"""PH2-D03-V2 FT00-03 的流式 owner reader、logical shard 与 checkpoint。"""
from __future__ import annotations

import gzip
import hashlib
import heapq
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V2_CHECKPOINT_FORMAT_VERSION,
    V2_CHECKPOINT_IDENTITY_FIELDS,
    V2_LOGICAL_SHARD_COUNT,
    V2_OWNER_KEYS,
    V2_RELEASE_KEY,
    V2RunIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_registry import (
    V2PackEntry,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    ArtifactManifest,
    DatasetContractError,
    canonical_json_bytes as dataset_canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_artifact_manifest,
)


class V2StreamingError(ValueError):
    """v2 流式读取、分片或 checkpoint 身份不满足严格合同。"""


_PUBLIC_VIEW_RULES = {
    "candidate": {
        "source": frozenset({None}),
        "observation": frozenset({"train"}),
    },
    "teacher": {
        "source": frozenset({None}),
        "observation": frozenset({"train"}),
        "teacher": frozenset({"train"}),
    },
    "dev": {
        "source": frozenset({None}),
        "observation": frozenset({"dev"}),
    },
    "shadow": {
        "source": frozenset({None}),
        "observation": frozenset({"train", "dev"}),
    },
}

_OWNER_PUBLIC_VIEWS = {
    "PH2_V2_CANDIDATE": "candidate",
    "PH2_V2_TEACHER": "teacher",
    "PH2_V2_DEV_CALIBRATOR": "dev",
    "PH2_V2_SHADOW_AUDITOR": "shadow",
}


def _sha256(value: Any, *, where: str) -> str:
    """要求小写规范 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise V2StreamingError(f"{where} 必须是小写 SHA-256")
    return value


def _key(value: Any, *, where: str, allow_empty: bool = False) -> tuple[int, ...]:
    """要求稳定正整数键，cursor 可用空 tuple 表示起点。"""
    if not isinstance(value, tuple):
        raise V2StreamingError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise V2StreamingError(f"{where} 不能为空")
    if any(type(item) is not int or item <= 0 for item in value):
        raise V2StreamingError(f"{where} 必须只含正严格整数")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    """把公开 POSIX 路径解析到 root 内并拒绝 symlink/旧/private 根。"""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise V2StreamingError("v2 payload 路径必须是 POSIX 文本")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise V2StreamingError("v2 payload 路径逃逸 root")
    lowered = tuple(part.casefold() for part in pure.parts)
    if "d03_v1" in lowered or "private" in lowered:
        raise V2StreamingError("v2 FT00-03 reader 不得读取旧/private 路径")
    repository = root.resolve()
    current = repository
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise V2StreamingError("v2 payload 路径不得经过 symlink")
    target = (repository / Path(*pure.parts)).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise V2StreamingError("v2 payload 路径缺失")
    return target


def _payload_relative(manifest_relative: str, file_relative: str) -> str:
    """将 pack 内文件身份拼到 manifest 所在目录。"""
    parent = PurePosixPath(manifest_relative).parent
    return PurePosixPath(parent, file_relative).as_posix()


def _sha256_file(path: Path) -> tuple[int, str]:
    """流式计算压缩 payload 的 transport size/SHA。"""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    except OSError as error:
        raise V2StreamingError("v2 payload transport 无法读取") from error
    return size, digest.hexdigest()


@dataclass(frozen=True)
class V2LogicalShardPlan:
    """固定数量、与 worker 配置无关的 deterministic logical shard。"""

    logical_shard_count: int = V2_LOGICAL_SHARD_COUNT

    def __post_init__(self) -> None:
        if type(self.logical_shard_count) is not int or self.logical_shard_count <= 0:
            raise V2StreamingError("v2 logical shard count 必须是正严格整数")
        if self.logical_shard_count != V2_LOGICAL_SHARD_COUNT:
            raise V2StreamingError("v2 logical shard count 漂移")

    def validate_index(self, value: Any) -> int:
        """校验一个 logical shard 下标。"""
        if (type(value) is not int
                or value < 0 or value >= self.logical_shard_count):
            raise V2StreamingError("v2 logical shard index 越界")
        return value

    def shard_for(self, record_key: tuple[int, ...]) -> int:
        """按稳定键摘要派生 shard，不依赖 worker 数或输入顺序。"""
        key = _key(record_key, where="v2 record key")
        digest = hashlib.sha256(dataset_canonical_json_bytes(list(key))).digest()
        return int.from_bytes(digest[:8], "big") % self.logical_shard_count


@dataclass(frozen=True)
class V2StreamWindow:
    """有限窗口及其可恢复 cursor。"""

    records: tuple[object, ...]
    cursor_record_key: tuple[int, ...]
    complete: int

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise V2StreamingError("v2 stream window records 类型非法")
        _key(self.cursor_record_key, where="v2 window cursor", allow_empty=True)
        if type(self.complete) is not int or self.complete not in (0, 1):
            raise V2StreamingError("v2 stream window complete 必须是 0/1")


@dataclass(frozen=True)
class V2StreamCheckpoint:
    """冻结 fresh/resume 共用的 checkpoint identity。"""

    checkpoint_format_version: int
    release_key: str
    run_identity_sha256: str
    owner_key: str
    pack_key: tuple[int, ...]
    source_state_sha256: str
    logical_shard_index: int
    cursor_record_key: tuple[int, ...]
    input_manifest_sha256: str

    def __post_init__(self) -> None:
        if (type(self.checkpoint_format_version) is not int
                or self.checkpoint_format_version != V2_CHECKPOINT_FORMAT_VERSION):
            raise V2StreamingError("v2 checkpoint format 漂移")
        if self.release_key != V2_RELEASE_KEY:
            raise V2StreamingError("v2 checkpoint release identity 漂移")
        _sha256(self.run_identity_sha256, where="v2 checkpoint run identity")
        if self.owner_key not in V2_OWNER_KEYS:
            raise V2StreamingError("v2 checkpoint owner 未注册")
        _key(self.pack_key, where="v2 checkpoint pack key")
        _sha256(self.source_state_sha256, where="v2 checkpoint source state")
        V2LogicalShardPlan().validate_index(self.logical_shard_index)
        _key(self.cursor_record_key, where="v2 checkpoint cursor", allow_empty=True)
        _sha256(self.input_manifest_sha256,
                where="v2 checkpoint input manifest")

    def to_dict(self) -> dict[str, Any]:
        """导出 canonical checkpoint object。"""
        return {
            "checkpoint_format_version": self.checkpoint_format_version,
            "cursor_record_key": list(self.cursor_record_key),
            "input_manifest_sha256": self.input_manifest_sha256,
            "logical_shard_index": self.logical_shard_index,
            "owner_key": self.owner_key,
            "pack_key": list(self.pack_key),
            "release_key": self.release_key,
            "run_identity_sha256": self.run_identity_sha256,
            "source_state_sha256": self.source_state_sha256,
        }

    def sha256(self) -> str:
        """返回 checkpoint canonical SHA-256。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "V2StreamCheckpoint":
        """严格回读 checkpoint，拒绝未知/缺失字段。"""
        fields = {"checkpoint_format_version", *V2_CHECKPOINT_IDENTITY_FIELDS}
        if not isinstance(value, dict) or set(value) != fields:
            raise V2StreamingError("v2 checkpoint 字段不精确")
        if (not isinstance(value["pack_key"], list)
                or not isinstance(value["cursor_record_key"], list)):
            raise V2StreamingError("v2 checkpoint key 必须是数组")
        return cls(
            value["checkpoint_format_version"],
            str(value["release_key"]),
            str(value["run_identity_sha256"]),
            str(value["owner_key"]),
            _key(tuple(value["pack_key"]), where="v2 checkpoint pack key"),
            str(value["source_state_sha256"]),
            value["logical_shard_index"],
            _key(tuple(value["cursor_record_key"]), where="v2 checkpoint cursor",
                 allow_empty=True),
            str(value["input_manifest_sha256"]),
        )

    def validate_against(
            self,
            run_identity: V2RunIdentity,
            entry: V2PackEntry,
            *,
            owner_key: str,
            source_state_sha256: str,
            ) -> None:
        """将 checkpoint 与当前 run、pack、owner 和 source state 逐项绑定。"""
        if not isinstance(run_identity, V2RunIdentity):
            raise V2StreamingError("v2 checkpoint run identity 类型非法")
        if not isinstance(entry, V2PackEntry):
            raise V2StreamingError("v2 checkpoint pack entry 类型非法")
        if self.release_key != run_identity.release_key:
            raise V2StreamingError("v2 checkpoint release 与 run 不一致")
        if self.run_identity_sha256 != run_identity.sha256():
            raise V2StreamingError("v2 checkpoint run identity 不一致")
        if self.input_manifest_sha256 != run_identity.input_manifest_sha256:
            raise V2StreamingError("v2 checkpoint input manifest 不一致")
        if self.pack_key != entry.pack_key:
            raise V2StreamingError("v2 checkpoint pack 不一致")
        if self.owner_key != owner_key:
            raise V2StreamingError("v2 checkpoint owner 不一致")
        if self.source_state_sha256 != source_state_sha256:
            raise V2StreamingError("v2 checkpoint source state 不一致")


@dataclass
class V2StreamReader:
    """按 manifest 身份流式读取公开 pack，始终保持记录级内存。"""

    repository_root: str | Path
    entry: V2PackEntry

    def __post_init__(self) -> None:
        if not isinstance(self.entry, V2PackEntry):
            raise V2StreamingError("v2 stream reader entry 类型非法")
        self.repository_root = Path(self.repository_root).resolve()

    def _manifest(self) -> ArtifactManifest:
        """读取并回验 manifest transport/content 双身份，不读取 payload。"""
        path = _safe_path(self.repository_root, self.entry.manifest_relative_path)
        try:
            manifest = read_artifact_manifest(path)
        except (DatasetArtifactIOError, DatasetContractError, OSError, ValueError) as error:
            raise V2StreamingError("v2 stream manifest 无法读取") from error
        if not isinstance(manifest, ArtifactManifest):
            raise V2StreamingError("v2 stream manifest 类型非法")
        size, digest = _sha256_file(path)
        if (size != self.entry.manifest_size_bytes
                or digest != self.entry.manifest_sha256
                or manifest.content_sha256() != self.entry.manifest_content_sha256):
            raise V2StreamingError("v2 stream manifest identity 漂移")
        try:
            validated = validate_v2_record(manifest.to_dict())
        except (DatasetContractError, ValueError) as error:
            raise V2StreamingError("v2 stream manifest schema 无效") from error
        if validated != manifest:
            raise V2StreamingError("v2 stream manifest schema 回读漂移")
        return manifest

    @staticmethod
    def _visible(identity: ArtifactFileIdentity, view_kind: str) -> bool:
        """按 FT00-03 public owner 规则判断文件可见性。"""
        rules = _PUBLIC_VIEW_RULES.get(view_kind)
        if rules is None:
            raise V2StreamingError(
                "v2 FT00-03 只允许 candidate/teacher/dev/shadow public view，"
                "private evaluator 留给后续独立切片")
        allowed_splits = rules.get(identity.owner_kind)
        if allowed_splits is None:
            return False
        return identity.split in allowed_splits

    def _files(
            self,
            manifest: ArtifactManifest,
            view_kind: str,
            split: str | None,
            ) -> tuple[ArtifactFileIdentity, ...]:
        """返回规范顺序的可见 owner 文件身份。"""
        if split is not None and split not in {"train", "dev", "held_out", "adversarial", "wall"}:
            raise V2StreamingError("v2 stream split 未注册")
        if view_kind not in _PUBLIC_VIEW_RULES:
            self._visible(manifest.files[0], view_kind)
        files = tuple(
            item for item in manifest.files
            if self._visible(item, view_kind)
            and (split is None or item.owner_kind == "source" or item.split == split)
        )
        if not files:
            raise V2StreamingError("v2 stream view 没有可见文件")
        return files

    def _file_records(
            self,
            manifest: ArtifactManifest,
            identity: ArtifactFileIdentity,
            ) -> Iterator[object]:
        """逐条读取单个 gzip JSONL，并在 EOF 验证完整身份。"""
        relative = _payload_relative(
            self.entry.manifest_relative_path, identity.relative_path)
        path = _safe_path(self.repository_root, relative)
        transport_size, transport_sha = _sha256_file(path)
        if (transport_size != identity.transport_size_bytes
                or transport_sha != identity.transport_sha256):
            raise V2StreamingError("v2 stream payload transport identity 漂移")
        content_digest = hashlib.sha256()
        content_size = 0
        count = 0
        previous_key: tuple[int, ...] | None = None
        first_key: tuple[int, ...] | None = None
        last_key: tuple[int, ...] | None = None
        try:
            with path.open("rb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                            raise V2StreamingError(
                                f"v2 stream 第 {line_number} 行换行非法")
                        content_digest.update(line)
                        content_size += len(line)
                        value = parse_canonical_json_bytes(
                            line[:-1], require_object=True)
                        assert isinstance(value, dict)
                        record = validate_v2_record(value)
                        if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                            raise V2StreamingError("v2 stream record kind 漂移")
                        if (identity.split is not None
                                and getattr(record, "split", identity.split) != identity.split):
                            raise V2StreamingError("v2 stream record split 漂移")
                        key = record.stable_key.components
                        if previous_key is not None and key <= previous_key:
                            raise V2StreamingError("v2 stream record key 未严格排序")
                        previous_key = key
                        if first_key is None:
                            first_key = key
                        last_key = key
                        count += 1
                        yield record
        except (OSError, EOFError, DatasetContractError, ValueError) as error:
            if isinstance(error, V2StreamingError):
                raise
            raise V2StreamingError("v2 stream gzip/JSONL 内容损坏") from error
        if (count != identity.record_count
                or content_size != identity.content_size_bytes
                or content_digest.hexdigest() != identity.content_sha256
                or first_key != (
                    identity.first_record_key.components
                    if identity.first_record_key is not None else None)
                or last_key != (
                    identity.last_record_key.components
                    if identity.last_record_key is not None else None)):
            raise V2StreamingError("v2 stream payload content identity 漂移")

    def iter_records(
            self,
            view_kind: str,
            *,
            split: str | None = None,
            shard_index: int | None = None,
            cursor_record_key: tuple[int, ...] = (),
            ) -> Iterator[object]:
        """按稳定键 merge 多个 owner 文件，保持常量级记录内存。"""
        manifest = self._manifest()
        files = self._files(manifest, view_kind, split)
        shard_plan = V2LogicalShardPlan()
        if shard_index is not None:
            shard_plan.validate_index(shard_index)
        cursor = _key(cursor_record_key, where="v2 stream cursor", allow_empty=True)
        streams = [iter(self._file_records(manifest, item)) for item in files]
        heap: list[tuple[tuple[int, ...], int, object]] = []
        for index, stream in enumerate(streams):
            try:
                record = next(stream)
            except StopIteration:
                continue
            heapq.heappush(heap, (record.stable_key.components, index, record))
        previous_key: tuple[int, ...] | None = None
        while heap:
            key, index, record = heapq.heappop(heap)
            if previous_key is not None and key == previous_key:
                raise V2StreamingError("v2 stream stable key 跨 owner 重复")
            previous_key = key
            try:
                following = next(streams[index])
            except StopIteration:
                following = None
            if following is not None:
                heapq.heappush(
                    heap, (following.stable_key.components, index, following))
            if cursor and key <= cursor:
                continue
            if shard_index is not None and shard_plan.shard_for(key) != shard_index:
                continue
            yield record

    def iter_windows(
            self,
            view_kind: str,
            *,
            window_size: int,
            split: str | None = None,
            shard_index: int | None = None,
            cursor_record_key: tuple[int, ...] = (),
            ) -> Iterator[V2StreamWindow]:
        """按窗口消费流，窗口边界携带 fresh/resume cursor。"""
        if type(window_size) is not int or window_size <= 0:
            raise V2StreamingError("v2 stream window_size 必须是正严格整数")
        pending: V2StreamWindow | None = None
        current: list[object] = []
        for record in self.iter_records(
                view_kind,
                split=split,
                shard_index=shard_index,
                cursor_record_key=cursor_record_key):
            current.append(record)
            if len(current) == window_size:
                page = V2StreamWindow(
                    tuple(current),
                    current[-1].stable_key.components,
                    0,
                )
                if pending is not None:
                    yield pending
                pending = page
                current = []
        if current:
            page = V2StreamWindow(
                tuple(current),
                current[-1].stable_key.components,
                0,
            )
            if pending is not None:
                yield pending
            pending = page
        if pending is None:
            yield V2StreamWindow((), cursor_record_key, 1)
        else:
            yield replace(pending, complete=1)

    def checkpoint(
            self,
            run_identity: V2RunIdentity,
            *,
            owner_key: str,
            shard_index: int,
            cursor_record_key: tuple[int, ...],
            source_state_sha256: str,
            ) -> V2StreamCheckpoint:
        """形成绑定当前 run、pack 和 source state 的 checkpoint。"""
        if not isinstance(run_identity, V2RunIdentity):
            raise V2StreamingError("v2 stream run identity 类型非法")
        checkpoint = V2StreamCheckpoint(
            V2_CHECKPOINT_FORMAT_VERSION,
            V2_RELEASE_KEY,
            run_identity.sha256(),
            owner_key,
            self.entry.pack_key,
            source_state_sha256,
            V2LogicalShardPlan().validate_index(shard_index),
            _key(cursor_record_key, where="v2 checkpoint cursor", allow_empty=True),
            run_identity.input_manifest_sha256,
        )
        checkpoint.validate_against(
            run_identity,
            self.entry,
            owner_key=owner_key,
            source_state_sha256=source_state_sha256,
        )
        return checkpoint

    def iter_from_checkpoint(
            self,
            checkpoint: V2StreamCheckpoint,
            run_identity: V2RunIdentity,
            *,
            owner_key: str,
            source_state_sha256: str,
            window_size: int,
            ) -> Iterator[V2StreamWindow]:
        """回验 checkpoint 后从 cursor/shard 继续流式读取。"""
        if not isinstance(checkpoint, V2StreamCheckpoint):
            raise V2StreamingError("v2 stream checkpoint 类型非法")
        checkpoint.validate_against(
            run_identity,
            self.entry,
            owner_key=owner_key,
            source_state_sha256=source_state_sha256,
        )
        view_kind = _OWNER_PUBLIC_VIEWS.get(owner_key)
        if view_kind is None:
            raise V2StreamingError("v2 checkpoint owner 没有 public view")
        return self.iter_windows(
            view_kind,
            window_size=window_size,
            shard_index=checkpoint.logical_shard_index,
            cursor_record_key=checkpoint.cursor_record_key,
        )


__all__ = [
    "V2LogicalShardPlan",
    "V2StreamCheckpoint",
    "V2StreamReader",
    "V2StreamWindow",
    "V2StreamingError",
]
