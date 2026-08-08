"""PH2-D03-V2 的 variable-count pack registry 与无写入 trainer preflight。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V2_EXECUTION_STAGES,
    V2_RELEASE_KEY,
    V2_RUN_SCALE_KEYS,
    V2_SCALE_RECORD_LIMITS,
    V2_SPLITS,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    validate_v2_record,
    validate_v2_record_set,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactManifest,
    DatasetContractError,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_artifact_manifest,
)


class V2RegistryError(ValueError):
    """v2 manifest、pack registry 或 trainer preflight 不满足严格合同。"""


_PACK_STAGES = V2_EXECUTION_STAGES[1:-1]


def _strict_sha256(value: Any, *, where: str) -> str:
    """要求小写、恰好 64 位十六进制 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise V2RegistryError(f"{where} 必须是小写 SHA-256")
    return value


def _strict_positive_int(value: Any, *, where: str) -> int:
    """要求严格正整数，拒绝 bool。"""
    if type(value) is not int or value <= 0:
        raise V2RegistryError(f"{where} 必须是正严格整数")
    return value


def _strict_nonnegative_int(value: Any, *, where: str) -> int:
    """要求严格非负整数，拒绝 bool。"""
    if type(value) is not int or value < 0:
        raise V2RegistryError(f"{where} 必须是非负严格整数")
    return value


def _key_tuple(value: Any, *, where: str) -> tuple[int, ...]:
    """校验 registry 内部使用的稳定正整数键。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item <= 0 for item in value)):
        raise V2RegistryError(f"{where} 必须是正整数 tuple")
    return value


def _split_counts(
        value: Any,
        *,
        where: str,
        allow_empty: bool = True,
        ) -> tuple[tuple[str, int], ...]:
    """校验 split 计数的唯一性、规范顺序和严格整数。"""
    if not isinstance(value, tuple):
        raise V2RegistryError(f"{where} 必须是 tuple")
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    last_rank = -1
    for item in value:
        if (not isinstance(item, tuple) or len(item) != 2
                or not isinstance(item[0], str)):
            raise V2RegistryError(f"{where} 项非法")
        split, count = item
        if split not in V2_SPLITS or split in seen:
            raise V2RegistryError(f"{where} split 重复或未知")
        if not allow_empty and count <= 0:
            raise V2RegistryError(f"{where} count 不得为空")
        _strict_nonnegative_int(count, where=f"{where}.{split}")
        rank = V2_SPLITS.index(split)
        if rank <= last_rank:
            raise V2RegistryError(f"{where} 顺序非法")
        seen.add(split)
        last_rank = rank
        result.append((split, count))
    return tuple(result)


def _manifest_relative(relative: Any) -> str:
    """校验 manifest 的安全 POSIX 相对路径，不访问文件系统。"""
    if (not isinstance(relative, str) or not relative or "\\" in relative):
        raise V2RegistryError("v2 manifest path 必须是 POSIX 文本")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise V2RegistryError("v2 manifest path 必须是安全 POSIX 相对路径")
    return relative


def _safe_manifest_path(root: Path, relative: str) -> Path:
    """把公开 v2 manifest 相对路径解析到 root 内并拒绝旧/private 根。"""
    _manifest_relative(relative)
    pure = PurePosixPath(relative)
    lowered = tuple(part.casefold() for part in pure.parts)
    if "d03_v1" in lowered or "private" in lowered:
        raise V2RegistryError("v2 registry 不得读取旧 v1 或 private manifest")
    repository = root.resolve()
    candidate = repository / Path(*pure.parts)
    current = repository
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise V2RegistryError("v2 manifest path 不得经过 symlink")
    target = candidate.resolve()
    if not target.is_relative_to(repository) or target.is_symlink():
        raise V2RegistryError("v2 manifest path 缺失或是 symlink")
    if not target.is_file():
        raise V2RegistryError("v2 manifest path 缺失")
    return target


def _manifest_digest(path: Path) -> tuple[int, str]:
    """只读取 manifest 本体并形成 transport 字节身份。"""
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _stage_rank(stage: str) -> int:
    """返回 W 阶段在 v2 执行序中的序号，FT00/PW 不可作为 pack stage。"""
    if stage not in _PACK_STAGES:
        raise V2RegistryError("v2 pack earliest stage 必须是 W-02..W-09")
    return V2_EXECUTION_STAGES.index(stage)


@dataclass(frozen=True)
class V2PackEntry:
    """由单个 v2 ArtifactManifest 派生的 immutable pack 计数与身份。"""

    pack_key: tuple[int, ...]
    manifest_relative_path: str
    manifest_size_bytes: int
    manifest_sha256: str
    manifest_content_sha256: str
    source_key: str
    license_partition: str
    earliest_stage: str
    w_stages: tuple[str, ...]
    source_ref_count: int
    observation_counts: tuple[tuple[str, int], ...]
    teacher_evidence_count: int
    evaluator_label_counts: tuple[tuple[str, int], ...]
    total_record_count: int
    source_cluster_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _key_tuple(self.pack_key, where="v2 pack key")
        relative = _manifest_relative(self.manifest_relative_path)
        lowered = tuple(part.casefold() for part in PurePosixPath(relative).parts)
        if "d03_v1" in lowered or "private" in lowered:
            raise V2RegistryError("v2 pack manifest 不得指向旧/private 路径")
        _strict_positive_int(self.manifest_size_bytes, where="v2 manifest size")
        _strict_sha256(self.manifest_sha256, where="v2 manifest SHA-256")
        _strict_sha256(self.manifest_content_sha256,
                       where="v2 manifest content SHA-256")
        if (not isinstance(self.source_key, str) or not self.source_key
                or not isinstance(self.license_partition, str)
                or not self.license_partition):
            raise V2RegistryError("v2 pack source/license 非法")
        _stage_rank(self.earliest_stage)
        if (not isinstance(self.w_stages, tuple) or not self.w_stages
                or len(set(self.w_stages)) != len(self.w_stages)
                or any(stage not in _PACK_STAGES for stage in self.w_stages)
                or tuple(sorted(self.w_stages, key=V2_EXECUTION_STAGES.index))
                != self.w_stages
                or self.w_stages[0] != self.earliest_stage):
            raise V2RegistryError("v2 pack w_stages 顺序或 earliest stage 非法")
        source_count = _strict_positive_int(
            self.source_ref_count, where="v2 pack source_ref_count")
        teacher_count = _strict_nonnegative_int(
            self.teacher_evidence_count, where="v2 pack teacher_evidence_count")
        observations = _split_counts(
            self.observation_counts, where="v2 pack observation_counts")
        evaluators = _split_counts(
            self.evaluator_label_counts, where="v2 pack evaluator_label_counts")
        total = _strict_positive_int(
            self.total_record_count, where="v2 pack total_record_count")
        expected_total = (
            source_count + sum(count for _, count in observations)
            + teacher_count + sum(count for _, count in evaluators)
        )
        if total != expected_total:
            raise V2RegistryError("v2 pack total count 与 owner/split 求和不一致")
        clusters = tuple(
            _key_tuple(key, where="v2 pack source cluster")
            for key in self.source_cluster_keys
        )
        if (not clusters or tuple(sorted(clusters)) != clusters
                or len(set(clusters)) != len(clusters)):
            raise V2RegistryError("v2 pack source cluster 必须唯一且规范排序")

    @property
    def train_observation_count(self) -> int:
        """返回该 pack 的 train Observation 数，不把其他 owner 冒充训练量。"""
        return dict(self.observation_counts).get("train", 0)

    @property
    def train_input_count(self) -> int:
        """返回 source/train Observation/teacher 三类 candidate 输入总数。"""
        return self.source_ref_count + self.train_observation_count + self.teacher_evidence_count

    def to_dict(self) -> dict[str, Any]:
        """导出 pack registry entry。"""
        return {
            "earliest_stage": self.earliest_stage,
            "evaluator_label_counts": [
                {"count": count, "split": split}
                for split, count in self.evaluator_label_counts
            ],
            "license_partition": self.license_partition,
            "manifest_content_sha256": self.manifest_content_sha256,
            "manifest_relative_path": self.manifest_relative_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_size_bytes": self.manifest_size_bytes,
            "observation_counts": [
                {"count": count, "split": split}
                for split, count in self.observation_counts
            ],
            "pack_key": list(self.pack_key),
            "source_cluster_keys": [list(key) for key in self.source_cluster_keys],
            "source_key": self.source_key,
            "source_ref_count": self.source_ref_count,
            "teacher_evidence_count": self.teacher_evidence_count,
            "total_record_count": self.total_record_count,
            "w_stages": list(self.w_stages),
        }

    @classmethod
    def from_manifest(
            cls,
            root: Path,
            relative: str,
            ) -> "V2PackEntry":
        """只读 manifest、验证 v2 schema 并派生 owner/split 计数。"""
        path = _safe_manifest_path(root, relative)
        try:
            manifest = read_artifact_manifest(path)
        except (DatasetArtifactIOError, DatasetContractError, OSError, ValueError) as error:
            raise V2RegistryError("v2 pack manifest 无法读取") from error
        if not isinstance(manifest, ArtifactManifest):
            raise V2RegistryError("v2 pack manifest 类型非法")
        try:
            validate_v2_record(manifest.to_dict())
        except (D03ContractError, ValueError) as error:
            raise V2RegistryError("v2 pack manifest schema 校验失败") from error
        manifest_size, manifest_sha = _manifest_digest(path)
        observation_counts: dict[str, int] = {split: 0 for split in V2_SPLITS}
        evaluator_counts: dict[str, int] = {split: 0 for split in V2_SPLITS}
        source_count = teacher_count = 0
        for item in manifest.files:
            if item.record_kind == "source_ref":
                source_count += item.record_count
            elif item.record_kind == "observation":
                assert item.split is not None
                observation_counts[item.split] += item.record_count
            elif item.record_kind == "teacher_evidence":
                teacher_count += item.record_count
            elif item.record_kind == "evaluator_label":
                assert item.split is not None
                evaluator_counts[item.split] += item.record_count
        return cls(
            manifest.stable_key.components,
            relative,
            manifest_size,
            manifest_sha,
            manifest.content_sha256(),
            manifest.source_key,
            manifest.license_partition,
            manifest.earliest_invalidated_stage,
            manifest.w_stages,
            source_count,
            tuple((split, observation_counts[split]) for split in V2_SPLITS
                  if observation_counts[split]),
            teacher_count,
            tuple((split, evaluator_counts[split]) for split in V2_SPLITS
                  if evaluator_counts[split]),
            manifest.record_count,
            tuple(key.components for key in manifest.source_cluster_keys),
        )


@dataclass(frozen=True)
class V2RegistrySnapshot:
    """v2 registry 的变量计数汇总，不包含任何 payload。"""

    pack_count: int
    total_record_count: int
    source_ref_count: int
    observation_counts: tuple[tuple[str, int], ...]
    teacher_evidence_count: int
    evaluator_label_counts: tuple[tuple[str, int], ...]
    source_cluster_count: int
    manifest_identities: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        _strict_positive_int(self.pack_count, where="v2 registry pack_count")
        total = _strict_positive_int(
            self.total_record_count, where="v2 registry total_record_count")
        source = _strict_positive_int(
            self.source_ref_count, where="v2 registry source_ref_count")
        teacher = _strict_nonnegative_int(
            self.teacher_evidence_count, where="v2 registry teacher_evidence_count")
        observations = _split_counts(
            self.observation_counts, where="v2 registry observation_counts")
        evaluators = _split_counts(
            self.evaluator_label_counts, where="v2 registry evaluator_label_counts")
        expected = (
            source + sum(count for _, count in observations)
            + teacher + sum(count for _, count in evaluators)
        )
        if total != expected:
            raise V2RegistryError("v2 registry total count 与 owner/split 求和不一致")
        _strict_positive_int(self.source_cluster_count,
                             where="v2 registry source_cluster_count")
        if not isinstance(self.manifest_identities, tuple):
            raise V2RegistryError("v2 registry manifest identities 类型非法")
        identities: list[tuple[str, int, str]] = []
        for item in self.manifest_identities:
            if not isinstance(item, tuple) or len(item) != 3:
                raise V2RegistryError("v2 registry manifest identity 项非法")
            relative, size, digest = item
            _manifest_relative(relative)
            _strict_positive_int(size, where="v2 manifest identity size")
            _strict_sha256(digest, where="v2 manifest identity SHA-256")
            identities.append(item)
        if len(identities) != self.pack_count or len(set(identities)) != len(identities):
            raise V2RegistryError("v2 registry manifest identities 不闭合")

    def to_dict(self) -> dict[str, Any]:
        """导出不含 payload 的 registry snapshot。"""
        return {
            "evaluator_label_counts": [list(item) for item in self.evaluator_label_counts],
            "manifest_identities": [list(item) for item in self.manifest_identities],
            "observation_counts": [list(item) for item in self.observation_counts],
            "pack_count": self.pack_count,
            "source_cluster_count": self.source_cluster_count,
            "source_ref_count": self.source_ref_count,
            "total_record_count": self.total_record_count,
            "training_input_count": (
                self.source_ref_count
                + dict(self.observation_counts).get("train", 0)
                + self.teacher_evidence_count
            ),
            "teacher_evidence_count": self.teacher_evidence_count,
        }


@dataclass(frozen=True)
class V2PackRegistry:
    """严格按 manifest 派生的 variable-count pack catalog。"""

    release_key: str
    entries: tuple[V2PackEntry, ...]

    def __post_init__(self) -> None:
        if self.release_key != V2_RELEASE_KEY:
            raise V2RegistryError("v2 registry release identity 漂移")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise V2RegistryError("v2 registry entries 不能为空")
        if any(not isinstance(item, V2PackEntry) for item in self.entries):
            raise V2RegistryError("v2 registry entry 类型非法")
        if tuple(sorted(self.entries, key=lambda item: item.pack_key)) != self.entries:
            raise V2RegistryError("v2 registry pack 顺序漂移")
        if len({item.pack_key for item in self.entries}) != len(self.entries):
            raise V2RegistryError("v2 registry pack key 重复")
        if len({item.manifest_relative_path for item in self.entries}) != len(self.entries):
            raise V2RegistryError("v2 registry manifest path 重复")
        clusters = [cluster for item in self.entries for cluster in item.source_cluster_keys]
        if len(clusters) != len(set(clusters)):
            raise V2RegistryError("v2 registry source cluster 跨 pack 重叠")

    @classmethod
    def from_manifest_paths(
            cls,
            repository_root: str | Path,
            manifest_paths: Iterable[str],
            ) -> "V2PackRegistry":
        """只读多个 v2 manifest，不打开任何 JSONL payload。"""
        root = Path(repository_root).resolve()
        paths = tuple(manifest_paths)
        if not paths:
            raise V2RegistryError("v2 registry manifest paths 不能为空")
        entries = tuple(sorted(
            (V2PackEntry.from_manifest(root, relative) for relative in paths),
            key=lambda item: item.pack_key,
        ))
        return cls(V2_RELEASE_KEY, entries)

    def snapshot(self) -> V2RegistrySnapshot:
        """派生所有 owner/split 计数和 manifest 双身份。"""
        observations: dict[str, int] = {split: 0 for split in V2_SPLITS}
        evaluators: dict[str, int] = {split: 0 for split in V2_SPLITS}
        for item in self.entries:
            for split, count in item.observation_counts:
                observations[split] += count
            for split, count in item.evaluator_label_counts:
                evaluators[split] += count
        return V2RegistrySnapshot(
            len(self.entries),
            sum(item.total_record_count for item in self.entries),
            sum(item.source_ref_count for item in self.entries),
            tuple((split, observations[split]) for split in V2_SPLITS if observations[split]),
            sum(item.teacher_evidence_count for item in self.entries),
            tuple((split, evaluators[split]) for split in V2_SPLITS if evaluators[split]),
            sum(len(item.source_cluster_keys) for item in self.entries),
            tuple((item.manifest_relative_path, item.manifest_size_bytes, item.manifest_sha256)
                  for item in self.entries),
        )

    def train_plan(
            self,
            stage_key: str,
            *,
            scale_key: str = "P0",
            max_records: int | None = None,
            ) -> "V2TrainPlan":
        """按 earliest stage 派生 candidate train pack 计划，不读取 payload。"""
        if stage_key not in _PACK_STAGES:
            raise V2RegistryError("v2 train plan stage 必须是 W-02..W-09")
        if scale_key not in V2_RUN_SCALE_KEYS:
            raise V2RegistryError("v2 train plan scale 未激活")
        if max_records is not None:
            _strict_positive_int(max_records, where="v2 train plan max_records")
        rank = V2_EXECUTION_STAGES.index(stage_key)
        selected = tuple(item for item in self.entries
                         if _stage_rank(item.earliest_stage) <= rank
                         and item.train_observation_count > 0)
        if not selected:
            raise V2RegistryError("v2 train plan 没有可见 train pack")
        source_count = sum(item.source_ref_count for item in selected)
        observation_count = sum(item.train_observation_count for item in selected)
        teacher_count = sum(item.teacher_evidence_count for item in selected)
        total = source_count + observation_count + teacher_count
        limit = V2_SCALE_RECORD_LIMITS[scale_key]
        if total > limit or (max_records is not None and total > max_records):
            raise V2RegistryError("v2 train plan 超过 scale/resource budget")
        manifest_identity = tuple(
            (item.manifest_relative_path, item.manifest_sha256)
            for item in selected
        )
        digest = hashlib.sha256(canonical_json_bytes({
            "observation_count": observation_count,
            "release_key": V2_RELEASE_KEY,
            "scale_key": scale_key,
            "source_count": source_count,
            "stage_key": stage_key,
            "teacher_count": teacher_count,
            "manifests": [list(item) for item in manifest_identity],
        })).hexdigest()
        return V2TrainPlan(
            V2_RELEASE_KEY, stage_key, scale_key,
            tuple(item.pack_key for item in selected),
            source_count, observation_count, teacher_count, total, digest,
        )


@dataclass(frozen=True)
class V2TrainPlan:
    """冻结一组 variable-count train owner 输入和其 manifest commitment。"""

    release_key: str
    stage_key: str
    scale_key: str
    pack_keys: tuple[tuple[int, ...], ...]
    source_ref_count: int
    observation_count: int
    teacher_evidence_count: int
    total_input_count: int
    manifest_commitment: str

    def __post_init__(self) -> None:
        if self.release_key != V2_RELEASE_KEY or self.stage_key not in _PACK_STAGES:
            raise V2RegistryError("v2 train plan identity 非法")
        if self.scale_key not in V2_RUN_SCALE_KEYS or not self.pack_keys:
            raise V2RegistryError("v2 train plan scale/pack 非法")
        if (not isinstance(self.pack_keys, tuple)
                or any(not isinstance(key, tuple) for key in self.pack_keys)):
            raise V2RegistryError("v2 train plan pack key 类型非法")
        for key in self.pack_keys:
            _key_tuple(key, where="v2 train plan pack key")
        if (tuple(sorted(self.pack_keys)) != self.pack_keys
                or len(set(self.pack_keys)) != len(self.pack_keys)):
            raise V2RegistryError("v2 train plan pack key 顺序或唯一性非法")
        source = _strict_positive_int(
            self.source_ref_count, where="v2 train plan source_ref_count")
        observation = _strict_positive_int(
            self.observation_count, where="v2 train plan observation_count")
        teacher = _strict_nonnegative_int(
            self.teacher_evidence_count, where="v2 train plan teacher_evidence_count")
        total = _strict_positive_int(
            self.total_input_count, where="v2 train plan total_input_count")
        if total != source + observation + teacher:
            raise V2RegistryError("v2 train plan count 不闭合")
        if total > V2_SCALE_RECORD_LIMITS[self.scale_key]:
            raise V2RegistryError("v2 train plan 超过 scale budget")
        _strict_sha256(self.manifest_commitment,
                       where="v2 train plan commitment")

    def to_dict(self) -> dict[str, Any]:
        """导出 train plan。"""
        return {
            "manifest_commitment": self.manifest_commitment,
            "observation_count": self.observation_count,
            "pack_keys": [list(key) for key in self.pack_keys],
            "release_key": self.release_key,
            "scale_key": self.scale_key,
            "source_ref_count": self.source_ref_count,
            "stage_key": self.stage_key,
            "teacher_evidence_count": self.teacher_evidence_count,
            "total_input_count": self.total_input_count,
        }


@dataclass(frozen=True)
class V2GenericTrainingResult:
    """预训练 generic lane 的纯校验结果，明确没有 Candidate/host 写入。"""

    plan: V2TrainPlan
    source_ref_count: int
    observation_count: int
    teacher_evidence_count: int
    input_commitment: str
    candidate_writes: int
    core_writes: int
    teacher_calls: int

    def __post_init__(self) -> None:
        if (type(self.source_ref_count) is not int
                or type(self.observation_count) is not int
                or type(self.teacher_evidence_count) is not int
                or self.source_ref_count != self.plan.source_ref_count
                or self.observation_count != self.plan.observation_count
                or self.teacher_evidence_count != self.plan.teacher_evidence_count):
            raise V2RegistryError("v2 generic trainer count 与 plan 漂移")
        if any(type(value) is not int or value != 0 for value in (
                self.candidate_writes, self.core_writes, self.teacher_calls)):
            raise V2RegistryError("FT00 generic trainer 不得产生训练/teacher 写入")
        _strict_sha256(self.input_commitment,
                       where="v2 generic trainer input commitment")


class V2GenericTrainer:
    """在 FT00 只做 variable-count、owner 和 canonical commitment 校验。"""

    def prepare(self, registry: V2PackRegistry, plan: V2TrainPlan) -> V2TrainPlan:
        """确认计划来自当前 registry，仍不打开 train payload。"""
        if not isinstance(registry, V2PackRegistry) or not isinstance(plan, V2TrainPlan):
            raise V2RegistryError("v2 generic trainer 输入类型非法")
        actual = registry.train_plan(plan.stage_key, scale_key=plan.scale_key)
        if actual != plan:
            raise V2RegistryError("v2 train plan 与当前 registry 漂移")
        return plan

    def validate_train_records(
            self,
            plan: V2TrainPlan,
            values: Iterable[dict[str, Any]],
            *,
            teacher_owner_key: tuple[int, ...],
            evaluator_owner_key: tuple[int, ...],
            ) -> V2GenericTrainingResult:
        """验证 variable-count train records，拒绝 evaluator label 和任何持久化写。"""
        if not isinstance(plan, V2TrainPlan):
            raise V2RegistryError("v2 generic trainer plan 类型非法")
        teacher_key = _key_tuple(teacher_owner_key, where="v2 teacher owner key")
        evaluator_key = _key_tuple(evaluator_owner_key, where="v2 evaluator owner key")
        if teacher_key == evaluator_key:
            raise V2RegistryError("v2 teacher/evaluator owner 不得相同")
        records = tuple(values)
        if any(not isinstance(item, dict) for item in records):
            raise V2RegistryError("v2 generic trainer record 类型非法")
        allowed_kinds = {"source_ref", "observation", "teacher_evidence"}
        for item in records:
            kind = item.get("record_kind")
            if kind not in allowed_kinds:
                raise V2RegistryError(
                    "v2 generic trainer 只接受 source_ref/observation/teacher_evidence")
            if kind == "observation" and item.get("split") != "train":
                raise V2RegistryError("v2 generic trainer 只接受 train observation")
        try:
            checked = validate_v2_record_set(
                list(records),
                teacher_owner_key=teacher_key,
                evaluator_owner_key=evaluator_key,
            )
        except (DatasetContractError, D03ContractError) as error:
            raise V2RegistryError("v2 generic trainer train record 校验失败") from error
        sources = tuple(item for item in checked if isinstance(item, SourceRefRecord))
        observations = tuple(item for item in checked if isinstance(item, ObservationRecord))
        teachers = tuple(item for item in checked if isinstance(item, TeacherEvidenceRecord))
        if (len(sources) != plan.source_ref_count
                or len(observations) != plan.observation_count
                or len(teachers) != plan.teacher_evidence_count):
            raise V2RegistryError("v2 generic trainer variable count 与 plan 不一致")
        encoded = tuple(sorted(
            canonical_json_bytes(item.to_dict())
            for item in (*sources, *observations, *teachers)
        ))
        return V2GenericTrainingResult(
            plan, len(sources), len(observations), len(teachers),
            hashlib.sha256(b"".join(encoded)).hexdigest(), 0, 0, 0,
        )


__all__ = [
    "V2GenericTrainer", "V2GenericTrainingResult", "V2PackEntry",
    "V2PackRegistry", "V2RegistryError", "V2RegistrySnapshot", "V2TrainPlan",
]
