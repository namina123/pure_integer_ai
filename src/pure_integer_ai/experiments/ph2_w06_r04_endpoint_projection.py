"""W06-R04 train endpoint_id 到跨命题 canonical identity 的追加投影。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.ph2_authored_mereology_course import (
    read_authored_mereology_seeds,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    compile_relation_seed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06_IDENTITY_VERSIONS
from pure_integer_ai.experiments.ph2_w06_r04_shared import (
    W06_R04_RUNTIME_NAMESPACE,
)


W06_R04_ENDPOINT_PROJECTION_PATH = (
    "data/ph2/manifests/w06_r04_endpoint_projection_v1.json"
)
W06_R04_SAMPLE_PATH = (
    "data/ph2/authored_relation_mereology_seed_v1.jsonl.sample"
)
W06_R04_SOURCE_OVERLAY_PATH = (
    "data/ph2/manifests/w06_source_semantic_overlay_v1.json"
)
W06_R04_STAGE_PATH = (
    "data/ph2/manifests/d03_v1/stages/w06_stage_manifest_v1.json"
)
W06_R04_EXPECTED_PARENT_SHA256 = {
    W06_R04_SAMPLE_PATH:
        "0bce17b2fff1397a62a919390fc9432394ecfb1192b23f8e2d126a772f2326cc",
    W06_R04_SOURCE_OVERLAY_PATH:
        "f5cae297254191dffb5bcacdafbdc461dcd1cf3a1340de27d9a8c98c598bfbbc",
    W06_R04_STAGE_PATH:
        "a9beda13955e4708b5f2bb7f4d2b106be1bdf709c82acaefcfa95ca7d276e00a",
}


class W06R04EndpointProjectionError(RuntimeError):
    """R04 endpoint overlay、parent 或映射发生漂移。"""


def _sha256(path: Path) -> str:
    """流式读取 parent 文件并返回小写 SHA-256。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise W06R04EndpointProjectionError(
            f"无法读取 endpoint projection 依赖：{path}") from error
    return digest.hexdigest()


def _canonical_endpoint(endpoint_id: str, object_kind: int) -> ObjectIdentity:
    """以 authored endpoint_id 的无损整数序构造 canonical identity。"""
    if not isinstance(endpoint_id, str) or not endpoint_id:
        raise W06R04EndpointProjectionError("endpoint_id 必须是非空文本")
    if type(object_kind) is not int or object_kind <= 0:
        raise W06R04EndpointProjectionError("endpoint object kind 非法")
    return ObjectIdentity(
        object_kind,
        (
            W06_R04_RUNTIME_NAMESPACE,
            40,
            len(endpoint_id),
            *(ord(item) for item in endpoint_id),
        ),
        versions=W06_IDENTITY_VERSIONS,
    )


@dataclass(frozen=True)
class W06R04EndpointProjectionEntry:
    """一条来源局部 endpoint 到 canonical endpoint 的显式映射。"""

    local_endpoint: ObjectIdentity
    canonical_endpoint: ObjectIdentity

    def __post_init__(self) -> None:
        """禁止 projection 改变 object kind。"""
        if (not isinstance(self.local_endpoint, ObjectIdentity)
                or not isinstance(self.canonical_endpoint, ObjectIdentity)):
            raise TypeError("endpoint projection 必须使用 ObjectIdentity")
        if self.local_endpoint.object_kind != self.canonical_endpoint.object_kind:
            raise W06R04EndpointProjectionError(
                "endpoint projection 不得改变 object kind")

    def stable_key(self) -> tuple[int, ...]:
        """返回 local/canonical 完整键。"""
        local = self.local_endpoint.stable_key()
        canonical = self.canonical_endpoint.stable_key()
        return len(local), *local, len(canonical), *canonical


class W06R04EndpointProjection:
    """只按已发布完整 local identity 查询，不读取 surface。"""

    def __init__(
            self, entries: tuple[W06R04EndpointProjectionEntry, ...],
            ) -> None:
        """建立 local 到 canonical 的无冲突查表。"""
        if (not isinstance(entries, tuple) or not entries
                or any(not isinstance(item, W06R04EndpointProjectionEntry)
                       for item in entries)):
            raise W06R04EndpointProjectionError("endpoint projection entries 非法")
        normalized = tuple(sorted(
            entries, key=W06R04EndpointProjectionEntry.stable_key))
        if entries != normalized:
            raise W06R04EndpointProjectionError("endpoint projection 未规范排序")
        mapping: dict[ObjectIdentity, ObjectIdentity] = {}
        for item in entries:
            prior = mapping.get(item.local_endpoint)
            if prior is not None and prior != item.canonical_endpoint:
                raise W06R04EndpointProjectionError("同一 local endpoint 映射冲突")
            mapping[item.local_endpoint] = item.canonical_endpoint
        if len(mapping) != len(entries):
            raise W06R04EndpointProjectionError("endpoint projection local key 重复")
        self.entries = entries
        self._mapping = mapping

    def resolve(self, endpoint: ObjectIdentity) -> ObjectIdentity:
        """未知 endpoint 保持完整身份，已登记 endpoint 返回显式 canonical。"""
        if not isinstance(endpoint, ObjectIdentity):
            raise TypeError("endpoint projection lookup 类型非法")
        return self._mapping.get(endpoint, endpoint)

    def clone_for_evaluation(self) -> "W06R04EndpointProjection":
        """返回相同不可变映射的评测副本。"""
        return W06R04EndpointProjection(self.entries)

    def state_key(self) -> tuple[int, ...]:
        """返回 projection 版本和全部 entry 的稳定键。"""
        values = [W06_R04_RUNTIME_NAMESPACE, 1, len(self.entries)]
        for item in self.entries:
            key = item.stable_key()
            values.extend((len(key), *key))
        return tuple(values)


def build_w06_r04_endpoint_projection(repo_root: str | Path) -> dict[str, Any]:
    """从公开 authored train seed 重建无 surface canonical endpoint overlay。"""
    root = Path(repo_root)
    parents = {}
    for relative, expected in W06_R04_EXPECTED_PARENT_SHA256.items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise W06R04EndpointProjectionError(
                f"R04 endpoint projection parent 漂移：{relative}")
        parents[relative] = actual

    seeds = read_authored_mereology_seeds(root / W06_R04_SAMPLE_PATH)
    train = tuple(seed for seed in seeds if seed.split == "train")
    accepted = train
    if len(train) != 7:
        raise W06R04EndpointProjectionError("R04 train inventory 漂移")

    rows = []
    canonical_kinds: dict[str, int] = {}
    for seed in accepted:
        compiled = compile_relation_seed(seed)
        payload = compiled.observation_payload.to_value()
        endpoints = payload.get("endpoints")
        if not isinstance(endpoints, list) or len(endpoints) != len(seed.endpoints):
            raise W06R04EndpointProjectionError("compiled endpoint inventory 漂移")
        for source, compiled_endpoint in zip(
                seed.endpoints, endpoints, strict=True):
            if not isinstance(compiled_endpoint, dict):
                raise W06R04EndpointProjectionError("compiled endpoint 非 object")
            local = ObjectIdentity.from_stable_key(tuple(
                compiled_endpoint["endpoint_key"]))
            if (local.object_kind != source.object_kind
                    or compiled_endpoint.get("object_kind") != source.object_kind):
                raise W06R04EndpointProjectionError("compiled endpoint kind 漂移")
            prior_kind = canonical_kinds.get(source.endpoint_id)
            if prior_kind is not None and prior_kind != source.object_kind:
                raise W06R04EndpointProjectionError(
                    "同一 endpoint_id 跨记录 object kind 冲突")
            canonical_kinds[source.endpoint_id] = source.object_kind
            canonical = _canonical_endpoint(
                source.endpoint_id, source.object_kind)
            rows.append({
                "canonical_endpoint_key": list(canonical.stable_key()),
                "endpoint_id": source.endpoint_id,
                "local_endpoint_key": list(local.stable_key()),
                "object_kind": source.object_kind,
                "seed_id": seed.seed_id,
            })
    rows.sort(key=lambda item: item["local_endpoint_key"])
    if len({tuple(item["local_endpoint_key"]) for item in rows}) != len(rows):
        raise W06R04EndpointProjectionError("local endpoint key 重复")

    return {
        "artifact_kind": "PH2_W06_R04_ENDPOINT_PROJECTION",
        "artifact_version": "PH2-W06-R04-ENDPOINT-PROJECTION-V1",
        "entries": rows,
        "execution_state": {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "OPEN_GENERATION": "NE_NOT_YET_EVALUABLE",
            "W06_STARTED": 0,
            "W07_STARTED": 0,
            "teacher_calls": 0,
        },
        "format_version": 1,
        "parent_identities": parents,
        "projection_policy": {
            "canonical_endpoint_count": len(canonical_kinds),
            "held_out_mapping_count": 0,
            "identity_basis": "AUTHORED_ENDPOINT_ID_NOT_SURFACE",
            "local_endpoint_count": len(rows),
            "rejected_type_mismatch_count": 0,
            "train_seed_count": len(accepted),
        },
        "status": "W06_R04_ENDPOINT_PROJECTION_PASS",
    }


def canonical_w06_r04_endpoint_projection_bytes(
        repo_root: str | Path,
        ) -> bytes:
    """返回 R04 projection 的规范 JSON bytes，并固定末尾换行。"""
    return canonical_json_bytes(
        build_w06_r04_endpoint_projection(repo_root)) + b"\n"


def _projection_from_value(value: dict[str, Any]) -> W06R04EndpointProjection:
    """从规范 JSON object 恢复可注入 runtime 的 projection。"""
    rows = value.get("entries")
    if not isinstance(rows, list) or not rows:
        raise W06R04EndpointProjectionError("endpoint projection entries 缺失")
    entries = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
                "canonical_endpoint_key", "endpoint_id", "local_endpoint_key",
                "object_kind", "seed_id"}:
            raise W06R04EndpointProjectionError("endpoint projection entry 漂移")
        try:
            local = ObjectIdentity.from_stable_key(
                tuple(row["local_endpoint_key"]))
            canonical = ObjectIdentity.from_stable_key(
                tuple(row["canonical_endpoint_key"]))
        except Exception as error:
            raise W06R04EndpointProjectionError(
                "endpoint projection identity 非法") from error
        if (canonical != _canonical_endpoint(
                row["endpoint_id"], row["object_kind"])
                or local.object_kind != row["object_kind"]):
            raise W06R04EndpointProjectionError("endpoint projection 内容漂移")
        entries.append(W06R04EndpointProjectionEntry(local, canonical))
    entries.sort(key=W06R04EndpointProjectionEntry.stable_key)
    return W06R04EndpointProjection(tuple(entries))


def read_w06_r04_endpoint_projection(
        path: str | Path,
        ) -> W06R04EndpointProjection:
    """严格回读 canonical overlay 并返回可注入 R-04 的 resolver。"""
    target = Path(path)
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise W06R04EndpointProjectionError(
            "无法读取 R04 endpoint projection") from error
    if not payload.endswith(b"\n") or payload == b"\n":
        raise W06R04EndpointProjectionError("endpoint projection 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except DatasetContractError as error:
        raise W06R04EndpointProjectionError(
            "endpoint projection 不是 canonical JSON") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise W06R04EndpointProjectionError("endpoint projection 字节漂移")
    assert isinstance(value, dict)
    if (value.get("artifact_kind") != "PH2_W06_R04_ENDPOINT_PROJECTION"
            or value.get("status") != "W06_R04_ENDPOINT_PROJECTION_PASS"):
        raise W06R04EndpointProjectionError("endpoint projection 合同字段漂移")
    policy = value.get("projection_policy")
    if (not isinstance(policy, dict)
            or policy.get("held_out_mapping_count") != 0
            or policy.get("train_seed_count") != 7
            or policy.get("rejected_type_mismatch_count") != 0):
        raise W06R04EndpointProjectionError("endpoint projection policy 漂移")
    projection = _projection_from_value(value)
    if (len(projection.entries) != policy.get("local_endpoint_count")
            or len({item.canonical_endpoint for item in projection.entries})
            != policy.get("canonical_endpoint_count")):
        raise W06R04EndpointProjectionError("endpoint projection count 漂移")
    return projection


def publish_w06_r04_endpoint_projection(
        repo_root: str | Path,
        output_path: str | Path | None = None,
        ) -> Path:
    """以排他创建发布 overlay，已有路径一律拒绝覆盖。"""
    root = Path(repo_root)
    target = root / (output_path or W06_R04_ENDPOINT_PROJECTION_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_w06_r04_endpoint_projection_bytes(root)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W06R04EndpointProjectionError(
            "R04 endpoint projection 已存在，禁止覆盖") from error
    return target


__all__ = [
    "W06R04EndpointProjection",
    "W06R04EndpointProjectionEntry",
    "W06R04EndpointProjectionError",
    "W06_R04_ENDPOINT_PROJECTION_PATH",
    "build_w06_r04_endpoint_projection",
    "canonical_w06_r04_endpoint_projection_bytes",
    "publish_w06_r04_endpoint_projection",
    "read_w06_r04_endpoint_projection",
]
