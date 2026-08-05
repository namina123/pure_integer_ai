"""W09 Git 外 rotation package 的构造、metadata freeze 与只读 transport。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    StableRecordKey,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    DatasetArtifactIOError,
    read_record_artifact,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09TrainingPayload
from pure_integer_ai.experiments.ph2_w09_inference import (
    W09_INFERENCE_PAYLOAD_KINDS,
)


W09_ROTATION_VERSION = "PH2-W09-ROTATION-V1"
W09_ROTATION_MANIFEST_NAME = "w09_rotation_package_v1.json"
W09_ROTATION_PACK_KEY = "W09-INDEPENDENT-ROTATION-20260805-V1"
_LOCATION_ROTATION = {
    "东门": "甲台",
    "西门": "乙台",
    "云台": "星港",
    "河台": "石湾",
    "三号库": "七号柜",
    "北川": "青浦",
    "南岭": "赤丘",
    "北库": "上舱",
    "南库": "下舱",
    "东塔": "前站",
    "西台": "后站",
}


class W09RotationError(RuntimeError):
    """rotation package 的 owner、identity、路径或 commitment 漂移。"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _commitment(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W09RotationError("rotation relative path 非法")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise W09RotationError("rotation relative path 越界")
    return path.as_posix()


def _is_reparse(path: Path) -> bool:
    try:
        value = path.stat(follow_symlinks=False)
    except OSError as error:
        raise W09RotationError("rotation path 无法 stat") from error
    return bool(getattr(value, "st_file_attributes", 0) & 0x400)


def _root(value: str | Path, *, create: bool = False) -> Path:
    path = Path(value)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink() or _is_reparse(path):
        raise W09RotationError("rotation root 不是普通目录")
    return path.resolve()


def _resolve(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / Path(*PurePosixPath(_safe_relative(relative)).parts)).resolve()
    if not path.is_relative_to(root):
        raise W09RotationError("rotation payload path 越界")
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.exists() and (current.is_symlink() or _is_reparse(current)):
            raise W09RotationError("rotation payload path 含 link/reparse")
    return path


def _rotate_text(value: str) -> str:
    result = value
    for before, after in _LOCATION_ROTATION.items():
        result = result.replace(before, after)
    return result


def _rotate_value(value: object, *, field: str = "") -> object:
    """只替换 identifier/text，不改变状态、枚举、逻辑位或结构。"""
    if isinstance(value, dict):
        result = {
            str(key): _rotate_value(item, field=str(key))
            for key, item in sorted(value.items())
        }
        if isinstance(result.get("text"), str) and "sha256" in result:
            result["sha256"] = _sha256(str(result["text"]).encode("utf-8"))
        if isinstance(result.get("raw_text"), str) and "raw_sha256" in result:
            result["raw_sha256"] = _sha256(str(result["raw_text"]).encode("utf-8"))
        if isinstance(result.get("raw_observation"), dict) and "raw_observation_sha256" in result:
            result["raw_observation_sha256"] = _sha256(canonical_json_bytes(result["raw_observation"]))
        return result
    if isinstance(value, list):
        return [_rotate_value(item, field=field) for item in value]
    if isinstance(value, str):
        lowered = field.lower()
        if any(token in lowered for token in ("surface", "text", "fragment")):
            return _rotate_text(value)
        if lowered.endswith(("_id", "_key")) or lowered in {
            "candidate_id", "record_key", "seed_id", "source_key",
        }:
            return f"R09::{value}" if value else value
        if value.startswith(("T_", "GG03_T", "tr_", "teacher-")):
            return f"R09::{value}"
        return _rotate_text(value)
    return value


def _record_key(kind: int, ordinal: int) -> StableRecordKey:
    return StableRecordKey((90910, kind, ordinal))


@dataclass(frozen=True)
class W09RotationRecords:
    """package builder 产生的三 owner record inventory。"""

    source_refs: tuple[object, ...]
    observations: tuple[ObservationRecord, ...]
    labels: tuple[EvaluatorLabelRecord, ...]
    training_observation_commitment: str
    rotation_case_commitment: str

    def __post_init__(self) -> None:
        if len(self.observations) != 309 or len(self.labels) != 309:
            raise W09RotationError("rotation case/label count 不闭合")
        if {item.stable_key for item in self.observations} != {
            item.observation_key for item in self.labels
        }:
            raise W09RotationError("rotation label 引用不闭合")
        if {item.payload_kind for item in self.observations} != set(W09_INFERENCE_PAYLOAD_KINDS):
            raise W09RotationError("rotation payload kind inventory 不闭合")
        if self.training_observation_commitment == self.rotation_case_commitment:
            raise W09RotationError("rotation case 未离开 train identity/content")


def build_w09_rotation_records(payload: W09TrainingPayload) -> W09RotationRecords:
    """从 train-only material 构造新 identity/text rotation；不接收 Candidate。"""
    if not isinstance(payload, W09TrainingPayload):
        raise TypeError("rotation builder 只接受 W09TrainingPayload")
    evidence = {item.observation_key: item for item in payload.training_evidence}
    old_to_new = {
        item.stable_key: _record_key(10, ordinal)
        for ordinal, item in enumerate(payload.observations, start=1)
    }
    observations: list[ObservationRecord] = []
    labels: list[EvaluatorLabelRecord] = []
    for ordinal, item in enumerate(payload.observations, start=1):
        typed = CanonicalJsonObject.from_value(
            _rotate_value(item.typed_payload.to_value())
        )
        observation = replace(
            item,
            artifact_key=_record_key(1, 1),
            stable_key=old_to_new[item.stable_key],
            split="held_out",
            dedup_cluster_key=_record_key(20, ordinal),
            content_group_key=_record_key(21, ordinal),
            template_group_key=_record_key(22, ordinal),
            shape_group_key=_record_key(23, ordinal),
            typed_payload=typed,
            supersedes_key=(
                old_to_new.get(item.supersedes_key)
                if item.supersedes_key is not None else None
            ),
            prerequisite_keys=tuple(
                old_to_new[key] for key in item.prerequisite_keys
                if key in old_to_new
            ),
            logical_order=ordinal,
        )
        teacher = evidence[item.stable_key]
        teacher_value = teacher.typed_evidence.to_value()
        state = teacher_value.get("expected_state", "TRUE")
        result = teacher_value.get("expected_payload", teacher_value)
        label = EvaluatorLabelRecord(
            1,
            1,
            1,
            item.dataset_key,
            _record_key(2, 1),
            _record_key(30, ordinal),
            observation.stable_key,
            _record_key(31, ordinal),
            str(state),
            CanonicalJsonObject.from_value(_rotate_value(result)),
            2048,
            1,
            "W-09",
            _record_key(32, 1),
        )
        observations.append(observation)
        labels.append(label)
    training_commitment = _commitment([
        [list(item.stable_key.components), item.typed_payload.to_value()]
        for item in payload.observations
    ])
    rotation_commitment = _commitment([
        [list(item.stable_key.components), item.typed_payload.to_value()]
        for item in observations
    ])
    return W09RotationRecords(
        payload.source_refs,
        tuple(observations),
        tuple(labels),
        training_commitment,
        rotation_commitment,
    )


@dataclass(frozen=True)
class W09RotationManifest:
    """外部 package 的三 owner file identity 与派生 commitment。"""

    package_version: str
    source_identity: ArtifactFileIdentity
    observation_identity: ArtifactFileIdentity
    label_identity: ArtifactFileIdentity
    payload_kind_inventory: tuple[str, ...]
    training_observation_commitment: str
    rotation_case_commitment: str
    package_commitment: str

    def __post_init__(self) -> None:
        if self.package_version != W09_ROTATION_VERSION:
            raise W09RotationError("rotation version 漂移")
        identities = (self.source_identity, self.observation_identity, self.label_identity)
        if any(not isinstance(item, ArtifactFileIdentity) for item in identities):
            raise W09RotationError("rotation file identity 非法")
        if tuple(item.owner_kind for item in identities) != ("source", "observation", "evaluator"):
            raise W09RotationError("rotation owner 顺序漂移")
        if self.observation_identity.record_count != self.label_identity.record_count or self.observation_identity.record_count != 309:
            raise W09RotationError("rotation Observation/label count 漂移")
        if self.payload_kind_inventory != tuple(sorted(W09_INFERENCE_PAYLOAD_KINDS)):
            raise W09RotationError("rotation payload kinds 不闭合")
        for value in (
            self.training_observation_commitment,
            self.rotation_case_commitment,
            self.package_commitment,
        ):
            if len(value) != 64:
                raise W09RotationError("rotation commitment 非法")
        if self.package_commitment != _commitment(self._identity_dict()):
            raise W09RotationError("rotation package commitment 漂移")

    def _identity_dict(self) -> dict[str, object]:
        return {
            "label_identity": self.label_identity.to_dict(),
            "observation_identity": self.observation_identity.to_dict(),
            "package_version": self.package_version,
            "payload_kind_inventory": list(self.payload_kind_inventory),
            "rotation_case_commitment": self.rotation_case_commitment,
            "source_identity": self.source_identity.to_dict(),
            "training_observation_commitment": self.training_observation_commitment,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": "PH2_W09_ROTATION_PACKAGE_MANIFEST",
            "format_version": 1,
            **self._identity_dict(),
            "package_commitment": self.package_commitment,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W09RotationManifest":
        if value.get("artifact_kind") != "PH2_W09_ROTATION_PACKAGE_MANIFEST" or value.get("format_version") != 1:
            raise W09RotationError("rotation manifest envelope 漂移")
        try:
            return cls(
                str(value["package_version"]),
                ArtifactFileIdentity.from_dict(value["source_identity"]),
                ArtifactFileIdentity.from_dict(value["observation_identity"]),
                ArtifactFileIdentity.from_dict(value["label_identity"]),
                tuple(str(item) for item in value["payload_kind_inventory"]),
                str(value["training_observation_commitment"]),
                str(value["rotation_case_commitment"]),
                str(value["package_commitment"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise W09RotationError("rotation manifest fields 损坏") from error


def _manifest(
    source: ArtifactFileIdentity,
    observation: ArtifactFileIdentity,
    label: ArtifactFileIdentity,
    records: W09RotationRecords,
) -> W09RotationManifest:
    partial = W09RotationManifest.__new__(W09RotationManifest)
    object.__setattr__(partial, "package_version", W09_ROTATION_VERSION)
    object.__setattr__(partial, "source_identity", source)
    object.__setattr__(partial, "observation_identity", observation)
    object.__setattr__(partial, "label_identity", label)
    object.__setattr__(partial, "payload_kind_inventory", tuple(sorted(W09_INFERENCE_PAYLOAD_KINDS)))
    object.__setattr__(partial, "training_observation_commitment", records.training_observation_commitment)
    object.__setattr__(partial, "rotation_case_commitment", records.rotation_case_commitment)
    object.__setattr__(partial, "package_commitment", "")
    commitment = _commitment(partial._identity_dict())
    return W09RotationManifest(
        W09_ROTATION_VERSION,
        source,
        observation,
        label,
        tuple(sorted(W09_INFERENCE_PAYLOAD_KINDS)),
        records.training_observation_commitment,
        records.rotation_case_commitment,
        commitment,
    )


def write_w09_rotation_package(root: str | Path, records: W09RotationRecords) -> tuple[Path, str, W09RotationManifest]:
    """排他写三 owner artifact 和 manifest；不覆盖任何既存 family。"""
    if not isinstance(records, W09RotationRecords):
        raise TypeError("rotation records 类型非法")
    target = _root(root, create=True)
    source = write_record_artifact(
        records.source_refs,
        target,
        ArtifactWriteSpec(
            "source_ref", "source", "source_refs.jsonl.gz", None, "CC0-1.0",
            tuple(sorted({item.source_cluster_key for item in records.source_refs})),
        ),
    )
    observations = write_record_artifact(
        records.observations,
        target,
        ArtifactWriteSpec(
            "observation", "observation", "observations/held_out.jsonl.gz",
            "held_out", "CC0-1.0",
            tuple(item.dedup_cluster_key for item in records.observations),
        ),
    )
    labels = write_record_artifact(
        records.labels,
        target,
        ArtifactWriteSpec(
            "evaluator_label", "evaluator", "owners/evaluator/held_out.labels.jsonl.gz",
            "held_out", "CC0-1.0",
            tuple(item.observation_key for item in records.labels),
        ),
    )
    manifest = _manifest(source, observations, labels, records)
    path = target / W09_ROTATION_MANIFEST_NAME
    try:
        with path.open("xb") as handle:
            encoded = manifest.canonical_bytes()
            handle.write(encoded)
    except FileExistsError as error:
        raise W09RotationError("rotation manifest 不可覆盖") from error
    return path, _sha256(encoded), manifest


def read_w09_rotation_manifest(root: str | Path, *, expected_sha256: str | None = None) -> W09RotationManifest:
    """只回读 package metadata；不读取 Observation 或 label payload。"""
    target = _root(root)
    path = _resolve(target, W09_ROTATION_MANIFEST_NAME)
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W09RotationError("rotation manifest 无法读取") from error
    if expected_sha256 is not None and _sha256(payload) != expected_sha256:
        raise W09RotationError("rotation manifest SHA 漂移")
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W09RotationError("rotation manifest 非 canonical")
    result = W09RotationManifest.from_dict(value)
    if result.canonical_bytes() != payload:
        raise W09RotationError("rotation manifest readback 漂移")
    return result


def read_w09_rotation_binding(root: str | Path, identity: ArtifactFileIdentity) -> tuple[object, ...]:
    """按 manifest-bound identity 读取一个 owner 文件并核验 transport。"""
    if not isinstance(identity, ArtifactFileIdentity):
        raise TypeError("rotation binding identity 类型非法")
    target = _root(root)
    path = _resolve(target, identity.relative_path)
    artifact_root = path.parents[len(PurePosixPath(identity.relative_path).parts) - 1]
    try:
        result = read_record_artifact(artifact_root, identity)
    except (DatasetArtifactIOError, OSError, ValueError) as error:
        raise W09RotationError("rotation binding transport 失败") from error
    if len(result) != identity.record_count:
        raise W09RotationError("rotation binding count 漂移")
    return result


def validate_w09_rotation_metadata(root: str | Path, manifest: W09RotationManifest) -> None:
    """family freeze 前只 stat 三 owner file，不读取 payload bytes。"""
    target = _root(root)
    allowed = {W09_ROTATION_MANIFEST_NAME}
    for identity in (manifest.source_identity, manifest.observation_identity, manifest.label_identity):
        path = _resolve(target, identity.relative_path)
        if not path.is_file() or path.stat().st_size != identity.transport_size_bytes:
            raise W09RotationError("rotation binding metadata 漂移")
        allowed.add(identity.relative_path)
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*") if path.is_file()
    }
    if actual != allowed:
        raise W09RotationError("rotation package 含未绑定文件")


__all__ = [
    "W09RotationError",
    "W09RotationManifest",
    "W09RotationRecords",
    "W09_ROTATION_MANIFEST_NAME",
    "W09_ROTATION_PACK_KEY",
    "W09_ROTATION_VERSION",
    "build_w09_rotation_records",
    "read_w09_rotation_binding",
    "read_w09_rotation_manifest",
    "validate_w09_rotation_metadata",
    "write_w09_rotation_package",
]
