"""``CONFLICT_SET`` semantic label codec 与 prediction-seal 后严格读取。"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_family import (
    FAMILY_FREEZE_MANIFEST_NAME,
    ConflictSetFamilyFreeze,
    read_conflict_set_family_freeze,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_protocol import (
    ConflictSetSemanticLabelRecord,
    ConflictSetSemanticPredictionSeal,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    TRANSPORT_ROOT_NAMESPACE,
    ConflictSetPrivateArtifact,
    ConflictSetRunGuard,
    ConflictSetRunIntent,
    consume_conflict_set_run_guard,
    strict_conflict_set_relative_path,
)


# object-model: exception
class ConflictSetFormalPrivateError(ValueError):
    """private label transport 或 pre-label 运行序违反冻结合同。"""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _labels_content(
        records: tuple[ConflictSetSemanticLabelRecord, ...],
        ) -> bytes:
    if (not isinstance(records, tuple) or not records
            or any(not isinstance(item, ConflictSetSemanticLabelRecord)
                   for item in records)):
        raise TypeError("semantic labels 必须是非空 typed tuple")
    keys = tuple(item.observation_stable_key_sha256 for item in records)
    if keys != tuple(sorted(set(keys))):
        raise ConflictSetFormalPrivateError(
            "semantic labels 必须按唯一 Observation identity 排序")
    return b"".join(canonical_json_line(item.to_dict()) for item in records)


def conflict_set_semantic_label_commitment_sha256(
        records: tuple[ConflictSetSemanticLabelRecord, ...],
        ) -> str:
    """返回与 transport 编码无关的 semantic label inventory commitment。"""
    _labels_content(records)
    return _sha(canonical_json_bytes([item.to_dict() for item in records]))


def encode_conflict_set_semantic_labels(
        records: tuple[ConflictSetSemanticLabelRecord, ...],
        *, compressed: bool,
        ) -> tuple[bytes, bytes]:
    """确定性编码 label；返回 transport bytes 与未压缩 canonical JSONL。"""
    if type(compressed) is not bool:
        raise TypeError("compressed 必须是 bool")
    content = _labels_content(records)
    return (
        gzip.compress(content, mtime=0) if compressed else content,
        content,
    )


def publish_conflict_set_semantic_labels(
        records: tuple[ConflictSetSemanticLabelRecord, ...],
        *,
        run_root: str | Path,
        artifact: ConflictSetPrivateArtifact,
        ) -> dict[str, object]:
    """按冻结 private-label artifact 不可覆盖物化 owner label transport。"""
    if not isinstance(artifact, ConflictSetPrivateArtifact):
        raise TypeError("private label artifact 类型错误")
    if artifact.role != "private_labels":
        raise ConflictSetFormalPrivateError(
            "publisher 只接受 private_labels artifact")
    root = _run_root(run_root)
    target = _artifact_path(root, artifact)
    if target.name.endswith(".jsonl.gz"):
        transport, content = encode_conflict_set_semantic_labels(
            records, compressed=True)
    elif target.name.endswith(".jsonl"):
        transport, content = encode_conflict_set_semantic_labels(
            records, compressed=False)
    else:
        raise ConflictSetFormalPrivateError(
            "private label transport 必须为 .jsonl 或 .jsonl.gz")
    if (artifact.record_count != len(records)
            or artifact.transport_size_bytes != len(transport)
            or artifact.content_size_bytes != len(content)
            or artifact.transport_sha256 != _sha(transport)
            or artifact.content_sha256 != _sha(content)):
        raise ConflictSetFormalPrivateError(
            "private label artifact commitment 与 records 不一致")
    if target.exists() or target.is_symlink():
        raise ConflictSetFormalPrivateError(
            "private label transport 已存在且不可覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ConflictSetFormalPrivateError(
            "private label transport parent 不得为 symlink")
    with target.open("xb") as handle:
        handle.write(transport)
    if target.read_bytes() != transport:
        raise ConflictSetFormalPrivateError(
            "private label transport 写入回读漂移")
    return {
        "artifact_kind": "PH2_GG03_CONFLICT_SET_SEMANTIC_LABEL_PUBLICATION_V1",
        "content_sha256": _sha(content),
        "content_size_bytes": len(content),
        "label_commitment_sha256": (
            conflict_set_semantic_label_commitment_sha256(records)),
        "record_count": len(records),
        "relative_path": artifact.relative_path,
        "status": "PUBLISHED",
        "transport_sha256": _sha(transport),
        "transport_size_bytes": len(transport),
    }


def parse_conflict_set_semantic_label_content(
        content: bytes,
        ) -> tuple[ConflictSetSemanticLabelRecord, ...]:
    """严格解析完整 canonical label JSONL，不接受尾随空行或未知字段。"""
    if (not isinstance(content, bytes) or not content
            or not content.endswith(b"\n") or content.endswith(b"\n\n")):
        raise ConflictSetFormalPrivateError(
            "semantic label content 必须是非空 canonical JSONL")
    records = []
    for line in content.splitlines(keepends=True):
        try:
            value = parse_canonical_json_bytes(
                line[:-1], require_object=True)
            if canonical_json_line(value) != line:
                raise ValueError("record bytes are not canonical")
            records.append(ConflictSetSemanticLabelRecord.from_dict(value))
        except (TypeError, ValueError) as error:
            raise ConflictSetFormalPrivateError(
                "semantic label record 非 canonical 或合同漂移") from error
    result = tuple(records)
    _labels_content(result)
    return result


def _artifact(
        freeze: ConflictSetFamilyFreeze, role: str,
        ) -> ConflictSetPrivateArtifact:
    matches = tuple(item for item in freeze.transport.artifacts
                    if item.role == role)
    if len(matches) != 1:
        raise ConflictSetFormalPrivateError(
            f"family freeze 缺唯一 {role} artifact")
    return matches[0]


def _run_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise ConflictSetFormalPrivateError(
            "formal run root 必须位于 K 盘")
    return root


def _artifact_path(
        run_root: Path,
        artifact: ConflictSetPrivateArtifact,
        ) -> Path:
    relative = strict_conflict_set_relative_path(
        artifact.relative_path, where=f"{artifact.role}.relative_path")
    target = (run_root / Path(*relative.split("/"))).resolve()
    namespace = (run_root / TRANSPORT_ROOT_NAMESPACE).resolve()
    if (not target.is_relative_to(namespace)
            or target.is_symlink()):
        raise ConflictSetFormalPrivateError(
            f"{artifact.role} artifact 路径越界或为 symlink")
    return target


def _read_prediction_seal(
        run_root: Path,
        artifact: ConflictSetPrivateArtifact,
        prediction: ConflictSetSemanticPredictionSeal,
        ) -> None:
    target = _artifact_path(run_root, artifact)
    if not target.is_file():
        raise ConflictSetFormalPrivateError(
            "private label read 前 prediction seal 不存在")
    try:
        value = read_canonical_object(target)
    except Exception as error:
        raise ConflictSetFormalPrivateError(
            "prediction seal 无法 canonical 回读") from error
    if (value != prediction.to_dict()
            or _sha(canonical_json_bytes(value)) != prediction.sha256()):
        raise ConflictSetFormalPrivateError("prediction seal identity 漂移")


def _read_consumed_guard(
        family: Path,
        available: ConflictSetRunGuard,
        ) -> ConflictSetRunGuard:
    available_path = family / "guard.available.json"
    consumed_path = family / "guard.consumed.json"
    intent_path = family / "run.intent.json"
    if (available_path.exists() or not consumed_path.is_file()
            or consumed_path.is_symlink() or not intent_path.is_file()
            or intent_path.is_symlink()):
        raise ConflictSetFormalPrivateError(
            "private label read 前 guard/intent 尚未持久消费")
    try:
        consumed = ConflictSetRunGuard.from_dict(
            read_canonical_object(consumed_path))
        intent = ConflictSetRunIntent.from_dict(
            read_canonical_object(intent_path))
    except Exception as error:
        raise ConflictSetFormalPrivateError(
            "consumed guard/intent 无法严格回读") from error
    expected, expected_intent = consume_conflict_set_run_guard(available)
    if consumed != expected or intent != expected_intent:
        raise ConflictSetFormalPrivateError(
            "consumed guard/intent identity 漂移")
    return consumed


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetPrivateLabelRead:
    """一次 seal 后 label 读取的安全内存结果。"""

    records: tuple[ConflictSetSemanticLabelRecord, ...]
    label_commitment_sha256: str
    transport_size_bytes: int
    content_size_bytes: int
    read_count: int = 1

    def __post_init__(self) -> None:
        if (not self.records
                or conflict_set_semantic_label_commitment_sha256(self.records)
                != self.label_commitment_sha256):
            raise ConflictSetFormalPrivateError(
                "private label read commitment 漂移")
        if (type(self.transport_size_bytes) is not int
                or self.transport_size_bytes <= 0
                or type(self.content_size_bytes) is not int
                or self.content_size_bytes <= 0
                or self.read_count != 1):
            raise ConflictSetFormalPrivateError(
                "private label read 计数或字节审计非法")


def read_conflict_set_semantic_labels_after_prediction_seal(
        *,
        run_root: str | Path,
        family_root: str | Path,
        freeze: ConflictSetFamilyFreeze,
        prediction: ConflictSetSemanticPredictionSeal,
        ) -> ConflictSetPrivateLabelRead:
    """先核 guard 与 prediction seal，再按 freeze 唯一读取 private label。"""
    if (not isinstance(freeze, ConflictSetFamilyFreeze)
            or not isinstance(prediction, ConflictSetSemanticPredictionSeal)):
        raise TypeError("family freeze 或 prediction seal 类型错误")
    run = _run_root(run_root)
    family = Path(family_root).resolve()
    if (not family.is_dir() or not family.is_relative_to(run)
            or family.is_symlink()):
        raise ConflictSetFormalPrivateError(
            "formal family root 必须位于同一 run root")
    try:
        published_freeze = read_conflict_set_family_freeze(
            family / FAMILY_FREEZE_MANIFEST_NAME)
    except Exception as error:
        raise ConflictSetFormalPrivateError(
            "private label read 前 family freeze 无法严格回读") from error
    if published_freeze != freeze:
        raise ConflictSetFormalPrivateError(
            "private label read 前 family freeze identity 漂移")
    if (prediction.family_manifest_sha256 != freeze.sha256()
            or prediction.family_commitment_sha256
            != freeze.family_commitment_sha256
            or prediction.candidate_manifest_sha256
            != freeze.transport.candidate_manifest_sha256):
        raise ConflictSetFormalPrivateError(
            "prediction seal 未绑定当前 family/candidate")
    _read_consumed_guard(family, freeze.available_guard)
    _read_prediction_seal(
        run, _artifact(freeze, "prediction_seal"), prediction)
    labels_artifact = _artifact(freeze, "private_labels")
    target = _artifact_path(run, labels_artifact)
    if not target.is_file():
        raise ConflictSetFormalPrivateError("private label transport 不存在")
    try:
        transport = target.read_bytes()
    except OSError as error:
        raise ConflictSetFormalPrivateError(
            "private label transport 不可读") from error
    if (labels_artifact.transport_sha256 != _sha(transport)
            or labels_artifact.transport_size_bytes != len(transport)):
        raise ConflictSetFormalPrivateError(
            "private label transport identity 漂移")
    try:
        content = (
            gzip.decompress(transport)
            if target.name.endswith(".jsonl.gz") else transport)
    except (OSError, EOFError) as error:
        raise ConflictSetFormalPrivateError(
            "private label gzip transport 非法") from error
    if (labels_artifact.content_sha256 != _sha(content)
            or labels_artifact.content_size_bytes != len(content)):
        raise ConflictSetFormalPrivateError(
            "private label content identity 漂移")
    records = parse_conflict_set_semantic_label_content(content)
    if (len(records) != labels_artifact.record_count
            or tuple(item.observation_stable_key_sha256 for item in records)
            != tuple(item.observation_stable_key_sha256
                     for item in prediction.records)):
        raise ConflictSetFormalPrivateError(
            "private label inventory 未覆盖 prediction seal")
    return ConflictSetPrivateLabelRead(
        records,
        conflict_set_semantic_label_commitment_sha256(records),
        len(transport),
        len(content),
    )


__all__ = [
    "ConflictSetFormalPrivateError",
    "ConflictSetPrivateLabelRead",
    "conflict_set_semantic_label_commitment_sha256",
    "encode_conflict_set_semantic_labels",
    "publish_conflict_set_semantic_labels",
    "parse_conflict_set_semantic_label_content",
    "read_conflict_set_semantic_labels_after_prediction_seal",
]
