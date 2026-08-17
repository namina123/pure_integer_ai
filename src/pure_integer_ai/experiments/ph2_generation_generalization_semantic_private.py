"""GG-03 V2 语义标签的 K 盘发布与 seal 后物化。"""
from __future__ import annotations

import gzip
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    sha256_text,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    EvaluationOneShotGuard,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.runtime import (
    verify_evaluation_guard_consumed,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationEvaluationFamilyError,
    GenerationGeneralizationObservationInventoryIdentity,
    GenerationGeneralizationPrivateLabelOwnerReceipt,
    generation_generalization_sha256_bytes,
    strict_generation_generalization_relative_path,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_labels import (
    PRIVATE_OWNER_RECEIPT_NAME,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    GenerationGeneralizationSemanticLabelRecord,
    generation_generalization_semantic_verdict_contract_sha256,
)


def _private_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if (not root.is_dir() or root.drive.upper() != "K:"
            or root.name != "private-label-owner"):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic private label owner root 非法")
    return root


def _label_path(root: Path, relative_path: str) -> Path:
    relative = strict_generation_generalization_relative_path(
        relative_path, where="GG-03 semantic private label")
    target = (root / Path(*relative.split("/"))).resolve()
    if not target.is_relative_to(root):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic private label 路径越界")
    return target


def _content_bytes(
        records: tuple[GenerationGeneralizationSemanticLabelRecord, ...],
        ) -> bytes:
    if (not records
            or len({item.observation_stable_key_sha256 for item in records})
            != len(records)):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label inventory 顺序或唯一性非法")
    return b"".join(canonical_json_line(item.to_dict()) for item in records)


def _label_commitment(
        records: tuple[GenerationGeneralizationSemanticLabelRecord, ...],
        ) -> str:
    return generation_generalization_sha256_bytes(canonical_json_bytes(
        [item.to_dict() for item in records]))


def publish_generation_generalization_semantic_labels(
        records: tuple[GenerationGeneralizationSemanticLabelRecord, ...],
        *,
        private_label_root: str | Path,
        label_relative_path: str,
        observation_inventory: GenerationGeneralizationObservationInventoryIdentity,
        verdict_contract_sha256: str,
        ) -> dict[str, object]:
    """不可覆盖发布 V2 语义标签和仅含元数据的 owner receipt。"""
    root = _private_root(private_label_root)
    if not isinstance(
            observation_inventory,
            GenerationGeneralizationObservationInventoryIdentity):
        raise TypeError("GG-03 semantic owner inventory 类型错误")
    sha256_text(
        verdict_contract_sha256,
        where="GG-03 semantic owner verdict contract SHA",
    )
    if verdict_contract_sha256 != (
            generation_generalization_semantic_verdict_contract_sha256()):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic verdict contract 漂移")
    expected = tuple(
        (item.stable_key_sha256, item.requirements)
        for item in observation_inventory.records)
    actual = tuple(
        (item.observation_stable_key_sha256, item.requirements)
        for item in records)
    if actual != expected:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic labels 未精确覆盖 Observation inventory")
    label_path = _label_path(root, label_relative_path)
    receipt_path = root / PRIVATE_OWNER_RECEIPT_NAME
    if label_path.exists() or receipt_path.exists():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label 或 owner receipt 已存在")
    content = _content_bytes(records)
    if label_path.name.endswith(".jsonl.gz"):
        transport = gzip.compress(content, mtime=0)
    elif label_path.suffix == ".jsonl":
        transport = content
    else:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label 必须为 .jsonl 或 .jsonl.gz")
    if label_path.parent != root:
        label_path.parent.mkdir(parents=True, exist_ok=False)
    with label_path.open("xb") as handle:
        handle.write(transport)
    receipt = GenerationGeneralizationPrivateLabelOwnerReceipt(
        observation_inventory.transport_sha256,
        label_relative_path,
        len(transport),
        generation_generalization_sha256_bytes(transport),
        len(content),
        generation_generalization_sha256_bytes(content),
        len(records),
        _label_commitment(records),
        verdict_contract_sha256,
    )
    write_immutable_json(receipt.to_dict(), receipt_path)
    if read_canonical_object(receipt_path) != receipt.to_dict():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic owner receipt 回读漂移")
    return {
        "label_commitment_sha256": receipt.label_commitment_sha256,
        "label_transport_sha256": receipt.label_transport_sha256,
        "owner_receipt_sha256": generation_generalization_sha256_bytes(
            receipt_path.read_bytes()),
        "record_count": len(records),
        "status": receipt.status,
    }


def read_generation_generalization_semantic_labels_after_guard(
        *,
        family_root: str | Path,
        expected_guard: EvaluationOneShotGuard,
        prediction_seal_path: str | Path,
        prediction_seal_sha256: str,
        private_label_root: str | Path,
        owner_receipt: GenerationGeneralizationPrivateLabelOwnerReceipt,
        observation_inventory: GenerationGeneralizationObservationInventoryIdentity,
        ) -> tuple[GenerationGeneralizationSemanticLabelRecord, ...]:
    """仅在 guard 已消费且 prediction seal 严格闭合后读取 V2 标签。"""
    family = Path(family_root).resolve()
    verify_evaluation_guard_consumed(family, expected_guard)
    prediction_path = Path(prediction_seal_path).resolve()
    sha256_text(
        prediction_seal_sha256,
        where="GG-03 semantic prediction seal SHA",
    )
    if (not prediction_path.is_relative_to(family)
            or not prediction_path.is_file()):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label read 前 prediction seal 不存在")
    try:
        prediction_value = read_canonical_object(prediction_path)
    except Exception as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label read 前 prediction seal 不可回读") from error
    if generation_generalization_sha256_bytes(canonical_json_bytes(
            prediction_value)) != prediction_seal_sha256:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic prediction seal identity 漂移")
    root = _private_root(private_label_root)
    if (not isinstance(
            owner_receipt, GenerationGeneralizationPrivateLabelOwnerReceipt)
            or not isinstance(
                observation_inventory,
                GenerationGeneralizationObservationInventoryIdentity)):
        raise TypeError("GG-03 semantic label materialization 输入类型错误")
    if (owner_receipt.verdict_contract_sha256
            != generation_generalization_semantic_verdict_contract_sha256()
            or owner_receipt.observation_inventory_sha256
            != observation_inventory.transport_sha256
            or owner_receipt.label_record_count
            != observation_inventory.record_count):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic owner receipt 与 inventory/contract 漂移")
    target = _label_path(root, owner_receipt.label_relative_path)
    try:
        transport = target.read_bytes()
    except OSError as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label transport 不可读") from error
    if (len(transport) != owner_receipt.label_transport_size_bytes
            or generation_generalization_sha256_bytes(transport)
            != owner_receipt.label_transport_sha256):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label transport identity 漂移")
    try:
        content = (
            gzip.decompress(transport)
            if target.name.endswith(".jsonl.gz") else transport)
    except (OSError, EOFError) as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label gzip 非法") from error
    if (len(content) != owner_receipt.label_content_size_bytes
            or generation_generalization_sha256_bytes(content)
            != owner_receipt.label_content_sha256
            or not content.endswith(b"\n") or content.endswith(b"\n\n")):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label content identity 漂移")
    records = []
    for line in content.splitlines(keepends=True):
        try:
            raw = parse_canonical_json_bytes(line[:-1], require_object=True)
        except ValueError as error:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic label JSONL 非法") from error
        if canonical_json_line(raw) != line:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic label record 非 canonical")
        records.append(GenerationGeneralizationSemanticLabelRecord.from_dict(
            raw))
    result = tuple(records)
    expected = tuple(
        (item.stable_key_sha256, item.requirements)
        for item in observation_inventory.records)
    actual = tuple(
        (item.observation_stable_key_sha256, item.requirements)
        for item in result)
    if (len(result) != owner_receipt.label_record_count
            or actual != expected
            or _label_commitment(result)
            != owner_receipt.label_commitment_sha256):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic label inventory commitment 漂移")
    return result


__all__ = [
    "publish_generation_generalization_semantic_labels",
    "read_generation_generalization_semantic_labels_after_guard",
]
