"""GG-03 formal surface label、owner publication 与 guard 后 materialization。"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    sha256_text,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationEvaluationFamilyError,
    GenerationGeneralizationObservationInventoryIdentity,
    GenerationGeneralizationPrivateLabelOwnerReceipt,
    generation_generalization_sha256_bytes,
    strict_generation_generalization_relative_path,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    generation_generalization_evaluation_requirements,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    EvaluationOneShotGuard,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.runtime import (
    verify_evaluation_guard_consumed,
)


FORMAL_LABEL_ARTIFACT_KIND = "PH2_GG03_FORMAL_SURFACE_LABEL_V1"
FORMAL_LABEL_SPLIT = "held_out"
FORMAL_LABEL_STATUSES = ("PASS", "FAIL", "NE")
PRIVATE_OWNER_RECEIPT_NAME = "owner-receipt.json"
_LABEL_FIELDS = {
    "accepted_surface_sha256", "artifact_kind", "format_version",
    "observation_stable_key_sha256", "rejected_surface_sha256",
    "requirements", "split",
}


def generation_generalization_surface_sha256(surface: str) -> str:
    """把 owner 独立确认的完整 Unicode surface 映射为 exact SHA-256。"""
    if (not isinstance(surface, str) or not surface
            or surface.strip() != surface):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 formal surface 文本非法")
    return hashlib.sha256(surface.encode("utf-8")).hexdigest()


def generation_generalization_observation_key_sha256(
        observation: GenerationGeneralizationEvaluationObservation,
        ) -> str:
    """返回与 family inventory 完全相同的 Observation stable-key SHA。"""
    if not isinstance(
            observation, GenerationGeneralizationEvaluationObservation):
        raise TypeError("GG-03 formal label Observation 类型错误")
    return generation_generalization_sha256_bytes(canonical_json_bytes(
        list(observation.stable_key())))


def _hash_tuple(value: object, *, where: str, minimum: int) -> tuple[str, ...]:
    """核验 canonical、有序、互异 surface SHA 集。"""
    if (not isinstance(value, (list, tuple)) or len(value) < minimum
            or any(not isinstance(item, str) for item in value)):
        raise GenerationGeneralizationEvaluationFamilyError(
            f"{where} 数量或类型非法")
    result = tuple(value)
    for item in result:
        sha256_text(item, where=where)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise GenerationGeneralizationEvaluationFamilyError(
            f"{where} 必须 canonical 且唯一")
    return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class GenerationGeneralizationFormalLabelRecord:
    """一条 Observation 的多合法 surface set 与负向 exact set。"""

    observation_stable_key_sha256: str
    accepted_surface_sha256: tuple[str, ...]
    rejected_surface_sha256: tuple[str, ...]
    requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        sha256_text(
            self.observation_stable_key_sha256,
            where="GG-03 formal label Observation SHA")
        accepted = _hash_tuple(
            self.accepted_surface_sha256,
            where="GG-03 accepted surface SHA", minimum=2)
        rejected = _hash_tuple(
            self.rejected_surface_sha256,
            where="GG-03 rejected surface SHA", minimum=1)
        if set(accepted) & set(rejected):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 accepted/rejected surface set 重叠")
        if (not isinstance(self.requirements, tuple) or not self.requirements
                or self.requirements != tuple(
                    item for item in INDEPENDENT_VERIFIER_REQUIREMENTS
                    if item in self.requirements)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 formal label requirement 顺序非法")

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted_surface_sha256": list(self.accepted_surface_sha256),
            "artifact_kind": FORMAL_LABEL_ARTIFACT_KIND,
            "format_version": 1,
            "observation_stable_key_sha256": (
                self.observation_stable_key_sha256),
            "rejected_surface_sha256": list(self.rejected_surface_sha256),
            "requirements": list(self.requirements),
            "split": FORMAL_LABEL_SPLIT,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GenerationGeneralizationFormalLabelRecord":
        raw = exact_dict(value, _LABEL_FIELDS, where="GG-03 formal label")
        if (raw["artifact_kind"] != FORMAL_LABEL_ARTIFACT_KIND
                or raw["format_version"] != 1
                or raw["split"] != FORMAL_LABEL_SPLIT):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 formal label kind/version/split 漂移")
        return cls(
            str(raw["observation_stable_key_sha256"]),
            tuple(str(item) for item in raw["accepted_surface_sha256"]),
            tuple(str(item) for item in raw["rejected_surface_sha256"]),
            tuple(str(item) for item in raw["requirements"]),
        )

    def verdict_for_surface_sha256(self, surface_sha256: str) -> str:
        """以 frozen exact set 返回 PASS/FAIL/NE，不猜测未登记 surface。"""
        sha256_text(surface_sha256, where="GG-03 predicted surface SHA")
        if surface_sha256 in self.accepted_surface_sha256:
            return "PASS"
        if surface_sha256 in self.rejected_surface_sha256:
            return "FAIL"
        return "NE"


def build_generation_generalization_formal_label_record(
        observation: GenerationGeneralizationEvaluationObservation,
        *,
        accepted_surfaces: Iterable[str],
        rejected_surfaces: Iterable[str],
        ) -> GenerationGeneralizationFormalLabelRecord:
    """owner 从独立审定 surface 集建立不含明文的 formal label。"""
    accepted = tuple(sorted({
        generation_generalization_surface_sha256(item)
        for item in accepted_surfaces
    }))
    rejected = tuple(sorted({
        generation_generalization_surface_sha256(item)
        for item in rejected_surfaces
    }))
    return GenerationGeneralizationFormalLabelRecord(
        generation_generalization_observation_key_sha256(observation),
        accepted,
        rejected,
        generation_generalization_evaluation_requirements(observation),
    )


def _content_bytes(
        records: tuple[GenerationGeneralizationFormalLabelRecord, ...],
        ) -> bytes:
    """形成按 Observation identity 排序的 canonical JSONL 内容。"""
    if (not records
            or len({item.observation_stable_key_sha256 for item in records})
            != len(records)):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 formal label inventory 顺序或唯一性非法")
    return b"".join(canonical_json_line(item.to_dict()) for item in records)


def _label_commitment(
        records: tuple[GenerationGeneralizationFormalLabelRecord, ...],
        ) -> str:
    """冻结 label record 的规范对象 identity，与 gzip transport 分离。"""
    return generation_generalization_sha256_bytes(canonical_json_bytes(
        [item.to_dict() for item in records]))


def _private_root(value: str | Path) -> Path:
    """要求 owner publication 位于 K 盘 `private-label-owner` 根。"""
    root = Path(value).resolve()
    if (not root.is_dir() or root.drive.upper() != "K:"
            or root.name != "private-label-owner"):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label owner root 非法")
    return root


def _label_path(root: Path, relative_path: str) -> Path:
    """把安全 POSIX relative path 恢复到 private owner root。"""
    relative = strict_generation_generalization_relative_path(
        relative_path, where="GG-03 private label")
    target = (root / Path(*relative.split("/"))).resolve()
    if not target.is_relative_to(root):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label 路径越界")
    return target


def publish_generation_generalization_private_labels(
        records: tuple[GenerationGeneralizationFormalLabelRecord, ...],
        *,
        private_label_root: str | Path,
        label_relative_path: str,
        observation_inventory: GenerationGeneralizationObservationInventoryIdentity,
        verdict_contract_sha256: str,
        ) -> dict[str, object]:
    """owner 侧不可覆盖发布 label transport 与 metadata-only receipt。"""
    root = _private_root(private_label_root)
    if not isinstance(
            observation_inventory,
            GenerationGeneralizationObservationInventoryIdentity):
        raise TypeError("GG-03 owner Observation inventory identity 类型错误")
    sha256_text(
        verdict_contract_sha256, where="GG-03 owner verdict contract SHA")
    expected = tuple(
        (item.stable_key_sha256, item.requirements)
        for item in observation_inventory.records)
    actual = tuple(
        (item.observation_stable_key_sha256, item.requirements)
        for item in records)
    if actual != expected:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 formal labels 未精确覆盖冻结 Observation inventory")
    label_path = _label_path(root, label_relative_path)
    receipt_path = root / PRIVATE_OWNER_RECEIPT_NAME
    if label_path.exists() or receipt_path.exists():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label 或 owner receipt 已存在")
    content = _content_bytes(records)
    if label_path.name.endswith(".jsonl.gz"):
        transport = gzip.compress(content, mtime=0)
    elif label_path.suffix == ".jsonl":
        transport = content
    else:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label 必须为 .jsonl 或 .jsonl.gz")
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
            "GG-03 private owner receipt 回读漂移")
    return {
        "label_commitment_sha256": receipt.label_commitment_sha256,
        "label_transport_sha256": receipt.label_transport_sha256,
        "owner_receipt_sha256": generation_generalization_sha256_bytes(
            receipt_path.read_bytes()),
        "record_count": len(records),
        "status": receipt.status,
    }


def read_generation_generalization_private_labels_after_guard(
        *,
        family_root: str | Path,
        expected_guard: EvaluationOneShotGuard,
        prediction_seal_path: str | Path,
        prediction_seal_sha256: str,
        private_label_root: str | Path,
        owner_receipt: GenerationGeneralizationPrivateLabelOwnerReceipt,
        observation_inventory: GenerationGeneralizationObservationInventoryIdentity,
        ) -> tuple[GenerationGeneralizationFormalLabelRecord, ...]:
    """formal guard 消费后按 receipt 物化并严格回验全部 private labels。"""
    family = Path(family_root).resolve()
    verify_evaluation_guard_consumed(family, expected_guard)
    prediction_path = Path(prediction_seal_path).resolve()
    sha256_text(
        prediction_seal_sha256, where="GG-03 prediction seal SHA")
    if (not prediction_path.is_relative_to(family)
            or not prediction_path.is_file()):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label read 前 prediction seal 不存在")
    try:
        prediction_value = read_canonical_object(prediction_path)
    except Exception as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label read 前 prediction seal 不可回读") from error
    if generation_generalization_sha256_bytes(
            canonical_json_bytes(prediction_value)) != prediction_seal_sha256:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label read 前 prediction seal identity 漂移")
    root = _private_root(private_label_root)
    if (not isinstance(owner_receipt, GenerationGeneralizationPrivateLabelOwnerReceipt)
            or not isinstance(
                observation_inventory,
                GenerationGeneralizationObservationInventoryIdentity)):
        raise TypeError("GG-03 private label materialization 输入类型错误")
    if (owner_receipt.observation_inventory_sha256
            != observation_inventory.transport_sha256
            or owner_receipt.label_record_count
            != observation_inventory.record_count):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private owner receipt 与 Observation inventory 漂移")
    target = _label_path(root, owner_receipt.label_relative_path)
    try:
        transport = target.read_bytes()
    except OSError as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label transport 不可读") from error
    if (len(transport) != owner_receipt.label_transport_size_bytes
            or generation_generalization_sha256_bytes(transport)
            != owner_receipt.label_transport_sha256):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label transport identity 漂移")
    try:
        content = (
            gzip.decompress(transport)
            if target.name.endswith(".jsonl.gz") else transport)
    except (OSError, EOFError) as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label gzip 非法") from error
    if (len(content) != owner_receipt.label_content_size_bytes
            or generation_generalization_sha256_bytes(content)
            != owner_receipt.label_content_sha256
            or not content.endswith(b"\n") or content.endswith(b"\n\n")):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label content identity 漂移")
    records = []
    for line in content.splitlines(keepends=True):
        try:
            raw = parse_canonical_json_bytes(line[:-1], require_object=True)
        except ValueError as error:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 private label JSONL 非法") from error
        if canonical_json_line(raw) != line:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 private label record 非 canonical")
        records.append(GenerationGeneralizationFormalLabelRecord.from_dict(raw))
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
            "GG-03 private label inventory commitment 漂移")
    return result


__all__ = [
    "FORMAL_LABEL_ARTIFACT_KIND",
    "FORMAL_LABEL_SPLIT",
    "FORMAL_LABEL_STATUSES",
    "GenerationGeneralizationFormalLabelRecord",
    "build_generation_generalization_formal_label_record",
    "generation_generalization_observation_key_sha256",
    "generation_generalization_surface_sha256",
    "publish_generation_generalization_private_labels",
    "read_generation_generalization_private_labels_after_guard",
]
