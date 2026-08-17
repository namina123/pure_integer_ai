"""GG-03 V2 语义 prediction seal、aggregate 与安全 publication 值。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    sha256_text,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationEvaluationFamilyError,
    GenerationGeneralizationObservationInventoryIdentity,
    generation_generalization_sha256_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationActualRun,
    GenerationGeneralizationEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_labels import (
    FORMAL_LABEL_STATUSES,
    generation_generalization_observation_key_sha256,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_protocol import (
    GenerationGeneralizationFormalDimensionResult,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    GenerationGeneralizationSemanticLabelRecord,
    build_actual_generation_generalization_semantic_projection,
    generation_generalization_semantic_verdict_contract_sha256,
)


SEMANTIC_PREDICTION_SEAL_ARTIFACT_KIND = (
    "PH2_GG03_FORMAL_SEMANTIC_PREDICTION_SEAL_V2")
SEMANTIC_FORMAL_AGGREGATE_ARTIFACT_KIND = (
    "PH2_GG03_FORMAL_SEMANTIC_AGGREGATE_V2")
SEMANTIC_FORMAL_RUNTIME_RECEIPT_ARTIFACT_KIND = (
    "PH2_GG03_FORMAL_SEMANTIC_RUNTIME_RECEIPT_V2")
SEMANTIC_FORMAL_FAILURE_SEAL_ARTIFACT_KIND = (
    "PH2_GG03_FORMAL_SEMANTIC_FAILURE_SEAL_V2")
SEMANTIC_FORMAL_FAILURE_DIAGNOSTIC_ARTIFACT_KIND = (
    "PH2_GG03_FORMAL_SEMANTIC_FAILURE_DIAGNOSTIC_V2")
SEMANTIC_PREDICTION_SEAL_STATUS = (
    "SEMANTIC_PREDICTIONS_SEALED_LABELS_UNREAD")


def _status_from_counts(passed: int, failed: int, ne: int) -> str:
    """按 FAIL > NE > PASS 从完整非空计数导出三态。"""
    if any(type(value) is not int or value < 0 for value in (
            passed, failed, ne)):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic status count 非法")
    if passed + failed + ne <= 0:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic status denominator 为空")
    if failed:
        return "FAIL"
    if ne:
        return "NE"
    return "PASS"


def _combined_status(internal: str, semantic: str) -> str:
    """把 actual verifier 与 private semantic verdict 作为同一 hard conjunct。"""
    if (internal not in FORMAL_LABEL_STATUSES
            or semantic not in FORMAL_LABEL_STATUSES):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic combined status 非法")
    if "FAIL" in {internal, semantic}:
        return "FAIL"
    if "NE" in {internal, semantic}:
        return "NE"
    return "PASS"


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class GenerationGeneralizationSemanticPredictionRecord:
    """一条 actual run 的 label-blind 语义投影与 result 内容锁。"""

    observation_stable_key_sha256: str
    semantic_projection_sha256: str | None
    run_sha256: str
    requirement_statuses: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        sha256_text(
            self.observation_stable_key_sha256,
            where="GG-03 semantic prediction Observation SHA",
        )
        if self.semantic_projection_sha256 is not None:
            sha256_text(
                self.semantic_projection_sha256,
                where="GG-03 semantic prediction projection SHA",
            )
        sha256_text(
            self.run_sha256, where="GG-03 semantic prediction run SHA")
        requirements = tuple(item[0] for item in self.requirement_statuses)
        if (not requirements or requirements != tuple(
                item for item in INDEPENDENT_VERIFIER_REQUIREMENTS
                if item in requirements)
                or any(status not in FORMAL_LABEL_STATUSES
                       for _requirement, status in self.requirement_statuses)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic prediction requirement status 顺序非法")

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_stable_key_sha256": (
                self.observation_stable_key_sha256),
            "projection_status": (
                "AVAILABLE"
                if self.semantic_projection_sha256 is not None
                else "UNAVAILABLE"),
            "requirement_statuses": [
                {"requirement": requirement, "status": status}
                for requirement, status in self.requirement_statuses
            ],
            "run_sha256": self.run_sha256,
            "semantic_projection_sha256": self.semantic_projection_sha256,
        }


def _prediction_record(
        run: GenerationGeneralizationEvaluationActualRun,
        ) -> GenerationGeneralizationSemanticPredictionRecord:
    """从 actual run 提取不含 surface 明文的语义 prediction record。"""
    if not isinstance(run, GenerationGeneralizationEvaluationActualRun):
        raise TypeError("GG-03 semantic prediction actual run 类型错误")
    projection = build_actual_generation_generalization_semantic_projection(
        run)
    return GenerationGeneralizationSemanticPredictionRecord(
        generation_generalization_observation_key_sha256(run.observation),
        None if projection is None else projection.sha256(),
        generation_generalization_sha256_bytes(canonical_json_bytes(
            list(run.stable_key()))),
        tuple((item.requirement, item.status) for item in run.requirements),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationSemanticPredictionSeal:
    """首次 label read 前全部 candidate 语义投影的不可变安全投影。"""

    family_manifest_sha256: str
    family_commitment_sha256: str
    candidate_payload_sha256: str
    policy_sha256: str
    batch_sha256: str
    verdict_contract_sha256: str
    records: tuple[GenerationGeneralizationSemanticPredictionRecord, ...]
    teacher_call_count: int = 0
    label_read_count: int = 0
    host_learning_write_count: int = 0
    status: str = SEMANTIC_PREDICTION_SEAL_STATUS

    def __post_init__(self) -> None:
        for name in (
                "family_manifest_sha256", "family_commitment_sha256",
                "candidate_payload_sha256", "policy_sha256", "batch_sha256",
                "verdict_contract_sha256"):
            sha256_text(
                getattr(self, name),
                where=f"GG-03 semantic prediction seal {name}",
            )
        if self.verdict_contract_sha256 != (
                generation_generalization_semantic_verdict_contract_sha256()):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic prediction seal verdict contract 漂移")
        if (not self.records
                or len({item.observation_stable_key_sha256
                        for item in self.records}) != len(self.records)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic prediction seal record 顺序或唯一性漂移")
        if any(getattr(self, name) != 0 for name in (
                "teacher_call_count", "label_read_count",
                "host_learning_write_count")):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic prediction seal 零调用/零写审计失败")
        if self.status != SEMANTIC_PREDICTION_SEAL_STATUS:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic prediction seal status 漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": SEMANTIC_PREDICTION_SEAL_ARTIFACT_KIND,
            "batch_sha256": self.batch_sha256,
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "family_commitment_sha256": self.family_commitment_sha256,
            "family_manifest_sha256": self.family_manifest_sha256,
            "format_version": 2,
            "host_learning_write_count": self.host_learning_write_count,
            "label_read_count": self.label_read_count,
            "policy_sha256": self.policy_sha256,
            "records": [item.to_dict() for item in self.records],
            "status": self.status,
            "teacher_call_count": self.teacher_call_count,
            "verdict_contract_sha256": self.verdict_contract_sha256,
        }

    def sha256(self) -> str:
        return generation_generalization_sha256_bytes(
            canonical_json_bytes(self.to_dict()))


def build_generation_generalization_semantic_prediction_seal(
        batch: GenerationGeneralizationEvaluationBatch,
        *,
        family_manifest_sha256: str,
        family_commitment_sha256: str,
        candidate_payload_sha256: str,
        policy_sha256: str,
        observation_inventory: GenerationGeneralizationObservationInventoryIdentity,
        ) -> GenerationGeneralizationSemanticPredictionSeal:
    """封存同一 actual batch 的语义 projection 后才允许读取标签。"""
    if (not isinstance(batch, GenerationGeneralizationEvaluationBatch)
            or not isinstance(
                observation_inventory,
                GenerationGeneralizationObservationInventoryIdentity)):
        raise TypeError("GG-03 semantic prediction seal 输入类型错误")
    records = tuple(_prediction_record(item) for item in batch.runs)
    expected = tuple(
        item.stable_key_sha256 for item in observation_inventory.records)
    actual = tuple(
        item.observation_stable_key_sha256 for item in records)
    if actual != expected:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic prediction seal 未精确覆盖 Observation inventory")
    return GenerationGeneralizationSemanticPredictionSeal(
        family_manifest_sha256,
        family_commitment_sha256,
        candidate_payload_sha256,
        policy_sha256,
        generation_generalization_sha256_bytes(canonical_json_bytes(
            list(batch.stable_key()))),
        generation_generalization_semantic_verdict_contract_sha256(),
        records,
        sum(item.teacher_call_count for item in batch.runs),
        sum(item.label_read_count for item in batch.runs),
        sum(item.host_learning_write_count for item in batch.runs),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationSemanticFormalAggregate:
    """不暴露 surface/label 的 V2 六路 hard-conjunct aggregate。"""

    family_manifest_sha256: str
    family_commitment_sha256: str
    prediction_seal_sha256: str
    label_commitment_sha256: str
    verdict_contract_sha256: str
    dimensions: tuple[GenerationGeneralizationFormalDimensionResult, ...]
    label_read_count: int
    label_record_count: int
    label_transport_bytes: int
    status: str
    failure_phase: str = "NONE"

    def __post_init__(self) -> None:
        for name in (
                "family_manifest_sha256", "family_commitment_sha256",
                "prediction_seal_sha256", "label_commitment_sha256",
                "verdict_contract_sha256"):
            sha256_text(
                getattr(self, name), where=f"GG-03 semantic aggregate {name}")
        if self.verdict_contract_sha256 != (
                generation_generalization_semantic_verdict_contract_sha256()):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic aggregate verdict contract 漂移")
        if tuple(item.requirement for item in self.dimensions) != (
                INDEPENDENT_VERIFIER_REQUIREMENTS):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic aggregate dimension order 漂移")
        expected = "FAIL" if any(
            item.status == "FAIL" for item in self.dimensions) else (
            "NE" if any(item.status == "NE" for item in self.dimensions)
            else "PASS")
        if self.status != expected:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic aggregate status 漂移")
        if any(type(value) is not int or value < 0 for value in (
                self.label_read_count, self.label_record_count,
                self.label_transport_bytes)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic aggregate label audit 非法")
        if (self.failure_phase == "NONE"
                and (self.label_read_count != 1
                     or self.label_record_count <= 0)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic completed aggregate label audit 未闭合")
        if not isinstance(self.failure_phase, str) or not self.failure_phase:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic aggregate failure phase 非法")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": SEMANTIC_FORMAL_AGGREGATE_ARTIFACT_KIND,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "failure_phase": self.failure_phase,
            "family_commitment_sha256": self.family_commitment_sha256,
            "family_manifest_sha256": self.family_manifest_sha256,
            "format_version": 2,
            "label_commitment_sha256": self.label_commitment_sha256,
            "label_read_count": self.label_read_count,
            "label_record_count": self.label_record_count,
            "label_transport_bytes": self.label_transport_bytes,
            "prediction_seal_sha256": self.prediction_seal_sha256,
            "status": self.status,
            "verdict_contract_sha256": self.verdict_contract_sha256,
            "write_counts": {
                "candidate_writes": 0,
                "evaluator_label_writes": 0,
                "host_learning_writes": 0,
                "memory_learning_writes": 0,
                "teacher_calls": 0,
            },
        }

    def sha256(self) -> str:
        return generation_generalization_sha256_bytes(
            canonical_json_bytes(self.to_dict()))


def build_generation_generalization_semantic_formal_aggregate(
        prediction: GenerationGeneralizationSemanticPredictionSeal,
        labels: tuple[GenerationGeneralizationSemanticLabelRecord, ...],
        *,
        label_commitment_sha256: str,
        label_transport_bytes: int,
        ) -> GenerationGeneralizationSemanticFormalAggregate:
    """逐 Observation 合取 actual verifier 与 private semantic verdict。"""
    if (not isinstance(
            prediction, GenerationGeneralizationSemanticPredictionSeal)
            or not labels or len(labels) != len(prediction.records)):
        raise TypeError("GG-03 semantic aggregate 输入类型或数量错误")
    prediction_keys = tuple(
        item.observation_stable_key_sha256 for item in prediction.records)
    label_keys = tuple(
        item.observation_stable_key_sha256 for item in labels)
    if prediction_keys != label_keys:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic prediction/label Observation identity 漂移")
    scored: list[dict[str, Any]] = []
    for predicted, label in zip(prediction.records, labels, strict=True):
        if tuple(
                item[0] for item in predicted.requirement_statuses
                ) != label.requirements:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic prediction/label requirement 漂移")
        semantic_status = label.verdict_for_projection_sha256(
            predicted.semantic_projection_sha256)
        scored.append({
            "observation_stable_key_sha256": (
                predicted.observation_stable_key_sha256),
            "requirement_statuses": [
                {
                    "requirement": requirement,
                    "status": _combined_status(internal, semantic_status),
                }
                for requirement, internal in predicted.requirement_statuses
            ],
            "semantic_projection_status": semantic_status,
        })
    dimensions = []
    for requirement in INDEPENDENT_VERIFIER_REQUIREMENTS:
        statuses = tuple(
            item["status"]
            for row in scored
            for item in row["requirement_statuses"]
            if item["requirement"] == requirement)
        counts = tuple(
            statuses.count(status) for status in FORMAL_LABEL_STATUSES)
        dimensions.append(GenerationGeneralizationFormalDimensionResult(
            requirement,
            _status_from_counts(*counts),
            len(statuses),
            counts[0], counts[1], counts[2],
            generation_generalization_sha256_bytes(canonical_json_bytes({
                "requirement": requirement,
                "rows": [
                    row for row in scored
                    if any(item["requirement"] == requirement
                           for item in row["requirement_statuses"])
                ],
                "verdict_contract_sha256": (
                    prediction.verdict_contract_sha256),
            })),
        ))
    statuses = tuple(item.status for item in dimensions)
    overall = (
        "FAIL" if "FAIL" in statuses
        else "NE" if "NE" in statuses
        else "PASS")
    return GenerationGeneralizationSemanticFormalAggregate(
        prediction.family_manifest_sha256,
        prediction.family_commitment_sha256,
        prediction.sha256(),
        label_commitment_sha256,
        prediction.verdict_contract_sha256,
        tuple(dimensions),
        1,
        len(labels),
        label_transport_bytes,
        overall,
    )


def build_generation_generalization_semantic_unavailable_aggregate(
        observation_inventory: GenerationGeneralizationObservationInventoryIdentity,
        *,
        family_manifest_sha256: str,
        family_commitment_sha256: str,
        label_commitment_sha256: str,
        prediction_seal_sha256: str,
        failure_phase: str,
        label_read_count: int,
        label_record_count: int,
        label_transport_bytes: int,
        ) -> GenerationGeneralizationSemanticFormalAggregate:
    """guard 后异常统一封存为六路 NE，并保留真实 label 审计计数。"""
    rows = tuple(
        GenerationGeneralizationFormalDimensionResult(
            requirement,
            "NE",
            planned,
            0,
            0,
            planned,
            generation_generalization_sha256_bytes(canonical_json_bytes({
                "failure_phase": failure_phase,
                "planned_count": planned,
                "requirement": requirement,
                "verdict_contract_sha256": (
                    generation_generalization_semantic_verdict_contract_sha256()),
            })),
        )
        for requirement, planned in observation_inventory.requirement_counts
    )
    return GenerationGeneralizationSemanticFormalAggregate(
        family_manifest_sha256,
        family_commitment_sha256,
        prediction_seal_sha256,
        label_commitment_sha256,
        generation_generalization_semantic_verdict_contract_sha256(),
        rows,
        label_read_count,
        label_record_count,
        label_transport_bytes,
        "NE",
        failure_phase,
    )


def build_generation_generalization_semantic_publication(
        aggregate: GenerationGeneralizationSemanticFormalAggregate,
        *,
        failure_diagnostic_sha256: str | None = None,
        ) -> tuple[dict[str, object], dict[str, object] | None,
                   dict[str, object] | None]:
    """V2 PASS 只发布 receipt，FAIL/NE 只发布最小 failure seal。"""
    if not isinstance(
            aggregate, GenerationGeneralizationSemanticFormalAggregate):
        raise TypeError("GG-03 semantic publication aggregate 类型错误")
    operational_failure = aggregate.failure_phase != "NONE"
    if operational_failure:
        if failure_diagnostic_sha256 is None:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic operational failure 缺安全 diagnostic")
        sha256_text(
            failure_diagnostic_sha256,
            where="GG-03 semantic failure diagnostic SHA",
        )
    elif failure_diagnostic_sha256 is not None:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic capability publication 不得携 diagnostic")
    decision = {
        "aggregate_sha256": aggregate.sha256(),
        "artifact_kind": "PH2_GG03_FORMAL_SEMANTIC_PUBLICATION_DECISION_V2",
        "format_version": 2,
        "publication_artifact": (
            "runtime_receipt.json" if aggregate.status == "PASS"
            else "failure_seal.json"),
        "status": aggregate.status,
    }
    if failure_diagnostic_sha256 is not None:
        decision["failure_diagnostic_sha256"] = failure_diagnostic_sha256
    receipt = None
    failure = None
    if aggregate.status == "PASS":
        receipt = {
            "aggregate_sha256": aggregate.sha256(),
            "artifact_kind": (
                SEMANTIC_FORMAL_RUNTIME_RECEIPT_ARTIFACT_KIND),
            "family_commitment_sha256": aggregate.family_commitment_sha256,
            "family_manifest_sha256": aggregate.family_manifest_sha256,
            "format_version": 2,
            "status": "PASS",
            "verdict_contract_sha256": aggregate.verdict_contract_sha256,
        }
    else:
        failure = {
            "aggregate_sha256": aggregate.sha256(),
            "artifact_kind": SEMANTIC_FORMAL_FAILURE_SEAL_ARTIFACT_KIND,
            "failure_phase": aggregate.failure_phase,
            "family_commitment_sha256": aggregate.family_commitment_sha256,
            "family_manifest_sha256": aggregate.family_manifest_sha256,
            "format_version": 2,
            "status": aggregate.status,
            "verdict_contract_sha256": aggregate.verdict_contract_sha256,
        }
        if failure_diagnostic_sha256 is not None:
            failure["failure_diagnostic_sha256"] = (
                failure_diagnostic_sha256)
    return decision, receipt, failure


__all__ = [
    "SEMANTIC_FORMAL_AGGREGATE_ARTIFACT_KIND",
    "SEMANTIC_FORMAL_FAILURE_SEAL_ARTIFACT_KIND",
    "SEMANTIC_FORMAL_FAILURE_DIAGNOSTIC_ARTIFACT_KIND",
    "SEMANTIC_FORMAL_RUNTIME_RECEIPT_ARTIFACT_KIND",
    "SEMANTIC_PREDICTION_SEAL_ARTIFACT_KIND",
    "SEMANTIC_PREDICTION_SEAL_STATUS",
    "GenerationGeneralizationSemanticFormalAggregate",
    "GenerationGeneralizationSemanticPredictionRecord",
    "GenerationGeneralizationSemanticPredictionSeal",
    "build_generation_generalization_semantic_formal_aggregate",
    "build_generation_generalization_semantic_prediction_seal",
    "build_generation_generalization_semantic_publication",
    "build_generation_generalization_semantic_unavailable_aggregate",
]
