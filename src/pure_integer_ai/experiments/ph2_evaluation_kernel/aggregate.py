"""Safe hard-conjunct aggregate with no averaged score."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    positive,
    sha256_text,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.manifest import (
    EvaluationKernelManifest,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
    EvaluationRunAudit,
    EvaluationResultSet,
)


EVALUATION_AGGREGATE_KIND = "PH2_EVALUATION_KERNEL_AGGREGATE"
EVALUATION_AGGREGATE_VERSION = "PH2-EVALUATION-KERNEL-AGGREGATE-V1"


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationAggregate:
    """Canonical result publication; every declared hard conjunct remains visible."""

    artifact_kind: str
    artifact_version: str
    manifest_sha256: str
    family_commitment: str
    stage_key: str
    run_id: int
    result_set: EvaluationResultSet
    run_audit: EvaluationRunAudit
    status: str

    def __post_init__(self) -> None:
        if (self.artifact_kind != EVALUATION_AGGREGATE_KIND
                or self.artifact_version != EVALUATION_AGGREGATE_VERSION):
            raise EvaluationKernelContractError("evaluation aggregate identity drifted")
        sha256_text(self.manifest_sha256, where="aggregate manifest")
        sha256_text(self.family_commitment, where="aggregate family")
        if positive(self.run_id, where="aggregate run_id") != 1:
            raise EvaluationKernelContractError("evaluation aggregate run_id must be one")
        if not isinstance(self.result_set, EvaluationResultSet):
            raise EvaluationKernelContractError("evaluation aggregate result set invalid")
        if not isinstance(self.run_audit, EvaluationRunAudit):
            raise EvaluationKernelContractError("evaluation aggregate run audit invalid")
        if self.status != self.result_set.status:
            raise EvaluationKernelContractError("evaluation aggregate status drifted")
        if self.status != "BLOCKED" and self.run_audit.audit_state != "COMPLETE":
            raise EvaluationKernelContractError(
                "non-BLOCKED aggregate requires a complete run audit")
        validate_v2_safe_report(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "family_commitment": self.family_commitment,
            "manifest_sha256": self.manifest_sha256,
            "result_set": self.result_set.to_dict(),
            "run_audit": self.run_audit.to_dict(),
            "run_id": self.run_id,
            "stage_key": self.stage_key,
            "status": self.status,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationAggregate":
        raw = exact_dict(value, {
            "artifact_kind", "artifact_version", "family_commitment",
            "manifest_sha256", "result_set", "run_audit", "run_id",
            "stage_key", "status",
        }, where="EvaluationAggregate")
        return cls(
            str(raw["artifact_kind"]), str(raw["artifact_version"]),
            str(raw["manifest_sha256"]), str(raw["family_commitment"]),
            str(raw["stage_key"]), raw["run_id"],
            EvaluationResultSet.from_dict(raw["result_set"]),
            EvaluationRunAudit.from_dict(raw["run_audit"]),
            str(raw["status"]),
        )


def build_evaluation_aggregate(
        manifest: EvaluationKernelManifest,
        result_set: EvaluationResultSet,
        run_audit: EvaluationRunAudit,
        *,
        run_id: int = 1,
        ) -> EvaluationAggregate:
    """Bind exact plugin output to its immutable family manifest."""
    if (not isinstance(manifest, EvaluationKernelManifest)
            or not isinstance(result_set, EvaluationResultSet)
            or not isinstance(run_audit, EvaluationRunAudit)):
        raise EvaluationKernelContractError("evaluation aggregate inputs invalid")
    keys = tuple(item.result_key for item in result_set.results)
    roles = tuple(item.role for item in result_set.results)
    expected_roles = (
        *("BEARING" for _ in manifest.bearing_dimension_keys),
        "GENERATION",
        *("SUPPORT" for _ in manifest.support_dimension_keys),
    )
    if keys != manifest.hard_conjunct_keys or roles != expected_roles:
        raise EvaluationKernelContractError(
            "plugin results do not exactly match manifest keys and roles")
    owner = manifest.owner_binding
    budget = manifest.resource_budget
    if run_audit.audit_state == "COMPLETE":
        if (run_audit.source_ref_count != owner.source_ref_count
                or run_audit.pair_count != owner.pair_count
                or run_audit.private_record_reads > budget.max_records
                or run_audit.private_payload_gets > budget.max_payload_gets
                or run_audit.transport_bytes_read > budget.max_payload_bytes
                or run_audit.logic_operations > budget.max_logic_operations):
            raise EvaluationKernelContractError(
                "evaluation aggregate resource/read audit drifted")
    elif result_set.status != "BLOCKED":
        raise EvaluationKernelContractError(
            "only BLOCKED aggregate may carry an unavailable audit")
    return EvaluationAggregate(
        EVALUATION_AGGREGATE_KIND, EVALUATION_AGGREGATE_VERSION,
        manifest.sha256(), manifest.family_commitment, manifest.stage_key,
        run_id, result_set, run_audit, result_set.status,
    )


__all__ = [
    "EVALUATION_AGGREGATE_KIND",
    "EVALUATION_AGGREGATE_VERSION",
    "EvaluationAggregate",
    "build_evaluation_aggregate",
]
