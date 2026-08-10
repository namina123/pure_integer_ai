"""Status-separated runtime receipt and failure seal projections."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Protocol

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    sha256_text,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.aggregate import (
    EvaluationAggregate,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EVALUATION_RESULT_STATUSES,
    EvaluationKernelContractError,
)


_PUBLICATION_ARTIFACTS = {
    "PASS": "RUNTIME_RECEIPT",
    "FAIL": "CAPABILITY_FAILURE_SEAL",
    "NE": "NOT_EVALUABLE_SEAL",
    "BLOCKED": "INFRASTRUCTURE_BLOCK_SEAL",
}


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPublicationDecision:
    """A PASS may publish a receipt; every other state seals and stops."""

    aggregate_sha256: str
    status: str
    publication_artifact: str
    runtime_receipt_allowed: int
    failure_seal_required: int

    def __post_init__(self) -> None:
        sha256_text(self.aggregate_sha256, where="publication aggregate")
        if self.status not in EVALUATION_RESULT_STATUSES:
            raise EvaluationKernelContractError("publication status is invalid")
        if self.publication_artifact != _PUBLICATION_ARTIFACTS[self.status]:
            raise EvaluationKernelContractError("publication artifact/status drifted")
        expected = (1, 0) if self.status == "PASS" else (0, 1)
        if (self.runtime_receipt_allowed, self.failure_seal_required) != expected:
            raise EvaluationKernelContractError("publication gate was weakened")

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "failure_seal_required": self.failure_seal_required,
            "publication_artifact": self.publication_artifact,
            "runtime_receipt_allowed": self.runtime_receipt_allowed,
            "status": self.status,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationRuntimeReceipt:
    """Minimal PASS-only public receipt projection."""

    manifest_sha256: str
    family_commitment: str
    aggregate_sha256: str
    stage_key: str
    run_id: int
    status: str = "PASS"

    def __post_init__(self) -> None:
        for name in ("manifest_sha256", "family_commitment", "aggregate_sha256"):
            sha256_text(getattr(self, name), where=f"runtime receipt {name}")
        if type(self.run_id) is not int or self.run_id != 1 or self.status != "PASS":
            raise EvaluationKernelContractError("runtime receipt must seal one PASS run")
        validate_v2_safe_report(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "family_commitment": self.family_commitment,
            "manifest_sha256": self.manifest_sha256,
            "run_id": self.run_id,
            "stage_key": self.stage_key,
            "status": self.status,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationFailureSeal:
    """Minimal non-PASS seal; it contains no case, label, path or message."""

    manifest_sha256: str
    family_commitment: str
    aggregate_sha256: str
    stage_key: str
    run_id: int
    status: str
    seal_kind: str

    def __post_init__(self) -> None:
        for name in ("manifest_sha256", "family_commitment", "aggregate_sha256"):
            sha256_text(getattr(self, name), where=f"failure seal {name}")
        if type(self.run_id) is not int or self.run_id != 1:
            raise EvaluationKernelContractError("failure seal run_id must be one")
        if self.status not in ("FAIL", "NE", "BLOCKED"):
            raise EvaluationKernelContractError("failure seal cannot represent PASS")
        if self.seal_kind != _PUBLICATION_ARTIFACTS[self.status]:
            raise EvaluationKernelContractError("failure seal kind drifted")
        validate_v2_safe_report(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "family_commitment": self.family_commitment,
            "manifest_sha256": self.manifest_sha256,
            "run_id": self.run_id,
            "seal_kind": self.seal_kind,
            "stage_key": self.stage_key,
            "status": self.status,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def build_publication_decision(
        aggregate: EvaluationAggregate,
        ) -> EvaluationPublicationDecision:
    if not isinstance(aggregate, EvaluationAggregate):
        raise EvaluationKernelContractError("publication aggregate type invalid")
    return EvaluationPublicationDecision(
        aggregate.sha256(), aggregate.status,
        _PUBLICATION_ARTIFACTS[aggregate.status],
        int(aggregate.status == "PASS"), int(aggregate.status != "PASS"),
    )


def build_runtime_receipt(aggregate: EvaluationAggregate) -> EvaluationRuntimeReceipt:
    """Build a receipt only for a PASS aggregate."""
    decision = build_publication_decision(aggregate)
    if decision.runtime_receipt_allowed != 1:
        raise EvaluationKernelContractError("non-PASS aggregate cannot publish receipt")
    return EvaluationRuntimeReceipt(
        aggregate.manifest_sha256, aggregate.family_commitment,
        aggregate.sha256(), aggregate.stage_key, aggregate.run_id,
    )


def build_failure_seal(aggregate: EvaluationAggregate) -> EvaluationFailureSeal:
    """Build a non-PASS family seal and reject PASS input."""
    decision = build_publication_decision(aggregate)
    if decision.failure_seal_required != 1:
        raise EvaluationKernelContractError("PASS aggregate cannot publish failure seal")
    return EvaluationFailureSeal(
        aggregate.manifest_sha256, aggregate.family_commitment,
        aggregate.sha256(), aggregate.stage_key, aggregate.run_id,
        aggregate.status, decision.publication_artifact,
    )


# object-model: interface; representation=protocol; interop=pending
class EvaluationPublicationBackend(Protocol):
    """Append-only publication boundary implemented by the generic runtime."""

    def publish_receipt(self, receipt: EvaluationRuntimeReceipt) -> None:
        """Publish exactly one PASS receipt."""

    def publish_failure_seal(self, seal: EvaluationFailureSeal) -> None:
        """Publish exactly one non-PASS seal."""


__all__ = [
    "EvaluationFailureSeal",
    "EvaluationPublicationBackend",
    "EvaluationPublicationDecision",
    "EvaluationRuntimeReceipt",
    "build_failure_seal",
    "build_publication_decision",
    "build_runtime_receipt",
]
