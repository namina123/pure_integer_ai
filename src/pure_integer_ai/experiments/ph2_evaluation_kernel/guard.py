"""Pure one-shot guard state transition for generic evaluation runtimes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Protocol

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    sha256_text,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.manifest import (
    EvaluationKernelManifest,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)


EVALUATION_GUARD_STATES = ("AVAILABLE", "CONSUMED")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationOneShotGuard:
    """All identities required before the only formal run can start."""

    manifest_sha256: str
    family_commitment: str
    owner_receipt_sha256: str
    candidate_artifact_sha256: str
    plugin_semantic_sha256: str
    run_id: int
    state: str
    formal_run_count_before: int = 0
    private_payload_reads_before: int = 0

    def __post_init__(self) -> None:
        for name in (
                "manifest_sha256", "family_commitment", "owner_receipt_sha256",
                "candidate_artifact_sha256", "plugin_semantic_sha256"):
            sha256_text(getattr(self, name), where=f"evaluation guard {name}")
        if type(self.run_id) is not int or self.run_id != 1:
            raise EvaluationKernelContractError("evaluation guard run_id must be one")
        if self.state not in EVALUATION_GUARD_STATES:
            raise EvaluationKernelContractError("evaluation guard state is invalid")
        if (type(self.formal_run_count_before) is not int
                or type(self.private_payload_reads_before) is not int
                or self.formal_run_count_before != 0
                or self.private_payload_reads_before != 0):
            raise EvaluationKernelContractError(
                "evaluation guard requires an unused zero-read family")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "family_commitment": self.family_commitment,
            "formal_run_count_before": self.formal_run_count_before,
            "manifest_sha256": self.manifest_sha256,
            "owner_receipt_sha256": self.owner_receipt_sha256,
            "plugin_semantic_sha256": self.plugin_semantic_sha256,
            "private_payload_reads_before": self.private_payload_reads_before,
            "run_id": self.run_id,
            "state": self.state,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationOneShotGuard":
        raw = exact_dict(value, {
            "candidate_artifact_sha256", "family_commitment",
            "formal_run_count_before", "manifest_sha256",
            "owner_receipt_sha256", "plugin_semantic_sha256",
            "private_payload_reads_before", "run_id", "state",
        }, where="EvaluationOneShotGuard")
        return cls(
            str(raw["manifest_sha256"]), str(raw["family_commitment"]),
            str(raw["owner_receipt_sha256"]),
            str(raw["candidate_artifact_sha256"]),
            str(raw["plugin_semantic_sha256"]), raw["run_id"],
            str(raw["state"]), raw["formal_run_count_before"],
            raw["private_payload_reads_before"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationRunIntent:
    """Immutable proof that the available guard was consumed before payload read."""

    manifest_sha256: str
    family_commitment: str
    consumed_guard_sha256: str
    run_id: int
    state: str = "FORMAL_RUN_INTENT_FROZEN"

    def __post_init__(self) -> None:
        for name in (
                "manifest_sha256", "family_commitment", "consumed_guard_sha256"):
            sha256_text(getattr(self, name), where=f"evaluation intent {name}")
        if type(self.run_id) is not int or self.run_id != 1:
            raise EvaluationKernelContractError("evaluation intent run_id must be one")
        if self.state != "FORMAL_RUN_INTENT_FROZEN":
            raise EvaluationKernelContractError("evaluation intent state drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumed_guard_sha256": self.consumed_guard_sha256,
            "family_commitment": self.family_commitment,
            "manifest_sha256": self.manifest_sha256,
            "run_id": self.run_id,
            "state": self.state,
        }


def build_available_guard(manifest: EvaluationKernelManifest) -> EvaluationOneShotGuard:
    """Create the only AVAILABLE guard from a validated immutable manifest."""
    if not isinstance(manifest, EvaluationKernelManifest):
        raise EvaluationKernelContractError("evaluation guard manifest type invalid")
    return build_available_guard_for_identity(
        manifest.sha256(), manifest.family_commitment,
        manifest.owner_binding.owner_receipt_sha256,
        manifest.candidate_artifact_sha256,
        manifest.plugin.semantic_sha256,
    )


def build_available_guard_for_identity(
        manifest_sha256: str,
        family_commitment: str,
        owner_receipt_sha256: str,
        candidate_artifact_sha256: str,
        plugin_semantic_sha256: str,
        ) -> EvaluationOneShotGuard:
    """按任意不可变 family identity 建立共享的一次性 available guard。"""
    return EvaluationOneShotGuard(
        manifest_sha256, family_commitment, owner_receipt_sha256,
        candidate_artifact_sha256, plugin_semantic_sha256,
        1, "AVAILABLE",
    )


def consume_guard(
        guard: EvaluationOneShotGuard,
        ) -> tuple[EvaluationOneShotGuard, EvaluationRunIntent]:
    """Return a consumed guard and intent; a consumed guard cannot transition again."""
    if not isinstance(guard, EvaluationOneShotGuard) or guard.state != "AVAILABLE":
        raise EvaluationKernelContractError("evaluation guard is already consumed")
    consumed = EvaluationOneShotGuard(
        guard.manifest_sha256, guard.family_commitment,
        guard.owner_receipt_sha256, guard.candidate_artifact_sha256,
        guard.plugin_semantic_sha256, guard.run_id, "CONSUMED",
        guard.formal_run_count_before, guard.private_payload_reads_before,
    )
    intent = EvaluationRunIntent(
        consumed.manifest_sha256, consumed.family_commitment,
        consumed.sha256(), consumed.run_id,
    )
    return consumed, intent


# object-model: interface; representation=protocol; interop=pending
class EvaluationGuardBackend(Protocol):
    """Append-only persistence boundary implemented by the generic runtime."""

    def publish_available(self, guard: EvaluationOneShotGuard) -> None:
        """Persist the only available guard, rejecting different existing content."""

    def consume(self) -> EvaluationRunIntent:
        """Atomically replace AVAILABLE with CONSUMED and persist run intent."""


__all__ = [
    "EVALUATION_GUARD_STATES",
    "EvaluationGuardBackend",
    "EvaluationOneShotGuard",
    "EvaluationRunIntent",
    "build_available_guard",
    "build_available_guard_for_identity",
    "consume_guard",
]
