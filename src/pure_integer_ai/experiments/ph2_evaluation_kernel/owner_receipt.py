"""Payload-free owner receipt projection for a new evaluation family."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    nonnegative,
    positive,
    sha256_text,
    text,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationOwnerBinding:
    """Safe metadata binding; private payload, labels and paths remain opaque."""

    owner_key: str
    owner_receipt_sha256: str
    metadata_sha256: str
    file_inventory_sha256: str
    payload_commitment: str
    case_commitment: str
    label_commitment: str
    cluster_commitment: str
    source_ref_count: int
    observation_count: int
    label_count: int
    pair_count: int
    formal_run_count_before: int = 0
    private_payload_reads_before: int = 0

    def __post_init__(self) -> None:
        text(self.owner_key, where="evaluation owner_key")
        for name in (
                "owner_receipt_sha256", "metadata_sha256", "file_inventory_sha256",
                "payload_commitment",
                "case_commitment", "label_commitment", "cluster_commitment"):
            sha256_text(getattr(self, name), where=f"evaluation owner {name}")
        for name in ("source_ref_count", "observation_count", "label_count", "pair_count"):
            positive(getattr(self, name), where=f"evaluation owner {name}")
        if (self.observation_count != self.label_count
                or self.pair_count != self.observation_count):
            raise EvaluationKernelContractError(
                "evaluation owner observation/label/pair counts must close")
        for name in ("formal_run_count_before", "private_payload_reads_before"):
            nonnegative(getattr(self, name), where=f"evaluation owner {name}")
            if getattr(self, name) != 0:
                raise EvaluationKernelContractError(
                    "evaluation owner must bind an unused zero-read family")
        validate_v2_safe_report(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_commitment": self.case_commitment,
            "cluster_commitment": self.cluster_commitment,
            "formal_run_count_before": self.formal_run_count_before,
            "file_inventory_sha256": self.file_inventory_sha256,
            "label_commitment": self.label_commitment,
            "label_count": self.label_count,
            "metadata_sha256": self.metadata_sha256,
            "observation_count": self.observation_count,
            "owner_key": self.owner_key,
            "owner_receipt_sha256": self.owner_receipt_sha256,
            "pair_count": self.pair_count,
            "payload_commitment": self.payload_commitment,
            "private_payload_reads_before": self.private_payload_reads_before,
            "source_ref_count": self.source_ref_count,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationOwnerBinding":
        raw = exact_dict(value, {
            "case_commitment", "cluster_commitment", "file_inventory_sha256",
            "formal_run_count_before",
            "label_commitment", "label_count", "metadata_sha256",
            "observation_count", "owner_key", "owner_receipt_sha256",
            "pair_count", "payload_commitment", "private_payload_reads_before",
            "source_ref_count",
        }, where="EvaluationOwnerBinding")
        return cls(
            str(raw["owner_key"]), str(raw["owner_receipt_sha256"]),
            str(raw["metadata_sha256"]), str(raw["file_inventory_sha256"]),
            str(raw["payload_commitment"]),
            str(raw["case_commitment"]), str(raw["label_commitment"]),
            str(raw["cluster_commitment"]), raw["source_ref_count"],
            raw["observation_count"], raw["label_count"], raw["pair_count"],
            raw["formal_run_count_before"], raw["private_payload_reads_before"],
        )


__all__ = ["EvaluationOwnerBinding"]
