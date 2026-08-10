"""Exact source, cluster and ordinal bindings for one evaluation family."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    positive,
    sha256_text,
    text,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)


EVALUATION_SOURCE_SPLITS = (
    "train", "dev", "shadow_a", "shadow_b", "shadow_c",
    "quarantine", "held_out", "adversarial", "wall",
    "formal_reserve_a", "formal_reserve_b",
)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class EvaluationSourceSlice:
    """A contiguous cluster/ordinal slice, never an unbounded repository claim."""

    source_key: str
    split: str
    cluster_key: str
    ordinal_start: int
    ordinal_end: int
    record_count: int
    source_ref_commitment: str

    def __post_init__(self) -> None:
        text(self.source_key, where="source slice source_key")
        if self.split not in EVALUATION_SOURCE_SPLITS:
            raise EvaluationKernelContractError("source slice split is not registered")
        text(self.cluster_key, where="source slice cluster_key")
        positive(self.ordinal_start, where="source slice ordinal_start")
        positive(self.ordinal_end, where="source slice ordinal_end")
        positive(self.record_count, where="source slice record_count")
        if self.ordinal_end < self.ordinal_start:
            raise EvaluationKernelContractError("source slice ordinal range is reversed")
        if self.record_count != self.ordinal_end - self.ordinal_start + 1:
            raise EvaluationKernelContractError(
                "source slice count must exactly match its ordinal range")
        sha256_text(
            self.source_ref_commitment, where="source slice SourceRef commitment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_key": self.cluster_key,
            "ordinal_end": self.ordinal_end,
            "ordinal_start": self.ordinal_start,
            "record_count": self.record_count,
            "source_key": self.source_key,
            "source_ref_commitment": self.source_ref_commitment,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationSourceSlice":
        raw = exact_dict(value, {
            "cluster_key", "ordinal_end", "ordinal_start", "record_count",
            "source_key", "source_ref_commitment", "split",
        }, where="EvaluationSourceSlice")
        return cls(
            str(raw["source_key"]), str(raw["split"]),
            str(raw["cluster_key"]), raw["ordinal_start"], raw["ordinal_end"],
            raw["record_count"], str(raw["source_ref_commitment"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationSourceBinding:
    """Immutable source contract plus non-overlapping exact slices."""

    source_contract_sha256: str
    slices: tuple[EvaluationSourceSlice, ...]
    source_ref_commitment: str

    def __post_init__(self) -> None:
        sha256_text(self.source_contract_sha256, where="source contract")
        sha256_text(self.source_ref_commitment, where="source binding SourceRef")
        if (not isinstance(self.slices, tuple) or not self.slices
                or any(not isinstance(item, EvaluationSourceSlice)
                       for item in self.slices)):
            raise EvaluationKernelContractError("source binding slices are invalid")
        if self.slices != tuple(sorted(self.slices)):
            raise EvaluationKernelContractError("source binding slices must be canonical")
        previous: dict[tuple[str, str, str], int] = {}
        for item in self.slices:
            key = (item.source_key, item.split, item.cluster_key)
            if item.ordinal_start <= previous.get(key, 0):
                raise EvaluationKernelContractError(
                    "source binding contains overlapping ordinal slices")
            previous[key] = item.ordinal_end

    @property
    def record_count(self) -> int:
        return sum(item.record_count for item in self.slices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "slices": [item.to_dict() for item in self.slices],
            "source_contract_sha256": self.source_contract_sha256,
            "source_ref_commitment": self.source_ref_commitment,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationSourceBinding":
        raw = exact_dict(value, {
            "record_count", "slices", "source_contract_sha256",
            "source_ref_commitment",
        }, where="EvaluationSourceBinding")
        if not isinstance(raw["slices"], list):
            raise EvaluationKernelContractError("source binding slices must be an array")
        result = cls(
            str(raw["source_contract_sha256"]),
            tuple(EvaluationSourceSlice.from_dict(item) for item in raw["slices"]),
            str(raw["source_ref_commitment"]),
        )
        if raw["record_count"] != result.record_count:
            raise EvaluationKernelContractError("source binding record_count drifted")
        return result


__all__ = [
    "EVALUATION_SOURCE_SPLITS",
    "EvaluationSourceBinding",
    "EvaluationSourceSlice",
]
