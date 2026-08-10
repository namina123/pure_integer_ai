"""Stage plugin declaration and structural invocation contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Protocol

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    positive,
    sha256_text,
    string_tuple,
    text,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
    EvaluationRunAudit,
    EvaluationResultSet,
)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPluginDeclaration:
    """Immutable plugin identity, independent of a Python file path."""

    plugin_key: str
    plugin_version: str
    stage_key: str
    module_key: str
    symbol_key: str
    semantic_sha256: str
    result_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
                "plugin_key", "plugin_version", "stage_key",
                "module_key", "symbol_key"):
            text(getattr(self, name), where=f"evaluation plugin {name}")
        if any(character in self.module_key for character in ("/", "\\", ":")):
            raise EvaluationKernelContractError(
                "evaluation plugin module_key must not be a physical path")
        sha256_text(self.semantic_sha256, where="evaluation plugin semantic identity")
        normalized = string_tuple(self.result_keys, where="evaluation plugin result keys")
        if normalized != self.result_keys:
            raise EvaluationKernelContractError("evaluation plugin result keys drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_key": self.module_key,
            "plugin_key": self.plugin_key,
            "plugin_version": self.plugin_version,
            "result_keys": list(self.result_keys),
            "semantic_sha256": self.semantic_sha256,
            "stage_key": self.stage_key,
            "symbol_key": self.symbol_key,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationPluginDeclaration":
        raw = exact_dict(value, {
            "module_key", "plugin_key", "plugin_version", "result_keys",
            "semantic_sha256", "stage_key", "symbol_key",
        }, where="EvaluationPluginDeclaration")
        return cls(
            str(raw["plugin_key"]), str(raw["plugin_version"]),
            str(raw["stage_key"]), str(raw["module_key"]),
            str(raw["symbol_key"]), str(raw["semantic_sha256"]),
            tuple(str(item) for item in raw["result_keys"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPluginRunContext:
    """Payload-free run identity supplied by the generic runtime."""

    manifest_sha256: str
    family_commitment: str
    source_binding_sha256: str
    owner_binding_sha256: str
    run_id: int

    def __post_init__(self) -> None:
        for name in (
                "manifest_sha256", "family_commitment",
                "source_binding_sha256", "owner_binding_sha256"):
            sha256_text(getattr(self, name), where=f"plugin context {name}")
        if positive(self.run_id, where="plugin context run_id") != 1:
            raise EvaluationKernelContractError("formal evaluation run_id must be one")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPluginOutcome:
    """The only generic runtime output accepted from a stage plugin."""

    result_set: EvaluationResultSet
    run_audit: EvaluationRunAudit

    def __post_init__(self) -> None:
        if (not isinstance(self.result_set, EvaluationResultSet)
                or not isinstance(self.run_audit, EvaluationRunAudit)):
            raise EvaluationKernelContractError("evaluation plugin outcome drifted")


# object-model: interface; representation=protocol; interop=pending
class StageEvaluationPlugin(Protocol):
    """A stage plugin receives only authorized records and returns safe results."""

    @property
    def declaration(self) -> EvaluationPluginDeclaration:
        """Return the immutable declaration bound by the family manifest."""

    def evaluate(
            self,
            context: EvaluationPluginRunContext,
            records: Iterable[object],
            ) -> EvaluationPluginOutcome:
        """Consume the supplied bounded stream exactly once."""


__all__ = [
    "EvaluationPluginDeclaration",
    "EvaluationPluginOutcome",
    "EvaluationPluginRunContext",
    "StageEvaluationPlugin",
]
