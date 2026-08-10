"""Shared execution facade for public-only experimental capability plugins."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_RESOURCE_HARD_LIMITS,
    V2_ZERO_CALL_WINDOW_COUNT,
    V2StageEvaluationPolicy,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2WriteAccount,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginOutcome,
    EvaluationPluginRunContext,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationDimensionResult,
    EvaluationKernelContractError,
    EvaluationResultSet,
    EvaluationRunAudit,
)
from pure_integer_ai.experiments.ph2_evaluation_public_source import (
    EvaluationPublicObservationEvidencePair,
    EvaluationPublicSourceRecord,
)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPublicProbe:
    """One public capability or support hard conjunct."""

    key: str
    status: str
    evidence_sha256: str
    operations: int

    def __post_init__(self) -> None:
        if (not isinstance(self.key, str) or not self.key
                or self.status not in {"PASS", "FAIL", "NE", "BLOCKED"}
                or not isinstance(self.evidence_sha256, str)
                or len(self.evidence_sha256) != 64
                or type(self.operations) is not int
                or self.operations < 0):
            raise EvaluationKernelContractError("public capability probe drifted")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPublicCapabilityRun:
    """Stage-specific capabilities plus rollback and a clone state signature."""

    probes: tuple[EvaluationPublicProbe, ...]
    rollback_probe: EvaluationPublicProbe
    state_signature: str
    operations: int

    def __post_init__(self) -> None:
        if (not isinstance(self.probes, tuple) or not self.probes
                or any(not isinstance(item, EvaluationPublicProbe)
                       for item in self.probes)
                or not isinstance(self.rollback_probe, EvaluationPublicProbe)
                or not isinstance(self.state_signature, str)
                or len(self.state_signature) != 64
                or type(self.operations) is not int
                or self.operations < 0):
            raise EvaluationKernelContractError(
                "public capability run projection drifted")


def build_evaluation_public_probe(
        key: str,
        *,
        evaluated: bool,
        passed: bool,
        evidence: dict[str, int],
        operations: int,
        ) -> EvaluationPublicProbe:
    """Build a fail-closed PASS/FAIL/NE probe from safe integer evidence."""
    if type(evaluated) is not bool or type(passed) is not bool:
        raise TypeError("public probe evaluated/passed flags must be bool")
    status = "PASS" if evaluated and passed else "FAIL" if evaluated else "NE"
    return EvaluationPublicProbe(
        key,
        status,
        _sha({
            "evidence": evidence,
            "evaluated": int(evaluated),
            "passed": int(passed),
            "result_key": key,
            "status": status,
        }),
        operations,
    )


def evaluation_public_plugin_semantic_sha256(
        repository: str | Path,
        *,
        plugin_key: str,
        plugin_version: str,
        relative_paths: tuple[str, ...],
        ) -> str:
    """Hash one public plugin and its exact active semantic file set."""
    root = Path(repository).resolve()
    if (not isinstance(plugin_key, str) or not plugin_key
            or not isinstance(plugin_version, str) or not plugin_version
            or not isinstance(relative_paths, tuple) or not relative_paths):
        raise TypeError("public plugin semantic inputs are invalid")
    rows = []
    for relative in relative_paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise EvaluationKernelContractError(
                "public plugin semantic path escaped repository") from error
        if not path.is_file() or path.is_symlink():
            raise EvaluationKernelContractError(
                "public plugin semantic file is unavailable")
        rows.append({
            "relative_path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return _sha({
        "files": rows,
        "plugin_key": plugin_key,
        "plugin_version": plugin_version,
    })


def _dimension_result(
        probe: EvaluationPublicProbe,
        role: str,
        ) -> EvaluationDimensionResult:
    counts = {
        "PASS": (1, 0, 0, 0),
        "FAIL": (0, 1, 0, 0),
        "NE": (0, 0, 1, 0),
        "BLOCKED": (0, 0, 0, 1),
    }[probe.status]
    return EvaluationDimensionResult(
        probe.key, role, probe.status, 1, *counts, probe.evidence_sha256)


def run_evaluation_public_plugin(
        context: EvaluationPluginRunContext,
        records: Iterable[object],
        *,
        policy: V2StageEvaluationPolicy,
        payload_factory: Callable[[tuple, tuple, tuple], object],
        capability_runner: Callable[[object], EvaluationPublicCapabilityRun],
        max_logic_operations: int,
        ) -> EvaluationPluginOutcome:
    """Run a stage capability twice and add the four shared V2 supports."""
    if (not isinstance(context, EvaluationPluginRunContext)
            or not isinstance(policy, V2StageEvaluationPolicy)
            or not callable(payload_factory)
            or not callable(capability_runner)
            or type(max_logic_operations) is not int
            or max_logic_operations <= 0):
        raise TypeError("public plugin runtime inputs are invalid")
    values = tuple(records)
    source_records = tuple(
        item for item in values
        if isinstance(item, EvaluationPublicSourceRecord))
    pairs = tuple(
        item for item in values
        if isinstance(item, EvaluationPublicObservationEvidencePair))
    if (values != (*source_records, *pairs)
            or not source_records or not pairs
            or any(item.source_binding_sha256
                   != context.source_binding_sha256 for item in values)):
        raise EvaluationKernelContractError(
            "public plugin stream order or binding drifted")
    payload = payload_factory(
        tuple(item.record for item in source_records),
        tuple(item.observation for item in pairs),
        tuple(item.evidence for item in pairs),
    )
    primary = capability_runner(payload)
    clone = capability_runner(payload)
    if (not isinstance(primary, EvaluationPublicCapabilityRun)
            or not isinstance(clone, EvaluationPublicCapabilityRun)
            or tuple(item.key for item in primary.probes)
            != (*policy.bearing_dimension_keys,
                policy.generation_hard_conjunct_key)):
        raise EvaluationKernelContractError(
            "public stage capability projection order drifted")
    logic_operations = primary.operations + clone.operations
    transport_payload = {
        "pairs": [
            {
                "evidence": item.evidence.to_dict(),
                "observation": item.observation.to_dict(),
            }
            for item in pairs
        ],
        "sources": [item.record.to_dict() for item in source_records],
    }
    transport_bytes = len(canonical_json_bytes(transport_payload))
    record_reads = len(source_records) + len(pairs) * 2
    resource = build_evaluation_public_probe(
        policy.hard_conjunct_keys[-4],
        evaluated=True,
        passed=(
            logic_operations <= max_logic_operations
            and logic_operations
            <= V2_RESOURCE_HARD_LIMITS["max_logic_operations"]
            and transport_bytes <= V2_RESOURCE_HARD_LIMITS["max_payload_bytes"]
            and record_reads <= V2_RESOURCE_HARD_LIMITS["max_records"]
        ),
        evidence={
            "logic_operations": logic_operations,
            "record_reads": record_reads,
            "transport_bytes": transport_bytes,
        },
        operations=0,
    )
    zero_call = build_evaluation_public_probe(
        policy.hard_conjunct_keys[-2],
        evaluated=True,
        passed=V2_ZERO_CALL_WINDOW_COUNT == 3,
        evidence={
            "companion_calls": 0,
            "llm_calls": 0,
            "teacher_calls": 0,
            "zero_call_window_count": V2_ZERO_CALL_WINDOW_COUNT,
        },
        operations=0,
    )
    primary_identity = tuple(
        (item.key, item.status, item.evidence_sha256)
        for item in (*primary.probes, primary.rollback_probe)
    )
    clone_identity = tuple(
        (item.key, item.status, item.evidence_sha256)
        for item in (*clone.probes, clone.rollback_probe)
    )
    clone_probe = build_evaluation_public_probe(
        policy.hard_conjunct_keys[-1],
        evaluated=True,
        passed=(primary_identity == clone_identity
                and primary.state_signature == clone.state_signature),
        evidence={
            "capability_projection_equal": int(
                primary_identity == clone_identity),
            "state_projection_equal": int(
                primary.state_signature == clone.state_signature),
        },
        operations=0,
    )
    probes = (
        *primary.probes,
        resource,
        primary.rollback_probe,
        zero_call,
        clone_probe,
    )
    roles = (
        *("BEARING" for _ in policy.bearing_dimension_keys),
        "GENERATION",
        *("SUPPORT" for _ in range(4)),
    )
    result_set = EvaluationResultSet(tuple(
        _dimension_result(probe, role)
        for probe, role in zip(probes, roles, strict=True)
    ))
    return EvaluationPluginOutcome(
        result_set,
        EvaluationRunAudit(
            "COMPLETE",
            len(source_records),
            len(pairs),
            record_reads,
            record_reads,
            transport_bytes,
            logic_operations,
            V2_ZERO_CALL_WINDOW_COUNT,
            V2WriteAccount(),
        ),
    )


__all__ = [
    "EvaluationPublicCapabilityRun",
    "EvaluationPublicProbe",
    "build_evaluation_public_probe",
    "evaluation_public_plugin_semantic_sha256",
    "run_evaluation_public_plugin",
]
