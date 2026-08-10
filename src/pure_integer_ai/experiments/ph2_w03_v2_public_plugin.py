"""Thin shared-kernel facade for the experimental public W-03 consumer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_RESOURCE_HARD_LIMITS,
    V2_STAGE_EVALUATION_POLICIES,
    V2_ZERO_CALL_WINDOW_COUNT,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2WriteAccount,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
    EvaluationPluginOutcome,
    EvaluationPluginRunContext,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationDimensionResult,
    EvaluationKernelContractError,
    EvaluationResultSet,
    EvaluationRunAudit,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_v2_public_projection import (
    W03V2PublicProbe,
    run_w03_v2_public_capability,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicObservationEvidencePair,
    W03V2PublicSourceRecord,
)


W03_V2_PUBLIC_PLUGIN_KEY = "W03-V2-PUBLIC-CAPABILITY-EXPERIMENTAL"
W03_V2_PUBLIC_PLUGIN_VERSION = "V1"
W03_V2_PUBLIC_EXPERIMENTAL = 1
W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM = 0
W03_V2_PUBLIC_W03_STARTED = 0
W03_V2_PUBLIC_MAX_LOGIC_OPERATIONS = 100_000
W03_V2_PUBLIC_PLUGIN_FILES = (
    "src/pure_integer_ai/experiments/ph2_w03_v2_public_source.py",
    "src/pure_integer_ai/experiments/ph2_w03_v2_public_projection.py",
    "src/pure_integer_ai/experiments/ph2_w03_v2_public_plugin.py",
    "src/pure_integer_ai/experiments/ph2_w03_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w03_adapter_extractors.py",
    "src/pure_integer_ai/experiments/ph2_w03_understanding.py",
    "src/pure_integer_ai/experiments/ph2_w03_understanding_contract.py",
    "src/pure_integer_ai/experiments/ph2_w03_generation.py",
    "src/pure_integer_ai/experiments/ph2_w03_generation_contract.py",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-03"
    )


def _support_probe(
        key: str,
        *,
        passed: bool,
        evidence: dict[str, int],
        ) -> W03V2PublicProbe:
    status = "PASS" if passed else "FAIL"
    return W03V2PublicProbe(
        key,
        status,
        _sha({
            "evidence": evidence,
            "evaluated": 1,
            "passed": int(passed),
            "result_key": key,
            "status": status,
        }),
        0,
    )


def _dimension_result(
        probe: W03V2PublicProbe,
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


def w03_v2_public_plugin_semantic_sha256(
        repository: str | Path,
        ) -> str:
    """Hash the public plugin and the exact active W-03 code it projects."""
    root = Path(repository).resolve()
    rows = []
    for relative in W03_V2_PUBLIC_PLUGIN_FILES:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise EvaluationKernelContractError(
                "W-03 public plugin semantic path escaped repository") from error
        if not path.is_file() or path.is_symlink():
            raise EvaluationKernelContractError(
                "W-03 public plugin semantic file is unavailable")
        rows.append({
            "relative_path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return _sha({
        "files": rows,
        "plugin_key": W03_V2_PUBLIC_PLUGIN_KEY,
        "plugin_version": W03_V2_PUBLIC_PLUGIN_VERSION,
    })


# object-model: service-value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicCapabilityPlugin:
    """Evaluate public records only; never create or consume a formal family."""

    declaration: EvaluationPluginDeclaration

    def __post_init__(self) -> None:
        policy = _policy()
        expected = (
            self.declaration.plugin_key == W03_V2_PUBLIC_PLUGIN_KEY
            and self.declaration.plugin_version == W03_V2_PUBLIC_PLUGIN_VERSION
            and self.declaration.stage_key == "W-03"
            and self.declaration.module_key
            == "ph2.w03.v2.public_capability.experimental"
            and self.declaration.symbol_key == "evaluate"
            and self.declaration.result_keys == policy.hard_conjunct_keys
        )
        if not expected:
            raise EvaluationKernelContractError(
                "W-03 public plugin declaration drifted")

    def evaluate(
            self,
            context: EvaluationPluginRunContext,
            records: Iterable[object],
            ) -> EvaluationPluginOutcome:
        """Run the current W-03 behavior on a bounded public-only stream."""
        if not isinstance(context, EvaluationPluginRunContext):
            raise TypeError("W-03 public plugin context is invalid")
        values = tuple(records)
        source_records = tuple(
            item for item in values if isinstance(item, W03V2PublicSourceRecord))
        pairs = tuple(
            item for item in values
            if isinstance(item, W03V2PublicObservationEvidencePair))
        if (values != (*source_records, *pairs)
                or not source_records or not pairs
                or any(item.source_binding_sha256
                       != context.source_binding_sha256
                       for item in values)):
            raise EvaluationKernelContractError(
                "W-03 public plugin stream order or binding drifted")
        training_payload = W03TrainingPayload(
            tuple(item.record for item in source_records),
            tuple(item.observation for item in pairs),
            tuple(item.evidence for item in pairs),
        )
        primary = run_w03_v2_public_capability(training_payload)
        clone = run_w03_v2_public_capability(training_payload)
        policy = _policy()
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
        resource = _support_probe(
            policy.hard_conjunct_keys[-4],
            passed=(
                logic_operations <= W03_V2_PUBLIC_MAX_LOGIC_OPERATIONS
                and logic_operations
                <= V2_RESOURCE_HARD_LIMITS["max_logic_operations"]
                and transport_bytes
                <= V2_RESOURCE_HARD_LIMITS["max_payload_bytes"]
                and record_reads <= V2_RESOURCE_HARD_LIMITS["max_records"]
            ),
            evidence={
                "logic_operations": logic_operations,
                "record_reads": record_reads,
                "transport_bytes": transport_bytes,
            },
        )
        zero_call = _support_probe(
            policy.hard_conjunct_keys[-2],
            passed=V2_ZERO_CALL_WINDOW_COUNT == 3,
            evidence={
                "companion_calls": 0,
                "llm_calls": 0,
                "teacher_calls": 0,
                "zero_call_window_count": V2_ZERO_CALL_WINDOW_COUNT,
            },
        )
        primary_identity = tuple(
            (item.key, item.status, item.evidence_sha256)
            for item in (*primary.probes, primary.rollback_probe)
        )
        clone_identity = tuple(
            (item.key, item.status, item.evidence_sha256)
            for item in (*clone.probes, clone.rollback_probe)
        )
        clone_probe = _support_probe(
            policy.hard_conjunct_keys[-1],
            passed=(
                primary_identity == clone_identity
                and primary.state_signature == clone.state_signature
            ),
            evidence={
                "capability_projection_equal": int(
                    primary_identity == clone_identity),
                "state_projection_equal": int(
                    primary.state_signature == clone.state_signature),
            },
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
        source_count = len(source_records)
        pair_count = len(pairs)
        return EvaluationPluginOutcome(
            result_set,
            EvaluationRunAudit(
                "COMPLETE",
                source_count,
                pair_count,
                record_reads,
                record_reads,
                transport_bytes,
                logic_operations,
                V2_ZERO_CALL_WINDOW_COUNT,
                V2WriteAccount(),
            ),
        )


def build_w03_v2_public_capability_plugin(
        repository: str | Path,
        ) -> W03V2PublicCapabilityPlugin:
    """Bind the live public plugin semantic identity to its declaration."""
    declaration = EvaluationPluginDeclaration(
        W03_V2_PUBLIC_PLUGIN_KEY,
        W03_V2_PUBLIC_PLUGIN_VERSION,
        "W-03",
        "ph2.w03.v2.public_capability.experimental",
        "evaluate",
        w03_v2_public_plugin_semantic_sha256(repository),
        _policy().hard_conjunct_keys,
    )
    return W03V2PublicCapabilityPlugin(declaration)


__all__ = [
    "W03_V2_PUBLIC_EXPERIMENTAL",
    "W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM",
    "W03_V2_PUBLIC_PLUGIN_FILES",
    "W03_V2_PUBLIC_PLUGIN_KEY",
    "W03_V2_PUBLIC_PLUGIN_VERSION",
    "W03_V2_PUBLIC_W03_STARTED",
    "W03V2PublicCapabilityPlugin",
    "build_w03_v2_public_capability_plugin",
    "w03_v2_public_plugin_semantic_sha256",
]
