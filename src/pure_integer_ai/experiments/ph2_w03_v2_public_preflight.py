"""Public P0-P2 orchestration for the experimental W-03 V2 plugin."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginOutcome,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.preflight import (
    EvaluationPreflightCheck,
    EvaluationPreflightLayer,
    build_preflight_layer,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_plugin import (
    W03_V2_PUBLIC_EXPERIMENTAL,
    W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
    W03_V2_PUBLIC_W03_STARTED,
    W03V2PublicCapabilityPlugin,
    w03_v2_public_plugin_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
    build_w03_v2_public_run_context,
)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-03"
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicPreflight:
    """Real public P0-P2 evidence without any formal readiness claim."""

    p0: EvaluationPreflightLayer
    p1: EvaluationPreflightLayer
    p2: EvaluationPreflightLayer
    outcome: EvaluationPluginOutcome
    experimental: int = W03_V2_PUBLIC_EXPERIMENTAL
    formal_mastery_claim: int = W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM
    w03_started: int = W03_V2_PUBLIC_W03_STARTED

    def __post_init__(self) -> None:
        if (tuple(item.layer_key for item in (self.p0, self.p1, self.p2))
                != ("P0", "P1", "P2")
                or (self.experimental, self.formal_mastery_claim,
                    self.w03_started) != (1, 0, 0)):
            raise EvaluationKernelContractError(
                "W-03 public preflight crossed the formal boundary")


def build_w03_v2_public_preflight(
        repository: str | Path,
        batch: W03V2PublicEvaluationBatch,
        plugin: W03V2PublicCapabilityPlugin,
        ) -> W03V2PublicPreflight:
    """Produce P0/P1/P2 only; P3/P4 and formal guard remain unavailable."""
    if (not isinstance(batch, W03V2PublicEvaluationBatch)
            or not isinstance(plugin, W03V2PublicCapabilityPlugin)):
        raise TypeError("W-03 public preflight inputs are invalid")
    p0 = build_preflight_layer("P0", (
        EvaluationPreflightCheck(
            "P0_PUBLIC_LICENSE_CLOSURE", "PASS",
            batch.source_binding.source_ref_commitment),
        EvaluationPreflightCheck(
            "P0_SOURCE_FIRST_STREAM", "PASS",
            _sha({
                "record_commitment": batch.record_commitment,
                "source_count": len(batch.source_records),
            })),
        EvaluationPreflightCheck(
            "P0_TRAIN_PAIR_CLOSURE", "PASS",
            _sha({
                "pair_count": len(batch.pairs),
                "source_binding": batch.source_binding.sha256(),
            })),
    ))
    live_semantic = w03_v2_public_plugin_semantic_sha256(repository)
    declaration_pass = (
        plugin.declaration.result_keys == _policy().hard_conjunct_keys
        and plugin.declaration.stage_key == "W-03"
    )
    p1 = build_preflight_layer("P1", (
        EvaluationPreflightCheck(
            "P1_EXPERIMENTAL_STATE_ZERO", "PASS",
            _sha({
                "experimental": W03_V2_PUBLIC_EXPERIMENTAL,
                "formal_mastery_claim": W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
                "w03_started": W03_V2_PUBLIC_W03_STARTED,
            })),
        EvaluationPreflightCheck(
            "P1_PLUGIN_DECLARATION",
            "PASS" if declaration_pass else "FAIL",
            plugin.declaration.sha256()),
        EvaluationPreflightCheck(
            "P1_PLUGIN_SEMANTIC_IDENTITY",
            "PASS" if plugin.declaration.semantic_sha256 == live_semantic
            else "FAIL",
            live_semantic),
    ))
    context = build_w03_v2_public_run_context(batch, plugin.declaration)
    outcome = plugin.evaluate(context, batch.records)
    p2 = build_preflight_layer("P2", tuple(
        EvaluationPreflightCheck(
            f"P2_{item.result_key}", item.status, item.evidence_sha256)
        for item in outcome.result_set.results
    ))
    return W03V2PublicPreflight(p0, p1, p2, outcome)


__all__ = [
    "W03V2PublicPreflight",
    "build_w03_v2_public_preflight",
]
