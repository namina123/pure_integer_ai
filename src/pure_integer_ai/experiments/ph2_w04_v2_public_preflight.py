"""W-04 typed facade over shared public P0-P2 orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginOutcome,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.preflight import (
    EvaluationPreflightLayer,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)
from pure_integer_ai.experiments.ph2_evaluation_public_preflight import (
    build_evaluation_public_preflight_layers,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_plugin import (
    W04_V2_PUBLIC_EXPERIMENTAL,
    W04_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
    W04_V2_PUBLIC_W04_STARTED,
    W04V2PublicCapabilityPlugin,
    w04_v2_public_plugin_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-04"
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04V2PublicPreflight:
    """Real W-04 public P0-P2 evidence without a formal readiness claim."""

    p0: EvaluationPreflightLayer
    p1: EvaluationPreflightLayer
    p2: EvaluationPreflightLayer
    outcome: EvaluationPluginOutcome
    experimental: int = W04_V2_PUBLIC_EXPERIMENTAL
    formal_mastery_claim: int = W04_V2_PUBLIC_FORMAL_MASTERY_CLAIM
    w04_started: int = W04_V2_PUBLIC_W04_STARTED

    def __post_init__(self) -> None:
        if (tuple(item.layer_key for item in (self.p0, self.p1, self.p2))
                != ("P0", "P1", "P2")
                or (self.experimental, self.formal_mastery_claim,
                    self.w04_started) != (1, 0, 0)):
            raise EvaluationKernelContractError(
                "W-04 public preflight crossed the formal boundary")


def build_w04_v2_public_preflight(
        repository: str | Path,
        batch: W04V2PublicEvaluationBatch,
        plugin: W04V2PublicCapabilityPlugin,
        ) -> W04V2PublicPreflight:
    """Produce W-04 P0/P1/P2 only; P3/P4 and guard remain unavailable."""
    if not isinstance(plugin, W04V2PublicCapabilityPlugin):
        raise TypeError("W-04 public plugin type is invalid")
    layers = build_evaluation_public_preflight_layers(
        batch,
        plugin,
        expected_stage_key="W-04",
        expected_result_keys=_policy().hard_conjunct_keys,
        live_semantic_sha256=w04_v2_public_plugin_semantic_sha256(repository),
    )
    return W04V2PublicPreflight(
        layers.p0, layers.p1, layers.p2, layers.outcome)


__all__ = [
    "W04V2PublicPreflight",
    "build_w04_v2_public_preflight",
]
