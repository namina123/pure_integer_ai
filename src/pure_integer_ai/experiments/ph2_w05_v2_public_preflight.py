"""W-05 typed facade over shared public P0-P2 orchestration."""
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
from pure_integer_ai.experiments.ph2_w05_v2_public_plugin import (
    W05_V2_PUBLIC_EXPERIMENTAL,
    W05_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
    W05_V2_PUBLIC_W05_STARTED,
    W05V2PublicCapabilityPlugin,
    w05_v2_public_plugin_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-05"
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicPreflight:
    """Real W-05 public P0-P2 evidence without a formal readiness claim."""

    p0: EvaluationPreflightLayer
    p1: EvaluationPreflightLayer
    p2: EvaluationPreflightLayer
    outcome: EvaluationPluginOutcome
    experimental: int = W05_V2_PUBLIC_EXPERIMENTAL
    formal_mastery_claim: int = W05_V2_PUBLIC_FORMAL_MASTERY_CLAIM
    w05_started: int = W05_V2_PUBLIC_W05_STARTED

    def __post_init__(self) -> None:
        if (tuple(item.layer_key for item in (self.p0, self.p1, self.p2))
                != ("P0", "P1", "P2")
                or (self.experimental, self.formal_mastery_claim,
                    self.w05_started) != (1, 0, 0)):
            raise EvaluationKernelContractError(
                "W-05 public preflight crossed the formal boundary")


def build_w05_v2_public_preflight(
        repository: str | Path,
        batch: W05V2PublicEvaluationBatch,
        plugin: W05V2PublicCapabilityPlugin,
        ) -> W05V2PublicPreflight:
    """Produce W-05 P0/P1/P2 only; P3/P4 and guard remain unavailable."""
    if not isinstance(plugin, W05V2PublicCapabilityPlugin):
        raise TypeError("W-05 public plugin type is invalid")
    layers = build_evaluation_public_preflight_layers(
        batch,
        plugin,
        expected_stage_key="W-05",
        expected_result_keys=_policy().hard_conjunct_keys,
        live_semantic_sha256=w05_v2_public_plugin_semantic_sha256(repository),
    )
    return W05V2PublicPreflight(
        layers.p0, layers.p1, layers.p2, layers.outcome)


__all__ = [
    "W05V2PublicPreflight",
    "build_w05_v2_public_preflight",
]
