"""Thin shared-kernel facade for the experimental public W-05 consumer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
    EvaluationPluginOutcome,
    EvaluationPluginRunContext,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)
from pure_integer_ai.experiments.ph2_evaluation_public_plugin import (
    evaluation_public_plugin_semantic_sha256,
    run_evaluation_public_plugin,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_projection import (
    run_w05_v2_public_capability,
)


W05_V2_PUBLIC_PLUGIN_KEY = "W05-V2-PUBLIC-CAPABILITY-EXPERIMENTAL"
W05_V2_PUBLIC_PLUGIN_VERSION = "V1"
W05_V2_PUBLIC_EXPERIMENTAL = 1
W05_V2_PUBLIC_FORMAL_MASTERY_CLAIM = 0
W05_V2_PUBLIC_W05_STARTED = 0
W05_V2_PUBLIC_MAX_LOGIC_OPERATIONS = 100_000
W05_V2_PUBLIC_PLUGIN_FILES = (
    "src/pure_integer_ai/experiments/ph2_evaluation_public_source.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_public_plugin.py",
    "src/pure_integer_ai/experiments/ph2_w05_v2_public_source.py",
    "src/pure_integer_ai/experiments/ph2_w05_v2_public_projection.py",
    "src/pure_integer_ai/experiments/ph2_w05_v2_public_plugin.py",
    "src/pure_integer_ai/experiments/ph2_w05_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w05_learning.py",
    "src/pure_integer_ai/experiments/ph2_w05_understanding.py",
    "src/pure_integer_ai/experiments/ph2_w05_reasoning.py",
    "src/pure_integer_ai/experiments/ph2_w05_generation.py",
    "src/pure_integer_ai/experiments/ph2_w05_generation_contract.py",
)


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-05"
    )


def w05_v2_public_plugin_semantic_sha256(
        repository: str | Path,
        ) -> str:
    """Hash the public plugin and exact active W-05 code it projects."""
    return evaluation_public_plugin_semantic_sha256(
        repository,
        plugin_key=W05_V2_PUBLIC_PLUGIN_KEY,
        plugin_version=W05_V2_PUBLIC_PLUGIN_VERSION,
        relative_paths=W05_V2_PUBLIC_PLUGIN_FILES,
    )


# object-model: service-value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05V2PublicCapabilityPlugin:
    """Evaluate public records only; never create or consume a formal family."""

    declaration: EvaluationPluginDeclaration

    def __post_init__(self) -> None:
        policy = _policy()
        expected = (
            self.declaration.plugin_key == W05_V2_PUBLIC_PLUGIN_KEY
            and self.declaration.plugin_version == W05_V2_PUBLIC_PLUGIN_VERSION
            and self.declaration.stage_key == "W-05"
            and self.declaration.module_key
            == "ph2.w05.v2.public_capability.experimental"
            and self.declaration.symbol_key == "evaluate"
            and self.declaration.result_keys == policy.hard_conjunct_keys
        )
        if not expected:
            raise EvaluationKernelContractError(
                "W-05 public plugin declaration drifted")

    def evaluate(
            self,
            context: EvaluationPluginRunContext,
            records: Iterable[object],
            ) -> EvaluationPluginOutcome:
        """Run current W-05 behavior on a bounded public-only stream."""
        return run_evaluation_public_plugin(
            context,
            records,
            policy=_policy(),
            payload_factory=W05TrainingPayload,
            capability_runner=run_w05_v2_public_capability,
            max_logic_operations=W05_V2_PUBLIC_MAX_LOGIC_OPERATIONS,
        )


def build_w05_v2_public_capability_plugin(
        repository: str | Path,
        ) -> W05V2PublicCapabilityPlugin:
    """Bind live public plugin semantic identity to its declaration."""
    declaration = EvaluationPluginDeclaration(
        W05_V2_PUBLIC_PLUGIN_KEY,
        W05_V2_PUBLIC_PLUGIN_VERSION,
        "W-05",
        "ph2.w05.v2.public_capability.experimental",
        "evaluate",
        w05_v2_public_plugin_semantic_sha256(repository),
        _policy().hard_conjunct_keys,
    )
    return W05V2PublicCapabilityPlugin(declaration)


__all__ = [
    "W05_V2_PUBLIC_EXPERIMENTAL",
    "W05_V2_PUBLIC_FORMAL_MASTERY_CLAIM",
    "W05_V2_PUBLIC_PLUGIN_FILES",
    "W05_V2_PUBLIC_PLUGIN_KEY",
    "W05_V2_PUBLIC_PLUGIN_VERSION",
    "W05_V2_PUBLIC_W05_STARTED",
    "W05V2PublicCapabilityPlugin",
    "build_w05_v2_public_capability_plugin",
    "w05_v2_public_plugin_semantic_sha256",
]
