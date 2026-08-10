"""Thin shared-kernel facade for the experimental public W-04 consumer."""
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
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_projection import (
    run_w04_v2_public_capability,
)


W04_V2_PUBLIC_PLUGIN_KEY = "W04-V2-PUBLIC-CAPABILITY-EXPERIMENTAL"
W04_V2_PUBLIC_PLUGIN_VERSION = "V1"
W04_V2_PUBLIC_EXPERIMENTAL = 1
W04_V2_PUBLIC_FORMAL_MASTERY_CLAIM = 0
W04_V2_PUBLIC_W04_STARTED = 0
W04_V2_PUBLIC_MAX_LOGIC_OPERATIONS = 100_000
W04_V2_PUBLIC_PLUGIN_FILES = (
    "src/pure_integer_ai/experiments/ph2_evaluation_public_source.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_public_plugin.py",
    "src/pure_integer_ai/experiments/ph2_w04_v2_public_source.py",
    "src/pure_integer_ai/experiments/ph2_w04_v2_public_projection.py",
    "src/pure_integer_ai/experiments/ph2_w04_v2_public_plugin.py",
    "src/pure_integer_ai/experiments/ph2_w04_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w04_learning.py",
    "src/pure_integer_ai/experiments/ph2_w04_understanding.py",
    "src/pure_integer_ai/experiments/ph2_w04_reasoning.py",
    "src/pure_integer_ai/experiments/ph2_w04_generation.py",
    "src/pure_integer_ai/experiments/ph2_w04_generation_contract.py",
)


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-04"
    )


def w04_v2_public_plugin_semantic_sha256(
        repository: str | Path,
        ) -> str:
    """Hash the public plugin and exact active W-04 code it projects."""
    return evaluation_public_plugin_semantic_sha256(
        repository,
        plugin_key=W04_V2_PUBLIC_PLUGIN_KEY,
        plugin_version=W04_V2_PUBLIC_PLUGIN_VERSION,
        relative_paths=W04_V2_PUBLIC_PLUGIN_FILES,
    )


# object-model: service-value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04V2PublicCapabilityPlugin:
    """Evaluate public records only; never create or consume a formal family."""

    declaration: EvaluationPluginDeclaration

    def __post_init__(self) -> None:
        policy = _policy()
        expected = (
            self.declaration.plugin_key == W04_V2_PUBLIC_PLUGIN_KEY
            and self.declaration.plugin_version == W04_V2_PUBLIC_PLUGIN_VERSION
            and self.declaration.stage_key == "W-04"
            and self.declaration.module_key
            == "ph2.w04.v2.public_capability.experimental"
            and self.declaration.symbol_key == "evaluate"
            and self.declaration.result_keys == policy.hard_conjunct_keys
        )
        if not expected:
            raise EvaluationKernelContractError(
                "W-04 public plugin declaration drifted")

    def evaluate(
            self,
            context: EvaluationPluginRunContext,
            records: Iterable[object],
            ) -> EvaluationPluginOutcome:
        """Run current W-04 behavior on a bounded public-only stream."""
        return run_evaluation_public_plugin(
            context,
            records,
            policy=_policy(),
            payload_factory=W04TrainingPayload,
            capability_runner=run_w04_v2_public_capability,
            max_logic_operations=W04_V2_PUBLIC_MAX_LOGIC_OPERATIONS,
        )


def build_w04_v2_public_capability_plugin(
        repository: str | Path,
        ) -> W04V2PublicCapabilityPlugin:
    """Bind live public plugin semantic identity to its declaration."""
    declaration = EvaluationPluginDeclaration(
        W04_V2_PUBLIC_PLUGIN_KEY,
        W04_V2_PUBLIC_PLUGIN_VERSION,
        "W-04",
        "ph2.w04.v2.public_capability.experimental",
        "evaluate",
        w04_v2_public_plugin_semantic_sha256(repository),
        _policy().hard_conjunct_keys,
    )
    return W04V2PublicCapabilityPlugin(declaration)


__all__ = [
    "W04_V2_PUBLIC_EXPERIMENTAL",
    "W04_V2_PUBLIC_FORMAL_MASTERY_CLAIM",
    "W04_V2_PUBLIC_PLUGIN_FILES",
    "W04_V2_PUBLIC_PLUGIN_KEY",
    "W04_V2_PUBLIC_PLUGIN_VERSION",
    "W04_V2_PUBLIC_W04_STARTED",
    "W04V2PublicCapabilityPlugin",
    "build_w04_v2_public_capability_plugin",
    "w04_v2_public_plugin_semantic_sha256",
]
