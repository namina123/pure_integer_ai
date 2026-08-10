"""Thin shared-kernel facade for the experimental public W-03 consumer."""
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
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_v2_public_projection import (
    run_w03_v2_public_capability,
)


W03_V2_PUBLIC_PLUGIN_KEY = "W03-V2-PUBLIC-CAPABILITY-EXPERIMENTAL"
W03_V2_PUBLIC_PLUGIN_VERSION = "V1"
W03_V2_PUBLIC_EXPERIMENTAL = 1
W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM = 0
W03_V2_PUBLIC_W03_STARTED = 0
W03_V2_PUBLIC_MAX_LOGIC_OPERATIONS = 100_000
W03_V2_PUBLIC_PLUGIN_FILES = (
    "src/pure_integer_ai/experiments/ph2_evaluation_public_source.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_public_plugin.py",
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


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-03"
    )


def w03_v2_public_plugin_semantic_sha256(
        repository: str | Path,
        ) -> str:
    """Hash the public plugin and the exact active W-03 code it projects."""
    return evaluation_public_plugin_semantic_sha256(
        repository,
        plugin_key=W03_V2_PUBLIC_PLUGIN_KEY,
        plugin_version=W03_V2_PUBLIC_PLUGIN_VERSION,
        relative_paths=W03_V2_PUBLIC_PLUGIN_FILES,
    )


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
        return run_evaluation_public_plugin(
            context,
            records,
            policy=_policy(),
            payload_factory=W03TrainingPayload,
            capability_runner=run_w03_v2_public_capability,
            max_logic_operations=W03_V2_PUBLIC_MAX_LOGIC_OPERATIONS,
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
