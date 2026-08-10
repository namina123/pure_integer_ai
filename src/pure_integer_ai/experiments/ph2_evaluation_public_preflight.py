"""Shared P0-P2 orchestration for public-only experimental stage plugins."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
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
from pure_integer_ai.experiments.ph2_evaluation_public_source import (
    EvaluationPublicBatch,
    build_evaluation_public_run_context,
)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPublicPreflightLayers:
    """Generic public P0-P2 layers and their real plugin outcome."""

    p0: EvaluationPreflightLayer
    p1: EvaluationPreflightLayer
    p2: EvaluationPreflightLayer
    outcome: EvaluationPluginOutcome

    def __post_init__(self) -> None:
        if (tuple(item.layer_key for item in (self.p0, self.p1, self.p2))
                != ("P0", "P1", "P2")
                or not isinstance(self.outcome, EvaluationPluginOutcome)):
            raise EvaluationKernelContractError(
                "public preflight layer projection drifted")


def build_evaluation_public_preflight_layers(
        batch: EvaluationPublicBatch,
        plugin: object,
        *,
        expected_stage_key: str,
        expected_result_keys: tuple[str, ...],
        live_semantic_sha256: str,
        ) -> EvaluationPublicPreflightLayers:
    """Run shared P0/P1/P2 checks without creating P3/P4 or a guard."""
    declaration = getattr(plugin, "declaration", None)
    evaluate = getattr(plugin, "evaluate", None)
    if (not isinstance(batch, EvaluationPublicBatch)
            or not isinstance(declaration, EvaluationPluginDeclaration)
            or not callable(evaluate)
            or not isinstance(expected_stage_key, str)
            or not expected_stage_key
            or not isinstance(expected_result_keys, tuple)
            or not expected_result_keys
            or not isinstance(live_semantic_sha256, str)
            or len(live_semantic_sha256) != 64):
        raise TypeError("public preflight inputs are invalid")
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
    declaration_pass = (
        declaration.result_keys == expected_result_keys
        and declaration.stage_key == expected_stage_key
    )
    p1 = build_preflight_layer("P1", (
        EvaluationPreflightCheck(
            "P1_EXPERIMENTAL_STATE_ZERO", "PASS",
            _sha({
                "experimental": 1,
                "formal_mastery_claim": 0,
                "stage_started": 0,
            })),
        EvaluationPreflightCheck(
            "P1_PLUGIN_DECLARATION",
            "PASS" if declaration_pass else "FAIL",
            declaration.sha256()),
        EvaluationPreflightCheck(
            "P1_PLUGIN_SEMANTIC_IDENTITY",
            "PASS" if declaration.semantic_sha256 == live_semantic_sha256
            else "FAIL",
            live_semantic_sha256),
    ))
    context = build_evaluation_public_run_context(batch, declaration)
    outcome = evaluate(context, batch.records)
    if not isinstance(outcome, EvaluationPluginOutcome):
        raise EvaluationKernelContractError(
            "public stage plugin outcome type drifted")
    p2 = build_preflight_layer("P2", tuple(
        EvaluationPreflightCheck(
            f"P2_{item.result_key}", item.status, item.evidence_sha256)
        for item in outcome.result_set.results
    ))
    return EvaluationPublicPreflightLayers(p0, p1, p2, outcome)


__all__ = [
    "EvaluationPublicPreflightLayers",
    "build_evaluation_public_preflight_layers",
]
