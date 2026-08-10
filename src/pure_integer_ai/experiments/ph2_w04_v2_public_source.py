"""W-04 typed facade over the shared public evaluation source adapter."""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
    EvaluationPluginRunContext,
)
from pure_integer_ai.experiments.ph2_evaluation_public_source import (
    EvaluationPublicBatch,
    EvaluationPublicObservationEvidencePair,
    EvaluationPublicSourceError,
    EvaluationPublicSourceRecord,
    build_evaluation_public_batch,
    build_evaluation_public_run_context,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload


W04_V2_PUBLIC_SOURCE_ADAPTER_VERSION = "PH2-W04-V2-PUBLIC-SOURCE-V1"
W04V2PublicEvaluationBatch = EvaluationPublicBatch
W04V2PublicObservationEvidencePair = EvaluationPublicObservationEvidencePair
W04V2PublicSourceError = EvaluationPublicSourceError
W04V2PublicSourceRecord = EvaluationPublicSourceRecord


def build_w04_v2_public_evaluation_batch(
        payload: W04TrainingPayload,
        ) -> W04V2PublicEvaluationBatch:
    """Bind a train-only W-04 payload to the shared public source contract."""
    if not isinstance(payload, W04TrainingPayload):
        raise TypeError("public W-04 source adapter requires W04TrainingPayload")
    return build_evaluation_public_batch(
        stage_key="W-04",
        adapter_version=W04_V2_PUBLIC_SOURCE_ADAPTER_VERSION,
        source_refs=payload.source_refs,
        observations=payload.observations,
        teacher_evidence=payload.teacher_evidence,
    )


def w04_v2_public_training_payload(
        batch: W04V2PublicEvaluationBatch,
        ) -> W04TrainingPayload:
    """Recover the W-04 typed payload from a shared public batch."""
    if not isinstance(batch, EvaluationPublicBatch):
        raise TypeError("public W-04 batch type is invalid")
    sources, observations, evidence = batch.training_records()
    return W04TrainingPayload(sources, observations, evidence)


def build_w04_v2_public_run_context(
        batch: W04V2PublicEvaluationBatch,
        declaration: EvaluationPluginDeclaration,
        ) -> EvaluationPluginRunContext:
    """Build the W-04 experimental context without a formal family."""
    return build_evaluation_public_run_context(batch, declaration)


__all__ = [
    "W04_V2_PUBLIC_SOURCE_ADAPTER_VERSION",
    "W04V2PublicEvaluationBatch",
    "W04V2PublicObservationEvidencePair",
    "W04V2PublicSourceError",
    "W04V2PublicSourceRecord",
    "build_w04_v2_public_evaluation_batch",
    "build_w04_v2_public_run_context",
    "w04_v2_public_training_payload",
]
