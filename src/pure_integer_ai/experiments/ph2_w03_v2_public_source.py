"""W-03 typed facade over the shared public evaluation source adapter."""
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
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload


W03_V2_PUBLIC_SOURCE_ADAPTER_VERSION = "PH2-W03-V2-PUBLIC-SOURCE-V1"
W03V2PublicEvaluationBatch = EvaluationPublicBatch
W03V2PublicObservationEvidencePair = EvaluationPublicObservationEvidencePair
W03V2PublicSourceError = EvaluationPublicSourceError
W03V2PublicSourceRecord = EvaluationPublicSourceRecord


def build_w03_v2_public_evaluation_batch(
        payload: W03TrainingPayload,
        ) -> W03V2PublicEvaluationBatch:
    """Bind a W-03 typed payload to the shared public source contract."""
    if not isinstance(payload, W03TrainingPayload):
        raise TypeError("public W-03 source adapter requires W03TrainingPayload")
    return build_evaluation_public_batch(
        stage_key="W-03",
        adapter_version=W03_V2_PUBLIC_SOURCE_ADAPTER_VERSION,
        source_refs=payload.source_refs,
        observations=payload.observations,
        teacher_evidence=payload.teacher_evidence,
    )


def w03_v2_public_training_payload(
        batch: W03V2PublicEvaluationBatch,
        ) -> W03TrainingPayload:
    """Recover the W-03 typed payload from a shared public batch."""
    if not isinstance(batch, EvaluationPublicBatch):
        raise TypeError("public W-03 batch type is invalid")
    sources, observations, evidence = batch.training_records()
    return W03TrainingPayload(sources, observations, evidence)


def build_w03_v2_public_run_context(
        batch: W03V2PublicEvaluationBatch,
        declaration: EvaluationPluginDeclaration,
        ) -> EvaluationPluginRunContext:
    """Build the W-03 experimental context without a formal family."""
    return build_evaluation_public_run_context(batch, declaration)


__all__ = [
    "W03_V2_PUBLIC_SOURCE_ADAPTER_VERSION",
    "W03V2PublicEvaluationBatch",
    "W03V2PublicObservationEvidencePair",
    "W03V2PublicSourceError",
    "W03V2PublicSourceRecord",
    "build_w03_v2_public_evaluation_batch",
    "build_w03_v2_public_run_context",
    "w03_v2_public_training_payload",
]
