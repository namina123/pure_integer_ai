"""W-05 typed facade over the shared public evaluation source adapter."""
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
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload


W05_V2_PUBLIC_SOURCE_ADAPTER_VERSION = "PH2-W05-V2-PUBLIC-SOURCE-V1"
W05V2PublicEvaluationBatch = EvaluationPublicBatch
W05V2PublicObservationEvidencePair = EvaluationPublicObservationEvidencePair
W05V2PublicSourceError = EvaluationPublicSourceError
W05V2PublicSourceRecord = EvaluationPublicSourceRecord


def build_w05_v2_public_evaluation_batch(
        payload: W05TrainingPayload,
        ) -> W05V2PublicEvaluationBatch:
    """Bind a train-only W-05 payload to the shared public source contract."""
    if not isinstance(payload, W05TrainingPayload):
        raise TypeError("public W-05 source adapter requires W05TrainingPayload")
    return build_evaluation_public_batch(
        stage_key="W-05",
        adapter_version=W05_V2_PUBLIC_SOURCE_ADAPTER_VERSION,
        source_refs=payload.source_refs,
        observations=payload.observations,
        teacher_evidence=payload.teacher_evidence,
    )


def w05_v2_public_training_payload(
        batch: W05V2PublicEvaluationBatch,
        ) -> W05TrainingPayload:
    """Recover the W-05 typed payload from a shared public batch."""
    if not isinstance(batch, EvaluationPublicBatch):
        raise TypeError("public W-05 batch type is invalid")
    sources, observations, evidence = batch.training_records()
    return W05TrainingPayload(sources, observations, evidence)


def build_w05_v2_public_run_context(
        batch: W05V2PublicEvaluationBatch,
        declaration: EvaluationPluginDeclaration,
        ) -> EvaluationPluginRunContext:
    """Build the W-05 experimental context without a formal family."""
    return build_evaluation_public_run_context(batch, declaration)


__all__ = [
    "W05_V2_PUBLIC_SOURCE_ADAPTER_VERSION",
    "W05V2PublicEvaluationBatch",
    "W05V2PublicObservationEvidencePair",
    "W05V2PublicSourceError",
    "W05V2PublicSourceRecord",
    "build_w05_v2_public_evaluation_batch",
    "build_w05_v2_public_run_context",
    "w05_v2_public_training_payload",
]
