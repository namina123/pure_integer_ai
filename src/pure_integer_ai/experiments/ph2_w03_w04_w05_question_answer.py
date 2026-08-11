"""在显式 W-03/W-04/W-05 链上运行无标签窄问答。"""
from __future__ import annotations

import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_public_source import (
    EvaluationPublicBatch,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer_contract import (
    W03W04W05AnswerProof,
    W03W04W05QuestionAnswerError,
    W03W04W05QuestionAnswerResult,
    W03W04W05QuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical import (
    run_w03_w04_w05_vertical_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalResult,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05V2PublicCandidateProjection,
    W05V2PublicGenerationProjection,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)


W03_W04_W05_QUESTION_REQUEST_SHA256 = (
    "a64a8e74e04b1043417d12aa41b4d7f7188e5dc693cec89bbf78b9ae89d16f92")
W03_W04_W05_QUESTION_ANSWER_RESULT_SHA256 = (
    "c2198868c7a34ace066b55e6a6f83ae854ec0dcdc5acd5b0aac5a097627693e2")
W03_W04_W05_QUESTION_STATE_SHA256 = (
    "318a3a30dd1948fa7707e3966850b68454b447c77fefaf04ab237749bcbc78e3")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _batch_projection(batch: EvaluationPublicBatch) -> dict[str, object]:
    """序列化全部不可变公开学习记录，供只读门复核。"""
    return {
        "pairs": [
            {
                "evidence": item.evidence.to_dict(),
                "observation": item.observation.to_dict(),
                "source_binding_sha256": item.source_binding_sha256,
            }
            for item in batch.pairs
        ],
        "record_commitment": batch.record_commitment,
        "source_binding": batch.source_binding.to_dict(),
        "source_records": [
            {
                "record": item.record.to_dict(),
                "source_binding_sha256": item.source_binding_sha256,
            }
            for item in batch.source_records
        ],
        "transport_bytes": batch.transport_bytes,
    }


def public_w03_w04_w05_state_sha256(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        ) -> str:
    """Commit the complete immutable public state consumed by one answer."""
    if (not isinstance(w03_batch, EvaluationPublicBatch)
            or not isinstance(w04_batch, EvaluationPublicBatch)
            or not isinstance(w05_batch, EvaluationPublicBatch)):
        raise TypeError("vertical question public state inputs are invalid")
    return _sha({
        "w03": _batch_projection(w03_batch),
        "w04": _batch_projection(w04_batch),
        "w05": _batch_projection(w05_batch),
    })


def _stop(
        request: W03W04W05QuestionRequest,
        status: str,
        vertical: W03W04W05VerticalResult,
        state_sha256: str,
        ) -> W03W04W05QuestionAnswerResult:
    return W03W04W05QuestionAnswerResult(
        request,
        status,
        None,
        None,
        vertical,
        state_sha256,
        state_sha256,
    )


def _candidate(
        vertical: W03W04W05VerticalResult,
        ) -> W05V2PublicCandidateProjection | None:
    if vertical.link is None:
        return None
    w05 = vertical.w04_w05.w05_result
    matches = tuple(
        item for item in w05.candidates
        if item.proposition_key == vertical.link.proposition_key
        and item.active == 1
        and item.lifecycle_status == "ACTIVE"
        and item.reasoning_status == "AUTHORIZED")
    return matches[0] if len(matches) == 1 else None


def _generation_options(
        vertical: W03W04W05VerticalResult,
        candidate: W05V2PublicCandidateProjection,
        ) -> tuple[W05V2PublicGenerationProjection, ...]:
    bridge = vertical.w04_w05.link
    if bridge is None:
        return ()
    return tuple(
        item for item in vertical.w04_w05.w05_result.generation_options
        if item.target_proposition_key == candidate.proposition_key
        and item.target_predicate_key == candidate.predicate_key
        and item.target_source_ref_key == candidate.source_ref_key
        and item.target_source_commitment == candidate.source_commitment
        and item.context_key == candidate.context_key
        and item.occurrence_order == candidate.occurrence_order
        and item.occurrence_order == bridge.occurrence_order
        and item.role_binding_keys
        == tuple(value.identity_key for value in candidate.role_bindings)
        and item.role_binding_keys == bridge.role_binding_keys
    )


def _generation_function_sha256(
        option: W05V2PublicGenerationProjection,
        ) -> str:
    """忽略构造证据来源，只比较目标生成行为是否逐字段相同。"""
    value = option.to_dict()
    for key in (
            "construction_source_commitment",
            "construction_source_proposition_key",
            "construction_source_ref_key"):
        value.pop(key)
    return _sha(value)


def project_w03_w04_w05_question_answer(
        request: W03W04W05QuestionRequest,
        vertical: W03W04W05VerticalResult,
        *,
        state_sha256: str,
        ) -> W03W04W05QuestionAnswerResult:
    """只选择唯一已学 RoleBinding filler，其余情况闭锁停止。"""
    if (not isinstance(request, W03W04W05QuestionRequest)
            or not isinstance(vertical, W03W04W05VerticalResult)
            or not isinstance(state_sha256, str)
            or len(state_sha256) != 64):
        raise TypeError("vertical question answer inputs are invalid")
    if vertical.status != "BRIDGED" or vertical.link is None:
        return _stop(
            request,
            "CLARIFY" if vertical.status == "CLARIFY" else "UNKNOWN",
            vertical,
            state_sha256,
        )
    if (request.source_record_key is not None
            and request.source_record_key != vertical.link.source_ref_key):
        return _stop(request, "UNKNOWN", vertical, state_sha256)
    if len(request.target_role_keys) != 1:
        return _stop(request, "CLARIFY", vertical, state_sha256)
    w05 = vertical.w04_w05.w05_result
    if (w05.status != "UNIQUE"
            or w05.selected_reasoning_status != "AUTHORIZED"
            or w05.generation_status != "READY"):
        return _stop(
            request,
            "CLARIFY" if w05.status == "MULTI" else "UNKNOWN",
            vertical,
            state_sha256,
        )
    candidate = _candidate(vertical)
    if candidate is None:
        return _stop(request, "UNKNOWN", vertical, state_sha256)
    bridge = vertical.w04_w05.link
    if (bridge is None
            or candidate.source_record_key != vertical.link.source_ref_key
            or candidate.source_commitment != vertical.link.source_commitment
            or candidate.predicate_key != vertical.link.predicate_key
            or candidate.occurrence_order != bridge.occurrence_order
            or tuple(item.identity_key for item in candidate.role_bindings)
            != bridge.role_binding_keys):
        return _stop(request, "UNKNOWN", vertical, state_sha256)
    generation = _generation_options(vertical, candidate)
    if not generation:
        return _stop(
            request,
            "UNKNOWN",
            vertical,
            state_sha256,
        )
    functional_identities = {
        _generation_function_sha256(item) for item in generation}
    if len(functional_identities) != 1:
        return _stop(
            request,
            "CLARIFY",
            vertical,
            state_sha256,
        )
    target_role = request.target_role_keys[0]
    bindings = tuple(
        item for item in candidate.role_bindings
        if item.role_key == target_role)
    if len(bindings) != 1:
        return _stop(
            request,
            "CLARIFY" if len(bindings) > 1 else "UNKNOWN",
            vertical,
            state_sha256,
        )
    binding = bindings[0]
    occurrences = tuple(
        item for item in candidate.occurrences
        if item.semantic_object_key == binding.filler_key
        and item.identity_key in generation[0].occurrence_order)
    if len(occurrences) != 1:
        return _stop(
            request,
            "CLARIFY" if len(occurrences) > 1 else "UNKNOWN",
            vertical,
            state_sha256,
        )
    occurrence = occurrences[0]
    option = min(generation, key=lambda item: _sha(item.to_dict()))
    proof = W03W04W05AnswerProof(
        vertical.link.source_ref_key,
        candidate.source_ref_key,
        vertical.link.source_commitment,
        vertical.link.w03_observation_key,
        vertical.link.w04_observation_key,
        vertical.link.w05_observation_key,
        candidate.proposition_key,
        candidate.predicate_key,
        binding.identity_key,
        binding.role_key,
        binding.filler_key,
        occurrence.identity_key,
        occurrence.start,
        occurrence.end,
        candidate.reasoning_status,
        w05.generation_status,
        option.construction_key,
        _sha(option.to_dict()),
        option.surface,
    )
    return W03W04W05QuestionAnswerResult(
        request,
        "ANSWER",
        occurrence.surface_fragment,
        proof,
        vertical,
        state_sha256,
        state_sha256,
    )


def run_w03_w04_w05_question_answer(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        request: W03W04W05QuestionRequest,
        *,
        overlay_validation_sha256: str,
        ) -> W03W04W05QuestionAnswerResult:
    """运行 FT08，按结构回答，并证明公开学习状态保持只读。"""
    if (not isinstance(w03_batch, EvaluationPublicBatch)
            or not isinstance(w04_batch, EvaluationPublicBatch)
            or not isinstance(w05_batch, EvaluationPublicBatch)
            or not isinstance(request, W03W04W05QuestionRequest)):
        raise TypeError("vertical question answer run inputs are invalid")
    before = public_w03_w04_w05_state_sha256(
        w03_batch, w04_batch, w05_batch)
    vertical = run_w03_w04_w05_vertical_query(
        w03_batch,
        w04_batch,
        w05_batch,
        request.vertical_query,
        overlay_validation_sha256=overlay_validation_sha256,
    )
    after = public_w03_w04_w05_state_sha256(
        w03_batch, w04_batch, w05_batch)
    if before != after:
        raise W03W04W05QuestionAnswerError(
            "vertical question changed learned public state")
    return project_w03_w04_w05_question_answer(
        request,
        vertical,
        state_sha256=before,
    )


__all__ = [
    "W03_W04_W05_QUESTION_ANSWER_RESULT_SHA256",
    "W03_W04_W05_QUESTION_REQUEST_SHA256",
    "W03_W04_W05_QUESTION_STATE_SHA256",
    "public_w03_w04_w05_state_sha256",
    "project_w03_w04_w05_question_answer",
    "run_w03_w04_w05_question_answer",
]
