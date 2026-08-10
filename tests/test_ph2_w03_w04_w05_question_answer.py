"""FT09 纵向链上的无标签、来源绑定窄问答。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    compile_authored_primitive_atomic_bridge_course,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    compile_authored_semantic_primitive_bridge_course,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer import (
    W03_W04_W05_QUESTION_ANSWER_RESULT_SHA256,
    W03_W04_W05_QUESTION_REQUEST_SHA256,
    W03_W04_W05_QUESTION_STATE_SHA256,
    run_w03_w04_w05_question_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer_contract import (
    W03W04W05QuestionAnswerError,
    W03W04W05QuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical import (
    run_w03_w04_w05_vertical_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalQuery,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay import (
    VERTICAL_CONTEXT,
    VERTICAL_SURFACE,
    build_w03_w04_w05_vertical_overlay,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    build_w04_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    build_w05_v2_public_evaluation_batch,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SEMANTIC_PRIMITIVE_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_semantic_primitive_bridge_seed_v1.jsonl.sample")
PRIMITIVE_MAP_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_primitive_atomic_bridge_map_v1.jsonl.sample")
ATOMIC_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_primitive_atomic_bridge_seed_v1.jsonl.sample")


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft09_vertical_qa")
    base = compile_authored_semantic_primitive_bridge_course(
        SEMANTIC_PRIMITIVE_SAMPLE, root / "base")
    donor = compile_authored_primitive_atomic_bridge_course(
        PRIMITIVE_MAP_SAMPLE, ATOMIC_SAMPLE, root / "donor")
    return build_w03_w04_w05_vertical_overlay(base, donor)


def _vertical_query(*, generation: int = 1, learned: bool = True):
    context = VERTICAL_CONTEXT if learned else "微风使得树叶轻动。"
    return W03W04W05VerticalQuery(
        VERTICAL_SURFACE,
        context,
        context,
        allow_generation=generation,
    )


def _vertical_result(overlay):
    return run_w03_w04_w05_vertical_query(
        overlay.w03_batch,
        overlay.w04_batch,
        overlay.w05_batch,
        _vertical_query(),
        overlay_validation_sha256=overlay.validation_sha256,
    )


@pytest.fixture(scope="module")
def learned_roles(overlay):
    result = _vertical_result(overlay)
    candidate = result.w04_w05.w05_result.candidates[0]
    return tuple(sorted(item.role_key for item in candidate.role_bindings))


def _request(learned_roles, *, roles=None, learned=True, generation=1,
             source=None):
    selected_roles = learned_roles[:1] if roles is None else roles
    question_surface = (
        "什么使得河水上涨？" if learned else "什么使得树叶轻动？")
    return W03W04W05QuestionRequest(
        question_surface,
        _vertical_query(generation=generation, learned=learned),
        tuple(sorted(selected_roles)),
        source,
    )


def _run(overlay, request, *, w04=None, w05=None):
    return run_w03_w04_w05_question_answer(
        overlay.w03_batch,
        overlay.w04_batch if w04 is None else w04,
        overlay.w05_batch if w05 is None else w05,
        request,
        overlay_validation_sha256=overlay.validation_sha256,
    )


def test_answer_comes_only_from_learned_role_reasoning_and_generation(
        overlay, learned_roles) -> None:
    request = _request(learned_roles)
    serialized = request.to_dict()

    result = _run(overlay, request)

    assert set(serialized) == {
        "question_surface",
        "source_record_key",
        "target_role_keys",
        "vertical_query",
    }
    assert result.status == "ANSWER"
    assert result.answer_surface == "暴雨"
    assert result.proof is not None
    assert result.proof.reasoning_status == "AUTHORIZED"
    assert result.proof.generation_status == "READY"
    assert result.proof.generated_proposition_surface == VERTICAL_CONTEXT
    assert result.state_before_sha256 == result.state_after_sha256
    assert result.vertical_result.status == "BRIDGED"
    actual_request_sha256 = hashlib.sha256(canonical_json_bytes(
        request.to_dict())).hexdigest()
    assert actual_request_sha256 == W03_W04_W05_QUESTION_REQUEST_SHA256
    assert result.sha256() == W03_W04_W05_QUESTION_ANSWER_RESULT_SHA256
    assert result.state_before_sha256 == W03_W04_W05_QUESTION_STATE_SHA256


def test_unlearned_question_is_unknown_without_generation(
        overlay, learned_roles) -> None:
    result = _run(overlay, _request(learned_roles, learned=False))

    assert result.status == "UNKNOWN"
    assert result.answer_surface is None
    assert result.proof is None
    assert result.vertical_result.status == "UNKNOWN"


def test_ambiguous_typed_target_roles_require_clarification(
        overlay, learned_roles) -> None:
    result = _run(
        overlay,
        _request(learned_roles, roles=learned_roles),
    )

    assert result.status == "CLARIFY"
    assert result.answer_surface is None
    assert result.proof is None
    assert result.vertical_result.status == "BRIDGED"


def test_w03_w04_break_prevents_answer(overlay, learned_roles) -> None:
    observations = tuple(
        replace(item.observation, prerequisite_keys=())
        if item.observation.stable_key == overlay.base_w04_observation.stable_key
        else item.observation
        for item in overlay.w04_batch.pairs
    )
    broken = build_w04_v2_public_evaluation_batch(W04TrainingPayload(
        tuple(item.record for item in overlay.w04_batch.source_records),
        observations,
        tuple(item.evidence for item in overlay.w04_batch.pairs),
    ))

    result = _run(overlay, _request(learned_roles), w04=broken)

    assert result.status == "UNKNOWN"
    assert result.vertical_result.w03_w04.status == "UNKNOWN"
    assert result.vertical_result.w04_w05.status == "BRIDGED"


def test_w04_w05_break_prevents_answer(overlay, learned_roles) -> None:
    observations = tuple(
        replace(item.observation, prerequisite_keys=())
        if item.observation.stable_key == overlay.overlay_w05_observation.stable_key
        else item.observation
        for item in overlay.w05_batch.pairs
    )
    broken = build_w05_v2_public_evaluation_batch(W05TrainingPayload(
        tuple(item.record for item in overlay.w05_batch.source_records),
        observations,
        tuple(item.evidence for item in overlay.w05_batch.pairs),
    ))

    result = _run(overlay, _request(learned_roles), w05=broken)

    assert result.status == "UNKNOWN"
    assert result.vertical_result.w03_w04.status == "BRIDGED"
    assert result.vertical_result.w04_w05.status == "UNKNOWN"


def test_source_mismatch_is_unknown(overlay, learned_roles) -> None:
    request = _request(learned_roles, source=(1, 9_999_999))

    result = _run(overlay, request)

    assert result.status == "UNKNOWN"
    assert result.vertical_result.status == "BRIDGED"


def test_generation_is_a_hard_answer_conjunct(
        overlay, learned_roles) -> None:
    result = _run(overlay, _request(learned_roles, generation=0))

    assert result.status == "UNKNOWN"
    assert result.vertical_result.status == "BRIDGED"
    assert result.vertical_result.w04_w05.w05_result.generation_status == "NOT_RUN"


def test_repeated_question_is_read_only_and_proof_tampering_is_rejected(
        overlay, learned_roles) -> None:
    request = _request(learned_roles)
    before = tuple(
        batch.record_commitment
        for batch in (overlay.w03_batch, overlay.w04_batch, overlay.w05_batch))

    first = _run(overlay, request)
    second = _run(overlay, request)

    assert first.sha256() == second.sha256()
    assert first.state_before_sha256 == first.state_after_sha256
    assert before == tuple(
        batch.record_commitment
        for batch in (overlay.w03_batch, overlay.w04_batch, overlay.w05_batch))
    assert first.proof is not None
    with pytest.raises(
            W03W04W05QuestionAnswerError,
            match="generated proposition surface|Generation option"):
        replace(
            first,
            proof=replace(
                first.proof,
                generated_proposition_surface="伪造生成。",
            ),
        )
