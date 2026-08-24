"""M4 命题资格结果到多段可读回答组织的纵切。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_capsule_evidence_bridge import (
    run_capsule_evidence_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_capsule_response_organization import (
    SEGMENT_CLAIM,
    SEGMENT_QUALIFIER,
    SEGMENT_SUPPORT,
    ResponseOrganizationError,
    organize_capsule_response,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)

from test_conversation_capsule_evidence_bridge import _fixture


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime():
    closure = load_public_source_payload_closure_from_root(_ROOT)
    return build_public_dialogue_runtime_v1(closure)


def _transition(runtime, state="SUPPORTED"):
    values = _fixture(runtime, qualification_state=state)
    return run_capsule_evidence_dialogue_turn(*values, runtime)


def test_answer_is_claim_then_support_and_replayable(runtime) -> None:
    transition = _transition(runtime)
    plan = organize_capsule_response(
        transition,
        support_surfaces=("依据：已绑定来源证据。",),
    )

    assert plan.response_act == "ANSWER"
    assert tuple(item.segment_kind for item in plan.segments) == (
        SEGMENT_CLAIM, SEGMENT_SUPPORT)
    assert plan.output_bytes.count(10) == 1
    assert plan.output_surface.splitlines()[1] == "依据：已绑定来源证据。"
    assert plan.replay_key
    assert plan.canonical_record()


def test_unknown_has_fallback_without_claim(runtime) -> None:
    plan = organize_capsule_response(
        _transition(runtime, "UNKNOWN"),
        fallback_surfaces=("当前证据不足，无法确认。",),
    )

    assert plan.response_act == "UNKNOWN"
    assert tuple(item.segment_kind for item in plan.segments) == (SEGMENT_QUALIFIER,)
    assert "无法确认" in plan.output_surface
    assert all(item.segment_kind != SEGMENT_CLAIM for item in plan.segments)


def test_answer_without_support_is_rejected(runtime) -> None:
    with pytest.raises(ResponseOrganizationError, match="claim 和 support"):
        organize_capsule_response(_transition(runtime))
