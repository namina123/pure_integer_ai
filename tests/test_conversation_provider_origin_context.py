"""DLG-RAW-11B 混合 Frame/provider-origin context 纯状态专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationTurnState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    PROVIDER_ORIGIN_ANCHOR_STATUS_NONE,
    PROVIDER_ORIGIN_CATALOG_IDENTITY_DOMAIN_V1,
    PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
    ProviderOriginAnchorProjectionV1,
    ProviderOriginProviderBindingV1,
    project_provider_origin_anchor_v1,
    provider_origin_legacy_proof_from_same_dispatch_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_APPEND_ACCEPTED,
    MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
    MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
    FrameQuestionAnswerTurnV2,
    ProviderOriginContextTurnV1,
    start_mixed_conversation_context_v2,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
    PublicProofSentenceProviderResultV1,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    PUBLIC_SENTENCE_DEMO_ROUTE_EXACT,
    build_public_sentence_demo_catalog,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    run_sparse_qa_query_with_typed_proof,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_public_sparse_qa_runtime_snapshot,
)


_ROOT = Path(__file__).resolve().parents[1]
_CONVERSATION_KEY = (65111, 1, 1)


def _source() -> SourceRef:
    """构造稳定的纯整数 Frame citation，不携带路径或 surface。"""
    return SourceRef(65111, 2, 3, GLOBAL_OWNER_SCOPE, VersionBundle())


def _frame_turn() -> ConversationTurnState:
    """构造一条具有完整 legacy typed record 的 Frame wrapper 输入。"""
    legacy_context = start_conversation_context((65111, 9, 1))
    return ConversationTurnState(
        0,
        (65111, 11, 1),
        (65111, 12, 1),
        (65111, 13, 1),
        (65111, 14, 1),
        ObjectIdentity(OBJECT_CONCEPT, (65111, 15, 1)),
        ((65111, 16, 1),),
        (_source(),),
        ((65111, 17, 1),),
        (65111, 18, 1),
        (65111, 19, 1),
        legacy_context.read(0),
    )


@pytest.fixture(scope="module")
def accepted_anchor() -> ProviderOriginAnchorProjectionV1:
    """只读取公开 frozen W03-W05 snapshot，构造同次 proof 的真实 ANSWER anchor。"""
    runtime = load_public_sparse_qa_runtime_snapshot(
        _ROOT / "data" / "ph2" / "sparse_qa_runtime_snapshot_v1.json",
        repository=_ROOT,
    )
    catalog = build_public_sentence_demo_catalog(runtime)
    route = next(item for item in catalog.routes
                 if item.route_kind == PUBLIC_SENTENCE_DEMO_ROUTE_EXACT)
    proof = run_sparse_qa_query_with_typed_proof(runtime, route.request)
    carrier = provider_origin_legacy_proof_from_same_dispatch_v1(proof)
    assert carrier is not None
    provider_identity = (1,) * 32
    runtime_identity = (2,) * 32
    catalog_record = (3, 5, 8)
    binding = ProviderOriginProviderBindingV1(
        PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
        provider_identity,
        runtime_identity,
        catalog_record,
        tuple(portable_sha256_v1(
            PROVIDER_ORIGIN_CATALOG_IDENTITY_DOMAIN_V1,
            (catalog_record,),
        )),
    )
    raw = tuple(route.request.question_surface.encode("utf-8"))
    result = PublicProofSentenceProviderResultV1(
        PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
        DLG_RAW_ACCEPT,
        intake_raw_conversation_vector(raw),
        provider_identity,
        runtime_identity,
        catalog_record,
        demo_record=(1,),
        route_kind=route.route_kind,
        source_record_key=carrier.source_record_key,
        output_scalars=carrier.generated_proposition_scalars,
        output_bytes=carrier.generated_proposition_u8,
        context_policy=PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1,
    )
    anchor = project_provider_origin_anchor_v1(binding, result, carrier)
    assert anchor.accepted
    return anchor


def _state_after_frame():
    """形成一条经 read(0) admission 写入的 V2 Frame state。"""
    initial = start_mixed_conversation_context_v2(_CONVERSATION_KEY)
    frame = _frame_turn()
    admission = initial.admit_frame_qa_run(frame, initial.read(0))
    assert admission.accepted
    return admission.after, frame


def test_frame_append_wraps_legacy_turn_and_records_prior_read_witness() -> None:
    """FRAME_QA_RUN 保留 legacy typed chain，并把 write origin 与 prior read 显式入 record。"""
    initial = start_mixed_conversation_context_v2(_CONVERSATION_KEY)
    frame = _frame_turn()
    prior_read = initial.read(0)

    admission = initial.admit_frame_qa_run(frame, prior_read)

    assert admission.result_code == MIXED_CONTEXT_APPEND_ACCEPTED
    assert admission.context_write_origin == MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN
    assert admission.after.revision == 1
    assert isinstance(admission.appended_turn, FrameQuestionAnswerTurnV2)
    assert admission.appended_turn.frame_turn == frame
    assert admission.appended_turn.prior_read_witness == prior_read.witness
    assert admission.after.latest_frame_target_turn(frame.target_key) == admission.appended_turn
    assert all(type(item) is int and item >= 0
               for item in admission.after.canonical_record())


def test_provider_answer_appends_tagged_projection_with_no_consumed_reference(
        accepted_anchor: ProviderOriginAnchorProjectionV1,
        ) -> None:
    """真实 provider ANCHOR_ANSWER 只形成 provider tagged turn，不借用 Frame 字段。"""
    state, _frame = _state_after_frame()
    prior_read = state.read(1)

    admission = state.admit_provider_origin_projection(
        accepted_anchor,
        prior_read,
    )

    assert admission.result_code == MIXED_CONTEXT_APPEND_ACCEPTED
    assert (admission.context_write_origin
            == MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION)
    assert isinstance(admission.appended_turn, ProviderOriginContextTurnV1)
    assert admission.appended_turn.anchor_projection == accepted_anchor
    assert (admission.appended_turn.provider_result_identity_u8
            == accepted_anchor.provider_result_identity_u8)
    assert admission.appended_turn.consumed_reference == ()
    assert admission.after.revision == 2


def test_provider_tail_cannot_jump_back_to_prior_frame_target(
        accepted_anchor: ProviderOriginAnchorProjectionV1,
        ) -> None:
    """target helper 只看可见尾轮；provider 写入后不得回溯早先 Frame。"""
    state, frame = _state_after_frame()
    assert state.latest_frame_target_turn(frame.target_key) is not None
    after_provider = state.admit_provider_origin_projection(
        accepted_anchor,
        state.read(1),
    ).after

    assert after_provider.latest_frame_target_turn(frame.target_key) is None
    assert after_provider.read(2).latest_frame_target_turn(frame.target_key) is None


def test_anchor_none_is_explicit_no_write_rejection() -> None:
    """ANCHOR_NONE 不消费传入 read，不追加 turn，且以 origin NONE 记录拒绝。"""
    state, _frame = _state_after_frame()
    anchor_none = ProviderOriginAnchorProjectionV1(
        PROVIDER_ORIGIN_ANCHOR_STATUS_NONE)

    admission = state.admit_provider_origin_projection(anchor_none, state.read(1))

    assert not admission.accepted
    assert admission.result_code == MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE
    assert admission.context_write_origin == MIXED_CONTEXT_WRITE_ORIGIN_NONE
    assert admission.prior_read is None
    assert admission.appended_turn is None
    assert admission.after == state
    assert admission.after.digest() == state.digest()


def test_mixed_snapshot_digest_is_stable_for_equal_inputs(
        accepted_anchor: ProviderOriginAnchorProjectionV1,
        ) -> None:
    """独立构造的相同 input/state 必须产生逐字节相同 canonical snapshot digest。"""
    first_a, _frame_a = _state_after_frame()
    first_b, _frame_b = _state_after_frame()
    final_a = first_a.admit_provider_origin_projection(
        accepted_anchor,
        first_a.read(1),
    ).after
    final_b = first_b.admit_provider_origin_projection(
        accepted_anchor,
        first_b.read(1),
    ).after

    assert final_a.canonical_record() == final_b.canonical_record()
    assert final_a.digest() == final_b.digest()
    assert len(final_a.digest()) == 32
    assert all(type(item) is int and 0 <= item <= 255
               for item in final_a.digest())
