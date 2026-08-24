"""DLG-RAW-12 V3 append-only provider/focus context 纯状态专项。"""
from __future__ import annotations

from dataclasses import replace

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
    PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
    PROVIDER_ORIGIN_ANCHOR_STATUS_NONE,
    PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
    PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1,
    ProviderOriginAnchorProjectionV1,
    ProviderOriginOccurrenceV1,
    ProviderOriginRoleBindingV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context import (
    MIXED_FOCUS_CONTEXT_APPEND_REJECT_ADMISSION,
    MIXED_FOCUS_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
    MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL,
    MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS,
    MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1,
    MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1,
    MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE,
    FrameQuestionAnswerTurnV3,
    ProviderOriginContextTurnV3,
    ProviderOriginFollowupFocusTurnV1,
    ProviderOriginFocusAdmissionV1,
    ProviderOriginFocusContextError,
    start_mixed_conversation_focus_context_v3,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    intake_raw_conversation_vector,
)


_CONVERSATION_KEY = (91201, 1, 1)
_LEGACY_KEY = (91201, 2, 1)


def _key(value: int) -> tuple[int, ...]:
    """构造只服务于本专项的稳定整数 key。"""
    return (91201, value, 1)


def _digest(value: int) -> tuple[int, ...]:
    """构造格式正确并相互可区分的 raw u8[32] identity。"""
    return (value,) * 32


def _identity(domain: bytes, record: tuple[int, ...]) -> tuple[int, ...]:
    """按公开 portable SHA framing 形成专项 carrier identity。"""
    return tuple(portable_sha256_v1(domain, (record,)))


def _foreign_record(identity: tuple[int, ...]) -> tuple[int, ...]:
    """构造带显式末尾 raw identity 段的外部 canonical record 见证。"""
    return (1, 32, *identity)


def _anchor() -> ProviderOriginAnchorProjectionV1:
    """构造具有两个同 source occurrence 的最小已接纳 anchor。"""
    first_binding = ProviderOriginRoleBindingV1(
        _key(101), _key(111), _key(121), 0)
    second_binding = ProviderOriginRoleBindingV1(
        _key(102), _key(112), _key(122), 1)
    first_occurrence = ProviderOriginOccurrenceV1(
        _key(131), first_binding.filler_key, 0, 0, 1)
    second_occurrence = ProviderOriginOccurrenceV1(
        _key(132), second_binding.filler_key, 1, 1, 2)
    return ProviderOriginAnchorProjectionV1(
        PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
        PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
        _digest(1),
        _digest(2),
        _digest(3),
        _digest(4),
        _digest(5),
        _digest(6),
        _key(141),
        _key(142),
        _digest(7),
        _key(143),
        _key(144),
        _key(145),
        _key(146),
        _key(147),
        PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1,
        _key(148),
        first_binding.binding_key,
        first_binding.role_key,
        first_binding.filler_key,
        first_occurrence.occurrence_key,
        first_occurrence.start,
        first_occurrence.end,
        (first_binding, second_binding),
        (first_occurrence, second_occurrence),
        (65, 66),
        (65, 66),
    )


def _binding_and_occurrence(
        anchor: ProviderOriginAnchorProjectionV1,
        index: int,
        ) -> tuple[ProviderOriginRoleBindingV1, ProviderOriginOccurrenceV1]:
    """按固定测试序读取一组同 anchor role/occurrence。"""
    return anchor.ordered_role_bindings[index], anchor.ordered_occurrences[index]


def _admission(
        anchor: ProviderOriginAnchorProjectionV1,
        *,
        reference_index: int,
        target_index: int,
        marker: int,
        ) -> ProviderOriginFocusAdmissionV1:
    """构造一条完整 V3 admission，其 target 只能是 anchor 原始 slice。"""
    reference_binding, reference_occurrence = _binding_and_occurrence(
        anchor, reference_index)
    target_binding, target_occurrence = _binding_and_occurrence(anchor, target_index)
    catalog_identity = _digest(30 + marker)
    form_identity = _digest(40 + marker)
    candidate_identity = _digest(50 + marker)
    input_record = intake_raw_conversation_vector((88, marker)).canonical_record()
    target_scalars = anchor.output_scalars[
        target_occurrence.start:target_occurrence.end]
    target_u8 = anchor.output_u8[
        target_occurrence.start:target_occurrence.end]
    readback_record = intake_raw_conversation_vector(target_u8).canonical_record()
    return ProviderOriginFocusAdmissionV1(
        _foreign_record(catalog_identity),
        catalog_identity,
        _foreign_record(form_identity),
        form_identity,
        _foreign_record(candidate_identity),
        candidate_identity,
        input_record,
        _identity(MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1, input_record),
        readback_record,
        _identity(
            MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1,
            readback_record,
        ),
        anchor.provider_kind,
        anchor.provider_identity_u8,
        anchor.runtime_identity_u8,
        anchor.catalog_record_identity_u8,
        anchor.provider_result_identity_u8,
        anchor.anchor_identity_u8,
        anchor.source_record_key,
        anchor.source_ref_stable_key,
        anchor.source_commitment_u8,
        anchor.w03_observation_key,
        anchor.w04_observation_key,
        anchor.w05_observation_key,
        anchor.generation_construction_key,
        anchor.proposition_key,
        anchor.predicate_key,
        anchor.relation_kind_code,
        reference_binding.binding_key,
        reference_binding.role_key,
        reference_binding.filler_key,
        reference_occurrence.occurrence_key,
        reference_occurrence.start,
        reference_occurrence.end,
        target_binding.binding_key,
        target_binding.role_key,
        target_binding.filler_key,
        target_occurrence.occurrence_key,
        target_occurrence.start,
        target_occurrence.end,
        target_scalars,
        target_u8,
    )


def _source() -> SourceRef:
    """构造 Frame payload 所需的无 surface citation。"""
    return SourceRef(91201, 3, 4, GLOBAL_OWNER_SCOPE, VersionBundle())


def _frame_turn() -> ConversationTurnState:
    """构造带真实 legacy read 的 Frame typed turn。"""
    legacy = start_conversation_context(_LEGACY_KEY)
    return ConversationTurnState(
        0,
        _key(201),
        _key(202),
        _key(203),
        _key(204),
        ObjectIdentity(OBJECT_CONCEPT, _key(205)),
        (_key(206),),
        (_source(),),
        (_key(207),),
        _key(208),
        _key(209),
        legacy.read(0),
    )


def _after_provider() -> tuple[object, ProviderOriginAnchorProjectionV1]:
    """形成最小 V3 provider tail，供 focus admission 专项复用。"""
    anchor = _anchor()
    initial = start_mixed_conversation_focus_context_v3(_CONVERSATION_KEY)
    provider = initial.admit_provider_origin_projection(anchor, initial.read(0))
    assert provider.accepted
    assert isinstance(provider.appended_turn, ProviderOriginContextTurnV3)
    return provider.after, anchor


def test_provider_to_focus_then_focus_to_focus_is_append_only() -> None:
    """success path 必须形成 provider -> focus -> focus 三条独立 typed event。"""
    state, anchor = _after_provider()
    first = state.admit_provider_origin_followup_focus(
        _admission(anchor, reference_index=0, target_index=1, marker=1),
        state.read(1),
    )
    assert first.accepted
    assert isinstance(first.appended_turn, ProviderOriginFollowupFocusTurnV1)
    second = first.after.admit_provider_origin_followup_focus(
        _admission(anchor, reference_index=1, target_index=0, marker=2),
        first.after.read(1),
    )

    assert second.accepted
    assert second.after.revision == 3
    assert tuple(type(turn) for turn in second.after.turns) == (
        ProviderOriginContextTurnV3,
        ProviderOriginFollowupFocusTurnV1,
        ProviderOriginFollowupFocusTurnV1,
    )
    assert (second.appended_turn.parent_turn_identity_u8
            == first.appended_turn.turn_identity_u8)
    assert second.appended_turn.current_occurrence_key == anchor.focus_occurrence_key
    assert all(type(item) is int and item >= 0
               for item in second.after.canonical_record())


def test_wrong_read_anchor_none_and_frame_tail_are_noop() -> None:
    """错误 witness、ANCHOR_NONE 和 Frame tail 均不得追加或回溯旧 provider。"""
    state, anchor = _after_provider()
    valid = _admission(anchor, reference_index=0, target_index=1, marker=3)

    wrong_limit = state.admit_provider_origin_followup_focus(valid, state.read(0))
    assert not wrong_limit.accepted
    assert wrong_limit.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL
    assert wrong_limit.context_write_origin == MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE
    assert wrong_limit.after == state

    foreign = start_mixed_conversation_focus_context_v3((91201, 8, 1)).read(0)
    wrong_read = state.admit_provider_origin_followup_focus(valid, foreign)
    assert not wrong_read.accepted
    assert wrong_read.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS
    assert wrong_read.after == state

    none = ProviderOriginAnchorProjectionV1(PROVIDER_ORIGIN_ANCHOR_STATUS_NONE)
    none_result = state.admit_provider_origin_projection(none, state.read(1))
    assert not none_result.accepted
    assert none_result.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_ANCHOR_NONE
    assert none_result.prior_read is None
    assert none_result.after == state

    initial = start_mixed_conversation_focus_context_v3((91201, 9, 1))
    frame = initial.admit_frame_qa_run(_frame_turn(), initial.read(0))
    assert frame.accepted
    assert isinstance(frame.appended_turn, FrameQuestionAnswerTurnV3)
    frame_tail = frame.after.admit_provider_origin_followup_focus(
        valid,
        frame.after.read(1),
    )
    assert not frame_tail.accepted
    assert frame_tail.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL
    assert frame_tail.after == frame.after


def test_initial_admissions_require_exact_read_zero() -> None:
    """空 context 的 read(1) 即使可见为空，也不得冒充首轮 read(0)。"""
    anchor = _anchor()
    initial = start_mixed_conversation_focus_context_v3((91201, 10, 1))

    provider = initial.admit_provider_origin_projection(anchor, initial.read(1))
    assert not provider.accepted
    assert provider.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS
    assert provider.after == initial

    frame = initial.admit_frame_qa_run(_frame_turn(), initial.read(1))
    assert not frame.accepted
    assert frame.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS
    assert frame.after == initial


def test_admission_and_parent_identity_drift_fail_closed() -> None:
    """anchor/current occurrence/identity 漂移只能失败，不能写入替代 event。"""
    state, anchor = _after_provider()
    valid = _admission(anchor, reference_index=0, target_index=1, marker=4)
    first = state.admit_provider_origin_followup_focus(valid, state.read(1))
    assert first.accepted

    mismatched_reference = _admission(
        anchor, reference_index=0, target_index=1, marker=5)
    second = first.after.admit_provider_origin_followup_focus(
        mismatched_reference,
        first.after.read(1),
    )
    assert not second.accepted
    assert second.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_ADMISSION
    assert second.after == first.after

    with pytest.raises(ProviderOriginFocusContextError, match="identity 漂移"):
        replace(valid, admission_identity_u8=(0,) * 32)
