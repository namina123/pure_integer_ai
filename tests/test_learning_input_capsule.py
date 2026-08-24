"""B1 双学习平面 capsule 的最小能力回归。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_CONFLICT,
    ADMISSION_DUPLICATE,
    ADMISSION_SCOPE_MISMATCH,
    CoreDelta,
    CoreLearningState,
    EVENT_ASSERTION,
    EVENT_CONFLICT,
    EVENT_REVISION,
    EVENT_TOMBSTONE,
    LearningInputCapsule,
    LearningInputCapsuleError,
    LearningReplayReceipt,
    PromotionRequest,
    RuntimeMemoryEvent,
    RuntimeMemoryState,
    append_runtime_event,
    consume_core_delta,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope


def _source(number: int) -> SourceRef:
    return SourceRef(number, number, number, GLOBAL_OWNER_SCOPE, VersionBundle())


def _capsule(number: int = 1, *, scope_number: int = 7,
             delta: int = 1, parent: tuple[int, ...] = ()) -> LearningInputCapsule:
    source = _source(number)
    return LearningInputCapsule(
        source=source,
        scope=session_scope(scope_number, source=source),
        version_key=(1, number),
        parent_version_key=parent,
        language=1,
        modality=2,
        raw_content_digest=digest_bytes(f"raw-{number}-{delta}".encode()),
        structural_units=((number, delta), (delta, number)),
        authority_key=(9, number),
        license_id="CC0-1.0",
        split=1,
        delta_sequence=delta,
    )


def test_capsule_document_and_integer_record_round_trip() -> None:
    capsule = _capsule()
    restored = LearningInputCapsule.from_document(capsule.to_document())
    assert restored == capsule
    assert capsule.identity_key == restored.identity_key
    assert all(type(item) is int for item in capsule.canonical_record)


def test_capsule_rejects_scope_source_version_crossing() -> None:
    source = _source(1)
    other = _source(2)
    with pytest.raises(LearningInputCapsuleError):
        LearningInputCapsule(
            source, session_scope(1, source=other), (1,), (), 1, 1,
            digest_bytes(b"x"), ((1,),), (1,), "CC0-1.0", 1, 1)


def test_core_delta_consumes_new_once_and_rejects_changed_content() -> None:
    capsule = _capsule()
    state = CoreLearningState(capsule.scope.stable_key(), (100,))
    delta = CoreDelta((100,), capsule, graph_diff=(3, 4))
    state, outcome = consume_core_delta(state, delta)
    assert outcome == ADMISSION_ACCEPTED
    assert len(state.consumed_item_ledger) == 1
    state_again, outcome = consume_core_delta(state, delta)
    assert outcome == ADMISSION_DUPLICATE
    assert state_again == state

    changed = CoreDelta((100,), _capsule(delta=2), graph_diff=(3, 4))
    # The changed capsule has a new identity and is accepted as a real delta;
    # a same-identity changed record is tested by replacing the graph projection.
    state, outcome = consume_core_delta(state, changed)
    assert outcome == ADMISSION_ACCEPTED
    collided = CoreDelta((100,), capsule, graph_diff=(99,))
    _, outcome = consume_core_delta(state, collided)
    assert outcome == ADMISSION_CONFLICT


def test_core_delta_isolated_by_scope() -> None:
    capsule = _capsule(scope_number=7)
    other = _capsule(scope_number=8)
    state = CoreLearningState(capsule.scope.stable_key(), (100,))
    _, outcome = consume_core_delta(state, CoreDelta((100,), other))
    assert outcome == ADMISSION_SCOPE_MISMATCH


def test_runtime_revision_tombstone_and_duplicate() -> None:
    capsule = _capsule()
    state = RuntimeMemoryState(capsule.scope.stable_key())
    first = RuntimeMemoryEvent(capsule, (800,))
    state, outcome = append_runtime_event(state, first)
    assert outcome == ADMISSION_ACCEPTED
    duplicate_state, outcome = append_runtime_event(state, first)
    assert outcome == ADMISSION_DUPLICATE
    assert duplicate_state == state

    revised_capsule = _capsule(delta=2, parent=(1, 1))
    revision = RuntimeMemoryEvent(
        revised_capsule, (800,), EVENT_REVISION, 2, first.event_key)
    state, outcome = append_runtime_event(state, revision)
    assert outcome == ADMISSION_ACCEPTED
    tombstone = RuntimeMemoryEvent(
        _capsule(delta=3, parent=(1, 2)), (800,), EVENT_TOMBSTONE, 3,
        revision.event_key, True)
    state, outcome = append_runtime_event(state, tombstone)
    assert outcome == ADMISSION_ACCEPTED
    assert [item.revision for item in state.events] == [1, 2, 3]


def test_runtime_same_revision_preserves_conflict_instead_of_overwriting() -> None:
    capsule = _capsule()
    state = RuntimeMemoryState(capsule.scope.stable_key())
    first = RuntimeMemoryEvent(capsule, (801,))
    state, _ = append_runtime_event(state, first)
    competing = RuntimeMemoryEvent(_capsule(delta=2), (801,))
    state, outcome = append_runtime_event(state, competing)
    assert outcome == ADMISSION_CONFLICT
    assert len(state.events) == 2
    assert state.events[-1].event_kind == EVENT_CONFLICT


def test_promotion_is_explicit_and_source_scoped() -> None:
    capsule = _capsule()
    event = RuntimeMemoryEvent(capsule, (900,))
    request = PromotionRequest(
        event.event_key, capsule.source, capsule.scope,
        ((11, 12),), (4, 5), (6, 7), (8,))
    assert request.stable_key()
    with pytest.raises(LearningInputCapsuleError):
        PromotionRequest(
            event.event_key, _source(2), capsule.scope,
            ((11, 12),), (4, 5), (6, 7), (8,))


def test_replay_receipts_bind_both_projection_kinds() -> None:
    capsule = _capsule()
    delta = CoreDelta((100,), capsule, graph_diff=(3,))
    event = RuntimeMemoryEvent(capsule, (901,))
    core_receipt = LearningReplayReceipt.from_core_delta(
        delta, output_identity=(500,), replay_key=(700,))
    runtime_receipt = LearningReplayReceipt.from_runtime_event(
        event, replay_key=(701,))
    assert core_receipt.projection_kind != runtime_receipt.projection_kind
    assert core_receipt.stable_key()
    assert runtime_receipt.stable_key()
