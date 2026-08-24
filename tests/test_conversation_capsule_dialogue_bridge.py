"""B1 capsule -> Runtime Memory -> raw dialogue 的单条端到端验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_DUPLICATE,
    LearningInputCapsule,
    RuntimeMemoryState,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_capsule_dialogue_bridge import (
    CapsuleDialogueBridgeError,
    run_capsule_dialogue_turn,
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
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    start_public_frame_dialogue,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime():
    closure = load_public_source_payload_closure_from_root(_ROOT)
    return build_public_dialogue_runtime_v1(closure)


def _input(runtime):
    frame = next(
        item for item in runtime.base_catalog.frames
        if item.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    raw = frame.surface_bytes
    source = SourceRef(88001, 1, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    capsule = LearningInputCapsule(
        source=source,
        scope=session_scope(88001, source=source),
        version_key=(1, 88001),
        parent_version_key=(),
        language=1,
        modality=1,
        raw_content_digest=digest_bytes(bytes(raw)),
        structural_units=((1, len(raw)),),
        authority_key=(1, 88001),
        license_id="CC0-1.0",
        split=1,
        delta_sequence=1,
    )
    return capsule, raw


def test_capsule_input_reaches_memory_and_real_dialogue(runtime) -> None:
    capsule, raw = _input(runtime)
    state = start_public_frame_dialogue((88001, 1))
    memory = RuntimeMemoryState(capsule.scope.stable_key())

    transition = run_capsule_dialogue_turn(
        capsule, raw, state, memory, runtime)

    assert transition.admission_status == ADMISSION_ACCEPTED
    assert transition.memory_before == memory
    assert len(transition.memory_after.events) == 1
    assert transition.dialogue_turn.before == state
    assert transition.dialogue_turn.after.next_operation_ordinal == 2
    assert transition.receipt.input_identity == capsule.identity_key
    assert transition.canonical_record()


def test_replay_is_duplicate_and_turn_receipt_are_identical(runtime) -> None:
    capsule, raw = _input(runtime)
    state = start_public_frame_dialogue((88001, 2))
    memory = RuntimeMemoryState(capsule.scope.stable_key())
    first = run_capsule_dialogue_turn(capsule, raw, state, memory, runtime)
    replay = run_capsule_dialogue_turn(
        capsule, raw, state, first.memory_after, runtime)

    assert replay.admission_status == ADMISSION_DUPLICATE
    assert replay.memory_after == first.memory_after
    assert replay.dialogue_turn == first.dialogue_turn
    assert replay.receipt == first.receipt


def test_digest_and_scope_mismatch_fail_closed(runtime) -> None:
    capsule, raw = _input(runtime)
    state = start_public_frame_dialogue((88001, 3))
    memory = RuntimeMemoryState(capsule.scope.stable_key())
    with pytest.raises(CapsuleDialogueBridgeError, match="digest"):
        run_capsule_dialogue_turn(capsule, (*raw, 0), state, memory, runtime)
    with pytest.raises(CapsuleDialogueBridgeError, match="scope"):
        run_capsule_dialogue_turn(
            capsule, raw, state,
            RuntimeMemoryState((999,)), runtime)
