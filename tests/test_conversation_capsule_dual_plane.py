"""B2 同一 capsule 的 Core/Runtime/Dialogue 双平面纵切。"""
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
    CoreDelta,
    CoreLearningState,
    LearningInputCapsule,
    RuntimeMemoryState,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_capsule_dual_plane import (
    CapsuleDualPlaneError,
    run_capsule_dual_plane_turn,
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
    source = SourceRef(88002, 1, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    capsule = LearningInputCapsule(
        source=source,
        scope=session_scope(88002, source=source),
        version_key=(1, 88002),
        parent_version_key=(),
        language=1,
        modality=1,
        raw_content_digest=digest_bytes(bytes(raw)),
        structural_units=((1, len(raw)),),
        authority_key=(1, 88002),
        license_id="CC0-1.0",
        split=1,
        delta_sequence=1,
    )
    return capsule, raw


def _states(capsule):
    core = CoreLearningState(capsule.scope.stable_key(), (700,))
    delta = CoreDelta((700,), capsule, graph_diff=(11, 12))
    memory = RuntimeMemoryState(capsule.scope.stable_key())
    return core, delta, memory


def test_one_capsule_changes_core_runtime_and_dialogue(runtime) -> None:
    capsule, raw = _input(runtime)
    core, delta, memory = _states(capsule)
    dialogue = start_public_frame_dialogue((88002, 1))

    transition = run_capsule_dual_plane_turn(
        capsule, delta, core, raw, dialogue, memory, runtime)

    assert transition.core_admission_status == ADMISSION_ACCEPTED
    assert transition.core_after.consumed_item_ledger == (capsule.identity_key,)
    assert transition.core_after.deltas == (delta,)
    assert len(transition.dialogue.memory_after.events) == 1
    assert transition.dialogue.dialogue_turn.after.next_operation_ordinal == 2
    assert transition.core_receipt.input_identity == capsule.identity_key
    assert transition.canonical_record()


def test_dual_plane_replay_is_idempotent_on_both_ledgers(runtime) -> None:
    capsule, raw = _input(runtime)
    core, delta, memory = _states(capsule)
    dialogue = start_public_frame_dialogue((88002, 2))
    first = run_capsule_dual_plane_turn(
        capsule, delta, core, raw, dialogue, memory, runtime)
    replay = run_capsule_dual_plane_turn(
        capsule,
        delta,
        first.core_after,
        raw,
        dialogue,
        first.dialogue.memory_after,
        runtime,
    )

    assert replay.core_admission_status == ADMISSION_DUPLICATE
    assert replay.core_after == first.core_after
    assert replay.dialogue.memory_after == first.dialogue.memory_after
    assert replay.core_receipt == first.core_receipt
    assert replay.dialogue.receipt == first.dialogue.receipt


def test_core_conflict_stops_before_runtime_append(runtime) -> None:
    capsule, raw = _input(runtime)
    core, delta, memory = _states(capsule)
    dialogue = start_public_frame_dialogue((88002, 3))
    first = run_capsule_dual_plane_turn(
        capsule, delta, core, raw, dialogue, memory, runtime)
    changed = CoreDelta((700,), capsule, graph_diff=(99,))

    with pytest.raises(CapsuleDualPlaneError, match="Core delta 被拒绝"):
        run_capsule_dual_plane_turn(
            capsule,
            changed,
            first.core_after,
            raw,
            dialogue,
            first.dialogue.memory_after,
            runtime,
        )
    assert len(first.dialogue.memory_after.events) == 1
    assert changed.stable_key() != delta.stable_key()
