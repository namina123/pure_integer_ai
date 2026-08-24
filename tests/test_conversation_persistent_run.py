"""M5 双平面检查点的跨进程式写入、恢复和按索引读取。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    CoreDelta,
    CoreLearningState,
    LearningInputCapsule,
    RuntimeMemoryState,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_capsule_dual_plane import (
    run_capsule_dual_plane_turn,
)
from pure_integer_ai.experiments.conversation_persistent_run import (
    PersistentDialogueCheckpoint,
    recover_dialogue_checkpoint,
    write_dialogue_checkpoint,
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
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT))


def _input(runtime):
    frame = next(item for item in runtime.base_catalog.frames
                 if item.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    raw = frame.surface_bytes
    source = SourceRef(88101, 1, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    capsule = LearningInputCapsule(
        source=source,
        scope=session_scope(88101, source=source),
        version_key=(1, 88101),
        parent_version_key=(),
        language=1,
        modality=1,
        raw_content_digest=digest_bytes(bytes(raw)),
        structural_units=((1, len(raw)),),
        authority_key=(1, 88101),
        license_id="CC0-1.0",
        split=1,
        delta_sequence=1,
    )
    return capsule, raw


def test_checkpoint_survives_close_and_uses_one_event_read(tmp_path, runtime) -> None:
    capsule, raw = _input(runtime)
    core = CoreLearningState(capsule.scope.stable_key(), (701,))
    delta = CoreDelta((701,), capsule, graph_diff=(17, 19))
    memory = RuntimeMemoryState(capsule.scope.stable_key())
    dialogue = start_public_frame_dialogue((88101, 1))
    first = run_capsule_dual_plane_turn(
        capsule, delta, core, raw, dialogue, memory, runtime)

    root = tmp_path / "m5-root"
    root.mkdir()
    from pure_integer_ai.storage.k_run_boundary import open_existing_run_root
    capability = open_existing_run_root(root, require_k_drive=False)
    first_checkpoint = PersistentDialogueCheckpoint(
        1, first.core_after, first.dialogue.memory_after,
        first.dialogue.dialogue_turn.after)
    write_dialogue_checkpoint(capability, first_checkpoint, runtime)

    recovered = recover_dialogue_checkpoint(root, runtime, require_k_drive=False)
    assert recovered.checkpoint.core_state == first.core_after
    assert recovered.checkpoint.runtime_state == first.dialogue.memory_after
    assert recovered.indexed_event_count == 1
    event, reads = recovered.query_event(first.dialogue.runtime_event.memory_item_key)
    assert event == first.dialogue.runtime_event
    assert reads == 1

    second = run_capsule_dual_plane_turn(
        capsule, delta, recovered.checkpoint.core_state, raw,
        recovered.checkpoint.dialogue_state,
        recovered.checkpoint.runtime_state, runtime)
    second_checkpoint = PersistentDialogueCheckpoint(
        2, second.core_after, second.dialogue.memory_after,
        second.dialogue.dialogue_turn.after,
        recovered.checkpoint_identity)
    write_dialogue_checkpoint(capability, second_checkpoint, runtime)

    restarted = recover_dialogue_checkpoint(root, runtime, require_k_drive=False)
    assert restarted.checkpoint.ordinal == 2
    assert restarted.checkpoint.previous_identity == recovered.checkpoint_identity
    assert restarted.checkpoint_identity == second_checkpoint.identity(runtime)
    assert restarted.checkpoint.dialogue_state == second.dialogue.dialogue_turn.after
    assert restarted.query_event(first.dialogue.runtime_event.memory_item_key)[1] == 1
