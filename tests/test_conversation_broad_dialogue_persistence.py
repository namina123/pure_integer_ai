from pathlib import Path

from pure_integer_ai.experiments.conversation_broad_dialogue_persistence import (
    recover_broad_dialogue_checkpoint,
    write_broad_dialogue_checkpoint,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueCitation,
    DialogueTurn,
)
from pure_integer_ai.experiments.conversation_broad_runtime_memory_bridge import (
    append_dialogue_turn_to_runtime_memory,
    empty_runtime_memory_for_conversation,
)
from pure_integer_ai.storage.k_run_boundary import create_new_run_root


def _turn(ordinal: int) -> DialogueTurn:
    return DialogueTurn(
        ordinal,
        "问题" + str(ordinal),
        "答案" + str(ordinal),
        "表面" + str(ordinal),
        "ANSWER",
        "来源" + str(ordinal),
        "https://example.invalid/" + str(ordinal),
        (ordinal + 1, ordinal + 2),
        "检索" + str(ordinal),
        (DialogueCitation(
            "证据" + str(ordinal),
            "来源" + str(ordinal),
            "https://example.invalid/" + str(ordinal),
            "CC-BY-4.0", "作者" + str(ordinal),
            (1, 2, ordinal + 3, 0, 0, 0, 1, 0, 0, 0, 0)),),
    )


def test_broad_dialogue_checkpoint_recovers_hot_history_without_text_files(tmp_path: Path) -> None:
    root_path = tmp_path / "broad-session"
    root = create_new_run_root(root_path, require_k_drive=False)
    state = BroadDialogueState((7, 8, 9), 1, (_turn(0),))
    write_broad_dialogue_checkpoint(root, state)

    recovered = recover_broad_dialogue_checkpoint(root_path, require_k_drive=False)
    assert recovered.checkpoint.state == state
    assert recovered.indexed_turn_count == 1
    assert recovered.query_turn(0) == (_turn(0), 1)
    assert recovered.query_turn(99) == (None, 1)

    second = BroadDialogueState((7, 8, 9), 2, (_turn(0), _turn(1)))
    memory = empty_runtime_memory_for_conversation(second.conversation_key)
    memory = append_dialogue_turn_to_runtime_memory(
        memory, second.conversation_key, _turn(0)).memory_after
    memory = append_dialogue_turn_to_runtime_memory(
        memory, second.conversation_key, _turn(1)).memory_after
    write_broad_dialogue_checkpoint(root, second, runtime_memory_state=memory)
    restarted = recover_broad_dialogue_checkpoint(root_path, require_k_drive=False)
    assert restarted.checkpoint.ordinal == 2
    assert restarted.checkpoint.previous_identity == recovered.checkpoint_identity
    assert restarted.checkpoint.state == second
    assert restarted.query_turn(1) == (_turn(1), 1)
    assert restarted.query_relevant_turns("问题0", limit=1) == (_turn(0),)
    extended = restarted.with_turn(_turn(2))
    assert extended.query_relevant_turns("问题2", limit=1) == (_turn(2),)
    item = memory.events[1].memory_item_key
    event, reads = restarted.query_runtime_memory_item(item)
    assert event == memory.events[1]
    assert reads == 1


def test_runtime_only_operation_can_persist_without_fake_dialogue_turn(tmp_path: Path) -> None:
    root_path = tmp_path / "runtime-only-session"
    root = create_new_run_root(root_path, require_k_drive=False)
    state = BroadDialogueState((21, 22), 0, ())
    memory = empty_runtime_memory_for_conversation(state.conversation_key)
    write_broad_dialogue_checkpoint(root, state, runtime_memory_state=memory)
    recovered = recover_broad_dialogue_checkpoint(
        root_path, require_k_drive=False)
    assert recovered.checkpoint.ordinal == 1
    assert recovered.checkpoint.state == state
    assert recovered.checkpoint.runtime_memory_state == memory
