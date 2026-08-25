from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_DUPLICATE,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueTurn,
)
from pure_integer_ai.experiments.conversation_broad_runtime_memory_bridge import (
    append_dialogue_turn_to_runtime_memory,
    empty_runtime_memory_for_conversation,
    replay_dialogue_state_to_runtime_memory,
)


def _turn(ordinal: int) -> DialogueTurn:
    return DialogueTurn(
        ordinal, f"用户问题{ordinal}", f"回答{ordinal}",
        f"表面{ordinal}", "ANSWER", "来源", "https://example.invalid",
        (ordinal + 1, ordinal + 2), f"用户问题{ordinal}",
    )


def test_dialogue_user_turn_enters_existing_runtime_memory_idempotently() -> None:
    conversation_key = (17, 3, 9)
    memory = empty_runtime_memory_for_conversation(conversation_key)
    first = append_dialogue_turn_to_runtime_memory(memory, conversation_key, _turn(0))
    replay = append_dialogue_turn_to_runtime_memory(
        first.memory_after, conversation_key, _turn(0))
    assert first.admission_status == ADMISSION_ACCEPTED
    assert replay.admission_status == ADMISSION_DUPLICATE
    assert replay.memory_after == first.memory_after
    assert len(first.memory_after.events) == 1
    assert first.event.capsule.raw_content_digest
    assert first.event.capsule.scope.stable_key() == memory.scope_key


def test_bounded_dialogue_state_replay_rebuilds_runtime_events() -> None:
    key = (4, 5, 6)
    state = BroadDialogueState(key, 2, (_turn(0), _turn(1)))
    memory = replay_dialogue_state_to_runtime_memory(state)
    assert len(memory.events) == 2
    assert memory.events[0].memory_item_key != memory.events[1].memory_item_key
