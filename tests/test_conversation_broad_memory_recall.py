from pure_integer_ai.experiments.conversation_broad_memory_recall import (
    BroadDialogueMemoryRecallIndex,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import DialogueTurn


def _turn(ordinal: int, question: str, answer: str | None,
          status: str) -> DialogueTurn:
    return DialogueTurn(
        ordinal, question, answer, answer, status, None, None,
        (ordinal + 1,) * 32,
    )


def test_incremental_memory_recall_learns_replay_shape_and_rejects_topic_only() -> None:
    statement = _turn(
        0, "请记住：我的兴趣是研究整数图和长期记忆。", None, "UNKNOWN")
    first_recall = "你还记得我刚才说的兴趣吗？"
    turns = [statement]
    index = BroadDialogueMemoryRecallIndex(tuple(turns))

    assert index.recall(first_recall) == statement.question
    turns.append(_turn(1, first_recall, statement.question, "ANSWER"))
    index.append(turns[-1])
    turns.append(_turn(
        2, "长期记忆是否意味着每轮都读取全部历史？", None, "UNKNOWN"))
    index.append(turns[-1])
    assert index.recall("长期记忆会一直降低查询性能吗？") is None

    style = _turn(
        3, "请记住：回答要清楚自然，不应编造事实。", None, "UNKNOWN")
    turns.append(style)
    index.append(style)
    second_recall = "你还记得我对回答方式的要求吗？"
    assert index.recall(second_recall) == style.question
    turns.append(_turn(4, second_recall, style.question, "ANSWER"))
    index.append(turns[-1])

    recovered = BroadDialogueMemoryRecallIndex(tuple(turns))
    combined = recovered.recall("你还能回忆我的研究兴趣和回答要求吗？")
    assert combined is not None
    assert statement.question in combined
    assert style.question in combined


def test_clarify_statement_is_recalled_without_promoting_it_to_answer() -> None:
    statement = _turn(
        0, "请记住：我偏好先给出结论，再说明证据。", None, "CLARIFY")
    index = BroadDialogueMemoryRecallIndex((statement,))
    assert index.recall("你还记得我的回答偏好吗？") == statement.question
