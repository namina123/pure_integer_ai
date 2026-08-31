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
        0, "请记住：我的回答偏好是先给出结论，再说明证据。", None, "CLARIFY")
    index = BroadDialogueMemoryRecallIndex((statement,))
    assert index.recall("你还记得我的回答偏好吗？") == statement.question


def test_initial_bootstrap_rejects_single_weak_overlap() -> None:
    prior = _turn(
        0, "请比较一下oneplus ace2与realme gt neo5这两款手机。",
        None, "UNKNOWN")
    index = BroadDialogueMemoryRecallIndex((prior,))
    assert index.recall("如何在mac机器上添加ssh密钥？") is None


def test_bootstrap_diffuses_across_one_intervening_statement() -> None:
    interest = _turn(
        0, "请记住：我的兴趣是研究整数图和长期记忆。", None, "UNKNOWN")
    preference = _turn(
        1, "以后谈到项目时，优先提醒性能约束。", None, "UNKNOWN")
    recall = "你还记得我刚才说的兴趣吗？"
    index = BroadDialogueMemoryRecallIndex((interest, preference))
    assert index.recall(recall) == interest.question

    index.append(_turn(2, recall, interest.question, "ANSWER"))
    recovered = BroadDialogueMemoryRecallIndex((
        interest, preference, _turn(2, recall, interest.question, "ANSWER")))
    assert recovered.recall(
        "你还记得我的长期研究兴趣吗？") == interest.question


def test_clarify_single_overlap_does_not_bootstrap_recall() -> None:
    prior = _turn(
        0, "如何写一个最简单的 Python 程序。", None, "CLARIFY")
    index = BroadDialogueMemoryRecallIndex((prior,))
    assert index.recall("那你如何看待 UBI?") is None


def test_local_question_suffix_cannot_poison_recall_shape() -> None:
    prior = _turn(
        0, "我刚才最开始问了什么？", None, "UNKNOWN")
    unrelated = "不存在的独立对象_20260829是什么？"
    index = BroadDialogueMemoryRecallIndex((prior,))
    assert index.recall(unrelated) is None

    # Even if a legacy run already contains the accidental replay, the pair
    # lacks independent scalar evidence and must not become a recall shape.
    poisoned = _turn(1, unrelated, prior.question, "ANSWER")
    recovered = BroadDialogueMemoryRecallIndex((prior, poisoned))
    assert recovered.recall("青石台的颜色是什么？") is None
    assert recovered.recall("夜间模式与设置有什么关系？") is None
