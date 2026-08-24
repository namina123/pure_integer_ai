"""公开无标签 response-act planning 输入的专项回归。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_public_reference_planning,
    compile_public_response_act_planning,
    public_response_act_planning_input_from_episode,
)


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_BASE = 20994


def test_public_input_and_planning_ignore_episode_response_act_label() -> None:
    """替换同一 episode 的 response_act 不得改变公开输入、候选或选择立场。"""
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.question.answer_plan.response_act == "CLARIFY")
    relabeled = replace(
        episode,
        question=replace(
            episode.question,
            answer_plan=replace(
                episode.question.answer_plan,
                response_act="UNKNOWN",
            ),
        ),
    )
    branch = language_branch_identity((_BASE, 1))
    original_input = public_response_act_planning_input_from_episode(episode)
    relabeled_input = public_response_act_planning_input_from_episode(relabeled)
    original = compile_public_response_act_planning(original_input, branch)
    relabeled_build = compile_public_response_act_planning(
        relabeled_input, branch)
    content = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 2, index))
        for index in range(1, 6)
    ))
    policy = EvidenceAnswerPolicyProtocol(*tuple(
        minimal_instruction_identity((_BASE, 3, index))
        for index in range(1, 5)
    ))
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(content, policy),
    )

    assert original_input.canonical_record() == relabeled_input.canonical_record()
    assert all(type(item) is int for item in original_input.stable_key())
    assert tuple(
        (item.proposition_id, item.candidate.state, item.candidate.stable_key())
        for item in original.candidate_bindings
    ) == tuple(
        (item.proposition_id, item.candidate.state, item.candidate.stable_key())
        for item in relabeled_build.candidate_bindings
    )
    assert original.planning.stable_key() == relabeled_build.planning.stable_key()
    assert selector.select(original.planning).stance == (
        selector.select(relabeled_build.planning).stance)
    assert selector.select(original.planning).stance == content.clarify
    assert not hasattr(original, "episode")


def test_public_inputs_form_unknown_clarify_and_conflict_planning() -> None:
    """三类 Evidence 状态均可由公开输入形成 planning 和对应选择立场。"""
    episodes = read_grounded_answer_episodes(_SAMPLE)
    branch = language_branch_identity((_BASE, 10))
    content = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 11, index))
        for index in range(1, 6)
    ))
    policy = EvidenceAnswerPolicyProtocol(*tuple(
        minimal_instruction_identity((_BASE, 12, index))
        for index in range(1, 5)
    ))
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(content, policy),
    )

    expected = {
        "UNKNOWN": content.unknown,
        "CLARIFY": content.clarify,
        "CONFLICT": content.conflict,
    }
    for response_act, stance in expected.items():
        episode = next(
            item for item in episodes
            if item.question.answer_plan.response_act == response_act)
        planning_input = public_response_act_planning_input_from_episode(
            episode)
        build = compile_public_response_act_planning(planning_input, branch)

        assert build.planning.goal.target_branch == branch
        assert selector.select(build.planning).stance == stance
        assert all(type(item) is int for item in planning_input.canonical_record())


def test_public_reference_planning_uses_explicit_order_not_answer_labels() -> None:
    """双 claim planning 只消费 Evidence 和调用方冻结的显式结构顺序。"""
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.episode_id == "train-grounded-reference-event-v2")
    original_plan = episode.question.answer_plan
    # 课程对象在常规构造时会用 ANSWER/reference 标签互相验签。这里直接遮蔽
    # 该训练专属字段，证明 public projection 的读取边界没有碰它。
    object.__setattr__(episode.question, "answer_plan", object())
    try:
        label_free_input = public_response_act_planning_input_from_episode(
            episode)
    finally:
        object.__setattr__(episode.question, "answer_plan", original_plan)
    branch = language_branch_identity((_BASE, 20))
    order = ("p-year", "p-registration")
    original = compile_public_reference_planning(
        public_response_act_planning_input_from_episode(episode),
        branch,
        order,
    )
    label_free_build = compile_public_reference_planning(
        label_free_input,
        branch,
        order,
    )

    assert tuple(
        item.proposition_id for item in original.candidate_bindings) == (
            "p-registration", "p-year")
    assert tuple(
        item.proposition.stable_key() for item in original.planning.candidates) == tuple(
        item.proposition.stable_key()
            for item in label_free_build.planning.candidates)
    assert original.planning.stable_key() == label_free_build.planning.stable_key()
    assert all(item.state.support and not item.state.refute
               for item in original.planning.candidates)
