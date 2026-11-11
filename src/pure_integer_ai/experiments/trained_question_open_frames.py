"""把来源化问式的表面角色次序连接到完整Core来源，不改变事实绑定。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame


def question_observed_role_contract(question, roles) -> bool:
    """保留来源完整角色成员，普通WH按训练语序消费，指代仍保持原来源序。

    只允许省略一个明确答案角色，不增加角色或把答案混入观察槽；此合同不
    授予事实，当前跨度、filler、证据和绑定唯一性仍由各实际消费者核验。
    """
    if len(set(roles)) != len(roles):
        return False
    if question.is_slot_question:
        if question.answer_role_key not in roles:
            return False
        expected = tuple(role for role in roles if role != question.answer_role_key)
        if question.is_reference_slot:
            return question.roles == expected
    else:
        expected = roles
    return (len(question.roles) == len(expected)
            and len(set(question.roles)) == len(question.roles)
            and set(question.roles) == set(expected))


def bound_question_open_frames(questions, facts) -> tuple[OpenRoleFrame, ...]:
    """按完整角色成员授权受训表面次序；WH只省略一个原答案角色。

    来源bindings的次序不是问句的表面次序。完整问式可使用已训练的角色
    排列，但不能增删或重复角色。此函数只建立解析框架，当前filler、
    Core正证据和唯一绑定仍由后续共享消费者核验。
    """
    frames = []
    for question in questions:
        for fact in facts:
            roles = tuple(item.role.stable_key() for item in fact.bindings)
            if fact.predicate.stable_key() != question.predicate_key:
                continue
            if not question_observed_role_contract(question, roles):
                continue
            frames.append(OpenRoleFrame(
                fact.proposition.stable_key(), question.predicate_key,
                (fact.source_hash,), question.roles, question.gaps,
                (0, len(question.roles) + sum(map(len, question.gaps)))))
    return tuple(sorted(set(frames), key=lambda item: item.stable_key()))
