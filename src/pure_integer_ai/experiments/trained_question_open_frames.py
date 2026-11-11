"""把来源化问式的表面角色次序连接到完整Core来源，不改变事实绑定。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame


def bound_question_open_frames(questions, facts) -> tuple[OpenRoleFrame, ...]:
    """按完整角色成员授权问式次序；WH仍保留原省略角色及顺序合同。

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
            if question.is_slot_question:
                if (len(set(roles)) != len(roles)
                        or question.answer_role_key not in roles
                        or question.roles != tuple(
                            role for role in roles
                            if role != question.answer_role_key)):
                    continue
            elif (len(set(roles)) != len(roles)
                    or len(roles) != len(question.roles)
                    or set(roles) != set(question.roles)):
                continue
            frames.append(OpenRoleFrame(
                fact.proposition.stable_key(), question.predicate_key,
                (fact.source_hash,), question.roles, question.gaps,
                (0, len(question.roles) + sum(map(len, question.gaps)))))
    return tuple(sorted(set(frames), key=lambda item: item.stable_key()))
