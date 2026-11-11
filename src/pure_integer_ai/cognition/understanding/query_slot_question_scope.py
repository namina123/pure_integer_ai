"""以完整训练 WH 语法限定关系角色的投影位置，保留原始观察和来源路径。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_open_roles import match_open_roles
from pure_integer_ai.cognition.understanding.query_relation_questions import TrainedRelationQuestionFrame
from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
    available_slot_question_frames, slot_contract,
)

_RoleSpan = tuple[tuple[int, ...], int, int]


def slot_question_projection_roles(
        frames: tuple[TrainedRelationQuestionFrame, ...],
        values: tuple[int, ...],
        source_ref: tuple[int, ...],
        *, role_span_groups: dict[tuple[int, ...], tuple[frozenset[_RoleSpan], ...]],
        ) -> dict[tuple[int, ...], frozenset[_RoleSpan]]:
    """恢复每个无歧义谓词问式允许的已观察角色与精确输入跨度。

    全输入匹配的训练帧及来源化组合须在操作、答案角色、模式和全部已观察
    跨度上达成同一合同。同合同内只要有当前同一来源命题逐槽支持的解析，
    未落地的前缀吞入解析不参与跨度竞争；不同合同、多个已落地解析仍保留。
    返回值是可删除重建的位置索引，后续绑定仍须核验完整父帧和 Core 角色。
    """
    available = available_slot_question_frames(frames, values)
    if not available:
        return {}
    by_frame = {item.open_frame().stable_key(): item for item in available}
    contracts = {}
    parses = match_open_roles(tuple(item.open_frame() for item in available),
                              values, source_ref)
    for parsed in parses:
        frame = by_frame[parsed.frame.stable_key()]
        observed = tuple((item.role, item.start, item.end, item.values)
                         for item in parsed.bindings)
        contracts.setdefault(frame.predicate_key, {}).setdefault(
            slot_contract(frame), set()).add(observed)
    result = {}
    for predicate, choices_by_contract in sorted(contracts.items()):
        if len(choices_by_contract) != 1:
            continue
        choices, = choices_by_contract.values()
        grounded = {
            choice for choice in choices
            if any(frozenset((role, start, end) for role, start, end, _values
                             in choice).issubset(group)
                   for group in role_span_groups.get(predicate, ()))
        }
        selected = grounded or choices
        if len(selected) == 1:
            result[predicate] = frozenset(
                (role, start, end) for role, start, end, _values
                in next(iter(selected)))
    return result
