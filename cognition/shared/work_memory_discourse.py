"""把 G-02 typed 篇章/命题计划投影到 A-02 transient 内容。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_structure_plan import (
    DiscoursePlan,
    PropositionPlan,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    validate_semantic_identity,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.work_memory_content import (
    WorkMemoryContentItem,
    WorkMemoryOccurrenceAnchor,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_DISCOURSE_PROJECTION_VERSION = 1
_TRACE_CONTEXT = 1
_TRACE_OPEN_QUESTION = 2
_TRACE_SELECTED_PROPOSITION = 3


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为投影报告中的可变长键增加长度边界。"""
    return len(value), *value


@dataclass(frozen=True)
class WorkMemoryDiscourseRoles:
    """声明 G-02 三类 typed 输出各自写入哪个开放 WorkMemory Role。"""

    context: ObjectIdentity
    open_question: ObjectIdentity
    selected_proposition: ObjectIdentity

    def __post_init__(self) -> None:
        values = (self.context, self.open_question, self.selected_proposition)
        if any(
                not isinstance(item, ObjectIdentity)
                or item.object_kind != OBJECT_ROLE
                for item in values):
            raise TypeError("G-02 WorkMemory roles 必须是一等 Role")
        for item in values:
            validate_semantic_identity(item)
        if len(set(values)) != len(values):
            raise ValueError("G-02 三类投影必须使用不同 Role")

    def stable_key(self) -> tuple[int, ...]:
        """返回三类输出到开放 Role 的完整映射键。"""
        return (
            _DISCOURSE_PROJECTION_VERSION,
            *_packed(self.context.stable_key()),
            *_packed(self.open_question.stable_key()),
            *_packed(self.selected_proposition.stable_key()),
        )


@dataclass(frozen=True)
class WorkMemoryDiscourseProjection:
    """记录一次 G-02 计划实际写入的全部 A-02 内容项。"""

    discourse_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    roles: WorkMemoryDiscourseRoles
    items: tuple[WorkMemoryContentItem, ...]

    def __post_init__(self) -> None:
        for label, key in (
                ("discourse", self.discourse_key),
                ("proposition", self.proposition_key)):
            if not isinstance(key, tuple) or not key:
                raise ValueError(f"G-02 {label} key 必须是非空 tuple")
            assert_int(*key, _where=f"WorkMemoryDiscourseProjection.{label}")
        if not isinstance(self.roles, WorkMemoryDiscourseRoles):
            raise TypeError("G-02 projection roles 类型错误")
        if (not isinstance(self.items, tuple)
                or any(not isinstance(item, WorkMemoryContentItem)
                       for item in self.items)):
            raise TypeError("G-02 projection items 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回输入计划、Role 映射及实际内容引用。"""
        result = [
            _DISCOURSE_PROJECTION_VERSION,
            *_packed(self.discourse_key),
            *_packed(self.proposition_key),
            *_packed(self.roles.stable_key()),
            len(self.items),
        ]
        for item in self.items:
            result.extend(_packed(item.content_ref()))
        return tuple(result)


def _anchors_by_source(
        anchors: tuple[WorkMemoryOccurrenceAnchor, ...],
        ) -> dict[SourceRef, WorkMemoryOccurrenceAnchor]:
    """建立严格的一来源一 occurrence anchor 映射，拒绝稳定序代选。"""
    if (not isinstance(anchors, tuple) or not anchors
            or any(not isinstance(item, WorkMemoryOccurrenceAnchor)
                   for item in anchors)):
        raise TypeError("G-02 WorkMemory projection 必须提供 occurrence anchors")
    result: dict[SourceRef, WorkMemoryOccurrenceAnchor] = {}
    for anchor in anchors:
        if anchor.source in result:
            raise ValueError("G-02 每个来源必须恰有一个显式 occurrence anchor")
        result[anchor.source] = anchor
    return result


def _context_source(
        discourse: DiscoursePlan,
        propositions: PropositionPlan,
        anchors: dict[SourceRef, WorkMemoryOccurrenceAnchor],
        ) -> SourceRef:
    """确定 context 的唯一来源；多来源无声明时拒绝猜测。"""
    if discourse.declaration_source is not None:
        if discourse.declaration_source not in anchors:
            raise ValueError("G-02 discourse declaration source 缺少 occurrence anchor")
        return discourse.declaration_source
    proposition_sources = {
        item.source for item in propositions.propositions
    }
    if len(proposition_sources) == 1:
        source = next(iter(proposition_sources))
        if source not in anchors:
            raise ValueError("G-02 Proposition source 缺少 occurrence anchor")
        return source
    if not proposition_sources and len(anchors) == 1:
        return next(iter(anchors))
    raise ValueError("多来源 G-02 context 必须携带唯一 declaration source")


def project_generation_plans_to_work_memory(
        work_memory: WorkMemory,
        discourse: DiscoursePlan,
        propositions: PropositionPlan,
        *,
        roles: WorkMemoryDiscourseRoles,
        anchors: tuple[WorkMemoryOccurrenceAnchor, ...],
        logical_seq_start: int,
        trace: tuple[int, ...],
        ) -> WorkMemoryDiscourseProjection:
    """原子预演后写入 context、开放问题和 selected Proposition typed 状态。"""
    if not isinstance(work_memory, WorkMemory):
        raise TypeError("G-02 WorkMemory projection 需要 WorkMemory")
    if not isinstance(discourse, DiscoursePlan):
        raise TypeError("G-02 WorkMemory projection discourse 类型错误")
    if not isinstance(propositions, PropositionPlan):
        raise TypeError("G-02 WorkMemory projection propositions 类型错误")
    if not isinstance(roles, WorkMemoryDiscourseRoles):
        raise TypeError("G-02 WorkMemory projection roles 类型错误")
    assert_int(
        logical_seq_start,
        _where="project_generation_plans_to_work_memory.logical_seq_start",
    )
    if type(logical_seq_start) is not int or logical_seq_start < 0:
        raise ValueError("G-02 logical_seq_start 必须是严格非负整数")
    if not isinstance(trace, tuple) or not trace:
        raise ValueError("G-02 WorkMemory projection trace 不能为空")
    assert_int(*trace, _where="project_generation_plans_to_work_memory.trace")
    if discourse.selection_key != propositions.selection_key:
        raise ValueError("G-02 discourse 与 PropositionPlan selection 不一致")
    if set(discourse.candidate_keys) != {
            item.candidate_key for item in propositions.propositions}:
        raise ValueError("G-02 discourse 与 PropositionPlan candidate 集不一致")
    anchor_map = _anchors_by_source(anchors)
    context_source = _context_source(discourse, propositions, anchor_map)
    store = work_memory.require_content_store()
    pending: list[WorkMemoryContentItem] = []

    def append_item(
            role: ObjectIdentity,
            value: ObjectIdentity,
            source: SourceRef,
            trace_kind: int,
            ordinal: int,
            ) -> None:
        """按来源精确选择 anchor，并为一个 typed 输出建立内容项。"""
        anchor = anchor_map.get(source)
        if anchor is None:
            raise ValueError("G-02 typed 输出来源缺少 occurrence anchor")
        pending.append(WorkMemoryContentItem(
            role,
            value,
            anchor,
            store.scope_for_role(role),
            logical_seq_start + len(pending),
            (*trace, trace_kind, ordinal),
        ))

    for index, value in enumerate(discourse.context):
        append_item(
            roles.context, value, context_source, _TRACE_CONTEXT, index)
    for index, obligation in enumerate(discourse.open_questions):
        append_item(
            roles.open_question,
            obligation.proposition.template,
            obligation.source,
            _TRACE_OPEN_QUESTION,
            index,
        )
    for index, proposition in enumerate(propositions.propositions):
        append_item(
            roles.selected_proposition,
            proposition.proposition.template,
            proposition.source,
            _TRACE_SELECTED_PROPOSITION,
            index,
        )

    preview = store.clone()
    for item in pending:
        preview.put(item)
    committed = tuple(work_memory.put_content(item) for item in pending)
    return WorkMemoryDiscourseProjection(
        discourse.stable_key(),
        propositions.stable_key(),
        roles,
        committed,
    )


__all__ = [
    "WorkMemoryDiscourseProjection",
    "WorkMemoryDiscourseRoles",
    "project_generation_plans_to_work_memory",
]
