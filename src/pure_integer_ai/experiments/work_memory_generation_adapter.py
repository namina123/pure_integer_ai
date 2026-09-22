"""将已完成 G-02 结构计划接入 A-02/M-02 WorkMemory。

This adapter is deliberately runtime-only.  It consumes typed generation
objects and occurrence anchors, never a surface string, and returns only
integer stable keys for persistence in the session trace.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.attractor_state import AttractorDependency
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructurePlan,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    OBJECT_VARIABLE,
    ObjectIdentity,
    SourceRef,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    SCOPE_EPISODE,
    SCOPE_QUERY,
    ScopeIdentity,
    document_scope,
    episode_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.situation_state import (
    CurrentSituationProjection,
    SituationEventLog,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.work_memory_content import (
    WorkMemoryContentProtocol,
    WorkMemoryOccurrenceAnchor,
    WorkMemoryRoleDefinition,
)
from pure_integer_ai.cognition.shared.work_memory_discourse import (
    WorkMemoryDiscourseProjection,
    WorkMemoryDiscourseRoles,
    project_generation_plans_to_work_memory,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_ADAPTER_VERSION = 1
_ROLE_NAMESPACE = 91590
_INSTRUCTION_NAMESPACE = 91591
_DEFAULT_SESSION_ID = 1
_SEMANTIC_KINDS = frozenset({
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT, OBJECT_PROPOSITION,
    OBJECT_SET_EXPR, OBJECT_VARIABLE,
})


@dataclass(frozen=True, slots=True)
class WorkMemoryGenerationProjection:
    """一个已验证投影及其可持久化整数摘要。"""

    projection: WorkMemoryDiscourseProjection
    situation: CurrentSituationProjection
    work_memory_state: tuple[int, ...]
    situation_state: tuple[int, ...]
    trace: tuple[int, ...]

    def integer_trace(self) -> dict[str, object]:
        """返回不含文本和 dataclass 的纯整数报告。"""
        return {
            "version": _ADAPTER_VERSION,
            "projection": list(self.projection.stable_key()),
            "work_memory_state": list(self.work_memory_state),
            "situation_state": list(self.situation_state),
            "trace": list(self.trace),
            "item_count": len(self.projection.items),
            "entry_count": len(self.situation.entries()),
        }


def _role(source: SourceRef, ordinal: int) -> ObjectIdentity:
    return role_identity(
        (_ROLE_NAMESPACE, ordinal), owner=source.owner, versions=source.versions)


def _instruction(source: SourceRef, ordinal: int) -> ObjectIdentity:
    return minimal_instruction_identity(
        (_INSTRUCTION_NAMESPACE, ordinal),
        owner=source.owner,
        versions=source.versions,
    )


def _scope_roles(source: SourceRef) -> WorkMemoryDiscourseRoles:
    return WorkMemoryDiscourseRoles(_role(source, 1), _role(source, 2), _role(source, 3))


def _allowed(values: tuple[ObjectIdentity, ...], fallback: int) -> tuple[int, ...]:
    kinds = {item.object_kind for item in values if item.object_kind in _SEMANTIC_KINDS}
    return tuple(sorted(kinds or {fallback}))


def _kind(value: ObjectIdentity) -> str:
    return {
        OBJECT_ENTITY: "ENTITY",
        OBJECT_EVENT: "EVENT",
        OBJECT_PROPOSITION: "PROPOSITION",
    }.get(value.object_kind, "UNRESOLVED_STATE")


def _close(work_memory: WorkMemory) -> None:
    """Close all opened A-09 scopes while retaining captured integer keys."""
    if work_memory.active_query_scope is not None:
        work_memory.end_query()
    if work_memory.active_episode_scope is not None:
        work_memory.end_episode()
    if work_memory.active_document_scope is not None:
        work_memory.end_document()
    if work_memory.active_session_scope is not None:
        work_memory.end_session()


def project_generation_to_work_memory(
        event_log,
        structure: GenerationStructurePlan,
        anchors: tuple[WorkMemoryOccurrenceAnchor, ...],
        *,
        logical_seq_start: int,
        query_local_id: int,
        trace: tuple[int, ...],
        session_local_id: int = _DEFAULT_SESSION_ID,
        ) -> WorkMemoryGenerationProjection:
    """在单个 query 内原子投影 G-02 context/question/proposition。

    The source and occurrence anchors are explicit.  Missing or mismatched
    anchors raise instead of silently manufacturing a WorkMemory value.
    """
    if not isinstance(structure, GenerationStructurePlan):
        raise TypeError("structure 必须是 GenerationStructurePlan")
    if not isinstance(anchors, tuple) or not anchors:
        raise ValueError("WorkMemory 投影必须提供 occurrence anchors")
    if type(logical_seq_start) is not int or logical_seq_start < 0:
        raise ValueError("logical_seq_start 必须是非负整数")
    if type(query_local_id) is not int or query_local_id <= 0:
        raise ValueError("query_local_id 必须是正整数")
    if type(session_local_id) is not int or session_local_id <= 0:
        raise ValueError("session_local_id 必须是正整数")
    if not isinstance(trace, tuple) or not trace:
        raise ValueError("WorkMemory 投影 trace 不能为空")
    assert_int(*trace, _where="project_generation_to_work_memory.trace")
    source = structure.selection.request.goal.source
    if any(item.source != source for item in anchors):
        raise ValueError("WorkMemory occurrence anchor source 与 generation source 不一致")
    roles = _scope_roles(source)
    context_values = tuple(structure.discourse.context)
    question_values = tuple(
        item.proposition.template for item in structure.discourse.open_questions)
    proposition_values = tuple(
        item.proposition.template for item in structure.propositions.propositions)
    protocol = WorkMemoryContentProtocol((
        WorkMemoryRoleDefinition(
            roles.context, _allowed(context_values, OBJECT_CONCEPT),
            SCOPE_DOCUMENT, max(1, len(context_values))),
        WorkMemoryRoleDefinition(
            roles.open_question, _allowed(question_values, OBJECT_PROPOSITION),
            SCOPE_QUERY, max(1, len(question_values))),
        WorkMemoryRoleDefinition(
            roles.selected_proposition, _allowed(proposition_values, OBJECT_PROPOSITION),
            SCOPE_EPISODE, max(1, len(proposition_values))),
    ), max(16, len(context_values) + len(question_values) + len(proposition_values) + 4))
    work_memory = WorkMemory()
    work_memory.configure_content(protocol)
    session = session_scope(
        session_local_id, owner=source.owner, versions=source.versions)
    document = document_scope(source, parent=session)
    episode = episode_scope(1, parent=document)
    query = query_scope(query_local_id, parent=episode)
    work_memory.begin_session(session)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode)
    work_memory.begin_query(query)
    try:
        projection = project_generation_plans_to_work_memory(
            work_memory,
            structure.discourse,
            structure.propositions,
            roles=roles,
            anchors=anchors,
            logical_seq_start=logical_seq_start,
            trace=trace,
        )
        if not projection.items:
            raise ValueError("G-02 结构计划没有可投影的 typed 内容")
        dependencies = {
            item.content_ref(): (
                AttractorDependency(
                    _instruction(source, index), item.value),
            )
            for index, item in enumerate(projection.items, 1)
        }
        item_roles = {item.role for item in projection.items}
        kind_by_role = {
            role: kind
            for role, kind in (
                (roles.context, _kind(context_values[0])
                 if context_values else "UNRESOLVED_STATE"),
                (roles.open_question, "OPEN_QUESTION"),
                (roles.selected_proposition, "PROPOSITION"),
            )
            if role in item_roles
        }
        situation = CurrentSituationProjection.from_work_memory_discourse(
            SituationEventLog(event_log, source, query),
            work_memory.require_content_store(),
            query,
            projection,
            kind_by_role=kind_by_role,
            dependencies_by_content_ref=dependencies,
        )
        result = WorkMemoryGenerationProjection(
            projection,
            situation,
            work_memory.content_state_key(),
            situation.state_key(),
            tuple(trace),
        )
    except BaseException:
        _close(work_memory)
        raise
    _close(work_memory)
    return result


def integer_trace_projection(value: object) -> object:
    """递归保留运行 trace 的整数结构，不携带内部文本标签。"""
    if type(value) is int:
        return value
    if type(value) is bool:
        return int(value)
    if isinstance(value, (tuple, list)):
        return [integer_trace_projection(item) for item in value
                if item is not None and not isinstance(item, str)]
    if isinstance(value, dict):
        return {
            str(key): integer_trace_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item is not None and not isinstance(item, str)
        }
    return None


def memory_structure_snapshot(memory) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """读取 Interaction Memory 当前全部 O/H/E 权威整数身份。"""
    active = memory.active_structures()
    return (
        tuple(sorted(item.observation_key for item in active.observations)),
        tuple(sorted(item.hypothesis_key for item in active.hypotheses)),
        tuple(sorted(item.evidence_key for item in active.evidence)),
    )


def query_work_memory_projection(bridge, query: dict[str, object], *,
                                 query_local_id: int) -> dict[str, object]:
    """把同次 QueryState 的真实 G-02 计划投影到 A-02 WorkMemory。

    开放回应使用本次输入 occurrence；闭合 Core 回应使用命题图中已验证的
    source anchor。函数只消费现役运行对象，不加载课程、token index 或表层词表。
    """
    structure = None
    anchors: tuple[WorkMemoryOccurrenceAnchor, ...] = ()
    pending = bridge._pending_response_delivery
    if pending is not None:
        _state, response = pending
        # Generic UNKNOWN delivery carries a ResponsePlan directly and has no
        # GenerationSurfacePreview/typed structure.  It is still a valid
        # graph-generated response; WorkMemory projection is only defined for
        # G-02 typed plans, so leave an explicit integer status instead of
        # dereferencing a non-existent preview and aborting the turn.
        preview = getattr(response, "preview", None)
        if preview is None:
            return {"version": 1, "status": 0, "reason": 3}
        structure = preview.request.structure
        occurrences = bridge._input_role_occurrences
        if occurrences:
            occurrence = occurrences[0]
            identity = occurrence.occurrence
            typed = bridge.memory.occurrence_index.ontology.resolve(identity)
            if typed is None:
                typed = bridge.memory.occurrence_index.ontology.materialize(identity)
            record = bridge.memory.occurrence_index.read(typed)
            anchors = (WorkMemoryOccurrenceAnchor(
                typed, identity, record.source, record.scope,
            ),)
    if structure is None and query.get("response_plan"):
        claims = query["response_plan"].get("claim_refs", ())
        if claims:
            proposition_key = tuple(claims[0])
            generation = bridge.generation_by_proposition.get(proposition_key)
            fact = bridge.core_fact_index.get(proposition_key)
            if generation is not None and fact is not None:
                structure = generation.structure_plan
                if structure is not None:
                    source = structure.selection.request.goal.source
                    source_anchor = bridge.core_runtime.generation_input(
                        fact.proposition).proposition.definition.source_anchor
                    if source_anchor.object_kind != OBJECT_OCCURRENCE:
                        raise ValueError("Core source_anchor 不是 Occurrence")
                    typed = bridge.memory.occurrence_index.ontology.resolve(source_anchor)
                    if typed is None:
                        typed = bridge.memory.occurrence_index.ontology.materialize(
                            source_anchor)
                    scope = document_scope(
                        source,
                        parent=session_scope(
                            1, owner=source.owner, versions=source.versions),
                    )
                    anchors = (WorkMemoryOccurrenceAnchor(
                        typed, source_anchor, source, scope,
                    ),)
    if structure is None or not anchors:
        return {"version": 1, "status": 0, "reason": 1}
    try:
        result = project_generation_to_work_memory(
            bridge.memory.intake.event_log,
            structure,
            anchors,
            logical_seq_start=query_local_id,
            query_local_id=query_local_id,
            trace=(91592, query_local_id),
        )
    except (TypeError, ValueError, RuntimeError):
        return {"version": 1, "status": 0, "reason": 2}
    return {"version": 1, "status": 1, **result.integer_trace()}


__all__ = [
    "WorkMemoryGenerationProjection",
    "integer_trace_projection",
    "memory_structure_snapshot",
    "project_generation_to_work_memory",
    "query_work_memory_projection",
]
