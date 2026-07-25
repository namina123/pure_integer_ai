"""把 L-03 指代 occurrence 与 L-05B2B 语义恢复接到 F-00。

本模块不解释代词词面，也不把旧图 ``ConceptRef`` 升格为 Entity。旧引用只用于
扩充 A-01 候选；真正的查询锚点必须是 A-01 当前唯一 adopted 的 typed Occurrence，
命题和 Evidence 继续由语义图与 H-00 只读恢复。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_OCCURRENCE,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
)
from pure_integer_ai.cognition.understanding.occurrence_reference import (
    OccurrenceReferenceEvidence,
    OccurrenceReferenceRequest,
    OccurrenceReferenceResolution,
    OccurrenceReferenceRevision,
    OccurrenceReferenceRuntime,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.language_semantic_course import (
    LanguageSemanticCourseInput,
)
from pure_integer_ai.experiments.language_semantic_query import (
    LanguageSemanticQueryRun,
    LanguageSemanticQueryRuntime,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _instruction(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验 route 和执行原因是注入的一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def _legacy_ref(
        value: tuple[int, int] | None, *, label: str,
        ) -> tuple[int, int] | None:
    """核验可选旧图引用只承担解析证据，不恢复任何对象语义。"""
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} 必须是二元节点引用")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value) or min(value) <= 0:
        raise ValueError(f"{label} 必须使用严格正整数")
    return value


@dataclass(frozen=True)
class ReferenceAnchorResolution:
    """保存 pronoun occurrence 到 typed antecedent 和语义恢复的完整结果。"""

    current: LanguageSemanticCourseInput
    reference_occurrence: TypedRef
    candidate_hint_refs: tuple[tuple[int, int], ...]
    reference_resolution: OccurrenceReferenceResolution
    antecedent_occurrence: TypedRef | None
    semantic: LanguageSemanticQueryRun | None
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验未解析与已解析两种形态，并保留同一来源和 runtime scope。"""
        if not isinstance(self.current, LanguageSemanticCourseInput):
            raise TypeError("reference resolution current 类型错误")
        if (not isinstance(self.reference_occurrence, TypedRef)
                or self.reference_occurrence.object_kind != OBJECT_OCCURRENCE):
            raise ValueError("reference occurrence 必须是 typed Occurrence")
        if (not isinstance(self.candidate_hint_refs, tuple)
                or any(_legacy_ref(
                    item, label="reference candidate hint ref") is None
                    for item in self.candidate_hint_refs)):
            raise TypeError("reference candidate hints 必须是 legacy ref tuple")
        if len(set(self.candidate_hint_refs)) != len(
                self.candidate_hint_refs):
            raise ValueError("reference candidate hints 不得重复")
        if not isinstance(
                self.reference_resolution, OccurrenceReferenceResolution):
            raise TypeError("reference resolution 必须来自 A-01 runtime")
        if (self.reference_resolution.request.reference
                != self.reference_occurrence):
            raise ValueError("A-01 resolution 替换了 reference occurrence")
        if self.antecedent_occurrence is None:
            if self.semantic is not None:
                raise ValueError("未定位 antecedent 时不得携带语义恢复")
            if self.reference_resolution.winner is not None:
                raise ValueError("A-01 已有唯一 winner 时不得丢弃 antecedent")
        else:
            if (not isinstance(self.antecedent_occurrence, TypedRef)
                    or self.antecedent_occurrence.object_kind
                    != OBJECT_OCCURRENCE):
                raise ValueError("antecedent 必须是 typed Occurrence")
            if not isinstance(self.semantic, LanguageSemanticQueryRun):
                raise TypeError("已定位 antecedent 时必须携带语义恢复")
            if (self.reference_resolution.winner is None
                    or self.reference_resolution.winner.antecedent
                    != self.antecedent_occurrence):
                raise ValueError("semantic antecedent 不是 A-01 唯一 winner")
            resolved = self.semantic.input_value.current
            if (resolved.source != self.current.source
                    or resolved.occurrence_scope
                    != self.current.occurrence_scope
                    or resolved.runtime_scope != self.current.runtime_scope
                    or resolved.occurrences
                    != (self.antecedent_occurrence,)):
                raise ValueError("reference 语义恢复替换了来源、scope 或 antecedent")
        if not isinstance(self.trace, tuple) or not self.trace:
            raise ValueError("reference resolution trace 必须是非空 tuple")
        assert_int(*self.trace, _where="reference resolution trace")
        if any(type(item) is not int for item in self.trace):
            raise ValueError("reference resolution trace 必须使用严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回输入 scope、解析证据、typed 锚点、语义请求和 trace 的稳定键。"""
        current_occurrence = self.reference_occurrence.stable_key()
        antecedent = (
            () if self.antecedent_occurrence is None
            else self.antecedent_occurrence.stable_key()
        )
        request = (
            () if self.semantic is None or self.semantic.request is None
            else self.semantic.request.stable_key()
        )
        return (
            1,
            *_packed(self.current.source.stable_key()),
            *_packed(self.current.occurrence_scope.stable_key()),
            *_packed(self.current.runtime_scope.stable_key()),
            *_packed(current_occurrence),
            len(self.candidate_hint_refs),
            *(value for item in self.candidate_hint_refs
              for value in _packed(item)),
            *_packed(self.reference_resolution.stable_key()),
            *_packed(antecedent),
            *_packed(request),
            *_packed(self.trace),
        )


class OccurrenceReferenceSemanticResolver:
    """以 legacy 提示扩候选，再消费 A-01 singleton adopted 恢复语义。"""

    def __init__(
            self,
            occurrence_index: OccurrenceIndex,
            semantic_runtime: LanguageSemanticQueryRuntime,
            reference_runtime: OccurrenceReferenceRuntime,
            antecedent_occurrences: tuple[TypedRef, ...],
            *,
            timestamp_seq: int,
            evidence: tuple[OccurrenceReferenceEvidence, ...] = (),
            revisions: tuple[OccurrenceReferenceRevision, ...] = (),
            commit_reference: bool = False,
            ) -> None:
        """绑定 A-01 owner、语义恢复 owner、有界窗口和调用方 Evidence。"""
        if not isinstance(occurrence_index, OccurrenceIndex):
            raise TypeError("reference resolver 需要 OccurrenceIndex")
        if not isinstance(semantic_runtime, LanguageSemanticQueryRuntime):
            raise TypeError("reference resolver 需要 LanguageSemanticQueryRuntime")
        if not isinstance(reference_runtime, OccurrenceReferenceRuntime):
            raise TypeError("reference resolver 需要 OccurrenceReferenceRuntime")
        if reference_runtime.occurrence_index is not occurrence_index:
            raise ValueError("reference resolver 必须复用同一 OccurrenceIndex owner")
        if (not isinstance(antecedent_occurrences, tuple)
                or any(not isinstance(item, TypedRef)
                       or item.object_kind != OBJECT_OCCURRENCE
                       for item in antecedent_occurrences)):
            raise TypeError("antecedent window 必须是 typed Occurrence tuple")
        if len(set(antecedent_occurrences)) != len(antecedent_occurrences):
            raise ValueError("antecedent window 不得重复 occurrence")
        assert_int(timestamp_seq, _where="reference resolver timestamp_seq")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("reference resolver timestamp_seq 不得为负")
        if (not isinstance(evidence, tuple)
                or any(not isinstance(item, OccurrenceReferenceEvidence)
                       for item in evidence)):
            raise TypeError("reference resolver evidence 类型错误")
        if (not isinstance(revisions, tuple)
                or any(not isinstance(item, OccurrenceReferenceRevision)
                       for item in revisions)):
            raise TypeError("reference resolver revisions 类型错误")
        if type(commit_reference) is not bool:
            raise TypeError("commit_reference 必须是 bool")
        self.occurrence_index = occurrence_index
        self.semantic_runtime = semantic_runtime
        self.reference_runtime = reference_runtime
        self.antecedent_occurrences = antecedent_occurrences
        self.timestamp_seq = timestamp_seq
        self.evidence = evidence
        self.revisions = revisions
        self.commit_reference = commit_reference

    def resolve(
            self, current: LanguageSemanticCourseInput,
            ) -> ReferenceAnchorResolution:
        """从 legacy 提示形成全量候选，仅消费 A-01 当前 singleton adopted。"""
        if not isinstance(current, LanguageSemanticCourseInput):
            raise TypeError("reference resolver 输入类型错误")
        if not current.read_only:
            raise ValueError("reference resolver 只接受 read-only 输入")
        if (len(current.occurrences) != 1
                or current.spans or current.active_senses):
            raise ValueError("reference resolver 需要唯一 occurrence 且无旁路 anchor")
        reference = current.occurrences[0]
        record = self.occurrence_index.read(reference)
        selected = tuple(sorted({
            item.legacy_ref
            for item in record.candidates
            if item.legacy_ref is not None
        }))
        matches = []
        for occurrence in self.antecedent_occurrences:
            candidate = self.occurrence_index.read(occurrence)
            legacy_candidates = {
                item.legacy_ref
                for item in candidate.candidates
                if item.legacy_ref is not None
            }
            if legacy_candidates.intersection(selected):
                matches.append(occurrence)
        reference_resolution = self.reference_runtime.resolve(
            OccurrenceReferenceRequest(
                reference,
                self.antecedent_occurrences,
                tuple(matches),
                current.runtime_scope,
                self.timestamp_seq,
                self.evidence,
                self.revisions,
            ),
            commit=self.commit_reference,
        )
        winner = reference_resolution.winner
        if winner is None:
            return ReferenceAnchorResolution(
                current,
                reference,
                selected,
                reference_resolution,
                None,
                None,
                (
                    2,
                    len(selected),
                    len(matches),
                    len(reference_resolution.adopted_candidates),
                ),
            )
        antecedent = winner.antecedent
        resolved_input = replace(
            current,
            occurrences=(antecedent,),
            spans=(),
            active_senses=(),
        )
        semantic = self.semantic_runtime.process(resolved_input)
        trace = (
            3,
            len(selected),
            *(value for item in selected for value in _packed(item)),
            *_packed(reference.stable_key()),
            *_packed(antecedent.stable_key()),
            *_packed(reference_resolution.decision.stable_key()),
            0 if semantic.request is None else 1,
        )
        return ReferenceAnchorResolution(
            current,
            reference,
            selected,
            reference_resolution,
            antecedent,
            semantic,
            trace,
        )


class SemanticReferenceQuestionExecutor:
    """执行 typed 指代恢复，并把匹配当前问题的语义候选投影到 F-00。"""

    def __init__(
            self,
            resolver: OccurrenceReferenceSemanticResolver,
            current: LanguageSemanticCourseInput,
            *,
            route: ObjectIdentity,
            executed_reason: ObjectIdentity,
            ) -> None:
        """绑定一次来源化 reference 输入、开放 route 和执行原因。"""
        if not isinstance(resolver, OccurrenceReferenceSemanticResolver):
            raise TypeError("reference question resolver 类型错误")
        if not isinstance(current, LanguageSemanticCourseInput):
            raise TypeError("reference question current 类型错误")
        self.resolver = resolver
        self.current = current
        self.route = _instruction(route, label="reference question route")
        self.executed_reason = _instruction(
            executed_reason,
            label="reference question executed reason",
        )

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """只采用与指代恢复 target、Evidence 方向、分支和 scope 全同的候选。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("reference question 需要 QuestionQuery")
        if query.route != self.route:
            raise ValueError("reference question 收到未注册 route")
        resolution = self.resolver.resolve(self.current)
        candidates = ()
        semantic_request = (
            None
            if resolution.semantic is None
            else resolution.semantic.request
        )
        if semantic_request is not None:
            goal = semantic_request.goal
            request = query.request
            if (goal.proposition == request.target
                    and goal.required == request.required
                    and goal.source == request.source
                    and goal.scope == request.response_scope
                    and goal.target_branch == request.target_branch):
                candidates = tuple(
                    item for item in semantic_request.candidates
                    if (item.proposition == request.target
                        and item.source == request.source
                        and item.scope == request.response_scope)
                )
        trace = (
            1,
            *_packed(query.stable_key()),
            *_packed(resolution.stable_key()),
            len(candidates),
            *(value for item in candidates
              for value in _packed(item.stable_key())),
        )
        return QuestionExecutionResult(
            query,
            self.executed_reason,
            candidates,
            trace,
        )


__all__ = [
    "OccurrenceReferenceSemanticResolver",
    "ReferenceAnchorResolution",
    "SemanticReferenceQuestionExecutor",
]
