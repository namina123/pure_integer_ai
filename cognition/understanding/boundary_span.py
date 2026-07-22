"""把句界 Hypothesis 物化为来源化 Span、StructureConcept 和选择断言。

候选根 Span 覆盖完整来源，零宽子 Span 表示内部边界锚点。消费者只读取唯一 active
选择断言；纠正通过 assertion supersede 追加历史，不删除旧候选或旧结构。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphStatement,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    HypothesisKey,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_HYPOTHESIS,
    TypedRef,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalClockIdentity,
)
from pure_integer_ai.cognition.understanding.boundary_hypothesis import (
    BoundaryDecision,
    BoundaryHypothesisProtocol,
    BoundaryResult,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.node_store import TIER_SHADOW

def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验结构和关系使用的开放整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value

@dataclass(frozen=True)
class BoundarySpanProtocol:
    """注入句界候选所需的一等结构、选择关系和逻辑时钟。"""

    hypothesis_protocol: BoundaryHypothesisProtocol
    span_protocol: SpanProtocol
    document_structure_key: tuple[int, ...]
    candidate_structure_key: tuple[int, ...]
    anchor_structure_key: tuple[int, ...]
    candidate_shape_namespace_key: tuple[int, ...]
    selection_relation_key: tuple[int, ...]
    withdrawal_relation_key: tuple[int, ...]
    selection_clock_kind: int

    def __post_init__(self) -> None:
        if not isinstance(
                self.hypothesis_protocol, BoundaryHypothesisProtocol):
            raise TypeError("hypothesis_protocol 类型非法")
        if not isinstance(self.span_protocol, SpanProtocol):
            raise TypeError("span_protocol 类型非法")
        keys = tuple(
            _strict_key(value, where=f"BoundarySpanProtocol.{label}")
            for label, value in (
                ("document_structure_key", self.document_structure_key),
                ("candidate_structure_key", self.candidate_structure_key),
                ("anchor_structure_key", self.anchor_structure_key),
                ("candidate_shape_namespace_key",
                 self.candidate_shape_namespace_key),
                ("selection_relation_key", self.selection_relation_key),
                ("withdrawal_relation_key", self.withdrawal_relation_key),
            )
        )
        if len(set(keys)) != len(keys):
            raise ValueError("句界结构、形状命名空间和选择关系键必须互不相同")
        if {self.selection_relation_key, self.withdrawal_relation_key} & {
                self.span_protocol.structure_relation_key,
                self.span_protocol.constituent_relation_key,
                self.span_protocol.occurrence_relation_key,
                self.span_protocol.candidate_relation_key,
                }:
            raise ValueError("句界选择和撤销关系不得复用 Span 基础关系")
        assert_int(
            self.selection_clock_kind,
            _where="BoundarySpanProtocol.selection_clock_kind",
        )
        if type(self.selection_clock_kind) is not int \
                or self.selection_clock_kind <= 0:
            raise ValueError("句界选择时钟 kind 必须为严格正整数")


@dataclass(frozen=True)
class BoundarySpanCandidate:
    """一个边界 Hypothesis 对应的候选根 Span 和零宽锚点。"""

    hypothesis: HypothesisKey
    root: TypedRef
    anchors: tuple[TypedRef, ...]
    candidate_statement: GraphStatement


@dataclass(frozen=True)
class BoundarySpanResult:
    """一次句界物化的候选、当前决定和新增图引用汇总。"""

    decision: BoundaryDecision
    document: TypedRef | None
    candidates: tuple[BoundarySpanCandidate, ...]
    selected_statement: GraphStatement | None
    span_refs: tuple[TypedRef, ...]
    statement_hashes: tuple[int, ...]


class BoundarySpanMaterializer:
    """维护句界 Span 候选、active 选择和 winner occurrence 成员。"""

    def __init__(self, spans: SpanIndex,
                 protocol: BoundarySpanProtocol) -> None:
        if not isinstance(spans, SpanIndex):
            raise TypeError("spans 必须是 SpanIndex")
        if not isinstance(protocol, BoundarySpanProtocol):
            raise TypeError("protocol 必须是 BoundarySpanProtocol")
        if spans.protocol != protocol.span_protocol:
            raise ValueError("SpanIndex 与 BoundarySpanProtocol 协议不一致")
        self.spans = spans
        self.protocol = protocol
        self._predicate: TypedRef | None = None
        self._withdrawal_predicate_ref: TypedRef | None = None
        self._candidates: dict[HypothesisKey, BoundarySpanCandidate] = {}

    def materialize(
            self, result: BoundaryResult, *,
            token_spans: tuple[tuple[int, int, int], ...],
            occurrence_refs: tuple[TypedRef, ...] = (),
            replace_selection: bool = False,
            ) -> BoundarySpanResult:
        """物化候选并消费唯一选择；无候选和历史决定时保持零写。"""
        if not isinstance(result, BoundaryResult):
            raise TypeError("result 必须是 BoundaryResult")
        if not isinstance(occurrence_refs, tuple):
            raise TypeError("occurrence_refs 必须是 TypedRef tuple")
        if type(replace_selection) is not bool:
            raise TypeError("replace_selection 必须是 bool")
        preview_decision = result.decision()
        preview_decision.token_cuts(token_spans)

        document = self._resolve_document(result)
        if not result.candidates and document is None:
            return BoundarySpanResult(
                preview_decision, None, (), None, (), ())
        if document is None:
            document = self._ensure_document(result)

        span_refs: list[TypedRef] = [document]
        statement_hashes: list[int] = []
        materialized: list[BoundarySpanCandidate] = []
        for snapshot in sorted(
                result.candidates,
                key=lambda item: item.hypothesis.stable_key()):
            candidate = self._materialize_candidate(
                result,
                snapshot.hypothesis,
                snapshot.candidate.anchors,
                statement_hashes,
            )
            materialized.append(candidate)
            span_refs.append(candidate.root)
            span_refs.extend(candidate.anchors)

        if result.selected_hypothesis is not None:
            selected_candidate = self._candidates.get(
                result.selected_hypothesis)
            if selected_candidate is None:
                raise ValueError("已决边界候选尚未物化")
            replace_candidate = replace_selection
            old_statement = self._active_selection(result, document)
            if (old_statement is not None
                    and old_statement.object != selected_candidate.root):
                old_hypothesis = self._hypothesis_for_candidate(
                    old_statement.object)
                old_snapshot = self._snapshot_for(result, old_hypothesis)
                replace_candidate = (
                    replace_candidate
                    or old_snapshot.snapshot.lifecycle != LIFECYCLE_ACTIVE)
            self._select(
                result,
                document,
                selected_candidate,
                statement_hashes,
                replace_existing=(replace_selection or bool(result.candidates)),
                replace_candidate=replace_candidate,
            )
        elif result.candidates:
            old_statement = self._active_selection(result, document)
            if old_statement is not None:
                self._withdraw_selection(
                    result,
                    old_statement,
                    statement_hashes,
                )

        selected_statement = self._active_selection(result, document)
        if selected_statement is None:
            return BoundarySpanResult(
                preview_decision,
                document,
                tuple(materialized),
                None,
                tuple(dict.fromkeys(span_refs)),
                tuple(sorted(set(statement_hashes))),
            )

        selected_hypothesis = self._hypothesis_for_candidate(
            selected_statement.object)
        selected_anchors = self._anchors_for_candidate(
            selected_statement.object)
        decision = BoundaryDecision(
            result.text,
            result.observation,
            result.scope,
            result.language_key,
            selected_hypothesis,
            selected_anchors,
        )
        decision.token_cuts(token_spans)
        span_refs.append(selected_statement.object)
        span_refs.extend(self._anchor_refs_for_candidate(
            selected_statement.object))
        self._link_occurrences(
            selected_statement.object,
            result,
            token_spans,
            occurrence_refs,
            statement_hashes,
        )
        return BoundarySpanResult(
            decision,
            document,
            tuple(materialized),
            selected_statement,
            tuple(dict.fromkeys(span_refs)),
            tuple(sorted(set(statement_hashes))),
        )

    def clone_for_context(
            self, spans: SpanIndex) -> "BoundarySpanMaterializer":
        """在评测 clone 的 SpanIndex 上重建独立物化器。"""
        return BoundarySpanMaterializer(spans, self.protocol)

    def supersede_selected(
            self, result: BoundaryResult, *,
            token_spans: tuple[tuple[int, int, int], ...],
            occurrence_refs: tuple[TypedRef, ...] = (),
            ) -> BoundarySpanResult:
        """显式应用已验证反馈，把当前选择替代为 result 的唯一 winner。"""
        if result.selected_hypothesis is None:
            raise ValueError("显式句界替代必须提供唯一已决 replacement")
        return self.materialize(
            result,
            token_spans=token_spans,
            occurrence_refs=occurrence_refs,
            replace_selection=True,
        )

    def _ensure_document(self, result: BoundaryResult) -> TypedRef:
        """物化本协议专用的来源文档根 Span。"""
        document_role = structure_concept_identity(
            self.protocol.document_structure_key)
        ordinal = self.spans.ensure_role_ordinal(document_role)
        span = self.spans.ensure_ref(
            source=result.observation,
            raw_text=result.text,
            scope=result.scope,
            members=((0, len(result.text)),),
            ordinal=ordinal,
            structures=(document_role,),
        )
        return span

    def _resolve_document(self, result: BoundaryResult) -> TypedRef | None:
        """只读解析已存在的本协议文档根，不因查询产生新图对象。"""
        ordinal = self.spans.resolve_role_ordinal(
            structure_concept_identity(self.protocol.document_structure_key))
        if ordinal is None:
            return None
        return self.spans.ontology.resolve(span_identity(
            result.observation,
            members=((0, len(result.text)),),
            ordinal=ordinal,
        ))

    def _materialize_candidate(
            self, result: BoundaryResult, hypothesis: HypothesisKey,
            anchors: tuple[int, ...],
            statement_hashes: list[int],
            ) -> BoundarySpanCandidate:
        """物化一个候选根、共享锚点和 Hypothesis link。"""
        cached = self._candidates.get(hypothesis)
        if cached is not None:
            return cached
        root_ordinal = self.spans.ensure_role_ordinal(
            hypothesis.object_identity())
        shape_key = (
            len(self.protocol.candidate_shape_namespace_key),
            *self.protocol.candidate_shape_namespace_key,
            len(anchors),
        )
        root = self.spans.ensure_ref(
            source=result.observation,
            raw_text=result.text,
            scope=result.scope,
            members=((0, len(result.text)),),
            ordinal=root_ordinal,
            structures=(
                structure_concept_identity(
                    self.protocol.candidate_structure_key),
                structure_concept_identity(shape_key),
            ),
        )
        anchor_refs: list[TypedRef] = []
        for member_ordinal, anchor in enumerate(anchors):
            anchor_role = structure_concept_identity(
                self.protocol.anchor_structure_key)
            anchor_ordinal = self.spans.ensure_role_ordinal(anchor_role)
            anchor = self.spans.ensure_ref(
                source=result.observation,
                raw_text=result.text,
                scope=result.scope,
                members=((anchor, anchor),),
                ordinal=anchor_ordinal,
                structures=(anchor_role,),
            )
            anchor_refs.append(anchor)
            constituent = self.spans.add_constituent(
                root,
                anchor,
                member_ordinal=member_ordinal,
            )
            statement_hashes.append(constituent.assertion_hash)
        candidate_statement = self.spans.add_candidate(
            hypothesis,
            root,
        )
        statement_hashes.append(candidate_statement.assertion_hash)
        candidate = BoundarySpanCandidate(
            hypothesis,
            root,
            tuple(anchor_refs),
            candidate_statement,
        )
        self._candidates[hypothesis] = candidate
        return candidate

    def _select(
            self, result: BoundaryResult, document: TypedRef,
            new: BoundarySpanCandidate,
            statement_hashes: list[int],
            *, replace_existing: bool, replace_candidate: bool,
            ) -> None:
        """追加或 supersede 唯一句界选择，并同步候选生命周期断言。"""
        old_statement = self._active_selection(result, document)
        if old_statement is not None and old_statement.object == new.root:
            return
        if old_statement is None:
            selected = self._selection_statement(result, document, new.root)
            statement_hashes.append(selected.assertion_hash)
            return
        if not replace_existing:
            return

        old_hypothesis = self._hypothesis_for_candidate(old_statement.object)
        if (
                old_hypothesis.hypothesis_kind
                != new.hypothesis.hypothesis_kind
                or old_hypothesis.competition_key
                != new.hypothesis.competition_key
                or old_hypothesis.scope != new.hypothesis.scope
                or old_hypothesis.observation != new.hypothesis.observation):
            raise ValueError("句界 replacement 不属于同一竞争组")
        clock = self.spans.scoped_identities.resume_clock(
            LogicalClockIdentity(
                result.scope,
                self.protocol.selection_clock_kind,
            ))
        timestamp = clock.advance()
        if replace_candidate:
            self.spans.validate_candidate_supersede(
                old_hypothesis,
                new.hypothesis,
                timestamp,
            )
        new_statement = self._selection_statement(
            result,
            document,
            new.root,
        )
        self.spans.scoped_identities.supersede(
            old_statement.assertion,
            new_statement.assertion,
            timestamp,
        )
        if replace_candidate:
            self.spans.supersede_candidate(
                old_hypothesis,
                new.hypothesis,
                timestamp,
            )
        statement_hashes.append(new_statement.assertion_hash)

    def _withdraw_selection(
            self, result: BoundaryResult, old_statement: GraphStatement,
            statement_hashes: list[int],
            ) -> None:
        """用图内 tombstone 撤销陈旧选择，并按 H-00 生命周期决定是否退出候选 link。"""
        old_hypothesis = self._hypothesis_for_candidate(old_statement.object)
        old_snapshot = self._snapshot_for(result, old_hypothesis)
        candidate = self._candidates.get(old_hypothesis)
        if candidate is None:
            raise ValueError("旧边界选择未出现在当前完整竞争快照中")
        clock = self.spans.scoped_identities.resume_clock(
            LogicalClockIdentity(
                result.scope,
                self.protocol.selection_clock_kind,
            ))
        timestamp = clock.advance()
        candidate_link = candidate.candidate_statement
        withdraw_candidate = (
            old_snapshot.snapshot.lifecycle != LIFECYCLE_ACTIVE)
        if (withdraw_candidate
                and self.spans.scoped_identities.assertion_is_superseded(
                    candidate_link.assertion_hash)):
            raise ValueError("已退出的边界 candidate link 仍被 active selection 引用")
        withdrawal = self.spans.ontology.relate(
            self._withdrawal_predicate(),
            candidate_link.subject,
            candidate.root,
            scope=result.scope,
            provenance_kind=result.observation.source_kind,
            content_version=result.observation.versions.parser.value,
            qualifiers=(len(result.language_key), *result.language_key),
        )
        self.spans.scoped_identities.supersede(
            old_statement.assertion,
            withdrawal.assertion,
            timestamp,
        )
        if withdraw_candidate:
            self.spans.scoped_identities.supersede(
                candidate_link.assertion,
                withdrawal.assertion,
                timestamp,
            )
        statement_hashes.append(withdrawal.assertion_hash)

    def _selection_statement(
            self, result: BoundaryResult, document: TypedRef,
            candidate: TypedRef,
            ) -> GraphStatement:
        """追加来源化选择断言，语言竞争键放入限定项。"""
        return self.spans.ontology.relate(
            self._selection_predicate(),
            document,
            candidate,
            scope=result.scope,
            provenance_kind=result.observation.source_kind,
            content_version=result.observation.versions.parser.value,
            qualifiers=(len(result.language_key), *result.language_key),
        )

    def _active_selection(
            self, result: BoundaryResult, document: TypedRef,
            ) -> GraphStatement | None:
        """读取当前语言竞争键下唯一未被替代的选择断言。"""
        qualifiers = (len(result.language_key), *result.language_key)
        statements = tuple(
            statement for statement in self.spans.ontology.statements(
                predicate=self._selection_predicate(),
                subject=document,
            )
            if (statement.assertion.qualifiers == qualifiers
                and not self.spans.scoped_identities.assertion_is_superseded(
                    statement.assertion_hash))
        )
        if len(statements) > 1:
            raise ValueError("同一来源和语言竞争键存在多个 active 句界选择")
        return None if not statements else statements[0]

    def _selection_predicate(self) -> TypedRef:
        """幂等物化调用方注入的句界选择 predicate。"""
        if self._predicate is None:
            self._predicate = self.spans.ontology.materialize(
                relation_concept_identity(
                    self.protocol.selection_relation_key),
                tier=TIER_SHADOW,
            )
        return self._predicate

    def _withdrawal_predicate(self) -> TypedRef:
        """幂等物化调用方注入的句界选择撤销 predicate。"""
        if self._withdrawal_predicate_ref is None:
            self._withdrawal_predicate_ref = self.spans.ontology.materialize(
                relation_concept_identity(
                    self.protocol.withdrawal_relation_key),
                tier=TIER_SHADOW,
            )
        return self._withdrawal_predicate_ref

    @staticmethod
    def _snapshot_for(result: BoundaryResult, hypothesis: HypothesisKey):
        """从完整竞争快照读取指定候选，缺失时拒绝按旧图猜测。"""
        matches = tuple(
            item for item in result.candidates
            if item.hypothesis == hypothesis)
        if len(matches) != 1:
            raise ValueError("当前边界结果未完整覆盖旧选择所在竞争组")
        return matches[0]

    def _hypothesis_for_candidate(self, candidate: TypedRef) -> HypothesisKey:
        """从候选 Span 的一等 Hypothesis link 恢复完整候选身份。"""
        links = self.spans.read(candidate).candidate_links
        if len(links) != 1:
            raise ValueError("句界候选 Span 没有唯一 Hypothesis link")
        identity = self.spans.ontology.identity_of(links[0].subject)
        if identity.object_kind != OBJECT_HYPOTHESIS:
            raise ValueError("句界候选 link 主体不是 Hypothesis")
        hypothesis = HypothesisKey.from_stable_key(identity.components)
        if (
                hypothesis.object_identity().owner != identity.owner
                or hypothesis.object_identity().versions != identity.versions):
            raise ValueError("句界 Hypothesis 图身份 owner/version 不一致")
        return hypothesis

    def _anchor_refs_for_candidate(
            self, candidate: TypedRef) -> tuple[TypedRef, ...]:
        """按 member ordinal 返回候选根的零宽锚点 Span。"""
        record = self.spans.read(candidate)
        anchors = tuple(statement.object for statement in record.constituents)
        for anchor in anchors:
            members = self.spans.read(anchor).members
            if len(members) != 1 or members[0][0] != members[0][1]:
                raise ValueError("句界候选 constituent 不是零宽锚点 Span")
        return anchors

    def _anchors_for_candidate(self, candidate: TypedRef) -> tuple[int, ...]:
        """从候选根递归关系恢复有序内部码点锚点。"""
        return tuple(
            self.spans.read(anchor).members[0][0]
            for anchor in self._anchor_refs_for_candidate(candidate)
        )

    def _link_occurrences(
            self, candidate: TypedRef, result: BoundaryResult,
            token_spans: tuple[tuple[int, int, int], ...],
            occurrence_refs: tuple[TypedRef, ...],
            statement_hashes: list[int],
            ) -> None:
        """把当前 winner occurrence 序绑定到已决候选根 Span。"""
        if not occurrence_refs:
            return
        if self.spans.occurrence_index is None:
            raise RuntimeError("句界 occurrence 锚定前必须装配 OccurrenceIndex")
        if len(occurrence_refs) != len(token_spans):
            raise ValueError("句界 token span 与 occurrence 数量不一致")
        for member_ordinal, (occurrence, token_span) in enumerate(zip(
                occurrence_refs, token_spans)):
            if not isinstance(occurrence, TypedRef):
                raise TypeError("occurrence_refs 只能包含 TypedRef")
            occurrence_record = self.spans.occurrence_index.read(occurrence)
            start, end, ordinal = token_span
            if (
                    occurrence_record.source != result.observation
                    or occurrence_record.scope != result.scope
                    or occurrence_record.start != start
                    or occurrence_record.end != end
                    or occurrence_record.ordinal != ordinal):
                raise ValueError("句界 occurrence 与 winner token span 不一致")
            statement = self.spans.add_occurrence(
                candidate,
                occurrence,
                member_ordinal=member_ordinal,
            )
            statement_hashes.append(statement.assertion_hash)


__all__ = [
    "BoundarySpanCandidate",
    "BoundarySpanMaterializer",
    "BoundarySpanProtocol",
    "BoundarySpanResult",
]
