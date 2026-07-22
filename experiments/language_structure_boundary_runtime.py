"""H-05 已采用结构预测到句界 Evidence 的防自证课程边界。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.understanding.boundary_hypothesis import (
    BoundaryCandidate,
    BoundaryEvidenceProfile,
    BoundaryEvidenceSpec,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.language_observation import (
    _boundary_language_key,
    _item_token_source_spans,
    _prepare_item_boundary,
)
from pure_integer_ai.experiments.language_structure_candidate_runtime import (
    StructureCandidateRecognitionTrace,
)


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验课程 proposal 和 provenance 使用的开放整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    """为变长稳定键加长度前缀，避免 provenance payload 拼接歧义。"""
    assert_int(*values, _where="structure_boundary.pack")
    return len(values), *values


@dataclass(frozen=True)
class StructureBoundaryEvidenceInput:
    """不暴露旧边界决定的完整文档结构预测和来源身份。"""

    text: str
    runtime_language: int
    language_key: tuple[int, ...]
    tokens: tuple[str, ...]
    observation: SourceRef
    scope: ScopeIdentity
    token_spans: tuple[tuple[int, int, int], ...]
    occurrences: tuple[ObjectIdentity, ...]
    structure_candidate: ObjectIdentity
    predicted: ObjectIdentity
    visible_inputs: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("structure-boundary 输入必须携带非空原文")
        assert_int(
            self.runtime_language,
            *self.language_key,
            _where="StructureBoundaryEvidenceInput",
        )
        if type(self.runtime_language) is not int or self.runtime_language <= 0:
            raise ValueError("runtime_language 必须为严格正整数")
        _strict_key(self.language_key, where="structure boundary language key")
        if (not isinstance(self.tokens, tuple) or not self.tokens
                or any(not isinstance(item, str) or not item
                       for item in self.tokens)):
            raise ValueError("structure-boundary tokens 必须是非空字符串 tuple")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("structure-boundary observation 类型非法")
        if (not isinstance(self.scope, ScopeIdentity)
                or self.scope.source != self.observation):
            raise ValueError("structure-boundary scope 必须指向 observation")
        if (not isinstance(self.token_spans, tuple)
                or len(self.token_spans) != len(self.tokens)
                or len(self.occurrences) != len(self.tokens)):
            raise ValueError("完整文档 token、span 和 occurrence 必须一一对应")
        if any(not isinstance(item, ObjectIdentity)
               or item.object_kind != OBJECT_OCCURRENCE
               for item in self.occurrences):
            raise TypeError("structure-boundary occurrences 类型非法")
        for name, value in (
                ("structure_candidate", self.structure_candidate),
                ("predicted", self.predicted)):
            if not isinstance(value, ObjectIdentity):
                raise TypeError(f"{name} 必须是 ObjectIdentity")
        if (not isinstance(self.visible_inputs, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.visible_inputs)):
            raise TypeError("visible_inputs 必须是 ObjectIdentity tuple")


@dataclass(frozen=True)
class StructureBoundaryEvidenceProposal:
    """课程依据结构 prediction 提出的边界三态 Evidence，不直接决定 winner。"""

    candidate: BoundaryCandidate
    stance: int
    reason_key: tuple[int, ...]
    payload: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, BoundaryCandidate):
            raise TypeError("boundary proposal candidate 类型非法")
        assert_int(self.stance, *self.payload, _where="boundary proposal")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("boundary proposal stance 未注册")
        _strict_key(self.reason_key, where="boundary proposal reason")
        if (not isinstance(self.payload, tuple)
                or any(type(item) is not int for item in self.payload)):
            raise ValueError("boundary proposal payload 必须是严格整数 tuple")


@runtime_checkable
class StructureBoundaryEvidenceMapper(Protocol):
    """课程把完整文档的已核验结构预测映射成零个或多个边界 proposal。"""

    def propose(
            self, input_value: StructureBoundaryEvidenceInput,
            ) -> tuple[StructureBoundaryEvidenceProposal, ...]:
        """只依据显式输入提出边界 Evidence，不读取候选图或旧边界决定。"""
        ...

    def clone_for_evaluation(self) -> "StructureBoundaryEvidenceMapper":
        """返回不共享可变课程状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple:
        """返回 mapper 的完整可比较状态。"""
        ...


@dataclass(frozen=True)
class StructureBoundaryIntegrationReport:
    """结构 trace eligibility、proposal 和实际边界 Evidence 写入计数。"""

    traces_seen: int = 0
    eligible_traces: int = 0
    non_support_blocked: int = 0
    inactive_blocked: int = 0
    read_only_blocked: int = 0
    partial_document_blocked: int = 0
    proposals: int = 0
    evidence_added: int = 0
    items_reparsed: int = 0


def _full_document_input(
        ctx, item: CollectedItem,
        trace: StructureCandidateRecognitionTrace,
        ) -> StructureBoundaryEvidenceInput | None:
    """只为与 item 完整 span/occurrence 序列一致的 trace 构造 mapper 输入。"""
    if item.raw_text is None or item.source_ref is None:
        return None
    token_spans = _item_token_source_spans(item)
    if token_spans is None or token_spans != trace.input_value.token_spans:
        return None
    occurrences = tuple(
        occurrence_identity(
            item.source_ref,
            start=start,
            end=end,
            ordinal=ordinal,
        )
        for start, end, ordinal in token_spans
    )
    if occurrences != trace.input_value.occurrences:
        return None
    return StructureBoundaryEvidenceInput(
        item.raw_text,
        item.lang,
        _boundary_language_key(ctx, item),
        tuple(item.tokens),
        item.source_ref,
        trace.input_value.scope,
        token_spans,
        occurrences,
        trace.mapped.candidate,
        trace.mapped.predicted,
        trace.mapped.visible_inputs,
    )


def _provenance_payload(
        trace: StructureCandidateRecognitionTrace,
        proposal: StructureBoundaryEvidenceProposal,
        ) -> tuple[int, ...]:
    """保存 structure Hypothesis、Evidence、decision、投影 Event 和 occurrence 来源链。"""
    outcome = trace.outcome
    if (outcome is None or trace.projection is None
            or not trace.projection.history):
        raise RuntimeError("eligible structure trace 缺少写链或投影 Event")
    hypothesis = trace.projection.candidate.hypothesis.stable_key()
    event = trace.projection.history[-1].definition.event.stable_key()
    occurrences = trace.input_value.occurrences
    return (
        *_pack(hypothesis),
        outcome.evidence.evidence_id,
        outcome.decision.decision_id,
        *_pack(event),
        len(occurrences),
        *(
            value
            for occurrence in occurrences
            for value in _pack(occurrence.stable_key())
        ),
        *_pack(proposal.payload),
    )


def _same_evidence_event(
        existing: BoundaryEvidenceSpec, *,
        candidate: BoundaryCandidate, stance: int,
        reason_key: tuple[int, ...], source: SourceRef,
        payload: tuple[int, ...],
        ) -> bool:
    """按完整来源和 provenance 识别同一边界事件，逻辑序不参与幂等身份。"""
    return (
        existing.candidate == candidate
        and existing.stance == stance
        and existing.reason_key == reason_key
        and existing.source == source
        and existing.payload == payload
    )


def apply_structure_boundary_evidence(
        ctx, corpus: list[CollectedItem],
        traces: tuple[StructureCandidateRecognitionTrace, ...],
        mapper: StructureBoundaryEvidenceMapper,
        ) -> StructureBoundaryIntegrationReport:
    """筛选 committed structure support，合并边界 Evidence 并重跑受影响训练 item。"""
    if not isinstance(mapper, StructureBoundaryEvidenceMapper):
        raise TypeError("mapper 必须实现 StructureBoundaryEvidenceMapper")
    items_by_source = {
        item.source_ref: item for item in corpus if item.source_ref is not None
    }
    if len(items_by_source) != sum(
            item.source_ref is not None for item in corpus):
        raise ValueError("structure-boundary corpus 存在重复 SourceRef")

    non_support = 0
    inactive = 0
    read_only = 0
    partial = 0
    eligible = 0
    proposal_count = 0
    evidence_added = 0
    changed_items: dict[SourceRef, CollectedItem] = {}
    pending_evidence: dict[SourceRef, list[BoundaryEvidenceSpec]] = {}
    next_timestamp_by_source: dict[SourceRef, int] = {}
    for trace in traces:
        if trace.outcome is None or trace.read_only:
            read_only += 1
            continue
        if trace.outcome.verification.stance != EVIDENCE_SUPPORT:
            non_support += 1
            continue
        if trace.projection is None:
            inactive += 1
            continue
        hypothesis = trace.projection.candidate.hypothesis
        active_state = ctx.candidate_projection_graph.protocol.active_state
        if (not trace.adopted
                or trace.projection.state != active_state
                or hypothesis not in trace.outcome.decision.adopted_hypotheses):
            inactive += 1
            continue
        item = items_by_source.get(trace.input_value.observation)
        if item is None:
            raise ValueError("structure trace 的 observation 不在当前训练 corpus")
        input_value = _full_document_input(ctx, item, trace)
        if input_value is None:
            partial += 1
            continue
        eligible += 1
        proposals = mapper.propose(input_value)
        if not isinstance(proposals, tuple) or any(
                not isinstance(value, StructureBoundaryEvidenceProposal)
                for value in proposals):
            raise TypeError("structure-boundary mapper 返回了非法 proposal")
        proposal_count += len(proposals)
        if not proposals:
            continue
        merged = pending_evidence.get(item.source_ref)
        if merged is None:
            existing = (
                () if item.boundary_profile is None
                else item.boundary_profile.evidence)
            merged = list(existing)
        for proposal in proposals:
            payload = _provenance_payload(trace, proposal)
            source = trace.outcome.verification.source
            if any(_same_evidence_event(
                    existing,
                    candidate=proposal.candidate,
                    stance=proposal.stance,
                    reason_key=proposal.reason_key,
                    source=source,
                    payload=payload,
                    ) for existing in merged):
                continue
            timestamp_seq = next_timestamp_by_source.get(item.source_ref)
            if timestamp_seq is None:
                timestamp_seq = ctx.boundary_hypothesis_engine.next_timestamp(
                    item.boundary_parse)
            evidence = BoundaryEvidenceSpec(
                proposal.candidate,
                proposal.stance,
                proposal.reason_key,
                timestamp_seq=timestamp_seq,
                source=source,
                payload=payload,
            )
            merged.append(evidence)
            evidence_added += 1
            changed_items[item.source_ref] = item
            pending_evidence[item.source_ref] = merged
            next_timestamp_by_source[item.source_ref] = timestamp_seq + 1

    for source, item in changed_items.items():
        item.boundary_profile = BoundaryEvidenceProfile(
            tuple(pending_evidence[source]))
    for source in sorted(changed_items, key=SourceRef.stable_key):
        _prepare_item_boundary(
            ctx,
            changed_items[source],
            commit_evidence=True,
            persist_graph=True,
        )
    return StructureBoundaryIntegrationReport(
        traces_seen=len(traces),
        eligible_traces=eligible,
        non_support_blocked=non_support,
        inactive_blocked=inactive,
        read_only_blocked=read_only,
        partial_document_blocked=partial,
        proposals=proposal_count,
        evidence_added=evidence_added,
        items_reparsed=len(changed_items),
    )


__all__ = [
    "StructureBoundaryEvidenceInput",
    "StructureBoundaryEvidenceMapper",
    "StructureBoundaryEvidenceProposal",
    "StructureBoundaryIntegrationReport",
    "apply_structure_boundary_evidence",
]
