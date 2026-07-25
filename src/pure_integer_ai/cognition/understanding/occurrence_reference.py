"""A-01 occurrence 级指代候选、Evidence 和后文回溯。

本模块只表达一次 reference Occurrence 与 antecedent Occurrence 的竞争关系。实体、
事件、篇章中心等有界内容状态属于 A-02；旧 ``ConceptRef`` 和位置近因只能在外层
扩充候选，不能进入本模块的采用判定。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
    ReplacementDirective,
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_OCCURRENCE,
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceRecord,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_REFERENCE_VERSION = 1
_EVIDENCE_HASHER = Hasher("occurrence_reference.evidence.v1")
_EVIDENCE_STANCES = frozenset({
    EVIDENCE_SUPPORT,
    EVIDENCE_REFUTE,
    EVIDENCE_UNKNOWN,
})


def _strict_key(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """核验开放协议键只包含严格整数，并按调用点决定是否允许为空。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长稳定键增加长度边界。"""
    return len(value), *value


def _occurrence(value: TypedRef, *, label: str) -> TypedRef:
    """核验运行期端点是一等 Occurrence 引用。"""
    if not isinstance(value, TypedRef):
        raise TypeError(f"{label} 必须是 TypedRef")
    if value.object_kind != OBJECT_OCCURRENCE:
        raise ValueError(f"{label} 必须是 Occurrence")
    return value


def _identity(
        value: ObjectIdentity, *, label: str, object_kind: int,
        ) -> ObjectIdentity:
    """核验协议注入的一等对象类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != object_kind:
        raise ValueError(f"{label} 对象类型错误")
    return value


def _candidate_key(
        reference: ObjectIdentity,
        antecedent: ObjectIdentity,
        ) -> tuple[int, ...]:
    """构造同时保留 reference 与 antecedent 完整身份的候选键。"""
    return (
        _REFERENCE_VERSION,
        *_pack(reference.stable_key()),
        *_pack(antecedent.stable_key()),
    )


def _competition_key(reference: ObjectIdentity) -> tuple[int, ...]:
    """构造同一次 reference 的 H-00 竞争组键。"""
    return _REFERENCE_VERSION, *_pack(reference.stable_key())


@dataclass(frozen=True)
class OccurrenceReferenceProtocol:
    """冻结开放候选 kind、形成语义和当前上下文预算。"""

    hypothesis_kind_key: tuple[int, ...]
    formation_dimension: ObjectIdentity
    formation_reason: ObjectIdentity
    max_preceding_distance: int
    max_following_distance: int
    max_context_occurrences: int
    max_candidates: int

    def __post_init__(self) -> None:
        _strict_key(
            self.hypothesis_kind_key,
            label="OccurrenceReferenceProtocol.hypothesis_kind_key",
        )
        _identity(
            self.formation_dimension,
            label="OccurrenceReferenceProtocol.formation_dimension",
            object_kind=OBJECT_CONCEPT,
        )
        _identity(
            self.formation_reason,
            label="OccurrenceReferenceProtocol.formation_reason",
            object_kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        assert_int(
            self.max_preceding_distance,
            self.max_following_distance,
            self.max_context_occurrences,
            self.max_candidates,
            _where="OccurrenceReferenceProtocol",
        )
        if (type(self.max_preceding_distance) is not int
                or self.max_preceding_distance <= 0):
            raise ValueError("max_preceding_distance 必须为严格正整数")
        if (type(self.max_following_distance) is not int
                or self.max_following_distance < 0):
            raise ValueError("max_following_distance 必须为非负严格整数")
        if (type(self.max_context_occurrences) is not int
                or self.max_context_occurrences <= 0):
            raise ValueError("max_context_occurrences 必须为严格正整数")
        if (type(self.max_candidates) is not int
                or self.max_candidates <= 0
                or self.max_candidates > self.max_context_occurrences):
            raise ValueError("max_candidates 必须位于当前上下文预算内")

    def stable_key(self) -> tuple[int, ...]:
        """返回协议语义和全部注入预算的完整稳定键。"""
        return (
            _REFERENCE_VERSION,
            *_pack(self.hypothesis_kind_key),
            *_pack(self.formation_dimension.stable_key()),
            *_pack(self.formation_reason.stable_key()),
            self.max_preceding_distance,
            self.max_following_distance,
            self.max_context_occurrences,
            self.max_candidates,
        )


@dataclass(frozen=True)
class OccurrenceReferenceEvidence:
    """调用方针对一个 antecedent 提交的来源化上下文 Evidence。"""

    antecedent: TypedRef
    dimension: ObjectIdentity
    stance: int
    reason: ObjectIdentity
    source: SourceRef
    timestamp_seq: int
    visible_occurrences: tuple[TypedRef, ...] = ()
    visible_objects: tuple[ObjectIdentity, ...] = ()
    trace: tuple[int, ...] = ()
    supersedes_evidence_id: int = 0

    def __post_init__(self) -> None:
        _occurrence(self.antecedent, label="reference Evidence antecedent")
        _identity(
            self.dimension,
            label="reference Evidence dimension",
            object_kind=OBJECT_CONCEPT,
        )
        _identity(
            self.reason,
            label="reference Evidence reason",
            object_kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        if not isinstance(self.source, SourceRef):
            raise TypeError("reference Evidence source 必须是 SourceRef")
        assert_int(
            self.stance,
            self.timestamp_seq,
            self.supersedes_evidence_id,
            _where="OccurrenceReferenceEvidence",
        )
        if type(self.stance) is not int or self.stance not in _EVIDENCE_STANCES:
            raise ValueError("reference Evidence stance 未注册")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise ValueError("reference Evidence 逻辑序不得为负")
        if (type(self.supersedes_evidence_id) is not int
                or self.supersedes_evidence_id < 0):
            raise ValueError("reference Evidence supersedes id 不得为负")
        if (not isinstance(self.visible_occurrences, tuple)
                or any(not isinstance(item, TypedRef)
                       or item.object_kind != OBJECT_OCCURRENCE
                       for item in self.visible_occurrences)):
            raise TypeError("visible_occurrences 必须是 Occurrence tuple")
        if len(set(self.visible_occurrences)) != len(
                self.visible_occurrences):
            raise ValueError("visible_occurrences 不得重复")
        if (not isinstance(self.visible_objects, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.visible_objects)):
            raise TypeError("visible_objects 必须是 ObjectIdentity tuple")
        if len(set(self.visible_objects)) != len(self.visible_objects):
            raise ValueError("visible_objects 不得重复")
        if not self.visible_occurrences and not self.visible_objects:
            raise ValueError("reference Evidence 必须声明至少一个可见输入")
        _strict_key(
            self.trace,
            label="OccurrenceReferenceEvidence.trace",
            allow_empty=True,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、维度、来源、可见输入和替代链的完整键。"""
        values: list[int] = [
            _REFERENCE_VERSION,
            *_pack(self.antecedent.stable_key()),
            *_pack(self.dimension.stable_key()),
            self.stance,
            *_pack(self.reason.stable_key()),
            *_pack(self.source.stable_key()),
            self.timestamp_seq,
            len(self.visible_occurrences),
        ]
        for occurrence in self.visible_occurrences:
            values.extend(_pack(occurrence.stable_key()))
        values.append(len(self.visible_objects))
        for identity in self.visible_objects:
            values.extend(_pack(identity.stable_key()))
        values.extend((
            *_pack(self.trace),
            self.supersedes_evidence_id,
        ))
        return tuple(values)


@dataclass(frozen=True)
class OccurrenceReferenceRevision:
    """以后文定向反驳把先前唯一 winner 替换为同组候选。"""

    rejected: TypedRef
    replacement: TypedRef
    reason: OccurrenceReferenceEvidence

    def __post_init__(self) -> None:
        _occurrence(self.rejected, label="reference revision rejected")
        _occurrence(self.replacement, label="reference revision replacement")
        if self.rejected == self.replacement:
            raise ValueError("reference revision 必须替换为不同 antecedent")
        if not isinstance(self.reason, OccurrenceReferenceEvidence):
            raise TypeError("reference revision reason 类型错误")
        if (self.reason.antecedent != self.rejected
                or self.reason.stance != EVIDENCE_REFUTE):
            raise ValueError("reference revision 必须引用 rejected 的定向反驳")

    def stable_key(self) -> tuple[int, ...]:
        """返回被替代者、替代者和完整理由 Evidence 键。"""
        return (
            _REFERENCE_VERSION,
            *_pack(self.rejected.stable_key()),
            *_pack(self.replacement.stable_key()),
            *_pack(self.reason.stable_key()),
        )


@dataclass(frozen=True)
class OccurrenceReferenceRequest:
    """一次有界 reference 竞争、上下文 Evidence 和可选回溯请求。"""

    reference: TypedRef
    window_occurrences: tuple[TypedRef, ...]
    candidate_occurrences: tuple[TypedRef, ...]
    runtime_scope: ScopeIdentity
    timestamp_seq: int
    evidence: tuple[OccurrenceReferenceEvidence, ...] = ()
    revisions: tuple[OccurrenceReferenceRevision, ...] = ()

    def __post_init__(self) -> None:
        _occurrence(self.reference, label="reference request reference")
        if (not isinstance(self.window_occurrences, tuple)
                or any(not isinstance(item, TypedRef)
                       or item.object_kind != OBJECT_OCCURRENCE
                       for item in self.window_occurrences)):
            raise TypeError("reference window 必须是 Occurrence tuple")
        if len(set(self.window_occurrences)) != len(
                self.window_occurrences):
            raise ValueError("reference window 不得重复 occurrence")
        if self.reference in self.window_occurrences:
            raise ValueError("reference 自身不得重复进入上下文窗口")
        if (not isinstance(self.candidate_occurrences, tuple)
                or any(not isinstance(item, TypedRef)
                       or item.object_kind != OBJECT_OCCURRENCE
                       for item in self.candidate_occurrences)):
            raise TypeError("reference candidates 必须是 Occurrence tuple")
        if len(set(self.candidate_occurrences)) != len(
                self.candidate_occurrences):
            raise ValueError("reference candidates 不得重复 occurrence")
        if not set(self.candidate_occurrences) <= set(
                self.window_occurrences):
            raise ValueError("reference candidates 必须来自声明的有界窗口")
        if not isinstance(self.runtime_scope, ScopeIdentity):
            raise TypeError("reference runtime_scope 类型错误")
        assert_int(self.timestamp_seq, _where="reference timestamp_seq")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise ValueError("reference timestamp_seq 不得为负")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(item, OccurrenceReferenceEvidence)
                       for item in self.evidence)):
            raise TypeError("reference evidence 类型错误")
        if len({item.stable_key() for item in self.evidence}) != len(
                self.evidence):
            raise ValueError("同一请求不得重复 reference Evidence")
        if (not isinstance(self.revisions, tuple)
                or any(not isinstance(item, OccurrenceReferenceRevision)
                       for item in self.revisions)):
            raise TypeError("reference revisions 类型错误")
        if len(self.revisions) > 1:
            raise ValueError("一次回溯只能替换先前唯一 winner")

    def stable_key(self) -> tuple[int, ...]:
        """返回窗口、候选、运行 scope、Evidence 和回溯的完整键。"""
        values: list[int] = [
            _REFERENCE_VERSION,
            *_pack(self.reference.stable_key()),
            len(self.window_occurrences),
        ]
        for occurrence in self.window_occurrences:
            values.extend(_pack(occurrence.stable_key()))
        values.append(len(self.candidate_occurrences))
        for occurrence in self.candidate_occurrences:
            values.extend(_pack(occurrence.stable_key()))
        values.extend((
            *_pack(self.runtime_scope.stable_key()),
            self.timestamp_seq,
            len(self.evidence),
        ))
        for evidence in self.evidence:
            values.extend(_pack(evidence.stable_key()))
        values.append(len(self.revisions))
        for revision in self.revisions:
            values.extend(_pack(revision.stable_key()))
        return tuple(values)


@dataclass(frozen=True)
class OccurrenceReferenceCandidate:
    """一个可回到双侧 Occurrence 图身份的 H-00 指代候选。"""

    hypothesis: HypothesisKey
    reference: TypedRef
    antecedent: TypedRef
    reference_identity: ObjectIdentity
    antecedent_identity: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("reference candidate hypothesis 类型错误")
        _occurrence(self.reference, label="reference candidate reference")
        _occurrence(self.antecedent, label="reference candidate antecedent")
        _identity(
            self.reference_identity,
            label="reference candidate reference identity",
            object_kind=OBJECT_OCCURRENCE,
        )
        _identity(
            self.antecedent_identity,
            label="reference candidate antecedent identity",
            object_kind=OBJECT_OCCURRENCE,
        )
        if self.hypothesis.candidate_key != _candidate_key(
                self.reference_identity, self.antecedent_identity):
            raise ValueError("reference candidate 与 H-00 candidate key 不一致")
        if self.hypothesis.competition_key != _competition_key(
                self.reference_identity):
            raise ValueError("reference candidate 与 H-00 competition 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回 H-00 候选和双侧图引用、图身份的完整键。"""
        return (
            _REFERENCE_VERSION,
            *_pack(self.hypothesis.stable_key()),
            *_pack(self.reference.stable_key()),
            *_pack(self.antecedent.stable_key()),
            *_pack(self.reference_identity.stable_key()),
            *_pack(self.antecedent_identity.stable_key()),
        )


@dataclass(frozen=True)
class OccurrenceReferenceResolution:
    """一次 A-01 决策及其候选、Evidence 和唯一消费投影。"""

    request: OccurrenceReferenceRequest
    candidates: tuple[OccurrenceReferenceCandidate, ...]
    evidence: tuple[EvidenceRecord, ...]
    decision: ResolverDecision | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, OccurrenceReferenceRequest):
            raise TypeError("reference resolution request 类型错误")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, OccurrenceReferenceCandidate)
                       for item in self.candidates)):
            raise TypeError("reference resolution candidates 类型错误")
        hypotheses = tuple(item.hypothesis for item in self.candidates)
        if len(set(hypotheses)) != len(hypotheses):
            raise ValueError("reference resolution candidates 不得重复")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(item, EvidenceRecord)
                       for item in self.evidence)):
            raise TypeError("reference resolution Evidence 类型错误")
        if self.decision is None:
            if self.candidates or self.evidence:
                raise ValueError("无 decision 的 reference resolution 必须为空")
        else:
            if not isinstance(self.decision, ResolverDecision):
                raise TypeError("reference resolution decision 类型错误")
            decision_hypotheses = tuple(
                item.hypothesis for item in self.decision.candidates)
            if set(decision_hypotheses) != set(hypotheses):
                raise ValueError("reference resolution 与 H-04 候选集合不一致")
            if any(item.hypothesis not in set(hypotheses)
                   for item in self.evidence):
                raise ValueError("reference resolution Evidence 越过候选组")

    @property
    def adopted_candidates(self) -> tuple[OccurrenceReferenceCandidate, ...]:
        """按 H-04 adopted 集合返回候选，不额外施加稳定序决胜。"""
        if self.decision is None:
            return ()
        by_hypothesis = {
            item.hypothesis: item for item in self.candidates
        }
        return tuple(
            by_hypothesis[item]
            for item in self.decision.adopted_hypotheses
        )

    @property
    def winner(self) -> OccurrenceReferenceCandidate | None:
        """仅在 adopted 集合恰有一个候选时返回可消费 winner。"""
        adopted = self.adopted_candidates
        return adopted[0] if len(adopted) == 1 else None

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、候选、Evidence 和 H-04 决策的完整稳定键。"""
        decision = () if self.decision is None else self.decision.stable_key()
        values: list[int] = [
            _REFERENCE_VERSION,
            *_pack(self.request.stable_key()),
            len(self.candidates),
        ]
        for candidate in self.candidates:
            values.extend(_pack(candidate.stable_key()))
        values.append(len(self.evidence))
        for evidence in self.evidence:
            values.extend(_pack(evidence.stable_key()))
        values.extend(_pack(decision))
        return tuple(values)


class OccurrenceReferenceRuntime:
    """在 A-09 生命周期内以 H-00/H-04 管理有界 occurrence 指代。"""

    def __init__(
            self,
            occurrence_index: OccurrenceIndex,
            work_memory: WorkMemory,
            protocol: OccurrenceReferenceProtocol,
            *,
            ledger: HypothesisLedger | None = None,
            resolver: HypothesisResolver | None = None,
            ) -> None:
        """绑定 occurrence owner、WorkMemory 生命周期和唯一候选状态 owner。"""
        if not isinstance(occurrence_index, OccurrenceIndex):
            raise TypeError("OccurrenceReferenceRuntime 需要 OccurrenceIndex")
        if not isinstance(work_memory, WorkMemory):
            raise TypeError("OccurrenceReferenceRuntime 需要 WorkMemory")
        if not isinstance(protocol, OccurrenceReferenceProtocol):
            raise TypeError("OccurrenceReferenceRuntime protocol 类型错误")
        target_ledger = HypothesisLedger() if ledger is None else ledger
        if not isinstance(target_ledger, HypothesisLedger):
            raise TypeError("OccurrenceReferenceRuntime ledger 类型错误")
        target_resolver = (
            HypothesisResolver(target_ledger)
            if resolver is None else resolver
        )
        if (not isinstance(target_resolver, HypothesisResolver)
                or target_resolver.ledger is not target_ledger):
            raise ValueError("OccurrenceReferenceRuntime resolver 必须绑定同一 ledger")
        self.occurrence_index = occurrence_index
        self.work_memory = work_memory
        self.protocol = protocol
        self.ledger = target_ledger
        self.resolver = target_resolver
        self._candidates: dict[
            HypothesisKey, OccurrenceReferenceCandidate] = {}

    def resolve(
            self,
            request: OccurrenceReferenceRequest,
            *,
            commit: bool = True,
            ) -> OccurrenceReferenceResolution:
        """预检后原子式形成候选、追加 Evidence 并执行 H-04 决策。"""
        if not isinstance(request, OccurrenceReferenceRequest):
            raise TypeError("OccurrenceReferenceRuntime request 类型错误")
        if type(commit) is not bool:
            raise TypeError("OccurrenceReferenceRuntime commit 必须是 bool")
        self._context(request)
        cloned_ledger = self.ledger.clone()
        cloned_resolver = self.resolver.clone(ledger=cloned_ledger)
        preview = self._apply(
            request,
            ledger=cloned_ledger,
            resolver=cloned_resolver,
            candidates=dict(self._candidates),
        )
        if not commit:
            return preview
        return self._apply(
            request,
            ledger=self.ledger,
            resolver=self.resolver,
            candidates=self._candidates,
        )

    def clone_for_context(
            self,
            occurrence_index: OccurrenceIndex,
            work_memory: WorkMemory,
            ) -> "OccurrenceReferenceRuntime":
        """复制 H-00/H-04 历史并重绑评测上下文的 occurrence 与生命周期 owner。"""
        cloned_ledger = self.ledger.clone()
        cloned = OccurrenceReferenceRuntime(
            occurrence_index,
            work_memory,
            self.protocol,
            ledger=cloned_ledger,
            resolver=self.resolver.clone(ledger=cloned_ledger),
        )
        cloned._candidates = dict(self._candidates)
        return cloned

    def state_key(self) -> tuple:
        """返回协议、候选定义、H-00 与 H-04 的完整可比较状态。"""
        return (
            self.protocol.stable_key(),
            tuple(
                self._candidates[key].stable_key()
                for key in sorted(
                    self._candidates,
                    key=lambda item: item.stable_key(),
                )
            ),
            self.ledger.state_key(),
            self.resolver.state_key(),
        )

    def _apply(
            self,
            request: OccurrenceReferenceRequest,
            *,
            ledger: HypothesisLedger,
            resolver: HypothesisResolver,
            candidates: dict[HypothesisKey, OccurrenceReferenceCandidate],
            ) -> OccurrenceReferenceResolution:
        """在指定状态 owner 上执行一遍已通过上下文校验的确定性事务。"""
        context = self._context(request)
        reference_record = context[request.reference]
        reference_identity = self.occurrence_index.ontology.identity_of(
            request.reference)
        request_candidates: list[OccurrenceReferenceCandidate] = []
        for antecedent in request.candidate_occurrences:
            antecedent_identity = self.occurrence_index.ontology.identity_of(
                antecedent)
            hypothesis = HypothesisKey(
                self.protocol.hypothesis_kind_key,
                _candidate_key(reference_identity, antecedent_identity),
                _competition_key(reference_identity),
                reference_record.scope,
                reference_record.source,
            )
            candidate = OccurrenceReferenceCandidate(
                hypothesis,
                request.reference,
                antecedent,
                reference_identity,
                antecedent_identity,
            )
            existing = candidates.get(hypothesis)
            if existing is not None and existing != candidate:
                raise ValueError("同一 reference Hypothesis 绑定了不同 occurrence")
            new_hypothesis = not ledger.has_hypothesis(hypothesis)
            ledger.register(hypothesis)
            candidates[hypothesis] = candidate
            request_candidates.append(candidate)
            if new_hypothesis:
                ledger.append_evidence(self._formation_evidence(
                    candidate,
                    source=reference_record.source,
                    timestamp_seq=request.timestamp_seq,
                ))

        if not request_candidates:
            if request.evidence or request.revisions:
                raise ValueError("无 reference candidate 时不得追加 Evidence 或回溯")
            return OccurrenceReferenceResolution(request, (), (), None)

        anchor = request_candidates[0].hypothesis
        existing_competition = tuple(
            item.hypothesis for item in ledger.competition(anchor))
        requested_hypotheses = {
            item.hypothesis for item in request_candidates
        }
        if set(existing_competition) != requested_hypotheses:
            raise ValueError("后续 reference 请求不得隐藏既有竞争候选")

        previous = resolver.decision_history(anchor)
        revisions = request.revisions
        if revisions:
            latest = previous[-1] if previous else None
            rejected = self._candidate_for_antecedent(
                request_candidates, revisions[0].rejected)
            rejected_snapshot = ledger.snapshot(rejected.hypothesis)
            if (rejected_snapshot.lifecycle == LIFECYCLE_ACTIVE
                    and (latest is None
                         or latest.adopted_hypotheses
                         != (rejected.hypothesis,))):
                raise ValueError("reference revision 只能替换先前唯一 winner")

        claims = [*request.evidence]
        claims.extend(revision.reason for revision in revisions)
        if len({item.stable_key() for item in claims}) != len(claims):
            raise ValueError("同一事务不得重复提交 reference Evidence")
        appended: dict[tuple[int, ...], EvidenceRecord] = {}
        for claim in claims:
            candidate = self._candidate_for_antecedent(
                request_candidates, claim.antecedent)
            record = self._evidence(
                request,
                claim,
                candidate,
                context=context,
            )
            appended[claim.stable_key()] = ledger.append_evidence(record)

        directives = tuple(
            ReplacementDirective(
                self._candidate_for_antecedent(
                    request_candidates, revision.rejected).hypothesis,
                self._candidate_for_antecedent(
                    request_candidates, revision.replacement).hypothesis,
                appended[revision.reason.stable_key()].evidence_id,
            )
            for revision in revisions
        )
        decision = resolver.resolve(
            anchor,
            timestamp_seq=request.timestamp_seq,
            replacements=directives,
        )
        by_hypothesis = {
            item.hypothesis: item for item in candidates.values()
        }
        all_candidates = tuple(
            by_hypothesis[item.hypothesis]
            for item in decision.candidates
        )
        all_evidence = tuple(sorted(
            (
                evidence
                for candidate in all_candidates
                for evidence in ledger.evidence_history(candidate.hypothesis)
            ),
            key=lambda item: (
                item.timestamp_seq,
                item.evidence_id,
                item.hypothesis.stable_key(),
            ),
        ))
        return OccurrenceReferenceResolution(
            request,
            all_candidates,
            all_evidence,
            decision,
        )

    def _context(
            self,
            request: OccurrenceReferenceRequest,
            ) -> dict[TypedRef, OccurrenceRecord]:
        """核验 A-09 生命周期、来源 scope、窗口预算和严格先行候选。"""
        reference = self.occurrence_index.read(request.reference)
        if self.work_memory.active_document_scope != reference.scope:
            raise ValueError("reference 必须绑定当前活动 document scope")
        if self.work_memory.active_episode_scope != request.runtime_scope:
            raise ValueError("reference 必须绑定当前活动 episode scope")
        if request.runtime_scope.source != reference.source:
            raise ValueError("reference runtime scope 与来源不一致")
        if len(request.window_occurrences) > self.protocol.max_context_occurrences:
            raise ValueError("reference window 超出注入的上下文预算")
        if len(request.candidate_occurrences) > self.protocol.max_candidates:
            raise ValueError("reference candidates 超出注入的候选预算")
        records = {request.reference: reference}
        for occurrence in request.window_occurrences:
            record = self.occurrence_index.read(occurrence)
            if record.source != reference.source or record.scope != reference.scope:
                raise ValueError("reference window 不得跨 SourceRef 或 occurrence scope")
            delta = record.document_index - reference.document_index
            if (delta < 0
                    and -delta > self.protocol.max_preceding_distance):
                raise ValueError("reference window 越过前文预算")
            if (delta > 0
                    and delta > self.protocol.max_following_distance):
                raise ValueError("reference window 越过后文预算")
            records[occurrence] = record
        for occurrence in request.candidate_occurrences:
            if records[occurrence].document_index >= reference.document_index:
                raise ValueError("reference antecedent 必须严格位于 reference 之前")
        return records

    def _formation_evidence(
            self,
            candidate: OccurrenceReferenceCandidate,
            *,
            source: SourceRef,
            timestamp_seq: int,
            ) -> EvidenceRecord:
        """为新候选追加 unknown 形成事件，不把候选出现误记为支持。"""
        payload = (
            _REFERENCE_VERSION,
            *_pack(self.protocol.formation_dimension.stable_key()),
            *_pack(candidate.reference_identity.stable_key()),
            *_pack(candidate.antecedent_identity.stable_key()),
        )
        evidence_id = _EVIDENCE_HASHER.h63((
            candidate.hypothesis.stable_key(),
            EVIDENCE_UNKNOWN,
            self.protocol.formation_reason.stable_key(),
            source.stable_key(),
            timestamp_seq,
            payload,
            0,
        )) or 1
        return EvidenceRecord(
            evidence_id,
            candidate.hypothesis,
            EVIDENCE_UNKNOWN,
            self.protocol.formation_reason.stable_key(),
            source,
            timestamp_seq,
            payload,
        )

    def _evidence(
            self,
            request: OccurrenceReferenceRequest,
            claim: OccurrenceReferenceEvidence,
            candidate: OccurrenceReferenceCandidate,
            *,
            context: dict[TypedRef, OccurrenceRecord],
            ) -> EvidenceRecord:
        """把调用方维度 Evidence 转为精确绑定候选的 H-00 追加事件。"""
        reference_record = context[request.reference]
        if claim.source != reference_record.source:
            raise ValueError("A-01 当前上下文 Evidence 不得跨 SourceRef")
        if claim.timestamp_seq != request.timestamp_seq:
            raise ValueError("reference Evidence 必须属于本次逻辑序")
        for occurrence in claim.visible_occurrences:
            if occurrence not in context:
                raise ValueError("reference Evidence 引用了窗口外 occurrence")
        visible_occurrences = tuple(
            self.occurrence_index.ontology.identity_of(item).stable_key()
            for item in claim.visible_occurrences
        )
        payload_values: list[int] = [
            _REFERENCE_VERSION,
            *_pack(claim.dimension.stable_key()),
            *_pack(candidate.reference_identity.stable_key()),
            *_pack(candidate.antecedent_identity.stable_key()),
            len(visible_occurrences),
        ]
        for identity_key in visible_occurrences:
            payload_values.extend(_pack(identity_key))
        payload_values.append(len(claim.visible_objects))
        for identity in claim.visible_objects:
            payload_values.extend(_pack(identity.stable_key()))
        payload_values.extend(_pack(claim.trace))
        payload = tuple(payload_values)
        evidence_id = _EVIDENCE_HASHER.h63((
            candidate.hypothesis.stable_key(),
            claim.stance,
            claim.reason.stable_key(),
            claim.source.stable_key(),
            claim.timestamp_seq,
            payload,
            claim.supersedes_evidence_id,
        )) or 1
        return EvidenceRecord(
            evidence_id,
            candidate.hypothesis,
            claim.stance,
            claim.reason.stable_key(),
            claim.source,
            claim.timestamp_seq,
            payload,
            claim.supersedes_evidence_id,
        )

    @staticmethod
    def _candidate_for_antecedent(
            candidates: list[OccurrenceReferenceCandidate],
            antecedent: TypedRef,
            ) -> OccurrenceReferenceCandidate:
        """按完整 typed occurrence 找到唯一候选，拒绝窗口外引用。"""
        matches = tuple(
            item for item in candidates if item.antecedent == antecedent
        )
        if len(matches) != 1:
            raise ValueError("reference Evidence 或 revision 引用了候选外 antecedent")
        return matches[0]


__all__ = [
    "OccurrenceReferenceCandidate",
    "OccurrenceReferenceEvidence",
    "OccurrenceReferenceProtocol",
    "OccurrenceReferenceRequest",
    "OccurrenceReferenceResolution",
    "OccurrenceReferenceRevision",
    "OccurrenceReferenceRuntime",
]
