"""H-06 跨 occurrence 顺序假设、证据累计和 context split。

本模块只管理完整身份、H-00 Evidence 与 append-only 生命周期，不解释具体语言、
顺序方向、邻接、距离、角色或 context 语义。所有这些含义都由图内一等对象和
调用方 verifier 注入；hash 只用于事件索引，不能替代完整身份。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    ArchiveDirective,
    HypothesisResolver,
    ReplacementDirective,
    ResolverDecision,
    TypedResolverScorer,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_OCCURRENCE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

_PATTERN_KEY_VERSION = 1
_OBSERVATION_KEY_VERSION = 1
_PROTOCOL_KEY_VERSION = 1
_EVIDENCE_PAYLOAD_VERSION = 1
_SPLIT_KEY_VERSION = 1
_EVIDENCE_HASHER = Hasher("order_hypothesis.evidence.v1")
_SPLIT_HASHER = Hasher("order_hypothesis.context_split.v1")
_STANCES = frozenset({
    EVIDENCE_SUPPORT,
    EVIDENCE_REFUTE,
    EVIDENCE_UNKNOWN,
})


def _integer_key(value, *, where: str,
                 allow_empty: bool = False) -> tuple[int, ...]:
    """校验开放整数键，并按调用位置决定是否允许空键。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _strict_nonnegative(value: int, *, where: str) -> int:
    """校验位置、逻辑序和可选 Evidence 引用为非负严格整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为非负严格整数")
    return value


def _identity(value, *, where: str,
              object_kind: int | None = None) -> ObjectIdentity:
    """校验字段是一等完整对象身份，并可限制其对象类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if object_kind is not None and value.object_kind != object_kind:
        raise ValueError(f"{where} 对象类型错误")
    return value


def _identity_sequence(
        values, *, where: str,
        allow_empty: bool = True) -> tuple[ObjectIdentity, ...]:
    """校验按稳定键规范化的一等对象序列，拒绝重复和顺序漂移。"""
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise ValueError(f"{where} 必须是 ObjectIdentity tuple")
    for index, value in enumerate(values):
        _identity(value, where=f"{where}[{index}]")
    expected = tuple(sorted(values, key=ObjectIdentity.stable_key))
    if values != expected or len(set(values)) != len(values):
        raise ValueError(f"{where} 必须按完整稳定键排序且不得重复")
    return values


def _pack_parts(parts: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """用显式长度边界拼接整数键，避免相邻字段产生歧义。"""
    packed: list[int] = [len(parts)]
    for part in parts:
        packed.extend((len(part), *part))
    return tuple(packed)


def _pack_identities(values: tuple[ObjectIdentity, ...]) -> tuple[int, ...]:
    """完整展开一等对象身份序列，不使用对象 hash 或运行时引用。"""
    return _pack_parts(tuple(value.stable_key() for value in values))


def _same_owner(values: tuple[ObjectIdentity, ...], *, where: str) -> None:
    """阻止一个模式把不同 owner 的图对象拼进同一学习身份。"""
    owners = {value.owner for value in values}
    if len(owners) != 1:
        raise ValueError(f"{where} 的一等对象 owner 必须一致")


@dataclass(frozen=True)
class OrderLearningProtocol:
    """定义一个版本化聚合学习集合及其开放事件协议键。"""

    hypothesis_kind: tuple[int, ...]
    support_reason_key: tuple[int, ...]
    refute_reason_key: tuple[int, ...]
    unknown_reason_key: tuple[int, ...]
    split_kind_key: tuple[int, ...]
    aggregate_source: SourceRef
    aggregate_scope: ScopeIdentity

    def __post_init__(self) -> None:
        for name in (
                "hypothesis_kind",
                "support_reason_key",
                "refute_reason_key",
                "unknown_reason_key",
                "split_kind_key"):
            _integer_key(getattr(self, name), where=f"OrderLearningProtocol.{name}")
        reasons = {
            self.support_reason_key,
            self.refute_reason_key,
            self.unknown_reason_key,
        }
        if len(reasons) != 3:
            raise ValueError("三种 Evidence stance 必须使用不同 reason key")
        if not isinstance(self.aggregate_source, SourceRef):
            raise TypeError("aggregate_source 必须是 SourceRef")
        if not isinstance(self.aggregate_scope, ScopeIdentity):
            raise TypeError("aggregate_scope 必须是 ScopeIdentity")
        if self.aggregate_scope.source != self.aggregate_source:
            raise ValueError("aggregate_scope 必须精确指向 aggregate_source")

    def stable_key(self) -> tuple[int, ...]:
        """保存聚合 manifest、scope 和全部注入协议键。"""
        return (
            _PROTOCOL_KEY_VERSION,
            *_pack_parts((
                self.hypothesis_kind,
                self.support_reason_key,
                self.refute_reason_key,
                self.unknown_reason_key,
                self.split_kind_key,
                self.aggregate_source.stable_key(),
                self.aggregate_scope.stable_key(),
            )),
        )

    def reason_for(self, stance: int) -> tuple[int, ...]:
        """把 verifier 三态映射到注入 reason，不在通用层定义理由语义。"""
        if stance == EVIDENCE_SUPPORT:
            return self.support_reason_key
        if stance == EVIDENCE_REFUTE:
            return self.refute_reason_key
        if stance == EVIDENCE_UNKNOWN:
            return self.unknown_reason_key
        raise ValueError("顺序 Evidence stance 未注册")


@dataclass(frozen=True)
class OrderPattern:
    """由图内对象定义的跨 occurrence 顺序候选，不绑定具体词面。"""

    language_branch: ObjectIdentity
    order_kind: ObjectIdentity
    structure_family: ObjectIdentity
    structure_candidate: ObjectIdentity
    first_slot: ObjectIdentity
    second_slot: ObjectIdentity
    constraint: ObjectIdentity
    context: ObjectIdentity
    conditions: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        _identity(
            self.language_branch,
            where="OrderPattern.language_branch",
            object_kind=OBJECT_LANGUAGE_BRANCH,
        )
        values = (
            self.language_branch,
            self.order_kind,
            self.structure_family,
            self.structure_candidate,
            self.first_slot,
            self.second_slot,
            self.constraint,
            self.context,
        )
        for name, value in zip((
                "language_branch", "order_kind", "structure_family",
                "structure_candidate", "first_slot", "second_slot",
                "constraint", "context"), values, strict=True):
            _identity(value, where=f"OrderPattern.{name}")
        _identity_sequence(self.conditions, where="OrderPattern.conditions")
        _same_owner((*values, *self.conditions), where="OrderPattern")
        if self.first_slot == self.second_slot:
            raise ValueError("顺序模式的两个 slot 必须不同")
        if self.first_slot.stable_key() > self.second_slot.stable_key():
            raise ValueError("slot pair 必须按完整对象稳定键规范化")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含 constraint 和 context 的完整候选身份。"""
        return (
            _PATTERN_KEY_VERSION,
            *_pack_identities((
                self.language_branch,
                self.order_kind,
                self.structure_family,
                self.structure_candidate,
                self.first_slot,
                self.second_slot,
                self.constraint,
                self.context,
                *self.conditions,
            )),
        )

    def competition_key(self) -> tuple[int, ...]:
        """返回同 context 下不同 constraint 直接竞争的完整边界。"""
        return (
            _PATTERN_KEY_VERSION,
            1,
            *_pack_identities((
                self.language_branch,
                self.order_kind,
                self.structure_family,
                self.structure_candidate,
                self.first_slot,
                self.second_slot,
                self.context,
                *self.conditions,
            )),
        )

    def split_family_key(self) -> tuple[int, ...]:
        """返回 context split 必须保持不变的结构和 slot 家族。"""
        return (
            _PATTERN_KEY_VERSION,
            2,
            *_pack_identities((
                self.language_branch,
                self.order_kind,
                self.structure_family,
                self.structure_candidate,
                self.first_slot,
                self.second_slot,
            )),
        )

    def context_key(self) -> tuple[int, ...]:
        """返回 context 与条件的完整身份，用于拒绝伪分裂。"""
        return _pack_identities((self.context, *self.conditions))

    def observation_projection_key(self) -> tuple[int, ...]:
        """返回观察必须真实提供的 typed 映射，不含待验证 constraint。"""
        return _pack_identities((
            self.language_branch,
            self.structure_family,
            self.structure_candidate,
            self.first_slot,
            self.second_slot,
            self.context,
            *self.conditions,
        ))


@dataclass(frozen=True)
class OrderObservation:
    """一条真实来源观察及其由调用方提供的结构、slot 和 context 映射。"""

    source: SourceRef
    scope: ScopeIdentity
    event_key: tuple[int, ...]
    language_branch: ObjectIdentity
    structure_family: ObjectIdentity
    structure_candidate: ObjectIdentity
    first_slot: ObjectIdentity
    second_slot: ObjectIdentity
    context: ObjectIdentity
    conditions: tuple[ObjectIdentity, ...]
    first_occurrence: ObjectIdentity
    second_occurrence: ObjectIdentity
    first_position: int
    second_position: int
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRef):
            raise TypeError("OrderObservation.source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("OrderObservation.scope 必须是 ScopeIdentity")
        if self.scope.source != self.source:
            raise ValueError("OrderObservation.scope 必须精确指向 source")
        _integer_key(self.event_key, where="OrderObservation.event_key")
        _integer_key(
            self.qualifiers,
            where="OrderObservation.qualifiers",
            allow_empty=True,
        )
        _identity(
            self.language_branch,
            where="OrderObservation.language_branch",
            object_kind=OBJECT_LANGUAGE_BRANCH,
        )
        values = (
            self.language_branch,
            self.structure_family,
            self.structure_candidate,
            self.first_slot,
            self.second_slot,
            self.context,
        )
        for name, value in zip((
                "language_branch", "structure_family", "structure_candidate",
                "first_slot", "second_slot", "context"), values, strict=True):
            _identity(value, where=f"OrderObservation.{name}")
        _identity_sequence(
            self.conditions, where="OrderObservation.conditions")
        _identity(
            self.first_occurrence,
            where="OrderObservation.first_occurrence",
            object_kind=OBJECT_OCCURRENCE,
        )
        _identity(
            self.second_occurrence,
            where="OrderObservation.second_occurrence",
            object_kind=OBJECT_OCCURRENCE,
        )
        if self.first_occurrence == self.second_occurrence:
            raise ValueError("顺序观察的两个 occurrence 必须不同")
        _strict_nonnegative(
            self.first_position, where="OrderObservation.first_position")
        _strict_nonnegative(
            self.second_position, where="OrderObservation.second_position")
        _same_owner(
            (*values, *self.conditions,
             self.first_occurrence, self.second_occurrence),
            where="OrderObservation",
        )
        if self.source.owner != self.language_branch.owner:
            raise ValueError("观察来源与语言分支 owner 不一致")

    def projection_key(self) -> tuple[int, ...]:
        """返回本观察声明的 typed 结构映射。"""
        return _pack_identities((
            self.language_branch,
            self.structure_family,
            self.structure_candidate,
            self.first_slot,
            self.second_slot,
            self.context,
            *self.conditions,
        ))

    def stable_key(self) -> tuple[int, ...]:
        """完整保存来源、scope、映射、occurrence、位置和 qualifier。"""
        return (
            _OBSERVATION_KEY_VERSION,
            *_pack_parts((
                self.source.stable_key(),
                self.scope.stable_key(),
                self.event_key,
                self.projection_key(),
                self.first_occurrence.stable_key(),
                self.second_occurrence.stable_key(),
                (self.first_position, self.second_position),
                self.qualifiers,
            )),
        )


@dataclass(frozen=True)
class OrderAssessment:
    """注入式 verifier 对单条顺序观察给出的三态结论和审计详情。"""

    stance: int
    detail_key: tuple[int, ...]

    def __post_init__(self) -> None:
        assert_int(self.stance, _where="OrderAssessment.stance")
        if self.stance not in _STANCES:
            raise ValueError("OrderAssessment.stance 未注册")
        _integer_key(self.detail_key, where="OrderAssessment.detail_key")


class OrderObservationVerifier(Protocol):
    """领域 verifier 的只读边界；通用层不解释观察位置和 constraint。"""

    def __call__(
            self, pattern: OrderPattern,
            observation: OrderObservation,
            ) -> OrderAssessment: ...


@dataclass(frozen=True)
class OrderContextSplitAssessment:
    """领域 verifier 对 parent→children context 细分给出的审计结论。"""

    accepted: bool
    detail_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("OrderContextSplitAssessment.accepted 必须是 bool")
        _integer_key(
            self.detail_key,
            where="OrderContextSplitAssessment.detail_key",
        )


class OrderContextSplitVerifier(Protocol):
    """领域层核验 child context 确实是 parent 的有效细分。"""

    def __call__(
            self, parent: OrderPattern,
            children: tuple[OrderPattern, ...],
            ) -> OrderContextSplitAssessment: ...


@dataclass(frozen=True)
class OrderEvidenceResult:
    """一次累计返回的完整候选、判断和 H-00 Evidence。"""

    pattern: OrderPattern
    hypothesis: HypothesisKey
    observation: OrderObservation
    assessment: OrderAssessment
    evidence: EvidenceRecord


@dataclass(frozen=True)
class OrderContextSplitEvent:
    """宽 context parent 到多个窄 context child 的 append-only 领域事件。"""

    event_id: int
    split_kind_key: tuple[int, ...]
    parent: HypothesisKey
    children: tuple[HypothesisKey, ...]
    reason_evidence_id: int
    timestamp_seq: int
    detail_key: tuple[int, ...]

    def __post_init__(self) -> None:
        assert_int(self.event_id, _where="OrderContextSplitEvent.event_id")
        if type(self.event_id) is not int or self.event_id <= 0:
            raise ValueError("split event_id 必须为严格正整数")
        _integer_key(
            self.split_kind_key,
            where="OrderContextSplitEvent.split_kind_key",
        )
        if not isinstance(self.parent, HypothesisKey):
            raise TypeError("split parent 必须是 HypothesisKey")
        if (not isinstance(self.children, tuple)
                or len(self.children) < 2
                or any(not isinstance(item, HypothesisKey)
                       for item in self.children)):
            raise ValueError("context split 至少需要两个 Hypothesis child")
        expected = tuple(sorted(
            self.children, key=HypothesisKey.stable_key))
        if self.children != expected or len(set(self.children)) != len(
                self.children):
            raise ValueError("split children 必须规范排序且不得重复")
        if self.parent in self.children:
            raise ValueError("split parent 不得同时作为 child")
        assert_int(
            self.reason_evidence_id,
            _where="OrderContextSplitEvent.reason_evidence_id",
        )
        if type(self.reason_evidence_id) is not int or self.reason_evidence_id <= 0:
            raise ValueError("split 必须引用严格正整数 Evidence id")
        _strict_nonnegative(
            self.timestamp_seq,
            where="OrderContextSplitEvent.timestamp_seq",
        )
        _integer_key(self.detail_key, where="OrderContextSplitEvent.detail_key")

    def identity_key(self) -> tuple[int, ...]:
        """返回不含 hash 索引的完整 split 事件内容。"""
        return (
            _SPLIT_KEY_VERSION,
            *_pack_parts((
                self.split_kind_key,
                self.parent.stable_key(),
                _pack_parts(tuple(
                    child.stable_key() for child in self.children)),
                (self.reason_evidence_id,),
                (self.timestamp_seq,),
                self.detail_key,
            )),
        )

    def stable_key(self) -> tuple[int, ...]:
        """保存 split hash 索引及其完整可碰撞核验内容。"""
        return self.event_id, *self.identity_key()


class OrderHypothesisEngine:
    """累计顺序 Evidence，并把显式解析委托给同一 H-00/H-04 owner。"""

    def __init__(
            self, protocol: OrderLearningProtocol, *,
            ledger: HypothesisLedger | None = None,
            resolver: HypothesisResolver | None = None,
            ) -> None:
        """绑定聚合协议、ledger 和 resolver，禁止两个 owner 状态分叉。"""
        if not isinstance(protocol, OrderLearningProtocol):
            raise TypeError("protocol 必须是 OrderLearningProtocol")
        target_ledger = HypothesisLedger() if ledger is None else ledger
        if not isinstance(target_ledger, HypothesisLedger):
            raise TypeError("ledger 必须是 HypothesisLedger")
        target_resolver = (
            HypothesisResolver(target_ledger)
            if resolver is None else resolver
        )
        if (not isinstance(target_resolver, HypothesisResolver)
                or target_resolver.ledger is not target_ledger):
            raise ValueError("resolver 必须绑定同一个 HypothesisLedger")
        self.protocol = protocol
        self.ledger = target_ledger
        self.resolver = target_resolver
        self._patterns: dict[HypothesisKey, OrderPattern] = {}
        self._evidence_slots: dict[tuple[int, ...], EvidenceRecord] = {}
        self._splits: dict[int, OrderContextSplitEvent] = {}
        self._split_by_parent: dict[HypothesisKey, int] = {}
        self._split_children: dict[HypothesisKey, set[HypothesisKey]] = {}

    def hypothesis_for(self, pattern: OrderPattern) -> HypothesisKey:
        """从完整模式和 aggregate manifest 构造稳定 H-00 候选身份。"""
        self._validate_pattern(pattern)
        return HypothesisKey(
            self.protocol.hypothesis_kind,
            pattern.stable_key(),
            pattern.competition_key(),
            self.protocol.aggregate_scope,
            self.protocol.aggregate_source,
        )

    def register_pattern(self, pattern: OrderPattern) -> HypothesisKey:
        """幂等登记模式，并从已有 H-06 Evidence 恢复其派生事件槽索引。"""
        hypothesis = self.hypothesis_for(pattern)
        existing = self._patterns.get(hypothesis)
        if existing is not None and existing != pattern:
            raise ValueError("同一 Hypothesis 身份绑定了不同 OrderPattern")
        self.ledger.register(hypothesis)
        self._patterns[hypothesis] = pattern
        self._index_active_evidence(hypothesis)
        return hypothesis

    def pattern_for_hypothesis(
            self, hypothesis: HypothesisKey) -> OrderPattern:
        """按完整 H-00 身份恢复已登记模式，并重新核验双向身份映射。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        pattern = self._patterns.get(hypothesis)
        if pattern is None:
            raise KeyError("Hypothesis 尚未登记 OrderPattern")
        if self.hypothesis_for(pattern) != hypothesis:
            raise ValueError("OrderPattern 与 Hypothesis 完整身份不一致")
        self.ledger.snapshot(hypothesis)
        return pattern

    def accumulate(
            self, pattern: OrderPattern, observation: OrderObservation,
            verifier: OrderObservationVerifier, *,
            timestamp_seq: int,
            supersedes_evidence_id: int = 0,
            ) -> OrderEvidenceResult:
        """验证单条观察后只追加 Evidence，不自动归档、替代或多数裁决。"""
        hypothesis = self.hypothesis_for(pattern)
        self._validate_observation(pattern, observation)
        _strict_nonnegative(timestamp_seq, where="accumulate.timestamp_seq")
        _strict_nonnegative(
            supersedes_evidence_id,
            where="accumulate.supersedes_evidence_id",
        )
        if not callable(verifier):
            raise TypeError("verifier 必须可调用")
        assessment = verifier(pattern, observation)
        if not isinstance(assessment, OrderAssessment):
            raise TypeError("verifier 必须返回 OrderAssessment")
        reason_key = self.protocol.reason_for(assessment.stance)
        payload = self._evidence_payload(observation, assessment)
        evidence_id = _EVIDENCE_HASHER.h63((
            hypothesis.stable_key(),
            observation.source.stable_key(),
            payload,
            assessment.stance,
            reason_key,
            supersedes_evidence_id,
        )) or 1
        evidence = EvidenceRecord(
            evidence_id,
            hypothesis,
            assessment.stance,
            reason_key,
            observation.source,
            timestamp_seq,
            payload=payload,
            supersedes_evidence_id=supersedes_evidence_id,
        )
        slot = self._evidence_slot(
            hypothesis, observation.source, observation.event_key)
        prior = self._evidence_slots.get(slot)
        if prior is not None and prior != evidence:
            if supersedes_evidence_id != prior.evidence_id:
                raise ValueError("同一顺序观察槽发生漂移且未显式 supersede")

        validation_ledger = self.ledger.clone()
        validation_ledger.register(hypothesis)
        validation_ledger.append_evidence(evidence)
        self.ledger.register(hypothesis)
        self.ledger.append_evidence(evidence)
        self._patterns[hypothesis] = pattern
        self._evidence_slots[slot] = evidence
        return OrderEvidenceResult(
            pattern, hypothesis, observation, assessment, evidence)

    def resolve(
            self, pattern: OrderPattern, *, timestamp_seq: int,
            scorers: tuple[TypedResolverScorer, ...] = (),
            replacements: tuple[ReplacementDirective, ...] = (),
            archives: tuple[ArchiveDirective, ...] = (),
            commit: bool = True,
            ) -> ResolverDecision:
        """显式调用 H-04；累计路径本身永不隐式触发生命周期退出。"""
        hypothesis = self.hypothesis_for(pattern)
        self.ledger.snapshot(hypothesis)
        return self.resolver.resolve(
            hypothesis,
            timestamp_seq=timestamp_seq,
            scorers=scorers,
            replacements=replacements,
            archives=archives,
            commit=commit,
        )

    def split_context(
            self, parent: OrderPattern,
            children: tuple[OrderPattern, ...], *,
            verifier: OrderContextSplitVerifier,
            reason_evidence_id: int,
            timestamp_seq: int,
            ) -> OrderContextSplitEvent:
        """原子预校验一对多 context split，再归档 parent 并保存领域事件。"""
        if not isinstance(children, tuple) or len(children) < 2:
            raise ValueError("context split 至少需要两个 child pattern")
        parent_hypothesis = self.hypothesis_for(parent)
        child_pairs = tuple(
            (self.hypothesis_for(child), child) for child in children)
        child_pairs = tuple(sorted(
            child_pairs, key=lambda item: item[0].stable_key()))
        child_hypotheses = tuple(item[0] for item in child_pairs)
        child_patterns = tuple(item[1] for item in child_pairs)
        if len(set(child_hypotheses)) != len(child_hypotheses):
            raise ValueError("context split child 不得重复")
        if parent_hypothesis in child_hypotheses:
            raise ValueError("context split parent 不得作为 child")
        for child in child_patterns:
            if child.split_family_key() != parent.split_family_key():
                raise ValueError("context split child 越过结构或 slot family")
        child_contexts = tuple(child.context_key() for child in child_patterns)
        if (len(set(child_contexts)) != len(child_contexts)
                or parent.context_key() in child_contexts):
            raise ValueError("context split 必须产生互异且更具体的 context")
        for hypothesis in (parent_hypothesis, *child_hypotheses):
            pattern = parent if hypothesis == parent_hypothesis else dict(
                child_pairs)[hypothesis]
            if self._patterns.get(hypothesis) != pattern:
                raise ValueError("split parent 和 children 必须先完整登记")
        if self.ledger.snapshot(parent_hypothesis).lifecycle != LIFECYCLE_ACTIVE:
            existing_id = self._split_by_parent.get(parent_hypothesis)
            if existing_id is None:
                raise ValueError("只有 active parent 可以首次 context split")
        for child in child_hypotheses:
            if self.ledger.snapshot(child).lifecycle != LIFECYCLE_ACTIVE:
                raise ValueError("context split child 必须保持 active")
        if not callable(verifier):
            raise TypeError("context split verifier 必须可调用")
        assessment = verifier(parent, child_patterns)
        if not isinstance(assessment, OrderContextSplitAssessment):
            raise TypeError("context split verifier 返回类型错误")
        if not assessment.accepted:
            raise ValueError("context split 未通过领域 verifier")
        _strict_nonnegative(timestamp_seq, where="split_context.timestamp_seq")
        assert_int(reason_evidence_id, _where="split_context.reason_evidence_id")
        if type(reason_evidence_id) is not int or reason_evidence_id <= 0:
            raise ValueError("split 必须引用严格正整数 Evidence id")
        event_without_id = OrderContextSplitEvent(
            1,
            self.protocol.split_kind_key,
            parent_hypothesis,
            child_hypotheses,
            reason_evidence_id,
            timestamp_seq,
            assessment.detail_key,
        )
        event_id = _SPLIT_HASHER.h63(event_without_id.identity_key()) or 1
        event = OrderContextSplitEvent(
            event_id,
            self.protocol.split_kind_key,
            parent_hypothesis,
            child_hypotheses,
            reason_evidence_id,
            timestamp_seq,
            assessment.detail_key,
        )
        existing = self._splits.get(event_id)
        if existing is not None and existing != event:
            raise ValueError("context split hash 碰撞绑定了不同完整事件")
        prior_id = self._split_by_parent.get(parent_hypothesis)
        if prior_id is not None and self._splits[prior_id] != event:
            raise ValueError("同一 parent 不得绑定不同 context split")
        if self._would_create_split_cycle(parent_hypothesis, child_hypotheses):
            raise ValueError("context split 不得形成环")

        directive = ArchiveDirective(parent_hypothesis, reason_evidence_id)
        self.resolver.resolve(
            parent_hypothesis,
            timestamp_seq=timestamp_seq,
            archives=(directive,),
            commit=False,
        )
        self.resolver.resolve(
            parent_hypothesis,
            timestamp_seq=timestamp_seq,
            archives=(directive,),
            commit=True,
        )
        self._splits[event_id] = event
        self._split_by_parent[parent_hypothesis] = event_id
        self._split_children[parent_hypothesis] = set(child_hypotheses)
        return event

    def split_history(
            self, pattern: OrderPattern,
            ) -> tuple[OrderContextSplitEvent, ...]:
        """返回一个模式作为 parent 或 child 参与的全部 split 历史。"""
        hypothesis = self.hypothesis_for(pattern)
        return tuple(sorted(
            (event for event in self._splits.values()
             if event.parent == hypothesis or hypothesis in event.children),
            key=lambda item: (item.timestamp_seq, item.event_id),
        ))

    def clear_derived_indexes(self) -> None:
        """删除可由 Evidence 和 split 事件重建的内存路由，不改权威状态。"""
        self._evidence_slots.clear()
        self._split_by_parent.clear()
        self._split_children.clear()

    def rebuild_derived_indexes(self) -> None:
        """从 H-00 active Evidence 和 append-only split 事件重建全部路由。"""
        self.clear_derived_indexes()
        for hypothesis in sorted(
                self._patterns, key=HypothesisKey.stable_key):
            self._index_active_evidence(hypothesis)
        for event in sorted(
                self._splits.values(), key=lambda item: item.stable_key()):
            prior = self._split_by_parent.get(event.parent)
            if prior is not None and prior != event.event_id:
                raise ValueError("split 历史中同一 parent 绑定了多个事件")
            self._split_by_parent[event.parent] = event.event_id
            self._split_children[event.parent] = set(event.children)

    def clone(self) -> "OrderHypothesisEngine":
        """复制 ledger、resolver、模式和 split 历史，隔离全部可变路由。"""
        cloned_ledger = self.ledger.clone()
        cloned_resolver = self.resolver.clone(ledger=cloned_ledger)
        cloned = OrderHypothesisEngine(
            self.protocol,
            ledger=cloned_ledger,
            resolver=cloned_resolver,
        )
        cloned._patterns = dict(self._patterns)
        cloned._splits = dict(self._splits)
        cloned.rebuild_derived_indexes()
        return cloned

    def state_key(self) -> tuple:
        """返回权威 ledger、resolver、模式与 split 事件的完整状态。"""
        return (
            self.protocol.stable_key(),
            self.ledger.state_key(),
            self.resolver.state_key(),
            tuple(
                (hypothesis.stable_key(), pattern.stable_key())
                for hypothesis, pattern in sorted(
                    self._patterns.items(),
                    key=lambda item: item[0].stable_key(),
                )
            ),
            tuple(
                self._splits[event_id].stable_key()
                for event_id in sorted(self._splits)
            ),
        )

    def _validate_pattern(self, pattern: OrderPattern) -> None:
        """核验模式 owner 与 aggregate manifest 隔离边界一致。"""
        if not isinstance(pattern, OrderPattern):
            raise TypeError("pattern 必须是 OrderPattern")
        if pattern.language_branch.owner != self.protocol.aggregate_source.owner:
            raise ValueError("OrderPattern owner 与 aggregate source 不一致")

    def _validate_observation(
            self, pattern: OrderPattern,
            observation: OrderObservation) -> None:
        """核验观察没有越过模式映射、owner 或真实来源边界。"""
        if not isinstance(observation, OrderObservation):
            raise TypeError("observation 必须是 OrderObservation")
        if observation.projection_key() != pattern.observation_projection_key():
            raise ValueError("观察的结构、slot 或 context 映射与模式不一致")
        if observation.source.owner != self.protocol.aggregate_source.owner:
            raise ValueError("观察来源 owner 与 aggregate source 不一致")

    def _evidence_payload(
            self, observation: OrderObservation,
            assessment: OrderAssessment) -> tuple[int, ...]:
        """把完整观察和 verifier 详情写入 Evidence payload，hash 只作索引。"""
        observation_key = observation.stable_key()
        return (
            _EVIDENCE_PAYLOAD_VERSION,
            len(observation.event_key),
            *observation.event_key,
            len(observation_key),
            *observation_key,
            len(assessment.detail_key),
            *assessment.detail_key,
        )

    @staticmethod
    def _evidence_slot(
            hypothesis: HypothesisKey, source: SourceRef,
            event_key: tuple[int, ...]) -> tuple[int, ...]:
        """构造同一真实来源事件槽的派生路由键。"""
        return _pack_parts((
            hypothesis.stable_key(),
            source.stable_key(),
            event_key,
        ))

    def _slot_from_evidence(
            self, evidence: EvidenceRecord) -> tuple[int, ...]:
        """从本协议 Evidence payload 读取事件槽，拒绝截断和尾随字段。"""
        payload = evidence.payload
        if len(payload) < 4 or payload[0] != _EVIDENCE_PAYLOAD_VERSION:
            raise ValueError("H-06 Evidence payload 版本或长度非法")
        event_size = payload[1]
        if event_size <= 0 or 2 + event_size >= len(payload):
            raise ValueError("H-06 Evidence event_key 被截断")
        event_key = payload[2:2 + event_size]
        cursor = 2 + event_size
        observation_size = payload[cursor]
        cursor += 1
        if observation_size <= 0 or cursor + observation_size >= len(payload):
            raise ValueError("H-06 Evidence observation 被截断")
        cursor += observation_size
        detail_size = payload[cursor]
        cursor += 1
        if detail_size <= 0 or cursor + detail_size != len(payload):
            raise ValueError("H-06 Evidence detail 长度非法")
        return self._evidence_slot(
            evidence.hypothesis, evidence.source, event_key)

    def _index_active_evidence(self, hypothesis: HypothesisKey) -> None:
        """按完整事件槽索引当前 H-06 Evidence，并保留外部 H-02A 反例。"""
        snapshot = self.ledger.snapshot(hypothesis)
        active_ids = frozenset((
            *snapshot.support_evidence_ids,
            *snapshot.refute_evidence_ids,
            *snapshot.unknown_evidence_ids,
        ))
        owned_reasons = frozenset({
            self.protocol.support_reason_key,
            self.protocol.refute_reason_key,
            self.protocol.unknown_reason_key,
        })
        for evidence in self.ledger.evidence_history(hypothesis):
            if (evidence.evidence_id not in active_ids
                    or evidence.reason_key not in owned_reasons):
                continue
            slot = self._slot_from_evidence(evidence)
            prior = self._evidence_slots.get(slot)
            if prior is not None and prior != evidence:
                raise ValueError("同一 H-06 事件槽存在多个 active Evidence")
            self._evidence_slots[slot] = evidence

    def _would_create_split_cycle(
            self, parent: HypothesisKey,
            children: tuple[HypothesisKey, ...]) -> bool:
        """沿既有 parent→children 图检查新边是否回到 parent。"""
        pending = list(children)
        visited: set[HypothesisKey] = set()
        while pending:
            current = pending.pop()
            if current == parent:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self._split_children.get(current, ()))
        return False


__all__ = [
    "OrderAssessment",
    "OrderContextSplitAssessment",
    "OrderContextSplitEvent",
    "OrderContextSplitVerifier",
    "OrderEvidenceResult",
    "OrderHypothesisEngine",
    "OrderLearningProtocol",
    "OrderObservation",
    "OrderObservationVerifier",
    "OrderPattern",
]
