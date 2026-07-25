"""H-05 候选的一等图定义和 append-only active 投影。

候选定义通过动态一等 predicate 写图；生命周期 Event 保存 H-00 当前 Evidence 与
H-04 decision 全键。图消费者只采用 active Event 投影，legacy tier、计数和内存 cache
均不能旁路。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.evidence_candidate import (
    ActiveEvidenceCandidate,
    CANDIDATE_AS_SUBJECT,
    CandidateBinding,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_EVENT,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    event_identity,
    semantic_source,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

_EVENT_VERSION = 1


class CandidateProjectionError(RuntimeError):
    """候选图定义、生命周期事件或恢复投影不一致。"""


def _strict_key(value, *, where: str,
                allow_empty: bool = False) -> tuple[int, ...]:
    """校验开放整数键，必要时允许空 replacement。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长完整键增加长度前缀。"""
    return len(value), *value


def _take(
        values: tuple[int, ...], cursor: int, *, label: str,
        allow_empty: bool = False) -> tuple[tuple[int, ...], int]:
    """从生命周期事件键读取一个长度前缀段。"""
    if cursor >= len(values):
        raise CandidateProjectionError(f"Event 缺少 {label} 长度")
    size = values[cursor]
    cursor += 1
    if size < 0 or (size == 0 and not allow_empty):
        raise CandidateProjectionError(f"Event {label} 长度非法")
    if cursor + size > len(values):
        raise CandidateProjectionError(f"Event {label} 被截断")
    return values[cursor:cursor + size], cursor + size


@dataclass(frozen=True)
class CandidateProjectionProtocol:
    """生命周期 predicate、状态、事件 kind 和命名空间的注入协议。"""

    event_candidate: ObjectIdentity
    event_kind: ObjectIdentity
    event_from_state: ObjectIdentity
    event_to_state: ObjectIdentity
    event_hypothesis: ObjectIdentity
    event_replacement: ObjectIdentity
    inactive_state: ObjectIdentity
    active_state: ObjectIdentity
    superseded_state: ObjectIdentity
    promotion_kind: ObjectIdentity
    refresh_kind: ObjectIdentity
    demotion_kind: ObjectIdentity
    supersede_kind: ObjectIdentity
    event_namespace_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验候选投影的 predicate、状态、事件 kind 和命名空间互异完整。"""
        predicates = self.predicate_identities()
        states_and_kinds = self.state_identities() + self.kind_identities()
        if any(not isinstance(item, ObjectIdentity)
               for item in (*predicates, *states_and_kinds)):
            raise TypeError("投影 predicate/state/kind 必须是 ObjectIdentity")
        if any(item.object_kind != OBJECT_CONCEPT
               for item in (*predicates, *states_and_kinds)):
            raise ValueError("投影 predicate/state/kind 必须是一等 Concept")
        all_identities = (*predicates, *states_and_kinds)
        if len(set(all_identities)) != len(all_identities):
            raise ValueError("投影 predicate/state/kind 必须互不相同")
        _strict_key(
            self.event_namespace_key,
            where="CandidateProjectionProtocol.event_namespace_key",
        )

    def predicate_identities(self) -> tuple[ObjectIdentity, ...]:
        """返回 Event 拓扑使用的全部 predicate 身份。"""
        return (
            self.event_candidate,
            self.event_kind,
            self.event_from_state,
            self.event_to_state,
            self.event_hypothesis,
            self.event_replacement,
        )

    def state_identities(self) -> tuple[ObjectIdentity, ...]:
        """返回 inactive、active 和 superseded 三个状态。"""
        return (
            self.inactive_state,
            self.active_state,
            self.superseded_state,
        )

    def kind_identities(self) -> tuple[ObjectIdentity, ...]:
        """返回 promotion、demotion 和 supersede 三个事件 kind。"""
        return (
            self.promotion_kind,
            self.refresh_kind,
            self.demotion_kind,
            self.supersede_kind,
        )

    def expected_kind(
            self, from_state: ObjectIdentity,
            to_state: ObjectIdentity) -> ObjectIdentity:
        """把合法状态转换映射到调用方注入的事件 kind。"""
        if from_state == self.inactive_state and to_state == self.active_state:
            return self.promotion_kind
        if from_state == self.active_state and to_state == self.active_state:
            return self.refresh_kind
        if from_state == self.active_state and to_state == self.inactive_state:
            return self.demotion_kind
        if (from_state == self.active_state
                and to_state == self.superseded_state):
            return self.supersede_kind
        raise ValueError("候选投影生命周期转换不合法")


@dataclass(frozen=True)
class CandidateProjectionEvent:
    """保存完整候选、Evidence、decision 和状态转换的图内 Event。"""

    event: ObjectIdentity
    definition: EvidenceCandidateDefinition
    event_kind: ObjectIdentity
    from_state: ObjectIdentity
    to_state: ObjectIdentity
    hypothesis: HypothesisKey
    evidence_keys: tuple[tuple[int, ...], ...]
    decision_key: tuple[int, ...]
    timestamp_seq: int
    replacement: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验生命周期 Event 与候选定义、Hypothesis 和逻辑序一致。"""
        if not isinstance(self.event, ObjectIdentity):
            raise TypeError("event 必须是 ObjectIdentity")
        if self.event.object_kind != OBJECT_EVENT:
            raise ValueError("event 必须是一等 Event")
        if not isinstance(self.definition, EvidenceCandidateDefinition):
            raise TypeError("definition 必须是 EvidenceCandidateDefinition")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        if self.hypothesis.candidate_key != self.definition.stable_key():
            raise ValueError("Event Hypothesis 未绑定完整候选定义")
        for name, identity in (
                ("event_kind", self.event_kind),
                ("from_state", self.from_state),
                ("to_state", self.to_state)):
            if not isinstance(identity, ObjectIdentity):
                raise TypeError(f"{name} 必须是 ObjectIdentity")
            if identity.object_kind != OBJECT_CONCEPT:
                raise ValueError(f"{name} 必须是一等 Concept")
        if not isinstance(self.evidence_keys, tuple) or not self.evidence_keys:
            raise ValueError("投影 Event 必须保存非空 Evidence 全键")
        checked = tuple(
            _strict_key(item, where=f"evidence_keys[{index}]")
            for index, item in enumerate(self.evidence_keys)
        )
        if len(set(checked)) != len(checked):
            raise ValueError("投影 Event 不得重复 Evidence 键")
        _strict_key(self.decision_key, where="decision_key")
        assert_int(self.timestamp_seq, _where="timestamp_seq")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise ValueError("timestamp_seq 必须为非负严格整数")
        if self.replacement is not None:
            if not isinstance(self.replacement, ObjectIdentity):
                raise TypeError("replacement 必须是 ObjectIdentity")
            if self.replacement == self.definition.candidate:
                raise ValueError("候选不得 supersede 为自身")
        if semantic_source(self.event) != self.hypothesis.observation:
            raise ValueError("Event 来源与 aggregate Hypothesis 不一致")


@dataclass(frozen=True)
class MaterializedCandidateDefinition:
    """候选图节点、Hypothesis 节点和定义 statement 的恢复结果。"""

    definition: EvidenceCandidateDefinition
    hypothesis: HypothesisKey
    candidate: TypedRef
    hypothesis_ref: TypedRef
    statements: tuple[GraphStatement, ...]


@dataclass(frozen=True)
class MaterializedCandidateProjectionEvent:
    """从图恢复的生命周期 Event 及其拓扑 statement。"""

    definition: CandidateProjectionEvent
    event: TypedRef
    statements: tuple[GraphStatement, ...]


@dataclass(frozen=True)
class CandidateGraphProjection:
    """一个候选当前状态、定义、完整历史和 replacement。"""

    candidate: MaterializedCandidateDefinition
    state: ObjectIdentity
    history: tuple[MaterializedCandidateProjectionEvent, ...]
    replacement: ObjectIdentity | None


class CandidateProjectionGraph:
    """写入候选定义和 lifecycle Event，并仅从图投影当前 active 集。"""

    def __init__(
            self, ontology: GraphOntology,
            protocol: CandidateProjectionProtocol) -> None:
        """绑定当前图并物化调用方注入的候选生命周期协议身份。"""
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if not isinstance(protocol, CandidateProjectionProtocol):
            raise TypeError("protocol 必须是 CandidateProjectionProtocol")
        self.ontology = ontology
        self.protocol = protocol
        self._protocol_refs = {
            identity: ontology.materialize(identity)
            for identity in (
                *protocol.predicate_identities(),
                *protocol.state_identities(),
                *protocol.kind_identities(),
            )
        }

    def preflight_definition(
            self, definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey, *, provenance_kind: int,
            epistemic_origin: int = 0, content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedCandidateDefinition | None:
        """零写核验候选定义可新建或精确重放，并返回已有完整定义。"""
        self._validate_definition(definition, hypothesis)
        self._validate_metadata(
            hypothesis.scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return self._preflight_definition(
            definition,
            hypothesis,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )

    def define(
            self, definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey, *, provenance_kind: int,
            epistemic_origin: int = 0, content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedCandidateDefinition:
        """预检候选所有动态 binding 后幂等写图，不写生命周期状态。"""
        existing = self.preflight_definition(
            definition,
            hypothesis,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        if existing is not None:
            return existing
        candidate_ref = self.ontology.materialize(definition.candidate)
        hypothesis_ref = self.ontology.materialize(
            hypothesis.object_identity())
        statements: list[GraphStatement] = []
        for binding in definition.bindings:
            statement = self.ontology.relate(
                self.ontology.materialize(binding.predicate),
                self.ontology.materialize(
                    binding.subject(definition.candidate)),
                self.ontology.materialize(
                    binding.object(definition.candidate)),
                scope=hypothesis.scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                qualifiers=(binding.ordinal, *qualifiers),
            )
            statements.append(statement)
        restored = self.read_definition(hypothesis)
        if (restored.definition != definition
                or restored.candidate != candidate_ref
                or restored.hypothesis_ref != hypothesis_ref):
            raise CandidateProjectionError("候选定义写后恢复不一致")
        expected_hashes = {item.assertion_hash for item in statements}
        if not expected_hashes <= {
                item.assertion_hash for item in restored.statements}:
            raise CandidateProjectionError("候选定义写后缺少 statement")
        return restored

    def read_definition(
            self, hypothesis: HypothesisKey) -> MaterializedCandidateDefinition:
        """从 Hypothesis candidate key 与动态图 statement 双向恢复候选定义。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        try:
            definition = EvidenceCandidateDefinition.from_stable_key(
                hypothesis.candidate_key)
        except (TypeError, ValueError) as exc:
            raise CandidateProjectionError("Hypothesis candidate 定义无法恢复") from exc
        self._validate_definition(definition, hypothesis)
        candidate_ref = self.ontology.resolve(definition.candidate)
        hypothesis_ref = self.ontology.resolve(hypothesis.object_identity())
        if candidate_ref is None or hypothesis_ref is None:
            raise CandidateProjectionError("候选或 Hypothesis 尚未物化")
        statements: list[GraphStatement] = []
        row_cache: dict[
            tuple[int, tuple[int, int]], tuple[GraphStatement, ...]
        ] = {}
        for binding in definition.bindings:
            predicate = self.ontology.resolve(binding.predicate)
            value = self.ontology.resolve(binding.value)
            if predicate is None or value is None:
                raise CandidateProjectionError("候选 binding 端点尚未物化")
            rows = self._cached_binding_rows(
                row_cache,
                candidate_ref,
                binding,
                predicate,
            )
            matching: list[GraphStatement] = []
            conflicting: list[GraphStatement] = []
            for row in rows:
                if not row.assertion.qualifiers:
                    continue
                if row.assertion.qualifiers[0] != binding.ordinal:
                    continue
                if (self._binding_row_matches(
                        row, candidate_ref, binding, value)
                        and row.assertion.scope == hypothesis.scope):
                    matching.append(row)
                else:
                    conflicting.append(row)
            if conflicting:
                raise CandidateProjectionError(
                    "同一 candidate predicate/ordinal 指向竞争端点")
            if len(matching) != 1:
                raise CandidateProjectionError("候选 binding 图拓扑缺失")
            statements.extend(matching)
        self._require_uniform_definition_metadata(
            tuple(statements), hypothesis.scope)
        return MaterializedCandidateDefinition(
            definition,
            hypothesis,
            candidate_ref,
            hypothesis_ref,
            tuple(sorted(
                statements, key=lambda item: item.assertion_hash)),
        )

    def append_event(
            self, definition: CandidateProjectionEvent, *,
            provenance_kind: int, epistemic_origin: int = 0,
            content_version: int = 0, qualifiers: tuple[int, ...] = (),
            ) -> MaterializedCandidateProjectionEvent:
        """核验当前状态和完整 Event 拓扑后 append-only 写入。"""
        self._validate_event(definition)
        self._validate_metadata(
            definition.hypothesis.scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        materialized = self.read_definition(definition.hypothesis)
        if materialized.definition != definition.definition:
            raise CandidateProjectionError("Event 与候选图定义不一致")
        existing = self.ontology.resolve(definition.event)
        if existing is not None and self._event_statements(existing):
            restored = self.read_event(existing)
            if restored.definition != definition:
                raise CandidateProjectionError("同一 Event 身份绑定不同拓扑")
            self._require_metadata(
                restored.statements,
                definition.hypothesis.scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                qualifiers=qualifiers,
            )
            return restored

        history = self.history(materialized.candidate)
        current = self.protocol.inactive_state
        if history:
            current = history[-1].definition.to_state
            if definition.timestamp_seq <= history[-1].definition.timestamp_seq:
                raise CandidateProjectionError("Event 逻辑序必须严格递增")
        if definition.from_state != current:
            raise CandidateProjectionError("Event from_state 与当前投影不一致")
        if definition.replacement is not None:
            replacement_ref = self.ontology.resolve(definition.replacement)
            if replacement_ref is None:
                raise CandidateProjectionError("replacement 候选尚未物化")
            replacement_projection = self.project(replacement_ref)
            if replacement_projection.state != self.protocol.active_state:
                raise CandidateProjectionError("replacement 候选必须 active")
            replacement_hypothesis = (
                replacement_projection.candidate.hypothesis)
            if not self._same_competition(
                    definition.hypothesis, replacement_hypothesis):
                raise CandidateProjectionError(
                    "replacement 必须属于完整同一竞争边界")

        event_ref = self.ontology.materialize(definition.event)
        targets = (
            (self.protocol.event_candidate, definition.definition.candidate),
            (self.protocol.event_kind, definition.event_kind),
            (self.protocol.event_from_state, definition.from_state),
            (self.protocol.event_to_state, definition.to_state),
            (self.protocol.event_hypothesis,
             definition.hypothesis.object_identity()),
        )
        statements: list[GraphStatement] = []
        for predicate_identity, value_identity in targets:
            statements.append(self.ontology.relate(
                self._protocol_refs[predicate_identity],
                event_ref,
                self.ontology.materialize(value_identity),
                scope=definition.hypothesis.scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                qualifiers=qualifiers,
            ))
        if definition.replacement is not None:
            statements.append(self.ontology.relate(
                self._protocol_refs[self.protocol.event_replacement],
                event_ref,
                self.ontology.materialize(definition.replacement),
                scope=definition.hypothesis.scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                qualifiers=qualifiers,
            ))
        restored = self.read_event(event_ref)
        if restored.definition != definition:
            raise CandidateProjectionError("Event 写后恢复不一致")
        return restored

    def read_event(
            self, event: TypedRef) -> MaterializedCandidateProjectionEvent:
        """从 Event identity 和六类图槽恢复完整生命周期事件。"""
        identity = self.ontology.identity_of(event)
        definition = self._parse_event_identity(identity)
        singleton = (
            (self.protocol.event_candidate,
             definition.definition.candidate, "candidate"),
            (self.protocol.event_kind, definition.event_kind, "event kind"),
            (self.protocol.event_from_state, definition.from_state, "from state"),
            (self.protocol.event_to_state, definition.to_state, "to state"),
            (self.protocol.event_hypothesis,
             definition.hypothesis.object_identity(), "Hypothesis"),
        )
        statements: list[GraphStatement] = []
        for predicate_identity, expected, label in singleton:
            targets, rows = self._targets(
                self._protocol_refs[predicate_identity], event)
            if len(rows) != 1 or len(targets) != 1 or self.ontology.identity_of(
                    targets[0]) != expected:
                raise CandidateProjectionError(
                    f"Event {label} 槽与 identity 不一致")
            statements.extend(rows)
        replacement_targets, replacement_rows = self._targets(
            self._protocol_refs[self.protocol.event_replacement], event)
        if definition.replacement is None:
            if replacement_targets:
                raise CandidateProjectionError("非 supersede Event 带 replacement")
        elif (len(replacement_rows) != 1
              or len(replacement_targets) != 1
              or self.ontology.identity_of(replacement_targets[0])
              != definition.replacement):
            raise CandidateProjectionError("Event replacement 槽不一致")
        statements.extend(replacement_rows)
        self._require_uniform_event_metadata(
            tuple(statements), definition.hypothesis.scope)
        return MaterializedCandidateProjectionEvent(
            definition,
            event,
            tuple(sorted(
                statements, key=lambda item: item.assertion_hash)),
        )

    def history(
            self, candidate: TypedRef
            ) -> tuple[MaterializedCandidateProjectionEvent, ...]:
        """读取候选完整 Event 历史并拒绝相同逻辑序的竞争事件。"""
        candidate_identity = self.ontology.identity_of(candidate)
        links = self.ontology.statements(
            predicate=self._protocol_refs[self.protocol.event_candidate],
            object_ref=candidate,
        )
        event_refs: dict[ObjectIdentity, TypedRef] = {}
        for link in links:
            identity = self.ontology.identity_of(link.subject)
            prior = event_refs.get(identity)
            if prior is not None and prior != link.subject:
                raise CandidateProjectionError("同一 Event 身份映射到多个节点")
            event_refs[identity] = link.subject
        restored = tuple(
            self.read_event(event_refs[key])
            for key in sorted(event_refs, key=ObjectIdentity.stable_key)
        )
        ordered = tuple(sorted(
            restored,
            key=lambda item: (
                item.definition.timestamp_seq,
                item.definition.event.stable_key(),
            ),
        ))
        timestamps = tuple(item.definition.timestamp_seq for item in ordered)
        if len(set(timestamps)) != len(timestamps):
            raise CandidateProjectionError("候选存在同逻辑序竞争 Event")
        if any(item.definition.definition.candidate != candidate_identity
               for item in ordered):
            raise CandidateProjectionError("候选 history 混入其他对象 Event")
        return ordered

    def project(self, candidate: TypedRef) -> CandidateGraphProjection:
        """顺序执行 Event 链，从图派生候选当前状态和 replacement。"""
        history = self.history(candidate)
        if not history:
            raise CandidateProjectionError("候选没有 lifecycle Event")
        first = history[0].definition
        materialized = self.read_definition(first.hypothesis)
        if materialized.candidate != candidate:
            raise CandidateProjectionError("Event Hypothesis 未绑定当前候选")
        state = self.protocol.inactive_state
        replacement = None
        for item in history:
            event = item.definition
            if (event.definition != materialized.definition
                    or event.hypothesis != materialized.hypothesis):
                raise CandidateProjectionError("Event 链混入不同候选定义")
            if event.from_state != state:
                raise CandidateProjectionError("Event 链 from_state 不连续")
            if event.event_kind != self.protocol.expected_kind(
                    event.from_state, event.to_state):
                raise CandidateProjectionError("Event kind 与状态转换不一致")
            if event.to_state == self.protocol.superseded_state:
                if event.replacement is None:
                    raise CandidateProjectionError("supersede Event 缺 replacement")
                replacement = event.replacement
            elif event.replacement is not None:
                raise CandidateProjectionError("非 supersede Event 带 replacement")
            state = event.to_state
        return CandidateGraphProjection(
            materialized, state, history, replacement)

    def active_for_binding(
            self, binding: CandidateBinding
            ) -> tuple[CandidateGraphProjection, ...]:
        """按一个完整 binding 查询 active 候选，不读取 legacy tier 或计数。"""
        if not isinstance(binding, CandidateBinding):
            raise TypeError("binding 必须是 CandidateBinding")
        predicate = self.ontology.resolve(binding.predicate)
        value = self.ontology.resolve(binding.value)
        if predicate is None or value is None:
            return ()
        rows = self.ontology.statements(
            predicate=predicate,
            subject=(
                value
                if binding.candidate_endpoint != CANDIDATE_AS_SUBJECT
                else None),
            object_ref=(
                value
                if binding.candidate_endpoint == CANDIDATE_AS_SUBJECT
                else None),
        )
        candidate_refs: dict[ObjectIdentity, TypedRef] = {}
        for row in rows:
            if (not row.assertion.qualifiers
                    or row.assertion.qualifiers[0] != binding.ordinal):
                continue
            candidate_ref = (
                row.subject
                if binding.candidate_endpoint == CANDIDATE_AS_SUBJECT
                else row.object)
            identity = self.ontology.identity_of(candidate_ref)
            candidate_refs[identity] = candidate_ref
        projections: list[CandidateGraphProjection] = []
        for identity in sorted(candidate_refs, key=ObjectIdentity.stable_key):
            history = self.history(candidate_refs[identity])
            if not history:
                continue
            projection = self.project(candidate_refs[identity])
            if binding not in projection.candidate.definition.bindings:
                raise CandidateProjectionError(
                    "active 候选出现定义外 binding 拓扑")
            if projection.state == self.protocol.active_state:
                projections.append(projection)
        return tuple(projections)

    def _event_identity(
            self, definition: EvidenceCandidateDefinition, *,
            event_kind: ObjectIdentity, from_state: ObjectIdentity,
            to_state: ObjectIdentity, hypothesis: HypothesisKey,
            evidence_keys: tuple[tuple[int, ...], ...],
            decision_key: tuple[int, ...], timestamp_seq: int,
            replacement: ObjectIdentity | None) -> ObjectIdentity:
        """构造保存完整转换内容的来源化 Event identity。"""
        replacement_key = (
            () if replacement is None else replacement.stable_key())
        key = (
            _EVENT_VERSION,
            *_pack(self.protocol.event_namespace_key),
            *_pack(definition.candidate.stable_key()),
            *_pack(event_kind.stable_key()),
            *_pack(from_state.stable_key()),
            *_pack(to_state.stable_key()),
            *_pack(hypothesis.stable_key()),
            len(evidence_keys),
            *(value for item in evidence_keys for value in _pack(item)),
            *_pack(decision_key),
            timestamp_seq,
            *_pack(replacement_key),
        )
        return event_identity(hypothesis.observation, key)

    def _parse_event_identity(
            self, identity: ObjectIdentity) -> CandidateProjectionEvent:
        """解析 Event declaration 并恢复候选、Evidence 和 decision 全键。"""
        if identity.object_kind != OBJECT_EVENT:
            raise CandidateProjectionError("lifecycle 引用不是 Event")
        source = semantic_source(identity)
        components = identity.components
        declaration, cursor = _take(
            components, 1 + len(source.stable_key()), label="declaration")
        if cursor != len(components):
            raise CandidateProjectionError("Event identity 含尾随数据")
        values = declaration
        if not values or values[0] != _EVENT_VERSION:
            raise CandidateProjectionError("Event key 版本非法")
        cursor = 1
        namespace, cursor = _take(values, cursor, label="namespace")
        if namespace != self.protocol.event_namespace_key:
            raise CandidateProjectionError("Event 不属于当前投影协议")
        candidate_key, cursor = _take(values, cursor, label="candidate")
        event_kind_key, cursor = _take(values, cursor, label="event kind")
        from_key, cursor = _take(values, cursor, label="from state")
        to_key, cursor = _take(values, cursor, label="to state")
        hypothesis_key, cursor = _take(values, cursor, label="Hypothesis")
        if cursor >= len(values):
            raise CandidateProjectionError("Event 缺 Evidence 数量")
        evidence_count = values[cursor]
        cursor += 1
        if evidence_count <= 0:
            raise CandidateProjectionError("Event Evidence 数量非法")
        evidence_keys: list[tuple[int, ...]] = []
        for index in range(evidence_count):
            evidence_key, cursor = _take(
                values, cursor, label=f"Evidence[{index}]")
            evidence_keys.append(evidence_key)
        decision_key, cursor = _take(values, cursor, label="decision")
        if cursor >= len(values):
            raise CandidateProjectionError("Event 缺 timestamp_seq")
        timestamp_seq = values[cursor]
        cursor += 1
        replacement_key, cursor = _take(
            values, cursor, label="replacement", allow_empty=True)
        if cursor != len(values):
            raise CandidateProjectionError("Event key 含尾随数据")
        try:
            hypothesis = HypothesisKey.from_stable_key(hypothesis_key)
            definition = EvidenceCandidateDefinition.from_stable_key(
                hypothesis.candidate_key)
            event_kind = ObjectIdentity.from_stable_key(event_kind_key)
            from_state = ObjectIdentity.from_stable_key(from_key)
            to_state = ObjectIdentity.from_stable_key(to_key)
            candidate = ObjectIdentity.from_stable_key(candidate_key)
            replacement = (
                None if not replacement_key
                else ObjectIdentity.from_stable_key(replacement_key))
        except (TypeError, ValueError) as exc:
            raise CandidateProjectionError("Event 内嵌完整身份无法恢复") from exc
        if (candidate != definition.candidate
                or hypothesis.observation != source):
            raise CandidateProjectionError("Event candidate/source 内外身份不一致")
        restored = CandidateProjectionEvent(
            identity,
            definition,
            event_kind,
            from_state,
            to_state,
            hypothesis,
            tuple(evidence_keys),
            decision_key,
            timestamp_seq,
            replacement,
        )
        self._validate_event(restored)
        return restored

    def _validate_event(self, event: CandidateProjectionEvent) -> None:
        """核验转换 kind、replacement 和完整 Event identity codec。"""
        if not isinstance(event, CandidateProjectionEvent):
            raise TypeError("event 必须是 CandidateProjectionEvent")
        expected_kind = self.protocol.expected_kind(
            event.from_state, event.to_state)
        if event.event_kind != expected_kind:
            raise ValueError("Event kind 与 from/to 不一致")
        if event.to_state == self.protocol.superseded_state:
            if event.replacement is None:
                raise ValueError("supersede Event 必须带 replacement")
        elif event.replacement is not None:
            raise ValueError("非 supersede Event 不得带 replacement")
        expected = self._event_identity(
            event.definition,
            event_kind=event.event_kind,
            from_state=event.from_state,
            to_state=event.to_state,
            hypothesis=event.hypothesis,
            evidence_keys=event.evidence_keys,
            decision_key=event.decision_key,
            timestamp_seq=event.timestamp_seq,
            replacement=event.replacement,
        )
        if expected != event.event:
            raise ValueError("Event identity 未完整编码转换内容")

    def _validate_definition(
            self, definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey) -> None:
        """核验 Hypothesis、候选定义、owner 和当前协议边界。"""
        if not isinstance(definition, EvidenceCandidateDefinition):
            raise TypeError("definition 必须是 EvidenceCandidateDefinition")
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        if (hypothesis.candidate_key != definition.stable_key()
                or hypothesis.competition_key != definition.competition_key):
            raise ValueError("Hypothesis 与候选定义完整键不一致")
        if definition.candidate.owner != hypothesis.observation.owner:
            raise ValueError("candidate owner 与 aggregate source 不一致")
        if definition.candidate.versions != hypothesis.observation.versions:
            raise ValueError("candidate version 与 aggregate source 不一致")
        protocol_identities = frozenset((
            *self.protocol.predicate_identities(),
            *self.protocol.state_identities(),
            *self.protocol.kind_identities(),
        ))
        if definition.candidate in protocol_identities:
            raise ValueError("candidate 不得复用 lifecycle 协议对象")
        if any(binding.predicate in protocol_identities
               for binding in definition.bindings):
            raise ValueError("候选 binding 不得复用 lifecycle predicate")

    def _preflight_definition(
            self, definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey, *, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...],
            ) -> MaterializedCandidateDefinition | None:
        """在零 statement 写状态下核验定义可新建、精确重放或必须拒绝。"""
        candidate = self.ontology.resolve(definition.candidate)
        if candidate is None:
            return None
        matches = 0
        row_cache: dict[
            tuple[int, tuple[int, int]], tuple[GraphStatement, ...]
        ] = {}
        for binding in definition.bindings:
            predicate = self.ontology.resolve(binding.predicate)
            value = self.ontology.resolve(binding.value)
            if predicate is None:
                continue
            rows = self._cached_binding_rows(
                row_cache,
                candidate,
                binding,
                predicate,
            )
            slot_rows = tuple(
                row for row in rows
                if row.assertion.qualifiers
                and row.assertion.qualifiers[0] == binding.ordinal)
            if not slot_rows:
                continue
            expected = (
                hypothesis.scope,
                provenance_kind,
                epistemic_origin,
                content_version,
                (binding.ordinal, *qualifiers),
            )
            exact = tuple(
                row for row in slot_rows
                if value is not None
                and self._binding_row_matches(
                    row, candidate, binding, value)
                and self._statement_metadata(row) == expected)
            if len(slot_rows) != 1 or len(exact) != 1:
                raise CandidateProjectionError(
                    "同一 candidate predicate/ordinal 已绑定竞争端点或元数据")
            matches += 1
        if matches == 0:
            return None
        if matches != len(definition.bindings):
            raise CandidateProjectionError("候选定义存在部分图拓扑")
        restored = self.read_definition(hypothesis)
        self._require_metadata(
            restored.statements,
            hypothesis.scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
            definition_ordinals=True,
        )
        return restored

    def _binding_rows(
            self, candidate: TypedRef, binding: CandidateBinding,
            predicate: TypedRef) -> tuple[GraphStatement, ...]:
        """按候选所在端点读取一个 binding 槽的全部 statement。"""
        if binding.candidate_endpoint == CANDIDATE_AS_SUBJECT:
            return self.ontology.statements(
                predicate=predicate,
                subject=candidate,
            )
        return self.ontology.statements(
            predicate=predicate,
            object_ref=candidate,
        )

    def _cached_binding_rows(
            self,
            cache: dict[
                tuple[int, tuple[int, int]], tuple[GraphStatement, ...]
            ],
            candidate: TypedRef,
            binding: CandidateBinding,
            predicate: TypedRef,
            ) -> tuple[GraphStatement, ...]:
        """在单次定义核验内复用同 predicate 和候选方向的完整 statement 集。"""
        key = binding.candidate_endpoint, predicate.node_ref()
        rows = cache.get(key)
        if rows is None:
            rows = self._binding_rows(candidate, binding, predicate)
            cache[key] = rows
        return rows

    @staticmethod
    def _binding_row_matches(
            row: GraphStatement, candidate: TypedRef,
            binding: CandidateBinding, value: TypedRef) -> bool:
        """核验 statement 的候选端、对端和 binding 方向完全一致。"""
        if binding.candidate_endpoint == CANDIDATE_AS_SUBJECT:
            return row.subject == candidate and row.object == value
        return row.subject == value and row.object == candidate

    def _event_statements(self, event: TypedRef) -> tuple[GraphStatement, ...]:
        """读取 Event 在当前协议全部槽位中的 statement。"""
        return tuple(
            row
            for predicate in self.protocol.predicate_identities()
            for row in self.ontology.statements(
                predicate=self._protocol_refs[predicate],
                subject=event,
            )
        )

    def _targets(
            self, predicate: TypedRef, subject: TypedRef
            ) -> tuple[tuple[TypedRef, ...], tuple[GraphStatement, ...]]:
        """按完整对象身份去重并稳定返回一个 Event 槽的端点。"""
        rows = self.ontology.statements(
            predicate=predicate,
            subject=subject,
        )
        targets: dict[ObjectIdentity, TypedRef] = {}
        for row in rows:
            identity = self.ontology.identity_of(row.object)
            prior = targets.get(identity)
            if prior is not None and prior != row.object:
                raise CandidateProjectionError("同一对象身份映射到多个节点")
            targets[identity] = row.object
        identities = tuple(sorted(targets, key=ObjectIdentity.stable_key))
        return (
            tuple(targets[item] for item in identities),
            tuple(sorted(rows, key=lambda item: item.assertion_hash)),
        )

    @staticmethod
    def _validate_metadata(
            scope: ScopeIdentity, *, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...]) -> None:
        """核验 projection assertion 使用来源化 scope 和开放整数元数据。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if not isinstance(qualifiers, tuple):
            raise TypeError("qualifiers 必须是整数 tuple")
        assert_int(
            provenance_kind,
            epistemic_origin,
            content_version,
            *qualifiers,
            _where="CandidateProjectionGraph.metadata",
        )
        if type(provenance_kind) is not int or provenance_kind <= 0:
            raise ValueError("provenance_kind 必须为严格正整数")
        if type(epistemic_origin) is not int or epistemic_origin < 0:
            raise ValueError("epistemic_origin 必须为非负严格整数")
        if type(content_version) is not int or content_version < 0:
            raise ValueError("content_version 必须为非负严格整数")
        if any(type(item) is not int for item in qualifiers):
            raise ValueError("qualifiers 必须使用严格整数")

    @staticmethod
    def _require_metadata(
            statements: tuple[GraphStatement, ...], scope: ScopeIdentity, *,
            provenance_kind: int, epistemic_origin: int,
            content_version: int, qualifiers: tuple[int, ...],
            definition_ordinals: bool = False) -> None:
        """精确重放时要求每条 Event statement 元数据完全一致。"""
        for statement in statements:
            expected_qualifiers = (
                statement.assertion.qualifiers[:1] + qualifiers
                if definition_ordinals else qualifiers)
            expected = (
                scope,
                provenance_kind,
                epistemic_origin,
                content_version,
                expected_qualifiers,
            )
            if CandidateProjectionGraph._statement_metadata(
                    statement) != expected:
                raise CandidateProjectionError("Event 精确重放元数据不一致")

    @staticmethod
    def _statement_metadata(statement: GraphStatement) -> tuple:
        """提取 statement 的完整来源和版本元数据供统一核验。"""
        assertion = statement.assertion
        return (
            assertion.scope,
            assertion.provenance_kind,
            assertion.epistemic_origin,
            assertion.content_version,
            assertion.qualifiers,
        )

    @classmethod
    def _require_uniform_definition_metadata(
            cls, statements: tuple[GraphStatement, ...],
            scope: ScopeIdentity) -> None:
        """要求候选定义各 binding 共享 scope 和除 ordinal 外的写入元数据。"""
        if not statements:
            raise CandidateProjectionError("候选定义没有 statement")
        normalized = set()
        for statement in statements:
            metadata = cls._statement_metadata(statement)
            qualifiers = metadata[-1]
            if metadata[0] != scope or not qualifiers:
                raise CandidateProjectionError("候选定义 scope 或 ordinal 缺失")
            normalized.add((*metadata[:-1], qualifiers[1:]))
        if len(normalized) != 1:
            raise CandidateProjectionError("候选定义 statement 元数据不一致")

    @classmethod
    def _require_uniform_event_metadata(
            cls, statements: tuple[GraphStatement, ...],
            scope: ScopeIdentity) -> None:
        """要求一个生命周期 Event 的全部槽使用同一精确 assertion 元数据。"""
        metadata = {cls._statement_metadata(item) for item in statements}
        if len(metadata) != 1 or next(iter(metadata))[0] != scope:
            raise CandidateProjectionError("Event statement 元数据不一致")

    @staticmethod
    def _same_competition(
            left: HypothesisKey, right: HypothesisKey) -> bool:
        """按 H-00 完整 kind、竞争组、scope 和 aggregate 来源核验替代边界。"""
        return (
            left.hypothesis_kind == right.hypothesis_kind
            and left.competition_key == right.competition_key
            and left.scope == right.scope
            and left.observation == right.observation
        )

    @staticmethod
    def _active_evidence(
            snapshot, hypothesis: HypothesisKey,
            decision: ResolverDecision, *, ledger) -> tuple[EvidenceRecord, ...]:
        """核验 decision 未陈旧，并从 ledger 恢复当前未被替代 Evidence。"""
        if decision.candidate(hypothesis).after != snapshot:
            raise CandidateProjectionError("H-04 decision 已陈旧")
        if ledger is None:
            return ()
        active_ids = frozenset((
            *snapshot.support_evidence_ids,
            *snapshot.refute_evidence_ids,
            *snapshot.unknown_evidence_ids,
        ))
        evidence = tuple(
            item for item in ledger.evidence_history(hypothesis)
            if item.evidence_id in active_ids
        )
        if not evidence or {item.evidence_id for item in evidence} != active_ids:
            raise CandidateProjectionError("无法完整恢复当前 active Evidence")
        return tuple(sorted(
            evidence, key=lambda item: (
                item.timestamp_seq, item.evidence_id)))


class EvidenceCandidateProjector:
    """连接 EvidenceCandidateEngine 与图内定义、晋升、降级和替代 Event。"""

    def __init__(
            self, engine: EvidenceCandidateEngine,
            graph: CandidateProjectionGraph) -> None:
        """绑定同一候选 engine 与图 facade，驱动生命周期投影。"""
        if not isinstance(engine, EvidenceCandidateEngine):
            raise TypeError("engine 必须是 EvidenceCandidateEngine")
        if not isinstance(graph, CandidateProjectionGraph):
            raise TypeError("graph 必须是 CandidateProjectionGraph")
        self.engine = engine
        self.graph = graph

    def promote(
            self, hypothesis: HypothesisKey, *, timestamp_seq: int,
            provenance_kind: int, epistemic_origin: int = 0,
            content_version: int = 0, qualifiers: tuple[int, ...] = (),
            ) -> CandidateGraphProjection:
        """只把 active+supported+adopted H-05 候选投影为图内 active。"""
        active = self.engine.active(hypothesis)
        if active is None:
            raise CandidateProjectionError(
                "只有 active+supported+adopted 候选可以晋升")
        evidence = self.graph._active_evidence(
            active.snapshot,
            hypothesis,
            active.decision,
            ledger=self.engine.ledger,
        )
        if timestamp_seq < active.decision.timestamp_seq or any(
                item.timestamp_seq > timestamp_seq for item in evidence):
            raise CandidateProjectionError("晋升逻辑序早于 Evidence 或 decision")
        materialized = self.graph.define(
            active.definition,
            hypothesis,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        history = self.graph.history(materialized.candidate)
        if history:
            projection = self.graph.project(materialized.candidate)
            if projection.state == self.graph.protocol.active_state:
                latest = projection.history[-1].definition
                evidence_keys = tuple(item.stable_key() for item in evidence)
                decision_key = active.decision.stable_key()
                if (latest.evidence_keys == evidence_keys
                        and latest.decision_key == decision_key):
                    return projection
                event = self._make_event(
                    active,
                    event_kind=self.graph.protocol.refresh_kind,
                    from_state=self.graph.protocol.active_state,
                    to_state=self.graph.protocol.active_state,
                    evidence=evidence,
                    timestamp_seq=timestamp_seq,
                )
                self.graph.append_event(
                    event,
                    provenance_kind=provenance_kind,
                    epistemic_origin=epistemic_origin,
                    content_version=content_version,
                    qualifiers=qualifiers,
                )
                return self.graph.project(materialized.candidate)
            if projection.state == self.graph.protocol.superseded_state:
                raise CandidateProjectionError("superseded 候选不得再次晋升")
        event = self._make_event(
            active,
            event_kind=self.graph.protocol.promotion_kind,
            from_state=self.graph.protocol.inactive_state,
            to_state=self.graph.protocol.active_state,
            evidence=evidence,
            timestamp_seq=timestamp_seq,
        )
        self.graph.append_event(
            event,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return self.graph.project(materialized.candidate)

    def demote(
            self, hypothesis: HypothesisKey, *, timestamp_seq: int,
            provenance_kind: int, epistemic_origin: int = 0,
            content_version: int = 0, qualifiers: tuple[int, ...] = (),
            ) -> CandidateGraphProjection:
        """候选失去 active supported adopted 后追加 active->inactive Event。"""
        definition = self.engine.definition(hypothesis)
        candidate_ref = self.graph.ontology.resolve(definition.candidate)
        if candidate_ref is None:
            raise CandidateProjectionError("候选尚未物化")
        projection = self.graph.project(candidate_ref)
        if projection.state != self.graph.protocol.active_state:
            raise CandidateProjectionError("只有 active 图投影可以降级")
        if self.engine.active(hypothesis) is not None:
            raise CandidateProjectionError("仍可采用的候选不得降级")
        decisions = self.engine.resolver.decision_history(hypothesis)
        if not decisions:
            raise CandidateProjectionError("降级缺少 H-04 decision")
        decision = decisions[-1]
        snapshot = self.engine.ledger.snapshot(hypothesis)
        if snapshot.lifecycle == LIFECYCLE_SUPERSEDED:
            raise CandidateProjectionError("superseded 候选必须写替代 Event")
        evidence = self.graph._active_evidence(
            snapshot,
            hypothesis,
            decision,
            ledger=self.engine.ledger,
        )
        if timestamp_seq < decision.timestamp_seq or any(
                item.timestamp_seq > timestamp_seq for item in evidence):
            raise CandidateProjectionError("降级逻辑序早于 Evidence 或 decision")
        event = self._make_event_from_parts(
            definition,
            hypothesis,
            decision,
            event_kind=self.graph.protocol.demotion_kind,
            from_state=self.graph.protocol.active_state,
            to_state=self.graph.protocol.inactive_state,
            evidence=evidence,
            timestamp_seq=timestamp_seq,
        )
        self.graph.append_event(
            event,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return self.graph.project(candidate_ref)

    def supersede(
            self, hypothesis: HypothesisKey,
            replacement: HypothesisKey, *, timestamp_seq: int,
            provenance_kind: int, epistemic_origin: int = 0,
            content_version: int = 0, qualifiers: tuple[int, ...] = (),
            ) -> CandidateGraphProjection:
        """把 H-00 已确认的同组替代追加为 active->superseded 图事件。"""
        definition = self.engine.definition(hypothesis)
        replacement_definition = self.engine.definition(replacement)
        candidate_ref = self.graph.ontology.resolve(definition.candidate)
        replacement_ref = self.graph.ontology.resolve(
            replacement_definition.candidate)
        if candidate_ref is None or replacement_ref is None:
            raise CandidateProjectionError("替代双方候选尚未物化")
        projection = self.graph.project(candidate_ref)
        replacement_projection = self.graph.project(replacement_ref)
        if projection.state != self.graph.protocol.active_state:
            raise CandidateProjectionError("只有 active 图候选可以被替代")
        if replacement_projection.state != self.graph.protocol.active_state:
            raise CandidateProjectionError("replacement 图候选必须 active")
        if not self.graph._same_competition(hypothesis, replacement):
            raise CandidateProjectionError("replacement 必须属于完整同一竞争边界")
        snapshot = self.engine.ledger.snapshot(hypothesis)
        transitions = self.engine.ledger.transition_history(hypothesis)
        if snapshot.lifecycle != LIFECYCLE_SUPERSEDED or not transitions:
            raise CandidateProjectionError("H-00 尚未确认 supersede")
        transition = transitions[-1]
        if transition.replacement != replacement:
            raise CandidateProjectionError("H-00 replacement 与图替代目标不一致")
        decisions = self.engine.resolver.decision_history(hypothesis)
        if not decisions:
            raise CandidateProjectionError("替代缺少 H-04 decision")
        decision = decisions[-1]
        evidence = self.graph._active_evidence(
            snapshot,
            hypothesis,
            decision,
            ledger=self.engine.ledger,
        )
        if timestamp_seq < decision.timestamp_seq or any(
                item.timestamp_seq > timestamp_seq for item in evidence):
            raise CandidateProjectionError("替代逻辑序早于 Evidence 或 decision")
        event = self._make_event_from_parts(
            definition,
            hypothesis,
            decision,
            event_kind=self.graph.protocol.supersede_kind,
            from_state=self.graph.protocol.active_state,
            to_state=self.graph.protocol.superseded_state,
            evidence=evidence,
            timestamp_seq=timestamp_seq,
            replacement=replacement_definition.candidate,
        )
        self.graph.append_event(
            event,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return self.graph.project(candidate_ref)

    def _make_event(
            self, active: ActiveEvidenceCandidate, *,
            event_kind: ObjectIdentity, from_state: ObjectIdentity,
            to_state: ObjectIdentity,
            evidence: tuple[EvidenceRecord, ...], timestamp_seq: int,
            replacement: ObjectIdentity | None = None,
            ) -> CandidateProjectionEvent:
        """从 active 候选投影转交完整 Event 构造参数。"""
        return self._make_event_from_parts(
            active.definition,
            active.hypothesis,
            active.decision,
            event_kind=event_kind,
            from_state=from_state,
            to_state=to_state,
            evidence=evidence,
            timestamp_seq=timestamp_seq,
            replacement=replacement,
        )

    def _make_event_from_parts(
            self, definition: EvidenceCandidateDefinition,
            hypothesis: HypothesisKey, decision: ResolverDecision, *,
            event_kind: ObjectIdentity, from_state: ObjectIdentity,
            to_state: ObjectIdentity,
            evidence: tuple[EvidenceRecord, ...], timestamp_seq: int,
            replacement: ObjectIdentity | None = None,
            ) -> CandidateProjectionEvent:
        """构造可逆 Event identity 并创建不可变领域事件。"""
        evidence_keys = tuple(item.stable_key() for item in evidence)
        event_identity_value = self.graph._event_identity(
            definition,
            event_kind=event_kind,
            from_state=from_state,
            to_state=to_state,
            hypothesis=hypothesis,
            evidence_keys=evidence_keys,
            decision_key=decision.stable_key(),
            timestamp_seq=timestamp_seq,
            replacement=replacement,
        )
        event = CandidateProjectionEvent(
            event_identity_value,
            definition,
            event_kind,
            from_state,
            to_state,
            hypothesis,
            evidence_keys,
            decision.stable_key(),
            timestamp_seq,
            replacement,
        )
        self.graph._validate_event(event)
        return event


__all__ = [
    "CandidateGraphProjection",
    "CandidateProjectionError",
    "CandidateProjectionEvent",
    "CandidateProjectionGraph",
    "CandidateProjectionProtocol",
    "EvidenceCandidateProjector",
    "MaterializedCandidateDefinition",
    "MaterializedCandidateProjectionEvent",
]
