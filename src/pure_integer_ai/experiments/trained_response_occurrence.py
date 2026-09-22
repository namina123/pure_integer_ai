"""把实际采用的来源角色接入 Memory Occurrence 图，不宣称同指或事实。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology, relation_concept_identity
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity, SourceRef, occurrence_identity, span_identity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION, MemoryObjectRef, ObservationPayload,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.semantic_object import context_scope_identity
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry, EvidenceEntry, FrontierEntry, QueryAnchor, QueryRoot, QueryState,
    ROOT_PENDING, SPACE_MEMORY, VisitedKey,
)
from pure_integer_ai.cognition.understanding.occurrence_index import OccurrenceIndex, OccurrenceProtocol
from pure_integer_ai.cognition.understanding.query_open_role_generation import OpenRoleGenerationContext
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, pack_record,
)
from pure_integer_ai.storage.occurrence import register_occurrence_tables
from pure_integer_ai.storage.node_store import register_node_tables


RESPONSE_OCCURRENCE_VERSION = 91534
_DELIVERY_ROLE = (RESPONSE_OCCURRENCE_VERSION, 1)
_OCCURRENCE_ROLE = (RESPONSE_OCCURRENCE_VERSION, 2)
_OBSERVED_SPAN = (RESPONSE_OCCURRENCE_VERSION, 3)
_SPEAKER = (RESPONSE_OCCURRENCE_VERSION, 4)
_INPUT_CONTEXT_ROLE = (RESPONSE_OCCURRENCE_VERSION, 7)
_DELIVERY_PROOF = (RESPONSE_OCCURRENCE_VERSION, 8)


def _compact_edge_qualifier(ordinal: int) -> tuple[int, ...]:
    """保留边序，不在每条边复制完整 proof/hypothesis 稳定键。"""
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("角色边 ordinal 必须是非负严格整数")
    return pack_record(RESPONSE_OCCURRENCE_VERSION, (ordinal,))


def _signed_record(tag: int, *parts: tuple[int, ...]) -> tuple[int, ...]:
    """把有符号整数逐值编码为 ``sign,magnitude``，不越过 SQLite 63 位域。"""
    encoded = []
    for part in parts:
        values = []
        for value in part:
            if type(value) is not int:
                raise ValueError("有符号记录只接受严格整数")
            values.extend((0, value) if value >= 0 else (1, -value - 1))
        encoded.append(tuple(values))
    return pack_record(tag, *encoded)


@dataclass(frozen=True, slots=True)
class ResponseRoleOccurrence:
    """一次已输出采用中的来源发生与角色；不是已消解的指代关系。"""

    occurrence: ObjectIdentity
    origin: ObjectIdentity
    role: ObjectIdentity
    proof_ref: ObjectIdentity
    delivery_context: ObjectIdentity
    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    ordinal: int
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]

    def stable_key(self) -> tuple[int, ...]:
        """保留全部端点、角色序、来源和作用域，不以内存引用定义身份。"""
        return _signed_record(RESPONSE_OCCURRENCE_VERSION, self.occurrence.stable_key(),
                              self.origin.stable_key(), self.role.stable_key(), self.proof_ref.stable_key(),
                              self.delivery_context.stable_key(),
                              self.observation.stable_key(), self.hypothesis.stable_key(),
                              (self.ordinal,), self.source_ref, self.scope_key)

    def query_binding(self) -> BindingEntry:
        """把角色发生绑定在对应输出证明中，防止与当前输入角色混同。"""
        return BindingEntry(_signed_record(RESPONSE_OCCURRENCE_VERSION, self.proof_ref.stable_key(),
                                           self.role.stable_key(), (self.ordinal,)),
                            self.occurrence.stable_key(), space=SPACE_MEMORY, scope_key=self.scope_key)

    def query_evidence(self) -> EvidenceEntry:
        """发生与采用来源可恢复，不把这种结构证据升级为指称内容的支持。"""
        return EvidenceEntry(self.source_ref, self.hypothesis.stable_key(), space=SPACE_MEMORY,
                             polarity=3, trust=1, evidence_key=self.stable_key(),
                             scope_key=self.scope_key, payload_key=self.stable_key())

    def seed(self, weight: int):
        """与三图其它根同时登记候选发生，权重只决定 frontier 优先级。"""
        key = self.stable_key()
        return (QueryRoot(SPACE_MEMORY, key, ROOT_PENDING, weight,
                          source_ref=self.source_ref, scope_key=self.scope_key),
                QueryAnchor(SPACE_MEMORY, key, kind=1, scope_key=self.scope_key),
                FrontierEntry((SPACE_MEMORY, *key), SPACE_MEMORY, target_key=key,
                              root_key=key, source_ref=self.source_ref, scope_key=self.scope_key,
                              owner_weight=weight, discourse_fit=1))

    def expand(self, state: QueryState, edge: FrontierEntry) -> QueryState:
        """将恢复并核验的来源发生加入共同 binding、evidence 和 visited。"""
        key = self.stable_key()
        if edge.owner_space != SPACE_MEMORY or edge.target_key != key:
            raise ValueError("Memory 发生节点与待展开 frontier 不一致")
        return state.with_(
            frontier=tuple(item for item in state.frontier if item != edge),
            depth=max(state.depth, edge.depth + 1),
            bindings=tuple(sorted({*state.bindings, self.query_binding()}, key=lambda item: item.stable_key())),
            evidence=tuple(sorted({*state.evidence, self.query_evidence()}, key=lambda item: item.stable_key())),
            visited=tuple(sorted({*state.visited, VisitedKey(SPACE_MEMORY, key, direction=edge.direction)},
                                 key=lambda item: item.stable_key())),
            node_count=state.node_count + 1, edge_count=state.edge_count + 1, read_count=state.read_count + 1)


@dataclass(frozen=True, slots=True)
class InputRoleOccurrence:
    """当前输入的动态角色发生；只提供来源化 UNKNOWN 证据，不声明同指。"""

    occurrence: ObjectIdentity
    origin: ObjectIdentity
    role: ObjectIdentity
    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    context: ObjectIdentity
    ordinal: int
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    values: tuple[int, ...]

    def stable_key(self) -> tuple[int, ...]:
        """稳定键保留 Observation、Hypothesis、span 和全部来源整数。"""
        return _signed_record(RESPONSE_OCCURRENCE_VERSION, (7,), self.occurrence.stable_key(),
                              self.origin.stable_key(), self.role.stable_key(),
                              self.observation.stable_key(), self.hypothesis.stable_key(),
                              self.context.stable_key(), (self.ordinal,), self.source_ref,
                              self.scope_key, self.values)

    def query_binding(self) -> BindingEntry:
        """把输入角色绑定到当前 Memory 假设，保持动态角色与概念分离。"""
        return BindingEntry(_signed_record(RESPONSE_OCCURRENCE_VERSION, (7,),
                                           self.hypothesis.stable_key(), self.role.stable_key(),
                                           (self.ordinal,)), self.occurrence.stable_key(),
                            space=SPACE_MEMORY, scope_key=self.scope_key)

    def query_evidence(self) -> EvidenceEntry:
        """输入发生只写 UNKNOWN，不能成为事实或 A-01 winner。"""
        return EvidenceEntry(self.source_ref, self.hypothesis.stable_key(), space=SPACE_MEMORY,
                             polarity=3, trust=1, evidence_key=self.stable_key(),
                             scope_key=self.scope_key, payload_key=self.stable_key())

    def seed(self, weight: int):
        """与 Observation/Hypothesis 和其它图根并行进入同一 frontier。"""
        key = self.stable_key()
        return (QueryRoot(SPACE_MEMORY, key, ROOT_PENDING, weight,
                          source_ref=self.source_ref, scope_key=self.scope_key),
                QueryAnchor(SPACE_MEMORY, key, kind=2, scope_key=self.scope_key),
                FrontierEntry((SPACE_MEMORY, *key), SPACE_MEMORY, target_key=key,
                              root_key=key, source_ref=self.source_ref, scope_key=self.scope_key,
                              owner_weight=weight, discourse_fit=1, required_slot_gain=1))

    def expand(self, state: QueryState, edge: FrontierEntry) -> QueryState:
        """展开输入角色发生并登记绑定、UNKNOWN evidence 与 visited。"""
        key = self.stable_key()
        if edge.owner_space != SPACE_MEMORY or edge.target_key != key:
            raise ValueError("输入角色发生节点与待展开 frontier 不一致")
        return state.with_(
            frontier=tuple(item for item in state.frontier if item != edge),
            depth=max(state.depth, edge.depth + 1),
            bindings=tuple(sorted({*state.bindings, self.query_binding()}, key=lambda item: item.stable_key())),
            evidence=tuple(sorted({*state.evidence, self.query_evidence()}, key=lambda item: item.stable_key())),
            visited=tuple(sorted({*state.visited, VisitedKey(SPACE_MEMORY, key, direction=edge.direction)},
                                 key=lambda item: item.stable_key())),
            node_count=state.node_count + 1, edge_count=state.edge_count + 1, read_count=state.read_count + 1)


@dataclass(frozen=True, slots=True)
class ReferenceCandidateSet:
    """跨轮 Role occurrence 候选集合；来源不同则保持 UNKNOWN，不裁决 winner。"""

    reference: InputRoleOccurrence
    antecedents: tuple[ResponseRoleOccurrence, ...]
    context_depths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reference, InputRoleOccurrence):
            raise TypeError("reference 必须是 InputRoleOccurrence")
        if (type(self.antecedents) is not tuple
                or any(not isinstance(item, ResponseRoleOccurrence)
                       for item in self.antecedents)
                or self.antecedents != tuple(sorted(
                    self.antecedents, key=lambda item: item.stable_key()))):
            raise ValueError("antecedents 必须是排序后的 ResponseRoleOccurrence tuple")
        if (not self.antecedents or type(self.context_depths) is not tuple
                or len(self.context_depths) != len(self.antecedents)
                or any(type(item) is not int or item < 0 for item in self.context_depths)):
            raise ValueError("reference candidates 必须保留每个 antecedent 的上下文深度")
        if any(item.source_ref == self.reference.source_ref for item in self.antecedents):
            raise ValueError("跨轮候选不得伪装成同来源 A-01 请求")

    @property
    def source_ref(self) -> tuple[int, ...]:
        """保留真实当前来源首字段，并封存全部跨轮来源键。

        Trace、Evidence 和 QueryHop 的轻量消费者约定 ``source_ref[0]`` 是
        当前来源 hash；完整候选来源仍作为后续整数记录保留，避免把协议 tag
        误报成来源。
        """
        packed = pack_record(RESPONSE_OCCURRENCE_VERSION, self.reference.source_ref,
                             *(item.source_ref for item in self.antecedents))
        return (self.reference.source_ref[0], *packed)

    def stable_key(self) -> tuple[int, ...]:
        """稳定键保留 reference、候选 Occurrence、scope 与 Role。"""
        return _signed_record(RESPONSE_OCCURRENCE_VERSION, (8,),
                              self.reference.stable_key(),
                              *(pack_record(RESPONSE_OCCURRENCE_VERSION,
                                            item.stable_key(), (depth,))
                                for item, depth in zip(self.antecedents, self.context_depths,
                                                       strict=True)))

    @property
    def hot_count(self) -> int:
        """一步显式上下文边内的候选数；只影响 frontier 优先级。"""
        return sum(depth <= 1 for depth in self.context_depths)

    @property
    def cold_count(self) -> int:
        """超过一步但仍在显式闭包中的候选数。"""
        return len(self.context_depths) - self.hot_count

    def query_binding(self) -> BindingEntry:
        """把候选集合绑定到输入发生；不绑定某个 antecedent winner。"""
        return BindingEntry(_signed_record(RESPONSE_OCCURRENCE_VERSION, (8,),
                                           self.reference.role.stable_key()),
                            self.stable_key(), space=SPACE_MEMORY,
                            scope_key=self.reference.scope_key)

    def query_evidence(self) -> EvidenceEntry:
        """候选集合只贡献 UNKNOWN Evidence，候选冲突由后续证据显式形成。"""
        return EvidenceEntry(self.source_ref, self.reference.hypothesis.stable_key(),
                             space=SPACE_MEMORY, polarity=3, trust=1,
                             evidence_key=self.stable_key(),
                             scope_key=self.reference.scope_key,
                             payload_key=self.stable_key())

    def seed(self, weight: int):
        """把跨轮候选集合与三图根同时送入共同 frontier。"""
        key = self.stable_key()
        return (QueryRoot(SPACE_MEMORY, key, ROOT_PENDING, weight,
                          source_ref=self.source_ref, scope_key=self.reference.scope_key),
                QueryAnchor(SPACE_MEMORY, key, kind=3, scope_key=self.reference.scope_key),
                FrontierEntry((SPACE_MEMORY, *key), SPACE_MEMORY, target_key=key,
                              root_key=key, source_ref=self.source_ref,
                              scope_key=self.reference.scope_key, owner_weight=weight,
                              required_slot_gain=1,
                              discourse_fit=1 if self.hot_count else 0,
                              recency_weight=1 if self.hot_count else 0))

    def expand(self, state: QueryState, edge: FrontierEntry) -> QueryState:
        """展开候选集合，保留全部 antecedent 及 UNKNOWN 证据。"""
        key = self.stable_key()
        if edge.owner_space != SPACE_MEMORY or edge.target_key != key:
            raise ValueError("reference candidate frontier 不匹配")
        return state.with_(
            frontier=tuple(item for item in state.frontier if item != edge),
            depth=max(state.depth, edge.depth + 1),
            bindings=tuple(sorted({*state.bindings, self.query_binding()},
                                  key=lambda item: item.stable_key())),
            evidence=tuple(sorted({*state.evidence, self.query_evidence()},
                                  key=lambda item: item.stable_key())),
            visited=tuple(sorted({*state.visited, VisitedKey(SPACE_MEMORY, key,
                                                              direction=edge.direction)},
                                 key=lambda item: item.stable_key())),
            node_count=state.node_count + 1, edge_count=state.edge_count + 1,
            read_count=state.read_count + 1)


def build_response_occurrence_index(events) -> OccurrenceIndex:
    """在同一个包外 Memory 空间复用既有发生索引和图对象，不写训练 Core。"""
    register_node_tables(events.backend)
    register_occurrence_tables(events.backend)
    # 每个图 facade 拥有自己的外部键恢复器；权威整数表仍是同一 backend。
    identities = ScopedIdentityStore(events.backend)
    ontology = GraphOntology(events.backend, space_id=events.memory_space_id,
                             space_identity=events.memory_space_identity,
                             scoped_identities=identities)
    return OccurrenceIndex(ontology, identities,
                           OccurrenceProtocol(_OBSERVED_SPAN, _SPEAKER))


def _expected_roles(context: OpenRoleGenerationContext, proof_ref: ObjectIdentity,
                    output_source: SourceRef) -> tuple[ResponseRoleOccurrence, ...]:
    """只从已完成生成的原角色边构造来源发生身份，不读取自然语言词表。"""
    if proof_ref.owner != context.source.owner or proof_ref.versions != context.source.versions:
        raise ValueError("角色发生的输出证明不属于原始生成来源")
    if output_source.owner != context.source.owner or output_source.versions != context.source.versions:
        raise ValueError("输出发生上下文与输入来源 owner/version 不一致")
    delivery_context = context_scope_identity(output_source, (RESPONSE_OCCURRENCE_VERSION, 5))
    return tuple(ResponseRoleOccurrence(
        occurrence_identity(span.source, start=span.start, end=span.end, ordinal=0),
        span.origin, span.role, proof_ref, delivery_context,
        context.observation, context.hypothesis, span.ordinal,
        context.source_ref, context.scope.stable_key(),
    ) for span in context.observed_spans())


def _speaker_context(observation: ObservationPayload) -> ObjectIdentity:
    """为原 Observation 的 speaker/context 声明建立来源化图对象，旧身份完整保留。"""
    original = observation.context.value()
    if not isinstance(original, ObjectIdentity):
        raise ValueError("发生上下文必须保留原 Observation 的一等声明")
    return context_scope_identity(observation.source,
        _signed_record(RESPONSE_OCCURRENCE_VERSION, (6,), original.stable_key()))


def _input_observation(memory, context: OpenRoleGenerationContext) -> ObservationPayload:
    """核对原始输入 Observation、owner 和活动 manifest，保留 speaker 来源。"""
    events = memory.intake.event_log.query(access=memory.candidate_index.access,
        event_kind=MEMORY_EVENT_OBSERVATION, object_ref=context.observation)
    if (len(events) != 1 or not isinstance(events[0].event.payload, ObservationPayload)
            or events[0].event.payload.source != context.source):
        raise ValueError("角色发生缺少唯一原始输入 Observation")
    memory.intake.require_current_manifest(context.source)
    return events[0].event.payload


def materialize_response_roles(memory, context: OpenRoleGenerationContext,
                               proof_ref: ObjectIdentity, output_source: SourceRef) -> tuple[ResponseRoleOccurrence, ...]:
    """由实际输出采用或显式旧证明迁移调用，幂等追加全部角色 occurrence 及边。"""
    observation = _input_observation(memory, context)
    index = memory.occurrence_index
    ontology = index.ontology
    source_record = index.source_repository.read(context.source_ref[0])
    if source_record.source_key != context.source.stable_key():
        raise ValueError("发生来源记录与原始生成 SourceRef 不一致")
    speaker = _speaker_context(observation)
    memory.intake.require_current_manifest(output_source)
    expected = _expected_roles(context, proof_ref, output_source)
    if expected:
        delivery_context = ontology.materialize(expected[0].delivery_context)
        proof = ontology.materialize(proof_ref)
        proof_predicate = ontology.materialize(
            relation_concept_identity(_DELIVERY_PROOF))
        proof_links = ontology.statements(
            predicate=proof_predicate, subject=delivery_context)
        if proof_links:
            if len(proof_links) != 1 or proof_links[0].object != proof:
                raise ValueError("输出证明 context 已绑定不同 proof")
        else:
            ontology.relate(
                proof_predicate, delivery_context, proof, scope=context.scope,
                provenance_kind=output_source.source_kind,
                content_version=output_source.versions.parser.value)
    for item, span in zip(expected, context.observed_spans(), strict=True):
        if tuple(map(ord, source_record.raw_text[span.start:span.end])) != span.values:
            raise ValueError("角色发生位置与实际生成的原始整数单元不一致")
        record = index.record(
            source=span.source, raw_text=source_record.raw_text, scope=context.scope,
            start=span.start, end=span.end, ordinal=0, segment_index=0,
            local_index=span.start, document_index=span.start, speaker=speaker,
            typed_candidates=(ontology.materialize(span.origin),))
        if ontology.identity_of(record.occurrence) != item.occurrence:
            raise ValueError("发生索引改变了来源身份")
        ontology.relate(ontology.materialize(relation_concept_identity(_OCCURRENCE_ROLE)),
                         record.occurrence, ontology.materialize(span.role), scope=context.scope,
                         provenance_kind=span.source.source_kind,
                         content_version=span.source.versions.parser.value, qualifiers=(span.ordinal,))
        ontology.relate(ontology.materialize(relation_concept_identity(_DELIVERY_ROLE)),
                         ontology.materialize(item.delivery_context), record.occurrence, scope=context.scope,
                         provenance_kind=span.source.source_kind,
                         content_version=span.source.versions.parser.value,
                         qualifiers=_compact_edge_qualifier(span.ordinal))
    return read_response_roles(memory, context, proof_ref, output_source, required=True)


def read_response_roles(memory, context: OpenRoleGenerationContext, proof_ref: ObjectIdentity, output_source: SourceRef,
                        *, required: bool = False) -> tuple[ResponseRoleOccurrence, ...]:
    """沿真实证明边只读恢复完整角色发生；旧历史无本体时不伪造节点或同指。"""
    index = memory.occurrence_index
    ontology = index.ontology
    expected = _expected_roles(context, proof_ref, output_source)
    proof = ontology.resolve(context_scope_identity(output_source, (RESPONSE_OCCURRENCE_VERSION, 5)))
    predicate = ontology.resolve(relation_concept_identity(_DELIVERY_ROLE))
    if proof is None or predicate is None:
        if required:
            raise ValueError("实际输出采用缺少角色发生图")
        return ()
    links = ontology.statements(predicate=predicate, subject=proof)
    if not links:
        if required:
            raise ValueError("实际输出证明缺少角色发生边")
        return ()
    proof_predicate = ontology.resolve(
        relation_concept_identity(_DELIVERY_PROOF))
    if proof_predicate is not None:
        proof_links = ontology.statements(
            predicate=proof_predicate, subject=proof)
        if proof_links and (
                len(proof_links) != 1
                or ontology.identity_of(proof_links[0].object) != proof_ref):
            raise ValueError("实际输出证明 context 的 proof 引用漂移")
    if len(links) != len(expected):
        raise ValueError("实际输出角色发生图没有完整覆盖原始角色")
    observation = _input_observation(memory, context)
    for item, span in zip(expected, context.observed_spans(), strict=True):
        qualifiers = _compact_edge_qualifier(item.ordinal)
        legacy = _signed_record(
            RESPONSE_OCCURRENCE_VERSION, (item.ordinal,), proof_ref.stable_key())
        matches = tuple(link for link in links
                        if link.assertion.qualifiers in {qualifiers, legacy})
        if len(matches) != 1:
            raise ValueError("输出证明角色序缺失或重复")
        link = matches[0]
        if (ontology.identity_of(link.object) != item.occurrence or link.assertion.scope != context.scope
                or link.assertion.provenance_kind != span.source.source_kind
                or link.assertion.content_version != span.source.versions.parser.value):
            raise ValueError("输出证明的角色发生来源漂移")
        record = index.read(link.object)
        if (record.source != span.source or record.scope != context.scope or record.ordinal != 0
                or record.start != span.start or record.end != span.end
                or tuple(map(ord, record.surface)) != span.values or record.speaker is None
                or ontology.identity_of(record.speaker) != _speaker_context(observation)
                or tuple(ontology.identity_of(value.typed_ref) for value in record.candidates)
                   != (span.origin,)):
            raise ValueError("角色发生的来源、原 Span、speaker 或整数单元漂移")
        role_predicate = ontology.resolve(relation_concept_identity(_OCCURRENCE_ROLE))
        if role_predicate is None or not any(
                ontology.identity_of(value.object) == span.role
                and value.assertion.qualifiers == (span.ordinal,) and value.assertion.scope == context.scope
                for value in ontology.statements(predicate=role_predicate, subject=record.occurrence)):
            raise ValueError("角色发生缺少原训练 Role 的有序图边")
    return expected


def _observation_payload(memory, observation: MemoryObjectRef) -> ObservationPayload:
    """读取唯一输入 Observation，保留原始来源与会话 context。"""
    events = memory.intake.event_log.query(access=memory.candidate_index.access,
        event_kind=MEMORY_EVENT_OBSERVATION, object_ref=observation)
    if len(events) != 1 or not isinstance(events[0].event.payload, ObservationPayload):
        raise ValueError("输入角色 Occurrence 缺少唯一 Observation")
    memory.intake.require_current_manifest(events[0].event.payload.source)
    return events[0].event.payload


def _input_role_context(source: SourceRef, candidate: OpenRelationCandidate,
                        observation: MemoryObjectRef, hypothesis: MemoryObjectRef) -> ObjectIdentity:
    """把输入解析的 O/H 端点和完整候选保存在一等来源化 ContextScope。"""
    if (observation.owner != source.owner or hypothesis.owner != source.owner
            or observation.versions != source.versions or hypothesis.versions != source.versions
            or observation.memory_space != hypothesis.memory_space):
        raise ValueError("输入角色 context 的来源、Memory 空间或版本不一致")
    return context_scope_identity(source, _signed_record(
        RESPONSE_OCCURRENCE_VERSION, (7,), observation.stable_key(),
        hypothesis.stable_key(), candidate.stable_key()))


def _expected_input_roles(memory, candidate: OpenRelationCandidate, observation: MemoryObjectRef,
                          hypothesis: MemoryObjectRef) -> tuple[InputRoleOccurrence, ...]:
    """从当前 Observation 来源和开放候选构造可核验的输入角色发生集合。"""
    if not isinstance(candidate, OpenRelationCandidate):
        raise TypeError("输入角色候选必须是 OpenRelationCandidate")
    payload = _observation_payload(memory, observation)
    source = payload.source
    source_record = memory.occurrence_index.source_repository.find(source.stable_key())
    if source_record is None:
        raise ValueError("输入角色 Occurrence 来源记录缺失")
    values = tuple(map(ord, source_record.raw_text))
    if candidate.values != values:
        raise ValueError("输入角色候选与 Observation 来源正文的整数单元不一致")
    scope = document_scope(source)
    context = _input_role_context(source, candidate, observation, hypothesis)
    return tuple(InputRoleOccurrence(
        occurrence_identity(source, start=span.start, end=span.end, ordinal=0),
        span_identity(source, members=((span.start, span.end),)),
        ObjectIdentity.from_stable_key(span.role), observation, hypothesis, context,
        span.ordinal, (source_record.source_hash, *source.stable_key()),
        scope.stable_key(), span.values,
    ) for span in candidate.bindings)


def materialize_input_roles(memory, candidate: OpenRelationCandidate, observation: MemoryObjectRef,
                            hypothesis: MemoryObjectRef) -> tuple[InputRoleOccurrence, ...]:
    """把当前输入的全部开放角色物化为来源 Occurrence，等待同轮 frontier 展开。"""
    payload = _observation_payload(memory, observation)
    source = payload.source
    scope = document_scope(source)
    index = memory.occurrence_index
    ontology = index.ontology
    source_record = index.source_repository.find(source.stable_key())
    if source_record is None:
        raise ValueError("输入角色 Occurrence 来源记录缺失")
    source_hash = source_record.source_hash
    speaker = _speaker_context(payload)
    context = _input_role_context(source, candidate, observation, hypothesis)
    ontology.materialize(speaker)
    ontology.materialize(context)
    result = []
    for span in candidate.bindings:
        origin = span_identity(source, members=((span.start, span.end),))
        ontology.materialize(origin)
        record = index.record(source=source, raw_text=source_record.raw_text, scope=scope,
                              start=span.start, end=span.end, ordinal=0, segment_index=0,
                              local_index=span.start, document_index=span.start,
                              speaker=speaker, typed_candidates=(ontology.materialize(origin),))
        role_identity = ObjectIdentity.from_stable_key(span.role)
        ontology.materialize(role_identity)
        ontology.relate(ontology.materialize(relation_concept_identity(_OCCURRENCE_ROLE)),
                        record.occurrence, ontology.materialize(role_identity), scope=scope,
                        provenance_kind=source.source_kind,
                        content_version=source.versions.parser.value, qualifiers=(span.ordinal,))
        ontology.relate(ontology.materialize(relation_concept_identity(_INPUT_CONTEXT_ROLE)),
                        ontology.materialize(context), record.occurrence, scope=scope,
                        provenance_kind=source.source_kind,
                        content_version=source.versions.parser.value,
                        qualifiers=_compact_edge_qualifier(span.ordinal))
        result.append(InputRoleOccurrence(
            ontology.identity_of(record.occurrence), origin, ObjectIdentity.from_stable_key(span.role),
            observation, hypothesis, context, span.ordinal, (source_hash, *source.stable_key()),
            scope.stable_key(), span.values))
    return tuple(sorted(result, key=lambda item: item.stable_key()))


def read_input_roles(memory, candidate: OpenRelationCandidate, observation: MemoryObjectRef,
                     hypothesis: MemoryObjectRef, *, required: bool = False) -> tuple[InputRoleOccurrence, ...]:
    """冷恢复输入角色 occurrence、context 边、role 边、speaker 和原整数 span。"""
    expected = _expected_input_roles(memory, candidate, observation, hypothesis)
    index = memory.occurrence_index
    ontology = index.ontology
    context = ontology.resolve(expected[0].context) if expected else None
    predicate = ontology.resolve(relation_concept_identity(_INPUT_CONTEXT_ROLE))
    if context is None or predicate is None:
        if required:
            raise ValueError("输入角色 Occurrence 图缺少 context 或 predicate")
        return ()
    links = ontology.statements(predicate=predicate, subject=context)
    if len(links) != len(expected):
        if required or links:
            raise ValueError("输入角色 context 没有完整覆盖开放角色")
        return ()
    payload = _observation_payload(memory, observation)
    role_predicate = ontology.resolve(relation_concept_identity(_OCCURRENCE_ROLE))
    if role_predicate is None:
        raise ValueError("输入角色 Occurrence 缺少 role predicate")
    for item, span in zip(expected, candidate.bindings, strict=True):
        qualifiers = _compact_edge_qualifier(item.ordinal)
        legacy = _signed_record(
            RESPONSE_OCCURRENCE_VERSION, (item.ordinal,), hypothesis.stable_key())
        matches = tuple(link for link in links
                        if link.assertion.qualifiers in {qualifiers, legacy})
        if len(matches) != 1 or ontology.identity_of(matches[0].object) != item.occurrence:
            raise ValueError("输入角色 context 边的序或发生身份漂移")
        record = index.read(matches[0].object)
        if (record.source != payload.source or record.scope != document_scope(payload.source)
                or record.start != span.start or record.end != span.end or record.ordinal != 0
                or tuple(map(ord, record.surface)) != item.values
                or record.speaker is None
                or ontology.identity_of(record.speaker) != _speaker_context(payload)
                or tuple(ontology.identity_of(value.typed_ref) for value in record.candidates)
                   != (item.origin,)):
            raise ValueError("输入角色 Occurrence 的来源、scope、speaker、span 或 origin 漂移")
        if not any(ontology.identity_of(value.object) == item.role
                   and value.assertion.qualifiers == (item.ordinal,)
                   and value.assertion.scope == document_scope(payload.source)
                   for value in ontology.statements(predicate=role_predicate, subject=record.occurrence)):
            raise ValueError("输入角色 Occurrence 缺少原 Role 图边")
    return tuple(sorted(expected, key=lambda item: item.stable_key()))


__all__ = [
    "InputRoleOccurrence", "RESPONSE_OCCURRENCE_VERSION", "ReferenceCandidateSet",
    "ResponseRoleOccurrence",
    "build_response_occurrence_index", "materialize_input_roles", "materialize_response_roles",
    "read_input_roles", "read_response_roles",
]
