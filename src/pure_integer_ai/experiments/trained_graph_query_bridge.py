"""三图库并行纵切：把训练后结构接到同一个 QueryState。

Core active relation、Interaction Memory Observation 与训练后 realization frame
分别产生独立 root；所有非空 root 同时进入一个 frontier，owner 权重只决定
扩展次序。Memory 没有 Hypothesis 时只贡献有来源的 Observation，不再把历史
原文或码点近邻伪装成语义边；Dialogue root 只能来自训练后的 Companion 图
结构，生产查询不加载 successor occurrence 或 exact-surface evidence。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.generation_observed_surface import (
    delivered_graph_surface_role,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD

from pure_integer_ai.cognition.shared.query_driver import (
    ExpansionOutcome,
    QueryExpander,
    QueryTracer,
    run_query,
    unvisited_frontier,
)
from pure_integer_ai.cognition.shared.query_termination import (
    evaluate_termination,
    from_query_state,
)
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY,
    ROOT_PENDING,
    SPACE_CORE,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_DIALOGUE_UNKNOWN,
    TERMINATION_NO_FRONTIER,
    TERMINATION_OPEN,
    BindingEntry,
    EvidenceEntry,
    FrontierEntry,
    QueryAnchor,
    QueryBudget,
    QueryRoot,
    QueryState,
    VisitedKey,
    seed_query,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_OBSERVATION,
    MEMORY_OBJECT_EVIDENCE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_OBSERVATION,
    MemoryLinkedRef,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
    STRUCTURE_ORDER_CONFLICT,
    InputRelationCandidate,
    QueryInputStructure,
    TrainedInputStructureProjector,
    graph_query_input_structure,
)
from pure_integer_ai.cognition.understanding.artifact_query_input import (
    ArtifactQueryInput,
)
from pure_integer_ai.cognition.understanding.query_structure_adapter import (
    RelationSurfaceStructureAdapter,
    SemanticCandidateInputView,
    SemanticCandidateStructureAdapter,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION,
    MEMORY_GENERIC_CANDIDATE_VERSION,
    MEMORY_CATEGORY_ARTIFACT,
    MEMORY_CATEGORY_OPEN_RELATION,
    MEMORY_CATEGORY_RESPONSE,
    frame_key,
    memory_competition_identity,
    memory_graph_input_object_key,
)
from pure_integer_ai.cognition.understanding.query_open_role_state import open_role_query_nodes
from pure_integer_ai.cognition.understanding.query_open_role_generation import (
    OpenRoleGenerationContext, prepare_open_role_generation,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    MemoryActiveStructures,
    TrainedDialogueMemoryGraph,
    memory_input_candidate_keys,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    GraphRelationGeneration,
    RelationGraphSurfaceGenerator,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.trained_query_evidence import (
    core_query_evidence,
    current_query_evidence,
    dialogue_query_evidence,
)
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_open_response_runtime import (
    GenericResponseConnectorQueryNode,
    generate_generic_responses,
    generate_open_responses,
    generic_response_query_nodes,
    response_query_nodes,
)
from pure_integer_ai.experiments.unknown_response_course import (
    RESPONSE_FEATURE_COLD_CONTEXT,
    RESPONSE_FEATURE_HOT_CONTEXT,
    RESPONSE_FEATURE_MULTI_HYPOTHESIS,
    RESPONSE_FEATURE_REFERENCE_OPEN,
    RESPONSE_FEATURE_REFERENCE_RESOLVED,
    RESPONSE_FEATURE_TOPIC,
    RESPONSE_FEATURE_TOPIC_GRAPH,
    RESPONSE_FEATURE_UNKNOWN,
    RESPONSE_FEATURE_DISCOURSE_RELATION,
    RESPONSE_FEATURE_RELATION_CHAIN,
    RESPONSE_FEATURE_QUD,
    RESPONSE_FEATURE_REVISION,
    RESPONSE_FEATURE_RELATION_CONFLICT,
    RESPONSE_FEATURE_EVENT_TIME,
    RESPONSE_FEATURE_ENTITY,
    RESPONSE_FEATURE_EVENT,
    RESPONSE_FEATURE_PROPERTY,
)
from pure_integer_ai.experiments.trained_response_delivery import (
    FactualResponseGeneration,
    GenericResponseGeneration,
    record_factual_response_delivery,
    record_generic_response_delivery,
    record_open_response_delivery,
)
from pure_integer_ai.experiments.trained_response_context import (
    DeliveredGraphSlotProjection,
    DeliveredGenericResponseContext,
    DeliveredFactualResponseContext,
    delivered_graph_slot_projections,
    restore_response_contexts,
)
from pure_integer_ai.experiments.trained_response_occurrence import (
    InputRoleOccurrence, ReferenceCandidateSet, materialize_input_roles,
)
from pure_integer_ai.experiments.trained_reference_resolution import (
    REFERENCE_CONFLICT,
    REFERENCE_RESOLVED,
    CrossTurnReferenceResolution,
    TrainedCrossTurnReferenceRuntime,
)
from pure_integer_ai.experiments.relation_capability_route import (
    RelationCapabilityRouteRegistry,
)
from pure_integer_ai.experiments.trained_typed_relation_bridge import (
    TypedRelationProjection,
    project_open_typed_relation,
    project_typed_relation,
)


def _response_plan_connector_key(response_plan):
    """Return the trained connector identity, excluding graph input links."""
    links = tuple(getattr(response_plan, "discourse_links", ()))
    connectors = tuple(
        item.stable_key() for item in links
        if isinstance(item, ObjectIdentity)
        and item.object_kind == OBJECT_STRUCTURE_CONCEPT
    )
    if len(connectors) != 1:
        raise ValueError("ResponsePlan must expose exactly one connector identity")
    return connectors[0]
from pure_integer_ai.experiments.trained_discourse_topic_runtime import (
    DiscourseTopicCandidate,
    derive_discourse_topic_candidates,
    graph_backed_discourse_topic_keys,
    matched_discourse_topic_hypothesis_keys,
    restore_discourse_topic_candidates,
    validate_discourse_topic_query_state,
)
from pure_integer_ai.experiments.discourse_topic_training_bridge import (
    discourse_topic_predicate_identities,
)
from pure_integer_ai.experiments.trained_discourse_topic_topology import (
    DISCOURSE_GRAPH_INPUT_ROLE,
    DiscourseTopicGraphTopology,
    read_discourse_topic_topology,
)
from pure_integer_ai.experiments.event_time_structure_training_bridge import (
    event_time_input_structure_identity,
    event_time_narrative_predicate_prefix,
    event_time_structure_predicate_identities,
)
from pure_integer_ai.experiments.trained_event_time_generation_binding import (
    GENERATION_BINDING_NAMESPACE,
    TrainedEventTimeGenerationBinding,
    read_event_time_generation_bindings,
)
from pure_integer_ai.experiments.trained_event_time_topology import (
    EventTimeGraphEdge,
    EventTimeGraphTopology,
    read_event_time_topology,
)

_TRACE_FORMAT = "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V3"

_CORE = SPACE_CORE
_MEMORY = SPACE_MEMORY
_DIALOGUE = SPACE_DIALOGUE

# Generic graph inputs are semantic identities, not permission to walk every
# persistence-support node attached to their assertion scope.  Role-binding,
# context-scope, occurrence and representation nodes are structural storage
# records; admitting them into the same frontier creates a scope-wide fanout
# that exhausts the shared budget before Memory/Dialogue can close.  Keep the
# semantic graph types available for arbitrary-depth traversal while filtering
# only those non-semantic persistence edges at each hop.
_GENERIC_SEMANTIC_OBJECT_KINDS = frozenset({
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT, OBJECT_PROPOSITION,
})
_EMPTY_ROOT_KEY = (0,)
_DEFAULT_SPACE_WEIGHTS = (300, 200, 100)
_ARTIFACT_ENVELOPE_ANCHOR_KIND = 4
_ARTIFACT_SEMANTIC_ANCHOR_KIND = 5
_ARTIFACT_ENVELOPE_ROLE = (91544, 1)
_ARTIFACT_LOCAL_ROLE = (91544, 2)
_ARTIFACT_PROJECTION_ROLE = (91544, 3)
_ARTIFACT_SEMANTIC_ROLE = (91544, 4)
_ARTIFACT_ENVELOPE_PREDICATE = (91544, 5)
_ARTIFACT_PROJECTION_PREDICATE = (91544, 6)
_ARTIFACT_REFERENCE_ROLE = (91544, 7)
_ARTIFACT_REFERENCE_SOURCE_ROLE = (91544, 8)
_ARTIFACT_REFERENCE_RELATION_ROLE = (91544, 9)
_ARTIFACT_REFERENCE_TARGET_ROLE = (91544, 10)
_ARTIFACT_REFERENCE_STATE_ROLE = (91544, 11)
_ARTIFACT_REFERENCE_FINGERPRINT_ROLE = (91544, 12)
_ARTIFACT_BRIDGE_ROLE = (91544, 13)
_ARTIFACT_BRIDGE_PREDICATE = (91544, 14)
_ARTIFACT_BRIDGE_EVIDENCE = (91544, 15)
_EVENT_TIME_GRAPH_ANCHOR_KIND = 6
_EVENT_TIME_GENERATION_ANCHOR_KIND = 7
_CORE_GRAPH_INPUT_ANCHOR_KIND = 8
_EVENT_TIME_GRAPH_EDGE = (91560, 1)
_EVENT_TIME_GRAPH_EVIDENCE = (91560, 2)
_INPUT_SEMANTIC_PROJECTION_EVIDENCE = (91569, 1)
_INPUT_RELATION_PROJECTION_EVIDENCE = (91569, 2)
_DISCOURSE_TOPIC_GRAPH_EVIDENCE = (91569, 3)
_MEMORY_DELIVERED_GRAPH_SURFACE_EVIDENCE = (91572, 2)


def _contains_integer(haystack: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[index:index + len(needle)] == needle
               for index in range(len(haystack) - len(needle) + 1))


@dataclass(frozen=True, slots=True)
class QueryHop:
    """一次纯整数多跳路径步骤。"""

    space: int
    proposition: tuple[int, ...]
    predicate: tuple[int, ...]
    from_filler: tuple[int, ...]
    to_filler: tuple[int, ...]
    surface_values: tuple[int, ...]
    depth: int
    source_hash: int
    # For a trained Core proposition, retain the complete relation-member
    # order that produced this hop.  This is an integer-only structural
    # carrier for graph-only multi-slot generation; it is not surface text
    # and is omitted from the compact trace, whose bindings/evidence already
    # carry the same proof.
    relation_fillers: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for label in (
                "proposition", "predicate", "from_filler", "to_filler",
                "surface_values"):
            value = getattr(self, label)
            if type(value) is not tuple or any(
                    type(item) is not int or item < 0 for item in value):
                raise ValueError(f"QueryHop.{label} 必须是非负整数 tuple")
        if self.space not in {_CORE, _MEMORY, _DIALOGUE}:
            raise ValueError("QueryHop.space 未注册")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("QueryHop.depth 必须是非负整数")
        if type(self.source_hash) is not int or self.source_hash <= 0:
            raise ValueError("QueryHop.source_hash 必须是正整数")
        if (type(self.relation_fillers) is not tuple
                or any(type(item) is not tuple or not item
                       or any(type(value) is not int or value < 0
                              for value in item)
                       for item in self.relation_fillers)):
            raise ValueError("QueryHop.relation_fillers 必须是整数 tuple 序列")


@dataclass(frozen=True, slots=True)
class ConceptAnchor:
    """输入 span 候选汇聚到的概念 filler。"""

    filler: tuple[int, ...]
    surface_values: tuple[int, ...]
    kind: int
    span_refs: tuple[tuple[int, ...], ...] = ()
    proposition_keys: tuple[tuple[int, ...], ...] = ()
    source_refs: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("filler", "surface_values"):
            value = getattr(self, name)
            if (type(value) is not tuple or not value
                    or any(type(item) is not int or item < 0
                           for item in value)):
                raise ValueError(f"ConceptAnchor.{name} 必须是非空整数 tuple")
        if type(self.kind) is not int or self.kind <= 0:
            raise ValueError("ConceptAnchor.kind 必须为正严格整数")
        for name in ("span_refs", "proposition_keys", "source_refs"):
            values = getattr(self, name)
            if (type(values) is not tuple or not values
                    or any(type(item) is not tuple or not item
                           for item in values)
                    or any(type(value) is not int or value < 0
                           for item in values for value in item)
                    or values != tuple(sorted(set(values)))):
                raise ValueError(
                    f"ConceptAnchor.{name} 必须是排序去重的非空整数 tuple 集")


@dataclass(frozen=True, slots=True)
class MemoryObservationAnchor:
    """从活动 Memory manifest 恢复的 Observation 根，不携带原文。"""

    observation_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    source_hash: int
    turn_seq: int
    speaker_kind: int
    hypothesis_keys: tuple[tuple[int, ...], ...]
    relation_refs: tuple[MemoryLinkedRef, ...] = ()
    context_depth: int = 0


@dataclass(frozen=True, slots=True)
class MemoryHypothesisAnchor:
    """已与当前输入拓扑匹配的开放 Memory Hypothesis。"""

    observation_key: tuple[int, ...]
    hypothesis_key: tuple[int, ...]
    hypothesis_kind: tuple[int, ...]
    candidate_key: tuple[int, ...]
    competition_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MemoryEvidenceAnchor:
    """与一个活动 Hypothesis 相连的来源化 Evidence。"""

    evidence_key: tuple[int, ...]
    hypothesis_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    stance: int
    detail: tuple[int, ...]
    payload_key: tuple[int, ...]
    supersedes_key: tuple[int, ...]


class _IntegerQueryTrace(QueryTracer):
    """记录共同 frontier/visited/root 的纯整数状态推进。"""

    def __init__(self, *, reference_mode: bool = False) -> None:
        if type(reference_mode) is not bool:
            raise TypeError("reference_mode 必须是 bool")
        self.reference_mode = reference_mode
        self.steps: list[dict[str, object]] = []
        # QueryState 中同一长整数身份会在多轮 frontier/visited/binding/
        # evidence 中反复出现。表只登记一次，步骤保留引用号，完整键仍可
        # 由消费者无损恢复；这避免把结构化 trace 误当作重复数据写入。
        self._key_ids: dict[tuple[int, ...], int] = {}
        self._key_table: list[tuple[int, ...]] = []
        self._legacy_item_keys: dict[object, tuple[int, ...]] = {}
        self._item_refs: dict[int, tuple[object, int]] = {}

    def _legacy_key(self, item) -> tuple[int, ...]:
        key = self._legacy_item_keys.get(item)
        if key is None:
            key = item.stable_key()
            self._legacy_item_keys[item] = key
        return key

    def _ref(self, key: tuple[int, ...]) -> int:
        if type(key) is not tuple or any(type(item) is not int for item in key):
            raise TypeError("Query trace key 必须是整数 tuple")
        ref = self._key_ids.get(key)
        if ref is None:
            ref = len(self._key_table) + 1
            self._key_ids[key] = ref
            self._key_table.append(key)
        return ref

    def _item_ref(self, item) -> int:
        identity = id(item)
        cached = self._item_refs.get(identity)
        if cached is not None and cached[0] is item:
            return cached[1]
        value = self._ref(item.stable_key())
        # 保留对象强引用直到本次 trace 结束，避免 CPython id 复用；对象
        # identity 只用于运行期缓存，输出引用仍完全由稳定整数键决定。
        self._item_refs[identity] = (item, value)
        return value

    def record(self, state: QueryState, reason: int) -> None:
        if not self.reference_mode:
            self.steps.append({
                "reason": reason,
                "depth": state.depth,
                "frontier": tuple(self._legacy_key(item)
                                  for item in state.frontier),
                "frontier_priority": [
                    [
                        item.owner_space,
                        item.owner_weight,
                        item.depth,
                        item.required_slot_gain,
                        item.evidence_support,
                        item.relation_fit,
                        item.discourse_fit,
                        item.source_trust,
                        item.recency_weight,
                    ]
                    for item in state.frontier
                ],
                "visited": tuple(self._legacy_key(item)
                                 for item in state.visited),
                "bindings": tuple(self._legacy_key(item)
                                  for item in state.bindings),
                "evidence": tuple(self._legacy_key(item)
                                  for item in state.evidence),
                "roots": tuple(item.stable_key() for item in state.roots),
                "termination_state": [
                    state.required_slots_open,
                    state.evidence_closed,
                    state.conflict_open,
                    state.best_score,
                    state.second_score,
                    len(state.best_candidate_key),
                    *state.best_candidate_key,
                    len(state.second_candidate_key),
                    *state.second_candidate_key,
                    state.marginal_gain,
                    state.cycle_hit,
                    state.generation_ready,
                ],
            })
            return
        self.steps.append({
            "reason": reason,
            "depth": state.depth,
            "frontier_refs": tuple(self._item_ref(item)
                                    for item in state.frontier),
            # 与边身份逐项对应，便于重放共同 frontier 的整数排序依据。
            "frontier_priority": [
                [
                    item.owner_space,
                    item.owner_weight,
                    item.depth,
                    item.required_slot_gain,
                    item.evidence_support,
                    item.relation_fit,
                    item.discourse_fit,
                    item.source_trust,
                    item.recency_weight,
                ]
                for item in state.frontier
            ],
            "visited_refs": tuple(self._item_ref(item)
                                  for item in state.visited),
            "binding_refs": tuple(self._item_ref(item)
                                   for item in state.bindings),
            "evidence_refs": tuple(self._item_ref(item)
                                    for item in state.evidence),
            "root_refs": tuple(self._item_ref(item) for item in state.roots),
            "termination_state": [
                state.required_slots_open,
                state.evidence_closed,
                state.conflict_open,
                state.best_score,
                state.second_score,
                self._ref(state.best_candidate_key)
                if state.best_candidate_key else 0,
                self._ref(state.second_candidate_key)
                if state.second_candidate_key else 0,
                state.marginal_gain,
                state.cycle_hit,
                state.generation_ready,
            ],
        })

    def export(self) -> dict[str, object] | tuple[dict[str, object], ...]:
        """输出无损整数引用 trace；表项按首次出现顺序稳定登记。"""
        if not self.reference_mode:
            return tuple(self.steps)
        return {
            "format": "PURE_INTEGER_QUERY_TRACE_TABLE_V1",
            "integer_key_table": [list(key) for key in self._key_table],
            "steps": list(self.steps),
        }


class TrainedGraphQueryBridge:
    """Core、Interaction Memory、Dialogue/Companion 的统一查询桥。"""

    def __init__(
            self,
            database: str | Path,
            *,
            memory_database: str | Path | None = None,
            tenant_id: int = 1,
            user_id: int = 1,
            session_id: int = 1,
            space_weights: tuple[int, int, int] = _DEFAULT_SPACE_WEIGHTS,
            surface_generator: RelationGraphSurfaceGenerator | None = None,
            semantic_input_views: tuple[SemanticCandidateInputView, ...] = (),
            # Successor occurrence evidence is diagnostic/training-only and
            # is rejected here so it cannot enter a production QueryState.
            successor_evidence: bool = False,
            ) -> None:
        if (type(space_weights) is not tuple
                or len(space_weights) != 3
                or any(type(item) is not int or item <= 0
                       for item in space_weights)
                or not (space_weights[0] > space_weights[1]
                        > space_weights[2])):
            raise ValueError(
                "space_weights 必须是 Core>Memory>Dialogue 的三个正整数")
        self.space_weights = space_weights
        self.surface_generator = surface_generator
        self.core_runtime = TrainedRelationGraphRuntime(database)
        if type(successor_evidence) is not bool:
            raise ValueError("successor_evidence 必须是 bool")
        if successor_evidence:
            raise ValueError(
                "production TrainedGraphQueryBridge 禁止 successor evidence；"
                "请使用独立训练/诊断入口")
        # The production bridge has no successor runtime at all.  Dialogue
        # roots must come from trained Companion graph records and the shared
        # Memory O/H/E state, so an exact surface lookup cannot enter the
        # QueryState or its evidence/frontier.
        self.dialogue_successor_runtime = None
        self.discourse_topic_topology: DiscourseTopicGraphTopology = (
            read_discourse_topic_topology(
                self.core_runtime.context.graph_ontology,
                discourse_topic_predicate_identities(),
            )
        )
        self.event_time_topology: EventTimeGraphTopology = (
            read_event_time_topology(
                self.core_runtime.context.graph_ontology,
                event_time_structure_predicate_identities(),
                event_time_narrative_predicate_prefix(),
            )
        )
        adjacency: dict[tuple[int, ...], list[EventTimeGraphEdge]] = {}
        for topology_edge in self.event_time_topology.edges:
            adjacency.setdefault(topology_edge.subject_key, []).append(topology_edge)
            adjacency.setdefault(topology_edge.object_key, []).append(topology_edge)
        self.event_time_graph_adjacency = {
            key: tuple(sorted(set(values), key=lambda item: item.stable_key()))
            for key, values in adjacency.items()
        }
        self.event_time_generation_bindings: tuple[
            TrainedEventTimeGenerationBinding, ...] = (
                read_event_time_generation_bindings(self.event_time_topology))
        facts = self.core_runtime.active_surface_facts()
        self.facts = facts
        self.kdconv_runtime = None
        kdconv_facts = ()
        try:
            from pure_integer_ai.experiments.kdconv_structure_bridge import (
                _register_tables, load_kdconv_structure_runtime,
            )
            _register_tables(self.core_runtime.backend)
            self.kdconv_runtime = load_kdconv_structure_runtime(
                self.core_runtime.context)
            # Release restore keeps KdConv as an integer posting index.  Its
            # ActiveRelationSurface objects are page-in'ed only after this
            # query's exact subject sequence is known.
            kdconv_facts = (self.kdconv_runtime.facts
                            if not self.kdconv_runtime._lazy else ())
        except (KeyError, RuntimeError, ValueError):
            self.kdconv_runtime = None
            kdconv_facts = ()
        self.kdconv_facts = tuple(kdconv_facts)
        self._semantic_input_views = tuple(semantic_input_views)
        # The carrier bridge is an optional extension of a trained model.  It
        # is restored read-only from the same SQLite so a query can connect a
        # projection to the complete existing semantic identity instead of
        # treating the carrier payload as a new fact.
        self._artifact_semantic_bindings = self._restore_artifact_semantic_bindings()
        self._artifact_bridge_available = bool(self._artifact_semantic_bindings)
        graph_structures = RelationSurfaceStructureAdapter.from_facts(facts)
        kdconv_structures = RelationSurfaceStructureAdapter.from_facts(
            self.kdconv_facts)
        semantic_structures = SemanticCandidateStructureAdapter.from_views(
            semantic_input_views)
        self.input_projector = TrainedInputStructureProjector(tuple(sorted(
            set((*graph_structures, *semantic_structures, *kdconv_structures)),
            key=lambda item: item.stable_key(),
        )), concept_links=restore_input_concept_links(self.core_runtime),
            open_role_frames=restore_open_role_frames(
                self.core_runtime.active_surface_frames(), facts))
        # Memory 写入和查询共享同一个训练后输入投影器。这样 O/H/E 的候选键
        # 只来自已训练图拓扑，而不是历史原文、posting 或字符特征。
        self.memory = None
        self._owns_memory = memory_database is not None
        if memory_database is not None:
            self.memory = TrainedDialogueMemoryGraph(
                memory_database,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                input_projector=self.input_projector,
            )
        self.filler_edges: dict[tuple[int, ...], tuple[ActiveRelationSurface, ...]] = {}
        for fact in facts:
            seen: set[tuple[int, ...]] = set()
            for binding in fact.bindings:
                key = binding.filler.stable_key()
                if key in seen:
                    continue
                seen.add(key)
                self.filler_edges.setdefault(key, ())
                if fact not in self.filler_edges[key]:
                    self.filler_edges[key] = self.filler_edges[key] + (fact,)
        for fact in self.kdconv_facts:
            for binding in fact.bindings:
                key = binding.filler.stable_key()
                self.filler_edges.setdefault(key, ())
                if fact not in self.filler_edges[key]:
                    self.filler_edges[key] = self.filler_edges[key] + (fact,)
        self._memory_observations: tuple[MemoryObservationAnchor, ...] = ()
        self._memory_hypotheses: tuple[MemoryHypothesisAnchor, ...] = ()
        self._memory_evidence: tuple[MemoryEvidenceAnchor, ...] = ()
        self._discourse_hypotheses: tuple[MemoryHypothesisAnchor, ...] = ()
        self._discourse_topic_candidates: tuple[DiscourseTopicCandidate, ...] = ()
        self._discourse_topic_graph_identity_keys: tuple[tuple[int, ...], ...] = ()
        self._discourse_topic_hypothesis_keys: tuple[tuple[int, ...], ...] = ()
        self._event_time_graph_input_keys: tuple[tuple[int, ...], ...] = ()
        self._core_graph_input_keys: tuple[tuple[int, ...], ...] = ()
        self._generic_graph_input_keys: tuple[tuple[int, ...], ...] = ()
        self._memory_graph_input_keys: tuple[tuple[int, ...], ...] = ()
        # Preserve the caller's exact integer graph identities separately from
        # the topology-expanded roots.  A frame input must remain auditable
        # even when its propositions are also seeded into the same frontier.
        self._discourse_graph_input_requested_keys: tuple[tuple[int, ...], ...] = ()
        self._discourse_graph_input_keys: tuple[tuple[int, ...], ...] = ()
        self._event_time_query_bindings: tuple[
            TrainedEventTimeGenerationBinding, ...] = ()
        self._event_time_query_frame_keys: frozenset[tuple[int, ...]] = frozenset()
        self._event_time_query_edge_keys: frozenset[tuple[int, ...]] = frozenset()
        self._memory_routes: tuple[tuple[int, ...], ...] = ()
        # Core proposition 与 Dialogue realization frame 的同图结构索引。
        self.core_fact_index: dict[tuple[int, ...], ActiveRelationSurface] = {}
        for fact in facts:
            self.core_fact_index[fact.proposition.stable_key()] = fact
        self.kdconv_fact_index = {
            fact.proposition.stable_key(): fact for fact in self.kdconv_facts}
        self.core_fact_index.update(self.kdconv_fact_index)
        self.dialogue_frame_index = {
            frame.proposition.stable_key(): frame
            for frame in self.core_runtime.active_surface_frames()
        }
        self.dialogue_frame_index.update({
            fact.proposition.stable_key(): fact
            for fact in self.kdconv_facts
        })
        # Frame registration is a graph property, independent of whether the
        # caller injects an optional surface renderer for this query.
        generation_keys = frozenset(self.dialogue_frame_index)
        self.relation_routes = RelationCapabilityRouteRegistry(
            facts,
            scope_keys={
                fact.proposition.stable_key(): self.core_runtime.generation_input(
                    fact.proposition).hypothesis.scope.stable_key()
                for fact in facts
            },
            generation_keys=generation_keys,
        )
        self._event_time_bindings_by_proposition = (
            self._validate_event_time_generation_bindings())
        self.typed_relation_projections: dict[
            tuple[int, ...], TypedRelationProjection] = {}
        self.generation_by_proposition: dict[
            tuple[int, ...], GraphRelationGeneration] = {}
        self.generation_ready_propositions: set[tuple[int, ...]] = set()
        self._generation_attempted: set[tuple[int, ...]] = set()
        self._pending_response_delivery = None
        self._input_observation_ref = None
        self._response_contexts = ()
        self._delivered_graph_slot_projections: tuple[
            DeliveredGraphSlotProjection, ...] = ()
        self._input_role_occurrences: tuple[InputRoleOccurrence, ...] = ()
        self._reference_candidate_sets: tuple[ReferenceCandidateSet, ...] = ()
        self._reference_resolutions: tuple[CrossTurnReferenceResolution, ...] = ()
        self._artifact_bridge_matches: tuple[object, ...] = ()
        self._artifact_bridge_missing: tuple[tuple[int, ...], ...] = ()
        self._reference_runtime = (
            None if self.memory is None else TrainedCrossTurnReferenceRuntime(
                self.core_runtime, self.input_projector, self.memory.occurrence_index))
        self._discourse_topic_query_projection: tuple[int, ...] = ()
        self._kdconv_query_propositions: set[tuple[int, ...]] = set()
        self._restored_factual_topic_keys: tuple[tuple[int, ...], ...] = ()

    def _activate_kdconv_facts(self, values: tuple[int, ...]) -> None:
        """按当前输入整数命中并接入 KdConv 事实，不恢复无关事实。"""
        runtime = self.kdconv_runtime
        if runtime is None or not runtime._lazy:
            return
        self._register_kdconv_facts(runtime.page_in_for_values(values))

    def _activate_kdconv_graph_objects(
            self,
            graph_object_keys: tuple[tuple[int, ...], ...],
            ) -> None:
        """按显式图身份 page-in 已学事实，再进入共同 QueryState。"""
        runtime = self.kdconv_runtime
        if runtime is None or not runtime._lazy or not graph_object_keys:
            return
        self._register_kdconv_facts(
            runtime.page_in_for_graph_object_keys(graph_object_keys))

    def _register_kdconv_facts(
            self,
            loaded: tuple[ActiveRelationSurface, ...],
            ) -> None:
        """把按需恢复的事实注册到现有 Core/输入/生成索引。"""
        if not loaded:
            return
        runtime = self.kdconv_runtime
        if runtime is None:
            raise RuntimeError("KdConv fact 注册缺少 runtime owner")
        for fact in loaded:
            key = fact.proposition.stable_key()
            self.kdconv_fact_index[key] = fact
            self.core_fact_index[key] = fact
            self.dialogue_frame_index[key] = fact
            for binding in fact.bindings:
                filler = binding.filler.stable_key()
                current = self.filler_edges.setdefault(filler, ())
                if fact not in current:
                    self.filler_edges[filler] = current + (fact,)
        self.kdconv_facts = tuple(sorted(
            runtime.facts, key=lambda item: item.proposition.stable_key()))
        graph_structures = RelationSurfaceStructureAdapter.from_facts(self.facts)
        kdconv_structures = RelationSurfaceStructureAdapter.from_facts(
            self.kdconv_facts)
        semantic_structures = SemanticCandidateStructureAdapter.from_views(
            self._semantic_input_views)
        concept_links = self.input_projector.concept_graph.links
        open_frames = self.input_projector.open_role_frames
        self.input_projector = TrainedInputStructureProjector(tuple(sorted(
            set((*graph_structures, *semantic_structures, *kdconv_structures)),
            key=lambda item: item.stable_key(),
        )), concept_links=concept_links, open_role_frames=open_frames)
        if self.memory is not None:
            self.memory.input_projector = self.input_projector
        if self._reference_runtime is not None:
            self._reference_runtime.input_projector = self.input_projector

    def _restore_artifact_semantic_bindings(self) -> tuple[object, ...]:
        """从训练库只读恢复 carrier projection 的完整语义桥引用。

        旧模型可能没有 bridge 扩展表，此时返回空集合；只要扩展表存在，
        任一 hash、parts、active proposition 或 Evidence 漂移都立即失败，
        不能以摘要或缺失字段继续查询。
        """
        try:
            from pure_integer_ai.experiments.artifact_semantic_training_bridge import (
                ArtifactSemanticBinding,
                ArtifactSemanticBridgeError,
                BRIDGE_PART_TABLE,
                BRIDGE_STATE_BOUND,
                BRIDGE_TABLE,
                _evidence_event_signature,
                _key_hash,
                _register_bridge_tables,
            )
            from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
        except ImportError:
            return ()
        try:
            _register_bridge_tables(self.core_runtime.backend)
        except RuntimeError as error:
            # A pre-bridge model has no physical extension table.  Other
            # schema failures must remain visible rather than being hidden as
            # an ordinary semantic miss.
            if "只读 SQLite 缺少表或列" in str(error):
                return ()
            raise
        rows = self.core_runtime.backend.select(BRIDGE_TABLE)
        if not rows:
            return ()
        # The bridge part table is large (139k+ integer rows) while the binding
        # table is intentionally small. Read parts once and group by binding;
        # repeated WHERE scans made cold startup scale with binding count.
        all_parts = self.core_runtime.backend.select(
            BRIDGE_PART_TABLE, order_by="binding_id")
        parts_by_binding: dict[int, list[dict[str, int]]] = {}
        for part in all_parts:
            parts_by_binding.setdefault(part["binding_id"], []).append(part)
        facts = {item.proposition.stable_key(): item for item in self.facts}
        bindings = []
        for row in sorted(rows, key=lambda item: item["binding_id"]):
            binding_id = row["binding_id"]
            parts = parts_by_binding.get(binding_id, ())
            grouped: dict[int, list[tuple[int, int]]] = {}
            for part in parts:
                grouped.setdefault(part["field_code"], []).append(
                    (part["part_ordinal"], part["part_value"]))

            def field(code: int, *, allow_empty: bool = False) -> tuple[int, ...]:
                values = grouped.get(code, ())
                ordered = tuple(value for _ordinal, value in sorted(values))
                if not ordered and not allow_empty:
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge 缺少完整 integer part")
                if tuple(ordinal for ordinal, _value in sorted(values)) != tuple(range(len(values))):
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge part ordinal 不连续")
                return ordered

            evidence_codes = sorted(code for code in grouped if code >= 100)
            if evidence_codes:
                evidence = tuple(field(code) for code in evidence_codes)
            else:
                # The first formal bridge schema stored the evidence count in
                # the row but not the large evidence-key parts.  Recover those
                # complete keys from the authoritative active proposition and
                # require an exact count; never treat the count/hash as the
                # evidence itself.
                proposition_key = field(3)
                fact = facts.get(proposition_key)
                if fact is None:
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge 缺少可恢复的 active proposition")
                history = tuple(sorted(
                    item.stable_key()
                    for item in self.core_runtime.evidence_history(
                        fact.proposition)))
                if len(history) != row["evidence_count"]:
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge evidence_count 与 active history 不一致")
                evidence = history
            if len(evidence) != row["evidence_count"]:
                raise ArtifactSemanticBridgeError(
                    "只读 bridge Evidence parts 数量与 evidence_count 不一致")
            binding = ArtifactSemanticBinding(
                row["carrier_code"], row["fact_ordinal"],
                row["projection_ordinal"], field(1), field(2), field(3),
                field(4), field(5), evidence, field(7),
                row["binding_state"],
            )
            expected_id = _key_hash(
                binding.projection_key, binding.semantic_object_key,
                binding.proposition_key)
            if binding_id != expected_id:
                raise ArtifactSemanticBridgeError(
                    "只读 bridge binding_id 与完整 identity 不一致")
            expected_hashes = {
                "projection_hash": _key_hash(binding.projection_key),
                "semantic_hash": _key_hash(binding.semantic_object_key),
                "proposition_hash": _key_hash(binding.proposition_key),
                "source_hash": _key_hash(binding.source_ref),
                "evidence_count": len(binding.evidence_keys),
            }
            if any(row[name] != value for name, value in expected_hashes.items()):
                raise ArtifactSemanticBridgeError(
                    "只读 bridge 摘要与完整 integer identity 不一致")
            if binding.binding_state == BRIDGE_STATE_BOUND:
                fact = facts.get(binding.proposition_key)
                if fact is None:
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge 指向非 active proposition")
                allowed = {
                    fact.predicate.stable_key(),
                    *(item.filler.stable_key() for item in fact.bindings),
                }
                if binding.semantic_object_key not in allowed:
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge semantic object 不是 proposition predicate/filler")
                active_history = self.core_runtime.evidence_history(
                    fact.proposition)
                # Carrier projections intentionally remap the Hypothesis
                # identity to a carrier-local semantic projection.  Validate
                # the complete underlying Evidence event instead of requiring
                # the remapped stable key to equal Core's Hypothesis key.
                try:
                    projection_evidence = tuple(
                        EvidenceRecord.from_stable_key(key)
                        for key in binding.evidence_keys)
                except (TypeError, ValueError) as error:
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge Evidence stable key 非法") from error
                if len(projection_evidence) != len(active_history):
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge Evidence 数量与 active proposition 历史不一致")
                projection_signatures = {
                    _evidence_event_signature(item)
                    for item in projection_evidence
                }
                active_signatures = {
                    _evidence_event_signature(item)
                    for item in active_history
                }
                if projection_signatures != active_signatures:
                    raise ArtifactSemanticBridgeError(
                        "只读 bridge Evidence 不属于 active proposition 历史")
            bindings.append(binding)
        natural = tuple(item.natural_key() for item in bindings)
        if len(set(natural)) != len(natural):
            raise ArtifactSemanticBridgeError("只读 bridge natural key 重复")
        return tuple(bindings)

    def _match_artifact_semantic_bindings(
            self, artifact_input: ArtifactQueryInput,
            ) -> tuple[tuple[object, ...], tuple[tuple[int, ...], ...]]:
        """按完整 projection identity 连接 carrier 与训练后 semantic graph。"""
        by_projection = {
            item.projection_key: item
            for item in self._artifact_semantic_bindings
            if getattr(item, "binding_state", 0) == 1
        }
        matched = []
        missing = []
        for projection in artifact_input.understanding_projections:
            key = projection.identity.stable_key()
            binding = by_projection.get(key)
            if binding is None:
                missing.append(key)
                continue
            if binding.semantic_object_key != projection.semantic_object.stable_key():
                raise ValueError("carrier projection 与训练 bridge semantic identity 不一致")
            if binding.proposition_key not in self.core_fact_index:
                raise ValueError("carrier projection bridge proposition 不在 active Core")
            matched.append(binding)
        return tuple(sorted(matched, key=lambda item: item.natural_key())), tuple(sorted(missing))

    def _validate_event_time_generation_bindings(
            self,
            ) -> dict[tuple[int, ...], tuple[TrainedEventTimeGenerationBinding, ...]]:
        """核验 Event/Time frame 精确复用当前 active fact 和 generation 对象。"""
        grouped: dict[
            tuple[int, ...], list[TrainedEventTimeGenerationBinding]] = {}
        ontology = self.core_runtime.context.graph_ontology
        topology_edges = {
            item.stable_key() for item in self.event_time_topology.edges}
        for binding in self.event_time_generation_bindings:
            fact = self.core_fact_index.get(binding.proposition_key)
            if fact is None or fact.predicate.stable_key() != binding.predicate_key:
                raise ValueError("Event/Time generation binding 未引用 active Core fact")
            route = self.relation_routes.for_proposition(binding.proposition_key)
            if (route is None or route.relation_kind not in {10, 11, 12, 13}
                    or route.generation_registered != 1):
                raise ValueError("Event/Time generation binding 未闭合 EVENT_TIME route")
            generation_input = self.core_runtime.generation_input(fact.proposition)
            definition = generation_input.proposition.definition
            structure = RelationSurfaceStructureAdapter.from_facts(
                (fact,))[0].relation_structure_key()
            if (event_time_input_structure_identity(structure).stable_key()
                    != binding.input_structure_key):
                raise ValueError("Event/Time generation binding 输入拓扑漂移")
            expected_roles = tuple(sorted(
                item.identity_for(definition.proposition).stable_key()
                for item in definition.bindings))
            expected_events = tuple(sorted(
                item.filler.stable_key() for item in definition.bindings
                if item.filler.object_kind in {OBJECT_EVENT, OBJECT_PROPOSITION}))
            if (binding.role_binding_keys != expected_roles
                    or binding.event_keys != expected_events
                    or binding.context_key != definition.context.stable_key()
                    or binding.source_ref != definition.source.stable_key()
                    or binding.scope_key
                    != generation_input.proposition.scope.stable_key()):
                raise ValueError("Event/Time generation binding 成员/source/scope 漂移")
            if (ontology.resolve(ObjectIdentity.from_stable_key(
                    binding.connector_key)) is None):
                raise ValueError("Event/Time generation binding connector 无法回读")
            if not set(binding.topology_edge_keys) <= topology_edges:
                raise ValueError("Event/Time generation binding 拓扑证据不完整")
            grouped.setdefault(binding.proposition_key, []).append(binding)
        return {
            key: tuple(sorted(values, key=lambda item: item.stable_key()))
            for key, values in grouped.items()
        }

    def close(self) -> None:
        """关闭只读 Core 与可选 Memory owner。"""
        self.core_runtime.close()
        if self.memory is not None:
            self.memory.close()
            self.memory = None

    def __enter__(self) -> "TrainedGraphQueryBridge":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _restore_memory_structures(
            self,
            input_structure: QueryInputStructure,
            artifact_input: ArtifactQueryInput | None = None,
            ) -> MemoryActiveStructures:
        """恢复和当前训练结构匹配的活动 O/H/E，不读表层或 posting。

        Memory 的语义入口是候选及共同图节点/命题引用。未匹配的活动记录
        仍保留在其 session 图中，但不会成为本请求
        的语义 root；调用方会显式留下一个 ``ROOT_EMPTY`` 证明该空间参加了查询。
        """
        empty = MemoryActiveStructures((), (), ())
        self._memory_observations = ()
        self._memory_hypotheses = ()
        self._memory_evidence = ()
        self._discourse_hypotheses = ()
        self._discourse_topic_candidates = derive_discourse_topic_candidates(
            input_structure)
        self._discourse_topic_graph_identity_keys = graph_backed_discourse_topic_keys(
            self.core_runtime.context.graph_ontology,
            self._discourse_topic_candidates,
        )
        self._discourse_topic_hypothesis_keys = ()
        self._memory_routes = ()
        self._response_contexts = ()
        self._delivered_graph_slot_projections = ()
        self._input_role_occurrences = ()
        self._reference_candidate_sets = ()
        self._reference_resolutions = ()
        if self.memory is None:
            return empty
        candidate_keys = tuple(sorted({
            *memory_input_candidate_keys(
                input_structure,
                session_id=self.memory.session_id,
                speaker_kind=1),
            *(() if artifact_input is None
              else artifact_input.memory_candidate_keys()),
        }))
        selected = self.memory.active_structures(
            candidate_keys=candidate_keys,
            observation_refs=(() if self._input_observation_ref is None else (self._input_observation_ref,)))
        self._response_contexts = restore_response_contexts(self.memory, selected)
        self._delivered_graph_slot_projections = (
            delivered_graph_slot_projections(self._response_contexts))
        if self._input_observation_ref is not None:
            candidates = {
                frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,),
                          item.stable_key()): item
                for item in input_structure.open_relations}
            occurrences = []
            for item in selected.hypotheses:
                candidate = candidates.get(item.candidate_key)
                if candidate is None or item.observation_key != self._input_observation_ref.stable_key():
                    continue
                occurrences.extend(materialize_input_roles(
                    self.memory, candidate, self._input_observation_ref,
                    MemoryObjectRef.from_stable_key(item.hypothesis_key)))
            self._input_role_occurrences = tuple(sorted(
                {item.stable_key(): item for item in occurrences}.values(),
                key=lambda item: item.stable_key()))
            prior_depths = {}
            prior_by_key = {}
            for context in self._response_contexts:
                for occurrence in context.role_occurrences:
                    key = occurrence.stable_key()
                    prior_by_key[key] = occurrence
                    prior_depths[key] = min(
                        prior_depths.get(key, context.context_depth), context.context_depth)
            candidate_sets = []
            for reference in self._input_role_occurrences:
                antecedents = tuple(prior_by_key[key] for key in sorted(prior_by_key)
                                    if prior_by_key[key].source_ref != reference.source_ref)
                if antecedents:
                    candidate_sets.append(ReferenceCandidateSet(
                        reference, antecedents,
                        tuple(prior_depths[item.stable_key()] for item in antecedents)))
            self._reference_candidate_sets = tuple(sorted(
                {item.stable_key(): item for item in candidate_sets}.values(),
                key=lambda item: item.stable_key()))
        self._memory_routes = selected.match_routes
        current_observation_key = (
            None if self._input_observation_ref is None
            else self._input_observation_ref.stable_key())
        self._memory_graph_input_keys = tuple(sorted({
            graph_key
            for item in selected.hypotheses
            if (current_observation_key is None
                or item.observation_key == current_observation_key)
            for graph_key in (memory_graph_input_object_key(
                item.candidate_key),)
            if graph_key is not None
        }))
        matched_hypotheses = {
            item.hypothesis_key: item for item in selected.hypotheses}
        matched_evidence = {
            item.evidence_key: item for item in selected.evidence}
        observation_keys = {item.observation_key for item in selected.observations}
        if any(item.observation_key not in observation_keys
               or any(key not in matched_evidence for key in item.evidence_keys)
               for item in selected.hypotheses):
            raise RuntimeError("Memory 结构恢复缺少 Observation/Evidence 连接")
        self._memory_observations = tuple(
            MemoryObservationAnchor(
                item.observation_key,
                item.source_ref,
                item.scope_key,
                item.source_hash,
                item.turn_seq,
                item.speaker_kind,
                tuple(key for key in item.hypothesis_keys
                      if key in matched_hypotheses),
                item.relation_refs,
                item.context_depth,
            )
            for item in selected.observations
        )
        self._memory_hypotheses = tuple(
            MemoryHypothesisAnchor(
                item.observation_key,
                item.hypothesis_key,
                item.hypothesis_kind,
                item.candidate_key,
                item.competition_key,
                item.source_ref,
                item.scope_key,
                tuple(key for key in item.evidence_keys
                      if key in matched_evidence),
            )
            for item in selected.hypotheses
        )
        self._memory_evidence = tuple(
            MemoryEvidenceAnchor(
                item.evidence_key,
                item.hypothesis_key,
                item.source_ref,
                item.scope_key,
                item.stance,
                item.detail,
                item.payload_key,
                item.supersedes_key,
            )
            for item in selected.evidence
        )
        self._discourse_hypotheses = tuple(
            item for item in self._memory_hypotheses
            if item.hypothesis_kind and item.hypothesis_kind[-1] in {
                4, 7, MEMORY_CATEGORY_OPEN_RELATION, MEMORY_CATEGORY_RESPONSE,
                MEMORY_CATEGORY_ARTIFACT}
            or (item.candidate_key
                and item.candidate_key[0] == MEMORY_GENERIC_CANDIDATE_VERSION)
        )
        # Current-input topics and topics recovered through the explicit
        # Observation/Memory closure share one candidate set.  The latter are
        # cold context, not a fallback source: their hypothesis/evidence roots
        # are already in this same QueryState and are ranked by depth below.
        self._discourse_topic_candidates = restore_discourse_topic_candidates(
            self._memory_hypotheses, self._discourse_topic_candidates)
        self._discourse_topic_graph_identity_keys = graph_backed_discourse_topic_keys(
            self.core_runtime.context.graph_ontology,
            self._discourse_topic_candidates,
        )
        self._discourse_topic_hypothesis_keys = matched_discourse_topic_hypothesis_keys(
            self._memory_hypotheses, self._discourse_topic_candidates)
        return selected

    def _reference_timestamp(self) -> int:
        """Return the current Observation sequence without using wall-clock time."""
        if self._input_observation_ref is None:
            return 0
        matches = tuple(item.turn_seq for item in self._memory_observations
                        if item.observation_key == self._input_observation_ref.stable_key())
        if len(matches) != 1:
            raise ValueError("current reference query has no unique Observation sequence")
        return matches[0]

    def _refresh_reference_resolutions(self, state: QueryState) -> QueryState:
        """Resolve only from Core facts already expanded in this QueryState."""
        if self._reference_runtime is None or not self._reference_candidate_sets:
            self._reference_resolutions = ()
            return state
        current = current_query_evidence(state.evidence)
        core_support = frozenset(
            item.hypothesis_key for item in current
            if item.space == _CORE and item.polarity == 1
        )
        resolutions = tuple(sorted((
            self._reference_runtime.resolve(
                candidate_set,
                self._response_contexts,
                core_support=core_support,
                timestamp_seq=self._reference_timestamp(),
            )
            for candidate_set in self._reference_candidate_sets
        ), key=lambda item: item.stable_key()))
        self._reference_resolutions = resolutions
        bindings = tuple(sorted({
            *state.bindings,
            *(binding for resolution in resolutions
              for binding in resolution.query_bindings()),
        }, key=lambda item: item.stable_key()))
        evidence = tuple(sorted({
            *state.evidence,
            *(entry for resolution in resolutions
              for entry in resolution.query_evidence()),
        }, key=lambda item: item.stable_key()))
        return state.with_(bindings=bindings, evidence=evidence)

    def _reference_generation_contexts(self) -> tuple:
        """Select an exact topic context only after an evidence-qualified winner."""
        required = tuple(item for item in self._reference_resolutions
                         if item.requires_resolution)
        if not required:
            return self._response_contexts
        if any(item.status != REFERENCE_RESOLVED for item in required):
            return ()
        keys = {item.winner_context_key for item in required}
        selected = tuple(item for item in self._response_contexts
                         if item.stable_key() in keys)
        if {item.stable_key() for item in selected} != keys:
            raise ValueError("reference winner topic context is not recoverable")
        return tuple(sorted(selected, key=lambda item: item.stable_key()))

    def _generic_response_feature_mask(self, hypotheses,
                                       input_structure=None) -> int:
        """Derive response-act conditions from the current shared graph view."""
        mask = RESPONSE_FEATURE_UNKNOWN
        # Generic Memory candidates are deliberately present for every unseen
        # input.  They must not manufacture a graph condition merely because
        # the projector emitted an open/partial relation shape.  Only a
        # relation whose trained predicate and all required fillers are closed
        # is a current-input graph signal for connector priority.
        complete_relation = bool(input_structure and any(
            item.state == STRUCTURE_CLOSED
            and item.required_count >= 2
            and item.required_count == item.support_count
            and item.predicate_coverage == 1
            for item in input_structure.relation_candidates))
        if self._response_contexts:
            # A shallow context must not permanently mask a deeper context
            # restored through the same Memory O/H/E closure: doing so makes
            # the trained COLD condition unreachable after the first turn.
            # Context depth is an already verified integer field; no surface
            # or recency text is inspected here.
            deepest = max(
                self._response_contexts,
                key=lambda item: (item.context_depth, item.stable_key()),
            )
            if deepest.context_depth <= 1:
                mask |= RESPONSE_FEATURE_HOT_CONTEXT
            else:
                mask |= RESPONSE_FEATURE_COLD_CONTEXT
        if self._discourse_topic_candidates:
            mask |= RESPONSE_FEATURE_TOPIC
        if self._discourse_topic_graph_identity_keys:
            mask |= RESPONSE_FEATURE_TOPIC_GRAPH
        if self._reference_candidate_sets:
            mask |= RESPONSE_FEATURE_REFERENCE_OPEN
        if len(tuple(hypotheses)) > 1:
            mask |= RESPONSE_FEATURE_MULTI_HYPOTHESIS
        relation_state = self._discourse_relation_feature_state(input_structure)
        if not complete_relation and not self._discourse_graph_input_keys:
            relation_state = (0, 0, 0, 0, 0, 0, 0)
        if relation_state[0]:
            mask |= RESPONSE_FEATURE_DISCOURSE_RELATION
        if relation_state[1]:
            mask |= RESPONSE_FEATURE_RELATION_CHAIN
        if relation_state[2]:
            mask |= RESPONSE_FEATURE_QUD
        if relation_state[3]:
            mask |= RESPONSE_FEATURE_REVISION
        if relation_state[4]:
            mask |= RESPONSE_FEATURE_RELATION_CONFLICT
        if ((complete_relation and self._event_time_query_bindings)
                or self._event_time_graph_input_keys):
            mask |= RESPONSE_FEATURE_EVENT_TIME
        # Generic graph roots are consumed by the same Core frontier and must
        # contribute the same graph-derived response conditions.  Keep the
        # source strictly in the integer identities already accepted by this
        # QueryState; no surface or fallback inference is involved.
        core_graph_input_kinds = {
            ObjectIdentity.from_stable_key(key).object_kind
            for key in (*self._core_graph_input_keys,
                        *self._generic_graph_input_keys)
        }
        if OBJECT_ENTITY in core_graph_input_kinds:
            mask |= RESPONSE_FEATURE_ENTITY
        if OBJECT_EVENT in core_graph_input_kinds:
            mask |= RESPONSE_FEATURE_EVENT
        if core_graph_input_kinds:
            mask |= RESPONSE_FEATURE_PROPERTY
        # A proposition graph root carries its ordered Core relation members;
        # derive their object-kind signal in this QueryState before connector
        # selection.  The root identity remains Proposition, so looking only
        # at root kinds previously hid the Entity condition of structural
        # graph-slot courses.  This reads restored Core facts only and never
        # inspects surface text or a vocabulary.
        explicit_graph_facts = tuple(
            self.core_fact_index.get(key)
            for key in (*self._core_graph_input_keys,
                        *self._generic_graph_input_keys,
                        *self._discourse_graph_input_keys)
            if key in self.core_fact_index
        )
        explicit_member_kinds = {
            binding.filler.object_kind
            for fact in explicit_graph_facts
            for binding in fact.bindings
        }
        if OBJECT_ENTITY in explicit_member_kinds:
            mask |= RESPONSE_FEATURE_ENTITY
        if OBJECT_EVENT in explicit_member_kinds:
            mask |= RESPONSE_FEATURE_EVENT
        if explicit_member_kinds & {OBJECT_ENTITY, OBJECT_EVENT}:
            mask |= RESPONSE_FEATURE_PROPERTY
        # Entity/event/property bits are graph-derived conditions.  Exact
        # trained filler spans may identify a known entity or event even when
        # the surrounding relation is still open.  PROPERTY means that the
        # same graph projection produced a trained relation candidate; it
        # does not assert that an open property is true.  Arbitrary input code
        # points therefore never create these features.
        if input_structure is not None:
            relation_propositions = {
                item.proposition_key
                for item in input_structure.relation_candidates
            }
            filler_kinds = {
                item.object_kind for item in input_structure.semantic_candidates
                if item.projection_kind == PROJECTION_FILLER
            }
            if complete_relation and OBJECT_ENTITY in filler_kinds:
                mask |= RESPONSE_FEATURE_ENTITY
            if complete_relation and (
                    OBJECT_EVENT in filler_kinds
                    or bool(self._event_time_query_bindings)):
                mask |= RESPONSE_FEATURE_EVENT
            # Property candidates are represented by the trained
            # filler/predicate topology.  They may guide an UNKNOWN response
            # act but cannot close Evidence or create a Core assertion.
            if complete_relation and relation_propositions and any(
                    item.projection_kind == PROJECTION_FILLER
                    and item.proposition_key in relation_propositions
                    for item in input_structure.semantic_candidates):
                mask |= RESPONSE_FEATURE_PROPERTY
        if (self._reference_resolutions
                and all(item.status == REFERENCE_RESOLVED
                        for item in self._reference_resolutions
                        if item.requires_resolution)):
            mask |= RESPONSE_FEATURE_REFERENCE_RESOLVED
        return mask

    def _discourse_relation_feature_state(self, input_structure=None):
        """Return integer-only W-08 relation state for this QueryState."""
        active = set()
        active.update(self._discourse_graph_input_keys)
        # A W-08 frame is itself a first-class Dialogue object.  Reference-
        # only and revision-only frames intentionally have no proposition
        # child; retain their frame identity so the same QueryState can
        # consume the trained marker edges instead of dropping them.
        frame_by_key = {
            frame.frame_key: frame
            for frame in self.discourse_topic_topology.frames
        }
        active_frames = {
            key: frame_by_key[key]
            for key in tuple(active)
            if key in frame_by_key
        }
        active.update(
            proposition_key
            for frame in active_frames.values()
            for proposition_key in frame.proposition_keys
        )
        if input_structure is not None:
            active.update(item.proposition_key
                          for item in input_structure.relation_candidates)
        active.update(item.proposition_key for item in self._discourse_topic_candidates)
        if not active:
            return (0, 0, 0, 0, 0, 0, 0)
        relations = tuple(item for item in self.discourse_topic_topology.relations
                          if item.subject_key in active or item.object_key in active)
        linked = tuple(item for item in relations
                       if item.subject_key in active and item.object_key in active)
        frames = tuple(
            frame for frame in self.discourse_topic_topology.frames
            if frame.frame_key in active_frames
            or active.intersection(frame.proposition_keys)
        )
        qud = int(any(frame.qud_keys for frame in frames))
        revision = int(any(frame.revision_keys for frame in frames))
        reference = int(any(frame.reference_keys for frame in frames))
        conflict = int(any(
            item.state == STRUCTURE_ORDER_CONFLICT
            for item in (() if input_structure is None
                         else input_structure.relation_candidates)))
        conflict = int(conflict or any(
            item.status == REFERENCE_CONFLICT
            for item in self._reference_resolutions))
        return (int(bool(relations)), int(bool(linked) or len(relations) > 1),
                qud, revision, conflict, len(frames), reference)

    def _generate_fact(
            self, fact: ActiveRelationSurface,
            ) -> GraphRelationGeneration:
        """经调用方注入的训练后 connector 或 relation frame 生成。"""
        if self.surface_generator is None:
            # _generate_surface 只允许训练/审计代码显式调用。生产 QueryState
            # 没有训练 connector 时必须保持无生成结果，不能回到来源正文。
            raise RuntimeError("缺少训练后 generation connector，拒绝来源表层直答")
        generated = self.surface_generator.generate_relation(
            self.core_runtime.generation_input(fact.proposition),
            fact,
        )
        # strict connector 的 structural fallback 会直接发射来源 claim 包络；
        # 它没有 R-01 Representation，不具备重新组织资格，必须 fail closed。
        if generated.connector is None or not generated.representations:
            raise RuntimeError("connector generation 缺少图内 realization")
        return generated

    def _prepare_generation_plans(
            self,
            proposition_keys: tuple[tuple[int, ...], ...],
            state: QueryState,
            ) -> None:
        """在共同 frontier 内消费 Dialogue 生成结构；三图必须已登记根。"""
        from pure_integer_ai.experiments.generation_organization import (
            plan_from_active_fact,
        )
        if {item.owner_space for item in state.roots} != {_CORE, _MEMORY, _DIALOGUE}:
            raise ValueError("生成准备必须属于已登记完整三图根的 QueryState")
        dialogue_roots = {item.root_key for item in state.roots if item.owner_space == _DIALOGUE}
        if any(key not in dialogue_roots for key in proposition_keys):
            raise ValueError("生成准备不得绕过当前 Dialogue 根")
        for key in proposition_keys:
            if key in self._generation_attempted:
                continue
            self._generation_attempted.add(key)
            fact = self.core_fact_index.get(key)
            if fact is None or key not in self.dialogue_frame_index:
                continue
            try:
                if key in self.kdconv_fact_index and self.kdconv_runtime is not None:
                    generated = self.kdconv_runtime.generate(fact)
                else:
                    generated = self._generate_fact(fact)
                plan = plan_from_active_fact(fact, generated)
            except (RuntimeError, TypeError, ValueError):
                continue
            if plan is not None:
                self.generation_by_proposition[key] = generated
                self.generation_ready_propositions.add(key)

    @staticmethod
    def anchor_concepts(
            input_structure: QueryInputStructure,
            ) -> tuple[ConceptAnchor, ...]:
        """从输入结构的全部 filler 投影汇聚概念 anchor，不重新匹配字符串。"""
        spans = {item.ref_key: item for item in input_structure.spans}
        # Keep the complete projection in QueryInputStructure, but only make
        # fillers that participate in a complete trained relation into Core
        # roots.  Standalone one-codepoint postings are query-local evidence;
        # expanding each as a root multiplies the frontier without adding a
        # graph-backed proposition and was the source of multi-gigabyte query
        # state growth on common characters.
        complete_propositions = {
            item.proposition_key for item in input_structure.relation_candidates
            if (item.state == STRUCTURE_CLOSED
                and item.required_count >= 2
                and item.required_count == item.support_count
                and item.predicate_coverage == 1)
        }
        # An open-role parse may still carry many exact filler postings from
        # the same integer input.  Those postings remain in the complete
        # QueryInputStructure and Memory O/H/E, but without a closed trained
        # relation they are not Core roots.  Expanding them here turns an
        # unknown role query into a graph-wide traversal and exhausts the
        # shared frontier before Dialogue can consume the open frame.
        if not complete_propositions:
            return ()
        grouped: dict[tuple[int, ...], list[object]] = {}
        for candidate in input_structure.semantic_candidates:
            if (candidate.projection_kind != PROJECTION_FILLER
                    or candidate.concept_route_key
                    or candidate.proposition_key not in complete_propositions):
                continue
            grouped.setdefault(candidate.candidate_key, []).append(candidate)
        result = []
        for filler, candidates in grouped.items():
            span_refs = tuple(sorted({item.span_ref for item in candidates}))
            surface_values = min(
                (spans[span_ref].values for span_ref in span_refs),
                key=lambda item: (len(item), item),
            )
            result.append(ConceptAnchor(
                filler,
                surface_values,
                1,
                span_refs,
                tuple(sorted({
                    item.proposition_key for item in candidates})),
                tuple(sorted({item.source_ref for item in candidates})),
            ))
        return tuple(sorted(result, key=lambda item: item.filler))

    @staticmethod
    def _pair_inverse_candidates(
            input_structure: QueryInputStructure,
            facts: tuple[ActiveRelationSurface, ...],
            ) -> tuple[tuple[InputRelationCandidate, ActiveRelationSurface], ...]:
        """Resolve a binary inverse query from two exact trained fillers.

        The projector intentionally leaves the predicate cue open when an
        inverse surface is unseen.  This bridge only closes the candidate when
        both filler identities and their trained role ordering match exactly;
        no lexical or character fallback is involved.
        """
        anchors = TrainedGraphQueryBridge.anchor_concepts(input_structure)
        filler_keys = {item.filler for item in anchors}
        if len(filler_keys) != 2:
            return ()
        matches = []
        for fact in facts:
            if len(fact.bindings) != 2:
                continue
            fact_fillers = {item.filler.stable_key() for item in fact.bindings}
            if fact_fillers != filler_keys:
                continue
            # Only the exact binary fact is eligible; a same-filler relation
            # with extra structure cannot be collapsed into a pair query.
            relation = next((item for item in input_structure.relation_candidates
                             if item.proposition_key == fact.proposition.stable_key()
                             and item.required_count == 2
                             and item.support_count == 2
                             and item.predicate_coverage == 0), None)
            if relation is not None:
                matches.append((relation, fact))
        return tuple(sorted(matches, key=lambda item: (
            item[0].proposition_key, item[0].structure_key)))

    def _rank_candidates(
            self,
            input_structure: QueryInputStructure,
            proposition_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[
                tuple[int, tuple[int, ...]],
                tuple[int, tuple[int, ...]],
            ]:
        """按已训练角色拓扑闭合度排名，不做词面或字符近邻评分。"""
        by_proposition: dict[tuple[int, ...], list[InputRelationCandidate]] = {}
        for candidate in input_structure.relation_candidates:
            by_proposition.setdefault(
                candidate.proposition_key, []).append(candidate)
        ranked = []
        for key in proposition_keys:
            candidates = by_proposition.get(key, ())
            score = max((item.score for item in candidates), default=0)
            ranked.append((score, key))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            return (0, ()), (0, ())
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else (0, ())
        return best, second

    @staticmethod
    def _margin_closed(best_score: int, second_score: int) -> bool:
        if best_score <= 0:
            return False
        if second_score <= 0:
            return True
        return (best_score - second_score) * 1000 >= second_score * 25

    def _validated_event_time_graph_inputs(
            self,
            graph_object_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[tuple[int, ...], ...]:
        """只接受能从当前只读 LC-05 topology 完整回读的显式对象键。"""
        if type(graph_object_keys) is not tuple:
            raise TypeError("graph_object_keys 必须是 tuple")
        if any(type(key) is not tuple or not key for key in graph_object_keys):
            raise ValueError("graph_object_keys 必须包含非空对象稳定键")
        if len(set(graph_object_keys)) != len(graph_object_keys):
            raise ValueError("graph_object_keys 不得重复")
        available = set(self.event_time_topology.root_object_keys)
        result = []
        ontology = self.core_runtime.context.graph_ontology
        for key in graph_object_keys:
            if key not in available:
                continue
            identity = ObjectIdentity.from_stable_key(key)
            if key not in available or ontology.resolve(identity) is None:
                raise ValueError("显式 graph object 不属于当前已训练 Event/Time topology")
            result.append(key)
        return tuple(sorted(result))

    def _validated_core_graph_inputs(
            self,
            graph_object_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[tuple[int, ...], ...]:
        """只接受当前 Core 关系图可展开且可由本体回读的一等 filler。"""
        if type(graph_object_keys) is not tuple:
            raise TypeError("graph_object_keys 必须是 tuple")
        if any(type(key) is not tuple or not key
               or any(type(value) is not int or value < 0 for value in key)
               for key in graph_object_keys):
            raise ValueError("graph_object_keys 必须是非空非负整数 tuple")
        if len(set(graph_object_keys)) != len(graph_object_keys):
            raise ValueError("graph_object_keys 不得重复")
        ontology = self.core_runtime.context.graph_ontology
        result = []
        for key in graph_object_keys:
            if key not in self.filler_edges:
                continue
            identity = ObjectIdentity.from_stable_key(key)
            if identity.object_kind not in {
                    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT}:
                continue
            if ontology.resolve(identity) is None:
                raise ValueError("显式 graph object 无法从当前 Core 本体回读")
            result.append(key)
        return tuple(sorted(result))

    def _validated_generic_graph_inputs(
            self,
            graph_object_keys: tuple[tuple[int, ...], ...],
            known_keys: set[tuple[int, ...]],
            ) -> tuple[tuple[int, ...], ...]:
        """Validate remaining explicit graph identities against the read-only ontology.

        Dedicated Event/Time and W-08 identities retain their specialized
        topology routes.  Every other accepted key must already be a
        materialized graph object in this model; this is an identity check,
        never a permissive integer or surface lookup.
        """
        ontology = self.core_runtime.context.graph_ontology
        result = []
        for key in graph_object_keys:
            if key in known_keys:
                continue
            identity = ObjectIdentity.from_stable_key(key)
            if ontology.resolve(identity) is None:
                raise ValueError("显式图输入未在当前只读图本体中物化")
            result.append(key)
        return tuple(sorted(result))

    def _validated_discourse_graph_inputs(
            self,
            graph_object_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[tuple[int, ...], ...]:
        """Accept explicit W-08 proposition/frame identities from the graph.

        These are first-class integer graph inputs, not text or source lookup
        keys.  A frame expands to its source-qualified propositions so relation
        and QUD/revision features can enter the same QueryState frontier.
        """
        if type(graph_object_keys) is not tuple:
            raise TypeError("graph_object_keys 必须是 tuple")
        if any(type(key) is not tuple or not key
               or any(type(value) is not int or value < 0 for value in key)
               for key in graph_object_keys):
            raise ValueError("graph_object_keys 必须是非空非负整数 tuple")
        if len(set(graph_object_keys)) != len(graph_object_keys):
            raise ValueError("graph_object_keys 不得重复")
        allowed = set(self.discourse_topic_topology.proposition_keys)
        # Keep every restored frame, including reference/revision-only
        # frames whose proposition list is intentionally empty.
        frame_to_props = {
            frame.frame_key: frame.proposition_keys
            for frame in self.discourse_topic_topology.frames
        }
        result = set()
        for key in graph_object_keys:
            if key in allowed:
                result.add(key)
            elif key in frame_to_props:
                # Keep the frame itself as a Dialogue root and add its
                # proposition identities as sibling roots.  Both are read
                # from the restored topology; neither is a surface lookup.
                result.add(key)
                result.update(frame_to_props[key])
        return tuple(sorted(result))

    def _match_event_time_generation_bindings(
            self,
            input_structure: QueryInputStructure,
            ) -> tuple[TrainedEventTimeGenerationBinding, ...]:
        """按闭合 proposition/predicate/结构身份绑定普通输入与 LC-05 frame。"""
        matches: set[TrainedEventTimeGenerationBinding] = set()
        for candidate in input_structure.relation_candidates:
            if candidate.state != STRUCTURE_CLOSED:
                continue
            input_key = event_time_input_structure_identity(
                candidate.structure_key).stable_key()
            for binding in self._event_time_bindings_by_proposition.get(
                    candidate.proposition_key, ()):
                if (binding.predicate_key == candidate.predicate_key
                        and binding.input_structure_key == input_key):
                    matches.add(binding)
        return tuple(sorted(matches, key=lambda item: item.stable_key()))

    def _augment_kdconv_input(
            self, input_structure: QueryInputStructure,
            ) -> QueryInputStructure:
        """将精确命中的 KdConv subject+attribute 汇聚为属性选择候选。

        value 可以不在当前输入中；它由同一 Core 图事实在本轮 frontier
        展开后绑定，不依赖 successor/raw surface 回放。
        """
        self._kdconv_query_propositions = set()
        if self.kdconv_runtime is not None and self.kdconv_runtime._lazy:
            before = len(self.kdconv_fact_index)
            self._activate_kdconv_facts(input_structure.token_values)
            if len(self.kdconv_fact_index) != before:
                # Re-project against the page-in'ed graph structures.  The
                # second pass is bounded by exact integer spans and carries
                # the same QueryState evidence contract as the eager path.
                input_structure = self.input_projector.project(
                    input_structure.token_values)
        if not self.kdconv_facts:
            return input_structure
        values = input_structure.token_values
        additions = []
        subject_matches: list[tuple[int, ActiveRelationSurface, object]] = []
        # Build the proposition posting once.  The old path scanned every
        # semantic candidate again for every KdConv fact, multiplying the
        # 123k-fact input pass into an avoidable quadratic query.
        candidates_by_proposition: dict[tuple[int, ...], list[object]] = {}
        for item in input_structure.semantic_candidates:
            candidates_by_proposition.setdefault(item.proposition_key, []).append(item)
        structures_by_proposition = {
            item.proposition_key: item for item in self.input_projector.structures
        }
        runtime = self.kdconv_runtime
        subject_keys: set[tuple[int, ...]] = set()
        if runtime is not None:
            codepoints = set(values)
            for codepoint in codepoints:
                subject_keys.update(runtime._subject_postings.get(codepoint, ()))
        for subject in sorted(subject_keys, key=lambda item: (len(item), item)):
            # A one-codepoint subject is not a stable entity anchor (it is
            # commonly punctuation or a grammatical atom).  It cannot select
            # a KdConv fact on its own; retain the generic Memory/Dialogue
            # path instead of emitting an unrelated property answer.
            if len(subject) < 2:
                continue
            if not _contains_integer(values, subject):
                continue
            for fact in runtime._facts_by_subject.get(subject, ()) if runtime is not None else ():
                cue = tuple(ord(item) for item in fact.cue)
                candidates = candidates_by_proposition.get(
                    fact.proposition.stable_key(), ())
                span_refs = tuple(sorted({item.span_ref for item in candidates}))
                if not span_refs:
                    continue
                structure = structures_by_proposition.get(
                    fact.proposition.stable_key())
                if structure is None:
                    continue
                if _contains_integer(values, cue):
                    additions.append(InputRelationCandidate(
                        fact.proposition.stable_key(), fact.predicate.stable_key(),
                        structure.relation_structure_key(), (91630, 1),
                        structure.source_ref, span_refs, 2, 2, 1, 1, 0,
                        STRUCTURE_CLOSED, 0))
                    self._kdconv_query_propositions.add(fact.proposition.stable_key())
                else:
                    row = runtime._fact_rows.get(
                        fact.proposition.stable_key()) if runtime is not None else None
                    subject_matches.append((
                        0 if row is None else row.support_count, fact, structure))
        if additions and runtime is not None:
            # The frozen source can expose the same subject/attribute/value
            # triple in multiple domain partitions. Those rows are distinct
            # provenance assertions, not competing answers; collapse them at
            # this query boundary by learned support.
            # A complete subject+attribute+value identity may be materialized
            # once per frozen domain.  Those rows are provenance duplicates,
            # not competing Memory topics.  Different values remain separate
            # candidates and therefore remain fail-closed competitors.
            by_semantic: dict[tuple[tuple[int, ...], ...], list[InputRelationCandidate]] = {}
            for candidate in additions:
                row = runtime._fact_rows[candidate.proposition_key]
                property_key = (
                    row.subject_values, row.attribute_values, row.value_values)
                by_semantic.setdefault(property_key, []).append(candidate)
            canonical: list[InputRelationCandidate] = []
            for candidates in by_semantic.values():
                canonical.append(max(
                    candidates,
                    key=lambda item: (
                        runtime._fact_rows[item.proposition_key].support_count,
                        tuple(reversed(item.proposition_key)),
                    )))
            additions = canonical
            self._kdconv_query_propositions = {
                item.proposition_key for item in additions}
        # A subject-only question is a graph-backed confirmation request.  It
        # does not guess across all attributes: one deterministic fact with
        # maximal learned support becomes the sole Dialogue candidate.  An
        # already closed non-KdConv relation has stronger same-input graph
        # evidence and must remain eligible for discourse/topic routing;
        # subject-only completion must not erase that relation.
        existing_closed_relation = any(
            item.state == STRUCTURE_CLOSED
            and item.required_count == item.support_count
            for item in input_structure.relation_candidates)
        if not additions and subject_matches and not existing_closed_relation:
            _support, fact, structure = max(
                subject_matches,
                key=lambda item: (
                    len(self.kdconv_runtime._fact_rows[
                        item[1].proposition.stable_key()].subject_values)
                    if self.kdconv_runtime is not None else 0,
                    item[0],
                    tuple(reversed(item[1].proposition.stable_key())),
                ),
            )
            additions.append(InputRelationCandidate(
                fact.proposition.stable_key(), fact.predicate.stable_key(),
                structure.relation_structure_key(), (91630, 2),
                structure.source_ref, tuple(sorted({
                    item.span_ref for item in input_structure.semantic_candidates
                    if item.proposition_key == fact.proposition.stable_key()
                    and item.projection_kind == PROJECTION_FILLER
                })), 1, 1, 1, 0, 0, STRUCTURE_CLOSED, 0))
            self._kdconv_query_propositions.add(fact.proposition.stable_key())
        if not additions:
            # Once the input already contains complete graph closures, open
            # segmentations are only unresolved alternatives for a different
            # query.  Keeping hundreds of them in this same frontier creates
            # a combinatorial Dialogue/Memory fan-out and can exhaust the
            # deterministic read budget before the closed topic routes run.
            # Retain every complete closed relation, except exact semantic
            # duplicates from separate frozen KdConv domain rows.  Never
            # collapse distinct values merely because their surface spans or
            # relation structure happen to match.
            closed = tuple(
                item for item in input_structure.relation_candidates
                if (item.state == STRUCTURE_CLOSED
                    and item.required_count == item.support_count))
            if closed:
                by_semantic = {}
                for candidate in closed:
                    row = (runtime._fact_rows.get(candidate.proposition_key)
                           if runtime is not None else None)
                    if row is None:
                        semantic_key = (0, candidate.proposition_key)
                    else:
                        semantic_key = (
                            1, row.subject_values, row.attribute_values,
                            row.value_values)
                    prior = by_semantic.get(semantic_key)
                    if (prior is None
                            or candidate.score > prior.score
                            or (candidate.score == prior.score
                                and candidate.proposition_key > prior.proposition_key)):
                        by_semantic[semantic_key] = candidate
                canonical_closed = tuple(sorted(
                    by_semantic.values(), key=lambda item: item.stable_key()))
                selected = frozenset(item.proposition_key
                                     for item in canonical_closed)
                return replace(
                    input_structure,
                    semantic_candidates=tuple(sorted(
                        (item for item in input_structure.semantic_candidates
                         if item.proposition_key in selected),
                        key=lambda item: item.stable_key())),
                    relation_candidates=tuple(sorted(
                        canonical_closed, key=lambda item: item.stable_key())),
                    carrier_candidates=tuple(sorted(
                        (item for item in input_structure.carrier_candidates
                         if item.structure_key in {
                             relation.structure_key
                             for relation in canonical_closed}),
                        key=lambda item: item.stable_key())),
                )
            return input_structure
        # An exact KdConv subject+attribute (or subject-only) closure is a
        # graph-backed disambiguation boundary.  The generic projector can
        # still have emitted thousands of one-codepoint partial candidates;
        # retaining those would manufacture a conflict against the exact
        # proposition.  Restrict this round's semantic/relation evidence to
        # the selected integer propositions, while leaving the trained graph
        # itself untouched for later queries.
        selected = frozenset(self._kdconv_query_propositions)
        semantic_candidates = tuple(
            item for item in input_structure.semantic_candidates
            if item.proposition_key in selected)
        relation_candidates = tuple(
            item for item in input_structure.relation_candidates
            if item.proposition_key in selected)
        selected_structure_keys = {
            item.structure_key for item in relation_candidates
        }
        selected_structure_keys.update(item.structure_key for item in additions)
        carrier_candidates = tuple(
            item for item in input_structure.carrier_candidates
            if item.structure_key in selected_structure_keys)
        input_structure = replace(
            input_structure,
            semantic_candidates=tuple(sorted(
                set(semantic_candidates), key=lambda item: item.stable_key())),
            relation_candidates=tuple(sorted(
                set(relation_candidates), key=lambda item: item.stable_key())),
            carrier_candidates=tuple(sorted(
                set(carrier_candidates), key=lambda item: item.stable_key())),
        )
        merged = {item.stable_key(): item
                  for item in input_structure.relation_candidates}
        merged.update({item.stable_key(): item for item in additions})
        return replace(input_structure,
                       relation_candidates=tuple(sorted(
                           merged.values(), key=lambda item: item.stable_key())))

    def _restore_factual_topic_input(
            self,
            input_structure: QueryInputStructure,
            ) -> QueryInputStructure:
        """Bind one prior delivered subject to the current predicate cue.

        The current user Observation explicitly links to the preceding
        delivered output Episode.  Only factual contexts reached through that
        edge are eligible.  KdConv predicate candidates must agree on one
        subject graph identity; ambiguous topics remain open.
        """
        if self.memory is None:
            return input_structure
        # The same canonical topic binding is needed before Memory.append as
        # it is during query().  Before append there is no current Observation
        # ref yet, so follow only the latest delivered Observation edge; this
        # keeps O/H/E and QueryState on one integer structure identity.
        if self._input_observation_ref is None:
            observation_refs = self.memory._preceding_response_refs()
            if not observation_refs:
                return input_structure
        else:
            observation_refs = (self._input_observation_ref,)
        structures = self.memory.active_structures(
            candidate_keys=(), observation_refs=observation_refs)
        contexts = tuple(
            item for item in restore_response_contexts(self.memory, structures)
            if isinstance(item, DeliveredFactualResponseContext)
        )
        if not contexts or self.kdconv_runtime is None:
            return input_structure
        subject_role = (91630, 1)
        def is_subject(identity: ObjectIdentity) -> bool:
            return (isinstance(identity, ObjectIdentity)
                    and identity.components[-2:] == subject_role)
        topics = {
            slot.filler.stable_key()
            for context in contexts
            for slot in context.slots
            if is_subject(slot.role)
        }
        if len(topics) != 1:
            return input_structure
        topic = next(iter(topics))
        # A complete attribute cue for the linked topic outranks an entity
        # name that is only a proper subspan of that cue.  For example, the
        # graph contains an entity named ``time`` as well as the prior
        # topic's longer ``production time`` attribute.  This is exact graph
        # span containment, not character similarity.
        prior_topic_cues = tuple(sorted({
            tuple(ord(value) for value in fact.cue)
            for fact in self.filler_edges.get(topic, ())
            if fact.proposition.stable_key() in self.kdconv_fact_index
            and any(is_subject(binding.role)
                    and binding.filler.stable_key() == topic
                    for binding in fact.bindings)
            and _contains_integer(
                input_structure.token_values,
                tuple(ord(value) for value in fact.cue))
        }, key=lambda item: (-len(item), item)))
        # A concept route can project a KdConv subject identity from a cue
        # span whose integers are not that subject's learned surface.  Such a
        # projection is useful frontier evidence, but it is not an explicit
        # topic switch.  Admit a switch only when a complete graph-indexed
        # subject sequence occurs exactly in the current integer input.  This
        # is the same subject posting boundary used by _augment_kdconv_input;
        # no surface similarity or language-specific inventory participates.
        explicit_topics = set()
        values = input_structure.token_values
        subject_keys: set[tuple[int, ...]] = set()
        for codepoint in set(values):
            subject_keys.update(
                self.kdconv_runtime._subject_postings.get(codepoint, ()))
        for subject_values in subject_keys:
            if (len(subject_values) < 2
                    or not _contains_integer(values, subject_values)):
                continue
            if any(len(subject_values) < len(cue)
                   and _contains_integer(cue, subject_values)
                   for cue in prior_topic_cues):
                continue
            for fact in self.kdconv_runtime._facts_by_subject.get(
                    subject_values, ()):
                subject = next((
                    binding for binding in fact.bindings
                    if is_subject(binding.role)
                ), None)
                if subject is not None:
                    explicit_topics.add(subject.filler.stable_key())
        if explicit_topics and explicit_topics != {topic}:
            # A non-overlapping subject span is an explicit topic switch.
            # Predicate/subject homographs remain eligible for the linked
            # prior topic because the current span does not disambiguate them.
            return input_structure
        predicate_keys = {
            item.predicate_key
            for item in input_structure.semantic_candidates
            if item.projection_kind != PROJECTION_FILLER
        }
        matches = []
        cue_span_refs: tuple[tuple[int, ...], ...] = ()
        structures_by_proposition = {
            item.proposition_key: item
            for item in self.input_projector.structures
        }
        if predicate_keys:
            for fact in self.kdconv_facts:
                if fact.predicate.stable_key() not in predicate_keys:
                    continue
                subject = next((
                    binding for binding in fact.bindings
                    if is_subject(binding.role)
                ), None)
                structure = structures_by_proposition.get(
                    fact.proposition.stable_key())
                if (subject is not None and subject.filler.stable_key() == topic
                        and structure is not None):
                    matches.append((fact, structure))
        else:
            # KdConv attribute cues are graph relation candidates even when
            # the surface projector has no standalone predicate member (the
            # normal form for short follow-ups such as "它出品时间是什么").
            # Recover only exact integer cue membership from those candidates;
            # no lexical table or character-neighbour heuristic is involved.
            input_values = input_structure.token_values
            for fact in self.filler_edges.get(topic, ()):
                if (fact.proposition.stable_key() not in self.kdconv_fact_index
                        or not _contains_integer(
                            input_values,
                            tuple(ord(value) for value in fact.cue))):
                    continue
                subject = next((
                    binding for binding in fact.bindings
                    if is_subject(binding.role)
                ), None)
                structure = structures_by_proposition.get(
                    fact.proposition.stable_key())
                if (subject is not None and subject.filler.stable_key() == topic
                        and structure is not None):
                    matches.append((fact, structure))
                    cue = tuple(ord(value) for value in fact.cue)
                    cue_span_refs = tuple(sorted({
                        span.ref_key for span in input_structure.spans
                        if span.values == cue
                    }))
        if not matches:
            # A follow-up may expose only a typed value cue (for example the
            # trained ``year`` predicate) while omitting the longer source
            # attribute surface (for example a date-valued attribute).  Keep
            # this route graph-derived: use only a predicate identity from
            # the KdConv graph and exact integer cue membership in the
            # candidate fact value.  It is not character-neighbour matching,
            # a language table, or a source-text lookup.  Ambiguous values
            # remain open and therefore fail closed.
            kdconv_cues = []
            spans_by_ref = {
                span.ref_key: span.values for span in input_structure.spans
            }
            for item in input_structure.semantic_candidates:
                if item.projection_kind == PROJECTION_FILLER:
                    continue
                try:
                    predicate_identity = ObjectIdentity.from_stable_key(
                        item.predicate_key)
                except (TypeError, ValueError):
                    continue
                components = predicate_identity.components
                if (len(components) < 2
                        or components[:2] != (91630, 1)):
                    continue
                values = spans_by_ref.get(item.span_ref, ())
                if values and values not in kdconv_cues:
                    kdconv_cues.append(values)
            if kdconv_cues:
                cue_matches = []
                cue_span_refs = tuple(sorted({
                    item.span_ref
                    for item in input_structure.semantic_candidates
                    if (item.projection_kind != PROJECTION_FILLER
                        and spans_by_ref.get(item.span_ref, ()) in kdconv_cues)
                }))
                for fact in self.kdconv_facts:
                    subject = next((
                        binding for binding in fact.bindings
                        if is_subject(binding.role)
                    ), None)
                    if subject is None or subject.filler.stable_key() != topic:
                        continue
                    row = self.kdconv_runtime._fact_rows[
                        fact.proposition.stable_key()]
                    if any(_contains_integer(row.value_values, cue)
                           for cue in kdconv_cues):
                        structure = structures_by_proposition.get(
                            fact.proposition.stable_key())
                        if structure is not None:
                            cue_matches.append((fact, structure))
                triples = {
                    (self.kdconv_runtime._fact_rows[
                        fact.proposition.stable_key()].subject_values,
                     self.kdconv_runtime._fact_rows[
                        fact.proposition.stable_key()].attribute_values,
                     self.kdconv_runtime._fact_rows[
                        fact.proposition.stable_key()].value_values)
                    for fact, _structure in cue_matches
                }
                if len(triples) == 1:
                    matches = cue_matches
                elif cue_matches:
                    # Several graph attributes can carry the same requested
                    # typed value (for example a full date and a year-only
                    # date).  Collapse only when their exact integer value
                    # prefix through the cue is identical.  The prefix is
                    # the answer-bearing graph segment; it is not a source
                    # surface replay or a character-neighbour heuristic.
                    answer_prefixes = set()
                    for fact, _structure in cue_matches:
                        row = self.kdconv_runtime._fact_rows[
                            fact.proposition.stable_key()]
                        for cue in kdconv_cues:
                            for start in range(
                                    len(row.value_values) - len(cue) + 1):
                                if row.value_values[
                                        start:start + len(cue)] == cue:
                                    answer_prefixes.add(
                                        (cue, row.value_values[:start + len(cue)]))
                                    break
                    if len(answer_prefixes) == 1:
                        # Keep one proposition for one exact answer-bearing
                        # integer prefix. Learned support is primary;
                        # shorter value structure is the deterministic tie
                        # break for a typed unit query; the stable key closes
                        # the ordering without adding a surface heuristic.
                        canonical = max(
                            cue_matches,
                            key=lambda item: (
                                self.kdconv_runtime._fact_rows[
                                    item[0].proposition.stable_key()].support_count,
                                -len(self.kdconv_runtime._fact_rows[
                                    item[0].proposition.stable_key()].value_values),
                                tuple(reversed(item[0].proposition.stable_key())),
                            ),
                        )
                        matches = [canonical]
        relations = {
            item.stable_key(): item for item in input_structure.relation_candidates
        }
        additions = []
        for fact, structure in matches:
            proposition_key = fact.proposition.stable_key()
            predicate_key = fact.predicate.stable_key()
            current = tuple(
                item for item in input_structure.semantic_candidates
                if item.proposition_key == proposition_key
                and item.projection_kind != PROJECTION_FILLER
            )
            if not current:
                # A typed value cue may be trained as its own predicate
                # proposition rather than as the longer KdConv attribute
                # proposition. Preserve the exact cue span as the binding
                # witness instead of dropping the graph fact here.
                span_refs = cue_span_refs
            else:
                span_refs = tuple(sorted({item.span_ref for item in current}))
            if not span_refs:
                continue
            additions.append(InputRelationCandidate(
                proposition_key,
                predicate_key,
                structure.relation_structure_key(),
                (91630, 3),
                structure.source_ref,
                span_refs,
                2,
                2,
                1,
                1,
                0,
                STRUCTURE_CLOSED,
                0,
            ))
        if not additions:
            return input_structure
        # Same subject/attribute/value across source partitions is supporting
        # provenance, not competing dialogue content.
        by_semantic = {}
        for candidate in additions:
            row = self.kdconv_runtime._fact_rows[candidate.proposition_key]
            key = (row.subject_values, row.attribute_values, row.value_values)
            prior = by_semantic.get(key)
            if (prior is None or row.support_count
                    > self.kdconv_runtime._fact_rows[
                        prior.proposition_key].support_count):
                by_semantic[key] = candidate
        additions = tuple(by_semantic.values())
        selected = {item.proposition_key for item in additions}
        relations.update({item.stable_key(): item for item in additions})
        relation_candidates = tuple(sorted((
            item for item in relations.values()
            if item.proposition_key in selected
        ), key=lambda item: item.stable_key()))
        semantic_candidates = tuple(sorted((
            item for item in input_structure.semantic_candidates
            if item.proposition_key in selected
        ), key=lambda item: item.stable_key()))
        self._kdconv_query_propositions = set(selected)
        self._restored_factual_topic_keys = (topic,)
        return replace(
            input_structure,
            semantic_candidates=semantic_candidates,
            relation_candidates=relation_candidates,
            open_relations=(),
            carrier_candidates=(),
        )

    def prepare_input_structure(
            self,
            surface: str | None,
            *,
            graph_object_keys: tuple[tuple[int, ...], ...] = (),
            ) -> QueryInputStructure:
        """Prepare one text, text+graph, or graph-only integer input."""
        if (type(graph_object_keys) is not tuple
                or any(type(key) is not tuple or not key
                       or any(type(item) is not int or item < 0 for item in key)
                       for key in graph_object_keys)
                or graph_object_keys != tuple(sorted(set(graph_object_keys)))):
            raise ValueError("graph_object_keys 必须规范排序去重")
        self._activate_kdconv_graph_objects(graph_object_keys)
        if surface is None:
            return graph_query_input_structure(graph_object_keys)
        if type(surface) is not str or not surface:
            raise ValueError("文字输入必须是非空字符串")
        structure = self._augment_kdconv_input(
            self.input_projector.project(tuple(ord(value) for value in surface)))
        if graph_object_keys:
            structure = replace(
                structure, graph_object_keys=graph_object_keys)
        return structure

    def query(self, surface: str | None = None, *,
               artifact_input: ArtifactQueryInput | None = None,
               graph_object_keys: tuple[tuple[int, ...], ...] = (),
               minimum_depth: int = 1, max_depth: int = 4,
               max_nodes: int = 4096, max_edges: int = 8192,
               max_reads: int = 16384,
               input_observation_ref: MemoryObjectRef | None = None,
               compact_trace: bool = False,
               input_structure: QueryInputStructure | None = None) -> dict[str, object]:
        """建立三图 roots 并执行一次共同 frontier 查询。"""
        if type(compact_trace) is not bool:
            raise TypeError("compact_trace 必须是 bool")
        if any(type(value) is not int or value <= 0
               for value in (max_nodes, max_edges, max_reads)):
            raise ValueError("query budgets must be positive strict integers")
        self._pending_response_delivery = None
        # 生成资格属于当前 QueryState；不能沿用上一轮已闭合命题。已物化
        # generation_by_proposition 可复用，但必须重新经过本轮 Dialogue
        # root 与槽位资格检查。
        self._generation_attempted.clear()
        self.generation_ready_propositions.clear()
        self.typed_relation_projections.clear()
        self._artifact_bridge_matches = ()
        self._artifact_bridge_missing = ()
        self._event_time_graph_input_keys = ()
        self._core_graph_input_keys = ()
        self._generic_graph_input_keys = ()
        self._memory_graph_input_keys = ()
        self._discourse_graph_input_requested_keys = ()
        self._discourse_graph_input_keys = ()
        self._event_time_query_bindings = ()
        self._event_time_query_frame_keys = frozenset()
        self._event_time_query_edge_keys = frozenset()
        self._restored_factual_topic_keys = ()
        if input_observation_ref is not None and (
                not isinstance(input_observation_ref, MemoryObjectRef)
                or input_observation_ref.object_kind != MEMORY_OBJECT_OBSERVATION or self.memory is None):
            raise ValueError("同次查询输入必须引用实际包外 Memory Observation")
        self._input_observation_ref = input_observation_ref
        if artifact_input is not None and not isinstance(artifact_input, ArtifactQueryInput):
            raise TypeError("artifact_input 必须是 ArtifactQueryInput")
        if artifact_input is not None and surface is not None:
            raise ValueError("文字和 artifact 输入必须明确二选一")
        if input_structure is not None and not isinstance(
                input_structure, QueryInputStructure):
            raise TypeError("input_structure 必须是 QueryInputStructure 或 None")
        if artifact_input is not None and input_structure is not None:
            raise ValueError("artifact_input 与 input_structure 不得同时提供")
        # KdConv 的发布运行时只常驻纯整数 posting。显式 Entity/Concept
        # 图输入必须先按完整稳定身份 page-in 相邻事实，随后才与常驻 Core
        # 使用完全相同的 filler/root/frontier 路径；这不是失败后的第二查询。
        self._activate_kdconv_graph_objects(graph_object_keys)
        self._event_time_graph_input_keys = self._validated_event_time_graph_inputs(
            graph_object_keys)
        self._core_graph_input_keys = self._validated_core_graph_inputs(
            graph_object_keys)
        self._discourse_graph_input_requested_keys = tuple(sorted(graph_object_keys))
        self._discourse_graph_input_keys = self._validated_discourse_graph_inputs(
            graph_object_keys)
        known_graph_keys = set(self.event_time_topology.root_object_keys)
        known_graph_keys.update(self._core_graph_input_keys)
        known_graph_keys.update(self.discourse_topic_topology.proposition_keys)
        known_graph_keys.update(
            frame.frame_key for frame in self.discourse_topic_topology.frames)
        self._generic_graph_input_keys = self._validated_generic_graph_inputs(
            graph_object_keys, known_graph_keys)
        known_graph_keys.update(self._generic_graph_input_keys)
        if artifact_input is None:
            if surface is None:
                if not graph_object_keys:
                    raise ValueError("graph-only query 必须携带 graph_object_keys")
                if input_structure is None:
                    input_structure = graph_query_input_structure(
                        graph_object_keys)
                elif (input_structure.token_values
                      or input_structure.graph_object_keys
                      != graph_object_keys):
                    raise ValueError("graph-only input_structure 与对象键不一致")
            else:
                if type(surface) is not str or not surface:
                    raise ValueError("query surface 必须是非空字符串")
                surface_values = tuple(ord(value) for value in surface)
                if input_structure is None:
                    input_structure = self.input_projector.project(surface_values)
                    input_structure = self._augment_kdconv_input(input_structure)
                elif input_structure.token_values != surface_values:
                    raise ValueError("input_structure 与 query surface 整数不一致")
                else:
                    # The caller may have prepared the graph-augmented structure
                    # for Memory intake.  Reconstruct only the proposition set
                    # consumed by the current QueryState; do not project again.
                    self._kdconv_query_propositions = {
                        item.proposition_key for item in input_structure.relation_candidates
                        if item.proposition_key in self.kdconv_fact_index
                    }
                if input_structure.graph_object_keys:
                    if input_structure.graph_object_keys != graph_object_keys:
                        raise ValueError("input_structure graph 对象键与 query 不一致")
                elif graph_object_keys:
                    input_structure = replace(
                        input_structure, graph_object_keys=graph_object_keys)
            query_key = input_structure.source_ref
        else:
            input_structure = artifact_input.query_input_structure(
                self.input_projector.structures)
            if graph_object_keys:
                input_structure = replace(
                    input_structure, graph_object_keys=graph_object_keys)
            query_key = artifact_input.stable_key()
            (self._artifact_bridge_matches,
             self._artifact_bridge_missing) = self._match_artifact_semantic_bindings(
                 artifact_input)
        if artifact_input is not None:
            input_structure = self._augment_kdconv_input(input_structure)
        # KdConv can close an explicit subject in the current input before
        # Memory restore.  Predicate-only follow-ups need the preceding
        # factual ResponsePlan topic, which is an explicit Episode edge and
        # must be recovered before the shared roots/frontier are seeded.
        input_structure = self._restore_factual_topic_input(input_structure)
        concept_anchors = self.anchor_concepts(input_structure)
        # The input projector can only close a relation when its trained
        # predicate span is present.  For an unseen inverse expression, use
        # the exact graph pair route above before ranking Dialogue roots.
        pair_inverse = self._pair_inverse_candidates(input_structure, self.facts)
        if pair_inverse:
            replacements = {
                relation.proposition_key: replace(
                    relation,
                    required_count=2,
                    support_count=2,
                    required_open=0,
                    state=STRUCTURE_CLOSED,
                    predicate_coverage=0,
                )
                for relation, _fact in pair_inverse
            }
            input_structure = replace(
                input_structure,
                relation_candidates=tuple(
                    sorted((replacements.get(item.proposition_key, item)
                            for item in input_structure.relation_candidates),
                           key=lambda item: item.stable_key())))
            concept_anchors = self.anchor_concepts(input_structure)
        # Filler projections without a complete trained relation are retained
        # in the input structure and Memory O/H/E, but they are not Core roots.
        # Otherwise every unmatched sentence expands dozens of one-slot
        # concept guesses until the shared budget is exhausted before the
        # generic Dialogue act can consume the same QueryState.
        complete_relation = any(
            item.state == STRUCTURE_CLOSED
            and item.required_count >= 2
            and item.required_count == item.support_count
            and item.predicate_coverage == 1
            for item in input_structure.relation_candidates)
        if (not complete_relation and not input_structure.open_relations
                and artifact_input is None):
            concept_anchors = ()
        self._event_time_query_bindings = (
            self._match_event_time_generation_bindings(input_structure))
        self._event_time_query_frame_keys = frozenset(
            item.frame_key for item in self._event_time_query_bindings)
        self._event_time_query_edge_keys = frozenset(
            edge_key
            for item in self._event_time_query_bindings
            for edge_key in item.topology_edge_keys)
        self._restore_memory_structures(input_structure, artifact_input)
        roots: list[QueryRoot] = []
        anchors: list[QueryAnchor] = []
        frontier: list[FrontierEntry] = []
        open_nodes = open_role_query_nodes(input_structure.open_relations)
        response_nodes = response_query_nodes(self.surface_generator, self.core_runtime,
                                               input_structure.open_relations)
        generic_hypotheses = tuple(
            item for item in self._discourse_hypotheses
            if (item.candidate_key
                and item.candidate_key[0] == MEMORY_GENERIC_CANDIDATE_VERSION
                and (self._input_observation_ref is None
                     or item.observation_key
                     == self._input_observation_ref.stable_key())))
        discourse_relation_feature_state = self._discourse_relation_feature_state(
            input_structure)
        generic_response_feature_mask = self._generic_response_feature_mask(
            generic_hypotheses, input_structure)
        generic_response_nodes = generic_response_query_nodes(
            self.surface_generator, generic_hypotheses, query_key,
            generic_response_feature_mask,
            input_structure=input_structure,
            explicit_graph_input_keys=(
                self._discourse_graph_input_requested_keys),
        )
        for node in response_nodes:
            root, anchor, edge = node.seed(self.space_weights[2])
            roots.append(root)
            anchors.append(anchor)
            frontier.append(edge)
        for node in generic_response_nodes:
            root, anchor, edge = node.seed(self.space_weights[2])
            roots.append(root)
            anchors.append(anchor)
            frontier.append(edge)
        for node in open_nodes:
            root, anchor, edge = node.seed(self.space_weights[node.space - 1])
            roots.append(root)
            anchors.append(anchor)
            frontier.append(edge)

        core_root_keys = set()
        for item in concept_anchors:
            source_hashes = tuple(sorted({
                fact.source_hash
                for fact in self.filler_edges.get(item.filler, ())
            }))
            source_ref = (len(source_hashes), *source_hashes)
            roots.append(QueryRoot(
                _CORE, item.filler, ROOT_PENDING,
                self.space_weights[0], source_ref=source_ref))
            anchors.append(QueryAnchor(_CORE, item.filler, kind=1))
            frontier.append(FrontierEntry(
                (_CORE, *item.filler), _CORE,
                target_key=item.filler,
                root_key=item.filler,
                source_ref=source_ref,
                direction=0,
                owner_weight=self.space_weights[0],
                depth=0,
                required_slot_gain=3,
            ))
            core_root_keys.add(item.filler)
        for topic_key in self._restored_factual_topic_keys:
            if topic_key in core_root_keys:
                continue
            source_hashes = tuple(sorted({
                fact.source_hash for fact in self.filler_edges.get(topic_key, ())
                if fact.proposition.stable_key() in self._kdconv_query_propositions
            }))
            source_ref = (len(source_hashes), *source_hashes)
            roots.append(QueryRoot(
                _CORE, topic_key, ROOT_PENDING,
                self.space_weights[0], source_ref=source_ref))
            anchors.append(QueryAnchor(_CORE, topic_key, kind=6))
            frontier.append(FrontierEntry(
                (_CORE, *topic_key), _CORE,
                target_key=topic_key,
                root_key=topic_key,
                source_ref=source_ref,
                direction=0,
                owner_weight=self.space_weights[0],
                depth=0,
                required_slot_gain=3,
                discourse_fit=2,
            ))
            core_root_keys.add(topic_key)
        for graph_key in self._core_graph_input_keys:
            if graph_key in core_root_keys:
                continue
            graph_facts = self.filler_edges[graph_key]
            source_hashes = tuple(sorted({
                fact.source_hash for fact in graph_facts
            }))
            source_ref = (len(source_hashes), *source_hashes)
            roots.append(QueryRoot(
                _CORE, graph_key, ROOT_PENDING, self.space_weights[0],
                source_ref=source_ref, scope_key=query_key))
            anchors.append(QueryAnchor(
                _CORE, graph_key, kind=_CORE_GRAPH_INPUT_ANCHOR_KIND,
                support=len(graph_facts), scope_key=query_key))
            frontier.append(FrontierEntry(
                (_CORE, _CORE_GRAPH_INPUT_ANCHOR_KIND, *graph_key), _CORE,
                target_key=graph_key, root_key=graph_key,
                scope_key=query_key, source_ref=source_ref,
                direction=0, owner_weight=self.space_weights[0], depth=0,
                required_slot_gain=2, evidence_support=1, relation_fit=2,
                source_trust=1,
            ))
            core_root_keys.add(graph_key)
        for graph_key in self._generic_graph_input_keys:
            if graph_key in core_root_keys:
                continue
            roots.append(QueryRoot(
                _CORE, graph_key, ROOT_PENDING, self.space_weights[0],
                source_ref=(21610, *graph_key), scope_key=query_key))
            anchors.append(QueryAnchor(
                _CORE, graph_key, kind=10, support=1, scope_key=query_key))
            frontier.append(FrontierEntry(
                (_CORE, 21610, 1, *graph_key), _CORE,
                target_key=graph_key, root_key=graph_key,
                scope_key=query_key, source_ref=(21610, *graph_key),
                direction=0, owner_weight=self.space_weights[0], depth=0,
                required_slot_gain=1, evidence_support=1, relation_fit=2,
                source_trust=1))
            core_root_keys.add(graph_key)
        for graph_key in self._event_time_graph_input_keys:
            if graph_key in core_root_keys:
                continue
            topology_edges = self.event_time_graph_adjacency[graph_key]
            source_refs = tuple(sorted({item.source_ref for item in topology_edges}))
            scopes = tuple(sorted({item.scope_key for item in topology_edges}))
            root_source_ref = frame_key(91560, *source_refs)
            root_scope = scopes[0] if len(scopes) == 1 else ()
            roots.append(QueryRoot(
                _CORE, graph_key, ROOT_PENDING, self.space_weights[0],
                source_ref=root_source_ref, scope_key=root_scope))
            anchors.append(QueryAnchor(
                _CORE, graph_key, kind=_EVENT_TIME_GRAPH_ANCHOR_KIND,
                support=len(topology_edges), scope_key=root_scope))
            frontier.append(FrontierEntry(
                (_CORE, *_EVENT_TIME_GRAPH_EDGE, *graph_key), _CORE,
                target_key=graph_key, root_key=graph_key,
                scope_key=root_scope, source_ref=root_source_ref,
                direction=0, owner_weight=self.space_weights[0], depth=0,
                required_slot_gain=2, evidence_support=1, relation_fit=2,
                source_trust=1,
            ))
            core_root_keys.add(graph_key)
        for binding in self._event_time_query_bindings:
            generation_frame_key = binding.frame_key
            if generation_frame_key in core_root_keys:
                continue
            topology_edges = self.event_time_graph_adjacency.get(generation_frame_key, ())
            if not topology_edges:
                raise RuntimeError("Event/Time generation frame 缺少 topology edge")
            roots.append(QueryRoot(
                _CORE, generation_frame_key, ROOT_PENDING, self.space_weights[0],
                source_ref=binding.source_ref,
                scope_key=binding.scope_key,
            ))
            anchors.append(QueryAnchor(
                _CORE, generation_frame_key, kind=_EVENT_TIME_GENERATION_ANCHOR_KIND,
                support=len(topology_edges), scope_key=binding.scope_key))
            frontier.append(FrontierEntry(
                (_CORE, *_EVENT_TIME_GRAPH_EDGE,
                 _EVENT_TIME_GENERATION_ANCHOR_KIND, *generation_frame_key),
                _CORE,
                target_key=generation_frame_key,
                root_key=generation_frame_key,
                scope_key=binding.scope_key,
                source_ref=binding.source_ref,
                direction=0,
                owner_weight=self.space_weights[0],
                depth=0,
                required_slot_gain=3,
                evidence_support=1,
                relation_fit=3,
                source_trust=1,
            ))
            core_root_keys.add(generation_frame_key)
        for graph_key in self._discourse_graph_input_keys:
            roots.append(QueryRoot(
                _DIALOGUE, graph_key, ROOT_PENDING,
                self.space_weights[2], source_ref=(21601, *graph_key),
                scope_key=query_key))
            anchors.append(QueryAnchor(
                _DIALOGUE, graph_key, kind=9, support=1,
                scope_key=query_key))
            frontier.append(FrontierEntry(
                (_DIALOGUE, 21601, 1, *graph_key), _DIALOGUE,
                target_key=graph_key, root_key=graph_key,
                scope_key=query_key, source_ref=(21601, *graph_key),
                direction=1, owner_weight=self.space_weights[2], depth=0,
                required_slot_gain=1, evidence_support=1, relation_fit=3,
                discourse_fit=3, source_trust=1))
        for graph_key in self._generic_graph_input_keys:
            roots.append(QueryRoot(
                _DIALOGUE, graph_key, ROOT_PENDING,
                self.space_weights[2], source_ref=(21610, 4, *graph_key),
                scope_key=query_key))
            anchors.append(QueryAnchor(
                _DIALOGUE, graph_key, kind=10, support=1,
                scope_key=query_key))
            frontier.append(FrontierEntry(
                (_DIALOGUE, 21610, 4, *graph_key), _DIALOGUE,
                target_key=graph_key, root_key=graph_key,
                scope_key=query_key, source_ref=(21610, 4, *graph_key),
                direction=1, owner_weight=self.space_weights[2], depth=0,
                required_slot_gain=1, evidence_support=1, relation_fit=1,
                discourse_fit=2, source_trust=1))
        if artifact_input is not None:
            artifact_source_ref = artifact_input.source_ref
            artifact_scope_key = artifact_input.envelope.scope.stable_key()
            envelope_key = artifact_input.envelope.identity.stable_key()
            roots.append(QueryRoot(
                _CORE, envelope_key, ROOT_PENDING, self.space_weights[0],
                source_ref=artifact_source_ref, scope_key=artifact_scope_key))
            anchors.append(QueryAnchor(
                _CORE, envelope_key, kind=_ARTIFACT_ENVELOPE_ANCHOR_KIND,
                support=max(1, len(artifact_input.understanding_projections)),
                scope_key=artifact_scope_key))
            frontier.append(FrontierEntry(
                (_CORE, *envelope_key), _CORE,
                target_key=envelope_key, root_key=envelope_key,
                scope_key=artifact_scope_key, source_ref=artifact_source_ref,
                direction=0, owner_weight=self.space_weights[0], depth=0,
                required_slot_gain=2, evidence_support=1, relation_fit=1,
            ))
            core_root_keys.add(envelope_key)
            semantic_objects = tuple(sorted({
                item.semantic_object.stable_key()
                for item in artifact_input.understanding_projections
            }))
            for semantic_key in semantic_objects:
                if semantic_key in core_root_keys:
                    continue
                roots.append(QueryRoot(
                    _CORE, semantic_key, ROOT_PENDING, self.space_weights[0],
                    source_ref=artifact_source_ref, scope_key=artifact_scope_key))
                anchors.append(QueryAnchor(
                    _CORE, semantic_key, kind=_ARTIFACT_SEMANTIC_ANCHOR_KIND,
                    support=1, scope_key=artifact_scope_key))
                frontier.append(FrontierEntry(
                    (_CORE, *semantic_key), _CORE,
                    target_key=semantic_key, root_key=semantic_key,
                    scope_key=artifact_scope_key, source_ref=artifact_source_ref,
                    direction=0, owner_weight=self.space_weights[0], depth=0,
                    required_slot_gain=2, evidence_support=1, relation_fit=1,
                ))
                core_root_keys.add(semantic_key)
        # 结构候选作为同一 QueryState 的关系 anchor 登记；它们只是开放
        # 解析候选，是否闭合仍由 Core/Evidence/终止谓词共同决定。
        for candidate in input_structure.relation_candidates:
            anchors.append(QueryAnchor(
                _CORE,
                candidate.structure_key,
                kind=2,
                support=candidate.support_count,
            ))
        if not core_root_keys and not any(node.space == _CORE for node in open_nodes):
            roots.append(QueryRoot(
                _CORE, _EMPTY_ROOT_KEY, ROOT_EMPTY,
                self.space_weights[0]))

        for item in self._memory_observations:
            roots.append(QueryRoot(
                _MEMORY, item.observation_key, ROOT_PENDING,
                self.space_weights[1],
                source_ref=item.source_ref,
                scope_key=item.scope_key,
            ))
            anchors.append(QueryAnchor(
                _MEMORY, item.observation_key, kind=1,
                scope_key=item.scope_key))
            frontier.append(FrontierEntry(
                (_MEMORY, *item.observation_key), _MEMORY,
                target_key=item.observation_key,
                root_key=item.observation_key,
                scope_key=item.scope_key,
                source_ref=item.source_ref,
                direction=0,
                owner_weight=self.space_weights[1],
                depth=0,
                required_slot_gain=3 if item.context_depth == 0 else 1,
                discourse_fit=1 if item.context_depth == 0 else 0,
                source_trust=1,
                recency_weight=item.turn_seq,
            ))
        for context in self._response_contexts:
            for occurrence in context.role_occurrences:
                root, anchor, entry = occurrence.seed(self.space_weights[1])
                roots.append(root)
                anchors.append(anchor)
                frontier.append(entry)
        for projection in self._delivered_graph_slot_projections:
            root, anchor, entry = projection.seed(self.space_weights[1])
            roots.append(root)
            anchors.append(anchor)
            frontier.append(entry)
        for occurrence in self._input_role_occurrences:
            root, anchor, entry = occurrence.seed(self.space_weights[1])
            roots.append(root)
            anchors.append(anchor)
            frontier.append(entry)
        for candidate_set in self._reference_candidate_sets:
            root, anchor, entry = candidate_set.seed(self.space_weights[1])
            roots.append(root)
            anchors.append(anchor)
            frontier.append(entry)
        if not self._memory_observations:
            roots.append(QueryRoot(
                _MEMORY, _EMPTY_ROOT_KEY, ROOT_EMPTY,
                self.space_weights[1]))

        # Dialogue/Companion 只能围绕已有 filler 概念 anchor 参与同轮查询。
        # 仅命中 predicate cue 或其它表层片段的输入不构成语义根，避免
        # 无 anchor 查询被训练 frame 旁路扩展。
        anchored_propositions = {
            item.proposition_key
            for item in input_structure.semantic_candidates
            if item.projection_kind == PROJECTION_FILLER
        }
        if self._restored_factual_topic_keys:
            anchored_propositions.update(self._kdconv_query_propositions)
        dialogue_keys = tuple(sorted({
            item.proposition_key
            for item in input_structure.relation_candidates
            if (item.proposition_key in self.dialogue_frame_index
                and item.proposition_key in anchored_propositions
                # A partially covered relation remains an open Core
                # candidate, but must not create a competing Dialogue frame.
                # Otherwise a filler shared by two propositions turns a
                # closed fact into a false ambiguity and blocks generation.
                and item.state == STRUCTURE_CLOSED)
        }))
        best, second = self._rank_candidates(
            input_structure, dialogue_keys)
        selected_required_open = 1
        if best[1]:
            # Candidate aggregation is score-based.  A lower-scoring open
            # interpretation for the same proposition must not keep the
            # winning closed Event/Time binding open; equally scoring open
            # interpretations remain represented and are handled by the
            # existing conflict/competition fields.
            winning = tuple(
                item for item in input_structure.relation_candidates
                if item.proposition_key == best[1] and item.score == best[0])
            selected_required_open = int(
                not winning or any(item.required_open > 0 for item in winning))
        input_order_conflict = int(any(
            item.state == STRUCTURE_ORDER_CONFLICT
            for item in input_structure.relation_candidates))
        for key in dialogue_keys:
            frame = self.dialogue_frame_index[key]
            source_ref = (frame.source_hash,)
            roots.append(QueryRoot(
                _DIALOGUE, key, ROOT_PENDING,
                self.space_weights[2], source_ref=source_ref))
            anchors.append(QueryAnchor(_DIALOGUE, key, kind=3))
            frontier.append(FrontierEntry(
                (_DIALOGUE, *key), _DIALOGUE,
                target_key=key,
                root_key=key,
                source_ref=source_ref,
                direction=1,
                owner_weight=self.space_weights[2],
                depth=0,
                required_slot_gain=3,
                evidence_support=1,
            ))
        dialogue_memory_keys = tuple(sorted({
            item.hypothesis_key for item in self._discourse_hypotheses
        }))
        discourse_topic_hypothesis_keys = set(
            self._discourse_topic_hypothesis_keys)
        topic_proposition_by_candidate = {
            item.candidate_key: item.proposition_key
            for item in self._discourse_topic_candidates
        }
        topic_graph_backed = set(self._discourse_topic_graph_identity_keys)
        topology_propositions = set(
            self.discourse_topic_topology.proposition_keys)
        observation_turns = {
            item.observation_key: item.turn_seq
            for item in self._memory_observations
        }
        observation_depths = {item.observation_key: item.context_depth for item in self._memory_observations}
        for key in dialogue_memory_keys:
            item = next(
                candidate for candidate in self._discourse_hypotheses
                if candidate.hypothesis_key == key)
            roots.append(QueryRoot(
                _DIALOGUE, key, ROOT_PENDING,
                self.space_weights[2], source_ref=item.source_ref,
                scope_key=item.scope_key,
            ))
            anchors.append(QueryAnchor(
                _DIALOGUE, key,
                kind=(7 if key in discourse_topic_hypothesis_keys else 1),
                scope_key=item.scope_key))
            topic_proposition = topic_proposition_by_candidate.get(
                item.candidate_key)
            topic_is_graph_backed = (
                topic_proposition in topic_graph_backed
                or topic_proposition in topology_propositions)
            topic_depth = observation_depths.get(item.observation_key, 0)
            # Depth is an integer priority signal inside the shared frontier:
            # current/hot topic context wins over cold history, while Core's
            # larger owner weight remains the dominant cross-space factor.
            if key in discourse_topic_hypothesis_keys:
                if topic_is_graph_backed and topic_depth == 0:
                    topic_fit = 5
                elif topic_is_graph_backed and topic_depth <= 1:
                    topic_fit = 4
                elif topic_depth == 0:
                    topic_fit = 4
                else:
                    topic_fit = 2
            else:
                topic_fit = 2 if topic_depth == 0 else 1
            frontier.append(FrontierEntry(
                (_DIALOGUE, *key), _DIALOGUE,
                target_key=key,
                root_key=key,
                scope_key=item.scope_key,
                source_ref=item.source_ref,
                direction=0,
                owner_weight=self.space_weights[2],
                depth=0,
                required_slot_gain=1,
                discourse_fit=topic_fit,
                source_trust=1,
                recency_weight=observation_turns[item.observation_key],
            ))
        if (not dialogue_keys and not dialogue_memory_keys
                and not any(node.space == _DIALOGUE for node in open_nodes)):
            roots.append(QueryRoot(
                _DIALOGUE, _EMPTY_ROOT_KEY, ROOT_EMPTY,
                self.space_weights[2]))

        # Preserve every exact trained filler projection in the shared state
        # even when its surrounding relation remains open and therefore does
        # not qualify as a Core frontier root.  These UNKNOWN records let the
        # response planner consume a known entity/event/concept without
        # asserting the unfinished proposition or re-reading input text.
        semantic_fillers = tuple(
            item for item in input_structure.semantic_candidates
            if item.projection_kind == PROJECTION_FILLER
        )
        filler_sets_by_role: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
        for item in semantic_fillers:
            identity = ObjectIdentity.from_stable_key(item.candidate_key)
            if identity.object_kind != item.object_kind:
                raise ValueError(
                    "input semantic filler identity kind drifted")
            filler_sets_by_role.setdefault(item.role_key, set()).add(
                item.candidate_key)
        initial_bindings = {}
        initial_evidence = {}
        for item in semantic_fillers:
            conflict = int(len(filler_sets_by_role[item.role_key]) > 1)
            binding = BindingEntry(
                item.role_key,
                item.candidate_key,
                space=_CORE,
                scope_key=query_key,
                conflict_kept=conflict,
            )
            projection_key = frame_key(
                _INPUT_SEMANTIC_PROJECTION_EVIDENCE[0],
                _INPUT_SEMANTIC_PROJECTION_EVIDENCE[1:],
                item.stable_key(),
            )
            evidence = EvidenceEntry(
                item.source_ref,
                item.proposition_key,
                space=_CORE,
                polarity=3,
                trust=1,
                evidence_key=projection_key,
                scope_key=query_key,
                payload_key=item.stable_key(),
            )
            initial_bindings[binding.stable_key()] = binding
            initial_evidence[evidence.stable_key()] = evidence
        # Relation candidates are graph objects in their own right.  Keep the
        # proposition identity available to the response planner without
        # asserting its truth: UNKNOWN evidence preserves open/conflicting
        # interpretations and ambiguity is rejected by the graph-slot binder.
        for item in input_structure.relation_candidates:
            proposition = ObjectIdentity.from_stable_key(item.proposition_key)
            if proposition.object_kind != OBJECT_PROPOSITION:
                raise ValueError(
                    "input relation proposition identity kind drifted")
            binding = BindingEntry(
                item.predicate_key,
                item.proposition_key,
                space=_CORE,
                scope_key=query_key,
                conflict_kept=int(item.state == STRUCTURE_ORDER_CONFLICT),
            )
            projection_key = frame_key(
                _INPUT_RELATION_PROJECTION_EVIDENCE[0],
                _INPUT_RELATION_PROJECTION_EVIDENCE[1:],
                item.stable_key(),
            )
            evidence = EvidenceEntry(
                item.source_ref,
                item.proposition_key,
                space=_CORE,
                polarity=3,
                trust=1,
                evidence_key=projection_key,
                scope_key=query_key,
                payload_key=item.stable_key(),
            )
            initial_bindings[binding.stable_key()] = binding
            initial_evidence[evidence.stable_key()] = evidence

        initial = seed_query(
            query_key,
            roots=tuple(roots),
            anchors=tuple(anchors),
            minimum_depth=minimum_depth,
            budget=QueryBudget(
                max_nodes=max_nodes, max_edges=max_edges,
                max_reads=max_reads, max_depth=max_depth),
            active_spaces=(_CORE, _MEMORY, _DIALOGUE),
        ).with_(
            bindings=tuple(initial_bindings[key]
                           for key in sorted(initial_bindings)),
            evidence=tuple(initial_evidence[key]
                           for key in sorted(initial_evidence)),
            required_slots_open=selected_required_open,
            best_score=best[0],
            second_score=second[0],
            best_candidate_key=best[1],
            second_candidate_key=second[1],
        )
        expander = _CrossSpaceExpander(
            self,
            input_structure=input_structure,
            artifact_input=artifact_input,
            artifact_bridge_matches=self._artifact_bridge_matches,
            input_order_conflict=input_order_conflict,
            input_required_slots_open=selected_required_open,
            response_nodes=response_nodes,
            generic_response_nodes=generic_response_nodes,
            generic_hypotheses=generic_hypotheses,
        )
        tracer = _IntegerQueryTrace(reference_mode=compact_trace)
        tracer.record(initial.with_(frontier=tuple(frontier)).canonical(),
                      TERMINATION_OPEN)
        state = run_query(
            initial.with_(frontier=tuple(frontier)),
            expander=expander,
            tracer=tracer,
        )
        # The last expansion can close the final Memory Evidence and prepare
        # the Dialogue realization at the same time.  Re-evaluate that fully
        # converged state once, without another graph read, because the driver
        # correctly stops when no unvisited frontier remains and otherwise
        # retains the pre-final closure flags.
        if (state.termination in {TERMINATION_NO_FRONTIER, 6}
                and not unvisited_frontier(state.frontier, state.visited)
                and all(root.status != ROOT_PENDING for root in state.roots)):
            state = expander._with_query_status(
                state.with_(termination=TERMINATION_OPEN, cycle_hit=0),
                None,
                marginal_gain=state.marginal_gain,
            )
            predicate = replace(
                from_query_state(state), frontier_count=0)
            final_reason = evaluate_termination(
                predicate, budgets=state.budget.stable_key())
            if final_reason != TERMINATION_OPEN:
                state = state.with_(termination=final_reason)
                tracer.record(state, final_reason)
        self._discourse_topic_query_projection = validate_discourse_topic_query_state(
            state,
            self._discourse_topic_candidates,
            self._discourse_topic_hypothesis_keys,
        )
        hops = tuple(sorted(
            expander.hops, key=lambda item: (item.depth, item.space,
                                             item.proposition)))
        # 结构生成：若 Core 侧证据闭合，把已绑定命题交给槽位重填，产出
        # 新表层（非整句回放）。生成失败/无闭合命题时返回 None，不猜。
        generation = None
        if hops and state.termination == TERMINATION_ANSWER_CLOSED:
            generation = self._generate_from_closed(
                hops, concept_anchors, state.best_candidate_key)
        # 阶段 C：把闭合命题经 ResponsePlan 组织并执行 token postcheck。生成
        # dict 只作为可重放 trace；真实发布承重回答从 ResponsePlan 渲染。若
        # plan 无法通过 postcheck（无结构证据/必填槽缺 token），则回答保持
        # None 由上层 fail-closed，不退回旧表层回放。
        response_plan = (
            self._response_plan_from_closed(
                hops, concept_anchors, state.best_candidate_key, state)
            if state.termination == TERMINATION_ANSWER_CLOSED
            else None
        )
        # 终止字段与可发布计划必须一致：闭合状态若无法构造并通过
        # ResponsePlan/token postcheck，立即降级为 NO_FRONTIER，绝不暴露
        # 空表层或伪造的 ANSWER_CLOSED。
        if state.termination == TERMINATION_ANSWER_CLOSED and response_plan is None:
            state = state.with_(
                termination=TERMINATION_NO_FRONTIER,
                evidence_closed=0,
                generation_ready=0,
            )
        if (response_plan is not None and response_plan.claim_refs
                and state.termination == TERMINATION_ANSWER_CLOSED
                and self._input_observation_ref is not None):
            self._pending_response_delivery = (
                state,
                FactualResponseGeneration(
                    response_plan,
                    self._input_observation_ref,
                    state.best_candidate_key,
                    (
                        91630,
                        state.depth,
                        state.node_count,
                        state.edge_count,
                        state.read_count,
                    ),
                    tuple(map(ord, response_plan.surface())),
                ),
            )
        if (state.termination in {TERMINATION_CLARIFY_MISSING_BINDING, TERMINATION_NO_FRONTIER}
                and not state.conflict_open
                # Generic and open-role candidates are produced from the same
                # converged QueryState.  When both exist, retain the generic
                # graph-backed candidate for the later shared arbitration;
                # the companion/open candidate must not overwrite it.
                and not expander.generic_response_generations
                and len(expander.open_response_generations) == 1):
            # 此候选已经在共同 frontier 内执行，这里只裁决非事实输出资格。
            # 不触发第二轮图库查询，也不覆盖已闭合的事实 ResponsePlan。
            response = expander.open_response_generations[0]
            self._pending_response_delivery = (state, response)
            response_plan = response.response_plan
            generation = {
                "response_act": list(response_plan.response_act.stable_key()),
                "connector": list(_response_plan_connector_key(response_plan)),
                "representations": [list(item.stable_key()) for item in response.representations],
                "discourse_context": [list(item.stable_key())
                                      for item in response.preview.request.structure.discourse.context],
                "trace": list(response.trace), "surface_values": list(map(ord, response_plan.surface())),
            }
        # Generic connectors have already participated as Dialogue roots and
        # completed G-00--G-03 inside the shared frontier. This final step only
        # adopts that computed non-factual candidate when no factual/open-role
        # plan is eligible; it does not query a second store or read a course.
        generic_unknown_keys = {
            item.hypothesis_key for item in self._memory_hypotheses
            if (item.candidate_key
                and item.candidate_key[0] == MEMORY_GENERIC_CANDIDATE_VERSION)
        }
        generic = (
            expander.generic_response_generations[0]
            if len(expander.generic_response_generations) == 1 else None)
        # A graph-backed generic act may coexist with a closed Core relation.
        # When its visible slots are qualified by the same input spans and
        # QueryState evidence, the trained Dialogue condition selects the
        # non-factual organization for this turn.  Core evidence remains in
        # the shared state; this is not a fallback query or a Core discard.
        generic_structural_preferred = bool(
            generic is not None and generic.graph_surfaces)
        # A context-only UNKNOWN connector has no graph surface proposal by
        # design, but it is still a complete trained Dialogue act.  When no
        # factual/open-role ResponsePlan exists, adopt that act directly from
        # the same QueryState instead of treating the empty graph-surface set
        # as absence of generation.
        generic_adoption_ready = bool(
            generic is not None
            and (response_plan is None or generic_structural_preferred))

        def generic_conflicts_closed() -> bool:
            """Close only conflicts consumed by an exact Memory/input join."""
            conflicts = tuple(
                item for item in state.bindings if item.conflict_kept == 1)
            if not conflicts:
                return True
            if generic is None or not generic.graph_surfaces:
                return False
            kind_by_category = {1: 16, 2: 17, 3: 7, 4: 7, 5: 4}
            selected_by_kind: dict[int, set[tuple[int, ...]]] = {}
            for category, proposal in generic.graph_surfaces:
                expected_kind = kind_by_category[category]
                if proposal.target.object_kind != expected_kind:
                    return False
                selected_by_kind.setdefault(expected_kind, set()).add(
                    proposal.target.stable_key())
            selected_targets = {
                value for values in selected_by_kind.values()
                for value in values
            }
            # A complete order-conflict does not lack a semantic role; only
            # its observed span order differs from the trained relation.  It
            # is consumed exactly when the generated multi-slot assignment is
            # the whole filler set of that same proposition.  Open/partial
            # relations and subset matches never enter this set.
            consumed_relation_keys = set()
            if input_structure is not None:
                for relation in input_structure.relation_candidates:
                    if (relation.state != STRUCTURE_ORDER_CONFLICT
                            or relation.support_count
                            != relation.required_count
                            or relation.required_open != 0):
                        continue
                    members = tuple(
                        item for item in input_structure.semantic_candidates
                        if (item.projection_kind == PROJECTION_FILLER
                            and item.proposition_key
                            == relation.proposition_key
                            and item.span_ref in relation.span_refs)
                    )
                    member_targets = {
                        item.candidate_key for item in members
                    }
                    if (len(members) == relation.role_coverage
                            and len({(item.member_ordinal, item.role_key)
                                     for item in members}) == len(members)
                            and member_targets == selected_targets):
                        consumed_relation_keys.add(
                            relation.proposition_key)
            relevant_by_role: dict[
                tuple[int, ...], set[tuple[int, ...]]] = {}
            for binding in conflicts:
                identity = None
                try:
                    identity = ObjectIdentity.from_stable_key(
                        binding.filler_key)
                except (TypeError, ValueError):
                    pass
                if binding.filler_key in consumed_relation_keys:
                    continue
                if (identity is None
                        or identity.object_kind not in selected_by_kind):
                    return False
                relevant_by_role.setdefault(binding.role_key, set()).add(
                    binding.filler_key)
            if not relevant_by_role:
                return bool(consumed_relation_keys)
            for fillers in relevant_by_role.values():
                selected = {
                    value for values in selected_by_kind.values()
                    for value in values if value in fillers
                }
                if len(selected) != 1:
                    return False
            return True

        generic_only_conflict = bool(expander.generic_response_generations) and bool(
            generic_unknown_keys) and not any(
                item.space == _CORE and item.polarity in {1, 2}
                for item in state.evidence
            ) and generic_conflicts_closed() and not any(
                item.hypothesis_key in generic_unknown_keys
                and item.polarity in {1, 2}
                for item in state.evidence
            )
        # A context-only UNKNOWN act has no graph surface with which to
        # consume a structural binding conflict.  When the same QueryState
        # contains no supported or refuted Core evidence, that conflict is
        # not a factual contradiction and must not suppress the already
        # generated dialogue act.  This keeps unknown-input response
        # selection on the shared frontier without making it a fallback
        # query or permitting an unsupported claim.
        generic_context_only_open = bool(
            expander.generic_response_generations
            and generic_unknown_keys
            and not any(
                item.space == _CORE and item.polarity in {1, 2}
                for item in state.evidence
            )
            and not any(
                item.hypothesis_key in generic_unknown_keys
                and item.polarity in {1, 2}
                for item in state.evidence
            )
            and all(not item.graph_surfaces
                     for item in expander.generic_response_generations)
        )
        generic_structural_closed = bool(
            generic_structural_preferred and generic_conflicts_closed())
        # An UNKNOWN act may be graph-complete while the input projector
        # retains an order/conflict candidate for an unfinished textual
        # relation.  With no supported/refuted Core evidence, that candidate
        # is not a factual contradiction: the graph-backed Dialogue act can
        # consume the explicit semantic root in this same QueryState.
        generic_structural_unknown = bool(
            generic_structural_preferred
            and not any(
                item.space == _CORE and item.polarity in {1, 2}
                for item in state.evidence))
        if (generic_adoption_ready
                and (not state.conflict_open or generic_only_conflict
                     or generic_structural_closed
                     or generic_structural_unknown
                     or generic_context_only_open)
                    and (state.termination not in {
                    TERMINATION_CLARIFY_CONFLICT,
                    TERMINATION_DIALOGUE_UNKNOWN}
                     or generic_adoption_ready)):
            if len(expander.generic_response_generations) == 1:
                generic = expander.generic_response_generations[0]
                response_plan = generic.response_plan
                unknown_surface = response_plan.surface()
                # Generic UNKNOWN is a completed, evidence-backed dialogue
                # act, not a factual answer and not an unresolved routing
                # failure.  Preserve evidence_closed=0 while giving the
                # shared QueryState an explicit non-factual terminal state.
                state = state.with_(
                    termination=TERMINATION_DIALOGUE_UNKNOWN,
                    generation_ready=1,
                )
                tracer.record(state, TERMINATION_DIALOGUE_UNKNOWN)
                if self._input_observation_ref is not None:
                    self._pending_response_delivery = (
                        state,
                        GenericResponseGeneration(
                            response_plan, self._input_observation_ref,
                            generic.trace, tuple(map(ord, unknown_surface)),
                            generic.graph_surfaces),
                    )
                generation = {
                    "kind": 3,
                    "response_act": list(response_plan.response_act.stable_key()),
                    "connector": list(_response_plan_connector_key(response_plan)),
                    "representations": [list(item.stable_key())
                                        for item in generic.representations],
                    "trace": list(generic.trace),
                    "surface_values": list(map(ord, unknown_surface)),
                }
        return self._finalize(
            state, hops, termination=state.termination,
            reason=_TERMINATION_NAMES.get(state.termination, "OPEN"),
            generation=generation,
            response_plan=(None if response_plan is None
                           else self._integer_response_plan(response_plan)),
            response_surface=(
                "" if response_plan is None else response_plan.surface()),
            frontier_trace=tracer.export(),
            visited=tuple(item.stable_key() for item in state.visited),
            expanded_edges=tuple(expander.expanded_edges),
            memory_routes=self._memory_routes,
            open_generation_contexts=tuple(expander.open_generation_contexts[key].stable_key()
                                           for key in sorted(expander.open_generation_contexts)),
            observed_surface_proposals=tuple(item.stable_key() for key in sorted(expander.observed_surfaces)
                                            for item in expander.observed_surfaces[key]),
            open_response_generations=tuple(item.stable_key() for item in expander.open_response_generations),
            generic_response_generations=tuple(
                item.stable_key() for item in expander.generic_response_generations),
            generic_response_feature_mask=generic_response_feature_mask,
            discourse_relation_feature_state=discourse_relation_feature_state,
            response_contexts=tuple(item.stable_key() for item in self._response_contexts),
            delivered_graph_slot_projections=tuple(
                item.stable_key()
                for item in self._delivered_graph_slot_projections),
            input_role_occurrences=tuple(item.stable_key() for item in self._input_role_occurrences),
            reference_candidate_sets=tuple(item.stable_key() for item in self._reference_candidate_sets),
            reference_resolutions=tuple(item.stable_key() for item in self._reference_resolutions),
            relation_capability_routes=tuple(
                expander.relation_capability_routes[key].trace()
                for key in sorted(expander.relation_capability_routes)),
            typed_relation_projections=tuple(
                expander.typed_relation_projections[key].trace()
                for key in sorted(expander.typed_relation_projections)),
            memory_competitions=tuple(
                (item.hypothesis_key, item.competition_key,
                 memory_competition_identity(item.candidate_key, item.competition_key,
                                             item.observation_key))
                for item in self._memory_hypotheses),
            discourse_topic_candidates=tuple(
                item.stable_key() for item in self._discourse_topic_candidates),
            discourse_topic_hypothesis_keys=self._discourse_topic_hypothesis_keys,
            discourse_topic_query_projection=self._discourse_topic_query_projection,
            discourse_topic_graph_identity_keys=self._discourse_topic_graph_identity_keys,
            discourse_topic_graph_topology=self.discourse_topic_topology.stable_key(),
            event_time_graph_input_keys=self._event_time_graph_input_keys,
            core_graph_input_keys=tuple(sorted({
                *self._core_graph_input_keys,
                *self._generic_graph_input_keys,
            })),
            generic_graph_input_keys=self._generic_graph_input_keys,
            memory_graph_input_keys=self._memory_graph_input_keys,
            discourse_graph_input_requested_keys=(
                self._discourse_graph_input_requested_keys),
            discourse_graph_input_keys=self._discourse_graph_input_keys,
            event_time_graph_topology=self.event_time_topology.integer_trace(),
            event_time_graph_expanded_edges=tuple(sorted(set(
                expander.event_time_graph_expanded_edges))),
            event_time_generation_bindings=tuple(
                item.integer_trace()
                for item in self.event_time_generation_bindings),
            event_time_query_bindings=tuple(
                item.integer_trace()
                for item in self._event_time_query_bindings),
            input_structure=input_structure,
            artifact_input=artifact_input,
            artifact_bridge_matches=tuple(
                item.natural_key() for item in self._artifact_bridge_matches),
            artifact_bridge_missing=self._artifact_bridge_missing,
            artifact_bridge_projection_keys=tuple(
                item.projection_key for item in self._artifact_bridge_matches),
            compact_trace=compact_trace,
        )

    def query_artifact(
            self, artifact_input: ArtifactQueryInput, *,
            graph_object_keys: tuple[tuple[int, ...], ...] = (),
            minimum_depth: int = 1, max_depth: int = 4,
            input_observation_ref: MemoryObjectRef | None = None,
            compact_trace: bool = False,
            ) -> dict[str, object]:
        """只读消费已验证 carrier；Memory 持久化须先由 append_artifact 显式完成。"""
        return self.query(
            artifact_input=artifact_input,
            graph_object_keys=graph_object_keys,
            minimum_depth=minimum_depth,
            max_depth=max_depth,
            input_observation_ref=input_observation_ref,
            compact_trace=compact_trace,
        )

    def acknowledge_response_delivery(self, surface: str):
        """宿主成功送出当前唯一回应后采用，不接收其他查询或未选 preview。"""
        pending = self._pending_response_delivery
        if pending is None:
            return None
        if self.memory is None:
            raise ValueError("回应采用必须写入包外 Interaction Memory")
        state, generation = pending
        if type(surface) is not str or surface != generation.response_plan.surface():
            raise ValueError("宿主送出的不是当前唯一 ResponsePlan")
        if isinstance(generation, FactualResponseGeneration):
            receipt = record_factual_response_delivery(
                self.memory, state, generation)
        elif isinstance(generation, GenericResponseGeneration):
            receipt = record_generic_response_delivery(
                self.memory, state, generation)
        else:
            receipt = record_open_response_delivery(
                self.memory, state, generation, tuple(map(ord, surface)))
        self._pending_response_delivery = None
        return receipt

    @staticmethod
    def _integer_response_plan(plan: object) -> dict[str, object]:
        """把 ResponsePlan 投影为纯整数 trace；字符串只留在输出边界。"""
        return {
            "response_act": list(plan.response_act.stable_key()),
            "claim_refs": [
                list(item.stable_key()) for item in plan.claim_refs],
            "event_refs": [
                list(item.stable_key()) for item in plan.event_refs],
            "memory_refs": [
                list(item.stable_key()) for item in plan.memory_refs],
            "slot_sequence": [
                {
                    "role": list(slot.role.stable_key()),
                    "filler": list(slot.filler.stable_key()),
                    "filler_surface_values": [
                        ord(value) for value in slot.filler_surface],
                    "required": 1 if slot.required else 0,
                    "source_hash": slot.source_hash,
                    "token_digest": list(slot.token_digest),
                    "allowed_node_kinds": list(slot.allowed_node_kinds),
                }
                for slot in plan.slot_sequence
            ],
            "discourse_links": [
                list(item.stable_key()) for item in plan.discourse_links],
            "scope_and_time": list(plan.scope_and_time),
            "evidence_refs": [list(item) for item in plan.evidence_refs],
            "style_ref": (
                [] if plan.style_ref is None
                else list(plan.style_ref.stable_key())),
            "carrier_ref": (
                [] if plan.carrier_ref is None
                else list(plan.carrier_ref.stable_key())),
            "realization_candidates": [
                {
                    "surface_values": [
                        ord(value) for value in item.surface],
                    "frame_proposition": list(
                        item.frame_proposition.stable_key()),
                    "frame_source_hash": item.frame_source_hash,
                    "slot_count": item.slot_count,
                }
                for item in plan.realization_candidates
            ],
            "stable_key": list(plan.stable_key()),
        }

    def _best_closed_fact(
            self,
            hops: tuple[QueryHop, ...],
            proposition_key: tuple[int, ...],
            ) -> tuple[ActiveRelationSurface, object] | None:
        """从 Core hops 反查可生成命题；无闭合事实返回 None。"""
        core_hops = tuple(
            item for item in hops
            if item.space == _CORE and item.proposition == proposition_key)
        for hop in core_hops:
            if proposition_key not in self.core_fact_index:
                continue
            fact = self.core_fact_index[proposition_key]
            generated = self.generation_by_proposition.get(proposition_key)
            if generated is None:
                continue
            if not generated.surface.strip():
                continue
            return fact, generated
        return None

    def _bind_event_time_response_plan(
            self,
            plan,
            proposition_key: tuple[int, ...],
            generated: GraphRelationGeneration,
            ):
        """把查询实际展开的 Event/Time frame 限定交给既有 ResponsePlan。"""
        bindings = tuple(
            item for item in self._event_time_query_bindings
            if item.proposition_key == proposition_key)
        if not bindings:
            return plan
        from pure_integer_ai.cognition.shared.generation_structure_plan import (
            GenerationStructurePlan,
        )
        if (generated.connector is None
                or not isinstance(generated.structure_plan, GenerationStructurePlan)):
            raise ValueError("Event/Time 回答缺少既有 connector/GenerationStructurePlan")
        connector_key = generated.connector.stable_key()
        if any(item.connector_key != connector_key for item in bindings):
            raise ValueError("Event/Time 回答 connector 与训练绑定漂移")
        planned_templates = {
            candidate.proposition.template.stable_key()
            for candidate in generated.structure_plan.selection.request.candidates
        }
        if proposition_key not in planned_templates:
            raise ValueError("GenerationStructurePlan 未消费选中 Event/Time Proposition")
        event_refs = tuple(sorted({
            *plan.event_refs,
            *(ObjectIdentity.from_stable_key(key)
              for item in bindings for key in item.event_keys),
        }, key=ObjectIdentity.stable_key))
        scope_and_time = [*plan.scope_and_time,
                          GENERATION_BINDING_NAMESPACE, len(bindings)]
        for binding in bindings:
            record = binding.scope_and_time_key()
            scope_and_time.extend((len(record), *record))
        discourse_links = tuple(sorted({
            *plan.discourse_links,
            *(ObjectIdentity.from_stable_key(item.frame_key)
              for item in bindings),
        }, key=ObjectIdentity.stable_key))
        return replace(
            plan,
            event_refs=event_refs,
            discourse_links=discourse_links,
            scope_and_time=tuple(scope_and_time),
        )

    def _response_plan_from_closed(
            self,
            hops: tuple[QueryHop, ...],
            anchors: tuple[ConceptAnchor, ...],
            proposition_key: tuple[int, ...],
            state: QueryState,
            ) -> object | None:
        """从已闭合 Core 事实构造可验证 ResponsePlan（阶段 C 发布承重路径）。

        只接受通过 token postcheck 的 plan；失败返回 None 由上层 fail-closed，
        不退回任何旧表层/近邻/整句回放。anchor 表层只作为 realization 证据。
        """
        from pure_integer_ai.experiments.generation_organization import (
            bind_query_evidence,
            plan_from_active_fact,
        )
        found = self._best_closed_fact(hops, proposition_key)
        if found is None:
            return None
        fact, generated = found
        try:
            plan = plan_from_active_fact(fact, generated)
            if plan is None:
                return None
            plan = bind_query_evidence(plan, state)
            typed = self.typed_relation_projections.get(proposition_key)
            if typed is not None:
                event_refs = plan.event_refs
                if typed.semantic_kind == 2:
                    event_refs = tuple(sorted({
                        *event_refs,
                        *(ObjectIdentity.from_stable_key(filler)
                          for _role, filler in typed.fields),
                    }, key=ObjectIdentity.stable_key))
                plan = replace(
                    plan,
                    event_refs=event_refs,
                    # bind_query_evidence already carries the typed
                    # EvidenceEntry; do not add a duplicate bare projection
                    # key that would violate delivery's exact evidence set.
                    evidence_refs=plan.evidence_refs,
                    scope_and_time=(
                        *plan.scope_and_time,
                        91600,
                        len(typed.stable_key()),
                        *typed.stable_key(),
                    ),
                )
            return self._bind_event_time_response_plan(
                plan, proposition_key, generated)
        except (TypeError, ValueError):
            return None

    def _generate_from_closed(
            self,
            hops: tuple[QueryHop, ...],
            anchors: tuple[ConceptAnchor, ...],
            proposition_key: tuple[int, ...],
            ) -> dict[str, object] | None:
        """从已闭合的 Core 多跳 claim 生成组织后的表层。

        槽位填充复用 runtime 的 _generate_surface：把命题的 RoleBinding 表层
        重填到同 predicate/role 的生成框架，产出由图内 token 组合的新句子。
        anchor 输入表层不作为槽位，只作为 realization 证据。
        """
        found = self._best_closed_fact(hops, proposition_key)
        if found is None:
            return None
        _fact, best = found
        return {
            "kind": 1,
            "surface_values": [ord(value) for value in best.surface],
            "frame_proposition": list(best.frame_proposition.stable_key()),
            "slot_count": best.slot_count,
            "connector": ([] if best.connector is None else list(best.connector.stable_key())),
            "representations": [list(item.stable_key()) for item in best.representations],
            "trace": list(best.trace),
            "structure_plan": (
                [] if best.structure_plan is None
                else list(best.structure_plan.stable_key())),
            "event_time_generation_bindings": [
                list(item.stable_key())
                for item in self._event_time_query_bindings
                if item.proposition_key == proposition_key],
            "anchor_surface_values": (
                list(anchors[0].surface_values) if anchors else []),
        }

    @staticmethod
    def _finalize(state: QueryState, hops: tuple[QueryHop, ...], *,
                  termination: int, reason: str,
                  generation: dict[str, object] | None = None,
                  response_plan: dict[str, object] | None = None,
                  response_surface: str = "",
                  frontier_trace: (
                      dict[str, object] | tuple[dict[str, object], ...]) = (),
                  visited: tuple[tuple[int, ...], ...] = (),
                  expanded_edges: tuple[tuple[int, ...], ...] = (),
                  memory_routes: tuple[tuple[int, ...], ...] = (),
                  open_generation_contexts: tuple[tuple[int, ...], ...] = (),
                  observed_surface_proposals: tuple[tuple[int, ...], ...] = (),
                  open_response_generations: tuple[tuple[int, ...], ...] = (),
                  generic_response_generations: tuple[tuple[int, ...], ...] = (),
                  generic_response_feature_mask: int = 0,
                  discourse_relation_feature_state: tuple[int, ...] = (),
                  response_contexts: tuple[tuple[int, ...], ...] = (),
                  delivered_graph_slot_projections: tuple[
                      tuple[int, ...], ...] = (),
                   input_role_occurrences: tuple[tuple[int, ...], ...] = (),
                  reference_candidate_sets: tuple[tuple[int, ...], ...] = (),
                  reference_resolutions: tuple[tuple[int, ...], ...] = (),
                  relation_capability_routes: tuple[
                      dict[str, object], ...] = (),
                  typed_relation_projections: tuple[
                      dict[str, object], ...] = (),
                  memory_competitions: tuple[
                      tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...] = (),
                  discourse_topic_candidates: tuple[tuple[int, ...], ...] = (),
                  discourse_topic_hypothesis_keys: tuple[tuple[int, ...], ...] = (),
                   discourse_topic_query_projection: tuple[int, ...] = (),
                   discourse_topic_graph_identity_keys: tuple[tuple[int, ...], ...] = (),
                   discourse_topic_graph_topology: tuple[int, ...] = (),
                   event_time_graph_input_keys: tuple[tuple[int, ...], ...] = (),
                   core_graph_input_keys: tuple[tuple[int, ...], ...] = (),
                   generic_graph_input_keys: tuple[tuple[int, ...], ...] = (),
                   memory_graph_input_keys: tuple[tuple[int, ...], ...] = (),
                   discourse_graph_input_requested_keys: tuple[tuple[int, ...], ...] = (),
                   discourse_graph_input_keys: tuple[tuple[int, ...], ...] = (),
                   event_time_graph_topology: dict[str, object] | None = None,
                   event_time_graph_expanded_edges: tuple[tuple[int, ...], ...] = (),
                   event_time_generation_bindings: tuple[dict[str, object], ...] = (),
                   event_time_query_bindings: tuple[dict[str, object], ...] = (),
                   artifact_bridge_matches: tuple[tuple[int, ...], ...] = (),
                  artifact_bridge_missing: tuple[tuple[int, ...], ...] = (),
                  artifact_bridge_projection_keys: tuple[tuple[int, ...], ...] = (),
                  input_structure: QueryInputStructure,
                  artifact_input: ArtifactQueryInput | None = None,
                  compact_trace: bool = False,
                  ) -> dict[str, object]:
        """把最终状态、多跳路径与终止原因组装为可重放 trace。"""
        if compact_trace:
            if not isinstance(frontier_trace, dict):
                raise TypeError("compact trace 缺少 frontier 引用表")
            raw_table = frontier_trace.get("integer_key_table")
            raw_steps = frontier_trace.get("steps")
            if not isinstance(raw_table, list) or not isinstance(raw_steps, list):
                raise TypeError("compact frontier trace 格式错误")
            key_table = [tuple(item) for item in raw_table]
            if any(any(type(value) is not int for value in key)
                   for key in key_table):
                raise TypeError("compact frontier key table 必须是纯整数")
            key_ids = {key: index for index, key in enumerate(key_table, 1)}
            if len(key_ids) != len(key_table):
                raise ValueError("compact frontier key table 不得重复")

            def ref(value: tuple[int, ...]) -> int:
                """在当前 trace 内驻留一个完整整数键并返回一基引用号。"""
                if type(value) is not tuple or any(
                        type(item) is not int for item in value):
                    raise TypeError("compact trace 只能驻留整数 tuple")
                if not value:
                    return 0
                current = key_ids.get(value)
                if current is None:
                    current = len(key_table) + 1
                    key_ids[value] = current
                    key_table.append(value)
                return current

            def tree(value):
                """把任意 trace 树中的整数序列替换为同表引用。"""
                if isinstance(value, (list, tuple)):
                    if all(type(item) is int for item in value):
                        return {"integer_key_ref": ref(tuple(value))}
                    return [tree(item) for item in value]
                if isinstance(value, dict):
                    return {key: tree(value[key]) for key in sorted(value)}
                return value

            result = {
                "format": "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V4",
                "schema_version": 4,
                "integer_key_encoding": "INTEGER_KEY_TABLE_V1",
                "input_structure": tree(input_structure.integer_trace()),
                "artifact_input": (
                    None if artifact_input is None
                    else tree(artifact_input.integer_trace())),
                "artifact_bridge": tree({
                    "available": int(bool(artifact_bridge_matches)
                                     or bool(artifact_bridge_missing)),
                    "matched": artifact_bridge_matches,
                    "projection_keys": artifact_bridge_projection_keys,
                    "missing": artifact_bridge_missing,
                }),
                "termination": termination,
                "termination_reason": reason,
                "depth": state.depth,
                "minimum_depth": state.minimum_depth,
                "node_count": state.node_count,
                "edge_count": state.edge_count,
                "read_count": state.read_count,
                "score": state.score,
                "hops": [
                    {
                        "space": item.space,
                        "proposition_ref": ref(item.proposition),
                        "predicate_ref": ref(item.predicate),
                        "from_filler_ref": ref(item.from_filler),
                        "to_filler_ref": ref(item.to_filler),
                        "relation_fillers_refs": [
                            ref(value) for value in item.relation_fillers],
                        "surface_values_ref": ref(item.surface_values),
                        "depth": item.depth,
                        "source_hash": item.source_hash,
                    }
                    for item in hops
                ],
                "evidence": [
                    {
                        "source_hash": item.source_ref[0],
                        "source_ref": ref(item.source_ref),
                        "space": item.space,
                        "hypothesis_ref": ref(item.hypothesis_key),
                        "polarity": item.polarity,
                        "trust": item.trust,
                        "evidence_ref": ref(item.evidence_key),
                        "scope_ref": ref(item.scope_key),
                        "payload_ref": ref(item.payload_key),
                        "supersedes_ref": ref(item.supersedes_key),
                    }
                    for item in state.evidence
                ],
                "bindings": [
                    {
                        "role_ref": ref(item.role_key),
                        "filler_ref": ref(item.filler_key),
                        "space": item.space,
                        "scope_ref": ref(item.scope_key),
                        "conflict_kept": item.conflict_kept,
                    }
                    for item in state.bindings
                ],
                "roots": [
                    {
                        "space": item.owner_space,
                        "root_ref": ref(item.root_key),
                        "status": item.status,
                        "priority_weight": item.priority_weight,
                        "source_ref": ref(item.source_ref),
                        "scope_ref": ref(item.scope_key),
                    }
                    for item in state.roots
                ],
                "anchors": [
                    {
                        "space": item.space,
                        "ref": ref(item.ref_key),
                        "kind": item.kind,
                        "support": item.support,
                        "scope_ref": ref(item.scope_key),
                    }
                    for item in state.anchors
                ],
                "active_spaces": list(state.active_spaces),
                "expanded_edge_refs": [ref(item) for item in expanded_edges],
                "memory_route_refs": [ref(item) for item in memory_routes],
                "open_generation_context_refs": [
                    ref(item) for item in open_generation_contexts],
                "observed_surface_proposal_refs": [
                    ref(item) for item in observed_surface_proposals],
                "open_response_generation_refs": [
                    ref(item) for item in open_response_generations],
                "generic_response_generation_refs": [
                    ref(item) for item in generic_response_generations],
                "generic_response_feature_mask": generic_response_feature_mask,
                "discourse_relation_feature_state": list(
                    discourse_relation_feature_state),
                "response_context_refs": [ref(item) for item in response_contexts],
                "delivered_graph_slot_projection_refs": [
                    ref(item) for item in delivered_graph_slot_projections],
                "input_role_occurrence_refs": [
                    ref(item) for item in input_role_occurrences],
                "reference_candidate_set_refs": [
                    ref(item) for item in reference_candidate_sets],
                "reference_resolution_refs": [
                    ref(item) for item in reference_resolutions],
                "relation_capability_routes": tree(relation_capability_routes),
                "typed_relation_projections": tree(typed_relation_projections),
                "memory_competitions": [
                    {
                        "hypothesis_ref": ref(hypothesis),
                        "declared_ref": ref(declared),
                        "effective_ref": ref(effective),
                    }
                    for hypothesis, declared, effective in memory_competitions
                ],
                "discourse_topic_candidate_refs": [
                    ref(item) for item in discourse_topic_candidates],
                "discourse_topic_hypothesis_refs": [
                    ref(item) for item in discourse_topic_hypothesis_keys],
                "discourse_topic_query_projection_ref": ref(
                    discourse_topic_query_projection),
                "discourse_topic_graph_identity_refs": [
                    ref(item) for item in discourse_topic_graph_identity_keys],
                "discourse_topic_graph_topology_ref": ref(
                    discourse_topic_graph_topology),
                "event_time_graph": tree({
                    "input_keys": event_time_graph_input_keys,
                    "topology": ({} if event_time_graph_topology is None
                                 else event_time_graph_topology),
                    "expanded_edges": event_time_graph_expanded_edges,
                    "generation_bindings": event_time_generation_bindings,
                    "query_generation_bindings": event_time_query_bindings,
                }),
                "event_time_graph_input_keys": [
                    list(item) for item in event_time_graph_input_keys],
                "core_graph_input_keys": [
                    list(item) for item in core_graph_input_keys],
                "generic_graph_input_keys": [
                    list(item) for item in generic_graph_input_keys],
                "memory_graph_input_keys": [
                    list(item) for item in memory_graph_input_keys],
                "discourse_graph_input_keys": [
                    list(item) for item in discourse_graph_input_keys],
                "discourse_graph_input_requested_keys": [
                    list(item) for item in discourse_graph_input_requested_keys],
                "frontier_trace": {
                    "format": frontier_trace.get("format"),
                    "steps": raw_steps,
                },
                "visited_refs": [ref(item) for item in visited],
                "termination_state": {
                    "required_slots_open": state.required_slots_open,
                    "evidence_closed": state.evidence_closed,
                    "conflict_open": state.conflict_open,
                    "best_score": state.best_score,
                    "second_score": state.second_score,
                    "best_candidate_ref": ref(state.best_candidate_key),
                    "second_candidate_ref": ref(state.second_candidate_key),
                    "marginal_gain": state.marginal_gain,
                    "cycle_hit": state.cycle_hit,
                    "generation_ready": state.generation_ready,
                },
                "generation": tree(generation),
                # WorkMemory 的同轮采用仍消费展开的 ResponsePlan；这里只保留
                # 一份，不再在 state_key/frontier trace 中重复扁平化。
                "response_plan": response_plan,
                "response_surface": response_surface,
                "query_ref": ref(state.query_key),
                "state_key": {
                    "encoding": "QUERY_STATE_COMPONENT_REFERENCES_V1",
                    "query_ref": ref(state.query_key),
                    "roots": [ref(item.stable_key()) for item in state.roots],
                    "anchors": [ref(item.stable_key()) for item in state.anchors],
                    "frontier": [ref(item.stable_key()) for item in state.frontier],
                    "bindings": [ref(item.stable_key()) for item in state.bindings],
                    "evidence": [ref(item.stable_key()) for item in state.evidence],
                    "visited": [ref(item.stable_key()) for item in state.visited],
                    "scalars": [
                        state.depth, state.minimum_depth, state.node_count,
                        state.edge_count, state.read_count, state.score,
                        state.required_slots_open, state.evidence_closed,
                        state.conflict_open, state.best_score,
                        state.second_score, ref(state.best_candidate_key),
                        ref(state.second_candidate_key), state.marginal_gain,
                        state.cycle_hit, state.generation_ready,
                        state.termination,
                    ],
                    "budget": list(state.budget.stable_key()),
                    "active_spaces": list(state.active_spaces),
                },
            }
            result["integer_key_table"] = [list(key) for key in key_table]
            return result
        bindings = tuple(
            {"role": list(item.role_key), "filler": list(item.filler_key),
             "space": item.space, "conflict_kept": item.conflict_kept}
            for item in state.bindings)
        evidence = tuple({
            "source_hash": item.source_ref[0],
            "source_ref": list(item.source_ref),
            "space": item.space,
            "hypothesis": list(item.hypothesis_key),
            "polarity": item.polarity,
            "trust": item.trust,
            "evidence_key": list(item.evidence_key),
            "scope": list(item.scope_key),
            "payload_key": list(item.payload_key),
            "supersedes": list(item.supersedes_key),
        } for item in state.evidence)
        return {
            "format": _TRACE_FORMAT,
            "schema_version": 3,
            "input_structure": input_structure.integer_trace(),
            "artifact_input": (
                None if artifact_input is None
                else artifact_input.integer_trace()),
            "artifact_bridge": {
                "available": int(bool(artifact_bridge_matches) or bool(artifact_bridge_missing)),
                "matched": [list(item) for item in artifact_bridge_matches],
                "projection_keys": [list(item) for item in artifact_bridge_projection_keys],
                "missing": [list(item) for item in artifact_bridge_missing],
            },
            "termination": termination,
            "termination_reason": reason,
            "depth": state.depth,
            "minimum_depth": state.minimum_depth,
            "node_count": state.node_count,
            "edge_count": state.edge_count,
            "read_count": state.read_count,
            "score": state.score,
            "hops": [
                {
                    "space": item.space,
                    "proposition": list(item.proposition),
                    "predicate": list(item.predicate),
                    "from_filler": list(item.from_filler),
                    "to_filler": list(item.to_filler),
                    "relation_fillers": [
                        list(value) for value in item.relation_fillers],
                    "surface_values": list(item.surface_values),
                    "depth": item.depth,
                    "source_hash": item.source_hash,
                }
                for item in hops
            ],
            "evidence": evidence,
            "bindings": bindings,
            "roots": [
                {
                    "space": item.owner_space,
                    "root": list(item.root_key),
                    "status": item.status,
                    "priority_weight": item.priority_weight,
                    "source_ref": list(item.source_ref),
                    "scope": list(item.scope_key),
                }
                for item in state.roots
            ],
            "anchors": [
                {
                    "space": item.space,
                    "ref": list(item.ref_key),
                    "kind": item.kind,
                    "support": item.support,
                    "scope": list(item.scope_key),
                }
                for item in state.anchors
            ],
            "active_spaces": list(state.active_spaces),
            "expanded_edges": [list(item) for item in expanded_edges],
            "memory_routes": [list(item) for item in memory_routes],
            "open_generation_contexts": [list(item) for item in open_generation_contexts],
            "observed_surface_proposals": [list(item) for item in observed_surface_proposals],
            "open_response_generations": [list(item) for item in open_response_generations],
            "generic_response_generations": [
                list(item) for item in generic_response_generations],
            "generic_response_feature_mask": generic_response_feature_mask,
            "discourse_relation_feature_state": list(
                discourse_relation_feature_state),
            "response_contexts": [list(item) for item in response_contexts],
            "delivered_graph_slot_projections": [
                list(item) for item in delivered_graph_slot_projections],
            "input_role_occurrences": [list(item) for item in input_role_occurrences],
            "reference_candidate_sets": [list(item) for item in reference_candidate_sets],
            "reference_resolutions": [list(item) for item in reference_resolutions],
            "relation_capability_routes": list(relation_capability_routes),
            "typed_relation_projections": list(typed_relation_projections),
            "memory_competitions": [
                {"hypothesis": list(hypothesis), "declared": list(declared),
                 "effective": list(effective)}
                for hypothesis, declared, effective in memory_competitions],
            "discourse_topic_candidates": [
                list(item) for item in discourse_topic_candidates],
            "discourse_topic_hypothesis_keys": [
                list(item) for item in discourse_topic_hypothesis_keys],
            "discourse_topic_query_projection": list(
                discourse_topic_query_projection),
            "discourse_topic_graph_identity_keys": [
                list(item) for item in discourse_topic_graph_identity_keys],
            "discourse_topic_graph_topology": list(
                discourse_topic_graph_topology),
            "event_time_graph": {
                "input_keys": [list(item) for item in event_time_graph_input_keys],
                "topology": ({} if event_time_graph_topology is None
                             else event_time_graph_topology),
                "expanded_edges": [
                    list(item) for item in event_time_graph_expanded_edges],
                "generation_bindings": list(event_time_generation_bindings),
                "query_generation_bindings": list(event_time_query_bindings),
            },
            "event_time_graph_input_keys": [
                list(item) for item in event_time_graph_input_keys],
            "core_graph_input_keys": [
                list(item) for item in core_graph_input_keys],
            "generic_graph_input_keys": [
                list(item) for item in generic_graph_input_keys],
            "memory_graph_input_keys": [
                list(item) for item in memory_graph_input_keys],
            "discourse_graph_input_keys": [
                list(item) for item in discourse_graph_input_keys],
            "discourse_graph_input_requested_keys": [
                list(item) for item in discourse_graph_input_requested_keys],
            "frontier_trace": (
                frontier_trace if isinstance(frontier_trace, dict)
                else list(frontier_trace)),
            "visited": [list(item) for item in visited],
            "termination_state": {
                "required_slots_open": state.required_slots_open,
                "evidence_closed": state.evidence_closed,
                "conflict_open": state.conflict_open,
                "best_score": state.best_score,
                "second_score": state.second_score,
                "best_candidate": list(state.best_candidate_key),
                "second_candidate": list(state.second_candidate_key),
                "marginal_gain": state.marginal_gain,
                "cycle_hit": state.cycle_hit,
                "generation_ready": state.generation_ready,
            },
            "generation": generation,
            "response_plan": response_plan,
            "response_surface": response_surface,
            "query_key": list(state.query_key),
            "state_key": list(state.stable_key()),
        }


_TERMINATION_NAMES = {
    TERMINATION_ANSWER_CLOSED: "ANSWER_CLOSED",
    TERMINATION_CLARIFY_CONFLICT: "CLARIFY_CONFLICT",
    TERMINATION_CLARIFY_MISSING_BINDING: "CLARIFY_MISSING_BINDING",
    TERMINATION_NO_FRONTIER: "NO_FRONTIER",
    5: "BUDGET_EXHAUSTED",
    6: "CYCLE_GUARD",
    7: "MARGINAL_CONVERGED",
    TERMINATION_DIALOGUE_UNKNOWN: "DIALOGUE_UNKNOWN",
}


class _CrossSpaceExpander(QueryExpander):
    """把共同 frontier 按 owner 分发给三个结构图扩展器。

    visited 只记录已作为 origin 扩展过的节点；新邻居入队但不在本轮标记为
    已访问，否则第二跳会被误判为环。owner 权重只参与排序，不能关闭空间。
    """

    canonical_output = 1

    def __init__(
            self,
            owner: TrainedGraphQueryBridge,
            *,
            input_structure: QueryInputStructure,
            artifact_input: ArtifactQueryInput | None = None,
            artifact_bridge_matches: tuple[object, ...] = (),
            input_order_conflict: int = 0,
            input_required_slots_open: int = 1,
            response_nodes: tuple = (),
            generic_response_nodes: tuple = (),
            generic_hypotheses: tuple = (),
            ) -> None:
        if input_order_conflict not in {0, 1}:
            raise ValueError("input_order_conflict 必须是 0 或 1")
        if input_required_slots_open not in {0, 1}:
            raise ValueError("input_required_slots_open 必须是 0 或 1")
        if artifact_input is not None and not isinstance(artifact_input, ArtifactQueryInput):
            raise TypeError("artifact_input 必须是 ArtifactQueryInput 或 None")
        self.owner = owner
        self.input_structure = input_structure
        self.artifact_input = artifact_input
        self.artifact_bridge_matches = tuple(artifact_bridge_matches)
        self.input_order_conflict = input_order_conflict
        self.input_required_slots_open = input_required_slots_open
        self.hops: list[QueryHop] = []
        self.expanded_edges: list[tuple[int, ...]] = []
        self.open_nodes = {(node.space, node.key): node
                           for node in open_role_query_nodes(input_structure.open_relations)}
        self.open_candidates = {
            frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,), item.stable_key()): item
            for item in input_structure.open_relations
        }
        self.open_generation_contexts: dict[tuple[int, ...], OpenRoleGenerationContext] = {}
        self.observed_surfaces = {}
        self.response_nodes = {node.key: node for node in response_nodes}
        self.generic_response_nodes = {
            node.key: node for node in generic_response_nodes}
        if set(self.response_nodes).intersection(self.generic_response_nodes):
            raise ValueError("open and generic response connector roots overlap")
        self.response_nodes.update(self.generic_response_nodes)
        self.generic_hypotheses = tuple(generic_hypotheses)
        self.response_occurrence_nodes = {item.stable_key(): item for context in owner._response_contexts
                                          for item in context.role_occurrences}
        self.input_occurrence_nodes = {item.stable_key(): item
                                       for item in owner._input_role_occurrences}
        self.reference_candidate_nodes = {item.stable_key(): item
                                          for item in owner._reference_candidate_sets}
        self.delivered_graph_slot_nodes = {
            item.stable_key(): item
            for item in owner._delivered_graph_slot_projections
        }
        self.relation_capability_routes = {}
        self.typed_relation_projections: dict[
            tuple[int, ...], TypedRelationProjection] = {}
        self.open_response_generations = ()
        self.generic_response_generations = ()
        self.event_time_graph_expanded_edges: list[tuple[int, ...]] = []
        self._response_generation_attempted = False

    def _source_hash(self, source_ref: tuple[int, ...], fallback: int) -> int:
        """将完整 SourceRef 稳定键转换为注册表 source hash。"""
        if not source_ref:
            return fallback
        if len(source_ref) != 11:
            raise ValueError("图结构 relation source_ref 不是完整 SourceRef")
        source = SourceRef.from_stable_key(source_ref)
        registry = self.owner.core_runtime.context.scoped_identity_store.registry
        source_hash = registry.find(
            IDENTITY_SOURCE_RECORD, source.stable_key())
        # Earlier append-only graphs can contain a complete SourceRef in the
        # statement scope without a materialized source-record header.  Core
        # Evidence uses the same deterministic registry identity in that case;
        # topology hops must preserve that identity instead of rejecting the
        # otherwise closed graph edge.
        return (source_hash if source_hash is not None else
                registry.identity_hash(
                    IDENTITY_SOURCE_RECORD, source.stable_key()))

    def _bind_open_generation_contexts(self, state: QueryState) -> QueryState:
        """Bind source-local generation spans only after their shared graph paths expand."""
        additions = []
        evidence = []
        for item in self.owner._discourse_hypotheses:
            if (self.owner._input_observation_ref is not None
                    and item.observation_key != self.owner._input_observation_ref.stable_key()):
                continue
            candidate = self.open_candidates.get(item.candidate_key)
            if candidate is None or item.hypothesis_key in self.open_generation_contexts:
                continue
            context = prepare_open_role_generation(
                state, candidate, observation_key=item.observation_key,
                hypothesis_key=item.hypothesis_key, competition_key=item.competition_key,
                source_ref=item.source_ref, scope_key=item.scope_key, evidence_keys=item.evidence_keys,
            )
            if context is not None:
                self.open_generation_contexts[item.hypothesis_key] = context
                additions.extend(context.query_bindings())
                reader = getattr(self.owner.surface_generator, "observed_role_surfaces", None)
                if reader is not None:
                    fact = self.owner.core_fact_index[candidate.frame.proposition]
                    # A trained relation may have a valid open-role frame but
                    # no role-level connector in this model revision.  Keep
                    # the shared O/H/E context and let generation remain
                    # unavailable; never abort the query or fall back to a
                    # source sentence/successor surface.
                    try:
                        proposals = reader(
                            self.owner.core_runtime.generation_input(fact.proposition),
                            fact, context.observed_spans())
                    except (RuntimeError, TypeError, ValueError):
                        proposals = ()
                    self.observed_surfaces[item.hypothesis_key] = proposals
                    for proposal in proposals:
                        key = proposal.stable_key()
                        additions.append(BindingEntry((91524, 1), key, space=_DIALOGUE,
                                                      scope_key=context.scope.stable_key()))
                        evidence.append(EvidenceEntry(
                            context.source_ref, item.hypothesis_key, space=_DIALOGUE, polarity=1,
                            evidence_key=(91524, 2, *proposal.origin.stable_key()),
                            scope_key=context.scope.stable_key(), payload_key=key,
                        ))
        if not additions:
            return state
        return state.with_(bindings=self._unique_bindings([*state.bindings, *additions]),
                           evidence=self._unique_evidence([*state.evidence, *evidence]))

    @staticmethod
    def _mark_root(state: QueryState, edge: FrontierEntry) -> QueryState:
        roots = tuple(sorted((
            root.expanded()
            if (root.owner_space == edge.owner_space
                and root.root_key == edge.root_key)
            else root
            for root in state.roots
        ), key=lambda item: (
            item.owner_space, item.status, item.priority_weight,
            len(item.root_key), item.root_key,
            len(item.source_ref), item.source_ref,
            len(item.scope_key), item.scope_key,
        )))
        return state.with_(roots=roots)

    def _distinct_relation_tie(self, state: QueryState) -> bool:
        """Return whether the ranked input tie contains different semantics.

        KdConv keeps domain provenance in separate physical rows.  Ranking
        must not turn two rows with the same integer subject/attribute/value
        identity into a false conflict, while genuinely different values or
        trained propositions must remain fail-closed.
        """
        best_key = state.best_candidate_key
        second_key = state.second_candidate_key
        if not best_key or not second_key or best_key == second_key:
            return False
        relations = tuple(
            item for item in self.input_structure.relation_candidates
            if item.proposition_key in {best_key, second_key})
        by_proposition = {}
        for item in relations:
            by_proposition.setdefault(item.proposition_key, []).append(item)
        best = by_proposition.get(best_key, ())
        second = by_proposition.get(second_key, ())
        if len(best) != 1 or len(second) != 1:
            return True

        def semantic_key(candidate):
            runtime = self.owner.kdconv_runtime
            row = (runtime._fact_rows.get(candidate.proposition_key)
                   if runtime is not None else None)
            if row is None:
                return (0, candidate.proposition_key, candidate.predicate_key,
                        candidate.structure_key, candidate.span_refs)
            return (1, row.subject_values, row.attribute_values,
                    row.value_values)

        return semantic_key(best[0]) != semantic_key(second[0])

    def _with_query_status(
            self,
            state: QueryState,
            edge: FrontierEntry | None,
            *,
            marginal_gain: int,
            ) -> QueryState:
        """把图扩展事实投影为综合终止字段，不在 driver 中写死闭合。"""
        if edge is not None:
            state = self._mark_root(state, edge)
        state = self.owner._refresh_reference_resolutions(state)
        state = self._bind_open_generation_contexts(state)
        frontier_exhausted = not unvisited_frontier(
            state.frontier, state.visited)
        if (not self._response_generation_attempted
                and frontier_exhausted
                and all(root.status != ROOT_PENDING for root in state.roots)):
            self._response_generation_attempted = True
            self.open_response_generations = generate_open_responses(
                self.owner.surface_generator, self.owner.core_runtime, state,
                tuple(self.open_generation_contexts[key] for key in sorted(self.open_generation_contexts)),
                self.owner._discourse_hypotheses, self.observed_surfaces,
                tuple(node for key, node in self.response_nodes.items()
                      if key not in self.generic_response_nodes),
                discourse_contexts=self.owner._reference_generation_contexts(),
                reference_resolutions=self.owner._reference_resolutions)
            augmented = []
            for response in self.open_response_generations:
                # The typed projection is already represented by the
                # QueryState EvidenceEntry payload. Preserve the exact state
                # evidence set so delivery validation remains authoritative.
                augmented.append(response)
            self.open_response_generations = tuple(augmented)
            if self.open_response_generations:
                generated_bindings = [BindingEntry(item.response_plan.response_act.stable_key(), item.stable_key(),
                                                   space=_DIALOGUE, scope_key=state.query_key)
                                      for item in self.open_response_generations]
                state = state.with_(bindings=self._unique_bindings([*state.bindings, *generated_bindings]))
            self.generic_response_generations = generate_generic_responses(
                self.owner.surface_generator,
                state,
                self.generic_hypotheses,
                tuple(self.generic_response_nodes.values()),
                self.owner._input_observation_ref,
                self.input_structure,
                self.owner._discourse_graph_input_requested_keys,
                tuple(self.hops),
            )
            if self.generic_response_generations:
                generated_bindings = [BindingEntry(
                    item.response_plan.response_act.stable_key(),
                    item.stable_key(),
                    space=_DIALOGUE,
                    scope_key=state.query_key,
                ) for item in self.generic_response_generations]
                state = state.with_(bindings=self._unique_bindings([
                    *state.bindings, *generated_bindings]))
        # 在综合终止字段计算前准备闭合命题的真实 Dialogue realization。
        # 之前仅在某个 Dialogue edge 恰好等于 best_candidate 时尝试，导致
        # frontier 已收敛但 generation_ready 仍被旧缓存污染，出现
        # ANSWER_CLOSED 而 response_surface 为空。这里要求本轮所有根已展开，
        # 并只为当前 Dialogue proposition roots 生成，保持三图同轮语义。
        if (frontier_exhausted
                and all(root.status != ROOT_PENDING for root in state.roots)):
            dialogue_roots = tuple(sorted({
                root.root_key for root in state.roots
                if root.owner_space == _DIALOGUE
                and root.root_key in self.owner.core_fact_index
            }))
            if dialogue_roots:
                self.owner._prepare_generation_plans(dialogue_roots, state)
        current_evidence = current_query_evidence(state.evidence)
        core_support = {
            item.hypothesis_key for item in current_evidence
            if (item.space == _CORE and item.polarity == 1
                and item.hypothesis_key in self.owner.core_fact_index)
        }
        core_support.update(
            item.hypothesis_key for item in current_evidence
            if item.space == _CORE and item.polarity == 1
            and item.hypothesis_key in self.owner.kdconv_fact_index)
        dialogue_support = {
            item.hypothesis_key for item in current_evidence
            if item.space == _DIALOGUE and item.polarity == 1
        }
        expected_memory_evidence = {
            item.evidence_key for item in self.owner._memory_evidence}
        bound_memory_evidence = {
            item.filler_key for item in state.bindings
            if (item.space == _MEMORY
                and item.role_key == (MEMORY_OBJECT_EVIDENCE,))
        }
        memory_complete = (
            not expected_memory_evidence
            or expected_memory_evidence <= bound_memory_evidence)
        selected = state.best_candidate_key
        closed = (
            bool(selected)
            and selected in core_support
            and selected in dialogue_support
            and selected in self.owner.generation_ready_propositions
            and memory_complete
            and (selected in self.owner._kdconv_query_propositions
                 or self.input_structure.concept_evidence_closed(
                     selected, frozenset(core_support)))
        )
        reference_open = any(
            item.requires_resolution and item.status != REFERENCE_RESOLVED
            for item in self.owner._reference_resolutions)
        reference_conflict = any(
            item.status == REFERENCE_CONFLICT
            for item in self.owner._reference_resolutions)
        closed = closed and not reference_open and not reference_conflict
        competition_keys = {
            item.hypothesis_key: memory_competition_identity(
                item.candidate_key, item.competition_key, item.observation_key)
            for item in self.owner._memory_hypotheses}
        reference_competition_keys = {
            candidate.hypothesis.stable_key(): candidate.hypothesis.competition_key
            for resolution in self.owner._reference_resolutions
            for candidate in resolution.candidates
        }
        competition_stances: dict[tuple[int, tuple[int, ...]], set[int]] = {}
        for item in current_evidence:
            if item.space == _MEMORY:
                occurrence = self.response_occurrence_nodes.get(item.hypothesis_key)
                if occurrence is not None:
                    if item != occurrence.query_evidence():
                        raise ValueError("角色发生证据不能替换为内容支持")
                    key = (_MEMORY, occurrence.stable_key())
                else:
                    competition = competition_keys.get(item.hypothesis_key)
                    if competition is None:
                        competition = reference_competition_keys.get(
                            item.hypothesis_key, item.hypothesis_key)
                    key = (_MEMORY, competition)
            elif item.space == _CORE:
                key = (_CORE, item.hypothesis_key)
            else:
                continue
            competition_stances.setdefault(key, set()).add(item.polarity)
        evidence_conflict = any({1, 2} <= stances
                                for stances in competition_stances.values())
        conflict_open = int(
            self.input_order_conflict
                or reference_conflict
            or any(item.conflict_kept == 1 for item in state.bindings)
            or evidence_conflict
            or (
                bool(state.second_candidate_key)
                and state.best_score > 0
                and state.best_score == state.second_score
                and self._distinct_relation_tie(state)
            )
        )
        return state.with_(
            required_slots_open=int(
                self.input_required_slots_open or not core_support or reference_open),
            evidence_closed=1 if closed else 0,
            conflict_open=conflict_open,
            best_score=state.best_score,
            second_score=state.second_score,
            marginal_gain=marginal_gain,
            generation_ready=1 if closed else 0,
        )

    @staticmethod
    def _unique_bindings(
            values: list[BindingEntry],
            ) -> tuple[BindingEntry, ...]:
        unique = tuple(sorted(values, key=lambda item: (
            item.space, len(item.role_key), item.role_key,
            len(item.filler_key), item.filler_key,
            len(item.scope_key), item.scope_key, item.conflict_kept)))
        return tuple(item for index, item in enumerate(unique)
                     if index == 0 or item != unique[index - 1])

    @staticmethod
    def _unique_evidence(
            values: list[EvidenceEntry],
            ) -> tuple[EvidenceEntry, ...]:
        unique = tuple(sorted(values, key=lambda item: (
            item.space, item.polarity, item.trust,
            len(item.source_ref), item.source_ref,
            len(item.hypothesis_key), item.hypothesis_key,
            len(item.evidence_key), item.evidence_key,
            len(item.scope_key), item.scope_key,
            len(item.payload_key), item.payload_key,
            len(item.supersedes_key), item.supersedes_key)))
        return tuple(item for index, item in enumerate(unique)
                     if index == 0 or item != unique[index - 1])

    @staticmethod
    def _unique_visited(
            values: list[VisitedKey],
            ) -> tuple[VisitedKey, ...]:
        unique = tuple(sorted(values, key=lambda item: (
            item.space, item.direction, len(item.node_key), item.node_key,
            len(item.binding_digest), item.binding_digest)))
        return tuple(item for index, item in enumerate(unique)
                     if index == 0 or item != unique[index - 1])

    def _record_relation_capability(self, route, bindings: list[BindingEntry],
                                    evidence: list[EvidenceEntry]) -> None:
        """Record route reachability without converting it to fact support."""
        self.relation_capability_routes[route.proposition_key] = route
        bindings.append(BindingEntry(
            role_key=route.binding_key,
            filler_key=route.stable_key(),
            space=_CORE,
            scope_key=route.scope_key,
        ))
        # Capability is represented by the binding and the dedicated trace
        # registry.  Do not add a synthetic Core EvidenceEntry: its payload
        # is a route record, not an EvidenceRecord, and therefore cannot be
        # consumed as claim evidence by closure or replay validators.

    def _record_typed_relation(
            self, fact: ActiveRelationSurface, route,
            bindings: list[BindingEntry], evidence: list[EvidenceEntry],
            ) -> None:
        """将已有 PROPERTY/EVENT_TIME 字段加入当前共同 QueryState。"""
        projection = project_typed_relation(fact, route)
        if projection is None:
            return
        self.typed_relation_projections[projection.proposition_key] = projection
        self.owner.typed_relation_projections[projection.proposition_key] = projection
        bindings.extend(projection.binding_entries())
        evidence.append(projection.evidence_entry())

    def _record_open_typed_relation(
            self, candidate, route, bindings: list[BindingEntry],
            evidence: list[EvidenceEntry],
            ) -> None:
        """把开放输入 span 作为 UNKNOWN typed 候选接入共同状态。"""
        projection = project_open_typed_relation(candidate, route)
        if projection is None:
            return
        self.typed_relation_projections[projection.proposition_key] = projection
        bindings.extend(projection.binding_entries())
        evidence.append(projection.evidence_entry())

    def _is_artifact_envelope_root(
            self, edge: FrontierEntry, origin: tuple[int, ...],
            ) -> bool:
        """仅将本次输入自身的 envelope root 解释为 carrier，不碰同形 Core 节点。"""
        if self.artifact_input is None:
            return False
        envelope_key = self.artifact_input.envelope.identity.stable_key()
        return (
            edge.owner_space == _CORE
            and origin == envelope_key
            and edge.root_key == envelope_key
            and edge.scope_key == self.artifact_input.envelope.scope.stable_key()
        )

    def _expand_artifact_envelope(
            self, state: QueryState, remaining: tuple[FrontierEntry, ...],
            edge: FrontierEntry, origin: tuple[int, ...],
            ) -> ExpansionOutcome:
        """展开 carrier 本体到明确 projection/Evidence，不从原始单元猜测语义。"""
        artifact = self.artifact_input
        if artifact is None:
            raise RuntimeError("artifact envelope 扩展缺少本次 ArtifactQueryInput")
        envelope_key = artifact.envelope.identity.stable_key()
        if origin != envelope_key:
            raise ValueError("artifact envelope root 与输入身份不一致")
        depth = edge.depth + 1
        scope_key = artifact.envelope.scope.stable_key()
        source_ref = artifact.source_ref
        source_marker = source_ref[0]
        bindings = list(state.bindings)
        evidence = list(state.evidence)
        hops: list[QueryHop] = []
        bindings.append(BindingEntry(
            _ARTIFACT_ENVELOPE_ROLE, envelope_key,
            space=_CORE, scope_key=scope_key))
        for anchor in artifact.anchors:
            bindings.append(BindingEntry(
                (*_ARTIFACT_LOCAL_ROLE, 1), anchor.identity.stable_key(),
                space=_CORE, scope_key=scope_key))
        for node in artifact.structure_nodes:
            bindings.append(BindingEntry(
                (*_ARTIFACT_LOCAL_ROLE, 2), node.identity.stable_key(),
                space=_CORE, scope_key=scope_key))
        for projection in artifact.understanding_projections:
            projection_key = projection.identity.stable_key()
            semantic_key = projection.semantic_object.stable_key()
            bindings.append(BindingEntry(
                _ARTIFACT_PROJECTION_ROLE, projection_key,
                space=_CORE, scope_key=scope_key))
            bindings.append(BindingEntry(
                _ARTIFACT_SEMANTIC_ROLE, semantic_key,
                space=_CORE, scope_key=scope_key))
            for item in projection.evidence:
                evidence.append(EvidenceEntry(
                    item.source.stable_key(), projection.hypothesis.stable_key(),
                    space=_CORE, polarity=item.stance, trust=1,
                    evidence_key=item.stable_key(),
                    scope_key=projection.scope.stable_key(),
                    payload_key=projection.stable_key(),
                ))
            bridge = next((item for item in self.artifact_bridge_matches
                           if item.projection_key == projection_key), None)
            if bridge is not None:
                # This is a structural bridge to an already learned Core
                # proposition.  It is not a new support claim, so bridge
                # Evidence remains UNKNOWN while the original projection
                # Evidence retains its own stance.
                bridge_key = bridge.natural_key()
                bindings.extend((
                    BindingEntry((*_ARTIFACT_BRIDGE_ROLE, 1), projection_key,
                                 space=_CORE, scope_key=scope_key),
                    BindingEntry((*_ARTIFACT_BRIDGE_ROLE, 2),
                                 bridge.semantic_object_key,
                                 space=_CORE, scope_key=scope_key),
                    BindingEntry((*_ARTIFACT_BRIDGE_ROLE, 3),
                                 bridge.proposition_key,
                                 space=_CORE, scope_key=scope_key),
                ))
                evidence.append(EvidenceEntry(
                    bridge.source_ref, bridge.proposition_key,
                    space=_CORE, polarity=3, trust=1,
                    evidence_key=(*_ARTIFACT_BRIDGE_EVIDENCE, *bridge_key),
                    scope_key=bridge.scope_key,
                    payload_key=bridge_key,
                ))
                hops.append(QueryHop(
                    _CORE, projection_key, _ARTIFACT_BRIDGE_PREDICATE,
                    semantic_key, bridge.proposition_key, (), depth,
                    source_marker))
            hops.append(QueryHop(
                _CORE, projection_key, _ARTIFACT_PROJECTION_PREDICATE,
                envelope_key, semantic_key, (), depth, source_marker))
        for reference in artifact.references:
            reference_key = reference.identity.stable_key()
            source_anchor_key = reference.anchor_identity.stable_key()
            bindings.append(BindingEntry(
                _ARTIFACT_REFERENCE_ROLE, reference_key,
                space=_CORE, scope_key=scope_key))
            bindings.append(BindingEntry(
                _ARTIFACT_REFERENCE_SOURCE_ROLE, source_anchor_key,
                space=_CORE, scope_key=scope_key))
            bindings.append(BindingEntry(
                _ARTIFACT_REFERENCE_RELATION_ROLE,
                reference.relation.stable_key(),
                space=_CORE, scope_key=scope_key))
            bindings.append(BindingEntry(
                _ARTIFACT_REFERENCE_STATE_ROLE, (reference.target_state,),
                space=_CORE, scope_key=scope_key))
            if reference.target_source is not None:
                bindings.append(BindingEntry(
                    _ARTIFACT_REFERENCE_TARGET_ROLE,
                    reference.target_source.stable_key(),
                    space=_CORE, scope_key=scope_key))
            if reference.target_anchor is not None:
                bindings.append(BindingEntry(
                    _ARTIFACT_REFERENCE_TARGET_ROLE,
                    reference.target_anchor.stable_key(),
                    space=_CORE, scope_key=scope_key))
            if reference.target_fingerprint:
                bindings.append(BindingEntry(
                    _ARTIFACT_REFERENCE_FINGERPRINT_ROLE,
                    reference.target_fingerprint,
                    space=_CORE, scope_key=scope_key))
            evidence.append(EvidenceEntry(
                reference.source.stable_key(), reference_key,
                space=_CORE, polarity=3, trust=1,
                evidence_key=reference_key, scope_key=reference.scope.stable_key(),
                payload_key=reference.stable_key(),
            ))
            target_key = (
                reference.target_anchor.stable_key()
                if reference.target_anchor is not None
                else (reference.target_source.stable_key()
                      if reference.target_source is not None
                      else source_anchor_key))
            hops.append(QueryHop(
                _CORE, reference_key, reference.relation.stable_key(),
                envelope_key, target_key, (), depth, source_marker))
        if not hops:
            hops.append(QueryHop(
                _CORE, envelope_key, _ARTIFACT_ENVELOPE_PREDICATE,
                envelope_key, envelope_key, (), depth, source_marker))
        self.hops.extend(hops)
        visited = self._unique_visited([
            *state.visited,
            VisitedKey(_CORE, origin, direction=edge.direction),
        ])
        next_state = state.with_(
            depth=max(state.depth, depth),
            frontier=remaining,
            visited=visited,
            bindings=self._unique_bindings(bindings),
            evidence=self._unique_evidence(evidence),
            node_count=(state.node_count + 1
                        + len(artifact.understanding_projections)
                        + len(artifact.references)),
            edge_count=state.edge_count + len(hops),
            read_count=(state.read_count + 1
                        + len(artifact.understanding_projections)
                        + len(artifact.references)),
            score=(state.score + len(hops)
                   + len(artifact.understanding_projections)
                   + len(artifact.references)),
        )
        next_state = self._with_query_status(
            next_state, edge, marginal_gain=(
                len(hops) + len(artifact.understanding_projections)
                + len(artifact.references)))
        return ExpansionOutcome(next_state, edge, len(hops))

    def expand(self, state: QueryState,
               edge: FrontierEntry) -> ExpansionOutcome:
        remaining = tuple(item for item in state.frontier if item != edge)
        origin = edge.target_key or tuple(edge.edge_key[1:])
        self.expanded_edges.append(edge.stable_key())
        if any(
                item.space == edge.owner_space
                and item.node_key == origin
                and item.direction == edge.direction
                for item in state.visited):
            next_state = self._with_query_status(
                state.with_(frontier=remaining), edge, marginal_gain=0)
            return ExpansionOutcome(
                next_state, edge, 0)
        space = edge.owner_space
        if self._is_artifact_envelope_root(edge, origin):
            return self._expand_artifact_envelope(state, remaining, edge, origin)
        occurrence = self.response_occurrence_nodes.get(origin) if space == _MEMORY else None
        if occurrence is not None:
            next_state = occurrence.expand(state, edge)
            self.hops.append(QueryHop(
                space, origin, occurrence.role.stable_key(), occurrence.proof_ref.stable_key(),
                occurrence.occurrence.stable_key(), (), edge.depth + 1, occurrence.source_ref[0]))
            next_state = self._with_query_status(next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        input_occurrence = self.input_occurrence_nodes.get(origin) if space == _MEMORY else None
        if input_occurrence is not None:
            next_state = input_occurrence.expand(state, edge)
            self.hops.append(QueryHop(
                space, origin, input_occurrence.role.stable_key(),
                input_occurrence.hypothesis.stable_key(),
                input_occurrence.occurrence.stable_key(), input_occurrence.values,
                edge.depth + 1, input_occurrence.source_ref[0]))
            next_state = self._with_query_status(next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        candidate_set = self.reference_candidate_nodes.get(origin) if space == _MEMORY else None
        if candidate_set is not None:
            next_state = candidate_set.expand(state, edge)
            self.hops.append(QueryHop(
                space, origin, candidate_set.reference.role.stable_key(),
                candidate_set.reference.hypothesis.stable_key(),
                candidate_set.stable_key(), (), edge.depth + 1,
                candidate_set.reference.source_ref[0]))
            next_state = self._with_query_status(next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        delivered_slot = (
            self.delivered_graph_slot_nodes.get(origin)
            if space == _MEMORY else None)
        if delivered_slot is not None:
            next_state = delivered_slot.expand(state, edge)
            self.hops.append(QueryHop(
                space,
                origin,
                delivered_graph_surface_role(
                    delivered_slot.category, delivered_slot.proof_ref),
                delivered_slot.hypothesis.stable_key(),
                delivered_slot.target.stable_key(),
                (),
                edge.depth + 1,
                delivered_slot.source_ref[0],
            ))
            next_state = self._with_query_status(
                next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        response_node = self.response_nodes.get(origin) if space == _DIALOGUE else None
        if response_node is not None:
            next_state = response_node.expand(state, edge)
            response_origin = (
                response_node.context_key
                if isinstance(response_node, GenericResponseConnectorQueryNode)
                else response_node.candidate.frame.schema_key())
            self.hops.append(QueryHop(
                space, origin, response_node.learned.stance.stable_key(),
                response_origin, response_node.learned.template.connector.stable_key(),
                (), edge.depth + 1, response_node.source_hash))
            next_state = self._with_query_status(next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        open_node = self.open_nodes.get((space, origin))
        if open_node is not None:
            next_state = open_node.expand(state, edge)
            if space == _CORE:
                route = self.owner.relation_routes.for_proposition(
                    open_node.frame.proposition)
                bindings = list(next_state.bindings)
                evidence = list(next_state.evidence)
                fact = self.owner.core_fact_index.get(open_node.frame.proposition)
                # Extension/open frames can be represented in the shared
                # frontier before a Core capability route exists (for example
                # an unseen Event/Time or KdConv role frame). Keep the open
                # structural evidence, but do not invent a route or crash the
                # query; factual closure remains fail-closed downstream.
                if route is not None:
                    self._record_relation_capability(route, bindings, evidence)
                if fact is not None and route is not None:
                    self._record_typed_relation(
                        fact, route, bindings, evidence)
                next_state = next_state.with_(
                    bindings=self._unique_bindings(bindings),
                    evidence=self._unique_evidence(evidence),
                )
            elif space == _DIALOGUE and open_node.candidate is not None:
                route = self.owner.relation_routes.for_proposition(
                    open_node.frame.proposition)
                bindings = list(next_state.bindings)
                evidence = list(next_state.evidence)
                if route is not None:
                    self._record_open_typed_relation(
                        open_node.candidate, route, bindings, evidence)
                next_state = next_state.with_(
                    bindings=self._unique_bindings(bindings),
                    evidence=self._unique_evidence(evidence),
                )
            self.hops.append(QueryHop(
                space, origin, open_node.frame.predicate,
                open_node.frame.schema_key(), origin, (), edge.depth + 1,
                open_node.frame.source_ref[0]))
            next_state = self._with_query_status(next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        if space == _CORE:
            outcome = self._expand_core(state, remaining, edge, origin)
        elif space == _MEMORY:
            outcome = self._expand_memory(state, remaining, edge, origin)
        elif space == _DIALOGUE:
            outcome = self._expand_dialogue(state, remaining, edge, origin)
        else:
            raise ValueError(f"query bridge 遇到未注册空间: {space}")
        return outcome

    def _expand_core(self, state: QueryState, remaining: tuple[FrontierEntry, ...],
                     edge: FrontierEntry,
                     origin: tuple[int, ...]) -> ExpansionOutcome:
        facts = self.owner.filler_edges.get(origin, ())
        if self.owner._kdconv_query_propositions:
            facts = tuple(
                fact for fact in facts
                if fact.proposition.stable_key()
                in self.owner._kdconv_query_propositions
                or fact.proposition.stable_key()
                not in self.owner.kdconv_fact_index
            )
        topology_edges = tuple(
            item for item in self.owner.event_time_graph_adjacency.get(origin, ())
            if ((self.owner._event_time_graph_input_keys
                 or item.stable_key()
                 in self.owner._event_time_query_edge_keys)
                and (item.subject_key == origin
                     or (item.object_key == origin and item.permits_reverse))))
        # A legal graph identity need not be a trained surface filler.  For
        # such roots, consume the authoritative graph statements directly in
        # this same Core frontier.  The marker is carried on descendant
        # edges, so arbitrary graph depth remains available without scanning
        # unrelated ontology objects.
        generic_statements = ()
        # An explicit generic identity opens the unfiltered first-hop route.
        # Descendant 21610 hops may continue to arbitrary depth, but only
        # inside the assertion scope carried by that hop.  Treating every
        # descendant as a new unscoped root made one connected graph consume
        # the entire ontology and starved lower-weight Memory/Dialogue roots.
        generic_root = origin in self.owner._generic_graph_input_keys
        generic_descendant = (
            bool(edge.source_ref)
            and edge.source_ref[0] == 21610
            and len(edge.source_ref) == 2
            and bool(edge.scope_key))
        generic_route = generic_root or generic_descendant
        # A proposition supplied as an explicit graph root is an active Core
        # relation in its own right.  Consume that complete fact in this same
        # frontier before inspecting raw ontology adjacency; the latter is
        # retained as structural evidence but must not stand in for the
        # proposition's ordered relation members.
        direct_fact = (
            self.owner.core_fact_index.get(origin)
            if generic_route else None)
        if direct_fact is not None and direct_fact not in facts:
            facts = (direct_fact, *facts)
        if generic_route and direct_fact is None:
            ontology = self.owner.core_runtime.context.graph_ontology
            identity = ObjectIdentity.from_stable_key(origin)
            ref = ontology.resolve(identity)
            if ref is None:
                raise RuntimeError("通用图输入在展开时丢失本体对象")
            statements = list(ontology.statements(subject=ref))
            statements.extend(ontology.statements(object_ref=ref))
            if generic_descendant:
                statements = [
                    item for item in statements
                    if item.assertion.scope.stable_key() == edge.scope_key]
            # A proposition's ontology scope also contains persistence
            # scaffolding (occurrence, role-binding, context-scope and
            # representation records).  Those records are evidence of how
            # the graph was materialized, not semantic graph slots.  They may
            # be retained by the source graph, but must not become recursive
            # frontier branches: doing so turns one proposition into a
            # scope-wide crawl and starves the lower-weight roots.
            statements = [
                item for item in statements
                if self.owner.core_runtime.context.graph_ontology.identity_of(
                    item.subject).object_kind in _GENERIC_SEMANTIC_OBJECT_KINDS
                and self.owner.core_runtime.context.graph_ontology.identity_of(
                    item.object).object_kind in _GENERIC_SEMANTIC_OBJECT_KINDS
            ]
            generic_statements = tuple(sorted({
                item.assertion_hash: item for item in statements
            }.values(), key=lambda item: item.assertion_hash))
        if not facts and not topology_edges and not generic_statements:
            visited = self._unique_visited([
                *state.visited,
                VisitedKey(_CORE, origin, direction=edge.direction),
            ])
            next_state = self._with_query_status(
                state.with_(frontier=remaining, visited=visited),
                edge,
                marginal_gain=0,
            )
            return ExpansionOutcome(
                next_state, edge, 0)
        depth = edge.depth + 1
        core_visited = {
            item.node_key for item in state.visited if item.space == _CORE}
        core_visited.add(origin)
        next_edges: list[FrontierEntry] = []
        new_bindings = list(state.bindings)
        new_evidence = list(state.evidence)
        hops: list[QueryHop] = []
        discovered: set[tuple[int, ...]] = set()
        for fact in facts:
            proposition = fact.proposition.stable_key()
            predicate = fact.predicate.stable_key()
            kdconv_fact = proposition in self.owner.kdconv_fact_index
            if kdconv_fact:
                source_scope = (fact.source_hash,)
                new_evidence.append(EvidenceEntry(
                    source_scope, proposition, space=_CORE, polarity=1,
                    trust=1, evidence_key=(91630, 3, *proposition),
                    scope_key=source_scope, payload_key=(91630, 4, *proposition)))
            else:
                source = self.owner.core_runtime.generation_input(fact.proposition)
                source_scope = source.hypothesis.scope.stable_key()
                route = self.owner.relation_routes.for_proposition(proposition)
                if route is None:
                    raise RuntimeError(
                        "active Core relation 没有 QueryState capability route")
                self._record_relation_capability(
                    route, new_bindings, new_evidence)
                self._record_typed_relation(
                    fact, route, new_bindings, new_evidence)
                new_evidence.extend(core_query_evidence(
                    fact.proposition,
                    self.owner.core_runtime.evidence_history(fact.proposition),
                    self.owner.core_runtime.context.scoped_identity_store.registry))
            for binding in fact.bindings:
                filler = binding.filler.stable_key()
                new_bindings.append(BindingEntry(
                    role_key=binding.role.stable_key(),
                    filler_key=filler,
                    space=_CORE,
                    scope_key=source_scope,
                ))
                if filler == origin or filler in core_visited:
                    continue
                if filler in discovered:
                    continue
                discovered.add(filler)
                hops.append(QueryHop(
                    _CORE, proposition, predicate, origin, filler,
                    tuple(ord(value) for value in binding.surface),
                    depth, fact.source_hash,
                    tuple(item.filler.stable_key() for item in fact.bindings)))
                next_edges.append(FrontierEntry(
                    (_CORE, *filler), _CORE,
                    target_key=filler,
                    root_key=edge.root_key,
                    scope_key=edge.scope_key,
                    source_ref=(fact.source_hash,),
                    direction=edge.direction,
                    owner_weight=self.owner.space_weights[0],
                    depth=depth, required_slot_gain=1,
                    evidence_support=1,
                    relation_fit=1,
                ))
        for statement in generic_statements:
            subject_key = self.owner.core_runtime.context.graph_ontology.identity_of(
                statement.subject).stable_key()
            object_key = self.owner.core_runtime.context.graph_ontology.identity_of(
                statement.object).stable_key()
            predicate_key = self.owner.core_runtime.context.graph_ontology.identity_of(
                statement.predicate).stable_key()
            outgoing = subject_key == origin
            filler = object_key if outgoing else subject_key
            direction = 1 if outgoing else 2
            role_key = predicate_key if outgoing else (21610, 2, *predicate_key)
            source_ref = (21610, statement.assertion_hash)
            scope_key = statement.assertion.scope.stable_key()
            assertion_key = statement.assertion.stable_key()
            # The explicit graph root is a first-class participant in the
            # same QueryState relation.  Without this marker, graph-only
            # multi-slot connectors can see the adjacent endpoint but cannot
            # prove the root/proposition slot from QueryState bindings.  Keep
            # the marker integer-only and tie it to the same assertion scope;
            # it is evidence, never a source-surface answer.
            new_bindings.append(BindingEntry(
                role_key=(21610, 1),
                filler_key=origin,
                space=_CORE,
                scope_key=scope_key,
            ))
            new_evidence.append(EvidenceEntry(
                source_ref,
                origin,
                space=_CORE,
                polarity=3,
                trust=1,
                evidence_key=(21610, 1, statement.assertion_hash),
                scope_key=scope_key,
                payload_key=assertion_key,
            ))
            new_bindings.append(BindingEntry(
                role_key=role_key,
                filler_key=filler,
                space=_CORE,
                scope_key=scope_key,
            ))
            new_evidence.append(EvidenceEntry(
                source_ref,
                origin,
                space=_CORE,
                polarity=3,
                trust=1,
                evidence_key=(21610, 3, statement.assertion_hash),
                scope_key=scope_key,
                payload_key=assertion_key,
            ))
            if filler in core_visited or filler in discovered:
                continue
            discovered.add(filler)
            hops.append(QueryHop(
                # Preserve the ontology traversal role in the hop.  Incoming
                # generic statements use the explicit inverse marker, which
                # lets response binding recover subject/object order without
                # stable-key sorting or a second graph query.
                _CORE, origin, role_key, origin, filler, (), depth,
                statement.assertion_hash,
                # The connector consumes the authoritative relation order from
                # this same Core assertion.  Traversal direction is only the
                # frontier route; it must never replace subject/object role
                # order with ``origin``/``filler`` order.
                (subject_key, object_key)))
            next_edges.append(FrontierEntry(
                (_CORE, 21610, direction, *filler,
                 statement.assertion_hash),
                _CORE,
                target_key=filler,
                root_key=edge.root_key,
                scope_key=scope_key,
                source_ref=source_ref,
                direction=direction,
                owner_weight=self.owner.space_weights[0],
                depth=depth,
                required_slot_gain=1,
                evidence_support=1,
                relation_fit=2,
                source_trust=1,
            ))
        for topology_edge in topology_edges:
            topology_key = topology_edge.stable_key()
            self.event_time_graph_expanded_edges.append(topology_key)
            if topology_edge.subject_key == origin:
                filler = topology_edge.object_key
                direction = 1
            elif topology_edge.object_key == origin:
                filler = topology_edge.subject_key
                direction = 2
            else:
                raise RuntimeError("Event/Time adjacency 包含不相邻 edge")
            new_bindings.append(BindingEntry(
                role_key=topology_edge.predicate_key,
                filler_key=filler,
                space=_CORE,
                scope_key=topology_edge.scope_key,
            ))
            new_evidence.append(EvidenceEntry(
                topology_edge.source_ref,
                topology_edge.subject_key,
                space=_CORE,
                polarity=3,
                trust=1,
                evidence_key=(*_EVENT_TIME_GRAPH_EVIDENCE, *topology_key),
                scope_key=topology_edge.scope_key,
                payload_key=topology_key,
            ))
            if filler in core_visited or filler in discovered:
                continue
            discovered.add(filler)
            hops.append(QueryHop(
                _CORE,
                topology_edge.subject_key,
                topology_edge.predicate_key,
                origin,
                filler,
                (),
                depth,
                self._source_hash(topology_edge.source_ref, 0),
            ))
            next_edges.append(FrontierEntry(
                (_CORE, *_EVENT_TIME_GRAPH_EDGE, direction,
                 *filler, *topology_key),
                _CORE,
                target_key=filler,
                root_key=edge.root_key,
                scope_key=topology_edge.scope_key,
                source_ref=topology_edge.source_ref,
                direction=direction,
                owner_weight=self.owner.space_weights[0],
                depth=depth,
                required_slot_gain=1,
                evidence_support=1,
                relation_fit=2,
                source_trust=1,
            ))
        self.hops.extend(hops)
        visited = self._unique_visited([
            *state.visited,
            VisitedKey(_CORE, origin, direction=edge.direction),
        ])
        next_state = state.with_(
                depth=max(state.depth, depth),
                frontier=tuple(sorted(
                    (*remaining, *next_edges),
                    key=lambda entry: entry.priority_tuple(), reverse=True)),
                visited=visited,
                bindings=self._unique_bindings(new_bindings),
                evidence=self._unique_evidence(new_evidence),
                node_count=state.node_count + len(facts) + len(topology_edges)
                + len(generic_statements),
                edge_count=state.edge_count + len(hops),
                read_count=state.read_count + len(facts) + len(topology_edges)
                + len(generic_statements),
                score=state.score + len(hops) + len(facts) + len(topology_edges)
                + len(generic_statements),
        )
        next_state = self._with_query_status(
            next_state,
            edge,
            marginal_gain=len(hops) + len(facts) + len(topology_edges)
            + len(generic_statements),
        )
        return ExpansionOutcome(
            next_state,
            edge,
            len(facts) + len(topology_edges) + len(generic_statements),
        )

    def _expand_memory(self, state: QueryState, remaining: tuple[FrontierEntry, ...],
                       edge: FrontierEntry,
                       origin: tuple[int, ...]) -> ExpansionOutcome:
        """沿匹配的 Observation -> Hypothesis -> Evidence 图展开。

        三层都由活动 manifest 恢复的对象引用驱动。这里不读 ``raw_text``、
        ``recent_turns()`` 或 posting；历史记录须经内容节点或显式观察关联边进入。
        """
        observation = next((
            item for item in self.owner._memory_observations
            if item.observation_key == origin), None)
        hypothesis = next((
            item for item in self.owner._memory_hypotheses
            if item.hypothesis_key == origin), None)
        evidence = next((
            item for item in self.owner._memory_evidence
            if item.evidence_key == origin), None)
        if observation is None and hypothesis is None and evidence is None:
            visited = self._unique_visited([
                *state.visited,
                VisitedKey(_MEMORY, origin, direction=edge.direction),
            ])
            next_state = self._with_query_status(
                state.with_(frontier=remaining, visited=visited),
                edge,
                marginal_gain=0,
            )
            return ExpansionOutcome(next_state, edge, 0)

        depth = edge.depth + 1
        next_edges: list[FrontierEntry] = []
        new_bindings = list(state.bindings)
        new_evidence = list(state.evidence)
        if observation is not None:
            # Observation only establishes its context. Evidence is added in
            # the Evidence layer so its support/refute/unknown stance remains
            # recoverable instead of being silently converted to support.
            new_bindings.append(BindingEntry(
                role_key=(MEMORY_OBJECT_OBSERVATION,),
                filler_key=observation.observation_key,
                space=_MEMORY,
                scope_key=observation.scope_key,
            ))
            hop = QueryHop(
                _MEMORY,
                observation.observation_key,
                (MEMORY_EVENT_OBSERVATION,),
                (MEMORY_OBJECT_OBSERVATION, observation.turn_seq,
                 observation.speaker_kind),
                observation.observation_key,
                (),
                depth,
                observation.source_hash,
            )
            for link in observation.relation_refs:
                new_bindings.append(BindingEntry((91533, 1), link.stable_key(), space=_MEMORY,
                                                  scope_key=observation.scope_key))
                target = link.value()
                if isinstance(target, MemoryObjectRef):
                    self.hops.append(QueryHop(_MEMORY, observation.observation_key, (91533, 1),
                                              observation.observation_key, target.stable_key(), (), depth,
                                              observation.source_hash))
                    if target.object_kind in {MEMORY_OBJECT_OBSERVATION, MEMORY_OBJECT_HYPOTHESIS,
                                               MEMORY_OBJECT_EVIDENCE}:
                        next_edges.append(FrontierEntry(
                            (_MEMORY, *target.stable_key()), _MEMORY, target_key=target.stable_key(),
                            root_key=edge.root_key, source_ref=observation.source_ref,
                            scope_key=observation.scope_key, direction=1,
                            owner_weight=self.owner.space_weights[1], depth=depth,
                            recency_weight=observation.turn_seq))
            for item in self.owner._memory_hypotheses:
                if (item.observation_key != observation.observation_key
                        or item.hypothesis_key not in observation.hypothesis_keys):
                    continue
                next_edges.append(FrontierEntry(
                    (_MEMORY, *item.hypothesis_key), _MEMORY,
                    target_key=item.hypothesis_key,
                    root_key=edge.root_key,
                    scope_key=item.scope_key,
                    source_ref=item.source_ref,
                    direction=edge.direction,
                    owner_weight=self.owner.space_weights[1],
                    depth=depth,
                    required_slot_gain=2,
                    evidence_support=1,
                    discourse_fit=1,
                    source_trust=1,
                    recency_weight=observation.turn_seq,
                ))
                # Evidence is an independently addressable graph layer.  Put
                # every lineage edge on this same frontier as soon as its
                # Observation is expanded, so an UNKNOWN stance cannot be
                # lost merely because a sibling Hypothesis was not selected
                # as a Dialogue root.
                for evidence_item in self.owner._memory_evidence:
                    if (evidence_item.hypothesis_key != item.hypothesis_key
                            or evidence_item.evidence_key not in item.evidence_keys):
                        continue
                    next_edges.append(FrontierEntry(
                        (_MEMORY, *evidence_item.evidence_key), _MEMORY,
                        target_key=evidence_item.evidence_key,
                        root_key=edge.root_key,
                        scope_key=evidence_item.scope_key,
                        source_ref=evidence_item.source_ref,
                        direction=edge.direction,
                        owner_weight=self.owner.space_weights[1],
                        depth=depth + 1,
                        required_slot_gain=1,
                        evidence_support=1,
                        source_trust=1,
                        recency_weight=observation.turn_seq,
                    ))
        elif hypothesis is not None:
            new_bindings.append(BindingEntry(
                role_key=hypothesis.hypothesis_kind,
                filler_key=hypothesis.candidate_key,
                space=_MEMORY,
                scope_key=hypothesis.scope_key,
            ))
            hop = QueryHop(
                _MEMORY,
                hypothesis.hypothesis_key,
                (MEMORY_EVENT_HYPOTHESIS,),
                hypothesis.observation_key,
                hypothesis.candidate_key,
                (),
                depth,
                hypothesis.source_ref[0],
            )
            for item in self.owner._memory_evidence:
                if (item.hypothesis_key != hypothesis.hypothesis_key
                        or item.evidence_key not in hypothesis.evidence_keys):
                    continue
                next_edges.append(FrontierEntry(
                    (_MEMORY, *item.evidence_key), _MEMORY,
                    target_key=item.evidence_key,
                    root_key=edge.root_key,
                    scope_key=item.scope_key,
                    source_ref=item.source_ref,
                    direction=edge.direction,
                    owner_weight=self.owner.space_weights[1],
                    depth=depth,
                    required_slot_gain=2,
                    evidence_support=1,
                    source_trust=1,
                    recency_weight=edge.recency_weight,
                ))
        else:
            assert evidence is not None
            new_bindings.append(BindingEntry(
                role_key=(MEMORY_OBJECT_EVIDENCE,),
                filler_key=evidence.evidence_key,
                space=_MEMORY,
                scope_key=evidence.scope_key,
            ))
            new_evidence.append(EvidenceEntry(
                evidence.source_ref,
                evidence.hypothesis_key,
                space=_MEMORY,
                polarity=evidence.stance,
                trust=1,
                evidence_key=evidence.evidence_key,
                scope_key=evidence.scope_key,
                payload_key=evidence.payload_key,
                supersedes_key=evidence.supersedes_key,
            ))
            # The Companion/Dialogue graph receives the same source-qualified
            # O/H/E evidence in this QueryState.  This is a parallel
            # projection, not a fallback route or a copied surface.
            new_evidence.append(EvidenceEntry(
                evidence.source_ref,
                evidence.hypothesis_key,
                space=_DIALOGUE,
                polarity=evidence.stance,
                trust=1,
                evidence_key=evidence.evidence_key,
                scope_key=evidence.scope_key,
                payload_key=evidence.payload_key,
                supersedes_key=evidence.supersedes_key,
            ))
            hop = QueryHop(
                _MEMORY,
                evidence.evidence_key,
                (MEMORY_EVENT_EVIDENCE,),
                evidence.hypothesis_key,
                evidence.evidence_key,
                (),
                depth,
                evidence.source_ref[0],
            )

        self.hops.append(hop)
        visited = self._unique_visited([
            *state.visited,
            VisitedKey(_MEMORY, origin, direction=edge.direction),
        ])
        next_state = state.with_(
                depth=max(state.depth, depth),
                frontier=tuple(sorted(
                    (*remaining, *next_edges),
                    key=lambda entry: entry.priority_tuple(), reverse=True)),
                visited=visited,
                bindings=self._unique_bindings(new_bindings),
                evidence=self._unique_evidence(new_evidence),
                node_count=state.node_count + 1,
                edge_count=state.edge_count + 1,
                read_count=state.read_count + 1,
                score=state.score + 1,
        )
        next_state = self._with_query_status(
            next_state, edge, marginal_gain=1)
        return ExpansionOutcome(next_state, edge, 1)

    def _expand_dialogue(
            self,
            state: QueryState,
            remaining: tuple[FrontierEntry, ...],
            edge: FrontierEntry,
            origin: tuple[int, ...],
            ) -> ExpansionOutcome:
        """展开训练后 ResponseAct/realization frame 到 Core claim 的桥。"""
        memory_hypothesis = next((
            item for item in self.owner._discourse_hypotheses
            if item.hypothesis_key == origin), None)
        if memory_hypothesis is not None:
            depth = edge.depth + 1
            binding = BindingEntry(
                role_key=memory_hypothesis.hypothesis_kind,
                filler_key=memory_hypothesis.candidate_key,
                space=_DIALOGUE,
                scope_key=memory_hypothesis.scope_key,
            )
            response_context = next((item for item in self.owner._response_contexts
                                     if item.hypothesis.stable_key() == origin), None)
            context_bindings = () if response_context is None else response_context.query_bindings()
            hypothesis_evidence = tuple(
                item for item in self.owner._memory_evidence
                if (item.hypothesis_key == memory_hypothesis.hypothesis_key
                    and item.evidence_key in memory_hypothesis.evidence_keys)
            )
            if not hypothesis_evidence:
                raise RuntimeError("Dialogue Memory Hypothesis 缺少关联 Evidence")
            evidence = list(EvidenceEntry(
                item.source_ref, item.hypothesis_key,
                space=_DIALOGUE, polarity=item.stance, trust=1,
                evidence_key=item.evidence_key, scope_key=item.scope_key,
                payload_key=item.payload_key,
                supersedes_key=item.supersedes_key,
            ) for item in hypothesis_evidence)
            next_edges: list[FrontierEntry] = []
            topology_hop_count = 0
            # A category-7 topic hypothesis may be backed by the physical
            # W-08 topology restored at startup.  Expand its proposition
            # relations in this same Dialogue frontier; the topology only
            # contributes evidence/ordering and never becomes a fallback
            # answer or a source-surface replay.
            topic = next((
                candidate for candidate in self.owner._discourse_topic_candidates
                if candidate.candidate_key == memory_hypothesis.candidate_key
            ), None)
            if topic is not None:
                if (topic.proposition_key
                        in self.owner._discourse_topic_graph_identity_keys):
                    topic_binding = BindingEntry(
                        role_key=DISCOURSE_GRAPH_INPUT_ROLE,
                        filler_key=topic.proposition_key,
                        space=_DIALOGUE,
                        scope_key=memory_hypothesis.scope_key,
                    )
                    context_bindings = (*context_bindings, topic_binding)
                    topic_evidence_key = frame_key(
                        _DISCOURSE_TOPIC_GRAPH_EVIDENCE[0],
                        _DISCOURSE_TOPIC_GRAPH_EVIDENCE[1:],
                        topic.stable_key(),
                    )
                    evidence.append(EvidenceEntry(
                        memory_hypothesis.source_ref,
                        topic.proposition_key,
                        space=_DIALOGUE,
                        polarity=3,
                        trust=1,
                        evidence_key=topic_evidence_key,
                        scope_key=memory_hypothesis.scope_key,
                        payload_key=topic.stable_key(),
                    ))
                topic_frames = tuple(
                    frame for frame in self.owner.discourse_topic_topology.frames
                    if topic.proposition_key in frame.proposition_keys
                )
                if topic_frames:
                    for relation in self.owner.discourse_topic_topology.relations:
                        if relation.subject_key != topic.proposition_key:
                            continue
                        target = relation.object_key
                        route_key = relation.stable_key()
                        new_bindings = list(context_bindings)
                        new_bindings.append(BindingEntry(
                            role_key=relation.predicate_key,
                            filler_key=target,
                            space=_DIALOGUE,
                            scope_key=memory_hypothesis.scope_key,
                        ))
                        context_bindings = tuple(new_bindings)
                        self.hops.append(QueryHop(
                            _DIALOGUE,
                            relation.subject_key,
                            relation.predicate_key,
                            relation.subject_key,
                            target,
                            (),
                            depth,
                            self._source_hash(
                                relation.source_ref,
                                memory_hypothesis.source_ref[0]),
                        ))
                        topology_hop_count += 1
                        next_edge = FrontierEntry(
                            (_DIALOGUE, *target, *route_key), _DIALOGUE,
                            target_key=target,
                            root_key=edge.root_key,
                            scope_key=memory_hypothesis.scope_key,
                            source_ref=(relation.source_ref
                                       if relation.source_ref else memory_hypothesis.source_ref),
                            direction=1,
                            owner_weight=self.owner.space_weights[2],
                            depth=depth,
                            required_slot_gain=0,
                            evidence_support=1,
                            discourse_fit=4,
                            source_trust=1,
                        )
                        next_edges.append(next_edge)
            hop = QueryHop(
                _DIALOGUE,
                memory_hypothesis.hypothesis_key,
                memory_hypothesis.hypothesis_kind,
                (_DIALOGUE, *memory_hypothesis.hypothesis_key),
                (_MEMORY, *memory_hypothesis.candidate_key),
                (),
                depth,
                memory_hypothesis.source_ref[0],
            )
            self.hops.append(hop)
            visited = self._unique_visited([
                *state.visited,
                VisitedKey(_DIALOGUE, origin, direction=edge.direction),
            ])
            next_state = state.with_(
                depth=max(state.depth, depth),
                frontier=tuple(sorted(
                    (*remaining, *next_edges),
                    key=lambda entry: entry.priority_tuple(), reverse=True)),
                visited=visited,
                bindings=self._unique_bindings([*state.bindings, binding, *context_bindings]),
                evidence=self._unique_evidence([*state.evidence, *evidence]),
                node_count=state.node_count + 1 + topology_hop_count,
                edge_count=state.edge_count + 1 + topology_hop_count,
                read_count=state.read_count + 1 + topology_hop_count,
                score=state.score + 1 + topology_hop_count,
            )
            next_state = self._with_query_status(
                next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        if origin in self.owner._discourse_graph_input_keys:
            # Explicit W-08 graph identities are first-class Dialogue roots.
            # Expand only topology relations/frames already restored from the
            # same SQLite graph; no source text or surface lookup is involved.
            depth = edge.depth + 1
            bindings = [BindingEntry(
                role_key=DISCOURSE_GRAPH_INPUT_ROLE,
                filler_key=origin, space=_DIALOGUE,
                scope_key=edge.scope_key)]
            evidence = []
            relation_count = 0
            for relation in self.owner.discourse_topic_topology.relations:
                if relation.subject_key != origin and relation.object_key != origin:
                    continue
                target = (relation.object_key if relation.subject_key == origin
                          else relation.subject_key)
                source_ref = relation.source_ref or (21601, *origin)
                bindings.append(BindingEntry(
                    role_key=relation.predicate_key, filler_key=target,
                    space=_DIALOGUE, scope_key=edge.scope_key))
                evidence.append(EvidenceEntry(
                    source_ref, origin, space=_DIALOGUE, polarity=3, trust=1,
                    evidence_key=(21601, 2, *relation.stable_key()),
                    scope_key=edge.scope_key, payload_key=relation.stable_key()))
                relation_count += 1
            for frame in self.owner.discourse_topic_topology.frames:
                if origin not in frame.proposition_keys:
                    continue
                source_ref = frame.source_ref or (21601, *frame.frame_key)
                evidence.append(EvidenceEntry(
                    source_ref, origin, space=_DIALOGUE, polarity=3, trust=1,
                    evidence_key=(21601, 3, *frame.stable_key()),
                    scope_key=edge.scope_key, payload_key=frame.stable_key()))
            self.hops.append(QueryHop(
                _DIALOGUE, origin, (21601, 1), origin, origin, (), depth,
                self._source_hash(
                    next((item.source_ref for item in self.owner.discourse_topic_topology.relations
                          if item.subject_key == origin and item.source_ref), ()),
                    21601)))
            visited = self._unique_visited([
                *state.visited,
                VisitedKey(_DIALOGUE, origin, direction=edge.direction),
            ])
            next_state = state.with_(
                depth=max(state.depth, depth), frontier=remaining,
                visited=visited,
                bindings=self._unique_bindings([*state.bindings, *bindings]),
                evidence=self._unique_evidence([*state.evidence, *evidence]),
                node_count=state.node_count + 1 + relation_count,
                edge_count=state.edge_count + 1 + relation_count,
                read_count=state.read_count + 1 + relation_count,
                score=state.score + 1 + relation_count,
            )
            next_state = self._with_query_status(
                next_state, edge, marginal_gain=1 + relation_count)
            return ExpansionOutcome(next_state, edge, 1 + relation_count)
        if origin in self.owner._generic_graph_input_keys:
            # Every accepted generic graph identity is consumed by Dialogue in
            # the same QueryState.  This marker is structural evidence only;
            # it cannot become a source-surface answer or a fallback route.
            depth = edge.depth + 1
            binding = BindingEntry(
                role_key=(21610, 4), filler_key=origin,
                space=_DIALOGUE, scope_key=edge.scope_key)
            evidence = EvidenceEntry(
                edge.source_ref, origin, space=_DIALOGUE, polarity=3, trust=1,
                evidence_key=(21610, 5, *origin), scope_key=edge.scope_key,
                payload_key=(21610, 6, *origin))
            self.hops.append(QueryHop(
                _DIALOGUE, origin, (21610, 4), origin, origin, (), depth,
                edge.source_ref[1] if len(edge.source_ref) > 1 else 0))
            visited = self._unique_visited([
                *state.visited,
                VisitedKey(_DIALOGUE, origin, direction=edge.direction),
            ])
            next_state = state.with_(
                depth=max(state.depth, depth), frontier=remaining,
                visited=visited,
                bindings=self._unique_bindings([*state.bindings, binding]),
                evidence=self._unique_evidence([*state.evidence, evidence]),
                node_count=state.node_count + 1,
                edge_count=state.edge_count + 1,
                read_count=state.read_count + 1,
                score=state.score + 1,
            )
            next_state = self._with_query_status(
                next_state, edge, marginal_gain=1)
            return ExpansionOutcome(next_state, edge, 1)
        fact = self.owner.core_fact_index.get(origin)
        frame = self.owner.dialogue_frame_index.get(origin)
        if fact is None or frame is None:
            visited = self._unique_visited([
                *state.visited,
                VisitedKey(_DIALOGUE, origin, direction=edge.direction),
            ])
            next_state = self._with_query_status(
                state.with_(frontier=remaining, visited=visited),
                edge,
                marginal_gain=0,
            )
            return ExpansionOutcome(next_state, edge, 0)
        from pure_integer_ai.experiments.generation_organization import (
            answer_response_act,
        )
        depth = edge.depth + 1
        if (origin == state.best_candidate_key and not self.input_order_conflict
                and self.owner._margin_closed(state.best_score, state.second_score)):
            self.owner._prepare_generation_plans((origin,), state)
        generated = self.owner.generation_by_proposition.get(origin)
        if generated is not None:
            frame = self.owner.dialogue_frame_index[generated.frame_proposition.stable_key()]
        if origin in self.owner.kdconv_fact_index:
            source_scope = (fact.source_hash,)
            source_hash = fact.source_hash
        else:
            source = self.owner.core_runtime.generation_input(frame.proposition)
            source_scope = source.hypothesis.scope.stable_key()
            source_hash = frame.source_hash
        binding = BindingEntry(
            role_key=answer_response_act().stable_key(),
            filler_key=origin,
            space=_DIALOGUE,
            scope_key=source_scope,
        )
        if origin in self.owner.kdconv_fact_index:
            evidence = EvidenceEntry(
                (source_hash,), origin, space=_DIALOGUE, polarity=1, trust=1,
                evidence_key=(91630, 5, *origin), scope_key=source_scope,
                payload_key=(91630, 6, *origin))
        else:
            evidence = dialogue_query_evidence(fact.proposition, frame, source, generated)
        hop = QueryHop(
            _DIALOGUE,
            origin,
            fact.predicate.stable_key(),
            (_DIALOGUE, *origin),
            (_CORE, *origin),
            (),
            depth,
            frame.source_hash,
        )
        self.hops.append(hop)
        visited = self._unique_visited([
            *state.visited,
            VisitedKey(_DIALOGUE, origin, direction=edge.direction),
        ])
        next_state = state.with_(
            depth=max(state.depth, depth),
            frontier=remaining,
            visited=visited,
            bindings=self._unique_bindings([*state.bindings, binding]),
            evidence=self._unique_evidence([*state.evidence, evidence]),
            node_count=state.node_count + 1,
            edge_count=state.edge_count + 1,
            read_count=state.read_count + 1,
            score=state.score + 1,
        )
        next_state = self._with_query_status(
            next_state, edge, marginal_gain=1)
        return ExpansionOutcome(next_state, edge, 1)


__all__ = [
    "ConceptAnchor",
    "MemoryEvidenceAnchor",
    "MemoryHypothesisAnchor",
    "MemoryObservationAnchor",
    "QueryHop",
    "TrainedGraphQueryBridge",
]
