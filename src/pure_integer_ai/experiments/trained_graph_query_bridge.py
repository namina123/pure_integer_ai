"""三图库并行纵切：把训练后结构接到同一个 QueryState。

Core active relation、Interaction Memory Observation 与训练后 realization frame
分别产生独立 root；所有非空 root 同时进入一个 frontier，owner 权重只决定
扩展次序。Memory 没有 Hypothesis 时只贡献有来源的 Observation，不再把历史
原文或码点近邻伪装成语义边；Dialogue root 只能来自训练后的 Companion 图
结构，生产查询不加载 successor occurrence 或 exact-surface evidence。
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import wraps
import os
from pathlib import Path

from pure_integer_ai.experiments.query_relation_arbitration import (
    core_relation_endpoint_identity,
    input_relation_competition,
    select_equivalent_slot_question_bindings,
    _surface_relation_topology_key,
    _trained_symmetric_relation_role_pair,
    _surface_symmetric_relation_topology_key,
    _surface_open_relation_topology_key,
    _surface_equivalent_relation_tie,
)
from pure_integer_ai.experiments.query_reference_source_consensus import (
    REFERENCE_SOURCE_CONSENSUS_VERSION, grounded_reference_source, reference_source_consensus,
)

from pure_integer_ai.experiments.relation_runtime_phase import (
    observed_relation_phase, relation_phase,
)

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.trained_core_relation_composition import (
    RelationCompositionBudget, pending_core_compositions,
    pending_composition_query_evidence, validate_relation_composition_provenance,
)
from pure_integer_ai.experiments.trained_relation_composition_graph import (
    read_relation_composition_graph,
)
from pure_integer_ai.experiments.trained_question_open_frames import (
    bound_question_open_frames, question_observed_role_contract,
)
from pure_integer_ai.experiments.verified_composition_query import (
    composition_output_is_current, consume_verified_composition,
)
from pure_integer_ai.cognition.understanding.query_relation_composition import (
    match_pending_relation_questions,
)
from pure_integer_ai.cognition.shared.generation_observed_surface import (
    delivered_graph_surface_role,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD
from pure_integer_ai.storage.source_record import SourceRecordRepository


_QUERY_SOURCE_PROVENANCE_HASHER = Hasher(
    "pure_integer_ai.query.source_provenance.v1")


def _memory_event_read_scope(operation):
    """在一次完整消费内复用经过物理修订校验的 Memory 事件解码。"""
    @wraps(operation)
    def scoped(self, *args, **kwargs):
        """保持原返回值与异常，并让 Memory append API 核验缓存修订。"""
        memory = getattr(self, "memory", None)
        if memory is None:
            return operation(self, *args, **kwargs)
        with memory.intake.event_log.read_scope():
            return operation(self, *args, **kwargs)
    return scoped

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
    TERMINATION_BUDGET_EXHAUSTED,
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
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    SPAN_SOURCE,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
    STRUCTURE_ORDER_CONFLICT,
    InputSpan,
    InputSpanRelation,
    InputRelationQuestionBinding,
    InputSemanticCandidate,
    TrainedInputDiscourseOperator,
    InputRelationCandidate,
    QueryInputStructure,
    TrainedInputStructureProjector,
    graph_query_input_structure,
    relation_candidate_semantic_groups,
    relation_candidate_semantic_key,
    scope_local_conjunction_candidate_keys,
    scope_local_conjunction_members,
    scope_local_fact_query_candidate_keys,
    scope_local_general_operation_candidate_keys,
)
from pure_integer_ai.cognition.understanding.query_input_provenance import (
    aggregate_input_provenance_duplicates as _aggregate_input_provenance_duplicates,
)
from pure_integer_ai.cognition.understanding.query_provenance_identity import (
    semantic_candidate_identity_key,
)
from pure_integer_ai.cognition.understanding.query_reference_context import (
    extended_reference_context_required, reference_marker_sources,
)
from pure_integer_ai.cognition.understanding.query_question_memory_view import (
    merge_reference_question_views, reference_question_open_candidate,
    retained_reference_question_gap_witness,
)
from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
    available_slot_question_frames, composed_slot_open_role_frames,
    restore_slot_question_frame,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
    _matching_open_relation_candidates,
    match_relation_question_frames,
    match_relation_slot_question_frames,
    relation_question_syntax_predicates,
)
from pure_integer_ai.cognition.understanding.query_scoped_relation_questions import (
    conjunction_has_complete_literal_scopes,
    match_scoped_relation_question_frames,
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
    MEMORY_CATEGORY_DISCOURSE,
    MEMORY_CATEGORY_OPEN_RELATION,
    MEMORY_CATEGORY_RESPONSE,
    MEMORY_CATEGORY_TIME,
    frame_key,
    memory_candidate_category,
    memory_competition_identity,
    memory_graph_input_object_key,
    memory_relation_candidate_key,
)
from pure_integer_ai.cognition.understanding.query_open_role_state import open_role_query_nodes
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate,
    OpenRoleFrame,
    match_open_roles,
)
from pure_integer_ai.cognition.understanding.query_relation_coverage import (
    has_uncovered_closed_relation_modifier,
)
from pure_integer_ai.cognition.understanding.query_assertion_scope import attach_assertion_scope_frames, SCOPE_QUERY_PROOF
from pure_integer_ai.experiments.trained_assertion_scope_graph import read_assertion_scope_graph
from pure_integer_ai.experiments.trained_assertion_scope_runtime import (
    assertion_scope_feature, assertion_scope_query_records, assertion_scope_response_allowed,
)
from pure_integer_ai.cognition.understanding.query_open_role_generation import (
    OpenRoleGenerationContext, prepare_open_role_generation,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    MemoryActiveStructures,
    MemoryObservationStructure,
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
from pure_integer_ai.experiments.input_dialogue_act_graph import (
    read_input_dialogue_act_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_graph import (
    read_input_discourse_operator_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_operation_graph import (
    MODE_FACT_QUERY_LEFT,
    MODE_FACT_QUERY_RIGHT,
    MODE_OPERATION_GENERAL,
    TrainedInputDiscourseOperationContract,
    read_input_discourse_operation_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_conjunction_graph import (
    read_input_discourse_conjunction_graph,
)
from pure_integer_ai.experiments.trained_revision_cue_graph import (
    read_revision_cue_graph,
)
from pure_integer_ai.experiments.natural_revision_runtime import (
    NaturalRevisionCommit,
    NaturalRevisionInputBinding,
    NaturalRevisionRuntime,
    NaturalRevisionTargetBinding,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    MODE_RELATION_PAIR_QUERY,
    read_relation_question_graph,
)
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_input_clarification import unresolved_role_context_allowed
from pure_integer_ai.experiments.input_projection_conflict import (
    conflicting_input_filler_groups as _conflicting_input_filler_groups,
    retained_projection_conflict_allowed,
)
from pure_integer_ai.experiments.symmetric_input_order import (
    symmetric_input_order_witnesses, unresolved_input_conflict_bindings,
)
from pure_integer_ai.experiments.trained_open_response_runtime import (
    GenericResponseConnectorQueryNode,
    _discourse_memory_refs,
    _reference_generation_provenance,
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
    resolve_factual_claim_input_hypotheses,
)
from pure_integer_ai.experiments.trained_response_context import (
    DeliveredGraphSlotProjection,
    DeliveredGenericResponseContext,
    DeliveredFactualResponseContext,
    delivered_graph_slot_projections,
    response_context_input_observations,
    restore_response_contexts,
)
from pure_integer_ai.experiments.trained_response_occurrence import (
    InputRoleOccurrence, ReferenceCandidateSet, materialize_input_roles,
)
from pure_integer_ai.experiments.memory_ohe_expansion_coverage import memory_ohe_expansion_coverage
from pure_integer_ai.experiments.trained_reference_resolution import (
    REFERENCE_CONFLICT,
    REFERENCE_RESOLVED,
    CrossTurnReferenceResolution,
    TrainedCrossTurnReferenceRuntime,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_REFERS,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
)
from pure_integer_ai.experiments.relation_capability_route import (
    RelationCapabilityRouteRegistry,
)
from pure_integer_ai.experiments.trained_typed_relation_bridge import (
    EVENT_TIME_RELATION_KINDS,
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


def _validate_input_discourse_graph_provenance(
        operator_metadata: dict[str, object] | None,
        operation_metadata: dict[str, object] | None,
        ) -> None:
    """Require operator and operation sidecars to share source/license identity."""
    if (operator_metadata is not None and operation_metadata is not None
            and (operator_metadata["source_identity"]
                 != operation_metadata["source_identity"]
                 or operator_metadata["license_identity"]
                 != operation_metadata["license_identity"])):
        raise ValueError(
            "operator graph and operation contract provenance must match")


def _generic_response_adoption_allowed(
        response_plan, generic_reference_plan, *,
        generic_structural_preferred: bool,
        resolved_reference_resolutions: tuple[object, ...]) -> bool:
    """Keep Companion arbitration from discarding a selected plan's provenance.

    A generic plan may be adopted when it is the only generated plan.  Once a
    factual/open plan exists, a graph-backed generic candidate can replace it
    only when this turn has no resolved cross-turn reference.  A resolved
    REFERS winner must remain attached to the plan that is actually delivered.
    """
    if generic_reference_plan is None:
        return False
    if response_plan is None:
        return True
    return bool(generic_structural_preferred
                and not resolved_reference_resolutions)


def _merge_memory_active_structures(
        *structures: MemoryActiveStructures) -> MemoryActiveStructures:
    """Merge independently rooted O/H/E closures without dropping conflicts."""
    observations = {}
    hypotheses = {}
    evidence = {}
    routes = set()
    for structure in structures:
        for rows, target, key_name in (
                (structure.observations, observations, "observation_key"),
                (structure.hypotheses, hypotheses, "hypothesis_key"),
                (structure.evidence, evidence, "evidence_key")):
            for row in rows:
                key = getattr(row, key_name)
                prior = target.get(key)
                if prior is not None and prior != row:
                    if (key_name == "observation_key"
                            and isinstance(prior, MemoryObservationStructure)
                            and isinstance(row, MemoryObservationStructure)
                            and prior.source_ref == row.source_ref
                            and prior.scope_key == row.scope_key
                            and prior.source_hash == row.source_hash
                            and prior.turn_seq == row.turn_seq
                            and prior.speaker_kind == row.speaker_kind
                            and prior.relation_refs == row.relation_refs):
                        # Candidate-filtered and zero-depth roots may expose
                        # different H subsets for the same Observation.  The
                        # root is authoritative for the union; path depth is
                        # metadata.  Any source/scope/relation drift still
                        # fails closed above.
                        target[key] = replace(
                            prior,
                            hypothesis_keys=tuple(sorted(set(
                                prior.hypothesis_keys
                                + row.hypothesis_keys))),
                            context_depth=min(
                                prior.context_depth, row.context_depth))
                        continue
                    raise RuntimeError("overlapping Memory closures disagree")
                target[key] = row
        routes.update(structure.match_routes)
    return MemoryActiveStructures(
        tuple(sorted(observations.values(),
                     key=lambda item: (item.turn_seq, item.observation_key))),
        tuple(sorted(hypotheses.values(), key=lambda item: item.hypothesis_key)),
        tuple(sorted(evidence.values(), key=lambda item: item.evidence_key)),
        tuple(sorted(routes)),
    )


def _root_response_context_inputs(memory, structures, contexts):
    """补齐实际交付的输入来源O及显式H/E依赖，不重新授权无关兄弟。"""
    required = response_context_input_observations(contexts)
    if not required:
        return structures
    inputs = memory.active_structures(
        candidate_keys=(), observation_refs=required, context_depth_limit=0)
    found = {item.observation_key for item in inputs.observations}
    if not {item.stable_key() for item in required} <= found:
        raise RuntimeError("delivered response input Observation is not recoverable")
    from pure_integer_ai.experiments.memory_provenance_inputs import retain_context_input_dependencies
    inputs = retain_context_input_dependencies(inputs, contexts)
    return _merge_memory_active_structures(structures, inputs)


def _validate_response_context_memory_closure(structures, contexts) -> None:
    """Require every adopted delivered-context O/H ref in this O/H/E closure."""
    available = (
        {item.observation_key for item in structures.observations}
        | {item.hypothesis_key for item in structures.hypotheses}
        | {item.evidence_key for item in structures.evidence}
    )
    if any(ref.stable_key() not in available
           for context in contexts for ref in context.memory_refs()):
        raise RuntimeError("response context Memory provenance is outside its O/H/E closure")


def _observed_input_role_keys(
        input_structure: QueryInputStructure,
        ) -> frozenset[tuple[int, ...]]:
    """Return trained role identities already filled by this input.

    Open-role frames contain every role, including roles whose integer span is
    already an observed filler.  Only the still-unfilled roles may become
    cross-turn REFERS candidates; the observed roles remain ordinary Memory
    evidence and must not create a second reference resolution.
    """
    observed = set()
    for item in input_structure.semantic_candidates:
        if item.projection_kind != PROJECTION_FILLER:
            continue
        observed.add(item.role_key)
    return frozenset(observed)
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
    read_event_time_generation_overlay_edges,
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
_PARTIAL_PROPERTY_ANCHOR_KIND = 13
_EVENT_TIME_GRAPH_EDGE = (91560, 1)
_EVENT_TIME_GRAPH_EVIDENCE = (91560, 2)
_INPUT_SEMANTIC_PROJECTION_EVIDENCE = (91569, 1)
_INPUT_RELATION_PROJECTION_EVIDENCE = (91569, 2)
_DISCOURSE_TOPIC_GRAPH_EVIDENCE = (91569, 3)
_MEMORY_DELIVERED_GRAPH_SURFACE_EVIDENCE = (91572, 2)
_INPUT_DISCOURSE_CONJUNCTION_ROLE = (91599, 4)
_INPUT_DISCOURSE_CONJUNCTION_EVIDENCE = (91599, 5)
_INPUT_FACT_QUERY_ROLE = (91672, 2)
_INPUT_FACT_QUERY_EVIDENCE = (91672, 3)
_INPUT_OPERATION_ROLE = (91672, 4)
_INPUT_OPERATION_EVIDENCE = (91672, 5)
_REFERENCE_TARGET_ADOPTION_ROLE = (91599, 6)
_REFERENCE_TARGET_ADOPTION_EVIDENCE = (91599, 7)
_REFERENCE_TARGET_ADOPTION_VERSION = 91599


def _closed_scope_conjunction_components(
        bindings: tuple[tuple[tuple[int, ...], tuple[int, ...],
                                tuple[tuple[int, ...], ...]], ...],
        generation_ready: set[tuple[int, ...]],
        ) -> tuple[tuple[int, ...], ...]:
    """Return one verified connected claim component, if one exists.

    Each trained conjunction is binary.  A chain of adjacent conjunctions can
    authorize a larger composition only when every edge is closed and every
    member was generated in this QueryState.  Disconnected edges, duplicate
    members, or an edge with an unready proposition remain fail-closed.
    """
    edges = tuple(sorted({
        tuple(sorted(set(members)))
        for _composition, _relation, members in bindings
        if len(set(members)) == 2 and set(members) <= generation_ready
    }))
    if not edges:
        return ()
    parent = {member: member for edge in edges for member in edge}

    def find(item):
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    components = {}
    for member in parent:
        components.setdefault(find(member), set()).add(member)
    if len(components) != 1:
        return ()
    component = tuple(sorted(next(iter(components.values()))))
    return (component,) if len(component) >= 2 else ()


def _composition_query_view(
        structure: QueryInputStructure,
        scope_bindings: tuple[tuple[tuple[int, ...], tuple[int, ...],
                                    tuple[tuple[int, ...], ...]], ...],
        kdconv_propositions: set[tuple[int, ...]],
        selected_relation_keys: set[tuple[int, ...]] | None = None,
        ) -> tuple[QueryInputStructure, set[tuple[int, ...]], set[tuple[int, ...]]]:
    """Keep audited alternatives while activating only closed scope members."""
    if not (structure.discourse_compositions or structure.discourse_gaps):
        return structure, set(kdconv_propositions), {
            item.stable_key() for item in structure.relation_candidates}
    # An unresolved gap between independently closed scopes is not an
    # operation.  Keep both exact relation projections in the same
    # QueryState so Core/Memory can consume them, while the caller retains
    # the unresolved-gap termination gate and cannot merge the scopes into a
    # factual multi-claim answer.  A real composition still follows the
    # audited scope-member selection below.
    if structure.discourse_gaps and not structure.discourse_compositions:
        return structure, set(kdconv_propositions), {
            item.stable_key() for item in structure.relation_candidates}
    members = {
        member
        for _composition, _relation, claim_keys in scope_bindings
        for member in claim_keys
    }
    referenced_relation_keys = {
        relation_key
        for composition in structure.discourse_compositions
        for group in composition.operand_relation_keys
        for relation_key in group
    }
    # An empty selection means that no local scope candidate was uniquely
    # identified. Preserve all referenced alternatives for audit/UNKNOWN
    # handling; clearing them would make the composition appear unparsed.
    active_relation_keys = (
        referenced_relation_keys
        if selected_relation_keys is None or not selected_relation_keys
        else set(selected_relation_keys))
    active_members = set(members)
    if not active_members:
        active_members = {
            item.proposition_key for item in structure.relation_candidates
            if item.stable_key() in active_relation_keys
        }
    active_span_refs = {
        span_ref
        for item in structure.relation_candidates
        if item.stable_key() in active_relation_keys
        for span_ref in item.span_refs
    }
    return (
        replace(
            structure,
            semantic_candidates=tuple(
                item for item in structure.semantic_candidates
                if (item.proposition_key in active_members
                    and item.span_ref in active_span_refs)),
        ),
        set(kdconv_propositions).intersection(active_members),
        active_relation_keys,
    )


def _selected_relation_view(
        structure: QueryInputStructure,
        relation_key: tuple[int, ...],
        ) -> tuple[QueryInputStructure, tuple[int, ...]]:
    """Collapse one authorized operation to its selected relation scope.

    The operation marker remains in the integer trace and operation contract,
    while its composition is consumed as a scope contract.  Every semantic
    span and carrier retained below is still named by the selected relation
    candidate; no surface-derived fallback is introduced.
    """
    candidates = {item.stable_key(): item for item in structure.relation_candidates}
    relation = candidates.get(relation_key)
    if relation is None or relation.state != STRUCTURE_CLOSED:
        raise ValueError("operation selected a non-closed relation")
    proposition = relation.proposition_key
    spans = set(relation.span_refs)
    semantic = tuple(sorted((
        item for item in structure.semantic_candidates
        if item.proposition_key == proposition and item.span_ref in spans),
        key=lambda item: item.stable_key()))
    carriers = tuple(sorted((
        item for item in structure.carrier_candidates
        if item.structure_key == relation.structure_key),
        key=lambda item: item.stable_key()))
    return replace(
        structure,
        carrier_candidates=carriers,
        semantic_candidates=semantic,
        relation_candidates=(relation,),
        open_relations=(),
        discourse_compositions=(),
        discourse_gaps=(),
    ), relation_key


def _has_closed_bound_relation_question(
        structure: QueryInputStructure,
        ) -> bool:
    """Return whether one exact question binding names a closed relation.

    Binary relation questions may intentionally omit a predicate span: the
    trained operation/frame supplies that predicate contract.  In that shape
    ``predicate_coverage`` is zero even though all required role spans are
    present.  The query root gate must preserve the selected relation's Core
    fillers while still requiring an exact binding and full structural closure.
    """
    relation_keys = {
        binding.relation_key
        for binding in structure.relation_question_bindings
    }
    return bool(relation_keys) and any(
        item.stable_key() in relation_keys
        and item.state == STRUCTURE_CLOSED
        and item.required_count >= 2
        and item.required_count == item.support_count
        and item.required_open == 0
        and item.predicate_coverage in {0, 1}
        for item in structure.relation_candidates)


def _partial_property_fact_matches(
        structure: QueryInputStructure,
        facts: tuple[ActiveRelationSurface, ...],
        ) -> tuple[tuple[InputRelationCandidate, ActiveRelationSurface], ...]:
    """Find one exact partially observed n-ary fact without guessing slots.

    A natural property question may expose only subject, attribute and
    predicate while the trained property relation also carries value,
    polarity, modality and intensity roles.  This helper is deliberately
    stricter than candidate ranking: one open relation must point to one
    active fact with the same proposition/predicate/source, every observed
    filler must match the fact role and filler identity, and the observed
    topology must be unique.  Omitted roles remain available only through the
    complete Core fact during query expansion.
    """
    if (not structure.relation_candidates
            or structure.discourse_compositions
            or structure.discourse_gaps
            or structure.relation_question_bindings
            or structure.pending_relation_composition_bindings):
        return ()
    facts_by_proposition = {
        fact.proposition.stable_key(): fact for fact in facts
    }
    span_refs = {item.ref_key for item in structure.spans}
    matches = []
    for relation in structure.relation_candidates:
        if (relation.state != STRUCTURE_OPEN
                or relation.conflict_count != 0
                or relation.predicate_coverage != 1
                or relation.required_open <= 0
                or relation.role_coverage < 2
                or relation.required_count <= 3):
            continue
        fact = facts_by_proposition.get(relation.proposition_key)
        if (fact is None
                or relation.source_ref != (fact.source_hash,)
                or relation.predicate_key != fact.predicate.stable_key()
                or relation.required_count != len(fact.bindings) + 1
                or len(fact.bindings) <= 2):
            continue
        fact_by_role = {
            item.role.stable_key(): item.filler.stable_key()
            for item in fact.bindings
        }
        if len(fact_by_role) != len(fact.bindings):
            continue
        observed_fillers = tuple(
            item for item in structure.semantic_candidates
            if (item.projection_kind == PROJECTION_FILLER
                and item.proposition_key == relation.proposition_key
                and item.predicate_key == relation.predicate_key
                and item.source_ref == relation.source_ref
                and item.span_ref in relation.span_refs
                and item.span_ref in span_refs)
        )
        predicate_members = tuple(
            item for item in structure.semantic_candidates
            if (item.projection_kind == PROJECTION_PREDICATE
                and item.proposition_key == relation.proposition_key
                and item.predicate_key == relation.predicate_key
                and item.source_ref == relation.source_ref
                and item.span_ref in relation.span_refs
                and item.span_ref in span_refs)
        )
        if (len(predicate_members) != 1
                or len(observed_fillers) != relation.role_coverage
                or len({item.role_key for item in observed_fillers})
                != len(observed_fillers)
                or len({item.span_ref for item in observed_fillers})
                != len(observed_fillers)):
            continue
        if set(item.span_ref for item in (*observed_fillers, *predicate_members)) != set(
                relation.span_refs):
            continue
        if any(
                item.role_key not in fact_by_role
                or fact_by_role[item.role_key] != item.candidate_key
                for item in observed_fillers):
            continue
        matches.append((relation, fact))
    identities = {
        (relation.proposition_key, relation.predicate_key,
         relation.source_ref,
         tuple(sorted((item.role.stable_key(), item.filler.stable_key())
                      for item in fact.bindings)))
        for relation, fact in matches
    }
    if len(identities) != 1:
        return ()
    return tuple(sorted(matches, key=lambda item: (
        item[0].proposition_key, item[0].structure_key,
        item[0].source_ref)))


def _has_unbound_relation_question_candidate(
        structure: QueryInputStructure,
        question_open_frames: tuple[OpenRoleFrame, ...],
        ) -> bool:
    """Detect a trained relation question that lacks its exact binding.

    Open predicate-bearing candidates retain the existing fail-closed rule.
    A closed substring candidate is also blocked when the current input ends
    with a trained question mark but no complete question frame bound. This
    never selects a proposition or lets a PURE_ALIAS route answer an
    unlicensed REFERS question by inverse fallback.
    """
    if (not question_open_frames
            or structure.relation_question_bindings
            or structure.pending_relation_composition_bindings
            or structure.discourse_compositions
            or structure.discourse_gaps):
        return False
    frame_contracts = {
        (item.proposition, item.predicate, item.source_ref, len(item.roles))
        for item in question_open_frames
    }
    values = getattr(structure, "token_values", ())
    question_suffix_contracts = {
        (item.proposition, item.predicate, item.source_ref, len(item.roles))
        for item in question_open_frames
        if (item.gaps[-1]
            and item.gaps[-1][-1] in {0x003F, 0xFF1F}
            and len(values) >= len(item.gaps[-1])
            and values[-len(item.gaps[-1]):] == item.gaps[-1])
    }
    for item in structure.relation_candidates:
        contract = (item.proposition_key, item.predicate_key,
                    item.source_ref, item.required_count - 1)
        if contract not in frame_contracts:
            continue
        if (item.state == STRUCTURE_OPEN
                and item.required_open > 0
                and item.predicate_coverage == 1):
            return True
        # The surface projector can close a relation from exact member
        # substrings while leaving an untrained modifier or polarity cue in
        # the gaps. A shared fact is not enough to answer a question: when a
        # trained question-ending is present, require the complete question
        # frame binding above before Core expansion.
        if (contract in question_suffix_contracts
                and item.state == STRUCTURE_CLOSED
                and item.required_open == 0
                and item.required_count == item.support_count
                and item.role_coverage == contract[3]
                and item.predicate_coverage == 1
                and item.conflict_count == 0):
            return True
    return False


def _has_unbound_event_time_question_candidate(
        structure: QueryInputStructure,
        event_time_predicate_keys: frozenset[tuple[int, ...]],
        ) -> bool:
    """Detect an open Event/Time relation that lacks a trained binding.

    Event/Time pair questions can arrive through the typed relation topology
    without a relation-question frame carrying the predicate span.  In that
    case the ordinary relation-question gate cannot see the missing slot:
    the candidate remains open and ``predicate_coverage`` may be zero.  The
    Event/Time graph still provides a strict predicate identity and an
    open-role parse, so stop at the same clarification boundary before any
    Core/Memory/Dialogue expansion.  This helper never selects a proposition
    or closes a relation.
    """
    if (not event_time_predicate_keys
            or structure.relation_question_bindings
            or structure.pending_relation_composition_bindings
            or structure.discourse_compositions
            or structure.discourse_gaps
            or not structure.open_relations):
        return False
    open_relation_contracts = {
        (item.frame.proposition, item.frame.predicate, item.frame.source_ref)
        for item in structure.open_relations
    }
    return any(
        item.state == STRUCTURE_OPEN
        and item.required_open > 0
        and item.predicate_key in event_time_predicate_keys
        and (item.proposition_key, item.predicate_key, item.source_ref)
        in open_relation_contracts
        for item in structure.relation_candidates)


def _fact_query_view(
        structure: QueryInputStructure,
        match: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        ) -> tuple[QueryInputStructure, tuple[int, ...]]:
    """Collapse one authorized unary query to its selected fact relation."""
    _composition_key, _operation_key, relation_key, _mode_key = match
    return _selected_relation_view(structure, relation_key)


def _general_operation_view(
        structure: QueryInputStructure,
        match: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        ) -> tuple[QueryInputStructure, tuple[int, ...]]:
    """Collapse one authorized general operation to its closed right scope."""
    _composition_key, _operation_key, relation_key = match
    return _selected_relation_view(structure, relation_key)


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
        # Query states reuse immutable entries across adjacent frontier
        # snapshots.  Keep only a bounded identity cache: it avoids rebuilding
        # long flattened stable keys while preventing a budget-exhausted trace
        # from retaining every historical state object.
        self._item_refs: dict[int, tuple[object, int]] = {}
        self._item_ref_cache_limit = 8192
        # Cache only stable integer keys.  The previous implementation kept a
        # strong reference to every immutable frontier/binding/evidence object
        # ever seen so CPython ``id`` reuse could not occur.  A query creates
        # thousands of short-lived states; retaining those objects made a
        # single ordinary turn grow to gigabytes.  The output is keyed by
        # stable integers, so deriving the key on each lookup preserves the
        # trace contract without retaining query-state history.

    def _legacy_key(self, item) -> tuple[int, ...]:
        return item.stable_key()

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
            self._item_refs.pop(identity, None)
            self._item_refs[identity] = cached
            return cached[1]
        ref = self._ref(item.stable_key())
        self._item_refs[identity] = (item, ref)
        while len(self._item_refs) > self._item_ref_cache_limit:
            self._item_refs.pop(next(iter(self._item_refs)))
        return ref

    def record(self, state: QueryState, reason: int) -> None:
        if not self.reference_mode:
            self.steps.append({
                "reason": reason,
                "depth": state.depth,
                "counters": [state.depth, state.node_count, state.edge_count, state.read_count],
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
            "counters": [state.depth, state.node_count, state.edge_count, state.read_count],
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
            input_discourse_operators: tuple[
                TrainedInputDiscourseOperator, ...] = (),
            input_discourse_graph_database: str | Path | None = None,
            input_dialogue_act_graph_database: str | Path | None = None,
            input_discourse_operation_graph_database: str | Path | None = None,
            input_discourse_conjunction_graph_database: str | Path | None = None,
            relation_question_graph_database: str | Path | None = None,
            assertion_scope_graph_database: str | Path | None = None,
            relation_composition_graph_database: str | Path | None = None,
            relation_composition_budget: RelationCompositionBudget = RelationCompositionBudget(),
            relation_composition_verification: bool = True,
            operation_program_graph_database: str | Path | None = None,
            operation_request_graph_database: str | Path | None = None,
            operation_output_graph_database: str | Path | None = None,
            event_time_generation_binding_overlay: str | Path | None = None,
            revision_cue_graph_database: str | Path | None = None,
            surface_artifact_database: str | Path | None = None,
            # Successor occurrence evidence is diagnostic/training-only and
            # is rejected here so it cannot enter a production QueryState.
            successor_evidence: bool = False,
            ) -> None:
        self._diagnostic_stage_file = os.environ.get(
            "PURE_INTEGER_QUERY_STAGE_FILE", "")
        self._diagnostic_stage("bridge_init_enter")
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
        self.assertion_scope_frames = (() if assertion_scope_graph_database is None
            else read_assertion_scope_graph(assertion_scope_graph_database))
        self.input_dialogue_act_graph = (
            None if input_dialogue_act_graph_database is None
            else read_input_dialogue_act_graph(
                input_dialogue_act_graph_database))
        # 这些 root 虽是单次查询路由状态，但扩展器及直接构造测试可能在
        # 首次 query 前读取；桥生命周期内先保持中性空集合。
        self._slot_question_core_roots: frozenset[tuple[int, ...]] = frozenset()
        self._partial_property_core_roots: frozenset[tuple[int, ...]] = frozenset()
        if revision_cue_graph_database is None:
            self.revision_cue_records = ()
            self.revision_cue_graph_metadata = None
            self.natural_revision_runtime = None
        else:
            (self.revision_cue_records,
             self.revision_cue_graph_metadata) = read_revision_cue_graph(
                revision_cue_graph_database)
            self.natural_revision_runtime = NaturalRevisionRuntime(
                self.revision_cue_records, scope_frames=self.assertion_scope_frames)
        if (type(input_discourse_operators) is not tuple
                or any(not isinstance(item, TrainedInputDiscourseOperator)
                       for item in input_discourse_operators)):
            raise TypeError("input_discourse_operators must be trained integer records")
        if input_discourse_graph_database is not None and input_discourse_operators:
            raise ValueError(
                "production operator graph 与外部 operator 注入不能同时使用")
        self.input_discourse_graph_metadata = None
        if input_discourse_graph_database is None:
            self.input_discourse_operators = tuple(sorted(
                set(input_discourse_operators), key=lambda item: item.stable_key()))
        else:
            (self.input_discourse_operators,
             self.input_discourse_graph_metadata) = (
                read_input_discourse_operator_graph(
                    input_discourse_graph_database))
        if input_discourse_operation_graph_database is None:
            self.input_discourse_operation_contracts = ()
            self.input_discourse_operation_graph_metadata = None
        else:
            (self.input_discourse_operation_contracts,
             self.input_discourse_operation_graph_metadata) = (
                read_input_discourse_operation_graph(
                    input_discourse_operation_graph_database))
        _validate_input_discourse_graph_provenance(
            self.input_discourse_graph_metadata,
            self.input_discourse_operation_graph_metadata,
        )
        if relation_question_graph_database is None:
            self.relation_question_frames = ()
            self.relation_question_graph_metadata = None
        else:
            (self.relation_question_frames,
             self.relation_question_graph_metadata) = read_relation_question_graph(
                 relation_question_graph_database)
        for metadata in (self.input_discourse_graph_metadata,
                         self.input_discourse_operation_graph_metadata):
            if (metadata is not None
                    and self.relation_question_graph_metadata is not None
                    and (metadata["source_identity"]
                         != self.relation_question_graph_metadata["source_identity"]
                         or metadata["license_identity"]
                         != self.relation_question_graph_metadata["license_identity"])):
                raise ValueError("question and operation graphs have different provenance")
        for metadata in (self.input_discourse_graph_metadata,
                         self.input_discourse_operation_graph_metadata,
                         self.relation_question_graph_metadata):
            if (metadata is not None
                    and self.revision_cue_graph_metadata is not None
                    and (metadata["source_identity"]
                         != self.revision_cue_graph_metadata["source_identity"]
                         or metadata["license_identity"]
                         != self.revision_cue_graph_metadata["license_identity"])):
                raise ValueError("revision cue and mounted input graphs have different provenance")
        known_operator_keys = {item.operation_key
                               for item in self.input_discourse_operators}
        if any(item.operation_key not in known_operator_keys
               for item in self.input_discourse_operation_contracts):
            raise ValueError(
                "operation contract graph references an operator absent from the operator graph")
        operation_modes = {}
        for item in self.input_discourse_operation_contracts:
            prior = operation_modes.get(item.operation_key)
            if prior is not None and prior != item.mode_key:
                raise ValueError("operation contract has conflicting modes")
            operation_modes[item.operation_key] = item.mode_key
        for frame in self.relation_question_frames:
            prior = operation_modes.get(frame.operation_key)
            if prior is not None and prior != MODE_RELATION_PAIR_QUERY:
                raise ValueError("question operation conflicts with an existing mode")
            operation_modes[frame.operation_key] = MODE_RELATION_PAIR_QUERY
        from pure_integer_ai.experiments.trained_operation_program_graph import read_operation_program_graph
        self.operation_programs_required = operation_program_graph_database is not None
        self.operation_program_witnesses = (() if operation_program_graph_database is None
            else read_operation_program_graph(operation_program_graph_database))
        if operation_output_graph_database is not None and not self.operation_programs_required:
            raise ValueError("output obligations require the current semantic program graph")
        from pure_integer_ai.experiments.operation_output_course import read_operation_output_graph
        self.operation_output_witnesses = (None if operation_output_graph_database is None else
            read_operation_output_graph(operation_output_graph_database, self.operation_program_witnesses))
        from pure_integer_ai.experiments.operation_request_course import read_operation_request_graph
        from pure_integer_ai.experiments.operation_request_consumer import (
            validate_operation_input_permissions, validate_operation_request_frames,
        )
        self.operation_request_frames = (() if operation_request_graph_database is None
            else read_operation_request_graph(operation_request_graph_database))
        validate_operation_request_frames(self.operation_request_frames, self.operation_program_witnesses)
        for frame in self.operation_request_frames:
            if frame.operation_key in operation_modes and operation_modes[frame.operation_key] != MODE_OPERATION_GENERAL:
                raise ValueError("request frame has a competing pair-query operation interface")
            operation_modes[frame.operation_key] = MODE_OPERATION_GENERAL
        for witness in self.operation_program_witnesses:
            if witness.program.operation_key not in operation_modes:
                raise ValueError("semantic operation program has no trained input contract")
            validate_operation_input_permissions(witness, (
                (self.input_discourse_operators, self.input_discourse_graph_metadata),
                (self.input_discourse_operation_contracts, self.input_discourse_operation_graph_metadata),
                (self.relation_question_frames, self.relation_question_graph_metadata),
            ))
        self.input_discourse_operation_modes = dict(operation_modes)
        self.input_discourse_fact_query_modes = {
            operation: mode for operation, mode in operation_modes.items()
            if mode in {MODE_FACT_QUERY_LEFT, MODE_FACT_QUERY_RIGHT}
        }
        if input_discourse_conjunction_graph_database is None:
            self.input_discourse_conjunctions = ()
            self.input_discourse_conjunction_graph_metadata = None
        else:
            (self.input_discourse_conjunctions,
             self.input_discourse_conjunction_graph_metadata) = (
                read_input_discourse_conjunction_graph(
                    input_discourse_conjunction_graph_database))
        if any(item.operation_key not in known_operator_keys
               for item in self.input_discourse_conjunctions):
            raise ValueError(
                "conjunction graph references an operator absent from the operator graph")
        for metadata in (self.input_discourse_graph_metadata,
                         self.relation_question_graph_metadata):
            if (metadata is not None
                    and self.input_discourse_conjunction_graph_metadata is not None
                    and (metadata["source_identity"]
                         != self.input_discourse_conjunction_graph_metadata["source_identity"]
                         or metadata["license_identity"]
                         != self.input_discourse_conjunction_graph_metadata["license_identity"])):
                raise ValueError("question, operator and conjunction graphs have different provenance")
        self.relation_composition_witnesses = (
            () if relation_composition_graph_database is None
            else read_relation_composition_graph(relation_composition_graph_database))
        if type(relation_composition_verification) is not bool:
            raise TypeError("relation composition verification must be bool")
        self.relation_composition_verification = relation_composition_verification
        self.relation_composition_budget = relation_composition_budget
        validate_relation_composition_provenance(
            self.relation_composition_witnesses, self.relation_question_graph_metadata)
        self.core_runtime = TrainedRelationGraphRuntime(
            database,
            surface_artifact_database=surface_artifact_database,
        )
        # Source metadata is read only to prove whether two physical integer
        # rows can be the same authored assertion.  The query path never
        # imports source text or an external dataset into the release package.
        self._source_records = SourceRecordRepository(
            self.core_runtime.backend,
            registry=self.core_runtime.context.scoped_identity_store.registry,
        )
        self._diagnostic_stage("core_runtime_ready")
        try:
            self.pending_relation_compositions = pending_core_compositions(
                self.core_runtime, self.relation_composition_witnesses,
                budget=relation_composition_budget)
        except BaseException:
            self.core_runtime.close()
            raise
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
        self._diagnostic_stage("topologies_ready")
        self.event_time_topology: EventTimeGraphTopology = (
            read_event_time_topology(
                self.core_runtime.context.graph_ontology,
                event_time_structure_predicate_identities(),
                event_time_narrative_predicate_prefix(),
            )
        )
        if event_time_generation_binding_overlay is not None:
            overlay_edges = read_event_time_generation_overlay_edges(
                event_time_generation_binding_overlay)
            if overlay_edges:
                edges = tuple(sorted(
                    set(self.event_time_topology.edges).union(overlay_edges),
                    key=lambda item: item.stable_key()))

                def objects(edge_kind: int) -> tuple[tuple[int, ...], ...]:
                    return tuple(sorted({
                        item.object_key for item in edges
                        if item.edge_kind == edge_kind
                    }))

                self.event_time_topology = EventTimeGraphTopology(
                    edges,
                    tuple(sorted({
                        item.subject_key for item in edges
                        if item.edge_kind in {
                            1, 2, 3, 4, 5, 6, 9, 18, 19, 20, 21, 22, 23,
                            24,
                        }
                    })),
                    objects(1), objects(2), objects(3), objects(4),
                    objects(5), objects(18), objects(19),
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
        self._diagnostic_stage(f"facts_ready:{len(facts)}")
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
        self._diagnostic_stage(f"kdconv_ready:{len(self.kdconv_facts)}")
        self._semantic_input_views = tuple(semantic_input_views)
        # The carrier bridge is an optional extension of a trained model.  It
        # is restored read-only from the same SQLite so a query can connect a
        # projection to the complete existing semantic identity instead of
        # treating the carrier payload as a new fact.
        # The artifact bridge part table is a large optional projection.  A
        # normal text/conjunction query never consumes it, so defer the full
        # integer-part restore until an ArtifactQueryInput actually arrives.
        # This keeps cold startup bounded by the active relation graph rather
        # than eagerly materializing an unrelated carrier corpus.
        self._artifact_semantic_bindings: tuple[object, ...] = ()
        self._artifact_semantic_bindings_loaded = False
        self._artifact_bridge_available = False
        self._diagnostic_stage("artifact_deferred")
        # Event/Time predicates have a frozen integer identity contract.  Keep
        # their inverse/pair closure gate available before the projector is
        # built; the gate is limited to these registered relation predicates.
        self._event_time_predicate_keys = frozenset(
            authored_relation_identity(kind).stable_key()
            for kind in EVENT_TIME_RELATION_KINDS)
        graph_structures = RelationSurfaceStructureAdapter.from_facts(facts)
        kdconv_structures = RelationSurfaceStructureAdapter.from_facts(
            self.kdconv_facts)
        semantic_structures = SemanticCandidateStructureAdapter.from_views(
            semantic_input_views)
        self._relation_question_open_frames = bound_question_open_frames(
            self.relation_question_frames, facts)
        self.input_projector = TrainedInputStructureProjector(tuple(sorted(
            set((*graph_structures, *semantic_structures, *kdconv_structures)),
            key=lambda item: item.stable_key(),
        )), concept_links=restore_input_concept_links(self.core_runtime),
            open_role_frames=tuple((*restore_open_role_frames(
                self.core_runtime.active_surface_frames(), facts),
                *self._relation_question_open_frames)),
            envelope_role_frames=restore_open_role_frames(
                self.core_runtime.active_surface_frames()),
            discourse_operators=self.input_discourse_operators,
            slot_question_frames=tuple(item for item in self.relation_question_frames
                                       if item.is_slot_question),
            complete_question_witness_frames=tuple(item for item in self.relation_question_frames
                                                  if not item.is_slot_question),
            pair_closure_requires_predicate_keys=(
                self._event_time_predicate_keys))
        self._diagnostic_stage("input_projector_ready")
        # Memory 写入和查询共享同一个训练后输入投影器。这样 O/H/E 的候选键
        # 只来自已训练图拓扑，而不是历史原文、posting 或字符特征。
        self.memory = None
        self._owns_memory = memory_database is not None
        if memory_database is not None:
            with relation_phase("memory_init", "TrainedGraphQueryBridge.__init__.memory"):
                self.memory = TrainedDialogueMemoryGraph(
                    memory_database,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    input_projector=self.input_projector,
                )
        self._diagnostic_stage("memory_ready")
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
        # Query-time Memory records are immutable after restoration.  Keep
        # integer-keyed indexes so every frontier hop can resolve O/H/E
        # objects directly instead of scanning the complete restored tuples.
        self._memory_observations_by_key: dict[
            tuple[int, ...], MemoryObservationAnchor] = {}
        self._memory_hypotheses_by_key: dict[
            tuple[int, ...], MemoryHypothesisAnchor] = {}
        self._memory_evidence_by_key: dict[
            tuple[int, ...], MemoryEvidenceAnchor] = {}
        self._memory_hypotheses_by_observation: dict[
            tuple[int, ...], tuple[MemoryHypothesisAnchor, ...]] = {}
        self._memory_evidence_by_hypothesis: dict[
            tuple[int, ...], tuple[MemoryEvidenceAnchor, ...]] = {}
        self._memory_active_object_keys: frozenset[
            tuple[int, tuple[int, ...]]] = frozenset()
        self._discourse_hypotheses: tuple[MemoryHypothesisAnchor, ...] = ()
        self._current_discourse_topic_candidates: tuple[
            DiscourseTopicCandidate, ...] = ()
        self._current_discourse_topic_graph_identity_keys: tuple[
            tuple[int, ...], ...] = ()
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
        self._input_dialogue_act_match = None
        self._input_observation_ref = None
        self._response_contexts = ()
        self._delivered_graph_slot_projections: tuple[
            DeliveredGraphSlotProjection, ...] = ()
        self._input_role_occurrences: tuple[InputRoleOccurrence, ...] = ()
        self._input_open_relations_by_hypothesis: dict[
            tuple[int, ...], OpenRelationCandidate] = {}
        self._reference_candidate_sets: tuple[ReferenceCandidateSet, ...] = ()
        self._reference_resolutions: tuple[CrossTurnReferenceResolution, ...] = ()
        self._reference_resolution_cache = OrderedDict()
        self._artifact_bridge_matches: tuple[object, ...] = ()
        self._artifact_bridge_missing: tuple[tuple[int, ...], ...] = ()
        self._reference_runtime = (
            None if self.memory is None else TrainedCrossTurnReferenceRuntime(
                self.core_runtime, self.input_projector, self.memory.occurrence_index))
        self._discourse_topic_query_projection: tuple[int, ...] = ()
        self._kdconv_query_propositions: set[tuple[int, ...]] = set()
        self._active_relation_keys: frozenset[tuple[int, ...]] = frozenset()
        self._fact_query_operations: tuple[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...
        ] = ()
        self._restored_factual_topic_keys: tuple[tuple[int, ...], ...] = ()
        self._diagnostic_stage("bridge_init_exit")

    def _diagnostic_stage(self, value: str) -> None:
        """Optionally append a startup marker for bounded read-only profiling."""
        target = getattr(self, "_diagnostic_stage_file", "")
        if not target:
            return
        try:
            with open(target, "a", encoding="ascii", newline="\n") as stream:
                stream.write(value + "\n")
        except OSError:
            pass

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
        )), concept_links=concept_links, open_role_frames=open_frames,
            envelope_role_frames=self.input_projector.envelope_role_frames,
            discourse_operators=self.input_discourse_operators,
            slot_question_frames=tuple(item for item in self.relation_question_frames
                                       if item.is_slot_question),
            complete_question_witness_frames=tuple(item for item in self.relation_question_frames
                                                  if not item.is_slot_question),
            pair_closure_requires_predicate_keys=(
                self._event_time_predicate_keys))
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
        if not self._artifact_semantic_bindings_loaded:
            self._artifact_semantic_bindings = (
                self._restore_artifact_semantic_bindings())
            self._artifact_semantic_bindings_loaded = True
            self._artifact_bridge_available = bool(
                self._artifact_semantic_bindings)
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

    @observed_relation_phase("memory_recover")
    def _restore_memory_structures(
            self,
            input_structure: QueryInputStructure,
            artifact_input: ArtifactQueryInput | None = None,
            ) -> MemoryActiveStructures:
        """在一次完整 Memory 恢复内复用物理事件解码。"""
        if self.memory is None:
            return self._restore_memory_structures_impl(
                input_structure, artifact_input)
        with self.memory.intake.event_log.read_scope():
            return self._restore_memory_structures_impl(
                input_structure, artifact_input)

    def _restore_memory_structures_impl(
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
        self._memory_observations_by_key = {}
        self._memory_hypotheses_by_key = {}
        self._memory_evidence_by_key = {}
        self._memory_hypotheses_by_observation = {}
        self._memory_evidence_by_hypothesis = {}
        self._memory_active_object_keys = frozenset()
        self._discourse_hypotheses = ()
        self._current_discourse_topic_candidates = (
            derive_discourse_topic_candidates(input_structure))
        self._current_discourse_topic_graph_identity_keys = (
            graph_backed_discourse_topic_keys(
            self.core_runtime.context.graph_ontology,
            self._current_discourse_topic_candidates,
        ))
        self._discourse_topic_candidates = self._current_discourse_topic_candidates
        self._discourse_topic_graph_identity_keys = (
            self._current_discourse_topic_graph_identity_keys)
        self._discourse_topic_hypothesis_keys = ()
        self._memory_routes = ()
        self._response_contexts = ()
        self._delivered_graph_slot_projections = ()
        self._input_role_occurrences = ()
        self._input_open_relations_by_hypothesis = {}
        self._reference_candidate_sets = ()
        self._reference_resolutions = ()
        self._current_reference_scope = None
        self._reference_resolution_cache = OrderedDict()
        if self._reference_runtime is not None:
            self._reference_runtime.clear_context_cache()
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
        # 写入与恢复共用当前指代义务；普通 WH 的已绑定观察角色即使
        # 同时出现在开放语法跨度中，也不能因此重开上一轮回答的角色。
        reference_context_required = extended_reference_context_required(
            input_structure, self.relation_question_frames)
        has_closed_relation = any(
            item.state == STRUCTURE_CLOSED
            and item.required_open == 0
            and item.required_count == item.support_count
            and item.predicate_coverage == 1
            for item in input_structure.relation_candidates)
        context_depth_limit = 1
        if not reference_context_required and (
                input_structure.relation_question_bindings
                or has_closed_relation):
            context_depth_limit = 0
        selected = self.memory.active_structures(
            candidate_keys=candidate_keys,
            observation_refs=(() if self._input_observation_ref is None else (self._input_observation_ref,)),
            # Production QueryState restores the current turn plus the
            # immediately linked delivered response.  The graph candidate
            # routes still select any directly evidenced prior hypothesis;
            # unrelated historical context edges must not expand recursively.
            context_depth_limit=context_depth_limit,
            recall_delivered_outputs=reference_context_required,
        )
        # Open predicate-covered relations may produce category-6/7 Memory
        # topic routes for bounded recall, but they are not active discourse
        # topics until the current typed relation is genuinely closed. Keep
        # those rows available to direct ``active_structures`` callers while
        # removing them from this production QueryState so they cannot supply
        # a topic root or block Core closure.
        closed_topic_keys = {
            item.candidate_key
            for item in self._discourse_topic_candidates
        }
        filtered_hypotheses = tuple(
            item for item in selected.hypotheses
            if not (
                item.candidate_key
                and item.candidate_key[0] == MEMORY_CANDIDATE_VERSION
                and memory_candidate_category(item.candidate_key)
                in {MEMORY_CATEGORY_TIME, MEMORY_CATEGORY_DISCOURSE}
                and item.candidate_key not in closed_topic_keys
            )
        )
        if len(filtered_hypotheses) != len(selected.hypotheses):
            kept_hypothesis_keys = {
                item.hypothesis_key for item in filtered_hypotheses}
            selected = MemoryActiveStructures(
                selected.observations,
                tuple(sorted(filtered_hypotheses,
                             key=lambda item: item.hypothesis_key)),
                tuple(sorted(
                    (item for item in selected.evidence
                     if item.hypothesis_key in kept_hypothesis_keys),
                    key=lambda item: item.evidence_key)),
                selected.match_routes,
            )
        self._response_contexts = restore_response_contexts(self.memory, selected)
        from pure_integer_ai.experiments.current_reference_scope import read_current_reference_scope
        self._current_reference_scope = read_current_reference_scope(
            self.memory, self._input_observation_ref, self._response_contexts)
        selected = _root_response_context_inputs(
            self.memory, selected, self._response_contexts)
        if reference_context_required:
            from pure_integer_ai.experiments.memory_reference_read_scope import retain_reference_role_dependencies
            selected = retain_reference_role_dependencies(
                self.memory, selected, self._response_contexts,
                self._input_observation_ref, self._current_reference_scope)
        _validate_response_context_memory_closure(
            selected, self._response_contexts)
        self._delivered_graph_slot_projections = (
            delivered_graph_slot_projections(self._response_contexts))
        if self._input_observation_ref is not None:
            candidates = {
                frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,),
                          item.stable_key()): item
                for item in input_structure.open_relations}
            occurrences = []
            input_open_relations_by_hypothesis = {}
            for item in selected.hypotheses:
                candidate = candidates.get(item.candidate_key)
                if candidate is None or item.observation_key != self._input_observation_ref.stable_key():
                    continue
                input_open_relations_by_hypothesis[item.hypothesis_key] = candidate
                occurrences.extend(materialize_input_roles(
                    self.memory, candidate, self._input_observation_ref,
                    MemoryObjectRef.from_stable_key(item.hypothesis_key)))
            self._input_open_relations_by_hypothesis = (
                input_open_relations_by_hypothesis)
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
            observed_role_keys = _observed_input_role_keys(input_structure)
            for reference in self._input_role_occurrences:
                if reference.role.stable_key() in observed_role_keys:
                    continue
                relation = input_open_relations_by_hypothesis.get(
                    reference.hypothesis.stable_key())
                if relation is None or not reference_marker_sources(
                        reference.values, reference.role.stable_key(),
                        relation.frame.predicate, self.relation_question_frames):
                    continue
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
        self._memory_observations_by_key = {
            item.observation_key: item for item in self._memory_observations}
        self._memory_hypotheses_by_key = {
            item.hypothesis_key: item for item in self._memory_hypotheses}
        self._memory_evidence_by_key = {
            item.evidence_key: item for item in self._memory_evidence}
        hypotheses_by_observation: dict[
                tuple[int, ...], list[MemoryHypothesisAnchor]] = {}
        for item in self._memory_hypotheses:
            hypotheses_by_observation.setdefault(
                item.observation_key, []).append(item)
        self._memory_hypotheses_by_observation = {
            key: tuple(sorted(values, key=lambda item: item.hypothesis_key))
            for key, values in hypotheses_by_observation.items()
        }
        evidence_by_hypothesis: dict[
                tuple[int, ...], list[MemoryEvidenceAnchor]] = {}
        for item in self._memory_evidence:
            evidence_by_hypothesis.setdefault(
                item.hypothesis_key, []).append(item)
        self._memory_evidence_by_hypothesis = {
            key: tuple(sorted(values, key=lambda item: item.evidence_key))
            for key, values in evidence_by_hypothesis.items()
        }
        self._memory_active_object_keys = frozenset({
            *((MEMORY_OBJECT_OBSERVATION, item.observation_key)
              for item in self._memory_observations),
            *((MEMORY_OBJECT_HYPOTHESIS, item.hypothesis_key)
              for item in self._memory_hypotheses),
            *((MEMORY_OBJECT_EVIDENCE, item.evidence_key)
              for item in self._memory_evidence),
        })
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
        observation = self._memory_observations_by_key.get(
            self._input_observation_ref.stable_key())
        if observation is None:
            raise ValueError("current reference query has no unique Observation sequence")
        return observation.turn_seq

    def _reference_relation_context(
            self,
            candidate_set: ReferenceCandidateSet,
            input_structure: QueryInputStructure,
            ) -> tuple[
                OpenRelationCandidate | None,
                tuple[tuple[tuple[int, ...], ObjectIdentity], ...],
            ]:
        """Resolve one open relation and its observed role fillers.

        A missing ``to`` role is the reference being resolved; its dynamic
        span has no semantic object yet.  REFERS evidence must therefore be
        authorized by the already observed role(s) in that same trained open
        frame.  Returning an empty tuple on any structural ambiguity keeps the
        resolver fail-closed instead of falling back to a whole-input or text
        proximity interpretation.
        """
        reference = candidate_set.reference
        relation_map = getattr(self, "_input_open_relations_by_hypothesis", None)
        if relation_map is None:
            # Lightweight callers used by the structural unit tests may invoke
            # this helper on an uninitialized bridge.  Preserve the old
            # fail-closed behavior only when exactly one current candidate
            # matches the reference span; production bridges always take the
            # hypothesis-keyed path above.
            relation_matches = tuple(
                item for item in input_structure.open_relations
                if any(
                    span.role == reference.role.stable_key()
                    and span.ordinal == reference.ordinal
                    and span.values == reference.values
                    and span.source_ref == input_structure.source_ref
                    for span in item.bindings))
            relation = relation_matches[0] if len(relation_matches) == 1 else None
        else:
            relation = relation_map.get(reference.hypothesis.stable_key())
        if relation is None or relation not in input_structure.open_relations:
            return None, ()
        matches = tuple(
            item for item in relation.bindings
            if (item.role == reference.role.stable_key()
                and item.ordinal == reference.ordinal
                and item.values == reference.values
                and item.source_ref == input_structure.source_ref))
        if len(matches) != 1:
            return None, ()
        observed = []
        for binding in relation.bindings:
            if binding.role == reference.role.stable_key():
                continue
            span_refs = {
                item.ref_key for item in input_structure.spans
                if (item.start == binding.start
                    and item.end == binding.end
                    and item.values == binding.values
                    and item.source_ref == input_structure.source_ref)
            }
            matches = tuple(
                item for item in input_structure.semantic_candidates
                if (item.projection_kind == PROJECTION_FILLER
                    and item.proposition_key == relation.frame.proposition
                    and item.predicate_key == relation.frame.predicate
                    and item.role_key == binding.role
                    and item.span_ref in span_refs))
            if len(matches) != 1:
                return None, ()
            try:
                observed.append((
                    binding.role,
                    ObjectIdentity.from_stable_key(matches[0].candidate_key),
                ))
            except (TypeError, ValueError):
                return None, ()
        expected_roles = {
            role for role in relation.frame.roles
            if role != reference.role.stable_key()
        }
        if ({role for role, _filler in observed} != expected_roles
                or len(observed) != len(expected_roles)):
            return None, ()
        fillers = tuple(sorted(
            set(observed), key=lambda item: (item[0], item[1].stable_key())))
        return relation, fillers

    def _reference_current_objects(
            self,
            candidate_set: ReferenceCandidateSet,
            input_structure: QueryInputStructure,
            ) -> tuple[ObjectIdentity, ...]:
        """Project observed fillers for the existing generic resolver paths."""
        _relation, fillers = self._reference_relation_context(
            candidate_set, input_structure)
        return tuple(sorted({filler for _role, filler in fillers},
                            key=lambda item: item.stable_key()))

    def _refresh_reference_resolutions(
            self,
            state: QueryState,
            input_structure: QueryInputStructure | None = None,
            ) -> QueryState:
        """Resolve only from Core facts already expanded in this QueryState."""
        if self._reference_runtime is None or not self._reference_candidate_sets:
            self._reference_resolutions = ()
            return state
        current = current_query_evidence(state.evidence)
        core_support = frozenset(
            item.hypothesis_key for item in current
            if item.space == _CORE and item.polarity == 1
        )
        timestamp_seq = self._reference_timestamp()
        cache = getattr(self, "_reference_resolution_cache", None)
        if cache is None:
            cache = self._reference_resolution_cache = OrderedDict()
        source_ref = (None if input_structure is None
                      else input_structure.source_ref)
        cache_key = (source_ref, core_support, timestamp_seq)
        cached = cache.get(cache_key)
        if cached is not None:
            if isinstance(cache, OrderedDict):
                cache.move_to_end(cache_key)
            resolutions, cached_bindings, cached_evidence = cached
            self._reference_resolutions = resolutions
            if (all(item in state.bindings for item in cached_bindings)
                    and all(item in state.evidence for item in cached_evidence)):
                return state
            return state.with_(
                bindings=tuple(sorted({*state.bindings, *cached_bindings},
                                      key=lambda item: item.stable_key())),
                evidence=tuple(sorted({*state.evidence, *cached_evidence},
                                      key=lambda item: item.stable_key())),
            )
        resolutions = []
        for candidate_set in self._reference_candidate_sets:
            relation_frame = None
            current_role_fillers = ()
            current_objects = None
            relation_answer_role = ()
            if input_structure is not None:
                relation_frame, current_role_fillers = (
                    self._reference_relation_context(
                        candidate_set, input_structure))
                current_objects = tuple(sorted(
                    {filler for _role, filler in current_role_fillers},
                    key=lambda item: item.stable_key()))
                slot_binding = next((
                    item for item in input_structure.relation_question_bindings
                    if item.answer_role_key), None)
                if slot_binding is not None:
                    slot_parts = self._relation_slot_question_parts(
                        input_structure, slot_binding)
                    if (slot_parts is not None
                            and candidate_set.reference.role.stable_key()
                            in slot_parts[4]):
                        # A reference slot keeps the unresolved endpoint in the
                        # generic open relation frame.  Its question answer role
                        # is therefore the same integer role as the reference,
                        # rather than an additional omitted relation role.  Passing
                        # it as ``relation_answer_role`` would make the resolver
                        # reject an otherwise exact REFERS/Core witness because
                        # that role is already present in ``relation_frame``.
                        if not slot_parts[0].is_reference_slot:
                            relation_answer_role = slot_binding.answer_role_key
            resolutions.append(self._reference_runtime.resolve(
                candidate_set,
                self._response_contexts,
                core_support=core_support,
                timestamp_seq=timestamp_seq,
                current_objects=current_objects,
                relation_frame=relation_frame,
                current_role_fillers=current_role_fillers,
                relation_answer_role=relation_answer_role,
                reference_scope=getattr(self, '_current_reference_scope', None),
            ))
        resolutions = tuple(resolutions)
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
        cache[cache_key] = (
            resolutions,
            tuple(item for item in bindings if item not in state.bindings),
            tuple(item for item in evidence if item not in state.evidence),
        )
        if isinstance(cache, OrderedDict):
            cache.move_to_end(cache_key)
            while len(cache) > 32:
                cache.popitem(last=False)
        return state.with_(bindings=bindings, evidence=evidence)

    def _reference_generation_contexts(self) -> tuple:
        """Select an exact topic context only after an evidence-qualified winner."""
        required = tuple(item for item in self._reference_resolutions
                         if item.requires_resolution)
        if not required:
            return ()
        if any(item.status != REFERENCE_RESOLVED for item in required):
            return ()
        keys = {key for item in required for key in item.support_context_keys}
        selected = tuple(item for item in self._response_contexts
                         if item.stable_key() in keys)
        if {item.stable_key() for item in selected} != keys:
            raise ValueError("reference winner topic context is not recoverable")
        return tuple(sorted(selected, key=lambda item: item.stable_key()))

    def _generic_response_feature_mask(self, hypotheses,
                                       input_structure=None) -> int:
        """Derive response-act conditions from the current shared graph view."""
        mask = RESPONSE_FEATURE_UNKNOWN
        if input_structure is not None:
            mask |= assertion_scope_feature(input_structure)
        dialogue_act_match = getattr(self, "_input_dialogue_act_match", None)
        if dialogue_act_match is not None:
            mask |= dialogue_act_match.response_feature_mask
        # Generic Memory candidates are deliberately present for every unseen
        # input.  They must not manufacture a graph condition merely because
        # the projector emitted an open/partial relation shape.  Only a
        # relation whose trained predicate and all required fillers are closed
        # is a current-input graph signal for connector priority.
        complete_relation = bool(input_structure
                                 and not input_structure.discourse_compositions
                                 and not has_uncovered_closed_relation_modifier(input_structure)
                                 and any(
            item.state == STRUCTURE_CLOSED
            and item.required_count >= 2
            and item.required_count == item.support_count
            and item.predicate_coverage == 1
            for item in input_structure.relation_candidates))
        linked_contexts = self._reference_generation_contexts()
        if linked_contexts:
            # A shallow context must not permanently mask a deeper context
            # restored through the same Memory O/H/E closure: doing so makes
            # the trained COLD condition unreachable after the first turn.
            # Context depth is an already verified integer field; no surface
            # or recency text is inspected here.
            deepest = max(
                linked_contexts,
                key=lambda item: (item.context_depth, item.stable_key()),
            )
            if deepest.context_depth <= 1:
                mask |= RESPONSE_FEATURE_HOT_CONTEXT
            else:
                mask |= RESPONSE_FEATURE_COLD_CONTEXT
        if self._current_discourse_topic_candidates:
            mask |= RESPONSE_FEATURE_TOPIC
        if self._current_discourse_topic_graph_identity_keys:
            mask |= RESPONSE_FEATURE_TOPIC_GRAPH
        if self._reference_candidate_sets:
            mask |= RESPONSE_FEATURE_REFERENCE_OPEN
        if len(tuple(hypotheses)) > 1:
            mask |= RESPONSE_FEATURE_MULTI_HYPOTHESIS
        if (input_structure is not None
                and (input_structure.discourse_compositions
                     or input_structure.discourse_gaps)):
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
        required_references = tuple(
            item for item in self._reference_resolutions
            if item.requires_resolution)
        if (required_references
                and all(item.status == REFERENCE_RESOLVED
                        for item in required_references)):
            mask |= RESPONSE_FEATURE_REFERENCE_RESOLVED
        return mask

    def _input_dialogue_act_trace(self) -> tuple[int, ...]:
        """Return the mounted graph and unique current-input match identity."""
        graph = getattr(self, "input_dialogue_act_graph", None)
        match = getattr(self, "_input_dialogue_act_match", None)
        if graph is None or match is None:
            return ()
        return graph.graph_id, *match.stable_key()

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
        active.update(
            item.proposition_key
            for item in self._current_discourse_topic_candidates)
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

    @observed_relation_phase("generation")
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
        if any(item.role_key == (SCOPE_QUERY_PROOF,) for item in state.bindings):
            # 内层命题可作证据，带作用域的当前陈述不能直接变成该命题的事实输出。
            return
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
            except (RuntimeError, TypeError, ValueError) as error:
                self._diagnostic_stage(
                    f"generation_failed:{len(key)}:{key[0] if key else 0}:"
                    f"{type(error).__name__}:"
                    f"{str(error).encode('ascii', 'backslashreplace').decode('ascii')}")
                continue
            if plan is not None:
                self.generation_by_proposition[key] = generated
                self.generation_ready_propositions.add(key)
                self._diagnostic_stage(
                    f"generation_ready:{len(key)}:{key[0] if key else 0}")
            else:
                self._diagnostic_stage(
                    f"generation_plan_none:{len(key)}:{key[0] if key else 0}")

    @staticmethod
    def anchor_concepts(
            input_structure: QueryInputStructure,
            *,
            allowed_predicate_keys: frozenset[tuple[int, ...]] | None = None,
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
                and (allowed_predicate_keys is None
                     or item.predicate_key in allowed_predicate_keys)
                and (item.predicate_coverage == 1
                     or (item.required_count == 2
                         and item.predicate_coverage == 0)))
        }
        relation_by_key = {
            item.stable_key(): item for item in input_structure.relation_candidates}
        complete_propositions.update(
            relation_by_key[binding.relation_key].proposition_key
            for binding in input_structure.relation_question_bindings
            if binding.answer_role_key
            and binding.relation_key in relation_by_key
            and relation_by_key[binding.relation_key].state == STRUCTURE_OPEN
            and relation_by_key[binding.relation_key].required_open == 1
            and relation_by_key[binding.relation_key].conflict_count == 0)
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
            *,
            allowed_predicate_keys: frozenset[tuple[int, ...]] | None = None,
            ) -> tuple[tuple[InputRelationCandidate, ActiveRelationSurface], ...]:
        """Resolve a binary inverse query from two exact trained fillers.

        The projector intentionally leaves the predicate cue open when an
        inverse surface is unseen.  This bridge only closes the candidate when
        both filler identities and their trained role ordering match exactly;
        no lexical or character fallback is involved.
        """
        matches = []
        for relation in input_structure.relation_candidates:
            if not (relation.required_count == 2
                    and relation.support_count == 2
                    and relation.predicate_coverage == 0):
                continue
            if (allowed_predicate_keys is not None
                    and relation.predicate_key not in allowed_predicate_keys):
                continue
            filler_keys = {
                item.candidate_key
                for item in input_structure.semantic_candidates
                if item.projection_kind == PROJECTION_FILLER
                and item.proposition_key == relation.proposition_key
                and item.source_ref == relation.source_ref
            }
            if len(filler_keys) != 2:
                continue
            for fact in facts:
                if len(fact.bindings) != 2:
                    continue
                if fact.predicate.stable_key() != relation.predicate_key:
                    continue
                fact_fillers = {
                    item.filler.stable_key() for item in fact.bindings}
                if fact_fillers == filler_keys:
                    matches.append((relation, fact))
        # An unseen inverse surface may omit the predicate cue entirely.  The
        # filler pair alone is therefore authoritative only when it identifies
        # one proposition/predicate.  If the same two fillers participate in
        # more than one relation (for example CAUSES and EVENT_BEFORE),
        # choosing by candidate score would silently answer a different
        # question.  Keep duplicate structural projections of one semantic
        # witness, but reject competing predicates/propositions fail-closed.
        semantic_matches = {
            (relation.proposition_key, relation.predicate_key,
             fact.proposition.stable_key())
            for relation, fact in matches
        }
        if len(semantic_matches) > 1:
            return ()
        return tuple(sorted(matches, key=lambda item: (
            item[0].proposition_key, item[0].structure_key)))

    _core_relation_endpoint_identity = staticmethod(core_relation_endpoint_identity)

    def _input_relation_competition(self, structure, *, question_predicate_keys):
        """使用只读 Core 事实索引执行当前输入范围内的纯整数竞争检查。"""
        return input_relation_competition(
            structure, self.core_fact_index,
            question_predicate_keys=question_predicate_keys,
            question_frames=self.relation_question_frames)

    def _rank_candidates(
            self,
            input_structure: QueryInputStructure,
            proposition_keys: tuple[tuple[int, ...], ...],
            *,
            active_relation_keys: frozenset[tuple[int, ...]] | None = None,
            ) -> tuple[
                tuple[int, tuple[int, ...]],
                tuple[int, tuple[int, ...]],
            ]:
        """按已训练角色拓扑闭合度排名，不做词面或字符近邻评分。"""
        by_semantic: dict[tuple[object, ...], list[InputRelationCandidate]] = {}
        for candidate in input_structure.relation_candidates:
            # An empty active set is the normal non-composition view: the
            # composition selector has no scope to narrow, so every current
            # relation candidate remains eligible.  Treating ``frozenset()``
            # as an explicit deny-all filter erased the best candidate for
            # complete natural relation questions after their exact binding
            # was attached, leaving Core evidence at zero and forcing the
            # generic UNKNOWN route.  A non-empty set remains the strict
            # scope filter used by verified conjunctions and target adoption.
            if (active_relation_keys
                    and candidate.stable_key() not in active_relation_keys):
                continue
            by_semantic.setdefault(
                relation_candidate_semantic_key(input_structure, candidate),
                []).append(candidate)
        ranked = []
        allowed = set(proposition_keys)
        for candidates in by_semantic.values():
            candidates = tuple(item for item in candidates
                               if item.proposition_key in allowed)
            if not candidates:
                continue
            score = max(item.score for item in candidates)
            representative = min(
                (item for item in candidates if item.score == score),
                key=lambda item: item.stable_key(),
            )
            ranked.append((score, representative.proposition_key))
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
            if (candidate.state != STRUCTURE_CLOSED
                    or candidate.predicate_coverage != 1):
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
        if additions and input_structure.discourse_compositions:
            # A compound discourse names its own relation identities.  Do not
            # admit unrelated KdConv page-in facts merely because their spans
            # share a subject with one operand; they would widen the common
            # frontier and turn a closed composition into an arbitrary
            # property search.  Referenced candidates remain in the structure
            # below so the composition contract can validate them.
            referenced_relation_keys = {
                relation_key
                for composition in input_structure.discourse_compositions
                for group in composition.operand_relation_keys
                for relation_key in group
            }
            additions = [
                item for item in additions
                if item.stable_key() in referenced_relation_keys]
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
                # A conjunction is a trained relation over the exact
                # candidate identities emitted by the projector.  KdConv
                # semantic deduplication must not discard one of those
                # identities before QueryInputStructure validates the
                # composition references.  Preserve every candidate named by
                # a discourse composition; later scope/conjunction checks
                # still decide whether it is authorized and closed.
                referenced_relation_keys = {
                    relation_key
                    for composition in input_structure.discourse_compositions
                    for group in composition.operand_relation_keys
                    for relation_key in group
                }
                if input_structure.discourse_compositions:
                    # Page-in may re-project many unrelated KdConv property
                    # candidates for the same repeated subject.  A discourse
                    # query admits only the exact relation identities named by
                    # its trained composition; unrelated closed properties are
                    # outside this QueryState and must not widen the frontier.
                    canonical_closed = tuple(
                        item for item in canonical_closed
                        if item.stable_key() in referenced_relation_keys)
                canonical_by_key = {
                    item.stable_key(): item for item in canonical_closed}
                for item in input_structure.relation_candidates:
                    if item.stable_key() in referenced_relation_keys:
                        canonical_by_key[item.stable_key()] = item
                canonical_closed = tuple(sorted(
                    canonical_by_key.values(), key=lambda item: item.stable_key()))
                semantic_selected = frozenset(
                    item.proposition_key for item in canonical_closed
                    if item.state == STRUCTURE_CLOSED
                    and item.required_open == 0)
                return replace(
                    input_structure,
                    semantic_candidates=tuple(sorted(
                        (item for item in input_structure.semantic_candidates
                         if item.proposition_key in semantic_selected),
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
        # itself untouched for later queries.  A discourse composition is a
        # separate integer contract: preserve every candidate and semantic
        # member it names so QueryInputStructure can validate the composition
        # after KdConv adds its exact property candidate.  Without this, a
        # valid conjunction can reference a candidate removed by this
        # property disambiguation boundary and fail as an unknown relation.
        selected = set(self._kdconv_query_propositions)
        referenced_relation_keys = {
            relation_key
            for composition in input_structure.discourse_compositions
            for group in composition.operand_relation_keys
            for relation_key in group
        }
        referenced_candidates = {
            item for item in input_structure.relation_candidates
            if (item.stable_key() in referenced_relation_keys
                and item.state == STRUCTURE_CLOSED
                and item.required_open == 0)
        }
        # Keep the exact relation identities named by the composition.  Using
        # proposition_key here would re-admit every partial surface candidate
        # for that proposition and manufacture a ranked tie in QueryState.
        selected.update(item.proposition_key for item in referenced_candidates)
        semantic_candidates = tuple(
            item for item in input_structure.semantic_candidates
            if item.proposition_key in selected)
        closed_referenced_relation_keys = {
            item.stable_key() for item in referenced_candidates}
        relation_candidates = tuple(
            item for item in input_structure.relation_candidates
            if item.proposition_key in self._kdconv_query_propositions
            or item.stable_key() in referenced_relation_keys)
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
        # A trained relation-question frame already owns the current
        # predicate/role scope.  Restoring a prior KdConv topic here would
        # replace that open REFERS frame before Memory can materialize its
        # observed role and invoke the cross-turn resolver.
        if (self.relation_question_frames
                and (input_structure.relation_question_bindings
                     or input_structure.open_relations)):
            return input_structure
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
            candidate_keys=(), observation_refs=observation_refs,
            context_depth_limit=0)
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

    def _normalize_operation_keys(
            self,
            operation_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[tuple[int, ...], ...]:
        """Validate explicit integer operation identities against the cold graph.

        An operation annotation is accepted only when its contract was restored
        from the read-only operation graph mounted on this bridge.  The
        annotation is an integer graph input; it is never inferred from a
        character, proposition tail, or receipt metadata.
        """
        if (type(operation_keys) is not tuple
                or any(type(key) is not tuple or not key
                       or any(type(value) is not int or value < 0 for value in key)
                       for key in operation_keys)
                or operation_keys != tuple(sorted(set(operation_keys)))):
            raise ValueError("operation_keys 必须是规范纯整数键")
        if any(key not in self.input_discourse_operation_modes
               for key in operation_keys):
            raise ValueError("输入 operation 未被当前 operation contract graph 授权")
        return operation_keys

    def _attach_operation_keys(
            self,
            input_structure: QueryInputStructure,
            operation_keys: tuple[tuple[int, ...], ...],
            ) -> QueryInputStructure:
        """Attach contract-validated operations to the current integer input."""
        # A prepared structure may already carry operation identities produced
        # by the trained integer surface projector.  An empty caller argument
        # must preserve those identities; otherwise the old runner could erase
        # a real parse or inject a course label after projection.
        if not operation_keys and input_structure.operation_keys:
            operation_keys = input_structure.operation_keys
        if (operation_keys and input_structure.tokens
                and not input_structure.operation_keys):
            raise ValueError(
                "自然语言输入的 operation 必须由训练整数 surface projector 命中")
        normalized = self._normalize_operation_keys(operation_keys)
        if (input_structure.operation_keys
                and input_structure.operation_keys != normalized):
            raise ValueError("input_structure operation_keys 与 query 不一致")
        if input_structure.operation_keys == normalized:
            return input_structure
        return replace(input_structure, operation_keys=normalized)

    def _relation_slot_question_parts(
            self, structure: QueryInputStructure,
            binding,
            ):
        """Validate a WH binding against one exact open input and Core fact."""
        if not binding.answer_role_key:
            return None
        try:
            frame = restore_slot_question_frame(
                binding.frame_key, trained_frames=self.relation_question_frames)
        except (TypeError, ValueError):
            return None
        if (frame is None or not frame.is_slot_question
                or frame.operation_key != binding.operation_key
                or frame.answer_role_key != binding.answer_role_key
                or structure.discourse_compositions or structure.discourse_gaps
                or len(binding.slot_span_refs) != len(frame.roles)):
            return None
        relation = next((item for item in structure.relation_candidates
                         if item.stable_key() == binding.relation_key), None)
        if (relation is None
                or relation.state not in {STRUCTURE_OPEN, STRUCTURE_CLOSED}
                or relation.predicate_key != frame.predicate_key
                or relation.conflict_count != 0
                or relation.predicate_coverage != 1):
            return None
        fact = self.core_fact_index.get(relation.proposition_key)
        if (fact is None or fact.predicate.stable_key() != frame.predicate_key
                or fact.source_hash <= 0):
            return None
        fact_roles = tuple(item.role.stable_key() for item in fact.bindings)
        if (len(set(fact_roles)) != len(fact_roles)
                or binding.answer_role_key not in fact_roles
                or not question_observed_role_contract(frame, fact_roles)
                or relation.source_ref != (fact.source_hash,)
                or relation.required_count != len(fact.bindings) + 1):
            return None
        spans = {item.ref_key: item for item in structure.spans}
        if any(ref not in spans or spans[ref].source_ref != structure.source_ref
               for ref in binding.slot_span_refs):
            return None
        fact_fillers = {item.role.stable_key(): item.filler.stable_key()
                        for item in fact.bindings}
        if len(fact_fillers) != len(fact.bindings):
            return None
        reference_answer_slot = frame.is_reference_slot
        resolutions = tuple(
            item for item in getattr(self, "_reference_resolutions", ())
            if item.requires_resolution)
        resolved_role = (
            resolutions[0].candidate_set.reference.role.stable_key()
            if len(resolutions) == 1
            and resolutions[0].status == REFERENCE_RESOLVED
            and (resolutions[0].candidate_set.reference.role.stable_key()
                 in frame.roles
                 or (reference_answer_slot
                     and resolutions[0].candidate_set.reference.role.stable_key()
                     == frame.answer_role_key))
            and resolutions[0].winner_target_key
            == fact_fillers[resolutions[0].candidate_set.reference.role.stable_key()]
            else ())
        filler_members = tuple(
            item for item in structure.semantic_candidates
            if item.projection_kind == PROJECTION_FILLER
            and item.proposition_key == relation.proposition_key
            and item.predicate_key == relation.predicate_key
            and item.source_ref == relation.source_ref
            and item.span_ref in relation.span_refs)
        predicate_members = tuple(
            item for item in structure.semantic_candidates
            if item.projection_kind == PROJECTION_PREDICATE
            and item.proposition_key == relation.proposition_key
            and item.predicate_key == relation.predicate_key
            and item.source_ref == relation.source_ref
            and item.span_ref in relation.span_refs)
        predicate_gap_witness = False
        if not predicate_members and relation.predicate_coverage == 1:
            # The predicate may be proved only by the exact trained slot
            # frame's gap.  Keep this branch structural: one open-role parse
            # must agree on proposition, role order and every learned gap.
            predicate_gap_matches = _matching_open_relation_candidates(
                structure, frame, relation.proposition_key,
                relation.source_ref, binding.slot_span_refs)
            predicate_gap_witness = (
                len(predicate_gap_matches) == 1
                and predicate_gap_matches[0].frame.roles == frame.roles
                and predicate_gap_matches[0].frame.gaps == frame.gaps)
            if not predicate_gap_witness and reference_answer_slot:
                predicate_gap_witness = retained_reference_question_gap_witness(
                    structure, frame, relation, binding)
            # After the first validated view, the exact open parse is
            # intentionally removed before Memory intake.  The retained
            # binding plus the promoted one-answer-slot topology is the same
            # integer witness for this second validation pass.
            if (not predicate_gap_witness
                    and not reference_answer_slot
                    and not structure.open_relations
                    and structure.relation_question_bindings
                    and relation.required_open == 1
                    and relation.support_count == relation.role_coverage + 1):
                predicate_gap_witness = True
        if (len(predicate_members) != 1 and not predicate_gap_witness):
            return None
        known_refs = []
        dynamic_roles = []
        for role, span_ref in zip(frame.roles, binding.slot_span_refs):
            members = tuple(item for item in filler_members
                            if item.role_key == role and item.span_ref == span_ref)
            if role == resolved_role and relation.state == STRUCTURE_CLOSED:
                if (len(members) != 1
                        or members[0].candidate_key != fact_fillers[role]):
                    return None
                known_refs.append(span_ref)
                dynamic_roles.append(role)
                continue
            if len(members) == 1:
                if members[0].candidate_key != fact_fillers[role]:
                    return None
                known_refs.append(span_ref)
                continue
            if members:
                return None
            if (role == resolved_role
                    and relation.state == STRUCTURE_CLOSED):
                known_refs.append(span_ref)
                dynamic_roles.append(role)
                continue
            open_matches = _matching_open_relation_candidates(
                structure, frame, relation.proposition_key,
                relation.source_ref, binding.slot_span_refs)
            if len(open_matches) != 1:
                return None
            span = spans[span_ref]
            open_bindings = tuple(item for item in open_matches[0].bindings
                                  if item.role == role
                                  and item.start == span.start
                                  and item.end == span.end
                                  and item.values == span.values)
            if len(open_bindings) != 1:
                return None
            dynamic_roles.append(role)
        # A reference slot omits the answer role from ``frame.roles`` by
        # design.  Its observed roles can all be exact Core fillers, so the
        # ordinary loop above has no missing observed role to mark dynamic.
        # Carry the omitted answer role explicitly from the trained frame
        # metadata; this keeps ordinary ``什么`` slots unchanged and avoids a
        # runtime word or surface list.
        if reference_answer_slot:
            if dynamic_roles or binding.answer_role_key != frame.answer_role_key:
                return None
            dynamic_roles.append(frame.answer_role_key)
        if (len(dynamic_roles) > 1
                or {item.role_key for item in filler_members} != (
                    {
                        role for role in frame.roles
                        if role not in dynamic_roles
                    }
                    | ({resolved_role}
                       if (relation.state == STRUCTURE_CLOSED
                           and resolved_role in dynamic_roles)
                       else set()))):
            return None
        if dynamic_roles:
            if resolved_role in dynamic_roles:
                if resolutions[0].winner_target_key != fact_fillers[resolved_role]:
                    return None
                if relation.state == STRUCTURE_OPEN:
                    if relation.role_coverage not in {
                            len(frame.roles) - len(dynamic_roles), len(frame.roles)}:
                        return None
                    if relation.role_coverage == len(frame.roles):
                        if (relation.required_open != 1
                                or relation.support_count != relation.role_coverage + 1):
                            return None
                    elif (relation.required_open != 1 + len(dynamic_roles)
                          or relation.support_count != relation.role_coverage + 1):
                        return None
                elif (relation.role_coverage != len(fact.bindings)
                      or relation.required_open != 0
                      or relation.support_count != relation.required_count):
                    return None
            elif reference_answer_slot:
                # The omitted answer role is the one required open slot;
                # every observed frame role must already be represented by a
                # current exact filler projection.
                if (relation.state != STRUCTURE_OPEN
                        or relation.role_coverage != len(frame.roles)
                        or relation.required_open != 1
                        or relation.support_count != relation.role_coverage + 1):
                    return None
            elif (relation.state != STRUCTURE_OPEN
                  or relation.role_coverage != len(frame.roles) - len(dynamic_roles)
                  or relation.required_open != 1 + len(dynamic_roles)
                  or relation.support_count != relation.role_coverage + 1):
                return None
        elif (relation.state != STRUCTURE_OPEN
              or relation.role_coverage != len(frame.roles)
              or relation.required_open != 1
              or relation.support_count != relation.role_coverage + 1):
            return None
        predicate_span = (predicate_members[0].span_ref
                          if predicate_members else None)
        reference_dynamic_span_refs = ()
        if reference_answer_slot and relation.state == STRUCTURE_CLOSED:
            # The authored reference question omits the answer role from its
            # slot list.  After target adoption the full open-role candidate
            # still carries that role's exact interval, so include the same
            # source span in the relation topology check.
            full_candidates = tuple(
                item for item in structure.open_relations
                if (item.frame.proposition == relation.proposition_key
                    and item.frame.predicate == relation.predicate_key
                    and item.source_ref == structure.source_ref
                    and item.values == structure.token_values
                    and any(role_binding.role == frame.answer_role_key
                            for role_binding in item.bindings)))
            dynamic_bindings = tuple(
                role_binding for item in full_candidates
                for role_binding in item.bindings
                if (role_binding.role == frame.answer_role_key
                    and role_binding.source_ref == structure.source_ref))
            dynamic_span_candidates = tuple(
                span.ref_key for span in spans.values()
                if any((span.start == item.start
                        and span.end == item.end
                        and span.values == item.values
                        and span.source_ref == structure.source_ref)
                       for item in dynamic_bindings))
            if len(full_candidates) != 1 or len(dynamic_bindings) != 1:
                return None
            reference_dynamic_span_refs = tuple(sorted(set(
                dynamic_span_candidates)))
            if len(reference_dynamic_span_refs) != 1:
                return None
        expected_relation_span_refs = tuple(sorted((
            *known_refs,
            *( () if predicate_span is None else (predicate_span,) ),
            *reference_dynamic_span_refs,
        )))
        if expected_relation_span_refs != relation.span_refs:
            return None
        if predicate_span is not None and predicate_span not in relation.span_refs:
            return None
        return (frame, relation, fact,
                fact_fillers[binding.answer_role_key], tuple(dynamic_roles))

    def _dynamic_slot_reference_blocked(
            self,
            structure: QueryInputStructure,
            *,
            frontier_exhausted: bool,
            ) -> bool:
        """Keep an anaphoric WH subject closed only after Memory resolves it.

        ``_relation_slot_question_parts`` deliberately validates the integer
        relation shape without depending on mutable QueryState.  The actual
        reference status is available only after Memory O/H/E refreshes during
        expansion, so this gate belongs at the shared generation/termination
        boundary.  A normal WH answer slot has no dynamic role and is never
        blocked here.
        """
        dynamic_roles = tuple(getattr(
            self, "_dynamic_slot_question_roles", ()))
        if not dynamic_roles:
            return False
        if len(dynamic_roles) != 1:
            return True
        if (self._input_observation_ref is None
                or self._reference_runtime is None):
            return True
        eligible = tuple(
            item for item in self._reference_candidate_sets
            if item.reference.role.stable_key() in dynamic_roles
            and self._reference_relation_context(item, structure)[0]
            is not None)
        # No preceding discourse, an unmatched role, or competing role
        # contexts cannot authorize a Core filler for the observed pronoun.
        if len(eligible) != 1:
            return True
        if not frontier_exhausted:
            return False
        required = tuple(item for item in self._reference_resolutions
                         if item.requires_resolution)
        if (len(required) != 1
                or any(item.status != REFERENCE_RESOLVED
                       for item in required)):
            return True
        resolution = required[0]
        if (resolution.candidate_set.reference.role.stable_key()
                not in dynamic_roles
                or resolution.winner_target_key is None):
            return True
        binding = next((item for item in structure.relation_question_bindings
                        if item.answer_role_key), None)
        if binding is None:
            return True
        relation = next((item for item in structure.relation_candidates
                         if item.stable_key() == binding.relation_key), None)
        fact = (None if relation is None else
                self.core_fact_index.get(relation.proposition_key))
        if fact is None:
            return True
        fillers = {
            item.role.stable_key(): item.filler.stable_key()
            for item in fact.bindings}
        role = resolution.candidate_set.reference.role.stable_key()
        return resolution.winner_target_key != fillers.get(role)

    def _memory_grounded_slot_binding(
            self, structure: QueryInputStructure,
            ) -> InputRelationQuestionBinding | None:
        """Select one dynamic WH proposition by exact delivered-role/Core identity."""
        if (not structure.relation_question_bindings
                or self._input_observation_ref is None
                or not self._reference_candidate_sets
                or self._reference_runtime is None):
            return None
        relation_by_key = {
            item.stable_key(): item for item in structure.relation_candidates}
        selected = []
        grounded_sources = []
        for binding in structure.relation_question_bindings:
            if not binding.answer_role_key:
                continue
            parts = self._relation_slot_question_parts(structure, binding)
            if parts is None:
                continue
            frame, relation, fact, _answer_filler, dynamic_roles = parts
            if len(dynamic_roles) != 1:
                continue
            dynamic_role = dynamic_roles[0]
            dynamic_span = None
            dynamic_span_values = ()
            if frame.is_reference_slot and dynamic_role not in frame.roles:
                # The reference slot omits its answer role from the trained
                # question frame, so the pronoun has no slot_span_ref.  Its
                # exact interval is carried by the restored full open-role
                # candidate and is recovered below for each candidate set.
                pass
            else:
                span_by_ref = {item.ref_key: item for item in structure.spans}
                dynamic_spans = tuple(
                    span_by_ref[span_ref]
                    for role, span_ref in zip(frame.roles, binding.slot_span_refs)
                    if role == dynamic_role and span_ref in span_by_ref)
                if len(dynamic_spans) != 1:
                    continue
                dynamic_span = dynamic_spans[0]
                dynamic_span_values = (dynamic_span.values,)
            relation_candidate = relation_by_key.get(binding.relation_key)
            if relation_candidate is None:
                continue
            input_candidates = []
            for candidate_set in self._reference_candidate_sets:
                reference = candidate_set.reference
                # The reference occurrence role comes from the generic
                # open-role graph, while ``dynamic_role`` is the authored
                # Core role identity carried by the relation-question frame.
                # They are distinct graph namespaces.  The exact integer
                # span is the shared witness between them; comparing the
                # role keys here would discard every valid cross-turn slot.
                if (dynamic_span is not None
                        and reference.values != dynamic_span_values[0]):
                    continue
                open_candidate = self._reference_relation_context(
                    candidate_set, structure)[0]
                if (open_candidate is None
                        or open_candidate.frame.proposition
                        != relation.proposition_key
                        or open_candidate.frame.predicate != frame.predicate_key
                        or relation_candidate.proposition_key
                        != relation.proposition_key):
                    continue
                candidate_dynamic_span = dynamic_span
                if candidate_dynamic_span is None:
                    reference_spans = tuple(
                        item for item in open_candidate.bindings
                        if (item.role == dynamic_role
                            and item.ordinal == reference.ordinal
                            and item.values == reference.values
                            and item.source_ref == structure.source_ref))
                    if len(reference_spans) != 1:
                        continue
                    candidate_dynamic_span = reference_spans[0]
                input_candidates.append((candidate_set, open_candidate,
                                         candidate_dynamic_span))
            if not input_candidates:
                continue
            # One input role can be materialized more than once when equivalent
            # trained frames or delivery proofs project the same integer
            # proposition.  Those wrappers are one semantic candidate only
            # when every antecedent resolves to the same graph object set.
            # Keep distinct object sets as a real ambiguity and fail closed.
            candidate_groups = {}
            for candidate_set, open_candidate, candidate_dynamic_span in input_candidates:
                antecedent_objects = frozenset(
                    identity.stable_key()
                    for antecedent in candidate_set.antecedents
                    for identity in self._reference_runtime.semantic_objects_for(
                        antecedent, self._response_contexts))
                group_key = (
                    binding.operation_key,
                    fact.proposition.stable_key(),
                    fact.predicate.stable_key(),
                    fact.source_hash,
                    binding.answer_role_key,
                    dynamic_role,
                    tuple(sorted(antecedent_objects)),
                    tuple(sorted(item.stable_key() for item in candidate_set.antecedents)),
                )
                candidate_groups.setdefault(
                    group_key, (candidate_set, candidate_dynamic_span))
            if len(candidate_groups) != 1:
                continue
            input_candidate, selected_dynamic_span = next(
                iter(candidate_groups.values()))
            target_filler_keys = {
                item.filler.stable_key()
                for item in fact.bindings
                if item.role.stable_key() == dynamic_role
            }
            if len(target_filler_keys) != 1:
                continue
            antecedent_object_keys = {
                identity.stable_key()
                for antecedent in input_candidate.antecedents
                if (frame.is_reference_slot or getattr(self, '_current_reference_scope', None) is None
                    or self._current_reference_scope.permits(
                        self._reference_runtime._context_for(antecedent, self._response_contexts)))
                for identity in self._reference_runtime.semantic_objects_for(
                    antecedent, self._response_contexts)
            }
            if target_filler_keys & antecedent_object_keys:
                semantic_key = (
                    binding.operation_key,
                    fact.proposition.stable_key(),
                    fact.predicate.stable_key(),
                    fact.source_hash,
                    binding.answer_role_key,
                    dynamic_role,
                    next(iter(target_filler_keys)),
                    tuple(sorted(antecedent_object_keys)),
                )
                selected.append((semantic_key,
                                 (selected_dynamic_span.start,
                                  selected_dynamic_span.end),
                                 binding))
                if frame.is_reference_slot:
                    matching_antecedents = tuple(
                        item for item in input_candidate.antecedents
                        if target_filler_keys & {
                            identity.stable_key() for identity in
                            self._reference_runtime.semantic_objects_for(item, self._response_contexts)})
                    if matching_antecedents:
                        grounded_sources.append(grounded_reference_source(
                            structure, binding, parts, selected_dynamic_span, input_candidate,
                            tuple(sorted({item.stable_key() for item in matching_antecedents}))))
        # Ignore parser envelopes and dynamic span coordinates while proving
        # semantic identity.  A distinct Core/antecedent group remains a real
        # ambiguity even when one of its spans is nested inside another.
        semantic_groups = {}
        for semantic_key, interval, binding in selected:
            semantic_groups.setdefault(semantic_key, []).append(
                (interval, binding))
        if len(semantic_groups) != 1:
            if len(grounded_sources) == len(selected):
                consensus = reference_source_consensus(grounded_sources)
                if consensus is not None:
                    self._memory_grounded_reference_source_consensus = consensus
                    return next(binding for _key, _span, binding in selected
                                if binding.stable_key() == consensus.selected_binding_key)
            return None
        span_bindings = {}
        for interval, binding in next(iter(semantic_groups.values())):
            prior = span_bindings.get(interval)
            if prior is None or binding.stable_key() < prior.stable_key():
                span_bindings[interval] = binding
        intervals = tuple(span_bindings)
        innermost = tuple(
            interval for interval in intervals
            if not any(
                other != interval
                and interval[0] <= other[0] and other[1] <= interval[1]
                and (interval[0], interval[1]) != (other[0], other[1])
                for other in intervals))
        if len(innermost) != 1:
            return None
        return span_bindings[innermost[0]]

    def _relation_slot_question_view(
            self, structure: QueryInputStructure,
            binding,
            ) -> QueryInputStructure | None:
        """Retain one open WH scope while limiting Core anchors to its slots."""
        parts = self._relation_slot_question_parts(structure, binding)
        if parts is None:
            return None
        frame, relation, _fact, _answer_filler, dynamic_roles = parts
        open_frame = OpenRoleFrame(
            relation.proposition_key, frame.predicate_key,
            relation.source_ref, frame.roles, frame.gaps,
            (0, len(frame.roles) + sum(map(len, frame.gaps))),
        )
        parses = match_open_roles(
            (open_frame,), structure.token_values, structure.source_ref)
        spans_by_interval = {
            (item.start, item.end): item.ref_key for item in structure.spans
        }
        matching_parses = tuple(
            parsed for parsed in parses
            if tuple(spans_by_interval.get((item.start, item.end))
                     for item in parsed.bindings) == binding.slot_span_refs)
        if len(matching_parses) != 1:
            return None
        semantic = tuple(sorted((
            item for item in structure.semantic_candidates
            if item.proposition_key == relation.proposition_key
            and item.source_ref == relation.source_ref
            and item.span_ref in relation.span_refs
        ), key=lambda item: item.stable_key()))
        carriers = tuple(sorted((
            item for item in structure.carrier_candidates
            if item.structure_key == relation.structure_key
            and item.source_ref == relation.source_ref
        ), key=lambda item: item.stable_key()))
        dynamic_open = ()
        if frame.is_reference_slot:
            reference_open = reference_question_open_candidate(
                structure, frame, relation, binding)
            if reference_open is None:
                return None
            dynamic_open = (reference_open,)
        elif dynamic_roles and relation.state == STRUCTURE_OPEN:
            dynamic_open = _matching_open_relation_candidates(
                structure, frame, relation.proposition_key,
                relation.source_ref, binding.slot_span_refs)
            if len(dynamic_open) != 1:
                return None
        return replace(
            structure,
            operation_keys=tuple(sorted({
                *structure.operation_keys, binding.operation_key,
            })),
            carrier_candidates=carriers,
            semantic_candidates=semantic,
            relation_candidates=(relation,),
            relation_question_bindings=(binding,),
            open_relations=dynamic_open,
            discourse_compositions=(),
            discourse_gaps=(),
        )

    def _relation_question_memory_view(
            self, structure: QueryInputStructure,
            ) -> QueryInputStructure:
        """逐项交接指代完整角色，多个来源保留竞争，不提前绑定实体。"""
        bindings = structure.relation_question_bindings
        if (not bindings or structure.discourse_compositions
                or any(not item.answer_role_key for item in bindings)):
            return structure
        if len(bindings) > 1:
            parts = tuple(self._relation_slot_question_parts(structure, item)
                          for item in bindings)
            if any(item is None or not item[0].is_reference_slot for item in parts):
                return structure
        views = tuple(self._relation_slot_question_view(structure, item)
                      for item in bindings)
        if any(item is None for item in views):
            raise ValueError("WH memory view lost its exact Core scope")
        return (views[0] if len(views) == 1
                else merge_reference_question_views(structure, views))

    def _attach_relation_question_binding(
            self, structure: QueryInputStructure,
            ) -> QueryInputStructure:
        if structure.assertion_scope_bindings or structure.operation_request_bindings:
            return structure
        if not self.relation_question_frames:
            if (structure.relation_question_bindings
                    or structure.pending_relation_composition_bindings):
                raise ValueError("question binding has no trained graph")
            return structure
        if not structure.tokens:
            if (structure.relation_question_bindings
                    or structure.pending_relation_composition_bindings):
                raise ValueError("graph-only input cannot carry question slots")
            return structure
        # Some trained question frames express the predicate only inside a
        # learned gap (for example a natural ``先于`` surface).  The generic
        # projector therefore sees both fillers but no predicate span and
        # leaves the relation OPEN.  Close that predicate slot only when one
        # complete train frame, one proposition and all observed role spans
        # agree exactly; the frame is syntax authority here, never a fact or
        # a lexical fallback.
        structure = self._close_relation_question_predicate_scope(structure)
        if structure.discourse_compositions:
            matches = match_scoped_relation_question_frames(
                self.relation_question_frames, structure,
                frozenset(item.operation_key
                          for item in self.input_discourse_conjunctions))
        else:
            complete = match_relation_question_frames(
                self.relation_question_frames, structure)
            slot = match_relation_slot_question_frames(
                self.relation_question_frames, structure,
                allow_dynamic_alternatives=True,
                allow_exact_alternatives=True)
            if len(slot) > 1:
                validated = tuple(
                    (item, self._relation_slot_question_parts(structure, item))
                    for item in slot)
                if all(parts is not None and not parts[4]
                       for _item, parts in validated):
                    slot = select_equivalent_slot_question_bindings(validated)
                elif any(parts is None or not parts[4]
                         for _item, parts in validated):
                    slot = ()
            slot = tuple(item for item in slot
                         if self._relation_slot_question_view(
                             structure, item) is not None)
            matches = tuple(sorted(set((*complete, *slot)),
                                   key=lambda item: item.stable_key()))
        # ``prepare_input_structure`` materializes a reference-slot binding
        # together with its full-role open candidate so Memory can observe the
        # omitted role.  Re-running the generic slot matcher on that already
        # projected view cannot reconstruct the authored frame (the current
        # candidate intentionally has one extra role), so it returns no match.
        # Reuse the binding only after the exact current integer/Core scope is
        # validated by the same slot validator; malformed or drifted bindings
        # still take the strict mismatch path below.
        if structure.relation_question_bindings and not matches:
            retained = tuple(self._relation_slot_question_parts(structure, item)
                             for item in structure.relation_question_bindings)
            if (all(item is not None for item in retained)
                    and (len(retained) == 1
                         or all(item[0].is_reference_slot for item in retained))
                    and all(self._relation_slot_question_view(structure, item) is not None
                            for item in structure.relation_question_bindings)):
                matches = structure.relation_question_bindings
        if structure.relation_question_bindings and (
                structure.relation_question_bindings != matches):
            raise ValueError("question binding does not match the current graph input")
        pending = (() if matches else match_pending_relation_questions(
            self.relation_question_frames, structure, self.pending_relation_compositions))
        if (structure.pending_relation_composition_bindings
                and structure.pending_relation_composition_bindings != pending):
            raise ValueError("pending relation query is not authorized by the current graph input")
        if pending:
            return replace(structure,
                operation_keys=tuple(sorted(set(structure.operation_keys) | {pending[0].operation_key})),
                pending_relation_composition_bindings=pending)
        if not matches:
            if any(self.input_discourse_operation_modes[key]
                   == MODE_RELATION_PAIR_QUERY for key in structure.operation_keys):
                # A licensed pair operation is still an integer input contract
                # when the current surface has no exact trained graph binding.
                # Keep the contract available for the query trace and let the
                # shared QueryState fail closed; raising here used to crash the
                # JSONL boundary before it could return an auditable refusal.
                return structure
            if (any(item.operation_key in {join.operation_key
                                           for join in self.input_discourse_conjunctions}
                    for item in structure.discourse_compositions)
                    and not conjunction_has_complete_literal_scopes(
                        structure, frozenset(item.operation_key
                            for item in self.input_discourse_conjunctions))):
                return replace(structure, discourse_compositions=tuple(sorted((
                    replace(item, operand_states=(STRUCTURE_OPEN, STRUCTURE_OPEN),
                            state=STRUCTURE_OPEN)
                    for item in structure.discourse_compositions),
                    key=lambda item: item.stable_key())))
            return structure
        question_operations = {item.operation_key for item in matches}
        if len(question_operations) != 1:
            raise ValueError("question scopes have conflicting operation contracts")
        operation_keys = tuple(sorted(
            set(structure.operation_keys) | question_operations))
        if (structure.operation_keys and not structure.discourse_compositions
                and structure.operation_keys != operation_keys):
            return structure
        if (structure.relation_question_bindings == matches
                and structure.operation_keys == operation_keys):
            return self._relation_question_memory_view(structure)
        prepared = replace(
            structure, operation_keys=operation_keys,
            relation_question_bindings=matches)
        return self._relation_question_memory_view(prepared)

    def _close_relation_question_predicate_scope(
            self, structure: QueryInputStructure,
            ) -> QueryInputStructure:
        """Close a predicate proved by one exact complete train question.

        The input projector can only mark a relation closed when a trained
        predicate span is present.  A complete relation-question frame is a
        second graph witness: its learned gaps may carry the predicate while
        the current input contains only exact filler spans.  Promote only a
        unique frame shape whose role spans are already represented by one
        semantic filler each.  Unknown, reversed, or competing frames remain
        open and continue through the fail-closed path.
        """
        if (not self.relation_question_frames or not structure.open_relations
                or not structure.relation_candidates
                or structure.relation_question_bindings):
            return structure
        complete_shapes = {
            (item.predicate_key, item.roles, item.gaps)
            for item in self.relation_question_frames
            if not item.is_slot_question
        }
        slot_shapes = {
            (item.predicate_key, item.roles, item.gaps)
            for item in available_slot_question_frames(
                self.relation_question_frames, structure.token_values)
        }
        # Keep every frame sharing a shape only as one syntax contract.  The
        # stable question source is provenance, while predicate/role/gap
        # shape is the input binding contract.
        promoted = {}
        spans_by_interval = {
            (span.start, span.end, span.values, span.source_ref): span.ref_key
            for span in structure.spans
        }
        for relation in structure.relation_candidates:
            if (relation.state != STRUCTURE_OPEN
                    or relation.predicate_coverage != 0
                    or relation.required_count not in {
                        relation.support_count + 1,
                        relation.support_count + 2,
                    }):
                continue
            candidates = []
            for open_relation in structure.open_relations:
                frame = open_relation.frame
                if (frame.proposition != relation.proposition_key
                        or frame.predicate != relation.predicate_key
                        or frame.source_ref != relation.source_ref
                        or len(frame.roles) != relation.role_coverage
                        or tuple(sorted(
                            spans_by_interval.get((binding.start, binding.end,
                                                    binding.values,
                                                    binding.source_ref))
                            for binding in open_relation.bindings
                            if spans_by_interval.get((binding.start, binding.end,
                                                       binding.values,
                                                       binding.source_ref))
                        )) != relation.span_refs):
                    continue
                shape = (frame.predicate, frame.roles, frame.gaps)
                if shape not in complete_shapes and shape not in slot_shapes:
                    continue
                role_refs = tuple(
                    spans_by_interval.get((binding.start, binding.end,
                                           binding.values, binding.source_ref))
                    for binding in open_relation.bindings)
                if (len(role_refs) != len(frame.roles)
                        or any(ref is None for ref in role_refs)
                        or len(set(role_refs)) != len(role_refs)):
                    continue
                valid = True
                for role, span_ref in zip(frame.roles, role_refs):
                    members = tuple(item for item in structure.semantic_candidates
                                    if (item.projection_kind == PROJECTION_FILLER
                                        and item.proposition_key
                                        == relation.proposition_key
                                        and item.predicate_key
                                        == relation.predicate_key
                                        and item.source_ref == relation.source_ref
                                        and item.role_key == role
                                        and item.span_ref == span_ref
                                        and item.span_ref in relation.span_refs))
                    if len(members) != 1:
                        valid = False
                        break
                if valid:
                    candidates.append((frame.predicate, frame.roles, frame.gaps))
            shapes = tuple(sorted(set(candidates)))
            if len(shapes) != 1:
                continue
            shape = shapes[0]
            if shape in slot_shapes:
                # A slot frame proves the omitted predicate, while its answer
                # endpoint remains open.  Retain that one open slot and do
                # not manufacture a predicate semantic span: the frame gap is
                # the only predicate witness in this input.
                support_count = relation.role_coverage + 1
                if support_count > relation.required_count:
                    continue
                promoted[relation.stable_key()] = replace(
                    relation,
                    support_count=support_count,
                    required_open=relation.required_count - support_count,
                    predicate_coverage=1,
                    state=STRUCTURE_OPEN,
                )
            else:
                promoted[relation.stable_key()] = replace(
                    relation,
                    support_count=relation.required_count,
                    required_open=0,
                    predicate_coverage=1,
                    state=STRUCTURE_CLOSED,
                )
        if not promoted:
            return structure
        return replace(
            structure,
            relation_candidates=tuple(sorted(
                (promoted.get(item.stable_key(), item)
                 for item in structure.relation_candidates),
                key=lambda item: item.stable_key())),
        )

    def _filter_relation_question_open_scope(
            self, structure: QueryInputStructure,
            ) -> QueryInputStructure:
        """Keep only train-question open parses with exact observed fillers.

        Question syntax is shared across the trained relation rows, so a
        binary frame can initially match every proposition with the same role
        order.  A dynamic follow-up role is allowed to remain open, while
        every observed role must match one exact current Core filler.  This
        narrows the open-role Memory entry before input roles are materialized
        and prevents unrelated facts from becoming reference candidates.
        """
        if (structure.assertion_scope_bindings or not self._relation_question_open_frames
                or not structure.open_relations
                or structure.relation_question_bindings):
            return structure
        if (self.pending_relation_compositions
                and match_pending_relation_questions(
                    self.relation_question_frames, structure,
                    self.pending_relation_compositions)):
            # The pending composition binding authorizes these exact input
            # slots.  Preserve their full Core premise projections for the
            # proof query, but do not turn them into ordinary open-role
            # Memory candidates.
            return replace(structure, open_relations=())
        # A conjunction has its own scope-local authority.  Keep the full
        # relation/semantic pool for question-frame binding, but narrow each
        # operand group to the unique candidate closed inside that operand.
        # Filtering the global pool here can remove the second clause before
        # ``match_scoped_relation_question_frames`` sees it.
        if structure.discourse_compositions:
            operations = frozenset(
                item.operation_key for item in self.input_discourse_conjunctions)
            selected = scope_local_conjunction_candidate_keys(
                structure, operations)
            if len(selected) != len(structure.discourse_compositions):
                return structure
            compositions = []
            for composition, (_composition_key, relation_keys) in zip(
                    structure.discourse_compositions, selected):
                groups = []
                for group, relation_key in zip(
                        composition.operand_relation_keys, relation_keys):
                    if relation_key not in group:
                        return structure
                    groups.append((relation_key,))
                compositions.append(replace(
                    composition, operand_relation_keys=tuple(groups)))
            return replace(
                structure,
                discourse_compositions=tuple(sorted(
                    compositions, key=lambda item: item.stable_key())))
        if structure.discourse_gaps:
            return structure
        spans = {
            (item.start, item.end, item.values, item.source_ref): item.ref_key
            for item in structure.spans
        }
        facts = self.core_fact_index
        question_frames_by_shape = {}
        for item in available_slot_question_frames(
                self.relation_question_frames, structure.token_values):
            question_frames_by_shape.setdefault(
                (item.predicate_key, item.roles, item.gaps), []).append(item)
        selected = []
        selected_props = set()
        for candidate in structure.open_relations:
            fact = facts.get(candidate.frame.proposition)
            if fact is None or candidate.frame.source_ref != (fact.source_hash,):
                continue
            fact_fillers = {
                item.role.stable_key(): item.filler.stable_key()
                for item in fact.bindings
            }
            if len(fact_fillers) != len(fact.bindings):
                continue
            question_frames = tuple(question_frames_by_shape.get((
                candidate.frame.predicate, candidate.frame.roles,
                candidate.frame.gaps), ()))
            if not question_frames:
                role_contract = (set(fact_fillers) == set(candidate.frame.roles)
                                 and len(fact_fillers) == len(candidate.frame.roles))
            else:
                role_contract = any(
                    question_frame.answer_role_key in fact_fillers
                    and question_observed_role_contract(question_frame, tuple(
                        item.role.stable_key() for item in fact.bindings))
                    and set(candidate.frame.roles) == set(question_frame.roles)
                    for question_frame in question_frames)
            if not role_contract:
                continue
            observed = 0
            dynamic_roles = []
            valid = True
            for binding in candidate.bindings:
                span_ref = spans.get((binding.start, binding.end,
                                      binding.values, candidate.source_ref))
                if span_ref is None:
                    dynamic_roles.append(binding.role)
                    continue
                matches = tuple(item for item in structure.semantic_candidates
                                if item.projection_kind == PROJECTION_FILLER
                                and item.proposition_key == candidate.frame.proposition
                                and item.predicate_key == candidate.frame.predicate
                                and item.role_key == binding.role
                                and item.span_ref == span_ref)
                if not matches:
                    dynamic_roles.append(binding.role)
                    continue
                if (len(matches) != 1
                        or matches[0].candidate_key != fact_fillers[binding.role]):
                    valid = False
                    break
                observed += 1
            if (valid and (observed or (question_frames
                                        and len(dynamic_roles) == 1))):
                selected.append(candidate)
                selected_props.add(candidate.frame.proposition)
        from pure_integer_ai.cognition.understanding.query_question_syntax_scope import grounded_question_open_scope
        selected = grounded_question_open_scope(selected, structure,
            question_frames_by_shape, self.relation_question_frames)
        selected_props = {candidate.frame.proposition for candidate in selected}
        if not selected:
            return replace(structure, open_relations=())
        # Keep the exact proposition scope for the subsequent Memory restore;
        # unrelated surface aliases cannot create extra cross-turn candidates.
        semantic = tuple(item for item in structure.semantic_candidates
                         if item.proposition_key in selected_props)
        relations = tuple(item for item in structure.relation_candidates
                          if item.proposition_key in selected_props)
        # The projector can expose the same semantic open slot through two
        # trained envelopes: the relation-question frame and the original
        # Core open-role frame.  Their envelope coordinates are provenance for
        # parsing, not separate hypotheses.  Collapse only when every
        # proposition/predicate/role/gap/source/value/binding field agrees,
        # preferring the question frame so the Memory candidate retains the
        # current REFERS scope rather than the duplicate Core envelope.
        question_frames = set((*self._relation_question_open_frames,
            *composed_slot_open_role_frames(self.relation_question_frames,
                self._relation_question_open_frames, structure.token_values)))
        semantic_open: dict[tuple[object, ...], OpenRelationCandidate] = {}
        for candidate in selected:
            semantic_key = (
                candidate.frame.proposition,
                candidate.frame.predicate,
                candidate.frame.roles,
                candidate.frame.gaps,
                candidate.frame.source_ref,
                candidate.source_ref,
                candidate.values,
                tuple(item.stable_key() for item in candidate.bindings),
            )
            prior = semantic_open.get(semantic_key)
            if (prior is None
                    or ((candidate.frame not in question_frames),
                        candidate.stable_key())
                    < ((prior.frame not in question_frames),
                       prior.stable_key())):
                semantic_open[semantic_key] = candidate
        retained_relation_keys = {
            item.stable_key() for item in relations}
        conjunction_operations = frozenset(
            item.operation_key for item in self.input_discourse_conjunctions)
        scope_bindings = scope_local_conjunction_candidate_keys(
            structure, conjunction_operations)
        scope_relation_by_ref = {}
        if len(scope_bindings) == len(structure.discourse_compositions):
            for composition, (_composition_key, relation_keys) in zip(
                    structure.discourse_compositions, scope_bindings):
                for scope_ref, relation_key in zip(
                        composition.operand_scope_refs, relation_keys):
                    scope_relation_by_ref[scope_ref] = relation_key
        # The discourse projector builds composition/gap references before
        # the question-specific scope filter runs. Keep those references
        # closed over the same retained relation candidates; otherwise the
        # narrowed structure names candidates that were deliberately removed
        # and QueryInputStructure rejects it before query routing.
        compositions = tuple(sorted((
            replace(item, operand_relation_keys=tuple(
                tuple(key for key in group if key in retained_relation_keys)
                if item.operand_scope_refs[index] not in scope_relation_by_ref
                or scope_relation_by_ref[item.operand_scope_refs[index]]
                not in retained_relation_keys
                else tuple(key for key in group if key == scope_relation_by_ref[
                    item.operand_scope_refs[index]])
                for index, group in enumerate(item.operand_relation_keys)))
            for item in structure.discourse_compositions),
            key=lambda item: item.stable_key()))
        gaps = tuple(sorted((
            replace(item,
                    left_relation_keys=tuple(
                        key for key in item.left_relation_keys
                        if key in retained_relation_keys),
                    right_relation_keys=tuple(
                        key for key in item.right_relation_keys
                        if key in retained_relation_keys))
            for item in structure.discourse_gaps),
            key=lambda item: item.stable_key()))
        return replace(
            structure,
            semantic_candidates=tuple(sorted(semantic,
                                             key=lambda item: item.stable_key())),
            relation_candidates=tuple(sorted(relations,
                                             key=lambda item: item.stable_key())),
            open_relations=tuple(sorted(semantic_open.values(),
                                        key=lambda item: item.stable_key())),
            discourse_compositions=compositions,
            discourse_gaps=gaps,
        )

    def _attach_operation_request_binding(self, structure):
        """独立请求变量只来自冷训练图；完整 request 只保留 reader scope。"""
        from pure_integer_ai.cognition.understanding.query_operation_requests import attach_operation_request
        prepared = attach_operation_request(self.operation_request_frames, structure)
        # A request frame is already a complete input contract.  Apply its
        # reader-only view even when no output-obligation graph is mounted;
        # otherwise ordinary open-role projections from the same surface can
        # leave unrelated frontier work and block the operation proof gate.
        from pure_integer_ai.cognition.understanding.query_operation_requests import operation_request_input_view
        from pure_integer_ai.cognition.understanding.query_operation_request_scope import pending_operation_request_input_view
        return pending_operation_request_input_view(
            self.operation_request_frames, operation_request_input_view(prepared))

    def _source_provenance_for_input(
            self, structure: QueryInputStructure,
            ) -> dict[tuple[int, ...], tuple[int, ...]]:
        """Recover integer-only proof keys for training sources in one input.

        A source hash alone identifies a physical registry row.  The release
        graph can contain the same authored assertion more than once under
        different physical ``source_id`` values, so duplicate aggregation is
        allowed only after the remaining source metadata agrees.  Missing or
        malformed records receive an isolated proof key and therefore remain
        fail-closed.
        """
        refs = {
            item.source_ref for item in structure.semantic_candidates
        }
        refs.update(item.source_ref for item in structure.relation_candidates)
        refs.update(item.source_ref for item in structure.carrier_candidates)
        refs.update(
            item.frame.source_ref for item in structure.open_relations)
        result: dict[tuple[int, ...], tuple[int, ...]] = {}
        for source_ref in sorted(refs):
            if len(source_ref) != 1:
                result[source_ref] = (0, *source_ref)
                continue
            source_hash = source_ref[0]
            try:
                record = self._source_records.read(source_hash)
                proof = (
                    1,
                    record.text_hash,
                    record.codepoint_count,
                    record.source_kind,
                    record.document_id,
                    record.corpus_version,
                    record.parser_version,
                    record.batch_id,
                    record.companion_type_hash,
                    record.companion_name_hash,
                    record.companion_assoc_id,
                    _QUERY_SOURCE_PROVENANCE_HASHER.h63(
                        record.license_id),
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                proof = (0, *source_ref)
            result[source_ref] = proof
        return result

    def _aggregate_input_provenance(
            self, structure: QueryInputStructure) -> QueryInputStructure:
        """Apply source-proof-aware aggregation at every query entry point."""
        return _aggregate_input_provenance_duplicates(
            structure,
            source_provenance=self._source_provenance_for_input(structure),
        )

    @observed_relation_phase("materialization")
    def prepare_input_structure(
            self,
            surface: str | None,
            *,
            graph_object_keys: tuple[tuple[int, ...], ...] = (),
            operation_keys: tuple[tuple[int, ...], ...] = (),
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
            return self._attach_operation_keys(
                graph_query_input_structure(graph_object_keys), operation_keys)
        if type(surface) is not str or not surface:
            raise ValueError("文字输入必须是非空字符串")
        structure = self._augment_kdconv_input(
            self.input_projector.project(tuple(ord(value) for value in surface)))
        structure = self._aggregate_input_provenance(structure)
        structure = attach_assertion_scope_frames(self.assertion_scope_frames, structure)
        # Complete trained requests own their exact reader scope before a
        # local WH parse can discard readers belonging to another predicate.
        structure = self._attach_operation_request_binding(structure)
        structure = self._filter_relation_question_open_scope(structure)
        from pure_integer_ai.cognition.understanding.query_anaphoric_slot_scope import attach_anaphoric_slot_syntax
        structure = attach_anaphoric_slot_syntax(
            structure, self.relation_question_frames, self.input_projector.structures)
        structure = self._attach_relation_question_binding(structure)
        structure = self._attach_operation_request_binding(structure)
        if graph_object_keys:
            structure = replace(
                structure, graph_object_keys=graph_object_keys)
        return self._attach_operation_keys(structure, operation_keys)

    def active_memory_input_structure(
            self, input_structure: QueryInputStructure,
            ) -> QueryInputStructure:
        """Return the scope-authorized relation view used for Memory intake."""
        if not isinstance(input_structure, QueryInputStructure):
            raise TypeError("input_structure 必须是 QueryInputStructure")
        if input_structure.assertion_scope_bindings:
            return _selected_relation_view(input_structure,
                input_structure.assertion_scope_bindings[0].relation_key)[0]
        if (any(item.operation_key in {join.operation_key
                                       for join in self.input_discourse_conjunctions}
                for item in input_structure.discourse_compositions)
                and self.relation_question_frames
                and not input_structure.relation_question_bindings
                and not conjunction_has_complete_literal_scopes(
                    input_structure, frozenset(item.operation_key
                        for item in self.input_discourse_conjunctions))):
            return replace(input_structure, semantic_candidates=(),
                           relation_candidates=(), carrier_candidates=(),
                           discourse_compositions=(), discourse_gaps=(),
                           open_relations=())
        if (input_structure.relation_question_bindings
                and not input_structure.discourse_compositions):
            if (len(input_structure.relation_question_bindings) == 1
                    and not input_structure.relation_question_bindings[0].answer_role_key):
                return _selected_relation_view(input_structure,
                    input_structure.relation_question_bindings[0].relation_key)[0]
            return self._relation_question_memory_view(input_structure)
        if not (input_structure.discourse_compositions
                or input_structure.discourse_gaps):
            return input_structure
        fact_query_matches = scope_local_fact_query_candidate_keys(
            input_structure,
            self.input_discourse_fact_query_modes,
        )
        if len(fact_query_matches) == 1:
            composition_key, operation_key, relation_key = fact_query_matches[0]
            mode_key = self.input_discourse_fact_query_modes[operation_key]
            prepared, _ = _fact_query_view(
                input_structure,
                (composition_key, operation_key, relation_key, mode_key),
            )
            return prepared
        general_operation_matches = scope_local_general_operation_candidate_keys(
            input_structure,
            frozenset(
                key for key, mode in self.input_discourse_operation_modes.items()
                if mode == MODE_OPERATION_GENERAL),
        )
        if len(general_operation_matches) == 1:
            prepared, _ = _general_operation_view(
                input_structure, general_operation_matches[0])
            return prepared
        conjunction_by_operation = {
            item.operation_key: item
            for item in self.input_discourse_conjunctions
        }
        local = scope_local_conjunction_candidate_keys(
            input_structure, frozenset(conjunction_by_operation))
        selected = {
            key for _composition, relation_keys in local for key in relation_keys
        }
        if not selected:
            return replace(input_structure, discourse_compositions=(),
                           discourse_gaps=(), open_relations=())
        relations = tuple(sorted(
            (item for item in input_structure.relation_candidates
             if item.stable_key() in selected),
            key=lambda item: item.stable_key()))
        proposition_keys = {item.proposition_key for item in relations}
        span_refs = {ref for item in relations for ref in item.span_refs}
        semantic = tuple(sorted(
            (item for item in input_structure.semantic_candidates
             if item.proposition_key in proposition_keys
             and item.span_ref in span_refs),
            key=lambda item: item.stable_key()))
        return replace(
            input_structure,
            semantic_candidates=semantic,
            relation_candidates=relations,
            discourse_compositions=(),
            discourse_gaps=(),
            open_relations=(),
        )

    @observed_relation_phase("memory_recover")
    def prepare_natural_revision_binding(
            self,
            input_structure: QueryInputStructure,
            *,
            memory_input_structure: QueryInputStructure | None = None,
            ) -> NaturalRevisionInputBinding | None:
        """Bind one trained correction cue before Interaction Memory writes."""
        if self.natural_revision_runtime is None:
            return None
        return self.natural_revision_runtime.bind_input(
            input_structure,
            memory_input_structure=memory_input_structure,
        )

    @observed_relation_phase("memory_recover")
    def resolve_natural_revision_target(
            self,
            input_structure: QueryInputStructure,
            binding: NaturalRevisionInputBinding,
            ) -> NaturalRevisionTargetBinding:
        """Resolve exactly one preceding Memory target for a bound cue."""
        if self.natural_revision_runtime is None:
            raise ValueError("no trained revision cue graph is mounted")
        if self.memory is None:
            raise ValueError("natural revision requires Interaction Memory")
        return self.natural_revision_runtime.resolve_target(
            self.memory, input_structure, binding)

    @observed_relation_phase("memory_commit")
    def commit_natural_revision(
            self,
            target: NaturalRevisionTargetBinding,
            source: SourceRef,
            ) -> NaturalRevisionCommit:
        """Append the current correction source as a superseding Evidence."""
        if self.natural_revision_runtime is None or self.memory is None:
            raise ValueError("no trained revision cue graph is mounted")
        return self.natural_revision_runtime.commit(
            self.memory, target, source)

    @observed_relation_phase("query")
    @_memory_event_read_scope
    def query(self, surface: str | None = None, *,
               artifact_input: ArtifactQueryInput | None = None,
               graph_object_keys: tuple[tuple[int, ...], ...] = (),
               minimum_depth: int = 1, max_depth: int = 4,
               max_nodes: int = 4096, max_edges: int = 8192,
               max_reads: int = 16384,
               input_observation_ref: MemoryObjectRef | None = None,
               compact_trace: bool = False,
               input_structure: QueryInputStructure | None = None,
               operation_keys: tuple[tuple[int, ...], ...] = (),
               ) -> dict[str, object]:
        """建立三图 roots 并执行一次共同 frontier 查询。"""
        if type(compact_trace) is not bool:
            raise TypeError("compact_trace 必须是 bool")
        if any(type(value) is not int or value <= 0
               for value in (max_nodes, max_edges, max_reads)):
            raise ValueError("query budgets must be positive strict integers")
        self._pending_response_delivery = None
        self._input_dialogue_act_match = None
        self._slot_question_core_roots: frozenset[tuple[int, ...]] = frozenset()
        self._partial_property_core_roots: frozenset[tuple[int, ...]] = frozenset()
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
        self._fact_query_operations = ()
        self._dynamic_slot_question_roles = ()
        self._dynamic_slot_query_blocked = False
        self._memory_grounded_reference_source_consensus = None
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
        if input_structure is not None and not operation_keys:
            operation_keys = input_structure.operation_keys
        operation_keys = self._normalize_operation_keys(operation_keys)
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
                if self.input_dialogue_act_graph is not None:
                    self._input_dialogue_act_match = (
                        self.input_dialogue_act_graph.match(surface_values))
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
            # 直接 query 与 prepare 后 query 在绑定问式前消费同一聚合视图。
            input_structure = self._aggregate_input_provenance(input_structure)
            input_structure = attach_assertion_scope_frames(self.assertion_scope_frames, input_structure)
            input_structure = self._attach_operation_request_binding(input_structure)
            input_structure = self._filter_relation_question_open_scope(input_structure)
            input_structure = self._attach_relation_question_binding(
                input_structure)
            input_structure = self._attach_operation_request_binding(input_structure)
            input_structure = self._attach_operation_keys(
                input_structure, operation_keys)
            query_key = input_structure.source_ref
        else:
            input_structure = artifact_input.query_input_structure(
                self.input_projector.structures)
            if graph_object_keys:
                input_structure = replace(
                    input_structure, graph_object_keys=graph_object_keys)
            input_structure = self._attach_operation_keys(
                input_structure, operation_keys)
            query_key = artifact_input.stable_key()
            (self._artifact_bridge_matches,
             self._artifact_bridge_missing) = self._match_artifact_semantic_bindings(
                 artifact_input)
        if artifact_input is not None:
            input_structure = self._augment_kdconv_input(input_structure)
        input_structure = self._aggregate_input_provenance(input_structure)
        # Callers may provide a projected structure directly instead of using
        # prepare_input_structure(). Apply the same train-question semantic
        # scope filter at the query boundary so both entry points materialize
        # one open Memory candidate for one logical slot.
        input_structure = self._filter_relation_question_open_scope(
            input_structure)
        # Preserve the exact predicate contract carried by a complete or WH
        # question parse.  A binary filler pair may belong to several Core
        # relations; when the requested relation is not uniquely closed, the
        # inverse fallback must not select a different predicate.
        syntax_predicates = relation_question_syntax_predicates(
            self.relation_question_frames, input_structure)
        if input_structure.assertion_scope_bindings:
            input_structure, _ = _selected_relation_view(input_structure,
                input_structure.assertion_scope_bindings[0].relation_key)
            syntax_predicates = (input_structure.relation_candidates[0].predicate_key,)
        # A complete current question binding is a stronger witness than the
        # set of all open frame parses.  Natural gaps can otherwise match
        # several predicate families that share the same filler spans; using
        # that broad set would erase the exact predicate contract and empty
        # the Core anchors.  Ambiguous bindings deliberately keep the broad
        # result and therefore remain fail-closed.
        if len(input_structure.relation_question_bindings) == 1:
            bound_relation = next((item for item in input_structure.relation_candidates
                                   if item.stable_key()
                                   == input_structure.relation_question_bindings[0].relation_key),
                                  None)
            if bound_relation is not None:
                syntax_predicates = (bound_relation.predicate_key,)
        question_predicate_keys = (
            None if not syntax_predicates
            else (frozenset(syntax_predicates)
                  if len(syntax_predicates) == 1 else frozenset()))
        # The exact predicate contract above remains strict for Core routing.
        # Input competition needs the full set of trained syntax predicates so
        # an ambiguous WH surface can still be rejected when its same-surface
        # endpoint identities compete; this set never authorizes an answer.
        input_competition_predicate_keys = (
            frozenset(syntax_predicates) if syntax_predicates else None)
        # KdConv can close an explicit subject in the current input before
        # Memory restore.  Predicate-only follow-ups need the preceding
        # factual ResponsePlan topic, which is an explicit Episode edge and
        # must be recovered before the shared roots/frontier are seeded.
        input_structure = self._restore_factual_topic_input(input_structure)
        # A mounted operation graph is a capability catalogue, not a global
        # switch for every query. Route into the program path only when this
        # exact current input has one operation key with a matching licensed
        # program; unrelated historical programs must not take over a REFERS
        # question sharing the same bridge.
        operation_program_keys = frozenset(
            witness.program.operation_key
            for witness in self.operation_program_witnesses)
        operation_program_route = (
            len(input_structure.operation_keys) == 1
            and input_structure.operation_keys[0] in operation_program_keys)
        if (len(input_structure.relation_question_bindings) == 1
                and not input_structure.discourse_compositions):
            binding = input_structure.relation_question_bindings[0]
            if binding.answer_role_key:
                view = self._relation_slot_question_view(input_structure, binding)
                if view is None:
                    raise ValueError("WH query view lost its exact Core scope")
                input_structure = view
            else:
                input_structure, _ = _selected_relation_view(
                    input_structure, binding.relation_key)
        fact_query_matches = scope_local_fact_query_candidate_keys(
            input_structure,
            self.input_discourse_fact_query_modes,
        )
        general_operation_matches = scope_local_general_operation_candidate_keys(
            input_structure,
            frozenset(
                key for key, mode in self.input_discourse_operation_modes.items()
                if mode == MODE_OPERATION_GENERAL),
        )
        # A mixed operation input may contain one authorized unary query over
        # a scope that is also joined by a distinct trained conjunction. Keep
        # the full conjunction structure for multi-claim evidence in that
        # case; a pure unary query still collapses to its selected relation.
        mixed_fact_query = False
        if len(fact_query_matches) == 1:
            conjunction_keys = frozenset(
                item.operation_key for item in self.input_discourse_conjunctions)
            mixed_fact_query = any(
                item.operation_key in conjunction_keys
                for item in input_structure.discourse_compositions
                if item.operation_key != fact_query_matches[0][1])
        if len(fact_query_matches) == 1:
            composition_key, operation_key, relation_key = fact_query_matches[0]
            mode_key = self.input_discourse_fact_query_modes[operation_key]
            self._fact_query_operations = ((
                composition_key, operation_key, mode_key, relation_key),)
            if not mixed_fact_query:
                input_structure, relation_key = _fact_query_view(
                    input_structure,
                    (composition_key, operation_key, relation_key, mode_key),
                )
        elif len(general_operation_matches) == 1:
            input_structure, _ = _general_operation_view(
                input_structure, general_operation_matches[0])
        composition_present = bool(
            input_structure.discourse_compositions
            or input_structure.discourse_gaps)
        gap_only = bool(
            input_structure.discourse_gaps
            and not input_structure.discourse_compositions)
        conjunction_by_operation = {
            item.operation_key: item
            for item in self.input_discourse_conjunctions
        }
        local_conjunctions = scope_local_conjunction_members(
            input_structure, frozenset(conjunction_by_operation))
        local_candidate_conjunctions = scope_local_conjunction_candidate_keys(
            input_structure, frozenset(conjunction_by_operation))
        composition_by_key = {
            item.stable_key(): item
            for item in input_structure.discourse_compositions
        }
        scope_conjunction_bindings = tuple(sorted(
            (composition_key,
             conjunction_by_operation[
                 composition_by_key[composition_key].operation_key].relation_key,
             members)
            for composition_key, members in local_conjunctions
            if composition_key in composition_by_key))
        input_structure, self._kdconv_query_propositions, active_relation_keys = (
            _composition_query_view(
                input_structure, scope_conjunction_bindings,
                self._kdconv_query_propositions,
                selected_relation_keys={
                    key for _composition, relation_keys
                    in local_candidate_conjunctions
                    for key in relation_keys
                }))
        active_candidates = {
            item.stable_key(): item
            for item in input_structure.relation_candidates
            if item.stable_key() in active_relation_keys
        }
        active_span_refs = {
            span_ref
            for item in active_candidates.values()
            for span_ref in item.span_refs
        }
        if composition_present:
            input_structure = replace(
                input_structure,
                semantic_candidates=tuple(
                    item for item in input_structure.semantic_candidates
                    if item.span_ref in active_span_refs),
                # Ambiguous open-role segmentations are retained in the
                # original structure for audit, but a verified conjunction
                # activates only its unique closed scope members.  Seeding
                # every open segmentation here recreates the frontier
                # explosion that the active composition view is meant to
                # prevent.
                open_relations=(),
            )
        self._active_relation_keys = frozenset(active_relation_keys)
        partial_property_matches = _partial_property_fact_matches(
            input_structure, tuple(self.core_fact_index.values()))
        self._partial_property_core_roots = frozenset(
            relation.proposition_key for relation, _fact in partial_property_matches)
        unbound_relation_question = _has_unbound_relation_question_candidate(
            input_structure, tuple((*self._relation_question_open_frames,
                *composed_slot_open_role_frames(self.relation_question_frames,
                    self._relation_question_open_frames, input_structure.token_values))))
        uncovered_closed_relation_modifier = (
            has_uncovered_closed_relation_modifier(input_structure))
        # A verified scope conjunction is the only composition route allowed
        # to seed Core relation roots.  Unverified/ambiguous compositions keep
        # the historical empty-anchor UNKNOWN path.
        concept_anchors = (
            self.anchor_concepts(
                input_structure,
                allowed_predicate_keys=question_predicate_keys)
            if (not composition_present or scope_conjunction_bindings or gap_only)
            else ())
        self._diagnostic_stage(
            f"concept_anchors:{len(concept_anchors)}:relations={len(input_structure.relation_candidates)}:"
            f"semantic={len(input_structure.semantic_candidates)}")
        # A multi-claim composition is authorized by its complete closed
        # scope join.  Ranking a single best/second proposition is not a
        # meaningful slot-closure signal for this route; keep the ordinary
        # best-candidate fields for traceability but let the conjunction
        # expander compute closure from all authorized members.
        composition_required_open = int(
            composition_present and not scope_conjunction_bindings)
        # The input projector can only close a relation when its trained
        # predicate span is present.  For an unseen inverse expression, use
        # the exact graph pair route above before ranking Dialogue roots.
        pair_inverse = (() if (composition_present and not gap_only)
                        or unbound_relation_question else
                        self._pair_inverse_candidates(
                            input_structure, self.facts,
                            allowed_predicate_keys=question_predicate_keys))
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
            concept_anchors = self.anchor_concepts(
                input_structure,
                allowed_predicate_keys=question_predicate_keys)
        # Filler projections without a complete trained relation are retained
        # in the input structure and Memory O/H/E, but they are not Core roots.
        # Otherwise every unmatched sentence expands dozens of one-slot
        # concept guesses until the shared budget is exhausted before the
        # generic Dialogue act can consume the same QueryState.
        slot_question_parts = None
        slot_question_binding = None
        if len(input_structure.relation_question_bindings) == 1:
            candidate_binding = input_structure.relation_question_bindings[0]
            if candidate_binding.answer_role_key:
                parts = self._relation_slot_question_parts(
                    input_structure, candidate_binding)
                if parts is not None:
                    slot_question_parts = parts
                    slot_question_binding = candidate_binding
                    self._dynamic_slot_question_roles = tuple(parts[4])
        # A relation-question frame may supply the predicate through its
        # integer operation binding rather than an observed predicate span.
        # Such a relation is structurally closed with ``predicate_coverage=0``
        # by design.  Keep its exact Core filler anchors alive; otherwise the
        # selected relation view has no open relations left and the generic
        # ``not complete_relation`` guard below erases every Core root before
        # the shared frontier can materialize positive evidence.  The binding
        # identity and full role/span closure remain mandatory, so an ordinary
        # open or unbound surface cannot use this route.
        question_bound_closed_relation = _has_closed_bound_relation_question(
            input_structure)
        complete_relation = bool(scope_conjunction_bindings or pair_inverse
                                  or slot_question_parts is not None
                                  or question_bound_closed_relation
                                  or self._partial_property_core_roots) or (
            (not composition_present or gap_only) and any(
                item.state == STRUCTURE_CLOSED
                and item.required_count >= 2
                and item.required_count == item.support_count
                and item.predicate_coverage == 1
                for item in input_structure.relation_candidates))
        if (not complete_relation and not input_structure.open_relations
                and artifact_input is None and not gap_only):
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
        self._diagnostic_stage("query_memory_restore_end")
        memory_grounded_slot_binding = None
        if input_structure.relation_question_bindings:
            memory_grounded_slot_binding = self._memory_grounded_slot_binding(
                input_structure)
            if memory_grounded_slot_binding is not None:
                selected_input = replace(
                    input_structure,
                    relation_question_bindings=(memory_grounded_slot_binding,),
                )
                selected_input = self._relation_slot_question_view(
                    selected_input, memory_grounded_slot_binding)
                if selected_input is None:
                    raise ValueError(
                        "Memory-grounded WH selection lost its exact Core scope")
                input_structure = selected_input
                slot_question_binding = memory_grounded_slot_binding
                slot_question_parts = self._relation_slot_question_parts(
                    input_structure, memory_grounded_slot_binding)
                if slot_question_parts is None:
                    raise ValueError(
                        "Memory-grounded WH selection lost its current Core fact")
                self._dynamic_slot_question_roles = tuple(
                    slot_question_parts[4])
                self._active_relation_keys = frozenset({
                    memory_grounded_slot_binding.relation_key})
                active_relation_keys = self._active_relation_keys
                concept_anchors = self.anchor_concepts(
                    input_structure,
                    allowed_predicate_keys=question_predicate_keys)
                self._event_time_query_bindings = (
                    self._match_event_time_generation_bindings(input_structure))
                self._event_time_query_frame_keys = frozenset(
                    item.frame_key for item in self._event_time_query_bindings)
                self._event_time_query_edge_keys = frozenset(
                    edge_key
                    for item in self._event_time_query_bindings
                    for edge_key in item.topology_edge_keys)
                complete_relation = True
                self._restore_memory_structures(input_structure, artifact_input)
                self._diagnostic_stage(
                    "query_memory_restore_after_memory_grounded_wh_selection")
        # An Event/Time candidate can be question-shaped even when no exact
        # relation-question frame matched the current surface.  Re-evaluate
        # after the optional Memory-grounded slot selection so that a valid
        # cross-turn binding is never shadowed by this fail-closed gate.
        unbound_event_time_question = (
            _has_unbound_event_time_question_candidate(
                input_structure, self._event_time_predicate_keys))
        if (slot_question_parts is not None
                and slot_question_parts[4]
                and self._input_observation_ref is not None):
            dynamic_roles = set(slot_question_parts[4])
            eligible = tuple(
                item for item in self._reference_candidate_sets
                if item.reference.role.stable_key() in dynamic_roles
                and self._reference_relation_context(item, input_structure)[0]
                is not None)
            if len(eligible) == 1:
                self._slot_question_core_roots = frozenset({
                    slot_question_parts[1].proposition_key})
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
        # A verified conjunction has a factual multi-claim ResponsePlan.  The
        # generic Companion act remains available for ordinary unknown input,
        # but its unrelated response roots must not keep this factual query in
        # ROOT_PENDING and block the same-turn relation generation gate.
        if composition_present and scope_conjunction_bindings:
            generic_response_nodes = ()
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

        from pure_integer_ai.experiments.operation_request_consumer import operation_request_query_seeds
        request_seeds = operation_request_query_seeds(input_structure, query_key, self.space_weights[0])
        request_targets = {root.root_key for root, _anchor, _entry in request_seeds}
        core_root_keys = set()
        for item in concept_anchors:
            # The complete request owns this root's current source/scope.
            # Its reader provenance remains in InputStructure; an unscoped
            # concept seed must not suppress the explicit request seed.
            if item.filler in request_targets:
                continue
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
        for proposition in self._slot_question_core_roots:
            if proposition in core_root_keys:
                continue
            fact = self.core_fact_index.get(proposition)
            if fact is None:
                continue
            source_ref = (fact.source_hash,)
            roots.append(QueryRoot(
                _CORE, proposition, ROOT_PENDING,
                self.space_weights[0], source_ref=source_ref,
                scope_key=query_key))
            anchors.append(QueryAnchor(
                _CORE, proposition, kind=12, scope_key=query_key))
            frontier.append(FrontierEntry(
                (_CORE, 92112, *proposition), _CORE,
                target_key=proposition, root_key=proposition,
                scope_key=query_key, source_ref=source_ref,
                owner_weight=self.space_weights[0], depth=0,
                required_slot_gain=3, evidence_support=1,
                relation_fit=3))
            core_root_keys.add(proposition)
        for proposition in self._partial_property_core_roots:
            if proposition in core_root_keys:
                continue
            fact = self.core_fact_index.get(proposition)
            if fact is None:
                continue
            source_ref = (fact.source_hash,)
            roots.append(QueryRoot(
                _CORE, proposition, ROOT_PENDING, self.space_weights[0],
                source_ref=source_ref, scope_key=query_key))
            anchors.append(QueryAnchor(
                _CORE, proposition, kind=_PARTIAL_PROPERTY_ANCHOR_KIND,
                scope_key=query_key))
            frontier.append(FrontierEntry(
                (_CORE, _PARTIAL_PROPERTY_ANCHOR_KIND, *proposition), _CORE,
                target_key=proposition, root_key=proposition,
                scope_key=query_key, source_ref=source_ref,
                owner_weight=self.space_weights[0], depth=0,
                required_slot_gain=3, evidence_support=1, relation_fit=3))
            core_root_keys.add(proposition)
        pending_by_key = {item.stable_key(): item for item in self.pending_relation_compositions}
        for root, anchor, entry in request_seeds:
            if root.root_key in core_root_keys:
                continue
            roots.append(root)
            anchors.append(anchor)
            frontier.append(entry)
            core_root_keys.add(root.root_key)
        for pending_binding in input_structure.pending_relation_composition_bindings:
            pending = pending_by_key[pending_binding.pending_key]
            for binding in pending.candidate.bindings:
                filler = binding.filler.stable_key()
                if filler in core_root_keys:
                    continue
                roots.append(QueryRoot(
                    _CORE, filler, ROOT_PENDING, self.space_weights[0],
                    source_ref=input_structure.source_ref, scope_key=query_key))
                anchors.append(QueryAnchor(_CORE, filler, kind=11, scope_key=query_key))
                frontier.append(FrontierEntry(
                    (_CORE, 92111, *filler), _CORE, target_key=filler, root_key=filler,
                    source_ref=input_structure.source_ref, scope_key=query_key,
                    owner_weight=self.space_weights[0], required_slot_gain=2, relation_fit=2))
                core_root_keys.add(filler)
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
            if (composition_present
                    and candidate.stable_key() not in self._active_relation_keys):
                continue
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
        event_time_query_propositions = frozenset(
            item.proposition_key for item in self._event_time_query_bindings)
        if self._restored_factual_topic_keys:
            anchored_propositions.update(self._kdconv_query_propositions)
        slot_question_proposition = (
            None if slot_question_parts is None
            else slot_question_parts[1].proposition_key)
        if (slot_question_parts is not None
                and not slot_question_parts[4]):
            anchored_propositions.add(slot_question_proposition)
        if memory_grounded_slot_binding is not None:
            anchored_propositions.add(slot_question_proposition)
        slot_question_relation_key = (
            None if slot_question_binding is None
            else slot_question_binding.relation_key)
        dialogue_keys = tuple(sorted({
            item.proposition_key
            for item in input_structure.relation_candidates
            if (item.proposition_key in self.dialogue_frame_index
                and item.proposition_key in anchored_propositions
                and (not composition_present
                     or item.stable_key() in self._active_relation_keys)
                # Once a real Event/Time generation binding is matched, only
                # its exact proposition/source is authorized for Dialogue.
                # Same-surface Event/Time rows without a connector remain
                # graph observations, but cannot compete as answer claims.
                and (item.predicate_key not in self._event_time_predicate_keys
                     or not event_time_query_propositions
                     or item.proposition_key in event_time_query_propositions)
                # A partially covered relation remains an open Core
                # candidate, but must not create a competing Dialogue frame.
                # Otherwise a filler shared by two propositions turns a
                # closed fact into a false ambiguity and blocks generation.
                and (item.state == STRUCTURE_CLOSED
                     or (item.proposition_key == slot_question_proposition
                         and item.stable_key() == slot_question_relation_key
                         and (item.required_open == 1
                              or (memory_grounded_slot_binding is not None
                                  and item.required_open
                                  == 1 + len(slot_question_parts[4]))))
                     or item.proposition_key in self._partial_property_core_roots))
        }))
        self._diagnostic_stage(
            f"dialogue_keys:{len(dialogue_keys)}:{len(set(item.proposition_key for item in input_structure.relation_candidates))}")
        best, second = self._rank_candidates(
            input_structure, dialogue_keys,
            active_relation_keys=self._active_relation_keys)
        selected_required_open = (
            int(bool(slot_question_parts[4]))
            if slot_question_parts is not None else 1)
        if self._partial_property_core_roots:
            selected_required_open = 0
        if best[1]:
            # Candidate aggregation is score-based.  A lower-scoring open
            # interpretation for the same proposition must not keep the
            # winning closed Event/Time binding open; equally scoring open
            # interpretations remain represented and are handled by the
            # existing conflict/competition fields.
            winning = tuple(
                item for item in input_structure.relation_candidates
                if item.proposition_key == best[1]
                and (not composition_present
                     or item.stable_key() in self._active_relation_keys)
                and item.score == best[0])
            selected_required_open = int(
                not winning or any(item.required_open > 0 for item in winning))
            if (slot_question_parts is not None
                    and best[1] == slot_question_proposition):
                selected_required_open = int(bool(slot_question_parts[4]))
        if self._partial_property_core_roots:
            selected_required_open = 0
        input_order_witnesses = symmetric_input_order_witnesses(
            input_structure, self.core_fact_index)
        explained_order_keys = {item.relation.stable_key() for item in input_order_witnesses}
        input_order_conflict = int(any(
            item.state == STRUCTURE_ORDER_CONFLICT
            and item.stable_key() not in explained_order_keys
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
        dialogue_memory_keys = tuple(sorted(
            self._discourse_topic_hypothesis_keys
            if (composition_present and scope_conjunction_bindings)
            else {
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
        discourse_hypotheses_by_key = {
            item.hypothesis_key: item for item in self._discourse_hypotheses}
        for key in dialogue_memory_keys:
            item = discourse_hypotheses_by_key[key]
            topic_root_empty = bool(
                composition_present and scope_conjunction_bindings)
            roots.append(QueryRoot(
                _DIALOGUE, key, ROOT_PENDING,
                self.space_weights[2], source_ref=item.source_ref,
                scope_key=item.scope_key,
            ))
            if topic_root_empty:
                roots[-1] = replace(roots[-1], status=ROOT_EMPTY)
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
            if topic_root_empty:
                continue
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
        conflicting_filler_groups = _conflicting_input_filler_groups(
            semantic_fillers)
        for item in semantic_fillers:
            identity = ObjectIdentity.from_stable_key(item.candidate_key)
            if identity.object_kind != item.object_kind:
                raise ValueError(
                    "input semantic filler identity kind drifted")
        initial_bindings = {}
        initial_evidence = {}
        for witness in input_order_witnesses:
            evidence = witness.query_evidence()
            initial_evidence[evidence.stable_key()] = evidence
        for binding, evidence in assertion_scope_query_records(input_structure, query_key):
            initial_bindings[binding.stable_key()] = binding
            initial_evidence[evidence.stable_key()] = evidence
        reference_consensus = self._memory_grounded_reference_source_consensus
        if reference_consensus is not None:
            selection_binding = BindingEntry(
                (REFERENCE_SOURCE_CONSENSUS_VERSION, 1), reference_consensus.stable_key(),
                space=_MEMORY, scope_key=query_key)
            initial_bindings[selection_binding.stable_key()] = selection_binding
        from pure_integer_ai.experiments.trained_input_clarification import unanchored_reference_question
        unanchored_reference = unanchored_reference_question(slot_question_parts, self._reference_candidate_sets)
        operation_binding_failure_reason = (
            "Event/Time relation lacks an exact trained question binding"
            if unbound_event_time_question else (
                "closed relation does not cover the complete current input"
                if uncovered_closed_relation_modifier else None))
        from pure_integer_ai.cognition.understanding.query_unlicensed_relation import (
            unlicensed_incomplete_relation_input,
        )
        if (operation_binding_failure_reason is None
                and artifact_input is None and not graph_object_keys
                and not self._event_time_query_bindings
                and unlicensed_incomplete_relation_input(
                    input_structure, self.relation_question_frames,
                    complete_relation=complete_relation)):
            operation_binding_failure_reason = (
                "incomplete relation lacks current roles or licensed reference")
        if unanchored_reference and operation_binding_failure_reason is None:
            operation_binding_failure_reason = "trained reference question lacks a source-qualified antecedent"
        from pure_integer_ai.experiments.query_pending_reference import unselected_reference_question
        unselected_reference = bool(
            memory_grounded_slot_binding is None
            and self._reference_candidate_sets
            and unselected_reference_question(input_structure, self.core_fact_index,
                                              self.relation_question_frames))
        if unselected_reference and operation_binding_failure_reason is None:
            operation_binding_failure_reason = "trained reference question lacks a unique source-qualified binding"
        from pure_integer_ai.experiments.operation_request_scope_query import pending_request_query_records
        pending_request_records = pending_request_query_records(self.operation_request_frames, input_structure)
        if pending_request_records:
            operation_binding_failure_reason = "trained operation request lacks unique complete current variables"
            for binding, evidence in pending_request_records:
                initial_bindings[binding.stable_key()] = binding
                initial_evidence[evidence.stable_key()] = evidence
        for item in semantic_fillers:
            conflict = int((item.proposition_key, item.source_ref, item.role_key)
                           in conflicting_filler_groups)
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
            if (composition_present
                    and item.stable_key() not in self._active_relation_keys):
                continue
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
        # Operation contracts are first-class integer graph inputs.  Their
        # binding/evidence is seeded into this same QueryState so a course row
        # cannot claim operation consumption merely by placing a key in a
        # receipt or split metadata.
        for operation_key in input_structure.operation_keys:
            mode_key = self.input_discourse_operation_modes[operation_key]
            proof_key = (
                *_INPUT_OPERATION_EVIDENCE,
                len(operation_key), *operation_key,
                len(mode_key), *mode_key,
            )
            if mode_key == MODE_RELATION_PAIR_QUERY:
                pending_matches = tuple(item for item in input_structure.pending_relation_composition_bindings
                                        if item.operation_key == operation_key)
                matches = tuple(
                    item for item in input_structure.relation_question_bindings
                    if item.operation_key == operation_key)
                if pending_matches:
                    proof_key += (92111, len(pending_matches[0].stable_key()),
                                  *pending_matches[0].stable_key())
                elif (not matches or (len(matches) != 1
                                   and not scope_conjunction_bindings)):
                    # An unresolved pair question must not authorize a Core
                    # answer.  A genuinely unknown/partial input may still
                    # take the trained context-only UNKNOWN Dialogue route;
                    # rejecting it here prevented that route from participating
                    # in the same QueryState and exposed an empty protocol
                    # response for unseen entities.  The exception is narrow:
                    # no closed relation, composition, explicit Core graph
                    # root, or Memory-grounded slot may be present.  Every
                    # other unbound pair query keeps the fail-closed
                    # CLARIFY_MISSING_BINDING gate below.
                    observation_key = (
                        None if self._input_observation_ref is None
                        else self._input_observation_ref.stable_key())
                    generic_unknown = any(
                        item.candidate_key
                        and item.candidate_key[0] == MEMORY_GENERIC_CANDIDATE_VERSION
                        and (observation_key is None
                             or item.observation_key == observation_key)
                        for item in self._discourse_hypotheses)
                    closed_text_relation = any(
                        item.state == STRUCTURE_CLOSED
                        and item.required_count >= 2
                        and item.required_count == item.support_count
                        and item.predicate_coverage == 1
                        for item in input_structure.relation_candidates)
                    allow_context_unknown = bool(
                        generic_unknown
                        and not complete_relation
                        and not closed_text_relation
                        and not composition_present
                        and not input_structure.pending_relation_composition_bindings
                        and not self._core_graph_input_keys
                        and not self._slot_question_core_roots)
                    if allow_context_unknown:
                        proof_key += (
                            92324, len(operation_key), *operation_key)
                    else:
                        operation_binding_failure_reason = (
                            "pair query lacks closed current-input bindings")
                        proof_key += (
                            92323, len(operation_key), *operation_key)
                elif len(matches) == 1:
                    proof_key += (len(matches[0].stable_key()), *matches[0].stable_key())
                else:
                    proof_key += (92029, len(matches), *(part for item in matches
                                    for part in (len(item.stable_key()),
                                                 *item.stable_key())))
            if input_structure.operation_request_bindings:
                request, = input_structure.operation_request_bindings
                proof_key += (92322, len(request.stable_key()), *request.stable_key())
            binding = BindingEntry(
                _INPUT_OPERATION_ROLE,
                operation_key,
                space=_DIALOGUE,
                scope_key=query_key,
            )
            evidence = EvidenceEntry(
                input_structure.source_ref,
                operation_key,
                space=_DIALOGUE,
                polarity=3,
                trust=1,
                evidence_key=proof_key,
                scope_key=query_key,
                payload_key=proof_key,
            )
            initial_bindings[binding.stable_key()] = binding
            initial_evidence[evidence.stable_key()] = evidence
        for composition_key, relation_key, members in scope_conjunction_bindings:
            proof_key = (
                *_INPUT_DISCOURSE_CONJUNCTION_EVIDENCE,
                len(composition_key), *composition_key,
                len(relation_key), *relation_key,
                len(members),
                *(value for member in members
                  for value in (len(member), *member)),
            )
            binding = BindingEntry(
                _INPUT_DISCOURSE_CONJUNCTION_ROLE, proof_key,
                space=_DIALOGUE, scope_key=query_key)
            evidence = EvidenceEntry(
                input_structure.source_ref, proof_key,
                space=_DIALOGUE, polarity=3, trust=1,
                evidence_key=proof_key, scope_key=query_key,
                payload_key=proof_key)
            initial_bindings[binding.stable_key()] = binding
            initial_evidence[evidence.stable_key()] = evidence
        for composition_key, operation_key, mode_key, relation_key in (
                self._fact_query_operations):
            proof_key = (
                *_INPUT_FACT_QUERY_EVIDENCE,
                len(composition_key), *composition_key,
                len(operation_key), *operation_key,
                len(mode_key), *mode_key,
                len(relation_key), *relation_key,
            )
            binding = BindingEntry(
                _INPUT_FACT_QUERY_ROLE, proof_key,
                space=_DIALOGUE, scope_key=query_key)
            evidence = EvidenceEntry(
                input_structure.source_ref, relation_key,
                space=_DIALOGUE, polarity=3, trust=1,
                evidence_key=proof_key, scope_key=query_key,
                payload_key=proof_key)
            initial_bindings[binding.stable_key()] = binding
            initial_evidence[evidence.stable_key()] = evidence
        if unbound_relation_question:
            operation_binding_failure_reason = (
                "trained relation-question frame lacks an exact current binding")

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
            required_slots_open=(composition_required_open
                                 if composition_present
                                 else selected_required_open),
            best_score=best[0],
            second_score=second[0],
            best_candidate_key=best[1],
            second_candidate_key=second[1],
        )
        self._diagnostic_stage(
            f"query_initial_ready:roots={len(roots)}:frontier={len(frontier)}:"
            f"bindings={len(initial.bindings)}:evidence={len(initial.evidence)}")
        expander = _CrossSpaceExpander(
            self,
            input_structure=input_structure,
            artifact_input=artifact_input,
            artifact_bridge_matches=self._artifact_bridge_matches,
            input_order_conflict=input_order_conflict,
            input_required_slots_open=(composition_required_open
                                       if composition_present
                                       else selected_required_open),
            scope_conjunction_bindings=scope_conjunction_bindings,
            response_nodes=response_nodes,
            generic_response_nodes=generic_response_nodes,
            generic_hypotheses=generic_hypotheses,
        )
        tracer = _IntegerQueryTrace(reference_mode=compact_trace)
        tracer.record(initial.with_(frontier=tuple(frontier)).canonical(),
                      TERMINATION_OPEN)
        # 完整端点竞争在事实搜索前拒绝；只允许同次受限 frontier 消费
        # 当前 O/H/E 和受训非事实生成节点，保留完整输入与最终冲突状态。
        input_relation_conflict = self._input_relation_competition(
            input_structure,
            question_predicate_keys=input_competition_predicate_keys,
        )
        if input_relation_conflict:
            from pure_integer_ai.experiments.trained_input_clarification import (
                clarification_generation_payload, run_input_clarification,
            )
            clarification_state, clarification = run_input_clarification(
                initial, tuple(frontier), generic_hypotheses,
                generic_response_nodes, expander=expander, tracer=tracer,
                input_role_occurrences=self._input_role_occurrences,
                input_relation_conflict=True)
            plan = None if clarification is None else clarification.response_plan
            surface = "" if plan is None else plan.surface()
            failure_termination = (clarification_state.termination
                if clarification_state.termination in {
                    TERMINATION_BUDGET_EXHAUSTED, 6}
                else TERMINATION_CLARIFY_CONFLICT)
            failed = clarification_state.with_(
                frontier=(),
                termination=failure_termination,
                conflict_open=1,
                evidence_closed=0,
                generation_ready=int(clarification is not None),
            ).canonical()
            if clarification is not None:
                self._pending_response_delivery = (
                    failed, GenericResponseGeneration(
                        plan, self._input_observation_ref,
                        clarification.trace, tuple(map(ord, surface)),
                        clarification.graph_surfaces))
            tracer.record(failed, failure_termination)
            self._diagnostic_stage(
                "query_input_relation_conflict_gate:frontier_cleared")
            return self._finalize(
                failed, tuple(expander.hops),
                termination=failure_termination,
                reason=_TERMINATION_NAMES[failure_termination],
                failure_reason="competing Core endpoint identities in input",
                generation=(None if clarification is None else
                    clarification_generation_payload(
                        clarification, _response_plan_connector_key(plan))),
                response_plan=None if plan is None else self._integer_response_plan(plan),
                response_surface=surface,
                frontier_trace=tracer.export(),
                visited=tuple(item.stable_key() for item in failed.visited),
                expanded_edges=tuple(expander.expanded_edges),
                generic_response_generations=tuple(
                    item.stable_key() for item in expander.generic_response_generations),
                operation_keys=input_structure.operation_keys,
                operation_contracts=tuple(
                    (key, self.input_discourse_operation_modes[key])
                    for key in input_structure.operation_keys),
                discourse_relation_feature_state=discourse_relation_feature_state,
                memory_routes=self._memory_routes,
                response_contexts=tuple(item.stable_key()
                                        for item in self._response_contexts),
                input_role_occurrences=tuple(
                    item.stable_key() for item in self._input_role_occurrences),
                reference_candidate_sets=tuple(
                    item.stable_key() for item in self._reference_candidate_sets),
                reference_resolutions=tuple(
                    item.stable_key() for item in self._reference_resolutions),
                generic_response_feature_mask=generic_response_feature_mask,
                input_dialogue_act_match=self._input_dialogue_act_trace(),
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
                event_time_generation_bindings=tuple(
                    item.integer_trace()
                    for item in self.event_time_generation_bindings),
                event_time_query_bindings=tuple(
                    item.integer_trace()
                    for item in self._event_time_query_bindings),
                input_structure=input_structure,
                artifact_input=artifact_input,
                compact_trace=compact_trace,
            )
        if operation_binding_failure_reason is not None:
            # 缺槽、未绑定问式及完整输入失败共用原非事实资格核验。
            # 当前 O/H/E 保留真实立场；冲突、事实证据和已展开查询仍关闭。
            from pure_integer_ai.experiments.trained_input_clarification import (
                clarification_failure_termination, clarification_generation_payload, run_input_clarification,
            )
            clarification_state, clarification = run_input_clarification(
                initial, tuple(frontier), generic_hypotheses,
                generic_response_nodes, expander=expander, tracer=tracer,
                input_role_occurrences=(self._input_role_occurrences
                    if unanchored_reference or unselected_reference else ()))
            clarification_plan = (None if clarification is None
                                  else clarification.response_plan)
            clarification_surface = ("" if clarification_plan is None
                                     else clarification_plan.surface())
            clarification_payload = (None if clarification is None else
                clarification_generation_payload(
                    clarification, _response_plan_connector_key(clarification_plan)))
            failure_termination = clarification_failure_termination(clarification_state)
            failed = clarification_state.with_(
                frontier=(),
                termination=failure_termination,
                required_slots_open=max(1, clarification_state.required_slots_open),
                evidence_closed=0,
                generation_ready=int(clarification is not None),
            ).canonical()
            if clarification is not None:
                self._pending_response_delivery = (
                    failed, GenericResponseGeneration(
                        clarification_plan, self._input_observation_ref,
                        clarification.trace, tuple(map(ord, clarification_surface)),
                        clarification.graph_surfaces))
            tracer.record(failed, failure_termination)
            return self._finalize(
                failed, tuple(expander.hops),
                termination=failure_termination,
                reason=_TERMINATION_NAMES[failure_termination],
                failure_reason=operation_binding_failure_reason,
                generation=clarification_payload,
                response_plan=(None if clarification_plan is None else
                               self._integer_response_plan(clarification_plan)),
                response_surface=clarification_surface,
                frontier_trace=tracer.export(),
                visited=tuple(item.stable_key() for item in failed.visited),
                expanded_edges=tuple(expander.expanded_edges),
                generic_response_generations=tuple(
                    item.stable_key() for item in expander.generic_response_generations),
                operation_keys=input_structure.operation_keys,
                operation_contracts=tuple(
                    (key, self.input_discourse_operation_modes[key])
                    for key in input_structure.operation_keys),
                discourse_relation_feature_state=discourse_relation_feature_state,
                memory_routes=self._memory_routes,
                response_contexts=tuple(item.stable_key()
                                        for item in self._response_contexts),
                input_role_occurrences=tuple(
                    item.stable_key() for item in self._input_role_occurrences),
                reference_candidate_sets=tuple(
                    item.stable_key() for item in self._reference_candidate_sets),
                reference_resolutions=tuple(
                    item.stable_key() for item in self._reference_resolutions),
                generic_response_feature_mask=generic_response_feature_mask,
                input_dialogue_act_match=self._input_dialogue_act_trace(),
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
                event_time_generation_bindings=tuple(
                    item.integer_trace()
                    for item in self.event_time_generation_bindings),
                event_time_query_bindings=tuple(
                    item.integer_trace()
                    for item in self._event_time_query_bindings),
                input_structure=input_structure,
                artifact_input=artifact_input,
                compact_trace=compact_trace,
            )
        state = run_query(
            initial.with_(frontier=tuple(frontier)),
            expander=expander,
            tracer=tracer,
        )
        self._diagnostic_stage(
            f"query_run_end:nodes={state.node_count}:edges={state.edge_count}:"
            f"reads={state.read_count}:termination={state.termination}")
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
        # The ordinary termination predicate intentionally rejects a tied
        # best/second candidate.  A verified scope conjunction is the one
        # graph relation that authorizes that tie to remain multi-claim: both
        # members are independently Core/Dialogue-supported and generated in
        # this same QueryState.  Promote only that unique closed relation;
        # unrelated ties stay UNKNOWN/CLARIFY.
        conjunction_claim_sets = _closed_scope_conjunction_components(
            scope_conjunction_bindings,
            self.generation_ready_propositions,
        )
        unvisited_count = len(unvisited_frontier(state.frontier, state.visited))
        roots_pending = sum(root.status == ROOT_PENDING for root in state.roots)
        promote_conjunction = bool(
            composition_present
            and len(conjunction_claim_sets) == 1
            and state.evidence_closed
            and not state.conflict_open
            and state.required_slots_open == 0
            and unvisited_count == 0
            and roots_pending == 0
            and state.termination != TERMINATION_ANSWER_CLOSED)
        self._diagnostic_stage(
            f"query_conjunction_gate:composition={int(composition_present)}:"
            f"components={len(conjunction_claim_sets)}:"
            f"evidence={state.evidence_closed}:conflict={state.conflict_open}:"
            f"required={state.required_slots_open}:frontier={unvisited_count}:"
            f"roots_pending={roots_pending}:termination={state.termination}:"
            f"promote={int(promote_conjunction)}")
        if promote_conjunction:
            state = state.with_(termination=TERMINATION_ANSWER_CLOSED)
            tracer.record(state, TERMINATION_ANSWER_CLOSED)
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
        if (not composition_present and not operation_program_route and hops
                and state.termination == TERMINATION_ANSWER_CLOSED):
            generation = self._generate_from_closed(
                hops, concept_anchors, state.best_candidate_key)
        # 阶段 C：把闭合命题经 ResponsePlan 组织并执行 token postcheck。生成
        # dict 只作为可重放 trace；真实发布承重回答从 ResponsePlan 渲染。若
        # plan 无法通过 postcheck（无结构证据/必填槽缺 token），则回答保持
        # None 由上层 fail-closed，不退回旧表层回放。
        response_plan = (
            self._response_plan_from_closed(
                hops, concept_anchors, state.best_candidate_key, state,
                reference_resolutions=self._reference_resolutions,
                discourse_contexts=self._reference_generation_contexts())
            if (not composition_present and not operation_program_route
                and state.termination == TERMINATION_ANSWER_CLOSED)
            else None
        )
        if (expander.composition_output is not None
                and state.termination == TERMINATION_ANSWER_CLOSED
                and composition_output_is_current(expander.composition_output, state)):
            response_plan = expander.composition_output.response_plan(state)
            generation = expander.composition_output.generation_value()
        if (composition_present and not operation_program_route
                and state.termination == TERMINATION_ANSWER_CLOSED):
            response_plan = self._multi_response_plan_from_closed(
                hops, state, scope_conjunction_bindings=scope_conjunction_bindings,
                require_predicate_contract=bool(input_structure.relation_question_bindings),
                reference_resolutions=self._reference_resolutions,
                discourse_contexts=self._reference_generation_contexts())
            if response_plan is not None and len(response_plan.claim_refs) >= 2:
                generation = {
                    "kind": 4,
                    "claim_keys": [list(item.stable_key())
                                   for item in response_plan.claim_refs],
                    "surface_values": list(map(ord, response_plan.surface())),
                    "scope_conjunction_bindings": [
                        [list(composition), list(relation),
                         [list(member) for member in members]]
                        for composition, relation, members
                        in scope_conjunction_bindings],
                }
        operation_unknown_execution = None
        if operation_program_route:
            from pure_integer_ai.experiments.operation_program_query import (
                consume_operation_program, operation_plan_matches,
            )
            from pure_integer_ai.experiments.operation_program_generation import generate_operation_response
            state, operation_execution = consume_operation_program(
                self.core_runtime, self.operation_program_witnesses, input_structure, state)
            if input_structure.operation_request_bindings:
                from pure_integer_ai.experiments.operation_request_consumer import close_operation_request_selection
                state = close_operation_request_selection(self, input_structure, state, operation_execution,
                                                         logical_conflict=expander.operation_logical_conflict)
            if state.termination == TERMINATION_CLARIFY_CONFLICT:
                tracer.record(state, TERMINATION_CLARIFY_CONFLICT)
            from pure_integer_ai.experiments.operation_companion_query import consume_operation_companion
            state, source_reads = consume_operation_companion(self.core_runtime,
                self.dialogue_frame_index, self.core_fact_index, operation_execution, state,
                owner_weight=self.space_weights[2], tracer=tracer)
            hops = tuple(sorted((*hops, *(QueryHop(_DIALOGUE, item.proposition, item.predicate,
                (_DIALOGUE, *item.proposition), (_CORE, *item.proposition), (),
                item.depth, item.source_hash) for item in source_reads)),
                key=lambda item: (item.depth, item.space, item.proposition)))
            operation_output_proof = None
            if self.operation_output_witnesses is not None:
                from pure_integer_ai.experiments.operation_output_query import consume_operation_output
                state, operation_output_proof = consume_operation_output(self.core_runtime,
                    self.operation_output_witnesses, self.operation_program_witnesses, operation_execution, state,
                    memory_hypotheses=self._memory_hypotheses,
                    current_observation_key=(() if self._input_observation_ref is None else
                                             self._input_observation_ref.stable_key()))
            state, response_plan, generation = generate_operation_response(
                self.core_runtime, self.surface_generator, operation_execution, state,
                output_witnesses=self.operation_output_witnesses, output_proof=operation_output_proof,
                output_programs=self.operation_program_witnesses)
            if not operation_plan_matches(operation_execution, response_plan):
                response_plan, generation = None, None
                state = state.with_(termination=(state.termination if state.termination in {
                    TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CLARIFY_CONFLICT}
                    else TERMINATION_CLARIFY_MISSING_BINDING),
                    evidence_closed=0, generation_ready=0,
                    required_slots_open=(state.required_slots_open
                        if state.termination == TERMINATION_CLARIFY_CONFLICT
                        else max(1, state.required_slots_open)))
            else:
                tracer.record(state, TERMINATION_ANSWER_CLOSED)
            from pure_integer_ai.experiments.operation_unknown_response import generate_unknown_operation_responses
            unknown_responses = generate_unknown_operation_responses(
                self.surface_generator, operation_execution, state, input_structure,
                expander.generic_hypotheses, tuple(expander.generic_response_nodes.values()),
                self._input_observation_ref, request_hypotheses=self._memory_hypotheses)
            if len(unknown_responses) == 1:
                expander.generic_response_generations = unknown_responses
                operation_unknown_execution = operation_execution
                state = state.with_(bindings=tuple(sorted(set((*state.bindings,
                    BindingEntry(unknown_responses[0].response_plan.response_act.stable_key(),
                        unknown_responses[0].stable_key(), space=_DIALOGUE, scope_key=state.query_key))),
                    key=lambda item: item.stable_key())))
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
            claim_keys = tuple(item.stable_key() for item in response_plan.claim_refs)
            claim_input_hypotheses = ()
            if self.memory is not None:
                claim_input_hypotheses = resolve_factual_claim_input_hypotheses(
                    self.memory, self._input_observation_ref, claim_keys)
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
                    claim_keys if len(response_plan.claim_refs) > 1 else (),
                    claim_input_hypotheses,
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
        resolved_reference_resolutions = tuple(
            item for item in self._reference_resolutions
            if item.requires_resolution and item.status == REFERENCE_RESOLVED)
        # A generic Companion plan is built without cross-turn inputs.  If it
        # is the only eligible plan, it must still carry the same resolved
        # REFERS provenance as every other generation kind; otherwise a later
        # generic adoption would silently discard the winner from the actual
        # ResponsePlan.  Keep this binding fail-closed: an unusable resolved
        # context cannot be emitted as an unqualified UNKNOWN response.
        generic_reference_plan = None
        if generic is not None:
            generic_reference_plan = generic.response_plan
            if resolved_reference_resolutions:
                try:
                    generic_reference_plan = self._bind_reference_response_plan(
                        generic_reference_plan,
                        state,
                        self._reference_generation_contexts(),
                        resolved_reference_resolutions,
                    )
                except (TypeError, ValueError):
                    generic_reference_plan = None
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
        generic_adoption_ready = _generic_response_adoption_allowed(
            response_plan,
            generic_reference_plan,
            generic_structural_preferred=generic_structural_preferred,
            resolved_reference_resolutions=resolved_reference_resolutions,
        )
        # The expander computes this gate while converging the shared
        # QueryState.  Keep the final generic arbitration on that same state
        # instead of referring to a query-local name that only exists inside
        # ``_with_query_status``.
        dynamic_slot_blocked = bool(
            getattr(self, "_dynamic_slot_query_blocked", False))

        def generic_conflicts_closed() -> bool:
            """Close only conflicts consumed by an exact Memory/input join."""
            conflicts = unresolved_input_conflict_bindings(state, input_order_witnesses)
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
        scoped_nonfactual_ready = assertion_scope_response_allowed(input_structure, state)
        unresolved_context_ready = bool(
            dynamic_slot_blocked and generic is not None
            and not generic.graph_surfaces and generic_unknown_keys
            and not resolved_reference_resolutions
            and generic_reference_plan is not None
            and not generic_reference_plan.claim_refs
            and not generic_reference_plan.claim_evidence_refs
            and not any(item.hypothesis_key in generic_unknown_keys
                        and item.polarity in (1, 2) for item in state.evidence)
            and unresolved_role_context_allowed(state, input_structure))
        if (not composition_present and generic_adoption_ready
                and (not dynamic_slot_blocked or unresolved_context_ready)
                and (not state.conflict_open or generic_only_conflict
                     or generic_structural_closed
                     or generic_structural_unknown
                     or generic_context_only_open or scoped_nonfactual_ready)
                    and (state.termination not in {
                    TERMINATION_CLARIFY_CONFLICT,
                    TERMINATION_DIALOGUE_UNKNOWN}
                     or generic_adoption_ready)):
            if len(expander.generic_response_generations) == 1:
                generic = expander.generic_response_generations[0]
                response_plan = generic_reference_plan
                unknown_surface = response_plan.surface()
                # Generic UNKNOWN is a completed, evidence-backed dialogue
                # act, not a factual answer and not an unresolved routing
                # failure.  Preserve evidence_closed=0 while giving the
                # shared QueryState an explicit non-factual terminal state.
                state = state.with_(
                    termination=(TERMINATION_CLARIFY_MISSING_BINDING
                        if unresolved_context_ready else (
                            TERMINATION_CLARIFY_CONFLICT
                            if scoped_nonfactual_ready and state.conflict_open
                            else TERMINATION_DIALOGUE_UNKNOWN)),
                    required_slots_open=(max(1, state.required_slots_open)
                        if unresolved_context_ready else state.required_slots_open),
                    evidence_closed=0,
                    generation_ready=1,
                )
                tracer.record(state, state.termination)
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
        if composition_present:
            factual_conjunction = bool(
                state.termination == TERMINATION_ANSWER_CLOSED
                and response_plan is not None
                and len(response_plan.claim_refs) >= 2)
            if factual_conjunction:
                pass
            else:
                self._pending_response_delivery = None
                generation = None
                response_plan = None
            generic = (expander.generic_response_generations[0]
                       if len(expander.generic_response_generations) == 1
                       else None)
            if factual_conjunction:
                pass
            elif (generic is not None and generic_reference_plan is not None
                  and not generic_reference_plan.claim_refs):
                response_plan = generic_reference_plan
                unknown_surface = response_plan.surface()
                state = state.with_(
                    termination=TERMINATION_DIALOGUE_UNKNOWN,
                    evidence_closed=0,
                    generation_ready=1,
                    required_slots_open=max(1, state.required_slots_open),
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
                    "representations": [
                        list(item.stable_key()) for item in generic.representations],
                    "trace": list(generic.trace),
                    "surface_values": list(map(ord, unknown_surface)),
                }
            else:
                state = state.with_(
                    termination=TERMINATION_NO_FRONTIER,
                    evidence_closed=0,
                    generation_ready=0,
                    required_slots_open=max(1, state.required_slots_open),
                )
                tracer.record(state, TERMINATION_NO_FRONTIER)
        if operation_program_route:
            # All later open/generic recovery paths remain subject to the same
            # semantic program. They cannot bypass a rejected factual Plan.
            from pure_integer_ai.experiments.operation_unknown_response import unknown_operation_plan_allowed
            unknown_delivery = (operation_unknown_execution is not None
                and unknown_operation_plan_allowed(operation_unknown_execution, state, response_plan, generation))
            if unknown_delivery:
                generation = {**generation,
                    'unknown_operation_execution': list(operation_unknown_execution.stable_key())}
            elif (not operation_plan_matches(operation_execution, response_plan)
                    or generation is None or "operation_execution" not in generation
                    or (self.operation_output_witnesses is not None
                        and (operation_output_proof is None
                            or generation.get("operation_output_proof") != list(operation_output_proof.stable_key())))):
                response_plan, generation = None, None
                self._pending_response_delivery = None
                if state.termination == TERMINATION_CLARIFY_CONFLICT:
                    state = state.with_(evidence_closed=0, generation_ready=0, conflict_open=1)
                else:
                    state = state.with_(termination=TERMINATION_CLARIFY_MISSING_BINDING,
                        evidence_closed=0, generation_ready=0,
                        required_slots_open=max(1, state.required_slots_open))
        if generation is not None and resolved_reference_resolutions:
            reference_contexts = self._reference_generation_contexts()
            (reference_objects, _reference_evidence, reference_memory,
             _reference_targets) = _reference_generation_provenance(
                 state, reference_contexts, resolved_reference_resolutions)
            generation = {
                **generation,
                "reference_provenance": {
                    "resolution_keys": [
                        list(item.stable_key())
                        for item in resolved_reference_resolutions],
                    "discourse_objects": [
                        list(item.stable_key()) for item in reference_objects],
                    "memory_refs": [
                        list(item.stable_key())
                        for item in sorted({
                            *reference_memory,
                            *_discourse_memory_refs(reference_contexts),
                        }, key=lambda item: item.stable_key())],
                },
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
            input_dialogue_act_match=self._input_dialogue_act_trace(),
            fact_query_operations=self._fact_query_operations,
            operation_keys=input_structure.operation_keys,
            operation_contracts=tuple(
                (key, self.input_discourse_operation_modes[key])
                for key in input_structure.operation_keys),
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
            input_structure=expander.input_structure,
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

    @observed_relation_phase("delivery")
    @_memory_event_read_scope
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
            "claim_evidence_refs": [
                {
                    "claim": list(claim),
                    "evidence": [list(ref) for ref in refs],
                }
                for claim, refs in plan.claim_evidence_refs
            ],
            "claim_slot_ranges": [
                {"claim": list(claim), "start": start, "end": end}
                for claim, start, end in plan.claim_slot_ranges
            ],
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
        # A closed trained relation can be consumed through its typed input
        # projection without emitting a direct proposition hop.  This is the
        # normal route for an open-frame/Event-Time binding: the shared
        # QueryState records the exact proposition, roles and evidence in
        # ``typed_relation_projections`` while Core traversal hops carry the
        # adjacent filler topology.  Accept that query-local proof, but keep
        # the direct-hop requirement for all other routes.
        projection_proof = proposition_key in self.typed_relation_projections
        event_time_proof = any(
            item.proposition_key == proposition_key
            for item in self._event_time_query_bindings)
        if not core_hops and not projection_proof and not event_time_proof:
            return None
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
        if projection_proof or event_time_proof:
            fact = self.core_fact_index.get(proposition_key)
            generated = self.generation_by_proposition.get(proposition_key)
            if (fact is not None and generated is not None
                    and generated.surface.strip()):
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

    @observed_relation_phase("response_plan")
    def _response_plan_from_closed(
            self,
            hops: tuple[QueryHop, ...],
            anchors: tuple[ConceptAnchor, ...],
            proposition_key: tuple[int, ...],
            state: QueryState,
            *,
            reference_resolutions: tuple[CrossTurnReferenceResolution, ...] = (),
            discourse_contexts: tuple[object, ...] = (),
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
            self._diagnostic_stage("single_plan_closed_fact:present=0")
            return None
        fact, generated = found
        phase = "build"
        try:
            plan = plan_from_active_fact(fact, generated)
            if plan is None:
                self._diagnostic_stage("single_plan_built:present=0")
                return None
            phase = "query_evidence"
            plan = bind_query_evidence(plan, state)
            phase = "typed_projection"
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
            phase = "reference"
            plan = self._bind_reference_response_plan(
                plan, state, discourse_contexts, reference_resolutions)
            phase = "event_time"
            return self._bind_event_time_response_plan(
                plan, proposition_key, generated)
        except (TypeError, ValueError) as error:
            self._diagnostic_stage(
                f"single_plan_rejected:phase={phase}:"
                f"error={str(error).encode('unicode_escape').decode('ascii')}")
            return None

    @staticmethod
    def _bind_reference_response_plan(
            plan,
            state: QueryState,
            discourse_contexts: tuple[object, ...],
            reference_resolutions: tuple[CrossTurnReferenceResolution, ...],
            ):
        """Carry an evidence-qualified REFERS winner into factual generation.

        The factual Core plan remains the selected claim, but its output is
        also authorized by the same resolved cross-turn provenance that was
        present in the shared QueryState.  This preserves the prior delivered
        response O/H and the winner O/H as typed Memory references; it does
        not infer a claim from the prior surface or add any text.
        """
        required = tuple(item for item in reference_resolutions
                         if item.requires_resolution)
        if not required:
            return plan
        if any(item.status != REFERENCE_RESOLVED for item in required):
            raise ValueError("factual ResponsePlan cannot consume unresolved REFERS")
        (reference_objects, _reference_evidence, reference_memory,
         reference_targets) = _reference_generation_provenance(
             state, discourse_contexts, reference_resolutions)
        memory_refs = set(plan.memory_refs)
        memory_refs.update(reference_memory)
        memory_refs.update(_discourse_memory_refs(discourse_contexts))
        discourse_links = tuple(sorted({
            *plan.discourse_links,
            *reference_objects,
            *(item.proof_ref for item in discourse_contexts),
        }, key=ObjectIdentity.stable_key))
        event_refs = tuple(sorted({
            *plan.event_refs,
            *(item for item in reference_targets
              if item.object_kind == OBJECT_EVENT),
        }, key=ObjectIdentity.stable_key))
        return replace(
            plan,
            discourse_links=discourse_links,
            event_refs=event_refs,
            memory_refs=tuple(sorted(memory_refs, key=lambda item: item.stable_key())),
        )

    @observed_relation_phase("response_plan")
    def _multi_response_plan_from_closed(
            self,
            hops: tuple[QueryHop, ...],
            state: QueryState,
            *,
            scope_conjunction_bindings: tuple[tuple[tuple[int, ...], tuple[int, ...],
                                                       tuple[tuple[int, ...], ...]], ...] = (),
            require_predicate_contract: bool = False,
            reference_resolutions: tuple[CrossTurnReferenceResolution, ...] = (),
            discourse_contexts: tuple[object, ...] = (),
            ) -> object | None:
        """Build a multi-claim plan only through an explicit trained composer.

        The composer owns the learned literal intervals and connector.  The
        bridge supplies only the current Core facts and QueryState; it never
        concatenates component surfaces or infers a conjunction in Python.
        """
        if (state.termination != TERMINATION_ANSWER_CLOSED
                or state.conflict_open or state.required_slots_open
                or not state.evidence_closed):
            self._diagnostic_stage(
                f"multi_plan_gate:termination={state.termination}:"
                f"conflict={state.conflict_open}:required={state.required_slots_open}:"
                f"evidence={state.evidence_closed}:result=reject")
            return None
        composer = getattr(self.surface_generator, "generate_composition", None)
        if not callable(composer):
            self._diagnostic_stage("multi_plan_composer:callable=0:result=reject")
            return None
        # A compound input may traverse many Core edges while only one
        # explicitly trained conjunction authorizes a multi-claim answer.
        # Select claims from that graph relation, never from hop count or
        # stable-key ordering.  Without one unique two-member binding the
        # composition remains UNKNOWN/CLARIFY.
        candidate_components = _closed_scope_conjunction_components(
            scope_conjunction_bindings,
            self.generation_ready_propositions,
        )
        self._diagnostic_stage(
            f"multi_plan_components:bindings={len(scope_conjunction_bindings)}:"
            f"components={len(candidate_components)}:"
            f"ready={len(self.generation_ready_propositions)}")
        if len(candidate_components) != 1:
            return None
        proposition_keys = candidate_components[0]
        if len(proposition_keys) < 2:
            self._diagnostic_stage(
                f"multi_plan_claims:count={len(proposition_keys)}:result=reject")
            return None
        # Connected-component detection intentionally treats the claim set as
        # unordered, but the trained input conjunction still carries an
        # ordered scope sequence. Preserve that sequence for composition so a
        # held-out clause permutation can select the same integer frame with
        # different current claim bindings. The set check prevents an
        # unrelated or partial edge from changing output order.
        ordered_members = []
        seen_members = set()
        for _composition, _relation, members in scope_conjunction_bindings:
            for member in members:
                # A connected chain repeats its shared endpoint in the next
                # edge.  Keep the first scope occurrence while rejecting any
                # member that falls outside the verified component.
                if member not in seen_members:
                    seen_members.add(member)
                    ordered_members.append(member)
        if set(ordered_members) != set(proposition_keys):
            self._diagnostic_stage(
                f"multi_plan_ordered_members:ordered={len(ordered_members)}:"
                f"component={len(proposition_keys)}:result=reject")
            return None
        proposition_keys = tuple(ordered_members)
        from pure_integer_ai.experiments.generation_organization import (
            ClosedGenerationEvidence,
            bind_query_evidence,
            build_multi_response_plan,
        )
        evidences = []
        components = []
        for key in proposition_keys:
            fact = self.core_fact_index.get(key)
            generated = self.generation_by_proposition.get(key)
            if fact is None or generated is None:
                self._diagnostic_stage(
                    f"multi_plan_fact_generation:fact={int(fact is not None)}:"
                    f"generation={int(generated is not None)}:result=reject")
                return None
            evidences.append(ClosedGenerationEvidence(
                fact.bindings,
                generated,
                key,
                fact.predicate.stable_key(),
                fact.source_hash,
                anchor_surface=fact.evidence_surface,
                connector=generated.connector,
            ))
            components.append((
                self.core_runtime.generation_input(fact.proposition),
                fact,
                generated,
            ))
        try:
            if require_predicate_contract:
                support = getattr(self.surface_generator, "supports_predicates", None)
                metadata = getattr(self.surface_generator, "metadata", None)
                predicates = tuple(item[1].predicate.stable_key()
                                   for item in components)
                predicate_support = callable(support) and bool(support(predicates))
                metadata_source = (isinstance(metadata, dict)
                                   and metadata.get("source_identity")
                                   == self.relation_question_graph_metadata["source_identity"])
                metadata_license = (isinstance(metadata, dict)
                                    and metadata.get("license_identity")
                                    == self.relation_question_graph_metadata["license_identity"])
                self._diagnostic_stage(
                    f"multi_plan_contract:support={int(predicate_support)}:"
                    f"source={int(metadata_source)}:license={int(metadata_license)}")
                if not (predicate_support and metadata_source and metadata_license):
                    return None
            composed = composer(tuple(components), state)
            self._diagnostic_stage(
                f"multi_plan_composed:present={int(composed is not None)}")
            slot_contract_builder = getattr(
                self.surface_generator, "claim_slot_contract", None)
            claim_slot_contract = (
                slot_contract_builder(tuple(components))
                if callable(slot_contract_builder) else None)
            self._diagnostic_stage(
                f"multi_plan_slot_contract:present={int(claim_slot_contract is not None)}")
            plan = build_multi_response_plan(
                tuple(evidences), composed,
                claim_slot_contract=claim_slot_contract)
            if plan is None:
                self._diagnostic_stage("multi_plan_built:present=0")
                return None
            plan = bind_query_evidence(plan, state)
            self._diagnostic_stage(
                f"multi_plan_built:present=1:claims={len(plan.claim_refs)}")
            return self._bind_reference_response_plan(
                plan, state, discourse_contexts, reference_resolutions)
        except (RuntimeError, TypeError, ValueError) as error:
            self._diagnostic_stage(
                f"multi_plan_error:{type(error).__name__}:result=reject")
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
                  failure_reason: str | None = None,
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
                  input_dialogue_act_match: tuple[int, ...] = (),
                  fact_query_operations: tuple[tuple[int, ...], ...] = (),
                  operation_keys: tuple[tuple[int, ...], ...] = (),
                  operation_contracts: tuple[
                      tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
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
            reference_cache: dict[int, tuple[tuple[int, ...], int]] = {}

            def ref(value: tuple[int, ...]) -> int:
                """驻留已经过状态或树边界校验的整数键并返回一基引用号。"""
                if type(value) is not tuple:
                    raise TypeError("compact trace 只能驻留整数 tuple")
                cached = reference_cache.get(id(value))
                if cached is not None and cached[0] is value:
                    return cached[1]
                if not value:
                    return 0
                current = key_ids.get(value)
                if current is None:
                    current = len(key_table) + 1
                    key_ids[value] = current
                    key_table.append(value)
                reference_cache[id(value)] = (value, current)
                while len(reference_cache) > 8192:
                    reference_cache.pop(next(iter(reference_cache)))
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
                 "failure_reason": failure_reason,
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
                "input_dialogue_act_match": list(input_dialogue_act_match),
                "fact_query_operations": [list(item)
                                           for item in fact_query_operations],
                "operation_keys": [list(item) for item in operation_keys],
                "operation_contracts": [
                    [list(key), list(mode)] for key, mode in operation_contracts],
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
                # Keep the query-scoped binding projection at the result
                # boundary as well as inside the grouped Event/Time trace.
                # Production callers consume this field to verify that the
                # binding selected by the current QueryState reached the
                # ResponsePlan/generation path; omitting it here would make
                # a real binding indistinguishable from an empty match.
                "event_time_query_bindings": [
                    tree(item) for item in event_time_query_bindings],
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
             "failure_reason": failure_reason,
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
            "input_dialogue_act_match": list(input_dialogue_act_match),
            "fact_query_operations": [list(item)
                                       for item in fact_query_operations],
            "operation_keys": [list(item) for item in operation_keys],
            "operation_contracts": [
                [list(key), list(mode)] for key, mode in operation_contracts],
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
            # See the compact projection above: this is the stable result
            # contract used by production receipts and independent audits.
            "event_time_query_bindings": [
                dict(item) for item in event_time_query_bindings],
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
            scope_conjunction_bindings: tuple[tuple[tuple[int, ...], tuple[int, ...],
                                                    tuple[tuple[int, ...], ...]], ...] = (),
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
        self.input_order_witnesses = symmetric_input_order_witnesses(
            input_structure, getattr(owner, 'core_fact_index', {}))
        self.input_structure = input_structure
        self.artifact_input = artifact_input
        self.artifact_bridge_matches = tuple(artifact_bridge_matches)
        self.input_order_conflict = input_order_conflict
        self.input_required_slots_open = input_required_slots_open
        self.operation_logical_conflict = True
        self.scope_conjunction_bindings = tuple(scope_conjunction_bindings)
        self.hops: list[QueryHop] = []
        self.expanded_edges: list[tuple[int, ...]] = []
        self.open_nodes = {(node.space, node.key): node
                           for node in open_role_query_nodes(input_structure.open_relations)}
        # Keep the source-derived open candidates available when a Core frame
        # is expanded.  The Core node intentionally carries only the trained
        # frame identity; known role fillers are admitted below only after an
        # exact input-graph validation against the active fact.
        self.open_candidates_by_frame = {}
        for candidate in input_structure.open_relations:
            self.open_candidates_by_frame.setdefault(
                candidate.frame.stable_key(), []).append(candidate)
        self.open_candidates_by_frame = {
            key: tuple(sorted(values, key=lambda item: item.stable_key()))
            for key, values in self.open_candidates_by_frame.items()
        }
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
        self._discourse_hypotheses_by_key = {
            item.hypothesis_key: item for item in self.owner._discourse_hypotheses}
        observation = self.owner._input_observation_ref
        observation_key = None if observation is None else observation.stable_key()
        self._input_discourse_hypotheses = tuple(
            item for item in self.owner._discourse_hypotheses
            if observation_key is None or item.observation_key == observation_key)
        self._expected_memory_evidence = frozenset(
            self.owner._memory_evidence_by_key)
        self._memory_ohe_expansion_coverage = None
        self._memory_competition_keys = {
            item.hypothesis_key: memory_competition_identity(
                item.candidate_key, item.competition_key, item.observation_key)
            for item in self.owner._memory_hypotheses}
        self._response_contexts_by_hypothesis = {
            item.hypothesis.stable_key(): item
            for item in self.owner._response_contexts}
        self._discourse_topics_by_candidate = {
            item.candidate_key: item
            for item in self.owner._discourse_topic_candidates}
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
        self._generation_input_cache: dict[ObjectIdentity, object] = {}
        self._evidence_history_cache: dict[ObjectIdentity, tuple[object, ...]] = {}
        self._route_cache: dict[tuple[int, ...], object | None] = {}
        self._typed_projection_cache: dict[
            tuple[int, ...], TypedRelationProjection | None] = {}
        self.open_response_generations = ()
        self.generic_response_generations = ()
        self.event_time_graph_expanded_edges: list[tuple[int, ...]] = []
        self._response_generation_attempted = False
        self._reference_adoption_record_key: tuple[int, ...] = ()
        self.composition_output = None
        self._composition_verification_attempted = False

    def _generation_input(self, proposition: ObjectIdentity):
        """复用本次查询的只读 Core 生成输入，不缓存查询状态或生成结果。"""
        cache = getattr(self, "_generation_input_cache", None)
        if cache is None:
            # 兼容只装配必要字段的旧诊断/专项 fixture。
            cache = self._generation_input_cache = {}
        key = (proposition.stable_key()
               if hasattr(proposition, "stable_key") else proposition)
        item = cache.get(key)
        if item is None:
            item = self.owner.core_runtime.generation_input(proposition)
            cache[key] = item
        return item

    def _evidence_history(self, proposition: ObjectIdentity) -> tuple[object, ...]:
        """按命题复用完整 H-00 历史，保留已替代证据参与当前裁决。"""
        cache = getattr(self, "_evidence_history_cache", None)
        if cache is None:
            cache = self._evidence_history_cache = {}
        key = (proposition.stable_key()
               if hasattr(proposition, "stable_key") else proposition)
        history = cache.get(key)
        if history is None:
            history = self.owner.core_runtime.evidence_history(proposition)
            cache[key] = history
        return history

    def _route(self, proposition_key: tuple[int, ...]):
        """复用冻结图中的关系路由；未找到的路由也保留同一查询内的结果。"""
        cache = getattr(self, "_route_cache", None)
        if cache is None:
            cache = self._route_cache = {}
        route = cache.get(proposition_key)
        if route is None and proposition_key not in cache:
            route = self.owner.relation_routes.for_proposition(proposition_key)
            cache[proposition_key] = route
        return route

    def _typed_projection(self, fact: ActiveRelationSurface, route):
        """按当前只读命题和固定路由复用 typed 投影，不替代权威图对象。"""
        cache = getattr(self, "_typed_projection_cache", None)
        if cache is None:
            cache = self._typed_projection_cache = {}
        key = fact.proposition.stable_key()
        if key not in cache:
            cache[key] = project_typed_relation(
                fact, route)
        return cache[key]

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
        for item in self._input_discourse_hypotheses:
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
                            self._generation_input(fact.proposition),
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
        if not best_key:
            return False
        if best_key == second_key:
            relations = tuple(
                item for item in self.input_structure.relation_candidates
                if item.proposition_key == best_key
                and (not self.owner._active_relation_keys
                     or item.stable_key() in self.owner._active_relation_keys))
            return len({
                relation_candidate_semantic_key(self.input_structure, item)
                for item in relations
            }) > 1
        if not second_key:
            return False
        # A three-claim QueryState may rank the two endpoints of a verified
        # chain (A-C, C-B) as best/second even though they are not one direct
        # conjunction edge.  Treat the unique closed, generation-ready
        # component as the authorization boundary; unrelated ties remain
        # fail-closed.
        connected_components = _closed_scope_conjunction_components(
            self.scope_conjunction_bindings,
            self.owner.generation_ready_propositions,
        )
        if any({best_key, second_key} <= set(component)
               for component in connected_components):
            return False
        relations = tuple(
            item for item in self.input_structure.relation_candidates
            if item.proposition_key in {best_key, second_key}
            and (not self.owner._active_relation_keys
                 or item.stable_key() in self.owner._active_relation_keys))
        semantic_keys = {
            relation_candidate_semantic_key(self.input_structure, item)
            for item in relations
        }
        # Multiple provenance rows with one exact input topology are support
        # for the same observed relation.  Only a second topology is a real
        # ambiguity; the original rows remain in the QueryState evidence.
        if _surface_equivalent_relation_tie(self.input_structure, relations):
            return False
        return len(semantic_keys) > 1

    @staticmethod
    def _reference_adoption_record(
            resolution: CrossTurnReferenceResolution,
            fact: ActiveRelationSurface,
            candidate: InputRelationCandidate,
            open_candidate,
            target: ObjectIdentity,
            ) -> tuple[int, ...]:
        """可逆地绑定完整指代支持、当前输入与 Core；不代替正向事实证据。"""
        fields = (
            resolution.stable_key(),
            fact.proposition.stable_key(),
            fact.predicate.stable_key(),
            candidate.stable_key(),
            open_candidate.stable_key(),
            resolution.adoption_support_key,
            target.stable_key(),
        )
        result = [_REFERENCE_TARGET_ADOPTION_VERSION, len(fields)]
        for field in fields:
            result.extend((len(field), *field))
        return tuple(result)

    def _validated_reference_target_adoption(
            self,
            state: QueryState,
            ) -> tuple[CrossTurnReferenceResolution, ActiveRelationSurface,
                       InputRelationCandidate, object,
                       ObjectIdentity] | None:
        """Return one exact REFERS winner/fact/input join, or ``None``.

        A resolved Memory winner is not sufficient by itself.  This gate
        joins it to the active REFERS proposition, its trained open frame,
        the current input's subject/predicate spans, and the missing target
        role.  No source text or target filler is recovered here.
        """
        del state  # Resolution status is already materialized on the owner.
        required = tuple(item for item in self.owner._reference_resolutions
                         if item.requires_resolution)
        if len(required) != 1:
            return None
        resolution = required[0]
        if (resolution.status != REFERENCE_RESOLVED
                or any(item.status != REFERENCE_RESOLVED
                       for item in self.owner._reference_resolutions
                       if item.requires_resolution)
                or any(item.status == REFERENCE_CONFLICT
                       for item in self.owner._reference_resolutions)):
            return None
        supporters = resolution.supporting_candidates
        target = resolution.winner_target
        if (not supporters or target is None or not resolution.adoption_support_key
                or resolution.winner_target_key != target.stable_key()):
            return None

        slot_bindings = tuple(
            item for item in getattr(
                self.input_structure, "relation_question_bindings", ())
            if item.answer_role_key)
        if len(slot_bindings) == 1:
            binding = slot_bindings[0]
            slot_parts = self.owner._relation_slot_question_parts(
                self.input_structure, binding)
            if slot_parts is not None:
                frame, relation, fact, _answer_filler, dynamic_roles = slot_parts
                reference = resolution.candidate_set.reference
                role_key = reference.role.stable_key()
                relation_map = self.owner._input_open_relations_by_hypothesis
                open_candidate = relation_map.get(
                    reference.hypothesis.stable_key())
                if (open_candidate is None
                        or open_candidate not in self.input_structure.open_relations
                        or role_key not in dynamic_roles
                        or (role_key not in frame.roles
                            and not (frame.is_reference_slot
                                     and role_key == frame.answer_role_key))
                        or reference.ordinal >= len(open_candidate.bindings)
                        or len(reference.source_ref) < 2):
                    return None
                try:
                    reference_source = SourceRef.from_stable_key(
                        reference.source_ref[1:])
                except (TypeError, ValueError):
                    return None
                dynamic = tuple(
                    item for item in open_candidate.bindings
                    if item.role == role_key
                    and item.ordinal == reference.ordinal
                    and item.values == reference.values
                    and item.source_ref == self.input_structure.source_ref)
                if len(dynamic) != 1:
                    return None
                role_fillers = {
                    item.role.stable_key(): item.filler
                    for item in fact.bindings}
                if role_fillers.get(role_key) != target:
                    return None
                try:
                    expected_occurrence = occurrence_identity(
                        reference_source, start=dynamic[0].start,
                        end=dynamic[0].end, ordinal=0)
                    expected_origin = span_identity(
                        reference_source,
                        members=((dynamic[0].start, dynamic[0].end),))
                except (TypeError, ValueError):
                    return None
                if (reference.occurrence != expected_occurrence
                        or reference.origin != expected_origin
                        or relation.proposition_key
                        != fact.proposition.stable_key()
                        or relation.predicate_key
                        != frame.predicate_key
                        or open_candidate.frame.proposition
                        != relation.proposition_key
                        or open_candidate.frame.predicate
                        != frame.predicate_key
                        or (not frame.is_reference_slot
                            and (open_candidate.frame.roles != frame.roles
                                 or open_candidate.frame.gaps != frame.gaps))
                        or (frame.is_reference_slot
                            and (frame.answer_role_key
                                 not in open_candidate.frame.roles
                                 or {role for role in open_candidate.frame.roles
                                     if role != frame.answer_role_key}
                                 != set(frame.roles)))):
                    return None
                return (resolution, fact, relation, open_candidate, target)

        predicate = authored_relation_identity(RELATION_REFERS)
        from_role = authored_relation_role_identity(ROLE_REFERS_FROM)
        to_role = authored_relation_role_identity(ROLE_REFERS_TO)
        predicate_key = predicate.stable_key()
        from_role_key = from_role.stable_key()
        to_role_key = to_role.stable_key()
        reference = resolution.candidate_set.reference
        # The current input occurrence is the missing target role.  The
        # observed ``from`` role authorizes the REFERS frame, while the
        # unresolved ``to`` role is the occurrence being resolved/adopted.
        if reference.role != to_role:
            return None
        if len(reference.source_ref) < 2:
            return None
        try:
            reference_source = SourceRef.from_stable_key(
                reference.source_ref[1:])
        except (TypeError, ValueError):
            return None

        facts = tuple(item for item in self.owner.facts
                      if item.predicate == predicate)
        matches = []
        for fact in facts:
            fact_bindings = {item.role: item.filler for item in fact.bindings}
            if (len(fact_bindings) != len(fact.bindings)
                    or set(fact_bindings) != {from_role, to_role}
                    or fact_bindings[to_role] != target):
                continue
            proposition_key = fact.proposition.stable_key()
            frame_candidates = tuple(
                item for item in self.input_structure.open_relations
                if (item.frame.proposition == proposition_key
                    and item.frame.predicate == predicate_key
                    and item.frame.source_ref == (fact.source_hash,)
                    and item.frame.roles == (from_role_key, to_role_key)
                    and item.source_ref == self.input_structure.source_ref
                    and item.values == self.input_structure.token_values))
            if len(frame_candidates) != 1:
                continue
            open_candidate = frame_candidates[0]
            by_role = {
                role: tuple(item for item in open_candidate.bindings
                            if item.role == role)
                for role in (from_role_key, to_role_key)
            }
            if any(len(by_role[role]) != 1 for role in (from_role_key, to_role_key)):
                continue
            from_span = by_role[from_role_key][0]
            target_span = by_role[to_role_key][0]
            if (from_span.source_ref != open_candidate.source_ref
                    or target_span.source_ref != open_candidate.source_ref
                    or target_span.ordinal != 1
                    or reference.ordinal != target_span.ordinal
                    or reference.values != target_span.values):
                continue
            try:
                expected_occurrence = occurrence_identity(
                    reference_source, start=target_span.start,
                    end=target_span.end, ordinal=0)
                expected_origin = span_identity(
                    reference_source,
                    members=((target_span.start, target_span.end),))
            except (TypeError, ValueError):
                continue
            if (reference.occurrence != expected_occurrence
                    or reference.origin != expected_origin):
                continue

            relation_candidates = tuple(
                item for item in self.input_structure.relation_candidates
                if (item.proposition_key == proposition_key
                    and item.predicate_key == predicate_key
                    and item.state in {STRUCTURE_OPEN, STRUCTURE_CLOSED}
                    and item.conflict_count == 0
                    and item.predicate_coverage == 1
                    and item.required_count == len(fact.bindings) + 1
                    and ((item.state == STRUCTURE_OPEN
                          and item.support_count == item.role_coverage + 1
                          and item.required_open == 1)
                         or (item.state == STRUCTURE_CLOSED
                             and item.support_count == item.required_count
                             and item.required_open == 0
                             and item.role_coverage == len(fact.bindings)))))
            if len(relation_candidates) != 1:
                continue
            relation = relation_candidates[0]

            fillers = tuple(
                item for item in self.input_structure.semantic_candidates
                if (item.proposition_key == proposition_key
                    and item.predicate_key == predicate_key
                    and item.projection_kind == PROJECTION_FILLER))
            predicates = tuple(
                item for item in self.input_structure.semantic_candidates
                if (item.proposition_key == proposition_key
                    and item.predicate_key == predicate_key
                    and item.projection_kind == PROJECTION_PREDICATE))
            from_fillers = tuple(item for item in fillers
                                 if item.role_key == from_role_key)
            target_fillers = tuple(item for item in fillers
                                   if item.role_key == to_role_key)
            from_span_refs = tuple(
                item.ref_key for item in self.input_structure.spans
                if (item.start == from_span.start
                    and item.end == from_span.end
                    and item.values == from_span.values
                    and item.source_ref == self.input_structure.source_ref))
            if (len(from_fillers) != 1 or target_fillers
                    or len(predicates) != 1
                    or from_fillers[0].candidate_key
                    != fact_bindings[from_role]
                    .stable_key()
                    or len(from_span_refs) != 1
                    or from_fillers[0].span_ref != from_span_refs[0]
                    or predicates[0].span_ref not in relation.span_refs
                    or from_fillers[0].span_ref not in relation.span_refs):
                continue
            if any(item.candidate_key == target.stable_key()
                   and item.projection_kind == PROJECTION_FILLER
                   for item in self.input_structure.semantic_candidates):
                continue
            if tuple(sorted({from_fillers[0].span_ref, predicates[0].span_ref})) \
                    != relation.span_refs:
                continue
            matches.append((resolution, fact, relation, open_candidate, target))
        if len(matches) != 1:
            return None
        return matches[0]

    def _reference_target_adoption_closed(self, state: QueryState) -> bool:
        """Check that a previously emitted adoption marker remains exact."""
        if self._reference_adoption_record_key:
            record = self._reference_adoption_record_key
            marker_key = (*_REFERENCE_TARGET_ADOPTION_EVIDENCE, *record)
            if any(
                    item.space == _CORE
                    and item.evidence_key == marker_key
                    and item.payload_key == record
                    and item.polarity == 3
                    for item in state.evidence):
                return True
        parts = self._validated_reference_target_adoption(state)
        if parts is None:
            return False
        _resolution, fact, candidate, open_candidate, target = parts
        record = self._reference_adoption_record(
            _resolution, fact, candidate, open_candidate, target)
        marker_key = (*_REFERENCE_TARGET_ADOPTION_EVIDENCE, *record)
        return any(
            item.space == _CORE
            and item.evidence_key == marker_key
            and item.payload_key == record
            and item.polarity == 3
            for item in state.evidence)

    def _materialized_reference_target_adoption_proposition(
            self, state: QueryState,
            ) -> tuple[int, ...] | None:
        """Recover the proposition from this QueryState's exact adoption proof."""
        record = self._reference_adoption_record_key
        if (not record or not self._reference_target_adoption_closed(state)
                or len(record) < 2
                or record[0] != _REFERENCE_TARGET_ADOPTION_VERSION
                or record[1] != 7):
            return None
        fields = []
        offset = 2
        for _ in range(record[1]):
            if offset >= len(record):
                return None
            width = record[offset]
            offset += 1
            end = offset + width
            if width <= 0 or end > len(record):
                return None
            fields.append(record[offset:end])
            offset = end
        if offset != len(record) or len(fields) != 7:
            return None

        proposition_key, predicate_key = fields[1], fields[2]
        fact = self.owner.core_fact_index.get(proposition_key)
        if (fact is None
                or fact.proposition.stable_key() != proposition_key
                or fact.predicate.stable_key() != predicate_key
                or proposition_key not in self.owner.dialogue_frame_index):
            return None
        resolutions = tuple(
            item for item in self.owner._reference_resolutions
            if item.requires_resolution)
        if len(resolutions) != 1:
            return None
        resolution = resolutions[0]
        if (resolution.status != REFERENCE_RESOLVED
                or resolution.stable_key() != fields[0]
                or resolution.adoption_support_key != fields[5]
                or resolution.winner_target_key != fields[6]
                or any(item.status == REFERENCE_CONFLICT
                       for item in self.owner._reference_resolutions)):
            return None
        if not any(
                item.space == _CORE
                and item.role_key == _REFERENCE_TARGET_ADOPTION_ROLE
                and item.filler_key == record
                for item in state.bindings):
            return None
        return proposition_key

    def _append_adopted_dialogue_root(
            self,
            state: QueryState,
            fact: ActiveRelationSurface,
            ) -> QueryState:
        """Add the adopted proposition as a normal Dialogue/Core frontier root."""
        proposition = fact.proposition.stable_key()
        frame = self.owner.dialogue_frame_index.get(proposition)
        if frame is None:
            return state
        source_ref = (frame.source_hash,)
        roots = list(state.roots)
        anchors = list(state.anchors)
        frontier = list(state.frontier)
        if not any(item.owner_space == _DIALOGUE
                   and item.root_key == proposition for item in roots):
            roots.append(QueryRoot(
                _DIALOGUE, proposition, ROOT_PENDING,
                self.owner.space_weights[2], source_ref=source_ref))
            anchors.append(QueryAnchor(_DIALOGUE, proposition, kind=3))
            frontier.append(FrontierEntry(
                (_DIALOGUE, *proposition), _DIALOGUE,
                target_key=proposition, root_key=proposition,
                source_ref=source_ref,
                direction=1, owner_weight=self.owner.space_weights[2],
                depth=0, required_slot_gain=3, evidence_support=1))
        return state.with_(
            roots=tuple(sorted(roots, key=lambda item: item.stable_key())),
            anchors=tuple(sorted(anchors, key=lambda item: item.stable_key())),
            frontier=tuple(sorted(frontier,
                                  key=lambda item: item.priority_tuple(),
                                  reverse=True)),
        )

    def _adopt_reference_target(self, state: QueryState) -> QueryState:
        """Materialize one validated REFERS target into the shared QueryState."""
        if self._reference_target_adoption_closed(state):
            return state
        parts = self._validated_reference_target_adoption(state)
        if parts is None:
            if any(item.requires_resolution
                   and item.status == REFERENCE_RESOLVED
                   for item in self.owner._reference_resolutions):
                self.owner._diagnostic_stage(
                    "reference_target_adoption:resolved_but_unjoined")
            return state
        resolution, fact, relation, open_candidate, target = parts
        record = self._reference_adoption_record(
            resolution, fact, relation, open_candidate, target)
        marker_key = (*_REFERENCE_TARGET_ADOPTION_EVIDENCE, *record)
        source = self._generation_input(fact.proposition)
        source_scope = source.hypothesis.scope.stable_key()
        target_key = target.stable_key()
        fact_key = fact.proposition.stable_key()
        resolved_role = resolution.candidate_set.reference.role.stable_key()
        dynamic_binding = tuple(
            item for item in open_candidate.bindings
            if item.role == resolved_role
            and item.ordinal == resolution.candidate_set.reference.ordinal
            and item.values == resolution.candidate_set.reference.values
            and item.source_ref == self.input_structure.source_ref)
        if len(dynamic_binding) != 1:
            return state
        dynamic = dynamic_binding[0]
        matching_spans = tuple(
            item for item in self.input_structure.spans
            if (item.start == dynamic.start
                and item.end == dynamic.end
                and item.values == dynamic.values
                and item.source_ref == self.input_structure.source_ref))
        source_spans = tuple(
            item for item in matching_spans if item.span_kind == SPAN_SOURCE)
        if len(source_spans) > 1:
            return state
        dynamic_span = (
            source_spans[0] if source_spans else
            InputSpan(self.input_structure.source_ref, dynamic.start,
                      dynamic.end, SPAN_SOURCE, dynamic.values))
        dynamic_span_ref = dynamic_span.ref_key
        fact_role_bindings = tuple(
            (index, item) for index, item in enumerate(fact.bindings)
            if item.role.stable_key() == resolved_role)
        if len(fact_role_bindings) != 1:
            return state
        span_ref = dynamic_span_ref
        member_ordinal, fact_role = fact_role_bindings[0]
        projection = InputSemanticCandidate(
            target.stable_key(), target.object_kind, PROJECTION_FILLER,
            span_ref, fact.proposition.stable_key(), fact.predicate.stable_key(),
            resolved_role, relation.source_ref, member_ordinal)
        bindings = list(state.bindings)
        evidence = list(state.evidence)
        if not any(item.space == _CORE and item.role_key == resolved_role
                   and item.filler_key == target_key
                   and item.scope_key == source_scope for item in bindings):
            bindings.append(BindingEntry(
                resolved_role, target_key,
                space=_CORE, scope_key=source_scope))
        if not any(item.space == _CORE and item.role_key == _REFERENCE_TARGET_ADOPTION_ROLE
                   and item.filler_key == record for item in bindings):
            bindings.append(BindingEntry(
                _REFERENCE_TARGET_ADOPTION_ROLE, record,
                space=_CORE, scope_key=source_scope))
        if not any(item.space == _CORE and item.evidence_key == marker_key
                   for item in evidence):
            evidence.append(EvidenceEntry(
                (fact.source_hash, *source.proposition.definition.source.stable_key()),
                fact_key, space=_CORE, polarity=3, trust=1,
                evidence_key=marker_key, scope_key=source_scope,
                payload_key=record))
        core_history = core_query_evidence(
            fact.proposition,
            self._evidence_history(fact.proposition),
            self.owner.core_runtime.context.scoped_identity_store.registry)
        evidence.extend(core_history)

        replacement = replace(
            relation,
            span_refs=tuple(sorted({*relation.span_refs, span_ref})),
            support_count=relation.required_count,
            role_coverage=len(fact.bindings),
            predicate_coverage=1,
            required_open=0,
            state=STRUCTURE_CLOSED,
            conflict_count=0,
        )
        spans = tuple(sorted({
            *self.input_structure.spans,
            dynamic_span,
        }, key=lambda item: item.stable_key()))
        precedes = []
        contains = []
        for left in spans:
            for right in spans:
                if left == right:
                    continue
                if left.end <= right.start:
                    precedes.append(InputSpanRelation(
                        left.ref_key, right.ref_key))
                if (left.start <= right.start and right.end <= left.end
                        and (left.start, left.end)
                        != (right.start, right.end)):
                    contains.append(InputSpanRelation(
                        left.ref_key, right.ref_key))
        self.input_structure = replace(
            self.input_structure,
            spans=spans,
            precedes=tuple(sorted(
                set(precedes), key=lambda item: item.stable_key())),
            contains=tuple(sorted(
                set(contains), key=lambda item: item.stable_key())),
            semantic_candidates=tuple(sorted({
                *self.input_structure.semantic_candidates, projection,
            }, key=lambda item: item.stable_key())),
            relation_candidates=tuple(sorted(
                (replacement if item == relation else item
                 for item in self.input_structure.relation_candidates),
                key=lambda item: item.stable_key())),
            relation_question_bindings=tuple(sorted((
                replace(item, relation_key=replacement.stable_key())
                if item.relation_key == relation.stable_key() else item
                for item in self.input_structure.relation_question_bindings
            ), key=lambda item: item.stable_key())),
        )
        active = set(self.owner._active_relation_keys)
        active.discard(relation.stable_key())
        active.add(replacement.stable_key())
        self.owner._active_relation_keys = frozenset(active)
        self.input_required_slots_open = 0
        propositions = tuple(sorted({
            item.proposition_key
            for item in self.input_structure.relation_candidates
            if item.stable_key() in self.owner._active_relation_keys
        }))
        best, second = self.owner._rank_candidates(
            self.input_structure, propositions,
            active_relation_keys=self.owner._active_relation_keys)
        state = state.with_(
            bindings=self._unique_bindings(bindings),
            evidence=self._unique_evidence(evidence),
            required_slots_open=0,
            best_score=best[0],
            second_score=second[0],
            best_candidate_key=best[1],
            second_candidate_key=second[1],
        )
        if self._reference_adoption_record_key != record:
            self._reference_adoption_record_key = record
            self.owner._diagnostic_stage(
                "reference_target_adoption:materialized")
            if not any(item.space == _CORE and item.proposition == fact_key
                       for item in self.hops):
                self.hops.append(QueryHop(
                    _CORE, fact_key, fact.predicate.stable_key(),
                    fact.bindings[0].filler.stable_key(),
                    fact.bindings[1].filler.stable_key(), (),
                    state.depth + 1, fact.source_hash,
                    tuple(item.filler.stable_key() for item in fact.bindings)))
        return self._append_adopted_dialogue_root(state, fact)

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
        pending_proofs = pending_composition_query_evidence(
            tuple(item for item in self.owner.pending_relation_compositions
                  if item.stable_key() in {binding.pending_key
                      for binding in self.input_structure.pending_relation_composition_bindings}), state)
        if pending_proofs:
            state = state.with_(evidence=self._unique_evidence([
                *state.evidence, *pending_proofs]))
        if not (self._reference_adoption_record_key
                and self._reference_target_adoption_closed(state)):
            state = self.owner._refresh_reference_resolutions(
                state, self.input_structure)
        state = self._adopt_reference_target(state)
        state = self._bind_open_generation_contexts(state)
        frontier_exhausted = not unvisited_frontier(
            state.frontier, state.visited)
        dynamic_slot_blocked = self.owner._dynamic_slot_reference_blocked(
            self.input_structure, frontier_exhausted=frontier_exhausted)
        self.owner._dynamic_slot_query_blocked = dynamic_slot_blocked
        if (not self._composition_verification_attempted
                and self.input_structure.pending_relation_composition_bindings
                and self.owner.relation_composition_verification
                and frontier_exhausted
                and all(root.status != ROOT_PENDING for root in state.roots)):
            self._composition_verification_attempted = True
            requested = {item.pending_key
                         for item in self.input_structure.pending_relation_composition_bindings}
            candidates = tuple(item for item in self.owner.pending_relation_compositions
                               if item.stable_key() in requested)
            if len(candidates) == 1:
                state, self.composition_output = consume_verified_composition(
                    self.owner.core_runtime, self.owner.relation_composition_witnesses,
                    candidates[0], state, self.owner.surface_generator,
                    self.owner.relation_composition_budget)
        # A conflicting input cannot authorize either an open-role or generic
        # response.  Running the six-layer surface planner here would spend
        # unbounded work enumerating every competing connector even though the
        # final contract must be CLARIFY/FAIL_CLOSED.  Keep the shared graph
        # evidence and terminate through the ordinary conflict predicate.
        scoped_nonfactual_ready = assertion_scope_response_allowed(self.input_structure, state)
        projection_conflict_ready = retained_projection_conflict_allowed(
            state, self.input_structure, restricted=True)
        response_generation_blocked = bool(
            (self.input_order_conflict and not projection_conflict_ready) or (state.conflict_open
                and not (scoped_nonfactual_ready or projection_conflict_ready)))
        context_only_dynamic = bool(
            dynamic_slot_blocked and unresolved_role_context_allowed(state, self.input_structure))
        if (not self._response_generation_attempted
                and frontier_exhausted
                and all(root.status != ROOT_PENDING for root in state.roots)
                and (not dynamic_slot_blocked or context_only_dynamic)):
            self._response_generation_attempted = True
            if response_generation_blocked:
                self.owner._diagnostic_stage(
                    "response_generation_deferred:conflict")
                self.open_response_generations = ()
                self.generic_response_generations = ()
            elif (not self.input_structure.assertion_scope_bindings
                  and not dynamic_slot_blocked and not projection_conflict_ready):
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
            # A uniquely supported Core proposition already has a trained
            # Dialogue realization path.  Do not build the generic UNKNOWN
            # connector for that same turn: the generic planner restores the
            # full R-01 closure and can dominate the query cost even though
            # its candidate is discarded once the factual claim closes.  The
            # gate is structural evidence only; unambiguous open and
            # unsupported inputs still take the generic path. Conflicts were
            # short-circuited above because their contract is fail-closed.
            core_fact_route_ready = bool(
                not self.input_structure.assertion_scope_bindings
                and not self.input_structure.discourse_compositions
                and not self.input_structure.discourse_gaps
                and not self.input_structure.pending_relation_composition_bindings
                and state.best_candidate_key
                and state.best_candidate_key in self.owner.core_fact_index
                and state.best_candidate_key in self.owner.dialogue_frame_index
                and state.required_slots_open == 0
                and state.conflict_open == 0
                and any(
                    item.space == _CORE
                    and item.polarity == 1
                    and item.hypothesis_key == state.best_candidate_key
                    for item in state.evidence))
            if response_generation_blocked:
                self.generic_response_generations = ()
            elif self.input_structure.operation_request_bindings:
                # 完整绑定请求先执行整个程序；部分Core候选既不授权回答，也不屏蔽UNKNOWN。
                self.owner._diagnostic_stage("generic_response_deferred:bound_operation_request")
                self.generic_response_generations = ()
            elif core_fact_route_ready:
                self.owner._diagnostic_stage(
                    "generic_response_deferred:closed_core_candidate")
                self.generic_response_generations = ()
            else:
                from pure_integer_ai.experiments.operation_request_scope_query import pending_request_context_allowed
                self.generic_response_generations = generate_generic_responses(
                    self.owner.surface_generator,
                    state,
                    self.generic_hypotheses,
                    tuple(self.generic_response_nodes.values()),
                    self.owner._input_observation_ref,
                    self.input_structure,
                    self.owner._discourse_graph_input_requested_keys,
                    tuple(self.hops),
                    context_only=(projection_conflict_ready or context_only_dynamic or pending_request_context_allowed(
                        state, self.input_structure, self.owner.operation_request_frames)),
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
            self.owner._diagnostic_stage(
                f"dialogue_roots:{len(dialogue_roots)}")
            if dialogue_roots and not dynamic_slot_blocked and not self.input_structure.assertion_scope_bindings:
                self.owner._prepare_generation_plans(dialogue_roots, state)
                # A tied conjunction cannot generate during the first
                # Dialogue-root visit because margin closure is intentionally
                # false.  Once all roots have converged, materialize the
                # trained Dialogue evidence for every successfully generated
                # claim into this same QueryState.  This is graph evidence,
                # not a source-surface replay.
                dialogue_evidence = []
                for proposition in dialogue_roots:
                    generated = self.owner.generation_by_proposition.get(
                        proposition)
                    fact = self.owner.core_fact_index.get(proposition)
                    frame = self.owner.dialogue_frame_index.get(proposition)
                    if generated is None or fact is None or frame is None:
                        continue
                    try:
                        if proposition in self.owner.kdconv_fact_index:
                            # KdConv facts have no Core H-00 generation input;
                            # preserve their integer source/evidence contract
                            # instead of passing the source scope tuple to the
                            # typed dialogue evidence projector.
                            source_scope = (fact.source_hash,)
                            dialogue_evidence.append(EvidenceEntry(
                                (fact.source_hash,), proposition,
                                space=_DIALOGUE, polarity=1, trust=1,
                                evidence_key=(91630, 5, *proposition),
                                scope_key=source_scope,
                                payload_key=(91630, 6, *proposition)))
                        else:
                            source = self._generation_input(frame.proposition)
                            dialogue_evidence.append(dialogue_query_evidence(
                                fact.proposition, frame, source, generated))
                    except (RuntimeError, TypeError, ValueError):
                        continue
                if dialogue_evidence:
                    state = state.with_(evidence=self._unique_evidence([
                        *state.evidence, *dialogue_evidence]))
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
        self.owner._diagnostic_stage(
            f"core_evidence_counts:{sum(item.space == _CORE for item in current_evidence)}:"
            f"support={sum(item.space == _CORE and item.polarity == 1 for item in current_evidence)}:"
            f"indexed={sum(item.space == _CORE and item.polarity == 1 and item.hypothesis_key in self.owner.core_fact_index for item in current_evidence)}")
        dialogue_support = {
            item.hypothesis_key for item in current_evidence
            if item.space == _DIALOGUE and item.polarity == 1
        }
        expected_memory_evidence = self._expected_memory_evidence
        bound_memory_evidence = {
            item.filler_key for item in state.bindings
            if (item.space == _MEMORY
                and item.role_key == (MEMORY_OBJECT_EVIDENCE,))
        }
        memory_complete = (
            not expected_memory_evidence
            or expected_memory_evidence <= bound_memory_evidence)
        selected = state.best_candidate_key
        partial_property_proof = (
            bool(selected)
            and selected in self.owner._partial_property_core_roots
            and selected in core_support
            and selected in dialogue_support
            and selected in self.owner.generation_ready_propositions
            and memory_complete)
        closed = (
            bool(selected)
            and selected in core_support
            and selected in dialogue_support
            and selected in self.owner.generation_ready_propositions
            and memory_complete
            and (partial_property_proof
                 or selected in self.owner._kdconv_query_propositions
                 or self.input_structure.concept_evidence_closed(
                     selected, frozenset(core_support)))
        )
        slot_binding = (
            self.input_structure.relation_question_bindings[0]
            if (len(self.input_structure.relation_question_bindings) == 1
                and self.input_structure.relation_question_bindings[0].answer_role_key)
            else None)
        if slot_binding is not None and selected:
            slot_parts = self.owner._relation_slot_question_parts(
                self.input_structure, slot_binding)
            if slot_parts is not None:
                (_slot_frame, slot_relation, slot_fact, answer_filler,
                 _dynamic_roles) = slot_parts
                source = self._generation_input(slot_fact.proposition)
                answer_binding_present = any(
                    item.space == _CORE
                    and item.role_key == slot_binding.answer_role_key
                    and item.filler_key == answer_filler
                    and item.scope_key == source.hypothesis.scope.stable_key()
                    for item in state.bindings)
                visible_bindings_present = all(any(
                    item.space == _CORE
                    and item.role_key == role
                    and item.filler_key == filler
                    and item.scope_key == source.hypothesis.scope.stable_key()
                    for item in state.bindings)
                    for role, filler in ((
                        semantic.role_key, semantic.candidate_key)
                        for semantic in self.input_structure.semantic_candidates
                        if semantic.proposition_key == slot_relation.proposition_key
                        and semantic.projection_kind == PROJECTION_FILLER))
                slot_closed = (
                    selected == slot_relation.proposition_key
                    and selected in core_support
                    and selected in dialogue_support
                    and selected in self.owner.generation_ready_propositions
                    and memory_complete
                    and answer_binding_present
                    and visible_bindings_present)
                closed = closed or slot_closed
        adoption = self._validated_reference_target_adoption(state)
        adoption_key = (None if adoption is None
                        else adoption[1].proposition.stable_key())
        if adoption_key is None:
            adoption_key = (
                self._materialized_reference_target_adoption_proposition(state))
        adoption_closed = bool(
            adoption_key is not None
            and self._reference_target_adoption_closed(state))
        if selected and adoption_closed and selected == adoption_key:
            # REFERS target adoption has no independent target projection, so
            # the ordinary span-chain closure cannot authorize it.  The
            # adoption marker is accepted only together with the current
            # positive Core/Dialogue support and generated realization.
            closed = bool(
                selected in core_support
                and selected in dialogue_support
                and selected in self.owner.generation_ready_propositions
                and memory_complete)
        conjunction_claim_sets = _closed_scope_conjunction_components(
            self.scope_conjunction_bindings,
            self.owner.generation_ready_propositions,
        )
        conjunction_closed = bool(
            len(conjunction_claim_sets) == 1
            and set(conjunction_claim_sets[0]) <= core_support
            and set(conjunction_claim_sets[0]) <= dialogue_support
            and set(conjunction_claim_sets[0])
            <= self.owner.generation_ready_propositions)
        self.owner._diagnostic_stage(
            f"closure_support:core={len(core_support)}:dialogue={len(dialogue_support)}:"
            f"ready={len(self.owner.generation_ready_propositions)}:"
            f"components={len(conjunction_claim_sets)}:closed={int(conjunction_closed)}")
        # A composition query is authorized only by its complete connected
        # claim component.  Do not let the ordinary best-candidate predicate
        # close one member early; that would terminate before all Dialogue
        # roots can materialize their trained generation evidence.
        if (self.input_structure.discourse_compositions
                or self.input_structure.discourse_gaps):
            closed = conjunction_closed
        else:
            closed = closed or conjunction_closed
        reference_open = any(
            item.requires_resolution and item.status != REFERENCE_RESOLVED
            for item in self.owner._reference_resolutions)
        reference_conflict = any(
            item.status == REFERENCE_CONFLICT
            for item in self.owner._reference_resolutions)
        # A dynamic WH role has an additional closure contract: the observed
        # surface must be joined to one resolved Memory reference before the
        # Core fact can authorize generation.  Keep the ordinary reference
        # statuses untouched while exposing this query-local gate to the
        # shared termination fields.
        reference_open = reference_open or dynamic_slot_blocked
        closed = closed and not reference_open and not reference_conflict
        if self.input_structure.pending_relation_composition_bindings:
            closed = (composition_output_is_current(self.composition_output, state)
                      and memory_complete and not reference_open and not reference_conflict)
        # A current-turn Memory refutation can disagree with a Core fact even
        # when the two spaces use different Hypothesis identities.  Join only
        # through the exact closed input relation route and the current
        # Observation; historical Memory rows remain unrelated context.
        current_memory_refutes_core = (
            self._current_memory_refutes_core_support(
                current_evidence, core_support))
        competition_keys = self._memory_competition_keys
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
        unresolved_input_conflicts = bool(unresolved_input_conflict_bindings(state, self.input_order_witnesses))
        self.operation_logical_conflict = bool(self.input_order_conflict or reference_conflict
            or unresolved_input_conflicts
            or evidence_conflict or current_memory_refutes_core or reference_open)
        conflict_open = int(
            self.input_order_conflict
                or reference_conflict
            or unresolved_input_conflicts
            or evidence_conflict
            or current_memory_refutes_core
            or (bool(state.second_candidate_key)
                and state.best_score > 0
                and state.best_score == state.second_score
                and self._distinct_relation_tie(state))
        )
        # 多个物理来源可以各自保留独立命题身份，但当它们已经在本轮
        # 完整消费为同一 predicate/role/span/object topology 时，竞争分数
        # 只表示 provenance 行数，不表示两个答案。终止谓词没有来源聚合
        # 语义，因此在所有根收敛、证据与生成均闭合且无冲突之后，把这一
        # 个查询局部的次佳分数归零；证据、ResponsePlan 和命题身份仍逐项
        # 保留。不同拓扑不会进入该门，继续走 margin/冲突 fail-closed。
        surface_tie_margin_resolved = False
        if (frontier_exhausted
                and all(root.status != ROOT_PENDING for root in state.roots)
                and closed
                and not conflict_open
                and state.best_candidate_key
                and state.second_candidate_key
                and state.best_score == state.second_score):
            tied_keys = {
                state.best_candidate_key, state.second_candidate_key}
            tied_relations = tuple(
                item for item in self.input_structure.relation_candidates
                if item.proposition_key in tied_keys
                and (not self.owner._active_relation_keys
                     or item.stable_key() in self.owner._active_relation_keys))
            surface_tie_margin_resolved = _surface_equivalent_relation_tie(
                self.input_structure, tied_relations)
            if surface_tie_margin_resolved:
                self.owner._diagnostic_stage(
                    "surface_tie_margin_resolved:provenance_rows")
        effective_second_score = (
            0 if surface_tie_margin_resolved else state.second_score)
        required_slots_open = int(
            (self.input_required_slots_open and not conjunction_closed
             and self.composition_output is None)
            or not core_support or reference_open)
        if self.input_structure.assertion_scope_bindings:
            closed = False
        result = state.with_(
            required_slots_open=required_slots_open,
            evidence_closed=1 if closed else 0,
            conflict_open=conflict_open,
            best_score=state.best_score,
            second_score=effective_second_score,
            marginal_gain=marginal_gain,
            generation_ready=1 if closed else 0,
        )
        self.owner._diagnostic_stage(
            f"query_status:input_slots={self.input_required_slots_open}:"
            f"reference_open={int(reference_open)}:"
            f"reference_conflict={int(reference_conflict)}:"
            f"core={len(core_support)}:dialogue={len(dialogue_support)}:"
            f"ready={len(self.owner.generation_ready_propositions)}:"
            f"components={len(conjunction_claim_sets)}:"
            f"conjunction_closed={int(conjunction_closed)}:"
            f"memory_complete={int(memory_complete)}:closed={int(closed)}:"
            f"required={result.required_slots_open}:"
            f"evidence={result.evidence_closed}:"
            f"conflict={result.conflict_open}:"
            f"generation={result.generation_ready}")
        return result

    def _current_memory_refutes_core_support(
            self,
            current_evidence: tuple[EvidenceEntry, ...],
            core_support: set[tuple[int, ...]],
            ) -> bool:
        """Detect a same-turn cross-space refutation for a closed relation.

        Memory relation candidates are namespaced by their full structural
        route, while Core evidence is keyed by the proposition identity.  The
        input relation is the only allowed bridge between those namespaces.
        """
        observation = self.owner._input_observation_ref
        if observation is None or not core_support:
            return False
        observation_key = observation.stable_key()
        proposition_by_candidate = {
            memory_relation_candidate_key(relation): relation.proposition_key
            for relation in self.input_structure.relation_candidates
            if (relation.state == STRUCTURE_CLOSED
                and relation.required_count >= 2
                and relation.required_count == relation.support_count)
        }
        if not proposition_by_candidate:
            return False
        current_hypotheses = {
            item.hypothesis_key: proposition_by_candidate[item.candidate_key]
            for item in self.owner._memory_hypotheses_by_observation.get(
                observation_key, ())
            if item.candidate_key in proposition_by_candidate
        }
        if not current_hypotheses:
            return False
        return any(
            item.space == _MEMORY
            and item.polarity == EVIDENCE_REFUTE
            and item.hypothesis_key in current_hypotheses
            and current_hypotheses[item.hypothesis_key] in core_support
            for item in current_evidence)

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
        projection = self._typed_projection(fact, route)
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

    def _record_open_relation_support(
            self, frame, fact, bindings: list[BindingEntry],
            evidence: list[EvidenceEntry],
            ) -> bool:
        """Admit only exact known-role support for an open Core relation.

        An open frame may contain one or more dynamic spans (for example the
        object in ``词项甲指向什么？``).  The frame itself proves only that a
        trained relation schema was recognized.  A Core proposition becomes
        supportable here only when every *known* input role has exactly one
        filler projection and that filler is the corresponding active fact
        filler.  Unknown roles stay unbound; ambiguous or conflicting
        projections return ``False`` and leave the query UNKNOWN.
        """
        candidates = self.open_candidates_by_frame.get(frame.stable_key(), ())
        if len(candidates) != 1:
            return False
        candidate = candidates[0]
        relation_candidates = tuple(
            item for item in self.input_structure.relation_candidates
            if item.proposition_key == fact.proposition.stable_key()
            and item.predicate_coverage == 1
            and item.state in {STRUCTURE_OPEN, STRUCTURE_CLOSED}
        )
        if len(relation_candidates) != 1:
            return False
        relation = relation_candidates[0]
        if relation.state == STRUCTURE_CLOSED:
            # A previously adopted REFERS target is already supported by the
            # exact Core history and marker.  Do not re-run the ordinary open
            # role projection, which correctly refuses to copy the target.
            return bool(
                any(item.space == _CORE
                    and item.hypothesis_key == fact.proposition.stable_key()
                    and item.polarity == 3
                    and item.evidence_key[:2]
                    == _REFERENCE_TARGET_ADOPTION_EVIDENCE
                    for item in evidence))
        fact_fillers = {
            item.role.stable_key(): item.filler.stable_key()
            for item in fact.bindings
        }
        if len(fact_fillers) != len(fact.bindings):
            return False
        span_matches = {}
        for member in self.input_structure.semantic_candidates:
            if member.projection_kind != PROJECTION_FILLER:
                continue
            span_matches.setdefault(
                (member.span_ref, member.role_key), []).append(member)
        known_fillers = {}
        for role_span in candidate.bindings:
            span_ref = tuple(
                item.ref_key for item in self.input_structure.spans
                if (item.start == role_span.start
                    and item.end == role_span.end
                    and item.values == role_span.values
                    and item.source_ref == candidate.source_ref)
            )
            if len(span_ref) > 1:
                return False
            if not span_ref:
                # A dynamic role may be represented only by the open-role
                # frame/binding, without an independently projected InputSpan.
                # Keep it unknown; never recover its filler from the active
                # Core fact.
                continue
            matches = tuple(span_matches.get((span_ref[0], role_span.role), ()))
            if not matches:
                # This role is a dynamic/unknown span.  It must not be
                # replaced by a filler recovered from the trained example.
                continue
            if len(matches) != 1:
                return False
            filler = matches[0].candidate_key
            if fact_fillers.get(role_span.role) != filler:
                return False
            known_fillers[role_span.role] = filler
        if not known_fillers:
            return False
        source = self._generation_input(fact.proposition)
        source_scope = source.hypothesis.scope.stable_key()
        # Only project bindings that were independently observed in the
        # current input.  Dynamic/unknown roles (for example the object in
        # ``词项甲指向什么？``) remain unbound; copying their active fact
        # filler here would turn a trained example into an answer hint.
        for role, filler in sorted(known_fillers.items()):
            bindings.append(BindingEntry(
                role_key=role, filler_key=filler, space=_CORE,
                scope_key=source_scope))
        evidence.extend(core_query_evidence(
            fact.proposition,
            self._evidence_history(fact.proposition),
            self.owner.core_runtime.context.scoped_identity_store.registry,
        ))
        return True

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
                route = self._route(open_node.frame.proposition)
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
                # An open role frame is not itself a fact.  Once its known
                # spans match one active Core fact exactly, however, carry
                # that fact's original Evidence into this QueryState so
                # cross-turn reference resolution can use the same support
                # boundary as a closed relation query.  This support check is
                # independent of typed-capability routes: older active facts
                # (notably REFERS) may have a trained frame without a route.
                if (fact is not None and open_node.candidate is None):
                    self._record_open_relation_support(
                        open_node.frame, fact, bindings, evidence)
                next_state = next_state.with_(
                    bindings=self._unique_bindings(bindings),
                    evidence=self._unique_evidence(evidence),
                )
            elif space == _DIALOGUE and open_node.candidate is not None:
                route = self._route(open_node.frame.proposition)
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

    @observed_relation_phase("core")
    def _expand_core(self, state: QueryState, remaining: tuple[FrontierEntry, ...],
                     edge: FrontierEntry,
                     origin: tuple[int, ...]) -> ExpansionOutcome:
        facts = self.owner.filler_edges.get(origin, ())
        partial_property_route = edge.root_key in self.owner._partial_property_core_roots
        if partial_property_route:
            if origin == edge.root_key:
                facts = tuple(
                    fact for fact in facts
                    if fact.proposition.stable_key() == edge.root_key)
            else:
                # The proposition root already consumed the complete fact.
                # Its filler descendants carry proof bindings only; they must
                # not reopen the global filler posting index.
                facts = ()
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
            if (generic_route
                or origin in self.owner._slot_question_core_roots
                or origin in self.owner._partial_property_core_roots)
            else None)
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
                source = self._generation_input(fact.proposition)
                source_scope = source.hypothesis.scope.stable_key()
                route = self._route(proposition)
                if route is None:
                    raise RuntimeError(
                        "active Core relation 没有 QueryState capability route")
                self._record_relation_capability(
                    route, new_bindings, new_evidence)
                self._record_typed_relation(
                    fact, route, new_bindings, new_evidence)
                new_evidence.extend(core_query_evidence(
                    fact.proposition,
                    self._evidence_history(fact.proposition),
                    self.owner.core_runtime.context.scoped_identity_store.registry))
            for binding in fact.bindings:
                filler = binding.filler.stable_key()
                new_bindings.append(BindingEntry(
                    role_key=binding.role.stable_key(),
                    filler_key=filler,
                    space=_CORE,
                    scope_key=source_scope,
                ))
                if filler == origin:
                    continue
                # 命题已在当前展开中完整核验，读取 hop 必须保留自己的
                # 来源；共享端点只去重后续 frontier，不能抹去命题证明。
                hops.append(QueryHop(
                    _CORE, proposition, predicate, origin, filler,
                    tuple(ord(value) for value in binding.surface),
                    depth, fact.source_hash,
                    tuple(item.filler.stable_key() for item in fact.bindings)))
                if filler in core_visited or filler in discovered:
                    continue
                discovered.add(filler)
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
        observation = self.owner._memory_observations_by_key.get(origin)
        hypothesis = self.owner._memory_hypotheses_by_key.get(origin)
        evidence = self.owner._memory_evidence_by_key.get(origin)
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
            # Observation relation_occurrences also contain delivery/role
            # occurrence and other materialization scaffolding.  Those edges
            # remain persisted and are seeded independently when they are
            # semantically active, but walking every one here makes a single
            # input Observation fan out into the whole session graph.  Keep
            # this frontier limited to the O/H/E objects admitted by the
            # current bounded restore; unrelated historical edges never enter
            # this QueryState.
            active_memory_keys = self.owner._memory_active_object_keys
            if self._memory_ohe_expansion_coverage is None:
                self._memory_ohe_expansion_coverage = memory_ohe_expansion_coverage(
                    state.roots, self.owner._memory_observations,
                    self.owner._memory_hypotheses, self.owner._memory_evidence)
            queued_memory_targets = set()
            for link in observation.relation_refs:
                new_bindings.append(BindingEntry((91533, 1), link.stable_key(), space=_MEMORY,
                                                  scope_key=observation.scope_key))
                target = link.value()
                if isinstance(target, MemoryObjectRef):
                    self.hops.append(QueryHop(_MEMORY, observation.observation_key, (91533, 1),
                                              observation.observation_key, target.stable_key(), (), depth,
                                              observation.source_hash))
                    target_identity = (target.object_kind, target.stable_key())
                    if (target_identity in active_memory_keys
                            and target_identity not in self._memory_ohe_expansion_coverage
                            and target_identity not in queued_memory_targets):
                        queued_memory_targets.add(target_identity)
                        next_edges.append(FrontierEntry(
                            (_MEMORY, *target.stable_key()), _MEMORY, target_key=target.stable_key(),
                            root_key=edge.root_key, source_ref=observation.source_ref,
                            scope_key=observation.scope_key, direction=1,
                            owner_weight=self.owner.space_weights[1], depth=depth,
                            recency_weight=observation.turn_seq))
            for item in self.owner._memory_hypotheses_by_observation.get(
                    observation.observation_key, ()):
                if item.hypothesis_key not in observation.hypothesis_keys:
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
                for evidence_item in self.owner._memory_evidence_by_hypothesis.get(
                        item.hypothesis_key, ()):
                    if evidence_item.evidence_key not in item.evidence_keys:
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
            for item in self.owner._memory_evidence_by_hypothesis.get(
                    hypothesis.hypothesis_key, ()):
                if item.evidence_key not in hypothesis.evidence_keys:
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
        memory_hypothesis = self._discourse_hypotheses_by_key.get(origin)
        if memory_hypothesis is not None:
            depth = edge.depth + 1
            binding = BindingEntry(
                role_key=memory_hypothesis.hypothesis_kind,
                filler_key=memory_hypothesis.candidate_key,
                space=_DIALOGUE,
                scope_key=memory_hypothesis.scope_key,
            )
            response_context = self._response_contexts_by_hypothesis.get(origin)
            context_bindings = () if response_context is None else response_context.query_bindings()
            hypothesis_evidence = tuple(
                item for item in self.owner._memory_evidence_by_hypothesis.get(
                    memory_hypothesis.hypothesis_key, ())
                if item.evidence_key in memory_hypothesis.evidence_keys
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
            topic = self._discourse_topics_by_candidate.get(
                memory_hypothesis.candidate_key)
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
            source = self._generation_input(frame.proposition)
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
