"""把训练后对话根、当前 O/H/E 和来源化角色接入既有六层生成消费者。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_observation import ObservedGenerationEvidence
from pure_integer_ai.cognition.shared.generation_observed_surface import (
    ObservedGraphSurfaceProposal,
    delivered_graph_surface_category,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal, GenerationCandidate, GenerationPlanningRequest,
    NonPropositionGenerationCandidate, NonPropositionGenerationGoal,
)
from pure_integer_ai.cognition.shared.generation_response import (
    ResponseActDiscourseRouter, ResponseActGenerationBinding, ResponseActGenerationRegistry,
    ResponseActPropositionRouter, ResponseActSyntaxRouter,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import GenerationStructurePlanner
from pure_integer_ai.cognition.shared.generation_surface import GenerationSurfacePreview, SurfaceSlotDirective
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT, OBJECT_LANGUAGE_ATOM,
    OBJECT_PROPOSITION, OBJECT_REPRESENTATION, ObjectIdentity, SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_driver import unvisited_frontier
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED, ROOT_PENDING, SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY,
    BindingEntry, EvidenceEntry, FrontierEntry, QueryAnchor, QueryRoot, QueryState, VisitedKey,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    RenderedSurface, UnicodeRepresentationRenderer, render_generation_preview, representation_parts,
)
from pure_integer_ai.cognition.shared.response_plan import ResponsePlan, ResponseRealization, ResponseSlot
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity, semantic_source,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.structure_order_consumer import StructureSlotValue
from pure_integer_ai.cognition.shared.typed_binding import BoundRoleBinding
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRelationCandidate, pack_record
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT,
    QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_open_role_generation import OpenRoleGenerationContext
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceRuntime, TypedGenerationSurfaceRequestBuilder,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorDiscourseMapper, LanguageConnectorExecutionRequestMapper,
    LanguageConnectorPropositionMapper, LanguageConnectorSyntaxMapper, LanguageGenerationConnectorRegistry,
)
from pure_integer_ai.experiments.response_generation_graph import (
    GENERIC_RESPONSE_GRAPH_CONCEPT,
    GENERIC_RESPONSE_GRAPH_ENTITY,
    GENERIC_RESPONSE_GRAPH_EVENT,
    GENERIC_RESPONSE_GRAPH_PROPOSITION,
    GENERIC_RESPONSE_GRAPH_ROLES,
    GENERIC_RESPONSE_GRAPH_TOPIC,
    RecoveredResponseConnector,
    generic_response_context_marker,
    generic_response_context_role,
    generic_response_context_type,
    generic_response_graph_role,
    generic_response_graph_type,
)
from pure_integer_ai.experiments.trained_discourse_topic_topology import (
    DISCOURSE_GRAPH_INPUT_ROLE,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    _ALIAS_BUDGET, _bound_relation_proposition, _execution_planner, _generation_protocols,
    _plan_generation, _runtime_policy, _surface_protocol,
)
from pure_integer_ai.experiments.unknown_response_course import (
    RESPONSE_FEATURE_DISCOURSE_RELATION,
    RESPONSE_FEATURE_ENTITY,
    RESPONSE_FEATURE_EVENT,
    RESPONSE_FEATURE_EVENT_TIME,
    RESPONSE_FEATURE_PROPERTY,
    RESPONSE_FEATURE_RELATION_CHAIN,
    RESPONSE_FEATURE_TOPIC_GRAPH,
    response_condition_priority,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD, IntegerIdentityRegistry


@dataclass(frozen=True, slots=True)
class ResponseConnectorQueryNode:
    """同一输入角色结构对应的全部训练动作根，不提前选择立场或答案。"""

    learned: RecoveredResponseConnector
    candidate: OpenRelationCandidate
    source_hash: int

    @property
    def key(self) -> tuple[int, ...]:
        """区分训练结构与当前输入实例，保留并列解析。"""
        return pack_record(91527, self.learned.template.connector.stable_key(), self.candidate.stable_key())

    @property
    def source_ref(self) -> tuple[int, ...]:
        """返回真实训练 SourceRecord 身份，不读取其正文。"""
        return self.source_hash, *self.learned.source.stable_key()

    def seed(self, weight: int):
        """查询开始时登记 Dialogue root/anchor/frontier，权重仅影响扩展顺序。"""
        scope = self.learned.scope.stable_key()
        return (QueryRoot(SPACE_DIALOGUE, self.key, ROOT_PENDING, weight,
                          source_ref=self.source_ref, scope_key=scope),
                QueryAnchor(SPACE_DIALOGUE, self.key, kind=3, scope_key=scope),
                FrontierEntry((SPACE_DIALOGUE, *self.key), SPACE_DIALOGUE, target_key=self.key,
                              root_key=self.key, source_ref=self.source_ref, scope_key=scope,
                              owner_weight=weight, required_slot_gain=1))

    def expand(self, state: QueryState, edge: FrontierEntry) -> QueryState:
        """语言结构支持只归于此根，不能升级未知关系的真值。"""
        if edge.owner_space != SPACE_DIALOGUE or edge.target_key != self.key:
            raise ValueError("对话结构根与待展开 frontier 不一致")
        evidence = EvidenceEntry(self.source_ref, self.key, space=SPACE_DIALOGUE, polarity=1,
                                 evidence_key=self.key, scope_key=self.learned.scope.stable_key(),
                                 payload_key=self.learned.stable_key())
        binding = BindingEntry(self.learned.stance.stable_key(), self.key, space=SPACE_DIALOGUE,
                               scope_key=self.learned.scope.stable_key())
        return state.with_(
            frontier=tuple(item for item in state.frontier if item != edge),
            depth=max(state.depth, edge.depth + 1),
            bindings=tuple(sorted({*state.bindings, binding}, key=lambda item: item.stable_key())),
            evidence=tuple(sorted({*state.evidence, evidence}, key=lambda item: item.stable_key())),
            visited=tuple(sorted({*state.visited, VisitedKey(SPACE_DIALOGUE, self.key, direction=edge.direction)},
                                 key=lambda item: item.stable_key())),
            node_count=state.node_count + 1, edge_count=state.edge_count + 1, read_count=state.read_count + 1)


@dataclass(frozen=True, slots=True)
class GenericResponseConnectorQueryNode:
    """A trained constant-member response act participating as a Dialogue root."""

    learned: RecoveredResponseConnector
    source_hash: int
    context_key: tuple[int, ...]
    priority: int

    def __post_init__(self) -> None:
        if (not isinstance(self.learned, RecoveredResponseConnector)
                or type(self.source_hash) is not int or self.source_hash <= 0
                or type(self.context_key) is not tuple or not self.context_key
                or any(type(value) is not int for value in self.context_key)
                or type(self.priority) is not int or self.priority <= 0):
            raise ValueError("generic response connector query node is incomplete")

    @property
    def key(self) -> tuple[int, ...]:
        return pack_record(
            91564, self.learned.template.connector.stable_key(), self.context_key)

    @property
    def source_ref(self) -> tuple[int, ...]:
        return self.source_hash, *self.learned.source.stable_key()

    def seed(self, weight: int):
        scope = self.learned.scope.stable_key()
        return (
            QueryRoot(SPACE_DIALOGUE, self.key, ROOT_PENDING, weight,
                      source_ref=self.source_ref, scope_key=scope),
            QueryAnchor(SPACE_DIALOGUE, self.key, kind=3, support=self.priority,
                        scope_key=scope),
            FrontierEntry(
                (SPACE_DIALOGUE, *self.key), SPACE_DIALOGUE,
                target_key=self.key, root_key=self.key,
                source_ref=self.source_ref, scope_key=scope,
                owner_weight=weight, required_slot_gain=1,
                evidence_support=1, discourse_fit=self.priority,
                source_trust=1,
            ),
        )

    def expand(self, state: QueryState, edge: FrontierEntry) -> QueryState:
        if edge.owner_space != SPACE_DIALOGUE or edge.target_key != self.key:
            raise ValueError("generic response connector frontier differs")
        evidence = EvidenceEntry(
            self.source_ref, self.context_key, space=SPACE_DIALOGUE,
            polarity=1, evidence_key=self.key,
            scope_key=self.learned.scope.stable_key(),
            payload_key=self.learned.stable_key(),
        )
        binding = BindingEntry(
            self.learned.stance.stable_key(), self.key,
            space=SPACE_DIALOGUE, scope_key=self.learned.scope.stable_key())
        return state.with_(
            frontier=tuple(item for item in state.frontier if item != edge),
            depth=max(state.depth, edge.depth + 1),
            bindings=tuple(sorted({*state.bindings, binding},
                                  key=lambda item: item.stable_key())),
            evidence=tuple(sorted({*state.evidence, evidence},
                                  key=lambda item: item.stable_key())),
            visited=tuple(sorted({
                *state.visited,
                VisitedKey(SPACE_DIALOGUE, self.key, direction=edge.direction),
            }, key=lambda item: item.stable_key())),
            node_count=state.node_count + 1,
            edge_count=state.edge_count + 1,
            read_count=state.read_count + 1,
        )


def response_query_nodes(runtime, core, candidates) -> tuple[ResponseConnectorQueryNode, ...]:
    """沿已训练角色结构匹配全部动作；不用来源文本、字相似度或失败后备条件。"""
    reader = getattr(runtime, "response_connectors", None)
    if reader is None:
        return ()
    learned = reader()
    if not learned:
        return ()
    source_identity = getattr(runtime, "source_identity_for", None)
    identities = None if source_identity is not None else IntegerIdentityRegistry(runtime.backend)
    facts = {item.proposition.stable_key(): item for item in core.active_surface_facts()}
    result = []
    for candidate in candidates:
        # Open-role candidates may be emitted by an extension structure
        # (KdConv/event-time) that has no learned generic response connector.
        # They remain represented in the shared input graph, but must not
        # crash the Dialogue owner by indexing a non-Core proposition.
        fact = facts.get(candidate.frame.proposition)
        if fact is None:
            continue
        schema = runtime._role_template(fact)
        for item in learned:
            if (item.template.language_branch != schema.language_branch
                    or item.template.proposition_structure != schema.proposition_structure
                    or item.template.predicate != schema.predicate):
                continue
            source_hash = (source_identity(item.source) if source_identity is not None
                           else identities.find(IDENTITY_SOURCE_RECORD, item.source.stable_key()))
            if source_hash is None:
                raise ValueError("训练对话结构丢失实际 SourceRecord 身份")
            result.append(ResponseConnectorQueryNode(item, candidate, source_hash))
    return tuple(sorted(result, key=lambda item: item.key))


def generic_response_query_nodes(runtime, hypotheses, query_key: tuple[int, ...],
                                 feature_mask: int = 0, *,
                                 input_structure: QueryInputStructure | None = None,
                                 explicit_graph_input_keys: tuple[
                                     tuple[int, ...], ...] = (),
                                 ) -> tuple[GenericResponseConnectorQueryNode, ...]:
    """Recover every eligible graph connector; priority chooses style only."""
    reader = getattr(runtime, "response_connectors", None)
    if reader is None or not hypotheses:
        return ()
    hypothesis_keys = tuple(sorted(item.hypothesis_key for item in hypotheses))
    context_key = pack_record(91564, query_key, *hypothesis_keys)
    source_identity = getattr(runtime, "source_identity_for", None)
    identities = None if source_identity is not None else IntegerIdentityRegistry(runtime.backend)
    eligible = []
    for learned in reader():
        owner = next((item for item in runtime._branches
                      if item.branch == learned.template.language_branch), None)
        if owner is None:
            continue
        values = owner.definition_graph.value_protocol
        protocols = _generation_protocols(
            runtime.context, owner.branch, retain_unknown_candidates=True)
        if learned.stance != protocols.content.unknown:
            continue
        contract = _generic_response_contract(
            learned, values, _surface_protocol(owner.branch))
        if contract is None:
            continue
        if (input_structure is not None
                and not _generic_graph_contract_bindable(
                    contract,
                    input_structure,
                    explicit_graph_input_keys,
                )):
            continue
        source_hash = (source_identity(learned.source) if source_identity is not None
                       else identities.find(
                           IDENTITY_SOURCE_RECORD, learned.source.stable_key()))
        if source_hash is None:
            raise ValueError("generic connector lacks its integer source identity")
        # A visible graph slot is a structural tie-break only.  It does not
        # remove context-only connectors from this QueryState; it ensures a
        # connector that can consume the current graph relation wins an equal
        # condition-strength tie over a legacy context-only realization.
        eligible.append((learned, source_hash, len(contract.graph_slots)))
    eligible.sort(key=lambda item: item[0].stable_key())
    if not eligible:
        return ()
    # An explicit graph root can be an unknown structural input, but it can
    # also accompany a fact whose Core relation is already closed.  The latter
    # must keep the trained factual/context organization as the primary route;
    # promoting a graph-slot connector solely from the PROPERTY feature made
    # closed Core queries lose their ResponsePlan.  Only an explicit graph
    # root with no closed Core claim may promote a bindable graph-slot act.
    graph_signal = (
        RESPONSE_FEATURE_ENTITY
        | RESPONSE_FEATURE_EVENT
        | RESPONSE_FEATURE_PROPERTY
        | RESPONSE_FEATURE_TOPIC_GRAPH
        | RESPONSE_FEATURE_DISCOURSE_RELATION
        | RESPONSE_FEATURE_RELATION_CHAIN
        | RESPONSE_FEATURE_EVENT_TIME
    )
    complete_core_relation = bool(
        input_structure is not None
        and any(
            relation.state == STRUCTURE_CLOSED
            and relation.required_count >= 2
            and relation.required_count == relation.support_count
            and relation.predicate_coverage == 1
            for relation in input_structure.relation_candidates))
    explicit_graph_root = bool(explicit_graph_input_keys)
    # Text+graph inputs need a relation-qualified Core projection before a
    # graph-slot connector can be selected.  An arbitrary proposition/object
    # key is still consumed by Core in this QueryState, but it cannot prove a
    # slot assignment by itself.  Graph-only input has no surface relation and
    # is the one case where the explicit graph root is the assignment source.
    graph_only_input = bool(
        input_structure is not None
        and not input_structure.token_values
        and bool(input_structure.graph_object_keys))
    # An explicit proposition/entity/event root is itself a closed structural
    # input, even when the accompanying text is intentionally unknown.  The
    # old gate required a closed *text* relation here, which made a valid
    # graph-root query select the context-only connector and hid the held-out
    # graph slots.  Keep Core precedence for a complete textual relation;
    # otherwise let the same QueryState graph root raise its structural act.
    graph_unknown_shape = (
        explicit_graph_root
        and bool(feature_mask & graph_signal)
        and (graph_only_input or not complete_core_relation))
    graph_preferred = graph_unknown_shape and any(
        item[2] > 0 for item in eligible)
    scored = tuple(
        (base_priority + (100000 if graph_preferred and item[2] > 0 else 0), item)
        for item in eligible
        for base_priority in (response_condition_priority(item[0].condition_key, feature_mask),)
        if base_priority > 0)
    matched = tuple((score, item) for score, item in scored if score > 0)
    if not matched:
        return ()

    # A purely unknown input has no graph-derived semantic signal.  In that
    # state a structural connector with visible graph slots cannot close, so
    # let the already-trained context-only connector win within the same
    # Dialogue root set.  This is selection over the shared QueryState
    # candidate pool, not a runtime fallback chain or a second graph query.
    if not graph_preferred:
        context_only = tuple((score, item) for score, item in matched
                             if item[2] == 0)
        if context_only:
            matched = context_only
    highest = max(score for score, _item in matched)
    preferred = tuple(item for score, item in matched if score == highest)
    highest_graph_slots = max(item[2] for item in preferred)
    preferred = tuple(item for item in preferred if item[2] == highest_graph_slots)
    selected_item = preferred[
        (sum(query_key) + sum(sum(key) for key in hypothesis_keys))
        % len(preferred)]
    if graph_preferred:
        # Keep every condition-compatible graph connector in this shared
        # QueryState candidate set.  A connector whose typed slots cannot be
        # closed by the already-expanded Core hops is not allowed to mask a
        # lower-priority connector that does close; generation performs that
        # contract check over the same state, with no second query or
        # database fallback.
        selected_scores = {
            item[0].stable_key() for _score, item in matched
            if item[2] > 0
        }
    else:
        selected_scores = {
            item[0].stable_key() for score, item in matched if score == highest
        }
    return tuple(GenericResponseConnectorQueryNode(
        learned, source_hash, context_key,
        response_condition_priority(learned.condition_key, feature_mask) * 1000
        + (1 if learned == selected_item[0] else 0),
    ) for learned, source_hash, _graph_slot_count in eligible
      if learned.stable_key() in selected_scores)


@dataclass(frozen=True, slots=True)
class _GenericGraphSlot:
    """One visible connector slot whose filler must come from QueryState."""

    binding: object
    category: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class _GenericResponseContract:
    """Validated silent Memory context plus visible graph-derived slots."""

    context: object
    graph_slots: tuple[_GenericGraphSlot, ...]


@dataclass(frozen=True, slots=True)
class _GenericGraphValue:
    """A unique graph filler and the QueryState records that qualified it."""

    slot: ObjectIdentity
    filler: ObjectIdentity
    evidence: tuple[tuple[int, ...], ...]
    observed_surface: ObservedGraphSurfaceProposal | None = None


@dataclass(frozen=True, slots=True)
class _GenericGraphSourceSpan:
    """Unique exact input span projected to one semantic graph identity."""

    input_source_ref: tuple[int, ...]
    start: int
    end: int
    values: tuple[int, ...]
    projection_keys: tuple[tuple[int, ...], ...]


def _generic_graph_source_span(
        input_structure: QueryInputStructure,
        filler: ObjectIdentity,
        ) -> _GenericGraphSourceSpan | None:
    """Recover one exact trained span; repeated/competing spans stay open."""
    if not isinstance(input_structure, QueryInputStructure):
        raise TypeError("generic graph surface requires QueryInputStructure")
    filler_key = filler.stable_key()
    spans = {item.ref_key: item for item in input_structure.spans}
    semantic = tuple(
        item for item in input_structure.semantic_candidates
        if item.projection_kind == PROJECTION_FILLER
        and item.candidate_key == filler_key
    )
    semantic_refs = {item.span_ref for item in semantic}
    if semantic:
        if len(semantic_refs) != 1:
            return None
        span = spans.get(next(iter(semantic_refs)))
        if span is None:
            raise ValueError("generic graph semantic projection lost its span")
        return _GenericGraphSourceSpan(
            input_structure.source_ref,
            span.start,
            span.end,
            span.values,
            tuple(sorted({item.stable_key() for item in semantic})),
        )

    relations = tuple(
        item for item in input_structure.relation_candidates
        if item.proposition_key == filler_key
    )
    if not relations:
        return None
    candidates = {}
    for relation in relations:
        members = tuple(spans.get(key) for key in relation.span_refs)
        if not members or any(item is None for item in members):
            raise ValueError("generic graph relation projection lost its spans")
        start = min(item.start for item in members)
        end = max(item.end for item in members)
        values = input_structure.token_values[start:end]
        key = start, end, values
        candidates.setdefault(key, []).append(relation.stable_key())
    if len(candidates) != 1:
        return None
    (start, end, values), projection_keys = next(iter(candidates.items()))
    return _GenericGraphSourceSpan(
        input_structure.source_ref,
        start,
        end,
        values,
        tuple(sorted(set(projection_keys))),
    )


def _generic_graph_contract_bindable(
        contract: _GenericResponseContract,
        input_structure: QueryInputStructure,
        explicit_graph_input_keys: tuple[tuple[int, ...], ...],
        ) -> bool:
    """在播种 Dialogue root 前排除本轮不可能闭合的图槽合同。"""
    if not contract.graph_slots:
        return True
    if (type(explicit_graph_input_keys) is not tuple
            or any(type(key) is not tuple or not key
                   or any(type(value) is not int or value < 0 for value in key)
                   for key in explicit_graph_input_keys)):
        raise ValueError("explicit graph inputs 必须是非空整数 tuple 集")
    category_for_kind = {
        OBJECT_ENTITY: GENERIC_RESPONSE_GRAPH_ENTITY,
        OBJECT_EVENT: GENERIC_RESPONSE_GRAPH_EVENT,
        OBJECT_PROPOSITION: GENERIC_RESPONSE_GRAPH_PROPOSITION,
        OBJECT_CONCEPT: GENERIC_RESPONSE_GRAPH_CONCEPT,
    }
    candidates: dict[int, set[tuple[int, ...]]] = {
        category: set() for category in GENERIC_RESPONSE_GRAPH_ROLES
    }
    for item in input_structure.semantic_candidates:
        if item.projection_kind != PROJECTION_FILLER:
            continue
        category = category_for_kind.get(item.object_kind)
        if category is not None:
            candidates[category].add(item.candidate_key)
    for item in input_structure.relation_candidates:
        if _generic_graph_source_span(
                input_structure,
                ObjectIdentity.from_stable_key(item.proposition_key)) is not None:
            candidates[GENERIC_RESPONSE_GRAPH_PROPOSITION].add(
                item.proposition_key)
    for key in explicit_graph_input_keys:
        try:
            identity = ObjectIdentity.from_stable_key(key)
        except (TypeError, ValueError):
            continue
        category = category_for_kind.get(identity.object_kind)
        if category is not None:
            candidates[category].add(key)
        if identity.object_kind == OBJECT_PROPOSITION:
            candidates[GENERIC_RESPONSE_GRAPH_TOPIC].add(key)
    if len(contract.graph_slots) == 1:
        slot = contract.graph_slots[0]
        return slot.ordinal == 1 and len(candidates[slot.category]) == 1

    # A graph-only input has no surface relation_candidates by design.  Keep
    # the trained multi-slot Dialogue roots alive when the explicit graph
    # object is real; final slot order is closed later from the Core hop
    # relation evidence, never from object-key ordering.
    if (input_structure.graph_object_keys
            and explicit_graph_input_keys):
        return True

    # 多槽仍要求一个当前输入中的完整关系提供角色序；仅有若干无序图键
    # 不能凭对象排序制造句法角色。
    for relation in input_structure.relation_candidates:
        if relation.state not in {STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT}:
            continue
        if (relation.support_count != relation.required_count
                or relation.required_open != 0):
            continue
        members = tuple(
            item for item in input_structure.semantic_candidates
            if (item.projection_kind == PROJECTION_FILLER
                and item.proposition_key == relation.proposition_key
                and item.span_ref in relation.span_refs)
        )
        available: dict[int, int] = {
            category: 0 for category in GENERIC_RESPONSE_GRAPH_ROLES
        }
        for member in members:
            category = category_for_kind.get(member.object_kind)
            if category is not None:
                available[category] += 1
        if all(available[item.category] >= item.ordinal
               for item in contract.graph_slots):
            return True
    return False


def _generic_representation_family(runtime, branch: ObjectIdentity) -> tuple[int, ...]:
    """Recover the single trained R-01 representation family once per branch."""
    cache = getattr(runtime, "_generic_response_representation_families", None)
    if cache is None:
        cache = {}
        setattr(runtime, "_generic_response_representation_families", cache)
    branch_key = branch.stable_key()
    cached = cache.get(branch_key)
    if cached is not None:
        return cached
    alias = runtime.alias_runtime(branch)
    families = {
        representation_parts(binding.filler)[0]
        for fact in alias.closure.consumer.lookup_relation(
            alias.selector.protocol.realizes_relation)
        for binding in fact.proposition.bindings
        if binding.filler.object_kind == OBJECT_REPRESENTATION
    }
    if len(families) != 1:
        raise ValueError("generic graph surface representation family is ambiguous")
    family = next(iter(families))
    cache[branch_key] = family
    return family


def _generic_response_contract(learned, values, surface_protocol):
    """Validate the silent context and any visible graph-derived Role slots."""
    template = learned.template
    marker = generic_response_context_marker(template.language_branch)
    if template.context != (marker,):
        return None
    roles = tuple(item for item in template.bindings
                  if item.source == values.role_filler_source)
    constants = tuple(item for item in template.bindings
                      if item.source == values.constant_source)
    if (not roles or len(constants) < 1
            or len(roles) + len(constants) != len(template.bindings)
            or any(item.constant is None
                   or item.constant.object_kind != OBJECT_LANGUAGE_ATOM
                   for item in constants)):
        raise ValueError("generic connector 必须含上下文 Role 和来源化语言原子")
    contexts = tuple(
        item for item in roles
        if item.role == generic_response_context_role(template.language_branch)
    )
    if len(contexts) != 1:
        raise ValueError("generic connector 必须含唯一 Memory 上下文 Role")
    context = contexts[0]
    slots = {item.slot: item for item in template.slots}
    directives = {item.slot: item for item in template.surface}
    if (context.role != generic_response_context_role(template.language_branch)
            or values.ordinal_value(context.ordinal) != 0
            or slots[context.slot].role != context.role
            or slots[context.slot].value_type
            != generic_response_context_type(template.language_branch)
            or directives[context.slot].action != surface_protocol.silent_action
            or directives[context.slot].surface_prefix_steps
            or context.slot == learned.marker_slot
            or any(directives[item.slot].action != surface_protocol.emit_action
                   for item in constants)):
        raise ValueError("generic connector 的上下文或 surface 合同漂移")
    graph_slots = []
    for binding in roles:
        if binding == context:
            continue
        matches = tuple(
            category for category in sorted(GENERIC_RESPONSE_GRAPH_ROLES)
            if binding.role == generic_response_graph_role(
                template.language_branch, category)
        )
        if len(matches) != 1:
            raise ValueError("generic graph slot Role category is ambiguous")
        category = matches[0]
        ordinal = values.ordinal_value(binding.ordinal)
        if (ordinal <= 0
                or slots[binding.slot].role != binding.role
                or slots[binding.slot].value_type
                != generic_response_graph_type(
                    template.language_branch, category)
                or directives[binding.slot].action
                != surface_protocol.emit_action):
            raise ValueError("generic graph slot contract drifted")
        graph_slots.append(_GenericGraphSlot(binding, category, ordinal))
    keys = tuple((item.category, item.ordinal) for item in graph_slots)
    if len(set(keys)) != len(keys):
        raise ValueError("generic graph slots duplicate category/ordinal")
    return _GenericResponseContract(
        context,
        tuple(sorted(
            graph_slots,
            key=lambda item: item.binding.slot.stable_key(),
        )),
    )


def _generic_context_binding(learned, values, surface_protocol):
    """Return the legacy context view after validating the full contract."""
    contract = _generic_response_contract(learned, values, surface_protocol)
    return None if contract is None else contract.context


def _graph_identity(value: tuple[int, ...]) -> ObjectIdentity | None:
    """Recover a complete ObjectIdentity without interpreting its components."""
    try:
        return ObjectIdentity.from_stable_key(value)
    except (TypeError, ValueError):
        return None


def _generic_graph_values(
        state: QueryState,
        contract: _GenericResponseContract,
        input_structure: QueryInputStructure,
        branch: ObjectIdentity,
        family: tuple[int, ...],
        source: SourceRef,
        scope: ScopeIdentity,
        observation: MemoryObjectRef,
        explicit_graph_input_keys: tuple[tuple[int, ...], ...] = (),
        graph_hops: tuple[object, ...] = (),
        ) -> tuple[_GenericGraphValue, ...] | None:
    """Bind every visible graph slot from unique same-query graph records."""
    if not contract.graph_slots:
        return ()
    by_category: dict[int, dict[ObjectIdentity, list[BindingEntry]]] = {
        category: {} for category in GENERIC_RESPONSE_GRAPH_ROLES
    }
    source_spans: dict[
        ObjectIdentity, _GenericGraphSourceSpan | None] = {}
    explicit_keys = frozenset(explicit_graph_input_keys)
    generic_core_evidence = any(
        item.space == SPACE_CORE
        and len(item.evidence_key) >= 2
        and item.evidence_key[:2] == (21610, 3)
        for item in state.evidence)
    graph_visible_keys = set(explicit_keys)
    if generic_core_evidence or explicit_keys:
        # Generic Core bindings are the identities reached by this query's
        # ontology statements.  They are eligible graph fillers even without
        # a text span; ordinary trained-surface bindings remain span-bound.
        for hop in graph_hops:
            if getattr(hop, "space", None) != SPACE_CORE:
                continue
            if explicit_keys and getattr(hop, "depth", None) != 1:
                continue
            endpoints = (getattr(hop, "from_filler", ()),
                         getattr(hop, "to_filler", ()))
            if explicit_keys and not (set(endpoints) & explicit_keys):
                continue
            graph_visible_keys.update(
                key for key in endpoints
                if type(key) is tuple and key)
            relation_fillers = getattr(hop, "relation_fillers", ())
            if (type(relation_fillers) is tuple
                    and all(type(key) is tuple and key
                            for key in relation_fillers)):
                graph_visible_keys.update(relation_fillers)
    kind_for_category = {
        GENERIC_RESPONSE_GRAPH_ENTITY: OBJECT_ENTITY,
        GENERIC_RESPONSE_GRAPH_EVENT: OBJECT_EVENT,
        GENERIC_RESPONSE_GRAPH_PROPOSITION: OBJECT_PROPOSITION,
        GENERIC_RESPONSE_GRAPH_CONCEPT: OBJECT_CONCEPT,
    }
    for binding in state.bindings:
        if binding.conflict_kept:
            continue
        identity = _graph_identity(binding.filler_key)
        if identity is None:
            continue
        source_span = _generic_graph_source_span(input_structure, identity)
        # A visible dynamic slot always needs an exact span in this input.
        # Historical graph objects remain active QueryState evidence, but
        # cannot compete for the surface unless the current trained
        # projection points to the same semantic identity.
        if source_span is None and identity.stable_key() not in graph_visible_keys:
            continue
        source_spans[identity] = source_span
        categories = []
        if (binding.space == SPACE_DIALOGUE
                and binding.role_key == DISCOURSE_GRAPH_INPUT_ROLE):
            categories.append(GENERIC_RESPONSE_GRAPH_TOPIC)
        delivered_category = delivered_graph_surface_category(
            binding.role_key)
        if delivered_category is not None:
            if binding.space != SPACE_MEMORY:
                raise ValueError(
                    "delivered graph-slot projection must stay in Memory")
            categories.append(delivered_category)
        for category, object_kind in kind_for_category.items():
            if (binding.space in (SPACE_CORE, SPACE_DIALOGUE)
                    and identity.object_kind == object_kind):
                categories.append(category)
        if (generic_core_evidence
                and binding.space in (SPACE_CORE, SPACE_DIALOGUE)
                and identity.object_kind == OBJECT_PROPOSITION):
            categories.append(GENERIC_RESPONSE_GRAPH_TOPIC)
        for category in categories:
            by_category[category].setdefault(identity, []).append(binding)

    # Relation-group members must be visible as ordinary Core role bindings in
    # this query.  Dialogue's generic graph-input marker intentionally binds
    # the explicit proposition root too, but that marker is structural input
    # evidence and must not consume a trained relation ordinal.
    core_visible = {
        identity
        for binding in state.bindings
        if binding.space == SPACE_CORE
        for identity in (_graph_identity(binding.filler_key),)
        if identity is not None and not binding.conflict_kept
    }

    selected_by_slot: dict[
        ObjectIdentity, tuple[ObjectIdentity, list[BindingEntry]]
    ] = {}
    if len(contract.graph_slots) > 1:
        # 多槽不能分别从全局候选集任取“第一个”。先按同一闭合关系的
        # proposition/role/member ordinal 构造完整赋值；只有所有可用关系
        # 给出同一组 filler 时才闭合。这样两个 Entity/Event/Concept 槽的
        # 次序来自训练图角色拓扑，而不是 ObjectIdentity 排序或表层位置猜测。
        assignments: dict[
            tuple[tuple[int, ...], ...],
            dict[ObjectIdentity, tuple[ObjectIdentity, list[BindingEntry]]],
        ] = {}
        for relation in input_structure.relation_candidates:
            if relation.state not in {
                    STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT}:
                continue
            # Order-conflict is eligible only when the input contains the
            # complete trained relation.  Missing-role/open candidates stay
            # rejected.  The unique assignment test below, followed by the
            # QueryState conflict closure, decides whether surface order can
            # be supplied by the trained response connector.
            if (relation.support_count != relation.required_count
                    or relation.required_open != 0):
                continue
            members = tuple(
                item for item in input_structure.semantic_candidates
                if (item.projection_kind == PROJECTION_FILLER
                    and item.proposition_key == relation.proposition_key
                    and item.span_ref in relation.span_refs)
            )
            if (len(members) != relation.role_coverage
                    or len({(item.member_ordinal, item.role_key)
                            for item in members}) != len(members)):
                continue
            grouped: dict[int, list[object]] = {
                category: [] for category in GENERIC_RESPONSE_GRAPH_ROLES
            }
            for member in members:
                for category, object_kind in kind_for_category.items():
                    if member.object_kind == object_kind:
                        grouped[category].append(member)
            for category in grouped:
                grouped[category].sort(key=lambda item: (
                    item.member_ordinal,
                    item.role_key,
                    item.span_ref,
                    item.candidate_key,
                ))

            assignment: dict[
                ObjectIdentity,
                tuple[ObjectIdentity, list[BindingEntry]],
            ] = {}
            complete = True
            for graph_slot in contract.graph_slots:
                candidates = grouped[graph_slot.category]
                index = graph_slot.ordinal - 1
                if index < 0 or index >= len(candidates):
                    complete = False
                    break
                filler = _graph_identity(candidates[index].candidate_key)
                if (filler is None
                        or filler not in by_category[graph_slot.category]):
                    complete = False
                    break
                assignment[graph_slot.binding.slot] = (
                    filler,
                    by_category[graph_slot.category][filler],
                )
            if not complete:
                continue
            assignment_key = tuple(
                assignment[item.binding.slot][0].stable_key()
                for item in contract.graph_slots
            )
            prior = assignments.get(assignment_key)
            if prior is None:
                assignments[assignment_key] = assignment
            else:
                # 同一语义赋值可有多条来源证据；合并 BindingEntry，但不把
                # 不同赋值压成一个候选。
                for slot, (filler, bindings) in assignment.items():
                    old_filler, old_bindings = prior[slot]
                    if old_filler != filler:
                        raise ValueError(
                            "relation-qualified graph assignment drifted")
                    prior[slot] = (
                        filler,
                        list({*old_bindings, *bindings}),
                    )
        if len(assignments) != 1:
            # Graph-only inputs intentionally carry no lexical relation
            # candidate.  Recover relation-qualified assignments from the
            # same QueryState Core hops.  Each generic hop is emitted with
            # the trained predicate role for outgoing traversal and the
            # explicit inverse role marker for incoming traversal; this keeps
            # subject/object order in graph evidence instead of guessing from
            # stable object keys.
            hop_assignments: dict[
                tuple[tuple[int, ...], ...],
                dict[ObjectIdentity, tuple[ObjectIdentity, list[BindingEntry]]],
            ] = {}
            # Generation may reach this function through the same QueryState
            # after the bridge has normalized requested graph keys.  The
            # canonical visible-key set is therefore the authoritative
            # explicit-root carrier; relying only on the optional call
            # argument made proposition slots disappear during generation.
            requested_roots = frozenset(explicit_graph_input_keys)
            root_candidates = requested_roots | frozenset(graph_visible_keys)
            requested_propositions = frozenset(
                key for key in root_candidates
                if (_graph_identity(key) is not None
                    and _graph_identity(key).object_kind == OBJECT_PROPOSITION))
            eligible_hops = []
            for hop in graph_hops:
                if getattr(hop, "space", None) != SPACE_CORE:
                    continue
                if requested_roots and getattr(hop, "depth", None) != 1:
                    continue
                endpoints = {
                    getattr(hop, "from_filler", ()),
                    getattr(hop, "to_filler", ()),
                }
                if requested_roots and not (endpoints & requested_roots):
                    continue
                fillers = getattr(hop, "relation_fillers", ())
                if (type(fillers) is tuple and fillers
                        and all(type(item) is tuple and item
                                for item in fillers)):
                    if (requested_propositions
                            and getattr(hop, "proposition", ())
                            not in requested_propositions):
                        continue
                    eligible_hops.append(hop)
            # A direct active Proposition hop contains the authoritative
            # ordered relation members.  Raw ontology hops from the same root
            # are structural evidence only and must not create competing
            # connector assignments.
            if not eligible_hops:
                eligible_hops = [
                    hop for hop in graph_hops
                    if getattr(hop, "space", None) == SPACE_CORE]
            for hop in eligible_hops:
                if getattr(hop, "space", None) != SPACE_CORE:
                    continue
                if requested_roots:
                    if getattr(hop, "depth", None) != 1:
                        continue
                    if not ({getattr(hop, "from_filler", ()),
                             getattr(hop, "to_filler", ())}
                            & requested_roots):
                        continue
                source_hash = getattr(hop, "source_hash", 0)
                if type(source_hash) is not int or source_hash <= 0:
                    continue
                from_key = getattr(hop, "from_filler", ())
                to_key = getattr(hop, "to_filler", ())
                role_key = getattr(hop, "predicate", ())
                relation_fillers = getattr(hop, "relation_fillers", ())
                if (type(from_key) is not tuple or not from_key
                        or type(to_key) is not tuple or not to_key
                        or type(role_key) is not tuple or not role_key):
                    continue
                if (type(relation_fillers) is tuple and relation_fillers
                        and all(type(item) is tuple and item
                                for item in relation_fillers)):
                    # Trained Core facts carry their complete ordered role
                    # fillers.  Keep the graph's order, but discard explicit
                    # proposition/topic roots that have no binding in the
                    # current QueryState.  Otherwise an unbound root shifts
                    # the ordinal sequence and prevents the actual role
                    # members from closing the connector contract.
                    ordered = tuple(
                        candidate for candidate in relation_fillers
                        if (identity := _graph_identity(candidate)) is not None
                        and identity in core_visible
                    )
                else:
                    inverse = (len(role_key) >= 2
                               and role_key[0] == 21610
                               and role_key[1] == 2)
                    subject_key, object_key = (
                        (to_key, from_key) if inverse else (from_key, to_key))
                    ordered = tuple(
                        candidate for candidate in (subject_key, object_key)
                        if (identity := _graph_identity(candidate)) is not None
                        and identity in core_visible
                    )
                grouped: dict[int, list[ObjectIdentity]] = {
                    category: [] for category in GENERIC_RESPONSE_GRAPH_ROLES}
                for candidate_key in ordered:
                    identity = _graph_identity(candidate_key)
                    if identity is None:
                        continue
                    for category, object_kind in kind_for_category.items():
                        if identity.object_kind == object_kind:
                            grouped[category].append(identity)
                    if identity.object_kind == OBJECT_PROPOSITION:
                        grouped[GENERIC_RESPONSE_GRAPH_TOPIC].append(identity)
                # An explicit proposition graph root is a first-class Core
                # binding in this same QueryState, but it is intentionally not
                # duplicated in ``relation_fillers`` (those are ordered
                # relation members).  Permit a topic/proposition slot to
                # consume that root without inventing an ordinal or reading
                # source text.  The explicit-root guard keeps ordinary text
                # relations on their existing member-only contract.
                if requested_propositions:
                    for root_key in requested_propositions:
                        root = _graph_identity(root_key)
                        if root is not None and root in by_category[
                                GENERIC_RESPONSE_GRAPH_PROPOSITION]:
                            grouped[GENERIC_RESPONSE_GRAPH_TOPIC].append(root)
                            grouped[GENERIC_RESPONSE_GRAPH_PROPOSITION].append(root)
                assignment: dict[
                    ObjectIdentity,
                    tuple[ObjectIdentity, list[BindingEntry]],
                ] = {}
                complete = True
                for graph_slot in contract.graph_slots:
                    members = grouped[graph_slot.category]
                    index = graph_slot.ordinal - 1
                    if index < 0 or index >= len(members):
                        complete = False
                        break
                    filler = members[index]
                    available = by_category[graph_slot.category].get(filler)
                    if not available:
                        complete = False
                        break
                    assignment[graph_slot.binding.slot] = (filler, available)
                if not complete:
                    continue
                assignment_key = tuple(
                    assignment[item.binding.slot][0].stable_key()
                    for item in contract.graph_slots)
                prior = hop_assignments.get(assignment_key)
                if prior is None:
                    hop_assignments[assignment_key] = assignment
                else:
                    for slot, (filler, bindings) in assignment.items():
                        old_filler, old_bindings = prior[slot]
                        if old_filler != filler:
                            raise ValueError("Core hop graph assignment drifted")
                        prior[slot] = (filler, list({*old_bindings, *bindings}))
            if len(hop_assignments) != 1:
                return None
            selected_by_slot = next(iter(hop_assignments.values()))
        else:
            selected_by_slot = next(iter(assignments.values()))

    result = []
    for graph_slot in contract.graph_slots:
        if selected_by_slot:
            selected = selected_by_slot.get(graph_slot.binding.slot)
            if selected is None:
                return None
            filler, bindings = selected
        else:
            candidates = by_category[graph_slot.category]
            # 单槽合同继续要求唯一候选；ordinal 不是对象排序请求。
            if graph_slot.ordinal != 1 or len(candidates) != 1:
                return None
            filler, bindings = next(iter(candidates.items()))
        proof = {item.stable_key() for item in bindings}
        scopes = {item.scope_key for item in bindings if item.scope_key}
        spaces = {item.space for item in bindings}
        if any(
                not binding.scope_key
                or not any(
                    evidence.space == binding.space
                    and evidence.scope_key == binding.scope_key
                    for evidence in state.evidence)
                for binding in bindings):
            return None
        qualified_evidence = tuple(
            item.stable_key()
            for item in state.evidence
            if item.space in spaces
            and item.scope_key in scopes
        )
        if not qualified_evidence:
            return None
        proof.update(qualified_evidence)
        proof_keys = tuple(sorted(proof))
        source_span = source_spans[filler]
        observed_surface = None
        current_scope = scope.stable_key()
        current_ohe = tuple(
            item.stable_key()
            for item in state.evidence
            if item.space in (SPACE_MEMORY, SPACE_DIALOGUE)
            and item.scope_key == current_scope
        )
        if not current_ohe:
            return None
        proof_keys = tuple(sorted({*proof_keys, *current_ohe}))
        if source_span is not None:
            observed_surface = ObservedGraphSurfaceProposal(
                filler,
                branch,
                source,
                scope,
                observation,
                source_span.input_source_ref,
                source_span.start,
                source_span.end,
                source_span.values,
                family,
                source_span.projection_keys,
                proof_keys,
            )
        result.append(_GenericGraphValue(
            graph_slot.binding.slot,
            filler,
            proof_keys,
            observed_surface,
        ))
    return tuple(sorted(result, key=lambda item: item.slot.stable_key()))


@dataclass(frozen=True, slots=True)
class _ObservedResponseDirectives:
    """原 G-03 的逐槽指令适配器；动态 Span 与训练常量走不同显式来源。"""

    node: ResponseConnectorQueryNode
    context: object
    binding: object
    proposals: tuple
    value_protocol: object

    def plan(self, structure, execution, branch):
        """保持同次 selection、完整槽值及训练 emit 指令，不补词或替换角色。"""
        template = self.node.learned.template
        if (not execution.complete or branch != template.language_branch
                or structure.selection.stable_key() != self.binding.selection_key
                or len(structure.syntax.sentences) != 1):
            raise ValueError("对话 G-03 上游计划或目标分支漂移")
        sentence = structure.syntax.sentences[0]
        if sentence.sentence != template.sentence:
            raise ValueError("对话 G-03 句式不是同一训练模板")
        sources = {item.slot: item for item in template.bindings}
        directives = {item.slot: item for item in template.surface}
        observed = {item.value.ordinal: item for item in self.proposals}
        result = []
        for value in sentence.values:
            source = sources[value.slot]
            directive = directives[value.slot]
            observed_key = ()
            if source.source == self.value_protocol.role_filler_source:
                ordinal = self.value_protocol.ordinal_value(source.ordinal)
                proposal = observed.get(ordinal)
                if proposal is None or proposal.origin != value.filler or proposal.value.role != source.role:
                    raise ValueError("对话动态槽没有同次来源化角色表示")
                observed_key = proposal.value.stable_key()
            elif source.source != self.value_protocol.constant_source or source.constant != value.filler:
                raise ValueError("对话常量槽不来自训练 connector")
            trace = pack_record(91528, self.node.key, self.binding.stable_key(), directive.stable_key())
            result.append(SurfaceSlotDirective(
                sentence.address, value.slot, directive.action, directive.instruction, trace,
                directive.surface_prefix_steps, _ALIAS_BUDGET,
                surface_use_key=pack_record(91528, self.context.observation.stable_key(),
                                            self.node.key, value.slot.stable_key()),
                observed_value_key=observed_key))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class _GenericResponseDirectives:
    """Bind silent Memory context and visible graph fillers into G-03."""

    node: GenericResponseConnectorQueryNode
    context: object
    binding: ResponseActGenerationBinding
    value_protocol: object
    surface_protocol: object
    graph_values: tuple[_GenericGraphValue, ...] = ()

    def plan(self, structure, execution, branch):
        """逐槽核验当前上下文绑定，并为静默槽关闭全部 R-01 请求。"""
        template = self.node.learned.template
        if (not execution.complete or branch != template.language_branch
                or structure.selection.stable_key() != self.binding.selection_key
                or len(structure.syntax.sentences) != 1):
            raise ValueError("generic G-03 上游计划或目标分支漂移")
        sentence = structure.syntax.sentences[0]
        if sentence.sentence != template.sentence:
            raise ValueError("generic G-03 句式不是同一训练模板")
        contract = _generic_response_contract(
            self.node.learned, self.value_protocol, self.surface_protocol)
        if contract is None:
            raise ValueError("generic G-03 缺少图内上下文合同")
        context_binding = contract.context
        context_identity = self.context.binding_identity()
        sources = {item.slot: item for item in template.bindings}
        directives = {item.slot: item for item in template.surface}
        graph_values = {item.slot: item for item in self.graph_values}
        if (set(graph_values)
                != {item.binding.slot for item in contract.graph_slots}):
            raise ValueError("generic G-03 graph fillers do not cover its contract")
        graph_categories = {
            item.binding.slot: item.category for item in contract.graph_slots
        }
        result = []
        for value in sentence.values:
            source = sources[value.slot]
            directive = directives[value.slot]
            silent = source == context_binding
            # A proposition supplied as an explicit graph root is structural
            # QueryState evidence, not an observed language span.  Keep it in
            # the typed connector and its proof, but make that slot silent so
            # G-03 never invents a surface from the proposition stable key or
            # falls back to source text.  Entity/event slots still require an
            # exact observed span below.
            graph_value = graph_values.get(value.slot)
            structural_silent = (
                not silent
                and graph_value is not None
                and graph_value.observed_surface is None
                and graph_value.filler.object_kind == OBJECT_PROPOSITION
                and graph_categories.get(value.slot) in {
                    GENERIC_RESPONSE_GRAPH_TOPIC,
                    GENERIC_RESPONSE_GRAPH_PROPOSITION,
                }
            )
            if structural_silent:
                if directive.surface_prefix_steps:
                    raise ValueError(
                        "proposition graph root cannot use surface prefix steps")
                silent = True
            if silent:
                if (value.filler != context_identity
                        and not structural_silent):
                    raise ValueError(
                        "generic 静默槽没有绑定当前 Memory 上下文")
                if (not structural_silent
                        and directive.action != self.surface_protocol.silent_action):
                    raise ValueError("generic 静默槽没有绑定当前 Memory 上下文")
                budget = None
                use_key = ()
            elif source.source == self.value_protocol.role_filler_source:
                if (graph_value is None
                        or value.filler != graph_value.filler
                        or directive.action != self.surface_protocol.emit_action):
                    raise ValueError(
                        "generic visible slot is not bound to current QueryState")
                budget = _ALIAS_BUDGET
                use_key = pack_record(
                    91568,
                    self.context.observation.stable_key(),
                    self.node.key,
                    value.slot.stable_key(),
                    *graph_value.evidence,
                )
                observed_value_key = (
                    () if graph_value.observed_surface is None
                    else graph_value.observed_surface.value_key
                )
                if observed_value_key and directive.surface_prefix_steps:
                    raise ValueError(
                        "generic observed graph slot cannot use R-01 prefix steps")
            else:
                if (source.source != self.value_protocol.constant_source
                        or source.constant != value.filler
                        or directive.action != self.surface_protocol.emit_action):
                    raise ValueError("generic 表层槽不来自训练 LanguageAtom")
                budget = _ALIAS_BUDGET
                use_key = pack_record(
                    91567,
                    self.context.observation.stable_key(),
                    self.node.key,
                    value.slot.stable_key(),
                )
                observed_value_key = ()
            if silent:
                observed_value_key = ()
            trace = pack_record(
                91567,
                self.node.key,
                self.binding.stable_key(),
                directive.stable_key(),
                self.context.stable_key(),
            )
            result.append(SurfaceSlotDirective(
                sentence.address,
                value.slot,
                (self.surface_protocol.silent_action
                 if structural_silent else directive.action),
                directive.instruction,
                trace,
                directive.surface_prefix_steps,
                budget,
                surface_use_key=use_key,
                observed_value_key=observed_value_key,
            ))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class OpenResponseGeneration:
    """已执行六层生成的非事实计划，保留查询根、完整候选和逐槽表示。"""

    node_key: tuple[int, ...]
    context_key: tuple[int, ...]
    response_plan: ResponsePlan
    representations: tuple[ObjectIdentity, ...]
    trace: tuple[int, ...]
    context: OpenRoleGenerationContext
    preview: GenerationSurfacePreview
    rendered: RenderedSurface

    def __post_init__(self):
        """保留原生成对象供输出采用，禁止从 trace 猜造新 preview。"""
        if (self.context.stable_key() != self.context_key or not self.preview.complete
                or self.preview.representations != self.representations
                or self.rendered.representations != self.representations
                or self.rendered.units != tuple(map(ord, self.response_plan.surface()))
                or self.preview.request.structure.selection.stance != self.response_plan.response_act):
            raise ValueError("开放回应的上下文、生成槽或渲染结果漂移")

    def stable_key(self):
        """完整生成 trace 与本次组织对象一起保存，不只留下输出摘要。"""
        return pack_record(91529, self.node_key, self.context_key, self.response_plan.stable_key(),
                           pack_record(1, *(item.stable_key() for item in self.representations)), self.trace)


def _observation_candidates(runtime, core, state, contexts, hypotheses, protocols):
    """复用实际 Core 模板做运行期 Span 替换，真实 O/H/E 单独进入原 G-00。"""
    facts = {item.proposition.stable_key(): item for item in core.active_surface_facts()}
    memory = {item.hypothesis_key: item for item in hypotheses}
    state_evidence = {item.stable_key(): item for item in state.evidence}
    result = {}
    for context in contexts:
        fact = facts[context.candidate.frame.proposition]
        original = core.generation_input(fact.proposition)
        template = runtime._role_template(fact)
        bound = _bound_relation_proposition(original, template, protocols.content.answer)
        bindings = []
        for span, role in zip(context.observed_spans(), fact.bindings, strict=True):
            typed = tuple(item for item in original.proposition.definition.bindings
                          if item.role == role.role and item.filler == role.filler)
            if len(typed) != 1 or span.role != role.role:
                raise ValueError("开放生成角色没有唯一训练模板绑定")
            bindings.append(BoundRoleBinding(span.role, span.origin, typed[0].ordinal))
        bound = replace(bound, bindings=tuple(bindings))
        hypothesis = memory[context.hypothesis.stable_key()]
        if (hypothesis.observation_key != context.observation.stable_key()
                or hypothesis.competition_key != context.competition_key):
            raise ValueError("开放生成 O/H 竞争身份漂移")
        evidence = ObservedGenerationEvidence(
            bound, context.source, context.scope, context.observation, context.hypothesis,
            hypothesis.hypothesis_kind, hypothesis.competition_key, context.observed_spans(),
            tuple(state_evidence[key] for key in context.evidence), context.stable_key())
        result[context.stable_key()] = GenerationCandidate(
            bound, evidence.state, semantic_source(bound.template), context.scope, (),
            observation_evidence=(evidence,))
    return result


def _reference_generation_provenance(state, discourse_contexts, reference_resolutions):
    """Require resolved graph evidence before exposing a cross-turn winner to G-02."""
    from pure_integer_ai.experiments.trained_reference_resolution import (
        REFERENCE_RESOLVED, CrossTurnReferenceResolution,
    )

    if (type(reference_resolutions) is not tuple
            or any(not isinstance(item, CrossTurnReferenceResolution)
                   for item in reference_resolutions)):
        raise TypeError("开放回应的 reference resolutions 类型错误")
    required = tuple(item for item in reference_resolutions if item.requires_resolution)
    if any(item.status != REFERENCE_RESOLVED for item in required):
        raise ValueError("开放回应不能消费未裁决或冲突的跨轮指代")
    contexts = {item.stable_key(): item for item in discourse_contexts}
    objects = set()
    evidence = set()
    memory_refs = set()
    targets = set()
    for resolution in required:
        winner = resolution.winner
        target = resolution.winner_target
        context = contexts.get(resolution.winner_context_key)
        if winner is None or target is None or context is None:
            raise ValueError("跨轮指代 winner 缺少可恢复的目标或篇章上下文")
        if winner.occurrence not in context.role_occurrences:
            raise ValueError("跨轮指代 winner occurrence 不属于其采用上下文")
        if any(item not in state.bindings for item in resolution.query_bindings()) or any(
                item not in state.evidence for item in resolution.query_evidence()):
            raise ValueError("跨轮指代 winner 尚未进入同一次 QueryState")
        objects.update(resolution.discourse_objects())
        evidence.add(resolution.stable_key())
        evidence.add(resolution.winner_context_key)
        evidence.update(item.stable_key() for item in resolution.query_evidence())
        memory_refs.update((winner.occurrence.observation, winner.occurrence.hypothesis))
        targets.add(target)
    return (
        tuple(sorted(objects, key=lambda item: item.stable_key())),
        tuple(sorted(evidence)),
        tuple(sorted(memory_refs, key=lambda item: item.stable_key())),
        tuple(sorted(targets, key=lambda item: item.stable_key())),
    )


def _active_memory_refs(state, hypotheses, *, selected_hypothesis_keys=None):
    """Convert selected restored O/H/E keys without copying unrelated history.

    QueryState still carries the complete parallel Memory evidence.  Delivery
    adopts only the hypothesis branch that produced the current response;
    embedding every historical branch into each Episode makes the append-only
    Use identity grow quadratically across turns.
    """
    if selected_hypothesis_keys is None:
        hypothesis_keys = {item.hypothesis_key for item in hypotheses}
    else:
        hypothesis_keys = set(selected_hypothesis_keys)
    evidence_keys = {
        key for item in hypotheses
        if item.hypothesis_key in hypothesis_keys
        for key in item.evidence_keys
    }
    active = set()
    for evidence in state.evidence:
        if evidence.space != SPACE_MEMORY or evidence.hypothesis_key not in hypothesis_keys:
            continue
        active.add(MemoryObjectRef.from_stable_key(evidence.hypothesis_key))
        if evidence.evidence_key in evidence_keys:
            active.add(MemoryObjectRef.from_stable_key(evidence.evidence_key))
    return active


def generate_open_responses(runtime, core, state: QueryState, contexts, hypotheses,
                            proposals_by_hypothesis, nodes, *, discourse_contexts=(),
                            reference_resolutions=()) -> tuple[OpenResponseGeneration, ...]:
    """在共享查询中执行所有闭合生成输入，不按其他图库是否失败启动。"""
    if not contexts or not nodes:
        return ()
    if (any(root.status == ROOT_PENDING for root in state.roots)
            or unvisited_frontier(state.frontier, state.visited)):
        raise ValueError("开放回应必须等同一次 frontier 的完整证据展开")
    for discourse in discourse_contexts:
        if any(binding not in state.bindings for binding in discourse.query_bindings()):
            raise ValueError("前序回应上下文尚未在本次共同 frontier 展开")
        if any(item.query_binding() not in state.bindings or item.query_evidence() not in state.evidence
               for item in discourse.role_occurrences):
            raise ValueError("前序角色发生尚未在本次共同 Memory frontier 展开")
    reference_objects, reference_evidence, reference_memory_refs, reference_targets = (
        _reference_generation_provenance(
            state, discourse_contexts, reference_resolutions))
    branches = {node.learned.template.language_branch for node in nodes}
    if len(branches) != 1:
        raise ValueError("多语言对话选择必须显式指定当前目标分支")
    branch = next(iter(branches))
    protocols = _generation_protocols(runtime.context, branch, retain_unknown_candidates=True)
    candidates = _observation_candidates(runtime, core, state, contexts, hypotheses, protocols)
    all_candidates = tuple(candidates.values())
    owner = next(item for item in runtime._branches if item.branch == branch)
    values = owner.definition_graph.value_protocol
    expanded = {(root.owner_space, root.root_key) for root in state.roots if root.status == ROOT_EXPANDED}
    result = []
    for context in contexts:
        candidate = candidates[context.stable_key()]
        request = GenerationPlanningRequest(AnswerGenerationGoal(
            protocols.content.answer, candidate.proposition, LogicEvidenceState(True, False),
            context.source, context.scope, branch), all_candidates)
        selection = protocols.selector.select(request)
        if selection.stance == protocols.content.answer:
            continue
        compatible_nodes = tuple(
            node for node in nodes
            if node.candidate == context.candidate
            and node.learned.stance == selection.stance)
        if len(compatible_nodes) > 1:
            # Multiple training observations can share one proposition schema.
            # Keep every connector as a Dialogue root/frontier participant, but
            # let the integer role protocol select the structurally preserving
            # organization for this input.  This is graph evidence (the
            # trained ordinal binding order), not source text, string distance,
            # or a fixed connector order.  A genuine tie remains ambiguous and
            # is not silently collapsed into an arbitrary answer.
            candidate_order = tuple(range(len(context.candidate.bindings)))

            def role_order_score(node):
                dynamic = tuple(
                    item for item in node.learned.template.bindings
                    if item.source == values.role_filler_source)
                ordinal_by_instruction = {
                    item.instruction: item.ordinal for item in values.ordinals}
                if any(item.ordinal not in ordinal_by_instruction
                       for item in dynamic):
                    return -1
                order = tuple(ordinal_by_instruction[item.ordinal]
                              for item in dynamic)
                if set(order) != set(candidate_order):
                    return -1
                exact = int(order == candidate_order)
                positional = sum(
                    int(index == ordinal)
                    for index, ordinal in enumerate(order))
                return exact * (len(order) + 1) + positional

            scored = tuple((role_order_score(node), node)
                           for node in compatible_nodes)
            highest = max(score for score, _node in scored)
            compatible_nodes = tuple(
                node for score, node in scored if score == highest)
        # The six-layer planner consumes one uniquely selected graph
        # organization.  If the training graph leaves a real tie, preserve
        # the ambiguity rather than choosing by stable-key order.
        if len(compatible_nodes) != 1:
            continue
        for node in compatible_nodes:
            if (SPACE_DIALOGUE, node.key) not in expanded or not any(
                    item.space == SPACE_DIALOGUE and item.evidence_key == node.key
                    and item.payload_key == node.learned.stable_key() for item in state.evidence):
                raise ValueError("开放回应没有同次已展开的训练动作证据")
            proposals = proposals_by_hypothesis.get(context.hypothesis.stable_key(), ())
            if len(proposals) != len(context.candidate.bindings):
                raise ValueError("开放回应必须保留全部角色表示")
            template = node.learned.template
            binding = node.learned.bind_context(context, selection, values)
            if discourse_contexts or reference_objects:
                binding = replace(binding,
                    discourse_context=tuple(sorted({identity for item in discourse_contexts
                                                     for identity in item.discourse_objects()}
                                                    | set(reference_objects),
                                                    key=lambda item: item.stable_key())),
                    evidence=tuple(sorted({*binding.evidence,
                                           *(item.stable_key() for item in discourse_contexts),
                                           *reference_evidence})))
            registry = ResponseActGenerationRegistry((node.learned.response_template(values),), (binding,))
            ordinary = LanguageGenerationConnectorRegistry(values, (template,))
            structure = GenerationStructurePlanner(
                ResponseActDiscourseRouter(LanguageConnectorDiscourseMapper(ordinary), registry),
                ResponseActPropositionRouter(LanguageConnectorPropositionMapper(), registry),
                ResponseActSyntaxRouter(LanguageConnectorSyntaxMapper(ordinary), registry))
            surface_protocol = _surface_protocol(branch)
            policy = _runtime_policy(template, surface_protocol)
            builder = TypedGenerationSurfaceRequestBuilder(
                surface_protocol, _execution_planner(template, owner.lifecycle),
                LanguageConnectorExecutionRequestMapper(policy.order_budget),
                _ObservedResponseDirectives(node, context, binding, proposals, values))
            surface = GenerationSurfaceRuntime(runtime.alias_runtime(branch), observed_surfaces=proposals)
            plan = _plan_generation(request, protocols, structure, builder, surface)
            preview = plan.layers[-1].artifact
            if not plan.complete or not isinstance(preview, GenerationSurfacePreview) or not preview.complete:
                raise ValueError("开放回应六层生成未完成，不能渲染局部表层")
            families = {representation_parts(item)[0] for item in preview.representations}
            if len(families) != 1:
                raise ValueError("开放回应的角色与语言原子表示族不同")
            renderer = UnicodeRepresentationRenderer(next(iter(families)), protocols.renderer)
            rendered = render_generation_preview(preview, renderer)
            slots = {slot.slot: slot for slot in template.slots}
            response_slots = tuple(ResponseSlot(
                slots[item.value.slot].role, item.value.filler,
                renderer.text(renderer.render((item.representation,))), True,
                context.source_ref[0] if item.observed_surface is not None else node.source_hash,
                allowed_node_kinds=(item.value.filler.object_kind,)) for item in preview.slots)
            memory_refs = {context.observation, context.hypothesis,
                           *reference_memory_refs}
            for discourse in discourse_contexts:
                memory_refs.update((discourse.observation, discourse.hypothesis,
                                    discourse.context.observation, discourse.context.hypothesis))
            memory_refs.update(_active_memory_refs(
                state, hypotheses,
                selected_hypothesis_keys={context.hypothesis.stable_key()}))
            discourse_links = tuple(sorted({
                *(item.proof_ref for item in discourse_contexts),
                *reference_objects,
            } - {template.connector}, key=lambda item: item.stable_key()))
            response = ResponsePlan(selection.stance, (), response_slots, (
                ResponseRealization(renderer.text(rendered), template.sentence, node.source_hash, len(preview.slots)),),
                tuple(item.stable_key() for item in state.evidence),
                scope_and_time=pack_record(1, context.source.stable_key(), context.scope.stable_key()),
                discourse_links=(template.connector, *discourse_links),
                event_refs=tuple(item for item in reference_targets
                                 if item.object_kind == OBJECT_EVENT),
                memory_refs=tuple(sorted(memory_refs, key=lambda item: item.stable_key())))
            result.append(OpenResponseGeneration(node.key, context.stable_key(), response, preview.representations,
                                                 pack_record(1, plan.stable_key(), rendered.stable_key(), binding.stable_key()),
                                                 context, preview, rendered))
    return tuple(sorted(result, key=lambda item: item.stable_key()))


@dataclass(frozen=True, slots=True)
class GenericResponseGenerationContext:
    """Current generic Observation and its graph candidates, with no surface."""

    query_key: tuple[int, ...]
    observation: MemoryObjectRef
    source: SourceRef
    scope: ScopeIdentity
    hypothesis_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (type(self.query_key) is not tuple or not self.query_key
                or any(type(value) is not int for value in self.query_key)
                or not isinstance(self.observation, MemoryObjectRef)
                or not isinstance(self.source, SourceRef)
                or not isinstance(self.scope, ScopeIdentity)
                or self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions
                or type(self.hypothesis_keys) is not tuple
                or not self.hypothesis_keys
                or self.hypothesis_keys != tuple(sorted(set(self.hypothesis_keys)))):
            raise ValueError("generic response generation context is incomplete")

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            91565,
            self.query_key,
            self.observation.stable_key(),
            self.source.stable_key(),
            self.scope.stable_key(),
            pack_record(1, *self.hypothesis_keys),
        )

    def binding_identity(self) -> ObjectIdentity:
        """把当前 Observation/Hypothesis 组封装为来源化一等 ContextScope。"""
        return context_scope_identity(self.source, self.stable_key())


@dataclass(frozen=True, slots=True)
class TrainedGenericResponseGeneration:
    """A completed connector generation for current generic Memory evidence."""

    node_key: tuple[int, ...]
    response_plan: ResponsePlan
    representations: tuple[ObjectIdentity, ...]
    trace: tuple[int, ...]
    context: GenericResponseGenerationContext
    preview: GenerationSurfacePreview
    rendered: RenderedSurface
    graph_surfaces: tuple[tuple[int, ObservedGraphSurfaceProposal], ...] = ()

    def __post_init__(self) -> None:
        if (not self.preview.complete
                or self.preview.representations != self.representations
                or self.rendered.representations != self.representations
                or self.rendered.units != tuple(map(ord, self.response_plan.surface()))
                or self.response_plan.claim_refs
                or self.preview.request.structure.selection.stance
                != self.response_plan.response_act):
            raise ValueError("trained generic response generation differs from its plan")
        if (type(self.graph_surfaces) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not int
                       or not isinstance(item[1], ObservedGraphSurfaceProposal)
                       for item in self.graph_surfaces)
                or len({(item[0], item[1].target)
                        for item in self.graph_surfaces})
                != len(self.graph_surfaces)):
            raise ValueError("trained generic graph surfaces are incomplete")

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            91566,
            self.node_key,
            self.context.stable_key(),
            self.response_plan.stable_key(),
            pack_record(1, *(item.stable_key() for item in self.representations)),
            self.trace,
        )


def generate_generic_responses(
        runtime,
        state: QueryState,
        hypotheses,
        nodes: tuple[GenericResponseConnectorQueryNode, ...],
        observation: MemoryObjectRef | None,
        input_structure: QueryInputStructure,
        explicit_graph_input_keys: tuple[tuple[int, ...], ...] = (),
        graph_hops: tuple[object, ...] = (),
        ) -> tuple[TrainedGenericResponseGeneration, ...]:
    """Execute one graph-prioritized response act through the existing six layers."""
    if runtime is None or not hypotheses or not nodes or observation is None:
        return ()
    if (any(root.status == ROOT_PENDING for root in state.roots)
            or unvisited_frontier(state.frontier, state.visited)
            or tuple(state.active_spaces) != (SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE)):
        raise ValueError("generic response requires the completed shared three-graph frontier")
    # Candidate priority orders all connectors discovered in this same
    # QueryState.  A high-priority connector whose typed graph contract cannot
    # close must not mask a lower-priority connector that does close; evaluate
    # the binding contract before committing the single generation route.
    ranked_nodes = tuple(sorted(
        nodes, key=lambda item: (-item.priority, item.key)))
    node = ranked_nodes[0]
    branches = {item.learned.template.language_branch for item in nodes}
    if len(branches) != 1:
        raise ValueError("generic response requires one explicit target branch")
    branch = next(iter(branches))
    protocols = _generation_protocols(
        runtime.context, branch, retain_unknown_candidates=True)
    if any(item.learned.stance != protocols.content.unknown
           for item in ranked_nodes):
        raise ValueError("generic connector is not the trained unknown response act")
    # A shared query may carry both a closed Core claim and residual Memory
    # hypotheses.  Generic UNKNOWN generation is legal only for hypotheses that
    # are explicitly UNKNOWN in both Memory and Dialogue for this query; do not
    # feed a supported/closed hypothesis into the non-proposition contract.
    unknown_keys = {
        item.hypothesis_key
        for item in state.evidence
        if item.space in (SPACE_MEMORY, SPACE_DIALOGUE)
        and item.polarity == 3
    }
    hypotheses = tuple(item for item in hypotheses
                       if item.hypothesis_key in unknown_keys)
    if not hypotheses:
        return ()
    sources = {item.source_ref for item in hypotheses}
    scopes = {item.scope_key for item in hypotheses}
    observations = {item.observation_key for item in hypotheses}
    if len(sources) != 1 or len(scopes) != 1 or observations != {observation.stable_key()}:
        raise ValueError("generic hypotheses do not share the current Observation source/scope")
    source_ref = next(iter(sources))
    source = SourceRef.from_stable_key(source_ref[1:])
    scope = ScopeIdentity.from_stable_key(next(iter(scopes)))
    context_key = pack_record(
        91565, state.query_key,
        pack_record(1, *(item.hypothesis_key for item in hypotheses)))
    query_evidence = tuple(state.evidence)
    candidates = tuple(NonPropositionGenerationCandidate(
        context_key,
        LogicEvidenceState(False, False),
        source,
        scope,
        observation,
        MemoryObjectRef.from_stable_key(item.hypothesis_key),
        item.hypothesis_kind,
        item.competition_key,
        state.stable_key(),
        query_evidence,
        tuple(state.active_spaces),
    ) for item in hypotheses)
    request = GenerationPlanningRequest(
        NonPropositionGenerationGoal(
            protocols.content.answer,
            context_key,
            LogicEvidenceState(True, False),
            source,
            scope,
            branch,
        ),
        candidates,
    )
    selection = protocols.selector.select(request)
    if (selection.stance != protocols.content.unknown
            or set(selection.selected_candidate_keys) != set(request.candidate_keys())):
        raise ValueError("generic response did not preserve the UNKNOWN O/H/E candidates")
    owner = next(item for item in runtime._branches if item.branch == branch)
    values = owner.definition_graph.value_protocol
    surface_protocol = _surface_protocol(branch)
    family = _generic_representation_family(runtime, branch)
    expanded = {(root.owner_space, root.root_key) for root in state.roots
                if root.status == ROOT_EXPANDED}
    contract = None
    graph_values = None
    for candidate_node in ranked_nodes:
        if ((SPACE_DIALOGUE, candidate_node.key) not in expanded
                or not any(
                    item.space == SPACE_DIALOGUE
                    and item.evidence_key == candidate_node.key
                    and item.payload_key == candidate_node.learned.stable_key()
                    for item in state.evidence)):
            continue
        candidate_contract = _generic_response_contract(
            candidate_node.learned, values, surface_protocol)
        if candidate_contract is None:
            continue
        candidate_values = _generic_graph_values(
            state,
            candidate_contract,
            input_structure,
            branch,
            family,
            source,
            scope,
            observation,
            explicit_graph_input_keys,
            graph_hops,
        )
        if candidate_values is None:
            continue
        node = candidate_node
        contract = candidate_contract
        graph_values = candidate_values
        break
    if graph_values is None:
        return ()
    template = node.learned.template
    context_binding = contract.context
    response_template = node.learned.response_template(values)
    context = GenericResponseGenerationContext(
        state.query_key,
        observation,
        source,
        scope,
        tuple(sorted(item.hypothesis_key for item in hypotheses)),
    )
    context_identity = context.binding_identity()
    content_values = (
        StructureSlotValue(context_binding.slot, context_identity),
        *(StructureSlotValue(item.slot, item.filler)
          for item in graph_values),
        *(StructureSlotValue(item.slot, item.constant)
          for item in template.bindings
          if (item.source == values.constant_source
              and item.slot != node.learned.marker_slot)),
    )
    # The binding records the graph connector and current Memory context.  A
    # complete QueryState is already carried by the shared frontier/evidence;
    # embedding state.stable_key() here made every later delivery proof copy
    # the entire prior frontier recursively.
    binding = ResponseActGenerationBinding(
        selection.stable_key(),
        response_template.stable_key(),
        source,
        scope,
        content_values,
        tuple(sorted({
            node.learned.stable_key(), context.stable_key(), state.query_key,
            *(key for item in graph_values for key in item.evidence),
        })),
        tuple(sorted(
            {context_identity, *(item.filler for item in graph_values)},
            key=lambda item: item.stable_key(),
        )),
    )
    registry = ResponseActGenerationRegistry((response_template,), (binding,))
    ordinary = LanguageGenerationConnectorRegistry(values, (template,))
    structure = GenerationStructurePlanner(
        ResponseActDiscourseRouter(LanguageConnectorDiscourseMapper(ordinary), registry),
        ResponseActPropositionRouter(LanguageConnectorPropositionMapper(), registry),
        ResponseActSyntaxRouter(LanguageConnectorSyntaxMapper(ordinary), registry),
    )
    policy = _runtime_policy(template, surface_protocol)
    builder = TypedGenerationSurfaceRequestBuilder(
        surface_protocol,
        _execution_planner(template, owner.lifecycle),
        LanguageConnectorExecutionRequestMapper(policy.order_budget),
        _GenericResponseDirectives(
            node, context, binding, values, surface_protocol, graph_values),
    )
    surface = GenerationSurfaceRuntime(
        runtime.alias_runtime(branch),
        observed_surfaces=tuple(
            item.observed_surface for item in graph_values
            if item.observed_surface is not None
        ),
    )
    plan = _plan_generation(request, protocols, structure, builder, surface)
    preview = plan.layers[-1].artifact
    if (not plan.complete or not isinstance(preview, GenerationSurfacePreview)
            or not preview.complete):
        if isinstance(preview, GenerationSurfacePreview):
            failed_slot = preview.slots[-1] if preview.slots else None
            reason_key = preview.reason.stable_key()
            slot_key = (() if failed_slot is None else
                        failed_slot.value.slot.stable_key())
            filler_key = (() if failed_slot is None else
                          failed_slot.value.filler.stable_key())
            raise ValueError(
                "generic response six-layer connector generation did not "
                f"complete: reason={reason_key}, slot={slot_key}, "
                f"filler={filler_key}")
        raise ValueError(
            "generic response six-layer connector generation did not "
            "produce a surface preview")
    families = {representation_parts(item)[0] for item in preview.representations}
    if len(families) != 1:
        raise ValueError("generic response representations do not share one family")
    renderer = UnicodeRepresentationRenderer(next(iter(families)), protocols.renderer)
    rendered = render_generation_preview(preview, renderer)
    if not rendered.units:
        raise ValueError(
            "generic response connector completed without emitted surface units: "
            f"connector={template.connector.stable_key()} "
            f"condition={template.constraint_set.stable_key()}")
    slots = {slot.slot: slot for slot in template.slots}
    response_slots = tuple(ResponseSlot(
        slots[item.value.slot].role,
        item.value.filler,
        renderer.text(renderer.render((item.representation,))),
        True,
        node.source_hash,
        allowed_node_kinds=(item.value.filler.object_kind,),
    ) for item in preview.slots if item.representation is not None)
    # Delivery 只采用本轮 Memory evidence 明确暴露的 O/H/E 引用；输入
    # Observation 已由 generation.input_observation 单独校验，不能凭对象
    # 身份额外扩大采用集合。
    # The generic act preserves every UNKNOWN candidate in its planning
    # evidence, but adopts one deterministic branch for the delivery Episode.
    # The remaining candidates stay query-local evidence and are never copied
    # into every Use payload.
    memory_refs = _active_memory_refs(
        state, hypotheses,
        selected_hypothesis_keys={hypotheses[0].hypothesis_key})
    response = ResponsePlan(
        selection.stance,
        (),
        response_slots,
        (ResponseRealization(
            renderer.text(rendered), template.sentence,
            node.source_hash, len(response_slots)),),
        tuple(item.stable_key() for item in state.evidence),
        scope_and_time=pack_record(1, source.stable_key(), scope.stable_key()),
        # Keep an explicitly supplied graph identity in the generated plan's
        # evidence-facing links even when it is the proposition/topic root
        # rather than a visible surface slot.  This proves that generation
        # consumed the requested graph input in the same QueryState and lets
        # delivery attribution account for the root without replaying it.
        discourse_links=tuple(sorted(
            {template.connector, *(item.filler for item in graph_values),
             *(ObjectIdentity.from_stable_key(key)
               for key in explicit_graph_input_keys)},
            key=lambda item: item.stable_key(),
        )),
        event_refs=tuple(sorted(
            (item.filler for item in graph_values
             if item.filler.object_kind == OBJECT_EVENT),
            key=lambda item: item.stable_key(),
        )),
        memory_refs=tuple(sorted(memory_refs, key=lambda item: item.stable_key())),
    )
    # Persist a bounded, replayable integer trace.  Planner/render artefacts
    # remain available in-process for postcheck and delivery; they must not be
    # copied into the append-only Memory proof on every turn.
    trace = pack_record(
        91566,
        node.key,
        state.query_key,
        context.stable_key(),
        response.response_act.stable_key(),
        pack_record(1, *(item.hypothesis_key for item in hypotheses)),
        pack_record(2, *(item.evidence_key for item in state.evidence
                         if item.space in (SPACE_MEMORY, SPACE_DIALOGUE))),
        pack_record(
            3,
            *(pack_record(
                item.filler.object_kind,
                item.slot.stable_key(),
                item.filler.stable_key(),
                *item.evidence,
            ) for item in graph_values),
        ),
    )
    category_by_slot = {
        item.binding.slot: item.category for item in contract.graph_slots}
    graph_surfaces = tuple(sorted((
        (category_by_slot[item.slot], item.observed_surface)
        for item in graph_values if item.observed_surface is not None
    ), key=lambda item: (item[0], item[1].stable_key())))
    return (TrainedGenericResponseGeneration(
        node.key, response, preview.representations, trace,
        context, preview, rendered, graph_surfaces),)


__all__ = [
    "GenericResponseConnectorQueryNode",
    "GenericResponseGenerationContext",
    "OpenResponseGeneration",
    "ResponseConnectorQueryNode",
    "TrainedGenericResponseGeneration",
    "generate_generic_responses",
    "generate_open_responses",
    "generic_response_query_nodes",
    "response_query_nodes",
]
