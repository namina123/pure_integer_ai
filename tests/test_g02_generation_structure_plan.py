"""G-02 篇章、命题、句法守恒和 G-00 中间三层接线测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentProtocol,
    AnswerContentSelector,
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationLayerDecision,
    GenerationLayerRegistration,
    GenerationPlanProtocol,
    GenerationPlanner,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    AnaphoraRequirement,
    DiscourseDependency,
    DiscoursePlan,
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationStructurePlan,
    GenerationStructurePlanner,
    GenerationSyntaxLayerResolver,
    PlannedProposition,
    PlannedSentence,
    PropositionPlan,
    PropositionSlotFiller,
    SyntaxLinearizationObligation,
    SyntaxPlan,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningBudget,
    ReasoningObligation,
    ReasoningPlanResult,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.structure_order import StructureSlotDefinition
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)


_T = LogicEvidenceState(True, False)


def _world(count: int = 3):
    """构造来源、query scope 和若干 bound proposition。"""
    source = SourceRef(
        10101, 10102, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = query_scope(1, parent=document_scope(source))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((10103, index))
        for index in range(1, 10)
    ))
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(source, (10104, index)),
            concept_identity((10105, index)),
            occurrence_identity(
                source, start=index, end=index + 1, ordinal=0),
            context_scope_identity(source, (10106, index)),
            (),
        )
        for index in range(1, count + 1)
    )
    graph = PropositionTemplateGraph(tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity((10107, index)),
        )
        for index, definition in enumerate(definitions, start=1)
    ))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((10108, 1)), failures))
    bound = tuple(
        substituter.substitute(
            definition.proposition, graph, BindingEnvironment())
        for definition in definitions
    )
    return source, scope, bound


def _candidate(
        bound,
        source,
        scope,
        index: int,
        *,
        unresolved: ReasoningObligation | None = None,
        ) -> GenerationCandidate:
    """构造独立 competition 的 supported 候选和可选 unresolved reasoning。"""
    hypothesis = HypothesisKey(
        (10110, 1),
        (10110, index),
        (10110, 100 + index),
        scope,
        source,
    )
    evidence = EvidenceRecord(
        300 + index,
        hypothesis,
        EVIDENCE_SUPPORT,
        (10111, index),
        source,
        300 + index,
    )
    reasoning = None
    if unresolved is not None:
        obligation = ReasoningObligation(bound, _T, source, scope)
        reasoning = ReasoningPlanResult(
            obligation,
            _T,
            False,
            minimal_instruction_identity((10112, index)),
            ReasoningBudget(1, 0, 0),
            1,
            0,
            unresolved=(unresolved,),
            evidence_ids=(evidence.evidence_id,),
            hypotheses=(hypothesis,),
        )
    return GenerationCandidate(
        bound, _T, source, scope, (evidence,), reasoning)


def _request(*, with_unresolved: bool = False, count: int = 2):
    """构造 G-01 answer selection 所需请求和候选。"""
    source, scope, bound = _world(max(3, count))
    unresolved = (
        ReasoningObligation(bound[2], _T, source, scope)
        if with_unresolved else None
    )
    candidates = tuple(
        _candidate(
            bound[index],
            source,
            scope,
            index + 1,
            unresolved=unresolved if index == 0 else None,
        )
        for index in range(count)
    )
    goal = AnswerGenerationGoal(
        minimal_instruction_identity((10113, 1)),
        bound[0],
        _T,
        source,
        scope,
    )
    return GenerationPlanningRequest(goal, candidates), unresolved


def _plan_protocol(seed: int = 10120) -> GenerationPlanProtocol:
    """注入 G-00 六层和结果身份。"""
    return GenerationPlanProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 11)
    ))


def _content_protocol(seed: int = 10130) -> AnswerContentProtocol:
    """注入 G-01 五种 stance。"""
    return AnswerContentProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 6)
    ))


class _Policy:
    """选择全部候选为 answer，或按调用方注入固定 decision。"""

    def __init__(self, protocol, decision=None) -> None:
        self.protocol = protocol
        self.decision = decision

    def select(self, request, artifacts):
        """不读取 token/path，只按 typed key 返回内容选择。"""
        del artifacts
        if self.decision is not None:
            return self.decision
        return AnswerContentDecision(
            self.protocol.answer,
            minimal_instruction_identity((10131, 1)),
            request.candidate_keys(),
            (),
            (10132, 1),
        )


def _selection(request, protocol=None):
    """构造全部候选 answer 的 G-01 selection。"""
    protocol = protocol or _content_protocol()
    selector = AnswerContentSelector(protocol, _Policy(protocol))
    return selector.select(request), selector, protocol


def _discourse(selection, *, dependencies=(), open_questions=None):
    """按 selection 构造 typed discourse plan。"""
    if open_questions is None:
        open_questions = tuple(
            obligation
            for candidate in selection.request.candidates
            if candidate.stable_key() in set(selection.selected_candidate_keys)
            and candidate.reasoning is not None
            for obligation in candidate.reasoning.unresolved
        )
    return DiscoursePlan(
        selection.stable_key(),
        selection.selected_candidate_keys,
        tuple(dependencies),
        tuple(open_questions),
        (concept_identity((10140, 1)),),
    )


def _propositions(selection, *, mutate=None):
    """逐候选保留完整 Evidence/Hypothesis，并允许测试注入单项篡改。"""
    items = []
    selected = set(selection.selected_candidate_keys)
    for candidate in selection.request.candidates:
        if candidate.stable_key() not in selected:
            continue
        planned = PlannedProposition(
            candidate.stable_key(),
            candidate.proposition,
            candidate.state,
            candidate.source,
            candidate.scope,
            candidate.evidence,
            candidate.hypotheses,
            (concept_identity((10141, len(items) + 1)),),
        )
        items.append(planned)
    if mutate is not None:
        items[0] = mutate(items[0])
    return PropositionPlan(selection.stable_key(), tuple(items))


def _syntax(
        selection,
        discourse,
        *,
        ordinals=None,
        future_anaphora: bool = False,
        bad_filler: bool = False,
        ):
    """为每个命题构造独立句、S-07 slot/value 和线性化义务。"""
    keys = discourse.topological_order
    if ordinals is None:
        ordinals = tuple(range(len(keys)))
    candidate_map = {
        candidate.stable_key(): candidate
        for candidate in selection.request.candidates
    }
    sentences = []
    obligations = []
    slots = []
    for index, (key, ordinal) in enumerate(zip(keys, ordinals), start=1):
        candidate = candidate_map[key]
        sentence = structure_concept_identity((10150, index))
        structure = structure_concept_identity((10151, index))
        slot = structure_concept_identity((10152, index))
        slot_definition = StructureSlotDefinition(
            structure,
            slot,
            role_identity(
                (10153, index),
                owner=candidate.source.owner,
                versions=candidate.source.versions,
            ),
            concept_identity((10154, index)),
        )
        filler = (
            concept_identity((10155, index))
            if bad_filler and index == 1
            else candidate.proposition.template
        )
        value = StructureSlotValue(slot, filler)
        planned = PlannedSentence(
            sentence,
            structure,
            ordinal,
            (key,),
            (slot_definition,),
            (value,),
            (PropositionSlotFiller(key, candidate.proposition, value),),
            minimal_instruction_identity((10156, index)),
            candidate.source,
            candidate.scope,
        )
        obligation = SyntaxLinearizationObligation(
            sentence,
            structure,
            (value,),
            (structure_concept_identity((10157, index)),),
            (concept_identity((10158, index)),),
            minimal_instruction_identity((10159, index)),
            candidate.source,
            candidate.scope,
        )
        sentences.append(planned)
        obligations.append(obligation)
        slots.append(slot)
    anaphora = ()
    if len(sentences) >= 2:
        requirement_sentence = sentences[0] if future_anaphora else sentences[1]
        requirement_slot = slots[0] if future_anaphora else slots[1]
        antecedent = keys[1] if future_anaphora else keys[0]
        anaphora = (AnaphoraRequirement(
            requirement_sentence.sentence,
            requirement_slot,
            antecedent,
            minimal_instruction_identity((10160, 1)),
            (10161, 1),
        ),)
    return SyntaxPlan(
        selection.stable_key(),
        tuple(sentences),
        anaphora,
        tuple(obligations),
    )


class _DiscourseMapper:
    """按注入 builder 返回 discourse plan。"""

    def __init__(self, builder) -> None:
        self.builder = builder
        self.calls = 0

    def plan(self, selection):
        """记录调用，证明未来层不会被提前执行。"""
        self.calls += 1
        return self.builder(selection)


class _PropositionMapper:
    """按注入 builder 返回 proposition plan。"""

    def __init__(self, builder) -> None:
        self.builder = builder
        self.calls = 0

    def plan(self, selection, discourse):
        """记录调用并返回命题计划。"""
        self.calls += 1
        return self.builder(selection, discourse)


class _SyntaxMapper:
    """按注入 builder 返回 syntax plan或显式失败。"""

    def __init__(self, builder) -> None:
        self.builder = builder
        self.calls = 0

    def plan(self, selection, discourse, propositions):
        """记录调用并返回句法计划。"""
        self.calls += 1
        return self.builder(selection, discourse, propositions)


def _planner(selection, *, dependencies=(), syntax_kwargs=None):
    """构造默认完整 G-02 planner 和三类可观测 mapper。"""
    syntax_kwargs = syntax_kwargs or {}
    discourse_mapper = _DiscourseMapper(
        lambda current: _discourse(current, dependencies=dependencies))
    proposition_mapper = _PropositionMapper(
        lambda current, discourse: _propositions(current))
    syntax_mapper = _SyntaxMapper(
        lambda current, discourse, propositions: _syntax(
            current, discourse, **syntax_kwargs))
    return (
        GenerationStructurePlanner(
            discourse_mapper, proposition_mapper, syntax_mapper),
        discourse_mapper,
        proposition_mapper,
        syntax_mapper,
    )


def test_g02_builds_dependency_ordered_proposition_and_syntax_plan() -> None:
    """多命题按 typed dependency 组织，Evidence、slot、照应和线性化义务完整保留。"""
    request, _ = _request()
    selection, _, _ = _selection(request)
    first, second = selection.selected_candidate_keys
    dependency = DiscourseDependency(
        first,
        second,
        structure_concept_identity((10170, 1)),
        minimal_instruction_identity((10171, 1)),
        (10172, 1),
    )
    planner, _, _, _ = _planner(selection, dependencies=(dependency,))

    plan = planner.plan(selection)

    assert plan.discourse.topological_order == (first, second)
    assert len(plan.propositions.propositions) == 2
    assert len(plan.syntax.sentences) == 2
    assert len(plan.syntax.linearization) == 2
    assert plan.syntax.anaphora[0].antecedent_candidate_key == first
    assert all(item.evidence for item in plan.propositions.propositions)


def test_g02_rejects_dependency_cycle_and_future_sentence_order() -> None:
    """篇章环直接失败，句 ordinal 也不能逆转已声明依赖。"""
    request, _ = _request()
    selection, _, _ = _selection(request)
    first, second = selection.selected_candidate_keys
    relation = structure_concept_identity((10180, 1))
    dependencies = (
        DiscourseDependency(
            first, second, relation,
            minimal_instruction_identity((10181, 1)), (10182, 1)),
        DiscourseDependency(
            second, first, relation,
            minimal_instruction_identity((10181, 2)), (10182, 2)),
    )
    with pytest.raises(ValueError, match="含环"):
        _discourse(selection, dependencies=dependencies)

    dependency = dependencies[:1]
    planner, _, _, _ = _planner(
        selection,
        dependencies=dependency,
        syntax_kwargs={"ordinals": (1, 0)},
    )
    with pytest.raises(ValueError, match="违反 discourse"):
        planner.plan(selection)


def test_g02_preserves_all_s05_open_questions() -> None:
    """selected reasoning 的 unresolved obligation 不能在 discourse 层消失。"""
    request, unresolved = _request(with_unresolved=True)
    selection, _, _ = _selection(request)
    discourse_mapper = _DiscourseMapper(
        lambda current: _discourse(current, open_questions=()))
    planner = GenerationStructurePlanner(
        discourse_mapper,
        _PropositionMapper(lambda current, discourse: _propositions(current)),
        _SyntaxMapper(lambda current, discourse, propositions: _syntax(
            current, discourse)),
    )
    with pytest.raises(ValueError, match="open question"):
        planner.plan_discourse(selection)

    valid = _discourse(selection)
    assert valid.open_questions == (unresolved,)


def test_g02_rejects_proposition_evidence_loss_and_slot_content_loss() -> None:
    """命题层不能丢 Evidence，句法 slot 也必须实际承载 selected Proposition。"""
    request, _ = _request()
    selection, _, _ = _selection(request)
    discourse = _discourse(selection)
    bad_propositions = _propositions(
        selection, mutate=lambda item: replace(item, evidence=()))
    planner, _, _, _ = _planner(selection)
    with pytest.raises(ValueError, match="Evidence 身份"):
        planner.plan_syntax(selection, discourse, bad_propositions)

    planner, _, _, _ = _planner(
        selection, syntax_kwargs={"bad_filler": True})
    with pytest.raises(ValueError, match="filler"):
        planner.plan(selection)


def test_g02_rejects_template_only_filler_for_different_binding() -> None:
    """同一 template 的另一绑定不能借共享 identity 冒充当前 Proposition filler。"""
    request, _ = _request(count=1)
    selection, _, _ = _selection(request)
    discourse = _discourse(selection)
    propositions = _propositions(selection)
    syntax = _syntax(selection, discourse)
    sentence = syntax.sentences[0]
    candidate = request.candidates[0]
    wrong_proposition = replace(
        candidate.proposition,
        instruction=minimal_instruction_identity((10189, 1)),
    )
    wrong_filler = PropositionSlotFiller(
        candidate.stable_key(),
        wrong_proposition,
        sentence.values[0],
    )
    bad_sentence = replace(
        sentence,
        proposition_fillers=(wrong_filler,),
    )
    bad_syntax = replace(syntax, sentences=(bad_sentence,))

    with pytest.raises(ValueError, match="BoundProposition"):
        GenerationStructurePlan(
            selection, discourse, propositions, bad_syntax)


def test_g02_allows_non_proposition_typed_slot_values() -> None:
    """句子可保留额外 typed slot，不能把全部句法成分都误作 Proposition。"""
    request, _ = _request(count=1)
    selection, _, _ = _selection(request)
    discourse = _discourse(selection)
    propositions = _propositions(selection)
    syntax = _syntax(selection, discourse)
    sentence = syntax.sentences[0]
    extra_slot = StructureSlotDefinition(
        sentence.structure,
        structure_concept_identity((10189, 2)),
        role_identity(
            (10189, 3),
            owner=sentence.source.owner,
            versions=sentence.source.versions,
        ),
        concept_identity((10189, 4)),
    )
    extra_value = StructureSlotValue(
        extra_slot.slot,
        concept_identity((10189, 5)),
    )
    expanded_sentence = replace(
        sentence,
        slots=(*sentence.slots, extra_slot),
        values=(*sentence.values, extra_value),
    )
    expanded_obligation = replace(
        syntax.linearization[0],
        values=expanded_sentence.values,
    )
    expanded_syntax = replace(
        syntax,
        sentences=(expanded_sentence,),
        linearization=(expanded_obligation,),
    )

    plan = GenerationStructurePlan(
        selection, discourse, propositions, expanded_syntax)

    assert len(plan.syntax.sentences[0].values) == 2


def test_g02_rejects_future_anaphora_and_requires_response_act_surface() -> None:
    """照应不能指向未来句，refuse 也必须经真实 slot 进入后续 surface。"""
    request, _ = _request()
    selection, _, _ = _selection(request)
    planner, _, _, _ = _planner(
        selection, syntax_kwargs={"future_anaphora": True})
    with pytest.raises(ValueError, match="未来 sentence"):
        planner.plan(selection)

    protocol = _content_protocol(10190)
    empty_decision = AnswerContentDecision(
        protocol.refuse,
        minimal_instruction_identity((10191, 1)),
        (),
        (),
        (10192, 1),
    )
    empty_selection = AnswerContentSelector(
        protocol, _Policy(protocol, empty_decision)).select(request)
    empty_discourse = _discourse(empty_selection)
    empty_props = PropositionPlan(empty_selection.stable_key(), ())
    empty_syntax = SyntaxPlan(empty_selection.stable_key(), (), (), ())
    with pytest.raises(ValueError, match="response-act sentence"):
        GenerationStructurePlan(
            empty_selection, empty_discourse, empty_props, empty_syntax)

    sentence = structure_concept_identity((10194, 1))
    structure = structure_concept_identity((10194, 2))
    slot = structure_concept_identity((10194, 3))
    slot_definition = StructureSlotDefinition(
        structure,
        slot,
        role_identity((10194, 4)),
        concept_identity((10194, 5)),
    )
    value = StructureSlotValue(slot, empty_selection.stance)
    planned = PlannedSentence(
        sentence,
        structure,
        0,
        (),
        (slot_definition,),
        (value,),
        (),
        minimal_instruction_identity((10194, 6)),
        request.goal.source,
        request.goal.scope,
        empty_selection.stance,
    )
    obligation = SyntaxLinearizationObligation(
        sentence,
        structure,
        (value,),
        (),
        (),
        minimal_instruction_identity((10194, 7)),
        request.goal.source,
        request.goal.scope,
    )
    response_syntax = SyntaxPlan(
        empty_selection.stable_key(),
        (planned,),
        (),
        (obligation,),
    )

    plan = GenerationStructurePlan(
        empty_selection, empty_discourse, empty_props, response_syntax)

    assert plan.syntax.sentences[0].response_act == protocol.refuse
    assert plan.syntax.sentences[0].values[0].filler == protocol.refuse


def test_g02_rejects_competing_anaphora_for_same_sentence_slot() -> None:
    """同一 slot 不能同时指向两个 antecedent，避免 surface 层私选。"""
    request, _ = _request()
    selection, _, _ = _selection(request)
    discourse = _discourse(selection)
    syntax = _syntax(selection, discourse)
    first = syntax.anaphora[0]
    competing = replace(
        first,
        antecedent_candidate_key=selection.selected_candidate_keys[1],
        instruction=minimal_instruction_identity((10193, 1)),
        trace=(10193, 2),
    )

    with pytest.raises(ValueError, match="重复声明 anaphora"):
        replace(syntax, anaphora=(first, competing))


def test_g02_future_layer_failure_does_not_pollute_current_layer() -> None:
    """discourse/proposition 增量入口不得提前调用失败的 syntax mapper。"""
    request, _ = _request()
    selection, _, _ = _selection(request)
    discourse_mapper = _DiscourseMapper(lambda current: _discourse(current))
    proposition_mapper = _PropositionMapper(
        lambda current, discourse: _propositions(current))

    def fail_syntax(current, discourse, propositions):
        """模拟未来 syntax mapper 失败。"""
        del current, discourse, propositions
        raise RuntimeError("syntax failure")

    syntax_mapper = _SyntaxMapper(fail_syntax)
    planner = GenerationStructurePlanner(
        discourse_mapper, proposition_mapper, syntax_mapper)

    discourse = planner.plan_discourse(selection)
    propositions = planner.plan_propositions(selection, discourse)
    assert discourse_mapper.calls == 1
    assert proposition_mapper.calls == 1
    assert syntax_mapper.calls == 0
    with pytest.raises(RuntimeError, match="syntax failure"):
        planner.plan_syntax(selection, discourse, propositions)


class _SurfacePlaceholder:
    """仅用于证明 G-00 surface 层仍由 G-03 之外的占位测试控制。"""

    def __init__(self, layer, protocol) -> None:
        self.layer = layer
        self.protocol = protocol

    def resolve(self, request, prior):
        """返回无领域含义的 complete，不能作为 G-03 验收。"""
        del request, prior
        return GenerationLayerDecision(
            self.layer,
            self.protocol.complete,
            minimal_instruction_identity((10200, 1)),
            payload=(1,),
            trace=(10201, 1),
        )


def test_g02_connects_g00_middle_layers_with_independent_recomputation() -> None:
    """G-00 前五层完整接线，三层 resolver 只核验自身及既有上游。"""
    request, _ = _request()
    plan_protocol = _plan_protocol(10210)
    selection, selector, _ = _selection(request, _content_protocol(10220))
    first, second = selection.selected_candidate_keys
    dependency = DiscourseDependency(
        first,
        second,
        structure_concept_identity((10221, 1)),
        minimal_instruction_identity((10222, 1)),
        (10223, 1),
    )
    structure_planner, discourse_mapper, proposition_mapper, syntax_mapper = (
        _planner(selection, dependencies=(dependency,)))
    layer_protocol = GenerationStructureLayerProtocol(*tuple(
        minimal_instruction_identity((10224, index))
        for index in range(1, 4)
    ))
    registrations = (
        GenerationLayerRegistration(
            plan_protocol.stance_layer,
            GenerationStanceLayerResolver(plan_protocol, selector),
        ),
        GenerationLayerRegistration(
            plan_protocol.content_layer,
            GenerationContentLayerResolver(plan_protocol, selector),
        ),
        GenerationLayerRegistration(
            plan_protocol.discourse_layer,
            GenerationDiscourseLayerResolver(
                plan_protocol, layer_protocol, selector, structure_planner),
        ),
        GenerationLayerRegistration(
            plan_protocol.proposition_layer,
            GenerationPropositionLayerResolver(
                plan_protocol, layer_protocol, selector, structure_planner),
        ),
        GenerationLayerRegistration(
            plan_protocol.syntax_layer,
            GenerationSyntaxLayerResolver(
                plan_protocol, layer_protocol, selector, structure_planner),
        ),
        GenerationLayerRegistration(
            plan_protocol.surface_layer,
            _SurfacePlaceholder(plan_protocol.surface_layer, plan_protocol),
        ),
    )

    plan = GenerationPlanner(plan_protocol, registrations).plan(request)

    assert plan.complete
    assert discourse_mapper.calls == 3
    assert proposition_mapper.calls == 2
    assert syntax_mapper.calls == 1
    assert plan.layers[2].payload
    assert plan.layers[3].payload
    assert plan.layers[4].payload
