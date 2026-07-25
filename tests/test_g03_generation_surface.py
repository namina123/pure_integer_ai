"""G-03 词形、照应、typed 顺序、原子采用和表示渲染对抗测试。"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
    AliasResolutionSelector,
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerDecision,
    GenerationLayerRegistration,
    GenerationPlanner,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlan,
    GenerationStructureExecutionPlanner,
    GenerationStructureExecutionRequest,
    SentenceStructureExecution,
    SentenceStructureExecutionBudget,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    AnaphoraRequirement,
    GenerationStructurePlan,
    GenerationStructurePlanner,
    PlannedSentence,
    PropositionSlotFiller,
    SyntaxLinearizationObligation,
    SyntaxPlan,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfacePreview,
    GenerationSurfaceProtocol,
    GenerationSurfaceRequest,
    SurfaceSlotDirective,
    SurfaceSlotPreview,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderHypothesisEngine,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    RenderedSurface,
    UnicodeRepresentationRenderer,
    render_generation_surface,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_CONSUMER_ACCEPTED,
    ORDER_CONSUMER_UNKNOWN,
    StructureOrderConsumer,
    StructureOrderLinearizationResult,
    StructureOrderSearchBudget,
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
    TypedGenerationSurfaceRequestBuilder,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRuntime,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT

from tests.test_g02_generation_structure_plan import (
    _Policy,
    _discourse,
    _plan_protocol,
    _propositions,
    _request,
    _selection,
)
from tests.test_r00_relation_closure import (
    _candidate_runtime,
    _cloned_graphs,
    _projection_protocol,
    _r01_definition,
    _r01_recognition,
    _relation_protocol,
    _semantic_graph,
    _source,
)
from tests.test_s07_structure_order import (
    _ResolvedRule,
    _SemanticsResolver,
    _active_plan,
    _consumer_protocol,
    _domain,
    _graphs,
    _learning_protocol,
    _pattern,
    _semantic_reasons,
)


_BASE = 13200


@dataclass
class _AliasFixture:
    """保存 G-03 测试使用的 active R-01 owner 和可克隆图。"""

    backend: DictBackend
    semantic_graph: object
    candidate_graph: CandidateProjectionGraph
    closure: RelationClosureRuntime
    runtime: AliasRelationRuntime
    protocol: AliasResolutionProtocol
    specs: dict[str, RelationClosureCandidateSpec]

    def close(self) -> None:
        """关闭测试后端。"""
        self.backend.close()


def _alias_fixture(
        branch: ObjectIdentity,
        realizations: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        references: tuple[tuple[ObjectIdentity, ObjectIdentity], ...] = (),
        ) -> _AliasFixture:
    """为给定权威对象建立 direct realizes 和可选方向 refers active facts。"""
    if branch.object_kind != OBJECT_LANGUAGE_BRANCH:
        raise ValueError("测试 branch 必须是 LanguageBranch")
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology, _projection_protocol())
    candidate_runtime = _candidate_runtime(candidate_graph)
    closure_protocol = _relation_protocol()

    alias_relation = concept_identity((_BASE + 1, 1))
    refers_relation = concept_identity((_BASE + 1, 2))
    realizes_relation = concept_identity((_BASE + 1, 3))
    alias_roles = (role_identity((_BASE + 2, 1)), role_identity((_BASE + 2, 2)))
    refers_roles = (role_identity((_BASE + 2, 3)), role_identity((_BASE + 2, 4)))
    realizes_roles = (
        role_identity((_BASE + 2, 5)),
        role_identity((_BASE + 2, 6)),
        role_identity((_BASE + 2, 7)),
    )
    alias_schema_identity = structure_concept_identity((_BASE + 3, 1))
    refers_schema_identity = structure_concept_identity((_BASE + 3, 2))
    realizes_schema_identity = structure_concept_identity((_BASE + 3, 3))
    alias_schema = RelationSchema(
        alias_schema_identity,
        alias_relation,
        (
            RelationSlotSchema(alias_roles[0], frozenset({OBJECT_PROPOSITION}), 1, 1),
            RelationSlotSchema(alias_roles[1], frozenset({OBJECT_PROPOSITION}), 1, 1),
        ),
    )
    refers_schema = RelationSchema(
        refers_schema_identity,
        refers_relation,
        (
            RelationSlotSchema(refers_roles[0], frozenset({OBJECT_PROPOSITION}), 1, 1),
            RelationSlotSchema(refers_roles[1], frozenset({OBJECT_PROPOSITION}), 1, 1),
        ),
    )
    realizes_schema = RelationSchema(
        realizes_schema_identity,
        realizes_relation,
        (
            RelationSlotSchema(
                realizes_roles[0],
                frozenset(origin.object_kind for origin, _ in realizations),
                1,
                1,
            ),
            RelationSlotSchema(
                realizes_roles[1],
                frozenset({realizations[0][1].object_kind}),
                1,
                1,
            ),
            RelationSlotSchema(
                realizes_roles[2], frozenset({branch.object_kind}), 1, 1),
        ),
    )

    definitions: list[tuple[str, object, RelationSchema]] = []
    for index, (origin, target) in enumerate(references, start=1):
        definition, _ = _r01_definition(
            _source(_BASE + 20 + index),
            family=_BASE + 100 + index,
            relation=refers_relation,
            schema_identity=refers_schema_identity,
            role_fillers=((refers_roles[0], origin), (refers_roles[1], target)),
        )
        definitions.append((f"refers_{index}", definition, refers_schema))
    for index, (origin, representation) in enumerate(realizations, start=1):
        definition, _ = _r01_definition(
            _source(_BASE + 40 + index),
            family=_BASE + 200 + index,
            relation=realizes_relation,
            schema_identity=realizes_schema_identity,
            role_fillers=(
                (realizes_roles[0], origin),
                (realizes_roles[1], representation),
                (realizes_roles[2], branch),
            ),
        )
        definitions.append((f"realizes_{index}", definition, realizes_schema))
    for _, definition, _ in definitions:
        semantic_graph.define_atomic(
            definition,
            scope=document_scope(definition.source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        closure_protocol,
        (alias_schema, refers_schema, realizes_schema),
        engine=candidate_runtime.engine,
    )
    closure = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        closure_protocol,
    )
    specs = {}
    for index, (name, definition, schema) in enumerate(definitions, start=1):
        spec = RelationClosureCandidateSpec(
            definition,
            schema,
            (_BASE + 300, index),
            (_source(_BASE + 301), _source(_BASE + 302)),
        )
        specs[name] = spec
        closure.form(spec)
        trace = closure.recognize(_r01_recognition(spec, _BASE + 400 + index))
        assert trace.active_fact is not None

    protocol = AliasResolutionProtocol(
        alias_relation,
        (alias_schema_identity,),
        *alias_roles,
        minimal_instruction_identity((_BASE + 4, 1)),
        refers_relation,
        (refers_schema_identity,),
        *refers_roles,
        minimal_instruction_identity((_BASE + 4, 2)),
        realizes_relation,
        (realizes_schema_identity,),
        *realizes_roles,
        minimal_instruction_identity((_BASE + 4, 3)),
        minimal_instruction_identity((_BASE + 4, 4)),
        minimal_instruction_identity((_BASE + 4, 5)),
        minimal_instruction_identity((_BASE + 4, 6)),
    )
    return _AliasFixture(
        backend,
        semantic_graph,
        candidate_graph,
        closure,
        AliasRelationRuntime(closure, AliasResolutionSelector(protocol)),
        protocol,
        specs,
    )


def _surface_protocol(seed: int = _BASE + 500) -> GenerationSurfaceProtocol:
    """构造互异的 emit/silent 和七类 surface reason。"""
    return GenerationSurfaceProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 10)
    ))


def _structure_plan(
        branch: ObjectIdentity,
        *,
        request: GenerationPlanningRequest | None = None,
        slot_definitions: tuple[StructureSlotDefinition, ...] | None = None,
        constraints: tuple[ObjectIdentity, ...] = (),
        anaphora: bool = False,
        ) -> GenerationStructurePlan:
    """构造同句双 Proposition，并允许注入真实 S-07 schema/constraint。"""
    if request is None:
        base_request, _ = _request(count=2)
        request = GenerationPlanningRequest(
            replace(base_request.goal, target_branch=branch),
            base_request.candidates,
        )
    elif request.goal.target_branch != branch:
        raise ValueError("G-03 测试请求目标分支不一致")
    selection, _, _ = _selection(request)
    discourse = _discourse(selection)
    propositions = _propositions(selection)
    if slot_definitions is None:
        structure = structure_concept_identity((_BASE + 510, 1))
        slot_definitions = tuple(
            StructureSlotDefinition(
                structure,
                structure_concept_identity((_BASE + 511, index)),
                role_identity((_BASE + 512, index)),
                concept_identity((_BASE + 513, index)),
            )
            for index in range(1, 3)
        )
    if len(slot_definitions) != 2:
        raise ValueError("G-03 测试结构必须恰有两个 slot")
    structure = slot_definitions[0].structure
    if any(item.structure != structure for item in slot_definitions):
        raise ValueError("G-03 测试 slot 必须属于同一 structure")
    candidate_map = {
        item.stable_key(): item for item in selection.request.candidates}
    keys = selection.selected_candidate_keys
    values = tuple(
        StructureSlotValue(
            slot.slot,
            candidate_map[key].proposition.template,
        )
        for slot, key in zip(slot_definitions, keys)
    )
    sentence_identity = structure_concept_identity((_BASE + 514, 1))
    sentence = PlannedSentence(
        sentence_identity,
        structure,
        0,
        keys,
        slot_definitions,
        values,
        tuple(
            PropositionSlotFiller(
                key, candidate_map[key].proposition, value)
            for key, value in zip(keys, values)
        ),
        minimal_instruction_identity((_BASE + 515, 1)),
        request.goal.source,
        request.goal.scope,
    )
    obligation = SyntaxLinearizationObligation(
        sentence_identity,
        structure,
        sentence.values,
        constraints,
        (),
        minimal_instruction_identity((_BASE + 516, 1)),
        request.goal.source,
        request.goal.scope,
    )
    requirements = ()
    if anaphora:
        requirements = (AnaphoraRequirement(
            sentence_identity,
            slot_definitions[1].slot,
            keys[0],
            minimal_instruction_identity((_BASE + 517, 1)),
            (_BASE + 517, 2),
        ),)
    syntax = SyntaxPlan(
        selection.stable_key(),
        (sentence,),
        requirements,
        (obligation,),
    )
    return GenerationStructurePlan(
        selection, discourse, propositions, syntax)


def _manual_execution(
        structure: GenerationStructurePlan,
        *,
        reverse: bool = False,
        ) -> GenerationStructureExecutionPlan:
    """为无 active constraint 的单句计划构造受契约核验的 accepted 结果。"""
    sentence = structure.syntax.sentences[0]
    obligation = structure.syntax.linearization[0]
    budget = StructureOrderSearchBudget(20)
    request = GenerationStructureExecutionRequest(
        structure.syntax,
        (SentenceStructureExecutionBudget(sentence.sentence, budget),),
    )
    values = tuple(reversed(obligation.values)) if reverse else obligation.values
    result = StructureOrderLinearizationResult(
        ORDER_CONSUMER_ACCEPTED,
        values,
        (),
        (),
        1,
    )
    execution = SentenceStructureExecution(
        obligation,
        sentence.slots,
        (),
        budget,
        result,
    )
    return GenerationStructureExecutionPlan(request, (execution,))


def _incomplete_execution(
        structure: GenerationStructurePlan,
        ) -> GenerationStructureExecutionPlan:
    """构造上游 S-07 unknown 结果，验证 G-03 不提前执行 surface mapper。"""
    sentence = structure.syntax.sentences[0]
    obligation = structure.syntax.linearization[0]
    budget = StructureOrderSearchBudget(1)
    request = GenerationStructureExecutionRequest(
        structure.syntax,
        (SentenceStructureExecutionBudget(sentence.sentence, budget),),
    )
    result = StructureOrderLinearizationResult(
        ORDER_CONSUMER_UNKNOWN,
        (),
        (),
        (minimal_instruction_identity((_BASE + 518, 1)),),
        1,
    )
    execution = SentenceStructureExecution(
        obligation,
        sentence.slots,
        (),
        budget,
        result,
    )
    return GenerationStructureExecutionPlan(request, (execution,))


def _directives(
        structure: GenerationStructurePlan,
        protocol: GenerationSurfaceProtocol,
        *,
        silent_slots: frozenset[ObjectIdentity] = frozenset(),
        prefix_steps: dict[ObjectIdentity, tuple[ObjectIdentity, ...]] | None = None,
        ) -> tuple[SurfaceSlotDirective, ...]:
    """为全部 planned slot 注入动作、预算、prefix policy 和唯一 use key。"""
    prefix_steps = prefix_steps or {}
    anaphora_slots = {
        (item.sentence, item.slot) for item in structure.syntax.anaphora}
    directives = []
    ordinal = 0
    for sentence in structure.syntax.sentences:
        for value in sentence.values:
            ordinal += 1
            silent = value.slot in silent_slots
            is_anaphora = (sentence.sentence, value.slot) in anaphora_slots
            directives.append(SurfaceSlotDirective(
                sentence=sentence.sentence,
                slot=value.slot,
                action=(protocol.silent_action if silent else protocol.emit_action),
                instruction=minimal_instruction_identity((_BASE + 520, ordinal)),
                trace=(_BASE + 521, ordinal),
                surface_prefix_steps=prefix_steps.get(value.slot, ()),
                surface_budget=(None if silent else AliasRouteSearchBudget(30, 30, 30)),
                surface_use_key=(() if silent else (_BASE + 522, ordinal)),
                reference_budget=(
                    AliasRouteSearchBudget(30, 30, 30)
                    if is_anaphora else None
                ),
                reference_use_key=(
                    (_BASE + 523, ordinal) if is_anaphora else ()
                ),
            ))
    return tuple(directives)


def _request_for(
        structure: GenerationStructurePlan,
        execution: GenerationStructureExecutionPlan,
        alias: AliasRelationRuntime,
        *,
        silent_slots: frozenset[ObjectIdentity] = frozenset(),
        prefix_steps: dict[ObjectIdentity, tuple[ObjectIdentity, ...]] | None = None,
        ) -> GenerationSurfaceRequest:
    """构造与 alias protocol 策略兼容的完整 G-03 请求。"""
    del alias
    protocol = _surface_protocol()
    return GenerationSurfaceRequest(
        protocol,
        structure,
        execution,
        structure.selection.request.goal.target_branch,
        _directives(
            structure,
            protocol,
            silent_slots=silent_slots,
            prefix_steps=prefix_steps,
        ),
    )


def _templates(
        structure: GenerationStructurePlan,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    """按 candidate key 返回双 Proposition template。"""
    candidates = {
        item.stable_key(): item for item in structure.selection.request.candidates}
    first, second = structure.selection.selected_candidate_keys
    return candidates[first].proposition.template, candidates[second].proposition.template


def test_g03_l05b1_order_drives_representation_order_and_unicode_rendering():
    """active S-07 反向约束真实决定 Representation 顺序，renderer 不补字符。"""
    order_backend = DictBackend()
    alias_fixture = None
    try:
        graphs = _graphs(order_backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(slot_count=2)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        promotion = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, kind=60),
            event=60,
            instance=60,
        )
        applies, skipped, unknown = _semantic_reasons()
        resolver = _SemanticsResolver({
            promotion.constraint.constraint: _ResolvedRule(
                domain.slots[1], domain.slots[0], True, False, 0, 0, None),
        }, applies_reason=applies, skipped_reason=skipped,
           unknown_reason=unknown)
        consumer = StructureOrderConsumer(
            graphs.lifecycle, resolver, _consumer_protocol())
        execution_planner = GenerationStructureExecutionPlanner(
            graphs.lifecycle, consumer)
        branch = language_branch_identity((_BASE + 530, 1))
        structure = _structure_plan(
            branch,
            slot_definitions=promotion.slots,
            constraints=(promotion.constraint.constraint,),
        )
        first, second = _templates(structure)
        family = (_BASE + 531, 1)
        rep_first = representation_identity(family, (0x7532,))
        rep_second = representation_identity(family, (0x5E8F,))
        alias_fixture = _alias_fixture(
            branch,
            ((first, rep_first), (second, rep_second)),
        )

        class _ExecutionRequests:
            """为当前单句 structure 注入固定搜索预算。"""

            def build(self, current):
                """返回逐句预算且不替换 SyntaxPlan。"""
                sentence = current.syntax.sentences[0]
                return GenerationStructureExecutionRequest(
                    current.syntax,
                    (SentenceStructureExecutionBudget(
                        sentence.sentence, StructureOrderSearchBudget(30)),),
                )

        class _DirectiveMapper:
            """为全部 slot 注入 direct-only surface 选择。"""

            def plan(self, current, execution, target_branch):
                """返回完整 emit 指令并核验目标分支。"""
                assert execution.complete
                assert target_branch == branch
                return _directives(current, _surface_protocol())

        surface_protocol = _surface_protocol()

        class _MatchingDirectiveMapper(_DirectiveMapper):
            """复用 builder 持有的同一 surface protocol。"""

            def plan(self, current, execution, target_branch):
                """返回与 request builder protocol 一致的指令。"""
                assert execution.complete
                assert target_branch == branch
                return _directives(current, surface_protocol)

        builder = TypedGenerationSurfaceRequestBuilder(
            surface_protocol,
            execution_planner,
            _ExecutionRequests(),
            _MatchingDirectiveMapper(),
        )
        request = builder.build(structure)
        run = GenerationSurfaceRuntime(alias_fixture.runtime).plan(request)

        assert run.complete
        assert request.execution.sentences[0].result.values == tuple(reversed(
            structure.syntax.sentences[0].values))
        assert run.plan.representations == (rep_second, rep_first)
        renderer = UnicodeRepresentationRenderer(
            family, minimal_instruction_identity((_BASE + 532, 1)))
        rendered = render_generation_surface(run.plan, renderer)
        assert renderer.text(rendered) == "序甲"
        assert rendered.units == (0x5E8F, 0x7532)
        with pytest.raises(ValueError, match="units/trace"):
            renderer.text(replace(rendered, units=(0x5E8F, 0x4E59)))

        missing_constraints = _structure_plan(
            branch,
            slot_definitions=promotion.slots,
            constraints=(),
        )
        with pytest.raises(ValueError, match="active S-07 constraint"):
            builder.build(missing_constraints)
    finally:
        if alias_fixture is not None:
            alias_fixture.close()
        order_backend.close()


def test_g03_late_ambiguity_keeps_all_prior_relation_uses_unwritten(monkeypatch):
    """前 slot 成功而后 slot 歧义时，整次 plan 不得留下任何前缀采用账。"""
    branch = language_branch_identity((_BASE + 540, 1))
    structure = _structure_plan(branch)
    execution = _manual_execution(structure)
    first, second = _templates(structure)
    family = (_BASE + 541, 1)
    rep_first = representation_identity(family, (0x7532,))
    rep_second_a = representation_identity(family, (0x4E00,))
    rep_second_b = representation_identity(family, (0x4E8C,))
    fixture = _alias_fixture(
        branch,
        (
            (first, rep_first),
            (second, rep_second_a),
            (second, rep_second_b),
        ),
    )
    try:
        calls = []
        original = fixture.runtime.preview_surface

        def counted(*args, **kwargs):
            """记录 surface 查询顺序后委托真实 R-01 runtime。"""
            calls.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(fixture.runtime, "preview_surface", counted)
        before = fixture.runtime.state_key()
        request = _request_for(structure, execution, fixture.runtime)

        run = GenerationSurfaceRuntime(fixture.runtime).plan(request)

        assert not run.complete
        assert run.preview.reason == request.protocol.surface_ambiguous_reason
        assert calls == [first, second]
        assert fixture.runtime.state_key() == before
        assert all(
            not fixture.closure.audit(spec).consumer_used
            for spec in fixture.specs.values()
        )
    finally:
        fixture.close()


def test_g03_first_missing_stops_future_queries_and_all_silent_is_rejected(
        monkeypatch):
    """首 slot missing 后未来零查询；每句全 silent 不能伪造完整空输出。"""
    branch = language_branch_identity((_BASE + 550, 1))
    structure = _structure_plan(branch)
    execution = _manual_execution(structure)
    first, second = _templates(structure)
    family = (_BASE + 551, 1)
    fixture = _alias_fixture(
        branch,
        ((second, representation_identity(family, (0x540E,))),),
    )
    try:
        calls = []
        original = fixture.runtime.preview_surface

        def counted(*args, **kwargs):
            """记录首个 missing 后是否仍访问未来 slot。"""
            calls.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(fixture.runtime, "preview_surface", counted)
        request = _request_for(structure, execution, fixture.runtime)
        run = GenerationSurfaceRuntime(fixture.runtime).plan(request)
        assert not run.complete
        assert run.preview.reason == request.protocol.surface_missing_reason
        assert calls == [first]

        slots = frozenset(
            value.slot for value in structure.syntax.sentences[0].values)
        with pytest.raises(ValueError, match="至少包含一个 emit"):
            _request_for(
                structure,
                execution,
                fixture.runtime,
                silent_slots=slots,
            )
    finally:
        fixture.close()


def test_g03_structure_incomplete_blocks_directive_mapper_and_r01_queries(
        monkeypatch):
    """L-05B1 未完成时 directive mapper 与 reference/surface 查询均必须零调用。"""
    branch = language_branch_identity((_BASE + 555, 1))
    structure = _structure_plan(branch)
    execution = _incomplete_execution(structure)
    first, second = _templates(structure)
    family = (_BASE + 556, 1)
    fixture = _alias_fixture(
        branch,
        (
            (first, representation_identity(family, (0x7532,))),
            (second, representation_identity(family, (0x5E8F,))),
        ),
    )
    try:
        class _IncompletePlanner(GenerationStructureExecutionPlanner):
            """返回预先构造的 incomplete execution，不访问 S-07 图。"""

            def __init__(self, value) -> None:
                self.value = value

            def execute(self, request):
                """核验 SyntaxPlan 后返回上游 unknown 结果。"""
                assert request.syntax == self.value.request.syntax
                return self.value

        class _ExecutionRequests:
            """为 builder 提供与 structure 一致的最小执行请求。"""

            def build(self, current):
                """返回当前 syntax 及唯一句预算。"""
                return execution.request

        class _ForbiddenDirectives:
            """记录并拒绝任何上游失败后的 surface mapper 调用。"""

            calls = 0

            def plan(self, current, current_execution, target_branch):
                """一旦被调用即证明失败传播越层。"""
                del current, current_execution, target_branch
                self.calls += 1
                raise AssertionError("structure incomplete 后不得调用 directive mapper")

        directives = _ForbiddenDirectives()
        builder = TypedGenerationSurfaceRequestBuilder(
            _surface_protocol(),
            _IncompletePlanner(execution),
            _ExecutionRequests(),
            directives,
        )
        request = builder.build(structure)
        assert request.directives == ()
        assert directives.calls == 0

        monkeypatch.setattr(
            fixture.runtime,
            "preview_reference",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("structure incomplete 后不得查询 reference")),
        )
        monkeypatch.setattr(
            fixture.runtime,
            "preview_surface",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("structure incomplete 后不得查询 surface")),
        )
        preview = GenerationSurfaceRuntime(fixture.runtime).preview(request)
        assert not preview.complete
        assert preview.reason == request.protocol.structure_incomplete_reason
        assert preview.slots == ()
    finally:
        fixture.close()


@pytest.mark.parametrize(
    ("reference_mode", "reason_name"),
    (
        ("missing", "reference_missing_reason"),
        ("ambiguous", "reference_ambiguous_reason"),
        ("mismatch", "reference_mismatch_reason"),
    ),
)
def test_g03_reference_failures_are_typed_and_stop_before_anaphora_surface(
        monkeypatch, reference_mode, reason_name):
    """照应 missing/ambiguous/mismatch 均分型失败且不查询该 slot 的词形。"""
    branch = language_branch_identity((_BASE + 560, 1))
    structure = _structure_plan(branch, anaphora=True)
    execution = _manual_execution(structure)
    first, second = _templates(structure)
    wrong = proposition_identity(
        structure.selection.request.goal.source, (_BASE + 561, 1))
    family = (_BASE + 562, 1)
    realizations = (
        (first, representation_identity(family, (0x5148,))),
        (second, representation_identity(family, (0x540E,))),
    )
    if reference_mode == "missing":
        references = ()
    elif reference_mode == "ambiguous":
        references = ((second, first), (second, wrong))
    else:
        references = ((second, wrong),)
    fixture = _alias_fixture(branch, realizations, references)
    try:
        surface_calls = []
        original = fixture.runtime.preview_surface

        def counted(*args, **kwargs):
            """记录 reference 失败后是否错误继续选择当前 slot surface。"""
            surface_calls.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(fixture.runtime, "preview_surface", counted)
        before = fixture.runtime.state_key()
        request = _request_for(structure, execution, fixture.runtime)
        run = GenerationSurfaceRuntime(fixture.runtime).plan(request)

        assert not run.complete
        assert run.preview.reason == getattr(request.protocol, reason_name)
        assert surface_calls == [first]
        assert fixture.runtime.state_key() == before
    finally:
        fixture.close()


def test_g03_anaphora_uses_reference_then_direct_surface_without_dereference():
    """同一 filler 先唯一命中 antecedent，再按显式 direct policy 选择自身词形。"""
    branch = language_branch_identity((_BASE + 570, 1))
    structure = _structure_plan(branch, anaphora=True)
    execution = _manual_execution(structure)
    first, second = _templates(structure)
    family = (_BASE + 571, 1)
    rep_first = representation_identity(family, (0x7532,))
    rep_second = representation_identity(family, (0x5B83,))
    fixture = _alias_fixture(
        branch,
        ((first, rep_first), (second, rep_second)),
        ((second, first),),
    )
    try:
        request = _request_for(structure, execution, fixture.runtime)
        run = GenerationSurfaceRuntime(fixture.runtime).plan(request)
        assert run.complete
        assert run.plan.representations == (rep_first, rep_second)
        assert len(run.plan.adoptions) == 3

        second_slot = structure.syntax.sentences[0].values[1].slot
        dereferencing = _request_for(
            structure,
            execution,
            fixture.runtime,
            prefix_steps={
                second_slot: (fixture.protocol.refers_step,),
            },
        )
        before = fixture.runtime.state_key()
        failed = GenerationSurfaceRuntime(fixture.runtime).plan(dereferencing)
        assert not failed.complete
        assert failed.preview.reason == dereferencing.protocol.surface_ambiguous_reason
        assert fixture.runtime.state_key() == before
    finally:
        fixture.close()


def test_g03_target_branch_and_proposal_policy_cannot_drift():
    """目标 branch、reference kind 和 surface prefix trace 任一漂移均 fail closed。"""
    branch = language_branch_identity((_BASE + 580, 1))
    other_branch = language_branch_identity((_BASE + 580, 2))
    structure = _structure_plan(branch)
    execution = _manual_execution(structure)
    first, second = _templates(structure)
    family = (_BASE + 581, 1)
    fixture = _alias_fixture(
        branch,
        (
            (first, representation_identity(family, (0x7532,))),
            (second, representation_identity(family, (0x5E8F,))),
        ),
    )
    try:
        protocol = _surface_protocol()
        directives = _directives(structure, protocol)
        missing_goal = replace(
            structure.selection.request.goal, target_branch=None)
        missing_request = GenerationPlanningRequest(
            missing_goal, structure.selection.request.candidates)
        missing_selection, _, _ = _selection(missing_request)
        missing_structure = GenerationStructurePlan(
            missing_selection,
            _discourse(missing_selection),
            _propositions(missing_selection),
            replace(
                structure.syntax,
                selection_key=missing_selection.stable_key(),
            ),
        )
        with pytest.raises(ValueError, match="缺少目标 LanguageBranch"):
            GenerationSurfaceRequest(
                protocol,
                missing_structure,
                _manual_execution(missing_structure),
                branch,
                _directives(missing_structure, protocol),
            )

        with pytest.raises(ValueError, match="generation goal"):
            GenerationSurfaceRequest(
                protocol, structure, execution, other_branch, directives)
        with pytest.raises(ValueError, match="精确覆盖"):
            GenerationSurfaceRequest(
                protocol,
                structure,
                execution,
                branch,
                directives[:-1],
            )

        wrong_branch = fixture.runtime.preview_surface(
            first,
            other_branch,
            budget=AliasRouteSearchBudget(20, 20, 20),
            allowed_prefix_steps=(),
        )
        slot = structure.syntax.sentences[0].values[0]
        preview_slot = SurfaceSlotPreview(
            directives[0], slot, surface=wrong_branch)
        request = GenerationSurfaceRequest(
            protocol, structure, execution, branch, directives)
        with pytest.raises(ValueError, match="目标分支"):
            GenerationSurfacePreview(
                request,
                protocol.surface_missing_reason,
                (preview_slot,),
            )

        bad_directive = replace(
            directives[0],
            surface_prefix_steps=(minimal_instruction_identity((_BASE + 582, 1)),),
        )
        bad_request = GenerationSurfaceRequest(
            protocol,
            structure,
            execution,
            branch,
            (bad_directive, directives[1]),
        )
        with pytest.raises(ValueError, match="未注册 R-01"):
            GenerationSurfaceRuntime(fixture.runtime).preview(bad_request)
    finally:
        fixture.close()


def test_g03_held_out_clone_and_renderer_family_are_isolated():
    """held-out 采用只写克隆账本，Unicode renderer 也拒绝其他表示族。"""
    branch = language_branch_identity((_BASE + 590, 1))
    structure = _structure_plan(branch)
    execution = _manual_execution(structure, reverse=True)
    first, second = _templates(structure)
    family = (_BASE + 591, 1)
    rep_first = representation_identity(family, (0x7532,))
    rep_second = representation_identity(family, (0x5E8F,))
    fixture = _alias_fixture(
        branch, ((first, rep_first), (second, rep_second)))
    clone = None
    try:
        host_backend = fixture.backend.snapshot()
        host_closure = fixture.closure.state_key()
        host_alias = fixture.runtime.state_key()
        clone, semantic_graph, candidate_graph = _cloned_graphs(fixture)
        closure = fixture.closure.clone_for_evaluation(
            semantic_graph, candidate_graph)
        alias = fixture.runtime.clone_for_runtime(closure)
        request = _request_for(structure, execution, alias)

        run = GenerationSurfaceRuntime(alias).plan(request)

        assert run.complete
        assert run.plan.representations == (rep_second, rep_first)
        assert alias.state_key() != host_alias
        assert fixture.backend.snapshot() == host_backend
        assert fixture.closure.state_key() == host_closure
        assert fixture.runtime.state_key() == host_alias

        renderer = UnicodeRepresentationRenderer(
            family, minimal_instruction_identity((_BASE + 592, 1)))
        wrong_family = representation_identity(
            (_BASE + 593, 1), (0x9519,))
        with pytest.raises(ValueError, match="未注册表示族"):
            renderer.render((wrong_family,))

        class _ReorderingRenderer:
            """故意替换输入顺序以验证通用边界守恒。"""

            def render(self, representations):
                """返回逆序输入，预期被 render_generation_surface 拒绝。"""
                return RenderedSurface(
                    minimal_instruction_identity((_BASE + 594, 1)),
                    tuple(reversed(representations)),
                    (1,),
                    (_BASE + 594, 2),
                )

        with pytest.raises(ValueError, match="替换或重排"):
            render_generation_surface(run.plan, _ReorderingRenderer())
    finally:
        if clone is not None:
            clone.close()
        fixture.close()


class _StaticMapper:
    """为 G-00 第六层重算提供确定的 G-02 三层 mapper。"""

    def __init__(self, value) -> None:
        self.value = value

    def plan(self, *args):
        """返回预先构造且仍由共享 planner 交叉核验的 typed 值。"""
        del args
        return self.value


class _StaticSurfaceBuilder:
    """把重建 structure 汇合到已核验的无约束 execution 和 directives。"""

    def __init__(self, protocol) -> None:
        self.protocol = protocol

    def build(self, structure):
        """从传入 structure 重建 request，不读取旧生成链。"""
        execution = _manual_execution(structure)
        return GenerationSurfaceRequest(
            self.protocol,
            structure,
            execution,
            structure.selection.request.goal.target_branch,
            _directives(structure, self.protocol),
        )


class _FixedLayerResolver:
    """为 G-00 前五层返回可被 surface resolver 独立复核的固定 decision。"""

    def __init__(self, layer, protocol, selected, payload, ordinal) -> None:
        self.layer = layer
        self.protocol = protocol
        self.selected = selected
        self.payload = payload
        self.ordinal = ordinal

    def resolve(self, request, prior):
        """返回当前层 complete decision，并保留输入调用 trace。"""
        del request, prior
        return GenerationLayerDecision(
            self.layer,
            self.protocol.complete,
            minimal_instruction_identity((_BASE + 600, self.ordinal)),
            self.selected,
            self.payload,
            (_BASE + 601, self.ordinal),
        )


def test_g03_connects_g00_surface_layer_and_propagates_typed_failure():
    """G-00 第六层独立重算前五层，成功提交，歧义则返回 typed failed。"""
    branch = language_branch_identity((_BASE + 610, 1))
    structure = _structure_plan(branch)
    first, second = _templates(structure)
    family = (_BASE + 611, 1)
    rep_first = representation_identity(family, (0x7532,))
    rep_second = representation_identity(family, (0x5E8F,))
    fixture = _alias_fixture(
        branch, ((first, rep_first), (second, rep_second)))
    try:
        planner_protocol = _plan_protocol(_BASE + 612)
        surface_protocol = _surface_protocol(_BASE + 613)
        structure_planner = GenerationStructurePlanner(
            _StaticMapper(structure.discourse),
            _StaticMapper(structure.propositions),
            _StaticMapper(structure.syntax),
        )
        surface_resolver = GenerationSurfaceLayerResolver(
            planner_protocol,
            AnswerContentSelector(
                structure.selection.protocol,
                _Policy(structure.selection.protocol),
            ),
            structure_planner,
            _StaticSurfaceBuilder(surface_protocol),
            GenerationSurfaceRuntime(fixture.runtime),
        )
        payloads = (
            structure.selection.stable_key(),
            structure.selection.stable_key(),
            structure.discourse.stable_key(),
            structure.propositions.stable_key(),
            structure.syntax.stable_key(),
        )
        registrations = []
        for ordinal, (layer, payload) in enumerate(zip(
                planner_protocol.layers()[:5], payloads), start=1):
            resolver = _FixedLayerResolver(
                layer,
                planner_protocol,
                structure.selection.selected_candidate_keys,
                payload,
                ordinal,
            )
            registrations.append(GenerationLayerRegistration(layer, resolver))
        registrations.append(GenerationLayerRegistration(
            planner_protocol.surface_layer, surface_resolver))
        planner = GenerationPlanner(planner_protocol, tuple(registrations))

        result = planner.plan(structure.selection.request)

        assert result.complete
        assert result.layers[-1].outcome == planner_protocol.complete
        assert result.layers[-1].reason == surface_protocol.complete_reason
    finally:
        fixture.close()

    ambiguous_fixture = _alias_fixture(
        branch,
        (
            (first, rep_first),
            (second, rep_second),
            (second, representation_identity(family, (0x4E8C,))),
        ),
    )
    try:
        planner_protocol = _plan_protocol(_BASE + 614)
        surface_protocol = _surface_protocol(_BASE + 615)
        structure_planner = GenerationStructurePlanner(
            _StaticMapper(structure.discourse),
            _StaticMapper(structure.propositions),
            _StaticMapper(structure.syntax),
        )
        surface_resolver = GenerationSurfaceLayerResolver(
            planner_protocol,
            AnswerContentSelector(
                structure.selection.protocol,
                _Policy(structure.selection.protocol),
            ),
            structure_planner,
            _StaticSurfaceBuilder(surface_protocol),
            GenerationSurfaceRuntime(ambiguous_fixture.runtime),
        )
        payloads = (
            structure.selection.stable_key(),
            structure.selection.stable_key(),
            structure.discourse.stable_key(),
            structure.propositions.stable_key(),
            structure.syntax.stable_key(),
        )
        registrations = tuple(
            GenerationLayerRegistration(
                layer,
                _FixedLayerResolver(
                    layer,
                    planner_protocol,
                    structure.selection.selected_candidate_keys,
                    payload,
                    ordinal,
                ),
            )
            for ordinal, (layer, payload) in enumerate(zip(
                planner_protocol.layers()[:5], payloads), start=1)
        ) + (GenerationLayerRegistration(
            planner_protocol.surface_layer, surface_resolver),)

        result = GenerationPlanner(
            planner_protocol, registrations).plan(
                structure.selection.request)

        assert not result.complete
        assert result.layers[-1].outcome == planner_protocol.failed
        assert result.layers[-1].reason == surface_protocol.surface_ambiguous_reason
        assert not any(
            ambiguous_fixture.closure.audit(spec).consumer_used
            for spec in ambiguous_fixture.specs.values()
        )
    finally:
        ambiguous_fixture.close()
