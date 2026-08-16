"""L-05B2B semantic predicate 到 S-07/R-01 connector 的对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    AnaphoraRequirement,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderHypothesisEngine,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderConsumer,
    StructureOrderSearchBudget,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundRoleBinding
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.language_generation_connector import (
    BoundPropositionAnaphoraDeclaration,
    BoundPropositionAnaphoraDeclarations,
    BoundPropositionAnaphoraLink,
    BoundPropositionDiscourseDeclaration,
    BoundPropositionDiscourseDeclarations,
    BoundPropositionDiscourseDependency,
    LanguageConnectorExecutionRequestMapper,
    LanguageConnectorOrdinalDefinition,
    LanguageConnectorSlotBinding,
    LanguageConnectorSurfaceDirective,
    LanguageConnectorSurfaceDirectiveMapper,
    LanguageConnectorSurfaceRuntimePolicy,
    LanguageConnectorTemplateRuntimePolicy,
    LanguageConnectorValueProtocol,
    LanguageGenerationConnector,
    LanguageGenerationConnectorError,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorRuntimePolicy,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g02_generation_structure_plan import (
    _request,
    _selection,
)
from tests.test_g03_generation_surface import (
    _alias_fixture,
    _surface_protocol,
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


_BASE = 15400


def _selection_with_role(branch, role, filler):
    """给单一 supported 命题增加一个真实 Role filler 和目标语言分支。"""
    request, _ = _request(count=1)
    candidate = request.candidates[0]
    proposition = replace(
        candidate.proposition,
        bindings=(BoundRoleBinding(role, filler),),
    )
    candidate = replace(candidate, proposition=proposition)
    goal = replace(
        request.goal,
        proposition=proposition,
        target_branch=branch,
    )
    current = GenerationPlanningRequest(goal, (candidate,))
    selection, _, _ = _selection(current)
    return selection


def test_bound_proposition_discourse_declaration_rebinds_query_scoped_candidates():
    """课程命题模板跨 query scope 重绑 candidate，重复命中不得私选。"""
    request, _unused = _request(count=2)
    first, second = request.candidates
    declaration = BoundPropositionDiscourseDeclaration(
        (first.proposition, second.proposition),
        (BoundPropositionDiscourseDependency(
            first.proposition,
            second.proposition,
            structure_concept_identity((_BASE + 90, 1)),
            minimal_instruction_identity((_BASE + 90, 2)),
            (_BASE + 90, 3),
        ),),
        request.goal.source,
        (_BASE + 90, 4),
    )
    provider = BoundPropositionDiscourseDeclarations((declaration,))

    first_scope = query_scope(
        1,
        parent=episode_scope(
            _BASE + 90,
            parent=document_scope(request.goal.source),
        ),
    )
    first_request = GenerationPlanningRequest(
        replace(request.goal, scope=first_scope),
        (replace(first, scope=first_scope), replace(second, scope=first_scope)),
    )
    first_selection, _unused_first, _unused_second = _selection(first_request)
    first_runtime = provider.declaration(first_selection)
    assert first_runtime is not None
    assert first_runtime.candidate_keys == first_selection.selected_candidate_keys

    second_scope = query_scope(
        2,
        parent=episode_scope(
            _BASE + 91,
            parent=document_scope(request.goal.source),
        ),
    )
    second_request = GenerationPlanningRequest(
        replace(request.goal, scope=second_scope),
        (replace(first, scope=second_scope), replace(second, scope=second_scope)),
    )
    second_selection, _unused_first, _unused_second = _selection(second_request)
    second_runtime = provider.declaration(second_selection)
    assert second_runtime is not None
    assert second_runtime.candidate_keys == second_selection.selected_candidate_keys
    assert second_runtime.stable_key() != first_runtime.stable_key()
    assert provider.clone_for_evaluation() is not provider
    assert provider.clone_for_evaluation().state_key() == provider.state_key()

    duplicate_request = GenerationPlanningRequest(
        replace(request.goal, scope=second_scope),
        (
            replace(first, scope=second_scope),
            replace(
                second,
                proposition=first.proposition,
                scope=second_scope,
            ),
        ),
    )
    duplicate_selection, _unused_first, _unused_second = _selection(
        duplicate_request)
    with pytest.raises(
            LanguageGenerationConnectorError,
            match="同一 BoundProposition 命中多个 selected candidate"):
        provider.declaration(duplicate_selection)


def test_anaphora_declaration_injects_runtime_sentence_and_rejects_future():
    """课程 link 按 exact candidate 集重绑句实例，未来 antecedent 由 G-02 拒绝。"""
    request, _unused = _request(count=2)
    branch = _domain(variant=98, slot_count=2).language
    planning = GenerationPlanningRequest(
        replace(request.goal, target_branch=branch),
        request.candidates,
    )
    selection, _unused_first, _unused_second = _selection(planning)
    first, second = planning.candidates
    value_protocol = LanguageConnectorValueProtocol(*tuple(
        minimal_instruction_identity((_BASE + 100, index))
        for index in range(1, 5)
    ))
    surface_protocol = _surface_protocol(_BASE + 101)
    templates = []
    policies = []
    sentence_slots = []
    for index, candidate in enumerate(planning.candidates, start=1):
        structure = structure_concept_identity((_BASE + 102, index, 1))
        slot = StructureSlotDefinition(
            structure,
            structure_concept_identity((_BASE + 102, index, 2)),
            role_identity((_BASE + 102, index, 3)),
            concept_identity((_BASE + 102, index, 4)),
        )
        connector_id = structure_concept_identity(
            (_BASE + 103, index, 1))
        sentence = structure_concept_identity((_BASE + 103, index, 2))
        templates.append(LanguageGenerationConnectorTemplate(
            connector_id,
            branch,
            candidate.proposition.structure,
            candidate.proposition.predicate,
            sentence,
            structure,
            (slot,),
            (LanguageConnectorSlotBinding(
                structure_concept_identity((_BASE + 104, index, 1)),
                slot.slot,
                value_protocol.proposition_source,
            ),),
            structure_concept_identity((_BASE + 104, index, 2)),
            (),
            structure_concept_identity((_BASE + 104, index, 3)),
            (),
            minimal_instruction_identity((_BASE + 104, index, 4)),
            minimal_instruction_identity((_BASE + 104, index, 5)),
            (LanguageConnectorSurfaceDirective(
                structure_concept_identity((_BASE + 104, index, 6)),
                slot.slot,
                surface_protocol.emit_action,
                minimal_instruction_identity((_BASE + 104, index, 7)),
                structure_concept_identity((_BASE + 104, index, 8)),
                (),
            ),),
        ))
        policies.append(LanguageConnectorTemplateRuntimePolicy(
            connector_id,
            (LanguageConnectorSurfaceRuntimePolicy(
                slot.slot,
                (_BASE + 105, index, 1),
                AliasRouteSearchBudget(16, 16, 16),
                (_BASE + 105, index, 2),
                (AliasRouteSearchBudget(16, 16, 16)
                 if index == 2 else None),
                ((_BASE + 105, index, 3) if index == 2 else ()),
            ),),
        ))
        sentence_slots.append((sentence, slot.slot))
    registry = LanguageGenerationConnectorRegistry(
        value_protocol, tuple(templates))
    runtime_policy = LanguageGenerationConnectorRuntimePolicy(
        (_BASE + 106, 1),
        StructureOrderSearchBudget(16),
        tuple(policies),
    )
    discourse = BoundPropositionDiscourseDeclarations((
        BoundPropositionDiscourseDeclaration(
            (first.proposition, second.proposition),
            (BoundPropositionDiscourseDependency(
                first.proposition,
                second.proposition,
                structure_concept_identity((_BASE + 107, 1)),
                minimal_instruction_identity((_BASE + 107, 2)),
                (_BASE + 107, 3),
            ),),
            planning.goal.source,
            (_BASE + 107, 4),
        ),
    ))

    def provider_for(antecedent, referring, sentence_slot, seed):
        sentence, slot = sentence_slot
        return BoundPropositionAnaphoraDeclarations((
            BoundPropositionAnaphoraDeclaration(
                (first.proposition, second.proposition),
                (BoundPropositionAnaphoraLink(
                    antecedent.proposition,
                    referring.proposition,
                    sentence,
                    slot,
                    minimal_instruction_identity((_BASE + 108, seed, 1)),
                    (_BASE + 108, seed, 2),
                ),),
                planning.goal.source,
                (_BASE + 108, seed, 3),
            ),
        ))

    provider = provider_for(first, second, sentence_slots[1], 1)
    connector = LanguageGenerationConnector(
        registry,
        runtime_policy,
        surface_protocol,
        discourse_declarations=discourse,
        anaphora_declarations=provider,
    )
    plan = connector.structure_planner().plan(selection)
    requirement = plan.syntax.anaphora[0]
    assert len(plan.syntax.sentences) == 2
    assert requirement.antecedent_candidate_key == first.stable_key()
    assert requirement.address == plan.syntax.sentences[1].address
    assert requirement.slot == sentence_slots[1][1]
    assert provider.clone_for_evaluation() is not provider
    assert provider.clone_for_evaluation().state_key() == provider.state_key()
    assert connector.anaphora_declarations is provider
    future = provider_for(second, first, sentence_slots[0], 2)
    future_connector = LanguageGenerationConnector(
        registry,
        runtime_policy,
        surface_protocol,
        discourse_declarations=discourse,
        anaphora_declarations=future,
    )
    with pytest.raises(ValueError, match="antecedent 不得位于未来"):
        future_connector.structure_planner().plan(selection)


def _connector(
        promotion,
        selection,
        role,
        *,
        mutate_role=None,
        duplicate_template: bool = False,
        ):
    """构造 predicate/Role 映射模板和逐槽 surface 策略。"""
    sources = tuple(
        minimal_instruction_identity((_BASE + 1, ordinal))
        for ordinal in range(1, 5)
    )
    role_ordinal = minimal_instruction_identity((_BASE + 1, 5))
    protocol = LanguageConnectorValueProtocol(
        *sources,
        (LanguageConnectorOrdinalDefinition(role_ordinal, 0),),
    )
    slots = promotion.slots
    surface_protocol = _surface_protocol(_BASE + 2)
    bindings = (
        LanguageConnectorSlotBinding(
            structure_concept_identity((_BASE + 18, 1)),
            slots[0].slot,
            protocol.proposition_source,
        ),
        LanguageConnectorSlotBinding(
            structure_concept_identity((_BASE + 18, 2)),
            slots[1].slot,
            protocol.predicate_source,
        ),
        LanguageConnectorSlotBinding(
            structure_concept_identity((_BASE + 18, 3)),
            slots[2].slot,
            protocol.role_filler_source,
            role=role if mutate_role is None else mutate_role,
            ordinal=role_ordinal,
        ),
    )
    directives = tuple(
        LanguageConnectorSurfaceDirective(
            structure_concept_identity((_BASE + 19, index + 1)),
            slot.slot,
            (surface_protocol.silent_action
             if index == 0 else surface_protocol.emit_action),
            minimal_instruction_identity((_BASE + 3, index + 1)),
            structure_concept_identity((_BASE + 20, index + 1)),
            (),
        )
        for index, slot in enumerate(slots)
    )
    surface_runtime = tuple(
        LanguageConnectorSurfaceRuntimePolicy(
            slot.slot,
            (_BASE + 4, index + 1),
            (None if index == 0 else AliasRouteSearchBudget(30, 30, 30)),
            (_BASE + 5, index + 1),
        )
        for index, slot in enumerate(slots)
    )
    candidate = selection.request.candidates[0]
    template = LanguageGenerationConnectorTemplate(
        structure_concept_identity((_BASE + 6, 1)),
        selection.request.goal.target_branch,
        candidate.proposition.structure,
        candidate.proposition.predicate,
        structure_concept_identity((_BASE + 7, 1)),
        slots[0].structure,
        slots,
        bindings,
        structure_concept_identity((_BASE + 21, 1)),
        (promotion.constraint.constraint,),
        structure_concept_identity((_BASE + 21, 2)),
        (),
        minimal_instruction_identity((_BASE + 8, 1)),
        minimal_instruction_identity((_BASE + 8, 2)),
        directives,
    )
    templates = (template,)
    if duplicate_template:
        templates = (
            template,
            replace(
                template,
                connector=structure_concept_identity((_BASE + 6, 2)),
            ),
        )
    registry = LanguageGenerationConnectorRegistry(
        protocol,
        templates,
    )
    runtime_policy = LanguageGenerationConnectorRuntimePolicy(
        (_BASE + 9, 1),
        StructureOrderSearchBudget(30),
        tuple(
            LanguageConnectorTemplateRuntimePolicy(
                item.connector,
                surface_runtime,
            )
            for item in templates
        ),
    )
    return LanguageGenerationConnector(
        registry,
        runtime_policy,
        surface_protocol,
    )


def _execution_planner(graphs, promotion):
    """把已晋升 S-07 constraint 装配为真实只读结构执行器。"""
    applies, skipped, unknown = _semantic_reasons()
    resolver = _SemanticsResolver({
        promotion.constraint.constraint: _ResolvedRule(
            promotion.slots[2].slot,
            promotion.slots[1].slot,
            True,
            False,
            0,
            0,
            None,
        ),
    }, applies_reason=applies, skipped_reason=skipped,
       unknown_reason=unknown)
    consumer = StructureOrderConsumer(
        graphs.lifecycle,
        resolver,
        _consumer_protocol(),
    )
    from pure_integer_ai.cognition.shared.generation_structure_execution import (
        GenerationStructureExecutionPlanner,
    )
    return GenerationStructureExecutionPlanner(graphs.lifecycle, consumer)


def test_connector_routes_predicate_and_role_through_active_s07_and_r01():
    """predicate 作为普通槽值，经 active 顺序和 alias fact 形成 Representation。"""
    order_backend = DictBackend()
    alias = None
    try:
        graphs = _graphs(order_backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=91, slot_count=3)
        promoter = OrderConstraintPromoter(
            engine,
            graphs.order_graph,
            graphs.lifecycle,
        )
        promotion = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, first=1, second=2, kind=91),
            event=91,
            instance=91,
        )
        role = promotion.slots[2].role
        filler = concept_identity((_BASE + 10, 1))
        selection = _selection_with_role(domain.language, role, filler)
        connector = _connector(promotion, selection, role)
        structure = connector.structure_planner().plan(selection)
        builder = connector.surface_request_builder(
            _execution_planner(graphs, promotion))
        request = builder.build(structure)

        predicate = selection.request.candidates[0].proposition.predicate
        family = (_BASE + 11, 1)
        predicate_representation = representation_identity(
            family, (0x56E0,))
        filler_representation = representation_identity(
            family, (0x679C,))
        alias = _alias_fixture(
            domain.language,
            (
                (predicate, predicate_representation),
                (filler, filler_representation),
            ),
        )
        before_alias = alias.runtime.state_key()
        before_legacy_edges = tuple(
            order_backend.select("edge", where=None))

        run = GenerationSurfaceRuntime(alias.runtime).plan(request)

        assert request.execution.complete is True
        assert request.execution.sentences[0].active_constraints
        assert run.complete is True
        assert set(run.plan.representations) == {
            predicate_representation,
            filler_representation,
        }
        assert len(run.plan.adoptions) == 2
        assert all(
            item.reference_budget is None and not item.reference_use_key
            for item in request.directives)
        assert alias.runtime.state_key() != before_alias
        assert tuple(order_backend.select("edge", where=None)) == (
            before_legacy_edges)
        assert not hasattr(structure.syntax, "token_seq")
        assert not hasattr(structure.syntax, "role_seq")
    finally:
        if alias is not None:
            alias.close()
        order_backend.close()


def test_connector_reference_policy_drives_g03_and_fails_closed():
    """生产 mapper 注入 reference 预算，并保留三类 G-03 分型失败。"""
    order_backend = DictBackend()
    try:
        graphs = _graphs(order_backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=97, slot_count=3)
        promotion = _active_plan(
            engine,
            OrderConstraintPromoter(
                engine, graphs.order_graph, graphs.lifecycle),
            domain,
            _pattern(domain, first=1, second=2, kind=97),
            event=97,
            instance=97,
        )
        role = promotion.slots[2].role
        filler = concept_identity((_BASE + 30, 1))
        selection = _selection_with_role(domain.language, role, filler)
        candidate = selection.request.candidates[0]
        antecedent = candidate.proposition.template
        origin = proposition_identity(
            candidate.source, (_BASE + 30, 6))
        base = _connector(promotion, selection, role)
        reference_slot = base.registry.templates[0].slots[0].slot
        template_policy = base.runtime_policy.templates[0]
        reference_surface_policy = next(
            item for item in template_policy.surface
            if item.slot == reference_slot)
        reference_budget = AliasRouteSearchBudget(30, 30, 30)
        with pytest.raises(ValueError, match="必须成对"):
            replace(
                reference_surface_policy,
                reference_budget=reference_budget,
            )
        with pytest.raises(ValueError, match="必须成对"):
            replace(
                reference_surface_policy,
                reference_use_key_suffix=(_BASE + 30, 4),
            )
        reference_surfaces = tuple(
            replace(
                item,
                surface_budget=reference_budget,
                reference_budget=reference_budget,
                reference_use_key_suffix=(_BASE + 30, 5),
            )
            if item.slot == reference_slot else item
            for item in template_policy.surface
        )
        reference_runtime = replace(
            base.runtime_policy,
            templates=(replace(
                template_policy,
                surface=reference_surfaces,
            ),),
        )
        template = base.registry.templates[0]
        proposition_source = base.registry.value_protocol.proposition_source
        constant_source = base.registry.value_protocol.constant_source
        reference_bindings = tuple(
            replace(
                item,
                source=constant_source,
                constant=origin,
            )
            if item.slot == reference_slot else replace(
                item,
                source=proposition_source,
            )
            if item.source == base.registry.value_protocol.predicate_source
            else item
            for item in template.bindings
        )
        reference_theory = tuple(
            replace(
                item,
                action=base.surface_protocol.emit_action,
            )
            if item.slot == reference_slot else item
            for item in template.surface
        )
        reference_registry = LanguageGenerationConnectorRegistry(
            base.registry.value_protocol,
            (replace(
                template,
                bindings=reference_bindings,
                surface=reference_theory,
            ),),
        )
        connector = LanguageGenerationConnector(
            reference_registry,
            reference_runtime,
            base.surface_protocol,
        )
        base_structure = connector.structure_planner().plan(selection)
        sentence = base_structure.syntax.sentences[0]
        requirement = AnaphoraRequirement(
            sentence.sentence,
            reference_slot,
            selection.selected_candidate_keys[0],
            minimal_instruction_identity((_BASE + 30, 2)),
            (_BASE + 30, 3),
            sentence.instance,
        )
        structure = replace(
            base_structure,
            syntax=replace(
                base_structure.syntax,
                anaphora=(requirement,),
            ),
        )
        execution_planner = _execution_planner(graphs, promotion)
        execution_mapper = LanguageConnectorExecutionRequestMapper(
            reference_runtime.order_budget)
        execution = execution_planner.execute(
            execution_mapper.build(structure))
        base_execution = execution_planner.execute(
            execution_mapper.build(base_structure))
        mapper = LanguageConnectorSurfaceDirectiveMapper(
            connector.registry,
            connector.runtime_policy,
            connector.surface_protocol,
        )
        directives = mapper.plan(
            structure, execution, domain.language)
        request = GenerationSurfaceRequest(
            connector.surface_protocol,
            structure,
            execution,
            domain.language,
            directives,
        )
        reference_directive = next(
            item for item in directives
            if item.slot == reference_slot)
        assert reference_directive.reference_budget == reference_budget
        assert reference_directive.reference_use_key
        assert reference_directive.reference_use_key != (
            reference_directive.surface_use_key)

        surface_only = tuple(
            replace(
                item,
                reference_budget=None,
                reference_use_key_suffix=(),
            )
            if item.slot == reference_slot else item
            for item in reference_surfaces
        )
        surface_only_runtime = replace(
            reference_runtime,
            templates=(replace(
                template_policy,
                surface=surface_only,
            ),),
        )
        missing_policy_mapper = LanguageConnectorSurfaceDirectiveMapper(
            reference_registry,
            surface_only_runtime,
            base.surface_protocol,
        )
        with pytest.raises(ValueError, match="anaphora 与 reference"):
            missing_policy_mapper.plan(
                structure, execution, domain.language)
        with pytest.raises(ValueError, match="anaphora 与 reference"):
            mapper.plan(
                base_structure, base_execution, domain.language)

        wrong = proposition_identity(
            candidate.source, (_BASE + 30, 8))
        family = (_BASE + 30, 7)
        realizations = (
            (origin, representation_identity(family, (0x547D,))),
            (antecedent, representation_identity(family, (0x56E0,))),
            (filler, representation_identity(family, (0x679C,))),
        )
        cases = (
            (((origin, antecedent),), None),
            ((), request.protocol.reference_missing_reason),
            (((origin, antecedent), (origin, wrong)),
             request.protocol.reference_ambiguous_reason),
            (((origin, wrong),), request.protocol.reference_mismatch_reason),
        )
        for references, reason in cases:
            alias = _alias_fixture(
                domain.language, realizations, references)
            try:
                before = alias.runtime.state_key()
                run = GenerationSurfaceRuntime(alias.runtime).plan(request)
                if reason is None:
                    assert run.complete
                    assert len(run.plan.adoptions) == 4
                    assert alias.runtime.state_key() != before
                else:
                    assert not run.complete
                    assert run.preview.reason == reason
                    assert alias.runtime.state_key() == before
            finally:
                alias.close()
    finally:
        order_backend.close()


def test_connector_theory_identity_excludes_run_local_policy():
    """预算、trace 和 Use 键变化不得污染 connector 语言理论身份。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=95, slot_count=3)
        promotion = _active_plan(
            engine,
            OrderConstraintPromoter(
                engine, graphs.order_graph, graphs.lifecycle),
            domain,
            _pattern(domain, first=1, second=2, kind=95),
            event=95,
            instance=95,
        )
        role = promotion.slots[2].role
        selection = _selection_with_role(
            domain.language,
            role,
            concept_identity((_BASE + 15, 1)),
        )
        connector = _connector(promotion, selection, role)
        theory_key = connector.registry.stable_key()
        template_policy = connector.runtime_policy.templates[0]
        emitted = template_policy.surface[1]
        changed_surface = tuple(
            replace(
                item,
                trace=(_BASE + 16, 1),
                surface_budget=AliasRouteSearchBudget(31, 31, 31),
                use_key_suffix=(_BASE + 16, 2),
            )
            if item.slot == emitted.slot else item
            for item in template_policy.surface
        )
        changed_policy = LanguageGenerationConnectorRuntimePolicy(
            (_BASE + 16, 3),
            StructureOrderSearchBudget(31),
            (replace(template_policy, surface=changed_surface),),
        )
        changed = LanguageGenerationConnector(
            connector.registry,
            changed_policy,
            connector.surface_protocol,
        )

        assert connector.registry.stable_key() == theory_key
        assert changed.registry.stable_key() == theory_key
        assert changed.runtime_policy.stable_key() != (
            connector.runtime_policy.stable_key())
        assert changed.stable_key() != connector.stable_key()
        assert not hasattr(connector.registry.templates[0].surface[0], "trace")
        assert not hasattr(
            connector.registry.templates[0].surface[0], "surface_budget")
    finally:
        backend.close()


def test_connector_requires_bidirectional_runtime_policy_coverage():
    """理论模板和逐槽运行策略缺任一方向覆盖时必须在装配期失败。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=96, slot_count=3)
        promotion = _active_plan(
            engine,
            OrderConstraintPromoter(
                engine, graphs.order_graph, graphs.lifecycle),
            domain,
            _pattern(domain, first=1, second=2, kind=96),
            event=96,
            instance=96,
        )
        role = promotion.slots[2].role
        selection = _selection_with_role(
            domain.language,
            role,
            concept_identity((_BASE + 17, 1)),
        )
        connector = _connector(promotion, selection, role)
        template_policy = connector.runtime_policy.templates[0]
        incomplete = replace(
            connector.runtime_policy,
            templates=(replace(
                template_policy,
                surface=template_policy.surface[:-1],
            ),),
        )

        with pytest.raises(ValueError, match="slot 与运行策略未双向覆盖"):
            LanguageGenerationConnector(
                connector.registry,
                incomplete,
                connector.surface_protocol,
            )
    finally:
        backend.close()


def test_connector_rejects_ambiguous_template_without_private_sorting():
    """同一 predicate/structure/branch 有两个模板时不得取稳定序首项。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=92, slot_count=3)
        promotion = _active_plan(
            engine,
            OrderConstraintPromoter(
                engine, graphs.order_graph, graphs.lifecycle),
            domain,
            _pattern(domain, first=1, second=2, kind=92),
            event=92,
            instance=92,
        )
        role = promotion.slots[2].role
        selection = _selection_with_role(
            domain.language,
            role,
            concept_identity((_BASE + 12, 1)),
        )
        connector = _connector(
            promotion,
            selection,
            role,
            duplicate_template=True,
        )

        with pytest.raises(
                LanguageGenerationConnectorError,
                match="歧义模板"):
            connector.structure_planner().plan(selection)
    finally:
        backend.close()


def test_connector_missing_role_fails_before_s07_or_r01():
    """模板要求不存在的 Role 时不得用其他 filler、位置或词面补齐。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=93, slot_count=3)
        promotion = _active_plan(
            engine,
            OrderConstraintPromoter(
                engine, graphs.order_graph, graphs.lifecycle),
            domain,
            _pattern(domain, first=1, second=2, kind=93),
            event=93,
            instance=93,
        )
        role = promotion.slots[2].role
        selection = _selection_with_role(
            domain.language,
            role,
            concept_identity((_BASE + 13, 1)),
        )
        connector = _connector(
            promotion,
            selection,
            role,
            mutate_role=promotion.slots[1].role,
        )
        before = backend.snapshot()

        with pytest.raises(
                LanguageGenerationConnectorError,
                match=r"Role\+ordinal 未唯一绑定"):
            connector.structure_planner().plan(selection)

        assert backend.snapshot() == before
    finally:
        backend.close()


def test_connector_rejects_ignored_ordinal_and_untyped_mapper_inputs():
    """非 Role ordinal 和非 typed mapper 输入必须显式失败，不得被执行静默忽略。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=94, slot_count=3)
        promotion = _active_plan(
            engine,
            OrderConstraintPromoter(
                engine, graphs.order_graph, graphs.lifecycle),
            domain,
            _pattern(domain, first=1, second=2, kind=94),
            event=94,
            instance=94,
        )
        role = promotion.slots[2].role
        selection = _selection_with_role(
            domain.language,
            role,
            concept_identity((_BASE + 14, 1)),
        )
        connector = _connector(promotion, selection, role)
        template = connector.registry.templates[0]
        predicate_source = connector.registry.value_protocol.predicate_source
        ordinal_instruction = (
            connector.registry.value_protocol.ordinals[0].instruction)
        bindings = tuple(
            replace(binding, ordinal=ordinal_instruction)
            if binding.source == predicate_source else binding
            for binding in template.bindings
        )

        with pytest.raises(ValueError, match="非 Role filler 槽位"):
            LanguageGenerationConnectorRegistry(
                connector.registry.value_protocol,
                (replace(template, bindings=bindings),),
            )

        execution_mapper = LanguageConnectorExecutionRequestMapper(
            StructureOrderSearchBudget(30))
        with pytest.raises(TypeError, match="GenerationStructurePlan"):
            execution_mapper.build(object())
        surface_mapper = LanguageConnectorSurfaceDirectiveMapper(
            connector.registry,
            connector.runtime_policy,
            connector.surface_protocol,
        )
        with pytest.raises(TypeError, match="GenerationStructurePlan"):
            surface_mapper.plan(object(), object(), domain.language)
    finally:
        backend.close()
