"""S-03 typed binding、捕获规避 substitution 和 STRUCT_BIND 适配测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_VARIABLE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    binder_identity,
    context_scope_identity,
    describe_variable,
    entity_identity,
    proposition_identity,
    role_identity,
    variable_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    BindingFrame,
    BoundProposition,
    ExactTypeCompatibilityResolver,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
    TypeCompatibilityResult,
    TypedBindingAssignment,
    TypedBindingError,
    TypedValue,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticBindingSpec,
    SemanticBuildPlan,
    SemanticBuilderProtocol,
    SemanticCandidateBuilder,
    SemanticFillerSpec,
    SemanticPropositionSpec,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.cognition.understanding.struct_bind_typed_adapter import (
    StructBindTypedAdapter,
    TypedStructBindEndpoint,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.storage.edge_types import EDGE_STRUCT_BIND
from pure_integer_ai.storage.node_store import TIER_PRIMARY


def _source(document_id: int = 1) -> SourceRef:
    """构造测试用来源全键，document_id 用于区分声明域。"""
    return SourceRef(
        9301, 9302, document_id, GLOBAL_OWNER_SCOPE, VersionBundle())


def _failures(seed: int = 9310) -> BindingFailureProtocol:
    """注入九个互异 MinimalInstruction，测试不依赖生产 reason 数值。"""
    reasons = tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 10)
    )
    return BindingFailureProtocol(*reasons)


def _definition(
        source: SourceRef, key: int,
        bindings: tuple[AtomicRoleBinding, ...],
        ) -> AtomicPropositionDefinition:
    """构造同来源、开放 Role 的 Proposition template。"""
    return AtomicPropositionDefinition(
        proposition_identity(source, (9320, key)),
        concept_identity((9321, key)),
        occurrence_identity(source, start=key, end=key + 1, ordinal=0),
        context_scope_identity(source, (9322, key)),
        bindings,
    )


def _template(
        definition: AtomicPropositionDefinition,
        introduced_binders: tuple[ObjectIdentity, ...] = (),
        structure: ObjectIdentity | None = None,
        ) -> ScopedPropositionTemplate:
    """为运行期测试注入结构身份，避免从 predicate 或位置猜逻辑形状。"""
    if structure is None:
        structure = structure_concept_identity(
            (9323, *definition.proposition.stable_key()))
    return ScopedPropositionTemplate(
        definition, structure, introduced_binders)


def _assignment(
        variable: ObjectIdentity, value: ObjectIdentity,
        value_type: ObjectIdentity, failures: BindingFailureProtocol,
        resolver=None,
        ) -> TypedBindingAssignment:
    """用注入 resolver 构造一个明确通过的测试赋值。"""
    return TypedBindingAssignment.create(
        variable,
        TypedValue(value, value_type),
        resolver=resolver or ExactTypeCompatibilityResolver(),
        failures=failures,
    )


def _binding(bound: BoundProposition, role: ObjectIdentity):
    """按完整 Role identity 读取唯一 bound filler，避免依赖 tuple 位置。"""
    matches = tuple(item for item in bound.bindings if item.role == role)
    assert len(matches) == 1
    return matches[0].filler


class _RejectResolver:
    """测试用显式不兼容 resolver，用于区分 refuted 与 unknown。"""

    def resolve(self, expected_type, actual_type):
        """对注入类型对返回明确 False，不解释具体类型意义。"""
        return TypeCompatibilityResult(expected_type, actual_type, False)


def test_describe_variable_roundtrips_full_identity_and_rejects_corruption():
    """Variable 必须恢复 Binder/local key/type 全键并拒绝嵌套损坏。"""
    source = _source()
    binder = binder_identity(source, (9330, 1))
    value_type = concept_identity((9330, 2))
    variable = variable_identity(binder, (9330, 3), value_type)

    descriptor = describe_variable(variable)
    assert descriptor.identity == variable
    assert descriptor.source == source
    assert descriptor.binder == binder
    assert descriptor.local_key == (9330, 3)
    assert descriptor.value_type == value_type

    components = list(variable.components)
    binder_length_index = 1 + len(source.stable_key())
    components[binder_length_index] += 1
    corrupt = ObjectIdentity(
        OBJECT_VARIABLE, tuple(components), variable.owner, variable.versions)
    with pytest.raises(ValueError):
        describe_variable(corrupt)


def test_same_local_key_under_different_binders_never_crosses_frames():
    """同 local key 的变量只按完整 Binder identity 查找。"""
    failures = _failures()
    source = _source()
    value_type = concept_identity((9340, 1))
    outer = binder_identity(source, (9340, 2))
    inner = binder_identity(source, (9340, 3))
    outer_var = variable_identity(outer, (1,), value_type)
    inner_var = variable_identity(inner, (1,), value_type)
    outer_value = entity_identity(source, (9340, 4))
    inner_value = entity_identity(source, (9340, 5))
    outer_frame = BindingFrame.create(
        outer,
        (_assignment(outer_var, outer_value, value_type, failures),),
        failures=failures,
    )
    inner_frame = BindingFrame.create(
        inner,
        (_assignment(inner_var, inner_value, value_type, failures),),
        failures=failures,
    )
    environment = BindingEnvironment().push(
        outer_frame, failures=failures).push(
            inner_frame, failures=failures)

    assert environment.resolve(
        outer_var, failures=failures).value.value == outer_value
    assert environment.resolve(
        inner_var, failures=failures).value.value == inner_value
    assert outer_var != inner_var


def test_type_unknown_and_explicit_rejection_have_distinct_reasons():
    """默认 unknown 与 resolver 明确拒绝必须分别 fail closed。"""
    failures = _failures()
    source = _source()
    binder = binder_identity(source, (9350, 1))
    expected = concept_identity((9350, 2))
    actual = concept_identity((9350, 3))
    variable = variable_identity(binder, (9350, 4), expected)
    value = TypedValue(entity_identity(source, (9350, 5)), actual)

    with pytest.raises(TypedBindingError) as unknown:
        TypedBindingAssignment.create(
            variable,
            value,
            resolver=ExactTypeCompatibilityResolver(),
            failures=failures,
        )
    assert unknown.value.failure.reason == failures.type_unknown
    assert unknown.value.failure.expected_type == expected
    assert unknown.value.failure.actual_type == actual

    with pytest.raises(TypedBindingError) as rejected:
        TypedBindingAssignment.create(
            variable,
            value,
            resolver=_RejectResolver(),
            failures=failures,
        )
    assert rejected.value.failure.reason == failures.type_rejected
    assert rejected.value.failure.reason != unknown.value.failure.reason


def test_frame_and_environment_report_structural_failure_reasons():
    """重复变量、Binder 错配、重复 frame 和未绑定均返回注入 reason。"""
    failures = _failures()
    source = _source()
    value_type = concept_identity((9360, 1))
    first_binder = binder_identity(source, (9360, 2))
    second_binder = binder_identity(source, (9360, 3))
    variable = variable_identity(first_binder, (9360, 4), value_type)
    missing = variable_identity(first_binder, (9360, 5), value_type)
    assignment = _assignment(
        variable, entity_identity(source, (9360, 6)), value_type, failures)

    with pytest.raises(TypedBindingError) as duplicate:
        BindingFrame.create(
            first_binder, (assignment, assignment), failures=failures)
    assert duplicate.value.failure.reason == failures.duplicate_variable

    with pytest.raises(TypedBindingError) as mismatch:
        BindingFrame.create(
            second_binder, (assignment,), failures=failures)
    assert mismatch.value.failure.reason == failures.binder_mismatch

    frame = BindingFrame.create(
        first_binder, (assignment,), failures=failures)
    environment = BindingEnvironment().push(frame, failures=failures)
    with pytest.raises(TypedBindingError) as conflict:
        environment.push(frame, failures=failures)
    assert conflict.value.failure.reason == failures.scope_conflict
    assert environment.lookup(missing) is None
    with pytest.raises(TypedBindingError) as unbound:
        environment.resolve(missing, failures=failures)
    assert unbound.value.failure.reason == failures.unbound_variable


def test_frame_and_environment_stable_keys_are_order_deterministic():
    """赋值输入乱序不改变 frame 或 environment 的运行期稳定键。"""
    failures = _failures()
    source = _source()
    value_type = concept_identity((9370, 1))
    binder = binder_identity(source, (9370, 2))
    first = variable_identity(binder, (9370, 3), value_type)
    second = variable_identity(binder, (9370, 4), value_type)
    assignments = (
        _assignment(
            first, entity_identity(source, (9370, 5)),
            value_type, failures),
        _assignment(
            second, entity_identity(source, (9370, 6)),
            value_type, failures),
    )
    direct = BindingFrame.create(binder, assignments, failures=failures)
    reverse = BindingFrame.create(
        binder, tuple(reversed(assignments)), failures=failures)
    assert direct == reverse
    assert direct.stable_key() == reverse.stable_key()
    assert BindingEnvironment((direct,)).stable_key() == (
        BindingEnvironment((reverse,)).stable_key())


def test_s02_variable_filler_substitutes_without_storage_writes():
    """S-02 产出的真实 Variable filler 可绑定，S-03 前后后端快照不变。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        occurrences = OccurrenceIndex(
            context.graph_ontology,
            context.scoped_identity_store,
            OccurrenceProtocol((9380, 1), (9380, 2)),
        )
        spans = SpanIndex(
            context.graph_ontology,
            context.scoped_identity_store,
            SpanProtocol((9381, 1), (9381, 2), (9381, 3), (9381, 4)),
            occurrences,
        )
        source = _source()
        scope = document_scope(source)
        occurrence = occurrences.record(
            source=source,
            raw_text="甲",
            scope=scope,
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
        ).occurrence
        span = spans.ensure_ref(
            source=source, raw_text="甲", scope=scope, members=((0, 1),))
        binder = binder_identity(source, (9382, 1))
        value_type = concept_identity((9382, 2))
        variable = variable_identity(binder, (9382, 3), value_type)
        upstream = HypothesisKey(
            (9383, 1), (9383, 2), (9383, 3), scope, source)
        plan = SemanticBuildPlan(
            upstream,
            (9384, 1),
            (),
            (SemanticPropositionSpec(
                (9384, 2),
                (9384, 3),
                concept_identity((9384, 4)),
                structure_concept_identity((9384, 5)),
                (SemanticBindingSpec(
                    role_identity((9384, 6)),
                    SemanticFillerSpec(external=variable),
                ),),
                source_anchor=occurrence,
            ),),
        )
        result = SemanticCandidateBuilder(
            spans,
            SemanticBuilderProtocol(
                minimal_instruction_identity((9385, 1)), (9385, 2)),
            occurrences,
        ).compile(span, plan)
        definition = result.propositions[0].definition
        failures = _failures()
        value = entity_identity(source, (9386, 1))
        frame = BindingFrame.create(
            binder,
            (_assignment(variable, value, value_type, failures),),
            failures=failures,
        )
        before = backend.snapshot()
        bound = PropositionSubstituter(SubstitutionProtocol(
            minimal_instruction_identity((9386, 2)), failures,
        )).substitute(
            definition.proposition,
                PropositionTemplateGraph((_template(
                definition, (binder,), result.propositions[0].spec.structure),)),
            BindingEnvironment((frame,)),
        )
        assert bound.bindings[0].filler == value
        assert bound.applied_variables == (variable,)
        assert backend.snapshot() == before
    finally:
        backend.close()


def test_nested_binders_substitute_by_identity_and_avoid_name_capture():
    """内外 Binder 的同 local key 不串，替换后的外层 Variable 不被内层捕获。"""
    failures = _failures()
    source = _source()
    value_type = concept_identity((9390, 1))
    outer_binder = binder_identity(source, (9390, 2))
    inner_binder = binder_identity(source, (9390, 3))
    outer_var = variable_identity(outer_binder, (1,), value_type)
    replacement_var = variable_identity(outer_binder, (2,), value_type)
    inner_var = variable_identity(inner_binder, (1,), value_type)
    outer_role = role_identity((9390, 4))
    inner_role = role_identity((9390, 5))
    free_outer_role = role_identity((9390, 6))
    nested_role = role_identity((9390, 7))
    inner = _definition(source, 2, (
        AtomicRoleBinding(inner_role, inner_var),
        AtomicRoleBinding(free_outer_role, outer_var),
    ))
    outer = _definition(source, 1, (
        AtomicRoleBinding(outer_role, outer_var),
        AtomicRoleBinding(nested_role, inner.proposition),
    ))
    inner_value = entity_identity(source, (9390, 8))
    outer_frame = BindingFrame.create(
        outer_binder,
        (_assignment(
            outer_var, replacement_var, value_type, failures),),
        failures=failures,
    )
    inner_frame = BindingFrame.create(
        inner_binder,
        (_assignment(inner_var, inner_value, value_type, failures),),
        failures=failures,
    )
    graph = PropositionTemplateGraph((
        _template(inner, (inner_binder,)),
        _template(outer, (outer_binder,)),
    ))
    bound = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((9390, 9)), failures,
    )).substitute(
        outer.proposition,
        graph,
        BindingEnvironment((outer_frame, inner_frame)),
    )

    assert _binding(bound, outer_role) == replacement_var
    nested = _binding(bound, nested_role)
    assert isinstance(nested, BoundProposition)
    assert _binding(nested, inner_role) == inner_value
    assert _binding(nested, free_outer_role) == replacement_var
    assert describe_variable(replacement_var).binder == outer_binder
    assert nested.introduced_binders == (inner_binder,)


def test_lexical_scope_rejects_sibling_leak_even_with_shared_template_memo():
    """共享子命题经合法路径缓存后，兄弟越域路径仍必须重新校验并失败。"""
    failures = _failures()
    source = _source()
    binder = binder_identity(source, (9400, 1))
    variable = variable_identity(
        binder, (9400, 2), concept_identity((9400, 3)))
    shared = _definition(source, 4, (
        AtomicRoleBinding(role_identity((9400, 4)), variable),))
    declaring_parent = _definition(source, 2, (
        AtomicRoleBinding(
            role_identity((9400, 5)), shared.proposition),))
    sibling_parent = _definition(source, 3, (
        AtomicRoleBinding(
            role_identity((9400, 6)), shared.proposition),))
    root = _definition(source, 1, (
        AtomicRoleBinding(
            role_identity((9400, 7)), declaring_parent.proposition),
        AtomicRoleBinding(
            role_identity((9400, 8)), sibling_parent.proposition),
    ))
    graph = PropositionTemplateGraph((
        _template(root),
        _template(declaring_parent, (binder,)),
        _template(sibling_parent),
        _template(shared),
    ))
    with pytest.raises(TypedBindingError) as error:
        PropositionSubstituter(SubstitutionProtocol(
            minimal_instruction_identity((9400, 9)), failures,
        )).substitute(root.proposition, graph, BindingEnvironment())
    assert error.value.failure.reason == failures.scope_conflict
    assert error.value.failure.variable == variable
    assert error.value.failure.binder == binder


def test_template_graph_rejects_duplicate_binder_introduction():
    """同一 Binder 由多个 template 引入会造成词法所有者歧义，必须拒绝。"""
    source = _source()
    binder = binder_identity(source, (9410, 1))
    first = _definition(source, 1, ())
    second = _definition(source, 2, ())
    with pytest.raises(ValueError, match="多个"):
        PropositionTemplateGraph((
            _template(first, (binder,)),
            _template(second, (binder,)),
        ))


def test_nested_input_order_is_deterministic_and_unknown_proposition_is_opaque():
    """template 输入乱序不改输出，未登记嵌套 Proposition 保持权威 opaque identity。"""
    failures = _failures()
    source = _source()
    opaque = proposition_identity(source, (9420, 1))
    child = _definition(source, 2, (
        AtomicRoleBinding(role_identity((9420, 2)), opaque),))
    root = _definition(source, 1, (
        AtomicRoleBinding(role_identity((9420, 3)), child.proposition),))
    protocol = SubstitutionProtocol(
        minimal_instruction_identity((9420, 4)), failures)
    direct = PropositionSubstituter(protocol).substitute(
        root.proposition,
        PropositionTemplateGraph((
            _template(root),
            _template(child),
        )),
        BindingEnvironment(),
    )
    reverse = PropositionSubstituter(protocol).substitute(
        root.proposition,
        PropositionTemplateGraph((
            _template(child),
            _template(root),
        )),
        BindingEnvironment(),
    )
    assert direct == reverse
    assert direct.stable_key() == reverse.stable_key()
    nested = direct.bindings[0].filler
    assert isinstance(nested, BoundProposition)
    assert nested.bindings[0].filler == opaque


def test_proposition_cycle_and_missing_root_use_structured_reasons():
    """模板环和未登记 root 均按注入协议失败，不依赖递归异常文字。"""
    failures = _failures()
    source = _source()
    first_id = proposition_identity(source, (9430, 1))
    second_id = proposition_identity(source, (9430, 2))
    first = replace(
        _definition(source, 1, (
            AtomicRoleBinding(role_identity((9430, 3)), second_id),)),
        proposition=first_id,
    )
    second = replace(
        _definition(source, 2, (
            AtomicRoleBinding(role_identity((9430, 4)), first_id),)),
        proposition=second_id,
    )
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((9430, 5)), failures))
    graph = PropositionTemplateGraph((
        _template(first),
        _template(second),
    ))
    with pytest.raises(TypedBindingError) as cycle:
        substituter.substitute(first_id, graph, BindingEnvironment())
    assert cycle.value.failure.reason == failures.proposition_cycle

    missing = proposition_identity(source, (9430, 6))
    with pytest.raises(TypedBindingError) as absent:
        substituter.substitute(missing, graph, BindingEnvironment())
    assert absent.value.failure.reason == failures.template_missing


def _add_struct_bind(context, source_slot, target_slot, *, order_index: int):
    """向测试后端追加一条合法 legacy STRUCT_BIND 边。"""
    context.edge_store.add(
        space_id_from=source_slot[0],
        local_id_from=source_slot[1],
        space_id_to=target_slot[0],
        local_id_to=target_slot[1],
        edge_type=EDGE_STRUCT_BIND,
        strength=1,
        source=9440,
        tier=TIER_PRIMARY,
        epistemic_origin=EPI_STRUCTURED,
        order_index=order_index,
    )


def test_struct_bind_adapter_requires_explicit_mapping_and_preserves_all_edges():
    """显式 endpoint 映射恢复全部竞争边，读取过程不修改后端。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        source_slot = (context.space_id, 100)
        target_a = (context.space_id, 200)
        target_b = (context.space_id, 300)
        _add_struct_bind(context, source_slot, target_b, order_index=9)
        _add_struct_bind(context, source_slot, target_a, order_index=2)
        source = _source()
        value_type = concept_identity((9441, 1))
        source_var = variable_identity(
            binder_identity(source, (9441, 2)), (9441, 3), value_type)
        target_var_a = variable_identity(
            binder_identity(source, (9441, 4)), (9441, 5), value_type)
        target_var_b = variable_identity(
            binder_identity(source, (9441, 6)), (9441, 7), value_type)
        endpoints = (
            TypedStructBindEndpoint(target_b, target_var_b),
            TypedStructBindEndpoint(source_slot, source_var),
            TypedStructBindEndpoint(target_a, target_var_a),
        )
        before = backend.snapshot()
        result = StructBindTypedAdapter(
            ExactTypeCompatibilityResolver(), _failures()).read_from(
                context.edge_store, source_slot, endpoints)

        assert len(result.correspondences) == 2
        assert result.failures == ()
        assert {item.target.slot_ref for item in result.correspondences} == {
            target_a, target_b}
        assert {item.target.variable for item in result.correspondences} == {
            target_var_a, target_var_b}
        assert {item.legacy_metadata.order_index
                for item in result.correspondences} == {2, 9}
        assert backend.snapshot() == before
    finally:
        backend.close()


def test_struct_bind_missing_mapping_and_type_unknown_fail_per_edge():
    """缺显式映射和类型 unknown 分别保留对应旧边及注入 reason。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        source_slot = (context.space_id, 101)
        missing_target = (context.space_id, 201)
        typed_target = (context.space_id, 301)
        _add_struct_bind(context, source_slot, missing_target, order_index=1)
        _add_struct_bind(context, source_slot, typed_target, order_index=2)
        source = _source()
        source_type = concept_identity((9450, 1))
        target_type = concept_identity((9450, 2))
        source_var = variable_identity(
            binder_identity(source, (9450, 3)), (9450, 4), source_type)
        target_var = variable_identity(
            binder_identity(source, (9450, 5)), (9450, 6), target_type)
        failures = _failures()
        result = StructBindTypedAdapter(
            ExactTypeCompatibilityResolver(), failures).read_from(
                context.edge_store,
                source_slot,
                (
                    TypedStructBindEndpoint(source_slot, source_var),
                    TypedStructBindEndpoint(typed_target, target_var),
                ),
            )

        assert result.correspondences == ()
        assert len(result.failures) == 2
        assert {item.failure.reason for item in result.failures} == {
            failures.legacy_mapping_missing,
            failures.type_unknown,
        }
        unknown = next(
            item for item in result.failures
            if item.failure.reason == failures.type_unknown)
        assert unknown.failure.expected_type == target_type
        assert unknown.failure.actual_type == source_type
    finally:
        backend.close()


def test_struct_bind_order_index_never_selects_or_changes_variable_mapping():
    """改变旧 order_index 只改变 trace，不改变显式 source/target Variable。"""
    mapped = []
    metadata = []
    for order_index in (0, 999):
        backend = DictBackend()
        try:
            context = make_train_context(backend)
            source_slot = (context.space_id, 102)
            target_slot = (context.space_id, 202)
            _add_struct_bind(
                context, source_slot, target_slot,
                order_index=order_index)
            source = _source()
            value_type = concept_identity((9460, 1))
            source_var = variable_identity(
                binder_identity(source, (9460, 2)), (9460, 3), value_type)
            target_var = variable_identity(
                binder_identity(source, (9460, 4)), (9460, 5), value_type)
            result = StructBindTypedAdapter(
                ExactTypeCompatibilityResolver(), _failures()).read_from(
                    context.edge_store,
                    source_slot,
                    (
                        TypedStructBindEndpoint(source_slot, source_var),
                        TypedStructBindEndpoint(target_slot, target_var),
                    ),
                )
            assert len(result.correspondences) == 1
            candidate = result.correspondences[0]
            mapped.append((candidate.source.variable, candidate.target.variable))
            metadata.append(candidate.legacy_metadata.order_index)
        finally:
            backend.close()
    assert mapped[0] == mapped[1]
    assert metadata == [0, 999]
