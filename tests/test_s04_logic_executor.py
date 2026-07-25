"""S-04 注入式复合逻辑执行、有限域量化和 trace 测试。"""
from __future__ import annotations

from itertools import product

import pytest

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    ConjunctionOperator,
    ConditionOperator,
    DisjunctionOperator,
    ExistentialOperator,
    FiniteQuantifierDomain,
    LogicAtomEvidence,
    LogicEvaluation,
    LogicEvidenceState,
    LogicExecutor,
    LogicFailureProtocol,
    LogicOperatorDefinition,
    LogicOperatorRegistry,
    ModalOperator,
    ModalResolution,
    NegationOperator,
    OperatorSlot,
    QuantifierDefinition,
    STATE_CONFLICTED,
    STATE_PROVISIONAL,
    STATE_REFUTED,
    STATE_UNKNOWN,
    UniversalOperator,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    binder_identity,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
    set_expr_identity,
    variable_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    ExactTypeCompatibilityResolver,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
    TypedValue,
)
from pure_integer_ai.storage.backend import DictBackend


_T = LogicEvidenceState(True, False)
_F = LogicEvidenceState(False, True)
_U = LogicEvidenceState(False, False)
_B = LogicEvidenceState(True, True)
_STATE_BY_STATUS = {
    STATE_PROVISIONAL: _T,
    STATE_REFUTED: _F,
    STATE_UNKNOWN: _U,
    STATE_CONFLICTED: _B,
}


def _source(document_id: int = 1) -> SourceRef:
    """构造共享 owner/version 且 document 可区分的测试来源。"""
    return SourceRef(
        9601, 9602, document_id, GLOBAL_OWNER_SCOPE, VersionBundle())


def _binding_failures(seed: int = 9610) -> BindingFailureProtocol:
    """注入 S-03 所需的九个互异绑定失败指令。"""
    return BindingFailureProtocol(*tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 10)
    ))


def _logic_failures(seed: int = 9620) -> LogicFailureProtocol:
    """注入 S-04 所需的九个互异逻辑失败指令。"""
    return LogicFailureProtocol(*tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 10)
    ))


def _definition(
        source: SourceRef,
        key: int,
        bindings: tuple[AtomicRoleBinding, ...] = (),
        ) -> AtomicPropositionDefinition:
    """构造不含真值、可作为原子或复合 template 的命题定义。"""
    return AtomicPropositionDefinition(
        proposition_identity(source, (9630, key)),
        concept_identity((9631, key)),
        occurrence_identity(source, start=key, end=key + 1, ordinal=0),
        context_scope_identity(source, (9632, key)),
        bindings,
    )


def _template(
        definition: AtomicPropositionDefinition,
        structure: ObjectIdentity,
        introduced_binders: tuple[ObjectIdentity, ...] = (),
        ) -> ScopedPropositionTemplate:
    """把命题和图中一等结构、词法 Binder 组合为 substitution template。"""
    return ScopedPropositionTemplate(
        definition, structure, introduced_binders)


def _bound(
        root: AtomicPropositionDefinition,
        templates: tuple[ScopedPropositionTemplate, ...],
        binding_failures: BindingFailureProtocol,
        environment: BindingEnvironment = BindingEnvironment(),
        inherited_binders: tuple[ObjectIdentity, ...] = (),
        ):
    """经真实 S-03 substitution 生成 S-04 只读运行期视图。"""
    graph = PropositionTemplateGraph(templates)
    protocol = SubstitutionProtocol(
        minimal_instruction_identity((9633, 1)), binding_failures)
    return (
        PropositionSubstituter(protocol).substitute(
            root.proposition,
            graph,
            environment,
            inherited_binders=inherited_binders,
        ),
        graph,
        protocol,
    )


class _InjectedAtomResolver:
    """按调用方函数读取 bound view，测试量化值和 scope 都不写死在执行器。"""

    def __init__(self, resolver) -> None:
        self._resolver = resolver
        self.calls: list[tuple[ObjectIdentity, tuple[int, ...]]] = []

    def resolve(self, proposition, *, source, scope):
        """把测试函数给出的状态包装为 source-bearing Evidence。"""
        self.calls.append((proposition.template, scope.stable_key()))
        resolved = self._resolver(proposition, scope)
        if resolved is None:
            return None
        state, evidence_id = resolved
        hypothesis = HypothesisKey(
            (9634, 1),
            (9634, evidence_id),
            (9634, 2),
            scope,
            source,
        )
        return LogicAtomEvidence(
            proposition.template,
            state,
            source,
            scope,
            hypothesis,
            (evidence_id,) if state.support else (),
            (evidence_id + 1000,) if state.refute else (),
            (evidence_id + 2000,) if state.status == STATE_UNKNOWN else (),
        )


class _InjectedQuantifierResolver:
    """按完整 Proposition template 注入本次运行的量化定义。"""

    def __init__(self, definitions) -> None:
        self._definitions = definitions

    def resolve(self, operator, proposition, context):
        """不按 operator 名称或位置猜 Binder/domain。"""
        resolved = self._definitions.get(proposition.template)
        if resolved is not None:
            assert operator.structure == proposition.structure
        return resolved


def _executor(
        definitions: tuple[LogicOperatorDefinition, ...],
        resolver,
        binding_failures: BindingFailureProtocol,
        logic_failures: LogicFailureProtocol,
        protocol: SubstitutionProtocol,
        ) -> LogicExecutor:
    """用全注入 registry、resolver 和 reason 协议构造执行器。"""
    return LogicExecutor(
        LogicOperatorRegistry(definitions),
        resolver,
        logic_failures,
        protocol,
        ExactTypeCompatibilityResolver(),
        binding_failures,
    )


def _binary_case(handler, left_state, right_state):
    """构造两个原子前提和一个开放二元结构并返回根求值。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    atom_structure = structure_concept_identity((9640, 1))
    operator_structure = structure_concept_identity((9640, 2))
    left_role = role_identity((9640, 3))
    right_role = role_identity((9640, 4))
    left = _definition(source, 1)
    right = _definition(source, 2)
    root = _definition(source, 3, (
        AtomicRoleBinding(left_role, left.proposition),
        AtomicRoleBinding(right_role, right.proposition),
    ))
    templates = (
        _template(left, atom_structure),
        _template(right, atom_structure),
        _template(root, operator_structure),
    )
    bound, graph, protocol = _bound(root, templates, failures)
    states = {
        left.proposition: (left_state, 1),
        right.proposition: (right_state, 2),
    }
    resolver = _InjectedAtomResolver(
        lambda proposition, _scope: states.get(proposition.template))
    definition = LogicOperatorDefinition(
        operator_structure,
        minimal_instruction_identity((9640, 5)),
        (OperatorSlot(left_role), OperatorSlot(right_role)),
        handler,
    )
    evaluation = _executor(
        (definition,), resolver, failures, logic_failures, protocol,
    ).evaluate(
        bound,
        source=source,
        scope=scope,
        graph=graph,
        environment=BindingEnvironment(),
    )
    return evaluation


def _quantifier_case(
        operator_type,
        states_by_value: dict[ObjectIdentity, LogicEvidenceState],
        *,
        closed: bool,
        actual_type: ObjectIdentity | None = None,
        ):
    """构造单变量有限域量化并返回结果、协议和后端无关对象。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    value_type = concept_identity((9650, 1))
    binder = binder_identity(source, (9650, 2))
    variable = variable_identity(binder, (9650, 3), value_type)
    body_role = role_identity((9650, 4))
    value_role = role_identity((9650, 5))
    atom_structure = structure_concept_identity((9650, 6))
    quantifier_structure = structure_concept_identity((9650, 7))
    body = _definition(source, 1, (
        AtomicRoleBinding(value_role, variable),))
    root = _definition(source, 2, (
        AtomicRoleBinding(body_role, body.proposition),))
    templates = (
        _template(body, atom_structure),
        _template(root, quantifier_structure, (binder,)),
    )
    bound, graph, protocol = _bound(root, templates, failures)
    values = tuple(
        TypedValue(value, actual_type or value_type)
        for value in states_by_value
    )
    domain = FiniteQuantifierDomain(
        set_expr_identity(source, (9650, 8)),
        values,
        closed,
        (concept_identity((9650, 9)),) if closed else (),
    )
    quantifier = QuantifierDefinition(
        binder, variable, OperatorSlot(body_role), domain)
    handler = operator_type()
    definition = LogicOperatorDefinition(
        quantifier_structure,
        minimal_instruction_identity((9650, 10)),
        (OperatorSlot(body_role),),
        handler,
    )

    def resolve(proposition, _scope):
        filler = next(
            item.filler for item in proposition.bindings
            if item.role == value_role)
        state = states_by_value.get(filler)
        if state is None:
            return None
        ordinal = tuple(states_by_value).index(filler)
        return state, 10 + ordinal

    resolver = _InjectedAtomResolver(resolve)
    evaluation = _executor(
        (definition,), resolver, failures, logic_failures, protocol,
    ).evaluate(
        bound,
        source=source,
        scope=scope,
        graph=graph,
        environment=BindingEnvironment(),
        quantifier_resolver=_InjectedQuantifierResolver({
            root.proposition: quantifier,
        }),
    )
    return evaluation, logic_failures, resolver


def test_four_states_roundtrip_and_not_preserves_unknown_and_conflict():
    """四态编码可逆，NOT 只交换证据位而不二值化 unknown/conflicted。"""
    for status, state in _STATE_BY_STATUS.items():
        assert LogicEvidenceState.from_status(status) == state
        assert state.status == status
    assert _T.negate() == _F
    assert _F.negate() == _T
    assert _U.negate() == _U
    assert _B.negate() == _B


@pytest.mark.parametrize(
    ("handler", "table"),
    (
        (ConjunctionOperator(), (
            (STATE_PROVISIONAL, STATE_REFUTED, STATE_UNKNOWN, STATE_CONFLICTED),
            (STATE_REFUTED, STATE_REFUTED, STATE_REFUTED, STATE_REFUTED),
            (STATE_UNKNOWN, STATE_REFUTED, STATE_UNKNOWN, STATE_REFUTED),
            (STATE_CONFLICTED, STATE_REFUTED, STATE_REFUTED, STATE_CONFLICTED),
        )),
        (DisjunctionOperator(), (
            (STATE_PROVISIONAL, STATE_PROVISIONAL, STATE_PROVISIONAL, STATE_PROVISIONAL),
            (STATE_PROVISIONAL, STATE_REFUTED, STATE_UNKNOWN, STATE_CONFLICTED),
            (STATE_PROVISIONAL, STATE_UNKNOWN, STATE_UNKNOWN, STATE_PROVISIONAL),
            (STATE_PROVISIONAL, STATE_CONFLICTED, STATE_PROVISIONAL, STATE_CONFLICTED),
        )),
        (ConditionOperator(), (
            (STATE_PROVISIONAL, STATE_REFUTED, STATE_UNKNOWN, STATE_CONFLICTED),
            (STATE_PROVISIONAL, STATE_PROVISIONAL, STATE_PROVISIONAL, STATE_PROVISIONAL),
            (STATE_PROVISIONAL, STATE_UNKNOWN, STATE_UNKNOWN, STATE_PROVISIONAL),
            (STATE_PROVISIONAL, STATE_CONFLICTED, STATE_PROVISIONAL, STATE_CONFLICTED),
        )),
    ),
)
def test_binary_operator_truth_tables(handler, table):
    """AND、OR、CONDITION 的十六格四态表必须保持证据位语义。"""
    statuses = (
        STATE_PROVISIONAL,
        STATE_REFUTED,
        STATE_UNKNOWN,
        STATE_CONFLICTED,
    )
    for left_index, right_index in product(range(4), repeat=2):
        evaluation = _binary_case(
            handler,
            _STATE_BY_STATUS[statuses[left_index]],
            _STATE_BY_STATUS[statuses[right_index]],
        )
        assert evaluation.status == table[left_index][right_index]


def test_not_uses_structure_identity_and_preserves_trace_evidence():
    """NOT 只按注册 StructureConcept 派发，并在根步骤保留前提 Evidence。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    atom_structure = structure_concept_identity((9660, 1))
    not_structure = structure_concept_identity((9660, 2))
    child_role = role_identity((9660, 3))
    child = _definition(source, 1)
    root = _definition(source, 2, (
        AtomicRoleBinding(child_role, child.proposition),))
    templates = (
        _template(child, atom_structure),
        _template(root, not_structure),
    )
    bound, graph, protocol = _bound(root, templates, failures)
    resolver = _InjectedAtomResolver(
        lambda proposition, _scope: (_B, 17)
        if proposition.template == child.proposition else None)
    definition = LogicOperatorDefinition(
        not_structure,
        minimal_instruction_identity((9660, 4)),
        (OperatorSlot(child_role),),
        NegationOperator(),
    )
    evaluation = _executor(
        (definition,), resolver, failures, logic_failures, protocol,
    ).evaluate(
        bound, source=source, scope=scope, graph=graph,
        environment=BindingEnvironment())

    assert evaluation.state == _B
    assert evaluation.evidence_ids == (17, 1017)
    assert len(evaluation.hypotheses) == 1
    assert len(evaluation.derivation) == 1
    assert evaluation.derivation[0].operator == not_structure
    assert evaluation.derivation[0].premises == (child.proposition,)
    assert evaluation.derivation[0].evidence_ids == evaluation.evidence_ids
    assert evaluation.derivation[0].hypotheses == evaluation.hypotheses


def test_unregistered_structure_and_atom_without_evidence_remain_unknown():
    """未注册结构不会按 predicate 或名字猜 operator，无原子证据时保持 unknown。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    structure = structure_concept_identity((9670, 1))
    atom = _definition(source, 1)
    bound, graph, protocol = _bound(
        atom, (_template(atom, structure),), failures)
    resolver = _InjectedAtomResolver(lambda _proposition, _scope: None)
    evaluation = _executor(
        (), resolver, failures, logic_failures, protocol,
    ).evaluate(
        bound, source=source, scope=scope, graph=graph,
        environment=BindingEnvironment())

    assert evaluation.status == STATE_UNKNOWN
    assert evaluation.failures[0].reason == logic_failures.atom_unknown
    assert evaluation.derivation == ()


def test_atom_resolver_source_and_scope_drift_are_rejected():
    """resolver 不能把其他 source 或 scope 的 Evidence 偷渡到当前求值。"""
    source = _source()
    scope = document_scope(source)
    other_scope = query_scope(2, parent=scope)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    atom = _definition(source, 1)
    structure = structure_concept_identity((9680, 1))
    bound, graph, protocol = _bound(
        atom, (_template(atom, structure),), failures)

    class _DriftResolver:
        """固定返回另一 scope，验证执行器在边界拒绝来源漂移。"""

        def resolve(self, proposition, *, source, scope):
            """返回结构合法但归属错误的 Evidence。"""
            return LogicAtomEvidence(
                proposition.template,
                _T,
                source,
                other_scope,
                None,
                (1,),
            )

    executor = _executor(
        (), _DriftResolver(), failures, logic_failures, protocol)
    with pytest.raises(ValueError, match="source/scope"):
        executor.evaluate(
            bound, source=source, scope=scope, graph=graph,
            environment=BindingEnvironment())


def test_scope_flip_changes_atom_result_without_changing_graph_identity():
    """同一命题在不同显式执行 scope 可得不同四态，图身份保持不变。"""
    source = _source()
    document = document_scope(source)
    first_scope = query_scope(1, parent=document)
    second_scope = query_scope(2, parent=document)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    atom = _definition(source, 1)
    structure = structure_concept_identity((9690, 1))
    bound, graph, protocol = _bound(
        atom, (_template(atom, structure),), failures)
    resolver = _InjectedAtomResolver(
        lambda _proposition, scope: (
            (_T, 1) if scope == first_scope else (_F, 2)))
    executor = _executor(
        (), resolver, failures, logic_failures, protocol)

    first = executor.evaluate(
        bound, source=source, scope=first_scope, graph=graph,
        environment=BindingEnvironment())
    second = executor.evaluate(
        bound, source=source, scope=second_scope, graph=graph,
        environment=BindingEnvironment())

    assert first.status == STATE_PROVISIONAL
    assert second.status == STATE_REFUTED
    assert first.proposition.template == second.proposition.template


def test_exists_witness_and_forall_counterexample_work_in_open_domain():
    """开放域允许 witness 支持 EXISTS、反例反驳 FORALL，但不要求域闭合。"""
    source = _source()
    first = entity_identity(source, (9700, 1))
    second = entity_identity(source, (9700, 2))
    states = {first: _F, second: _T}

    exists, _, _ = _quantifier_case(
        ExistentialOperator, states, closed=False)
    universal, _, _ = _quantifier_case(
        UniversalOperator, states, closed=False)

    assert exists.status == STATE_PROVISIONAL
    assert universal.status == STATE_REFUTED
    assert len(exists.branches) == 2
    assert len(universal.branches) == 2
    assert all(branch.assignment is not None for branch in exists.branches)
    assert all(branch.assignment is not None for branch in universal.branches)


def test_quantifier_without_runtime_domain_resolver_stays_unknown():
    """量化域不固定在 handler；当前运行未注入 resolver 时不执行 body。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    value_type = concept_identity((9705, 1))
    binder = binder_identity(source, (9705, 2))
    variable = variable_identity(binder, (9705, 3), value_type)
    body_role = role_identity((9705, 4))
    value_role = role_identity((9705, 5))
    atom_structure = structure_concept_identity((9705, 6))
    quantifier_structure = structure_concept_identity((9705, 7))
    body = _definition(source, 1, (
        AtomicRoleBinding(value_role, variable),))
    root = _definition(source, 2, (
        AtomicRoleBinding(body_role, body.proposition),))
    templates = (
        _template(body, atom_structure),
        _template(root, quantifier_structure, (binder,)),
    )
    bound, graph, protocol = _bound(root, templates, failures)
    resolver = _InjectedAtomResolver(
        lambda _proposition, _scope: (_T, 1))
    definition = LogicOperatorDefinition(
        quantifier_structure,
        minimal_instruction_identity((9705, 8)),
        (OperatorSlot(body_role),),
        ExistentialOperator(),
    )
    evaluation = _executor(
        (definition,), resolver, failures, logic_failures, protocol,
    ).evaluate(
        bound, source=source, scope=scope, graph=graph,
        environment=BindingEnvironment())

    assert evaluation.status == STATE_UNKNOWN
    assert evaluation.failures[0].reason == logic_failures.domain_missing
    assert resolver.calls == []


def test_open_domain_does_not_upgrade_current_exhaustion_to_quantified_truth():
    """开放域当前值全反驳或全支持都不能冒充完整 EXISTS/FORALL 结论。"""
    source = _source()
    first = entity_identity(source, (9710, 1))
    second = entity_identity(source, (9710, 2))
    exists, exists_failures, _ = _quantifier_case(
        ExistentialOperator, {first: _F, second: _F}, closed=False)
    universal, universal_failures, _ = _quantifier_case(
        UniversalOperator, {first: _T, second: _T}, closed=False)

    assert exists.status == STATE_UNKNOWN
    assert universal.status == STATE_UNKNOWN
    assert exists.failures[-1].reason == exists_failures.domain_incomplete
    assert universal.failures[-1].reason == universal_failures.domain_incomplete


@pytest.mark.parametrize(
    ("operator_type", "states", "closed_status", "open_status"),
    (
        (ExistentialOperator, (_B, _F), STATE_CONFLICTED, STATE_PROVISIONAL),
        (UniversalOperator, (_B, _T), STATE_CONFLICTED, STATE_REFUTED),
    ),
)
def test_quantifier_conflict_requires_the_relevant_domain_completeness(
        operator_type, states, closed_status, open_status):
    """量化冲突的全域方向只在 closed 域成立，开放域保留单向证据。"""
    source = _source()
    values = (
        entity_identity(source, (9720, 1)),
        entity_identity(source, (9720, 2)),
    )
    mapping = dict(zip(values, states))
    closed, _, _ = _quantifier_case(
        operator_type, mapping, closed=True)
    opened, _, _ = _quantifier_case(
        operator_type, mapping, closed=False)

    assert closed.status == closed_status
    assert opened.status == open_status


def test_closed_empty_domain_has_standard_finite_quantifier_results():
    """显式闭合空域反驳 EXISTS 并支持 FORALL，且不产生伪分支。"""
    exists, _, _ = _quantifier_case(
        ExistentialOperator, {}, closed=True)
    universal, _, _ = _quantifier_case(
        UniversalOperator, {}, closed=True)

    assert exists.status == STATE_REFUTED
    assert universal.status == STATE_PROVISIONAL
    assert exists.branches == ()
    assert universal.branches == ()


def test_quantifier_type_failure_is_unknown_with_structured_binding_reason():
    """域值类型不兼容时分支 fail closed，并保留 S-03 结构化失败。"""
    source = _source()
    value = entity_identity(source, (9730, 1))
    incompatible = concept_identity((9730, 2))
    evaluation, logic_failures, resolver = _quantifier_case(
        ExistentialOperator,
        {value: _T},
        closed=True,
        actual_type=incompatible,
    )

    assert evaluation.status == STATE_UNKNOWN
    assert evaluation.failures[0].reason == logic_failures.binding_failure
    assert evaluation.failures[0].binding_failure is not None
    assert evaluation.branches[0].assignment is None
    assert resolver.calls == []


def test_nested_quantifier_passes_only_active_ancestor_binders():
    """内层 body 可见已激活祖先 Binder，不能依赖环境中未声明的 frame。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    value_type = concept_identity((9740, 1))
    outer_binder = binder_identity(source, (9740, 2))
    inner_binder = binder_identity(source, (9740, 3))
    outer_variable = variable_identity(outer_binder, (9740, 4), value_type)
    inner_variable = variable_identity(inner_binder, (9740, 5), value_type)
    outer_body_role = role_identity((9740, 6))
    inner_body_role = role_identity((9740, 7))
    outer_value_role = role_identity((9740, 8))
    inner_value_role = role_identity((9740, 9))
    atom_structure = structure_concept_identity((9740, 10))
    outer_structure = structure_concept_identity((9740, 11))
    inner_structure = structure_concept_identity((9740, 12))
    body = _definition(source, 1, (
        AtomicRoleBinding(outer_value_role, outer_variable),
        AtomicRoleBinding(inner_value_role, inner_variable),
    ))
    inner = _definition(source, 2, (
        AtomicRoleBinding(inner_body_role, body.proposition),))
    outer = _definition(source, 3, (
        AtomicRoleBinding(outer_body_role, inner.proposition),))
    templates = (
        _template(body, atom_structure),
        _template(inner, inner_structure, (inner_binder,)),
        _template(outer, outer_structure, (outer_binder,)),
    )
    bound, graph, protocol = _bound(outer, templates, failures)
    outer_value = entity_identity(source, (9740, 13))
    inner_value = entity_identity(source, (9740, 14))
    outer_domain = FiniteQuantifierDomain(
        set_expr_identity(source, (9740, 15)),
        (TypedValue(outer_value, value_type),),
        True,
        (concept_identity((9740, 16)),),
    )
    inner_domain = FiniteQuantifierDomain(
        set_expr_identity(source, (9740, 17)),
        (TypedValue(inner_value, value_type),),
        True,
        (concept_identity((9740, 18)),),
    )
    outer_quantifier = QuantifierDefinition(
        outer_binder,
        outer_variable,
        OperatorSlot(outer_body_role),
        outer_domain,
    )
    inner_quantifier = QuantifierDefinition(
        inner_binder,
        inner_variable,
        OperatorSlot(inner_body_role),
        inner_domain,
    )
    definitions = (
        LogicOperatorDefinition(
            outer_structure,
            minimal_instruction_identity((9740, 19)),
            (OperatorSlot(outer_body_role),),
            ExistentialOperator(),
        ),
        LogicOperatorDefinition(
            inner_structure,
            minimal_instruction_identity((9740, 20)),
            (OperatorSlot(inner_body_role),),
            ExistentialOperator(),
        ),
    )

    def resolve(proposition, _scope):
        fillers = {item.role: item.filler for item in proposition.bindings}
        if (fillers.get(outer_value_role) == outer_value
                and fillers.get(inner_value_role) == inner_value):
            return _T, 31
        return None

    evaluation = _executor(
        definitions,
        _InjectedAtomResolver(resolve),
        failures,
        logic_failures,
        protocol,
    ).evaluate(
        bound,
        source=source,
        scope=scope,
        graph=graph,
        environment=BindingEnvironment(),
        quantifier_resolver=_InjectedQuantifierResolver({
            outer.proposition: outer_quantifier,
            inner.proposition: inner_quantifier,
        }),
    )

    assert evaluation.status == STATE_PROVISIONAL
    assert len(evaluation.branches) == 1
    assert evaluation.failures == ()
    assert [step.operator for step in evaluation.derivation] == [
        inner_structure,
        outer_structure,
    ]


def test_modal_without_resolver_is_unknown_and_resolution_keeps_provenance():
    """MODAL 无 resolver 时弃权，有 resolver 时保留新 scope、Evidence 和 Hypothesis。"""
    source = _source()
    scope = document_scope(source)
    modal_scope = query_scope(7, parent=scope)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    atom_structure = structure_concept_identity((9750, 1))
    modal_structure = structure_concept_identity((9750, 2))
    child_role = role_identity((9750, 3))
    child = _definition(source, 1)
    root = _definition(source, 2, (
        AtomicRoleBinding(child_role, child.proposition),))
    templates = (
        _template(child, atom_structure),
        _template(root, modal_structure),
    )
    bound, graph, protocol = _bound(root, templates, failures)
    atom_resolver = _InjectedAtomResolver(
        lambda proposition, _scope: (_T, 41)
        if proposition.template == child.proposition else None)
    definition = LogicOperatorDefinition(
        modal_structure,
        minimal_instruction_identity((9750, 4)),
        (OperatorSlot(child_role),),
        ModalOperator(),
    )
    executor = _executor(
        (definition,), atom_resolver, failures, logic_failures, protocol)
    missing = executor.evaluate(
        bound, source=source, scope=scope, graph=graph,
        environment=BindingEnvironment())
    assert missing.status == STATE_UNKNOWN
    assert missing.failures[0].reason == logic_failures.modal_unknown
    assert atom_resolver.calls == []

    modal_hypothesis = HypothesisKey(
        (9750, 5), (9750, 6), (9750, 7), modal_scope, source)

    class _ModalResolver:
        """注入受限 modal 结果，不读取 operator 名称。"""

        def resolve(self, operator, child_evaluation, context):
            """返回带独立 scope 和来源证据的 provisional 结果。"""
            assert operator.structure == modal_structure
            assert child_evaluation.status == STATE_PROVISIONAL
            assert context.scope == scope
            return ModalResolution(
                _T, source, modal_scope, (42,), (modal_hypothesis,))

    resolved = executor.evaluate(
        bound,
        source=source,
        scope=scope,
        graph=graph,
        environment=BindingEnvironment(),
        modal_resolver=_ModalResolver(),
    )
    assert resolved.status == STATE_PROVISIONAL
    assert resolved.scope == modal_scope
    assert resolved.evidence_ids == (41, 42)
    assert len(resolved.hypotheses) == 2
    assert resolved.derivation[-1].scope == modal_scope
    assert resolved.derivation[-1].evidence_ids == (41, 42)


def test_recursive_handler_returns_cycle_failure_and_next_run_is_clean():
    """同一运行路径递归复用命题会失败，后续独立求值不残留活动状态。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    logic_failures = _logic_failures()
    structure = structure_concept_identity((9760, 1))
    role = role_identity((9760, 2))
    child = _definition(source, 1)
    root = _definition(source, 2, (
        AtomicRoleBinding(role, child.proposition),))
    templates = (
        _template(child, structure_concept_identity((9760, 3))),
        _template(root, structure),
    )
    bound, graph, protocol = _bound(root, templates, failures)

    class _RecursiveHandler:
        """故意重入当前命题，用于验证活动路径检测。"""

        def apply(self, executor, definition, proposition, context):
            """把当前命题原样递归交回执行器。"""
            return executor._evaluate(proposition, context)

    definition = LogicOperatorDefinition(
        structure,
        minimal_instruction_identity((9760, 4)),
        (OperatorSlot(role),),
        _RecursiveHandler(),
    )
    executor = _executor(
        (definition,),
        _InjectedAtomResolver(lambda _proposition, _scope: None),
        failures,
        logic_failures,
        protocol,
    )
    first = executor.evaluate(
        bound, source=source, scope=scope, graph=graph,
        environment=BindingEnvironment())
    second = executor.evaluate(
        bound, source=source, scope=scope, graph=graph,
        environment=BindingEnvironment())

    assert first.failures[0].reason == logic_failures.evaluation_cycle
    assert second.stable_key() == first.stable_key()


def test_logic_execution_does_not_write_backend_or_materialize_bound_values():
    """S-04 运行结果只在内存返回，不写 Backend、Core 或新 Proposition。"""
    backend = DictBackend()
    try:
        before = backend.snapshot()
        evaluation = _binary_case(ConjunctionOperator(), _T, _T)
        assert evaluation.status == STATE_PROVISIONAL
        assert backend.snapshot() == before
    finally:
        backend.close()
