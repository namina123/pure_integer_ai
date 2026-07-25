"""R-08 NOT 首切片：H-05 采用、S-04 执行和 G-00/G-01 消费。"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
    CandidateRecognitionRequest,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentProtocol,
    AnswerContentSelector,
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationLayerDecision,
    GenerationLayerRegistration,
    GenerationPlanProtocol,
    GenerationPlanner,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    HypothesisKey,
)
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
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicDerivedEvidenceSeed,
    LogicOperatorCandidateProtocol,
    LogicOperatorCandidateSpec,
    build_logic_derived_evidence,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicAtomEvidence,
    LogicEvidenceState,
    LogicFailureProtocol,
    LogicOperatorDefinition,
    ModalOperator,
    ModalResolution,
    ConjunctionOperator,
    ConditionOperator,
    DisjunctionOperator,
    ExistentialOperator,
    FiniteQuantifierDomain,
    NegationOperator,
    OperatorSlot,
    QuantifierDefinition,
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
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.logic_closure_runtime import (
    LogicClosureExecutionPlan,
    LogicClosureFormationPlan,
    LogicClosureRecognitionPlan,
    LogicClosureRoundRequest,
    LogicClosureRuntime,
    TrainingLogicClosureRuntimeBuilder,
    install_logic_closure_runtime,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.training.stages import STAGE1_SKELETON


_T = LogicEvidenceState(True, False)
_F = LogicEvidenceState(False, True)


def _source(source_id: int) -> SourceRef:
    """构造共享 owner/version 且来源身份独立的测试来源。"""
    return SourceRef(
        18001,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _binding_failures(seed: int = 18010) -> BindingFailureProtocol:
    """注入 S-03 所需的绑定失败指令。"""
    return BindingFailureProtocol(*tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 10)
    ))


def _logic_failures(seed: int = 18020) -> LogicFailureProtocol:
    """注入 S-04 所需的逻辑失败指令。"""
    return LogicFailureProtocol(*tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 10)
    ))


def _definition(
        source: SourceRef,
        key: int,
        bindings: tuple[AtomicRoleBinding, ...] = (),
        ) -> AtomicPropositionDefinition:
    """构造不携带真值、可被 S-03 绑定的原子或复合命题定义。"""
    return AtomicPropositionDefinition(
        proposition_identity(source, (18030, key)),
        concept_identity((18031, key)),
        occurrence_identity(source, start=key, end=key + 1, ordinal=0),
        context_scope_identity(source, (18032, key)),
        bindings,
    )


@dataclass(frozen=True)
class _LogicWorld:
    """集中保存一个可重复执行的 NOT typed 命题图。"""

    source: SourceRef
    root: object
    child: AtomicPropositionDefinition
    graph: PropositionTemplateGraph
    structure: ObjectIdentity
    child_role: ObjectIdentity
    substitution: SubstitutionProtocol
    binding_failures: BindingFailureProtocol
    logic_failures: LogicFailureProtocol


def _logic_world(source: SourceRef | None = None) -> _LogicWorld:
    """构造 root、child、StructureConcept、Role 和真实 bound view。"""
    source = _source(1) if source is None else source
    child = _definition(source, 1)
    structure = structure_concept_identity((18040, 2))
    child_role = role_identity((18040, 3))
    root_definition = _definition(
        source,
        2,
        (AtomicRoleBinding(child_role, child.proposition),),
    )
    graph = PropositionTemplateGraph((
        ScopedPropositionTemplate(
            child, structure_concept_identity((18040, 1))),
        ScopedPropositionTemplate(root_definition, structure),
    ))
    binding_failures = _binding_failures()
    substitution = SubstitutionProtocol(
        minimal_instruction_identity((18040, 4)), binding_failures)
    root = PropositionSubstituter(substitution).substitute(
        root_definition.proposition,
        graph,
        BindingEnvironment(),
    )
    return _LogicWorld(
        source,
        root,
        child,
        graph,
        structure,
        child_role,
        substitution,
        binding_failures,
        _logic_failures(),
    )


@dataclass(frozen=True)
class _BinaryWorld:
    """集中保存 AND/OR 使用的两个 typed 子命题和根命题图。"""

    source: SourceRef
    root: object
    left: AtomicPropositionDefinition
    right: AtomicPropositionDefinition
    graph: PropositionTemplateGraph
    structure: ObjectIdentity
    left_role: ObjectIdentity
    right_role: ObjectIdentity
    substitution: SubstitutionProtocol
    binding_failures: BindingFailureProtocol
    logic_failures: LogicFailureProtocol


def _binary_world() -> _BinaryWorld:
    """构造带两个显式 Role 槽的 AND/OR bound root。"""
    source = _source(2)
    left = _definition(source, 11)
    right = _definition(source, 12)
    structure = structure_concept_identity((18043, 3))
    left_role = role_identity((18043, 4))
    right_role = role_identity((18043, 5))
    root_definition = _definition(
        source,
        13,
        (
            AtomicRoleBinding(left_role, left.proposition),
            AtomicRoleBinding(right_role, right.proposition),
        ),
    )
    graph = PropositionTemplateGraph((
        ScopedPropositionTemplate(
            left, structure_concept_identity((18043, 1))),
        ScopedPropositionTemplate(
            right, structure_concept_identity((18043, 2))),
        ScopedPropositionTemplate(root_definition, structure),
    ))
    binding_failures = _binding_failures(18044)
    substitution = SubstitutionProtocol(
        minimal_instruction_identity((18043, 6)), binding_failures)
    root = PropositionSubstituter(substitution).substitute(
        root_definition.proposition,
        graph,
        BindingEnvironment(),
    )
    return _BinaryWorld(
        source,
        root,
        left,
        right,
        graph,
        structure,
        left_role,
        right_role,
        substitution,
        binding_failures,
        _logic_failures(18045),
    )


@dataclass(frozen=True)
class _QuantifierWorld:
    """集中保存有限域量化所需的 Binder、Variable 和 body 图。"""

    source: SourceRef
    root: object
    body: AtomicPropositionDefinition
    graph: PropositionTemplateGraph
    structure: ObjectIdentity
    body_role: ObjectIdentity
    value_role: ObjectIdentity
    binder: ObjectIdentity
    variable: ObjectIdentity
    value_type: ObjectIdentity
    values: tuple[ObjectIdentity, ...]
    substitution: SubstitutionProtocol
    binding_failures: BindingFailureProtocol
    logic_failures: LogicFailureProtocol


def _quantifier_world() -> _QuantifierWorld:
    """构造带两个 typed 域值的量化根命题和可替换 body。"""
    source = _source(3)
    value_type = concept_identity((18046, 1))
    binder = binder_identity(source, (18046, 2))
    variable = variable_identity(binder, (18046, 3), value_type)
    body_role = role_identity((18046, 4))
    value_role = role_identity((18046, 5))
    body = _definition(
        source,
        21,
        (AtomicRoleBinding(value_role, variable),),
    )
    root_definition = _definition(
        source,
        22,
        (AtomicRoleBinding(body_role, body.proposition),),
    )
    structure = structure_concept_identity((18046, 7))
    graph = PropositionTemplateGraph((
        ScopedPropositionTemplate(
            body, structure_concept_identity((18046, 6))),
        ScopedPropositionTemplate(root_definition, structure, (binder,)),
    ))
    binding_failures = _binding_failures(18047)
    substitution = SubstitutionProtocol(
        minimal_instruction_identity((18046, 8)), binding_failures)
    root = PropositionSubstituter(substitution).substitute(
        root_definition.proposition,
        graph,
        BindingEnvironment(),
    )
    values = (
        entity_identity(source, (18046, 9)),
        entity_identity(source, (18046, 10)),
    )
    return _QuantifierWorld(
        source,
        root,
        body,
        graph,
        structure,
        body_role,
        value_role,
        binder,
        variable,
        value_type,
        values,
        substitution,
        binding_failures,
        _logic_failures(18048),
    )


@dataclass(frozen=True)
class _NestedWorld:
    """保存 NOT(MODAL(atom)) 的三层命题图和两个 operator 结构。"""

    source: SourceRef
    root: object
    atom: AtomicPropositionDefinition
    graph: PropositionTemplateGraph
    modal_structure: ObjectIdentity
    not_structure: ObjectIdentity
    modal_role: ObjectIdentity
    not_role: ObjectIdentity
    substitution: SubstitutionProtocol
    binding_failures: BindingFailureProtocol
    logic_failures: LogicFailureProtocol


def _nested_world() -> _NestedWorld:
    """构造外层 NOT、内层 MODAL 和来源化 atom 的嵌套 bound 图。"""
    source = _source(4)
    atom = _definition(source, 31)
    modal_role = role_identity((18049, 1))
    not_role = role_identity((18049, 2))
    modal_definition = _definition(
        source,
        32,
        (AtomicRoleBinding(modal_role, atom.proposition),),
    )
    root_definition = _definition(
        source,
        33,
        (AtomicRoleBinding(not_role, modal_definition.proposition),),
    )
    modal_structure = structure_concept_identity((18049, 4))
    not_structure = structure_concept_identity((18049, 5))
    graph = PropositionTemplateGraph((
        ScopedPropositionTemplate(
            atom, structure_concept_identity((18049, 3))),
        ScopedPropositionTemplate(modal_definition, modal_structure),
        ScopedPropositionTemplate(root_definition, not_structure),
    ))
    binding_failures = _binding_failures(18053)
    substitution = SubstitutionProtocol(
        minimal_instruction_identity((18049, 6)), binding_failures)
    root = PropositionSubstituter(substitution).substitute(
        root_definition.proposition,
        graph,
        BindingEnvironment(),
    )
    return _NestedWorld(
        source,
        root,
        atom,
        graph,
        modal_structure,
        not_structure,
        modal_role,
        not_role,
        substitution,
        binding_failures,
        _logic_failures(18054),
    )


def _candidate_protocol() -> LogicOperatorCandidateProtocol:
    """注入逻辑候选使用的结构、指令和有序槽 predicate。"""
    return LogicOperatorCandidateProtocol(
        concept_identity((18050, 1)),
        concept_identity((18050, 2)),
        concept_identity((18050, 3)),
    )


def _operator_spec(
        structure: ObjectIdentity,
        slots: tuple[ObjectIdentity, ...],
        handler,
        *,
        candidate_source_id: int,
        competition_key: tuple[int, ...] = (18062, 1),
        ) -> LogicOperatorCandidateSpec:
    """用同一候选本体协议装配任意显式 operator handler。"""
    return LogicOperatorCandidateSpec(
        proposition_identity(
            _source(candidate_source_id),
            (18060, candidate_source_id),
        ),
        LogicOperatorDefinition(
            structure,
            minimal_instruction_identity((18061, 1)),
            tuple(OperatorSlot(role) for role in slots),
            handler,
        ),
        competition_key,
        (_source(candidate_source_id + 1), _source(candidate_source_id + 2)),
    )


def _candidate_spec(
        world: _LogicWorld,
        *,
        candidate_source_id: int = 600,
        ) -> LogicOperatorCandidateSpec:
    """把 NOT StructureConcept 映射为独立一等 Proposition 候选。"""
    return _operator_spec(
        world.structure,
        (world.child_role,),
        NegationOperator(),
        candidate_source_id=candidate_source_id,
    )


def _runtime(
        spec: LogicOperatorCandidateSpec,
        *extra_specs: LogicOperatorCandidateSpec,
        ) -> tuple[DictBackend, LogicClosureRuntime]:
    """创建带真实 H-05 owner、图投影和 NOT runtime 的测试设施。"""
    backend = DictBackend()
    context = make_train_context(backend)
    projection_protocol = CandidateProjectionProtocol(*(
        tuple(concept_identity((18070, ordinal)) for ordinal in range(1, 14))
        + ((18071, 1),)
    ))
    graph = CandidateProjectionGraph(
        context.graph_ontology,
        projection_protocol,
    )
    aggregate = _source(690)
    candidate_runtime = CandidateLearningRuntime(
        EvidenceCandidateEngine(EvidenceCandidateProtocol(
            (18072, 1),
            (18072, 2),
            aggregate,
            document_scope(aggregate),
            2,
        )),
        graph,
        IndependentObjectVerifier(IndependentVerifierProtocol(
            concept_identity((18073, 1)),
            (18073, 2),
            (18073, 3),
            (18073, 4),
            (18073, 5),
        )),
        CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
    )
    return backend, LogicClosureRuntime(
        candidate_runtime,
        _candidate_protocol(),
        (spec, *extra_specs),
    )


class _ChildAtomResolver:
    """只按 child Proposition 和当前 scope 注入原子 Evidence。"""

    def __init__(self, world: _LogicWorld, support_scope) -> None:
        self.world = world
        self.support_scope = support_scope

    def resolve(self, proposition, *, source, scope):
        """返回 scope 决定的 child 状态，根命题没有原子旁路。"""
        if proposition.template != self.world.child.proposition:
            return None
        support = scope == self.support_scope
        state = _T if support else _F
        hypothesis = HypothesisKey(
            (18080, 1),
            proposition.template.stable_key(),
            (18080, 2),
            scope,
            source,
        )
        return LogicAtomEvidence(
            proposition.template,
            state,
            source,
            scope,
            hypothesis,
            (18081,) if support else (),
            () if support else (18082,),
        )


class _BinaryAtomResolver:
    """按两个完整 child Proposition 注入本次 AND/OR 的原子 Evidence。"""

    def __init__(self, world: _BinaryWorld, states) -> None:
        self.world = world
        self.states = states

    def resolve(self, proposition, *, source, scope):
        """只按 child identity 查找状态，不读取结构名或 surface。"""
        state = self.states.get(proposition.template)
        if state is None:
            return None
        evidence_id = 18083 + proposition.template.components[-1]
        hypothesis = HypothesisKey(
            (18084, 1),
            proposition.template.stable_key(),
            (18084, 2),
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
            (evidence_id,) if state.refute else (),
            (evidence_id,) if state == LogicEvidenceState(False, False)
            else (),
        )


def _activate(
        runtime: LogicClosureRuntime,
        spec: LogicOperatorCandidateSpec,
        hypothesis,
        ) -> object:
    """对已 forming 候选执行 prediction、独立 reveal 和 active projection。"""
    observation = _source(610)
    scope = document_scope(observation)
    event_key = (18090, 1)
    timestamps = runtime.candidate_runtime.next_timestamps(3)
    runtime.recognize(CandidateRecognitionRequest(
        hypothesis,
        observation,
        scope,
        event_key,
        (
            spec.definition.structure,
            spec.definition.instruction,
            *(slot.role for slot in spec.definition.slots),
        ),
        spec.candidate,
        RevealedObjectObservation(
            observation,
            scope,
            event_key,
            _source(611),
            (spec.candidate,),
            (),
            (18091, 1),
        ),
        *timestamps,
    ))
    return hypothesis


def _form_and_activate(
        runtime: LogicClosureRuntime,
        spec: LogicOperatorCandidateSpec,
        ) -> object:
    """通过真实 H-05 forming、prediction、独立 reveal 和 active projection。"""
    hypothesis = runtime.form(spec, timestamp_base=10)
    return _activate(runtime, spec, hypothesis)


def _execute(
        runtime: LogicClosureRuntime,
        world: _LogicWorld,
        *,
        scope,
        use_key: tuple[int, ...],
        support_scope,
        modal_resolver=None,
        ):
    """执行同一 bound root，并把所有执行依赖显式注入 runtime。"""
    return runtime.execute(
        world.root,
        use_key=use_key,
        source=world.source,
        scope=scope,
        graph=world.graph,
        environment=BindingEnvironment(),
        atom_resolver=_ChildAtomResolver(world, support_scope),
        failures=world.logic_failures,
        substitution=world.substitution,
        type_resolver=ExactTypeCompatibilityResolver(),
        binding_failures=world.binding_failures,
        modal_resolver=modal_resolver,
    )


def _execute_binary(
        runtime: LogicClosureRuntime,
        world: _BinaryWorld,
        *,
        left_state: LogicEvidenceState = _T,
        right_state: LogicEvidenceState = _F,
        use_key: tuple[int, ...] = (18150, 1),
        ):
    """执行二元 operator，并保持 AND/OR 的两个 Role 槽完全来源化。"""
    scope = document_scope(world.source)
    return runtime.execute(
        world.root,
        use_key=use_key,
        source=world.source,
        scope=scope,
        graph=world.graph,
        environment=BindingEnvironment(),
        atom_resolver=_BinaryAtomResolver(
            world,
            {
                world.left.proposition: left_state,
                world.right.proposition: right_state,
            },
        ),
        failures=world.logic_failures,
        substitution=world.substitution,
        type_resolver=ExactTypeCompatibilityResolver(),
        binding_failures=world.binding_failures,
    )


class _QuantifierAtomResolver:
    """从 substitution 后的 value Role 读取有限域分支 Evidence。"""

    def __init__(self, world: _QuantifierWorld, states) -> None:
        self.world = world
        self.states = states

    def resolve(self, proposition, *, source, scope):
        """只为量化 body 的完整 filler 注入对应四态。"""
        if proposition.template != self.world.body.proposition:
            return None
        filler = next(
            item.filler for item in proposition.bindings
            if item.role == self.world.value_role
        )
        state = self.states.get(filler)
        if state is None:
            return None
        evidence_id = 18160 + self.world.values.index(filler)
        hypothesis = HypothesisKey(
            (18161, 1),
            proposition.stable_key(),
            (18161, 2),
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
            (evidence_id + 10,) if state.refute else (),
            (evidence_id + 20,)
            if state == LogicEvidenceState(False, False) else (),
        )


class _QuantifierResolver:
    """按当前根命题注入 Binder、Variable 和开放或闭合有限域。"""

    def __init__(self, world: _QuantifierWorld, *, closed: bool) -> None:
        self.world = world
        domain = FiniteQuantifierDomain(
            set_expr_identity(world.source, (18162, 1)),
            tuple(TypedValue(value, world.value_type) for value in world.values),
            closed,
            (concept_identity((18162, 2)),) if closed else (),
        )
        self.definition = QuantifierDefinition(
            world.binder,
            world.variable,
            OperatorSlot(world.body_role),
            domain,
        )

    def resolve(self, operator, proposition, context):
        """只在 operator、根命题和执行来源都对齐时返回量化定义。"""
        if (operator.structure != self.world.structure
                or proposition.template != self.world.root.template
                or context.source != self.world.source):
            return None
        return self.definition


def _execute_quantifier(
        runtime: LogicClosureRuntime,
        world: _QuantifierWorld,
        *,
        states,
        closed: bool,
        use_key: tuple[int, ...],
        ):
    """通过真实临时 BindingFrame 执行有限域量化候选。"""
    scope = document_scope(world.source)
    return runtime.execute(
        world.root,
        use_key=use_key,
        source=world.source,
        scope=scope,
        graph=world.graph,
        environment=BindingEnvironment(),
        atom_resolver=_QuantifierAtomResolver(world, states),
        failures=world.logic_failures,
        substitution=world.substitution,
        type_resolver=ExactTypeCompatibilityResolver(),
        binding_failures=world.binding_failures,
        quantifier_resolver=_QuantifierResolver(world, closed=closed),
    )


class _InjectedModalResolver:
    """用独立 Evidence 把 child 结果投影到显式 modal scope。"""

    def __init__(self, world: _LogicWorld, scope) -> None:
        self.world = world
        self.scope = scope

    def resolve(self, operator, child, context):
        """只在结构、子命题和来源对齐时返回受限 modal 结果。"""
        if (operator.structure != self.world.structure
                or child.proposition.template != self.world.child.proposition
                or context.source != self.world.source):
            return None
        hypothesis = HypothesisKey(
            (18165, 1),
            child.proposition.stable_key(),
            (18165, 2),
            self.scope,
            self.world.source,
        )
        return ModalResolution(
            _T,
            self.world.source,
            self.scope,
            (18166,),
            (hypothesis,),
        )


class _NestedAtomResolver:
    """只为嵌套图最内层 atom 提供当前 scope 的支持 Evidence。"""

    def __init__(self, world: _NestedWorld) -> None:
        self.world = world

    def resolve(self, proposition, *, source, scope):
        """复合层没有原子旁路，只有 atom identity 可命中。"""
        if proposition.template != self.world.atom.proposition:
            return None
        hypothesis = HypothesisKey(
            (18168, 1),
            proposition.stable_key(),
            (18168, 2),
            scope,
            source,
        )
        return LogicAtomEvidence(
            proposition.template,
            _T,
            source,
            scope,
            hypothesis,
            (18169,),
        )


class _NestedModalResolver:
    """为嵌套图内层 MODAL 提供独立 scope 和 Evidence。"""

    def __init__(self, world: _NestedWorld, scope) -> None:
        self.world = world
        self.scope = scope

    def resolve(self, operator, child, context):
        """仅匹配内层 modal 结构和 atom，不干预外层 NOT。"""
        if (operator.structure != self.world.modal_structure
                or child.proposition.template != self.world.atom.proposition):
            return None
        hypothesis = HypothesisKey(
            (18170, 1),
            child.proposition.stable_key(),
            (18170, 2),
            self.scope,
            context.source,
        )
        return ModalResolution(
            _T,
            context.source,
            self.scope,
            (18171,),
            (hypothesis,),
        )


def _execute_nested(
        runtime: LogicClosureRuntime,
        world: _NestedWorld,
        *,
        input_scope,
        modal_scope,
        use_key: tuple[int, ...],
        ):
    """执行嵌套 operator，并保留内外两层 scope 的 derivation。"""
    return runtime.execute(
        world.root,
        use_key=use_key,
        source=world.source,
        scope=input_scope,
        graph=world.graph,
        environment=BindingEnvironment(),
        atom_resolver=_NestedAtomResolver(world),
        failures=world.logic_failures,
        substitution=world.substitution,
        type_resolver=ExactTypeCompatibilityResolver(),
        binding_failures=world.binding_failures,
        modal_resolver=_NestedModalResolver(world, modal_scope),
    )


class _GenerationPolicy:
    """把真实逻辑候选交给 G-01 的共享 selector，不读取 surface。"""

    def __init__(self, protocol: AnswerContentProtocol) -> None:
        self.protocol = protocol

    def select(self, request, artifacts):
        """只选择请求内满足目标四态的候选。"""
        del artifacts
        return AnswerContentDecision(
            self.protocol.answer,
            minimal_instruction_identity((18100, 1)),
            request.candidate_keys(),
            (),
            (18101, 1),
        )


class _CompleteGenerationLayer:
    """为 G-00 后四层提供无语义捷径的 complete 记录。"""

    def __init__(self, layer, protocol, seed: int) -> None:
        self.layer = layer
        self.protocol = protocol
        self.seed = seed

    def resolve(self, request, prior):
        """保留前层输入，只证明 G-00 能继续调度。"""
        del request, prior
        return GenerationLayerDecision(
            self.layer,
            self.protocol.complete,
            minimal_instruction_identity((18102, self.seed)),
            payload=(self.seed,),
            trace=(18103, self.seed),
        )


def _generation_request(bundle) -> GenerationPlanningRequest:
    """把逻辑派生 Evidence 组装成 G-00 typed 回答请求。"""
    candidate = bundle.candidate
    goal = AnswerGenerationGoal(
        minimal_instruction_identity((18104, 1)),
        candidate.proposition,
        candidate.state,
        candidate.source,
        candidate.scope,
    )
    return GenerationPlanningRequest(goal, (candidate,))


def test_not_without_active_evidence_stays_unknown():
    """只有 StructureConcept、handler 和 forming 不足以激活 NOT。"""
    world = _logic_world()
    spec = _candidate_spec(world)
    backend, runtime = _runtime(spec)
    try:
        runtime.form(spec, timestamp_base=10)
        assert runtime.registry_snapshot().registry.get(world.structure) is None
        execution = _execute(
            runtime,
            world,
            scope=document_scope(world.source),
            use_key=(18110, 1),
            support_scope=document_scope(world.source),
        )
        assert execution.evaluation.status == STATE_UNKNOWN
        assert execution.evaluation.derivation == ()
        assert execution.adoptions == ()
    finally:
        backend.close()


def test_not_adoption_trace_and_derived_candidate_reach_g00_g01():
    """active NOT 保留采用归因，并由独立派生 Evidence 进入 G-00/G-01。"""
    world = _logic_world()
    spec = _candidate_spec(world)
    backend, runtime = _runtime(spec)
    try:
        hypothesis = _form_and_activate(runtime, spec)
        scope = document_scope(world.source)
        execution = _execute(
            runtime,
            world,
            scope=scope,
            use_key=(18120, 1),
            support_scope=scope,
        )
        assert execution.evaluation.status == STATE_REFUTED
        assert len(execution.adoptions) == 1
        assert execution.adoptions[0].hypothesis == hypothesis
        assert execution.evaluation.derivation[0].operator == world.structure
        assert execution.adoptions[0].evidence

        content_hypothesis = HypothesisKey(
            (18121, 1),
            execution.evaluation.proposition.stable_key(),
            (18121, 2),
            scope,
            world.source,
        )
        bundle = build_logic_derived_evidence(
            execution,
            content_hypothesis,
            (LogicDerivedEvidenceSeed(
                EVIDENCE_REFUTE,
                18122,
                (18123, 1),
                18124,
                (18125, 1),
            ),),
        )
        assert bundle.candidate.state == execution.evaluation.state
        assert bundle.candidate.evidence == bundle.evidence
        assert bundle.evidence[0].hypothesis == content_hypothesis
        assert bundle.evidence[0].hypothesis != execution.adoptions[0].hypothesis

        request = _generation_request(bundle)
        content_protocol = AnswerContentProtocol(*tuple(
            minimal_instruction_identity((18130, index))
            for index in range(1, 6)
        ))
        selector = AnswerContentSelector(
            content_protocol,
            _GenerationPolicy(content_protocol),
        )
        plan_protocol = GenerationPlanProtocol(*tuple(
            minimal_instruction_identity((18131, index))
            for index in range(1, 11)
        ))
        registrations = [
            GenerationLayerRegistration(
                plan_protocol.stance_layer,
                GenerationStanceLayerResolver(plan_protocol, selector),
            ),
            GenerationLayerRegistration(
                plan_protocol.content_layer,
                GenerationContentLayerResolver(plan_protocol, selector),
            ),
        ]
        registrations.extend(
            GenerationLayerRegistration(
                layer,
                _CompleteGenerationLayer(layer, plan_protocol, index),
            )
            for index, layer in enumerate(plan_protocol.layers()[2:], start=1)
        )
        plan = GenerationPlanner(
            plan_protocol,
            tuple(registrations),
        ).plan(request)
        assert plan.complete is True
        assert plan.layers[0].payload == plan.layers[1].payload
        assert bundle.candidate.stable_key() in request.candidate_keys()
    finally:
        backend.close()


def test_not_scope_flip_changes_state_without_changing_graph_identity():
    """同一 typed 图在不同 execution scope 下改变原子证据，NOT 结果随之翻转。"""
    world = _logic_world()
    spec = _candidate_spec(world, candidate_source_id=620)
    backend, runtime = _runtime(spec)
    try:
        _form_and_activate(runtime, spec)
        scope_a = document_scope(world.source)
        scope_b = query_scope(2, parent=scope_a)
        root_key = world.root.stable_key()
        first = _execute(
            runtime,
            world,
            scope=scope_a,
            use_key=(18140, 1),
            support_scope=scope_a,
        )
        second = _execute(
            runtime,
            world,
            scope=scope_b,
            use_key=(18140, 2),
            support_scope=scope_a,
        )
        assert first.evaluation.state == _F
        assert second.evaluation.state == _T
        assert first.evaluation.proposition.template == world.root.template
        assert second.evaluation.proposition.template == world.root.template
        assert world.root.stable_key() == root_key
        assert world.graph.get(world.root.template).structure == world.structure
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("handler", "expected_status", "candidate_source_id"),
    (
        (ConjunctionOperator(), STATE_REFUTED, 651),
        (DisjunctionOperator(), STATE_PROVISIONAL, 652),
    ),
)
def test_and_or_use_the_same_h05_owner_and_typed_slots(
        handler, expected_status, candidate_source_id):
    """AND/OR 只替换 handler，仍经 active adoption 和有序 Role 槽执行。"""
    world = _binary_world()
    spec = _operator_spec(
        world.structure,
        (world.left_role, world.right_role),
        handler,
        candidate_source_id=candidate_source_id,
    )
    backend, runtime = _runtime(spec)
    try:
        _form_and_activate(runtime, spec)
        execution = _execute_binary(
            runtime,
            world,
        )
        assert execution.evaluation.status == expected_status
        assert len(execution.adoptions) == 1
        assert execution.adoptions[0].spec.candidate == spec.candidate
        assert execution.evaluation.derivation[0].operator == world.structure
        assert execution.evaluation.derivation[0].premises == (
            world.left.proposition,
            world.right.proposition,
        )
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("left_state", "right_state", "expected_status", "candidate_source_id"),
    (
        (_T, LogicEvidenceState(False, False), STATE_UNKNOWN, 661),
        (LogicEvidenceState(False, False), _T, STATE_PROVISIONAL, 662),
    ),
)
def test_condition_preserves_order_and_unknown_propagation(
        left_state, right_state, expected_status, candidate_source_id):
    """CONDITION 按 Role 槽顺序执行，unknown 不因前件位置被伪造成反驳。"""
    world = _binary_world()
    spec = _operator_spec(
        world.structure,
        (world.left_role, world.right_role),
        ConditionOperator(),
        candidate_source_id=candidate_source_id,
    )
    backend, runtime = _runtime(spec)
    try:
        _form_and_activate(runtime, spec)
        execution = _execute_binary(
            runtime,
            world,
            left_state=left_state,
            right_state=right_state,
            use_key=(18151, 1 if expected_status == STATE_UNKNOWN else 2),
        )
        assert execution.evaluation.status == expected_status
        assert len(execution.adoptions) == 1
    finally:
        backend.close()


def test_exists_requires_active_candidate_and_respects_domain_closure():
    """EXISTS 只在 adopted 后执行；开放域全反驳 unknown，显式 witness 支持。"""
    world = _quantifier_world()
    spec = _operator_spec(
        world.structure,
        (world.body_role,),
        ExistentialOperator(),
        candidate_source_id=670,
    )
    backend, runtime = _runtime(spec)
    try:
        scope = document_scope(world.source)
        hypothesis = runtime.form(spec, timestamp_base=10)
        before = _execute_quantifier(
            runtime,
            world,
            states={value: _F for value in world.values},
            closed=False,
            use_key=(18163, 1),
        )
        assert before.evaluation.status == STATE_UNKNOWN
        assert before.adoptions == ()

        _activate(runtime, spec, hypothesis)
        open_result = _execute_quantifier(
            runtime,
            world,
            states={value: _F for value in world.values},
            closed=False,
            use_key=(18163, 2),
        )
        witness_result = _execute_quantifier(
            runtime,
            world,
            states={world.values[0]: _T, world.values[1]: _F},
            closed=True,
            use_key=(18163, 3),
        )
        assert open_result.evaluation.status == STATE_UNKNOWN
        assert witness_result.evaluation.status == STATE_PROVISIONAL
        assert len(open_result.evaluation.branches) == len(world.values)
        assert len(witness_result.evaluation.branches) == len(world.values)
        assert len(witness_result.adoptions) == 1
        assert all(
            branch.assignment is not None
            for branch in witness_result.evaluation.branches
        )
        assert witness_result.evaluation.scope == scope
    finally:
        backend.close()


def test_forall_requires_closed_support_but_accepts_explicit_counterexample():
    """FORALL 不从开放域全支持证真，但任何显式反例都可定向 refute。"""
    world = _quantifier_world()
    spec = _operator_spec(
        world.structure,
        (world.body_role,),
        UniversalOperator(),
        candidate_source_id=680,
    )
    backend, runtime = _runtime(spec)
    try:
        _form_and_activate(runtime, spec)
        all_support = {value: _T for value in world.values}
        open_result = _execute_quantifier(
            runtime,
            world,
            states=all_support,
            closed=False,
            use_key=(18164, 1),
        )
        closed_result = _execute_quantifier(
            runtime,
            world,
            states=all_support,
            closed=True,
            use_key=(18164, 2),
        )
        counterexample = _execute_quantifier(
            runtime,
            world,
            states={world.values[0]: _T, world.values[1]: _F},
            closed=False,
            use_key=(18164, 3),
        )
        assert open_result.evaluation.status == STATE_UNKNOWN
        assert closed_result.evaluation.status == STATE_PROVISIONAL
        assert counterexample.evaluation.status == STATE_REFUTED
        assert len(closed_result.adoptions) == 1
        assert all(
            branch.assignment is not None
            for branch in closed_result.evaluation.branches
        )
    finally:
        backend.close()


def test_modal_requires_independent_resolution_and_preserves_scope_trace():
    """MODAL 缺 resolver unknown；独立 resolver 可改 scope 但不能改 source。"""
    world = _logic_world()
    spec = _operator_spec(
        world.structure,
        (world.child_role,),
        ModalOperator(),
        candidate_source_id=690,
    )
    backend, runtime = _runtime(spec)
    try:
        _form_and_activate(runtime, spec)
        input_scope = document_scope(world.source)
        missing = _execute(
            runtime,
            world,
            scope=input_scope,
            use_key=(18167, 1),
            support_scope=input_scope,
        )
        modal_scope = query_scope(73, parent=input_scope)
        resolved = _execute(
            runtime,
            world,
            scope=input_scope,
            use_key=(18167, 2),
            support_scope=input_scope,
            modal_resolver=_InjectedModalResolver(world, modal_scope),
        )
        assert missing.evaluation.status == STATE_UNKNOWN
        assert missing.adoptions == ()
        assert resolved.evaluation.status == STATE_PROVISIONAL
        assert resolved.evaluation.source == world.source
        assert resolved.evaluation.scope == modal_scope
        assert resolved.evaluation.evidence_ids == (18081, 18166)
        assert len(resolved.adoptions) == 1
    finally:
        backend.close()


def test_nested_scope_requires_both_active_operators_and_attributes_each_use():
    """NOT(MODAL(atom)) 消融内层回退 unknown，完整链保留两层 scope/adoption。"""
    world = _nested_world()
    modal_spec = _operator_spec(
        world.modal_structure,
        (world.modal_role,),
        ModalOperator(),
        candidate_source_id=700,
        competition_key=(18172, 1),
    )
    not_spec = _operator_spec(
        world.not_structure,
        (world.not_role,),
        NegationOperator(),
        candidate_source_id=710,
        competition_key=(18172, 2),
    )
    backend, runtime = _runtime(modal_spec, not_spec)
    try:
        modal_hypothesis = runtime.form(modal_spec, timestamp_base=10)
        not_hypothesis = runtime.form(not_spec, timestamp_base=12)
        _activate(runtime, not_spec, not_hypothesis)
        input_scope = document_scope(world.source)
        modal_scope = query_scope(74, parent=input_scope)
        ablated = _execute_nested(
            runtime,
            world,
            input_scope=input_scope,
            modal_scope=modal_scope,
            use_key=(18173, 1),
        )
        assert ablated.evaluation.status == STATE_UNKNOWN
        assert {item.spec.candidate for item in ablated.adoptions} == {
            not_spec.candidate,
        }

        _activate(runtime, modal_spec, modal_hypothesis)
        complete = _execute_nested(
            runtime,
            world,
            input_scope=input_scope,
            modal_scope=modal_scope,
            use_key=(18173, 2),
        )
        assert complete.evaluation.status == STATE_REFUTED
        assert complete.evaluation.scope == input_scope
        assert tuple(
            step.operator for step in complete.evaluation.derivation
        ) == (world.modal_structure, world.not_structure)
        assert tuple(
            step.scope for step in complete.evaluation.derivation
        ) == (modal_scope, input_scope)
        assert {item.spec.candidate for item in complete.adoptions} == {
            modal_spec.candidate,
            not_spec.candidate,
        }
        assert complete.evaluation.evidence_ids == (18169, 18171)
    finally:
        backend.close()


def test_candidate_identity_replacement_keeps_typed_not_shape():
    """替换候选来源身份不改变同一 StructureConcept 的指令和 Role 槽形状。"""
    world = _logic_world()
    first = _candidate_spec(world, candidate_source_id=630)
    second = _candidate_spec(world, candidate_source_id=640)
    protocol = _candidate_protocol()
    first_definition = first.candidate_definition(protocol)
    second_definition = second.candidate_definition(protocol)
    assert first.candidate != second.candidate
    assert first.definition.stable_key() == second.definition.stable_key()
    assert first_definition.bindings[1:] == second_definition.bindings[1:]
    assert first_definition.candidate != second_definition.candidate


def _production_projection_protocol() -> CandidateProjectionProtocol:
    """构造生产 caller 测试使用的独立 H-05 lifecycle 图协议。"""
    return CandidateProjectionProtocol(*(
        tuple(concept_identity((18200, ordinal)) for ordinal in range(1, 14))
        + ((18201, 1),)
    ))


def _production_builder(
        spec: LogicOperatorCandidateSpec,
        ) -> TrainingLogicClosureRuntimeBuilder:
    """构造使用 Core 训练历史而非进程内临时 ledger 的标准 builder。"""
    aggregate = _source(800)
    learning = EvidenceCandidateProtocol(
        (18202, 1),
        (18202, 2),
        aggregate,
        document_scope(aggregate),
        2,
    )
    history = TrainingHypothesisHistoryProtocol(
        (18204, 1),
        learning.hypothesis_kind_key,
        learning.aggregate_source,
        learning.aggregate_scope,
    )
    return TrainingLogicClosureRuntimeBuilder(
        (18204, 2),
        learning,
        _production_projection_protocol(),
        IndependentVerifierProtocol(
            concept_identity((18203, 1)),
            (18203, 2),
            (18203, 3),
            (18203, 4),
            (18203, 5),
        ),
        CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
        _candidate_protocol(),
        history,
        (spec,),
    )


_COURSE_NORMAL = 0
_COURSE_WRITE_READ_ONLY = 1
_COURSE_MISSING_REPLACEMENT = 2
_COURSE_WRONG_SCOPE = 3
_COURSE_DUPLICATE_ROUTE = 4


@dataclass(frozen=True)
class _ProductionCourse:
    """把来源 scope 映射为 NOT forming、独立核验和 typed 执行。"""

    world: _LogicWorld
    spec: LogicOperatorCandidateSpec
    mode: int = _COURSE_NORMAL

    def request(self, scope, *, read_only: bool) -> LogicClosureRoundRequest:
        """按测试模式返回训练写入或只读执行请求。"""
        effective_scope = (
            query_scope(899, parent=document_scope(self.world.source))
            if self.mode == _COURSE_WRONG_SCOPE else scope
        )
        should_write = (
            not read_only or self.mode == _COURSE_WRITE_READ_ONLY
        )
        formations = ()
        recognitions = ()
        if should_write:
            formations = (LogicClosureFormationPlan(self.spec, 10),)
            event_key = (18210, 1)
            replacement = (
                proposition_identity(_source(898), (18210, 2))
                if self.mode == _COURSE_MISSING_REPLACEMENT else None
            )
            recognition = LogicClosureRecognitionPlan(
                self.spec.candidate,
                self.world.source,
                event_key,
                (self.world.structure,),
                RevealedObjectObservation(
                    self.world.source,
                    effective_scope,
                    event_key,
                    _source(897),
                    supported_targets=(self.spec.candidate,),
                    trace=(18210, 3),
                ),
                replacement=replacement,
            )
            recognitions = (
                (recognition, recognition)
                if self.mode == _COURSE_DUPLICATE_ROUTE
                else (recognition,)
            )
        hypothesis = HypothesisKey(
            (18211, 1),
            self.world.root.stable_key(),
            (18211, 2),
            effective_scope,
            self.world.source,
        )
        execution = LogicClosureExecutionPlan(
            self.world.root,
            (18212, 1),
            self.world.source,
            self.world.graph,
            BindingEnvironment(),
            _ChildAtomResolver(self.world, effective_scope),
            self.world.logic_failures,
            self.world.substitution,
            ExactTypeCompatibilityResolver(),
            self.world.binding_failures,
            hypothesis,
            (LogicDerivedEvidenceSeed(
                EVIDENCE_REFUTE,
                18213,
                (18214, 1),
                100,
                (18215, 1),
            ),),
        )
        return LogicClosureRoundRequest(
            effective_scope,
            formations,
            recognitions,
            (execution,),
        )

    def clone_for_evaluation(self) -> "_ProductionCourse":
        """返回不共享调用游标的冻结课程副本。"""
        return _ProductionCourse(self.world, self.spec, self.mode)

    def state_key(self) -> tuple[int, ...]:
        """返回课程版本、模式和来源身份键。"""
        return (
            18216,
            1,
            self.mode,
            *self.world.source.stable_key(),
        )


def _production_fixture(*, mode: int = _COURSE_NORMAL):
    """构造安装到真实 TrainContext 的 R-08 builder/course 设施。"""
    ctx = make_train_context(DictBackend())
    world = _logic_world()
    spec = _candidate_spec(world, candidate_source_id=810)
    runtime = install_logic_closure_runtime(
        ctx,
        _production_builder(spec),
        _ProductionCourse(world, spec, mode),
    )
    return ctx, world, runtime


def test_production_course_assigns_logic_order_and_builds_g00_candidate():
    """生产 course 真实形成、激活、执行 NOT 并产出 G-00 typed 候选。"""
    ctx, world, runtime = _production_fixture()
    try:
        scope = document_scope(world.source)
        report = runtime.process(scope, read_only=False)
        assert len(report.formations) == 1
        assert len(report.recognitions) == 1
        assert len(report.executions) == 1
        recognition = report.recognitions[0]
        assert recognition.evidence.timestamp_seq > 11
        assert recognition.decision.timestamp_seq > recognition.evidence.timestamp_seq
        bundle = report.executions[0]
        assert bundle.execution.evaluation.status == STATE_REFUTED
        assert bundle.candidate.proposition == world.root
        assert bundle.candidate.evidence == bundle.evidence
        assert len(bundle.execution.adoptions) == 1
    finally:
        ctx.backend.close()


def test_production_course_clone_executes_read_only_without_host_write():
    """V-06 clone 可只读采用 active NOT，宿主图、owner 和报告保持不变。"""
    ctx, world, runtime = _production_fixture()
    try:
        scope = document_scope(world.source)
        runtime.process(scope, read_only=False)
        baseline_backend = ctx.backend.snapshot()
        baseline_owner = runtime.state_key()
        baseline_reports = tuple(ctx.logic_closure_reports)

        with isolated_evaluation(ctx, label="r08-held-out") as eval_ctx:
            report = eval_ctx.logic_closure_runtime.process(
                scope,
                read_only=True,
            )
            assert report.formations == ()
            assert report.recognitions == ()
            assert report.executions[0].execution.evaluation.status == STATE_REFUTED

        assert ctx.backend.snapshot() == baseline_backend
        assert runtime.state_key() == baseline_owner
        assert tuple(ctx.logic_closure_reports) == baseline_reports
    finally:
        ctx.backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_training_builder_restores_and_continues_h05_history(backend_type):
    """标准 builder 从 Core 历史恢复同态 owner，并可继续追加 recognition。"""
    ctx = make_train_context(backend_type())
    world = _logic_world()
    spec = _candidate_spec(world, candidate_source_id=815)
    builder = _production_builder(spec)
    runtime = install_logic_closure_runtime(
        ctx,
        builder,
        _ProductionCourse(world, spec),
    )
    try:
        scope = document_scope(world.source)
        runtime.process(scope, read_only=False)
        rebuilt = builder.build(ctx)
        assert rebuilt.state_key() == runtime.owner.state_key()
        execution = _execute(
            rebuilt,
            world,
            scope=scope,
            use_key=(18217, 1),
            support_scope=scope,
        )
        assert execution.evaluation.status == STATE_REFUTED
        assert len(execution.adoptions) == 1

        hypothesis = rebuilt.candidate_runtime.hypothesis_for_candidate(
            spec.candidate)
        timestamps = rebuilt.candidate_runtime.next_timestamps(3)
        event_key = (18217, 2)
        rebuilt.recognize(CandidateRecognitionRequest(
            hypothesis,
            world.source,
            scope,
            event_key,
            (world.structure,),
            spec.candidate,
            RevealedObjectObservation(
                world.source,
                scope,
                event_key,
                _source(896),
                supported_targets=(spec.candidate,),
                trace=(18217, 3),
            ),
            *timestamps,
        ))
        rebuilt_again = builder.build(ctx)
        assert rebuilt_again.state_key() == rebuilt.state_key()
        assert ctx.training_candidate_history.entries(builder.history_protocol)
    finally:
        ctx.backend.close()


def test_training_builder_accepts_preexisting_candidate_proposition():
    """候选 Proposition 本体可先存在，只有孤立 lifecycle/Hypothesis 才应阻断恢复。"""
    ctx = make_train_context(DictBackend())
    world = _logic_world()
    spec = _candidate_spec(world, candidate_source_id=818)
    ctx.graph_ontology.materialize(spec.candidate)
    try:
        runtime = install_logic_closure_runtime(
            ctx,
            _production_builder(spec),
            _ProductionCourse(world, spec),
        )
        report = runtime.process(
            document_scope(world.source),
            read_only=False,
        )
        assert report.recognitions[0].projection is not None
        assert report.executions[0].execution.evaluation.status == STATE_REFUTED
    finally:
        ctx.backend.close()


@pytest.mark.parametrize(
    ("mode", "read_only", "message"),
    (
        (_COURSE_WRITE_READ_ONLY, True, "read-only"),
        (_COURSE_MISSING_REPLACEMENT, False, "replacement"),
        (_COURSE_WRONG_SCOPE, False, "替换了 round scope"),
        (_COURSE_DUPLICATE_ROUTE, False, "重复 recognition 路由"),
    ),
)
def test_production_course_rejects_invalid_round_before_h05_write(
        mode, read_only, message):
    """只读写入、缺 replacement、错 scope 和重复路由均保持首写前失败。"""
    ctx, world, runtime = _production_fixture(mode=mode)
    try:
        baseline_backend = ctx.backend.snapshot()
        baseline_owner = runtime.state_key()
        with pytest.raises(ValueError, match=message):
            runtime.process(
                document_scope(world.source),
                read_only=read_only,
            )
        assert ctx.backend.snapshot() == baseline_backend
        assert runtime.state_key() == baseline_owner
    finally:
        ctx.backend.close()


def test_formal_train_calls_logic_course_and_returns_real_report(
        tmp_path, monkeypatch):
    """顶层 formal_train 成对安装 R-08，并由 round caller 返回真实执行报告。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    source = SourceRef(
        SOURCE_BARE_TEXT,
        18220,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )
    world = _logic_world(source)
    spec = _candidate_spec(world, candidate_source_id=820)
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r08-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            persist_graph_dump=False,
            language_logic_closure_builder=_production_builder(spec),
            language_logic_closure_course=_ProductionCourse(world, spec),
        ),
        [CollectedItem(
            tokens=["甲", "事实"],
            raw_text="甲事实",
            role_seq=[1, 1],
            source=SOURCE_BARE_TEXT,
            source_ref=source,
        )],
        backend=DictBackend(),
    )

    assert result.logic_closure_reports
    report = result.logic_closure_reports[-1]
    assert report.formations
    assert report.recognitions
    assert report.executions[0].execution.evaluation.status == STATE_REFUTED


def test_formal_train_rejects_partial_logic_configuration(tmp_path):
    """R-08 builder/course 缺任一项时必须在训练 round 前 fail closed。"""
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="r08-partial",
        active_training_stages=(),
        persist_graph_dump=False,
        language_logic_closure_builder=object(),
    )
    with pytest.raises(ValueError, match="必须成对配置"):
        formal_train(config, [], backend=DictBackend())


def test_formal_train_without_logic_configuration_keeps_optional_path_off(
        tmp_path):
    """未配置 R-08 时不安装 caller、不生成报告，也不要求任何逻辑协议。"""
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r08-off",
            active_training_stages=(),
            persist_graph_dump=False,
        ),
        [],
        backend=DictBackend(),
    )
    assert result.logic_closure_reports == ()


__all__ = []
