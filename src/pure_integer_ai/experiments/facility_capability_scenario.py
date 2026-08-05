"""用生产 C-00/C-01/C-02 owner 执行 F-01 Capability 复用场景。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pure_integer_ai.cognition.shared.artifact_binding import (
    ArtifactBindingChoice,
    ArtifactBindingEndpoint,
    ArtifactBindingRequest,
    ArtifactBindingValue,
)
from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorBudget,
    AttractorProtocol,
)
from pure_integer_ai.cognition.shared.capability_activation import (
    CapabilityActivationMapper,
    CapabilityObligationProjection,
)
from pure_integer_ai.cognition.shared.capability_candidate import (
    CapabilityCandidateProposal,
    CapabilityExample,
    CapabilityFormationRequest,
    CapabilityFormationRuntime,
    CapabilityStatusProtocol,
)
from pure_integer_ai.cognition.shared.capability_resolver import (
    CapabilityActivationAssessment,
    CapabilityResolverRoute,
)
from pure_integer_ai.cognition.shared.capability_verification import (
    CapabilityHeldOutCase,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactArgument,
    ArtifactAuthority,
    ArtifactInvocation,
    ArtifactParameter,
    ArtifactSchema,
    ExactArtifactCompatibilityResolver,
    FormalArtifact,
    FormalArtifactDefinition,
    artifact_identity,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    FormalArtifactBridge,
    FormalArtifactFailureProtocol,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchRuntimeConfig,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_CAPABILITY,
    MEMORY_OBJECT_OBSERVATION,
    MemoryEvent,
    MemoryLinkedRef,
    ObservationPayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_hot_set import (
    StableTopKSourcePolicy,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryCurrentQuery,
    MemoryQueryDefinition,
    MemoryQueryProtocol,
    MemoryQueryRoles,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    ActivationScore,
    ActivationScoreReason,
    MemoryAggregateFilter,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningObligation,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_OBSERVED,
    CLOCK_MEMORY_USED,
    CLOCK_QUERY,
    LogicalClock,
    LogicalClockIdentity,
    LogicalTimestamp,
    document_scope,
    episode_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    binder_identity,
    context_scope_identity,
    proposition_identity,
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
)
from pure_integer_ai.cognition.understanding.struct_bind_typed_adapter import (
    StructBindTypedAdapter,
    TypedStructBindEndpoint,
)
from pure_integer_ai.experiments.artifact_binding_runtime import (
    ArtifactBindingRuntime,
)
from pure_integer_ai.experiments.attractor_runtime import (
    install_attractor_runtime,
)
from pure_integer_ai.experiments.capability_execution_runtime import (
    CapabilityExecutionProtocol,
    CapabilityExecutionRuntime,
)
from pure_integer_ai.experiments.capability_memory_runtime import (
    CapabilityMemoryRuntime,
)
from pure_integer_ai.experiments.capability_verification_runtime import (
    CapabilityVerificationRuntime,
)
from pure_integer_ai.experiments.evaluation_protocol import CanonicalIdentity
from pure_integer_ai.experiments.memory_query_runtime import (
    install_memory_query_runtime,
)
from pure_integer_ai.experiments.memory_resolver_runtime import (
    install_memory_resolver_runtime,
)
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_ADD,
    OPCODE_HALT,
    OPCODE_LOAD,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_STRUCT_BIND
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
    MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.training.formal_artifact_vm import (
    RationalEqualityVerifier,
    RestrictedVMExecutor,
    encode_vm_program,
)
from pure_integer_ai.vm.graph_compile import Instruction


_ACCESS = MemoryAccessContext(0, 0, 0)
_PROFILE = TemperatureProfile(
    (53_000, 1),
    (
        TemperatureTier((53_000, 1), 0),
        TemperatureTier((53_000, 2), 1),
    ),
)


@dataclass(frozen=True)
class CapabilityEvidence:
    """保存真实形式绑定、M-08 Use 和 Core 不变证据。"""

    binding_success_count: int
    use_count: int
    core_before: CanonicalIdentity
    core_after: CanonicalIdentity

    def __post_init__(self) -> None:
        """要求能力被真实执行和使用，且运行期没有改写 Core。"""
        if (type(self.binding_success_count) is not int
                or self.binding_success_count <= 0):
            raise ValueError("F-01 C-02 缺少成功 binding run")
        if type(self.use_count) is not int or self.use_count <= 0:
            raise ValueError("F-01 C-02 缺少 Capability Use")
        if not isinstance(self.core_before, CanonicalIdentity):
            raise TypeError("F-01 C-02 core_before 类型错误")
        if not isinstance(self.core_after, CanonicalIdentity):
            raise TypeError("F-01 C-02 core_after 类型错误")
        if self.core_before != self.core_after:
            raise ValueError("F-01 C-02 改写了 Core")


def _source(document_id: int = 1) -> SourceRef:
    """构造带完整非零版本的形式任务来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        53_001,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(2),
            PrimitiveVersion(3),
            CurriculumVersion(4),
        ),
    )


def _instruction(source: SourceRef, *key: int) -> ObjectIdentity:
    """构造 owner/version 对齐的一等最小指令。"""
    return minimal_instruction_identity(
        key, owner=source.owner, versions=source.versions)


def _scope(source: SourceRef, local_id: int = 1) -> Any:
    """构造一次形式调用使用的 query scope。"""
    session = session_scope(
        1, owner=source.owner, versions=source.versions, source=source)
    document = document_scope(source, parent=session)
    episode = episode_scope(local_id, parent=document)
    return query_scope(local_id, parent=episode)


def _artifact(
        source: SourceRef,
        scope: Any,
        kind: ObjectIdentity,
        schema: ArtifactSchema,
        key: int,
        payload: tuple[int, ...],
        ) -> FormalArtifact:
    """构造完整来源化 typed Artifact。"""
    return FormalArtifact(
        artifact_identity(source, kind, schema, (key,), payload, scope),
        kind,
        schema,
        source,
        payload,
        scope,
    )


def _formal_failures() -> FormalArtifactFailureProtocol:
    """冻结 S-06 的十一种一等失败原因。"""
    reasons = tuple(minimal_instruction_identity((53_060, item))
                    for item in range(1, 12))
    return FormalArtifactFailureProtocol(*reasons)


def _formal_case(
        *,
        left_payload: tuple[int, ...] = (2, 1),
        right_payload: tuple[int, ...] = (3, 1),
        expected_payload: tuple[int, ...] = (5, 1),
        ) -> dict[str, Any]:
    """构造双参数加法 Artifact definition、调用和独立 verifier。"""
    source = _source()
    scope = _scope(source)
    number_type = concept_identity((53_010, 1))
    program_type = concept_identity((53_010, 2))
    proof_type = concept_identity((53_010, 3))
    unit = concept_identity((53_011, 1))
    unitless = concept_identity((53_011, 2))
    program_kind = concept_identity((53_012, 1))
    value_kind = concept_identity((53_012, 2))
    proof_kind = concept_identity((53_012, 3))
    executor_authority = ArtifactAuthority(
        concept_identity((53_013, 1)), concept_identity((53_013, 2)))
    verifier_authority = ArtifactAuthority(
        concept_identity((53_014, 1)), concept_identity((53_014, 2)))
    binder = binder_identity(source, (53_020, 1))
    left_var = variable_identity(binder, (53_020, 2), number_type)
    right_var = variable_identity(binder, (53_020, 3), number_type)
    number_schema = ArtifactSchema(number_type, unit)
    program_schema = ArtifactSchema(program_type, unitless)
    proof_schema = ArtifactSchema(proof_type, unitless)
    program = _artifact(
        source,
        None,
        program_kind,
        program_schema,
        1,
        encode_vm_program((
            Instruction(OPCODE_LOAD, (101,)),
            Instruction(OPCODE_LOAD, (102,)),
            Instruction(OPCODE_ADD),
            Instruction(OPCODE_HALT),
        )),
    )
    definition = FormalArtifactDefinition(
        program,
        (
            ArtifactParameter(left_var, number_schema, (101,)),
            ArtifactParameter(right_var, number_schema, (102,)),
        ),
        value_kind,
        number_schema,
        proof_kind,
        proof_schema,
        executor_authority,
        verifier_authority,
    )
    left = _artifact(source, scope, value_kind, number_schema, 2, left_payload)
    right = _artifact(source, scope, value_kind, number_schema, 3, right_payload)
    expected = _artifact(
        source, scope, value_kind, number_schema, 4, expected_payload)
    invocation = ArtifactInvocation(
        proposition_identity(source, (53_030, 1)),
        definition,
        (
            ArtifactArgument(left_var, left),
            ArtifactArgument(right_var, right),
        ),
        source,
        scope,
        (53_040, 1),
        expected,
    )
    return {
        "source": source,
        "scope": scope,
        "number_type": number_type,
        "definition": definition,
        "invocation": invocation,
        "executor": RestrictedVMExecutor(
            executor_authority,
            minimal_instruction_identity((53_050, 1)),
            100,
        ),
        "verifier": RationalEqualityVerifier(
            verifier_authority,
            minimal_instruction_identity((53_050, 2)),
            minimal_instruction_identity((53_050, 3)),
        ),
    }


def _bridge(case: dict[str, Any]) -> FormalArtifactBridge:
    """用精确类型/unit resolver 装配真实 S-06 bridge。"""
    return FormalArtifactBridge(
        ExactArtifactCompatibilityResolver(),
        ExactArtifactCompatibilityResolver(),
        case["executor"],
        case["verifier"],
        _formal_failures(),
    )


def _binding_failures() -> BindingFailureProtocol:
    """冻结 STRUCT_BIND adapter 的九种一等失败原因。"""
    return BindingFailureProtocol(*tuple(
        minimal_instruction_identity((53_070, item))
        for item in range(1, 10)
    ))


def _begin_query(ctx: Any, source: SourceRef, local_id: int = 1) -> Any:
    """打开 session/document/episode/query 完整生命周期。"""
    session = session_scope(
        1, owner=source.owner, versions=source.versions, source=source)
    document = document_scope(source, parent=session)
    episode = episode_scope(local_id, parent=document)
    query = query_scope(local_id, parent=episode)
    ctx.work_memory.begin_session(session)
    ctx.work_memory.begin_document(document)
    ctx.work_memory.begin_episode(episode)
    ctx.work_memory.begin_query(query)
    return query


def _end_query(ctx: Any) -> None:
    """逆序关闭所有 WorkMemory 生命周期。"""
    work = ctx.work_memory
    if work.active_query_scope is not None:
        work.end_query()
    if work.active_episode_scope is not None:
        work.end_episode()
    if work.active_document_scope is not None:
        work.end_document()
    if work.active_session_scope is not None:
        work.end_session()


def _add_edge(ctx: Any, source_slot: tuple[int, int],
              target_slot: tuple[int, int], *, order_index: int) -> None:
    """写入一条来源化 STRUCT_BIND 边供 A-06 运行期重读。"""
    ctx.edge_store.add(
        space_id_from=source_slot[0],
        local_id_from=source_slot[1],
        space_id_to=target_slot[0],
        local_id_to=target_slot[1],
        edge_type=EDGE_STRUCT_BIND,
        strength=1,
        source=53_080,
        tier=TIER_PRIMARY,
        epistemic_origin=EPI_STRUCTURED,
        order_index=order_index,
    )


def _binding_case(
        *, left: int = 2, right: int = 3, total: int = 5,
        ) -> tuple[Any, Any, dict[str, Any], ArtifactBindingRuntime,
                   ArtifactBindingRequest]:
    """装配真实边、显式选择、typed adapter 和 S-06 执行链。"""
    backend = DictBackend()
    ctx = make_train_context(backend, companion=True)
    formal = _formal_case(
        left_payload=(left, 1),
        right_payload=(right, 1),
        expected_payload=(total, 1),
    )
    query = _begin_query(ctx, formal["source"])
    definition = formal["definition"]
    source_binder = binder_identity(formal["source"], (53_090, 1))
    source_variables = (
        variable_identity(source_binder, (53_090, 2), formal["number_type"]),
        variable_identity(source_binder, (53_090, 3), formal["number_type"]),
    )
    source_slots = ((ctx.space_id, 53_091), (ctx.space_id, 53_092))
    target_slots = ((ctx.space_id, 53_093), (ctx.space_id, 53_094))
    for ordinal, (source_slot, target_slot) in enumerate(
            zip(source_slots, target_slots), start=1):
        _add_edge(ctx, source_slot, target_slot, order_index=10 - ordinal)
    endpoints = (
        ArtifactBindingEndpoint(source_slots[0], source_variables[0]),
        ArtifactBindingEndpoint(target_slots[0], definition.parameters[0].variable),
        ArtifactBindingEndpoint(source_slots[1], source_variables[1]),
        ArtifactBindingEndpoint(target_slots[1], definition.parameters[1].variable),
    )
    adapter = StructBindTypedAdapter(
        ExactTypeCompatibilityResolver(), _binding_failures())
    typed_endpoints = tuple(
        TypedStructBindEndpoint(item.slot_ref, item.variable)
        for item in endpoints
    )
    correspondences = tuple(
        adapter.read_from(
            ctx.edge_store, source_slot, typed_endpoints).correspondences[0]
        for source_slot in source_slots
    )
    reason = minimal_instruction_identity((53_095, 1))
    choices = tuple(
        ArtifactBindingChoice(
            endpoints[index * 2],
            endpoints[index * 2 + 1],
            correspondences[index].stable_key(),
            reason,
            (53_096, index + 1),
        )
        for index in range(2)
    )
    values = tuple(
        ArtifactBindingValue(
            endpoints[index * 2],
            formal["invocation"].arguments[index].value,
        )
        for index in range(2)
    )
    request = ArtifactBindingRequest(
        formal["invocation"].proposition,
        definition,
        formal["source"],
        query,
        (53_097, 1),
        endpoints,
        values,
        choices,
        formal["invocation"].expected,
    )
    runtime = ArtifactBindingRuntime(
        ctx.edge_store, adapter, _bridge(formal), ctx.work_memory)
    return backend, ctx, formal, runtime, request


def _successful_example(left: int, right: int, total: int) -> CapabilityExample:
    """执行真实 A-06 调用并冻结一个成功 Capability example。"""
    backend, ctx, _, runtime, request = _binding_case(
        left=left, right=right, total=total)
    try:
        run = runtime.run(request)
        if not run.succeeded:
            raise RuntimeError("F-01 Capability example 形式执行失败")
        return CapabilityExample(run)
    finally:
        _end_query(ctx)
        backend.close()


class _Former:
    """只提出由生产 scenario 构造的 candidate proposal。"""

    def __init__(self, proposal: CapabilityCandidateProposal) -> None:
        self.proposal = proposal

    def propose(self, formation_input: Any) -> CapabilityCandidateProposal:
        """返回 proposal；反召回和契约检查仍由 C-00 runtime 完成。"""
        del formation_input
        return self.proposal


def _candidate() -> Any:
    """从两个真实 A-06 示例形成 provisional Capability candidate。"""
    examples = (
        _successful_example(2, 3, 5),
        _successful_example(4, 6, 10),
    )
    first = examples[0].run.invocation.definition
    source = first.program.source
    bindings = ((201,), (202,))
    program = _artifact(
        source,
        None,
        first.program.artifact_kind,
        first.program.schema,
        53_101,
        encode_vm_program((
            Instruction(OPCODE_LOAD, bindings[0]),
            Instruction(OPCODE_LOAD, bindings[1]),
            Instruction(OPCODE_ADD),
            Instruction(OPCODE_HALT),
        )),
    )
    binder = binder_identity(source, (53_102, 1))
    parameters = tuple(
        replace(
            parameter,
            variable=variable_identity(
                binder,
                (53_102, index + 2),
                parameter.schema.value_type,
            ),
            executor_binding=bindings[index],
        )
        for index, parameter in enumerate(first.parameters)
    )
    definition = replace(first, program=program, parameters=parameters)
    proposal = CapabilityCandidateProposal(
        MemoryLinkedRef.object(concept_identity((53_103, 1))),
        definition,
        source,
        (proposition_identity(source, (53_104, 1)),),
        tuple(item.content_ref() for item in examples),
        minimal_instruction_identity((53_105, 1)),
        (53_106, 1, len(examples)),
    )
    status = CapabilityStatusProtocol(*tuple(
        minimal_instruction_identity((53_107, item))
        for item in range(1, 4)
    ))
    return CapabilityFormationRuntime(_Former(proposal)).form(
        CapabilityFormationRequest(examples, (), status))


def _verified_report() -> Any:
    """对未见参数执行 C-01 held-out 并返回 verified report。"""
    candidate = _candidate()
    source = _source(document_id=53_110)
    scope = _scope(source, local_id=53_110)
    definition = candidate.proposal.definition
    value_kind = candidate.examples[0].run.invocation.arguments[0].value.artifact_kind
    held_out = CapabilityHeldOutCase(
        source,
        scope,
        proposition_identity(source, (53_111, 1)),
        (
            _artifact(
                source, scope, value_kind,
                definition.parameters[0].schema, 53_112, (7, 1)),
            _artifact(
                source, scope, value_kind,
                definition.parameters[1].schema, 53_113, (9, 1)),
        ),
        _artifact(
            source, scope, definition.result_kind,
            definition.result_schema, 53_114, (16, 1)),
        (proposition_identity(source, (53_115, 1)),),
        ArtifactAuthority(
            concept_identity((53_116, 1)),
            concept_identity((53_116, 2)),
        ),
        _instruction(source, 53_117, 1),
        (53_118, 1),
        (53_119, 1),
    )
    return CapabilityVerificationRuntime(_bridge(_formal_case())).verify(
        candidate, held_out)


class _EmptyBaseline:
    """Capability query 不注入额外 Core 候选。"""

    def candidates(self, request: Any) -> tuple[Any, ...]:
        del request
        return ()

    def state_key(self) -> tuple[int, ...]:
        return 1, 53_120


class _OpenFilter:
    """提供一个无附加约束的 Hypothesis 索引分支。"""

    def filters(self, request: Any) -> tuple[MemoryAggregateFilter, ...]:
        del request
        return (MemoryAggregateFilter(),)

    def state_key(self) -> tuple[int, ...]:
        return 1, 53_121


class _ZeroScore:
    """默认 Hypothesis 路径的纯整数评分器，本场景不消费该路径。"""

    def score(self, request: Any, hypothesis: Any,
              aggregate: Any, sources: Any) -> ActivationScore:
        del request, hypothesis, aggregate, sources
        return ActivationScore(0, (ActivationScoreReason((53_122, 1), 0),))

    def state_key(self) -> tuple[int, ...]:
        return 1, 53_122


class _CapabilityScore:
    """只按当前 Capability 自己的使用统计形成确定性整数分。"""

    def assess(self, request: Any, capability: Any,
               usage: Any) -> CapabilityActivationAssessment:
        del capability
        value = 100 + usage.use_count * 10 - usage.failure_count * 20
        return CapabilityActivationAssessment(
            request.hypothesis_kind,
            ActivationScore(
                value,
                (ActivationScoreReason((53_123, 1), value),),
            ),
        )

    def state_key(self) -> tuple[int, ...]:
        return 1, 53_123


class _EmptyMapper:
    """默认 A-10 mapper 不投影非 Capability 候选。"""

    def project(self, request: Any, candidate: Any,
                obligations: Any) -> tuple[Any, ...]:
        del request, candidate, obligations
        return ()

    def clone_for_context(self, ctx: Any) -> "_EmptyMapper":
        del ctx
        return _EmptyMapper()

    def state_key(self) -> tuple[int, ...]:
        return 1, 53_124


class _UnusedRecompute:
    """本场景没有后文更新，任何重算调用都表示边界漂移。"""

    def recompute(self, activation: Any, update: Any) -> Any:
        del activation, update
        raise RuntimeError("F-01 Capability 场景不应触发 A-10 重算")

    def clone_for_context(self, ctx: Any) -> "_UnusedRecompute":
        del ctx
        return _UnusedRecompute()

    def state_key(self) -> tuple[int, ...]:
        return 1, 53_125


class _FirstObligation:
    """把真实 Capability 显式关联到调用方第一个当前义务。"""

    def select(self, request: Any, candidate: Any,
               obligations: Any) -> tuple[CapabilityObligationProjection, ...]:
        del request, candidate
        return (CapabilityObligationProjection(obligations[0], 0),)

    def state_key(self) -> tuple[int, ...]:
        return 1, 53_126


def _query_protocol(source: SourceRef, capability_kind: tuple[int, ...]
                    ) -> MemoryQueryProtocol:
    """构造只召回目标 Capability object kind 的 M-06 协议。"""
    roles = MemoryQueryRoles(*tuple(
        _instruction(source, 53_130, item) for item in range(1, 9)
    ))
    return MemoryQueryProtocol(
        roles,
        (MemoryQueryDefinition(
            _instruction(source, 53_131, 1),
            capability_kind,
            (roles.occurrence, roles.domain),
            1,
            MEMORY_OBJECT_CAPABILITY,
        ),),
    )


def _current(ctx: Any, source: SourceRef, scope: Any) -> MemoryCurrentQuery:
    """物化当前 typed 输入并构造 M-06 query。"""
    ontology = ctx.graph_ontology
    occurrence = ontology.materialize(occurrence_identity(
        source, start=0, end=1, ordinal=0))
    span = ontology.materialize(span_identity(
        source, members=((0, 1),), ordinal=0))
    semantic = ontology.materialize(proposition_identity(source, (53_132, 1)))
    structure = ontology.materialize(structure_concept_identity(
        (53_133, 1), owner=source.owner, versions=source.versions))
    timestamp = LogicalClock(
        LogicalClockIdentity(scope, CLOCK_QUERY)).advance()
    return MemoryCurrentQuery(
        scope,
        source,
        timestamp,
        (occurrence,),
        (span,),
        (semantic,),
        (structure,),
        concept_identity((53_134, 1), owner=source.owner,
                         versions=source.versions),
        concept_identity((53_134, 2), owner=source.owner,
                         versions=source.versions),
    )


def _goal(source: SourceRef, scope: Any) -> ReasoningObligation:
    """构造属于当前 query 且无预置答案的 typed 义务。"""
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (53_135, 1)),
        concept_identity((53_135, 2), owner=source.owner,
                         versions=source.versions),
        occurrence_identity(source, start=1, end=2, ordinal=1),
        context_scope_identity(source, (53_135, 3)),
        (),
    )
    template = ScopedPropositionTemplate(
        definition,
        structure_concept_identity(
            (53_135, 4), owner=source.owner, versions=source.versions),
    )
    graph = PropositionTemplateGraph((template,))
    failures = BindingFailureProtocol(*tuple(
        _instruction(source, 53_136, item) for item in range(1, 10)
    ))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        _instruction(source, 53_137, 1), failures))
    return ReasoningObligation(
        substituter.substitute(
            definition.proposition, graph, BindingEnvironment()),
        LogicEvidenceState(True, False),
        source,
        scope,
    )


def _core_refs(ctx: Any) -> tuple[Any, ...]:
    """在 Core 冻结前物化 Observation 所需的五种端点。"""
    ontology = ctx.graph_ontology
    return (
        ontology.materialize(concept_identity((53_140, 1))),
        ontology.materialize(concept_identity((53_140, 2))),
        ontology.materialize(concept_identity((53_140, 3))),
        ontology.materialize(concept_identity((53_140, 4))),
        ontology.materialize(relation_concept_identity((53_140, 5))),
    )


def _append_observation(ctx: Any, source: SourceRef,
                        refs: tuple[Any, ...]) -> Any:
    """向 interact Memory 追加一条字段齐全的真实 Observation。"""
    episode = episode_scope(1, parent=document_scope(source))
    context, concept, structure, proposition, relation = refs
    observed_at = LogicalClock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_OBSERVED)).advance()
    payload = ObservationPayload(
        source,
        MemoryLinkedRef.core(context),
        (concept,),
        (concept, structure),
        structure,
        (proposition,),
        (MemoryLinkedRef.core(relation),),
        observed_at,
    )
    ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_OBSERVATION,
        payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    return ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_OBSERVATION, ref, episode, payload))


def _batch_config() -> MemoryBatchRuntimeConfig:
    """构造 C-02 发布使用的 M-10/K-02 配置。"""
    return MemoryBatchRuntimeConfig(
        _PROFILE,
        (53_000, 1),
        SegmentDependency(
            MEMORY_BATCH_CORE_DEPENDENCY_KEY,
            (53_141, 1),
            (53_141, 2),
        ),
        SegmentBudget(8, 1_000_000),
        SegmentBudget(64, 2_000_000),
    )


def _source_dependency(batch_id: int) -> SegmentDependency:
    """绑定 Capability Memory 发布的来源批次依赖。"""
    return SegmentDependency(
        MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
        (53_142, batch_id),
        (53_143, batch_id),
    )


def _created_at(source: SourceRef) -> LogicalTimestamp:
    """形成 candidate 来源 episode 内的创建逻辑时间。"""
    episode = episode_scope(53_144, parent=document_scope(source))
    return LogicalClock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_CREATED)).advance()


def _candidate_binding_request(
        runtime: ArtifactBindingRuntime,
        original: ArtifactBindingRequest,
        definition: FormalArtifactDefinition,
        proposition: ObjectIdentity,
        scope: Any,
        ) -> ArtifactBindingRequest:
    """用真实 STRUCT_BIND 槽位为 recovered Capability 重建 A-06 请求。"""
    source_endpoints = (original.endpoints[0], original.endpoints[2])
    target_endpoints = tuple(
        ArtifactBindingEndpoint(
            original.endpoints[index * 2 + 1].slot_ref,
            parameter.variable,
        )
        for index, parameter in enumerate(definition.parameters)
    )
    endpoints = tuple(
        item
        for pair in zip(source_endpoints, target_endpoints)
        for item in pair
    )
    typed_endpoints = tuple(
        TypedStructBindEndpoint(item.slot_ref, item.variable)
        for item in endpoints
    )
    correspondences = tuple(
        runtime.adapter.read_from(
            runtime.edge_store,
            source.slot_ref,
            typed_endpoints,
        ).correspondences[0]
        for source in source_endpoints
    )
    choices = tuple(
        ArtifactBindingChoice(
            source_endpoints[index],
            target_endpoints[index],
            correspondences[index].stable_key(),
            original.choices[index].reason,
            (53_145, index + 1),
        )
        for index in range(len(target_endpoints))
    )
    values = tuple(
        ArtifactBindingValue(
            source_endpoints[index],
            _artifact(
                original.source,
                scope,
                value.artifact.artifact_kind,
                value.artifact.schema,
                53_146 + index,
                value.artifact.payload,
            ),
        )
        for index, value in enumerate(original.values)
    )
    expected_source = original.expected
    expected = None if expected_source is None else _artifact(
        original.source,
        scope,
        expected_source.artifact_kind,
        expected_source.schema,
        53_149,
        expected_source.payload,
    )
    return ArtifactBindingRequest(
        proposition,
        definition,
        original.source,
        scope,
        (53_150, 1),
        endpoints,
        values,
        choices,
        expected,
    )


def run_capability_evidence() -> CapabilityEvidence:
    """运行 verified Capability 发布、召回、绑定和唯一 Use 闭环。"""
    report = _verified_report()
    backend, ctx, formal, binding_runtime, original = _binding_case()
    del formal
    try:
        source = report.candidate.proposal.source
        if source != original.source:
            raise RuntimeError("F-01 Capability 来源与 A-06 上下文漂移")
        refs = _core_refs(ctx)
        query_runtime = install_memory_query_runtime(
            ctx,
            _query_protocol(
                source,
                report.candidate.proposal.capability_kind.stable_key(),
            ),
            aggregates=ctx.memory_interact_aggregates,
        )
        resolver = MemoryOverlayResolver(
            ctx.memory_interact_aggregates,
            ctx.core_identity_catalog,
            _EmptyBaseline(),
            _OpenFilter(),
            _ZeroScore(),
            StableTopKSourcePolicy(),
        )
        resolver_runtime = install_memory_resolver_runtime(ctx, resolver)
        install_memory_batch_runtimes(ctx, _batch_config())
        capability_route = CapabilityResolverRoute(
            ctx.memory_interact_events, _CapabilityScore())
        resolver_runtime.register_route(capability_route)
        protocol = AttractorProtocol(*tuple(
            _instruction(source, 53_151, item) for item in range(1, 5)
        ))
        attractor = install_attractor_runtime(
            ctx,
            protocol,
            AttractorBudget(1, 1, 0),
            _EmptyMapper(),
            _UnusedRecompute(),
        )
        attractor.register_mapper_route(CapabilityActivationMapper(
            _FirstObligation(),
            activation_kind=_instruction(source, 53_152, 1),
            score_reason=_instruction(source, 53_152, 2),
            capability_dependency_role=_instruction(source, 53_152, 3),
        ))
        current = _current(ctx, source, original.scope)
        compilation = query_runtime.compile(current, access=_ACCESS)
        goal = _goal(source, original.scope)
        core_reader = CoreCanonicalStateReader(ctx)
        core_before = CanonicalIdentity.from_value(core_reader.read())
        published = CapabilityMemoryRuntime(ctx).publish_verified(
            report,
            batch_id=53_153,
            source_dependency=_source_dependency(53_153),
            created_at=_created_at(source),
        )
        observation = _append_observation(ctx, source, refs)
        state = attractor.resolve_and_activate(compilation, (goal,))
        request = _candidate_binding_request(
            binding_runtime,
            original,
            report.candidate.proposal.definition,
            goal.proposition.template,
            original.scope,
        )
        memory_use = install_memory_use_runtime(ctx)
        execution = CapabilityExecutionRuntime(
            ctx,
            binding_runtime,
            memory_use,
            CapabilityExecutionProtocol(
                _instruction(source, 53_154, 1),
                MemoryLinkedRef.object(concept_identity(
                    (53_154, 2), owner=source.owner,
                    versions=source.versions)),
                MemoryLinkedRef.object(concept_identity(
                    (53_154, 3), owner=source.owner,
                    versions=source.versions)),
            ),
        )
        result = execution.execute_frontier(
            request,
            input_observation_ref=observation.event.object_ref,
            used_at=LogicalTimestamp(state.current_timestamp.clock, 2),
            failed_at=LogicalTimestamp(
                LogicalClockIdentity(
                    published.recovered.capability_event.event.scope,
                    CLOCK_MEMORY_USED,
                ),
                1,
            ),
        )
        core_after = CanonicalIdentity.from_value(core_reader.read())
        uses = ctx.memory_interact_events.query(
            access=_ACCESS, event_kind=MEMORY_EVENT_USE)
        return CapabilityEvidence(
            int(result.binding_run.succeeded),
            len(uses),
            core_before,
            core_after,
        )
    finally:
        _end_query(ctx)
        backend.close()


__all__ = ["CapabilityEvidence", "run_capability_evidence"]
