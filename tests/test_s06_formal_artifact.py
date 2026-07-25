"""S-06 typed Artifact、形式域桥和 WorkMemory 生命周期对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactArgument,
    ArtifactAuthority,
    ArtifactCompatibilityResult,
    ArtifactInvocation,
    ArtifactParameter,
    ArtifactSchema,
    ExactArtifactCompatibilityResolver,
    FormalArtifact,
    FormalArtifactDefinition,
    artifact_identity,
    describe_artifact_identity,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactVerificationObservation,
    FormalArtifactBridge,
    FormalArtifactFailureProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import EPISTEMIC_UNKNOWN
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
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    binder_identity,
    proposition_identity,
    variable_identity,
)
from pure_integer_ai.cognition.shared.work_memory import (
    WorkMemory,
    WorkMemoryScopeError,
)
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_ADD,
    OPCODE_HALT,
    OPCODE_JMP,
    OPCODE_LOAD,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.training.formal_artifact_vm import (
    RationalEqualityVerifier,
    RestrictedVMExecutor,
    decode_vm_program,
    encode_vm_program,
)
from pure_integer_ai.vm.graph_compile import Instruction


def _source(document_id: int = 1) -> SourceRef:
    """构造带完整课程版本的形式任务来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        9901,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(2),
            PrimitiveVersion(3),
            CurriculumVersion(4),
        ),
    )


def _scopes(source: SourceRef, query_id: int = 1):
    """构造 WorkMemory 和调用共享的 session/document/episode/query 边界。"""
    session = session_scope(1, versions=source.versions)
    document = document_scope(source)
    episode = episode_scope(1, parent=document)
    query = query_scope(query_id, parent=episode)
    return session, document, episode, query


def _artifact(
        source: SourceRef,
        scope,
        kind: ObjectIdentity,
        schema: ArtifactSchema,
        key: int,
        payload: tuple[int, ...],
        ) -> FormalArtifact:
    """按测试声明键构造完整来源化 Artifact。"""
    return FormalArtifact(
        artifact_identity(source, kind, schema, (key,), payload, scope),
        kind,
        schema,
        source,
        payload,
        scope,
    )


def _failures() -> FormalArtifactFailureProtocol:
    """注入互异 MinimalInstruction，避免依赖异常文字判断失败类型。"""
    reasons = [minimal_instruction_identity((9960, index))
               for index in range(1, 12)]
    return FormalArtifactFailureProtocol(*reasons)


class _UnitResolver:
    """按测试注入的 unit 身份返回换算、unknown 或明确拒绝。"""

    def __init__(self, expected, convertible, rejected, support) -> None:
        self.expected = expected
        self.convertible = convertible
        self.rejected = rejected
        self.support = support

    def resolve(self, expected, actual):
        """不同 unit 的换算因子由 resolver 提供，不由 bridge 写死。"""
        if expected != self.expected:
            return ArtifactCompatibilityResult(expected, actual, None)
        if actual == expected:
            return ArtifactCompatibilityResult(expected, actual, True)
        if actual == self.convertible:
            return ArtifactCompatibilityResult(
                expected, actual, True, (self.support,), (1, 100))
        if actual == self.rejected:
            return ArtifactCompatibilityResult(expected, actual, False)
        return ArtifactCompatibilityResult(expected, actual, None)


class _DriftVerifier:
    """只在 verifier observation 边界注入指定字段漂移。"""

    def __init__(self, delegate, *, authority=None, source=None, scope=None):
        self.delegate = delegate
        self.authority = authority
        self.source = source
        self.scope = scope

    def verify(self, request):
        """保留真实比较结果，仅替换审计字段以验证 bridge fail closed。"""
        observation = self.delegate.verify(request)
        return replace(
            observation,
            authority=(self.authority
                       if self.authority is not None
                       else observation.authority),
            source=self.source if self.source is not None else observation.source,
            scope=self.scope if self.scope is not None else observation.scope,
        )


def _case(
        *, left_payload: tuple[int, ...] = (2, 1),
        right_payload: tuple[int, ...] = (3, 1),
        expected_payload: tuple[int, ...] = (5, 1),
        left_type: ObjectIdentity | None = None,
        left_unit: ObjectIdentity | None = None,
        instructions: tuple[Instruction, ...] | None = None,
        step_limit: int = 100,
        ):
    """构造完整双参数加法调用及其图身份、adapter 和失败协议。"""
    source = _source()
    _, _, _, scope = _scopes(source)
    number_type = concept_identity((9910, 1))
    program_type = concept_identity((9910, 2))
    proof_type = concept_identity((9910, 3))
    unit = concept_identity((9911, 1))
    unitless = concept_identity((9911, 2))
    program_kind = concept_identity((9912, 1))
    value_kind = concept_identity((9912, 2))
    proof_kind = concept_identity((9912, 3))
    executor_authority = ArtifactAuthority(
        concept_identity((9913, 1)), concept_identity((9913, 2)))
    verifier_authority = ArtifactAuthority(
        concept_identity((9914, 1)), concept_identity((9914, 2)))

    binder = binder_identity(source, (9920, 1))
    left_var = variable_identity(binder, (9920, 2), number_type)
    right_var = variable_identity(binder, (9920, 3), number_type)
    number_schema = ArtifactSchema(number_type, unit)
    program_schema = ArtifactSchema(program_type, unitless)
    proof_schema = ArtifactSchema(proof_type, unitless)
    if instructions is None:
        instructions = (
            Instruction(OPCODE_LOAD, (101,)),
            Instruction(OPCODE_LOAD, (102,)),
            Instruction(OPCODE_ADD),
            Instruction(OPCODE_HALT),
        )
    program = _artifact(
        source,
        None,
        program_kind,
        program_schema,
        1,
        encode_vm_program(instructions),
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
    left_schema = ArtifactSchema(left_type or number_type, left_unit or unit)
    left = _artifact(
        source, scope, value_kind, left_schema, 2, left_payload)
    right = _artifact(
        source, scope, value_kind, number_schema, 3, right_payload)
    expected = _artifact(
        source, scope, value_kind, number_schema, 4, expected_payload)
    proposition = proposition_identity(source, (9930, 1))
    invocation = ArtifactInvocation(
        proposition,
        definition,
        (
            ArtifactArgument(left_var, left),
            ArtifactArgument(right_var, right),
        ),
        source,
        scope,
        (9940, 1),
        expected,
    )
    vm_failure = minimal_instruction_identity((9950, 1))
    missing_expected = minimal_instruction_identity((9950, 2))
    malformed = minimal_instruction_identity((9950, 3))
    executor = RestrictedVMExecutor(
        executor_authority, vm_failure, step_limit)
    verifier = RationalEqualityVerifier(
        verifier_authority, missing_expected, malformed)
    return {
        "source": source,
        "scope": scope,
        "number_type": number_type,
        "unit": unit,
        "unitless": unitless,
        "definition": definition,
        "invocation": invocation,
        "executor": executor,
        "verifier": verifier,
        "failures": _failures(),
    }


def _bridge(case, *, unit_resolver=None, verifier=None):
    """用保守 type resolver 和调用方选择的 unit/verifier 组装桥。"""
    return FormalArtifactBridge(
        ExactArtifactCompatibilityResolver(),
        unit_resolver or ExactArtifactCompatibilityResolver(),
        case["executor"],
        verifier or case["verifier"],
        case["failures"],
    )


def test_artifact_identity_keeps_source_kind_and_full_key():
    """Artifact 身份可无损恢复 source、kind 和声明键，hash 不是权威身份。"""
    case = _case()
    program = case["definition"].program
    descriptor = describe_artifact_identity(program.identity)
    assert descriptor.source == case["source"]
    assert descriptor.artifact_kind == program.artifact_kind
    assert descriptor.schema == program.schema
    assert descriptor.scope is None
    assert descriptor.declaration_key == (1,)
    assert descriptor.payload == program.payload
    assert ObjectIdentity.from_stable_key(
        program.identity.stable_key()) == program.identity
    with pytest.raises((TypeError, ValueError)):
        ArtifactSchema(case["number_type"], None)


def test_restricted_vm_execution_and_independent_verifier_form_complete_result():
    """语言任务绑定参数后真实执行 VM，并由独立 expected 形成 value/proof Artifact。"""
    case = _case()
    result = _bridge(case).invoke(case["invocation"])

    assert result.succeeded is True
    assert result.value is not None and result.value.payload == (5, 1)
    assert result.proof is not None and result.proof.payload[0] == 1
    assert result.execution is not None
    assert result.execution.authority == case["definition"].executor
    assert result.verification is not None
    assert result.verification.authority == case["definition"].verifier
    assert describe_artifact_identity(result.value.identity).source == case["source"]
    assert ObjectIdentity.from_stable_key(
        result.proof.identity.stable_key()) == result.proof.identity


def test_formal_success_keeps_language_proposition_unknown():
    """VM 和 verifier 成功不能自动把自然语言 Proposition 升为 provisional。"""
    case = _case()
    result = _bridge(case).invoke(case["invocation"])
    assert result.succeeded is True
    assert result.proposition_state.status == EPISTEMIC_UNKNOWN
    assert result.proposition_state.support is False
    assert result.proposition_state.refute is False


def test_parameter_type_unknown_stops_before_execution():
    """参数类型缺兼容依据时 fail closed，VM 不得收到调用。"""
    wrong_type = concept_identity((9910, 99))
    case = _case(left_type=wrong_type)
    result = _bridge(case).invoke(case["invocation"])
    assert result.succeeded is False
    assert result.execution is None
    assert result.failures[0].reason == case["failures"].type_unknown
    assert result.failures[0].expected == case["number_type"]
    assert result.failures[0].actual == wrong_type


def test_missing_and_duplicate_parameters_fail_closed():
    """缺参数或同一 Variable 重复绑定都不能靠位置补齐。"""
    case = _case()
    invocation = case["invocation"]
    missing = replace(invocation, arguments=invocation.arguments[:1])
    duplicate = replace(
        invocation,
        arguments=(invocation.arguments[0], invocation.arguments[0]),
    )
    for candidate in (missing, duplicate):
        result = _bridge(case).invoke(candidate)
        assert result.execution is None
        assert any(item.reason == case["failures"].argument_shape
                   for item in result.failures)


def test_unit_conversion_payload_is_consumed_by_vm_adapter():
    """不同但兼容的 unit 必须先按 resolver 载荷精确换算再执行。"""
    centimetre = concept_identity((9911, 10))
    rejected = concept_identity((9911, 11))
    support = concept_identity((9911, 12))
    case = _case(left_payload=(200, 1), left_unit=centimetre)
    resolver = _UnitResolver(
        case["unit"], centimetre, rejected, support)
    result = _bridge(case, unit_resolver=resolver).invoke(case["invocation"])
    assert result.succeeded is True
    assert result.value is not None and result.value.payload == (5, 1)
    assert result.bound_arguments[0].unit_support == (support,)
    assert result.bound_arguments[0].unit_adapter_payload == (1, 100)


@pytest.mark.parametrize("mode", ["unknown", "rejected"])
def test_unit_unknown_or_rejected_stops_before_execution(mode):
    """unit unknown 与明确不兼容使用不同 reason，二者都不得进入 VM。"""
    convertible = concept_identity((9911, 20))
    rejected = concept_identity((9911, 21))
    unknown = concept_identity((9911, 22))
    support = concept_identity((9911, 23))
    actual = rejected if mode == "rejected" else unknown
    case = _case(left_unit=actual)
    resolver = _UnitResolver(
        case["unit"], convertible, rejected, support)
    result = _bridge(case, unit_resolver=resolver).invoke(case["invocation"])
    assert result.execution is None
    expected_reason = (
        case["failures"].unit_rejected
        if mode == "rejected"
        else case["failures"].unit_unknown)
    assert result.failures[0].reason == expected_reason


@pytest.mark.parametrize("drift", ["identity", "version", "source", "scope"])
def test_verifier_identity_version_source_and_scope_drift_fail_closed(drift):
    """verifier 任一审计坐标漂移都不能生成可信 proof result。"""
    case = _case()
    declared = case["definition"].verifier
    authority = declared
    source = None
    scope = None
    if drift == "identity":
        authority = ArtifactAuthority(
            concept_identity((9914, 99)), declared.version)
    elif drift == "version":
        authority = ArtifactAuthority(
            declared.identity, concept_identity((9914, 98)))
    elif drift == "source":
        source = _source(document_id=2)
    else:
        source_ref = case["source"]
        _, document, episode, _ = _scopes(source_ref)
        scope = query_scope(99, parent=episode)
        assert document == episode.parent
    verifier = _DriftVerifier(
        case["verifier"], authority=authority, source=source, scope=scope)
    result = _bridge(case, verifier=verifier).invoke(case["invocation"])
    assert result.succeeded is False
    assert result.value is not None
    assert result.proof is None
    assert result.failures[0].reason == (
        case["failures"].verifier_contract_drift)


def test_vm_step_limit_and_malformed_program_are_not_success():
    """无限回跳和损坏 program payload 均由受限 executor 显式拒绝。"""
    loop_case = _case(
        instructions=(Instruction(OPCODE_JMP, (0,)),), step_limit=3)
    loop_result = _bridge(loop_case).invoke(loop_case["invocation"])
    assert loop_result.succeeded is False
    assert loop_result.execution is not None
    assert loop_result.execution.executed is False
    assert loop_result.failures[0].reason == (
        loop_case["failures"].executor_rejected)

    case = _case()
    original = case["definition"].program
    malformed_payload = (1, 1, OPCODE_LOAD)
    program = FormalArtifact(
        artifact_identity(
            case["source"],
            original.artifact_kind,
            original.schema,
            (99,),
            malformed_payload,
            None,
        ),
        original.artifact_kind,
        original.schema,
        case["source"],
        malformed_payload,
        None,
    )
    definition = replace(case["definition"], program=program)
    invocation = replace(case["invocation"], definition=definition)
    malformed = _bridge(case).invoke(invocation)
    assert malformed.execution is not None
    assert malformed.execution.executed is False
    with pytest.raises(ValueError):
        decode_vm_program(program.payload)


def test_rejected_expected_produces_formal_proof_but_not_language_evidence():
    """形式值与 expected 不同会保留拒绝 proof，且语言状态仍为 unknown。"""
    case = _case(expected_payload=(6, 1))
    result = _bridge(case).invoke(case["invocation"])
    assert result.succeeded is False
    assert result.value is not None and result.value.payload == (5, 1)
    assert result.proof is not None and result.proof.payload[0] == 2
    assert result.verification is not None
    assert result.verification.accepted is False
    assert result.proposition_state.status == EPISTEMIC_UNKNOWN
    assert result.failures[0].reason == case["failures"].verifier_rejected


def test_work_memory_keeps_episode_artifacts_and_clears_query_trace():
    """query 结束只清调用 trace，当前值到 episode 结束才清理。"""
    case = _case()
    result = _bridge(case).invoke(case["invocation"])
    source = case["source"]
    session, document, episode, query = _scopes(source)
    work_memory = WorkMemory()
    work_memory.begin_session(session)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode)
    work_memory.begin_query(query)
    work_memory.record_artifact_result(result)

    assert len(work_memory.query_artifact_results) == 1
    assert result.value is not None
    assert work_memory.get_episode_artifact(result.value.identity) == result.value
    assert len(work_memory.episode_artifacts) == 5
    work_memory.end_query()
    assert work_memory.query_artifact_results == []
    assert len(work_memory.episode_artifacts) == 5
    work_memory.end_episode()
    assert work_memory.episode_artifacts == {}
    work_memory.end_document()
    work_memory.end_session()


def test_work_memory_rejects_cross_query_and_abort_clears_artifacts():
    """其他 query 的结果不能写入，异常中止也不能把 Artifact 留给下一 episode。"""
    case = _case()
    result = _bridge(case).invoke(case["invocation"])
    source = case["source"]
    session, document, episode, query = _scopes(source)
    work_memory = WorkMemory()
    work_memory.begin_session(session)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode)
    other_query = query_scope(2, parent=episode)
    work_memory.begin_query(other_query)
    with pytest.raises(WorkMemoryScopeError, match="当前 query"):
        work_memory.record_artifact_result(result)
    work_memory.end_query()
    work_memory.begin_query(query)
    work_memory.record_artifact_result(result)
    work_memory.abort_episode()
    assert work_memory.episode_artifacts == {}
    assert work_memory.query_artifact_results == []
    work_memory.end_document()
    work_memory.end_session()
