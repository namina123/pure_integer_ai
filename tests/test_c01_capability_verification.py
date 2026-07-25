"""C-01 独立 held-out、三态晋升和评测隔离测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.capability_candidate import (
    CapabilityFormationRequest,
    CapabilityFormationRuntime,
)
from pure_integer_ai.cognition.shared.capability_verification import (
    CapabilityHeldOutCase,
)
from pure_integer_ai.cognition.shared.formal_artifact import ArtifactAuthority
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactVerificationObservation,
)
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import proposition_identity
from pure_integer_ai.experiments.capability_verification_runtime import (
    CapabilityVerificationRuntime,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.numeric.symbol_domain import OPCODE_LOAD

from test_a06_artifact_binding_runtime import (
    _runtime_case,
    _start_work_memory,
    _stop_work_memory,
)
from test_c00_capability_candidate import (
    _Former,
    _proposal,
    _status_protocol,
    _successful_example,
)
from test_s06_formal_artifact import (
    _DriftVerifier,
    _artifact,
    _bridge,
    _case,
    _scopes,
    _source,
)


def _candidate():
    """从两个真实 A-06 示例形成尚未验证的 C-00 candidate。"""
    examples = (
        _successful_example(2, 3, 5),
        _successful_example(4, 6, 10),
    )
    proposal = _proposal(examples)
    return CapabilityFormationRuntime(_Former(proposal)).form(
        CapabilityFormationRequest(examples, (), _status_protocol()))


def _held_out(
        candidate,
        *,
        left=7,
        right=9,
        expected=16,
        document_id=20801,
        source=None,
        scope=None,
        specification_authority=None,
        ):
    """构造执行前冻结且与 examples 分源的双参数 held-out case。"""
    source = source or _source(document_id=document_id)
    if scope is None:
        _, _, _, scope = _scopes(source, query_id=document_id)
    definition = candidate.proposal.definition
    value_kind = candidate.examples[0].run.invocation.arguments[0].value.artifact_kind
    arguments = (
        _artifact(
            source, scope, value_kind,
            definition.parameters[0].schema, 20810, (left, 1)),
        _artifact(
            source, scope, value_kind,
            definition.parameters[1].schema, 20811, (right, 1)),
    )
    expected_artifact = _artifact(
        source, scope, definition.result_kind,
        definition.result_schema, 20812, (expected, 1))
    authority = specification_authority or ArtifactAuthority(
        concept_identity((20820, 1)),
        concept_identity((20820, 2)),
    )
    return CapabilityHeldOutCase(
        source,
        scope,
        proposition_identity(source, (20830, 1)),
        arguments,
        expected_artifact,
        (proposition_identity(source, (20831, 1)),),
        authority,
        minimal_instruction_identity(
            (20832, 1), owner=source.owner, versions=source.versions),
        (20833, 1),
        (20834, document_id),
    )


class _UnknownVerifier:
    """以正确 authority 和归属返回 unknown 的独立 verifier。"""

    def __init__(self, authority, reason):
        """绑定声明 authority 和注入式 unknown reason。"""
        self.authority = authority
        self.reason = reason

    def verify(self, request):
        """保留当前 source/scope，明确返回不可判定而不是拒绝。"""
        return ArtifactVerificationObservation(
            self.authority,
            request.invocation.source,
            request.invocation.scope,
            None,
            (),
            (),
            self.reason,
        )


def test_c01_unseen_parameters_promote_only_after_real_execution():
    """未见参数必须由 candidate program 执行并经独立 verifier 后晋升。"""
    candidate = _candidate()
    held_out = _held_out(candidate)

    report = CapabilityVerificationRuntime(_bridge(_case())).verify(
        candidate, held_out)

    assert report.verified is True
    assert report.previous_state == candidate.status_protocol.provisional
    assert report.state == candidate.status_protocol.verified
    assert report.result.succeeded is True
    assert report.result.value is not None
    assert report.result.value.payload == (16, 1)
    assert report.invocation.definition == candidate.proposal.definition
    assert report.invocation.expected == held_out.expected
    assert report.stable_key()


def test_c01_wrong_result_rejects_but_unknown_verifier_stays_provisional():
    """明确 wrong result 才 rejected，verifier unknown 不得伪晋升或误杀。"""
    candidate = _candidate()
    rejected = CapabilityVerificationRuntime(_bridge(_case())).verify(
        candidate, _held_out(candidate, expected=17))
    assert rejected.result.verification is not None
    assert rejected.result.verification.accepted is False
    assert rejected.state == candidate.status_protocol.rejected

    formal = _case()
    unknown_verifier = _UnknownVerifier(
        candidate.proposal.definition.verifier,
        minimal_instruction_identity((20840, 1)),
    )
    unknown = CapabilityVerificationRuntime(
        _bridge(formal, verifier=unknown_verifier)).verify(
            candidate, _held_out(candidate, document_id=20802))
    assert unknown.result.verification is not None
    assert unknown.result.verification.accepted is None
    assert unknown.state == candidate.status_protocol.provisional
    assert unknown.verified is False

    drift_authority = ArtifactAuthority(
        concept_identity((20841, 1)),
        concept_identity((20841, 2)),
    )
    drift = CapabilityVerificationRuntime(
        _bridge(
            formal,
            verifier=_DriftVerifier(
                formal["verifier"], authority=drift_authority),
        )).verify(candidate, _held_out(candidate, document_id=20806))
    assert drift.result.verification is not None
    assert drift.result.verification.accepted is True
    assert drift.result.succeeded is False
    assert drift.state == candidate.status_protocol.provisional


def test_c01_candidate_execution_failure_is_rejected():
    """candidate program 已进入 executor 后明确失败应 rejected，而非长期 pending。"""
    base = _candidate()
    proposal = _proposal(base.examples)
    definition = proposal.definition
    malformed_program = _artifact(
        definition.program.source,
        None,
        definition.program.artifact_kind,
        definition.program.schema,
        20842,
        (1, 1, OPCODE_LOAD),
    )
    malformed_proposal = replace(
        proposal,
        definition=replace(definition, program=malformed_program),
    )
    candidate = CapabilityFormationRuntime(_Former(malformed_proposal)).form(
        CapabilityFormationRequest(
            base.examples, (), base.status_protocol))

    report = CapabilityVerificationRuntime(_bridge(_case())).verify(
        candidate, _held_out(candidate, document_id=20807))

    assert report.result.execution is not None
    assert report.result.execution.executed is False
    assert report.state == candidate.status_protocol.rejected


def test_c01_rejects_example_replay_even_when_source_and_declaration_change():
    """换 held-out 来源、scope 和 Artifact 声明不能掩盖相同参数内容重放。"""
    candidate = _candidate()
    replay = _held_out(candidate, left=2, right=3, expected=5)

    with pytest.raises(ValueError, match="重放了 C-00 example"):
        CapabilityVerificationRuntime(_bridge(_case())).verify(
            candidate, replay)


def test_c01_requires_independent_specification_and_exact_contract():
    """规格 authority 或 typed 参数契约不独立时必须在 executor 前拒绝。"""
    candidate = _candidate()
    definition = candidate.proposal.definition
    dependent = _held_out(
        candidate,
        specification_authority=definition.verifier,
    )
    with pytest.raises(ValueError, match="specification authority 必须独立"):
        CapabilityVerificationRuntime(_bridge(_case())).verify(
            candidate, dependent)

    held_out = _held_out(candidate, document_id=20803)
    wrong_schema = candidate.examples[0].run.invocation.definition.program.schema
    wrong_value = _artifact(
        held_out.source,
        held_out.scope,
        held_out.arguments[0].artifact_kind,
        wrong_schema,
        20850,
        held_out.arguments[0].payload,
    )
    drift = replace(
        held_out,
        arguments=(wrong_value, held_out.arguments[1]),
    )
    with pytest.raises(ValueError, match="参数 schema"):
        CapabilityVerificationRuntime(_bridge(_case())).verify(
            candidate, drift)


def test_c01_rejects_prior_candidate_result_as_student_expected():
    """前一次 candidate result Artifact 不能回流充当下一次 held-out expected。"""
    candidate = _candidate()
    runtime = CapabilityVerificationRuntime(_bridge(_case()))
    held_out = _held_out(candidate, document_id=20804)
    first = runtime.verify(candidate, held_out)
    assert first.result.value is not None
    leaked = replace(
        held_out,
        expected=first.result.value,
        case_key=(20835, 1),
    )

    with pytest.raises(ValueError, match="不能回流"):
        runtime.verify(candidate, leaked)


def test_c01_v06_executes_in_clone_without_host_write():
    """V-06 clone 内可执行 held-out，宿主 backend 和 WorkMemory 保持不变。"""
    candidate = _candidate()
    backend, context, formal, _, _ = _runtime_case()
    try:
        host_backend = backend.snapshot()
        host_artifacts = dict(context.work_memory.episode_artifacts)
        host_results = list(context.work_memory.query_artifact_results)

        with isolated_evaluation(context, label="c01-capability-held-out") as clone:
            clone.work_memory.end_session()
            eval_source = SourceRef(
                formal["source"].source_kind,
                formal["source"].source_id,
                20805,
                clone.scope_owner,
                formal["source"].versions,
            )
            _, _, _, query = _start_work_memory(clone, eval_source)
            held_out = _held_out(
                candidate,
                source=eval_source,
                scope=query,
                document_id=20805,
            )

            report = CapabilityVerificationRuntime(_bridge(_case())).verify(
                candidate, held_out)

            assert report.verified is True
            assert report.invocation.source.owner == clone.scope_owner
            assert clone.work_memory.query_artifact_results == []
            _stop_work_memory(clone)

        assert backend.snapshot() == host_backend
        assert context.work_memory.episode_artifacts == host_artifacts
        assert context.work_memory.query_artifact_results == host_results
    finally:
        _stop_work_memory(context)
        backend.close()
