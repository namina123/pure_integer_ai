"""S-06 语言命题到形式域 executor/verifier 的 typed Artifact 桥。

桥只负责参数匹配、type/unit 三态兼容、authority/source/scope/version 核验和审计
结果组装。具体 payload、程序语义和验证算法由注入 adapter 实现；形式验证成功不会
自动成为自然语言 Proposition 的支持证据。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactArgument,
    ArtifactAuthority,
    ArtifactCompatibilityResolver,
    ArtifactCompatibilityResult,
    ArtifactInvocation,
    ArtifactParameter,
    FormalArtifact,
    artifact_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_RESULT_IDENTITY_VERSION = 1
_PROOF_IDENTITY_VERSION = 1
_PROOF_ACCEPTED = 1
_PROOF_REJECTED = 2
_PROOF_UNKNOWN = 3


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给 trace 中可变长身份段增加长度，避免整数串拼接歧义。"""
    return len(key), *key


def _strict_tuple(
        value: tuple[int, ...], *, label: str, allow_empty: bool = True,
        ) -> tuple[int, ...]:
    """校验 adapter 返回的纯整数 payload 或 trace。"""
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise ValueError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_reason(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验失败 reason 是调用方注入的一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    if ObjectIdentity.from_stable_key(identity.stable_key()) != identity:
        raise ValueError(f"{label} 完整身份不能稳定 round-trip")
    return identity


@dataclass(frozen=True)
class FormalArtifactFailureProtocol:
    """为各 fail-closed 边界注入互异的一等失败原因。"""

    argument_shape: ObjectIdentity
    type_rejected: ObjectIdentity
    type_unknown: ObjectIdentity
    unit_rejected: ObjectIdentity
    unit_unknown: ObjectIdentity
    resolver_drift: ObjectIdentity
    executor_rejected: ObjectIdentity
    executor_contract_drift: ObjectIdentity
    verifier_rejected: ObjectIdentity
    verifier_unknown: ObjectIdentity
    verifier_contract_drift: ObjectIdentity

    def __post_init__(self) -> None:
        reasons = self.reasons()
        if len(set(reasons)) != len(reasons):
            raise ValueError("Artifact failure reason 必须互不相同")
        for reason in reasons:
            _require_reason(reason, label="Artifact failure reason")

    def reasons(self) -> tuple[ObjectIdentity, ...]:
        """按协议槽位返回全部 reason，供台账和测试核对完整性。"""
        return (
            self.argument_shape,
            self.type_rejected,
            self.type_unknown,
            self.unit_rejected,
            self.unit_unknown,
            self.resolver_drift,
            self.executor_rejected,
            self.executor_contract_drift,
            self.verifier_rejected,
            self.verifier_unknown,
            self.verifier_contract_drift,
        )


@dataclass(frozen=True)
class FormalArtifactFailure:
    """一次 Artifact 调用失败的完整 typed trace。"""

    reason: ObjectIdentity
    proposition: ObjectIdentity
    parameter: ObjectIdentity | None = None
    expected: ObjectIdentity | None = None
    actual: ObjectIdentity | None = None
    upstream_reason: ObjectIdentity | None = None
    details: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_reason(self.reason, label="Artifact failure reason")
        if (not isinstance(self.proposition, ObjectIdentity)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("Artifact failure proposition 类型不匹配")
        if self.parameter is not None:
            if not isinstance(self.parameter, ObjectIdentity):
                raise TypeError("failure parameter 必须是 ObjectIdentity")
        for label, identity in (
                ("failure expected", self.expected),
                ("failure actual", self.actual)):
            if identity is not None and not isinstance(identity, ObjectIdentity):
                raise TypeError(f"{label} 必须是 ObjectIdentity")
        if self.upstream_reason is not None:
            _require_reason(
                self.upstream_reason, label="Artifact upstream reason")
        _strict_tuple(self.details, label="Artifact failure details")


@dataclass(frozen=True)
class BoundArtifactArgument:
    """通过 type/unit 校验后的参数绑定及兼容依据。"""

    parameter: ArtifactParameter
    argument: ArtifactArgument
    type_support: tuple[ObjectIdentity, ...] = ()
    unit_support: tuple[ObjectIdentity, ...] = ()
    unit_adapter_payload: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parameter, ArtifactParameter):
            raise TypeError("bound parameter 必须是 ArtifactParameter")
        if not isinstance(self.argument, ArtifactArgument):
            raise TypeError("bound argument 必须是 ArtifactArgument")
        if self.parameter.variable != self.argument.parameter:
            raise ValueError("bound argument 与 parameter Variable 不一致")
        for label, support in (
                ("type_support", self.type_support),
                ("unit_support", self.unit_support)):
            if not isinstance(support, tuple):
                raise TypeError(f"bound {label} 必须是 ObjectIdentity tuple")
            if any(not isinstance(item, ObjectIdentity) for item in support):
                raise TypeError(f"bound {label} 含非 ObjectIdentity")
        _strict_tuple(
            self.unit_adapter_payload,
            label="bound unit_adapter_payload",
        )


@dataclass(frozen=True)
class ArtifactExecutionRequest:
    """传给形式域 executor 的有序、已校验调用视图。"""

    invocation: ArtifactInvocation
    arguments: tuple[BoundArtifactArgument, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, ArtifactInvocation):
            raise TypeError("execution invocation 必须是 ArtifactInvocation")
        if not isinstance(self.arguments, tuple):
            raise TypeError("execution arguments 必须是 BoundArtifactArgument tuple")
        declared = self.invocation.definition.parameters
        if len(self.arguments) != len(declared):
            raise ValueError("execution arguments 未完整覆盖 definition 参数")
        for parameter, bound in zip(declared, self.arguments):
            if not isinstance(bound, BoundArtifactArgument):
                raise TypeError("execution arguments 含非 BoundArtifactArgument")
            if bound.parameter != parameter:
                raise ValueError("execution arguments 顺序或参数身份漂移")


@dataclass(frozen=True)
class ArtifactExecutionObservation:
    """executor 返回的来源化结果；bridge 仍须核验契约是否漂移。"""

    authority: ArtifactAuthority
    source: SourceRef
    scope: ScopeIdentity
    executed: bool
    output_payload: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()
    failure_reason: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.authority, ArtifactAuthority):
            raise TypeError("execution authority 必须是 ArtifactAuthority")
        if not isinstance(self.source, SourceRef):
            raise TypeError("execution source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("execution scope 必须是 ScopeIdentity")
        if type(self.executed) is not bool:
            raise TypeError("execution executed 必须是严格 bool")
        _strict_tuple(self.output_payload, label="execution output_payload")
        _strict_tuple(self.trace, label="execution trace")
        if self.executed and self.failure_reason is not None:
            raise ValueError("成功 execution 不得携带 failure_reason")
        if not self.executed and self.failure_reason is not None:
            _require_reason(
                self.failure_reason, label="execution failure_reason")


class FormalArtifactExecutor(Protocol):
    """形式域 adapter 的受限执行协议。"""

    def execute(
            self, request: ArtifactExecutionRequest,
            ) -> ArtifactExecutionObservation:
        """执行已校验调用，并返回显式 authority、source、scope 和 trace。"""
        ...


@dataclass(frozen=True)
class ArtifactVerificationRequest:
    """传给独立 verifier 的 invocation、执行观察、结果和可选规格。"""

    invocation: ArtifactInvocation
    execution: ArtifactExecutionObservation
    result: FormalArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, ArtifactInvocation):
            raise TypeError("verification invocation 必须是 ArtifactInvocation")
        if not isinstance(self.execution, ArtifactExecutionObservation):
            raise TypeError("verification execution 必须是 ArtifactExecutionObservation")
        if not self.execution.executed:
            raise ValueError("未成功执行的 observation 不得进入 verifier")
        if not isinstance(self.result, FormalArtifact):
            raise TypeError("verification result 必须是 FormalArtifact")
        if (self.result.source != self.invocation.source
                or self.result.scope != self.invocation.scope):
            raise ValueError("verification result 与 invocation 归属不一致")


@dataclass(frozen=True)
class ArtifactVerificationObservation:
    """verifier 的三态结论及完整运行归属。"""

    authority: ArtifactAuthority
    source: SourceRef
    scope: ScopeIdentity
    accepted: bool | None
    payload: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()
    failure_reason: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.authority, ArtifactAuthority):
            raise TypeError("verification authority 必须是 ArtifactAuthority")
        if not isinstance(self.source, SourceRef):
            raise TypeError("verification source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("verification scope 必须是 ScopeIdentity")
        if self.accepted is not None and type(self.accepted) is not bool:
            raise TypeError("verification accepted 必须是严格 bool 或 None")
        _strict_tuple(self.payload, label="verification payload")
        _strict_tuple(self.trace, label="verification trace")
        if self.accepted is True and self.failure_reason is not None:
            raise ValueError("accepted verification 不得携带 failure_reason")
        if self.failure_reason is not None:
            _require_reason(
                self.failure_reason, label="verification failure_reason")


class FormalArtifactVerifier(Protocol):
    """与 executor 分离的形式结果验证协议。"""

    def verify(
            self, request: ArtifactVerificationRequest,
            ) -> ArtifactVerificationObservation:
        """验证形式结果；缺规格或依据时返回 unknown，不能自证通过。"""
        ...


@dataclass(frozen=True)
class ArtifactInvocationResult:
    """一次调用的绑定、执行、proof 和语言 Evidence 隔离结果。"""

    invocation: ArtifactInvocation
    bound_arguments: tuple[BoundArtifactArgument, ...]
    execution: ArtifactExecutionObservation | None
    verification: ArtifactVerificationObservation | None
    value: FormalArtifact | None
    proof: FormalArtifact | None
    proposition_state: LogicEvidenceState
    failures: tuple[FormalArtifactFailure, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, ArtifactInvocation):
            raise TypeError("result invocation 必须是 ArtifactInvocation")
        if not isinstance(self.bound_arguments, tuple):
            raise TypeError("result bound_arguments 必须是 tuple")
        if any(not isinstance(item, BoundArtifactArgument)
               for item in self.bound_arguments):
            raise TypeError("result bound_arguments 类型非法")
        for label, observation, expected_type in (
                ("execution", self.execution, ArtifactExecutionObservation),
                ("verification", self.verification,
                 ArtifactVerificationObservation)):
            if observation is not None and not isinstance(
                    observation, expected_type):
                raise TypeError(f"result {label} 类型非法")
        for label, artifact in (("value", self.value), ("proof", self.proof)):
            if artifact is not None:
                if not isinstance(artifact, FormalArtifact):
                    raise TypeError(f"result {label} 必须是 FormalArtifact")
                if (artifact.source != self.invocation.source
                        or artifact.scope != self.invocation.scope):
                    raise ValueError(f"result {label} 与 invocation 归属不一致")
        if not isinstance(self.proposition_state, LogicEvidenceState):
            raise TypeError("result proposition_state 必须是 LogicEvidenceState")
        if self.proposition_state.support or self.proposition_state.refute:
            raise ValueError("S-06 形式结果不得直接生成语言 Proposition Evidence")
        if not isinstance(self.failures, tuple):
            raise TypeError("result failures 必须是 FormalArtifactFailure tuple")
        if any(not isinstance(item, FormalArtifactFailure)
               for item in self.failures):
            raise TypeError("result failures 类型非法")

    @property
    def succeeded(self) -> bool:
        """只在独立 verifier 接受且 value/proof 均完整时返回真。"""
        return (
            not self.failures
            and self.execution is not None
            and self.execution.executed
            and self.verification is not None
            and self.verification.accepted is True
            and self.value is not None
            and self.proof is not None
        )


class FormalArtifactBridge:
    """以注入 resolver、executor 和 verifier 完成一次 fail-closed 调用。"""

    def __init__(
            self,
            type_resolver: ArtifactCompatibilityResolver,
            unit_resolver: ArtifactCompatibilityResolver,
            executor: FormalArtifactExecutor,
            verifier: FormalArtifactVerifier,
            failures: FormalArtifactFailureProtocol,
            ) -> None:
        self.type_resolver = type_resolver
        self.unit_resolver = unit_resolver
        self.executor = executor
        self.verifier = verifier
        self.failures = failures

    def invoke(self, invocation: ArtifactInvocation) -> ArtifactInvocationResult:
        """绑定参数、执行并独立验证；任一身份或兼容边界异常均停止后续升级。"""
        if not isinstance(invocation, ArtifactInvocation):
            raise TypeError("invocation 必须是 ArtifactInvocation")
        unknown = LogicEvidenceState(False, False)
        bound, binding_failures = self._bind(invocation)
        if binding_failures:
            return ArtifactInvocationResult(
                invocation, bound, None, None, None, None,
                unknown, binding_failures)

        execution = self.executor.execute(
            ArtifactExecutionRequest(invocation, bound))
        execution_failure = self._execution_failure(invocation, execution)
        if execution_failure is not None:
            return ArtifactInvocationResult(
                invocation, bound, execution, None, None, None,
                unknown, (execution_failure,))

        value = self._result_artifact(invocation, execution)
        expected_failure = self._expected_failure(invocation, value)
        if expected_failure is not None:
            return ArtifactInvocationResult(
                invocation, bound, execution, None, value, None,
                unknown, (expected_failure,))

        verification = self.verifier.verify(
            ArtifactVerificationRequest(invocation, execution, value))
        contract_failure = self._verification_contract_failure(
            invocation, verification)
        if contract_failure is not None:
            return ArtifactInvocationResult(
                invocation, bound, execution, verification, value, None,
                unknown, (contract_failure,))

        proof = self._proof_artifact(invocation, value, verification)
        if verification.accepted is not True:
            reason = (
                self.failures.verifier_rejected
                if verification.accepted is False
                else self.failures.verifier_unknown)
            failure = FormalArtifactFailure(
                reason,
                invocation.proposition,
                upstream_reason=verification.failure_reason,
            )
            return ArtifactInvocationResult(
                invocation, bound, execution, verification, value, proof,
                unknown, (failure,))
        return ArtifactInvocationResult(
            invocation, bound, execution, verification, value, proof,
            unknown, ())

    def _bind(
            self, invocation: ArtifactInvocation,
            ) -> tuple[
                tuple[BoundArtifactArgument, ...],
                tuple[FormalArtifactFailure, ...],
                ]:
        """按 Variable 全身份绑定全部参数，并保留 type/unit resolver 依据。"""
        expected = {
            parameter.variable: parameter
            for parameter in invocation.definition.parameters
        }
        supplied: dict[ObjectIdentity, ArtifactArgument] = {}
        failures: list[FormalArtifactFailure] = []
        for argument in invocation.arguments:
            if argument.parameter in supplied or argument.parameter not in expected:
                failures.append(FormalArtifactFailure(
                    self.failures.argument_shape,
                    invocation.proposition,
                    parameter=argument.parameter,
                ))
                continue
            supplied[argument.parameter] = argument
        for parameter in invocation.definition.parameters:
            if parameter.variable not in supplied:
                failures.append(FormalArtifactFailure(
                    self.failures.argument_shape,
                    invocation.proposition,
                    parameter=parameter.variable,
                ))
        if failures:
            return (), tuple(failures)

        bound: list[BoundArtifactArgument] = []
        for parameter in invocation.definition.parameters:
            argument = supplied[parameter.variable]
            type_result = self.type_resolver.resolve(
                parameter.schema.value_type,
                argument.value.schema.value_type,
            )
            failure = self._compatibility_failure(
                invocation,
                parameter,
                type_result,
                expected_identity=parameter.schema.value_type,
                actual_identity=argument.value.schema.value_type,
                rejected_reason=self.failures.type_rejected,
                unknown_reason=self.failures.type_unknown,
            )
            if failure is not None:
                failures.append(failure)
                continue
            unit_result = self.unit_resolver.resolve(
                parameter.schema.unit,
                argument.value.schema.unit,
            )
            failure = self._compatibility_failure(
                invocation,
                parameter,
                unit_result,
                expected_identity=parameter.schema.unit,
                actual_identity=argument.value.schema.unit,
                rejected_reason=self.failures.unit_rejected,
                unknown_reason=self.failures.unit_unknown,
            )
            if failure is not None:
                failures.append(failure)
                continue
            bound.append(BoundArtifactArgument(
                parameter,
                argument,
                type_result.support,
                unit_result.support,
                unit_result.adapter_payload,
            ))
        if failures:
            return tuple(bound), tuple(failures)
        return tuple(bound), ()

    def _compatibility_failure(
            self,
            invocation: ArtifactInvocation,
            parameter: ArtifactParameter | None,
            result: ArtifactCompatibilityResult,
            *,
            expected_identity: ObjectIdentity,
            actual_identity: ObjectIdentity,
            rejected_reason: ObjectIdentity,
            unknown_reason: ObjectIdentity,
            ) -> FormalArtifactFailure | None:
        """核验 resolver 没有替换问题身份，并把三态结果映射为 typed failure。"""
        parameter_identity = None if parameter is None else parameter.variable
        if (not isinstance(result, ArtifactCompatibilityResult)
                or result.expected != expected_identity
                or result.actual != actual_identity):
            return FormalArtifactFailure(
                self.failures.resolver_drift,
                invocation.proposition,
                parameter=parameter_identity,
                expected=expected_identity,
                actual=actual_identity,
            )
        if result.compatible is True:
            return None
        reason = rejected_reason if result.compatible is False else unknown_reason
        return FormalArtifactFailure(
            reason,
            invocation.proposition,
            parameter=parameter_identity,
            expected=expected_identity,
            actual=actual_identity,
        )

    def _execution_failure(
            self,
            invocation: ArtifactInvocation,
            execution: ArtifactExecutionObservation,
            ) -> FormalArtifactFailure | None:
        """核验 executor authority 和当前上下文，拒绝来源、scope 或版本漂移。"""
        if not isinstance(execution, ArtifactExecutionObservation):
            raise TypeError("executor 必须返回 ArtifactExecutionObservation")
        if (execution.authority != invocation.definition.executor
                or execution.source != invocation.source
                or execution.scope != invocation.scope):
            return FormalArtifactFailure(
                self.failures.executor_contract_drift,
                invocation.proposition,
            )
        if not execution.executed:
            return FormalArtifactFailure(
                self.failures.executor_rejected,
                invocation.proposition,
                upstream_reason=execution.failure_reason,
            )
        return None

    def _expected_failure(
            self, invocation: ArtifactInvocation, value: FormalArtifact,
            ) -> FormalArtifactFailure | None:
        """在 verifier 前核验 expected 的 type/unit，防止异型规格参与比较。"""
        expected = invocation.expected
        if expected is None:
            return None
        type_result = self.type_resolver.resolve(
            invocation.definition.result_schema.value_type,
            expected.schema.value_type,
        )
        failure = self._compatibility_failure(
            invocation,
            None,
            type_result,
            expected_identity=invocation.definition.result_schema.value_type,
            actual_identity=expected.schema.value_type,
            rejected_reason=self.failures.type_rejected,
            unknown_reason=self.failures.type_unknown,
        )
        if failure is not None:
            return failure
        unit_result = self.unit_resolver.resolve(
            invocation.definition.result_schema.unit,
            expected.schema.unit,
        )
        return self._compatibility_failure(
            invocation,
            None,
            unit_result,
            expected_identity=invocation.definition.result_schema.unit,
            actual_identity=expected.schema.unit,
            rejected_reason=self.failures.unit_rejected,
            unknown_reason=self.failures.unit_unknown,
        )

    def _verification_contract_failure(
            self,
            invocation: ArtifactInvocation,
            verification: ArtifactVerificationObservation,
            ) -> FormalArtifactFailure | None:
        """核验 verifier 身份、版本和运行归属均与 definition/invocation 一致。"""
        if not isinstance(verification, ArtifactVerificationObservation):
            raise TypeError("verifier 必须返回 ArtifactVerificationObservation")
        if (verification.authority != invocation.definition.verifier
                or verification.source != invocation.source
                or verification.scope != invocation.scope):
            return FormalArtifactFailure(
                self.failures.verifier_contract_drift,
                invocation.proposition,
            )
        return None

    def _result_artifact(
            self,
            invocation: ArtifactInvocation,
            execution: ArtifactExecutionObservation,
            ) -> FormalArtifact:
        """从完整 invocation 和 executor observation 构造当前 scope 的 value Artifact。"""
        declaration_key = (
            _RESULT_IDENTITY_VERSION,
            *_packed(invocation.proposition.stable_key()),
            *_packed(invocation.definition.program.identity.stable_key()),
            *_packed(invocation.scope.stable_key()),
            *_packed(invocation.invocation_key),
            *_packed(execution.authority.stable_key()),
            *_packed(execution.output_payload),
        )
        identity = artifact_identity(
            invocation.source,
            invocation.definition.result_kind,
            invocation.definition.result_schema,
            declaration_key,
            execution.output_payload,
            invocation.scope,
        )
        return FormalArtifact(
            identity,
            invocation.definition.result_kind,
            invocation.definition.result_schema,
            invocation.source,
            execution.output_payload,
            invocation.scope,
        )

    def _proof_artifact(
            self,
            invocation: ArtifactInvocation,
            value: FormalArtifact,
            verification: ArtifactVerificationObservation,
            ) -> FormalArtifact:
        """构造保留 verifier/version/结论/trace 的当前 proof-result Artifact。"""
        if verification.accepted is True:
            state = _PROOF_ACCEPTED
        elif verification.accepted is False:
            state = _PROOF_REJECTED
        else:
            state = _PROOF_UNKNOWN
        payload = (
            state,
            *_packed(verification.payload),
            *_packed(verification.trace),
        )
        declaration_key = (
            _PROOF_IDENTITY_VERSION,
            *_packed(value.identity.stable_key()),
            *_packed(verification.authority.stable_key()),
            *_packed(invocation.source.stable_key()),
            *_packed(invocation.scope.stable_key()),
            *_packed(payload),
        )
        identity = artifact_identity(
            invocation.source,
            invocation.definition.proof_kind,
            invocation.definition.proof_schema,
            declaration_key,
            payload,
            invocation.scope,
        )
        return FormalArtifact(
            identity,
            invocation.definition.proof_kind,
            invocation.definition.proof_schema,
            invocation.source,
            payload,
            invocation.scope,
        )


__all__ = [
    "ArtifactExecutionObservation",
    "ArtifactExecutionRequest",
    "ArtifactInvocationResult",
    "ArtifactVerificationObservation",
    "ArtifactVerificationRequest",
    "BoundArtifactArgument",
    "FormalArtifactBridge",
    "FormalArtifactExecutor",
    "FormalArtifactFailure",
    "FormalArtifactFailureProtocol",
    "FormalArtifactVerifier",
]
