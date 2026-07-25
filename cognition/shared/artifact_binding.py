"""A-06 STRUCT_BIND 到 typed Artifact 调用的不可变领域对象。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactInvocation,
    FormalArtifact,
    FormalArtifactDefinition,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactExecutionObservation,
    ArtifactInvocationResult,
    ArtifactVerificationObservation,
    BoundArtifactArgument,
    FormalArtifactFailure,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_QUERY,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import describe_variable
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _strict_key(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """核验开放稳定键只含严格整数，并按调用点决定是否允许空键。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是 tuple")
    if not value and not allow_empty:
        raise ValueError(f"{label} 不能为空")
    assert_int(*value, _where=label)
    if any(type(part) is not int for part in value):
        raise ValueError(f"{label} 必须只含严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界，避免相邻字段连接歧义。"""
    return len(value), *value


def _slot(value: ConceptRef, *, label: str) -> ConceptRef:
    """核验 legacy 槽位仍是二元严格整数引用，不解释其领域含义。"""
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{label} 必须是二元 ConceptRef")
    assert_int(*value, _where=label)
    if any(type(part) is not int for part in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _instruction(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验选择理由是一等最小指令，而不是代码内置字符串。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return value


def _optional_identity(value: ObjectIdentity | None) -> tuple[int, ...]:
    """把可空一等身份编码为带存在位的完整稳定键。"""
    return (0,) if value is None else (1, *_packed(value.stable_key()))


def _optional_artifact(value: FormalArtifact | None) -> tuple[int, ...]:
    """把可空 Artifact 编码为带存在位的完整领域稳定键。"""
    return (0,) if value is None else (1, *_packed(value.stable_key()))


def _bound_argument_key(value: BoundArtifactArgument) -> tuple[int, ...]:
    """编码参数、实参、类型/单位依据和单位适配载荷。"""
    result = [
        *_packed(value.parameter.stable_key()),
        *_packed(value.argument.parameter.stable_key()),
        *_packed(value.argument.value.stable_key()),
        len(value.type_support),
    ]
    for support in value.type_support:
        result.extend(_packed(support.stable_key()))
    result.append(len(value.unit_support))
    for support in value.unit_support:
        result.extend(_packed(support.stable_key()))
    result.extend(_packed(value.unit_adapter_payload))
    return tuple(result)


def _execution_key(
        value: ArtifactExecutionObservation | None,
        ) -> tuple[int, ...]:
    """编码可空 executor 观察的 authority、归属、结果和失败依据。"""
    if value is None:
        return (0,)
    return (
        1,
        *_packed(value.authority.stable_key()),
        *_packed(value.source.stable_key()),
        *_packed(value.scope.stable_key()),
        int(value.executed),
        *_packed(value.output_payload),
        *_packed(value.trace),
        *_optional_identity(value.failure_reason),
    )


def _verification_key(
        value: ArtifactVerificationObservation | None,
        ) -> tuple[int, ...]:
    """编码可空 verifier 三态、authority、归属、载荷和失败依据。"""
    if value is None:
        return (0,)
    accepted = 0 if value.accepted is None else (2 if value.accepted else 1)
    return (
        1,
        *_packed(value.authority.stable_key()),
        *_packed(value.source.stable_key()),
        *_packed(value.scope.stable_key()),
        accepted,
        *_packed(value.payload),
        *_packed(value.trace),
        *_optional_identity(value.failure_reason),
    )


def _failure_key(value: FormalArtifactFailure) -> tuple[int, ...]:
    """编码一个形式失败的 reason、命题、参数、期望、实际和上游 trace。"""
    return (
        *_packed(value.reason.stable_key()),
        *_packed(value.proposition.stable_key()),
        *_optional_identity(value.parameter),
        *_optional_identity(value.expected),
        *_optional_identity(value.actual),
        *_optional_identity(value.upstream_reason),
        *_packed(value.details),
    )


def _invocation_result_key(value: ArtifactInvocationResult) -> tuple[int, ...]:
    """无损编码既有 S-06 结果，不要求其临时新增跨模块 stable_key 接口。"""
    result = [
        *_packed(value.invocation.stable_key()),
        len(value.bound_arguments),
    ]
    for argument in value.bound_arguments:
        result.extend(_packed(_bound_argument_key(argument)))
    result.extend((
        *_packed(_execution_key(value.execution)),
        *_packed(_verification_key(value.verification)),
        *_optional_artifact(value.value),
        *_optional_artifact(value.proof),
        *_packed(value.proposition_state.stable_key()),
        len(value.failures),
    ))
    for failure in value.failures:
        result.extend(_packed(_failure_key(failure)))
    return tuple(result)


@dataclass(frozen=True)
class ArtifactBindingEndpoint:
    """调用方声明的 legacy 槽位与完整 Variable 身份映射。"""

    slot_ref: ConceptRef
    variable: ObjectIdentity

    def __post_init__(self) -> None:
        """拒绝裸槽位或非 Variable 身份进入 A-06 选择协议。"""
        _slot(self.slot_ref, label="ArtifactBindingEndpoint.slot_ref")
        describe_variable(self.variable)

    def stable_key(self) -> tuple[int, ...]:
        """返回槽位和 Variable 全身份组成的无歧义稳定键。"""
        return (*self.slot_ref, *_packed(self.variable.stable_key()))


@dataclass(frozen=True)
class ArtifactBindingValue:
    """当前 episode 中由一个 source Variable 提供的 typed Artifact 值。"""

    endpoint: ArtifactBindingEndpoint
    artifact: FormalArtifact

    def __post_init__(self) -> None:
        """保证值绑定到完整 endpoint，具体类型与单位由 S-06 独立复核。"""
        if not isinstance(self.endpoint, ArtifactBindingEndpoint):
            raise TypeError("ArtifactBindingValue.endpoint 类型错误")
        if not isinstance(self.artifact, FormalArtifact):
            raise TypeError("ArtifactBindingValue.artifact 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回 endpoint 与 Artifact 完整身份、schema、scope 和载荷键。"""
        return (
            *_packed(self.endpoint.stable_key()),
            *_packed(self.artifact.stable_key()),
        )


@dataclass(frozen=True)
class ArtifactBindingChoice:
    """调用方对一条真实 STRUCT_BIND 候选的显式采用声明。"""

    source: ArtifactBindingEndpoint
    target: ArtifactBindingEndpoint
    correspondence_key: tuple[int, ...]
    reason: ObjectIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求候选完整键、注入式理由和非空 trace，禁止按顺序私选。"""
        if not isinstance(self.source, ArtifactBindingEndpoint):
            raise TypeError("ArtifactBindingChoice.source 类型错误")
        if not isinstance(self.target, ArtifactBindingEndpoint):
            raise TypeError("ArtifactBindingChoice.target 类型错误")
        _strict_key(
            self.correspondence_key,
            label="ArtifactBindingChoice.correspondence_key",
        )
        _instruction(self.reason, label="ArtifactBindingChoice.reason")
        _strict_key(self.trace, label="ArtifactBindingChoice.trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回端点、候选、选择理由和调用方 trace 的完整稳定键。"""
        return (
            *_packed(self.source.stable_key()),
            *_packed(self.target.stable_key()),
            *_packed(self.correspondence_key),
            *_packed(self.reason.stable_key()),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class ArtifactBindingRequest:
    """一次覆盖完整 Artifact definition 参数的 STRUCT_BIND 调用请求。"""

    proposition: ObjectIdentity
    definition: FormalArtifactDefinition
    source: SourceRef
    scope: ScopeIdentity
    invocation_key: tuple[int, ...]
    endpoints: tuple[ArtifactBindingEndpoint, ...]
    values: tuple[ArtifactBindingValue, ...]
    choices: tuple[ArtifactBindingChoice, ...]
    expected: FormalArtifact | None = None

    def __post_init__(self) -> None:
        """核验 endpoint、值和显式选择一一覆盖 definition 的全部参数。"""
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("ArtifactBindingRequest.proposition 类型错误")
        if not isinstance(self.definition, FormalArtifactDefinition):
            raise TypeError("ArtifactBindingRequest.definition 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("ArtifactBindingRequest.source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("ArtifactBindingRequest.scope 类型错误")
        if self.scope.scope_kind != SCOPE_QUERY:
            raise ValueError("ArtifactBindingRequest.scope 必须是 query scope")
        if (self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions
                or (self.scope.source is not None
                    and self.scope.source != self.source)):
            raise ValueError("ArtifactBindingRequest scope 与 source 不一致")
        _strict_key(
            self.invocation_key,
            label="ArtifactBindingRequest.invocation_key",
        )
        if not isinstance(self.endpoints, tuple) or not self.endpoints:
            raise ValueError("ArtifactBindingRequest.endpoints 不能为空")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("ArtifactBindingRequest.values 不能为空")
        if not isinstance(self.choices, tuple) or not self.choices:
            raise ValueError("ArtifactBindingRequest.choices 不能为空")
        if self.expected is not None and not isinstance(
                self.expected, FormalArtifact):
            raise TypeError("ArtifactBindingRequest.expected 类型错误")

        by_slot: dict[ConceptRef, ArtifactBindingEndpoint] = {}
        by_variable: dict[ObjectIdentity, ArtifactBindingEndpoint] = {}
        for endpoint in self.endpoints:
            if not isinstance(endpoint, ArtifactBindingEndpoint):
                raise TypeError("ArtifactBindingRequest.endpoints 含非法项")
            if endpoint.slot_ref in by_slot:
                raise ValueError("一个 slot_ref 不得映射多个 endpoint")
            if endpoint.variable in by_variable:
                raise ValueError("一个 Variable 不得映射多个 slot_ref")
            by_slot[endpoint.slot_ref] = endpoint
            by_variable[endpoint.variable] = endpoint

        values_by_variable: dict[ObjectIdentity, ArtifactBindingValue] = {}
        for value in self.values:
            if not isinstance(value, ArtifactBindingValue):
                raise TypeError("ArtifactBindingRequest.values 含非法项")
            if by_variable.get(value.endpoint.variable) != value.endpoint:
                raise ValueError("ArtifactBindingValue endpoint 未在请求中声明")
            if value.endpoint.variable in values_by_variable:
                raise ValueError("同一 source Variable 不得提供多个 Artifact 值")
            values_by_variable[value.endpoint.variable] = value

        choices_by_target: dict[ObjectIdentity, ArtifactBindingChoice] = {}
        for choice in self.choices:
            if not isinstance(choice, ArtifactBindingChoice):
                raise TypeError("ArtifactBindingRequest.choices 含非法项")
            if (by_variable.get(choice.source.variable) != choice.source
                    or by_variable.get(choice.target.variable) != choice.target):
                raise ValueError("ArtifactBindingChoice endpoint 未在请求中声明")
            if choice.source.variable not in values_by_variable:
                raise ValueError("ArtifactBindingChoice source 缺少当前 Artifact 值")
            if choice.target.variable in choices_by_target:
                raise ValueError("同一 target parameter 不得选择多个绑定")
            choices_by_target[choice.target.variable] = choice

        parameter_variables = tuple(
            parameter.variable for parameter in self.definition.parameters)
        if set(choices_by_target) != set(parameter_variables):
            raise ValueError("ArtifactBindingChoice 必须一一覆盖 definition 参数")
        if len(choices_by_target) != len(parameter_variables):
            raise ValueError("Artifact definition 参数身份重复或选择不唯一")

    def stable_key(self) -> tuple[int, ...]:
        """返回请求内定义、scope、值和显式选择的完整稳定键。"""
        expected_key = (
            () if self.expected is None else self.expected.stable_key())
        result = [
            *_packed(self.proposition.stable_key()),
            *_packed(self.definition.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.invocation_key),
            len(self.endpoints),
        ]
        for endpoint in self.endpoints:
            result.extend(_packed(endpoint.stable_key()))
        result.append(len(self.values))
        for value in self.values:
            result.extend(_packed(value.stable_key()))
        result.append(len(self.choices))
        for choice in self.choices:
            result.extend(_packed(choice.stable_key()))
        result.extend(_packed(expected_key))
        return tuple(result)


@dataclass(frozen=True)
class ArtifactBindingReadTrace:
    """一个 source slot 的全部 STRUCT_BIND 通过项和失败项稳定键。"""

    source_slot: ConceptRef
    correspondence_keys: tuple[tuple[int, ...], ...]
    failure_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """保留 reader 的全部候选和失败，不把读取顺序解释为选择。"""
        _slot(self.source_slot, label="ArtifactBindingReadTrace.source_slot")
        for label, keys in (
                ("correspondence_keys", self.correspondence_keys),
                ("failure_keys", self.failure_keys)):
            if not isinstance(keys, tuple):
                raise TypeError(f"ArtifactBindingReadTrace.{label} 必须是 tuple")
            for key in keys:
                _strict_key(key, label=f"ArtifactBindingReadTrace.{label}")

    def stable_key(self) -> tuple[int, ...]:
        """返回 source slot 及全部候选、失败键的确定性 trace。"""
        result = [*self.source_slot, len(self.correspondence_keys)]
        for key in self.correspondence_keys:
            result.extend(_packed(key))
        result.append(len(self.failure_keys))
        for key in self.failure_keys:
            result.extend(_packed(key))
        return tuple(result)


@dataclass(frozen=True)
class ArtifactBindingRun:
    """一次真实 STRUCT_BIND 采用、形式调用和 WorkMemory 记录的完整报告。"""

    request: ArtifactBindingRequest
    reads: tuple[ArtifactBindingReadTrace, ...]
    invocation: ArtifactInvocation
    result: ArtifactInvocationResult

    def __post_init__(self) -> None:
        """核验报告只挂接由当前请求构造的同一 invocation 与结果。"""
        if not isinstance(self.request, ArtifactBindingRequest):
            raise TypeError("ArtifactBindingRun.request 类型错误")
        if not isinstance(self.reads, tuple) or any(
                not isinstance(item, ArtifactBindingReadTrace)
                for item in self.reads):
            raise TypeError("ArtifactBindingRun.reads 类型错误")
        if not isinstance(self.invocation, ArtifactInvocation):
            raise TypeError("ArtifactBindingRun.invocation 类型错误")
        if not isinstance(self.result, ArtifactInvocationResult):
            raise TypeError("ArtifactBindingRun.result 类型错误")
        if self.result.invocation != self.invocation:
            raise ValueError("ArtifactBindingRun result 不属于当前 invocation")
        if (self.invocation.proposition != self.request.proposition
                or self.invocation.definition != self.request.definition
                or self.invocation.source != self.request.source
                or self.invocation.scope != self.request.scope
                or self.invocation.invocation_key
                != self.request.invocation_key
                or self.invocation.expected != self.request.expected):
            raise ValueError("ArtifactBindingRun invocation 与请求漂移")

    @property
    def succeeded(self) -> bool:
        """返回 S-06 executor 与独立 verifier 是否共同完成。"""
        return self.result.succeeded

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、reader trace、invocation 和形式结果的完整稳定键。"""
        result = [*_packed(self.request.stable_key()), len(self.reads)]
        for read in self.reads:
            result.extend(_packed(read.stable_key()))
        result.extend((
            *_packed(self.invocation.stable_key()),
            *_packed(_invocation_result_key(self.result)),
        ))
        return tuple(result)


__all__ = [
    "ArtifactBindingChoice",
    "ArtifactBindingEndpoint",
    "ArtifactBindingReadTrace",
    "ArtifactBindingRequest",
    "ArtifactBindingRun",
    "ArtifactBindingValue",
]
