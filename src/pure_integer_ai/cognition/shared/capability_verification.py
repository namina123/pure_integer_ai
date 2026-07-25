"""C-01 Capability held-out case、验证报告与三态转换领域对象。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.capability_candidate import (
    CapabilityCandidate,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactAuthority,
    ArtifactInvocation,
    FormalArtifact,
    FormalArtifactDefinition,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactInvocationResult,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_QUERY,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_CASE_DOMAIN = "capability.held_out.case.v1"
_INVOCATION_DOMAIN = "capability.held_out.invocation.v1"
_RESULT_DOMAIN = "capability.held_out.result.v1"
_CANDIDATE_DOMAIN = "capability.held_out.candidate.v1"


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(value), *value


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 held-out 声明键或 trace 只含严格整数且非空。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空 tuple")
    assert_int(*value, _where=label)
    if any(type(part) is not int for part in value):
        raise ValueError(f"{label} 必须只含严格整数")
    return value


def _instruction(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验 held-out 形成理由是一等 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return value


def capability_argument_content_key(
        definition: FormalArtifactDefinition,
        values: tuple[FormalArtifact, ...],
        ) -> tuple[int, ...]:
    """提取与来源、scope 和声明身份无关的有序参数内容键。"""
    if not isinstance(definition, FormalArtifactDefinition):
        raise TypeError("definition 必须是 FormalArtifactDefinition")
    if not isinstance(values, tuple) or any(
            not isinstance(item, FormalArtifact) for item in values):
        raise TypeError("values 必须是 FormalArtifact tuple")
    if len(values) != len(definition.parameters):
        raise ValueError("参数值数量必须精确覆盖 definition parameters")
    result = [len(values)]
    for parameter, value in zip(definition.parameters, values):
        result.extend(_packed(parameter.schema.stable_key()))
        result.extend(_packed(value.artifact_kind.stable_key()))
        result.extend(_packed(value.schema.stable_key()))
        result.extend(_packed(value.payload))
    return tuple(result)


def _optional_artifact(value: FormalArtifact | None) -> tuple[int, ...]:
    """把可选 Artifact 编为有显式存在位的稳定键。"""
    if value is None:
        return (0,)
    return 1, *_packed(value.stable_key())


def _optional_identity(value: ObjectIdentity | None) -> tuple[int, ...]:
    """把可选对象身份编为有显式存在位的稳定键。"""
    if value is None:
        return (0,)
    return 1, *_packed(value.stable_key())


def _result_audit_key(value: ArtifactInvocationResult) -> tuple[int, ...]:
    """编码 C-01 晋升所依赖的执行、验证、结果、proof 和失败证据。"""
    if not isinstance(value, ArtifactInvocationResult):
        raise TypeError("value 必须是 ArtifactInvocationResult")
    result = [*_packed(value.invocation.stable_key())]
    if value.execution is None:
        result.append(0)
    else:
        result.extend((
            1,
            *_packed(value.execution.authority.stable_key()),
            *_packed(value.execution.source.stable_key()),
            *_packed(value.execution.scope.stable_key()),
            1 if value.execution.executed else 0,
            *_packed(value.execution.output_payload),
            *_packed(value.execution.trace),
            *_optional_identity(value.execution.failure_reason),
        ))
    if value.verification is None:
        result.append(0)
    else:
        accepted = (
            1 if value.verification.accepted is True
            else 0 if value.verification.accepted is False
            else -1
        )
        result.extend((
            1,
            *_packed(value.verification.authority.stable_key()),
            *_packed(value.verification.source.stable_key()),
            *_packed(value.verification.scope.stable_key()),
            accepted,
            *_packed(value.verification.payload),
            *_packed(value.verification.trace),
            *_optional_identity(value.verification.failure_reason),
        ))
    result.extend((*_optional_artifact(value.value), *_optional_artifact(value.proof)))
    result.append(len(value.failures))
    for failure in value.failures:
        result.extend((
            *_packed(failure.reason.stable_key()),
            *_packed(failure.proposition.stable_key()),
            *_optional_identity(failure.parameter),
            *_optional_identity(failure.expected),
            *_optional_identity(failure.actual),
            *_optional_identity(failure.upstream_reason),
            *_packed(failure.details),
        ))
    return tuple(result)


@dataclass(frozen=True)
class CapabilityHeldOutCase:
    """执行前冻结的未见参数、独立 expected 与规格来源。"""

    source: SourceRef
    scope: ScopeIdentity
    proposition: ObjectIdentity
    arguments: tuple[FormalArtifact, ...]
    expected: FormalArtifact
    specification_refs: tuple[ObjectIdentity, ...]
    specification_authority: ArtifactAuthority
    formation_reason: ObjectIdentity
    formation_trace: tuple[int, ...]
    case_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 held-out case 在执行前已有完整 query 归属和规格证据。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("CapabilityHeldOutCase.source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("CapabilityHeldOutCase.scope 类型错误")
        if self.scope.scope_kind != SCOPE_QUERY:
            raise ValueError("Capability held-out scope 必须是 query")
        if (self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions
                or self.scope.source != self.source):
            raise ValueError("Capability held-out scope 与 source 不一致")
        if (not isinstance(self.proposition, ObjectIdentity)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("Capability held-out proposition 类型错误")
        if semantic_source(self.proposition) != self.source:
            raise ValueError("Capability held-out proposition 来源漂移")
        if not isinstance(self.arguments, tuple) or not self.arguments:
            raise ValueError("Capability held-out 至少需要一个参数")
        if any(not isinstance(item, FormalArtifact) for item in self.arguments):
            raise TypeError("Capability held-out arguments 类型错误")
        if any(item.source != self.source or item.scope != self.scope
               for item in self.arguments):
            raise ValueError("Capability held-out argument 归属漂移")
        if not isinstance(self.expected, FormalArtifact):
            raise TypeError("Capability held-out expected 类型错误")
        if self.expected.source != self.source or self.expected.scope != self.scope:
            raise ValueError("Capability held-out expected 归属漂移")
        if not isinstance(self.specification_refs, tuple) or not self.specification_refs:
            raise ValueError("Capability held-out 必须提供非空 specification refs")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.specification_refs):
            raise TypeError("Capability held-out specification refs 类型错误")
        if len(set(self.specification_refs)) != len(self.specification_refs):
            raise ValueError("Capability held-out specification refs 不得重复")
        if not isinstance(self.specification_authority, ArtifactAuthority):
            raise TypeError("Capability held-out specification authority 类型错误")
        _instruction(
            self.formation_reason,
            label="Capability held-out formation reason",
        )
        _strict_key(
            self.formation_trace,
            label="Capability held-out formation trace",
        )
        _strict_key(self.case_key, label="Capability held-out case key")

    def argument_content_key(
            self, definition: FormalArtifactDefinition,
            ) -> tuple[int, ...]:
        """按 candidate 参数契约返回去来源 held-out 输入内容键。"""
        return capability_argument_content_key(definition, self.arguments)

    def stable_key(self) -> tuple[int, ...]:
        """返回 case 全部输入、expected、规格 authority 和形成依据。"""
        result = [
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.proposition.stable_key()),
            len(self.arguments),
        ]
        for argument in self.arguments:
            result.extend(_packed(argument.stable_key()))
        result.extend((
            *_packed(self.expected.stable_key()),
            len(self.specification_refs),
        ))
        for specification in self.specification_refs:
            result.extend(_packed(specification.stable_key()))
        result.extend((
            *_packed(self.specification_authority.stable_key()),
            *_packed(self.formation_reason.stable_key()),
            *_packed(self.formation_trace),
            *_packed(self.case_key),
        ))
        return tuple(result)

    def content_ref(self) -> tuple[int, ...]:
        """为验证报告生成固定长度 held-out case 内容引用。"""
        return integer_tuple_fingerprint(self.stable_key(), domain=_CASE_DOMAIN)


def capability_verification_state(
        candidate: CapabilityCandidate,
        result: ArtifactInvocationResult,
        ) -> ObjectIdentity:
    """按明确接受、候选失败和不可归责 unknown 决定注入状态。"""
    if not isinstance(candidate, CapabilityCandidate):
        raise TypeError("candidate 必须是 CapabilityCandidate")
    if not isinstance(result, ArtifactInvocationResult):
        raise TypeError("result 必须是 ArtifactInvocationResult")
    protocol = candidate.status_protocol
    if result.succeeded:
        return protocol.verified
    if result.verification is not None and result.verification.accepted is False:
        return protocol.rejected
    if result.execution is not None and result.execution.executed is False:
        return protocol.rejected
    return protocol.provisional


@dataclass(frozen=True)
class CapabilityVerificationReport:
    """C-01 对一个 candidate 和独立 held-out case 的完整三态报告。"""

    candidate: CapabilityCandidate
    held_out: CapabilityHeldOutCase
    invocation: ArtifactInvocation
    result: ArtifactInvocationResult
    previous_state: ObjectIdentity
    state: ObjectIdentity

    def __post_init__(self) -> None:
        """核验报告绑定同一次调用且状态严格由结果和注入协议决定。"""
        if not isinstance(self.candidate, CapabilityCandidate):
            raise TypeError("CapabilityVerificationReport.candidate 类型错误")
        if not isinstance(self.held_out, CapabilityHeldOutCase):
            raise TypeError("CapabilityVerificationReport.held_out 类型错误")
        if not isinstance(self.invocation, ArtifactInvocation):
            raise TypeError("CapabilityVerificationReport.invocation 类型错误")
        if not isinstance(self.result, ArtifactInvocationResult):
            raise TypeError("CapabilityVerificationReport.result 类型错误")
        if self.result.invocation != self.invocation:
            raise ValueError("Capability verification result 不属于当前 invocation")
        if self.invocation.definition != self.candidate.proposal.definition:
            raise ValueError("Capability verification invocation 替换了 candidate definition")
        if self.invocation.expected != self.held_out.expected:
            raise ValueError("Capability verification invocation expected 漂移")
        if self.previous_state != self.candidate.status_protocol.provisional:
            raise ValueError("C-01 只接受 provisional candidate")
        expected_state = capability_verification_state(
            self.candidate, self.result)
        if self.state != expected_state:
            raise ValueError("Capability verification state 与结果不一致")

    @property
    def verified(self) -> bool:
        """返回当前 report 是否由独立 verifier 明确晋升为 verified。"""
        return self.state == self.candidate.status_protocol.verified

    def stable_key(self) -> tuple[int, ...]:
        """返回固定长度子对象引用和完整状态转换键。"""
        candidate_ref = integer_tuple_fingerprint(
            self.candidate.stable_key(), domain=_CANDIDATE_DOMAIN)
        invocation_ref = integer_tuple_fingerprint(
            self.invocation.stable_key(), domain=_INVOCATION_DOMAIN)
        result_ref = integer_tuple_fingerprint(
            _result_audit_key(self.result), domain=_RESULT_DOMAIN)
        return (
            *_packed(candidate_ref),
            *_packed(self.held_out.content_ref()),
            *_packed(invocation_ref),
            *_packed(result_ref),
            *_packed(self.previous_state.stable_key()),
            *_packed(self.state.stable_key()),
        )


__all__ = [
    "CapabilityHeldOutCase",
    "CapabilityVerificationReport",
    "capability_argument_content_key",
    "capability_verification_state",
]
