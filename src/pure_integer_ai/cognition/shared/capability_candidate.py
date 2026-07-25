"""C-00 由 typed Artifact 示例形成 provisional Capability 候选的领域对象。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.artifact_binding import ArtifactBindingRun
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactAuthority,
    ArtifactSchema,
    FormalArtifact,
    FormalArtifactDefinition,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_EXAMPLE_DOMAIN = "capability.candidate.example.v1"


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(value), *value


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验形成 trace 只含严格整数且非空。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空 tuple")
    assert_int(*value, _where=label)
    if any(type(part) is not int for part in value):
        raise ValueError(f"{label} 必须只含严格整数")
    return value


def _instruction(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验状态或形成理由是一等 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return value


def _definition_shape(
        value: FormalArtifactDefinition,
        ) -> tuple[int, ...]:
    """提取不依赖具体 Variable/程序身份的参数、结果和 proof 类型契约。"""
    result = [len(value.parameters)]
    for parameter in value.parameters:
        result.extend(_packed(parameter.schema.stable_key()))
    for key in (
            value.result_kind.stable_key(),
            value.result_schema.stable_key(),
            value.proof_kind.stable_key(),
            value.proof_schema.stable_key()):
        result.extend(_packed(key))
    return tuple(result)


def _program_payload_key(
        value: FormalArtifactDefinition,
        ) -> tuple[int, ...]:
    """返回解释器上下文中的程序载荷，识别只换 Artifact 身份的复制。"""
    return (
        *_packed(value.program.artifact_kind.stable_key()),
        *_packed(value.program.schema.stable_key()),
        *_packed(value.executor.stable_key()),
        *_packed(value.program.payload),
    )


def _execution_contract_key(
        value: FormalArtifactDefinition,
        ) -> tuple[int, ...]:
    """去除来源、声明和 Variable 身份后提取实际执行契约。"""
    result = [
        *_packed(value.program.artifact_kind.stable_key()),
        *_packed(value.program.schema.stable_key()),
        len(value.parameters),
    ]
    for parameter in value.parameters:
        result.extend(_packed(parameter.schema.stable_key()))
        result.extend(_packed(parameter.executor_binding))
    for key in (
            value.result_kind.stable_key(),
            value.result_schema.stable_key(),
            value.proof_kind.stable_key(),
            value.proof_schema.stable_key(),
            value.executor.stable_key(),
            value.verifier.stable_key()):
        result.extend(_packed(key))
    return tuple(result)


@dataclass(frozen=True)
class CapabilityStatusProtocol:
    """Capability candidate 的 provisional、verified 和 rejected 一等状态。"""

    provisional: ObjectIdentity
    verified: ObjectIdentity
    rejected: ObjectIdentity

    def __post_init__(self) -> None:
        """要求三种状态互异且都来自符号域最小指令。"""
        states = (self.provisional, self.verified, self.rejected)
        if len(set(states)) != len(states):
            raise ValueError("Capability 状态必须互不相同")
        for state in states:
            _instruction(state, label="Capability status")

    def stable_key(self) -> tuple[int, ...]:
        """返回三种注入状态的完整身份键。"""
        return tuple(
            part
            for state in (
                self.provisional, self.verified, self.rejected)
            for part in _packed(state.stable_key())
        )


@dataclass(frozen=True)
class CapabilityArgumentDemonstration:
    """former 可见的单个参数 schema 与实际输入 Artifact。"""

    schema: ArtifactSchema
    value: FormalArtifact

    def __post_init__(self) -> None:
        """只保留形成所需类型和值，不暴露 Variable 或 executor binding。"""
        if not isinstance(self.schema, ArtifactSchema):
            raise TypeError("Capability demonstration schema 类型错误")
        if not isinstance(self.value, FormalArtifact):
            raise TypeError("Capability demonstration value 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回参数 schema 和实际输入 Artifact 的完整稳定键。"""
        return (
            *_packed(self.schema.stable_key()),
            *_packed(self.value.stable_key()),
        )


@dataclass(frozen=True)
class CapabilityDemonstration:
    """former 可见的脱敏输入、独立验证结果、proof 与来源。"""

    source: SourceRef
    arguments: tuple[CapabilityArgumentDemonstration, ...]
    value: FormalArtifact
    proof: FormalArtifact
    result_kind: ObjectIdentity
    result_schema: ArtifactSchema
    proof_kind: ObjectIdentity
    proof_schema: ArtifactSchema
    verifier: ArtifactAuthority
    verification_payload: tuple[int, ...]
    verification_trace: tuple[int, ...]
    example_ref: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验投影完整且不依赖 example program、expected 或执行绑定。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("Capability demonstration source 类型错误")
        if not isinstance(self.arguments, tuple) or not self.arguments:
            raise ValueError("Capability demonstration 至少需要一个参数")
        if any(not isinstance(item, CapabilityArgumentDemonstration)
               for item in self.arguments):
            raise TypeError("Capability demonstration arguments 类型错误")
        if not isinstance(self.value, FormalArtifact):
            raise TypeError("Capability demonstration value 类型错误")
        if not isinstance(self.proof, FormalArtifact):
            raise TypeError("Capability demonstration proof 类型错误")
        if self.value.source != self.source or self.proof.source != self.source:
            raise ValueError("Capability demonstration 结果来源漂移")
        if any(item.value.source != self.source for item in self.arguments):
            raise ValueError("Capability demonstration 参数来源漂移")
        if self.value.artifact_kind != self.result_kind:
            raise ValueError("Capability demonstration result kind 漂移")
        if self.value.schema != self.result_schema:
            raise ValueError("Capability demonstration result schema 漂移")
        if self.proof.artifact_kind != self.proof_kind:
            raise ValueError("Capability demonstration proof kind 漂移")
        if self.proof.schema != self.proof_schema:
            raise ValueError("Capability demonstration proof schema 漂移")
        if not isinstance(self.verifier, ArtifactAuthority):
            raise TypeError("Capability demonstration verifier 类型错误")
        for label, key in (
                ("verification payload", self.verification_payload),
                ("verification trace", self.verification_trace)):
            if not isinstance(key, tuple):
                raise TypeError(f"Capability demonstration {label} 必须是 tuple")
            assert_int(*key, _where=f"Capability demonstration {label}")
            if any(type(part) is not int for part in key):
                raise ValueError(
                    f"Capability demonstration {label} 必须只含严格整数")
        if not isinstance(self.example_ref, tuple) or len(
                self.example_ref) != 34:
            raise ValueError("Capability demonstration example_ref 长度非法")
        assert_int(*self.example_ref, _where="Capability demonstration ref")

    def stable_key(self) -> tuple[int, ...]:
        """返回脱敏 demonstration 的来源、输入、结果和验证证据键。"""
        result = [*_packed(self.source.stable_key()), len(self.arguments)]
        for argument in self.arguments:
            result.extend(_packed(argument.stable_key()))
        for key in (
                self.value.stable_key(),
                self.proof.stable_key(),
                self.result_kind.stable_key(),
                self.result_schema.stable_key(),
                self.proof_kind.stable_key(),
                self.proof_schema.stable_key(),
                self.verifier.stable_key(),
                self.verification_payload,
                self.verification_trace,
                self.example_ref):
            result.extend(_packed(key))
        return tuple(result)


@dataclass(frozen=True)
class CapabilityFormationInput:
    """former 唯一可见的 demonstrations 与调用方注入规格。"""

    demonstrations: tuple[CapabilityDemonstration, ...]
    specifications: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        """要求至少两个脱敏示例，并拒绝重复或非法规格身份。"""
        if not isinstance(self.demonstrations, tuple) or len(
                self.demonstrations) < 2:
            raise ValueError("CapabilityFormationInput 至少需要两个 demonstrations")
        if any(not isinstance(item, CapabilityDemonstration)
               for item in self.demonstrations):
            raise TypeError("CapabilityFormationInput.demonstrations 类型错误")
        if not isinstance(self.specifications, tuple) or any(
                not isinstance(item, ObjectIdentity)
                for item in self.specifications):
            raise TypeError("CapabilityFormationInput.specifications 类型错误")
        if len(set(self.specifications)) != len(self.specifications):
            raise ValueError("CapabilityFormationInput specifications 不得重复")

    def example_refs(self) -> tuple[tuple[int, ...], ...]:
        """按输入顺序返回 runtime 生成的全部 example 内容引用。"""
        return tuple(item.example_ref for item in self.demonstrations)

    def stable_key(self) -> tuple[int, ...]:
        """返回全部脱敏 demonstrations 和开放规格的稳定键。"""
        result = [len(self.demonstrations)]
        for demonstration in self.demonstrations:
            result.extend(_packed(demonstration.stable_key()))
        result.append(len(self.specifications))
        for specification in self.specifications:
            result.extend(_packed(specification.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class CapabilityExample:
    """一个由 A-06 完成 binding、执行和独立 verifier 的审计示例。"""

    run: ArtifactBindingRun

    def __post_init__(self) -> None:
        """C-00 只接受形式成功的 A-06 run，不把失败尝试当正例。"""
        if not isinstance(self.run, ArtifactBindingRun):
            raise TypeError("CapabilityExample.run 类型错误")
        if self.run.succeeded is not True:
            raise ValueError("CapabilityExample 必须来自成功 A-06 run")

    def argument_key(self) -> tuple[int, ...]:
        """返回实际参数 Variable 与 Artifact 值，用于拒绝重复参数示例。"""
        result = [len(self.run.invocation.arguments)]
        for argument in self.run.invocation.arguments:
            result.extend(_packed(argument.parameter.stable_key()))
            result.extend(_packed(argument.value.stable_key()))
        return tuple(result)

    def content_ref(self) -> tuple[int, ...]:
        """对完整 typed run 生成固定长度内容引用，父对象不递归平铺。"""
        return integer_tuple_fingerprint(
            self.run.stable_key(), domain=_EXAMPLE_DOMAIN)

    def demonstration(self) -> CapabilityDemonstration:
        """从完整审计示例投影 former 可见字段，排除 program 和 expected。"""
        definition = self.run.invocation.definition
        result = self.run.result
        if (result.value is None or result.proof is None
                or result.verification is None
                or result.verification.accepted is not True):
            raise ValueError("成功 example 缺少独立验证后的 value/proof")
        arguments = tuple(
            CapabilityArgumentDemonstration(parameter.schema, argument.value)
            for parameter, argument in zip(
                definition.parameters, self.run.invocation.arguments)
        )
        if len(arguments) != len(definition.parameters):
            raise ValueError("Capability example 参数投影不完整")
        return CapabilityDemonstration(
            self.run.invocation.source,
            arguments,
            result.value,
            result.proof,
            definition.result_kind,
            definition.result_schema,
            definition.proof_kind,
            definition.proof_schema,
            result.verification.authority,
            result.verification.payload,
            result.verification.trace,
            self.content_ref(),
        )


@dataclass(frozen=True)
class CapabilityCandidateProposal:
    """former 从全部示例/规格提出的新程序、契约和形成依据。"""

    capability_kind: MemoryLinkedRef
    definition: FormalArtifactDefinition
    source: SourceRef
    preconditions: tuple[ObjectIdentity, ...]
    example_refs: tuple[tuple[int, ...], ...]
    formation_reason: ObjectIdentity
    formation_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 proposal 的一等类型、来源、开放前置条件和完整示例引用。"""
        if not isinstance(self.capability_kind, MemoryLinkedRef):
            raise TypeError("CapabilityCandidateProposal.kind 类型错误")
        if not isinstance(self.definition, FormalArtifactDefinition):
            raise TypeError("CapabilityCandidateProposal.definition 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("CapabilityCandidateProposal.source 类型错误")
        if not isinstance(self.preconditions, tuple) or any(
                not isinstance(item, ObjectIdentity)
                for item in self.preconditions):
            raise TypeError("CapabilityCandidateProposal.preconditions 类型错误")
        if len(set(self.preconditions)) != len(self.preconditions):
            raise ValueError("Capability 前置条件不得重复")
        if not isinstance(self.example_refs, tuple) or len(
                self.example_refs) < 2:
            raise ValueError("Capability proposal 至少引用两个 example")
        for ref in self.example_refs:
            if not isinstance(ref, tuple) or len(ref) != 34:
                raise ValueError("Capability example 内容引用长度非法")
            assert_int(*ref, _where="Capability example ref")
        _instruction(
            self.formation_reason, label="Capability formation reason")
        _strict_key(
            self.formation_trace, label="Capability formation trace")

    def contract_key(self) -> tuple[int, ...]:
        """返回供 C-02 CapabilityPayload 保存的程序契约和前置条件键。"""
        result = [*_packed(self.definition.stable_key()), len(self.preconditions)]
        for precondition in self.preconditions:
            result.extend(_packed(precondition.stable_key()))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 kind、契约、来源、示例引用和形成依据的完整键。"""
        result = [
            *_packed(self.capability_kind.stable_key()),
            *_packed(self.contract_key()),
            *_packed(self.source.stable_key()),
            len(self.example_refs),
        ]
        for ref in self.example_refs:
            result.extend(_packed(ref))
        result.extend((
            *_packed(self.formation_reason.stable_key()),
            *_packed(self.formation_trace),
        ))
        return tuple(result)


@dataclass(frozen=True)
class CapabilityFormationRequest:
    """C-00 examples、已召回 program 集和状态协议组成的形成请求。"""

    examples: tuple[CapabilityExample, ...]
    recalled_programs: tuple[FormalArtifactDefinition, ...]
    status_protocol: CapabilityStatusProtocol
    specifications: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        """要求至少两个参数不同的成功示例，并核验召回 program 身份。"""
        if not isinstance(self.examples, tuple) or len(self.examples) < 2:
            raise ValueError("CapabilityFormationRequest 至少需要两个 examples")
        if any(not isinstance(item, CapabilityExample)
               for item in self.examples):
            raise TypeError("CapabilityFormationRequest.examples 类型错误")
        argument_keys = tuple(item.argument_key() for item in self.examples)
        if len(set(argument_keys)) != len(argument_keys):
            raise ValueError("Capability examples 必须使用不同参数")
        if not isinstance(self.recalled_programs, tuple) or any(
                not isinstance(item, FormalArtifactDefinition)
                for item in self.recalled_programs):
            raise ValueError("recalled_programs 必须只含完整 program definition")
        recalled_keys = tuple(
            item.stable_key() for item in self.recalled_programs)
        if len(set(recalled_keys)) != len(recalled_keys):
            raise ValueError("recalled_programs 不得重复")
        if not isinstance(self.status_protocol, CapabilityStatusProtocol):
            raise TypeError("status_protocol 类型错误")
        if not isinstance(self.specifications, tuple) or any(
                not isinstance(item, ObjectIdentity)
                for item in self.specifications):
            raise TypeError("Capability specifications 类型错误")
        if len(set(self.specifications)) != len(self.specifications):
            raise ValueError("Capability specifications 不得重复")

    def example_refs(self) -> tuple[tuple[int, ...], ...]:
        """按请求证据顺序返回全部 example 固定长度内容引用。"""
        return tuple(item.content_ref() for item in self.examples)

    def formation_input(self) -> CapabilityFormationInput:
        """构造不含 example program、expected 和召回内容的 former 输入。"""
        return CapabilityFormationInput(
            tuple(item.demonstration() for item in self.examples),
            self.specifications,
        )


class CapabilityCandidateFormer(Protocol):
    """从脱敏 demonstrations 提出候选定义和形成依据的注入式 former。"""

    def propose(
            self, formation_input: CapabilityFormationInput,
            ) -> CapabilityCandidateProposal:
        """提出 provisional 候选；runtime 仍会独立执行反召回和契约核验。"""
        ...


@dataclass(frozen=True)
class CapabilityCandidate:
    """尚未经过 C-01 未见参数 held-out 的 provisional Capability。"""

    proposal: CapabilityCandidateProposal
    examples: tuple[CapabilityExample, ...]
    state: ObjectIdentity
    status_protocol: CapabilityStatusProtocol

    def __post_init__(self) -> None:
        """C-00 候选只能处于注入协议的 provisional 状态。"""
        if not isinstance(self.proposal, CapabilityCandidateProposal):
            raise TypeError("CapabilityCandidate.proposal 类型错误")
        if not isinstance(self.examples, tuple) or len(self.examples) < 2:
            raise ValueError("CapabilityCandidate.examples 不完整")
        if any(not isinstance(item, CapabilityExample) for item in self.examples):
            raise TypeError("CapabilityCandidate.examples 类型错误")
        if not isinstance(self.status_protocol, CapabilityStatusProtocol):
            raise TypeError("CapabilityCandidate.status_protocol 类型错误")
        if self.state != self.status_protocol.provisional:
            raise ValueError("C-00 只能形成 provisional Capability candidate")
        if self.proposal.example_refs != tuple(
                item.content_ref() for item in self.examples):
            raise ValueError("CapabilityCandidate proposal 未引用全部当前 examples")

    def stable_key(self) -> tuple[int, ...]:
        """返回 proposal、example 内容引用、状态与状态协议键。"""
        result = [
            *_packed(self.proposal.stable_key()),
            len(self.examples),
        ]
        for example in self.examples:
            result.extend(_packed(example.content_ref()))
        result.extend((
            *_packed(self.state.stable_key()),
            *_packed(self.status_protocol.stable_key()),
        ))
        return tuple(result)


class CapabilityFormationRuntime:
    """调用 former 并独立拒绝程序召回、示例遗漏和契约漂移。"""

    def __init__(self, former: CapabilityCandidateFormer) -> None:
        """绑定只负责提出 proposal 的注入式 former。"""
        if not hasattr(former, "propose"):
            raise TypeError("former 必须实现 CapabilityCandidateFormer")
        self.former = former

    def form(self, request: CapabilityFormationRequest) -> CapabilityCandidate:
        """形成 provisional candidate，不执行 held-out、Memory 写入或 Use。"""
        if not isinstance(request, CapabilityFormationRequest):
            raise TypeError("request 必须是 CapabilityFormationRequest")
        formation_input = request.formation_input()
        proposal = self.former.propose(formation_input)
        if not isinstance(proposal, CapabilityCandidateProposal):
            raise TypeError("former 必须返回 CapabilityCandidateProposal")
        if proposal.example_refs != formation_input.example_refs():
            raise ValueError("Capability proposal 未逐项引用当前全部 examples")

        candidate_program = proposal.definition.program.identity
        candidate_payload = _program_payload_key(proposal.definition)
        candidate_contract = _execution_contract_key(proposal.definition)
        existing_groups = (
            (
                "example",
                tuple(item.run.invocation.definition for item in request.examples),
            ),
            ("已召回", request.recalled_programs),
        )
        for label, definitions in existing_groups:
            for definition in definitions:
                if candidate_program == definition.program.identity:
                    raise ValueError(
                        f"{label} 中已有 program 不能冒充新 induction candidate")
                if candidate_payload == _program_payload_key(definition):
                    raise ValueError(
                        f"{label} program payload 不能换身份冒充 induction")
                if candidate_contract == _execution_contract_key(definition):
                    raise ValueError(
                        f"{label} program 执行契约不能换身份冒充 induction")

        expected_shape = _definition_shape(proposal.definition)
        for example in request.examples:
            if _definition_shape(
                    example.run.invocation.definition) != expected_shape:
                raise ValueError("Capability candidate 与 example 类型契约漂移")
        return CapabilityCandidate(
            proposal,
            request.examples,
            request.status_protocol.provisional,
            request.status_protocol,
        )


__all__ = [
    "CapabilityArgumentDemonstration",
    "CapabilityCandidate",
    "CapabilityCandidateFormer",
    "CapabilityCandidateProposal",
    "CapabilityDemonstration",
    "CapabilityExample",
    "CapabilityFormationInput",
    "CapabilityFormationRequest",
    "CapabilityFormationRuntime",
    "CapabilityStatusProtocol",
]
