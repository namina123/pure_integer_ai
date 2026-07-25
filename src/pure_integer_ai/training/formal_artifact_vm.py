"""S-06 typed Artifact 到受限 VM 和有理数独立 verifier 的 adapter。

通用 Artifact 桥不认识 opcode 或 Rational。本模块解释 program/value payload，按参数
声明中的 executor_binding 建 VM 环境，并复用既有 step limit。verifier 只比较独立
expected Artifact，不把执行器自己的输出当 expected。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.formal_artifact import ArtifactAuthority
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactExecutionObservation,
    ArtifactExecutionRequest,
    ArtifactVerificationObservation,
    ArtifactVerificationRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.rational import eq, make, mul
from pure_integer_ai.vm.graph_compile import Instruction
from pure_integer_ai.vm.vm_core import (
    DEFAULT_STEP_LIMIT,
    StepLimitExceeded,
    execute,
)


_VM_PROGRAM_CODEC_VERSION = 1


def _strict_int(value: int, *, label: str, nonnegative: bool = False) -> int:
    """校验 VM codec 的严格整数头字段。"""
    assert_int(value, _where=label)
    if type(value) is not int:
        raise ValueError(f"{label} 必须是严格整数")
    if nonnegative and value < 0:
        raise ValueError(f"{label} 必须是非负整数")
    return value


def _reason(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验 adapter 失败原因是一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def encode_vm_program(
        instructions: tuple[Instruction, ...] | list[Instruction],
        ) -> tuple[int, ...]:
    """把 VM 指令序列编码成 Artifact 的无歧义纯整数 payload。"""
    if not isinstance(instructions, (tuple, list)):
        raise TypeError("instructions 必须是 Instruction 序列")
    payload: list[int] = [_VM_PROGRAM_CODEC_VERSION, len(instructions)]
    for instruction in instructions:
        if not isinstance(instruction, Instruction):
            raise TypeError("instructions 必须只含 Instruction")
        payload.extend((instruction.opcode, len(instruction.args)))
        payload.extend(instruction.args)
    return tuple(payload)


def decode_vm_program(payload: tuple[int, ...]) -> tuple[Instruction, ...]:
    """从 Artifact payload 恢复 VM 指令，拒绝截断、尾随和错误 codec 版本。"""
    if not isinstance(payload, tuple) or len(payload) < 2:
        raise ValueError("VM program payload 长度非法")
    assert_int(*payload, _where="decode_vm_program.payload")
    if any(type(value) is not int for value in payload):
        raise ValueError("VM program payload 必须使用严格整数")
    if payload[0] != _VM_PROGRAM_CODEC_VERSION:
        raise ValueError("VM program payload codec 版本非法")
    count = _strict_int(
        payload[1], label="VM program instruction count", nonnegative=True)
    cursor = 2
    instructions: list[Instruction] = []
    for _ in range(count):
        if cursor + 2 > len(payload):
            raise ValueError("VM program instruction header 被截断")
        opcode = payload[cursor]
        arg_count = _strict_int(
            payload[cursor + 1],
            label="VM program argument count",
            nonnegative=True,
        )
        cursor += 2
        if cursor + arg_count > len(payload):
            raise ValueError("VM program instruction arguments 被截断")
        instructions.append(Instruction(
            opcode, payload[cursor:cursor + arg_count]))
        cursor += arg_count
    if cursor != len(payload):
        raise ValueError("VM program payload 含尾随数据")
    return tuple(instructions)


@dataclass(frozen=True)
class RestrictedVMExecutor:
    """以固定 authority 和 step limit 执行已绑定 program Artifact。"""

    authority: ArtifactAuthority
    failure_reason: ObjectIdentity
    step_limit: int = DEFAULT_STEP_LIMIT

    def __post_init__(self) -> None:
        if not isinstance(self.authority, ArtifactAuthority):
            raise TypeError("VM executor authority 必须是 ArtifactAuthority")
        _reason(self.failure_reason, label="VM executor failure_reason")
        _strict_int(
            self.step_limit, label="VM executor step_limit", nonnegative=True)

    def execute(
            self, request: ArtifactExecutionRequest,
            ) -> ArtifactExecutionObservation:
        """解码 program 和 Rational 参数后调用既有受限 VM，失败显式返回 unknown 路径。"""
        if not isinstance(request, ArtifactExecutionRequest):
            raise TypeError("request 必须是 ArtifactExecutionRequest")
        try:
            instructions = decode_vm_program(
                request.invocation.definition.program.payload)
            env = {}
            for bound in request.arguments:
                binding = bound.parameter.executor_binding
                if len(binding) != 1:
                    raise ValueError("VM parameter executor_binding 必须含一个 symbol id")
                payload = bound.argument.value.payload
                if len(payload) != 2:
                    raise ValueError("VM argument payload 必须是 Rational 二元组")
                value = make(payload[0], payload[1])
                conversion = bound.unit_adapter_payload
                if conversion:
                    if len(conversion) != 2:
                        raise ValueError("VM unit adapter payload 必须是 Rational 二元组")
                    value = mul(value, make(conversion[0], conversion[1]))
                env[binding[0]] = value
            value = execute(
                list(instructions), env, step_limit=self.step_limit)
        except (StepLimitExceeded, KeyError, IndexError, ValueError,
                ZeroDivisionError):
            return ArtifactExecutionObservation(
                self.authority,
                request.invocation.source,
                request.invocation.scope,
                False,
                (),
                (self.step_limit,),
                self.failure_reason,
            )
        return ArtifactExecutionObservation(
            self.authority,
            request.invocation.source,
            request.invocation.scope,
            True,
            (value.num, value.den),
            (len(instructions), self.step_limit),
        )


@dataclass(frozen=True)
class RationalEqualityVerifier:
    """用独立 expected Artifact 对 VM Rational 结果做精确交叉积验证。"""

    authority: ArtifactAuthority
    missing_expected_reason: ObjectIdentity
    malformed_value_reason: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.authority, ArtifactAuthority):
            raise TypeError("Rational verifier authority 必须是 ArtifactAuthority")
        _reason(
            self.missing_expected_reason,
            label="Rational verifier missing_expected_reason",
        )
        _reason(
            self.malformed_value_reason,
            label="Rational verifier malformed_value_reason",
        )

    def verify(
            self, request: ArtifactVerificationRequest,
            ) -> ArtifactVerificationObservation:
        """精确比较执行值和独立规格；缺规格或载荷损坏时保持 unknown。"""
        if not isinstance(request, ArtifactVerificationRequest):
            raise TypeError("request 必须是 ArtifactVerificationRequest")
        expected = request.invocation.expected
        if expected is None:
            return ArtifactVerificationObservation(
                self.authority,
                request.invocation.source,
                request.invocation.scope,
                None,
                (),
                (),
                self.missing_expected_reason,
            )
        try:
            if len(request.result.payload) != 2 or len(expected.payload) != 2:
                raise ValueError("Rational payload 长度非法")
            actual_value = make(
                request.result.payload[0], request.result.payload[1])
            expected_value = make(expected.payload[0], expected.payload[1])
        except (ValueError, ZeroDivisionError):
            return ArtifactVerificationObservation(
                self.authority,
                request.invocation.source,
                request.invocation.scope,
                None,
                (),
                (),
                self.malformed_value_reason,
            )
        accepted = eq(actual_value, expected_value)
        return ArtifactVerificationObservation(
            self.authority,
            request.invocation.source,
            request.invocation.scope,
            accepted,
            (
                actual_value.num,
                actual_value.den,
                expected_value.num,
                expected_value.den,
            ),
            (),
        )


__all__ = [
    "RationalEqualityVerifier",
    "RestrictedVMExecutor",
    "decode_vm_program",
    "encode_vm_program",
]
