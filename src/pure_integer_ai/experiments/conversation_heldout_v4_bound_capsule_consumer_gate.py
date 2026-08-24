"""DLG-05 v4 R04b P3-C2 的 bound capsule consumer 零写门。

本模块只接受已经回读的 C1c adapter result 与 P3-C0 caller gate input。它把 C1c
descriptor 的固定 child 精确绑定到 P3-C0 source root，然后执行既有的零写 dry-run。
它不暴露 capsule 路径或正文，也不直接读取 capsule、调用 runtime 或创建 artifact。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_heldout_v4_active_caller_gate import (
    ConversationHeldOutV4ActiveCallerGateError,
    ConversationHeldOutV4ActiveCallerGateInput,
    ConversationHeldOutV4ActiveCallerGateResult,
    V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY,
    run_v4_active_caller_zero_write_dry_run,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_capsule_adapter import (
    ConversationHeldOutV4RuntimeTaskCapsuleAdapterError,
    ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult,
    revalidate_v4_runtime_task_capsule_adapter,
)
from pure_integer_ai.storage.integer_codec import pack_key, strict_integer_tuple


V4_BOUND_CAPSULE_CONSUMER_GATE_COUNTS_SCHEMA = 1
V4_BOUND_CAPSULE_CONSUMER_GATE_RESULT_SCHEMA = 1

V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY = (
    "P3_BOUND_CAPSULE_CONSUMER_GATE_TEST_ONLY")


# object-model: exception
class ConversationHeldOutV4BoundCapsuleConsumerGateError(RuntimeError):
    """P3-C2 的 C1c capability、P3-B binding 或零写 caller gate 不成立。"""


def _fail(message: str) -> None:
    """统一产生不包含正文、root 或运行状态的 fail-closed 错误。"""
    raise ConversationHeldOutV4BoundCapsuleConsumerGateError(message)


def _text_scalars(value: str, *, label: str) -> tuple[int, ...]:
    """把有限状态文本转换为规范 Unicode scalar tuple。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} 必须是非空 str")
    result = tuple(ord(item) for item in value)
    if any(item < 1 or 0xD800 <= item <= 0xDFFF or item > 0x10FFFF
           for item in result):
        raise ValueError(f"{label} 含非法 Unicode scalar")
    return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BoundCapsuleConsumerGateCounts:
    """P3-C2 的一次 C1c 重验、三项 binding 与一次 P3-C0 dry-run 账户。"""

    adapter_revalidations: int
    descriptor_source_root_matches: int
    readback_matches: int
    transport_matches: int
    caller_dry_runs: int

    def __post_init__(self) -> None:
        """固定本切片的唯一允许控制流；所有计数必须各为一次。"""
        for label, item in zip(
                ("adapter_revalidations", "descriptor_source_root_matches",
                 "readback_matches", "transport_matches", "caller_dry_runs"),
                self.integer_stream()[1:]):
            if type(item) is not int or item != 1:
                raise ValueError(f"P3-C2 {label} 必须为 1")

    def integer_stream(self) -> tuple[int, ...]:
        """返回所有实际 gate 检查的规范整数序。"""
        return (
            V4_BOUND_CAPSULE_CONSUMER_GATE_COUNTS_SCHEMA,
            self.adapter_revalidations,
            self.descriptor_source_root_matches,
            self.readback_matches,
            self.transport_matches,
            self.caller_dry_runs,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BoundCapsuleConsumerGateInput:
    """P3-C2 唯一入口：C1c bound capability 加上 P3-C0 typed caller input。"""

    adapter_result: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult
    active_caller_gate_input: ConversationHeldOutV4ActiveCallerGateInput

    def __post_init__(self) -> None:
        """拒绝裸 capsule root、裸 C1c descriptor 或未类型化 caller 参数。"""
        if not isinstance(self.adapter_result,
                          ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult):
            raise TypeError("P3-C2 adapter_result 类型错误")
        if not isinstance(self.active_caller_gate_input,
                          ConversationHeldOutV4ActiveCallerGateInput):
            raise TypeError("P3-C2 active_caller_gate_input 类型错误")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BoundCapsuleConsumerGateResult:
    """P3-C2 的 payload-free bound gate evidence；不保留任何路径或 capsule。"""

    adapter_stable_key: tuple[int, ...]
    capsule_descriptor_stable_key: tuple[int, ...]
    active_caller_gate_result: ConversationHeldOutV4ActiveCallerGateResult
    counts: ConversationHeldOutV4BoundCapsuleConsumerGateCounts
    status: str

    def __post_init__(self) -> None:
        """绑定 path-free C1c identity、P3-C0 result 与唯一 test-only 状态。"""
        for label, item in (
                ("adapter_stable_key", self.adapter_stable_key),
                ("capsule_descriptor_stable_key",
                 self.capsule_descriptor_stable_key)):
            try:
                strict_integer_tuple(item, label=f"P3-C2 {label}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"P3-C2 {label} 非法") from exc
        if not isinstance(self.active_caller_gate_result,
                          ConversationHeldOutV4ActiveCallerGateResult):
            raise TypeError("P3-C2 active_caller_gate_result 类型错误")
        if not isinstance(self.counts, ConversationHeldOutV4BoundCapsuleConsumerGateCounts):
            raise TypeError("P3-C2 counts 类型错误")
        if (not self.active_caller_gate_result.test_transport
                or self.active_caller_gate_result.status
                != V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY
                or self.status != V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY):
            raise ValueError("P3-C2 result transport 或 status 不成立")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 root、capsule 或 payload 的跨阶段稳定 identity。"""
        result = [V4_BOUND_CAPSULE_CONSUMER_GATE_RESULT_SCHEMA]
        for value in (
                self.adapter_stable_key,
                self.capsule_descriptor_stable_key,
                self.active_caller_gate_result.stable_key(),
                self.counts.integer_stream(),
                _text_scalars(self.status, label="P3-C2 result status")):
            pack_key(result, value)
        return tuple(result)


def _require_bound_input(
        adapter: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult,
        caller_input: ConversationHeldOutV4ActiveCallerGateInput,
        ) -> None:
    """核验 C1c descriptor、P3-B evidence 和 transport 是同一个 bound capability。"""
    expected_source_root = (
        adapter.capsule_descriptor.parent_run_root.path
        / adapter.capsule_descriptor.relative_root)
    if caller_input.source_capsule_root != expected_source_root:
        _fail("P3-C2 source capsule root 未精确绑定 C1c descriptor child")
    try:
        caller_readback_key = caller_input.readback.stable_key()
        adapter_readback_key = adapter.readback.stable_key()
    except (TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundCapsuleConsumerGateError(
            "P3-C2 P3-B stable identity 无法比较") from exc
    if caller_readback_key != adapter_readback_key:
        _fail("P3-C2 P3-B readback 未绑定 C1c adapter")
    expected_transport = adapter.capsule_descriptor.parent_run_root.test_transport
    if (caller_input.test_transport != expected_transport
            or adapter.publication_run_root.test_transport != expected_transport
            or adapter.readback.publication_run_root.test_transport
            != expected_transport):
        _fail("P3-C2 C1c/P3-B/caller transport 不一致")


def run_v4_bound_capsule_consumer_zero_write_gate(
        value: ConversationHeldOutV4BoundCapsuleConsumerGateInput,
        ) -> ConversationHeldOutV4BoundCapsuleConsumerGateResult:
    """执行 P3-C2：先回读 C1c，再绑定 P3-C0 source root 并运行零写 dry-run。

    C1c revalidation 负责受预算的 capsule 物理核验；本模块只消费其 payload-free
    result，且绝不将 root 或 capsule 交还给调用方。它仍不执行任何 runtime 或 writer。
    """
    if not isinstance(value, ConversationHeldOutV4BoundCapsuleConsumerGateInput):
        raise TypeError("P3-C2 gate 必须接收 BoundCapsuleConsumerGateInput")
    try:
        adapter = revalidate_v4_runtime_task_capsule_adapter(value.adapter_result)
    except (ConversationHeldOutV4RuntimeTaskCapsuleAdapterError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundCapsuleConsumerGateError(
            "P3-C2 C1c adapter revalidation 未通过") from exc
    _require_bound_input(adapter, value.active_caller_gate_input)
    try:
        caller_result = run_v4_active_caller_zero_write_dry_run(
            value.active_caller_gate_input)
    except (ConversationHeldOutV4ActiveCallerGateError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4BoundCapsuleConsumerGateError(
            "P3-C2 P3-C0 zero-write dry-run 未通过") from exc
    if (caller_result.readback_stable_key != adapter.readback.stable_key()
            or caller_result.caller_code_identity
            != value.active_caller_gate_input.caller_code_identity
            or caller_result.logical_stage_name
            != value.active_caller_gate_input.logical_stage_name
            or not caller_result.test_transport
            or caller_result.status != V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY):
        _fail("P3-C2 P3-C0 result 未保持 C1c binding")
    counts = ConversationHeldOutV4BoundCapsuleConsumerGateCounts(1, 1, 1, 1, 1)
    return ConversationHeldOutV4BoundCapsuleConsumerGateResult(
        adapter.stable_key(),
        adapter.capsule_descriptor.stable_key(),
        caller_result,
        counts,
        V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY,
    )


__all__ = [
    "ConversationHeldOutV4BoundCapsuleConsumerGateCounts",
    "ConversationHeldOutV4BoundCapsuleConsumerGateError",
    "ConversationHeldOutV4BoundCapsuleConsumerGateInput",
    "ConversationHeldOutV4BoundCapsuleConsumerGateResult",
    "V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY",
    "run_v4_bound_capsule_consumer_zero_write_gate",
]
