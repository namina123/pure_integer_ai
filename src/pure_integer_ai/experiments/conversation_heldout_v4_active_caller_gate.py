"""DLG-05 v4 R04b P3-C0 的活动 runtime caller 零写前置门。

本模块只回读已封存的 P3-B evidence，并核验未来 external capsule 与 runtime artifact
root 的路径边界。它不读取 capsule payload、不运行 runtime、不调用 writer，且不创建任何
目录或文件。
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
    ConversationHeldOutV4ActiveRosterReadbackError,
    ConversationHeldOutV4ActiveRosterReadbackResult,
    V4_ACTIVE_ROSTER_READBACK_STATUS_COVERAGE_NE,
    V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY,
    revalidate_v4_active_roster_bidirectional_readback,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import pack_key, strict_integer_tuple


V4_ACTIVE_CALLER_GATE_COUNTS_SCHEMA = 1
V4_ACTIVE_CALLER_GATE_RESULT_SCHEMA = 1

V4_ACTIVE_CALLER_GATE_CALLER_KIND = "v4-runtime-artifact"
V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY = "P3_CALLER_GATE_TEST_ONLY"
V4_ACTIVE_CALLER_GATE_STATUS_DRY_RUN_NE = "P3_CALLER_GATE_DRY_RUN_NE"


# object-model: exception
class ConversationHeldOutV4ActiveCallerGateError(RuntimeError):
    """P3-C0 的 P3-B evidence、caller identity 或零写路径边界不成立。"""


def _fail(message: str) -> None:
    """统一产生不携带 capsule payload、绝对路径或运行状态的 fail-closed 错误。"""
    raise ConversationHeldOutV4ActiveCallerGateError(message)


def _require_stage_name(value: str) -> str:
    """限制 logical stage 为没有路径语义的短 ASCII 标识。"""
    if not isinstance(value, str) or not value or len(value) > 56:
        raise ValueError("P3-C0 logical_stage_name 长度非法")
    if not value[0].isascii() or not value[0].isalnum():
        raise ValueError("P3-C0 logical_stage_name 必须以 ASCII 字母或数字开始")
    if any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
           for item in value):
        raise ValueError("P3-C0 logical_stage_name 含非法字符")
    return value


def _text_scalars(value: str, *, label: str) -> tuple[int, ...]:
    """将有限状态文字转换为规范 Unicode scalar tuple。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} 必须是非空 str")
    result = tuple(ord(item) for item in value)
    if any(item < 1 or 0xD800 <= item <= 0xDFFF or item > 0x10FFFF
           for item in result):
        raise ValueError(f"{label} 含非法 Unicode scalar")
    return result


def _is_reparse(path: Path) -> bool:
    """在 resolve 前检测 Windows reparse，其他平台保持保守的普通文件语义。"""
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(stat, "st_file_attributes", 0) & 0x400)


def _require_absolute_path(value: Path, *, label: str) -> Path:
    """拒绝裸字符串、相对路径和含有父级逃逸语义的 prospective root。"""
    if not isinstance(value, Path):
        raise TypeError(f"{label} 必须是 Path")
    if not value.is_absolute() or ".." in value.parts:
        _fail(f"{label} 必须是不含 .. 的绝对 Path")
    return value


def _require_existing_normal_directory(value: Path, *, label: str) -> Path:
    """逐级拒绝 link/reparse 后核验 source root 或 target parent 为普通目录。"""
    raw = _require_absolute_path(value, label=label)
    for current in (raw, *raw.parents):
        if current.is_symlink() or _is_reparse(current):
            _fail(f"{label} 含链接或 reparse point")
        if not current.exists():
            _fail(f"{label} 不存在")
        if current == current.parent:
            break
    try:
        resolved = raw.resolve()
    except OSError as exc:
        raise ConversationHeldOutV4ActiveCallerGateError(
            f"{label} 无法解析") from exc
    if (not resolved.is_dir() or resolved.is_symlink() or _is_reparse(resolved)):
        _fail(f"{label} 不是普通目录")
    return resolved


def _require_absent_normal_target(value: Path, *, label: str) -> Path:
    """核验 future artifact root 仅为不存在的普通父目录下一个名字，不创建它。"""
    raw = _require_absolute_path(value, label=label)
    if not raw.name:
        _fail(f"{label} 必须有独立目录名")
    if os.path.lexists(raw) or raw.is_symlink() or _is_reparse(raw):
        _fail(f"{label} 必须此前不存在且不是链接")
    parent = _require_existing_normal_directory(raw.parent, label=f"{label} parent")
    target = parent / raw.name
    if os.path.lexists(target) or target.is_symlink() or _is_reparse(target):
        _fail(f"{label} 在普通父目录下已存在或不是普通 prospective root")
    return target


def _require_transport_drive(value: Path, *, test_transport: bool, label: str) -> None:
    """生产 dry-run 只允许 K 盘；测试路径必须显式选择 test transport。"""
    if type(test_transport) is not bool:
        raise TypeError("P3-C0 test_transport 必须是 bool")
    if not test_transport and value.drive.upper() != "K:":
        _fail(f"{label} 生产 root 必须位于 K 盘")


def _require_disjoint(source: Path, target: Path) -> None:
    """拒绝 source/target 相同、嵌套或会使 future writer 覆盖 source 的路径关系。"""
    if (source == target or source.is_relative_to(target)
            or target.is_relative_to(source)):
        _fail("P3-C0 source/target root 不得相同或嵌套")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveCallerGateCounts:
    """P3-C0 的唯一只读 P3-B 回读和所有必须为零的副作用账户。"""

    readback_revalidations: int
    root_boundary_checks: int
    payload_reads: int
    runtime_calls: int
    artifact_writes: int
    candidate_writes: int
    core_writes: int
    memory_writes: int
    companion_writes: int
    use_writes: int
    teacher_calls: int
    private_reads: int
    private_writes: int
    formal_reads: int
    formal_writes: int

    def __post_init__(self) -> None:
        """固定本切片的一个 evidence 回读、两个 root check 与所有零副作用。"""
        if (type(self.readback_revalidations) is not int
                or self.readback_revalidations != 1):
            raise ValueError("P3-C0 readback_revalidations 必须为 1")
        if type(self.root_boundary_checks) is not int or self.root_boundary_checks != 2:
            raise ValueError("P3-C0 root_boundary_checks 必须为 2")
        for label, value in zip(
                ("payload_reads", "runtime_calls", "artifact_writes",
                 "candidate_writes", "core_writes", "memory_writes",
                 "companion_writes", "use_writes", "teacher_calls",
                 "private_reads", "private_writes", "formal_reads",
                 "formal_writes"),
                self.integer_stream()[3:]):
            if type(value) is not int or value != 0:
                raise ValueError(f"P3-C0 {label} 必须为零")

    def integer_stream(self) -> tuple[int, ...]:
        """返回所有实际 effect 账户的规范整数序。"""
        return (
            V4_ACTIVE_CALLER_GATE_COUNTS_SCHEMA,
            self.readback_revalidations,
            self.root_boundary_checks,
            self.payload_reads,
            self.runtime_calls,
            self.artifact_writes,
            self.candidate_writes,
            self.core_writes,
            self.memory_writes,
            self.companion_writes,
            self.use_writes,
            self.teacher_calls,
            self.private_reads,
            self.private_writes,
            self.formal_reads,
            self.formal_writes,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveCallerGateInput:
    """P3-C0 唯一入口的 P3-B evidence、explicit roots 与 caller identity。"""

    readback: ConversationHeldOutV4ActiveRosterReadbackResult
    source_capsule_root: Path
    future_artifact_root: Path
    caller_code_identity: ProtocolKey
    test_transport: bool
    logical_stage_name: str

    def __post_init__(self) -> None:
        """仅接受 typed P3-B result、Path root、ProtocolKey 与显式 transport mode。"""
        if not isinstance(self.readback, ConversationHeldOutV4ActiveRosterReadbackResult):
            raise TypeError("P3-C0 readback 类型错误")
        _require_absolute_path(self.source_capsule_root, label="P3-C0 source_capsule_root")
        _require_absolute_path(self.future_artifact_root, label="P3-C0 future_artifact_root")
        if not isinstance(self.caller_code_identity, ProtocolKey):
            raise TypeError("P3-C0 caller_code_identity 必须是 ProtocolKey")
        if type(self.test_transport) is not bool:
            raise TypeError("P3-C0 test_transport 必须是 bool")
        _require_stage_name(self.logical_stage_name)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveCallerGateResult:
    """P3-C0 纯内存 dry-run evidence；不含 root、capsule 或 runtime payload。"""

    readback_stable_key: tuple[int, ...]
    caller_kind: str
    caller_code_identity: ProtocolKey
    test_transport: bool
    logical_stage_name: str
    counts: ConversationHeldOutV4ActiveCallerGateCounts
    status: str

    def __post_init__(self) -> None:
        """绑定 path-free P3-B identity、caller、模式、零写账户与唯一状态。"""
        try:
            strict_integer_tuple(
                self.readback_stable_key, label="P3-C0 readback_stable_key")
        except (TypeError, ValueError) as exc:
            raise ValueError("P3-C0 readback_stable_key 非法") from exc
        if self.caller_kind != V4_ACTIVE_CALLER_GATE_CALLER_KIND:
            raise ValueError("P3-C0 result caller_kind 未注册")
        if not isinstance(self.caller_code_identity, ProtocolKey):
            raise TypeError("P3-C0 result caller_code_identity 类型错误")
        if type(self.test_transport) is not bool:
            raise TypeError("P3-C0 result test_transport 必须是 bool")
        _require_stage_name(self.logical_stage_name)
        if not isinstance(self.counts, ConversationHeldOutV4ActiveCallerGateCounts):
            raise TypeError("P3-C0 result counts 类型错误")
        expected_status = (
            V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY if self.test_transport
            else V4_ACTIVE_CALLER_GATE_STATUS_DRY_RUN_NE)
        if self.status != expected_status:
            raise ValueError("P3-C0 result status 与 transport 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回无绝对路径、可供后续 P3-C1 显式绑定的 dry-run identity。"""
        result = [V4_ACTIVE_CALLER_GATE_RESULT_SCHEMA]
        for value in (
                self.readback_stable_key,
                _text_scalars(self.caller_kind, label="P3-C0 caller_kind"),
                self.caller_code_identity.components,
                (1 if self.test_transport else 0,),
                _text_scalars(self.logical_stage_name,
                              label="P3-C0 logical_stage_name"),
                self.counts.integer_stream(),
                _text_scalars(self.status, label="P3-C0 result status")):
            pack_key(result, value)
        return tuple(result)


def run_v4_active_caller_zero_write_dry_run(
        value: ConversationHeldOutV4ActiveCallerGateInput,
        ) -> ConversationHeldOutV4ActiveCallerGateResult:
    """执行 P3-C0：回读 P3-B 与路径边界，不读 payload、不调用 runtime 或 writer。

    source root 只被验证为普通目录，future target root 只被验证为不存在的普通 prospective
    directory。函数不会枚举 source、创建 target 或留下任何 publication，因此后续非 dry-run
    必须另行冻结 P3-B 到 typed capsule 的语义映射和实际读取计数。
    """
    if not isinstance(value, ConversationHeldOutV4ActiveCallerGateInput):
        raise TypeError("P3-C0 dry-run 必须接收 ConversationHeldOutV4ActiveCallerGateInput")
    try:
        readback = revalidate_v4_active_roster_bidirectional_readback(value.readback)
    except (ConversationHeldOutV4ActiveRosterReadbackError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveCallerGateError(
            "P3-C0 P3-B evidence readback 未通过") from exc
    if readback.publication_run_root.test_transport != value.test_transport:
        _fail("P3-C0 P3-B transport 与 caller gate mode 不一致")
    expected_readback_status = (
        V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY if value.test_transport
        else V4_ACTIVE_ROSTER_READBACK_STATUS_COVERAGE_NE)
    if readback.receipt.status != expected_readback_status:
        _fail("P3-C0 P3-B receipt status 与 caller gate mode 不一致")

    source = _require_existing_normal_directory(
        value.source_capsule_root, label="P3-C0 source capsule root")
    target = _require_absent_normal_target(
        value.future_artifact_root, label="P3-C0 future artifact root")
    _require_transport_drive(source, test_transport=value.test_transport,
                             label="P3-C0 source capsule root")
    _require_transport_drive(target, test_transport=value.test_transport,
                             label="P3-C0 future artifact root")
    _require_disjoint(source, target)

    counts = ConversationHeldOutV4ActiveCallerGateCounts(
        1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return ConversationHeldOutV4ActiveCallerGateResult(
        readback.stable_key(),
        V4_ACTIVE_CALLER_GATE_CALLER_KIND,
        value.caller_code_identity,
        value.test_transport,
        value.logical_stage_name,
        counts,
        (V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY if value.test_transport
         else V4_ACTIVE_CALLER_GATE_STATUS_DRY_RUN_NE),
    )


__all__ = [
    "ConversationHeldOutV4ActiveCallerGateCounts",
    "ConversationHeldOutV4ActiveCallerGateError",
    "ConversationHeldOutV4ActiveCallerGateInput",
    "ConversationHeldOutV4ActiveCallerGateResult",
    "V4_ACTIVE_CALLER_GATE_CALLER_KIND",
    "V4_ACTIVE_CALLER_GATE_STATUS_DRY_RUN_NE",
    "V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY",
    "run_v4_active_caller_zero_write_dry_run",
]
