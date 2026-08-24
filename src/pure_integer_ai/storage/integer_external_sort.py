"""K run capability 上的有界规范整数 framed stream 外排。

本模块只理解 P0 ``IntegerFramedStream`` 的 record tuple 和其封存 footer。它不解释
record 的字段、来源、训练、运行时或任何上层领域对象。调用方以稳定的 logical stage
name 承担排序键语义；``sort_key`` 必须是确定性的纯回调，并为每一条 record 返回严格整数
tuple。

临时 run 与最终输出都只能由 ``KRunRoot`` capability 通过排他创建得到。发生失败时，
本模块不会删除、覆盖、重命名或复用任何残片；未返回 result 的输出不构成已验证结果。
它继承 K boundary 对受控 ACL 的前提：标准库在 Windows 上不能单独证明 hostile 并发
junction 替换的全路径原子安全，故这里只报告可观测漂移并保持 fail-closed。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import os
from pathlib import Path
from typing import BinaryIO, Callable, Iterable

from pure_integer_ai.storage.integer_codec import (
    INTEGER_FRAMED_STREAM_MAGIC,
    INTEGER_FRAMED_STREAM_SCHEMA,
    IntegerFramedStreamError,
    IntegerFramedStreamFooter,
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
    encode_integer_tuple,
    encoded_integer_tuple_size,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunFileDigest,
    KRunFileIdentity,
    KRunRoot,
    capture_plain_file_identity,
    ensure_normal_relative_directory,
    open_exclusive_binary,
    open_plain_binary,
    relative_path,
    require_disjoint_run_roots,
    require_plain_file_identity,
)


_TEMPORARY_NAMESPACE = "integer-external-sort"
_SHA256_BYTE_COUNT = hashlib.sha256().digest_size
_MAX_STAGE_NAME_LENGTH = 96
_MAX_PATH_COUNTER = 99_999_999

INTEGER_EXTERNAL_SORT_BUDGET_SCHEMA = 1
INTEGER_EXTERNAL_SORT_INPUT_IDENTITY_SCHEMA = 1
INTEGER_EXTERNAL_SORT_IDENTITY_SCHEMA = 1
INTEGER_EXTERNAL_SORT_RESULT_SCHEMA = 1


# object-model: exception
class IntegerExternalSortError(RuntimeError):
    """外排输入、键、封存身份或固定输出合同不成立。"""


# object-model: exception
class IntegerExternalSortBudgetExceeded(IntegerExternalSortError):
    """外排的输入、批次、临时文件、输出或打开文件预算被超过。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class IntegerExternalSortBudget:
    """一次外排允许消费的全部有限物理与内存预算。

    ``max_batch_*`` 限制内存中等待排序的 record 和已验证 sort key 的规范编码规模；
    ``max_temporary_*`` 是所有新写临时 run 的累计上限，而不是清理阈值。输出记录数与
    payload 必须等于已接受输入，因此受 ``max_input_*`` 同时约束。
    """

    max_input_file_count: int
    max_input_physical_bytes: int
    max_input_record_count: int
    max_input_payload_bytes: int
    max_record_payload_bytes: int
    max_batch_record_count: int
    max_batch_payload_bytes: int
    max_batch_sort_key_bytes: int
    max_temporary_run_count: int
    max_temporary_record_count: int
    max_temporary_payload_bytes: int
    max_temporary_physical_bytes: int
    max_output_physical_bytes: int
    merge_fan_in: int
    max_open_files: int
    max_merge_pass_count: int

    def __post_init__(self) -> None:
        """拒绝隐式整数、零预算和无法满足有界归并句柄数的配置。"""
        values = (
            ("max_input_file_count", self.max_input_file_count),
            ("max_input_physical_bytes", self.max_input_physical_bytes),
            ("max_input_record_count", self.max_input_record_count),
            ("max_input_payload_bytes", self.max_input_payload_bytes),
            ("max_record_payload_bytes", self.max_record_payload_bytes),
            ("max_batch_record_count", self.max_batch_record_count),
            ("max_batch_payload_bytes", self.max_batch_payload_bytes),
            ("max_batch_sort_key_bytes", self.max_batch_sort_key_bytes),
            ("max_temporary_run_count", self.max_temporary_run_count),
            ("max_temporary_record_count", self.max_temporary_record_count),
            ("max_temporary_payload_bytes", self.max_temporary_payload_bytes),
            ("max_temporary_physical_bytes", self.max_temporary_physical_bytes),
            ("max_output_physical_bytes", self.max_output_physical_bytes),
            ("merge_fan_in", self.merge_fan_in),
            ("max_open_files", self.max_open_files),
            ("max_merge_pass_count", self.max_merge_pass_count),
        )
        for label, value in values:
            _require_positive_integer(value, label=label)
        if self.max_input_file_count > _MAX_PATH_COUNTER:
            raise ValueError("max_input_file_count 超过规范路径计数上限")
        if self.max_temporary_run_count > _MAX_PATH_COUNTER:
            raise ValueError("max_temporary_run_count 超过规范路径计数上限")
        if self.max_merge_pass_count > _MAX_PATH_COUNTER:
            raise ValueError("max_merge_pass_count 超过规范路径计数上限")
        if self.max_record_payload_bytes > self.max_input_payload_bytes:
            raise ValueError("max_record_payload_bytes 不得超过输入 payload 预算")
        if self.max_record_payload_bytes > self.max_batch_payload_bytes:
            raise ValueError("max_record_payload_bytes 不得超过 batch payload 预算")
        if self.max_batch_record_count > self.max_input_record_count:
            raise ValueError("max_batch_record_count 不得超过输入 record 预算")
        if self.merge_fan_in < 2:
            raise ValueError("merge_fan_in 至少为 2")
        if self.max_open_files < self.merge_fan_in + 1:
            raise ValueError("max_open_files 必须覆盖 fan-in readers 和一个 writer")

    def integer_stream(self) -> tuple[int, ...]:
        """返回含全部资源上限的版本化规范整数身份。"""
        return (
            INTEGER_EXTERNAL_SORT_BUDGET_SCHEMA,
            self.max_input_file_count,
            self.max_input_physical_bytes,
            self.max_input_record_count,
            self.max_input_payload_bytes,
            self.max_record_payload_bytes,
            self.max_batch_record_count,
            self.max_batch_payload_bytes,
            self.max_batch_sort_key_bytes,
            self.max_temporary_run_count,
            self.max_temporary_record_count,
            self.max_temporary_payload_bytes,
            self.max_temporary_physical_bytes,
            self.max_output_physical_bytes,
            self.merge_fan_in,
            self.max_open_files,
            self.max_merge_pass_count,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回不省略任何预算字段的可审计稳定键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class IntegerExternalSortInputIdentity:
    """一个完整消费并在读取前后复核过的输入 framed stream 身份。"""

    relative_path: Path
    footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """固定词法相对路径、完整 P0 footer 与物理字节身份。"""
        _validate_relative_path(self.relative_path, label="input identity relative_path")
        if not isinstance(self.footer, IntegerFramedStreamFooter):
            raise TypeError("input identity footer 必须是 IntegerFramedStreamFooter")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("input identity physical 必须是 KRunFileDigest")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 root 的输入路径、P0 footer 与 physical 身份。"""
        result = [INTEGER_EXTERNAL_SORT_INPUT_IDENTITY_SCHEMA]
        pack_key(result, _relative_path_integer_key(self.relative_path))
        pack_key(result, self.footer.integer_tuple())
        result.append(self.physical.byte_count)
        pack_key(result, self.physical.sha256)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回输入内容与相对定位的完整规范身份。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class IntegerExternalSortIdentity:
    """排序语义阶段、全部输入身份及固定输出的可审计身份。"""

    logical_stage_name: str
    input_identities: tuple[IntegerExternalSortInputIdentity, ...]
    output_relative_path: Path
    output_footer: IntegerFramedStreamFooter
    output_physical: KRunFileDigest

    def __post_init__(self) -> None:
        """拒绝缺失输入、非规范阶段名和未封存的输出身份字段。"""
        _validate_stage_name(self.logical_stage_name)
        if (not isinstance(self.input_identities, tuple)
                or not self.input_identities
                or any(not isinstance(item, IntegerExternalSortInputIdentity)
                       for item in self.input_identities)):
            raise ValueError("input_identities 必须是非空 input identity tuple")
        _validate_relative_path(
            self.output_relative_path,
            label="external sort output_relative_path",
        )
        if not isinstance(self.output_footer, IntegerFramedStreamFooter):
            raise TypeError("output_footer 必须是 IntegerFramedStreamFooter")
        if not isinstance(self.output_physical, KRunFileDigest):
            raise TypeError("output_physical 必须是 KRunFileDigest")

    def integer_stream(self) -> tuple[int, ...]:
        """返回排序阶段、全部输入与固定输出的完整规范身份。"""
        result = [INTEGER_EXTERNAL_SORT_IDENTITY_SCHEMA]
        pack_key(result, _stage_name_integer_key(self.logical_stage_name))
        result.append(len(self.input_identities))
        for item in self.input_identities:
            pack_key(result, item.integer_stream())
        pack_key(result, _relative_path_integer_key(self.output_relative_path))
        pack_key(result, self.output_footer.integer_tuple())
        result.append(self.output_physical.byte_count)
        pack_key(result, self.output_physical.sha256)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回不含输入或输出绝对 root 的完整排序身份。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class IntegerExternalSortResult:
    """成功外排的身份、临时物化累计量与归并句柄配置回执。"""

    identity: IntegerExternalSortIdentity
    temporary_run_count: int
    temporary_record_count: int
    temporary_payload_bytes: int
    temporary_physical_bytes: int
    initial_run_count: int
    merge_pass_count: int
    merge_fan_in: int
    max_open_files: int
    budget: IntegerExternalSortBudget

    def __post_init__(self) -> None:
        """固定所有累计整数与实际采用的 bounded merge 配置。"""
        if not isinstance(self.identity, IntegerExternalSortIdentity):
            raise TypeError("identity 必须是 IntegerExternalSortIdentity")
        for label, value in (
            ("temporary_run_count", self.temporary_run_count),
            ("temporary_record_count", self.temporary_record_count),
            ("temporary_payload_bytes", self.temporary_payload_bytes),
            ("temporary_physical_bytes", self.temporary_physical_bytes),
            ("initial_run_count", self.initial_run_count),
            ("merge_pass_count", self.merge_pass_count),
        ):
            _require_nonnegative_integer(value, label=label)
        _require_positive_integer(self.merge_fan_in, label="merge_fan_in")
        _require_positive_integer(self.max_open_files, label="max_open_files")
        if not isinstance(self.budget, IntegerExternalSortBudget):
            raise TypeError("budget 必须是 IntegerExternalSortBudget")
        if self.merge_fan_in < 2:
            raise ValueError("merge_fan_in 至少为 2")
        if self.max_open_files < self.merge_fan_in + 1:
            raise ValueError("max_open_files 必须覆盖 fan-in readers 和一个 writer")
        if (self.merge_fan_in != self.budget.merge_fan_in
                or self.max_open_files != self.budget.max_open_files):
            raise ValueError("result 的 fan-in/max-open 必须与 budget 精确一致")

    def integer_stream(self) -> tuple[int, ...]:
        """返回可供 cursor 或 receipt 绑定的完整有界外排结果身份。"""
        result = [INTEGER_EXTERNAL_SORT_RESULT_SCHEMA]
        pack_key(result, self.identity.integer_stream())
        result.extend((
            self.temporary_run_count,
            self.temporary_record_count,
            self.temporary_payload_bytes,
            self.temporary_physical_bytes,
            self.initial_run_count,
            self.merge_pass_count,
            self.merge_fan_in,
            self.max_open_files,
        ))
        pack_key(result, self.budget.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回不以文本序列化或绝对文件系统定位表达的结果稳定键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _PendingRecord:
    """一条尚未落盘的 record、排序键及其受预算约束的规范编码长度。"""

    record: tuple[int, ...]
    sort_key: tuple[int, ...]
    payload_bytes: int
    sort_key_bytes: int


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _RunDescriptor:
    """已封存临时 run 的路径、双重身份与稳定归并优先级起点。"""

    relative_path: Path
    footer: IntegerFramedStreamFooter
    physical: KRunFileDigest
    file_identity: KRunFileIdentity
    first_initial_run_index: int

    def __post_init__(self) -> None:
        """固定一个已封存临时 run 可被后续 merge 核验的全部小型元数据。"""
        _validate_relative_path(self.relative_path, label="temporary run relative_path")
        if not isinstance(self.footer, IntegerFramedStreamFooter):
            raise TypeError("temporary run footer 必须是 IntegerFramedStreamFooter")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("temporary run physical 必须是 KRunFileDigest")
        if not isinstance(self.file_identity, KRunFileIdentity):
            raise TypeError("temporary run file_identity 必须是 KRunFileIdentity")
        _require_nonnegative_integer(
            self.first_initial_run_index,
            label="temporary run first_initial_run_index",
        )


# object-model: resource_owner; representation=protocol; interop=pending
class _DigestingBinaryHandle:
    """把单一已核验文件句柄的成功读写字节绑定为有限 physical SHA-256。

    P0 reader/writer 是该 wrapper 的唯一调用者；wrapper 只累计底层实际返回成功的 bytes，
    不缓存全文，也不自行选择或重新打开路径。``max_bytes`` 在底层成功返回后立即检查，
    因而不会把越过物理预算的数据交给 P0 继续解析或纳入 result。
    """

    __slots__ = ("_byte_count", "_digest", "_max_bytes", "_stream")

    def __init__(self, stream: BinaryIO, *, max_bytes: int) -> None:
        """接管已由 K boundary 打开的二进制句柄，固定完整 physical 读取/写入预算。"""
        _require_positive_integer(max_bytes, label="digesting handle max_bytes")
        self._byte_count = 0
        self._digest = hashlib.sha256()
        self._max_bytes = max_bytes
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        """读取底层字节并在 P0 消费前将实际物理字节纳入同一 handle digest。"""
        remaining = self._max_bytes - self._byte_count
        if type(size) is int:
            requested = remaining + 1 if size < 0 else min(size, remaining + 1)
        else:
            requested = size
        data = self._stream.read(requested)
        if not isinstance(data, bytes):
            return data
        self._accept(data)
        return data

    def write(self, data: bytes | bytearray | memoryview) -> int:
        """写入底层句柄，并只累计其确认成功写出的前缀字节。"""
        if len(data) > self._max_bytes - self._byte_count:
            raise IntegerExternalSortBudgetExceeded(
                "external sort 同一 handle 物理字节超过预算")
        written = self._stream.write(data)
        if type(written) is int and 0 < written <= len(data):
            self._accept(bytes(memoryview(data)[:written]))
        return written

    def flush(self) -> None:
        """透传 P0 seal 所需的 flush，不把 flush 伪造为成功物理写入。"""
        self._stream.flush()

    def close(self) -> None:
        """关闭唯一底层句柄；关闭错误由 P0 resource owner 按 fail-closed 处理。"""
        self._stream.close()

    def physical(self) -> KRunFileDigest:
        """返回当前同一 handle 已实际成功处理的有限 physical 身份。"""
        return KRunFileDigest(self._byte_count, tuple(self._digest.digest()))

    def _accept(self, data: bytes) -> None:
        """在更新 digest 前拒绝超出明确物理预算的成功 I/O 字节。"""
        if len(data) > self._max_bytes - self._byte_count:
            raise IntegerExternalSortBudgetExceeded(
                "external sort 同一 handle 物理字节超过预算")
        self._digest.update(data)
        self._byte_count += len(data)


def external_sort_sealed_integer_records(
        input_root: KRunRoot,
        input_relative_paths: tuple[Path, ...],
        work_root: KRunRoot,
        *,
        output_relative_path: Path,
        logical_stage_name: str,
        sort_key: Callable[[tuple[int, ...]], tuple[int, ...]],
        budget: IntegerExternalSortBudget,
        ) -> IntegerExternalSortResult:
    """将多个已封存 P0 framed stream 稳定外排到 work root 的固定新输出。

    输入 tuple 的顺序是相同排序键 record 的稳定次序。所有输入都先以 P0 footer 和物理
    SHA-256 完整核验；batch 仅保存受 ``budget`` 限制的 record 与 sort key。临时 run
    只可排他新建，达到 fan-in 后通过有界多路归并逐轮缩减，最终输出同样以 P0 footer 和
    物理 SHA-256 绑定。任何失败都不删除残片、不会覆盖已存在 output，且不返回 result。
    """
    if not isinstance(budget, IntegerExternalSortBudget):
        raise TypeError("budget 必须是 IntegerExternalSortBudget")
    if not callable(sort_key):
        raise TypeError("sort_key 必须是可调用严格整数 tuple 回调")
    _validate_stage_name(logical_stage_name)
    input_paths = _validate_input_relative_paths(
        input_relative_paths,
        budget=budget,
    )
    output_relative = _validated_relative_path(
        output_relative_path,
        label="output_relative_path",
    )
    require_disjoint_run_roots(
        input_root,
        work_root,
        label="external sort input/work roots",
    )

    temporary_stage = Path(_TEMPORARY_NAMESPACE) / logical_stage_name
    if _is_relative_path_within(output_relative, temporary_stage):
        raise IntegerExternalSortError("output_relative_path 不得位于保留临时 run 命名空间")
    output_path = relative_path(
        work_root,
        output_relative,
        label="external sort output",
    )
    if os.path.lexists(output_path):
        raise IntegerExternalSortError("external sort output 已存在，禁止覆盖")

    if output_relative.parent.parts:
        ensure_normal_relative_directory(
            work_root,
            output_relative.parent,
            label="external sort output parent",
        )
    ensure_normal_relative_directory(
        work_root,
        temporary_stage,
        label="external sort temporary stage",
    )

    batch: list[_PendingRecord] = []
    batch_payload_bytes = 0
    batch_sort_key_bytes = 0
    runs: list[_RunDescriptor] = []
    input_identities: list[IntegerExternalSortInputIdentity] = []
    input_record_count = 0
    input_payload_bytes = 0
    input_physical_bytes = 0
    temporary_run_count = 0
    temporary_record_count = 0
    temporary_payload_bytes = 0
    temporary_physical_bytes = 0
    next_initial_run_index = 0

    def flush_batch() -> None:
        """排序当前有界 batch 并写出一个唯一封存初始 run，不复用任何已有路径。"""
        nonlocal batch_payload_bytes
        nonlocal batch_sort_key_bytes
        nonlocal next_initial_run_index
        nonlocal temporary_run_count
        nonlocal temporary_record_count
        nonlocal temporary_payload_bytes
        nonlocal temporary_physical_bytes
        if not batch:
            return
        if next_initial_run_index > _MAX_PATH_COUNTER:
            raise IntegerExternalSortBudgetExceeded("初始 run 路径计数超过上限")
        batch.sort(key=lambda item: item.sort_key)
        relative = _temporary_run_relative(
            temporary_stage,
            pass_index=0,
            run_index=next_initial_run_index,
        )
        run, totals = _write_pending_run(
            work_root,
            relative,
            batch,
            first_initial_run_index=next_initial_run_index,
            budget=budget,
            temporary_totals=(
                temporary_run_count,
                temporary_record_count,
                temporary_payload_bytes,
                temporary_physical_bytes,
            ),
            label="external sort initial run",
        )
        runs.append(run)
        (
            temporary_run_count,
            temporary_record_count,
            temporary_payload_bytes,
            temporary_physical_bytes,
        ) = totals
        next_initial_run_index += 1
        batch.clear()
        batch_payload_bytes = 0
        batch_sort_key_bytes = 0

    for input_index, relative in enumerate(input_paths):
        source_record_count = 0
        source_payload_bytes = 0
        input_identity = capture_plain_file_identity(
            input_root,
            relative,
            label=f"external sort input[{input_index}] capture",
        )
        if input_identity.byte_count > budget.max_input_physical_bytes - input_physical_bytes:
            raise IntegerExternalSortBudgetExceeded("external sort 输入物理字节累计超过预算")
        reader, reader_handle = _open_framed_reader(
            input_root,
            relative,
            budget=budget,
            max_physical_bytes=(
                budget.max_input_physical_bytes - input_physical_bytes),
            expected_identity=input_identity,
            label=f"external sort input[{input_index}]",
        )
        try:
            while True:
                record = reader.read_record()
                if record is None:
                    break
                payload_bytes = encoded_integer_tuple_size(record)
                if payload_bytes > budget.max_record_payload_bytes:
                    raise IntegerExternalSortBudgetExceeded(
                        "external sort 单条 record payload 超过预算")
                if input_record_count >= budget.max_input_record_count:
                    raise IntegerExternalSortBudgetExceeded(
                        "external sort 输入 record 累计超过预算")
                if payload_bytes > budget.max_input_payload_bytes - input_payload_bytes:
                    raise IntegerExternalSortBudgetExceeded(
                        "external sort 输入 payload 累计超过预算")
                key = _checked_sort_key(sort_key, record)
                key_bytes = encoded_integer_tuple_size(key)
                if key_bytes > budget.max_batch_sort_key_bytes:
                    raise IntegerExternalSortBudgetExceeded(
                        "external sort 单条 sort key 超过 batch 预算")
                if (batch and (
                        len(batch) >= budget.max_batch_record_count
                        or payload_bytes > budget.max_batch_payload_bytes - batch_payload_bytes
                        or key_bytes > budget.max_batch_sort_key_bytes - batch_sort_key_bytes
                )):
                    flush_batch()
                if len(batch) >= budget.max_batch_record_count:
                    raise IntegerExternalSortBudgetExceeded(
                        "external sort batch record 数超过预算")
                if payload_bytes > budget.max_batch_payload_bytes - batch_payload_bytes:
                    raise IntegerExternalSortBudgetExceeded(
                        "external sort batch payload 超过预算")
                if key_bytes > budget.max_batch_sort_key_bytes - batch_sort_key_bytes:
                    raise IntegerExternalSortBudgetExceeded(
                        "external sort batch sort key 超过预算")
                batch.append(_PendingRecord(
                    record=record,
                    sort_key=key,
                    payload_bytes=payload_bytes,
                    sort_key_bytes=key_bytes,
                ))
                batch_payload_bytes += payload_bytes
                batch_sort_key_bytes += key_bytes
                input_record_count += 1
                input_payload_bytes += payload_bytes
                source_record_count += 1
                source_payload_bytes += payload_bytes
            footer = reader.footer
            if footer is None:
                raise IntegerExternalSortError("external sort 输入未取得 sealed footer")
            if (footer.record_count != source_record_count
                    or footer.total_payload_bytes != source_payload_bytes):
                raise IntegerExternalSortError("external sort 输入 footer 与流式计数不一致")
        finally:
            reader.close()
        physical_from_handle = reader_handle.physical()
        if physical_from_handle.byte_count != input_identity.byte_count:
            raise IntegerExternalSortError("external sort 输入读取字节数与捕获身份不一致")
        if (physical_from_handle.byte_count
                > budget.max_input_physical_bytes - input_physical_bytes):
            raise IntegerExternalSortBudgetExceeded("external sort 输入物理字节累计超过预算")
        require_plain_file_identity(
            input_root,
            relative,
            input_identity,
            label=f"external sort input[{input_index}] post-read",
        )
        input_physical_bytes += input_identity.byte_count
        input_identities.append(IntegerExternalSortInputIdentity(
            relative_path=relative,
            footer=footer,
            physical=physical_from_handle,
        ))

    flush_batch()
    if input_payload_bytes > budget.max_input_payload_bytes:
        raise AssertionError("输入 payload 预算检查遗漏")
    if input_record_count > budget.max_input_record_count:
        raise AssertionError("输入 record 预算检查遗漏")
    if input_physical_bytes > budget.max_input_physical_bytes:
        raise AssertionError("输入物理字节预算检查遗漏")

    initial_run_count = len(runs)
    merge_pass_count = 0
    pass_index = 1
    while len(runs) > budget.merge_fan_in:
        if merge_pass_count >= budget.max_merge_pass_count:
            raise IntegerExternalSortBudgetExceeded("external sort merge pass 数超过预算")
        next_runs: list[_RunDescriptor] = []
        for group_index, start in enumerate(range(0, len(runs), budget.merge_fan_in)):
            group = tuple(runs[start:start + budget.merge_fan_in])
            if len(group) == 1:
                next_runs.append(group[0])
                continue
            relative = _temporary_run_relative(
                temporary_stage,
                pass_index=pass_index,
                run_index=group_index,
            )
            expected_records, expected_payload = _run_group_totals(group)
            predicted_upper = _stream_physical_upper_bound(
                expected_records,
                expected_payload,
                max_record_payload_bytes=budget.max_record_payload_bytes,
            )
            totals = _reserve_temporary_totals(
                (
                    temporary_run_count,
                    temporary_record_count,
                    temporary_payload_bytes,
                    temporary_physical_bytes,
                ),
                record_count=expected_records,
                payload_bytes=expected_payload,
                physical_bytes=predicted_upper,
                budget=budget,
            )
            footer, physical, file_identity = _merge_runs_to_new_stream(
                work_root,
                group,
                relative,
                sort_key=sort_key,
                budget=budget,
                physical_limit=predicted_upper,
                label="external sort temporary merge",
            )
            if physical.byte_count > predicted_upper:
                raise IntegerExternalSortError("temporary merge 物理字节超过预先上界")
            next_runs.append(_RunDescriptor(
                relative_path=relative,
                footer=footer,
                physical=physical,
                file_identity=file_identity,
                first_initial_run_index=group[0].first_initial_run_index,
            ))
            (
                temporary_run_count,
                temporary_record_count,
                temporary_payload_bytes,
                temporary_physical_bytes,
            ) = _replace_reserved_physical_bytes(
                totals,
                predicted_physical_bytes=predicted_upper,
                actual_physical_bytes=physical.byte_count,
            )
        runs = next_runs
        merge_pass_count += 1
        pass_index += 1

    if not runs:
        output_footer, output_physical, _output_file_identity = _write_empty_output(
            work_root,
            output_relative,
            budget=budget,
        )
    else:
        output_footer, output_physical, _output_file_identity = _merge_runs_to_new_stream(
            work_root,
            tuple(runs),
            output_relative,
            sort_key=sort_key,
            budget=budget,
            physical_limit=budget.max_output_physical_bytes,
            label="external sort output",
        )

    if (output_footer.record_count != input_record_count
            or output_footer.total_payload_bytes != input_payload_bytes):
        raise IntegerExternalSortError("external sort 输出 footer 与输入累计量不一致")
    identity = IntegerExternalSortIdentity(
        logical_stage_name=logical_stage_name,
        input_identities=tuple(input_identities),
        output_relative_path=output_relative,
        output_footer=output_footer,
        output_physical=output_physical,
    )
    return IntegerExternalSortResult(
        identity=identity,
        temporary_run_count=temporary_run_count,
        temporary_record_count=temporary_record_count,
        temporary_payload_bytes=temporary_payload_bytes,
        temporary_physical_bytes=temporary_physical_bytes,
        initial_run_count=initial_run_count,
        merge_pass_count=merge_pass_count,
        merge_fan_in=budget.merge_fan_in,
        max_open_files=budget.max_open_files,
        budget=budget,
    )


def _write_pending_run(
        work_root: KRunRoot,
        relative: Path,
        records: list[_PendingRecord],
        *,
        first_initial_run_index: int,
        budget: IntegerExternalSortBudget,
        temporary_totals: tuple[int, int, int, int],
        label: str,
        ) -> tuple[_RunDescriptor, tuple[int, int, int, int]]:
    """把一个已按 key 稳定排序的有界 batch 写成新封存 run 并回传累计量。"""
    if not records:
        raise ValueError("外排初始 run 不得从空 batch 写出")
    expected_footer, expected_physical_bytes = _exact_pending_stream_identity(records)
    totals = _reserve_temporary_totals(
        temporary_totals,
        record_count=expected_footer.record_count,
        payload_bytes=expected_footer.total_payload_bytes,
        physical_bytes=expected_physical_bytes,
        budget=budget,
    )
    _ensure_parent(work_root, relative, label=label)
    footer, physical, file_identity = _write_records_exclusively(
        work_root,
        relative,
        (item.record for item in records),
        expected_record_count=expected_footer.record_count,
        expected_payload_bytes=expected_footer.total_payload_bytes,
        max_physical_bytes=expected_physical_bytes,
        label=label,
    )
    if footer != expected_footer:
        raise IntegerExternalSortError("external sort 初始 run footer 不可重现")
    if physical.byte_count != expected_physical_bytes:
        raise IntegerExternalSortError("external sort 初始 run 物理字节不匹配")
    final_totals = _replace_reserved_physical_bytes(
        totals,
        predicted_physical_bytes=expected_physical_bytes,
        actual_physical_bytes=physical.byte_count,
    )
    return (
        _RunDescriptor(
            relative_path=relative,
            footer=footer,
            physical=physical,
            file_identity=file_identity,
            first_initial_run_index=first_initial_run_index,
        ),
        final_totals,
    )


def _merge_runs_to_new_stream(
        work_root: KRunRoot,
        runs: tuple[_RunDescriptor, ...],
        output_relative: Path,
        *,
        sort_key: Callable[[tuple[int, ...]], tuple[int, ...]],
        budget: IntegerExternalSortBudget,
        physical_limit: int,
        label: str,
        ) -> tuple[IntegerFramedStreamFooter, KRunFileDigest, KRunFileIdentity]:
    """有界打开多个已封存 run，以稳定 k-way merge 排他写出一个新 P0 stream。"""
    if not runs:
        raise ValueError("外排 merge 至少需要一个 run")
    if len(runs) > budget.merge_fan_in:
        raise IntegerExternalSortBudgetExceeded("external sort merge fan-in 超过预算")
    if len(runs) + 1 > budget.max_open_files:
        raise IntegerExternalSortBudgetExceeded("external sort merge 打开文件数超过预算")
    _require_positive_integer(physical_limit, label="external sort physical_limit")
    expected_record_count, expected_payload_bytes = _run_group_totals(runs)
    upper_bound = _stream_physical_upper_bound(
        expected_record_count,
        expected_payload_bytes,
        max_record_payload_bytes=budget.max_record_payload_bytes,
    )
    if upper_bound > physical_limit:
        raise IntegerExternalSortBudgetExceeded("external sort 输出物理字节上界超过预算")
    _ensure_parent(work_root, output_relative, label=label)

    readers: list[tuple[IntegerFramedStreamReader, _RunDescriptor, _DigestingBinaryHandle]] = []
    writer: IntegerFramedStreamWriter | None = None
    output_footer: IntegerFramedStreamFooter | None = None
    try:
        for index, run in enumerate(runs):
            reader, handle = _open_framed_reader(
                work_root,
                run.relative_path,
                budget=budget,
                max_physical_bytes=run.physical.byte_count,
                expected_identity=run.file_identity,
                label=f"{label} input[{index}]",
            )
            readers.append((reader, run, handle))
        writer, writer_handle = _open_framed_writer(
            work_root,
            output_relative,
            max_physical_bytes=physical_limit,
            label=label,
        )
        previous_keys: list[tuple[int, ...] | None] = [None] * len(readers)
        heap: list[tuple[tuple[int, ...], int, tuple[int, ...]]] = []
        for index, (reader, _run, _handle) in enumerate(readers):
            item = _next_merge_item(
                reader,
                reader_index=index,
                previous_keys=previous_keys,
                sort_key=sort_key,
            )
            if item is not None:
                heapq.heappush(heap, item)
        while heap:
            _key, reader_index, record = heapq.heappop(heap)
            writer.append(record)
            reader, _run, _handle = readers[reader_index]
            item = _next_merge_item(
                reader,
                reader_index=reader_index,
                previous_keys=previous_keys,
                sort_key=sort_key,
            )
            if item is not None:
                heapq.heappush(heap, item)
        for index, (reader, run, handle) in enumerate(readers):
            footer = reader.finish()
            if footer != run.footer:
                raise IntegerExternalSortError(
                    f"{label} input[{index}] sealed footer 漂移")
            if handle.physical() != run.physical:
                raise IntegerExternalSortError(
                    f"{label} input[{index}] 临时 run physical 身份漂移")
            reader.close()
            require_plain_file_identity(
                work_root,
                run.relative_path,
                run.file_identity,
                label=f"{label} input[{index}] post-read",
            )
        output_footer = writer.seal()
        if (output_footer.record_count != expected_record_count
                or output_footer.total_payload_bytes != expected_payload_bytes):
            raise IntegerExternalSortError(f"{label} 输出 footer 与 merge 输入不一致")
    finally:
        if writer is not None:
            writer.close()
        for reader, _run, _handle in readers:
            reader.close()
    if output_footer is None:
        raise AssertionError("external sort merge 未取得 output footer")
    output_physical = writer_handle.physical()
    output_file_identity = capture_plain_file_identity(
        work_root,
        output_relative,
        label=f"{label} output identity",
    )
    if output_physical.byte_count > physical_limit:
        raise IntegerExternalSortBudgetExceeded(f"{label} 输出物理字节超过预算")
    if output_physical.byte_count != output_file_identity.byte_count:
        raise IntegerExternalSortError(f"{label} 输出 handle 与路径字节身份漂移")
    _verify_actual_output_footer(
        work_root,
        output_relative,
        expected_footer=output_footer,
        expected_physical=output_physical,
        expected_identity=output_file_identity,
        budget=budget,
        label=label,
    )
    return output_footer, output_physical, output_file_identity


def _write_empty_output(
        work_root: KRunRoot,
        output_relative: Path,
        *,
        budget: IntegerExternalSortBudget,
        ) -> tuple[IntegerFramedStreamFooter, KRunFileDigest, KRunFileIdentity]:
    """把零 record 输入规范地写成一个封存空 P0 stream，而不创建临时 run。"""
    upper_bound = _stream_physical_upper_bound(
        0,
        0,
        max_record_payload_bytes=budget.max_record_payload_bytes,
    )
    if upper_bound > budget.max_output_physical_bytes:
        raise IntegerExternalSortBudgetExceeded("external sort 空输出物理字节上界超过预算")
    _ensure_parent(work_root, output_relative, label="external sort empty output")
    footer, physical, file_identity = _write_records_exclusively(
        work_root,
        output_relative,
        (),
        expected_record_count=0,
        expected_payload_bytes=0,
        max_physical_bytes=budget.max_output_physical_bytes,
        label="external sort empty output",
    )
    _verify_actual_output_footer(
        work_root,
        output_relative,
        expected_footer=footer,
        expected_physical=physical,
        expected_identity=file_identity,
        budget=budget,
        label="external sort empty output",
    )
    return footer, physical, file_identity


def _write_records_exclusively(
        root: KRunRoot,
        relative: Path,
        records: Iterable[tuple[int, ...]],
        *,
        expected_record_count: int,
        expected_payload_bytes: int,
        max_physical_bytes: int,
        label: str,
        ) -> tuple[IntegerFramedStreamFooter, KRunFileDigest, KRunFileIdentity]:
    """接管一个 capability 排他句柄，按给定流顺序写入并唯一 seal P0 stream。"""
    stream: BinaryIO | None = None
    handle: _DigestingBinaryHandle | None = None
    writer: IntegerFramedStreamWriter | None = None
    try:
        stream = open_exclusive_binary(root, relative, label=label)
        handle = _DigestingBinaryHandle(
            stream,
            max_bytes=max_physical_bytes,
        )
        writer = IntegerFramedStreamWriter.from_open_binary(
            handle,
            path=relative,
        )
        stream = None
        for record in records:
            writer.append(record)
        footer = writer.seal()
    finally:
        if writer is not None:
            writer.close()
        elif handle is not None:
            handle.close()
        elif stream is not None:
            stream.close()
    if (footer.record_count != expected_record_count
            or footer.total_payload_bytes != expected_payload_bytes):
        raise IntegerExternalSortError(f"{label} writer footer 与预期不一致")
    if handle is None:
        raise AssertionError("external sort writer 缺少 digesting handle")
    physical = handle.physical()
    file_identity = capture_plain_file_identity(root, relative, label=f"{label} identity")
    if physical.byte_count != file_identity.byte_count:
        raise IntegerExternalSortError(f"{label} writer handle 与路径字节身份漂移")
    return footer, physical, file_identity


def _open_framed_writer(
        root: KRunRoot,
        relative: Path,
        *,
        max_physical_bytes: int,
        label: str,
        ) -> tuple[IntegerFramedStreamWriter, _DigestingBinaryHandle]:
    """以 K capability 排他打开新文件并把唯一句柄所有权交给 P0 writer。"""
    stream = open_exclusive_binary(root, relative, label=label)
    handle = _DigestingBinaryHandle(stream, max_bytes=max_physical_bytes)
    try:
        return IntegerFramedStreamWriter.from_open_binary(handle, path=relative), handle
    except BaseException:
        try:
            handle.close()
        except OSError:
            pass
        raise


def _open_framed_reader(
        root: KRunRoot,
        relative: Path,
        *,
        budget: IntegerExternalSortBudget,
        max_physical_bytes: int,
        expected_identity: KRunFileIdentity,
        label: str,
        ) -> tuple[IntegerFramedStreamReader, _DigestingBinaryHandle]:
    """以 K capability 打开已核验文件并把句柄所有权交给有界 P0 reader。"""
    stream = open_plain_binary(
        root,
        relative,
        label=label,
        expected_identity=expected_identity,
    )
    handle = _DigestingBinaryHandle(stream, max_bytes=max_physical_bytes)
    try:
        return IntegerFramedStreamReader.from_open_binary(
            handle,
            path=relative,
            max_frame_bytes=budget.max_record_payload_bytes,
            max_record_count=budget.max_input_record_count,
            max_total_payload_bytes=budget.max_input_payload_bytes,
        ), handle
    except BaseException:
        try:
            handle.close()
        except OSError:
            pass
        raise


def _next_merge_item(
        reader: IntegerFramedStreamReader,
        *,
        reader_index: int,
        previous_keys: list[tuple[int, ...] | None],
        sort_key: Callable[[tuple[int, ...]], tuple[int, ...]],
        ) -> tuple[tuple[int, ...], int, tuple[int, ...]] | None:
    """读取一个 merge record，重算严格 key 并拒绝任何临时 run 键序倒退。"""
    record = reader.read_record()
    if record is None:
        return None
    key = _checked_sort_key(sort_key, record)
    previous = previous_keys[reader_index]
    if previous is not None and key < previous:
        raise IntegerExternalSortError("external sort 临时 run 的 sort key 非单调")
    previous_keys[reader_index] = key
    return key, reader_index, record


def _verify_actual_output_footer(
        work_root: KRunRoot,
        output_relative: Path,
        *,
        expected_footer: IntegerFramedStreamFooter,
        expected_physical: KRunFileDigest,
        expected_identity: KRunFileIdentity,
        budget: IntegerExternalSortBudget,
        label: str,
        ) -> None:
    """重新经同一打开 handle 验证最终 P0 footer、physical SHA 与路径身份。"""
    reader, handle = _open_framed_reader(
        work_root,
        output_relative,
        budget=budget,
        max_physical_bytes=expected_physical.byte_count,
        expected_identity=expected_identity,
        label=f"{label} footer verification",
    )
    try:
        footer = reader.finish()
    finally:
        reader.close()
    if footer != expected_footer:
        raise IntegerExternalSortError(f"{label} 输出路径 footer 漂移")
    if handle.physical() != expected_physical:
        raise IntegerExternalSortError(f"{label} 输出路径 physical 身份漂移")
    require_plain_file_identity(
        work_root,
        output_relative,
        expected_identity,
        label=f"{label} footer verification post-read",
    )


def _checked_sort_key(
        sort_key: Callable[[tuple[int, ...]], tuple[int, ...]],
        record: tuple[int, ...],
        ) -> tuple[int, ...]:
    """调用外部排序键回调，并将非严格整数 tuple 统一按 fail-closed 拒绝。"""
    try:
        key = sort_key(record)
    except BaseException as exc:
        raise IntegerExternalSortError("external sort sort_key 回调失败") from exc
    if type(key) is not tuple:
        raise IntegerExternalSortError("external sort sort_key 必须返回严格 tuple")
    try:
        strict_integer_tuple(key, label="external sort sort_key", empty=True)
    except (TypeError, ValueError) as exc:
        raise IntegerExternalSortError(
            "external sort sort_key 必须只含严格整数") from exc
    if any(type(item) is not int for item in key):
        raise IntegerExternalSortError("external sort sort_key 必须只含严格整数")
    return key


def _exact_pending_stream_identity(
        records: list[_PendingRecord],
        ) -> tuple[IntegerFramedStreamFooter, int]:
    """从有界 batch 预计算 writer 应产生的精确 footer 和完整物理字节数。"""
    digest = hashlib.sha256()
    total_payload_bytes = 0
    frame_length_bytes = 0
    for item in records:
        payload = encode_integer_tuple(item.record)
        if len(payload) != item.payload_bytes:
            raise IntegerExternalSortError("external sort batch payload 长度漂移")
        digest.update(payload)
        total_payload_bytes += len(payload)
        frame_length_bytes += _unsigned_varint_size(len(payload))
    footer = IntegerFramedStreamFooter(
        INTEGER_FRAMED_STREAM_SCHEMA,
        len(records),
        total_payload_bytes,
        tuple(digest.digest()),
    )
    physical_bytes = (
        len(INTEGER_FRAMED_STREAM_MAGIC)
        + frame_length_bytes
        + total_payload_bytes
        + 1
        + len(encode_integer_tuple(footer.integer_tuple()))
    )
    return footer, physical_bytes


def _stream_physical_upper_bound(
        record_count: int,
        payload_bytes: int,
        *,
        max_record_payload_bytes: int,
        ) -> int:
    """给定 P0 footer 总量计算不依赖正文的保守物理字节上界。"""
    _require_nonnegative_integer(record_count, label="stream record_count")
    _require_nonnegative_integer(payload_bytes, label="stream payload_bytes")
    _require_positive_integer(
        max_record_payload_bytes,
        label="stream max_record_payload_bytes",
    )
    if record_count == 0 and payload_bytes != 0:
        raise IntegerExternalSortError("零 record stream 不得具有 payload")
    maximum_footer = IntegerFramedStreamFooter(
        INTEGER_FRAMED_STREAM_SCHEMA,
        record_count,
        payload_bytes,
        (255,) * _SHA256_BYTE_COUNT,
    )
    return (
        len(INTEGER_FRAMED_STREAM_MAGIC)
        + record_count * _unsigned_varint_size(max_record_payload_bytes)
        + payload_bytes
        + 1
        + len(encode_integer_tuple(maximum_footer.integer_tuple()))
    )


def _run_group_totals(runs: tuple[_RunDescriptor, ...]) -> tuple[int, int]:
    """对少量 fan-in descriptors 累计 footer record 与 payload 总量。"""
    if not runs:
        raise ValueError("run group 不得为空")
    record_count = 0
    payload_bytes = 0
    previous_first_index = -1
    for run in runs:
        if run.first_initial_run_index <= previous_first_index:
            raise IntegerExternalSortError("external sort run 归并顺序不稳定")
        previous_first_index = run.first_initial_run_index
        record_count += run.footer.record_count
        payload_bytes += run.footer.total_payload_bytes
    return record_count, payload_bytes


def _reserve_temporary_totals(
        totals: tuple[int, int, int, int],
        *,
        record_count: int,
        payload_bytes: int,
        physical_bytes: int,
        budget: IntegerExternalSortBudget,
        ) -> tuple[int, int, int, int]:
    """在创建任何临时文件之前按其完整上界原子式预留所有累计预算。"""
    if (not isinstance(totals, tuple) or len(totals) != 4
            or any(type(value) is not int or value < 0 for value in totals)):
        raise TypeError("temporary totals 必须是四个非负严格整数")
    _require_nonnegative_integer(record_count, label="temporary record_count")
    _require_nonnegative_integer(payload_bytes, label="temporary payload_bytes")
    _require_positive_integer(physical_bytes, label="temporary physical_bytes")
    run_count, old_records, old_payload, old_physical = totals
    if run_count >= budget.max_temporary_run_count:
        raise IntegerExternalSortBudgetExceeded("external sort 临时 run 数超过预算")
    if record_count > budget.max_temporary_record_count - old_records:
        raise IntegerExternalSortBudgetExceeded("external sort 临时 record 累计超过预算")
    if payload_bytes > budget.max_temporary_payload_bytes - old_payload:
        raise IntegerExternalSortBudgetExceeded("external sort 临时 payload 累计超过预算")
    if physical_bytes > budget.max_temporary_physical_bytes - old_physical:
        raise IntegerExternalSortBudgetExceeded("external sort 临时物理字节累计超过预算")
    return (
        run_count + 1,
        old_records + record_count,
        old_payload + payload_bytes,
        old_physical + physical_bytes,
    )


def _replace_reserved_physical_bytes(
        totals: tuple[int, int, int, int],
        *,
        predicted_physical_bytes: int,
        actual_physical_bytes: int,
        ) -> tuple[int, int, int, int]:
    """以已验证实际字节替换预留上界，保留保守检查但让 result 记录真实累计量。"""
    _require_positive_integer(
        predicted_physical_bytes,
        label="predicted_physical_bytes",
    )
    _require_positive_integer(
        actual_physical_bytes,
        label="actual_physical_bytes",
    )
    if actual_physical_bytes > predicted_physical_bytes:
        raise IntegerExternalSortError("external sort 实际物理字节超过预留上界")
    run_count, record_count, payload_bytes, reserved_physical = totals
    if reserved_physical < predicted_physical_bytes:
        raise AssertionError("temporary physical 预留计数错误")
    return (
        run_count,
        record_count,
        payload_bytes,
        reserved_physical - predicted_physical_bytes + actual_physical_bytes,
    )


def _temporary_run_relative(
        temporary_stage: Path,
        *,
        pass_index: int,
        run_index: int,
        ) -> Path:
    """生成稳定、有限且不依赖运行时随机数的临时 run 相对路径。"""
    _require_nonnegative_integer(pass_index, label="temporary pass_index")
    _require_nonnegative_integer(run_index, label="temporary run_index")
    if pass_index > _MAX_PATH_COUNTER or run_index > _MAX_PATH_COUNTER:
        raise IntegerExternalSortBudgetExceeded("temporary run 路径计数超过上限")
    return (
        temporary_stage
        / "runs"
        / f"pass-{pass_index:08d}"
        / f"run-{run_index:08d}.pifrs"
    )


def _ensure_parent(root: KRunRoot, relative: Path, *, label: str) -> None:
    """显式建立或复核 output 的普通父目录，供边界层排他打开使用。"""
    _validate_relative_path(relative, label=f"{label} relative path")
    if relative.parent.parts:
        ensure_normal_relative_directory(
            root,
            relative.parent,
            label=f"{label} parent",
        )


def _validate_input_relative_paths(
        value: tuple[Path, ...],
        *,
        budget: IntegerExternalSortBudget,
        ) -> tuple[Path, ...]:
    """固定小型、非空、无重复的输入相对路径 tuple，保留调用方顺序作为稳定 tie 次序。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError("input_relative_paths 必须是非空 tuple[Path, ...]")
    if len(value) > budget.max_input_file_count:
        raise IntegerExternalSortBudgetExceeded("external sort 输入文件数超过预算")
    result: list[Path] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        relative = _validated_relative_path(
            item,
            label=f"input_relative_paths[{index}]",
        )
        normalized = os.path.normcase(str(relative))
        if normalized in seen:
            raise IntegerExternalSortError("external sort 输入路径不得重复")
        seen.add(normalized)
        result.append(relative)
    return tuple(result)


def _validated_relative_path(value: Path, *, label: str) -> Path:
    """返回语法已拒绝绝对、盘符、空和向上逃逸成分的相对 ``Path``。"""
    _validate_relative_path(value, label=label)
    return Path(*value.parts)


def _is_relative_path_within(value: Path, parent: Path) -> bool:
    """以 Windows ``normcase`` 语义判断两个已验证相对路径的成分级包含关系。"""
    value_parts = tuple(os.path.normcase(part) for part in value.parts)
    parent_parts = tuple(os.path.normcase(part) for part in parent.parts)
    return (len(value_parts) >= len(parent_parts)
            and value_parts[:len(parent_parts)] == parent_parts)


def _validate_relative_path(value: Path, *, label: str) -> None:
    """拒绝跨 root 路径；实际父链、链接和文件类型仍交给 K boundary 核验。"""
    if not isinstance(value, Path):
        raise TypeError(f"{label} 必须是 Path")
    if (value.is_absolute() or value.drive or value.root or not value.parts
            or ".." in value.parts
            or any(part in {"", ".", ".."} for part in value.parts)):
        raise ValueError(f"{label} 必须是非空且不含 .. 的相对 Path")


def _relative_path_integer_key(value: Path) -> tuple[int, ...]:
    """将相对路径按当前 normcase 与 POSIX 分隔符编码为 Unicode scalar 键。"""
    relative = _validated_relative_path(
        value,
        label="external sort relative path identity",
    )
    text = "/".join(os.path.normcase(relative.as_posix()).split("\\"))
    return _unicode_scalar_key(
        text,
        label="external sort relative path identity",
    )


def _stage_name_integer_key(value: str) -> tuple[int, ...]:
    """把已受 ASCII 合同约束的 logical stage 编成显式 scalar 键。"""
    _validate_stage_name(value)
    return _unicode_scalar_key(value, label="external sort logical stage")


def _unicode_scalar_key(value: str, *, label: str) -> tuple[int, ...]:
    """拒绝代理项并返回未经 Unicode 归一化的严格整数 scalar 序列。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} 必须是非空 str")
    result = tuple(ord(character) for character in value)
    if any(0xD800 <= scalar <= 0xDFFF for scalar in result):
        raise ValueError(f"{label} 不得含 Unicode 代理项")
    strict_integer_tuple(result, label=label)
    return result


def _validate_stage_name(value: str) -> None:
    """限制 logical stage 为可稳定映射到单个临时目录的 ASCII 标识。"""
    if not isinstance(value, str) or not value:
        raise TypeError("logical_stage_name 必须是非空 str")
    if len(value) > _MAX_STAGE_NAME_LENGTH:
        raise ValueError("logical_stage_name 超过长度上限")
    if not value[0].isalnum() or not value[0].isascii():
        raise ValueError("logical_stage_name 必须以 ASCII 字母或数字开始")
    if any(not character.isascii() or not (
            character.isalnum() or character in {"-", "_", "."})
           for character in value):
        raise ValueError("logical_stage_name 只能包含 ASCII 字母、数字、-、_、.")


def _unsigned_varint_size(value: int) -> int:
    """计算 P0 frame length 所用的最短 unsigned varint 字节数。"""
    _require_nonnegative_integer(value, label="unsigned varint value")
    size = (value.bit_length() + 6) // 7
    return size if size else 1


def _require_positive_integer(value: int, *, label: str) -> int:
    """拒绝 bool、零和负值，避免预算被隐式真值或无界含义放宽。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_nonnegative_integer(value: int, *, label: str) -> int:
    """拒绝 bool 与负值，固定累计量和规范长度为非负整数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


__all__ = [
    "INTEGER_EXTERNAL_SORT_BUDGET_SCHEMA",
    "INTEGER_EXTERNAL_SORT_IDENTITY_SCHEMA",
    "INTEGER_EXTERNAL_SORT_INPUT_IDENTITY_SCHEMA",
    "INTEGER_EXTERNAL_SORT_RESULT_SCHEMA",
    "IntegerExternalSortBudget",
    "IntegerExternalSortBudgetExceeded",
    "IntegerExternalSortError",
    "IntegerExternalSortIdentity",
    "IntegerExternalSortInputIdentity",
    "IntegerExternalSortResult",
    "external_sort_sealed_integer_records",
]
