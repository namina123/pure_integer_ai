"""存储协议共用的规范整数流编码和分帧读取。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class IntegerCodecError(ValueError):
    """整数流版本、varint、分帧或规范编码不完整。"""


# object-model: exception
class IntegerFramedStreamError(IntegerCodecError):
    """整数分帧流的物理边界、封存 footer 或完整性校验失败。"""


# object-model: exception
class IntegerFramedStreamBudgetExceeded(IntegerFramedStreamError):
    """整数分帧流的单帧、记录数或累计 payload 预算被超过。"""


# object-model: exception
class IntegerFramedStreamStateError(RuntimeError):
    """整数分帧流 writer 或 reader 的生命周期调用顺序非法。"""


INTEGER_FRAMED_STREAM_MAGIC = b"PIFRS\x01"
"""整数分帧流的固定物理 magic；schema 由 footer 的首字段声明。"""

INTEGER_FRAMED_STREAM_SCHEMA = 1
"""当前整数分帧流 footer 的固定 schema 整数。"""

_INTEGER_FRAMED_STREAM_SHA256_SIZE = hashlib.sha256().digest_size
_INTEGER_FRAMED_STREAM_FOOTER_FIELD_COUNT = (
    3 + _INTEGER_FRAMED_STREAM_SHA256_SIZE)


def strict_integer_tuple(
        value: tuple[int, ...], *, label: str, empty: bool = False,
        ) -> tuple[int, ...]:
    """核验协议值为严格整数 tuple，并按调用方要求处理空值。"""
    if not isinstance(value, tuple) or (not empty and not value):
        raise ValueError(f"{label} 必须是{'可空' if empty else '非空'}整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise ValueError(f"{label} 必须使用严格整数")
    return value


def encode_integer_tuple(values: tuple[int, ...]) -> bytes:
    """把严格整数 tuple 编成带元素计数的规范 zigzag varint 字节。"""
    strict_integer_tuple(values, label="integer codec values", empty=True)
    encoded = bytearray()
    _append_unsigned(encoded, len(values))
    for value in values:
        unsigned = value * 2 if value >= 0 else (-value * 2) - 1
        _append_unsigned(encoded, unsigned)
    return bytes(encoded)


def encoded_integer_tuple_size(values: tuple[int, ...]) -> int:
    """直接计算规范整数流字节数，不分配实际编码。"""
    strict_integer_tuple(values, label="integer codec values", empty=True)
    size = _unsigned_varint_size_validated(len(values))
    for value in values:
        size += _signed_varint_size_validated(value)
    return size


def encoded_framed_integer_tuple_size(
        values: tuple[tuple[int, ...], ...],
        ) -> int:
    """计算按长度分帧的多个整数键的规范编码字节数。"""
    if not isinstance(values, tuple):
        raise ValueError("framed integer codec values 必须是 tuple")
    item_count = 0
    for value in values:
        strict_integer_tuple(
            value, label="framed integer codec value", empty=True)
    return _encoded_framed_integer_tuple_size_validated(values)


def _encoded_framed_integer_tuple_size_validated(
        values: tuple[tuple[int, ...], ...],
        ) -> int:
    """计算调用方已经完整校验的分帧整数键编码长度。"""
    item_count = sum(1 + len(value) for value in values)
    size = _unsigned_varint_size_validated(item_count)
    for value in values:
        size += _signed_varint_size_validated(len(value))
        for item in value:
            size += _signed_varint_size_validated(item)
    return size


def decode_integer_tuple(data: bytes) -> tuple[int, ...]:
    """解码规范整数流，并拒绝截断、尾随字节和非最短 varint。"""
    if not isinstance(data, bytes) or not data:
        raise IntegerCodecError("integer codec data 必须是非空 bytes")
    cursor = 0
    size, cursor = _read_unsigned(data, cursor)
    result: list[int] = []
    for _ in range(size):
        unsigned, cursor = _read_unsigned(data, cursor)
        result.append(
            unsigned // 2 if unsigned % 2 == 0
            else -((unsigned + 1) // 2)
        )
    if cursor != len(data):
        raise IntegerCodecError("integer codec data 存在尾随字节")
    restored = tuple(result)
    if encode_integer_tuple(restored) != data:
        raise IntegerCodecError("integer codec data 不是规范编码")
    return restored


def pack_key(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长度整数键按长度分帧追加到目标流。"""
    strict_integer_tuple(value, label="packed integer key", empty=True)
    result.extend((len(value), *value))


def _append_unsigned(target: bytearray, value: int) -> None:
    """按最短 unsigned varint 形式追加一个非负整数。"""
    if type(value) is not int or value < 0:
        raise ValueError("unsigned varint 只能编码非负严格整数")
    while value >= 128:
        target.append((value & 127) | 128)
        value >>= 7
    target.append(value)


def _unsigned_varint_size(value: int) -> int:
    """返回非负严格整数的最短 unsigned varint 字节数。"""
    if type(value) is not int or value < 0:
        raise ValueError("unsigned varint 只能计算非负严格整数")
    return _unsigned_varint_size_validated(value)


def _unsigned_varint_size_validated(value: int) -> int:
    """返回调用方已校验非负整数的 varint 字节数。"""
    size = (value.bit_length() + 6) // 7
    return size if size else 1


def _signed_varint_size(value: int) -> int:
    """返回严格整数经 zigzag 后的最短 unsigned varint 字节数。"""
    if type(value) is not int:
        raise ValueError("zigzag varint 只能计算严格整数")
    return _signed_varint_size_validated(value)


def _signed_varint_size_validated(value: int) -> int:
    """返回调用方已校验严格整数的 zigzag varint 字节数。"""
    unsigned = value * 2 if value >= 0 else (-value * 2) - 1
    return _unsigned_varint_size_validated(unsigned)


def _read_unsigned(data: bytes, cursor: int) -> tuple[int, int]:
    """从指定位置读取一个最短 unsigned varint 及其后继位置。"""
    start = cursor
    value = 0
    shift = 0
    while True:
        if cursor >= len(data):
            raise IntegerCodecError("unsigned varint 被截断")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            break
        shift += 7
    canonical = bytearray()
    _append_unsigned(canonical, value)
    if bytes(canonical) != data[start:cursor]:
        raise IntegerCodecError("unsigned varint 不是最短编码")
    return value, cursor


def _write_all(stream: BinaryIO, data: bytes) -> None:
    """把一段有限字节完整写入资源 owner 持有的二进制流。"""
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if type(written) is not int or written <= 0 or written > len(view):
            raise OSError("整数分帧流写入未完成")
        view = view[written:]


def _read_exact(stream: BinaryIO, size: int, *, label: str) -> bytes:
    """只读取当前有限字段所需字节，短读立即按损坏输入拒绝。"""
    if type(size) is not int or size < 0:
        raise ValueError("整数分帧流读取长度必须是非负严格整数")
    result = bytearray()
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not isinstance(chunk, bytes) or not chunk:
            raise IntegerFramedStreamError(f"{label} 被截断")
        if len(chunk) > remaining:
            raise IntegerFramedStreamError(f"{label} 返回越界字节")
        result.extend(chunk)
        remaining -= len(chunk)
    return bytes(result)


def _read_bounded_unsigned(
        stream: BinaryIO,
        *,
        maximum: int,
        label: str,
        ) -> int:
    """流式读取规范 varint，并在超过字段上限前停止读取攻击输入。"""
    if type(maximum) is not int or maximum < 0:
        raise ValueError("整数分帧流字段上限必须是非负严格整数")
    encoded = bytearray()
    value = 0
    shift = 0
    for _ in range(_unsigned_varint_size_validated(maximum)):
        byte = _read_exact(stream, 1, label=label)[0]
        encoded.append(byte)
        value |= (byte & 127) << shift
        if value > maximum:
            raise IntegerFramedStreamBudgetExceeded(
                f"{label} 超过声明的读取预算")
        if byte < 128:
            canonical = bytearray()
            _append_unsigned(canonical, value)
            if bytes(canonical) != bytes(encoded):
                raise IntegerFramedStreamError(f"{label} 不是最短 varint")
            return value
        shift += 7
    raise IntegerFramedStreamError(
        f"{label} 非规范或超过声明的读取预算")


def _decode_zigzag(unsigned: int) -> int:
    """把已验证的 unsigned varint 值还原为一个严格整数。"""
    return unsigned // 2 if unsigned % 2 == 0 else -((unsigned + 1) // 2)


def _require_nonnegative_limit(value: int, *, label: str) -> int:
    """核验 reader 的显式硬预算为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _validate_framed_reader_limits(
        *,
        max_frame_bytes: int,
        max_record_count: int,
        max_total_payload_bytes: int,
        ) -> None:
    """在任何文件打开或句柄接管前完整核验 framed reader 的三项预算。"""
    _require_nonnegative_limit(
        max_frame_bytes, label="整数分帧流 max_frame_bytes")
    _require_nonnegative_limit(
        max_record_count, label="整数分帧流 max_record_count")
    _require_nonnegative_limit(
        max_total_payload_bytes,
        label="整数分帧流 max_total_payload_bytes",
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class IntegerFramedStreamFooter:
    """已封存整数分帧流的 schema、计数、payload 字节数和完整 digest。"""

    schema: int
    record_count: int
    total_payload_bytes: int
    payload_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """固定 footer schema、非负计数和完整 SHA-256 字节整数键。"""
        if type(self.schema) is not int or self.schema != INTEGER_FRAMED_STREAM_SCHEMA:
            raise ValueError("整数分帧流 footer schema 不兼容")
        _require_nonnegative_limit(
            self.record_count, label="整数分帧流 footer record_count")
        _require_nonnegative_limit(
            self.total_payload_bytes,
            label="整数分帧流 footer total_payload_bytes",
        )
        strict_integer_tuple(
            self.payload_sha256,
            label="整数分帧流 footer payload_sha256",
        )
        if (len(self.payload_sha256) != _INTEGER_FRAMED_STREAM_SHA256_SIZE
                or any(item < 0 or item > 255
                       for item in self.payload_sha256)):
            raise ValueError("整数分帧流 footer SHA-256 长度或字节范围错误")

    def integer_tuple(self) -> tuple[int, ...]:
        """返回 footer 的完整固定字段顺序的规范整数 tuple。"""
        return (
            self.schema,
            self.record_count,
            self.total_payload_bytes,
            *self.payload_sha256,
        )

    @classmethod
    def from_integer_tuple(
            cls,
            values: tuple[int, ...],
            ) -> "IntegerFramedStreamFooter":
        """从完整固定字段 tuple 重建 footer，不接受扩展尾字段。"""
        strict_integer_tuple(
            values,
            label="整数分帧流 footer values",
        )
        if len(values) != _INTEGER_FRAMED_STREAM_FOOTER_FIELD_COUNT:
            raise ValueError("整数分帧流 footer 字段数错误")
        return cls(
            values[0],
            values[1],
            values[2],
            values[3:],
        )


# object-model: resource_owner; representation=protocol; interop=pending
class IntegerFramedStreamWriter:
    """唯一持有新建目标的资源 owner，顺序追加纯整数记录并一次封存。"""

    __slots__ = (
        "path",
        "_digest",
        "_footer",
        "_poisoned",
        "_record_count",
        "_sealed",
        "_stream",
        "_total_payload_bytes",
    )

    def __init__(self, target: str | Path) -> None:
        """以排他新建方式写入 magic；root 选择和发布语义由上层负责。"""
        path = Path(target)
        stream = path.open("xb", buffering=0)
        self._initialize(path, stream)

    @classmethod
    def from_open_binary(
            cls, stream: BinaryIO, *, path: str | Path,
            ) -> "IntegerFramedStreamWriter":
        """接管一个已由上层排他打开的二进制写句柄并写入 magic。

        此入口不选择路径或实施文件系统策略；调用方必须先完成其物理边界核验。
        成功转交后 writer 独占并负责关闭 ``stream``，不能再由调用方写入。
        """
        if not all(callable(getattr(stream, name, None))
                   for name in ("write", "flush", "close")):
            raise TypeError("整数分帧流 writer stream 必须是可写二进制句柄")
        result = cls.__new__(cls)
        result._initialize(Path(path), stream)
        return result

    def _initialize(self, path: Path, stream: BinaryIO) -> None:
        """建立唯一 writer 生命周期并在任何初始写入失败时关闭已接管句柄。"""
        self.path = path
        self._digest = hashlib.sha256()
        self._footer: IntegerFramedStreamFooter | None = None
        self._poisoned = False
        self._record_count = 0
        self._sealed = False
        self._stream: BinaryIO | None = stream
        self._total_payload_bytes = 0
        try:
            _write_all(stream, INTEGER_FRAMED_STREAM_MAGIC)
        except BaseException:
            self._close_quietly()
            raise

    def __enter__(self) -> "IntegerFramedStreamWriter":
        """返回唯一 writer，退出上下文时只关闭资源而不自动封存。"""
        return self

    def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
            ) -> None:
        """结束 writer 生命周期；异常和未 seal 文件保持为未发布状态。"""
        self.close()

    @property
    def sealed(self) -> bool:
        """返回 footer 是否已完整写入并关闭 writer 资源。"""
        return self._sealed

    @property
    def footer(self) -> IntegerFramedStreamFooter | None:
        """返回 seal 后的固定 footer；未 seal 时不提供发布身份。"""
        return self._footer

    def append(self, record: tuple[int, ...]) -> None:
        """顺序追加一条严格整数 record，并只把该 payload 纳入 digest。"""
        stream = self._require_writable()
        strict_integer_tuple(record, label="整数分帧流 record", empty=True)
        payload = encode_integer_tuple(record)
        if not payload:
            raise IntegerFramedStreamError("整数分帧流 record payload 不得为空")
        length = bytearray()
        _append_unsigned(length, len(payload))
        try:
            _write_all(stream, bytes(length))
            _write_all(stream, payload)
            self._digest.update(payload)
            self._record_count += 1
            self._total_payload_bytes += len(payload)
        except BaseException:
            self._poison()
            raise

    def seal(self) -> IntegerFramedStreamFooter:
        """写入唯一 footer sentinel 后关闭资源，之后不允许继续追加或重复封存。"""
        stream = self._require_writable()
        footer = IntegerFramedStreamFooter(
            INTEGER_FRAMED_STREAM_SCHEMA,
            self._record_count,
            self._total_payload_bytes,
            tuple(self._digest.digest()),
        )
        try:
            _write_all(stream, b"\x00")
            _write_all(stream, encode_integer_tuple(footer.integer_tuple()))
            stream.flush()
            self.close()
            self._footer = footer
            self._sealed = True
            return footer
        except BaseException:
            self._poison()
            raise

    def close(self) -> None:
        """关闭 writer 资源；未 seal 文件保留给上层按未发布残片处理。"""
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()

    def _require_writable(self) -> BinaryIO:
        """要求 writer 仍处于唯一可追加且尚未封存的生命周期状态。"""
        if self._poisoned:
            raise IntegerFramedStreamStateError(
                "整数分帧流 writer 已因不完整写入失效")
        if self._sealed:
            raise IntegerFramedStreamStateError("整数分帧流 writer 已 seal")
        if self._stream is None:
            raise IntegerFramedStreamStateError("整数分帧流 writer 已关闭")
        return self._stream

    def _close_quietly(self) -> None:
        """在构造失败路径释放文件句柄，不掩盖原始写入异常。"""
        try:
            self.close()
        except OSError:
            pass

    def _poison(self) -> None:
        """物理写入失败后永久关闭 writer，避免残片被继续追加或错误封存。"""
        self._poisoned = True
        self._close_quietly()


# object-model: resource_owner; representation=protocol; interop=pending
class IntegerFramedStreamReader:
    """按 frame 读取已封存整数流；只保留当前 record 和固定 footer。"""

    __slots__ = (
        "path",
        "max_frame_bytes",
        "max_record_count",
        "max_total_payload_bytes",
        "_digest",
        "_failed",
        "_footer",
        "_record_count",
        "_stream",
        "_total_payload_bytes",
    )

    def __init__(
            self,
            source: str | Path,
            *,
            max_frame_bytes: int,
            max_record_count: int,
            max_total_payload_bytes: int,
            ) -> None:
        """打开既有流并只读取固定 magic，正文与 footer 保持惰性逐条验证。"""
        _validate_framed_reader_limits(
            max_frame_bytes=max_frame_bytes,
            max_record_count=max_record_count,
            max_total_payload_bytes=max_total_payload_bytes,
        )
        path = Path(source)
        stream = path.open("rb", buffering=0)
        self._initialize(
            path,
            stream,
            max_frame_bytes=max_frame_bytes,
            max_record_count=max_record_count,
            max_total_payload_bytes=max_total_payload_bytes,
        )

    @classmethod
    def from_open_binary(
            cls,
            stream: BinaryIO,
            *,
            path: str | Path,
            max_frame_bytes: int,
            max_record_count: int,
            max_total_payload_bytes: int,
            ) -> "IntegerFramedStreamReader":
        """接管一个已由上层核验的二进制读句柄并立即验证 magic。

        此入口不自行打开 ``path``，以便上层把目录能力和 no-follow 检查保持在实际
        handle 上。成功转交后 reader 独占并负责关闭 ``stream``。
        """
        if not all(callable(getattr(stream, name, None))
                   for name in ("read", "close")):
            raise TypeError("整数分帧流 reader stream 必须是可读二进制句柄")
        _validate_framed_reader_limits(
            max_frame_bytes=max_frame_bytes,
            max_record_count=max_record_count,
            max_total_payload_bytes=max_total_payload_bytes,
        )
        path_value = Path(path)
        result = cls.__new__(cls)
        result._initialize(
            path_value,
            stream,
            max_frame_bytes=max_frame_bytes,
            max_record_count=max_record_count,
            max_total_payload_bytes=max_total_payload_bytes,
        )
        return result

    def _initialize(
            self,
            path: Path,
            stream: BinaryIO,
            *,
            max_frame_bytes: int,
            max_record_count: int,
            max_total_payload_bytes: int,
            ) -> None:
        """建立有界 reader cursor；magic 失败时关闭已接管句柄。"""
        self.path = path
        self.max_frame_bytes = _require_nonnegative_limit(
            max_frame_bytes, label="整数分帧流 max_frame_bytes")
        self.max_record_count = _require_nonnegative_limit(
            max_record_count, label="整数分帧流 max_record_count")
        self.max_total_payload_bytes = _require_nonnegative_limit(
            max_total_payload_bytes,
            label="整数分帧流 max_total_payload_bytes",
        )
        self._digest = hashlib.sha256()
        self._failed = False
        self._footer: IntegerFramedStreamFooter | None = None
        self._record_count = 0
        self._stream: BinaryIO | None = stream
        self._total_payload_bytes = 0
        try:
            magic = _read_exact(
                self._require_open(),
                len(INTEGER_FRAMED_STREAM_MAGIC),
                label="整数分帧流 magic",
            )
            if magic != INTEGER_FRAMED_STREAM_MAGIC:
                raise IntegerFramedStreamError("整数分帧流 magic 不匹配")
        except BaseException:
            self._abort()
            raise

    def __enter__(self) -> "IntegerFramedStreamReader":
        """返回有界 cursor；调用方须消费至 footer 才能取得封存校验结果。"""
        return self

    def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
            ) -> None:
        """结束 reader 生命周期；提前关闭不会把未验证输入视为已封存。"""
        self.close()

    def __iter__(self) -> "IntegerFramedStreamReader":
        """返回自身作为逐条 record iterator，不聚合全体记录。"""
        return self

    def __next__(self) -> tuple[int, ...]:
        """返回下一条 record；footer 校验成功后以 StopIteration 结束。"""
        record = self.read_record()
        if record is None:
            raise StopIteration
        return record

    @property
    def footer(self) -> IntegerFramedStreamFooter | None:
        """只在完整读到且验证 footer 后返回封存身份。"""
        return self._footer

    @property
    def record_count(self) -> int:
        """返回当前已验证 payload record 数，不包含 footer。"""
        return self._record_count

    @property
    def total_payload_bytes(self) -> int:
        """返回当前已验证 payload 的累计字节数。"""
        return self._total_payload_bytes

    def read_record(self) -> tuple[int, ...] | None:
        """读取一个 frame，或在 footer 完整核验后返回 None。"""
        if self._failed:
            raise IntegerFramedStreamStateError("整数分帧流 reader 已因损坏输入关闭")
        if self._footer is not None:
            return None
        try:
            stream = self._require_open()
            payload_size = _read_bounded_unsigned(
                stream,
                maximum=self.max_frame_bytes,
                label="整数分帧流 frame length",
            )
            if payload_size == 0:
                self._footer = self._read_footer(stream)
                self.close()
                return None
            if self._record_count >= self.max_record_count:
                raise IntegerFramedStreamBudgetExceeded(
                    "整数分帧流超过 record 数预算")
            if (payload_size
                    > self.max_total_payload_bytes - self._total_payload_bytes):
                raise IntegerFramedStreamBudgetExceeded(
                    "整数分帧流超过累计 payload 字节预算")
            payload = _read_exact(
                stream,
                payload_size,
                label="整数分帧流 record payload",
            )
            record = decode_integer_tuple(payload)
            self._digest.update(payload)
            self._record_count += 1
            self._total_payload_bytes += payload_size
            return record
        except (IntegerCodecError, OSError, ValueError) as exc:
            self._failed = True
            self._abort()
            if isinstance(exc, IntegerFramedStreamError):
                raise
            raise IntegerFramedStreamError("整数分帧流 record 编码损坏") from exc

    def finish(self) -> IntegerFramedStreamFooter:
        """不聚合地消费余下 record，并返回经 digest/计数校验的 footer。"""
        while self.read_record() is not None:
            pass
        if self._footer is None:
            raise IntegerFramedStreamStateError("整数分帧流 reader 未取得 footer")
        return self._footer

    def close(self) -> None:
        """关闭 reader 资源；提前关闭不改变 footer 的未验证状态。"""
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()

    def _read_footer(self, stream: BinaryIO) -> IntegerFramedStreamFooter:
        """流式读取固定小 footer，核验 count、字节数、digest 与无尾随字节。"""
        field_count = _read_bounded_unsigned(
            stream,
            maximum=_INTEGER_FRAMED_STREAM_FOOTER_FIELD_COUNT,
            label="整数分帧流 footer field count",
        )
        if field_count != _INTEGER_FRAMED_STREAM_FOOTER_FIELD_COUNT:
            raise IntegerFramedStreamError("整数分帧流 footer 字段数错误")
        values = [
            _decode_zigzag(_read_bounded_unsigned(
                stream,
                maximum=INTEGER_FRAMED_STREAM_SCHEMA * 2,
                label="整数分帧流 footer schema",
            )),
            _decode_zigzag(_read_bounded_unsigned(
                stream,
                maximum=self.max_record_count * 2,
                label="整数分帧流 footer record_count",
            )),
            _decode_zigzag(_read_bounded_unsigned(
                stream,
                maximum=self.max_total_payload_bytes * 2,
                label="整数分帧流 footer total_payload_bytes",
            )),
        ]
        for index in range(_INTEGER_FRAMED_STREAM_SHA256_SIZE):
            values.append(_decode_zigzag(_read_bounded_unsigned(
                stream,
                maximum=255 * 2,
                label=f"整数分帧流 footer SHA-256[{index}]",
            )))
        footer = IntegerFramedStreamFooter.from_integer_tuple(tuple(values))
        if footer.record_count != self._record_count:
            raise IntegerFramedStreamError("整数分帧流 footer record_count 不匹配")
        if footer.total_payload_bytes != self._total_payload_bytes:
            raise IntegerFramedStreamError(
                "整数分帧流 footer total_payload_bytes 不匹配")
        if footer.payload_sha256 != tuple(self._digest.digest()):
            raise IntegerFramedStreamError("整数分帧流 footer SHA-256 不匹配")
        if stream.read(1):
            raise IntegerFramedStreamError("整数分帧流 footer 后存在尾随字节")
        return footer

    def _require_open(self) -> BinaryIO:
        """要求 reader 尚未关闭且尚未通过 footer 结束。"""
        if self._stream is None:
            raise IntegerFramedStreamStateError("整数分帧流 reader 已关闭")
        return self._stream

    def _abort(self) -> None:
        """在损坏或预算失败路径关闭句柄，避免后续继续消费不可信输入。"""
        try:
            self.close()
        except OSError:
            pass


@dataclass
class IntegerStreamReader:
    """在已解码整数 tuple 上执行带边界检查的顺序读取。"""

    values: tuple[int, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        """核验输入流和初始游标，禁止越界起步。"""
        strict_integer_tuple(
            self.values, label="integer stream reader values", empty=True)
        if type(self.cursor) is not int or not 0 <= self.cursor <= len(self.values):
            raise ValueError("integer stream reader cursor 越界")

    def read(self, *, label: str) -> int:
        """读取一个严格整数，流已结束时报告字段名。"""
        if self.cursor >= len(self.values):
            raise IntegerCodecError(f"整数流缺少字段: {label}")
        value = self.values[self.cursor]
        self.cursor += 1
        return value

    def read_nonnegative(self, *, label: str) -> int:
        """读取一个非负严格整数。"""
        value = self.read(label=label)
        if value < 0:
            raise IntegerCodecError(f"{label} 必须是非负整数")
        return value

    def read_positive(self, *, label: str) -> int:
        """读取一个正严格整数。"""
        value = self.read(label=label)
        if value <= 0:
            raise IntegerCodecError(f"{label} 必须是正整数")
        return value

    def read_key(self, *, label: str, empty: bool = False) -> tuple[int, ...]:
        """读取一个长度分帧整数键并核验是否允许为空。"""
        size = self.read_nonnegative(label=f"{label}.size")
        end = self.cursor + size
        if end > len(self.values):
            raise IntegerCodecError(f"{label} 被截断")
        value = self.values[self.cursor:end]
        self.cursor = end
        return strict_integer_tuple(value, label=label, empty=empty)

    def finish(self) -> None:
        """要求调用方已经消费完整流，拒绝未知尾字段。"""
        if self.cursor != len(self.values):
            raise IntegerCodecError("整数流存在未消费尾字段")


__all__ = [
    "IntegerCodecError",
    "IntegerFramedStreamBudgetExceeded",
    "IntegerFramedStreamError",
    "IntegerFramedStreamFooter",
    "IntegerFramedStreamReader",
    "IntegerFramedStreamStateError",
    "IntegerFramedStreamWriter",
    "IntegerStreamReader",
    "INTEGER_FRAMED_STREAM_MAGIC",
    "INTEGER_FRAMED_STREAM_SCHEMA",
    "decode_integer_tuple",
    "encoded_framed_integer_tuple_size",
    "encoded_integer_tuple_size",
    "encode_integer_tuple",
    "pack_key",
    "strict_integer_tuple",
]
