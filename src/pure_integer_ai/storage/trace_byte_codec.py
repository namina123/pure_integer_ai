"""按冻结的 LZ4 子集逐块编码诊断 bytes；不改变图记录的规范身份。"""
from __future__ import annotations

from collections.abc import Callable, Iterator


TRACE_BYTE_ENCODING = "PIA_LZ4_BLOCKS_V1"
MAGIC = b"PIALZ4B1"
BLOCK_BYTES = 65536


class TraceByteCodecError(ValueError):
    """物理字节流损坏、非规范或超过显式解码预算。"""


def _length(output: bytearray, value: int) -> None:
    """追加 LZ4 长度扩展，255 的倍数也必须有末尾零。"""
    while value >= 255:
        output.append(255)
        value -= 255
    output.append(value)


def _encode_block(raw: bytes) -> bytes:
    """最近四字节位置、贪心最长匹配；登记顺序与 tie break 唯一。"""
    positions: dict[bytes, int] = {}
    output = bytearray()
    cursor = anchor = 0
    size = len(raw)
    while cursor <= size - 12:
        prefix = raw[cursor:cursor + 4]
        previous = positions.get(prefix)
        positions[prefix] = cursor
        if previous is None:
            cursor += 1
            continue
        end = cursor + 4
        # 完全相等的有界字节块可整体越过；首个差异仍逐字节定位。
        while end + 64 <= size - 5:
            previous_end = previous + end - cursor
            if raw[end:end + 64] != raw[previous_end:previous_end + 64]:
                break
            end += 64
        while end < size - 5 and raw[end] == raw[previous + end - cursor]:
            end += 1
        literal_size, match_size = cursor - anchor, end - cursor - 4
        output.append((min(literal_size, 15) << 4) | min(match_size, 15))
        if literal_size >= 15:
            _length(output, literal_size - 15)
        output.extend(raw[anchor:cursor])
        output.extend((cursor - previous).to_bytes(2, "little"))
        if match_size >= 15:
            _length(output, match_size - 15)
        # 匹配区间以offset为周期；较早四字节窗口必被后继同窗覆盖。
        # 末尾三个窗口可能跨出匹配区间，必须按原顺序全部登记。
        last_period = max(cursor + 1, end - (cursor - previous) - 3)
        for position in range(last_period, end):
            positions[raw[position:position + 4]] = position
        cursor = anchor = end
    literal_size = size - anchor
    output.append(min(literal_size, 15) << 4)
    if literal_size >= 15:
        _length(output, literal_size - 15)
    output.extend(raw[anchor:])
    return bytes(output)


def iter_encoded_trace_bytes(raw: bytes, *, checkpoint: Callable[[], None] | None = None) -> Iterator[bytes]:
    """逐块产生唯一编码，保存方无需同时持有完整压缩副本。"""
    if not isinstance(raw, bytes) or not raw or len(raw) >= 1 << 64:
        raise TraceByteCodecError("trace bytes must be nonempty and fit uint64")
    yield MAGIC + len(raw).to_bytes(8, "little")
    for start in range(0, len(raw), BLOCK_BYTES):
        if checkpoint is not None:
            checkpoint()
        block = raw[start:start + BLOCK_BYTES]
        encoded = _encode_block(block)
        stored_raw = len(encoded) >= len(block)
        payload = block if stored_raw else encoded
        yield (len(payload) | (int(stored_raw) << 31)).to_bytes(4, "little") + payload


def encode_trace_bytes(raw: bytes, *, checkpoint: Callable[[], None] | None = None) -> bytes:
    """为小型调用者和固定向量合并逐块编码，字节与流式保存相同。"""
    return b"".join(iter_encoded_trace_bytes(raw, checkpoint=checkpoint))


def _extended(data: bytes, cursor: int, value: int, maximum: int) -> tuple[int, int]:
    """读取长度扩展，在申请输出空间前拒绝越界。"""
    if value == 15:
        while True:
            if cursor >= len(data):
                raise TraceByteCodecError("truncated LZ4 length")
            item = data[cursor]
            cursor += 1
            value += item
            if value > maximum:
                raise TraceByteCodecError("LZ4 length exceeds block budget")
            if item != 255:
                break
    return value, cursor


def _decode_block(data: bytes, expected: int) -> bytes:
    """只允许当前块内的正 offset；重叠复制按 LZ4 字节序展开。"""
    output = bytearray()
    cursor = 0
    while cursor < len(data):
        token = data[cursor]
        cursor += 1
        literals, cursor = _extended(data, cursor, token >> 4, expected - len(output))
        if literals > expected - len(output) or cursor + literals > len(data):
            raise TraceByteCodecError("truncated or oversized LZ4 literals")
        output.extend(data[cursor:cursor + literals])
        cursor += literals
        if cursor == len(data):
            break
        if cursor + 2 > len(data):
            raise TraceByteCodecError("truncated LZ4 offset")
        distance = int.from_bytes(data[cursor:cursor + 2], "little")
        cursor += 2
        if not 1 <= distance <= len(output):
            raise TraceByteCodecError("LZ4 offset is outside the current block")
        length, cursor = _extended(data, cursor, token & 15, expected - len(output))
        length += 4
        if length > expected - len(output):
            raise TraceByteCodecError("LZ4 match exceeds block budget")
        while length:
            count = min(length, distance)
            output.extend(output[-distance:][:count])
            length -= count
    if len(output) != expected:
        raise TraceByteCodecError("decoded block length differs")
    return bytes(output)


def decode_trace_bytes(data: bytes, *, max_decoded_bytes: int) -> bytes:
    """有界恢复完整原始 bytes，并重编码核对唯一的规范物理表示。"""
    if type(max_decoded_bytes) is not int or max_decoded_bytes <= 0:
        raise TraceByteCodecError("decoded byte budget must be a positive strict integer")
    if not isinstance(data, bytes) or len(data) < 16 or data[:8] != MAGIC:
        raise TraceByteCodecError("trace byte codec header differs")
    expected = int.from_bytes(data[8:16], "little")
    if not 1 <= expected <= max_decoded_bytes:
        raise TraceByteCodecError("decoded trace exceeds byte budget")
    output = bytearray()
    cursor = 16
    while len(output) < expected:
        if cursor + 4 > len(data):
            raise TraceByteCodecError("truncated block header")
        word = int.from_bytes(data[cursor:cursor + 4], "little")
        cursor += 4
        stored_raw, size = word >> 31, word & 0x7fffffff
        block_size = min(BLOCK_BYTES, expected - len(output))
        if not 1 <= size <= block_size or cursor + size > len(data):
            raise TraceByteCodecError("stored block size differs or is truncated")
        payload = data[cursor:cursor + size]
        cursor += size
        block = payload if stored_raw else _decode_block(payload, block_size)
        if len(block) != block_size:
            raise TraceByteCodecError("raw block length differs")
        encoded = _encode_block(block)
        canonical_raw = len(encoded) >= len(block)
        if stored_raw != int(canonical_raw) or payload != (block if canonical_raw else encoded):
            raise TraceByteCodecError("noncanonical trace block encoding")
        output.extend(block)
    if cursor != len(data):
        raise TraceByteCodecError("trailing bytes after trace blocks")
    return bytes(output)
