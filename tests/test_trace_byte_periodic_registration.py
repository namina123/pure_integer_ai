"""周期后缀登记与冻结的完整逐位置参考编码逐字节相同。"""
import hashlib
from itertools import product

import pytest

from pure_integer_ai.storage.trace_byte_codec import _encode_block


def _reference_block(raw):
    """冻结原最近窗口/贪心算法，只作为独立物理字节参考。"""
    positions, output = {}, bytearray()
    cursor = anchor = 0
    size = len(raw)

    def length(value):
        """按原协议逐字节记录扩展长度。"""
        while value >= 255:
            output.append(255)
            value -= 255
        output.append(value)

    while cursor <= size - 12:
        prefix = raw[cursor:cursor + 4]
        previous = positions.get(prefix)
        positions[prefix] = cursor
        if previous is None:
            cursor += 1
            continue
        end = cursor + 4
        while end < size - 5 and raw[end] == raw[previous + end - cursor]:
            end += 1
        literals, match = cursor - anchor, end - cursor - 4
        output.append((min(literals, 15) << 4) | min(match, 15))
        if literals >= 15:
            length(literals - 15)
        output.extend(raw[anchor:cursor])
        output.extend((cursor - previous).to_bytes(2, 'little'))
        if match >= 15:
            length(match - 15)
        for position in range(cursor + 1, end):
            positions[raw[position:position + 4]] = position
        cursor = anchor = end
    literals = size - anchor
    output.append(min(literals, 15) << 4)
    if literals >= 15:
        length(literals - 15)
    output.extend(raw[anchor:])
    return bytes(output)


@pytest.mark.parametrize('period', [1, 2, 3, 4, 5, 7, 16, 31, 63, 127, 257])
def test_long_overlap_and_partial_final_window_match_original(period):
    """短offset长匹配和末尾跨界窗口必须保留原最近位置。"""
    seed = b''.join(hashlib.sha256(i.to_bytes(4, 'little')).digest()
                    for i in range((period + 31) // 32))[:period]
    raw = (seed * (6000 // period + 1))[:6000] + b'!tail!' + seed * 3 + b'final'
    assert _encode_block(raw) == _reference_block(raw)


def test_small_alphabets_and_adjacent_matches_preserve_all_windows():
    """穷举短周期及间隔拼接，覆盖重复窗口竞争与多个后继匹配。"""
    for size in range(1, 8):
        for values in product((0, 1), repeat=size):
            seed = bytes(values)
            raw = seed * 12 + b'XYZ' + seed[::-1] * 5 + seed + b'end'
            assert _encode_block(raw) == _reference_block(raw)


def test_incompressible_and_mixed_blocks_match_original():
    """原样块、长字面和混合局部周期保留完整字节结果。"""
    random = b''.join(hashlib.sha256(i.to_bytes(4, 'little')).digest() for i in range(1500))
    for raw in (random, random[:1000] + b'abcd' * 3000 + random[1000:3000],
                random[:17000] * 3 + random[-1000:]):
        assert len(raw) <= 65536
        assert _encode_block(raw) == _reference_block(raw)
