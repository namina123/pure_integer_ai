"""逐值最短varint充要条件与原完整重编码参考保持相同结果。"""
import pytest

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError, _read_unsigned, decode_integer_tuple, encode_integer_tuple,
)


def _reference(data, cursor):
    """冻结旧逐值读取和最短重编码逻辑，只用于独立回归对照。"""
    start, value, shift = cursor, 0, 0
    while True:
        if cursor >= len(data):
            raise IntegerCodecError('unsigned varint 被截断')
        byte = data[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            break
        shift += 7
    canonical = bytearray()
    remaining = value
    while remaining >= 128:
        canonical.append((remaining & 127) | 128)
        remaining >>= 7
    canonical.append(remaining)
    if bytes(canonical) != data[start:cursor]:
        raise IntegerCodecError('unsigned varint 不是最短编码')
    return value, cursor


def test_all_one_and_two_byte_sequences_match_original_reader():
    """穷举所有单字节及高位续接的二字节，涵盖截断和非最短表示。"""
    for first in range(256):
        values = [bytes((first,))]
        if first >= 128:
            values.extend(bytes((first, second)) for second in range(256))
        for data in values:
            try:
                expected = _reference(data, 0)
            except IntegerCodecError as error:
                with pytest.raises(IntegerCodecError, match=str(error)):
                    _read_unsigned(data, 0)
            else:
                assert _read_unsigned(data, 0) == expected


@pytest.mark.parametrize('width', [7, 14, 21, 63, 64, 127, 255, 511])
def test_signed_boundaries_round_trip_without_width_loss(width):
    """任意宽有符号整数和计数均保留原规范字节及完整成员。"""
    values = (0, 1, -1, (1 << width) - 1, 1 << width, -(1 << width), -(1 << width) - 1)
    encoded = encode_integer_tuple(values)
    assert decode_integer_tuple(encoded) == values


@pytest.mark.parametrize('data', [b'\x80\x00', b'\x81\x00', b'\xff\x80\x00',
    b'\x81\x80\x80\x00', b'\x80', b'\x01\x80\x00', b'\x01\x01\x00'])
def test_noncanonical_count_members_truncation_and_trailing_bytes_are_rejected(data):
    """逐值快捷校验和整流校验共同拒绝全部异常编码。"""
    with pytest.raises(IntegerCodecError):
        decode_integer_tuple(data)


@pytest.mark.parametrize('cursor', [-3, -2, -1, 0, 1, 2])
def test_existing_offset_behavior_matches_original(cursor):
    """不改变带前后缀读取以及旧宿主负索引的返回或失败。"""
    for data in (b'\x00\x01\x02', b'\x81\x01\x00', b'\x80\x00\x01'):
        try:
            expected = _reference(data, cursor)
        except (IntegerCodecError, IndexError) as error:
            with pytest.raises(type(error), match=str(error)):
                _read_unsigned(data, cursor)
        else:
            assert _read_unsigned(data, cursor) == expected
