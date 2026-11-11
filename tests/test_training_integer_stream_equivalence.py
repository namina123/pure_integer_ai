"""训练整数流固定向量、跨 word 边界及非法物理表示。"""
import pytest

from pure_integer_ai.storage.training_candidate_event import (
    TrainingCandidateEventIntegrityError,
    decode_integer_stream,
    encode_integer_stream,
)


def _stream(raw, size):
    """只组装冻结的物理 byte 序列，不调用被测 varint 算法。"""
    return (1, size, len(raw), *(int.from_bytes(raw[start:start + 7], "little")
                              for start in range(0, len(raw), 7)))


@pytest.mark.parametrize("values,raw", [
    ((0,), b"\x00"),
    ((-1, 1), b"\x01\x02"),
    ((63, -64, 64, -65), b"\x7e\x7f\x80\x01\x81\x01"),
    (((1 << 63) - 1,), b"\xfe\xff\xff\xff\xff\xff\xff\xff\xff\x01"),
    ((-(1 << 63),), b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"),
    ((1 << 130,), b"\x80" * 18 + b"\x20"),
])
def test_integer_stream_matches_frozen_little_endian_vectors(values, raw):
    """边界及任意精度整数须逐 word 符合固定外部 byte 向量。"""
    expected = _stream(raw, len(values))
    assert encode_integer_stream(values) == expected
    assert decode_integer_stream(expected) == values


@pytest.mark.parametrize("raw,size,message", [
    (b"\x80\x00", 1, "不是规范编码"),
    (b"\x81\x00", 1, "不是规范编码"),
    (b"\x80" * 8 + b"\x00", 1, "不是规范编码"),
    (b"\x01" * 6 + b"\x80\x00", 7, "不是规范编码"),
    (b"\x01\x80", 1, "varint 被截断"),
    (b"\x80" * 9, 1, "varint 被截断"),
    (b"\x00\x00", 1, "原始长度不一致"),
    (b"\x80\x00", 2, "原始长度不一致"),
])
def test_integer_stream_keeps_noncanonical_and_truncated_failures(raw, size, message):
    """过长零组、跨 word 截断及原长度漂移保持原拒绝次序。"""
    with pytest.raises(TrainingCandidateEventIntegrityError, match=message):
        decode_integer_stream(_stream(raw, size))


@pytest.mark.parametrize("word", [True, -1, 1 << 56])
def test_integer_stream_rejects_non_integer_or_out_of_range_word(word):
    """word 的严格整数类型及 56-bit 上界不能被宿主 byte 转换放宽。"""
    with pytest.raises(TrainingCandidateEventIntegrityError, match="56-bit"):
        decode_integer_stream((1, 1, 1, word))


def test_integer_stream_checks_canonical_form_without_second_encode(monkeypatch):
    """规范性核验须覆盖整个已读流，不能再次物化同一完整编码。"""
    def forbidden(_values):
        """捕获重复编码，不改变待测输入。"""
        raise AssertionError("unexpected full stream re-encode")
    monkeypatch.setattr("pure_integer_ai.storage.training_candidate_event.encode_integer_stream", forbidden)
    assert decode_integer_stream(_stream(b"\x80\x01", 1)) == (64,)
    with pytest.raises(TrainingCandidateEventIntegrityError, match="不是规范编码"):
        decode_integer_stream(_stream(b"\x80\x00", 1))
