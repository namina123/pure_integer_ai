"""规范整数流长度直算与实际编码的逐边界等价回归。"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.integer_codec import (
    encoded_framed_integer_tuple_size,
    encoded_integer_tuple_size,
    encode_integer_tuple,
    pack_key,
)
from pure_integer_ai.storage.sealed_segment import SegmentRecord


_BOUNDARY_VALUES = (
    0,
    1,
    -1,
    63,
    -64,
    64,
    -65,
    8191,
    -8192,
    8192,
    -8193,
    2 ** 63,
    -(2 ** 63),
    2 ** 511,
    -(2 ** 511),
)


@pytest.mark.parametrize(
    "values",
    (
        (),
        _BOUNDARY_VALUES,
        tuple(range(127)),
        tuple(range(128)),
    ),
)
def test_encoded_integer_tuple_size_matches_canonical_bytes(
        values: tuple[int, ...],
        ) -> None:
    assert encoded_integer_tuple_size(values) == len(
        encode_integer_tuple(values))


@pytest.mark.parametrize("invalid", ([1], (True,), (1.0,)))
def test_encoded_integer_tuple_size_rejects_non_strict_integer_tuple(
        invalid: object,
        ) -> None:
    with pytest.raises(Exception) as encoded_error:
        encode_integer_tuple(invalid)  # type: ignore[arg-type]
    with pytest.raises(type(encoded_error.value)):
        encoded_integer_tuple_size(invalid)  # type: ignore[arg-type]


def test_segment_record_size_matches_unchanged_canonical_encoding() -> None:
    record = SegmentRecord(
        (1, 63, 64, 8192),
        (-1, -64, -65, -8192, -8193, 2 ** 127),
    )
    assert record.size_bytes() == len(
        encode_integer_tuple(record.integer_stream()))


@pytest.mark.parametrize(
    "values",
    (
        (),
        ((),),
        ((1, 63, 64), (-1, -64, -65)),
        (tuple(range(127)), tuple(range(128))),
        ((2 ** 511,), (-(2 ** 511),)),
    ),
)
def test_framed_integer_tuple_size_matches_packed_canonical_bytes(
        values: tuple[tuple[int, ...], ...],
        ) -> None:
    stream: list[int] = []
    for value in values:
        pack_key(stream, value)
    assert encoded_framed_integer_tuple_size(values) == len(
        encode_integer_tuple(tuple(stream)))


def test_framed_integer_tuple_size_rejects_non_tuple_container() -> None:
    with pytest.raises(ValueError, match="必须是 tuple"):
        encoded_framed_integer_tuple_size([])  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", (([1],), ((True,),), ((1.0,),)))
def test_framed_integer_tuple_size_rejects_invalid_keys(
        invalid: object,
        ) -> None:
    with pytest.raises(Exception):
        encoded_framed_integer_tuple_size(invalid)  # type: ignore[arg-type]
