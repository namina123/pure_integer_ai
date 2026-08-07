"""高频 segment 值结构的 slots、canonical 与兼容性回归。"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import asdict, replace

import pytest

from pure_integer_ai.storage.sealed_segment import (
    SealedSegment,
    SegmentBudget,
    SegmentRecord,
)


_RECORD = SegmentRecord((1, 2), (3, 4, 5, 6, 7, 8))
_STRUCT_VALUES = (
    SegmentBudget(1000, 1_000_000),
    _RECORD,
)
_SEALED = SealedSegment((11,), (12,), (13,), (), 14, (_RECORD,))
_SEALED_CANONICAL_SHA256 = (
    "f5a4f44527d919871d01e009763fbca049b363108ccc7e45d4f54804104a0312"
)


@pytest.mark.parametrize("value", _STRUCT_VALUES)
def test_segment_value_structs_have_no_instance_dict_and_round_trip(
        value: object,
        ) -> None:
    assert not hasattr(value, "__dict__")
    assert replace(value) == value
    assert asdict(replace(value)) == asdict(value)
    assert hash(replace(value)) == hash(value)
    assert pickle.loads(pickle.dumps(value, protocol=5)) == value


def test_sealed_segment_canonical_bytes_are_unchanged() -> None:
    assert hashlib.sha256(_SEALED.to_bytes()).hexdigest() == (
        _SEALED_CANONICAL_SHA256
    )
    assert SealedSegment.from_bytes(_SEALED.to_bytes()) == _SEALED
    assert pickle.loads(pickle.dumps(_SEALED, protocol=5)) == _SEALED
    assert bytes(_SEALED.checksum_key) == hashlib.sha256(
        _SEALED.to_bytes()).digest()
