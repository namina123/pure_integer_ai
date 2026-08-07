"""高频纯整数值结构的 slots 与兼容性回归。"""
from __future__ import annotations

import pickle
from dataclasses import asdict, replace

import pytest

from pure_integer_ai.crosscut.integer.valtypes import FixedQuotient, Rational


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (Rational(3, 7), {"num": 3, "den": 7}),
        (
            FixedQuotient(M=3, r=1, k=2, b=7),
            {"M": 3, "r": 1, "k": 2, "b": 7},
        ),
    ),
)
def test_integer_value_structs_have_no_instance_dict_and_round_trip(
        value: object,
        expected: dict[str, int],
        ) -> None:
    assert not hasattr(value, "__dict__")
    assert asdict(value) == expected
    assert replace(value) == value
    assert hash(replace(value)) == hash(value)
    assert pickle.loads(pickle.dumps(value, protocol=5)) == value
