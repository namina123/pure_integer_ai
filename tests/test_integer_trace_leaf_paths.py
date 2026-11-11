"""严格整数叶子快路径必须保留规范字节、完整成员和原拒绝诊断。"""
from enum import IntEnum
import json

import pytest

from pure_integer_ai.experiments.batch_case_provenance import (
    CaseProvenanceError, _integer_projection,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError, canonical_json_bytes, parse_canonical_json_bytes,
)


def test_large_signed_integer_sequences_keep_every_canonical_byte():
    values = list(range(-2048, 2049)) + [10 ** 90, -(10 ** 90)]
    value = {"整数": values, "tuple": (0, 7), "bool": True, "null": None}
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    assert canonical_json_bytes(value) == expected
    assert parse_canonical_json_bytes(expected, require_object=True) == {
        **value, "tuple": [0, 7]}


@pytest.mark.parametrize("bad", [1.25, float("nan"), float("inf")])
def test_float_after_long_integer_prefix_retains_exact_nested_error_path(bad):
    value = {"rows": [[7], list(range(4096)) + [bad]]}
    with pytest.raises(DatasetContractError) as error:
        canonical_json_bytes(value)
    assert str(error.value) == "canonical_json.rows[1][4096] 禁止浮点"


def test_integer_subclass_never_receives_strict_integer_permission():
    non_strict = IntEnum("NonStrictInteger", {"VALUE": 7}).VALUE
    with pytest.raises(DatasetContractError) as error:
        canonical_json_bytes({"rows": [1, non_strict]})
    assert str(error.value) == "canonical_json.rows[1] 含 JSON 不支持类型 NonStrictInteger"
    with pytest.raises(CaseProvenanceError, match="unexpected non-integer"):
        _integer_projection({"rows": [1, non_strict]})


def test_integer_dict_value_does_not_bypass_invalid_object_key_check():
    with pytest.raises(DatasetContractError) as error:
        canonical_json_bytes({"rows": [{7: 3}]})
    assert str(error.value) == "canonical_json.rows[0] 的 object key 必须是非空字符串"


def test_projection_preserves_complete_members_and_copies_mutable_sequences():
    original = {"format": "trace", "rows": [list(range(4097)), (-2, 0, 8)], "null": None}
    projected = _integer_projection(original)
    assert projected == {"rows": [list(range(4097)), [-2, 0, 8]], "null": None}
    projected["rows"][0][4096] = 99
    assert original["rows"][0][4096] == 4096
    projected["rows"][1].append(10)
    assert original["rows"][1] == (-2, 0, 8)


@pytest.mark.parametrize("bad", [True, False, 0.5, "unlicensed", b"7", {1: 2}])
def test_projection_rejects_noninteger_after_long_integer_prefix(bad):
    with pytest.raises(CaseProvenanceError):
        _integer_projection({"rows": [list(range(4096)) + [bad]]})


def test_projection_only_strips_explicit_boundary_text_paths():
    assert _integer_projection({"response_surface": "原实际输出", "rows": [0, 1]}) == {
        "rows": [0, 1]}
    with pytest.raises(CaseProvenanceError, match="rows.response_surface"):
        _integer_projection({"rows": {"response_surface": "隐藏正文"}})
