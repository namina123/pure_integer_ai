"""请求组合学习必须保留完整父来源、授权位置和原生图冷恢复。"""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef
from pure_integer_ai.cognition.understanding.query_operation_requests import FRAME, OperationRequestFrame
from pure_integer_ai.experiments.operation_request_course import materialize_operation_request_graph, read_operation_request_graph
from pure_integer_ai.experiments.operation_request_gap_learning import compile_train_operation_request_gap_record
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_w06_identity_protocol import W06_IDENTITY_VERSIONS


def _parent(value, gaps):
    """纯整数父记录只表示语法和 reader，不携带实体或答案。"""
    digest = hashlib.sha256(bytes((value,))).digest()
    source = SourceRef(FRAME, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1,
                       0, GLOBAL_OWNER_SCOPE, W06_IDENTITY_VERSIONS)
    return OperationRequestFrame((70,), source, tuple(digest), (61,), (62,),
                                 ((1, (71,), (72,), (4,)),), gaps)


@pytest.fixture
def instruction():
    parents = (_parent(1, ((97,), (98,))), _parent(2, ((99,), (100,))))
    return {"owner": "teacher", "split": "train", "license_id": "CC0-1.0",
            "parent_frame_keys": [list(frame.stable_key()) for frame in parents],
            "base_parent_ordinal": 0, "authorized_gap_ordinals": [0], "gap_parent_ordinals": [1, 0]}


def test_gap_learning_produces_a_deterministic_native_graph_with_its_own_source(instruction, tmp_path):
    """新来源绑定整个组合指令，冷图逐项恢复原生 reader 和组合后的语言间隔。"""
    frame = compile_train_operation_request_gap_record(instruction)
    assert frame.gaps == ((99,), (98,)) and frame.readers == ((1, (71,), (72,), (4,)),)
    assert frame.source_sha256 == tuple(hashlib.sha256(canonical_json_line(instruction)).digest())
    assert compile_train_operation_request_gap_record(deepcopy(instruction)) == frame
    database = tmp_path / "requests.sqlite3"
    materialize_operation_request_graph((frame,), database)
    assert read_operation_request_graph(database) == (frame,)


@pytest.mark.parametrize("field,value", [("owner", "evaluator"), ("split", "held_out"), ("license_id", "unknown")])
def test_gap_learning_cannot_consume_evaluator_or_unlicensed_instructions(instruction, field, value):
    instruction[field] = value
    with pytest.raises(ValueError, match="licensed teacher train"):
        compile_train_operation_request_gap_record(instruction)


@pytest.mark.parametrize("field,value", [("operation_key", (701,)), ("source_identity", (611,)),
                                        ("license_identity", (621,)),
                                        ("readers", ((1, (71,), (721,), (4,)),))])
def test_incompatible_parent_contracts_cannot_be_combined(instruction, field, value):
    """同一表面间隔不能把另一操作、角色或来源的授权带入当前请求。"""
    parent = OperationRequestFrame.from_stable_key(tuple(instruction["parent_frame_keys"][1]))
    instruction["parent_frame_keys"][1] = list(replace(parent, **{field: value}).stable_key())
    with pytest.raises(ValueError, match="cannot mix"):
        compile_train_operation_request_gap_record(instruction)


def test_instruction_cannot_modify_an_unlicensed_gap(instruction):
    instruction["gap_parent_ordinals"] = [0, 1]
    with pytest.raises(ValueError, match="unauthorized"):
        compile_train_operation_request_gap_record(instruction)


def test_existing_syntax_keeps_its_original_source(instruction):
    instruction["gap_parent_ordinals"] = [0, 0]
    with pytest.raises(ValueError, match="already exists"):
        compile_train_operation_request_gap_record(instruction)
