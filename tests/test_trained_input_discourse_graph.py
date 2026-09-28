"""训练后输入话语 operator 图的物化与冷恢复验证。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    TrainedInputDiscourseOperator,
)
from pure_integer_ai.experiments.trained_input_discourse_graph import (
    materialize_input_discourse_operator_graph,
    read_input_discourse_operator_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_conjunction_graph import (
    TrainedInputDiscourseConjunction,
    materialize_input_discourse_conjunction_graph,
    read_input_discourse_conjunction_graph,
)


def test_operator_graph_cold_restore_preserves_integer_surface_and_identity(
        tmp_path):
    operators = (
        TrainedInputDiscourseOperator((91591, 1), (65292,)),
        TrainedInputDiscourseOperator((91591, 2), (65307,)),
    )
    source = (91595, 1, 11, 22)
    license_key = (91596, 1)
    database = tmp_path / "operator-graph.sqlite3"

    receipt = materialize_input_discourse_operator_graph(
        operators, database,
        source_identity=source,
        license_identity=license_key,
    )
    restored, metadata = read_input_discourse_operator_graph(database)

    assert receipt["operator_count"] == 2
    assert tuple(item.stable_key() for item in restored) == tuple(
        item.stable_key() for item in sorted(operators, key=lambda item: item.stable_key()))
    assert metadata["source_identity"] == source
    assert metadata["license_identity"] == license_key
    assert metadata["operator_count"] == 2


def test_operator_graph_rejects_reusing_existing_path(tmp_path):
    operator = TrainedInputDiscourseOperator((91591, 1), (65292,))
    database = tmp_path / "operator-graph.sqlite3"
    materialize_input_discourse_operator_graph(
        (operator,), database,
        source_identity=(91595, 1), license_identity=(91596, 1))

    try:
        materialize_input_discourse_operator_graph(
            (operator,), database,
            source_identity=(91595, 1), license_identity=(91596, 1))
    except ValueError as exc:
        assert "新路径" in str(exc)
    else:
        raise AssertionError("operator graph must not overwrite an existing run artifact")


def test_conjunction_graph_cold_restore_preserves_integer_contract(tmp_path):
    database = tmp_path / "conjunction-graph.sqlite3"
    conjunctions = (
        TrainedInputDiscourseConjunction((91591, 1)),
        TrainedInputDiscourseConjunction((91591, 2)),
    )
    receipt = materialize_input_discourse_conjunction_graph(
        conjunctions, database,
        source_identity=(91650, 1, 11, 22),
        license_identity=(91651, 1),
    )
    restored, metadata = read_input_discourse_conjunction_graph(database)
    assert receipt["conjunction_count"] == 2
    assert tuple(item.stable_key() for item in restored) == tuple(
        item.stable_key() for item in sorted(conjunctions,
                                             key=lambda item: item.stable_key()))
    assert metadata["conjunction_count"] == 2
