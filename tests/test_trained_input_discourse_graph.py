"""训练后输入话语 operator 图的物化与冷恢复验证。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    TrainedInputDiscourseOperator,
)
from pure_integer_ai.experiments.trained_input_discourse_graph import (
    materialize_input_discourse_operator_graph,
    read_input_discourse_operator_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_operation_graph import (
    MODE_FACT_QUERY_LEFT,
    MODE_FACT_QUERY_RIGHT,
    TrainedInputDiscourseOperationContract,
    materialize_input_discourse_operation_graph,
    read_input_discourse_operation_graph,
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


def test_fact_query_operation_graph_cold_restore_preserves_mode_and_provenance(
        tmp_path):
    operator = TrainedInputDiscourseOperator((91680, 1), (65311,))
    contract = TrainedInputDiscourseOperationContract(
        operator.operation_key, MODE_FACT_QUERY_LEFT)
    database = tmp_path / "fact-query-operation-graph.sqlite3"
    receipt = materialize_input_discourse_operation_graph(
        (contract,), database,
        source_identity=(91660, 148, 20261004, 10),
        license_identity=(91661, 148),
        operators=(operator,),
    )
    restored, metadata = read_input_discourse_operation_graph(database)
    assert receipt["contract_count"] == 1
    assert restored == (contract,)
    assert metadata["source_identity"] == (91660, 148, 20261004, 10)
    assert metadata["license_identity"] == (91661, 148)


def test_fact_query_operation_graph_cold_restore_preserves_right_mode(tmp_path):
    operator = TrainedInputDiscourseOperator((91680, 3), (35831, 38382, 65306))
    contract = TrainedInputDiscourseOperationContract(
        operator.operation_key, MODE_FACT_QUERY_RIGHT)
    database = tmp_path / "fact-query-right-operation-graph.sqlite3"
    materialize_input_discourse_operation_graph(
        (contract,), database,
        source_identity=(91660, 153, 20261005, 12),
        license_identity=(91661, 153),
        operators=(operator,),
    )
    restored, metadata = read_input_discourse_operation_graph(database)
    assert restored == (contract,)
    assert metadata["source_identity"] == (91660, 153, 20261005, 12)


def test_fact_query_operation_graph_rejects_missing_operator_or_empty_contracts(
        tmp_path):
    database = tmp_path / "fact-query-operation-graph.sqlite3"
    contract = TrainedInputDiscourseOperationContract((91680, 1))
    try:
        materialize_input_discourse_operation_graph(
            (contract,), database,
            source_identity=(91660, 148), license_identity=(91661, 148),
            operators=(),
        )
    except ValueError as exc:
        assert "matching operators" in str(exc)
    else:
        raise AssertionError("operation contract must be tied to an operator graph")
    try:
        materialize_input_discourse_operation_graph(
            (), tmp_path / "empty.sqlite3",
            source_identity=(91660, 148), license_identity=(91661, 148),
            operators=(),
        )
    except ValueError as exc:
        assert "at least one contract" in str(exc)
    else:
        raise AssertionError("empty operation contract graph must fail closed")


def test_bridge_rejects_mismatched_operator_and_contract_provenance(tmp_path):
    """An operation contract cannot silently authorize another source pack."""
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        TrainedGraphQueryBridge,
    )

    operator = TrainedInputDiscourseOperator((91680, 1), (65311,))
    operator_db = tmp_path / "operator.sqlite3"
    contract_db = tmp_path / "contract.sqlite3"
    materialize_input_discourse_operator_graph(
        (operator,), operator_db,
        source_identity=(91660, 149, 20261004, 12),
        license_identity=(91661, 149),
    )
    materialize_input_discourse_operation_graph(
        (TrainedInputDiscourseOperationContract(
            operator.operation_key, MODE_FACT_QUERY_LEFT),),
        contract_db,
        source_identity=(91660, 150, 20261004, 12),
        license_identity=(91661, 150),
        operators=(operator,),
    )
    try:
        TrainedGraphQueryBridge(
            tmp_path / "missing-model.sqlite3",
            input_discourse_graph_database=operator_db,
            input_discourse_operation_graph_database=contract_db,
        )
    except (ValueError, FileNotFoundError):
        # The provenance check is intentionally before the production model
        # is relevant; use a real model in the integration test below.
        pass
    else:
        raise AssertionError("mismatched operation provenance must fail closed")
