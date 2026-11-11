"""Query cases are split independently from teacher facts and evaluator labels."""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.experiments.operation_query_split_course import (
    OPERATION_DUTIES, OperationQuerySplitError, audit_operation_query_splits,
    read_train_operation_queries,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _course_files(root):
    queries, labels = {}, {}
    for split in ("train", "held_out", "negative", "conflict"):
        owner = "teacher" if split == "train" else "evaluator"
        base = root / ("train" if split == "train" else f"owners/evaluator/{split}")
        rows = tuple({
            "case_id": f"{split}-{ordinal:02d}",
            "chain_id": f"{split}-chain-{ordinal:02d}",
            "duty": OPERATION_DUTIES[(ordinal - 1) % len(OPERATION_DUTIES)],
            "question": f"{split} case {ordinal:02d} 自然语言请求。",
            "split": split, "owner": owner, "license_id": "CC0-1.0",
        } for ordinal in range(1, 21))
        queries[split] = base / "queries.jsonl"
        _write_jsonl(queries[split], rows)
        if split != "train":
            policy = {"held_out": "ANSWER_CLOSED", "negative": "UNKNOWN",
                      "conflict": "CLARIFY_CONFLICT"}[split]
            labels[split] = root / f"owners/evaluator/{split}-labels.jsonl"
            _write_jsonl(labels[split], tuple({
                "case_id": row["case_id"], "policy": policy,
                "split": split, "owner": "evaluator", "license_id": "CC0-1.0",
            } for row in rows))
    return queries, labels


def _audit(paths, labels):
    return audit_operation_query_splits(
        paths["train"], paths["held_out"], paths["negative"], paths["conflict"],
        labels["held_out"], labels["negative"], labels["conflict"])


def test_four_way_query_splits_are_separate_from_facts_and_labels(tmp_path):
    paths, labels = _course_files(tmp_path)
    courses, audit = _audit(paths, labels)

    assert {split: len(rows) for split, rows in courses.items()} == {
        "train": 20, "held_out": 20, "negative": 20, "conflict": 20}
    assert audit.status == "PASS_QUERY_SPLIT_STRUCTURE_ONLY"
    assert audit.train_duty_count == audit.held_out_duty_count == 8
    assert audit.semantic_behavior_qualification == "UNKNOWN_NOT_RUN"
    assert audit.novel_topology_status == "UNKNOWN_NOT_REPRESENTED"
    assert audit.production_authorized == audit.free_dialogue_complete == 0
    assert all(not hasattr(case, "seed_ids") for rows in courses.values() for case in rows)
    assert len(read_train_operation_queries(paths["train"])) == 20


@pytest.mark.parametrize("leak", ["chain", "question", "case_id"])
def test_cross_split_identity_or_input_leak_is_rejected(tmp_path, leak):
    paths, labels = _course_files(tmp_path)
    held_out = [json.loads(line) for line in paths["held_out"].read_bytes().splitlines()]
    train = [json.loads(line) for line in paths["train"].read_bytes().splitlines()]
    field = {"chain": "chain_id", "question": "question", "case_id": "case_id"}[leak]
    held_out[0][field] = train[0][field]
    _write_jsonl(paths["held_out"], held_out)

    with pytest.raises(OperationQuerySplitError, match="crosses query splits|cross query splits"):
        _audit(paths, labels)


def test_evaluator_label_policy_is_bound_to_its_physical_case_owner(tmp_path):
    paths, labels = _course_files(tmp_path)
    rows = [json.loads(line) for line in labels["negative"].read_bytes().splitlines()]
    rows[0]["policy"] = "ANSWER_CLOSED"
    _write_jsonl(labels["negative"], rows)

    with pytest.raises(OperationQuerySplitError, match="evaluator labels"):
        _audit(paths, labels)


def test_train_reader_refuses_evaluator_owned_inputs(tmp_path):
    paths, _labels = _course_files(tmp_path)

    with pytest.raises(OperationQuerySplitError, match="train input only"):
        read_train_operation_queries(paths["held_out"])


def test_query_schema_rejects_fact_seed_or_answer_payloads(tmp_path):
    paths, labels = _course_files(tmp_path)
    rows = [json.loads(line) for line in paths["train"].read_bytes().splitlines()]
    rows[0]["seed_ids"] = ["teacher-fact-01"]
    _write_jsonl(paths["train"], rows)

    with pytest.raises(OperationQuerySplitError, match="fields drifted"):
        _audit(paths, labels)
