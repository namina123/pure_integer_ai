"""真实 learning 中 support 与 conflict 的完整消费分母回归。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_typed_relation_bridge import build_integer_w06_learning_runtime
from pure_integer_ai.experiments.independent_relation_fact_source import compile_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_graph_course import write_integer_relation_graph_course
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.relation_training_integrity import verify_relation_training_integrity
from pure_integer_ai.experiments.run_conversation_training import dialogue_semantic_protocols
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend


def test_real_supported_and_conflicted_candidates_are_both_complete(tmp_path):
    """冲突保留 H-00 证据，缺失课程候选与额外候选仍必须拒绝。"""
    sample = Path(__file__).resolve().parents[1] / "data/ph2/authored_wave1_composite_operation_facts_v3.jsonl.sample"
    rows = [json.loads(line) for line in sample.read_bytes().splitlines()]
    chosen = [next(row for row in rows if row["sample_role"] == role) for role in ("support", "conflict")]
    source = tmp_path / "data/ph2/facts.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8") + b"\n" for row in chosen))
    facts = compile_independent_relation_fact_source(source, tmp_path / "teacher")
    course_path = tmp_path / "course.int.json"
    write_integer_relation_graph_course(course_path, facts.course)
    backend = SQLiteBackend(str(tmp_path / "core.sqlite3"))
    try:
        context = make_train_context(backend)
        _, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(context, occurrence_protocol=occurrence, span_protocol=span)
        learning = build_integer_w06_learning_runtime(backend, context, (course_path,), require_full_curriculum=False)
        result = verify_relation_training_integrity(learning, facts.course)
        assert (result["candidate_count"], result["active_candidate_count"], result["conflict_candidate_count"]) == (2, 1, 1)
        assert result["active_proposition_keys"] == [list(record.proposition_key)
            for record in facts.course.records if record.sample_role_code == 1]
        with pytest.raises(ValueError, match="complete licensed course"):
            verify_relation_training_integrity(learning, replace(facts.course, records=facts.course.records[:1]))
        with pytest.raises(ValueError, match="lifecycle or projection"):
            verify_relation_training_integrity(learning, replace(facts.course, records=tuple(
                replace(record, sample_role_code=1) for record in facts.course.records)))
    finally:
        backend.close()
