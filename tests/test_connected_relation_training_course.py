"""独立教师课程的 owner 隔离、问式组合和完整来源冷读边界。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.connected_relation_training_course import (
    compile_relation_training_course,
)
from pure_integer_ai.experiments.integer_relation_connected_course import _SPEC, _seed
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseCommonError, publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import compile_relation_seed
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION, RECORD_SOURCE_REF, canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetValidationError, validate_artifact_manifest,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    materialize_relation_question_graph, read_relation_question_graph,
)


def _source(tmp_path, *, evaluation=False, identical_questions=False):
    """创建两个虚构教师来源，不读取正式或历史评测数据。"""
    directory = tmp_path / "data" / "ph2"
    directory.mkdir(parents=True)
    rows = [
        {"anchor": "两种称呼", "case_id": "a", "relation": "PURE_ALIAS",
         "sample_role": "support", "seed_id": "a", "split": "train",
         "surface": "晨亭与西亭是同一处设施的两种称呼。",
         "terms": [["晨亭", "concept", 101], ["西亭", "concept", 102]]},
        {"anchor": "两种称呼", "case_id": "b", "relation": "PURE_ALIAS",
         "sample_role": "support", "seed_id": "b",
         "split": "held_out" if evaluation else "train",
         "surface": "秋亭与北亭是同一处设施的两种称呼。",
         "terms": [["秋亭", "concept", 103], ["北亭", "concept", 104]]},
    ]
    cases = [
        {"case_id": "qa", "input": "请判断：晨亭与西亭是什么关系？",
         "operation": "ASSERT_RELATION", "seed_ids": ["a"], "split": "train"},
        {"case_id": "qb", "input": ("请判断：秋亭与北亭是什么关系？"
         if identical_questions else "依据记录：秋亭与北亭是什么关系，对吗？"),
         "operation": "ASSERT_RELATION", "seed_ids": ["b"], "split": "train"},
    ]
    source, questions = directory / "facts.jsonl", directory / "questions.jsonl"
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    questions.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in cases))
    return source, questions, rows


def test_train_only_source_has_no_evaluator_and_cold_question_provenance(tmp_path):
    source, questions, _ = _source(tmp_path)
    result = compile_relation_training_course(source, questions, tmp_path / "pack")
    assert result.pack.manifest.splits == ("train",)
    assert result.source_audit["teacher_count"] == 2
    assert result.source_audit["evaluator_count"] == 0
    assert result.source_audit["status"] == "PASS_SOURCE_PACK_ONLY"
    assert result.question_audit["contract_count"] == 1
    assert result.question_audit["production_authorized"] == 0
    assert len(result.train_course.records) == 2
    assert len(result.question_frames) == 2
    target = tmp_path / "questions.sqlite3"
    materialize_relation_question_graph(
        result.question_frames, target,
        source_identity=result.train_course.source_identity,
        license_identity=result.train_course.license_identity)
    restored, metadata = read_relation_question_graph(target)
    assert restored == result.question_frames
    assert metadata["source_identity"] == result.train_course.source_identity
    assert all(frame.source_case_key[0] == 92029 for frame in restored)
    assert not (result.pack.pack_root / "owners" / "evaluator").exists()
    records = {kind: [] for kind in (RECORD_SOURCE_REF, RECORD_OBSERVATION)}
    for identity in result.pack.manifest.files:
        if identity.record_kind in records:
            records[identity.record_kind].extend(read_record_artifact(result.pack.pack_root, identity))
    with pytest.raises(DatasetValidationError, match="四类"):
        validate_artifact_manifest(
            result.pack.manifest, tuple(records[RECORD_SOURCE_REF]), tuple(records[RECORD_OBSERVATION]))


@pytest.mark.parametrize("evaluation,identical", [(True, False), (False, True)])
def test_unqualified_owner_or_one_surface_fails_before_pack_write(tmp_path, evaluation, identical):
    source, questions, _ = _source(
        tmp_path, evaluation=evaluation, identical_questions=identical)
    with pytest.raises(ValueError, match="evaluator|two distinct"):
        compile_relation_training_course(source, questions, tmp_path / "pack")
    assert not (tmp_path / "pack").exists()


def test_common_publisher_preserves_dual_owner_default_and_rejects_mixed_train_only(tmp_path):
    source, _questions, rows = _source(tmp_path)
    seeds = tuple(compile_relation_seed(_seed(row, number), semantic_identity=True)
                  for number, row in enumerate(rows, 1))
    spec = replace(_SPEC, pack_name="default-owner-boundary")
    with pytest.raises(AuthoredCourseCommonError, match="双 owner"):
        publish_authored_course(seeds, source, tmp_path / "default", spec)
    mixed = (seeds[0], replace(seeds[1], split="held_out", label_owner="evaluator"))
    with pytest.raises(AuthoredCourseCommonError, match="evaluation owner"):
        publish_authored_course(
            mixed, source, tmp_path / "mixed", spec, training_only=True)
    assert not (tmp_path / "mixed").exists()
