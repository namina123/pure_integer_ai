"""纯整数 relation graph-course 的冷恢复和 Core 接线回归。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_integer_w06_adapter,
    build_integer_w06_learning_runtime,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    IntegerRelationGraphCourseError,
    integer_course_from_w06_adapter,
    integer_relation_graph_course_to_w06_adapter,
    read_integer_relation_graph_course,
    write_integer_relation_graph_course,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    adapt_w06_training_payload,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
)
import pure_integer_ai.experiments.run_conversation_training as training_runner
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    GRAPH_RELATION_GENERATION_UNAVAILABLE,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "ph2_w06_dataset_artifacts" / "source_semantic_overlay_v1" / "packs"


def _authored_adapter():
    source = []
    observations = []
    teachers = []
    for manifest_path in sorted(PACK_ROOT.glob(
            "AUTHORED_CC0_V1--CC0-1.0--*/manifest.json")):
        manifest = read_artifact_manifest(manifest_path)
        for identity in manifest.files:
            records = read_record_artifact(manifest_path.parent, identity)
            if identity.record_kind == RECORD_SOURCE_REF:
                source.extend(records)
            elif identity.record_kind == RECORD_OBSERVATION:
                observations.extend(
                    item for item in records if item.split == "train")
            elif identity.record_kind == RECORD_TEACHER_EVIDENCE:
                teachers.extend(records)
    return adapt_w06_training_payload(W06TrainingPayload(
        tuple(source), tuple(observations), tuple(teachers)))


def _write_course(tmp_path: Path) -> tuple[Path, object]:
    adapter = _authored_adapter()
    course = integer_course_from_w06_adapter(
        adapter,
        source_identity=(91660, 999, 20261008, 1),
        license_identity=(91661, 999, 1),
    )
    path = tmp_path / "relation-course.int.json"
    write_integer_relation_graph_course(path, course)
    return path, adapter


def test_integer_course_round_trip_preserves_w06_payload_and_rejection(tmp_path):
    path, authored = _write_course(tmp_path)
    restored_course = read_integer_relation_graph_course(path)
    restored = integer_relation_graph_course_to_w06_adapter(restored_course)

    assert len(restored_course.records) == len(authored.observations) + len(
        authored.rejections)
    assert len(restored.candidates) == len(authored.candidates)
    assert len(restored.rejections) == len(authored.rejections)
    old_payload = {
        item.observation.stable_key: item.observation.typed_payload.to_value()
        for item in authored.observations
    }
    new_payload = {
        item.observation.stable_key: item.observation.typed_payload.to_value()
        for item in restored.observations
    }
    assert old_payload == new_payload
    old_evidence = {
        item.observation.stable_key: item.teacher_record.typed_evidence.to_value()
        for item in authored.evidence
    }
    new_evidence = {
        item.observation.stable_key: item.teacher_record.typed_evidence.to_value()
        for item in restored.evidence
    }
    assert old_evidence == new_evidence
    assert {
        item.observation.stable_key for item in restored.rejection_evidence
    } == {item.observation.stable_key for item in authored.rejection_evidence}
    serialized = path.read_text(encoding="ascii")
    assert "raw_text" not in serialized
    assert '"surface":' not in serialized
    assert '"answer"' not in serialized


def test_integer_course_rejects_nested_text_and_noncanonical_json(tmp_path):
    path, _ = _write_course(tmp_path)
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["records"][0]["consumer_request"]["leak"] = "source prose"
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(IntegerRelationGraphCourseError):
        read_integer_relation_graph_course(path)


def test_integer_course_consumes_shared_sqlite_core_and_readonly_runtime(tmp_path):
    course_path, _ = _write_course(tmp_path)
    database = tmp_path / "training.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(
            context, occurrence_protocol=occurrence, span_protocol=span)
        runtime = build_integer_w06_learning_runtime(
            backend, context, (course_path,))
        backend.commit()
        report = runtime.report()
        assert report.candidate_count == 40
        assert report.schema_rejection_count == 1
        assert report.active_candidate_count > 0
        assert backend.count("graph_statement") > 0
    finally:
        backend.close()

    with TrainedRelationGraphRuntime(database) as restored:
        assert restored.snapshot.candidate_identities
        fact = restored.active_surface_facts()[0]
        query = " / ".join((fact.cue, *(item.surface for item in fact.bindings)))
        result = restored.query(query)
        assert result.result_code == GRAPH_RELATION_GENERATION_UNAVAILABLE
        assert result.answer is None


def test_integer_bridge_reads_only_integer_course(tmp_path):
    course_path, _ = _write_course(tmp_path)
    adapter = build_integer_w06_adapter((course_path,))
    assert len(adapter.candidates) == 40
    assert len(adapter.rejections) == 1


def test_integer_course_cli_is_a_separate_runner_input(monkeypatch, tmp_path):
    """CLI 的整数课程参数不能落入普通 dialogue pack 参数。"""
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"ok": 1}

    monkeypatch.setattr(training_runner, "run_conversation_training", fake_runner)
    course_path = tmp_path / "course.int.json"
    course_path.write_text("placeholder", encoding="ascii")
    assert training_runner.main([
        "--run-root", r"K:\\pure_integer_ai_work\\test-run",
        "--integer-relation-course", str(course_path),
        "--no-default-courses",
    ]) == 0
    assert captured["integer_relation_course_paths"] == (str(course_path),)
    assert captured["extra_course_paths"] == ()
    assert captured["include_default_courses"] is False
