"""D-02B 首类 AUTHORED_CC0_V1 sense/概念边界编译器 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_sense_course import (
    LICENSE_ID,
    PACK_NAME,
    SOURCE_KEY,
    AuthoredSenseCourseError,
    compile_authored_sense_course,
    read_authored_sense_seeds,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    validate_stage_visibility,
)


SAMPLE_PATH = Path("data/ph2/authored_sense_seed_v1.jsonl.sample")


def _sample_values() -> list[dict]:
    """读取仓库极小 sample 为独立可修改 JSON object 列表。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """按统一规范 JSONL 写测试 seed，不复用编译器 parser。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _records_by_kind(build, kind: str):
    """读取 manifest 中指定 record kind 的全部物理文件。"""
    records = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            records.extend(read_record_artifact(build.pack_root, identity))
    return tuple(records)


def test_sample_license_is_explicit_cc0_and_seed_families_are_independent():
    """仓库 sample 有独立数据许可，teacher/evaluator family 与模板互斥。"""
    license_text = Path("data/ph2/DATA_LICENSE.md").read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: CC0-1.0" in license_text
    seeds = read_authored_sense_seeds(SAMPLE_PATH)
    assert len(seeds) == 6
    assert {seed.sample_role for seed in seeds} == {
        "support", "refute", "conflict", "supersede"}
    teacher_families = {seed.family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_families = {seed.family for seed in seeds if seed.label_owner == "evaluator"}
    teacher_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "teacher"
    }
    evaluator_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "evaluator"
    }
    assert teacher_families.isdisjoint(evaluator_families)
    assert teacher_templates.isdisjoint(evaluator_templates)
    assert LICENSE_ID == "CC0-1.0"


def test_compiler_writes_deterministic_pack_with_private_owner_separation(tmp_path):
    """两次编译产生相同 manifest/transport，Observation 不含私有 expected。"""
    first = compile_authored_sense_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_sense_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-03",)
    assert first.validation.source_ref_count == 6
    assert first.validation.observation_count == 6
    assert first.validation.teacher_evidence_count == 4
    assert first.validation.evaluator_label_count == 2
    assert first.validation.source_cluster_count == 2
    assert read_artifact_manifest(first.pack_root / "manifest.json") == first.manifest
    assert (first.pack_root / "manifest.json").read_bytes() == (
        second.pack_root / "manifest.json").read_bytes()
    for first_file, second_file in zip(first.manifest.files, second.manifest.files):
        assert first_file == second_file
        assert (first.pack_root / first_file.relative_path).read_bytes() == (
            second.pack_root / second_file.relative_path).read_bytes()

    sources = _records_by_kind(first, RECORD_SOURCE_REF)
    observations = _records_by_kind(first, RECORD_OBSERVATION)
    teachers = _records_by_kind(first, RECORD_TEACHER_EVIDENCE)
    evaluators = _records_by_kind(first, RECORD_EVALUATOR_LABEL)
    assert all(isinstance(item, SourceRefRecord) for item in sources)
    assert all(isinstance(item, ObservationRecord) for item in observations)
    assert all(isinstance(item, TeacherEvidenceRecord) for item in teachers)
    assert all(isinstance(item, EvaluatorLabelRecord) for item in evaluators)
    assert len(sources) == 6
    assert len(observations) == 6
    assert len(teachers) == 4
    assert len(evaluators) == 2
    for item in observations:
        payload = item.typed_payload.to_value()
        assert set(payload) == {
            "candidate_sense", "context", "query_kind", "surface"}
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
    assert all("expected_state" in item.typed_evidence.to_value()
               for item in teachers)
    assert all(item.expected_state in {"TRUE", "FALSE"}
               for item in evaluators)


def test_split_clusters_stage_views_and_supersede_target_are_directly_auditable(
        tmp_path):
    """train/held-out 来源簇互斥，owner 视图隔离，supersede 指向更早记录。"""
    build = compile_authored_sense_course(SAMPLE_PATH, tmp_path)
    sources = _records_by_kind(build, RECORD_SOURCE_REF)
    observations = _records_by_kind(build, RECORD_OBSERVATION)
    teachers = _records_by_kind(build, RECORD_TEACHER_EVIDENCE)
    evaluators = _records_by_kind(build, RECORD_EVALUATOR_LABEL)
    source_index = {item.stable_key: item for item in sources}
    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(item for item in observations if item.split == "held_out")
    train_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in train
    }
    held_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in held_out
    }
    assert train_clusters.isdisjoint(held_clusters)
    validate_stage_visibility(
        train, teachers, (), current_stage="W-03", view_kind="training")
    validate_stage_visibility(
        held_out, (), evaluators, current_stage="W-03", view_kind="evaluation")
    superseders = [item for item in observations if item.sample_role == "supersede"]
    assert len(superseders) == 1
    superseder = superseders[0]
    target = next(
        item for item in observations if item.stable_key == superseder.supersedes_key)
    assert target.logical_order < superseder.logical_order
    assert target.split == superseder.split == "train"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"), "CC0-1.0"),
        (lambda rows: rows[1].__setitem__("seed_id", rows[0]["seed_id"]), "重复"),
        (lambda rows: rows[4].__setitem__("split", "train"), "split"),
        (lambda rows: (
            rows[0].__setitem__("sample_role", "supersede"),
            rows[0].__setitem__("supersedes_seed_id", rows[1]["seed_id"]),
        ), "更早"),
    ],
)
def test_bad_license_duplicate_seed_owner_split_and_future_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、重复 seed、owner/split 串线和未来替代均不能进入编译。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredSenseCourseError, match=message):
        read_authored_sense_seeds(bad)


def test_noncanonical_or_float_seed_and_existing_pack_fail_closed(tmp_path):
    """非规范 JSON、浮点和同版本覆盖均被拒绝。"""
    bad = tmp_path / "bad.sample"
    bad.write_bytes(b'{"candidate_sense":1.5}\n')
    with pytest.raises(AuthoredSenseCourseError, match="规范 JSON"):
        read_authored_sense_seeds(bad)

    build = compile_authored_sense_course(SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredSenseCourseError, match="已存在"):
        compile_authored_sense_course(SAMPLE_PATH, tmp_path / "release")


def test_manifest_transport_mutation_remains_detectable(tmp_path):
    """首类 compiler 产生的每个 transport 仍受 D-02A 双 hash reader 保护。"""
    build = compile_authored_sense_course(SAMPLE_PATH, tmp_path)
    identity = next(
        item for item in build.manifest.files
        if item.record_kind == RECORD_OBSERVATION and item.split == "held_out")
    path = build.pack_root / identity.relative_path
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(bytes(payload))
    with pytest.raises(DatasetArtifactIOError, match="transport SHA-256"):
        read_record_artifact(build.pack_root, identity)
