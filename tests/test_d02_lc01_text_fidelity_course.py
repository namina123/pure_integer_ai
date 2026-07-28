"""D-02 LC-01 文本观察保真与 LC-15 初版目标课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_text_fidelity_course import (
    COURSE_MANIFEST_PATH,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredTextFidelityCourseError,
    build_text_fidelity_course_manifest,
    compile_authored_text_fidelity_course,
    initial_learning_objectives,
    read_authored_text_fidelity_seeds,
    validate_text_fidelity_payload,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_language_course_contract import (
    LANGUAGE_OBJECTIVE_KEYS,
    TEXT_FIDELITY_EVALUATOR_DIMENSIONS,
    LanguageCourseContractError,
    read_language_course_manifest,
    write_language_course_manifest,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


SAMPLE_PATH = Path("data/ph2/authored_text_fidelity_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "5ab2968031a345302c1bf91081e0028e6198d777fd9c6199b695ad2dbdc5d116")


def _sample_values() -> list[dict]:
    """读取仓库 sample 为独立可修改 JSON object 列表。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """用统一规范 JSONL 写负例，不复用课程 parser。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _records_by_kind(build, kind: str):
    """读取 pack 中一种 record kind 的全部物理文件。"""
    return tuple(
        record
        for identity in build.manifest.files
        if identity.record_kind == kind
        for record in read_record_artifact(build.pack_root, identity)
    )


def test_sample_freezes_seven_families_both_owners_and_all_initial_objectives():
    """teacher/evaluator 均覆盖七类样本，目标分型无省略且族/template 互斥。"""
    seeds = read_authored_text_fidelity_seeds(SAMPLE_PATH)
    assert len(seeds) == 16
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 8
    assert {item.sample_family for item in teacher} == set(SAMPLE_FAMILIES)
    assert {item.sample_family for item in evaluator} == set(SAMPLE_FAMILIES)
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {key for item in seeds for key in item.objective_keys} == set(
        LANGUAGE_OBJECTIVE_KEYS)
    assert tuple(item.objective_key for item in initial_learning_objectives()) == (
        LANGUAGE_OBJECTIVE_KEYS)
    assert all(item.runtime_pass_authority == 0
               for item in initial_learning_objectives())


def test_payload_preserves_raw_receipt_lattice_loss_and_integer_lengths():
    """raw 不变、候选不私选、receipt/hash/损失和整数长度均可重算。"""
    seeds = read_authored_text_fidelity_seeds(SAMPLE_PATH)
    audits = [validate_text_fidelity_payload(item.observation_payload())
              for item in seeds]
    assert all(len(item.raw_sha256) == 64 for item in audits)
    assert all(len(item.derived_sha256) == 64 for item in audits)
    negative = next(item for item in seeds
                    if item.seed_id == "teacher-whitespace-negative-v1")
    payload = negative.observation_payload().to_value()
    assert payload["raw_observation"]["text"] == "甲  乙"
    assert payload["derived_candidate"]["text"] == "甲乙"
    assert payload["information_loss"] == 1
    assert payload["normalization_receipt"]["operations"][0][
        "reversible"] == 0
    assert payload["description_length"] == {
        "derived_unit_count": 2,
        "longer_by_units": 0,
        "raw_unit_count": 4,
        "shorter_by_units": 2,
        "unit_kind": "UNICODE_CODE_POINT",
    }
    lattice = [item.observation_payload().to_value() for item in seeds
               if item.candidate_group == "teacher-segmentation-lattice-v1"]
    assert len(lattice) == 2
    assert {item["selection_state"] for item in lattice} == {"UNSELECTED"}
    assert len({tuple(item["derived_candidate"]["segments"])
                for item in lattice}) == 2


def test_compiler_reuses_four_record_contract_and_is_transport_deterministic(
        tmp_path):
    """两次独立编译产生相同四 owner pack，Observation 不含 expected。"""
    first = compile_authored_text_fidelity_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_text_fidelity_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 48
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-02",)
    assert first.validation.source_ref_count == 16
    assert first.validation.observation_count == 16
    assert first.validation.teacher_evidence_count == 8
    assert first.validation.evaluator_label_count == 8
    assert first.validation.source_cluster_count == 2
    assert read_artifact_manifest(first.pack_root / "manifest.json") == (
        first.manifest)
    for identity in first.manifest.files:
        other = next(item for item in second.manifest.files
                     if item.relative_path == identity.relative_path)
        assert identity == other
        assert (first.pack_root / identity.relative_path).read_bytes() == (
            second.pack_root / other.relative_path).read_bytes()

    sources = _records_by_kind(first, RECORD_SOURCE_REF)
    observations = _records_by_kind(first, RECORD_OBSERVATION)
    teachers = _records_by_kind(first, RECORD_TEACHER_EVIDENCE)
    evaluators = _records_by_kind(first, RECORD_EVALUATOR_LABEL)
    assert all(isinstance(item, SourceRefRecord) for item in sources)
    assert all(isinstance(item, ObservationRecord) for item in observations)
    assert all(isinstance(item, TeacherEvidenceRecord) for item in teachers)
    assert all(isinstance(item, EvaluatorLabelRecord) for item in evaluators)
    for item in observations:
        assert item.payload_kind == PAYLOAD_KIND
        payload = item.typed_payload.to_value()
        validate_text_fidelity_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
    assert {item.expected_payload.to_value()["dimension"]
            for item in evaluators} == set(TEXT_FIDELITY_EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    """课程控制 manifest 绑定 pack/目标/evaluator/retention 且不可覆盖。"""
    build = compile_authored_text_fidelity_course(SAMPLE_PATH, tmp_path / "pack")
    manifest = build_text_fidelity_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.objective_taxonomy_status == "INITIAL_FROZEN"
    assert manifest.capability_exit_states.to_value() == {
        "RAW_TEXT_NOISE": "COURSE_FROZEN",
        "TYPED_LEARNING_OBJECTIVES": "PARTIAL_COURSE",
    }
    assert manifest.pack_record_count == 48
    assert manifest.pack_manifest_sha256 == build.manifest.sha256()
    assert all(value == 0 for value in (
        manifest.d03_published,
        manifest.w01_started,
        manifest.formal_training_runs,
        manifest.teacher_calls,
        manifest.learning_state_writes,
        manifest.mastered_claims,
        manifest.readiness_claims,
    ))
    path = tmp_path / "course.json"
    write_language_course_manifest(manifest, path)
    assert read_language_course_manifest(path) == manifest
    write_language_course_manifest(manifest, path)
    path.write_bytes(b'{"damaged":1}\n')
    with pytest.raises(LanguageCourseContractError, match="内容不同"):
        write_language_course_manifest(manifest, path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"), "CC0-1.0"),
        (lambda rows: rows[8].__setitem__("split", "train"), "split"),
        (lambda rows: rows[3].__setitem__(
            "candidate_group", "isolated-candidate"), "lattice"),
        (lambda rows: rows[5].__setitem__(
            "supersedes_seed_id", rows[7]["seed_id"]), "更早"),
        (lambda rows: rows.pop(), "七类"),
    ],
)
def test_bad_license_owner_lattice_future_revision_and_missing_family_fail_closed(
        tmp_path, mutate, message):
    """坏许可、owner 串线、私选 lattice、未来 revision 和漏 family 均失败。"""
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredTextFidelityCourseError, match=message):
        read_authored_text_fidelity_seeds(path)


def test_payload_damage_and_noncanonical_or_float_seed_fail_closed(tmp_path):
    """raw hash、私选、静默损失、非规范 JSON 和 float 都不能通过。"""
    seed = read_authored_text_fidelity_seeds(SAMPLE_PATH)[0]
    value = seed.observation_payload().to_value()
    value["raw_observation"]["text"] = "damaged"
    with pytest.raises(AuthoredTextFidelityCourseError, match="raw Observation"):
        validate_text_fidelity_payload(value)
    value = seed.observation_payload().to_value()
    value["selection_state"] = "SELECTED"
    with pytest.raises(AuthoredTextFidelityCourseError, match="私选"):
        validate_text_fidelity_payload(value)
    value = seed.observation_payload().to_value()
    value["information_loss"] = 1
    with pytest.raises(AuthoredTextFidelityCourseError, match="不可逆"):
        validate_text_fidelity_payload(value)

    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"candidate_id":1.5}\n')
    with pytest.raises(AuthoredTextFidelityCourseError, match="损坏"):
        read_authored_text_fidelity_seeds(path)


def test_formal_repository_course_manifest_and_external_pack_are_exact():
    """正式课程 manifest、sample 和外部 pack 可逐 hash/record 回读。"""
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    manifest = read_language_course_manifest(repository / COURSE_MANIFEST_PATH)
    assert manifest.sha256() == FORMAL_COURSE_MANIFEST_SHA256
    sample = repository / Path(*manifest.sample_relative_path.split("/"))
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == manifest.sample_sha256
    pack_manifest = workspace / Path(
        *manifest.pack_manifest_relative_path.split("/"))
    assert manifest.pack_manifest_relative_path.startswith(
        FORMAL_ARTIFACT_RELATIVE_ROOT + "/packs/")
    assert hashlib.sha256(pack_manifest.read_bytes()).hexdigest() == (
        manifest.pack_manifest_sha256)
    restored = read_artifact_manifest(pack_manifest)
    assert restored.sha256() == manifest.pack_manifest_sha256
    assert restored.record_count == manifest.pack_record_count == 48
    assert restored.splits == manifest.pack_splits == ("train", "held_out")
    assert (
        manifest.d03_published,
        manifest.w01_started,
        manifest.formal_training_runs,
        manifest.teacher_calls,
        manifest.learning_state_writes,
        manifest.mastered_claims,
        manifest.readiness_claims,
    ) == (0, 0, 0, 0, 0, 0, 0)
