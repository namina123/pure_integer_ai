"""D-02 LC-03 多词表达与构式身份课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_construction_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredConstructionCourseError,
    build_construction_course_manifest,
    compile_authored_construction_course,
    read_authored_construction_seeds,
    validate_construction_payload,
)
from pure_integer_ai.experiments.ph2_capability_course_contract import (
    CapabilityCourseContractError,
    read_capability_course_manifest,
    write_capability_course_manifest,
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
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


SAMPLE_PATH = Path("data/ph2/authored_construction_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "516153e354010461bb08df0d3ad0879a70c4a19a08a2d2018a84d34ef8041a6a")


def _sample_values() -> list[dict]:
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _records_by_kind(build, kind: str):
    return tuple(
        record
        for identity in build.manifest.files
        if identity.record_kind == kind
        for record in read_record_artifact(build.pack_root, identity)
    )


def test_sample_freezes_both_owners_and_complete_construction_taxonomy():
    """双 owner 均有七类样本、十类候选和一条 anti-literal 负基线。"""
    seeds = read_authored_construction_seeds(SAMPLE_PATH)
    assert len(seeds) == 24
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 12
    expected_kinds = {
        "AMBIGUOUS", "ANTI_LITERAL", "COMPOSITIONAL", "DISCONTINUOUS",
        "GENERATION", "PARTIAL_LEXICALIZED", "RETENTION", "REVISION",
        "UNKNOWN", "WHOLE_LEXICALIZED",
    }
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.candidate_kind for item in owner} == expected_kinds
        literal = tuple(item for item in owner
                        if item.baseline_kind == "LITERAL_TOKEN_SUM_ONLY")
        assert len(literal) == 1
        assert literal[0].expected_state == "FALSE"
        assert validate_construction_payload(
            literal[0].observation_payload()).literal_only == 1
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {item.evaluation_dimension for item in evaluator} == set(
        EVALUATOR_DIMENSIONS)


def test_same_surface_same_proposition_and_held_out_recombination_are_direct():
    """保留同表层异构式、同命题不同构式和两个 filler×construction 新配对。"""
    seeds = read_authored_construction_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for prefix, owner in (("teacher", teacher), ("evaluator", evaluator)):
        ambiguous = tuple(
            item for item in owner
            if item.candidate_group == f"{prefix}-ambiguous-surface-v1")
        assert len(ambiguous) == 2
        assert len({item.observed_text for item in ambiguous}) == 1
        assert len({item.construction_identity.to_value()["construction_key"]
                    for item in ambiguous}) == 2
        by_proposition: dict[str, set[str]] = {}
        for item in owner:
            by_proposition.setdefault(item.proposition_group, set()).add(
                item.construction_family)
        assert any(len(families) >= 2 for families in by_proposition.values())

    teacher_fillers = {item.filler_family for item in teacher}
    teacher_constructions = {item.construction_family for item in teacher}
    teacher_pairs = {
        (item.filler_family, item.construction_family) for item in teacher
    }
    recombinations = {
        (item.filler_family, item.construction_family) for item in evaluator
        if item.filler_family in teacher_fillers
        and item.construction_family in teacher_constructions
        and (item.filler_family, item.construction_family) not in teacher_pairs
    }
    assert ("FILLER_COLOR", "CONSTRUCTION_BA") in recombinations
    assert ("FILLER_DOOR", "CONSTRUCTION_COMPOSITIONAL") in recombinations


def test_spans_slots_event_core_generation_and_revision_are_typed():
    """非连续 span、固定/可变 slot、event core、隐藏生成和本 split 修订均可直验。"""
    seeds = read_authored_construction_seeds(SAMPLE_PATH)
    discontinuous = tuple(
        item for item in seeds if item.candidate_kind == "DISCONTINUOUS")
    assert len(discontinuous) == 2
    assert all(validate_construction_payload(
        item.observation_payload()).discontinuous_span_count == 1
               for item in discontinuous)

    for item in seeds:
        payload = item.observation_payload().to_value()
        assert payload["selection_state"] == "UNSELECTED"
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        if item.candidate_kind == "GENERATION":
            assert payload["observed_surface"]["target_hidden"] == 1
            assert payload["generation_constraint"]["output_surface_hidden"] == 1
            for surface in item.expected_payload.to_value()["accepted_surfaces"]:
                assert surface not in payload["observed_surface"]["text"]
        if item.candidate_kind == "REVISION":
            target = next(seed for seed in seeds
                          if seed.seed_id == item.supersedes_seed_id)
            assert target.label_owner == item.label_owner
            assert target.split == item.split
            assert target.logical_order < item.logical_order
            slots = payload["slots"]
            assert {slot["fixed"] for slot in slots} == {0, 1}


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    """两次编译 bit-identical，Observation 与私有 label 物理分账。"""
    first = compile_authored_construction_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_construction_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 72
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-03",)
    assert first.validation.source_ref_count == 24
    assert first.validation.observation_count == 24
    assert first.validation.teacher_evidence_count == 12
    assert first.validation.evaluator_label_count == 12
    assert first.validation.source_cluster_count == 2
    assert read_artifact_manifest(first.pack_root / "manifest.json") == first.manifest
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
        validate_construction_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    """控制 manifest 绑定课程、组合轴、消融、pack 和全部零执行事实。"""
    build = compile_authored_construction_course(SAMPLE_PATH, tmp_path / "pack")
    manifest = build_construction_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-03",)
    assert manifest.capability_keys == ("MULTIWORD_CONSTRUCTION",)
    assert manifest.pack_record_count == 72
    assert manifest.pack_manifest_sha256 == build.manifest.sha256()
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
    path = tmp_path / "course.json"
    write_capability_course_manifest(manifest, path)
    assert read_capability_course_manifest(path) == manifest
    write_capability_course_manifest(manifest, path)
    path.write_bytes(b'{"damaged":1}\n')
    with pytest.raises(CapabilityCourseContractError, match="内容不同"):
        write_capability_course_manifest(manifest, path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"), "CC0-1.0"),
        (lambda rows: rows[12].__setitem__("split", "train"), "split"),
        (lambda rows: rows.pop(8), "构式候选族|同表层"),
        (lambda rows: rows[9].__setitem__(
            "supersedes_seed_id", rows[10]["seed_id"]), "更早"),
        (lambda rows: rows[10].__setitem__(
            "retention_anchor_id", rows[11]["seed_id"]), "更早"),
        (lambda rows: rows[12].__setitem__(
            "evaluation_dimension", rows[13]["evaluation_dimension"]),
         "evaluator 维度"),
    ],
)
def test_bad_license_owner_lattice_future_links_and_dimension_fail_closed(
        tmp_path, mutate, message):
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredConstructionCourseError, match=message):
        read_authored_construction_seeds(path)


def test_missing_held_out_recombination_fails_closed(tmp_path):
    """两条 evaluator 新配对一旦退化成 teacher 已见配对就拒绝。"""
    rows = _sample_values()
    rows[12]["filler_family"] = rows[0]["filler_family"]
    rows[12]["construction_family"] = rows[0]["construction_family"]
    rows[13]["filler_family"] = rows[1]["filler_family"]
    rows[13]["construction_family"] = rows[1]["construction_family"]
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredConstructionCourseError, match="held-out"):
        read_authored_construction_seeds(path)


def test_payload_hash_span_slot_event_generation_and_baseline_damage_fail_closed():
    """坏 hash/span/slot/event、生成泄漏和伪构式对象均不得通过。"""
    seeds = read_authored_construction_seeds(SAMPLE_PATH)
    positive = seeds[0]
    value = positive.observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredConstructionCourseError, match="SHA-256"):
        validate_construction_payload(value)

    value = positive.observation_payload().to_value()
    value["spans"][0]["members"][0][1] = 999
    with pytest.raises(AuthoredConstructionCourseError, match="越界"):
        validate_construction_payload(value)

    value = positive.observation_payload().to_value()
    value["slots"][0]["filler_span_id"] = "missing"
    with pytest.raises(AuthoredConstructionCourseError, match="未知 filler"):
        validate_construction_payload(value)

    value = positive.observation_payload().to_value()
    value["event_core_mapping"]["role_bindings"][0]["slot_id"] = "missing"
    with pytest.raises(AuthoredConstructionCourseError, match="未知或重复 slot"):
        validate_construction_payload(value)

    generation = next(item for item in seeds if item.candidate_kind == "GENERATION")
    value = generation.observation_payload().to_value()
    value["generation_constraint"]["output_surface_hidden"] = 0
    with pytest.raises(AuthoredConstructionCourseError, match="hidden"):
        validate_construction_payload(value)

    literal = next(item for item in seeds
                   if item.baseline_kind == "LITERAL_TOKEN_SUM_ONLY")
    value = literal.observation_payload().to_value()
    value["construction_identity"]["present"] = 1
    with pytest.raises(AuthoredConstructionCourseError, match="缺身份"):
        validate_construction_payload(value)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    """非规范 JSON、float 和覆盖既有 pack 均失败。"""
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredConstructionCourseError, match="损坏"):
        read_authored_construction_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredConstructionCourseError, match="非规范 JSON|损坏"):
        read_authored_construction_seeds(path)
    compile_authored_construction_course(SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_construction_course(SAMPLE_PATH, tmp_path / "release")


def test_formal_repository_course_manifest_and_external_pack_are_exact():
    """正式 LC-03 manifest、sample 和外部 pack 可逐 hash/record 回读。"""
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    manifest = read_capability_course_manifest(repository / COURSE_MANIFEST_PATH)
    assert manifest.sha256() == FORMAL_COURSE_MANIFEST_SHA256
    sample = repository / Path(*manifest.sample_relative_path.split("/"))
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == manifest.sample_sha256
    pack_manifest = workspace / Path(*manifest.pack_manifest_relative_path.split("/"))
    assert manifest.pack_manifest_relative_path.startswith(
        FORMAL_ARTIFACT_RELATIVE_ROOT + "/packs/")
    assert hashlib.sha256(pack_manifest.read_bytes()).hexdigest() == (
        manifest.pack_manifest_sha256)
    restored = read_artifact_manifest(pack_manifest)
    assert restored.sha256() == manifest.pack_manifest_sha256
    assert restored.record_count == manifest.pack_record_count == 72
    assert restored.splits == manifest.pack_splits == ("train", "held_out")
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
