"""D-02 LC-02 生产性形态与构词课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_morphology_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredMorphologyCourseError,
    build_morphology_course_manifest,
    compile_authored_morphology_course,
    read_authored_morphology_seeds,
    validate_morphology_payload,
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


SAMPLE_PATH = Path("data/ph2/authored_morphology_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "ecd55e43951772e707b4f68ae9e58d9a183e37572d38aa19edfd79a12799eb3e")


def _sample_values() -> list[dict]:
    """读取仓库 sample 为可独立破坏的 JSON object。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """用统一规范 JSONL 写测试负例。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _records_by_kind(build, kind: str):
    """读取 pack 中某一 record kind 的全部物理文件。"""
    return tuple(
        record
        for identity in build.manifest.files
        if identity.record_kind == kind
        for record in read_record_artifact(build.pack_root, identity)
    )


def test_sample_freezes_both_owners_seven_families_and_full_morphology_taxonomy():
    """双 owner 均覆盖七类、九候选类和六种承重形态关系。"""
    seeds = read_authored_morphology_seeds(SAMPLE_PATH)
    assert len(seeds) == 22
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 11
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.candidate_kind for item in owner} == {
            "AFFIXATION", "COMPOUND", "DICTIONARY_REPLAY", "EXCEPTION",
            "GENERATION", "REDUPLICATION", "RETENTION", "SEGMENTATION",
            "UNKNOWN",
        }
        assert {
            relation
            for item in owner
            for relation in validate_morphology_payload(
                item.observation_payload()).relation_kinds
        } == {
            "ATTACHES_AFFIX", "COMPOUND_COMPONENT", "EXCEPTION_TO",
            "FILLS_SLOT", "HAS_STEM", "REDUPLICATES",
        }
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})


def test_held_out_stem_construction_recombination_and_ambiguous_lattices_are_direct():
    """evaluator 至少有两个轴已见但配对未见的组合，并保留双候选切分。"""
    seeds = read_authored_morphology_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    teacher_stems = {item.stem_family for item in teacher}
    teacher_constructions = {item.construction_family for item in teacher}
    teacher_pairs = {
        (item.stem_family, item.construction_family) for item in teacher
    }
    recombinations = {
        (item.stem_family, item.construction_family) for item in evaluator
        if (item.stem_family in teacher_stems
            and item.construction_family in teacher_constructions
            and (item.stem_family, item.construction_family) not in teacher_pairs)
    }
    assert len(recombinations) >= 2
    assert ("STEM_NOUN", "SUFFIX_HUA") in recombinations
    assert ("STEM_PROPERTY", "COMPOUND_MODIFIER_HEAD") in recombinations
    for prefix in ("teacher", "evaluator"):
        candidates = tuple(
            item for item in seeds
            if item.candidate_group == f"{prefix}-segmentation-lattice-v1")
        assert len(candidates) == 2
        assert len({tuple(
            (unit.to_value()["unit_kind"], unit.to_value()["surface"])
            for unit in item.analysis_units) for item in candidates}) == 2
        assert {item.observation_payload().to_value()["selection_state"]
                for item in candidates} == {"UNSELECTED"}


def test_reverse_generation_hides_target_and_dictionary_replay_is_negative():
    """反向生成答案只在私有 label，词典整词回放不能通过生产性形态维。"""
    seeds = read_authored_morphology_seeds(SAMPLE_PATH)
    generations = tuple(item for item in seeds if item.candidate_kind == "GENERATION")
    assert len(generations) == 2
    for item in generations:
        payload = item.observation_payload().to_value()
        assert payload["observed_surface"]["target_hidden"] == 1
        assert payload["generation_constraint"]["output_surface_hidden"] == 1
        assert item.expected_payload.to_value()["accepted_surfaces"]
        for surface in item.expected_payload.to_value()["accepted_surfaces"]:
            assert surface not in payload["observed_surface"]["text"]
    replay = tuple(
        item for item in seeds if item.baseline_kind == "DICTIONARY_REPLAY_ONLY")
    assert len(replay) == 2
    assert {item.expected_state for item in replay} == {"FALSE"}
    assert all(not item.relations for item in replay)
    assert all(validate_morphology_payload(
        item.observation_payload()).dictionary_replay_only == 1 for item in replay)


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    """两次独立编译产生同一四 owner pack，Observation 不含 expected。"""
    first = compile_authored_morphology_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_morphology_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 66
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-02",)
    assert first.validation.source_ref_count == 22
    assert first.validation.observation_count == 22
    assert first.validation.teacher_evidence_count == 11
    assert first.validation.evaluator_label_count == 11
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
        validate_morphology_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert {item.expected_payload.to_value()["analysis_key"]
            for item in evaluators}
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    """控制 manifest 绑定组合轴、消融、pack 和全部零执行事实。"""
    build = compile_authored_morphology_course(SAMPLE_PATH, tmp_path / "pack")
    manifest = build_morphology_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-02",)
    assert manifest.capability_keys == ("MORPHOLOGY_WORD_FORM",)
    assert manifest.pack_record_count == 66
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
        (lambda rows: rows[11].__setitem__("split", "train"), "split"),
        (lambda rows: rows.pop(16), "歧义切分"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[9]["seed_id"]), "更早"),
        (lambda rows: rows.pop(), "evaluator"),
    ],
)
def test_bad_license_owner_lattice_future_revision_and_missing_family_fail_closed(
        tmp_path, mutate, message):
    """坏许可、owner 串线、坏 lattice、未来 revision 和漏族均失败。"""
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredMorphologyCourseError, match=message):
        read_authored_morphology_seeds(path)


def test_payload_hash_slot_generation_and_dictionary_damage_fail_closed():
    """坏 hash、漏 slot、生成泄漏和词典伪关系均不得通过。"""
    seeds = read_authored_morphology_seeds(SAMPLE_PATH)
    value = seeds[0].observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredMorphologyCourseError, match="SHA-256"):
        validate_morphology_payload(value)

    value = seeds[0].observation_payload().to_value()
    value["morphology_relations"] = [
        item for item in value["morphology_relations"]
        if item["relation_kind"] != "FILLS_SLOT"]
    value["morphology_relations"][1]["order_index"] = 2
    with pytest.raises(AuthoredMorphologyCourseError, match="stem/slot"):
        validate_morphology_payload(value)

    generation = next(item for item in seeds if item.candidate_kind == "GENERATION")
    value = generation.observation_payload().to_value()
    value["generation_constraint"]["output_surface_hidden"] = 0
    with pytest.raises(AuthoredMorphologyCourseError, match="hidden"):
        validate_morphology_payload(value)

    replay = next(item for item in seeds
                  if item.baseline_kind == "DICTIONARY_REPLAY_ONLY")
    value = replay.observation_payload().to_value()
    value["morphology_relations"] = [{
        "order_index": 1,
        "relation_kind": "HAS_STEM",
        "slot_key": "SURFACE_FORM",
        "source_unit_id": "c",
        "target_unit_id": "c",
    }]
    with pytest.raises(AuthoredMorphologyCourseError, match="词典回放"):
        validate_morphology_payload(value)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    """非规范 JSON、float 和覆盖既有 pack 均失败。"""
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredMorphologyCourseError, match="损坏"):
        read_authored_morphology_seeds(path)
    compile_authored_morphology_course(SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_morphology_course(SAMPLE_PATH, tmp_path / "release")


def test_formal_repository_course_manifest_and_external_pack_are_exact():
    """正式 LC-02 manifest、sample 和外部 pack 可逐 hash/record 回读。"""
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    manifest = read_capability_course_manifest(repository / COURSE_MANIFEST_PATH)
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
    assert restored.record_count == manifest.pack_record_count == 66
    assert restored.splits == manifest.pack_splits == ("train", "held_out")
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
