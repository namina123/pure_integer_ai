"""D-02 LC-04 递归结构与联合 parse 课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_recursive_parse_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredRecursiveParseCourseError,
    build_recursive_parse_course_manifest,
    compile_authored_recursive_parse_course,
    read_authored_recursive_parse_seeds,
    validate_recursive_parse_payload,
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


SAMPLE_PATH = Path("data/ph2/authored_recursive_parse_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "d6cf629d1d4e7441743d501cbe6e48e863bac14367d2bc7fd6504f8977b05c09")


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


def test_sample_freezes_both_owners_and_complete_recursive_parse_taxonomy():
    """双 owner 均覆盖七类样本、十一候选类和单树负基线。"""
    seeds = read_authored_recursive_parse_seeds(SAMPLE_PATH)
    assert len(seeds) == 24
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 12
    expected_kinds = {
        "AMBIGUOUS", "COORDINATED", "DISCONTINUOUS_DEPENDENCY",
        "GENERATION", "NESTED", "OPTIONAL", "PRESELECTED_TREE", "REPEATED",
        "RETENTION", "REVISION", "UNKNOWN",
    }
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.candidate_kind for item in owner} == expected_kinds
        baseline = tuple(item for item in owner
                         if item.baseline_kind == "PRESELECTED_TREE_ONLY")
        assert len(baseline) == 1
        assert baseline[0].expected_state == "FALSE"
        assert validate_recursive_parse_payload(
            baseline[0].observation_payload()).single_tree_only == 1
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {item.evaluation_dimension for item in evaluator} == set(
        EVALUATOR_DIMENSIONS)


def test_joint_parse_competition_and_held_out_depth_recombination_are_direct():
    """歧义保留两棵未选树，evaluator 有两个 filler×depth 新配对。"""
    seeds = read_authored_recursive_parse_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner in (teacher, evaluator):
        ambiguous = next(item for item in owner
                         if item.candidate_kind == "AMBIGUOUS")
        payload = ambiguous.observation_payload().to_value()
        assert len(payload["parse_candidates"]) == 2
        assert payload["selection_state"] == "UNSELECTED"
        assert len({item["parse_key"] for item in payload["parse_candidates"]}) == 2

    teacher_fillers = {item.filler_family for item in teacher}
    teacher_depths = {item.depth_family for item in teacher}
    teacher_pairs = {(item.filler_family, item.depth_family) for item in teacher}
    recombinations = {
        (item.filler_family, item.depth_family) for item in evaluator
        if item.filler_family in teacher_fillers
        and item.depth_family in teacher_depths
        and (item.filler_family, item.depth_family) not in teacher_pairs
    }
    assert ("FILLER_ENTITY", "DEPTH_REPEATED") in recombinations
    assert ("FILLER_ACTION", "DEPTH_SHALLOW") in recombinations


def test_optional_repeat_coordination_nesting_discontinuity_and_roles_are_typed():
    """可空、重复、协调、嵌套、非连续和 Role/scope 都由 payload 直接承载。"""
    seeds = read_authored_recursive_parse_seeds(SAMPLE_PATH)
    by_kind = {}
    for item in seeds:
        by_kind.setdefault(item.candidate_kind, item)
    optional = by_kind["OPTIONAL"].observation_payload().to_value()
    assert any(node["optional"] == 1
               for node in optional["parse_candidates"][0]["nodes"])
    repeated = by_kind["REPEATED"].observation_payload().to_value()
    repeat_nodes = [node for node in repeated["parse_candidates"][0]["nodes"]
                    if node["role_key"] == "ENTITY"]
    assert {node["repeat_ordinal"] for node in repeat_nodes} == {0, 1}
    coordinated = by_kind["COORDINATED"].observation_payload().to_value()
    assert any(edge["edge_kind"] == "COORDINATES"
               for edge in coordinated["parse_candidates"][0]["edges"])
    nested = validate_recursive_parse_payload(by_kind["NESTED"].observation_payload())
    assert nested.maximum_depth >= 3
    discontinuous = validate_recursive_parse_payload(
        by_kind["DISCONTINUOUS_DEPENDENCY"].observation_payload())
    assert discontinuous.discontinuous_node_count >= 1


def test_reverse_linearization_hides_surface_and_reparse_is_local():
    """生成目标只在私有 label；revision 只替代同 split 且 parser version 前进。"""
    seeds = read_authored_recursive_parse_seeds(SAMPLE_PATH)
    generations = tuple(item for item in seeds if item.candidate_kind == "GENERATION")
    assert len(generations) == 2
    for item in generations:
        payload = item.observation_payload().to_value()
        assert payload["observed_surface"]["target_hidden"] == 1
        assert payload["generation_constraint"]["output_surface_hidden"] == 1
        for surface in item.expected_payload.to_value()["accepted_surfaces"]:
            assert surface not in payload["observed_surface"]["text"]
    for revision in (item for item in seeds if item.candidate_kind == "REVISION"):
        target = next(item for item in seeds
                      if item.seed_id == revision.supersedes_seed_id)
        assert revision.family == target.family
        assert revision.split == target.split
        old = max(item.to_value()["parser_version"]
                  for item in target.parse_candidates)
        new = min(item.to_value()["parser_version"]
                  for item in revision.parse_candidates)
        assert new > old


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    first = compile_authored_recursive_parse_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_recursive_parse_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 72
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-05",)
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
        validate_recursive_parse_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    build = compile_authored_recursive_parse_course(SAMPLE_PATH, tmp_path / "pack")
    manifest = build_recursive_parse_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-04",)
    assert manifest.capability_keys == ("RECURSIVE_PARSE",)
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
        (lambda rows: rows.pop(5), "parse 候选族|sample family|competition"),
        (lambda rows: rows[9].__setitem__(
            "supersedes_seed_id", rows[10]["seed_id"]), "更早"),
        (lambda rows: rows[9]["parse_candidates"][0].__setitem__(
            "parser_version", 1), "version"),
        (lambda rows: rows[10].__setitem__(
            "retention_anchor_id", rows[11]["seed_id"]), "更早"),
        (lambda rows: rows[12].__setitem__(
            "evaluation_dimension", rows[13]["evaluation_dimension"]),
         "evaluator 维度"),
    ],
)
def test_bad_license_owner_family_future_links_and_dimension_fail_closed(
        tmp_path, mutate, message):
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredRecursiveParseCourseError, match=message):
        read_authored_recursive_parse_seeds(path)


def test_missing_held_out_depth_recombination_fails_closed(tmp_path):
    rows = _sample_values()
    rows[12]["filler_family"] = rows[0]["filler_family"]
    rows[12]["depth_family"] = rows[0]["depth_family"]
    rows[13]["filler_family"] = rows[1]["filler_family"]
    rows[13]["depth_family"] = rows[1]["depth_family"]
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredRecursiveParseCourseError, match="held-out"):
        read_authored_recursive_parse_seeds(path)


def test_payload_hash_parent_budget_edge_selection_and_generation_fail_closed():
    seeds = read_authored_recursive_parse_seeds(SAMPLE_PATH)
    positive = seeds[0]
    value = positive.observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredRecursiveParseCourseError, match="SHA-256"):
        validate_recursive_parse_payload(value)

    value = positive.observation_payload().to_value()
    value["parse_candidates"][0]["nodes"][1]["parent_node_id"] = "content"
    with pytest.raises(AuthoredRecursiveParseCourseError, match="环"):
        validate_recursive_parse_payload(value)

    nested = next(item for item in seeds if item.candidate_kind == "NESTED")
    value = nested.observation_payload().to_value()
    value["parse_budget"]["max_depth"] = 2
    with pytest.raises(AuthoredRecursiveParseCourseError, match="depth 超预算"):
        validate_recursive_parse_payload(value)

    value = positive.observation_payload().to_value()
    value["parse_candidates"][0]["edges"][0]["to_node_id"] = "missing"
    with pytest.raises(AuthoredRecursiveParseCourseError, match="端点非法"):
        validate_recursive_parse_payload(value)

    value = positive.observation_payload().to_value()
    value["selection_state"] = "SELECTED"
    with pytest.raises(AuthoredRecursiveParseCourseError, match="预选"):
        validate_recursive_parse_payload(value)

    generation = next(item for item in seeds if item.candidate_kind == "GENERATION")
    value = generation.observation_payload().to_value()
    value["generation_constraint"]["output_surface_hidden"] = 0
    with pytest.raises(AuthoredRecursiveParseCourseError, match="hidden"):
        validate_recursive_parse_payload(value)


def test_ambiguous_candidate_drop_and_single_tree_joint_forgery_fail_closed():
    seeds = read_authored_recursive_parse_seeds(SAMPLE_PATH)
    ambiguous = next(item for item in seeds if item.candidate_kind == "AMBIGUOUS")
    value = ambiguous.observation_payload().to_value()
    value["parse_candidates"].pop()
    with pytest.raises(AuthoredRecursiveParseCourseError, match="不得先选"):
        validate_recursive_parse_payload(value)
    baseline = next(item for item in seeds
                    if item.baseline_kind == "PRESELECTED_TREE_ONLY")
    value = baseline.observation_payload().to_value()
    value["generation_constraint"]["requires_joint_parse"] = 1
    with pytest.raises(AuthoredRecursiveParseCourseError, match="joint parse"):
        validate_recursive_parse_payload(value)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredRecursiveParseCourseError, match="损坏"):
        read_authored_recursive_parse_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredRecursiveParseCourseError, match="非规范 JSON|损坏"):
        read_authored_recursive_parse_seeds(path)
    compile_authored_recursive_parse_course(SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_recursive_parse_course(SAMPLE_PATH, tmp_path / "release")


def test_formal_repository_course_manifest_and_external_pack_are_exact():
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
