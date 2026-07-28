"""D-02 LC-05 Event/State、时间锚、区间和体貌课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_event_time_aspect_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredEventTimeAspectCourseError,
    build_event_time_aspect_course_manifest,
    compile_authored_event_time_aspect_course,
    default_event_time_aspect_sample_bytes,
    read_authored_event_time_aspect_seeds,
    validate_event_time_aspect_payload,
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


SAMPLE_PATH = Path("data/ph2/authored_event_time_aspect_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "8ebc3070b1ab624138618838340b7ecf98798ab03cb5b87170c0220f2c439555")


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


def test_sample_is_generated_exactly_and_freezes_both_owners():
    """正式 sample 必须等于规范生成物，双 owner 覆盖七族和十四候选。"""
    assert SAMPLE_PATH.read_bytes() == default_event_time_aspect_sample_bytes()
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    assert len(seeds) == 28
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 14
    expected_kinds = {
        "AMBIGUOUS_ANCHOR", "ANCHORED_INTERVAL", "COMPLETED", "DURATIVE",
        "EVENT", "GENERATION", "HABITUAL_ITERATIVE",
        "IMPLICIT_NOW_BASELINE", "NARRATIVE_ORDER", "RETENTION", "REVISION",
        "STATE", "SURFACE_ORDER_BASELINE", "UNKNOWN",
    }
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.candidate_kind for item in owner} == expected_kinds
        assert {item.baseline_kind for item in owner} == {
            "IMPLICIT_NOW_ASSUMPTION", "SURFACE_ORDER_ONLY",
            "TYPED_TEMPORAL_OBJECTS_PRESENT",
        }
        negatives = tuple(item for item in owner if item.baseline_kind != (
            "TYPED_TEMPORAL_OBJECTS_PRESENT"))
        assert len(negatives) == 2
        assert all(item.expected_state == "FALSE" for item in negatives)
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {item.evaluation_dimension for item in evaluator} == set(
        EVALUATOR_DIMENSIONS)


def test_event_state_anchor_interval_and_aspect_taxonomy_is_direct():
    """Event/State 分账及锚、区间、四种体貌均由 typed payload 直接承载。"""
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    by_kind = {}
    for item in seeds:
        by_kind.setdefault(item.candidate_kind, item)
    event = by_kind["EVENT"].observation_payload().to_value()
    state = by_kind["STATE"].observation_payload().to_value()
    assert {item["semantic_kind"] for item in event["event_state_candidates"]} == {
        "EVENT"}
    assert {item["semantic_kind"] for item in state["event_state_candidates"]} == {
        "STATE"}
    assert all(item["carrier_kind"] == "EVENT"
               for payload in (event, state)
               for item in payload["event_state_candidates"])

    anchored = by_kind["ANCHORED_INTERVAL"].observation_payload().to_value()
    assert len(anchored["temporal_anchors"]) == 2
    assert len(anchored["intervals"]) == 1
    assert anchored["intervals"][0]["order_known"] == 1
    durative = by_kind["DURATIVE"].observation_payload().to_value()
    assert {item["aspect_kind"] for item in durative["aspect_profiles"]} == {
        "DURATIVE"}
    completed = by_kind["COMPLETED"].observation_payload().to_value()
    profile = completed["aspect_profiles"][0]
    assert (profile["aspect_kind"], profile["bounded"], profile["completed"]) == (
        "COMPLETED", 1, 1)
    repeated = by_kind["HABITUAL_ITERATIVE"].observation_payload().to_value()
    assert {item["aspect_kind"] for item in repeated["aspect_profiles"]} == {
        "HABITUAL", "ITERATIVE"}
    assert next(item for item in repeated["aspect_profiles"]
                if item["aspect_kind"] == "ITERATIVE")["iteration_count"] == 3


def test_narrative_ambiguity_unknown_and_negative_baselines_do_not_guess():
    """叙事顺序、同表层双锚和 unknown 显式；现实 now/文本顺序基线为 FALSE。"""
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    by_kind = {}
    for item in seeds:
        by_kind.setdefault(item.candidate_kind, item)
    narrative = by_kind["NARRATIVE_ORDER"].observation_payload().to_value()
    assert [item["relation_kind"] for item in narrative["narrative_relations"]] == [
        "BEFORE"]
    ambiguous = by_kind["AMBIGUOUS_ANCHOR"].observation_payload().to_value()
    assert ambiguous["selection_state"] == "UNSELECTED"
    assert len(ambiguous["temporal_anchors"]) == 2
    assert len({item["anchor_id"] for item in ambiguous["aspect_profiles"]}) == 2
    unknown = by_kind["UNKNOWN"].observation_payload().to_value()
    assert unknown["temporal_anchors"][0]["anchor_kind"] == "UNKNOWN"
    assert unknown["aspect_profiles"][0]["aspect_kind"] == "UNKNOWN"

    implicit = by_kind["IMPLICIT_NOW_BASELINE"]
    implicit_payload = implicit.observation_payload().to_value()
    assert implicit.expected_state == "FALSE"
    assert implicit_payload["temporal_anchors"] == []
    assert implicit_payload["intervals"] == []
    surface = by_kind["SURFACE_ORDER_BASELINE"]
    assert surface.expected_state == "FALSE"
    assert surface.observation_payload().to_value()[
        "narrative_relations"][0]["relation_kind"] == "UNKNOWN"


def test_revision_retention_generation_and_context_scope_are_local():
    """时间修正只换锚，retention 指向更早样本，生成目标隐藏且保持三层。"""
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    for revision in (item for item in seeds if item.candidate_kind == "REVISION"):
        target = next(item for item in seeds
                      if item.seed_id == revision.supersedes_seed_id)
        old = target.observation_payload().to_value()
        new = revision.observation_payload().to_value()
        assert old["event_state_candidates"] == new["event_state_candidates"]
        assert old["temporal_anchors"] != new["temporal_anchors"]
        assert new["revision_receipt"]["revision_scope"] == "ANCHOR_ONLY"
        assert new["revision_receipt"]["dependency_keys"] == ["TIME_ANCHOR"]
        assert new["revision_receipt"]["raw_observation_preserved"] == 1
    for retention in (item for item in seeds if item.candidate_kind == "RETENTION"):
        anchor = next(item for item in seeds
                      if item.seed_id == retention.retention_anchor_id)
        assert anchor.logical_order < retention.logical_order
        assert anchor.split == retention.split
    for generation in (item for item in seeds if item.candidate_kind == "GENERATION"):
        payload = generation.observation_payload().to_value()
        constraint = payload["generation_constraint"]
        assert payload["observed_surface"]["target_hidden"] == 1
        assert constraint["output_surface_hidden"] == 1
        assert constraint["anchor_preservation_required"] == 1
        assert constraint["aspect_preservation_required"] == 1
        assert constraint["event_state_preservation_required"] == 1
        for surface in generation.expected_payload.to_value()["accepted_surfaces"]:
            assert surface not in payload["observed_surface"]["text"]
    for seed in seeds:
        payload = seed.observation_payload().to_value()
        scope = payload["surface_scope"]["context_scope_key"]
        assert all(item["context_scope_key"] == scope
                   for item in payload["event_state_candidates"])
        assert all(item["context_scope_key"] == scope
                   for item in payload["temporal_anchors"])


def test_held_out_event_aspect_anchor_recombination_is_direct():
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    teacher_events = {item.event_kind for item in teacher}
    teacher_aspects = {item.aspect_family for item in teacher}
    teacher_anchors = {item.anchor_family for item in teacher}
    teacher_triples = {
        (item.event_kind, item.aspect_family, item.anchor_family)
        for item in teacher
    }
    recombinations = {
        (item.event_kind, item.aspect_family, item.anchor_family)
        for item in evaluator
        if item.event_kind in teacher_events
        and item.aspect_family in teacher_aspects
        and item.anchor_family in teacher_anchors
        and (item.event_kind, item.aspect_family, item.anchor_family)
        not in teacher_triples
    }
    assert ("EVENT", "DURATIVE", "CALENDAR") in recombinations
    assert ("STATE", "COMPLETED", "DEICTIC") in recombinations


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    first = compile_authored_event_time_aspect_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_event_time_aspect_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 84
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-05",)
    assert first.validation.source_ref_count == 28
    assert first.validation.observation_count == 28
    assert first.validation.teacher_evidence_count == 14
    assert first.validation.evaluator_label_count == 14
    assert first.validation.source_cluster_count == 26
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
        validate_event_time_aspect_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    build = compile_authored_event_time_aspect_course(
        SAMPLE_PATH, tmp_path / "pack")
    manifest = build_event_time_aspect_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-05",)
    assert manifest.capability_keys == ("EVENT_TIME_ASPECT",)
    assert manifest.pack_record_count == 84
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
        (lambda rows: rows[14].__setitem__("split", "train"), "split"),
        (lambda rows: rows.pop(5), "时间候选族|sample family"),
        (lambda rows: rows[10].__setitem__(
            "supersedes_seed_id", rows[11]["seed_id"]), "更早"),
        (lambda rows: rows[12].__setitem__(
            "retention_anchor_id", rows[13]["seed_id"]), "更早"),
        (lambda rows: rows[14].__setitem__(
            "evaluation_dimension", rows[15]["evaluation_dimension"]),
         "evaluator 维度"),
    ],
)
def test_bad_license_owner_family_future_links_and_dimension_fail_closed(
        tmp_path, mutate, message):
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredEventTimeAspectCourseError, match=message):
        read_authored_event_time_aspect_seeds(path)


def test_missing_held_out_triple_recombination_fails_closed(tmp_path):
    rows = _sample_values()
    teacher_triples = [
        (row["event_kind"], row["aspect_family"], row["anchor_family"])
        for row in rows[:14]
    ]
    for index, row in enumerate(rows[14:]):
        event_kind, aspect_family, anchor_family = teacher_triples[index]
        row["event_kind"] = event_kind
        row["aspect_family"] = aspect_family
        row["anchor_family"] = anchor_family
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="held-out"):
        read_authored_event_time_aspect_seeds(path)


def test_payload_hash_scope_budget_and_reference_fail_closed():
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    positive = seeds[0]
    value = positive.observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="SHA-256"):
        validate_event_time_aspect_payload(value)

    value = positive.observation_payload().to_value()
    value["event_state_candidates"][0]["context_scope_key"] = "OTHER"
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="ContextScope"):
        validate_event_time_aspect_payload(value)

    anchored = next(item for item in seeds
                    if item.candidate_kind == "ANCHORED_INTERVAL")
    value = anchored.observation_payload().to_value()
    value["resource_budget"]["max_anchors"] = 1
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="超预算"):
        validate_event_time_aspect_payload(value)

    value = anchored.observation_payload().to_value()
    value["intervals"][0]["end_anchor_id"] = "MISSING"
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="引用非法"):
        validate_event_time_aspect_payload(value)


def test_object_anchor_aspect_and_baseline_ablations_fail_closed():
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    state = next(item for item in seeds if item.candidate_kind == "STATE")
    value = state.observation_payload().to_value()
    value["event_state_candidates"][0]["semantic_kind"] = "EVENT"
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="State 分账"):
        validate_event_time_aspect_payload(value)

    anchored = next(item for item in seeds
                    if item.candidate_kind == "ANCHORED_INTERVAL")
    value = anchored.observation_payload().to_value()
    value["temporal_anchors"].pop()
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="引用非法|缺锚"):
        validate_event_time_aspect_payload(value)

    completed = next(item for item in seeds if item.candidate_kind == "COMPLETED")
    value = completed.observation_payload().to_value()
    value["aspect_profiles"][0]["completed"] = 0
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="完成体"):
        validate_event_time_aspect_payload(value)

    implicit = next(item for item in seeds
                    if item.candidate_kind == "IMPLICIT_NOW_BASELINE")
    value = implicit.observation_payload().to_value()
    value["baseline_kind"] = "TYPED_TEMPORAL_OBJECTS_PRESENT"
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="现实 now"):
        validate_event_time_aspect_payload(value)

    surface = next(item for item in seeds
                   if item.candidate_kind == "SURFACE_ORDER_BASELINE")
    value = surface.observation_payload().to_value()
    value["narrative_relations"][0]["relation_kind"] = "BEFORE"
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="PRECEDES"):
        validate_event_time_aspect_payload(value)


def test_revision_identity_generation_preservation_and_unknown_fail_closed():
    seeds = read_authored_event_time_aspect_seeds(SAMPLE_PATH)
    rows = _sample_values()
    rows[10]["event_state_candidates"][0]["proposition_key"] = "REPLACED"
    path_rows = rows
    assert path_rows[10]["candidate_kind"] == "REVISION"

    generation = next(item for item in seeds if item.candidate_kind == "GENERATION")
    value = generation.observation_payload().to_value()
    value["generation_constraint"]["aspect_preservation_required"] = 0
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="逐层保持"):
        validate_event_time_aspect_payload(value)

    unknown = next(item for item in seeds if item.candidate_kind == "UNKNOWN")
    value = unknown.observation_payload().to_value()
    value["temporal_anchors"][0]["anchor_kind"] = "CALENDAR"
    value["aspect_profiles"][0]["aspect_kind"] = "NONE"
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="unknown"):
        validate_event_time_aspect_payload(value)


def test_revision_cannot_replace_event_identity(tmp_path):
    rows = _sample_values()
    rows[10]["event_state_candidates"][0]["proposition_key"] = "REPLACED"
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="Event/State 身份"):
        read_authored_event_time_aspect_seeds(path)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="损坏"):
        read_authored_event_time_aspect_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredEventTimeAspectCourseError, match="非规范 JSON|损坏"):
        read_authored_event_time_aspect_seeds(path)
    compile_authored_event_time_aspect_course(SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_event_time_aspect_course(SAMPLE_PATH, tmp_path / "release")


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
    assert restored.record_count == manifest.pack_record_count == 84
    assert restored.splits == manifest.pack_splits == ("train", "held_out")
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
