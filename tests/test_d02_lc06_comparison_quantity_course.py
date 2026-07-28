"""D-02 LC-06 comparison, degree, quantity, range, and measure course T0."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_comparison_quantity_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredComparisonQuantityCourseError,
    build_comparison_quantity_course_manifest,
    compile_authored_comparison_quantity_course,
    default_comparison_quantity_sample_bytes,
    read_authored_comparison_quantity_seeds,
    validate_comparison_quantity_payload,
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


SAMPLE_PATH = Path("data/ph2/authored_comparison_quantity_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "beb51b2eb72b4827b107190fd0d9fac551fea7903c2d5fe42d8ec0e229a939bc")


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


def _first_by_kind(seeds) -> dict[str, object]:
    result = {}
    for seed in seeds:
        result.setdefault(seed.candidate_kind, seed)
    return result


def test_sample_is_generated_exactly_and_freezes_both_owners():
    """The formal sample is canonical and both owners cover all LC-06 classes."""
    assert SAMPLE_PATH.read_bytes() == default_comparison_quantity_sample_bytes()
    seeds = read_authored_comparison_quantity_seeds(SAMPLE_PATH)
    assert len(seeds) == 28
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 14
    expected_kinds = {
        "AMBIGUOUS_STANDARD", "APPROXIMATE_EXACT",
        "BARE_PROPERTY_BASELINE", "COMPARISON", "DEGREE", "GENERATION",
        "MEASURE", "QUANTITY_COUNT", "QUANTIFIER_SCOPE", "RANGE",
        "RETENTION", "REVISION", "UNIT_ERASURE_BASELINE", "UNKNOWN",
    }
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.candidate_kind for item in owner} == expected_kinds
        assert {item.baseline_kind for item in owner} == {
            "BARE_PROPERTY_ONLY", "TYPED_COMPARISON_OBJECTS_PRESENT",
            "UNIT_ERASURE",
        }
        negatives = tuple(item for item in owner if item.baseline_kind != (
            "TYPED_COMPARISON_OBJECTS_PRESENT"))
        assert len(negatives) == 2
        assert all(item.expected_state == "FALSE" for item in negatives)
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {item.evaluation_dimension for item in evaluator} == set(
        EVALUATOR_DIMENSIONS)


def test_comparison_degree_quantity_measure_and_range_are_direct():
    """Each claimed LC-06 dimension is carried by typed objects, not text cues."""
    by_kind = _first_by_kind(read_authored_comparison_quantity_seeds(SAMPLE_PATH))
    comparison = by_kind["COMPARISON"].observation_payload().to_value()
    assert len(comparison["object_candidates"]) == 2
    assert comparison["scale_definitions"][0]["direction"] == "INCREASING"
    assert comparison["standard_candidates"][0]["standard_kind"] == "EXPLICIT"
    assert comparison["comparison_candidates"][0]["operator"] == "GT"

    degree = by_kind["DEGREE"].observation_payload().to_value()
    assert degree["scale_definitions"][0]["threshold_num"] == 5
    assert degree["scale_definitions"][0]["threshold_den"] == 1
    measure = by_kind["MEASURE"].observation_payload().to_value()
    assert measure["unit_definitions"][0]["canonical_unit_id"] == "METER"
    assert measure["quantity_candidates"][0]["unit_id"]

    ranged = by_kind["RANGE"].observation_payload().to_value()
    quantity = ranged["quantity_candidates"][0]
    assert quantity["exactness"] == "RANGE"
    assert (quantity["lower_num"], quantity["amount_num"],
            quantity["upper_num"]) == (10, 12, 14)
    approximate = by_kind["APPROXIMATE_EXACT"].observation_payload().to_value()
    assert {item["exactness"] for item in approximate["quantity_candidates"]} == {
        "APPROXIMATE", "EXACT"}

    counted = by_kind["QUANTITY_COUNT"].observation_payload().to_value()
    assert counted["object_candidates"][0]["object_kind"] == "SET_EXPR"
    assert counted["quantity_candidates"][0]["exactness"] == "COUNT"
    assert counted["quantifier_scopes"][0]["quantifier_kind"] == "COUNT_EXACT"
    scoped = by_kind["QUANTIFIER_SCOPE"].observation_payload().to_value()
    assert [item["scope_order"] for item in scoped["quantifier_scopes"]] == [1, 2]
    assert [item["quantifier_kind"] for item in scoped["quantifier_scopes"]] == [
        "COUNT_EXACT", "FORALL"]


def test_ambiguity_unknown_and_two_negative_baselines_do_not_guess():
    by_kind = _first_by_kind(read_authored_comparison_quantity_seeds(SAMPLE_PATH))
    ambiguous = by_kind["AMBIGUOUS_STANDARD"]
    payload = ambiguous.observation_payload().to_value()
    assert ambiguous.expected_state == "CONFLICT"
    assert payload["selection_state"] == "UNSELECTED"
    assert len(payload["standard_candidates"]) == 2
    assert len(payload["comparison_candidates"]) == 2

    unknown = by_kind["UNKNOWN"]
    payload = unknown.observation_payload().to_value()
    assert unknown.expected_state == "UNKNOWN"
    assert payload["scale_definitions"][0]["direction"] == "UNKNOWN"
    assert payload["standard_candidates"][0]["standard_kind"] == "UNKNOWN"
    assert payload["quantity_candidates"][0]["exactness"] == "UNKNOWN"

    bare = by_kind["BARE_PROPERTY_BASELINE"]
    payload = bare.observation_payload().to_value()
    assert bare.expected_state == "FALSE"
    assert payload["scale_definitions"] == []
    assert payload["standard_candidates"] == []
    assert payload["comparison_candidates"] == []
    erased = by_kind["UNIT_ERASURE_BASELINE"]
    payload = erased.observation_payload().to_value()
    assert erased.expected_state == "FALSE"
    assert payload["unit_definitions"] == []
    assert payload["quantity_candidates"][0]["unit_id"] == ""


def test_revision_retention_generation_and_context_scope_are_local():
    seeds = read_authored_comparison_quantity_seeds(SAMPLE_PATH)
    for revision in (item for item in seeds if item.candidate_kind == "REVISION"):
        target = next(item for item in seeds
                      if item.seed_id == revision.supersedes_seed_id)
        old = target.observation_payload().to_value()
        new = revision.observation_payload().to_value()
        for key in (
                "object_candidates", "scale_definitions", "standard_candidates",
                "unit_definitions", "comparison_candidates", "quantifier_scopes"):
            assert old[key] == new[key]
        assert old["quantity_candidates"] != new["quantity_candidates"]
        assert new["revision_receipt"]["revision_scope"] == "QUANTITY_ONLY"
        assert new["revision_receipt"]["dependency_keys"] == ["QUANTITY"]
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
        assert all(constraint[key] == 1 for key in (
            "comparison_preservation_required", "quantity_preservation_required",
            "quantifier_scope_preservation_required", "scale_preservation_required",
            "standard_preservation_required", "unit_preservation_required",
        ))
        for surface in generation.expected_payload.to_value()["accepted_surfaces"]:
            assert surface not in payload["observed_surface"]["text"]
    for seed in seeds:
        payload = seed.observation_payload().to_value()
        scope = payload["surface_scope"]["context_scope_key"]
        for key in ("object_candidates", "scale_definitions",
                    "standard_candidates", "unit_definitions",
                    "comparison_candidates"):
            assert all(item["context_scope_key"] == scope for item in payload[key])


def test_held_out_object_scale_unit_recombination_is_direct():
    seeds = read_authored_comparison_quantity_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    teacher_objects = {item.object_family for item in teacher}
    teacher_scales = {item.scale_family for item in teacher}
    teacher_units = {item.unit_family for item in teacher}
    teacher_triples = {
        (item.object_family, item.scale_family, item.unit_family)
        for item in teacher}
    recombinations = {
        (item.object_family, item.scale_family, item.unit_family)
        for item in evaluator
        if item.object_family in teacher_objects
        and item.scale_family in teacher_scales
        and item.unit_family in teacher_units
        and (item.object_family, item.scale_family, item.unit_family)
        not in teacher_triples
    }
    assert {
        ("EVENT", "TEMPERATURE", "CELSIUS"),
        ("SET", "DURATION", "HOUR"),
        ("SET", "LENGTH", "METER"),
    }.issubset(recombinations)


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    first = compile_authored_comparison_quantity_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_comparison_quantity_course(
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
        validate_comparison_quantity_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    build = compile_authored_comparison_quantity_course(
        SAMPLE_PATH, tmp_path / "pack")
    manifest = build_comparison_quantity_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-06",)
    assert manifest.capability_keys == ("COMPARISON_QUANTITY_MEASURE",)
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
        (lambda rows: rows.pop(5), "候选族|sample family"),
        (lambda rows: rows[10].__setitem__(
            "supersedes_seed_id", rows[11]["seed_id"]), "更早"),
        (lambda rows: rows[11].__setitem__(
            "retention_anchor_id", rows[12]["seed_id"]), "更早"),
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
    with pytest.raises(AuthoredComparisonQuantityCourseError, match=message):
        read_authored_comparison_quantity_seeds(path)


def test_missing_held_out_triple_recombination_fails_closed(tmp_path):
    rows = _sample_values()
    component_keys = (
        "comparison_candidates", "generation_constraint", "object_candidates",
        "object_family", "quantifier_scopes", "quantity_candidates",
        "scale_definitions", "scale_family", "standard_candidates",
        "surface_scope", "unit_definitions", "unit_family",
    )
    for teacher, evaluator in zip(rows[:14], rows[14:], strict=True):
        for key in component_keys:
            evaluator[key] = teacher[key]
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="held-out"):
        read_authored_comparison_quantity_seeds(path)


def test_payload_hash_scope_budget_and_reference_fail_closed():
    seeds = read_authored_comparison_quantity_seeds(SAMPLE_PATH)
    positive = next(item for item in seeds if item.candidate_kind == "COMPARISON")
    value = positive.observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="SHA-256"):
        validate_comparison_quantity_payload(value)

    value = positive.observation_payload().to_value()
    value["object_candidates"][0]["context_scope_key"] = "OTHER"
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="ContextScope"):
        validate_comparison_quantity_payload(value)

    value = positive.observation_payload().to_value()
    value["resource_budget"]["max_objects"] = 1
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="超预算"):
        validate_comparison_quantity_payload(value)

    value = positive.observation_payload().to_value()
    value["quantity_candidates"][0]["object_id"] = "MISSING"
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="未知对象"):
        validate_comparison_quantity_payload(value)


def test_rational_unit_and_quantifier_scope_fail_closed():
    by_kind = _first_by_kind(read_authored_comparison_quantity_seeds(SAMPLE_PATH))
    ranged = by_kind["RANGE"].observation_payload().to_value()
    ranged["quantity_candidates"][0]["lower_num"] = 15
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="上下界"):
        validate_comparison_quantity_payload(ranged)

    measure = by_kind["MEASURE"].observation_payload().to_value()
    measure["unit_definitions"][0]["dimension_key"] = "MASS"
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="维度不一致"):
        validate_comparison_quantity_payload(measure)

    scoped = by_kind["QUANTIFIER_SCOPE"].observation_payload().to_value()
    scoped["quantifier_scopes"][1]["scope_order"] = 3
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="必须连续"):
        validate_comparison_quantity_payload(scoped)

    counted = by_kind["QUANTITY_COUNT"].observation_payload().to_value()
    quantity = counted["quantity_candidates"][0]
    quantity["amount_den"] = quantity["lower_den"] = quantity["upper_den"] = 2
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="计数必须是整数"):
        validate_comparison_quantity_payload(counted)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("object_family", "SET", "object_family"),
        ("scale_family", "MASS", "scale_family"),
        ("unit_family", "KILOGRAM", "unit_family"),
    ],
)
def test_combination_axes_cannot_lie_about_typed_payload(field, value, message):
    seed = next(item for item in read_authored_comparison_quantity_seeds(SAMPLE_PATH)
                if item.candidate_kind == "MEASURE")
    payload = seed.observation_payload().to_value()
    payload["split_identity"][field] = value
    payload["split_identity"]["combination_key"] = (
        f'{payload["split_identity"]["object_family"]}::'
        f'{payload["split_identity"]["scale_family"]}::'
        f'{payload["split_identity"]["unit_family"]}')
    with pytest.raises(AuthoredComparisonQuantityCourseError, match=message):
        validate_comparison_quantity_payload(payload)


def test_baseline_revision_generation_and_unknown_ablations_fail_closed(tmp_path):
    seeds = read_authored_comparison_quantity_seeds(SAMPLE_PATH)
    by_kind = _first_by_kind(seeds)
    bare = by_kind["BARE_PROPERTY_BASELINE"].observation_payload().to_value()
    bare["baseline_kind"] = "TYPED_COMPARISON_OBJECTS_PRESENT"
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="裸 PROPERTY"):
        validate_comparison_quantity_payload(bare)

    erased = by_kind["UNIT_ERASURE_BASELINE"].observation_payload().to_value()
    erased["baseline_kind"] = "TYPED_COMPARISON_OBJECTS_PRESENT"
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="单位擦除"):
        validate_comparison_quantity_payload(erased)

    generation = by_kind["GENERATION"].observation_payload().to_value()
    generation["generation_constraint"]["unit_preservation_required"] = 0
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="逐层保持"):
        validate_comparison_quantity_payload(generation)

    unknown = by_kind["UNKNOWN"].observation_payload().to_value()
    unknown["scale_definitions"][0]["direction"] = "INCREASING"
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="unknown"):
        validate_comparison_quantity_payload(unknown)

    rows = _sample_values()
    rows[10]["object_candidates"][0]["proposition_key"] = "REPLACED"
    path = tmp_path / "bad-revision.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="不得替换"):
        read_authored_comparison_quantity_seeds(path)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredComparisonQuantityCourseError, match="损坏"):
        read_authored_comparison_quantity_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredComparisonQuantityCourseError,
                       match="非规范 JSON|损坏"):
        read_authored_comparison_quantity_seeds(path)
    compile_authored_comparison_quantity_course(SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_comparison_quantity_course(
            SAMPLE_PATH, tmp_path / "release")


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
