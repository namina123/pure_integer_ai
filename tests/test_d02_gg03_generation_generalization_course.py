"""GG-03 D-02E.4 多合法表达、组合 split 与四-owner pack 测试。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_generation_generalization_course import (
    CAPABILITY_KEYS,
    COURSE_MANIFEST_PATH,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    TASK_KEYS,
    AuthoredGenerationGeneralizationCourseError,
    build_default_generation_generalization_seed_values,
    build_generation_generalization_course_manifest,
    compile_authored_generation_generalization_course,
    default_generation_generalization_sample_bytes,
    read_authored_generation_generalization_seeds,
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
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    ABLATION_KEYS,
    BASELINE_KINDS,
    CANDIDATE_CASES,
    COMBINATION_KEY_AXES,
    COURSE_FAMILIES,
    EVALUATOR_DIMENSIONS,
    EXPECTED_PAYLOAD_KEYS,
    INDEPENDENT_VERIFIER_REQUIREMENTS,
    PAYLOAD_KEYS,
    PAYLOAD_KIND,
    RETENTION_PROTOCOLS,
    VERIFIER_NE_CONDITIONS,
    GenerationGeneralizationContractError,
    validate_generation_generalization_expected,
    validate_generation_generalization_payload,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


SAMPLE_PATH = Path(
    "data/ph2/authored_generation_generalization_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "f9f1b119658c2387541b0f1db8b6d82189bd793677151aa72eb5f51eecfab778")


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
    assert SAMPLE_PATH.read_bytes() == default_generation_generalization_sample_bytes()
    seeds = read_authored_generation_generalization_seeds(SAMPLE_PATH)
    assert len(seeds) == 28
    for owner_key in ("teacher", "evaluator"):
        owner = tuple(item for item in seeds if item.label_owner == owner_key)
        assert len(owner) == 14
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.course_family for item in owner} == set(COURSE_FAMILIES)
        assert {item.candidate_case for item in owner} == set(CANDIDATE_CASES)
    teacher = {item.family for item in seeds if item.label_owner == "teacher"}
    evaluator = {item.family for item in seeds if item.label_owner == "evaluator"}
    assert teacher.isdisjoint(evaluator)


def test_observation_has_five_unselected_layers_and_no_private_output_label():
    for seed in read_authored_generation_generalization_seeds(SAMPLE_PATH):
        payload = seed.observation_payload.to_value()
        audit = validate_generation_generalization_payload(seed.observation_payload)
        assert tuple(payload) == PAYLOAD_KEYS
        assert audit.choice_kinds == CHOICE_KINDS
        assert len(audit.surface_candidate_ids) == 3
        assert all(item["selection_state"] == "UNSELECTED"
                   for item in payload["choice_candidates"])
        assert all(item["complete_answer_template"] == 0
                   for item in payload["choice_candidates"])
        assert all(item["outcome_broadcast"] == 0
                   for item in payload["choice_candidates"])
        assert all(item["complete_answer"] == 0
                   for item in payload["surface_candidates"])
        assert "expected_payload" not in payload
        assert "accepted_surface_variants" not in payload


def test_private_labels_accept_sets_and_require_six_independent_verifiers():
    for seed in read_authored_generation_generalization_seeds(SAMPLE_PATH):
        expected = seed.expected_payload.to_value()
        assert tuple(expected) == EXPECTED_PAYLOAD_KEYS
        assert len(expected["accepted_surface_candidate_ids"]) == 2
        assert len(expected["accepted_surface_variants"]) == 2
        assert expected["surface_set_comparison"] == "SET_OR_CONSTRAINT"
        assert expected["unique_expected_string_forbidden"] == 1
        assert tuple(sorted(expected["independent_verifier_requirements"])) == (
            INDEPENDENT_VERIFIER_REQUIREMENTS)
        assert all(expected["independent_verifier_requirements"].values())
        serialized = seed.observation_payload.payload.decode("utf-8")
        assert not any(item in serialized
                       for item in expected["accepted_surface_variants"])


def test_held_out_complete_combinations_are_unseen_but_non_source_axes_are_seen():
    seeds = read_authored_generation_generalization_seeds(SAMPLE_PATH)
    teacher = tuple(validate_generation_generalization_payload(
        item.observation_payload) for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(validate_generation_generalization_payload(
        item.observation_payload) for item in seeds if item.label_owner == "evaluator")
    assert {item.combination_key for item in teacher}.isdisjoint(
        {item.combination_key for item in evaluator})
    for axis_index, axis in enumerate(COMBINATION_KEY_AXES):
        train_values = {item.combination_values[axis_index] for item in teacher}
        held_values = {item.combination_values[axis_index] for item in evaluator}
        if axis == "source_cluster":
            assert train_values.isdisjoint(held_values)
        else:
            assert held_values <= train_values
    assert all(item.combination_key.count("::") == 9
               for item in (*teacher, *evaluator))


def test_semantic_source_reference_context_and_stance_failures_are_layered():
    seeds = read_authored_generation_generalization_seeds(SAMPLE_PATH)
    by_case = {item.candidate_case: item for item in seeds
               if item.label_owner == "evaluator"}
    for case, dimension in {
        "RELATION_ROLE_SCOPE_DRIFT": "SEMANTIC_ROLE_SCOPE_POLARITY",
        "STRUCTURE_SLOT_ORDER_DRIFT": "STRUCTURE_SLOT_ORDER",
        "SOURCE_UNCERTAINTY_PRESERVATION": "SOURCE_UNCERTAINTY_CITATION",
        "REFERENCE_RECOVERABILITY": "ADDRESSEE_RECOVERABILITY",
        "ADDRESSEE_CONTEXT_NEGATIVE": "FAILURE_LAYER_LOCALIZATION",
        "STANCE_CONTENT_WORDING": "STANCE_CONTENT_WORDING_SEPARATION",
    }.items():
        seed = by_case[case]
        expected = seed.expected_payload.to_value()
        assert seed.evaluation_dimension == dimension
        target_states = [item["state"] for item in expected["choice_layer_states"]
                         if item["state"] != "TRUE"]
        assert len(target_states) <= 1


def test_revision_retention_and_use_replay_remain_evidence_only():
    seeds = read_authored_generation_generalization_seeds(SAMPLE_PATH)
    index = {item.seed_id: item for item in seeds}
    revisions = tuple(item for item in seeds if item.sample_family == "REVISION")
    retentions = tuple(item for item in seeds if item.sample_family == "RETENTION")
    assert len(revisions) == len(retentions) == 2
    for seed in revisions:
        target = index[seed.supersedes_seed_id]
        assert target.family == seed.family
        assert target.split == seed.split
        assert target.logical_order < seed.logical_order
    for seed in retentions:
        target = index[seed.retention_anchor_id]
        assert target.family == seed.family
        replay = seed.observation_payload.to_value()["replay_evidence"]
        assert replay["replay_kind"] == "EVIDENCE_ONLY"
        assert replay["assessment_update_present"] == 0
        assert replay["complete_template_promotion_forbidden"] == 1
        assert replay["exact_use_ids"]


def test_exact_memory_control_is_present_but_cannot_be_generalization_pass():
    controls = tuple(
        item for item in read_authored_generation_generalization_seeds(SAMPLE_PATH)
        if item.candidate_case == "EXACT_MEMORY_CONTROL")
    assert len(controls) == 2
    for seed in controls:
        audit = validate_generation_generalization_payload(seed.observation_payload)
        assert audit.exact_memory_control == 1
        assert seed.expected_state == "FALSE"
        assert seed.evaluation_dimension == "EXACT_MEMORY_BASELINE_REJECT"


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    first = compile_authored_generation_generalization_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_generation_generalization_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 84
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-09",)
    assert first.validation.source_ref_count == 28
    assert first.validation.observation_count == 28
    assert first.validation.teacher_evidence_count == 14
    assert first.validation.evaluator_label_count == 14
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
        validate_generation_generalization_payload(item.typed_payload)
        assert "expected_payload" not in item.typed_payload.to_value()
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    build = compile_authored_generation_generalization_course(
        SAMPLE_PATH, tmp_path / "pack")
    manifest = build_generation_generalization_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == TASK_KEYS
    assert manifest.capability_keys == CAPABILITY_KEYS
    assert manifest.pack_record_count == 84
    assert manifest.payload_keys == PAYLOAD_KEYS
    assert manifest.evaluator_dimensions == EVALUATOR_DIMENSIONS
    assert manifest.combination_axes == tuple(sorted(
        axis.upper() for axis in COMBINATION_KEY_AXES))
    assert manifest.baseline_kinds == BASELINE_KINDS
    assert manifest.ablation_keys == ABLATION_KEYS
    assert manifest.retention_protocols == RETENTION_PROTOCOLS
    assert manifest.verifier_ne_conditions == VERIFIER_NE_CONDITIONS
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
        (lambda rows: rows.pop(4), "sample family|课程族|candidate case|维度"),
        (lambda rows: rows[9].__setitem__(
            "supersedes_seed_id", rows[11]["seed_id"]), "更早"),
        (lambda rows: rows[13]["observation_payload"].__setitem__(
            "retention_anchor_id", rows[11]["seed_id"]), "retention"),
    ],
)
def test_bad_license_owner_coverage_and_future_links_fail_closed(
        tmp_path, mutate, message):
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredGenerationGeneralizationCourseError,
                       match=message):
        read_authored_generation_generalization_seeds(path)


def test_combination_axis_leak_and_unseen_component_fail_closed(tmp_path):
    rows = _sample_values()
    train_combo = rows[0]["observation_payload"]["combination_split"]
    held_combo = rows[14]["observation_payload"]["combination_split"]
    for axis in COMBINATION_KEY_AXES:
        held_combo[axis] = train_combo[axis]
    held_combo["combination_key"] = train_combo["combination_key"]
    path = tmp_path / "leak.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredGenerationGeneralizationCourseError,
                       match="完整组合泄漏|source cluster"):
        read_authored_generation_generalization_seeds(path)

    rows = _sample_values()
    split = rows[14]["observation_payload"]["combination_split"]
    split["structure_family"] = "NEVER_TRAIN_STRUCTURE"
    split["combination_key"] = "::".join(
        split[axis] for axis in COMBINATION_KEY_AXES)
    path = tmp_path / "unseen.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredGenerationGeneralizationCourseError,
                       match="structure_family"):
        read_authored_generation_generalization_seeds(path)


def test_payload_hash_template_broadcast_reference_and_unique_string_fail_closed():
    row = _sample_values()[0]
    payload = row["observation_payload"]
    variants = (
        lambda value: CanonicalJsonObject.from_value(value))
    bad = json.loads(json.dumps(payload, ensure_ascii=False))
    bad["observed_surface"]["sha256"] = "0" * 64
    with pytest.raises(GenerationGeneralizationContractError, match="hash"):
        validate_generation_generalization_payload(variants(bad))
    bad = json.loads(json.dumps(payload, ensure_ascii=False))
    bad["choice_candidates"][0]["complete_answer_template"] = 1
    with pytest.raises(GenerationGeneralizationContractError, match="模板"):
        validate_generation_generalization_payload(variants(bad))
    bad = json.loads(json.dumps(payload, ensure_ascii=False))
    bad["choice_candidates"][1]["outcome_broadcast"] = 1
    with pytest.raises(GenerationGeneralizationContractError, match="广播"):
        validate_generation_generalization_payload(variants(bad))
    bad = json.loads(json.dumps(payload, ensure_ascii=False))
    bad["context_contract"]["addressee_context"][
        "recoverable_reference_ids"] = [999999]
    with pytest.raises(GenerationGeneralizationContractError, match="shared visible"):
        validate_generation_generalization_payload(variants(bad))
    bad = json.loads(json.dumps(payload, ensure_ascii=False))
    bad["surface_constraints"]["unique_string_comparison"] = 1
    with pytest.raises(GenerationGeneralizationContractError, match="多合法"):
        validate_generation_generalization_payload(variants(bad))


def test_private_label_single_variant_leak_and_missing_verifier_fail_closed():
    row = _sample_values()[0]
    observation = CanonicalJsonObject.from_value(row["observation_payload"])
    expected = row["expected_payload"]

    bad = json.loads(json.dumps(expected, ensure_ascii=False))
    bad["accepted_surface_variants"] = bad["accepted_surface_variants"][:1]
    with pytest.raises(GenerationGeneralizationContractError, match="变体不足"):
        validate_generation_generalization_expected(
            CanonicalJsonObject.from_value(bad), expected_state=row["expected_state"],
            evaluation_dimension=row["evaluation_dimension"],
            observation_payload=observation)
    bad = json.loads(json.dumps(expected, ensure_ascii=False))
    bad["independent_verifier_requirements"].pop(
        INDEPENDENT_VERIFIER_REQUIREMENTS[0])
    with pytest.raises(GenerationGeneralizationContractError, match="verifier"):
        validate_generation_generalization_expected(
            CanonicalJsonObject.from_value(bad), expected_state=row["expected_state"],
            evaluation_dimension=row["evaluation_dimension"],
            observation_payload=observation)
    bad_observation = json.loads(json.dumps(row["observation_payload"],
                                            ensure_ascii=False))
    leaked = expected["accepted_surface_variants"][0]
    bad_observation["observed_surface"]["text"] += leaked
    bad_observation["observed_surface"]["sha256"] = hashlib.sha256(
        bad_observation["observed_surface"]["text"].encode("utf-8")).hexdigest()
    with pytest.raises(GenerationGeneralizationContractError, match="泄漏"):
        validate_generation_generalization_expected(
            CanonicalJsonObject.from_value(expected),
            expected_state=row["expected_state"],
            evaluation_dimension=row["evaluation_dimension"],
            observation_payload=CanonicalJsonObject.from_value(bad_observation))


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredGenerationGeneralizationCourseError, match="损坏"):
        read_authored_generation_generalization_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredGenerationGeneralizationCourseError,
                       match="非规范 JSON|损坏"):
        read_authored_generation_generalization_seeds(path)
    compile_authored_generation_generalization_course(
        SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_generation_generalization_course(
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
