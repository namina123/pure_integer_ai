"""D-02 LC-14 转述、引语、holder 与认识 scope 课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_attribution_quotation_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredAttributionQuotationCourseError,
    build_attribution_quotation_course_manifest,
    compile_authored_attribution_quotation_course,
    default_attribution_quotation_sample_bytes,
    read_authored_attribution_quotation_seeds,
    validate_attribution_quotation_payload,
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


SAMPLE_PATH = Path(
    "data/ph2/authored_attribution_quotation_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "8201c1d9b825adf4f69c0de1c19d8b21aa9c1d3ba6bf16002ee51fe603d588d1")


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
    assert SAMPLE_PATH.read_bytes() == default_attribution_quotation_sample_bytes()
    seeds = read_authored_attribution_quotation_seeds(SAMPLE_PATH)
    assert len(seeds) == 34
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 17
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert len({item.candidate_kind for item in owner}) == 17
        assert {item.baseline_kind for item in owner} == {
            "QUOTE_BOUNDARY_SURFACE_ONLY",
            "REPORTED_AS_CURRENT_FACT",
            "TYPED_ATTRIBUTION_OBJECTS_PRESENT",
        }
        negatives = tuple(item for item in owner if item.baseline_kind != (
            "TYPED_ATTRIBUTION_OBJECTS_PRESENT"))
        assert len(negatives) == 2
        assert all(item.expected_state == "FALSE" for item in negatives)
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {item.evaluation_dimension for item in evaluator} == set(
        EVALUATOR_DIMENSIONS)


def test_reported_content_never_enters_current_projection():
    for seed in read_authored_attribution_quotation_seeds(SAMPLE_PATH):
        payload = seed.observation_payload().to_value()
        assert all(item["current_projection_allowed"] == 0
                   for item in payload["proposition_candidates"])
        assert all(item["current_projection_allowed"] == 0
                   for item in payload["attribution_candidates"])
        assert payload["selection_state"] == "UNSELECTED"


def test_claim_belief_hypothesis_and_nested_holder_are_typed():
    by_kind = _first_by_kind(
        read_authored_attribution_quotation_seeds(SAMPLE_PATH))
    expected = {
        "CLAIM": ("CLAIM", "ASSERTED", "REPORTED"),
        "BELIEF": ("BELIEF", "HELD", "UNCERTAIN"),
        "HYPOTHESIS": ("HYPOTHESIS", "HYPOTHESIZED", "UNCERTAIN"),
    }
    for candidate_kind, states in expected.items():
        item = by_kind[candidate_kind].observation_payload().to_value()[
            "attribution_candidates"][0]
        assert (item["attribution_kind"], item["epistemic_state"],
                item["uncertainty_state"]) == states
        assert item["holder_id"] and item["source_ref_key"]
    nested = by_kind["NESTED_HOLDER"].observation_payload().to_value()[
        "attribution_candidates"]
    assert len(nested) == 2
    assert nested[1]["parent_attribution_id"] == nested[0]["attribution_id"]
    assert nested[1]["nested_holder_id"]


def test_quote_span_and_paraphrase_version_are_raw_bound():
    by_kind = _first_by_kind(
        read_authored_attribution_quotation_seeds(SAMPLE_PATH))
    direct = by_kind["DIRECT_QUOTATION"].observation_payload().to_value()
    quote = direct["quotation_spans"][0]
    text = direct["observed_surface"]["text"]
    assert text[quote["start"]:quote["end"]] == quote["exact_text"]
    assert quote["boundary_state"] == "MATCH"
    assert quote["version_kind"] == "EXACT_QUOTE"
    paraphrase = by_kind["PARAPHRASE"].observation_payload().to_value()
    assert len(paraphrase["quotation_spans"]) == 2
    source, derived = paraphrase["quotation_spans"]
    assert derived["version_kind"] == "PARAPHRASE"
    assert derived["paraphrase_of_quote_id"] == source["quote_id"]
    assert derived["attribution_id"] != source["attribution_id"]


def test_pronoun_and_tense_transfer_preserve_holder_and_anchor():
    by_kind = _first_by_kind(
        read_authored_attribution_quotation_seeds(SAMPLE_PATH))
    for candidate_kind, transfer_kind in (
            ("PRONOUN_TRANSFER", "PRONOUN"),
            ("TENSE_TRANSFER", "TENSE")):
        receipt = by_kind[candidate_kind].observation_payload().to_value()[
            "transfer_receipt"]
        assert receipt["transfer_kind"] == transfer_kind
        assert receipt["source_form"] and receipt["target_form"]
        assert receipt["holder_preserved"] == 1
        assert receipt["temporal_anchor_preserved"] == 1


def test_denial_conflict_ambiguity_and_unknown_do_not_flatten_scope():
    by_kind = _first_by_kind(
        read_authored_attribution_quotation_seeds(SAMPLE_PATH))
    denial = by_kind["LATER_DENIAL"].observation_payload().to_value()[
        "attribution_candidates"]
    assert {item["epistemic_state"] for item in denial} == {
        "ASSERTED", "DENIED"}
    conflict = by_kind["SOURCE_CONFLICT"].observation_payload().to_value()[
        "attribution_candidates"]
    assert len({item["holder_id"] for item in conflict}) == 2
    assert {item["epistemic_state"] for item in conflict} == {"CONFLICT"}
    ambiguous = by_kind["AMBIGUOUS_SCOPE"].observation_payload().to_value()[
        "attribution_candidates"]
    assert {item["candidate_state"] for item in ambiguous} == {"COMPETING"}
    unknown = by_kind["UNKNOWN"].observation_payload().to_value()[
        "attribution_candidates"]
    assert {item["candidate_state"] for item in unknown} == {"UNKNOWN"}


def test_revision_changes_only_attribution_state_and_version():
    seeds = read_authored_attribution_quotation_seeds(SAMPLE_PATH)
    revision = next(item for item in seeds if item.candidate_kind == "REVISION")
    target = next(item for item in seeds if item.seed_id == revision.supersedes_seed_id)
    old = target.observation_payload().to_value()
    new = revision.observation_payload().to_value()
    assert old["proposition_candidates"] == new["proposition_candidates"]
    assert old["quotation_spans"] == new["quotation_spans"]
    assert old["transfer_receipt"] == new["transfer_receipt"]
    before = old["attribution_candidates"][0]
    after = new["attribution_candidates"][0]
    assert before["attribution_id"] == after["attribution_id"]
    assert before["holder_id"] == after["holder_id"]
    assert before["content_proposition_id"] == after["content_proposition_id"]
    assert before["candidate_state"] == "SUPPORTED"
    assert after["candidate_state"] == "REFUTED"
    assert before["candidate_version"] == 1
    assert after["candidate_version"] == 2


def test_retention_and_generation_are_read_only_and_hidden():
    seeds = read_authored_attribution_quotation_seeds(SAMPLE_PATH)
    for retention in (item for item in seeds if item.candidate_kind == "RETENTION"):
        anchor = next(item for item in seeds
                      if item.seed_id == retention.retention_anchor_id)
        assert anchor.logical_order < retention.logical_order
        assert anchor.split == retention.split
        assert anchor.candidate_kind == "BELIEF"
    for generation in (item for item in seeds if item.candidate_kind == "GENERATION"):
        payload = generation.observation_payload().to_value()
        constraint = payload["generation_constraint"]
        assert payload["observed_surface"]["target_hidden"] == 1
        assert constraint["direction"] == "ATTRIBUTION_TO_SURFACE"
        preservation = {
            key: value for key, value in constraint.items()
            if key.endswith("_required")}
        assert preservation and set(preservation.values()) == {1}
        for surface in generation.expected_payload.to_value()["accepted_surfaces"]:
            assert surface not in payload["observed_surface"]["text"]


def test_held_out_attribution_holder_source_recombination_is_direct():
    seeds = read_authored_attribution_quotation_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    axes = (
        {item.attribution_family for item in teacher},
        {item.holder_role for item in teacher},
        {item.source_channel for item in teacher},
    )
    teacher_triples = {
        (item.attribution_family, item.holder_role, item.source_channel)
        for item in teacher}
    held = {
        (item.attribution_family, item.holder_role, item.source_channel)
        for item in evaluator
        if item.attribution_family in axes[0]
        and item.holder_role in axes[1]
        and item.source_channel in axes[2]
        and (item.attribution_family, item.holder_role, item.source_channel)
        not in teacher_triples}
    assert len(held) >= 4
    assert ("CLAIM", "ORGANIZATION", "SPEECH") in held
    assert ("BELIEF", "SYSTEM", "DOCUMENT") in held


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    first = compile_authored_attribution_quotation_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_attribution_quotation_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 102
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-08",)
    assert first.validation.source_ref_count == 34
    assert first.validation.observation_count == 34
    assert first.validation.teacher_evidence_count == 17
    assert first.validation.evaluator_label_count == 17
    assert first.validation.source_cluster_count == 32
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
        validate_attribution_quotation_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    build = compile_authored_attribution_quotation_course(
        SAMPLE_PATH, tmp_path / "pack")
    manifest = build_attribution_quotation_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-14",)
    assert manifest.capability_keys == (
        "ATTRIBUTION_QUOTATION_PERSPECTIVE", "SOURCE_UNCERTAINTY_REALITY")
    assert manifest.pack_record_count == 102
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
        (lambda rows: rows[17].__setitem__("split", "train"), "owner/split"),
        (lambda rows: rows.pop(5), "候选族|sample family"),
        (lambda rows: rows[14].__setitem__(
            "supersedes_seed_id", rows[16]["seed_id"]), "更早|revision"),
        (lambda rows: rows[15].__setitem__(
            "retention_anchor_id", rows[16]["seed_id"]), "更早"),
        (lambda rows: rows[17].__setitem__(
            "evaluation_dimension", rows[18]["evaluation_dimension"]),
         "evaluator 维度"),
    ],
)
def test_bad_license_owner_family_future_links_and_dimension_fail_closed(
        tmp_path, mutate, message):
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredAttributionQuotationCourseError, match=message):
        read_authored_attribution_quotation_seeds(path)


def test_projection_hash_scope_budget_and_reference_fail_closed():
    seed = read_authored_attribution_quotation_seeds(SAMPLE_PATH)[0]
    value = seed.observation_payload().to_value()
    value["proposition_candidates"][0]["current_projection_allowed"] = 1
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="current projection"):
        validate_attribution_quotation_payload(value)

    value = seed.observation_payload().to_value()
    value["attribution_candidates"][0]["current_projection_allowed"] = 1
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="不得自动推出"):
        validate_attribution_quotation_payload(value)

    value = seed.observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredAttributionQuotationCourseError, match="SHA-256"):
        validate_attribution_quotation_payload(value)

    value = seed.observation_payload().to_value()
    value["proposition_candidates"][0]["context_scope_key"] = "OTHER"
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="ContextScope"):
        validate_attribution_quotation_payload(value)

    value = seed.observation_payload().to_value()
    value["resource_budget"]["max_attributions"] = 0
    with pytest.raises(AuthoredAttributionQuotationCourseError, match="正严格整数"):
        validate_attribution_quotation_payload(value)

    value = seed.observation_payload().to_value()
    value["attribution_candidates"][0]["content_proposition_id"] = "MISSING"
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="未知 Proposition"):
        validate_attribution_quotation_payload(value)


def test_quote_transfer_negative_and_generation_ablations_fail_closed():
    by_kind = _first_by_kind(
        read_authored_attribution_quotation_seeds(SAMPLE_PATH))
    direct = by_kind["DIRECT_QUOTATION"].observation_payload().to_value()
    direct["quotation_spans"][0]["end"] -= 1
    with pytest.raises(AuthoredAttributionQuotationCourseError, match="原话 span"):
        validate_attribution_quotation_payload(direct)

    paraphrase = by_kind["PARAPHRASE"].observation_payload().to_value()
    paraphrase["quotation_spans"][1]["paraphrase_of_quote_id"] = "MISSING"
    with pytest.raises(AuthoredAttributionQuotationCourseError, match="更早原话"):
        validate_attribution_quotation_payload(paraphrase)

    pronoun = by_kind["PRONOUN_TRANSFER"].observation_payload().to_value()
    pronoun["transfer_receipt"]["holder_preserved"] = 0
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="保留 holder"):
        validate_attribution_quotation_payload(pronoun)

    boundary = by_kind["QUOTE_BOUNDARY_BASELINE"].observation_payload().to_value()
    boundary["quotation_spans"][0]["boundary_state"] = "MATCH"
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="坏引语边界"):
        validate_attribution_quotation_payload(boundary)

    generation = by_kind["GENERATION"].observation_payload().to_value()
    generation["generation_constraint"]["uncertainty_preservation_required"] = 0
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="逐层保持"):
        validate_attribution_quotation_payload(generation)


def test_revision_identity_and_held_out_axis_forgery_fail_closed(tmp_path):
    rows = _sample_values()
    rows[14]["attribution_candidates"][0]["holder_id"] = "REPLACED"
    path = tmp_path / "bad-revision.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="holder/content/scope"):
        read_authored_attribution_quotation_seeds(path)

    seed = read_authored_attribution_quotation_seeds(SAMPLE_PATH)[0]
    value = seed.observation_payload().to_value()
    value["split_identity"]["holder_role"] = "SYSTEM"
    value["split_identity"]["combination_key"] = "CLAIM::SYSTEM::DOCUMENT"
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="holder_role"):
        validate_attribution_quotation_payload(value)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredAttributionQuotationCourseError, match="损坏"):
        read_authored_attribution_quotation_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredAttributionQuotationCourseError,
                       match="非规范 JSON|损坏"):
        read_authored_attribution_quotation_seeds(path)
    compile_authored_attribution_quotation_course(
        SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_attribution_quotation_course(
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
    assert restored.record_count == manifest.pack_record_count == 102
    assert restored.splits == manifest.pack_splits == ("train", "held_out")
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
