"""D-02 LC-08 开放集、最小澄清与主动补证课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_open_set_clarification_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredOpenSetClarificationCourseError,
    build_open_set_clarification_course_manifest,
    compile_authored_open_set_clarification_course,
    default_open_set_clarification_sample_bytes,
    read_authored_open_set_clarification_seeds,
    validate_open_set_clarification_payload,
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
    "data/ph2/authored_open_set_clarification_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "4191cb06e44e30699e38ffadde0daf4adcec42599ddde88a2af483f79b3fce12")


def _sample_values() -> list[dict]:
    """回读正式 sample 为可破坏的测试对象。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """用共用规范 JSON serializer 写破坏样本。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _records_by_kind(build, kind: str):
    """从课程 pack 读取指定 owner 的 record。"""
    return tuple(
        record
        for identity in build.manifest.files
        if identity.record_kind == kind
        for record in read_record_artifact(build.pack_root, identity)
    )


def _first_by_kind(seeds) -> dict[str, object]:
    """取各候选类的首个样本，避免依赖 owner 文本。"""
    result = {}
    for seed in seeds:
        result.setdefault(seed.candidate_kind, seed)
    return result


def test_sample_is_generated_exactly_and_freezes_both_owners():
    """正式 sample 必须可重建，双 owner 各覆盖十七类候选。"""
    assert SAMPLE_PATH.read_bytes() == default_open_set_clarification_sample_bytes()
    seeds = read_authored_open_set_clarification_seeds(SAMPLE_PATH)
    assert len(seeds) == 34
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 17
    expected_kinds = {
        "ACCESS_BLOCKED", "ACTIVE_EVIDENCE_REQUEST", "AMBIGUOUS_BRANCH",
        "ANSWER_EVIDENCE_UPDATE", "BUDGET_BLOCKED", "GENERATION",
        "INSUFFICIENT_GUESS_BASELINE", "KNOWN_SUFFICIENT_NO_QUESTION",
        "MINIMAL_CLARIFICATION", "NEW_CONSTRUCTION_DETECTION",
        "NEW_SENSE_AMBIGUITY", "NEW_USAGE_DETECTION", "NEW_WORD_DETECTION",
        "OVERQUESTION_BASELINE", "RETENTION", "REVISION", "UNKNOWN",
    }
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.candidate_kind for item in owner} == expected_kinds
        assert {item.open_set_kind for item in owner} == {
            "KNOWN", "NEW_CONSTRUCTION", "NEW_SENSE", "NEW_USAGE",
            "NEW_WORD", "UNKNOWN",
        }
        negatives = tuple(item for item in owner if item.baseline_kind != (
            "TYPED_OPEN_SET_OBJECTS_PRESENT"))
        assert len(negatives) == 2
        assert all(item.expected_state == "FALSE" for item in negatives)
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {item.evaluation_dimension for item in evaluator} == set(
        EVALUATOR_DIMENSIONS)


def test_four_novelty_types_are_direct_typed_candidates():
    """新词、新义、新构式和新用法必须由 novelty 与 branch 直接承载。"""
    by_kind = _first_by_kind(
        read_authored_open_set_clarification_seeds(SAMPLE_PATH))
    mapping = {
        "NEW_CONSTRUCTION_DETECTION": ("NEW_CONSTRUCTION", "CONSTRUCTION"),
        "NEW_SENSE_AMBIGUITY": ("NEW_SENSE", "SENSE"),
        "NEW_USAGE_DETECTION": ("NEW_USAGE", "USAGE"),
        "NEW_WORD_DETECTION": ("NEW_WORD", "LEXEME"),
    }
    for candidate_kind, (open_kind, object_kind) in mapping.items():
        payload = by_kind[candidate_kind].observation_payload().to_value()
        assert payload["novelty_profile"]["open_set_kind"] == open_kind
        assert payload["novelty_profile"]["detection_state"] == "NOVEL"
        assert {item["object_kind"] for item in payload[
            "candidate_branches"]} == {object_kind}
        assert payload["selection_state"] == "UNSELECTED"


def test_missing_reasons_and_stop_states_remain_separate():
    """unknown、ambiguous、budget 与 access 不得被压成同一停止原因。"""
    by_kind = _first_by_kind(
        read_authored_open_set_clarification_seeds(SAMPLE_PATH))
    cases = {
        "NEW_SENSE_AMBIGUITY": ("AMBIGUOUS", "VALID", "AVAILABLE"),
        "ACTIVE_EVIDENCE_REQUEST": ("UNKNOWN", "VALID", "AVAILABLE"),
        "BUDGET_BLOCKED": ("BUDGET", "NE", "AVAILABLE"),
        "ACCESS_BLOCKED": ("ACCESS", "NE", "NO_ACCESS"),
        "UNKNOWN": ("UNKNOWN", "UNKNOWN", "UNKNOWN"),
    }
    observed = set()
    for kind, expected in cases.items():
        payload = by_kind[kind].observation_payload().to_value()
        actual = (
            payload["missing_information_obligations"][0]["blocking_reason"],
            payload["clarification_plan"]["plan_state"],
            payload["evidence_request"]["access_state"],
        )
        assert actual == expected
        observed.add(actual)
    assert len(observed) == len(cases)


def test_minimal_question_active_request_and_no_question_are_typed():
    """最小澄清、主动补证和证据充分停止必须使用不同计划。"""
    by_kind = _first_by_kind(
        read_authored_open_set_clarification_seeds(SAMPLE_PATH))
    minimal = by_kind["MINIMAL_CLARIFICATION"].observation_payload().to_value()
    plan = minimal["clarification_plan"]
    assert plan["question_kind"] == "MINIMAL_BRANCH_ELIMINATION"
    assert len(plan["target_candidate_ids"]) == 2
    assert plan["expected_eliminated_branches"] == 1
    assert plan["minimality_checked"] == 1

    active = by_kind["ACTIVE_EVIDENCE_REQUEST"].observation_payload().to_value()
    assert active["clarification_plan"]["question_kind"] == "EVIDENCE_REQUEST"
    assert active["evidence_request"]["path_kind"] == "SOURCE_LOOKUP"
    assert active["evidence_request"]["status"] == "OPEN"

    sufficient = by_kind[
        "KNOWN_SUFFICIENT_NO_QUESTION"].observation_payload().to_value()
    assert sufficient["clarification_plan"]["question_kind"] == "NO_QUESTION"
    assert sufficient["evidence_request"]["status"] == "NOT_REQUIRED"
    assert sufficient["missing_information_obligations"][0]["status"] == (
        "NOT_REQUIRED")


def test_negative_baselines_reject_overquestion_and_insufficient_guess():
    """证据充分仍追问和证据不足直接猜测必须是独立 FALSE label。"""
    by_kind = _first_by_kind(
        read_authored_open_set_clarification_seeds(SAMPLE_PATH))
    over = by_kind["OVERQUESTION_BASELINE"]
    guess = by_kind["INSUFFICIENT_GUESS_BASELINE"]
    assert over.expected_state == guess.expected_state == "FALSE"
    assert over.baseline_kind == "SUFFICIENT_EVIDENCE_OVERQUESTION"
    assert guess.baseline_kind == "INSUFFICIENT_EVIDENCE_GUESS"
    assert over.observation_payload().to_value()[
        "clarification_plan"]["plan_state"] == "INVALID"
    assert guess.observation_payload().to_value()[
        "clarification_plan"]["plan_state"] == "INVALID"


def test_answer_updates_only_dependent_candidates():
    """澄清回答只可更新 obligation 依赖候选并保留 raw。"""
    by_kind = _first_by_kind(
        read_authored_open_set_clarification_seeds(SAMPLE_PATH))
    payload = by_kind[
        "ANSWER_EVIDENCE_UPDATE"].observation_payload().to_value()
    receipt = payload["clarification_receipt"]
    obligation = payload["missing_information_obligations"][0]
    assert receipt["update_scope"] == "CLARIFICATION_DEPENDENCIES_ONLY"
    assert receipt["raw_observation_preserved"] == 1
    assert set(receipt["affected_candidate_ids"]) == set(
        obligation["candidate_ids"])
    assert set(receipt["affected_candidate_ids"]).isdisjoint(
        receipt["unaffected_candidate_ids"])
    assert obligation["status"] == "SATISFIED"
    assert payload["evidence_request"]["status"] == "SATISFIED"


def test_revision_preserves_unaffected_third_candidate_bit_identically():
    """revision 只能推进前两候选，第三候选必须逐字段不变。"""
    seeds = read_authored_open_set_clarification_seeds(SAMPLE_PATH)
    for revision in (item for item in seeds if item.candidate_kind == "REVISION"):
        target = next(item for item in seeds
                      if item.seed_id == revision.supersedes_seed_id)
        old = target.observation_payload().to_value()
        new = revision.observation_payload().to_value()
        old_by_id = {item["candidate_id"]: item
                     for item in old["candidate_branches"]}
        new_by_id = {item["candidate_id"]: item
                     for item in new["candidate_branches"]}
        unaffected = new["clarification_receipt"]["unaffected_candidate_ids"]
        assert len(unaffected) == 1
        assert old_by_id[unaffected[0]] == new_by_id[unaffected[0]]
        assert new["missing_information_obligations"][0]["status"] == (
            "SATISFIED")
        assert new["evidence_request"]["status"] == "SATISFIED"


def test_retention_and_generation_are_read_only_and_hidden():
    """retention 只指旧锚；生成问题只存在于私有 label。"""
    seeds = read_authored_open_set_clarification_seeds(SAMPLE_PATH)
    for retention in (item for item in seeds if item.candidate_kind == "RETENTION"):
        anchor = next(item for item in seeds
                      if item.seed_id == retention.retention_anchor_id)
        assert anchor.logical_order < retention.logical_order
        assert anchor.split == retention.split
        assert anchor.candidate_kind == "NEW_CONSTRUCTION_DETECTION"
    for generation in (item for item in seeds if item.candidate_kind == "GENERATION"):
        payload = generation.observation_payload().to_value()
        constraint = payload["generation_constraint"]
        assert payload["observed_surface"]["target_hidden"] == 1
        assert constraint["output_surface_hidden"] == 1
        assert constraint["direction"] == "OBLIGATION_TO_QUESTION"
        preservation = {
            key: value for key, value in constraint.items()
            if key.endswith("_preservation_required")}
        assert preservation and set(preservation.values()) == {1}
        for surface in generation.expected_payload.to_value()["accepted_surfaces"]:
            assert surface not in payload["observed_surface"]["text"]


def test_held_out_open_set_clarification_evidence_recombination_is_direct():
    """held-out 完整三轴未见，而三个单轴都必须在 teacher 出现。"""
    seeds = read_authored_open_set_clarification_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    teacher_open = {item.open_set_kind for item in teacher}
    teacher_clarification = {item.clarification_need for item in teacher}
    teacher_evidence = {item.evidence_path for item in teacher}
    teacher_triples = {
        (item.open_set_kind, item.clarification_need, item.evidence_path)
        for item in teacher}
    recombinations = {
        (item.open_set_kind, item.clarification_need, item.evidence_path)
        for item in evaluator
        if item.open_set_kind in teacher_open
        and item.clarification_need in teacher_clarification
        and item.evidence_path in teacher_evidence
        and (item.open_set_kind, item.clarification_need, item.evidence_path)
        not in teacher_triples
    }
    assert len(recombinations) == 5
    assert {
        ("NEW_CONSTRUCTION", "EVIDENCE_REQUEST", "SOURCE_LOOKUP"),
        ("NEW_SENSE", "MINIMAL_BRANCH_ELIMINATION", "SOURCE_LOOKUP"),
        ("NEW_WORD", "EVIDENCE_REQUEST", "LOCAL_OBSERVATION"),
    }.issubset(recombinations)


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    """编译只复用四 owner record，且两次 transport 字节一致。"""
    first = compile_authored_open_set_clarification_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_open_set_clarification_course(
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
        validate_open_set_clarification_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    """课程 manifest 必须不可覆盖、零执行并绑定两项能力。"""
    build = compile_authored_open_set_clarification_course(
        SAMPLE_PATH, tmp_path / "pack")
    manifest = build_open_set_clarification_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-08",)
    assert manifest.capability_keys == (
        "OPEN_SET_CONTINUAL_LEARNING", "PRAGMATIC_CLARIFICATION_REPAIR")
    assert manifest.sample_relative_path == (
        "data/ph2/authored_open_set_clarification_seed_v1.jsonl.sample")
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
        (lambda rows: rows[17].__setitem__("split", "train"), "split"),
        (lambda rows: rows.pop(5), "候选族|sample family"),
        (lambda rows: rows[14].__setitem__(
            "supersedes_seed_id", rows[16]["seed_id"]), "更早"),
        (lambda rows: rows[15].__setitem__(
            "retention_anchor_id", rows[16]["seed_id"]), "更早"),
        (lambda rows: rows[17].__setitem__(
            "evaluation_dimension", rows[18]["evaluation_dimension"]),
         "evaluator 维度"),
    ],
)
def test_bad_license_owner_family_future_links_and_dimension_fail_closed(
        tmp_path, mutate, message):
    """坏许可、owner、覆盖、未来引用和 evaluator 维均须停线。"""
    rows = _sample_values()
    mutate(rows)
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match=message):
        read_authored_open_set_clarification_seeds(path)


def test_missing_held_out_triple_recombination_fails_closed(tmp_path):
    """evaluator 若只重放 teacher 三轴，不得称组合 held-out。"""
    rows = _sample_values()
    source_kinds = {
        "BUDGET_STOP": ["SOURCE_REF"],
        "CLARIFICATION_ANSWER": ["CLARIFICATION_ANSWER"],
        "LOCAL_OBSERVATION": ["LOCAL_OBSERVATION"],
        "NO_ACCESS": ["SOURCE_REF"],
        "NONE": [],
        "SOURCE_LOOKUP": ["SOURCE_REF"],
        "UNKNOWN": ["UNKNOWN_SOURCE"],
    }
    for teacher, evaluator in zip(rows[:17], rows[17:], strict=True):
        path_kind = teacher["evidence_path"]
        evaluator["evidence_path"] = path_kind
        request = evaluator["evidence_request"]
        request["path_kind"] = path_kind
        request["required_source_kinds"] = source_kinds[path_kind]
        request["access_state"] = teacher["evidence_request"]["access_state"]
        request["budget_state"] = teacher["evidence_request"]["budget_state"]
        request["status"] = teacher["evidence_request"]["status"]
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match="held-out"):
        read_authored_open_set_clarification_seeds(path)


def test_payload_hash_scope_budget_reference_and_split_fail_closed():
    """坏 hash、跨 scope、超预算、悬空引用和伪组合轴均须停线。"""
    seed = next(item for item in read_authored_open_set_clarification_seeds(
        SAMPLE_PATH) if item.candidate_kind == "MINIMAL_CLARIFICATION")
    value = seed.observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match="SHA-256"):
        validate_open_set_clarification_payload(value)

    value = seed.observation_payload().to_value()
    value["novelty_profile"]["context_scope_key"] = "OTHER"
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="ContextScope"):
        validate_open_set_clarification_payload(value)

    value = seed.observation_payload().to_value()
    value["resource_budget"]["max_candidate_branches"] = 1
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match="超预算"):
        validate_open_set_clarification_payload(value)

    value = seed.observation_payload().to_value()
    value["candidate_branches"][0]["depends_on_obligation_id"] = "MISSING"
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match="依赖"):
        validate_open_set_clarification_payload(value)

    value = seed.observation_payload().to_value()
    value["split_identity"]["evidence_path"] = "SOURCE_LOOKUP"
    value["split_identity"]["combination_key"] = (
        "NEW_SENSE::MINIMAL_BRANCH_ELIMINATION::SOURCE_LOOKUP")
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="evidence_path"):
        validate_open_set_clarification_payload(value)


def test_obligation_plan_request_and_baseline_ablations_fail_closed():
    """删除补证链、破坏最小性或伪造负基线身份必须拒绝。"""
    by_kind = _first_by_kind(
        read_authored_open_set_clarification_seeds(SAMPLE_PATH))
    active = by_kind["ACTIVE_EVIDENCE_REQUEST"].observation_payload().to_value()
    active["evidence_request"]["candidate_ids"] = []
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="绑定 candidate"):
        validate_open_set_clarification_payload(active)

    minimal = by_kind["MINIMAL_CLARIFICATION"].observation_payload().to_value()
    minimal["clarification_plan"]["expected_eliminated_branches"] = 2
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match="严格子分支"):
        validate_open_set_clarification_payload(minimal)

    budget = by_kind["BUDGET_BLOCKED"].observation_payload().to_value()
    budget["missing_information_obligations"][0][
        "required_evidence_kind"] = "SOURCE_EVIDENCE"
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="required Evidence"):
        validate_open_set_clarification_payload(budget)

    over = by_kind["OVERQUESTION_BASELINE"].observation_payload().to_value()
    over["baseline_kind"] = "TYPED_OPEN_SET_OBJECTS_PRESENT"
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="FALSE 基线|身份错配"):
        validate_open_set_clarification_payload(over)


def test_revision_generation_and_unknown_ablations_fail_closed(tmp_path):
    """修正越界、生成漏保持和 unknown 猜测均不得通过。"""
    seeds = read_authored_open_set_clarification_seeds(SAMPLE_PATH)
    by_kind = _first_by_kind(seeds)
    generation = by_kind["GENERATION"].observation_payload().to_value()
    generation["generation_constraint"]["uncertainty_preservation_required"] = 0
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match="逐层保持"):
        validate_open_set_clarification_payload(generation)

    unknown = by_kind["UNKNOWN"].observation_payload().to_value()
    unknown["novelty_profile"]["detection_state"] = "NOVEL"
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="novelty detection"):
        validate_open_set_clarification_payload(unknown)

    rows = _sample_values()
    revision = rows[14]
    revision["candidate_branches"][2]["candidate_version"] = 2
    path = tmp_path / "bad-revision.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="非依赖|bit-identical"):
        read_authored_open_set_clarification_seeds(path)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    """float、非规范 JSON 和 pack 覆盖都必须停线。"""
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredOpenSetClarificationCourseError, match="损坏"):
        read_authored_open_set_clarification_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredOpenSetClarificationCourseError,
                       match="非规范 JSON|损坏"):
        read_authored_open_set_clarification_seeds(path)
    compile_authored_open_set_clarification_course(
        SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_open_set_clarification_course(
            SAMPLE_PATH, tmp_path / "release")


def test_formal_repository_course_manifest_and_external_pack_are_exact():
    """正式课程 manifest 和外置 pack 必须逐 hash 回读且保持零执行。"""
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
