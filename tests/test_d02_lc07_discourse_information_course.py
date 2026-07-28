"""D-02 LC-07 篇章关系、预设、信息结构与 QUD 课程 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_discourse_information_course import (
    COURSE_MANIFEST_PATH,
    EVALUATOR_DIMENSIONS,
    FORMAL_ARTIFACT_RELATIVE_ROOT,
    PACK_NAME,
    PAYLOAD_KIND,
    AuthoredDiscourseInformationCourseError,
    build_discourse_information_course_manifest,
    compile_authored_discourse_information_course,
    default_discourse_information_sample_bytes,
    read_authored_discourse_information_seeds,
    validate_discourse_information_payload,
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
    "data/ph2/authored_discourse_information_seed_v1.jsonl.sample")
FORMAL_COURSE_MANIFEST_SHA256 = (
    "791d8ee6f658fb0a7c3291ac6a9754fd1df414d27d99912c53be062f08a60ef5")


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
    """正式 sample 必须可重建，双 owner 各自覆盖完整候选族。"""
    assert SAMPLE_PATH.read_bytes() == default_discourse_information_sample_bytes()
    seeds = read_authored_discourse_information_seeds(SAMPLE_PATH)
    assert len(seeds) == 32
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    assert len(teacher) == len(evaluator) == 16
    expected_kinds = {
        "AMBIGUOUS_RELATION", "CAUSE", "CONCESSION", "CONTRAST",
        "ELABORATION", "GENERATION", "GIVEN_NEW",
        "NO_CONNECTIVE_BASELINE", "PRESUPPOSITION_CANCELLATION",
        "PRESUPPOSITION_PROJECTION", "QUD", "RETENTION", "REVISION",
        "TOPIC_FOCUS", "UNKNOWN", "WRONG_CONNECTIVE_BASELINE",
    }
    for owner in (teacher, evaluator):
        assert {item.sample_family for item in owner} == set(SAMPLE_FAMILIES)
        assert {item.candidate_kind for item in owner} == expected_kinds
        assert {item.baseline_kind for item in owner} == {
            "NO_CONNECTIVE_ONLY", "TYPED_DISCOURSE_OBJECTS_PRESENT",
            "WRONG_CONNECTIVE_ONLY",
        }
        negatives = tuple(item for item in owner if item.baseline_kind != (
            "TYPED_DISCOURSE_OBJECTS_PRESENT"))
        assert len(negatives) == 2
        assert all(item.expected_state == "FALSE" for item in negatives)
    assert {item.family for item in teacher}.isdisjoint(
        {item.family for item in evaluator})
    assert {item.template_family for item in teacher}.isdisjoint(
        {item.template_family for item in evaluator})
    assert {item.evaluation_dimension for item in evaluator} == set(
        EVALUATOR_DIMENSIONS)


def test_four_discourse_relations_are_direct_typed_candidates():
    """四类篇章关系必须由 typed 端点和候选状态承载。"""
    by_kind = _first_by_kind(
        read_authored_discourse_information_seeds(SAMPLE_PATH))
    for kind in ("ELABORATION", "CONTRAST", "CAUSE", "CONCESSION"):
        payload = by_kind[kind].observation_payload().to_value()
        relation = payload["discourse_relations"][0]
        proposition_ids = {
            item["proposition_id"] for item in payload["proposition_candidates"]}
        assert relation["relation_kind"] == kind
        assert relation["candidate_state"] == "SUPPORTED"
        assert relation["provisional"] == 1
        assert relation["source_proposition_id"] in proposition_ids
        assert relation["target_proposition_id"] in proposition_ids


def test_presupposition_information_structure_and_qud_are_direct():
    """预设、topic/focus、given/new 与 QUD 都必须有独立 typed 对象。"""
    by_kind = _first_by_kind(
        read_authored_discourse_information_seeds(SAMPLE_PATH))
    projected = by_kind[
        "PRESUPPOSITION_PROJECTION"].observation_payload().to_value()
    cancelled = by_kind[
        "PRESUPPOSITION_CANCELLATION"].observation_payload().to_value()
    assert projected["presupposition_obligations"][0]["status"] == "PROJECTED"
    assert cancelled["presupposition_obligations"][0]["status"] == "CANCELLED"
    assert projected["presupposition_obligations"][0][
        "dependency_relation_ids"]

    topic_focus = by_kind["TOPIC_FOCUS"].observation_payload().to_value()
    information = topic_focus["information_structure"][0]
    assert information["status_family"] == "TOPIC_FOCUS"
    assert information["topic_key"] and information["focus_key"]
    assert len(information["contrast_set_keys"]) == 2
    given_new = by_kind["GIVEN_NEW"].observation_payload().to_value()
    assert given_new["information_structure"][0]["status_family"] == "MIXED"
    qud = by_kind["QUD"].observation_payload().to_value()["qud_candidates"][0]
    assert qud["status"] == "OPEN"
    assert qud["provisional"] == 1


def test_negative_ambiguity_and_unknown_rows_do_not_guess():
    """两类表层负基线、歧义竞争和 unknown 必须 fail closed。"""
    by_kind = _first_by_kind(
        read_authored_discourse_information_seeds(SAMPLE_PATH))
    absent = by_kind["NO_CONNECTIVE_BASELINE"].observation_payload().to_value()
    assert absent["discourse_relations"] == []
    assert absent["baseline_kind"] == "NO_CONNECTIVE_ONLY"
    wrong = by_kind[
        "WRONG_CONNECTIVE_BASELINE"].observation_payload().to_value()
    assert wrong["discourse_relations"][0]["connective_state"] == "MISMATCH"
    assert wrong["discourse_relations"][0]["candidate_state"] == "REFUTED"
    ambiguous = by_kind["AMBIGUOUS_RELATION"].observation_payload().to_value()
    assert {item["candidate_state"]
            for item in ambiguous["discourse_relations"]} == {"COMPETING"}
    assert len({item["relation_kind"]
                for item in ambiguous["discourse_relations"]}) == 2
    unknown = by_kind["UNKNOWN"].observation_payload().to_value()
    assert unknown["discourse_relations"][0]["relation_kind"] == "UNKNOWN"
    assert unknown["information_structure"][0]["status_family"] == "UNKNOWN"
    assert unknown["qud_candidates"][0]["status"] == "UNKNOWN"


def test_revision_changes_only_relation_state_and_version():
    """后文修正不得替换 Proposition、预设、信息结构或 QUD。"""
    seeds = read_authored_discourse_information_seeds(SAMPLE_PATH)
    for revision in (item for item in seeds if item.candidate_kind == "REVISION"):
        target = next(item for item in seeds
                      if item.seed_id == revision.supersedes_seed_id)
        old = target.observation_payload().to_value()
        new = revision.observation_payload().to_value()
        for key in (
                "proposition_candidates", "presupposition_obligations",
                "information_structure", "qud_candidates"):
            assert old[key] == new[key]
        old_relation = old["discourse_relations"][0]
        new_relation = new["discourse_relations"][0]
        for key in (
                "connective_state", "context_scope_key", "provisional",
                "relation_id", "relation_kind", "source_proposition_id",
                "target_proposition_id"):
            assert old_relation[key] == new_relation[key]
        assert old_relation["candidate_state"] == "SUPPORTED"
        assert new_relation["candidate_state"] == "REFUTED"
        assert (old_relation["candidate_version"],
                new_relation["candidate_version"]) == (1, 2)
        assert new["revision_receipt"]["revision_scope"] == (
            "DISCOURSE_RELATION_ONLY")
        assert new["revision_receipt"]["raw_observation_preserved"] == 1


def test_retention_and_generation_are_read_only_and_hidden():
    """retention 只指旧锚；生成答案只存在于私有 label。"""
    seeds = read_authored_discourse_information_seeds(SAMPLE_PATH)
    for retention in (item for item in seeds if item.candidate_kind == "RETENTION"):
        anchor = next(item for item in seeds
                      if item.seed_id == retention.retention_anchor_id)
        assert anchor.logical_order < retention.logical_order
        assert anchor.split == retention.split
        assert anchor.candidate_kind == "CONCESSION"
    for generation in (item for item in seeds if item.candidate_kind == "GENERATION"):
        payload = generation.observation_payload().to_value()
        constraint = payload["generation_constraint"]
        assert payload["observed_surface"]["target_hidden"] == 1
        assert constraint["output_surface_hidden"] == 1
        preservation = {
            key: value for key, value in constraint.items()
            if key.endswith("_preservation_required")}
        assert preservation and set(preservation.values()) == {1}
        assert payload["presupposition_obligations"]
        for surface in generation.expected_payload.to_value()["accepted_surfaces"]:
            assert surface not in payload["observed_surface"]["text"]


def test_held_out_relation_information_qud_recombination_is_direct():
    """held-out 合取未见，但 relation、信息状态和 QUD 单轴均已见。"""
    seeds = read_authored_discourse_information_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    teacher_relations = {item.relation_family for item in teacher}
    teacher_information = {item.information_status_family for item in teacher}
    teacher_qud = {item.qud_family for item in teacher}
    teacher_triples = {
        (item.relation_family, item.information_status_family, item.qud_family)
        for item in teacher}
    recombinations = {
        (item.relation_family, item.information_status_family, item.qud_family)
        for item in evaluator
        if item.relation_family in teacher_relations
        and item.information_status_family in teacher_information
        and item.qud_family in teacher_qud
        and (item.relation_family, item.information_status_family,
             item.qud_family) not in teacher_triples
    }
    assert {
        ("CAUSE", "MIXED", "ANSWERED"),
        ("CONCESSION", "NEW", "ANSWERED"),
        ("CONTRAST", "GIVEN", "OPEN"),
        ("ELABORATION", "NEW", "ANSWERED"),
    }.issubset(recombinations)


def test_compiler_reuses_four_records_and_is_transport_deterministic(tmp_path):
    """编译只复用四 owner record，且两次 transport 字节一致。"""
    first = compile_authored_discourse_information_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_discourse_information_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.record_count == 96
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-08",)
    assert first.validation.source_ref_count == 32
    assert first.validation.observation_count == 32
    assert first.validation.teacher_evidence_count == 16
    assert first.validation.evaluator_label_count == 16
    assert first.validation.source_cluster_count == 30
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
        validate_discourse_information_payload(item.typed_payload)
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert "accepted_surfaces" not in payload
    assert len({item.dimension_key for item in evaluators}) == len(
        EVALUATOR_DIMENSIONS)


def test_course_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    """课程 manifest 必须不可覆盖、零执行并绑定两项能力。"""
    build = compile_authored_discourse_information_course(
        SAMPLE_PATH, tmp_path / "pack")
    manifest = build_discourse_information_course_manifest(SAMPLE_PATH, build)
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.task_keys == ("LC-07",)
    assert manifest.capability_keys == (
        "DISCOURSE_INFORMATION_STRUCTURE", "REFERENCE_DISCOURSE_REVISION")
    assert manifest.sample_relative_path == (
        "data/ph2/authored_discourse_information_seed_v1.jsonl.sample")
    assert manifest.pack_record_count == 96
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
        (lambda rows: rows[16].__setitem__("split", "train"), "split"),
        (lambda rows: rows.pop(5), "候选族|sample family"),
        (lambda rows: rows[13].__setitem__(
            "supersedes_seed_id", rows[15]["seed_id"]), "更早"),
        (lambda rows: rows[14].__setitem__(
            "retention_anchor_id", rows[15]["seed_id"]), "更早"),
        (lambda rows: rows[16].__setitem__(
            "evaluation_dimension", rows[17]["evaluation_dimension"]),
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
    with pytest.raises(AuthoredDiscourseInformationCourseError, match=message):
        read_authored_discourse_information_seeds(path)


def test_missing_held_out_triple_recombination_fails_closed(tmp_path):
    """evaluator 若只重放 teacher 完整合取，不能称组合 held-out。"""
    rows = _sample_values()
    for teacher, evaluator in zip(rows[:16], rows[16:], strict=True):
        status = teacher["information_status_family"]
        evaluator["information_status_family"] = status
        information = evaluator["information_structure"][0]
        token = information["information_id"].removesuffix("_INFORMATION_1")
        information["status_family"] = status
        information["topic_key"] = ""
        information["focus_key"] = ""
        information["contrast_set_keys"] = []
        if status == "GIVEN":
            information["topic_key"] = f"{token}_TOPIC"
        elif status == "NEW":
            information["focus_key"] = f"{token}_FOCUS"
        elif status == "MIXED":
            information["topic_key"] = f"{token}_TOPIC"
            information["focus_key"] = f"{token}_FOCUS"
        elif status == "TOPIC_FOCUS":
            information["topic_key"] = f"{token}_TOPIC"
            information["focus_key"] = f"{token}_FOCUS"
            information["contrast_set_keys"] = [
                f"{token}_ALTERNATIVE_A", f"{token}_ALTERNATIVE_B"]
        evaluator["qud_family"] = teacher["qud_family"]
        evaluator["qud_candidates"][0]["status"] = teacher["qud_family"]
    path = tmp_path / "bad.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="held-out"):
        read_authored_discourse_information_seeds(path)


def test_payload_hash_scope_budget_and_reference_fail_closed():
    """坏 hash、跨 scope、超预算和悬空端点都不得通过。"""
    seed = next(item for item in read_authored_discourse_information_seeds(
        SAMPLE_PATH) if item.candidate_kind == "ELABORATION")
    value = seed.observation_payload().to_value()
    value["observed_surface"]["text"] = "损坏"
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="SHA-256"):
        validate_discourse_information_payload(value)

    value = seed.observation_payload().to_value()
    value["proposition_candidates"][0]["context_scope_key"] = "OTHER"
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="ContextScope"):
        validate_discourse_information_payload(value)

    value = seed.observation_payload().to_value()
    value["resource_budget"]["max_propositions"] = 1
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="超预算"):
        validate_discourse_information_payload(value)

    value = seed.observation_payload().to_value()
    value["discourse_relations"][0]["target_proposition_id"] = "MISSING"
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="端点"):
        validate_discourse_information_payload(value)


def test_relation_baselines_information_qud_and_presupposition_ablations_fail():
    """删除承重 typed 层或伪造连接词状态必须被 validator 拒绝。"""
    by_kind = _first_by_kind(
        read_authored_discourse_information_seeds(SAMPLE_PATH))
    absent = by_kind["NO_CONNECTIVE_BASELINE"].observation_payload().to_value()
    absent["baseline_kind"] = "TYPED_DISCOURSE_OBJECTS_PRESENT"
    with pytest.raises(AuthoredDiscourseInformationCourseError,
                       match="无 connective|身份错配"):
        validate_discourse_information_payload(absent)

    wrong = by_kind[
        "WRONG_CONNECTIVE_BASELINE"].observation_payload().to_value()
    wrong["discourse_relations"][0]["candidate_state"] = "SUPPORTED"
    with pytest.raises(AuthoredDiscourseInformationCourseError,
                       match="错误 connective"):
        validate_discourse_information_payload(wrong)

    projected = by_kind[
        "PRESUPPOSITION_PROJECTION"].observation_payload().to_value()
    obligation_id = projected[
        "presupposition_obligations"][0]["obligation_id"]
    projected["presupposition_obligations"] = []
    projected["generation_constraint"]["input_candidate_ids"].remove(
        obligation_id)
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="预设投射"):
        validate_discourse_information_payload(projected)

    topic = by_kind["TOPIC_FOCUS"].observation_payload().to_value()
    topic["information_structure"][0]["contrast_set_keys"].pop()
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="对立集"):
        validate_discourse_information_payload(topic)

    qud = by_kind["QUD"].observation_payload().to_value()
    qud["qud_candidates"][0]["status"] = "ANSWERED"
    qud["split_identity"]["qud_family"] = "ANSWERED"
    qud["split_identity"]["combination_key"] = (
        f'{qud["split_identity"]["relation_family"]}::'
        f'{qud["split_identity"]["information_status_family"]}::ANSWERED')
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="开放 QUD"):
        validate_discourse_information_payload(qud)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("relation_family", "CONTRAST", "relation_family"),
        ("information_status_family", "NEW", "information family"),
        ("qud_family", "OPEN", "QUD family"),
    ],
)
def test_combination_axes_cannot_lie_about_typed_payload(field, value, message):
    """组合轴必须从 typed payload 派生，不能只改标签绕过。"""
    seed = next(item for item in read_authored_discourse_information_seeds(
        SAMPLE_PATH) if item.candidate_kind == "ELABORATION")
    payload = seed.observation_payload().to_value()
    payload["split_identity"][field] = value
    payload["split_identity"]["combination_key"] = (
        f'{payload["split_identity"]["relation_family"]}::'
        f'{payload["split_identity"]["information_status_family"]}::'
        f'{payload["split_identity"]["qud_family"]}')
    with pytest.raises(AuthoredDiscourseInformationCourseError, match=message):
        validate_discourse_information_payload(payload)


def test_revision_generation_and_unknown_ablations_fail_closed(tmp_path):
    """修正越界、生成漏保持和 unknown 猜测均不得通过。"""
    seeds = read_authored_discourse_information_seeds(SAMPLE_PATH)
    by_kind = _first_by_kind(seeds)
    generation = by_kind["GENERATION"].observation_payload().to_value()
    generation["generation_constraint"]["qud_preservation_required"] = 0
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="逐层保持"):
        validate_discourse_information_payload(generation)

    unknown = by_kind["UNKNOWN"].observation_payload().to_value()
    unknown["information_structure"][0]["focus_key"] = "GUESSED"
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="unknown"):
        validate_discourse_information_payload(unknown)

    rows = _sample_values()
    rows[13]["proposition_candidates"][0]["proposition_key"] = "REPLACED"
    path = tmp_path / "bad-revision.sample"
    _write_values(path, rows)
    with pytest.raises(AuthoredDiscourseInformationCourseError,
                       match="不得替换"):
        read_authored_discourse_information_seeds(path)


def test_noncanonical_float_and_existing_pack_fail_closed(tmp_path):
    """float、非规范 JSON 和 pack 覆盖都必须停线。"""
    path = tmp_path / "bad.sample"
    path.write_bytes(b'{"logical_order":1.5}\n')
    with pytest.raises(AuthoredDiscourseInformationCourseError, match="损坏"):
        read_authored_discourse_information_seeds(path)
    rows = _sample_values()
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with pytest.raises(AuthoredDiscourseInformationCourseError,
                       match="非规范 JSON|损坏"):
        read_authored_discourse_information_seeds(path)
    compile_authored_discourse_information_course(
        SAMPLE_PATH, tmp_path / "release")
    with pytest.raises(Exception, match="已存在"):
        compile_authored_discourse_information_course(
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
    assert restored.record_count == manifest.pack_record_count == 96
    assert restored.splits == manifest.pack_splits == ("train", "held_out")
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
