"""D-02E.1 AUTHORED_CC0_V1 discourse/revision 资料包 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.parser_revision import (
    ParserAnchorRevision,
    parser_lineage_key,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    ScopeIdentity,
)
from pure_integer_ai.experiments.ph2_authored_discourse_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_discourse_course,
    read_authored_discourse_seeds,
)
from pure_integer_ai.experiments.ph2_authored_discourse_schema import (
    VARIANT_KINDS,
    AuthoredDiscourseCourseError,
)
from pure_integer_ai.experiments.ph2_authored_nested_course import (
    compile_authored_nested_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    validate_stage_visibility,
)


SAMPLE_PATH = Path(
    "data/ph2/authored_discourse_revision_seed_v1.jsonl.sample")
NESTED_SAMPLE_PATH = Path(
    "data/ph2/authored_logic_nested_scope_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_discourse_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_discourse_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_discourse_course.py"),
)


def _sample_values() -> list[dict]:
    """读取仓库 sample 为独立可修改 object。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """按统一 canonical writer 写测试 JSONL。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _write_json_with_float(path: Path, values: list[dict]) -> None:
    """绕过合同 writer 写 float parser 负例。"""
    path.write_bytes(b"".join(
        (json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8")
        for row in values
    ))


def _records(build, kind: str):
    """读取 pack 内指定 record kind。"""
    out = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            out.extend(read_record_artifact(build.pack_root, identity))
    return tuple(out)


def _identity(value) -> ObjectIdentity:
    """从 payload 恢复一等对象身份。"""
    return ObjectIdentity.from_stable_key(tuple(value))


def test_sample_covers_nine_variants_reference_revision_and_isolation():
    """sample 覆盖九类、四 role、有界指代、局部 revision 和 owner 隔离。"""
    seeds = read_authored_discourse_seeds(SAMPLE_PATH)
    assert len(seeds) == 12
    assert {item.variant_kind for item in seeds} == VARIANT_KINDS
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    teacher_families = {
        item.family for item in seeds if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in seeds if item.label_owner == "evaluator"}
    assert teacher_families and evaluator_families
    assert teacher_families.isdisjoint(evaluator_families)
    assert any(item.reference_plan is not None for item in seeds)
    assert any(item.parser_revision is not None for item in seeds)


def test_later_reinterpretation_is_bounded_and_local_not_fifo():
    """后文修正显式替换候选并只列受影响 query，不靠裸 FIFO。"""
    seed = next(item for item in read_authored_discourse_seeds(SAMPLE_PATH)
                if item.variant_kind == "LATER_REINTERPRETATION")
    plan = seed.reference_plan
    revision = seed.parser_revision
    assert plan is not None and revision is not None
    assert plan.rejected_occurrence_id == "ship-a"
    assert plan.replacement_occurrence_id == "ship-b"
    assert set(plan.candidate_occurrence_ids) <= set(plan.window_occurrence_ids)
    assert plan.impacted_query_ids == revision.recompute_query_ids == (901,)
    assert revision.affected_occurrence_ids == ("ship-a",)
    assert revision.unaffected_occurrence_ids == ("ref-pronoun",)


def test_compiler_is_bit_identical_partitioned_and_expected_private(tmp_path):
    """两目录 bit-identical，12/12/9/3 分账且 expected 不进入 Observation。"""
    first = compile_authored_discourse_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_discourse_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.content_sha256() == (
        "420a98334a9fb9ce91e116bc991daaa544316cfbd19cf23fa3da942a35703c95")
    assert first.manifest.w_stages == ("W-08",)
    assert first.validation.source_ref_count == 12
    assert first.validation.observation_count == 12
    assert first.validation.teacher_evidence_count == 9
    assert first.validation.evaluator_label_count == 3
    assert first.validation.source_cluster_count == 2
    assert read_artifact_manifest(first.pack_root / "manifest.json") == (
        first.manifest)
    assert (first.pack_root / "manifest.json").read_bytes() == (
        second.pack_root / "manifest.json").read_bytes()
    for left, right in zip(first.manifest.files, second.manifest.files):
        assert left == right
        assert (first.pack_root / left.relative_path).read_bytes() == (
            second.pack_root / right.relative_path).read_bytes()
    observations = _records(first, RECORD_OBSERVATION)
    assert len(_records(first, RECORD_SOURCE_REF)) == 12
    assert len(observations) == 12
    assert len(_records(first, RECORD_TEACHER_EVIDENCE)) == 9
    assert len(_records(first, RECORD_EVALUATOR_LABEL)) == 3
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_payload_roundtrips_source_scope_span_and_occurrence_identity(tmp_path):
    """逐条恢复 SourceRef/document scope/Span/Occurrence，不以 surface 作身份。"""
    build = compile_authored_discourse_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        source = SourceRef.from_stable_key(tuple(payload["source_ref_key"]))
        scope = ScopeIdentity.from_stable_key(tuple(
            payload["document_scope_key"]))
        assert scope.scope_kind == SCOPE_DOCUMENT
        assert scope.source == source
        for item in payload["paragraph_spans"]:
            identity = _identity(item["span_key"])
            assert identity.object_kind == OBJECT_SPAN
            assert identity == span_identity(
                source,
                members=((item["start"], item["end"]),),
                ordinal=item["ordinal"],
            )
        for item in payload["occurrences"]:
            identity = _identity(item["occurrence_key"])
            assert identity.object_kind == OBJECT_OCCURRENCE
            assert identity == occurrence_identity(
                source,
                start=item["start"],
                end=item["end"],
                ordinal=item["ordinal"],
            )
        assert payload["surface_cue_authoritative"] == 0
        assert payload["bare_reference_fifo_authoritative"] == 0
        assert payload["old_occurrences_rewritten"] == 0
        assert payload["unaffected_recomputed"] == 0
        assert payload["whole_document_recomputed"] == 0


def test_reference_payload_preserves_explicit_window_candidates_and_budget(
        tmp_path):
    """reference 只消费声明窗口/候选，身份和预算可直接复核。"""
    build = compile_authored_discourse_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        plan = payload["reference_plan"]
        if plan is None:
            continue
        by_id = {
            item["occurrence_id"]: item["occurrence_key"]
            for item in payload["occurrences"]}
        assert plan["reference_occurrence_key"] == by_id[
            plan["reference_occurrence_id"]]
        assert plan["window_occurrence_keys"] == [
            by_id[item] for item in plan["window_occurrence_ids"]]
        assert plan["candidate_occurrence_keys"] == [
            by_id[item] for item in plan["candidate_occurrence_ids"]]
        assert set(plan["candidate_occurrence_ids"]) <= set(
            plan["window_occurrence_ids"])
        assert len(plan["candidate_occurrence_ids"]) <= (
            payload["consumer_request"]["max_candidates"])


def test_revision_payload_uses_cross_version_source_and_parser_anchor_revision(
        tmp_path):
    """revision 真实跨 ParserVersion，affected anchor 局部映射且 unaffected 不重算。"""
    build = compile_authored_discourse_course(SAMPLE_PATH, tmp_path)
    plans = [
        item.typed_payload.to_value()["parser_revision_plan"]
        for item in _records(build, RECORD_OBSERVATION)
        if item.typed_payload.to_value()["parser_revision_plan"] is not None
    ]
    assert len(plans) == 2
    for plan in plans:
        old_source = SourceRef.from_stable_key(tuple(plan["old_source_key"]))
        new_source = SourceRef.from_stable_key(tuple(plan["new_source_key"]))
        old_scope = ScopeIdentity.from_stable_key(tuple(
            plan["old_document_scope_key"]))
        new_scope = ScopeIdentity.from_stable_key(tuple(
            plan["new_document_scope_key"]))
        assert parser_lineage_key(old_source) == parser_lineage_key(new_source)
        assert old_source.versions.parser.value == plan["old_parser_version"]
        assert new_source.versions.parser.value == plan["new_parser_version"]
        assert old_source.versions.parser < new_source.versions.parser
        assert old_scope.source == old_source
        assert new_scope.source == new_source
        for item in plan["affected_occurrences"]:
            anchor = ParserAnchorRevision(
                _identity(item["old_occurrence_key"]),
                tuple(_identity(value)
                      for value in item["replacement_occurrence_keys"]),
            )
            assert list(anchor.stable_key()) == item["revision_key"]
        for item in plan["unaffected_occurrences"]:
            assert _identity(item["old_occurrence_key"]) != _identity(
                item["new_occurrence_key"])
        assert plan["recompute_query_ids"]


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """来源簇、owner、W-08 视图及同 family 修订链均可复核。"""
    build = compile_authored_discourse_course(SAMPLE_PATH, tmp_path)
    sources = _records(build, RECORD_SOURCE_REF)
    observations = _records(build, RECORD_OBSERVATION)
    teachers = _records(build, RECORD_TEACHER_EVIDENCE)
    evaluators = _records(build, RECORD_EVALUATOR_LABEL)
    source_index = {item.stable_key: item for item in sources}
    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(item for item in observations if item.split == "held_out")
    train_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in train}
    held_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in held_out}
    assert train_clusters.isdisjoint(held_clusters)
    assert {item.owner_key for item in teachers}.isdisjoint(
        {item.owner_key for item in evaluators})
    validate_stage_visibility(
        train, teachers, (), current_stage="W-08", view_kind="training")
    validate_stage_visibility(
        held_out, (), evaluators, current_stage="W-08", view_kind="evaluation")
    superseder = next(
        item for item in observations if item.sample_role == "supersede")
    target = next(
        item for item in observations
        if item.stable_key == superseder.supersedes_key)
    assert target.logical_order < superseder.logical_order
    assert target.split == superseder.split == "train"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].__setitem__("extra", 1), "字段集合"),
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"),
         "CC0-1.0"),
        (lambda rows: rows[1].__setitem__(
            "seed_id", rows[0]["seed_id"]), "重复"),
        (lambda rows: rows[10].__setitem__("split", "train"),
         "label_owner 与 split"),
        (lambda rows: rows[0].__setitem__("variant_kind", "OTHER"),
         "variant"),
        (lambda rows: rows[3].__setitem__("reference_plan", None),
         "reference variant"),
        (lambda rows: rows[5]["paragraphs"][1].__setitem__("ordinal", 0),
         "paragraph ordinal"),
        (lambda rows: rows[0]["occurrences"][0].__setitem__(
            "surface_fragment", "错舟"), "与 surface 不一致"),
        (lambda rows: rows[1]["consumer_request"].__setitem__(
            "max_context_chars", 2), "context 字符预算"),
        (lambda rows: rows[3]["consumer_request"].__setitem__(
            "max_candidates", 0), "正严格整数"),
        (lambda rows: rows[3]["reference_plan"][
            "window_occurrence_ids"].append("unknown"), "未知 occurrence"),
        (lambda rows: rows[3]["reference_plan"].__setitem__(
            "window_occurrence_ids", ["ship-a", "ref-pronoun"]),
         "自身不得进入 window"),
        (lambda rows: rows[7]["parser_revision"].__setitem__(
            "new_parser_version", 1), "严格更高"),
        (lambda rows: rows[7]["parser_revision"].__setitem__(
            "unaffected_occurrence_ids", []), "affected/unaffected"),
        (lambda rows: rows[7]["parser_revision"].__setitem__(
            "affected_occurrence_ids", ["old-anchor", "new-anchor"]),
         "完整覆盖 affected"),
        (lambda rows: rows[7]["parser_revision"].__setitem__(
            "unaffected_occurrence_ids", ["new-anchor"]),
         "replacement 不得冒充"),
        (lambda rows: rows[8]["reference_plan"].__setitem__(
            "replacement_occurrence_id", ""), "必须同时声明"),
        (lambda rows: rows[7]["consumer_request"].__setitem__(
            "max_recompute_queries", 0), "正严格整数"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[10]["seed_id"]), "更早"),
        (lambda rows: (
            rows[11].__setitem__("sample_role", "supersede"),
            rows[11].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[11].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "跨 family/split"),
        (lambda rows: (
            rows[6].__setitem__("perturbation_kind", "NONE"),
            rows[11].__setitem__("perturbation_kind", "NONE"),
        ),
         "缺少必需反向破坏"),
    ],
)
def test_bad_license_split_span_reference_revision_budget_and_chain_fail_closed(
        tmp_path, mutate, message):
    """坏许可/split/span/reference/revision/预算/恢复链均不能进 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredDiscourseCourseError, match=message):
        read_authored_discourse_seeds(bad)


def test_nested_manifest_does_not_drift(tmp_path):
    """新增 discourse 合同不得改变已关闭 nested artifact 身份。"""
    nested = compile_authored_nested_course(
        NESTED_SAMPLE_PATH, tmp_path / "nested")
    assert nested.manifest.content_sha256() == (
        "1908cacff3ad11398598dfa2248f99b8ac01d312d6737ed6a4a9f46aa6b7d0f7")


def test_sample_hash_float_noncanonical_existing_pack_and_cues_fail_closed(
        tmp_path):
    """钉住 sample；float/非规范/覆盖失败，源码不搬 cue/expected/训练状态。"""
    assert hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest() == (
        "cf8092eb5824b3b0c7d39c3ddaa2dec1855894d64c2d8fbec93dc57ff95eb9a8")
    rows = _sample_values()
    rows[0]["consumer_request"]["max_candidates"] = 1.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredDiscourseCourseError, match="规范 JSON"):
        read_authored_discourse_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"consumer_request": {}}\n')
    with pytest.raises(AuthoredDiscourseCourseError, match="规范 JSON"):
        read_authored_discourse_seeds(bad_json)
    build = compile_authored_discourse_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredDiscourseCourseError, match="发布失败"):
        compile_authored_discourse_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "surface.startswith",
            "expected_state ==",
            "teacher_payload",
            "mastered =",
            "window_occurrences[-1]"}:
        assert token not in source
