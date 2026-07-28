"""D-02E.2 AUTHORED_CC0_V1 typed question-answer 资料包 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_QUERY,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.ph2_authored_discourse_course import (
    compile_authored_discourse_course,
)
from pure_integer_ai.experiments.ph2_authored_qa_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_qa_course,
    read_authored_qa_seeds,
)
from pure_integer_ai.experiments.ph2_authored_qa_schema import (
    QUESTION_KINDS,
    AuthoredQACourseError,
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


SAMPLE_PATH = Path("data/ph2/authored_question_answer_seed_v1.jsonl.sample")
DISCOURSE_SAMPLE_PATH = Path(
    "data/ph2/authored_discourse_revision_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_qa_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_qa_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_qa_course.py"),
)


def _sample_values() -> list[dict]:
    """读取仓库 sample 为独立可修改 JSON object。"""
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


def _bound(value: dict) -> BoundProposition:
    """恢复本包无 Binder/Role 的 target/candidate BoundProposition。"""
    assert value["bindings"] == []
    return BoundProposition(
        _identity(value["template_key"]),
        _identity(value["instruction_key"]),
        _identity(value["predicate_key"]),
        _identity(value["structure_key"]),
        _identity(value["source_anchor_key"]),
        _identity(value["context_key"]),
        tuple(_identity(item) for item in value["introduced_binder_keys"]),
        (),
        tuple(_identity(item) for item in value["applied_variable_keys"]),
    )


def _selection_class(payload: dict) -> str:
    """只按 route、reference singleton、Evidence 和 competition 独立判立场。"""
    if payload["route_registered"] == 0:
        return "REFUSE"
    reference = payload["reference_resolution"]
    if reference is not None and reference["winner_occurrence_key"] is None:
        return "CLARIFY" if len(
            reference["adopted_occurrence_keys"]) > 1 else "UNKNOWN"
    required = payload["question_request"]["required_state"]
    candidates = payload["candidate_propositions"]
    if any(item["state"] == {"refute": 1, "support": 1}
           for item in candidates):
        return "CONFLICT"
    eligible = [
        item for item in candidates
        if ((not required["support"] or item["state"]["support"])
            and (not required["refute"] or item["state"]["refute"]))
    ]
    grouped = {}
    for item in eligible:
        grouped.setdefault(tuple(item["competition_key"]), set()).add(
            tuple(item["candidate_key"]))
    if any(len(values) > 1 for values in grouped.values()):
        return "CLARIFY"
    if len(eligible) == 1 and eligible[0]["matches_request_target"] == 1:
        return "ANSWER"
    return "UNKNOWN"


def test_sample_covers_fact_relation_logic_reference_and_nonanswer_modes():
    """sample 覆盖八类问题、五种 selection 结果、四 role 和隔离扰动。"""
    seeds = read_authored_qa_seeds(SAMPLE_PATH)
    assert len(seeds) == 12
    assert {item.question_kind for item in seeds} == QUESTION_KINDS
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    assert {item.route_status for item in seeds} == {
        "REGISTERED", "UNSUPPORTED"}
    assert any(item.reference_resolution is not None for item in seeds)
    assert any(not item.candidates for item in seeds)


def test_compiler_is_bit_identical_partitioned_and_expected_private(tmp_path):
    """两目录 bit-identical，12/12/9/3 分账且 expected 不进入 Observation。"""
    first = compile_authored_qa_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_qa_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.content_sha256() == (
        "2785a4e55474ae90ba20828410fde4dc69faca1e861c78c77aff8cf6de55ad62")
    assert first.manifest.w_stages == ("W-09",)
    assert first.validation.source_ref_count == 12
    assert first.validation.observation_count == 12
    assert first.validation.teacher_evidence_count == 9
    assert first.validation.evaluator_label_count == 3
    assert first.validation.source_cluster_count == 2
    assert read_artifact_manifest(first.pack_root / "manifest.json") == (
        first.manifest)
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


def test_question_request_target_scope_and_candidate_roundtrip(tmp_path):
    """逐条恢复真实 QuestionRequest、query scopes 和候选 BoundProposition。"""
    build = compile_authored_qa_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        source = SourceRef.from_stable_key(tuple(payload["source_ref_key"]))
        value = payload["question_request"]
        target = _bound(value["target"])
        evidence_scope = ScopeIdentity.from_stable_key(tuple(
            value["evidence_scope_key"]))
        response_scope = ScopeIdentity.from_stable_key(tuple(
            value["response_scope_key"]))
        request = QuestionRequest(
            _identity(value["query_kind_key"]),
            _identity(value["intent_key"]),
            _identity(value["goal_kind_key"]),
            target,
            LogicEvidenceState(
                bool(value["required_state"]["support"]),
                bool(value["required_state"]["refute"]),
            ),
            evidence_scope,
            response_scope,
            tuple(value["trace"]),
        )
        assert list(request.stable_key()) == value["request_key"]
        assert request.source == source
        assert evidence_scope.scope_kind == SCOPE_QUERY
        assert response_scope.scope_kind == SCOPE_QUERY
        assert evidence_scope != response_scope
        assert evidence_scope.parent == response_scope.parent
        for item in payload["candidate_propositions"]:
            candidate = _bound(item["bound_proposition"])
            assert list(candidate.stable_key()) == item["candidate_key"]
            if item["matches_request_target"]:
                assert candidate == target


def test_independent_selection_oracle_covers_answer_unknown_clarify_refuse_conflict(
        tmp_path):
    """不读 expected/teacher，只按 typed 采用依据得到五种结果。"""
    build = compile_authored_qa_course(SAMPLE_PATH, tmp_path)
    classes = {
        _selection_class(item.typed_payload.to_value())
        for item in _records(build, RECORD_OBSERVATION)
    }
    assert classes == {"ANSWER", "UNKNOWN", "CLARIFY", "REFUSE", "CONFLICT"}
    by_kind = {
        payload["question_kind"]: _selection_class(payload)
        for payload in (
            item.typed_payload.to_value()
            for item in _records(build, RECORD_OBSERVATION))
        if payload["question_kind"] not in {"EXPLICIT_FACT", "REFERENCE_SCOPE"}
    }
    assert by_kind["FINITE_LOGIC"] == "ANSWER"
    assert by_kind["UNKNOWN"] == "UNKNOWN"
    assert by_kind["AMBIGUOUS"] == "CLARIFY"
    assert by_kind["UNSUPPORTED"] == "REFUSE"
    assert by_kind["CONFLICT"] == "CONFLICT"


def test_reference_resolution_requires_singleton_adopted_winner_without_tiebreak(
        tmp_path):
    """A-01 winner 仅来自 singleton adopted，不按稳定序或 surface 私选。"""
    build = compile_authored_qa_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        resolution = payload["reference_resolution"]
        if resolution is None:
            continue
        source = SourceRef.from_stable_key(tuple(payload["source_ref_key"]))
        assert resolution["singleton_winner_required"] == 1
        assert resolution["stable_order_tiebreaker"] == 0
        assert len(resolution["adopted_occurrence_keys"]) == 1
        assert resolution["winner_occurrence_key"] == (
            resolution["adopted_occurrence_keys"][0])
        assert resolution["winner_occurrence_key"] in (
            resolution["candidate_occurrence_keys"])
        assert _identity(resolution["winner_occurrence_key"]) == (
            occurrence_identity(
                source,
                start=_identity(resolution["winner_occurrence_key"]).components[-3],
                end=_identity(resolution["winner_occurrence_key"]).components[-2],
                ordinal=_identity(
                    resolution["winner_occurrence_key"]).components[-1],
            ))


def test_selection_basis_forbids_expected_teacher_fixture_surface_and_nonempty(
        tmp_path):
    """采用依据只认 required/Evidence/reference，不认 expected、teacher 或非空输出。"""
    build = compile_authored_qa_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        assert payload["fixture_authoritative"] == 0
        assert payload["nonempty_output_authoritative"] == 0
        assert payload["surface_cue_authoritative"] == 0
        assert payload["selection_basis"] == {
            "candidate_evidence_state_authoritative": 1,
            "conflict_precedes_answer": 1,
            "expected_authoritative": 0,
            "required_direction_authoritative": 1,
            "teacher_authoritative": 0,
            "unique_reference_winner_required": 1,
        }


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """来源簇、owner、W-09 视图及同 family reference 修订可复核。"""
    build = compile_authored_qa_course(SAMPLE_PATH, tmp_path)
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
        train, teachers, (), current_stage="W-09", view_kind="training")
    validate_stage_visibility(
        held_out, (), evaluators, current_stage="W-09", view_kind="evaluation")
    superseder = next(
        item for item in observations if item.sample_role == "supersede")
    target = next(item for item in observations
                  if item.stable_key == superseder.supersedes_key)
    assert target.logical_order < superseder.logical_order
    assert target.split == superseder.split == "train"


def _unknown_candidate() -> dict:
    """构造会使 UNKNOWN seed 非法的可决候选。"""
    return {
        "candidate_id": "known",
        "competition_id": 5,
        "end": 5,
        "evidence_ids": [599],
        "evidence_refute": 0,
        "evidence_support": 1,
        "ordinal": 0,
        "predicate_kind": 5,
        "proposition_local_id": 501,
        "start": 0,
        "surface_fragment": "远处有声响",
    }


def _unsupported_candidate() -> dict:
    """构造坐标有效但 route 未注册时禁止出现的候选。"""
    return {
        "candidate_id": "unsupported-candidate",
        "competition_id": 7,
        "end": 6,
        "evidence_ids": [799],
        "evidence_refute": 0,
        "evidence_support": 1,
        "ordinal": 0,
        "predicate_kind": 7,
        "proposition_local_id": 701,
        "start": 1,
        "surface_fragment": "未注册类型",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].__setitem__("extra", 1), "字段集合"),
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"),
         "CC0-1.0"),
        (lambda rows: rows[1].__setitem__("seed_id", rows[0]["seed_id"]),
         "重复"),
        (lambda rows: rows[10].__setitem__("split", "train"),
         "label_owner 与 split"),
        (lambda rows: rows[0].__setitem__("question_kind", "OTHER"),
         "question kind"),
        (lambda rows: rows[0].__setitem__("route_status", "OTHER"),
         "route status"),
        (lambda rows: (
            rows[0].__setitem__("required_support", 0),
            rows[0].__setitem__("required_refute", 0),
        ), "至少声明"),
        (lambda rows: rows[0].__setitem__(
            "response_scope_local_id", rows[0]["query_scope_local_id"]),
         "不得混用"),
        (lambda rows: rows[0]["target"].__setitem__(
            "surface_fragment", "灯灭着"), "与 context 不一致"),
        (lambda rows: rows[5]["candidates"][1].__setitem__(
            "candidate_id", rows[5]["candidates"][0]["candidate_id"]),
         "candidate_id 重复"),
        (lambda rows: rows[5]["candidates"][1].__setitem__("ordinal", 0),
         "candidate ordinal"),
        (lambda rows: rows[0]["candidates"][0].__setitem__(
            "evidence_ids", [101, 101]), "evidence_ids 非法或重复"),
        (lambda rows: rows[0]["candidates"][0].__setitem__(
            "evidence_ids", []), "四态可用性"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "max_context_chars", 2), "context 超过字符预算"),
        (lambda rows: rows[5]["consumer_request"].__setitem__(
            "max_candidates", 1), "candidates 超过预算"),
        (lambda rows: rows[2]["consumer_request"].__setitem__(
            "max_evidence_items", 1), "Evidence 超过预算"),
        (lambda rows: rows[3].__setitem__("reference_resolution", None),
         "reference question"),
        (lambda rows: rows[3]["reference_resolution"][
            "candidate_occurrence_ids"].append("unknown"),
         "未知 occurrence"),
        (lambda rows: (
            rows[3]["reference_resolution"].__setitem__(
                "candidate_occurrence_ids", ["ref-pronoun"]),
            rows[3]["reference_resolution"].__setitem__(
                "adopted_occurrence_ids", ["ref-pronoun"]),
            rows[3]["reference_resolution"].__setitem__(
                "winner_occurrence_id", "ref-pronoun"),
        ),
         "自身不得成为 antecedent"),
        (lambda rows: rows[3]["reference_resolution"].__setitem__(
            "winner_occurrence_id", "ref-pronoun"), "同一 winner"),
        (lambda rows: rows[6].__setitem__(
            "candidates", [_unsupported_candidate()]), "unsupported QA"),
        (lambda rows: rows[5]["candidates"][1].__setitem__(
            "competition_id", 7), "同 competition 多命题"),
        (lambda rows: rows[7]["candidates"][0].__setitem__(
            "evidence_refute", 0), "CONFLICT QA"),
        (lambda rows: rows[4]["candidates"].append(_unknown_candidate()),
         "UNKNOWN QA"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[10]["seed_id"]), "更早"),
        (lambda rows: (
            rows[11].__setitem__("sample_role", "supersede"),
            rows[11].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[11].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "跨 family/split"),
        (lambda rows: (
            rows[7].__setitem__("perturbation_kind", "NONE"),
            rows[11].__setitem__("perturbation_kind", "NONE"),
        ), "缺少必需反向破坏"),
    ],
)
def test_bad_license_split_target_candidate_reference_route_budget_fail_closed(
        tmp_path, mutate, message):
    """坏 owner/target/candidate/reference/route/预算/恢复链均不能进 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredQACourseError, match=message):
        read_authored_qa_seeds(bad)


def test_discourse_manifest_does_not_drift(tmp_path):
    """新增 QA 合同不得改变已关闭 discourse artifact 身份。"""
    discourse = compile_authored_discourse_course(
        DISCOURSE_SAMPLE_PATH, tmp_path / "discourse")
    assert discourse.manifest.content_sha256() == (
        "420a98334a9fb9ce91e116bc991daaa544316cfbd19cf23fa3da942a35703c95")


def test_sample_hash_float_noncanonical_existing_pack_and_shortcuts_fail_closed(
        tmp_path):
    """钉住 sample；float/非规范/覆盖失败，源码不搬 expected/surface/排序捷径。"""
    assert hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest() == (
        "2a8ee92c16b4157679ce9e373dbf7c73b35bfd58fc1f843f15852c4a48821770")
    rows = _sample_values()
    rows[0]["consumer_request"]["max_candidates"] = 1.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredQACourseError, match="规范 JSON"):
        read_authored_qa_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"candidates": []}\n')
    with pytest.raises(AuthoredQACourseError, match="规范 JSON"):
        read_authored_qa_seeds(bad_json)
    build = compile_authored_qa_course(SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredQACourseError, match="发布失败"):
        compile_authored_qa_course(SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "expected_state ==",
            "surface.startswith",
            "winner = sorted",
            "if output",
            "fixture_answer"}:
        assert token not in source
