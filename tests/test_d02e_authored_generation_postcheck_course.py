"""D-02E.3 AUTHORED_CC0_V1 generation adoption/postcheck 资料包 T0。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentProtocol,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSourceRequirement,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_QUERY,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.ph2_authored_generation_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_generation_course,
    read_authored_generation_seeds,
)
from pure_integer_ai.experiments.ph2_authored_generation_schema import (
    GENERATION_CASES,
    GENERATION_STANCES,
    AuthoredGenerationCourseError,
)
from pure_integer_ai.experiments.ph2_authored_qa_course import (
    compile_authored_qa_course,
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
    "data/ph2/authored_generation_postcheck_seed_v1.jsonl.sample")
QA_SAMPLE_PATH = Path(
    "data/ph2/authored_question_answer_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_generation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_generation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_generation_course.py"),
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
    """恢复本包无 Binder/Role 的候选 BoundProposition。"""
    assert value["bindings"] == []
    return BoundProposition(
        _identity(value["template_key"]),
        _identity(value["instruction_key"]),
        _identity(value["predicate_key"]),
        _identity(value["structure_key"]),
        _identity(value["source_anchor_key"]),
        _identity(value["context_key"]),
        (),
        (),
        (),
    )


def _protocol(value: list[int]) -> AnswerContentProtocol:
    """从五个 packed MinimalInstruction 恢复 AnswerContentProtocol。"""
    cursor = 0
    identities = []
    for _ in range(5):
        size = value[cursor]
        cursor += 1
        identities.append(_identity(value[cursor:cursor + size]))
        cursor += size
    assert cursor == len(value)
    return AnswerContentProtocol(*identities)


def _adoption_class(payload: dict) -> str:
    """只按候选 Evidence 四态独立得到五种 adoption 立场。"""
    candidates = payload["candidate_propositions"]
    if any(item["state"] == {"refute": 1, "support": 1}
           for item in candidates):
        return "CONFLICT"
    decisive = [
        item for item in candidates
        if item["state"]["support"] or item["state"]["refute"]]
    if len(decisive) > 1:
        return "CLARIFY"
    if len(decisive) == 1:
        return "ANSWER"
    return "UNKNOWN" if candidates else "REFUSE"


def _postcheck_class(payload: dict) -> str:
    """只按 actual evidence source、citation/trust 和 source match 判 G-04。"""
    postcheck = payload["postcheck"]
    if not postcheck["enabled"]:
        return "NOT_RUN"
    for item in postcheck["requirements"]:
        requirement = item["requirement"]
        evidence = {tuple(value)
                    for value in requirement["evidence_source_keys"]}
        cited = {tuple(value) for value in item["cited_source_keys"]}
        trusted = {tuple(value) for value in item["trusted_source_keys"]}
        refuted = {tuple(value) for value in item["refuted_source_keys"]}
        if not item["source_match"]:
            return "SOURCE_FAIL"
        if requirement["citation_required"] and not evidence <= cited:
            return "CITATION_FAIL"
        if (requirement["trust_required"]
                and (not evidence <= trusted or evidence & refuted)):
            return "TRUST_FAIL"
    return "PASS"


def test_sample_covers_adoption_postcheck_cases_stances_roles_and_perturbations():
    """sample 覆盖九类 case、五 stance、四 role、citation/trust/source 失败。"""
    seeds = read_authored_generation_seeds(SAMPLE_PATH)
    assert len(seeds) == 12
    assert {item.generation_case for item in seeds} == GENERATION_CASES
    assert {item.stance for item in seeds} == GENERATION_STANCES
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    assert any(item.postcheck_enabled for item in seeds)
    assert any(not item.postcheck_enabled for item in seeds)


def test_compiler_is_bit_identical_partitioned_and_expected_private(tmp_path):
    """两目录 bit-identical，12/12/9/3 分账且 adoption expected 不泄漏。"""
    first = compile_authored_generation_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_generation_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.content_sha256() == (
        "15fe0e47834cf23c27076f7b18dd74199e0a90f584213614eb0f201ee69adc2a")
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
        assert "generation_case" not in payload
        if payload["task_kind"] == "ADOPTION":
            assert payload["postcheck"]["prior_adoption"] is None


def test_candidate_bound_owner_source_scope_and_protocol_roundtrip(tmp_path):
    """恢复候选 BoundProposition、owner SourceRef/query scope 和五 stance protocol。"""
    build = compile_authored_generation_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        source = SourceRef.from_stable_key(tuple(payload["source_ref_key"]))
        scope = ScopeIdentity.from_stable_key(tuple(
            payload["response_scope_key"]))
        assert scope.scope_kind == SCOPE_QUERY
        assert scope.source == source
        protocol = _protocol(payload["adoption_request"]["protocol_key"])
        assert len(protocol.stances()) == 5
        candidate_keys = []
        for item in payload["candidate_propositions"]:
            candidate = _bound(item["bound_proposition"])
            assert list(candidate.stable_key()) == item["candidate_key"]
            assert item["owner_source_key"] == payload["source_ref_key"]
            assert item["owner_scope_key"] == payload["response_scope_key"]
            assert len(item["evidence_source_ids"]) == len(
                item["evidence_source_keys"])
            candidate_keys.append(item["candidate_key"])
        assert candidate_keys == payload["adoption_request"]["candidate_keys"]


def test_adoption_oracle_covers_five_stances_without_expected_or_prior_decision(
        tmp_path):
    """ADOPTION 只按候选四态得到五种立场，Observation 不携带先验选择。"""
    build = compile_authored_generation_course(SAMPLE_PATH, tmp_path)
    adoption = [
        item.typed_payload.to_value()
        for item in _records(build, RECORD_OBSERVATION)
        if item.typed_payload.to_value()["task_kind"] == "ADOPTION"
    ]
    assert {_adoption_class(item) for item in adoption} == GENERATION_STANCES
    for payload in adoption:
        assert payload["postcheck"]["prior_adoption"] is None
        assert payload["postcheck"]["requirements"] == []
        assert "stance" not in payload["adoption_request"]
        assert "selected_candidate_ids" not in payload["adoption_request"]


def test_postcheck_oracle_covers_pass_and_three_independent_failure_dimensions(
        tmp_path):
    """G-04 独立核验实际 Evidence source、citation、trust 和 source 归属。"""
    build = compile_authored_generation_course(SAMPLE_PATH, tmp_path)
    postchecks = [
        item.typed_payload.to_value()
        for item in _records(build, RECORD_OBSERVATION)
        if item.typed_payload.to_value()["task_kind"] == "POSTCHECK"
    ]
    assert {_postcheck_class(item) for item in postchecks} == {
        "PASS", "CITATION_FAIL", "TRUST_FAIL", "SOURCE_FAIL"}
    assert all(item["postcheck"]["renderer_complete"] == 1
               for item in postchecks)
    assert all(item["postcheck"]["surface_units"] > 0
               for item in postchecks)


def test_prior_adoption_source_requirement_and_same_run_keys_roundtrip(tmp_path):
    """恢复真实 AnswerContentDecision/GenerationSourceRequirement 并钉住同次 key。"""
    build = compile_authored_generation_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        postcheck = payload["postcheck"]
        assert postcheck["same_run_key"] == (
            payload["adoption_request"]["same_run_key"])
        if not postcheck["enabled"]:
            continue
        protocol = _protocol(payload["adoption_request"]["protocol_key"])
        prior = postcheck["prior_adoption"]
        assert prior is not None
        decision = AnswerContentDecision(
            _identity(prior["stance_key"]),
            _identity(prior["reason_key"]),
            tuple(tuple(item) for item in prior["selected_candidate_keys"]),
            (),
            tuple(prior["trace"]),
        )
        assert decision.stance in protocol.stances()
        assert tuple(postcheck["same_run_key"]) == tuple(
            decision.trace[1:1 + len(postcheck["same_run_key"])])
        for item in postcheck["requirements"]:
            value = item["requirement"]
            requirement = GenerationSourceRequirement(
                tuple(value["candidate_key"]),
                SourceRef.from_stable_key(tuple(value["source_key"])),
                ScopeIdentity.from_stable_key(tuple(value["scope_key"])),
                bool(value["citation_required"]),
                bool(value["trust_required"]),
                tuple(value["trace"]),
                tuple(SourceRef.from_stable_key(tuple(source))
                      for source in value["evidence_source_keys"]),
            )
            assert list(requirement.stable_key()) == value["requirement_key"]
            assert tuple(postcheck["same_run_key"]) == tuple(
                requirement.trace[1:1 + len(postcheck["same_run_key"])])


def test_shortcut_and_memory_flags_are_zero(tmp_path):
    """非空 surface/renderer/expected/teacher/fixture 不判通过，Memory 正式关闭。"""
    build = compile_authored_generation_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        assert payload["expected_authoritative"] == 0
        assert payload["fixture_authoritative"] == 0
        assert payload["teacher_authoritative"] == 0
        assert payload["surface_cue_authoritative"] == 0
        assert payload["nonempty_surface_authoritative"] == 0
        assert payload["renderer_success_is_postcheck_pass"] == 0
        assert payload["memory_evidence_enabled"] == 0
        assert payload["memory_commit_enabled"] == 0


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """来源簇、owner、W-09 视图及同 family postcheck 修订可复核。"""
    build = compile_authored_generation_course(SAMPLE_PATH, tmp_path)
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


def _valid_refuse_candidate() -> dict:
    """构造坐标有效但 REFUSE case 禁止出现的候选。"""
    return {
        "candidate_id": "forbidden",
        "end": 4,
        "evidence_refute": 0,
        "evidence_source_ids": [41],
        "evidence_support": 1,
        "ordinal": 0,
        "predicate_kind": 4,
        "proposition_local_id": 401,
        "start": 0,
        "surface_fragment": "请求越权",
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
        (lambda rows: rows[0].__setitem__("generation_case", "OTHER"),
         "case"),
        (lambda rows: rows[0].__setitem__("stance", "OTHER"), "stance"),
        (lambda rows: rows[0].__setitem__(
            "selected_candidate_ids", ["unknown"]), "不属于请求"),
        (lambda rows: rows[2]["candidates"][1].__setitem__(
            "candidate_id", rows[2]["candidates"][0]["candidate_id"]),
         "candidate_id 重复"),
        (lambda rows: rows[2]["candidates"][1].__setitem__("ordinal", 0),
         "ordinal"),
        (lambda rows: rows[0]["candidates"][0].__setitem__(
            "surface_fragment", "灯灭着"), "与 context 不一致"),
        (lambda rows: rows[0]["candidates"][0].__setitem__(
            "evidence_source_ids", [1, 1]), "evidence_source_ids 非法或重复"),
        (lambda rows: rows[0].__setitem__("selected_candidate_ids", []),
         "ANSWER 必须采用唯一候选"),
        (lambda rows: rows[1].__setitem__(
            "selected_candidate_ids", ["unknown"]), "非 ANSWER stance"),
        (lambda rows: rows[5].__setitem__("surface_units", 0),
         "surface units 与 renderer"),
        (lambda rows: rows[5].__setitem__("renderer_complete", 0),
         "surface units 与 renderer"),
        (lambda rows: rows[5].__setitem__("source_requirements", []),
         "精确覆盖"),
        (lambda rows: rows[5]["source_requirements"][0].__setitem__(
            "candidate_id", "unknown"), "精确覆盖"),
        (lambda rows: rows[5]["source_requirements"][0][
            "cited_source_ids"].append(999), "越过 Evidence 来源"),
        (lambda rows: rows[7]["source_requirements"][0].__setitem__(
            "trusted_source_ids", [81,82]), "同时 trusted/refuted"),
        (lambda rows: (
            rows[5]["source_requirements"][0].__setitem__(
                "citation_required", 0),
            rows[5]["source_requirements"][0].__setitem__(
                "trust_required", 0),
        ), "至少要求"),
        (lambda rows: rows[2]["consumer_request"].__setitem__(
            "max_candidates", 1), "candidates 超过预算"),
        (lambda rows: rows[5]["consumer_request"].__setitem__(
            "max_evidence_sources", 1), "Evidence sources 超过预算"),
        (lambda rows: rows[5]["consumer_request"].__setitem__(
            "max_surface_units", 3), "surface units 超过预算"),
        (lambda rows: rows[5]["consumer_request"].__setitem__(
            "max_postcheck_checks", 0), "正严格整数"),
        (lambda rows: rows[1]["candidates"][0].__setitem__(
            "evidence_support", 1), "ADOPTION_UNKNOWN"),
        (lambda rows: rows[2].__setitem__(
            "candidates", [rows[2]["candidates"][0]]),
         "ADOPTION_AMBIGUOUS"),
        (lambda rows: rows[3].__setitem__(
            "candidates", [_valid_refuse_candidate()]), "ADOPTION_REFUSE"),
        (lambda rows: rows[4]["candidates"][0].__setitem__(
            "evidence_refute", 0), "ADOPTION_CONFLICT"),
        (lambda rows: rows[5]["source_requirements"][0].__setitem__(
            "cited_source_ids", [61]), "POSTCHECK_PASS"),
        (lambda rows: rows[6]["source_requirements"][0].__setitem__(
            "cited_source_ids", [71,72]), "citation fail case"),
        (lambda rows: (
            rows[7]["source_requirements"][0].__setitem__(
                "trusted_source_ids", [81,82]),
            rows[7]["source_requirements"][0].__setitem__(
                "refuted_source_ids", []),
        ), "trust fail case"),
        (lambda rows: rows[8]["source_requirements"][0].__setitem__(
            "source_match", 1), "source fail case"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[10]["seed_id"]), "更早"),
        (lambda rows: (
            rows[11].__setitem__("sample_role", "supersede"),
            rows[11].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[11].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "跨 family/split"),
        (lambda rows: (
            rows[4].__setitem__("perturbation_kind", "NONE"),
            rows[7].__setitem__("perturbation_kind", "NONE"),
        ),
         "缺少必需反向破坏"),
    ],
)
def test_bad_owner_adoption_requirement_observation_budget_and_chain_fail_closed(
        tmp_path, mutate, message):
    """坏 owner/adoption/source observation/预算/恢复链均不能进 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredGenerationCourseError, match=message):
        read_authored_generation_seeds(bad)


def test_qa_manifest_does_not_drift(tmp_path):
    """新增 generation 合同不得改变已关闭 QA artifact 身份。"""
    qa = compile_authored_qa_course(QA_SAMPLE_PATH, tmp_path / "qa")
    assert qa.manifest.content_sha256() == (
        "2785a4e55474ae90ba20828410fde4dc69faca1e861c78c77aff8cf6de55ad62")


def test_sample_hash_float_noncanonical_existing_pack_and_shortcuts_fail_closed(
        tmp_path):
    """钉住 sample；float/非规范/覆盖失败，源码不搬非空/renderer/expected 捷径。"""
    assert hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest() == (
        "e0546aa8ce17812d606c27ccc0a561a9eb6b6b6a8829bec9f833ba2a30e58e4a")
    rows = _sample_values()
    rows[0]["consumer_request"]["max_candidates"] = 1.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredGenerationCourseError, match="规范 JSON"):
        read_authored_generation_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"candidates": []}\n')
    with pytest.raises(AuthoredGenerationCourseError, match="规范 JSON"):
        read_authored_generation_seeds(bad_json)
    build = compile_authored_generation_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredGenerationCourseError, match="发布失败"):
        compile_authored_generation_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "expected_state ==",
            "surface.startswith",
            "if rendered",
            "if output",
            "teacher_answer"}:
        assert token not in source
