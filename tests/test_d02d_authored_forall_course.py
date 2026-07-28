"""D-02D.5 AUTHORED_CC0_V1 typed FORALL 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    OBJECT_VARIABLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorCandidateProtocol,
    LogicOperatorCandidateSpec,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    FiniteQuantifierDomain,
    LogicEvidenceState,
    LogicOperatorDefinition,
    OperatorSlot,
    QuantifierDefinition,
    UniversalOperator,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
    TypedValue,
)
from pure_integer_ai.experiments.ph2_authored_exists_course import (
    compile_authored_exists_course,
)
from pure_integer_ai.experiments.ph2_authored_forall_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_forall_course,
    read_authored_forall_seeds,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    authored_logic_instruction_identity,
    authored_logic_operator_identity,
    authored_logic_role_identity,
    authored_logic_structure_identity,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    INSTRUCTION_FORALL,
    OPERATOR_FORALL,
    ROLE_FORALL_BODY,
    ROLE_FORALL_VALUE,
    STRUCTURE_FORALL,
)
from pure_integer_ai.experiments.ph2_authored_quantifier_compile import (
    authored_quantifier_value_type,
)
from pure_integer_ai.experiments.ph2_authored_quantifier_schema import (
    LICENSE_ID,
    REQUEST_QUANTIFIER_EXECUTION,
    SOURCE_KEY,
    AuthoredQuantifierCourseError,
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


SAMPLE_PATH = Path("data/ph2/authored_logic_forall_seed_v1.jsonl.sample")
EXISTS_SAMPLE_PATH = Path(
    "data/ph2/authored_logic_exists_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_quantifier_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_quantifier_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_forall_course.py"),
)


def _sample_values() -> list[dict]:
    """读取仓库 sample 为独立可修改 JSON object 列表。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """按统一规范写测试 JSONL。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _write_json_with_float(path: Path, values: list[dict]) -> None:
    """绕过合同 writer 落一个 float parser 负例。"""
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
    """读取一个 pack 内指定 record kind。"""
    out = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            out.extend(read_record_artifact(build.pack_root, identity))
    return tuple(out)


def _identity(value) -> ObjectIdentity:
    """从规范整数列表恢复一等对象身份。"""
    return ObjectIdentity.from_stable_key(tuple(value))


def _bound(value: dict) -> BoundProposition:
    """递归恢复量词 root/body 及 Variable filler。"""
    bindings = []
    for item in value["bindings"]:
        filler_value = item["filler"]
        if filler_value["kind"] == "identity":
            filler = _identity(filler_value["identity_key"])
        else:
            assert filler_value["kind"] == "bound_proposition"
            filler = _bound(filler_value["bound"])
        bindings.append(BoundRoleBinding(
            _identity(item["role_key"]), filler, item["ordinal"]))
    return BoundProposition(
        _identity(value["template_key"]),
        _identity(value["instruction_key"]),
        _identity(value["predicate_key"]),
        _identity(value["structure_key"]),
        _identity(value["source_anchor_key"]),
        _identity(value["context_key"]),
        tuple(_identity(item) for item in value["introduced_binder_keys"]),
        tuple(bindings),
        tuple(_identity(item) for item in value["applied_variable_keys"]),
    )


def _forall(states: tuple[LogicEvidenceState, ...], closed: bool):
    """按 S-04 FORALL 聚合公式独立计算四态。"""
    if any(item.refute for item in states):
        return LogicEvidenceState(
            closed and all(item.support for item in states), True)
    if closed and all(item.support for item in states):
        return LogicEvidenceState(True, False)
    return LogicEvidenceState(False, False)


def test_sample_covers_support_counterexample_open_empty_type_and_swap():
    """FORALL 覆盖完整支持、反例、开放域、空域、类型失败和修订链。"""
    seeds = read_authored_forall_seeds(SAMPLE_PATH)
    assert len(seeds) == 12
    assert LICENSE_ID == "CC0-1.0"
    assert {item.operator_family for item in seeds} == {"FORALL"}
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    states = set()
    for seed in seeds:
        assert seed.operator_kind == OPERATOR_FORALL
        assert seed.structure_kind == STRUCTURE_FORALL
        assert seed.instruction_kind == INSTRUCTION_FORALL
        assert seed.body_role_kind == ROLE_FORALL_BODY
        assert seed.value_role_kind == ROLE_FORALL_VALUE
        assert seed.consumer_request.request_kind == (
            REQUEST_QUANTIFIER_EXECUTION)
        if seed.perturbation_kind != "DOMAIN_TYPE_MISMATCH":
            value_states = tuple(LogicEvidenceState(
                bool(item.evidence_support), bool(item.evidence_refute))
                for item in seed.domain.values)
            result = _forall(value_states, bool(seed.domain.closed))
            states.add((result.support, result.refute))
    assert states == {
        (True, False), (False, True), (False, False), (True, True)}


def test_closed_open_counterexample_and_empty_domain_are_distinct():
    """开放域全支持不证真，反例不需闭域，闭合空域按全称语义支持。"""
    seeds = read_authored_forall_seeds(SAMPLE_PATH)
    closed = next(item for item in seeds
                  if item.seed_id == "teacher-forall-closed-all-support-v1")
    opened = next(item for item in seeds
                  if item.seed_id == "teacher-forall-open-all-support-v1")
    counterexample = next(
        item for item in seeds
        if item.seed_id == "teacher-forall-open-counterexample-v1")
    empty = next(item for item in seeds
                 if item.seed_id == "teacher-forall-closed-empty-v1")
    closed_states = tuple(LogicEvidenceState(
        bool(item.evidence_support), bool(item.evidence_refute))
        for item in closed.domain.values)
    open_states = tuple(LogicEvidenceState(
        bool(item.evidence_support), bool(item.evidence_refute))
        for item in opened.domain.values)
    counterexample_states = tuple(LogicEvidenceState(
        bool(item.evidence_support), bool(item.evidence_refute))
        for item in counterexample.domain.values)
    assert closed_states == open_states
    assert _forall(closed_states, True) == LogicEvidenceState(True, False)
    assert _forall(open_states, False) == LogicEvidenceState(False, False)
    assert _forall(counterexample_states, False) == LogicEvidenceState(
        False, True)
    assert _forall((), bool(empty.domain.closed)) == LogicEvidenceState(
        True, False)


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，12/12/9/3 分账并保持 expected 私有。"""
    first = compile_authored_forall_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_forall_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.w_stages == ("W-07",)
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


def test_payload_roundtrips_binder_domain_universal_candidate_and_budget(
        tmp_path):
    """恢复 Binder/Variable/domain、Universal candidate、bound root 和预算。"""
    build = compile_authored_forall_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        definition_value = payload["operator_definition"]
        definition = LogicOperatorDefinition(
            _identity(definition_value["structure_key"]),
            _identity(definition_value["instruction_key"]),
            tuple(OperatorSlot(
                _identity(item["role_key"]), item["ordinal"])
                for item in definition_value["slots"]),
            UniversalOperator(),
        )
        assert definition.structure == authored_logic_structure_identity(
            STRUCTURE_FORALL)
        assert definition.instruction == authored_logic_instruction_identity(
            INSTRUCTION_FORALL)
        assert definition.slots == (OperatorSlot(
            authored_logic_role_identity(ROLE_FORALL_BODY), 0),)
        protocol_value = payload["candidate_protocol"]
        protocol = LogicOperatorCandidateProtocol(
            _identity(protocol_value["structure_predicate_key"]),
            _identity(protocol_value["instruction_predicate_key"]),
            _identity(protocol_value["slot_predicate_key"]),
        )
        candidate_value = payload["candidate_spec"]
        sources = tuple(SourceRef.from_stable_key(tuple(item))
                        for item in candidate_value["forming_source_keys"])
        spec = LogicOperatorCandidateSpec(
            _identity(candidate_value["candidate_key"]),
            definition,
            tuple(candidate_value["competition_key"]),
            sources,
        )
        assert len(spec.candidate_definition(protocol).bindings) == 3
        quantifier_value = payload["quantifier_definition"]
        domain_value = quantifier_value["domain"]
        domain = FiniteQuantifierDomain(
            _identity(domain_value["domain_key"]),
            tuple(TypedValue(
                _identity(item["value_key"]),
                _identity(item["type_key"]),
            ) for item in domain_value["values"]),
            bool(domain_value["closed"]),
            tuple(_identity(item)
                  for item in domain_value["closure_evidence_keys"]),
        )
        quantifier = QuantifierDefinition(
            _identity(quantifier_value["binder_key"]),
            _identity(quantifier_value["variable_key"]),
            OperatorSlot(
                _identity(quantifier_value["body_slot"]["role_key"]),
                quantifier_value["body_slot"]["ordinal"],
            ),
            domain,
        )
        assert quantifier.binder.object_kind == OBJECT_BINDER
        assert quantifier.variable.object_kind == OBJECT_VARIABLE
        assert quantifier.domain.domain.object_kind == OBJECT_SET_EXPR
        assert quantifier.body_slot.role == authored_logic_role_identity(
            ROLE_FORALL_BODY)
        assert _identity(quantifier_value["value_type_key"]) == (
            authored_quantifier_value_type(1))
        root = _bound(payload["bound_root"])
        assert root.template == spec.candidate
        assert root.template.object_kind == OBJECT_PROPOSITION
        assert root.predicate == authored_logic_operator_identity(
            OPERATOR_FORALL)
        assert root.introduced_binders == (quantifier.binder,)
        assert len(root.bindings) == 1
        body = root.bindings[0].filler
        assert isinstance(body, BoundProposition)
        assert body.bindings[0].role == authored_logic_role_identity(
            ROLE_FORALL_VALUE)
        assert body.bindings[0].filler == quantifier.variable
        evidence_index = {
            tuple(item["value_key"]): item for item in payload["value_evidence"]}
        assert set(evidence_index) == {
            item.value.stable_key() for item in domain.values}
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_QUANTIFIER_EXECUTION
        assert request["budget"] == {
            "max_branches": 16,
            "max_depth": 6,
            "max_domain_values": 8,
            "max_steps": 64,
        }
        scope = ScopeIdentity.from_stable_key(tuple(request["scope_key"]))
        assert scope.source == sources[0]
        assert payload["closed_domain_requires_evidence"] == 1
        assert payload["open_domain_current_all_support_is_true"] == 0
        assert payload["explicit_counterexample_requires_closed_domain"] == 0
        assert payload["closed_empty_domain_vacuous_support"] == 1
        assert payload["surface_cue_authoritative"] == 0


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-07 视图和同 quantifier 修订可复核。"""
    build = compile_authored_forall_course(SAMPLE_PATH, tmp_path)
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
        train, teachers, (), current_stage="W-07", view_kind="training")
    validate_stage_visibility(
        held_out, (), evaluators, current_stage="W-07", view_kind="evaluation")
    superseder = next(
        item for item in observations if item.sample_role == "supersede")
    target = next(
        item for item in observations if item.stable_key == superseder.supersedes_key)
    assert target.logical_order < superseder.logical_order
    assert target.split == superseder.split == "train"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"),
         "CC0-1.0"),
        (lambda rows: rows[1].__setitem__("seed_id", rows[0]["seed_id"]),
         "重复"),
        (lambda rows: rows[9].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("operator_family", "EXISTS"),
         "profile"),
        (lambda rows: rows[0].__setitem__("operator_kind", 5), "profile"),
        (lambda rows: rows[0].__setitem__("structure_kind", 5), "profile"),
        (lambda rows: rows[0].__setitem__("instruction_kind", 5), "profile"),
        (lambda rows: rows[0].__setitem__("body_role_kind", 6),
         "Role profile"),
        (lambda rows: rows[0]["domain"].__setitem__(
            "closure_evidence_local_ids", []), "closure evidence"),
        (lambda rows: rows[1]["domain"].__setitem__(
            "closure_evidence_local_ids", [1]), "open quantifier"),
        (lambda rows: rows[0]["domain"]["values"][1].__setitem__(
            "value_id", rows[0]["domain"]["values"][0]["value_id"]),
         "重复"),
        (lambda rows: rows[0]["domain"]["values"][0].__setitem__(
            "actual_type_kind", 2), "非类型扰动"),
        (lambda rows: rows[5]["domain"]["values"][0].__setitem__(
            "actual_type_kind", 1), "必须有错类型"),
        (lambda rows: rows[0]["domain"]["values"][0].__setitem__(
            "evidence_support", 2), "0/1"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "max_domain_values", 1), "超过 consumer"),
        (lambda rows: rows[0]["body"].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[9]["seed_id"]), "更早"),
        (lambda rows: (
            rows[10].__setitem__("sample_role", "supersede"),
            rows[10].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[10].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "family/split/operator"),
        (lambda rows: rows[4].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_license_profile_domain_type_budget_span_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、Binder/domain/type/预算、span 和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredQuantifierCourseError, match=message):
        read_authored_forall_seeds(bad)


def test_exists_manifest_does_not_drift_when_forall_semantics_are_added(
        tmp_path):
    """FORALL 专属字段不得改变已关闭 EXISTS artifact 身份。"""
    build = compile_authored_exists_course(EXISTS_SAMPLE_PATH, tmp_path)
    assert build.manifest.content_sha256() == (
        "d8910e29a82f7c6fc6df6c1c03c1ef5e19a56c411e2b690dc7124fbc69155964")


def test_float_noncanonical_existing_pack_and_legacy_are_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬字符串量词或闭世界默认。"""
    rows = _sample_values()
    rows[0]["operator_kind"] = 6.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredQuantifierCourseError, match="规范 JSON"):
        read_authored_forall_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredQuantifierCourseError, match="规范 JSON"):
        read_authored_forall_seeds(bad_json)
    build = compile_authored_forall_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredQuantifierCourseError, match="发布失败"):
        compile_authored_forall_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "surface.startswith",
            "closed=True",
            "token_seq",
            "expected_state =="}:
        assert token not in source
