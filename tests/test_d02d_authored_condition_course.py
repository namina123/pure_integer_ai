"""D-02D.3 AUTHORED_CC0_V1 typed CONDITION 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorCandidateProtocol,
    LogicOperatorCandidateSpec,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    ConditionOperator,
    LogicEvidenceState,
    LogicOperatorDefinition,
    OperatorSlot,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.ph2_authored_condition_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_condition_course,
    read_authored_condition_seeds,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    authored_logic_instruction_identity,
    authored_logic_operator_identity,
    authored_logic_role_identity,
    authored_logic_structure_identity,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    INSTRUCTION_CONDITION,
    LICENSE_ID,
    OPERATOR_CONDITION,
    REQUEST_LOGIC_EXECUTION,
    ROLE_CONDITION_ANTECEDENT,
    ROLE_CONDITION_CONSEQUENT,
    SOURCE_KEY,
    STRUCTURE_CONDITION,
    AuthoredLogicCourseError,
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


SAMPLE_PATH = Path("data/ph2/authored_logic_condition_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_logic_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_logic_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_condition_course.py"),
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
    """递归恢复 payload 中的 BoundProposition。"""
    bindings = tuple(BoundRoleBinding(
        _identity(item["role_key"]),
        _bound(item["filler"]["bound"]),
        item["ordinal"],
    ) for item in value["bindings"])
    return BoundProposition(
        _identity(value["template_key"]),
        _identity(value["instruction_key"]),
        _identity(value["predicate_key"]),
        _identity(value["structure_key"]),
        _identity(value["source_anchor_key"]),
        _identity(value["context_key"]),
        tuple(_identity(item) for item in value["introduced_binder_keys"]),
        bindings,
        tuple(_identity(item) for item in value["applied_variable_keys"]),
    )


def _condition(
        antecedent: LogicEvidenceState,
        consequent: LogicEvidenceState) -> LogicEvidenceState:
    """按 S-04 冻结 material implication evidence-bit 公式计算。"""
    return LogicEvidenceState(
        antecedent.refute or consequent.support,
        antecedent.support and consequent.refute,
    )


def test_sample_covers_four_states_ordered_roles_and_relation_confusions():
    """CONDITION 覆盖四态、有序前后件、因果/时序混淆和恢复链。"""
    seeds = read_authored_condition_seeds(SAMPLE_PATH)
    assert len(seeds) == 13
    assert LICENSE_ID == "CC0-1.0"
    assert {item.operator_family for item in seeds} == {"CONDITION"}
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    states = set()
    for seed in seeds:
        assert seed.operator_kind == OPERATOR_CONDITION
        assert seed.structure_kind == STRUCTURE_CONDITION
        assert seed.instruction_kind == INSTRUCTION_CONDITION
        assert len(seed.operands) == len(seed.bindings) == 2
        assert {item.object_kind for item in seed.operands} == {
            OBJECT_PROPOSITION}
        assert [item.role_kind for item in seed.bindings] == [
            ROLE_CONDITION_ANTECEDENT,
            ROLE_CONDITION_CONSEQUENT,
        ]
        operand_by_id = {item.operand_id: item for item in seed.operands}
        ordered = tuple(LogicEvidenceState(
            bool(operand_by_id[item.operand_id].evidence_support),
            bool(operand_by_id[item.operand_id].evidence_refute),
        ) for item in seed.bindings)
        result = _condition(*ordered)
        states.add((result.support, result.refute))
    assert states == {
        (True, False), (False, True), (False, False), (True, True)}


def test_antecedent_consequent_swap_changes_noncommutative_result():
    """前后件交换真实改变 material implication，不按 surface 私自纠正。"""
    seed = next(
        item for item in read_authored_condition_seeds(SAMPLE_PATH)
        if item.perturbation_kind == "ANTECEDENT_CONSEQUENT_SWAP")
    operand_by_id = {item.operand_id: item for item in seed.operands}
    ordered = tuple(LogicEvidenceState(
        bool(operand_by_id[item.operand_id].evidence_support),
        bool(operand_by_id[item.operand_id].evidence_refute),
    ) for item in seed.bindings)
    assert _condition(*ordered) != _condition(*reversed(ordered))


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，13/13/9/4 分账并保持 expected 私有。"""
    first = compile_authored_condition_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_condition_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.w_stages == ("W-07",)
    assert first.validation.source_ref_count == 13
    assert first.validation.observation_count == 13
    assert first.validation.teacher_evidence_count == 9
    assert first.validation.evaluator_label_count == 4
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
    assert len(_records(first, RECORD_SOURCE_REF)) == 13
    assert len(observations) == 13
    assert len(_records(first, RECORD_TEACHER_EVIDENCE)) == 9
    assert len(_records(first, RECORD_EVALUATOR_LABEL)) == 4
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_payload_roundtrips_condition_candidate_roles_scope_and_budget(tmp_path):
    """恢复 CONDITION definition、H-05 candidate、前后件 branch、scope 和预算。"""
    build = compile_authored_condition_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        definition_value = payload["operator_definition"]
        definition = LogicOperatorDefinition(
            _identity(definition_value["structure_key"]),
            _identity(definition_value["instruction_key"]),
            tuple(OperatorSlot(
                _identity(item["role_key"]), item["ordinal"])
                for item in definition_value["slots"]),
            ConditionOperator(),
        )
        assert definition.structure == authored_logic_structure_identity(
            STRUCTURE_CONDITION)
        assert definition.instruction == authored_logic_instruction_identity(
            INSTRUCTION_CONDITION)
        assert definition.slots == (
            OperatorSlot(authored_logic_role_identity(
                ROLE_CONDITION_ANTECEDENT), 0),
            OperatorSlot(authored_logic_role_identity(
                ROLE_CONDITION_CONSEQUENT), 0),
        )
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
        assert len(spec.candidate_definition(protocol).bindings) == 4
        root = _bound(payload["bound_root"])
        assert root.template == spec.candidate
        assert root.predicate == authored_logic_operator_identity(
            OPERATOR_CONDITION)
        assert root.structure == authored_logic_structure_identity(
            STRUCTURE_CONDITION)
        assert [item.role for item in root.bindings] == [
            authored_logic_role_identity(ROLE_CONDITION_ANTECEDENT),
            authored_logic_role_identity(ROLE_CONDITION_CONSEQUENT),
        ]
        evidence = payload["operand_evidence"]
        states = tuple(LogicEvidenceState(
            bool(item["support"]), bool(item["refute"])) for item in evidence)
        assert isinstance(_condition(*states), LogicEvidenceState)
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_LOGIC_EXECUTION
        assert request["budget"] == {
            "max_branches": 8,
            "max_depth": 4,
            "max_steps": 24,
        }
        scope = ScopeIdentity.from_stable_key(tuple(request["scope_key"]))
        assert scope.source == sources[0]
        assert payload["closed_world_assumed"] == 0
        assert payload["surface_cue_authoritative"] == 0


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-07 视图和同 operator 修订可复核。"""
    build = compile_authored_condition_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[0].__setitem__("license_id", "UNKNOWN"), "CC0-1.0"),
        (lambda rows: rows[1].__setitem__("seed_id", rows[0]["seed_id"]),
         "重复"),
        (lambda rows: rows[9].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("operator_family", "CAUSES"),
         "profile"),
        (lambda rows: rows[0].__setitem__("operator_kind", 2), "profile"),
        (lambda rows: rows[0].__setitem__("structure_kind", 2), "profile"),
        (lambda rows: rows[0].__setitem__("instruction_kind", 2), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 5),
         "slot 重复"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("ordinal", 1),
         "Role profile"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "object_kind", 17), "Proposition"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "evidence_support", 2), "0/1"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "max_steps", 0), "max_steps"),
        (lambda rows: rows[0].__setitem__("nesting_depth", 2), "单层"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[9]["seed_id"]), "更早"),
        (lambda rows: (
            rows[10].__setitem__("sample_role", "supersede"),
            rows[10].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[10].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "family/split/operator"),
        (lambda rows: rows[6].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_license_profile_role_budget_span_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、operator、Role、四态位、预算和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredLogicCourseError, match=message):
        read_authored_condition_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_are_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬 CAUSES、PRECEDES 或 surface cue。"""
    rows = _sample_values()
    rows[0]["operator_kind"] = 4.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredLogicCourseError, match="规范 JSON"):
        read_authored_condition_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredLogicCourseError, match="规范 JSON"):
        read_authored_condition_seeds(bad_json)
    build = compile_authored_condition_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredLogicCourseError, match="发布失败"):
        compile_authored_condition_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_CAUSES",
            "EDGE_PRECEDES",
            "token_seq",
            "surface.startswith",
            "closed_world=True"}:
        assert token not in source
