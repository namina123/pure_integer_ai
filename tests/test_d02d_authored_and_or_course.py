"""D-02D.2 AUTHORED_CC0_V1 typed AND/OR 资料包 T0。"""
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
    ConjunctionOperator,
    DisjunctionOperator,
    LogicEvidenceState,
    LogicOperatorDefinition,
    OperatorSlot,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.ph2_authored_and_or_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_and_or_course,
    read_authored_and_or_seeds,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    authored_logic_instruction_identity,
    authored_logic_operator_identity,
    authored_logic_role_identity,
    authored_logic_structure_identity,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    INSTRUCTION_AND,
    INSTRUCTION_OR,
    LICENSE_ID,
    OPERATOR_AND,
    OPERATOR_OR,
    REQUEST_LOGIC_EXECUTION,
    ROLE_AND_OPERAND,
    ROLE_OR_OPERAND,
    SOURCE_KEY,
    STRUCTURE_AND,
    STRUCTURE_OR,
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


SAMPLE_PATH = Path("data/ph2/authored_logic_and_or_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_logic_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_logic_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_and_or_course.py"),
)
PROFILE = {
    "AND": (
        OPERATOR_AND,
        STRUCTURE_AND,
        INSTRUCTION_AND,
        ROLE_AND_OPERAND,
        ConjunctionOperator,
    ),
    "OR": (
        OPERATOR_OR,
        STRUCTURE_OR,
        INSTRUCTION_OR,
        ROLE_OR_OPERAND,
        DisjunctionOperator,
    ),
}


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
    bindings = []
    for item in value["bindings"]:
        filler_value = item["filler"]
        assert filler_value["kind"] == "bound_proposition"
        bindings.append(BoundRoleBinding(
            _identity(item["role_key"]),
            _bound(filler_value["bound"]),
            item["ordinal"],
        ))
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


def _combine(family: str, states: tuple[LogicEvidenceState, ...]):
    """按 S-04 冻结 evidence-bit 公式独立计算 AND/OR 四态。"""
    if family == "AND":
        return LogicEvidenceState(
            all(item.support for item in states),
            any(item.refute for item in states),
        )
    assert family == "OR"
    return LogicEvidenceState(
        any(item.support for item in states),
        all(item.refute for item in states),
    )


def test_sample_covers_both_operators_four_states_and_branch_perturbations():
    """AND/OR 各自覆盖四态，并含 order/branch/operator 反向破坏。"""
    seeds = read_authored_and_or_seeds(SAMPLE_PATH)
    assert len(seeds) == 15
    assert LICENSE_ID == "CC0-1.0"
    assert {item.operator_family for item in seeds} == set(PROFILE)
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    assert any(
        item.perturbation_kind == "CONTENT_REPLACEMENT"
        and item.split == "held_out"
        for item in seeds
    )
    output_states = {"AND": set(), "OR": set()}
    for seed in seeds:
        profile = PROFILE[seed.operator_family]
        assert seed.operator_kind == profile[0]
        assert seed.structure_kind == profile[1]
        assert seed.instruction_kind == profile[2]
        assert len(seed.operands) == len(seed.bindings) == 2
        assert {item.object_kind for item in seed.operands} == {
            OBJECT_PROPOSITION}
        assert {item.role_kind for item in seed.bindings} == {profile[3]}
        assert {item.ordinal for item in seed.bindings} == {0, 1}
        states = tuple(LogicEvidenceState(
            bool(item.evidence_support), bool(item.evidence_refute))
            for item in seed.operands)
        output_states[seed.operator_family].add(
            (_combine(seed.operator_family, states).support,
             _combine(seed.operator_family, states).refute))
    assert output_states["AND"] == {
        (True, False), (False, True), (False, False), (True, True)}
    assert output_states["OR"] == {
        (True, False), (False, True), (False, False), (True, True)}


def test_operand_order_swap_changes_trace_order_not_commutative_result():
    """交换 OR slot ordinal 会改变显式 trace，但不改变 evidence-bit 结果。"""
    seed = next(
        item for item in read_authored_and_or_seeds(SAMPLE_PATH)
        if item.perturbation_kind == "OPERAND_ORDER_SWAP")
    by_ordinal = {
        item.ordinal: item.operand_id for item in seed.bindings}
    assert by_ordinal == {0: "door-close", 1: "light-on"}
    operand_by_id = {item.operand_id: item for item in seed.operands}
    ordered = tuple(LogicEvidenceState(
        bool(operand_by_id[by_ordinal[index]].evidence_support),
        bool(operand_by_id[by_ordinal[index]].evidence_refute),
    ) for index in (0, 1))
    assert _combine("OR", ordered) == _combine("OR", tuple(reversed(ordered)))


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，15/15/11/4 分账并保持 expected 私有。"""
    first = compile_authored_and_or_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_and_or_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.w_stages == ("W-07",)
    assert first.validation.source_ref_count == 15
    assert first.validation.observation_count == 15
    assert first.validation.teacher_evidence_count == 11
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
    assert len(_records(first, RECORD_SOURCE_REF)) == 15
    assert len(observations) == 15
    assert len(_records(first, RECORD_TEACHER_EVIDENCE)) == 11
    assert len(_records(first, RECORD_EVALUATOR_LABEL)) == 4
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_payload_roundtrips_operator_candidate_two_branches_and_budget(tmp_path):
    """恢复双 operator definition、H-05 candidate、两个 branch、scope 和预算。"""
    build = compile_authored_and_or_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        family = payload["operator_family"]
        profile = PROFILE[family]
        definition_value = payload["operator_definition"]
        definition = LogicOperatorDefinition(
            _identity(definition_value["structure_key"]),
            _identity(definition_value["instruction_key"]),
            tuple(OperatorSlot(
                _identity(item["role_key"]), item["ordinal"])
                for item in definition_value["slots"]),
            profile[4](),
        )
        assert definition.structure == authored_logic_structure_identity(
            profile[1])
        assert definition.instruction == authored_logic_instruction_identity(
            profile[2])
        assert definition.slots == tuple(OperatorSlot(
            authored_logic_role_identity(profile[3]), index)
            for index in (0, 1))
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
        assert root.predicate == authored_logic_operator_identity(profile[0])
        assert root.structure == authored_logic_structure_identity(profile[1])
        assert len(root.bindings) == 2
        assert all(isinstance(item.filler, BoundProposition)
                   for item in root.bindings)
        evidence = payload["operand_evidence"]
        assert len(evidence) == 2
        states = tuple(LogicEvidenceState(
            bool(item["support"]), bool(item["refute"])) for item in evidence)
        assert isinstance(_combine(family, states), LogicEvidenceState)
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_LOGIC_EXECUTION
        assert request["root_key"] == candidate_value["candidate_key"]
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
    build = compile_authored_and_or_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[11].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("operator_family", "XOR"),
         "family"),
        (lambda rows: rows[0].__setitem__("operator_kind", 3), "profile"),
        (lambda rows: rows[0].__setitem__("structure_kind", 3), "profile"),
        (lambda rows: rows[0].__setitem__("instruction_kind", 3), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 3),
         "Role profile"),
        (lambda rows: rows[0]["bindings"][1].__setitem__("ordinal", 0),
         "slot 重复"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "object_kind", 17), "Proposition"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "evidence_refute", 2), "0/1"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "max_branches", 0), "max_branches"),
        (lambda rows: rows[0].__setitem__("nesting_depth", 2), "单层"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[10].__setitem__(
            "supersedes_seed_id", rows[11]["seed_id"]), "更早"),
        (lambda rows: (
            rows[12].__setitem__("sample_role", "supersede"),
            rows[12].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[12].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "family/split/operator"),
        (lambda rows: rows[8].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_license_profile_branch_budget_span_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、operator、branch、四态位、预算和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredLogicCourseError, match=message):
        read_authored_and_or_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_are_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬旧边、surface 分支或闭世界默认。"""
    rows = _sample_values()
    rows[0]["operator_kind"] = 2.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredLogicCourseError, match="规范 JSON"):
        read_authored_and_or_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredLogicCourseError, match="规范 JSON"):
        read_authored_and_or_seeds(bad_json)
    build = compile_authored_and_or_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredLogicCourseError, match="发布失败"):
        compile_authored_and_or_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_AND",
            "EDGE_OR",
            "token_seq",
            "surface.startswith",
            "closed_world=True"}:
        assert token not in source
