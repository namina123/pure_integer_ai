"""D-02D.1 AUTHORED_CC0_V1 typed NOT 资料包 T0。"""
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
    LogicEvidenceState,
    LogicOperatorDefinition,
    NegationOperator,
    OperatorSlot,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    authored_logic_instruction_identity,
    authored_logic_operator_identity,
    authored_logic_role_identity,
    authored_logic_structure_identity,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    INSTRUCTION_NOT,
    LICENSE_ID,
    OPERATOR_NOT,
    REQUEST_LOGIC_EXECUTION,
    ROLE_NOT_OPERAND,
    SOURCE_KEY,
    STRUCTURE_NOT,
    AuthoredLogicCourseError,
)
from pure_integer_ai.experiments.ph2_authored_not_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_not_course,
    read_authored_not_seeds,
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


SAMPLE_PATH = Path("data/ph2/authored_logic_not_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_logic_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_logic_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_not_course.py"),
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
    """递归恢复 payload 中的不可物化 BoundProposition。"""
    bindings = []
    for item in value["bindings"]:
        filler_value = item["filler"]
        if filler_value["kind"] == "identity":
            filler = _identity(filler_value["identity_key"])
        else:
            assert filler_value["kind"] == "bound_proposition"
            filler = _bound(filler_value["bound"])
        bindings.append(BoundRoleBinding(
            _identity(item["role_key"]),
            filler,
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


def _nested_leaf(root: BoundProposition):
    """返回连续一元 NOT 层数和最终原子 BoundProposition。"""
    depth = 0
    current = root
    not_structure = authored_logic_structure_identity(STRUCTURE_NOT)
    while current.structure == not_structure:
        assert len(current.bindings) == 1
        filler = current.bindings[0].filler
        assert isinstance(filler, BoundProposition)
        depth += 1
        current = filler
    return depth, current


def test_sample_covers_four_states_targets_open_world_and_order():
    """NOT 覆盖四态、双重否定、target/scope、开放世界和恢复链。"""
    seeds = read_authored_not_seeds(SAMPLE_PATH)
    assert len(seeds) == 13
    assert LICENSE_ID == "CC0-1.0"
    assert {item.operator_family for item in seeds} == {"NOT"}
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    assert any(
        item.perturbation_kind == "CONTENT_REPLACEMENT"
        and item.split == "held_out"
        for item in seeds
    )
    evidence_states = {
        (item.operands[0].evidence_support, item.operands[0].evidence_refute)
        for item in seeds
    }
    assert evidence_states == {(1, 0), (0, 1), (0, 0), (1, 1)}
    for seed in seeds:
        assert seed.operator_kind == OPERATOR_NOT
        assert seed.structure_kind == STRUCTURE_NOT
        assert seed.instruction_kind == INSTRUCTION_NOT
        assert seed.consumer_request.request_kind == REQUEST_LOGIC_EXECUTION
        assert len(seed.operands) == len(seed.bindings) == 1
        assert seed.operands[0].object_kind == OBJECT_PROPOSITION
        assert seed.bindings[0].role_kind == ROLE_NOT_OPERAND
        assert seed.nesting_depth == (
            2 if seed.perturbation_kind == "DOUBLE_NEGATION" else 1)


def test_four_state_negation_is_open_world_and_double_negation_is_explicit():
    """NOT 交换证据位，unknown/conflict 不二值化，双重否定需两次执行。"""
    supported = LogicEvidenceState(True, False)
    refuted = LogicEvidenceState(False, True)
    unknown = LogicEvidenceState(False, False)
    conflicted = LogicEvidenceState(True, True)
    assert supported.negate() == refuted
    assert refuted.negate() == supported
    assert unknown.negate() == unknown
    assert conflicted.negate() == conflicted
    assert supported.negate().negate() == supported


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，13/13/9/4 分账并保持 expected 私有。"""
    first = compile_authored_not_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_not_course(SAMPLE_PATH, tmp_path / "second")
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


def test_payload_roundtrips_operator_candidate_bound_root_and_budget(tmp_path):
    """恢复 operator definition、H-05 candidate、嵌套 root、四态输入和 scope。"""
    build = compile_authored_not_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        definition_value = payload["operator_definition"]
        definition = LogicOperatorDefinition(
            _identity(definition_value["structure_key"]),
            _identity(definition_value["instruction_key"]),
            tuple(OperatorSlot(
                _identity(item["role_key"]),
                item["ordinal"],
            ) for item in definition_value["slots"]),
            NegationOperator(),
        )
        assert definition.structure == authored_logic_structure_identity(
            STRUCTURE_NOT)
        assert definition.instruction == authored_logic_instruction_identity(
            INSTRUCTION_NOT)
        assert definition.slots == (
            OperatorSlot(authored_logic_role_identity(ROLE_NOT_OPERAND), 0),)
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
        assert len(spec.candidate_definition(protocol).bindings) == (
            candidate_value["graph_binding_count"])
        root = _bound(payload["bound_root"])
        assert root.template == spec.candidate
        assert root.predicate == authored_logic_operator_identity(OPERATOR_NOT)
        depth, leaf = _nested_leaf(root)
        assert depth == payload["nesting_depth"]
        assert leaf.template.object_kind == OBJECT_PROPOSITION
        evidence = payload["operand_evidence"]
        assert len(evidence) == 1
        assert _identity(evidence[0]["template_key"]) == leaf.template
        state = LogicEvidenceState(
            bool(evidence[0]["support"]),
            bool(evidence[0]["refute"]),
        )
        result = state
        for _ in range(depth):
            result = result.negate()
        assert isinstance(result, LogicEvidenceState)
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_LOGIC_EXECUTION
        assert request["root_key"] == candidate_value["candidate_key"]
        assert request["budget"] == {
            "max_branches": 4,
            "max_depth": 4,
            "max_steps": 16,
        }
        scope = ScopeIdentity.from_stable_key(tuple(request["scope_key"]))
        assert scope.source == sources[0]
        assert payload["closed_world_assumed"] == 0
        assert payload["surface_cue_authoritative"] == 0


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-07 视图和 parser revision 可复核。"""
    build = compile_authored_not_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[0].__setitem__("operator_family", "AND"),
         "profile"),
        (lambda rows: rows[0].__setitem__("operator_kind", 2), "profile"),
        (lambda rows: rows[0].__setitem__("structure_kind", 2), "profile"),
        (lambda rows: rows[0].__setitem__("instruction_kind", 2), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 2),
         "Role profile"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "object_kind", 17), "Proposition"),
        (lambda rows: rows[0]["operands"][0].__setitem__(
            "evidence_support", 2), "0/1"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 2), "request kind"),
        (lambda rows: rows[4]["consumer_request"].__setitem__(
            "max_depth", 1), "max_depth"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "operand_id", "missing"), "未知 operand"),
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
        (lambda rows: rows[4].__setitem__("nesting_depth", 1),
         "double negation"),
    ],
)
def test_bad_license_profile_operand_budget_span_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、operator、四态位、预算、target 和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredLogicCourseError, match=message):
        read_authored_not_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_are_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬旧边、token 序或闭世界默认。"""
    rows = _sample_values()
    rows[0]["operator_kind"] = 1.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredLogicCourseError, match="规范 JSON"):
        read_authored_not_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredLogicCourseError, match="规范 JSON"):
        read_authored_not_seeds(bad_json)
    build = compile_authored_not_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredLogicCourseError, match="发布失败"):
        compile_authored_not_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_NOT",
            "build_negation",
            "token_seq",
            "slot_seq",
            "closed_world=True"}:
        assert token not in source
