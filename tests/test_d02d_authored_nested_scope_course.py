"""D-02D.7 AUTHORED_CC0_V1 typed 嵌套作用域资料包 T0。"""
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
    ExistentialOperator,
    FiniteQuantifierDomain,
    LogicEvidenceState,
    LogicOperatorDefinition,
    ModalOperator,
    ModalResolution,
    NegationOperator,
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
from pure_integer_ai.experiments.ph2_authored_forall_course import (
    compile_authored_forall_course,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    authored_logic_instruction_identity,
    authored_logic_operator_identity,
    authored_logic_role_identity,
    authored_logic_structure_identity,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    OPERATOR_EXISTS,
    OPERATOR_FORALL,
    OPERATOR_MODAL,
    OPERATOR_NOT,
)
from pure_integer_ai.experiments.ph2_authored_modal_course import (
    compile_authored_modal_course,
)
from pure_integer_ai.experiments.ph2_authored_modal_schema import (
    RESOLVER_RESOLVED,
)
from pure_integer_ai.experiments.ph2_authored_nested_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_nested_course,
    read_authored_nested_seeds,
)
from pure_integer_ai.experiments.ph2_authored_nested_schema import (
    AuthoredNestedCourseError,
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
    "data/ph2/authored_logic_nested_scope_seed_v1.jsonl.sample")
MODAL_SAMPLE_PATH = Path(
    "data/ph2/authored_logic_modal_seed_v1.jsonl.sample")
FORALL_SAMPLE_PATH = Path(
    "data/ph2/authored_logic_forall_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_nested_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_nested_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_nested_course.py"),
)

_T = LogicEvidenceState(True, False)
_F = LogicEvidenceState(False, True)
_U = LogicEvidenceState(False, False)


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
    """递归恢复 nested root、operator child 和 Variable filler。"""
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


def _exists(states: tuple[LogicEvidenceState, ...], closed: bool):
    """独立执行 EXISTS 四态聚合。"""
    if any(item.support for item in states):
        return LogicEvidenceState(
            True, closed and all(item.refute for item in states))
    if closed and all(item.refute for item in states):
        return _F
    return _U


def _forall(states: tuple[LogicEvidenceState, ...], closed: bool):
    """独立执行 FORALL 四态聚合。"""
    if any(item.refute for item in states):
        return LogicEvidenceState(
            closed and all(item.support for item in states), True)
    if closed and all(item.support for item in states):
        return _T
    return _U


def _evaluate_nested(seed) -> LogicEvidenceState:
    """按 layer chain、candidate availability 和 resolver 独立计算四态。"""
    def evaluate(index: int, branch_state: LogicEvidenceState | None = None):
        if index == len(seed.layers):
            if branch_state is not None:
                return branch_state
            return LogicEvidenceState(
                bool(seed.leaf.evidence_support),
                bool(seed.leaf.evidence_refute),
            )
        layer = seed.layers[index]
        if not layer.candidate_available:
            return _U
        if layer.operator_family in {"EXISTS", "FORALL"}:
            assert seed.quantifier is not None
            states = tuple(evaluate(
                index + 1,
                LogicEvidenceState(
                    bool(item.evidence_support),
                    bool(item.evidence_refute),
                ),
            ) for item in seed.quantifier.domain.values)
            if layer.operator_family == "EXISTS":
                return _exists(states, bool(seed.quantifier.domain.closed))
            return _forall(states, bool(seed.quantifier.domain.closed))
        child = evaluate(index + 1, branch_state)
        if layer.operator_family == "NOT":
            return LogicEvidenceState(child.refute, child.support)
        assert layer.operator_family == "MODAL"
        resolver = layer.modal_resolver
        assert resolver is not None
        if resolver.status != RESOLVER_RESOLVED:
            return _U
        return LogicEvidenceState(
            bool(resolver.resolution_support),
            bool(resolver.resolution_refute),
        )

    return evaluate(0)


def _handler(family: str):
    """按 payload family 恢复现役 handler。"""
    return {
        "NOT": NegationOperator,
        "MODAL": ModalOperator,
        "EXISTS": ExistentialOperator,
        "FORALL": UniversalOperator,
    }[family]()


def test_sample_covers_modal_quantifier_order_depth_missing_and_revision():
    """nested 覆盖异构层序、量词翻转、三层深度、缺内层和恢复链。"""
    seeds = read_authored_nested_seeds(SAMPLE_PATH)
    assert len(seeds) == 12
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    chains = {tuple(layer.operator_family for layer in item.layers)
              for item in seeds}
    assert {
        ("NOT", "MODAL"),
        ("MODAL", "NOT"),
        ("NOT", "EXISTS"),
        ("EXISTS", "NOT"),
        ("NOT", "FORALL"),
        ("NOT", "MODAL", "NOT"),
    }.issubset(chains)
    states = {
        (_evaluate_nested(item).support, _evaluate_nested(item).refute)
        for item in seeds}
    assert states == {
        (True, False), (False, True), (False, False), (True, True)}


def test_quantifier_not_scope_swap_changes_result_on_same_domain_states():
    """同一 T/F domain 下 NOT(EXISTS) 与 EXISTS(NOT) 必须给出不同结果。"""
    seeds = read_authored_nested_seeds(SAMPLE_PATH)
    not_exists = next(
        item for item in seeds
        if item.seed_id == "teacher-nested-not-exists-v1")
    exists_not = next(
        item for item in seeds
        if item.seed_id == "teacher-nested-exists-not-v1")
    left_states = tuple(
        (item.evidence_support, item.evidence_refute)
        for item in not_exists.quantifier.domain.values)
    right_states = tuple(
        (item.evidence_support, item.evidence_refute)
        for item in exists_not.quantifier.domain.values)
    assert left_states == right_states == ((1, 0), (0, 1))
    assert _evaluate_nested(not_exists) == _F
    assert _evaluate_nested(exists_not) == _T


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，12/12/9/3 分账并保持 expected 私有。"""
    first = compile_authored_nested_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_nested_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
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


def test_payload_roundtrips_bound_tree_candidates_scope_binder_and_budget(
        tmp_path):
    """恢复异构 tree、每层 candidate/scope、量词 Binder/domain 和预算。"""
    build = compile_authored_nested_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        protocol_value = payload["candidate_protocol"]
        protocol = LogicOperatorCandidateProtocol(
            _identity(protocol_value["structure_predicate_key"]),
            _identity(protocol_value["instruction_predicate_key"]),
            _identity(protocol_value["slot_predicate_key"]),
        )
        root = _bound(payload["bound_root"])
        chain = []
        current = root
        while current.bindings and isinstance(
                current.bindings[0].filler, BoundProposition):
            chain.append(current)
            current = current.bindings[0].filler
        assert len(chain) == len(payload["layers"])
        source = SourceRef.from_stable_key(tuple(
            payload["layers"][0]["candidate_spec"][
                "forming_source_keys"][0]))
        input_scope = ScopeIdentity.from_stable_key(tuple(
            payload["consumer_request"]["scope_key"]))
        assert input_scope.source == source
        quantifier_seen = False
        for layer_value, bound in zip(payload["layers"], chain):
            definition_value = layer_value["operator_definition"]
            definition = LogicOperatorDefinition(
                _identity(definition_value["structure_key"]),
                _identity(definition_value["instruction_key"]),
                tuple(OperatorSlot(
                    _identity(item["role_key"]), item["ordinal"])
                    for item in definition_value["slots"]),
                _handler(layer_value["operator_family"]),
            )
            assert definition.structure == authored_logic_structure_identity(
                layer_value["operator_kind"])
            assert definition.instruction == authored_logic_instruction_identity(
                layer_value["operator_kind"])
            assert bound.template == _identity(layer_value["bound_key"])
            assert bound.predicate == authored_logic_operator_identity(
                layer_value["operator_kind"])
            candidate_value = layer_value["candidate_spec"]
            spec = LogicOperatorCandidateSpec(
                _identity(candidate_value["candidate_key"]),
                definition,
                tuple(candidate_value["competition_key"]),
                tuple(SourceRef.from_stable_key(tuple(item))
                      for item in candidate_value["forming_source_keys"]),
            )
            assert spec.candidate == bound.template
            assert len(spec.candidate_definition(protocol).bindings) == 3
            modal_plan = layer_value["modal_resolution_plan"]
            if modal_plan is not None and modal_plan["status"] == RESOLVER_RESOLVED:
                output_scope = ScopeIdentity.from_stable_key(tuple(
                    modal_plan["output_scope_key"]))
                resolution = ModalResolution(
                    LogicEvidenceState(
                        bool(modal_plan["resolution_state"]["support"]),
                        bool(modal_plan["resolution_state"]["refute"]),
                    ),
                    SourceRef.from_stable_key(tuple(modal_plan["source_key"])),
                    output_scope,
                    tuple(modal_plan["evidence_ids"]),
                )
                assert resolution.source == source
                assert output_scope == input_scope or output_scope.parent == input_scope
            quantifier_value = layer_value["quantifier_definition"]
            if quantifier_value is not None:
                quantifier_seen = True
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
                assert bound.introduced_binders == (quantifier.binder,)
                assert current.bindings[0].filler == quantifier.variable
        assert quantifier_seen == any(
            item["quantifier_definition"] is not None
            for item in payload["layers"])
        assert payload["derivation_order"] == [
            item["layer_id"] for item in reversed(payload["layers"])]
        assert [item["layer_id"] for item in payload["derivation_trace"]] == (
            payload["derivation_order"])
        assert payload["consumer_request"]["budget"] == {
            "max_branches": 16,
            "max_depth": 4,
            "max_domain_values": 8,
            "max_resolver_calls": 1,
            "max_steps": 64,
        }
        assert payload["same_source_required"] == 1
        assert payload["scope_order_authoritative"] == 1
        assert payload["surface_cue_authoritative"] == 0


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-07 视图和同 chain 修订可复核。"""
    build = compile_authored_nested_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[0].__setitem__(
            "operator_registry", "OTHER"), "operator registry"),
        (lambda rows: rows[0]["layers"][0].__setitem__(
            "operator_kind", 2), "profile"),
        (lambda rows: rows[0]["layers"][0].__setitem__(
            "role_kind", 2), "profile"),
        (lambda rows: rows[0]["layers"][1].__setitem__(
            "layer_id", rows[0]["layers"][0]["layer_id"]), "layer_id 重复"),
        (lambda rows: rows[0].__setitem__("layers", rows[0]["layers"][:1]),
         "至少有两个"),
        (lambda rows: rows[0]["layers"].reverse(), "outer-to-inner"),
        (lambda rows: rows[0]["leaf"].__setitem__("object_kind", 1),
         "Proposition"),
        (lambda rows: rows[10]["consumer_request"].__setitem__(
            "max_depth", 2), "depth"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "max_resolver_calls", 0), "正严格整数"),
        (lambda rows: rows[2].__setitem__("quantifier", None),
         "缺少 Binder/domain"),
        (lambda rows: rows[2]["quantifier"].__setitem__(
            "layer_id", "outer-not"), "恰好对应"),
        (lambda rows: rows[2]["quantifier"].__setitem__(
            "value_role_kind", 9), "value Role"),
        (lambda rows: rows[2]["quantifier"]["domain"]["values"][0].__setitem__(
            "actual_type_kind", 2), "错类型"),
        (lambda rows: rows[5]["layers"][0].__setitem__(
            "candidate_available", 0), "只缺一个内层"),
        (lambda rows: rows[0]["layers"][1].__setitem__(
            "candidate_available", 0), "非 missing"),
        (lambda rows: rows[10].__setitem__(
            "layers", rows[10]["layers"][:2]), "至少三层"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[9]["seed_id"]), "更早"),
        (lambda rows: (
            rows[11].__setitem__("sample_role", "supersede"),
            rows[11].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[11].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "family/split/layer chain"),
        (lambda rows: rows[11].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_registry_profile_layer_scope_quantifier_budget_and_chain_fail_closed(
        tmp_path, mutate, message):
    """坏 registry/profile/layer/scope/quantifier/预算和恢复链均拒绝。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredNestedCourseError, match=message):
        read_authored_nested_seeds(bad)


def test_modal_and_forall_manifests_do_not_drift(tmp_path):
    """新增 nested 合同不得改变已关闭 MODAL/FORALL artifact 身份。"""
    modal = compile_authored_modal_course(
        MODAL_SAMPLE_PATH, tmp_path / "modal")
    forall = compile_authored_forall_course(
        FORALL_SAMPLE_PATH, tmp_path / "forall")
    assert modal.manifest.sha256() == (
        "950f57d61ddb8ca04912f94b2e00def553c88d14c8dda0bf58a8bb04a3675f2a")
    assert forall.manifest.sha256() == (
        "598cbd1cac8dde4ae194a1e378ed8534f72ef713d14064c2f3d17c37bb06c4b8")


def test_float_noncanonical_existing_pack_and_legacy_are_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬 surface、expected 或固定层数直通。"""
    rows = _sample_values()
    rows[0]["consumer_request"]["max_depth"] = 4.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredNestedCourseError, match="规范 JSON"):
        read_authored_nested_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredNestedCourseError, match="规范 JSON"):
        read_authored_nested_seeds(bad_json)
    build = compile_authored_nested_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredNestedCourseError, match="发布失败"):
        compile_authored_nested_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "surface.startswith",
            "token_seq",
            "expected_state ==",
            "len(seed.layers) == 2"}:
        assert token not in source
