"""D-02D.6 AUTHORED_CC0_V1 typed MODAL 资料包 T0。"""
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
    ModalOperator,
    ModalResolution,
    OperatorSlot,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.ph2_authored_exists_course import (
    compile_authored_exists_course,
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
    INSTRUCTION_MODAL,
    OPERATOR_MODAL,
    ROLE_MODAL_CHILD,
    STRUCTURE_MODAL,
)
from pure_integer_ai.experiments.ph2_authored_modal_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_modal_course,
    read_authored_modal_seeds,
)
from pure_integer_ai.experiments.ph2_authored_modal_schema import (
    AuthoredModalCourseError,
    RESOLVER_BUDGET_UNDECIDED,
    RESOLVER_DENIED,
    RESOLVER_MISSING,
    RESOLVER_RESOLVED,
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


SAMPLE_PATH = Path("data/ph2/authored_logic_modal_seed_v1.jsonl.sample")
EXISTS_SAMPLE_PATH = Path(
    "data/ph2/authored_logic_exists_seed_v1.jsonl.sample")
FORALL_SAMPLE_PATH = Path(
    "data/ph2/authored_logic_forall_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_logic_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_modal_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_modal_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_modal_course.py"),
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
    """递归恢复 modal root 和 child。"""
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


def _modal_state(seed) -> LogicEvidenceState:
    """只按 resolver 合同独立恢复 modal 四态，不读取 expected。"""
    if seed.resolver.status != RESOLVER_RESOLVED:
        return LogicEvidenceState(False, False)
    return LogicEvidenceState(
        bool(seed.resolver.resolution_support),
        bool(seed.resolver.resolution_refute),
    )


def test_sample_covers_resolver_scope_child_states_budget_and_revision():
    """MODAL 覆盖独立 resolver、三类 scope、child 四态、预算和修订链。"""
    seeds = read_authored_modal_seeds(SAMPLE_PATH)
    assert len(seeds) == 13
    logic_seeds = [item.logic for item in seeds]
    assert {item.operator_family for item in logic_seeds} == {"MODAL"}
    assert {item.sample_role for item in logic_seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in logic_seeds})
    assert {item.resolver.status for item in seeds} == {
        RESOLVER_RESOLVED,
        RESOLVER_MISSING,
        RESOLVER_DENIED,
        RESOLVER_BUDGET_UNDECIDED,
    }
    child_states = {
        (bool(item.logic.operands[0].evidence_support),
         bool(item.logic.operands[0].evidence_refute))
        for item in seeds}
    result_states = {
        (_modal_state(item).support, _modal_state(item).refute)
        for item in seeds}
    assert child_states == {
        (True, False), (False, True), (False, False), (True, True)}
    assert result_states == child_states
    for item in seeds:
        logic = item.logic
        assert logic.operator_kind == OPERATOR_MODAL
        assert logic.structure_kind == STRUCTURE_MODAL
        assert logic.instruction_kind == INSTRUCTION_MODAL
        assert logic.bindings[0].role_kind == ROLE_MODAL_CHILD
        assert logic.nesting_depth == 1
        assert item.max_resolver_calls == 1


def test_child_state_is_not_modal_result_and_unresolved_paths_are_unknown():
    """child true 可被 resolver 反驳，缺失、拒绝和预算未决都保持 unknown。"""
    seeds = read_authored_modal_seeds(SAMPLE_PATH)
    flipped = next(
        item for item in seeds
        if item.logic.seed_id == "teacher-modal-child-true-resolution-false-v1")
    child = flipped.logic.operands[0]
    assert (child.evidence_support, child.evidence_refute) == (1, 0)
    assert _modal_state(flipped) == LogicEvidenceState(False, True)
    unresolved = {
        item.resolver.status: item for item in seeds
        if item.resolver.status != RESOLVER_RESOLVED}
    assert set(unresolved) == {
        RESOLVER_MISSING,
        RESOLVER_DENIED,
        RESOLVER_BUDGET_UNDECIDED,
    }
    assert all(_modal_state(item) == LogicEvidenceState(False, False)
               for item in unresolved.values())


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，13/13/10/3 分账并保持 expected 私有。"""
    first = compile_authored_modal_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_modal_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.w_stages == ("W-07",)
    assert first.validation.source_ref_count == 13
    assert first.validation.observation_count == 13
    assert first.validation.teacher_evidence_count == 10
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
    assert len(_records(first, RECORD_SOURCE_REF)) == 13
    assert len(observations) == 13
    assert len(_records(first, RECORD_TEACHER_EVIDENCE)) == 10
    assert len(_records(first, RECORD_EVALUATOR_LABEL)) == 3
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_payload_roundtrips_modal_candidate_scope_source_and_resolution(
        tmp_path):
    """恢复 Modal handler、child、candidate、scope、source、Evidence 和预算。"""
    build = compile_authored_modal_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        definition_value = payload["operator_definition"]
        definition = LogicOperatorDefinition(
            _identity(definition_value["structure_key"]),
            _identity(definition_value["instruction_key"]),
            tuple(OperatorSlot(
                _identity(item["role_key"]), item["ordinal"])
                for item in definition_value["slots"]),
            ModalOperator(),
        )
        assert definition.structure == authored_logic_structure_identity(
            STRUCTURE_MODAL)
        assert definition.instruction == authored_logic_instruction_identity(
            INSTRUCTION_MODAL)
        assert definition.slots == (OperatorSlot(
            authored_logic_role_identity(ROLE_MODAL_CHILD), 0),)
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
        root = _bound(payload["bound_root"])
        assert root.template == spec.candidate
        assert root.template.object_kind == OBJECT_PROPOSITION
        assert root.predicate == authored_logic_operator_identity(OPERATOR_MODAL)
        assert len(root.bindings) == 1
        assert isinstance(root.bindings[0].filler, BoundProposition)
        assert root.bindings[0].role == authored_logic_role_identity(
            ROLE_MODAL_CHILD)
        plan = payload["modal_resolution_plan"]
        source = SourceRef.from_stable_key(tuple(plan["source_key"]))
        input_scope = ScopeIdentity.from_stable_key(tuple(
            plan["input_scope_key"]))
        assert source == sources[0]
        assert input_scope.source == source
        if plan["status"] == RESOLVER_RESOLVED:
            output_scope = ScopeIdentity.from_stable_key(tuple(
                plan["output_scope_key"]))
            state = LogicEvidenceState(
                bool(plan["resolution_state"]["support"]),
                bool(plan["resolution_state"]["refute"]),
            )
            resolution = ModalResolution(
                state, source, output_scope, tuple(plan["evidence_ids"]))
            assert resolution.source == input_scope.source
            if output_scope.scope_kind == SCOPE_DOCUMENT:
                assert output_scope == input_scope
            else:
                assert output_scope.parent == input_scope
        else:
            assert plan["output_scope_key"] is None
            assert plan["resolution_state"] == {"refute": 0, "support": 0}
            assert plan["evidence_ids"] == []
        assert plan["source_unchanged"] == 1
        assert payload["consumer_request"]["budget"] == {
            "max_branches": 8,
            "max_depth": 4,
            "max_resolver_calls": 1,
            "max_steps": 32,
        }
        assert payload["resolver_required"] == 1
        assert payload["child_evaluated_without_resolver"] == 0
        assert payload["child_state_is_modal_result"] == 0
        assert payload["surface_cue_authoritative"] == 0


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-07 视图和同 modal 修订可复核。"""
    build = compile_authored_modal_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[0].__setitem__("extra", 1), "字段集合"),
        (lambda rows: rows[0]["logic"].__setitem__("license_id", "UNKNOWN"),
         "logic seed 非法"),
        (lambda rows: rows[1]["logic"].__setitem__(
            "seed_id", rows[0]["logic"]["seed_id"]), "重复"),
        (lambda rows: rows[10]["logic"].__setitem__("split", "train"),
         "logic seed 非法"),
        (lambda rows: rows[0]["logic"].__setitem__(
            "operator_family", "NOT"), "profile"),
        (lambda rows: rows[0]["logic"].__setitem__("operator_kind", 1),
         "profile"),
        (lambda rows: rows[0]["logic"].__setitem__("structure_kind", 1),
         "profile"),
        (lambda rows: rows[0]["logic"].__setitem__("instruction_kind", 1),
         "profile"),
        (lambda rows: rows[0]["logic"]["bindings"][0].__setitem__(
            "role_kind", 1), "Proposition child Role"),
        (lambda rows: rows[0]["logic"]["operands"][0].__setitem__(
            "object_kind", 1), "logic seed 非法"),
        (lambda rows: rows[0]["logic"].__setitem__("nesting_depth", 2),
         "不得提前嵌套"),
        (lambda rows: rows[0].__setitem__("max_resolver_calls", 33),
         "超过 step"),
        (lambda rows: rows[0]["resolver"].__setitem__("status", "OTHER"),
         "status"),
        (lambda rows: rows[0]["resolver"].__setitem__(
            "source_mode", "OTHER_SOURCE"), "不得改变 source"),
        (lambda rows: rows[0]["resolver"].__setitem__("scope_kind", 2),
         "scope kind"),
        (lambda rows: (
            rows[1]["resolver"].__setitem__("scope_kind", 1),
            rows[1]["resolver"].__setitem__("scope_local_id", 1),
        ), "不得伪造 local id"),
        (lambda rows: rows[0]["resolver"].__setitem__("scope_local_id", 0),
         "local id"),
        (lambda rows: rows[0]["resolver"].__setitem__("evidence_ids", []),
         "必须有 Evidence"),
        (lambda rows: rows[4]["resolver"].__setitem__(
            "resolution_support", 1), "不得伪造 resolution"),
        (lambda rows: rows[0]["resolver"].__setitem__(
            "evidence_ids", [101, 101]), "非法或重复"),
        (lambda rows: rows[9]["logic"].__setitem__(
            "supersedes_seed_id", rows[10]["logic"]["seed_id"]), "更早"),
        (lambda rows: (
            rows[11]["logic"].__setitem__("sample_role", "supersede"),
            rows[11]["logic"].__setitem__(
                "perturbation_kind", "PARSER_REVISION"),
            rows[11]["logic"].__setitem__(
                "supersedes_seed_id", rows[0]["logic"]["seed_id"]),
        ), "跨 family/split/operator"),
        (lambda rows: rows[7]["logic"].__setitem__(
            "perturbation_kind", "NONE"), "缺少"),
    ],
)
def test_bad_profile_resolver_source_scope_budget_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏 profile、resolver、source、scope、预算和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredModalCourseError, match=message):
        read_authored_modal_seeds(bad)


def test_exists_and_forall_manifests_do_not_drift(tmp_path):
    """新增 modal 坐标和 handler 不得改变已关闭量词 artifact 身份。"""
    exists = compile_authored_exists_course(
        EXISTS_SAMPLE_PATH, tmp_path / "exists")
    forall = compile_authored_forall_course(
        FORALL_SAMPLE_PATH, tmp_path / "forall")
    assert exists.manifest.content_sha256() == (
        "d8910e29a82f7c6fc6df6c1c03c1ef5e19a56c411e2b690dc7124fbc69155964")
    assert forall.manifest.content_sha256() == (
        "db37efe6192806c9186bedcd96229dc13933f3892aede721957d16087e378015")


def test_float_noncanonical_existing_pack_and_legacy_are_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬 surface modal 或 child 直通。"""
    rows = _sample_values()
    rows[0]["max_resolver_calls"] = 1.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredModalCourseError, match="规范 JSON"):
        read_authored_modal_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"logic": {}}\n')
    with pytest.raises(AuthoredModalCourseError, match="规范 JSON"):
        read_authored_modal_seeds(bad_json)
    build = compile_authored_modal_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredModalCourseError, match="发布失败"):
        compile_authored_modal_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "surface.startswith",
            "token_seq",
            "expected_state ==",
            "child.state"}:
        assert token not in source
