"""D-02C.7 AUTHORED_CC0_V1 typed CAUSES 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.causal_execution import (
    CausalEndpointProtocol,
    causal_endpoints,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.experiments.causal_relation_runtime import (
    CausalVerificationProtocol,
)
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.ph2_authored_causes_course import (
    CAUSAL_EXECUTION_RULE_KIND,
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_causes_course,
    read_authored_causes_seeds,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
    authored_relation_rule_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_CAUSES,
    RELATION_EVENT_AFTER,
    RELATION_EVENT_BEFORE,
    RELATION_EVENT_SAME,
    RELATION_EVENT_UNKNOWN,
    REQUEST_CAUSAL_VERIFICATION,
    ROLE_CAUSE,
    ROLE_EFFECT,
    SCHEMA_CAUSES,
    SOURCE_KEY,
    AuthoredRelationCourseError,
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
from pure_integer_ai.experiments.verification_orchestration import ProtocolKey


SAMPLE_PATH = Path("data/ph2/authored_relation_causes_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_causes_course.py"),
)
TEMPORAL_RELATION_KINDS = (
    RELATION_EVENT_BEFORE,
    RELATION_EVENT_AFTER,
    RELATION_EVENT_SAME,
    RELATION_EVENT_UNKNOWN,
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


def _endpoint_by_role(seed):
    """按冻结 Role 坐标恢复 seed endpoint。"""
    endpoint_index = {item.endpoint_id: item for item in seed.endpoints}
    return {
        item.role_kind: endpoint_index[item.endpoint_id]
        for item in seed.bindings
    }


def _definition_and_schema(payload: dict):
    """从 Observation payload 恢复 typed 命题和 relation schema。"""
    candidate = payload["candidate_definition"]
    proposition = _identity(candidate["proposition_key"])
    bindings = []
    for value in candidate["role_bindings"]:
        binding = AtomicRoleBinding(
            _identity(value["role_key"]),
            _identity(value["filler_key"]),
            value["ordinal"],
        )
        assert binding.identity_for(proposition) == _identity(
            value["binding_key"])
        assert _identity(value["binding_key"]).object_kind == (
            OBJECT_ROLE_BINDING)
        bindings.append(binding)
    definition = AtomicPropositionDefinition(
        proposition,
        _identity(candidate["predicate_key"]),
        _identity(candidate["source_anchor_key"]),
        _identity(candidate["context_key"]),
        tuple(bindings),
    )
    schema_value = payload["relation_schema"]
    schema = RelationSchema(
        _identity(schema_value["schema_key"]),
        _identity(schema_value["relation_key"]),
        tuple(RelationSlotSchema(
            _identity(value["role_key"]),
            frozenset(value["allowed_object_kinds"]),
            value["min_count"],
            value["max_count"],
        ) for value in schema_value["slots"]),
    )
    schema.validate_definition(definition)
    return definition, schema


def test_sample_covers_typed_endpoints_independent_evidence_and_order():
    """CAUSES 覆盖独立证据、反事实边界、Event/Proposition 与恢复链。"""
    seeds = read_authored_causes_seeds(SAMPLE_PATH)
    assert len(seeds) == 14
    assert LICENSE_ID == "CC0-1.0"
    assert {item.relation_family for item in seeds} == {"CAUSES"}
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    assert any(
        item.perturbation_kind == "CONTENT_REPLACEMENT"
        and item.split == "held_out"
        for item in seeds
    )
    endpoint_kinds = set()
    for seed in seeds:
        assert seed.relation_kind == RELATION_CAUSES
        assert seed.schema_kind == SCHEMA_CAUSES
        assert seed.directionality == DIRECTION_FORWARD
        assert {item.role_kind for item in seed.bindings} == {
            ROLE_CAUSE, ROLE_EFFECT}
        assert seed.consumer_request.request_kind == (
            REQUEST_CAUSAL_VERIFICATION)
        assert all(frozenset(item.allowed_object_kinds) == {
            OBJECT_EVENT, OBJECT_PROPOSITION} for item in seed.bindings)
        endpoint_kinds.update(item.object_kind for item in seed.endpoints)
    assert endpoint_kinds == {OBJECT_EVENT, OBJECT_PROPOSITION}


def test_direction_reversal_only_swaps_cause_effect_role_fillers():
    """方向扰动只交换 Role filler，surface、端点和 consumer 保持不变。"""
    seeds = {
        item.seed_id: item for item in read_authored_causes_seeds(SAMPLE_PATH)
    }
    forward = seeds["teacher-switch-causes-light-v1"]
    reversed_seed = seeds["teacher-switch-light-reversed-v1"]
    assert forward.surface == reversed_seed.surface
    assert forward.endpoints == reversed_seed.endpoints
    assert forward.consumer_request == reversed_seed.consumer_request
    forward_by_role = _endpoint_by_role(forward)
    reversed_by_role = _endpoint_by_role(reversed_seed)
    assert reversed_by_role[ROLE_CAUSE] == forward_by_role[ROLE_EFFECT]
    assert reversed_by_role[ROLE_EFFECT] == forward_by_role[ROLE_CAUSE]


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，14/14/10/4 分账并保持 expected 私有。"""
    first = compile_authored_causes_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_causes_course(SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.w_stages == ("W-06",)
    assert first.validation.source_ref_count == 14
    assert first.validation.observation_count == 14
    assert first.validation.teacher_evidence_count == 10
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
    assert len(_records(first, RECORD_SOURCE_REF)) == 14
    assert len(observations) == 14
    assert len(_records(first, RECORD_TEACHER_EVIDENCE)) == 10
    assert len(_records(first, RECORD_EVALUATOR_LABEL)) == 4
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_payload_roundtrips_causal_protocol_and_separates_event_time(tmp_path):
    """恢复 CAUSES endpoint/protocol，时间只作必要条件且不写回因果。"""
    build = compile_authored_causes_course(SAMPLE_PATH, tmp_path)
    expected_temporal = {
        authored_relation_identity(kind) for kind in TEMPORAL_RELATION_KINDS}
    dimension_keys = set()
    verifier_keys = set()
    target_keys = set()
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        definition, schema = _definition_and_schema(payload)
        protocol = payload["causal_protocol"]
        endpoint_protocol = CausalEndpointProtocol(
            _identity(protocol["relation_key"]),
            _identity(protocol["cause_role_key"]),
            _identity(protocol["effect_role_key"]),
            _identity(protocol["execution_instruction_key"]),
        )
        assert endpoint_protocol.relation == schema.relation
        assert endpoint_protocol.cause_role == authored_relation_role_identity(
            ROLE_CAUSE)
        assert endpoint_protocol.effect_role == authored_relation_role_identity(
            ROLE_EFFECT)
        assert endpoint_protocol.execution_instruction == (
            authored_relation_rule_identity(CAUSAL_EXECUTION_RULE_KIND))
        cause, effect = causal_endpoints(definition, endpoint_protocol)
        verification = CausalVerificationProtocol(
            ProtocolKey(tuple(protocol["dimension_key"])),
            ProtocolKey(tuple(protocol["verifier_key"])),
            ProtocolKey(tuple(protocol["evidence_target_kind_key"])),
        )
        dimension_keys.add(verification.dimension.stable_key())
        verifier_keys.add(verification.verifier.stable_key())
        target_keys.add(verification.evidence_target_kind.stable_key())
        scope = ScopeIdentity.from_stable_key(tuple(protocol["scope_key"]))
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_CAUSAL_VERIFICATION
        temporal_relations = tuple(
            _identity(value) for value in request["temporal_relations"])
        temporal = EventTimeVerificationRequest(scope, temporal_relations)
        assert set(temporal.relations) == expected_temporal
        assert endpoint_protocol.relation not in temporal.relations
        assert request["causal_relation_key"] == protocol["relation_key"]
        candidates = request["candidate_endpoints"]
        if observation.perturbation_kind == "DIRECTION_REVERSAL":
            assert _identity(candidates["cause_key"]) == effect
            assert _identity(candidates["effect_key"]) == cause
        else:
            assert _identity(candidates["cause_key"]) == cause
            assert _identity(candidates["effect_key"]) == effect
        assert request["budget"] == {
            "max_evidence_requests": 20,
            "max_relations": 8,
            "max_witness_inputs": 8,
        }
        assert protocol["independent_witness_required"] == 1
        assert protocol["forming_source_reusable_as_witness"] == 0
        assert protocol["temporal_support_sufficient"] == 0
        assert protocol["precedence_implies_causation"] == 0
        assert protocol["causal_implies_event_time_fact"] == 0
        assert protocol["counterfactual_verdict_claimed"] == 0
        assert protocol["occurrence_order_consumed"] == 0
        assert protocol["structure_order_consumed"] == 0
    assert len(dimension_keys) == 1
    assert len(verifier_keys) == 1
    assert len(target_keys) == 1


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-06 视图和同 relation 修订可复核。"""
    build = compile_authored_causes_course(SAMPLE_PATH, tmp_path)
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
        train, teachers, (), current_stage="W-06", view_kind="training")
    validate_stage_visibility(
        held_out, (), evaluators, current_stage="W-06", view_kind="evaluation")
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
        (lambda rows: rows[10].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("relation_family", "PRECEDES"),
         "profile"),
        (lambda rows: rows[0].__setitem__("relation_kind", 10), "profile"),
        (lambda rows: rows[0].__setitem__("directionality", 1), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "allowed_object_kinds", [4]), "allowed_object_kinds"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "object_kind", 4), "allowed_object_kinds"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 99),
         "Role profile"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 6), "causal"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "effect_endpoint_id", "press-switch"), "不得相同"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "max_witness_inputs", 0), "max_witness_inputs"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[9].__setitem__(
            "supersedes_seed_id", rows[10]["seed_id"]), "更早"),
        (lambda rows: (
            rows[11].__setitem__("sample_role", "supersede"),
            rows[11].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[11].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "family/split/relation"),
        (lambda rows: rows[7].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_license_profile_query_types_budget_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、Role、consumer、预算、span 与恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredRelationCourseError, match=message):
        read_authored_causes_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_are_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬旧边、token 序或 reward cue。"""
    rows = _sample_values()
    rows[0]["relation_kind"] = 14.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_causes_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_causes_seeds(bad_json)
    build = compile_authored_causes_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredRelationCourseError, match="发布失败"):
        compile_authored_causes_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_CAUSES",
            "build_causes_edges",
            "token_seq",
            "order_index",
            "causes_reward"}:
        assert token not in source
