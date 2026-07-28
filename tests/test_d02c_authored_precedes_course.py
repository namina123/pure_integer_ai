"""D-02C.6 AUTHORED_CC0_V1 typed PRECEDES/event-time 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_DIRECTION_UNKNOWN,
    EVENT_TIME_SAME,
    ResolvedEventTimeRelation,
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
from pure_integer_ai.experiments.event_time_runtime import (
    EventTimeEvidenceRequest,
)
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationProtocol,
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.ph2_authored_precedes_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_precedes_course,
    read_authored_precedes_seeds,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_EVENT_AFTER,
    RELATION_EVENT_BEFORE,
    RELATION_EVENT_SAME,
    RELATION_EVENT_UNKNOWN,
    REQUEST_EVENT_TIME_VERIFICATION,
    ROLE_EVENT_AFTER_OBJECT,
    ROLE_EVENT_AFTER_SUBJECT,
    ROLE_EVENT_BEFORE_OBJECT,
    ROLE_EVENT_BEFORE_SUBJECT,
    ROLE_EVENT_SAME_OBJECT,
    ROLE_EVENT_SAME_SUBJECT,
    ROLE_EVENT_UNKNOWN_OBJECT,
    ROLE_EVENT_UNKNOWN_SUBJECT,
    SCHEMA_EVENT_AFTER,
    SCHEMA_EVENT_BEFORE,
    SCHEMA_EVENT_SAME,
    SCHEMA_EVENT_UNKNOWN,
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


SAMPLE_PATH = Path("data/ph2/authored_relation_precedes_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_precedes_course.py"),
)
EXPECTED_PROFILE = {
    "EVENT_BEFORE": (
        RELATION_EVENT_BEFORE,
        SCHEMA_EVENT_BEFORE,
        ROLE_EVENT_BEFORE_SUBJECT,
        ROLE_EVENT_BEFORE_OBJECT,
        EVENT_TIME_BEFORE,
    ),
    "EVENT_AFTER": (
        RELATION_EVENT_AFTER,
        SCHEMA_EVENT_AFTER,
        ROLE_EVENT_AFTER_SUBJECT,
        ROLE_EVENT_AFTER_OBJECT,
        EVENT_TIME_AFTER,
    ),
    "EVENT_SAME": (
        RELATION_EVENT_SAME,
        SCHEMA_EVENT_SAME,
        ROLE_EVENT_SAME_SUBJECT,
        ROLE_EVENT_SAME_OBJECT,
        EVENT_TIME_SAME,
    ),
    "EVENT_UNKNOWN": (
        RELATION_EVENT_UNKNOWN,
        SCHEMA_EVENT_UNKNOWN,
        ROLE_EVENT_UNKNOWN_SUBJECT,
        ROLE_EVENT_UNKNOWN_OBJECT,
        EVENT_TIME_DIRECTION_UNKNOWN,
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


def _endpoint_by_role(seed):
    """按冻结 Role 坐标恢复 seed endpoint。"""
    endpoint_index = {item.endpoint_id: item for item in seed.endpoints}
    return {
        item.role_kind: endpoint_index[item.endpoint_id]
        for item in seed.bindings
    }


def test_sample_covers_four_directions_endpoint_kinds_and_order_boundaries():
    """before/after/same/unknown 齐全，并含 Event/Proposition 与两类混淆。"""
    seeds = read_authored_precedes_seeds(SAMPLE_PATH)
    assert len(seeds) == 13
    assert LICENSE_ID == "CC0-1.0"
    assert {item.relation_family for item in seeds} == set(EXPECTED_PROFILE)
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
        relation, schema, subject_role, object_role, _ = EXPECTED_PROFILE[
            seed.relation_family]
        assert seed.relation_kind == relation
        assert seed.schema_kind == schema
        assert seed.directionality == DIRECTION_FORWARD
        assert {item.role_kind for item in seed.bindings} == {
            subject_role, object_role}
        assert seed.consumer_request.request_kind == (
            REQUEST_EVENT_TIME_VERIFICATION)
        assert all(frozenset(item.allowed_object_kinds) == {
            OBJECT_EVENT, OBJECT_PROPOSITION} for item in seed.bindings)
        endpoint_kinds.update(item.object_kind for item in seed.endpoints)
    assert endpoint_kinds == {OBJECT_EVENT, OBJECT_PROPOSITION}


def test_direction_reversal_and_after_preserve_declared_event_time_axes():
    """方向翻转只换 Role filler，AFTER 保留独立 relation 和端点方向。"""
    seeds = {
        item.seed_id: item for item in read_authored_precedes_seeds(SAMPLE_PATH)
    }
    before = seeds["teacher-open-before-enter-v1"]
    reversed_seed = seeds["teacher-open-before-enter-reversed-v1"]
    after = seeds["teacher-enter-after-open-v1"]
    assert before.surface == reversed_seed.surface
    assert before.endpoints == reversed_seed.endpoints
    assert before.consumer_request == reversed_seed.consumer_request
    before_by_role = _endpoint_by_role(before)
    reversed_by_role = _endpoint_by_role(reversed_seed)
    assert reversed_by_role[ROLE_EVENT_BEFORE_SUBJECT] == (
        before_by_role[ROLE_EVENT_BEFORE_OBJECT])
    assert reversed_by_role[ROLE_EVENT_BEFORE_OBJECT] == (
        before_by_role[ROLE_EVENT_BEFORE_SUBJECT])
    assert {item.local_id for item in after.endpoints} == {
        item.local_id for item in before.endpoints}
    assert after.relation_kind != before.relation_kind
    assert after.relation_family == "EVENT_AFTER"


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，13/13/9/4 分账并保持 expected 私有。"""
    first = compile_authored_precedes_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_precedes_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.w_stages == ("W-06",)
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


def test_payload_roundtrips_event_evidence_resolver_and_verification_request(
        tmp_path):
    """恢复 relation schema、Evidence 路由、四方向 resolver 和独立 R-09 请求。"""
    build = compile_authored_precedes_course(SAMPLE_PATH, tmp_path)
    directions = set()
    dimension_keys = set()
    verifier_keys = set()
    hypothesis_keys = set()
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        candidate = payload["candidate_definition"]
        proposition = _identity(candidate["proposition_key"])
        assert proposition.object_kind == OBJECT_PROPOSITION
        bindings = []
        for value in candidate["role_bindings"]:
            role = _identity(value["role_key"])
            filler = _identity(value["filler_key"])
            binding = AtomicRoleBinding(role, filler, value["ordinal"])
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
        protocol = payload["event_time_protocol"]
        resolved = ResolvedEventTimeRelation(
            _identity(protocol["relation_key"]),
            protocol["direction"],
            tuple(protocol["detail_key"]),
        )
        assert resolved.relation == schema.relation
        directions.add(resolved.direction)
        dimension = ProtocolKey(tuple(protocol["dimension_key"]))
        verifier = ProtocolKey(tuple(protocol["verifier_key"]))
        verification_protocol = EventTimeVerificationProtocol(
            dimension, verifier)
        dimension_keys.add(verification_protocol.dimension.stable_key())
        verifier_keys.add(verification_protocol.verifier.stable_key())
        hypothesis_keys.add(tuple(protocol["hypothesis_kind_key"]))
        assert protocol["occurrence_order_consumed"] == 0
        assert protocol["structure_order_consumed"] == 0
        assert protocol["causes_effect"] == 0
        scope = ScopeIdentity.from_stable_key(tuple(protocol["scope_key"]))
        verification = EventTimeVerificationRequest(
            scope, (resolved.relation,))
        assert verification.relations == (schema.relation,)
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_EVENT_TIME_VERIFICATION
        assert request["relations"] == [protocol["relation_key"]]
        assert request["budget"] == {
            "max_evidence_requests": 20,
            "max_relations": 8,
        }
        filler_by_role = {
            item.role: item.filler for item in definition.canonical_bindings()
        }
        subject_role = _identity(protocol["subject_role_key"])
        object_role = _identity(protocol["object_role_key"])
        evidence_request = EventTimeEvidenceRequest(
            schema.relation,
            proposition,
            subject_role,
            0,
            filler_by_role[subject_role],
            object_role,
            0,
            filler_by_role[object_role],
            1,
        )
        candidate_endpoints = request["candidate_endpoints"]
        if observation.perturbation_kind == "DIRECTION_REVERSAL":
            assert _identity(candidate_endpoints["subject_key"]) != (
                evidence_request.subject)
            assert _identity(candidate_endpoints["object_key"]) != (
                evidence_request.object_identity)
        else:
            assert _identity(candidate_endpoints["subject_key"]) == (
                evidence_request.subject)
            assert _identity(candidate_endpoints["object_key"]) == (
                evidence_request.object_identity)
    assert directions == {
        EVENT_TIME_BEFORE,
        EVENT_TIME_AFTER,
        EVENT_TIME_SAME,
        EVENT_TIME_DIRECTION_UNKNOWN,
    }
    assert len(dimension_keys) == 1
    assert len(verifier_keys) == 1
    assert len(hypothesis_keys) == 1


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-06 视图和同 direction 修订可复核。"""
    build = compile_authored_precedes_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[9].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("relation_family", "PRECEDES"),
         "family"),
        (lambda rows: rows[0].__setitem__("relation_kind", 11), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "allowed_object_kinds", [17]), "slot 类型"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "object_kind", 4), "allowed_object_kinds"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 99),
         "Role profile"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 2), "consumer"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "object_endpoint_id", "open-door"), "不得相同"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[8].__setitem__(
            "supersedes_seed_id", rows[10]["seed_id"]), "更早"),
        (lambda rows: (
            rows[11].__setitem__("sample_role", "supersede"),
            rows[11].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[11].__setitem__("supersedes_seed_id", rows[2]["seed_id"]),
        ), "family/split/relation"),
        (lambda rows: rows[5].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_license_direction_profile_query_types_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、方向、Role、query、端点类型和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredRelationCourseError, match=message):
        read_authored_precedes_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_is_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬用旧 PRECEDES、token 或 CAUSES。"""
    rows = _sample_values()
    rows[0]["relation_kind"] = 10.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_precedes_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_precedes_seeds(bad_json)
    build = compile_authored_precedes_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredRelationCourseError, match="发布失败"):
        compile_authored_precedes_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_PRECEDES",
            "EDGE_CAUSES",
            "token_seq",
            "role_precedes"}:
        assert token not in source
    precedes_source = SOURCE_PATHS[-1].read_text(encoding="utf-8")
    for token in {
            "RELATION_CAUSES",
            "REQUEST_CAUSAL_VERIFICATION",
            "causal_protocol"}:
        assert token not in precedes_source
