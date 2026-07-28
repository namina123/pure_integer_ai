"""D-02C.1 AUTHORED_CC0_V1 typed PURE_ALIAS/REFERS 资料包 T0。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
    SameKindConstraint,
)
from pure_integer_ai.experiments.ph2_authored_alias_refers_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_alias_refers_course,
    read_authored_alias_refers_seeds,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    DIRECTION_SYMMETRIC,
    LICENSE_ID,
    RELATION_PURE_ALIAS,
    RELATION_REFERS,
    REQUEST_REFERENCE_RESOLUTION,
    ROLE_ALIAS_LEFT,
    ROLE_ALIAS_RIGHT,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
    SCHEMA_PURE_ALIAS,
    SCHEMA_REFERS,
    SOURCE_KEY,
    AuthoredRelationCourseError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
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
    "data/ph2/authored_relation_alias_refers_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_alias_refers_course.py"),
)


def _sample_values() -> list[dict]:
    """读取仓库 sample 为独立可修改 JSON object 列表。"""
    return [json.loads(line) for line in SAMPLE_PATH.read_text(
        encoding="utf-8").splitlines()]


def _write_values(path: Path, values: list[dict]) -> None:
    """按 D-02 规范 JSONL writer 写 parser 负例。"""
    path.write_bytes(b"".join(canonical_json_line(value) for value in values))


def _write_json_with_float(path: Path, values: list[dict]) -> None:
    """绕过合同 writer，落一个只因 float 非法的输入。"""
    path.write_bytes(b"".join(
        (json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8")
        for row in values
    ))


def _records_by_kind(build, kind: str):
    """读取 manifest 中指定 record kind 的全部物理文件。"""
    records = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            records.extend(read_record_artifact(build.pack_root, identity))
    return tuple(records)


def _identity(value) -> ObjectIdentity:
    """从 payload 的严格整数列表恢复一等对象身份。"""
    assert isinstance(value, list)
    assert all(type(item) is int for item in value)
    return ObjectIdentity.from_stable_key(tuple(value))


def _all_keys(value) -> set[str]:
    """递归收集 JSON object key，确认新 pack 没有降级 edge/subtype 字段。"""
    if isinstance(value, dict):
        return set(value) | {
            key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_sample_covers_both_relations_consumer_recovery_and_breakage():
    """两类 relation 均有正例，并覆盖 held-out 内容替换、冲突和恢复。"""
    seeds = read_authored_alias_refers_seeds(SAMPLE_PATH)
    assert len(seeds) == 9
    assert LICENSE_ID == "CC0-1.0"
    assert {seed.relation_family for seed in seeds} == {"PURE_ALIAS", "REFERS"}
    assert {seed.sample_role for seed in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        seed.perturbation_kind for seed in seeds})
    assert any(
        seed.perturbation_kind == "CONTENT_REPLACEMENT"
        and seed.split == "held_out"
        for seed in seeds
    )
    teacher_families = {seed.family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_families = {
        seed.family for seed in seeds if seed.label_owner == "evaluator"}
    teacher_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "evaluator"}
    assert teacher_families.isdisjoint(evaluator_families)
    assert teacher_templates.isdisjoint(evaluator_templates)
    for seed in seeds:
        request = seed.consumer_request
        assert request.request_kind == REQUEST_REFERENCE_RESOLUTION
        AliasRouteSearchBudget(
            request.max_facts, request.max_states, request.max_routes)


def test_profile_uses_typed_alias_symmetry_and_directional_refers_roles():
    """PURE_ALIAS 与 REFERS 使用不同整数 relation/schema/Role 和方向合同。"""
    seeds = read_authored_alias_refers_seeds(SAMPLE_PATH)
    for seed in seeds:
        roles = {item.role_kind for item in seed.bindings}
        if seed.relation_family == "PURE_ALIAS":
            assert seed.relation_kind == RELATION_PURE_ALIAS
            assert seed.schema_kind == SCHEMA_PURE_ALIAS
            assert seed.directionality == DIRECTION_SYMMETRIC
            assert roles == {ROLE_ALIAS_LEFT, ROLE_ALIAS_RIGHT}
            assert len({item.object_kind for item in seed.endpoints}) == 1
        else:
            assert seed.relation_kind == RELATION_REFERS
            assert seed.schema_kind == SCHEMA_REFERS
            assert seed.directionality == DIRECTION_FORWARD
            assert roles == {ROLE_REFERS_FROM, ROLE_REFERS_TO}

    by_id = {seed.seed_id: seed for seed in seeds}
    forward = by_id["teacher-pronoun-refers-v1"]
    reversed_seed = by_id["teacher-pronoun-reversed-v1"]
    assert forward.surface == reversed_seed.surface
    assert forward.endpoints == reversed_seed.endpoints
    assert forward.consumer_request == reversed_seed.consumer_request
    assert {
        item.role_kind: item.endpoint_id for item in forward.bindings
    } != {
        item.role_kind: item.endpoint_id for item in reversed_seed.bindings
    }
    revised = by_id["teacher-pronoun-refers-v2"]
    assert revised.supersedes_seed_id == forward.seed_id
    assert revised.perturbation_kind == "PARSER_REVISION"


def test_compiler_writes_bit_identical_owner_separated_pack(tmp_path):
    """两目录 bit-identical，9/9/5/4 分账且 Observation 不含 expected。"""
    first = compile_authored_alias_refers_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_alias_refers_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.splits == ("train", "held_out")
    assert first.manifest.w_stages == ("W-06",)
    assert first.validation.source_ref_count == 9
    assert first.validation.observation_count == 9
    assert first.validation.teacher_evidence_count == 5
    assert first.validation.evaluator_label_count == 4
    assert first.validation.source_cluster_count == 2
    assert read_artifact_manifest(first.pack_root / "manifest.json") == (
        first.manifest)
    assert (first.pack_root / "manifest.json").read_bytes() == (
        second.pack_root / "manifest.json").read_bytes()
    for first_file, second_file in zip(first.manifest.files, second.manifest.files):
        assert first_file == second_file
        assert (first.pack_root / first_file.relative_path).read_bytes() == (
            second.pack_root / second_file.relative_path).read_bytes()

    sources = _records_by_kind(first, RECORD_SOURCE_REF)
    observations = _records_by_kind(first, RECORD_OBSERVATION)
    teachers = _records_by_kind(first, RECORD_TEACHER_EVIDENCE)
    evaluators = _records_by_kind(first, RECORD_EVALUATOR_LABEL)
    assert all(isinstance(item, SourceRefRecord) for item in sources)
    assert all(isinstance(item, ObservationRecord) for item in observations)
    assert all(isinstance(item, TeacherEvidenceRecord) for item in teachers)
    assert all(isinstance(item, EvaluatorLabelRecord) for item in evaluators)
    assert (len(sources), len(observations), len(teachers), len(evaluators)) == (
        9, 9, 5, 4)
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload
        assert {"edge_type", "subtype"}.isdisjoint(_all_keys(payload))


def test_payload_roundtrips_relation_schema_proposition_and_consumer_budget(
        tmp_path):
    """每条 Observation 可重建现役 RelationSchema、命题和 consumer 预算。"""
    build = compile_authored_alias_refers_course(SAMPLE_PATH, tmp_path)
    for observation in _records_by_kind(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        candidate = payload["candidate_definition"]
        proposition = _identity(candidate["proposition_key"])
        predicate = _identity(candidate["predicate_key"])
        source_anchor = _identity(candidate["source_anchor_key"])
        context = _identity(candidate["context_key"])
        assert proposition.object_kind == OBJECT_PROPOSITION
        assert predicate.object_kind == OBJECT_CONCEPT
        assert source_anchor.object_kind == OBJECT_OCCURRENCE
        assert context.object_kind == OBJECT_CONTEXT_SCOPE

        bindings = []
        for value in candidate["role_bindings"]:
            role = _identity(value["role_key"])
            filler = _identity(value["filler_key"])
            binding_identity = _identity(value["binding_key"])
            binding = AtomicRoleBinding(role, filler, value["ordinal"])
            assert role.object_kind == OBJECT_ROLE
            assert binding_identity.object_kind == OBJECT_ROLE_BINDING
            assert binding.identity_for(proposition) == binding_identity
            bindings.append(binding)
        definition = AtomicPropositionDefinition(
            proposition, predicate, source_anchor, context, tuple(bindings))

        schema_value = payload["relation_schema"]
        schema_identity = _identity(schema_value["schema_key"])
        relation_identity = _identity(schema_value["relation_key"])
        assert schema_identity.object_kind == OBJECT_STRUCTURE_CONCEPT
        slots = tuple(
            RelationSlotSchema(
                _identity(value["role_key"]),
                frozenset(value["allowed_object_kinds"]),
                value["min_count"],
                value["max_count"],
            )
            for value in schema_value["slots"]
        )
        constraints = tuple(
            SameKindConstraint(
                _identity(value["constraint_key"]),
                tuple(_identity(key) for key in value["role_keys"]),
            )
            for value in schema_value["constraints"]
        )
        schema = RelationSchema(
            schema_identity, relation_identity, slots, constraints)
        assert schema.validate_definition(definition) == definition
        if payload["directionality"] == DIRECTION_SYMMETRIC:
            assert len(schema.same_kind_constraints) == 1
        else:
            assert schema.same_kind_constraints == ()

        request = payload["consumer_request"]
        origin = _identity(request["origin_key"])
        endpoint_keys = {
            tuple(value["endpoint_key"]) for value in payload["endpoints"]}
        assert origin.stable_key() in endpoint_keys
        assert request["request_kind"] == REQUEST_REFERENCE_RESOLUTION
        budget = request["budget"]
        AliasRouteSearchBudget(
            budget["max_facts"], budget["max_states"], budget["max_routes"])


def test_split_clusters_stage_views_and_supersede_are_auditable(tmp_path):
    """train/held-out 来源簇和 owner 隔离，parser revision 只指向更早同类记录。"""
    build = compile_authored_alias_refers_course(SAMPLE_PATH, tmp_path)
    sources = _records_by_kind(build, RECORD_SOURCE_REF)
    observations = _records_by_kind(build, RECORD_OBSERVATION)
    teachers = _records_by_kind(build, RECORD_TEACHER_EVIDENCE)
    evaluators = _records_by_kind(build, RECORD_EVALUATOR_LABEL)
    source_index = {item.stable_key: item for item in sources}
    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(item for item in observations if item.split == "held_out")
    train_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in train
    }
    held_clusters = {
        source_index[item.source_ref_key].source_cluster_key for item in held_out
    }
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
        (lambda rows: rows[1].__setitem__("seed_id", rows[0]["seed_id"]), "重复"),
        (lambda rows: rows[5].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("relation_family", "LEGACY_ALIAS"),
         "family"),
        (lambda rows: rows[0].__setitem__("relation_kind", RELATION_REFERS),
         "profile"),
        (lambda rows: rows[0].__setitem__("schema_kind", SCHEMA_REFERS),
         "profile"),
        (lambda rows: rows[0].__setitem__("directionality", DIRECTION_FORWARD),
         "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 99),
         "Role profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "role_registry", "LEGACY_ROLE"), "Role registry"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__("object_kind", 20),
         "object kind"),
        (lambda rows: rows[1]["endpoints"][0].__setitem__("local_id", 1),
         "不得伪造"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "endpoint_id", "missing"), "未知 endpoint"),
        (lambda rows: rows[0]["bindings"][1].__setitem__("role_kind", 1),
         "slot 重复"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "origin_endpoint_id", "missing"), "consumer origin"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 99), "request kind"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "target_object_kinds", [999]), "非权威"),
        (lambda rows: (
            rows[0].__setitem__("sample_role", "supersede"),
            rows[0].__setitem__("supersedes_seed_id", rows[1]["seed_id"]),
        ), "更早"),
        (lambda rows: rows[4].__setitem__(
            "supersedes_seed_id", rows[0]["seed_id"]), "family/split/relation"),
    ],
)
def test_bad_license_profile_identity_consumer_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、profile、endpoint、consumer 和恢复链均不能进入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredRelationCourseError, match=message):
        read_authored_alias_refers_seeds(bad)


def test_float_noncanonical_json_and_existing_pack_fail_closed(tmp_path):
    """float、非规范 JSON 和同版本覆盖均 fail closed。"""
    rows = _sample_values()
    rows[0]["relation_kind"] = 1.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_alias_refers_seeds(bad_float)

    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_alias_refers_seeds(bad_json)

    build = compile_authored_alias_refers_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredRelationCourseError, match="发布失败"):
        compile_authored_alias_refers_course(
            SAMPLE_PATH, tmp_path / "release")


def test_new_course_does_not_import_or_emit_legacy_refers_edge_contract():
    """D-02C 只输出 typed relation，不搬用 legacy edge/subtype/surface loader。"""
    forbidden = {
        "EDGE_REFERS_TO",
        "SUBTYPE_PURE_ALIAS",
        "build_refers_stable_edge",
        "alias_facts",
    }
    strings = set()
    source_text = ""
    for path in SOURCE_PATHS:
        source = path.read_text(encoding="utf-8")
        source_text += source
        strings.update(
            node.value for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    assert forbidden.isdisjoint(strings)
    assert all(token not in source_text for token in forbidden)
    sample_fragments = {
        endpoint["surface_fragment"]
        for row in _sample_values()
        for endpoint in row["endpoints"]
    }
    assert sample_fragments.isdisjoint(strings)
