"""D-02C.4 AUTHORED_CC0_V1 typed MEREOLOGY 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.mereology_relation import (
    MereologyBudget,
    MereologyPattern,
    MereologyProtocol,
    MereologyRelationProtocol,
    MereologyStatement,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    InverseRule,
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.experiments.ph2_authored_mereology_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_mereology_course,
    read_authored_mereology_seeds,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_HAS_PART,
    RELATION_PART_OF,
    REQUEST_MEREOLOGY_QUERY,
    ROLE_HAS_PART_PART,
    ROLE_HAS_PART_WHOLE,
    ROLE_PART_OF_PART,
    ROLE_PART_OF_WHOLE,
    SCHEMA_HAS_PART,
    SCHEMA_PART_OF,
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


SAMPLE_PATH = Path(
    "data/ph2/authored_relation_mereology_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_mereology_course.py"),
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


def test_sample_covers_variants_inverse_direction_and_relation_confusion():
    """PART_OF/HAS_PART 各有正例，逆向、类别混淆和恢复均独立。"""
    seeds = read_authored_mereology_seeds(SAMPLE_PATH)
    assert len(seeds) == 11
    assert LICENSE_ID == "CC0-1.0"
    assert {item.relation_family for item in seeds} == {
        "PART_OF", "HAS_PART"}
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.perturbation_kind for item in seeds})
    assert any(
        item.perturbation_kind == "CONTENT_REPLACEMENT"
        and item.split == "held_out"
        for item in seeds
    )
    for seed in seeds:
        assert seed.directionality == DIRECTION_FORWARD
        assert seed.consumer_request.request_kind == REQUEST_MEREOLOGY_QUERY
        if seed.relation_family == "PART_OF":
            assert seed.relation_kind == RELATION_PART_OF
            assert seed.schema_kind == SCHEMA_PART_OF
            assert {item.role_kind for item in seed.bindings} == {
                ROLE_PART_OF_PART, ROLE_PART_OF_WHOLE}
        else:
            assert seed.relation_kind == RELATION_HAS_PART
            assert seed.schema_kind == SCHEMA_HAS_PART
            assert {item.role_kind for item in seed.bindings} == {
                ROLE_HAS_PART_PART, ROLE_HAS_PART_WHOLE}
        assert {item.object_kind for item in seed.endpoints} == {OBJECT_ENTITY}
        assert all(item.allowed_object_kinds == (OBJECT_ENTITY,)
                   for item in seed.bindings)


def test_direction_and_inverse_relation_break_only_declared_axes():
    """方向翻转只换 Role filler，inverse mismatch 只换 relation/schema。"""
    seeds = {
        item.seed_id: item for item in read_authored_mereology_seeds(SAMPLE_PATH)
    }
    supported = seeds["teacher-wheel-part-of-car-v1"]
    reversed_seed = seeds["teacher-wheel-part-reversed-v1"]
    assert supported.surface == reversed_seed.surface
    assert supported.endpoints == reversed_seed.endpoints
    assert supported.consumer_request == reversed_seed.consumer_request
    supported_by_role = _endpoint_by_role(supported)
    reversed_by_role = _endpoint_by_role(reversed_seed)
    assert reversed_by_role[ROLE_PART_OF_PART] == (
        supported_by_role[ROLE_PART_OF_WHOLE])
    assert reversed_by_role[ROLE_PART_OF_WHOLE] == (
        supported_by_role[ROLE_PART_OF_PART])

    has_part = seeds["teacher-car-has-wheel-v1"]
    inverse_mismatch = seeds["teacher-car-contains-as-part-of-v1"]
    assert has_part.surface == inverse_mismatch.surface
    assert has_part.endpoints == inverse_mismatch.endpoints
    assert has_part.consumer_request == inverse_mismatch.consumer_request
    assert has_part.relation_kind != inverse_mismatch.relation_kind
    assert has_part.schema_kind != inverse_mismatch.schema_kind
    has_values = sorted(
        item.local_id for item in _endpoint_by_role(has_part).values())
    mismatch_values = sorted(
        item.local_id for item in _endpoint_by_role(inverse_mismatch).values())
    assert has_values == mismatch_values


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，11/11/7/4 分账并保持 expected 私有。"""
    first = compile_authored_mereology_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_mereology_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.w_stages == ("W-06",)
    assert first.validation.source_ref_count == 11
    assert first.validation.observation_count == 11
    assert first.validation.teacher_evidence_count == 7
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
    assert len(_records(first, RECORD_SOURCE_REF)) == 11
    assert len(observations) == 11
    assert len(_records(first, RECORD_TEACHER_EVIDENCE)) == 7
    assert len(_records(first, RECORD_EVALUATOR_LABEL)) == 4
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_payload_roundtrips_mereology_protocol_inverse_query_and_no_default_rule(
        tmp_path):
    """恢复两 relation、显式 inverse、statement/query，且无默认传递。"""
    build = compile_authored_mereology_course(SAMPLE_PATH, tmp_path)
    protocols = {}
    inverse_rules = set()
    statements = []
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
        protocol_value = payload["mereology_protocol"]
        relation_protocol = MereologyRelationProtocol(
            schema,
            _identity(protocol_value["canonical_part_role_key"]),
            _identity(protocol_value["canonical_whole_role_key"]),
        )
        protocols[relation_protocol.relation] = relation_protocol
        assert protocol_value["transitive_rules"] == []
        assert protocol_value["composition_rules"] == []
        assert protocol_value["irreflexive_rules"] == []
        assert len(protocol_value["inverse_rules"]) == 1
        inverse_value = protocol_value["inverse_rules"][0]
        inverse_rules.add(InverseRule(
            _identity(inverse_value["rule_key"]),
            _identity(inverse_value["premise_relation_key"]),
            _identity(inverse_value["premise_left_role_key"]),
            _identity(inverse_value["premise_right_role_key"]),
            _identity(inverse_value["result_relation_key"]),
            _identity(inverse_value["result_left_role_key"]),
            _identity(inverse_value["result_right_role_key"]),
        ))
        filler_by_role = {
            item.role: item.filler for item in definition.canonical_bindings()
        }
        statement = MereologyStatement(
            relation_protocol.relation,
            filler_by_role[relation_protocol.part_role],
            filler_by_role[relation_protocol.whole_role],
        )
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_MEREOLOGY_QUERY
        pattern = MereologyPattern(
            _identity(request["pattern"]["relation_key"]),
            _identity(request["pattern"]["part_key"]),
            _identity(request["pattern"]["whole_key"]),
        )
        budget = MereologyBudget(
            request["budget"]["max_direct_facts"],
            request["budget"]["max_closure_statements"],
            request["budget"]["max_rule_applications"],
            request["budget"]["max_options"],
        )
        assert budget.stable_key() == (20, 40, 40, 20)
        if observation.perturbation_kind == "DIRECTION_REVERSAL":
            assert not pattern.matches(statement)
        else:
            assert pattern.matches(statement)
        statements.append(statement)
    assert len(protocols) == 2
    assert len(inverse_rules) == 1
    family = MereologyProtocol(
        tuple(protocols.values()),
        inverse_rules=tuple(inverse_rules),
    )
    assert family.transitive_rules == ()
    assert family.composition_rules == ()
    assert family.irreflexive_rules == ()
    for statement in statements:
        family.validate_statement(statement)


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-06 视图和同 relation 修订可复核。"""
    build = compile_authored_mereology_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[7].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("relation_family", "MEMBER"),
         "family"),
        (lambda rows: rows[0].__setitem__("relation_kind", 7), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "allowed_object_kinds", [4]), "allowed_object_kinds"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "object_kind", 18), "allowed_object_kinds"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 99),
         "Role profile"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 2), "consumer"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "part_endpoint_id", "car"), "不得相同"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "whole_endpoint_id", "wheel"), "不得相同"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[6].__setitem__(
            "supersedes_seed_id", rows[8]["seed_id"]), "更早"),
        (lambda rows: (
            rows[9].__setitem__("sample_role", "supersede"),
            rows[9].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[9].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "family/split/relation"),
        (lambda rows: rows[9].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_license_profile_query_types_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、relation、Role、query、类型和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredRelationCourseError, match=message):
        read_authored_mereology_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_is_fail_closed(tmp_path):
    """float/非规范/覆盖失败，课程源码不搬用旧 MEREOLOGY 入口。"""
    rows = _sample_values()
    rows[0]["relation_kind"] = 6.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_mereology_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_mereology_seeds(bad_json)
    build = compile_authored_mereology_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredRelationCourseError, match="发布失败"):
        compile_authored_mereology_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_MEREOLOGY",
            "mereology_facts",
            "EDGE_IS_A",
            "ancestor_map"}:
        assert token not in source
