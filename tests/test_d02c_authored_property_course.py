"""D-02C.3 AUTHORED_CC0_V1 六维 typed PROPERTY 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.property_relation import (
    MappingPropertyIntensityResolver,
    PropertyClaim,
    PropertyPattern,
    PropertyQueryBudget,
    PropertyRelationProtocol,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.crosscut.integer.valtypes import Rational
from pure_integer_ai.experiments.ph2_authored_property_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_property_course,
    read_authored_property_seeds,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_PROPERTY,
    REQUEST_PROPERTY_SELECTION,
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_INTENSITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_VALUE,
    SCHEMA_PROPERTY,
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
    "data/ph2/authored_relation_property_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_property_course.py"),
)
PROPERTY_ROLES = (
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_VALUE,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_INTENSITY,
)
EXPECTED_ALLOWED = (
    frozenset({OBJECT_ENTITY}),
    frozenset({OBJECT_CONCEPT}),
    frozenset({OBJECT_CONCEPT, OBJECT_ENTITY}),
    frozenset({OBJECT_CONCEPT}),
    frozenset({OBJECT_CONCEPT}),
    frozenset({OBJECT_CONCEPT}),
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
    endpoint_index = {
        item.endpoint_id: item for item in seed.relation.endpoints
    }
    return {
        item.role_kind: endpoint_index[item.endpoint_id]
        for item in seed.relation.bindings
    }


def test_sample_covers_six_roles_rational_query_and_recovery():
    """PROPERTY 有六维正例、两种 value 类型、独立 query 和完整扰动。"""
    seeds = read_authored_property_seeds(SAMPLE_PATH)
    assert len(seeds) == 11
    assert LICENSE_ID == "CC0-1.0"
    assert {item.relation.relation_family for item in seeds} == {"PROPERTY"}
    assert {item.relation.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        item.relation.perturbation_kind for item in seeds})
    assert any(
        item.relation.perturbation_kind == "CONTENT_REPLACEMENT"
        and item.relation.split == "held_out"
        for item in seeds
    )
    value_kinds = set()
    for seed in seeds:
        relation = seed.relation
        assert relation.relation_kind == RELATION_PROPERTY
        assert relation.schema_kind == SCHEMA_PROPERTY
        assert relation.directionality == DIRECTION_FORWARD
        assert relation.consumer_request.request_kind == (
            REQUEST_PROPERTY_SELECTION)
        binding_by_role = {
            item.role_kind: item for item in relation.bindings
        }
        assert set(binding_by_role) == set(PROPERTY_ROLES)
        assert tuple(
            frozenset(binding_by_role[role].allowed_object_kinds)
            for role in PROPERTY_ROLES
        ) == EXPECTED_ALLOWED
        endpoint_by_role = _endpoint_by_role(seed)
        value_kinds.add(endpoint_by_role[ROLE_PROPERTY_VALUE].object_kind)
        assert endpoint_by_role[ROLE_PROPERTY_SUBJECT].object_kind == (
            OBJECT_ENTITY)
        assert endpoint_by_role[ROLE_PROPERTY_ATTRIBUTE].object_kind == (
            OBJECT_CONCEPT)
        assert endpoint_by_role[ROLE_PROPERTY_INTENSITY].object_kind == (
            OBJECT_CONCEPT)
        assert seed.intensity_num > 0
        assert seed.intensity_den > 0
    assert value_kinds == {OBJECT_CONCEPT, OBJECT_ENTITY}


def test_role_value_and_intensity_mutations_break_only_declared_axes():
    """Role 交换、value 替换和强度替换分别保持其他 seed 维度。"""
    seeds = {
        item.relation.seed_id: item
        for item in read_authored_property_seeds(SAMPLE_PATH)
    }
    supported = seeds["teacher-maple-red-v1"]
    swapped = seeds["teacher-maple-role-mismatch-v1"]
    assert supported.relation.surface == swapped.relation.surface
    assert supported.relation.endpoints == swapped.relation.endpoints
    assert supported.intensity_num == swapped.intensity_num == 1
    assert supported.intensity_den == swapped.intensity_den == 1
    supported_by_role = _endpoint_by_role(supported)
    swapped_by_role = _endpoint_by_role(swapped)
    assert swapped_by_role[ROLE_PROPERTY_ATTRIBUTE] == (
        supported_by_role[ROLE_PROPERTY_VALUE])
    assert swapped_by_role[ROLE_PROPERTY_VALUE] == (
        supported_by_role[ROLE_PROPERTY_ATTRIBUTE])
    for role in (
            ROLE_PROPERTY_SUBJECT,
            ROLE_PROPERTY_POLARITY,
            ROLE_PROPERTY_MODALITY,
            ROLE_PROPERTY_INTENSITY):
        assert swapped_by_role[role] == supported_by_role[role]

    high = seeds["teacher-maple-high-intensity-v1"]
    high_by_role = _endpoint_by_role(high)
    for role in PROPERTY_ROLES[:-1]:
        assert high_by_role[role].local_id == supported_by_role[role].local_id
    assert high_by_role[ROLE_PROPERTY_INTENSITY].local_id != (
        supported_by_role[ROLE_PROPERTY_INTENSITY].local_id)
    assert (high.intensity_num, high.intensity_den) == (3, 2)

    blue = seeds["teacher-maple-blue-v1"]
    blue_by_role = _endpoint_by_role(blue)
    for role in PROPERTY_ROLES:
        if role == ROLE_PROPERTY_VALUE:
            assert blue_by_role[role].local_id != supported_by_role[role].local_id
        else:
            assert blue_by_role[role].local_id == supported_by_role[role].local_id


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，11/11/7/4 分账并保持 expected 私有。"""
    first = compile_authored_property_course(SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_property_course(SAMPLE_PATH, tmp_path / "second")
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


def test_payload_roundtrips_property_protocol_claim_rational_and_query(
        tmp_path):
    """每条 payload 可恢复六 Role protocol、claim、Rational 和查询预算。"""
    build = compile_authored_property_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        candidate = payload["candidate_definition"]
        proposition = _identity(candidate["proposition_key"])
        predicate = _identity(candidate["predicate_key"])
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
            predicate,
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
        ordered_roles = tuple(
            slot.role for slot in sorted(
                schema.slots,
                key=lambda item: item.role.components[-1],
            )
        )
        assert tuple(role.components[-1] for role in ordered_roles) == (
            PROPERTY_ROLES)
        protocol = PropertyRelationProtocol(schema, *ordered_roles)
        filler_by_role = {
            item.role: item.filler for item in definition.canonical_bindings()
        }
        claim = PropertyClaim(*tuple(
            filler_by_role[role] for role in protocol.roles()))
        rational = payload["rational_role_values"]
        assert len(rational) == 1
        assert _identity(rational[0]["role_key"]) == protocol.intensity_role
        assert _identity(rational[0]["filler_key"]) == claim.intensity
        resolver = MappingPropertyIntensityResolver(((
            claim.intensity,
            Rational(rational[0]["num"], rational[0]["den"]),
        ),))
        assert resolver.resolve(claim.intensity) == Rational(
            rational[0]["num"], rational[0]["den"])
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_PROPERTY_SELECTION
        pattern = PropertyPattern(
            _identity(request["pattern"]["subject_key"]),
            _identity(request["pattern"]["attribute_key"]),
        )
        budget = PropertyQueryBudget(
            request["budget"]["max_direct_facts"],
            request["budget"]["max_options"],
        )
        assert budget.stable_key() == (20, 10)
        if observation.perturbation_kind == "ROLE_MISMATCH":
            assert not pattern.matches(claim)
        else:
            assert pattern.matches(claim)


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-06 视图和同 relation 修订可复核。"""
    build = compile_authored_property_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[0].__setitem__("relation_kind", 4), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "allowed_object_kinds", [4]), "allowed_object_kinds"),
        (lambda rows: rows[0]["bindings"].pop(), "覆盖全部 endpoint"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 2), "非 PROPERTY"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "subject_endpoint_id", "red"), "subject 锚"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "attribute_endpoint_id", "red"), "attribute 锚"),
        (lambda rows: rows[0]["property_intensity"].__setitem__("den", 0),
         "intensity"),
        (lambda rows: rows[0]["property_intensity"].__setitem__("extra", 1),
         "字段集合"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[6].__setitem__(
            "supersedes_seed_id", rows[8]["seed_id"]), "更早"),
        (lambda rows: (
            rows[9].__setitem__("sample_role", "supersede"),
            rows[9].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[9].__setitem__("supersedes_seed_id", rows[0]["seed_id"]),
        ), "family/split/relation"),
    ],
)
def test_bad_license_profile_query_rational_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、六维 profile、query、Rational 和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredRelationCourseError, match=message):
        read_authored_property_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_is_fail_closed(tmp_path):
    """float/非规范/覆盖失败，课程源码不搬用旧 PROPERTY 写边入口。"""
    rows = _sample_values()
    rows[0]["property_intensity"]["num"] = 1.5
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_property_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_property_seeds(bad_json)
    build = compile_authored_property_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredRelationCourseError, match="发布失败"):
        compile_authored_property_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_PROPERTY",
            "ATTR_PROP_",
            "build_property_edges",
            "property_claims"}:
        assert token not in source
