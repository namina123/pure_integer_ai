"""D-02C.2 AUTHORED_CC0_V1 typed SUBSET/MEMBER 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    OBJECT_SET_EXPR,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_MEMBER,
    RELATION_SUBSET,
    REQUEST_RELATION_EVALUATION,
    ROLE_MEMBER_ELEMENT,
    ROLE_MEMBER_SET,
    ROLE_SUBSET_CHILD,
    ROLE_SUBSET_PARENT,
    SCHEMA_MEMBER,
    SCHEMA_SUBSET,
    SOURCE_KEY,
    AuthoredRelationCourseError,
)
from pure_integer_ai.experiments.ph2_authored_subset_member_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_subset_member_course,
    read_authored_subset_member_seeds,
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
    "data/ph2/authored_relation_subset_member_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_subset_member_course.py"),
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


def test_sample_covers_both_relations_types_consumer_and_recovery():
    """SUBSET/MEMBER 各有正例，反例覆盖方向、类型、内容、伪关系和冲突。"""
    seeds = read_authored_subset_member_seeds(SAMPLE_PATH)
    assert len(seeds) == 10
    assert LICENSE_ID == "CC0-1.0"
    assert {seed.relation_family for seed in seeds} == {"SUBSET", "MEMBER"}
    assert {seed.sample_role for seed in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert REQUIRED_PERTURBATIONS.issubset({
        seed.perturbation_kind for seed in seeds})
    assert any(
        seed.perturbation_kind == "CONTENT_REPLACEMENT"
        and seed.split == "held_out"
        for seed in seeds
    )
    for seed in seeds:
        assert seed.directionality == DIRECTION_FORWARD
        assert seed.consumer_request.request_kind == REQUEST_RELATION_EVALUATION
        roles = {item.role_kind for item in seed.bindings}
        if seed.relation_family == "SUBSET":
            assert seed.relation_kind == RELATION_SUBSET
            assert seed.schema_kind == SCHEMA_SUBSET
            assert roles == {ROLE_SUBSET_CHILD, ROLE_SUBSET_PARENT}
            if seed.perturbation_kind != "TYPE_MISMATCH":
                assert {item.object_kind for item in seed.endpoints} == {
                    OBJECT_SET_EXPR}
        else:
            assert seed.relation_kind == RELATION_MEMBER
            assert seed.schema_kind == SCHEMA_MEMBER
            assert roles == {ROLE_MEMBER_ELEMENT, ROLE_MEMBER_SET}


def test_direction_type_and_supersede_break_only_declared_axes():
    """方向翻转只换 Role filler，类型错配显式保留，revision 绑定更早 SUBSET。"""
    seeds = {
        seed.seed_id: seed for seed in read_authored_subset_member_seeds(
            SAMPLE_PATH)
    }
    supported = seeds["teacher-sparrow-subset-v1"]
    reversed_seed = seeds["teacher-sparrow-subset-reversed-v1"]
    assert supported.surface == reversed_seed.surface
    assert supported.endpoints == reversed_seed.endpoints
    assert supported.consumer_request == reversed_seed.consumer_request
    assert {
        item.role_kind: item.endpoint_id for item in supported.bindings
    } != {
        item.role_kind: item.endpoint_id for item in reversed_seed.bindings
    }
    mismatch = seeds["teacher-member-type-mismatch-v1"]
    endpoint_by_id = {item.endpoint_id: item for item in mismatch.endpoints}
    endpoint_by_role = {
        item.role_kind: endpoint_by_id[item.endpoint_id]
        for item in mismatch.bindings
    }
    assert endpoint_by_role[ROLE_MEMBER_ELEMENT].object_kind == OBJECT_SET_EXPR
    assert endpoint_by_role[ROLE_MEMBER_SET].object_kind == OBJECT_ENTITY
    revised = seeds["teacher-sparrow-subset-v2"]
    assert revised.supersedes_seed_id == supported.seed_id
    assert revised.relation_family == supported.relation_family == "SUBSET"


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，10/10/6/4 分账并保持 expected 私有。"""
    first = compile_authored_subset_member_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_subset_member_course(
        SAMPLE_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == SOURCE_KEY
    assert first.manifest.license_partition == LICENSE_ID
    assert first.manifest.w_stages == ("W-06",)
    assert first.validation.source_ref_count == 10
    assert first.validation.observation_count == 10
    assert first.validation.teacher_evidence_count == 6
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
    assert len(_records(first, RECORD_SOURCE_REF)) == 10
    assert len(observations) == 10
    assert len(_records(first, RECORD_TEACHER_EVIDENCE)) == 6
    assert len(_records(first, RECORD_EVALUATOR_LABEL)) == 4
    for observation in observations:
        payload = observation.typed_payload.to_value()
        assert "expected_state" not in payload
        assert "expected_payload" not in payload


def test_payload_roundtrips_setexpr_relation_schema_and_atomic_definition(
        tmp_path):
    """每条候选可恢复 SetExpr/Entity endpoint、RelationSchema 和原子命题。"""
    build = compile_authored_subset_member_course(SAMPLE_PATH, tmp_path)
    for observation in _records(build, RECORD_OBSERVATION):
        payload = observation.typed_payload.to_value()
        endpoint_kinds = {
            _identity(item["endpoint_key"]).object_kind
            for item in payload["endpoints"]
        }
        assert endpoint_kinds.issubset({OBJECT_ENTITY, OBJECT_SET_EXPR})
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
        assert schema.validate_definition(definition) == definition
        assert payload["consumer_request"]["request_kind"] == (
            REQUEST_RELATION_EVALUATION)


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 owner、W-06 视图和同类 supersede 均可直接复核。"""
    build = compile_authored_subset_member_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[1].__setitem__("seed_id", rows[0]["seed_id"]), "重复"),
        (lambda rows: rows[6].__setitem__("split", "train"), "split"),
        (lambda rows: rows[0].__setitem__("relation_family", "IS_A"), "family"),
        (lambda rows: rows[0].__setitem__("relation_kind", RELATION_MEMBER),
         "profile"),
        (lambda rows: rows[0].__setitem__("schema_kind", SCHEMA_MEMBER),
         "profile"),
        (lambda rows: rows[0].__setitem__("directionality", 1), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 99),
         "Role profile"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__("object_kind", 4),
         "SetExpr"),
        (lambda rows: rows[1]["endpoints"][1].__setitem__("object_kind", 16),
         "set endpoint"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 1), "request kind"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "endpoint_id", "missing"), "未知 endpoint"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: (
            rows[0].__setitem__("sample_role", "supersede"),
            rows[0].__setitem__("supersedes_seed_id", rows[1]["seed_id"]),
        ), "更早"),
        (lambda rows: rows[5].__setitem__(
            "supersedes_seed_id", rows[1]["seed_id"]), "family/split/relation"),
    ],
)
def test_bad_license_profile_types_consumer_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、profile、类型、consumer 和恢复链均不能进入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredRelationCourseError, match=message):
        read_authored_subset_member_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_is_a_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬用旧 IS_A 写边入口。"""
    rows = _sample_values()
    rows[0]["relation_kind"] = 3.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_subset_member_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_subset_member_seeds(bad_json)
    build = compile_authored_subset_member_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredRelationCourseError, match="发布失败"):
        compile_authored_subset_member_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {"EDGE_IS_A", "build_is_a_edge", "legacy_mapper"}:
        assert token not in source
