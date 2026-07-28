"""D-02C.5 AUTHORED_CC0_V1 typed SIMILAR/ANTONYM 资料包 T0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
)
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPair,
    SymmetricPairPattern,
    SymmetricRelationBudget,
    SymmetricRelationProtocol,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    IrreflexiveRule,
    RelationSchema,
    RelationSlotSchema,
    SameKindConstraint,
    SymmetricRule,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_SYMMETRIC,
    LICENSE_ID,
    RELATION_ANTONYM,
    RELATION_SIMILAR,
    REQUEST_SYMMETRIC_PAIR_QUERY,
    ROLE_ANTONYM_LEFT,
    ROLE_ANTONYM_RIGHT,
    ROLE_SIMILAR_LEFT,
    ROLE_SIMILAR_RIGHT,
    SCHEMA_ANTONYM,
    SCHEMA_SIMILAR,
    SOURCE_KEY,
    AuthoredRelationCourseError,
)
from pure_integer_ai.experiments.ph2_authored_semantic_pair_course import (
    PACK_NAME,
    REQUIRED_PERTURBATIONS,
    compile_authored_semantic_pair_course,
    read_authored_semantic_pair_seeds,
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
from pure_integer_ai.experiments.semantic_pair_runtime import (
    SemanticPairBudget,
)


SAMPLE_PATH = Path(
    "data/ph2/authored_relation_similar_antonym_seed_v1.jsonl.sample")
SOURCE_PATHS = (
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_schema.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_relation_compile.py"),
    Path("src/pure_integer_ai/experiments/ph2_authored_semantic_pair_course.py"),
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


def _endpoint_local_ids(seed) -> frozenset[int]:
    """返回 pair 两端的局部语义 id。"""
    return frozenset(item.local_id for item in seed.endpoints)


def test_sample_covers_two_channels_symmetry_alias_boundary_and_recovery():
    """SIMILAR/ANTONYM 各有正例，pair 反向和 alias 混淆独立覆盖。"""
    seeds = read_authored_semantic_pair_seeds(SAMPLE_PATH)
    assert len(seeds) == 11
    assert LICENSE_ID == "CC0-1.0"
    assert {item.relation_family for item in seeds} == {
        "SIMILAR", "ANTONYM"}
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
        assert seed.directionality == DIRECTION_SYMMETRIC
        assert seed.consumer_request.request_kind == (
            REQUEST_SYMMETRIC_PAIR_QUERY)
        if seed.relation_family == "SIMILAR":
            assert seed.relation_kind == RELATION_SIMILAR
            assert seed.schema_kind == SCHEMA_SIMILAR
            assert {item.role_kind for item in seed.bindings} == {
                ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT}
        else:
            assert seed.relation_kind == RELATION_ANTONYM
            assert seed.schema_kind == SCHEMA_ANTONYM
            assert {item.role_kind for item in seed.bindings} == {
                ROLE_ANTONYM_LEFT, ROLE_ANTONYM_RIGHT}
        assert {item.object_kind for item in seed.endpoints} == {OBJECT_CONCEPT}
        assert all(item.allowed_object_kinds == (OBJECT_CONCEPT,)
                   for item in seed.bindings)


def test_pair_reversal_canonicalizes_but_cross_channel_stays_distinct():
    """反向 surface 保留同 pair，换成 ANTONYM 后 relation 身份仍独立。"""
    seeds = {
        item.seed_id: item
        for item in read_authored_semantic_pair_seeds(SAMPLE_PATH)
    }
    direct = seeds["teacher-rapid-fast-similar-v1"]
    reversed_seed = seeds["teacher-fast-rapid-reversed-v1"]
    cross = seeds["teacher-similar-as-antonym-v1"]
    assert _endpoint_local_ids(direct) == _endpoint_local_ids(reversed_seed)
    assert direct.relation_kind == reversed_seed.relation_kind
    assert direct.schema_kind == reversed_seed.schema_kind
    assert [item.surface_fragment for item in direct.endpoints] == [
        "迅速", "快速"]
    assert [item.surface_fragment for item in reversed_seed.endpoints] == [
        "快速", "迅速"]
    assert _endpoint_local_ids(direct) == _endpoint_local_ids(cross)
    assert direct.relation_kind != cross.relation_kind
    assert direct.schema_kind != cross.schema_kind
    alias = seeds["teacher-alias-as-similar-v1"]
    assert alias.relation_family == "SIMILAR"
    assert alias.perturbation_kind == "ALIAS_CONFUSION"


def test_compiler_is_bit_identical_and_owner_separated(tmp_path):
    """两目录 bit-identical，11/11/7/4 分账并保持 expected 私有。"""
    first = compile_authored_semantic_pair_course(
        SAMPLE_PATH, tmp_path / "first")
    second = compile_authored_semantic_pair_course(
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


def test_payload_roundtrips_independent_channels_rules_pair_and_query(tmp_path):
    """恢复两个协议、独立 kind/owner、显式对称规则和 ANTONYM 反自反。"""
    build = compile_authored_semantic_pair_course(SAMPLE_PATH, tmp_path)
    protocols = {}
    channel_keys = {}
    hypothesis_keys = {}
    reversed_pairs = []
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
        constraints = tuple(SameKindConstraint(
            _identity(value["constraint_key"]),
            tuple(_identity(role) for role in value["role_keys"]),
        ) for value in schema_value["constraints"])
        schema = RelationSchema(
            _identity(schema_value["schema_key"]),
            _identity(schema_value["relation_key"]),
            tuple(RelationSlotSchema(
                _identity(value["role_key"]),
                frozenset(value["allowed_object_kinds"]),
                value["min_count"],
                value["max_count"],
            ) for value in schema_value["slots"]),
            constraints,
        )
        schema.validate_definition(definition)
        protocol_value = payload["semantic_pair_protocol"]
        symmetric_value = protocol_value["symmetric_rule"]
        symmetric = SymmetricRule(
            _identity(symmetric_value["rule_key"]),
            _identity(symmetric_value["relation_key"]),
            _identity(symmetric_value["left_role_key"]),
            _identity(symmetric_value["right_role_key"]),
        )
        irreflexive = None
        if protocol_value["irreflexive_rules"]:
            assert len(protocol_value["irreflexive_rules"]) == 1
            rule_value = protocol_value["irreflexive_rules"][0]
            irreflexive = IrreflexiveRule(
                _identity(rule_value["rule_key"]),
                _identity(rule_value["relation_key"]),
                _identity(rule_value["left_role_key"]),
                _identity(rule_value["right_role_key"]),
            )
        protocol = SymmetricRelationProtocol(
            schema,
            _identity(protocol_value["left_role_key"]),
            _identity(protocol_value["right_role_key"]),
            symmetric,
            irreflexive,
        )
        protocols[protocol.relation] = protocol
        channel_keys[protocol.relation] = tuple(
            protocol_value["channel_protocol_key"])
        hypothesis_keys[protocol.relation] = tuple(
            protocol_value["hypothesis_kind_key"])
        assert protocol_value["alias_promotion"] == 0
        assert protocol_value["inverse_rules"] == []
        assert protocol_value["transitive_rules"] == []
        assert protocol_value["discovery_writes_use"] == 0
        assert protocol_value["exact_use_requires_context"] == 1
        filler_by_role = {
            item.role: item.filler for item in definition.canonical_bindings()
        }
        pair = protocol.pair(
            filler_by_role[protocol.left_role],
            filler_by_role[protocol.right_role],
        )
        request = payload["consumer_request"]
        assert request["request_kind"] == REQUEST_SYMMETRIC_PAIR_QUERY
        pattern = SymmetricPairPattern(
            _identity(request["pattern"]["endpoint_key"]),
            _identity(request["pattern"]["counterpart_key"]),
        )
        assert pattern.exact_pair(protocol) == pair
        budget = SymmetricRelationBudget(
            request["budget"]["max_direct_facts"],
            request["budget"]["max_options"],
        )
        total = SemanticPairBudget(
            request["budget"]["max_total_direct_facts"])
        assert budget.stable_key() == (20, 20)
        assert total.stable_key() == (40,)
        if observation.perturbation_kind in {"NONE", "PAIR_REVERSAL"}:
            if pair.relation == _identity(schema_value["relation_key"]):
                reversed_pairs.append((observation.perturbation_kind, pair))
    assert len(protocols) == 2
    assert len(set(channel_keys.values())) == 2
    assert len(set(hypothesis_keys.values())) == 2
    similar = next(
        item for item in protocols.values()
        if item.relation.components[-1] == RELATION_SIMILAR)
    antonym = next(
        item for item in protocols.values()
        if item.relation.components[-1] == RELATION_ANTONYM)
    assert similar.irreflexive_rule is None
    assert antonym.irreflexive_rule is not None
    rapid_pairs = [pair for kind, pair in reversed_pairs
                   if kind in {"NONE", "PAIR_REVERSAL"}
                   and {item.components[-1] for item in pair.endpoints()} == {
                       91, 92}]
    assert len(rapid_pairs) == 2
    assert rapid_pairs[0] == rapid_pairs[1]


def test_split_owner_stage_and_supersede_are_auditable(tmp_path):
    """split 来源簇、双 label owner、W-06 视图和同 channel 修订可复核。"""
    build = compile_authored_semantic_pair_course(SAMPLE_PATH, tmp_path)
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
        (lambda rows: rows[0].__setitem__("relation_family", "PURE_ALIAS"),
         "family"),
        (lambda rows: rows[0].__setitem__("relation_kind", 9), "profile"),
        (lambda rows: rows[0]["bindings"][0].__setitem__(
            "allowed_object_kinds", [16]), "allowed_object_kinds"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "object_kind", 16), "allowed_object_kinds"),
        (lambda rows: rows[0]["bindings"][0].__setitem__("role_kind", 99),
         "Role profile"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "request_kind", 2), "consumer"),
        (lambda rows: rows[0]["consumer_request"].__setitem__(
            "counterpart_endpoint_id", "rapid"), "不得相同"),
        (lambda rows: rows[0]["endpoints"][0].__setitem__(
            "surface_fragment", "错误"), "span 与 surface"),
        (lambda rows: rows[6].__setitem__(
            "supersedes_seed_id", rows[8]["seed_id"]), "更早"),
        (lambda rows: (
            rows[9].__setitem__("sample_role", "supersede"),
            rows[9].__setitem__("perturbation_kind", "PARSER_REVISION"),
            rows[9].__setitem__("supersedes_seed_id", rows[1]["seed_id"]),
        ), "family/split/relation"),
        (lambda rows: rows[2].__setitem__("perturbation_kind", "NONE"),
         "缺少"),
    ],
)
def test_bad_license_channel_profile_query_types_and_supersede_fail_closed(
        tmp_path, mutate, message):
    """坏许可、channel、Role、query、类型和恢复链均不能入 pack。"""
    rows = _sample_values()
    mutate(rows)
    bad = tmp_path / "bad.sample"
    _write_values(bad, rows)
    with pytest.raises(AuthoredRelationCourseError, match=message):
        read_authored_semantic_pair_seeds(bad)


def test_float_noncanonical_existing_pack_and_legacy_is_fail_closed(tmp_path):
    """float/非规范/覆盖失败，源码不搬旧 pair、alias 或 inverse 入口。"""
    rows = _sample_values()
    rows[0]["relation_kind"] = 8.0
    bad_float = tmp_path / "float.sample"
    _write_json_with_float(bad_float, rows)
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_semantic_pair_seeds(bad_float)
    bad_json = tmp_path / "noncanonical.sample"
    bad_json.write_bytes(b'{"surface": "x"}\n')
    with pytest.raises(AuthoredRelationCourseError, match="规范 JSON"):
        read_authored_semantic_pair_seeds(bad_json)
    build = compile_authored_semantic_pair_course(
        SAMPLE_PATH, tmp_path / "release")
    assert build.pack_root.exists()
    with pytest.raises(AuthoredRelationCourseError, match="发布失败"):
        compile_authored_semantic_pair_course(
            SAMPLE_PATH, tmp_path / "release")
    source = "".join(path.read_text(encoding="utf-8") for path in SOURCE_PATHS)
    for token in {
            "EDGE_SIMILAR",
            "EDGE_ANTONYM",
            "verify_inverse",
            "alias_facts",
            "build_refers_stable_edge"}:
        assert token not in source
