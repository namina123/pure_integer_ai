"""D-02C.5 SIMILAR/ANTONYM 双 channel typed 极小 pack。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import OBJECT_CONCEPT
from pure_integer_ai.cognition.shared.typed_relation import (
    IrreflexiveRule,
    SymmetricRule,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
    authored_relation_rule_identity,
    compile_relation_seed,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_SYMMETRIC,
    LICENSE_ID,
    RELATION_ANTONYM,
    RELATION_SIMILAR,
    REQUEST_SYMMETRIC_PAIR_QUERY,
    REQUIRED_SAMPLE_ROLES,
    ROLE_ANTONYM_LEFT,
    ROLE_ANTONYM_RIGHT,
    ROLE_SIMILAR_LEFT,
    ROLE_SIMILAR_RIGHT,
    SCHEMA_ANTONYM,
    SCHEMA_SIMILAR,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--similar-antonym-v1"
STAGE = "W-06"
SUBSTAGE = "SIMILAR_ANTONYM"
SIMILAR_SYMMETRIC_RULE_KIND = 2
ANTONYM_SYMMETRIC_RULE_KIND = 3
ANTONYM_IRREFLEXIVE_RULE_KIND = 4
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "PAIR_REVERSAL",
    "RELATION_CONFUSION",
    "ALIAS_CONFUSION",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
_PROFILE = {
    "SIMILAR": (
        RELATION_SIMILAR,
        SCHEMA_SIMILAR,
        ROLE_SIMILAR_LEFT,
        ROLE_SIMILAR_RIGHT,
        SIMILAR_SYMMETRIC_RULE_KIND,
        None,
        (20501, 1),
        (20502, 1),
    ),
    "ANTONYM": (
        RELATION_ANTONYM,
        SCHEMA_ANTONYM,
        ROLE_ANTONYM_LEFT,
        ROLE_ANTONYM_RIGHT,
        ANTONYM_SYMMETRIC_RULE_KIND,
        ANTONYM_IRREFLEXIVE_RULE_KIND,
        (20501, 2),
        (20502, 2),
    ),
}
_SPEC = AuthoredCourseSpec(
    SOURCE_KEY,
    LICENSE_ID,
    1,
    1,
    1,
    1,
    1,
    PACK_NAME,
    STAGE,
    SUBSTAGE,
    "authored-similar-antonym-seed-v1",
    "urn:pure-integer-ai:ph2:authored-similar-antonym-v1",
    "Pure Integer AI PH2 authored typed similar antonym seed",
    "SEMANTIC_PAIR_RELATION_LABEL",
    "similar-antonym",
    100,
)


def _identity_key(identity) -> list[int]:
    """把一等对象身份投影为规范严格整数列表。"""
    return list(identity.stable_key())


def _protocol_payload(seed: AuthoredRelationSeed) -> dict:
    """输出双 channel owner、显式对称规则及可选反自反规则。"""
    profile = _PROFILE[seed.relation_family]
    relation = authored_relation_identity(profile[0])
    left_role = authored_relation_role_identity(profile[2])
    right_role = authored_relation_role_identity(profile[3])
    symmetric = SymmetricRule(
        authored_relation_rule_identity(profile[4]),
        relation,
        left_role,
        right_role,
    )
    irreflexive = []
    if profile[5] is not None:
        rule = IrreflexiveRule(
            authored_relation_rule_identity(profile[5]),
            relation,
            left_role,
            right_role,
        )
        irreflexive = [{
            "left_role_key": _identity_key(rule.left_role),
            "relation_key": _identity_key(rule.relation),
            "right_role_key": _identity_key(rule.right_role),
            "rule_key": _identity_key(rule.rule),
        }]
    return {
        "alias_promotion": 0,
        "channel_protocol_key": list(profile[7]),
        "discovery_writes_use": 0,
        "exact_use_requires_context": 1,
        "hypothesis_kind_key": list(profile[6]),
        "inverse_rules": [],
        "irreflexive_rules": irreflexive,
        "left_role_key": _identity_key(left_role),
        "right_role_key": _identity_key(right_role),
        "symmetric_rule": {
            "left_role_key": _identity_key(symmetric.left_role),
            "relation_key": _identity_key(symmetric.relation),
            "right_role_key": _identity_key(symmetric.right_role),
            "rule_key": _identity_key(symmetric.rule),
        },
        "transitive_rules": [],
    }


def _compile_semantic_pair_seed(
        seed: AuthoredRelationSeed) -> AuthoredCompiledSeed:
    """把双 owner 和显式 symmetric protocol 附加到共享 payload。"""
    base = compile_relation_seed(seed)
    value = base.observation_payload.to_value()
    protocol = _protocol_payload(seed)
    value["semantic_pair_protocol"] = protocol
    payload = CanonicalJsonObject.from_value(value)
    return replace(
        base,
        observation_payload=payload,
        dedup_parts=(seed.surface, value),
        content_parts=(
            seed.surface,
            value["candidate_definition"],
            protocol,
        ),
        shape_parts=(
            *base.shape_parts,
            "semantic_pair_protocol_v1",
            protocol["hypothesis_kind_key"],
        ),
    )


def _validate_profile(seed: AuthoredRelationSeed) -> None:
    """核对 channel、同型 Role、对称 query 和端点类型。"""
    profile = _PROFILE.get(seed.relation_family)
    if profile is None:
        raise AuthoredRelationCourseError("semantic pair relation family 未注册")
    relation, schema, left_role, right_role = profile[:4]
    if (seed.relation_kind != relation
            or seed.schema_kind != schema
            or seed.directionality != DIRECTION_SYMMETRIC):
        raise AuthoredRelationCourseError(
            "semantic pair relation profile 坐标漂移")
    if len(seed.endpoints) != 2 or len(seed.bindings) != 2:
        raise AuthoredRelationCourseError(
            "semantic pair 必须恰有两个 endpoint")
    binding_by_role = {item.role_kind: item for item in seed.bindings}
    if set(binding_by_role) != {left_role, right_role}:
        raise AuthoredRelationCourseError(
            "semantic pair left/right Role profile 漂移")
    if any(frozenset(item.allowed_object_kinds) != {OBJECT_CONCEPT}
           for item in seed.bindings):
        raise AuthoredRelationCourseError("semantic pair Role slot 类型漂移")
    if any(item.object_kind != OBJECT_CONCEPT for item in seed.endpoints):
        raise AuthoredRelationCourseError(
            "semantic pair endpoint 必须是 Concept")
    request = seed.consumer_request
    if request.request_kind != REQUEST_SYMMETRIC_PAIR_QUERY:
        raise AuthoredRelationCourseError(
            "semantic pair consumer request kind 漂移")
    bound = {
        binding_by_role[left_role].endpoint_id,
        binding_by_role[right_role].endpoint_id,
    }
    requested = {
        request.origin_endpoint_id,
        request.counterpart_endpoint_id,
    }
    if requested != bound:
        raise AuthoredRelationCourseError(
            "semantic pair query endpoint/counterpart 锚漂移")


def read_authored_semantic_pair_seeds(
        path: str | Path) -> tuple[AuthoredRelationSeed, ...]:
    """读取规范 SIMILAR/ANTONYM JSONL，并核对双 owner 与恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredRelationCourseError(
            "semantic pair sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredRelationCourseError(
            "semantic pair sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredRelationCourseError(
                f"semantic pair 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"semantic pair 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seed = AuthoredRelationSeed.from_dict(value)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"semantic pair 第 {line_number} 行 payload 非法") from error
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredRelationCourseError("semantic pair seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredRelationCourseError(
            "semantic pair logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredRelationCourseError(
                "semantic pair supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.relation_family != seed.relation_family):
            raise AuthoredRelationCourseError(
                "semantic pair supersede 不得跨 family/split/relation")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredRelationCourseError(
                "semantic pair supersede 必须是 parser revision")
    teacher_families = {
        item.family for item in seeds if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in seeds if item.label_owner == "evaluator"}
    teacher_templates = {
        item.template_family for item in seeds if item.label_owner == "teacher"}
    evaluator_templates = {
        item.template_family for item in seeds
        if item.label_owner == "evaluator"}
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates):
        raise AuthoredRelationCourseError(
            "semantic pair teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredRelationCourseError(
            "semantic pair 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredRelationCourseError(
            "semantic pair 缺少必需反向破坏")
    supported = {
        item.relation_family for item in seeds
        if item.sample_role in {"support", "supersede"}
    }
    if supported != set(_PROFILE):
        raise AuthoredRelationCourseError(
            "SIMILAR/ANTONYM 均须有独立正例")
    return tuple(seeds)


def compile_authored_semantic_pair_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02C.5 typed SIMILAR/ANTONYM 极小 pack。"""
    seeds = read_authored_semantic_pair_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(_compile_semantic_pair_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError(
            "semantic pair pack 发布失败") from error


__all__ = [
    "ANTONYM_IRREFLEXIVE_RULE_KIND",
    "ANTONYM_SYMMETRIC_RULE_KIND",
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "SIMILAR_SYMMETRIC_RULE_KIND",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_semantic_pair_course",
    "read_authored_semantic_pair_seeds",
]
