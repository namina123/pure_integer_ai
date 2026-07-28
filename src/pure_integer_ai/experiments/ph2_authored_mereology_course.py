"""D-02C.4 MEREOLOGY typed relation family 极小 pack。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import OBJECT_ENTITY
from pure_integer_ai.cognition.shared.typed_relation import InverseRule
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
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_HAS_PART,
    RELATION_PART_OF,
    REQUEST_MEREOLOGY_QUERY,
    REQUIRED_SAMPLE_ROLES,
    ROLE_HAS_PART_PART,
    ROLE_HAS_PART_WHOLE,
    ROLE_PART_OF_PART,
    ROLE_PART_OF_WHOLE,
    SCHEMA_HAS_PART,
    SCHEMA_PART_OF,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--mereology-v1"
STAGE = "W-06"
SUBSTAGE = "MEREOLOGY"
INVERSE_RULE_KIND = 1
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "DIRECTION_REVERSAL",
    "INVERSE_RELATION",
    "RELATION_CONFUSION",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
_PROFILE = {
    "PART_OF": (
        RELATION_PART_OF,
        SCHEMA_PART_OF,
        ROLE_PART_OF_PART,
        ROLE_PART_OF_WHOLE,
    ),
    "HAS_PART": (
        RELATION_HAS_PART,
        SCHEMA_HAS_PART,
        ROLE_HAS_PART_PART,
        ROLE_HAS_PART_WHOLE,
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
    "authored-mereology-seed-v1",
    "urn:pure-integer-ai:ph2:authored-mereology-v1",
    "Pure Integer AI PH2 authored typed mereology seed",
    "MEREOLOGY_RELATION_LABEL",
    "mereology",
    100,
)


def _identity_key(identity) -> list[int]:
    """把一等对象身份投影为规范严格整数列表。"""
    return list(identity.stable_key())


def _inverse_rule() -> InverseRule:
    """构造本 pack 显式 PART_OF -> HAS_PART inverse rule。"""
    return InverseRule(
        authored_relation_rule_identity(INVERSE_RULE_KIND),
        authored_relation_identity(RELATION_PART_OF),
        authored_relation_role_identity(ROLE_PART_OF_PART),
        authored_relation_role_identity(ROLE_PART_OF_WHOLE),
        authored_relation_identity(RELATION_HAS_PART),
        authored_relation_role_identity(ROLE_HAS_PART_WHOLE),
        authored_relation_role_identity(ROLE_HAS_PART_PART),
    )


def _protocol_payload(seed: AuthoredRelationSeed) -> dict:
    """输出 canonical Role、显式 inverse 及空的默认传递/复合规则集。"""
    profile = _PROFILE[seed.relation_family]
    part_role = authored_relation_role_identity(profile[2])
    whole_role = authored_relation_role_identity(profile[3])
    inverse = _inverse_rule()
    return {
        "canonical_part_role_key": _identity_key(part_role),
        "canonical_whole_role_key": _identity_key(whole_role),
        "composition_rules": [],
        "inverse_rules": [{
            "premise_left_role_key": _identity_key(
                inverse.premise_left_role),
            "premise_relation_key": _identity_key(
                inverse.premise_relation),
            "premise_right_role_key": _identity_key(
                inverse.premise_right_role),
            "result_left_role_key": _identity_key(
                inverse.result_left_role),
            "result_relation_key": _identity_key(
                inverse.result_relation),
            "result_right_role_key": _identity_key(
                inverse.result_right_role),
            "rule_key": _identity_key(inverse.rule),
        }],
        "irreflexive_rules": [],
        "transitive_rules": [],
    }


def _compile_mereology_seed(seed: AuthoredRelationSeed) -> AuthoredCompiledSeed:
    """把显式 mereology protocol 附加到共享 typed relation payload。"""
    base = compile_relation_seed(seed)
    value = base.observation_payload.to_value()
    protocol = _protocol_payload(seed)
    value["mereology_protocol"] = protocol
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
            "mereology_protocol_v1",
            INVERSE_RULE_KIND,
        ),
    )


def _validate_profile(seed: AuthoredRelationSeed) -> None:
    """核对 relation variant、canonical Role、端点类型和 query 方向。"""
    profile = _PROFILE.get(seed.relation_family)
    if profile is None:
        raise AuthoredRelationCourseError("mereology relation family 未注册")
    relation, schema, part_role, whole_role = profile
    if (seed.relation_kind != relation
            or seed.schema_kind != schema
            or seed.directionality != DIRECTION_FORWARD):
        raise AuthoredRelationCourseError("mereology relation profile 坐标漂移")
    if len(seed.endpoints) != 2 or len(seed.bindings) != 2:
        raise AuthoredRelationCourseError("mereology 必须恰有 part/whole 两端")
    binding_by_role = {item.role_kind: item for item in seed.bindings}
    if set(binding_by_role) != {part_role, whole_role}:
        raise AuthoredRelationCourseError("mereology part/whole Role profile 漂移")
    if any(frozenset(item.allowed_object_kinds) != {OBJECT_ENTITY}
           for item in seed.bindings):
        raise AuthoredRelationCourseError("mereology Role slot 类型漂移")
    endpoint_index = {item.endpoint_id: item for item in seed.endpoints}
    if any(item.object_kind != OBJECT_ENTITY for item in seed.endpoints):
        raise AuthoredRelationCourseError("mereology endpoint 必须是 Entity")
    request = seed.consumer_request
    if request.request_kind != REQUEST_MEREOLOGY_QUERY:
        raise AuthoredRelationCourseError("mereology consumer request kind 漂移")
    declared_part = binding_by_role[part_role].endpoint_id
    declared_whole = binding_by_role[whole_role].endpoint_id
    if seed.perturbation_kind == "DIRECTION_REVERSAL":
        if (request.origin_endpoint_id != declared_whole
                or request.whole_endpoint_id != declared_part):
            raise AuthoredRelationCourseError(
                "mereology direction reversal 未只交换 part/whole")
    elif (request.origin_endpoint_id != declared_part
          or request.whole_endpoint_id != declared_whole):
        raise AuthoredRelationCourseError("mereology query part/whole 锚漂移")
    if endpoint_index[request.origin_endpoint_id] == endpoint_index[
            request.whole_endpoint_id]:
        raise AuthoredRelationCourseError("mereology query 端点不得相同")


def read_authored_mereology_seeds(
        path: str | Path) -> tuple[AuthoredRelationSeed, ...]:
    """读取规范 MEREOLOGY JSONL，并核对覆盖面、owner 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredRelationCourseError("mereology sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredRelationCourseError("mereology sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredRelationCourseError(
                f"mereology 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"mereology 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seed = AuthoredRelationSeed.from_dict(value)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"mereology 第 {line_number} 行 payload 非法") from error
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredRelationCourseError("mereology seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredRelationCourseError("mereology logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredRelationCourseError("mereology supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.relation_family != seed.relation_family):
            raise AuthoredRelationCourseError(
                "mereology supersede 不得跨 family/split/relation")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredRelationCourseError(
                "mereology supersede 必须是 parser revision")
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
            "mereology teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredRelationCourseError("mereology 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredRelationCourseError("mereology 缺少必需反向破坏")
    supported = {
        item.relation_family for item in seeds
        if item.sample_role in {"support", "supersede"}
    }
    if supported != set(_PROFILE):
        raise AuthoredRelationCourseError(
            "PART_OF/HAS_PART 均须有独立正例")
    return tuple(seeds)


def compile_authored_mereology_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02C.4 typed MEREOLOGY 极小 pack。"""
    seeds = read_authored_mereology_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(_compile_mereology_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError("mereology pack 发布失败") from error


__all__ = [
    "INVERSE_RULE_KIND",
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_mereology_course",
    "read_authored_mereology_seeds",
]
