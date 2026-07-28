"""D-02C.2 SUBSET/MEMBER 极小 pack 的 profile、读取和发布 facade。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_SET_EXPR,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    compile_relation_seed,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    LICENSE_ID,
    RELATION_MEMBER,
    RELATION_SUBSET,
    REQUEST_RELATION_EVALUATION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_MEMBER_ELEMENT,
    ROLE_MEMBER_SET,
    ROLE_SUBSET_CHILD,
    ROLE_SUBSET_PARENT,
    SCHEMA_MEMBER,
    SCHEMA_SUBSET,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--subset-member-v1"
STAGE = "W-06"
SUBSTAGE = "SUBSET_MEMBER"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "DIRECTION_REVERSAL",
    "TYPE_MISMATCH",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
_PROFILE = {
    "SUBSET": (
        RELATION_SUBSET,
        SCHEMA_SUBSET,
        frozenset({ROLE_SUBSET_CHILD, ROLE_SUBSET_PARENT}),
    ),
    "MEMBER": (
        RELATION_MEMBER,
        SCHEMA_MEMBER,
        frozenset({ROLE_MEMBER_ELEMENT, ROLE_MEMBER_SET}),
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
    "authored-subset-member-seed-v1",
    "urn:pure-integer-ai:ph2:authored-subset-member-v1",
    "Pure Integer AI PH2 authored typed subset/member seed",
    "SUBSET_MEMBER_RELATION_LABEL",
    "subset-member",
    100,
)


def _validate_profile(seed: AuthoredRelationSeed) -> None:
    """核对 SUBSET/MEMBER 坐标、Role、方向、consumer 和 endpoint 类型。"""
    profile = _PROFILE.get(seed.relation_family)
    if profile is None:
        raise AuthoredRelationCourseError("subset/member relation family 未注册")
    relation, schema, roles = profile
    if (seed.relation_kind != relation
            or seed.schema_kind != schema
            or seed.directionality != DIRECTION_FORWARD):
        raise AuthoredRelationCourseError("subset/member relation profile 坐标漂移")
    actual_roles = {item.role_kind for item in seed.bindings}
    if len(seed.bindings) != 2 or actual_roles != roles:
        raise AuthoredRelationCourseError("subset/member Role profile 漂移")
    if seed.consumer_request.request_kind != REQUEST_RELATION_EVALUATION:
        raise AuthoredRelationCourseError("subset/member consumer request kind 漂移")
    endpoint_index = {item.endpoint_id: item for item in seed.endpoints}
    endpoint_by_role = {
        item.role_kind: endpoint_index[item.endpoint_id]
        for item in seed.bindings
    }
    if seed.perturbation_kind == "TYPE_MISMATCH":
        return
    if seed.relation_family == "SUBSET":
        if any(endpoint_by_role[role].object_kind != OBJECT_SET_EXPR
               for role in roles):
            raise AuthoredRelationCourseError("SUBSET endpoint 必须都是 SetExpr")
        return
    if endpoint_by_role[ROLE_MEMBER_ELEMENT].object_kind not in {
            OBJECT_ENTITY, OBJECT_SET_EXPR}:
        raise AuthoredRelationCourseError("MEMBER element endpoint 类型非法")
    if endpoint_by_role[ROLE_MEMBER_SET].object_kind != OBJECT_SET_EXPR:
        raise AuthoredRelationCourseError("MEMBER set endpoint 必须是 SetExpr")


def read_authored_subset_member_seeds(
        path: str | Path) -> tuple[AuthoredRelationSeed, ...]:
    """读取规范 SUBSET/MEMBER JSONL，并核对覆盖面、profile 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredRelationCourseError("subset/member sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredRelationCourseError("subset/member sample 必须非空并以换行结束")
    seeds: list[AuthoredRelationSeed] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredRelationCourseError(
                f"subset/member 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"subset/member 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seed = AuthoredRelationSeed.from_dict(value)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"subset/member 第 {line_number} 行 payload 非法") from error
        _validate_profile(seed)
        seeds.append(seed)
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise AuthoredRelationCourseError("subset/member seed_id 重复")
    orders = [seed.logical_order for seed in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredRelationCourseError("subset/member logical_order 必须严格递增")
    index = {seed.seed_id: seed for seed in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredRelationCourseError("subset/member supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.relation_family != seed.relation_family):
            raise AuthoredRelationCourseError(
                "subset/member supersede 不得跨 family/split/relation")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredRelationCourseError("subset/member supersede 必须是 parser revision")
    teacher_families = {seed.family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_families = {seed.family for seed in seeds if seed.label_owner == "evaluator"}
    teacher_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "teacher"
    }
    evaluator_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "evaluator"
    }
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates):
        raise AuthoredRelationCourseError("teacher/evaluator family 必须非空且互斥")
    if {seed.sample_role for seed in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredRelationCourseError("subset/member 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            seed.perturbation_kind for seed in seeds}):
        raise AuthoredRelationCourseError("subset/member 缺少必需反向破坏")
    supported_relations = {
        seed.relation_family for seed in seeds
        if seed.sample_role in {"support", "supersede"}
    }
    if supported_relations != set(_PROFILE):
        raise AuthoredRelationCourseError("SUBSET/MEMBER 均须有独立正例")
    return tuple(seeds)


def compile_authored_subset_member_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02C.2 typed SUBSET/MEMBER 极小 pack。"""
    seeds = read_authored_subset_member_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_relation_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError(
            "subset/member pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_subset_member_course",
    "read_authored_subset_member_seeds",
]
