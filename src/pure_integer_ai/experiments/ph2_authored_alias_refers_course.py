"""D-02C.1 PURE_ALIAS/REFERS 极小 pack 的 profile、读取和发布 facade。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
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
    DIRECTION_SYMMETRIC,
    LICENSE_ID,
    RELATION_PURE_ALIAS,
    RELATION_REFERS,
    REQUEST_REFERENCE_RESOLUTION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_ALIAS_LEFT,
    ROLE_ALIAS_RIGHT,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
    SCHEMA_PURE_ALIAS,
    SCHEMA_REFERS,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--alias-refers-v1"
STAGE = "W-06"
SUBSTAGE = "PURE_ALIAS_REFERS"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "DIRECTION_REVERSAL",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
_PROFILE = {
    "PURE_ALIAS": (
        RELATION_PURE_ALIAS,
        SCHEMA_PURE_ALIAS,
        DIRECTION_SYMMETRIC,
        frozenset({ROLE_ALIAS_LEFT, ROLE_ALIAS_RIGHT}),
    ),
    "REFERS": (
        RELATION_REFERS,
        SCHEMA_REFERS,
        DIRECTION_FORWARD,
        frozenset({ROLE_REFERS_FROM, ROLE_REFERS_TO}),
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
    "authored-alias-refers-seed-v1",
    "urn:pure-integer-ai:ph2:authored-alias-refers-v1",
    "Pure Integer AI PH2 authored typed alias/refers seed",
    "ALIAS_REFERS_RELATION_LABEL",
    "pure-alias-refers",
    100,
)


def _validate_profile(seed: AuthoredRelationSeed) -> None:
    """核对首个关系包的 relation/schema/direction/Role 和 consumer 坐标。"""
    profile = _PROFILE.get(seed.relation_family)
    if profile is None:
        raise AuthoredRelationCourseError("alias/refers relation family 未注册")
    relation, schema, direction, roles = profile
    if (seed.relation_kind != relation
            or seed.schema_kind != schema
            or seed.directionality != direction):
        raise AuthoredRelationCourseError("alias/refers relation profile 坐标漂移")
    actual_roles = {item.role_kind for item in seed.bindings}
    if len(seed.bindings) != 2 or actual_roles != roles:
        raise AuthoredRelationCourseError("alias/refers Role profile 漂移")
    if seed.consumer_request.request_kind != REQUEST_REFERENCE_RESOLUTION:
        raise AuthoredRelationCourseError("alias/refers consumer request kind 漂移")
    AliasRouteSearchBudget(
        seed.consumer_request.max_facts,
        seed.consumer_request.max_states,
        seed.consumer_request.max_routes,
    )
    if seed.relation_family == "PURE_ALIAS":
        kinds = {item.object_kind for item in seed.endpoints}
        if len(kinds) != 1:
            raise AuthoredRelationCourseError("PURE_ALIAS endpoint 必须同型")


def read_authored_alias_refers_seeds(
        path: str | Path) -> tuple[AuthoredRelationSeed, ...]:
    """读取规范 alias/refers JSONL，并核对覆盖面、profile 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredRelationCourseError("alias/refers sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredRelationCourseError("alias/refers sample 必须非空并以换行结束")
    seeds: list[AuthoredRelationSeed] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredRelationCourseError(
                f"alias/refers 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"alias/refers 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seed = AuthoredRelationSeed.from_dict(value)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"alias/refers 第 {line_number} 行 payload 非法") from error
        _validate_profile(seed)
        seeds.append(seed)
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise AuthoredRelationCourseError("alias/refers seed_id 重复")
    orders = [seed.logical_order for seed in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredRelationCourseError("alias/refers logical_order 必须严格递增")
    index = {seed.seed_id: seed for seed in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredRelationCourseError("alias/refers supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.relation_family != seed.relation_family):
            raise AuthoredRelationCourseError(
                "alias/refers supersede 不得跨 family/split/relation")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredRelationCourseError("alias/refers supersede 必须是 parser revision")
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
        raise AuthoredRelationCourseError("alias/refers 必须覆盖四种 sample role")
    perturbations = {seed.perturbation_kind for seed in seeds}
    if not REQUIRED_PERTURBATIONS.issubset(perturbations):
        raise AuthoredRelationCourseError("alias/refers 缺少必需反向破坏")
    supported_relations = {
        seed.relation_family for seed in seeds
        if seed.sample_role in {"support", "supersede"}
    }
    if supported_relations != set(_PROFILE):
        raise AuthoredRelationCourseError("alias/refers 两类 relation 均须有独立正例")
    return tuple(seeds)


def compile_authored_alias_refers_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02C.1 typed PURE_ALIAS/REFERS 极小 pack。"""
    seeds = read_authored_alias_refers_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_relation_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError(
            "alias/refers pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_alias_refers_course",
    "read_authored_alias_refers_seeds",
]
