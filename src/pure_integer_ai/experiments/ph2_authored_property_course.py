"""D-02C.3 PROPERTY 六维 typed 极小 pack 的读取、校验和发布。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
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
    RELATION_PROPERTY,
    REQUEST_PROPERTY_SELECTION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_INTENSITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_VALUE,
    SCHEMA_PROPERTY,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--property-v1"
STAGE = "W-06"
SUBSTAGE = "PROPERTY"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "ROLE_MISMATCH",
    "VALUE_REPLACEMENT",
    "INTENSITY_REPLACEMENT",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
_INTENSITY_FIELDS = frozenset({"den", "num"})
_ROLE_KINDS = frozenset({
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_VALUE,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_INTENSITY,
})
_ALLOWED_BY_ROLE = {
    ROLE_PROPERTY_SUBJECT: frozenset({OBJECT_ENTITY}),
    ROLE_PROPERTY_ATTRIBUTE: frozenset({OBJECT_CONCEPT}),
    ROLE_PROPERTY_VALUE: frozenset({OBJECT_CONCEPT, OBJECT_ENTITY}),
    ROLE_PROPERTY_POLARITY: frozenset({OBJECT_CONCEPT}),
    ROLE_PROPERTY_MODALITY: frozenset({OBJECT_CONCEPT}),
    ROLE_PROPERTY_INTENSITY: frozenset({OBJECT_CONCEPT}),
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
    "authored-property-seed-v1",
    "urn:pure-integer-ai:ph2:authored-property-v1",
    "Pure Integer AI PH2 authored typed property seed",
    "PROPERTY_RELATION_LABEL",
    "property",
    100,
)


@dataclass(frozen=True)
class AuthoredPropertySeed:
    """一条共享 relation seed 及其显式 intensity Rational 解释。"""

    relation: AuthoredRelationSeed
    intensity_num: int
    intensity_den: int

    def __post_init__(self) -> None:
        if not isinstance(self.relation, AuthoredRelationSeed):
            raise TypeError("property relation seed 类型错误")
        if (type(self.intensity_num) is not int or self.intensity_num <= 0
                or type(self.intensity_den) is not int
                or self.intensity_den <= 0):
            raise AuthoredRelationCourseError(
                "property intensity 必须是正严格整数 Rational")

    @classmethod
    def from_dict(cls, value: Any) -> "AuthoredPropertySeed":
        """拆出 PROPERTY 专属 Rational，再交给共享 relation 合同。"""
        if not isinstance(value, dict) or "property_intensity" not in value:
            raise AuthoredRelationCourseError("property seed 缺少 intensity")
        intensity = value["property_intensity"]
        if not isinstance(intensity, dict) or set(intensity) != _INTENSITY_FIELDS:
            raise AuthoredRelationCourseError("property intensity 字段集合漂移")
        relation_value = dict(value)
        del relation_value["property_intensity"]
        return cls(
            AuthoredRelationSeed.from_dict(relation_value),
            intensity["num"],
            intensity["den"],
        )


def _validate_profile(seed: AuthoredPropertySeed) -> None:
    """核对 R-03 六 Role、精确 slot 类型、查询锚和 intensity filler。"""
    relation = seed.relation
    if (relation.relation_family != "PROPERTY"
            or relation.relation_kind != RELATION_PROPERTY
            or relation.schema_kind != SCHEMA_PROPERTY
            or relation.directionality != DIRECTION_FORWARD):
        raise AuthoredRelationCourseError("PROPERTY relation profile 坐标漂移")
    if len(relation.bindings) != 6 or len(relation.endpoints) != 6:
        raise AuthoredRelationCourseError("PROPERTY 必须恰有六个 Role filler")
    binding_by_role = {item.role_kind: item for item in relation.bindings}
    if set(binding_by_role) != _ROLE_KINDS:
        raise AuthoredRelationCourseError("PROPERTY 六 Role profile 漂移")
    for role, allowed in _ALLOWED_BY_ROLE.items():
        if frozenset(binding_by_role[role].allowed_object_kinds) != allowed:
            raise AuthoredRelationCourseError("PROPERTY Role slot 类型漂移")
    endpoint_index = {
        item.endpoint_id: item for item in relation.endpoints
    }
    endpoint_by_role = {
        role: endpoint_index[binding.endpoint_id]
        for role, binding in binding_by_role.items()
    }
    if any(endpoint_by_role[role].object_kind not in allowed
           for role, allowed in _ALLOWED_BY_ROLE.items()):
        raise AuthoredRelationCourseError("PROPERTY endpoint 类型非法")
    request = relation.consumer_request
    if request.request_kind != REQUEST_PROPERTY_SELECTION:
        raise AuthoredRelationCourseError("PROPERTY consumer request kind 漂移")
    if request.origin_endpoint_id != binding_by_role[
            ROLE_PROPERTY_SUBJECT].endpoint_id:
        raise AuthoredRelationCourseError("PROPERTY query subject 锚漂移")
    expected_attribute = binding_by_role[
        ROLE_PROPERTY_ATTRIBUTE].endpoint_id
    if relation.perturbation_kind == "ROLE_MISMATCH":
        if (request.attribute_endpoint_id == expected_attribute
                or request.attribute_endpoint_id != binding_by_role[
                    ROLE_PROPERTY_VALUE].endpoint_id):
            raise AuthoredRelationCourseError(
                "PROPERTY Role mismatch 未只交换 attribute/value")
    elif request.attribute_endpoint_id != expected_attribute:
        raise AuthoredRelationCourseError("PROPERTY query attribute 锚漂移")


def read_authored_property_seeds(
        path: str | Path) -> tuple[AuthoredPropertySeed, ...]:
    """读取规范 PROPERTY JSONL，并核对覆盖面、owner 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredRelationCourseError("property sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredRelationCourseError("property sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredRelationCourseError(
                f"property 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"property 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seed = AuthoredPropertySeed.from_dict(value)
        except DatasetContractError as error:
            raise AuthoredRelationCourseError(
                f"property 第 {line_number} 行 payload 非法") from error
        _validate_profile(seed)
        seeds.append(seed)
    relations = [item.relation for item in seeds]
    if len({item.seed_id for item in relations}) != len(relations):
        raise AuthoredRelationCourseError("property seed_id 重复")
    orders = [item.logical_order for item in relations]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredRelationCourseError("property logical_order 必须严格递增")
    index = {item.seed_id: item for item in relations}
    for relation in relations:
        if not relation.supersedes_seed_id:
            continue
        target = index.get(relation.supersedes_seed_id)
        if target is None or target.logical_order >= relation.logical_order:
            raise AuthoredRelationCourseError("property supersede 必须指向更早 seed")
        if (target.family != relation.family
                or target.split != relation.split
                or target.relation_family != relation.relation_family):
            raise AuthoredRelationCourseError(
                "property supersede 不得跨 family/split/relation")
        if relation.perturbation_kind != "PARSER_REVISION":
            raise AuthoredRelationCourseError("property supersede 必须是 parser revision")
    teacher_families = {
        item.family for item in relations if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in relations if item.label_owner == "evaluator"}
    teacher_templates = {
        item.template_family for item in relations
        if item.label_owner == "teacher"}
    evaluator_templates = {
        item.template_family for item in relations
        if item.label_owner == "evaluator"}
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates):
        raise AuthoredRelationCourseError(
            "property teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in relations} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredRelationCourseError("property 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in relations}):
        raise AuthoredRelationCourseError("property 缺少必需反向破坏")
    if not any(item.sample_role in {"support", "supersede"}
               for item in relations):
        raise AuthoredRelationCourseError("PROPERTY 必须有独立正例")
    return tuple(seeds)


def compile_authored_property_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02C.3 六维 typed PROPERTY 极小 pack。"""
    seeds = read_authored_property_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_relation_seed(
                item.relation,
                rational_role_values=((
                    ROLE_PROPERTY_INTENSITY,
                    item.intensity_num,
                    item.intensity_den,
                ),),
            ) for item in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError("property pack 发布失败") from error


__all__ = [
    "AuthoredPropertySeed",
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_property_course",
    "read_authored_property_seeds",
]
