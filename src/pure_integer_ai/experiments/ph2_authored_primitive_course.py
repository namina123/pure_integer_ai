"""把 AUTHORED_CC0_V1 极小 seed 编译为 W-04 原语与 surface 对应 pack。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.operator_primitives import (
    OP_ADD,
    OP_GE,
    OP_GT,
    OP_LE,
    OP_LT,
    OP_MUL,
    OP_SUB,
)
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_CAUSES,
    REL_EQUAL,
    REL_MEMBER,
    REL_MEREOLOGY,
    REL_PRECEDES,
    REL_PROPERTY,
    REL_SIMILAR,
    REL_SUBSET,
)
from pure_integer_ai.cognition.shared.symbol_types import (
    TYPE_ATTR_MARKER,
    TYPE_CAUSES,
    TYPE_CMP,
    TYPE_COPULA,
    TYPE_NEGATION,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
    DatasetContractError,
    parse_canonical_json_bytes,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--primitive-v1"
STAGE = "W-04"
SUBSTAGE = "PRIMITIVE_SURFACE_MAPPING"

_ALLOWED_PRIMITIVES = {
    "relation": frozenset({
        REL_SUBSET, REL_MEMBER, REL_EQUAL, REL_CAUSES, REL_PRECEDES,
        REL_MEREOLOGY, REL_PROPERTY, REL_SIMILAR,
    }),
    "operator": frozenset({OP_ADD, OP_SUB, OP_MUL, OP_GT, OP_LT, OP_GE, OP_LE}),
    "symbol_type": frozenset({
        TYPE_NEGATION, TYPE_COPULA, TYPE_CMP, TYPE_CAUSES, TYPE_ATTR_MARKER,
    }),
}
_SEED_FIELDS = frozenset({
    "context",
    "expected_payload",
    "expected_state",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "perturbation_kind",
    "primitive_kind",
    "primitive_registry",
    "sample_role",
    "seed_id",
    "split",
    "supersedes_seed_id",
    "surface_form",
    "template_family",
})
_REQUIRED_ROLES = frozenset({"support", "refute", "conflict", "supersede"})

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
    "authored-primitive-seed-v1",
    "urn:pure-integer-ai:ph2:authored-primitive-v1",
    "Pure Integer AI PH2 authored primitive seed",
    "PRIMITIVE_SURFACE_LABEL",
    "primitive-surface-mapping",
    100,
)


class AuthoredPrimitiveCourseError(RuntimeError):
    """原创 primitive seed 的坐标、owner、许可或替代序非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 primitive seed 字段为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredPrimitiveCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredPrimitiveCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求 primitive kind 和逻辑序为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredPrimitiveCourseError(f"{where} 必须是正严格整数")
    return value


@dataclass(frozen=True)
class PrimitiveSurfaceSeed:
    """一条文字 alias 到既有冻结整数原语坐标的私有 owner seed。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    surface_form: str
    context: str
    primitive_registry: str
    primitive_kind: int
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("surface_form", self.surface_form),
                ("context", self.context),
                ("primitive_registry", self.primitive_registry),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"PrimitiveSurfaceSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="PrimitiveSurfaceSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredPrimitiveCourseError("label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredPrimitiveCourseError("label_owner 与 split 不一致")
        if self.sample_role not in _REQUIRED_ROLES:
            raise AuthoredPrimitiveCourseError("sample_role 不属于 primitive 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredPrimitiveCourseError("supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredPrimitiveCourseError("非 supersede seed 不得声明替代目标")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredPrimitiveCourseError("expected_state 非四态")
        _positive_int(self.primitive_kind, where="PrimitiveSurfaceSeed.primitive_kind")
        allowed = _ALLOWED_PRIMITIVES.get(self.primitive_registry)
        if allowed is None or self.primitive_kind not in allowed:
            raise AuthoredPrimitiveCourseError("primitive registry/kind 不是 active 冻结坐标")
        _positive_int(self.logical_order, where="PrimitiveSurfaceSeed.logical_order")

    def compiled(self) -> AuthoredCompiledSeed:
        """去除私有 seed 外壳并生成共用发布器所需 typed seed。"""
        coordinate = {
            "kind": self.primitive_kind,
            "registry": self.primitive_registry,
        }
        return AuthoredCompiledSeed(
            self.seed_id,
            self.family,
            self.template_family,
            self.label_owner,
            self.split,
            self.sample_role,
            "PrimitiveSurfaceQuery",
            CanonicalJsonObject.from_value({
                "candidate_primitive": coordinate,
                "context": self.context,
                "query_kind": "primitive_surface_mapping",
                "surface_form": self.surface_form,
            }),
            self.expected_state,
            self.expected_payload,
            self.perturbation_kind,
            self.supersedes_seed_id,
            self.logical_order,
            (
                self.surface_form, self.context,
                self.primitive_registry, self.primitive_kind,
            ),
            (self.surface_form, self.context),
            ("primitive_surface_query_v1", self.primitive_registry),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PrimitiveSurfaceSeed":
        """从严格字段集合恢复 primitive seed。"""
        if set(value) != _SEED_FIELDS:
            raise AuthoredPrimitiveCourseError("primitive seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredPrimitiveCourseError("primitive seed 必须是 CC0-1.0")
        return cls(
            str(value["seed_id"]),
            str(value["family"]),
            str(value["template_family"]),
            str(value["label_owner"]),
            str(value["split"]),
            str(value["sample_role"]),
            str(value["surface_form"]),
            str(value["context"]),
            str(value["primitive_registry"]),
            value["primitive_kind"],
            str(value["expected_state"]),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            str(value["perturbation_kind"]),
            str(value["supersedes_seed_id"]),
            value["logical_order"],
        )


def read_authored_primitive_seeds(
        path: str | Path) -> tuple[PrimitiveSurfaceSeed, ...]:
    """读取规范 primitive JSONL，并核对 family、角色和 supersede 顺序。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredPrimitiveCourseError("primitive sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredPrimitiveCourseError("primitive sample 必须非空并以换行结束")
    seeds: list[PrimitiveSurfaceSeed] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredPrimitiveCourseError(
                f"primitive sample 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredPrimitiveCourseError(
                f"primitive sample 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seeds.append(PrimitiveSurfaceSeed.from_dict(value))
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise AuthoredPrimitiveCourseError("primitive seed_id 重复")
    orders = [seed.logical_order for seed in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredPrimitiveCourseError("primitive logical_order 必须严格递增")
    index = {seed.seed_id: seed for seed in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredPrimitiveCourseError("primitive supersede 必须指向更早 seed")
        if target.family != seed.family or target.split != seed.split:
            raise AuthoredPrimitiveCourseError("primitive supersede 不得跨 family/split")
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
        raise AuthoredPrimitiveCourseError("teacher/evaluator family 必须非空且互斥")
    if {seed.sample_role for seed in seeds} != _REQUIRED_ROLES:
        raise AuthoredPrimitiveCourseError("primitive sample 必须覆盖四种 sample role")
    return tuple(seeds)


def compile_authored_primitive_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 W-04 原语/surface 极小 pack，不把 cue 命中写成判据。"""
    seeds = read_authored_primitive_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(seed.compiled() for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredPrimitiveCourseError(
            "primitive pack 发布失败") from error


__all__ = [
    "AuthoredPrimitiveCourseError",
    "LICENSE_ID",
    "PACK_NAME",
    "PrimitiveSurfaceSeed",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_primitive_course",
    "read_authored_primitive_seeds",
]
