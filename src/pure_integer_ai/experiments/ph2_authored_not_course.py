"""D-02D.1 NOT 四态、target 和作用边界 typed 极小 pack。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    compile_logic_seed,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    INSTRUCTION_NOT,
    LICENSE_ID,
    OPERATOR_NOT,
    REQUEST_LOGIC_EXECUTION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_NOT_OPERAND,
    SOURCE_KEY,
    STRUCTURE_NOT,
    AuthoredLogicCourseError,
    AuthoredLogicSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--not-v1"
STAGE = "W-07"
SUBSTAGE = "NOT"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "DOUBLE_NEGATION",
    "TARGET_REPLACEMENT",
    "SCOPE_TARGET_SHIFT",
    "PSEUDO_OPERATOR",
    "CLOSED_WORLD_CONFUSION",
    "REFUTE_EVIDENCE_CONFUSION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
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
    "authored-not-seed-v1",
    "urn:pure-integer-ai:ph2:authored-not-v1",
    "Pure Integer AI PH2 authored typed NOT seed",
    "NOT_FOUR_STATE_LABEL",
    "not",
    100,
)


def _validate_profile(seed: AuthoredLogicSeed) -> None:
    """核对 NOT 一元 Role、四态 operand、嵌套深度和执行预算。"""
    if (seed.operator_family != "NOT"
            or seed.operator_kind != OPERATOR_NOT
            or seed.structure_kind != STRUCTURE_NOT
            or seed.instruction_kind != INSTRUCTION_NOT):
        raise AuthoredLogicCourseError("NOT operator profile 坐标漂移")
    if len(seed.operands) != 1 or len(seed.bindings) != 1:
        raise AuthoredLogicCourseError("NOT 必须恰有一个 operand")
    binding = seed.bindings[0]
    if binding.role_kind != ROLE_NOT_OPERAND or binding.ordinal != 0:
        raise AuthoredLogicCourseError("NOT operand Role profile 漂移")
    if binding.operand_id != seed.operands[0].operand_id:
        raise AuthoredLogicCourseError("NOT target binding 漂移")
    if seed.consumer_request.request_kind != REQUEST_LOGIC_EXECUTION:
        raise AuthoredLogicCourseError("NOT consumer request kind 漂移")
    if seed.perturbation_kind == "DOUBLE_NEGATION":
        if seed.nesting_depth != 2:
            raise AuthoredLogicCourseError("double negation 必须有两层 NOT")
    elif seed.nesting_depth != 1:
        raise AuthoredLogicCourseError("单层 NOT nesting_depth 漂移")


def read_authored_not_seeds(
        path: str | Path) -> tuple[AuthoredLogicSeed, ...]:
    """读取规范 NOT JSONL，并核对 owner、扰动和 supersede 恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredLogicCourseError("NOT sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredLogicCourseError("NOT sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredLogicCourseError(
                f"NOT 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredLogicCourseError(
                f"NOT 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seed = AuthoredLogicSeed.from_dict(value)
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredLogicCourseError("NOT seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredLogicCourseError("NOT logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredLogicCourseError("NOT supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.operator_family != seed.operator_family):
            raise AuthoredLogicCourseError(
                "NOT supersede 不得跨 family/split/operator")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredLogicCourseError(
                "NOT supersede 必须是 parser revision")
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
        raise AuthoredLogicCourseError(
            "NOT teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredLogicCourseError("NOT 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredLogicCourseError("NOT 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_not_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02D.1 typed NOT 极小 pack。"""
    seeds = read_authored_not_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_logic_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredLogicCourseError("NOT pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_not_course",
    "read_authored_not_seeds",
]
