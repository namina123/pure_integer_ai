"""D-02D.2 AND/OR 多分支开放世界 typed 极小 pack。"""
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
    INSTRUCTION_AND,
    INSTRUCTION_OR,
    LICENSE_ID,
    OPERATOR_AND,
    OPERATOR_OR,
    REQUEST_LOGIC_EXECUTION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_AND_OPERAND,
    ROLE_OR_OPERAND,
    SOURCE_KEY,
    STRUCTURE_AND,
    STRUCTURE_OR,
    AuthoredLogicCourseError,
    AuthoredLogicSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--and-or-v1"
STAGE = "W-07"
SUBSTAGE = "AND_OR"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "OPERAND_ORDER_SWAP",
    "BRANCH_REPLACEMENT",
    "OPERATOR_CONFUSION",
    "PSEUDO_OPERATOR",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
})
_PROFILE = {
    "AND": (OPERATOR_AND, STRUCTURE_AND, INSTRUCTION_AND, ROLE_AND_OPERAND),
    "OR": (OPERATOR_OR, STRUCTURE_OR, INSTRUCTION_OR, ROLE_OR_OPERAND),
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
    "authored-and-or-seed-v1",
    "urn:pure-integer-ai:ph2:authored-and-or-v1",
    "Pure Integer AI PH2 authored typed AND OR seed",
    "AND_OR_FOUR_STATE_LABEL",
    "and-or",
    100,
)


def _validate_profile(seed: AuthoredLogicSeed) -> None:
    """核对 AND/OR 双分支 Role、operator 和执行预算。"""
    profile = _PROFILE.get(seed.operator_family)
    if profile is None:
        raise AuthoredLogicCourseError("AND/OR operator family 未注册")
    if (seed.operator_kind != profile[0]
            or seed.structure_kind != profile[1]
            or seed.instruction_kind != profile[2]):
        raise AuthoredLogicCourseError("AND/OR operator profile 坐标漂移")
    if len(seed.operands) != 2 or len(seed.bindings) != 2:
        raise AuthoredLogicCourseError("AND/OR 必须恰有两个 operand")
    if ({item.role_kind for item in seed.bindings} != {profile[3]}
            or {item.ordinal for item in seed.bindings} != {0, 1}):
        raise AuthoredLogicCourseError("AND/OR operand Role profile 漂移")
    if seed.consumer_request.request_kind != REQUEST_LOGIC_EXECUTION:
        raise AuthoredLogicCourseError("AND/OR consumer request kind 漂移")
    if seed.nesting_depth != 1:
        raise AuthoredLogicCourseError("AND/OR 当前 pack 必须是单层 operator")


def read_authored_and_or_seeds(
        path: str | Path) -> tuple[AuthoredLogicSeed, ...]:
    """读取规范 AND/OR JSONL，并核对双 family、owner 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredLogicCourseError("AND/OR sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredLogicCourseError("AND/OR sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredLogicCourseError(
                f"AND/OR 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredLogicCourseError(
                f"AND/OR 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seed = AuthoredLogicSeed.from_dict(value)
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredLogicCourseError("AND/OR seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredLogicCourseError("AND/OR logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredLogicCourseError("AND/OR supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.operator_family != seed.operator_family):
            raise AuthoredLogicCourseError(
                "AND/OR supersede 不得跨 family/split/operator")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredLogicCourseError(
                "AND/OR supersede 必须是 parser revision")
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
            "AND/OR teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredLogicCourseError("AND/OR 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredLogicCourseError("AND/OR 缺少必需反向破坏")
    if {item.operator_family for item in seeds} != set(_PROFILE):
        raise AuthoredLogicCourseError("AND 与 OR 必须全部覆盖")
    return tuple(seeds)


def compile_authored_and_or_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02D.2 typed AND/OR 极小 pack。"""
    seeds = read_authored_and_or_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_logic_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredLogicCourseError("AND/OR pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_and_or_course",
    "read_authored_and_or_seeds",
]
