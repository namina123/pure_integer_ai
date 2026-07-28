"""D-02D.3 CONDITION 有序前后件四态 typed 极小 pack。"""
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
    INSTRUCTION_CONDITION,
    LICENSE_ID,
    OPERATOR_CONDITION,
    REQUEST_LOGIC_EXECUTION,
    REQUIRED_SAMPLE_ROLES,
    ROLE_CONDITION_ANTECEDENT,
    ROLE_CONDITION_CONSEQUENT,
    SOURCE_KEY,
    STRUCTURE_CONDITION,
    AuthoredLogicCourseError,
    AuthoredLogicSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--condition-v1"
STAGE = "W-07"
SUBSTAGE = "CONDITION"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "BRANCH_REPLACEMENT",
    "ANTECEDENT_CONSEQUENT_SWAP",
    "CAUSAL_CONFUSION",
    "TEMPORAL_CONFUSION",
    "PSEUDO_OPERATOR",
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
    "authored-condition-seed-v1",
    "urn:pure-integer-ai:ph2:authored-condition-v1",
    "Pure Integer AI PH2 authored typed CONDITION seed",
    "CONDITION_FOUR_STATE_LABEL",
    "condition",
    100,
)


def _validate_profile(seed: AuthoredLogicSeed) -> None:
    """核对 CONDITION 前件/后件有序 Role、operator 和预算。"""
    if (seed.operator_family != "CONDITION"
            or seed.operator_kind != OPERATOR_CONDITION
            or seed.structure_kind != STRUCTURE_CONDITION
            or seed.instruction_kind != INSTRUCTION_CONDITION):
        raise AuthoredLogicCourseError("CONDITION operator profile 坐标漂移")
    if len(seed.operands) != 2 or len(seed.bindings) != 2:
        raise AuthoredLogicCourseError("CONDITION 必须恰有两个 operand")
    if [item.role_kind for item in seed.bindings] != [
            ROLE_CONDITION_ANTECEDENT,
            ROLE_CONDITION_CONSEQUENT,
    ] or any(item.ordinal != 0 for item in seed.bindings):
        raise AuthoredLogicCourseError("CONDITION 前后件 Role profile 漂移")
    if seed.consumer_request.request_kind != REQUEST_LOGIC_EXECUTION:
        raise AuthoredLogicCourseError("CONDITION consumer request kind 漂移")
    if seed.nesting_depth != 1:
        raise AuthoredLogicCourseError("CONDITION 当前 pack 必须是单层 operator")


def read_authored_condition_seeds(
        path: str | Path) -> tuple[AuthoredLogicSeed, ...]:
    """读取规范 CONDITION JSONL，并核对 owner、扰动和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredLogicCourseError("CONDITION sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredLogicCourseError(
            "CONDITION sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredLogicCourseError(
                f"CONDITION 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredLogicCourseError(
                f"CONDITION 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seed = AuthoredLogicSeed.from_dict(value)
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredLogicCourseError("CONDITION seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredLogicCourseError("CONDITION logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredLogicCourseError(
                "CONDITION supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.operator_family != seed.operator_family):
            raise AuthoredLogicCourseError(
                "CONDITION supersede 不得跨 family/split/operator")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredLogicCourseError(
                "CONDITION supersede 必须是 parser revision")
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
            "CONDITION teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredLogicCourseError("CONDITION 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredLogicCourseError("CONDITION 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_condition_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02D.3 typed CONDITION 极小 pack。"""
    seeds = read_authored_condition_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_logic_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredLogicCourseError("CONDITION pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_condition_course",
    "read_authored_condition_seeds",
]
