"""D-02D.4 EXISTS witness、closed/open 域 typed 极小 pack。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    INSTRUCTION_EXISTS,
    OPERATOR_EXISTS,
    REQUIRED_SAMPLE_ROLES,
    ROLE_EXISTS_BODY,
    ROLE_EXISTS_VALUE,
    STRUCTURE_EXISTS,
)
from pure_integer_ai.experiments.ph2_authored_quantifier_compile import (
    compile_quantifier_seed,
)
from pure_integer_ai.experiments.ph2_authored_quantifier_schema import (
    LICENSE_ID,
    REQUEST_QUANTIFIER_EXECUTION,
    SOURCE_KEY,
    AuthoredQuantifierCourseError,
    AuthoredQuantifierSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--exists-v1"
STAGE = "W-07"
SUBSTAGE = "EXISTS"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "DOMAIN_CLOSURE_CONFUSION",
    "DOMAIN_TYPE_MISMATCH",
    "EMPTY_DOMAIN_CONFUSION",
    "QUANTIFIER_SWAP",
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
    "authored-exists-seed-v1",
    "urn:pure-integer-ai:ph2:authored-exists-v1",
    "Pure Integer AI PH2 authored typed EXISTS seed",
    "EXISTS_FINITE_DOMAIN_LABEL",
    "exists",
    100,
)


def _validate_profile(seed: AuthoredQuantifierSeed) -> None:
    """核对 EXISTS Binder、Variable、body Role、domain 类型和预算。"""
    if (seed.operator_family != "EXISTS"
            or seed.operator_kind != OPERATOR_EXISTS
            or seed.structure_kind != STRUCTURE_EXISTS
            or seed.instruction_kind != INSTRUCTION_EXISTS):
        raise AuthoredQuantifierCourseError("EXISTS operator profile 坐标漂移")
    if (seed.body_role_kind != ROLE_EXISTS_BODY
            or seed.value_role_kind != ROLE_EXISTS_VALUE):
        raise AuthoredQuantifierCourseError("EXISTS body/value Role profile 漂移")
    if seed.consumer_request.request_kind != REQUEST_QUANTIFIER_EXECUTION:
        raise AuthoredQuantifierCourseError("EXISTS consumer request kind 漂移")
    mismatched = {
        item.value_id for item in seed.domain.values
        if item.actual_type_kind != seed.value_type_kind}
    if seed.perturbation_kind == "DOMAIN_TYPE_MISMATCH":
        if not mismatched:
            raise AuthoredQuantifierCourseError(
                "EXISTS type mismatch 扰动必须有错类型 value")
    elif mismatched:
        raise AuthoredQuantifierCourseError(
            "EXISTS 非类型扰动不得混入错类型 value")


def read_authored_exists_seeds(
        path: str | Path) -> tuple[AuthoredQuantifierSeed, ...]:
    """读取规范 EXISTS JSONL，并核对 owner、domain 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredQuantifierCourseError("EXISTS sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredQuantifierCourseError(
            "EXISTS sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredQuantifierCourseError(
                f"EXISTS 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredQuantifierCourseError(
                f"EXISTS 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seed = AuthoredQuantifierSeed.from_dict(value)
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredQuantifierCourseError("EXISTS seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredQuantifierCourseError(
            "EXISTS logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredQuantifierCourseError(
                "EXISTS supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or target.operator_family != seed.operator_family):
            raise AuthoredQuantifierCourseError(
                "EXISTS supersede 不得跨 family/split/operator")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredQuantifierCourseError(
                "EXISTS supersede 必须是 parser revision")
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
        raise AuthoredQuantifierCourseError(
            "EXISTS teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredQuantifierCourseError("EXISTS 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredQuantifierCourseError("EXISTS 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_exists_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02D.4 typed EXISTS 极小 pack。"""
    seeds = read_authored_exists_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_quantifier_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredQuantifierCourseError("EXISTS pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_exists_course",
    "read_authored_exists_seeds",
]
