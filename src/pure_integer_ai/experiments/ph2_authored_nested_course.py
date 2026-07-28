"""D-02D.7 异构 operator 嵌套、scope 翻转和 lexical Binder 极小 pack。"""
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
    INSTRUCTION_FORALL,
    INSTRUCTION_MODAL,
    INSTRUCTION_NOT,
    OPERATOR_EXISTS,
    OPERATOR_FORALL,
    OPERATOR_MODAL,
    OPERATOR_NOT,
    REQUIRED_SAMPLE_ROLES,
    ROLE_EXISTS_BODY,
    ROLE_EXISTS_VALUE,
    ROLE_FORALL_BODY,
    ROLE_FORALL_VALUE,
    ROLE_MODAL_CHILD,
    ROLE_NOT_OPERAND,
    SOURCE_KEY,
    STRUCTURE_EXISTS,
    STRUCTURE_FORALL,
    STRUCTURE_MODAL,
    STRUCTURE_NOT,
)
from pure_integer_ai.experiments.ph2_authored_nested_compile import (
    compile_nested_seed,
)
from pure_integer_ai.experiments.ph2_authored_nested_schema import (
    AuthoredNestedCourseError,
    AuthoredNestedSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--nested-scope-v1"
STAGE = "W-07"
SUBSTAGE = "NESTED_SCOPE"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "MODAL_SCOPE_SHIFT",
    "QUANTIFIER_SWAP",
    "MISSING_INNER_OPERATOR",
    "BUDGET_UNDECIDED",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
    "DEPTH_REPLACEMENT",
    "PSEUDO_OPERATOR",
})
_SPEC = AuthoredCourseSpec(
    SOURCE_KEY,
    "CC0-1.0",
    1,
    1,
    1,
    1,
    1,
    PACK_NAME,
    STAGE,
    SUBSTAGE,
    "authored-nested-scope-seed-v1",
    "urn:pure-integer-ai:ph2:authored-nested-scope-v1",
    "Pure Integer AI PH2 authored typed nested scope seed",
    "NESTED_SCOPE_LABEL",
    "nested_scope",
    100,
)
_PROFILES = {
    "NOT": (
        OPERATOR_NOT,
        STRUCTURE_NOT,
        INSTRUCTION_NOT,
        ROLE_NOT_OPERAND,
    ),
    "MODAL": (
        OPERATOR_MODAL,
        STRUCTURE_MODAL,
        INSTRUCTION_MODAL,
        ROLE_MODAL_CHILD,
    ),
    "EXISTS": (
        OPERATOR_EXISTS,
        STRUCTURE_EXISTS,
        INSTRUCTION_EXISTS,
        ROLE_EXISTS_BODY,
    ),
    "FORALL": (
        OPERATOR_FORALL,
        STRUCTURE_FORALL,
        INSTRUCTION_FORALL,
        ROLE_FORALL_BODY,
    ),
}


def _validate_profile(seed: AuthoredNestedSeed) -> None:
    """核对每层 operator 坐标、inner availability 和量词 value Role。"""
    for layer in seed.layers:
        expected = _PROFILES.get(layer.operator_family)
        actual = (
            layer.operator_kind,
            layer.structure_kind,
            layer.instruction_kind,
            layer.role_kind,
        )
        if expected is None or actual != expected:
            raise AuthoredNestedCourseError(
                "nested operator profile 坐标漂移")
    unavailable = [
        index for index, layer in enumerate(seed.layers)
        if not layer.candidate_available]
    if seed.perturbation_kind == "MISSING_INNER_OPERATOR":
        if len(unavailable) != 1 or unavailable[0] == 0:
            raise AuthoredNestedCourseError(
                "missing inner operator 必须只缺一个内层 candidate")
    elif unavailable:
        raise AuthoredNestedCourseError(
            "非 missing 扰动不得隐藏 nested candidate")
    if seed.perturbation_kind == "DEPTH_REPLACEMENT" and len(seed.layers) < 3:
        raise AuthoredNestedCourseError(
            "depth replacement 必须增加到至少三层")
    if seed.quantifier is not None:
        layer = next(
            item for item in seed.layers
            if item.layer_id == seed.quantifier.layer_id)
        expected_value_role = (
            ROLE_EXISTS_VALUE
            if layer.operator_family == "EXISTS"
            else ROLE_FORALL_VALUE
        )
        if seed.quantifier.value_role_kind != expected_value_role:
            raise AuthoredNestedCourseError(
                "nested quantifier value Role 漂移")


def read_authored_nested_seeds(
        path: str | Path) -> tuple[AuthoredNestedSeed, ...]:
    """读取规范 nested JSONL，并核对 owner、层序、来源簇和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredNestedCourseError("nested sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredNestedCourseError(
            "nested sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredNestedCourseError(
                f"nested 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredNestedCourseError(
                f"nested 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seed = AuthoredNestedSeed.from_dict(value)
        _validate_profile(seed)
        seeds.append(seed)
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredNestedCourseError("nested seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredNestedCourseError(
            "nested logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredNestedCourseError(
                "nested supersede 必须指向更早 seed")
        if (target.family != seed.family
                or target.split != seed.split
                or tuple(item.operator_family for item in target.layers)
                != tuple(item.operator_family for item in seed.layers)):
            raise AuthoredNestedCourseError(
                "nested supersede 不得跨 family/split/layer chain")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredNestedCourseError(
                "nested supersede 必须是 parser revision")
    teacher_families = {
        item.family for item in seeds if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in seeds if item.label_owner == "evaluator"}
    teacher_templates = {
        item.template_family for item in seeds
        if item.label_owner == "teacher"}
    evaluator_templates = {
        item.template_family for item in seeds
        if item.label_owner == "evaluator"}
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates):
        raise AuthoredNestedCourseError(
            "nested teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredNestedCourseError("nested 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredNestedCourseError("nested 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_nested_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02D.7 typed nested scope 极小 pack。"""
    seeds = read_authored_nested_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_nested_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredNestedCourseError("nested pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_nested_course",
    "read_authored_nested_seeds",
]
