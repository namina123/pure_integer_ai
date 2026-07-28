"""D-02D.6 MODAL 独立 resolver、scope 和预算极小 pack。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.identity import OBJECT_PROPOSITION
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    INSTRUCTION_MODAL,
    LICENSE_ID,
    OPERATOR_MODAL,
    REQUIRED_SAMPLE_ROLES,
    ROLE_MODAL_CHILD,
    SOURCE_KEY,
    STRUCTURE_MODAL,
)
from pure_integer_ai.experiments.ph2_authored_modal_compile import (
    compile_modal_seed,
)
from pure_integer_ai.experiments.ph2_authored_modal_schema import (
    AuthoredModalCourseError,
    AuthoredModalSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--modal-v1"
STAGE = "W-07"
SUBSTAGE = "MODAL"
REQUIRED_PERTURBATIONS = frozenset({
    "CONTENT_REPLACEMENT",
    "MODAL_SCOPE_SHIFT",
    "PSEUDO_OPERATOR",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
    "RESOLVER_MISSING",
    "RESOLVER_DENIED",
    "BUDGET_UNDECIDED",
    "BRANCH_REPLACEMENT",
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
    "authored-modal-seed-v1",
    "urn:pure-integer-ai:ph2:authored-modal-v1",
    "Pure Integer AI PH2 authored typed MODAL seed",
    "MODAL_RESOLVER_LABEL",
    "modal",
    100,
)


def _validate_profile(seed: AuthoredModalSeed) -> None:
    """核对 MODAL unary Proposition、Role、坐标和 resolver 预算。"""
    logic = seed.logic
    if (logic.operator_family != "MODAL"
            or logic.operator_kind != OPERATOR_MODAL
            or logic.structure_kind != STRUCTURE_MODAL
            or logic.instruction_kind != INSTRUCTION_MODAL):
        raise AuthoredModalCourseError("MODAL operator profile 坐标漂移")
    if (len(logic.operands) != 1 or len(logic.bindings) != 1
            or logic.bindings[0].role_kind != ROLE_MODAL_CHILD
            or logic.bindings[0].ordinal != 0
            or logic.operands[0].object_kind != OBJECT_PROPOSITION):
        raise AuthoredModalCourseError(
            "MODAL 必须有一个 Proposition child Role")
    if logic.nesting_depth != 1:
        raise AuthoredModalCourseError("MODAL 单包不得提前嵌套作用域")


def read_authored_modal_seeds(
        path: str | Path) -> tuple[AuthoredModalSeed, ...]:
    """读取规范 MODAL JSONL，并核对 owner、resolver 和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredModalCourseError("MODAL sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredModalCourseError(
            "MODAL sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredModalCourseError(
                f"MODAL 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredModalCourseError(
                f"MODAL 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seed = AuthoredModalSeed.from_dict(value)
        _validate_profile(seed)
        seeds.append(seed)
    logic_seeds = [item.logic for item in seeds]
    if len({item.seed_id for item in logic_seeds}) != len(logic_seeds):
        raise AuthoredModalCourseError("MODAL seed_id 重复")
    orders = [item.logical_order for item in logic_seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredModalCourseError(
            "MODAL logical_order 必须严格递增")
    index = {item.seed_id: item for item in logic_seeds}
    for logic in logic_seeds:
        if not logic.supersedes_seed_id:
            continue
        target = index.get(logic.supersedes_seed_id)
        if target is None or target.logical_order >= logic.logical_order:
            raise AuthoredModalCourseError(
                "MODAL supersede 必须指向更早 seed")
        if (target.family != logic.family
                or target.split != logic.split
                or target.operator_family != logic.operator_family):
            raise AuthoredModalCourseError(
                "MODAL supersede 不得跨 family/split/operator")
        if logic.perturbation_kind != "PARSER_REVISION":
            raise AuthoredModalCourseError(
                "MODAL supersede 必须是 parser revision")
    teacher_families = {
        item.family for item in logic_seeds if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in logic_seeds if item.label_owner == "evaluator"}
    teacher_templates = {
        item.template_family for item in logic_seeds
        if item.label_owner == "teacher"}
    evaluator_templates = {
        item.template_family for item in logic_seeds
        if item.label_owner == "evaluator"}
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates):
        raise AuthoredModalCourseError(
            "MODAL teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in logic_seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredModalCourseError("MODAL 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in logic_seeds}):
        raise AuthoredModalCourseError("MODAL 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_modal_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02D.6 typed MODAL 极小 pack。"""
    seeds = read_authored_modal_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_modal_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredModalCourseError("MODAL pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_modal_course",
    "read_authored_modal_seeds",
]
