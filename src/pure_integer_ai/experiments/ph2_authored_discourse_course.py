"""D-02E.1 中文变体、篇章、reference 和持续修正极小 pack。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_discourse_compile import (
    compile_discourse_seed,
)
from pure_integer_ai.experiments.ph2_authored_discourse_schema import (
    LICENSE_ID,
    REQUIRED_SAMPLE_ROLES,
    SOURCE_KEY,
    VARIANT_KINDS,
    AuthoredDiscourseCourseError,
    AuthoredDiscourseSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--discourse-revision-v1"
STAGE = "W-08"
SUBSTAGE = "DISCOURSE_REVISION"
REQUIRED_PERTURBATIONS = frozenset({
    "NONE",
    "CONTENT_REPLACEMENT",
    "TARGET_REPLACEMENT",
    "SCOPE_TARGET_SHIFT",
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
    "authored-discourse-revision-seed-v1",
    "urn:pure-integer-ai:ph2:authored-discourse-revision-v1",
    "Pure Integer AI PH2 authored Chinese discourse and revision seed",
    "DISCOURSE_REVISION_LABEL",
    "discourse_revision",
    100,
)


def read_authored_discourse_seeds(
        path: str | Path) -> tuple[AuthoredDiscourseSeed, ...]:
    """读取规范 discourse JSONL，并核对隔离、覆盖与恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredDiscourseCourseError(
            "discourse sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredDiscourseCourseError(
            "discourse sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredDiscourseCourseError(
                f"discourse 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredDiscourseCourseError(
                f"discourse 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seeds.append(AuthoredDiscourseSeed.from_dict(value))
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredDiscourseCourseError("discourse seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredDiscourseCourseError(
            "discourse logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredDiscourseCourseError(
                "discourse supersede 必须指向更早 seed")
        if target.family != seed.family or target.split != seed.split:
            raise AuthoredDiscourseCourseError(
                "discourse supersede 不得跨 family/split")
        if (seed.perturbation_kind != "PARSER_REVISION"
                or seed.parser_revision is None):
            raise AuthoredDiscourseCourseError(
                "discourse supersede 必须携带 parser revision")
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
        raise AuthoredDiscourseCourseError(
            "discourse teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredDiscourseCourseError(
            "discourse 必须覆盖四种 sample role")
    if {item.variant_kind for item in seeds} != VARIANT_KINDS:
        raise AuthoredDiscourseCourseError(
            "discourse 必须精确覆盖全部九类 variant")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredDiscourseCourseError(
            "discourse 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_discourse_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02E.1 discourse/revision 极小 pack。"""
    seeds = read_authored_discourse_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_discourse_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredDiscourseCourseError(
            "discourse pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_discourse_course",
    "read_authored_discourse_seeds",
]
