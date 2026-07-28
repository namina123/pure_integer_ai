"""D-02E.2 显式事实、关系、逻辑、reference 与不可答 QA 极小 pack。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_qa_compile import compile_qa_seed
from pure_integer_ai.experiments.ph2_authored_qa_schema import (
    LICENSE_ID,
    QUESTION_KINDS,
    REQUIRED_SAMPLE_ROLES,
    SOURCE_KEY,
    AuthoredQACourseError,
    AuthoredQASeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--question-answer-v1"
STAGE = "W-09"
SUBSTAGE = "QUESTION_ANSWER"
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
    "authored-question-answer-seed-v1",
    "urn:pure-integer-ai:ph2:authored-question-answer-v1",
    "Pure Integer AI PH2 authored typed question answer seed",
    "QUESTION_ANSWER_LABEL",
    "question_answer",
    100,
)


def read_authored_qa_seeds(path: str | Path) -> tuple[AuthoredQASeed, ...]:
    """读取规范 QA JSONL，并核对问题族、owner、扰动和恢复链。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredQACourseError("QA sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredQACourseError("QA sample 必须非空并以换行结束")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredQACourseError(
                f"QA 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredQACourseError(
                f"QA 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seeds.append(AuthoredQASeed.from_dict(value))
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredQACourseError("QA seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredQACourseError("QA logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredQACourseError("QA supersede 必须指向更早 seed")
        if target.family != seed.family or target.split != seed.split:
            raise AuthoredQACourseError("QA supersede 不得跨 family/split")
        if seed.perturbation_kind != "PARSER_REVISION":
            raise AuthoredQACourseError(
                "QA supersede 必须是 parser revision 扰动")
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
        raise AuthoredQACourseError(
            "QA teacher/evaluator family 必须非空且互斥")
    if {item.sample_role for item in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredQACourseError("QA 必须覆盖四种 sample role")
    if {item.question_kind for item in seeds} != QUESTION_KINDS:
        raise AuthoredQACourseError("QA 必须精确覆盖全部 question kind")
    if not REQUIRED_PERTURBATIONS.issubset({
            item.perturbation_kind for item in seeds}):
        raise AuthoredQACourseError("QA 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_qa_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 D-02E.2 typed QA 极小 pack。"""
    seeds = read_authored_qa_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_qa_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredQACourseError("QA pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "REQUIRED_PERTURBATIONS",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_qa_course",
    "read_authored_qa_seeds",
]
