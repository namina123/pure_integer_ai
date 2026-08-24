"""DLG-RAW 的公开、无标签 grounded-answer 课程投影。

本模块只消费 canonical JSONL transport 中 V3 所需的公开字段。它不构造
``GroundedAnswerEpisode``，因此不会对 ``answer_plan``、``surfaces``、
``reference_course`` 或其 surface label 执行任何语义读取、校验或分派。完整课程
raw bytes 仍由调用方的 SHA-256 content lock 覆盖；本投影只决定公开 Evidence 到
planning input 的映射。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningEvidence,
    GroundedResponseActPlanningInput,
)


_ARTIFACT_KIND = "PH2_GROUNDED_ANSWER_EPISODE_V1"
_LICENSE_ID = "CC0-1.0"
_SPLITS = frozenset({"train", "dev", "held_out"})


# object-model: exception; interop=DLG-RAW-05B
class PublicGroundedAnswerCourseProjectionError(ValueError):
    """公开课程不能无标签投影为 question/Evidence planning 输入。"""


def _text(value: Any, *, label: str) -> str:
    """接受无首尾空白的 transport text，拒绝隐式值转换。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PublicGroundedAnswerCourseProjectionError(
            f"{label} 必须是无首尾空白的非空文本")
    return value


def _positive(value: Any, *, label: str) -> int:
    """接受正严格整数，不让 bool/float 进入公开 planning。"""
    if type(value) is not int or value <= 0:
        raise PublicGroundedAnswerCourseProjectionError(
            f"{label} 必须是正严格整数")
    return value


def _bit(value: Any, *, label: str) -> int:
    """接受精确 0/1 Evidence 方向位。"""
    if type(value) is not int or value not in (0, 1):
        raise PublicGroundedAnswerCourseProjectionError(
            f"{label} 必须是严格整数 0/1")
    return value


def _field(value: dict[str, Any], key: str, *, label: str) -> Any:
    """读取一个白名单字段；缺失时 fail closed，不枚举私有标签字段。"""
    if key not in value:
        raise PublicGroundedAnswerCourseProjectionError(f"{label} 缺字段 {key}")
    return value[key]


def _evidence(value: Any, *, ordinal: int) -> GroundedResponseActPlanningEvidence:
    """从一条白名单 Evidence object 建立无标签 planning record。"""
    label = f"public course evidence[{ordinal}]"
    if not isinstance(value, dict):
        raise PublicGroundedAnswerCourseProjectionError(f"{label} 必须是 object")
    return GroundedResponseActPlanningEvidence(
        _text(_field(value, "evidence_id", label=label),
              label=f"{label}.evidence_id"),
        _text(_field(value, "proposition_id", label=label),
              label=f"{label}.proposition_id"),
        _text(_field(value, "source_id", label=label),
              label=f"{label}.source_id"),
        _positive(_field(value, "scope_id", label=label),
                  label=f"{label}.scope_id"),
        _text(_field(value, "claim_text", label=label),
              label=f"{label}.claim_text"),
        _text(_field(value, "evidence_text", label=label),
              label=f"{label}.evidence_text"),
        _bit(_field(value, "support", label=label),
             label=f"{label}.support"),
        _bit(_field(value, "refute", label=label),
             label=f"{label}.refute"),
    )


# object-model: value; representation=struct; interop=DLG-RAW-05B
@dataclass(frozen=True, slots=True)
class PublicGroundedAnswerCourseProjection:
    """一个公开 episode 的问题表面和无标签 Evidence planning 输入。"""

    episode_id: str
    question_surface: str
    planning_input: GroundedResponseActPlanningInput

    def __post_init__(self) -> None:
        """冻结 public projection 的三项可观察输入并校验交叉 identity。"""
        episode_id = _text(self.episode_id, label="public projection episode_id")
        question_surface = _text(
            self.question_surface, label="public projection question_surface")
        if not isinstance(self.planning_input, GroundedResponseActPlanningInput):
            raise TypeError("public projection planning_input 类型错误")
        if self.planning_input.episode_id != episode_id:
            raise PublicGroundedAnswerCourseProjectionError(
                "public projection episode identity 与 planning 漂移")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "question_surface", question_surface)


def _line_header(value: Any, *, line_number: int) -> tuple[str, str]:
    """仅读取每行 transport/header，供唯一 target 的选择和重复拒绝使用。

    非 target 行可能是 UNKNOWN、CONFLICT 或其他不携带 Evidence 的公开课程记录。
    因此它们不能被要求满足回答型 episode 的 Evidence 条件，更不能因而阻断 target
    episode 的无标签投影。
    """
    label = f"public course line {line_number}"
    if not isinstance(value, dict):
        raise PublicGroundedAnswerCourseProjectionError(f"{label} 根必须是 object")
    episode_id = _text(_field(value, "episode_id", label=label),
                       label=f"{label}.episode_id")
    split = _text(_field(value, "split", label=label), label=f"{label}.split")
    if split not in _SPLITS:
        raise PublicGroundedAnswerCourseProjectionError(f"{label} split 未注册")
    return episode_id, split


def _project_target_line(
        value: Any,
        *,
        episode_id: str,
        line_number: int,
        ) -> PublicGroundedAnswerCourseProjection:
    """只在命中 target 后读取其公开 question/Evidence 白名单字段。"""
    label = f"public course line {line_number}"
    if _field(value, "artifact_kind", label=label) != _ARTIFACT_KIND:
        raise PublicGroundedAnswerCourseProjectionError(f"{label} artifact_kind 漂移")
    if _field(value, "license_id", label=label) != _LICENSE_ID:
        raise PublicGroundedAnswerCourseProjectionError(f"{label} license_id 漂移")
    schema = _field(value, "schema_version", label=label)
    if type(schema) is not int or schema not in (1, 2):
        raise PublicGroundedAnswerCourseProjectionError(f"{label} schema_version 漂移")
    question = _field(value, "question", label=label)
    if not isinstance(question, dict):
        raise PublicGroundedAnswerCourseProjectionError(f"{label}.question 必须是 object")
    evidence_value = _field(question, "evidence", label=f"{label}.question")
    if not isinstance(evidence_value, list) or not evidence_value:
        raise PublicGroundedAnswerCourseProjectionError(
            f"{label}.question.evidence 必须是非空 list")
    evidence = tuple(
        _evidence(item, ordinal=index)
        for index, item in enumerate(evidence_value, start=1))
    evidence_scope = _positive(
        _field(question, "evidence_scope_id", label=f"{label}.question"),
        label=f"{label}.question.evidence_scope_id")
    response_scope = _positive(
        _field(question, "response_scope_id", label=f"{label}.question"),
        label=f"{label}.question.response_scope_id")
    if evidence_scope == response_scope:
        raise PublicGroundedAnswerCourseProjectionError(
            f"{label}.question Evidence/response scope 不得相同")
    if any(item.scope_id != evidence_scope for item in evidence):
        raise PublicGroundedAnswerCourseProjectionError(
            f"{label}.question Evidence scope 漂移")
    planning_input = GroundedResponseActPlanningInput(
        episode_id,
        _text(_field(question, "typed_intent", label=f"{label}.question"),
              label=f"{label}.question.typed_intent"),
        evidence_scope,
        response_scope,
        evidence,
    )
    projection = PublicGroundedAnswerCourseProjection(
        episode_id,
        _text(_field(question, "question_surface", label=f"{label}.question"),
              label=f"{label}.question.question_surface"),
        planning_input,
    )
    return projection


def project_public_grounded_answer_from_payload(
        payload: bytes,
        episode_id: str,
        *,
        train_only: bool = True,
        ) -> PublicGroundedAnswerCourseProjection:
    """从 content-locked canonical JSONL 选择一个公开、无标签 episode projection。

    JSON parser 在这里仅是 canonical byte transport adapter：本函数随后只读取白名单
    object 字段。它绝不构造 full course object，也不调用 answer-plan/surface/reference
    的解析器或 verifier。
    """
    if not isinstance(payload, bytes):
        raise TypeError("public grounded course payload 必须是 bytes")
    target_id = _text(episode_id, label="public projection target episode_id")
    if type(train_only) is not bool:
        raise TypeError("public projection train_only 必须是严格 bool")
    if (not payload or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise PublicGroundedAnswerCourseProjectionError(
            "public grounded course JSONL 换行非法")
    matches: list[PublicGroundedAnswerCourseProjection] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(payload[:-1].split(b"\n"), start=1):
        if not line:
            raise PublicGroundedAnswerCourseProjectionError(
                f"public grounded course 第 {line_number} 行为空")
        try:
            value = parse_canonical_json_bytes(line, require_object=True)
        except DatasetContractError as error:
            raise PublicGroundedAnswerCourseProjectionError(
                f"public grounded course 第 {line_number} 行非 canonical JSON") from error
        current_id, split = _line_header(value, line_number=line_number)
        if current_id in seen_ids:
            raise PublicGroundedAnswerCourseProjectionError(
                "public grounded course episode_id 重复")
        seen_ids.add(current_id)
        if current_id == target_id:
            if train_only and split != "train":
                raise PublicGroundedAnswerCourseProjectionError(
                    "public grounded course target 不是 TRAIN episode")
            matches.append(_project_target_line(
                value,
                episode_id=current_id,
                line_number=line_number,
            ))
    if len(matches) != 1:
        raise PublicGroundedAnswerCourseProjectionError(
            "public grounded course 找不到唯一 target episode")
    return matches[0]


__all__ = [
    "PublicGroundedAnswerCourseProjection",
    "PublicGroundedAnswerCourseProjectionError",
    "project_public_grounded_answer_from_payload",
]
