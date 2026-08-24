"""DLG-RAW-05B 公开课程字段选择投影专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedAnswerEpisode,
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    public_response_act_planning_input_from_episode,
)
from pure_integer_ai.experiments.public_grounded_answer_course_projection import (
    PublicGroundedAnswerCourseProjectionError,
    project_public_grounded_answer_from_payload,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _REPOSITORY_ROOT / "data/ph2/grounded_answer_train_v1.jsonl.sample"
_EPISODE_ID = "train-grounded-reference-event-v2"


def _ignored_fields_mutated_payload() -> bytes:
    """仅改写禁入字段，保持顶层 JSONL transport 为 canonical bytes。"""
    lines = []
    for line in _SAMPLE.read_bytes().splitlines():
        value = parse_canonical_json_bytes(line, require_object=True)
        if value["episode_id"] == _EPISODE_ID:
            value["question"]["answer_plan"] = {"ignored": [1, 2, 3]}
            value["reference_course"] = {"ignored": "reference labels"}
            value["surfaces"] = {"ignored": {"accepted": "not consulted"}}
        lines.append(canonical_json_line(value))
    return b"".join(lines)


def _public_field_mutated_payload(kind: str) -> bytes:
    """只变更目标 episode 的一个白名单字段，保留其他 raw 记录。"""
    lines = []
    for line in _SAMPLE.read_bytes().splitlines():
        value = parse_canonical_json_bytes(line, require_object=True)
        if value["episode_id"] == _EPISODE_ID:
            if kind == "public-text":
                value["question"]["typed_intent"] = ""
            elif kind == "train-boundary":
                value["split"] = "dev"
            elif kind == "strict-integer-bit":
                value["question"]["evidence"][0]["support"] = True
            else:
                raise AssertionError("未知 public field mutation")
        lines.append(canonical_json_line(value))
    return b"".join(lines)


def _non_target_question_malformed_payload() -> bytes:
    """损坏非目标 question，确认 selection 前不会语义投影它。"""
    lines = []
    changed = False
    for line in _SAMPLE.read_bytes().splitlines():
        value = parse_canonical_json_bytes(line, require_object=True)
        if not changed and value["episode_id"] != _EPISODE_ID:
            value["question"] = {
                "answer_plan": {"must_not_be_read": True},
                "evidence": "not-a-list",
                "typed_intent": False,
            }
            changed = True
        lines.append(canonical_json_line(value))
    assert changed
    return b"".join(lines)


def test_public_projection_matches_existing_visible_evidence_only() -> None:
    """同一公开 course 的白名单投影必须等于旧 reader 的 Evidence projection。"""
    legacy_episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.episode_id == _EPISODE_ID)
    legacy_input = public_response_act_planning_input_from_episode(
        legacy_episode)
    projection = project_public_grounded_answer_from_payload(
        _SAMPLE.read_bytes(), _EPISODE_ID)

    assert projection.question_surface == legacy_episode.question.question_surface
    assert projection.planning_input.canonical_record() == legacy_input.canonical_record()


def test_projection_never_constructs_full_grounded_answer_episode(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """V3 的公开 projection 不得回退到会解析 answer/surface 标签的旧 episode reader。"""
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("public projection 不得构造 GroundedAnswerEpisode")

    monkeypatch.setattr(GroundedAnswerEpisode, "from_dict", classmethod(forbidden))
    projection = project_public_grounded_answer_from_payload(
        _SAMPLE.read_bytes(), _EPISODE_ID)

    assert projection.planning_input.episode_id == _EPISODE_ID


def test_ignored_answer_and_surface_fields_do_not_define_public_projection() -> None:
    """禁入字段可变而 question/Evidence 不变时，公开 planning 必须逐记录一致。"""
    baseline = project_public_grounded_answer_from_payload(
        _SAMPLE.read_bytes(), _EPISODE_ID)
    changed = project_public_grounded_answer_from_payload(
        _ignored_fields_mutated_payload(), _EPISODE_ID)

    assert changed.question_surface == baseline.question_surface
    assert changed.planning_input.canonical_record() == (
        baseline.planning_input.canonical_record())


def test_projection_does_not_project_non_target_question_fields() -> None:
    """扫描 non-target 行只可读 identity/split，不能被其 question 结构阻断。"""
    baseline = project_public_grounded_answer_from_payload(
        _SAMPLE.read_bytes(), _EPISODE_ID)
    changed = project_public_grounded_answer_from_payload(
        _non_target_question_malformed_payload(), _EPISODE_ID)

    assert changed.question_surface == baseline.question_surface
    assert changed.planning_input.canonical_record() == (
        baseline.planning_input.canonical_record())


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("public-text", "typed_intent"),
        ("train-boundary", "target 不是 TRAIN"),
        ("strict-integer-bit", "support"),
    ),
    ids=("public-text", "train-boundary", "strict-integer-bit"),
)
def test_public_projection_rejects_public_field_drift(
        kind: str,
        message: str,
        ) -> None:
    """白名单字段损坏、split 漂移或 bool 伪整数必须 fail closed。"""
    changed = _public_field_mutated_payload(kind)

    with pytest.raises(PublicGroundedAnswerCourseProjectionError, match=message):
        project_public_grounded_answer_from_payload(changed, _EPISODE_ID)
