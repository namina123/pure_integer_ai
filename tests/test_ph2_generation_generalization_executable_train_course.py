"""E-01 六路 TRAIN executable case catalog 聚焦合同。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_course import (
    GenerationGeneralizationExecutableTrainCourseError,
    read_generation_generalization_executable_train_course,
)


CASE_PATH = Path(
    "data/ph2/generation_generalization_executable_train_case_v1.jsonl.sample")
GROUNDED_PATH = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def test_executable_train_course_reuses_six_source_bound_cases_without_answers():
    """六路 case 精确引用 CC0 TRAIN episode，不复制 surface/label。"""
    course = read_generation_generalization_executable_train_course(
        CASE_PATH, GROUNDED_PATH)

    assert tuple(item.requirement for item in course.cases) == (
        INDEPENDENT_VERIFIER_REQUIREMENTS)
    assert len(course.cases) == 6
    assert len(course.source_episodes) == 4
    assert course.stable_key() == course.stable_key()
    assert all(course.episode_for(item).split == "train"
               for item in course.cases)
    assert course.case_for_requirement(
        "ADDRESSEE_RECOVERABILITY").reference_strategy == (
            "ANTECEDENT_REFERENCE")
    assert course.case_for_requirement(
        "STRUCTURE_SLOT_ORDER").reference_strategy == (
            "EXPLICIT_REPETITION")
    raw = CASE_PATH.read_bytes().lower()
    assert b'"surface"' not in raw
    assert b'"accepted' not in raw
    assert b'"rejected' not in raw
    assert b'"expected' not in raw

    with pytest.raises(
            GenerationGeneralizationExecutableTrainCourseError,
            match="精确覆盖六路"):
        replace(course, cases=course.cases[:-1])

    task = course.case_for_requirement("COMMUNICATIVE_TASK")
    with pytest.raises(
            GenerationGeneralizationExecutableTrainCourseError,
            match="requirement/runtime/strategy/act"):
        replace(task, response_act="ANSWER")
