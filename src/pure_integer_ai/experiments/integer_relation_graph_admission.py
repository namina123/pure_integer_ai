"""Join one supervised train course with label-free evaluation graph courses."""
from __future__ import annotations

from collections.abc import Mapping

from pure_integer_ai.experiments.integer_graph_novelty import (
    IntegerGraphNoveltyCase,
    validate_integer_graph_novelty_splits,
)
from pure_integer_ai.experiments.integer_relation_evaluation_graph import (
    EVALUATION_SPLITS,
    IntegerRelationEvaluationGraphRecord,
    IntegerRelationEvaluationGraphSplitCourse,
)
from pure_integer_ai.experiments.integer_relation_graph_split_course import (
    IntegerRelationGraphSplitCourse,
    IntegerRelationGraphSplitCourseError,
    SPLITS,
)


class IntegerRelationGraphAdmissionError(ValueError):
    """Train/evaluation ownership is not closed or labels cross the boundary."""


def build_integer_relation_graph_admission_splits(
        train_course: IntegerRelationGraphSplitCourse,
        evaluation_courses: Mapping[
            str, IntegerRelationEvaluationGraphSplitCourse],
        ) -> dict[str, tuple[IntegerGraphNoveltyCase, ...]]:
    """Build novelty inputs without accepting supervised records for eval.

    The train codec intentionally retains its teacher labels. Evaluation must
    use the separate label-free record and course types; the result only feeds
    structural novelty validation, never a learner.
    """
    if (not isinstance(train_course, IntegerRelationGraphSplitCourse)
            or train_course.split != "train"):
        raise IntegerRelationGraphAdmissionError(
            "exactly one supervised train course is required")
    if (not isinstance(evaluation_courses, Mapping)
            or set(evaluation_courses) != set(EVALUATION_SPLITS)):
        raise IntegerRelationGraphAdmissionError(
            "held_out, negative, and conflict evaluation courses are required")

    identity = (
        train_course.owner_identity,
        train_course.source_identity,
        train_course.license_identity,
    )
    case_keys = {case.case_key for case in train_course.cases}
    result: dict[str, tuple[IntegerGraphNoveltyCase, ...]] = {
        "train": train_course.novelty_cases(),
    }
    for split in EVALUATION_SPLITS:
        course = evaluation_courses[split]
        if (not isinstance(course, IntegerRelationEvaluationGraphSplitCourse)
                or course.split != split):
            raise IntegerRelationGraphAdmissionError(
                f"{split} must use the label-free evaluation course type")
        if identity != (
                course.owner_identity,
                course.source_identity,
                course.license_identity):
            raise IntegerRelationGraphAdmissionError(
                f"{split} owner/source/license identity drifted")

        cases = []
        for case in course.cases:
            assignment = case.assignment
            if assignment.split != split or assignment.case_key in case_keys:
                raise IntegerRelationGraphAdmissionError(
                    f"{split} case identity or split drifted")
            case_keys.add(assignment.case_key)
            if not case.records:
                raise IntegerRelationGraphAdmissionError(
                    f"{split} contains no graph-accepted case for topology audit")
            if any(type(row) is not IntegerRelationEvaluationGraphRecord
                   for row in case.records):
                raise IntegerRelationGraphAdmissionError(
                    f"{split} contains a non-label-free evaluation record")
            cases.append(case.to_novelty_case())
        if not cases:
            raise IntegerRelationGraphAdmissionError(
                f"{split} contains no evaluation cases")
        result[split] = tuple(cases)

    if set(result) != set(SPLITS):
        raise IntegerRelationGraphAdmissionError("admission split set drifted")
    return result


def validate_integer_relation_graph_admission(
        train_course: IntegerRelationGraphSplitCourse,
        evaluation_courses: Mapping[
            str, IntegerRelationEvaluationGraphSplitCourse],
        *,
        minimum_novel_held_out: int = 20,
        minimum_relation_families: int = 5,
        ) -> dict[str, object]:
    """Validate topology across isolated course types; do not qualify behavior."""
    try:
        splits = build_integer_relation_graph_admission_splits(
            train_course, evaluation_courses)
    except IntegerRelationGraphAdmissionError:
        raise
    except (IntegerRelationGraphSplitCourseError, TypeError, ValueError) as error:
        raise IntegerRelationGraphAdmissionError(
            "admission split resolution failed") from error

    report = validate_integer_graph_novelty_splits(
        splits,
        minimum_novel_held_out=minimum_novel_held_out,
        minimum_relation_families=minimum_relation_families,
    )
    report.update({
        "split_course_contract": "PASS_TRAIN_AND_LABEL_FREE_EVALUATION",
        "owner_identity": list(train_course.owner_identity),
        "source_identity": list(train_course.source_identity),
        "license_identity": list(train_course.license_identity),
        "evaluation_record_contract": "PASS_LABEL_FREE_TYPED_RECORDS",
        "source_license_qualification": "UNKNOWN_requires_source_manifest_audit",
        "upstream_cluster_qualification": "UNKNOWN_requires_source_cluster_audit",
        "label_owner_qualification": "UNKNOWN_requires_independent_label_owner_audit",
        "unauthorized_qualification": "UNKNOWN_not_measured_by_graph_admission",
        "free_dialogue_complete": 0,
    })
    return report


__all__ = [
    "IntegerRelationGraphAdmissionError",
    "build_integer_relation_graph_admission_splits",
    "validate_integer_relation_graph_admission",
]
