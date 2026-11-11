"""Explicit four-way containers for integer graph-course admission.

This training/admission container carries the bundle owner, source/license
identity, frozen cases and the complete records referenced by those cases.
It does not compile evaluation labels or qualify their owner/provenance.
The existing train-only compiler and live runner remain separate.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pure_integer_ai.experiments.integer_graph_novelty import (
    IntegerGraphNoveltyCase,
    validate_integer_graph_novelty_splits,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    IntegerRelationGraphCourseError,
    IntegerRelationGraphRecord,
    # Reuse the frozen record codec without changing the live compiler.
    # A public codec can replace these imports after the active run ends.
    _record_dict,
    _record_from_dict,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)


INTEGER_RELATION_GRAPH_SPLIT_COURSE_FORMAT = (
    "PURE_INTEGER_RELATION_GRAPH_SPLIT_COURSE_V1")
SPLITS = ("train", "held_out", "negative", "conflict")
_SPLIT_CODES = {name: index for index, name in enumerate(SPLITS, 1)}
_TOP_LEVEL_FIELDS = frozenset({
    "format", "split", "split_code", "owner_identity", "source_identity",
    "license_identity", "case_count", "record_count", "cases", "records",
    "free_dialogue_complete",
})
_CASE_FIELDS = frozenset({
    "case_key", "input_values", "dialogue_chain_keys", "record_keys",
})


class IntegerRelationGraphSplitCourseError(ValueError):
    """Split-course metadata or record ownership is not closed."""


def _key(value: Any, *, where: str, positive: bool = False) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int or item < 0
                   or (positive and item <= 0) for item in value)):
        raise IntegerRelationGraphSplitCourseError(
            f"{where} must be a nonempty integer key")
    return tuple(value)


def _unicode_values(value: Any, *, where: str) -> tuple[int, ...]:
    result = _key(value, where=where)
    if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in result):
        raise IntegerRelationGraphSplitCourseError(
            f"{where} must contain Unicode scalar values")
    return result


@dataclass(frozen=True, slots=True)
class IntegerRelationGraphSplitCase:
    """One frozen input case and the records that justify it."""

    case_key: tuple[int, ...]
    input_values: tuple[int, ...]
    dialogue_chain_keys: tuple[tuple[int, ...], ...]
    record_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_key", _key(self.case_key, where="case_key"))
        object.__setattr__(
            self, "input_values", _unicode_values(
                self.input_values, where="input_values"))
        if (not isinstance(self.dialogue_chain_keys, (list, tuple))
                or not self.dialogue_chain_keys):
            raise IntegerRelationGraphSplitCourseError(
                "dialogue_chain_keys must not be empty")
        chains = tuple(_key(item, where="dialogue_chain_key")
                       for item in self.dialogue_chain_keys)
        if len(set(chains)) != len(chains):
            raise IntegerRelationGraphSplitCourseError(
                "dialogue_chain_keys must be unique")
        object.__setattr__(self, "dialogue_chain_keys", chains)
        if (not isinstance(self.record_keys, (list, tuple))
                or not self.record_keys):
            raise IntegerRelationGraphSplitCourseError(
                "record_keys must not be empty")
        records = tuple(_key(item, where="case.record_key", positive=True)
                        for item in self.record_keys)
        if len(set(records)) != len(records):
            raise IntegerRelationGraphSplitCourseError(
                "case record_keys must be unique")
        object.__setattr__(self, "record_keys", records)

    def to_novelty_case(
            self, records: Mapping[tuple[int, ...], IntegerRelationGraphRecord],
            ) -> IntegerGraphNoveltyCase:
        """Resolve only the explicitly owned records for novelty admission."""
        try:
            resolved = tuple(records[key] for key in self.record_keys)
        except KeyError as error:
            raise IntegerRelationGraphSplitCourseError(
                f"case record is missing: {error.args[0]}") from error
        return IntegerGraphNoveltyCase(
            self.case_key, self.input_values, resolved, self.dialogue_chain_keys)


@dataclass(frozen=True, slots=True)
class IntegerRelationGraphSplitCourse:
    """A single split with closed case-to-record ownership."""

    split: str
    owner_identity: tuple[int, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    cases: tuple[IntegerRelationGraphSplitCase, ...]
    records: tuple[IntegerRelationGraphRecord, ...]

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise IntegerRelationGraphSplitCourseError(
                f"unknown split: {self.split!r}")
        object.__setattr__(
            self, "owner_identity", _key(
                self.owner_identity, where="owner_identity", positive=True))
        object.__setattr__(
            self, "source_identity", _key(
                self.source_identity, where="source_identity", positive=True))
        object.__setattr__(
            self, "license_identity", _key(
                self.license_identity, where="license_identity", positive=True))
        if not isinstance(self.cases, (tuple, list)) or not self.cases:
            raise IntegerRelationGraphSplitCourseError("split has no cases")
        cases = tuple(self.cases)
        if any(not isinstance(item, IntegerRelationGraphSplitCase)
               for item in cases):
            raise IntegerRelationGraphSplitCourseError("case type is invalid")
        if len({item.case_key for item in cases}) != len(cases):
            raise IntegerRelationGraphSplitCourseError("case identities are duplicated")
        if tuple(sorted(cases, key=lambda item: item.case_key)) != cases:
            raise IntegerRelationGraphSplitCourseError("cases must be canonically sorted")
        object.__setattr__(self, "cases", cases)
        if not isinstance(self.records, (tuple, list)) or not self.records:
            raise IntegerRelationGraphSplitCourseError("split has no records")
        records = tuple(self.records)
        if any(not isinstance(item, IntegerRelationGraphRecord)
               for item in records):
            raise IntegerRelationGraphSplitCourseError("record type is invalid")
        if len({item.observation_key for item in records}) != len(records):
            raise IntegerRelationGraphSplitCourseError(
                "record observation identities are duplicated")
        if tuple(sorted(records, key=lambda item: item.stable_key())) != records:
            raise IntegerRelationGraphSplitCourseError(
                "records must be canonically sorted")
        object.__setattr__(self, "records", records)
        by_observation = {item.observation_key: item for item in records}
        referenced = [key for case in cases for key in case.record_keys]
        if len(set(referenced)) != len(referenced):
            raise IntegerRelationGraphSplitCourseError(
                "a record may belong to only one case")
        if set(referenced) != set(by_observation):
            raise IntegerRelationGraphSplitCourseError(
                "cases must reference every record exactly once")

    @property
    def split_code(self) -> int:
        return _SPLIT_CODES[self.split]

    def novelty_cases(self) -> tuple[IntegerGraphNoveltyCase, ...]:
        by_observation = {item.observation_key: item for item in self.records}
        return tuple(case.to_novelty_case(by_observation) for case in self.cases)


def _case_dict(case: IntegerRelationGraphSplitCase) -> dict[str, object]:
    return {
        "case_key": list(case.case_key),
        "input_values": list(case.input_values),
        "dialogue_chain_keys": [list(item) for item in case.dialogue_chain_keys],
        "record_keys": [list(item) for item in case.record_keys],
    }


def _case_from_dict(value: object) -> IntegerRelationGraphSplitCase:
    if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
        raise IntegerRelationGraphSplitCourseError("case fields drifted")
    try:
        return IntegerRelationGraphSplitCase(
            _key(value["case_key"], where="case_key"),
            _unicode_values(value["input_values"], where="input_values"),
            tuple(tuple(item) for item in value["dialogue_chain_keys"]),
            tuple(tuple(item) for item in value["record_keys"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, IntegerRelationGraphSplitCourseError):
            raise
        raise IntegerRelationGraphSplitCourseError("case fields are invalid") from error


def write_integer_relation_graph_split_course(
        path: str | Path,
        course: IntegerRelationGraphSplitCourse,
        ) -> str:
    """Write one split with a deterministic, text-free integer envelope."""
    if not isinstance(course, IntegerRelationGraphSplitCourse):
        raise TypeError("course type is invalid")
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    payload = {
        "format": INTEGER_RELATION_GRAPH_SPLIT_COURSE_FORMAT,
        "split": course.split,
        "split_code": course.split_code,
        "owner_identity": list(course.owner_identity),
        "source_identity": list(course.source_identity),
        "license_identity": list(course.license_identity),
        "case_count": len(course.cases),
        "record_count": len(course.records),
        "cases": [_case_dict(item) for item in course.cases],
        "records": [_record_dict(item) for item in course.records],
        "free_dialogue_complete": 0,
    }
    raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")) + "\n").encode("ascii")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def read_integer_relation_graph_split_course(
        path: str | Path,
        ) -> IntegerRelationGraphSplitCourse:
    """Cold-read one split; all case references must resolve exactly."""
    raw = Path(path).resolve().read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise IntegerRelationGraphSplitCourseError(
            "split course must end with one newline")
    try:
        payload = parse_canonical_json_bytes(raw[:-1], require_object=True)
    except ValueError as error:
        raise IntegerRelationGraphSplitCourseError(
            "split course JSON is not canonical") from error
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise IntegerRelationGraphSplitCourseError("split course fields drifted")
    if payload.get("format") != INTEGER_RELATION_GRAPH_SPLIT_COURSE_FORMAT:
        raise IntegerRelationGraphSplitCourseError("split course format drifted")
    split = payload.get("split")
    if (split not in SPLITS or type(payload.get("split_code")) is not int
            or payload["split_code"] != _SPLIT_CODES[split]):
        raise IntegerRelationGraphSplitCourseError("split code drifted")
    if (type(payload.get("free_dialogue_complete")) is not int
            or payload["free_dialogue_complete"] != 0):
        raise IntegerRelationGraphSplitCourseError(
            "free_dialogue_complete must remain zero")
    try:
        cases_raw = payload["cases"]
        records_raw = payload["records"]
        if not isinstance(cases_raw, list) or not isinstance(records_raw, list):
            raise IntegerRelationGraphSplitCourseError(
                "cases and records must be lists")
        cases = tuple(_case_from_dict(item) for item in cases_raw)
        records = tuple(_record_from_dict(item) for item in records_raw)
        course = IntegerRelationGraphSplitCourse(
            split,
            _key(payload["owner_identity"], where="owner_identity", positive=True),
            _key(payload["source_identity"], where="source_identity", positive=True),
            _key(payload["license_identity"], where="license_identity", positive=True),
            cases,
            records,
        )
    except (KeyError, TypeError, ValueError, IntegerRelationGraphCourseError) as error:
        if isinstance(error, IntegerRelationGraphSplitCourseError):
            raise
        raise IntegerRelationGraphSplitCourseError(
            "split course records are invalid") from error
    if (type(payload.get("case_count")) is not int
            or payload["case_count"] != len(course.cases)):
        raise IntegerRelationGraphSplitCourseError("case count drifted")
    if (type(payload.get("record_count")) is not int
            or payload["record_count"] != len(course.records)):
        raise IntegerRelationGraphSplitCourseError("record count drifted")
    return course


def build_integer_graph_novelty_splits(
        courses: Mapping[str, IntegerRelationGraphSplitCourse],
        ) -> dict[str, tuple[IntegerGraphNoveltyCase, ...]]:
    """Join four cold-read courses only when owner/source/license agree."""
    if not isinstance(courses, Mapping) or set(courses) != set(SPLITS):
        raise IntegerRelationGraphSplitCourseError(
            "all four split courses are required")
    values = tuple(courses[name] for name in SPLITS)
    if any(not isinstance(item, IntegerRelationGraphSplitCourse)
           for item in values):
        raise IntegerRelationGraphSplitCourseError("split course type is invalid")
    if any(item.split != name for name, item in zip(SPLITS, values, strict=True)):
        raise IntegerRelationGraphSplitCourseError("split course name drifted")
    identity_fields = ("owner_identity", "source_identity", "license_identity")
    for field in identity_fields:
        if len({getattr(item, field) for item in values}) != 1:
            raise IntegerRelationGraphSplitCourseError(
                f"split {field} drifted")
    case_keys = [case.case_key for item in values for case in item.cases]
    if len(set(case_keys)) != len(case_keys):
        raise IntegerRelationGraphSplitCourseError("case identities cross split")
    return {name: courses[name].novelty_cases() for name in SPLITS}


def validate_integer_relation_graph_split_courses(
        courses: Mapping[str, IntegerRelationGraphSplitCourse], *,
        minimum_novel_held_out: int = 20,
        minimum_relation_families: int = 5,
        ) -> dict[str, object]:
    """Run the real topology admission after the split contract closes."""
    splits = build_integer_graph_novelty_splits(courses)
    report = validate_integer_graph_novelty_splits(
        splits,
        minimum_novel_held_out=minimum_novel_held_out,
        minimum_relation_families=minimum_relation_families,
    )
    report["split_course_contract"] = "PASS"
    report["owner_identity"] = list(next(iter(courses.values())).owner_identity)
    report["source_identity"] = list(next(iter(courses.values())).source_identity)
    report["license_identity"] = list(next(iter(courses.values())).license_identity)
    report["label_owner_qualification"] = "UNKNOWN_requires_independent_label_owner_audit"
    report["unauthorized_qualification"] = "UNKNOWN_not_measured_by_four_split_container"
    return report


__all__ = [
    "INTEGER_RELATION_GRAPH_SPLIT_COURSE_FORMAT",
    "IntegerRelationGraphSplitCase",
    "IntegerRelationGraphSplitCourse",
    "IntegerRelationGraphSplitCourseError",
    "SPLITS",
    "build_integer_graph_novelty_splits",
    "read_integer_relation_graph_split_course",
    "validate_integer_relation_graph_split_courses",
    "write_integer_relation_graph_split_course",
]
