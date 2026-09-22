"""Narrow licensed JSONL loader for three-graph dialogue sessions.

V3 keeps every semantic surface as Unicode-scalar integers.  Older licensed
V1/V2 rows remain readable at this intake boundary, but are normalized to the
same integer value object immediately.  There is no token index, vocabulary,
segmentation, answer lookup, or source-body fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


DIALOGUE_SPEAKER_USER = 1
DIALOGUE_SPEAKER_ASSISTANT = 2
_SPEAKERS = frozenset({DIALOGUE_SPEAKER_USER, DIALOGUE_SPEAKER_ASSISTANT})
_FORMATS = frozenset({
    "PURE_INTEGER_AI_OASST1_DIALOGUE_COURSE_V2",
    "PURE_INTEGER_AI_OPENASSISTANT_DIALOGUE_COURSE_V2",
    "PURE_INTEGER_AI_KDCONV_DIALOGUE_COURSE_V1",
    "PURE_INTEGER_AI_LLM_ASSISTED_DIALOGUE_COURSE_V1",
})
STRUCTURAL_DIALOGUE_COURSE_V3 = (
    "PURE_INTEGER_AI_STRUCTURAL_DIALOGUE_COURSE_V3")
_FORMATS = frozenset({*_FORMATS, STRUCTURAL_DIALOGUE_COURSE_V3})
_SPLITS = frozenset({"train", "heldout", "negative"})
_FORBIDDEN_INDEX_FIELDS = frozenset({
    "token_index_file", "token_index_ordinal", "token_index_sha256",
    "aggregate_index_file", "aggregate_index_ordinal",
    "aggregate_index_sha256",
})


class StructuralDialogueCourseError(ValueError):
    """The explicit turn course is malformed or reaches a forbidden route."""


@dataclass(frozen=True, slots=True)
class StructuralDialogueTurn:
    turn_ordinal: int
    speaker_role: int
    message_id_values: tuple[int, ...]
    surface_values: tuple[int, ...]
    graph_object_keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if (type(self.turn_ordinal) is not int or self.turn_ordinal <= 0
                or type(self.speaker_role) is not int
                or self.speaker_role not in _SPEAKERS
                or type(self.message_id_values) is not tuple
                or not self.message_id_values
                or any(type(value) is not int or value < 0
                       for value in self.message_id_values)
                or type(self.surface_values) is not tuple
                or not self.surface_values
                or any(type(value) is not int
                       or not 0 <= value <= 0x10FFFF
                       or 0xD800 <= value <= 0xDFFF
                       for value in self.surface_values)):
            raise StructuralDialogueCourseError(
                "dialogue turn fields are incomplete")
        if (type(self.graph_object_keys) is not tuple
                or any(type(key) is not tuple or not key
                       or any(type(value) is not int or value < 0 for value in key)
                       for key in self.graph_object_keys)
                or len(set(self.graph_object_keys)) != len(self.graph_object_keys)):
            raise StructuralDialogueCourseError(
                "dialogue turn graph inputs are incomplete")

    @property
    def message_id(self) -> str:
        """Compatibility view for legacy bridge code; storage stays integer-only."""
        return "".join(map(chr, self.message_id_values))

    @property
    def surface(self) -> str:
        """Ephemeral bridge view; no string surface is retained in the course."""
        return "".join(map(chr, self.surface_values))


@dataclass(frozen=True, slots=True)
class StructuralDialogueCase:
    case_id_values: tuple[int, ...]
    split: str
    license_id: str
    dialogue_turns: tuple[StructuralDialogueTurn, ...]

    def __post_init__(self) -> None:
        turns = self.dialogue_turns
        if (type(self.case_id_values) is not tuple
                or not self.case_id_values
                or any(type(value) is not int or value < 0
                       for value in self.case_id_values)
                or self.split not in _SPLITS
                or type(self.license_id) is not str or not self.license_id
                or type(turns) is not tuple or len(turns) < 2
                or tuple(item.turn_ordinal for item in turns)
                != tuple(range(1, len(turns) + 1))
                or any(item.speaker_role != (
                    DIALOGUE_SPEAKER_USER if index % 2 else
                    DIALOGUE_SPEAKER_ASSISTANT)
                    for index, item in enumerate(turns, 1))):
            raise StructuralDialogueCourseError(
                "dialogue case order/license is incomplete")

    @property
    def case_id(self) -> str:
        """Compatibility view for existing evaluation indexes."""
        return "".join(map(chr, self.case_id_values))


@dataclass(frozen=True, slots=True)
class StructuralDialoguePack:
    cases: tuple[StructuralDialogueCase, ...]
    pack_sha256: str

    def __post_init__(self) -> None:
        if (not self.cases or type(self.pack_sha256) is not str
                or len(self.pack_sha256) != 64
                or any(value not in "0123456789abcdef"
                       for value in self.pack_sha256)):
            raise StructuralDialogueCourseError(
                "dialogue pack identity is incomplete")


def _surface_values(value: object, *, integer_format: bool) -> tuple[int, ...]:
    """Normalize one licensed boundary value without retaining source text."""
    if integer_format:
        if not isinstance(value, list):
            raise StructuralDialogueCourseError(
                "V3 dialogue turn surface_values must be a list")
        return tuple(value)
    if type(value) is not str or not value or value != value.strip():
        raise StructuralDialogueCourseError(
            "legacy dialogue turn surface is incomplete")
    return tuple(map(ord, value))


def _turn(value: object, *, integer_format: bool) -> StructuralDialogueTurn:
    if not isinstance(value, dict):
        raise StructuralDialogueCourseError("dialogue turn must be an object")
    if integer_format:
        if "surface" in value or "message_id" in value:
            raise StructuralDialogueCourseError(
                "V3 dialogue turn forbids string semantic fields")
        surface = value.get("surface_values")
        message_id = value.get("message_id_values")
        if not isinstance(message_id, list):
            raise StructuralDialogueCourseError(
                "V3 dialogue turn message_id_values must be a list")
        message_id = tuple(message_id)
    else:
        if "surface_values" in value or "message_id_values" in value:
            raise StructuralDialogueCourseError(
                "legacy dialogue turn cannot mix semantic encodings")
        surface = value.get("surface")
        legacy_message_id = value.get("message_id")
        if type(legacy_message_id) is not str or not legacy_message_id:
            raise StructuralDialogueCourseError(
                "legacy dialogue turn message_id is incomplete")
        message_id = tuple(map(ord, legacy_message_id))
    raw_graph_keys = value.get("graph_object_keys", ())
    if raw_graph_keys is None:
        raw_graph_keys = ()
    if raw_graph_keys == ():
        raw_graph_keys = []
    if not isinstance(raw_graph_keys, list):
        raise StructuralDialogueCourseError(
            "graph_object_keys must be an integer-array list")
    graph_keys = tuple(tuple(item) for item in raw_graph_keys)
    return StructuralDialogueTurn(
        value.get("turn_ordinal"),
        value.get("speaker_role"),
        message_id,
        _surface_values(surface, integer_format=integer_format),
        graph_keys,
    )


def _case(value: object) -> StructuralDialogueCase:
    if not isinstance(value, dict):
        raise StructuralDialogueCourseError("dialogue course row must be an object")
    course_format = value.get("format")
    if course_format not in _FORMATS:
        raise StructuralDialogueCourseError("dialogue course format is not registered")
    integer_format = course_format == STRUCTURAL_DIALOGUE_COURSE_V3
    if any(field in value for field in _FORBIDDEN_INDEX_FIELDS):
        raise StructuralDialogueCourseError(
            "dialogue session forbids token/aggregate index routes")
    raw_turns = value.get("dialogue_turns")
    if not isinstance(raw_turns, list):
        raise StructuralDialogueCourseError("dialogue_turns must be a list")
    turns = tuple(_turn(item, integer_format=integer_format)
                  for item in raw_turns)
    if len(turns) < 2:
        raise StructuralDialogueCourseError(
            "dialogue case requires at least two turns")
    path_count = value.get("path_turn_count")
    path_ids = value.get(
        "path_message_id_values" if integer_format else "path_message_ids")
    if integer_format:
        if ("path_message_ids" in value
                or not isinstance(path_ids, list)
                or any(not isinstance(item, list) for item in path_ids)):
            raise StructuralDialogueCourseError(
                "V3 path identities must be integer arrays")
        path_ids = tuple(tuple(item) for item in path_ids)
    elif isinstance(path_ids, list):
        path_ids = tuple(
            tuple(map(ord, item)) if isinstance(item, str) else item
            for item in path_ids
        )
    response_surface = (
        value.get("response_surface_values") if integer_format
        else value.get("response_surface")
    )
    expected_response = (
        list(turns[-1].surface_values) if integer_format
        else "".join(map(chr, turns[-1].surface_values))
    )
    forbidden_response_field = (
        "response_surface" if integer_format else "response_surface_values")
    if (forbidden_response_field in value
            or type(path_count) is not int or path_count != len(turns)
            or type(path_ids) is not tuple
            or tuple(path_ids) != tuple(
                item.message_id_values for item in turns)
            or response_surface != expected_response
            or value.get("response_turn_ordinal") != turns[-1].turn_ordinal
            or value.get("prompt_turn_ordinal") != turns[-2].turn_ordinal):
        raise StructuralDialogueCourseError(
            "dialogue path commitment differs from explicit turns")
    if integer_format:
        if any(field in value for field in ("sample_id", "item_id")):
            raise StructuralDialogueCourseError(
                "V3 dialogue case forbids string identity")
        case_id = value.get("sample_id_values")
        if not isinstance(case_id, list) or not case_id:
            case_id = value.get("item_id_values")
        if not isinstance(case_id, list):
            raise StructuralDialogueCourseError(
                "V3 dialogue case requires integer identity")
        case_id = tuple(case_id)
    else:
        case_id = value.get("sample_id")
        if not isinstance(case_id, str) or not case_id:
            case_id = value.get("item_id")
        if not isinstance(case_id, str) or not case_id:
            raise StructuralDialogueCourseError(
                "legacy dialogue case identity is incomplete")
        case_id = tuple(map(ord, case_id))
    return StructuralDialogueCase(
        case_id,
        value.get("split"),
        value.get("license_id"),
        turns,
    )


def load_structural_dialogue_course(
        paths: Iterable[str | Path]) -> StructuralDialoguePack:
    """Load explicit turns without importing legacy token-index facilities."""
    resolved = tuple(Path(item).resolve(strict=True) for item in paths)
    if not resolved:
        raise StructuralDialogueCourseError("dialogue course path is empty")
    digest = hashlib.sha256()
    cases = []
    seen = set()
    for path in resolved:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise StructuralDialogueCourseError(
                "dialogue course must be UTF-8 JSONL") from exc
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                case = _case(json.loads(line))
            except json.JSONDecodeError as exc:
                raise StructuralDialogueCourseError(
                    f"invalid JSONL at {path.name}:{line_number}") from exc
            if case.case_id_values in seen:
                raise StructuralDialogueCourseError(
                    "dialogue case identity is duplicated")
            seen.add(case.case_id_values)
            cases.append(case)
    return StructuralDialoguePack(tuple(cases), digest.hexdigest())


__all__ = [
    "DIALOGUE_SPEAKER_ASSISTANT",
    "DIALOGUE_SPEAKER_USER",
    "STRUCTURAL_DIALOGUE_COURSE_V3",
    "StructuralDialogueCase",
    "StructuralDialogueCourseError",
    "StructuralDialoguePack",
    "StructuralDialogueTurn",
    "load_structural_dialogue_course",
]
