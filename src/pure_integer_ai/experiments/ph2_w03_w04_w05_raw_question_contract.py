"""原始问题构造、来源绑定实例与 FT09 结果的不可变合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer_contract import (
    W03W04W05QuestionAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalResult,
)


RAW_QUESTION_SEGMENT_KINDS = {
    "BOUNDARY", "PREDICATE", "ROLE_FILLER", "VARIABLE"}
RAW_QUESTION_STATUSES = {"ANSWER", "CLARIFY", "UNKNOWN"}


# object-model: exception
class W03W04W05RawQuestionError(ValueError):
    """问题构造、匹配实例或原始问题结果发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(
        value: object,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> str:
    if (not isinstance(value, str) or value.strip() != value
            or (not allow_empty and not value)):
        raise W03W04W05RawQuestionError(
            f"{where} is not canonical text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05RawQuestionError(
            f"{where} is not a strict integer key")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W03W04W05RawQuestionError(f"{where} is not SHA-256")
    return value


def _selected_candidate(vertical: W03W04W05VerticalResult):
    if vertical.status != "BRIDGED" or vertical.link is None:
        raise W03W04W05RawQuestionError(
            "question construction requires a bridged vertical result")
    w05 = vertical.w04_w05.w05_result
    candidates = tuple(
        item for item in w05.candidates
        if item.proposition_key == vertical.link.proposition_key)
    if (w05.status != "UNIQUE" or len(candidates) != 1
            or candidates[0].active != 1
            or candidates[0].lifecycle_status != "ACTIVE"
            or candidates[0].reasoning_status != "AUTHORIZED"):
        raise W03W04W05RawQuestionError(
            "question construction lacks one authorized Proposition")
    return candidates[0]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionPatternSegment:
    """一个数据学习所得的变量、谓词、角色填充或边界槽。"""

    kind: str
    role_ordinal: int | None
    literal_surface: str | None

    def __post_init__(self) -> None:
        if self.kind not in RAW_QUESTION_SEGMENT_KINDS:
            raise W03W04W05RawQuestionError(
                "raw question segment kind drifted")
        if self.kind in {"VARIABLE", "ROLE_FILLER"}:
            if (type(self.role_ordinal) is not int
                    or self.role_ordinal < 0):
                raise W03W04W05RawQuestionError(
                    "raw question role ordinal drifted")
        elif self.role_ordinal is not None:
            raise W03W04W05RawQuestionError(
                "non-role question segment published a role ordinal")
        if self.kind in {"VARIABLE", "BOUNDARY"}:
            _text(self.literal_surface, where="question literal segment")
        elif self.literal_surface is not None:
            raise W03W04W05RawQuestionError(
                "dynamic question segment published a literal surface")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "literal_surface": self.literal_surface,
            "role_ordinal": self.role_ordinal,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionPattern:
    """由公开样本和一个已学命题验证的问题构造模式。"""

    construction_id: str
    source_key: str
    license_id: str
    sample_sha256: str
    logical_order: int
    language: str
    primitive_registry: str
    primitive_kind: int
    segments: tuple[RawQuestionPatternSegment, ...]
    exemplar_vertical_sha256: str

    def __post_init__(self) -> None:
        for name in (
                "construction_id", "source_key", "license_id", "language",
                "primitive_registry"):
            _text(getattr(self, name), where=f"question pattern {name}")
        _sha256(self.sample_sha256, where="question pattern sample")
        _sha256(
            self.exemplar_vertical_sha256,
            where="question pattern exemplar",
        )
        if type(self.logical_order) is not int or self.logical_order <= 0:
            raise W03W04W05RawQuestionError(
                "question pattern logical order drifted")
        if type(self.primitive_kind) is not int or self.primitive_kind <= 0:
            raise W03W04W05RawQuestionError(
                "question pattern primitive kind drifted")
        if (not isinstance(self.segments, tuple) or not self.segments
                or any(not isinstance(item, RawQuestionPatternSegment)
                       for item in self.segments)):
            raise W03W04W05RawQuestionError(
                "question pattern segments drifted")
        kinds = tuple(item.kind for item in self.segments)
        if (kinds.count("VARIABLE") != 1
                or kinds.count("PREDICATE") != 1
                or kinds.count("BOUNDARY") != 1
                or kinds.count("ROLE_FILLER") < 1):
            raise W03W04W05RawQuestionError(
                "question pattern lacks required structural segments")
        roles = tuple(
            item.role_ordinal for item in self.segments
            if item.role_ordinal is not None)
        if (len(set(roles)) != len(roles)
                or set(roles) != set(range(len(roles)))):
            raise W03W04W05RawQuestionError(
                "question pattern role ordinals are not closed")

    def to_dict(self) -> dict[str, object]:
        return {
            "construction_id": self.construction_id,
            "exemplar_vertical_sha256": self.exemplar_vertical_sha256,
            "language": self.language,
            "license_id": self.license_id,
            "logical_order": self.logical_order,
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "sample_sha256": self.sample_sha256,
            "segments": [item.to_dict() for item in self.segments],
            "source_key": self.source_key,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionConstructionSegment:
    """模式应用到一个已学 Proposition 后的来源化问题片段。"""

    kind: str
    surface: str
    role_key: tuple[int, ...] | None
    occurrence_key: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if self.kind not in RAW_QUESTION_SEGMENT_KINDS:
            raise W03W04W05RawQuestionError(
                "question construction segment kind drifted")
        _text(self.surface, where="question construction surface")
        if self.kind == "VARIABLE":
            if self.role_key is None or self.occurrence_key is not None:
                raise W03W04W05RawQuestionError(
                    "question variable segment identity drifted")
        elif self.kind == "ROLE_FILLER":
            if self.role_key is None or self.occurrence_key is None:
                raise W03W04W05RawQuestionError(
                    "question role filler identity drifted")
        elif self.kind == "PREDICATE":
            if self.role_key is not None or self.occurrence_key is None:
                raise W03W04W05RawQuestionError(
                    "question predicate identity drifted")
        elif self.role_key is not None or self.occurrence_key is not None:
            raise W03W04W05RawQuestionError(
                "question boundary published semantic identity")
        if self.role_key is not None:
            _strict_key(self.role_key, where="question construction role")
        if self.occurrence_key is not None:
            _strict_key(
                self.occurrence_key,
                where="question construction occurrence",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "occurrence_key": (
                None if self.occurrence_key is None
                else list(self.occurrence_key)
            ),
            "role_key": (
                None if self.role_key is None else list(self.role_key)),
            "surface": self.surface,
        }


def _validate_construction(
        value: "RawQuestionConstruction",
        ) -> None:
    vertical = value.vertical_result
    if vertical.link is None:
        raise W03W04W05RawQuestionError(
            "question construction lacks a vertical link")
    if ((vertical.link.primitive_registry, vertical.link.primitive_kind)
            != (value.pattern.primitive_registry, value.pattern.primitive_kind)):
        raise W03W04W05RawQuestionError(
            "question pattern primitive does not match learned knowledge")
    candidate = _selected_candidate(vertical)
    bindings = candidate.role_bindings
    role_ordinals = tuple(
        item.role_ordinal for item in value.pattern.segments
        if item.role_ordinal is not None)
    if len(bindings) != len(role_ordinals):
        raise W03W04W05RawQuestionError(
            "question pattern role inventory does not match Proposition")
    target_role = None
    for pattern_segment, segment in zip(value.pattern.segments, value.segments):
        if pattern_segment.kind != segment.kind:
            raise W03W04W05RawQuestionError(
                "question construction segment order drifted")
        if pattern_segment.kind in {"VARIABLE", "BOUNDARY"}:
            if segment.surface != pattern_segment.literal_surface:
                raise W03W04W05RawQuestionError(
                    "question literal segment escaped its learned pattern")
        if pattern_segment.kind == "VARIABLE":
            binding = bindings[pattern_segment.role_ordinal]
            if segment.role_key != binding.role_key:
                raise W03W04W05RawQuestionError(
                    "question variable target role drifted")
            target_role = binding.role_key
        elif pattern_segment.kind == "PREDICATE":
            occurrences = tuple(
                item for item in candidate.occurrences
                if item.identity_key == vertical.link.predicate_occurrence_key)
            if (len(occurrences) != 1
                    or segment.occurrence_key != occurrences[0].identity_key
                    or segment.surface != occurrences[0].surface_fragment):
                raise W03W04W05RawQuestionError(
                    "question predicate segment escaped learned occurrence")
        elif pattern_segment.kind == "ROLE_FILLER":
            binding = bindings[pattern_segment.role_ordinal]
            occurrences = tuple(
                item for item in candidate.occurrences
                if item.semantic_object_key == binding.filler_key)
            if (len(occurrences) != 1
                    or segment.role_key != binding.role_key
                    or segment.occurrence_key != occurrences[0].identity_key
                    or segment.surface != occurrences[0].surface_fragment):
                raise W03W04W05RawQuestionError(
                    "question role filler escaped learned RoleBinding")
    if target_role is None or value.target_role_key != target_role:
        raise W03W04W05RawQuestionError(
            "question construction target role is absent")
    if value.question_surface != "".join(
            item.surface for item in value.segments):
        raise W03W04W05RawQuestionError(
            "question construction surface is not segment-derived")
    target_binding = tuple(
        item for item in bindings if item.role_key == value.target_role_key)
    if len(target_binding) != 1:
        raise W03W04W05RawQuestionError(
            "question target RoleBinding is not unique")
    target_occurrences = tuple(
        item for item in candidate.occurrences
        if item.semantic_object_key == target_binding[0].filler_key)
    if (len(target_occurrences) != 1
            or target_occurrences[0].surface_fragment in value.question_surface):
        raise W03W04W05RawQuestionError(
            "question construction leaked the target filler surface")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionConstruction:
    """一个问题模式在来源绑定已学 Proposition 上的可查询实例。"""

    pattern: RawQuestionPattern
    vertical_result: W03W04W05VerticalResult
    segments: tuple[RawQuestionConstructionSegment, ...]
    question_surface: str
    target_role_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.pattern, RawQuestionPattern)
                or not isinstance(self.vertical_result, W03W04W05VerticalResult)
                or not isinstance(self.segments, tuple)
                or len(self.segments) != len(self.pattern.segments)
                or any(not isinstance(item, RawQuestionConstructionSegment)
                       for item in self.segments)):
            raise W03W04W05RawQuestionError(
                "raw question construction projection drifted")
        _text(self.question_surface, where="constructed question surface")
        _strict_key(self.target_role_key, where="constructed target role")
        _validate_construction(self)

    @property
    def source_record_key(self) -> tuple[int, ...]:
        if self.vertical_result.link is None:
            raise RuntimeError("question construction link is unavailable")
        return self.vertical_result.link.source_ref_key

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern": self.pattern.to_dict(),
            "question_surface": self.question_surface,
            "segments": [item.to_dict() for item in self.segments],
            "target_role_key": list(self.target_role_key),
            "vertical_result": self.vertical_result.to_dict(),
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionRequest:
    """只含原始问题表层与可选知识来源域的外部请求。"""

    question_surface: str
    source_record_key: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _text(self.question_surface, where="raw question surface")
        if self.source_record_key is not None:
            _strict_key(
                self.source_record_key,
                where="raw question source record",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "question_surface": self.question_surface,
            "source_record_key": (
                None if self.source_record_key is None
                else list(self.source_record_key)
            ),
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionAnswerResult:
    """原始问题构造匹配以及可选 FT09 执行结果。"""

    request: RawQuestionRequest
    status: str
    answer_surface: str | None
    matched_construction_sha256s: tuple[str, ...]
    selected_construction: RawQuestionConstruction | None
    typed_result: W03W04W05QuestionAnswerResult | None
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.request, RawQuestionRequest)
                or self.status not in RAW_QUESTION_STATUSES
                or not isinstance(self.matched_construction_sha256s, tuple)):
            raise W03W04W05RawQuestionError(
                "raw question answer result drifted")
        for item in self.matched_construction_sha256s:
            _sha256(item, where="matched question construction")
        if (tuple(sorted(self.matched_construction_sha256s))
                != self.matched_construction_sha256s
                or len(set(self.matched_construction_sha256s))
                != len(self.matched_construction_sha256s)):
            raise W03W04W05RawQuestionError(
                "matched question constructions are not canonical")
        count = len(self.matched_construction_sha256s)
        if count == 0:
            if (self.status != "UNKNOWN" or self.answer_surface is not None
                    or self.selected_construction is not None
                    or self.typed_result is not None):
                raise W03W04W05RawQuestionError(
                    "unmatched raw question published a result")
        elif count > 1:
            if (self.status != "CLARIFY" or self.answer_surface is not None
                    or self.selected_construction is not None
                    or self.typed_result is not None):
                raise W03W04W05RawQuestionError(
                    "ambiguous raw question selected a construction")
        else:
            if (not isinstance(
                    self.selected_construction, RawQuestionConstruction)
                    or not isinstance(
                        self.typed_result, W03W04W05QuestionAnswerResult)):
                raise W03W04W05RawQuestionError(
                    "unique raw question lacks FT09 execution")
            construction = self.selected_construction
            typed = self.typed_result
            if (construction.sha256()
                    != self.matched_construction_sha256s[0]
                    or construction.question_surface
                    != self.request.question_surface
                    or typed.request.question_surface
                    != self.request.question_surface
                    or typed.request.vertical_query
                    != construction.vertical_result.query
                    or typed.request.target_role_keys
                    != (construction.target_role_key,)
                    or typed.request.source_record_key
                    != self.request.source_record_key
                    or self.status != typed.status
                    or self.answer_surface != typed.answer_surface):
                raise W03W04W05RawQuestionError(
                    "raw question result escaped its construction or FT09")
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05RawQuestionError(
                "raw question boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_surface": self.answer_surface,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "matched_construction_sha256s": list(
                self.matched_construction_sha256s),
            "request": self.request.to_dict(),
            "selected_construction": (
                None if self.selected_construction is None
                else self.selected_construction.to_dict()
            ),
            "status": self.status,
            "typed_result": (
                None if self.typed_result is None
                else self.typed_result.to_dict()
            ),
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "RAW_QUESTION_SEGMENT_KINDS",
    "RAW_QUESTION_STATUSES",
    "RawQuestionAnswerResult",
    "RawQuestionConstruction",
    "RawQuestionConstructionSegment",
    "RawQuestionPattern",
    "RawQuestionPatternSegment",
    "RawQuestionRequest",
    "W03W04W05RawQuestionError",
]
