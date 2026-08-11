"""FT28 通用原始定义问式与来源绑定回答合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition_contract import (
    W04W05SourceBoundPropositionQueryResult,
    W05SourceBoundProposition,
)


W05_RAW_DEFINITION_STATUSES = {
    "ANSWER", "AMBIGUOUS", "CONFLICT", "UNKNOWN", "CLARIFY"}


# object-model: exception
class W05RawDefinitionQAError(ValueError):
    """定义问式、来源候选或回答 trace 发生合同漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W05RawDefinitionQAError(f"{where} is not canonical text")
    return value


def _optional_text(value: object, *, where: str) -> str | None:
    if value is None:
        return None
    return _text(value, where=where)


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W05RawDefinitionQAError(
            f"{where} is not a strict integer key")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W05RawDefinitionQAError(
            f"{where} is not a SHA-256 digest")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05DefinitionQuestionConstruction:
    """不绑定任何词项或答案的通用定义问句构式。"""

    construction_key: tuple[int, ...]
    construction_id: str
    language: str
    leading_literal: str
    trailing_literal: str
    boundary_marks: tuple[str, ...]

    def __post_init__(self) -> None:
        _key(self.construction_key, where="definition construction key")
        _text(self.construction_id, where="definition construction id")
        _text(self.language, where="definition construction language")
        if (not isinstance(self.leading_literal, str)
                or not isinstance(self.trailing_literal, str)
                or (not self.leading_literal and not self.trailing_literal)
                or self.leading_literal.strip() != self.leading_literal
                or self.trailing_literal.strip() != self.trailing_literal):
            raise W05RawDefinitionQAError(
                "definition construction literals drifted")
        if (not isinstance(self.boundary_marks, tuple)
                or not self.boundary_marks
                or any(not isinstance(item, str)
                       or item.strip() != item
                       for item in self.boundary_marks)
                or len(set(self.boundary_marks))
                != len(self.boundary_marks)):
            raise W05RawDefinitionQAError(
                "definition construction boundaries drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "boundary_marks": list(self.boundary_marks),
            "construction_id": self.construction_id,
            "construction_key": list(self.construction_key),
            "language": self.language,
            "leading_literal": self.leading_literal,
            "trailing_literal": self.trailing_literal,
        }


def default_definition_question_constructions(
        ) -> tuple[W05DefinitionQuestionConstruction, ...]:
    """返回当前公开的两种中文定义问式结构。"""
    boundaries = ("", "?", "？")
    return (
        W05DefinitionQuestionConstruction(
            (1, 28, 5, 1),
            "ZH_DEFINITION_PREFIX_SHENME_SHI",
            "zh",
            "什么是",
            "",
            boundaries,
        ),
        W05DefinitionQuestionConstruction(
            (1, 28, 5, 2),
            "ZH_DEFINITION_SUFFIX_SHI_SHENME_YISI",
            "zh",
            "",
            "是什么意思",
            boundaries,
        ),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05RawDefinitionRequest:
    """一个不携带 expected answer 的原始定义问题。"""

    question_surface: str
    context_text: str | None = None
    language: str = "zh"

    def __post_init__(self) -> None:
        _text(self.question_surface, where="definition question surface")
        _optional_text(self.context_text, where="definition question context")
        _text(self.language, where="definition question language")

    def to_dict(self) -> dict[str, object]:
        return {
            "context_text": self.context_text,
            "language": self.language,
            "question_surface": self.question_surface,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05DefinitionQuestionMatch:
    """通用构式对原始问题的一个无答案匹配。"""

    construction: W05DefinitionQuestionConstruction
    term_surface: str
    boundary_mark: str

    def __post_init__(self) -> None:
        if not isinstance(
                self.construction, W05DefinitionQuestionConstruction):
            raise TypeError("definition match construction type is invalid")
        _text(self.term_surface, where="definition match term")
        if self.boundary_mark not in self.construction.boundary_marks:
            raise W05RawDefinitionQAError(
                "definition match boundary escaped its construction")

    def to_dict(self) -> dict[str, object]:
        return {
            "boundary_mark": self.boundary_mark,
            "construction": self.construction.to_dict(),
            "term_surface": self.term_surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05RawDefinitionTrace:
    """从问式匹配到来源 proposition 选择的可审计纯身份路径。"""

    request_sha256: str
    matched_construction_keys: tuple[tuple[int, ...], ...]
    proposition_result_sha256: str | None
    primitive_projection_sha256: str | None
    proposition_projection_sha256: str | None
    definition_candidate_keys: tuple[tuple[int, ...], ...]
    selected_proposition_keys: tuple[tuple[int, ...], ...]
    decision_status: str

    def __post_init__(self) -> None:
        _sha256(self.request_sha256, where="definition trace request")
        for name in (
                "matched_construction_keys", "definition_candidate_keys",
                "selected_proposition_keys"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, tuple) for item in values)):
                raise W05RawDefinitionQAError(
                    f"definition trace {name} drifted")
            for item in values:
                _key(item, where=f"definition trace {name} item")
        for name in (
                "proposition_result_sha256",
                "primitive_projection_sha256",
                "proposition_projection_sha256"):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, where=f"definition trace {name}")
        if self.decision_status not in W05_RAW_DEFINITION_STATUSES:
            raise W05RawDefinitionQAError(
                "definition trace decision status drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_status": self.decision_status,
            "definition_candidate_keys": [
                list(item) for item in self.definition_candidate_keys],
            "matched_construction_keys": [
                list(item) for item in self.matched_construction_keys],
            "primitive_projection_sha256": (
                self.primitive_projection_sha256),
            "proposition_projection_sha256": (
                self.proposition_projection_sha256),
            "proposition_result_sha256": self.proposition_result_sha256,
            "request_sha256": self.request_sha256,
            "selected_proposition_keys": [
                list(item) for item in self.selected_proposition_keys],
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def _expected_status(
        result: W04W05SourceBoundPropositionQueryResult,
        definitions: tuple[W05SourceBoundProposition, ...],
        ) -> tuple[str, str | None, tuple[W05SourceBoundProposition, ...]]:
    if result.status != "UNIQUE":
        mapped = {
            "AMBIGUOUS": "AMBIGUOUS",
            "CONFLICT": "CONFLICT",
            "CLARIFY": "CLARIFY",
            "UNKNOWN": "UNKNOWN",
        }
        return mapped[result.status], None, ()
    texts = tuple(sorted(set(
        item.definition_text for item in definitions
        if item.definition_text is not None)))
    if not texts:
        return "UNKNOWN", None, ()
    if len(texts) > 1:
        return "CLARIFY", None, ()
    selected = tuple(
        item for item in definitions if item.definition_text == texts[0])
    return "ANSWER", texts[0], selected


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05RawDefinitionAnswerResult:
    """原始定义问题的来源绑定回答或诚实拒答结果。"""

    request: W05RawDefinitionRequest
    status: str
    matches: tuple[W05DefinitionQuestionMatch, ...]
    term_surface: str | None
    proposition_result: W04W05SourceBoundPropositionQueryResult | None
    definition_candidates: tuple[W05SourceBoundProposition, ...]
    selected_propositions: tuple[W05SourceBoundProposition, ...]
    answer_text: str | None
    trace: W05RawDefinitionTrace
    trace_commitment_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.request, W05RawDefinitionRequest)
                or self.status not in W05_RAW_DEFINITION_STATUSES
                or not isinstance(self.matches, tuple)
                or any(not isinstance(item, W05DefinitionQuestionMatch)
                       for item in self.matches)
                or not isinstance(self.definition_candidates, tuple)
                or any(not isinstance(item, W05SourceBoundProposition)
                       for item in self.definition_candidates)
                or not isinstance(self.selected_propositions, tuple)
                or any(not isinstance(item, W05SourceBoundProposition)
                       for item in self.selected_propositions)
                or not isinstance(self.trace, W05RawDefinitionTrace)):
            raise W05RawDefinitionQAError(
                "definition answer projection drifted")
        _optional_text(self.term_surface, where="definition answer term")
        _optional_text(self.answer_text, where="definition answer text")
        _sha256(
            self.trace_commitment_sha256,
            where="definition answer trace commitment",
        )
        if self.trace.sha256() != self.trace_commitment_sha256:
            raise W05RawDefinitionQAError(
                "definition answer trace commitment drifted")
        if self.trace.request_sha256 != self.request.sha256():
            raise W05RawDefinitionQAError(
                "definition answer request trace drifted")
        construction_keys = tuple(
            item.construction.construction_key for item in self.matches)
        if self.trace.matched_construction_keys != construction_keys:
            raise W05RawDefinitionQAError(
                "definition answer construction trace drifted")
        if len(self.matches) != 1:
            expected = "UNKNOWN" if not self.matches else "CLARIFY"
            if (self.status != expected or self.term_surface is not None
                    or self.proposition_result is not None
                    or self.definition_candidates
                    or self.selected_propositions
                    or self.answer_text is not None):
                raise W05RawDefinitionQAError(
                    "unmatched or multi-matched question published knowledge")
        else:
            match = self.matches[0]
            result = self.proposition_result
            if (self.term_surface != match.term_surface
                    or not isinstance(
                        result, W04W05SourceBoundPropositionQueryResult)
                    or result.query.surface != match.term_surface
                    or result.query.context_text != self.request.context_text
                    or result.query.language != self.request.language):
                raise W05RawDefinitionQAError(
                    "definition answer escaped its term query")
            expected_definitions = tuple(
                item for item in result.propositions
                if item.primitive.active == 1
                and item.relation_kind == "DEFINITION"
                and item.definition_text is not None)
            if self.definition_candidates != expected_definitions:
                raise W05RawDefinitionQAError(
                    "definition answer candidate filtering drifted")
            expected_status, expected_answer, expected_selected = (
                _expected_status(result, expected_definitions))
            if (self.status != expected_status
                    or self.answer_text != expected_answer
                    or self.selected_propositions != expected_selected):
                raise W05RawDefinitionQAError(
                    "definition answer uniqueness decision drifted")
            if (self.trace.proposition_result_sha256 != result.sha256()
                    or self.trace.primitive_projection_sha256
                    != result.primitive_projection_sha256
                    or self.trace.proposition_projection_sha256
                    != result.proposition_projection_sha256):
                raise W05RawDefinitionQAError(
                    "definition answer proposition trace drifted")
        if self.trace.decision_status != self.status:
            raise W05RawDefinitionQAError(
                "definition answer decision trace drifted")
        if self.trace.definition_candidate_keys != tuple(
                item.proposition_key for item in self.definition_candidates):
            raise W05RawDefinitionQAError(
                "definition candidate key trace drifted")
        if self.trace.selected_proposition_keys != tuple(
                item.proposition_key for item in self.selected_propositions):
            raise W05RawDefinitionQAError(
                "selected proposition key trace drifted")
        if self.status != "ANSWER" and (
                self.answer_text is not None or self.selected_propositions):
            raise W05RawDefinitionQAError(
                "non-answer result published an answer")
        if (self.experimental, self.formal_mastery_claim,
                self.w03_started, self.w04_started,
                self.w05_started) != (1, 0, 0, 0, 0):
            raise W05RawDefinitionQAError(
                "definition answer formal boundary drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_text": self.answer_text,
            "definition_candidates": [
                item.to_dict() for item in self.definition_candidates],
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "matches": [item.to_dict() for item in self.matches],
            "proposition_result": (
                None if self.proposition_result is None
                else self.proposition_result.to_dict()
            ),
            "request": self.request.to_dict(),
            "selected_propositions": [
                item.to_dict() for item in self.selected_propositions],
            "status": self.status,
            "term_surface": self.term_surface,
            "trace": self.trace.to_dict(),
            "trace_commitment_sha256": self.trace_commitment_sha256,
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W05_RAW_DEFINITION_STATUSES",
    "W05DefinitionQuestionConstruction",
    "W05DefinitionQuestionMatch",
    "W05RawDefinitionAnswerResult",
    "W05RawDefinitionQAError",
    "W05RawDefinitionRequest",
    "W05RawDefinitionTrace",
    "default_definition_question_constructions",
]
