"""FT29 source-bound definition display and citation contracts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_inline_ast import (
    MEDIAWIKI_INLINE_PARSER_VERSION,
    MediaWikiInlineParseError,
    project_mediawiki_inline,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseSourceRef,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition_contract import (
    W05SourceBoundProposition,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa_contract import (
    W05RawDefinitionAnswerResult,
)


W05_DEFINITION_DISPLAY_STATUSES = {
    "AMBIGUOUS_RENDERING",
    "DISPLAY",
    "MALFORMED_MARKUP",
    "NO_SOURCE_ANSWER",
    "SOURCE_NOT_UNIQUE",
    "UNSUPPORTED_MARKUP",
}


# object-model: exception
class W05DefinitionRenderingError(ValueError):
    """The FT29 display, citation, or commitment contract drifted."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W05DefinitionRenderingError(f"{where} is not a SHA-256 digest")
    return value


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W05DefinitionRenderingError(f"{where} is not an integer key")
    return value


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W05DefinitionRenderingError(f"{where} is not canonical text")
    return value


def _optional_text(value: object, *, where: str) -> str | None:
    if value is None:
        return None
    return _text(value, where=where)


def _citation_payload(
        proposition_key: tuple[int, ...],
        primitive_key: tuple[int, ...],
        entry_key: tuple[int, ...],
        sense_key: tuple[int, ...],
        concept_key: tuple[int, ...],
        observation_key: tuple[int, ...],
        source_ref: W03PublicSenseSourceRef,
        raw_source_sha256: str,
        source_answer_trace_commitment_sha256: str,
        proposition_query_record_commitment_sha256: str,
        primitive_projection_sha256: str,
        proposition_projection_sha256: str,
        epistemic_status: str,
        truth_status: str,
        ) -> dict[str, object]:
    return {
        "concept_key": list(concept_key),
        "entry_key": list(entry_key),
        "epistemic_status": epistemic_status,
        "observation_key": list(observation_key),
        "primitive_key": list(primitive_key),
        "primitive_projection_sha256": primitive_projection_sha256,
        "proposition_key": list(proposition_key),
        "proposition_projection_sha256": proposition_projection_sha256,
        "proposition_query_record_commitment_sha256": (
            proposition_query_record_commitment_sha256),
        "raw_source_sha256": raw_source_sha256,
        "sense_key": list(sense_key),
        "source_answer_trace_commitment_sha256": (
            source_answer_trace_commitment_sha256),
        "source_ref": source_ref.to_dict(),
        "truth_status": truth_status,
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05DefinitionCitationProjection:
    """One citation closed over the selected FT28 proposition chain."""

    proposition_key: tuple[int, ...]
    primitive_key: tuple[int, ...]
    entry_key: tuple[int, ...]
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    source_ref: W03PublicSenseSourceRef
    raw_source_sha256: str
    source_answer_trace_commitment_sha256: str
    proposition_query_record_commitment_sha256: str
    primitive_projection_sha256: str
    proposition_projection_sha256: str
    epistemic_status: str
    truth_status: str
    citation_commitment_sha256: str

    def __post_init__(self) -> None:
        for name in (
                "proposition_key", "primitive_key", "entry_key", "sense_key",
                "concept_key", "observation_key"):
            _key(getattr(self, name), where=f"definition citation {name}")
        if not isinstance(self.source_ref, W03PublicSenseSourceRef):
            raise TypeError("definition citation SourceRef type is invalid")
        for name in (
                "raw_source_sha256",
                "source_answer_trace_commitment_sha256",
                "proposition_query_record_commitment_sha256",
                "primitive_projection_sha256",
                "proposition_projection_sha256",
                "citation_commitment_sha256"):
            _sha256(getattr(self, name), where=f"definition citation {name}")
        _text(self.epistemic_status, where="definition citation epistemic status")
        _text(self.truth_status, where="definition citation truth status")
        payload = _citation_payload(
            self.proposition_key,
            self.primitive_key,
            self.entry_key,
            self.sense_key,
            self.concept_key,
            self.observation_key,
            self.source_ref,
            self.raw_source_sha256,
            self.source_answer_trace_commitment_sha256,
            self.proposition_query_record_commitment_sha256,
            self.primitive_projection_sha256,
            self.proposition_projection_sha256,
            self.epistemic_status,
            self.truth_status,
        )
        if self.citation_commitment_sha256 != _sha(payload):
            raise W05DefinitionRenderingError(
                "definition citation commitment drifted")

    def to_dict(self) -> dict[str, object]:
        value = _citation_payload(
            self.proposition_key,
            self.primitive_key,
            self.entry_key,
            self.sense_key,
            self.concept_key,
            self.observation_key,
            self.source_ref,
            self.raw_source_sha256,
            self.source_answer_trace_commitment_sha256,
            self.proposition_query_record_commitment_sha256,
            self.primitive_projection_sha256,
            self.proposition_projection_sha256,
            self.epistemic_status,
            self.truth_status,
        )
        value["citation_commitment_sha256"] = self.citation_commitment_sha256
        return value


def definition_citation_projection(
        answer: W05RawDefinitionAnswerResult,
        proposition: W05SourceBoundProposition,
        ) -> W05DefinitionCitationProjection:
    """Project a citation without widening the selected proposition set."""
    if (not isinstance(answer, W05RawDefinitionAnswerResult)
            or not isinstance(proposition, W05SourceBoundProposition)):
        raise TypeError("definition citation inputs are invalid")
    if (answer.status != "ANSWER" or answer.answer_text is None
            or answer.proposition_result is None
            or proposition not in answer.selected_propositions
            or proposition.definition_text != answer.answer_text):
        raise W05DefinitionRenderingError(
            "definition citation escaped the FT28 selection")
    primitive = proposition.primitive
    result = answer.proposition_result
    values = (
        proposition.proposition_key,
        primitive.primitive_key,
        primitive.entry_key,
        primitive.sense_key,
        primitive.concept_key,
        primitive.observation_key,
        primitive.source_ref,
        hashlib.sha256(answer.answer_text.encode("utf-8")).hexdigest(),
        answer.trace_commitment_sha256,
        result.record_commitment_sha256,
        result.primitive_projection_sha256,
        result.proposition_projection_sha256,
        proposition.epistemic_status,
        proposition.truth_status,
    )
    commitment = _sha(_citation_payload(*values))
    return W05DefinitionCitationProjection(*values, commitment)


def _render_status(code: str) -> str:
    if code == "AMBIGUOUS_LINK":
        return "AMBIGUOUS_RENDERING"
    if code in {
            "NESTED_MARKUP", "UNKNOWN_TEMPLATE", "UNSUPPORTED_INLINE_MARKUP",
            "UNSUPPORTED_LINK_TARGET", "UNSUPPORTED_VARIABLE"}:
        return "UNSUPPORTED_MARKUP"
    return "MALFORMED_MARKUP"


def _expected_values(
        answer: W05RawDefinitionAnswerResult,
        ) -> tuple[
            str,
            str | None,
            str | None,
            tuple[W05DefinitionCitationProjection, ...],
            str | None,
            str | None,
            str | None,
        ]:
    if not isinstance(answer, W05RawDefinitionAnswerResult):
        raise TypeError("definition display answer type is invalid")
    if answer.status != "ANSWER":
        return "NO_SOURCE_ANSWER", None, None, (), None, None, None
    if answer.answer_text is None:
        raise W05DefinitionRenderingError("ANSWER has no raw source text")
    citations = tuple(
        definition_citation_projection(answer, item)
        for item in answer.selected_propositions)
    if len(answer.selected_propositions) != 1:
        return (
            "SOURCE_NOT_UNIQUE", answer.answer_text, None, citations,
            None, None, None,
        )
    try:
        projection = project_mediawiki_inline(answer.answer_text)
    except MediaWikiInlineParseError as error:
        return (
            _render_status(error.code),
            answer.answer_text,
            None,
            citations,
            None,
            None,
            error.code,
        )
    return (
        "DISPLAY",
        answer.answer_text,
        projection.display_text,
        citations,
        projection.document.ast_sha256,
        projection.projection_sha256,
        None,
    )


def _display_payload(
        *,
        status: str,
        raw_source_text: str | None,
        display_text: str | None,
        citations: tuple[W05DefinitionCitationProjection, ...],
        ast_sha256: str | None,
        inline_projection_sha256: str | None,
        failure_code: str | None,
        source_answer_sha256: str,
        source_answer_status: str,
        source_answer_trace_commitment_sha256: str,
        experimental: int,
        formal_mastery_claim: int,
        w03_started: int,
        w04_started: int,
        w05_started: int,
        ) -> dict[str, object]:
    return {
        "ast_sha256": ast_sha256,
        "citations": [item.to_dict() for item in citations],
        "display_text": display_text,
        "experimental": experimental,
        "failure_code": failure_code,
        "formal_mastery_claim": formal_mastery_claim,
        "inline_projection_sha256": inline_projection_sha256,
        "parser_version": MEDIAWIKI_INLINE_PARSER_VERSION,
        "raw_source_text": raw_source_text,
        "source_answer_sha256": source_answer_sha256,
        "source_answer_status": source_answer_status,
        "source_answer_trace_commitment_sha256": (
            source_answer_trace_commitment_sha256),
        "status": status,
        "w03_started": w03_started,
        "w04_started": w04_started,
        "w05_started": w05_started,
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W05DefinitionDisplayResult:
    """An opt-in display projection that retains the exact FT28 answer."""

    source_answer: W05RawDefinitionAnswerResult
    status: str
    raw_source_text: str | None
    display_text: str | None
    citations: tuple[W05DefinitionCitationProjection, ...]
    ast_sha256: str | None
    inline_projection_sha256: str | None
    failure_code: str | None
    source_answer_sha256: str
    source_answer_trace_commitment_sha256: str
    display_projection_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.source_answer, W05RawDefinitionAnswerResult)
                or self.status not in W05_DEFINITION_DISPLAY_STATUSES
                or not isinstance(self.citations, tuple)
                or any(not isinstance(item, W05DefinitionCitationProjection)
                       for item in self.citations)):
            raise W05DefinitionRenderingError(
                "definition display projection structure drifted")
        _optional_text(self.raw_source_text, where="definition raw source")
        _optional_text(self.display_text, where="definition display text")
        _optional_text(self.failure_code, where="definition display failure")
        for name in (
                "source_answer_sha256",
                "source_answer_trace_commitment_sha256",
                "display_projection_sha256"):
            _sha256(getattr(self, name), where=f"definition display {name}")
        if (self.source_answer_sha256 != self.source_answer.sha256()
                or self.source_answer_trace_commitment_sha256
                != self.source_answer.trace_commitment_sha256):
            raise W05DefinitionRenderingError(
                "definition display escaped the FT28 trace")
        expected = _expected_values(self.source_answer)
        actual = (
            self.status,
            self.raw_source_text,
            self.display_text,
            self.citations,
            self.ast_sha256,
            self.inline_projection_sha256,
            self.failure_code,
        )
        if actual != expected:
            raise W05DefinitionRenderingError(
                "definition display projection does not match its source")
        if self.ast_sha256 is not None:
            _sha256(self.ast_sha256, where="definition display AST")
        if self.inline_projection_sha256 is not None:
            _sha256(
                self.inline_projection_sha256,
                where="definition inline projection",
            )
        if (self.experimental, self.formal_mastery_claim,
                self.w03_started, self.w04_started,
                self.w05_started) != (1, 0, 0, 0, 0):
            raise W05DefinitionRenderingError(
                "definition display formal boundary drifted")
        payload = _display_payload(
            status=self.status,
            raw_source_text=self.raw_source_text,
            display_text=self.display_text,
            citations=self.citations,
            ast_sha256=self.ast_sha256,
            inline_projection_sha256=self.inline_projection_sha256,
            failure_code=self.failure_code,
            source_answer_sha256=self.source_answer_sha256,
            source_answer_status=self.source_answer.status,
            source_answer_trace_commitment_sha256=(
                self.source_answer_trace_commitment_sha256),
            experimental=self.experimental,
            formal_mastery_claim=self.formal_mastery_claim,
            w03_started=self.w03_started,
            w04_started=self.w04_started,
            w05_started=self.w05_started,
        )
        if self.display_projection_sha256 != _sha(payload):
            raise W05DefinitionRenderingError(
                "definition display commitment drifted")

    @classmethod
    def from_source_answer(
            cls,
            answer: W05RawDefinitionAnswerResult,
            ) -> "W05DefinitionDisplayResult":
        values = _expected_values(answer)
        source_answer_sha256 = answer.sha256()
        payload = _display_payload(
            status=values[0],
            raw_source_text=values[1],
            display_text=values[2],
            citations=values[3],
            ast_sha256=values[4],
            inline_projection_sha256=values[5],
            failure_code=values[6],
            source_answer_sha256=source_answer_sha256,
            source_answer_status=answer.status,
            source_answer_trace_commitment_sha256=(
                answer.trace_commitment_sha256),
            experimental=1,
            formal_mastery_claim=0,
            w03_started=0,
            w04_started=0,
            w05_started=0,
        )
        return cls(
            answer,
            *values,
            source_answer_sha256,
            answer.trace_commitment_sha256,
            _sha(payload),
        )

    def to_dict(self) -> dict[str, object]:
        value = _display_payload(
            status=self.status,
            raw_source_text=self.raw_source_text,
            display_text=self.display_text,
            citations=self.citations,
            ast_sha256=self.ast_sha256,
            inline_projection_sha256=self.inline_projection_sha256,
            failure_code=self.failure_code,
            source_answer_sha256=self.source_answer_sha256,
            source_answer_status=self.source_answer.status,
            source_answer_trace_commitment_sha256=(
                self.source_answer_trace_commitment_sha256),
            experimental=self.experimental,
            formal_mastery_claim=self.formal_mastery_claim,
            w03_started=self.w03_started,
            w04_started=self.w04_started,
            w05_started=self.w05_started,
        )
        value["display_projection_sha256"] = self.display_projection_sha256
        return value

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W05_DEFINITION_DISPLAY_STATUSES",
    "W05DefinitionCitationProjection",
    "W05DefinitionDisplayResult",
    "W05DefinitionRenderingError",
    "definition_citation_projection",
]
