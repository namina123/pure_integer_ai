"""FT28 通用定义问式解析与来源绑定 raw definition QA。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition import (
    W04W05SourceBoundPropositionRuntime,
    query_w04_w05_source_bound_propositions,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa_contract import (
    W05DefinitionQuestionConstruction,
    W05DefinitionQuestionMatch,
    W05RawDefinitionAnswerResult,
    W05RawDefinitionRequest,
    W05RawDefinitionTrace,
    default_definition_question_constructions,
)


def _match_construction(
        request: W05RawDefinitionRequest,
        construction: W05DefinitionQuestionConstruction,
        ) -> tuple[W05DefinitionQuestionMatch, ...]:
    if request.language != construction.language:
        return ()
    matches = []
    marked_boundaries = tuple(
        item for item in construction.boundary_marks if item)
    terminal_marks = tuple(
        item for item in marked_boundaries
        if request.question_surface.endswith(item))
    boundaries = terminal_marks or (
        ("",) if "" in construction.boundary_marks else ())
    for boundary in boundaries:
        surface = request.question_surface
        if boundary:
            if not surface.endswith(boundary):
                continue
            surface = surface[:-len(boundary)]
        if (construction.leading_literal
                and not surface.startswith(construction.leading_literal)):
            continue
        if (construction.trailing_literal
                and not surface.endswith(construction.trailing_literal)):
            continue
        start = len(construction.leading_literal)
        end = (
            len(surface) - len(construction.trailing_literal)
            if construction.trailing_literal else len(surface)
        )
        term = surface[start:end]
        if not term or term.strip() != term:
            continue
        matches.append(W05DefinitionQuestionMatch(
            construction, term, boundary))
    return tuple(matches)


def match_w05_raw_definition_question(
        request: W05RawDefinitionRequest,
        *,
        constructions: tuple[W05DefinitionQuestionConstruction, ...]
        | None = None,
        ) -> tuple[W05DefinitionQuestionMatch, ...]:
    """以通用构式匹配问题，不查询词项或答案数据。"""
    if not isinstance(request, W05RawDefinitionRequest):
        raise TypeError("definition matching requires a typed request")
    values = (
        default_definition_question_constructions()
        if constructions is None else constructions
    )
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, W05DefinitionQuestionConstruction)
                   for item in values)):
        raise TypeError("definition constructions are invalid")
    return tuple(
        match
        for construction in values
        for match in _match_construction(request, construction)
    )


def _decision(
        proposition_result,
        definitions,
        ) -> tuple[str, str | None, tuple[object, ...]]:
    if proposition_result.status != "UNIQUE":
        return {
            "AMBIGUOUS": "AMBIGUOUS",
            "CONFLICT": "CONFLICT",
            "CLARIFY": "CLARIFY",
            "UNKNOWN": "UNKNOWN",
        }[proposition_result.status], None, ()
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


def _result_without_query(
        request: W05RawDefinitionRequest,
        matches: tuple[W05DefinitionQuestionMatch, ...],
        ) -> W05RawDefinitionAnswerResult:
    status = "UNKNOWN" if not matches else "CLARIFY"
    trace = W05RawDefinitionTrace(
        request.sha256(),
        tuple(item.construction.construction_key for item in matches),
        None,
        None,
        None,
        (),
        (),
        status,
    )
    return W05RawDefinitionAnswerResult(
        request,
        status,
        matches,
        None,
        None,
        (),
        (),
        None,
        trace,
        trace.sha256(),
    )


def answer_w05_raw_definition_question(
        runtime: W04W05SourceBoundPropositionRuntime,
        request: W05RawDefinitionRequest,
        ) -> W05RawDefinitionAnswerResult:
    """只在构式、来源状态和 definition 文本都唯一时返回原文定义。"""
    if (not isinstance(runtime, W04W05SourceBoundPropositionRuntime)
            or not isinstance(request, W05RawDefinitionRequest)):
        raise TypeError("definition QA inputs are invalid")
    matches = match_w05_raw_definition_question(request)
    if len(matches) != 1:
        return _result_without_query(request, matches)
    term = matches[0].term_surface
    proposition_result = query_w04_w05_source_bound_propositions(
        runtime,
        W03PublicSenseQuery(
            term, request.context_text, request.language),
    )
    definitions = tuple(
        item for item in proposition_result.propositions
        if item.primitive.active == 1
        and item.relation_kind == "DEFINITION"
        and item.definition_text is not None)
    status, answer, selected = _decision(
        proposition_result, definitions)
    trace = W05RawDefinitionTrace(
        request.sha256(),
        tuple(item.construction.construction_key for item in matches),
        proposition_result.sha256(),
        proposition_result.primitive_projection_sha256,
        proposition_result.proposition_projection_sha256,
        tuple(item.proposition_key for item in definitions),
        tuple(item.proposition_key for item in selected),
        status,
    )
    return W05RawDefinitionAnswerResult(
        request,
        status,
        matches,
        term,
        proposition_result,
        definitions,
        selected,
        answer,
        trace,
        trace.sha256(),
    )


def answer_w05_raw_definition_batch(
        runtime: W04W05SourceBoundPropositionRuntime,
        requests: tuple[W05RawDefinitionRequest, ...],
        ) -> tuple[W05RawDefinitionAnswerResult, ...]:
    """在同一不可变 runtime 上回答有界定义问题批次。"""
    if (not isinstance(requests, tuple) or not requests
            or any(not isinstance(item, W05RawDefinitionRequest)
                   for item in requests)):
        raise TypeError("definition QA batch is invalid")
    return tuple(
        answer_w05_raw_definition_question(runtime, item)
        for item in requests)


__all__ = [
    "answer_w05_raw_definition_batch",
    "answer_w05_raw_definition_question",
    "match_w05_raw_definition_question",
]
