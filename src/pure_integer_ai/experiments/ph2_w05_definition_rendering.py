"""FT29 opt-in rendering of an already selected FT28 raw definition."""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w05_definition_rendering_contract import (
    W05DefinitionDisplayResult,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa_contract import (
    W05RawDefinitionAnswerResult,
)


def render_w05_definition_answer(
        answer: W05RawDefinitionAnswerResult,
        ) -> W05DefinitionDisplayResult:
    """Render only the exact candidate set authorized by FT28."""
    if not isinstance(answer, W05RawDefinitionAnswerResult):
        raise TypeError("definition rendering requires an FT28 answer")
    return W05DefinitionDisplayResult.from_source_answer(answer)


def render_w05_definition_batch(
        answers: tuple[W05RawDefinitionAnswerResult, ...],
        ) -> tuple[W05DefinitionDisplayResult, ...]:
    """Render a bounded answer batch without retaining additional state."""
    if (not isinstance(answers, tuple) or not answers
            or any(not isinstance(item, W05RawDefinitionAnswerResult)
                   for item in answers)):
        raise TypeError("definition rendering batch is invalid")
    return tuple(render_w05_definition_answer(item) for item in answers)


__all__ = [
    "render_w05_definition_answer",
    "render_w05_definition_batch",
]
