"""把显式 grounded-answer pattern 映射为 GG-01 lexical choice。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_HYPOTHESIS,
    ObjectIdentity,
    SourceRef,
    concept_identity,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceCondition,
    GenerationChoiceHypothesis,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorVariant,
)


_NAMESPACE = 20950


# object-model: exception
class GroundedAnswerChoiceError(ValueError):
    """variant 与 typed generation candidate 不能形成同一 lexical choice。"""


def _positive_fingerprint(values: tuple[int, ...], *, domain: str) -> int:
    """把内容引用压成稳定正整数，仅用于开放 SourceRef 路由字段。"""
    fingerprint = integer_tuple_fingerprint(values, domain=domain)
    value = int.from_bytes(bytes(fingerprint[2:10]), "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


def _owned_identity(
        kind: int,
        values: tuple[int, ...],
        candidate: GenerationCandidate,
        ) -> ObjectIdentity:
    """在当前 query owner/version 中建立内容引用身份。"""
    return ObjectIdentity(
        kind,
        integer_tuple_fingerprint(
            values, domain="grounded.answer.lexical.identity.v1"),
        candidate.scope.owner,
        candidate.scope.versions,
    )


def grounded_answer_generation_context(
        variant: GroundedAnswerConnectorVariant,
        candidate: GenerationCandidate,
        ) -> ObjectIdentity:
    """建立不含 selected connector 的同次 grounded generation 上下文。"""
    if not isinstance(variant, GroundedAnswerConnectorVariant):
        raise TypeError("grounded generation context variant 类型错误")
    if not isinstance(candidate, GenerationCandidate):
        raise TypeError("grounded generation context candidate 类型错误")
    template = variant.template
    if (candidate.proposition.structure != template.proposition_structure
            or candidate.proposition.predicate != template.predicate):
        raise GroundedAnswerChoiceError(
            "grounded generation context 与 candidate match key 漂移")
    return _owned_identity(
        OBJECT_CONTEXT_SCOPE,
        (*candidate.scope.stable_key(),
         *candidate.proposition.stable_key(),
         *template.language_branch.stable_key()),
        candidate,
    )


def build_grounded_answer_lexical_choice(
        variant: GroundedAnswerConnectorVariant,
        candidate: GenerationCandidate,
        ) -> GenerationChoiceHypothesis:
    """把调用者已显式采用的 pattern 映射为 GG-01 lexical 候选。"""
    if not isinstance(variant, GroundedAnswerConnectorVariant):
        raise TypeError("grounded lexical variant 类型错误")
    if not isinstance(candidate, GenerationCandidate):
        raise TypeError("grounded lexical candidate 类型错误")
    template = variant.template
    if (candidate.proposition.structure != template.proposition_structure
            or candidate.proposition.predicate != template.predicate):
        raise GroundedAnswerChoiceError(
            "lexical choice 与 generation candidate match key 漂移")
    theory_key = template.connector.stable_key()
    context = grounded_answer_generation_context(variant, candidate)
    condition = concept_identity(
        integer_tuple_fingerprint(
            (*context.stable_key(), *candidate.proposition.stable_key()),
            domain="grounded.answer.lexical.condition.v1",
        ),
        owner=candidate.scope.owner,
        versions=candidate.scope.versions,
    )
    required = tuple(sorted((
        template.language_branch,
        template.proposition_structure,
        template.predicate,
    ), key=ObjectIdentity.stable_key))
    choice_condition = GenerationChoiceCondition(
        condition,
        context,
        required,
        (),
        candidate.scope,
    )
    competition_key = integer_tuple_fingerprint(
        (*candidate.proposition.template.stable_key(),
         *choice_condition.stable_key()),
        domain="grounded.answer.lexical.competition.v1",
    )
    choice_identity = _owned_identity(
        OBJECT_HYPOTHESIS,
        (*competition_key,
         *template.connector.stable_key(),
         variant.option.pattern_id),
        candidate,
    )
    forming_source = SourceRef(
        _NAMESPACE,
        _positive_fingerprint(
            theory_key, domain="grounded.answer.lexical.source.v1"),
        variant.option.pattern_id,
        candidate.scope.owner,
        candidate.scope.versions,
    )
    return GenerationChoiceHypothesis(
        choice_identity,
        "LEXICAL_REALIZATION_CHOICE",
        candidate.proposition.template,
        choice_condition,
        template.connector,
        (forming_source,),
        competition_key,
        candidate.scope,
    )


__all__ = [
    "GroundedAnswerChoiceError",
    "build_grounded_answer_lexical_choice",
    "grounded_answer_generation_context",
]
