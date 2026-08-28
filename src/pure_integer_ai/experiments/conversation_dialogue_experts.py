"""General-first learned dialogue expert routing with learned activation.

No domain names or keyword lists are embedded here.  Activation features are
derived from integer feature counts in a domain model relative to the general
model.  A domain expert is queried only after the general expert declines and
the current turn supplies domain evidence, optionally reinforced by already
loaded bounded history.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from pure_integer_ai.experiments.conversation_learned_dialogue_response import (
    LearnedDialogueResponseModel,
    LearnedDialogueResponseResult,
    LearnedDialogueResponseRuntime,
    dialogue_prompt_features,
)


MIN_DOMAIN_FEATURE_SUPPORT = 2
DOMAIN_RARE_DIVISOR = 512
MIN_CURRENT_DOMAIN_FEATURES = 3
MIN_CURRENT_WITH_HISTORY_FEATURES = 1
MIN_HISTORY_DOMAIN_FEATURES = 2
DOMAIN_MIN_SIMILARITY_PERMILLE = 700


def _feature_support(
        model: LearnedDialogueResponseModel,
        ) -> dict[tuple[int, ...], int]:
    counts: dict[int, int] = {}
    for feature, _fragment, count in model.feature_fragment_counts:
        counts[feature] = counts.get(feature, 0) + count
    return {model.features[ordinal]: count
            for ordinal, count in counts.items()}


def learned_domain_activation_features(
        general: LearnedDialogueResponseModel,
        domain: LearnedDialogueResponseModel,
        ) -> frozenset[tuple[int, ...]]:
    """Derive bounded domain-discriminative n-grams from learned counts."""
    if (not isinstance(general, LearnedDialogueResponseModel)
            or not isinstance(domain, LearnedDialogueResponseModel)):
        raise TypeError("domain activation 需要 learned response model")
    general_counts = _feature_support(general)
    domain_counts = _feature_support(domain)
    maximum_support = max(
        MIN_DOMAIN_FEATURE_SUPPORT,
        domain.train_count // DOMAIN_RARE_DIVISOR)
    features = []
    for feature, domain_count in domain_counts.items():
        if (len(feature) < 2
                or not MIN_DOMAIN_FEATURE_SUPPORT <= domain_count
                <= maximum_support):
            continue
        general_count = general_counts.get(feature, 0)
        # Fourfold normalized enrichment keeps general conversational forms
        # out while retaining recurring entities and domain constructions.
        if (general_count > 0
                and domain_count * general.train_count
                < 4 * general_count * domain.train_count):
            continue
        features.append(feature)
    return frozenset(features)


def learned_grounded_domain_activation_features(
        general: LearnedDialogueResponseModel, *,
        grounded_feature_counts: dict[tuple[int, ...], int],
        conversational_feature_counts: dict[tuple[int, ...], int],
        grounded_document_count: int,
        conversational_document_count: int,
        ) -> frozenset[tuple[int, ...]]:
    """Learn domain gates enriched in grounded versus conversational rows."""
    if (not isinstance(general, LearnedDialogueResponseModel)
            or grounded_document_count <= 0
            or conversational_document_count <= 0):
        raise ValueError("grounded domain activation 统计非法")
    general_counts = _feature_support(general)
    maximum_support = max(
        MIN_DOMAIN_FEATURE_SUPPORT,
        grounded_document_count // DOMAIN_RARE_DIVISOR)
    features = []
    for feature, grounded_count in grounded_feature_counts.items():
        if (len(feature) < 2
                or not MIN_DOMAIN_FEATURE_SUPPORT <= grounded_count
                <= maximum_support):
            continue
        conversational_count = conversational_feature_counts.get(feature, 0)
        if (conversational_count > 0
                and grounded_count * conversational_document_count
                < 4 * conversational_count * grounded_document_count):
            continue
        general_count = general_counts.get(feature, 0)
        if (general_count > 0
                and grounded_count * general.train_count
                < 4 * general_count * grounded_document_count):
            continue
        features.append(feature)
    return frozenset(features)


# object-model: value; representation=struct; interop=dialogue-expert-v1
@dataclass(frozen=True, slots=True)
class LearnedDialogueDomainExpert:
    """One runtime paired with its reproducible learned activation set."""

    runtime: LearnedDialogueResponseRuntime
    activation_features: frozenset[tuple[int, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, LearnedDialogueResponseRuntime):
            raise TypeError("domain expert runtime 非法")
        if (not isinstance(self.activation_features, frozenset)
                or not self.activation_features):
            raise ValueError("domain expert activation 不能为空")


# object-model: derived_cache; representation=runtime; interop=dialogue-expert-v1
@dataclass(slots=True)
class LazyLearnedDialogueDomainExpert:
    """Activation evidence plus a one-shot loader for one embedded expert."""

    activation_features: frozenset[tuple[int, ...]]
    loader: Callable[[], LearnedDialogueResponseRuntime]
    runtime: LearnedDialogueResponseRuntime | None = None

    def load(self) -> LearnedDialogueResponseRuntime:
        if self.runtime is None:
            value = self.loader()
            if not isinstance(value, LearnedDialogueResponseRuntime):
                raise TypeError("lazy domain expert loader 返回值非法")
            self.runtime = value
        return self.runtime


# object-model: derived_cache; representation=runtime; interop=dialogue-expert-v1
class LearnedDialogueExpertRouter:
    """Query a general expert first, then uniquely activated domain experts."""

    __slots__ = ("general", "domains")

    def __init__(
            self, general: LearnedDialogueResponseRuntime,
            domains: tuple[LearnedDialogueResponseRuntime, ...] = (),
            *, lazy_domains: tuple[
                tuple[frozenset[tuple[int, ...]],
                      Callable[[], LearnedDialogueResponseRuntime]], ...] = (),
            ) -> None:
        if not isinstance(general, LearnedDialogueResponseRuntime):
            raise TypeError("general dialogue expert 非法")
        if (not isinstance(domains, tuple)
                or any(not isinstance(item, LearnedDialogueResponseRuntime)
                       for item in domains)):
            raise TypeError("domain dialogue experts 非法")
        self.general = general
        eager = tuple(
            LearnedDialogueDomainExpert(
                runtime,
                learned_domain_activation_features(
                    general.model, runtime.model))
            for runtime in domains
        )
        lazy = tuple(
            LazyLearnedDialogueDomainExpert(features, loader)
            for features, loader in lazy_domains
        )
        if any(not item.activation_features for item in lazy):
            raise ValueError("lazy domain expert activation 不能为空")
        self.domains = (*eager, *lazy)

    @staticmethod
    def _activation_counts(
            expert: LearnedDialogueDomainExpert, prompt: str,
            history: tuple[tuple[int, str], ...],
            ) -> tuple[int, int]:
        current = frozenset(dialogue_prompt_features(prompt))
        current_count = len(current & expert.activation_features)
        history_features: set[tuple[int, ...]] = set()
        for role, surface in history[-8:]:
            # Only an explicit user turn is routing evidence.  Assistant
            # surfaces are model output and must not recursively activate a
            # domain expert on a later generic follow-up.
            if role != 1:
                continue
            history_features.update(dialogue_prompt_features(surface))
        history_count = len(history_features & expert.activation_features)
        return current_count, history_count

    def respond(
            self, prompt: str, *,
            history: tuple[tuple[int, str], ...] = (),
            minimum_fragment_occurrences: int = 1,
            minimum_similarity_permille: int,
            ) -> LearnedDialogueResponseResult:
        general = self.general.respond(
            prompt, history=history,
            minimum_fragment_occurrences=minimum_fragment_occurrences,
            minimum_similarity_permille=minimum_similarity_permille)
        candidates = []
        for ordinal, expert in enumerate(self.domains):
            current_count, history_count = self._activation_counts(
                expert, prompt, history)
            activated = (
                current_count >= MIN_CURRENT_DOMAIN_FEATURES
                or current_count >= MIN_CURRENT_WITH_HISTORY_FEATURES
                and history_count >= MIN_HISTORY_DOMAIN_FEATURES)
            if not activated:
                continue
            runtime = (expert.load()
                       if isinstance(expert, LazyLearnedDialogueDomainExpert)
                       else expert.runtime)
            result = runtime.respond(
                prompt, history=history,
                minimum_fragment_occurrences=minimum_fragment_occurrences,
                minimum_similarity_permille=max(
                    minimum_similarity_permille,
                    DOMAIN_MIN_SIMILARITY_PERMILLE),
            )
            if result.used:
                candidates.append((
                    current_count, history_count,
                    result.similarity_permille, result.shared_feature_count,
                    -ordinal, ordinal, result,
                ))
        candidates.sort(reverse=True, key=lambda item: item[:-1])
        if not candidates:
            return general
        best = candidates[0]
        if len(candidates) > 1 and candidates[1][:4] == best[:4]:
            return general
        result = best[-1]
        return replace(
            result,
            reason="learned_domain_expert_" + result.reason,
            trace=(6, best[5], best[0], best[1], *result.trace),
        )

    def close(self) -> None:
        self.general.close()
        for expert in self.domains:
            runtime = expert.runtime
            if runtime is not None:
                runtime.close()


__all__ = [
    "LazyLearnedDialogueDomainExpert", "LearnedDialogueDomainExpert",
    "LearnedDialogueExpertRouter",
    "learned_domain_activation_features",
    "learned_grounded_domain_activation_features",
]
