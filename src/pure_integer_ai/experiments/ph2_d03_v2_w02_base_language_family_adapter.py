"""Explicit lzh-to-zh scope adapter for W-02 base Candidate prediction."""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.experiments.ph2_dataset_contract import ObservationRecord
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02CandidatePrediction,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02DevCandidateIndex,
    predict_w02_dev_observation,
)


W02_BASE_LANGUAGE_FAMILY_ADAPTER_VERSION = (
    "PH2-D03-V2-W02-BASE-LANGUAGE-FAMILY-ADAPTER-V1"
)
W02_BASE_SCOPE_LANGUAGE = "zh"
W02_BASE_ADAPTABLE_SOURCE_LANGUAGES = ("lzh", "zh")


# object-model: exception
class W02BaseLanguageFamilyAdapterError(RuntimeError):
    """A base-only language-family view changed identity or payload."""


def adapt_w02_observation_for_base_candidate(
        observation: ObservationRecord) -> ObservationRecord:
    """Return a base-only zh view while preserving the original observation."""
    if not isinstance(observation, ObservationRecord):
        raise TypeError("W-02 base language adapter requires ObservationRecord")
    if (observation.w_stage != "W-02"
            or observation.payload_kind != "typed_carrier"
            or observation.language not in W02_BASE_ADAPTABLE_SOURCE_LANGUAGES):
        raise W02BaseLanguageFamilyAdapterError(
            "W-02 base language adapter scope is not authorized")
    if observation.language == W02_BASE_SCOPE_LANGUAGE:
        return observation
    adapted = replace(observation, language=W02_BASE_SCOPE_LANGUAGE)
    expected = observation.to_dict()
    expected["language"] = W02_BASE_SCOPE_LANGUAGE
    if adapted.to_dict() != expected:
        raise W02BaseLanguageFamilyAdapterError(
            "W-02 base language adapter changed non-language fields")
    return adapted


def predict_w02_dev_observation_language_family(
        index: W02DevCandidateIndex,
        observation: ObservationRecord,
        ) -> tuple[W02CandidatePrediction, int]:
    """Run the unchanged base predictor through the explicit scope adapter."""
    adapted = adapt_w02_observation_for_base_candidate(observation)
    return predict_w02_dev_observation(index, adapted)


__all__ = [
    "W02_BASE_ADAPTABLE_SOURCE_LANGUAGES",
    "W02_BASE_LANGUAGE_FAMILY_ADAPTER_VERSION",
    "W02_BASE_SCOPE_LANGUAGE",
    "W02BaseLanguageFamilyAdapterError",
    "adapt_w02_observation_for_base_candidate",
    "predict_w02_dev_observation_language_family",
]
