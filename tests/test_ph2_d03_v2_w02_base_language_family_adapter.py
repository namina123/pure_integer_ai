"""Public metamorphic tests for the W-02 lzh base adapter."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_base_language_family_adapter import (
    W02BaseLanguageFamilyAdapterError,
    adapt_w02_observation_for_base_candidate,
    predict_w02_dev_observation_language_family,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_base_language_family_probe import (
    _public_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02CarrierRule,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02DevCandidateIndex,
    predict_w02_dev_observation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    authorize_w02_morphology_source_routes,
)


def _index() -> W02DevCandidateIndex:
    rule = W02CarrierRule(
        "plain_text", "", "", "carrier_root", "language_content")
    return W02DevCandidateIndex(
        {"plain_text": ((rule, 1),)}, (), {}, (), 0, "1" * 64, 1)


def test_lzh_base_view_changes_only_language_and_not_original() -> None:
    _, _, observation, _ = _public_fixture()
    before = observation.to_dict()
    adapted = adapt_w02_observation_for_base_candidate(observation)

    assert observation.to_dict() == before
    assert observation.language == "lzh"
    assert adapted.language == "zh"
    assert adapted.to_dict() == {**before, "language": "zh"}


def test_lzh_base_prediction_is_metamorphic_to_zh_view() -> None:
    _, zh_observation, lzh_observation, _ = _public_fixture()
    baseline, baseline_operations = predict_w02_dev_observation(
        _index(), zh_observation)
    adapted, adapted_operations = predict_w02_dev_observation_language_family(
        _index(), lzh_observation)

    assert adapted.to_dict() == baseline.to_dict()
    assert adapted_operations == baseline_operations


def test_route_keeps_original_lzh_and_rejects_adapted_zh_identity() -> None:
    source, _, observation, capability = _public_fixture()
    routes = authorize_w02_morphology_source_routes(
        (source,), (capability,))
    adapted = adapt_w02_observation_for_base_candidate(observation)

    assert routes.permits(observation)
    assert not routes.permits(adapted)


def test_adapter_rejects_unregistered_language() -> None:
    _, _, observation, _ = _public_fixture()
    with pytest.raises(W02BaseLanguageFamilyAdapterError, match="scope"):
        adapt_w02_observation_for_base_candidate(
            replace(observation, language="en"))
