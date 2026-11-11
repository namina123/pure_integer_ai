from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def training_runner(monkeypatch):
    scripts_dir = str(Path(__file__).parents[1] / "scripts")
    monkeypatch.syspath_prepend(scripts_dir)
    preflight = importlib.import_module(
        "preflight_natural_conjunction_k32_cross_turn_ohe")
    for name in (
            "ROOT", "COURSE_ROOT", "OPERATOR_GRAPH", "CONJUNCTION_GRAPH",
            "COMPOSITION_GRAPH", "SOURCE", "LICENSE", "Q2"):
        monkeypatch.setattr(preflight, name, getattr(preflight, name))
    return importlib.import_module(
        "run_natural_conjunction_k32_cross_turn_ohe_training")


def test_summary_uses_reference_resolution_constants(training_runner):
    resolved = SimpleNamespace(status=training_runner.REFERENCE_RESOLVED)
    resolved.stable_key = lambda: (91, 3)
    bridge = SimpleNamespace(_reference_resolutions=(resolved,))
    result = {
        "response_plan": {"memory_refs": [1], "discourse_links": [2]},
        "generation": {"representations": [1]},
    }

    summary = training_runner._query_summary(
        "resolved", (10,), result, bridge, None)

    assert summary["reference_status_counts"] == {
        "not_applicable": 0, "open": 0, "resolved": 1, "conflict": 0,
    }
    assert summary["reference_applicable"] == 1
    assert summary["reference_resolved"] == 1
    assert summary["reference_conflict"] == 0
    assert summary["reference_response_plan_winners"] == 1


def test_summary_decodes_compact_adoption_refs_and_composition_generation(
        training_runner):
    bridge = SimpleNamespace(_reference_resolutions=())
    result = {
        "integer_key_table": [[91599, 7, 99], [91599, 6]],
        "evidence": [{
            "space": 1, "polarity": 3, "evidence_ref": 1,
        }],
        "bindings": [{"space": 1, "role_ref": 2}],
        "response_plan": {"slot_sequence": []},
        "generation": {
            "kind": 4,
            "claim_keys": [{"integer_key_ref": 3}],
            "surface_values": {"integer_key_ref": 4},
        },
    }

    summary = training_runner._query_summary(
        "composition", (10,), result, bridge, None)

    assert summary["target_adoption_marker"] == 1
    assert summary["target_adoption_binding"] == 1
    assert summary["generation"] == 1
