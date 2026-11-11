"""Audit the three completed original calls; never regenerate their answers."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.experiments.natural_relation_preflight import write_preflight_json
from pure_integer_ai.experiments.operation_program_call_audit import (
    OPERATION_EXECUTION_ROLE, audit_saved_operation_counterfactuals, restore_saved_operation_question,
    verify_removed_operation_trace,
)
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_question_graph import read_relation_question_graph


@pytest.fixture(scope="module")
def saved():
    root = os.environ.get("PIA_SAVED_OPERATION_SOURCE")
    core = os.environ.get("PIA_SAVED_COMPOSITION_SOURCE")
    if root is None or core is None:
        pytest.skip("requires explicitly supplied original operation and Core artifacts")
    return Path(root).resolve(strict=True), Path(core).resolve(strict=True)


def test_original_program_counterfactuals_restore_physical_core_ohe_plan_delivery(saved):
    source, core = saved
    result = audit_saved_operation_counterfactuals(source, core)
    assert result["status"] == "PASS_DEVELOPMENT_PHYSICAL_PROGRAM_PROJECTION_CORE_OHE_PLAN_DELIVERY_COUNTERFACTUAL"
    assert result["source_write_count"] == result["query_reexecution_count"] == result["generation_reexecution_count"] == 0
    assert result["qualified_semantic_operation_families"] == result["free_dialogue_complete"] == 0
    assert result["held_out_novelty"] == "UNKNOWN_TRAIN_QUESTION"
    assert [case["termination"] for case in result["cases"]] == [1, 1, 3]
    assert result["cases"][0]["response_surface"] != result["cases"][1]["response_surface"]
    assert not result["cases"][2]["response_surface"]
    target = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    if target:
        write_preflight_json(Path(target), "operation_counterfactual_cold_audit.json", result)


@pytest.mark.parametrize("field", ["relation_candidates", "semantic_candidates", "relation_question_bindings", "token_values"])
def test_selected_saved_question_rejects_missing_relation_roles_or_input(saved, field):
    source, core_source = saved
    value = json.loads((source / "cases/001/trace.int.json").read_bytes())
    report = json.loads((source / "cases/001/call_receipt.json").read_bytes())
    changed = deepcopy(value)
    changed["input_structure"][field] = []
    frames, _ = read_relation_question_graph(core_source / "questions.sqlite3")
    with TrainedRelationGraphRuntime(core_source / "core.sqlite3") as core:
        surfaces = core.active_surface_facts()
        projector = TrainedInputStructureProjector(RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(core),
            open_role_frames=restore_open_role_frames(core.active_surface_frames(), surfaces))
        with pytest.raises(ValueError, match="saved input differs"):
            restore_saved_operation_question(changed, report, projector, frames)


@pytest.mark.parametrize("field", ["evidence_closed", "generation_ready", "response_surface"])
def test_removed_program_cannot_retain_factual_output(saved, field):
    source, _ = saved
    value = json.loads((source / "cases/003/trace.int.json").read_bytes())
    report = json.loads((source / "cases/003/call_receipt.json").read_bytes())
    if field == "response_surface":
        value[field] = "unauthorized"
    else:
        value["state_key"]["scalars"][7 if field == "evidence_closed" else 15] = 1
    with pytest.raises(ValueError, match="removed operation acquired"):
        verify_removed_operation_trace(value, report)


@pytest.mark.parametrize("field", ["bindings", "evidence"])
def test_removed_program_rejects_transplanted_execution_authority(saved, field):
    source, _ = saved
    value = json.loads((source / "cases/003/trace.int.json").read_bytes())
    report = json.loads((source / "cases/003/call_receipt.json").read_bytes())
    member = (BindingEntry(OPERATION_EXECUTION_ROLE, (1,)) if field == "bindings"
              else EvidenceEntry((1,), (2,), polarity=3, evidence_key=(92230, 1)))
    value["integer_key_table"].append(list(member.stable_key()))
    value["state_key"][field].append(len(value["integer_key_table"]))
    with pytest.raises(ValueError, match="removed operation retained"):
        verify_removed_operation_trace(value, report)
