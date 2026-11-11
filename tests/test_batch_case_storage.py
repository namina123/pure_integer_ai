from __future__ import annotations

from copy import deepcopy
import json

import pytest

from pure_integer_ai.experiments.batch_case_provenance import (
    capture_factual_case_provenance,
)
from pure_integer_ai.experiments.batch_case_storage import (
    CaseArtifactStorageError,
    build_compact_case_provenance,
    compact_case_provenance_bytes,
    load_case_provenance,
    restore_compact_case_provenance,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


def _captured() -> tuple[dict, bytes]:
    """建立满足原V1消费者的最小整数trace与同次delivery证据。"""
    query = {
        "format": "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V4",
        "schema_version": 4,
        "integer_key_encoding": "INTEGER_KEY_TABLE_V1",
        "integer_key_table": [[100], [65, 65306, 88], [88], [7], [50], [60], [90]],
        "query_ref": 1,
        "input_structure": {
            "source_ref": {"integer_key_ref": 1},
            "token_values": {"integer_key_ref": 2},
        },
        "state_key": {
            "query_ref": 1, "active_spaces": [1, 2, 3], "evidence": [7],
            "scalars": [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        },
        "active_spaces": [1, 2, 3], "termination": 1,
        "hops": [
            {"space": 2, "proposition_ref": 4, "predicate_ref": 5, "source_hash": 17},
            {"space": 3, "proposition_ref": 0, "predicate_ref": 6, "source_hash": 18},
        ],
        "memory_route_refs": [5], "expanded_edge_refs": [],
        "operation_keys": [[22]], "operation_contracts": [[[22], [91672, 3]]],
        "response_plan": {
            "response_act": [33], "stable_key": [44], "claim_refs": [[7]],
            "evidence_refs": [[90]],
            "claim_evidence_refs": [{"claim": [7], "evidence": [[90]]}],
        },
        "response_surface": "X",
        "generation": {"surface_values": {"integer_key_ref": 3}},
        "frontier_trace": {"format": "INTEGER_FRONTIER_TRACE_V1", "steps": []},
    }
    delivery = {
        "proof_ref": [101], "output_source": [102], "episode_event_hash": 19,
        "use_event_hashes": [20], "claim_count": 1, "evidence_ref_count": 1,
        "delivered": 1,
    }
    trace_bytes = canonical_json_bytes(query) + b"\n"
    captured = capture_factual_case_provenance(
        query_result=query, query_values=(65, 65306, 88), expected_claim=(7,),
        expected_operation=(22,), input_observation=(103,), input_source=(104,),
        delivery_summary=delivery)
    return captured, trace_bytes


def test_compact_reference_restores_exact_legacy_audit_structure(tmp_path) -> None:
    captured, trace_bytes = _captured()
    (tmp_path / "trace.int.json").write_bytes(trace_bytes)
    raw = compact_case_provenance_bytes(
        captured, trace_path="trace.int.json", trace_bytes=trace_bytes)
    compact = json.loads(raw)
    assert "trace" not in compact["provenance"]
    assert restore_compact_case_provenance(compact, root=tmp_path) == captured
    saved = tmp_path / "case_provenance.json"
    saved.write_bytes(raw)
    assert load_case_provenance(saved, root=tmp_path) == captured
    full = tmp_path / "full-provenance.json"
    full.write_bytes(canonical_json_bytes(captured) + b"\n")
    assert load_case_provenance(full) == captured


@pytest.mark.parametrize("path", [
    "../trace.int.json", "/trace.int.json", "x\\trace.int.json", "D:/trace.int.json",
])
def test_compact_reference_rejects_nonlocal_or_nonportable_path(path: str) -> None:
    captured, trace_bytes = _captured()
    with pytest.raises(CaseArtifactStorageError, match="POSIX relative path|escapes"):
        build_compact_case_provenance(captured, trace_path=path, trace_bytes=trace_bytes)


@pytest.mark.parametrize("field,new_value", [
    ("byte_count", 1),
    ("sha256", "0" * 64),
    ("integer_projection_sha256", "f" * 64),
    ("path", "../trace.int.json"),
])
def test_restore_rejects_changed_trace_reference(field: str, new_value: object,
                                                tmp_path) -> None:
    captured, trace_bytes = _captured()
    (tmp_path / "trace.int.json").write_bytes(trace_bytes)
    compact = deepcopy(build_compact_case_provenance(
        captured, trace_path="trace.int.json", trace_bytes=trace_bytes))
    compact["trace_ref"][field] = new_value
    with pytest.raises(CaseArtifactStorageError):
        restore_compact_case_provenance(compact, root=tmp_path)


def test_restore_rejects_trace_symlink_escape(tmp_path) -> None:
    captured, trace_bytes = _captured()
    outside = tmp_path.parent / "outside-trace.json"
    outside.write_bytes(trace_bytes)
    try:
        (tmp_path / "linked-trace.json").symlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip("symlinks are unavailable in this Windows test environment: " + str(error))
    compact = build_compact_case_provenance(
        captured, trace_path="linked-trace.json", trace_bytes=trace_bytes)
    with pytest.raises(CaseArtifactStorageError, match="outside"):
        restore_compact_case_provenance(compact, root=tmp_path)


def test_compact_reference_rejects_mismatched_projection_and_boolean_completion() -> None:
    captured, trace_bytes = _captured()
    changed = deepcopy(captured)
    changed["trace"]["operation_keys"] = [[23]]
    with pytest.raises(CaseArtifactStorageError, match="projection"):
        build_compact_case_provenance(
            changed, trace_path="trace.int.json", trace_bytes=trace_bytes)


def test_compact_metadata_is_an_independent_snapshot() -> None:
    captured, trace_bytes = _captured()
    compact = build_compact_case_provenance(
        captured, trace_path="trace.int.json", trace_bytes=trace_bytes)
    captured["delivery_summary"]["proof_ref"][0] = 999
    assert compact["provenance"]["delivery_summary"]["proof_ref"] == [101]


@pytest.mark.parametrize("location", ["outer", "inner"])
def test_restore_rejects_boolean_completion_flags(location, tmp_path) -> None:
    captured, trace_bytes = _captured()
    compact = build_compact_case_provenance(
        captured, trace_path="trace.int.json", trace_bytes=trace_bytes)
    record = compact if location == "outer" else compact["provenance"]
    record["free_dialogue_complete"] = False
    with pytest.raises(CaseArtifactStorageError):
        restore_compact_case_provenance(compact, root=tmp_path)


def test_load_rejects_provenance_outside_the_declared_root(tmp_path) -> None:
    captured, trace_bytes = _captured()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    saved = tmp_path / "provenance.json"
    saved.write_bytes(compact_case_provenance_bytes(
        captured, trace_path="trace.int.json", trace_bytes=trace_bytes))
    with pytest.raises(CaseArtifactStorageError, match="outside"):
        load_case_provenance(saved, root=isolated)
    changed = deepcopy(captured)
    changed["free_dialogue_complete"] = False
    with pytest.raises(CaseArtifactStorageError, match="free_dialogue_complete"):
        build_compact_case_provenance(
            changed, trace_path="trace.int.json", trace_bytes=trace_bytes)
