"""Probe pairwise Core graph-only inputs through the heldout overlay."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import scripts.run_core_graph_only_consumption_20260924g as base
from scripts.run_core_graph_only_heldout_20260924i import (
    _CanonicalWithHeldout,
    _GraphBridgeWithHeldout,
)


DATABASE = base.DATABASE
INVENTORY = base.INVENTORY
ROOT = Path(os.environ.get(
    "PURE_INTEGER_CORE_PAIR_RUN_ROOT",
    r"K:/pure_integer_ai_work/current_runs/core-graph-pair-heldout-20260924m"))
FREE_DIALOGUE_COMPLETE = 0


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(
        (json.dumps(value, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")) + "\n").encode("ascii"))


def _inventory_by_kind() -> dict[int, tuple[int, ...]]:
    payload = json.loads(INVENTORY.read_text(encoding="ascii"))
    if payload[:2] != [91580, 1] or len(payload[2]) != 3:
        raise ValueError("inventory schema is not closed")
    result: dict[int, tuple[int, ...]] = {}
    for kind, values in payload[2]:
        if len(values) != 1:
            raise ValueError("inventory must contain one key per object kind")
        key = tuple(values[0])
        if not key or any(type(item) is not int or item < 0 for item in key):
            raise ValueError("inventory key is not a non-negative integer tuple")
        result[int(kind)] = key
    if set(result) != {4, 16, 17}:
        raise ValueError("inventory does not cover Entity/Event/Concept")
    return result


def _trace(result: dict[str, object], surface: str) -> object:
    return base._integer_projection({
        "termination": result.get("termination"),
        "termination_state": result.get("termination_state"),
        "active_spaces": result.get("active_spaces"),
        "roots": result.get("roots"),
        "anchors": result.get("anchors"),
        "frontier_trace": result.get("frontier_trace"),
        "bindings": result.get("bindings"),
        "evidence": result.get("evidence"),
        "visited": result.get("visited"),
        "expanded_edges": result.get("expanded_edges"),
        "generation": result.get("generation"),
        "response_plan": result.get("response_plan"),
        "response_surface_values": list(map(ord, surface)),
        "core_graph_input_keys": result.get("core_graph_input_keys"),
        "generic_graph_input_keys": result.get("generic_graph_input_keys"),
        "discourse_graph_input_keys": result.get("discourse_graph_input_keys"),
        "event_time_graph_input_keys": result.get("event_time_graph_input_keys"),
        "generic_response_feature_mask": result.get(
            "generic_response_feature_mask"),
    })


def main() -> int:
    if ROOT.exists():
        raise ValueError(f"run root must be new: {ROOT}")
    ROOT.mkdir(parents=True)
    before = base._sha256(DATABASE)
    keys = _inventory_by_kind()
    pairs = (
        tuple(sorted((keys[16], keys[4]))),
        tuple(sorted((keys[16], keys[17]))),
        tuple(sorted((keys[17], keys[4]))),
    )
    rows: list[dict[str, object]] = []
    with _CanonicalWithHeldout(DATABASE) as canonical:
        for ordinal, pair in enumerate(pairs, 1):
            checkpoint = ROOT / f"pair-{ordinal:03d}.checkpoint.json"
            _write_json(checkpoint, {
                "format": "PURE_INTEGER_CORE_GRAPH_PAIR_CHECKPOINT_V1",
                "ordinal": ordinal,
                "graph_input_keys": [list(item) for item in pair],
                "stage": "STARTED",
                "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
            })
            try:
                with _GraphBridgeWithHeldout(
                        DATABASE,
                        memory_database=ROOT / "session.sqlite3",
                        session_id=92140 + ordinal) as bridge:
                    appended = bridge.memory.append_graph_input(
                        pair, speaker_kind=1)
                    observation = bridge.memory.intake.result_for_source(
                        appended.source).observation_ref
                    _write_json(checkpoint, {
                        "format": "PURE_INTEGER_CORE_GRAPH_PAIR_CHECKPOINT_V1",
                        "ordinal": ordinal,
                        "graph_input_keys": [list(item) for item in pair],
                        "stage": "INTAKE_COMPLETE",
                        "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                    })
                    result = bridge.query(
                        None, max_depth=64, graph_object_keys=pair,
                        input_observation_ref=observation)
                    surface = result.get("response_surface", "")
                    delivery = None
                    if result.get("generation") and surface:
                        delivery = bridge.acknowledge_response_delivery(surface)
                        if delivery is None:
                            raise RuntimeError("pair response lacked delivery")
                    trace_path = ROOT / f"pair-{ordinal:03d}.int.json.gz"
                    trace_sha = base._write_trace(_trace(result, surface), trace_path)
                    core_keys = {tuple(item) for item in result.get(
                        "core_graph_input_keys", ())}
                    all_keys = {tuple(item) for item in (
                        *result.get("core_graph_input_keys", ()),
                        *result.get("generic_graph_input_keys", ()),
                        *result.get("discourse_graph_input_keys", ()),
                        *result.get("event_time_graph_input_keys", ()),
                    )}
                    state = result.get("termination_state", {})
                    row = {
                        "ordinal": ordinal,
                        "graph_input_keys": [list(item) for item in pair],
                        "status": "OK",
                        "active_spaces": result.get("active_spaces"),
                        "termination": result.get("termination"),
                        "generation": int(bool(result.get("generation"))),
                        "generation_ready": int(state.get("generation_ready", 0)),
                        "required_slots_open": int(
                            state.get("required_slots_open", 0)),
                        "core_consumed_count": sum(item in core_keys for item in pair),
                        "all_consumed_count": sum(item in all_keys for item in pair),
                        "response_surface_value_count": len(surface),
                        "delivery": int(delivery is not None),
                        "trace_file": trace_path.name,
                        "trace_sha256": trace_sha,
                    }
                    rows.append(row)
                    _write_json(checkpoint, {
                        "format": "PURE_INTEGER_CORE_GRAPH_PAIR_CHECKPOINT_V1",
                        "ordinal": ordinal,
                        "graph_input_keys": [list(item) for item in pair],
                        "stage": "COMPLETE",
                        "termination": result.get("termination"),
                        "generation": row["generation"],
                        "core_consumed_count": row["core_consumed_count"],
                        "delivery": row["delivery"],
                        "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                    })
            except BaseException as exc:
                error_path = ROOT / f"pair-{ordinal:03d}.error.txt"
                error_path.write_text(
                    "".join(traceback.format_exception(exc)), encoding="utf-8")
                rows.append({
                    "ordinal": ordinal,
                    "graph_input_keys": [list(item) for item in pair],
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error_file": error_path.name,
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                })
                _write_json(checkpoint, {
                    "format": "PURE_INTEGER_CORE_GRAPH_PAIR_CHECKPOINT_V1",
                    "ordinal": ordinal,
                    "graph_input_keys": [list(item) for item in pair],
                    "stage": "ERROR_CAPTURED",
                    "error_type": type(exc).__name__,
                    "error_file": error_path.name,
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                })
    after = base._sha256(DATABASE)
    successful = [row for row in rows if row.get("status") == "OK"]
    receipt = {
        "format": "PURE_INTEGER_CORE_GRAPH_PAIR_HELDOUT_V1",
        "integer_payload": 1,
        "database_model_sha256_before": before,
        "database_model_sha256_after": after,
        "model_read_only": int(before == after),
        "pair_count": len(rows),
        "pair_success_count": len(successful),
        "pair_error_count": len(rows) - len(successful),
        "three_graph_pair_query_count": sum(
            set(row.get("active_spaces", ())) == {1, 2, 3}
            for row in successful),
        "both_core_consumed_pair_count": sum(
            row.get("core_consumed_count", 0) == 2 for row in successful),
        "both_all_consumed_pair_count": sum(
            row.get("all_consumed_count", 0) == 2 for row in successful),
        "generation_count": sum(row.get("generation", 0) for row in successful),
        "delivery_count": sum(row.get("delivery", 0) for row in successful),
        "termination_codes": [row.get("termination") for row in successful],
        "rows": rows,
        "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
        "source_text_rows_added": 0,
        "database_copy_count": 0,
        "source_body_answer_route": 0,
        "successor_answer_route": 0,
        "character_nearest_route": 0,
        "language_vocabulary_route": 0,
        "opencc_route": 0,
        "integer_trace": 1,
    }
    _write_json(ROOT / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0 if not rows or all(row.get("status") == "OK" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
