"""Probe each Core graph object in an isolated session with durable row checkpoints."""
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
    "PURE_INTEGER_CORE_GRAPH_RUN_ROOT",
    r"K:/pure_integer_ai_work/current_runs/core-graph-only-heldout-20260924j"))
FREE_DIALOGUE_COMPLETE = 0


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(
        (json.dumps(value, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")) + "\n").encode("ascii"))


def _row_trace(result: dict[str, object], surface: str) -> object:
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


def _row_summary(key: tuple[int, ...], result: dict[str, object],
                 surface: str, delivery: object) -> dict[str, object]:
    generated = bool(result.get("generation"))
    core_keys = {tuple(item) for item in result.get("core_graph_input_keys", ())}
    all_keys = {tuple(item) for item in (
        *result.get("core_graph_input_keys", ()),
        *result.get("generic_graph_input_keys", ()),
        *result.get("discourse_graph_input_keys", ()),
        *result.get("event_time_graph_input_keys", ()),
    )}
    return {
        "ordinal": 0,
        "graph_input_key": list(key),
        "status": "OK",
        "termination": result.get("termination"),
        "generation": int(generated),
        "generation_ready": int(result.get(
            "termination_state", {}).get("generation_ready", 0)),
        "required_slots_open": int(
            result.get("termination_state", {}).get("required_slots_open", 0)),
        "active_spaces": result.get("active_spaces"),
        "core_consumed": int(key in core_keys),
        "all_consumed": int(key in all_keys),
        "response_surface_value_count": len(surface),
        "delivery": int(delivery is not None),
        "trace_file": "",
        "trace_sha256": "",
    }


def main() -> int:
    if ROOT.exists():
        raise ValueError(f"run root must be new: {ROOT}")
    ROOT.mkdir(parents=True)
    before = base._sha256(DATABASE)
    keys = base._inventory_keys()
    rows: list[dict[str, object]] = []
    base.TrainedGenerationConnectorRuntime = _CanonicalWithHeldout
    base.TrainedGraphQueryBridge = _GraphBridgeWithHeldout
    with _CanonicalWithHeldout(DATABASE) as canonical:
        for ordinal, key in enumerate(keys, 1):
            checkpoint = ROOT / f"query-{ordinal:03d}.checkpoint.json"
            _write_json(checkpoint, {
                "format": "PURE_INTEGER_CORE_GRAPH_ONLY_HELDOUT_CHECKPOINT_V1",
                "ordinal": ordinal,
                "graph_input_key": list(key),
                "stage": "STARTED",
                "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
            })
            row: dict[str, object]
            try:
                # A fresh bridge/session makes a failed topology query unable to
                # contaminate the next object's O/H/E or QueryState.
                with _GraphBridgeWithHeldout(
                        DATABASE,
                        memory_database=ROOT / "session.sqlite3",
                        session_id=92130 + ordinal) as bridge:
                    appended = bridge.memory.append_graph_input(
                        (key,), speaker_kind=1)
                    observation = bridge.memory.intake.result_for_source(
                        appended.source).observation_ref
                    _write_json(checkpoint, {
                        "format": "PURE_INTEGER_CORE_GRAPH_ONLY_HELDOUT_CHECKPOINT_V1",
                        "ordinal": ordinal,
                        "graph_input_key": list(key),
                        "stage": "INTAKE_COMPLETE",
                        "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                    })
                    result = bridge.query(
                        None, max_depth=64, graph_object_keys=(key,),
                        input_observation_ref=observation)
                    surface = result.get("response_surface", "")
                    delivery = None
                    if result.get("generation") and surface:
                        delivery = bridge.acknowledge_response_delivery(surface)
                        if delivery is None:
                            raise RuntimeError("generated graph response lacked delivery")
                    trace = _row_trace(result, surface)
                    trace_path = ROOT / f"query-{ordinal:03d}.int.json.gz"
                    trace_sha = base._write_trace(trace, trace_path)
                    row = _row_summary(key, result, surface, delivery)
                    row.update({
                        "ordinal": ordinal,
                        "trace_file": trace_path.name,
                        "trace_sha256": trace_sha,
                    })
                    _write_json(checkpoint, {
                        "format": "PURE_INTEGER_CORE_GRAPH_ONLY_HELDOUT_CHECKPOINT_V1",
                        "ordinal": ordinal,
                        "graph_input_key": list(key),
                        "stage": "COMPLETE",
                        "termination": result.get("termination"),
                        "generation": int(bool(result.get("generation"))),
                        "delivery": int(delivery is not None),
                        "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                    })
            except BaseException as exc:  # preserve row evidence and continue
                error_path = ROOT / f"query-{ordinal:03d}.error.txt"
                error_path.write_text(
                    "".join(traceback.format_exception(exc)), encoding="utf-8")
                row = {
                    "ordinal": ordinal,
                    "graph_input_key": list(key),
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error_file": error_path.name,
                    "trace_file": "",
                    "trace_sha256": "",
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                }
                _write_json(checkpoint, {
                    "format": "PURE_INTEGER_CORE_GRAPH_ONLY_HELDOUT_CHECKPOINT_V1",
                    "ordinal": ordinal,
                    "graph_input_key": list(key),
                    "stage": "ERROR_CAPTURED",
                    "error_type": type(exc).__name__,
                    "error_file": error_path.name,
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                })
            rows.append(row)

    after = base._sha256(DATABASE)
    successful = [row for row in rows if row.get("status") == "OK"]
    receipt = {
        "format": "PURE_INTEGER_CORE_GRAPH_ONLY_HELDOUT_CONSUMPTION_V2",
        "integer_payload": 1,
        "database_model_sha256_before": before,
        "database_model_sha256_after": after,
        "model_read_only": int(before == after),
        "query_count": len(rows),
        "query_success_count": len(successful),
        "query_error_count": len(rows) - len(successful),
        "three_graph_query_count": sum(
            set(row.get("active_spaces", ())) == {1, 2, 3}
            for row in successful),
        "core_graph_input_consumed_count": sum(
            row.get("core_consumed", 0) for row in successful),
        "consumed_graph_input_count": sum(
            row.get("all_consumed", 0) for row in successful),
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
