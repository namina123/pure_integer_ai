"""Probe a three-object Core graph-only input through the heldout overlay."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import scripts.run_core_graph_pair_heldout_20260924m as pair_probe


DATABASE = pair_probe.DATABASE
INVENTORY = pair_probe.INVENTORY
ROOT = Path(os.environ.get(
    "PURE_INTEGER_CORE_TRIPLE_RUN_ROOT",
    r"K:/pure_integer_ai_work/current_runs/core-graph-triple-heldout-20260924p"))
FREE_DIALOGUE_COMPLETE = 0


def main() -> int:
    if ROOT.exists():
        raise ValueError(f"run root must be new: {ROOT}")
    ROOT.mkdir(parents=True)
    before = pair_probe.base._sha256(DATABASE)
    keys = pair_probe._inventory_by_kind()
    variant = os.environ.get("PURE_INTEGER_CORE_TRIPLE_VARIANT", "primary")
    if variant == "primary":
        triple = tuple(sorted((keys[4], keys[16], keys[17])))
    elif variant == "alternate":
        # The alternate keys are selected deterministically from the same
        # trained Core graph, not synthesized or copied from a surface.
        from scripts.select_core_graph_input_keys import select_core_graph_input_keys
        selected = dict(select_core_graph_input_keys(DATABASE, per_kind=2))
        triple = tuple(sorted((selected[4][0], selected[16][1], selected[17][0])))
    else:
        raise ValueError(f"unknown triple variant: {variant}")
    checkpoint = ROOT / "triple-001.checkpoint.json"
    pair_probe._write_json(checkpoint, {
        "format": "PURE_INTEGER_CORE_GRAPH_TRIPLE_CHECKPOINT_V1",
        "graph_input_keys": [list(item) for item in triple],
        "stage": "STARTED",
        "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
    })
    rows: list[dict[str, object]] = []
    try:
        with pair_probe._CanonicalWithHeldout(DATABASE) as canonical:
            with pair_probe._GraphBridgeWithHeldout(
                    DATABASE,
                    memory_database=ROOT / "session.sqlite3",
                    session_id=92150) as bridge:
                appended = bridge.memory.append_graph_input(
                    triple, speaker_kind=1)
                observation = bridge.memory.intake.result_for_source(
                    appended.source).observation_ref
                pair_probe._write_json(checkpoint, {
                    "format": "PURE_INTEGER_CORE_GRAPH_TRIPLE_CHECKPOINT_V1",
                    "graph_input_keys": [list(item) for item in triple],
                    "stage": "INTAKE_COMPLETE",
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                })
                result = bridge.query(
                    None, max_depth=64, graph_object_keys=triple,
                    input_observation_ref=observation)
                surface = result.get("response_surface", "")
                delivery = None
                if result.get("generation") and surface:
                    delivery = bridge.acknowledge_response_delivery(surface)
                    if delivery is None:
                        raise RuntimeError("triple response lacked delivery")
                trace_path = ROOT / "triple-001.int.json.gz"
                trace_sha = pair_probe.base._write_trace(
                    pair_probe._trace(result, surface), trace_path)
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
                    "ordinal": 1,
                    "graph_input_keys": [list(item) for item in triple],
                    "status": "OK",
                    "active_spaces": result.get("active_spaces"),
                    "termination": result.get("termination"),
                    "generation": int(bool(result.get("generation"))),
                    "generation_ready": int(state.get("generation_ready", 0)),
                    "required_slots_open": int(
                        state.get("required_slots_open", 0)),
                    "core_consumed_count": sum(item in core_keys for item in triple),
                    "all_consumed_count": sum(item in all_keys for item in triple),
                    "response_surface_value_count": len(surface),
                    "delivery": int(delivery is not None),
                    "trace_file": trace_path.name,
                    "trace_sha256": trace_sha,
                }
                rows.append(row)
                pair_probe._write_json(checkpoint, {
                    "format": "PURE_INTEGER_CORE_GRAPH_TRIPLE_CHECKPOINT_V1",
                    "graph_input_keys": [list(item) for item in triple],
                    "stage": "COMPLETE",
                    "termination": result.get("termination"),
                    "generation": row["generation"],
                    "core_consumed_count": row["core_consumed_count"],
                    "delivery": row["delivery"],
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                })
    except BaseException as exc:
        error_path = ROOT / "triple-001.error.txt"
        error_path.write_text(
            "".join(traceback.format_exception(exc)), encoding="utf-8")
        rows.append({
            "ordinal": 1,
            "graph_input_keys": [list(item) for item in triple],
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error_file": error_path.name,
            "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
        })
        pair_probe._write_json(checkpoint, {
            "format": "PURE_INTEGER_CORE_GRAPH_TRIPLE_CHECKPOINT_V1",
            "graph_input_keys": [list(item) for item in triple],
            "stage": "ERROR_CAPTURED",
            "error_type": type(exc).__name__,
            "error_file": error_path.name,
            "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
        })
    after = pair_probe.base._sha256(DATABASE)
    successful = [row for row in rows if row.get("status") == "OK"]
    receipt = {
        "format": "PURE_INTEGER_CORE_GRAPH_TRIPLE_HELDOUT_V1",
        "integer_payload": 1,
        "database_model_sha256_before": before,
        "database_model_sha256_after": after,
        "model_read_only": int(before == after),
        "triple_count": len(rows),
        "triple_success_count": len(successful),
        "triple_error_count": len(rows) - len(successful),
        "three_graph_triple_query_count": sum(
            set(row.get("active_spaces", ())) == {1, 2, 3}
            for row in successful),
        "all_core_consumed_triple_count": sum(
            row.get("core_consumed_count", 0) == 3 for row in successful),
        "all_consumed_triple_count": sum(
            row.get("all_consumed_count", 0) == 3 for row in successful),
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
    pair_probe._write_json(ROOT / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0 if not rows or all(row.get("status") == "OK" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
