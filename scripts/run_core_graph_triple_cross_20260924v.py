"""Probe cross-identity three-object Core graph composition."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import scripts.run_core_graph_pair_heldout_20260924m as pair_probe
from scripts.select_core_graph_input_keys import select_core_graph_input_keys


DATABASE = pair_probe.DATABASE
COURSE = pair_probe.base.Path(
    r"K:/pure_integer_ai_work/current_runs/core-graph-heldout-20260920r5/"
    r"heldout_response_course.int.json")
MANIFEST = pair_probe.base.Path(
    r"K:/pure_integer_ai_work/current_runs/core-graph-heldout-20260920r5/"
    r"heldout_manifest.json")
ROOT = Path(os.environ.get(
    "PURE_INTEGER_CORE_TRIPLE_CROSS_RUN_ROOT",
    r"K:/pure_integer_ai_work/current_runs/core-graph-triple-cross-heldout-20260924v"))
FREE_DIALOGUE_COMPLETE = 0


def _write_json(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n").encode("ascii"))


def _trace(result: dict[str, object], surface: str) -> object:
    return pair_probe.base._integer_projection({
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


def _triples() -> tuple[tuple[tuple[int, ...], ...], ...]:
    selected = dict(select_core_graph_input_keys(DATABASE, per_kind=2))
    entities = selected[4]
    events = selected[16]
    concepts = selected[17]
    matched = {
        tuple(sorted((entities[1], events[0], concepts[1]))),
        tuple(sorted((entities[0], events[1], concepts[0]))),
    }
    triples = tuple(
        tuple(sorted((entity, event, concept)))
        for entity in entities
        for event in events
        for concept in concepts
        if tuple(sorted((entity, event, concept))) not in matched
    )
    if len(triples) != 6:
        raise RuntimeError("cross triple set is not the expected six combinations")
    return triples


def main() -> int:
    if ROOT.exists():
        raise ValueError(f"run root must be new: {ROOT}")
    ROOT.mkdir(parents=True)
    before = pair_probe.base._sha256(DATABASE)
    triples = _triples()
    rows: list[dict[str, object]] = []
    with pair_probe._CanonicalWithHeldout(DATABASE) as canonical:
        for ordinal, triple in enumerate(triples, 1):
            checkpoint = ROOT / f"triple-{ordinal:03d}.checkpoint.json"
            _write_json(checkpoint, {
                "format": "PURE_INTEGER_CORE_GRAPH_CROSS_TRIPLE_CHECKPOINT_V1",
                "ordinal": ordinal,
                "graph_input_keys": [list(item) for item in triple],
                "stage": "STARTED",
                "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
            })
            try:
                with pair_probe._GraphBridgeWithHeldout(
                        DATABASE,
                        memory_database=ROOT / "session.sqlite3",
                        session_id=92200 + ordinal) as bridge:
                    appended = bridge.memory.append_graph_input(
                        triple, speaker_kind=1)
                    observation = bridge.memory.intake.result_for_source(
                        appended.source).observation_ref
                    _write_json(checkpoint, {
                        "format": "PURE_INTEGER_CORE_GRAPH_CROSS_TRIPLE_CHECKPOINT_V1",
                        "ordinal": ordinal,
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
                            raise RuntimeError("cross triple response lacked delivery")
                    trace_path = ROOT / f"triple-{ordinal:03d}.int.json.gz"
                    trace_sha = pair_probe.base._write_trace(
                        _trace(result, surface), trace_path)
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
                    _write_json(checkpoint, {
                        "format": "PURE_INTEGER_CORE_GRAPH_CROSS_TRIPLE_CHECKPOINT_V1",
                        "ordinal": ordinal,
                        "graph_input_keys": [list(item) for item in triple],
                        "stage": "COMPLETE",
                        "termination": result.get("termination"),
                        "generation": row["generation"],
                        "core_consumed_count": row["core_consumed_count"],
                        "delivery": row["delivery"],
                        "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                    })
            except BaseException as exc:
                error_path = ROOT / f"triple-{ordinal:03d}.error.txt"
                error_path.write_text(
                    "".join(traceback.format_exception(exc)), encoding="utf-8")
                rows.append({
                    "ordinal": ordinal,
                    "graph_input_keys": [list(item) for item in triple],
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error_file": error_path.name,
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                })
                _write_json(checkpoint, {
                    "format": "PURE_INTEGER_CORE_GRAPH_CROSS_TRIPLE_CHECKPOINT_V1",
                    "ordinal": ordinal,
                    "graph_input_keys": [list(item) for item in triple],
                    "stage": "ERROR_CAPTURED",
                    "error_type": type(exc).__name__,
                    "error_file": error_path.name,
                    "free_dialogue_complete": FREE_DIALOGUE_COMPLETE,
                })
    after = pair_probe.base._sha256(DATABASE)
    successful = [row for row in rows if row.get("status") == "OK"]
    receipt = {
        "format": "PURE_INTEGER_CORE_GRAPH_CROSS_TRIPLE_HELDOUT_V1",
        "integer_payload": 1,
        "database_model_sha256_before": before,
        "database_model_sha256_after": after,
        "model_read_only": int(before == after),
        "cross_triple_count": len(rows),
        "cross_triple_success_count": len(successful),
        "cross_triple_error_count": len(rows) - len(successful),
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
    _write_json(ROOT / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0 if len(successful) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
