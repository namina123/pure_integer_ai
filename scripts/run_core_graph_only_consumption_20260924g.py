"""Consume three real Core object identities through graph-only QueryState input."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)


DATABASE = Path(
    r"K:/pure_integer_ai_work/current_runs/"
    r"discourse-relation-response-graph-20260917b/training.sqlite3")
INVENTORY = Path(
    r"K:/pure_integer_ai_work/current_runs/"
    r"core-graph-input-inventory-20260923e/core_graph_input_keys.int.json")
ROOT = Path(
    r"K:/pure_integer_ai_work/current_runs/"
    r"core-graph-only-consumption-20260924g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_projection(value: object) -> object:
    if type(value) is int:
        return value
    if isinstance(value, (list, tuple)):
        return [_integer_projection(item) for item in value
                if item is not None and not isinstance(item, str)]
    if isinstance(value, dict):
        return {str(key): _integer_projection(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if item is not None and not isinstance(item, str)}
    return None


def _write_trace(value: object, path: Path) -> str:
    payload = (json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("ascii")
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_keys() -> tuple[tuple[int, ...], ...]:
    payload = json.loads(INVENTORY.read_text(encoding="ascii"))
    if payload[:2] != [91580, 1] or len(payload[2]) != 3:
        raise ValueError("inventory schema is not closed")
    keys = []
    for kind, values in payload[2]:
        if len(values) != 1:
            raise ValueError("inventory must contain one key per object kind")
        key = tuple(values[0])
        if not key or any(type(item) is not int or item < 0 for item in key):
            raise ValueError("inventory key is not a non-negative integer tuple")
        keys.append(key)
    return tuple(keys)


def main() -> int:
    if ROOT.exists():
        raise ValueError(f"run root must be new: {ROOT}")
    ROOT.mkdir(parents=True)
    before = _sha256(DATABASE)
    keys = _inventory_keys()
    receipt_rows = []
    with TrainedGenerationConnectorRuntime(DATABASE) as canonical:
        with TrainedGraphQueryBridge(
                DATABASE, memory_database=ROOT / "session.sqlite3",
                session_id=92120) as bridge:
            for ordinal, key in enumerate(keys, 1):
                appended = bridge.memory.append_graph_input((key,), speaker_kind=1)
                observation = bridge.memory.intake.result_for_source(
                    appended.source).observation_ref
                result = bridge.query(
                    None, max_depth=64, graph_object_keys=(key,),
                    input_observation_ref=observation)
                generated = bool(result.get("generation"))
                surface = result.get("response_surface", "")
                delivery = None
                if generated and surface:
                    delivery = bridge.acknowledge_response_delivery(surface)
                    if delivery is None:
                        raise RuntimeError("graph-only generation lacked delivery")
                trace = _integer_projection({
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
                trace_sha = _write_trace(trace, ROOT / f"query-{ordinal:03d}.int.json.gz")
                receipt_rows.append({
                    "ordinal": ordinal,
                    "graph_input_key": list(key),
                    "termination": result.get("termination"),
                    "generation": int(generated),
                    "generation_ready": int(result.get("generation_ready", 0)),
                    "active_spaces": result.get("active_spaces"),
                    "core_consumed": int(key in set(result.get("core_graph_input_keys", ()))),
                    "all_consumed": int(key in set(
                        (*result.get("core_graph_input_keys", ()),
                         *result.get("generic_graph_input_keys", ()),
                         *result.get("discourse_graph_input_keys", ()),
                         *result.get("event_time_graph_input_keys", ())))),
                    "response_surface_value_count": len(surface),
                    "delivery": int(delivery is not None),
                    "trace_sha256": trace_sha,
                })
    after = _sha256(DATABASE)
    receipt = {
        "format": "PURE_INTEGER_CORE_GRAPH_ONLY_CONSUMPTION_V1",
        "integer_payload": 1,
        "database_model_sha256_before": before,
        "database_model_sha256_after": after,
        "model_read_only": int(before == after),
        "query_count": len(receipt_rows),
        "three_graph_query_count": sum(
            set(row["active_spaces"]) == {1, 2, 3} for row in receipt_rows),
        "graph_input_count": len(receipt_rows),
        "core_graph_input_consumed_count": sum(row["core_consumed"] for row in receipt_rows),
        "consumed_graph_input_count": sum(row["all_consumed"] for row in receipt_rows),
        "generation_count": sum(row["generation"] for row in receipt_rows),
        "delivery_count": sum(row["delivery"] for row in receipt_rows),
        "termination_codes": [row["termination"] for row in receipt_rows],
        "rows": receipt_rows,
        "free_dialogue_complete": 0,
        "source_text_rows_added": 0,
        "database_copy_count": 0,
        "source_body_answer_route": 0,
        "successor_answer_route": 0,
        "character_nearest_route": 0,
        "language_vocabulary_route": 0,
        "opencc_route": 0,
        "integer_trace": 1,
    }
    (ROOT / "receipt.json").write_bytes(
        (json.dumps(receipt, ensure_ascii=True, sort_keys=True,
                    separators=(",", ":")) + "\n").encode("ascii"))
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
