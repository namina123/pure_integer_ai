"""Audit the durable graph-only Core probe without rerunning the model."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


RUN = Path(r"K:/pure_integer_ai_work/current_runs/core-graph-only-heldout-20260924k")
OUT = Path(
    r"K:/pure_integer_ai_work/audits/core-graph-only-qualification-20260924k")
INVENTORY = Path(
    r"K:/pure_integer_ai_work/current_runs/core-graph-input-inventory-20260923e/core_graph_input_keys.int.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="ascii") as stream:
        return json.load(stream)


def main() -> int:
    receipt = json.loads((RUN / "receipt.json").read_text(encoding="ascii"))
    inventory = json.loads(INVENTORY.read_text(encoding="ascii"))
    keys = tuple(tuple(item[1][0]) for item in inventory[2])
    rows = []
    failures = []
    for ordinal, (key, expected) in enumerate(zip(keys, receipt["rows"]), 1):
        trace_path = RUN / expected["trace_file"]
        trace = _load(trace_path)
        trace_sha = _sha256(trace_path)
        state = trace.get("termination_state", {})
        core_keys = {tuple(item) for item in trace.get("core_graph_input_keys", ())}
        ok = (
            expected["ordinal"] == ordinal
            and tuple(expected["graph_input_key"]) == key
            and expected["trace_sha256"] == trace_sha
            and tuple(trace.get("active_spaces", ())) == (1, 2, 3)
            and key in core_keys
            and expected["core_consumed"] == 1
            and expected["all_consumed"] == 1
            and expected["status"] == "OK"
        )
        row = {
            "ordinal": ordinal,
            "graph_input_key": list(key),
            "status": "PASS" if ok else "FAIL",
            "active_spaces": trace.get("active_spaces"),
            "core_consumed": expected["core_consumed"],
            "all_consumed": expected["all_consumed"],
            "termination": trace.get("termination"),
            "generation": int(bool(trace.get("generation"))),
            "generation_ready": int(state.get("generation_ready", 0)),
            "required_slots_open": int(state.get("required_slots_open", 0)),
            "delivery": expected["delivery"],
            "trace_sha256": trace_sha,
            "trace_file": trace_path.name,
        }
        rows.append(row)
        if not ok:
            failures.append(ordinal)

    generated_ready = sum(row["generation_ready"] for row in rows)
    generated = sum(row["generation"] for row in rows)
    deliveries = sum(row["delivery"] for row in rows)
    status = (
        "PASS_GRAPH_ONLY_CORE_CONSUMPTION_PARTIAL_GENERATION"
        if not failures and generated == 2 and generated_ready == 2
        and deliveries == 2 and receipt.get("free_dialogue_complete") == 0
        else "FAIL"
    )
    report = {
        "format": "PURE_INTEGER_CORE_GRAPH_ONLY_QUALIFICATION_V1",
        "status": status,
        "run_root": str(RUN),
        "receipt_sha256": _sha256(RUN / "receipt.json"),
        "inventory_sha256": _sha256(INVENTORY),
        "query_count": len(rows),
        "three_graph_query_count": sum(
            tuple(row["active_spaces"]) == (1, 2, 3) for row in rows),
        "core_graph_input_consumed_count": sum(
            row["core_consumed"] for row in rows),
        "consumed_graph_input_count": sum(row["all_consumed"] for row in rows),
        "generation_count": generated,
        "generation_ready_count": generated_ready,
        "delivery_count": deliveries,
        "termination_codes": [row["termination"] for row in rows],
        "partial_generation_is_explicit": 1,
        "event_missing_binding_is_fail_closed": int(
            rows[1]["termination"] == 3 and rows[1]["generation"] == 0
            and rows[1]["required_slots_open"] == 1),
        "database_model_read_only": int(
            receipt.get("model_read_only") == 1
            and receipt.get("database_model_sha256_before")
            == receipt.get("database_model_sha256_after")),
        "source_text_rows_added": receipt.get("source_text_rows_added"),
        "database_copy_count": receipt.get("database_copy_count"),
        "forbidden_routes": {
            "source_body_answer_route": receipt.get("source_body_answer_route"),
            "successor_answer_route": receipt.get("successor_answer_route"),
            "character_nearest_route": receipt.get("character_nearest_route"),
            "language_vocabulary_route": receipt.get("language_vocabulary_route"),
            "opencc_route": receipt.get("opencc_route"),
        },
        "free_dialogue_complete": receipt.get("free_dialogue_complete"),
        "integer_trace": receipt.get("integer_trace"),
        "rows": rows,
        "failed_ordinals": failures,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "qualification.json").write_bytes(
        (json.dumps(report, ensure_ascii=True, sort_keys=True,
                    separators=(",", ":")) + "\n").encode("ascii"))
    print(json.dumps(report, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
