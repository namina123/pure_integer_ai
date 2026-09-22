"""Audit pairwise Core graph composition from durable integer traces."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


RUN = Path(r"K:/pure_integer_ai_work/current_runs/core-graph-pair-heldout-20260924o")
OUT = Path(
    r"K:/pure_integer_ai_work/audits/core-graph-pair-qualification-20260924o")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="ascii") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("pair trace must be an integer-projected object")
    return value


def _response_plan_refs(plan: object) -> set[tuple[int, ...]]:
    if not isinstance(plan, dict):
        return set()
    refs: set[tuple[int, ...]] = set()
    for field in ("claim_refs", "event_refs", "discourse_links", "evidence_refs"):
        value = plan.get(field, ())
        if isinstance(value, list):
            refs.update(tuple(item) for item in value if isinstance(item, list))
    carrier = plan.get("carrier_ref", ())
    if isinstance(carrier, list) and carrier:
        refs.add(tuple(carrier))
    slots = plan.get("slot_sequence", ())
    if isinstance(slots, list):
        for slot in slots:
            if isinstance(slot, dict) and isinstance(slot.get("filler"), list):
                refs.add(tuple(slot["filler"]))
    return refs


def main() -> int:
    receipt_path = RUN / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    rows = []
    failures = []
    for expected in receipt.get("rows", ()):
        ordinal = int(expected["ordinal"])
        pair = {tuple(item) for item in expected["graph_input_keys"]}
        trace_path = RUN / expected["trace_file"]
        trace = _trace(trace_path)
        state = trace.get("termination_state", {})
        trace_core = {tuple(item) for item in trace.get(
            "core_graph_input_keys", ())}
        plan_refs = _response_plan_refs(trace.get("response_plan"))
        conditions = {
            "trace_hash": expected.get("trace_sha256") == _sha256(trace_path),
            "active_spaces": trace.get("active_spaces") == [1, 2, 3],
            "both_core_consumed": pair <= trace_core,
            "both_plan_referenced": pair <= plan_refs,
            "termination_closed": trace.get("termination") == 8,
            "generation_ready": state.get("generation_ready") == 1,
            "generation_present": isinstance(trace.get("generation"), dict),
            "delivery": expected.get("delivery") == 1,
            "status_ok": expected.get("status") == "OK",
        }
        ok = all(conditions.values())
        row = {
            "ordinal": ordinal,
            "graph_input_keys": [list(item) for item in expected[
                "graph_input_keys"]],
            "status": "PASS" if ok else "FAIL",
            "active_spaces": trace.get("active_spaces"),
            "core_consumed_count": expected.get("core_consumed_count"),
            "all_consumed_count": expected.get("all_consumed_count"),
            "plan_reference_count": sum(item in plan_refs for item in pair),
            "termination": trace.get("termination"),
            "generation_ready": int(state.get("generation_ready", 0)),
            "delivery": expected.get("delivery"),
            "trace_file": trace_path.name,
            "trace_sha256": _sha256(trace_path),
            "conditions": conditions,
        }
        rows.append(row)
        if not ok:
            failures.append(ordinal)

    forbidden = {
        "source_body_answer_route": receipt.get("source_body_answer_route"),
        "successor_answer_route": receipt.get("successor_answer_route"),
        "character_nearest_route": receipt.get("character_nearest_route"),
        "language_vocabulary_route": receipt.get("language_vocabulary_route"),
        "opencc_route": receipt.get("opencc_route"),
    }
    status = (
        "PASS_PAIRWISE_CORE_GRAPH_COMPOSITION"
        if not failures
        and len(rows) == 3
        and all(value == 0 for value in forbidden.values())
        and receipt.get("model_read_only") == 1
        and receipt.get("free_dialogue_complete") == 0
        else "FAIL"
    )
    report = {
        "format": "PURE_INTEGER_CORE_GRAPH_PAIR_QUALIFICATION_V1",
        "status": status,
        "run_root": str(RUN),
        "receipt_sha256": _sha256(receipt_path),
        "pair_count": len(rows),
        "three_graph_pair_query_count": sum(
            row["active_spaces"] == [1, 2, 3] for row in rows),
        "both_core_consumed_pair_count": sum(
            row["core_consumed_count"] == 2 for row in rows),
        "both_response_plan_referenced_pair_count": sum(
            row["plan_reference_count"] == 2 for row in rows),
        "generation_count": receipt.get("generation_count"),
        "generation_ready_count": sum(row["generation_ready"] for row in rows),
        "delivery_count": receipt.get("delivery_count"),
        "termination_codes": receipt.get("termination_codes"),
        "canonical_model_read_only": receipt.get("model_read_only"),
        "canonical_sha_before": receipt.get("database_model_sha256_before"),
        "canonical_sha_after": receipt.get("database_model_sha256_after"),
        "source_text_rows_added": receipt.get("source_text_rows_added"),
        "database_copy_count": receipt.get("database_copy_count"),
        "forbidden_routes": forbidden,
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
