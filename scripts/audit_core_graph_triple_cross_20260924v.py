"""Audit cross-identity three-object Core graph traces."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

from scripts.audit_core_graph_pair_heldout_20260924o import _response_plan_refs


RUN = Path(os.environ.get(
    "PURE_INTEGER_CORE_TRIPLE_CROSS_AUDIT_RUN_ROOT",
    r"K:/pure_integer_ai_work/current_runs/core-graph-triple-cross-heldout-20260924v"))
OUT = Path(os.environ.get(
    "PURE_INTEGER_CORE_TRIPLE_CROSS_AUDIT_OUT",
    r"K:/pure_integer_ai_work/audits/core-graph-triple-cross-qualification-20260924v"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_row(row: dict[str, object]) -> dict[str, object]:
    keys = {tuple(item) for item in row["graph_input_keys"]}
    trace_path = RUN / row["trace_file"]
    with gzip.open(trace_path, "rt", encoding="ascii") as stream:
        trace = json.load(stream)
    state = trace.get("termination_state", {})
    trace_core = {tuple(item) for item in trace.get(
        "core_graph_input_keys", ())}
    plan_refs = _response_plan_refs(trace.get("response_plan"))
    conditions = {
        "trace_hash": row.get("trace_sha256") == _sha256(trace_path),
        "active_spaces": trace.get("active_spaces") == [1, 2, 3],
        "all_core_consumed": keys <= trace_core,
        "all_plan_referenced": keys <= plan_refs,
        "termination_closed": trace.get("termination") == 8,
        "generation_ready": state.get("generation_ready") == 1,
        "generation_present": isinstance(trace.get("generation"), dict),
        "delivery": row.get("delivery") == 1,
        "status_ok": row.get("status") == "OK",
    }
    return {
        "ordinal": row.get("ordinal"),
        "graph_input_keys": row.get("graph_input_keys"),
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "termination": trace.get("termination"),
        "generation_ready": state.get("generation_ready"),
        "delivery": row.get("delivery"),
        "trace_file": trace_path.name,
        "trace_sha256": _sha256(trace_path),
        "plan_reference_count": sum(item in plan_refs for item in keys),
    }


def main() -> int:
    receipt_path = RUN / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    rows = []
    for row in receipt.get("rows", ()):
        if row.get("status") != "OK":
            rows.append({
                "ordinal": row.get("ordinal"),
                "graph_input_keys": row.get("graph_input_keys"),
                "status": "FAIL",
                "conditions": {"status_ok": False},
            })
        else:
            rows.append(_audit_row(row))
    forbidden = {
        "source_body_answer_route": receipt.get("source_body_answer_route"),
        "successor_answer_route": receipt.get("successor_answer_route"),
        "character_nearest_route": receipt.get("character_nearest_route"),
        "language_vocabulary_route": receipt.get("language_vocabulary_route"),
        "opencc_route": receipt.get("opencc_route"),
    }
    all_rows_pass = bool(rows) and all(row["status"] == "PASS" for row in rows)
    status = (
        "PASS_CROSS_TRIPLE_CORE_GRAPH_COMPOSITION"
        if all_rows_pass
        and all(value == 0 for value in forbidden.values())
        and receipt.get("model_read_only") == 1
        and receipt.get("free_dialogue_complete") == 0
        else "FAIL_CROSS_TRIPLE_CORE_GRAPH_COMPOSITION")
    report = {
        "format": "PURE_INTEGER_CORE_GRAPH_CROSS_TRIPLE_QUALIFICATION_V1",
        "status": status,
        "run_root": str(RUN),
        "receipt_sha256": _sha256(receipt_path),
        "cross_triple_count": receipt.get("cross_triple_count"),
        "three_graph_triple_query_count": receipt.get(
            "three_graph_triple_query_count"),
        "all_core_consumed_triple_count": receipt.get(
            "all_core_consumed_triple_count"),
        "all_consumed_triple_count": receipt.get("all_consumed_triple_count"),
        "generation_count": receipt.get("generation_count"),
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
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "qualification.json").write_bytes((
        json.dumps(report, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")) + "\n").encode("ascii"))
    print(json.dumps(report, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
