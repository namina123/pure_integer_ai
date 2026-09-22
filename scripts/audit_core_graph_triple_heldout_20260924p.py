"""Audit triple-object Core graph composition from its integer trace."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from scripts.audit_core_graph_pair_heldout_20260924o import _response_plan_refs


RUN = Path(r"K:/pure_integer_ai_work/current_runs/core-graph-triple-heldout-20260924p")
OUT = Path(
    r"K:/pure_integer_ai_work/audits/core-graph-triple-qualification-20260924p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    receipt_path = RUN / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    expected = receipt["rows"][0]
    keys = {tuple(item) for item in expected["graph_input_keys"]}
    trace_path = RUN / expected["trace_file"]
    with gzip.open(trace_path, "rt", encoding="ascii") as stream:
        trace = json.load(stream)
    state = trace.get("termination_state", {})
    trace_core = {tuple(item) for item in trace.get(
        "core_graph_input_keys", ())}
    plan_refs = _response_plan_refs(trace.get("response_plan"))
    conditions = {
        "trace_hash": expected.get("trace_sha256") == _sha256(trace_path),
        "active_spaces": trace.get("active_spaces") == [1, 2, 3],
        "all_core_consumed": keys <= trace_core,
        "all_plan_referenced": keys <= plan_refs,
        "termination_closed": trace.get("termination") == 8,
        "generation_ready": state.get("generation_ready") == 1,
        "generation_present": isinstance(trace.get("generation"), dict),
        "delivery": expected.get("delivery") == 1,
        "status_ok": expected.get("status") == "OK",
    }
    forbidden = {
        "source_body_answer_route": receipt.get("source_body_answer_route"),
        "successor_answer_route": receipt.get("successor_answer_route"),
        "character_nearest_route": receipt.get("character_nearest_route"),
        "language_vocabulary_route": receipt.get("language_vocabulary_route"),
        "opencc_route": receipt.get("opencc_route"),
    }
    ok = all(conditions.values())
    status = (
        "PASS_TRIPLE_CORE_GRAPH_COMPOSITION"
        if ok and all(value == 0 for value in forbidden.values())
        and receipt.get("model_read_only") == 1
        and receipt.get("free_dialogue_complete") == 0
        else "FAIL"
    )
    report = {
        "format": "PURE_INTEGER_CORE_GRAPH_TRIPLE_QUALIFICATION_V1",
        "status": status,
        "run_root": str(RUN),
        "receipt_sha256": _sha256(receipt_path),
        "triple_count": receipt.get("triple_count"),
        "three_graph_triple_query_count": receipt.get(
            "three_graph_triple_query_count"),
        "all_core_consumed_triple_count": receipt.get(
            "all_core_consumed_triple_count"),
        "all_response_plan_referenced": int(keys <= plan_refs),
        "plan_reference_count": sum(item in plan_refs for item in keys),
        "generation_count": receipt.get("generation_count"),
        "generation_ready_count": int(state.get("generation_ready", 0)),
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
        "conditions": conditions,
        "graph_input_keys": [list(item) for item in expected[
            "graph_input_keys"]],
        "trace_file": trace_path.name,
        "trace_sha256": _sha256(trace_path),
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
