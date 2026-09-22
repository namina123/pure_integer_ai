"""Build an isolated integer held-out overlay for the live three-slot shape."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(r"K:/pure_integer_ai_work/current_runs/broad-structural-heldout-course-20260923b")
EVENT = 65536

def literal(*units: int) -> list[object]: return [1, list(units)]
def graph(category: int, ordinal: int) -> list[int]: return [2, category, ordinal]
def row(index: int, mask: int) -> list[object]:
    return [[91561, mask, EVENT, 10000 + index], [
        literal(0x5B00 + index, 0x5B20 + index), graph(1, 1), graph(1, 2),
        graph(4, 1), literal(0x5B40 + index, 0x5B60 + index, 0x5B80 + index),
        literal(0x5BA0 + index, 0x5BC0 + index, 0x3002),
    ]]

def main() -> int:
    if ROOT.exists(): raise ValueError("heldout course root must be new")
    masks = (163969, 131201, 163969)
    rows = [row(i, mask) for i, mask in enumerate(masks, 1)]
    payload = json.dumps([91562, 3, [[i, *item] for i, item in enumerate(rows, 1)]], ensure_ascii=True, separators=(",", ":")).encode("ascii")
    manifest = json.dumps({
        "attribution": "Project-owned integer graph composition annotations; fillers are supplied by QueryState.",
        "course_sha256": hashlib.sha256(payload).hexdigest(), "format": "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2",
        "free_dialogue_claim": 0, "graph_role_categories": [1, 4], "graph_slot_count": 9,
        "license_ids": ["MIT"], "minimum_graph_slots_per_variant": 3,
        "official_url": "https://opensource.org/license/mit", "relation_qualified_graph_slots": 9,
        "rights_basis": "Project-owned integer graph composition.", "schema_version": 2,
        "selection_rule": "Specific forbidden Event feature plus the completed three-graph QueryState and ordered typed graph roles.",
        "source_id": "broad-structural-heldout-20260923b", "source_kind": "project_owned_structural_annotation",
        "source_version": 1, "split": "heldout", "variant_count": 3,
        "whole_sentence_representation_count": 0,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ROOT.mkdir(parents=True)
    (ROOT / "heldout_response_course.int.json").write_bytes(payload)
    (ROOT / "heldout_manifest.json").write_bytes(manifest)
    print(json.dumps({"root": str(ROOT), "course_sha256": hashlib.sha256(payload).hexdigest(), "variant_count": 3}, ensure_ascii=True, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
