"""Build a new held-out overlay for the two independently closed shapes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"K:/pure_integer_ai_work/current_runs/broad-structural-heldout-course-20260923d")
EVENT = 65536


def literal(*units: int) -> list[object]:
    return [1, list(units)]


def graph(category: int, ordinal: int) -> list[int]:
    return [2, category, ordinal]


def row(index: int, mask: int) -> list[object]:
    return [[91561, mask, EVENT, 11000 + index], [
        literal(0x5C00 + index, 0x5C20 + index),
        graph(1, 1),
        graph(1, 2),
        graph(4, 1),
        literal(0x5C40 + index, 0x5C60 + index, 0x5C80 + index),
        literal(0x5CA0 + index, 0x5CC0 + index, 0x3002),
    ]]


def main() -> int:
    if ROOT.exists():
        raise ValueError(f"heldout course root must be new: {ROOT}")
    masks = (163969, 163969)
    rows = [row(i, mask) for i, mask in enumerate(masks, 1)]
    payload = json.dumps(
        [91562, 3, [[i, *item] for i, item in enumerate(rows, 1)]],
        ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")
    manifest = json.dumps({
        "attribution": "Project-owned integer graph composition annotations; fillers are supplied by QueryState.",
        "course_sha256": hashlib.sha256(payload).hexdigest(),
        "format": "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2",
        "free_dialogue_claim": 0,
        "graph_role_categories": [1, 4],
        "graph_slot_count": 6,
        "license_ids": ["MIT"],
        "minimum_graph_slots_per_variant": 3,
        "official_url": "https://opensource.org/license/mit",
        "relation_qualified_graph_slots": 6,
        "rights_basis": "Project-owned integer graph composition.",
        "schema_version": 2,
        "selection_rule": "Specific forbidden Event feature plus the independently closed three-graph QueryState and ordered typed graph roles.",
        "source_id": "broad-structural-heldout-20260923d",
        "source_kind": "project_owned_structural_annotation",
        "source_version": 1,
        "split": "heldout",
        "variant_count": 2,
        "whole_sentence_representation_count": 0,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ROOT.mkdir(parents=True)
    (ROOT / "heldout_response_course.int.json").write_bytes(payload)
    (ROOT / "heldout_manifest.json").write_bytes(manifest)
    print(json.dumps({
        "root": str(ROOT),
        "course_sha256": hashlib.sha256(payload).hexdigest(),
        "variant_count": 2,
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
