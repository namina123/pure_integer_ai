"""Build a fresh integer-only three-slot graph-composition course."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(r"K:/pure_integer_ai_work/current_runs/broad-structural-course-20260922o")
UNKNOWN, RELATION = 1, 512
ENTITY, EVENT, PROPERTY, EVENT_TIME = 32768, 65536, 131072, 16384
HOT, COLD, MULTI, CONFLICT, REVISION, CHAIN = 2, 4, 128, 256, 4096, 1024
TOPIC, TOPIC_GRAPH, REF_OPEN, REF_RESOLVED = 8, 16, 64, 32


def literal(*units: int) -> list[object]:
    return [1, list(units)]


def graph(category: int, ordinal: int) -> list[int]:
    return [2, category, ordinal]


def row(index: int, mask: int,
        roles: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> list[object]:
    lead = (0x5600 + index, 0x5620 + index)
    middle = (0x5640 + index, 0x5660 + index, 0x5680 + index)
    tail = (0x56A0 + index, 0x56C0 + index, 0x3002)
    return [[91561, mask, 0, 9700 + index],
            [literal(*lead), *(graph(*role) for role in roles),
             literal(*middle), literal(*tail)]]


def payload(rows: tuple[list[object], ...]) -> bytes:
    return json.dumps([91562, 3, [[i, *value] for i, value in enumerate(rows, 1)]],
                      ensure_ascii=True, separators=(",", ":")).encode("ascii")


def manifest(data: bytes, split: str, count: int) -> bytes:
    return json.dumps({
        "attribution": "Project-owned integer graph composition annotations; fillers are supplied by QueryState.",
        "course_sha256": hashlib.sha256(data).hexdigest(),
        "format": "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2",
        "free_dialogue_claim": 0,
        "graph_role_categories": [1, 2, 3, 4, 5],
        "graph_slot_count": 3 * count,
        "license_ids": ["MIT"],
        "minimum_graph_slots_per_variant": 3,
        "official_url": "https://opensource.org/license/mit",
        "relation_qualified_graph_slots": 3 * count,
        "rights_basis": "Project-owned integer graph composition.",
        "schema_version": 2,
        "selection_rule": "Completed three-graph QueryState plus ordered typed graph roles and a structural proposition root.",
        "source_id": "broad-structural-course-20260922o-" + split,
        "source_kind": "project_owned_structural_annotation",
        "source_version": 1,
        "split": split,
        "variant_count": count,
        "whole_sentence_representation_count": 0,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def validate(rows: tuple[list[object], ...]) -> None:
    variants, literals = set(), set()
    for condition, parts in rows:
        roles = tuple(tuple(part[1:]) for part in parts if part[0] == 2)
        if len(roles) != 3 or len(set(roles)) != 3:
            raise ValueError("each variant requires three distinct graph roles")
        key = (tuple(condition), roles)
        if key in variants:
            raise ValueError("duplicate semantic variant")
        variants.add(key)
        for part in parts:
            if part[0] == 1:
                atom = tuple(part[1])
                if atom in literals:
                    raise ValueError("duplicate literal chunk")
                literals.add(atom)


def main() -> int:
    if ROOT.exists():
        raise ValueError("course output directory must be new")
    observed = (
        (163969, ((1, 1), (1, 2), (4, 1))),
        (131201, ((4, 1), (4, 2), (3, 1))),
        (163969, ((1, 1), (5, 1), (4, 1))),
    )
    states = (HOT | MULTI, COLD | MULTI, HOT | CONFLICT, COLD | CONFLICT,
              HOT | REVISION, COLD | REVISION, HOT | TOPIC_GRAPH,
              COLD | TOPIC_GRAPH, HOT | EVENT_TIME | TOPIC,
              COLD | EVENT_TIME | TOPIC, HOT | CHAIN, COLD | CHAIN,
              HOT | REF_OPEN, COLD | REF_RESOLVED, HOT | TOPIC, COLD | TOPIC)
    objects = (ENTITY, EVENT, PROPERTY, EVENT_TIME)
    triples = (((1, 1), (1, 2), (4, 1)),
               ((4, 1), (4, 2), (3, 1)),
               ((1, 1), (5, 1), (4, 1)),
               ((2, 1), (2, 2), (4, 1)))
    specs = list(observed)
    for i, state in enumerate(states):
        for obj in objects:
            mask = UNKNOWN | RELATION | state | obj
            if any(mask == item[0] for item in observed):
                continue
            specs.append((mask, triples[i % len(triples)]))
    if len(specs) < 64:
        raise ValueError("course inventory is incomplete")
    heldout_specs, development_specs = tuple(specs[:20]), tuple(specs[20:64])
    heldout = tuple(row(i, mask, roles) for i, (mask, roles) in enumerate(heldout_specs, 1))
    development = tuple(row(i + 20, mask, roles) for i, (mask, roles) in enumerate(development_specs, 1))
    validate(development)
    validate(heldout)
    dev_keys = {(tuple(item[0]), tuple(tuple(part[1:]) for part in item[1] if part[0] == 2)) for item in development}
    out_keys = {(tuple(item[0]), tuple(tuple(part[1:]) for part in item[1] if part[0] == 2)) for item in heldout}
    if dev_keys & out_keys:
        raise ValueError("development/held-out semantic overlap")
    ROOT.mkdir(parents=True)
    result = {"root": str(ROOT), "integer_payload": 1}
    for split, items in (("development", development), ("heldout", heldout)):
        data = payload(items)
        (ROOT / f"{split}_response_course.int.json").write_bytes(data)
        (ROOT / f"{split}_manifest.json").write_bytes(manifest(data, split, len(items)))
        result[f"{split}_variant_count"] = len(items)
        result[f"{split}_course_sha256"] = hashlib.sha256(data).hexdigest()
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
