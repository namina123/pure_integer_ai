"""Build a fresh integer-only three-slot course from one observed topology."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT = Path(r"K:/pure_integer_ai_work/current_runs/broad-structural-course-20260922q")
UNKNOWN, RELATION = 1, 512
ENTITY, EVENT, PROPERTY, EVENT_TIME = 32768, 65536, 131072, 16384
HOT, COLD, MULTI, CONFLICT, REVISION, CHAIN, QUD = 2, 4, 128, 256, 4096, 1024, 2048
TOPIC, TOPIC_GRAPH, REF_OPEN, REF_RESOLVED = 8, 16, 64, 32
ROLES = ((1, 1), (1, 2), (4, 1))
def literal(*units: int) -> list[object]: return [1, list(units)]
def graph(category: int, ordinal: int) -> list[int]: return [2, category, ordinal]
def row(index: int, mask: int) -> list[object]:
    return [[91561, mask, 0, 9900 + index], [literal(0x5800 + index, 0x5820 + index), *(graph(*role) for role in ROLES), literal(0x5840 + index, 0x5860 + index, 0x5880 + index), literal(0x58A0 + index, 0x58C0 + index, 0x3002)]]
def payload(rows: tuple[list[object], ...]) -> bytes: return json.dumps([91562, 3, [[i, *value] for i, value in enumerate(rows, 1)]], ensure_ascii=True, separators=(",", ":")).encode("ascii")
def manifest(data: bytes, split: str, count: int) -> bytes:
    return json.dumps({"attribution":"Project-owned integer graph composition annotations; fillers are supplied by QueryState.","course_sha256":hashlib.sha256(data).hexdigest(),"format":"STRUCTURAL_RESPONSE_COURSE_LEDGER_V2","free_dialogue_claim":0,"graph_role_categories":[1,4],"graph_slot_count":3*count,"license_ids":["MIT"],"minimum_graph_slots_per_variant":3,"official_url":"https://opensource.org/license/mit","relation_qualified_graph_slots":3*count,"rights_basis":"Project-owned integer graph composition.","schema_version":2,"selection_rule":"Completed three-graph QueryState plus the observed ordered typed relation and structural proposition root.","source_id":"broad-structural-course-20260922q-"+split,"source_kind":"project_owned_structural_annotation","source_version":1,"split":split,"variant_count":count,"whole_sentence_representation_count":0}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
def validate(rows: tuple[list[object], ...]) -> None:
    variants, literals = set(), set()
    for condition, parts in rows:
        roles = tuple(tuple(part[1:]) for part in parts if part[0] == 2)
        if roles != ROLES: raise ValueError("course role topology drifted")
        key = (tuple(condition), roles)
        if key in variants: raise ValueError("duplicate semantic variant")
        variants.add(key)
        for part in parts:
            if part[0] == 1:
                atom = tuple(part[1])
                if atom in literals: raise ValueError("duplicate literal chunk")
                literals.add(atom)
def main() -> int:
    if ROOT.exists(): raise ValueError("course output directory must be new")
    states = (HOT|MULTI,COLD|MULTI,HOT|CONFLICT,COLD|CONFLICT,HOT|REVISION,COLD|REVISION,HOT|TOPIC_GRAPH,COLD|TOPIC_GRAPH,HOT|QUD|TOPIC,COLD|QUD|TOPIC,HOT|CHAIN,COLD|CHAIN,HOT|REF_OPEN,COLD|REF_RESOLVED,HOT|TOPIC,COLD|TOPIC)
    objects = (ENTITY, EVENT, PROPERTY, EVENT_TIME); masks = tuple(UNKNOWN|RELATION|state|obj for state in states for obj in objects)
    if len(masks) != 64 or len(set(masks)) != 64: raise ValueError("course inventory is incomplete")
    rows = tuple(row(i, mask) for i, mask in enumerate(masks)); development, heldout = rows[:44], rows[44:]
    validate(development); validate(heldout)
    dev_literals={tuple(part[1]) for item in development for part in item[1] if part[0]==1}; held_literals={tuple(part[1]) for item in heldout for part in item[1] if part[0]==1}
    if dev_literals & held_literals: raise ValueError("development/heldout literals overlap")
    ROOT.mkdir(parents=True); result={"root":str(ROOT),"integer_payload":1}
    for split, items in (("development",development),("heldout",heldout)):
        data=payload(items); (ROOT/f"{split}_response_course.int.json").write_bytes(data); (ROOT/f"{split}_manifest.json").write_bytes(manifest(data,split,len(items))); result[f"{split}_variant_count"]=len(items); result[f"{split}_course_sha256"]=hashlib.sha256(data).hexdigest()
    print(json.dumps(result, ensure_ascii=True, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
