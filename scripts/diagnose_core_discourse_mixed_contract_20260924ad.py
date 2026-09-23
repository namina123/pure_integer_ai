"""Diagnose graph-slot closure for one mixed heldout QueryState."""
from __future__ import annotations

import os
from pathlib import Path

import scripts.run_core_discourse_mixed_20260924x as probe
from pure_integer_ai.experiments import trained_open_response_runtime as open_runtime


DATABASE = Path(
    r"K:/pure_integer_ai_work/current_runs/discourse-relation-response-graph-20260917b/training.sqlite3")
COURSE = Path(
    r"K:/pure_integer_ai_work/current_runs/core-discourse-mixed-course-20260924ab/heldout_response_course.int.json")
MANIFEST = Path(
    r"K:/pure_integer_ai_work/current_runs/core-discourse-mixed-course-20260924ab/heldout_manifest.json")
ROOT = Path(r"K:/pure_integer_ai_work/current_runs/core-discourse-mixed-diagnose-20260924af")


def _compact_hop(hop):
    return {
        "space": getattr(hop, "space", None),
        "depth": getattr(hop, "depth", None),
        "from": getattr(hop, "from_filler", ()),
        "to": getattr(hop, "to_filler", ()),
        "predicate": getattr(hop, "predicate", ()),
        "relation_fillers": getattr(hop, "relation_fillers", ()),
        "proposition": getattr(hop, "proposition", ()),
    }


def main() -> int:
    if ROOT.exists():
        raise ValueError(f"diagnostic root must be new: {ROOT}")
    ROOT.mkdir(parents=True)
    original = open_runtime._generic_graph_values

    def wrapped(*args, **kwargs):
        state, contract, input_structure = args[:3]
        result = original(*args, **kwargs)
        slots = [(item.category, item.ordinal) for item in contract.graph_slots]
        explicit_keys = kwargs.get(
            "explicit_graph_input_keys", args[8] if len(args) > 8 else ())
        hops = kwargs.get("graph_hops", args[9] if len(args) > 9 else ())
        roots_by_kind = {}
        for key in explicit_keys:
            identity = open_runtime._graph_identity(key)
            if identity is not None:
                roots_by_kind[identity.object_kind] = (
                    roots_by_kind.get(identity.object_kind, 0) + 1)
        binding_identities = {
            (item.space, item.filler_key)
            for item in state.bindings
            if item.space == open_runtime.SPACE_CORE and not item.conflict_kept
        }
        if ({category for category, _ordinal in slots}
                != {open_runtime.GENERIC_RESPONSE_GRAPH_ENTITY,
                    open_runtime.GENERIC_RESPONSE_GRAPH_EVENT,
                    open_runtime.GENERIC_RESPONSE_GRAPH_CONCEPT,
                    open_runtime.GENERIC_RESPONSE_GRAPH_TOPIC}
                or any(ordinal != 1 for _category, ordinal in slots)):
            return result
        binding_counts = {}
        for item in state.bindings:
            if item.conflict_kept:
                continue
            identity = open_runtime._graph_identity(item.filler_key)
            if identity is not None and item.filler_key in explicit_keys:
                count_key = (identity.object_kind, item.space)
                binding_counts[count_key] = binding_counts.get(count_key, 0) + 1
        payload = {
            "slots": slots,
            "filled_slot_count": None if result is None else len(result),
            "explicit_root_kind_counts": roots_by_kind,
            "explicit_root_core_binding_count": sum(
                (open_runtime.SPACE_CORE, key) in binding_identities
                for key in explicit_keys),
            "explicit_root_binding_counts_by_kind_space": {
                str(key): value for key, value in binding_counts.items()},
            "hop_count": len(hops),
            "core_hop_depth_counts": {
                depth: sum(1 for item in hops
                          if getattr(item, "space", None)
                          == open_runtime.SPACE_CORE
                          and getattr(item, "depth", None) == depth)
                for depth in (1, 2, 3, 4, 8, 16, 32, 64)
            },
            "relation_candidate_count": len(input_structure.relation_candidates),
        }
        print(payload, flush=True)
        return result

    open_runtime._generic_graph_values = wrapped
    try:
        with probe._CanonicalWithHeldout(DATABASE):
            triples = probe._core_triples()
            with probe._GraphBridgeWithHeldout(
                    DATABASE, memory_database=ROOT / "session.sqlite3",
                    session_id=92412) as bridge:
                discourse_key = probe._qud_propositions(bridge)[0]
                graph_keys = tuple(sorted((*triples[0], discourse_key)))
                appended = bridge.memory.append_graph_input(graph_keys, speaker_kind=1)
                observation = bridge.memory.intake.result_for_source(
                    appended.source).observation_ref
                result = bridge.query(
                    None, max_depth=64, graph_object_keys=graph_keys,
                    input_observation_ref=observation)
                print({"termination": result.get("termination"),
                       "generation": int(bool(result.get("generation"))),
                       "feature_mask": result.get(
                           "generic_response_feature_mask")}, flush=True)
    finally:
        open_runtime._generic_graph_values = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
