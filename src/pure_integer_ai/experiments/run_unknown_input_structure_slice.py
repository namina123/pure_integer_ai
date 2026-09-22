"""Run the integer-only unknown-input structure path through all three graphs.

This is a construction artifact, not a completion claim.  Inputs that do not
match a trained semantic topology are retained as source-scoped generic
Memory candidates.  The bridge still creates Core, Interaction Memory and
Dialogue roots in one QueryState; generic candidates carry UNKNOWN evidence
and can never become Core facts or successor/surface responses.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_UNKNOWN
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_GENERIC_CANDIDATE_VERSION,
    memory_candidate_category,
    memory_input_candidate_keys,
)
from pure_integer_ai.cognition.understanding.query_input_structure import STRUCTURE_CLOSED
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.heldout_response_runtime import (
    HeldoutResponseRuntime,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _write_trace(value: object, target: Path) -> str:
    with target.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(_json_bytes(value))
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_only(value: object, path: tuple[object, ...] = ()) -> None:
    """Reject strings, floats and booleans in the persisted query projection."""
    if type(value) is int:
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _integer_only(item, (*path, index))
        return
    if type(value) is dict:
        for key, item in value.items():
            _integer_only(item, (*path, key))
        return
    raise ValueError(
        f"unknown input trace contains a non-integer value at {path}: {value!r}")


def _projection(result: dict[str, object], *,
                heldout_connector_selected: int = 0,
                core_precedence: int = 0) -> list[object]:
    """Keep only the complete integer QueryState fields needed for recovery."""
    if heldout_connector_selected not in {0, 1}:
        raise ValueError("heldout connector marker must be zero or one")
    if core_precedence not in {0, 1}:
        raise ValueError("Core precedence marker must be zero or one")
    projection = [
        result["query_key"],
        result["active_spaces"],
        result["roots"],
        result["anchors"],
        result["frontier_trace"],
        result["bindings"],
        result["evidence"],
        result["visited"],
        result["expanded_edges"],
        result["termination"],
        result["termination_state"],
        result["discourse_topic_query_projection"],
        [] if result["response_plan"] is None else result["response_plan"],
        [] if result["generation"] is None else result["generation"],
        result["generic_response_generations"],
        result["generic_response_feature_mask"],
        list(map(ord, result["response_surface"])),
        result.get("discourse_graph_input_requested_keys", ()),
        result.get("discourse_graph_input_keys", ()),
        result.get("event_time_graph_input_keys", ()),
        (*result.get("core_graph_input_keys", ()),
         *result.get("generic_graph_input_keys", ())),
        heldout_connector_selected,
        core_precedence,
    ]
    _integer_only(projection)
    return projection


def _frontier_count(frontier_trace: object) -> int:
    """Read the final frontier size from either trace encoding.

    The bridge's current integer trace is a dict containing ``steps``;
    older runs used a tuple/list whose third field was the frontier.  The
    statistic is observational only and must not normalize or persist the
    trace itself.
    """
    if not frontier_trace:
        return 0
    if isinstance(frontier_trace, dict):
        steps = frontier_trace.get("steps")
        if not isinstance(steps, list) or not steps:
            return 0
        last = steps[-1]
        if not isinstance(last, dict):
            return 0
        refs = last.get("frontier_refs")
        if refs is None:
            refs = last.get("frontier")
        return len(refs) if isinstance(refs, (list, tuple)) else 0
    last = frontier_trace[-1]
    if isinstance(last, (list, tuple)) and len(last) > 2:
        return len(last[2]) if isinstance(last[2], (list, tuple)) else 0
    return 0


def _trace_graph_keys(value: object, *, label: str) -> tuple[tuple[int, ...], ...]:
    """Normalize graph-key rows emitted by the full or compact trace."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an integer key sequence")
    result = []
    for key in value:
        if (not isinstance(key, (list, tuple)) or not key
                or any(type(item) is not int or item < 0 for item in key)):
            raise ValueError(f"{label} contains an invalid integer key")
        result.append(tuple(key))
    return tuple(result)


def run_slice(database: Path, run_root: Path, inputs: tuple[str, ...], *,
              session_id: int, heldout_course: Path | None = None,
              heldout_manifest: Path | None = None,
              graph_object_keys: tuple[tuple[int, ...], ...] = (),
              ) -> dict[str, object]:
    """Persist real unknown-input O/H/E and same-turn three-graph traces."""
    database = database.resolve(strict=True)
    root = run_root.resolve()
    if database.drive.upper() != "K:" or root.drive.upper() != "K:":
        raise ValueError("unknown-input slice requires explicit K drive paths")
    if root.exists() or not inputs or type(session_id) is not int or session_id <= 0:
        raise ValueError("unknown-input slice arguments are invalid")
    if (type(graph_object_keys) is not tuple
            or any(type(key) is not tuple or not key
                   or any(type(value) is not int or value < 0 for value in key)
                   for key in graph_object_keys)
            or len(set(graph_object_keys)) != len(graph_object_keys)
            or (graph_object_keys and len(graph_object_keys) != len(inputs))):
        raise ValueError(
            "graph_object_keys must be unique integer keys aligned one per input")
    canonical_sha_before = _sha256(database)
    root.mkdir(parents=True)
    session = root / "session.sqlite3"
    rows: list[dict[str, object]] = []
    snapshots: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
    if (heldout_course is None) != (heldout_manifest is None):
        raise ValueError("heldout course and manifest must be supplied together")
    heldout_course_sha = None
    heldout_manifest_sha = None
    heldout_manifest_payload: dict[str, object] | None = None
    with TrainedGenerationConnectorRuntime(database) as canonical:
        generator_context = (HeldoutResponseRuntime(
            canonical, course_path=heldout_course,
            manifest_path=heldout_manifest)
            if heldout_course is not None else nullcontext(canonical))
        with generator_context as generator, TrainedGraphQueryBridge(
                database, memory_database=session, session_id=session_id,
                surface_generator=generator) as bridge:
            if heldout_course is not None:
                heldout_course_sha = hashlib.sha256(heldout_course.resolve(strict=True).read_bytes()).hexdigest()
                heldout_manifest_sha = hashlib.sha256(heldout_manifest.resolve(strict=True).read_bytes()).hexdigest()
                heldout_manifest_payload = json.loads(
                    heldout_manifest.resolve(strict=True).read_text(encoding="ascii"))
            for ordinal, surface in enumerate(inputs, 1):
                if type(surface) is not str or not surface:
                    raise ValueError("unknown input must be non-empty text")
                projected = bridge.input_projector.project(tuple(map(ord, surface)))
                if (any(
                        item.state == STRUCTURE_CLOSED and item.required_count >= 2
                        and item.required_count == item.support_count
                        and item.predicate_coverage == 1
                        for item in projected.relation_candidates)):
                    raise ValueError("input unexpectedly matched a trained topology")
                appended = bridge.memory.append(surface, speaker_kind=1, stance=EVIDENCE_UNKNOWN)
                observation = bridge.memory.intake.result_for_source(appended.source).observation_ref
                candidate_keys = memory_input_candidate_keys(projected, session_id=session_id, speaker_kind=1)
                generic = tuple(key for key in candidate_keys if key[0] == MEMORY_GENERIC_CANDIDATE_VERSION)
                if len(generic) != 2 or {memory_candidate_category(key) for key in generic} != {6, 7}:
                    raise ValueError("unknown input did not produce generic time/discourse candidates")
                active = bridge.memory.active_structures(candidate_keys=candidate_keys, observation_refs=(observation,))
                hypotheses = tuple(item for item in active.hypotheses if item.candidate_key in generic)
                evidence = tuple(item for item in active.evidence if item.hypothesis_key in {item.hypothesis_key for item in hypotheses})
                if len(hypotheses) != 2 or len(evidence) != 2 or any(item.stance != 3 for item in evidence):
                    raise ValueError("generic unknown input O/H/E is not complete UNKNOWN")
                query_graph_keys = (() if not graph_object_keys else
                                    (graph_object_keys[ordinal - 1],))
                result = bridge.query(
                    surface, max_depth=64,
                    graph_object_keys=query_graph_keys,
                    input_observation_ref=observation)
                requested_graph_keys = _trace_graph_keys(
                    result.get("discourse_graph_input_requested_keys", ()),
                    label="requested graph inputs")
                consumed_graph_keys = set(_trace_graph_keys(
                    result.get("discourse_graph_input_keys", ()),
                    label="Dialogue graph inputs"))
                consumed_graph_keys.update(_trace_graph_keys(
                    result.get("event_time_graph_input_keys", ()),
                    label="Event/Time graph inputs"))
                core_consumed_graph_keys = set(_trace_graph_keys(
                    result.get("core_graph_input_keys", ()),
                    label="Core graph inputs"))
                core_consumed_graph_keys.update(_trace_graph_keys(
                    result.get("generic_graph_input_keys", ()),
                    label="Generic Core graph inputs"))
                consumed_graph_keys.update(core_consumed_graph_keys)
                if (query_graph_keys
                        and (requested_graph_keys != query_graph_keys
                             or not set(query_graph_keys) <= consumed_graph_keys)):
                    raise ValueError(
                        "explicit graph input was not consumed by the same QueryState")
                if result["active_spaces"] != [1, 2, 3]:
                    raise ValueError("unknown input QueryState did not retain all graph spaces")
                restored_context_count = len(result["response_contexts"])
                if (ordinal > 1 and rows and rows[-1].get("heldout_match")
                        and restored_context_count < 1):
                    raise ValueError("prior delivered response did not re-enter Dialogue context")
                hypothesis_keys = {item.hypothesis_key for item in hypotheses}
                dialogue_roots = {tuple(row["root"]) for row in result["roots"] if row["space"] == 3}
                if not hypothesis_keys <= dialogue_roots:
                    raise ValueError("generic Memory hypotheses missing Dialogue roots")
                generated = bool(
                    result["response_surface"] and result["generation"]
                    and result["generation"].get("kind") == 3
                    and result["generic_response_generations"])
                heldout_connector_selected = bool(
                    generated and heldout_course is not None
                    and hasattr(generator, "is_heldout_connector")
                    and generator.is_heldout_connector(tuple(
                        result["generation"].get("connector", ()))))
                response_plan = result["response_plan"] or {}
                planned_graph_keys = {
                    tuple(key) for field in (
                        "claim_refs", "event_refs", "discourse_links")
                    for key in response_plan.get(field, ())
                    if isinstance(key, list)
                }
                planned_graph_keys.update(
                    tuple(slot["filler"])
                    for slot in response_plan.get("slot_sequence", ())
                    if (isinstance(slot, dict)
                        and isinstance(slot.get("filler"), list))
                )
                # A factual Core connector may legitimately win over the
                # lower-priority heldout/Dialogue connector.  Record this as
                # explicit graph precedence rather than treating it as a
                # missing response or allowing the companion overlay to win.
                core_precedence = bool(
                    generated and not heldout_connector_selected
                    and query_graph_keys
                    and set(query_graph_keys) <= planned_graph_keys
                    and core_consumed_graph_keys
                    and set(query_graph_keys) <= core_consumed_graph_keys)
                # Runs without an overlay use this field as the existing
                # generic-connector match gate.  Qualification runs load an
                # overlay and must prove that its connector, not a canonical
                # tie, produced the selected ResponsePlan.
                matched = (
                    heldout_connector_selected or core_precedence
                    if heldout_course is not None else generated)
                if not matched and heldout_course is None:
                    raise ValueError("unknown input did not produce the trained generic connector act")
                generation_consumed_graph_keys = (
                    set(query_graph_keys) & planned_graph_keys)
                delivery_summary = {"use_event_hashes": (), "outcome_event_hashes": (),
                                    "delivered": 0, "fact_claims_added": 0}
                if matched:
                    if (not result["generation"].get("connector")
                            or not result["generation"].get("representations")
                            or len(result["generic_response_generations"]) != 1):
                        raise ValueError("unknown input did not complete the trained six-layer connector")
                    if (heldout_connector_selected
                            and (response_plan.get("claim_refs")
                                 or not response_plan.get("memory_refs"))):
                        raise ValueError("unknown response plan unexpectedly carries a claim")
                    delivery = bridge.acknowledge_response_delivery(result["response_surface"])
                    if delivery is None:
                        raise ValueError("unknown response was not adopted into Interaction Memory")
                    delivery_summary = delivery.summary()
                    if (delivery_summary["delivered"] != 1 or delivery_summary["fact_claims_added"] != 0
                            or not delivery_summary["use_event_hashes"]
                            or len(delivery_summary["use_event_hashes"]) != len(delivery_summary["outcome_event_hashes"])):
                        raise ValueError("unknown response delivery attribution is incomplete")
                trace_sha = _write_trace(
                    _projection(
                        result,
                        heldout_connector_selected=int(
                            heldout_connector_selected),
                        core_precedence=int(core_precedence)),
                    root / f"query-{ordinal:03d}.int.json.gz")
                rows.append({"ordinal": ordinal, "generic_candidate_count": len(generic),
                    "hypothesis_count": len(hypotheses), "evidence_count": len(evidence),
                    "active_space_count": len(result["active_spaces"]), "dialogue_root_count": len(hypothesis_keys),
                    "response_surface_value_count": len(result["response_surface"]),
                    "response_slot_count": len(response_plan.get("slot_sequence", ())),
                    "delivery_use_count": len(delivery_summary["use_event_hashes"]),
                    "delivery_outcome_count": len(delivery_summary["outcome_event_hashes"]),
                    "restored_response_context_count": restored_context_count,
                    "generic_connector_generation_count": len(result["generic_response_generations"]),
                    "termination_code": result["termination"], "trace_sha256": trace_sha,
                    "heldout_match": int(heldout_connector_selected),
                    "heldout_connector_selected": int(heldout_connector_selected),
                    "core_precedence": int(core_precedence),
                    "graph_input_count": len(query_graph_keys),
                    "graph_input_keys": [list(key) for key in query_graph_keys],
                    "consumed_graph_input_count": len(
                        set(query_graph_keys) & consumed_graph_keys),
                    "core_graph_input_consumed_count": len(
                        set(query_graph_keys) & core_consumed_graph_keys),
                    "generation_consumed_graph_input_count": len(
                        generation_consumed_graph_keys),
                    "root_count": len(result["roots"]),
                    "frontier_count": _frontier_count(result["frontier_trace"]),
                    "visited_count": len(result["visited"]),
                    "query_evidence_count": len(result["evidence"]),
                    "ohe_observation_count": len(bridge._memory_observations),
                    "ohe_hypothesis_count": len(bridge._memory_hypotheses),
                    "ohe_evidence_count": len(bridge._memory_evidence)})
                state = bridge.memory.active_structures()
                snapshots.append((tuple(item.observation_key for item in state.observations),
                                  tuple(item.hypothesis_key for item in state.hypotheses),
                                  tuple(item.evidence_key for item in state.evidence)))
    with TrainedGraphQueryBridge(
            database, memory_database=session, session_id=session_id) as restored:
        state = restored.memory.active_structures()
        restored_snapshot = (
            tuple(item.observation_key for item in state.observations),
            tuple(item.hypothesis_key for item in state.hypotheses),
            tuple(item.evidence_key for item in state.evidence),
        )
    if not snapshots or restored_snapshot != snapshots[-1]:
        raise ValueError("unknown input O/H/E cold restore mismatch")
    canonical_sha_after = _sha256(database)
    if canonical_sha_after != canonical_sha_before:
        raise ValueError("heldout consumption changed canonical SQLite")
    receipt = {
        "protocol": 1,
        "queried_user_count": len(rows),
        "active_spaces": [1, 2, 3],
        "schema_version": 1,
        "query_count": len(rows),
        "unknown_query_count": len(rows),
        "three_graph_query_count": sum(item["active_space_count"] == 3 for item in rows),
        "generic_candidate_count": sum(item["generic_candidate_count"] for item in rows),
        "hypothesis_count": sum(item["hypothesis_count"] for item in rows),
        "evidence_count": sum(item["evidence_count"] for item in rows),
        "cold_restore_equal": int(restored_snapshot == snapshots[-1]),
        "model_read_only": 1,
        "canonical_model_sha256_before": canonical_sha_before,
        "canonical_model_sha256_after": canonical_sha_after,
        "database_copy_count": 0,
        "source_text_rows_added": 0,
        "source_body_answer_route": 0,
        "successor_answer_route": 0,
        "character_nearest_route": 0,
        "opencc_route": 0,
        "language_vocabulary_route": 0,
        "core_training_performed": 0,
        "response_generation_performed": 1,
        "unknown_response_count": len(rows),
        "termination_codes": [item["termination_code"] for item in rows],
        "response_delivery_count": sum(
            item["delivery_use_count"] > 0 for item in rows),
        "delivery_count": sum(
            item["delivery_use_count"] > 0 for item in rows),
        "response_delivery_use_count": sum(
            item["delivery_use_count"] for item in rows),
        "response_delivery_outcome_count": sum(
            item["delivery_outcome_count"] for item in rows),
        "graph_segment_count": sum(
            item["response_slot_count"] for item in rows),
        "generic_connector_generation_count": sum(
            item["generic_connector_generation_count"] for item in rows),
        "restored_response_context_count": sum(
            item["restored_response_context_count"] for item in rows),
        "free_dialogue_complete": 0,
        "heldout_course_sha256": heldout_course_sha,
        "heldout_manifest_sha256": heldout_manifest_sha,
        "heldout_consumption": int(heldout_course is not None),
        "heldout_owner_verified": int(heldout_course is not None),
        "heldout_match_count": sum(item["heldout_match"] for item in rows),
        "heldout_connector_selected_count": sum(
            item["heldout_connector_selected"] for item in rows),
        "core_precedence_count": sum(item["core_precedence"] for item in rows),
        "integer_trace": 1,
        "graph_input_count": len(graph_object_keys),
        "consumed_graph_input_count": sum(
            item["consumed_graph_input_count"] for item in rows),
        "core_graph_input_consumed_count": sum(
            item["core_graph_input_consumed_count"] for item in rows),
        "generation_consumed_graph_input_count": sum(
            item["generation_consumed_graph_input_count"] for item in rows),
        "after_ohe": [
            rows[-1]["ohe_observation_count"],
            rows[-1]["ohe_hypothesis_count"],
            rows[-1]["ohe_evidence_count"],
        ],
        "semantic_variant_registry_normalized": 1,
        "heldout_whole_sentence_representation_count": 0,
        "heldout_graph_role_categories": (
            heldout_manifest_payload.get("graph_role_categories", [])
            if heldout_manifest_payload is not None else []),
        "heldout_minimum_graph_slots_per_variant": 1,
        "rows": rows,
    }
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--session-id", required=True, type=int)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--heldout-course", type=Path)
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--graph-object-key", action="append", default=[])
    args = parser.parse_args()
    graph_object_keys = tuple(
        tuple(int(item, 10) for item in value.split(","))
        for value in args.graph_object_key)
    result = run_slice(args.database, args.run_root, tuple(args.input),
                       session_id=args.session_id,
                       heldout_course=args.heldout_course,
                       heldout_manifest=args.heldout_manifest,
                       graph_object_keys=graph_object_keys)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_trace_graph_keys", "run_slice", "main"]
