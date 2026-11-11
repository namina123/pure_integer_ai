"""Offline conformance check for the portable Core/Dialogue witness bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(key: tuple[int, ...]) -> tuple[int, ...]:
    from pure_integer_ai.experiments import trained_open_response_runtime

    identity = trained_open_response_runtime._graph_identity(key)
    return () if identity is None else identity.stable_key()


def _response_plan_keys(result: dict[str, object]) -> set[tuple[int, ...]]:
    from pure_integer_ai.experiments.run_trained_relation_graph_terminal import (
        _response_plan_graph_keys,
    )

    return _response_plan_graph_keys(result)


def _relation_fact_rows(
        relation_facts: dict[
            tuple[tuple[int, ...], tuple[int, ...]], set[tuple[int, ...]]
        ],
        ) -> list[dict[str, object]]:
    return [
        {"proposition": list(proposition),
         "predicate": list(predicate),
         "root_overlap_count": len(overlap)}
        for (proposition, predicate), overlap in sorted(relation_facts.items())
    ]


def _source_text_counts(database: Path) -> dict[str, int]:
    connection = sqlite3.connect(
        "file:" + str(database) + "?mode=ro", uri=True)
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        result = {
            "source_record_rows": 0,
            "source_record_text_rows": 0,
            "text_assoc_rows": 0,
        }
        if "source_record" in tables:
            result["source_record_rows"] = int(connection.execute(
                "SELECT COUNT(*) FROM source_record").fetchone()[0])
            columns = {row[1] for row in connection.execute(
                "PRAGMA table_info(source_record)")}
            if "raw_text" in columns:
                result["source_record_text_rows"] = int(connection.execute(
                    "SELECT COUNT(*) FROM source_record "
                    "WHERE raw_text IS NOT NULL").fetchone()[0])
        if "text_assoc" in tables:
            result["text_assoc_rows"] = int(connection.execute(
                "SELECT COUNT(*) FROM text_assoc").fetchone()[0])
        return result
    finally:
        connection.close()


def run_conformance(
        *, bundle_root: str | Path, model_root: str | Path,
        session_path: str | Path, output_path: str | Path,
        ) -> dict[str, object]:
    """Re-run the committed four-root witness without loading the project."""
    root = Path(bundle_root).resolve(strict=True)
    model = Path(model_root).resolve(strict=True)
    session = Path(session_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    for path, label in ((session, "session"), (output, "output")):
        try:
            path.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError(f"{label} must be outside the portable bundle")
    if session == model or session.is_dir():
        raise ValueError("session must be a new file outside the model")
    try:
        session.relative_to(model)
    except ValueError:
        pass
    else:
        raise ValueError("session must be outside the model root")
    if session.exists():
        raise FileExistsError(session)
    if output == session:
        raise ValueError("session and output must use different paths")
    if output.exists():
        raise FileExistsError(output)

    contract_path = root / "conformance" / "witness_contract.json"
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    graph_keys = tuple(tuple(row) for row in contract["input_keys"])
    if (len(graph_keys) != 4 or len(set(graph_keys)) != 4
            or any(type(value) is not int for key in graph_keys for value in key)):
        raise ValueError("portable witness contract is not four unique integer keys")
    canonical_model_sha = contract["canonical_model_sha256"]
    model_before = _sha256(model / "model" / "training.sqlite3")
    if model_before != canonical_model_sha:
        raise ValueError("portable model SHA does not match the witness contract")

    from pure_integer_ai.experiments.heldout_response_runtime import (
        HeldoutResponseRuntime,
    )
    from pure_integer_ai.experiments.trained_generation_connector_runtime import (
        TrainedGenerationConnectorRuntime,
    )
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        TrainedGraphQueryBridge,
    )
    from pure_integer_ai.experiments import trained_open_response_runtime

    course = root / "conformance" / "heldout_response_course.int.json"
    heldout_manifest = root / "conformance" / "heldout_manifest.json"
    captured: list[dict[str, object]] = []
    original = trained_open_response_runtime._generic_graph_values

    def capture(*args, **kwargs):
        state, response_contract, _input_structure = args[:3]
        result = original(*args, **kwargs)
        slots = tuple((item.category, item.ordinal)
                      for item in response_contract.graph_slots)
        if (len(slots) != 4 or {ordinal for _, ordinal in slots} != {1}
                or {category for category, _ in slots} != {1, 2, 3, 5}):
            return result
        slot_categories = {
            item.binding.slot: (item.category, item.ordinal)
            for item in response_contract.graph_slots
        }
        explicit = kwargs.get(
            "explicit_graph_input_keys", args[8] if len(args) > 8 else ())
        hops = kwargs.get("graph_hops", args[9] if len(args) > 9 else ())
        roots = {_stable_key(key) for key in explicit}
        parent: dict[tuple[int, ...], tuple[int, ...]] = {}

        def find(key: tuple[int, ...]) -> tuple[int, ...]:
            parent.setdefault(key, key)
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(left: tuple[int, ...], right: tuple[int, ...]) -> None:
            first, second = find(left), find(right)
            if first != second:
                parent[second] = first

        relation_facts: dict[
            tuple[tuple[int, ...], tuple[int, ...]], set[tuple[int, ...]]
        ] = {}
        for hop in hops:
            if (getattr(hop, "space", None) != trained_open_response_runtime.SPACE_CORE
                    or getattr(hop, "depth", 65) != 1):
                continue
            fillers = tuple(getattr(hop, "relation_fillers", ()))
            members = sorted({
                _stable_key(key) for key in fillers if _stable_key(key)
            })
            identity = (
                _stable_key(getattr(hop, "proposition", ())),
                tuple(getattr(hop, "predicate", ())),
            )
            overlap = { _stable_key(key) for key in members } & roots
            if identity[0] and len(members) >= 2 and overlap:
                relation_facts.setdefault(identity, set()).update(overlap)
            if members:
                for member in members:
                    find(_stable_key(member))
                for member in members[1:]:
                    union(_stable_key(members[0]), _stable_key(member))

        slot_rows = []
        for slot in result or ():
            if slot.slot not in slot_categories:
                continue
            slot_rows.append({
                "category": slot_categories[slot.slot][0],
                "ordinal": slot_categories[slot.slot][1],
                "filler": list(slot.filler.stable_key()),
                "evidence_count": len(slot.evidence),
                "explicit_root": int(slot.filler.stable_key() in roots),
            })
        root_groups = {find(key) for key in roots if key}
        captured.append({
            "slot_rows": slot_rows,
            "relation_facts": _relation_fact_rows(relation_facts),
            "root_group_count": len(root_groups),
            "state_evidence_count": len(state.evidence),
        })
        return result

    trained_open_response_runtime._generic_graph_values = capture
    generation = None
    delivery = None
    result: dict[str, object] | None = None
    try:
        with TrainedGenerationConnectorRuntime(model / "model" / "training.sqlite3") as canonical:
            with HeldoutResponseRuntime(
                    canonical, course_path=course,
                    manifest_path=heldout_manifest) as heldout:
                with TrainedGraphQueryBridge(
                        model / "model" / "training.sqlite3",
                        memory_database=session, session_id=92924,
                        surface_generator=heldout) as bridge:
                    appended = bridge.memory.append_graph_input(
                        graph_keys, speaker_kind=1)
                    observation = bridge.memory.intake.result_for_source(
                        appended.source).observation_ref
                    result = bridge.query(
                        None, max_depth=64, graph_object_keys=graph_keys,
                        input_observation_ref=observation)
                    generation = result.get("generation")
                    connector = (generation.get("connector")
                                 if isinstance(generation, dict) else None)
                    owner_verified = int(
                        isinstance(connector, (list, tuple))
                        and heldout.is_heldout_connector(tuple(connector)))
                    surface = result.get("response_surface", "")
                    if generation and surface:
                        delivery = bridge.acknowledge_response_delivery(surface)
    finally:
        trained_open_response_runtime._generic_graph_values = original

    if result is None:
        raise RuntimeError("portable QueryState did not execute")
    model_after = _sha256(model / "model" / "training.sqlite3")
    query_input_keys = set(graph_keys)
    consumed = set()
    for field in ("core_graph_input_keys", "generic_graph_input_keys",
                  "discourse_graph_input_keys", "event_time_graph_input_keys"):
        consumed.update(tuple(key) for key in result.get(field, ()))
    plan_keys = _response_plan_keys(result)
    candidate = captured[0] if captured else {}
    slot_rows = candidate.get("slot_rows", [])
    relation_facts = candidate.get("relation_facts", [])
    root_kinds = {key[0] for key in graph_keys}
    distinct_predicates = {tuple(row["predicate"]) for row in relation_facts}
    local_cross_root = any(row["root_overlap_count"] >= 2
                           for row in relation_facts)
    active_spaces = sorted(result.get("active_spaces", ()))
    source_counts = _source_text_counts(model / "model" / "training.sqlite3")
    required_ok = (
        len(graph_keys) == 4 and root_kinds == {4, 7, 16, 17}
        and len({_stable_key(key) for key in graph_keys}) == 4
        and all(_stable_key(key) for key in graph_keys)
        and len(captured) == 1
        and len(slot_rows) == 4
        and {row["category"] for row in slot_rows} == {1, 2, 3, 5}
        and all(row["ordinal"] == 1 and row["evidence_count"] > 0
                and row["explicit_root"] == 1 for row in slot_rows)
        and len(distinct_predicates) >= 2 and local_cross_root
        and candidate.get("root_group_count", 0) >= 2
        and set(result.get("termination_state", {}).get(
            "active_spaces", active_spaces)) >= {1, 2, 3}
        and query_input_keys <= consumed and query_input_keys <= plan_keys
        and result.get("termination") == 8
        and source_counts["source_record_rows"] == 6604
        and source_counts["source_record_text_rows"] == 6604
        and source_counts["text_assoc_rows"] == 0
        and bool(generation) and bool(result.get("termination_state", {}).get(
            "generation_ready", 0)) and owner_verified == 1
        and delivery is not None and model_before == model_after
    )
    if not required_ok:
        raise RuntimeError("portable Core/Dialogue witness conformance failed")

    payload = {
        "format": "PURE_INTEGER_PORTABLE_CORE_WITNESS_CONFORMANCE_V1",
        "status": "PASS",
        "model_sha256": model_after,
        "runtime_dependencies": [],
        "active_spaces": active_spaces,
        "termination": result.get("termination"),
        "query_state_four_slot_count": len(slot_rows),
        "same_query_evidence_slots": sum(
            row["evidence_count"] > 0 for row in slot_rows),
        "distinct_core_predicates": len(distinct_predicates),
        "local_multi_root_relation_witness": int(local_cross_root),
        "explicit_root_connectivity_groups": candidate.get(
            "root_group_count", 0),
        "input_keys_consumed": len(query_input_keys & consumed),
        "input_keys_in_response_plan": len(query_input_keys & plan_keys),
        "heldout_connector_owner_verified": owner_verified,
        "generation_ready": int(bool(result.get("termination_state", {}).get(
            "generation_ready", 0))),
        "response_delivery": int(delivery is not None),
        "response_surface_sha256": hashlib.sha256(
            "".join(chr(value) for value in map(ord, result.get(
                "response_surface", ""))).encode("utf-8")).hexdigest(),
        "model_read_only": int(model_before == model_after),
        "source_text_rows": source_counts,
        "free_dialogue_complete": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n", encoding="ascii", newline="\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_conformance(
        bundle_root=args.bundle_root,
        model_root=args.model_root,
        session_path=args.session,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
