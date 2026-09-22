"""审计自由对话训练前/发布前的三图消费资格。

This is a read-only, standard-library-only audit.  It binds the latest heldout
reference receipt to the exact model SQLite and checks that every recorded
query used the three graph spaces, O/H/E, graph traversal state, and trained
generation.  A passing result is deliberately *not* a free-dialogue claim;
the claim remains zero until broad training and independent release succeed.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys


FORMAT = "PURE_INTEGER_FREE_DIALOGUE_QUALIFICATION_AUDIT_V1"
EXPECTED_SPACES = (1, 2, 3)
GRAPH_FILLER_KINDS = frozenset({6, 7, 16, 17})
REQUIRED_RECEIPT = (
    "cold_restore_equal", "core_training_performed", "delivery_count",
    "model_read_only", "reference_candidate_set_count",
    "reference_resolution_count", "reference_resolved_count",
    "role_occurrence_count", "queries",
)
FORBIDDEN_ROUTE_KEYS = (
    "source_body_answer_route", "successor_answer_route",
    "character_nearest_route", "opencc_route", "language_vocabulary_route",
)


class QualificationAuditError(ValueError):
    """资格收据或训练谱系不闭合。"""


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationAuditError(f"{label} 不可回读") from error
    if not isinstance(value, dict):
        raise QualificationAuditError(f"{label} 必须是 object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise QualificationAuditError(f"{label} 必须是整数")
    return value


def _walk_route_keys(value: object, path: str = "root") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key in FORBIDDEN_ROUTE_KEYS and item not in (0, False, None):
                found.append(f"{path}.{key}")
            found.extend(_walk_route_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_route_keys(item, f"{path}[{index}]"))
    return tuple(found)


def _trace(path: Path) -> dict[str, object]:
    try:
        with gzip.open(path, "rt", encoding="ascii", newline="") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationAuditError(f"trace 不可回读: {path.name}") from error
    if isinstance(value, list):
        # The runtime's compact integer trace is a positional projection.  It
        # keeps the same QueryState fields as the object form while avoiding
        # non-integer protocol labels in the persisted payload.
        if len(value) < 17:
            raise QualificationAuditError(f"整数 trace 字段不完整: {path.name}")
        # Preserve the positional payload length before replacing it with the
        # normalized object.  The graph-input witness fields live at slots
        # 17..22; checking ``len(value)`` after replacement silently dropped
        # those fields and made a real consumed input appear absent.
        compact = value
        compact_len = len(compact)
        value = {
            "active_spaces": compact[1],
            "roots": compact[2],
            "frontier_trace": compact[4],
            "bindings": compact[5],
            "evidence": compact[6],
            "visited": compact[7],
            "response_plan": compact[12] or None,
            "generation": compact[13] or None,
            # Held-out runtime generation is a graph-slot connector act;
            # this marker is derived from the trace, never supplied by text.
            "heldout_match": int(
                compact[21] if compact_len >= 22
                else (isinstance(compact[13], dict)
                      and compact[13].get("kind") == 3)),
            "core_precedence": int(compact[22]) if compact_len >= 23 else 0,
        }
        if compact_len >= 20:
            value.update({
                "discourse_graph_input_requested_keys": compact[17],
                "discourse_graph_input_keys": compact[18],
                "event_time_graph_input_keys": compact[19],
            })
        if compact_len >= 21:
            value["core_graph_input_keys"] = compact[20]
    if not isinstance(value, dict):
        raise QualificationAuditError(f"trace 必须是 object: {path.name}")
    bad = _walk_route_keys(value)
    if bad:
        raise QualificationAuditError(f"trace 启用了禁止 route: {bad[0]}")
    query = value.get("query", value)
    if not isinstance(query, dict):
        raise QualificationAuditError(f"trace query 必须是 object: {path.name}")
    if query.get("active_spaces") != list(EXPECTED_SPACES):
        raise QualificationAuditError(f"trace 未并行使用三图: {path.name}")
    # QueryState serializes the bounded frontier as ``frontier_trace``; keep
    # the audit aligned with that compact trace contract.
    for field in ("roots", "frontier_trace", "bindings", "evidence", "visited"):
        if not isinstance(query.get(field), list):
            raise QualificationAuditError(f"trace 缺少 {field}: {path.name}")
    root_spaces = {
        item.get("space") for item in query["roots"]
        if isinstance(item, dict)
    }
    if root_spaces != set(EXPECTED_SPACES):
        raise QualificationAuditError(f"trace roots 未覆盖三图: {path.name}")
    evidence = query["evidence"]
    spaces = {item.get("space") for item in evidence if isinstance(item, dict)}
    if (not {2, 3} <= spaces
            or not spaces <= set(EXPECTED_SPACES)):
        raise QualificationAuditError(
            f"trace evidence 未保留 Memory/Dialogue: {path.name}")
    if ((query.get("generation") is not None
         and not isinstance(query.get("generation"), dict))
            or (query.get("response_plan") is not None
                and not isinstance(query.get("response_plan"), dict))):
        raise QualificationAuditError(f"trace generation/plan 编码无效: {path.name}")
    return query


def _trace_composed_graph_slot(
        query: dict[str, object], *, minimum_slots: int = 1,
        ) -> bool:
    """Require a visible semantic graph filler in the selected heldout plan."""
    if type(minimum_slots) is not int or minimum_slots <= 0:
        raise QualificationAuditError("minimum graph slots 必须是正整数")
    if query.get("heldout_match") != 1 and query.get("core_precedence") != 1:
        return False
    generation = query.get("generation")
    response_plan = query.get("response_plan")
    if not isinstance(generation, dict) or not isinstance(response_plan, dict):
        return False
    representations = generation.get("representations")
    slots = response_plan.get("slot_sequence")
    if (generation.get("kind") != 3
            or not isinstance(representations, list)
            or len(representations) < 3
            or not isinstance(slots, list)):
        return False
    # The graph-slot connector emits language-atom representations at the
    # final ResponsePlan boundary.  The semantic graph role is therefore
    # carried by the connector/generation trace, not by the slot filler kind
    # (which is deliberately a surface atom).  A matched kind-3 generation
    # with the required response slots is the compact integer witness.
    return len(slots) >= minimum_slots and all(
        isinstance(slot, dict)
        and isinstance(slot.get("filler"), list)
        and bool(slot["filler"])
        and isinstance(slot.get("filler_surface_values"), list)
        and bool(slot["filler_surface_values"])
        and slot.get("required") == 1
        for slot in slots
    )


def _stdlib_dependency_audit(source_root: Path) -> dict[str, object]:
    """Reject external imports reachable from the production mainline.

    Historical source-pack adapters remain discoverable as an isolated count;
    they are not allowed to become a runtime dependency merely by existing in
    the repository.
    """
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    package = "pure_integer_ai"
    files = {
        ".".join(path.relative_to(source_root).with_suffix("").parts): path
        for path in source_root.rglob("*.py")
        if path.relative_to(source_root).parts
    }
    graph: dict[str, set[str]] = {}
    parse_errors: list[str] = []
    imports_by_module: dict[str, set[str]] = {}
    for module, path in sorted(files.items()):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            parse_errors.append(f"{path}:{error}")
            continue
        imported: set[str] = set()
        package_parts = module.split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package_parts[:len(package_parts) - node.level + 1]
                    imported.add(".".join(base + ([node.module] if node.module else [])))
                elif node.module:
                    imported.add(node.module)
        imports_by_module[module] = imported
        graph[module] = {
            item if item in files else item.rsplit(".", 1)[0]
            for item in imported
            if item in files or item.rsplit(".", 1)[0] in files
        }
    roots = (
        "pure_integer_ai.experiments.trained_graph_query_bridge",
        "pure_integer_ai.experiments.trained_generation_connector_runtime",
        "pure_integer_ai.experiments.trained_dialogue_session_projection",
        "pure_integer_ai.experiments.run_unknown_input_structure_slice",
        "pure_integer_ai.experiments.run_open_role_memory_slice",
        "pure_integer_ai.experiments.formal_train",
        "pure_integer_ai.experiments.materialize_generic_response_generation",
        "pure_integer_ai.experiments.trained_graph_release",
    )
    reachable: set[str] = set()
    pending = [item for item in roots if item in files]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(graph.get(current, ()))
    mainline_external: list[str] = []
    isolated_external: list[str] = []
    for module in sorted(files):
        for name in sorted(imports_by_module.get(module, ())):
            root = name.split(".", 1)[0]
            if not root or root in stdlib or root == package:
                continue
            target = f"{module}:{root}"
            (mainline_external if module in reachable else isolated_external).append(target)
    return {
        "mainline_modules": len(reachable),
        "external_imports": sorted(set(mainline_external)),
        "isolated_external_imports": sorted(set(isolated_external)),
        "parse_errors": parse_errors,
        "stdlib_only": int(not mainline_external and not parse_errors),
    }


def audit(*, database: str | Path, heldout_root: str | Path,
          source_root: str | Path | None = None,
          hash_model: bool = True) -> dict[str, object]:
    model = Path(database).resolve(strict=True)
    heldout = Path(heldout_root).resolve(strict=True)
    if model.drive.upper() != "K:" or heldout.drive.upper() != "K:":
        raise QualificationAuditError("model 与 heldout 必须位于 K 盘")
    receipt = _read_json(heldout / "receipt.json", "heldout receipt")
    modern = receipt.get("protocol") == 1
    if modern:
        query_count = _require_int(
            receipt.get("queried_user_count"),
            "queried_user_count", minimum=1)
        before_sha = receipt.get("canonical_model_sha256_before")
        after_sha = receipt.get("canonical_model_sha256_after")
        after_ohe = receipt.get("after_ohe")
        checks = {
            "three_graph_query_state": (
                receipt.get("active_spaces") == list(EXPECTED_SPACES)
                and receipt.get("three_graph_query_count") == query_count),
            "cold_restore": receipt.get("cold_restore_equal") == 1,
            "model_read_only": receipt.get("model_read_only") == 1,
            "canonical_unchanged": (
                isinstance(before_sha, str) and before_sha == after_sha),
            "database_not_copied": receipt.get("database_copy_count") == 0,
            "source_text_not_added": receipt.get("source_text_rows_added") == 0,
            "source_body_not_used": receipt.get("source_body_answer_route") == 0,
            "successor_not_used": receipt.get("successor_answer_route") == 0,
            "character_nearest_not_used": receipt.get("character_nearest_route") == 0,
            "opencc_not_used": receipt.get("opencc_route") == 0,
            "language_vocabulary_not_used": receipt.get("language_vocabulary_route") == 0,
            "delivery": _require_int(
                receipt.get("delivery_count"),
                "delivery_count", minimum=1) <= query_count,
            "heldout_graph_slot_consumed": _require_int(
                receipt.get("heldout_match_count"),
                "heldout_match_count", minimum=0) >= 1,
            "heldout_connector_owner_verified": (
                receipt.get("heldout_consumption") == 1
                and receipt.get("heldout_owner_verified") == 1),
            "ohe_persisted": (
                isinstance(after_ohe, list) and len(after_ohe) == 3
                and all(type(value) is int and value > 0
                        for value in after_ohe)),
            "integer_trace": receipt.get("integer_trace") == 1,
            "normalized_graph_slot_provenance": (
                receipt.get("semantic_variant_registry_normalized") == 1
                and receipt.get(
                    "heldout_whole_sentence_representation_count") == 0
                and isinstance(
                    receipt.get("heldout_graph_role_categories"), list)
                and bool(receipt["heldout_graph_role_categories"])
                and receipt["heldout_graph_role_categories"]
                == sorted(set(receipt["heldout_graph_role_categories"]))
                and all(type(value) is int and 1 <= value <= 5
                        for value in receipt[
                            "heldout_graph_role_categories"])),
            "not_claimed_complete": receipt.get("free_dialogue_complete") == 0,
        }
        graph_input_count = receipt.get("graph_input_count", 0)
        if graph_input_count:
            graph_input_count = _require_int(
                graph_input_count, "graph_input_count", minimum=1)
            checks["explicit_graph_inputs_consumed"] = (
                _require_int(
                    receipt.get("consumed_graph_input_count"),
                    "consumed_graph_input_count", minimum=0)
                == graph_input_count)
        trace_paths = tuple(sorted(heldout.glob("query-*.json.gz")))
        trace_names = tuple(path.name for path in trace_paths)
        if len(trace_paths) != query_count:
            raise QualificationAuditError("session trace 数量与查询数不一致")
    else:
        if receipt.get("schema_version") != 1:
            raise QualificationAuditError("heldout receipt schema 不兼容")
        for field in REQUIRED_RECEIPT:
            if field not in receipt:
                raise QualificationAuditError(f"heldout receipt 缺少 {field}")
        checks = {
            "three_graph_query_state": receipt.get("active_spaces_contract", [1, 2, 3]) == [1, 2, 3],
            "cold_restore": receipt.get("cold_restore_equal") == 1,
            "model_read_only": receipt.get("model_read_only") == 1,
            "core_not_trained_during_probe": receipt.get("core_training_performed") == 0,
            "delivery": _require_int(receipt["delivery_count"], "delivery_count", minimum=2) >= 2,
            "role_occurrences": _require_int(receipt["role_occurrence_count"], "role_occurrence_count", minimum=2) >= 2,
            "reference_candidates": _require_int(receipt["reference_candidate_set_count"], "reference_candidate_set_count", minimum=2) >= 2,
            "reference_resolution": _require_int(receipt["reference_resolution_count"], "reference_resolution_count", minimum=2) >= 2,
            "reference_resolved": _require_int(receipt["reference_resolved_count"], "reference_resolved_count", minimum=1) >= 1,
            "not_claimed_complete": receipt.get("free_dialogue_complete") == 0,
        }
        trace_names = tuple(item.get("trace") for item in receipt["queries"] if isinstance(item, dict))
        if len(trace_names) != len(receipt["queries"]) or any(not isinstance(name, str) for name in trace_names):
            raise QualificationAuditError("queries 缺少 trace 名称")
        trace_paths = tuple(heldout / name for name in trace_names)
    traces = tuple(_trace(path) for path in trace_paths)
    checks["trace_count"] = len(traces) >= 2
    # Unknown/non-claim turns deliberately carry evidence only from
    # Interaction Memory and Dialogue.  Core evidence must remain absent;
    # requiring all three spaces here would reject the intended fail-closed
    # unknown path even though the QueryState itself retains all three roots.
    trace_evidence_spaces = {
        item.get("space")
        for trace in traces
        for item in trace.get("evidence", ())
        if isinstance(item, dict)
    }
    checks["trace_evidence_spaces"] = (
        {2, 3} <= trace_evidence_spaces <= set(EXPECTED_SPACES))
    if modern:
        checks["trace_heldout_match"] = sum(
            int(item.get("heldout_match", 0))
            + int(item.get("core_precedence", 0))
            for item in traces) >= len(traces)
        minimum_slots = _require_int(
            receipt.get("heldout_minimum_graph_slots_per_variant", 1),
            "heldout_minimum_graph_slots_per_variant", minimum=1)
        checks["trace_composed_graph_slot"] = any(
            _trace_composed_graph_slot(item, minimum_slots=minimum_slots)
            for item in traces)
        if receipt.get("graph_input_count", 0):
            requested_count = 0
            consumed_count = 0
            trace_graph_inputs_closed = True
            trace_generation_graph_inputs_closed = True
            for item in traces:
                requested = {
                    tuple(key) for key in item.get(
                        "discourse_graph_input_requested_keys", ())
                    if isinstance(key, list)
                }
                consumed = {
                    tuple(key) for key in item.get(
                        "discourse_graph_input_keys", ())
                    if isinstance(key, list)
                }
                consumed.update(
                    tuple(key) for key in item.get(
                        "event_time_graph_input_keys", ())
                    if isinstance(key, list)
                )
                consumed.update(
                    tuple(key) for key in item.get(
                        "core_graph_input_keys", ())
                    if isinstance(key, list)
                )
                requested_count += len(requested)
                consumed_count += len(requested & consumed)
                trace_graph_inputs_closed &= bool(requested) and requested <= consumed
                response_plan = item.get("response_plan")
                planned = set()
                if isinstance(response_plan, dict):
                    planned.update(
                        tuple(key) for field in (
                            "claim_refs", "event_refs", "discourse_links")
                        for key in response_plan.get(field, ())
                        if isinstance(key, list)
                    )
                    planned.update(
                        tuple(slot["filler"])
                        for slot in response_plan.get("slot_sequence", ())
                        if (isinstance(slot, dict)
                            and isinstance(slot.get("filler"), list))
                    )
                trace_generation_graph_inputs_closed &= (
                    bool(requested) and requested <= planned
                    and (item.get("heldout_match") == 1
                         or item.get("core_precedence") == 1))
            checks["trace_explicit_graph_inputs_consumed"] = (
                trace_graph_inputs_closed
                and requested_count == receipt["graph_input_count"]
                and consumed_count == receipt["graph_input_count"])
            if "generation_consumed_graph_input_count" in receipt:
                checks["explicit_graph_inputs_consumed_by_generation"] = (
                    _require_int(
                        receipt.get("generation_consumed_graph_input_count"),
                        "generation_consumed_graph_input_count",
                        minimum=0,
                    ) == receipt["graph_input_count"]
                    and trace_generation_graph_inputs_closed)
    else:
        checks["trace_reference_evidence"] = any(
            sum(item.get("reference_evidence_by_dimension", [0, 0, 0, 0, 0])[-1:]) > 0
            for item in receipt["queries"] if isinstance(item, dict))
    dependency = _stdlib_dependency_audit(Path(source_root).resolve() if source_root else Path(__file__).resolve().parents[2])
    checks["stdlib_only"] = dependency["stdlib_only"] == 1
    model_sha = _sha256(model) if hash_model else None
    result = {
        "format": FORMAT, "schema_version": 1,
        "model_path": model.name, "model_bytes": model.stat().st_size,
        "model_sha256": model_sha,
        "heldout_root": heldout.name,
        "trace_names": list(trace_names),
        "checks": checks,
        "dependency": dependency,
        "free_dialogue_complete": 0,
        "qualification_status": "PASS_HELDOUT_CONSUMPTION_ONLY" if all(checks.values()) else "BLOCKED",
        "broad_training_required": 1,
        "independent_release_required": 1,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--heldout-root", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-model-hash", action="store_true")
    args = parser.parse_args(argv)
    result = audit(database=args.database, heldout_root=args.heldout_root,
                   source_root=args.source_root, hash_model=not args.skip_model_hash)
    output = Path(args.output).resolve()
    if output.exists():
        raise QualificationAuditError("audit output 已存在，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True,
                                  separators=(",", ":")) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0 if result["qualification_status"] == "PASS_HELDOUT_CONSUMPTION_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
