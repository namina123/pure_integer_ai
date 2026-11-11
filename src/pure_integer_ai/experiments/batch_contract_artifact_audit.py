"""Audit an ended connector-contract batch without querying or restoring graphs.

The V1 runner persisted aggregate rows, not authoritative QueryState/plan/delivery
provenance. Consistent aggregates can never qualify language capability. This
reader does not import the runner, touch its databases, or repair its evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from pure_integer_ai.experiments.batch_run_recovery import ProcessProbe, probe_process


class BatchArtifactEvidenceError(ValueError):
    """Evidence cannot be read safely or has changed during the audit."""


_BASE = "PURE_INTEGER_RELATION_GENERATION_BATCH_"
_FORMATS = {
    "run_identity.json": _BASE + "PREFLIGHT_V1",
    "source_license.json": _BASE + "SOURCE_LICENSE_V1",
    "split_manifest.json": _BASE + "SPLIT_V1",
    "run_state.json": _BASE + "RUN_STATE_V1",
    "case_checkpoint.json": _BASE + "CASE_CHECKPOINT_V1",
    "preflight.json": _BASE + "PREFLIGHT_V1",
    "checkpoint.json": _BASE + "PREFLIGHT_V1",
    "trace.int.json": _BASE + "TRACE_V1",
    "failure.json": _BASE + "FAILURE_V1",
}
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _object(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise BatchArtifactEvidenceError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _key(value: object) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(type(item) is int and item >= 0 for item in value))


class _Snapshot:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest: dict[str, dict | None] = {}

    def read(self, name: str) -> dict | None:
        path = self.root / name
        if not path.exists():
            self.manifest[name] = None
            return None
        if path.is_symlink() or path.resolve().parent != self.root:
            raise BatchArtifactEvidenceError(f"artifact escapes run directory: {name}")
        if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
            raise BatchArtifactEvidenceError(f"artifact is not a bounded JSON file: {name}")
        data = path.read_bytes()
        if len(data) > _MAX_JSON_BYTES:
            raise BatchArtifactEvidenceError(f"artifact exceeds JSON read budget: {name}")
        self.manifest[name] = {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        }
        try:
            value = json.loads(data.decode("ascii"), object_pairs_hook=_object)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BatchArtifactEvidenceError(f"unreadable JSON artifact: {name}") from error
        if not isinstance(value, dict):
            raise BatchArtifactEvidenceError(f"artifact must be an object: {name}")
        return value

    def verify_unchanged(self) -> None:
        for name, prior in self.manifest.items():
            path = self.root / name
            if prior is None:
                if path.exists():
                    raise BatchArtifactEvidenceError(f"artifact appeared during audit: {name}")
            elif (not path.is_file() or path.is_symlink()
                  or path.resolve().parent != self.root
                  or path.stat().st_size != prior["bytes"]
                  or hashlib.sha256(path.read_bytes()).hexdigest() != prior["sha256"]):
                raise BatchArtifactEvidenceError(f"artifact changed during audit: {name}")


def inspect_completed_batch_contract(
        run_root: str | Path, *, owner_pids: tuple[int, ...],
        expected_covered_count: int, expected_uncovered_count: int,
        probe: Callable[[int], ProcessProbe] = probe_process) -> dict:
    """Check only the original V1 artifacts after independently observed owners exit.

    Freeze expected counts and owner identities before calling this function.
    LIVE/UNKNOWN owners reject the audit before any artifact/database is read.
    Missing files remain UNKNOWN; contradictions are FAIL. Neither outcome can
    be promoted to Wave 1, even if the legacy receipt called all cases closed.
    """
    if (type(expected_covered_count) is not int or not 1 <= expected_covered_count <= 64
            or type(expected_uncovered_count) is not int or expected_uncovered_count < 0):
        raise BatchArtifactEvidenceError("invalid frozen coverage expectation")
    if (not isinstance(owner_pids, tuple) or not owner_pids
            or any(type(pid) is not int or not 0 < pid <= 0xFFFFFFFF for pid in owner_pids)
            or len(set(owner_pids)) != len(owner_pids)):
        raise BatchArtifactEvidenceError("independently observed owner PIDs are required")
    owners = []
    for pid in owner_pids:
        result = probe(pid)
        if result.pid != pid or result.status != "EXITED":
            raise BatchArtifactEvidenceError(f"owner PID {pid} is not proven inactive")
        owners.append({"pid": pid, "status": result.status, "reason": result.reason})
    root = Path(run_root).resolve(strict=True)
    if not root.is_dir():
        raise BatchArtifactEvidenceError("run root must be a directory")
    snapshot = _Snapshot(root)
    artifacts = {name: snapshot.read(name) for name in _FORMATS}
    issues: list[str] = []
    for name, value in artifacts.items():
        if value is None:
            continue
        if value.get("format") != _FORMATS[name]:
            issues.append(f"{name}:format_drift")
        if name not in {"source_license.json", "trace.int.json"}:
            if value.get("identity") != root.name:
                issues.append(f"{name}:identity_drift")
        if type(value.get("free_dialogue_complete")) is not int \
                or value["free_dialogue_complete"] != 0:
            issues.append(f"{name}:free_dialogue_not_frozen_zero")
    identity = artifacts["run_identity.json"] or {}
    source = artifacts["source_license.json"] or {}
    split = artifacts["split_manifest.json"] or {}
    receipt = artifacts["preflight.json"] or {}
    if identity and source:
        for field in ("source_identity", "license_identity", "course_sha256"):
            if identity.get(field) != source.get(field):
                issues.append(f"source_license.json:{field}_drift")
        if source.get("license") != "CC0-1.0":
            issues.append("source_license.json:license_drift")
        for field in ("source_body_rows_added", "whole_sentence_or_answer_table"):
            if type(source.get(field)) is not int or source[field] != 0:
                issues.append(f"source_license.json:{field}_not_zero")
    if split:
        held_out = split.get("held_out")
        if (not isinstance(held_out, dict) or held_out.get("status")
                != "UNKNOWN_REUSED_ACTIVE_TRAINING_PROPOSITIONS"):
            issues.append("split_manifest.json:held_out_boundary_drift")
        for name in ("negative", "conflict"):
            boundary = split.get(name)
            if not isinstance(boundary, dict) or boundary.get("status") != "BOUNDED_NOT_RUN":
                issues.append(f"split_manifest.json:{name}_boundary_drift")
    if receipt:
        if identity and receipt.get("course_sha256") != identity.get("course_sha256"):
            issues.append("preflight.json:course_sha256_drift")
        for field, expected in (("production_not_started", 1),
                                ("core_training_performed", 1),
                                ("database_copy_count", 0),
                                ("source_body_rows_added", 0),
                                ("whole_sentence_or_answer_table", 0)):
            if type(receipt.get(field)) is not int or receipt[field] != expected:
                issues.append(f"preflight.json:{field}_drift")
    state = artifacts["run_state.json"] or {}
    for field in ("pid", "worker_pid"):
        if field in state and (type(state[field]) is not int or state[field] not in owner_pids):
            raise BatchArtifactEvidenceError(f"run-state {field} was not independently probed")

    checkpoint = artifacts["case_checkpoint.json"] or {}
    cases = checkpoint.get("cases", [])
    if (not isinstance(cases, list) or len(cases) > expected_covered_count
            or any(not isinstance(row, dict) for row in cases)):
        raise BatchArtifactEvidenceError("invalid bounded case checkpoint")
    case_numbers = [row.get("case") for row in cases]
    if (any(type(number) is not int for number in case_numbers)
            or case_numbers != list(range(1, len(cases) + 1))):
        issues.append("case_checkpoint.json:noncontiguous_prefix")
    if artifacts["case_checkpoint.json"] is not None:
        for field, expected in (("completed_case_count", len(cases)), ("next_case", len(cases) + 1)):
            if type(checkpoint.get(field)) is not int or checkpoint[field] != expected:
                issues.append(f"case_checkpoint.json:{field}_drift")

    rows = []
    propositions = []
    for number, row in enumerate(cases, 1):
        prefix = f"case_{number:03d}"
        for field in ("proposition_key", "operation_key", "query_values", "surface_values"):
            if not _key(row.get(field)) and not (field == "surface_values" and row.get(field) == []):
                issues.append(f"{prefix}:{field}_invalid")
        proposition = row.get("proposition_key")
        if _key(proposition):
            propositions.append(tuple(proposition))
        flag = row.get("factual_closed")
        if type(flag) is not int or flag not in {0, 1}:
            issues.append(f"{prefix}:factual_closed_invalid")
        if flag == 1:
            for field, expected in (("termination", 1), ("response_plan_present", 1),
                                    ("generation_present", 1), ("delivery_called", 1)):
                if type(row.get(field)) is not int or row[field] != expected:
                    issues.append(f"{prefix}:{field}_closure_drift")
            for field in ("claim_count", "claim_evidence_count"):
                if type(row.get(field)) is not int or row[field] <= 0:
                    issues.append(f"{prefix}:{field}_closure_drift")
            if not _key(row.get("surface_values")):
                issues.append(f"{prefix}:closed_without_output")
            for field in ("operation_keys_inferred", "operation_keys_consumed"):
                if not isinstance(row.get(field), list) or row.get("operation_key") not in row[field]:
                    issues.append(f"{prefix}:{field}_operation_drift")
        config_name = f"case_worker_{number:03d}.json"
        result_name = f"case_result_{number:03d}.json"
        config, worker_result = snapshot.read(config_name), snapshot.read(result_name)
        for name, value, expected_format in (
                (config_name, config, _BASE + "CASE_WORKER_V1"),
                (result_name, worker_result, _BASE + "CASE_RESULT_V1")):
            if value is not None and (value.get("format") != expected_format
                    or value.get("identity") != root.name
                    or type(value.get("case")) is not int or value["case"] != number
                    or type(value.get("free_dialogue_complete")) is not int
                    or value["free_dialogue_complete"] != 0):
                issues.append(f"{name}:envelope_drift")
        if config is not None:
            if any(config.get(field) != row.get(field) for field in ("proposition_key", "operation_key")):
                issues.append(f"{config_name}:target_drift")
            cue, query = config.get("operation_cue"), row.get("query_values")
            if not _key(cue) or not _key(query) or query[:len(cue) + 1] != cue + [65306]:
                issues.append(f"{config_name}:query_cue_drift")
        if worker_result is not None:
            raw_row = {key: value for key, value in row.items()
                       if key not in {"memory_database", "memory_scope"}}
            if worker_result.get("result") != raw_row:
                issues.append(f"{result_name}:checkpoint_result_drift")
        rows.append({
            "case": number, "reported_factual_closed": flag,
            "worker_result_consistency": "CHECKED" if worker_result is not None else "UNKNOWN_missing",
            "target_claim_in_plan": "UNKNOWN_not_saved_by_v1_runner",
            "query_state_three_graph_ohe": "UNKNOWN_not_saved_by_v1_runner",
            "delivery_proof": "UNKNOWN_not_saved_by_v1_runner",
            "legacy_reported_core_hop_count": row.get("core_hop_count"),
            "legacy_reported_memory_route_count": row.get("memory_route_count"),
            "qualified_core_hop_count": None, "qualified_memory_route_count": None,
        })
    if len(set(propositions)) != len(propositions):
        issues.append("case_checkpoint.json:duplicate_covered_proposition")

    closed_count = sum(type(row.get("factual_closed")) is int and row["factual_closed"] == 1 for row in cases)
    final = artifacts["checkpoint.json"] or {}
    if receipt:
        for field, expected in (("positive_case_count", len(cases)), ("covered_fact_count", len(cases)),
                                ("positive_closed_count", closed_count),
                                ("uncovered_fact_count", expected_uncovered_count),
                                ("active_fact_count", expected_covered_count + expected_uncovered_count)):
            if type(receipt.get(field)) is not int or receipt[field] != expected:
                issues.append(f"preflight.json:{field}_drift")
        if receipt.get("cases") != cases or len(cases) != expected_covered_count:
            issues.append("preflight.json:coverage_checkpoint_drift")
        expected_status = ("PASS_BATCH_CONNECTOR_QUERY_DELIVERY" if closed_count == len(cases) and cases
                           else "BLOCKED_BATCH_QUERY")
        if receipt.get("status") != expected_status:
            issues.append("preflight.json:status_drift")
        uncovered = receipt.get("uncovered_facts")
        if (not isinstance(uncovered, list) or len(uncovered) != expected_uncovered_count
                or any(not isinstance(row, dict) or not _key(row.get("proposition_key"))
                       or row.get("status") != "UNKNOWN_no_materialized_connector" for row in uncovered)):
            issues.append("preflight.json:uncovered_boundary_drift")
        else:
            keys = [tuple(row["proposition_key"]) for row in uncovered]
            if len(set(keys)) != len(keys) or set(keys) & set(propositions):
                issues.append("preflight.json:covered_uncovered_overlap")
    if final:
        saved = snapshot.manifest["preflight.json"]
        if saved is not None and final.get("preflight_sha256") != saved["sha256"]:
            issues.append("checkpoint.json:preflight_sha256_drift")
        if receipt and final.get("stage") != receipt.get("status"):
            issues.append("checkpoint.json:stage_drift")
    aggregate_trace = artifacts["trace.int.json"]
    if aggregate_trace is not None and aggregate_trace.get("cases") != cases:
        issues.append("trace.int.json:aggregate_case_drift")
    missing = [name for name in _FORMATS if name != "failure.json" and artifacts[name] is None]
    if not missing and state.get("status") != "COMPLETE":
        issues.append("run_state.json:nonterminal_final_bundle")
    failure = artifacts["failure.json"] or {}
    state_epoch = state.get("updated_epoch_ns")
    failure_epoch = failure.get("updated_epoch_ns")
    current_failure = (
        state.get("status") == "CASE_WORKER_FAILED"
        and type(state_epoch) is int and type(failure_epoch) is int
        and failure_epoch >= state_epoch
        and failure.get("stage") == "queries"
        and failure.get("identity") == root.name
    )
    snapshot.verify_unchanged()
    return {
        "format": "PURE_INTEGER_BATCH_CONTRACT_ARTIFACT_AUDIT_V1",
        "identity": root.name, "run_root": str(root),
        "status": ("FAIL_ARTIFACT_CONTRADICTION" if issues else "FAIL_CURRENT_CASE_WORKER" if current_failure
                   else "UNKNOWN_INCOMPLETE_ARTIFACTS" if missing else "CONSISTENT_AGGREGATES_ONLY"),
        "scope": "V1 connector contract aggregates; no query replay",
        "owners": owners, "artifact_sha256": snapshot.manifest,
        "missing_artifacts": missing, "issues": issues,
        "expected_covered_count": expected_covered_count,
        "expected_uncovered_count": expected_uncovered_count,
        "completed_case_count": len(cases), "reported_closed_count": closed_count,
        "case_evidence": rows,
        "failure_attempt_qualification": (
            "CURRENT_FAILURE_CORRELATED_WITH_TERMINAL_STATE" if current_failure
            else "UNKNOWN_no_durable_attempt_identity"),
        "novelty": "UNKNOWN_reuses_active_training_propositions",
        "negative": "UNKNOWN_not_run_by_contract_fixture",
        "conflict": "UNKNOWN_not_run_by_contract_fixture",
        "operation_behavior": "UNKNOWN_single_gate_operation",
        "resource_qualification": "UNKNOWN_requires_separate_metadata_and_runtime_evidence",
        "full_chain_provenance": "UNKNOWN_not_saved_by_v1_runner",
        "wave1_qualification": "UNKNOWN_not_a_language_qualification_run",
        "free_dialogue_complete": 0,
    }
