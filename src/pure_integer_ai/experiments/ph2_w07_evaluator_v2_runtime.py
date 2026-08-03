"""W-07 evaluator v2：细粒度游标、持久 ledger 与一次性安全封存。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_FORMAL_RUN_ID,
    W07_GENERATION_ABLATION_KEY,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w07_evaluator import evaluate_w07_case
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07_EVALUATOR_FAILURE_PHASES,
    W07_EVALUATOR_PHASES,
    W07_PRIVATE_CASE_NAME,
    W07_PRIVATE_CLUSTER_NAME,
    W07_PRIVATE_FAMILY_FREEZE_NAME,
    W07_PRIVATE_HARD_REQUIREMENTS,
    W07_PRIVATE_LABEL_NAME,
    W07_PRIVATE_SCHEMA_NAME,
    W07_PRIVATE_SOURCE_NAME,
    W07PrivateEvaluationError,
    decode_w07_private_documents,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_runtime import (
    W07EvaluatorInjectedFault,
    W07EvaluatorInfrastructureError,
    W07PrivateEvaluatorRunResult,
    _build_consumer_suite,
    _candidate_config,
    _candidate_documents,
    _dump_root,
    _read_canonical,
    _sha256,
    _strict_sha1,
    _tree_digest,
    _validate_roots,
    _write_exclusive,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_v2_contract import (
    W07_V2_EVALUATOR_VERSION,
    W07_V2_FAILURE_KINDS,
    W07_V2_NONE,
    W07_V2_OPERATIONS,
    W07_V2_PRIVATE_AGGREGATE_NAME,
    W07_V2_PRIVATE_RECOMMENDATION_NAME,
    W07V2AblationProgress,
    W07V2DiagnosticCursor,
    public_safe_w07_v2_aggregate,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_v2_family import (
    consume_w07_v2_private_first_run_guard,
)
from pure_integer_ai.experiments.ph2_w07_runtime import load_w07_public_dump


_ABLATION_PHASES = dict(zip(
    W07_PUBLIC_ABLATION_KEYS,
    W07_EVALUATOR_PHASES[5:13],
    strict=True,
))


@dataclass(frozen=True)
class W07V2PrivateEvaluatorRuntimeConfig:
    repository_root: str | Path
    candidate_root: str | Path
    family_root: str | Path
    execution_root: str | Path
    evaluator_public_head_commit_sha1: str
    fault_cursor: W07V2DiagnosticCursor | None = None

    def __post_init__(self) -> None:
        if (self.fault_cursor is not None
                and not isinstance(self.fault_cursor, W07V2DiagnosticCursor)):
            raise TypeError("W-07 v2 fault cursor type drift")


def _family_documents_v2(
        family_root: Path,
        family_freeze_sha256: str,
        evaluator_head: str,
        ) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    freeze, freeze_bytes = _read_canonical(
        family_root / W07_PRIVATE_FAMILY_FREEZE_NAME,
        label="private v2 family freeze")
    expected_fields = {
        "ablation_order", "artifact_kind", "candidate_contract_sha256",
        "candidate_host_freeze_sha256", "case_commitment",
        "cluster_commitment", "diagnostic_contract",
        "evaluator_public_head_commit_sha1", "evaluator_version",
        "family_key", "file_inventory", "formal_run_count",
        "format_version", "hard_requirements", "label_commitment",
        "owner_key", "payload_commitment", "self_excluded",
    }
    expected_diagnostic = {
        "dimension_order": list(W07_PUBLIC_DIMENSION_KEYS),
        "failure_kinds": list(W07_V2_FAILURE_KINDS),
        "failure_phases": list(W07_EVALUATOR_FAILURE_PHASES),
        "operations": list(W07_V2_OPERATIONS),
    }
    if (_sha256(freeze_bytes) != family_freeze_sha256
            or set(freeze) != expected_fields
            or freeze.get("artifact_kind")
            != "PH2_W07_PRIVATE_FAMILY_V2_FREEZE"
            or freeze.get("format_version") != 2
            or freeze.get("evaluator_version") != W07_V2_EVALUATOR_VERSION
            or freeze.get("formal_run_count") != 0
            or freeze.get("self_excluded") != 1
            or freeze.get("ablation_order") != list(W07_PUBLIC_ABLATION_KEYS)
            or freeze.get("hard_requirements")
            != list(W07_PRIVATE_HARD_REQUIREMENTS)
            or freeze.get("diagnostic_contract") != expected_diagnostic
            or freeze.get("evaluator_public_head_commit_sha1")
            != evaluator_head):
        raise W07EvaluatorInfrastructureError(
            "private v2 family freeze drift")
    names = (
        W07_PRIVATE_SOURCE_NAME,
        W07_PRIVATE_SCHEMA_NAME,
        W07_PRIVATE_CASE_NAME,
        W07_PRIVATE_LABEL_NAME,
        W07_PRIVATE_CLUSTER_NAME,
    )
    inventory = freeze.get("file_inventory")
    if not isinstance(inventory, list) or len(inventory) != len(names):
        raise W07EvaluatorInfrastructureError(
            "private v2 family inventory drift")
    by_name = {
        item.get("path"): item for item in inventory if isinstance(item, dict)
    }
    if set(by_name) != set(names):
        raise W07EvaluatorInfrastructureError(
            "private v2 family inventory names drift")
    payloads = []
    for name in names:
        path = family_root / name
        if not path.is_file() or path.is_symlink():
            raise W07EvaluatorInfrastructureError(
                "private v2 family file missing")
        payload = path.read_bytes()
        row = by_name[name]
        if (set(row) != {"path", "sha256", "size_bytes"}
                or row.get("sha256") != _sha256(payload)
                or row.get("size_bytes") != len(payload)):
            raise W07EvaluatorInfrastructureError(
                "private v2 family file identity drift")
        payloads.append(payload)
    return freeze, tuple(payloads)


def _enter(
        config: W07V2PrivateEvaluatorRuntimeConfig,
        cursor: W07V2DiagnosticCursor,
        ) -> None:
    if config.fault_cursor == cursor:
        raise W07EvaluatorInjectedFault(cursor.operation)


def _failure_kind(error: Exception) -> str:
    if isinstance(error, W07EvaluatorInjectedFault):
        return "INJECTED"
    if isinstance(error, sqlite3.Error):
        return "STORAGE"
    if isinstance(error, (TimeoutError, RecursionError)):
        return "RESOURCE"
    if isinstance(error, MemoryError):
        return "MEMORY"
    if isinstance(error, (W07PrivateEvaluationError,
                          W07EvaluatorInfrastructureError)):
        return "DOMAIN_CONTRACT"
    if error.__class__.__module__.startswith("pure_integer_ai"):
        return "DOMAIN_CONTRACT"
    if isinstance(error, (TypeError, ValueError)):
        return "TYPE_VALUE"
    if isinstance(error, OSError):
        return "OS"
    return "UNEXPECTED"


def _ledger_commitment(rows: tuple[dict[str, int | str], ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(rows))).hexdigest()


def _safe_owner_audit(
        repository: Path,
        candidate: Path,
        label_path: Path,
        *,
        repository_before: str,
        candidate_before: str,
        label_before: str,
        ) -> tuple[int, int, int, int]:
    try:
        host_writes = int(_tree_digest(candidate) != candidate_before)
        label_writes = int(_sha256(label_path.read_bytes()) != label_before)
        public_writes = int(_tree_digest(repository) != repository_before)
        return host_writes, label_writes, public_writes, 1
    except Exception:
        return 1, 1, 1, 0


def _attach_infrastructure(
        aggregate: dict[str, object],
        *,
        baseline_audit: dict[str, int],
        ledger_audit: tuple[dict[str, int | str], ...],
        ledgers_committed: int,
        ledgers_closed: int,
        owner_audit_complete: int,
        clone_digest_match: int,
        clone_dump_readback: int,
        clone_host_copy_match: int,
        host_copy_unchanged: int,
        carrier_scope_ok: int,
        artifact_counts: dict[str, int],
        evaluator_head: str,
        ) -> None:
    aggregate["infrastructure"] = {
        "baseline_consumer_audit": baseline_audit,
        "candidate_inventory_match": clone_digest_match,
        "carrier_projection_count": artifact_counts.get(
            "CARRIER_PROJECTION", 0),
        "carrier_scope_digest_match": carrier_scope_ok,
        "clone_dump_readback": clone_dump_readback,
        "clone_host_copy_match": clone_host_copy_match,
        "evaluator_public_head_commit_sha1": evaluator_head,
        "host_copy_unchanged": host_copy_unchanged,
        "ledger_audit": list(ledger_audit),
        "ledger_audit_commitment": (
            _ledger_commitment(ledger_audit) if ledger_audit else "0" * 64),
        "ledger_count": len(ledger_audit),
        "ledgers_closed": ledgers_closed,
        "ledgers_committed": ledgers_committed,
        "logic_scope_cell_count": artifact_counts.get("LOGIC_SCOPE_CELL", 0),
        "logic_use_count": artifact_counts.get("LOGIC_USE", 0),
        "owner_audit_complete": owner_audit_complete,
    }


def _forbidden_tokens(private_payload) -> tuple[bytes, ...]:
    return (
        *(item.case_key.encode("utf-8") for item in private_payload.cases),
        *(item.label_key.encode("utf-8") for item in private_payload.labels),
        b"surface", b"expected", b"private" + b"_path", b"message",
    )


def run_w07_v2_private_evaluation_once(
        config: W07V2PrivateEvaluatorRuntimeConfig,
        *,
        family_freeze_sha256: str,
        ) -> W07PrivateEvaluatorRunResult:
    """消费一个全新 v2 guard，并发布 PASS、FAIL 或可定位 NE。"""
    if not isinstance(config, W07V2PrivateEvaluatorRuntimeConfig):
        raise TypeError("W-07 v2 evaluator config type drift")
    evaluator_head = _strict_sha1(
        config.evaluator_public_head_commit_sha1, label="evaluator HEAD")
    repository = Path(config.repository_root).resolve()
    candidate = Path(config.candidate_root).resolve()
    family_root = Path(config.family_root).resolve()
    execution = Path(config.execution_root).resolve()
    _validate_roots(repository, candidate, family_root, execution)
    contract, contract_bytes, host, host_bytes = _candidate_documents(
        repository, candidate)
    family, documents = _family_documents_v2(
        family_root, family_freeze_sha256, evaluator_head)
    if (family.get("candidate_contract_sha256") != _sha256(contract_bytes)
            or family.get("candidate_host_freeze_sha256")
            != _sha256(host_bytes)):
        raise W07EvaluatorInfrastructureError(
            "private v2 family candidate binding drift")

    candidate_before = _tree_digest(candidate)
    label_path = family_root / W07_PRIVATE_LABEL_NAME
    label_before = _sha256(label_path.read_bytes())
    repository_before = _tree_digest(repository)
    _guard_path, guard_sha = consume_w07_v2_private_first_run_guard(
        family_root, family_freeze_sha256=family_freeze_sha256)

    cursor = W07V2DiagnosticCursor(
        "PAYLOAD_DECODE", "ENTER_PHASE")
    private_payload = None
    suite = None
    baseline: list[Any] = []
    ablations: list[W07V2AblationProgress] = []
    current_ablation_key: str | None = None
    current_ablation: list[Any] = []
    gate_passes: list[bool] = []
    baseline_audit: dict[str, int] = {}
    ledger_audit: tuple[dict[str, int | str], ...] = ()
    ledgers_committed = 0
    ledgers_closed = 0
    clone_digest_match = 0
    clone_dump_readback = 0
    clone_ok = 0
    carrier_scope_ok = 0
    host_copy_unchanged = 0
    artifact_counts: dict[str, int] = {}
    host_copy: Path | None = None
    host_copy_sha = ""

    try:
        _enter(config, cursor)
        cursor = W07V2DiagnosticCursor(
            "PAYLOAD_DECODE", "DECODE_DOCUMENTS")
        _enter(config, cursor)
        private_payload = decode_w07_private_documents(*documents)
        if private_payload.evaluator_public_head_commit_sha1 != evaluator_head:
            raise W07EvaluatorInfrastructureError(
                "private v2 source evaluator HEAD drift")

        cursor = W07V2DiagnosticCursor(
            "CLONE_LOAD", "CREATE_EXECUTION_ROOT")
        _enter(config, cursor)
        execution.mkdir(parents=True, exist_ok=False)
        cursor = W07V2DiagnosticCursor("CLONE_LOAD", "LOAD_CLONE")
        _enter(config, cursor)
        dump_root = _dump_root(candidate)
        clone_outcome = load_w07_public_dump(_candidate_config(
            config,
            contract,
            run_root=dump_root,
            sqlite_path=execution / "clone.sqlite",
        ))
        clone_dump_readback = int(clone_outcome.dump_readback)

        cursor = W07V2DiagnosticCursor(
            "HOST_COPY", "COPY_HOST_MANIFEST")
        _enter(config, cursor)
        dump_path = dump_root / f"w07_run_{W07_FORMAL_RUN_ID:020d}" / (
            "w07_dump_manifest.json")
        host_copy = execution / "host_copy.dump"
        shutil.copyfile(dump_path, host_copy)
        dump_sha = _sha256(dump_path.read_bytes())
        host_copy_sha = _sha256(host_copy.read_bytes())
        clone_ok = int(dump_sha == host_copy_sha)

        cursor = W07V2DiagnosticCursor(
            "CLONE_COMPARE", "COMPARE_CLONE")
        _enter(config, cursor)
        host_digest = host["host_evidence"]["host_digests"]
        clone_digest_match = int(all((
            clone_outcome.logical_state_digest == host_digest["logical"],
            clone_outcome.candidate_digest == host_digest["candidate"],
            clone_outcome.logic_digest == host_digest["logic"],
            clone_outcome.source_evidence_digest == host_digest["source_evidence"],
            clone_outcome.active_projection_digest == host_digest["active_projection"],
            clone_outcome.carrier_scope_digest == host_digest["carrier_scope"],
            clone_outcome.transaction_digest == host_digest["transaction"],
        )))
        artifact_counts = dict(clone_outcome.artifact_counts)
        carrier_scope_ok = int(all((
            clone_outcome.carrier_scope_digest == host_digest["carrier_scope"],
            artifact_counts.get("CARRIER_PROJECTION") == 9,
            artifact_counts.get("LOGIC_SCOPE_CELL") == 189,
            artifact_counts.get("LOGIC_USE") == 21,
        )))
        if not clone_digest_match or not carrier_scope_ok:
            raise W07EvaluatorInfrastructureError(
                "candidate clone or carrier scope drift")

        cursor = W07V2DiagnosticCursor("BASELINE", "ENTER_PHASE")
        _enter(config, cursor)
        cursor = W07V2DiagnosticCursor("BASELINE", "BUILD_SUITE")
        _enter(config, cursor)
        suite, _ = _build_consumer_suite(config, contract, execution)
        cursor = W07V2DiagnosticCursor("BASELINE", "COMMIT_LEDGERS")
        _enter(config, cursor)
        suite.commit()
        ledgers_committed = 1
        cursor = W07V2DiagnosticCursor("BASELINE", "AUDIT_LEDGERS")
        _enter(config, cursor)
        ledger_audit = suite.ledger_audit()
        if (len(ledger_audit) != 7
                or any(item["table_count"] <= 0
                       or item["nonempty_table_count"] <= 0
                       or item["row_count"] <= 0
                       for item in ledger_audit)):
            raise W07EvaluatorInfrastructureError(
                "W-07 v2 committed ledger audit failed")

        for case in private_payload.cases:
            cursor = W07V2DiagnosticCursor(
                "BASELINE", "EVALUATE_CASE",
                dimension_key=case.dimension_key)
            _enter(config, cursor)
            baseline.append(evaluate_w07_case(
                suite, case, evaluation_ordinal=0))
        baseline_audit = suite.audit()

        for ordinal, key in enumerate(W07_PUBLIC_ABLATION_KEYS):
            phase = _ABLATION_PHASES[key]
            current_ablation_key = key
            current_ablation = []
            cursor = W07V2DiagnosticCursor(
                phase, "ENTER_PHASE", ablation_key=key)
            _enter(config, cursor)
            disabled = key.removesuffix("-ABLATION")
            for case in private_payload.cases:
                cursor = W07V2DiagnosticCursor(
                    phase,
                    "EVALUATE_CASE",
                    ablation_key=key,
                    dimension_key=case.dimension_key,
                )
                _enter(config, cursor)
                current_ablation.append(evaluate_w07_case(
                    suite,
                    case,
                    disabled_dimension=disabled,
                    evaluation_ordinal=ordinal + 1,
                ))
            cursor = W07V2DiagnosticCursor(
                phase, "ASSEMBLE_ABLATION", ablation_key=key)
            _enter(config, cursor)
            progress = W07V2AblationProgress(
                key, tuple(current_ablation))
            statuses = tuple(item.status for item in progress.results)
            expected = tuple(
                "FAIL" if index == ordinal else "PASS"
                for index in range(len(W07_PUBLIC_DIMENSION_KEYS)))
            gate_passes.append(
                statuses == expected
                and not any(item.ne_count for item in progress.results))
            ablations.append(progress)
            current_ablation_key = None
            current_ablation = []

        cursor = W07V2DiagnosticCursor("INTEGRITY", "CLOSE_LEDGERS")
        _enter(config, cursor)
        suite.close()
        suite = None
        ledgers_closed = 1
        host_copy_unchanged = int(
            host_copy is not None
            and _sha256(host_copy.read_bytes()) == host_copy_sha)

        cursor = W07V2DiagnosticCursor("INTEGRITY", "AUDIT_OWNERS")
        _enter(config, cursor)
        host_writes, label_writes, public_writes, owner_audit_complete = (
            _safe_owner_audit(
                repository,
                candidate,
                label_path,
                repository_before=repository_before,
                candidate_before=candidate_before,
                label_before=label_before,
            ))
        if (not owner_audit_complete
                or any((host_writes, label_writes, public_writes))
                or not host_copy_unchanged):
            raise W07EvaluatorInfrastructureError(
                "W-07 v2 evaluator owner isolation failed")

        cursor = W07V2DiagnosticCursor(
            "REPORT_SAFETY", "ASSEMBLE_REPORT")
        _enter(config, cursor)
        aggregate = public_safe_w07_v2_aggregate(
            tuple(baseline),
            tuple(ablations),
            family_commitment=family["family_key"],
            payload_commitment=family["payload_commitment"],
            case_commitment=family["case_commitment"],
            label_commitment=family["label_commitment"],
            cluster_commitment=family["cluster_commitment"],
            formal_run_count=1,
            host_writes=host_writes,
            label_writes=label_writes,
            public_repo_writes=public_writes,
            failure_kind=W07_V2_NONE,
            cursor=cursor,
            ablation_gates_passed=all(gate_passes),
        )
        _attach_infrastructure(
            aggregate,
            baseline_audit=baseline_audit,
            ledger_audit=ledger_audit,
            ledgers_committed=ledgers_committed,
            ledgers_closed=ledgers_closed,
            owner_audit_complete=owner_audit_complete,
            clone_digest_match=clone_digest_match,
            clone_dump_readback=clone_dump_readback,
            clone_host_copy_match=clone_ok,
            host_copy_unchanged=host_copy_unchanged,
            carrier_scope_ok=carrier_scope_ok,
            artifact_counts=artifact_counts,
            evaluator_head=evaluator_head,
        )
        generation_progress = next(
            item for item in ablations
            if item.ablation_key == W07_GENERATION_ABLATION_KEY)
        aggregate["generation_ablation_statuses"] = [
            item.status for item in generation_progress.results]

        cursor = W07V2DiagnosticCursor(
            "REPORT_SAFETY", "ENCODE_REPORT")
        _enter(config, cursor)
        aggregate["diagnostic_cursor"] = cursor.to_safe_dict()
        encoded = canonical_json_bytes(aggregate)
        if any(token in encoded for token in _forbidden_tokens(private_payload)):
            raise W07EvaluatorInfrastructureError(
                "W-07 v2 safe aggregate leaked forbidden fields")
        cursor = W07V2DiagnosticCursor(
            "REPORT_SAFETY", "PUBLISH_REPORT")
        _enter(config, cursor)
        aggregate["diagnostic_cursor"] = cursor.to_safe_dict()
        encoded = canonical_json_bytes(aggregate)
        aggregate_path = family_root / "publication" / (
            W07_V2_PRIVATE_AGGREGATE_NAME)
        aggregate_sha = _write_exclusive(aggregate_path, encoded)
        recommendation_path = None
        recommendation_sha = None
        if aggregate["status"] == "PASS" and all(gate_passes):
            recommendation = canonical_json_bytes({
                "aggregate_sha256": aggregate_sha,
                "artifact_kind": (
                    "PH2_W07_RUNTIME_RECEIPT_V2_RECOMMENDATION"),
                "candidate_contract_sha256": family[
                    "candidate_contract_sha256"],
                "candidate_host_freeze_sha256": family[
                    "candidate_host_freeze_sha256"],
                "evaluator_public_head_commit_sha1": evaluator_head,
                "evaluator_version": W07_V2_EVALUATOR_VERSION,
                "family_commitment": family["family_key"],
                "formal_run_count": 1,
                "format_version": 2,
                "recommend_runtime_receipt": 1,
            })
            recommendation_path = family_root / "publication" / (
                W07_V2_PRIVATE_RECOMMENDATION_NAME)
            recommendation_sha = _write_exclusive(
                recommendation_path, recommendation)
        return W07PrivateEvaluatorRunResult(
            str(aggregate["status"]),
            aggregate_path,
            aggregate_sha,
            recommendation_path,
            recommendation_sha,
            family_freeze_sha256,
            guard_sha,
        )
    except Exception as error:
        failure_kind = _failure_kind(error)
        if suite is not None:
            if not ledger_audit:
                try:
                    ledger_audit = suite.ledger_audit()
                except Exception:
                    ledger_audit = ()
            try:
                suite.close()
                ledgers_closed = 1
            except Exception:
                ledgers_closed = 0
            suite = None
        if host_copy is not None and host_copy.is_file():
            try:
                host_copy_unchanged = int(
                    _sha256(host_copy.read_bytes()) == host_copy_sha)
            except Exception:
                host_copy_unchanged = 0
        host_writes, label_writes, public_writes, owner_audit_complete = (
            _safe_owner_audit(
                repository,
                candidate,
                label_path,
                repository_before=repository_before,
                candidate_before=candidate_before,
                label_before=label_before,
            ))
        progress = list(ablations)
        if current_ablation_key is not None:
            progress.append(W07V2AblationProgress(
                current_ablation_key, tuple(current_ablation)))
        aggregate = public_safe_w07_v2_aggregate(
            tuple(baseline),
            tuple(progress),
            family_commitment=family["family_key"],
            payload_commitment=family["payload_commitment"],
            case_commitment=family["case_commitment"],
            label_commitment=family["label_commitment"],
            cluster_commitment=family["cluster_commitment"],
            formal_run_count=1,
            host_writes=host_writes,
            label_writes=label_writes,
            public_repo_writes=public_writes,
            failure_kind=failure_kind,
            cursor=cursor,
            ablation_gates_passed=False,
        )
        _attach_infrastructure(
            aggregate,
            baseline_audit=baseline_audit,
            ledger_audit=ledger_audit,
            ledgers_committed=ledgers_committed,
            ledgers_closed=ledgers_closed,
            owner_audit_complete=owner_audit_complete,
            clone_digest_match=clone_digest_match,
            clone_dump_readback=clone_dump_readback,
            clone_host_copy_match=clone_ok,
            host_copy_unchanged=host_copy_unchanged,
            carrier_scope_ok=carrier_scope_ok,
            artifact_counts=artifact_counts,
            evaluator_head=evaluator_head,
        )
        encoded = canonical_json_bytes(aggregate)
        if (private_payload is not None
                and any(token in encoded
                        for token in _forbidden_tokens(private_payload))):
            raise W07EvaluatorInfrastructureError(
                "W-07 v2 failure aggregate leaked forbidden fields")
        aggregate_path = family_root / "publication" / (
            W07_V2_PRIVATE_AGGREGATE_NAME)
        aggregate_sha = _write_exclusive(aggregate_path, encoded)
        return W07PrivateEvaluatorRunResult(
            "NE",
            aggregate_path,
            aggregate_sha,
            None,
            None,
            family_freeze_sha256,
            guard_sha,
        )


__all__ = [
    "W07V2PrivateEvaluatorRuntimeConfig",
    "run_w07_v2_private_evaluation_once",
]
