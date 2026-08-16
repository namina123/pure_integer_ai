"""Generic one-shot runtime for new manifest/plugin evaluation families."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Protocol

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorBoundaryContract,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.aggregate import (
    EvaluationAggregate,
    build_evaluation_aggregate,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    EvaluationOneShotGuard,
    EvaluationRunIntent,
    build_available_guard,
    consume_guard,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.manifest import (
    EvaluationKernelManifest,
    publish_evaluation_manifest,
    read_evaluation_manifest,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.identity import (
    evaluation_kernel_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginOutcome,
    EvaluationPluginRunContext,
    StageEvaluationPlugin,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.preflight import (
    EvaluationFormalReadyReceipt,
    assert_formal_ready_receipt,
    publish_formal_ready_receipt,
    read_formal_ready_receipt,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.private_io import (
    AuthorizedEvaluationFile,
    EvaluationFileIdentity,
    authorize_evaluation_files,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.publication import (
    EvaluationFailureSeal,
    EvaluationPublicationDecision,
    EvaluationRuntimeReceipt,
    build_failure_seal,
    build_publication_decision,
    build_runtime_receipt,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationDimensionResult,
    EvaluationKernelContractError,
    EvaluationResultSet,
    EvaluationRunAudit,
)


EVALUATION_FAMILY_MANIFEST = "family.manifest.json"
EVALUATION_GUARD_AVAILABLE = "guard.available.json"
EVALUATION_GUARD_CONSUMED = "guard.consumed.json"
EVALUATION_RUN_INTENT = "run.intent.json"
EVALUATION_AGGREGATE = "publication/aggregate.json"
EVALUATION_PUBLICATION_DECISION = "publication/decision.json"
EVALUATION_RUNTIME_RECEIPT = "publication/runtime_receipt.json"
EVALUATION_FAILURE_SEAL = "publication/failure_seal.json"
EVALUATION_FORMAL_READY_RECEIPT = "formal_ready.receipt.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _existing_family_root(value: str | Path) -> Path:
    original = Path(value)
    if original.is_symlink():
        raise EvaluationKernelContractError("evaluation family root symlink rejected")
    is_junction = getattr(original, "is_junction", None)
    if is_junction is not None and is_junction():
        raise EvaluationKernelContractError("evaluation family root junction rejected")
    root = original.resolve()
    if not root.is_dir():
        raise EvaluationKernelContractError("evaluation family root is missing")
    return root


def _new_family_root(value: str | Path) -> Path:
    original = Path(value)
    if original.exists() or original.is_symlink():
        raise EvaluationKernelContractError("evaluation family root must be new")
    parent = original.parent.resolve()
    if not parent.is_dir():
        raise EvaluationKernelContractError("evaluation family parent is missing")
    return (parent / original.name).resolve()


def _guard_at(root: Path, name: str) -> EvaluationOneShotGuard:
    return EvaluationOneShotGuard.from_dict(read_canonical_object(root / name))


def publish_evaluation_family(
        manifest: EvaluationKernelManifest,
        family_root: str | Path,
        ) -> Path:
    """Publish a new Git-external family manifest and its only available guard."""
    if not isinstance(manifest, EvaluationKernelManifest):
        raise EvaluationKernelContractError("evaluation family manifest type invalid")
    root = _new_family_root(family_root)
    root.mkdir()
    publish_evaluation_manifest(manifest, root / EVALUATION_FAMILY_MANIFEST)
    write_immutable_json(
        build_available_guard(manifest).to_dict(), root / EVALUATION_GUARD_AVAILABLE)
    if read_evaluation_family_manifest(root) != manifest:
        raise EvaluationKernelContractError("evaluation family manifest readback drifted")
    if _guard_at(root, EVALUATION_GUARD_AVAILABLE) != build_available_guard(manifest):
        raise EvaluationKernelContractError("evaluation family available guard drifted")
    return root


def read_evaluation_family_manifest(
        family_root: str | Path,
        ) -> EvaluationKernelManifest:
    """Read the family manifest from a real non-linked root."""
    root = _existing_family_root(family_root)
    return read_evaluation_manifest(root / EVALUATION_FAMILY_MANIFEST)


def consume_evaluation_family_guard(
        family_root: str | Path,
        manifest: EvaluationKernelManifest,
        ) -> EvaluationRunIntent:
    """Consume AVAILABLE before any decompressed private record is requested."""
    if not isinstance(manifest, EvaluationKernelManifest):
        raise EvaluationKernelContractError("evaluation guard manifest type invalid")
    return consume_evaluation_guard_once(
        family_root, build_available_guard(manifest))


def consume_evaluation_guard_once(
        family_root: str | Path,
        expected_available: EvaluationOneShotGuard,
        ) -> EvaluationRunIntent:
    """按共享 guard identity 原子占用任意 family 的唯一 formal run。"""
    root = _existing_family_root(family_root)
    if (not isinstance(expected_available, EvaluationOneShotGuard)
            or expected_available.state != "AVAILABLE"):
        raise EvaluationKernelContractError(
            "evaluation expected available guard invalid")
    available_path = root / EVALUATION_GUARD_AVAILABLE
    consumed_path = root / EVALUATION_GUARD_CONSUMED
    intent_path = root / EVALUATION_RUN_INTENT
    if (not available_path.is_file() or consumed_path.exists()
            or intent_path.exists()):
        raise EvaluationKernelContractError("evaluation family guard already consumed")
    available = _guard_at(root, EVALUATION_GUARD_AVAILABLE)
    if available != expected_available:
        raise EvaluationKernelContractError("evaluation available guard identity drifted")
    consumed, intent = consume_guard(available)
    write_immutable_json(consumed.to_dict(), consumed_path)
    write_immutable_json(intent.to_dict(), intent_path)
    available_path.unlink()
    if _guard_at(root, EVALUATION_GUARD_CONSUMED) != consumed:
        raise EvaluationKernelContractError("evaluation consumed guard readback drifted")
    if read_canonical_object(intent_path) != intent.to_dict():
        raise EvaluationKernelContractError("evaluation run intent readback drifted")
    return intent


def verify_evaluation_family_consumed(
        family_root: str | Path,
        manifest: EvaluationKernelManifest,
        ) -> EvaluationRunIntent:
    """Fail closed unless consumed guard and intent form one exact lineage."""
    if not isinstance(manifest, EvaluationKernelManifest):
        raise EvaluationKernelContractError("evaluation guard manifest type invalid")
    return verify_evaluation_guard_consumed(
        family_root, build_available_guard(manifest))


def verify_evaluation_guard_consumed(
        family_root: str | Path,
        expected_available: EvaluationOneShotGuard,
        ) -> EvaluationRunIntent:
    """回验任意 family 的 consumed guard 与 immutable intent 唯一 lineage。"""
    root = _existing_family_root(family_root)
    if (not isinstance(expected_available, EvaluationOneShotGuard)
            or expected_available.state != "AVAILABLE"):
        raise EvaluationKernelContractError(
            "evaluation expected available guard invalid")
    if (root / EVALUATION_GUARD_AVAILABLE).exists():
        raise EvaluationKernelContractError("evaluation available guard still exists")
    consumed = _guard_at(root, EVALUATION_GUARD_CONSUMED)
    expected_consumed, expected_intent = consume_guard(expected_available)
    if consumed != expected_consumed:
        raise EvaluationKernelContractError("evaluation consumed guard drifted")
    raw = read_canonical_object(root / EVALUATION_RUN_INTENT)
    intent = EvaluationRunIntent(
        str(raw.get("manifest_sha256", "")),
        str(raw.get("family_commitment", "")),
        str(raw.get("consumed_guard_sha256", "")),
        raw.get("run_id", 0),
        str(raw.get("state", "")),
    )
    if (intent.to_dict() != raw or intent != expected_intent
            or intent.consumed_guard_sha256 != consumed.sha256()):
        raise EvaluationKernelContractError("evaluation consumed lineage drifted")
    return intent


# object-model: interface; representation=protocol; interop=pending
class EvaluationRecordLoader(Protocol):
    """Stage source adapter invoked only after guard consumption."""

    def __call__(
            self,
            context: EvaluationPluginRunContext,
            files: tuple[AuthorizedEvaluationFile, ...],
            ) -> Iterable[object]:
        """Yield the stage's validated, source-first bounded record stream."""


def preflight_evaluation_family(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        family_root: str | Path,
        files: tuple[EvaluationFileIdentity, ...],
        plugin: StageEvaluationPlugin,
        ) -> tuple[EvaluationKernelManifest, tuple[AuthorizedEvaluationFile, ...]]:
    """P3-style transport-only preflight; it never opens decompressed content."""
    root = _existing_family_root(family_root)
    manifest = read_evaluation_family_manifest(root)
    if manifest.kernel_semantic_sha256 != evaluation_kernel_semantic_sha256(
            _REPOSITORY_ROOT):
        raise EvaluationKernelContractError("evaluation live kernel identity drifted")
    if plugin.declaration != manifest.plugin:
        raise EvaluationKernelContractError("evaluation stage plugin identity drifted")
    if not (root / EVALUATION_GUARD_AVAILABLE).is_file():
        raise EvaluationKernelContractError("evaluation family guard already consumed")
    if (_guard_at(root, EVALUATION_GUARD_AVAILABLE)
            != build_available_guard(manifest)):
        raise EvaluationKernelContractError("evaluation preflight guard drifted")
    if any((root / name).exists() for name in (
            EVALUATION_GUARD_CONSUMED, EVALUATION_RUN_INTENT,
            EVALUATION_AGGREGATE, EVALUATION_PUBLICATION_DECISION,
            EVALUATION_RUNTIME_RECEIPT, EVALUATION_FAILURE_SEAL)):
        raise EvaluationKernelContractError("evaluation family is no longer unused")
    return manifest, authorize_evaluation_files(boundary, roots, manifest, files)


def publish_evaluation_family_formal_ready(
        family_root: str | Path,
        receipt: EvaluationFormalReadyReceipt,
        ) -> Path:
    """Bind an externally completed PASS P0-P4 receipt to this family root."""
    root = _existing_family_root(family_root)
    manifest = read_evaluation_family_manifest(root)
    return publish_formal_ready_receipt(
        manifest, receipt, root / EVALUATION_FORMAL_READY_RECEIPT)


def _validate_plugin_outcome(
        manifest: EvaluationKernelManifest,
        outcome: EvaluationPluginOutcome,
        ) -> None:
    if not isinstance(outcome, EvaluationPluginOutcome):
        raise EvaluationKernelContractError("evaluation plugin returned invalid output")
    audit = outcome.run_audit
    owner = manifest.owner_binding
    budget = manifest.resource_budget
    if audit.audit_state != "COMPLETE":
        raise EvaluationKernelContractError("plugin audit is incomplete")
    if (audit.source_ref_count != owner.source_ref_count
            or audit.pair_count != owner.pair_count
            or audit.private_record_reads > budget.max_records
            or audit.private_payload_gets > budget.max_payload_gets
            or audit.transport_bytes_read > budget.max_payload_bytes
            or audit.logic_operations > budget.max_logic_operations):
        raise EvaluationKernelContractError("evaluation plugin resource/read audit drifted")


def _blocked_result_set(manifest: EvaluationKernelManifest) -> EvaluationResultSet:
    roles = (
        *("BEARING" for _ in manifest.bearing_dimension_keys),
        "GENERATION",
        *("SUPPORT" for _ in manifest.support_dimension_keys),
    )
    evidence = hashlib.sha256(canonical_json_bytes({
        "family_commitment": manifest.family_commitment,
        "phase": "FORMAL_RUNTIME_BLOCK",
        "plugin_semantic_sha256": manifest.plugin.semantic_sha256,
    })).hexdigest()
    return EvaluationResultSet(tuple(
        EvaluationDimensionResult(
            key, role, "BLOCKED", 1, 0, 0, 0, 1, evidence)
        for key, role in zip(manifest.hard_conjunct_keys, roles, strict=True)
    ))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationFamilyPublication:
    """In-memory result of one sealed formal family."""

    aggregate: EvaluationAggregate
    decision: EvaluationPublicationDecision
    runtime_receipt: EvaluationRuntimeReceipt | None
    failure_seal: EvaluationFailureSeal | None

    def __post_init__(self) -> None:
        if (self.decision.aggregate_sha256 != self.aggregate.sha256()
                or self.decision.status != self.aggregate.status):
            raise EvaluationKernelContractError("publication decision lineage drifted")
        if self.aggregate.status == "PASS":
            if self.runtime_receipt is None or self.failure_seal is not None:
                raise EvaluationKernelContractError("PASS publication projection drifted")
        elif self.runtime_receipt is not None or self.failure_seal is None:
            raise EvaluationKernelContractError("non-PASS publication projection drifted")


def _publish_evaluation_result(
        root: Path,
        aggregate: EvaluationAggregate,
        ) -> EvaluationFamilyPublication:
    if any((root / name).exists() for name in (
            EVALUATION_AGGREGATE, EVALUATION_PUBLICATION_DECISION,
            EVALUATION_RUNTIME_RECEIPT, EVALUATION_FAILURE_SEAL)):
        raise EvaluationKernelContractError("evaluation result already published")
    decision = build_publication_decision(aggregate)
    write_immutable_json(aggregate.to_dict(), root / EVALUATION_AGGREGATE)
    write_immutable_json(
        decision.to_dict(), root / EVALUATION_PUBLICATION_DECISION)
    receipt = None
    seal = None
    if aggregate.status == "PASS":
        receipt = build_runtime_receipt(aggregate)
        write_immutable_json(receipt.to_dict(), root / EVALUATION_RUNTIME_RECEIPT)
    else:
        seal = build_failure_seal(aggregate)
        write_immutable_json(seal.to_dict(), root / EVALUATION_FAILURE_SEAL)
    if read_canonical_object(root / EVALUATION_AGGREGATE) != aggregate.to_dict():
        raise EvaluationKernelContractError("evaluation aggregate readback drifted")
    return EvaluationFamilyPublication(aggregate, decision, receipt, seal)


def run_evaluation_family_once(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        family_root: str | Path,
        files: tuple[EvaluationFileIdentity, ...],
        plugin: StageEvaluationPlugin,
        record_loader: EvaluationRecordLoader,
        ) -> EvaluationFamilyPublication:
    """Authorize, consume, stream, aggregate and seal exactly one formal run."""
    root = _existing_family_root(family_root)
    manifest, authorized = preflight_evaluation_family(
        boundary, roots, root, files, plugin)
    ready_path = root / EVALUATION_FORMAL_READY_RECEIPT
    if not ready_path.is_file():
        raise EvaluationKernelContractError("evaluation formal-ready receipt is missing")
    assert_formal_ready_receipt(manifest, read_formal_ready_receipt(ready_path))
    consume_evaluation_family_guard(root, manifest)
    context = EvaluationPluginRunContext(
        manifest.sha256(), manifest.family_commitment,
        manifest.source_binding.sha256(), manifest.owner_binding.sha256(), 1)
    try:
        records = record_loader(context, authorized)
        outcome = plugin.evaluate(context, records)
        _validate_plugin_outcome(manifest, outcome)
        authorize_evaluation_files(boundary, roots, manifest, files)
        aggregate = build_evaluation_aggregate(
            manifest, outcome.result_set, outcome.run_audit)
    except Exception:
        aggregate = build_evaluation_aggregate(
            manifest, _blocked_result_set(manifest),
            EvaluationRunAudit.blocked_unavailable())
    verify_evaluation_family_consumed(root, manifest)
    return _publish_evaluation_result(root, aggregate)


__all__ = [
    "EVALUATION_AGGREGATE",
    "EVALUATION_FAILURE_SEAL",
    "EVALUATION_FORMAL_READY_RECEIPT",
    "EVALUATION_FAMILY_MANIFEST",
    "EVALUATION_GUARD_AVAILABLE",
    "EVALUATION_GUARD_CONSUMED",
    "EVALUATION_PUBLICATION_DECISION",
    "EVALUATION_RUN_INTENT",
    "EVALUATION_RUNTIME_RECEIPT",
    "EvaluationFamilyPublication",
    "EvaluationRecordLoader",
    "consume_evaluation_family_guard",
    "consume_evaluation_guard_once",
    "preflight_evaluation_family",
    "publish_evaluation_family_formal_ready",
    "publish_evaluation_family",
    "read_evaluation_family_manifest",
    "run_evaluation_family_once",
    "verify_evaluation_family_consumed",
    "verify_evaluation_guard_consumed",
]
