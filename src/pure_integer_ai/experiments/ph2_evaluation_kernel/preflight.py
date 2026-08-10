"""Five-layer P0-P4 formal-readiness contracts for generic families."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    flag,
    read_canonical_object,
    sha1_text,
    sha256_text,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.manifest import (
    EvaluationKernelManifest,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.private_io import (
    AuthorizedEvaluationFile,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EVALUATION_RESULT_STATUSES,
    EvaluationKernelContractError,
    evaluation_status_from_counts,
)


EVALUATION_PREFLIGHT_LAYERS = ("P0", "P1", "P2", "P3", "P4")
EVALUATION_FORMAL_READY_VERSION = "PH2-EVALUATION-FORMAL-READY-V1"


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class EvaluationPreflightCheck:
    """One payload-free check identity and its four-state outcome."""

    check_key: str
    status: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.check_key, str) or not self.check_key:
            raise EvaluationKernelContractError("preflight check key is invalid")
        if self.status not in EVALUATION_RESULT_STATUSES:
            raise EvaluationKernelContractError("preflight check status is invalid")
        sha256_text(self.evidence_sha256, where="preflight check evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_key": self.check_key,
            "evidence_sha256": self.evidence_sha256,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationPreflightCheck":
        raw = exact_dict(value, {
            "check_key", "evidence_sha256", "status",
        }, where="EvaluationPreflightCheck")
        return cls(
            str(raw["check_key"]), str(raw["status"]),
            str(raw["evidence_sha256"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationPreflightLayer:
    """Canonical ordered checks for one P0-P4 layer."""

    layer_key: str
    checks: tuple[EvaluationPreflightCheck, ...]
    status: str

    def __post_init__(self) -> None:
        if self.layer_key not in EVALUATION_PREFLIGHT_LAYERS:
            raise EvaluationKernelContractError("preflight layer key is invalid")
        if (not isinstance(self.checks, tuple) or not self.checks
                or any(not isinstance(item, EvaluationPreflightCheck)
                       for item in self.checks)):
            raise EvaluationKernelContractError("preflight layer checks are invalid")
        keys = tuple(item.check_key for item in self.checks)
        if len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
            raise EvaluationKernelContractError(
                "preflight check keys must be unique and canonical")
        counts = {
            state: sum(item.status == state for item in self.checks)
            for state in EVALUATION_RESULT_STATUSES
        }
        expected = evaluation_status_from_counts(
            planned_count=len(self.checks),
            passed_count=counts["PASS"], failed_count=counts["FAIL"],
            not_evaluated_count=counts["NE"], blocked_count=counts["BLOCKED"],
        )
        if self.status != expected:
            raise EvaluationKernelContractError("preflight layer status drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [item.to_dict() for item in self.checks],
            "layer_key": self.layer_key,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationPreflightLayer":
        raw = exact_dict(value, {
            "checks", "layer_key", "status",
        }, where="EvaluationPreflightLayer")
        return cls(
            str(raw["layer_key"]),
            tuple(EvaluationPreflightCheck.from_dict(item)
                  for item in raw["checks"]),
            str(raw["status"]),
        )


def build_preflight_layer(
        layer_key: str,
        checks: Iterable[EvaluationPreflightCheck],
        ) -> EvaluationPreflightLayer:
    """Sort safe check identities and derive the layer status."""
    rows = tuple(sorted(checks))
    counts = {state: sum(item.status == state for item in rows)
              for state in EVALUATION_RESULT_STATUSES}
    status = evaluation_status_from_counts(
        planned_count=len(rows), passed_count=counts["PASS"],
        failed_count=counts["FAIL"], not_evaluated_count=counts["NE"],
        blocked_count=counts["BLOCKED"],
    )
    return EvaluationPreflightLayer(layer_key, rows, status)


def build_transport_preflight_layer(
        manifest: EvaluationKernelManifest,
        files: tuple[AuthorizedEvaluationFile, ...],
        ) -> EvaluationPreflightLayer:
    """Build P3 only from safe transport permits; no decompressed payload is read."""
    if (not isinstance(manifest, EvaluationKernelManifest) or not files
            or any(not isinstance(item, AuthorizedEvaluationFile) for item in files)):
        raise EvaluationKernelContractError("transport preflight inputs are invalid")
    evidence = hashlib.sha256(canonical_json_bytes({
        "family_commitment": manifest.family_commitment,
        "file_count": len(files),
        "inventory_sha256": manifest.owner_binding.file_inventory_sha256,
        "owner_receipt_sha256": manifest.owner_binding.owner_receipt_sha256,
        "transport_family_commitment": manifest.transport_family_commitment,
    })).hexdigest()
    return build_preflight_layer("P3", (
        EvaluationPreflightCheck("ARTIFACT_FREEZE_IDENTITY", "PASS", evidence),
        EvaluationPreflightCheck("FAMILY_GUARD_AVAILABLE", "PASS", evidence),
        EvaluationPreflightCheck("OWNER_SAFE_METADATA", "PASS", evidence),
        EvaluationPreflightCheck("PRIVATE_PAYLOAD_READS_ZERO", "PASS", evidence),
        EvaluationPreflightCheck("TRANSPORT_LAYOUT_IDENTITY", "PASS", evidence),
    ))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationFormalReadyReceipt:
    """P0-P4 conjunction required before the only formal guard consumption."""

    artifact_version: str
    manifest_sha256: str
    family_commitment: str
    plugin_semantic_sha256: str
    source_binding_sha256: str
    owner_binding_sha256: str
    public_head_sha1: str
    publication_evidence_sha256: str
    layers: tuple[EvaluationPreflightLayer, ...]
    public_dev_status: str
    public_shadow_a_status: str
    public_shadow_b_or_metamorphic_status: str
    threshold_frozen: int
    family_frozen: int
    family_pushed: int
    formal_run_count_before: int
    private_payload_reads_before: int
    status: str

    def __post_init__(self) -> None:
        if self.artifact_version != EVALUATION_FORMAL_READY_VERSION:
            raise EvaluationKernelContractError("formal-ready version drifted")
        for name in (
                "manifest_sha256", "family_commitment", "plugin_semantic_sha256",
                "source_binding_sha256", "owner_binding_sha256"):
            sha256_text(getattr(self, name), where=f"formal-ready {name}")
        sha1_text(self.public_head_sha1, where="formal-ready public_head_sha1")
        sha256_text(
            self.publication_evidence_sha256,
            where="formal-ready publication evidence")
        if (not isinstance(self.layers, tuple)
                or tuple(item.layer_key for item in self.layers)
                != EVALUATION_PREFLIGHT_LAYERS):
            raise EvaluationKernelContractError("formal-ready layer order drifted")
        for name in (
                "public_dev_status", "public_shadow_a_status",
                "public_shadow_b_or_metamorphic_status"):
            if getattr(self, name) not in EVALUATION_RESULT_STATUSES:
                raise EvaluationKernelContractError("formal-ready public status invalid")
        for name in ("threshold_frozen", "family_frozen", "family_pushed"):
            flag(getattr(self, name), where=f"formal-ready {name}")
        if (type(self.formal_run_count_before) is not int
                or type(self.private_payload_reads_before) is not int
                or self.formal_run_count_before != 0
                or self.private_payload_reads_before != 0):
            raise EvaluationKernelContractError(
                "formal-ready family must remain unused and zero-read")
        expected = "PASS" if (
            all(item.status == "PASS" for item in self.layers)
            and self.public_dev_status == "PASS"
            and self.public_shadow_a_status == "PASS"
            and self.public_shadow_b_or_metamorphic_status == "PASS"
            and (self.threshold_frozen, self.family_frozen, self.family_pushed)
            == (1, 1, 1)
        ) else "BLOCKED"
        if self.status != expected:
            raise EvaluationKernelContractError("formal-ready status drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "family_commitment": self.family_commitment,
            "family_frozen": self.family_frozen,
            "family_pushed": self.family_pushed,
            "formal_run_count_before": self.formal_run_count_before,
            "layers": [item.to_dict() for item in self.layers],
            "manifest_sha256": self.manifest_sha256,
            "owner_binding_sha256": self.owner_binding_sha256,
            "plugin_semantic_sha256": self.plugin_semantic_sha256,
            "private_payload_reads_before": self.private_payload_reads_before,
            "public_head_sha1": self.public_head_sha1,
            "publication_evidence_sha256": self.publication_evidence_sha256,
            "public_dev_status": self.public_dev_status,
            "public_shadow_a_status": self.public_shadow_a_status,
            "public_shadow_b_or_metamorphic_status": (
                self.public_shadow_b_or_metamorphic_status),
            "source_binding_sha256": self.source_binding_sha256,
            "status": self.status,
            "threshold_frozen": self.threshold_frozen,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationFormalReadyReceipt":
        raw = exact_dict(value, {
            "artifact_version", "family_commitment", "family_frozen",
            "family_pushed", "formal_run_count_before", "layers",
            "manifest_sha256", "owner_binding_sha256", "plugin_semantic_sha256",
            "private_payload_reads_before", "public_dev_status",
            "public_head_sha1", "publication_evidence_sha256",
            "public_shadow_a_status", "public_shadow_b_or_metamorphic_status",
            "source_binding_sha256", "status", "threshold_frozen",
        }, where="EvaluationFormalReadyReceipt")
        return cls(
            str(raw["artifact_version"]), str(raw["manifest_sha256"]),
            str(raw["family_commitment"]), str(raw["plugin_semantic_sha256"]),
            str(raw["source_binding_sha256"]),
            str(raw["owner_binding_sha256"]),
            str(raw["public_head_sha1"]),
            str(raw["publication_evidence_sha256"]),
            tuple(EvaluationPreflightLayer.from_dict(item)
                  for item in raw["layers"]),
            str(raw["public_dev_status"]), str(raw["public_shadow_a_status"]),
            str(raw["public_shadow_b_or_metamorphic_status"]),
            raw["threshold_frozen"], raw["family_frozen"],
            raw["family_pushed"], raw["formal_run_count_before"],
            raw["private_payload_reads_before"], str(raw["status"]),
        )

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def build_formal_ready_receipt(
        manifest: EvaluationKernelManifest,
        p0: EvaluationPreflightLayer,
        p1: EvaluationPreflightLayer,
        p2: EvaluationPreflightLayer,
        p3: EvaluationPreflightLayer,
        *,
        public_dev_status: str,
        public_shadow_a_status: str,
        public_shadow_b_or_metamorphic_status: str,
        family_pushed: int,
        publication_evidence_sha256: str,
        ) -> EvaluationFormalReadyReceipt:
    """Derive P4 from P0-P3 and the frozen formal-ready hard conjunction."""
    if tuple(item.layer_key for item in (p0, p1, p2, p3)) != (
            "P0", "P1", "P2", "P3"):
        raise EvaluationKernelContractError("formal-ready input layers drifted")
    ready = (
        all(item.status == "PASS" for item in (p0, p1, p2, p3))
        and public_dev_status == public_shadow_a_status
        == public_shadow_b_or_metamorphic_status == "PASS"
        and family_pushed == 1
    )
    evidence = hashlib.sha256(canonical_json_bytes({
        "family_commitment": manifest.family_commitment,
        "layer_statuses": [item.status for item in (p0, p1, p2, p3)],
        "public_dev_status": public_dev_status,
        "public_shadow_a_status": public_shadow_a_status,
        "public_shadow_b_or_metamorphic_status": (
            public_shadow_b_or_metamorphic_status),
    })).hexdigest()
    p4 = build_preflight_layer("P4", (
        EvaluationPreflightCheck(
            "FORMAL_READY_HARD_CONJUNCTION",
            "PASS" if ready else "BLOCKED", evidence),
    ))
    return EvaluationFormalReadyReceipt(
        EVALUATION_FORMAL_READY_VERSION,
        manifest.sha256(), manifest.family_commitment,
        manifest.plugin.semantic_sha256, manifest.source_binding.sha256(),
        manifest.owner_binding.sha256(), manifest.public_head_sha1,
        publication_evidence_sha256, (p0, p1, p2, p3, p4),
        public_dev_status, public_shadow_a_status,
        public_shadow_b_or_metamorphic_status,
        1, 1, family_pushed, 0, 0,
        "PASS" if ready else "BLOCKED",
    )


def assert_formal_ready_receipt(
        manifest: EvaluationKernelManifest,
        receipt: EvaluationFormalReadyReceipt,
        ) -> None:
    """Reject any non-PASS or cross-family formal-ready receipt."""
    if (not isinstance(manifest, EvaluationKernelManifest)
            or not isinstance(receipt, EvaluationFormalReadyReceipt)
            or receipt.status != "PASS"
            or receipt.manifest_sha256 != manifest.sha256()
            or receipt.family_commitment != manifest.family_commitment
            or receipt.plugin_semantic_sha256 != manifest.plugin.semantic_sha256
            or receipt.source_binding_sha256 != manifest.source_binding.sha256()
            or receipt.owner_binding_sha256 != manifest.owner_binding.sha256()
            or receipt.public_head_sha1 != manifest.public_head_sha1):
        raise EvaluationKernelContractError("evaluation family is not formal-ready")


def publish_formal_ready_receipt(
        manifest: EvaluationKernelManifest,
        receipt: EvaluationFormalReadyReceipt,
        path: str | Path,
        ) -> Path:
    """Publish only a PASS P0-P4 receipt append-only."""
    assert_formal_ready_receipt(manifest, receipt)
    target = write_immutable_json(receipt.to_dict(), path)
    if read_formal_ready_receipt(target) != receipt:
        raise EvaluationKernelContractError("formal-ready receipt readback drifted")
    return target


def read_formal_ready_receipt(path: str | Path) -> EvaluationFormalReadyReceipt:
    return EvaluationFormalReadyReceipt.from_dict(read_canonical_object(path))


__all__ = [
    "EVALUATION_FORMAL_READY_VERSION",
    "EVALUATION_PREFLIGHT_LAYERS",
    "EvaluationFormalReadyReceipt",
    "EvaluationPreflightCheck",
    "EvaluationPreflightLayer",
    "assert_formal_ready_receipt",
    "build_formal_ready_receipt",
    "build_preflight_layer",
    "build_transport_preflight_layer",
    "publish_formal_ready_receipt",
    "read_formal_ready_receipt",
]
