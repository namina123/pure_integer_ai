"""Metadata-only owner handoff for the independent ``CONFLICT_SET`` family.

This boundary opens only the owner metadata record.  It never resolves a
private payload path, reads a label transport, or creates a formal family.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_handoff import (
    CAPABILITY_KEY,
    CODE_IDENTITY,
    EVALUATOR_OWNER,
    FAMILY_NAMESPACE,
    SOURCE_OWNER,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    ARTIFACT_ROLES,
    ConflictSetPrivateArtifact,
    ConflictSetPrivateProtocolError,
    ConflictSetPrivateTransport,
    ConflictSetRunGuard,
    build_conflict_set_run_guard,
)


OWNER_METADATA_ARTIFACT_KIND = "PH2_GG03_CONFLICT_SET_OWNER_METADATA_V1"
OWNER_METADATA_FORMAT_VERSION = 1
OWNER_METADATA_STATUS = "SEALED_UNREAD"
OWNER_METADATA_FILENAMES = frozenset({
    "owner-metadata.jsonl",
    "owner-receipt.json",
    "owner-receipt.jsonl",
})


class ConflictSetOwnerMetadataError(ValueError):
    """Owner metadata cannot be accepted without opening private payload."""


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConflictSetOwnerMetadataError(f"{where} must be non-empty text")
    return value


def _sha(value: object, *, where: str) -> str:
    result = _text(value, where=where)
    if (len(result) != 64 or result != result.lower()
            or any(char not in "0123456789abcdef" for char in result)):
        raise ConflictSetOwnerMetadataError(
            f"{where} must be a lowercase SHA-256")
    return result


def _zero(value: object, *, where: str) -> int:
    if type(value) is not int or value != 0:
        raise ConflictSetOwnerMetadataError(f"{where} must be zero")
    return value


def _exact(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ConflictSetOwnerMetadataError(
            f"{where} has missing or unknown fields")
    return value


def _receipt_payload(
        *,
        owner_id: str,
        family_namespace: str,
        capability_key: str,
        code_identity: str,
        source_owner: str,
        transport_commitment_sha256: str,
        artifacts: tuple[ConflictSetPrivateArtifact, ...],
        formal_run_count_before: int,
        private_payload_reads_before: int,
        host_learning_writes_before: int,
        label_writes_before: int,
        clone_evaluation_writes_before: int,
        teacher_api_llm_calls_before: int,
        status: str,
        ) -> dict[str, object]:
    """Build the self-committed metadata body without the receipt field."""
    return {
        "artifact_kind": OWNER_METADATA_ARTIFACT_KIND,
        "artifacts": [item.to_dict() for item in artifacts],
        "capability_key": capability_key,
        "clone_evaluation_writes_before": clone_evaluation_writes_before,
        "code_identity": code_identity,
        "family_namespace": family_namespace,
        "formal_run_count_before": formal_run_count_before,
        "format_version": OWNER_METADATA_FORMAT_VERSION,
        "host_learning_writes_before": host_learning_writes_before,
        "label_writes_before": label_writes_before,
        "private_payload_reads_before": private_payload_reads_before,
        "source_owner": source_owner,
        "status": status,
        "teacher_api_llm_calls_before": teacher_api_llm_calls_before,
        "transport_commitment_sha256": transport_commitment_sha256,
        "owner_id": owner_id,
    }


def _validate_artifacts(
        artifacts: tuple[ConflictSetPrivateArtifact, ...],
        ) -> None:
    if (not isinstance(artifacts, tuple)
            or any(not isinstance(item, ConflictSetPrivateArtifact)
                   for item in artifacts)):
        raise ConflictSetOwnerMetadataError(
            "owner metadata artifact inventory is not typed")
    roles = tuple(item.role for item in artifacts)
    if (roles != ARTIFACT_ROLES
            or len(set(item.relative_path for item in artifacts))
            != len(artifacts)):
        raise ConflictSetOwnerMetadataError(
            "owner metadata artifact inventory is incomplete or reordered")
    by_role = {item.role: item for item in artifacts}
    if (by_role["observation_pack"].record_count
            != by_role["private_labels"].record_count):
        raise ConflictSetOwnerMetadataError(
            "owner metadata observation/label counts do not close")


@dataclass(frozen=True, slots=True)
class ConflictSetOwnerMetadata:
    """Safe owner projection; private payload paths remain opaque."""

    owner_id: str
    family_namespace: str
    capability_key: str
    code_identity: str
    source_owner: str
    transport_commitment_sha256: str
    owner_receipt_sha256: str
    artifacts: tuple[ConflictSetPrivateArtifact, ...]
    formal_run_count_before: int = 0
    private_payload_reads_before: int = 0
    host_learning_writes_before: int = 0
    label_writes_before: int = 0
    clone_evaluation_writes_before: int = 0
    teacher_api_llm_calls_before: int = 0
    status: str = OWNER_METADATA_STATUS

    def __post_init__(self) -> None:
        if (self.owner_id != EVALUATOR_OWNER
                or self.family_namespace != FAMILY_NAMESPACE
                or self.capability_key != CAPABILITY_KEY
                or self.code_identity != CODE_IDENTITY
                or self.source_owner != SOURCE_OWNER):
            raise ConflictSetOwnerMetadataError(
                "owner metadata identity is not the independent family")
        _sha(self.transport_commitment_sha256,
             where="owner metadata transport commitment")
        _sha(self.owner_receipt_sha256, where="owner metadata receipt")
        if self.status != OWNER_METADATA_STATUS:
            raise ConflictSetOwnerMetadataError(
                "owner metadata status must remain SEALED_UNREAD")
        _validate_artifacts(self.artifacts)
        for name in (
                "formal_run_count_before", "private_payload_reads_before",
                "host_learning_writes_before", "label_writes_before",
                "clone_evaluation_writes_before",
                "teacher_api_llm_calls_before"):
            _zero(getattr(self, name), where=f"owner metadata {name}")
        expected = hashlib.sha256(canonical_json_bytes(
            _receipt_payload(
                owner_id=self.owner_id,
                family_namespace=self.family_namespace,
                capability_key=self.capability_key,
                code_identity=self.code_identity,
                source_owner=self.source_owner,
                transport_commitment_sha256=self.transport_commitment_sha256,
                artifacts=self.artifacts,
                formal_run_count_before=self.formal_run_count_before,
                private_payload_reads_before=self.private_payload_reads_before,
                host_learning_writes_before=self.host_learning_writes_before,
                label_writes_before=self.label_writes_before,
                clone_evaluation_writes_before=(
                    self.clone_evaluation_writes_before),
                teacher_api_llm_calls_before=(
                    self.teacher_api_llm_calls_before),
                status=self.status,
            ))).hexdigest()
        if self.owner_receipt_sha256 != expected:
            raise ConflictSetOwnerMetadataError(
                "owner metadata receipt commitment drifted")

    def to_dict(self) -> dict[str, object]:
        value = _receipt_payload(
            owner_id=self.owner_id,
            family_namespace=self.family_namespace,
            capability_key=self.capability_key,
            code_identity=self.code_identity,
            source_owner=self.source_owner,
            transport_commitment_sha256=self.transport_commitment_sha256,
            artifacts=self.artifacts,
            formal_run_count_before=self.formal_run_count_before,
            private_payload_reads_before=self.private_payload_reads_before,
            host_learning_writes_before=self.host_learning_writes_before,
            label_writes_before=self.label_writes_before,
            clone_evaluation_writes_before=self.clone_evaluation_writes_before,
            teacher_api_llm_calls_before=self.teacher_api_llm_calls_before,
            status=self.status,
        )
        value["owner_receipt_sha256"] = self.owner_receipt_sha256
        return value

    def canonical_bytes(self) -> bytes:
        """Return one canonical metadata JSONL record, including its newline."""
        return canonical_json_line(self.to_dict())

    def metadata_sha256(self) -> str:
        """Hash the complete metadata record, never a private payload."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetOwnerMetadata":
        raw = _exact(value, {
            "artifact_kind", "artifacts", "capability_key",
            "clone_evaluation_writes_before", "code_identity",
            "family_namespace", "formal_run_count_before", "format_version",
            "host_learning_writes_before", "label_writes_before",
            "owner_id", "owner_receipt_sha256", "private_payload_reads_before",
            "source_owner", "status", "teacher_api_llm_calls_before",
            "transport_commitment_sha256",
        }, where="owner metadata")
        if (raw["artifact_kind"] != OWNER_METADATA_ARTIFACT_KIND
                or raw["format_version"] != OWNER_METADATA_FORMAT_VERSION):
            raise ConflictSetOwnerMetadataError(
                "owner metadata artifact kind or version is invalid")
        if not isinstance(raw["artifacts"], list):
            raise ConflictSetOwnerMetadataError(
                "owner metadata artifacts must be an array")
        try:
            artifacts = tuple(
                ConflictSetPrivateArtifact.from_dict(item)
                for item in raw["artifacts"])
        except (ConflictSetPrivateProtocolError, TypeError, ValueError) as error:
            raise ConflictSetOwnerMetadataError(
                "owner metadata artifact is invalid") from error
        return cls(
            raw["owner_id"], raw["family_namespace"], raw["capability_key"],
            raw["code_identity"], raw["source_owner"],
            raw["transport_commitment_sha256"], raw["owner_receipt_sha256"],
            artifacts, raw["formal_run_count_before"],
            raw["private_payload_reads_before"],
            raw["host_learning_writes_before"], raw["label_writes_before"],
            raw["clone_evaluation_writes_before"],
            raw["teacher_api_llm_calls_before"], raw["status"],
        )


def validate_conflict_set_owner_metadata(
        transport: ConflictSetPrivateTransport,
        metadata: ConflictSetOwnerMetadata,
        ) -> None:
    """Close owner metadata against transport without reading any payload."""
    if (not isinstance(transport, ConflictSetPrivateTransport)
            or not isinstance(metadata, ConflictSetOwnerMetadata)):
        raise TypeError("transport or owner metadata type is invalid")
    if metadata.transport_commitment_sha256 != transport.commitment_sha256():
        raise ConflictSetOwnerMetadataError(
            "owner metadata transport commitment drifted")
    if metadata.artifacts != transport.artifacts:
        raise ConflictSetOwnerMetadataError(
            "owner metadata artifact inventory drifted")


def build_conflict_set_private_transport_from_owner_metadata(
        metadata: ConflictSetOwnerMetadata,
        *,
        public_preflight_manifest_sha256: str,
        observation_pack_sha256: str,
        source_manifest_sha256: str,
        candidate_manifest_sha256: str,
        ) -> ConflictSetPrivateTransport:
    """Assemble candidate transport from owner metadata, then close its hash."""
    if not isinstance(metadata, ConflictSetOwnerMetadata):
        raise TypeError("owner metadata type is invalid")
    try:
        transport = ConflictSetPrivateTransport(
            FAMILY_NAMESPACE, CAPABILITY_KEY, CODE_IDENTITY,
            EVALUATOR_OWNER, SOURCE_OWNER,
            public_preflight_manifest_sha256, observation_pack_sha256,
            source_manifest_sha256, candidate_manifest_sha256,
            metadata.artifacts,
        )
    except ConflictSetPrivateProtocolError as error:
        raise ConflictSetOwnerMetadataError(
            "owner metadata cannot assemble a valid transport") from error
    validate_conflict_set_owner_metadata(transport, metadata)
    return transport


def build_conflict_set_run_guard_from_owner_metadata(
        transport: ConflictSetPrivateTransport,
        metadata: ConflictSetOwnerMetadata,
        ) -> ConflictSetRunGuard:
    """Create the available guard only after owner metadata closes exactly."""
    validate_conflict_set_owner_metadata(transport, metadata)
    return build_conflict_set_run_guard(
        transport, owner_receipt_sha256=metadata.owner_receipt_sha256)


def parse_conflict_set_owner_metadata_bytes(
        payload: bytes,
        ) -> ConflictSetOwnerMetadata:
    """Parse one canonical owner metadata JSONL record only."""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise ConflictSetOwnerMetadataError(
            "owner metadata must be one JSONL record")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except (TypeError, ValueError) as error:
        raise ConflictSetOwnerMetadataError(
            "owner metadata JSON is not canonical") from error
    if canonical_json_line(value) != payload:
        raise ConflictSetOwnerMetadataError(
            "owner metadata JSON bytes are not canonical")
    return ConflictSetOwnerMetadata.from_dict(value)


def read_conflict_set_owner_metadata(
        path: str | Path,
        ) -> ConflictSetOwnerMetadata:
    """Read only the explicitly supplied owner metadata file."""
    target = Path(path)
    if (target.name not in OWNER_METADATA_FILENAMES
            or target.is_symlink() or not target.is_file()):
        raise ConflictSetOwnerMetadataError("owner metadata file is missing")
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise ConflictSetOwnerMetadataError(
            "owner metadata file cannot be read") from error
    return parse_conflict_set_owner_metadata_bytes(payload)


__all__ = [
    "OWNER_METADATA_ARTIFACT_KIND",
    "OWNER_METADATA_FILENAMES",
    "OWNER_METADATA_FORMAT_VERSION",
    "OWNER_METADATA_STATUS",
    "ConflictSetOwnerMetadata",
    "ConflictSetOwnerMetadataError",
    "build_conflict_set_private_transport_from_owner_metadata",
    "build_conflict_set_run_guard_from_owner_metadata",
    "parse_conflict_set_owner_metadata_bytes",
    "read_conflict_set_owner_metadata",
    "validate_conflict_set_owner_metadata",
]
