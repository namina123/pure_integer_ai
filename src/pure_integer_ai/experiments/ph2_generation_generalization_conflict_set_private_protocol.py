"""Independent metadata-only private transport protocol for ``CONFLICT_SET``.

The protocol freezes ownership, artifact roles, commitments, and a one-shot
run guard.  It intentionally never opens a private label, candidate payload,
or formal artifact; those actions belong to a later isolated runner.
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
    ARTIFACT_ROLES,
    CAPABILITY_KEY,
    CODE_IDENTITY,
    EVALUATOR_OWNER,
    FAMILY_NAMESPACE,
    SOURCE_OWNER,
)


ARTIFACT_KIND = "PH2_GG03_CONFLICT_SET_PRIVATE_PROTOCOL_V1"
FORMAT_VERSION = 1
PROTOCOL_STATUS = "PRIVATE_TRANSPORT_PROTOCOL_FROZEN"
TRANSPORT_ROOT_NAMESPACE = "gg03-conflict-set-v1"
RUN_STATES = ("AVAILABLE", "CONSUMED")
RUN_INTENT_STATE = "FORMAL_RUN_INTENT_FROZEN"
ARTIFACT_MATERIALIZATION_STATES = ("MATERIALIZED", "RESERVED")
_FORBIDDEN_PATH_PARTS = frozenset({
    "PH2_GG03_EXECUTABLE_EVALUATION_FAMILY_FREEZE_V1",
    "PH2_GG03_EXECUTABLE_SEMANTIC_EVALUATION_FAMILY_FREEZE_V2",
    "formal-family-20260818-a",
    "v8",
    "v10",
})
_ROLE_SPECS = {
    "code_freeze": (CODE_IDENTITY, "PUBLIC", "CODE_FREEZE", "MATERIALIZED"),
    "observation_pack": (SOURCE_OWNER, "PUBLIC", "OBSERVATION", "MATERIALIZED"),
    "source_manifest": (SOURCE_OWNER, "PUBLIC", "SOURCE", "MATERIALIZED"),
    "candidate_manifest": (CODE_IDENTITY, "PUBLIC", "CANDIDATE", "MATERIALIZED"),
    "public_preflight": (CODE_IDENTITY, "PUBLIC", "PUBLIC_PREFLIGHT", "MATERIALIZED"),
    "private_labels": (EVALUATOR_OWNER, "PRIVATE", "PRIVATE_LABEL", "MATERIALIZED"),
    "prediction_seal": (EVALUATOR_OWNER, "PRIVATE", "PREDICTION", "RESERVED"),
    "aggregate_report": (EVALUATOR_OWNER, "PUBLIC", "PUBLICATION", "RESERVED"),
    "runtime_receipt": (EVALUATOR_OWNER, "PUBLIC", "PUBLICATION", "RESERVED"),
    "formal_failure_report": (EVALUATOR_OWNER, "PUBLIC", "PUBLICATION", "RESERVED"),
}


class ConflictSetPrivateProtocolError(ValueError):
    """The new private transport protocol is not safe to freeze."""


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConflictSetPrivateProtocolError(f"{where} must be non-empty text")
    return value


def _sha(value: object, *, where: str, length: int = 64) -> str:
    result = _text(value, where=where)
    if (len(result) != length
            or result != result.lower()
            or any(char not in "0123456789abcdef" for char in result)):
        raise ConflictSetPrivateProtocolError(
            f"{where} must be a hexadecimal SHA-{length * 4}")
    return result


def _positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConflictSetPrivateProtocolError(
            f"{where} must be a positive strict integer")
    return value


def _zero(value: object, *, where: str) -> int:
    if type(value) is not int or value != 0:
        raise ConflictSetPrivateProtocolError(f"{where} must be zero")
    return value


def _exact(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ConflictSetPrivateProtocolError(
            f"{where} has missing or unknown fields")
    return value


def _relative(value: object, *, where: str) -> str:
    result = _text(value, where=where)
    path = Path(result)
    if (path.is_absolute() or "\\" in result
            or any(part in {"", ".", ".."} for part in result.split("/"))):
        raise ConflictSetPrivateProtocolError(
            f"{where} must be a safe POSIX relative path")
    if not result.startswith(TRANSPORT_ROOT_NAMESPACE + "/"):
        raise ConflictSetPrivateProtocolError(
            f"{where} must stay inside the new transport namespace")
    if any(part.lower() in {item.lower() for item in _FORBIDDEN_PATH_PARTS}
           for part in result.split("/")):
        raise ConflictSetPrivateProtocolError(
            f"{where} points at a legacy family")
    return result


@dataclass(frozen=True, slots=True, order=True)
class ConflictSetPrivateArtifact:
    """Metadata and commitment for one role; content is never read here."""

    role: str
    owner: str
    visibility: str
    phase: str
    relative_path: str
    transport_sha256: str | None
    transport_size_bytes: int
    content_sha256: str | None
    content_size_bytes: int
    record_count: int
    materialization: str = "MATERIALIZED"

    def __post_init__(self) -> None:
        spec = _ROLE_SPECS.get(self.role)
        if spec is None or self.role not in ARTIFACT_ROLES:
            raise ConflictSetPrivateProtocolError("artifact role is not frozen")
        expected_owner, expected_visibility, expected_phase, expected_materialization = spec
        if (self.owner != expected_owner
                or self.visibility != expected_visibility
                or self.phase != expected_phase
                or self.materialization != expected_materialization):
            raise ConflictSetPrivateProtocolError(
                "artifact role owner/visibility/phase/materialization drifted")
        _relative(self.relative_path, where="artifact.relative_path")
        if self.materialization == "MATERIALIZED":
            _sha(self.transport_sha256, where="artifact.transport_sha256")
            _sha(self.content_sha256, where="artifact.content_sha256")
            for name in (
                    "transport_size_bytes", "content_size_bytes", "record_count"):
                _positive(getattr(self, name), where=f"artifact.{name}")
        elif self.materialization == "RESERVED":
            if self.transport_sha256 is not None or self.content_sha256 is not None:
                raise ConflictSetPrivateProtocolError(
                    "reserved artifact cannot claim content commitment")
            for name in (
                    "transport_size_bytes", "content_size_bytes", "record_count"):
                _zero(getattr(self, name), where=f"reserved artifact.{name}")
        else:
            raise ConflictSetPrivateProtocolError(
                "artifact materialization state is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "materialization": self.materialization,
            "owner": self.owner,
            "phase": self.phase,
            "record_count": self.record_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetPrivateArtifact":
        raw = _exact(value, {
            "content_sha256", "content_size_bytes", "materialization", "owner",
            "phase", "record_count", "relative_path", "role",
            "transport_sha256", "transport_size_bytes", "visibility",
        }, where="private_artifact")
        return cls(
            raw["role"], raw["owner"], raw["visibility"], raw["phase"],
            raw["relative_path"], raw["transport_sha256"],
            raw["transport_size_bytes"], raw["content_sha256"],
            raw["content_size_bytes"], raw["record_count"],
            raw["materialization"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetPrivateTransport:
    """Complete metadata inventory before any private payload read."""

    family_namespace: str
    capability_key: str
    code_identity: str
    evaluator_owner: str
    source_owner: str
    public_preflight_manifest_sha256: str
    observation_pack_sha256: str
    source_manifest_sha256: str
    candidate_manifest_sha256: str
    artifacts: tuple[ConflictSetPrivateArtifact, ...]
    status: str = PROTOCOL_STATUS
    formal_run_count_before: int = 0
    private_payload_reads_before: int = 0
    host_learning_writes_before: int = 0
    label_writes_before: int = 0
    clone_evaluation_writes_before: int = 0
    teacher_api_llm_calls_before: int = 0

    def __post_init__(self) -> None:
        if (self.family_namespace != FAMILY_NAMESPACE
                or self.capability_key != CAPABILITY_KEY
                or self.code_identity != CODE_IDENTITY
                or self.evaluator_owner != EVALUATOR_OWNER
                or self.source_owner != SOURCE_OWNER
                or self.evaluator_owner == self.source_owner):
            raise ConflictSetPrivateProtocolError(
                "private transport identity is not independent")
        for name in (
                "public_preflight_manifest_sha256", "observation_pack_sha256",
                "source_manifest_sha256", "candidate_manifest_sha256"):
            _sha(getattr(self, name), where=name)
        if self.status != PROTOCOL_STATUS:
            raise ConflictSetPrivateProtocolError("private transport status is invalid")
        if (not isinstance(self.artifacts, tuple)
                or any(not isinstance(item, ConflictSetPrivateArtifact)
                       for item in self.artifacts)):
            raise ConflictSetPrivateProtocolError(
                "private artifact inventory is incomplete or noncanonical")
        roles = tuple(item.role for item in self.artifacts)
        if (len(set(roles)) != len(roles) or roles != ARTIFACT_ROLES
                or len({item.relative_path for item in self.artifacts})
                != len(self.artifacts)):
            raise ConflictSetPrivateProtocolError(
                "private artifact inventory is incomplete or noncanonical")
        expected_content = {
            "public_preflight": self.public_preflight_manifest_sha256,
            "observation_pack": self.observation_pack_sha256,
            "source_manifest": self.source_manifest_sha256,
            "candidate_manifest": self.candidate_manifest_sha256,
        }
        for item in self.artifacts:
            expected = expected_content.get(item.role)
            if expected is not None and item.content_sha256 != expected:
                raise ConflictSetPrivateProtocolError(
                    f"artifact {item.role} commitment drifted")
        for name in (
                "formal_run_count_before", "private_payload_reads_before",
                "host_learning_writes_before", "label_writes_before",
                "clone_evaluation_writes_before", "teacher_api_llm_calls_before"):
            _zero(getattr(self, name), where=name)

    def commitment_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [item.to_dict() for item in self.artifacts],
            "artifact_kind": ARTIFACT_KIND,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "capability_key": self.capability_key,
            "clone_evaluation_writes_before": (
                self.clone_evaluation_writes_before),
            "code_identity": self.code_identity,
            "evaluator_owner": self.evaluator_owner,
            "family_namespace": self.family_namespace,
            "formal_run_count_before": self.formal_run_count_before,
            "format_version": FORMAT_VERSION,
            "host_learning_writes_before": self.host_learning_writes_before,
            "label_writes_before": self.label_writes_before,
            "observation_pack_sha256": self.observation_pack_sha256,
            "private_payload_reads_before": self.private_payload_reads_before,
            "public_preflight_manifest_sha256": (
                self.public_preflight_manifest_sha256),
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_owner": self.source_owner,
            "status": self.status,
            "teacher_api_llm_calls_before": self.teacher_api_llm_calls_before,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetPrivateTransport":
        raw = _exact(value, {
            "artifact_kind", "artifacts", "candidate_manifest_sha256", "capability_key",
            "clone_evaluation_writes_before", "code_identity", "evaluator_owner",
            "family_namespace", "formal_run_count_before", "format_version",
            "host_learning_writes_before", "label_writes_before",
            "observation_pack_sha256", "private_payload_reads_before",
            "public_preflight_manifest_sha256", "source_manifest_sha256",
            "source_owner", "status", "teacher_api_llm_calls_before",
        }, where="private_transport")
        if (raw["artifact_kind"] != ARTIFACT_KIND
                or raw["format_version"] != FORMAT_VERSION):
            raise ConflictSetPrivateProtocolError(
                "private transport artifact kind or format version is invalid")
        if not isinstance(raw["artifacts"], list):
            raise ConflictSetPrivateProtocolError(
                "private transport artifacts must be an array")
        return cls(
            raw["family_namespace"], raw["capability_key"], raw["code_identity"],
            raw["evaluator_owner"], raw["source_owner"],
            raw["public_preflight_manifest_sha256"],
            raw["observation_pack_sha256"], raw["source_manifest_sha256"],
            raw["candidate_manifest_sha256"], tuple(
                ConflictSetPrivateArtifact.from_dict(item)
                for item in raw["artifacts"]
            ), raw["status"], raw["formal_run_count_before"],
            raw["private_payload_reads_before"],
            raw["host_learning_writes_before"], raw["label_writes_before"],
            raw["clone_evaluation_writes_before"],
            raw["teacher_api_llm_calls_before"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetRunGuard:
    """Independent one-shot state for the future formal runner."""

    transport_commitment_sha256: str
    owner_receipt_sha256: str
    candidate_manifest_sha256: str
    run_id: int = 1
    state: str = "AVAILABLE"
    formal_run_count_before: int = 0
    private_payload_reads_before: int = 0

    def __post_init__(self) -> None:
        _sha(self.transport_commitment_sha256,
             where="guard.transport_commitment_sha256")
        _sha(self.owner_receipt_sha256, where="guard.owner_receipt_sha256")
        _sha(self.candidate_manifest_sha256,
             where="guard.candidate_manifest_sha256")
        if self.run_id != 1 or type(self.run_id) is not int:
            raise ConflictSetPrivateProtocolError("guard run_id must be one")
        if self.state not in RUN_STATES:
            raise ConflictSetPrivateProtocolError("guard state is invalid")
        _zero(self.formal_run_count_before,
              where="guard.formal_run_count_before")
        _zero(self.private_payload_reads_before,
              where="guard.private_payload_reads_before")

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "formal_run_count_before": self.formal_run_count_before,
            "owner_receipt_sha256": self.owner_receipt_sha256,
            "private_payload_reads_before": self.private_payload_reads_before,
            "run_id": self.run_id,
            "state": self.state,
            "transport_commitment_sha256": self.transport_commitment_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetRunGuard":
        raw = _exact(value, {
            "candidate_manifest_sha256", "formal_run_count_before",
            "owner_receipt_sha256", "private_payload_reads_before", "run_id",
            "state", "transport_commitment_sha256",
        }, where="run_guard")
        return cls(
            raw["transport_commitment_sha256"], raw["owner_receipt_sha256"],
            raw["candidate_manifest_sha256"], raw["run_id"], raw["state"],
            raw["formal_run_count_before"],
            raw["private_payload_reads_before"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetRunIntent:
    """Proof that the guard was consumed before any future payload read."""

    transport_commitment_sha256: str
    consumed_guard_sha256: str
    run_id: int = 1
    state: str = RUN_INTENT_STATE

    def __post_init__(self) -> None:
        _sha(self.transport_commitment_sha256,
             where="intent.transport_commitment_sha256")
        _sha(self.consumed_guard_sha256,
             where="intent.consumed_guard_sha256")
        if self.run_id != 1 or type(self.run_id) is not int:
            raise ConflictSetPrivateProtocolError("intent run_id must be one")
        if self.state != RUN_INTENT_STATE:
            raise ConflictSetPrivateProtocolError("intent state is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "consumed_guard_sha256": self.consumed_guard_sha256,
            "run_id": self.run_id,
            "state": self.state,
            "transport_commitment_sha256": self.transport_commitment_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetRunIntent":
        raw = _exact(value, {
            "consumed_guard_sha256", "run_id", "state",
            "transport_commitment_sha256",
        }, where="run_intent")
        return cls(
            raw["transport_commitment_sha256"],
            raw["consumed_guard_sha256"], raw["run_id"], raw["state"],
        )


def build_conflict_set_private_transport(
        *,
        public_preflight_manifest_sha256: str,
        observation_pack_sha256: str,
        source_manifest_sha256: str,
        candidate_manifest_sha256: str,
        artifacts: tuple[ConflictSetPrivateArtifact, ...],
        ) -> ConflictSetPrivateTransport:
    """Build metadata only; no argument is interpreted as a filesystem path."""
    return ConflictSetPrivateTransport(
        FAMILY_NAMESPACE, CAPABILITY_KEY, CODE_IDENTITY,
        EVALUATOR_OWNER, SOURCE_OWNER,
        public_preflight_manifest_sha256, observation_pack_sha256,
        source_manifest_sha256, candidate_manifest_sha256, artifacts,
    )


def build_conflict_set_run_guard(
        transport: ConflictSetPrivateTransport,
        *,
        owner_receipt_sha256: str,
        ) -> ConflictSetRunGuard:
    """Build an unused AVAILABLE guard from a validated transport commitment."""
    if not isinstance(transport, ConflictSetPrivateTransport):
        raise TypeError("transport type is invalid")
    return ConflictSetRunGuard(
        transport.commitment_sha256(), owner_receipt_sha256,
        transport.candidate_manifest_sha256,
    )


def consume_conflict_set_run_guard(
        guard: ConflictSetRunGuard,
        ) -> tuple[ConflictSetRunGuard, ConflictSetRunIntent]:
    """Consume once; a consumed guard cannot be consumed or replayed."""
    if not isinstance(guard, ConflictSetRunGuard) or guard.state != "AVAILABLE":
        raise ConflictSetPrivateProtocolError("guard is already consumed")
    consumed = ConflictSetRunGuard(
        guard.transport_commitment_sha256, guard.owner_receipt_sha256,
        guard.candidate_manifest_sha256, guard.run_id, "CONSUMED",
        guard.formal_run_count_before, guard.private_payload_reads_before,
    )
    return consumed, ConflictSetRunIntent(
        consumed.transport_commitment_sha256, consumed.sha256(), consumed.run_id,
    )


def assert_conflict_set_transport_matches_public_freeze(
        transport: ConflictSetPrivateTransport,
        *,
        public_preflight_manifest_sha256: str,
        observation_pack_sha256: str,
        source_manifest_sha256: str,
        candidate_manifest_sha256: str,
        ) -> None:
    """Reject public/private commitment drift without opening private content."""
    if not isinstance(transport, ConflictSetPrivateTransport):
        raise TypeError("transport type is invalid")
    expected = {
        "public_preflight_manifest_sha256": public_preflight_manifest_sha256,
        "observation_pack_sha256": observation_pack_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
    }
    for name, value in expected.items():
        _sha(value, where=f"expected.{name}")
        if getattr(transport, name) != value:
            raise ConflictSetPrivateProtocolError(
                f"private transport {name} drifted")


def parse_conflict_set_private_transport_bytes(
        payload: bytes) -> ConflictSetPrivateTransport:
    """Parse one canonical metadata-only transport record."""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise ConflictSetPrivateProtocolError(
            "private transport must be one JSONL record")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except (TypeError, ValueError) as error:
        raise ConflictSetPrivateProtocolError(
            "private transport JSON is not canonical") from error
    if canonical_json_line(value) != payload:
        raise ConflictSetPrivateProtocolError(
            "private transport JSON bytes are not canonical")
    return ConflictSetPrivateTransport.from_dict(value)


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_MATERIALIZATION_STATES",
    "FORMAT_VERSION",
    "PROTOCOL_STATUS",
    "RUN_INTENT_STATE",
    "RUN_STATES",
    "TRANSPORT_ROOT_NAMESPACE",
    "ConflictSetPrivateArtifact",
    "ConflictSetPrivateProtocolError",
    "ConflictSetPrivateTransport",
    "ConflictSetRunGuard",
    "ConflictSetRunIntent",
    "assert_conflict_set_transport_matches_public_freeze",
    "build_conflict_set_private_transport",
    "build_conflict_set_run_guard",
    "consume_conflict_set_run_guard",
    "parse_conflict_set_private_transport_bytes",
]
