"""Runtime owner/path firewall for the PH2-D03-V2 evaluator boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_EVALUATOR_STAGES,
    V2_OWNER_ROOT_POLICIES,
    V2_PRIVATE_SPLITS,
    V2_ZERO_WRITE_FIELDS,
    V2EvaluatorBoundaryError,
    V2EvaluatorBoundaryContract,
    V2OwnerRootPolicy,
    V2PrivateFamilyRegistration,
    V2ReportExposureError,
    validate_v2_safe_report,
)


V2_ACCESS_OPERATIONS = ("READ_SOURCE", "READ_OBSERVATION", "READ_TEACHER", "READ_LABEL")
V2_EXPOSURE_KINDS = (
    "ABSOLUTE_PATH",
    "FORBIDDEN_FIELD",
    "NON_STRING_KEY",
    "PATH_STRING",
    "UNSAFE_VALUE_TYPE",
)
V2_EXPOSURE_PHASES = (
    "REPORT_BUILD",
    "REPORT_SERIALIZE",
    "REPORT_PUBLISH",
)
V2_EXPOSURE_ARTIFACT_KIND = "PH2_D03_V2_EVALUATOR_EXPOSURE_INCIDENT"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not _SHA256.fullmatch(value)):
        raise V2EvaluatorBoundaryError(f"{where} must be lowercase SHA-256")
    return value


def _positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise V2EvaluatorBoundaryError(f"{where} must be positive")
    return value


def _safe_relative_path(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise V2EvaluatorBoundaryError("v2 evaluator relative path is invalid")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
            or "//" in value):
        raise V2EvaluatorBoundaryError("v2 evaluator path traversal rejected")
    return path.parts


def _root_is_safe(root: Path) -> Path:
    original = Path(root)
    if original.is_symlink():
        raise V2EvaluatorBoundaryError("v2 evaluator root symlink rejected")
    is_junction = getattr(original, "is_junction", None)
    if is_junction is not None and is_junction():
        raise V2EvaluatorBoundaryError("v2 evaluator root junction rejected")
    target = original.resolve()
    if not target.is_dir():
        raise V2EvaluatorBoundaryError("v2 evaluator root must be a real directory")
    return target


def _roots_disjoint(roots: tuple[Path, ...]) -> None:
    for index, first in enumerate(roots):
        for second in roots[index + 1:]:
            if first == second or first.is_relative_to(second) or second.is_relative_to(first):
                raise V2EvaluatorBoundaryError("v2 evaluator physical roots overlap")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2PhysicalRoots:
    """Opaque physical roots; no root path is serialized into a public report."""

    candidate_train: Path
    teacher_train: Path
    dev_calibration: Path
    shadow_audit: Path
    private_evaluator: Path
    exposure_ledger: Path

    def __post_init__(self) -> None:
        values = tuple(
            _root_is_safe(item) for item in (
                self.candidate_train, self.teacher_train, self.dev_calibration,
                self.shadow_audit, self.private_evaluator, self.exposure_ledger,
            )
        )
        _roots_disjoint(values)
        for field, value in zip(self.__dataclass_fields__, values, strict=True):
            object.__setattr__(self, field, value)

    @classmethod
    def from_paths(
            cls,
            candidate_train: str | Path,
            teacher_train: str | Path,
            dev_calibration: str | Path,
            shadow_audit: str | Path,
            private_evaluator: str | Path,
            exposure_ledger: str | Path,
            ) -> "V2PhysicalRoots":
        return cls(
            Path(candidate_train), Path(teacher_train), Path(dev_calibration),
            Path(shadow_audit), Path(private_evaluator), Path(exposure_ledger),
        )

    def by_root_key(self, root_key: str) -> Path:
        values = {
            "CANDIDATE_TRAIN_ROOT": self.candidate_train,
            "TEACHER_TRAIN_ROOT": self.teacher_train,
            "DEV_CALIBRATION_ROOT": self.dev_calibration,
            "SHADOW_AUDIT_ROOT": self.shadow_audit,
            "PRIVATE_EVALUATOR_ROOT": self.private_evaluator,
        }
        try:
            return values[root_key]
        except KeyError as error:
            raise V2EvaluatorBoundaryError("v2 evaluator root key is not registered") from error


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2WriteAccount:
    """Host and learning writes that evaluation authorization must keep at zero."""

    assessment_writes: int = 0
    candidate_writes: int = 0
    clock_writes: int = 0
    companion_writes: int = 0
    core_writes: int = 0
    evaluator_label_writes: int = 0
    evidence_writes: int = 0
    host_writes: int = 0
    memory_writes: int = 0
    use_writes: int = 0

    def __post_init__(self) -> None:
        if tuple(self.__dataclass_fields__) != V2_ZERO_WRITE_FIELDS:
            raise V2EvaluatorBoundaryError("v2 zero-write fields drifted")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in V2_ZERO_WRITE_FIELDS):
            raise V2EvaluatorBoundaryError("v2 evaluation write account is invalid")

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, name) == 0 for name in V2_ZERO_WRITE_FIELDS)

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in V2_ZERO_WRITE_FIELDS}


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2AccessRequest:
    """One read request against a registered owner root."""

    stage_key: str
    owner_key: str
    split: str
    record_kind: str
    relative_path: str
    content_sha256: str
    content_size_bytes: int
    purpose: str
    candidate_freeze_sha256: str | None = None
    code_freeze_sha256: str | None = None
    write_account: V2WriteAccount = V2WriteAccount()

    def __post_init__(self) -> None:
        if self.stage_key not in V2_EVALUATOR_STAGES:
            raise V2EvaluatorBoundaryError("v2 access stage is invalid")
        if not isinstance(self.owner_key, str) or not self.owner_key:
            raise V2EvaluatorBoundaryError("v2 access owner is invalid")
        if self.split not in ("train", "dev", *V2_PRIVATE_SPLITS):
            raise V2EvaluatorBoundaryError("v2 access split is invalid")
        if self.record_kind not in dict((kind, layout) for kind, layout in (
                ("source_ref", ""), ("observation", ""),
                ("teacher_evidence", ""), ("evaluator_label", ""))):
            raise V2EvaluatorBoundaryError("v2 access record kind is invalid")
        _safe_relative_path(self.relative_path)
        _sha256(self.content_sha256, where="v2 access content")
        _positive(self.content_size_bytes, where="v2 access content size")
        if not isinstance(self.purpose, str) or not self.purpose:
            raise V2EvaluatorBoundaryError("v2 access purpose is invalid")
        if not isinstance(self.write_account, V2WriteAccount):
            raise V2EvaluatorBoundaryError("v2 access write account is invalid")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2AccessPermit:
    """Authorized target plus a safe public identity without physical path."""

    owner_key: str
    root_key: str
    stage_key: str
    split: str
    record_kind: str
    target_path: Path
    content_sha256: str
    content_size_bytes: int
    path_commitment: str

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "owner_key": self.owner_key,
            "path_commitment": self.path_commitment,
            "record_kind": self.record_kind,
            "root_key": self.root_key,
            "split": self.split,
            "stage_key": self.stage_key,
        }


def _layout_matches(request: V2AccessRequest) -> bool:
    expected = {
        "source_ref": "source/source_refs.jsonl.gz",
        "observation": f"observations/{request.split}.jsonl.gz",
        "teacher_evidence": "teacher/train.evidence.jsonl.gz",
        "evaluator_label": f"evaluator/{request.split}.labels.jsonl.gz",
    }[request.record_kind]
    return request.relative_path == expected


def _has_link_component(root: Path, parts: tuple[str, ...]) -> bool:
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
        is_junction = getattr(current, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
    return False


def _resolve_file(root: Path, relative_path: str) -> Path:
    parts = _safe_relative_path(relative_path)
    if _has_link_component(root, parts):
        raise V2EvaluatorBoundaryError("v2 evaluator symlink component rejected")
    target = (root / Path(*parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise V2EvaluatorBoundaryError("v2 evaluator target is outside registered root")
    return target


def _file_digest(target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as handle:
        while chunk := handle.read(131_072):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _policy_for(owner_key: str) -> V2OwnerRootPolicy:
    for policy in V2_OWNER_ROOT_POLICIES:
        if policy.owner_key == owner_key:
            return policy
    raise V2EvaluatorBoundaryError("v2 evaluator owner is not registered")


def _validate_freeze(value: str | None, *, required: int, expected: str, where: str) -> None:
    if required:
        if value != expected:
            raise V2EvaluatorBoundaryError(f"v2 {where} freeze does not match")
    elif value is not None:
        raise V2EvaluatorBoundaryError(f"v2 {where} freeze must be omitted")


def _path_commitment(root_key: str, relative_path: str) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "relative_path": relative_path,
        "root_key": root_key,
    })).hexdigest()


def _incident_directory(root: Path, family_commitment: str) -> Path:
    return root / family_commitment


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2ExposureIncident:
    """Safe append-only incident; it never stores a path, label or surface."""

    incident_id: int
    family_commitment: str
    exposure_kind: str
    phase: str
    evidence_sha256: str
    artifact_kind: str = V2_EXPOSURE_ARTIFACT_KIND
    format_version: int = 1
    blind_pass_eligible: int = 0
    terminal_status: str = "EXPOSED_FAMILY_BLOCKED"

    def __post_init__(self) -> None:
        _positive(self.incident_id, where="v2 exposure incident id")
        _sha256(self.family_commitment, where="v2 exposure family")
        if self.exposure_kind not in V2_EXPOSURE_KINDS:
            raise V2EvaluatorBoundaryError("v2 exposure kind is invalid")
        if self.phase not in V2_EXPOSURE_PHASES:
            raise V2EvaluatorBoundaryError("v2 exposure phase is invalid")
        _sha256(self.evidence_sha256, where="v2 exposure evidence")
        if (self.artifact_kind != V2_EXPOSURE_ARTIFACT_KIND
                or self.format_version != 1 or self.blind_pass_eligible != 0
                or self.terminal_status != "EXPOSED_FAMILY_BLOCKED"):
            raise V2EvaluatorBoundaryError("v2 exposure incident identity drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "blind_pass_eligible": self.blind_pass_eligible,
            "evidence_sha256": self.evidence_sha256,
            "exposure_kind": self.exposure_kind,
            "family_commitment": self.family_commitment,
            "format_version": self.format_version,
            "incident_id": self.incident_id,
            "phase": self.phase,
            "terminal_status": self.terminal_status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2ExposureIncident":
        raw = exact_dict(value, {
            "artifact_kind", "blind_pass_eligible", "evidence_sha256",
            "exposure_kind", "family_commitment", "format_version", "incident_id",
            "phase", "terminal_status",
        }, where="V2ExposureIncident")
        return cls(
            raw["incident_id"], str(raw["family_commitment"]),
            str(raw["exposure_kind"]), str(raw["phase"]),
            str(raw["evidence_sha256"]), str(raw["artifact_kind"]),
            raw["format_version"], raw["blind_pass_eligible"],
            str(raw["terminal_status"]),
        )


def incident_evidence_commitment(
        family_commitment: str, exposure_kind: str, phase: str) -> str:
    """Commit only to incident metadata, never to the offending report."""
    _sha256(family_commitment, where="v2 incident family")
    return hashlib.sha256(canonical_json_bytes({
        "exposure_kind": exposure_kind,
        "family_commitment": family_commitment,
        "phase": phase,
    })).hexdigest()


def read_v2_exposure_incidents(
        roots: V2PhysicalRoots,
        family_commitment: str,
        ) -> tuple[V2ExposureIncident, ...]:
    """Read a family's safe incident metadata and fail closed on ledger drift."""
    if not isinstance(roots, V2PhysicalRoots):
        raise V2EvaluatorBoundaryError("v2 physical roots are invalid")
    family = _sha256(family_commitment, where="v2 exposure family")
    directory = _incident_directory(roots.exposure_ledger, family)
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        raise V2EvaluatorBoundaryError("v2 exposure family ledger is invalid")
    files = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    incidents: list[V2ExposureIncident] = []
    for ordinal, target in enumerate(files, start=1):
        expected_name = f"incident-{ordinal:08d}.json"
        if target.name != expected_name or not target.is_file() or target.is_symlink():
            raise V2EvaluatorBoundaryError("v2 exposure ledger sequence drifted")
        incident = V2ExposureIncident.from_dict(read_canonical_object(target))
        if incident.family_commitment != family or incident.incident_id != ordinal:
            raise V2EvaluatorBoundaryError("v2 exposure incident family drifted")
        validate_v2_safe_report(incident.to_dict())
        incidents.append(incident)
    return tuple(incidents)


def append_v2_exposure_incident(
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        *,
        exposure_kind: str,
        phase: str,
        ) -> V2ExposureIncident:
    """Append one irreversible safe incident without storing offending content."""
    if (not isinstance(roots, V2PhysicalRoots)
            or not isinstance(registration, V2PrivateFamilyRegistration)):
        raise V2EvaluatorBoundaryError("v2 exposure append inputs are invalid")
    existing = read_v2_exposure_incidents(roots, registration.family_commitment)
    incident = V2ExposureIncident(
        len(existing) + 1,
        registration.family_commitment,
        exposure_kind,
        phase,
        incident_evidence_commitment(
            registration.family_commitment, exposure_kind, phase),
    )
    directory = _incident_directory(
        roots.exposure_ledger, registration.family_commitment)
    directory.mkdir(exist_ok=True)
    target = directory / f"incident-{incident.incident_id:08d}.json"
    write_immutable_json(incident.to_dict(), target)
    if V2ExposureIncident.from_dict(read_canonical_object(target)) != incident:
        raise V2EvaluatorBoundaryError("v2 exposure incident readback drifted")
    return incident


def assert_v2_blind_family_eligible(
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        ) -> None:
    """Permanently reject blind authorization after the first exposure."""
    if not isinstance(registration, V2PrivateFamilyRegistration):
        raise V2EvaluatorBoundaryError("v2 family registration is invalid")
    if read_v2_exposure_incidents(roots, registration.family_commitment):
        raise V2EvaluatorBoundaryError("v2 exposed family is not blind-pass eligible")


def audit_v2_safe_report(
        value: object,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        *,
        phase: str,
        ) -> None:
    """Validate a report and append an incident before propagating exposure."""
    if phase not in V2_EXPOSURE_PHASES:
        raise V2EvaluatorBoundaryError("v2 report audit phase is invalid")
    try:
        validate_v2_safe_report(value)
    except V2ReportExposureError as error:
        append_v2_exposure_incident(
            roots, registration,
            exposure_kind=error.exposure_kind,
            phase=phase,
        )
        raise


def authorize_v2_access(
        contract: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        request: V2AccessRequest,
        *,
        registration: V2PrivateFamilyRegistration | None = None,
        ) -> V2AccessPermit:
    """Authorize one hash-bound read without parsing or returning its payload."""
    if (not isinstance(contract, V2EvaluatorBoundaryContract)
            or not isinstance(roots, V2PhysicalRoots)
            or not isinstance(request, V2AccessRequest)):
        raise V2EvaluatorBoundaryError("v2 access inputs are invalid")
    policy = next(
        (item for item in contract.owner_root_policies
         if item.owner_key == request.owner_key),
        None,
    )
    if policy is None or policy != _policy_for(request.owner_key):
        raise V2EvaluatorBoundaryError("v2 access owner policy drifted")
    if (request.split not in policy.allowed_splits
            or request.record_kind not in policy.allowed_record_kinds
            or request.purpose != policy.purpose
            or not request.write_account.is_zero
            or not _layout_matches(request)):
        raise V2EvaluatorBoundaryError("v2 access owner/split/path/write boundary rejected")
    is_private = policy.owner_key == "PH2_V2_PRIVATE_EVALUATOR"
    if is_private:
        if (not isinstance(registration, V2PrivateFamilyRegistration)
                or registration.stage_key != request.stage_key
                or request.split not in V2_PRIVATE_SPLITS):
            raise V2EvaluatorBoundaryError("v2 private family registration is missing")
        assert_v2_blind_family_eligible(roots, registration)
        _validate_freeze(
            request.candidate_freeze_sha256,
            required=policy.candidate_freeze_required,
            expected=registration.candidate_freeze_sha256,
            where="candidate",
        )
        _validate_freeze(
            request.code_freeze_sha256,
            required=policy.code_freeze_required,
            expected=registration.code_freeze_sha256,
            where="code",
        )
    else:
        if registration is not None:
            raise V2EvaluatorBoundaryError("v2 private registration crossed public owner")
        _validate_freeze(
            request.candidate_freeze_sha256, required=0, expected="", where="candidate")
        _validate_freeze(
            request.code_freeze_sha256, required=0, expected="", where="code")
    root = roots.by_root_key(policy.root_key)
    target = _resolve_file(root, request.relative_path)
    size, digest = _file_digest(target)
    if size != request.content_size_bytes or digest != request.content_sha256:
        raise V2EvaluatorBoundaryError("v2 evaluator transport identity drifted")
    permit = V2AccessPermit(
        request.owner_key, policy.root_key, request.stage_key, request.split,
        request.record_kind, target, digest, size,
        _path_commitment(policy.root_key, request.relative_path),
    )
    validate_v2_safe_report(permit.to_safe_dict())
    return permit


__all__ = [
    "V2_ACCESS_OPERATIONS", "V2_EXPOSURE_ARTIFACT_KIND", "V2_EXPOSURE_KINDS",
    "V2_EXPOSURE_PHASES", "V2AccessPermit", "V2AccessRequest",
    "V2ExposureIncident", "V2PhysicalRoots", "V2WriteAccount",
    "append_v2_exposure_incident", "assert_v2_blind_family_eligible",
    "audit_v2_safe_report", "authorize_v2_access",
    "incident_evidence_commitment", "read_v2_exposure_incidents",
]
