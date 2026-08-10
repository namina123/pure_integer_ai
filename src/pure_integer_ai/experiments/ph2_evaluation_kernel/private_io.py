"""Generic private inventory validation and V2 firewall authorization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    positive,
    sha256_text,
    text,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_PRIVATE_SPLITS,
    V2EvaluatorBoundaryContract,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2AccessPermit,
    V2AccessRequest,
    V2PhysicalRoots,
    V2WriteAccount,
    authorize_v2_access,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.manifest import (
    EvaluationKernelManifest,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)


EVALUATION_PRIVATE_RECORD_KINDS = (
    "source_ref", "observation", "evaluator_label")


def _expected_layout(split: str, record_kind: str) -> tuple[str, str]:
    if record_kind == "source_ref":
        if split:
            raise EvaluationKernelContractError("private SourceRef split must be empty")
        return "PRIVATE_SOURCE", "source/source_refs.jsonl.gz"
    if split not in V2_PRIVATE_SPLITS:
        raise EvaluationKernelContractError("private evaluation split is invalid")
    if record_kind == "observation":
        return f"PRIVATE_{split.upper()}_OBSERVATION", f"observations/{split}.jsonl.gz"
    if record_kind == "evaluator_label":
        return f"PRIVATE_{split.upper()}_LABEL", f"evaluator/{split}.labels.jsonl.gz"
    raise EvaluationKernelContractError("private evaluation record kind is invalid")


def _safe_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or path.is_absolute()
            or path.as_posix() != value or ".." in path.parts):
        raise EvaluationKernelContractError("private inventory relative path is unsafe")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class EvaluationFileIdentity:
    """Transport and decompressed content identity from safe owner metadata."""

    layout_key: str
    split: str
    record_kind: str
    relative_path: str
    transport_size_bytes: int
    transport_sha256: str
    content_size_bytes: int
    content_sha256: str
    record_count: int

    def __post_init__(self) -> None:
        text(self.layout_key, where="evaluation file layout_key")
        if not isinstance(self.split, str):
            raise EvaluationKernelContractError("evaluation file split must be text")
        if self.record_kind not in EVALUATION_PRIVATE_RECORD_KINDS:
            raise EvaluationKernelContractError("evaluation file record kind invalid")
        _safe_relative_path(self.relative_path)
        expected_key, expected_path = _expected_layout(self.split, self.record_kind)
        if self.layout_key != expected_key or self.relative_path != expected_path:
            raise EvaluationKernelContractError("evaluation file layout drifted")
        for name in (
                "transport_size_bytes", "content_size_bytes", "record_count"):
            positive(getattr(self, name), where=f"evaluation file {name}")
        sha256_text(self.transport_sha256, where="evaluation file transport")
        sha256_text(self.content_sha256, where="evaluation file content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "layout_key": self.layout_key,
            "record_count": self.record_count,
            "record_kind": self.record_kind,
            "relative_path": self.relative_path,
            "split": self.split,
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationFileIdentity":
        raw = exact_dict(value, {
            "content_sha256", "content_size_bytes", "layout_key",
            "record_count", "record_kind", "relative_path", "split",
            "transport_sha256", "transport_size_bytes",
        }, where="EvaluationFileIdentity")
        return cls(
            str(raw["layout_key"]), str(raw["split"]),
            str(raw["record_kind"]), str(raw["relative_path"]),
            raw["transport_size_bytes"], str(raw["transport_sha256"]),
            raw["content_size_bytes"], str(raw["content_sha256"]),
            raw["record_count"],
        )


def evaluation_file_inventory_sha256(
        files: tuple[EvaluationFileIdentity, ...],
        ) -> str:
    """Commit to the full ordered inventory without exposing it publicly."""
    if (not isinstance(files, tuple) or not files
            or any(not isinstance(item, EvaluationFileIdentity) for item in files)):
        raise EvaluationKernelContractError("evaluation file inventory is invalid")
    return hashlib.sha256(canonical_json_bytes(
        [item.to_dict() for item in files])).hexdigest()


def validate_evaluation_file_inventory(
        manifest: EvaluationKernelManifest,
        files: tuple[EvaluationFileIdentity, ...],
        ) -> None:
    """Require source-first and exact observation/label pairs for used splits."""
    if not isinstance(manifest, EvaluationKernelManifest):
        raise EvaluationKernelContractError("evaluation inventory manifest invalid")
    if evaluation_file_inventory_sha256(files) != (
            manifest.owner_binding.file_inventory_sha256):
        raise EvaluationKernelContractError("evaluation file inventory commitment drifted")
    if files[0].record_kind != "source_ref" or files[0].record_count != (
            manifest.source_binding.record_count):
        raise EvaluationKernelContractError("evaluation SourceRef inventory drifted")
    source_splits = tuple(dict.fromkeys(
        item.split for item in manifest.source_binding.slices))
    private_splits = tuple(split for split in V2_PRIVATE_SPLITS if split in source_splits)
    if set(source_splits) != set(private_splits) or not private_splits:
        raise EvaluationKernelContractError(
            "formal source binding must use registered private splits only")
    expected = ["PRIVATE_SOURCE"]
    for split in private_splits:
        expected.extend((
            f"PRIVATE_{split.upper()}_OBSERVATION",
            f"PRIVATE_{split.upper()}_LABEL",
        ))
    if tuple(item.layout_key for item in files) != tuple(expected):
        raise EvaluationKernelContractError("evaluation file inventory order drifted")
    observation_count = 0
    label_count = 0
    for split in private_splits:
        observation = next(item for item in files if (
            item.split == split and item.record_kind == "observation"))
        label = next(item for item in files if (
            item.split == split and item.record_kind == "evaluator_label"))
        if observation.record_count != label.record_count:
            raise EvaluationKernelContractError(
                "evaluation split observation/label counts differ")
        observation_count += observation.record_count
        label_count += label.record_count
    owner = manifest.owner_binding
    if (observation_count != owner.observation_count
            or label_count != owner.label_count
            or observation_count != owner.pair_count):
        raise EvaluationKernelContractError("evaluation owner inventory counts drifted")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class AuthorizedEvaluationFile:
    """A private identity paired with its firewall permit; never serialize the path."""

    identity: EvaluationFileIdentity
    permit: V2AccessPermit

    def __post_init__(self) -> None:
        if (not isinstance(self.identity, EvaluationFileIdentity)
                or not isinstance(self.permit, V2AccessPermit)):
            raise EvaluationKernelContractError("authorized evaluation file invalid")
        if (self.permit.root_key != "PRIVATE_EVALUATOR_ROOT"
                or self.permit.record_kind != self.identity.record_kind
                or self.permit.content_sha256 != self.identity.transport_sha256
                or self.permit.content_size_bytes != self.identity.transport_size_bytes):
            raise EvaluationKernelContractError(
                "evaluation file identity and firewall permit differ")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.identity.content_sha256,
            "content_size_bytes": self.identity.content_size_bytes,
            "layout_key": self.identity.layout_key,
            "record_count": self.identity.record_count,
            "transport": self.permit.to_safe_dict(),
        }


def authorize_evaluation_files(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        manifest: EvaluationKernelManifest,
        files: tuple[EvaluationFileIdentity, ...],
        ) -> tuple[AuthorizedEvaluationFile, ...]:
    """Reuse the existing owner/path firewall for a manifest-bound inventory."""
    if (not isinstance(boundary, V2EvaluatorBoundaryContract)
            or not isinstance(roots, V2PhysicalRoots)):
        raise EvaluationKernelContractError("evaluation firewall inputs invalid")
    validate_evaluation_file_inventory(manifest, files)
    registration = manifest.transport_registration()
    first_private_split = next(
        item.split for item in files if item.record_kind == "observation")
    authorized = []
    for identity in files:
        request = V2AccessRequest(
            manifest.stage_key,
            "PH2_V2_PRIVATE_EVALUATOR",
            identity.split or first_private_split,
            identity.record_kind,
            identity.relative_path,
            identity.transport_sha256,
            identity.transport_size_bytes,
            "PRIVATE_EVALUATION",
            registration.candidate_freeze_sha256,
            registration.code_freeze_sha256,
            V2WriteAccount(),
        )
        authorized.append(AuthorizedEvaluationFile(
            identity,
            authorize_v2_access(
                boundary, roots, request, registration=registration),
        ))
    return tuple(authorized)


__all__ = [
    "EVALUATION_PRIVATE_RECORD_KINDS",
    "AuthorizedEvaluationFile",
    "EvaluationFileIdentity",
    "authorize_evaluation_files",
    "evaluation_file_inventory_sha256",
    "validate_evaluation_file_inventory",
]
