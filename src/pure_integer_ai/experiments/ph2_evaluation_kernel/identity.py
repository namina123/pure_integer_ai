"""Live semantic identity for the complete generic evaluation kernel."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03FileIdentity,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)


EVALUATION_KERNEL_IDENTITY_VERSION = "PH2-EVALUATION-KERNEL-IDENTITY-V1"
EVALUATION_KERNEL_MODULES = (
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/__init__.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/aggregate.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/guard.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/identity.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/manifest.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/owner_receipt.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/plugin.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/preflight.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/private_io.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/publication.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/records.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/runtime.py",
    "src/pure_integer_ai/experiments/ph2_evaluation_kernel/source_binding.py",
)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationKernelIdentity:
    """Exact module inventory and the semantic SHA committed by new families."""

    identity_version: str
    files: tuple[D03FileIdentity, ...]
    semantic_sha256: str

    def __post_init__(self) -> None:
        if self.identity_version != EVALUATION_KERNEL_IDENTITY_VERSION:
            raise EvaluationKernelContractError("evaluation kernel identity version drifted")
        if (not isinstance(self.files, tuple)
                or tuple(item.relative_path for item in self.files)
                != EVALUATION_KERNEL_MODULES):
            raise EvaluationKernelContractError("evaluation kernel module inventory drifted")
        expected = hashlib.sha256(canonical_json_bytes({
            "files": [item.to_dict() for item in self.files],
            "identity_version": self.identity_version,
        })).hexdigest()
        if self.semantic_sha256 != expected:
            raise EvaluationKernelContractError("evaluation kernel semantic SHA drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [item.to_dict() for item in self.files],
            "identity_version": self.identity_version,
            "semantic_sha256": self.semantic_sha256,
        }


def build_evaluation_kernel_identity(
        repository_root: str | Path,
        ) -> EvaluationKernelIdentity:
    """Hash every registered kernel module from the live public repository."""
    repository = Path(repository_root).resolve()
    files = []
    for relative in EVALUATION_KERNEL_MODULES:
        target = repository / Path(*relative.split("/"))
        if target.is_symlink() or not target.is_file():
            raise EvaluationKernelContractError("evaluation kernel module is missing")
        payload = target.read_bytes()
        files.append(D03FileIdentity(
            relative, len(payload), hashlib.sha256(payload).hexdigest()))
    value = {
        "files": [item.to_dict() for item in files],
        "identity_version": EVALUATION_KERNEL_IDENTITY_VERSION,
    }
    return EvaluationKernelIdentity(
        EVALUATION_KERNEL_IDENTITY_VERSION,
        tuple(files),
        hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    )


def evaluation_kernel_semantic_sha256(repository_root: str | Path) -> str:
    """Return the live kernel semantic SHA used by family builders."""
    return build_evaluation_kernel_identity(repository_root).semantic_sha256


__all__ = [
    "EVALUATION_KERNEL_IDENTITY_VERSION",
    "EVALUATION_KERNEL_MODULES",
    "EvaluationKernelIdentity",
    "build_evaluation_kernel_identity",
    "evaluation_kernel_semantic_sha256",
]
