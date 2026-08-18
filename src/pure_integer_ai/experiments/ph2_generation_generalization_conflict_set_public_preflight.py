"""Public preflight freeze for the independent ``CONFLICT_SET`` family.

The freeze binds only public source code and the label-free handoff sample.
It does not create a candidate, read private labels, or execute a formal run.
"""
from __future__ import annotations

import ast
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
    FAMILY_NAMESPACE,
    NEGATIVE_MATRIX_CASE_COUNT,
    ConflictSetOwnerHandoffError,
    read_conflict_set_owner_handoff,
)


ARTIFACT_KIND = "PH2_GG03_CONFLICT_SET_PUBLIC_PREFLIGHT_FREEZE_V1"
FORMAT_VERSION = 1
PUBLIC_PREFLIGHT_STATUS = "PUBLIC_PREFLIGHT_FROZEN"
PUBLIC_POSITIVE_RUNTIME_CASE_COUNT = 2
PUBLIC_SAMPLE_RELATIVE_PATH = "data/ph2/conflict_set_owner_handoff_v1.jsonl.sample"
ROOT_MODULES = (
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract",
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_connector",
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_order",
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_handoff",
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_postcheck",
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_surface",
)


class ConflictSetPublicPreflightError(ValueError):
    """The public preflight freeze is incomplete or inconsistent."""


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConflictSetPublicPreflightError(f"{where} must be non-empty text")
    return value


def _sha(value: object, *, where: str, length: int) -> str:
    result = _text(value, where=where).lower()
    if (len(result) != length
            or any(char not in "0123456789abcdef" for char in result)):
        raise ConflictSetPublicPreflightError(
            f"{where} must be a hexadecimal SHA-{length * 4}")
    return result


def _positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConflictSetPublicPreflightError(
            f"{where} must be a positive strict integer")
    return value


def _zero(value: object, *, where: str) -> int:
    if type(value) is not int or value != 0:
        raise ConflictSetPublicPreflightError(f"{where} must be zero")
    return value


def _exact(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ConflictSetPublicPreflightError(
            f"{where} has missing or unknown fields")
    return value


def _relative(value: object, *, where: str) -> str:
    result = _text(value, where=where)
    path = Path(result)
    if (path.is_absolute() or "\\" in result
            or any(part in {"", ".", ".."} for part in result.split("/"))):
        raise ConflictSetPublicPreflightError(
            f"{where} must be a safe POSIX relative path")
    return result


def _module_file(repository: Path, module: str) -> Path | None:
    relative = Path("src", *module.split("."))
    direct = repository / relative.with_suffix(".py")
    package = repository / relative / "__init__.py"
    if direct.is_file():
        return direct.resolve()
    if package.is_file():
        return package.resolve()
    return None


def _imported_modules(path: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ConflictSetPublicPreflightError(
            f"cannot parse code closure file: {path}") from error
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                item.name for item in node.names
                if item.name == "pure_integer_ai"
                or item.name.startswith("pure_integer_ai."))
        elif (isinstance(node, ast.ImportFrom) and node.level == 0
              and isinstance(node.module, str)
              and (node.module == "pure_integer_ai"
                   or node.module.startswith("pure_integer_ai."))):
            modules.add(node.module)
            modules.update(
                f"{node.module}.{item.name}" for item in node.names)
    return tuple(sorted(modules))


@dataclass(frozen=True, slots=True, order=True)
class ConflictSetCodeFile:
    """One repository-relative production file in the public code closure."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _relative(self.relative_path, where="code_file.relative_path")
        _positive(self.size_bytes, where="code_file.size_bytes")
        _sha(self.sha256, where="code_file.sha256", length=64)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetCodeFile":
        raw = _exact(value, {"relative_path", "sha256", "size_bytes"},
                     where="code_file")
        return cls(raw["relative_path"], raw["size_bytes"], raw["sha256"])


@dataclass(frozen=True, slots=True)
class ConflictSetCodeIdentity:
    """Deterministic aggregate identity of the public runtime import closure."""

    files: tuple[ConflictSetCodeFile, ...]
    aggregate_sha256: str

    def __post_init__(self) -> None:
        if (not self.files or tuple(sorted(self.files)) != self.files
                or len({item.relative_path for item in self.files})
                != len(self.files)):
            raise ConflictSetPublicPreflightError(
                "code identity files must be sorted and unique")
        _sha(self.aggregate_sha256,
             where="code_identity.aggregate_sha256", length=64)
        expected = hashlib.sha256(canonical_json_bytes(
            [item.to_dict() for item in self.files])).hexdigest()
        if expected != self.aggregate_sha256:
            raise ConflictSetPublicPreflightError(
                "code identity aggregate hash drifted")
        relative_paths = {item.relative_path for item in self.files}
        required = {
            f"src/{Path(*module.split('.')).with_suffix('.py').as_posix()}"
            for module in ROOT_MODULES
        }
        if not required <= relative_paths:
            raise ConflictSetPublicPreflightError(
                "code identity omitted a conflict-set root module")

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetCodeIdentity":
        raw = _exact(value, {"aggregate_sha256", "files"},
                     where="code_identity")
        if not isinstance(raw["files"], list):
            raise ConflictSetPublicPreflightError(
                "code_identity.files must be an array")
        return cls(
            tuple(ConflictSetCodeFile.from_dict(item) for item in raw["files"]),
            raw["aggregate_sha256"],
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_conflict_set_code_identity(
        repository_root: str | Path) -> ConflictSetCodeIdentity:
    """Freeze the complete import closure of the public conflict-set runtime."""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise ConflictSetPublicPreflightError("repository root is invalid")
    pending = list(ROOT_MODULES)
    visited: set[Path] = set()
    while pending:
        module = pending.pop()
        path = _module_file(repository, module)
        if path is None or path in visited:
            continue
        visited.add(path)
        pending.extend(_imported_modules(path))
    if not visited:
        raise ConflictSetPublicPreflightError("public code closure is empty")
    files = tuple(sorted(
        ConflictSetCodeFile(
            path.relative_to(repository).as_posix(),
            path.stat().st_size,
            _sha256_file(path),
        )
        for path in visited
    ))
    return ConflictSetCodeIdentity(
        files,
        hashlib.sha256(canonical_json_bytes(
            [item.to_dict() for item in files])).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ConflictSetPublicPreflightFreeze:
    """Public freeze metadata before private evaluator creation."""

    code_identity: ConflictSetCodeIdentity
    public_head_sha1: str
    sample_relative_path: str
    sample_sha256: str
    sample_size_bytes: int
    positive_runtime_case_count: int
    negative_matrix_case_count: int
    status: str = PUBLIC_PREFLIGHT_STATUS
    teacher_api_llm_call_count: int = 0
    private_label_read_count: int = 0
    formal_run_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.code_identity, ConflictSetCodeIdentity):
            raise TypeError("code_identity type is invalid")
        _sha(self.public_head_sha1, where="public_head_sha1", length=40)
        if self.sample_relative_path != PUBLIC_SAMPLE_RELATIVE_PATH:
            raise ConflictSetPublicPreflightError(
                "public sample path is not frozen")
        _sha(self.sample_sha256, where="sample_sha256", length=64)
        _positive(self.sample_size_bytes, where="sample_size_bytes")
        if self.positive_runtime_case_count != PUBLIC_POSITIVE_RUNTIME_CASE_COUNT:
            raise ConflictSetPublicPreflightError(
                "positive runtime case count is not frozen")
        if self.negative_matrix_case_count != NEGATIVE_MATRIX_CASE_COUNT:
            raise ConflictSetPublicPreflightError(
                "negative matrix case count is not frozen")
        if self.status != PUBLIC_PREFLIGHT_STATUS:
            raise ConflictSetPublicPreflightError("preflight status is invalid")
        for name in (
                "teacher_api_llm_call_count", "private_label_read_count",
                "formal_run_count"):
            _zero(getattr(self, name), where=name)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": ARTIFACT_KIND,
            "capability_key": CAPABILITY_KEY,
            "code_identity": self.code_identity.to_dict(),
            "family_namespace": FAMILY_NAMESPACE,
            "formal_run_count": self.formal_run_count,
            "format_version": FORMAT_VERSION,
            "negative_matrix_case_count": self.negative_matrix_case_count,
            "positive_runtime_case_count": self.positive_runtime_case_count,
            "private_label_read_count": self.private_label_read_count,
            "public_head_sha1": self.public_head_sha1,
            "sample_relative_path": self.sample_relative_path,
            "sample_sha256": self.sample_sha256,
            "sample_size_bytes": self.sample_size_bytes,
            "status": self.status,
            "teacher_api_llm_call_count": self.teacher_api_llm_call_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetPublicPreflightFreeze":
        raw = _exact(value, {
            "artifact_kind", "capability_key", "code_identity",
            "family_namespace", "formal_run_count", "format_version",
            "negative_matrix_case_count", "positive_runtime_case_count",
            "private_label_read_count", "public_head_sha1",
            "sample_relative_path", "sample_sha256", "sample_size_bytes",
            "status", "teacher_api_llm_call_count",
        }, where="public_preflight_freeze")
        if (raw["artifact_kind"] != ARTIFACT_KIND
                or raw["format_version"] != FORMAT_VERSION
                or raw["capability_key"] != CAPABILITY_KEY
                or raw["family_namespace"] != FAMILY_NAMESPACE):
            raise ConflictSetPublicPreflightError(
                "public preflight identity or version drifted")
        return cls(
            ConflictSetCodeIdentity.from_dict(raw["code_identity"]),
            raw["public_head_sha1"], raw["sample_relative_path"],
            raw["sample_sha256"], raw["sample_size_bytes"],
            raw["positive_runtime_case_count"],
            raw["negative_matrix_case_count"], raw["status"],
            raw["teacher_api_llm_call_count"],
            raw["private_label_read_count"], raw["formal_run_count"],
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_conflict_set_public_preflight_freeze(
        repository_root: str | Path,
        *,
        public_head_sha1: str,
        ) -> ConflictSetPublicPreflightFreeze:
    """Bind the actual public sample and code closure without running evaluators."""
    repository = Path(repository_root).resolve()
    sample = (repository / Path(*PUBLIC_SAMPLE_RELATIVE_PATH.split("/"))).resolve()
    if not sample.is_file() or not sample.is_relative_to(repository):
        raise ConflictSetPublicPreflightError("public sample is missing")
    try:
        handoff = read_conflict_set_owner_handoff(sample)
    except (OSError, ValueError, ConflictSetOwnerHandoffError) as error:
        raise ConflictSetPublicPreflightError(
            "public handoff sample failed contract readback") from error
    if (handoff.capability_key != CAPABILITY_KEY
            or handoff.family_namespace != FAMILY_NAMESPACE
            or handoff.public_preflight.negative_matrix_case_count
            != NEGATIVE_MATRIX_CASE_COUNT):
        raise ConflictSetPublicPreflightError(
            "public sample is bound to a different family")
    payload = sample.read_bytes()
    return ConflictSetPublicPreflightFreeze(
        build_conflict_set_code_identity(repository),
        public_head_sha1,
        PUBLIC_SAMPLE_RELATIVE_PATH,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        PUBLIC_POSITIVE_RUNTIME_CASE_COUNT,
        NEGATIVE_MATRIX_CASE_COUNT,
    )


def parse_conflict_set_public_preflight_bytes(
        payload: bytes) -> ConflictSetPublicPreflightFreeze:
    """Parse one canonical public preflight freeze record."""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise ConflictSetPublicPreflightError("freeze must be one JSONL record")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except (TypeError, ValueError) as error:
        raise ConflictSetPublicPreflightError(
            "freeze JSON is not canonical") from error
    if canonical_json_line(value) != payload:
        raise ConflictSetPublicPreflightError("freeze JSON bytes are not canonical")
    return ConflictSetPublicPreflightFreeze.from_dict(value)


def read_conflict_set_public_preflight(
        path: str | Path) -> ConflictSetPublicPreflightFreeze:
    """Read one public freeze record without side effects."""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ConflictSetPublicPreflightError("freeze file is unreadable") from error
    return parse_conflict_set_public_preflight_bytes(payload)


__all__ = [
    "ARTIFACT_KIND",
    "FORMAT_VERSION",
    "PUBLIC_PREFLIGHT_STATUS",
    "PUBLIC_POSITIVE_RUNTIME_CASE_COUNT",
    "PUBLIC_SAMPLE_RELATIVE_PATH",
    "ROOT_MODULES",
    "ConflictSetCodeFile",
    "ConflictSetCodeIdentity",
    "ConflictSetPublicPreflightError",
    "ConflictSetPublicPreflightFreeze",
    "build_conflict_set_code_identity",
    "build_conflict_set_public_preflight_freeze",
    "parse_conflict_set_public_preflight_bytes",
    "read_conflict_set_public_preflight",
]
