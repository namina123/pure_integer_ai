"""``CONFLICT_SET`` 正式运行前的 metadata-only family freeze 合同。"""
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
    CODE_IDENTITY,
    FAMILY_NAMESPACE,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_metadata import (
    ConflictSetOwnerMetadata,
    ConflictSetOwnerMetadataError,
    build_conflict_set_run_guard_from_owner_metadata,
    validate_conflict_set_owner_metadata,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    ConflictSetPrivateProtocolError,
    ConflictSetPrivateTransport,
    ConflictSetRunGuard,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_public_preflight import (
    ConflictSetPublicPreflightError,
    ConflictSetPublicPreflightFreeze,
    parse_conflict_set_public_preflight_bytes,
)


FAMILY_FREEZE_ARTIFACT_KIND = "PH2_GG03_CONFLICT_SET_FAMILY_FREEZE_V1"
FAMILY_FREEZE_FORMAT_VERSION = 1
FAMILY_FREEZE_STATUS = "FROZEN_NOT_RUN_PRIVATE_LABELS_UNREAD"
PUBLIC_PREFLIGHT_MANIFEST_RELATIVE_PATH = (
    "data/ph2/manifests/gg03_conflict_set_public_preflight_v1.json")
PUBLIC_PREFLIGHT_MANIFEST_SHA256 = (
    "5e1ba013d2889108169678370319cc43e7f492dd2b2f1a53d88a678767afa7f4")
FAMILY_CODE_ROOT_MODULES = (
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol",
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_metadata",
    "pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_family",
)


# object-model: exception
class ConflictSetFamilyFreezeError(ValueError):
    """family freeze 无法在不读取 private payload 的条件下闭合。"""


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConflictSetFamilyFreezeError(f"{where} must be non-empty text")
    return value


def _sha(value: object, *, where: str) -> str:
    result = _text(value, where=where)
    if (len(result) != 64 or result != result.lower()
            or any(char not in "0123456789abcdef" for char in result)):
        raise ConflictSetFamilyFreezeError(
            f"{where} must be a lowercase SHA-256")
    return result


def _positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConflictSetFamilyFreezeError(
            f"{where} must be a positive strict integer")
    return value


def _zero(value: object, *, where: str) -> int:
    if type(value) is not int or value != 0:
        raise ConflictSetFamilyFreezeError(f"{where} must be zero")
    return value


def _exact(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ConflictSetFamilyFreezeError(
            f"{where} has missing or unknown fields")
    return value


def _relative(value: object, *, where: str) -> str:
    result = _text(value, where=where)
    if (Path(result).is_absolute() or "\\" in result
            or any(part in {"", ".", ".."} for part in result.split("/"))):
        raise ConflictSetFamilyFreezeError(
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
        raise ConflictSetFamilyFreezeError(
            f"cannot parse family code closure file: {path}") from error
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
            modules.update(f"{node.module}.{item.name}" for item in node.names)
    return tuple(sorted(modules))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class ConflictSetFamilyCodeFile:
    """family/transport/owner 代码闭包中的一份仓库相对文件身份。"""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _relative(self.relative_path, where="family_code_file.relative_path")
        _positive(self.size_bytes, where="family_code_file.size_bytes")
        _sha(self.sha256, where="family_code_file.sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetFamilyCodeFile":
        raw = _exact(value, {"relative_path", "sha256", "size_bytes"},
                     where="family_code_file")
        return cls(raw["relative_path"], raw["size_bytes"], raw["sha256"])


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetFamilyCodeIdentity:
    """当前 private transport、owner reader 与 family freeze 的代码身份。"""

    files: tuple[ConflictSetFamilyCodeFile, ...]
    aggregate_sha256: str
    root_modules: tuple[str, ...] = FAMILY_CODE_ROOT_MODULES

    def __post_init__(self) -> None:
        if self.root_modules != FAMILY_CODE_ROOT_MODULES:
            raise ConflictSetFamilyFreezeError(
                "family code identity root modules drifted")
        if (not self.files or tuple(sorted(self.files)) != self.files
                or len({item.relative_path for item in self.files})
                != len(self.files)):
            raise ConflictSetFamilyFreezeError(
                "family code identity files must be sorted and unique")
        _sha(self.aggregate_sha256,
             where="family_code_identity.aggregate_sha256")
        expected = hashlib.sha256(canonical_json_bytes(
            [item.to_dict() for item in self.files])).hexdigest()
        if self.aggregate_sha256 != expected:
            raise ConflictSetFamilyFreezeError(
                "family code identity aggregate hash drifted")
        relative_paths = {item.relative_path for item in self.files}
        required = {
            f"src/{Path(*module.split('.')).with_suffix('.py').as_posix()}"
            for module in FAMILY_CODE_ROOT_MODULES
        }
        if not required <= relative_paths:
            raise ConflictSetFamilyFreezeError(
                "family code identity omitted a required root module")

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "files": [item.to_dict() for item in self.files],
            "root_modules": list(self.root_modules),
        }

    def canonical_bytes(self) -> bytes:
        """返回 owner 可物化为 code-freeze 的规范 JSONL 字节。"""
        return canonical_json_line(self.to_dict())

    def commitment_sha256(self) -> str:
        """返回 code-freeze 规范传输的 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetFamilyCodeIdentity":
        raw = _exact(value, {"aggregate_sha256", "files", "root_modules"},
                     where="family_code_identity")
        if not isinstance(raw["files"], list) or not isinstance(
                raw["root_modules"], list):
            raise ConflictSetFamilyFreezeError(
                "family code identity arrays are invalid")
        return cls(
            tuple(ConflictSetFamilyCodeFile.from_dict(item)
                  for item in raw["files"]),
            raw["aggregate_sha256"], tuple(raw["root_modules"]),
        )


def build_conflict_set_family_code_identity(
        repository_root: str | Path,
        ) -> ConflictSetFamilyCodeIdentity:
    """递归冻结 private protocol、owner reader 和 family 模块 import 闭包。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise ConflictSetFamilyFreezeError("repository root is invalid")
    pending = list(FAMILY_CODE_ROOT_MODULES)
    visited: set[Path] = set()
    while pending:
        module = pending.pop()
        path = _module_file(repository, module)
        if path is None or path in visited:
            continue
        visited.add(path)
        pending.extend(_imported_modules(path))
    if not visited:
        raise ConflictSetFamilyFreezeError("family code closure is empty")
    files = tuple(sorted(
        ConflictSetFamilyCodeFile(
            path.relative_to(repository).as_posix(),
            path.stat().st_size,
            _sha256_file(path),
        )
        for path in visited
    ))
    return ConflictSetFamilyCodeIdentity(
        files,
        hashlib.sha256(canonical_json_bytes(
            [item.to_dict() for item in files])).hexdigest(),
    )


def _artifact_inventory_sha256(transport: ConflictSetPrivateTransport) -> str:
    return hashlib.sha256(canonical_json_bytes(
        [item.to_dict() for item in transport.artifacts])).hexdigest()


def _family_body(
        *,
        public_preflight: ConflictSetPublicPreflightFreeze,
        family_code_identity: ConflictSetFamilyCodeIdentity,
        transport: ConflictSetPrivateTransport,
        owner_metadata: ConflictSetOwnerMetadata,
        available_guard: ConflictSetRunGuard,
        artifact_inventory_sha256: str,
        family_namespace: str,
        capability_key: str,
        public_code_identity: str,
        public_preflight_manifest_relative_path: str,
        public_preflight_manifest_sha256: str,
        unique_formal_run_limit: int,
        formal_run_count_before: int,
        private_payload_reads_before: int,
        host_learning_writes_before: int,
        label_writes_before: int,
        clone_evaluation_writes_before: int,
        teacher_api_llm_calls_before: int,
        status: str,
        ) -> dict[str, object]:
    return {
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "artifact_kind": FAMILY_FREEZE_ARTIFACT_KIND,
        "available_guard": available_guard.to_dict(),
        "capability_key": capability_key,
        "clone_evaluation_writes_before": clone_evaluation_writes_before,
        "family_code_identity": family_code_identity.to_dict(),
        "family_namespace": family_namespace,
        "formal_run_count_before": formal_run_count_before,
        "format_version": FAMILY_FREEZE_FORMAT_VERSION,
        "host_learning_writes_before": host_learning_writes_before,
        "label_writes_before": label_writes_before,
        "owner_metadata": owner_metadata.to_dict(),
        "private_payload_reads_before": private_payload_reads_before,
        "public_code_identity": public_code_identity,
        "public_preflight": public_preflight.to_dict(),
        "public_preflight_manifest_relative_path": (
            public_preflight_manifest_relative_path),
        "public_preflight_manifest_sha256": (
            public_preflight_manifest_sha256),
        "status": status,
        "teacher_api_llm_calls_before": teacher_api_llm_calls_before,
        "transport": transport.to_dict(),
        "unique_formal_run_limit": unique_formal_run_limit,
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetFamilyFreeze:
    """真实 owner 到达后可发布，但尚未运行的完整 pre-run manifest。"""

    public_preflight: ConflictSetPublicPreflightFreeze
    family_code_identity: ConflictSetFamilyCodeIdentity
    transport: ConflictSetPrivateTransport
    owner_metadata: ConflictSetOwnerMetadata
    available_guard: ConflictSetRunGuard
    artifact_inventory_sha256: str
    family_commitment_sha256: str
    family_namespace: str = FAMILY_NAMESPACE
    capability_key: str = CAPABILITY_KEY
    public_code_identity: str = CODE_IDENTITY
    public_preflight_manifest_relative_path: str = (
        PUBLIC_PREFLIGHT_MANIFEST_RELATIVE_PATH)
    public_preflight_manifest_sha256: str = PUBLIC_PREFLIGHT_MANIFEST_SHA256
    unique_formal_run_limit: int = 1
    formal_run_count_before: int = 0
    private_payload_reads_before: int = 0
    host_learning_writes_before: int = 0
    label_writes_before: int = 0
    clone_evaluation_writes_before: int = 0
    teacher_api_llm_calls_before: int = 0
    status: str = FAMILY_FREEZE_STATUS

    def __post_init__(self) -> None:
        if (not isinstance(self.public_preflight,
                           ConflictSetPublicPreflightFreeze)
                or not isinstance(self.family_code_identity,
                                  ConflictSetFamilyCodeIdentity)
                or not isinstance(self.transport, ConflictSetPrivateTransport)
                or not isinstance(self.owner_metadata,
                                  ConflictSetOwnerMetadata)
                or not isinstance(self.available_guard, ConflictSetRunGuard)):
            raise TypeError("family freeze typed inputs are invalid")
        if (self.family_namespace != FAMILY_NAMESPACE
                or self.capability_key != CAPABILITY_KEY
                or self.public_code_identity != CODE_IDENTITY):
            raise ConflictSetFamilyFreezeError(
                "family freeze identity drifted")
        if self.public_preflight_manifest_relative_path != (
                PUBLIC_PREFLIGHT_MANIFEST_RELATIVE_PATH):
            raise ConflictSetFamilyFreezeError(
                "public preflight manifest path drifted")
        _sha(self.public_preflight_manifest_sha256,
             where="public_preflight_manifest_sha256")
        if (self.public_preflight_manifest_sha256
                != PUBLIC_PREFLIGHT_MANIFEST_SHA256
                or self.public_preflight.sha256()
                != PUBLIC_PREFLIGHT_MANIFEST_SHA256):
            raise ConflictSetFamilyFreezeError(
                "frozen public preflight V1 drifted")
        if self.transport.public_preflight_manifest_sha256 != (
                PUBLIC_PREFLIGHT_MANIFEST_SHA256):
            raise ConflictSetFamilyFreezeError(
                "transport does not bind frozen public preflight V1")
        try:
            validate_conflict_set_owner_metadata(
                self.transport, self.owner_metadata)
            expected_guard = build_conflict_set_run_guard_from_owner_metadata(
                self.transport, self.owner_metadata)
        except (ConflictSetOwnerMetadataError,
                ConflictSetPrivateProtocolError) as error:
            raise ConflictSetFamilyFreezeError(
                "owner metadata does not close the family transport") from error
        if self.available_guard != expected_guard:
            raise ConflictSetFamilyFreezeError(
                "available guard drifted from owner metadata")
        by_role = {item.role: item for item in self.transport.artifacts}
        code_freeze = by_role["code_freeze"]
        code_bytes = self.family_code_identity.canonical_bytes()
        code_sha = hashlib.sha256(code_bytes).hexdigest()
        if (code_freeze.transport_sha256 != code_sha
                or code_freeze.content_sha256 != code_sha
                or code_freeze.transport_size_bytes != len(code_bytes)
                or code_freeze.content_size_bytes != len(code_bytes)
                or code_freeze.record_count != 1):
            raise ConflictSetFamilyFreezeError(
                "code-freeze artifact does not bind current family code")
        public_artifact = by_role["public_preflight"]
        public_bytes = self.public_preflight.canonical_bytes()
        if (public_artifact.transport_sha256
                != PUBLIC_PREFLIGHT_MANIFEST_SHA256
                or public_artifact.content_sha256
                != PUBLIC_PREFLIGHT_MANIFEST_SHA256
                or public_artifact.transport_size_bytes != len(public_bytes)
                or public_artifact.content_size_bytes != len(public_bytes)
                or public_artifact.record_count != 1):
            raise ConflictSetFamilyFreezeError(
                "public-preflight artifact does not bind frozen V1 bytes")
        _sha(self.artifact_inventory_sha256,
             where="artifact_inventory_sha256")
        if self.artifact_inventory_sha256 != _artifact_inventory_sha256(
                self.transport):
            raise ConflictSetFamilyFreezeError(
                "artifact inventory commitment drifted")
        if (type(self.unique_formal_run_limit) is not int
                or self.unique_formal_run_limit != 1):
            raise ConflictSetFamilyFreezeError(
                "unique formal run limit must be one")
        for name in (
                "formal_run_count_before", "private_payload_reads_before",
                "host_learning_writes_before", "label_writes_before",
                "clone_evaluation_writes_before",
                "teacher_api_llm_calls_before"):
            _zero(getattr(self, name), where=name)
        if self.status != FAMILY_FREEZE_STATUS:
            raise ConflictSetFamilyFreezeError(
                "family freeze status is invalid")
        _sha(self.family_commitment_sha256,
             where="family_commitment_sha256")
        expected_commitment = hashlib.sha256(canonical_json_bytes(
            self._body())).hexdigest()
        if self.family_commitment_sha256 != expected_commitment:
            raise ConflictSetFamilyFreezeError(
                "family freeze self-commitment drifted")

    def _body(self) -> dict[str, object]:
        return _family_body(
            public_preflight=self.public_preflight,
            family_code_identity=self.family_code_identity,
            transport=self.transport,
            owner_metadata=self.owner_metadata,
            available_guard=self.available_guard,
            artifact_inventory_sha256=self.artifact_inventory_sha256,
            family_namespace=self.family_namespace,
            capability_key=self.capability_key,
            public_code_identity=self.public_code_identity,
            public_preflight_manifest_relative_path=(
                self.public_preflight_manifest_relative_path),
            public_preflight_manifest_sha256=(
                self.public_preflight_manifest_sha256),
            unique_formal_run_limit=self.unique_formal_run_limit,
            formal_run_count_before=self.formal_run_count_before,
            private_payload_reads_before=self.private_payload_reads_before,
            host_learning_writes_before=self.host_learning_writes_before,
            label_writes_before=self.label_writes_before,
            clone_evaluation_writes_before=(
                self.clone_evaluation_writes_before),
            teacher_api_llm_calls_before=self.teacher_api_llm_calls_before,
            status=self.status,
        )

    def to_dict(self) -> dict[str, object]:
        value = self._body()
        value["family_commitment_sha256"] = self.family_commitment_sha256
        return value

    def canonical_bytes(self) -> bytes:
        """返回一条包含换行的 canonical family-freeze JSONL。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回完整 family-freeze transport SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetFamilyFreeze":
        raw = _exact(value, {
            "artifact_inventory_sha256", "artifact_kind",
            "available_guard", "capability_key",
            "clone_evaluation_writes_before", "family_code_identity",
            "family_commitment_sha256", "family_namespace",
            "formal_run_count_before", "format_version",
            "host_learning_writes_before", "label_writes_before",
            "owner_metadata", "private_payload_reads_before",
            "public_code_identity", "public_preflight",
            "public_preflight_manifest_relative_path",
            "public_preflight_manifest_sha256", "status",
            "teacher_api_llm_calls_before", "transport",
            "unique_formal_run_limit",
        }, where="family_freeze")
        if (raw["artifact_kind"] != FAMILY_FREEZE_ARTIFACT_KIND
                or raw["format_version"] != FAMILY_FREEZE_FORMAT_VERSION):
            raise ConflictSetFamilyFreezeError(
                "family freeze artifact kind or version is invalid")
        try:
            return cls(
                ConflictSetPublicPreflightFreeze.from_dict(
                    raw["public_preflight"]),
                ConflictSetFamilyCodeIdentity.from_dict(
                    raw["family_code_identity"]),
                ConflictSetPrivateTransport.from_dict(raw["transport"]),
                ConflictSetOwnerMetadata.from_dict(raw["owner_metadata"]),
                ConflictSetRunGuard.from_dict(raw["available_guard"]),
                raw["artifact_inventory_sha256"],
                raw["family_commitment_sha256"],
                raw["family_namespace"], raw["capability_key"],
                raw["public_code_identity"],
                raw["public_preflight_manifest_relative_path"],
                raw["public_preflight_manifest_sha256"],
                raw["unique_formal_run_limit"],
                raw["formal_run_count_before"],
                raw["private_payload_reads_before"],
                raw["host_learning_writes_before"],
                raw["label_writes_before"],
                raw["clone_evaluation_writes_before"],
                raw["teacher_api_llm_calls_before"], raw["status"],
            )
        except (ConflictSetOwnerMetadataError,
                ConflictSetPrivateProtocolError,
                ConflictSetPublicPreflightError, TypeError, ValueError) as error:
            if isinstance(error, ConflictSetFamilyFreezeError):
                raise
            raise ConflictSetFamilyFreezeError(
                "family freeze nested contract is invalid") from error


def build_conflict_set_family_freeze(
        *,
        public_preflight: ConflictSetPublicPreflightFreeze,
        family_code_identity: ConflictSetFamilyCodeIdentity,
        transport: ConflictSetPrivateTransport,
        owner_metadata: ConflictSetOwnerMetadata,
        ) -> ConflictSetFamilyFreeze:
    """只从已核验 metadata 构造 freeze；不打开路径或发布任何文件。"""
    if (not isinstance(public_preflight, ConflictSetPublicPreflightFreeze)
            or not isinstance(family_code_identity,
                              ConflictSetFamilyCodeIdentity)
            or not isinstance(transport, ConflictSetPrivateTransport)
            or not isinstance(owner_metadata, ConflictSetOwnerMetadata)):
        raise TypeError("family freeze build inputs are invalid")
    guard = build_conflict_set_run_guard_from_owner_metadata(
        transport, owner_metadata)
    inventory_sha = _artifact_inventory_sha256(transport)
    body = _family_body(
        public_preflight=public_preflight,
        family_code_identity=family_code_identity,
        transport=transport,
        owner_metadata=owner_metadata,
        available_guard=guard,
        artifact_inventory_sha256=inventory_sha,
        family_namespace=FAMILY_NAMESPACE,
        capability_key=CAPABILITY_KEY,
        public_code_identity=CODE_IDENTITY,
        public_preflight_manifest_relative_path=(
            PUBLIC_PREFLIGHT_MANIFEST_RELATIVE_PATH),
        public_preflight_manifest_sha256=PUBLIC_PREFLIGHT_MANIFEST_SHA256,
        unique_formal_run_limit=1,
        formal_run_count_before=0,
        private_payload_reads_before=0,
        host_learning_writes_before=0,
        label_writes_before=0,
        clone_evaluation_writes_before=0,
        teacher_api_llm_calls_before=0,
        status=FAMILY_FREEZE_STATUS,
    )
    return ConflictSetFamilyFreeze(
        public_preflight, family_code_identity, transport, owner_metadata,
        guard, inventory_sha,
        hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    )


def parse_conflict_set_family_freeze_bytes(
        payload: bytes,
        ) -> ConflictSetFamilyFreeze:
    """严格回读一条 canonical family-freeze JSONL。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise ConflictSetFamilyFreezeError(
            "family freeze must be one JSONL record")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except (TypeError, ValueError) as error:
        raise ConflictSetFamilyFreezeError(
            "family freeze JSON is not canonical") from error
    if canonical_json_line(value) != payload:
        raise ConflictSetFamilyFreezeError(
            "family freeze JSON bytes are not canonical")
    return ConflictSetFamilyFreeze.from_dict(value)


def assert_conflict_set_family_freeze_matches_live_public_code(
        freeze: ConflictSetFamilyFreeze,
        repository_root: str | Path,
        ) -> None:
    """只读重算 public manifest 与当前 family 代码身份，不读取 owner payload。"""
    if not isinstance(freeze, ConflictSetFamilyFreeze):
        raise TypeError("family freeze type is invalid")
    repository = Path(repository_root).resolve()
    target = (
        repository
        / Path(*PUBLIC_PREFLIGHT_MANIFEST_RELATIVE_PATH.split("/"))
    ).resolve()
    if (not target.is_file() or target.is_symlink()
            or not target.is_relative_to(repository)):
        raise ConflictSetFamilyFreezeError(
            "frozen public preflight manifest is missing")
    try:
        payload = target.read_bytes()
        public = parse_conflict_set_public_preflight_bytes(payload)
    except (OSError, ConflictSetPublicPreflightError) as error:
        raise ConflictSetFamilyFreezeError(
            "frozen public preflight manifest cannot be revalidated") from error
    if (hashlib.sha256(payload).hexdigest()
            != PUBLIC_PREFLIGHT_MANIFEST_SHA256
            or public != freeze.public_preflight):
        raise ConflictSetFamilyFreezeError(
            "live public preflight V1 drifted")
    if build_conflict_set_family_code_identity(
            repository) != freeze.family_code_identity:
        raise ConflictSetFamilyFreezeError(
            "live family code identity drifted")


__all__ = [
    "FAMILY_CODE_ROOT_MODULES",
    "FAMILY_FREEZE_ARTIFACT_KIND",
    "FAMILY_FREEZE_FORMAT_VERSION",
    "FAMILY_FREEZE_STATUS",
    "PUBLIC_PREFLIGHT_MANIFEST_RELATIVE_PATH",
    "PUBLIC_PREFLIGHT_MANIFEST_SHA256",
    "ConflictSetFamilyCodeFile",
    "ConflictSetFamilyCodeIdentity",
    "ConflictSetFamilyFreeze",
    "ConflictSetFamilyFreezeError",
    "assert_conflict_set_family_freeze_matches_live_public_code",
    "build_conflict_set_family_code_identity",
    "build_conflict_set_family_freeze",
    "parse_conflict_set_family_freeze_bytes",
]
