"""W08 外部私密包 V1 的 metadata、owner firewall 与 transport 合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    EvaluatorLabelRecord,
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    W08PrivateEvaluationError,
    evidence_commitment,
    strict_sha256,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_INFERENCE_PAYLOAD_KINDS,
)


W08_EXTERNAL_PRIVATE_PACKAGE_V1 = "W08_EXTERNAL_PRIVATE_PACKAGE_V1"
W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME = "external_private_package_v1.json"
W08_EXTERNAL_PACKAGE_MANIFEST_KIND = "PH2_W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST"
W08_EXTERNAL_PACKAGE_BINDING_KINDS = ("observation", "evaluator")


class W08ExternalPrivatePackageError(W08PrivateEvaluationError):
    """外部私密包 manifest、路径或 owner 隔离失败。"""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise W08ExternalPrivatePackageError(f"W08 external {label} 非法")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    path = PurePosixPath(_text(value, label=label))
    text = path.as_posix()
    if path.is_absolute() or ".." in path.parts or "\\" in text or text != value:
        raise W08ExternalPrivatePackageError(f"W08 external {label} 必须是安全 POSIX 相对路径")
    return text


def _is_reparse(path: Path) -> bool:
    try:
        stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise W08ExternalPrivatePackageError("W08 external 路径无法 stat") from error
    # Windows FILE_ATTRIBUTE_REPARSE_POINT，同时覆盖 junction。
    return bool(getattr(stat, "st_file_attributes", 0) & 0x400)


def _validated_root(value: str | Path, *, create: bool = False) -> Path:
    raw = Path(value)
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    if not raw.is_dir() or raw.is_symlink() or _is_reparse(raw):
        raise W08ExternalPrivatePackageError(
            "W08 external package root 非普通目录或为 link/reparse"
        )
    resolved = raw.resolve()
    if resolved.is_symlink() or _is_reparse(resolved):
        raise W08ExternalPrivatePackageError(
            "W08 external package root 非普通目录或为 link/reparse"
        )
    return resolved


def _resolve_external_path(root: Path, relative_path: str, *, label: str) -> Path:
    root = root.resolve()
    if root.is_symlink() or _is_reparse(root):
        raise W08ExternalPrivatePackageError("W08 external payload root 不得是 link/reparse")
    safe = _safe_relative_path(relative_path, label=label)
    target = (root / Path(*PurePosixPath(safe).parts)).resolve()
    if not target.is_relative_to(root):
        raise W08ExternalPrivatePackageError("W08 external payload path 越界")
    current = root
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.is_symlink() or (
            current.exists() and _is_reparse(current)
        ):
            raise W08ExternalPrivatePackageError("W08 external payload path 含 link/reparse")
    return target


@dataclass(frozen=True, order=True)
class W08ExternalPrivateBinding:
    """一个 logical pack 的单 owner 文件身份，只含 metadata。"""

    pack_key: str
    identity: ArtifactFileIdentity
    payload_kind_inventory: tuple[str, ...]
    cluster_commitment: str

    @property
    def relative_path(self) -> str:
        return self.identity.relative_path

    @property
    def access_phase(self) -> str:
        return "EXTERNAL_PRIVATE_PAYLOAD"

    def __post_init__(self) -> None:
        _text(self.pack_key, label="pack key")
        if not isinstance(self.identity, ArtifactFileIdentity):
            raise W08ExternalPrivatePackageError("W08 external binding identity 类型非法")
        if self.identity.owner_kind not in W08_EXTERNAL_PACKAGE_BINDING_KINDS:
            raise W08ExternalPrivatePackageError("W08 external binding owner 非法")
        if self.identity.owner_kind == "observation" and (
            self.identity.record_kind != "observation" or self.identity.split != "held_out"
        ):
            raise W08ExternalPrivatePackageError("W08 observation owner/split/kind 漂移")
        if self.identity.owner_kind == "evaluator" and (
            self.identity.record_kind != "evaluator_label" or self.identity.split != "held_out"
        ):
            raise W08ExternalPrivatePackageError("W08 label owner/split/kind 漂移")
        if self.identity.owner_kind == "observation":
            expected = tuple(sorted(set(self.payload_kind_inventory)))
            if not expected or any(item not in W08_INFERENCE_PAYLOAD_KINDS for item in expected):
                raise W08ExternalPrivatePackageError("W08 external payload kind inventory 非法")
            if expected != self.payload_kind_inventory:
                raise W08ExternalPrivatePackageError("W08 external payload kind inventory 非 canonical")
        elif self.payload_kind_inventory:
            raise W08ExternalPrivatePackageError("W08 evaluator binding 不得携带 payload kind inventory")
        strict_sha256(self.cluster_commitment, label="external cluster commitment")

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_commitment": self.cluster_commitment,
            "identity": self.identity.to_dict(),
            "pack_key": self.pack_key,
            "payload_kind_inventory": list(self.payload_kind_inventory),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "W08ExternalPrivateBinding":
        try:
            identity = ArtifactFileIdentity.from_dict(value["identity"])  # type: ignore[arg-type]
            return cls(
                str(value["pack_key"]),
                identity,
                tuple(str(item) for item in value["payload_kind_inventory"]),  # type: ignore[union-attr]
                str(value["cluster_commitment"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise W08ExternalPrivatePackageError("W08 external binding metadata 损坏") from error


def _binding_commitment(binding: W08ExternalPrivateBinding) -> dict[str, object]:
    return {
        "cluster_commitment": binding.cluster_commitment,
        "identity": binding.identity.to_dict(),
        "pack_key": binding.pack_key,
        "payload_kind_inventory": list(binding.payload_kind_inventory),
    }


def _derived_commitments(
    bindings: tuple[W08ExternalPrivateBinding, ...],
) -> tuple[str, str, str, str]:
    observations = tuple(
        item for item in bindings if item.identity.owner_kind == "observation"
    )
    labels = tuple(
        item for item in bindings if item.identity.owner_kind == "evaluator"
    )
    case_commitment = evidence_commitment([[
        item.pack_key,
        item.identity.content_identity_dict(),
        list(item.payload_kind_inventory),
        item.cluster_commitment,
    ] for item in observations])
    label_commitment = evidence_commitment([[
        item.pack_key,
        item.identity.content_identity_dict(),
        item.cluster_commitment,
    ] for item in labels])
    cluster_commitment = evidence_commitment([
        [item.pack_key, item.identity.owner_kind, item.cluster_commitment]
        for item in bindings
    ])
    payload_commitment = evidence_commitment([[
        item.pack_key,
        item.identity.owner_kind,
        item.identity.content_sha256,
        item.identity.transport_sha256,
    ] for item in bindings])
    return (
        case_commitment,
        label_commitment,
        cluster_commitment,
        payload_commitment,
    )


@dataclass(frozen=True)
class W08ExternalPrivatePackageManifest:
    """冻结 logical pack、双 owner 文件身份及全部 commitment。"""

    package_version: str
    bindings: tuple[W08ExternalPrivateBinding, ...]
    payload_kind_inventory: tuple[str, ...]
    cluster_commitment: str
    case_commitment: str
    label_commitment: str
    payload_commitment: str
    package_commitment: str

    def __post_init__(self) -> None:
        if self.package_version != W08_EXTERNAL_PRIVATE_PACKAGE_V1:
            raise W08ExternalPrivatePackageError("W08 external package version 漂移")
        if not self.bindings:
            raise W08ExternalPrivatePackageError("W08 external package bindings 为空")
        ordered = tuple(sorted(self.bindings, key=lambda item: (item.pack_key, item.identity.owner_kind)))
        if ordered != self.bindings:
            raise W08ExternalPrivatePackageError("W08 external bindings 非 canonical")
        keys = [(item.pack_key, item.identity.owner_kind) for item in self.bindings]
        paths = [item.identity.relative_path for item in self.bindings]
        if len(keys) != len(set(keys)) or len(paths) != len(set(paths)):
            raise W08ExternalPrivatePackageError("W08 external binding 重复")
        packs = sorted({item.pack_key for item in self.bindings})
        for pack in packs:
            pack_bindings = tuple(
                item for item in self.bindings if item.pack_key == pack
            )
            if len(pack_bindings) != 2:
                raise W08ExternalPrivatePackageError(
                    "W08 external 双 owner inventory 不闭合"
                )
            observation = next(
                (item for item in pack_bindings if item.identity.owner_kind == "observation"),
                None,
            )
            label = next(
                (item for item in pack_bindings if item.identity.owner_kind == "evaluator"),
                None,
            )
            if (
                observation is None
                or label is None
                or observation.identity.record_count <= 0
                or observation.identity.record_count != label.identity.record_count
                or observation.cluster_commitment != label.cluster_commitment
            ):
                raise W08ExternalPrivatePackageError(
                    "W08 external 双 owner count/cluster 不闭合"
                )
        inventory = tuple(sorted(set(self.payload_kind_inventory)))
        if inventory != self.payload_kind_inventory or inventory != tuple(sorted(W08_INFERENCE_PAYLOAD_KINDS)):
            raise W08ExternalPrivatePackageError("W08 external 五 payload kind inventory 不闭合")
        strict_sha256(self.cluster_commitment, label="external cluster commitment")
        strict_sha256(self.case_commitment, label="external case commitment")
        strict_sha256(self.label_commitment, label="external label commitment")
        strict_sha256(self.payload_commitment, label="external payload commitment")
        strict_sha256(self.package_commitment, label="external package commitment")
        derived = _derived_commitments(self.bindings)
        if derived != (
            self.case_commitment,
            self.label_commitment,
            self.cluster_commitment,
            self.payload_commitment,
        ):
            raise W08ExternalPrivatePackageError(
                "W08 external derived commitment 漂移"
            )
        if tuple(sorted({kind for item in self.bindings for kind in item.payload_kind_inventory})) != inventory:
            raise W08ExternalPrivatePackageError("W08 external payload kind binding inventory 不闭合")
        expected = _package_commitment_value(self)
        if expected != self.package_commitment:
            raise W08ExternalPrivatePackageError("W08 external package commitment 漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": W08_EXTERNAL_PACKAGE_MANIFEST_KIND,
            "bindings": [item.to_dict() for item in self.bindings],
            "case_commitment": self.case_commitment,
            "cluster_commitment": self.cluster_commitment,
            "format_version": 1,
            "label_commitment": self.label_commitment,
            "package_commitment": self.package_commitment,
            "package_version": self.package_version,
            "payload_commitment": self.payload_commitment,
            "payload_kind_inventory": list(self.payload_kind_inventory),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "W08ExternalPrivatePackageManifest":
        if value.get("artifact_kind") != W08_EXTERNAL_PACKAGE_MANIFEST_KIND or value.get("format_version") != 1:
            raise W08ExternalPrivatePackageError("W08 external package manifest kind/version 漂移")
        try:
            return cls(
                str(value["package_version"]),
                tuple(W08ExternalPrivateBinding.from_dict(item) for item in value["bindings"]),  # type: ignore[arg-type]
                tuple(str(item) for item in value["payload_kind_inventory"]),  # type: ignore[union-attr]
                str(value["cluster_commitment"]),
                str(value["case_commitment"]),
                str(value["label_commitment"]),
                str(value["payload_commitment"]),
                str(value["package_commitment"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise W08ExternalPrivatePackageError("W08 external package manifest 损坏") from error


def _package_commitment_value(manifest: W08ExternalPrivatePackageManifest) -> str:
    return evidence_commitment({
        "bindings": [_binding_commitment(item) for item in manifest.bindings],
        "case_commitment": manifest.case_commitment,
        "cluster_commitment": manifest.cluster_commitment,
        "label_commitment": manifest.label_commitment,
        "package_version": manifest.package_version,
        "payload_commitment": manifest.payload_commitment,
        "payload_kind_inventory": list(manifest.payload_kind_inventory),
    })


def build_w08_external_private_manifest(
    bindings: Iterable[W08ExternalPrivateBinding],
    *,
    payload_kind_inventory: tuple[str, ...] | None = None,
) -> W08ExternalPrivatePackageManifest:
    """从 evaluator owner 提供的文件 metadata 生成 manifest，不读取 payload。"""
    supplied = tuple(bindings)
    if any(not isinstance(item, W08ExternalPrivateBinding) for item in supplied):
        raise W08ExternalPrivatePackageError("W08 external binding 类型非法")
    binding_tuple = tuple(sorted(
        supplied, key=lambda item: (item.pack_key, item.identity.owner_kind)
    ))
    observations = tuple(item for item in binding_tuple if item.identity.owner_kind == "observation")
    labels = tuple(item for item in binding_tuple if item.identity.owner_kind == "evaluator")
    if {item.pack_key for item in observations} != {item.pack_key for item in labels}:
        raise W08ExternalPrivatePackageError("W08 external observation/label pack 不闭合")
    inventory = payload_kind_inventory or tuple(sorted({kind for item in observations for kind in item.payload_kind_inventory}))
    inventory = tuple(sorted(set(inventory)))
    (
        case_commitment,
        label_commitment,
        cluster_commitment,
        payload_commitment,
    ) = _derived_commitments(binding_tuple)
    package_commitment = evidence_commitment({
        "bindings": [_binding_commitment(item) for item in binding_tuple],
        "case_commitment": case_commitment,
        "cluster_commitment": cluster_commitment,
        "label_commitment": label_commitment,
        "package_version": W08_EXTERNAL_PRIVATE_PACKAGE_V1,
        "payload_commitment": payload_commitment,
        "payload_kind_inventory": list(inventory),
    })
    return W08ExternalPrivatePackageManifest(
        W08_EXTERNAL_PRIVATE_PACKAGE_V1,
        binding_tuple,
        inventory,
        cluster_commitment,
        case_commitment,
        label_commitment,
        payload_commitment,
        package_commitment,
    )


def write_w08_external_private_manifest(
    root: str | Path,
    manifest: W08ExternalPrivatePackageManifest,
) -> tuple[Path, str]:
    """独占写 manifest；payload 文件必须在 evaluator owner 侧先行完成。"""
    if not isinstance(manifest, W08ExternalPrivatePackageManifest):
        raise W08ExternalPrivatePackageError("W08 external manifest 类型非法")
    root_path = _validated_root(root, create=True)
    target = _resolve_external_path(root_path, W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME, label="manifest path")
    try:
        with target.open("xb") as handle:
            payload = manifest.canonical_bytes()
            handle.write(payload)
    except FileExistsError as error:
        raise W08ExternalPrivatePackageError("W08 external manifest 不可覆盖") from error
    return target, _sha256(payload)


def read_w08_external_private_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> W08ExternalPrivatePackageManifest:
    """只读取并核验 manifest metadata；不读取 Observation/label payload。"""
    root_path = _validated_root(root)
    raw_path = Path(manifest_path)
    if raw_path.is_absolute():
        if raw_path.name != W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME:
            raise W08ExternalPrivatePackageError("W08 external manifest path/name 漂移")
        if raw_path.parent.resolve() != root_path:
            raise W08ExternalPrivatePackageError("W08 external manifest path 越界")
        relative = raw_path.name
    else:
        relative = raw_path.as_posix()
    if relative != W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME:
        raise W08ExternalPrivatePackageError("W08 external manifest path/name 漂移")
    target = _resolve_external_path(root_path, relative, label="manifest path")
    try:
        payload = target.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W08ExternalPrivatePackageError("W08 external manifest 无法读取") from error
    if expected_sha256 is not None and _sha256(payload) != strict_sha256(expected_sha256, label="external manifest"):
        raise W08ExternalPrivatePackageError("W08 external manifest SHA 漂移")
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W08ExternalPrivatePackageError("W08 external manifest 非 canonical object")
    manifest = W08ExternalPrivatePackageManifest.from_dict(value)
    if manifest.canonical_bytes() != payload:
        raise W08ExternalPrivatePackageError(
            "W08 external manifest 字段 inventory 漂移"
        )
    return manifest


def validate_w08_external_private_package_metadata(
    root: str | Path,
    manifest: W08ExternalPrivatePackageManifest,
) -> None:
    """冻结前只核验路径、owner 和 manifest metadata，不读取 payload bytes。"""
    if not isinstance(manifest, W08ExternalPrivatePackageManifest):
        raise W08ExternalPrivatePackageError("W08 external manifest 类型非法")
    root_path = _validated_root(root)
    manifest_path = _resolve_external_path(root_path, W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME, label="manifest path")
    if not manifest_path.is_file():
        raise W08ExternalPrivatePackageError("W08 external manifest 缺失")
    for binding in manifest.bindings:
        target = _resolve_external_path(root_path, binding.identity.relative_path, label="payload path")
        if not target.is_file():
            raise W08ExternalPrivatePackageError("W08 external payload 文件缺失")
    expected_files = {
        W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME,
        *(item.identity.relative_path for item in manifest.bindings),
    }
    actual_files: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        root_path, followlinks=False
    ):
        current = Path(directory)
        for name in directory_names:
            child = current / name
            if child.is_symlink() or _is_reparse(child):
                raise W08ExternalPrivatePackageError(
                    "W08 external package 含未授权 link/reparse"
                )
        for name in file_names:
            child = current / name
            if child.is_symlink() or _is_reparse(child):
                raise W08ExternalPrivatePackageError(
                    "W08 external package 含未授权 link/reparse"
                )
            actual_files.add(child.relative_to(root_path).as_posix())
    if actual_files != expected_files:
        raise W08ExternalPrivatePackageError(
            "W08 external package 文件 inventory 漂移"
        )


def read_w08_external_private_binding(
    root: str | Path,
    binding: W08ExternalPrivateBinding,
) -> tuple[ObservationRecord, ...] | tuple[EvaluatorLabelRecord, ...]:
    """guard 后读取单个 owner artifact，并由 identity 复核 transport/content/hash/count。"""
    if not isinstance(binding, W08ExternalPrivateBinding):
        raise W08ExternalPrivatePackageError("W08 external binding 类型非法")
    root_path = _validated_root(root)
    target = _resolve_external_path(root_path, binding.identity.relative_path, label="payload path")
    if not target.is_file():
        raise W08ExternalPrivatePackageError("W08 external payload 文件缺失")
    try:
        records = read_record_artifact(root_path, binding.identity)
    except (DatasetArtifactIOError, OSError) as error:
        raise W08ExternalPrivatePackageError("W08 external payload transport/hash/count 失败") from error
    if binding.identity.owner_kind == "observation":
        if any(not isinstance(item, ObservationRecord) for item in records):
            raise W08ExternalPrivatePackageError("W08 external Observation record kind 漂移")
        observed = tuple(sorted({item.payload_kind for item in records}))
        if observed != binding.payload_kind_inventory:
            raise W08ExternalPrivatePackageError("W08 external payload kind inventory 漂移")
        return records  # type: ignore[return-value]
    if any(not isinstance(item, EvaluatorLabelRecord) for item in records):
        raise W08ExternalPrivatePackageError("W08 external label record kind 漂移")
    return records  # type: ignore[return-value]


__all__ = [
    "W08_EXTERNAL_PACKAGE_MANIFEST_KIND",
    "W08_EXTERNAL_PACKAGE_BINDING_KINDS",
    "W08_EXTERNAL_PRIVATE_PACKAGE_MANIFEST_NAME",
    "W08_EXTERNAL_PRIVATE_PACKAGE_V1",
    "W08ExternalPrivateBinding",
    "W08ExternalPrivatePackageError",
    "W08ExternalPrivatePackageManifest",
    "build_w08_external_private_manifest",
    "read_w08_external_private_binding",
    "read_w08_external_private_manifest",
    "validate_w08_external_private_package_metadata",
    "write_w08_external_private_manifest",
]
