"""原始数据只读 manifest 与完整性校验接口。

本模块位于 experiments 边界，允许文件路径、编码和许可等文本元数据。任何进入核心图的
绑定仍必须是严格整数元组。原始文件只读，manifest 只能写到 raw root 之外的新版本目录。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class ManifestIntegrityError(RuntimeError):
    """manifest、原始文件或只读边界不一致。"""


def _strict_nonnegative(value: int, *, where: str) -> int:
    """校验 manifest 整数字段为严格非负整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为严格非负整数")
    return value


def _integer_key(values: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验进入核心的 manifest 绑定为非空严格整数元组。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{where} 必须是非空严格整数元组")
    assert_int(*values, _where=where)
    if any(type(value) is not int for value in values):
        raise ValueError(f"{where} 必须使用严格整数")
    return values


def _relative_path(value: str) -> str:
    """规范化可移植相对路径并拒绝逃逸 raw root。"""
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("manifest 文件路径必须位于 raw root 内")
    return path.as_posix()


@dataclass(frozen=True, order=True)
class ManifestBinding:
    """外部元数据名称到核心整数身份键的显式绑定。"""

    name: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ManifestBinding.name 不能为空")
        _integer_key(self.values, where=f"ManifestBinding[{self.name}]")
        if any(value < 0 for value in self.values):
            raise ValueError("ManifestBinding 核心整数键不得为负")

    def to_dict(self) -> dict[str, Any]:
        """转换为确定字段顺序无关的 JSON 对象。"""
        return {"name": self.name, "values": list(self.values)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManifestBinding":
        """从 JSON 对象恢复整数绑定。"""
        return cls(str(value["name"]), tuple(value["values"]))


@dataclass(frozen=True, order=True)
class RawFileManifest:
    """一个原始文件的内容指纹、格式和扫描统计。"""

    relative_path: str
    sha256: str
    size_bytes: int
    encoding: str
    file_format: str
    parser_kind: str
    property_namespace: str = ""
    property_name: str = ""
    record_count: int = 0
    anomaly_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path))
        digest = self.sha256.lower()
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise ValueError("RawFileManifest.sha256 必须为 64 位十六进制")
        object.__setattr__(self, "sha256", digest)
        _strict_nonnegative(self.size_bytes, where="RawFileManifest.size_bytes")
        _strict_nonnegative(
            self.record_count, where="RawFileManifest.record_count")
        _strict_nonnegative(
            self.anomaly_count, where="RawFileManifest.anomaly_count")
        if not self.file_format or not self.parser_kind:
            raise ValueError("RawFileManifest 必须声明格式和 parser_kind")

    def to_dict(self) -> dict[str, Any]:
        """转换为可规范序列化的 JSON 对象。"""
        return {
            "anomaly_count": self.anomaly_count,
            "encoding": self.encoding,
            "file_format": self.file_format,
            "parser_kind": self.parser_kind,
            "property_name": self.property_name,
            "property_namespace": self.property_namespace,
            "record_count": self.record_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RawFileManifest":
        """从 JSON 对象恢复文件 manifest。"""
        return cls(
            str(value["relative_path"]),
            str(value["sha256"]),
            value["size_bytes"],
            str(value["encoding"]),
            str(value["file_format"]),
            str(value["parser_kind"]),
            str(value.get("property_namespace", "")),
            str(value.get("property_name", "")),
            value.get("record_count", 0),
            value.get("anomaly_count", 0),
        )


@dataclass(frozen=True)
class RawDatasetManifest:
    """一批不可变原始文件及其 adapter/许可/核心绑定版本。"""

    dataset_name: str
    dataset_version: str
    adapter_version: int
    parser_version: int
    license_id: str
    files: tuple[RawFileManifest, ...]
    bindings: tuple[ManifestBinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset_name or not self.dataset_version or not self.license_id:
            raise ValueError("数据集名称、版本和许可标识不能为空")
        _strict_nonnegative(
            self.adapter_version, where="RawDatasetManifest.adapter_version")
        _strict_nonnegative(
            self.parser_version, where="RawDatasetManifest.parser_version")
        if not self.files:
            raise ValueError("RawDatasetManifest.files 不能为空")
        object.__setattr__(self, "files", tuple(sorted(self.files)))
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings)))
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("RawDatasetManifest 文件路径重复")
        names = [item.name for item in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("RawDatasetManifest 绑定名称重复")

    def to_dict(self) -> dict[str, Any]:
        """转换为规范 JSON 所需的稳定对象。"""
        return {
            "adapter_version": self.adapter_version,
            "bindings": [item.to_dict() for item in sorted(self.bindings)],
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "files": [item.to_dict() for item in sorted(self.files)],
            "license_id": self.license_id,
            "parser_version": self.parser_version,
        }

    def canonical_bytes(self) -> bytes:
        """返回 bit-identical UTF-8 规范 JSON。"""
        text = json.dumps(
            self.to_dict(), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"))
        return (text + "\n").encode("utf-8")

    def sha256(self) -> str:
        """返回不依赖输出路径的 manifest 内容哈希。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def binding(self, name: str) -> tuple[int, ...]:
        """按名称读取唯一核心整数绑定，缺失时 fail closed。"""
        matches = [item.values for item in self.bindings if item.name == name]
        if len(matches) != 1:
            raise ManifestIntegrityError(f"manifest 绑定 {name!r} 不唯一或缺失")
        return matches[0]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RawDatasetManifest":
        """从已解析 JSON 恢复数据集 manifest。"""
        return cls(
            str(value["dataset_name"]),
            str(value["dataset_version"]),
            value["adapter_version"],
            value["parser_version"],
            str(value["license_id"]),
            tuple(RawFileManifest.from_dict(item) for item in value["files"]),
            tuple(ManifestBinding.from_dict(item) for item in value.get(
                "bindings", ())),
        )


def sha256_file(path: str | Path) -> str:
    """以固定块大小流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> RawDatasetManifest:
    """严格按 UTF-8 读取 manifest，并拒绝非对象根。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestIntegrityError("manifest 根必须是 JSON 对象")
    return RawDatasetManifest.from_dict(value)


def verify_manifest(manifest: RawDatasetManifest,
                    raw_root: str | Path) -> None:
    """逐文件核验存在性、大小和 SHA-256，任一不符立即失败。"""
    root = Path(raw_root).resolve()
    for item in manifest.files:
        path = (root / item.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ManifestIntegrityError(
                f"manifest 原始文件缺失或越界: {item.relative_path}")
        if path.stat().st_size != item.size_bytes:
            raise ManifestIntegrityError(
                f"manifest 文件大小变化: {item.relative_path}")
        if sha256_file(path) != item.sha256:
            raise ManifestIntegrityError(
                f"manifest 文件哈希变化: {item.relative_path}")


@contextmanager
def open_verified_text(raw_root: str | Path,
                       item: RawFileManifest) -> Iterator[TextIO]:
    """在前后两次内容核验之间以严格编码只读打开文本文件。"""
    root = Path(raw_root).resolve()
    path = (root / item.relative_path).resolve()
    if not path.is_relative_to(root):
        raise ManifestIntegrityError("原始文件路径逃逸 raw root")
    if not item.encoding:
        raise ManifestIntegrityError("二进制文件不能通过文本适配器读取")
    if not path.is_file() or path.stat().st_size != item.size_bytes:
        raise ManifestIntegrityError(f"原始文件缺失或大小变化: {item.relative_path}")
    if sha256_file(path) != item.sha256:
        raise ManifestIntegrityError(f"原始文件哈希变化: {item.relative_path}")
    try:
        with path.open("r", encoding=item.encoding, errors="strict", newline="") as handle:
            yield handle
    finally:
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ManifestIntegrityError(
                f"读取期间原始文件大小变化: {item.relative_path}")
        if sha256_file(path) != item.sha256:
            raise ManifestIntegrityError(
                f"读取期间原始文件哈希变化: {item.relative_path}")


@contextmanager
def open_verified_binary(raw_root: str | Path,
                         item: RawFileManifest) -> Iterator[BinaryIO]:
    """在前后完整性核验之间只读打开二进制流，供 byte span 适配器使用。"""
    root = Path(raw_root).resolve()
    path = (root / item.relative_path).resolve()
    if not path.is_relative_to(root):
        raise ManifestIntegrityError("原始文件路径逃逸 raw root")
    if not path.is_file() or path.stat().st_size != item.size_bytes:
        raise ManifestIntegrityError(f"原始文件缺失或大小变化: {item.relative_path}")
    if sha256_file(path) != item.sha256:
        raise ManifestIntegrityError(f"原始文件哈希变化: {item.relative_path}")
    try:
        with path.open("rb") as handle:
            yield handle
    finally:
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ManifestIntegrityError(
                f"读取期间原始文件大小变化: {item.relative_path}")
        if sha256_file(path) != item.sha256:
            raise ManifestIntegrityError(
                f"读取期间原始文件哈希变化: {item.relative_path}")


def write_manifest(manifest: RawDatasetManifest, output_path: str | Path, *,
                   raw_root: str | Path) -> Path:
    """在 raw root 外幂等写 manifest；既有不同内容必须换版本目录。"""
    root = Path(raw_root).resolve()
    path = Path(output_path).resolve()
    if path.is_relative_to(root):
        raise ManifestIntegrityError("manifest 输出不得位于原始数据目录")
    payload = manifest.canonical_bytes()
    if path.exists():
        if path.read_bytes() != payload:
            raise ManifestIntegrityError("既有 manifest 内容不同，必须使用新版本目录")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    return path


__all__ = [
    "ManifestBinding",
    "ManifestIntegrityError",
    "RawDatasetManifest",
    "RawFileManifest",
    "open_verified_binary",
    "open_verified_text",
    "read_manifest",
    "sha256_file",
    "verify_manifest",
    "write_manifest",
]
