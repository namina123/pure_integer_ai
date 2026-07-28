"""PH2 固定 Git commit 第三方来源的逐文件 blob/local hash snapshot 合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import sha256_path


class GitSnapshotError(RuntimeError):
    """Git commit、tracked file、许可或 snapshot manifest 发生漂移。"""


def _text(value: Any, *, where: str) -> str:
    """要求 Git snapshot 文本非空且无首尾空白。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise GitSnapshotError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _positive(value: Any, *, where: str) -> int:
    """要求版本为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise GitSnapshotError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    """要求文件大小/计数为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise GitSnapshotError(f"{where} 必须是非负严格整数")
    return value


def _digest(value: Any, length: int, *, where: str) -> str:
    """校验 SHA-1/SHA-256 小写十六进制。"""
    text = _text(value, where=where).lower()
    if (len(text) != length
            or any(character not in "0123456789abcdef" for character in text)):
        raise GitSnapshotError(f"{where} 摘要非法")
    return text


def _relative(value: Any, *, where: str) -> str:
    """要求 tracked file 使用安全 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise GitSnapshotError(f"{where} 必须是安全 POSIX 相对路径")
    return text


@dataclass(frozen=True, order=True)
class GitTrackedFileManifest:
    """一个固定 commit tracked file 的 Git blob 和本地字节双身份。"""

    relative_path: str
    git_blob_sha1: str
    local_sha256: str
    size_bytes: int
    file_kind: str
    split: str
    record_count: int
    anomaly_count: int
    parser_event_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "relative_path",
            _relative(self.relative_path, where="GitTrackedFileManifest.path"))
        object.__setattr__(
            self, "git_blob_sha1",
            _digest(self.git_blob_sha1, 40, where="GitTrackedFileManifest.blob"))
        object.__setattr__(
            self, "local_sha256",
            _digest(self.local_sha256, 64, where="GitTrackedFileManifest.sha256"))
        _nonnegative(self.size_bytes, where="GitTrackedFileManifest.size_bytes")
        _text(self.file_kind, where="GitTrackedFileManifest.file_kind")
        if self.split and self.split not in {"train", "dev", "held_out"}:
            raise GitSnapshotError("GitTrackedFileManifest.split 非法")
        _nonnegative(self.record_count, where="GitTrackedFileManifest.record_count")
        _nonnegative(self.anomaly_count, where="GitTrackedFileManifest.anomaly_count")
        object.__setattr__(
            self,
            "parser_event_sha256",
            _digest(
                self.parser_event_sha256,
                64,
                where="GitTrackedFileManifest.parser_event_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """导出 tracked file 规范身份。"""
        return {
            "anomaly_count": self.anomaly_count,
            "file_kind": self.file_kind,
            "git_blob_sha1": self.git_blob_sha1,
            "local_sha256": self.local_sha256,
            "parser_event_sha256": self.parser_event_sha256,
            "record_count": self.record_count,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GitTrackedFileManifest":
        """从规范 object 恢复 tracked file。"""
        return cls(
            str(value["relative_path"]),
            str(value["git_blob_sha1"]),
            str(value["local_sha256"]),
            value["size_bytes"],
            str(value["file_kind"]),
            str(value["split"]),
            value["record_count"],
            value["anomaly_count"],
            str(value["parser_event_sha256"]),
        )


@dataclass(frozen=True)
class GitSourceSnapshotManifest:
    """冻结官方 repository/tag/commit、许可和所有参与文件。"""

    format_version: int
    source_key: str
    repository_url: str
    tag: str
    commit_sha1: str
    license_id: str
    license_evidence_path: str
    redistribution_policy: str
    adapter_version: int
    parser_version: int
    files: tuple[GitTrackedFileManifest, ...]
    parser_report: CanonicalJsonObject

    def __post_init__(self) -> None:
        for name, value in (
                ("format_version", self.format_version),
                ("adapter_version", self.adapter_version),
                ("parser_version", self.parser_version)):
            _positive(value, where=f"GitSourceSnapshotManifest.{name}")
        _text(self.source_key, where="GitSourceSnapshotManifest.source_key")
        if not self.repository_url.startswith("https://github.com/"):
            raise GitSnapshotError("Git snapshot repository_url 必须是 GitHub HTTPS")
        _text(self.tag, where="GitSourceSnapshotManifest.tag")
        object.__setattr__(
            self, "commit_sha1",
            _digest(self.commit_sha1, 40, where="GitSourceSnapshotManifest.commit"))
        _text(self.license_id, where="GitSourceSnapshotManifest.license_id")
        object.__setattr__(
            self,
            "license_evidence_path",
            _relative(
                self.license_evidence_path,
                where="GitSourceSnapshotManifest.license_evidence_path",
            ),
        )
        if self.redistribution_policy != "PUBLIC":
            raise GitSnapshotError("当前 Git snapshot 只接受许可明确的 PUBLIC 来源")
        if not isinstance(self.files, tuple) or not self.files:
            raise GitSnapshotError("Git snapshot files 不能为空")
        if any(not isinstance(item, GitTrackedFileManifest) for item in self.files):
            raise GitSnapshotError("Git snapshot files 类型错误")
        object.__setattr__(self, "files", tuple(sorted(self.files)))
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise GitSnapshotError("Git snapshot tracked path 重复")
        if self.license_evidence_path not in paths:
            raise GitSnapshotError("Git snapshot 缺少 license evidence file")
        if not isinstance(self.parser_report, CanonicalJsonObject):
            raise GitSnapshotError("Git snapshot parser_report 类型错误")

    def to_dict(self) -> dict[str, Any]:
        """导出 Git snapshot 的规范 JSON object。"""
        return {
            "adapter_version": self.adapter_version,
            "commit_sha1": self.commit_sha1,
            "files": [item.to_dict() for item in self.files],
            "format_version": self.format_version,
            "license_evidence_path": self.license_evidence_path,
            "license_id": self.license_id,
            "parser_report": self.parser_report.to_value(),
            "parser_version": self.parser_version,
            "redistribution_policy": self.redistribution_policy,
            "repository_url": self.repository_url,
            "source_key": self.source_key,
            "tag": self.tag,
        }

    def canonical_bytes(self) -> bytes:
        """返回一个换行结尾的规范 manifest 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回完整 Git snapshot manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GitSourceSnapshotManifest":
        """从规范 object 恢复 Git snapshot。"""
        return cls(
            value["format_version"],
            str(value["source_key"]),
            str(value["repository_url"]),
            str(value["tag"]),
            str(value["commit_sha1"]),
            str(value["license_id"]),
            str(value["license_evidence_path"]),
            str(value["redistribution_policy"]),
            value["adapter_version"],
            value["parser_version"],
            tuple(GitTrackedFileManifest.from_dict(item)
                  for item in value["files"]),
            CanonicalJsonObject.from_value(value["parser_report"]),
        )


def read_git_snapshot_manifest(path: str | Path) -> GitSourceSnapshotManifest:
    """严格读取规范 Git snapshot manifest。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise GitSnapshotError("Git snapshot manifest 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise GitSnapshotError("Git snapshot manifest 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = GitSourceSnapshotManifest.from_dict(value)
    except Exception as error:
        raise GitSnapshotError("Git snapshot manifest 合同或编码损坏") from error
    if manifest.canonical_bytes() != payload:
        raise GitSnapshotError("Git snapshot manifest 规范字节漂移")
    return manifest


def git_blob_sha1_path(path: str | Path) -> str:
    """按 Git blob object 格式计算文件 SHA-1，不依赖工作树 Git 配置。"""
    source = Path(path)
    try:
        size = source.stat().st_size
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {size}\0".encode("ascii"))
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise GitSnapshotError("Git blob 文件无法读取") from error
    return digest.hexdigest()


def write_git_snapshot_manifest(
        manifest: GitSourceSnapshotManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等核对 Git snapshot manifest。"""
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise GitSnapshotError("Git snapshot manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise GitSnapshotError("Git snapshot manifest 无法发布") from error
    return target


def verify_git_snapshot(
        manifest: GitSourceSnapshotManifest,
        repository_root: str | Path,
        *,
        resolved_commit_sha1: str,
        resolved_git_blob_sha1s: Mapping[str, str] | None = None,
        ) -> None:
    """核对 commit、本地字节，并可核对调用方只读解析的 Git blobs。"""
    commit = _digest(
        resolved_commit_sha1, 40, where="verify_git_snapshot.commit")
    if commit != manifest.commit_sha1:
        raise GitSnapshotError("Git snapshot commit 漂移")
    if resolved_git_blob_sha1s is not None:
        expected_paths = {item.relative_path for item in manifest.files}
        if set(resolved_git_blob_sha1s) != expected_paths:
            raise GitSnapshotError("Git snapshot resolved blob path 集合漂移")
        for item in manifest.files:
            resolved_blob = _digest(
                resolved_git_blob_sha1s[item.relative_path],
                40,
                where=f"verify_git_snapshot.blob.{item.relative_path}",
            )
            if resolved_blob != item.git_blob_sha1:
                raise GitSnapshotError("Git snapshot tracked file blob SHA-1 漂移")
    root = Path(repository_root).resolve()
    for item in manifest.files:
        path = (root / Path(*PurePosixPath(item.relative_path).parts)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise GitSnapshotError("Git snapshot tracked file 缺失或逃逸")
        if path.stat().st_size != item.size_bytes:
            raise GitSnapshotError("Git snapshot tracked file size 漂移")
        if sha256_path(path) != item.local_sha256:
            raise GitSnapshotError("Git snapshot tracked file SHA-256 漂移")


__all__ = [
    "GitSnapshotError",
    "GitSourceSnapshotManifest",
    "GitTrackedFileManifest",
    "git_blob_sha1_path",
    "read_git_snapshot_manifest",
    "verify_git_snapshot",
    "write_git_snapshot_manifest",
]
