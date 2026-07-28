"""PH2 第三方原始字节、HTTP 响应头和许可裁决的不可覆盖 snapshot 合同。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)


LICENSE_STATUSES = ("MATCH", "CONFLICT", "UNKNOWN")
SNAPSHOT_POLICIES = ("PUBLIC", "LOCAL_ONLY", "BLOCKED")


class RawSnapshotError(RuntimeError):
    """原始字节、HTTP metadata、许可或 snapshot manifest 不一致。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 snapshot 文本无首尾空白；可选允许空值。"""
    if not isinstance(value, str) or value.strip() != value:
        raise RawSnapshotError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise RawSnapshotError(f"{where} 不能为空")
    return value


def _positive(value: Any, *, where: str) -> int:
    """要求版本为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise RawSnapshotError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    """要求计数为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise RawSnapshotError(f"{where} 必须是非负严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求小写或可归一的小写 SHA-256。"""
    text = _text(value, where=where).lower()
    if (len(text) != 64
            or any(character not in "0123456789abcdef" for character in text)):
        raise RawSnapshotError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求 manifest 只保存可迁移 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise RawSnapshotError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def sha256_path(path: str | Path) -> str:
    """以固定块大小流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HttpCaptureMetadata:
    """最终 HTTP 响应中用于重建 snapshot 的稳定头字段。"""

    status_code: int
    response_url: str
    response_date_gmt: str
    last_modified_gmt: str
    etag: str
    content_type: str
    content_length: int
    response_headers_sha256: str
    captured_at_utc: str = ""

    def __post_init__(self) -> None:
        if self.status_code != 200:
            raise RawSnapshotError("snapshot HTTP status 必须为 200")
        if not self.response_url.startswith("https://"):
            raise RawSnapshotError("snapshot response_url 必须是 HTTPS")
        for name, value in (
                ("response_date_gmt", self.response_date_gmt),
                ("last_modified_gmt", self.last_modified_gmt),
                ("etag", self.etag),
                ("content_type", self.content_type)):
            _text(value, where=f"HttpCaptureMetadata.{name}")
        if not self.response_date_gmt.endswith(" GMT"):
            raise RawSnapshotError("snapshot Date 必须是 GMT")
        if not self.last_modified_gmt.endswith(" GMT"):
            raise RawSnapshotError("snapshot Last-Modified 必须是 GMT")
        _nonnegative(
            self.content_length, where="HttpCaptureMetadata.content_length")
        object.__setattr__(
            self,
            "response_headers_sha256",
            _sha256(
                self.response_headers_sha256,
                where="HttpCaptureMetadata.response_headers_sha256",
            ),
        )
        if self.captured_at_utc and not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
                r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z",
                self.captured_at_utc):
            raise RawSnapshotError(
                "snapshot captured_at_utc 必须是 UTC ISO-8601")

    def to_dict(self) -> dict[str, Any]:
        """导出 HTTP capture 的规范 JSON object。"""
        result = {
            "content_length": self.content_length,
            "content_type": self.content_type,
            "etag": self.etag,
            "last_modified_gmt": self.last_modified_gmt,
            "response_date_gmt": self.response_date_gmt,
            "response_headers_sha256": self.response_headers_sha256,
            "response_url": self.response_url,
            "status_code": self.status_code,
        }
        if self.captured_at_utc:
            result["captured_at_utc"] = self.captured_at_utc
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HttpCaptureMetadata":
        """从严格字段恢复 HTTP capture。"""
        return cls(
            value["status_code"],
            str(value["response_url"]),
            str(value["response_date_gmt"]),
            str(value["last_modified_gmt"]),
            str(value["etag"]),
            str(value["content_type"]),
            value["content_length"],
            str(value["response_headers_sha256"]),
            str(value.get("captured_at_utc", "")),
        )


def read_http_capture(
        headers_path: str | Path,
        *,
        response_url: str,
        captured_at_utc: str = "",
        ) -> HttpCaptureMetadata:
    """读取 curl dump-header 的最终 200 响应，拒绝缺字段或重复字段。"""
    path = Path(headers_path)
    try:
        payload = path.read_bytes()
        text = payload.decode("iso-8859-1")
    except (OSError, UnicodeError) as error:
        raise RawSnapshotError("HTTP capture 无法读取") from error
    blocks = [block for block in text.replace("\r\n", "\n").split("\n\n")
              if block.strip()]
    if not blocks:
        raise RawSnapshotError("HTTP capture 为空")
    lines = [line for line in blocks[-1].split("\n") if line]
    if not lines or not lines[0].startswith("HTTP/"):
        raise RawSnapshotError("HTTP capture 缺少 status line")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2 or not parts[1].isdigit():
        raise RawSnapshotError("HTTP capture status line 非法")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            raise RawSnapshotError("HTTP capture header 行非法")
        name, value = line.split(":", 1)
        normalized = name.strip().casefold()
        if normalized in headers:
            raise RawSnapshotError("HTTP capture 最终响应头重复")
        headers[normalized] = value.strip()
    required = ("date", "last-modified", "etag", "content-type", "content-length")
    if any(name not in headers for name in required):
        raise RawSnapshotError("HTTP capture 缺少必需头字段")
    try:
        content_length = int(headers["content-length"])
    except ValueError as error:
        raise RawSnapshotError("HTTP Content-Length 非整数") from error
    return HttpCaptureMetadata(
        int(parts[1]),
        response_url,
        headers["date"],
        headers["last-modified"],
        headers["etag"],
        headers["content-type"],
        content_length,
        sha256_path(path),
        captured_at_utc,
    )


@dataclass(frozen=True)
class ThirdPartyRawSnapshotManifest:
    """冻结第三方原始字节、获取证据、许可冲突和 parser 扫描摘要。"""

    format_version: int
    source_key: str
    snapshot_id: str
    official_url: str
    raw_relative_path: str
    raw_sha256: str
    raw_size_bytes: int
    compression: str
    encoding: str
    adapter_version: int
    parser_version: int
    http: HttpCaptureMetadata
    expected_license_id: str
    observed_license_id: str
    license_evidence_url: str
    license_status: str
    redistribution_policy: str
    release_eligible: int
    blocker_code: str
    attribution: str
    declared_record_count: int
    parsed_record_count: int
    anomaly_count: int
    parser_event_sha256: str
    parser_report: CanonicalJsonObject

    def __post_init__(self) -> None:
        for name, value in (
                ("format_version", self.format_version),
                ("adapter_version", self.adapter_version),
                ("parser_version", self.parser_version)):
            _positive(value, where=f"ThirdPartyRawSnapshotManifest.{name}")
        for name, value in (
                ("source_key", self.source_key),
                ("snapshot_id", self.snapshot_id),
                ("compression", self.compression),
                ("encoding", self.encoding),
                ("expected_license_id", self.expected_license_id),
                ("observed_license_id", self.observed_license_id),
                ("attribution", self.attribution)):
            _text(value, where=f"ThirdPartyRawSnapshotManifest.{name}")
        if not self.official_url.startswith("https://"):
            raise RawSnapshotError("snapshot official_url 必须是 HTTPS")
        object.__setattr__(
            self, "raw_relative_path",
            _relative_path(
                self.raw_relative_path, where="snapshot.raw_relative_path"))
        object.__setattr__(
            self, "raw_sha256",
            _sha256(self.raw_sha256, where="snapshot.raw_sha256"))
        object.__setattr__(
            self, "parser_event_sha256",
            _sha256(
                self.parser_event_sha256,
                where="snapshot.parser_event_sha256"))
        if not isinstance(self.parser_report, CanonicalJsonObject):
            raise RawSnapshotError("snapshot.parser_report 类型错误")
        _nonnegative(self.raw_size_bytes, where="snapshot.raw_size_bytes")
        if not isinstance(self.http, HttpCaptureMetadata):
            raise RawSnapshotError("snapshot.http 类型错误")
        if self.raw_size_bytes != self.http.content_length:
            raise RawSnapshotError("raw size 与 HTTP Content-Length 不一致")
        if not self.license_evidence_url.startswith("https://"):
            raise RawSnapshotError("license_evidence_url 必须是 HTTPS")
        if self.license_status not in LICENSE_STATUSES:
            raise RawSnapshotError("license_status 非法")
        if self.redistribution_policy not in SNAPSHOT_POLICIES:
            raise RawSnapshotError("snapshot redistribution_policy 非法")
        if self.release_eligible not in {0, 1}:
            raise RawSnapshotError("release_eligible 必须为 0/1")
        _text(
            self.blocker_code,
            where="snapshot.blocker_code",
            allow_empty=True,
        )
        for name, value in (
                ("declared_record_count", self.declared_record_count),
                ("parsed_record_count", self.parsed_record_count),
                ("anomaly_count", self.anomaly_count)):
            _nonnegative(value, where=f"snapshot.{name}")
        if self.license_status == "MATCH":
            if self.expected_license_id != self.observed_license_id:
                raise RawSnapshotError("MATCH 许可 id 不一致")
            if not self.release_eligible or self.redistribution_policy != "PUBLIC":
                raise RawSnapshotError("MATCH snapshot 必须可公开")
            if self.blocker_code:
                raise RawSnapshotError("MATCH snapshot 不得有 blocker")
        else:
            if self.release_eligible or self.redistribution_policy != "BLOCKED":
                raise RawSnapshotError("冲突/未知许可 snapshot 必须 BLOCKED")
            if not self.blocker_code:
                raise RawSnapshotError("冲突/未知许可 snapshot 必须有 blocker")

    def to_dict(self) -> dict[str, Any]:
        """导出 snapshot manifest 的规范 JSON object。"""
        return {
            "adapter_version": self.adapter_version,
            "anomaly_count": self.anomaly_count,
            "attribution": self.attribution,
            "blocker_code": self.blocker_code,
            "compression": self.compression,
            "declared_record_count": self.declared_record_count,
            "encoding": self.encoding,
            "expected_license_id": self.expected_license_id,
            "format_version": self.format_version,
            "http": self.http.to_dict(),
            "license_evidence_url": self.license_evidence_url,
            "license_status": self.license_status,
            "observed_license_id": self.observed_license_id,
            "official_url": self.official_url,
            "parsed_record_count": self.parsed_record_count,
            "parser_event_sha256": self.parser_event_sha256,
            "parser_report": self.parser_report.to_value(),
            "parser_version": self.parser_version,
            "raw_relative_path": self.raw_relative_path,
            "raw_sha256": self.raw_sha256,
            "raw_size_bytes": self.raw_size_bytes,
            "redistribution_policy": self.redistribution_policy,
            "release_eligible": self.release_eligible,
            "snapshot_id": self.snapshot_id,
            "source_key": self.source_key,
        }

    def canonical_bytes(self) -> bytes:
        """返回以一个换行结束的规范 manifest 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回 snapshot manifest 规范 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ThirdPartyRawSnapshotManifest":
        """从规范 object 恢复 snapshot manifest。"""
        return cls(
            value["format_version"],
            str(value["source_key"]),
            str(value["snapshot_id"]),
            str(value["official_url"]),
            str(value["raw_relative_path"]),
            str(value["raw_sha256"]),
            value["raw_size_bytes"],
            str(value["compression"]),
            str(value["encoding"]),
            value["adapter_version"],
            value["parser_version"],
            HttpCaptureMetadata.from_dict(value["http"]),
            str(value["expected_license_id"]),
            str(value["observed_license_id"]),
            str(value["license_evidence_url"]),
            str(value["license_status"]),
            str(value["redistribution_policy"]),
            value["release_eligible"],
            str(value["blocker_code"]),
            str(value["attribution"]),
            value["declared_record_count"],
            value["parsed_record_count"],
            value["anomaly_count"],
            str(value["parser_event_sha256"]),
            CanonicalJsonObject.from_value(value["parser_report"]),
        )


def read_raw_snapshot_manifest(
        path: str | Path) -> ThirdPartyRawSnapshotManifest:
    """严格读取规范 snapshot manifest。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise RawSnapshotError("snapshot manifest 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise RawSnapshotError("snapshot manifest 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = ThirdPartyRawSnapshotManifest.from_dict(value)
    except Exception as error:
        raise RawSnapshotError("snapshot manifest 合同或规范编码损坏") from error
    if manifest.canonical_bytes() != payload:
        raise RawSnapshotError("snapshot manifest 规范字节漂移")
    return manifest


def write_raw_snapshot_manifest(
        manifest: ThirdPartyRawSnapshotManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等核对 snapshot manifest，禁止原地改版。"""
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise RawSnapshotError("snapshot manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise RawSnapshotError("snapshot manifest 无法发布") from error
    return target


def verify_raw_snapshot(
        manifest: ThirdPartyRawSnapshotManifest,
        raw_root: str | Path,
        ) -> None:
    """逐字节复核 raw path、size 和 SHA-256，不因 manifest 存在而信任。"""
    root = Path(raw_root).resolve()
    path = (root / Path(*PurePosixPath(
        manifest.raw_relative_path).parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RawSnapshotError("snapshot raw 文件缺失或逃逸 root")
    if path.stat().st_size != manifest.raw_size_bytes:
        raise RawSnapshotError("snapshot raw size 漂移")
    if sha256_path(path) != manifest.raw_sha256:
        raise RawSnapshotError("snapshot raw SHA-256 漂移")


__all__ = [
    "HttpCaptureMetadata",
    "LICENSE_STATUSES",
    "RawSnapshotError",
    "SNAPSHOT_POLICIES",
    "ThirdPartyRawSnapshotManifest",
    "read_http_capture",
    "read_raw_snapshot_manifest",
    "sha256_path",
    "verify_raw_snapshot",
    "write_raw_snapshot_manifest",
]
