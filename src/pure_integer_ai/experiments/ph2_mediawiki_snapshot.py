"""PH2 MediaWiki 定日 multistream dump 的不可覆盖 snapshot 合同。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiScanReport,
)


FORMAT_VERSION = 1
ADAPTER_VERSION = 1
PARSER_VERSION = 1
LICENSE_ID = "CC-BY-SA-4.0"
LICENSE_EVIDENCE_URL = (
    "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
)
WIKTIONARY_ATTRIBUTION_POLICY = (
    "Wiktionary contributors; retain page title, page id, revision id, "
    "revision timestamp, contributor metadata, source URL, and "
    "CC-BY-SA-4.0 notice in every redistributed derived record"
)
WIKIPEDIA_ATTRIBUTION_POLICY = (
    "Wikipedia contributors; retain page title, page id, revision id, "
    "revision timestamp, contributor metadata, source URL, and "
    "CC-BY-SA-4.0 notice in every redistributed derived record"
)
# 保留冻结 Wiktionary snapshot 已使用的向后兼容公开名称。
ATTRIBUTION_POLICY = WIKTIONARY_ATTRIBUTION_POLICY

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MD5_RE = re.compile(r"[0-9a-f]{32}")
_PROJECT_RE = re.compile(r"[a-z][a-z0-9]*")
_DATE_RE = re.compile(r"[0-9]{8}")
_CHECKSUM_LINE_RE = re.compile(r"([0-9a-f]{40})  ([^\r\n]+)")


class MediaWikiSnapshotError(RuntimeError):
    """MediaWiki 官方证据、raw 身份或 parser report 不一致。"""


def attribution_policy_for_project(project: str) -> str:
    """按显式授权的项目族返回对应署名政策。"""
    if project == "zhwiktionary":
        return WIKTIONARY_ATTRIBUTION_POLICY
    if project == "zhwiki":
        return WIKIPEDIA_ATTRIBUTION_POLICY
    raise MediaWikiSnapshotError(
        "MediaWiki project has no frozen attribution policy")


def source_key_for_project(project: str, dump_date: str) -> str:
    """按显式项目映射返回 PH2 来源键。"""
    prefixes = {
        "zhwiki": "ZHWIKIPEDIA",
        "zhwiktionary": "ZHWIKTIONARY",
    }
    try:
        prefix = prefixes[project]
    except KeyError as error:
        raise MediaWikiSnapshotError(
            "MediaWiki project has no frozen source key") from error
    return f"{prefix}_{dump_date}"


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise MediaWikiSnapshotError(f"{where} must be non-empty text")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise MediaWikiSnapshotError(f"{where} must be a positive integer")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise MediaWikiSnapshotError(
            f"{where} must be a nonnegative integer")
    return value


def _hash(value: Any, *, where: str, pattern: re.Pattern[str]) -> str:
    text = _text(value, where=where).lower()
    if pattern.fullmatch(text) is None:
        raise MediaWikiSnapshotError(f"{where} has an invalid digest")
    return text


def _sha1(value: Any, *, where: str) -> str:
    return _hash(value, where=where, pattern=_SHA1_RE)


def _sha256(value: Any, *, where: str) -> str:
    return _hash(value, where=where, pattern=_SHA256_RE)


def _md5(value: Any, *, where: str) -> str:
    return _hash(value, where=where, pattern=_MD5_RE)


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise MediaWikiSnapshotError(
            f"{where} must be a safe POSIX relative path")
    return text


def _require_keys(
        value: Any,
        expected: set[str],
        *,
        where: str,
        ) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise MediaWikiSnapshotError(f"{where} fields are not exact")
    return value


def _resolve_under(root: Path, relative_path: str) -> Path:
    path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    if not path.is_relative_to(root):
        raise MediaWikiSnapshotError("MediaWiki snapshot path escapes root")
    return path


def _file_hashes(path: Path) -> tuple[int, str, str]:
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                sha1.update(block)
                sha256.update(block)
    except OSError as error:
        raise MediaWikiSnapshotError("MediaWiki evidence file unreadable") from error
    return size, sha1.hexdigest(), sha256.hexdigest()


def _sha256_path(path: Path) -> str:
    return _file_hashes(path)[2]


def _json_no_duplicates(payload: bytes) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MediaWikiSnapshotError(
                    "dumpstatus contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_float(value: str) -> None:
        raise MediaWikiSnapshotError(
            f"dumpstatus contains unsupported numeric token {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MediaWikiSnapshotError("dumpstatus JSON is damaged") from error
    if not isinstance(value, dict):
        raise MediaWikiSnapshotError("dumpstatus root must be an object")
    return value


def _dump_names(project: str, dump_date: str) -> tuple[str, str]:
    prefix = f"{project}-{dump_date}-pages-articles-multistream"
    return f"{prefix}.xml.bz2", f"{prefix}-index.txt.bz2"


def _official_base(project: str, dump_date: str) -> str:
    return f"https://dumps.wikimedia.org/{project}/{dump_date}"


def _validate_report(report: MediaWikiScanReport, source_key: str) -> None:
    if not isinstance(report, MediaWikiScanReport):
        raise MediaWikiSnapshotError("parser report has the wrong type")
    if report.source_key != source_key:
        raise MediaWikiSnapshotError("parser report source does not match")
    for name, value in (
            ("xml_sha256", report.xml_sha256),
            ("index_sha256", report.index_sha256),
            ("event_sha256", report.event_sha256)):
        _sha256(value, where=f"parser report {name}")
    for name in (
            "page_count", "main_namespace_count", "skipped_namespace_count",
            "valid_page_count", "anomaly_count", "text_size_bytes",
            "template_count", "max_page_text_bytes", "max_template_count",
            "xml_event_count", "work_unit_count"):
        _nonnegative(getattr(report, name), where=f"parser report {name}")
    if report.full_eof_verified != 1:
        raise MediaWikiSnapshotError("parser report is not a verified full EOF")
    if (report.page_count != report.main_namespace_count
            + report.skipped_namespace_count):
        raise MediaWikiSnapshotError("parser report namespace counts disagree")
    if (report.main_namespace_count != report.valid_page_count
            + report.anomaly_count):
        raise MediaWikiSnapshotError("parser report page outcomes disagree")
    if report.work_unit_count != (
            report.xml_event_count + report.page_count
            + report.text_size_bytes + report.template_count):
        raise MediaWikiSnapshotError("parser report work count disagrees")
    codes = report.anomaly_codes.to_value()
    if not isinstance(codes, dict):
        raise MediaWikiSnapshotError("parser anomaly codes are invalid")
    for code, count in codes.items():
        _text(code, where="parser anomaly code")
        _positive(count, where=f"parser anomaly count {code}")
    evidence = report.anomaly_evidence.to_value()
    if (not isinstance(evidence, dict) or set(evidence) != {"items"}
            or not isinstance(evidence["items"], list)):
        raise MediaWikiSnapshotError("parser anomaly evidence is invalid")
    if (sum(codes.values()) != report.anomaly_count
            or len(evidence["items"]) != report.anomaly_count):
        raise MediaWikiSnapshotError("parser anomaly totals disagree")


_REPORT_KEYS = {
    "anomaly_codes", "anomaly_count", "anomaly_evidence", "event_sha256",
    "full_eof_verified", "index_sha256", "main_namespace_count",
    "max_page_text_bytes", "max_template_count", "page_count",
    "skipped_namespace_count", "source_key", "template_count",
    "text_size_bytes", "valid_page_count", "work_unit_count",
    "xml_event_count", "xml_sha256",
}


def read_mediawiki_scan_report(path: str | Path) -> MediaWikiScanReport:
    """回读一份规范保存的 report，不重新扫描 dump。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise MediaWikiSnapshotError("saved parser report newline is invalid")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        _require_keys(value, _REPORT_KEYS, where="saved parser report")
        report = MediaWikiScanReport.from_dict(value)
    except MediaWikiSnapshotError:
        raise
    except Exception as error:
        raise MediaWikiSnapshotError("saved parser report is damaged") from error
    _validate_report(report, report.source_key)
    if canonical_json_line(report.to_dict()) != payload:
        raise MediaWikiSnapshotError("saved parser report is not canonical")
    return report


@dataclass(frozen=True)
class MediaWikiSnapshotAcquisition:
    """仅在构建定日 dump snapshot 时使用的可迁移路径。"""

    dumpstatus_relative_path: str
    dumpstatus_headers_relative_path: str
    checksum_relative_path: str
    checksum_headers_relative_path: str
    xml_relative_path: str
    xml_headers_relative_path: str
    index_relative_path: str
    index_headers_relative_path: str
    report_relative_paths: tuple[str, str]

    def __post_init__(self) -> None:
        for name in (
                "dumpstatus_relative_path",
                "dumpstatus_headers_relative_path",
                "checksum_relative_path",
                "checksum_headers_relative_path",
                "xml_relative_path",
                "xml_headers_relative_path",
                "index_relative_path",
                "index_headers_relative_path"):
            object.__setattr__(
                self, name,
                _relative_path(getattr(self, name), where=f"acquisition {name}"),
            )
        if (not isinstance(self.report_relative_paths, tuple)
                or len(self.report_relative_paths) != 2):
            raise MediaWikiSnapshotError("acquisition requires two reports")
        reports = tuple(_relative_path(
            item, where="acquisition report path")
            for item in self.report_relative_paths)
        if len(set(reports)) != 2:
            raise MediaWikiSnapshotError("acquisition report paths repeat")
        object.__setattr__(self, "report_relative_paths", reports)


@dataclass(frozen=True)
class MediaWikiOfficialEvidence:
    """一份官方 metadata 及其完整响应头 hash。"""

    role: str
    relative_path: str
    headers_relative_path: str
    official_url: str
    size_bytes: int
    sha256: str
    response_headers_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"DUMPSTATUS", "PROJECT_SHA1SUMS"}:
            raise MediaWikiSnapshotError("official evidence role is invalid")
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path, where="official evidence path"))
        object.__setattr__(self, "headers_relative_path", _relative_path(
            self.headers_relative_path, where="official headers path"))
        if not self.official_url.startswith("https://dumps.wikimedia.org/"):
            raise MediaWikiSnapshotError("official evidence URL is invalid")
        _positive(self.size_bytes, where="official evidence size")
        object.__setattr__(self, "sha256", _sha256(
            self.sha256, where="official evidence SHA-256"))
        object.__setattr__(self, "response_headers_sha256", _sha256(
            self.response_headers_sha256,
            where="official response headers SHA-256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "headers_relative_path": self.headers_relative_path,
            "official_url": self.official_url,
            "relative_path": self.relative_path,
            "response_headers_sha256": self.response_headers_sha256,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MediaWikiOfficialEvidence":
        _require_keys(value, {
            "headers_relative_path", "official_url", "relative_path",
            "response_headers_sha256", "role", "sha256", "size_bytes",
        }, where="official evidence")
        return cls(
            str(value["role"]), str(value["relative_path"]),
            str(value["headers_relative_path"]), str(value["official_url"]),
            value["size_bytes"], str(value["sha256"]),
            str(value["response_headers_sha256"]),
        )


@dataclass(frozen=True)
class MediaWikiCompressedFileIdentity:
    """同时绑定官方摘要与本地摘要的压缩 raw 字节身份。"""

    role: str
    raw_relative_path: str
    headers_relative_path: str
    official_url: str
    compressed_size_bytes: int
    upstream_sha1: str
    upstream_md5: str
    local_sha256: str
    response_headers_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"XML", "INDEX"}:
            raise MediaWikiSnapshotError("compressed file role is invalid")
        object.__setattr__(self, "raw_relative_path", _relative_path(
            self.raw_relative_path, where="compressed raw path"))
        object.__setattr__(self, "headers_relative_path", _relative_path(
            self.headers_relative_path, where="compressed headers path"))
        if not self.official_url.startswith("https://dumps.wikimedia.org/"):
            raise MediaWikiSnapshotError("compressed file URL is invalid")
        _positive(self.compressed_size_bytes, where="compressed file size")
        object.__setattr__(self, "upstream_sha1", _sha1(
            self.upstream_sha1, where="compressed upstream SHA-1"))
        object.__setattr__(self, "upstream_md5", _md5(
            self.upstream_md5, where="compressed upstream MD5"))
        for name in (
                "local_sha256", "response_headers_sha256", "content_sha256"):
            object.__setattr__(self, name, _sha256(
                getattr(self, name), where=f"compressed {name}"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "compressed_size_bytes": self.compressed_size_bytes,
            "content_sha256": self.content_sha256,
            "headers_relative_path": self.headers_relative_path,
            "local_sha256": self.local_sha256,
            "official_url": self.official_url,
            "raw_relative_path": self.raw_relative_path,
            "response_headers_sha256": self.response_headers_sha256,
            "role": self.role,
            "upstream_md5": self.upstream_md5,
            "upstream_sha1": self.upstream_sha1,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "MediaWikiCompressedFileIdentity":
        _require_keys(value, {
            "compressed_size_bytes", "content_sha256",
            "headers_relative_path", "local_sha256", "official_url",
            "raw_relative_path", "response_headers_sha256", "role",
            "upstream_md5", "upstream_sha1",
        }, where="compressed file identity")
        return cls(
            str(value["role"]), str(value["raw_relative_path"]),
            str(value["headers_relative_path"]), str(value["official_url"]),
            value["compressed_size_bytes"], str(value["upstream_sha1"]),
            str(value["upstream_md5"]), str(value["local_sha256"]),
            str(value["response_headers_sha256"]),
            str(value["content_sha256"]),
        )


@dataclass(frozen=True)
class MediaWikiParserReportIdentity:
    """一份已保存 full-EOF parser report 的文件身份。"""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path, where="parser report path"))
        _positive(self.size_bytes, where="parser report size")
        object.__setattr__(self, "sha256", _sha256(
            self.sha256, where="parser report SHA-256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "MediaWikiParserReportIdentity":
        _require_keys(value, {"relative_path", "sha256", "size_bytes"},
                      where="parser report identity")
        return cls(
            str(value["relative_path"]), value["size_bytes"],
            str(value["sha256"]),
        )


@dataclass(frozen=True)
class MediaWikiDumpSnapshotManifest:
    """同时携带双官方来源和双遍相等证据的定日 multistream snapshot。"""

    format_version: int
    source_key: str
    snapshot_id: str
    project: str
    dump_date: str
    dumpstatus_schema_version: str
    dump_job: str
    dump_job_status: str
    dump_job_updated: str
    adapter_version: int
    parser_version: int
    license_id: str
    license_evidence_url: str
    attribution_policy: str
    redistribution_policy: str
    release_eligible: int
    definitive_truth_authoritative: int
    compression: str
    official_evidence: tuple[MediaWikiOfficialEvidence, ...]
    raw_files: tuple[MediaWikiCompressedFileIdentity, ...]
    parser_reports: tuple[MediaWikiParserReportIdentity, ...]
    double_pass_equal: int
    final_parser_report: MediaWikiScanReport

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise MediaWikiSnapshotError("manifest format version is invalid")
        _text(self.snapshot_id, where="manifest snapshot id")
        if (_PROJECT_RE.fullmatch(self.project) is None
                or _DATE_RE.fullmatch(self.dump_date) is None):
            raise MediaWikiSnapshotError("manifest project/date is invalid")
        if self.source_key != source_key_for_project(
                self.project, self.dump_date):
            raise MediaWikiSnapshotError("manifest source key is invalid")
        if self.dumpstatus_schema_version != "0.8":
            raise MediaWikiSnapshotError("dumpstatus schema version is invalid")
        if (self.dump_job != "articlesmultistreamdumprecombine"
                or self.dump_job_status != "done"):
            raise MediaWikiSnapshotError("dump job is not complete")
        _text(self.dump_job_updated, where="dump job updated")
        if (self.adapter_version != ADAPTER_VERSION
                or self.parser_version != PARSER_VERSION):
            raise MediaWikiSnapshotError("adapter/parser version is invalid")
        if (self.license_id != LICENSE_ID
                or self.license_evidence_url != LICENSE_EVIDENCE_URL
                or self.attribution_policy
                != attribution_policy_for_project(self.project)):
            raise MediaWikiSnapshotError("license or attribution is invalid")
        if (self.redistribution_policy != "PUBLIC"
                or self.release_eligible != 1):
            raise MediaWikiSnapshotError("redistribution decision is invalid")
        if self.definitive_truth_authoritative != 0:
            raise MediaWikiSnapshotError("external text cannot be definitive truth")
        if self.compression != "bzip2":
            raise MediaWikiSnapshotError("compression is invalid")
        if (not isinstance(self.official_evidence, tuple)
                or not all(isinstance(item, MediaWikiOfficialEvidence)
                           for item in self.official_evidence)
                or {item.role for item in self.official_evidence}
                != {"DUMPSTATUS", "PROJECT_SHA1SUMS"}
                or len(self.official_evidence) != 2):
            raise MediaWikiSnapshotError("official evidence set is incomplete")
        object.__setattr__(self, "official_evidence", tuple(sorted(
            self.official_evidence,
            key=lambda item: {"DUMPSTATUS": 0, "PROJECT_SHA1SUMS": 1}[
                item.role],
        )))
        if (not isinstance(self.raw_files, tuple)
                or not all(isinstance(item, MediaWikiCompressedFileIdentity)
                           for item in self.raw_files)
                or {item.role for item in self.raw_files} != {"XML", "INDEX"}
                or len(self.raw_files) != 2):
            raise MediaWikiSnapshotError("compressed raw file set is incomplete")
        object.__setattr__(self, "raw_files", tuple(sorted(
            self.raw_files,
            key=lambda item: {"XML": 0, "INDEX": 1}[item.role],
        )))
        if (not isinstance(self.parser_reports, tuple)
                or not all(isinstance(item, MediaWikiParserReportIdentity)
                           for item in self.parser_reports)
                or len(self.parser_reports) != 2
                or len({item.relative_path for item in self.parser_reports}) != 2
                or len({item.sha256 for item in self.parser_reports}) != 1):
            raise MediaWikiSnapshotError("double-pass report identities disagree")
        object.__setattr__(self, "parser_reports", tuple(sorted(
            self.parser_reports, key=lambda item: item.relative_path)))
        if self.double_pass_equal != 1:
            raise MediaWikiSnapshotError("double-pass equality is not proven")
        _validate_report(self.final_parser_report, self.source_key)
        base = _official_base(self.project, self.dump_date)
        xml_name, index_name = _dump_names(self.project, self.dump_date)
        official = {item.role: item for item in self.official_evidence}
        expected_official = {
            "DUMPSTATUS": ("dumpstatus.json", f"{base}/dumpstatus.json"),
            "PROJECT_SHA1SUMS": (
                f"{self.project}-{self.dump_date}-sha1sums.txt",
                f"{base}/{self.project}-{self.dump_date}-sha1sums.txt",
            ),
        }
        for role, (name, url) in expected_official.items():
            item = official[role]
            if (PurePosixPath(item.relative_path).name != name
                    or item.official_url != url):
                raise MediaWikiSnapshotError(
                    "official evidence project/date binding is invalid")
        raw = {item.role: item for item in self.raw_files}
        for role, name in (("XML", xml_name), ("INDEX", index_name)):
            item = raw[role]
            if (PurePosixPath(item.raw_relative_path).name != name
                    or item.official_url != f"{base}/{name}"):
                raise MediaWikiSnapshotError(
                    "compressed file project/date binding is invalid")
        if (raw["XML"].content_sha256
                != self.final_parser_report.xml_sha256
                or raw["INDEX"].content_sha256
                != self.final_parser_report.index_sha256):
            raise MediaWikiSnapshotError(
                "compressed files and parser content hashes disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "attribution_policy": self.attribution_policy,
            "compression": self.compression,
            "definitive_truth_authoritative": (
                self.definitive_truth_authoritative),
            "double_pass_equal": self.double_pass_equal,
            "dump_date": self.dump_date,
            "dump_job": self.dump_job,
            "dump_job_status": self.dump_job_status,
            "dump_job_updated": self.dump_job_updated,
            "dumpstatus_schema_version": self.dumpstatus_schema_version,
            "final_parser_report": self.final_parser_report.to_dict(),
            "format_version": self.format_version,
            "license_evidence_url": self.license_evidence_url,
            "license_id": self.license_id,
            "official_evidence": [
                item.to_dict() for item in self.official_evidence],
            "parser_reports": [item.to_dict() for item in self.parser_reports],
            "parser_version": self.parser_version,
            "project": self.project,
            "raw_files": [item.to_dict() for item in self.raw_files],
            "redistribution_policy": self.redistribution_policy,
            "release_eligible": self.release_eligible,
            "snapshot_id": self.snapshot_id,
            "source_key": self.source_key,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "MediaWikiDumpSnapshotManifest":
        _require_keys(value, {
            "adapter_version", "attribution_policy", "compression",
            "definitive_truth_authoritative", "double_pass_equal",
            "dump_date", "dump_job", "dump_job_status", "dump_job_updated",
            "dumpstatus_schema_version", "final_parser_report",
            "format_version", "license_evidence_url", "license_id",
            "official_evidence", "parser_reports", "parser_version",
            "project", "raw_files", "redistribution_policy",
            "release_eligible", "snapshot_id", "source_key",
        }, where="MediaWiki snapshot manifest")
        report_value = _require_keys(
            value["final_parser_report"], _REPORT_KEYS,
            where="embedded parser report")
        return cls(
            value["format_version"], str(value["source_key"]),
            str(value["snapshot_id"]), str(value["project"]),
            str(value["dump_date"]),
            str(value["dumpstatus_schema_version"]),
            str(value["dump_job"]), str(value["dump_job_status"]),
            str(value["dump_job_updated"]), value["adapter_version"],
            value["parser_version"], str(value["license_id"]),
            str(value["license_evidence_url"]),
            str(value["attribution_policy"]),
            str(value["redistribution_policy"]), value["release_eligible"],
            value["definitive_truth_authoritative"],
            str(value["compression"]),
            tuple(MediaWikiOfficialEvidence.from_dict(item)
                  for item in value["official_evidence"]),
            tuple(MediaWikiCompressedFileIdentity.from_dict(item)
                  for item in value["raw_files"]),
            tuple(MediaWikiParserReportIdentity.from_dict(item)
                  for item in value["parser_reports"]),
            value["double_pass_equal"],
            MediaWikiScanReport.from_dict(report_value),
        )


def _read_dumpstatus(
        path: Path,
        *,
        project: str,
        dump_date: str,
        ) -> tuple[str, str, dict[str, dict[str, Any]]]:
    try:
        root = _json_no_duplicates(path.read_bytes())
        schema_version = str(root["version"])
        job = root["jobs"]["articlesmultistreamdumprecombine"]
        status = str(job["status"])
        updated = str(job["updated"])
        files = job["files"]
    except (OSError, KeyError, TypeError) as error:
        raise MediaWikiSnapshotError("dumpstatus structure is invalid") from error
    if schema_version != "0.8" or status != "done" or not isinstance(files, dict):
        raise MediaWikiSnapshotError("dumpstatus job is not a completed v0.8 job")
    xml_name, index_name = _dump_names(project, dump_date)
    selected: dict[str, dict[str, Any]] = {}
    for name in (xml_name, index_name):
        item = files.get(name)
        if not isinstance(item, dict):
            raise MediaWikiSnapshotError("dumpstatus lacks a required dump file")
        try:
            selected[name] = {
                "md5": _md5(item["md5"], where="dumpstatus MD5"),
                "sha1": _sha1(item["sha1"], where="dumpstatus SHA-1"),
                "size": _positive(item["size"], where="dumpstatus size"),
                "url": _text(item["url"], where="dumpstatus URL"),
            }
        except KeyError as error:
            raise MediaWikiSnapshotError(
                "dumpstatus file identity is incomplete") from error
        if selected[name]["url"] != f"/{project}/{dump_date}/{name}":
            raise MediaWikiSnapshotError("dumpstatus URL does not bind project/date")
    return schema_version, updated, selected


def _read_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise MediaWikiSnapshotError("project checksum file is unreadable") from error
    result: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise MediaWikiSnapshotError("project checksum line is invalid")
        digest, name = match.groups()
        if name in result:
            raise MediaWikiSnapshotError("project checksum filename repeats")
        result[name] = digest
    return result


def _evidence_identity(
        *,
        root: Path,
        role: str,
        relative_path: str,
        headers_relative_path: str,
        official_url: str,
        ) -> MediaWikiOfficialEvidence:
    path = _resolve_under(root, relative_path)
    headers = _resolve_under(root, headers_relative_path)
    size, _, digest = _file_hashes(path)
    return MediaWikiOfficialEvidence(
        role, relative_path, headers_relative_path, official_url, size, digest,
        _sha256_path(headers),
    )


def build_mediawiki_dump_snapshot(
        *,
        raw_root: str | Path,
        source_key: str,
        project: str,
        dump_date: str,
        snapshot_id: str,
        acquisition: MediaWikiSnapshotAcquisition,
        ) -> MediaWikiDumpSnapshotManifest:
    """只用已保存 report 与 raw hash 构建，绝不调用 parser。"""
    root = Path(raw_root).resolve()
    if not isinstance(acquisition, MediaWikiSnapshotAcquisition):
        raise MediaWikiSnapshotError("acquisition has the wrong type")
    if source_key != source_key_for_project(project, dump_date):
        raise MediaWikiSnapshotError("source key does not bind project/date")
    base = _official_base(project, dump_date)
    dumpstatus_path = _resolve_under(root, acquisition.dumpstatus_relative_path)
    checksum_path = _resolve_under(root, acquisition.checksum_relative_path)
    schema_version, updated, status_files = _read_dumpstatus(
        dumpstatus_path, project=project, dump_date=dump_date)
    checksums = _read_checksums(checksum_path)
    xml_name, index_name = _dump_names(project, dump_date)
    reports: list[MediaWikiScanReport] = []
    report_identities: list[MediaWikiParserReportIdentity] = []
    for relative_path in acquisition.report_relative_paths:
        path = _resolve_under(root, relative_path)
        report = read_mediawiki_scan_report(path)
        _validate_report(report, source_key)
        size, _, digest = _file_hashes(path)
        reports.append(report)
        report_identities.append(MediaWikiParserReportIdentity(
            relative_path, size, digest))
    if reports[0] != reports[1]:
        raise MediaWikiSnapshotError("saved double-pass reports disagree")
    report = reports[0]
    raw_inputs = (
        ("XML", xml_name, acquisition.xml_relative_path,
         acquisition.xml_headers_relative_path, report.xml_sha256),
        ("INDEX", index_name, acquisition.index_relative_path,
         acquisition.index_headers_relative_path, report.index_sha256),
    )
    raw_files: list[MediaWikiCompressedFileIdentity] = []
    for role, name, relative_path, headers_path, content_sha256 in raw_inputs:
        status = status_files[name]
        checksum_sha1 = checksums.get(name)
        if checksum_sha1 != status["sha1"]:
            raise MediaWikiSnapshotError(
                "dumpstatus and project checksum SHA-1 disagree")
        path = _resolve_under(root, relative_path)
        size, local_sha1, local_sha256 = _file_hashes(path)
        if size != status["size"] or local_sha1 != status["sha1"]:
            raise MediaWikiSnapshotError(
                "compressed raw size or upstream SHA-1 disagrees")
        raw_files.append(MediaWikiCompressedFileIdentity(
            role, relative_path, headers_path, f"{base}/{name}", size,
            local_sha1, status["md5"], local_sha256,
            _sha256_path(_resolve_under(root, headers_path)), content_sha256,
        ))
    official_evidence = (
        _evidence_identity(
            root=root, role="DUMPSTATUS",
            relative_path=acquisition.dumpstatus_relative_path,
            headers_relative_path=acquisition.dumpstatus_headers_relative_path,
            official_url=f"{base}/dumpstatus.json"),
        _evidence_identity(
            root=root, role="PROJECT_SHA1SUMS",
            relative_path=acquisition.checksum_relative_path,
            headers_relative_path=acquisition.checksum_headers_relative_path,
            official_url=f"{base}/{project}-{dump_date}-sha1sums.txt"),
    )
    return MediaWikiDumpSnapshotManifest(
        FORMAT_VERSION, source_key, snapshot_id, project, dump_date,
        schema_version, "articlesmultistreamdumprecombine", "done", updated,
        ADAPTER_VERSION, PARSER_VERSION, LICENSE_ID, LICENSE_EVIDENCE_URL,
        attribution_policy_for_project(project), "PUBLIC", 1, 0, "bzip2",
        official_evidence,
        tuple(raw_files), tuple(report_identities), 1, report,
    )


def read_mediawiki_dump_snapshot(
        path: str | Path,
        ) -> MediaWikiDumpSnapshotManifest:
    """严格恢复一份规范 MediaWiki snapshot manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise MediaWikiSnapshotError("snapshot manifest newline is invalid")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = MediaWikiDumpSnapshotManifest.from_dict(value)
    except MediaWikiSnapshotError:
        raise
    except Exception as error:
        raise MediaWikiSnapshotError("snapshot manifest is damaged") from error
    if manifest.canonical_bytes() != payload:
        raise MediaWikiSnapshotError("snapshot manifest is not canonical")
    return manifest


def write_mediawiki_dump_snapshot(
        manifest: MediaWikiDumpSnapshotManifest,
        path: str | Path,
        ) -> Path:
    """独占发布或核对幂等，禁止覆盖已有版本。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise MediaWikiSnapshotError("snapshot manifest has the wrong type")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise MediaWikiSnapshotError(
                "MediaWiki snapshot exists with different content")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise MediaWikiSnapshotError("MediaWiki snapshot cannot be published") from error
    return target


def verify_mediawiki_dump_snapshot(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        raw_root: str | Path,
        ) -> None:
    """不解压、不重扫 parser，只核 raw、官方证据和 report 字节。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise MediaWikiSnapshotError("snapshot manifest has the wrong type")
    root = Path(raw_root).resolve()
    official = {item.role: item for item in manifest.official_evidence}
    for item in manifest.official_evidence:
        path = _resolve_under(root, item.relative_path)
        headers = _resolve_under(root, item.headers_relative_path)
        size, _, digest = _file_hashes(path)
        if (size != item.size_bytes or digest != item.sha256
                or _sha256_path(headers) != item.response_headers_sha256):
            raise MediaWikiSnapshotError("official evidence identity drifted")
    _, updated, status_files = _read_dumpstatus(
        _resolve_under(root, official["DUMPSTATUS"].relative_path),
        project=manifest.project, dump_date=manifest.dump_date)
    if updated != manifest.dump_job_updated:
        raise MediaWikiSnapshotError("dumpstatus update identity drifted")
    checksums = _read_checksums(_resolve_under(
        root, official["PROJECT_SHA1SUMS"].relative_path))
    raw_by_role = {item.role: item for item in manifest.raw_files}
    xml_name, index_name = _dump_names(manifest.project, manifest.dump_date)
    for role, name in (("XML", xml_name), ("INDEX", index_name)):
        item = raw_by_role[role]
        status = status_files[name]
        if (checksums.get(name) != status["sha1"]
                or item.upstream_sha1 != status["sha1"]
                or item.upstream_md5 != status["md5"]):
            raise MediaWikiSnapshotError("official checksum evidence drifted")
        size, sha1, sha256 = _file_hashes(
            _resolve_under(root, item.raw_relative_path))
        if (size != item.compressed_size_bytes
                or sha1 != item.upstream_sha1
                or sha256 != item.local_sha256
                or _sha256_path(_resolve_under(root, item.headers_relative_path))
                != item.response_headers_sha256):
            raise MediaWikiSnapshotError("compressed raw identity drifted")
    restored_reports: list[MediaWikiScanReport] = []
    for item in manifest.parser_reports:
        path = _resolve_under(root, item.relative_path)
        size, _, digest = _file_hashes(path)
        if size != item.size_bytes or digest != item.sha256:
            raise MediaWikiSnapshotError("saved parser report identity drifted")
        restored_reports.append(read_mediawiki_scan_report(path))
    if (restored_reports[0] != restored_reports[1]
            or restored_reports[0] != manifest.final_parser_report):
        raise MediaWikiSnapshotError("saved parser reports no longer agree")


__all__ = [
    "ADAPTER_VERSION",
    "ATTRIBUTION_POLICY",
    "FORMAT_VERSION",
    "LICENSE_EVIDENCE_URL",
    "LICENSE_ID",
    "PARSER_VERSION",
    "WIKIPEDIA_ATTRIBUTION_POLICY",
    "WIKTIONARY_ATTRIBUTION_POLICY",
    "MediaWikiCompressedFileIdentity",
    "MediaWikiDumpSnapshotManifest",
    "MediaWikiOfficialEvidence",
    "MediaWikiParserReportIdentity",
    "MediaWikiSnapshotAcquisition",
    "MediaWikiSnapshotError",
    "attribution_policy_for_project",
    "build_mediawiki_dump_snapshot",
    "read_mediawiki_dump_snapshot",
    "read_mediawiki_scan_report",
    "source_key_for_project",
    "verify_mediawiki_dump_snapshot",
    "write_mediawiki_dump_snapshot",
]
