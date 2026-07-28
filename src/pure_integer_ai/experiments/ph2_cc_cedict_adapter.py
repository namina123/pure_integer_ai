"""CC_CEDICT_20260725 gzip 的严格解析、许可审计和双遍稳定扫描。"""
from __future__ import annotations

import gzip
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import sha256_path


SOURCE_KEY = "CC_CEDICT_20260725"
SNAPSHOT_ID = "20260725"
OFFICIAL_URL = (
    "https://www.mdbg.net/chinese/export/cedict/"
    "cedict_1_0_ts_utf-8_mdbg.txt.gz")
LICENSE_EVIDENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
EXPECTED_LICENSE_ID = "CC-BY-SA-3.0"
OBSERVED_LICENSE_ID = "CC-BY-SA-4.0"
ADAPTER_VERSION = 1
PARSER_VERSION = 1

_ENTRY = re.compile(r"^(\S+) (\S+) \[([^\]]+)\] /(.+)/$")
_LICENSE_URLS = {
    "https://creativecommons.org/licenses/by-sa/3.0/": "CC-BY-SA-3.0",
    "https://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
}


class CcCedictAdapterError(RuntimeError):
    """CC-CEDICT gzip、行格式、header metadata 或许可证据损坏。"""


@dataclass(frozen=True)
class CcCedictEntry:
    """一条只作词形/候选 Evidence 的词典行，不把英文 gloss 当真值。"""

    line_number: int
    traditional: str
    simplified: str
    pinyin: str
    glosses: tuple[str, ...]
    raw_line_sha256: str

    def __post_init__(self) -> None:
        if type(self.line_number) is not int or self.line_number <= 0:
            raise CcCedictAdapterError("CEDICT line_number 必须是正严格整数")
        for name, value in (
                ("traditional", self.traditional),
                ("simplified", self.simplified),
                ("pinyin", self.pinyin)):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise CcCedictAdapterError(f"CEDICT {name} 非法")
        if (not isinstance(self.glosses, tuple) or not self.glosses
                or any(not isinstance(item, str) or not item
                       or item.strip() != item for item in self.glosses)):
            raise CcCedictAdapterError("CEDICT glosses 非法")
        if (len(self.raw_line_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.raw_line_sha256)):
            raise CcCedictAdapterError("CEDICT raw_line_sha256 非法")

    def to_dict(self) -> dict[str, Any]:
        """导出 parser event，显式钉死 gloss 非权威。"""
        return {
            "english_gloss_authoritative": 0,
            "glosses": list(self.glosses),
            "line_number": self.line_number,
            "pinyin": self.pinyin,
            "raw_line_sha256": self.raw_line_sha256,
            "simplified": self.simplified,
            "traditional": self.traditional,
        }


@dataclass(frozen=True)
class CcCedictAnomaly:
    """保留坏行位置、原始摘要和确定性分类。"""

    line_number: int
    code: str
    raw_line_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """导出不含可逆全文的 anomaly event。"""
        return {
            "code": self.code,
            "line_number": self.line_number,
            "raw_line_sha256": self.raw_line_sha256,
        }


@dataclass(frozen=True)
class CcCedictLicenseAudit:
    """比较执行文档许可分区与实际 snapshot header。"""

    expected_license_id: str
    observed_license_id: str
    evidence_url: str
    status: str
    blocker_code: str

    def __post_init__(self) -> None:
        if self.status not in {"MATCH", "CONFLICT"}:
            raise CcCedictAdapterError("CEDICT license status 非法")
        if self.status == "MATCH" and self.blocker_code:
            raise CcCedictAdapterError("MATCH 许可不得有 blocker")
        if self.status == "CONFLICT" and not self.blocker_code:
            raise CcCedictAdapterError("CONFLICT 许可必须有 blocker")

    def to_dict(self) -> dict[str, str]:
        """导出许可审计。"""
        return {
            "blocker_code": self.blocker_code,
            "evidence_url": self.evidence_url,
            "expected_license_id": self.expected_license_id,
            "observed_license_id": self.observed_license_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class CcCedictScanReport:
    """一次完整或 sample 扫描的计数、metadata 和规范事件摘要。"""

    compressed_sha256: str
    compressed_size_bytes: int
    line_count: int
    header_count: int
    entry_count: int
    anomaly_count: int
    declared_entry_count: int
    terminal_newline_present: int
    metadata: tuple[tuple[str, str], ...]
    event_sha256: str
    license_audit: CcCedictLicenseAudit
    english_gloss_authoritative: int = 0

    def __post_init__(self) -> None:
        for name, value in (
                ("compressed_size_bytes", self.compressed_size_bytes),
                ("line_count", self.line_count),
                ("header_count", self.header_count),
                ("entry_count", self.entry_count),
                ("anomaly_count", self.anomaly_count),
                ("declared_entry_count", self.declared_entry_count)):
            if type(value) is not int or value < 0:
                raise CcCedictAdapterError(f"CEDICT {name} 非法")
        if self.terminal_newline_present not in {0, 1}:
            raise CcCedictAdapterError(
                "CEDICT terminal_newline_present 必须为 0/1")
        if self.english_gloss_authoritative != 0:
            raise CcCedictAdapterError("CEDICT gloss 不得成为中文逻辑真值")
        if len({name for name, _ in self.metadata}) != len(self.metadata):
            raise CcCedictAdapterError("CEDICT metadata key 重复")

    def to_dict(self) -> dict[str, Any]:
        """导出规范扫描摘要。"""
        return {
            "anomaly_count": self.anomaly_count,
            "compressed_sha256": self.compressed_sha256,
            "compressed_size_bytes": self.compressed_size_bytes,
            "declared_entry_count": self.declared_entry_count,
            "english_gloss_authoritative": 0,
            "entry_count": self.entry_count,
            "event_sha256": self.event_sha256,
            "header_count": self.header_count,
            "license_audit": self.license_audit.to_dict(),
            "line_count": self.line_count,
            "metadata": {name: value for name, value in self.metadata},
            "terminal_newline_present": self.terminal_newline_present,
        }


def _line_sha256(line: str) -> str:
    """计算不含换行的原始 UTF-8 行摘要。"""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def parse_cc_cedict_entry(
        line: str,
        *,
        line_number: int,
        ) -> CcCedictEntry:
    """严格解析 traditional simplified [pinyin] /gloss/ 行。"""
    if not isinstance(line, str) or not line or line.strip() != line:
        raise CcCedictAdapterError("CEDICT data line 空或有首尾空白")
    match = _ENTRY.fullmatch(line)
    if match is None:
        raise CcCedictAdapterError("CEDICT data line 格式非法")
    traditional, simplified, pinyin, gloss_blob = match.groups()
    glosses = tuple(gloss_blob.split("/"))
    if any(not gloss for gloss in glosses):
        raise CcCedictAdapterError("CEDICT gloss 为空")
    return CcCedictEntry(
        line_number,
        traditional,
        simplified,
        pinyin,
        glosses,
        _line_sha256(line),
    )


def _metadata(header_lines: list[str]) -> tuple[tuple[str, str], ...]:
    """解析 `#! key=value`，拒绝重复或空 key/value。"""
    values: dict[str, str] = {}
    for line in header_lines:
        if not line.startswith("#! "):
            continue
        content = line[3:]
        if "=" not in content:
            raise CcCedictAdapterError("CEDICT metadata header 非 key=value")
        name, value = content.split("=", 1)
        if not name or not value or name in values:
            raise CcCedictAdapterError("CEDICT metadata 空或重复")
        values[name] = value
    return tuple(sorted(values.items()))


def audit_cc_cedict_license(
        metadata: tuple[tuple[str, str], ...],
        *,
        expected_license_id: str = EXPECTED_LICENSE_ID,
        ) -> CcCedictLicenseAudit:
    """只信 `#! license=` URL，并对文档冻结分区做显式冲突裁决。"""
    values = dict(metadata)
    evidence_url = values.get("license", "")
    observed = _LICENSE_URLS.get(evidence_url)
    if observed is None:
        raise CcCedictAdapterError("CEDICT header 缺少已注册 license URL")
    status = "MATCH" if observed == expected_license_id else "CONFLICT"
    return CcCedictLicenseAudit(
        expected_license_id,
        observed,
        evidence_url,
        status,
        "" if status == "MATCH" else "LICENSE_PARTITION_MISMATCH",
    )


def _iter_lines(path: Path) -> Iterator[tuple[int, str, int]]:
    """以 strict UTF-8 和 universal newline 流式读取 gzip。"""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
            for line_number, raw in enumerate(handle, start=1):
                yield (
                    line_number,
                    raw.rstrip("\r\n"),
                    1 if raw.endswith(("\n", "\r")) else 0,
                )
    except (OSError, UnicodeError, EOFError) as error:
        raise CcCedictAdapterError("CEDICT gzip/UTF-8 损坏") from error


def scan_cc_cedict_gzip(
        path: str | Path,
        *,
        expected_compressed_sha256: str,
        complete_snapshot: bool,
        ) -> CcCedictScanReport:
    """前后核验压缩 hash，扫描 header/data/anomaly 并形成事件摘要。"""
    source = Path(path)
    if not source.is_file():
        raise CcCedictAdapterError("CEDICT gzip 缺失")
    expected = expected_compressed_sha256.lower()
    if sha256_path(source) != expected:
        raise CcCedictAdapterError("CEDICT compressed SHA-256 不一致")
    digest = hashlib.sha256()
    headers: list[str] = []
    entries = 0
    anomalies = 0
    line_count = 0
    terminal_newline_present = 0
    for line_number, line, has_line_end in _iter_lines(source):
        line_count = line_number
        terminal_newline_present = has_line_end
        if line.startswith("#"):
            headers.append(line)
            event = {
                "kind": "header",
                "line_number": line_number,
                "raw_line_sha256": _line_sha256(line),
            }
        elif not line:
            anomalies += 1
            event = CcCedictAnomaly(
                line_number, "EMPTY_DATA_LINE", _line_sha256(line)).to_dict()
            event["kind"] = "anomaly"
        else:
            try:
                entry = parse_cc_cedict_entry(line, line_number=line_number)
            except CcCedictAdapterError:
                anomalies += 1
                event = CcCedictAnomaly(
                    line_number,
                    "MALFORMED_ENTRY",
                    _line_sha256(line),
                ).to_dict()
                event["kind"] = "anomaly"
            else:
                entries += 1
                event = entry.to_dict()
                event["kind"] = "entry"
        digest.update(canonical_json_line(event))
    metadata = _metadata(headers)
    values = dict(metadata)
    declared_text = values.get("entries")
    if declared_text is None or not declared_text.isdigit():
        raise CcCedictAdapterError("CEDICT header entries 非整数或缺失")
    declared = int(declared_text)
    if complete_snapshot and (anomalies or entries != declared):
        raise CcCedictAdapterError("CEDICT 完整快照记录数/异常与 header 不一致")
    if sha256_path(source) != expected:
        raise CcCedictAdapterError("CEDICT 读取期间压缩字节变化")
    return CcCedictScanReport(
        expected,
        source.stat().st_size,
        line_count,
        len(headers),
        entries,
        anomalies,
        declared,
        terminal_newline_present,
        metadata,
        digest.hexdigest(),
        audit_cc_cedict_license(metadata),
    )


def read_cc_cedict_sample(
        path: str | Path,
        ) -> tuple[CcCedictEntry, ...]:
    """读取许可允许的极小明文 sample，并拒绝任何坏行。"""
    sample = Path(path)
    try:
        payload = sample.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise CcCedictAdapterError("CEDICT sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise CcCedictAdapterError("CEDICT sample 换行非法")
    headers: list[str] = []
    entries: list[CcCedictEntry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#"):
            if entries:
                raise CcCedictAdapterError("CEDICT sample header 必须位于 data 前")
            headers.append(line)
        else:
            entries.append(parse_cc_cedict_entry(
                line, line_number=line_number))
    audit = audit_cc_cedict_license(_metadata(headers))
    if audit.status != "CONFLICT":
        raise CcCedictAdapterError("当前 sample 未复现冻结许可冲突")
    if not entries:
        raise CcCedictAdapterError("CEDICT sample 缺少 data row")
    return tuple(entries)


__all__ = [
    "ADAPTER_VERSION",
    "CcCedictAdapterError",
    "CcCedictAnomaly",
    "CcCedictEntry",
    "CcCedictLicenseAudit",
    "CcCedictScanReport",
    "EXPECTED_LICENSE_ID",
    "LICENSE_EVIDENCE_URL",
    "OBSERVED_LICENSE_ID",
    "OFFICIAL_URL",
    "PARSER_VERSION",
    "SNAPSHOT_ID",
    "SOURCE_KEY",
    "audit_cc_cedict_license",
    "parse_cc_cedict_entry",
    "read_cc_cedict_sample",
    "scan_cc_cedict_gzip",
]
