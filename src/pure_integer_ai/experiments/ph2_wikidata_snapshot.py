"""Wikidata bounded fixed-revision 多实体 snapshot manifest。"""
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
from pure_integer_ai.experiments.ph2_wikidata_adapter import (
    WikidataAdapterError,
    WikidataEntityScanReport,
    scan_wikidata_entity_file,
)
from pure_integer_ai.experiments.ph2_wikidata_allowlist import (
    LICENSE_ID,
    SOURCE_KEY,
    WikidataPropertyRule,
    WikidataRevisionAllowlist,
    wikidata_entity_url,
)


FORMAT_VERSION = 1
ADAPTER_VERSION = 1
PARSER_VERSION = 1
LICENSE_EVIDENCE_URL = "https://www.wikidata.org/wiki/Wikidata:Licensing"
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z"
)


class WikidataSnapshotError(RuntimeError):
    """Wikidata HTTP capture、raw identity 或 manifest 不一致。"""


def _text(value: Any, *, where: str) -> str:
    """要求非空无首尾空白文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise WikidataSnapshotError(f"{where} 必须是非空文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    """要求正严格整数。"""
    if type(value) is not int or value <= 0:
        raise WikidataSnapshotError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    """要求非负严格整数。"""
    if type(value) is not int or value < 0:
        raise WikidataSnapshotError(f"{where} 必须是非负严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求小写 SHA-256 文本。"""
    text = _text(value, where=where).lower()
    if (len(text) != 64
            or any(character not in "0123456789abcdef" for character in text)):
        raise WikidataSnapshotError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求可迁移且不能逃逸 root 的 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise WikidataSnapshotError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _sha256_path(path: Path) -> str:
    """以固定块大小流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WikidataHttpCapture:
    """只公开安全稳定字段并绑定完整原始响应头 hash。"""

    status_code: int
    response_url: str
    response_date_gmt: str
    last_modified_gmt: str
    content_type: str
    response_headers_sha256: str
    captured_at_utc: str

    def __post_init__(self) -> None:
        if self.status_code != 200:
            raise WikidataSnapshotError("Wikidata HTTP status 必须为 200")
        if not self.response_url.startswith(
                "https://www.wikidata.org/wiki/Special:EntityData/"):
            raise WikidataSnapshotError("Wikidata response URL 非官方 EntityData")
        for name, value in (
                ("response_date_gmt", self.response_date_gmt),
                ("last_modified_gmt", self.last_modified_gmt),
                ("content_type", self.content_type)):
            _text(value, where=f"WikidataHttpCapture.{name}")
        if (not self.response_date_gmt.endswith(" GMT")
                or not self.last_modified_gmt.endswith(" GMT")):
            raise WikidataSnapshotError("Wikidata HTTP 日期必须为 GMT")
        if self.content_type.casefold() != "application/json; charset=utf-8":
            raise WikidataSnapshotError("Wikidata content-type 非 JSON UTF-8")
        object.__setattr__(
            self,
            "response_headers_sha256",
            _sha256(
                self.response_headers_sha256,
                where="WikidataHttpCapture.response_headers_sha256",
            ),
        )
        if not isinstance(self.captured_at_utc, str) or not _UTC_RE.fullmatch(
                self.captured_at_utc):
            raise WikidataSnapshotError("Wikidata captured_at_utc 非 UTC ISO-8601")

    def to_dict(self) -> dict[str, Any]:
        """导出不含 cookie、client IP 或其他敏感 header 的 object。"""
        return {
            "captured_at_utc": self.captured_at_utc,
            "content_type": self.content_type,
            "last_modified_gmt": self.last_modified_gmt,
            "response_date_gmt": self.response_date_gmt,
            "response_headers_sha256": self.response_headers_sha256,
            "response_url": self.response_url,
            "status_code": self.status_code,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WikidataHttpCapture":
        """从 manifest 恢复安全 HTTP capture。"""
        return cls(
            value["status_code"],
            str(value["response_url"]),
            str(value["response_date_gmt"]),
            str(value["last_modified_gmt"]),
            str(value["content_type"]),
            str(value["response_headers_sha256"]),
            str(value["captured_at_utc"]),
        )


def read_wikidata_http_capture(
        path: str | Path,
        *,
        response_url: str,
        captured_at_utc: str,
        ) -> WikidataHttpCapture:
    """读取 curl headers，只选择安全字段并保留完整 header hash。"""
    header_path = Path(path)
    try:
        payload = header_path.read_bytes()
        text = payload.decode("iso-8859-1")
    except (OSError, UnicodeError) as error:
        raise WikidataSnapshotError("Wikidata headers 无法读取") from error
    blocks = [
        block for block in text.replace("\r\n", "\n").split("\n\n")
        if block.strip()
    ]
    if not blocks:
        raise WikidataSnapshotError("Wikidata headers 为空")
    lines = [line for line in blocks[-1].split("\n") if line]
    if not lines or not lines[0].startswith("HTTP/"):
        raise WikidataSnapshotError("Wikidata status line 缺失")
    status_parts = lines[0].split(" ", 2)
    if len(status_parts) < 2 or not status_parts[1].isdigit():
        raise WikidataSnapshotError("Wikidata status line 非法")
    selected: dict[str, str] = {}
    selected_names = {"date", "last-modified", "content-type"}
    for line in lines[1:]:
        if ":" not in line:
            raise WikidataSnapshotError("Wikidata header 行非法")
        name, raw_value = line.split(":", 1)
        normalized = name.strip().casefold()
        if normalized not in selected_names:
            continue
        if normalized in selected:
            raise WikidataSnapshotError("Wikidata 安全 header 重复")
        selected[normalized] = raw_value.strip()
    if set(selected) != selected_names:
        raise WikidataSnapshotError("Wikidata 安全 header 缺失")
    return WikidataHttpCapture(
        int(status_parts[1]),
        response_url,
        selected["date"],
        selected["last-modified"],
        selected["content-type"],
        _sha256_path(header_path),
        captured_at_utc,
    )


@dataclass(frozen=True)
class WikidataAcquisitionFile:
    """构建 manifest 时使用的 raw/header 相对路径与请求时刻。"""

    qid: str
    revision: int
    raw_relative_path: str
    headers_relative_path: str
    captured_at_utc: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"Q[1-9][0-9]*", self.qid):
            raise WikidataSnapshotError("acquisition QID 非法")
        _positive(self.revision, where="acquisition revision")
        object.__setattr__(
            self,
            "raw_relative_path",
            _relative_path(self.raw_relative_path, where="acquisition raw path"),
        )
        object.__setattr__(
            self,
            "headers_relative_path",
            _relative_path(
                self.headers_relative_path, where="acquisition headers path"),
        )
        if not _UTC_RE.fullmatch(self.captured_at_utc):
            raise WikidataSnapshotError("acquisition UTC 非法")


@dataclass(frozen=True)
class WikidataEntitySnapshot:
    """一个 QID 的固定 revision、raw identity、split 与 parser evidence。"""

    qid: str
    revision: int
    split: str
    cluster_id: str
    purpose_keys: tuple[str, ...]
    raw_relative_path: str
    raw_sha256: str
    raw_size_bytes: int
    http: WikidataHttpCapture
    parser_report: WikidataEntityScanReport

    def __post_init__(self) -> None:
        if not re.fullmatch(r"Q[1-9][0-9]*", self.qid):
            raise WikidataSnapshotError("entity snapshot QID 非法")
        _positive(self.revision, where="entity snapshot revision")
        if self.split not in {"train", "dev", "held_out"}:
            raise WikidataSnapshotError("entity snapshot split 非法")
        _text(self.cluster_id, where="entity snapshot cluster")
        if (not isinstance(self.purpose_keys, tuple) or not self.purpose_keys
                or tuple(sorted(set(self.purpose_keys))) != self.purpose_keys):
            raise WikidataSnapshotError("entity snapshot purpose_keys 非法")
        object.__setattr__(
            self,
            "raw_relative_path",
            _relative_path(
                self.raw_relative_path, where="entity snapshot raw path"),
        )
        object.__setattr__(
            self,
            "raw_sha256",
            _sha256(self.raw_sha256, where="entity snapshot raw SHA-256"),
        )
        _positive(self.raw_size_bytes, where="entity snapshot raw size")
        if not isinstance(self.http, WikidataHttpCapture):
            raise WikidataSnapshotError("entity snapshot HTTP 类型非法")
        if self.http.response_url != wikidata_entity_url(
                self.qid, revision=self.revision):
            raise WikidataSnapshotError("entity snapshot URL/revision 不一致")
        if not isinstance(self.parser_report, WikidataEntityScanReport):
            raise WikidataSnapshotError("entity snapshot parser report 类型非法")
        report = self.parser_report
        if (report.qid != self.qid or report.revision != self.revision
                or report.raw_sha256 != self.raw_sha256
                or report.raw_size_bytes != self.raw_size_bytes):
            raise WikidataSnapshotError("entity snapshot parser/raw identity 不一致")

    def to_dict(self) -> dict[str, Any]:
        """导出单实体 snapshot object。"""
        return {
            "cluster_id": self.cluster_id,
            "http": self.http.to_dict(),
            "parser_report": self.parser_report.to_dict(),
            "purpose_keys": list(self.purpose_keys),
            "qid": self.qid,
            "raw_relative_path": self.raw_relative_path,
            "raw_sha256": self.raw_sha256,
            "raw_size_bytes": self.raw_size_bytes,
            "revision": self.revision,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WikidataEntitySnapshot":
        """从 manifest 恢复单实体 snapshot。"""
        return cls(
            str(value["qid"]),
            value["revision"],
            str(value["split"]),
            str(value["cluster_id"]),
            tuple(str(item) for item in value["purpose_keys"]),
            str(value["raw_relative_path"]),
            str(value["raw_sha256"]),
            value["raw_size_bytes"],
            WikidataHttpCapture.from_dict(value["http"]),
            WikidataEntityScanReport.from_dict(value["parser_report"]),
        )


@dataclass(frozen=True)
class WikidataRevisionSnapshotManifest:
    """冻结 allowlist v2 对应的 11 个 fixed-revision EntityData 文件。"""

    format_version: int
    source_key: str
    snapshot_id: str
    allowlist_revision: int
    allowlist_sha256: str
    adapter_version: int
    parser_version: int
    license_id: str
    license_evidence_url: str
    attribution: str
    redistribution_policy: str
    entities: tuple[WikidataEntitySnapshot, ...]
    aggregate_report: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise WikidataSnapshotError("Wikidata manifest format_version 非法")
        if self.source_key != SOURCE_KEY:
            raise WikidataSnapshotError("Wikidata manifest source_key 非法")
        _text(self.snapshot_id, where="Wikidata manifest snapshot_id")
        _positive(self.allowlist_revision, where="allowlist revision")
        object.__setattr__(
            self,
            "allowlist_sha256",
            _sha256(self.allowlist_sha256, where="allowlist SHA-256"),
        )
        if (self.adapter_version != ADAPTER_VERSION
                or self.parser_version != PARSER_VERSION):
            raise WikidataSnapshotError("Wikidata adapter/parser version 非法")
        if self.license_id != LICENSE_ID:
            raise WikidataSnapshotError("Wikidata manifest license 非法")
        if self.license_evidence_url != LICENSE_EVIDENCE_URL:
            raise WikidataSnapshotError("Wikidata license evidence URL 非法")
        _text(self.attribution, where="Wikidata attribution")
        if self.redistribution_policy != "PUBLIC":
            raise WikidataSnapshotError("Wikidata redistribution policy 非法")
        if not isinstance(self.entities, tuple) or not self.entities:
            raise WikidataSnapshotError("Wikidata manifest entities 不能为空")
        object.__setattr__(
            self,
            "entities",
            tuple(sorted(self.entities, key=lambda item: int(item.qid[1:]))),
        )
        qids = [item.qid for item in self.entities]
        if len(qids) != len(set(qids)):
            raise WikidataSnapshotError("Wikidata manifest QID 重复")
        cluster_splits: dict[str, str] = {}
        for item in self.entities:
            prior = cluster_splits.setdefault(item.cluster_id, item.split)
            if prior != item.split:
                raise WikidataSnapshotError("Wikidata manifest cluster 跨 split")
        if not isinstance(self.aggregate_report, CanonicalJsonObject):
            raise WikidataSnapshotError("Wikidata aggregate report 类型非法")

    def to_dict(self) -> dict[str, Any]:
        """导出多实体 snapshot 规范 object。"""
        return {
            "adapter_version": self.adapter_version,
            "aggregate_report": self.aggregate_report.to_value(),
            "allowlist_revision": self.allowlist_revision,
            "allowlist_sha256": self.allowlist_sha256,
            "attribution": self.attribution,
            "entities": [item.to_dict() for item in self.entities],
            "format_version": self.format_version,
            "license_evidence_url": self.license_evidence_url,
            "license_id": self.license_id,
            "parser_version": self.parser_version,
            "redistribution_policy": self.redistribution_policy,
            "snapshot_id": self.snapshot_id,
            "source_key": self.source_key,
        }

    def canonical_bytes(self) -> bytes:
        """返回唯一换行结尾的规范 manifest 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回 manifest 规范 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WikidataRevisionSnapshotManifest":
        """从规范 object 恢复多实体 snapshot。"""
        return cls(
            value["format_version"],
            str(value["source_key"]),
            str(value["snapshot_id"]),
            value["allowlist_revision"],
            str(value["allowlist_sha256"]),
            value["adapter_version"],
            value["parser_version"],
            str(value["license_id"]),
            str(value["license_evidence_url"]),
            str(value["attribution"]),
            str(value["redistribution_policy"]),
            tuple(WikidataEntitySnapshot.from_dict(item)
                  for item in value["entities"]),
            CanonicalJsonObject.from_value(dict(value["aggregate_report"])),
        )


def _resolve_under(root: Path, relative_path: str) -> Path:
    """把安全相对路径解析到 root 内并拒绝逃逸。"""
    path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    if not path.is_relative_to(root):
        raise WikidataSnapshotError("Wikidata raw path 逃逸 root")
    return path


def _merge_counts(target: dict[str, int], values: dict[str, Any]) -> None:
    """合并 aggregate 中的纯整数计数。"""
    for key, value in values.items():
        _nonnegative(value, where=f"aggregate count {key}")
        target[key] = target.get(key, 0) + value


def _aggregate_reports(
        entities: tuple[WikidataEntitySnapshot, ...],
        ) -> CanonicalJsonObject:
    """按 QID 顺序合并 parser 统计并形成独立 aggregate event hash。"""
    digest = hashlib.sha256()
    totals = {
        "anomaly_count": 0,
        "qualifier_snak_count": 0,
        "reference_count": 0,
        "reference_snak_count": 0,
        "selected_statement_count": 0,
        "statement_count": 0,
        "valid_statement_count": 0,
    }
    rank_counts: dict[str, int] = {}
    snaktype_counts: dict[str, int] = {}
    datatype_counts: dict[str, int] = {}
    selected_property_counts: dict[str, int] = {}
    for entity in entities:
        report = entity.parser_report
        digest.update(canonical_json_line({
            "event_kind": "WIKIDATA_ENTITY_REPORT_V1",
            "event_sha256": report.event_sha256,
            "qid": report.qid,
            "revision": report.revision,
        }))
        for key in totals:
            totals[key] += getattr(report, key)
        _merge_counts(rank_counts, report.rank_counts.to_value())
        _merge_counts(snaktype_counts, report.snaktype_counts.to_value())
        _merge_counts(datatype_counts, report.datatype_counts.to_value())
        _merge_counts(
            selected_property_counts,
            report.selected_property_counts.to_value(),
        )
    return CanonicalJsonObject.from_value({
        **totals,
        "datatype_counts": datatype_counts,
        "entity_count": len(entities),
        "event_sha256": digest.hexdigest(),
        "rank_counts": rank_counts,
        "selected_property_counts": selected_property_counts,
        "snaktype_counts": snaktype_counts,
    })


def build_wikidata_revision_snapshot(
        allowlist: WikidataRevisionAllowlist,
        *,
        raw_root: str | Path,
        acquisition_files: tuple[WikidataAcquisitionFile, ...],
        snapshot_id: str,
        ) -> WikidataRevisionSnapshotManifest:
    """从实际 raw/header 构建 allowlist 绑定的 fixed-revision manifest。"""
    if not isinstance(allowlist, WikidataRevisionAllowlist):
        raise WikidataSnapshotError("Wikidata allowlist 类型非法")
    root = Path(raw_root).resolve()
    files_by_qid = {item.qid: item for item in acquisition_files}
    if (len(files_by_qid) != len(acquisition_files)
            or set(files_by_qid) != {item.qid for item in allowlist.entities}):
        raise WikidataSnapshotError("acquisition QID 集与 allowlist 不一致")
    entities: list[WikidataEntitySnapshot] = []
    for seed in allowlist.entities:
        acquisition = files_by_qid[seed.qid]
        raw_path = _resolve_under(root, acquisition.raw_relative_path)
        headers_path = _resolve_under(root, acquisition.headers_relative_path)
        if not raw_path.is_file() or not headers_path.is_file():
            raise WikidataSnapshotError("Wikidata raw/header 文件缺失")
        response_url = wikidata_entity_url(
            seed.qid, revision=acquisition.revision)
        http = read_wikidata_http_capture(
            headers_path,
            response_url=response_url,
            captured_at_utc=acquisition.captured_at_utc,
        )
        try:
            report = scan_wikidata_entity_file(
                raw_path,
                expected_qid=seed.qid,
                expected_revision=acquisition.revision,
                property_rules=allowlist.properties,
            )
        except WikidataAdapterError as error:
            raise WikidataSnapshotError("Wikidata parser 扫描失败") from error
        entities.append(WikidataEntitySnapshot(
            seed.qid,
            acquisition.revision,
            seed.split,
            seed.cluster_id,
            seed.purpose_keys,
            acquisition.raw_relative_path,
            report.raw_sha256,
            report.raw_size_bytes,
            http,
            report,
        ))
    frozen_entities = tuple(entities)
    return WikidataRevisionSnapshotManifest(
        FORMAT_VERSION,
        SOURCE_KEY,
        snapshot_id,
        allowlist.allowlist_revision,
        allowlist.sha256(),
        ADAPTER_VERSION,
        PARSER_VERSION,
        LICENSE_ID,
        LICENSE_EVIDENCE_URL,
        "Wikidata contributors; fixed EntityData revisions retained",
        "PUBLIC",
        frozen_entities,
        _aggregate_reports(frozen_entities),
    )


def read_wikidata_revision_snapshot(
        path: str | Path,
        ) -> WikidataRevisionSnapshotManifest:
    """严格读取规范 Wikidata snapshot manifest。"""
    try:
        payload = Path(path).read_bytes()
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = WikidataRevisionSnapshotManifest.from_dict(value)
    except Exception as error:
        raise WikidataSnapshotError("Wikidata snapshot manifest 无法恢复") from error
    if (not payload.endswith(b"\n") or payload.endswith(b"\n\n")
            or manifest.canonical_bytes() != payload):
        raise WikidataSnapshotError("Wikidata snapshot manifest 规范字节漂移")
    return manifest


def write_wikidata_revision_snapshot(
        manifest: WikidataRevisionSnapshotManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布 snapshot manifest，禁止原地覆盖不同内容。"""
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise WikidataSnapshotError("Wikidata snapshot 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise WikidataSnapshotError("Wikidata snapshot 无法发布") from error
    return target


def verify_wikidata_revision_snapshot(
        manifest: WikidataRevisionSnapshotManifest,
        *,
        raw_root: str | Path,
        property_rules: tuple[WikidataPropertyRule, ...],
        ) -> None:
    """逐实体重算 raw hash 与 parser report，拒绝只信 manifest。"""
    root = Path(raw_root).resolve()
    for entity in manifest.entities:
        raw_path = _resolve_under(root, entity.raw_relative_path)
        if (not raw_path.is_file()
                or raw_path.stat().st_size != entity.raw_size_bytes
                or _sha256_path(raw_path) != entity.raw_sha256):
            raise WikidataSnapshotError("Wikidata raw identity 漂移")
        try:
            report = scan_wikidata_entity_file(
                raw_path,
                expected_qid=entity.qid,
                expected_revision=entity.revision,
                property_rules=property_rules,
                expected_sha256=entity.raw_sha256,
            )
        except WikidataAdapterError as error:
            raise WikidataSnapshotError("Wikidata raw parser verify 失败") from error
        if report != entity.parser_report:
            raise WikidataSnapshotError("Wikidata parser report 漂移")


__all__ = [
    "ADAPTER_VERSION",
    "FORMAT_VERSION",
    "LICENSE_EVIDENCE_URL",
    "PARSER_VERSION",
    "WikidataAcquisitionFile",
    "WikidataEntitySnapshot",
    "WikidataHttpCapture",
    "WikidataRevisionSnapshotManifest",
    "WikidataSnapshotError",
    "build_wikidata_revision_snapshot",
    "read_wikidata_http_capture",
    "read_wikidata_revision_snapshot",
    "verify_wikidata_revision_snapshot",
    "write_wikidata_revision_snapshot",
]
