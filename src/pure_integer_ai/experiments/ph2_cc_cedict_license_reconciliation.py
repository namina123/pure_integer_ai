"""CC-CEDICT 官方许可分歧、历史 blocker 与 LC-12 缺口封存。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_language_baseline_catalog import (
    build_course_coverage_ledger,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CapabilityCourseCoverageLedger,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import (
    read_raw_snapshot_manifest,
    sha256_path,
)


FORMAT_VERSION = 1
ARTIFACT_VERSION = "CC-CEDICT-20260725-license-reconciliation-v1"
SOURCE_KEY = "CC_CEDICT_20260725"
HISTORICAL_MANIFEST_PATH = (
    "data/ph2/manifests/cc_cedict_20260725.raw_snapshot.json")
HISTORICAL_MANIFEST_SHA256 = (
    "7c148fd121d90aab616b8fb804631cd92f3b1d522c4776ff2cd1a3c3036886fc")
RECONCILIATION_MANIFEST_PATH = (
    "data/ph2/manifests/"
    "cc_cedict_20260725.license_reconciliation_v1.json")

_EVIDENCE_KEYS = (
    "CC_CEDICT_PROJECT_WIKI_GENERAL",
    "MDBG_CURRENT_DOWNLOAD_PAGE",
    "MDBG_SNAPSHOT_RAW_HEADER",
)
_STAGE_KEYS = ("W-02", "W-03")
_LICENSE_URLS = {
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_HEX = frozenset("0123456789abcdef")
_LC12_GAPS = {
    "MORPHOLOGY_WORD_FORM": "W02_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL",
    "MULTIWORD_CONSTRUCTION": "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL",
    "RAW_TEXT_NOISE": "W02_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL",
    "SOURCE_UNCERTAINTY_REALITY": (
        "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL"),
}


class CcCedictLicenseReconciliationError(RuntimeError):
    """许可对账 artifact 非规范、历史证据漂移或被错误放行。"""


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value or "\x00" in value):
        raise CcCedictLicenseReconciliationError(f"{where} 非法")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise CcCedictLicenseReconciliationError(f"{where} 必须为 0/1")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise CcCedictLicenseReconciliationError(f"{where} 必须为正整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise CcCedictLicenseReconciliationError(f"{where} 非 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where).replace("\\", "/")
    path = PurePosixPath(text)
    if (path.is_absolute() or not path.parts or text != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise CcCedictLicenseReconciliationError(f"{where} 非安全相对路径")
    return text


def _strict_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, str) for item in value)):
        raise CcCedictLicenseReconciliationError(f"{where} 必须为非空 tuple[str]")
    normalized = tuple(_text(item, where=where) for item in value)
    if normalized != tuple(sorted(set(normalized))):
        raise CcCedictLicenseReconciliationError(f"{where} 必须排序且无重复")
    return normalized


def _require_keys(value: dict[str, Any], expected: set[str], *, where: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise CcCedictLicenseReconciliationError(f"{where} 字段集合非法")


@dataclass(frozen=True)
class CcCedictOfficialLicenseEvidence:
    """一份官方页面或 snapshot header 的作用域化许可声明。"""

    evidence_key: str
    authority: str
    evidence_scope: str
    source_url: str
    observed_license_id: str
    license_url: str
    captured_utc: str
    payload_size_bytes: int
    payload_sha256: str
    evidence_locator: str

    def __post_init__(self) -> None:
        for name in (
                "evidence_key", "authority", "evidence_scope", "source_url",
                "observed_license_id", "license_url", "captured_utc",
                "evidence_locator"):
            object.__setattr__(self, name, _text(
                getattr(self, name), where=f"evidence {name}"))
        if not self.source_url.startswith("https://"):
            raise CcCedictLicenseReconciliationError(
                "evidence source_url 必须为 HTTPS")
        expected_url = _LICENSE_URLS.get(self.observed_license_id)
        if expected_url != self.license_url:
            raise CcCedictLicenseReconciliationError(
                "evidence license id/URL 不一致")
        if _UTC.fullmatch(self.captured_utc) is None:
            raise CcCedictLicenseReconciliationError(
                "evidence captured_utc 非规范 UTC")
        _positive(self.payload_size_bytes, where="evidence payload_size_bytes")
        object.__setattr__(self, "payload_sha256", _sha256(
            self.payload_sha256, where="evidence payload_sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "captured_utc": self.captured_utc,
            "evidence_key": self.evidence_key,
            "evidence_locator": self.evidence_locator,
            "evidence_scope": self.evidence_scope,
            "license_url": self.license_url,
            "observed_license_id": self.observed_license_id,
            "payload_sha256": self.payload_sha256,
            "payload_size_bytes": self.payload_size_bytes,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "CcCedictOfficialLicenseEvidence":
        _require_keys(value, {
            "authority", "captured_utc", "evidence_key", "evidence_locator",
            "evidence_scope", "license_url", "observed_license_id",
            "payload_sha256", "payload_size_bytes", "source_url",
        }, where="CcCedictOfficialLicenseEvidence")
        return cls(
            str(value["evidence_key"]), str(value["authority"]),
            str(value["evidence_scope"]), str(value["source_url"]),
            str(value["observed_license_id"]), str(value["license_url"]),
            str(value["captured_utc"]), value["payload_size_bytes"],
            str(value["payload_sha256"]), str(value["evidence_locator"]),
        )


@dataclass(frozen=True)
class CcCedictAlternativeCoverage:
    """许可阻断后某 W 阶段的合法替代来源与独立性缺口。"""

    stage_key: str
    status: str
    source_keys: tuple[str, ...]
    license_ids: tuple[str, ...]
    course_gap_codes: tuple[str, ...]
    independence_limitations: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.stage_key not in _STAGE_KEYS:
            raise CcCedictLicenseReconciliationError("alternative stage 非法")
        if self.status != "PARTIAL_ALTERNATIVE":
            raise CcCedictLicenseReconciliationError(
                "alternative 不得冒充完整覆盖")
        for name in (
                "source_keys", "license_ids", "course_gap_codes",
                "independence_limitations", "evidence_refs"):
            object.__setattr__(self, name, _strict_tuple(
                getattr(self, name), where=f"alternative {name}"))
        if "CC_CEDICT_20260725" in self.source_keys:
            raise CcCedictLicenseReconciliationError(
                "BLOCKED CC-CEDICT 不得列为替代来源")

    def to_dict(self) -> dict[str, Any]:
        return {
            "course_gap_codes": list(self.course_gap_codes),
            "evidence_refs": list(self.evidence_refs),
            "independence_limitations": list(self.independence_limitations),
            "license_ids": list(self.license_ids),
            "source_keys": list(self.source_keys),
            "stage_key": self.stage_key,
            "status": self.status,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "CcCedictAlternativeCoverage":
        _require_keys(value, {
            "course_gap_codes", "evidence_refs", "independence_limitations",
            "license_ids", "source_keys", "stage_key", "status",
        }, where="CcCedictAlternativeCoverage")
        return cls(
            str(value["stage_key"]), str(value["status"]),
            tuple(str(item) for item in value["source_keys"]),
            tuple(str(item) for item in value["license_ids"]),
            tuple(str(item) for item in value["course_gap_codes"]),
            tuple(str(item) for item in value["independence_limitations"]),
            tuple(str(item) for item in value["evidence_refs"]),
        )


def build_cc_cedict_lc12_supplement() -> CapabilityCourseCoverageLedger:
    """在 LC-12 全量 DAG 上登记 W-02/W-03 的来源许可缺口。"""
    base = build_course_coverage_ledger()
    records = []
    for record in base.records:
        gap = _LC12_GAPS.get(record.capability_key)
        if gap is None:
            records.append(record)
            continue
        records.append(replace(
            record,
            external_prerequisites=tuple(sorted({
                *record.external_prerequisites,
                gap,
            })),
            evidence_refs=tuple(sorted({
                *record.evidence_refs,
                RECONCILIATION_MANIFEST_PATH,
            })),
        ))
    return CapabilityCourseCoverageLedger(
        FORMAT_VERSION,
        "LC-12-source-gap-supplement-cc-cedict-v1",
        tuple(records),
    )


@dataclass(frozen=True)
class CcCedictLicenseReconciliationManifest:
    """当前官方证据分叉下的 fail-closed 决断。"""

    format_version: int
    artifact_version: str
    source_key: str
    historical_manifest_relative_path: str
    historical_manifest_sha256: str
    historical_license_verdict: str
    historical_blocker_code: str
    official_evidence: tuple[CcCedictOfficialLicenseEvidence, ...]
    official_evidence_consistent: int
    license_verdict: str
    blocker_code: str
    redistribution_policy: str
    release_eligible: int
    public_source_pack_emitted: int
    attribution: str
    alternative_coverage: tuple[CcCedictAlternativeCoverage, ...]
    lc12_supplement: CapabilityCourseCoverageLedger
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise CcCedictLicenseReconciliationError("format_version 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise CcCedictLicenseReconciliationError("artifact_version 非法")
        if self.source_key != SOURCE_KEY:
            raise CcCedictLicenseReconciliationError("source_key 非法")
        path = _relative_path(
            self.historical_manifest_relative_path,
            where="historical_manifest_relative_path")
        if path != HISTORICAL_MANIFEST_PATH:
            raise CcCedictLicenseReconciliationError("historical manifest 路径非法")
        object.__setattr__(self, "historical_manifest_sha256", _sha256(
            self.historical_manifest_sha256,
            where="historical_manifest_sha256"))
        if self.historical_manifest_sha256 != HISTORICAL_MANIFEST_SHA256:
            raise CcCedictLicenseReconciliationError("historical manifest hash 漂移")
        if (self.historical_license_verdict != "BLOCKED"
                or self.historical_blocker_code != "LICENSE_PARTITION_MISMATCH"):
            raise CcCedictLicenseReconciliationError(
                "历史 LICENSE_PARTITION_MISMATCH 必须保持 BLOCKED")
        if (not isinstance(self.official_evidence, tuple)
                or not all(isinstance(item, CcCedictOfficialLicenseEvidence)
                           for item in self.official_evidence)):
            raise CcCedictLicenseReconciliationError("official evidence 类型非法")
        object.__setattr__(self, "official_evidence", tuple(sorted(
            self.official_evidence, key=lambda item: item.evidence_key)))
        if tuple(item.evidence_key for item in self.official_evidence) != _EVIDENCE_KEYS:
            raise CcCedictLicenseReconciliationError("official evidence 必须三项齐全")
        observed = {item.observed_license_id for item in self.official_evidence}
        if observed != {"CC-BY-SA-3.0", "CC-BY-SA-4.0"}:
            raise CcCedictLicenseReconciliationError("官方许可分歧证据不完整")
        _flag(self.official_evidence_consistent,
              where="official_evidence_consistent")
        if self.official_evidence_consistent != 0:
            raise CcCedictLicenseReconciliationError("分歧证据不得标成一致")
        if (self.license_verdict != "BLOCKED"
                or self.blocker_code != "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE"
                or self.redistribution_policy != "BLOCKED"):
            raise CcCedictLicenseReconciliationError("许可分歧必须 fail-closed")
        if (_flag(self.release_eligible, where="release_eligible") != 0
                or _flag(self.public_source_pack_emitted,
                         where="public_source_pack_emitted") != 0):
            raise CcCedictLicenseReconciliationError(
                "许可分歧不得 release 或产出 source pack")
        _text(self.attribution, where="attribution")
        if (not isinstance(self.alternative_coverage, tuple)
                or not all(isinstance(item, CcCedictAlternativeCoverage)
                           for item in self.alternative_coverage)):
            raise CcCedictLicenseReconciliationError("alternative coverage 类型非法")
        object.__setattr__(self, "alternative_coverage", tuple(sorted(
            self.alternative_coverage, key=lambda item: item.stage_key)))
        if tuple(item.stage_key for item in self.alternative_coverage) != _STAGE_KEYS:
            raise CcCedictLicenseReconciliationError("W-02/W-03 替代评估必须齐全")
        if not isinstance(self.lc12_supplement, CapabilityCourseCoverageLedger):
            raise CcCedictLicenseReconciliationError("LC-12 supplement 类型非法")
        records = {
            item.capability_key: item for item in self.lc12_supplement.records
        }
        for capability_key, gap in _LC12_GAPS.items():
            record = records[capability_key]
            if (gap not in record.external_prerequisites
                    or RECONCILIATION_MANIFEST_PATH not in record.evidence_refs):
                raise CcCedictLicenseReconciliationError("LC-12 来源缺口未登记")
        frozen = {
            item.capability_key for item in self.lc12_supplement.records
            if item.exit_state == "COURSE_FROZEN"
        }
        allowed_frozen = {
            item for item in frozen if item not in _LC12_GAPS
        } | {"RAW_TEXT_NOISE"}
        morphology = records["MORPHOLOGY_WORD_FORM"]
        if morphology.exit_state == "COURSE_FROZEN":
            if ("LC02_MORPHOLOGY_COURSE_V1"
                    not in morphology.external_prerequisites
                    or "data/ph2/manifests/lc02_morphology_course_v1.json"
                    not in morphology.evidence_refs):
                raise CcCedictLicenseReconciliationError(
                    "独立 LC-02 课程冻结缺直接证据")
            allowed_frozen.add("MORPHOLOGY_WORD_FORM")
        multiword = records["MULTIWORD_CONSTRUCTION"]
        if multiword.exit_state == "COURSE_FROZEN":
            if ("LC03_CONSTRUCTION_COURSE_V1"
                    not in multiword.external_prerequisites
                    or "data/ph2/manifests/lc03_construction_course_v1.json"
                    not in multiword.evidence_refs):
                raise CcCedictLicenseReconciliationError(
                    "独立 LC-03 课程冻结缺直接证据")
            allowed_frozen.add("MULTIWORD_CONSTRUCTION")
        if frozen - allowed_frozen:
            raise CcCedictLicenseReconciliationError(
                "许可对账不得冒充 CC-CEDICT 相关课程已冻结")
        state = self.execution_state.to_value()
        expected_state = {
            "companion_writes": 0,
            "core_learning_writes": 0,
            "d03_published": 0,
            "formal_training_runs": 0,
            "mastered_claims": 0,
            "memory_learning_writes": 0,
            "readiness_claims": 0,
            "teacher_calls": 0,
            "use_learning_writes": 0,
            "w01_started": 0,
        }
        if state != expected_state:
            raise CcCedictLicenseReconciliationError("execution_state 非零或缺项")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alternative_coverage": [
                item.to_dict() for item in self.alternative_coverage],
            "artifact_version": self.artifact_version,
            "attribution": self.attribution,
            "blocker_code": self.blocker_code,
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "historical_blocker_code": self.historical_blocker_code,
            "historical_license_verdict": self.historical_license_verdict,
            "historical_manifest_relative_path": (
                self.historical_manifest_relative_path),
            "historical_manifest_sha256": self.historical_manifest_sha256,
            "lc12_supplement": self.lc12_supplement.to_dict(),
            "license_verdict": self.license_verdict,
            "official_evidence": [item.to_dict() for item in self.official_evidence],
            "official_evidence_consistent": self.official_evidence_consistent,
            "public_source_pack_emitted": self.public_source_pack_emitted,
            "redistribution_policy": self.redistribution_policy,
            "release_eligible": self.release_eligible,
            "source_key": self.source_key,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]
            ) -> "CcCedictLicenseReconciliationManifest":
        _require_keys(value, {
            "alternative_coverage", "artifact_version", "attribution",
            "blocker_code", "execution_state", "format_version",
            "historical_blocker_code", "historical_license_verdict",
            "historical_manifest_relative_path", "historical_manifest_sha256",
            "lc12_supplement", "license_verdict", "official_evidence",
            "official_evidence_consistent", "public_source_pack_emitted",
            "redistribution_policy", "release_eligible", "source_key",
        }, where="CcCedictLicenseReconciliationManifest")
        return cls(
            value["format_version"], str(value["artifact_version"]),
            str(value["source_key"]),
            str(value["historical_manifest_relative_path"]),
            str(value["historical_manifest_sha256"]),
            str(value["historical_license_verdict"]),
            str(value["historical_blocker_code"]),
            tuple(CcCedictOfficialLicenseEvidence.from_dict(item)
                  for item in value["official_evidence"]),
            value["official_evidence_consistent"],
            str(value["license_verdict"]), str(value["blocker_code"]),
            str(value["redistribution_policy"]), value["release_eligible"],
            value["public_source_pack_emitted"], str(value["attribution"]),
            tuple(CcCedictAlternativeCoverage.from_dict(item)
                  for item in value["alternative_coverage"]),
            CapabilityCourseCoverageLedger.from_dict(
                dict(value["lc12_supplement"])),
            CanonicalJsonObject.from_value(dict(value["execution_state"])),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_cc_cedict_license_reconciliation(
        ) -> CcCedictLicenseReconciliationManifest:
    """从切片 3 的直接官方证据构建当前 BLOCKED 决断。"""
    captured = "2026-07-26T15:28:28Z"
    evidence = (
        CcCedictOfficialLicenseEvidence(
            "CC_CEDICT_PROJECT_WIKI_GENERAL",
            "CC-CEDICT project wiki",
            "PROJECT_GENERAL_LICENSE_STATEMENT",
            "https://cc-cedict.org/wiki/_export/raw/start",
            "CC-BY-SA-3.0",
            _LICENSE_URLS["CC-BY-SA-3.0"],
            captured,
            2280,
            "9619670a47ea3b8aff446cdbde44cf354b7d061f3fd570d61e83c0014968da59",
            "raw wiki paragraph beginning CC-CEDICT is licensed under",
        ),
        CcCedictOfficialLicenseEvidence(
            "MDBG_CURRENT_DOWNLOAD_PAGE",
            "MDBG",
            "CURRENT_DOWNLOAD_LICENSE_STATEMENT",
            "https://www.mdbg.net/chinese/dictionary?page=cc-cedict",
            "CC-BY-SA-4.0",
            _LICENSE_URLS["CC-BY-SA-4.0"],
            captured,
            15197,
            "56f697cb55390035480602747590d97b5dfb275301e0c231d0d35301f6bdb07b",
            "HTML rel=license link adjacent to the current gzip download",
        ),
        CcCedictOfficialLicenseEvidence(
            "MDBG_SNAPSHOT_RAW_HEADER",
            "MDBG",
            "SNAPSHOT_FILE_HEADER",
            "https://www.mdbg.net/chinese/export/cedict/"
            "cedict_1_0_ts_utf-8_mdbg.txt.gz",
            "CC-BY-SA-4.0",
            _LICENSE_URLS["CC-BY-SA-4.0"],
            captured,
            3965460,
            "c745acaa8d549e6fd3a6cadadf5481c018eef0a0e3dbb2c704c3969c9f1685d3",
            "gzip header License paragraph and #! license metadata",
        ),
    )
    alternatives = (
        CcCedictAlternativeCoverage(
            "W-02",
            "PARTIAL_ALTERNATIVE",
            ("AUTHORED_CC0_V1", "UD_ZH_GSDSIMP_R2_18", "ZHWIKTIONARY_20260701"),
            ("CC-BY-SA-4.0", "CC0-1.0"),
            ("MORPHOLOGY_INVENTORY_SOURCE_GAP", "WORD_BOUNDARY_PACK_NOT_FROZEN"),
            (
                "CC_CEDICT_LEXICON_FAMILY_ABSENT",
                "UNIFIED_SOURCE_PACK_NOT_FROZEN",
                "WIKTIONARY_HELD_OUT_CANNOT_DOUBLE_AS_TRAIN",
            ),
            (
                "data/ph2/manifests/ud_zh_gsdsimp_r2_18.git_snapshot.json",
                "data/ph2/manifests/zhwiktionary_20260701.multistream_snapshot.json",
                "tests/test_d02_authored_primitive_course.py",
            ),
        ),
        CcCedictAlternativeCoverage(
            "W-03",
            "PARTIAL_ALTERNATIVE",
            (
                "AUTHORED_CC0_V1", "CONCEPTNET_5_7_0",
                "WIKIDATA_REVISION_V1", "ZHWIKTIONARY_20260701",
            ),
            ("CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"),
            ("LEXICAL_SENSE_PACK_NOT_FROZEN", "MULTIWORD_SOURCE_GAP"),
            (
                "ENGLISH_GLOSS_NOT_TRUTH",
                "NO_CROSS_SOURCE_PASS_YET",
                "SOURCE_RELATION_FAMILY_NOT_EQUIVALENT_TO_LEXICON",
                "UNIFIED_SOURCE_PACK_NOT_FROZEN",
            ),
            (
                "data/ph2/manifests/conceptnet_5_7_0.raw_snapshot.json",
                "data/ph2/manifests/wikidata_revision_v1.pinned_snapshot.json",
                "data/ph2/manifests/zhwiktionary_20260701.multistream_snapshot.json",
                "tests/test_d02_authored_sense_course.py",
            ),
        ),
    )
    return CcCedictLicenseReconciliationManifest(
        FORMAT_VERSION,
        ARTIFACT_VERSION,
        SOURCE_KEY,
        HISTORICAL_MANIFEST_PATH,
        HISTORICAL_MANIFEST_SHA256,
        "BLOCKED",
        "LICENSE_PARTITION_MISMATCH",
        evidence,
        0,
        "BLOCKED",
        "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE",
        "BLOCKED",
        0,
        0,
        "CC-CEDICT project and MDBG; redistribution remains blocked until "
        "the official license scope converges",
        alternatives,
        build_cc_cedict_lc12_supplement(),
        CanonicalJsonObject.from_value({
            "companion_writes": 0,
            "core_learning_writes": 0,
            "d03_published": 0,
            "formal_training_runs": 0,
            "mastered_claims": 0,
            "memory_learning_writes": 0,
            "readiness_claims": 0,
            "teacher_calls": 0,
            "use_learning_writes": 0,
            "w01_started": 0,
        }),
    )


def read_cc_cedict_license_reconciliation(
        path: str | Path,
        ) -> CcCedictLicenseReconciliationManifest:
    source = Path(path)
    try:
        payload = source.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CcCedictLicenseReconciliationError(
            "许可对账 manifest 无法读取") from error
    if not isinstance(value, dict):
        raise CcCedictLicenseReconciliationError("许可对账 manifest 顶层非对象")
    manifest = CcCedictLicenseReconciliationManifest.from_dict(value)
    if payload != manifest.canonical_bytes():
        raise CcCedictLicenseReconciliationError("许可对账 manifest 非规范字节")
    return manifest


def write_cc_cedict_license_reconciliation(
        manifest: CcCedictLicenseReconciliationManifest,
        path: str | Path,
        ) -> None:
    destination = Path(path)
    payload = manifest.canonical_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise CcCedictLicenseReconciliationError(
                "许可对账 manifest 已存在且内容不同")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise CcCedictLicenseReconciliationError(
            "许可对账 manifest 并发占用") from error


def verify_cc_cedict_license_reconciliation(
        manifest: CcCedictLicenseReconciliationManifest,
        repository_root: str | Path,
        ) -> None:
    root = Path(repository_root).resolve()
    historical = (root / manifest.historical_manifest_relative_path).resolve()
    try:
        historical.relative_to(root)
    except ValueError as error:
        raise CcCedictLicenseReconciliationError(
            "historical manifest 路径逃逸") from error
    if not historical.is_file():
        raise CcCedictLicenseReconciliationError("historical manifest 缺失")
    if sha256_path(historical) != manifest.historical_manifest_sha256:
        raise CcCedictLicenseReconciliationError("historical manifest hash 不一致")
    old = read_raw_snapshot_manifest(historical)
    if (old.source_key != SOURCE_KEY or old.license_status != "CONFLICT"
            or old.redistribution_policy != "BLOCKED"
            or old.release_eligible != 0
            or old.blocker_code != "LICENSE_PARTITION_MISMATCH"):
        raise CcCedictLicenseReconciliationError("历史 blocker 语义被改动")


__all__ = [
    "ARTIFACT_VERSION",
    "CcCedictAlternativeCoverage",
    "CcCedictLicenseReconciliationError",
    "CcCedictLicenseReconciliationManifest",
    "CcCedictOfficialLicenseEvidence",
    "FORMAT_VERSION",
    "HISTORICAL_MANIFEST_PATH",
    "HISTORICAL_MANIFEST_SHA256",
    "RECONCILIATION_MANIFEST_PATH",
    "SOURCE_KEY",
    "build_cc_cedict_lc12_supplement",
    "build_cc_cedict_license_reconciliation",
    "read_cc_cedict_license_reconciliation",
    "verify_cc_cedict_license_reconciliation",
    "write_cc_cedict_license_reconciliation",
]
