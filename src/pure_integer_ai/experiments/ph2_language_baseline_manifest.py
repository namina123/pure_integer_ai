"""LG/LC/MD/GG 的 D-03 前基线 manifest 与文件级公开审计边界。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Pattern

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
    CapabilityCourseCoverageLedger,
    LanguageCapabilityCoverageLedger,
    LanguageCoverageContractError,
    VerifierCapabilityRegistry,
)


FORMAT_VERSION = 1
PUBLIC_GATE_STATUSES = ("BLOCKED", "CLEAR")
AUDIT_ROW_STATES = ("MISSING", "PARTIAL", "PRESENT")
GG03_EXIT_STATES = ("COURSE_FROZEN", "MISSING")

MD_BASELINE_KEYS = (
    "FIXED_TOP_K",
    "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP",
    "RECENCY_HOT_ONLY",
    "TYPED_FIXED_RING",
)

MD_SAMPLE_GROUP_KEYS = (
    "ACCESS_BUDGET_GROUNDING_DISTINCT",
    "COLD_CORRECT_HOT_DISTRACTOR",
    "EXACT_MEMORY_STRUCTURE_DISTANCE_HELD_OUT",
    "GENERATION_UNAUTHORIZED_ACTIVATION",
    "HOT_COLD_CONFLICT",
    "HOT_CORRECT",
    "INDEXED_DIRECT_ROUTE",
    "LOCAL_REVISION_BIT_IDENTICAL",
    "MULTICENTER_OBLIGATIONS",
    "NO_ANSWER_UNKNOWN",
)

MD_HARD_INVARIANT_KEYS = (
    "BUDGET_NOT_UNKNOWN",
    "HARD_VETO_NOT_OFFSET",
    "HELD_OUT_COMBINATION_NOT_TRAIN",
    "HOLDOUT_EVALUATOR_HOST_WRITES_ZERO",
    "UNAUTHORIZED_GENERATION_ZERO",
    "UNRELATED_REVISION_CHANGES_ZERO",
)

GG_COURSE_FAMILY_KEYS = (
    "CONTEXT_ADDRESSEE_CONDITION",
    "ELLIPSIS_EXPLICIT_CONTRAST",
    "LEXICAL_STRUCTURE_RECOMBINATION",
    "MULTIPLE_LEGAL_SURFACES",
    "MULTIPROPOSITION_ORDER_REVISION",
    "REFERENCE_RECOVERABILITY",
    "SEMANTIC_DRIFT_NEGATIVES",
    "SOURCE_UNCERTAINTY_QUALIFICATION",
    "STANCE_CONTENT_WORDING_SEPARATION",
    "USE_OUTCOME_NOT_TEMPLATE",
)

GG_COMBINATION_AXIS_KEYS = (
    "ADDRESSEE_RECOVERABILITY_FAMILY",
    "COMMUNICATIVE_DISCOURSE_STRATEGY",
    "CONTEXT_CONDITION_FAMILY",
    "DIRECTION",
    "LEXICAL_REALIZATION_FAMILY",
    "OBLIGATION_KIND",
    "PARSER_COURSE_VERSION",
    "PROPOSITION_LOGIC_SHAPE",
    "SOURCE_CLUSTER",
    "STRUCTURE_CONCEPT_FAMILY",
)

GG_STAGE_KEYS = (
    "D02E3",
    "G00",
    "G01",
    "G02",
    "G03",
    "G04",
    "G05",
    "W02",
    "W03",
    "W04",
    "W05",
    "W06",
    "W07",
    "W08",
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class LanguageBaselineManifestError(RuntimeError):
    """语言基线 manifest、公开扫描或预注册证据不一致。"""


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise LanguageBaselineManifestError(
            f"{where} 必须是非空无首尾空白文本")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise LanguageBaselineManifestError(f"{where} 必须是非负严格整数")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise LanguageBaselineManifestError(f"{where} 必须是正严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise LanguageBaselineManifestError(f"{where} 必须为 0/1")
    return value


def _sha1(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if _SHA1_RE.fullmatch(text) is None:
        raise LanguageBaselineManifestError(f"{where} 必须是 SHA-1")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise LanguageBaselineManifestError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise LanguageBaselineManifestError(
            f"{where} 必须是安全 POSIX 相对路径")
    return text


def _strict_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise LanguageBaselineManifestError(f"{where} 必须是 tuple")
    result = tuple(_text(item, where=where) for item in value)
    if not allow_empty and not result:
        raise LanguageBaselineManifestError(f"{where} 不得为空")
    if tuple(sorted(set(result))) != result:
        raise LanguageBaselineManifestError(f"{where} 必须排序且去重")
    return result


def _require_keys(
        value: Any,
        expected: set[str] | tuple[str, ...],
        *,
        where: str,
        ) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise LanguageBaselineManifestError(f"{where} 字段不精确")
    return value


def _hash_path(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
    except OSError as error:
        raise LanguageBaselineManifestError("公开候选文件无法读取") from error
    return size, digest.hexdigest()


def _resolve_under(root: Path, relative_path: str) -> Path:
    path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    if not path.is_relative_to(root):
        raise LanguageBaselineManifestError("公开候选路径逃逸仓库")
    return path


@dataclass(frozen=True)
class PublicFileIdentity:
    """最终待提交集合中的一个可迁移文件身份。"""

    relative_path: str
    category: str
    size_bytes: int
    sha256: str
    text_file: int
    line_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path, where="public file path"))
        _text(self.category, where="public file category")
        _nonnegative(self.size_bytes, where="public file size")
        object.__setattr__(self, "sha256", _sha256(
            self.sha256, where="public file SHA-256"))
        _flag(self.text_file, where="public file text_file")
        _nonnegative(self.line_count, where="public file line_count")
        if not self.text_file and self.line_count != 0:
            raise LanguageBaselineManifestError("binary file line_count 必须为 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "line_count": self.line_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "text_file": self.text_file,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PublicFileIdentity":
        _require_keys(value, {
            "category", "line_count", "relative_path", "sha256",
            "size_bytes", "text_file",
        }, where="PublicFileIdentity")
        return cls(
            str(value["relative_path"]), str(value["category"]),
            value["size_bytes"], str(value["sha256"]),
            value["text_file"], value["line_count"],
        )


@dataclass(frozen=True)
class PublicPatternFinding:
    """不复制命中文字，只保存规则键、位置和整行 hash。"""

    rule_key: str
    relative_path: str
    line_number: int
    line_sha256: str

    def __post_init__(self) -> None:
        _text(self.rule_key, where="pattern rule_key")
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path, where="pattern finding path"))
        _positive(self.line_number, where="pattern finding line_number")
        object.__setattr__(self, "line_sha256", _sha256(
            self.line_sha256, where="pattern finding line SHA-256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "line_sha256": self.line_sha256,
            "relative_path": self.relative_path,
            "rule_key": self.rule_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PublicPatternFinding":
        _require_keys(value, {
            "line_number", "line_sha256", "relative_path", "rule_key",
        }, where="PublicPatternFinding")
        return cls(
            str(value["rule_key"]), str(value["relative_path"]),
            value["line_number"], str(value["line_sha256"]),
        )


@dataclass(frozen=True)
class PublicGateBaseline:
    """LG-00 的扫描口径、当前阻断和最终重扫义务。"""

    scope_file_count: int
    scanned_text_file_count: int
    binary_paths: tuple[str, ...]
    unreadable_paths: tuple[str, ...]
    legacy_rule_keys: tuple[str, ...]
    legacy_findings: tuple[PublicPatternFinding, ...]
    legacy_status: str
    secret_rule_keys: tuple[str, ...]
    secret_findings: tuple[PublicPatternFinding, ...]
    secret_status: str
    final_rescan_required: int
    public_release_allowed: int

    def __post_init__(self) -> None:
        _nonnegative(self.scope_file_count, where="public gate scope count")
        _nonnegative(
            self.scanned_text_file_count, where="public gate scanned count")
        for name in (
                "binary_paths", "unreadable_paths", "legacy_rule_keys",
                "secret_rule_keys"):
            values = _strict_tuple(
                getattr(self, name), where=f"public gate {name}",
                allow_empty=name in {"binary_paths", "unreadable_paths"})
            if name.endswith("paths"):
                values = tuple(_relative_path(
                    item, where=f"public gate {name}") for item in values)
            object.__setattr__(self, name, values)
        for name in ("legacy_findings", "secret_findings"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or not all(isinstance(item, PublicPatternFinding)
                               for item in values)):
                raise LanguageBaselineManifestError(
                    f"public gate {name} 类型非法")
            object.__setattr__(self, name, tuple(sorted(
                values,
                key=lambda item: (
                    item.relative_path, item.line_number, item.rule_key),
            )))
        if self.legacy_status not in PUBLIC_GATE_STATUSES:
            raise LanguageBaselineManifestError("legacy status 非法")
        if self.secret_status not in PUBLIC_GATE_STATUSES:
            raise LanguageBaselineManifestError("secret status 非法")
        _flag(self.final_rescan_required, where="final_rescan_required")
        _flag(self.public_release_allowed, where="public_release_allowed")
        expected_legacy = "BLOCKED" if self.legacy_findings else "CLEAR"
        expected_secret = "BLOCKED" if (
            self.secret_findings or self.binary_paths or self.unreadable_paths
        ) else "CLEAR"
        if (self.legacy_status != expected_legacy
                or self.secret_status != expected_secret):
            raise LanguageBaselineManifestError("public gate verdict 不诚实")
        if (self.scanned_text_file_count + len(self.binary_paths)
                + len(self.unreadable_paths) != self.scope_file_count):
            raise LanguageBaselineManifestError("public gate 扫描范围未闭合")
        if self.final_rescan_required != 1 or self.public_release_allowed != 0:
            raise LanguageBaselineManifestError(
                "baseline 不得冒充最终公开门通过")

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary_paths": list(self.binary_paths),
            "final_rescan_required": self.final_rescan_required,
            "legacy_findings": [
                item.to_dict() for item in self.legacy_findings],
            "legacy_rule_keys": list(self.legacy_rule_keys),
            "legacy_status": self.legacy_status,
            "public_release_allowed": self.public_release_allowed,
            "scanned_text_file_count": self.scanned_text_file_count,
            "scope_file_count": self.scope_file_count,
            "secret_findings": [
                item.to_dict() for item in self.secret_findings],
            "secret_rule_keys": list(self.secret_rule_keys),
            "secret_status": self.secret_status,
            "unreadable_paths": list(self.unreadable_paths),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PublicGateBaseline":
        _require_keys(value, {
            "binary_paths", "final_rescan_required", "legacy_findings",
            "legacy_rule_keys", "legacy_status", "public_release_allowed",
            "scanned_text_file_count", "scope_file_count", "secret_findings",
            "secret_rule_keys", "secret_status", "unreadable_paths",
        }, where="PublicGateBaseline")
        return cls(
            value["scope_file_count"], value["scanned_text_file_count"],
            tuple(str(item) for item in value["binary_paths"]),
            tuple(str(item) for item in value["unreadable_paths"]),
            tuple(str(item) for item in value["legacy_rule_keys"]),
            tuple(PublicPatternFinding.from_dict(item)
                  for item in value["legacy_findings"]),
            str(value["legacy_status"]),
            tuple(str(item) for item in value["secret_rule_keys"]),
            tuple(PublicPatternFinding.from_dict(item)
                  for item in value["secret_findings"]),
            str(value["secret_status"]), value["final_rescan_required"],
            value["public_release_allowed"],
        )


@dataclass(frozen=True)
class MDProbePreRegistration:
    """MD-00 的结果不可见预注册。"""

    preregistration_version: str
    decision_state: str
    results_observed: int
    baseline_keys: tuple[str, ...]
    sample_group_keys: tuple[str, ...]
    metric_contracts: CanonicalJsonObject
    hard_invariant_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    threshold_policy: CanonicalJsonObject
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.preregistration_version, where="MD prereg version")
        if self.decision_state != "PRE_REGISTERED":
            raise LanguageBaselineManifestError("MD-00 决断状态非法")
        if self.results_observed != 0:
            raise LanguageBaselineManifestError("MD-00 不得先看结果")
        for name, exact in (
                ("baseline_keys", MD_BASELINE_KEYS),
                ("sample_group_keys", MD_SAMPLE_GROUP_KEYS),
                ("hard_invariant_keys", MD_HARD_INVARIANT_KEYS)):
            values = _strict_tuple(getattr(self, name), where=f"MD {name}")
            if values != exact:
                raise LanguageBaselineManifestError(f"MD {name} 未列全")
        object.__setattr__(self, "ablation_keys", _strict_tuple(
            self.ablation_keys, where="MD ablation_keys"))
        object.__setattr__(self, "evidence_refs", _strict_tuple(
            self.evidence_refs, where="MD evidence_refs"))
        metrics = self.metric_contracts.to_value()
        _require_keys(metrics, {
            "audit", "quality", "resource",
        }, where="MD metric_contracts")
        for group, values in metrics.items():
            if (not isinstance(values, list) or not values
                    or tuple(sorted(set(values))) != tuple(values)):
                raise LanguageBaselineManifestError(
                    f"MD metric group {group} 非法")
        threshold = self.threshold_policy.to_value()
        _require_keys(threshold, {
            "comparison_order", "decision_rule", "freeze_before_run",
            "hard_zero_policy", "primary_candidate", "resource_ceiling_rule",
            "theater_rule",
        }, where="MD threshold_policy")
        if (threshold["freeze_before_run"] != 1
                or threshold["hard_zero_policy"] != 1):
            raise LanguageBaselineManifestError("MD threshold 未冻结")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_keys": list(self.ablation_keys),
            "baseline_keys": list(self.baseline_keys),
            "decision_state": self.decision_state,
            "evidence_refs": list(self.evidence_refs),
            "hard_invariant_keys": list(self.hard_invariant_keys),
            "metric_contracts": self.metric_contracts.to_value(),
            "preregistration_version": self.preregistration_version,
            "results_observed": self.results_observed,
            "sample_group_keys": list(self.sample_group_keys),
            "threshold_policy": self.threshold_policy.to_value(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MDProbePreRegistration":
        _require_keys(value, {
            "ablation_keys", "baseline_keys", "decision_state",
            "evidence_refs", "hard_invariant_keys", "metric_contracts",
            "preregistration_version", "results_observed",
            "sample_group_keys", "threshold_policy",
        }, where="MDProbePreRegistration")
        return cls(
            str(value["preregistration_version"]),
            str(value["decision_state"]), value["results_observed"],
            tuple(str(item) for item in value["baseline_keys"]),
            tuple(str(item) for item in value["sample_group_keys"]),
            CanonicalJsonObject.from_value(dict(value["metric_contracts"])),
            tuple(str(item) for item in value["hard_invariant_keys"]),
            tuple(str(item) for item in value["ablation_keys"]),
            CanonicalJsonObject.from_value(dict(value["threshold_policy"])),
            tuple(str(item) for item in value["evidence_refs"]),
        )


@dataclass(frozen=True)
class GenerationCoverageAuditRow:
    """GG-00 中一个课程族、组合轴或阶段的当前事实。"""

    row_key: str
    state: str
    gap_code: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.row_key, where="GG row_key")
        if self.state not in AUDIT_ROW_STATES:
            raise LanguageBaselineManifestError("GG row state 非法")
        _text(self.gap_code, where="GG gap_code")
        object.__setattr__(self, "evidence_refs", _strict_tuple(
            self.evidence_refs, where="GG evidence_refs"))
        if self.state == "PRESENT" and self.gap_code != "NONE":
            raise LanguageBaselineManifestError("GG PRESENT 不得带 gap")
        if self.state != "PRESENT" and self.gap_code == "NONE":
            raise LanguageBaselineManifestError("GG 缺口必须有 gap_code")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_refs": list(self.evidence_refs),
            "gap_code": self.gap_code,
            "row_key": self.row_key,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationCoverageAuditRow":
        _require_keys(value, {
            "evidence_refs", "gap_code", "row_key", "state",
        }, where="GenerationCoverageAuditRow")
        return cls(
            str(value["row_key"]), str(value["state"]),
            str(value["gap_code"]),
            tuple(str(item) for item in value["evidence_refs"]),
        )


@dataclass(frozen=True)
class GenerationCoverageAudit:
    """GG-00 的课程族、完整组合轴和阶段配对缺口矩阵。"""

    audit_version: str
    gg03_exit_state: str
    course_family_rows: tuple[GenerationCoverageAuditRow, ...]
    combination_axis_rows: tuple[GenerationCoverageAuditRow, ...]
    stage_rows: tuple[GenerationCoverageAuditRow, ...]

    def __post_init__(self) -> None:
        _text(self.audit_version, where="GG audit_version")
        if self.gg03_exit_state not in GG03_EXIT_STATES:
            raise LanguageBaselineManifestError("GG-03 exit state 非法")
        for name, exact in (
                ("course_family_rows", GG_COURSE_FAMILY_KEYS),
                ("combination_axis_rows", GG_COMBINATION_AXIS_KEYS),
                ("stage_rows", GG_STAGE_KEYS)):
            rows = getattr(self, name)
            if (not isinstance(rows, tuple)
                    or not all(isinstance(item, GenerationCoverageAuditRow)
                               for item in rows)):
                raise LanguageBaselineManifestError(f"GG {name} 类型非法")
            rows = tuple(sorted(rows, key=lambda item: item.row_key))
            object.__setattr__(self, name, rows)
            if tuple(item.row_key for item in rows) != exact:
                raise LanguageBaselineManifestError(f"GG {name} 未列全")
        all_rows = (
            *self.course_family_rows,
            *self.combination_axis_rows,
            *self.stage_rows,
        )
        all_present = all(item.state == "PRESENT" for item in all_rows)
        if self.gg03_exit_state == "COURSE_FROZEN" and not all_present:
            raise LanguageBaselineManifestError(
                "GG-03 COURSE_FROZEN 不得隐藏课程、组合轴或阶段缺口")
        if self.gg03_exit_state == "MISSING" and all_present:
            raise LanguageBaselineManifestError(
                "GG-03 MISSING 与完整 PRESENT 证据冲突")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": self.audit_version,
            "combination_axis_rows": [
                item.to_dict() for item in self.combination_axis_rows],
            "course_family_rows": [
                item.to_dict() for item in self.course_family_rows],
            "gg03_exit_state": self.gg03_exit_state,
            "stage_rows": [item.to_dict() for item in self.stage_rows],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationCoverageAudit":
        _require_keys(value, {
            "audit_version", "combination_axis_rows", "course_family_rows",
            "gg03_exit_state", "stage_rows",
        }, where="GenerationCoverageAudit")
        return cls(
            str(value["audit_version"]), str(value["gg03_exit_state"]),
            tuple(GenerationCoverageAuditRow.from_dict(item)
                  for item in value["course_family_rows"]),
            tuple(GenerationCoverageAuditRow.from_dict(item)
                  for item in value["combination_axis_rows"]),
            tuple(GenerationCoverageAuditRow.from_dict(item)
                  for item in value["stage_rows"]),
        )


@dataclass(frozen=True)
class LanguageBaselineManifest:
    """切片 2 的 LG-00、LC-00/11/12、MD-00 和 GG-00 合并证据。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    head_sha1: str
    origin_master_sha1: str
    tracked_change_count: int
    staged_change_count: int
    untracked_file_count: int
    inventory_exclusions: tuple[str, ...]
    file_inventory: tuple[PublicFileIdentity, ...]
    paper_files: tuple[PublicFileIdentity, ...]
    public_gate: PublicGateBaseline
    capability_ledger: LanguageCapabilityCoverageLedger
    verifier_registry: VerifierCapabilityRegistry
    course_coverage_ledger: CapabilityCourseCoverageLedger
    md00_preregistration: MDProbePreRegistration
    gg00_audit: GenerationCoverageAudit
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise LanguageBaselineManifestError("baseline format_version 非法")
        _text(self.artifact_version, where="baseline artifact_version")
        if self.artifact_status != "BASELINE_FROZEN":
            raise LanguageBaselineManifestError("baseline artifact_status 非法")
        object.__setattr__(self, "head_sha1", _sha1(
            self.head_sha1, where="baseline HEAD"))
        object.__setattr__(self, "origin_master_sha1", _sha1(
            self.origin_master_sha1, where="baseline origin/master"))
        if self.head_sha1 != self.origin_master_sha1:
            raise LanguageBaselineManifestError("HEAD 与 origin/master 不一致")
        for name in (
                "tracked_change_count", "staged_change_count",
                "untracked_file_count"):
            _nonnegative(getattr(self, name), where=f"baseline {name}")
        if self.tracked_change_count or self.staged_change_count:
            raise LanguageBaselineManifestError("baseline tracked/staged 非零")
        exclusions = tuple(_relative_path(
            item, where="baseline inventory exclusion")
            for item in _strict_tuple(
                self.inventory_exclusions,
                where="baseline inventory_exclusions"))
        object.__setattr__(self, "inventory_exclusions", exclusions)
        for name in ("file_inventory", "paper_files"):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values
                    or not all(isinstance(item, PublicFileIdentity)
                               for item in values)):
                raise LanguageBaselineManifestError(f"baseline {name} 非法")
            values = tuple(sorted(values, key=lambda item: item.relative_path))
            object.__setattr__(self, name, values)
            paths = tuple(item.relative_path for item in values)
            if len(paths) != len(set(paths)):
                raise LanguageBaselineManifestError(f"baseline {name} 路径重复")
        if (len(self.file_inventory) + len(self.inventory_exclusions)
                != self.untracked_file_count):
            raise LanguageBaselineManifestError("baseline 未跟踪清单计数不闭合")
        if self.public_gate.scope_file_count != len(self.file_inventory):
            raise LanguageBaselineManifestError("public gate 与清单范围不一致")
        if not isinstance(self.public_gate, PublicGateBaseline):
            raise LanguageBaselineManifestError("public gate 类型非法")
        if not isinstance(
                self.capability_ledger, LanguageCapabilityCoverageLedger):
            raise LanguageBaselineManifestError("capability ledger 类型非法")
        if not isinstance(self.verifier_registry, VerifierCapabilityRegistry):
            raise LanguageBaselineManifestError("verifier registry 类型非法")
        if not isinstance(
                self.course_coverage_ledger, CapabilityCourseCoverageLedger):
            raise LanguageBaselineManifestError("course ledger 类型非法")
        if not isinstance(self.md00_preregistration, MDProbePreRegistration):
            raise LanguageBaselineManifestError("MD prereg 类型非法")
        if not isinstance(self.gg00_audit, GenerationCoverageAudit):
            raise LanguageBaselineManifestError("GG audit 类型非法")
        verifier_keys = {
            item.verifier_key for item in self.verifier_registry.records
        }
        for entry in self.capability_ledger.entries:
            if not set(entry.verifier_keys).issubset(verifier_keys):
                raise LanguageBaselineManifestError(
                    "capability 引用了未注册 verifier")
        if ({item.capability_key for item in self.capability_ledger.entries}
                != set(CAPABILITY_KEYS)):
            raise LanguageBaselineManifestError("capability 前沿不完整")
        execution = self.execution_state.to_value()
        _require_keys(execution, {
            "companion_writes", "core_learning_writes", "d03_published",
            "formal_training_runs", "mastered_claims", "memory_learning_writes",
            "readiness_claims", "teacher_calls", "use_learning_writes",
            "w01_started",
        }, where="baseline execution_state")
        if any(value != 0 for value in execution.values()):
            raise LanguageBaselineManifestError("baseline 出现禁用执行状态")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "capability_ledger": self.capability_ledger.to_dict(),
            "course_coverage_ledger": self.course_coverage_ledger.to_dict(),
            "execution_state": self.execution_state.to_value(),
            "file_inventory": [item.to_dict() for item in self.file_inventory],
            "format_version": self.format_version,
            "gg00_audit": self.gg00_audit.to_dict(),
            "head_sha1": self.head_sha1,
            "inventory_exclusions": list(self.inventory_exclusions),
            "md00_preregistration": self.md00_preregistration.to_dict(),
            "origin_master_sha1": self.origin_master_sha1,
            "paper_files": [item.to_dict() for item in self.paper_files],
            "public_gate": self.public_gate.to_dict(),
            "staged_change_count": self.staged_change_count,
            "tracked_change_count": self.tracked_change_count,
            "untracked_file_count": self.untracked_file_count,
            "verifier_registry": self.verifier_registry.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LanguageBaselineManifest":
        _require_keys(value, {
            "artifact_status", "artifact_version", "capability_ledger",
            "course_coverage_ledger", "execution_state", "file_inventory",
            "format_version", "gg00_audit", "head_sha1",
            "inventory_exclusions", "md00_preregistration",
            "origin_master_sha1", "paper_files", "public_gate",
            "staged_change_count", "tracked_change_count",
            "untracked_file_count", "verifier_registry",
        }, where="LanguageBaselineManifest")
        try:
            return cls(
                value["format_version"], str(value["artifact_version"]),
                str(value["artifact_status"]), str(value["head_sha1"]),
                str(value["origin_master_sha1"]),
                value["tracked_change_count"], value["staged_change_count"],
                value["untracked_file_count"],
                tuple(str(item) for item in value["inventory_exclusions"]),
                tuple(PublicFileIdentity.from_dict(item)
                      for item in value["file_inventory"]),
                tuple(PublicFileIdentity.from_dict(item)
                      for item in value["paper_files"]),
                PublicGateBaseline.from_dict(value["public_gate"]),
                LanguageCapabilityCoverageLedger.from_dict(
                    value["capability_ledger"]),
                VerifierCapabilityRegistry.from_dict(
                    value["verifier_registry"]),
                CapabilityCourseCoverageLedger.from_dict(
                    value["course_coverage_ledger"]),
                MDProbePreRegistration.from_dict(value["md00_preregistration"]),
                GenerationCoverageAudit.from_dict(value["gg00_audit"]),
                CanonicalJsonObject.from_value(dict(value["execution_state"])),
            )
        except LanguageCoverageContractError as error:
            raise LanguageBaselineManifestError(
                "language coverage nested contract damaged") from error


def inventory_public_files(
        repo_root: str | Path,
        relative_paths: tuple[str, ...],
        ) -> tuple[PublicFileIdentity, ...]:
    """按调用方冻结的 Git 路径集合生成稳定文件级清单。"""
    root = Path(repo_root).resolve()
    paths = tuple(sorted(_relative_path(
        item, where="inventory input path") for item in relative_paths))
    if len(paths) != len(set(paths)):
        raise LanguageBaselineManifestError("inventory input path 重复")
    result: list[PublicFileIdentity] = []
    for relative_path in paths:
        path = _resolve_under(root, relative_path)
        if not path.is_file():
            raise LanguageBaselineManifestError("inventory 文件缺失")
        payload = path.read_bytes()
        size, digest = _hash_path(path)
        try:
            text = payload.decode("utf-8")
        except UnicodeError:
            text_file = 0
            line_count = 0
        else:
            text_file = 1
            line_count = len(text.splitlines())
        top = PurePosixPath(relative_path).parts[0]
        category = {
            "data": "DATA",
            "src": "SOURCE",
            "tests": "TEST",
        }.get(top, "OTHER")
        result.append(PublicFileIdentity(
            relative_path, category, size, digest, text_file, line_count))
    return tuple(result)


def scan_public_patterns(
        repo_root: str | Path,
        inventory: tuple[PublicFileIdentity, ...],
        rules: tuple[tuple[str, Pattern[str]], ...],
        ) -> tuple[tuple[PublicPatternFinding, ...], tuple[str, ...], tuple[str, ...]]:
    """扫描文本但不把敏感命中文字复制进结果。"""
    root = Path(repo_root).resolve()
    findings: list[PublicPatternFinding] = []
    binary: list[str] = []
    unreadable: list[str] = []
    rule_keys = tuple(key for key, _ in rules)
    if tuple(sorted(set(rule_keys))) != rule_keys:
        raise LanguageBaselineManifestError("scan rule keys 必须排序去重")
    for item in inventory:
        if not item.text_file:
            binary.append(item.relative_path)
            continue
        path = _resolve_under(root, item.relative_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            unreadable.append(item.relative_path)
            continue
        for line_number, line in enumerate(lines, start=1):
            for rule_key, pattern in rules:
                if pattern.search(line) is None:
                    continue
                findings.append(PublicPatternFinding(
                    rule_key,
                    item.relative_path,
                    line_number,
                    hashlib.sha256(line.encode("utf-8")).hexdigest(),
                ))
    return tuple(findings), tuple(sorted(binary)), tuple(sorted(unreadable))


def read_language_baseline_manifest(
        path: str | Path,
        ) -> LanguageBaselineManifest:
    """严格回读规范语言基线 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise LanguageBaselineManifestError("baseline newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = LanguageBaselineManifest.from_dict(value)
    except LanguageBaselineManifestError:
        raise
    except Exception as error:
        raise LanguageBaselineManifestError("baseline manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise LanguageBaselineManifestError("baseline manifest 非规范字节")
    return manifest


def verify_language_baseline_files(
        manifest: LanguageBaselineManifest,
        *,
        repo_root: str | Path,
        ) -> None:
    """逐字节核当前候选、论文和自引用排除文件，拒绝清单漂移。"""
    if not isinstance(manifest, LanguageBaselineManifest):
        raise LanguageBaselineManifestError("baseline manifest 类型非法")
    root = Path(repo_root).resolve()
    current = inventory_public_files(
        root, tuple(item.relative_path for item in manifest.file_inventory))
    if current != manifest.file_inventory:
        raise LanguageBaselineManifestError("baseline public file identity 漂移")
    papers = inventory_public_files(
        root, tuple(item.relative_path for item in manifest.paper_files))
    if papers != manifest.paper_files:
        raise LanguageBaselineManifestError("baseline paper identity 漂移")
    for relative_path in manifest.inventory_exclusions:
        if not _resolve_under(root, relative_path).is_file():
            raise LanguageBaselineManifestError("baseline exclusion 文件缺失")


def write_language_baseline_manifest(
        manifest: LanguageBaselineManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布语言基线，禁止原地覆盖版本。"""
    if not isinstance(manifest, LanguageBaselineManifest):
        raise LanguageBaselineManifestError("baseline manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise LanguageBaselineManifestError(
                "language baseline 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise LanguageBaselineManifestError("language baseline 无法发布") from error
    return target


__all__ = [
    "AUDIT_ROW_STATES",
    "GG03_EXIT_STATES",
    "FORMAT_VERSION",
    "GG_COMBINATION_AXIS_KEYS",
    "GG_COURSE_FAMILY_KEYS",
    "GG_STAGE_KEYS",
    "GenerationCoverageAudit",
    "GenerationCoverageAuditRow",
    "LanguageBaselineManifest",
    "LanguageBaselineManifestError",
    "MD_BASELINE_KEYS",
    "MD_HARD_INVARIANT_KEYS",
    "MD_SAMPLE_GROUP_KEYS",
    "MDProbePreRegistration",
    "PUBLIC_GATE_STATUSES",
    "PublicFileIdentity",
    "PublicGateBaseline",
    "PublicPatternFinding",
    "inventory_public_files",
    "read_language_baseline_manifest",
    "scan_public_patterns",
    "verify_language_baseline_files",
    "write_language_baseline_manifest",
]
