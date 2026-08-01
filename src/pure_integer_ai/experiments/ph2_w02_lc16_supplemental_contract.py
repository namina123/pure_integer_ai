"""W-02 LC-16 九载体补充资格的只读合同与公开 receipt 结构。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    DIRECTIONS,
    IN_SCOPE_CARRIER_KEYS,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    SAMPLE_KINDS,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_W02_LC16_SUPPLEMENTAL_QUALIFICATION"
ARTIFACT_VERSION = "PH2-W02-LC16-SUPPLEMENTAL-20260801-A"
ARTIFACT_STATUS = "W02_LC16_SUPPLEMENTAL_FROZEN_NOT_RUN"
MANIFEST_PATH = "data/ph2/manifests/w02_lc16_supplemental_v1.json"
OVERLAY_PATH = "data/ph2/manifests/d03_lc16_successor_overlay_v1.json"
OVERLAY_SHA256 = (
    "6cb9ab991ff41ecd87905f446ed5d75b2ad83e9d6f43124e2a69e15e7135083d")
W02_PARENT_RECEIPT_SHA256 = (
    "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb")

BOUNDARY_WITHDRAWAL = "W-02-BOUNDARY_WITHDRAWAL_BY_CARRIER"
MULTI_CANDIDATE = "W-02-MULTI_CANDIDATE_BOUNDARY_BY_CARRIER"
NEW_CONTENT_MORPHOLOGY = "W-02-NEW_CONTENT_MORPHOLOGY_BY_CARRIER"
OOV = "W-02-OOV_BY_CARRIER"
GENERATION_HARD_CONJUNCT = "W-02-LC16-GENERATION-HARD-CONJUNCT"
BEARING_DIMENSIONS = (
    BOUNDARY_WITHDRAWAL,
    MULTI_CANDIDATE,
    NEW_CONTENT_MORPHOLOGY,
    OOV,
)
EVALUATION_ORDER = (*BEARING_DIMENSIONS, GENERATION_HARD_CONJUNCT)
ABLATION_ORDER = tuple(f"{item}-ABLATION" for item in BEARING_DIMENSIONS)
STATUSES = ("PASS", "FAIL", "NE")
QUALIFICATION_STATUSES = (*STATUSES, "BLOCKED")
HOST_DIGEST_KEYS = ("core", "cursor", "logical", "memory", "use")
CASE_COUNT = len(IN_SCOPE_CARRIER_KEYS) * len(SAMPLE_KINDS)
DIRECTION_EVALUATION_COUNT = CASE_COUNT * len(DIRECTIONS)
MAX_WORKERS = 4
MAX_PAYLOAD_BYTES = 134217728
MAX_PAYLOAD_READS = 131072
MAX_LOGIC_OPERATIONS = 2000000

EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_STARTED": 0,
    "formal_training_runs": 0,
    "llm_calls": 0,
    "memory_learning_writes": 0,
    "receipt_published": 0,
    "runtime_observed": 0,
    "teacher_calls": 0,
    "w02_lc16_supplemental_qualified": 0,
}

_PRIVATE_KEYS = frozenset({
    "accepted_surfaces", "expected_payload", "expected_surface",
    "evaluator_label_payload", "raw_observation", "private_payload",
})


class W02Lc16SupplementalError(RuntimeError):
    """补充资格输入、合取结果或发布身份不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    """要求 object 字段集合与合同完全一致。"""
    if not isinstance(value, dict) or set(value) != expected:
        raise W02Lc16SupplementalError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    """校验非空、无首尾空白文本。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise W02Lc16SupplementalError(f"{where} 文本非法")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """校验小写 SHA-256 文本。"""
    checked = _text(value, where=where).lower()
    if len(checked) != 64 or any(item not in "0123456789abcdef" for item in checked):
        raise W02Lc16SupplementalError(f"{where} 必须是小写 SHA-256")
    return checked


def _nonnegative(value: Any, *, where: str) -> int:
    """校验不含 bool 的非负严格整数。"""
    if type(value) is not int or value < 0:
        raise W02Lc16SupplementalError(f"{where} 必须是非负严格整数")
    return value


def _positive(value: Any, *, where: str) -> int:
    """校验不含 bool 的正严格整数。"""
    checked = _nonnegative(value, where=where)
    if checked == 0:
        raise W02Lc16SupplementalError(f"{where} 必须为正")
    return checked


def _status(value: Any, *, where: str, blocked: bool = False) -> str:
    """校验公开状态枚举。"""
    allowed = QUALIFICATION_STATUSES if blocked else STATUSES
    if value not in allowed:
        raise W02Lc16SupplementalError(f"{where} 状态非法")
    return value


def _stable_digest(value: Any) -> str:
    """对不含 private 内容的证据结构计算规范摘要。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _dimension_status(value: "SupplementalDimensionResult") -> str:
    """按 fail/NE 优先规则推导一个维度状态。"""
    if value.fail_count:
        return "FAIL"
    if value.ne_count:
        return "NE"
    if value.pass_count == value.case_count:
        return "PASS"
    return "FAIL"


@dataclass(frozen=True, order=True)
class SupplementalDimensionResult:
    """一个补充资格维度的计数、状态和不可逆证据摘要。"""

    dimension_key: str
    status: str
    pass_count: int
    case_count: int
    fail_count: int
    ne_count: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.dimension_key not in EVALUATION_ORDER:
            raise W02Lc16SupplementalError("补充维度未预注册")
        _status(self.status, where=self.dimension_key)
        for name in ("pass_count", "case_count", "fail_count", "ne_count"):
            _nonnegative(getattr(self, name), where=f"{self.dimension_key}.{name}")
        if self.case_count != self.pass_count + self.fail_count + self.ne_count:
            raise W02Lc16SupplementalError("补充维度计数不闭合")
        if self.case_count <= 0 or self.status != _dimension_status(self):
            raise W02Lc16SupplementalError("补充维度状态与计数不一致")
        _sha256(self.evidence_sha256, where="补充维度 evidence")

    def to_dict(self) -> dict[str, Any]:
        """导出不含 private 内容的维度摘要。"""
        return {
            "case_count": self.case_count,
            "dimension_key": self.dimension_key,
            "evidence_sha256": self.evidence_sha256,
            "fail_count": self.fail_count,
            "ne_count": self.ne_count,
            "pass_count": self.pass_count,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SupplementalDimensionResult":
        """从精确公开 object 恢复维度摘要。"""
        raw = _exact(value, {
            "case_count", "dimension_key", "evidence_sha256", "fail_count",
            "ne_count", "pass_count", "status",
        }, where="SupplementalDimensionResult")
        return cls(
            str(raw["dimension_key"]), str(raw["status"]), raw["pass_count"],
            raw["case_count"], raw["fail_count"], raw["ne_count"],
            str(raw["evidence_sha256"]),
        )


@dataclass(frozen=True, order=True)
class SupplementalAblationResult:
    """一个正交消融的四维状态与生成硬合取状态。"""

    ablation_key: str
    targeted_dimension: str
    dimension_statuses: tuple[tuple[str, str], ...]
    generation_status: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.ablation_key not in ABLATION_ORDER:
            raise W02Lc16SupplementalError("补充消融未预注册")
        index = ABLATION_ORDER.index(self.ablation_key)
        if self.targeted_dimension != BEARING_DIMENSIONS[index]:
            raise W02Lc16SupplementalError("补充消融目标错位")
        if tuple(item[0] for item in self.dimension_statuses) != BEARING_DIMENSIONS:
            raise W02Lc16SupplementalError("补充消融维度顺序漂移")
        if any(
                not isinstance(item, tuple) or len(item) != 2
                or item[1] not in STATUSES
                for item in self.dimension_statuses):
            raise W02Lc16SupplementalError("补充消融状态非法")
        _status(self.generation_status, where="补充消融 generation")
        _sha256(self.evidence_sha256, where="补充消融 evidence")

    def to_dict(self) -> dict[str, Any]:
        """导出消融摘要。"""
        return {
            "ablation_key": self.ablation_key,
            "dimension_statuses": [
                {"dimension_key": key, "status": status}
                for key, status in self.dimension_statuses
            ],
            "evidence_sha256": self.evidence_sha256,
            "generation_status": self.generation_status,
            "targeted_dimension": self.targeted_dimension,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SupplementalAblationResult":
        """从精确公开 object 恢复消融摘要。"""
        raw = _exact(value, {
            "ablation_key", "dimension_statuses", "evidence_sha256",
            "generation_status", "targeted_dimension",
        }, where="SupplementalAblationResult")
        statuses = []
        for item in raw["dimension_statuses"]:
            nested = _exact(item, {"dimension_key", "status"}, where="ablation dimension")
            statuses.append((str(nested["dimension_key"]), str(nested["status"])))
        return cls(
            str(raw["ablation_key"]), str(raw["targeted_dimension"]),
            tuple(statuses), str(raw["generation_status"]),
            str(raw["evidence_sha256"]),
        )


@dataclass(frozen=True, order=True)
class SupplementalDirectionResult:
    """单个 carrier/case/direction 的独立 evaluator 公开结果。"""

    carrier_key: str
    case_key: tuple[int, ...]
    owner_key: tuple[int, ...]
    direction: str
    dimensions: tuple[SupplementalDimensionResult, ...]
    generation: SupplementalDimensionResult
    ablations: tuple[SupplementalAblationResult, ...]
    independent_reveal_status: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.carrier_key not in IN_SCOPE_CARRIER_KEYS:
            raise W02Lc16SupplementalError("补充结果 carrier 未登记")
        StableRecordKey.from_value(list(self.case_key), where="supplemental case_key")
        StableRecordKey.from_value(list(self.owner_key), where="supplemental owner_key")
        if self.direction not in DIRECTIONS:
            raise W02Lc16SupplementalError("补充结果 direction 未登记")
        if tuple(item.dimension_key for item in self.dimensions) != BEARING_DIMENSIONS:
            raise W02Lc16SupplementalError("补充结果 bearing 顺序漂移")
        if self.generation.dimension_key != GENERATION_HARD_CONJUNCT:
            raise W02Lc16SupplementalError("补充结果 generation 缺失")
        if any(item.case_count != 1 for item in (*self.dimensions, self.generation)):
            raise W02Lc16SupplementalError("单方向结果 case_count 必须为 1")
        if tuple(item.ablation_key for item in self.ablations) != ABLATION_ORDER:
            raise W02Lc16SupplementalError("补充结果 ablation 顺序漂移")
        _status(self.independent_reveal_status, where="independent reveal", blocked=True)
        _sha256(self.evidence_sha256, where="direction evidence")

    def to_public_dict(self) -> dict[str, Any]:
        """导出不含 private payload 的单方向结果。"""
        return {
            "ablations": [item.to_dict() for item in self.ablations],
            "carrier_key": self.carrier_key,
            "case_key": list(self.case_key),
            "dimensions": [item.to_dict() for item in self.dimensions],
            "direction": self.direction,
            "evidence_sha256": self.evidence_sha256,
            "generation": self.generation.to_dict(),
            "independent_reveal_status": self.independent_reveal_status,
            "owner_key": list(self.owner_key),
        }


@dataclass(frozen=True, order=True)
class SupplementalCarrierSummary:
    """按 carrier 折叠 21 个方向结果的公开摘要。"""

    carrier_key: str
    direction_evaluations: int
    dimensions: tuple[SupplementalDimensionResult, ...]
    generation: SupplementalDimensionResult
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.carrier_key not in IN_SCOPE_CARRIER_KEYS:
            raise W02Lc16SupplementalError("carrier summary 未登记")
        if self.direction_evaluations != len(SAMPLE_KINDS) * len(DIRECTIONS):
            raise W02Lc16SupplementalError("carrier summary direction 数量错误")
        if tuple(item.dimension_key for item in self.dimensions) != BEARING_DIMENSIONS:
            raise W02Lc16SupplementalError("carrier summary bearing 顺序漂移")
        if self.generation.dimension_key != GENERATION_HARD_CONJUNCT:
            raise W02Lc16SupplementalError("carrier summary generation 缺失")
        _sha256(self.evidence_sha256, where="carrier summary evidence")

    def to_dict(self) -> dict[str, Any]:
        """导出 carrier 摘要。"""
        return {
            "carrier_key": self.carrier_key,
            "direction_evaluations": self.direction_evaluations,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "evidence_sha256": self.evidence_sha256,
            "generation": self.generation.to_dict(),
        }


@dataclass(frozen=True, order=True)
class W02Lc16SupplementalReport:
    """独立 supplemental family 的 append-only 公开 receipt 投影。"""

    status: str
    parent_overlay_sha256: str
    w02_parent_receipt_sha256: str
    case_count: int
    direction_evaluations: int
    dimensions: tuple[SupplementalDimensionResult, ...]
    generation: SupplementalDimensionResult
    ablations: tuple[SupplementalAblationResult, ...]
    carrier_summaries: tuple[SupplementalCarrierSummary, ...]
    host_digests_before: dict[str, str]
    host_digests_after: dict[str, str]
    private_path_reads: int
    private_payload_bytes: int
    private_payload_reads: int
    evaluator_label_reads: int
    evaluator_label_writes: int
    host_write_count: int
    independent_evaluator_module_separate: int
    consumer_result_builder_reused: int
    runtime_observed: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        _status(self.status, where="supplemental report", blocked=True)
        if self.parent_overlay_sha256 != OVERLAY_SHA256:
            raise W02Lc16SupplementalError("parent overlay SHA 漂移")
        if self.w02_parent_receipt_sha256 != W02_PARENT_RECEIPT_SHA256:
            raise W02Lc16SupplementalError("W-02 parent receipt SHA 漂移")
        if self.case_count != CASE_COUNT or self.direction_evaluations != DIRECTION_EVALUATION_COUNT:
            raise W02Lc16SupplementalError("supplemental case/direction 数量错误")
        if tuple(item.dimension_key for item in self.dimensions) != BEARING_DIMENSIONS:
            raise W02Lc16SupplementalError("report bearing 顺序漂移")
        if self.generation.dimension_key != GENERATION_HARD_CONJUNCT:
            raise W02Lc16SupplementalError("report generation 缺失")
        if tuple(item.ablation_key for item in self.ablations) != ABLATION_ORDER:
            raise W02Lc16SupplementalError("report ablation 顺序漂移")
        if tuple(item.carrier_key for item in self.carrier_summaries) != IN_SCOPE_CARRIER_KEYS:
            raise W02Lc16SupplementalError("report carrier summary 未闭合")
        self._verify_host()
        for name in (
                "private_path_reads", "private_payload_bytes",
                "private_payload_reads", "evaluator_label_reads"):
            _nonnegative(getattr(self, name), where=name)
        if (self.runtime_observed == 1
                and (self.private_path_reads == 0
                     or self.private_payload_reads == 0
                     or self.evaluator_label_reads == 0)):
            raise W02Lc16SupplementalError(
                "已观察 supplemental runtime 必须有 private read evidence")
        if self.private_payload_bytes > MAX_PAYLOAD_BYTES or self.private_payload_reads > MAX_PAYLOAD_READS:
            raise W02Lc16SupplementalError("supplemental payload budget 超限")
        if self.evaluator_label_writes != 0 or self.host_write_count != 0:
            raise W02Lc16SupplementalError("supplemental evaluator 必须零写")
        if self.independent_evaluator_module_separate != 1 or self.consumer_result_builder_reused != 0:
            raise W02Lc16SupplementalError("supplemental evaluator independence 失败")
        if self.runtime_observed not in (0, 1):
            raise W02Lc16SupplementalError("runtime_observed 非法")
        _sha256(self.evidence_sha256, where="supplemental report evidence")
        if self.runtime_observed == 0 and self.status != "BLOCKED":
            raise W02Lc16SupplementalError("未运行 supplemental 只能 BLOCKED")

    def _verify_host(self) -> None:
        """要求五个 host 摘要集合完整且 before/after 不变。"""
        if (set(self.host_digests_before) != set(HOST_DIGEST_KEYS)
                or set(self.host_digests_after) != set(HOST_DIGEST_KEYS)):
            raise W02Lc16SupplementalError("host digest 集合不完整")
        for value in (*self.host_digests_before.values(), *self.host_digests_after.values()):
            _sha256(value, where="host digest")
        if self.host_digests_before != self.host_digests_after:
            raise W02Lc16SupplementalError("supplemental 改变 host digest")

    def to_public_dict(self) -> dict[str, Any]:
        """导出公开 receipt，排除方向细节和任何 private payload。"""
        return {
            "ablations": [item.to_dict() for item in self.ablations],
            "artifact_kind": ARTIFACT_KIND,
            "artifact_version": ARTIFACT_VERSION,
            "carrier_summaries": [item.to_dict() for item in self.carrier_summaries],
            "case_count": self.case_count,
            "consumer_result_builder_reused": self.consumer_result_builder_reused,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "direction_evaluations": self.direction_evaluations,
            "evidence_sha256": self.evidence_sha256,
            "generation": self.generation.to_dict(),
            "host_digests_after": dict(sorted(self.host_digests_after.items())),
            "host_digests_before": dict(sorted(self.host_digests_before.items())),
            "host_write_count": self.host_write_count,
            "independent_evaluator_module_separate": self.independent_evaluator_module_separate,
            "parent_overlay_sha256": self.parent_overlay_sha256,
            "private_reads": {
                "evaluator_label_reads": self.evaluator_label_reads,
                "private_path_reads": self.private_path_reads,
                "private_payload_bytes": self.private_payload_bytes,
                "private_payload_reads": self.private_payload_reads,
            },
            "runtime_observed": self.runtime_observed,
            "status": self.status,
            "w02_parent_receipt_sha256": self.w02_parent_receipt_sha256,
        }

    def canonical_bytes(self) -> bytes:
        """返回单尾换行的 canonical receipt bytes。"""
        return canonical_json_bytes(self.to_public_dict()) + b"\n"

    @classmethod
    def from_public_dict(cls, value: Any) -> "W02Lc16SupplementalReport":
        """从公开 receipt 恢复最小 report 骨架。"""
        raw = _exact(value, {
            "ablations", "artifact_kind", "artifact_version", "carrier_summaries",
            "case_count", "consumer_result_builder_reused", "dimensions",
            "direction_evaluations", "evidence_sha256", "generation",
            "host_digests_after", "host_digests_before", "host_write_count",
            "independent_evaluator_module_separate", "parent_overlay_sha256",
            "private_reads", "runtime_observed", "status",
            "w02_parent_receipt_sha256",
        }, where="W02Lc16SupplementalReport")
        if raw["artifact_kind"] != ARTIFACT_KIND or raw["artifact_version"] != ARTIFACT_VERSION:
            raise W02Lc16SupplementalError("supplemental receipt artifact identity 漂移")
        private_reads = _exact(raw["private_reads"], {
            "evaluator_label_reads", "private_path_reads",
            "private_payload_bytes", "private_payload_reads",
        }, where="supplemental private_reads")
        return cls(
            str(raw["status"]), str(raw["parent_overlay_sha256"]),
            str(raw["w02_parent_receipt_sha256"]), raw["case_count"],
            raw["direction_evaluations"],
            tuple(SupplementalDimensionResult.from_dict(item) for item in raw["dimensions"]),
            SupplementalDimensionResult.from_dict(raw["generation"]),
            tuple(SupplementalAblationResult.from_dict(item) for item in raw["ablations"]),
            tuple(_carrier_summary_from_dict(item) for item in raw["carrier_summaries"]),
            dict(raw["host_digests_before"]), dict(raw["host_digests_after"]),
            private_reads["private_path_reads"], private_reads["private_payload_bytes"],
            private_reads["private_payload_reads"], private_reads["evaluator_label_reads"],
            0, raw["host_write_count"], raw["independent_evaluator_module_separate"],
            raw["consumer_result_builder_reused"], raw["runtime_observed"],
            str(raw["evidence_sha256"]),
        )


def _carrier_summary_from_dict(value: Any) -> SupplementalCarrierSummary:
    """从公开 object 恢复 carrier 摘要。"""
    raw = _exact(value, {
        "carrier_key", "direction_evaluations", "dimensions",
        "evidence_sha256", "generation",
    }, where="SupplementalCarrierSummary")
    return SupplementalCarrierSummary(
        str(raw["carrier_key"]), raw["direction_evaluations"],
        tuple(SupplementalDimensionResult.from_dict(item) for item in raw["dimensions"]),
        SupplementalDimensionResult.from_dict(raw["generation"]),
        str(raw["evidence_sha256"]),
    )


@dataclass(frozen=True, order=True)
class SupplementalManifestFile:
    """预注册 manifest 的文件身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(_text(self.relative_path, where="manifest path"))
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_path or path.as_posix() != self.relative_path:
            raise W02Lc16SupplementalError("manifest path 逃逸")
        _positive(self.byte_count, where="manifest byte_count")
        _sha256(self.sha256, where="manifest file sha256")

    def to_dict(self) -> dict[str, Any]:
        """导出文件身份。"""
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SupplementalManifestFile":
        """从精确 object 恢复文件身份。"""
        raw = _exact(value, {"byte_count", "relative_path", "role", "sha256"}, where="SupplementalManifestFile")
        return cls(str(raw["relative_path"]), str(raw["role"]), raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True, order=True)
class W02Lc16SupplementalManifest:
    """冻结 W-02 supplemental 课程、预算、边界和零执行状态。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    parent_overlay_path: str
    parent_overlay_sha256: str
    w02_parent_receipt_sha256: str
    scope: CanonicalJsonObject
    execution_state: CanonicalJsonObject
    dependencies: tuple[SupplementalManifestFile, ...]
    evidence_files: tuple[SupplementalManifestFile, ...]

    def __post_init__(self) -> None:
        if (self.format_version != FORMAT_VERSION
                or self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise W02Lc16SupplementalError("supplemental manifest artifact identity 漂移")
        if self.parent_overlay_path != OVERLAY_PATH or self.parent_overlay_sha256 != OVERLAY_SHA256:
            raise W02Lc16SupplementalError("supplemental overlay parent 漂移")
        _sha256(self.w02_parent_receipt_sha256, where="manifest W-02 receipt SHA")
        if self.w02_parent_receipt_sha256 != W02_PARENT_RECEIPT_SHA256:
            raise W02Lc16SupplementalError("manifest W-02 receipt 承诺漂移")
        expected_scope = {
            "ablation_count": len(ABLATION_ORDER),
            "case_count": CASE_COUNT,
            "carrier_count": len(IN_SCOPE_CARRIER_KEYS),
            "direction_count": len(DIRECTIONS),
            "direction_evaluations": DIRECTION_EVALUATION_COUNT,
            "independent_evaluator_module_separate": 1,
            "max_host_writes": 0,
            "max_logic_operations": MAX_LOGIC_OPERATIONS,
            "max_payload_bytes": MAX_PAYLOAD_BYTES,
            "max_payload_reads": MAX_PAYLOAD_READS,
            "max_workers": MAX_WORKERS,
            "sample_kind_count": len(SAMPLE_KINDS),
            "w02_bearing_dimension_count": len(BEARING_DIMENSIONS),
        }
        if self.scope.to_value() != expected_scope:
            raise W02Lc16SupplementalError("supplemental scope 漂移")
        expected_state = dict(EXECUTION_STATE)
        if self.execution_state.to_value() != expected_state:
            raise W02Lc16SupplementalError("supplemental execution state 漂移")
        if (self.dependencies != tuple(sorted(self.dependencies, key=lambda item: item.role))
                or tuple(item.role for item in self.dependencies) != ("D03_LC16_OVERLAY", "W02_PARENT_RECEIPT_COMMITMENT")
                or len({item.relative_path for item in self.dependencies}) != len(self.dependencies)
                or self.evidence_files != tuple(sorted(self.evidence_files, key=lambda item: item.role))
                or tuple(item.role for item in self.evidence_files) != ("CATALOG", "CONTRACT", "EVALUATOR", "PUBLICATION", "TEST")):
            raise W02Lc16SupplementalError("supplemental 文件证据未闭合")

    def to_dict(self) -> dict[str, Any]:
        """导出 canonical manifest object。"""
        return {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "parent_overlay_path": self.parent_overlay_path,
            "parent_overlay_sha256": self.parent_overlay_sha256,
            "scope": self.scope.to_value(),
            "w02_parent_receipt_sha256": self.w02_parent_receipt_sha256,
        }

    def canonical_bytes(self) -> bytes:
        """返回单尾换行的 canonical manifest bytes。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回 manifest canonical SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "W02Lc16SupplementalManifest":
        """从精确 object 恢复预注册 manifest。"""
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version", "dependencies",
            "evidence_files", "execution_state", "format_version",
            "parent_overlay_path", "parent_overlay_sha256", "scope",
            "w02_parent_receipt_sha256",
        }, where="W02Lc16SupplementalManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise W02Lc16SupplementalError("supplemental manifest artifact_kind 非法")
        return cls(
            raw["format_version"], str(raw["artifact_version"]),
            str(raw["artifact_status"]), str(raw["parent_overlay_path"]),
            str(raw["parent_overlay_sha256"]), str(raw["w02_parent_receipt_sha256"]),
            CanonicalJsonObject.from_value(raw["scope"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
            tuple(SupplementalManifestFile.from_dict(item) for item in raw["dependencies"]),
            tuple(SupplementalManifestFile.from_dict(item) for item in raw["evidence_files"]),
        )


def read_w02_lc16_supplemental_manifest(path: str | Path) -> W02Lc16SupplementalManifest:
    """严格回读 canonical supplemental manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise W02Lc16SupplementalError("supplemental manifest newline 非法")
        manifest = W02Lc16SupplementalManifest.from_dict(
            parse_canonical_json_bytes(payload[:-1], require_object=True))
    except W02Lc16SupplementalError:
        raise
    except Exception as error:
        raise W02Lc16SupplementalError("supplemental manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise W02Lc16SupplementalError("supplemental manifest 非 canonical bytes")
    return manifest


def write_w02_lc16_supplemental_manifest(
        manifest: W02Lc16SupplementalManifest, path: str | Path,
        ) -> Path:
    """以逐字节幂等、异内容拒绝方式写入预注册 manifest。"""
    if not isinstance(manifest, W02Lc16SupplementalManifest):
        raise W02Lc16SupplementalError("supplemental manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise W02Lc16SupplementalError("supplemental manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    return target


__all__ = [
    "ABLATION_ORDER", "ARTIFACT_KIND", "ARTIFACT_STATUS", "ARTIFACT_VERSION",
    "BEARING_DIMENSIONS", "CASE_COUNT", "DIRECTION_EVALUATION_COUNT", "DIRECTIONS",
    "EXECUTION_STATE", "GENERATION_HARD_CONJUNCT", "HOST_DIGEST_KEYS",
    "IN_SCOPE_CARRIER_KEYS", "MANIFEST_PATH", "MAX_LOGIC_OPERATIONS",
    "MAX_PAYLOAD_BYTES", "MAX_PAYLOAD_READS", "MAX_WORKERS", "OVERLAY_PATH",
    "OVERLAY_SHA256", "QUALIFICATION_STATUSES", "SAMPLE_KINDS", "STATUSES",
    "W02Lc16SupplementalError", "W02Lc16SupplementalManifest",
    "W02Lc16SupplementalReport", "SupplementalAblationResult",
    "SupplementalCarrierSummary", "SupplementalDimensionResult",
    "SupplementalDirectionResult", "SupplementalManifestFile",
    "W02_PARENT_RECEIPT_SHA256", "read_w02_lc16_supplemental_manifest",
    "write_w02_lc16_supplemental_manifest",
]
