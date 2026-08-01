"""D-03 LC-16 后继 overlay 的可复用严格记录类型。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import (
    COURSE_STATE,
    COVERAGE_STATES,
    EVIDENCE_ROLES,
    PARSER_IDENTITIES,
    RENDERER_KEY,
    RENDERER_MODE,
    RENDERER_VERSION,
    SCOPE_KEYS,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    DIRECTIONS,
    IN_SCOPE_CARRIER_KEYS,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    SAMPLE_KINDS,
    SAMPLE_SPLITS,
)


class D03Lc16OverlayError(RuntimeError):
    """D-03 LC-16 后继 overlay 的身份、覆盖或边界不闭合。"""


def exact_dict(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    """要求输入为字段集合精确相等的 object。"""
    if not isinstance(value, dict) or set(value) != expected:
        raise D03Lc16OverlayError(f"{where} 字段不精确")
    return value


def text(
        value: Any, *, where: str, allow_empty: bool = False,
        ) -> str:
    """校验无首尾空白的规范文本。"""
    if not isinstance(value, str) or value.strip() != value:
        raise D03Lc16OverlayError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise D03Lc16OverlayError(f"{where} 不得为空")
    return value


def positive(value: Any, *, where: str) -> int:
    """校验正严格整数并拒绝 bool。"""
    if type(value) is not int or value <= 0:
        raise D03Lc16OverlayError(f"{where} 必须是正严格整数")
    return value


def nonnegative(value: Any, *, where: str) -> int:
    """校验非负严格整数并拒绝 bool。"""
    if type(value) is not int or value < 0:
        raise D03Lc16OverlayError(f"{where} 必须是非负严格整数")
    return value


def bit(value: Any, *, where: str) -> int:
    """校验严格 0/1 整数。"""
    if type(value) is not int or value not in (0, 1):
        raise D03Lc16OverlayError(f"{where} 必须是 0/1 严格整数")
    return value


def sha256(value: Any, *, where: str) -> str:
    """校验小写 SHA-256 文本。"""
    checked = text(value, where=where).lower()
    if (len(checked) != 64
            or any(item not in "0123456789abcdef" for item in checked)):
        raise D03Lc16OverlayError(f"{where} 必须是小写 SHA-256")
    return checked


def relative_path(value: Any, *, where: str) -> str:
    """校验不逃逸的 POSIX 仓库相对路径。"""
    checked = text(value, where=where)
    path = PurePosixPath(checked)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or "\\" in checked or path.as_posix() != checked
            or ":" in path.parts[0]):
        raise D03Lc16OverlayError(f"{where} 必须是安全 POSIX 相对路径")
    return checked


def string_tuple(
        value: tuple[str, ...], *, where: str, allow_empty: bool = False,
        ) -> tuple[str, ...]:
    """校验有序去重的规范文本 tuple。"""
    if (not isinstance(value, tuple) or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item
                   or item.strip() != item for item in value)
            or len(set(value)) != len(value)):
        raise D03Lc16OverlayError(f"{where} 必须是去重文本 tuple")
    return value


@dataclass(frozen=True, order=True)
class OverlayFileIdentity:
    """冻结一个公开文件的相对路径、字节数和 SHA-256。"""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        relative_path(self.relative_path, where="file relative_path")
        positive(self.size_bytes, where="file size_bytes")
        sha256(self.sha256, where="file sha256")

    def to_dict(self) -> dict[str, Any]:
        """导出规范文件身份 object。"""
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayFileIdentity":
        """从精确 object 恢复文件身份。"""
        raw = exact_dict(value, {"relative_path", "sha256", "size_bytes"},
                         where="OverlayFileIdentity")
        return cls(str(raw["relative_path"]), raw["size_bytes"],
                   str(raw["sha256"]))


@dataclass(frozen=True, order=True)
class OverlayEvidenceFile:
    """把 overlay 自身的合同、目录和测试绑定到文件身份。"""

    role: str
    file_identity: OverlayFileIdentity

    def __post_init__(self) -> None:
        if self.role not in EVIDENCE_ROLES:
            raise D03Lc16OverlayError("evidence role 未登记")
        if not isinstance(self.file_identity, OverlayFileIdentity):
            raise D03Lc16OverlayError("evidence file identity 类型非法")

    def to_dict(self) -> dict[str, Any]:
        """导出角色和文件身份。"""
        return {"file_identity": self.file_identity.to_dict(), "role": self.role}

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayEvidenceFile":
        """从精确 object 恢复证据文件。"""
        raw = exact_dict(value, {"file_identity", "role"},
                         where="OverlayEvidenceFile")
        return cls(str(raw["role"]),
                   OverlayFileIdentity.from_dict(raw["file_identity"]))


@dataclass(frozen=True, order=True)
class OverlayCourseCase:
    """绑定一个已物化小批样本的 owner、split 与 payload 身份。"""

    case_key: StableRecordKey
    sample_kind: str
    owner_key: StableRecordKey
    split: str
    directions: tuple[str, ...]
    materialization_size_bytes: int
    materialization_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_key, StableRecordKey):
            raise D03Lc16OverlayError("course case_key 类型非法")
        if self.sample_kind not in SAMPLE_KINDS:
            raise D03Lc16OverlayError("course sample_kind 未登记")
        if not isinstance(self.owner_key, StableRecordKey):
            raise D03Lc16OverlayError("course owner_key 类型非法")
        if self.split != SAMPLE_SPLITS[self.sample_kind]:
            raise D03Lc16OverlayError("course owner/split 与样本类型不一致")
        if self.directions != DIRECTIONS:
            raise D03Lc16OverlayError("course directions 必须精确覆盖三向")
        positive(self.materialization_size_bytes,
                 where="course materialization_size_bytes")
        sha256(self.materialization_sha256,
               where="course materialization_sha256")

    def to_dict(self) -> dict[str, Any]:
        """导出课程 case 身份。"""
        return {
            "case_key": self.case_key.to_list(),
            "directions": list(self.directions),
            "materialization_sha256": self.materialization_sha256,
            "materialization_size_bytes": self.materialization_size_bytes,
            "owner_key": self.owner_key.to_list(),
            "sample_kind": self.sample_kind,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayCourseCase":
        """从精确 object 恢复课程 case。"""
        raw = exact_dict(value, {
            "case_key", "directions", "materialization_sha256",
            "materialization_size_bytes", "owner_key", "sample_kind", "split",
        }, where="OverlayCourseCase")
        return cls(
            StableRecordKey.from_value(raw["case_key"], where="case_key"),
            str(raw["sample_kind"]),
            StableRecordKey.from_value(raw["owner_key"], where="owner_key"),
            str(raw["split"]),
            tuple(str(item) for item in raw["directions"]),
            raw["materialization_size_bytes"],
            str(raw["materialization_sha256"]),
        )


@dataclass(frozen=True, order=True)
class OverlayCarrierCourse:
    """冻结单一 carrier 的课程、parser、renderer、预算和七类 payload。"""

    carrier_key: str
    course_state: str
    manifest_identity: OverlayFileIdentity
    sample_identity: OverlayFileIdentity
    parser_package: str
    parser_version: str
    renderer_key: str
    renderer_version: str
    renderer_mode: str
    budget: CanonicalJsonObject
    cases: tuple[OverlayCourseCase, ...]

    def __post_init__(self) -> None:
        if self.carrier_key not in IN_SCOPE_CARRIER_KEYS:
            raise D03Lc16OverlayError("course carrier 未登记")
        if self.course_state != COURSE_STATE:
            raise D03Lc16OverlayError("course state 漂移")
        if (not isinstance(self.manifest_identity, OverlayFileIdentity)
                or not isinstance(self.sample_identity, OverlayFileIdentity)):
            raise D03Lc16OverlayError("course file identity 类型非法")
        text(self.parser_package, where="course parser_package")
        text(self.parser_version, where="course parser_version")
        if (self.parser_package, self.parser_version) != PARSER_IDENTITIES[
                self.carrier_key]:
            raise D03Lc16OverlayError("course parser identity 漂移")
        if (self.renderer_key != RENDERER_KEY
                or self.renderer_version != RENDERER_VERSION
                or self.renderer_mode != RENDERER_MODE):
            raise D03Lc16OverlayError("course renderer identity 漂移")
        if not isinstance(self.budget, CanonicalJsonObject):
            raise D03Lc16OverlayError("course budget 类型非法")
        budget = self.budget.to_value()
        if (not isinstance(budget, dict)
                or budget.get("carrier_key") != self.carrier_key
                or budget.get("max_cases") != len(SAMPLE_KINDS)):
            raise D03Lc16OverlayError("course budget 与 carrier 不一致")
        if (not isinstance(self.cases, tuple)
                or tuple(item.sample_kind for item in self.cases) != SAMPLE_KINDS
                or len({item.case_key for item in self.cases}) != len(self.cases)
                or len({item.owner_key for item in self.cases}) != len(self.cases)):
            raise D03Lc16OverlayError("course 七类 case 未精确闭合")

    def to_dict(self) -> dict[str, Any]:
        """导出单载体课程总账。"""
        return {
            "budget": self.budget.to_value(),
            "carrier_key": self.carrier_key,
            "cases": [item.to_dict() for item in self.cases],
            "course_state": self.course_state,
            "manifest_identity": self.manifest_identity.to_dict(),
            "parser_package": self.parser_package,
            "parser_version": self.parser_version,
            "renderer_key": self.renderer_key,
            "renderer_mode": self.renderer_mode,
            "renderer_version": self.renderer_version,
            "sample_identity": self.sample_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayCarrierCourse":
        """从精确 object 恢复单载体课程。"""
        raw = exact_dict(value, {
            "budget", "carrier_key", "cases", "course_state",
            "manifest_identity", "parser_package", "parser_version",
            "renderer_key", "renderer_mode", "renderer_version",
            "sample_identity",
        }, where="OverlayCarrierCourse")
        return cls(
            str(raw["carrier_key"]), str(raw["course_state"]),
            OverlayFileIdentity.from_dict(raw["manifest_identity"]),
            OverlayFileIdentity.from_dict(raw["sample_identity"]),
            str(raw["parser_package"]), str(raw["parser_version"]),
            str(raw["renderer_key"]), str(raw["renderer_version"]),
            str(raw["renderer_mode"]),
            CanonicalJsonObject.from_value(raw["budget"]),
            tuple(OverlayCourseCase.from_dict(item) for item in raw["cases"]),
        )


@dataclass(frozen=True, order=True)
class OverlayScopeRecord:
    """冻结一个 W-02 至 W-09 typed Artifact scope 的入口和失效后缀。"""

    scope_key: str
    earliest_stage: str
    evaluator_key: str
    current_state: str
    failure_suffix: tuple[str, ...]

    def __post_init__(self) -> None:
        text(self.scope_key, where="scope key")
        text(self.earliest_stage, where="scope earliest_stage")
        text(self.evaluator_key, where="scope evaluator_key")
        if self.current_state not in COVERAGE_STATES:
            raise D03Lc16OverlayError("scope current_state 未登记")
        string_tuple(self.failure_suffix, where="scope failure_suffix")

    def to_dict(self) -> dict[str, Any]:
        """导出阶段 scope。"""
        return {
            "current_state": self.current_state,
            "earliest_stage": self.earliest_stage,
            "evaluator_key": self.evaluator_key,
            "failure_suffix": list(self.failure_suffix),
            "scope_key": self.scope_key,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayScopeRecord":
        """从精确 object 恢复阶段 scope。"""
        raw = exact_dict(value, {
            "current_state", "earliest_stage", "evaluator_key",
            "failure_suffix", "scope_key",
        }, where="OverlayScopeRecord")
        return cls(
            str(raw["scope_key"]), str(raw["earliest_stage"]),
            str(raw["evaluator_key"]), str(raw["current_state"]),
            tuple(str(item) for item in raw["failure_suffix"]),
        )


@dataclass(frozen=True, order=True)
class OverlayCoverageCell:
    """显式登记一个 carrier×direction×scope 单元的当前事实状态。"""

    carrier_key: str
    direction: str
    scope_key: str
    evaluator_key: str
    current_state: str

    def __post_init__(self) -> None:
        if self.carrier_key not in IN_SCOPE_CARRIER_KEYS:
            raise D03Lc16OverlayError("coverage carrier 未登记")
        if self.direction not in DIRECTIONS:
            raise D03Lc16OverlayError("coverage direction 未登记")
        if self.scope_key not in SCOPE_KEYS:
            raise D03Lc16OverlayError("coverage scope 未登记")
        text(self.evaluator_key, where="coverage evaluator_key")
        if self.current_state not in COVERAGE_STATES:
            raise D03Lc16OverlayError("coverage current_state 未登记")

    @property
    def key(self) -> tuple[str, str, str]:
        """返回 carrier、方向与 scope 的复合键。"""
        return self.carrier_key, self.direction, self.scope_key

    def to_dict(self) -> dict[str, Any]:
        """导出 coverage cell。"""
        return {
            "carrier_key": self.carrier_key,
            "current_state": self.current_state,
            "direction": self.direction,
            "evaluator_key": self.evaluator_key,
            "scope_key": self.scope_key,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayCoverageCell":
        """从精确 object 恢复 coverage cell。"""
        raw = exact_dict(value, {
            "carrier_key", "current_state", "direction", "evaluator_key",
            "scope_key",
        }, where="OverlayCoverageCell")
        return cls(
            str(raw["carrier_key"]), str(raw["direction"]),
            str(raw["scope_key"]), str(raw["evaluator_key"]),
            str(raw["current_state"]),
        )


@dataclass(frozen=True, order=True)
class OverlayEvaluatorBoundary:
    """声明独立 evaluator 可判维度、NE 边界和 NE 聚合策略。"""

    boundary_key: str
    evaluator_key: str
    owner_key: str
    direction: str
    implementation_state: str
    decidable_dimensions: tuple[str, ...]
    ne_conditions: tuple[str, ...]
    ne_policy: str

    def __post_init__(self) -> None:
        for name in (
                "boundary_key", "evaluator_key", "owner_key", "direction",
                "implementation_state", "ne_policy"):
            text(getattr(self, name), where=f"evaluator {name}")
        decidable = string_tuple(
            self.decidable_dimensions, where="evaluator decidable_dimensions")
        conditions = string_tuple(
            self.ne_conditions, where="evaluator ne_conditions")
        if set(decidable) & set(conditions):
            raise D03Lc16OverlayError("evaluator decidable 与 NE 边界重叠")

    def to_dict(self) -> dict[str, Any]:
        """导出 evaluator 边界。"""
        return {
            "boundary_key": self.boundary_key,
            "decidable_dimensions": list(self.decidable_dimensions),
            "direction": self.direction,
            "evaluator_key": self.evaluator_key,
            "implementation_state": self.implementation_state,
            "ne_conditions": list(self.ne_conditions),
            "ne_policy": self.ne_policy,
            "owner_key": self.owner_key,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayEvaluatorBoundary":
        """从精确 object 恢复 evaluator 边界。"""
        raw = exact_dict(value, {
            "boundary_key", "decidable_dimensions", "direction",
            "evaluator_key", "implementation_state", "ne_conditions",
            "ne_policy", "owner_key",
        }, where="OverlayEvaluatorBoundary")
        return cls(
            str(raw["boundary_key"]), str(raw["evaluator_key"]),
            str(raw["owner_key"]), str(raw["direction"]),
            str(raw["implementation_state"]),
            tuple(str(item) for item in raw["decidable_dimensions"]),
            tuple(str(item) for item in raw["ne_conditions"]),
            str(raw["ne_policy"]),
        )


@dataclass(frozen=True, order=True)
class OverlayResourceBudget:
    """冻结一个 supplemental family 的 records、评测和 I/O 硬预算。"""

    family_key: str
    max_records: int
    max_direction_evaluations: int
    max_payload_bytes: int
    max_payload_reads: int
    max_logic_operations: int
    max_workers: int
    max_host_writes: int

    def __post_init__(self) -> None:
        text(self.family_key, where="resource family_key")
        for name in (
                "max_records", "max_direction_evaluations", "max_payload_bytes",
                "max_payload_reads", "max_logic_operations", "max_workers"):
            positive(getattr(self, name), where=f"resource {name}")
        nonnegative(self.max_host_writes, where="resource max_host_writes")
        if (self.max_records != len(IN_SCOPE_CARRIER_KEYS) * len(SAMPLE_KINDS)
                or self.max_direction_evaluations
                != self.max_records * len(DIRECTIONS)
                or self.max_host_writes != 0):
            raise D03Lc16OverlayError("supplemental resource 课程规模或只读边界漂移")

    def to_dict(self) -> dict[str, Any]:
        """导出 supplemental 资源预算。"""
        return {
            "family_key": self.family_key,
            "max_direction_evaluations": self.max_direction_evaluations,
            "max_host_writes": self.max_host_writes,
            "max_logic_operations": self.max_logic_operations,
            "max_payload_bytes": self.max_payload_bytes,
            "max_payload_reads": self.max_payload_reads,
            "max_records": self.max_records,
            "max_workers": self.max_workers,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayResourceBudget":
        """从精确 object 恢复 supplemental 资源预算。"""
        raw = exact_dict(value, {
            "family_key", "max_direction_evaluations", "max_host_writes",
            "max_logic_operations", "max_payload_bytes", "max_payload_reads",
            "max_records", "max_workers",
        }, where="OverlayResourceBudget")
        return cls(
            str(raw["family_key"]), raw["max_records"],
            raw["max_direction_evaluations"], raw["max_payload_bytes"],
            raw["max_payload_reads"], raw["max_logic_operations"],
            raw["max_workers"], raw["max_host_writes"],
        )


@dataclass(frozen=True, order=True)
class OverlayFailureDependency:
    """把一个失败条件绑定到完整下游重验后缀。"""

    failure_key: str
    invalidation_suffix: tuple[str, ...]

    def __post_init__(self) -> None:
        text(self.failure_key, where="failure key")
        string_tuple(self.invalidation_suffix, where="invalidation suffix")

    def to_dict(self) -> dict[str, Any]:
        """导出失败键与失效后缀。"""
        return {
            "failure_key": self.failure_key,
            "invalidation_suffix": list(self.invalidation_suffix),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayFailureDependency":
        """从精确 object 恢复失败依赖。"""
        raw = exact_dict(value, {"failure_key", "invalidation_suffix"},
                         where="OverlayFailureDependency")
        return cls(str(raw["failure_key"]),
                   tuple(str(item) for item in raw["invalidation_suffix"]))


@dataclass(frozen=True, order=True)
class OverlayGenerationAccount:
    """分别核算来源化回放与开放生成，不允许聚合或相互代过。"""

    account_key: str
    current_status: str
    evaluator_boundary_key: str
    runtime_evidenced: int
    included_in_current_directional_evidence: int
    required_for_j_lc_w09: int
    failure_suffix: tuple[str, ...]
    aggregate_with_other_account: int = 0

    def __post_init__(self) -> None:
        text(self.account_key, where="generation account_key")
        text(self.current_status, where="generation current_status")
        text(self.evaluator_boundary_key, where="generation evaluator key",
             allow_empty=True)
        for name in (
                "runtime_evidenced", "included_in_current_directional_evidence",
                "required_for_j_lc_w09", "aggregate_with_other_account"):
            bit(getattr(self, name), where=f"generation {name}")
        if self.aggregate_with_other_account != 0:
            raise D03Lc16OverlayError("generation 两类账户禁止聚合")
        string_tuple(self.failure_suffix, where="generation failure_suffix")

    def to_dict(self) -> dict[str, Any]:
        """导出独立 generation 账户。"""
        return {
            "account_key": self.account_key,
            "aggregate_with_other_account": self.aggregate_with_other_account,
            "current_status": self.current_status,
            "evaluator_boundary_key": self.evaluator_boundary_key,
            "failure_suffix": list(self.failure_suffix),
            "included_in_current_directional_evidence": (
                self.included_in_current_directional_evidence),
            "required_for_j_lc_w09": self.required_for_j_lc_w09,
            "runtime_evidenced": self.runtime_evidenced,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayGenerationAccount":
        """从精确 object 恢复 generation 账户。"""
        raw = exact_dict(value, {
            "account_key", "aggregate_with_other_account", "current_status",
            "evaluator_boundary_key", "failure_suffix",
            "included_in_current_directional_evidence",
            "required_for_j_lc_w09", "runtime_evidenced",
        }, where="OverlayGenerationAccount")
        return cls(
            str(raw["account_key"]), str(raw["current_status"]),
            str(raw["evaluator_boundary_key"]), raw["runtime_evidenced"],
            raw["included_in_current_directional_evidence"],
            raw["required_for_j_lc_w09"],
            tuple(str(item) for item in raw["failure_suffix"]),
            raw["aggregate_with_other_account"],
        )


__all__ = [
    "D03Lc16OverlayError", "OverlayCarrierCourse", "OverlayCourseCase",
    "OverlayCoverageCell", "OverlayEvaluatorBoundary", "OverlayEvidenceFile",
    "OverlayFailureDependency", "OverlayFileIdentity", "OverlayGenerationAccount",
    "OverlayResourceBudget", "OverlayScopeRecord", "bit", "exact_dict",
    "nonnegative", "positive", "relative_path", "sha256", "string_tuple", "text",
]
