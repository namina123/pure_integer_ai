"""D-03 九阶段身份、资料可见性、评测、预算和恢复合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    FORMAT_VERSION,
    STAGE_KEYS,
    W06_SUBSTAGE_KEYS,
    W07_SUBSTAGE_KEYS,
    enum_text,
    exact_dict,
    flag,
    positive,
    string_tuple,
    text,
    validate_zero_execution_state,
)


STAGE_ARTIFACT_KIND = "PH2_D03_STAGE_MANIFEST"
SPLITS = ("train", "dev", "held_out", "adversarial", "wall")
CANDIDATE_ALLOWED_SPLITS = ("train",)
CANDIDATE_FORBIDDEN_SPLITS = ("dev", "held_out", "adversarial", "wall")
AGGREGATION_POLICY = "ALL_BEARING_DIMENSIONS_MUST_PASS"
NE_POLICIES = ("BLOCK", "ALLOW_NON_BEARING")
OWNER_KEYS = (
    "PH2_TRAIN_CANDIDATE",
    "PH2_TRAINING_EVIDENCE",
    "PH2_PRIVATE_EVALUATOR",
)
RUN_ID_POLICY = "NEW_POSITIVE_INTEGER_REQUIRED"
REQUIRED_FAILURE_POINTS = (
    "BEFORE_FIRST_SHARD",
    "AFTER_PARTIAL_SHARD",
    "BEFORE_MERGE_PREVIEW",
    "AFTER_MERGE_BEFORE_COMMIT",
    "AFTER_COMMIT_BEFORE_CURSOR",
    "AFTER_MANIFEST_PUBLISH",
)


@dataclass(frozen=True)
class D03StageIdentity:
    """冻结一个 W 阶段的序号、立即前置和内部严格子序。"""

    stage_key: str
    ordinal: int
    prerequisite_stage_keys: tuple[str, ...]
    substage_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        enum_text(self.stage_key, STAGE_KEYS, where="stage_key")
        positive(self.ordinal, where="stage ordinal")
        expected_ordinal = STAGE_KEYS.index(self.stage_key) + 1
        if self.ordinal != expected_ordinal:
            raise D03ContractError("stage 顺序与 ordinal 不一致")
        if not isinstance(self.prerequisite_stage_keys, tuple):
            raise D03ContractError("stage prerequisite 必须是 tuple")
        expected_prerequisites = (
            () if self.ordinal == 1 else (STAGE_KEYS[self.ordinal - 2],)
        )
        if self.prerequisite_stage_keys != expected_prerequisites:
            raise D03ContractError("stage 前置顺序缺失、跳级或成环")
        if not isinstance(self.substage_keys, tuple):
            raise D03ContractError("stage substage 必须是 tuple")
        expected_substages: tuple[str, ...] = ()
        if self.stage_key == "W-06":
            expected_substages = W06_SUBSTAGE_KEYS
        elif self.stage_key == "W-07":
            expected_substages = W07_SUBSTAGE_KEYS
        if self.substage_keys != expected_substages:
            raise D03ContractError("W-06/W-07 子序不完整或乱序")

    def to_dict(self) -> dict[str, Any]:
        """导出阶段身份。"""
        return {
            "ordinal": self.ordinal,
            "prerequisite_stage_keys": list(self.prerequisite_stage_keys),
            "stage_key": self.stage_key,
            "substage_keys": list(self.substage_keys),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03StageIdentity":
        """从严格 object 恢复阶段身份。"""
        raw = exact_dict(value, {
            "ordinal", "prerequisite_stage_keys", "stage_key", "substage_keys",
        }, where="D03StageIdentity")
        return cls(
            str(raw["stage_key"]), raw["ordinal"],
            string_tuple(
                raw["prerequisite_stage_keys"], where="prerequisite stages",
                allow_empty=True,
            ),
            string_tuple(
                raw["substage_keys"], where="substage keys", allow_empty=True,
            ),
        )


@dataclass(frozen=True)
class StageDataVisibility:
    """冻结 candidate 白名单、future 明拒集合和双私有 owner 视图。"""

    train_pack_keys: tuple[str, ...]
    future_pack_keys: tuple[str, ...]
    dev_pack_keys: tuple[str, ...]
    held_out_pack_keys: tuple[str, ...]
    evaluator_pack_keys: tuple[str, ...]
    candidate_owner: str
    teacher_owner: str
    evaluator_owner: str
    candidate_allowed_splits: tuple[str, ...]
    candidate_forbidden_splits: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
                "train_pack_keys", "future_pack_keys", "dev_pack_keys",
                "held_out_pack_keys", "evaluator_pack_keys"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise D03ContractError(f"visibility {name} 必须是 tuple")
            normalized = tuple(sorted(
                string_tuple(values, where=name, allow_empty=True)))
            object.__setattr__(self, name, normalized)
        if set(self.train_pack_keys) & set(self.future_pack_keys):
            raise D03ContractError("train 白名单与 future 明拒集合重叠")
        owners = (self.candidate_owner, self.teacher_owner, self.evaluator_owner)
        if owners != OWNER_KEYS or len(set(owners)) != 3:
            raise D03ContractError("candidate/teacher/evaluator owner 必须物理分离")
        if self.candidate_allowed_splits != CANDIDATE_ALLOWED_SPLITS:
            raise D03ContractError("candidate split 只能是 train")
        if self.candidate_forbidden_splits != CANDIDATE_FORBIDDEN_SPLITS:
            raise D03ContractError("candidate forbidden split 不完整")

    def to_dict(self) -> dict[str, Any]:
        """导出阶段资料可见性。"""
        return {
            "candidate_allowed_splits": list(self.candidate_allowed_splits),
            "candidate_forbidden_splits": list(self.candidate_forbidden_splits),
            "candidate_owner": self.candidate_owner,
            "dev_pack_keys": list(self.dev_pack_keys),
            "evaluator_owner": self.evaluator_owner,
            "evaluator_pack_keys": list(self.evaluator_pack_keys),
            "future_pack_keys": list(self.future_pack_keys),
            "held_out_pack_keys": list(self.held_out_pack_keys),
            "teacher_owner": self.teacher_owner,
            "train_pack_keys": list(self.train_pack_keys),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StageDataVisibility":
        """从严格 object 恢复阶段资料可见性。"""
        raw = exact_dict(value, {
            "candidate_allowed_splits", "candidate_forbidden_splits",
            "candidate_owner", "dev_pack_keys", "evaluator_owner",
            "evaluator_pack_keys", "future_pack_keys", "held_out_pack_keys",
            "teacher_owner", "train_pack_keys",
        }, where="StageDataVisibility")
        return cls(
            string_tuple(raw["train_pack_keys"], where="train packs", allow_empty=True),
            string_tuple(raw["future_pack_keys"], where="future packs", allow_empty=True),
            string_tuple(raw["dev_pack_keys"], where="dev packs", allow_empty=True),
            string_tuple(raw["held_out_pack_keys"], where="held-out packs", allow_empty=True),
            string_tuple(raw["evaluator_pack_keys"], where="evaluator packs", allow_empty=True),
            str(raw["candidate_owner"]), str(raw["teacher_owner"]),
            str(raw["evaluator_owner"]),
            string_tuple(raw["candidate_allowed_splits"], where="allowed splits"),
            string_tuple(raw["candidate_forbidden_splits"], where="forbidden splits"),
        )


@dataclass(frozen=True, order=True)
class EvaluationThreshold:
    """冻结一个不可被均值掩盖的预注册 evaluator 维度阈值。"""

    dimension_key: str
    min_pass_numerator: int
    min_pass_denominator: int
    max_fail_count: int
    bearing: int
    ne_policy: str
    preregistered: int

    def __post_init__(self) -> None:
        text(self.dimension_key, where="threshold dimension")
        positive(self.min_pass_denominator, where="threshold denominator")
        if (type(self.min_pass_numerator) is not int
                or self.min_pass_numerator < 0
                or self.min_pass_numerator > self.min_pass_denominator):
            raise D03ContractError("threshold numerator 非法")
        if type(self.max_fail_count) is not int or self.max_fail_count < 0:
            raise D03ContractError("threshold max_fail_count 非法")
        flag(self.bearing, where="threshold bearing")
        enum_text(self.ne_policy, NE_POLICIES, where="threshold NE policy")
        flag(self.preregistered, where="threshold preregistered")
        if self.preregistered != 1:
            raise D03ContractError("evaluator 阈值必须在运行前预注册")
        if self.bearing == 1 and self.ne_policy != "BLOCK":
            raise D03ContractError("承重维度 FAIL/NE 必须阻断")

    def to_dict(self) -> dict[str, Any]:
        """导出预注册维度阈值。"""
        return {
            "bearing": self.bearing,
            "dimension_key": self.dimension_key,
            "max_fail_count": self.max_fail_count,
            "min_pass_denominator": self.min_pass_denominator,
            "min_pass_numerator": self.min_pass_numerator,
            "ne_policy": self.ne_policy,
            "preregistered": self.preregistered,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationThreshold":
        """从严格 object 恢复预注册维度阈值。"""
        raw = exact_dict(value, {
            "bearing", "dimension_key", "max_fail_count",
            "min_pass_denominator", "min_pass_numerator", "ne_policy",
            "preregistered",
        }, where="EvaluationThreshold")
        return cls(
            str(raw["dimension_key"]), raw["min_pass_numerator"],
            raw["min_pass_denominator"], raw["max_fail_count"],
            raw["bearing"], str(raw["ne_policy"]), raw["preregistered"],
        )


@dataclass(frozen=True)
class StageEvaluationBinding:
    """绑定 evaluator 身份、owner、逐维阈值、消融和连续窗口。"""

    evaluator_key: str
    evaluator_version: str
    owner_key: str
    aggregation_policy: str
    thresholds: tuple[EvaluationThreshold, ...]
    ablation_keys: tuple[str, ...]
    continuous_window_count: int

    def __post_init__(self) -> None:
        text(self.evaluator_key, where="evaluator key")
        text(self.evaluator_version, where="evaluator version")
        if self.owner_key != "PH2_PRIVATE_EVALUATOR":
            raise D03ContractError("evaluator owner 必须是私有 owner")
        if self.aggregation_policy != AGGREGATION_POLICY:
            raise D03ContractError("承重维度不得被均值或 aggregate 掩盖")
        if (not isinstance(self.thresholds, tuple) or not self.thresholds
                or any(not isinstance(item, EvaluationThreshold)
                       for item in self.thresholds)):
            raise D03ContractError("evaluator thresholds 不能为空")
        thresholds = tuple(sorted(self.thresholds))
        if len({item.dimension_key for item in thresholds}) != len(thresholds):
            raise D03ContractError("evaluator threshold dimension 重复")
        if not any(item.bearing == 1 for item in thresholds):
            raise D03ContractError("evaluator 至少有一个承重维度")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "ablation_keys", tuple(sorted(
            string_tuple(self.ablation_keys, where="ablation keys"))))
        positive(self.continuous_window_count, where="continuous window")

    def to_dict(self) -> dict[str, Any]:
        """导出阶段 evaluator 绑定。"""
        return {
            "ablation_keys": list(self.ablation_keys),
            "aggregation_policy": self.aggregation_policy,
            "continuous_window_count": self.continuous_window_count,
            "evaluator_key": self.evaluator_key,
            "evaluator_version": self.evaluator_version,
            "owner_key": self.owner_key,
            "thresholds": [item.to_dict() for item in self.thresholds],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StageEvaluationBinding":
        """从严格 object 恢复阶段 evaluator 绑定。"""
        raw = exact_dict(value, {
            "ablation_keys", "aggregation_policy", "continuous_window_count",
            "evaluator_key", "evaluator_version", "owner_key", "thresholds",
        }, where="StageEvaluationBinding")
        if not isinstance(raw["thresholds"], list):
            raise D03ContractError("thresholds 必须是数组")
        return cls(
            str(raw["evaluator_key"]), str(raw["evaluator_version"]),
            str(raw["owner_key"]), str(raw["aggregation_policy"]),
            tuple(EvaluationThreshold.from_dict(item) for item in raw["thresholds"]),
            string_tuple(raw["ablation_keys"], where="ablation keys"),
            raw["continuous_window_count"],
        )


@dataclass(frozen=True)
class StageResourceBudget:
    """冻结 records、I/O、逻辑、重算、worker 和 checkpoint 硬上限。"""

    max_records: int
    max_segments: int
    max_payload_gets: int
    max_payload_bytes: int
    max_logic_operations: int
    max_recompute_objects: int
    max_workers: int
    max_checkpoint_count: int

    def __post_init__(self) -> None:
        for name in (
                "max_records", "max_segments", "max_payload_gets",
                "max_payload_bytes", "max_logic_operations",
                "max_recompute_objects", "max_workers",
                "max_checkpoint_count"):
            positive(getattr(self, name), where=f"预算 {name}")
        if self.max_workers != 4:
            raise D03ContractError("预算 max_workers 必须覆盖 1/2/4")

    def to_dict(self) -> dict[str, Any]:
        """导出阶段资源预算。"""
        return {name: getattr(self, name) for name in (
            "max_checkpoint_count", "max_logic_operations", "max_payload_bytes",
            "max_payload_gets", "max_recompute_objects", "max_records",
            "max_segments", "max_workers",
        )}

    @classmethod
    def from_dict(cls, value: Any) -> "StageResourceBudget":
        """从严格 object 恢复阶段资源预算。"""
        raw = exact_dict(value, {
            "max_checkpoint_count", "max_logic_operations", "max_payload_bytes",
            "max_payload_gets", "max_recompute_objects", "max_records",
            "max_segments", "max_workers",
        }, where="StageResourceBudget")
        return cls(
            raw["max_records"], raw["max_segments"], raw["max_payload_gets"],
            raw["max_payload_bytes"], raw["max_logic_operations"],
            raw["max_recompute_objects"], raw["max_workers"],
            raw["max_checkpoint_count"],
        )


@dataclass(frozen=True)
class StageRecoveryBinding:
    """冻结 run/cursor/base fence、逻辑 shard、barrier 和失败恢复点。"""

    run_id_policy: str
    cursor_version: str
    base_fence_required: int
    logical_shard_count: int
    allowed_worker_counts: tuple[int, ...]
    merge_barrier_key: str
    failure_point_keys: tuple[str, ...]
    fresh_resume_equivalent: int

    def __post_init__(self) -> None:
        if self.run_id_policy != RUN_ID_POLICY:
            raise D03ContractError("run id policy 非法")
        text(self.cursor_version, where="cursor version")
        flag(self.base_fence_required, where="base fence")
        if self.base_fence_required != 1:
            raise D03ContractError("base fence 不得缺失")
        positive(self.logical_shard_count, where="logical shard count")
        if self.allowed_worker_counts != (1, 2, 4):
            raise D03ContractError("worker counts 必须严格为 1/2/4")
        text(self.merge_barrier_key, where="merge barrier")
        if self.failure_point_keys != REQUIRED_FAILURE_POINTS:
            raise D03ContractError("failure points 不完整或乱序")
        flag(self.fresh_resume_equivalent, where="fresh/resume equivalence")
        if self.fresh_resume_equivalent != 1:
            raise D03ContractError("fresh/resume 必须同起点同后缀")

    def to_dict(self) -> dict[str, Any]:
        """导出阶段恢复绑定。"""
        return {
            "allowed_worker_counts": list(self.allowed_worker_counts),
            "base_fence_required": self.base_fence_required,
            "cursor_version": self.cursor_version,
            "failure_point_keys": list(self.failure_point_keys),
            "fresh_resume_equivalent": self.fresh_resume_equivalent,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "run_id_policy": self.run_id_policy,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StageRecoveryBinding":
        """从严格 object 恢复阶段恢复绑定。"""
        raw = exact_dict(value, {
            "allowed_worker_counts", "base_fence_required", "cursor_version",
            "failure_point_keys", "fresh_resume_equivalent",
            "logical_shard_count", "merge_barrier_key", "run_id_policy",
        }, where="StageRecoveryBinding")
        workers = raw["allowed_worker_counts"]
        if (not isinstance(workers, list)
                or any(type(item) is not int for item in workers)):
            raise D03ContractError("worker counts 类型非法")
        return cls(
            str(raw["run_id_policy"]), str(raw["cursor_version"]),
            raw["base_fence_required"], raw["logical_shard_count"],
            tuple(workers), str(raw["merge_barrier_key"]),
            string_tuple(raw["failure_point_keys"], where="failure points"),
            raw["fresh_resume_equivalent"],
        )


@dataclass(frozen=True)
class D03StageManifest:
    """合取一个阶段的身份、可见性、评测、预算、恢复和零执行状态。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    release_key: str
    stage_identity: D03StageIdentity
    data_visibility: StageDataVisibility
    evaluation_binding: StageEvaluationBinding
    resource_budget: StageResourceBudget
    recovery_binding: StageRecoveryBinding
    execution_state: dict[str, int]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise D03ContractError("stage format_version 非法")
        if self.artifact_kind != STAGE_ARTIFACT_KIND:
            raise D03ContractError("stage artifact_kind 非法")
        text(self.artifact_version, where="stage artifact version")
        text(self.release_key, where="stage release key")
        for name, expected in (
                ("stage_identity", D03StageIdentity),
                ("data_visibility", StageDataVisibility),
                ("evaluation_binding", StageEvaluationBinding),
                ("resource_budget", StageResourceBudget),
                ("recovery_binding", StageRecoveryBinding)):
            if not isinstance(getattr(self, name), expected):
                raise D03ContractError(f"stage {name} 类型非法")
        object.__setattr__(self, "execution_state", validate_zero_execution_state(
            self.execution_state))

    def to_dict(self) -> dict[str, Any]:
        """导出规范阶段 manifest。"""
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "data_visibility": self.data_visibility.to_dict(),
            "evaluation_binding": self.evaluation_binding.to_dict(),
            "execution_state": dict(self.execution_state),
            "format_version": self.format_version,
            "recovery_binding": self.recovery_binding.to_dict(),
            "release_key": self.release_key,
            "resource_budget": self.resource_budget.to_dict(),
            "stage_identity": self.stage_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "D03StageManifest":
        """从严格 object 恢复阶段 manifest。"""
        raw = exact_dict(value, {
            "artifact_kind", "artifact_version", "data_visibility",
            "evaluation_binding", "execution_state", "format_version",
            "recovery_binding", "release_key", "resource_budget",
            "stage_identity",
        }, where="D03StageManifest")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["release_key"]),
            D03StageIdentity.from_dict(raw["stage_identity"]),
            StageDataVisibility.from_dict(raw["data_visibility"]),
            StageEvaluationBinding.from_dict(raw["evaluation_binding"]),
            StageResourceBudget.from_dict(raw["resource_budget"]),
            StageRecoveryBinding.from_dict(raw["recovery_binding"]),
            raw["execution_state"],
        )


def validate_stage_manifest_set(stages: tuple[D03StageManifest, ...]) -> None:
    """要求唯一九阶段按严格顺序且共享同一 D-03 release key。"""
    if (not isinstance(stages, tuple) or len(stages) != len(STAGE_KEYS)
            or any(not isinstance(item, D03StageManifest) for item in stages)):
        raise D03ContractError("D-03 必须包含唯一九阶段")
    keys = tuple(item.stage_identity.stage_key for item in stages)
    if keys != STAGE_KEYS:
        raise D03ContractError("D-03 九阶段顺序漂移")
    if len({item.release_key for item in stages}) != 1:
        raise D03ContractError("D-03 九阶段 release identity 不一致")


__all__ = [
    "AGGREGATION_POLICY",
    "D03StageIdentity",
    "D03StageManifest",
    "EvaluationThreshold",
    "OWNER_KEYS",
    "REQUIRED_FAILURE_POINTS",
    "RUN_ID_POLICY",
    "STAGE_ARTIFACT_KIND",
    "StageDataVisibility",
    "StageEvaluationBinding",
    "StageRecoveryBinding",
    "StageResourceBudget",
    "validate_stage_manifest_set",
]
