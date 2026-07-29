"""W-02 四 bearing dimension、正交消融和公开安全摘要合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NE = "NE"
STATUSES = (STATUS_FAIL, STATUS_NE, STATUS_PASS)

GENERATION_HARD_CONJUNCT = "W-02-GENERATION-HARD-CONJUNCT"
W02_BEARING_DIMENSIONS = (
    "W-02-BOUNDARY_WITHDRAWAL",
    "W-02-MULTI_CANDIDATE",
    "W-02-NEW_CONTENT_MORPHOLOGY",
    "W-02-OOV",
)
W02_EVALUATION_ORDER = (*W02_BEARING_DIMENSIONS, GENERATION_HARD_CONJUNCT)
W02_ABLATION_ORDER = tuple(
    f"{item}-ABLATION" for item in W02_BEARING_DIMENSIONS)


class W02EvaluationError(RuntimeError):
    """W-02 evaluator 阈值、消融、隔离或公开摘要合同损坏。"""


def _sha256(value: object, *, where: str) -> str:
    """要求公开 evidence identity 是规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W02EvaluationError(f"{where} 必须是小写 SHA-256")
    return value


def _count(value: object, *, where: str, positive: bool = False) -> int:
    """要求计数为非负或正严格整数。"""
    if type(value) is not int or value < int(positive):
        raise W02EvaluationError(f"{where} 计数非法")
    return value


def _status(value: object, *, where: str) -> str:
    """要求状态严格落在 PASS/FAIL/NE。"""
    if value not in STATUSES:
        raise W02EvaluationError(f"{where} 状态非法")
    assert isinstance(value, str)
    return value


@dataclass(frozen=True)
class W02DimensionResult:
    """一个 1/1、fail=0、NE=BLOCK 维度的公开计数和证据摘要。"""

    dimension_key: str
    status: str
    pass_count: int
    case_count: int
    fail_count: int
    ne_count: int
    evidence_sha256: str
    min_pass_numerator: int = 1
    min_pass_denominator: int = 1
    max_fail_count: int = 0
    ne_policy: str = "BLOCK"

    def __post_init__(self) -> None:
        if self.dimension_key not in W02_EVALUATION_ORDER:
            raise W02EvaluationError("dimension key 未预注册")
        status = _status(self.status, where=self.dimension_key)
        for name in ("pass_count", "case_count", "fail_count", "ne_count"):
            _count(getattr(self, name), where=f"dimension.{name}")
        if self.case_count != self.pass_count + self.fail_count + self.ne_count:
            raise W02EvaluationError("dimension case 计数不闭合")
        if self.case_count <= 0:
            raise W02EvaluationError("bearing/generation dimension 不得无 case")
        if (self.min_pass_numerator != 1 or self.min_pass_denominator != 1
                or self.max_fail_count != 0 or self.ne_policy != "BLOCK"):
            raise W02EvaluationError("D-03 W-02 threshold 被改写")
        expected = (
            STATUS_FAIL if self.fail_count
            else STATUS_NE if self.ne_count
            else STATUS_PASS if self.pass_count == self.case_count
            else STATUS_FAIL
        )
        if status != expected:
            raise W02EvaluationError("dimension status 与 1/1 合取计数不一致")
        _sha256(self.evidence_sha256, where="dimension evidence")

    def to_public_dict(self) -> dict[str, Any]:
        """只导出状态、计数、阈值和 evidence digest。"""
        return {
            "case_count": self.case_count,
            "dimension_key": self.dimension_key,
            "evidence_sha256": self.evidence_sha256,
            "fail_count": self.fail_count,
            "max_fail_count": self.max_fail_count,
            "min_pass_denominator": self.min_pass_denominator,
            "min_pass_numerator": self.min_pass_numerator,
            "ne_count": self.ne_count,
            "ne_policy": self.ne_policy,
            "pass_count": self.pass_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class W02AblationResult:
    """一个预注册消融对四 bearing dimension 和生成硬合取的状态矩阵。"""

    ablation_key: str
    targeted_dimension: str
    dimension_statuses: tuple[tuple[str, str], ...]
    generation_status: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.ablation_key not in W02_ABLATION_ORDER:
            raise W02EvaluationError("ablation key 未预注册")
        position = W02_ABLATION_ORDER.index(self.ablation_key)
        if self.targeted_dimension != W02_BEARING_DIMENSIONS[position]:
            raise W02EvaluationError("ablation 与目标 bearing dimension 错位")
        if (not isinstance(self.dimension_statuses, tuple)
                or tuple(item[0] for item in self.dimension_statuses)
                != W02_BEARING_DIMENSIONS):
            raise W02EvaluationError("ablation dimension 顺序漂移")
        if any(
                not isinstance(item, tuple) or len(item) != 2
                or item[0] not in W02_BEARING_DIMENSIONS
                or item[1] not in STATUSES
                for item in self.dimension_statuses):
            raise W02EvaluationError("ablation dimension status 非法")
        _status(self.generation_status, where="ablation generation")
        _sha256(self.evidence_sha256, where="ablation evidence")

    def status_for(self, dimension: str) -> str:
        """读取某 bearing dimension 在本消融下的状态。"""
        return dict(self.dimension_statuses)[dimension]

    def to_public_dict(self) -> dict[str, Any]:
        """导出无 private case 内容的消融矩阵。"""
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


@dataclass(frozen=True)
class W02PrivateEvaluationReport:
    """private evaluator 唯一允许离开 clone 的安全结果。"""

    status: str
    dimensions: tuple[W02DimensionResult, ...]
    generation: W02DimensionResult
    ablations: tuple[W02AblationResult, ...]
    generation_consumer_disabled_status: str
    host_digests_before: dict[str, str]
    host_digests_after: dict[str, str]
    private_path_reads: int
    private_payload_bytes: int
    held_out_observation_reads: int
    evaluator_label_reads: int
    ud_observation_reads: int
    evaluator_label_writes: int
    host_write_count: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        _status(self.status, where="evaluation")
        if tuple(item.dimension_key for item in self.dimensions) != (
                W02_BEARING_DIMENSIONS):
            raise W02EvaluationError("bearing dimension 顺序或集合漂移")
        if self.generation.dimension_key != GENERATION_HARD_CONJUNCT:
            raise W02EvaluationError("生成附加硬合取缺失")
        if tuple(item.ablation_key for item in self.ablations) != (
                W02_ABLATION_ORDER):
            raise W02EvaluationError("四项 ablation 顺序或集合漂移")
        _status(
            self.generation_consumer_disabled_status,
            where="generation consumer disabled",
        )
        expected_keys = {"core", "cursor", "logical", "memory", "use"}
        if (not isinstance(self.host_digests_before, dict)
                or not isinstance(self.host_digests_after, dict)
                or set(self.host_digests_before) != expected_keys
                or set(self.host_digests_after) != expected_keys):
            raise W02EvaluationError("host digest 维度不完整")
        for value in (*self.host_digests_before.values(),
                      *self.host_digests_after.values()):
            _sha256(value, where="host digest")
        if self.host_digests_before != self.host_digests_after:
            raise W02EvaluationError("private evaluator 改变了 host digest")
        for name, positive in (
                ("private_path_reads", True),
                ("private_payload_bytes", True),
                ("held_out_observation_reads", True),
                ("evaluator_label_reads", True),
                ("ud_observation_reads", True),
                ("evaluator_label_writes", False),
                ("host_write_count", False)):
            _count(getattr(self, name), where=name, positive=positive)
        if self.private_path_reads != 7:
            raise W02EvaluationError("W-02 evaluator 必须恰读 7 条 private path")
        if self.evaluator_label_writes != 0 or self.host_write_count != 0:
            raise W02EvaluationError("evaluator label/host write 必须为 0")
        _sha256(self.evidence_sha256, where="evaluation evidence")

    def to_public_dict(self) -> dict[str, Any]:
        """导出严格不含 private Observation、label 或 expected surface 的报告。"""
        return {
            "ablations": [item.to_public_dict() for item in self.ablations],
            "aggregation_policy": "ALL_BEARING_DIMENSIONS_MUST_PASS",
            "dimensions": [item.to_public_dict() for item in self.dimensions],
            "evidence_sha256": self.evidence_sha256,
            "format_version": 1,
            "generation": self.generation.to_public_dict(),
            "generation_consumer_disabled_status": (
                self.generation_consumer_disabled_status),
            "host_digests_after": dict(sorted(self.host_digests_after.items())),
            "host_digests_before": dict(sorted(self.host_digests_before.items())),
            "host_write_count": self.host_write_count,
            "private_reads": {
                "evaluator_label_reads": self.evaluator_label_reads,
                "held_out_observation_reads": self.held_out_observation_reads,
                "private_path_reads": self.private_path_reads,
                "private_payload_bytes": self.private_payload_bytes,
                "ud_observation_reads": self.ud_observation_reads,
            },
            "status": self.status,
        }


def aggregate_w02_evaluation(
        *,
        dimensions: tuple[W02DimensionResult, ...],
        generation: W02DimensionResult,
        ablations: tuple[W02AblationResult, ...],
        generation_consumer_disabled_status: str,
        host_digests_before: dict[str, str],
        host_digests_after: dict[str, str],
        private_path_reads: int,
        private_payload_bytes: int,
        held_out_observation_reads: int,
        evaluator_label_reads: int,
        ud_observation_reads: int,
        evaluator_label_writes: int,
        host_write_count: int,
        evidence_sha256: str,
        ) -> W02PrivateEvaluationReport:
    """按 fail 优先、NE 次之的硬合取形成安全报告。"""
    if tuple(item.dimension_key for item in dimensions) != (
            W02_BEARING_DIMENSIONS):
        raise W02EvaluationError("bearing dimension 输入漂移")
    if generation.dimension_key != GENERATION_HARD_CONJUNCT:
        raise W02EvaluationError("generation 输入不是附加硬合取")
    baseline = {item.dimension_key: item.status for item in dimensions}
    all_baseline_pass = all(item == STATUS_PASS for item in baseline.values())
    if tuple(item.ablation_key for item in ablations) != W02_ABLATION_ORDER:
        raise W02EvaluationError("ablation 输入顺序漂移")
    for item in ablations:
        if all_baseline_pass:
            if item.status_for(item.targeted_dimension) == STATUS_PASS:
                raise W02EvaluationError("ablation 未击穿目标 bearing dimension")
            for dimension, status in item.dimension_statuses:
                if dimension != item.targeted_dimension and status != baseline[dimension]:
                    raise W02EvaluationError("ablation 非正交地击穿其他 bearing dimension")
    statuses = [item.status for item in dimensions] + [generation.status]
    if STATUS_FAIL in statuses:
        overall = STATUS_FAIL
    elif STATUS_NE in statuses:
        overall = STATUS_NE
    elif (generation_consumer_disabled_status != STATUS_FAIL
          or any(item.status_for(item.targeted_dimension) == STATUS_PASS
                 for item in ablations)):
        overall = STATUS_FAIL
    else:
        overall = STATUS_PASS
    return W02PrivateEvaluationReport(
        overall,
        dimensions,
        generation,
        ablations,
        generation_consumer_disabled_status,
        dict(host_digests_before),
        dict(host_digests_after),
        private_path_reads,
        private_payload_bytes,
        held_out_observation_reads,
        evaluator_label_reads,
        ud_observation_reads,
        evaluator_label_writes,
        host_write_count,
        evidence_sha256,
    )


__all__ = [
    "GENERATION_HARD_CONJUNCT",
    "STATUS_FAIL",
    "STATUS_NE",
    "STATUS_PASS",
    "W02_ABLATION_ORDER",
    "W02_BEARING_DIMENSIONS",
    "W02_EVALUATION_ORDER",
    "W02AblationResult",
    "W02DimensionResult",
    "W02EvaluationError",
    "W02PrivateEvaluationReport",
    "aggregate_w02_evaluation",
]
