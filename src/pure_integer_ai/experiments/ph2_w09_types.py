"""W-09 窗口、三向消费、维度、消融和恢复的一等类型。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_ALL_DIMENSION_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
    W09_FAILURE_POINT_KEYS,
    W09_RESOURCE_BUDGET,
    W09_STOP_STATES,
)


class W09TypeError(RuntimeError):
    """W-09 typed 对象字段、身份或方向发生漂移。"""


class TeacherExitPhase(str, Enum):
    TRAINING_MATERIAL_SOURCE = "TRAINING_MATERIAL_SOURCE"
    DEV_CALIBRATION_ONLY = "DEV_CALIBRATION_ONLY"
    SHADOW_ERROR_ONLY = "SHADOW_ERROR_ONLY"
    ZERO_CALL_WINDOW = "ZERO_CALL_WINDOW"


class W09ResultState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NE = "NE"


def _sha256(value: object, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W09TypeError(f"{where} 不是 SHA-256")
    return value


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != 32
        or any(type(item) is not int or not 0 <= item <= 255 for item in value)
    ):
        raise W09TypeError(f"{where} 不是稳定整数键")
    return value


def _result_state(value: object, *, where: str) -> W09ResultState:
    try:
        return W09ResultState(value)
    except (TypeError, ValueError) as error:
        raise W09TypeError(f"{where} 结果状态非法") from error


@dataclass(frozen=True)
class W09ResourceAudit:
    """单个窗口的逐项资源用量和冻结上限。"""

    used: tuple[tuple[str, int], ...]
    limits: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if dict(self.limits) != W09_RESOURCE_BUDGET:
            raise W09TypeError("W-09 resource limits 漂移")
        if tuple(sorted(self.used)) != self.used or len(dict(self.used)) != len(self.used):
            raise W09TypeError("W-09 resource audit 必须规范排序且字段唯一")
        used = dict(self.used)
        if set(used) != set(W09_RESOURCE_BUDGET):
            raise W09TypeError("W-09 resource audit 字段不完整")
        if any(
            type(value) is not int or value < 0 or value > W09_RESOURCE_BUDGET[key]
            for key, value in used.items()
        ):
            raise W09TypeError("W-09 resource audit 超限或类型非法")

    @classmethod
    def zero(cls) -> "W09ResourceAudit":
        return cls(
            tuple((key, 0) for key in sorted(W09_RESOURCE_BUDGET)),
            tuple(sorted(W09_RESOURCE_BUDGET.items())),
        )

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({"limits": dict(self.limits), "used": dict(self.used)})


@dataclass(frozen=True)
class W09WindowIdentity:
    """一个 teacher-exit 窗口的不可混合输入、输出与审计身份。"""

    phase: TeacherExitPhase
    window_ordinal: int
    input_commitment: str
    candidate_identity: str
    teacher_call_count: int
    consumer_output_commitments: tuple[tuple[str, str], ...]
    resource_audit: W09ResourceAudit
    rollback_audit_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.phase, TeacherExitPhase):
            raise W09TypeError("W-09 teacher exit phase 类型非法")
        if type(self.window_ordinal) is not int or not 1 <= self.window_ordinal <= 3:
            raise W09TypeError("W-09 window ordinal 必须在 1..3")
        _sha256(self.input_commitment, where="window input")
        _sha256(self.candidate_identity, where="window candidate")
        _sha256(self.rollback_audit_sha256, where="window rollback audit")
        if type(self.teacher_call_count) is not int or self.teacher_call_count < 0:
            raise W09TypeError("W-09 teacher call counter 非法")
        if self.phase is TeacherExitPhase.ZERO_CALL_WINDOW and self.teacher_call_count != 0:
            raise W09TypeError("W-09 zero-call window 发生 teacher call")
        if not isinstance(self.resource_audit, W09ResourceAudit):
            raise W09TypeError("W-09 window 缺少 resource audit")
        if tuple(key for key, _ in self.consumer_output_commitments) != W09_CONSUMER_KEYS:
            raise W09TypeError("W-09 window 必须独立绑定 U/R/G 输出")
        for key, commitment in self.consumer_output_commitments:
            if key not in W09_CONSUMER_KEYS:
                raise W09TypeError("W-09 window consumer 非法")
            _sha256(commitment, where=f"{key} output")
        if len({value for _, value in self.consumer_output_commitments}) != 3:
            raise W09TypeError("W-09 window 不得共享三向输出身份")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "candidate_identity": self.candidate_identity,
            "consumer_output_commitments": dict(self.consumer_output_commitments),
            "input_commitment": self.input_commitment,
            "phase": self.phase.value,
            "resource_audit": list(self.resource_audit.stable_key()),
            "rollback_audit_sha256": self.rollback_audit_sha256,
            "teacher_call_count": self.teacher_call_count,
            "window_ordinal": self.window_ordinal,
        })


@dataclass(frozen=True)
class W09ConsumerRequest:
    consumer_key: str
    request_key: tuple[int, ...]
    input_commitment: str

    def __post_init__(self) -> None:
        if self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09TypeError("W-09 consumer request direction 非法")
        _key(self.request_key, where="consumer request")
        _sha256(self.input_commitment, where="consumer input")


@dataclass(frozen=True)
class W09ConsumerChoice:
    consumer_key: str
    request_key: tuple[int, ...]
    choice_key: tuple[int, ...]
    selected_candidate_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09TypeError("W-09 consumer choice direction 非法")
        _key(self.request_key, where="choice request")
        _key(self.choice_key, where="choice")
        _key(self.selected_candidate_key, where="selected candidate")


@dataclass(frozen=True)
class W09UseOutcome:
    consumer_key: str
    request_key: tuple[int, ...]
    choice_key: tuple[int, ...]
    selected_candidate_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]
    stop_state: str

    def __post_init__(self) -> None:
        if self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09TypeError("W-09 Use/outcome direction 非法")
        for name in (
            "request_key",
            "choice_key",
            "selected_candidate_key",
            "use_key",
            "outcome_key",
        ):
            _key(getattr(self, name), where=name)
        if self.stop_state not in W09_STOP_STATES:
            raise W09TypeError("W-09 Use/outcome stop state 非法")


@dataclass(frozen=True)
class W09VerifierResult:
    consumer_key: str
    request_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]
    verifier_key: tuple[int, ...]
    status: W09ResultState
    failure_kind: str

    def __post_init__(self) -> None:
        if self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09TypeError("W-09 verifier direction 非法")
        for name in ("request_key", "use_key", "outcome_key", "verifier_key"):
            _key(getattr(self, name), where=name)
        _result_state(self.status, where="verifier")
        if not isinstance(self.failure_kind, str) or not self.failure_kind:
            raise W09TypeError("W-09 verifier failure kind 非法")
        if self.status is W09ResultState.PASS and self.failure_kind != "NONE":
            raise W09TypeError("W-09 PASS verifier 不得携带 failure")
        if self.status is not W09ResultState.PASS and self.failure_kind == "NONE":
            raise W09TypeError("W-09 FAIL/NE verifier 必须携带枚举 failure")


@dataclass(frozen=True)
class W09DirectionalResult:
    request: W09ConsumerRequest
    choice: W09ConsumerChoice
    use_outcome: W09UseOutcome
    verifier: W09VerifierResult

    def __post_init__(self) -> None:
        if not all(isinstance(item, expected) for item, expected in (
            (self.request, W09ConsumerRequest),
            (self.choice, W09ConsumerChoice),
            (self.use_outcome, W09UseOutcome),
            (self.verifier, W09VerifierResult),
        )):
            raise W09TypeError("W-09 directional result typed object 不完整")
        consumers = {
            self.request.consumer_key,
            self.choice.consumer_key,
            self.use_outcome.consumer_key,
            self.verifier.consumer_key,
        }
        if len(consumers) != 1:
            raise W09TypeError("W-09 directional result 混合了 consumer")
        if not (
            self.request.request_key
            == self.choice.request_key
            == self.use_outcome.request_key
            == self.verifier.request_key
            and self.choice.choice_key == self.use_outcome.choice_key
            and self.choice.selected_candidate_key
            == self.use_outcome.selected_candidate_key
            and self.use_outcome.use_key == self.verifier.use_key
            and self.use_outcome.outcome_key == self.verifier.outcome_key
        ):
            raise W09TypeError("W-09 directional result 引用链不闭合")


@dataclass(frozen=True)
class W09DimensionResult:
    dimension_key: str
    window_ordinal: int
    directional_results: tuple[W09DirectionalResult, ...]
    status: W09ResultState

    def __post_init__(self) -> None:
        if self.dimension_key not in W09_DIMENSION_KEYS:
            raise W09TypeError("W-09 dimension key 非法")
        if type(self.window_ordinal) is not int or not 1 <= self.window_ordinal <= 3:
            raise W09TypeError("W-09 dimension window 非法")
        if any(
            not isinstance(item, W09DirectionalResult)
            for item in self.directional_results
        ):
            raise W09TypeError("W-09 dimension directional result 类型非法")
        if tuple(item.request.consumer_key for item in self.directional_results) != W09_CONSUMER_KEYS:
            raise W09TypeError("W-09 dimension 必须独立持有 U/R/G result")
        request_keys = {item.request.request_key for item in self.directional_results}
        choice_keys = {item.choice.choice_key for item in self.directional_results}
        use_keys = {item.use_outcome.use_key for item in self.directional_results}
        outcome_keys = {item.use_outcome.outcome_key for item in self.directional_results}
        if any(len(keys) != 3 for keys in (request_keys, choice_keys, use_keys, outcome_keys)):
            raise W09TypeError("W-09 dimension 三向结果共享身份")
        states = tuple(item.verifier.status for item in self.directional_results)
        expected = (
            W09ResultState.PASS
            if all(item is W09ResultState.PASS for item in states)
            else W09ResultState.FAIL
            if any(item is W09ResultState.FAIL for item in states)
            else W09ResultState.NE
        )
        if self.status is not expected:
            raise W09TypeError("W-09 dimension status 不是三向硬合取")


@dataclass(frozen=True)
class W09AblationResult:
    ablation_key: str
    target_dimension_key: str
    component_disabled: int
    dimension_statuses: tuple[tuple[str, W09ResultState], ...]
    status: W09ResultState

    def __post_init__(self) -> None:
        if self.ablation_key not in W09_ABLATION_KEYS:
            raise W09TypeError("W-09 ablation key 非法")
        if self.target_dimension_key not in W09_ALL_DIMENSION_KEYS:
            raise W09TypeError("W-09 ablation target 非法")
        if self.ablation_key != f"{self.target_dimension_key}-ABLATION":
            raise W09TypeError("W-09 ablation key 与 target 不一致")
        if self.component_disabled != 1:
            raise W09TypeError("W-09 ablation 未真实禁用 component")
        if tuple(key for key, _ in self.dimension_statuses) != W09_ALL_DIMENSION_KEYS:
            raise W09TypeError("W-09 ablation 必须逐维发布状态")
        statuses = {key: _result_state(value, where=key) for key, value in self.dimension_statuses}
        bearing_others_pass = all(
            statuses[key] is W09ResultState.PASS
            for key in W09_DIMENSION_KEYS
            if key != self.target_dimension_key
        )
        target_state = statuses[self.target_dimension_key]
        if self.target_dimension_key not in W09_DIMENSION_KEYS and target_state is W09ResultState.NE:
            expected = W09ResultState.NE
        elif target_state is W09ResultState.FAIL and bearing_others_pass:
            expected = W09ResultState.PASS
        else:
            expected = W09ResultState.FAIL
        if self.status is not expected:
            raise W09TypeError("W-09 ablation status 不是目标击穿/非目标保持")


@dataclass(frozen=True)
class W09StopDecision:
    stop_state: str
    failure_kind: str
    resource_audit: W09ResourceAudit
    publication_allowed: int

    def __post_init__(self) -> None:
        if self.stop_state not in W09_STOP_STATES:
            raise W09TypeError("W-09 stop state 非法")
        if not isinstance(self.failure_kind, str) or not self.failure_kind:
            raise W09TypeError("W-09 stop failure kind 非法")
        if not isinstance(self.resource_audit, W09ResourceAudit):
            raise W09TypeError("W-09 stop 缺少 resource audit")
        if self.publication_allowed not in {0, 1}:
            raise W09TypeError("W-09 publication flag 非法")
        if self.publication_allowed and self.stop_state != "RESOLVED":
            raise W09TypeError("W-09 非 RESOLVED 不得发布")
        if self.stop_state == "RESOLVED" and self.failure_kind != "NONE":
            raise W09TypeError("W-09 RESOLVED 不得携带 failure")
        if self.stop_state != "RESOLVED" and self.failure_kind == "NONE":
            raise W09TypeError("W-09 非 RESOLVED 必须携带 failure")


@dataclass(frozen=True)
class W09RollbackReceipt:
    failure_point_key: str
    base_identity: str
    preview_identity: str
    restored_identity: str
    leaked_write_count: int

    def __post_init__(self) -> None:
        if self.failure_point_key not in W09_FAILURE_POINT_KEYS:
            raise W09TypeError("W-09 rollback failure point 非法")
        _sha256(self.base_identity, where="rollback base")
        _sha256(self.preview_identity, where="rollback preview")
        _sha256(self.restored_identity, where="rollback restored")
        if self.restored_identity != self.base_identity or self.leaked_write_count != 0:
            raise W09TypeError("W-09 rollback 未恢复 base 或发生泄漏写")


@dataclass(frozen=True)
class W09CloneReceipt:
    source_identity: str
    clone_identity: str
    pre_learning_output_identity: str
    post_learning_output_identity: str
    isolated_learning_write_count: int
    source_write_count: int

    def __post_init__(self) -> None:
        for name in (
            "source_identity",
            "clone_identity",
            "pre_learning_output_identity",
            "post_learning_output_identity",
        ):
            _sha256(getattr(self, name), where=name)
        if self.source_identity == self.clone_identity:
            raise W09TypeError("W-09 clone 必须具有独立 identity")
        if type(self.isolated_learning_write_count) is not int or self.isolated_learning_write_count <= 0:
            raise W09TypeError("W-09 clone 未发生隔离后续学习")
        if self.source_write_count != 0:
            raise W09TypeError("W-09 clone 学习污染 source")
        if self.pre_learning_output_identity == self.post_learning_output_identity:
            raise W09TypeError("W-09 clone 后续学习未改变 clone output")


__all__ = [
    "TeacherExitPhase",
    "W09AblationResult",
    "W09CloneReceipt",
    "W09ConsumerChoice",
    "W09ConsumerRequest",
    "W09DimensionResult",
    "W09DirectionalResult",
    "W09ResourceAudit",
    "W09ResultState",
    "W09RollbackReceipt",
    "W09StopDecision",
    "W09TypeError",
    "W09UseOutcome",
    "W09VerifierResult",
    "W09WindowIdentity",
]
