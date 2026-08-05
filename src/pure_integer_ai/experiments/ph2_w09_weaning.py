"""W-09 typed 教师退出协议及其可执行、失败关闭 runtime。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Callable

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_RESOURCE_BUDGET,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    W09_DEV_OWNER,
    W09FrozenContract,
    W09_OWNER_KEY,
    W09_TRAINING_MATERIAL_OWNER,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09TrainingPayload
from pure_integer_ai.experiments.ph2_dataset_owner_records import TeacherEvidenceRecord
from pure_integer_ai.experiments.ph2_w09_types import (
    TeacherExitPhase,
    W09WindowIdentity,
)


W09_TYPED_WEANING_VERSION = 1
W09_ZERO_CALL_WINDOWS_REQUIRED = 3
W09_TYPED_WEANING_BLOCKER = "W-09_typed_weaning_protocol_missing"
W09_ZERO_CALL_WINDOWS_PENDING = "W-09_zero_call_windows_pending"
W09_TYPED_WEANING_READY = "W-09_typed_weaning_ready"


class W09WeaningError(RuntimeError):
    """W-09 typed 断奶协议、phase 或审计输入不合法。"""


def _sha256(value: object, *, where: str) -> str:
    """把稳定整数身份编码成 SHA-256，并拒绝非公开可审计输入。"""
    try:
        key = digest_value(_commitment_value(value))
    except (TypeError, ValueError, OverflowError) as error:
        raise W09WeaningError(f"{where} commitment is invalid") from error
    return hashlib.sha256(bytes(key)).hexdigest()


def _commitment_value(value: object) -> object:
    """把 typed 对象投影为只含稳定键的 canonical JSON 值。"""
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return value
    if isinstance(value, Enum):
        return value.value
    stable = getattr(value, "stable_key", None)
    if callable(stable):
        return {"stable_key": list(stable())}
    stable_reader = getattr(stable, "stable_key", None)
    if callable(stable_reader):
        return {"stable_key": list(stable_reader())}
    if isinstance(stable, tuple):
        return {"stable_key": list(stable)}
    if isinstance(value, dict):
        return {str(key): _commitment_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_commitment_value(item) for item in value]
    raise W09WeaningError("W-09 commitment contains an unstable object")


def _require_sha(value: object, *, where: str) -> str:
    """校验一个已经冻结的十六进制 SHA-256 字符串。"""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W09WeaningError(f"{where} commitment is invalid")
    return value


def _int_tuple(value: object, *, where: str, nonempty: bool = True) -> tuple[int, ...]:
    """校验只含严格整数的稳定键。"""
    if not isinstance(value, tuple) or (nonempty and not value):
        raise W09WeaningError(f"{where} key is invalid")
    if any(type(item) is not int or item < 0 or item > 255 for item in value):
        raise W09WeaningError(f"{where} key is invalid")
    return value


def _state_key(owner: object, *, where: str) -> tuple[int, ...]:
    """读取注入 owner 的非空稳定键，阻断对象地址充当身份。"""
    reader = getattr(owner, "state_key", None)
    if callable(reader):
        key = reader()
        if isinstance(key, tuple) and key and all(type(item) is int for item in key):
            return key
    return digest_value((where, type(owner).__module__, type(owner).__qualname__))


def _call(owner: object, methods: tuple[str, ...], *args: object) -> object:
    """调用一个显式注入 owner；未知 owner 不允许静默 fallback。"""
    if callable(owner):
        return owner(*args)
    for name in methods:
        method = getattr(owner, name, None)
        if callable(method):
            return method(*args)
    raise W09WeaningError("W-09 injected phase owner protocol is incomplete")


def _counter(owner: object, names: tuple[str, ...]) -> int:
    """读取 owner 的严格非负调用计数；没有计数器时返回零。"""
    for name in names:
        value = getattr(owner, name, None)
        if callable(value):
            value = value()
        if value is not None:
            if type(value) is not int or value < 0:
                raise W09WeaningError("W-09 call counter is invalid")
            return value
    return 0


def _host_snapshot(ctx: object) -> object:
    """读取 host backend 的可比较快照，不执行任何写入。"""
    backend = getattr(ctx, "backend", None)
    snapshot = getattr(backend, "snapshot", None)
    if callable(snapshot):
        return snapshot()
    reader = getattr(ctx, "state_key", None)
    if callable(reader):
        return reader()
    return None


def _output_snapshot(ctx: object, report: object) -> object:
    """读取 production 输出身份；默认只读 stage4 report 的稳定键。"""
    readers = tuple(
        getattr(ctx, name, None)
        for name in (
            "w09_output_state_reader",
            "w09_choice_state_reader",
            "w09_use_outcome_state_reader",
            "w09_learning_state_reader",
        )
    )
    if any(callable(reader) for reader in readers):
        return tuple(reader() if callable(reader) else None for reader in readers)
    stable = getattr(report, "stable_key", None)
    if callable(stable):
        return stable()
    return _state_key(report, where="stage4 report")


def _forbidden_field(value: object) -> bool:
    """递归检查 teacher material 不得携带答案或 evaluator 字段。"""
    if isinstance(value, dict):
        for key, item in value.items():
            if any(token in str(key).lower() for token in ("expected", "label", "evaluator")):
                return True
            if _forbidden_field(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_forbidden_field(item) for item in value)
    return False


@dataclass(frozen=True)
class W09TypedWeaningProtocol:
    """冻结 W-09 phase 顺序、来源身份、阈值、资源和三个窗口输入。"""

    authority_sha256: str
    registry_identity: tuple[int, ...]
    candidate_identity: str
    input_commitment: str
    threshold_key: tuple[int, ...]
    resource_limits: tuple[tuple[str, int], ...] = tuple(sorted(W09_RESOURCE_BUDGET.items()))
    window_input_commitments: tuple[str, ...] = ()
    version: int = W09_TYPED_WEANING_VERSION
    owner_key: str = W09_OWNER_KEY
    window_count: int = W09_ZERO_CALL_WINDOWS_REQUIRED
    phase_order: tuple[TeacherExitPhase, ...] = (
        TeacherExitPhase.TRAINING_MATERIAL_SOURCE,
        TeacherExitPhase.DEV_CALIBRATION_ONLY,
        TeacherExitPhase.SHADOW_ERROR_ONLY,
        TeacherExitPhase.ZERO_CALL_WINDOW,
    )

    def __post_init__(self) -> None:
        """校验协议身份和不可变执行边界。"""
        _require_sha(self.authority_sha256, where="authority")
        _require_sha(self.candidate_identity, where="candidate")
        _require_sha(self.input_commitment, where="input")
        _int_tuple(self.registry_identity, where="registry")
        _int_tuple(self.threshold_key, where="threshold")
        if type(self.version) is not int or self.version != W09_TYPED_WEANING_VERSION:
            raise W09WeaningError("W-09 typed weaning version drifted")
        if self.owner_key != W09_OWNER_KEY:
            raise W09WeaningError("W-09 typed weaning owner drifted")
        if type(self.window_count) is not int or self.window_count != W09_ZERO_CALL_WINDOWS_REQUIRED:
            raise W09WeaningError("W-09 requires exactly three windows")
        if self.phase_order != (
            TeacherExitPhase.TRAINING_MATERIAL_SOURCE,
            TeacherExitPhase.DEV_CALIBRATION_ONLY,
            TeacherExitPhase.SHADOW_ERROR_ONLY,
            TeacherExitPhase.ZERO_CALL_WINDOW,
        ):
            raise W09WeaningError("W-09 phase order drifted")
        if self.resource_limits != tuple(sorted(W09_RESOURCE_BUDGET.items())):
            raise W09WeaningError("W-09 typed weaning resource limits drifted")
        commitments = self.window_input_commitments
        if not commitments:
            commitments = tuple(
                _sha256((self.input_commitment, ordinal), where="window")
                for ordinal in range(1, self.window_count + 1)
            )
            object.__setattr__(self, "window_input_commitments", commitments)
        for item in commitments:
            _require_sha(item, where="window input")
        if (len(commitments) != self.window_count
                or len(set(commitments)) != self.window_count):
            raise W09WeaningError("W-09 window input commitments are invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含全部冻结字段的协议整数身份。"""
        return digest_value({
            "authority_sha256": self.authority_sha256,
            "candidate_identity": self.candidate_identity,
            "input_commitment": self.input_commitment,
            "owner_key": self.owner_key,
            "phase_order": [item.value for item in self.phase_order],
            "registry_identity": list(self.registry_identity),
            "resource_limits": dict(self.resource_limits),
            "threshold_key": list(self.threshold_key),
            "version": self.version,
            "window_count": self.window_count,
            "window_input_commitments": list(self.window_input_commitments),
        })


@dataclass(frozen=True)
class W09PhaseAudit:
    """记录一个退出 phase 的只读、调用计数和独立输出审计。"""

    phase: TeacherExitPhase
    input_commitment: str
    output_commitment: str
    teacher_call_count: int = 0
    api_call_count: int = 0
    llm_call_count: int = 0
    host_write_count: int = 0
    candidate_write_count: int = 0
    production_output_changed: int = 0
    training_material_count: int = 0
    dev_read_count: int = 0
    shadow_error_count: int = 0
    shadow_audit_write_count: int = 0

    def __post_init__(self) -> None:
        """校验 phase 专属 owner 边界和零 live-call 不变量。"""
        if not isinstance(self.phase, TeacherExitPhase):
            raise W09WeaningError("W-09 phase audit type is invalid")
        _require_sha(self.input_commitment, where="phase input")
        _require_sha(self.output_commitment, where="phase output")
        counters = (
            self.teacher_call_count, self.api_call_count, self.llm_call_count,
            self.host_write_count, self.candidate_write_count,
            self.production_output_changed, self.training_material_count,
            self.dev_read_count, self.shadow_error_count,
            self.shadow_audit_write_count,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise W09WeaningError("W-09 phase audit counter is invalid")
        if any(value != 0 for value in (
                self.teacher_call_count, self.api_call_count, self.llm_call_count,
                self.host_write_count, self.candidate_write_count)):
            raise W09WeaningError("W-09 phase performed forbidden live call or host write")
        if self.production_output_changed:
            raise W09WeaningError("W-09 phase changed production output")
        if self.phase is TeacherExitPhase.TRAINING_MATERIAL_SOURCE:
            if self.training_material_count <= 0 or any(
                    value != 0 for value in (
                        self.dev_read_count, self.shadow_error_count,
                        self.shadow_audit_write_count)):
                raise W09WeaningError("W-09 training-material phase is incomplete")
        elif self.phase is TeacherExitPhase.DEV_CALIBRATION_ONLY:
            if self.dev_read_count <= 0 or any(
                    value != 0 for value in (
                        self.training_material_count, self.shadow_error_count)):
                raise W09WeaningError("W-09 dev phase is not read-only or incomplete")
        elif self.phase is TeacherExitPhase.SHADOW_ERROR_ONLY:
            if self.shadow_error_count <= 0 or self.shadow_audit_write_count <= 0:
                raise W09WeaningError("W-09 shadow phase has no independent audit")
            if any(value != 0 for value in (
                    self.training_material_count, self.dev_read_count)):
                raise W09WeaningError("W-09 shadow phase consumed another phase input")
        elif self.phase is TeacherExitPhase.ZERO_CALL_WINDOW:
            raise W09WeaningError("zero-call windows require W09WindowIdentity")

    def stable_key(self) -> tuple[int, ...]:
        """返回不依赖对象地址的 phase 审计键。"""
        return digest_value({
            "api_call_count": self.api_call_count,
            "candidate_write_count": self.candidate_write_count,
            "dev_read_count": self.dev_read_count,
            "host_write_count": self.host_write_count,
            "input_commitment": self.input_commitment,
            "llm_call_count": self.llm_call_count,
            "output_commitment": self.output_commitment,
            "phase": self.phase.value,
            "production_output_changed": self.production_output_changed,
            "shadow_audit_write_count": self.shadow_audit_write_count,
            "shadow_error_count": self.shadow_error_count,
            "teacher_call_count": self.teacher_call_count,
            "training_material_count": self.training_material_count,
        })


@dataclass(frozen=True)
class W09WeaningReport:
    """汇总前三个退出 phase 和尚未完成的三个零调用窗口。"""

    protocol_key: tuple[int, ...]
    phase_audits: tuple[W09PhaseAudit, ...]
    windows: tuple[W09WindowIdentity, ...] = ()
    required_window_count: int = W09_ZERO_CALL_WINDOWS_REQUIRED
    ready: bool = False
    blockers: tuple[str, ...] = (W09_ZERO_CALL_WINDOWS_PENDING,)

    def __post_init__(self) -> None:
        """确保 phase 全覆盖、窗口连续且 ready 不能绕过零调用证据。"""
        _int_tuple(self.protocol_key, where="report protocol")
        if self.required_window_count != W09_ZERO_CALL_WINDOWS_REQUIRED:
            raise W09WeaningError("W-09 report window count drifted")
        expected = (
            TeacherExitPhase.TRAINING_MATERIAL_SOURCE,
            TeacherExitPhase.DEV_CALIBRATION_ONLY,
            TeacherExitPhase.SHADOW_ERROR_ONLY,
        )
        if tuple(item.phase for item in self.phase_audits) != expected:
            raise W09WeaningError("W-09 report must contain the three pre-window phases")
        if tuple(item.window_ordinal for item in self.windows) != tuple(
                range(1, len(self.windows) + 1)):
            raise W09WeaningError("W-09 window ordinals must be contiguous")
        if len(self.windows) > self.required_window_count:
            raise W09WeaningError("W-09 report has too many windows")
        if self.ready != (len(self.windows) == self.required_window_count):
            raise W09WeaningError("W-09 ready must be driven by all zero-call windows")
        if self.ready:
            if self.blockers:
                raise W09WeaningError("W-09 ready report cannot have blockers")
        elif W09_ZERO_CALL_WINDOWS_PENDING not in self.blockers:
            raise W09WeaningError("W-09 incomplete report must expose pending blocker")

    @property
    def complete(self) -> bool:
        """返回三个连续窗口是否都已完成。"""
        return self.ready

    def stable_key(self) -> tuple[int, ...]:
        """返回完整报告的稳定键。"""
        return digest_value({
            "blockers": list(self.blockers),
            "phase_audits": [item.stable_key() for item in self.phase_audits],
            "protocol_key": list(self.protocol_key),
            "ready": int(self.ready),
            "required_window_count": self.required_window_count,
            "windows": [item.stable_key() for item in self.windows],
        })


@dataclass
class W09FrozenTeacherEvidenceSource:
    """只暴露 W09-01 冻结 Teacher Evidence 的训练材料 reader。"""

    context: W09FrozenContract
    payload: W09TrainingPayload
    owner_key: str = W09_TRAINING_MATERIAL_OWNER
    read_count: int = 0

    def __post_init__(self) -> None:
        """确认 source 只能绑定 frozen contract 的 training Evidence。"""
        if not isinstance(self.context, W09FrozenContract):
            raise W09WeaningError("W-09 training source context is invalid")
        if not isinstance(self.payload, W09TrainingPayload):
            raise W09WeaningError("W-09 training source payload is invalid")
        if self.owner_key != W09_TRAINING_MATERIAL_OWNER:
            raise W09WeaningError("W-09 training source owner is invalid")
        if not self.payload.training_evidence:
            raise W09WeaningError("W-09 training source Evidence is empty")

    def read(self, _ctx: object, _report: object) -> tuple[object, ...]:
        """返回冻结 Evidence；不访问 teacher，也不写 host。"""
        self.read_count += 1
        return self.payload.training_evidence

    def state_key(self) -> tuple[int, ...]:
        """返回 source contract、Evidence 和只读次数的稳定键。"""
        return digest_value({
            "context": list(self.context.stable_key()),
            "evidence": [_commitment_value(item)
                         for item in self.payload.training_evidence],
            "owner_key": self.owner_key,
            "read_count": self.read_count,
        })


@dataclass
class W09DevCalibrationOwner:
    """只读消费 WIKIDATA dev registry 的校准 owner。"""

    context: W09FrozenContract
    items: tuple[object, ...]
    owner_key: str = W09_DEV_OWNER
    dev_read_count: int = 0

    def __post_init__(self) -> None:
        """确认 dev owner、pack registry 和输入非空。"""
        if not isinstance(self.context, W09FrozenContract):
            raise W09WeaningError("W-09 dev context is invalid")
        if self.owner_key != W09_DEV_OWNER or not isinstance(self.items, tuple) or not self.items:
            raise W09WeaningError("W-09 dev calibration owner is invalid")
        if self.context.dev_pack_keys != (
                "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1",):
            raise W09WeaningError("W-09 dev registry drifted")
        if any(_forbidden_field(
                item.to_dict() if callable(getattr(item, "to_dict", None)) else item)
                for item in self.items):
            raise W09WeaningError("W-09 dev calibration input contains answer fields")

    def calibrate(self, _ctx: object, _report: object) -> tuple[object, ...]:
        """读取 dev Observation 并返回校准材料，不写 Candidate 或 production host。"""
        self.dev_read_count += len(self.items)
        return self.items

    def state_key(self) -> tuple[int, ...]:
        """返回 dev owner 的 registry、输入和读取计数身份。"""
        return digest_value({
            "context": list(self.context.stable_key()),
            "dev_read_count": self.dev_read_count,
            "items": [_commitment_value(item) for item in self.items],
            "owner_key": self.owner_key,
        })


@dataclass
class W09ShadowErrorAudit:
    """只写独立 shadow error audit 的 owner，不反馈 candidate/output/learning。"""

    owner_key: str = "PH2_W09_SHADOW_ERROR_AUDIT_OWNER"
    records: list[str] | None = None

    def __post_init__(self) -> None:
        """初始化独立审计列表并拒绝伪装为 dev 或 candidate owner。"""
        if self.owner_key != "PH2_W09_SHADOW_ERROR_AUDIT_OWNER":
            raise W09WeaningError("W-09 shadow audit owner is invalid")
        if self.records is None:
            self.records = []

    def record(self, _ctx: object, errors: object) -> int:
        """只记录错误输入 commitment，不把错误改写为候选或输出。"""
        assert self.records is not None
        self.records.append(w09_commitment(errors))
        return 1

    @property
    def audit_write_count(self) -> int:
        """返回独立审计写入数量。"""
        return len(self.records or ())

    def state_key(self) -> tuple[int, ...]:
        """返回 shadow audit 的独立、可回读身份。"""
        return digest_value({
            "owner_key": self.owner_key,
            "records": tuple(self.records or ()),
        })


class W09TypedWeaningRuntime:
    """按冻结 phase 顺序执行训练材料、dev、shadow，并提供零调用窗口设施。"""

    def __init__(
            self,
            protocol: W09TypedWeaningProtocol,
            *,
            training_material_source: Any = None,
            dev_calibrator: Any = None,
            shadow_auditor: Any = None,
            frozen_contract: W09FrozenContract | None = None,
            teacher_call_counter: Callable[[object], int] | None = None,
            api_call_counter: Callable[[object], int] | None = None,
            llm_call_counter: Callable[[object], int] | None = None,
            host_state_reader: Callable[[object], object] | None = None,
            output_state_reader: Callable[[object], object] | None = None,
            ) -> None:
        """绑定四个 phase owner；缺少任一真实 caller 时运行必须 fail closed。"""
        if not isinstance(protocol, W09TypedWeaningProtocol):
            raise W09WeaningError("W-09 typed weaning protocol type is invalid")
        self.protocol = protocol
        if frozen_contract is not None:
            if (
                protocol.authority_sha256 != frozen_contract.authority_sha256
                or protocol.registry_identity != frozen_contract.stable_key()
                or protocol.owner_key != frozen_contract.owner_key
            ):
                raise W09WeaningError("W-09 authority or registry identity drifted")
        self.frozen_contract = frozen_contract
        self.training_material_source = training_material_source
        self.dev_calibrator = dev_calibrator
        self.shadow_auditor = shadow_auditor
        self._teacher_counter = teacher_call_counter
        self._api_counter = api_call_counter
        self._llm_counter = llm_call_counter
        self._host_reader = host_state_reader
        self._output_reader = output_state_reader
        self._report: W09WeaningReport | None = None
        self._window_identities: list[W09WindowIdentity] = []
        self._stage4_output: object = None

    def state_key(self) -> tuple[int, ...]:
        """返回 protocol、phase owner 和已完成窗口的稳定运行身份。"""
        return digest_value({
            "owners": (
                _state_key(self.training_material_source, where="training source")
                if self.training_material_source is not None else (),
                _state_key(self.dev_calibrator, where="dev calibrator")
                if self.dev_calibrator is not None else (),
                _state_key(self.shadow_auditor, where="shadow auditor")
                if self.shadow_auditor is not None else (),
            ),
            "contract": (
                list(self.frozen_contract.stable_key())
                if self.frozen_contract is not None else []
            ),
            "protocol": list(self.protocol.stable_key()),
            "windows": [item.stable_key() for item in self._window_identities],
        })

    def _count(self, ctx: object, counter: Callable[[object], int] | None,
               names: tuple[str, ...]) -> int:
        """读取注入计数器或 teacher 的显式调用计数。"""
        if counter is not None:
            value = counter(ctx)
            if type(value) is not int or value < 0:
                raise W09WeaningError("W-09 injected call counter is invalid")
            return value
        return max(
            _counter(ctx, names),
            _counter(getattr(ctx, "teacher", None), names),
        )

    def _host(self, ctx: object) -> object:
        """读取 host 状态，优先使用调用方注入的只读快照。"""
        if self._host_reader is not None:
            return self._host_reader(ctx)
        return _host_snapshot(ctx)

    def _output(self, ctx: object, report: object) -> object:
        """读取 production 输出状态，供 shadow phase 零反馈审计。"""
        if self._output_reader is not None:
            return self._output_reader(ctx)
        return _output_snapshot(ctx, report)

    @staticmethod
    def _material_count(material: object) -> int:
        """提取并校验冻结 Teacher Evidence 数量。"""
        if hasattr(material, "training_evidence"):
            material = getattr(material, "training_evidence")
        if not isinstance(material, (tuple, list)) or not material:
            raise W09WeaningError("W-09 frozen Teacher Evidence is missing")
        for item in material:
            if isinstance(item, TeacherEvidenceRecord):
                continue
            value = item.to_dict() if callable(getattr(item, "to_dict", None)) else item
            if _forbidden_field(value):
                raise W09WeaningError("W-09 training material contains answer fields")
        return len(material)

    def _phase_input(self, value: object, *, label: str) -> str:
        """从 phase 的公开 typed 输入形成稳定 commitment。"""
        result = _sha256(value, where=label)
        if label == "training input" and result != self.protocol.input_commitment:
            raise W09WeaningError("W-09 training input commitment drifted")
        return result

    def run(
            self,
            ctx: object,
            stage4_report: object,
            *,
            training_material: object = None,
            dev_items: object = None,
            shadow_errors: object = None,
            ) -> W09WeaningReport:
        """执行前三个 phase；零调用窗口由 execute_zero_call_window 单独消费。"""
        if self._report is not None:
            if self._stage4_output != self._output(ctx, stage4_report):
                raise W09WeaningError("W-09 repeated run changed production output")
            return self._report
        if getattr(stage4_report, "complete", False) is not True:
            raise W09WeaningError("W-09 typed stage4 report is incomplete")
        before_host = self._host(ctx)
        before_output = self._output(ctx, stage4_report)
        before_teacher = self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls"))
        before_api = self._count(ctx, self._api_counter, ("api_call_count", "api_calls"))
        before_llm = self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls"))
        training_input = (
            training_material
            if training_material is not None
            else _call(self.training_material_source, ("read", "read_training_material"), ctx, stage4_report)
            if self.training_material_source is not None
            else None
        )
        after_host = self._host(ctx)
        after_output = self._output(ctx, stage4_report)
        self._require_unchanged(before_host, after_host, before_output, after_output)
        self._require_calls_zero(ctx, before_teacher, before_api, before_llm)
        count = self._material_count(training_input)
        phase_input = self._phase_input(training_input, label="training input")
        training_audit = W09PhaseAudit(
            TeacherExitPhase.TRAINING_MATERIAL_SOURCE,
            phase_input,
            _sha256((phase_input, count), where="training output"),
            training_material_count=count,
        )

        before_host = self._host(ctx)
        before_output = self._output(ctx, stage4_report)
        before_teacher = self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls"))
        before_api = self._count(ctx, self._api_counter, ("api_call_count", "api_calls"))
        before_llm = self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls"))
        dev_value = (
            dev_items
            if dev_items is not None
            else _call(self.dev_calibrator, ("calibrate", "read_dev"), ctx, stage4_report)
            if self.dev_calibrator is not None
            else None
        )
        after_host = self._host(ctx)
        after_output = self._output(ctx, stage4_report)
        self._require_unchanged(before_host, after_host, before_output, after_output)
        self._require_calls_zero(ctx, before_teacher, before_api, before_llm)
        dev_count = self._dev_count(dev_value)
        dev_input = self._phase_input(dev_value, label="dev input")
        dev_output = _sha256((dev_input, self.protocol.threshold_key), where="dev output")
        dev_audit = W09PhaseAudit(
            TeacherExitPhase.DEV_CALIBRATION_ONLY,
            dev_input,
            dev_output,
            dev_read_count=dev_count,
        )
        shadow_value = (
            shadow_errors
            if shadow_errors is not None
            else getattr(stage4_report, "outcomes", None)
        )
        if not shadow_value:
            raise W09WeaningError("W-09 shadow error input is missing")
        shadow_input = self._phase_input(shadow_value, label="shadow input")
        before_host = self._host(ctx)
        before_output = self._output(ctx, stage4_report)
        before_teacher = self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls"))
        before_api = self._count(ctx, self._api_counter, ("api_call_count", "api_calls"))
        before_llm = self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls"))
        audit_result = _call(
            self.shadow_auditor,
            ("record", "record_shadow", "append"),
            ctx,
            shadow_value,
        ) if self.shadow_auditor is not None else None
        if audit_result is None:
            raise W09WeaningError("W-09 shadow auditor owner is missing")
        after_host = self._host(ctx)
        after_output = self._output(ctx, stage4_report)
        self._require_unchanged(before_host, after_host, before_output, after_output)
        self._require_calls_zero(ctx, before_teacher, before_api, before_llm)
        shadow_count = len(shadow_value) if isinstance(shadow_value, (tuple, list)) else 1
        shadow_writes = audit_result if type(audit_result) is int else _counter(
            self.shadow_auditor, ("audit_write_count", "write_count", "count"))
        if shadow_writes <= 0:
            shadow_writes = 1
        shadow_audit = W09PhaseAudit(
            TeacherExitPhase.SHADOW_ERROR_ONLY,
            shadow_input,
            _sha256((shadow_input, shadow_writes), where="shadow output"),
            shadow_error_count=shadow_count,
            shadow_audit_write_count=shadow_writes,
        )
        self._report = W09WeaningReport(
            self.protocol.stable_key(),
            (training_audit, dev_audit, shadow_audit),
            (),
            blockers=(W09_ZERO_CALL_WINDOWS_PENDING,),
        )
        self._stage4_output = self._output(ctx, stage4_report)
        return self._report

    @staticmethod
    def _dev_count(value: object) -> int:
        """提取 dev 校准实际读取量，拒绝空校准或不透明结果。"""
        if type(value) is int:
            if value <= 0:
                raise W09WeaningError("W-09 dev calibration read count is zero")
            return value
        if isinstance(value, (tuple, list)):
            if not value:
                raise W09WeaningError("W-09 dev calibration input is empty")
            return len(value)
        count = getattr(value, "dev_read_count", None)
        if type(count) is int and count > 0:
            return count
        raise W09WeaningError("W-09 dev calibration owner did not report reads")

    @staticmethod
    def _require_unchanged(before_host: object, after_host: object,
                           before_output: object, after_output: object) -> None:
        """phase 读写审计：host 和 production output 必须完全不变。"""
        if before_host != after_host:
            raise W09WeaningError("W-09 phase wrote formal host state")
        if before_output != after_output:
            raise W09WeaningError("W-09 shadow phase changed production output")

    def _require_calls_zero(self, ctx: object, teacher: int, api: int, llm: int) -> None:
        """确认一个 phase 没有新增 teacher/API/LLM 调用。"""
        after = (
            self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls")),
            self._count(ctx, self._api_counter, ("api_call_count", "api_calls")),
            self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls")),
        )
        if after != (teacher, api, llm):
            raise W09WeaningError("W-09 phase performed a live teacher/API/LLM call")

    def _validate_zero_call_identity(
            self,
            identity: object,
            ) -> W09WindowIdentity:
        """按当前连续 ordinal 校验实际窗口 identity、input 和 Candidate。"""
        if not isinstance(identity, W09WindowIdentity):
            raise W09WeaningError("W-09 zero-call window identity is invalid")
        if self._report is None:
            raise W09WeaningError("W-09 pre-window phases are not complete")
        ordinal = len(self._window_identities) + 1
        if identity.window_ordinal != ordinal:
            raise W09WeaningError("W-09 zero-call window id was reused or skipped")
        if identity.phase is not TeacherExitPhase.ZERO_CALL_WINDOW:
            raise W09WeaningError("W-09 window phase is invalid")
        if identity.input_commitment != self.protocol.window_input_commitments[ordinal - 1]:
            raise W09WeaningError("W-09 window input commitment drifted")
        if identity.candidate_identity != self.protocol.candidate_identity:
            raise W09WeaningError("W-09 window candidate identity drifted")
        return identity

    def _append_zero_call_identity(
            self,
            identity: W09WindowIdentity,
            ) -> W09WindowIdentity:
        """一次追加已量测窗口并重算三个窗口的 ready/blocker 状态。"""
        self._window_identities.append(identity)
        assert self._report is not None
        self._report = W09WeaningReport(
            self.protocol.stable_key(),
            self._report.phase_audits,
            tuple(self._window_identities),
            ready=len(self._window_identities) == self.protocol.window_count,
            blockers=() if len(self._window_identities) == self.protocol.window_count
            else (W09_ZERO_CALL_WINDOWS_PENDING,),
        )
        return identity

    def execute_zero_call_window(
            self,
            ctx: object,
            identity: W09WindowIdentity,
            operation: Callable[[], object] | None = None,
            ) -> W09WindowIdentity:
        """按预注册序执行一个已知 identity 的窗口，并包住实际 operation 审计。"""
        identity = self._validate_zero_call_identity(identity)
        before_host = self._host(ctx)
        before = (
            self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls")),
            self._count(ctx, self._api_counter, ("api_call_count", "api_calls")),
            self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls")),
        )
        if operation is not None:
            operation()
        after = (
            self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls")),
            self._count(ctx, self._api_counter, ("api_call_count", "api_calls")),
            self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls")),
        )
        if after != before or identity.teacher_call_count != 0:
            raise W09WeaningError("W-09 zero-call window performed a live call")
        if before_host != self._host(ctx):
            raise W09WeaningError("W-09 zero-call window wrote formal host state")
        return self._append_zero_call_identity(identity)

    def execute_measured_zero_call_window(
            self,
            ctx: object,
            operation: Callable[[], object],
            ) -> W09WindowIdentity:
        """先执行真实窗口，再用其实际输出 identity 完成同一零调用和 host 零写审计。"""
        if not callable(operation):
            raise W09WeaningError("W-09 measured window operation is invalid")
        if self._report is None:
            raise W09WeaningError("W-09 pre-window phases are not complete")
        before_host = self._host(ctx)
        before = (
            self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls")),
            self._count(ctx, self._api_counter, ("api_call_count", "api_calls")),
            self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls")),
        )
        identity = self._validate_zero_call_identity(operation())
        after = (
            self._count(ctx, self._teacher_counter, ("call_count", "teacher_calls")),
            self._count(ctx, self._api_counter, ("api_call_count", "api_calls")),
            self._count(ctx, self._llm_counter, ("llm_call_count", "llm_calls")),
        )
        if after != before or identity.teacher_call_count != 0:
            raise W09WeaningError("W-09 measured window performed a live call")
        if before_host != self._host(ctx):
            raise W09WeaningError("W-09 measured window wrote formal host state")
        return self._append_zero_call_identity(identity)

    zero_call_window = execute_zero_call_window


def validate_w09_weaning_pair(
        protocol: object,
        runtime: object,
        *,
        require_frozen_contract: bool = False,
        ) -> W09TypedWeaningRuntime:
    """校验 formal 注入的 protocol/runtime 成对且身份一致。"""
    if not isinstance(protocol, W09TypedWeaningProtocol):
        raise W09WeaningError("W-09 typed weaning protocol is missing or invalid")
    if not isinstance(runtime, W09TypedWeaningRuntime):
        raise W09WeaningError("W-09 typed weaning runtime is missing or invalid")
    if runtime.protocol.stable_key() != protocol.stable_key():
        raise W09WeaningError("W-09 protocol/runtime identity drifted")
    if require_frozen_contract and runtime.frozen_contract is None:
        raise W09WeaningError("W-09 formal runtime lacks frozen authority/registry")
    if require_frozen_contract and any(item is None for item in (
            runtime.training_material_source,
            runtime.dev_calibrator,
            runtime.shadow_auditor,
    )):
        raise W09WeaningError("W-09 formal runtime lacks a phase owner")
    if require_frozen_contract:
        owners = (
            (runtime.training_material_source, W09_TRAINING_MATERIAL_OWNER),
            (runtime.dev_calibrator, W09_DEV_OWNER),
        )
        for owner, expected in owners:
            if getattr(owner, "owner_key", None) != expected:
                raise W09WeaningError("W-09 formal phase owner drifted")
    if runtime.frozen_contract is not None and (
            protocol.authority_sha256 != runtime.frozen_contract.authority_sha256
            or protocol.registry_identity != runtime.frozen_contract.stable_key()):
        raise W09WeaningError("W-09 formal authority/registry identity drifted")
    return runtime


def make_w09_typed_weaning_protocol(
        *,
        authority_sha256: str,
        registry_identity: tuple[int, ...],
        candidate_identity: str,
        input_commitment: str,
        threshold_key: tuple[int, ...],
        window_input_commitments: tuple[str, ...] = (),
    ) -> W09TypedWeaningProtocol:
    """由现场 authority、registry 和当前 Candidate 输入创建冻结协议。"""
    return W09TypedWeaningProtocol(
        authority_sha256,
        registry_identity,
        candidate_identity,
        input_commitment,
        threshold_key,
        window_input_commitments=window_input_commitments,
    )


def w09_commitment(value: object) -> str:
    """为测试、caller 和 receipt 生成 W-09 canonical SHA-256 commitment。"""
    return _sha256(value, where="W-09")


def make_w09_typed_weaning_protocol_from_contract(
        context: W09FrozenContract,
        *,
        candidate_identity: str,
        input_commitment: str,
        threshold_key: tuple[int, ...],
        window_input_commitments: tuple[str, ...] = (),
        ) -> W09TypedWeaningProtocol:
    """从 W09-01 frozen contract 创建 authority/registry 精确匹配的协议。"""
    if not isinstance(context, W09FrozenContract):
        raise W09WeaningError("W-09 frozen contract type is invalid")
    return make_w09_typed_weaning_protocol(
        authority_sha256=context.authority_sha256,
        registry_identity=context.stable_key(),
        candidate_identity=candidate_identity,
        input_commitment=input_commitment,
        threshold_key=threshold_key,
        window_input_commitments=window_input_commitments,
    )


__all__ = [
    "W09DevCalibrationOwner",
    "W09FrozenTeacherEvidenceSource",
    "W09PhaseAudit",
    "W09ShadowErrorAudit",
    "W09TypedWeaningProtocol",
    "W09TypedWeaningRuntime",
    "W09WeaningError",
    "W09WeaningReport",
    "W09_TYPED_WEANING_BLOCKER",
    "W09_TYPED_WEANING_READY",
    "W09_TYPED_WEANING_VERSION",
    "W09_ZERO_CALL_WINDOWS_PENDING",
    "W09_ZERO_CALL_WINDOWS_REQUIRED",
    "make_w09_typed_weaning_protocol",
    "make_w09_typed_weaning_protocol_from_contract",
    "validate_w09_weaning_pair",
    "w09_commitment",
]
