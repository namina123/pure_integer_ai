"""版本化课程阶段计划、评测报告和 hard gate 运行时。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.curriculum_mastery import (
    CurriculumMasteryIntegrityError,
    CurriculumMasteryStore,
    CurriculumReportFaultInjector,
    CurriculumStageReportRecord,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    encode_integer_tuple,
    pack_key,
    strict_integer_tuple,
)


_REPORT_PAYLOAD_VERSION = 1
_HASH_BYTES = 7


class CurriculumHardGateError(RuntimeError):
    """课程顺序、前置 mastery 或评测完整性不足，必须停止训练。"""


class CurriculumEvaluatorDriftError(CurriculumHardGateError):
    """评测器 state key 在绑定后发生变化，旧报告不得继续使用。"""


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放协议键为非空严格整数 tuple。"""
    return strict_integer_tuple(value, label=label)


def _stable_hash(value: tuple[int, ...]) -> int:
    """把规范整数键压成 SQLite 安全的正整数索引。"""
    digest = hashlib.sha256(encode_integer_tuple(value)).digest()
    return int.from_bytes(digest[:_HASH_BYTES], "little") + 1


@dataclass(frozen=True)
class CurriculumArtifactVersions:
    """一次 mastery 判定绑定的数据、代码、原语和课程完整版本键。"""

    data_key: tuple[int, ...]
    code_key: tuple[int, ...]
    primitive_key: tuple[int, ...]
    curriculum_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝缺失或非整数版本，禁止以隐式当前版本恢复。"""
        for label, value in (
                ("data_key", self.data_key),
                ("code_key", self.code_key),
                ("primitive_key", self.primitive_key),
                ("curriculum_key", self.curriculum_key)):
            _key(value, label=f"CurriculumArtifactVersions.{label}")

    def stable_key(self) -> tuple[int, ...]:
        """返回保留四个版本边界的规范整数键。"""
        result = [_REPORT_PAYLOAD_VERSION]
        for value in (
                self.data_key, self.code_key,
                self.primitive_key, self.curriculum_key):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True)
class CurriculumStagePlan:
    """由调用方注入的严格阶段顺序及可跳过属性。"""

    ordered_stage_keys: tuple[tuple[int, ...], ...]
    skippable_stage_keys: frozenset[tuple[int, ...]] = frozenset()

    def __post_init__(self) -> None:
        """核验阶段唯一、有序且 skippable 只引用计划内阶段。"""
        if not isinstance(self.ordered_stage_keys, tuple) or not self.ordered_stage_keys:
            raise ValueError("课程计划必须包含至少一个阶段")
        normalized = tuple(
            _key(value, label="CurriculumStagePlan.stage_key")
            for value in self.ordered_stage_keys
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("课程计划阶段键不得重复")
        if (not isinstance(self.skippable_stage_keys, frozenset)
                or any(value not in normalized
                       for value in self.skippable_stage_keys)):
            raise ValueError("skippable_stage_keys 必须是计划内阶段的 frozenset")
        for value in self.skippable_stage_keys:
            _key(value, label="CurriculumStagePlan.skippable_stage_key")

    def index(self, stage_key: tuple[int, ...]) -> int:
        """返回阶段在严格计划中的位置，不存在时硬失败。"""
        stage_key = _key(stage_key, label="curriculum stage key")
        try:
            return self.ordered_stage_keys.index(stage_key)
        except ValueError as exc:
            raise CurriculumHardGateError("阶段不在当前课程计划中") from exc

    def preceding(self, stage_key: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        """返回指定阶段之前必须全部掌握的严格前置序列。"""
        return self.ordered_stage_keys[:self.index(stage_key)]

    def is_skippable(self, stage_key: tuple[int, ...]) -> bool:
        """判断同版本已掌握阶段是否允许在恢复时跳过。"""
        self.index(stage_key)
        return stage_key in self.skippable_stage_keys


@dataclass(frozen=True)
class CurriculumGateCheck:
    """评测报告中的一个独立检查及其纯整数证据。"""

    check_key: tuple[int, ...]
    passed: bool
    evidence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验检查身份、布尔结论和不可省略的证据键。"""
        _key(self.check_key, label="CurriculumGateCheck.check_key")
        if type(self.passed) is not bool:
            raise TypeError("CurriculumGateCheck.passed 必须是 bool")
        _key(self.evidence_key, label="CurriculumGateCheck.evidence_key")


@dataclass(frozen=True)
class CurriculumStageEvaluation:
    """评测器为一个阶段返回的完整分维检查集合。"""

    checks: tuple[CurriculumGateCheck, ...]

    def __post_init__(self) -> None:
        """要求至少一个检查且检查身份唯一。"""
        if (not isinstance(self.checks, tuple) or not self.checks
                or any(not isinstance(item, CurriculumGateCheck)
                       for item in self.checks)):
            raise ValueError("阶段评测必须包含 CurriculumGateCheck")
        keys = tuple(item.check_key for item in self.checks)
        if len(set(keys)) != len(keys):
            raise ValueError("阶段评测 check_key 不得重复")

    @property
    def passed(self) -> bool:
        """仅当全部分维检查通过时返回真。"""
        return all(item.passed for item in self.checks)


@dataclass(frozen=True)
class CurriculumStageEvaluationRequest:
    """发送给注入式评测器的阶段、版本、前置报告和宿主证据。"""

    stage_key: tuple[int, ...]
    versions: CurriculumArtifactVersions
    prerequisite_reports: tuple[tuple[tuple[int, ...], int], ...]
    evidence: Any


@runtime_checkable
class CurriculumStageEvaluator(Protocol):
    """不依赖具体课程语义的版本化阶段评测协议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回评测逻辑、阈值和依赖的完整版本键。"""
        ...

    def evaluate(
            self,
            request: CurriculumStageEvaluationRequest,
            ) -> CurriculumStageEvaluation:
        """只读评估当前阶段证据并返回分维结果。"""
        ...


@dataclass(frozen=True)
class _CurriculumReportPayload:
    """阶段报告 payload 的规范逻辑结构。"""

    stage_key: tuple[int, ...]
    version_key: tuple[int, ...]
    evaluator_key: tuple[int, ...]
    prerequisite_reports: tuple[tuple[tuple[int, ...], int], ...]
    checks: tuple[CurriculumGateCheck, ...]

    def to_key(self) -> tuple[int, ...]:
        """编码为可由持久层 parts 保存的规范整数 tuple。"""
        result = [_REPORT_PAYLOAD_VERSION]
        pack_key(result, self.stage_key)
        pack_key(result, self.version_key)
        pack_key(result, self.evaluator_key)
        result.append(len(self.prerequisite_reports))
        for stage_key, report_hash in self.prerequisite_reports:
            pack_key(result, stage_key)
            result.append(report_hash)
        result.append(len(self.checks))
        for check in self.checks:
            pack_key(result, check.check_key)
            result.append(1 if check.passed else 0)
            pack_key(result, check.evidence_key)
        return tuple(result)

    @classmethod
    def from_key(cls, value: tuple[int, ...]) -> _CurriculumReportPayload:
        """严格解码报告 payload，拒绝未知版本和尾随字段。"""
        try:
            reader = IntegerStreamReader(value)
            if reader.read_positive(label="curriculum report payload version") != 1:
                raise CurriculumMasteryIntegrityError("阶段报告 payload 版本未知")
            stage_key = reader.read_key(label="curriculum report stage key")
            version_key = reader.read_key(label="curriculum report version key")
            evaluator_key = reader.read_key(label="curriculum report evaluator key")
            prerequisite_count = reader.read_nonnegative(
                label="curriculum report prerequisite count")
            prerequisites = []
            for _ in range(prerequisite_count):
                prerequisites.append((
                    reader.read_key(label="curriculum prerequisite stage key"),
                    reader.read_positive(label="curriculum prerequisite report hash"),
                ))
            check_count = reader.read_positive(label="curriculum report check count")
            checks = []
            for _ in range(check_count):
                check_key = reader.read_key(label="curriculum report check key")
                passed = reader.read_nonnegative(label="curriculum report check passed")
                if passed not in {0, 1}:
                    raise CurriculumMasteryIntegrityError("阶段报告检查通过位非法")
                checks.append(CurriculumGateCheck(
                    check_key,
                    bool(passed),
                    reader.read_key(label="curriculum report check evidence"),
                ))
            reader.finish()
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, CurriculumMasteryIntegrityError):
                raise
            raise CurriculumMasteryIntegrityError("阶段报告 payload 非法") from exc
        return cls(
            stage_key,
            version_key,
            evaluator_key,
            tuple(prerequisites),
            tuple(checks),
        )


class CurriculumMasteryRuntime:
    """从正式报告派生 mastery，并执行严格顺序和失效传播。"""

    def __init__(
            self,
            backend: StorageBackend,
            plan: CurriculumStagePlan,
            versions: CurriculumArtifactVersions,
            evaluator: CurriculumStageEvaluator,
            ) -> None:
        """绑定后端、注入计划、完整版本和稳定评测器。"""
        if not isinstance(plan, CurriculumStagePlan):
            raise TypeError("plan 必须是 CurriculumStagePlan")
        if not isinstance(versions, CurriculumArtifactVersions):
            raise TypeError("versions 必须是 CurriculumArtifactVersions")
        if not isinstance(evaluator, CurriculumStageEvaluator):
            raise TypeError("evaluator 未实现 CurriculumStageEvaluator")
        self.store = CurriculumMasteryStore(backend)
        self.plan = plan
        self.versions = versions
        self.evaluator = evaluator
        self._bound_evaluator_key = self._read_evaluator_key()

    def _read_evaluator_key(self) -> tuple[int, ...]:
        """读取并核验评测器当前稳定键。"""
        return _key(
            self.evaluator.state_key(),
            label="CurriculumStageEvaluator.state_key",
        )

    def _require_evaluator_stable(self) -> tuple[int, ...]:
        """拒绝绑定后漂移的评测器，避免旧报告跨逻辑复用。"""
        current = self._read_evaluator_key()
        if current != self._bound_evaluator_key:
            raise CurriculumEvaluatorDriftError("课程评测器 state key 已漂移")
        return current

    def current_report(
            self,
            stage_key: tuple[int, ...],
            ) -> CurriculumStageReportRecord | None:
        """返回当前版本、评测器和前置报告下仍有效的 mastery 报告。"""
        evaluator_key = self._require_evaluator_stable()
        version_key = self.versions.stable_key()
        return self._current_report(
            stage_key,
            evaluator_key=evaluator_key,
            version_key=version_key,
            cache={},
        )

    def _current_report(
            self,
            stage_key: tuple[int, ...],
            *,
            evaluator_key: tuple[int, ...],
            version_key: tuple[int, ...],
            cache: dict[tuple[int, ...], CurriculumStageReportRecord | None],
            ) -> CurriculumStageReportRecord | None:
        """在一次判定内缓存前置结果，避免严格课程重复递归读取。"""
        stage_key = _key(stage_key, label="curriculum current stage key")
        self.plan.index(stage_key)
        if stage_key in cache:
            return cache[stage_key]
        record = self.store.latest(
            _stable_hash(stage_key),
            _stable_hash(version_key),
        )
        if record is None:
            cache[stage_key] = None
            return None
        payload = _CurriculumReportPayload.from_key(record.payload)
        if payload.stage_key != stage_key:
            raise CurriculumMasteryIntegrityError("阶段报告 stage hash 碰撞")
        if payload.version_key != version_key:
            raise CurriculumMasteryIntegrityError("阶段报告 version hash 碰撞")
        if record.stage_hash != _stable_hash(payload.stage_key):
            raise CurriculumMasteryIntegrityError("阶段报告 stage hash 不匹配")
        if record.version_hash != _stable_hash(payload.version_key):
            raise CurriculumMasteryIntegrityError("阶段报告 version hash 不匹配")
        if record.evaluator_hash != _stable_hash(payload.evaluator_key):
            raise CurriculumMasteryIntegrityError("阶段报告 evaluator hash 不匹配")
        if payload.evaluator_key != evaluator_key:
            cache[stage_key] = None
            return None
        evaluation = CurriculumStageEvaluation(payload.checks)
        if record.passed != int(evaluation.passed):
            raise CurriculumMasteryIntegrityError("阶段报告通过位与检查不一致")
        if not evaluation.passed:
            cache[stage_key] = None
            return None
        expected_prerequisites = []
        for prerequisite_key in self.plan.preceding(stage_key):
            prerequisite = self._current_report(
                prerequisite_key,
                evaluator_key=evaluator_key,
                version_key=version_key,
                cache=cache,
            )
            if prerequisite is None:
                cache[stage_key] = None
                return None
            expected_prerequisites.append(
                (prerequisite_key, prerequisite.report_hash))
        if payload.prerequisite_reports != tuple(expected_prerequisites):
            cache[stage_key] = None
            return None
        cache[stage_key] = record
        return record

    def is_mastered(self, stage_key: tuple[int, ...]) -> bool:
        """判断阶段是否在当前完整依赖下真正 mastered。"""
        return self.current_report(stage_key) is not None

    def prepare(
            self,
            requested_stage_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[tuple[int, ...], ...]:
        """在训练写入前校验顺序和前置，并返回本次必须执行的后缀。"""
        if not isinstance(requested_stage_keys, tuple) or not requested_stage_keys:
            raise CurriculumHardGateError("本次课程请求不能为空")
        normalized = tuple(
            _key(value, label="curriculum requested stage key")
            for value in requested_stage_keys
        )
        if len(set(normalized)) != len(normalized):
            raise CurriculumHardGateError("本次课程请求阶段不得重复")
        indexes = tuple(self.plan.index(value) for value in normalized)
        if indexes != tuple(sorted(indexes)):
            raise CurriculumHardGateError("本次课程请求不符合严格阶段顺序")
        evaluator_key = self._require_evaluator_stable()
        version_key = self.versions.stable_key()
        cache: dict[
            tuple[int, ...], CurriculumStageReportRecord | None,
        ] = {}

        def mastered(stage_key: tuple[int, ...]) -> bool:
            """复用本次 prepare 的前置报告缓存。"""
            return self._current_report(
                stage_key,
                evaluator_key=evaluator_key,
                version_key=version_key,
                cache=cache,
            ) is not None

        requested_set = set(normalized)
        for stage_key in normalized:
            for prerequisite_key in self.plan.preceding(stage_key):
                if prerequisite_key not in requested_set and not mastered(
                        prerequisite_key):
                    raise CurriculumHardGateError("课程前置阶段尚未 mastered")
        first_pending = None
        for position, stage_key in enumerate(normalized):
            if (not self.plan.is_skippable(stage_key)
                    or not mastered(stage_key)):
                first_pending = position
                break
        if first_pending is None:
            return ()
        pending = normalized[first_pending:]
        start_index = self.plan.index(pending[0])
        end_index = self.plan.index(pending[-1])
        required_suffix = self.plan.ordered_stage_keys[start_index:end_index + 1]
        if pending != required_suffix:
            raise CurriculumHardGateError("待执行课程后缀不得跳过中间阶段")
        return pending

    def evaluate_and_record(
            self,
            stage_key: tuple[int, ...],
            evidence: Any,
            *,
            required_checks: tuple[CurriculumGateCheck, ...] = (),
            fault_injector: CurriculumReportFaultInjector | None = None,
            ) -> CurriculumStageReportRecord:
        """核验前置、调用评测器并追加唯一正式报告。"""
        stage_key = _key(stage_key, label="curriculum evaluated stage key")
        self.plan.index(stage_key)
        evaluator_key = self._require_evaluator_stable()
        version_key = self.versions.stable_key()
        cache: dict[
            tuple[int, ...], CurriculumStageReportRecord | None,
        ] = {}
        prerequisites = []
        for prerequisite_key in self.plan.preceding(stage_key):
            report = self._current_report(
                prerequisite_key,
                evaluator_key=evaluator_key,
                version_key=version_key,
                cache=cache,
            )
            if report is None:
                raise CurriculumHardGateError("课程前置阶段尚未 mastered")
            prerequisites.append((prerequisite_key, report.report_hash))
        request = CurriculumStageEvaluationRequest(
            stage_key,
            self.versions,
            tuple(prerequisites),
            evidence,
        )
        evaluation = self.evaluator.evaluate(request)
        if not isinstance(evaluation, CurriculumStageEvaluation):
            raise TypeError("课程评测器必须返回 CurriculumStageEvaluation")
        self._require_evaluator_stable()
        if (not isinstance(required_checks, tuple)
                or any(not isinstance(item, CurriculumGateCheck)
                       for item in required_checks)):
            raise TypeError("required_checks 必须是 CurriculumGateCheck tuple")
        checks = evaluation.checks + required_checks
        if len({item.check_key for item in checks}) != len(checks):
            raise CurriculumHardGateError("课程评测检查身份冲突")
        complete = CurriculumStageEvaluation(checks)
        payload = _CurriculumReportPayload(
            stage_key,
            version_key,
            evaluator_key,
            tuple(prerequisites),
            complete.checks,
        ).to_key()
        stage_hash = _stable_hash(stage_key)
        version_hash = _stable_hash(version_key)
        sequence = self.store.next_sequence(stage_hash, version_hash)
        passed = int(complete.passed)
        report_hash = _stable_hash((
            stage_hash,
            version_hash,
            _stable_hash(evaluator_key),
            sequence,
            passed,
            *payload,
        ))
        return self.store.append(
            CurriculumStageReportRecord(
                report_hash,
                stage_hash,
                version_hash,
                _stable_hash(evaluator_key),
                sequence,
                passed,
                payload,
            ),
            fault_injector=fault_injector,
        )


@dataclass(frozen=True)
class CurriculumMasteryProtocol:
    """把通用课程计划映射到训练宿主阶段整数。"""

    plan: CurriculumStagePlan
    versions: CurriculumArtifactVersions
    evaluator: CurriculumStageEvaluator
    stage_bindings: tuple[tuple[int, tuple[int, ...]], ...]
    host_gate_check_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验训练阶段和课程阶段一一映射且宿主检查键完整。"""
        if not isinstance(self.plan, CurriculumStagePlan):
            raise TypeError("plan 必须是 CurriculumStagePlan")
        if not isinstance(self.versions, CurriculumArtifactVersions):
            raise TypeError("versions 必须是 CurriculumArtifactVersions")
        if not isinstance(self.evaluator, CurriculumStageEvaluator):
            raise TypeError("evaluator 未实现 CurriculumStageEvaluator")
        if not isinstance(self.stage_bindings, tuple) or not self.stage_bindings:
            raise ValueError("stage_bindings 不能为空")
        training_stages = []
        stage_keys = []
        for training_stage, stage_key in self.stage_bindings:
            if type(training_stage) is not int:
                raise TypeError("训练阶段映射键必须是严格整数")
            normalized = _key(stage_key, label="curriculum stage binding")
            self.plan.index(normalized)
            training_stages.append(training_stage)
            stage_keys.append(normalized)
        if (len(set(training_stages)) != len(training_stages)
                or len(set(stage_keys)) != len(stage_keys)):
            raise ValueError("stage_bindings 必须是一一映射")
        _key(self.host_gate_check_key, label="host_gate_check_key")

    def stage_keys_for(
            self,
            training_stages: tuple[int, ...],
            ) -> tuple[tuple[int, ...], ...]:
        """按训练请求顺序解析课程阶段键，缺失映射时硬失败。"""
        mapping = dict(self.stage_bindings)
        try:
            stage_keys = tuple(mapping[stage] for stage in training_stages)
        except KeyError as exc:
            raise CurriculumHardGateError("训练阶段缺少课程 mastery 映射") from exc
        indexes = tuple(self.plan.index(value) for value in stage_keys)
        if indexes != tuple(sorted(indexes)):
            raise CurriculumHardGateError("训练阶段映射违反课程严格顺序")
        return stage_keys

    def bind(self, backend: StorageBackend) -> CurriculumMasteryRuntime:
        """在当前已注册后端上构造无全局状态的 mastery 运行时。"""
        return CurriculumMasteryRuntime(
            backend,
            self.plan,
            self.versions,
            self.evaluator,
        )


__all__ = [
    "CurriculumArtifactVersions",
    "CurriculumEvaluatorDriftError",
    "CurriculumGateCheck",
    "CurriculumHardGateError",
    "CurriculumMasteryProtocol",
    "CurriculumMasteryRuntime",
    "CurriculumStageEvaluation",
    "CurriculumStageEvaluationRequest",
    "CurriculumStageEvaluator",
    "CurriculumStagePlan",
]
