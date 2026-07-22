"""H-02A 通用确定性扰动、变换 trace 和定向负 Evidence 协议。

核心不解释语言、角色、cue 或逻辑原语，只接受完整一等对象序列、调用方变换键和
三态 verifier。只有 verifier 明确确认语义对象或受测约束已改变，才允许向其列出的
Hypothesis 追加 refute；表面等价变换不写负证据，无法判断时只写 unknown。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


ASSESSMENT_CHANGED = 1
ASSESSMENT_EQUIVALENT = 2
ASSESSMENT_UNKNOWN = 3
_ASSESSMENTS = frozenset({
    ASSESSMENT_CHANGED,
    ASSESSMENT_EQUIVALENT,
    ASSESSMENT_UNKNOWN,
})

_TRACE_VERSION = 1
_EVIDENCE_HASHER = Hasher("pure_integer_ai.perturbation_evidence.v1")


def _integer_key(value, *, where: str) -> tuple[int, ...]:
    """校验调用方注入的开放整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _object_sequence(value, *, where: str) -> tuple[ObjectIdentity, ...]:
    """校验变换对象必须是一等图对象身份而非 surface 或运行时 ref。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是 ObjectIdentity tuple")
    if any(not isinstance(item, ObjectIdentity) for item in value):
        raise TypeError(f"{where} 只能包含 ObjectIdentity")
    return value


def _pack_objects(values: tuple[ObjectIdentity, ...]) -> tuple[int, ...]:
    """按长度前缀保存完整对象身份，禁止摘要替代 trace。"""
    packed: list[int] = [len(values)]
    for value in values:
        key = value.stable_key()
        packed.extend((len(key), *key))
    return tuple(packed)


@dataclass(frozen=True)
class PerturbationProtocol:
    """注入反驳、unknown 理由以及同源重复诊断键。"""

    refute_reason_key: tuple[int, ...]
    unknown_reason_key: tuple[int, ...]
    duplicate_transform_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验三类调用方键非空、纯整数且互不混用。"""
        keys = tuple(
            _integer_key(value, where=f"PerturbationProtocol.{name}")
            for name, value in (
                ("refute_reason_key", self.refute_reason_key),
                ("unknown_reason_key", self.unknown_reason_key),
                ("duplicate_transform_key", self.duplicate_transform_key),
            )
        )
        if len(set(keys)) != len(keys):
            raise ValueError("扰动 Evidence 理由和重复诊断键必须互不相同")


def _positions(
        value, *, size: int, where: str) -> tuple[int, ...]:
    """校验一侧受影响位置严格递增、唯一且不越过该侧序列。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if (any(type(index) is not int or index < 0 or index >= size
            for index in value)
            or tuple(sorted(set(value))) != value):
        raise ValueError(f"{where} 必须严格递增、唯一且位于对应序列内")
    return value


def _derived_affected_positions(
        original: tuple[ObjectIdentity, ...],
        transformed: tuple[ObjectIdentity, ...],
        output_to_input: tuple[int, ...],
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """由完整映射推导输入/输出变化位置，并核验保留对象身份未被偷换。"""
    input_to_outputs: list[list[int]] = [[] for _ in original]
    affected_output: list[int] = []
    for output_index, input_index in enumerate(output_to_input):
        if input_index < 0:
            affected_output.append(output_index)
            continue
        if transformed[output_index] != original[input_index]:
            raise ValueError("非 -1 映射的变换前后对象身份必须完全相同")
        input_to_outputs[input_index].append(output_index)
        if input_index != output_index:
            affected_output.append(output_index)
    affected_input = tuple(
        input_index
        for input_index, output_indexes in enumerate(input_to_outputs)
        if output_indexes != [input_index]
    )
    return affected_input, tuple(affected_output)


@dataclass(frozen=True)
class PerturbationTrace:
    """一次变换的完整对象、位置映射、影响位置、来源和审计元数据。"""

    transform_key: tuple[int, ...]
    original: tuple[ObjectIdentity, ...]
    transformed: tuple[ObjectIdentity, ...]
    output_to_input: tuple[int, ...]
    affected_input_positions: tuple[int, ...]
    affected_output_positions: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    metadata_keys: tuple[tuple[int, ...], ...] = ()
    duplicate_of: SourceRef | None = None

    def __post_init__(self) -> None:
        """核验完整身份、双侧变化位置、来源和映射能够互相恢复。"""
        _integer_key(self.transform_key, where="PerturbationTrace.transform_key")
        _object_sequence(self.original, where="PerturbationTrace.original")
        _object_sequence(self.transformed, where="PerturbationTrace.transformed")
        if not self.original or not self.transformed:
            raise ValueError("扰动前后对象序列均不能为空")
        if not isinstance(self.output_to_input, tuple):
            raise TypeError("output_to_input 必须是整数 tuple")
        assert_int(*self.output_to_input, _where="PerturbationTrace.mapping")
        if len(self.output_to_input) != len(self.transformed):
            raise ValueError("output_to_input 必须与 transformed 等长")
        if any(
                type(index) is not int
                or index < -1
                or index >= len(self.original)
                for index in self.output_to_input):
            raise ValueError("output_to_input 只能引用原序列位置或使用 -1")
        _positions(
            self.affected_input_positions,
            size=len(self.original),
            where="PerturbationTrace.affected_input_positions",
        )
        _positions(
            self.affected_output_positions,
            size=len(self.transformed),
            where="PerturbationTrace.affected_output_positions",
        )
        derived_input, derived_output = _derived_affected_positions(
            self.original,
            self.transformed,
            self.output_to_input,
        )
        if self.affected_input_positions != derived_input:
            raise ValueError("affected_input_positions 与完整位置映射不一致")
        if self.affected_output_positions != derived_output:
            raise ValueError("affected_output_positions 与完整位置映射不一致")
        if not isinstance(self.source, SourceRef):
            raise TypeError("PerturbationTrace.source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("PerturbationTrace.scope 必须是 ScopeIdentity")
        if self.scope.source != self.source:
            raise ValueError("扰动 scope 必须指向同一 SourceRef")
        if not isinstance(self.metadata_keys, tuple):
            raise TypeError("metadata_keys 必须是整数 tuple 的 tuple")
        tuple(
            _integer_key(value, where="PerturbationTrace.metadata_key")
            for value in self.metadata_keys
        )
        if self.duplicate_of is not None:
            if not isinstance(self.duplicate_of, SourceRef):
                raise TypeError("duplicate_of 必须是 SourceRef 或 None")
            if self.duplicate_of != self.source:
                raise ValueError("同源重复诊断必须引用同一 SourceRef")
            if self.original != self.transformed:
                raise ValueError("同源重复诊断不得改变对象内容")
            if (self.output_to_input != tuple(range(len(self.original)))
                    or self.affected_input_positions
                    or self.affected_output_positions):
                raise ValueError("同源重复诊断必须使用无变化的身份位置映射")
        elif self.original == self.transformed:
            raise ValueError("非重复扰动必须真实改变对象序列")

    @property
    def is_duplicate_diagnostic(self) -> bool:
        """判断该 trace 是否只记录同一来源精确重放。"""
        return self.duplicate_of is not None

    def stable_key(self) -> tuple[int, ...]:
        """返回不丢对象、来源、映射和元数据的完整整数 trace。"""
        original = _pack_objects(self.original)
        transformed = _pack_objects(self.transformed)
        source = self.source.stable_key()
        scope = self.scope.stable_key()
        metadata: list[int] = [len(self.metadata_keys)]
        for key in self.metadata_keys:
            metadata.extend((len(key), *key))
        duplicate = (
            () if self.duplicate_of is None
            else self.duplicate_of.stable_key()
        )
        return (
            _TRACE_VERSION,
            len(self.transform_key),
            *self.transform_key,
            len(original),
            *original,
            len(transformed),
            *transformed,
            len(self.output_to_input),
            *self.output_to_input,
            len(self.affected_input_positions),
            *self.affected_input_positions,
            len(self.affected_output_positions),
            *self.affected_output_positions,
            len(source),
            *source,
            len(scope),
            *scope,
            len(metadata),
            *metadata,
            len(duplicate),
            *duplicate,
        )


def build_permutation_trace(
        units: tuple[ObjectIdentity, ...], *,
        output_order: tuple[int, ...],
        transform_key: tuple[int, ...],
        source: SourceRef,
        scope: ScopeIdentity,
        metadata_keys: tuple[tuple[int, ...], ...] = (),
        ) -> PerturbationTrace:
    """按调用方完整排列生成确定性乱序 trace，拒绝丢项、重复和恒等排列。"""
    units = _object_sequence(units, where="build_permutation_trace.units")
    if not isinstance(output_order, tuple):
        raise TypeError("output_order 必须是整数 tuple")
    assert_int(*output_order, _where="build_permutation_trace.output_order")
    if (len(output_order) != len(units)
            or tuple(sorted(output_order)) != tuple(range(len(units)))):
        raise ValueError("output_order 必须是原序列位置的完整排列")
    if output_order == tuple(range(len(units))):
        raise ValueError("恒等排列不是扰动")
    transformed = tuple(units[index] for index in output_order)
    affected_input, affected_output = _derived_affected_positions(
        units,
        transformed,
        output_order,
    )
    return PerturbationTrace(
        transform_key,
        units,
        transformed,
        output_order,
        affected_input,
        affected_output,
        source,
        scope,
        metadata_keys,
    )


def build_replacement_trace(
        original: tuple[ObjectIdentity, ...],
        transformed: tuple[ObjectIdentity, ...], *,
        output_to_input: tuple[int, ...],
        transform_key: tuple[int, ...],
        source: SourceRef,
        scope: ScopeIdentity,
        metadata_keys: tuple[tuple[int, ...], ...] = (),
        ) -> PerturbationTrace:
    """记录调用方已构造的增删替换，不赋予替换对象任何固定语义。"""
    original = _object_sequence(
        original, where="build_replacement_trace.original")
    transformed = _object_sequence(
        transformed, where="build_replacement_trace.transformed")
    if not isinstance(output_to_input, tuple):
        raise TypeError("output_to_input 必须是整数 tuple")
    assert_int(*output_to_input, _where="build_replacement_trace.mapping")
    if len(output_to_input) != len(transformed):
        raise ValueError("output_to_input 必须与 transformed 等长")
    if any(
            type(index) is not int
            or index < -1
            or index >= len(original)
            for index in output_to_input):
        raise ValueError("output_to_input 只能引用原序列位置或使用 -1")
    affected_input, affected_output = _derived_affected_positions(
        original,
        transformed,
        output_to_input,
    )
    return PerturbationTrace(
        transform_key,
        original,
        transformed,
        output_to_input,
        affected_input,
        affected_output,
        source,
        scope,
        metadata_keys,
    )


@dataclass(frozen=True)
class PerturbationAssessment:
    """verifier 对变换的三态裁决及可被定向反驳的候选集合。"""

    verdict: int
    detail_key: tuple[int, ...]
    refuted_hypotheses: tuple[HypothesisKey, ...] = ()

    def __post_init__(self) -> None:
        """核验三态裁决与允许反驳的候选集合保持一致。"""
        assert_int(self.verdict, _where="PerturbationAssessment.verdict")
        if type(self.verdict) is not int or self.verdict not in _ASSESSMENTS:
            raise ValueError("PerturbationAssessment.verdict 未注册")
        if not isinstance(self.refuted_hypotheses, tuple):
            raise TypeError("refuted_hypotheses 必须是 HypothesisKey tuple")
        if any(not isinstance(item, HypothesisKey)
               for item in self.refuted_hypotheses):
            raise TypeError("refuted_hypotheses 只能包含 HypothesisKey")
        if len(set(self.refuted_hypotheses)) != len(
                self.refuted_hypotheses):
            raise ValueError("refuted_hypotheses 不得重复")
        _integer_key(self.detail_key, where="PerturbationAssessment.detail_key")
        if self.verdict == ASSESSMENT_CHANGED:
            if not self.refuted_hypotheses:
                raise ValueError("CHANGED 裁决必须指出可反驳的相关 Hypothesis")
        elif self.refuted_hypotheses:
            raise ValueError("EQUIVALENT/UNKNOWN 裁决不得携带反驳候选")


PerturbationVerifier = Callable[
    [PerturbationTrace, tuple[HypothesisKey, ...]],
    PerturbationAssessment,
]


@dataclass(frozen=True)
class PerturbationEvidenceResult:
    """一次扰动裁决及实际写入 H-00 的候选快照。"""

    trace: PerturbationTrace
    assessment: PerturbationAssessment
    snapshots: tuple[HypothesisSnapshot, ...]
    evidence_ids: tuple[int, ...]


class PerturbationEngine:
    """执行注入式 verifier，并把变化或 unknown 定向写入既有 H-00 ledger。"""

    def __init__(
            self, protocol: PerturbationProtocol, *,
            ledger: HypothesisLedger,
            ) -> None:
        """绑定调用方协议和真正拥有候选的 H-00 ledger。"""
        if not isinstance(protocol, PerturbationProtocol):
            raise TypeError("protocol 必须是 PerturbationProtocol")
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("ledger 必须是 HypothesisLedger")
        self.protocol = protocol
        self.ledger = ledger

    def evaluate(
            self, trace: PerturbationTrace, *,
            candidates: tuple[HypothesisKey, ...],
            verifier: PerturbationVerifier,
            evidence_source: SourceRef,
            timestamp_seq: int,
            ) -> PerturbationEvidenceResult:
        """核验 trace 后追加定向 Evidence；重复诊断永远不得作为语义反例。"""
        if not isinstance(trace, PerturbationTrace):
            raise TypeError("trace 必须是 PerturbationTrace")
        if not isinstance(candidates, tuple) or not candidates:
            raise ValueError("扰动评测至少需要一个候选 Hypothesis")
        if any(not isinstance(item, HypothesisKey) for item in candidates):
            raise TypeError("candidates 只能包含 HypothesisKey")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidates 不得重复")
        for candidate in candidates:
            self.ledger.snapshot(candidate)
            if (candidate.observation != trace.source
                    or candidate.scope != trace.scope):
                raise ValueError("扰动候选必须与 trace 使用同一来源和 scope")
        if not callable(verifier):
            raise TypeError("verifier 必须可调用")
        if not isinstance(evidence_source, SourceRef):
            raise TypeError("evidence_source 必须是 SourceRef")
        assert_int(timestamp_seq, _where="PerturbationEngine.timestamp_seq")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("timestamp_seq 必须为非负严格整数")
        if (trace.is_duplicate_diagnostic
                and trace.transform_key
                != self.protocol.duplicate_transform_key):
            raise ValueError("同源重复 trace 必须使用协议注入的重复变换键")

        assessment = verifier(trace, candidates)
        if not isinstance(assessment, PerturbationAssessment):
            raise TypeError("verifier 必须返回 PerturbationAssessment")
        candidate_set = set(candidates)
        if any(item not in candidate_set
               for item in assessment.refuted_hypotheses):
            raise ValueError("verifier 不得反驳本次 candidates 外的 Hypothesis")
        if (trace.is_duplicate_diagnostic
                and assessment.verdict == ASSESSMENT_CHANGED):
            raise ValueError("同源精确重复不得被裁决为语义变化")

        if assessment.verdict == ASSESSMENT_EQUIVALENT:
            return PerturbationEvidenceResult(trace, assessment, (), ())
        if assessment.verdict == ASSESSMENT_CHANGED:
            targets = assessment.refuted_hypotheses
            stance = EVIDENCE_REFUTE
            reason_key = self.protocol.refute_reason_key
        else:
            targets = candidates
            stance = EVIDENCE_UNKNOWN
            reason_key = self.protocol.unknown_reason_key

        trace_key = trace.stable_key()
        evidence_ids: list[int] = []
        snapshots: list[HypothesisSnapshot] = []
        for hypothesis in targets:
            evidence_id = _EVIDENCE_HASHER.h63((
                hypothesis.stable_key(),
                trace_key,
                assessment.verdict,
                assessment.detail_key,
                evidence_source.stable_key(),
                timestamp_seq,
            )) or 1
            self.ledger.append_evidence(EvidenceRecord(
                evidence_id,
                hypothesis,
                stance,
                reason_key,
                evidence_source,
                timestamp_seq,
                payload=(
                    assessment.verdict,
                    len(assessment.detail_key),
                    *assessment.detail_key,
                    len(trace_key),
                    *trace_key,
                ),
            ))
            evidence_ids.append(evidence_id)
            snapshots.append(self.ledger.snapshot(hypothesis))
        return PerturbationEvidenceResult(
            trace,
            assessment,
            tuple(snapshots),
            tuple(evidence_ids),
        )


class SourceDuplicateLedger:
    """按 SourceRef 和事件键识别精确重放，拒绝同源同槽内容漂移。"""

    def __init__(self, duplicate_transform_key: tuple[int, ...]) -> None:
        """建立只以完整来源和事件键寻址的进程内重复登记表。"""
        self.duplicate_transform_key = _integer_key(
            duplicate_transform_key,
            where="SourceDuplicateLedger.duplicate_transform_key",
        )
        self._events: dict[
            tuple[int, ...], tuple[tuple[ObjectIdentity, ...], ScopeIdentity]
        ] = {}

    def register(
            self, units: tuple[ObjectIdentity, ...], *,
            source: SourceRef,
            scope: ScopeIdentity,
            event_key: tuple[int, ...],
            ) -> PerturbationTrace | None:
        """首次观察登记，精确重放返回诊断 trace，内容漂移则 fail closed。"""
        units = _object_sequence(units, where="SourceDuplicateLedger.units")
        if not units:
            raise ValueError("同源重复诊断的对象序列不能为空")
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if not isinstance(scope, ScopeIdentity) or scope.source != source:
            raise ValueError("scope 必须指向同一 SourceRef")
        event_key = _integer_key(
            event_key,
            where="SourceDuplicateLedger.event_key",
        )
        source_key = source.stable_key()
        slot = (
            len(source_key),
            *source_key,
            len(event_key),
            *event_key,
        )
        existing = self._events.get(slot)
        if existing is None:
            self._events[slot] = (units, scope)
            return None
        if existing != (units, scope):
            raise ValueError("同一 SourceRef 事件槽出现内容或 scope 漂移")
        return PerturbationTrace(
            self.duplicate_transform_key,
            units,
            units,
            tuple(range(len(units))),
            (),
            (),
            source,
            scope,
            metadata_keys=(event_key,),
            duplicate_of=source,
        )

    def clone(self) -> "SourceDuplicateLedger":
        """复制完整来源事件，供 V-06 沙箱独立记录重放。"""
        cloned = SourceDuplicateLedger(self.duplicate_transform_key)
        cloned._events = dict(self._events)
        return cloned

    def state_key(self) -> tuple:
        """返回完整事件身份、对象和 scope，不以摘要替代诊断依据。"""
        return tuple(
            (
                slot,
                _pack_objects(value[0]),
                value[1].stable_key(),
            )
            for slot, value in sorted(self._events.items())
        )


__all__ = [
    "ASSESSMENT_CHANGED",
    "ASSESSMENT_EQUIVALENT",
    "ASSESSMENT_UNKNOWN",
    "PerturbationAssessment",
    "PerturbationEngine",
    "PerturbationEvidenceResult",
    "PerturbationProtocol",
    "PerturbationTrace",
    "PerturbationVerifier",
    "SourceDuplicateLedger",
    "build_permutation_trace",
    "build_replacement_trace",
]
