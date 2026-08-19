"""DLG-05 独立对话 held-out family 的 typed 切分与 label-late 协议。

本模块只冻结评测边界，不执行问答、Memory 或生成。case 只携带输入内容
身份、会话 turn 结构和预注册轴；evaluator label 独立保存，必须在 selection-first
execution 完成后才能合并。这样可以先把会话评测施工面固定下来，避免把 expected
答案偷偷变成运行时查询输入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)


class ConversationHeldOutProtocolError(RuntimeError):
    """held-out family、label 或 observation 不闭合。"""


def _key(value: tuple[int, ...], *, label: str) -> ProtocolKey:
    """核验非空协议键并返回 value object。"""
    if not isinstance(value, tuple) or not value:
        raise ConversationHeldOutProtocolError(f"{label} 必须是非空整数 tuple")
    return ProtocolKey(value)


def _positive(value: int, *, label: str) -> int:
    """核验正严格整数。"""
    if type(value) is not int or value <= 0:
        raise ConversationHeldOutProtocolError(f"{label} 必须是正严格整数")
    return value


def _nonnegative(value: int, *, label: str) -> int:
    """核验非负严格整数。"""
    if type(value) is not int or value < 0:
        raise ConversationHeldOutProtocolError(f"{label} 必须是非负严格整数")
    return value


def _identity(value: CanonicalIdentity, *, label: str) -> CanonicalIdentity:
    """核验保留完整载荷的规范身份。"""
    if not isinstance(value, CanonicalIdentity):
        raise TypeError(f"{label} 必须是 CanonicalIdentity")
    return value


def _key_tuple(
        values: tuple[ProtocolKey, ...],
        *,
        label: str,
        allow_empty: bool = False,
        ) -> tuple[ProtocolKey, ...]:
    """核验协议键 tuple 的唯一性和稳定排序。"""
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise ConversationHeldOutProtocolError(f"{label} 不能为空")
    if any(not isinstance(item, ProtocolKey) for item in values):
        raise TypeError(f"{label} 含非法 ProtocolKey")
    if len(set(values)) != len(values):
        raise ConversationHeldOutProtocolError(f"{label} 不得重复")
    return values


def _key_sequence(
        values: tuple[ProtocolKey, ...],
        *,
        label: str,
        ) -> tuple[ProtocolKey, ...]:
    """核验有序协议序列；会话回合允许重复 response-act。"""
    if not isinstance(values, tuple) or not values:
        raise ConversationHeldOutProtocolError(f"{label} 不能为空")
    if any(not isinstance(item, ProtocolKey) for item in values):
        raise TypeError(f"{label} 含非法 ProtocolKey")
    return values


def _response_key(value: ProtocolKey, *, label: str) -> ProtocolKey:
    """核验 response-act 属于 DLG-05 四态协议。"""
    if value not in {
            RESPONSE_ANSWER,
            RESPONSE_UNKNOWN,
            RESPONSE_CLARIFY,
            RESPONSE_CONFLICT}:
        raise ConversationHeldOutProtocolError(f"{label} response-act 未注册")
    return value


def _candidate_keys(
        values: tuple[tuple[int, ...], ...],
        *,
        label: str,
        ) -> tuple[tuple[int, ...], ...]:
    """核验候选完整整数身份序列。"""
    if not isinstance(values, tuple):
        raise TypeError(f"{label} 必须是 tuple")
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, tuple) or not value:
            raise ConversationHeldOutProtocolError(
                f"{label}[{index}] 必须是非空整数 tuple")
        if any(type(item) is not int or item < 0 for item in value):
            raise ConversationHeldOutProtocolError(
                f"{label}[{index}] 必须使用非负严格整数")
        result.append(value)
    if len(set(result)) != len(result):
        raise ConversationHeldOutProtocolError(f"{label} 不得重复")
    return tuple(result)


def _pack_key(result: list[int], value: ProtocolKey) -> None:
    """把可变长度 ProtocolKey 以边界帧写入稳定键。"""
    result.extend((len(value.components), *value.components))


# 轴键只表达评测维度，不把自然语言枚举写进 runtime 语义层。
AXIS_SYNONYM = ProtocolKey((1, 1))
AXIS_ORDER = ProtocolKey((1, 2))
AXIS_OMISSION = ProtocolKey((1, 3))
AXIS_EXPLICIT_REPEAT = ProtocolKey((1, 4))
AXIS_PROPOSITION_REFERENCE = ProtocolKey((1, 5))
AXIS_EVENT_REFERENCE = ProtocolKey((1, 6))
AXIS_UNSEEN_SOURCE = ProtocolKey((1, 7))
AXIS_UNSEEN_RELATION = ProtocolKey((1, 8))
AXIS_CONFLICT = ProtocolKey((1, 9))
AXIS_MEMORY_MISS = ProtocolKey((1, 10))
AXIS_SCOPE_DRIFT = ProtocolKey((1, 11))
AXIS_ROLLBACK = ProtocolKey((1, 12))
AXIS_MEMORY_CAUSAL = ProtocolKey((1, 13))

MEMORY_OFF = ProtocolKey((2, 0))
MEMORY_ON = ProtocolKey((2, 1))

RESPONSE_ANSWER = ProtocolKey((3, 1))
RESPONSE_UNKNOWN = ProtocolKey((3, 2))
RESPONSE_CLARIFY = ProtocolKey((3, 3))
RESPONSE_CONFLICT = ProtocolKey((3, 4))

CONTEXT_FRESH = ProtocolKey((4, 1))
CONTEXT_CARRY = ProtocolKey((4, 2))
CONTEXT_EXPLICIT_REPEAT = ProtocolKey((4, 3))
CONTEXT_SCOPE_CHANGE = ProtocolKey((4, 4))

ROLLBACK_NONE = ProtocolKey((5, 0))
ROLLBACK_READ_ONLY = ProtocolKey((5, 1))

REFERENCE_NONE = ProtocolKey((6, 0))
REFERENCE_PROPOSITION = ProtocolKey((6, 2))
REFERENCE_EVENT = ProtocolKey((6, 3))


@dataclass(frozen=True, slots=True)
class ConversationHeldOutTurn:
    """一个不含 expected/label 的 typed 会话回合。"""

    turn_key: ProtocolKey
    ordinal: int
    content: CanonicalIdentity
    source_key: ProtocolKey
    scope_key: ProtocolKey
    context_mode: ProtocolKey
    memory_mode: ProtocolKey
    reference_mode: ProtocolKey
    rollback_mode: ProtocolKey

    def __post_init__(self) -> None:
        _key(self.turn_key.components, label="turn key")
        _positive(self.ordinal, label="turn ordinal")
        _identity(self.content, label="turn content")
        for name, value in (
                ("source_key", self.source_key),
                ("scope_key", self.scope_key),
                ("context_mode", self.context_mode),
                ("memory_mode", self.memory_mode),
                ("reference_mode", self.reference_mode),
                ("rollback_mode", self.rollback_mode)):
            if not isinstance(value, ProtocolKey):
                raise TypeError(f"turn {name} 必须是 ProtocolKey")
        if self.memory_mode not in {MEMORY_OFF, MEMORY_ON}:
            raise ConversationHeldOutProtocolError("turn memory_mode 未注册")
        if self.context_mode not in {
                CONTEXT_FRESH,
                CONTEXT_CARRY,
                CONTEXT_EXPLICIT_REPEAT,
                CONTEXT_SCOPE_CHANGE,
        }:
            raise ConversationHeldOutProtocolError("turn context_mode 未注册")
        if self.reference_mode not in {
                REFERENCE_NONE,
                REFERENCE_PROPOSITION,
                REFERENCE_EVENT,
        }:
            raise ConversationHeldOutProtocolError("turn reference_mode 未注册")
        if self.rollback_mode not in {ROLLBACK_NONE, ROLLBACK_READ_ONLY}:
            raise ConversationHeldOutProtocolError("turn rollback_mode 未注册")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含完整 content payload 的纯整数身份键。"""
        payload = self.content.payload
        result = [1]
        _pack_key(result, self.turn_key)
        result.append(self.ordinal)
        for value in (
                self.source_key,
                self.scope_key,
                self.context_mode,
                self.memory_mode,
                self.reference_mode,
                self.rollback_mode):
            _pack_key(result, value)
        result.extend((len(payload), *payload))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutCase:
    """一个独立 held-out case；没有任何 evaluator label 字段。"""

    case_key: ProtocolKey
    family_key: ProtocolKey
    axis_keys: tuple[ProtocolKey, ...]
    dedup_cluster: CanonicalIdentity
    provenance_cluster: CanonicalIdentity
    turns: tuple[ConversationHeldOutTurn, ...]

    def __post_init__(self) -> None:
        _key(self.case_key.components, label="case key")
        _key(self.family_key.components, label="case family key")
        _key_tuple(self.axis_keys, label="case axis keys")
        _identity(self.dedup_cluster, label="case dedup cluster")
        _identity(self.provenance_cluster, label="case provenance cluster")
        if (not isinstance(self.turns, tuple) or not self.turns
                or any(not isinstance(item, ConversationHeldOutTurn)
                       for item in self.turns)):
            raise ConversationHeldOutProtocolError("case turns 必须是非空 typed tuple")
        ordinals = tuple(item.ordinal for item in self.turns)
        if ordinals != tuple(sorted(ordinals)):
            raise ConversationHeldOutProtocolError("case turns 必须按 ordinal 排序")
        if len(set(item.turn_key for item in self.turns)) != len(self.turns):
            raise ConversationHeldOutProtocolError("case turn key 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回 case、轴、簇和全部 turn 的稳定整数键。"""
        result = [1]
        _pack_key(result, self.case_key)
        _pack_key(result, self.family_key)
        result.append(len(self.axis_keys))
        for key in self.axis_keys:
            _pack_key(result, key)
        for identity in (self.dedup_cluster, self.provenance_cluster):
            result.extend((len(identity.payload), *identity.payload))
        result.append(len(self.turns))
        for turn in self.turns:
            key = turn.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutManifest:
    """不含标签的 family manifest，供 selection-first runner 使用。"""

    version: int
    family_key: ProtocolKey
    train_contents: tuple[CanonicalIdentity, ...]
    train_dedup_clusters: tuple[CanonicalIdentity, ...]
    train_provenance_clusters: tuple[CanonicalIdentity, ...]
    cases: tuple[ConversationHeldOutCase, ...]
    required_axes: tuple[ProtocolKey, ...]
    required_memory_modes: tuple[ProtocolKey, ...]

    def __post_init__(self) -> None:
        _positive(self.version, label="manifest version")
        _key(self.family_key.components, label="manifest family key")
        for name, values in (
                ("train_contents", self.train_contents),
                ("train_dedup_clusters", self.train_dedup_clusters),
                ("train_provenance_clusters", self.train_provenance_clusters)):
            if not isinstance(values, tuple):
                raise TypeError(f"manifest {name} 必须是 tuple")
            if any(not isinstance(item, CanonicalIdentity) for item in values):
                raise TypeError(f"manifest {name} 含非法 identity")
            if len(set(item.payload for item in values)) != len(values):
                raise ConversationHeldOutProtocolError(f"manifest {name} 不得重复")
        if (not isinstance(self.cases, tuple) or not self.cases
                or any(not isinstance(item, ConversationHeldOutCase)
                       for item in self.cases)):
            raise ConversationHeldOutProtocolError("manifest cases 必须非空")
        if any(item.family_key != self.family_key for item in self.cases):
            raise ConversationHeldOutProtocolError("case family key 漂移")
        case_keys = tuple(item.case_key for item in self.cases)
        if len(set(case_keys)) != len(case_keys):
            raise ConversationHeldOutProtocolError("case key 不得重复")
        _key_tuple(self.required_axes, label="manifest required axes")
        _key_tuple(self.required_memory_modes, label="manifest required memory modes")
        if set(self.required_memory_modes) != {MEMORY_OFF, MEMORY_ON}:
            raise ConversationHeldOutProtocolError(
                "manifest 必须同时覆盖 Memory OFF/ON")
        observed_axes = {
            axis for case in self.cases for axis in case.axis_keys
        }
        missing_axes = set(self.required_axes) - observed_axes
        if missing_axes:
            raise ConversationHeldOutProtocolError(
                f"manifest 缺少 required held-out axes: {sorted(missing_axes)}")
        observed_modes = {
            turn.memory_mode for case in self.cases for turn in case.turns
        }
        if not set(self.required_memory_modes).issubset(observed_modes):
            raise ConversationHeldOutProtocolError(
                "manifest cases 未覆盖全部 Memory mode")
        train_contents = {item.payload for item in self.train_contents}
        train_dedup = {item.payload for item in self.train_dedup_clusters}
        train_provenance = {item.payload for item in self.train_provenance_clusters}
        for case in self.cases:
            if case.dedup_cluster.payload in train_dedup:
                raise ConversationHeldOutProtocolError(
                    "held-out dedup cluster 泄漏到 train")
            if case.provenance_cluster.payload in train_provenance:
                raise ConversationHeldOutProtocolError(
                    "held-out provenance cluster 泄漏到 train")
            if any(turn.content.payload in train_contents for turn in case.turns):
                raise ConversationHeldOutProtocolError(
                    "held-out content 泄漏到 train")

    def stable_key(self) -> tuple[int, ...]:
        """返回不包含 label 的完整 manifest 整数键。"""
        result = [self.version]
        _pack_key(result, self.family_key)
        for values in (
                self.train_contents,
                self.train_dedup_clusters,
                self.train_provenance_clusters):
            result.append(len(values))
            for item in values:
                result.extend((len(item.payload), *item.payload))
        result.append(len(self.cases))
        for case in self.cases:
            key = case.stable_key()
            result.extend((len(key), *key))
        for values in (self.required_axes, self.required_memory_modes):
            result.append(len(values))
            for item in values:
                _pack_key(result, item)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutLabel:
    """独立 evaluator label；不会进入 case 或 selection-first executor。"""

    case_key: ProtocolKey
    response_act: ProtocolKey
    turn_response_acts: tuple[ProtocolKey, ...]
    selected_candidate_keys: tuple[tuple[int, ...], ...]
    cited_source_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _key(self.case_key.components, label="label case key")
        _response_key(self.response_act, label="label")
        _key_sequence(self.turn_response_acts, label="label turn response acts")
        for value in self.turn_response_acts:
            _response_key(value, label="label turn")
        _candidate_keys(
            self.selected_candidate_keys,
            label="label selected candidates",
        )
        _candidate_keys(self.cited_source_keys, label="label cited sources")


@dataclass(frozen=True, slots=True)
class ConversationHeldOutLabelSet:
    """与 manifest 分离的完整标签包。"""

    manifest_key: tuple[int, ...]
    labels: tuple[ConversationHeldOutLabel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_key, tuple) or not self.manifest_key:
            raise ConversationHeldOutProtocolError("label set manifest key 非法")
        if any(type(item) is not int for item in self.manifest_key):
            raise ConversationHeldOutProtocolError("label set manifest key 非整数")
        if (not isinstance(self.labels, tuple) or not self.labels
                or any(not isinstance(item, ConversationHeldOutLabel)
                       for item in self.labels)):
            raise ConversationHeldOutProtocolError("label set labels 必须非空")
        keys = tuple(item.case_key for item in self.labels)
        if len(set(keys)) != len(keys):
            raise ConversationHeldOutProtocolError("label case key 不得重复")


@dataclass(frozen=True, slots=True)
class ConversationHeldOutObservation:
    """selection-first 执行产物，不携带 expected label。"""

    case_key: ProtocolKey
    response_act: ProtocolKey
    turn_response_acts: tuple[ProtocolKey, ...]
    selected_candidate_keys: tuple[tuple[int, ...], ...]
    cited_source_keys: tuple[tuple[int, ...], ...]
    context_revision: int
    memory_receipt_keys: tuple[tuple[int, ...], ...]
    memory_causal_proven: int = 0
    proven_axis_keys: tuple[ProtocolKey, ...] = ()

    def __post_init__(self) -> None:
        _key(self.case_key.components, label="observation case key")
        _response_key(self.response_act, label="observation")
        _key_sequence(self.turn_response_acts, label="observation turn acts")
        for value in self.turn_response_acts:
            _response_key(value, label="observation turn")
        _candidate_keys(
            self.selected_candidate_keys,
            label="observation selected candidates",
        )
        _candidate_keys(self.cited_source_keys, label="observation cited sources")
        _nonnegative(self.context_revision, label="observation context revision")
        if not isinstance(self.memory_receipt_keys, tuple):
            raise TypeError("observation memory receipts 必须是 tuple")
        for key in self.memory_receipt_keys:
            if not isinstance(key, tuple) or not key or any(
                    type(item) is not int or item < 0 for item in key):
                raise ConversationHeldOutProtocolError(
                    "observation memory receipt key 非法")
        if type(self.memory_causal_proven) is not int or (
                self.memory_causal_proven not in (0, 1)):
            raise ConversationHeldOutProtocolError(
                "observation memory causal proof 必须为 0/1")
        if (not isinstance(self.proven_axis_keys, tuple)
                or any(not isinstance(item, ProtocolKey)
                       for item in self.proven_axis_keys)
                or len(set(self.proven_axis_keys))
                != len(self.proven_axis_keys)):
            raise ConversationHeldOutProtocolError(
                "observation proven axis keys 非法")
        object.__setattr__(self, "proven_axis_keys", tuple(sorted(
            self.proven_axis_keys, key=lambda item: item.components)))

    def stable_key(self) -> tuple[int, ...]:
        """返回 selection-first observation 的完整整数键。"""
        result = [1]
        _pack_key(result, self.case_key)
        _pack_key(result, self.response_act)
        result.append(len(self.turn_response_acts))
        for value in self.turn_response_acts:
            _pack_key(result, value)
        for values, label in (
                (self.selected_candidate_keys, "selected candidates"),
                (self.cited_source_keys, "cited sources"),
                (self.memory_receipt_keys, "memory receipts")):
            result.append(len(values))
            for value in values:
                if not isinstance(value, tuple) \
                        or any(type(item) is not int for item in value):
                    raise ConversationHeldOutProtocolError(
                        f"observation {label} stable key 含非法 segment")
                result.extend((len(value), *value))
        result.append(len(self.proven_axis_keys))
        for value in self.proven_axis_keys:
            _pack_key(result, value)
        result.extend((self.context_revision, self.memory_causal_proven))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutEvaluation:
    """label-late 合并后的可审计摘要。"""

    manifest_key: tuple[int, ...]
    total: int
    passed: int
    failed_case_keys: tuple[ProtocolKey, ...]
    memory_off_total: int
    memory_on_total: int

    @property
    def complete(self) -> bool:
        """返回是否每个 held-out case 都有一致 observation。"""
        return self.total > 0 and self.total == self.passed


def run_selection_first(
        manifest: ConversationHeldOutManifest,
        execute: Callable[[ConversationHeldOutCase], ConversationHeldOutObservation],
        ) -> tuple[ConversationHeldOutObservation, ...]:
    """只把无 label manifest case 交给 executor，按 case 顺序收集 observation。"""
    if not isinstance(manifest, ConversationHeldOutManifest):
        raise TypeError("selection-first manifest 类型错误")
    if not callable(execute):
        raise TypeError("selection-first execute 必须可调用")
    observations = []
    for case in manifest.cases:
        observation = execute(case)
        if not isinstance(observation, ConversationHeldOutObservation):
            raise TypeError("selection-first executor 返回错误 observation")
        if observation.case_key != case.case_key:
            raise ConversationHeldOutProtocolError(
                "selection-first observation 替换了 case key")
        observations.append(observation)
    return tuple(observations)


def evaluate_label_late(
        manifest: ConversationHeldOutManifest,
        labels: ConversationHeldOutLabelSet,
        observations: tuple[ConversationHeldOutObservation, ...],
        ) -> ConversationHeldOutEvaluation:
    """在 observation 完成后才合并独立 labels，返回逐 case 结果。"""
    if not isinstance(manifest, ConversationHeldOutManifest):
        raise TypeError("label-late manifest 类型错误")
    if not isinstance(labels, ConversationHeldOutLabelSet):
        raise TypeError("label-late labels 类型错误")
    if labels.manifest_key != manifest.stable_key():
        raise ConversationHeldOutProtocolError(
            "label set manifest key 与执行 manifest 不一致")
    if (not isinstance(observations, tuple)
            or any(not isinstance(item, ConversationHeldOutObservation)
                   for item in observations)):
        raise TypeError("label-late observations 类型错误")
    expected_cases = {case.case_key for case in manifest.cases}
    label_by_case = {item.case_key: item for item in labels.labels}
    observation_by_case = {item.case_key: item for item in observations}
    if set(label_by_case) != expected_cases:
        raise ConversationHeldOutProtocolError(
            "label set 未逐 case 覆盖 manifest")
    if set(observation_by_case) != expected_cases:
        raise ConversationHeldOutProtocolError(
            "observations 未逐 case 覆盖 manifest")
    failed = []
    memory_off_total = 0
    memory_on_total = 0
    for case in manifest.cases:
        label = label_by_case[case.case_key]
        observation = observation_by_case[case.case_key]
        modes = {turn.memory_mode for turn in case.turns}
        if MEMORY_OFF in modes:
            memory_off_total += 1
        if MEMORY_ON in modes:
            memory_on_total += 1
        if not (
                len(label.turn_response_acts) == len(case.turns)
                and len(observation.turn_response_acts) == len(case.turns)
                and observation.response_act == label.response_act
                and observation.turn_response_acts == label.turn_response_acts
                and observation.selected_candidate_keys
                == label.selected_candidate_keys
                and observation.cited_source_keys == label.cited_source_keys):
            failed.append(case.case_key)
    return ConversationHeldOutEvaluation(
        manifest.stable_key(),
        len(manifest.cases),
        len(manifest.cases) - len(failed),
        tuple(failed),
        memory_off_total,
        memory_on_total,
    )


__all__ = [
    "AXIS_CONFLICT",
    "AXIS_EVENT_REFERENCE",
    "AXIS_EXPLICIT_REPEAT",
    "AXIS_MEMORY_MISS",
    "AXIS_MEMORY_CAUSAL",
    "AXIS_OMISSION",
    "AXIS_ORDER",
    "AXIS_PROPOSITION_REFERENCE",
    "AXIS_ROLLBACK",
    "AXIS_SCOPE_DRIFT",
    "AXIS_SYNONYM",
    "AXIS_UNSEEN_RELATION",
    "AXIS_UNSEEN_SOURCE",
    "CONTEXT_CARRY",
    "CONTEXT_EXPLICIT_REPEAT",
    "CONTEXT_FRESH",
    "CONTEXT_SCOPE_CHANGE",
    "REFERENCE_EVENT",
    "REFERENCE_NONE",
    "REFERENCE_PROPOSITION",
    "ROLLBACK_NONE",
    "ROLLBACK_READ_ONLY",
    "ConversationHeldOutCase",
    "ConversationHeldOutEvaluation",
    "ConversationHeldOutLabel",
    "ConversationHeldOutLabelSet",
    "ConversationHeldOutManifest",
    "ConversationHeldOutObservation",
    "ConversationHeldOutProtocolError",
    "ConversationHeldOutTurn",
    "MEMORY_OFF",
    "MEMORY_ON",
    "RESPONSE_ANSWER",
    "RESPONSE_CLARIFY",
    "RESPONSE_CONFLICT",
    "RESPONSE_UNKNOWN",
    "evaluate_label_late",
    "run_selection_first",
]
