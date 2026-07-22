"""Memory 不可变事件对象及其稳定整数协议。

本模块只定义事件真源，不计算当前聚合、召回分数或巩固门。语言、关系、意图、
信号和影响类型均通过一等引用注入；这里固定的整数只用于对象种类和序列化协议。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.formal_artifact import (
    describe_artifact_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    HypothesisKey,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_ARTIFACT,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    TypedRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    AssertionIdentity,
    CLOCK_MEMORY_RESOLVED,
    LogicalTimestamp,
    ScopeIdentity,
    SCOPE_SESSION,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
)


MEMORY_OBJECT_OBSERVATION = 1
MEMORY_OBJECT_EPISODE = 2
MEMORY_OBJECT_HYPOTHESIS = 3
MEMORY_OBJECT_EVIDENCE = 4
MEMORY_OBJECT_USE = 5
MEMORY_OBJECT_ARTIFACT = 6
MEMORY_OBJECT_CAPABILITY = 7
MEMORY_OBJECT_LEGACY_IMPORT = 8
MEMORY_OBJECT_PARSE_FAILURE = 9
MEMORY_OBJECT_INTAKE_MANIFEST = 10
MEMORY_OBJECT_RESOLUTION = 11
_MEMORY_OBJECT_KINDS = frozenset({
    MEMORY_OBJECT_OBSERVATION,
    MEMORY_OBJECT_EPISODE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_EVIDENCE,
    MEMORY_OBJECT_USE,
    MEMORY_OBJECT_ARTIFACT,
    MEMORY_OBJECT_CAPABILITY,
    MEMORY_OBJECT_LEGACY_IMPORT,
    MEMORY_OBJECT_PARSE_FAILURE,
    MEMORY_OBJECT_INTAKE_MANIFEST,
    MEMORY_OBJECT_RESOLUTION,
})

MEMORY_EVENT_OBSERVATION = 1
MEMORY_EVENT_EPISODE = 2
MEMORY_EVENT_HYPOTHESIS = 3
MEMORY_EVENT_EVIDENCE = 4
MEMORY_EVENT_USE = 5
MEMORY_EVENT_ARTIFACT = 6
MEMORY_EVENT_CAPABILITY = 7
MEMORY_EVENT_RETENTION = 8
MEMORY_EVENT_LIFECYCLE = 9
MEMORY_EVENT_LEGACY_IMPORT = 10
MEMORY_EVENT_PARSE_FAILURE = 11
MEMORY_EVENT_INTAKE_MANIFEST = 12
MEMORY_EVENT_DERIVATION = 13
MEMORY_EVENT_RESOLUTION = 14

INTAKE_OUTCOME_SUCCESS = 1
INTAKE_OUTCOME_FAILURE = 2

INTAKE_DERIVED_OBSERVATION = 1
INTAKE_DERIVED_HYPOTHESIS = 2
INTAKE_DERIVED_EVIDENCE = 3
INTAKE_DERIVED_FAILURE = 4
INTAKE_DERIVED_MANIFEST = 5
_INTAKE_DERIVED_OBJECT_KINDS = {
    INTAKE_DERIVED_OBSERVATION: MEMORY_OBJECT_OBSERVATION,
    INTAKE_DERIVED_HYPOTHESIS: MEMORY_OBJECT_HYPOTHESIS,
    INTAKE_DERIVED_EVIDENCE: MEMORY_OBJECT_EVIDENCE,
    INTAKE_DERIVED_FAILURE: MEMORY_OBJECT_PARSE_FAILURE,
    INTAKE_DERIVED_MANIFEST: MEMORY_OBJECT_INTAKE_MANIFEST,
}

RETENTION_EPISODIC = 1
RETENTION_CONSOLIDATED = 2

REFERENCE_CORE_TYPED = 1
REFERENCE_MEMORY_OBJECT = 2
REFERENCE_OBJECT_IDENTITY = 3
REFERENCE_ASSERTION = 4
_REFERENCE_KINDS = frozenset({
    REFERENCE_CORE_TYPED,
    REFERENCE_MEMORY_OBJECT,
    REFERENCE_OBJECT_IDENTITY,
    REFERENCE_ASSERTION,
})

LEGACY_TABLE_MEMORY_ITEM = 1
LEGACY_TABLE_EXPERIENCE_COUNT = 2

TIME_AXIS_CREATED = 1
TIME_AXIS_OBSERVED = 2
TIME_AXIS_USED = 3

_OBJECT_REF_VERSION = 1
_LINK_REF_VERSION = 1
_EVENT_VERSION = 1
_PAYLOAD_VERSION = 1


def _strict_tuple(value: tuple[int, ...], *, where: str,
                  allow_empty: bool = False) -> tuple[int, ...]:
    """校验开放整数键，拒绝 bool、浮点、截断空键和整数子类。"""
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise ValueError(f"{where} 必须是{'可空' if allow_empty else '非空'}整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _positive(value: int, *, where: str) -> int:
    """校验协议 kind 为严格正整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须为严格正整数")
    return value


def _nonnegative(value: int, *, where: str) -> int:
    """校验序号和旧字段为严格非负整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为非负严格整数")
    return value


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长整数键增加长度边界。"""
    return len(key), *key


def _take(values: tuple[int, ...], cursor: int, *, where: str,
          allow_empty: bool = False) -> tuple[tuple[int, ...], int]:
    """从长度前缀流读取一段，并拒绝负长度、非法空段和截断。"""
    if cursor >= len(values):
        raise ValueError(f"{where} 缺少长度")
    size = values[cursor]
    cursor += 1
    if size < 0 or (size == 0 and not allow_empty):
        raise ValueError(f"{where} 长度非法")
    end = cursor + size
    if end > len(values):
        raise ValueError(f"{where} 被截断")
    return values[cursor:end], end


def _finish(values: tuple[int, ...], cursor: int, *, where: str) -> None:
    """核验稳定键已被完整消费，拒绝尾随字段。"""
    if cursor != len(values):
        raise ValueError(f"{where} 含尾随字段")


def source_reparse_lineage_key(source: SourceRef) -> tuple[int, ...]:
    """返回只排除 parser version 的来源谱系键，供显式重新解析核验。"""
    if not isinstance(source, SourceRef):
        raise TypeError("source 必须是 SourceRef")
    versions = source.versions
    return (
        source.source_kind,
        source.source_id,
        source.document_id,
        *source.owner.stable_key(),
        versions.corpus.value,
        versions.primitive.value,
        versions.curriculum.value,
    )


@dataclass(frozen=True, order=True)
class MemoryObjectRef:
    """一个 Memory 空间内带 owner、版本和开放对象键的稳定引用。"""

    memory_space: SpaceIdentity
    owner: OwnerScope
    versions: VersionBundle
    object_kind: int
    object_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验引用只能属于 Memory，且对象种类和键均已注册。"""
        if not isinstance(self.memory_space, SpaceIdentity):
            raise TypeError("MemoryObjectRef.memory_space 必须是 SpaceIdentity")
        if self.memory_space.space_type != SPACE_TYPE_MEMORY:
            raise ValueError("MemoryObjectRef 只能属于 Memory 空间")
        if not isinstance(self.owner, OwnerScope):
            raise TypeError("MemoryObjectRef.owner 必须是 OwnerScope")
        if not isinstance(self.versions, VersionBundle):
            raise TypeError("MemoryObjectRef.versions 必须是 VersionBundle")
        _positive(self.object_kind, where="MemoryObjectRef.object_kind")
        if self.object_kind not in _MEMORY_OBJECT_KINDS:
            raise ValueError("MemoryObjectRef.object_kind 未注册")
        _strict_tuple(self.object_key, where="MemoryObjectRef.object_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含稳定空间、owner、版本、种类和完整对象键的身份。"""
        return (
            _OBJECT_REF_VERSION,
            *self.memory_space.stable_key(),
            *self.owner.stable_key(),
            *self.versions.stable_key(),
            self.object_kind,
            *_pack(self.object_key),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "MemoryObjectRef":
        """从完整整数键恢复 Memory 对象引用。"""
        key = _strict_tuple(key, where="MemoryObjectRef.stable_key")
        if len(key) < 14 or key[0] != _OBJECT_REF_VERSION:
            raise ValueError("MemoryObjectRef 稳定键版本或长度非法")
        object_key, cursor = _take(
            key, 13, where="MemoryObjectRef.object_key")
        _finish(key, cursor, where="MemoryObjectRef.stable_key")
        return cls(
            SpaceIdentity(*key[1:4]),
            OwnerScope(*key[4:8]),
            VersionBundle(
                CorpusVersion(key[8]),
                ParserVersion(key[9]),
                PrimitiveVersion(key[10]),
                CurriculumVersion(key[11]),
            ),
            key[12],
            object_key,
        )


@dataclass(frozen=True, order=True)
class MemoryLinkedRef:
    """事件字段可引用的 Core、Memory、权威对象或 scoped assertion。"""

    reference_kind: int
    reference_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """逐类 round-trip 引用，禁止开放裸整数伪装成一等对象。"""
        _positive(self.reference_kind, where="MemoryLinkedRef.reference_kind")
        if self.reference_kind not in _REFERENCE_KINDS:
            raise ValueError("MemoryLinkedRef.reference_kind 未注册")
        key = _strict_tuple(
            self.reference_key, where="MemoryLinkedRef.reference_key")
        restored = self.value()
        if restored.stable_key() != key:
            raise ValueError("MemoryLinkedRef 引用不能稳定 round-trip")

    @classmethod
    def core(cls, ref: TypedRef) -> "MemoryLinkedRef":
        """把已物化 Core typed ref 包装为事件引用。"""
        if not isinstance(ref, TypedRef):
            raise TypeError("core ref 必须是 TypedRef")
        return cls(REFERENCE_CORE_TYPED, ref.stable_key())

    @classmethod
    def memory(cls, ref: MemoryObjectRef) -> "MemoryLinkedRef":
        """把 Memory 对象引用包装为通用事件引用。"""
        if not isinstance(ref, MemoryObjectRef):
            raise TypeError("memory ref 必须是 MemoryObjectRef")
        return cls(REFERENCE_MEMORY_OBJECT, ref.stable_key())

    @classmethod
    def object(cls, identity: ObjectIdentity) -> "MemoryLinkedRef":
        """把权威非编址对象身份包装为事件引用。"""
        if not isinstance(identity, ObjectIdentity):
            raise TypeError("object identity 必须是 ObjectIdentity")
        return cls(REFERENCE_OBJECT_IDENTITY, identity.stable_key())

    @classmethod
    def assertion(cls, identity: AssertionIdentity) -> "MemoryLinkedRef":
        """把来源化 scoped assertion 包装为事件引用。"""
        if not isinstance(identity, AssertionIdentity):
            raise TypeError("assertion identity 必须是 AssertionIdentity")
        return cls(REFERENCE_ASSERTION, identity.stable_key())

    def value(self) -> TypedRef | MemoryObjectRef | ObjectIdentity | AssertionIdentity:
        """按引用种类恢复领域对象，不解释其语义。"""
        parsers = {
            REFERENCE_CORE_TYPED: TypedRef.from_stable_key,
            REFERENCE_MEMORY_OBJECT: MemoryObjectRef.from_stable_key,
            REFERENCE_OBJECT_IDENTITY: ObjectIdentity.from_stable_key,
            REFERENCE_ASSERTION: AssertionIdentity.from_stable_key,
        }
        parser = parsers.get(self.reference_kind)
        if parser is None:
            raise ValueError("MemoryLinkedRef.reference_kind 未注册")
        return parser(self.reference_key)

    def stable_key(self) -> tuple[int, ...]:
        """返回带种类和长度边界的引用键。"""
        return (
            _LINK_REF_VERSION,
            self.reference_kind,
            *_pack(self.reference_key),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "MemoryLinkedRef":
        """从完整整数键恢复分型引用。"""
        key = _strict_tuple(key, where="MemoryLinkedRef.stable_key")
        if len(key) < 4 or key[0] != _LINK_REF_VERSION:
            raise ValueError("MemoryLinkedRef 稳定键版本或长度非法")
        reference_key, cursor = _take(
            key, 2, where="MemoryLinkedRef.reference_key")
        _finish(key, cursor, where="MemoryLinkedRef.stable_key")
        return cls(key[1], reference_key)

    def core_refs(self) -> tuple[TypedRef, ...]:
        """返回该引用直接包含的 Core typed 端点。"""
        value = self.value()
        if isinstance(value, TypedRef):
            return (value,)
        if isinstance(value, AssertionIdentity):
            return value.subject, value.object
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回该引用直接包含的 Memory 对象端点。"""
        value = self.value()
        return (value,) if isinstance(value, MemoryObjectRef) else ()


class MemoryPayload(Protocol):
    """所有事件载荷必须提供稳定键、引用集合和主逻辑时间。"""

    def stable_key(self) -> tuple[int, ...]: ...
    def timestamp(self) -> LogicalTimestamp: ...
    def core_refs(self) -> tuple[TypedRef, ...]: ...
    def memory_refs(self) -> tuple[MemoryObjectRef, ...]: ...


def _link_keys(refs: tuple[MemoryLinkedRef, ...]) -> tuple[int, ...]:
    """把有序引用序列编码为逐项长度前缀流。"""
    result: list[int] = [len(refs)]
    for ref in refs:
        result.extend(_pack(ref.stable_key()))
    return tuple(result)


def _take_links(values: tuple[int, ...], cursor: int, *, where: str
                ) -> tuple[tuple[MemoryLinkedRef, ...], int]:
    """恢复有序通用引用序列，并保持重复与次序。"""
    if cursor >= len(values):
        raise ValueError(f"{where} 缺少引用数量")
    count = values[cursor]
    cursor += 1
    if count < 0:
        raise ValueError(f"{where} 引用数量非法")
    result: list[MemoryLinkedRef] = []
    for index in range(count):
        key, cursor = _take(
            values, cursor, where=f"{where}[{index}]")
        result.append(MemoryLinkedRef.from_stable_key(key))
    return tuple(result), cursor


def _memory_keys(refs: tuple[MemoryObjectRef, ...]) -> tuple[int, ...]:
    """把有序 Memory 引用序列编码为逐项长度前缀流。"""
    result: list[int] = [len(refs)]
    for ref in refs:
        result.extend(_pack(ref.stable_key()))
    return tuple(result)


def _take_memory_refs(values: tuple[int, ...], cursor: int, *, where: str
                      ) -> tuple[tuple[MemoryObjectRef, ...], int]:
    """恢复有序 Memory 对象引用序列。"""
    if cursor >= len(values):
        raise ValueError(f"{where} 缺少引用数量")
    count = values[cursor]
    cursor += 1
    if count < 0:
        raise ValueError(f"{where} 引用数量非法")
    result: list[MemoryObjectRef] = []
    for index in range(count):
        key, cursor = _take(
            values, cursor, where=f"{where}[{index}]")
        result.append(MemoryObjectRef.from_stable_key(key))
    return tuple(result), cursor


def _typed_keys(refs: tuple[TypedRef, ...]) -> tuple[int, ...]:
    """把固定长度 TypedRef 序列编码为有数量边界的整数流。"""
    return (
        len(refs),
        *(value for ref in refs for value in ref.stable_key()),
    )


def _take_typed_refs(values: tuple[int, ...], cursor: int, *, where: str
                     ) -> tuple[tuple[TypedRef, ...], int]:
    """按 TypedRef 自描述长度恢复一组 Core 引用。"""
    if cursor >= len(values):
        raise ValueError(f"{where} 缺少引用数量")
    count = values[cursor]
    cursor += 1
    if count < 0:
        raise ValueError(f"{where} 引用数量非法")
    result: list[TypedRef] = []
    for index in range(count):
        if cursor >= len(values):
            raise ValueError(f"{where}[{index}] 被截断")
        # TypedRef 当前稳定键固定 11 项；解析器继续负责字段核验。
        end = cursor + 11
        if end > len(values):
            raise ValueError(f"{where}[{index}] 被截断")
        result.append(TypedRef.from_stable_key(values[cursor:end]))
        cursor = end
    return tuple(result), cursor


def _payload_refs(refs: tuple[MemoryLinkedRef, ...]
                  ) -> tuple[tuple[TypedRef, ...], tuple[MemoryObjectRef, ...]]:
    """汇总通用引用中的 Core 和 Memory 端点，供 facade 做完整性核验。"""
    core: list[TypedRef] = []
    memory: list[MemoryObjectRef] = []
    for ref in refs:
        core.extend(ref.core_refs())
        memory.extend(ref.memory_refs())
    return tuple(core), tuple(memory)


@dataclass(frozen=True)
class ObservationPayload:
    """某来源和上下文中观察到的对象、顺序、结构、命题与关系出现。"""

    source: SourceRef
    context: MemoryLinkedRef
    concept_refs: tuple[TypedRef, ...]
    ordered_refs: tuple[TypedRef, ...]
    structure_ref: TypedRef | None
    proposition_refs: tuple[TypedRef, ...]
    relation_occurrences: tuple[MemoryLinkedRef, ...]
    observed_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验来源、上下文、引用集合和 observed 时钟完整。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("Observation.source 必须是 SourceRef")
        if not isinstance(self.context, MemoryLinkedRef):
            raise TypeError("Observation.context 必须是一等引用")
        for name, refs in (
                ("concept_refs", self.concept_refs),
                ("ordered_refs", self.ordered_refs),
                ("proposition_refs", self.proposition_refs)):
            if not isinstance(refs, tuple) or any(
                    not isinstance(ref, TypedRef) for ref in refs):
                raise TypeError(f"Observation.{name} 必须是 TypedRef tuple")
        if self.structure_ref is not None and not isinstance(
                self.structure_ref, TypedRef):
            raise TypeError("Observation.structure_ref 必须是 TypedRef 或 None")
        if not isinstance(self.relation_occurrences, tuple) or any(
                not isinstance(ref, MemoryLinkedRef)
                for ref in self.relation_occurrences):
            raise TypeError("Observation.relation_occurrences 类型错误")
        if not isinstance(self.observed_at, LogicalTimestamp):
            raise TypeError("Observation.observed_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回可无损恢复全部观察字段的稳定键。"""
        structure = (() if self.structure_ref is None
                     else self.structure_ref.stable_key())
        return (
            _PAYLOAD_VERSION,
            *_pack(self.source.stable_key()),
            *_pack(self.context.stable_key()),
            *_typed_keys(self.concept_refs),
            *_typed_keys(self.ordered_refs),
            *_pack(structure),
            *_typed_keys(self.proposition_refs),
            *_link_keys(self.relation_occurrences),
            *_pack(self.observed_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ObservationPayload":
        """从完整键恢复 Observation，拒绝截断和尾随。"""
        key = _strict_tuple(key, where="Observation.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Observation payload 版本未注册")
        source_key, cursor = _take(key, 1, where="Observation.source")
        context_key, cursor = _take(key, cursor, where="Observation.context")
        concepts, cursor = _take_typed_refs(
            key, cursor, where="Observation.concept_refs")
        ordered, cursor = _take_typed_refs(
            key, cursor, where="Observation.ordered_refs")
        structure_key, cursor = _take(
            key, cursor, where="Observation.structure_ref", allow_empty=True)
        propositions, cursor = _take_typed_refs(
            key, cursor, where="Observation.proposition_refs")
        relations, cursor = _take_links(
            key, cursor, where="Observation.relation_occurrences")
        timestamp_key, cursor = _take(
            key, cursor, where="Observation.observed_at")
        _finish(key, cursor, where="Observation.stable_key")
        return cls(
            SourceRef.from_stable_key(source_key),
            MemoryLinkedRef.from_stable_key(context_key),
            concepts,
            ordered,
            None if not structure_key else TypedRef.from_stable_key(structure_key),
            propositions,
            relations,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 Observation 的 observed 逻辑时间。"""
        return self.observed_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """返回观察直接引用的全部 Core typed identity。"""
        relation_core, _ = _payload_refs(self.relation_occurrences)
        structure = () if self.structure_ref is None else (self.structure_ref,)
        return (
            *self.context.core_refs(), *self.concept_refs,
            *self.ordered_refs, *structure, *self.proposition_refs,
            *relation_core,
        )

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回上下文或关系出现中显式引用的 Memory 对象。"""
        _, relation_memory = _payload_refs(self.relation_occurrences)
        return (*self.context.memory_refs(), *relation_memory)


@dataclass(frozen=True)
class ParseFailurePayload:
    """一次来源解析在 Core 物化前失败的结构化记录。"""

    source: SourceRef
    failure_kind: MemoryLinkedRef
    batch_id: int
    parser_version: int
    diagnostic_key: tuple[int, ...]
    failed_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验失败类型由一等引用注入，批次和 parser 版本可回溯。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("ParseFailure.source 必须是 SourceRef")
        if not isinstance(self.failure_kind, MemoryLinkedRef):
            raise TypeError("ParseFailure.failure_kind 必须是一等引用")
        _positive(self.batch_id, where="ParseFailure.batch_id")
        _positive(self.parser_version, where="ParseFailure.parser_version")
        if self.parser_version != self.source.versions.parser.value:
            raise ValueError("ParseFailure parser version 与来源版本不一致")
        _strict_tuple(
            self.diagnostic_key, where="ParseFailure.diagnostic_key")
        if not isinstance(self.failed_at, LogicalTimestamp):
            raise TypeError("ParseFailure.failed_at 必须是 LogicalTimestamp")

    def identity_key(self) -> tuple[int, ...]:
        """返回每个精确来源版本唯一一次解析失败的对象键。"""
        return (_PAYLOAD_VERSION, *_pack(self.source.stable_key()))

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、失败类型、批次、诊断和逻辑时间的完整稳定键。"""
        return (
            _PAYLOAD_VERSION,
            *_pack(self.source.stable_key()),
            *_pack(self.failure_kind.stable_key()),
            self.batch_id,
            self.parser_version,
            *_pack(self.diagnostic_key),
            *_pack(self.failed_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ParseFailurePayload":
        """从稳定键恢复解析失败，拒绝字段截断或尾随。"""
        key = _strict_tuple(key, where="ParseFailure.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("ParseFailure payload 版本未注册")
        source_key, cursor = _take(key, 1, where="ParseFailure.source")
        failure_key, cursor = _take(
            key, cursor, where="ParseFailure.failure_kind")
        if cursor + 2 > len(key):
            raise ValueError("ParseFailure 批次或 parser version 被截断")
        batch_id, parser_version = key[cursor:cursor + 2]
        diagnostic_key, cursor = _take(
            key, cursor + 2, where="ParseFailure.diagnostic_key")
        timestamp_key, cursor = _take(
            key, cursor, where="ParseFailure.failed_at")
        _finish(key, cursor, where="ParseFailure.stable_key")
        return cls(
            SourceRef.from_stable_key(source_key),
            MemoryLinkedRef.from_stable_key(failure_key),
            batch_id,
            parser_version,
            diagnostic_key,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回解析失败发生的逻辑时间。"""
        return self.failed_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """返回失败类型引用中显式携带的 Core identity。"""
        return self.failure_kind.core_refs()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回失败类型引用中显式携带的 Memory identity。"""
        return self.failure_kind.memory_refs()


@dataclass(frozen=True, order=True)
class IntakeDerivedBinding:
    """manifest 中一个稳定 lineage key 到派生 Memory 对象的绑定。"""

    binding_kind: int
    lineage_key: tuple[int, ...]
    object_ref: MemoryObjectRef

    def __post_init__(self) -> None:
        """核验绑定种类、开放谱系键和对象种类严格一致。"""
        expected = _INTAKE_DERIVED_OBJECT_KINDS.get(self.binding_kind)
        if expected is None:
            raise ValueError("IntakeDerivedBinding.binding_kind 未注册")
        _strict_tuple(
            self.lineage_key, where="IntakeDerivedBinding.lineage_key")
        if (not isinstance(self.object_ref, MemoryObjectRef)
                or self.object_ref.object_kind != expected):
            raise ValueError("IntakeDerivedBinding 对象种类与 binding kind 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回绑定种类、完整谱系键和对象引用。"""
        return (
            self.binding_kind,
            *_pack(self.lineage_key),
            *_pack(self.object_ref.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "IntakeDerivedBinding":
        """从稳定键恢复一个派生绑定。"""
        key = _strict_tuple(key, where="IntakeDerivedBinding.stable_key")
        lineage_key, cursor = _take(
            key, 1, where="IntakeDerivedBinding.lineage_key")
        object_key, cursor = _take(
            key, cursor, where="IntakeDerivedBinding.object_ref")
        _finish(key, cursor, where="IntakeDerivedBinding.stable_key")
        return cls(
            key[0], lineage_key, MemoryObjectRef.from_stable_key(object_key))


@dataclass(frozen=True)
class IntakeManifestPayload:
    """一次来源版本摄入的成功或失败派生清单。"""

    source: SourceRef
    batch_id: int
    parser_version: int
    outcome_kind: int
    bindings: tuple[IntakeDerivedBinding, ...]
    supersedes_manifest_ref: MemoryObjectRef | None
    completed_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验 outcome、来源版本、派生一一对应和显式 reparse 前驱。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("IntakeManifest.source 必须是 SourceRef")
        _positive(self.batch_id, where="IntakeManifest.batch_id")
        _positive(self.parser_version, where="IntakeManifest.parser_version")
        if self.parser_version != self.source.versions.parser.value:
            raise ValueError("IntakeManifest parser version 与来源版本不一致")
        if self.outcome_kind not in {
                INTAKE_OUTCOME_SUCCESS, INTAKE_OUTCOME_FAILURE}:
            raise ValueError("IntakeManifest.outcome_kind 未注册")
        if (not isinstance(self.bindings, tuple) or not self.bindings
                or any(not isinstance(item, IntakeDerivedBinding)
                       for item in self.bindings)):
            raise TypeError("IntakeManifest.bindings 必须是非空绑定 tuple")
        binding_keys = {
            (item.binding_kind, item.lineage_key) for item in self.bindings
        }
        if len(binding_keys) != len(self.bindings):
            raise ValueError("IntakeManifest 同种派生含重复 lineage key")
        refs = {item.object_ref for item in self.bindings}
        if len(refs) != len(self.bindings):
            raise ValueError("IntakeManifest 不得用一个对象冒充多个派生绑定")
        spaces = {item.object_ref.memory_space for item in self.bindings}
        if len(spaces) != 1 or any(
                item.object_ref.owner != self.source.owner
                or item.object_ref.versions != self.source.versions
                for item in self.bindings):
            raise ValueError("IntakeManifest 派生对象空间、owner 或版本漂移")

        grouped = {
            kind: {
                item.lineage_key for item in self.bindings
                if item.binding_kind == kind
            }
            for kind in _INTAKE_DERIVED_OBJECT_KINDS
        }
        if self.outcome_kind == INTAKE_OUTCOME_SUCCESS:
            if (len(grouped[INTAKE_DERIVED_OBSERVATION]) != 1
                    or grouped[INTAKE_DERIVED_FAILURE]
                    or grouped[INTAKE_DERIVED_MANIFEST]
                    or grouped[INTAKE_DERIVED_HYPOTHESIS]
                    != grouped[INTAKE_DERIVED_EVIDENCE]):
                raise ValueError("成功 manifest 必须含一个 Observation 和逐候选唯一 Evidence")
        elif (len(grouped[INTAKE_DERIVED_FAILURE]) != 1
              or any(grouped[kind] for kind in (
                  INTAKE_DERIVED_OBSERVATION,
                  INTAKE_DERIVED_HYPOTHESIS,
                  INTAKE_DERIVED_EVIDENCE,
                  INTAKE_DERIVED_MANIFEST,
              ))):
            raise ValueError("失败 manifest 只能含一个 ParseFailure")

        prior = self.supersedes_manifest_ref
        if prior is not None:
            if (not isinstance(prior, MemoryObjectRef)
                    or prior.object_kind != MEMORY_OBJECT_INTAKE_MANIFEST
                    or prior.memory_space not in spaces
                    or prior.owner != self.source.owner):
                raise ValueError("IntakeManifest reparse 前驱引用非法")
            prior_source = SourceRef.from_stable_key(prior.object_key)
            if (source_reparse_lineage_key(prior_source)
                    != source_reparse_lineage_key(self.source)
                    or prior_source.versions.parser.value
                    >= self.parser_version):
                raise ValueError("IntakeManifest 前驱不是同谱系更早 parser 版本")
        if not isinstance(self.completed_at, LogicalTimestamp):
            raise TypeError("IntakeManifest.completed_at 必须是 LogicalTimestamp")

    def identity_key(self) -> tuple[int, ...]:
        """返回当前 Memory 空间内每个精确 SourceRef 唯一的 manifest 键。"""
        return self.source.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、结果、全部派生绑定、前驱和逻辑时间。"""
        values: list[int] = [
            _PAYLOAD_VERSION,
            *_pack(self.source.stable_key()),
            self.batch_id,
            self.parser_version,
            self.outcome_kind,
            len(self.bindings),
        ]
        for binding in self.bindings:
            values.extend(_pack(binding.stable_key()))
        values.extend(_pack(
            () if self.supersedes_manifest_ref is None
            else self.supersedes_manifest_ref.stable_key()))
        values.extend(_pack(self.completed_at.stable_key()))
        return tuple(values)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "IntakeManifestPayload":
        """从稳定键恢复完整 intake manifest。"""
        key = _strict_tuple(key, where="IntakeManifest.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("IntakeManifest payload 版本未注册")
        source_key, cursor = _take(key, 1, where="IntakeManifest.source")
        if cursor + 4 > len(key):
            raise ValueError("IntakeManifest 固定字段被截断")
        batch_id, parser_version, outcome_kind, binding_count = (
            key[cursor:cursor + 4])
        cursor += 4
        if binding_count <= 0:
            raise ValueError("IntakeManifest binding_count 非法")
        bindings: list[IntakeDerivedBinding] = []
        for _ in range(binding_count):
            binding_key, cursor = _take(
                key, cursor, where="IntakeManifest.binding")
            bindings.append(IntakeDerivedBinding.from_stable_key(binding_key))
        prior_key, cursor = _take(
            key, cursor, where="IntakeManifest.supersedes", allow_empty=True)
        timestamp_key, cursor = _take(
            key, cursor, where="IntakeManifest.completed_at")
        _finish(key, cursor, where="IntakeManifest.stable_key")
        return cls(
            SourceRef.from_stable_key(source_key),
            batch_id,
            parser_version,
            outcome_kind,
            tuple(bindings),
            None if not prior_key else MemoryObjectRef.from_stable_key(prior_key),
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回本次摄入完成的逻辑时间。"""
        return self.completed_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """manifest 只列 Memory 派生对象，不复制其 Core 引用。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回全部派生对象和可选前驱 manifest。"""
        prior = (() if self.supersedes_manifest_ref is None
                 else (self.supersedes_manifest_ref,))
        return tuple(item.object_ref for item in self.bindings) + prior


@dataclass(frozen=True)
class EpisodePayload:
    """一次输入、候选、选择、输出、使用声明和结果的完整处理记录。"""

    input_observation_ref: MemoryObjectRef
    intent_ref: MemoryLinkedRef | None
    candidate_refs: tuple[MemoryLinkedRef, ...]
    selected_path_ref: MemoryLinkedRef | None
    output_ref: MemoryLinkedRef | None
    used_memory_refs: tuple[MemoryObjectRef, ...]
    result_signal_refs: tuple[MemoryLinkedRef, ...]
    failure_ref: MemoryLinkedRef | None
    round_seq: int
    session_ref: ScopeIdentity
    created_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验 Episode 的对象分型、session 边界和逻辑创建序。"""
        if (not isinstance(self.input_observation_ref, MemoryObjectRef)
                or self.input_observation_ref.object_kind
                != MEMORY_OBJECT_OBSERVATION):
            raise ValueError("Episode.input_observation_ref 必须指向 Observation")
        for name, value in (
                ("intent_ref", self.intent_ref),
                ("selected_path_ref", self.selected_path_ref),
                ("output_ref", self.output_ref),
                ("failure_ref", self.failure_ref)):
            if value is not None and not isinstance(value, MemoryLinkedRef):
                raise TypeError(f"Episode.{name} 必须是一等引用或 None")
        for name, refs, expected in (
                ("candidate_refs", self.candidate_refs, MemoryLinkedRef),
                ("used_memory_refs", self.used_memory_refs, MemoryObjectRef),
                ("result_signal_refs", self.result_signal_refs, MemoryLinkedRef)):
            if not isinstance(refs, tuple) or any(
                    not isinstance(ref, expected) for ref in refs):
                raise TypeError(f"Episode.{name} 类型错误")
        _nonnegative(self.round_seq, where="Episode.round_seq")
        if (not isinstance(self.session_ref, ScopeIdentity)
                or self.session_ref.scope_kind != SCOPE_SESSION):
            raise ValueError("Episode.session_ref 必须是 session ScopeIdentity")
        if not isinstance(self.created_at, LogicalTimestamp):
            raise TypeError("Episode.created_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回 Episode 全字段稳定键；used refs 不因此自动生成 Use。"""
        optional = (
            self.intent_ref,
            self.selected_path_ref,
            self.output_ref,
            self.failure_ref,
        )
        result: list[int] = [
            _PAYLOAD_VERSION,
            *_pack(self.input_observation_ref.stable_key()),
        ]
        for ref in optional[:1]:
            result.extend(_pack(() if ref is None else ref.stable_key()))
        result.extend(_link_keys(self.candidate_refs))
        for ref in optional[1:3]:
            result.extend(_pack(() if ref is None else ref.stable_key()))
        result.extend(_memory_keys(self.used_memory_refs))
        result.extend(_link_keys(self.result_signal_refs))
        result.extend(_pack(
            () if self.failure_ref is None else self.failure_ref.stable_key()))
        result.extend((self.round_seq,))
        result.extend(_pack(self.session_ref.stable_key()))
        result.extend(_pack(self.created_at.stable_key()))
        return tuple(result)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "EpisodePayload":
        """从稳定键恢复一次完整 Episode。"""
        key = _strict_tuple(key, where="Episode.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Episode payload 版本未注册")
        observation_key, cursor = _take(
            key, 1, where="Episode.input_observation_ref")
        intent_key, cursor = _take(
            key, cursor, where="Episode.intent_ref", allow_empty=True)
        candidates, cursor = _take_links(
            key, cursor, where="Episode.candidate_refs")
        selected_key, cursor = _take(
            key, cursor, where="Episode.selected_path_ref", allow_empty=True)
        output_key, cursor = _take(
            key, cursor, where="Episode.output_ref", allow_empty=True)
        used, cursor = _take_memory_refs(
            key, cursor, where="Episode.used_memory_refs")
        signals, cursor = _take_links(
            key, cursor, where="Episode.result_signal_refs")
        failure_key, cursor = _take(
            key, cursor, where="Episode.failure_ref", allow_empty=True)
        if cursor >= len(key):
            raise ValueError("Episode 缺少 round_seq")
        round_seq = key[cursor]
        cursor += 1
        session_key, cursor = _take(key, cursor, where="Episode.session_ref")
        timestamp_key, cursor = _take(key, cursor, where="Episode.created_at")
        _finish(key, cursor, where="Episode.stable_key")
        optional = lambda value: (
            None if not value else MemoryLinkedRef.from_stable_key(value))
        return cls(
            MemoryObjectRef.from_stable_key(observation_key),
            optional(intent_key),
            candidates,
            optional(selected_key),
            optional(output_key),
            used,
            signals,
            optional(failure_key),
            round_seq,
            ScopeIdentity.from_stable_key(session_key),
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 Episode 创建逻辑时间。"""
        return self.created_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """汇总 Episode 各字段中直接引用的 Core identity。"""
        links = tuple(
            ref for ref in (
                self.intent_ref, self.selected_path_ref,
                self.output_ref, self.failure_ref)
            if ref is not None
        ) + self.candidate_refs + self.result_signal_refs
        core, _ = _payload_refs(links)
        return core

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回 Episode 引用的 Observation、候选和声明使用对象。"""
        links = tuple(
            ref for ref in (
                self.intent_ref, self.selected_path_ref,
                self.output_ref, self.failure_ref)
            if ref is not None
        ) + self.candidate_refs + self.result_signal_refs
        _, memory = _payload_refs(links)
        return (self.input_observation_ref, *self.used_memory_refs, *memory)


@dataclass(frozen=True)
class HypothesisPayload:
    """H-00 候选声明及其初始 retention/lifecycle，不包含派生计数。"""

    hypothesis: HypothesisKey
    initial_retention: int
    initial_lifecycle: int
    created_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """强制新候选从 episodic/active 开始，后续状态只走事件。"""
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("Hypothesis payload 必须包含 HypothesisKey")
        if self.initial_retention != RETENTION_EPISODIC:
            raise ValueError("新 Hypothesis 必须从 EPISODIC 开始")
        if self.initial_lifecycle != LIFECYCLE_ACTIVE:
            raise ValueError("新 Hypothesis 必须从 ACTIVE 开始")
        if not isinstance(self.created_at, LogicalTimestamp):
            raise TypeError("Hypothesis.created_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回候选声明、初始双轴和创建时钟的稳定键。"""
        return (
            _PAYLOAD_VERSION,
            *_pack(self.hypothesis.stable_key()),
            self.initial_retention,
            self.initial_lifecycle,
            *_pack(self.created_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "HypothesisPayload":
        """从完整声明键恢复 Hypothesis payload。"""
        key = _strict_tuple(key, where="HypothesisPayload.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Hypothesis payload 版本未注册")
        hypothesis_key, cursor = _take(key, 1, where="Hypothesis.key")
        if cursor + 2 > len(key):
            raise ValueError("Hypothesis 初始状态被截断")
        retention, lifecycle = key[cursor:cursor + 2]
        timestamp_key, cursor = _take(
            key, cursor + 2, where="Hypothesis.created_at")
        _finish(key, cursor, where="HypothesisPayload.stable_key")
        return cls(
            HypothesisKey.from_stable_key(hypothesis_key),
            retention,
            lifecycle,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 Hypothesis 创建逻辑时间。"""
        return self.created_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """H-00 开放键不冒充 Core typed identity。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """候选声明本身不隐式引用其他 Memory 对象。"""
        return ()


@dataclass(frozen=True)
class EvidencePayload:
    """某来源或 Episode 以类型化信号支持、反对或保留 Hypothesis。"""

    hypothesis_ref: MemoryObjectRef
    stance: int
    signal_ref: MemoryLinkedRef | None
    legacy_reason_key: tuple[int, ...]
    source: SourceRef | None
    episode_ref: MemoryObjectRef | None
    detail: tuple[int, ...]
    supersedes_ref: MemoryObjectRef | None
    observed_at: LogicalTimestamp
    compatibility_record_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验证据目标、来源、信号、替代链和兼容记录的互斥契约。"""
        if (not isinstance(self.hypothesis_ref, MemoryObjectRef)
                or self.hypothesis_ref.object_kind
                != MEMORY_OBJECT_HYPOTHESIS):
            raise ValueError("Evidence.hypothesis_ref 必须指向 Hypothesis")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("Evidence.stance 未注册")
        if (self.signal_ref is None) == (not self.legacy_reason_key):
            raise ValueError("Evidence 必须且只能提供 typed signal 或 legacy reason")
        if self.signal_ref is not None and not isinstance(
                self.signal_ref, MemoryLinkedRef):
            raise TypeError("Evidence.signal_ref 必须是一等引用或 None")
        _strict_tuple(
            self.legacy_reason_key, where="Evidence.legacy_reason_key",
            allow_empty=True)
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError("Evidence.source 必须是 SourceRef 或 None")
        if self.episode_ref is not None and (
                not isinstance(self.episode_ref, MemoryObjectRef)
                or self.episode_ref.object_kind != MEMORY_OBJECT_EPISODE):
            raise ValueError("Evidence.episode_ref 必须指向 Episode")
        if self.source is None and self.episode_ref is None:
            raise ValueError("Evidence 必须保留 source 或 episode 来源")
        _strict_tuple(self.detail, where="Evidence.detail", allow_empty=True)
        if self.supersedes_ref is not None and (
                not isinstance(self.supersedes_ref, MemoryObjectRef)
                or self.supersedes_ref.object_kind != MEMORY_OBJECT_EVIDENCE):
            raise ValueError("Evidence.supersedes_ref 必须指向 Evidence")
        if not isinstance(self.observed_at, LogicalTimestamp):
            raise TypeError("Evidence.observed_at 必须是 LogicalTimestamp")
        _strict_tuple(
            self.compatibility_record_key,
            where="Evidence.compatibility_record_key",
            allow_empty=True,
        )
        if self.compatibility_record_key and not self.legacy_reason_key:
            raise ValueError("H-00 兼容记录只能伴随显式 legacy reason")

    def stable_key(self) -> tuple[int, ...]:
        """返回证据来源、信号、方向、替代链和观察时钟的完整键。"""
        optional = lambda value: () if value is None else value.stable_key()
        return (
            _PAYLOAD_VERSION,
            *_pack(self.hypothesis_ref.stable_key()),
            self.stance,
            *_pack(optional(self.signal_ref)),
            *_pack(self.legacy_reason_key),
            *_pack(optional(self.source)),
            *_pack(optional(self.episode_ref)),
            *_pack(self.detail),
            *_pack(optional(self.supersedes_ref)),
            *_pack(self.observed_at.stable_key()),
            *_pack(self.compatibility_record_key),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "EvidencePayload":
        """从完整键恢复 Evidence 及其显式兼容边界。"""
        key = _strict_tuple(key, where="Evidence.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Evidence payload 版本未注册")
        hypothesis_key, cursor = _take(key, 1, where="Evidence.hypothesis")
        if cursor >= len(key):
            raise ValueError("Evidence 缺少 stance")
        stance = key[cursor]
        signal_key, cursor = _take(
            key, cursor + 1, where="Evidence.signal", allow_empty=True)
        legacy_key, cursor = _take(
            key, cursor, where="Evidence.legacy_reason", allow_empty=True)
        source_key, cursor = _take(
            key, cursor, where="Evidence.source", allow_empty=True)
        episode_key, cursor = _take(
            key, cursor, where="Evidence.episode", allow_empty=True)
        detail, cursor = _take(
            key, cursor, where="Evidence.detail", allow_empty=True)
        supersedes_key, cursor = _take(
            key, cursor, where="Evidence.supersedes", allow_empty=True)
        timestamp_key, cursor = _take(key, cursor, where="Evidence.observed_at")
        compatibility, cursor = _take(
            key, cursor, where="Evidence.compatibility", allow_empty=True)
        _finish(key, cursor, where="Evidence.stable_key")
        return cls(
            MemoryObjectRef.from_stable_key(hypothesis_key),
            stance,
            None if not signal_key else MemoryLinkedRef.from_stable_key(signal_key),
            legacy_key,
            None if not source_key else SourceRef.from_stable_key(source_key),
            None if not episode_key else MemoryObjectRef.from_stable_key(episode_key),
            detail,
            None if not supersedes_key else MemoryObjectRef.from_stable_key(supersedes_key),
            LogicalTimestamp.from_stable_key(timestamp_key),
            compatibility,
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回证据观察逻辑时间。"""
        return self.observed_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """返回 typed signal 直接引用的 Core 端点。"""
        return () if self.signal_ref is None else self.signal_ref.core_refs()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回 Hypothesis、Episode、被替代 Evidence 和 signal Memory 端点。"""
        result = [self.hypothesis_ref]
        if self.episode_ref is not None:
            result.append(self.episode_ref)
        if self.supersedes_ref is not None:
            result.append(self.supersedes_ref)
        if self.signal_ref is not None:
            result.extend(self.signal_ref.memory_refs())
        return tuple(result)


@dataclass(frozen=True)
class UsePayload:
    """某 Memory 对象实际影响一个 Episode 决策及其结果的独立事件。"""

    memory_ref: MemoryObjectRef
    episode_ref: MemoryObjectRef
    influence_kind: MemoryLinkedRef
    outcome_ref: MemoryLinkedRef | None
    used_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验 Use 必须显式指向既有 Memory 对象和 Episode。"""
        if not isinstance(self.memory_ref, MemoryObjectRef):
            raise TypeError("Use.memory_ref 必须是 MemoryObjectRef")
        if self.memory_ref.object_kind == MEMORY_OBJECT_USE:
            raise ValueError("Use 不得把另一个 Use 冒充被使用知识")
        if (not isinstance(self.episode_ref, MemoryObjectRef)
                or self.episode_ref.object_kind != MEMORY_OBJECT_EPISODE):
            raise ValueError("Use.episode_ref 必须指向 Episode")
        if not isinstance(self.influence_kind, MemoryLinkedRef):
            raise TypeError("Use.influence_kind 必须是一等引用")
        if self.outcome_ref is not None and not isinstance(
                self.outcome_ref, MemoryLinkedRef):
            raise TypeError("Use.outcome_ref 必须是一等引用或 None")
        if not isinstance(self.used_at, LogicalTimestamp):
            raise TypeError("Use.used_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回使用对象、Episode、影响类型、结果和 used 时钟的完整键。"""
        outcome = () if self.outcome_ref is None else self.outcome_ref.stable_key()
        return (
            _PAYLOAD_VERSION,
            *_pack(self.memory_ref.stable_key()),
            *_pack(self.episode_ref.stable_key()),
            *_pack(self.influence_kind.stable_key()),
            *_pack(outcome),
            *_pack(self.used_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "UsePayload":
        """从稳定键恢复一次显式 Use。"""
        key = _strict_tuple(key, where="Use.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Use payload 版本未注册")
        memory_key, cursor = _take(key, 1, where="Use.memory_ref")
        episode_key, cursor = _take(key, cursor, where="Use.episode_ref")
        influence_key, cursor = _take(key, cursor, where="Use.influence_kind")
        outcome_key, cursor = _take(
            key, cursor, where="Use.outcome_ref", allow_empty=True)
        timestamp_key, cursor = _take(key, cursor, where="Use.used_at")
        _finish(key, cursor, where="Use.stable_key")
        return cls(
            MemoryObjectRef.from_stable_key(memory_key),
            MemoryObjectRef.from_stable_key(episode_key),
            MemoryLinkedRef.from_stable_key(influence_key),
            None if not outcome_key else MemoryLinkedRef.from_stable_key(outcome_key),
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 Use 的 used 逻辑时间。"""
        return self.used_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """返回影响类型和结果直接引用的 Core 端点。"""
        result = list(self.influence_kind.core_refs())
        if self.outcome_ref is not None:
            result.extend(self.outcome_ref.core_refs())
        return tuple(result)

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回被使用对象、Episode 以及可选 Memory 类型/结果端点。"""
        result = [self.memory_ref, self.episode_ref]
        result.extend(self.influence_kind.memory_refs())
        if self.outcome_ref is not None:
            result.extend(self.outcome_ref.memory_refs())
        return tuple(result)


@dataclass(frozen=True)
class ArtifactPayload:
    """可长期留存的 typed Artifact 身份及其来源 Observation。"""

    artifact: ObjectIdentity
    observation_ref: MemoryObjectRef | None
    created_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验 Artifact 身份可自描述恢复，且来源引用类型正确。"""
        if (not isinstance(self.artifact, ObjectIdentity)
                or self.artifact.object_kind != OBJECT_ARTIFACT):
            raise ValueError("Artifact payload 必须包含 OBJECT_ARTIFACT 身份")
        describe_artifact_identity(self.artifact)
        if self.observation_ref is not None and (
                not isinstance(self.observation_ref, MemoryObjectRef)
                or self.observation_ref.object_kind
                != MEMORY_OBJECT_OBSERVATION):
            raise ValueError("Artifact.observation_ref 必须指向 Observation")
        if not isinstance(self.created_at, LogicalTimestamp):
            raise TypeError("Artifact.created_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回 Artifact、可选 Observation 和创建时钟。"""
        observation = (() if self.observation_ref is None
                       else self.observation_ref.stable_key())
        return (
            _PAYLOAD_VERSION,
            *_pack(self.artifact.stable_key()),
            *_pack(observation),
            *_pack(self.created_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ArtifactPayload":
        """从稳定键恢复 Artifact 声明。"""
        key = _strict_tuple(key, where="ArtifactPayload.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Artifact payload 版本未注册")
        artifact_key, cursor = _take(key, 1, where="Artifact.identity")
        observation_key, cursor = _take(
            key, cursor, where="Artifact.observation", allow_empty=True)
        timestamp_key, cursor = _take(key, cursor, where="Artifact.created_at")
        _finish(key, cursor, where="ArtifactPayload.stable_key")
        return cls(
            ObjectIdentity.from_stable_key(artifact_key),
            None if not observation_key else MemoryObjectRef.from_stable_key(observation_key),
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 Artifact 创建逻辑时间。"""
        return self.created_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """Artifact 内部权威对象身份不降级为 TypedRef。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回可选来源 Observation。"""
        return () if self.observation_ref is None else (self.observation_ref,)


@dataclass(frozen=True)
class CapabilityPayload:
    """可召回能力的类型、程序 Artifact、调用契约、证据和创建时间。"""

    capability_kind: MemoryLinkedRef
    program_ref: MemoryObjectRef
    contract_key: tuple[int, ...]
    evidence_refs: tuple[MemoryObjectRef, ...]
    created_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验能力只引用已声明程序 Artifact 和 Evidence。"""
        if not isinstance(self.capability_kind, MemoryLinkedRef):
            raise TypeError("Capability.kind 必须是一等引用")
        if (not isinstance(self.program_ref, MemoryObjectRef)
                or self.program_ref.object_kind != MEMORY_OBJECT_ARTIFACT):
            raise ValueError("Capability.program_ref 必须指向 Artifact")
        _strict_tuple(self.contract_key, where="Capability.contract_key")
        if not isinstance(self.evidence_refs, tuple) or any(
                not isinstance(ref, MemoryObjectRef)
                or ref.object_kind != MEMORY_OBJECT_EVIDENCE
                for ref in self.evidence_refs):
            raise ValueError("Capability.evidence_refs 必须只指向 Evidence")
        if not isinstance(self.created_at, LogicalTimestamp):
            raise TypeError("Capability.created_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回能力类型、程序、完整契约、证据和创建时钟。"""
        return (
            _PAYLOAD_VERSION,
            *_pack(self.capability_kind.stable_key()),
            *_pack(self.program_ref.stable_key()),
            *_pack(self.contract_key),
            *_memory_keys(self.evidence_refs),
            *_pack(self.created_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "CapabilityPayload":
        """从稳定键恢复 Capability 声明。"""
        key = _strict_tuple(key, where="Capability.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Capability payload 版本未注册")
        kind_key, cursor = _take(key, 1, where="Capability.kind")
        program_key, cursor = _take(key, cursor, where="Capability.program")
        contract_key, cursor = _take(key, cursor, where="Capability.contract")
        evidence, cursor = _take_memory_refs(
            key, cursor, where="Capability.evidence")
        timestamp_key, cursor = _take(key, cursor, where="Capability.created_at")
        _finish(key, cursor, where="Capability.stable_key")
        return cls(
            MemoryLinkedRef.from_stable_key(kind_key),
            MemoryObjectRef.from_stable_key(program_key),
            contract_key,
            evidence,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 Capability 创建逻辑时间。"""
        return self.created_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """返回能力类型直接引用的 Core 端点。"""
        return self.capability_kind.core_refs()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回程序 Artifact、证据和可选 Memory 类型端点。"""
        return (
            self.program_ref,
            *self.evidence_refs,
            *self.capability_kind.memory_refs(),
        )


@dataclass(frozen=True)
class ResolutionPayload:
    """H-04 完整决策及其候选引用的 append-only 声明。"""

    decision_key: tuple[int, ...]
    hypothesis_refs: tuple[MemoryObjectRef, ...]
    resolved_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验 decision codec、候选顺序和 Memory 引用逐项一致。"""
        decision = ResolverDecision.from_stable_key(self.decision_key)
        if (not isinstance(self.hypothesis_refs, tuple)
                or not self.hypothesis_refs
                or any(not isinstance(ref, MemoryObjectRef)
                       or ref.object_kind != MEMORY_OBJECT_HYPOTHESIS
                       for ref in self.hypothesis_refs)):
            raise ValueError("Resolution.hypothesis_refs 必须指向 Hypothesis")
        expected = tuple(
            item.hypothesis.stable_key() for item in decision.candidates)
        actual = tuple(ref.object_key for ref in self.hypothesis_refs)
        if actual != expected:
            raise ValueError("Resolution 候选引用与 decision trace 顺序不一致")
        spaces = {ref.memory_space for ref in self.hypothesis_refs}
        owners = {ref.owner for ref in self.hypothesis_refs}
        versions = {ref.versions for ref in self.hypothesis_refs}
        if len(spaces) != 1 or len(owners) != 1 or len(versions) != 1:
            raise ValueError("Resolution 候选必须属于同一 Memory owner/version")
        if not isinstance(self.resolved_at, LogicalTimestamp):
            raise TypeError("Resolution.resolved_at 必须是 LogicalTimestamp")
        anchor = decision.candidates[0].hypothesis
        if (self.resolved_at.clock.scope != anchor.scope
                or self.resolved_at.clock.clock_kind != CLOCK_MEMORY_RESOLVED
                or self.resolved_at.seq != decision.timestamp_seq + 1):
            raise ValueError("Resolution resolved_at 与 H-04 逻辑序不一致")

    def identity_key(self) -> tuple[int, ...]:
        """返回不含 M-03 适配时钟的原始 H-04 决策完整键。"""
        return self.decision_key

    def stable_key(self) -> tuple[int, ...]:
        """返回决策、候选引用和 resolved 时钟的完整载荷键。"""
        return (
            _PAYLOAD_VERSION,
            *_pack(self.decision_key),
            *_memory_keys(self.hypothesis_refs),
            *_pack(self.resolved_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ResolutionPayload":
        """从稳定键恢复 H-04 决策声明。"""
        key = _strict_tuple(key, where="Resolution.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Resolution payload 版本未注册")
        decision_key, cursor = _take(
            key, 1, where="Resolution.decision")
        hypotheses, cursor = _take_memory_refs(
            key, cursor, where="Resolution.hypotheses")
        timestamp_key, cursor = _take(
            key, cursor, where="Resolution.resolved_at")
        _finish(key, cursor, where="Resolution.stable_key")
        return cls(
            decision_key,
            hypotheses,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 H-04 决策的适配逻辑时间。"""
        return self.resolved_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """H-04 决策不隐式声明 Core 对象。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回 decision trace 中按稳定顺序排列的 Hypothesis 引用。"""
        return self.hypothesis_refs


@dataclass(frozen=True)
class RetentionTransitionPayload:
    """对象从 episodic 到 consolidated 的单向保留状态事件。"""

    target_ref: MemoryObjectRef
    from_state: int
    to_state: int
    reason_evidence_refs: tuple[MemoryObjectRef, ...]
    changed_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """只允许有 Evidence 理由的 EPISODIC 到 CONSOLIDATED 转换。"""
        if not isinstance(self.target_ref, MemoryObjectRef):
            raise TypeError("Retention.target_ref 必须是 MemoryObjectRef")
        if (self.from_state, self.to_state) != (
                RETENTION_EPISODIC, RETENTION_CONSOLIDATED):
            raise ValueError("retention 只允许 EPISODIC→CONSOLIDATED")
        if not self.reason_evidence_refs or any(
                not isinstance(ref, MemoryObjectRef)
                or ref.object_kind != MEMORY_OBJECT_EVIDENCE
                for ref in self.reason_evidence_refs):
            raise ValueError("retention 转换必须引用 Evidence")
        if not isinstance(self.changed_at, LogicalTimestamp):
            raise TypeError("Retention.changed_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回目标、单向状态、理由和转换时钟。"""
        return (
            _PAYLOAD_VERSION,
            *_pack(self.target_ref.stable_key()),
            self.from_state,
            self.to_state,
            *_memory_keys(self.reason_evidence_refs),
            *_pack(self.changed_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "RetentionTransitionPayload":
        """从稳定键恢复 retention 转换事件。"""
        key = _strict_tuple(key, where="Retention.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Retention payload 版本未注册")
        target_key, cursor = _take(key, 1, where="Retention.target")
        if cursor + 2 > len(key):
            raise ValueError("Retention 状态被截断")
        from_state, to_state = key[cursor:cursor + 2]
        reasons, cursor = _take_memory_refs(
            key, cursor + 2, where="Retention.reasons")
        timestamp_key, cursor = _take(key, cursor, where="Retention.changed_at")
        _finish(key, cursor, where="Retention.stable_key")
        return cls(
            MemoryObjectRef.from_stable_key(target_key),
            from_state,
            to_state,
            reasons,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 retention 转换逻辑时间。"""
        return self.changed_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """retention 转换不隐式引用 Core。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回转换目标和理由 Evidence。"""
        return self.target_ref, *self.reason_evidence_refs


@dataclass(frozen=True)
class LifecycleTransitionPayload:
    """对象 active/superseded/archived 的不可变单向转换。"""

    target_ref: MemoryObjectRef
    from_state: int
    to_state: int
    reason_evidence_refs: tuple[MemoryObjectRef, ...]
    replacement_ref: MemoryObjectRef | None
    changed_at: LogicalTimestamp
    compatibility_transition_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验生命周期只前进，superseded 必须有同类 replacement。"""
        if not isinstance(self.target_ref, MemoryObjectRef):
            raise TypeError("Lifecycle.target_ref 必须是 MemoryObjectRef")
        allowed = {
            (LIFECYCLE_ACTIVE, LIFECYCLE_SUPERSEDED),
            (LIFECYCLE_ACTIVE, LIFECYCLE_ARCHIVED),
            (LIFECYCLE_SUPERSEDED, LIFECYCLE_ARCHIVED),
        }
        if (self.from_state, self.to_state) not in allowed:
            raise ValueError("lifecycle 转换方向非法")
        if not self.reason_evidence_refs or any(
                not isinstance(ref, MemoryObjectRef)
                or ref.object_kind != MEMORY_OBJECT_EVIDENCE
                for ref in self.reason_evidence_refs):
            raise ValueError("lifecycle 转换必须引用 Evidence")
        if self.to_state == LIFECYCLE_SUPERSEDED:
            if (not isinstance(self.replacement_ref, MemoryObjectRef)
                    or self.replacement_ref.object_kind
                    != self.target_ref.object_kind
                    or self.replacement_ref == self.target_ref):
                raise ValueError("superseded 必须指定不同的同类 replacement")
        elif self.replacement_ref is not None:
            raise ValueError("archived 转换不得携带 replacement")
        if not isinstance(self.changed_at, LogicalTimestamp):
            raise TypeError("Lifecycle.changed_at 必须是 LogicalTimestamp")
        _strict_tuple(
            self.compatibility_transition_key,
            where="Lifecycle.compatibility_transition_key",
            allow_empty=True,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回目标、状态方向、理由、replacement 和转换时钟。"""
        replacement = (() if self.replacement_ref is None
                       else self.replacement_ref.stable_key())
        return (
            _PAYLOAD_VERSION,
            *_pack(self.target_ref.stable_key()),
            self.from_state,
            self.to_state,
            *_memory_keys(self.reason_evidence_refs),
            *_pack(replacement),
            *_pack(self.changed_at.stable_key()),
            *_pack(self.compatibility_transition_key),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "LifecycleTransitionPayload":
        """从稳定键恢复 lifecycle 转换事件。"""
        key = _strict_tuple(key, where="Lifecycle.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Lifecycle payload 版本未注册")
        target_key, cursor = _take(key, 1, where="Lifecycle.target")
        if cursor + 2 > len(key):
            raise ValueError("Lifecycle 状态被截断")
        from_state, to_state = key[cursor:cursor + 2]
        reasons, cursor = _take_memory_refs(
            key, cursor + 2, where="Lifecycle.reasons")
        replacement_key, cursor = _take(
            key, cursor, where="Lifecycle.replacement", allow_empty=True)
        timestamp_key, cursor = _take(key, cursor, where="Lifecycle.changed_at")
        compatibility, cursor = _take(
            key, cursor, where="Lifecycle.compatibility", allow_empty=True)
        _finish(key, cursor, where="Lifecycle.stable_key")
        return cls(
            MemoryObjectRef.from_stable_key(target_key),
            from_state,
            to_state,
            reasons,
            None if not replacement_key else MemoryObjectRef.from_stable_key(replacement_key),
            LogicalTimestamp.from_stable_key(timestamp_key),
            compatibility,
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 lifecycle 转换逻辑时间。"""
        return self.changed_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """lifecycle 转换不隐式引用 Core。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回转换目标、理由和可选 replacement。"""
        replacement = () if self.replacement_ref is None else (self.replacement_ref,)
        return self.target_ref, *self.reason_evidence_refs, *replacement


@dataclass(frozen=True)
class DerivationTransitionPayload:
    """parser 重新解析导致的派生对象替代或归档事件。"""

    target_ref: MemoryObjectRef
    from_state: int
    to_state: int
    replacement_ref: MemoryObjectRef | None
    prior_source: SourceRef
    replacement_source: SourceRef
    binding_kind: int
    lineage_key: tuple[int, ...]
    changed_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验同来源谱系只由更高 parser version 推进，且不借用 Evidence。"""
        if not isinstance(self.target_ref, MemoryObjectRef):
            raise TypeError("Derivation.target_ref 必须是 MemoryObjectRef")
        if (self.from_state != LIFECYCLE_ACTIVE
                or self.to_state not in {
                    LIFECYCLE_SUPERSEDED, LIFECYCLE_ARCHIVED}):
            raise ValueError("Derivation 只允许 active 到 superseded/archived")
        expected_kind = _INTAKE_DERIVED_OBJECT_KINDS.get(self.binding_kind)
        if expected_kind is None:
            raise ValueError("Derivation.binding_kind 未注册")
        if self.target_ref.object_kind != expected_kind:
            raise ValueError("Derivation target 与 binding kind 不一致")
        if (not isinstance(self.prior_source, SourceRef)
                or not isinstance(self.replacement_source, SourceRef)):
            raise TypeError("Derivation 必须携带前后 SourceRef")
        if (source_reparse_lineage_key(self.prior_source)
                != source_reparse_lineage_key(self.replacement_source)
                or self.replacement_source.versions.parser.value
                <= self.prior_source.versions.parser.value):
            raise ValueError("Derivation 必须指向同谱系更高 parser version")
        if (self.target_ref.owner != self.prior_source.owner
                or self.target_ref.versions != self.prior_source.versions):
            raise ValueError("Derivation target 未绑定旧来源 owner/version")
        _strict_tuple(self.lineage_key, where="Derivation.lineage_key")
        if self.to_state == LIFECYCLE_SUPERSEDED:
            if (not isinstance(self.replacement_ref, MemoryObjectRef)
                    or self.replacement_ref.object_kind != expected_kind
                    or self.replacement_ref.owner
                    != self.replacement_source.owner
                    or self.replacement_ref.versions
                    != self.replacement_source.versions
                    or self.replacement_ref == self.target_ref):
                raise ValueError("Derivation superseded 缺少同类新版本 replacement")
        elif self.replacement_ref is not None:
            raise ValueError("Derivation archived 不得携带 replacement")
        if not isinstance(self.changed_at, LogicalTimestamp):
            raise TypeError("Derivation.changed_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回目标、前后来源、谱系和逻辑转换时间。"""
        replacement = (() if self.replacement_ref is None
                       else self.replacement_ref.stable_key())
        return (
            _PAYLOAD_VERSION,
            *_pack(self.target_ref.stable_key()),
            self.from_state,
            self.to_state,
            *_pack(replacement),
            *_pack(self.prior_source.stable_key()),
            *_pack(self.replacement_source.stable_key()),
            self.binding_kind,
            *_pack(self.lineage_key),
            *_pack(self.changed_at.stable_key()),
        )

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...]) -> "DerivationTransitionPayload":
        """从稳定键恢复 parser 派生版本转换。"""
        key = _strict_tuple(key, where="Derivation.stable_key")
        if key[0] != _PAYLOAD_VERSION:
            raise ValueError("Derivation payload 版本未注册")
        target_key, cursor = _take(key, 1, where="Derivation.target")
        if cursor + 2 > len(key):
            raise ValueError("Derivation 状态被截断")
        from_state, to_state = key[cursor:cursor + 2]
        replacement_key, cursor = _take(
            key, cursor + 2, where="Derivation.replacement", allow_empty=True)
        prior_key, cursor = _take(
            key, cursor, where="Derivation.prior_source")
        source_key, cursor = _take(
            key, cursor, where="Derivation.replacement_source")
        if cursor >= len(key):
            raise ValueError("Derivation 缺少 binding_kind")
        binding_kind = key[cursor]
        lineage_key, cursor = _take(
            key, cursor + 1, where="Derivation.lineage_key")
        timestamp_key, cursor = _take(
            key, cursor, where="Derivation.changed_at")
        _finish(key, cursor, where="Derivation.stable_key")
        return cls(
            MemoryObjectRef.from_stable_key(target_key),
            from_state,
            to_state,
            None if not replacement_key else MemoryObjectRef.from_stable_key(
                replacement_key),
            SourceRef.from_stable_key(prior_key),
            SourceRef.from_stable_key(source_key),
            binding_kind,
            lineage_key,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回 derivation lifecycle 的逻辑时间。"""
        return self.changed_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """parser 版本替代不隐式声明 Core 语义。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """返回被替代目标和可选同类 replacement。"""
        replacement = () if self.replacement_ref is None else (self.replacement_ref,)
        return self.target_ref, *replacement


@dataclass(frozen=True)
class LegacyImportPayload:
    """旧表原字段的显式导入事件，不赋予 Hypothesis/Evidence/Use 语义。"""

    table_kind: int
    row_key: tuple[int, ...]
    field_values: tuple[int, ...]
    imported_at: LogicalTimestamp

    def __post_init__(self) -> None:
        """只接受已登记旧表种类和无损整数行载荷。"""
        if self.table_kind not in {
                LEGACY_TABLE_MEMORY_ITEM, LEGACY_TABLE_EXPERIENCE_COUNT}:
            raise ValueError("LegacyImport.table_kind 未注册")
        _strict_tuple(self.row_key, where="LegacyImport.row_key")
        _strict_tuple(self.field_values, where="LegacyImport.field_values")
        if not isinstance(self.imported_at, LogicalTimestamp):
            raise TypeError("LegacyImport.imported_at 必须是 LogicalTimestamp")

    def stable_key(self) -> tuple[int, ...]:
        """返回旧表种类、原主键、原字段和值及导入时钟。"""
        return (
            _PAYLOAD_VERSION,
            self.table_kind,
            *_pack(self.row_key),
            *_pack(self.field_values),
            *_pack(self.imported_at.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "LegacyImportPayload":
        """从稳定键恢复明确标记的 legacy import。"""
        key = _strict_tuple(key, where="LegacyImport.stable_key")
        if len(key) < 5 or key[0] != _PAYLOAD_VERSION:
            raise ValueError("LegacyImport payload 版本或长度非法")
        row_key, cursor = _take(key, 2, where="LegacyImport.row_key")
        fields, cursor = _take(key, cursor, where="LegacyImport.field_values")
        timestamp_key, cursor = _take(key, cursor, where="LegacyImport.imported_at")
        _finish(key, cursor, where="LegacyImport.stable_key")
        return cls(
            key[1], row_key, fields,
            LogicalTimestamp.from_stable_key(timestamp_key),
        )

    def timestamp(self) -> LogicalTimestamp:
        """返回旧行显式导入的逻辑时间。"""
        return self.imported_at

    def core_refs(self) -> tuple[TypedRef, ...]:
        """旧裸整数列不得自动升级为 Core identity。"""
        return ()

    def memory_refs(self) -> tuple[MemoryObjectRef, ...]:
        """旧裸整数列不得自动升级为 Memory identity。"""
        return ()


_PAYLOAD_TYPES = {
    MEMORY_EVENT_OBSERVATION: ObservationPayload,
    MEMORY_EVENT_EPISODE: EpisodePayload,
    MEMORY_EVENT_HYPOTHESIS: HypothesisPayload,
    MEMORY_EVENT_EVIDENCE: EvidencePayload,
    MEMORY_EVENT_USE: UsePayload,
    MEMORY_EVENT_ARTIFACT: ArtifactPayload,
    MEMORY_EVENT_CAPABILITY: CapabilityPayload,
    MEMORY_EVENT_RETENTION: RetentionTransitionPayload,
    MEMORY_EVENT_LIFECYCLE: LifecycleTransitionPayload,
    MEMORY_EVENT_LEGACY_IMPORT: LegacyImportPayload,
    MEMORY_EVENT_PARSE_FAILURE: ParseFailurePayload,
    MEMORY_EVENT_INTAKE_MANIFEST: IntakeManifestPayload,
    MEMORY_EVENT_DERIVATION: DerivationTransitionPayload,
    MEMORY_EVENT_RESOLUTION: ResolutionPayload,
}

_DECLARATION_OBJECT_KINDS = {
    MEMORY_EVENT_OBSERVATION: MEMORY_OBJECT_OBSERVATION,
    MEMORY_EVENT_EPISODE: MEMORY_OBJECT_EPISODE,
    MEMORY_EVENT_HYPOTHESIS: MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_EVENT_EVIDENCE: MEMORY_OBJECT_EVIDENCE,
    MEMORY_EVENT_USE: MEMORY_OBJECT_USE,
    MEMORY_EVENT_ARTIFACT: MEMORY_OBJECT_ARTIFACT,
    MEMORY_EVENT_CAPABILITY: MEMORY_OBJECT_CAPABILITY,
    MEMORY_EVENT_LEGACY_IMPORT: MEMORY_OBJECT_LEGACY_IMPORT,
    MEMORY_EVENT_PARSE_FAILURE: MEMORY_OBJECT_PARSE_FAILURE,
    MEMORY_EVENT_INTAKE_MANIFEST: MEMORY_OBJECT_INTAKE_MANIFEST,
    MEMORY_EVENT_RESOLUTION: MEMORY_OBJECT_RESOLUTION,
}


@dataclass(frozen=True)
class MemoryEvent:
    """带对象引用、scope、完整逻辑时钟和类型化载荷的不可变事件。"""

    event_kind: int
    object_ref: MemoryObjectRef
    scope: ScopeIdentity
    payload: MemoryPayload

    def __post_init__(self) -> None:
        """核验事件种类、对象身份、owner/version 和 payload 主时间一致。"""
        payload_type = _PAYLOAD_TYPES.get(self.event_kind)
        if payload_type is None:
            raise ValueError("MemoryEvent.event_kind 未注册")
        if not isinstance(self.object_ref, MemoryObjectRef):
            raise TypeError("MemoryEvent.object_ref 必须是 MemoryObjectRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("MemoryEvent.scope 必须是 ScopeIdentity")
        if not isinstance(self.payload, payload_type):
            raise TypeError("MemoryEvent.payload 与 event_kind 不匹配")
        if (self.scope.owner != self.object_ref.owner
                or self.scope.versions != self.object_ref.versions):
            raise ValueError("MemoryEvent scope 与对象 owner/version 不一致")
        timestamp = self.payload.timestamp()
        if (timestamp.clock.scope.owner != self.object_ref.owner
                or timestamp.clock.scope.versions != self.object_ref.versions):
            raise ValueError("MemoryEvent timestamp 与对象 owner/version 不一致")
        expected_kind = _DECLARATION_OBJECT_KINDS.get(self.event_kind)
        if expected_kind is not None:
            if self.object_ref.object_kind != expected_kind:
                raise ValueError("MemoryEvent 对象种类与声明事件不匹配")
            expected_key = self._declaration_object_key()
            if self.object_ref.object_key != expected_key:
                raise ValueError("MemoryEvent 对象键与声明 payload 不一致")
        elif self.object_ref != self.payload.target_ref:
            raise ValueError("状态转换事件必须以 target_ref 作为 object_ref")

    def _declaration_object_key(self) -> tuple[int, ...]:
        """按对象种类返回稳定对象键，创建时间不污染既有本体身份。"""
        if isinstance(self.payload, HypothesisPayload):
            return self.payload.hypothesis.stable_key()
        if isinstance(self.payload, ArtifactPayload):
            return self.payload.artifact.stable_key()
        if isinstance(self.payload, (ParseFailurePayload,
                                     IntakeManifestPayload,
                                     ResolutionPayload)):
            return self.payload.identity_key()
        return self.payload.stable_key()

    @property
    def timestamp(self) -> LogicalTimestamp:
        """返回事件的完整主逻辑时间戳。"""
        return self.payload.timestamp()

    @property
    def time_axis(self) -> int:
        """返回 created/observed/used 物理索引轴，不推断领域真值。"""
        if self.event_kind in {
                MEMORY_EVENT_OBSERVATION,
                MEMORY_EVENT_EVIDENCE,
                MEMORY_EVENT_PARSE_FAILURE,
        }:
            return TIME_AXIS_OBSERVED
        if self.event_kind == MEMORY_EVENT_USE:
            return TIME_AXIS_USED
        return TIME_AXIS_CREATED

    @property
    def is_declaration(self) -> bool:
        """返回该事件是否声明一个新的不可变 Memory 对象。"""
        return self.event_kind in _DECLARATION_OBJECT_KINDS

    def stable_key(self) -> tuple[int, ...]:
        """返回事件种类、对象、scope、timestamp 和 payload 的完整身份。"""
        return (
            _EVENT_VERSION,
            self.event_kind,
            *_pack(self.object_ref.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.timestamp.stable_key()),
            *_pack(self.payload.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "MemoryEvent":
        """从完整整数身份恢复类型化事件并复核主时间。"""
        key = _strict_tuple(key, where="MemoryEvent.stable_key")
        if len(key) < 6 or key[0] != _EVENT_VERSION:
            raise ValueError("MemoryEvent 稳定键版本或长度非法")
        object_key, cursor = _take(key, 2, where="MemoryEvent.object_ref")
        scope_key, cursor = _take(key, cursor, where="MemoryEvent.scope")
        timestamp_key, cursor = _take(key, cursor, where="MemoryEvent.timestamp")
        payload_key, cursor = _take(key, cursor, where="MemoryEvent.payload")
        _finish(key, cursor, where="MemoryEvent.stable_key")
        payload_type = _PAYLOAD_TYPES.get(key[1])
        if payload_type is None:
            raise ValueError("MemoryEvent.event_kind 未注册")
        payload = payload_type.from_stable_key(payload_key)
        event = cls(
            key[1],
            MemoryObjectRef.from_stable_key(object_key),
            ScopeIdentity.from_stable_key(scope_key),
            payload,
        )
        if event.timestamp != LogicalTimestamp.from_stable_key(timestamp_key):
            raise ValueError("MemoryEvent 外层 timestamp 与 payload 不一致")
        return event


def memory_object_ref(memory_space: SpaceIdentity, object_kind: int,
                      object_key: tuple[int, ...], *, owner: OwnerScope,
                      versions: VersionBundle) -> MemoryObjectRef:
    """用显式 Memory 空间和 owner/version 构造对象引用，不从端点推断 owner。"""
    return MemoryObjectRef(
        memory_space, owner, versions, object_kind, object_key)


def payload_from_stable_key(event_kind: int,
                            key: tuple[int, ...]) -> MemoryPayload:
    """按事件种类恢复类型化 payload，供正规化持久层重建完整事件。"""
    payload_type = _PAYLOAD_TYPES.get(event_kind)
    if payload_type is None:
        raise ValueError("Memory event payload kind 未注册")
    return payload_type.from_stable_key(key)


def declaration_object_key(event_kind: int,
                           payload: MemoryPayload) -> tuple[int, ...]:
    """从声明 payload 恢复对象键，状态事件不得调用该函数。"""
    expected_kind = _DECLARATION_OBJECT_KINDS.get(event_kind)
    if expected_kind is None:
        raise ValueError("状态事件没有新的声明对象键")
    if isinstance(payload, HypothesisPayload):
        return payload.hypothesis.stable_key()
    if isinstance(payload, ArtifactPayload):
        return payload.artifact.stable_key()
    if isinstance(payload, (ParseFailurePayload, IntakeManifestPayload,
                            ResolutionPayload)):
        return payload.identity_key()
    return payload.stable_key()


__all__ = [
    "ArtifactPayload",
    "CapabilityPayload",
    "DerivationTransitionPayload",
    "EpisodePayload",
    "EvidencePayload",
    "INTAKE_DERIVED_EVIDENCE",
    "INTAKE_DERIVED_FAILURE",
    "INTAKE_DERIVED_HYPOTHESIS",
    "INTAKE_DERIVED_MANIFEST",
    "INTAKE_DERIVED_OBSERVATION",
    "INTAKE_OUTCOME_FAILURE",
    "INTAKE_OUTCOME_SUCCESS",
    "IntakeDerivedBinding",
    "IntakeManifestPayload",
    "LEGACY_TABLE_EXPERIENCE_COUNT",
    "LEGACY_TABLE_MEMORY_ITEM",
    "LifecycleTransitionPayload",
    "LegacyImportPayload",
    "MEMORY_EVENT_ARTIFACT",
    "MEMORY_EVENT_CAPABILITY",
    "MEMORY_EVENT_DERIVATION",
    "MEMORY_EVENT_EPISODE",
    "MEMORY_EVENT_EVIDENCE",
    "MEMORY_EVENT_HYPOTHESIS",
    "MEMORY_EVENT_INTAKE_MANIFEST",
    "MEMORY_EVENT_LEGACY_IMPORT",
    "MEMORY_EVENT_LIFECYCLE",
    "MEMORY_EVENT_OBSERVATION",
    "MEMORY_EVENT_PARSE_FAILURE",
    "MEMORY_EVENT_RETENTION",
    "MEMORY_EVENT_RESOLUTION",
    "MEMORY_EVENT_USE",
    "MEMORY_OBJECT_ARTIFACT",
    "MEMORY_OBJECT_CAPABILITY",
    "MEMORY_OBJECT_EPISODE",
    "MEMORY_OBJECT_EVIDENCE",
    "MEMORY_OBJECT_HYPOTHESIS",
    "MEMORY_OBJECT_INTAKE_MANIFEST",
    "MEMORY_OBJECT_LEGACY_IMPORT",
    "MEMORY_OBJECT_OBSERVATION",
    "MEMORY_OBJECT_PARSE_FAILURE",
    "MEMORY_OBJECT_RESOLUTION",
    "MEMORY_OBJECT_USE",
    "MemoryEvent",
    "MemoryLinkedRef",
    "MemoryObjectRef",
    "ObservationPayload",
    "ParseFailurePayload",
    "ResolutionPayload",
    "RETENTION_CONSOLIDATED",
    "RETENTION_EPISODIC",
    "RetentionTransitionPayload",
    "TIME_AXIS_CREATED",
    "TIME_AXIS_OBSERVED",
    "TIME_AXIS_USED",
    "UsePayload",
    "memory_object_ref",
    "declaration_object_key",
    "payload_from_stable_key",
    "source_reparse_lineage_key",
]
