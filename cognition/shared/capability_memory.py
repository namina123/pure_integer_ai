"""C-02 verified Capability 的可逆契约及 Memory 恢复边界。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.capability_candidate import (
    CapabilityStatusProtocol,
)
from pure_integer_ai.cognition.shared.capability_verification import (
    CapabilityVerificationReport,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactAuthority,
    FormalArtifactDefinition,
    formal_artifact_definition_from_stable_key,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_ARTIFACT,
    MEMORY_EVENT_CAPABILITY,
    MEMORY_OBJECT_ARTIFACT,
    MEMORY_OBJECT_CAPABILITY,
    ArtifactPayload,
    CapabilityPayload,
    MemoryLinkedRef,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_CONTRACT_VERSION = 1
_PROVENANCE_DOMAIN = "capability.memory.verified.provenance.v1"


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长整数键增加长度边界。"""
    return len(value), *value


def _strict_key(
        value: tuple[int, ...], *, label: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """核验 contract 字段为严格整数 tuple。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{label} 必须是{'可空' if allow_empty else '非空'} tuple")
    if value:
        assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须只含严格整数")
    return value


def _take(
        values: tuple[int, ...], cursor: int, *, label: str,
        allow_empty: bool = False,
        ) -> tuple[tuple[int, ...], int]:
    """读取长度前缀字段，并拒绝负长度、截断和非法空段。"""
    if cursor >= len(values):
        raise ValueError(f"{label} 缺少长度")
    size = values[cursor]
    cursor += 1
    if type(size) is not int or size < 0 or (size == 0 and not allow_empty):
        raise ValueError(f"{label} 长度非法")
    end = cursor + size
    if end > len(values):
        raise ValueError(f"{label} 被截断")
    return values[cursor:end], end


def _status_protocol_from_key(
        key: tuple[int, ...],
        ) -> CapabilityStatusProtocol:
    """从三个有界 ObjectIdentity 段恢复 Capability 状态协议。"""
    values = _strict_key(key, label="Capability status protocol")
    states = []
    cursor = 0
    for label in ("provisional", "verified", "rejected"):
        state_key, cursor = _take(
            values, cursor, label=f"Capability status {label}")
        states.append(ObjectIdentity.from_stable_key(state_key))
    if cursor != len(values):
        raise ValueError("Capability status protocol 含尾随字段")
    return CapabilityStatusProtocol(*states)


def _report_fields(
        key: tuple[int, ...],
        ) -> tuple[tuple[int, ...], ...]:
    """拆解 C-01 六段报告引用，供 contract 核验状态和来源证明绑定。"""
    values = _strict_key(key, label="Capability verification report")
    result = []
    cursor = 0
    for label in (
            "candidate", "held_out", "invocation", "result",
            "previous_state", "state"):
        field, cursor = _take(
            values, cursor, label=f"Capability report {label}")
        result.append(field)
    if cursor != len(values):
        raise ValueError("Capability verification report 含尾随字段")
    if any(len(field) != 34 for field in result[:4]):
        raise ValueError("Capability verification report 内容引用长度非法")
    return tuple(result)


def _verified_provenance_body(
        capability_kind: MemoryLinkedRef,
        definition: FormalArtifactDefinition,
        preconditions: tuple[ObjectIdentity, ...],
        report_key: tuple[int, ...],
        status_protocol: CapabilityStatusProtocol,
        previous_state: ObjectIdentity,
        state: ObjectIdentity,
        candidate_source: SourceRef,
        candidate_example_sources: tuple[SourceRef, ...],
        held_out_source: SourceRef,
        verification_authority: ArtifactAuthority,
        specification_authority: ArtifactAuthority,
        ) -> tuple[int, ...]:
    """编码不含自引用 fingerprint 的完整 verified provenance。"""
    result = [
        _CONTRACT_VERSION,
        *_packed(capability_kind.stable_key()),
        *_packed(definition.stable_key()),
        len(preconditions),
    ]
    for precondition in preconditions:
        result.extend(_packed(precondition.stable_key()))
    result.extend((
        *_packed(report_key),
        *_packed(status_protocol.stable_key()),
        *_packed(previous_state.stable_key()),
        *_packed(state.stable_key()),
        *_packed(candidate_source.stable_key()),
        len(candidate_example_sources),
    ))
    for source in candidate_example_sources:
        result.extend(_packed(source.stable_key()))
    result.extend((
        *_packed(held_out_source.stable_key()),
        *_packed(verification_authority.stable_key()),
        *_packed(specification_authority.stable_key()),
    ))
    return tuple(result)


@dataclass(frozen=True)
class VerifiedCapabilityContract:
    """可从 CapabilityPayload 恢复并独立核验的 verified 调用契约。"""

    capability_kind: MemoryLinkedRef
    definition: FormalArtifactDefinition
    preconditions: tuple[ObjectIdentity, ...]
    report_key: tuple[int, ...]
    status_protocol: CapabilityStatusProtocol
    previous_state: ObjectIdentity
    state: ObjectIdentity
    candidate_source: SourceRef
    candidate_example_sources: tuple[SourceRef, ...]
    held_out_source: SourceRef
    verification_authority: ArtifactAuthority
    specification_authority: ArtifactAuthority
    provenance_ref: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验定义、状态、双侧来源、authority 和 C-01 内容引用闭合。"""
        if not isinstance(self.capability_kind, MemoryLinkedRef):
            raise TypeError("Capability contract kind 必须是一等引用")
        if not isinstance(self.definition, FormalArtifactDefinition):
            raise TypeError("Capability contract definition 类型错误")
        if not isinstance(self.preconditions, tuple) or any(
                not isinstance(item, ObjectIdentity)
                for item in self.preconditions):
            raise TypeError("Capability contract preconditions 类型错误")
        if len(set(self.preconditions)) != len(self.preconditions):
            raise ValueError("Capability contract preconditions 不得重复")
        if not isinstance(self.status_protocol, CapabilityStatusProtocol):
            raise TypeError("Capability contract status protocol 类型错误")
        if self.previous_state != self.status_protocol.provisional:
            raise ValueError("Capability contract 前态必须是 provisional")
        if self.state != self.status_protocol.verified:
            raise ValueError("Capability contract 只接受 verified 状态")
        report_fields = _report_fields(self.report_key)
        if (ObjectIdentity.from_stable_key(report_fields[4])
                != self.previous_state
                or ObjectIdentity.from_stable_key(report_fields[5])
                != self.state):
            raise ValueError("Capability contract 状态与 C-01 report 漂移")
        if not isinstance(self.candidate_source, SourceRef):
            raise TypeError("Capability contract candidate source 类型错误")
        if self.definition.program.source != self.candidate_source:
            raise ValueError("Capability definition program 与 candidate 来源漂移")
        if (not isinstance(self.candidate_example_sources, tuple)
                or not self.candidate_example_sources
                or any(not isinstance(item, SourceRef)
                       for item in self.candidate_example_sources)):
            raise TypeError("Capability candidate example sources 类型错误")
        source_keys = tuple(
            item.stable_key() for item in self.candidate_example_sources)
        if source_keys != tuple(sorted(set(source_keys))):
            raise ValueError("Capability candidate example sources 必须唯一稳定排序")
        if not isinstance(self.held_out_source, SourceRef):
            raise TypeError("Capability contract held-out source 类型错误")
        if (self.held_out_source == self.candidate_source
                or self.held_out_source in self.candidate_example_sources):
            raise ValueError("Capability held-out source 未与 candidate 侧分离")
        if self.verification_authority != self.definition.verifier:
            raise ValueError("Capability verification authority 与 definition 漂移")
        if self.specification_authority in (
                self.definition.executor, self.definition.verifier):
            raise ValueError("Capability specification authority 不独立")
        _strict_key(
            self.provenance_ref, label="Capability verified provenance")
        expected = integer_tuple_fingerprint(
            self._provenance_body(), domain=_PROVENANCE_DOMAIN)
        if self.provenance_ref != expected:
            raise ValueError("Capability verified provenance 内容引用漂移")

    @classmethod
    def from_report(
            cls,
            report: CapabilityVerificationReport,
            ) -> "VerifiedCapabilityContract":
        """只把 C-01 明确 verified 报告编码为可长期恢复的契约。"""
        if not isinstance(report, CapabilityVerificationReport):
            raise TypeError("report 必须是 CapabilityVerificationReport")
        if not report.verified or not report.result.succeeded:
            raise ValueError("C-02 只接受 C-01 明确 verified report")
        verification = report.result.verification
        if verification is None or verification.accepted is not True:
            raise ValueError("verified report 缺少独立 verifier accepted")
        candidate = report.candidate
        proposal = candidate.proposal
        example_sources = tuple(sorted(
            {item.run.invocation.source for item in candidate.examples},
            key=lambda item: item.stable_key(),
        ))
        provenance_body = _verified_provenance_body(
            proposal.capability_kind,
            proposal.definition,
            proposal.preconditions,
            report.stable_key(),
            candidate.status_protocol,
            report.previous_state,
            report.state,
            proposal.source,
            example_sources,
            report.held_out.source,
            verification.authority,
            report.held_out.specification_authority,
        )
        return cls(
            proposal.capability_kind,
            proposal.definition,
            proposal.preconditions,
            report.stable_key(),
            candidate.status_protocol,
            report.previous_state,
            report.state,
            proposal.source,
            example_sources,
            report.held_out.source,
            verification.authority,
            report.held_out.specification_authority,
            integer_tuple_fingerprint(
                provenance_body, domain=_PROVENANCE_DOMAIN),
        )

    def _provenance_body(self) -> tuple[int, ...]:
        """编码除自引用 fingerprint 外的全部 verified provenance 字段。"""
        return _verified_provenance_body(
            self.capability_kind,
            self.definition,
            self.preconditions,
            self.report_key,
            self.status_protocol,
            self.previous_state,
            self.state,
            self.candidate_source,
            self.candidate_example_sources,
            self.held_out_source,
            self.verification_authority,
            self.specification_authority,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 provenance 字段及其域分离内容引用。"""
        return (*self._provenance_body(), *_packed(self.provenance_ref))

    @classmethod
    def from_stable_key(
            cls,
            key: tuple[int, ...],
            ) -> "VerifiedCapabilityContract":
        """从 Capability contract_key 对称恢复 verified 契约。"""
        values = _strict_key(key, label="VerifiedCapabilityContract.stable_key")
        if values[0] != _CONTRACT_VERSION:
            raise ValueError("Capability contract 版本未注册")
        capability_kind_key, cursor = _take(
            values, 1, label="Capability contract kind")
        definition_key, cursor = _take(
            values, cursor, label="Capability contract definition")
        if cursor >= len(values):
            raise ValueError("Capability contract 缺少 precondition 数量")
        precondition_count = values[cursor]
        cursor += 1
        if type(precondition_count) is not int or precondition_count < 0:
            raise ValueError("Capability contract precondition 数量非法")
        preconditions = []
        for _ in range(precondition_count):
            item, cursor = _take(
                values, cursor, label="Capability contract precondition")
            preconditions.append(ObjectIdentity.from_stable_key(item))
        report_key, cursor = _take(
            values, cursor, label="Capability contract report")
        protocol_key, cursor = _take(
            values, cursor, label="Capability contract status protocol")
        previous_key, cursor = _take(
            values, cursor, label="Capability contract previous state")
        state_key, cursor = _take(
            values, cursor, label="Capability contract state")
        candidate_source_key, cursor = _take(
            values, cursor, label="Capability contract candidate source")
        if cursor >= len(values):
            raise ValueError("Capability contract 缺少 example source 数量")
        source_count = values[cursor]
        cursor += 1
        if type(source_count) is not int or source_count <= 0:
            raise ValueError("Capability contract example source 数量非法")
        example_sources = []
        for _ in range(source_count):
            item, cursor = _take(
                values, cursor, label="Capability contract example source")
            example_sources.append(SourceRef.from_stable_key(item))
        held_out_source_key, cursor = _take(
            values, cursor, label="Capability contract held-out source")
        verification_key, cursor = _take(
            values, cursor, label="Capability contract verifier")
        specification_key, cursor = _take(
            values, cursor, label="Capability contract specification authority")
        provenance_ref, cursor = _take(
            values, cursor, label="Capability contract provenance")
        if cursor != len(values):
            raise ValueError("Capability contract 含尾随字段")
        definition = formal_artifact_definition_from_stable_key(definition_key)
        contract = cls(
            MemoryLinkedRef.from_stable_key(capability_kind_key),
            definition,
            tuple(preconditions),
            report_key,
            _status_protocol_from_key(protocol_key),
            ObjectIdentity.from_stable_key(previous_key),
            ObjectIdentity.from_stable_key(state_key),
            SourceRef.from_stable_key(candidate_source_key),
            tuple(example_sources),
            SourceRef.from_stable_key(held_out_source_key),
            _authority_from_key(verification_key),
            _authority_from_key(specification_key),
            provenance_ref,
        )
        if contract.verification_authority.stable_key() != verification_key:
            raise ValueError("Capability contract verifier 恢复漂移")
        if contract.stable_key() != values:
            raise ValueError("Capability contract 恢复结果与输入不一致")
        return contract


def _authority_from_key(key: tuple[int, ...]) -> ArtifactAuthority:
    """延迟导入共享 authority codec，保持 contract parser 单向依赖。"""
    from pure_integer_ai.cognition.shared.formal_artifact import (
        artifact_authority_from_stable_key,
    )
    return artifact_authority_from_stable_key(key)


@dataclass(frozen=True)
class RecoveredCapability:
    """由同一 Memory 空间的 Artifact 与 Capability 声明恢复的能力。"""

    capability_ref: MemoryObjectRef
    capability_event: MaterializedMemoryEvent
    artifact_event: MaterializedMemoryEvent
    payload: CapabilityPayload
    contract: VerifiedCapabilityContract
    definition: FormalArtifactDefinition


def recover_verified_capability(
        event_log: MemoryEventLog,
        capability_ref: MemoryObjectRef,
        *,
        access: MemoryAccessContext,
        ) -> RecoveredCapability:
    """只读恢复唯一已激活 Capability，并逐字段核对程序及 verified provenance。"""
    if not isinstance(event_log, MemoryEventLog):
        raise TypeError("event_log 必须是 MemoryEventLog")
    if (not isinstance(capability_ref, MemoryObjectRef)
            or capability_ref.object_kind != MEMORY_OBJECT_CAPABILITY):
        raise ValueError("capability_ref 必须指向 Capability")
    capability_events = event_log.query(
        access=access,
        event_kind=MEMORY_EVENT_CAPABILITY,
        object_ref=capability_ref,
    )
    if len(capability_events) != 1 or not isinstance(
            capability_events[0].event.payload, CapabilityPayload):
        raise ValueError("Capability 没有唯一可见声明")
    payload = capability_events[0].event.payload
    contract = VerifiedCapabilityContract.from_stable_key(
        payload.contract_key)
    if payload.capability_kind != contract.capability_kind:
        raise ValueError("Capability kind 与 verified contract 漂移")
    artifact_events = event_log.query(
        access=access,
        event_kind=MEMORY_EVENT_ARTIFACT,
        object_ref=payload.program_ref,
    )
    if len(artifact_events) != 1 or not isinstance(
            artifact_events[0].event.payload, ArtifactPayload):
        raise ValueError("Capability program 没有唯一可见 Artifact 声明")
    artifact = artifact_events[0].event.payload
    definition = contract.definition
    if (payload.program_ref.object_kind != MEMORY_OBJECT_ARTIFACT
            or payload.program_ref.object_key
            != definition.program.identity.stable_key()
            or artifact.artifact != definition.program.identity):
        raise ValueError("Capability program Artifact 与 definition 漂移")
    if (definition.program.source != contract.candidate_source
            or definition.executor != contract.definition.executor
            or definition.verifier != contract.verification_authority):
        raise ValueError("Capability 恢复后的来源或 authority 漂移")
    return RecoveredCapability(
        capability_ref,
        capability_events[0],
        artifact_events[0],
        payload,
        contract,
        definition,
    )


__all__ = [
    "RecoveredCapability",
    "VerifiedCapabilityContract",
    "recover_verified_capability",
]
