"""精确授权投影驱动的多中心 agenda 与共享冷记录读取。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.identity import OwnerScope, SourceRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.free_text_recall_runtime import (
    AclFirstExactRecallReader,
    EvidenceFormedCenter,
    ExactRecallResult,
    TypedRecallPayload,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    RecallBudget,
    RecallCitation,
)


_READY = "READY"
_STOPPED = "STOPPED"
_REJECTION_STATES = {
    "ACL_DENIED",
    "CENTER_UNBOUND",
    "DESCRIPTOR_MISMATCH",
    "MANIFEST_STALE",
    "POLICY_STALE",
    "READ_FAILED",
    "RECORD_MISMATCH",
    "SCOPE_MISMATCH",
    "SEGMENT_MISMATCH",
    "SEGMENT_NOT_ISOLATED",
    "SOURCE_MISMATCH",
    "VERSION_MISMATCH",
}
_CENTER_STATES = {_READY, _STOPPED, *_REJECTION_STATES}


class AuthorizedCenterRuntimeError(RuntimeError):
    """授权投影、物理位置或多中心状态不闭合。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """要求协议身份是非空正严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item <= 0 for item in value)):
        raise AuthorizedCenterRuntimeError(f"{where} 必须是正严格整数 tuple")
    return value


def _integer_stream(
        value: tuple[int, ...], *, where: str,
        ) -> tuple[int, ...]:
    """要求内容引用流非空、仅含非负严格整数，但允许合法零位。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise AuthorizedCenterRuntimeError(
            f"{where} 必须是非负严格整数 tuple")
    return value


def _stable(domain: int, *parts: int) -> StableRecordKey:
    """从完整整数输入形成确定性、正整数稳定键。"""
    if type(domain) is not int or domain <= 0:
        raise AuthorizedCenterRuntimeError("stable domain 非法")
    if any(type(item) is not int for item in parts):
        raise AuthorizedCenterRuntimeError("stable parts 非法")
    raw = ":".join(str(item) for item in (domain, *parts)).encode("ascii")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((domain, value if value else 1))


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长稳定键按长度分帧写入结果。"""
    result.extend((len(value), *value))


@dataclass(frozen=True, order=True)
class CenterAuthorizationBinding:
    """中心到唯一记录、来源、scope、版本和物理段的授权绑定。"""

    center_key: StableRecordKey
    descriptor_key: tuple[int, ...]
    record_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    version_key: tuple[int, ...]
    owner: OwnerScope
    segment_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验绑定内部的来源、owner、scope 和版本不可互相漂移。"""
        if not isinstance(self.center_key, StableRecordKey):
            raise TypeError("authorization center_key 类型错误")
        for name in (
                "descriptor_key", "record_key", "version_key", "segment_key"):
            _strict_key(getattr(self, name), where=f"authorization {name}")
        if not isinstance(self.source, SourceRef):
            raise TypeError("authorization source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("authorization scope 类型错误")
        if not isinstance(self.owner, OwnerScope):
            raise TypeError("authorization owner 类型错误")
        if (self.source.owner != self.owner
                or self.scope.owner != self.owner
                or self.scope.source != self.source
                or self.scope.versions != self.source.versions):
            raise AuthorizedCenterRuntimeError("authorization source/scope/owner 漂移")

    def stable_key(self) -> tuple[int, ...]:
        """返回精确绑定的完整稳定整数键。"""
        result: list[int] = []
        for value in (
                self.center_key.components,
                self.descriptor_key,
                self.record_key,
                self.source.stable_key(),
                self.scope.stable_key(),
                self.version_key,
                self.owner.stable_key(),
                self.segment_key):
            _pack(result, value)
        return tuple(result)


@dataclass(frozen=True)
class CenterAuthorizationProjection:
    """一个 policy epoch 对当前 location epoch 的只读授权投影。"""

    projection_key: StableRecordKey
    policy_key: tuple[int, ...]
    policy_epoch: int
    manifest_key: tuple[int, ...]
    manifest_epoch: int
    access: MemoryAccessContext
    bindings: tuple[CenterAuthorizationBinding, ...]

    def __post_init__(self) -> None:
        """规范化绑定并拒绝同一中心或记录出现互相冲突的授权。"""
        if not isinstance(self.projection_key, StableRecordKey):
            raise TypeError("authorization projection_key 类型错误")
        _strict_key(self.policy_key, where="authorization policy_key")
        _strict_key(self.manifest_key, where="authorization manifest_key")
        for name in ("policy_epoch", "manifest_epoch"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise AuthorizedCenterRuntimeError(f"{name} 必须是正严格整数")
        if not isinstance(self.access, MemoryAccessContext):
            raise TypeError("authorization access 类型错误")
        if (not isinstance(self.bindings, tuple) or not self.bindings
                or any(not isinstance(item, CenterAuthorizationBinding)
                       for item in self.bindings)):
            raise TypeError("authorization bindings 类型错误")
        bindings = tuple(sorted(self.bindings))
        center_keys = tuple(item.center_key for item in bindings)
        if len(center_keys) != len(set(center_keys)):
            raise AuthorizedCenterRuntimeError("同一 center 不得重复授权")
        record_contracts: dict[tuple[int, ...], tuple[object, ...]] = {}
        for item in bindings:
            contract = (
                item.descriptor_key,
                item.source,
                item.scope,
                item.version_key,
                item.owner,
                item.segment_key,
            )
            previous = record_contracts.setdefault(item.record_key, contract)
            if previous != contract:
                raise AuthorizedCenterRuntimeError("同一 record 的授权合同冲突")
        object.__setattr__(self, "bindings", bindings)

    def stable_key(self) -> tuple[int, ...]:
        """返回 policy、manifest、access 和全部精确绑定的稳定键。"""
        result: list[int] = []
        for value in (
                self.projection_key.components,
                self.policy_key,
                self.manifest_key,
                self.access.stable_key()):
            _pack(result, value)
        result.extend((self.policy_epoch, self.manifest_epoch, len(self.bindings)))
        for binding in self.bindings:
            _pack(result, binding.stable_key())
        return tuple(result)


@dataclass(frozen=True, order=True)
class AuthorizedCenterObligation:
    """一个中心独占的 frontier 与授权消费义务。"""

    obligation_key: StableRecordKey
    center_key: StableRecordKey
    frontier_key: StableRecordKey
    binding_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验义务身份、center 和 frontier 均为独立稳定身份。"""
        for name in ("obligation_key", "center_key", "frontier_key"):
            if not isinstance(getattr(self, name), StableRecordKey):
                raise TypeError(f"authorized obligation {name} 类型错误")
        _integer_stream(
            self.binding_key, where="authorized obligation binding_key")


@dataclass(frozen=True, order=True)
class AuthorizedCenterReceipt:
    """一个中心独占的停止状态和共享读取归因收据。"""

    receipt_key: StableRecordKey
    obligation_key: StableRecordKey
    center_key: StableRecordKey
    frontier_key: StableRecordKey
    state: str
    shared_read_key: StableRecordKey | None
    physical_payload_gets: int
    reused_payload: int
    citations: tuple[RecallCitation, ...]

    def __post_init__(self) -> None:
        """核验读取 owner、复用者和拒绝状态的 payload 边界。"""
        for name in (
                "receipt_key", "obligation_key", "center_key", "frontier_key"):
            if not isinstance(getattr(self, name), StableRecordKey):
                raise TypeError(f"authorized receipt {name} 类型错误")
        if self.state not in _CENTER_STATES:
            raise AuthorizedCenterRuntimeError("authorized receipt state 非法")
        if (self.shared_read_key is not None
                and not isinstance(self.shared_read_key, StableRecordKey)):
            raise TypeError("authorized receipt shared_read_key 类型错误")
        for name in ("physical_payload_gets", "reused_payload"):
            if getattr(self, name) not in (0, 1):
                raise AuthorizedCenterRuntimeError(f"{name} 必须是 0/1")
        if self.physical_payload_gets and self.reused_payload:
            raise AuthorizedCenterRuntimeError("同一 center 不得同时读取并复用")
        if (not isinstance(self.citations, tuple)
                or any(not isinstance(item, RecallCitation)
                       for item in self.citations)):
            raise TypeError("authorized receipt citations 类型错误")
        if self.state == _READY:
            if self.shared_read_key is None or not self.citations:
                raise AuthorizedCenterRuntimeError("READY center 缺共享读或 citation")
            if self.physical_payload_gets + self.reused_payload != 1:
                raise AuthorizedCenterRuntimeError("READY center 缺读取归因")
        elif self.state == _STOPPED:
            if self.shared_read_key is None or self.citations:
                raise AuthorizedCenterRuntimeError(
                    "STOPPED center 必须保留读归因但移除 citation")
            if self.physical_payload_gets + self.reused_payload != 1:
                raise AuthorizedCenterRuntimeError("STOPPED center 缺读取归因")
        elif any((
                self.physical_payload_gets,
                self.reused_payload,
                len(self.citations),
        )):
            raise AuthorizedCenterRuntimeError("非 READY center 不得暴露 payload 归因")


@dataclass(frozen=True, order=True)
class AuthorizedCenterState:
    """一个中心独立的 obligation、receipt、frontier 和可选 payload。"""

    center: EvidenceFormedCenter
    obligation: AuthorizedCenterObligation
    receipt: AuthorizedCenterReceipt
    payload: TypedRecallPayload | None

    def __post_init__(self) -> None:
        """核验中心身份贯穿义务和收据，拒绝状态不携带 payload。"""
        if not isinstance(self.center, EvidenceFormedCenter):
            raise TypeError("authorized state center 类型错误")
        if not isinstance(self.obligation, AuthorizedCenterObligation):
            raise TypeError("authorized state obligation 类型错误")
        if not isinstance(self.receipt, AuthorizedCenterReceipt):
            raise TypeError("authorized state receipt 类型错误")
        center_key = self.center.center_key
        if (self.obligation.center_key != center_key
                or self.receipt.center_key != center_key
                or self.receipt.obligation_key != self.obligation.obligation_key
                or self.receipt.frontier_key != self.obligation.frontier_key):
            raise AuthorizedCenterRuntimeError("center obligation/receipt 身份漂移")
        if self.receipt.state == _READY:
            if not isinstance(self.payload, TypedRecallPayload):
                raise TypeError("READY center 缺 typed payload")
        elif self.payload is not None:
            raise AuthorizedCenterRuntimeError("非 READY center 不得暴露 payload")


@dataclass(frozen=True, order=True)
class AuthorizedRecordRead:
    """一个唯一授权 record 的单次物理读取审计。"""

    shared_read_key: StableRecordKey
    record_key: tuple[int, ...]
    exact: ExactRecallResult

    def __post_init__(self) -> None:
        """核验共享读取确实返回当前 record 的已解决 payload。"""
        if not isinstance(self.shared_read_key, StableRecordKey):
            raise TypeError("shared read key 类型错误")
        _strict_key(self.record_key, where="shared read record_key")
        if not isinstance(self.exact, ExactRecallResult):
            raise TypeError("shared read exact 类型错误")
        if (self.exact.payload is None
                or self.exact.receipt.stop_reason != "RESOLVED"):
            raise AuthorizedCenterRuntimeError("共享 record read 未解决")
        if self.exact.receipt.result_keys != (StableRecordKey(self.record_key),):
            raise AuthorizedCenterRuntimeError("共享 record read 返回其他记录")


@dataclass(frozen=True)
class AuthorizedCenterAgendaRun:
    """多中心独立状态和按 record 去重后的物理读取报告。"""

    projection_key: StableRecordKey
    states: tuple[AuthorizedCenterState, ...]
    record_reads: tuple[AuthorizedRecordRead, ...]
    host_write_count: int = 0
    private_label_read_count: int = 0

    def __post_init__(self) -> None:
        """核验中心、frontier、义务和读身份唯一且物理读恰好逐记录一次。"""
        if not isinstance(self.projection_key, StableRecordKey):
            raise TypeError("agenda projection_key 类型错误")
        if (not isinstance(self.states, tuple) or not self.states
                or any(not isinstance(item, AuthorizedCenterState)
                       for item in self.states)):
            raise TypeError("agenda states 类型错误")
        if (not isinstance(self.record_reads, tuple)
                or any(not isinstance(item, AuthorizedRecordRead)
                       for item in self.record_reads)):
            raise TypeError("agenda record_reads 类型错误")
        states = tuple(sorted(self.states, key=lambda item: item.center.center_key))
        reads = tuple(sorted(self.record_reads, key=lambda item: item.record_key))
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "record_reads", reads)
        for values, label in (
                ((item.center.center_key for item in states), "center"),
                ((item.obligation.obligation_key for item in states), "obligation"),
                ((item.obligation.frontier_key for item in states), "frontier"),
                ((item.receipt.receipt_key for item in states), "receipt"),
                ((item.record_key for item in reads), "record read"),
                ((item.shared_read_key for item in reads), "shared read")):
            materialized = tuple(values)
            if len(materialized) != len(set(materialized)):
                raise AuthorizedCenterRuntimeError(f"agenda {label} 身份重复")
        consumers = tuple(
            item for item in states
            if item.receipt.state in {_READY, _STOPPED})
        if sum(
                item.receipt.physical_payload_gets
                for item in consumers) != len(reads):
            raise AuthorizedCenterRuntimeError("每个共享 record 必须恰有一个读取 owner")
        read_keys = {item.shared_read_key for item in reads}
        if any(
                item.receipt.shared_read_key not in read_keys
                for item in consumers):
            raise AuthorizedCenterRuntimeError("center 指向未知共享读取")
        if self.host_write_count != 0 or self.private_label_read_count != 0:
            raise AuthorizedCenterRuntimeError("authorized agenda 不得写 host 或读私有标签")

    def state(self, center_key: StableRecordKey) -> AuthorizedCenterState:
        """按精确中心身份读取独立状态，未知中心失败。"""
        matches = tuple(
            item for item in self.states if item.center.center_key == center_key)
        if len(matches) != 1:
            raise KeyError(f"未知或重复 center: {center_key}")
        return matches[0]

    def stop_center(self, center_key: StableRecordKey) -> "AuthorizedCenterAgendaRun":
        """只关闭指定中心的 frontier，不改变共享读取或其他中心状态。"""
        selected = self.state(center_key)
        if selected.receipt.state != _READY:
            raise AuthorizedCenterRuntimeError("只有 READY center 可进入 STOPPED")
        stopped_receipt = replace(
            selected.receipt,
            state=_STOPPED,
            citations=(),
        )
        stopped = replace(selected, receipt=stopped_receipt, payload=None)
        states = tuple(
            stopped if item.center.center_key == center_key else item
            for item in self.states)
        return replace(self, states=states)


class AuthorizedCenterAgendaRuntime:
    """在 payload page-in 前核完身份、来源、scope、版本、ACL 和物理隔离。"""

    def __init__(self, reader: AclFirstExactRecallReader) -> None:
        """绑定现有 ACL-first exact reader，不拥有 payload repository。"""
        if not isinstance(reader, AclFirstExactRecallReader):
            raise TypeError("authorized agenda reader 类型错误")
        self.reader = reader

    @staticmethod
    def _obligation(
            projection: CenterAuthorizationProjection,
            center: EvidenceFormedCenter,
            binding: CenterAuthorizationBinding | None,
            ) -> AuthorizedCenterObligation:
        """为每个中心建立不会因共享记录而合并的 frontier 和义务。"""
        binding_key = (
            center.center_key.components if binding is None
            else binding.stable_key())
        projection_parts = projection.projection_key.components
        frontier = _stable(
            9301, *projection_parts, *center.center_key.components)
        obligation = _stable(
            9302, *projection_parts, *center.center_key.components,
            *binding_key)
        return AuthorizedCenterObligation(
            obligation,
            center.center_key,
            frontier,
            binding_key,
        )

    @staticmethod
    def _rejected(
            projection: CenterAuthorizationProjection,
            center: EvidenceFormedCenter,
            binding: CenterAuthorizationBinding | None,
            state: str,
            ) -> AuthorizedCenterState:
        """形成不含共享读、citation 或 payload 的预检拒绝状态。"""
        if state not in _REJECTION_STATES:
            raise AuthorizedCenterRuntimeError("预检拒绝 state 非法")
        obligation = AuthorizedCenterAgendaRuntime._obligation(
            projection, center, binding)
        receipt = AuthorizedCenterReceipt(
            _stable(9303, *obligation.obligation_key.components, *map(ord, state)),
            obligation.obligation_key,
            center.center_key,
            obligation.frontier_key,
            state,
            None,
            0,
            0,
            (),
        )
        return AuthorizedCenterState(center, obligation, receipt, None)

    def _precheck(
            self,
            center: EvidenceFormedCenter,
            binding: CenterAuthorizationBinding | None,
            projection: CenterAuthorizationProjection,
            *,
            current_policy_epoch: int,
            ) -> str | None:
        """只读 manifest 元数据并返回首个拒绝原因；不得读取 segment payload。"""
        if projection.policy_epoch != current_policy_epoch:
            return "POLICY_STALE"
        manifest = self.reader.store.current_manifest()
        if (manifest is None
                or manifest.publish_epoch != projection.manifest_epoch
                or manifest.manifest_key != projection.manifest_key):
            return "MANIFEST_STALE"
        if binding is None:
            return "CENTER_UNBOUND"
        entry = center.index_entry
        if binding.center_key != center.center_key:
            return "CENTER_UNBOUND"
        if binding.record_key != entry.record_key:
            return "RECORD_MISMATCH"
        if binding.source != entry.source:
            return "SOURCE_MISMATCH"
        if binding.scope != entry.scope:
            return "SCOPE_MISMATCH"
        if (binding.owner != entry.source.owner
                or not projection.access.can_read(binding.owner)):
            return "ACL_DENIED"
        if binding.descriptor_key != self.reader.descriptor_key:
            return "DESCRIPTOR_MISMATCH"
        matches = tuple(
            item for item in manifest.entries
            if (item.descriptor_key == binding.descriptor_key
                and item.segment_key == binding.segment_key))
        if len(matches) != 1:
            return "SEGMENT_MISMATCH"
        location = matches[0]
        if location.version_key != binding.version_key:
            return "VERSION_MISMATCH"
        if (location.key_range.lower_key != binding.record_key
                or location.key_range.upper_key != binding.record_key):
            return "SEGMENT_NOT_ISOLATED"
        return None

    def run(
            self,
            centers: tuple[EvidenceFormedCenter, ...],
            current: MemoryCurrentQuery,
            projection: CenterAuthorizationProjection,
            budget: RecallBudget,
            *,
            reader_key_prefix: tuple[int, ...],
            current_policy_epoch: int,
            ) -> AuthorizedCenterAgendaRun:
        """预检所有中心，按唯一 record page-in 一次，并复制独立消费状态。"""
        if (not isinstance(centers, tuple) or not centers
                or any(not isinstance(item, EvidenceFormedCenter)
                       for item in centers)):
            raise TypeError("authorized agenda centers 类型错误")
        if len({item.center_key for item in centers}) != len(centers):
            raise AuthorizedCenterRuntimeError("agenda center identity 不得重复")
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("authorized agenda current 类型错误")
        if not isinstance(projection, CenterAuthorizationProjection):
            raise TypeError("authorized agenda projection 类型错误")
        if not isinstance(budget, RecallBudget):
            raise TypeError("authorized agenda budget 类型错误")
        _strict_key(reader_key_prefix, where="authorized reader_key_prefix")
        if type(current_policy_epoch) is not int or current_policy_epoch <= 0:
            raise AuthorizedCenterRuntimeError(
                "current_policy_epoch 必须是正严格整数")
        binding_map = {item.center_key: item for item in projection.bindings}
        accepted: dict[tuple[int, ...], list[
            tuple[EvidenceFormedCenter, CenterAuthorizationBinding]]] = {}
        states: list[AuthorizedCenterState] = []
        for center in sorted(centers):
            binding = binding_map.get(center.center_key)
            rejection = self._precheck(
                center,
                binding,
                projection,
                current_policy_epoch=current_policy_epoch,
            )
            if rejection is not None:
                states.append(self._rejected(
                    projection, center, binding, rejection))
                continue
            assert binding is not None
            accepted.setdefault(binding.record_key, []).append((center, binding))
        record_reads: list[AuthorizedRecordRead] = []
        for read_index, record_key in enumerate(sorted(accepted), start=1):
            consumers = sorted(
                accepted[record_key], key=lambda item: item[0].center_key)
            representative, _ = consumers[0]
            exact = self.reader.read(
                representative,
                current,
                projection.access,
                budget,
                reader_key=(*reader_key_prefix, read_index),
            )
            if exact.payload is None or exact.receipt.stop_reason != "RESOLVED":
                states.extend(self._rejected(
                    projection, center, binding, "READ_FAILED")
                    for center, binding in consumers)
                continue
            shared_read_key = _stable(
                9304,
                *projection.projection_key.components,
                *record_key,
                *exact.receipt.obligation_key.components,
            )
            record_reads.append(AuthorizedRecordRead(
                shared_read_key, record_key, exact))
            for consumer_index, (center, binding) in enumerate(
                    consumers, start=1):
                obligation = self._obligation(projection, center, binding)
                receipt = AuthorizedCenterReceipt(
                    _stable(
                        9305,
                        *obligation.obligation_key.components,
                        *shared_read_key.components,
                    ),
                    obligation.obligation_key,
                    center.center_key,
                    obligation.frontier_key,
                    _READY,
                    shared_read_key,
                    int(consumer_index == 1),
                    int(consumer_index != 1),
                    exact.receipt.citations,
                )
                states.append(AuthorizedCenterState(
                    center, obligation, receipt, exact.payload))
        return AuthorizedCenterAgendaRun(
            projection.projection_key,
            tuple(states),
            tuple(record_reads),
        )


__all__ = [
    "AuthorizedCenterAgendaRun",
    "AuthorizedCenterAgendaRuntime",
    "AuthorizedCenterObligation",
    "AuthorizedCenterReceipt",
    "AuthorizedCenterRuntimeError",
    "AuthorizedCenterState",
    "AuthorizedRecordRead",
    "CenterAuthorizationBinding",
    "CenterAuthorizationProjection",
]
