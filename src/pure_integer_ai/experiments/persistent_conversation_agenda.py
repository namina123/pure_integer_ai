"""只保存 center/query/record key 的 append-only 持久会话 agenda。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Callable

from pure_integer_ai.experiments.authorized_center_runtime import (
    AuthorizedCenterAgendaRun,
)
from pure_integer_ai.experiments.free_text_recall_runtime import (
    EvidenceFormedCenter,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
)
from pure_integer_ai.storage.segment_repository import (
    AppendOnlyObjectRepository,
)


OBJECT_KIND_CONVERSATION_AGENDA = 6
AGENDA_OPEN = 1
AGENDA_RESOLVED = 2
AGENDA_BLOCKED = 3
_LIFECYCLES = frozenset({AGENDA_OPEN, AGENDA_RESOLVED, AGENDA_BLOCKED})
_FORMAT_VERSION = 1
_DIGEST_SIZE = 32


class PersistentConversationAgendaError(RuntimeError):
    """agenda identity、revision 链、center metadata 或持久化不闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """按统一整数 codec 分帧写入键。"""
    pack_key(result, value)


def _digest(values: tuple[int, ...]) -> tuple[int, ...]:
    """返回 append-only snapshot 的完整 SHA-256 字节元组。"""
    return tuple(hashlib.sha256(encode_integer_tuple(values)).digest())


def _digest_key(
        value: tuple[int, ...], *, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """核验空初始指针或完整 SHA-256 字节元组。"""
    if allow_empty and value == ():
        return value
    if (not isinstance(value, tuple) or len(value) != _DIGEST_SIZE
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise PersistentConversationAgendaError("agenda digest 非法")
    return value


@dataclass(frozen=True, order=True)
class PersistentAgendaCenter:
    """持久账本中的 center、typed query、record、依赖和消费收据。"""

    center_key: StableRecordKey
    query_key: StableRecordKey
    record_key: StableRecordKey
    dependencies: tuple[StableRecordKey, ...]
    lifecycle: int
    last_logical_seq: int
    consumer_receipt_keys: tuple[StableRecordKey, ...] = ()

    def __post_init__(self) -> None:
        """核验 key 类型、依赖 DAG 局部形状、生命周期和逻辑序。"""
        for name in ("center_key", "query_key", "record_key"):
            if not isinstance(getattr(self, name), StableRecordKey):
                raise TypeError(f"persistent agenda {name} 类型错误")
        if (not isinstance(self.dependencies, tuple)
                or any(not isinstance(item, StableRecordKey)
                       for item in self.dependencies)):
            raise TypeError("agenda dependencies 类型错误")
        dependencies = tuple(sorted(set(self.dependencies)))
        if dependencies != self.dependencies:
            raise PersistentConversationAgendaError(
                "agenda dependencies 必须排序去重")
        if self.center_key in dependencies:
            raise PersistentConversationAgendaError("center 不得依赖自身")
        if self.lifecycle not in _LIFECYCLES:
            raise PersistentConversationAgendaError("agenda lifecycle 未注册")
        if type(self.last_logical_seq) is not int or self.last_logical_seq < 0:
            raise PersistentConversationAgendaError("agenda logical seq 非法")
        if (not isinstance(self.consumer_receipt_keys, tuple)
                or any(not isinstance(item, StableRecordKey)
                       for item in self.consumer_receipt_keys)):
            raise TypeError("agenda consumer receipts 类型错误")
        receipts = tuple(sorted(set(self.consumer_receipt_keys)))
        if receipts != self.consumer_receipt_keys:
            raise PersistentConversationAgendaError(
                "agenda consumer receipts 必须排序去重")

    @classmethod
    def from_formed_center(
            cls,
            center: EvidenceFormedCenter,
            query_key: StableRecordKey,
            *,
            dependencies: tuple[StableRecordKey, ...] = (),
            lifecycle: int = AGENDA_OPEN,
            logical_seq: int = 0,
            ) -> "PersistentAgendaCenter":
        """从真实 Evidence center 建立不含 payload 的长期引用。"""
        if not isinstance(center, EvidenceFormedCenter):
            raise TypeError("formed center 类型错误")
        if not isinstance(query_key, StableRecordKey):
            raise TypeError("agenda query_key 类型错误")
        return cls(
            center.center_key,
            query_key,
            StableRecordKey(tuple(center.index_entry.record_key)),
            dependencies,
            lifecycle,
            logical_seq,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 center 全部长期元数据，不含任何事实 payload。"""
        result: list[int] = []
        for value in (
                self.center_key.components,
                self.query_key.components,
                self.record_key.components):
            _pack(result, value)
        result.append(len(self.dependencies))
        for item in self.dependencies:
            _pack(result, item.components)
        result.extend((self.lifecycle, self.last_logical_seq))
        result.append(len(self.consumer_receipt_keys))
        for item in self.consumer_receipt_keys:
            _pack(result, item.components)
        return tuple(result)


def _validate_dependency_graph(
        centers: tuple[PersistentAgendaCenter, ...],
        ) -> None:
    """要求所有依赖指向同一 snapshot center 且整图无环。"""
    by_key = {item.center_key: item for item in centers}
    if len(by_key) != len(centers):
        raise PersistentConversationAgendaError("agenda center key 重复")
    for item in centers:
        if any(dependency not in by_key for dependency in item.dependencies):
            raise PersistentConversationAgendaError("agenda dependency 指向外部 center")
    active: set[StableRecordKey] = set()
    complete: set[StableRecordKey] = set()

    def visit(key: StableRecordKey) -> None:
        if key in complete:
            return
        if key in active:
            raise PersistentConversationAgendaError("agenda dependency 形成环")
        active.add(key)
        for dependency in by_key[key].dependencies:
            visit(dependency)
        active.remove(key)
        complete.add(key)

    for key in sorted(by_key):
        visit(key)


@dataclass(frozen=True)
class PersistentConversationAgenda:
    """一个 append-only revision 的小型持久中心账本。"""

    agenda_key: StableRecordKey
    revision: int
    previous_digest: tuple[int, ...]
    centers: tuple[PersistentAgendaCenter, ...]

    def __post_init__(self) -> None:
        """核验 revision 链指针、中心排序和依赖 DAG。"""
        if not isinstance(self.agenda_key, StableRecordKey):
            raise TypeError("persistent agenda key 类型错误")
        if type(self.revision) is not int or self.revision < 0:
            raise PersistentConversationAgendaError("agenda revision 非法")
        _digest_key(self.previous_digest, allow_empty=self.revision == 0)
        if (self.revision == 0) != (self.previous_digest == ()):
            raise PersistentConversationAgendaError("agenda previous digest 链断裂")
        if (not isinstance(self.centers, tuple)
                or any(not isinstance(item, PersistentAgendaCenter)
                       for item in self.centers)):
            raise TypeError("persistent agenda centers 类型错误")
        ordered = tuple(sorted(self.centers, key=lambda item: item.center_key))
        if ordered != self.centers:
            raise PersistentConversationAgendaError("agenda centers 未规范排序")
        _validate_dependency_graph(self.centers)

    def stable_key(self) -> tuple[int, ...]:
        """返回 agenda identity、revision、前驱和全部 center metadata。"""
        result = [
            _FORMAT_VERSION,
            *self.agenda_key.components,
            self.revision,
        ]
        _pack(result, self.previous_digest)
        result.append(len(self.centers))
        for item in self.centers:
            _pack(result, item.stable_key())
        return tuple(result)

    def digest(self) -> tuple[int, ...]:
        """返回当前完整 snapshot 摘要，用作下一 revision 指针。"""
        return _digest(self.stable_key())


def _serialize(snapshot: PersistentConversationAgenda) -> bytes:
    """把 agenda snapshot 编为规范整数流。"""
    values: list[int] = [_FORMAT_VERSION]
    _pack(values, snapshot.agenda_key.components)
    values.append(snapshot.revision)
    _pack(values, snapshot.previous_digest)
    values.append(len(snapshot.centers))
    for center in snapshot.centers:
        for key in (
                center.center_key,
                center.query_key,
                center.record_key):
            _pack(values, key.components)
        values.append(len(center.dependencies))
        for dependency in center.dependencies:
            _pack(values, dependency.components)
        values.extend((center.lifecycle, center.last_logical_seq))
        values.append(len(center.consumer_receipt_keys))
        for receipt in center.consumer_receipt_keys:
            _pack(values, receipt.components)
    return encode_integer_tuple(tuple(values))


def _deserialize(payload: bytes) -> PersistentConversationAgenda:
    """从规范整数流恢复 agenda，并拒绝截断或尾字段。"""
    try:
        reader = IntegerStreamReader(decode_integer_tuple(payload))
        version = reader.read_positive(label="agenda format")
        if version != _FORMAT_VERSION:
            raise PersistentConversationAgendaError("agenda format 不兼容")
        agenda_key = StableRecordKey(reader.read_key(label="agenda key"))
        revision = reader.read_nonnegative(label="agenda revision")
        previous = reader.read_key(
            label="agenda previous digest", empty=True)
        center_count = reader.read_nonnegative(label="agenda center count")
        centers = []
        for index in range(center_count):
            center_key = StableRecordKey(reader.read_key(
                label=f"center[{index}].center_key"))
            query_key = StableRecordKey(reader.read_key(
                label=f"center[{index}].query_key"))
            record_key = StableRecordKey(reader.read_key(
                label=f"center[{index}].record_key"))
            dependency_count = reader.read_nonnegative(
                label=f"center[{index}].dependency_count")
            dependencies = tuple(
                StableRecordKey(reader.read_key(
                    label=f"center[{index}].dependency[{ordinal}]"))
                for ordinal in range(dependency_count)
            )
            lifecycle = reader.read_positive(
                label=f"center[{index}].lifecycle")
            logical_seq = reader.read_nonnegative(
                label=f"center[{index}].logical_seq")
            receipt_count = reader.read_nonnegative(
                label=f"center[{index}].receipt_count")
            receipts = tuple(
                StableRecordKey(reader.read_key(
                    label=f"center[{index}].receipt[{ordinal}]"))
                for ordinal in range(receipt_count)
            )
            centers.append(PersistentAgendaCenter(
                center_key,
                query_key,
                record_key,
                dependencies,
                lifecycle,
                logical_seq,
                receipts,
            ))
        reader.finish()
        return PersistentConversationAgenda(
            agenda_key,
            revision,
            previous,
            tuple(centers),
        )
    except PersistentConversationAgendaError:
        raise
    except (IntegerCodecError, TypeError, ValueError) as error:
        raise PersistentConversationAgendaError("agenda payload 损坏") from error


def _object_identity(
        agenda_key: StableRecordKey, revision: int,
        ) -> tuple[int, ...]:
    """形成 repository 中不替代完整 agenda key 的 revision identity。"""
    return (
        _FORMAT_VERSION,
        len(agenda_key.components),
        *agenda_key.components,
        revision,
    )


def _identity_matches(
        identity: tuple[int, ...], agenda_key: StableRecordKey,
        ) -> bool:
    """判断 sealed object identity 是否属于目标 agenda。"""
    size = len(agenda_key.components)
    return (
        len(identity) == size + 3
        and identity[0] == _FORMAT_VERSION
        and identity[1] == size
        and identity[2:2 + size] == agenda_key.components
        and type(identity[-1]) is int
        and identity[-1] >= 0
    )


class PersistentConversationAgendaStore:
    """在通用 seal-last object repository 上管理一个 agenda revision 链。"""

    def __init__(
            self,
            repository: AppendOnlyObjectRepository,
            commit: Callable[[], None],
            ) -> None:
        """绑定 append-only repository 和显式事务 owner。"""
        if not isinstance(repository, AppendOnlyObjectRepository):
            raise TypeError("agenda repository 协议错误")
        if not callable(commit):
            raise TypeError("agenda commit 必须可调用")
        self.repository = repository
        self.commit = commit

    def create(
            self,
            agenda_key: StableRecordKey,
            centers: tuple[PersistentAgendaCenter, ...],
            ) -> PersistentConversationAgenda:
        """独占创建 revision 0，不读取或复制任何长期事实 payload。"""
        if self._descriptors(agenda_key):
            raise PersistentConversationAgendaError("agenda 已存在")
        snapshot = PersistentConversationAgenda(
            agenda_key,
            0,
            (),
            tuple(sorted(centers, key=lambda item: item.center_key)),
        )
        self._put(snapshot)
        return snapshot

    def load(
            self, agenda_key: StableRecordKey,
            ) -> PersistentConversationAgenda:
        """只读取小型 agenda metadata，并重验完整 revision/digest 链。"""
        descriptors = self._descriptors(agenda_key)
        if not descriptors:
            raise KeyError(f"persistent agenda 不存在: {agenda_key.components}")
        snapshots = []
        for descriptor in descriptors:
            snapshot = _deserialize(self.repository.get(
                OBJECT_KIND_CONVERSATION_AGENDA,
                descriptor.identity_key,
            ))
            if (snapshot.agenda_key != agenda_key
                    or descriptor.identity_key != _object_identity(
                        agenda_key, snapshot.revision)):
                raise PersistentConversationAgendaError(
                    "agenda object identity 与 payload 漂移")
            snapshots.append(snapshot)
        snapshots.sort(key=lambda item: item.revision)
        if tuple(item.revision for item in snapshots) != tuple(
                range(len(snapshots))):
            raise PersistentConversationAgendaError("agenda revision 不连续")
        previous = None
        for snapshot in snapshots:
            expected = () if previous is None else previous.digest()
            if snapshot.previous_digest != expected:
                raise PersistentConversationAgendaError("agenda digest 链断裂")
            previous = snapshot
        assert previous is not None
        return previous

    def advance(
            self,
            current: PersistentConversationAgenda,
            centers: tuple[PersistentAgendaCenter, ...],
            ) -> PersistentConversationAgenda:
        """比较当前 revision 后追加新 snapshot，拒绝静态身份倒退或删除。"""
        if not isinstance(current, PersistentConversationAgenda):
            raise TypeError("current agenda 类型错误")
        latest = self.load(current.agenda_key)
        if latest != current:
            raise PersistentConversationAgendaError("agenda revision 已被并发推进")
        ordered = tuple(sorted(centers, key=lambda item: item.center_key))
        self._validate_transition(current.centers, ordered)
        next_snapshot = PersistentConversationAgenda(
            current.agenda_key,
            current.revision + 1,
            current.digest(),
            ordered,
        )
        self._put(next_snapshot)
        return next_snapshot

    def record_authorized_run(
            self,
            current: PersistentConversationAgenda,
            run: AuthorizedCenterAgendaRun,
            *,
            lifecycle_by_center: tuple[
                tuple[StableRecordKey, int, int], ...] = (),
            ) -> PersistentConversationAgenda:
        """只追加各 center 独立 receipt/lifecycle，不保存 run payload。"""
        if not isinstance(run, AuthorizedCenterAgendaRun):
            raise TypeError("authorized agenda run 类型错误")
        lifecycle_map = {
            key: (lifecycle, logical_seq)
            for key, lifecycle, logical_seq in lifecycle_by_center
        }
        if len(lifecycle_map) != len(lifecycle_by_center):
            raise PersistentConversationAgendaError("lifecycle center 重复")
        state_by_center = {
            item.center.center_key: item for item in run.states
        }
        current_keys = {item.center_key for item in current.centers}
        if set(state_by_center) - current_keys:
            raise PersistentConversationAgendaError(
                "authorized run 含 agenda 外 center")
        next_centers = []
        for center in current.centers:
            state = state_by_center.get(center.center_key)
            receipts = center.consumer_receipt_keys
            if state is not None:
                receipts = tuple(sorted(set((
                    *receipts,
                    state.receipt.receipt_key,
                ))))
            lifecycle, logical_seq = lifecycle_map.get(
                center.center_key,
                (center.lifecycle, center.last_logical_seq),
            )
            next_centers.append(replace(
                center,
                lifecycle=lifecycle,
                last_logical_seq=logical_seq,
                consumer_receipt_keys=receipts,
            ))
        if set(lifecycle_map) - {item.center_key for item in current.centers}:
            raise PersistentConversationAgendaError("lifecycle 指向外部 center")
        return self.advance(current, tuple(next_centers))

    def _put(self, snapshot: PersistentConversationAgenda) -> None:
        """seal-last 发布单个完整 snapshot，再由显式 owner 提交。"""
        self.repository.put(
            OBJECT_KIND_CONVERSATION_AGENDA,
            _object_identity(snapshot.agenda_key, snapshot.revision),
            _serialize(snapshot),
        )
        self.commit()

    def _descriptors(self, agenda_key: StableRecordKey):
        """按完整 key 过滤 agenda metadata descriptors。"""
        if not isinstance(agenda_key, StableRecordKey):
            raise TypeError("agenda_key 类型错误")
        return tuple(
            item
            for item in self.repository.list_kind(
                OBJECT_KIND_CONVERSATION_AGENDA)
            if _identity_matches(item.identity_key, agenda_key)
        )

    @staticmethod
    def _validate_transition(
            before: tuple[PersistentAgendaCenter, ...],
            after: tuple[PersistentAgendaCenter, ...],
            ) -> None:
        """禁止删除 center、替换 query/record/dependency 或倒退 lifecycle/receipt。"""
        previous = {item.center_key: item for item in before}
        current = {item.center_key: item for item in after}
        if not set(previous).issubset(current):
            raise PersistentConversationAgendaError("agenda 不得删除 center")
        allowed = {
            AGENDA_OPEN: {AGENDA_OPEN, AGENDA_RESOLVED, AGENDA_BLOCKED},
            AGENDA_BLOCKED: {AGENDA_BLOCKED, AGENDA_OPEN, AGENDA_RESOLVED},
            AGENDA_RESOLVED: {AGENDA_RESOLVED},
        }
        for key, old in previous.items():
            new = current[key]
            if (old.query_key != new.query_key
                    or old.record_key != new.record_key
                    or old.dependencies != new.dependencies):
                raise PersistentConversationAgendaError(
                    "agenda center 静态 identity 漂移")
            if new.lifecycle not in allowed[old.lifecycle]:
                raise PersistentConversationAgendaError("agenda lifecycle 倒退")
            if new.last_logical_seq < old.last_logical_seq:
                raise PersistentConversationAgendaError("agenda logical seq 倒退")
            if not set(old.consumer_receipt_keys).issubset(
                    new.consumer_receipt_keys):
                raise PersistentConversationAgendaError("agenda receipt 被删除")
            if (new.lifecycle != old.lifecycle
                    or new.consumer_receipt_keys != old.consumer_receipt_keys):
                if new.last_logical_seq <= old.last_logical_seq:
                    raise PersistentConversationAgendaError(
                        "agenda 状态变化必须推进 logical seq")


@dataclass(frozen=True)
class PersistentAgendaBinding:
    """从持久 metadata 匹配出的当前真实 center 集。"""

    agenda: PersistentConversationAgenda
    centers: tuple[EvidenceFormedCenter, ...]

    def __post_init__(self) -> None:
        """核验绑定只包含 agenda 中 open center 且 identity/record 精确。"""
        if not isinstance(self.agenda, PersistentConversationAgenda):
            raise TypeError("agenda binding snapshot 类型错误")
        if (not isinstance(self.centers, tuple)
                or any(not isinstance(item, EvidenceFormedCenter)
                       for item in self.centers)):
            raise TypeError("agenda binding centers 类型错误")


class PersistentConversationAgendaRuntime:
    """把持久 center/record metadata 重新绑定到当前 Evidence center。"""

    @staticmethod
    def bind(
            agenda: PersistentConversationAgenda,
            available: tuple[EvidenceFormedCenter, ...],
            ) -> PersistentAgendaBinding:
        """按 exact center+record 绑定 open center；agenda 自身不授予 payload 权限。"""
        if not isinstance(agenda, PersistentConversationAgenda):
            raise TypeError("persistent agenda 类型错误")
        if (not isinstance(available, tuple)
                or any(not isinstance(item, EvidenceFormedCenter)
                       for item in available)):
            raise TypeError("available centers 类型错误")
        by_key = {item.center_key: item for item in available}
        if len(by_key) != len(available):
            raise PersistentConversationAgendaError("available center key 重复")
        bound = []
        for reference in agenda.centers:
            if reference.lifecycle != AGENDA_OPEN:
                continue
            center = by_key.get(reference.center_key)
            if center is None:
                raise PersistentConversationAgendaError(
                    "open agenda center 无当前 Evidence 形成结果")
            if tuple(center.index_entry.record_key) != (
                    reference.record_key.components):
                raise PersistentConversationAgendaError(
                    "open agenda center record identity 漂移")
            bound.append(center)
        return PersistentAgendaBinding(
            agenda,
            tuple(sorted(bound)),
        )


__all__ = [
    "AGENDA_BLOCKED",
    "AGENDA_OPEN",
    "AGENDA_RESOLVED",
    "OBJECT_KIND_CONVERSATION_AGENDA",
    "PersistentAgendaBinding",
    "PersistentAgendaCenter",
    "PersistentConversationAgenda",
    "PersistentConversationAgendaError",
    "PersistentConversationAgendaRuntime",
    "PersistentConversationAgendaStore",
]
