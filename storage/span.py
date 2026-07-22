"""一等 Span 的来源详情和完整成员区间存储。

Span 图对象由 graph_object 保存；本模块只保存可回源的规范化成员、同位序号、scope
和 parser version。结构类型、父子角色、候选状态与替代关系属于上层图断言，不进入本表。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT


SPAN_TABLE = "span"
SPAN_MEMBER_TABLE = "span_member"

SPAN_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("source_hash", TYPE_INT),
    ("scope_hash", TYPE_INT),
    ("member_count", TYPE_INT),
    ("ordinal", TYPE_INT),
    ("parser_version", TYPE_INT),
    ("envelope_start", TYPE_INT),
    ("envelope_end", TYPE_INT),
]
SPAN_MEMBER_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("member_ordinal", TYPE_INT),
    ("start", TYPE_INT),
    ("end", TYPE_INT),
]


class SpanStorageIntegrityError(RuntimeError):
    """Span 详情或成员出现缺失、重复和冲突。"""


@dataclass(frozen=True, order=True)
class SpanStorageRecord:
    """一个 Span 的来源、scope、成员数量和包络详情。"""

    space_id: int
    local_id: int
    source_hash: int
    scope_hash: int
    member_count: int
    ordinal: int
    parser_version: int
    envelope_start: int
    envelope_end: int


@dataclass(frozen=True, order=True)
class SpanMemberStorageRecord:
    """Span 的一个规范化来源区间。"""

    space_id: int
    local_id: int
    member_ordinal: int
    start: int
    end: int


def register_span_tables(backend: StorageBackend) -> None:
    """注册 Span 详情和成员两张 append-only 核心表。"""
    backend.register_table(
        SPAN_TABLE,
        SPAN_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("space_id", "local_id"),
            ("source_hash",),
            ("scope_hash",),
        ],
        core=True,
    )
    backend.register_table(
        SPAN_MEMBER_TABLE,
        SPAN_MEMBER_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("space_id", "local_id"),
            ("start", "end"),
        ],
        core=True,
    )


def _validate_record(record: SpanStorageRecord) -> None:
    """校验 Span 详情使用严格整数且包络允许零宽。"""
    values = tuple(record.__dict__.values())
    assert_int(*values, _where="SpanStorageRecord")
    if any(type(value) is not int for value in values):
        raise ValueError("SpanStorageRecord 必须使用严格整数")
    if min(record.space_id, record.local_id, record.source_hash,
           record.scope_hash, record.member_count) <= 0:
        raise ValueError("Span 编址、来源、scope 和成员数必须为正")
    if (record.ordinal < 0 or record.parser_version < 0
            or record.envelope_start < 0
            or record.envelope_end < record.envelope_start):
        raise ValueError("Span ordinal、版本或包络非法")


def _validate_members(
        record: SpanStorageRecord,
        members: tuple[tuple[int, int], ...]) -> None:
    """核验成员已规范化，且数量和包络与详情一致。"""
    if not isinstance(members, tuple) or len(members) != record.member_count:
        raise ValueError("Span 成员数量与详情不一致")
    previous_end = -1
    has_zero_width = False
    for index, member in enumerate(members):
        if not isinstance(member, tuple) or len(member) != 2:
            raise ValueError("Span member 必须是二元区间")
        start, end = member
        assert_int(start, end, _where=f"Span member[{index}]")
        if (type(start) is not int or type(end) is not int
                or start < 0 or end < start):
            raise ValueError("Span member 边界非法")
        if start == end:
            has_zero_width = True
        if index and start <= previous_end:
            raise ValueError("Span members 必须有序、不重叠且不相邻")
        previous_end = end
    if has_zero_width and len(members) != 1:
        raise ValueError("零宽 Span 不得与其他成员混合")
    if (members[0][0] != record.envelope_start
            or members[-1][1] != record.envelope_end):
        raise ValueError("Span 包络与完整成员不一致")


class SpanStore:
    """维护 Span 详情和完整成员的幂等 append-only 写入。"""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        self._verified_by_ref: dict[
            tuple[int, int],
            tuple[SpanStorageRecord, tuple[tuple[int, int], ...]],
        ] = {}

    def add(
            self, record: SpanStorageRecord,
            members: tuple[tuple[int, int], ...]
            ) -> SpanStorageRecord:
        """幂等追加 Span；同图身份的任何详情或成员冲突均拒绝。"""
        if not isinstance(record, SpanStorageRecord):
            raise TypeError("SpanStore.add 需要 SpanStorageRecord")
        _validate_record(record)
        _validate_members(record, members)
        ref = (record.space_id, record.local_id)
        cached = self._verified_by_ref.get(ref)
        if cached is not None:
            if cached != (record, members):
                raise SpanStorageIntegrityError("Span 身份已绑定冲突详情")
            return cached[0]
        existing = self._backend.select(SPAN_TABLE, where={
            "space_id": record.space_id,
            "local_id": record.local_id,
        })
        if existing:
            restored, restored_members = self.read(
                record.space_id, record.local_id)
            if restored != record or restored_members != members:
                raise SpanStorageIntegrityError("Span 身份已绑定冲突详情")
            self._verified_by_ref[ref] = (restored, restored_members)
            return restored
        self._backend.insert(SPAN_TABLE, dict(record.__dict__))
        for member_ordinal, (start, end) in enumerate(members):
            self._backend.insert(SPAN_MEMBER_TABLE, {
                "space_id": record.space_id,
                "local_id": record.local_id,
                "member_ordinal": member_ordinal,
                "start": start,
                "end": end,
            })
        # backend 成功接收全部已严格校验的 append-only 行后，输入即是本次事务的核验结果。
        # 任一 insert 抛错时不会写缓存；外部迁移或故障注入仍须 clear 后从权威表完整回读。
        self._verified_by_ref[ref] = (record, members)
        return record

    def read(
            self, space_id: int, local_id: int
            ) -> tuple[SpanStorageRecord, tuple[tuple[int, int], ...]]:
        """按图节点回读唯一 Span 详情和全部有序成员。"""
        assert_int(space_id, local_id, _where="SpanStore.read")
        rows = self._backend.select(SPAN_TABLE, where={
            "space_id": space_id,
            "local_id": local_id,
        })
        if len(rows) != 1:
            raise SpanStorageIntegrityError("Span 没有唯一详情记录")
        record = SpanStorageRecord(**rows[0])
        _validate_record(record)
        member_rows = self._backend.select(SPAN_MEMBER_TABLE, where={
            "space_id": space_id,
            "local_id": local_id,
        })
        if len(member_rows) != record.member_count:
            raise SpanStorageIntegrityError("Span 成员数量缺失或重复")
        ordered = sorted(member_rows, key=lambda row: row["member_ordinal"])
        if [row["member_ordinal"] for row in ordered] != list(
                range(record.member_count)):
            raise SpanStorageIntegrityError("Span member ordinal 不连续")
        members = tuple((row["start"], row["end"]) for row in ordered)
        _validate_members(record, members)
        return record, members

    def span_count(self) -> int:
        """返回当前后端唯一 Span 详情数。"""
        return len(self._backend.select(SPAN_TABLE, where=None))

    def clear_runtime_caches(self) -> None:
        """外部 load、迁移或故障注入后清空已核验幂等缓存。"""
        self._verified_by_ref.clear()


__all__ = [
    "SPAN_MEMBER_TABLE",
    "SPAN_TABLE",
    "SpanMemberStorageRecord",
    "SpanStorageIntegrityError",
    "SpanStorageRecord",
    "SpanStore",
    "register_span_tables",
]
