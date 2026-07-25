"""持久化恢复包的版本、segment、manifest 与迁移协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.segment_dependency import (
    SegmentDependency as RecoveryDependency,
    canonical_dependencies,
)


RECOVERY_FORMAT_VERSION = 1

RECOVERY_SEGMENT_SPACE = 1
RECOVERY_SEGMENT_GLOBAL = 2
RECOVERY_SEGMENT_CURSOR = 3

FAULT_DUMP_BEFORE_SEGMENT = 1
FAULT_DUMP_AFTER_SEGMENT = 2
FAULT_DUMP_BEFORE_MANIFEST = 3
FAULT_DUMP_AFTER_MANIFEST = 4
FAULT_DUMP_BEFORE_PUBLISH = 5
FAULT_LOAD_AFTER_PREFLIGHT = 6
FAULT_LOAD_AFTER_TABLE = 7
FAULT_LOAD_BEFORE_COMMIT = 8


def strict_key(
        value: tuple[int, ...], *, label: str, empty: bool = False,
        ) -> tuple[int, ...]:
    """核验恢复协议中的完整稳定整数键。"""
    if not isinstance(value, tuple) or (not empty and not value):
        raise ValueError(f"{label} 必须是{'\u53ef\u7a7a' if empty else '\u975e\u7a7a'}整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise ValueError(f"{label} 必须使用严格整数")
    return value


def positive(value: int, *, label: str) -> int:
    """核验版本、epoch 和文件大小为正严格整数。"""
    assert_int(value, _where=label)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def nonnegative(value: int, *, label: str) -> int:
    """核验行数、水位和序号为非负严格整数。"""
    assert_int(value, _where=label)
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


@dataclass(frozen=True, order=True)
class RecoverySegment:
    """一个已封存物理 segment 的完整身份和读取边界。"""

    segment_key: tuple[int, ...]
    filename: str
    publish_epoch: int
    read_fence: int
    row_count: int
    size_bytes: int
    checksum_key: tuple[int, ...]
    table_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """核验 segment 键、单文件名、epoch、尺寸、校验和表计数。"""
        strict_key(self.segment_key, label="recovery segment_key")
        if (not isinstance(self.filename, str) or not self.filename
                or "/" in self.filename or "\\" in self.filename
                or self.filename in {".", ".."}):
            raise ValueError("recovery segment filename 必须是单文件名")
        positive(self.publish_epoch, label="recovery segment publish_epoch")
        nonnegative(self.read_fence, label="recovery segment read_fence")
        nonnegative(self.row_count, label="recovery segment row_count")
        nonnegative(self.size_bytes, label="recovery segment size_bytes")
        strict_key(self.checksum_key, label="recovery segment checksum_key")
        if not isinstance(self.table_counts, tuple):
            raise TypeError("recovery segment table_counts 必须是 tuple")
        normalized: list[tuple[str, int]] = []
        for table, count in self.table_counts:
            if not isinstance(table, str) or not table:
                raise ValueError("recovery segment table 必须是非空字符串")
            normalized.append((table, nonnegative(
                count, label="recovery segment table count")))
        normalized.sort()
        if len({table for table, _ in normalized}) != len(normalized):
            raise ValueError("recovery segment table_counts 不得重复表")
        object.__setattr__(self, "table_counts", tuple(normalized))

    def to_payload(self) -> dict[str, Any]:
        """转为 manifest 中的确定性 segment 记录。"""
        return {
            "segment_key": list(self.segment_key),
            "filename": self.filename,
            "publish_epoch": self.publish_epoch,
            "read_fence": self.read_fence,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "checksum_key": list(self.checksum_key),
            "table_counts": [[table, count]
                             for table, count in self.table_counts],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RecoverySegment":
        """从 manifest 严格恢复 segment 记录。"""
        if not isinstance(payload, dict):
            raise RecoveryIntegrityError("recovery segment 必须是 object")
        try:
            table_counts = tuple(
                (item[0], item[1]) for item in payload["table_counts"])
            return cls(
                tuple(payload["segment_key"]),
                payload["filename"],
                payload["publish_epoch"],
                payload["read_fence"],
                payload["row_count"],
                payload["size_bytes"],
                tuple(payload["checksum_key"]),
                table_counts,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RecoveryIntegrityError("recovery segment 字段非法") from exc


@dataclass(frozen=True, order=True)
class RecoveryTableState:
    """完整快照中一张表的逻辑行数和有序内容校验。"""

    table: str
    row_count: int
    checksum_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验表名、逻辑行数和完整校验键。"""
        if not isinstance(self.table, str) or not self.table:
            raise ValueError("recovery table state 表名必须非空")
        nonnegative(self.row_count, label="recovery table row_count")
        strict_key(self.checksum_key, label="recovery table checksum_key")

    def to_payload(self) -> dict[str, Any]:
        """转为 manifest 中的表状态记录。"""
        return {
            "table": self.table,
            "row_count": self.row_count,
            "checksum_key": list(self.checksum_key),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RecoveryTableState":
        """从 manifest 严格恢复表状态。"""
        if not isinstance(payload, dict):
            raise RecoveryIntegrityError("recovery table state 必须是 object")
        try:
            return cls(
                payload["table"],
                payload["row_count"],
                tuple(payload["checksum_key"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RecoveryIntegrityError("recovery table state 字段非法") from exc


@dataclass(frozen=True)
class RecoveryManifest:
    """一个已发布 run 的 schema、依赖、空间和 segment 真值源。"""

    run_id: str
    format_version: int
    publish_epoch: int
    version_key: tuple[int, ...]
    dependencies: tuple[RecoveryDependency, ...]
    space_ids: tuple[int, ...]
    table_order: tuple[str, ...]
    schema: dict[str, dict[str, Any]]
    schema_checksum_key: tuple[int, ...]
    table_states: tuple[RecoveryTableState, ...]
    segments: tuple[RecoverySegment, ...]

    def __post_init__(self) -> None:
        """核验 run manifest 的完整性、唯一性和 epoch 一致性。"""
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("recovery manifest run_id 必须非空")
        positive(self.format_version, label="recovery format_version")
        positive(self.publish_epoch, label="recovery publish_epoch")
        strict_key(self.version_key, label="recovery version_key", empty=True)
        dependencies = canonical_dependencies(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        if any(type(item) is not int or item <= 0 for item in self.space_ids):
            raise ValueError("recovery space_ids 必须是正严格整数")
        spaces = tuple(sorted(self.space_ids))
        if len(set(spaces)) != len(spaces):
            raise ValueError("recovery space_ids 不得重复")
        object.__setattr__(self, "space_ids", spaces)
        if (not isinstance(self.table_order, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.table_order)):
            raise TypeError("recovery table_order 必须是非空表名 tuple")
        if len(set(self.table_order)) != len(self.table_order):
            raise ValueError("recovery table_order 不得重复")
        if set(self.table_order) != set(self.schema):
            raise ValueError("recovery table_order 与 schema 表集不一致")
        strict_key(self.schema_checksum_key,
                   label="recovery schema_checksum_key")
        if (not isinstance(self.table_states, tuple)
                or any(not isinstance(item, RecoveryTableState)
                       for item in self.table_states)):
            raise TypeError("recovery table_states 必须是 RecoveryTableState tuple")
        table_states = tuple(sorted(self.table_states))
        if tuple(item.table for item in table_states) != tuple(sorted(self.table_order)):
            raise ValueError("recovery table_states 必须完整覆盖 table_order")
        object.__setattr__(self, "table_states", table_states)
        if (not isinstance(self.segments, tuple)
                or any(not isinstance(item, RecoverySegment)
                       for item in self.segments)):
            raise TypeError("recovery segments 必须是 RecoverySegment tuple")
        segments = tuple(sorted(self.segments, key=lambda item: item.segment_key))
        if len({item.segment_key for item in segments}) != len(segments):
            raise ValueError("recovery segment_key 不得重复")
        if len({item.filename for item in segments}) != len(segments):
            raise ValueError("recovery segment filename 不得重复")
        if any(item.publish_epoch != self.publish_epoch for item in segments):
            raise ValueError("recovery segment publish_epoch 交叉")
        space_segment_ids = tuple(sorted(
            item.segment_key[1] for item in segments
            if len(item.segment_key) == 2
            and item.segment_key[0] == RECOVERY_SEGMENT_SPACE))
        if space_segment_ids != spaces:
            raise ValueError("recovery space segment 与 space_ids 不一致")
        object.__setattr__(self, "segments", segments)

    def to_payload(self) -> dict[str, Any]:
        """转为可封印的 canonical manifest 对象。"""
        return {
            "run_id": self.run_id,
            "format_version": self.format_version,
            "publish_epoch": self.publish_epoch,
            "version_key": list(self.version_key),
            "dependencies": [item.to_payload() for item in self.dependencies],
            "space_ids": list(self.space_ids),
            "table_order": list(self.table_order),
            "schema": self.schema,
            "schema_checksum_key": list(self.schema_checksum_key),
            "table_states": [item.to_payload() for item in self.table_states],
            "segments": [item.to_payload() for item in self.segments],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RecoveryManifest":
        """从已迁移到当前格式的 payload 构造 manifest。"""
        if not isinstance(payload, dict):
            raise RecoveryIntegrityError("run manifest 必须是 object")
        try:
            schema = payload["schema"]
            if not isinstance(schema, dict):
                raise TypeError
            return cls(
                payload["run_id"],
                payload["format_version"],
                payload["publish_epoch"],
                tuple(payload["version_key"]),
                tuple(RecoveryDependency.from_payload(item)
                      for item in payload["dependencies"]),
                tuple(payload["space_ids"]),
                tuple(payload["table_order"]),
                schema,
                tuple(payload["schema_checksum_key"]),
                tuple(RecoveryTableState.from_payload(item)
                      for item in payload["table_states"]),
                tuple(RecoverySegment.from_payload(item)
                      for item in payload["segments"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RecoveryIntegrityError):
                raise
            raise RecoveryIntegrityError("run manifest 字段非法") from exc


@dataclass(frozen=True)
class RecoveryLoadResult:
    """一次恢复后的空间、游标、manifest 和实际写表。"""

    space_ids: tuple[int, ...]
    cursor_payload: dict[str, Any] | None
    manifest: RecoveryManifest
    loaded_tables: tuple[str, ...]


class RecoveryError(RuntimeError):
    """恢复包发布、加载或迁移的公共失败基类。"""


class RecoveryIntegrityError(RecoveryError):
    """manifest、segment、schema、依赖或校验和不一致。"""


class RecoveryConflictError(RecoveryError):
    """目标 run 已发布，或目标 backend 与恢复包内容漂移。"""


class RecoveryMigrationError(RecoveryError):
    """旧格式没有显式迁移路径，或迁移未到达当前版本。"""


@runtime_checkable
class RecoveryFaultInjector(Protocol):
    """可在发布和加载承重边界注入故障的最小协议。"""

    def hit(self, point: int, context: dict[str, Any]) -> None:
        """在指定故障点观测上下文，需模拟失败时直接抛异常。"""
        ...


MigrationFunction = Callable[[dict[str, Any]], dict[str, Any]]


class RecoveryMigrationRegistry:
    """按相邻格式版本管理显式、确定的 manifest 迁移。"""

    def __init__(self) -> None:
        """创建无全局可变状态的空迁移注册表。"""
        self._steps: dict[int, tuple[int, MigrationFunction]] = {}

    def register(
            self, from_version: int, to_version: int,
            migrate: MigrationFunction,
            ) -> None:
        """注册唯一的相邻版本迁移，拒绝跨级和静默覆盖。"""
        nonnegative(from_version, label="migration from_version")
        positive(to_version, label="migration to_version")
        if to_version != from_version + 1:
            raise ValueError("recovery migration 必须按相邻版本注册")
        if not callable(migrate):
            raise TypeError("recovery migration 必须可调用")
        previous = self._steps.get(from_version)
        candidate = (to_version, migrate)
        if previous is not None and previous != candidate:
            raise ValueError("同一 recovery 源版本不得重复定义")
        self._steps[from_version] = candidate

    def migrate(
            self, payload: dict[str, Any], *, target_version: int,
            ) -> dict[str, Any]:
        """依次执行显式迁移，缺任一段时 fail closed。"""
        positive(target_version, label="migration target_version")
        if not isinstance(payload, dict):
            raise RecoveryMigrationError("migration payload 必须是 object")
        current = payload
        visited: set[int] = set()
        while True:
            version = current.get("format_version")
            if type(version) is not int or version < 0:
                raise RecoveryMigrationError("manifest 缺少合法 format_version")
            if version == target_version:
                return current
            if version > target_version or version in visited:
                raise RecoveryMigrationError("manifest 迁移方向非法")
            visited.add(version)
            step = self._steps.get(version)
            if step is None:
                raise RecoveryMigrationError(
                    f"缺少 recovery format {version} 的迁移路径")
            to_version, migrate = step
            migrated = migrate(current)
            if not isinstance(migrated, dict):
                raise RecoveryMigrationError("manifest migration 必须返回 object")
            if migrated.get("format_version") != to_version:
                raise RecoveryMigrationError("manifest migration 未到达声明目标版本")
            current = migrated


def hit_fault(
        injector: RecoveryFaultInjector | None,
        point: int,
        context: dict[str, Any],
        ) -> None:
    """调用可选故障注入器，未注入时保持零副作用。"""
    if injector is None:
        return
    if not isinstance(injector, RecoveryFaultInjector):
        raise TypeError("fault injector 未实现 hit 协议")
    injector.hit(point, dict(context))


__all__ = [
    "FAULT_DUMP_AFTER_MANIFEST",
    "FAULT_DUMP_AFTER_SEGMENT",
    "FAULT_DUMP_BEFORE_MANIFEST",
    "FAULT_DUMP_BEFORE_PUBLISH",
    "FAULT_DUMP_BEFORE_SEGMENT",
    "FAULT_LOAD_AFTER_PREFLIGHT",
    "FAULT_LOAD_AFTER_TABLE",
    "FAULT_LOAD_BEFORE_COMMIT",
    "RECOVERY_FORMAT_VERSION",
    "RECOVERY_SEGMENT_CURSOR",
    "RECOVERY_SEGMENT_GLOBAL",
    "RECOVERY_SEGMENT_SPACE",
    "RecoveryConflictError",
    "RecoveryDependency",
    "RecoveryError",
    "RecoveryFaultInjector",
    "RecoveryIntegrityError",
    "RecoveryLoadResult",
    "RecoveryManifest",
    "RecoveryMigrationError",
    "RecoveryMigrationRegistry",
    "RecoverySegment",
    "RecoveryTableState",
    "hit_fault",
]
