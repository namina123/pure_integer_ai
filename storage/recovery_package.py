"""恢复包的原子发布、封印核验、幂等加载和失败回滚。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from pure_integer_ai.storage import paths
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.recovery_protocol import (
    FAULT_DUMP_AFTER_MANIFEST,
    FAULT_DUMP_AFTER_SEGMENT,
    FAULT_DUMP_BEFORE_MANIFEST,
    FAULT_DUMP_BEFORE_PUBLISH,
    FAULT_DUMP_BEFORE_SEGMENT,
    FAULT_LOAD_AFTER_PREFLIGHT,
    FAULT_LOAD_AFTER_TABLE,
    FAULT_LOAD_BEFORE_COMMIT,
    RECOVERY_FORMAT_VERSION,
    RECOVERY_SEGMENT_CURSOR,
    RECOVERY_SEGMENT_GLOBAL,
    RECOVERY_SEGMENT_SPACE,
    RecoveryConflictError,
    RecoveryDependency,
    RecoveryFaultInjector,
    RecoveryIntegrityError,
    RecoveryLoadResult,
    RecoveryManifest,
    RecoveryMigrationRegistry,
    RecoverySegment,
    RecoveryTableState,
    hit_fault,
)


_SEGMENT_HEADER_KIND = "zero_ai_recovery_segment"
_SPACE_TABLE = "space"


@dataclass
class _OpenSegment:
    """发布期间一个正在写入的 segment 句柄与计数。"""

    segment_key: tuple[int, ...]
    filename: str
    handle: BinaryIO
    row_count: int
    table_counts: dict[str, int]


@dataclass(frozen=True)
class _TableSpan:
    """已验证 segment 中某表连续记录的字节范围。"""

    start: int
    end: int
    row_count: int


@dataclass(frozen=True)
class _SegmentIndex:
    """已封存 segment 的文件路径与按表偏移索引。"""

    segment: RecoverySegment
    path: str
    table_spans: dict[str, _TableSpan]


@dataclass(frozen=True)
class InspectedRecoveryPackage:
    """完成 manifest、schema、segment 和依赖预检后的恢复包。"""

    manifest: RecoveryManifest
    indexes: tuple[_SegmentIndex, ...]
    cursor_payload: dict[str, Any] | None


def _canonical_bytes(value: object) -> bytes:
    """生成无多余空白、键有序的 UTF-8 JSON 字节。"""
    return (json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("utf-8")


def _checksum_key(data: bytes) -> tuple[int, ...]:
    """返回 SHA-256 的完整字节整数键，不以截断 hash 代替身份。"""
    return tuple(hashlib.sha256(data).digest())


def _checksum_path(path: str) -> tuple[tuple[int, ...], int]:
    """流式计算文件完整校验键和字节大小。"""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return tuple(digest.digest()), size


def _write_bytes(path: str, data: bytes) -> None:
    """完整写入并 fsync 单文件，使封印对应已推送的字节。"""
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """严格序列化存储行，禁止把未知类型静默降级成字符串。"""
    if not isinstance(row, dict):
        raise TypeError("recovery row 必须是 dict")
    result: dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(key, str) or not key:
            raise TypeError("recovery row 列名必须是非空字符串")
        if key in {"_table", "_ordinal", "_kind"}:
            raise ValueError(f"存储表使用了恢复保留列: {key}")
        if value is None or type(value) in {int, str, bool}:
            result[key] = value
            continue
        raise TypeError(
            f"recovery row 不支持列 {key!r} 的值类型 {type(value).__name__}")
    return result


def _schema_payload(
        snapshot: dict[str, dict[str, Any]],
        table_order: tuple[str, ...],
        ) -> dict[str, dict[str, Any]]:
    """将 backend schema 快照规范化为可比较、可封印的 payload。"""
    result: dict[str, dict[str, Any]] = {}
    for table in table_order:
        try:
            meta = snapshot[table]
            columns = tuple(meta["columns"])
            col_types = tuple(tuple(item) for item in meta["col_types"])
            indexes = tuple(tuple(item) for item in meta.get("indexes", ()))
            recovery_key = tuple(meta.get("recovery_key", ()))
            discipline = meta["discipline"]
            core = meta["core"]
        except (KeyError, TypeError) as exc:
            raise RecoveryIntegrityError(f"后端 schema 元数据不完整: {table}") from exc
        if tuple(item[0] for item in col_types) != columns:
            raise RecoveryIntegrityError(f"后端 schema 列和类型漂移: {table}")
        if type(discipline) is not int or type(core) is not bool:
            raise RecoveryIntegrityError(f"后端 schema 纪律字段非法: {table}")
        if (len(set(recovery_key)) != len(recovery_key)
                or any(item not in columns for item in recovery_key)):
            raise RecoveryIntegrityError(
                f"后端 schema recovery_key 非法: {table}")
        result[table] = {
            "columns": list(columns),
            "col_types": [list(item) for item in col_types],
            "discipline": discipline,
            "core": core,
            "indexes": [list(item) for item in sorted(indexes)],
            "recovery_key": list(recovery_key),
        }
    return result


def _normalize_manifest_schema(
        schema: object,
        table_order: tuple[str, ...],
        ) -> dict[str, dict[str, Any]]:
    """核验 manifest schema 字段并返回 canonical 形式。"""
    if not isinstance(schema, dict) or set(schema) != set(table_order):
        raise RecoveryIntegrityError("manifest schema 与 table_order 不一致")
    snapshot: dict[str, dict[str, Any]] = {}
    for table in table_order:
        meta = schema.get(table)
        if not isinstance(meta, dict):
            raise RecoveryIntegrityError(f"manifest schema 表字段非法: {table}")
        try:
            columns = tuple(meta["columns"])
            col_types = tuple(tuple(item) for item in meta["col_types"])
            indexes = tuple(tuple(item) for item in meta["indexes"])
            recovery_key = tuple(meta.get("recovery_key", ()))
            discipline = meta["discipline"]
            core = meta["core"]
        except (KeyError, TypeError) as exc:
            raise RecoveryIntegrityError(f"manifest schema 字段不完整: {table}") from exc
        if (any(not isinstance(item, str) or not item for item in columns)
                or len(set(columns)) != len(columns)):
            raise RecoveryIntegrityError(f"manifest schema 列非法: {table}")
        if (tuple(item[0] for item in col_types) != columns
                or any(len(item) != 2 or not isinstance(item[1], str)
                       for item in col_types)):
            raise RecoveryIntegrityError(f"manifest schema 列类型非法: {table}")
        if (type(discipline) is not int or type(core) is not bool
                or any(any(not isinstance(column, str) or not column
                           for column in index) for index in indexes)):
            raise RecoveryIntegrityError(f"manifest schema 元数据非法: {table}")
        if (any(not isinstance(item, str) or not item
                for item in recovery_key)
                or len(set(recovery_key)) != len(recovery_key)
                or any(item not in columns for item in recovery_key)):
            raise RecoveryIntegrityError(
                f"manifest schema recovery_key 非法: {table}")
        snapshot[table] = {
            "columns": columns,
            "col_types": col_types,
            "discipline": discipline,
            "core": core,
            "indexes": indexes,
            "recovery_key": recovery_key,
        }
    return _schema_payload(snapshot, table_order)


def _table_scope(columns: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """从 schema 列协议判定表的权威空间归属，不按表名写死。"""
    if "space_id" in columns:
        return "primary", ("space_id",)
    endpoints = tuple(column for column in ("space_id_from", "space_id_to")
                      if column in columns)
    if endpoints:
        return "endpoints", endpoints
    return "global", ()


def _row_space_ids(
        row: dict[str, Any], columns: tuple[str, ...],
        ) -> tuple[int, ...]:
    """按表的权威空间列返回行应出现的 segment 集。"""
    kind, scope_columns = _table_scope(columns)
    if kind == "global":
        return ()
    values: list[int] = []
    for column in scope_columns:
        value = row.get(column)
        if type(value) is not int or value <= 0:
            raise RecoveryIntegrityError(
                f"空间化表行缺少合法 {column}")
        if value not in values:
            values.append(value)
    return tuple(values)


def _select_tables(
        backend: StorageBackend,
        tables: tuple[str, ...] | None,
        *, include_registered_tables: bool,
        ) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    """根据显式表集和可选注册表闭包冻结恢复 schema。"""
    schema_snapshot = backend.schema_snapshot()
    if tables is None:
        selected = tuple(sorted(schema_snapshot))
    else:
        if not isinstance(tables, tuple) or any(
                not isinstance(item, str) or not item for item in tables):
            raise TypeError("recovery tables 必须是表名 tuple 或 None")
        if len(set(tables)) != len(tables):
            raise ValueError("recovery tables 不得重复")
        selected_list = list(tables)
        if include_registered_tables:
            selected_list.extend(
                table for table in sorted(schema_snapshot)
                if table not in set(selected_list))
        selected = tuple(selected_list)
    missing = tuple(table for table in selected if table not in schema_snapshot)
    if missing:
        raise RecoveryIntegrityError(f"恢复表未在 backend 注册: {missing}")
    return selected, schema_snapshot


def _space_inventory(backend: StorageBackend) -> tuple[int, ...]:
    """从注册的 space 表读取当前物理空间清单。"""
    schema = backend.schema_snapshot()
    if _SPACE_TABLE not in schema:
        raise RecoveryIntegrityError("完整恢复需要已注册 space 表")
    result = []
    for row in backend.select(_SPACE_TABLE, where=None):
        value = row.get("space_id")
        if type(value) is not int or value <= 0:
            raise RecoveryIntegrityError("space 表包含非法 space_id")
        result.append(value)
    if len(set(result)) != len(result):
        raise RecoveryIntegrityError("space 表包含重复 space_id")
    return tuple(sorted(result))


def registered_space_ids(backend: StorageBackend) -> tuple[int, ...]:
    """返回 backend 当前已注册的全部空间，供完整 run dump 避免手工枚举。"""
    return _space_inventory(backend)


def _prepare_staging(run_dir: str, run_id: str) -> tuple[str, str]:
    """创建同盘 staging，拒绝覆盖已发布 manifest，允许 run 目录已存 metrics。"""
    os.makedirs(run_dir, exist_ok=True)
    final_dir = paths.run_dir_of(run_dir, run_id)
    staging_dir = paths.run_staging_dir(run_dir, run_id)
    published_manifest = os.path.join(final_dir, paths.RUN_MANIFEST_NAME)
    if os.path.isfile(published_manifest):
        raise RecoveryConflictError(f"拒绝覆盖已发布 run: {run_id}")
    if os.path.exists(final_dir) and not os.path.isdir(final_dir):
        raise RecoveryConflictError("run 发布路径被非目录对象占用")
    os.makedirs(final_dir, exist_ok=True)
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    elif os.path.exists(staging_dir):
        raise RecoveryConflictError("recovery staging 路径被非目录对象占用")
    os.makedirs(staging_dir)
    return staging_dir, final_dir


def _publish_staging(staging_dir: str, final_dir: str) -> None:
    """先发布不可见的 segment 和 seal，最后原子替换 manifest 可见性指针。"""
    manifest_name = paths.RUN_MANIFEST_NAME
    names = sorted(name for name in os.listdir(staging_dir)
                   if name != manifest_name)
    moved: list[str] = []
    manifest_published = False
    try:
        for name in names:
            source = os.path.join(staging_dir, name)
            target = os.path.join(final_dir, name)
            os.replace(source, target)
            moved.append(target)
        os.replace(
            os.path.join(staging_dir, manifest_name),
            os.path.join(final_dir, manifest_name),
        )
        manifest_published = True
        try:
            os.rmdir(staging_dir)
        except OSError:
            pass
    except BaseException:
        if not manifest_published:
            for target in moved:
                if os.path.isfile(target):
                    os.remove(target)
        raise


def _open_segment(
        staging_dir: str,
        segment_key: tuple[int, ...],
        filename: str,
        publish_epoch: int,
        fault_injector: RecoveryFaultInjector | None,
        ) -> _OpenSegment:
    """创建 segment 并先写入包含身份与 epoch 的协议头。"""
    hit_fault(fault_injector, FAULT_DUMP_BEFORE_SEGMENT, {
        "filename": filename,
        "segment_key": segment_key,
        "publish_epoch": publish_epoch,
    })
    path = os.path.join(staging_dir, filename)
    handle = open(path, "wb")
    header = {
        "_kind": _SEGMENT_HEADER_KIND,
        "format_version": RECOVERY_FORMAT_VERSION,
        "segment_key": list(segment_key),
        "publish_epoch": publish_epoch,
    }
    handle.write(_canonical_bytes(header))
    return _OpenSegment(segment_key, filename, handle, 0, {})


def _write_record(
        segment: _OpenSegment,
        table: str,
        ordinal: int,
        row: dict[str, Any],
        ) -> None:
    """向 segment 追加一条带原表序号的行，供跨空间去重和原序恢复。"""
    record = {"_table": table, "_ordinal": ordinal, **row}
    segment.handle.write(_canonical_bytes(record))
    segment.row_count += 1
    segment.table_counts[table] = segment.table_counts.get(table, 0) + 1


def _finalize_segment(
        staging_dir: str,
        segment: _OpenSegment,
        publish_epoch: int,
        fault_injector: RecoveryFaultInjector | None,
        ) -> RecoverySegment:
    """刷写、关闭并以完整校验键封存 segment。"""
    segment.handle.flush()
    os.fsync(segment.handle.fileno())
    segment.handle.close()
    path = os.path.join(staging_dir, segment.filename)
    checksum_key, size_bytes = _checksum_path(path)
    result = RecoverySegment(
        segment.segment_key,
        segment.filename,
        publish_epoch,
        segment.row_count,
        segment.row_count,
        size_bytes,
        checksum_key,
        tuple(segment.table_counts.items()),
    )
    hit_fault(fault_injector, FAULT_DUMP_AFTER_SEGMENT, {
        "filename": segment.filename,
        "segment_key": segment.segment_key,
        "row_count": segment.row_count,
    })
    return result


def _write_cursor_segment(
        staging_dir: str,
        payload: dict[str, Any],
        publish_epoch: int,
        fault_injector: RecoveryFaultInjector | None,
        ) -> RecoverySegment:
    """将训练游标作为同一恢复包的封存 companion segment 写入。"""
    if not isinstance(payload, dict):
        raise TypeError("cursor payload 必须是 dict")
    filename = paths.CURSOR_STATE_NAME
    segment_key = (RECOVERY_SEGMENT_CURSOR,)
    hit_fault(fault_injector, FAULT_DUMP_BEFORE_SEGMENT, {
        "filename": filename,
        "segment_key": segment_key,
        "publish_epoch": publish_epoch,
    })
    data = _canonical_bytes(payload)
    path = os.path.join(staging_dir, filename)
    _write_bytes(path, data)
    result = RecoverySegment(
        segment_key,
        filename,
        publish_epoch,
        1,
        1,
        len(data),
        _checksum_key(data),
        (),
    )
    hit_fault(fault_injector, FAULT_DUMP_AFTER_SEGMENT, {
        "filename": filename,
        "segment_key": segment_key,
        "row_count": 1,
    })
    return result


def publish_recovery_package(
        backend: StorageBackend,
        run_dir: str,
        run_id: str,
        *,
        spaces: tuple[int, ...],
        tables: tuple[str, ...] | None = None,
        include_registered_tables: bool = False,
        require_all_spaces: bool = False,
        version_key: tuple[int, ...] = (),
        dependencies: tuple[RecoveryDependency, ...] = (),
        publish_epoch: int = 1,
        cursor_payload: dict[str, Any] | None = None,
        row_selector: Callable[[str, dict[str, Any]], bool] | None = None,
        fault_injector: RecoveryFaultInjector | None = None,
        ) -> RecoveryManifest:
    """在 staging 完成全部 segment 与封印后，以单次目录替换发布 run。"""
    if (not isinstance(spaces, tuple)
            or any(type(item) is not int or item <= 0 for item in spaces)):
        raise TypeError("recovery spaces 必须是正严格整数 tuple")
    space_ids = tuple(sorted(spaces))
    if not space_ids or len(set(space_ids)) != len(space_ids):
        raise ValueError("recovery spaces 必须非空且不重复")
    if type(publish_epoch) is not int or publish_epoch <= 0:
        raise ValueError("recovery publish_epoch 必须是正严格整数")
    if require_all_spaces and _space_inventory(backend) != space_ids:
        raise RecoveryIntegrityError(
            "完整恢复包必须覆盖 backend 当前全部空间")
    table_order, schema_snapshot = _select_tables(
        backend, tables,
        include_registered_tables=include_registered_tables,
    )
    schema = _schema_payload(schema_snapshot, table_order)
    schema_checksum_key = _checksum_key(_canonical_bytes(schema))
    staging_dir, final_dir = _prepare_staging(run_dir, run_id)
    opened: list[_OpenSegment] = []
    try:
        by_space: dict[int, _OpenSegment] = {}
        for space_id in space_ids:
            filename = f"space_{space_id}.dump"
            segment = _open_segment(
                staging_dir,
                (RECOVERY_SEGMENT_SPACE, space_id),
                filename,
                publish_epoch,
                fault_injector,
            )
            opened.append(segment)
            by_space[space_id] = segment
        global_tables = tuple(
            table for table in table_order
            if _table_scope(tuple(schema[table]["columns"]))[0] == "global")
        global_segment = None
        if global_tables:
            global_segment = _open_segment(
                staging_dir,
                (RECOVERY_SEGMENT_GLOBAL,),
                "global_identity.dump",
                publish_epoch,
                fault_injector,
            )
            opened.append(global_segment)

        table_states: list[RecoveryTableState] = []
        selected_spaces = set(space_ids)
        for table in table_order:
            columns = tuple(schema[table]["columns"])
            scope_kind, _ = _table_scope(columns)
            logical_digest = hashlib.sha256()
            logical_count = 0
            for ordinal, raw_row in enumerate(backend.select(table, where=None)):
                row = _serialize_row(raw_row)
                if row_selector is not None:
                    selected = row_selector(table, dict(row))
                    if type(selected) is not bool:
                        raise TypeError("recovery row_selector 必须返回严格 bool")
                    if not selected:
                        continue
                row_spaces = _row_space_ids(row, columns)
                if scope_kind == "global":
                    if global_segment is None:
                        raise RecoveryIntegrityError("全局表缺少 global segment")
                    _write_record(global_segment, table, ordinal, row)
                else:
                    targets = tuple(
                        space_id for space_id in row_spaces
                        if space_id in selected_spaces)
                    if not targets:
                        if require_all_spaces:
                            raise RecoveryIntegrityError(
                                f"表 {table} 存在不属于已声明空间的行")
                        continue
                    for space_id in targets:
                        _write_record(by_space[space_id], table, ordinal, row)
                logical_record = {
                    "_ordinal": ordinal,
                    **row,
                }
                logical_digest.update(_canonical_bytes(logical_record))
                logical_count += 1
            table_states.append(RecoveryTableState(
                table,
                logical_count,
                tuple(logical_digest.digest()),
            ))

        segments = [
            _finalize_segment(
                staging_dir, segment, publish_epoch, fault_injector)
            for segment in opened
        ]
        opened.clear()
        if cursor_payload is not None:
            segments.append(_write_cursor_segment(
                staging_dir,
                cursor_payload,
                publish_epoch,
                fault_injector,
            ))
        manifest = RecoveryManifest(
            run_id,
            RECOVERY_FORMAT_VERSION,
            publish_epoch,
            tuple(version_key),
            tuple(dependencies),
            space_ids,
            table_order,
            schema,
            schema_checksum_key,
            tuple(table_states),
            tuple(segments),
        )
        hit_fault(fault_injector, FAULT_DUMP_BEFORE_MANIFEST, {
            "run_id": run_id,
            "publish_epoch": publish_epoch,
        })
        manifest_bytes = _canonical_bytes(manifest.to_payload())
        _write_bytes(
            os.path.join(staging_dir, paths.RUN_MANIFEST_NAME),
            manifest_bytes,
        )
        _write_bytes(
            os.path.join(staging_dir, paths.RUN_MANIFEST_SEAL_NAME),
            (hashlib.sha256(manifest_bytes).hexdigest() + "\n").encode("ascii"),
        )
        hit_fault(fault_injector, FAULT_DUMP_AFTER_MANIFEST, {
            "run_id": run_id,
            "publish_epoch": publish_epoch,
        })
        hit_fault(fault_injector, FAULT_DUMP_BEFORE_PUBLISH, {
            "run_id": run_id,
            "publish_epoch": publish_epoch,
        })
        _publish_staging(staging_dir, final_dir)
        return manifest
    except BaseException:
        for segment in opened:
            if not segment.handle.closed:
                segment.handle.close()
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
        raise


def _load_manifest_payload(
        run_dir: str,
        run_id: str,
        migrations: RecoveryMigrationRegistry | None,
        ) -> RecoveryManifest:
    """先核验独立封印，再解析并显式迁移 manifest。"""
    manifest_path = paths.run_manifest_path(run_dir, run_id)
    seal_path = paths.run_manifest_seal_path(run_dir, run_id)
    if not os.path.isfile(manifest_path) or not os.path.isfile(seal_path):
        raise RecoveryIntegrityError("run manifest 或独立封印缺失")
    with open(manifest_path, "rb") as handle:
        manifest_bytes = handle.read()
    with open(seal_path, "rb") as handle:
        try:
            expected_seal = handle.read().decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise RecoveryIntegrityError("run manifest 封印非法") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_seal:
        raise RecoveryIntegrityError("run manifest 封印不匹配")
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryIntegrityError("run manifest 不是完整 JSON") from exc
    if not isinstance(payload, dict):
        raise RecoveryIntegrityError("run manifest 根必须是 object")
    version = payload.get("format_version")
    if version != RECOVERY_FORMAT_VERSION:
        if migrations is None:
            raise RecoveryIntegrityError(
                f"不兼容 recovery format: {version}")
        payload = migrations.migrate(
            payload, target_version=RECOVERY_FORMAT_VERSION)
    manifest = RecoveryManifest.from_payload(payload)
    if manifest.run_id != run_id:
        raise RecoveryIntegrityError("manifest run_id 与路径不一致")
    normalized_schema = _normalize_manifest_schema(
        manifest.schema, manifest.table_order)
    if _checksum_key(_canonical_bytes(normalized_schema)) != manifest.schema_checksum_key:
        raise RecoveryIntegrityError("manifest schema 校验不匹配")
    if normalized_schema != manifest.schema:
        manifest = RecoveryManifest(
            manifest.run_id,
            manifest.format_version,
            manifest.publish_epoch,
            manifest.version_key,
            manifest.dependencies,
            manifest.space_ids,
            manifest.table_order,
            normalized_schema,
            manifest.schema_checksum_key,
            manifest.table_states,
            manifest.segments,
        )
    return manifest


def _validate_record(
        record: object,
        manifest: RecoveryManifest,
        segment: RecoverySegment,
        ) -> tuple[str, int, dict[str, Any]]:
    """核验单条 segment 记录的表、原序、列集和空间归属。"""
    if not isinstance(record, dict):
        raise RecoveryIntegrityError("segment 记录必须是 object")
    table = record.get("_table")
    ordinal = record.get("_ordinal")
    if table not in manifest.schema or type(ordinal) is not int or ordinal < 0:
        raise RecoveryIntegrityError("segment 记录表名或 ordinal 非法")
    row = {key: value for key, value in record.items()
           if key not in {"_table", "_ordinal"}}
    columns = tuple(manifest.schema[table]["columns"])
    if not set(row).issubset(columns):
        raise RecoveryIntegrityError(f"segment 记录含未注册列: {table}")
    row = _serialize_row(row)
    scope_kind, _ = _table_scope(columns)
    key_kind = segment.segment_key[0]
    if key_kind == RECOVERY_SEGMENT_GLOBAL:
        if scope_kind != "global":
            raise RecoveryIntegrityError("空间化表不得出现在 global segment")
    elif key_kind == RECOVERY_SEGMENT_SPACE:
        if scope_kind == "global" or len(segment.segment_key) != 2:
            raise RecoveryIntegrityError("全局表或非法键不得出现在 space segment")
        if segment.segment_key[1] not in _row_space_ids(row, columns):
            raise RecoveryIntegrityError("segment 记录与空间归属不一致")
    else:
        raise RecoveryIntegrityError("数据记录出现在非数据 segment")
    return table, ordinal, row


def _index_data_segment(
        run_root: str,
        manifest: RecoveryManifest,
        segment: RecoverySegment,
        ) -> _SegmentIndex:
    """单次扫描 segment，同时核验封印、头、物理计数并记录表偏移。"""
    path = os.path.join(run_root, segment.filename)
    if not os.path.isfile(path):
        raise RecoveryIntegrityError(f"recovery segment 缺失: {segment.filename}")
    checksum_key, size_bytes = _checksum_path(path)
    if checksum_key != segment.checksum_key or size_bytes != segment.size_bytes:
        raise RecoveryIntegrityError(f"recovery segment 校验不匹配: {segment.filename}")
    spans: dict[str, _TableSpan] = {}
    counts: dict[str, int] = {}
    with open(path, "rb") as handle:
        header_line = handle.readline()
        try:
            header = json.loads(header_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryIntegrityError("segment 头非法") from exc
        expected_header = {
            "_kind": _SEGMENT_HEADER_KIND,
            "format_version": RECOVERY_FORMAT_VERSION,
            "segment_key": list(segment.segment_key),
            "publish_epoch": segment.publish_epoch,
        }
        if header != expected_header:
            raise RecoveryIntegrityError("segment 头身份或 epoch 不匹配")
        previous_table: str | None = None
        span_start = handle.tell()
        span_count = 0
        row_count = 0
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                if previous_table is not None:
                    spans[previous_table] = _TableSpan(
                        span_start, start, span_count)
                break
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RecoveryIntegrityError("segment 记录不是完整 JSON") from exc
            table, _, _ = _validate_record(record, manifest, segment)
            if previous_table is None:
                previous_table = table
                span_start = start
                span_count = 0
            elif table != previous_table:
                if table in spans:
                    raise RecoveryIntegrityError("segment 同表记录必须连续")
                spans[previous_table] = _TableSpan(
                    span_start, start, span_count)
                previous_table = table
                span_start = start
                span_count = 0
            span_count += 1
            row_count += 1
            counts[table] = counts.get(table, 0) + 1
    if row_count != segment.row_count or segment.read_fence != row_count:
        raise RecoveryIntegrityError("segment row_count 或 read_fence 不匹配")
    if tuple(sorted(counts.items())) != segment.table_counts:
        raise RecoveryIntegrityError("segment table_counts 不匹配")
    return _SegmentIndex(segment, path, spans)


def _read_cursor_segment(
        run_root: str,
        segment: RecoverySegment,
        ) -> dict[str, Any]:
    """核验并读取与 run 同次发布的游标 segment。"""
    path = os.path.join(run_root, segment.filename)
    if not os.path.isfile(path):
        raise RecoveryIntegrityError("cursor segment 缺失")
    checksum_key, size_bytes = _checksum_path(path)
    if checksum_key != segment.checksum_key or size_bytes != segment.size_bytes:
        raise RecoveryIntegrityError("cursor segment 校验不匹配")
    if segment.row_count != 1 or segment.read_fence != 1:
        raise RecoveryIntegrityError("cursor segment read_fence 非法")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryIntegrityError("cursor segment 不是完整 JSON") from exc
    if not isinstance(payload, dict):
        raise RecoveryIntegrityError("cursor segment 根必须是 object")
    return payload


def inspect_recovery_package(
        run_dir: str,
        run_id: str,
        *,
        expected_version_key: tuple[int, ...] | None = None,
        expected_dependencies: tuple[RecoveryDependency, ...] | None = None,
        expected_publish_epoch: int | None = None,
        migrations: RecoveryMigrationRegistry | None = None,
        ) -> InspectedRecoveryPackage:
    """在不修改 backend 的前提下完成 manifest、依赖、epoch 和全部 segment 预检。"""
    manifest = _load_manifest_payload(run_dir, run_id, migrations)
    if (expected_version_key is not None
            and manifest.version_key != tuple(expected_version_key)):
        raise RecoveryIntegrityError("recovery version_key 不兼容")
    if (expected_dependencies is not None
            and manifest.dependencies != tuple(sorted(expected_dependencies))):
        raise RecoveryIntegrityError("recovery dependencies 不兼容")
    if (expected_publish_epoch is not None
            and manifest.publish_epoch != expected_publish_epoch):
        raise RecoveryIntegrityError("recovery publish_epoch 不兼容")
    run_root = paths.run_dir_of(run_dir, run_id)
    indexes: list[_SegmentIndex] = []
    cursor_payload = None
    cursor_seen = False
    global_seen = False
    for segment in manifest.segments:
        kind = segment.segment_key[0]
        if kind == RECOVERY_SEGMENT_CURSOR:
            if cursor_seen or len(segment.segment_key) != 1:
                raise RecoveryIntegrityError("cursor segment 重复或键非法")
            cursor_seen = True
            cursor_payload = _read_cursor_segment(run_root, segment)
            continue
        if kind == RECOVERY_SEGMENT_GLOBAL:
            if global_seen or len(segment.segment_key) != 1:
                raise RecoveryIntegrityError("global segment 重复或键非法")
            global_seen = True
        elif kind != RECOVERY_SEGMENT_SPACE or len(segment.segment_key) != 2:
            raise RecoveryIntegrityError("manifest 包含未注册 segment 类型")
        indexes.append(_index_data_segment(run_root, manifest, segment))
    requires_global = any(
        _table_scope(tuple(manifest.schema[table]["columns"]))[0] == "global"
        for table in manifest.table_order)
    if requires_global != global_seen:
        raise RecoveryIntegrityError("global segment 与 schema 全局表集不一致")
    return InspectedRecoveryPackage(
        manifest,
        tuple(sorted(indexes, key=lambda item: item.segment.segment_key)),
        cursor_payload,
    )


def _read_table_rows(
        package: InspectedRecoveryPackage,
        table: str,
        ) -> list[dict[str, Any]]:
    """按原表 ordinal 合并各 segment 记录，只对跨空间物理副本去重。"""
    by_ordinal: dict[int, dict[str, Any]] = {}
    for index in package.indexes:
        span = index.table_spans.get(table)
        if span is None:
            continue
        with open(index.path, "rb") as handle:
            handle.seek(span.start)
            seen = 0
            while handle.tell() < span.end:
                line = handle.readline()
                if not line:
                    raise RecoveryIntegrityError("segment 表范围提前结束")
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RecoveryIntegrityError("segment 表记录非法") from exc
                actual_table, ordinal, row = _validate_record(
                    record, package.manifest, index.segment)
                if actual_table != table:
                    raise RecoveryIntegrityError("segment 表偏移索引漂移")
                previous = by_ordinal.get(ordinal)
                if previous is not None and previous != row:
                    raise RecoveryIntegrityError("同一原表 ordinal 的跨空间副本漂移")
                by_ordinal[ordinal] = row
                seen += 1
            if seen != span.row_count:
                raise RecoveryIntegrityError("segment 表偏移行数不匹配")
    ordered = sorted(by_ordinal.items())
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for ordinal, row in ordered:
        digest.update(_canonical_bytes({"_ordinal": ordinal, **row}))
        rows.append(row)
    state_map = {item.table: item for item in package.manifest.table_states}
    expected = state_map[table]
    if len(rows) != expected.row_count or tuple(digest.digest()) != expected.checksum_key:
        raise RecoveryIntegrityError(f"表 {table} 逻辑行数或校验不匹配")
    return rows


def _target_schema(
        backend: StorageBackend,
        manifest: RecoveryManifest,
        ) -> None:
    """比较目标 backend 已注册 schema，未显式迁移的差异一律拒绝。"""
    target_snapshot = backend.schema_snapshot()
    missing = tuple(table for table in manifest.table_order
                    if table not in target_snapshot)
    if missing:
        raise RecoveryIntegrityError(f"目标 backend 缺少恢复表: {missing}")
    target = _schema_payload(target_snapshot, manifest.table_order)
    if target != manifest.schema:
        raise RecoveryIntegrityError("目标 backend schema 与恢复包不兼容")


def load_recovery_package(
        backend: StorageBackend,
        run_dir: str,
        run_id: str,
        *,
        expected_version_key: tuple[int, ...] | None = None,
        expected_dependencies: tuple[RecoveryDependency, ...] | None = None,
        expected_publish_epoch: int | None = None,
        migrations: RecoveryMigrationRegistry | None = None,
        fault_injector: RecoveryFaultInjector | None = None,
        ) -> RecoveryLoadResult:
    """先全量预检，再幂等加载；任一写入失败恢复 backend 数据与水位。"""
    package = inspect_recovery_package(
        run_dir,
        run_id,
        expected_version_key=expected_version_key,
        expected_dependencies=expected_dependencies,
        expected_publish_epoch=expected_publish_epoch,
        migrations=migrations,
    )
    _target_schema(backend, package.manifest)
    tables_to_load: list[str] = []
    for table in package.manifest.table_order:
        rows = _read_table_rows(package, table)
        existing = backend.select(table, where=None)
        if existing == rows:
            continue
        if existing:
            raise RecoveryConflictError(
                f"目标表 {table} 非空且与恢复包漂移")
        tables_to_load.append(table)
    hit_fault(fault_injector, FAULT_LOAD_AFTER_PREFLIGHT, {
        "run_id": run_id,
        "publish_epoch": package.manifest.publish_epoch,
        "table_count": len(tables_to_load),
    })
    state = backend.recovery_state_snapshot()
    try:
        floor_by_space: dict[int, int] = {}
        for table in package.manifest.table_order:
            rows = _read_table_rows(package, table)
            if table in tables_to_load:
                for row in rows:
                    backend.insert(table, row)
                hit_fault(fault_injector, FAULT_LOAD_AFTER_TABLE, {
                    "run_id": run_id,
                    "table": table,
                    "row_count": len(rows),
                })
            for row in rows:
                space_id = row.get("space_id")
                local_id = row.get("local_id")
                if type(space_id) is int and type(local_id) is int:
                    floor_by_space[space_id] = max(
                        local_id, floor_by_space.get(space_id, 0))
        for space_id, floor in sorted(floor_by_space.items()):
            backend.advance_id_pool(space_id, floor)
        hit_fault(fault_injector, FAULT_LOAD_BEFORE_COMMIT, {
            "run_id": run_id,
            "publish_epoch": package.manifest.publish_epoch,
        })
        backend.commit()
    except BaseException:
        backend.restore_recovery_state(state)
        backend.commit()
        raise
    return RecoveryLoadResult(
        package.manifest.space_ids,
        package.cursor_payload,
        package.manifest,
        tuple(tables_to_load),
    )


def _recovery_row_key(
        table: str,
        row: dict[str, Any],
        columns: tuple[str, ...],
        ) -> tuple[Any, ...]:
    """按 manifest 声明的完整恢复键提取一行身份。"""
    if not columns:
        raise RecoveryIntegrityError(
            f"表 {table} 未声明 recovery_key，禁止增量合并")
    try:
        return tuple(row[column] for column in columns)
    except KeyError as exc:
        raise RecoveryIntegrityError(
            f"表 {table} 的 recovery_key 缺少列 {exc.args[0]}") from exc


def merge_recovery_package(
        backend: StorageBackend,
        run_dir: str,
        run_id: str,
        *,
        expected_version_key: tuple[int, ...] | None = None,
        expected_dependencies: tuple[RecoveryDependency, ...] | None = None,
        expected_publish_epoch: int | None = None,
        migrations: RecoveryMigrationRegistry | None = None,
        fault_injector: RecoveryFaultInjector | None = None,
        ) -> RecoveryLoadResult:
    """按每表完整恢复键把增量包合并到既有依赖状态，同键漂移立即拒绝。"""
    package = inspect_recovery_package(
        run_dir,
        run_id,
        expected_version_key=expected_version_key,
        expected_dependencies=expected_dependencies,
        expected_publish_epoch=expected_publish_epoch,
        migrations=migrations,
    )
    _target_schema(backend, package.manifest)
    missing_by_table: dict[str, tuple[dict[str, Any], ...]] = {}
    for table in package.manifest.table_order:
        rows = _read_table_rows(package, table)
        recovery_key = tuple(
            package.manifest.schema[table].get("recovery_key", ()))
        incoming: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            key = _recovery_row_key(table, row, recovery_key)
            previous = incoming.get(key)
            if previous is not None:
                if previous != row:
                    raise RecoveryIntegrityError(
                        f"恢复包表 {table} 同键含不同记录")
                raise RecoveryIntegrityError(
                    f"恢复包表 {table} 同键含重复记录")
            incoming[key] = row
        existing: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in backend.select(table, where=None):
            key = _recovery_row_key(table, row, recovery_key)
            previous = existing.get(key)
            if previous is not None:
                raise RecoveryConflictError(
                    f"目标表 {table} 已存在重复 recovery_key")
            existing[key] = row
        missing: list[dict[str, Any]] = []
        for key, row in incoming.items():
            previous = existing.get(key)
            if previous is None:
                missing.append(row)
                continue
            if previous != row:
                raise RecoveryConflictError(
                    f"目标表 {table} 同 recovery_key 内容漂移")
        missing_by_table[table] = tuple(missing)
    hit_fault(fault_injector, FAULT_LOAD_AFTER_PREFLIGHT, {
        "run_id": run_id,
        "publish_epoch": package.manifest.publish_epoch,
        "table_count": len(missing_by_table),
    })
    state = backend.recovery_state_snapshot()
    loaded_tables: list[str] = []
    try:
        floor_by_space: dict[int, int] = {}
        for table in package.manifest.table_order:
            missing = missing_by_table[table]
            for row in missing:
                backend.insert(table, row)
            if missing:
                loaded_tables.append(table)
            hit_fault(fault_injector, FAULT_LOAD_AFTER_TABLE, {
                "run_id": run_id,
                "table": table,
                "row_count": len(missing),
            })
            for row in _read_table_rows(package, table):
                space_id = row.get("space_id")
                local_id = row.get("local_id")
                if type(space_id) is int and type(local_id) is int:
                    floor_by_space[space_id] = max(
                        local_id, floor_by_space.get(space_id, 0))
        for space_id, floor in sorted(floor_by_space.items()):
            backend.advance_id_pool(space_id, floor)
        hit_fault(fault_injector, FAULT_LOAD_BEFORE_COMMIT, {
            "run_id": run_id,
            "publish_epoch": package.manifest.publish_epoch,
        })
        backend.commit()
    except BaseException:
        backend.restore_recovery_state(state)
        backend.commit()
        raise
    return RecoveryLoadResult(
        package.manifest.space_ids,
        package.cursor_payload,
        package.manifest,
        tuple(loaded_tables),
    )


__all__ = [
    "InspectedRecoveryPackage",
    "inspect_recovery_package",
    "load_recovery_package",
    "merge_recovery_package",
    "publish_recovery_package",
    "registered_space_ids",
]
