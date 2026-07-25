"""V-03 持久化、恢复、续训、clone、迁移和故障对抗探针。"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from pure_integer_ai.cognition.shared.edge_types import EDGE_REFERS_TO
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    ParserVersion,
    PrimitiveVersion,
    VersionBundle,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage import paths
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.recovery_protocol import (
    FAULT_DUMP_AFTER_SEGMENT,
    FAULT_DUMP_BEFORE_MANIFEST,
    FAULT_DUMP_BEFORE_PUBLISH,
    FAULT_LOAD_AFTER_TABLE,
    RECOVERY_FORMAT_VERSION,
    RECOVERY_SEGMENT_SPACE,
    RecoveryConflictError,
    RecoveryDependency,
    RecoveryIntegrityError,
    RecoveryMigrationRegistry,
)
from pure_integer_ai.training.cursor import (
    CursorState,
    cursor_state_from_payload,
    dump_run,
    load_run_package,
)


_VERSIONS = VersionBundle(
    CorpusVersion(11),
    ParserVersion(12),
    PrimitiveVersion(13),
    CurriculumVersion(14),
)
_DEPENDENCIES = (
    RecoveryDependency((501, 1), (7, 1), (9, 8, 7)),
)


class _RaiseAt:
    """在指定恢复故障点抛异常的测试注入器。"""

    def __init__(self, point: int) -> None:
        """记录目标故障点和命中上下文。"""
        self.point = point
        self.contexts: list[dict] = []

    def hit(self, point: int, context: dict) -> None:
        """命中目标点时中断发布或加载。"""
        if point != self.point:
            return
        self.contexts.append(context)
        raise RuntimeError(f"fault:{point}")


def _backend(kind: str, tmp_path, label: str):
    """构造 Dict、SQLite 内存或 SQLite 文件后端。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite_memory":
        return SQLiteBackend(":memory:")
    if kind == "sqlite_file":
        return SQLiteBackend(str(tmp_path / f"{label}.sqlite"))
    raise ValueError(f"未知 backend kind: {kind}")


def _seed(backend):
    """建立含 Companion、两层 Memory 和跨空间边的最小恢复状态。"""
    ctx = make_train_context(backend, companion=True)
    spaces = tuple(row["space_id"] for row in backend.select(
        "space", order_by="space_id"))
    core_id, companion_id, memory_read_id, _ = spaces
    core_local = backend.next_id(core_id)
    companion_local = backend.next_id(companion_id)
    ctx.node_store.put(core_id, core_local, node_type=1)
    ctx.node_store.put(companion_id, companion_local, node_type=1)
    ctx.edge_store.add(
        space_id_from=core_id,
        local_id_from=core_local,
        space_id_to=companion_id,
        local_id_to=companion_local,
        edge_type=EDGE_REFERS_TO,
        source=1,
    )
    ctx.memory_read.put(1, 991)
    assert ctx.memory_read.space_id == memory_read_id
    return ctx, spaces


def _dump(
        backend,
        tmp_path,
        run_id: str = "base",
        *,
        fault_injector=None,
        ) -> tuple[str, tuple[int, ...]]:
    """以全注册表、全空间和游标发布一个完整恢复包。"""
    run_dir = str(tmp_path / "runs")
    spaces = tuple(row["space_id"] for row in backend.select(
        "space", order_by="space_id"))
    dumped = dump_run(
        backend,
        run_dir,
        run_id,
        spaces=list(spaces),
        tables=None,
        require_all_spaces=True,
        versions=_VERSIONS,
        dependencies=_DEPENDENCIES,
        cursor_state=CursorState(
            base_run_id="seed",
            run_id=run_id,
            completed={1, 2},
            non_skippable={3},
        ),
        fault_injector=fault_injector,
    )
    assert dumped == list(spaces)
    return run_dir, spaces


def _manifest_payload(run_dir: str, run_id: str = "base") -> dict:
    """读取测试将要故意破坏的 run manifest。"""
    with open(paths.run_manifest_path(run_dir, run_id), encoding="utf-8") as handle:
        return json.load(handle)


def _rewrite_manifest(
        run_dir: str,
        payload: dict,
        run_id: str = "base",
        ) -> None:
    """以协议 canonical 形状重写 manifest 并同步封印。"""
    data = (json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("utf-8")
    with open(paths.run_manifest_path(run_dir, run_id), "wb") as handle:
        handle.write(data)
    with open(paths.run_manifest_seal_path(run_dir, run_id), "w",
              encoding="ascii") as handle:
        handle.write(hashlib.sha256(data).hexdigest() + "\n")


@pytest.mark.parametrize("kind", ("dict", "sqlite_memory", "sqlite_file"))
def test_fresh_dump_load_resume_clone_and_repeat_are_identical(
        kind: str, tmp_path,
        ):
    """fresh/dump/load/resume/clone 在三类后端上恢复同一状态，重放零倍增。"""
    source = _backend(kind, tmp_path, "source")
    target = None
    try:
        _, spaces = _seed(source)
        expected = source.snapshot()
        run_dir, _ = _dump(source, tmp_path)
        assert os.path.isfile(paths.run_manifest_path(run_dir, "base"))
        assert os.path.isfile(paths.run_manifest_seal_path(run_dir, "base"))
        assert os.path.isfile(paths.cursor_state_path(run_dir, "base"))

        target = _backend(kind, tmp_path, "target")
        make_train_context(target, companion=True)
        result = load_run_package(
            target,
            run_dir,
            "base",
            expected_versions=_VERSIONS,
            expected_dependencies=_DEPENDENCIES,
        )
        assert result.space_ids == spaces
        assert result.loaded_tables
        restored_cursor = cursor_state_from_payload(
            result.cursor_payload, fallback_run_id="base")
        assert restored_cursor == CursorState(
            base_run_id="seed",
            run_id="base",
            completed={1, 2},
            non_skippable={3},
        )
        assert target.snapshot() == expected
        assert target.count("edge") == 1
        assert target.next_id(spaces[0]) == 2

        before_repeat = target.recovery_state_snapshot()
        repeated = load_run_package(
            target,
            run_dir,
            "base",
            expected_versions=_VERSIONS,
            expected_dependencies=_DEPENDENCIES,
        )
        assert repeated.loaded_tables == ()
        after_repeat = target.recovery_state_snapshot()
        assert after_repeat["tables"] == before_repeat["tables"]
        assert after_repeat["isa_edge_gen"] == before_repeat["isa_edge_gen"]
        assert after_repeat["id_pool"] == {
            **before_repeat["id_pool"],
            spaces[0]: 2,
        }
    finally:
        source.close()
        if target is not None:
            target.close()


@pytest.mark.parametrize("point", (
    FAULT_DUMP_AFTER_SEGMENT,
    FAULT_DUMP_BEFORE_MANIFEST,
    FAULT_DUMP_BEFORE_PUBLISH,
))
def test_dump_fault_never_publishes_manifest(point: int, tmp_path):
    """segment、manifest 或发布前故障都不得留下可恢复权威点。"""
    backend = DictBackend()
    try:
        _seed(backend)
        injector = _RaiseAt(point)
        with pytest.raises(RuntimeError, match="fault"):
            _dump(backend, tmp_path, fault_injector=injector)
        run_dir = str(tmp_path / "runs")
        assert injector.contexts
        assert not os.path.isfile(paths.run_manifest_path(run_dir, "base"))
    finally:
        backend.close()


def test_load_fault_rolls_back_tables_and_id_watermarks(tmp_path):
    """加载中途失败必须恢复全表、id pool 和 IS_A 代次。"""
    source = DictBackend()
    target = DictBackend()
    try:
        _seed(source)
        run_dir, _ = _dump(source, tmp_path)
        make_train_context(target, companion=True)
        before = target.recovery_state_snapshot()
        with pytest.raises(RuntimeError, match="fault"):
            load_run_package(
                target,
                run_dir,
                "base",
                expected_versions=_VERSIONS,
                expected_dependencies=_DEPENDENCIES,
                fault_injector=_RaiseAt(FAULT_LOAD_AFTER_TABLE),
            )
        assert target.recovery_state_snapshot() == before
    finally:
        source.close()
        target.close()


def test_missing_companion_segment_and_segment_tamper_fail_closed(tmp_path):
    """Companion segment 缺失和任意 segment 字节漂移都不得进入 backend。"""
    source = DictBackend()
    try:
        _, spaces = _seed(source)
        run_dir, _ = _dump(source, tmp_path)
        companion_path = paths.space_dump_path(run_dir, "base", spaces[1])
        os.remove(companion_path)
        target = DictBackend()
        try:
            make_train_context(target, companion=True)
            before = target.recovery_state_snapshot()
            with pytest.raises(RecoveryIntegrityError, match="segment"):
                load_run_package(target, run_dir, "base")
            assert target.recovery_state_snapshot() == before
        finally:
            target.close()

        source_two = DictBackend()
        try:
            _seed(source_two)
            run_dir_two, _ = _dump(source_two, tmp_path, "tamper")
            path = paths.space_dump_path(run_dir_two, "tamper", spaces[0])
            with open(path, "ab") as handle:
                handle.write(b"{}\n")
            target_two = DictBackend()
            try:
                make_train_context(target_two, companion=True)
                with pytest.raises(RecoveryIntegrityError, match="\u6821\u9a8c"):
                    load_run_package(target_two, run_dir_two, "tamper")
            finally:
                target_two.close()
        finally:
            source_two.close()
    finally:
        source.close()


def test_half_manifest_wrong_version_and_dependency_fail_closed(tmp_path):
    """半写 manifest、错 Core 版本和依赖漂移都必须在写前拒绝。"""
    source = DictBackend()
    try:
        _seed(source)
        run_dir, _ = _dump(source, tmp_path)
        target = DictBackend()
        try:
            make_train_context(target, companion=True)
            with pytest.raises(RecoveryIntegrityError, match="version_key"):
                load_run_package(
                    target,
                    run_dir,
                    "base",
                    expected_versions=VersionBundle(
                        CorpusVersion(11),
                        ParserVersion(12),
                        PrimitiveVersion(99),
                        CurriculumVersion(14),
                    ),
                )
            with pytest.raises(RecoveryIntegrityError, match="dependencies"):
                load_run_package(
                    target,
                    run_dir,
                    "base",
                    expected_dependencies=(
                        RecoveryDependency((501, 1), (7, 2), (9, 8, 7)),
                    ),
                )
        finally:
            target.close()

        manifest_path = paths.run_manifest_path(run_dir, "base")
        with open(manifest_path, "rb") as handle:
            content = handle.read()
        with open(manifest_path, "wb") as handle:
            handle.write(content[:max(1, len(content) // 2)])
        target_half = DictBackend()
        try:
            make_train_context(target_half, companion=True)
            with pytest.raises(RecoveryIntegrityError, match="\u5c01\u5370"):
                load_run_package(target_half, run_dir, "base")
        finally:
            target_half.close()
    finally:
        source.close()


def test_epoch_cross_and_partial_destination_fail_closed(tmp_path):
    """segment epoch 交叉和目标表部分状态不得被静默合并。"""
    source = DictBackend()
    try:
        _, spaces = _seed(source)
        run_dir, _ = _dump(source, tmp_path)
        payload = _manifest_payload(run_dir)
        payload["segments"][0]["publish_epoch"] = 2
        _rewrite_manifest(run_dir, payload)
        target = DictBackend()
        try:
            make_train_context(target, companion=True)
            with pytest.raises(RecoveryIntegrityError):
                load_run_package(target, run_dir, "base")
        finally:
            target.close()

        source_two = DictBackend()
        target_two = DictBackend()
        try:
            _seed(source_two)
            run_dir_two, _ = _dump(source_two, tmp_path, "partial")
            ctx_two = make_train_context(target_two, companion=True)
            ctx_two.node_store.put(spaces[0], 99, node_type=1)
            before = target_two.recovery_state_snapshot()
            with pytest.raises(RecoveryConflictError, match="concept_node"):
                load_run_package(target_two, run_dir_two, "partial")
            assert target_two.recovery_state_snapshot() == before
        finally:
            source_two.close()
            target_two.close()
    finally:
        source.close()


def test_old_manifest_requires_explicit_migration(tmp_path):
    """旧格式无注册路径时拒绝，注册确定迁移后才可加载。"""
    source = DictBackend()
    try:
        _seed(source)
        run_dir, _ = _dump(source, tmp_path)
        payload = _manifest_payload(run_dir)
        payload["format_version"] = 0
        _rewrite_manifest(run_dir, payload)

        rejected = DictBackend()
        try:
            make_train_context(rejected, companion=True)
            with pytest.raises(RecoveryIntegrityError, match="format"):
                load_run_package(rejected, run_dir, "base")
        finally:
            rejected.close()

        registry = RecoveryMigrationRegistry()

        def migrate_zero(payload: dict) -> dict:
            """将测试旧 manifest 显式标记为当前格式。"""
            migrated = dict(payload)
            migrated["format_version"] = RECOVERY_FORMAT_VERSION
            return migrated

        registry.register(0, RECOVERY_FORMAT_VERSION, migrate_zero)
        migrated = DictBackend()
        try:
            make_train_context(migrated, companion=True)
            result = load_run_package(
                migrated,
                run_dir,
                "base",
                migrations=registry,
            )
            assert result.manifest.format_version == RECOVERY_FORMAT_VERSION
            assert migrated.count("edge") == 1
        finally:
            migrated.close()
    finally:
        source.close()


def test_manifest_cannot_drop_declared_companion_or_overwrite_run(tmp_path):
    """manifest 不得删除已声明 Companion，已发布 run 也不得覆盖。"""
    source = DictBackend()
    try:
        _, spaces = _seed(source)
        run_dir, _ = _dump(source, tmp_path)
        with pytest.raises(RecoveryConflictError, match="\u8986\u76d6"):
            _dump(source, tmp_path)

        payload = _manifest_payload(run_dir)
        payload["segments"] = [
            segment for segment in payload["segments"]
            if segment["segment_key"] != [RECOVERY_SEGMENT_SPACE, spaces[1]]
        ]
        _rewrite_manifest(run_dir, payload)
        target = DictBackend()
        try:
            make_train_context(target, companion=True)
            with pytest.raises(RecoveryIntegrityError):
                load_run_package(target, run_dir, "base")
        finally:
            target.close()
    finally:
        source.close()
