"""PW-00A 正式装载、双阶段发布、恢复与 Core 保护专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.formal_post_weaning import (
    FormalPostWeaningLoadRequest,
    PW00A_START_PUBLISHED,
    PW00A_START_RESUMED,
)
from pure_integer_ai.cognition.shared.post_weaning import PostWeaningIntakeRequest
from pure_integer_ai.cognition.shared.types import WEANING_POST, WEANING_PRE
from pure_integer_ai.experiments.pw00a_formal_runtime import (
    PW00AFormalRuntime,
    PW00AFormalStartupError,
)
from pure_integer_ai.experiments.pw00a_formal_transaction import (
    PW00A_EVENT_ABORTED,
    PW00A_EVENT_PREPARED,
    PW00A_EVENT_PUBLISHED,
    PW00AFormalEventStore,
)
from pure_integer_ai.experiments.pw00a_inference_artifact import ARTIFACT_PATH
from pure_integer_ai.storage.backend import (
    CoreOwnerWriteProtectionError,
    DictBackend,
    SQLiteBackend,
)
from test_pw00_post_weaning_runtime import (
    _build_runtime,
    _Parser,
    _restore_runtime,
    _source,
)


ROOT = Path(__file__).resolve().parents[1]


def _file_key(path: Path) -> tuple[int, ...]:
    """返回测试装载请求使用的文件 SHA-256 字节键。"""
    return tuple(hashlib.sha256(path.read_bytes()).digest())


def _authority_reader(root, path):
    """测试只替代尚未发布的 authority 文件内容，不替代 artifact 摘要绑定。"""
    del root, path
    return {
        "status": "PW00A_FORMAL_LOAD_AUTHORITY_EVIDENCED",
        "readiness_transition": {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS_REPUBLISHED": 1,
            "PW00A_STARTED": 0,
        },
    }


def _authority_path(tmp_path: Path) -> Path:
    """形成一个只由注入 reader 解释的固定字节占位。"""
    path = tmp_path / "pw00a-authority-test.json"
    path.write_bytes(b"{}\n")
    return path


def _request(manifest, authority_path: Path, *, run_id: int, epoch: int):
    """从已核验 PW-00 设施清单形成正式装载请求。"""
    return FormalPostWeaningLoadRequest(
        run_id,
        epoch,
        manifest.runtime_owner,
        _file_key(authority_path),
        _file_key(ROOT / ARTIFACT_PATH),
        manifest.routes,
        manifest.probe,
        manifest.budget,
        (30101, run_id, epoch),
    )


def _start(ctx, request, authority_path: Path):
    """用真实 W09 推理 artifact 和测试 authority 启动。"""
    return PW00AFormalRuntime.start(
        ctx,
        request,
        repository_root=ROOT,
        authority_path=authority_path,
        authority_reader=_authority_reader,
    )


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_pw00a_fresh_publish_loads_state_and_protects_core(
        backend_type,
        tmp_path: Path,
        ) -> None:
    """双后端首次发布后只允许 Memory 增长，Core owner 写在物理层拒绝。"""
    backend, ctx, routes, manifest, _, _, _ = _build_runtime(backend_type())
    authority_path = _authority_path(tmp_path)
    try:
        runtime = _start(ctx, _request(
            manifest, authority_path, run_id=1, epoch=1), authority_path)

        assert ctx.weaning_phase == WEANING_POST
        assert runtime.startup_report.status == PW00A_START_PUBLISHED
        assert runtime.inference_adapter.state_sha256 == (
            "4df2369120d91512c503a6b0977adcf9b25ef78ade6605d88ce3232bed02ccc8")
        assert ctx.backend.owner_write_protection_state() == (
            ctx.core_space.space_id,)
        assert tuple(
            item.event_kind for item in PW00AFormalEventStore(backend).events(1)
        ) == (PW00A_EVENT_PREPARED, PW00A_EVENT_PUBLISHED)

        source = _source(301)
        run = runtime.run_intake(PostWeaningIntakeRequest(
            routes.reading,
            source,
            "正式阅读来源",
            "license-formal-read",
            401,
            parser=_Parser(source, 31),
            trace=(30120, 1),
        ))
        assert run.report.core_unchanged
        core_row = backend.select(
            "concept_node",
            where={"space_id": ctx.core_space.space_id},
            limit=1,
        )[0]
        with pytest.raises(CoreOwnerWriteProtectionError):
            backend.update(
                "concept_node",
                {
                    "space_id": ctx.core_space.space_id,
                    "local_id": core_row["local_id"],
                },
                {"version_head": core_row["version_head"]},
            )
    finally:
        backend.close()


def test_pw00a_sqlite_restart_resumes_published_state(tmp_path: Path) -> None:
    """SQLite 真重开后从 PUBLISHED 恢复 phase、推理状态、ID 水位和 Core 保护。"""
    database = tmp_path / "pw00a-restart.sqlite3"
    authority_path = _authority_path(tmp_path)
    first_backend = SQLiteBackend(str(database))
    first, projection_key, request = None, None, None
    try:
        _, ctx, _, manifest, _, _, projection = _build_runtime(first_backend)
        request = _request(manifest, authority_path, run_id=1, epoch=1)
        first = _start(ctx, request, authority_path)
        projection_key = projection.stable_key()
        assert first.startup_report.status == PW00A_START_PUBLISHED
    finally:
        first_backend.close()

    second_backend = SQLiteBackend(str(database))
    try:
        ctx, _, _, _, _ = _restore_runtime(second_backend, projection_key)
        resumed = _start(ctx, request, authority_path)

        assert resumed.startup_report.status == PW00A_START_RESUMED
        assert ctx.weaning_phase == WEANING_POST
        assert second_backend.owner_write_protection_state() == (
            ctx.core_space.space_id,)
        schema = second_backend.schema_snapshot()
        for space in (ctx.core_space, ctx.memory_read, ctx.memory_interact):
            direct_ids = []
            for table, meta in schema.items():
                if {"space_id", "local_id"}.issubset(meta["columns"]):
                    direct_ids.extend(
                        row["local_id"] for row in second_backend.select(
                            table, where={"space_id": space.space_id})
                    )
            assert second_backend.id_pool_floor(space.space_id) >= max(
                direct_ids, default=0)
        assert len(PW00AFormalEventStore(second_backend).events(1)) == 2
    finally:
        second_backend.close()


def test_pw00a_publish_failure_appends_aborted_and_allows_new_epoch(
        monkeypatch,
        tmp_path: Path,
        ) -> None:
    """PUBLISHED 故障恢复 phase/保护并封存 ABORTED，新 run 才能继续。"""
    backend, ctx, _, manifest, _, _, _ = _build_runtime(DictBackend())
    authority_path = _authority_path(tmp_path)
    first_request = _request(manifest, authority_path, run_id=1, epoch=1)
    original = PW00AFormalEventStore.published

    def fail_publish(self, **kwargs):
        """在 PREPARED 已提交后注入发布故障。"""
        del self, kwargs
        raise RuntimeError("PW00A publish injection")

    try:
        monkeypatch.setattr(PW00AFormalEventStore, "published", fail_publish)
        with pytest.raises(RuntimeError, match="injection"):
            _start(ctx, first_request, authority_path)
        assert ctx.weaning_phase == WEANING_PRE
        assert backend.owner_write_protection_state() == ()
        assert tuple(
            item.event_kind for item in PW00AFormalEventStore(backend).events(1)
        ) == (PW00A_EVENT_PREPARED, PW00A_EVENT_ABORTED)

        monkeypatch.setattr(PW00AFormalEventStore, "published", original)
        second = _start(
            ctx,
            _request(manifest, authority_path, run_id=2, epoch=2),
            authority_path,
        )
        assert second.startup_report.status == PW00A_START_PUBLISHED
        assert ctx.weaning_phase == WEANING_POST
    finally:
        backend.close()


def test_pw00a_prepared_only_requires_explicit_abort(tmp_path: Path) -> None:
    """crash 留下的 PREPARED 不可自动领养，显式 ABORTED 后才能换 epoch。"""
    backend, ctx, _, manifest, _, _, _ = _build_runtime(DictBackend())
    authority_path = _authority_path(tmp_path)
    store = PW00AFormalEventStore(backend)
    store.prepared(
        run_id=1,
        publish_epoch=1,
        manifest_sha256="1" * 64,
        payload={"owner_watermark_key": [1], "pw00a_started": 0},
    )
    try:
        with pytest.raises(PW00AFormalStartupError, match="prepared-only"):
            _start(
                ctx,
                _request(manifest, authority_path, run_id=2, epoch=2),
                authority_path,
            )
        PW00AFormalRuntime.abort_prepared(ctx, run_id=1, trace=(30130, 1))
        runtime = _start(
            ctx,
            _request(manifest, authority_path, run_id=2, epoch=2),
            authority_path,
        )
        assert runtime.startup_report.status == PW00A_START_PUBLISHED
        assert tuple(
            item.event_kind for item in store.events(1)
        ) == (PW00A_EVENT_PREPARED, PW00A_EVENT_ABORTED)
    finally:
        backend.close()
