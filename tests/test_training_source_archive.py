"""新来源正文从首次写入即归档于模型之外，Core 保留完整整数来源身份。"""
from contextlib import contextmanager

import pytest

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef, VersionBundle
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.incremental_relation_model import IncrementalRelationTrainingRuntime
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.run_conversation_training import dialogue_semantic_protocols
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.source_record import (
    SourceRecordIntegrityError, SourceRecordRepository, register_source_record_table,
)
from pure_integer_ai.storage.span import SpanStore


@contextmanager
def _training(core_path, archive_path, *, read_only=False):
    """两个独立 SQLite owner 显式共享 Core 的权威整数来源 registry。"""
    core = SQLiteBackend(str(core_path), read_only=read_only)
    archive = SQLiteBackend(str(archive_path), read_only=read_only)
    try:
        register_source_record_table(core)
        register_source_record_table(archive)
        context = make_train_context(core)
        sources = SourceRecordRepository(archive, registry=context.scoped_identity_store.registry)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(context, occurrence_protocol=occurrence,
            span_protocol=span, source_records=sources)
        yield core, archive, context, sources
    finally:
        archive.close()
        core.close()


def test_source_body_never_enters_core_and_cold_span_uses_explicit_archive(tmp_path):
    """真实 Span、来源完整键和文本 hash 均保留；Core 没有新增正文或空正文替身。"""
    core_path, archive_path = tmp_path / "model.sqlite3", tmp_path / "sources.sqlite3"
    source = SourceRef(92457, 1, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    text = "甲属于乙"
    with _training(core_path, archive_path) as (core, archive, context, _sources):
        source_hash = context.occurrence_index.ensure_source(source, text)
        span = context.span_index.ensure_ref(source=source, raw_text=text,
            scope=document_scope(source), members=((0, 1),))
        assert core.select("source_record", where=None) == []
        assert archive.select("source_record", where={"source_hash": source_hash})[0]["raw_text"] == text
        core.commit()
        archive.commit()
    before = core_path.read_bytes()
    with _training(core_path, archive_path, read_only=True) as (core, _archive, context, sources):
        record = context.span_index.read(span)
        assert record.source == source and record.raw_text == text and record.members == ((0, 1),)
        assert sources.read(source_hash).source_key == source.stable_key()
        # 普通生产恢复不能静默读取训练归档；训练视图须显式给出该来源 owner。
        core_sources = SourceRecordRepository(core, registry=context.scoped_identity_store.registry)
        spans = SpanStore(core)
        with pytest.raises(SourceRecordIntegrityError):
            TrainedRelationGraphRuntime._span_surface(span, spans, core_sources)
        training = object.__new__(IncrementalRelationTrainingRuntime)
        training._archived_sources = sources
        assert training._span_surface(span, spans, core_sources) == ("甲", source_hash, 0, 1, text)
    assert core_path.read_bytes() == before


def test_archive_keeps_source_content_integrity_after_cold_read(tmp_path):
    """正文被替换时继续执行原全文 hash/长度校验，不能只凭来源 hash 接受。"""
    core_path, archive_path = tmp_path / "model.sqlite3", tmp_path / "sources.sqlite3"
    source = SourceRef(92457, 2, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    with _training(core_path, archive_path) as (core, archive, context, sources):
        source_hash = context.occurrence_index.ensure_source(source, "甲属于乙")
        archive._conn.execute("UPDATE source_record SET raw_text='甲更正乙'")
        sources.clear_runtime_caches()
        with pytest.raises(SourceRecordIntegrityError):
            sources.read(source_hash)
        assert core.select("source_record", where=None) == []


def test_archive_cannot_replace_the_core_identity_registry(tmp_path):
    """另一个 registry 即使能够保存正文，也不能为当前 Core 定义来源身份。"""
    core_path, archive_path = tmp_path / "model.sqlite3", tmp_path / "sources.sqlite3"
    with _training(core_path, archive_path) as (_core, archive, context, _sources):
        _semantic, occurrence, span = dialogue_semantic_protocols()
        with pytest.raises(ValueError, match="registry"):
            install_language_graph_protocols(context, occurrence_protocol=occurrence,
                span_protocol=span, source_records=SourceRecordRepository(archive))


def test_archive_cannot_be_installed_only_on_half_the_language_boundary(tmp_path):
    """显式来源归档不能与已经存在的默认 occurrence 来源 owner 混装。"""
    core_path, archive_path = tmp_path / "model.sqlite3", tmp_path / "sources.sqlite3"
    with _training(core_path, archive_path) as (_core, _archive, context, sources):
        _semantic, _occurrence, span = dialogue_semantic_protocols()
        with pytest.raises(ValueError, match="同时安装 occurrence"):
            install_language_graph_protocols(context, span_protocol=span, source_records=sources)
