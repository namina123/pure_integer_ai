"""M-01 SourceRecord、Companion assoc、精确切片和恢复链对抗。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    InputPayload,
    LANG_ZH,
    MODALITY_LANGUAGE,
    STAGE_POST_WEANING_READ,
    Segment,
    SpaceContext,
)
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.source_intake import (
    SourceIntake,
    SourceIntakeIntegrityError,
)
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.source_record import (
    SourceRecordIntegrityError,
    SourceRecordMetadata,
    SourceRecordRepository,
)
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import (
    CompanionIntegrityError,
    CompanionSpace,
)
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _backend(kind: str):
    """构造并 bootstrap 两类受支持 backend。"""
    backend = DictBackend() if kind == "dict" else SQLiteBackend(":memory:")
    bootstrap(backend)
    return backend


def _source(document_id: int, *, source_id: int = 71) -> SourceRef:
    """构造带 parser version 的稳定裸文本来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(9)),
    )


def _spaces(backend):
    """按固定身份顺序装配 Core、两层 Memory 与 Companion。"""
    registry = SpaceRegistry(backend)
    core = AbstractSpace.create(registry, "core")
    memory_read = MemorySpace.create(registry, "memory_read")
    memory_interact = MemorySpace.create(registry, "memory_interact")
    companion = CompanionSpace.create(registry, "companion")
    return core, memory_read, memory_interact, companion


def _intake(backend, companion):
    """复用 scoped identity registry 构造来源仓库和 intake。"""
    scoped = ScopedIdentityStore(backend)
    repository = SourceRecordRepository(
        backend, registry=scoped.registry)
    return SourceIntake(repository, companion), repository


@pytest.mark.parametrize("backend_kind", ["dict", "sqlite"])
def test_same_text_different_sources_preserve_provenance(backend_kind: str):
    """相同文本 hash 的不同 SourceRef 必须保留两条来源和两个 assoc。"""
    backend = _backend(backend_kind)
    _, _, _, companion = _spaces(backend)
    intake, repository = _intake(backend, companion)

    first = intake.ensure(_source(1), "同文", license_id="L-A", batch_id=3)
    second = intake.ensure(_source(2), "同文", license_id="L-A", batch_id=3)

    assert first.text_hash == second.text_hash
    assert first.source_hash != second.source_hash
    assert first.companion_assoc_id != second.companion_assoc_id
    assert repository.source_count() == 2
    backend.close()


def test_source_intake_is_idempotent_and_reads_exact_slice():
    """同来源重放不新增 assoc，并能回到精确码点区间。"""
    backend = _backend("dict")
    _, _, _, companion = _spaces(backend)
    intake, _ = _intake(backend, companion)
    source = _source(1)

    first = intake.ensure(source, "甲乙丙", license_id="L-B", batch_id=8)
    second = intake.ensure(source, "甲乙丙", license_id="L-B", batch_id=8)
    sliced = intake.read_slice(source, 1, 3)

    assert first == second
    assert len(companion.all_items()) == 1
    assert sliced.text == "乙丙"
    assert sliced.source_hash == first.source_hash
    with pytest.raises(SourceIntakeIntegrityError, match="漂移"):
        intake.ensure(source, "甲乙丙", license_id="L-C", batch_id=8)
    backend.close()


@pytest.mark.parametrize("backend_kind", ["dict", "sqlite"])
def test_companion_instances_allocate_interleaved_assoc_ids(backend_kind: str):
    """同一空间的多个实例交错写入时必须从持久层水位连续分配。"""
    backend = _backend(backend_kind)
    _, _, _, first = _spaces(backend)
    second = CompanionSpace.create(SpaceRegistry(backend), "companion")

    first_id = first.put_text("第一条")
    second_id = second.put_text("第二条")
    third_id = first.put_text("第三条")

    assert (first_id, second_id, third_id) == (1, 2, 3)
    assert first.assoc_identity(first_id).space == second.identity
    assert second.assoc_identity(second_id).assoc_id == second_id
    backend.close()


def test_companion_rejects_duplicate_space_registration_row():
    """运行时 space_id 出现重复注册行时不得任选一行继续读写。"""
    backend = _backend("dict")
    _, _, _, companion = _spaces(backend)
    row = backend.select("space", where={"space_id": companion.space_id})[0]
    backend.insert("space", row)

    with pytest.raises(CompanionIntegrityError, match="唯一空间注册行"):
        CompanionSpace(SpaceRegistry(backend), backend, companion.space_id)
    backend.close()


def test_source_metadata_rejects_partial_batch_and_companion_drift():
    """legacy metadata 不得只留 batch，完整绑定还必须核验 Companion 来源 kind。"""
    with pytest.raises(ValueError, match="单独声明 batch"):
        SourceRecordMetadata(batch_id=1)

    backend = _backend("dict")
    _, _, _, companion = _spaces(backend)
    intake, repository = _intake(backend, companion)
    source = _source(1)
    assoc_id = companion.put_text("来源", meta=source.source_kind + 1)
    repository.put_complete(
        source.stable_key(),
        "来源",
        metadata=SourceRecordMetadata(
            "L-meta",
            0,
            companion.identity.type_hash,
            companion.identity.name_hash,
            assoc_id,
        ),
    )

    with pytest.raises(SourceIntakeIntegrityError, match="来源内容"):
        intake.read_slice(source, 0, 2)
    backend.close()


def test_legacy_source_record_cannot_be_silently_upgraded():
    """append-only legacy 来源缺 metadata 时必须显式迁移，不能就地补写。"""
    backend = _backend("dict")
    _, _, _, companion = _spaces(backend)
    intake, repository = _intake(backend, companion)
    source = _source(1)
    repository.put(source.stable_key(), "旧来源")

    with pytest.raises(SourceRecordIntegrityError, match="legacy"):
        intake.ensure(source, "旧来源", license_id="L-D", batch_id=1)
    backend.close()


def test_post_weaning_observe_archives_source_before_graph_write():
    """训练后 Observe 先完成来源链，且 assoc_id 不作为图边端点。"""
    backend = _backend("dict")
    core, memory_read, memory_interact, companion = _spaces(backend)
    intake, repository = _intake(backend, companion)
    ctx = SpaceContext(
        core, memory_read, memory_interact, companion,
        stage=STAGE_POST_WEANING_READ, memory_active=True)
    source = _source(1)
    raw = InputPayload(
        segments=[Segment(
            seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
            domain=DOMAIN_TEXT, tokens=["甲"])],
        source=SOURCE_BARE_TEXT,
        stage=STAGE_POST_WEANING_READ,
        source_ref=source,
        raw_text="甲",
        source_license_id="L-E",
        source_batch_id=5,
    )

    ObservePipeline(ctx, source_intake=intake).observe(raw)

    record = repository.find(source.stable_key())
    assert record is not None and record.metadata_complete
    assoc_id = record.companion_assoc_id
    assert companion.read(assoc_id)["text"] == "甲"
    assert all(
        row["space_id_from"] != companion.space_id
        and row["space_id_to"] != companion.space_id
        for row in backend.select("edge"))
    backend.close()


def test_post_weaning_observe_missing_source_dependency_writes_nothing():
    """缺许可或 Companion intake 时，训练后 Observe 在任何节点写入前失败。"""
    backend = _backend("dict")
    core, memory_read, memory_interact, companion = _spaces(backend)
    ctx = SpaceContext(
        core, memory_read, memory_interact, companion,
        stage=STAGE_POST_WEANING_READ, memory_active=True)
    raw = InputPayload(
        segments=[Segment(
            seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
            domain=DOMAIN_TEXT, tokens=["甲"])],
        source=SOURCE_BARE_TEXT,
        stage=STAGE_POST_WEANING_READ,
        source_ref=_source(1),
        raw_text="甲",
    )
    before_nodes = backend.select("concept_node")

    with pytest.raises(RuntimeError, match="来源入口"):
        ObservePipeline(ctx).observe(raw)

    assert backend.select("concept_node") == before_nodes
    backend.close()


def test_post_weaning_route_dependency_precedes_source_intake():
    """Memory 路由设施缺失时，完整来源也不得先写入 Companion 或 SourceRecord。"""
    backend = _backend("dict")
    core, _, memory_interact, companion = _spaces(backend)
    intake, repository = _intake(backend, companion)
    ctx = SpaceContext(
        core, None, memory_interact, companion,
        stage=STAGE_POST_WEANING_READ, memory_active=True)
    raw = InputPayload(
        segments=[Segment(
            seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
            domain=DOMAIN_TEXT, tokens=["甲"])],
        source=SOURCE_BARE_TEXT,
        stage=STAGE_POST_WEANING_READ,
        source_ref=_source(1),
        raw_text="甲",
        source_license_id="L-route",
        source_batch_id=1,
    )

    with pytest.raises(RuntimeError, match="MemoryRead"):
        ObservePipeline(ctx, source_intake=intake).observe(raw)

    assert repository.source_count() == 0
    assert companion.all_items() == []
    assert backend.select("concept_node") == []
    backend.close()


def test_dump_load_restores_assoc_identity_and_watermark(tmp_path):
    """dump/load 后来源绑定可回读，下一 assoc 从既有最大值继续。"""
    first_backend = _backend("dict")
    core, memory_read, memory_interact, companion = _spaces(first_backend)
    intake, _ = _intake(first_backend, companion)
    source = _source(1)
    stored = intake.ensure(source, "恢复文本", license_id="L-F", batch_id=13)
    dump_run(
        first_backend, str(tmp_path), "source-run",
        spaces=[
            core.space_id,
            memory_read.space_id,
            memory_interact.space_id,
            companion.space_id,
        ],
        tables=DUMP_TABLES,
    )

    restored_backend = _backend("dict")
    _, _, _, restored_companion = _spaces(restored_backend)
    load_run(restored_backend, str(tmp_path), "source-run")
    restored_intake, restored_repository = _intake(
        restored_backend, restored_companion)

    restored = restored_repository.find(source.stable_key())
    assert restored == stored
    assert restored_intake.read_slice(source, 0, 2).text == "恢复"
    next_record = restored_intake.ensure(
        _source(2), "下一条", license_id="L-F", batch_id=14)
    assert next_record.companion_assoc_id == stored.companion_assoc_id + 1
    restored_backend.close()
    first_backend.close()
