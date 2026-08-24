"""P3-B active roster 与 P2 双向回读的 test-transport 专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_active_caller_gate as caller_gate_module
import pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback as readback_module
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion, CurriculumVersion, OwnerScope, ParserVersion,
    PrimitiveVersion, SourceRef, VersionBundle, VISIBILITY_USER,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeBudget,
    ConversationHeldOutV4ActiveIntakeInput,
    ConversationHeldOutV4ActiveIntakeMapping,
    ConversationHeldOutV4ActiveIntakeObservationBinding,
    ConversationHeldOutV4ActiveIntakeRosterFactories,
    ConversationHeldOutV4ActiveIntakeSourceBinding,
    ConversationHeldOutV4ActiveIntakeTeacherPolicy,
    materialize_v4_active_intake,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
    ConversationHeldOutV4ActiveRosterReadbackBudget,
    ConversationHeldOutV4ActiveRosterReadbackError,
    ConversationHeldOutV4ActiveRosterReadbackInput,
    V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY,
    materialize_v4_active_roster_bidirectional_readback,
    revalidate_v4_active_roster_bidirectional_readback,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_caller_gate import (
    V4_ACTIVE_CALLER_GATE_CALLER_KIND,
    V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY,
    ConversationHeldOutV4ActiveCallerGateInput,
    run_v4_active_caller_zero_write_dry_run,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4LineageNode, ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_SIDE_HELD_OUT, V4_PROVENANCE_SIDE_TRAINING,
    V4_PROVENANCE_UPSTREAM_SHA256,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceCatalogBudget,
    ConversationHeldOutV4ProvenanceCatalogInput,
    build_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag import (
    ConversationHeldOutV4ProvenanceDagBudget,
    build_v4_provenance_direct_dag,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_closure import (
    ConversationHeldOutV4ProvenanceClosureBudget,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_merge import (
    ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
    advance_v4_provenance_cross_side_merge,
    publish_v4_provenance_cross_side_merge,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_projection import (
    ConversationHeldOutV4ProvenanceProjectionBudget,
    ConversationHeldOutV4ProvenanceProjectionResult,
    ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
    V4_PROVENANCE_PROJECTION_STREAM_CONTENT,
    V4_PROVENANCE_PROJECTION_STREAM_LINEAGE,
    V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF,
    decode_v4_provenance_content_projection_record,
    decode_v4_provenance_lineage_projection_record,
    decode_v4_provenance_source_ref_projection_record,
    encode_v4_provenance_content_projection_record,
    encode_v4_provenance_lineage_projection_record,
    encode_v4_provenance_source_ref_projection_record,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import CanonicalJsonObject, StableRecordKey
from pure_integer_ai.experiments.ph2_dataset_records import ObservationRecord, SourceRefRecord
from pure_integer_ai.storage.integer_codec import (
    IntegerFramedStreamReader, IntegerFramedStreamWriter, pack_key,
)
from pure_integer_ai.storage.integer_external_sort import IntegerExternalSortBudget
from pure_integer_ai.storage.k_run_boundary import (
    KRunFileDigest, create_new_run_root, ensure_normal_relative_directory,
    open_exclusive_binary, open_plain_binary, write_exclusive_bytes,
)


def _key(value: int) -> StableRecordKey:
    """生成完整 PH2 record identity。"""
    return StableRecordKey((value, value + 10_000))


def _protocol(value: int) -> ProtocolKey:
    """生成完整、可排序的协议 identity。"""
    return ProtocolKey((value, value + 20_000))


def _sha256(value: bytes) -> tuple[int, ...]:
    """生成实际消费内容的完整 SHA-256 tuple。"""
    return tuple(hashlib.sha256(value).digest())


def _payload(value: dict) -> CanonicalJsonObject:
    """构造 test-only typed PH2 payload。"""
    return CanonicalJsonObject.from_value(value)


def _source_ref(value: int) -> SourceRef:
    """构造不依赖 record key 的完整 P1 来源本体。"""
    return SourceRef(
        100 + value, 200 + value, value,
        OwnerScope(9, 8, 0, VISIBILITY_USER),
        VersionBundle(CorpusVersion(10 + value), ParserVersion(20 + value),
                      PrimitiveVersion(30 + value), CurriculumVersion(40 + value)),
    )


def _source(value: int, cluster: int) -> SourceRefRecord:
    """构造有独立快照、cluster 与 local content identity 的公开 source record。"""
    body = f"p3b-source-{value}".encode("ascii")
    upstream = hashlib.sha256(f"upstream-{value}".encode("ascii")).hexdigest()
    return SourceRefRecord(
        1, 1, 1, _key(1), _key(2), _key(value), "P3B_TEST_SOURCE",
        f"snapshot-{value}", f"revision-{value}",
        f"https://example.invalid/p3b/{value}", f"p3b/{value}",
        f"sha256:{upstream}", hashlib.sha256(body).hexdigest(), "CC0-1.0",
        "PUBLIC", "P3-B test source", 1,
        _payload({"document": value, "start": 0, "end": 1}), 0, _key(cluster),
    )


def _snapshot(source: SourceRefRecord, source_ref: SourceRef) -> ConversationHeldOutV4SnapshotIdentity:
    """按 SourceRefRecord 的冻结公开字段建立 P1 snapshot。"""
    _algorithm, upstream = source.upstream_checksum.split(":", 1)
    return ConversationHeldOutV4SnapshotIdentity(
        source_ref, tuple(map(ord, source.official_url)),
        tuple(map(ord, source.revision_id)), tuple(map(ord, source.snapshot_id)),
        V4_PROVENANCE_UPSTREAM_SHA256, tuple(bytes.fromhex(upstream)),
        tuple(bytes.fromhex(source.local_sha256)), _protocol(10), _protocol(20),
        _protocol(30),
    )


def _observation(value: int, source: SourceRefRecord, split: str) -> ObservationRecord:
    """构造仅用于 roster binding 的公开 Observation。"""
    return ObservationRecord(
        1, 1, 1, _key(1), _key(2), _key(value), "W-01", "p3b-readback",
        split, "zh", "typed-proposition", source.stable_key, "CC0-1.0",
        _key(31), _key(32), _key(33), _key(34), "forming", "support", "Proposition",
        _payload({"test": value}), "NONE", None, (), 1,
    )


def _sort_budget() -> IntegerExternalSortBudget:
    """覆盖小型真实外排的固定资源上限。"""
    return IntegerExternalSortBudget(
        max_input_file_count=16, max_input_physical_bytes=2 * 1024 * 1024,
        max_input_record_count=128, max_input_payload_bytes=2 * 1024 * 1024,
        max_record_payload_bytes=64 * 1024, max_batch_record_count=2,
        max_batch_payload_bytes=128 * 1024, max_batch_sort_key_bytes=128 * 1024,
        max_temporary_run_count=128, max_temporary_record_count=512,
        max_temporary_payload_bytes=4 * 1024 * 1024,
        max_temporary_physical_bytes=8 * 1024 * 1024,
        max_output_physical_bytes=2 * 1024 * 1024, merge_fan_in=2,
        max_open_files=3, max_merge_pass_count=16,
    )


def _active_budget() -> ConversationHeldOutV4ActiveIntakeBudget:
    """容纳两侧 source/Observation roster 的 P3-A 小型预算。"""
    return ConversationHeldOutV4ActiveIntakeBudget(
        16, 16, 16, 16, 32 * 1024, 128 * 1024, 256 * 1024, 64 * 1024,
        ConversationHeldOutV4ProvenanceCatalogBudget(
            4, 256 * 1024, 512 * 1024, 32 * 1024, 64, 128 * 1024),
        _sort_budget(),
    )


def _p3b_budget() -> ConversationHeldV4ActiveRosterReadbackBudget:
    """构造 P3-B raw、sort、receipt 与 manifest 的有限预算。"""
    return ConversationHeldV4ActiveRosterReadbackBudget(
        64, 64, 32 * 1024, 32 * 1024, 256 * 1024, 512 * 1024,
        128 * 1024, 256 * 1024, 128 * 1024, _sort_budget())


# 减少测试 fixture 中的长类型名；不改变生产 API。
ConversationHeldV4ActiveRosterReadbackBudget = ConversationHeldOutV4ActiveRosterReadbackBudget


def _active_result(tmp_path: Path):
    """物化包含两个 disjoint cluster、各两条 leaf 的真实 P3-A roster。"""
    train_source, held_source = _source(101, 701), _source(102, 702)
    train_ref, held_ref = _source_ref(1), _source_ref(2)
    train_snapshot, held_snapshot = (_snapshot(train_source, train_ref),
                                     _snapshot(held_source, held_ref))
    train_node = ConversationHeldOutV4LineageNode(_protocol(401), train_snapshot)
    held_node = ConversationHeldOutV4LineageNode(_protocol(402), held_snapshot)
    train_observation = _observation(201, train_source, "train")
    held_observation = _observation(202, held_source, "held_out")
    source_bindings = (
        ConversationHeldOutV4ActiveIntakeSourceBinding(
            train_source, train_ref, train_snapshot, train_node.node_key, "train", _protocol(501)),
        ConversationHeldOutV4ActiveIntakeSourceBinding(
            held_source, held_ref, held_snapshot, held_node.node_key, "held_out", _protocol(502)),
    )
    observation_bindings = (
        ConversationHeldOutV4ActiveIntakeObservationBinding(
            train_observation, train_ref, _sha256(b"train-observation"), train_snapshot,
            train_node.node_key, train_source.source_cluster_key, _protocol(501)),
        ConversationHeldOutV4ActiveIntakeObservationBinding(
            held_observation, held_ref, _sha256(b"held-observation"), held_snapshot,
            held_node.node_key, held_source.source_cluster_key, _protocol(502)),
    )
    root = lambda name: create_new_run_root(  # noqa: E731
        tmp_path / name, require_k_drive=False, label=f"P3-B active {name}")
    request = ConversationHeldOutV4ActiveIntakeInput(
        root("active-staging"), root("active-work"), root("active-publication"),
        _protocol(601), _protocol(602),
        ConversationHeldOutV4ActiveIntakeTeacherPolicy((_protocol(900),), ("W-01",), 0),
        ConversationHeldOutV4ActiveIntakeRosterFactories(
            lambda: (train_node, held_node), lambda: source_bindings,
            lambda: observation_bindings, lambda: ()),
        _active_budget(), "p3b-active-test",
    )
    return materialize_v4_active_intake(request)


def _read_active_mappings(active) -> tuple[ConversationHeldOutV4ActiveIntakeMapping, ...]:
    """通过 P0 reader 取回 P3-A 显式 mapping，不从测试对象偷取 leaf。"""
    descriptor = active.mapping_stream
    with open_plain_binary(active.publication_run_root, descriptor.relative_path,
                           label="P3-B test active mapping") as stream:
        with IntegerFramedStreamReader.from_open_binary(
                stream, path=descriptor.relative_path, max_frame_bytes=32 * 1024,
                max_record_count=64, max_total_payload_bytes=128 * 1024) as reader:
            result = tuple(ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(item)
                           for item in reader)
            assert reader.finish() == descriptor.p0_footer
    return result


def _projection_budget() -> ConversationHeldOutV4ProvenanceProjectionBudget:
    """为 synthetic B3 sealed streams 提供与其 descriptors 一致的冻结预算。"""
    closure_budget = ConversationHeldOutV4ProvenanceClosureBudget(
        16, 32, 64, 64, 8, 4, 64 * 1024, 128, 2 * 1024 * 1024,
        2 * 1024 * 1024, 32 * 1024 * 1024, 3, _sort_budget())
    return ConversationHeldOutV4ProvenanceProjectionBudget(
        16, 16, 64, 4, 8, 64, 64 * 1024, 128, 2 * 1024 * 1024,
        2 * 1024 * 1024, 64 * 1024 * 1024, 4, _sort_budget(), closure_budget)


def _dag_budget() -> ConversationHeldOutV4ProvenanceDagBudget:
    """为每侧 P2-A catalog 到真实 B1 的最小可重放范围固定预算。"""
    return ConversationHeldOutV4ProvenanceDagBudget(
        max_node_count=16,
        max_leaf_count=16,
        max_parent_request_count=16,
        max_direct_edge_count=16,
        max_parents_per_node=4,
        max_leaves_per_node=16,
        max_record_payload_bytes=64 * 1024,
        max_stream_record_count=64,
        max_stream_payload_bytes=2 * 1024 * 1024,
        max_stream_physical_bytes=2 * 1024 * 1024,
        max_total_materialized_physical_bytes=32 * 1024 * 1024,
        max_open_files=3,
        external_sort_budget=_sort_budget(),
    )


def _pack(*parts: tuple[int, ...]) -> tuple[int, ...]:
    """复刻 B3 的长度分帧 tuple 排序语义。"""
    value: list[int] = []
    for part in parts:
        pack_key(value, part)
    return tuple(value)


def _projection_sort_key(channel: int, record: tuple[int, ...]) -> tuple[int, ...]:
    """按公开 B3 projection primary key、side 与 full leaf 排序。"""
    if channel == V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF:
        source, side, leaf = decode_v4_provenance_source_ref_projection_record(record)
        return _pack(source.stable_key(), (side,), leaf.integer_stream())
    if channel == V4_PROVENANCE_PROJECTION_STREAM_CONTENT:
        content, side, leaf = decode_v4_provenance_content_projection_record(record)
        return _pack(content, (side,), leaf.integer_stream())
    namespace, node, snapshot, side, leaf = decode_v4_provenance_lineage_projection_record(record)
    return _pack((namespace,), node.components, snapshot.integer_stream(), (side,),
                 leaf.integer_stream())


def _write_sealed(root, relative: Path, records: tuple[tuple[int, ...], ...]):
    """仅经 K-run capability 写出 sealed P0 test stream。"""
    ensure_normal_relative_directory(root, relative.parent, label="P3-B test parent")
    stream = open_exclusive_binary(root, relative, label="P3-B test stream")
    writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
    try:
        for record in records:
            writer.append(record)
        return writer.seal()
    finally:
        writer.close()


def _side_catalog_and_b1(tmp_path: Path, name: str, mappings, split: str):
    """从 P3-A mapping 的 full leaf/snapshot 重建单侧 P2-A catalog 并运行真实 B1。

    这里不借用 P3-A catalog 的跨侧内容，也不伪造 B1 identity：每个 side 只写出本
    side 所消费的 leaf 和其显式 lineage node。B3 stream 仍由此 fixture 封存，但它的
    direct_dag_stable_key 必须精确绑定本函数返回的 B1 结果。
    """
    selected = tuple(item for item in mappings if item.split == split)
    if not selected:
        raise AssertionError("P3-B test fixture 要求每侧至少一个 active mapping")
    nodes = {}
    for item in selected:
        node = ConversationHeldOutV4LineageNode(item.lineage_node_key, item.snapshot)
        nodes[node.node_key.components] = node
    catalog_root = create_new_run_root(
        tmp_path / f"{name}-catalog", require_k_drive=False,
        label="P3-B side P2-A catalog test transport")
    node_paths = []
    for index, node in enumerate(sorted(nodes.values(), key=lambda item: item.node_key.components)):
        relative = Path("nodes") / f"{index:03d}.pifrs"
        _write_sealed(catalog_root, relative, (node.integer_stream(),))
        node_paths.append(relative)
    leaf_paths = []
    for index, mapping in enumerate(sorted(selected, key=lambda item: item.leaf.stable_key())):
        relative = Path("leaves") / f"{index:03d}.pifrs"
        _write_sealed(catalog_root, relative, (mapping.leaf.integer_stream(),))
        leaf_paths.append(relative)
    catalog = build_v4_provenance_stream_catalog(
        ConversationHeldOutV4ProvenanceCatalogInput(
            catalog_root, tuple(node_paths), tuple(leaf_paths),
            ConversationHeldOutV4ProvenanceCatalogBudget(
                max_total_shards=32,
                max_physical_bytes_per_shard=256 * 1024,
                max_total_physical_bytes=4 * 1024 * 1024,
                max_frame_bytes=64 * 1024,
                max_records_per_stream=64,
                max_total_payload_bytes_per_stream=2 * 1024 * 1024,
            )))
    b3_work_root = create_new_run_root(
        tmp_path / f"{name}-b3-work", require_k_drive=False,
        label="P3-B side B1/B3 work test transport")
    direct = build_v4_provenance_direct_dag(
        catalog, b3_work_root, budget=_dag_budget(),
        logical_stage_name=f"p3b-{name}-b1")
    return selected, catalog_root, catalog, b3_work_root, direct


def _b3_result(tmp_path: Path, name: str, mappings, split: str, *, drop_last: bool = False):
    """以 P3-A 的真实 full leaf 写三个正常排序的 B3 test-transport projections。"""
    expected_side = V4_PROVENANCE_SIDE_TRAINING if split == "train" else V4_PROVENANCE_SIDE_HELD_OUT
    full_selected, catalog_root, catalog, root, direct = _side_catalog_and_b1(
        tmp_path, name, mappings, split)
    selected = list(full_selected)
    if drop_last:
        selected.pop()
    descriptors = {}
    for channel, short in (
            (V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF, "source"),
            (V4_PROVENANCE_PROJECTION_STREAM_CONTENT, "content"),
            (V4_PROVENANCE_PROJECTION_STREAM_LINEAGE, "lineage")):
        records = []
        for item in selected:
            if channel == V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF:
                record = encode_v4_provenance_source_ref_projection_record(
                    item.leaf.source_ref, expected_side, item.leaf)
            elif channel == V4_PROVENANCE_PROJECTION_STREAM_CONTENT:
                record = encode_v4_provenance_content_projection_record(
                    item.leaf.content_sha256, expected_side, item.leaf)
            else:
                record = encode_v4_provenance_lineage_projection_record(
                    1, item.lineage_node_key, item.snapshot, expected_side, item.leaf)
            records.append(record)
        records = tuple(sorted(records, key=lambda item: _projection_sort_key(channel, item)))
        relative = Path(f"b3-{short}") / "projection.pifrs"
        footer = _write_sealed(root, relative, records)
        payload = (root.path / relative).read_bytes()
        descriptors[channel] = ConversationHeldOutV4ProvenanceProjectionStreamDescriptor(
            channel, relative, footer,
            KRunFileDigest(len(payload), tuple(hashlib.sha256(payload).digest())))
    budget = _projection_budget()
    result = ConversationHeldOutV4ProvenanceProjectionResult(
        f"{name}.b3", direct.stable_key(), (2, expected_side),
        descriptors[V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF],
        descriptors[V4_PROVENANCE_PROJECTION_STREAM_CONTENT],
        descriptors[V4_PROVENANCE_PROJECTION_STREAM_LINEAGE], len(selected), len(selected),
        len(selected), 0, budget)
    return catalog_root, catalog, root, direct, result


def _merge_budget() -> ConversationHeldOutV4ProvenanceCrossSideMergeBudget:
    """覆盖三路 P2-C cursor、receipt 与 manifest 的小型固定预算。"""
    return ConversationHeldOutV4ProvenanceCrossSideMergeBudget(
        64 * 1024, 64, 256 * 1024, 512 * 1024, 2 * 1024 * 1024, 3,
        64 * 1024, 64 * 1024, 64 * 1024, 64 * 1024, 32)


def _p2_publication(tmp_path: Path, train_root, train, held_root, held):
    """按公开 P2-C 接口完成三通道 cursor 并封存 publication。"""
    output = create_new_run_root(tmp_path / "p2-publication", require_k_drive=False,
                                 label="P3-B P2-C test transport")
    budget, code = _merge_budget(), _protocol(811)
    cursor = None
    for _ in range(3):
        cursor = advance_v4_provenance_cross_side_merge(
            train, train_root, held, held_root, output, code_identity=code,
            budget=budget, logical_stage_name="p3b-p2c", resume_cursor=cursor)
    assert cursor is not None
    return output, publish_v4_provenance_cross_side_merge(
        train, train_root, held, held_root, output, completed_cursor=cursor,
        code_identity=code, budget=budget, logical_stage_name="p3b-p2c")


def _request(tmp_path: Path, *, drop_train_leaf: bool = False):
    """组成完整 P3-A -> B3 -> P2-C -> P3-B test-only 入口。"""
    active = _active_result(tmp_path)
    mappings = _read_active_mappings(active)
    train_catalog_root, train_catalog, train_root, train_direct, train = _b3_result(
        tmp_path, "train-b3", mappings, "train", drop_last=drop_train_leaf)
    held_catalog_root, held_catalog, held_root, held_direct, held = _b3_result(
        tmp_path, "held-b3", mappings, "held_out")
    p2_root, p2 = _p2_publication(tmp_path, train_root, train, held_root, held)
    root = lambda name: create_new_run_root(  # noqa: E731
        tmp_path / name, require_k_drive=False, label=f"P3-B {name}")
    request = ConversationHeldOutV4ActiveRosterReadbackInput(
        active, train_catalog, train_direct, train, train_root,
        held_catalog, held_direct, held, held_root, p2, p2_root,
        root("p3b-staging"), root("p3b-work"), root("p3b-publication"),
        _protocol(812), _p3b_budget(), "p3b-readback")
    return request, train_catalog_root, train_catalog, train_direct, held_catalog_root, held_catalog, held_direct


def test_active_roster_bidirectional_readback_closes_real_p3a_and_p2_test_transport(tmp_path):
    """P3-B 必须回读 P3-A/P2-C 并以 full leaf 精确闭合两侧三投影。"""
    request, *_ = _request(tmp_path)
    result = materialize_v4_active_roster_bidirectional_readback(request)

    assert result.receipt.status == V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY
    assert result.receipt.training_catalog_stable_key == request.training_catalog.stable_key()
    assert result.receipt.held_out_catalog_stable_key == request.held_out_catalog.stable_key()
    assert result.receipt.training_b1_stable_key == request.training_b1_result.stable_key()
    assert result.receipt.held_out_b1_stable_key == request.held_out_b1_result.stable_key()
    assert result.receipt.counts.integer_stream() == (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    assert revalidate_v4_active_roster_bidirectional_readback(result) == result
    assert {
        path.relative_to(request.publication_run_root.path).as_posix()
        for path in request.publication_run_root.path.rglob("*") if path.is_file()
    } == {"readback/receipt.pifrs", "manifest.pii"}


def test_active_roster_bidirectional_readback_rejects_p2_leaf_missing_from_active_roster(tmp_path):
    """P2 少一条 train full leaf 时，不得发布 P3-B receipt 或 manifest。"""
    request, *_ = _request(tmp_path, drop_train_leaf=True)
    with pytest.raises(ConversationHeldOutV4ActiveRosterReadbackError, match="leaf set"):
        materialize_v4_active_roster_bidirectional_readback(request)
    assert not (request.publication_run_root.path / "readback" / "receipt.pifrs").exists()
    assert not (request.publication_run_root.path / "manifest.pii").exists()


def test_active_roster_readback_rejects_direct_b1_identity_drift_before_outputs(tmp_path):
    """B3 的 direct B1 key 漂移必须在 P3-B staging/work/publication 零写入时停止。"""
    request, *_ = _request(tmp_path)
    drifted_train = replace(
        request.training_b3_result,
        direct_dag_stable_key=(*request.training_b3_result.direct_dag_stable_key, 99),
    )
    request = replace(request, training_b3_result=drifted_train)

    with pytest.raises(ConversationHeldOutV4ActiveRosterReadbackError):
        materialize_v4_active_roster_bidirectional_readback(request)
    for root in (request.staging_run_root, request.work_run_root,
                 request.publication_run_root):
        assert not any(root.path.iterdir())


def test_active_roster_readback_revalidation_rejects_extra_published_file(tmp_path):
    """两文件 publication 封存后出现任意额外文件都必须使回读失败。"""
    request, *_ = _request(tmp_path)
    result = materialize_v4_active_roster_bidirectional_readback(request)
    write_exclusive_bytes(request.publication_run_root, "unexpected.pii", b"residue",
                          label="P3-B test unexpected publication residue")

    with pytest.raises(ConversationHeldOutV4ActiveRosterReadbackError, match="publication"):
        revalidate_v4_active_roster_bidirectional_readback(result)


def test_active_roster_readback_rejects_prepopulated_output_before_input_reads(tmp_path):
    """P3-B 不得接管或覆盖既有 output root，即使其中只有一个普通文件。"""
    request, *_ = _request(tmp_path)
    write_exclusive_bytes(request.publication_run_root, "preexisting.pii", b"residue",
                          label="P3-B test preexisting output")

    with pytest.raises(ConversationHeldOutV4ActiveRosterReadbackError, match="fresh"):
        materialize_v4_active_roster_bidirectional_readback(request)
    assert not (request.publication_run_root.path / "readback" / "receipt.pifrs").exists()


def test_active_caller_gate_revalidates_real_p3b_without_capsule_or_target_io(tmp_path):
    """P3-C0 只回读 P3-B，并对 future runtime roots 维持严格零 payload/write。"""
    request, *_ = _request(tmp_path)
    readback = materialize_v4_active_roster_bidirectional_readback(request)
    source = tmp_path / "future-capsule-source"
    source.mkdir()
    target = tmp_path / "future-runtime-artifact"
    source_before = tuple(source.iterdir())
    parent_before = tuple(tmp_path.iterdir())

    result = run_v4_active_caller_zero_write_dry_run(
        ConversationHeldOutV4ActiveCallerGateInput(
            readback, source, target, _protocol(913), True, "p3c0-dry-run"))

    assert result.readback_stable_key == readback.stable_key()
    assert result.caller_kind == V4_ACTIVE_CALLER_GATE_CALLER_KIND
    assert result.status == V4_ACTIVE_CALLER_GATE_STATUS_TEST_ONLY
    assert result.counts.integer_stream() == (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert tuple(source.iterdir()) == source_before
    assert tuple(tmp_path.iterdir()) == parent_before
    assert not target.exists()


def test_active_roster_readback_module_stays_outside_runtime_private_and_trainer_boundaries():
    """P3-B 只能读取 provenance evidence，不得接入 runtime、private 或训练实现。"""
    source = Path(readback_module.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(token in name for name in imported for token in (
        "runtime", "candidate", "private", "formal", "owner", "trainer", "sqlite"))
    assert "tempfile" not in source


def test_active_caller_gate_stays_outside_capsule_runtime_writer_and_training_boundaries():
    """P3-C0 只能验证 P3-B evidence 与路径边界，不能提前读取或调用 future caller。"""
    source = Path(caller_gate_module.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(token in name for name in imported for token in (
        "external_input_capsule", "candidate_runtime", "runtime_artifact", "trainer",
        "sqlite", "tempfile", "private", "formal", "owner"))
