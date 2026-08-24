"""DLG-05 v4 R04b P2-B1 直接 DAG/leaf endpoint 闭合专项。"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag as dag_module
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_USER,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4LineageNode,
    ConversationHeldOutV4ProvenanceLeaf,
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_SIDE_TRAINING,
    V4_PROVENANCE_UPSTREAM_SHA1,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceCatalogBudget,
    ConversationHeldOutV4ProvenanceCatalogInput,
    ConversationHeldOutV4ProvenanceScalableError,
    build_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag import (
    ConversationHeldOutV4ProvenanceDagBudget,
    ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded,
    ConversationHeldOutV4ProvenanceScalableDagError,
    V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE,
    V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT,
    V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT,
    build_v4_provenance_direct_dag,
    decode_v4_provenance_direct_edge_record,
    decode_v4_provenance_leaf_endpoint_record,
    decode_v4_provenance_node_endpoint_record,
    revalidate_v4_provenance_direct_dag_result,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
)
from pure_integer_ai.storage.integer_external_sort import IntegerExternalSortBudget
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    ensure_normal_relative_directory,
    open_exclusive_binary,
    open_plain_binary,
    create_new_run_root,
    open_existing_run_root,
)


def _scalars(value: str) -> tuple[int, ...]:
    """把 test-only 可读字符串转换成未经归一化的 Unicode scalar tuple。"""
    return tuple(ord(character) for character in value)


def _sha1(value: str) -> tuple[int, ...]:
    """产生完整 SHA-1 tuple，供 snapshot 上游版本 identity 使用。"""
    return tuple(hashlib.sha1(value.encode("utf-8")).digest())


def _sha256(value: str) -> tuple[int, ...]:
    """产生完整 SHA-256 tuple，供内容和本体 test transport 使用。"""
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _key(value: int) -> ProtocolKey:
    """构造排序可读且非摘要的完整 ProtocolKey。"""
    return ProtocolKey((value, value + 10_000))


def _source_ref(value: int) -> SourceRef:
    """构造保留 owner 与版本边界的完整十一整数 SourceRef。"""
    return SourceRef(
        80 + value,
        900 + value,
        value,
        OwnerScope(
            tenant_id=3,
            user_id=7,
            session_id=0,
            visibility=VISIBILITY_USER,
        ),
        VersionBundle(
            CorpusVersion(101 + value),
            ParserVersion(201 + value),
            PrimitiveVersion(301 + value),
            CurriculumVersion(401 + value),
        ),
    )


def _snapshot(value: int, *, source_value: int) -> ConversationHeldOutV4SnapshotIdentity:
    """构造含 URI、revision、hash、许可和代码身份的最小 P1 snapshot。"""
    return ConversationHeldOutV4SnapshotIdentity(
        _source_ref(source_value),
        _scalars(f"https://example.invalid/b1/{value}"),
        _scalars(f"revision-{value}"),
        _scalars(f"snapshot-{value}"),
        V4_PROVENANCE_UPSTREAM_SHA1,
        _sha1(f"upstream-{value}"),
        _sha256(f"local-{value}"),
        _key(1_000 + value),
        _key(2_000 + value),
        _key(3_000 + value),
    )


def _node(
        value: int,
        *,
        source_value: int,
        parents: tuple[ProtocolKey, ...] = (),
        ) -> ConversationHeldOutV4LineageNode:
    """构造规范排序 parent 的完整 P1 node。"""
    return ConversationHeldOutV4LineageNode(
        _key(value),
        _snapshot(value, source_value=source_value),
        parents,
    )


def _leaf(
        value: int,
        *,
        source_value: int,
        node_key: ProtocolKey,
        ) -> ConversationHeldOutV4ProvenanceLeaf:
    """构造训练侧 leaf；B1 将真实 join 它的完整 SourceRef。"""
    return ConversationHeldOutV4ProvenanceLeaf(
        V4_PROVENANCE_SIDE_TRAINING,
        _source_ref(source_value),
        _sha256(f"content-{value}"),
        node_key,
        _key(7_000 + value),
    )


def _source_root(tmp_path: Path, name: str = "source"):
    """建立 P2-A source test transport capability；它不需要 B1 fresh token。"""
    path = tmp_path / name
    path.mkdir()
    return open_existing_run_root(
        path,
        require_k_drive=False,
        label="B1 source test transport",
    )


def _work_root(tmp_path: Path, name: str = "work"):
    """以排他新建能力建立 B1 work root，测试生产形状的 fresh-root 合同。"""
    return create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label="B1 work test transport",
    )


def _write_sealed(root, relative: Path, records: tuple[tuple[int, ...], ...]) -> None:
    """仅经 capability 与 P0 from-open 写入一个 sealed test transport shard。"""
    if relative.parent.parts:
        ensure_normal_relative_directory(root, relative.parent, label="B1 test parent")
    stream = open_exclusive_binary(root, relative, label="B1 test P0 output")
    writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
    try:
        for record in records:
            writer.append(record)
        writer.seal()
    finally:
        writer.close()


def _catalog_budget() -> ConversationHeldOutV4ProvenanceCatalogBudget:
    """提供能覆盖多 shard P1 test transport 的固定 P2-A 读取预算。"""
    return ConversationHeldOutV4ProvenanceCatalogBudget(
        max_total_shards=12,
        max_physical_bytes_per_shard=64 * 1024,
        max_total_physical_bytes=512 * 1024,
        max_frame_bytes=32 * 1024,
        max_records_per_stream=32,
        max_total_payload_bytes_per_stream=48 * 1024,
    )


def _sort_budget(**overrides) -> IntegerExternalSortBudget:
    """小 batch 强制外排，同时给 B1 typed P0 stream 留出足够固定上限。"""
    values = {
        "max_input_file_count": 12,
        "max_input_physical_bytes": 512 * 1024,
        "max_input_record_count": 128,
        "max_input_payload_bytes": 256 * 1024,
        "max_record_payload_bytes": 32 * 1024,
        "max_batch_record_count": 1,
        "max_batch_payload_bytes": 48 * 1024,
        "max_batch_sort_key_bytes": 4 * 1024,
        "max_temporary_run_count": 64,
        "max_temporary_record_count": 1_024,
        "max_temporary_payload_bytes": 512 * 1024,
        "max_temporary_physical_bytes": 768 * 1024,
        "max_output_physical_bytes": 96 * 1024,
        "merge_fan_in": 2,
        "max_open_files": 3,
        "max_merge_pass_count": 16,
    }
    values.update(overrides)
    return IntegerExternalSortBudget(**values)


def _dag_budget(**overrides) -> ConversationHeldOutV4ProvenanceDagBudget:
    """提供带每类、每 stream、累计物化和 handle 上限的 B1 完整预算。"""
    values = {
        "max_node_count": 16,
        "max_leaf_count": 16,
        "max_parent_request_count": 32,
        "max_direct_edge_count": 32,
        "max_parents_per_node": 4,
        "max_leaves_per_node": 4,
        "max_record_payload_bytes": 32 * 1024,
        "max_stream_record_count": 64,
        "max_stream_payload_bytes": 256 * 1024,
        "max_stream_physical_bytes": 96 * 1024,
        "max_total_materialized_physical_bytes": 4 * 1024 * 1024,
        "max_open_files": 3,
        "external_sort_budget": _sort_budget(),
    }
    values.update(overrides)
    return ConversationHeldOutV4ProvenanceDagBudget(**values)


def _build_catalog(
        root,
        *,
        nodes: tuple[ConversationHeldOutV4LineageNode, ...],
        leaves: tuple[ConversationHeldOutV4ProvenanceLeaf, ...],
        ):
    """把已全局排序 P1 records 分为多个 sealed catalog shard。"""
    ordered_nodes = tuple(sorted(nodes, key=lambda item: item.node_key.components))
    ordered_leaves = tuple(sorted(leaves, key=lambda item: item.stable_key()))
    node_paths: list[Path] = []
    leaf_paths: list[Path] = []
    for index, node in enumerate(ordered_nodes):
        relative = Path("nodes") / f"{index:03d}.pifrs"
        _write_sealed(root, relative, (node.integer_stream(),))
        node_paths.append(relative)
    for index, leaf in enumerate(ordered_leaves):
        relative = Path("leaves") / f"{index:03d}.pifrs"
        _write_sealed(root, relative, (leaf.integer_stream(),))
        leaf_paths.append(relative)
    return build_v4_provenance_stream_catalog(
        ConversationHeldOutV4ProvenanceCatalogInput(
            root,
            tuple(node_paths),
            tuple(leaf_paths),
            _catalog_budget(),
        ))


def _read_records(root, descriptor, budget):
    """经 K capability 和 P0 reader 回读一个 B1 descriptor，不将 files 聚入内存。"""
    stream = open_plain_binary(
        root,
        descriptor.work_relative_path,
        label="B1 test descriptor read",
    )
    reader = IntegerFramedStreamReader.from_open_binary(
        stream,
        path=descriptor.work_relative_path,
        max_frame_bytes=budget.max_record_payload_bytes,
        max_record_count=budget.max_stream_record_count,
        max_total_payload_bytes=budget.max_stream_payload_bytes,
    )
    try:
        records = tuple(reader)
        assert reader.footer == descriptor.p0_footer
        return records
    finally:
        reader.close()


def _tree_file_identity(root: Path) -> tuple[tuple[str, bytes], ...]:
    """记录 test transport tree 的全部文件字节，证明 B1 readback 零写入。"""
    return tuple(sorted(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).digest())
        for path in root.rglob("*")
        if path.is_file()
    ))


def _fixture_catalog(tmp_path: Path):
    """构造跨 shard 且 parent/leaf 键故意与最终 join key 乱序的小 DAG。"""
    source = _source_root(tmp_path)
    first = _node(10, source_value=10)
    second = _node(20, source_value=20)
    # node 顺序为 10,20,30,40，但 parent request 写出顺序为 20,10，必须外排。
    third = _node(30, source_value=100, parents=(second.node_key,))
    fourth = _node(40, source_value=1, parents=(first.node_key,))
    # leaf 的 SourceRef 排序先为 node 40，再为 node 30，B1 必须按 lineage key 重新外排。
    first_leaf = _leaf(1, source_value=1, node_key=fourth.node_key)
    second_leaf = _leaf(2, source_value=100, node_key=third.node_key)
    catalog = _build_catalog(
        source,
        nodes=(first, second, third, fourth),
        leaves=(first_leaf, second_leaf),
    )
    return source, catalog, (first, second, third, fourth), (first_leaf, second_leaf)


def test_direct_dag_closes_cross_shard_endpoints_and_external_parent_leaf_orders(tmp_path):
    """B1 必须流式闭合跨 shard 的 node/leaf，并保留完整 parent snapshot 而非摘要。"""
    _source, catalog, nodes, leaves = _fixture_catalog(tmp_path)
    work = _work_root(tmp_path)
    budget = _dag_budget()

    result = build_v4_provenance_direct_dag(
        catalog,
        work,
        budget=budget,
        logical_stage_name="dlg05-v4-p2b1.v1",
    )

    assert result.node_count == 4
    assert result.parent_request_count == 2
    assert result.direct_edge_count == 2
    assert result.leaf_count == 2
    assert result.leaf_endpoint_count == 2
    assert result.node_endpoint_stream.stream_kind == V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT
    assert result.direct_edge_stream.stream_kind == V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE
    assert result.leaf_endpoint_stream.stream_kind == V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT
    assert str(work.path) not in str(result.stable_key())

    endpoint_records = _read_records(work, result.node_endpoint_stream, budget)
    assert tuple(
        decode_v4_provenance_node_endpoint_record(record)[0]
        for record in endpoint_records
    ) == tuple(node.node_key for node in nodes)

    edges = tuple(
        decode_v4_provenance_direct_edge_record(record)
        for record in _read_records(work, result.direct_edge_stream, budget)
    )
    assert tuple((child, parent) for child, parent, _snapshot_value in edges) == (
        (nodes[3].node_key, nodes[0].node_key),
        (nodes[2].node_key, nodes[1].node_key),
    )
    assert edges[0][2] == nodes[0].snapshot
    assert edges[1][2] == nodes[1].snapshot

    endpoints = tuple(
        decode_v4_provenance_leaf_endpoint_record(record)
        for record in _read_records(work, result.leaf_endpoint_stream, budget)
    )
    assert tuple(leaf.lineage_node_key for leaf, _snapshot_value in endpoints) == (
        nodes[2].node_key,
        nodes[3].node_key,
    )
    assert tuple(leaf for leaf, _snapshot_value in endpoints) == (leaves[1], leaves[0])
    assert all(
        leaf.source_ref.stable_key() == snapshot.source_ref.stable_key()
        for leaf, snapshot in endpoints
    )


def test_direct_dag_streams_shared_parent_and_leaf_groups_with_bounded_groups(
        tmp_path):
    """同一 parent/lineage node 的多条关系必须逐条闭合，不能退化为单条或全量缓存。"""
    source = _source_root(tmp_path, "shared-group-source")
    root = _node(10, source_value=10)
    shared_parent = _node(20, source_value=20)
    first_child = _node(
        30,
        source_value=30,
        parents=(shared_parent.node_key,),
    )
    second_child = _node(
        40,
        source_value=40,
        parents=(shared_parent.node_key,),
    )
    first_leaf = _leaf(1, source_value=20, node_key=shared_parent.node_key)
    second_leaf = _leaf(2, source_value=20, node_key=shared_parent.node_key)
    catalog = _build_catalog(
        source,
        nodes=(root, shared_parent, first_child, second_child),
        leaves=(first_leaf, second_leaf),
    )
    budget = _dag_budget(max_leaves_per_node=2)

    result = build_v4_provenance_direct_dag(
        catalog,
        _work_root(tmp_path, "shared-group-work"),
        budget=budget,
        logical_stage_name="shared-group.v1",
    )

    assert result.direct_edge_count == 2
    assert result.leaf_endpoint_count == 2
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded,
                       match="leaf group"):
        build_v4_provenance_direct_dag(
            catalog,
            _work_root(tmp_path, "shared-group-over-budget-work"),
            budget=_dag_budget(max_leaves_per_node=1),
            logical_stage_name="shared-group-over-budget.v1",
        )


def test_revalidates_b1_catalog_to_six_descriptors_with_reopened_read_only_root(
        tmp_path):
    """B1 公开回读必须接受 reopen capability，并且不改写 catalog 或封存 work tree。"""
    source, catalog, _nodes, _leaves = _fixture_catalog(tmp_path)
    work = _work_root(tmp_path, "b1-readback-work")
    result = build_v4_provenance_direct_dag(
        catalog, work, budget=_dag_budget(), logical_stage_name="b1-readback.v1")
    reopened = open_existing_run_root(
        work.path,
        require_k_drive=False,
        label="B1 readback reopened test transport",
    )
    before_source = _tree_file_identity(source.path)
    before_work = _tree_file_identity(work.path)

    assert revalidate_v4_provenance_direct_dag_result(
        result, reopened, catalog=catalog) is result
    assert _tree_file_identity(source.path) == before_source
    assert _tree_file_identity(work.path) == before_work


def test_b1_readback_rejects_catalog_identity_mismatch_and_sealed_physical_drift(
        tmp_path):
    """catalog-to-result stable key 和每条 B1 descriptor physical 都不可替换。"""
    source, catalog, _nodes, _leaves = _fixture_catalog(tmp_path)
    work = _work_root(tmp_path, "b1-readback-drift-work")
    result = build_v4_provenance_direct_dag(
        catalog, work, budget=_dag_budget(), logical_stage_name="b1-readback-drift.v1")
    before_source = _tree_file_identity(source.path)
    other_source = _source_root(tmp_path, "b1-readback-other-source")
    other_node = _node(90, source_value=90)
    other_catalog = _build_catalog(other_source, nodes=(other_node,), leaves=())

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError,
                       match="catalog stable key"):
        revalidate_v4_provenance_direct_dag_result(
            result, work, catalog=other_catalog)

    target = work.path / result.direct_edge_stream.work_relative_path
    with target.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError,
                       match="漂移|physical"):
        revalidate_v4_provenance_direct_dag_result(
            result, work, catalog=catalog)
    assert _tree_file_identity(source.path) == before_source


def test_direct_dag_rejects_unknown_parent_unknown_leaf_and_source_ref_mismatch(tmp_path):
    """任一缺失 endpoint 或完整十一整数 SourceRef 漂移都不能产生 B1 result。"""
    parent_source = _source_root(tmp_path, "unknown-parent-source")
    root = _node(10, source_value=10)
    unknown_parent_child = _node(
        20,
        source_value=20,
        parents=(_key(999),),
    )
    parent_catalog = _build_catalog(
        parent_source,
        nodes=(root, unknown_parent_child),
        leaves=(),
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError, match="未知 node"):
        build_v4_provenance_direct_dag(
            parent_catalog,
            _work_root(tmp_path, "unknown-parent-work"),
            budget=_dag_budget(),
            logical_stage_name="unknown-parent.v1",
        )

    leaf_source = _source_root(tmp_path, "unknown-leaf-source")
    known = _node(10, source_value=10)
    unknown_leaf = _leaf(1, source_value=10, node_key=_key(999))
    leaf_catalog = _build_catalog(
        leaf_source,
        nodes=(known,),
        leaves=(unknown_leaf,),
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError, match="未知 node"):
        build_v4_provenance_direct_dag(
            leaf_catalog,
            _work_root(tmp_path, "unknown-leaf-work"),
            budget=_dag_budget(),
            logical_stage_name="unknown-leaf.v1",
        )

    mismatch_source = _source_root(tmp_path, "mismatch-source")
    known = _node(10, source_value=10)
    mismatch_leaf = _leaf(1, source_value=11, node_key=known.node_key)
    mismatch_catalog = _build_catalog(
        mismatch_source,
        nodes=(known,),
        leaves=(mismatch_leaf,),
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError, match="SourceRef"):
        build_v4_provenance_direct_dag(
            mismatch_catalog,
            _work_root(tmp_path, "mismatch-work"),
            budget=_dag_budget(),
            logical_stage_name="mismatch.v1",
        )

    version_source = _source_root(tmp_path, "version-mismatch-source")
    known = _node(10, source_value=10)
    baseline_leaf = _leaf(1, source_value=10, node_key=known.node_key)
    baseline_ref = baseline_leaf.source_ref
    version_only_ref = SourceRef(
        baseline_ref.source_kind,
        baseline_ref.source_id,
        baseline_ref.document_id,
        baseline_ref.owner,
        VersionBundle(
            baseline_ref.versions.corpus,
            ParserVersion(baseline_ref.versions.parser.value + 1),
            baseline_ref.versions.primitive,
            baseline_ref.versions.curriculum,
        ),
    )
    version_mismatch_leaf = ConversationHeldOutV4ProvenanceLeaf(
        baseline_leaf.side,
        version_only_ref,
        baseline_leaf.content_sha256,
        baseline_leaf.lineage_node_key,
        baseline_leaf.consumed_input_identity,
    )
    assert sum(
        left != right
        for left, right in zip(
            baseline_ref.stable_key(),
            version_only_ref.stable_key(),
            strict=True,
        )
    ) == 1
    version_catalog = _build_catalog(
        version_source,
        nodes=(known,),
        leaves=(version_mismatch_leaf,),
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError, match="SourceRef"):
        build_v4_provenance_direct_dag(
            version_catalog,
            _work_root(tmp_path, "version-mismatch-work"),
            budget=_dag_budget(),
            logical_stage_name="version-mismatch.v1",
        )


def test_direct_dag_rejects_budget_nonfresh_nonempty_and_overlapping_work_roots(tmp_path):
    """B1 不能扩大节点预算、复用旧根、接受脏根或与 catalog 相同的工作目录。"""
    source, catalog, _nodes, _leaves = _fixture_catalog(tmp_path)
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded, match="node 数"):
        build_v4_provenance_direct_dag(
            catalog,
            _work_root(tmp_path, "budget-work"),
            budget=_dag_budget(max_node_count=1),
            logical_stage_name="budget.v1",
        )

    parent_source = _source_root(tmp_path, "parent-budget-source")
    first_parent = _node(10, source_value=10)
    second_parent = _node(20, source_value=20)
    two_parent_child = _node(
        30,
        source_value=30,
        parents=(first_parent.node_key, second_parent.node_key),
    )
    parent_catalog = _build_catalog(
        parent_source,
        nodes=(first_parent, second_parent, two_parent_child),
        leaves=(),
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded, match="parent 数"):
        build_v4_provenance_direct_dag(
            parent_catalog,
            _work_root(tmp_path, "parent-budget-work"),
            budget=_dag_budget(max_parents_per_node=1),
            logical_stage_name="parent-budget.v1",
        )

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded, match="external sort"):
        build_v4_provenance_direct_dag(
            catalog,
            _work_root(tmp_path, "external-sort-budget-work"),
            budget=_dag_budget(external_sort_budget=_sort_budget(
                max_input_record_count=1,
                max_batch_record_count=1,
            )),
            logical_stage_name="external-sort-budget.v1",
        )

    existing_empty = open_existing_run_root(
        _work_root(tmp_path, "reopened-work").path,
        require_k_drive=False,
        label="B1 reopened work",
    )
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        build_v4_provenance_direct_dag(
            catalog,
            existing_empty,
            budget=_dag_budget(),
            logical_stage_name="reopened.v1",
        )

    nonempty = _work_root(tmp_path, "nonempty-work")
    ensure_normal_relative_directory(nonempty, "old-stage")
    with pytest.raises(KRunBoundaryError, match="必须为空"):
        build_v4_provenance_direct_dag(
            catalog,
            nonempty,
            budget=_dag_budget(),
            logical_stage_name="nonempty.v1",
        )

    with pytest.raises(KRunBoundaryError, match="相同|嵌套"):
        build_v4_provenance_direct_dag(
            catalog,
            source,
            budget=_dag_budget(),
            logical_stage_name="overlap.v1",
        )


def test_direct_dag_detects_catalog_identity_drift_before_replay(tmp_path):
    """冻结 catalog 之后替换任一 P0 shard，B1 的首个 revalidate 必须 fail closed。"""
    source, catalog, nodes, _leaves = _fixture_catalog(tmp_path)
    original_relative = catalog.node_streams[0].relative_path
    alternate = Path("alternate.pifrs")
    _write_sealed(source, alternate, (_node(11, source_value=11).integer_stream(),))
    os.replace(source.path / alternate, source.path / original_relative)

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="stable key|sealed P0"):
        build_v4_provenance_direct_dag(
            catalog,
            _work_root(tmp_path),
            budget=_dag_budget(),
            logical_stage_name="drift.v1",
        )
    assert nodes[0].node_key.components != _key(11).components


def test_direct_dag_rechecks_frozen_descriptor_on_second_catalog_consumption(
        tmp_path, monkeypatch):
    """即使首个 catalog revalidate 已通过，后续 node replay 仍须逐字段拒绝替换。"""
    source, catalog, _nodes, _leaves = _fixture_catalog(tmp_path)
    original_relative = catalog.node_streams[0].relative_path
    alternate = Path("second-pass-alternate.pifrs")
    _write_sealed(source, alternate, (_node(12, source_value=12).integer_stream(),))
    original_revalidate = dag_module.revalidate_v4_provenance_stream_catalog

    def revalidate_then_replace(value):
        """模拟 catalog 完整复核结束后、B1 第二次消费前 source 发生可观测替换。"""
        result = original_revalidate(value)
        os.replace(source.path / alternate, source.path / original_relative)
        return result

    monkeypatch.setattr(
        dag_module,
        "revalidate_v4_provenance_stream_catalog",
        revalidate_then_replace,
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError, match="冻结 descriptor"):
        build_v4_provenance_direct_dag(
            catalog,
            _work_root(tmp_path),
            budget=_dag_budget(),
            logical_stage_name="second-pass-drift.v1",
        )


def test_direct_dag_rejects_leaf_input_replacement_before_leaf_external_sort(
        tmp_path, monkeypatch):
    """leaf 在 catalog 重验后、实际外排前被替换时，冻结 descriptor 仍必须拒绝。"""
    source, catalog, nodes, _leaves = _fixture_catalog(tmp_path)
    original_relative = catalog.leaf_streams[0].relative_path
    alternate = Path("leaf-second-pass-alternate.pifrs")
    replacement = _leaf(99, source_value=1, node_key=nodes[3].node_key)
    _write_sealed(source, alternate, (replacement.integer_stream(),))
    original_join = dag_module._join_direct_edges

    def join_then_replace(*args, **kwargs):
        """把替换点固定在 direct edge 闭合后、leaf 外排实际读取前。"""
        result = original_join(*args, **kwargs)
        os.replace(source.path / alternate, source.path / original_relative)
        return result

    monkeypatch.setattr(dag_module, "_join_direct_edges", join_then_replace)
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableDagError,
                       match="identity 漂移"):
        build_v4_provenance_direct_dag(
            catalog,
            _work_root(tmp_path),
            budget=_dag_budget(),
            logical_stage_name="leaf-second-pass-drift.v1",
        )


def test_direct_dag_is_deterministic_and_has_no_runtime_or_unsafe_domain_imports(tmp_path):
    """相同 catalog/budget 的独立 fresh root 必须产生相同无路径结果，模块保持 P2 边界。"""
    _source, catalog, _nodes, _leaves = _fixture_catalog(tmp_path)
    budget = _dag_budget()
    first_work = _work_root(tmp_path, "first-work")
    second_work = _work_root(tmp_path, "second-work")
    first = build_v4_provenance_direct_dag(
        catalog,
        first_work,
        budget=budget,
        logical_stage_name="deterministic.v1",
    )
    second = build_v4_provenance_direct_dag(
        catalog,
        second_work,
        budget=budget,
        logical_stage_name="deterministic.v1",
    )
    assert first.stable_key() == second.stable_key()
    for first_descriptor, second_descriptor in zip((
            first.node_endpoint_stream,
            first.parent_request_stream,
            first.sorted_parent_request_stream,
            first.direct_edge_stream,
            first.sorted_leaf_stream,
            first.leaf_endpoint_stream,
    ), (
            second.node_endpoint_stream,
            second.parent_request_stream,
            second.sorted_parent_request_stream,
            second.direct_edge_stream,
            second.sorted_leaf_stream,
            second.leaf_endpoint_stream,
    ), strict=True):
        assert first_descriptor.stable_key() == second_descriptor.stable_key()

    source_path = Path(dag_module.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    local_imports = []
    path_open_calls = []
    for item in ast.walk(tree):
        if isinstance(item, ast.ImportFrom) and item.module and item.module.startswith(
                "pure_integer_ai"):
            local_imports.append(item.module)
        if (isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "open"):
            path_open_calls.append(item.lineno)
    assert "ConversationHeldOutV4LineageFreeze" not in source_text
    assert "json" not in source_text
    assert "sqlite" not in source_text
    assert not any(any(token in item for token in (
        "runtime", "owner", "private", "formal", "family",
    )) for item in local_imports)
    assert path_open_calls == []
