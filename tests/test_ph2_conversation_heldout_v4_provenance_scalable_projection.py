"""DLG-05 v4 R04b P2-B3 三路可扩展 provenance 投影专项。"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_projection as projection_module
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
    V4_PROVENANCE_SIDE_HELD_OUT,
    V4_PROVENANCE_SIDE_TRAINING,
    V4_PROVENANCE_UPSTREAM_SHA1,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceCatalogBudget,
    ConversationHeldOutV4ProvenanceCatalogInput,
    build_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_closure import (
    ConversationHeldOutV4ProvenanceClosureBudget,
    ConversationHeldOutV4ProvenanceClosureResult,
    ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
    V4_PROVENANCE_CLOSURE_STREAM_KNOWN,
    build_v4_provenance_transitive_closure,
    decode_v4_provenance_closure_pair_record,
    encode_v4_provenance_closure_pair_record,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag import (
    ConversationHeldOutV4ProvenanceDagBudget,
    build_v4_provenance_direct_dag,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_projection import (
    ConversationHeldOutV4ProvenanceProjectionBudget,
    ConversationHeldOutV4ProvenanceScalableProjectionBudgetExceeded,
    ConversationHeldOutV4ProvenanceScalableProjectionError,
    V4_PROVENANCE_PROJECTION_NAMESPACE_LINEAGE_NODE,
    V4_PROVENANCE_PROJECTION_STREAM_CONTENT,
    V4_PROVENANCE_PROJECTION_STREAM_LINEAGE,
    V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF,
    build_v4_provenance_leaf_projections,
    decode_v4_provenance_content_projection_record,
    decode_v4_provenance_lineage_projection_record,
    decode_v4_provenance_source_ref_projection_record,
    revalidate_v4_provenance_leaf_projections,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
    pack_key,
)
from pure_integer_ai.storage.integer_external_sort import IntegerExternalSortBudget
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    KRunFileDigest,
    create_new_run_root,
    ensure_normal_relative_directory,
    open_existing_run_root,
    open_exclusive_binary,
    open_plain_binary,
)


def _scalars(value: str) -> tuple[int, ...]:
    """把 test-only 文本映射为未经归一化的 Unicode scalar tuple。"""
    return tuple(ord(item) for item in value)


def _sha1(value: str) -> tuple[int, ...]:
    """生成完整 SHA-1 bytes，用作受控上游 snapshot 身份。"""
    return tuple(hashlib.sha1(value.encode("utf-8")).digest())


def _sha256(value: str) -> tuple[int, ...]:
    """生成完整 SHA-256 bytes，用作 leaf content 或 local snapshot 身份。"""
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _key(value: int) -> ProtocolKey:
    """构造完整、有序且可区分的 ProtocolKey。"""
    return ProtocolKey((value, value + 30_000))


def _source_ref(value: int) -> SourceRef:
    """构造含 owner/version 全十一整数的 SourceRef。"""
    return SourceRef(
        80 + value,
        800 + value,
        value,
        OwnerScope(tenant_id=4, user_id=8, session_id=0,
                   visibility=VISIBILITY_USER),
        VersionBundle(
            CorpusVersion(100 + value),
            ParserVersion(200 + value),
            PrimitiveVersion(300 + value),
            CurriculumVersion(400 + value),
        ),
    )


def _snapshot(value: int, *, revision: str | None = None) -> ConversationHeldOutV4SnapshotIdentity:
    """构造一个完整 node snapshot，可仅改变 revision 来模拟全 identity 漂移。"""
    return ConversationHeldOutV4SnapshotIdentity(
        _source_ref(value),
        _scalars(f"https://example.invalid/b3/{value}"),
        _scalars(revision if revision is not None else f"revision-{value}"),
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
        source_value: int | None = None,
        parents: tuple[ProtocolKey, ...] = (),
        ) -> ConversationHeldOutV4LineageNode:
    """构造 P1 typed node，允许 parent DAG 与来源身份独立编排。"""
    actual_source = value if source_value is None else source_value
    return ConversationHeldOutV4LineageNode(
        _key(value),
        _snapshot(actual_source),
        parents,
    )


def _leaf(
        value: int,
        *,
        source_value: int,
        node_key: ProtocolKey,
        side: int = V4_PROVENANCE_SIDE_TRAINING,
        content_value: str | None = None,
        ) -> ConversationHeldOutV4ProvenanceLeaf:
    """构造指向完整 endpoint 的 leaf；不同 value 保持消费输入 identity 不同。"""
    return ConversationHeldOutV4ProvenanceLeaf(
        side,
        _source_ref(source_value),
        _sha256(content_value if content_value is not None else f"content-{value}"),
        node_key,
        _key(5_000 + value),
    )


def _source_root(tmp_path: Path, name: str):
    """创建显式 test transport source capability。"""
    return create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label="B3 source test transport",
    )


def _work_root(tmp_path: Path, name: str):
    """创建 B1/B2/B3 共用的原始 create-new work capability。"""
    return create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label="B3 work test transport",
    )


def _write_sealed(root, relative: Path, records: tuple[tuple[int, ...], ...]):
    """只经 capability/P0 bridge 写入封存测试 stream。"""
    if relative.parent.parts:
        ensure_normal_relative_directory(root, relative.parent,
                                         label="B3 test shard parent")
    stream = open_exclusive_binary(root, relative, label="B3 test shard")
    writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
    try:
        for record in records:
            writer.append(record)
        return writer.seal()
    finally:
        writer.close()


def _catalog_budget() -> ConversationHeldOutV4ProvenanceCatalogBudget:
    """提供小型多 shard P2-A 输入的明确资源上限。"""
    return ConversationHeldOutV4ProvenanceCatalogBudget(
        max_total_shards=32,
        max_physical_bytes_per_shard=96 * 1024,
        max_total_physical_bytes=2 * 1024 * 1024,
        max_frame_bytes=48 * 1024,
        max_records_per_stream=128,
        max_total_payload_bytes_per_stream=1 * 1024 * 1024,
    )


def _sort_budget(**overrides) -> IntegerExternalSortBudget:
    """小 batch 强制每次 B1/B2/B3 sort 走受限的真实外排与多轮 merge。"""
    values = {
        "max_input_file_count": 32,
        "max_input_physical_bytes": 2 * 1024 * 1024,
        "max_input_record_count": 512,
        "max_input_payload_bytes": 2 * 1024 * 1024,
        "max_record_payload_bytes": 48 * 1024,
        "max_batch_record_count": 1,
        "max_batch_payload_bytes": 96 * 1024,
        "max_batch_sort_key_bytes": 96 * 1024,
        "max_temporary_run_count": 1_024,
        "max_temporary_record_count": 8_192,
        "max_temporary_payload_bytes": 16 * 1024 * 1024,
        "max_temporary_physical_bytes": 24 * 1024 * 1024,
        "max_output_physical_bytes": 2 * 1024 * 1024,
        "merge_fan_in": 2,
        "max_open_files": 3,
        "max_merge_pass_count": 32,
    }
    values.update(overrides)
    return IntegerExternalSortBudget(**values)


def _dag_budget() -> ConversationHeldOutV4ProvenanceDagBudget:
    """覆盖 B1 node/leaf endpoint 与原始外排 stage 的完整预算。"""
    return ConversationHeldOutV4ProvenanceDagBudget(
        max_node_count=64,
        max_leaf_count=64,
        max_parent_request_count=128,
        max_direct_edge_count=128,
        max_parents_per_node=8,
        max_leaves_per_node=8,
        max_record_payload_bytes=48 * 1024,
        max_stream_record_count=256,
        max_stream_payload_bytes=2 * 1024 * 1024,
        max_stream_physical_bytes=2 * 1024 * 1024,
        max_total_materialized_physical_bytes=96 * 1024 * 1024,
        max_open_files=3,
        external_sort_budget=_sort_budget(),
    )


def _closure_budget() -> ConversationHeldOutV4ProvenanceClosureBudget:
    """覆盖 B2 strict ancestor fixed point 的受控资源上限。"""
    return ConversationHeldOutV4ProvenanceClosureBudget(
        max_node_count=64,
        max_direct_edge_count=128,
        max_known_pair_count=512,
        max_candidate_pair_count_per_round=512,
        max_closure_round_count=16,
        max_parents_per_node=8,
        max_record_payload_bytes=48 * 1024,
        max_stream_record_count=512,
        max_stream_payload_bytes=4 * 1024 * 1024,
        max_stream_physical_bytes=2 * 1024 * 1024,
        max_total_materialized_physical_bytes=256 * 1024 * 1024,
        max_open_files=3,
        external_sort_budget=_sort_budget(),
    )


def _projection_budget(**overrides) -> ConversationHeldOutV4ProvenanceProjectionBudget:
    """提供三路 B3 raw/sort 及 ancestor endpoint 对齐的明确预算。"""
    values = {
        "max_node_count": 64,
        "max_leaf_count": 64,
        "max_known_pair_count": 512,
        "max_leaves_per_node": 8,
        "max_ancestors_per_node": 32,
        "max_lineage_projection_record_count": 1_024,
        "max_record_payload_bytes": 48 * 1024,
        "max_stream_record_count": 1_024,
        "max_stream_payload_bytes": 8 * 1024 * 1024,
        "max_stream_physical_bytes": 4 * 1024 * 1024,
        "max_total_materialized_physical_bytes": 512 * 1024 * 1024,
        "max_open_files": 4,
        "external_sort_budget": _sort_budget(),
        "closure_replay_budget": _closure_budget(),
    }
    values.update(overrides)
    return ConversationHeldOutV4ProvenanceProjectionBudget(**values)


def _build_catalog(
        root,
        nodes: tuple[ConversationHeldOutV4LineageNode, ...],
        leaves: tuple[ConversationHeldOutV4ProvenanceLeaf, ...],
        ):
    """把 deliberately split 且全局排序的 P1 records 写为 sealed P2-A catalog。"""
    node_paths: list[Path] = []
    leaf_paths: list[Path] = []
    for index, node in enumerate(sorted(nodes, key=lambda item: item.node_key.components)):
        relative = Path("nodes") / f"{index:03d}.pifrs"
        _write_sealed(root, relative, (node.integer_stream(),))
        node_paths.append(relative)
    for index, leaf in enumerate(sorted(leaves, key=lambda item: item.stable_key())):
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


def _build_pipeline(
        tmp_path: Path,
        *,
        name: str,
        logical_name: str | None = None,
        nodes: tuple[ConversationHeldOutV4LineageNode, ...],
        leaves: tuple[ConversationHeldOutV4ProvenanceLeaf, ...],
        ):
    """构造同一 original work capability 上连续 B1/B2 的公开 typed test transport。"""
    source = _source_root(tmp_path, f"{name}-source")
    catalog = _build_catalog(source, nodes, leaves)
    work = _work_root(tmp_path, f"{name}-work")
    stage_name = name if logical_name is None else logical_name
    direct = build_v4_provenance_direct_dag(
        catalog,
        work,
        budget=_dag_budget(),
        logical_stage_name=f"{stage_name}.b1",
    )
    closure = build_v4_provenance_transitive_closure(
        direct,
        work,
        budget=_closure_budget(),
        logical_stage_name=f"{stage_name}.b2",
    )
    return work, direct, closure


def _read_records(work, descriptor, budget):
    """经 work capability/P0 reader 回读 B3 sorted descriptor 的 record tuple。"""
    stream = open_plain_binary(
        work,
        descriptor.work_relative_path,
        label="B3 test projection read",
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


def _packed(*parts: tuple[int, ...]) -> tuple[int, ...]:
    """复用协议的长度分帧排序语义，供专项断言验证输出严格排序。"""
    result: list[int] = []
    for part in parts:
        pack_key(result, part)
    return tuple(result)


def _chain_fixture(tmp_path: Path, name: str, *, logical_name: str | None = None):
    """构造 root/middle/tip chain 与三条跨 side/source/content 排序的 leaves。"""
    root = _node(10, source_value=10)
    middle = _node(20, source_value=20, parents=(root.node_key,))
    tip = _node(30, source_value=30, parents=(middle.node_key,))
    leaves = (
        _leaf(1, source_value=30, node_key=tip.node_key,
              side=V4_PROVENANCE_SIDE_TRAINING, content_value="z-content"),
        _leaf(2, source_value=20, node_key=middle.node_key,
              side=V4_PROVENANCE_SIDE_HELD_OUT, content_value="a-content"),
        _leaf(3, source_value=10, node_key=root.node_key,
              side=V4_PROVENANCE_SIDE_TRAINING, content_value="m-content"),
    )
    return (*_build_pipeline(
        tmp_path,
        name=name,
        logical_name=logical_name,
        nodes=(root, middle, tip),
        leaves=leaves,
    ), (root, middle, tip), leaves)


def test_three_independent_projections_preserve_full_leaf_and_all_strict_ancestors(tmp_path):
    """B3 必须独立排序三通道，并对每条 leaf 写 self 加上全部 strict ancestors。"""
    work, direct, closure, nodes, leaves = _chain_fixture(tmp_path, "three")
    budget = _projection_budget()

    result = build_v4_provenance_leaf_projections(
        direct,
        closure,
        work,
        budget=budget,
        logical_stage_name="dlg05-v4-p2b3.v1",
    )

    assert result.source_ref_projection_count == len(leaves)
    assert result.content_projection_count == len(leaves)
    assert result.lineage_projection_count == 6
    assert result.source_ref_projection_stream.stream_kind == (
        V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF)
    assert result.content_projection_stream.stream_kind == (
        V4_PROVENANCE_PROJECTION_STREAM_CONTENT)
    assert result.lineage_projection_stream.stream_kind == (
        V4_PROVENANCE_PROJECTION_STREAM_LINEAGE)
    assert str(work.path) not in str(result.stable_key())

    source_records = tuple(map(
        decode_v4_provenance_source_ref_projection_record,
        _read_records(work, result.source_ref_projection_stream, budget),
    ))
    assert tuple(item[2] for item in source_records) == tuple(sorted(
        leaves,
        key=lambda leaf: _packed(
            leaf.source_ref.stable_key(), (leaf.side,), leaf.integer_stream()),
    ))
    assert tuple(
        _packed(item[0].stable_key(), (item[1],), item[2].integer_stream())
        for item in source_records
    ) == tuple(sorted(
        _packed(item[0].stable_key(), (item[1],), item[2].integer_stream())
        for item in source_records
    ))

    content_records = tuple(map(
        decode_v4_provenance_content_projection_record,
        _read_records(work, result.content_projection_stream, budget),
    ))
    assert tuple(item[2] for item in content_records) == tuple(sorted(
        leaves,
        key=lambda leaf: _packed(
            leaf.content_sha256, (leaf.side,), leaf.integer_stream()),
    ))
    assert tuple(
        _packed(item[0], (item[1],), item[2].integer_stream())
        for item in content_records
    ) == tuple(sorted(
        _packed(item[0], (item[1],), item[2].integer_stream())
        for item in content_records
    ))

    lineage_records = tuple(map(
        decode_v4_provenance_lineage_projection_record,
        _read_records(work, result.lineage_projection_stream, budget),
    ))
    snapshots = {node.node_key.components: node.snapshot for node in nodes}
    expected_ancestors = {
        leaves[0].consumed_input_identity.components: (
            nodes[0].node_key.components,
            nodes[1].node_key.components,
            nodes[2].node_key.components,
        ),
        leaves[1].consumed_input_identity.components: (
            nodes[0].node_key.components,
            nodes[1].node_key.components,
        ),
        leaves[2].consumed_input_identity.components: (
            nodes[0].node_key.components,
        ),
    }
    actual_ancestors: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for namespace, ancestor, snapshot, side, leaf in lineage_records:
        assert namespace == V4_PROVENANCE_PROJECTION_NAMESPACE_LINEAGE_NODE
        assert side == leaf.side
        assert snapshot == snapshots[ancestor.components]
        actual_ancestors.setdefault(leaf.consumed_input_identity.components, []).append(
            ancestor.components)
    assert {
        key: tuple(sorted(value))
        for key, value in actual_ancestors.items()
    } == {
        key: tuple(sorted(value))
        for key, value in expected_ancestors.items()
    }
    lineage_keys = tuple(
        _packed(
            (namespace,), ancestor.components, snapshot.integer_stream(), (side,),
            leaf.integer_stream())
        for namespace, ancestor, snapshot, side, leaf in lineage_records
    )
    assert lineage_keys == tuple(sorted(lineage_keys))


def test_same_source_or_content_different_full_leafs_are_preserved_without_verdict(tmp_path):
    """相同 SourceRef/content 的不同 leaf 只能并存投影，B3 不得去重或作 duplicate verdict。"""
    node = _node(10, source_value=10)
    first = _leaf(1, source_value=10, node_key=node.node_key,
                  side=V4_PROVENANCE_SIDE_TRAINING, content_value="same")
    second = _leaf(2, source_value=10, node_key=node.node_key,
                   side=V4_PROVENANCE_SIDE_HELD_OUT, content_value="same")
    work, direct, closure = _build_pipeline(
        tmp_path,
        name="same",
        nodes=(node,),
        leaves=(first, second),
    )
    budget = _projection_budget()

    result = build_v4_provenance_leaf_projections(
        direct, closure, work, budget=budget, logical_stage_name="same.v1")

    source = tuple(map(
        decode_v4_provenance_source_ref_projection_record,
        _read_records(work, result.source_ref_projection_stream, budget),
    ))
    content = tuple(map(
        decode_v4_provenance_content_projection_record,
        _read_records(work, result.content_projection_stream, budget),
    ))
    lineage = tuple(map(
        decode_v4_provenance_lineage_projection_record,
        _read_records(work, result.lineage_projection_stream, budget),
    ))
    assert len(source) == len(content) == len(lineage) == 2
    assert {item[2].consumed_input_identity for item in source} == {
        first.consumed_input_identity, second.consumed_input_identity}
    assert {item[2].consumed_input_identity for item in content} == {
        first.consumed_input_identity, second.consumed_input_identity}
    assert {item[4].consumed_input_identity for item in lineage} == {
        first.consumed_input_identity, second.consumed_input_identity}
    assert not hasattr(result, "duplicate_verdict")
    assert not hasattr(result, "conflict_verdict")


@pytest.mark.parametrize("target", ("b1-parent", "b2-edge", "b2-frontier"))
def test_revalidates_nonprojection_b1_and_b2_descriptors_before_any_b3_stage(tmp_path, target):
    """即使不参与最终 projection 的 B1/B2 descriptor 被替换，B3 也必须在输出前拒绝。"""
    work, direct, closure, _nodes, _leaves = _chain_fixture(tmp_path, f"drift-{target}")
    descriptor = {
        "b1-parent": direct.parent_request_stream,
        "b2-edge": closure.edge_by_child_stream,
        "b2-frontier": closure.empty_frontier_stream,
    }[target]
    replacement = Path(f"replacement-{target}.pifrs")
    _write_sealed(work, replacement, ((99_999,),))
    os.replace(work.path / replacement, work.path / descriptor.work_relative_path)

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableProjectionError,
                       match="漂移|descriptor"):
        build_v4_provenance_leaf_projections(
            direct,
            closure,
            work,
            budget=_projection_budget(),
            logical_stage_name=f"drift-{target}.v1",
        )
    assert not (work.path / "stage-14-projection-known-by-ancestor-sort").exists()


def _write_b2_descriptor(root, stage: str, record: tuple[int, ...]):
    """为 forged B2 result 写一条封存 P0，并返回完整 closure descriptor。"""
    stage_root = create_new_run_root(
        root.path / stage,
        require_k_drive=False,
        label="B3 forged closure stage",
    )
    relative = Path("known.pifrs")
    footer = _write_sealed(stage_root, relative, (record,))
    payload = (stage_root.path / relative).read_bytes()
    return ConversationHeldOutV4ProvenanceClosureStreamDescriptor(
        V4_PROVENANCE_CLOSURE_STREAM_KNOWN,
        Path(stage) / relative,
        footer,
        KRunFileDigest(len(payload), tuple(hashlib.sha256(payload).digest())),
    )


@pytest.mark.parametrize(
    "forgery",
    ("snapshot", "unknown-ancestor", "existing-non-ancestor"),
)
def test_rejects_forged_b2_known_ancestor_that_passes_its_own_physical_descriptor(tmp_path, forgery):
    """B3 必须逐 record 对齐 B1 独立 replay closure，而非只验 P0 SHA 或 endpoint 存在。"""
    work, direct, closure, nodes, _leaves = _chain_fixture(tmp_path, f"forged-{forgery}")
    known_records = _read_records(work, closure.known_ancestor_stream,
                                  _projection_budget())
    descendant, ancestor, snapshot = decode_v4_provenance_closure_pair_record(
        known_records[0])
    if forgery == "snapshot":
        forged_ancestor = ancestor
        forged_snapshot = _snapshot(
            ancestor.components[0],
            revision="forged-revision",
        )
        assert forged_snapshot.source_ref == snapshot.source_ref
        assert forged_snapshot != snapshot
    elif forgery == "unknown-ancestor":
        forged_ancestor = _key(999)
        forged_snapshot = _snapshot(999)
    else:
        descendant = nodes[0].node_key
        forged_ancestor = nodes[1].node_key
        forged_snapshot = nodes[1].snapshot
        assert descendant != forged_ancestor
    forged = encode_v4_provenance_closure_pair_record(
        descendant, forged_ancestor, forged_snapshot)
    known_descriptor = _write_b2_descriptor(
        work, f"forged-known-{forgery}", forged)
    forged_result = ConversationHeldOutV4ProvenanceClosureResult(
        closure.logical_stage_name,
        closure.direct_dag_stable_key,
        closure.edge_by_child_stream,
        known_descriptor,
        closure.empty_frontier_stream,
        closure.node_count,
        closure.direct_edge_count,
        1,
        closure.closure_round_count,
        closure.materialized_physical_bytes,
        closure.budget,
    )

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableProjectionError,
                       match="replay closure|不一致"):
        build_v4_provenance_leaf_projections(
            direct,
            forged_result,
            work,
            budget=_projection_budget(),
            logical_stage_name=f"forged-{forgery}.v1",
        )


def test_rejects_reopened_work_root_lineage_budget_and_existing_b3_stage(tmp_path):
    """B3 不能接受重开 capability、扩大 lineage 乘积预算或覆盖已有 sibling stage。"""
    work, direct, closure, _nodes, _leaves = _chain_fixture(tmp_path, "bounds")
    reopened = open_existing_run_root(
        work.path,
        require_k_drive=False,
        label="B3 reopened work",
    )
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        build_v4_provenance_leaf_projections(
            direct, closure, reopened, budget=_projection_budget(),
            logical_stage_name="reopened.v1")

    too_small = _projection_budget(
        max_leaf_count=3,
        max_lineage_projection_record_count=3,
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableProjectionBudgetExceeded,
                       match="lineage projection"):
        build_v4_provenance_leaf_projections(
            direct, closure, work, budget=too_small, logical_stage_name="small.v1")

    work2, direct2, closure2, _nodes2, _leaves2 = _chain_fixture(tmp_path, "collision")
    budget = _projection_budget()
    build_v4_provenance_leaf_projections(
        direct2, closure2, work2, budget=budget, logical_stage_name="collision.v1")
    with pytest.raises(KRunBoundaryError, match="已存在|必须为空|必须此前不存在"):
        build_v4_provenance_leaf_projections(
            direct2, closure2, work2, budget=budget, logical_stage_name="collision.v1")


def test_is_deterministic_and_does_not_cross_p2_boundary(tmp_path):
    """相同 typed inputs 的独立 work roots 必须同 identity，模块不引入 runtime/发布依赖。"""
    first_work, first_direct, first_closure, _nodes, _leaves = _chain_fixture(
        tmp_path, "deterministic-first", logical_name="deterministic-input")
    second_work, second_direct, second_closure, _nodes, _leaves = _chain_fixture(
        tmp_path, "deterministic-second", logical_name="deterministic-input")
    budget = _projection_budget()
    first = build_v4_provenance_leaf_projections(
        first_direct, first_closure, first_work,
        budget=budget, logical_stage_name="deterministic.v1")
    second = build_v4_provenance_leaf_projections(
        second_direct, second_closure, second_work,
        budget=budget, logical_stage_name="deterministic.v1")
    assert first.stable_key() == second.stable_key()
    assert first.source_ref_projection_stream.stable_key() == (
        second.source_ref_projection_stream.stable_key())
    assert first.content_projection_stream.stable_key() == (
        second.content_projection_stream.stable_key())
    assert first.lineage_projection_stream.stable_key() == (
        second.lineage_projection_stream.stable_key())

    source_path = Path(projection_module.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    local_imports: list[str] = []
    path_open_calls: list[int] = []
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


def _tree_file_identity(root: Path) -> tuple[tuple[str, bytes], ...]:
    """记录 test transport work tree 的所有文件字节，证明只读回放无写入。"""
    return tuple(sorted(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).digest())
        for path in root.rglob("*")
        if path.is_file()
    ))


def _single_side_projection(tmp_path: Path, name: str):
    """构造只含 TRAIN leaf 的独立 B3 result，供指定 side 的回读边界使用。"""
    root = _node(10, source_value=10)
    tip = _node(20, source_value=20, parents=(root.node_key,))
    leaves = (
        _leaf(1, source_value=10, node_key=root.node_key,
              side=V4_PROVENANCE_SIDE_TRAINING),
        _leaf(2, source_value=20, node_key=tip.node_key,
              side=V4_PROVENANCE_SIDE_TRAINING),
    )
    work, direct, closure = _build_pipeline(
        tmp_path, name=name, nodes=(root, tip), leaves=leaves)
    result = build_v4_provenance_leaf_projections(
        direct, closure, work, budget=_projection_budget(),
        logical_stage_name=f"{name}.b3")
    return work, result


def test_revalidates_all_sealed_projection_descriptors_read_only_for_expected_side(tmp_path):
    """P2-B3 回读必须逐条消费三路封存输出，并保持 work tree 完全不变。"""
    work, result = _single_side_projection(tmp_path, "readback")
    before = _tree_file_identity(work.path)

    returned = revalidate_v4_provenance_leaf_projections(
        result, work, expected_side=V4_PROVENANCE_SIDE_TRAINING)

    assert returned is result
    assert returned.stable_key() == result.stable_key()
    assert _tree_file_identity(work.path) == before


def test_revalidation_accepts_reopened_sealed_work_capability(tmp_path):
    """后续 P3-B 进程可仅以 open-existing capability 回读封存 B3 输出。"""
    work, result = _single_side_projection(tmp_path, "readback-reopened")
    reopened = open_existing_run_root(
        work.path,
        require_k_drive=False,
        label="B3 readback reopened test transport",
    )
    before = _tree_file_identity(work.path)

    assert revalidate_v4_provenance_leaf_projections(
        result, reopened, expected_side=V4_PROVENANCE_SIDE_TRAINING) is result
    assert _tree_file_identity(work.path) == before


def test_revalidation_rejects_wrong_side_and_descriptor_physical_drift(tmp_path):
    """指定 side 和冻结 P0 physical 身份均是回读硬边界。"""
    work, result = _single_side_projection(tmp_path, "readback-drift")
    before = _tree_file_identity(work.path)
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableProjectionError,
                       match="side.*expected_side"):
        revalidate_v4_provenance_leaf_projections(
            result, work, expected_side=V4_PROVENANCE_SIDE_HELD_OUT)
    assert _tree_file_identity(work.path) == before

    target = work.path / result.content_projection_stream.work_relative_path
    with target.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableProjectionError,
                       match="漂移|physical"):
        revalidate_v4_provenance_leaf_projections(
            result, work, expected_side=V4_PROVENANCE_SIDE_TRAINING)
