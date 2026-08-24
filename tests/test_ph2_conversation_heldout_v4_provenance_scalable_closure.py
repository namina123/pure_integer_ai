"""DLG-05 v4 R04b P2-B2 外部严格祖先闭包专项。"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_closure as closure_module
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
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_UPSTREAM_SHA1,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceCatalogBudget,
    ConversationHeldOutV4ProvenanceCatalogInput,
    build_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_closure import (
    ConversationHeldOutV4ProvenanceClosureBudget,
    ConversationHeldOutV4ProvenanceScalableClosureBudgetExceeded,
    ConversationHeldOutV4ProvenanceScalableClosureError,
    build_v4_provenance_transitive_closure,
    decode_v4_provenance_closure_pair_record,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag import (
    ConversationHeldOutV4ProvenanceDagBudget,
    build_v4_provenance_direct_dag,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
)
from pure_integer_ai.storage.integer_external_sort import IntegerExternalSortBudget
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    create_new_run_root,
    ensure_normal_relative_directory,
    open_existing_run_root,
    open_exclusive_binary,
    open_plain_binary,
)


def _scalars(value: str) -> tuple[int, ...]:
    """把 test-only 文字映射为未归一化的 Unicode scalar tuple。"""
    return tuple(ord(item) for item in value)


def _sha1(value: str) -> tuple[int, ...]:
    """生成完整 SHA-1 tuple，作为 upstream snapshot 身份。"""
    return tuple(hashlib.sha1(value.encode("utf-8")).digest())


def _sha256(value: str) -> tuple[int, ...]:
    """生成完整 SHA-256 tuple，作为 local snapshot 身份。"""
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _key(value: int) -> ProtocolKey:
    """构造完整且有序可读的 ProtocolKey。"""
    return ProtocolKey((value, value + 20_000))


def _source_ref(value: int) -> SourceRef:
    """构造包含 owner/version 的完整十一整数 SourceRef。"""
    return SourceRef(
        60 + value,
        600 + value,
        value,
        OwnerScope(tenant_id=3, user_id=7, session_id=0,
                   visibility=VISIBILITY_USER),
        VersionBundle(
            CorpusVersion(100 + value),
            ParserVersion(200 + value),
            PrimitiveVersion(300 + value),
            CurriculumVersion(400 + value),
        ),
    )


def _node(value: int, *, parents: tuple[ProtocolKey, ...] = ()) -> ConversationHeldOutV4LineageNode:
    """构造完整 snapshot 和直接 parent 的 P1 test transport node。"""
    return ConversationHeldOutV4LineageNode(
        _key(value),
        ConversationHeldOutV4SnapshotIdentity(
            _source_ref(value),
            _scalars(f"https://example.invalid/b2/{value}"),
            _scalars(f"revision-{value}"),
            _scalars(f"snapshot-{value}"),
            V4_PROVENANCE_UPSTREAM_SHA1,
            _sha1(f"upstream-{value}"),
            _sha256(f"local-{value}"),
            _key(1_000 + value),
            _key(2_000 + value),
            _key(3_000 + value),
        ),
        parents,
    )


def _source_root(tmp_path: Path, name: str = "source"):
    """打开显式 D 盘 test transport source capability。"""
    path = tmp_path / name
    path.mkdir()
    return open_existing_run_root(
        path,
        require_k_drive=False,
        label="B2 source test transport",
    )


def _work_root(tmp_path: Path, name: str = "work"):
    """排他创建带私有 token 的 B1/B2 test transport work capability。"""
    return create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label="B2 work test transport",
    )


def _write_sealed(root, relative: Path, records: tuple[tuple[int, ...], ...]) -> None:
    """仅经 K boundary 和 P0 handle bridge 写入封存 catalog shard。"""
    if relative.parent.parts:
        ensure_normal_relative_directory(root, relative.parent,
                                         label="B2 test shard parent")
    stream = open_exclusive_binary(root, relative, label="B2 test shard")
    writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
    try:
        for record in records:
            writer.append(record)
        writer.seal()
    finally:
        writer.close()


def _catalog_budget() -> ConversationHeldOutV4ProvenanceCatalogBudget:
    """提供小型 B2 test transport 的 P2-A 明确输入预算。"""
    return ConversationHeldOutV4ProvenanceCatalogBudget(
        max_total_shards=16,
        max_physical_bytes_per_shard=64 * 1024,
        max_total_physical_bytes=512 * 1024,
        max_frame_bytes=32 * 1024,
        max_records_per_stream=64,
        max_total_payload_bytes_per_stream=48 * 1024,
    )


def _sort_budget(**overrides) -> IntegerExternalSortBudget:
    """小 batch 强制 B1/B2 走真实 bounded external sort。"""
    values = {
        "max_input_file_count": 16,
        "max_input_physical_bytes": 512 * 1024,
        "max_input_record_count": 128,
        "max_input_payload_bytes": 256 * 1024,
        "max_record_payload_bytes": 32 * 1024,
        "max_batch_record_count": 1,
        "max_batch_payload_bytes": 48 * 1024,
        "max_batch_sort_key_bytes": 4 * 1024,
        "max_temporary_run_count": 128,
        "max_temporary_record_count": 2_048,
        "max_temporary_payload_bytes": 1 * 1024 * 1024,
        "max_temporary_physical_bytes": 2 * 1024 * 1024,
        "max_output_physical_bytes": 128 * 1024,
        "merge_fan_in": 2,
        "max_open_files": 3,
        "max_merge_pass_count": 16,
    }
    values.update(overrides)
    return IntegerExternalSortBudget(**values)


def _dag_budget() -> ConversationHeldOutV4ProvenanceDagBudget:
    """覆盖 B1 source node、empty leaf 与 external-sort stage 的冻结预算。"""
    return ConversationHeldOutV4ProvenanceDagBudget(
        max_node_count=32,
        max_leaf_count=32,
        max_parent_request_count=64,
        max_direct_edge_count=64,
        max_parents_per_node=4,
        max_leaves_per_node=4,
        max_record_payload_bytes=32 * 1024,
        max_stream_record_count=128,
        max_stream_payload_bytes=512 * 1024,
        max_stream_physical_bytes=128 * 1024,
        max_total_materialized_physical_bytes=16 * 1024 * 1024,
        max_open_files=3,
        external_sort_budget=_sort_budget(),
    )


def _closure_budget(**overrides) -> ConversationHeldOutV4ProvenanceClosureBudget:
    """提供足以闭合小 DAG、又固定所有 B2 stage 上限的资源预算。"""
    values = {
        "max_node_count": 32,
        "max_direct_edge_count": 64,
        "max_known_pair_count": 256,
        "max_candidate_pair_count_per_round": 256,
        "max_closure_round_count": 8,
        "max_parents_per_node": 4,
        "max_record_payload_bytes": 32 * 1024,
        "max_stream_record_count": 256,
        "max_stream_payload_bytes": 1 * 1024 * 1024,
        "max_stream_physical_bytes": 128 * 1024,
        "max_total_materialized_physical_bytes": 64 * 1024 * 1024,
        "max_open_files": 3,
        "external_sort_budget": _sort_budget(),
    }
    values.update(overrides)
    return ConversationHeldOutV4ProvenanceClosureBudget(**values)


def _build_catalog(root, nodes: tuple[ConversationHeldOutV4LineageNode, ...]):
    """把按 node key 排序的 P1 nodes 写为多 shard P2-A sealed catalog。"""
    node_paths: list[Path] = []
    for index, node in enumerate(sorted(nodes, key=lambda item: item.node_key.components)):
        relative = Path("nodes") / f"{index:03d}.pifrs"
        _write_sealed(root, relative, (node.integer_stream(),))
        node_paths.append(relative)
    return build_v4_provenance_stream_catalog(
        ConversationHeldOutV4ProvenanceCatalogInput(
            root,
            tuple(node_paths),
            (),
            _catalog_budget(),
        ))


def _build_b1(tmp_path: Path, nodes: tuple[ConversationHeldOutV4LineageNode, ...], *, name: str):
    """建立 B1 result 与原 work capability，供 B2 严格在其 stage sibling 中继续施工。"""
    source = _source_root(tmp_path, f"{name}-source")
    catalog = _build_catalog(source, nodes)
    work = _work_root(tmp_path, f"{name}-work")
    result = build_v4_provenance_direct_dag(
        catalog,
        work,
        budget=_dag_budget(),
        logical_stage_name=f"{name}.b1",
    )
    return work, result


def _read_pairs(work, descriptor, budget):
    """从 root-relative B2 descriptor 回读完整 strict ancestor pair，不读取裸字节。"""
    stream = open_plain_binary(work, descriptor.work_relative_path,
                               label="B2 closure descriptor read")
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
        return tuple(decode_v4_provenance_closure_pair_record(item)
                     for item in records)
    finally:
        reader.close()


def test_transitive_closure_covers_all_node_dag_and_excludes_self_pairs(tmp_path):
    """三层 chain 必须得到 strict direct+transitive ancestors，不能只留下 direct edge。"""
    first = _node(10)
    second = _node(20, parents=(first.node_key,))
    third = _node(30, parents=(second.node_key,))
    fourth = _node(40, parents=(third.node_key,))
    work, b1 = _build_b1(tmp_path, (first, second, third, fourth), name="chain")
    budget = _closure_budget()

    result = build_v4_provenance_transitive_closure(
        b1,
        work,
        budget=budget,
        logical_stage_name="chain.b2",
    )

    pairs = _read_pairs(work, result.known_ancestor_stream, budget)
    assert {(descendant, ancestor) for descendant, ancestor, _snapshot in pairs} == {
        (second.node_key, first.node_key),
        (third.node_key, second.node_key),
        (third.node_key, first.node_key),
        (fourth.node_key, third.node_key),
        (fourth.node_key, second.node_key),
        (fourth.node_key, first.node_key),
    }
    assert all(descendant != ancestor for descendant, ancestor, _snapshot in pairs)
    assert result.known_pair_count == 6
    assert result.empty_frontier_stream.p0_footer.record_count == 0
    assert result.closure_round_count == 3
    assert str(work.path) not in str(result.stable_key())


def test_transitive_closure_detects_cycle_in_node_subgraph_without_leaf(tmp_path):
    """即使无 leaf，B2 也必须闭合整张 node 图并在 self candidate 处拒绝环。"""
    first = _node(10, parents=(_key(20),))
    second = _node(20, parents=(first.node_key,))
    work, b1 = _build_b1(tmp_path, (first, second), name="cycle")

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableClosureError,
                       match="谱系环"):
        build_v4_provenance_transitive_closure(
            b1,
            work,
            budget=_closure_budget(),
            logical_stage_name="cycle.b2",
        )


def test_transitive_closure_deduplicates_diamond_paths_without_dropping_snapshot(tmp_path):
    """两个 parent 路径产生同一 strict ancestor 时，known 只保留一个完整 snapshot pair。"""
    root = _node(10)
    left = _node(20, parents=(root.node_key,))
    right = _node(30, parents=(root.node_key,))
    child = _node(40, parents=(left.node_key, right.node_key))
    work, b1 = _build_b1(tmp_path, (root, left, right, child), name="diamond")
    budget = _closure_budget(max_parents_per_node=2)

    result = build_v4_provenance_transitive_closure(
        b1,
        work,
        budget=budget,
        logical_stage_name="diamond.b2",
    )

    pairs = _read_pairs(work, result.known_ancestor_stream, budget)
    child_root = [
        snapshot
        for descendant, ancestor, snapshot in pairs
        if descendant == child.node_key and ancestor == root.node_key
    ]
    assert child_root == [root.snapshot]
    assert result.known_pair_count == 5


def test_transitive_closure_rejects_round_budget_and_reopened_work_capability(tmp_path):
    """未达固定点的 frontier 不能越过 round 上限，也不能由 reopen capability 续跑。"""
    first = _node(10)
    second = _node(20, parents=(first.node_key,))
    third = _node(30, parents=(second.node_key,))
    work, b1 = _build_b1(tmp_path, (first, second, third), name="budget")

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableClosureBudgetExceeded,
                       match="round"):
        build_v4_provenance_transitive_closure(
            b1,
            work,
            budget=_closure_budget(max_closure_round_count=1),
            logical_stage_name="budget.b2",
        )

    reopened = open_existing_run_root(
        work.path,
        require_k_drive=False,
        label="B2 reopened work",
    )
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        build_v4_provenance_transitive_closure(
            b1,
            reopened,
            budget=_closure_budget(),
            logical_stage_name="reopened.b2",
        )

    limited_work, limited_b1 = _build_b1(
        tmp_path,
        (first, second, third),
        name="initial-known-budget",
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableClosureBudgetExceeded,
                       match="initial known"):
        build_v4_provenance_transitive_closure(
            limited_b1,
            limited_work,
            budget=_closure_budget(max_known_pair_count=1),
            logical_stage_name="initial-known-budget.b2",
        )


def test_closure_module_has_no_runtime_json_sqlite_or_unsafe_filesystem_fallbacks():
    """B2 只能依赖 P0/P1/P2-A/B1/K boundary/external sort，不能引入资格或运行时域。"""
    source_path = Path(closure_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    imports = []
    path_open_calls = []
    destructive_calls = []
    for item in ast.walk(tree):
        if isinstance(item, ast.ImportFrom) and item.module and item.module.startswith(
                "pure_integer_ai"):
            imports.append(item.module)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
            if item.func.attr == "open":
                path_open_calls.append(item.lineno)
            if item.func.attr in {"remove", "unlink", "rmdir", "rename", "replace"}:
                destructive_calls.append(item.lineno)
    assert "ConversationHeldOutV4LineageFreeze" not in source
    assert "json" not in source
    assert "sqlite" not in source
    assert not any(any(token in item for token in (
        "runtime", "owner", "private", "formal", "family",
    )) for item in imports)
    assert path_open_calls == []
    assert destructive_calls == []
