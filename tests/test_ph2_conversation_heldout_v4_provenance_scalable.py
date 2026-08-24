"""DLG-05 v4 R04b P2-A 有界 typed provenance stream catalog 专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable as scalable_module
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
    V4_PROVENANCE_STREAM_CATALOG_SCHEMA,
    V4_PROVENANCE_STREAM_KIND_LEAF,
    V4_PROVENANCE_STREAM_KIND_NODE,
    build_v4_provenance_stream_catalog,
    revalidate_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import IntegerFramedStreamWriter
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    ensure_normal_relative_directory,
    open_exclusive_binary,
    open_existing_run_root,
)


def _scalars(value: str) -> tuple[int, ...]:
    """把可读测试文字编码为未经归一化的 Unicode scalar tuple。"""
    return tuple(ord(character) for character in value)


def _sha1(value: str) -> tuple[int, ...]:
    """生成完整 SHA-1 字节 tuple，供 snapshot 上游身份测试使用。"""
    return tuple(hashlib.sha1(value.encode("utf-8")).digest())


def _sha256(value: str) -> tuple[int, ...]:
    """生成完整 SHA-256 字节 tuple，供 content 与物理身份断言使用。"""
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _bytes_sha256(value: bytes) -> tuple[int, ...]:
    """按原始物理字节计算 SHA-256，避免文本重编码改变 framed stream 身份。"""
    return tuple(hashlib.sha256(value).digest())


def _key(value: int) -> ProtocolKey:
    """构造顺序清晰且完整的 P1 ProtocolKey。"""
    return ProtocolKey((value, value + 1000))


def _source_ref(value: int) -> SourceRef:
    """构造保留 owner 和版本字段的完整 P1 SourceRef。"""
    return SourceRef(
        40 + value,
        500 + value,
        value,
        OwnerScope(
            tenant_id=3,
            user_id=7,
            session_id=0,
            visibility=VISIBILITY_USER,
        ),
        VersionBundle(
            CorpusVersion(11 + value),
            ParserVersion(21 + value),
            PrimitiveVersion(31 + value),
            CurriculumVersion(41 + value),
        ),
    )


def _snapshot(value: int, *, source_value: int) -> ConversationHeldOutV4SnapshotIdentity:
    """构造具备所有固定 identity 字段的最小 snapshot。"""
    return ConversationHeldOutV4SnapshotIdentity(
        _source_ref(source_value),
        _scalars(f"https://example.invalid/provenance/{value}"),
        _scalars(f"revision-{value}"),
        _scalars(f"snapshot-{value}"),
        V4_PROVENANCE_UPSTREAM_SHA1,
        _sha1(f"upstream-{value}"),
        _sha256(f"local-{value}"),
        _key(100 + value),
        _key(200 + value),
        _key(300 + value),
    )


def _node(value: int, *, source_value: int | None = None) -> ConversationHeldOutV4LineageNode:
    """构造无 parent 的单一 node record，供 P2-A 机械输入验证使用。"""
    actual_source = value if source_value is None else source_value
    return ConversationHeldOutV4LineageNode(
        _key(value),
        _snapshot(value, source_value=actual_source),
    )


def _leaf(
        value: int, *, source_value: int, node_key: ProtocolKey,
        ) -> ConversationHeldOutV4ProvenanceLeaf:
    """构造完整 leaf；P2-A 只验证其自身表示与排序，不做 endpoint join。"""
    return ConversationHeldOutV4ProvenanceLeaf(
        V4_PROVENANCE_SIDE_TRAINING,
        _source_ref(source_value),
        _sha256(f"content-{value}"),
        node_key,
        _key(700 + value),
    )


def _root(tmp_path):
    """显式创建 test transport capability，不把临时目录伪装为 K 盘生产路径。"""
    root = open_existing_run_root(
        tmp_path,
        require_k_drive=False,
        label="P2-A test transport root",
    )
    assert root.test_transport is True
    return root


def _write_sealed(root, relative: Path, records: tuple[tuple[int, ...], ...]):
    """经 K boundary 排他写入一个 sealed P0 test stream。"""
    if relative.parent.parts:
        ensure_normal_relative_directory(
            root,
            relative.parent,
            label="P2-A test stream parent",
        )
    with open_exclusive_binary(root, relative, label="P2-A test stream") as stream:
        with IntegerFramedStreamWriter.from_open_binary(
                stream,
                path=root.path / relative,
        ) as writer:
            for record in records:
                writer.append(record)
            return writer.seal()


def _write_unsealed(root, relative: Path, record: tuple[int, ...]) -> None:
    """保留一个没有 footer 的 P0 残片，验证 catalog 不会把它当输入身份。"""
    if relative.parent.parts:
        ensure_normal_relative_directory(
            root,
            relative.parent,
            label="P2-A test partial parent",
        )
    with open_exclusive_binary(root, relative, label="P2-A test partial") as stream:
        writer = IntegerFramedStreamWriter.from_open_binary(
            stream,
            path=root.path / relative,
        )
        writer.append(record)
        writer.close()


def _budget(**overrides) -> ConversationHeldOutV4ProvenanceCatalogBudget:
    """提供能容纳小型专项输入的默认 P2-A 预算。"""
    values = {
        "max_total_shards": 8,
        "max_physical_bytes_per_shard": 64 * 1024,
        "max_total_physical_bytes": 256 * 1024,
        "max_frame_bytes": 16 * 1024,
        "max_records_per_stream": 32,
        "max_total_payload_bytes_per_stream": 32 * 1024,
    }
    values.update(overrides)
    return ConversationHeldOutV4ProvenanceCatalogBudget(**values)


def _input(
        root, *, node_paths: tuple[Path, ...], leaf_paths: tuple[Path, ...] = (),
        budget: ConversationHeldOutV4ProvenanceCatalogBudget | None = None,
        ) -> ConversationHeldOutV4ProvenanceCatalogInput:
    """构造仅含 K capability、相对路径和显式预算的生产形状输入。"""
    return ConversationHeldOutV4ProvenanceCatalogInput(
        root,
        node_paths,
        leaf_paths,
        _budget() if budget is None else budget,
    )


def test_catalog_streams_typed_shards_and_retains_full_physical_identities(tmp_path):
    """多个有序 node/leaf shard 应只形成小 catalog，不聚合 P1 records。"""
    root = _root(tmp_path)
    first_node = _node(10)
    second_node = _node(20)
    first_leaf = _leaf(10, source_value=10, node_key=first_node.node_key)
    second_leaf = _leaf(20, source_value=20, node_key=second_node.node_key)
    ordered_leaves = tuple(sorted((first_leaf, second_leaf), key=lambda item: item.stable_key()))

    node_first_footer = _write_sealed(
        root,
        Path("nodes/000.ints"),
        (first_node.integer_stream(),),
    )
    node_second_footer = _write_sealed(
        root,
        Path("nodes/001.ints"),
        (second_node.integer_stream(),),
    )
    leaf_first_footer = _write_sealed(
        root,
        Path("leaves/000.ints"),
        (ordered_leaves[0].integer_stream(),),
    )
    leaf_second_footer = _write_sealed(
        root,
        Path("leaves/001.ints"),
        (ordered_leaves[1].integer_stream(),),
    )

    catalog_input = _input(
        root,
        node_paths=(Path("nodes/000.ints"), Path("nodes/001.ints")),
        leaf_paths=(Path("leaves/000.ints"), Path("leaves/001.ints")),
    )
    catalog = build_v4_provenance_stream_catalog(catalog_input)

    assert tuple(item.stream_kind for item in catalog.node_streams) == (
        V4_PROVENANCE_STREAM_KIND_NODE,
        V4_PROVENANCE_STREAM_KIND_NODE,
    )
    assert tuple(item.stream_kind for item in catalog.leaf_streams) == (
        V4_PROVENANCE_STREAM_KIND_LEAF,
        V4_PROVENANCE_STREAM_KIND_LEAF,
    )
    assert tuple(item.p0_footer for item in catalog.node_streams) == (
        node_first_footer,
        node_second_footer,
    )
    assert tuple(item.p0_footer for item in catalog.leaf_streams) == (
        leaf_first_footer,
        leaf_second_footer,
    )
    all_streams = catalog.node_streams + catalog.leaf_streams
    assert catalog.budget == catalog_input.budget
    assert catalog.total_physical_byte_count == sum(
        item.physical_byte_count for item in all_streams)
    for item in all_streams:
        payload = (root.path / item.relative_path).read_bytes()
        assert item.physical_byte_count == len(payload)
        assert item.physical_sha256 == _bytes_sha256(payload)


@pytest.mark.parametrize("records", [
    (lambda: (_node(20).integer_stream(), _node(10).integer_stream())),
    (lambda: (_node(10).integer_stream(), _node(10).integer_stream())),
])
def test_catalog_rejects_noncanonical_or_duplicate_node_identity_within_shard(
        tmp_path, records):
    """一个 node stream 内倒序或重复 key 均不能被 catalog 正常化。"""
    root = _root(tmp_path)
    _write_sealed(root, Path("nodes/bad.ints"), records())

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="严格递增"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/bad.ints"),),
        ))


def test_catalog_rejects_cross_shard_order_drift_and_typed_path_alias(tmp_path):
    """声明顺序就是全局顺序，且同一物理 shard 不能同时冒充 node 和 leaf。"""
    root = _root(tmp_path)
    _write_sealed(root, Path("nodes/first.ints"), (_node(20).integer_stream(),))
    _write_sealed(root, Path("nodes/second.ints"), (_node(10).integer_stream(),))

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="跨 shard"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/first.ints"), Path("nodes/second.ints")),
        ))

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="不得重复"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/first.ints"),),
            leaf_paths=(Path("nodes/first.ints"),),
        ))


def test_catalog_rejects_node_leaf_type_confusion_and_unsealed_stream(tmp_path):
    """leaf 不能作为 node 解码，缺 footer 的残片也不能取得 stream identity。"""
    root = _root(tmp_path)
    node = _node(10)
    leaf = _leaf(10, source_value=10, node_key=node.node_key)
    _write_sealed(root, Path("nodes/leaf-as-node.ints"), (leaf.integer_stream(),))
    _write_unsealed(root, Path("nodes/partial.ints"), node.integer_stream())

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="P1 node record"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/leaf-as-node.ints"),),
        ))
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="sealed P0"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/partial.ints"),),
        ))


def test_catalog_defers_leaf_source_ref_endpoint_to_streaming_dag_join(tmp_path):
    """P2-A 不得建立 node 字典；leaf endpoint 一致性留给后续外排 join。"""
    root = _root(tmp_path)
    node = _node(10, source_value=1)
    mismatched_leaf = _leaf(10, source_value=999, node_key=node.node_key)
    _write_sealed(root, Path("nodes/only.ints"), (node.integer_stream(),))
    _write_sealed(root, Path("leaves/deferred.ints"), (mismatched_leaf.integer_stream(),))

    catalog = build_v4_provenance_stream_catalog(_input(
        root,
        node_paths=(Path("nodes/only.ints"),),
        leaf_paths=(Path("leaves/deferred.ints"),),
    ))

    assert len(catalog.node_streams) == 1
    assert len(catalog.leaf_streams) == 1


def test_catalog_stable_key_is_schema_backed_and_excludes_absolute_root(tmp_path):
    """后续计划可绑定路径相对 identity、footer、物理 SHA/size 与预算而不泄露 root。"""
    root = _root(tmp_path)
    node = _node(10)
    _write_sealed(root, Path("nodes/only.ints"), (node.integer_stream(),))
    catalog = build_v4_provenance_stream_catalog(_input(
        root,
        node_paths=(Path("nodes/only.ints"),),
    ))
    other_directory = tmp_path / "other-root"
    other_directory.mkdir()
    other_root = _root(other_directory)
    rebound = replace(catalog, root=other_root)
    changed_identity = replace(
        catalog.node_streams[0],
        physical_sha256=(
            catalog.node_streams[0].physical_sha256[0] ^ 1,
            *catalog.node_streams[0].physical_sha256[1:],
        ),
    )
    changed_physical = replace(catalog, node_streams=(changed_identity,))
    changed_budget = replace(
        catalog,
        budget=_budget(max_total_physical_bytes=512 * 1024),
    )

    assert catalog.integer_stream()[0] == V4_PROVENANCE_STREAM_CATALOG_SCHEMA
    assert catalog.stable_key() == catalog.integer_stream()
    assert catalog.stable_key() == rebound.stable_key()
    assert catalog.stable_key() != changed_physical.stable_key()
    assert catalog.stable_key() != changed_budget.stable_key()


def test_catalog_revalidation_rebuilds_frozen_input_and_rejects_tampered_identity(
        tmp_path):
    """重验只能使用 catalog 自带 root/路径/预算，且结果必须逐整数等于冻结 key。"""
    root = _root(tmp_path)
    node = _node(10)
    _write_sealed(root, Path("nodes/only.ints"), (node.integer_stream(),))
    catalog = build_v4_provenance_stream_catalog(_input(
        root,
        node_paths=(Path("nodes/only.ints"),),
    ))
    tampered_identity = replace(
        catalog.node_streams[0],
        physical_sha256=(
            catalog.node_streams[0].physical_sha256[0] ^ 1,
            *catalog.node_streams[0].physical_sha256[1:],
        ),
    )
    tampered_catalog = replace(catalog, node_streams=(tampered_identity,))

    assert revalidate_v4_provenance_stream_catalog(catalog) == catalog
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="stable key"):
        revalidate_v4_provenance_stream_catalog(tampered_catalog)


def test_catalog_rejects_path_closure_and_directory_input(tmp_path):
    """向上逃逸、重复节点和目录都不能成为 P2-A 声明的输入 closure。"""
    root = _root(tmp_path)
    node = _node(10)
    _write_sealed(root, Path("nodes/only.ints"), (node.integer_stream(),))
    ensure_normal_relative_directory(root, Path("nodes/directory"), label="P2-A directory")

    with pytest.raises(TypeError, match="KRunRoot"):
        ConversationHeldOutV4ProvenanceCatalogInput(
            tmp_path,
            (Path("nodes/only.ints"),),
            (),
            _budget(),
        )
    with pytest.raises(ValueError, match="相对 Path"):
        _input(root, node_paths=(Path("../outside.ints"),))
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="不得重复"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/only.ints"), Path("nodes/only.ints")),
        ))
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="sealed P0"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/directory"),),
        ))


def test_catalog_fails_closed_for_catalog_reader_and_physical_budgets(
        tmp_path, monkeypatch):
    """shard 数、P0 record 数与物理字节任一超过预算都不能产生 catalog。"""
    root = _root(tmp_path)
    first = _node(10)
    second = _node(20)
    _write_sealed(root, Path("nodes/one.ints"), (first.integer_stream(),))
    _write_sealed(root, Path("nodes/two.ints"), (second.integer_stream(),))

    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="shard 数"):
        _input(
            root,
            node_paths=(Path("nodes/one.ints"), Path("nodes/two.ints")),
            budget=_budget(max_total_shards=1),
        )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="sealed P0"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/one.ints"),),
            budget=_budget(max_records_per_stream=0),
        ))
    def must_not_open(*args, **kwargs):
        """已知 lstat 长度超限时，不得再打开或读取该 P0 stream。"""
        raise AssertionError("物理长度超限不得打开 shard")

    monkeypatch.setattr(scalable_module, "open_plain_binary", must_not_open)
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="物理字节"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/one.ints"),),
            budget=_budget(max_physical_bytes_per_shard=1),
        ))


def test_catalog_detects_post_read_physical_identity_drift(tmp_path, monkeypatch):
    """读后 O(1) identity 复核失败时，catalog 不得保留已解码的 shard 身份。"""
    root = _root(tmp_path)
    node = _node(10)
    _write_sealed(root, Path("nodes/only.ints"), (node.integer_stream(),))
    original_require_identity = scalable_module.require_plain_file_identity
    calls = 0

    def drifted_identity(*args, **kwargs):
        """模拟 P0 完整读取后 boundary 观察到路径对象身份已经漂移。"""
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_require_identity(*args, **kwargs)
        raise KRunBoundaryError("injected physical identity drift")

    monkeypatch.setattr(
        scalable_module,
        "require_plain_file_identity",
        drifted_identity,
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceScalableError, match="sealed P0"):
        build_v4_provenance_stream_catalog(_input(
            root,
            node_paths=(Path("nodes/only.ints"),),
        ))
    assert calls == 2


def test_catalog_uses_only_p1_p0_and_k_boundary_not_freeze_or_runtime_dependencies():
    """P2-A 不得把 test-only 全量对象、评测流程或数据存储引入生产入口。"""
    source_path = Path(scalable_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    local_imports = []
    for item in ast.walk(tree):
        if isinstance(item, ast.ImportFrom) and item.module and item.module.startswith(
                "pure_integer_ai"):
            local_imports.append(item.module)
    assert "ConversationHeldOutV4LineageFreeze" not in source
    assert all("r02" not in item and "r03" not in item for item in local_imports)
    assert all(all(token not in item for token in (
        "source_qualification", "runtime", "owner", "private", "formal"))
               for item in local_imports)
    assert "json" not in source
    assert "sqlite" not in source
    assert not any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "open"
        for item in ast.walk(tree)
    )
