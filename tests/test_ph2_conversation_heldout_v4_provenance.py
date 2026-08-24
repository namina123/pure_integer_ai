"""DLG-05 v4 R04b P1 谱系本体的整数、闭合与边界专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_provenance as provenance_module
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
    ConversationHeldOutV4LineageFreeze,
    ConversationHeldOutV4LineageNode,
    ConversationHeldOutV4ProvenanceError,
    ConversationHeldOutV4ProvenanceLeaf,
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_FREEZE_SCHEMA,
    V4_PROVENANCE_SIDE_HELD_OUT,
    V4_PROVENANCE_SIDE_TRAINING,
    V4_PROVENANCE_UPSTREAM_SHA1,
    V4_PROVENANCE_UPSTREAM_SHA256,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import pack_key


def _scalars(value: str) -> tuple[int, ...]:
    """把测试文字转成未经归一化的 scalar tuple。"""
    return tuple(ord(character) for character in value)


def _sha1(value: str) -> tuple[int, ...]:
    """生成仅供 typed-contract 测试使用的完整 SHA-1 字节 tuple。"""
    return tuple(hashlib.sha1(value.encode("utf-8")).digest())


def _sha256(value: str) -> tuple[int, ...]:
    """生成仅供 typed-contract 测试使用的完整 SHA-256 字节 tuple。"""
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _key(value: int) -> ProtocolKey:
    """构造规范排序测试可读的完整 ProtocolKey。"""
    return ProtocolKey((value, value + 1000))


def _snapshot(
        value: int, *, source_ref: SourceRef | None = None,
        upstream_algorithm: int = V4_PROVENANCE_UPSTREAM_SHA1,
        ) -> ConversationHeldOutV4SnapshotIdentity:
    """构造携带全部 snapshot 字段的最小公开 test-only 本体。"""
    upstream = (
        _sha1(f"upstream-{value}")
        if upstream_algorithm == V4_PROVENANCE_UPSTREAM_SHA1
        else _sha256(f"upstream-{value}")
    )
    return ConversationHeldOutV4SnapshotIdentity(
        _source_ref(value) if source_ref is None else source_ref,
        _scalars(f"https://example.invalid/source/{value}"),
        _scalars(f"revision-{value}"),
        _scalars(f"snapshot-{value}"),
        upstream_algorithm,
        upstream,
        _sha256(f"local-{value}"),
        _key(100 + value),
        _key(200 + value),
        _key(300 + value),
    )


def _source_ref(value: int) -> SourceRef:
    """构造保留 owner 与四类版本的完整十一整数 SourceRef。"""
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


def _leaf(
        *, side: int, source_value: int, node_key: ProtocolKey,
        input_value: int, content_value: str,
        ) -> ConversationHeldOutV4ProvenanceLeaf:
    """构造一个完整 leaf，显式绑定实际消费的输入身份。"""
    return ConversationHeldOutV4ProvenanceLeaf(
        side,
        _source_ref(source_value),
        _sha256(content_value),
        node_key,
        _key(input_value),
    )


def _freeze() -> ConversationHeldOutV4LineageFreeze:
    """构造两个根和一个双亲 transform 的规范小型 DAG。"""
    root_a = ConversationHeldOutV4LineageNode(
        _key(10), _snapshot(10, source_ref=_source_ref(1)))
    root_b = ConversationHeldOutV4LineageNode(
        _key(20), _snapshot(20, source_ref=_source_ref(2)))
    transform = ConversationHeldOutV4LineageNode(
        _key(30),
        _snapshot(30, source_ref=_source_ref(3)),
        (root_a.node_key, root_b.node_key),
    )
    leaves = (
        _leaf(
            side=V4_PROVENANCE_SIDE_TRAINING,
            source_value=3,
            node_key=transform.node_key,
            input_value=701,
            content_value="training-content",
        ),
        _leaf(
            side=V4_PROVENANCE_SIDE_HELD_OUT,
            source_value=2,
            node_key=root_b.node_key,
            input_value=702,
            content_value="held-out-content",
        ),
    )
    return ConversationHeldOutV4LineageFreeze(
        (root_a, root_b, transform),
        tuple(sorted(leaves, key=lambda item: item.stable_key())),
    )


def test_lineage_freeze_round_trips_full_integer_ontology_and_source_versions():
    """多 parent DAG、完整 snapshot 和十一整数 SourceRef 必须逐字节可逆。"""
    freeze = _freeze()
    payload = freeze.to_bytes()

    restored = ConversationHeldOutV4LineageFreeze.from_bytes(payload)

    assert restored == freeze
    assert restored.to_bytes() == payload
    assert restored.canonical_sha256() == tuple(hashlib.sha256(payload).digest())
    assert len(restored.nodes) == 3
    assert restored.nodes[2].parent_node_keys == (
        restored.nodes[0].node_key,
        restored.nodes[1].node_key,
    )
    assert restored.nodes[2].stable_key() == restored.nodes[2].integer_stream()
    assert restored.nodes[2].snapshot.stable_key() == (
        restored.nodes[2].snapshot.integer_stream())
    assert any(
        item.source_ref == restored.nodes[2].snapshot.source_ref
        for item in restored.leaves
    )
    assert {item.side for item in restored.leaves} == {
        V4_PROVENANCE_SIDE_TRAINING,
        V4_PROVENANCE_SIDE_HELD_OUT,
    }
    for leaf in restored.leaves:
        assert len(leaf.source_ref.stable_key()) == 11
        assert SourceRef.from_stable_key(leaf.source_ref.stable_key()) == leaf.source_ref
        assert ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(
            leaf.integer_stream()) == leaf


def test_snapshot_identity_retains_every_required_field_and_supports_sha_algorithms():
    """URI、版本、两个 digest、许可/代码身份任一漂移都不得被整数键吞没。"""
    baseline = _snapshot(1)
    sha256_upstream = _snapshot(1, upstream_algorithm=V4_PROVENANCE_UPSTREAM_SHA256)
    variants = (
        replace(baseline, official_uri_scalars=_scalars("https://example.invalid/other")),
        replace(baseline, revision_scalars=_scalars("revision-other")),
        replace(baseline, snapshot_scalars=_scalars("snapshot-other")),
        replace(baseline, upstream_digest=_sha1("upstream-other")),
        replace(baseline, local_sha256=_sha256("local-other")),
        replace(baseline, source_ref=_source_ref(999)),
        replace(baseline, license_review_artifact_identity=_key(401)),
        replace(baseline, ingest_code_identity=_key(402)),
        replace(baseline, transform_code_identity=_key(403)),
        sha256_upstream,
    )

    assert all(item.integer_stream() != baseline.integer_stream()
               for item in variants)
    assert ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
        sha256_upstream.integer_stream()) == sha256_upstream

    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        replace(baseline, official_uri_scalars=(0xD800,))
    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        replace(baseline, revision_scalars=(), snapshot_scalars=())
    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        replace(baseline, upstream_digest_algorithm=99)
    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        replace(baseline, upstream_digest_algorithm=V4_PROVENANCE_UPSTREAM_SHA256)
    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        replace(baseline, local_sha256=(1,) * 31)
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="URI"):
        replace(baseline, official_uri_scalars=_scalars("relative/path"))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="空白"):
        replace(baseline, official_uri_scalars=_scalars("https://example.invalid/a b"))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="空白"):
        replace(baseline, official_uri_scalars=_scalars("https://example.invalid/a\u2009b"))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="控制"):
        replace(baseline, official_uri_scalars=_scalars("https://example.invalid/a\x00b"))

    urn = replace(
        baseline,
        official_uri_scalars=_scalars("urn:example:source:1"),
    )
    assert ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
        urn.integer_stream()) == urn


def test_lineage_node_and_freeze_reject_noncanonical_or_unclosed_dag():
    """parent 排序、闭合、无环与 node 身份唯一性均在 freeze 前 fail closed。"""
    snapshot = _snapshot(50)
    first = _key(10)
    second = _key(20)
    node = _key(30)

    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="排序"):
        ConversationHeldOutV4LineageNode(node, snapshot, (second, first))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="重复"):
        ConversationHeldOutV4LineageNode(node, snapshot, (first, first))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="自身"):
        ConversationHeldOutV4LineageNode(node, snapshot, (node,))

    missing_parent = ConversationHeldOutV4LineageNode(node, snapshot, (first,))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="未闭合"):
        ConversationHeldOutV4LineageFreeze((missing_parent,))

    cycle_a = ConversationHeldOutV4LineageNode(first, _snapshot(51), (second,))
    cycle_b = ConversationHeldOutV4LineageNode(second, _snapshot(52), (first,))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="环"):
        ConversationHeldOutV4LineageFreeze((cycle_a, cycle_b))

    duplicate_a = ConversationHeldOutV4LineageNode(first, _snapshot(53))
    duplicate_b = ConversationHeldOutV4LineageNode(first, _snapshot(54))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="不得重复"):
        ConversationHeldOutV4LineageFreeze((duplicate_a, duplicate_b))


def test_leaf_keeps_same_source_distinct_inputs_but_requires_existing_node():
    """P1 不合并同 SourceRef 的不同 leaf，同时拒绝悬空 node 和 exact 重复。"""
    root = ConversationHeldOutV4LineageNode(
        _key(10), _snapshot(60, source_ref=_source_ref(8)))
    first = _leaf(
        side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=8,
        node_key=root.node_key,
        input_value=801,
        content_value="same-source-content",
    )
    second = _leaf(
        side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=8,
        node_key=root.node_key,
        input_value=802,
        content_value="same-source-content",
    )
    freeze = ConversationHeldOutV4LineageFreeze(
        (root,),
        tuple(sorted((first, second), key=lambda item: item.stable_key())),
    )

    assert len(freeze.leaves) == 2
    assert freeze.leaves[0].source_ref == freeze.leaves[1].source_ref
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="不得重复"):
        ConversationHeldOutV4LineageFreeze((root,), (first, first))
    unknown = _leaf(
        side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=9,
        node_key=_key(999),
        input_value=803,
        content_value="unknown-node",
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="未知"):
        ConversationHeldOutV4LineageFreeze((root,), (unknown,))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="side"):
        replace(first, side=99)
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="digest"):
        replace(first, content_sha256=(1,) * 31)


def _unsafe_source_ref(value: object) -> SourceRef:
    """仅为边界测试绕过 dataclass 构造，模拟带 bool/非严格整数的伪 SourceRef。"""
    source = object.__new__(SourceRef)
    object.__setattr__(source, "source_kind", value)
    object.__setattr__(source, "source_id", 501)
    object.__setattr__(source, "document_id", 1)
    object.__setattr__(source, "owner", OwnerScope(
        tenant_id=3,
        user_id=7,
        session_id=0,
        visibility=VISIBILITY_USER,
    ))
    object.__setattr__(source, "versions", VersionBundle(
        CorpusVersion(1),
        ParserVersion(2),
        PrimitiveVersion(3),
        CurriculumVersion(4),
    ))
    return source


def test_snapshot_and_leaf_reject_bool_or_non_strict_source_ref_before_freeze():
    """snapshot 与 leaf 都必须经完整十一整数键重建，不能接受 bool 或浮点伪来源。"""
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="严格"):
        _snapshot(70, source_ref=_unsafe_source_ref(True))

    root = ConversationHeldOutV4LineageNode(_key(70), _snapshot(70))
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="严格"):
        ConversationHeldOutV4ProvenanceLeaf(
            V4_PROVENANCE_SIDE_TRAINING,
            _unsafe_source_ref(1.5),
            _sha256("non-strict-source"),
            root.node_key,
            _key(870),
        )


def test_lineage_freeze_rejects_cross_source_and_version_leaf_mismatch():
    """leaf 必须精确属于目标 node 的 snapshot，来源或任一 SourceRef 版本漂移都拒绝。"""
    expected_source = _source_ref(80)
    root = ConversationHeldOutV4LineageNode(
        _key(80), _snapshot(80, source_ref=expected_source))
    cross_source = _leaf(
        side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=81,
        node_key=root.node_key,
        input_value=881,
        content_value="cross-source",
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="SourceRef"):
        ConversationHeldOutV4LineageFreeze((root,), (cross_source,))

    changed_versions = VersionBundle(
        CorpusVersion(expected_source.versions.corpus.value + 1),
        expected_source.versions.parser,
        expected_source.versions.primitive,
        expected_source.versions.curriculum,
    )
    cross_version_source = SourceRef(
        expected_source.source_kind,
        expected_source.source_id,
        expected_source.document_id,
        expected_source.owner,
        changed_versions,
    )
    cross_version = ConversationHeldOutV4ProvenanceLeaf(
        V4_PROVENANCE_SIDE_HELD_OUT,
        cross_version_source,
        _sha256("cross-version"),
        root.node_key,
        _key(882),
    )
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="SourceRef"):
        ConversationHeldOutV4LineageFreeze((root,), (cross_version,))


def test_integer_transport_rejects_schema_tail_truncation_and_noncanonical_order():
    """P1 transport 必须拒绝未知 schema、尾字段、损坏字节和不规范列表顺序。"""
    freeze = _freeze()
    values = freeze.integer_stream()

    wrong_schema = list(values)
    wrong_schema[0] = 99
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="schema"):
        ConversationHeldOutV4LineageFreeze.from_integer_stream(tuple(wrong_schema))
    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        ConversationHeldOutV4LineageFreeze.from_integer_stream((*values, 1))
    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        ConversationHeldOutV4LineageFreeze.from_bytes(freeze.to_bytes()[:-1])
    with pytest.raises(ConversationHeldOutV4ProvenanceError):
        ConversationHeldOutV4LineageFreeze.from_bytes(freeze.to_bytes() + b"\x00")

    noncanonical = [V4_PROVENANCE_FREEZE_SCHEMA, len(freeze.nodes)]
    for node in reversed(freeze.nodes):
        pack_key(noncanonical, node.integer_stream())
    noncanonical.append(len(freeze.leaves))
    for leaf in freeze.leaves:
        pack_key(noncanonical, leaf.integer_stream())
    with pytest.raises(ConversationHeldOutV4ProvenanceError, match="排序"):
        ConversationHeldOutV4LineageFreeze.from_integer_stream(
            tuple(noncanonical))


def test_provenance_module_keeps_integer_only_no_io_dependency_boundary():
    """P1 只能依赖 identity/codec 基元，不能接回 R04、runtime、I/O 或秘密域。"""
    source_path = Path(provenance_module.__file__)
    tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
    allowed_local_imports = {
        "pure_integer_ai.cognition.shared.identity": {"SourceRef"},
        "pure_integer_ai.cognition.shared.unicode_representation": {
            "validate_unicode_scalars",
        },
        "pure_integer_ai.experiments.evaluation_protocol": {"ProtocolKey"},
        "pure_integer_ai.storage.integer_codec": {
            "IntegerCodecError",
            "IntegerStreamReader",
            "decode_integer_tuple",
            "encode_integer_tuple",
            "pack_key",
            "strict_integer_tuple",
        },
    }
    dynamic_imports = []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("pure_integer_ai")
                       for alias in node.names)
        elif (isinstance(node, ast.ImportFrom) and node.module
              and node.module.startswith("pure_integer_ai")):
            assert node.level == 0
            assert node.module in allowed_local_imports
            assert {alias.name for alias in node.names} == allowed_local_imports[
                node.module]
            assert all(alias.name != "*" for alias in node.names)
        elif isinstance(node, ast.Call):
            if ((isinstance(node.func, ast.Name) and node.func.id == "__import__")
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module")):
                dynamic_imports.append(node.lineno)

    assert dynamic_imports == []
    assert names.isdisjoint({
        "Path", "open", "read_bytes", "write_bytes", "json", "sqlite3",
        "IntegerFramedStreamWriter", "IntegerFramedStreamReader",
    })
    for forbidden_name in (
            "audit_v4_independent_source_qualification",
            "read_v4_external_input_capsule",
            "run_v4_candidate_runtime",
            "write_v4_runtime_artifact",
            "read_v4_owner_metadata"):
        assert not hasattr(provenance_module, forbidden_name)
