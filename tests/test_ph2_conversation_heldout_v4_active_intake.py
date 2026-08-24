"""P3-A 活动摄取边界与 roster binding 的有界 test transport 专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_active_intake as intake_module
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
from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeBudget,
    ConversationHeldOutV4ActiveIntakeError,
    ConversationHeldOutV4ActiveIntakeInput,
    ConversationHeldOutV4ActiveIntakeObservationBinding,
    ConversationHeldOutV4ActiveIntakeRosterFactories,
    ConversationHeldOutV4ActiveIntakeSourceBinding,
    ConversationHeldOutV4ActiveIntakeTeacherBinding,
    ConversationHeldOutV4ActiveIntakeTeacherPolicy,
    V4_ACTIVE_INTAKE_STATUS_TEST_ONLY,
    load_v4_active_intake_publication,
    materialize_v4_active_intake,
    revalidate_v4_active_intake,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4LineageNode,
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_UPSTREAM_SHA256,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceCatalogBudget,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import (
    CanonicalJsonObject,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_dataset_owner_records import (
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_records import (
    ObservationRecord,
    SourceRefRecord,
)
from pure_integer_ai.storage.integer_external_sort import IntegerExternalSortBudget
from pure_integer_ai.storage.k_run_boundary import (
    create_new_run_root,
    open_plain_binary,
    write_exclusive_bytes,
)
from pure_integer_ai.storage.integer_codec import IntegerFramedStreamReader


def _sha256(value: bytes) -> tuple[int, ...]:
    """构造完整 SHA-256 字节 tuple，模拟明确的实际消费内容 identity。"""
    return tuple(hashlib.sha256(value).digest())


def _key(value: int) -> StableRecordKey:
    """构造短小但完整的 PH2 StableRecordKey。"""
    return StableRecordKey((value, value + 1000))


def _protocol(value: int) -> ProtocolKey:
    """构造完整 ProtocolKey，不让测试依赖字符串或摘要身份。"""
    return ProtocolKey((value, value + 10_000))


def _payload(value: dict) -> CanonicalJsonObject:
    """构造 PH2 record 的 typed JSON；它不能进入 P3 mapping artifact。"""
    return CanonicalJsonObject.from_value(value)


def _source_ref(value: int) -> SourceRef:
    """生成完整 P1 SourceRef，刻意区别于 PH2 StableRecordKey。"""
    return SourceRef(
        40 + value,
        500 + value,
        value,
        OwnerScope(3, 7, 0, VISIBILITY_USER),
        VersionBundle(
            CorpusVersion(10 + value),
            ParserVersion(20 + value),
            PrimitiveVersion(30 + value),
            CurriculumVersion(40 + value),
        ),
    )


def _source(
        record_id: int, cluster_id: int, *, snapshot_id: str = "snapshot-v1",
        ) -> SourceRefRecord:
    """建立与 P1 snapshot 精确对应的最小公开 source record。"""
    source_bytes = f"source-{record_id}".encode("utf-8")
    upstream = hashlib.sha256(f"upstream-{record_id}".encode("utf-8")).hexdigest()
    local = hashlib.sha256(source_bytes).hexdigest()
    return SourceRefRecord(
        1, 1, 1,
        _key(60_001), _key(60_002), _key(record_id),
        "P3_TEST_SOURCE", snapshot_id, f"revision-{record_id}",
        f"https://example.invalid/p3/source/{record_id}",
        f"p3/source/{record_id}", f"sha256:{upstream}", local,
        "CC0-1.0", "PUBLIC", "P3 test source", 1,
        _payload({"document": record_id, "byte_start": 0, "byte_end": 1}),
        0, _key(cluster_id),
    )


def _snapshot(source: SourceRefRecord, source_ref: SourceRef) -> ConversationHeldOutV4SnapshotIdentity:
    """从 source record 的公开元数据构造唯一合法的显式 P1 snapshot binding。"""
    algorithm, upstream = source.upstream_checksum.split(":", 1)
    assert algorithm == "sha256"
    return ConversationHeldOutV4SnapshotIdentity(
        source_ref,
        tuple(ord(item) for item in source.official_url),
        tuple(ord(item) for item in source.revision_id),
        tuple(ord(item) for item in source.snapshot_id),
        V4_PROVENANCE_UPSTREAM_SHA256,
        tuple(bytes.fromhex(upstream)),
        tuple(bytes.fromhex(source.local_sha256)),
        _protocol(100),
        _protocol(200),
        _protocol(300),
    )


def _observation(record_id: int, source: SourceRefRecord, *, split: str) -> ObservationRecord:
    """构造包含刻意可见 payload 标记的 Observation，检查 P3 不会落正文。"""
    return ObservationRecord(
        1, 1, 1,
        _key(60_001), _key(60_002), _key(record_id),
        "W-01", "p3-active-intake", split, "zh", "typed-proposition",
        source.stable_key, "CC0-1.0",
        _key(70_001), _key(70_002), _key(70_003), _key(70_004),
        "forming", "support", "Proposition",
        _payload({"opaque_payload": "P3-OBSERVATION-BODY-MUST-NOT-LEAK"}),
        "NONE", None, (), 1,
    )


def _teacher(
        record_id: int, observation: ObservationRecord, source: SourceRefRecord,
        ) -> TeacherEvidenceRecord:
    """构造 permit 可接受的 teacher record；其 evidence 正文同样不得落 P3 mapping。"""
    return TeacherEvidenceRecord(
        1, 1, 1,
        _key(60_001), _key(60_002), _key(record_id),
        observation.stable_key, "FORM_REVEAL",
        _payload({"opaque_evidence": "P3-TEACHER-BODY-MUST-NOT-LEAK"}),
        source.stable_key, "W-01", 0, _key(90_001),
    )


def _external_sort_budget() -> IntegerExternalSortBudget:
    """提供覆盖小型 P3-A P0/sort 物化的冻结外排预算。"""
    return IntegerExternalSortBudget(
        max_input_file_count=1,
        max_input_physical_bytes=256 * 1024,
        max_input_record_count=64,
        max_input_payload_bytes=128 * 1024,
        max_record_payload_bytes=32 * 1024,
        max_batch_record_count=16,
        max_batch_payload_bytes=64 * 1024,
        max_batch_sort_key_bytes=32 * 1024,
        max_temporary_run_count=64,
        max_temporary_record_count=256,
        max_temporary_payload_bytes=512 * 1024,
        max_temporary_physical_bytes=1024 * 1024,
        max_output_physical_bytes=256 * 1024,
        merge_fan_in=2,
        max_open_files=3,
        max_merge_pass_count=16,
    )


def _budget() -> ConversationHeldOutV4ActiveIntakeBudget:
    """构造与三类小型 P3 roster 一致的 catalog 与 external-sort 联合预算。"""
    return ConversationHeldOutV4ActiveIntakeBudget(
        max_lineage_node_count=16,
        max_source_record_count=16,
        max_observation_record_count=16,
        max_teacher_record_count=16,
        max_mapping_record_payload_bytes=32 * 1024,
        max_stream_payload_bytes=128 * 1024,
        max_stream_physical_bytes=256 * 1024,
        max_manifest_bytes=64 * 1024,
        catalog_budget=ConversationHeldOutV4ProvenanceCatalogBudget(
            max_total_shards=4,
            max_physical_bytes_per_shard=256 * 1024,
            max_total_physical_bytes=512 * 1024,
            max_frame_bytes=32 * 1024,
            max_records_per_stream=64,
            max_total_payload_bytes_per_stream=128 * 1024,
        ),
        external_sort_budget=_external_sort_budget(),
    )


def _root(tmp_path: Path, name: str):
    """创建明确 D 盘 test transport root；生产默认 K 盘策略不在本专项绕过。"""
    return create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label=f"P3-A test {name}",
    )


def _request(tmp_path: Path, *,
             source: SourceRefRecord | None = None,
             observation: ObservationRecord | None = None,
             teacher: TeacherEvidenceRecord | None = None,
             source_split: str = "train",
             observation_content: bytes = b"actual observation content",
             teacher_content: bytes = b"actual teacher content",
             nodes: tuple[ConversationHeldOutV4LineageNode, ...] | None = None,
             source_bindings: tuple[ConversationHeldOutV4ActiveIntakeSourceBinding, ...] | None = None,
             observation_bindings: tuple[ConversationHeldOutV4ActiveIntakeObservationBinding, ...] | None = None,
             teacher_bindings: tuple[ConversationHeldOutV4ActiveIntakeTeacherBinding, ...] | None = None,
             ) -> ConversationHeldOutV4ActiveIntakeInput:
    """生成完整一次 test-only P3-A request，可由测试精确替换某条 binding。"""
    actual_source = source or _source(10_001, 80_001)
    source_ref = _source_ref(1)
    snapshot = _snapshot(actual_source, source_ref)
    node = ConversationHeldOutV4LineageNode(_protocol(400), snapshot)
    actual_observation = observation or _observation(20_001, actual_source, split="train")
    actual_teacher = teacher or _teacher(30_001, actual_observation, actual_source)
    source_binding = ConversationHeldOutV4ActiveIntakeSourceBinding(
        actual_source, source_ref, snapshot, node.node_key, source_split, _protocol(500))
    observation_binding = ConversationHeldOutV4ActiveIntakeObservationBinding(
        actual_observation, source_ref, _sha256(observation_content), snapshot,
        node.node_key, actual_source.source_cluster_key, _protocol(500))
    teacher_binding = ConversationHeldOutV4ActiveIntakeTeacherBinding(
        actual_teacher, source_ref, _sha256(teacher_content), snapshot,
        node.node_key, "train", actual_source.source_cluster_key, "CC0-1.0")
    policy = ConversationHeldOutV4ActiveIntakeTeacherPolicy(
        (_protocol_from_owner(actual_teacher.owner_key),), ("W-01",), 0)
    return ConversationHeldOutV4ActiveIntakeInput(
        _root(tmp_path, "staging"),
        _root(tmp_path, "work"),
        _root(tmp_path, "publication"),
        _protocol(700),
        _protocol(800),
        policy,
        ConversationHeldOutV4ActiveIntakeRosterFactories(
            lambda: nodes if nodes is not None else (node,),
            lambda: source_bindings if source_bindings is not None else (source_binding,),
            lambda: (observation_bindings if observation_bindings is not None
                     else (observation_binding,)),
            lambda: teacher_bindings if teacher_bindings is not None else (teacher_binding,),
        ),
        _budget(),
        "p3a-test",
    )


def _protocol_from_owner(value: StableRecordKey) -> ProtocolKey:
    """以完整 teacher owner stable key 形成 policy 中的显式 ProtocolKey。"""
    return ProtocolKey(value.components)


def test_active_intake_materializes_test_only_catalog_mapping_and_manifest(tmp_path):
    """D 盘仅经显式 test transport 生成 exact 四文件 publication，绝不升级 coverage。"""
    request = _request(tmp_path)
    result = materialize_v4_active_intake(request)

    assert result.status == V4_ACTIVE_INTAKE_STATUS_TEST_ONLY
    assert result.catalog.root == request.publication_run_root
    assert result.catalog.node_streams[0].relative_path == Path("catalog/nodes.pifrs")
    assert result.catalog.leaf_streams[0].relative_path == Path("catalog/leaves.pifrs")
    assert result.mapping_stream.relative_path == Path("mapping/records.pifrs")
    assert result.counts.integer_stream() == (1, 1, 1, 1)
    assert {
        path.relative_to(request.publication_run_root.path).as_posix()
        for path in request.publication_run_root.path.rglob("*") if path.is_file()
    } == {
        "catalog/nodes.pifrs", "catalog/leaves.pifrs", "mapping/records.pifrs",
        "manifest.pii",
    }
    assert revalidate_v4_active_intake(result) == result

    mapping_payload = (
        request.publication_run_root.path / "mapping" / "records.pifrs").read_bytes()
    assert b"P3-OBSERVATION-BODY-MUST-NOT-LEAK" not in mapping_payload
    assert b"P3-TEACHER-BODY-MUST-NOT-LEAK" not in mapping_payload
    with open_plain_binary(
            request.publication_run_root,
            Path("mapping/records.pifrs"),
            label="P3-A mapping test read") as stream:
        with IntegerFramedStreamReader.from_open_binary(
                stream,
                path=Path("mapping/records.pifrs"),
                max_frame_bytes=32 * 1024,
                max_record_count=16,
                max_total_payload_bytes=128 * 1024,
        ) as reader:
            mappings = tuple(
                intake_module.ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(item)
                for item in reader)
            reader.finish()
    assert {
        item.content_sha256 for item in mappings
    } >= {
        _sha256(b"actual observation content"),
        _sha256(b"actual teacher content"),
    }


def test_active_intake_publication_loader_returns_current_mapping_members_only(tmp_path):
    """公开 loader 只返回 current P3-A stream 的无正文成员，并在篡改后先拒绝。"""
    request = _request(tmp_path)
    result = materialize_v4_active_intake(request)

    publication = load_v4_active_intake_publication(result)

    assert publication.result == result
    assert len(publication.mappings) == result.counts.mapping_record_count
    mapping_path = request.publication_run_root.path / "mapping" / "records.pifrs"
    with open_plain_binary(
            request.publication_run_root,
            Path("mapping/records.pifrs"),
            label="P3-A public loader independent membership read",
    ) as stream:
        with IntegerFramedStreamReader.from_open_binary(
                stream,
                path=Path("mapping/records.pifrs"),
                max_frame_bytes=result.budget.max_mapping_record_payload_bytes,
                max_record_count=result.counts.mapping_record_count,
                max_total_payload_bytes=result.budget.max_stream_payload_bytes,
        ) as reader:
            direct_members = tuple(
                intake_module.ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(raw)
                for raw in reader
            )
            reader.finish()
    assert publication.mappings == direct_members

    mapping_path.write_bytes(mapping_path.read_bytes() + b"x")
    with pytest.raises(ConversationHeldOutV4ActiveIntakeError):
        load_v4_active_intake_publication(result)


def test_active_intake_rejects_missing_source_binding_before_publication(tmp_path):
    """Observation 引用一个未登记 source record 时，source merge join 必须 fail closed。"""
    source = _source(10_001, 80_001)
    broken_observation = _observation(20_001, source, split="train")
    broken_observation = replace(broken_observation, source_ref_key=_key(77_777))
    request = _request(tmp_path, source=source, observation=broken_observation)

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="source join"):
        materialize_v4_active_intake(request)
    assert not (request.publication_run_root.path / "manifest.pii").exists()


def test_active_intake_rejects_cluster_cross_split(tmp_path):
    """同一 source cluster 即使 SourceRef 不同，也不能一侧 train 一侧 held-out。"""
    source_a = _source(10_001, 80_001)
    source_b = _source(10_002, 80_001)
    ref_a = _source_ref(1)
    ref_b = _source_ref(2)
    snapshot_a = _snapshot(source_a, ref_a)
    snapshot_b = _snapshot(source_b, ref_b)
    node_a = ConversationHeldOutV4LineageNode(_protocol(401), snapshot_a)
    node_b = ConversationHeldOutV4LineageNode(_protocol(402), snapshot_b)
    observation_a = _observation(20_001, source_a, split="train")
    observation_b = _observation(20_002, source_b, split="held_out")
    binding_a = ConversationHeldOutV4ActiveIntakeSourceBinding(
        source_a, ref_a, snapshot_a, node_a.node_key, "train", _protocol(501))
    binding_b = ConversationHeldOutV4ActiveIntakeSourceBinding(
        source_b, ref_b, snapshot_b, node_b.node_key, "held_out", _protocol(502))
    obs_binding_a = ConversationHeldOutV4ActiveIntakeObservationBinding(
        observation_a, ref_a, _sha256(b"train input"), snapshot_a, node_a.node_key,
        source_a.source_cluster_key, _protocol(501))
    obs_binding_b = ConversationHeldOutV4ActiveIntakeObservationBinding(
        observation_b, ref_b, _sha256(b"heldout input"), snapshot_b, node_b.node_key,
        source_b.source_cluster_key, _protocol(502))
    request = _request(
        tmp_path,
        source=source_a,
        observation=observation_a,
        nodes=(node_a, node_b),
        source_bindings=(binding_a, binding_b),
        observation_bindings=(obs_binding_a, obs_binding_b),
        teacher_bindings=(),
    )

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="cluster.*split"):
        materialize_v4_active_intake(request)


def test_active_intake_rejects_teacher_outside_frozen_policy(tmp_path):
    """任何未声明 owner/stage/withdrawal 的 teacher 都不能穿过 intake 边界。"""
    request = _request(tmp_path)
    denied_policy = ConversationHeldOutV4ActiveIntakeTeacherPolicy(
        (_protocol(99_999),), ("W-01",), 0)
    request = replace(request, teacher_policy=denied_policy)

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="permit policy"):
        materialize_v4_active_intake(request)


def test_active_intake_rejects_unknown_lineage_node_without_in_memory_lookup(tmp_path):
    """binding 指向没有被 node factory 物化的 node 时，外排 merge join 必须拒绝。"""
    request = _request(tmp_path)
    source_binding = tuple(request.roster_factories.source_binding_factory())[0]
    observation_binding = tuple(request.roster_factories.observation_binding_factory())[0]
    unknown_node = _protocol(99_001)
    broken_source = replace(source_binding, lineage_node_key=unknown_node)
    broken_observation = replace(observation_binding, lineage_node_key=unknown_node)
    factories = replace(
        request.roster_factories,
        source_binding_factory=lambda: (broken_source,),
        observation_binding_factory=lambda: (broken_observation,),
        teacher_binding_factory=lambda: (),
    )
    request = replace(request, roster_factories=factories)

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="未登记 lineage node"):
        materialize_v4_active_intake(request)


def test_active_intake_rejects_duplicate_record_key(tmp_path):
    """重复 active record key 不能借 external-sort 的稳定并列顺序混过。"""
    request = _request(tmp_path)
    source_binding = tuple(request.roster_factories.source_binding_factory())[0]
    factories = replace(
        request.roster_factories,
        source_binding_factory=lambda: (source_binding, source_binding),
    )
    request = replace(request, roster_factories=factories)

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="mapping record key"):
        materialize_v4_active_intake(request)


def test_active_intake_rejects_duplicate_lineage_node(tmp_path):
    """同一 node key 的两份 P0 node 也不能作为重复来源的隐性旁路。"""
    request = _request(tmp_path)
    node = tuple(request.roster_factories.lineage_node_factory())[0]
    request = replace(
        request,
        roster_factories=replace(
            request.roster_factories,
            lineage_node_factory=lambda: (node, node),
        ),
    )

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="sorted node key"):
        materialize_v4_active_intake(request)


def test_active_intake_rejects_extra_publication_file_after_manifest(tmp_path):
    """manifest-last 后的额外文件使 exact closure 失效，不能被回读 API 忽略。"""
    request = _request(tmp_path)
    result = materialize_v4_active_intake(request)
    write_exclusive_bytes(
        request.publication_run_root,
        "extra.pifrs",
        b"unexpected",
        label="P3-A extra publication test file")

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="catalog 回读失败"):
        revalidate_v4_active_intake(result)


def test_active_intake_manifest_and_mapping_drift_fail_closed(tmp_path):
    """任何 publication payload 变化都会在 P3-B 可调用的回读入口被拒绝。"""
    request = _request(tmp_path)
    result = materialize_v4_active_intake(request)
    mapping_path = request.publication_run_root.path / "mapping" / "records.pifrs"
    payload = mapping_path.read_bytes()
    mapping_path.write_bytes(payload + b"x")

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError):
        revalidate_v4_active_intake(result)


def test_active_intake_rejects_nonfresh_prepopulated_root(tmp_path):
    """P3-A 不得接管、覆盖或继续预存 output；每次必须有新 capability。"""
    request = _request(tmp_path)
    write_exclusive_bytes(
        request.publication_run_root,
        "preexisting.bin",
        b"not-an-active-intake-output",
        label="P3-A preexisting test file")

    with pytest.raises(ConversationHeldOutV4ActiveIntakeError, match="新建/隔离"):
        materialize_v4_active_intake(request)


def test_active_intake_module_has_no_runtime_or_unsafe_storage_dependency():
    """P3-A 只能依赖 P0/P1、PH2 record 与 K boundary，不能偷接训练或临时存储。"""
    source_path = Path(intake_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    local_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                "pure_integer_ai"):
            local_imports.append(node.module)
    assert all(all(token not in item for token in (
        "runtime", "candidate", "source_qualification", "private", "formal",
        "v2_generic", "v2_streaming")) for item in local_imports)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint({"tempfile", "sqlite3", "environ", "getenv"})
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"read_bytes", "write_bytes"}
        for node in ast.walk(tree)
    )
