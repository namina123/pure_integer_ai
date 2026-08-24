"""DLG-05 v4 R04b P2-C 跨侧 exact merge 内核专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_merge as merge_module

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
    ConversationHeldOutV4ProvenanceLeaf,
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_SIDE_HELD_OUT,
    V4_PROVENANCE_SIDE_TRAINING,
    V4_PROVENANCE_UPSTREAM_SHA1,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_merge import (
    ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
    ConversationHeldOutV4ProvenanceCrossSideMergeBudgetExceeded,
    ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact,
    ConversationHeldOutV4ProvenanceCrossSideMergeError,
    ConversationHeldOutV4ProvenanceCrossSideOverlapError,
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
    advance_v4_provenance_cross_side_merge,
    build_v4_provenance_cross_side_merge_channel as _raw_build_v4_provenance_cross_side_merge_channel,
    load_v4_provenance_cross_side_merge_publication,
    load_v4_provenance_cross_side_merge_resume_cursor,
    publish_v4_provenance_cross_side_merge,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_projection import (
    ConversationHeldOutV4ProvenanceProjectionBudget,
    ConversationHeldOutV4ProvenanceProjectionResult,
    ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
    encode_v4_provenance_content_projection_record,
    encode_v4_provenance_lineage_projection_record,
    encode_v4_provenance_source_ref_projection_record,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
    decode_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunFileDigest,
    KRunBoundaryError,
    create_new_run_root,
    open_existing_run_root,
    open_exclusive_binary,
    open_plain_binary,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_closure import (
    ConversationHeldOutV4ProvenanceClosureBudget,
)
from pure_integer_ai.storage.integer_external_sort import IntegerExternalSortBudget


def _scalars(value: str) -> tuple[int, ...]:
    """把 test-only 字符串映射为未归一化 scalar tuple。"""
    return tuple(ord(item) for item in value)


def _digest(value: str, algorithm: str = "sha256") -> tuple[int, ...]:
    """生成完整 bytes tuple，避免测试借用短摘要。"""
    return tuple(getattr(hashlib, algorithm)(value.encode("utf-8")).digest())


def _key(value: int) -> ProtocolKey:
    """构造可区分的完整 ProtocolKey。"""
    return ProtocolKey((value, value + 40_000))


def _source_ref(value: int) -> SourceRef:
    """构造完整 owner/version SourceRef。"""
    return SourceRef(
        100 + value,
        1_000 + value,
        value,
        OwnerScope(tenant_id=9, user_id=11, session_id=0,
                   visibility=VISIBILITY_USER),
        VersionBundle(
            CorpusVersion(200 + value),
            ParserVersion(300 + value),
            PrimitiveVersion(400 + value),
            CurriculumVersion(500 + value),
        ),
    )


def _snapshot(value: int) -> ConversationHeldOutV4SnapshotIdentity:
    """构造 lineage primary key 所需的完整 snapshot 本体。"""
    return ConversationHeldOutV4SnapshotIdentity(
        _source_ref(value),
        _scalars(f"https://example.invalid/p2c/{value}"),
        _scalars(f"revision-{value}"),
        _scalars(f"snapshot-{value}"),
        V4_PROVENANCE_UPSTREAM_SHA1,
        _digest(f"upstream-{value}", "sha1"),
        _digest(f"local-{value}"),
        _key(1_000 + value),
        _key(2_000 + value),
        _key(3_000 + value),
    )


def _leaf(value: int, *, source_value: int, side: int,
          content: str) -> ConversationHeldOutV4ProvenanceLeaf:
    """构造带完整 side/source/content/consumed identity 的 B3 leaf。"""
    return ConversationHeldOutV4ProvenanceLeaf(
        side,
        _source_ref(source_value),
        _digest(content),
        _key(5_000 + source_value),
        _key(9_000 + value),
    )


def _budget(**overrides) -> ConversationHeldOutV4ProvenanceCrossSideMergeBudget:
    """提供覆盖小型双 reader/一 writer transport 的明确冻结预算。"""
    values = {
        "max_record_payload_bytes": 32 * 1024,
        "max_stream_record_count": 64,
        "max_stream_payload_bytes": 256 * 1024,
        "max_stream_physical_bytes": 512 * 1024,
        "max_total_materialized_physical_bytes": 2 * 1024 * 1024,
        "max_open_files": 3,
        "max_intent_physical_bytes": 64 * 1024,
        "max_cursor_physical_bytes": 64 * 1024,
        "max_receipt_physical_bytes": 64 * 1024,
        "max_manifest_payload_bytes": 64 * 1024,
        "max_pre_manifest_file_count": 32,
    }
    values.update(overrides)
    return ConversationHeldOutV4ProvenanceCrossSideMergeBudget(**values)


def _b3_projection_budget() -> ConversationHeldOutV4ProvenanceProjectionBudget:
    """构造仅供 P2-C synthetic B3 result wrapper 的完整冻结预算。"""
    sort_budget = IntegerExternalSortBudget(
        max_input_file_count=16,
        max_input_physical_bytes=2 * 1024 * 1024,
        max_input_record_count=128,
        max_input_payload_bytes=2 * 1024 * 1024,
        max_record_payload_bytes=64 * 1024,
        max_batch_record_count=4,
        max_batch_payload_bytes=128 * 1024,
        max_batch_sort_key_bytes=128 * 1024,
        max_temporary_run_count=128,
        max_temporary_record_count=512,
        max_temporary_payload_bytes=4 * 1024 * 1024,
        max_temporary_physical_bytes=8 * 1024 * 1024,
        max_output_physical_bytes=2 * 1024 * 1024,
        merge_fan_in=2,
        max_open_files=3,
        max_merge_pass_count=16,
    )
    closure_budget = ConversationHeldOutV4ProvenanceClosureBudget(
        max_node_count=16,
        max_direct_edge_count=32,
        max_known_pair_count=64,
        max_candidate_pair_count_per_round=64,
        max_closure_round_count=8,
        max_parents_per_node=4,
        max_record_payload_bytes=64 * 1024,
        max_stream_record_count=128,
        max_stream_payload_bytes=2 * 1024 * 1024,
        max_stream_physical_bytes=2 * 1024 * 1024,
        max_total_materialized_physical_bytes=32 * 1024 * 1024,
        max_open_files=3,
        external_sort_budget=sort_budget,
    )
    return ConversationHeldOutV4ProvenanceProjectionBudget(
        max_node_count=16,
        max_leaf_count=16,
        max_known_pair_count=64,
        max_leaves_per_node=4,
        max_ancestors_per_node=8,
        max_lineage_projection_record_count=64,
        max_record_payload_bytes=64 * 1024,
        max_stream_record_count=128,
        max_stream_payload_bytes=2 * 1024 * 1024,
        max_stream_physical_bytes=2 * 1024 * 1024,
        max_total_materialized_physical_bytes=64 * 1024 * 1024,
        max_open_files=4,
        external_sort_budget=sort_budget,
        closure_replay_budget=closure_budget,
    )


def _record(channel: int, leaf: ConversationHeldOutV4ProvenanceLeaf,
            *, lineage_value: int | None = None) -> tuple[int, ...]:
    """以 B3 的公开 canonical encoder 构造一个完整 projection record。"""
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
        return encode_v4_provenance_source_ref_projection_record(
            leaf.source_ref, leaf.side, leaf)
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
        return encode_v4_provenance_content_projection_record(
            leaf.content_sha256, leaf.side, leaf)
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE:
        if lineage_value is None:
            raise AssertionError("lineage record 缺测试 ancestor")
        return encode_v4_provenance_lineage_projection_record(
            1,
            _key(7_000 + lineage_value),
            _snapshot(lineage_value),
            leaf.side,
            leaf,
        )
    raise AssertionError("测试 channel 未注册")


def _sort_key(channel: int, record: tuple[int, ...]) -> tuple[int, ...]:
    """按已知 test fixture primary/side/leaf 字段排序输入，模拟 B3 output。"""
    from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_projection import (
        decode_v4_provenance_content_projection_record,
        decode_v4_provenance_lineage_projection_record,
        decode_v4_provenance_source_ref_projection_record,
    )

    def pack(*parts: tuple[int, ...]) -> tuple[int, ...]:
        result: list[int] = []
        from pure_integer_ai.storage.integer_codec import pack_key
        for part in parts:
            pack_key(result, part)
        return tuple(result)

    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
        source, side, leaf = decode_v4_provenance_source_ref_projection_record(record)
        return pack(source.stable_key(), (side,), leaf.integer_stream())
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
        content, side, leaf = decode_v4_provenance_content_projection_record(record)
        return pack(content, (side,), leaf.integer_stream())
    namespace, ancestor, snapshot, side, leaf = (
        decode_v4_provenance_lineage_projection_record(record))
    return pack(
        (namespace,), ancestor.components, snapshot.integer_stream(), (side,),
        leaf.integer_stream())


def _write_projection(
        tmp_path: Path,
        name: str,
        channel: int,
        records: tuple[tuple[int, ...], ...],
        ):
    """在独立 B3-like work root 写入一条封存 projection descriptor。"""
    work = create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label="P2-C test B3 work root",
    )
    stage = create_new_run_root(
        work.path / "b3-projection",
        require_k_drive=False,
        label="P2-C test B3 stage",
    )
    relative = Path("projection.pifrs")
    stream = open_exclusive_binary(stage, relative, label="P2-C test input")
    writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
    try:
        for record in records:
            writer.append(record)
        footer = writer.seal()
    finally:
        writer.close()
    payload = (stage.path / relative).read_bytes()
    descriptor = ConversationHeldOutV4ProvenanceProjectionStreamDescriptor(
        channel,
        Path("b3-projection") / relative,
        footer,
        KRunFileDigest(len(payload), tuple(hashlib.sha256(payload).digest())),
    )
    return work, descriptor


def _write_b3_result(
        tmp_path: Path,
        name: str,
        *, side: int, source_value: int, content: str,
        lineage_value: int,
        ):
    """构造一侧三个已排序 descriptor 的 synthetic B3 result，不模拟真实资格。"""
    work = create_new_run_root(
        tmp_path / name,
        require_k_drive=False,
        label="P2-C synthetic B3 result root",
    )
    leaf = _leaf(source_value, source_value=source_value, side=side,
                 content=content)
    descriptors: dict[int, ConversationHeldOutV4ProvenanceProjectionStreamDescriptor] = {}
    for channel, short_name in (
            (V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF, "source"),
            (V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT, "content"),
            (V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE, "lineage")):
        stage_name = f"b3-{short_name}"
        stage = create_new_run_root(
            work.path / stage_name,
            require_k_drive=False,
            label="P2-C synthetic B3 result stage",
        )
        relative = Path("projection.pifrs")
        record = _record(channel, leaf, lineage_value=lineage_value)
        stream = open_exclusive_binary(stage, relative, label="P2-C synthetic B3 input")
        writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
        try:
            writer.append(record)
            footer = writer.seal()
        finally:
            writer.close()
        payload = (stage.path / relative).read_bytes()
        descriptors[channel] = ConversationHeldOutV4ProvenanceProjectionStreamDescriptor(
            channel,
            Path(stage_name) / relative,
            footer,
            KRunFileDigest(
                len(payload), tuple(hashlib.sha256(payload).digest())),
        )
    budget = _b3_projection_budget()
    result = ConversationHeldOutV4ProvenanceProjectionResult(
        f"{name}.b3",
        (1, source_value),
        (2, source_value),
        descriptors[V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF],
        descriptors[V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT],
        descriptors[V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE],
        1,
        1,
        1,
        0,
        budget,
    )
    return work, result


def _read_merged(root, descriptor, budget):
    """用 capability/P0 bridge 读取 P2-C sealed output，供精确断言。"""
    stream = open_plain_binary(root, descriptor.output_relative_path,
                               label="P2-C test merged read")
    reader = IntegerFramedStreamReader.from_open_binary(
        stream,
        path=descriptor.output_relative_path,
        max_frame_bytes=budget.max_record_payload_bytes,
        max_record_count=budget.max_stream_record_count,
        max_total_payload_bytes=budget.max_stream_payload_bytes,
    )
    try:
        result = tuple(reader)
        assert reader.footer == descriptor.p0_footer
        return result
    finally:
        reader.close()


def _complete_three_channel_pair(
        training_result, training_root, held_out_result, held_out_root, output,
        *, code_identity, budget, logical_stage_name):
    """按公开 C1 接口推进固定三通道，供 C2 只验证最终发布边界。"""
    cursor = None
    for _ in range(3):
        cursor = advance_v4_provenance_cross_side_merge(
            training_result,
            training_root,
            held_out_result,
            held_out_root,
            output,
            code_identity=code_identity,
            budget=budget,
            logical_stage_name=logical_stage_name,
            resume_cursor=cursor,
        )
    assert cursor is not None
    return cursor


def _b3_descriptor_for_channel(result, channel):
    """只在 test transport 中按固定公开三路取 synthetic B3 descriptor。"""
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
        return result.source_ref_projection_stream
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
        return result.content_projection_stream
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE:
        return result.lineage_projection_stream
    raise AssertionError("unknown P2-C channel")


def _reopen_p2c_roots(training_root, held_root, output_root):
    """模拟新进程只持有已打开 capability，而不保留 create-new token。"""
    return (
        open_existing_run_root(
            training_root.path, require_k_drive=False,
            label="P2-C reopened training root"),
        open_existing_run_root(
            held_root.path, require_k_drive=False,
            label="P2-C reopened held-out root"),
        open_existing_run_root(
            output_root.path, require_k_drive=False,
            label="P2-C reopened output root"),
    )


def build_v4_provenance_cross_side_merge_channel(*args, **kwargs):
    """为仅验证单 channel 内核的 fixture 补齐显式 synthetic frozen identity。"""
    channel = kwargs.get("channel")
    if type(channel) is not int:
        raise AssertionError("P2-C test build 缺 channel")
    kwargs.setdefault("training_b3_stable_key", (710_000 + channel,))
    kwargs.setdefault("held_out_b3_stable_key", (720_000 + channel,))
    kwargs.setdefault("code_identity", _key(730_000 + channel))
    return _raw_build_v4_provenance_cross_side_merge_channel(*args, **kwargs)


@pytest.mark.parametrize(
    "channel",
    (
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
    ),
)
def test_zero_cross_side_merge_preserves_full_records_and_uses_one_writer(channel, tmp_path):
    """三种完整 primary 均可在两 reader/一 writer 下确定性 zero-overlap merge。"""
    training_leaf = _leaf(1, source_value=10, side=V4_PROVENANCE_SIDE_TRAINING,
                          content="training")
    held_leaf = _leaf(2, source_value=20, side=V4_PROVENANCE_SIDE_HELD_OUT,
                      content="held")
    training_record = _record(channel, training_leaf, lineage_value=10)
    held_record = _record(channel, held_leaf, lineage_value=20)
    training, training_descriptor = _write_projection(
        tmp_path, f"{channel}-training", channel, (training_record,))
    held_out, held_out_descriptor = _write_projection(
        tmp_path, f"{channel}-held", channel, (held_record,))
    output = create_new_run_root(
        tmp_path / f"{channel}-output",
        require_k_drive=False,
        label="P2-C test output root",
    )
    budget = _budget()

    result = build_v4_provenance_cross_side_merge_channel(
        training, training_descriptor, held_out, held_out_descriptor, output,
        channel=channel, budget=budget, logical_stage_name=f"zero-{channel}.v1")

    assert result.training_input_record_count == result.held_out_input_record_count == 1
    assert result.training_exact_duplicate_count == result.held_out_exact_duplicate_count == 0
    assert result.emitted_record_count == 2
    assert result.materialized_physical_bytes == (
        result.intent.physical.byte_count + result.output.physical.byte_count)
    assert _read_merged(output, result.output, budget) == tuple(sorted(
        (training_record, held_record), key=lambda item: _sort_key(channel, item)))


@pytest.mark.parametrize(
    "channel",
    (
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
    ),
)
def test_cross_side_same_primary_is_rejected_even_when_full_leaves_differ(channel, tmp_path):
    """primary equality 不能被不同 side/full leaf 掩盖。"""
    training_leaf = _leaf(1, source_value=10, side=V4_PROVENANCE_SIDE_TRAINING,
                          content="same-content")
    held_leaf = _leaf(2, source_value=10, side=V4_PROVENANCE_SIDE_HELD_OUT,
                      content="same-content")
    training_record = _record(channel, training_leaf, lineage_value=10)
    held_record = _record(channel, held_leaf, lineage_value=10)
    training, training_descriptor = _write_projection(
        tmp_path, f"overlap-{channel}-training", channel, (training_record,))
    held_out, held_out_descriptor = _write_projection(
        tmp_path, f"overlap-{channel}-held", channel, (held_record,))
    output = create_new_run_root(
        tmp_path / f"overlap-{channel}-output",
        require_k_drive=False,
        label="P2-C overlap output",
    )

    with pytest.raises(ConversationHeldOutV4ProvenanceCrossSideOverlapError,
                       match="cross-side overlap"):
        build_v4_provenance_cross_side_merge_channel(
            training, training_descriptor, held_out, held_out_descriptor, output,
            channel=channel, budget=_budget(),
            logical_stage_name=f"overlap-{channel}.v1")
    assert not (output.path / "manifest.pii").exists()


def test_same_side_exact_duplicate_is_counted_and_only_written_once(tmp_path):
    """同一完整 B3 record 重复只能形成 exact duplicate，不吞掉不同 primary。"""
    first = _leaf(1, source_value=10, side=V4_PROVENANCE_SIDE_TRAINING,
                  content="first")
    second = _leaf(2, source_value=20, side=V4_PROVENANCE_SIDE_HELD_OUT,
                   content="second")
    first_record = _record(V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF, first)
    second_record = _record(V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF, second)
    training, training_descriptor = _write_projection(
        tmp_path, "duplicate-training", V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        (first_record, first_record))
    held_out, held_out_descriptor = _write_projection(
        tmp_path, "duplicate-held", V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        (second_record,))
    output = create_new_run_root(
        tmp_path / "duplicate-output", require_k_drive=False,
        label="P2-C duplicate output")
    budget = _budget()

    result = build_v4_provenance_cross_side_merge_channel(
        training, training_descriptor, held_out, held_out_descriptor, output,
        channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        budget=budget, logical_stage_name="duplicate.v1")

    assert result.training_input_record_count == 2
    assert result.training_exact_duplicate_count == 1
    assert result.emitted_record_count == 2
    assert _read_merged(output, result.output, budget) == (first_record, second_record)


@pytest.mark.parametrize(
    "channel",
    (
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
    ),
)
def test_same_primary_different_full_leaf_is_preserved(channel, tmp_path):
    """同侧 primary 相同但完整 leaf 不同不是 duplicate，三条都必须保留。"""
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
        first = _leaf(1, source_value=71, side=V4_PROVENANCE_SIDE_TRAINING,
                      content="first")
        second = _leaf(2, source_value=71, side=V4_PROVENANCE_SIDE_TRAINING,
                       content="second")
        first_lineage, second_lineage = 71, 72
    elif channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
        first = _leaf(1, source_value=71, side=V4_PROVENANCE_SIDE_TRAINING,
                      content="same")
        second = _leaf(2, source_value=72, side=V4_PROVENANCE_SIDE_TRAINING,
                       content="same")
        first_lineage, second_lineage = 71, 72
    else:
        first = _leaf(1, source_value=71, side=V4_PROVENANCE_SIDE_TRAINING,
                      content="first")
        second = _leaf(2, source_value=72, side=V4_PROVENANCE_SIDE_TRAINING,
                       content="second")
        first_lineage = second_lineage = 99
    held = _leaf(3, source_value=73, side=V4_PROVENANCE_SIDE_HELD_OUT,
                 content="held")
    first_record = _record(channel, first, lineage_value=first_lineage)
    second_record = _record(channel, second, lineage_value=second_lineage)
    held_record = _record(channel, held, lineage_value=173)
    training_records = tuple(sorted(
        (first_record, second_record), key=lambda item: _sort_key(channel, item)))
    held_records = (held_record,)
    training, training_descriptor = _write_projection(
        tmp_path, f"same-primary-{channel}-training", channel, training_records)
    held_out, held_descriptor = _write_projection(
        tmp_path, f"same-primary-{channel}-held", channel, held_records)
    output = create_new_run_root(
        tmp_path / f"same-primary-{channel}-output", require_k_drive=False,
        label="P2-C same primary different leaf output")
    budget = _budget()

    result = build_v4_provenance_cross_side_merge_channel(
        training, training_descriptor, held_out, held_descriptor, output,
        channel=channel, budget=budget, logical_stage_name="same-primary.v1")

    assert result.training_exact_duplicate_count == 0
    assert result.held_out_exact_duplicate_count == 0
    assert result.emitted_record_count == 3
    assert _read_merged(output, result.output, budget) == tuple(sorted(
        (first_record, second_record, held_record),
        key=lambda item: _sort_key(channel, item)))


def test_rejects_wrong_side_before_output_and_descriptor_drift(tmp_path):
    """expected side 与 physical descriptor 任一漂移都不得进入 output stage。"""
    wrong_leaf = _leaf(1, source_value=10, side=V4_PROVENANCE_SIDE_HELD_OUT,
                       content="wrong")
    held_leaf = _leaf(2, source_value=20, side=V4_PROVENANCE_SIDE_HELD_OUT,
                      content="held")
    wrong_record = _record(V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT, wrong_leaf)
    held_record = _record(V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT, held_leaf)
    training, training_descriptor = _write_projection(
        tmp_path, "wrong-side-training", V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        (wrong_record,))
    held_out, held_out_descriptor = _write_projection(
        tmp_path, "wrong-side-held", V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        (held_record,))
    output = create_new_run_root(
        tmp_path / "wrong-side-output", require_k_drive=False,
        label="P2-C wrong-side output")

    with pytest.raises(ConversationHeldOutV4ProvenanceCrossSideMergeError,
                       match="side"):
        build_v4_provenance_cross_side_merge_channel(
            training, training_descriptor, held_out, held_out_descriptor, output,
            channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
            budget=_budget(), logical_stage_name="wrong-side.v1")

    clean_training = _leaf(3, source_value=30, side=V4_PROVENANCE_SIDE_TRAINING,
                           content="clean")
    clean_record = _record(V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
                           clean_training)
    drift_training, drift_descriptor = _write_projection(
        tmp_path, "drift-training", V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        (clean_record,))
    replacement = drift_training.path / "replacement.pifrs"
    replacement.write_bytes(b"drift")
    replacement.replace(drift_training.path / drift_descriptor.work_relative_path)
    output_drift = create_new_run_root(
        tmp_path / "drift-output", require_k_drive=False,
        label="P2-C drift output")
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        build_v4_provenance_cross_side_merge_channel(
            drift_training, drift_descriptor, held_out, held_out_descriptor,
            output_drift, channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
            budget=_budget(), logical_stage_name="drift.v1")
    assert not any(output_drift.path.iterdir())


def test_budget_stage_collision_and_reopened_capability_fail_closed(tmp_path):
    """不允许以过小预算、既有 stage 或重开 root 伪造可恢复 merge。"""
    training_leaf = _leaf(1, source_value=10, side=V4_PROVENANCE_SIDE_TRAINING,
                          content="training")
    held_leaf = _leaf(2, source_value=20, side=V4_PROVENANCE_SIDE_HELD_OUT,
                      content="held")
    training_record = _record(V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
                              training_leaf)
    held_record = _record(V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF, held_leaf)
    training, training_descriptor = _write_projection(
        tmp_path, "bounds-training", V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        (training_record,))
    held_out, held_out_descriptor = _write_projection(
        tmp_path, "bounds-held", V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        (held_record,))
    output = create_new_run_root(
        tmp_path / "bounds-output", require_k_drive=False,
        label="P2-C bounds output")

    with pytest.raises(ConversationHeldOutV4ProvenanceCrossSideMergeBudgetExceeded):
        build_v4_provenance_cross_side_merge_channel(
            training, training_descriptor, held_out, held_out_descriptor, output,
            channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
            budget=_budget(max_stream_record_count=1), logical_stage_name="bounds.v1")
    with pytest.raises(KRunBoundaryError, match="已存在|不能为空|不存在"):
        build_v4_provenance_cross_side_merge_channel(
            training, training_descriptor, held_out, held_out_descriptor, output,
            channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
            budget=_budget(), logical_stage_name="collision.v1")

    reopened_training = open_existing_run_root(
        training.path, require_k_drive=False, label="P2-C reopened training")
    fresh_output = create_new_run_root(
        tmp_path / "reopened-output", require_k_drive=False,
        label="P2-C reopened output")
    with pytest.raises(TypeError, match="allow_reopened_roots"):
        build_v4_provenance_cross_side_merge_channel(
            training, training_descriptor, held_out, held_out_descriptor,
            fresh_output, channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
            budget=_budget(), logical_stage_name="reopened-public.v1",
            allow_reopened_roots=True)
    assert not any(fresh_output.path.iterdir())
    with pytest.raises(KRunBoundaryError, match="create_new_run_root"):
        build_v4_provenance_cross_side_merge_channel(
            reopened_training, training_descriptor, held_out, held_out_descriptor,
            fresh_output, channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
            budget=_budget(), logical_stage_name="reopened.v1")


def test_three_channel_pair_gate_advances_only_through_sealed_cursors(tmp_path):
    """P2-C1 每次只完成一个固定通道，并以 cursor 绑定完整 B3 pair 与 code identity。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "cursor-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=10, content="training", lineage_value=10)
    held_root, held_result = _write_b3_result(
        tmp_path, "cursor-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=20, content="held", lineage_value=20)
    output = create_new_run_root(
        tmp_path / "cursor-output", require_k_drive=False,
        label="P2-C cursor output")
    budget = _budget()
    code_identity = _key(12_000)

    first = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-cursor.v1")
    assert isinstance(first, ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact)
    assert [item.channel for item in first.cursor.completed_channel_results] == [
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF]
    assert first.cursor.training_b3_stable_key == training_result.stable_key()
    assert first.cursor.held_out_b3_stable_key == held_result.stable_key()
    assert first.descriptor.p0_footer.record_count == 1

    second = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-cursor.v1", resume_cursor=first)
    third = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-cursor.v1", resume_cursor=second)
    assert [item.channel for item in third.cursor.completed_channel_results] == [
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
    ]
    assert third.cursor.is_complete
    assert not (output.path / "manifest.pii").exists()

    with pytest.raises(ConversationHeldOutV4ProvenanceCrossSideMergeError,
                       match="已完成"):
        advance_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-cursor.v1", resume_cursor=third)


def test_pair_gate_revalidates_all_inputs_and_cursor_before_next_output(tmp_path):
    """任一未轮到的 B3 stream 或 sealed cursor 漂移均阻止后续 channel stage。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "gate-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=10, content="training", lineage_value=10)
    held_root, held_result = _write_b3_result(
        tmp_path, "gate-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=20, content="held", lineage_value=20)
    budget = _budget()
    code_identity = _key(12_100)
    output = create_new_run_root(
        tmp_path / "gate-output", require_k_drive=False,
        label="P2-C gate output")

    held_content = held_result.content_projection_stream
    (held_root.path / held_content.work_relative_path).write_bytes(b"drift")
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        advance_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-gate.v1")
    assert not any(output.path.iterdir())

    clean_training_root, clean_training_result = _write_b3_result(
        tmp_path, "cursor-drift-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=30, content="training", lineage_value=30)
    clean_held_root, clean_held_result = _write_b3_result(
        tmp_path, "cursor-drift-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=40, content="held", lineage_value=40)
    clean_output = create_new_run_root(
        tmp_path / "cursor-drift-output", require_k_drive=False,
        label="P2-C cursor drift output")
    first = advance_v4_provenance_cross_side_merge(
        clean_training_result, clean_training_root,
        clean_held_result, clean_held_root, clean_output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-cursor-drift.v1")
    (clean_output.path / first.descriptor.output_relative_path).write_bytes(b"drift")
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        advance_v4_provenance_cross_side_merge(
            clean_training_result, clean_training_root,
            clean_held_result, clean_held_root, clean_output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-cursor-drift.v1", resume_cursor=first)
    assert not (clean_output.path / "stage-21-cross-side-content").exists()


@pytest.mark.parametrize("completed_prefix", (0, 1, 2))
def test_seal_before_cursor_recovery_replays_each_channel_without_rewrite(
        completed_prefix, tmp_path):
    """三个固定通道都可接管 seal 后 cursor 前的完整 P0，不重写输出。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"orphan-training-{completed_prefix}",
        side=V4_PROVENANCE_SIDE_TRAINING, source_value=100 + completed_prefix,
        content="training", lineage_value=100 + completed_prefix)
    held_root, held_result = _write_b3_result(
        tmp_path, f"orphan-held-{completed_prefix}",
        side=V4_PROVENANCE_SIDE_HELD_OUT, source_value=200 + completed_prefix,
        content="held", lineage_value=200 + completed_prefix)
    output = create_new_run_root(
        tmp_path / f"orphan-output-{completed_prefix}", require_k_drive=False,
        label="P2-C orphan recovery output")
    budget = _budget()
    code_identity = _key(13_000 + completed_prefix)
    cursor = None
    for _ in range(completed_prefix):
        cursor = advance_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-orphan.v1", resume_cursor=cursor)
    channel = (
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
    )[completed_prefix]
    orphan = build_v4_provenance_cross_side_merge_channel(
        training_root, _b3_descriptor_for_channel(training_result, channel),
        held_root, _b3_descriptor_for_channel(held_result, channel), output,
        channel=channel,
        training_b3_stable_key=training_result.stable_key(),
        held_out_b3_stable_key=held_result.stable_key(),
        code_identity=code_identity,
        budget=budget,
        logical_stage_name="p2c-orphan.v1")
    orphan_path = output.path / orphan.output.output_relative_path
    orphan_bytes = orphan_path.read_bytes()

    recovered = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-orphan.v1", resume_cursor=cursor)

    assert recovered.cursor.completed_channel_results[-1].output == orphan.output
    assert len(recovered.cursor.completed_channel_results) == completed_prefix + 1
    assert orphan_path.read_bytes() == orphan_bytes
    assert not (output.path / "stage-26-cross-side-resource-receipt").exists()
    assert not (output.path / "manifest.pii").exists()


def test_unverifiable_sealed_channel_residue_fails_without_new_cursor(tmp_path):
    """orphan stage 含未知残片时只能保留残片并拒绝接管。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "bad-orphan-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=310, content="training", lineage_value=310)
    held_root, held_result = _write_b3_result(
        tmp_path, "bad-orphan-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=320, content="held", lineage_value=320)
    output = create_new_run_root(
        tmp_path / "bad-orphan-output", require_k_drive=False,
        label="P2-C bad orphan output")
    budget = _budget()
    code_identity = _key(13_100)
    first = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-bad-orphan.v1")
    channel = V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT
    orphan = build_v4_provenance_cross_side_merge_channel(
        training_root, _b3_descriptor_for_channel(training_result, channel),
        held_root, _b3_descriptor_for_channel(held_result, channel), output,
        channel=channel,
        training_b3_stable_key=training_result.stable_key(),
        held_out_b3_stable_key=held_result.stable_key(),
        code_identity=code_identity,
        budget=budget,
        logical_stage_name="p2c-bad-orphan.v1")
    residue = output.path / orphan.output.output_relative_path.parent / "residue.pii"
    residue.write_bytes(b"unadopted")

    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        advance_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-bad-orphan.v1", resume_cursor=first)
    assert residue.exists()
    assert not (output.path / "stage-24-cross-side-cursor").exists()
    assert not (output.path / "stage-26-cross-side-resource-receipt").exists()
    assert not (output.path / "manifest.pii").exists()


def test_complete_cursor_publishes_receipt_then_manifest_last_once(tmp_path):
    """三路零交集最终 cursor 只能发布一份回读一致的 receipt/manifest。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "publish-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=410, content="training", lineage_value=410)
    held_root, held_result = _write_b3_result(
        tmp_path, "publish-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=420, content="held", lineage_value=420)
    output = create_new_run_root(
        tmp_path / "publish-output", require_k_drive=False,
        label="P2-C publication output")
    budget = _budget()
    code_identity = _key(13_200)
    cursor = _complete_three_channel_pair(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publish.v1")

    publication = publish_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        completed_cursor=cursor, code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publish.v1")
    receipt = publication.receipt_artifact
    assert receipt.descriptor.output_relative_path == (
        Path("stage-26-cross-side-resource-receipt") / "resource-receipt.pifrs")
    assert receipt.receipt.final_cursor_stable_key == cursor.stable_key()
    assert len(receipt.receipt.pre_receipt_files) == 9
    assert len(publication.manifest.pre_manifest_files) == 10
    assert decode_integer_tuple((output.path / "manifest.pii").read_bytes()) == (
        publication.manifest.integer_stream())
    receipt_stream = open_plain_binary(
        output, receipt.descriptor.output_relative_path,
        label="P2-C test receipt read")
    reader = IntegerFramedStreamReader.from_open_binary(
        receipt_stream, path=receipt.descriptor.output_relative_path,
        max_frame_bytes=budget.max_record_payload_bytes,
        max_record_count=budget.max_stream_record_count,
        max_total_payload_bytes=budget.max_stream_payload_bytes)
    try:
        assert tuple(reader) == (receipt.receipt.integer_stream(),)
        assert reader.footer == receipt.descriptor.p0_footer
    finally:
        reader.close()
    manifest_bytes = (output.path / "manifest.pii").read_bytes()
    with pytest.raises(ConversationHeldOutV4ProvenanceCrossSideMergeError,
                       match="manifest"):
        publish_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            completed_cursor=cursor, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-publish.v1")
    assert (output.path / "manifest.pii").read_bytes() == manifest_bytes


def test_reopened_published_run_is_reconstructed_read_only(tmp_path):
    """新进程可从固定 inputs 完整回读已发布 P2-C，且不写任何 output 文件。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "publication-loader-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=450, content="training", lineage_value=450)
    held_root, held_result = _write_b3_result(
        tmp_path, "publication-loader-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=460, content="held", lineage_value=460)
    output = create_new_run_root(
        tmp_path / "publication-loader-output", require_k_drive=False,
        label="P2-C publication loader output")
    budget = _budget()
    code_identity = _key(13_250)
    cursor = _complete_three_channel_pair(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publication-loader.v1")
    published = publish_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        completed_cursor=cursor, code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publication-loader.v1")
    before = {
        path.relative_to(output.path): path.read_bytes()
        for path in output.path.rglob("*") if path.is_file()
    }
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)

    loaded = load_v4_provenance_cross_side_merge_publication(
        training_result, reopened_training, held_result, reopened_held, reopened_output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publication-loader.v1")

    assert loaded.stable_key() == published.stable_key()
    after = {
        path.relative_to(output.path): path.read_bytes()
        for path in output.path.rglob("*") if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("drift_kind", ("manifest", "receipt", "cursor", "output"))
def test_publication_loader_rejects_any_registered_file_drift(drift_kind, tmp_path):
    """已发布闭包任一关键文件变化都不得被回读器重解释或补写。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"publication-loader-drift-training-{drift_kind}",
        side=V4_PROVENANCE_SIDE_TRAINING, source_value=470,
        content="training", lineage_value=470)
    held_root, held_result = _write_b3_result(
        tmp_path, f"publication-loader-drift-held-{drift_kind}",
        side=V4_PROVENANCE_SIDE_HELD_OUT, source_value=480,
        content="held", lineage_value=480)
    output = create_new_run_root(
        tmp_path / f"publication-loader-drift-output-{drift_kind}",
        require_k_drive=False, label="P2-C publication loader drift output")
    budget = _budget()
    code_identity = _key(13_260)
    cursor = _complete_three_channel_pair(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publication-loader-drift.v1")
    publish_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        completed_cursor=cursor, code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publication-loader-drift.v1")
    target = {
        "manifest": output.path / "manifest.pii",
        "receipt": output.path / "stage-26-cross-side-resource-receipt" / "resource-receipt.pifrs",
        "cursor": output.path / "stage-23-cross-side-cursor" / "cursor.pifrs",
        "output": output.path / "stage-20-cross-side-source-ref" / "merged.pifrs",
    }[drift_kind]
    target.write_bytes(b"drift")
    before = target.read_bytes()
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)

    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        load_v4_provenance_cross_side_merge_publication(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-publication-loader-drift.v1")
    assert target.read_bytes() == before


def test_publication_loader_rejects_unknown_final_closure_residue(tmp_path):
    """manifest 存在也不能掩盖 publication root 中未登记普通文件。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "publication-loader-residue-training",
        side=V4_PROVENANCE_SIDE_TRAINING, source_value=490,
        content="training", lineage_value=490)
    held_root, held_result = _write_b3_result(
        tmp_path, "publication-loader-residue-held",
        side=V4_PROVENANCE_SIDE_HELD_OUT, source_value=500,
        content="held", lineage_value=500)
    output = create_new_run_root(
        tmp_path / "publication-loader-residue-output", require_k_drive=False,
        label="P2-C publication loader residue output")
    budget = _budget()
    code_identity = _key(13_270)
    cursor = _complete_three_channel_pair(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publication-loader-residue.v1")
    publish_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        completed_cursor=cursor, code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-publication-loader-residue.v1")
    residue = output.path / "unknown-publication-residue.pii"
    residue.write_bytes(b"residue")
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)

    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        load_v4_provenance_cross_side_merge_publication(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-publication-loader-residue.v1")
    assert residue.read_bytes() == b"residue"


@pytest.mark.parametrize("mutate_registered_file", (False, "output", "intent"))
def test_receipt_recovery_rejects_drift_and_allows_unchanged_manifest_retry(
        mutate_registered_file, tmp_path, monkeypatch):
    """receipt seal 后的失败可续接；已声明文件漂移则不得补写 manifest。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"receipt-recovery-training-{mutate_registered_file}",
        side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=510, content="training", lineage_value=510)
    held_root, held_result = _write_b3_result(
        tmp_path, f"receipt-recovery-held-{mutate_registered_file}",
        side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=520, content="held", lineage_value=520)
    output = create_new_run_root(
        tmp_path / f"receipt-recovery-output-{mutate_registered_file}",
        require_k_drive=False,
        label="P2-C receipt recovery output")
    budget = _budget()
    code_identity = _key(13_300)
    cursor = _complete_three_channel_pair(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-receipt-recovery.v1")
    original_publish_manifest_last = merge_module.publish_manifest_last

    def stop_before_manifest(*args, **kwargs):
        raise RuntimeError("intentional manifest interruption")

    monkeypatch.setattr(merge_module, "publish_manifest_last", stop_before_manifest)
    with pytest.raises(RuntimeError, match="intentional manifest interruption"):
        publish_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            completed_cursor=cursor, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-receipt-recovery.v1")
    receipt_path = (output.path / "stage-26-cross-side-resource-receipt"
                    / "resource-receipt.pifrs")
    receipt_bytes = receipt_path.read_bytes()
    assert receipt_path.exists()
    assert not (output.path / "manifest.pii").exists()

    monkeypatch.setattr(merge_module, "publish_manifest_last", original_publish_manifest_last)
    if mutate_registered_file:
        first_channel = cursor.cursor.completed_channel_results[0]
        relative = (
            first_channel.intent.output_relative_path
            if mutate_registered_file == "intent"
            else first_channel.output.output_relative_path)
        drift = output.path / relative
        drift.write_bytes(b"drift")
        with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                            KRunBoundaryError)):
            publish_v4_provenance_cross_side_merge(
                training_result, training_root, held_result, held_root, output,
                completed_cursor=cursor, code_identity=code_identity, budget=budget,
                logical_stage_name="p2c-receipt-recovery.v1")
        assert receipt_path.read_bytes() == receipt_bytes
        assert not (output.path / "manifest.pii").exists()
        return
    publication = publish_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        completed_cursor=cursor, code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-receipt-recovery.v1")
    assert publication.manifest_file.relative_path == Path("manifest.pii")
    assert receipt_path.read_bytes() == receipt_bytes


def test_publication_rejects_unknown_residue_before_receipt(tmp_path):
    """任何未登记普通文件都阻止 receipt 与 manifest，而不是被静默忽略。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "residue-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=610, content="training", lineage_value=610)
    held_root, held_result = _write_b3_result(
        tmp_path, "residue-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=620, content="held", lineage_value=620)
    output = create_new_run_root(
        tmp_path / "residue-output", require_k_drive=False,
        label="P2-C residue output")
    budget = _budget()
    code_identity = _key(13_400)
    cursor = _complete_three_channel_pair(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-residue.v1")
    (output.path / "unknown-residue.pii").write_bytes(b"residue")

    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        publish_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            completed_cursor=cursor, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-residue.v1")
    assert not (output.path / "stage-26-cross-side-resource-receipt").exists()
    assert not (output.path / "manifest.pii").exists()


@pytest.mark.parametrize("completed_channel_count", (1, 2, 3))
def test_reopened_cursor_loader_rebuilds_explicit_prefix_and_can_resume(
        completed_channel_count, tmp_path):
    """新进程重开三 root 后，指定 prefix 只能由重放结果重建并继续一条新 stage。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"loader-training-{completed_channel_count}",
        side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=710 + completed_channel_count, content="training",
        lineage_value=710 + completed_channel_count)
    held_root, held_result = _write_b3_result(
        tmp_path, f"loader-held-{completed_channel_count}",
        side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=720 + completed_channel_count, content="held",
        lineage_value=720 + completed_channel_count)
    output = create_new_run_root(
        tmp_path / f"loader-output-{completed_channel_count}",
        require_k_drive=False, label="P2-C loader output")
    budget = _budget()
    code_identity = _key(13_500 + completed_channel_count)
    cursor = None
    for _ in range(completed_channel_count):
        cursor = advance_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader.v1", resume_cursor=cursor)
    assert cursor is not None
    prior_files = {
        path.relative_to(output.path): path.read_bytes()
        for path in output.path.rglob("*") if path.is_file()
    }
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)

    loaded = load_v4_provenance_cross_side_merge_resume_cursor(
        training_result, reopened_training, held_result, reopened_held, reopened_output,
        completed_channel_count=completed_channel_count,
        code_identity=code_identity, budget=budget, logical_stage_name="p2c-loader.v1")

    assert loaded.stable_key() == cursor.stable_key()
    if completed_channel_count == 3:
        with pytest.raises(ConversationHeldOutV4ProvenanceCrossSideMergeError,
                           match="已完成"):
            advance_v4_provenance_cross_side_merge(
                training_result, reopened_training, held_result, reopened_held,
                reopened_output, code_identity=code_identity, budget=budget,
                logical_stage_name="p2c-loader.v1", resume_cursor=loaded)
        return
    resumed = advance_v4_provenance_cross_side_merge(
        training_result, reopened_training, held_result, reopened_held, reopened_output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader.v1", resume_cursor=loaded)
    assert len(resumed.cursor.completed_channel_results) == completed_channel_count + 1
    for relative, payload in prior_files.items():
        assert (output.path / relative).read_bytes() == payload


@pytest.mark.parametrize("completed_prefix", (0, 1, 2))
def test_reopened_orphan_output_is_replayed_and_only_its_cursor_is_written(
        completed_prefix, tmp_path):
    """重开后可接管每个 channel 的 orphan output，且不重写已 seal 字节。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"loader-orphan-training-{completed_prefix}",
        side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=810 + completed_prefix, content="training",
        lineage_value=810 + completed_prefix)
    held_root, held_result = _write_b3_result(
        tmp_path, f"loader-orphan-held-{completed_prefix}",
        side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=820 + completed_prefix, content="held",
        lineage_value=820 + completed_prefix)
    output = create_new_run_root(
        tmp_path / f"loader-orphan-output-{completed_prefix}",
        require_k_drive=False, label="P2-C loader orphan output")
    budget = _budget()
    code_identity = _key(13_600 + completed_prefix)
    cursor = None
    for _ in range(completed_prefix):
        cursor = advance_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-orphan.v1", resume_cursor=cursor)
    channel = (
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
        V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
    )[completed_prefix]
    orphan = build_v4_provenance_cross_side_merge_channel(
        training_root, _b3_descriptor_for_channel(training_result, channel),
        held_root, _b3_descriptor_for_channel(held_result, channel), output,
        channel=channel,
        training_b3_stable_key=training_result.stable_key(),
        held_out_b3_stable_key=held_result.stable_key(),
        code_identity=code_identity,
        budget=budget,
        logical_stage_name="p2c-loader-orphan.v1")
    orphan_path = output.path / orphan.output.output_relative_path
    orphan_bytes = orphan_path.read_bytes()
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)
    if completed_prefix == 0:
        resumed = advance_v4_provenance_cross_side_merge(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-orphan.v1")
    else:
        loaded = load_v4_provenance_cross_side_merge_resume_cursor(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, completed_channel_count=completed_prefix,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-orphan.v1")
        assert cursor is not None
        assert loaded.stable_key() == cursor.stable_key()
        resumed = advance_v4_provenance_cross_side_merge(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-orphan.v1", resume_cursor=loaded)
    assert len(resumed.cursor.completed_channel_results) == completed_prefix + 1
    assert orphan_path.read_bytes() == orphan_bytes


@pytest.mark.parametrize(
    "drift_kind",
    ("stage", "code", "budget", "b3", "missing_intent", "damaged_intent"),
)
def test_reopened_orphan_requires_its_original_frozen_intent(drift_kind, tmp_path):
    """无 cursor orphan 只能按其封存的完整身份接管，不能用当前参数重解释。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"intent-drift-training-{drift_kind}",
        side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=1_010, content="training", lineage_value=1_010)
    held_root, held_result = _write_b3_result(
        tmp_path, f"intent-drift-held-{drift_kind}",
        side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=1_020, content="held", lineage_value=1_020)
    output = create_new_run_root(
        tmp_path / f"intent-drift-output-{drift_kind}",
        require_k_drive=False, label="P2-C intent drift output")
    budget = _budget()
    code_identity = _key(13_800)
    channel = V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF
    orphan = build_v4_provenance_cross_side_merge_channel(
        training_root, _b3_descriptor_for_channel(training_result, channel),
        held_root, _b3_descriptor_for_channel(held_result, channel), output,
        channel=channel,
        training_b3_stable_key=training_result.stable_key(),
        held_out_b3_stable_key=held_result.stable_key(),
        code_identity=code_identity,
        budget=budget,
        logical_stage_name="p2c-intent-drift.v1")
    intent_path = output.path / orphan.intent.output_relative_path
    merged_path = output.path / orphan.output.output_relative_path
    intent_bytes = intent_path.read_bytes()
    merged_bytes = merged_path.read_bytes()
    attempted_training_result = training_result
    attempted_code_identity = code_identity
    attempted_budget = budget
    attempted_stage_name = "p2c-intent-drift.v1"
    if drift_kind == "stage":
        attempted_stage_name = "p2c-intent-drift.v2"
    elif drift_kind == "code":
        attempted_code_identity = _key(13_801)
    elif drift_kind == "budget":
        attempted_budget = _budget(max_intent_physical_bytes=128 * 1024)
    elif drift_kind == "b3":
        attempted_training_result = replace(
            training_result,
            direct_dag_stable_key=(91_000, 1),
        )
    elif drift_kind == "missing_intent":
        intent_path.unlink()
    elif drift_kind == "damaged_intent":
        intent_path.write_bytes(b"damaged-intent")
    else:
        raise AssertionError("unknown intent drift kind")
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)

    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        advance_v4_provenance_cross_side_merge(
            attempted_training_result, reopened_training,
            held_result, reopened_held, reopened_output,
            code_identity=attempted_code_identity,
            budget=attempted_budget,
            logical_stage_name=attempted_stage_name)
    if drift_kind not in {"missing_intent", "damaged_intent"}:
        assert intent_path.read_bytes() == intent_bytes
    assert merged_path.read_bytes() == merged_bytes
    assert not (output.path / "stage-23-cross-side-cursor").exists()


def test_channel_intent_is_sealed_before_merged_and_unpaired_stage_stops(tmp_path, monkeypatch):
    """intent-first 中断保留可审计残片，但没有 merged output 时绝不补 cursor。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "intent-first-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=1_030, content="training", lineage_value=1_030)
    held_root, held_result = _write_b3_result(
        tmp_path, "intent-first-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=1_040, content="held", lineage_value=1_040)
    output = create_new_run_root(
        tmp_path / "intent-first-output", require_k_drive=False,
        label="P2-C intent-first output")
    budget = _budget()
    code_identity = _key(13_900)
    original_open_exclusive = merge_module.open_exclusive_binary

    def interrupt_after_intent(root, relative, **kwargs):
        if Path(relative) == Path("merged.pifrs"):
            assert (root.path / "intent.pifrs").is_file()
            raise RuntimeError("intent-first interruption")
        return original_open_exclusive(root, relative, **kwargs)

    monkeypatch.setattr(merge_module, "open_exclusive_binary", interrupt_after_intent)
    with pytest.raises(RuntimeError, match="intent-first interruption"):
        advance_v4_provenance_cross_side_merge(
            training_result, training_root, held_result, held_root, output,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-intent-first.v1")
    stage = output.path / "stage-20-cross-side-source-ref"
    assert (stage / "intent.pifrs").is_file()
    assert not (stage / "merged.pifrs").exists()
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        advance_v4_provenance_cross_side_merge(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-intent-first.v1")
    assert not (output.path / "stage-23-cross-side-cursor").exists()


def test_advance_revalidates_all_cursor_prefixes_after_loader(tmp_path):
    """loader 返回后旧 cursor 漂移也必须阻止 advance 写下一通道。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "loader-recheck-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=1_050, content="training", lineage_value=1_050)
    held_root, held_result = _write_b3_result(
        tmp_path, "loader-recheck-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=1_060, content="held", lineage_value=1_060)
    output = create_new_run_root(
        tmp_path / "loader-recheck-output", require_k_drive=False,
        label="P2-C loader recheck output")
    budget = _budget()
    code_identity = _key(14_000)
    first = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader-recheck.v1")
    second = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader-recheck.v1", resume_cursor=first)
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)
    loaded = load_v4_provenance_cross_side_merge_resume_cursor(
        training_result, reopened_training, held_result, reopened_held,
        reopened_output, completed_channel_count=2,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader-recheck.v1")
    assert loaded.stable_key() == second.stable_key()
    (output.path / "stage-23-cross-side-cursor" / "cursor.pifrs").write_bytes(
        b"drift")
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        advance_v4_provenance_cross_side_merge(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-recheck.v1", resume_cursor=loaded)
    assert not (output.path / "stage-22-cross-side-lineage").exists()
    assert not (output.path / "stage-25-cross-side-cursor").exists()


@pytest.mark.parametrize("cursor_stage", (23, 24, 25))
def test_reopened_loader_rejects_each_historical_cursor_drift(cursor_stage, tmp_path):
    """任一已完成 prefix cursor 物理漂移都不能被后续 checkpoint 掩盖。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"loader-cursor-{cursor_stage}-training",
        side=V4_PROVENANCE_SIDE_TRAINING, source_value=910,
        content="training", lineage_value=910)
    held_root, held_result = _write_b3_result(
        tmp_path, f"loader-cursor-{cursor_stage}-held",
        side=V4_PROVENANCE_SIDE_HELD_OUT, source_value=920,
        content="held", lineage_value=920)
    output = create_new_run_root(
        tmp_path / f"loader-cursor-{cursor_stage}-output", require_k_drive=False,
        label="P2-C loader historical cursor output")
    budget = _budget()
    code_identity = _key(13_700 + cursor_stage)
    _complete_three_channel_pair(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader-historical-cursor.v1")
    cursor_path = output.path / f"stage-{cursor_stage}-cross-side-cursor" / "cursor.pifrs"
    cursor_path.write_bytes(b"drift")
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        load_v4_provenance_cross_side_merge_resume_cursor(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, completed_channel_count=3,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-historical-cursor.v1")


def test_reopened_loader_rejects_cursor_drift_and_unknown_closure(tmp_path):
    """loader 不会把可读 P0 或未知普通文件当作可恢复 prefix。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "loader-reject-training", side=V4_PROVENANCE_SIDE_TRAINING,
        source_value=910, content="training", lineage_value=910)
    held_root, held_result = _write_b3_result(
        tmp_path, "loader-reject-held", side=V4_PROVENANCE_SIDE_HELD_OUT,
        source_value=920, content="held", lineage_value=920)
    output = create_new_run_root(
        tmp_path / "loader-reject-output", require_k_drive=False,
        label="P2-C loader reject output")
    budget = _budget()
    code_identity = _key(13_700)
    cursor = advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader-reject.v1")
    cursor_path = output.path / cursor.descriptor.output_relative_path
    cursor_bytes = cursor_path.read_bytes()
    cursor_path.write_bytes(b"drift")
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)

    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        load_v4_provenance_cross_side_merge_resume_cursor(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, completed_channel_count=1,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-reject.v1")
    assert not (output.path / "stage-21-cross-side-content").exists()
    cursor_path.write_bytes(cursor_bytes)
    (output.path / "unknown-loader-residue.pii").write_bytes(b"residue")
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        load_v4_provenance_cross_side_merge_resume_cursor(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, completed_channel_count=1,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-reject.v1")


def test_reopened_orphan_rejects_damaged_merged_framing(tmp_path):
    """孤儿 merged P0 直接损坏时必须在恢复前统一 fail closed。"""
    training_root, training_result = _write_b3_result(
        tmp_path, "loader-damaged-orphan-training",
        side=V4_PROVENANCE_SIDE_TRAINING, source_value=930,
        content="training", lineage_value=930)
    held_root, held_result = _write_b3_result(
        tmp_path, "loader-damaged-orphan-held",
        side=V4_PROVENANCE_SIDE_HELD_OUT, source_value=940,
        content="held", lineage_value=940)
    output = create_new_run_root(
        tmp_path / "loader-damaged-orphan-output", require_k_drive=False,
        label="P2-C loader damaged orphan output")
    budget = _budget()
    code_identity = _key(13_800)
    orphan = build_v4_provenance_cross_side_merge_channel(
        training_root, _b3_descriptor_for_channel(
            training_result, V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF),
        held_root, _b3_descriptor_for_channel(
            held_result, V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF),
        output, channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
        training_b3_stable_key=training_result.stable_key(),
        held_out_b3_stable_key=held_result.stable_key(),
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader-damaged-orphan.v1")
    (output.path / orphan.output.output_relative_path).write_bytes(b"damaged-p0")
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)
    with pytest.raises(ConversationHeldOutV4ProvenanceCrossSideMergeError):
        advance_v4_provenance_cross_side_merge(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-damaged-orphan.v1")
    assert not (output.path / "stage-23-cross-side-cursor").exists()


@pytest.mark.parametrize("missing_name", ("intent.pifrs", "merged.pifrs"))
def test_reopened_loader_rejects_cursor_without_channel_closure(
        missing_name, tmp_path):
    """cursor 不能脱离对应 intent/output 闭包独立成为可恢复 state。"""
    training_root, training_result = _write_b3_result(
        tmp_path, f"loader-missing-{missing_name}-training",
        side=V4_PROVENANCE_SIDE_TRAINING, source_value=950,
        content="training", lineage_value=950)
    held_root, held_result = _write_b3_result(
        tmp_path, f"loader-missing-{missing_name}-held",
        side=V4_PROVENANCE_SIDE_HELD_OUT, source_value=960,
        content="held", lineage_value=960)
    output = create_new_run_root(
        tmp_path / f"loader-missing-{missing_name}-output", require_k_drive=False,
        label="P2-C loader missing channel closure output")
    budget = _budget()
    code_identity = _key(13_900)
    advance_v4_provenance_cross_side_merge(
        training_result, training_root, held_result, held_root, output,
        code_identity=code_identity, budget=budget,
        logical_stage_name="p2c-loader-missing-channel.v1")
    (output.path / "stage-20-cross-side-source-ref" / missing_name).unlink()
    reopened_training, reopened_held, reopened_output = _reopen_p2c_roots(
        training_root, held_root, output)
    with pytest.raises((ConversationHeldOutV4ProvenanceCrossSideMergeError,
                        KRunBoundaryError)):
        load_v4_provenance_cross_side_merge_resume_cursor(
            training_result, reopened_training, held_result, reopened_held,
            reopened_output, completed_channel_count=1,
            code_identity=code_identity, budget=budget,
            logical_stage_name="p2c-loader-missing-channel.v1")
    assert not (output.path / "stage-21-cross-side-content").exists()


def test_merge_module_remains_pure_integer_provenance_boundary():
    """P2-C core 不得跨入 runtime、owner、private/formal、JSON、SQLite 或 freeze transport。"""
    module_path = Path(__file__).parents[1] / "src" / "pure_integer_ai" / "experiments" / (
        "conversation_heldout_v4_provenance_scalable_merge.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    forbidden = ("json", "sqlite", "runtime", "owner", "private", "formal")
    assert not any(token in module for module in imported for token in forbidden)
    assert "LineageFreeze" not in module_path.read_text(encoding="utf-8")
