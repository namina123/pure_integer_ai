"""DLG-RAW-07 typed RAW-04 snapshot 的有界 codec 与真实恢复专项。"""
from __future__ import annotations

from pathlib import Path
from shutil import copy2

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    OBJECT_CONCEPT,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextState,
    ConversationTurnState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    ConversationRawDialogueState,
    run_public_frame_dialogue_turn,
    start_public_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_snapshot import (
    ConversationRawDialogueSnapshotError,
    decode_public_frame_dialogue_snapshot_bytes,
    encode_public_frame_dialogue_snapshot_bytes,
    restore_public_frame_dialogue_state,
    snapshot_public_frame_dialogue_state,
)
from pure_integer_ai.experiments.conversation_raw_intake import encode_utf8_v1
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3,
)
_ROOT = Path(__file__).resolve().parents[1]


def _source(ordinal: int) -> SourceRef:
    """构造具完整 owner/version 的纯整数 source identity。"""
    return SourceRef(65070, ordinal + 1, ordinal, GLOBAL_OWNER_SCOPE,
                     VersionBundle())


def _turn(ordinal: int, context_read) -> ConversationTurnState:
    """构造不含 surface 的最小完整 typed context turn。"""
    return ConversationTurnState(
        ordinal,
        (65001, 71, ordinal, 1),
        (65001, 72, ordinal, 1),
        (65001, 73, ordinal, 1),
        (65001, 74, ordinal, 1),
        ObjectIdentity(OBJECT_CONCEPT, (65001, 75, ordinal, 1)),
        ((65001, 76, ordinal, 1), (65001, 76, ordinal, 2)),
        (_source(ordinal),),
        ((65001, 77, ordinal, 1),),
        (65001, 78, 1),
        (65001, 79, ordinal, 1),
        context_read,
    )


def _typed_state() -> ConversationRawDialogueState:
    """形成两轮可恢复 state，第二轮显式消费第一轮的 typed read。"""
    initial = start_conversation_context((65001, 80, 1))
    first = _turn(0, initial.read(0))
    after_first = ConversationContextState(
        initial.conversation_key,
        1,
        initial.digest(),
        (first,),
    )
    second = _turn(1, after_first.read(1))
    after_second = ConversationContextState(
        after_first.conversation_key,
        2,
        after_first.digest(),
        (*after_first.turns, second),
    )
    return ConversationRawDialogueState(
        after_second.conversation_key,
        3,
        after_second,
    )


def _public_runtime_from_root(root: Path) -> PublicDialogueRuntimeV1:
    """经 host 冻结完整 public closure，再构建无路径 dialogue runtime。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(root))


def _copy_public_source_root(tmp_path: Path) -> Path:
    """复制整个公开资源闭包，供跨物理根 A/B snapshot 恢复验收。"""
    root = tmp_path / "public-dialogue-root-b"
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative = logical_key.decode("ascii")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(_ROOT / relative, destination)
    return root


def _alias_surface(runtime: PublicDialogueRuntimeV1) -> tuple[int, ...]:
    """仅从 runtime 已验证的 V2 family/binding 拼接真实 alias 首句。"""
    slot = runtime.source_bound_slot_catalog
    binding = next(item for item in slot.bindings
                   if item.binding_key == "north-east-side-entrance-time-v2")
    expected = tuple(ord(character) for character in "北川站东侧入口何时启用？")
    surfaces = tuple(
        (*family.prefix_scalars, *binding.entity_scalars,
         *family.suffix_scalars)
        for family in slot.families
        if (*family.prefix_scalars, *binding.entity_scalars,
            *family.suffix_scalars) == expected
    )
    assert surfaces == (expected,)
    return expected


def _drift_legacy_v1_slot_transport(root: Path) -> None:
    """只改变 V2 runtime 不读取的旧 V1 transport，以产生不同 logical closure。"""
    source = root / "data/ph2/dlg_raw_public_source_bound_slot_v1.jsonl.sample"
    payload = source.read_bytes()
    assert payload
    source.write_bytes(b"X" + payload[1:])


def _target_anchor_follow_up(runtime: PublicDialogueRuntimeV1) -> tuple[int, ...]:
    """按冻结 scalar 精确选取既有 target-anchor follow-up 输入。"""
    scalars = tuple(ord(item) for item in "它是在什么时候启用的？")
    matches = runtime.active_catalog.matching_frames(scalars)
    assert len(matches) == 1
    return matches[0].surface_bytes


@pytest.fixture(scope="module")
def public_runtime() -> PublicDialogueRuntimeV1:
    """本地公开 root 的 runtime 只构造一次，避免无关重复 I/O。"""
    return _public_runtime_from_root(_ROOT)


def test_typed_state_record_and_bytes_roundtrip_without_surface_payload(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """完整 typed state 必须 decode/re-encode 相等，且 bytes 不是 Python object dump。"""
    state = _typed_state()
    record = snapshot_public_frame_dialogue_state(state, public_runtime)
    restored = restore_public_frame_dialogue_state(record, public_runtime)
    payload = encode_public_frame_dialogue_snapshot_bytes(state, public_runtime)
    restored_bytes = decode_public_frame_dialogue_snapshot_bytes(
        payload, public_runtime)

    assert restored == state
    assert restored_bytes == state
    assert snapshot_public_frame_dialogue_state(restored, public_runtime) == record
    assert encode_public_frame_dialogue_snapshot_bytes(restored, public_runtime) == payload
    assert b"\xe4\xb8\x9c\xe4\xbe\xa7\xe5\x85\xa5\xe5\x8f\xa3" not in payload


def test_snapshot_rejects_binding_schema_truncation_and_tail_drift(
        public_runtime: PublicDialogueRuntimeV1) -> None:
    """恢复必须在 runtime 前拒绝 binding/record/transport 的结构漂移。"""
    state = _typed_state()
    record = snapshot_public_frame_dialogue_state(state, public_runtime)
    payload = encode_public_frame_dialogue_snapshot_bytes(state, public_runtime)
    drifted_binding = (*record[:2], record[2] + 1, *record[3:])

    with pytest.raises(ConversationRawDialogueSnapshotError, match="binding"):
        restore_public_frame_dialogue_state(drifted_binding, public_runtime)
    with pytest.raises(ConversationRawDialogueSnapshotError, match="截断|长度越界"):
        restore_public_frame_dialogue_state(record[:-1], public_runtime)
    with pytest.raises(ConversationRawDialogueSnapshotError, match="尾随"):
        restore_public_frame_dialogue_state((*record, 0), public_runtime)
    with pytest.raises(ConversationRawDialogueSnapshotError, match="截断|尾随|length 越界"):
        decode_public_frame_dialogue_snapshot_bytes(payload[:-1], public_runtime)


def test_alias_first_turn_restores_across_physical_roots_then_runs_follow_up(
        tmp_path: Path) -> None:
    """A 的动态 alias 首句必须能由 B 恢复并走真实 target-anchor follow-up。"""
    root_b = _copy_public_source_root(tmp_path)
    runtime_a = _public_runtime_from_root(_ROOT)
    runtime_b = _public_runtime_from_root(root_b)
    assert (runtime_a.source_payload_closure.closure_identity
            == runtime_b.source_payload_closure.closure_identity)
    assert runtime_a.binding_record() == runtime_b.binding_record()
    assert runtime_a.runtime_identity() == runtime_b.runtime_identity()
    surface = _alias_surface(runtime_a)
    assert not runtime_a.active_catalog.matching_frames(surface)
    initial = start_public_frame_dialogue((65001, 81, 1))
    cache_a = PublicCoursePreparationCache()
    first = run_public_frame_dialogue_turn(
        initial,
        encode_utf8_v1(surface),
        runtime_a,
        preparation_cache=cache_a,
    )
    assert first.answer.accepted
    assert bytes(first.answer.output_bytes).decode("utf-8") == "北川站东门于2024年启用。"

    snapshot = encode_public_frame_dialogue_snapshot_bytes(first.after, runtime_a)
    restored = decode_public_frame_dialogue_snapshot_bytes(snapshot, runtime_b)
    assert restored == first.after
    follow_up = _target_anchor_follow_up(runtime_a)
    continuous = run_public_frame_dialogue_turn(
        first.after,
        follow_up,
        runtime_a,
        preparation_cache=cache_a,
    )
    resumed = run_public_frame_dialogue_turn(
        restored,
        follow_up,
        runtime_b,
        preparation_cache=PublicCoursePreparationCache(),
    )

    assert continuous.answer.accepted and resumed.answer.accepted
    assert continuous.answer.result_code == resumed.answer.result_code
    assert continuous.context_written == resumed.context_written == 1
    assert continuous.answer.output_bytes == resumed.answer.output_bytes
    assert bytes(resumed.answer.output_bytes).decode("utf-8") == "北川站东门于2024年启用。"
    assert (encode_public_frame_dialogue_snapshot_bytes(
        continuous.after, runtime_a)
            == encode_public_frame_dialogue_snapshot_bytes(
                resumed.after, runtime_b))


def test_v3_snapshot_rejects_different_legacy_closure_before_state_restore(
        tmp_path: Path) -> None:
    """V3 snapshot 即使语义 catalog 相同，也不得恢复到不同 closure identity。"""
    root_b = _copy_public_source_root(tmp_path)
    _drift_legacy_v1_slot_transport(root_b)
    runtime_a = _public_runtime_from_root(_ROOT)
    runtime_b = _public_runtime_from_root(root_b)

    assert (runtime_a.source_payload_closure.closure_identity
            != runtime_b.source_payload_closure.closure_identity)
    assert runtime_a.base_catalog.canonical_record() == runtime_b.base_catalog.canonical_record()
    assert runtime_a.active_catalog.canonical_record() == runtime_b.active_catalog.canonical_record()
    assert (runtime_a.source_bound_slot_catalog.catalog_schema
            == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3)
    assert (runtime_b.source_bound_slot_catalog.catalog_schema
            == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3)
    assert (runtime_a.source_bound_slot_catalog.manifest_sha256
            == runtime_b.source_bound_slot_catalog.manifest_sha256)
    assert runtime_a.binding_record() != runtime_b.binding_record()

    initial = start_public_frame_dialogue((65001, 81, 2))
    first = run_public_frame_dialogue_turn(
        initial,
        encode_utf8_v1(_alias_surface(runtime_a)),
        runtime_a,
        preparation_cache=PublicCoursePreparationCache(),
    )
    assert first.answer.accepted
    snapshot = encode_public_frame_dialogue_snapshot_bytes(first.after, runtime_a)

    with pytest.raises(ConversationRawDialogueSnapshotError, match="binding"):
        decode_public_frame_dialogue_snapshot_bytes(snapshot, runtime_b)
