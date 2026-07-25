"""Core 训练候选事件物理编码与完整性对抗。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TRAINING_HISTORY_EVIDENCE,
    TrainingCandidateHistoryLog,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.training_candidate_event import (
    TRAINING_CANDIDATE_EVENT_PART_TABLE,
    TRAINING_CANDIDATE_EVENT_TABLE,
    TrainingCandidateEventIntegrityError,
    decode_integer_stream,
    encode_integer_stream,
)


def test_training_candidate_integer_stream_round_trips_arbitrary_integers():
    """整数流编码不得截断负数、零、63-bit 边界或任意精度整数。"""
    values = (
        0,
        1,
        -1,
        127,
        -128,
        (1 << 63) - 1,
        -(1 << 63),
        1 << 130,
        -(1 << 129),
    )

    encoded = encode_integer_stream(values)

    assert decode_integer_stream(encoded) == values
    assert encode_integer_stream(decode_integer_stream(encoded)) == encoded


def test_training_candidate_integer_stream_rejects_noncanonical_tail_bits():
    """最后一个 56-bit word 的未使用字节非零时必须拒绝恢复。"""
    encoded = encode_integer_stream((1, 2, 3))
    byte_size = encoded[2]
    used = byte_size % 7
    assert used > 0
    damaged = (
        *encoded[:-1],
        encoded[-1] | (1 << (used * 8)),
    )

    with pytest.raises(
            TrainingCandidateEventIntegrityError,
            match="填充位非零"):
        decode_integer_stream(damaged)


def test_training_candidate_integer_stream_rejects_overlong_varint():
    """同一整数的非规范 varint 不能形成第二种物理表示。"""
    encoded = encode_integer_stream((0,))
    overlong_word = 128
    damaged = (
        encoded[0],
        encoded[1],
        2,
        overlong_word,
    )

    with pytest.raises(
            TrainingCandidateEventIntegrityError,
            match="不是规范编码"):
        decode_integer_stream(damaged)


def _history_fixture():
    """构造最小 Core 历史及一个含显式逻辑序的领域事件。"""
    backend = DictBackend()
    bootstrap(backend)
    source = SourceRef(1, 2, 3, GLOBAL_OWNER_SCOPE, VersionBundle())
    protocol = TrainingHypothesisHistoryProtocol(
        (4,),
        (5,),
        source,
        document_scope(source),
    )
    history = TrainingCandidateHistoryLog(backend, 6)
    history.append(protocol, TRAINING_HISTORY_EVIDENCE, 7, (8,))
    return backend, history, protocol


def test_training_candidate_history_rejects_tampered_event_sequence():
    """物理逻辑序即使仍为合法整数，也必须与完整事件信封一致。"""
    backend, history, protocol = _history_fixture()
    snapshot = backend.snapshot()
    snapshot[TRAINING_CANDIDATE_EVENT_TABLE][0]["event_seq"] = 9
    backend.load_snapshot(snapshot)

    with pytest.raises(
            TrainingCandidateEventIntegrityError,
            match="逻辑序与信封漂移"):
        history.entries(protocol)


def test_training_candidate_history_rejects_rekeyed_event_hash():
    """同时篡改信封与 chunk 外键时仍须按完整事件内容重算 hash。"""
    backend, history, protocol = _history_fixture()
    snapshot = backend.snapshot()
    old_hash = snapshot[TRAINING_CANDIDATE_EVENT_TABLE][0]["event_hash"]
    changed_hash = old_hash - 1 if old_hash > 1 else 2
    snapshot[TRAINING_CANDIDATE_EVENT_TABLE][0]["event_hash"] = changed_hash
    for row in snapshot[TRAINING_CANDIDATE_EVENT_PART_TABLE]:
        row["event_hash"] = changed_hash
    backend.load_snapshot(snapshot)

    with pytest.raises(
            TrainingCandidateEventIntegrityError,
            match="完整事件键不一致"):
        history.entries(protocol)
