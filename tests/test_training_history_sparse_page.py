"""候选历史分页必须完整恢复请求事件，并避免无关 chunk 扫描。"""
from pathlib import Path

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.training_candidate_event import (
    TRAINING_CANDIDATE_EVENT_PART_TABLE, TrainingCandidateEventIntegrityError,
    TrainingCandidateEventRecord, TrainingCandidateEventRecordStore,
)


def _records(backend, count):
    """创建有限且互不重名的完整物理事件，payload 不含领域文字。"""
    bootstrap(backend)
    store = TrainingCandidateEventRecordStore(backend)
    records = []
    for index in range(1, count + 1):
        payload = tuple(index * 100 + part for part in range(19))
        record = TrainingCandidateEventRecord(index, 6, 7, 1, index, 19, 19)
        store.add(record, payload)
        records.append(record)
    return store, tuple(records)


@pytest.mark.parametrize('kind', ('dict', 'sqlite'))
def test_training_history_reads_only_requested_pages_and_matches_single_reads(tmp_path, monkeypatch, kind):
    """跨页、乱序请求与两个后端均返回全部原 payload，逐页读取上界固定。"""
    backend = DictBackend() if kind == 'dict' else SQLiteBackend(str(tmp_path / 'history.sqlite3'))
    try:
        store, records = _records(backend, 131)
        selected = tuple(reversed(records[1:130]))
        expected = {record.event_hash: store.read_payload(record) for record in selected}
        read, pages = backend.select, []
        select_in = backend.select_in

        def bounded_select(table, *args, **kwargs):
            """阻止恢复时读取未被请求的整张 chunk 表。"""
            if table == TRAINING_CANDIDATE_EVENT_PART_TABLE:
                raise AssertionError('unbounded history chunk scan')
            return read(table, *args, **kwargs)

        def capture_page(table, column, values, *args, **kwargs):
            """记录真实索引读取页，不能用统计替代完整输出核验。"""
            if table == TRAINING_CANDIDATE_EVENT_PART_TABLE:
                assert column == 'event_hash' and 1 <= len(values) <= 128
                pages.append(values)
            return select_in(table, column, values, *args, **kwargs)

        monkeypatch.setattr(backend, 'select', bounded_select)
        monkeypatch.setattr(backend, 'select_in', capture_page)
        assert store.read_payload_many(selected) == expected
        assert tuple(value for page in pages for value in page) == tuple(record.event_hash for record in selected)
        assert len(pages) == 2
        assert store.read_payload_many(()) == {}
    finally:
        if kind == 'sqlite':
            backend.close()


@pytest.mark.parametrize('damage,message', [
    ('missing', 'chunk 数量不完整'),
    ('duplicate', 'chunk 数量不完整'),
    ('space', '空间或顺序漂移'),
    ('ordinal', '空间或顺序漂移'),
    ('padding', '填充位必须为零'),
    ('width', '宽度非法'),
    ('size', 'payload_size 不一致'),
])
def test_training_history_page_keeps_original_integrity_rejections(damage, message):
    """任何命中块的损坏仍失败，不能因稀疏读取放行半写或漂移。"""
    backend = DictBackend()
    store, records = _records(backend, 2)
    snapshot = backend.snapshot()
    parts = snapshot[TRAINING_CANDIDATE_EVENT_PART_TABLE]
    first = next(row for row in parts if row['event_hash'] == records[0].event_hash)
    last = next(row for row in parts if row['event_hash'] == records[0].event_hash and row['chunk_index'] == 1)
    if damage == 'missing':
        parts.remove(first)
    elif damage == 'duplicate':
        parts.append(dict(first))
    elif damage == 'space':
        first['space_id'] += 1
    elif damage == 'ordinal':
        first['chunk_index'] = 2
    elif damage == 'padding':
        last['part_15'] = 1
    elif damage == 'width':
        first['part_size'] = 17
    elif damage == 'size':
        last['part_size'] -= 1
        last['part_02'] = 0
    backend.load_snapshot(snapshot)
    with pytest.raises(TrainingCandidateEventIntegrityError, match=message):
        store.read_payload_many(records)


def test_training_history_page_rejects_duplicate_requests_before_io(monkeypatch):
    """重复 event_hash 必须在任一页读取前拒绝，不能由结果字典去重。"""
    backend = DictBackend()
    store, records = _records(backend, 1)
    def forbidden(*args, **kwargs):
        """重复请求不得触及任何 chunk 页。"""
        raise AssertionError('unexpected chunk read')
    monkeypatch.setattr(backend, 'select_in', forbidden)
    with pytest.raises(TrainingCandidateEventIntegrityError, match='重复 event_hash'):
        store.read_payload_many((records[0], records[0]))
