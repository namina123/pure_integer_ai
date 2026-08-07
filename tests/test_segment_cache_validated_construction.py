"""热缓存 owner 的已验证冻结记录重建路径。"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.sealed_segment import SegmentBudget, SegmentRecord
from pure_integer_ai.storage.segment_cache import (
    CachedSegmentRecord,
    SegmentPageCache,
)


def test_internal_cache_transitions_do_not_repeat_public_validation(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    calls = 0
    original = CachedSegmentRecord.__post_init__

    def counted(value: CachedSegmentRecord) -> None:
        nonlocal calls
        calls += 1
        original(value)

    monkeypatch.setattr(CachedSegmentRecord, "__post_init__", counted)
    first = SegmentRecord((1,), (11,))
    second = SegmentRecord((2,), (22,))
    cache = SegmentPageCache(SegmentBudget(2, 1_000_000))

    loaded = cache.page_in((99,), (first,))
    assert loaded[0].record == first
    assert cache.get((99,), (1,)) is not None
    assert cache.pin((99,), (1,)).pin_count == 1
    assert cache.unpin((99,), (1,)).pin_count == 0
    assert cache.put_dirty((99,), second).dirty is True
    flushed: list[CachedSegmentRecord] = []
    assert cache.flush_dirty(
        lambda records: flushed.extend(records)) == 1
    assert flushed[0].record == second
    assert calls == 0


def test_public_cached_record_constructor_still_validates() -> None:
    record = SegmentRecord((1,), (11,))
    with pytest.raises(TypeError, match="dirty"):
        CachedSegmentRecord((99,), record, 1, 1, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="access_seq"):
        CachedSegmentRecord((99,), record, False, 0, 0)
