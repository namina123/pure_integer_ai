"""热缓存 owner 的已验证冻结记录重建路径。"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.sealed_segment import SegmentBudget, SegmentRecord
from pure_integer_ai.storage.segment_cache import (
    CachedSegmentRecord,
    SegmentCacheError,
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


def test_clear_clean_cache_does_not_recompute_record_sizes(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    records = tuple(
        SegmentRecord((index,), (index + 10,)) for index in range(3))
    cache = SegmentPageCache(SegmentBudget(3, 1_000_000))
    cache.page_in((99,), records)

    def forbidden_size(_record: SegmentRecord) -> int:
        raise AssertionError("clean cache clear must not recompute sizes")

    monkeypatch.setattr(SegmentRecord, "size_bytes", forbidden_size)
    assert cache.clear() == 3
    assert cache.object_count == 0
    assert cache.size_bytes == 0


def test_owner_validated_get_skips_key_rescan_but_public_get_stays_guarded(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    import pure_integer_ai.storage.segment_cache as segment_cache_module

    record = SegmentRecord((1,), (11,))
    cache = SegmentPageCache(SegmentBudget(2, 1_000_000))
    cache.page_in((99,), (record,))

    def forbidden_key_scan(*_args, **_kwargs):
        raise AssertionError("owner-validated get must not rescan keys")

    monkeypatch.setattr(
        segment_cache_module, "strict_integer_tuple", forbidden_key_scan)
    assert cache._get_validated((99,), (1,)) is not None
    with pytest.raises(AssertionError, match="must not rescan"):
        cache.get((99,), (1,))


def test_clear_rechecks_reentrant_dirty_state_after_flush() -> None:
    first = SegmentRecord((1,), (11,))
    second = SegmentRecord((2,), (22,))
    cache = SegmentPageCache(SegmentBudget(3, 1_000_000))
    cache.put_dirty((99,), first)

    def reentrant_flush(_records) -> None:
        cache.put_dirty((99,), second)

    with pytest.raises(SegmentCacheError, match="dirty"):
        cache.clear(flush=reentrant_flush)
    assert cache.object_count == 2
    assert cache.get((99,), (2,)) is not None


def test_evict_all_clean_reuses_clear_without_recomputing_sizes(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    records = tuple(
        SegmentRecord((index,), (index + 10,)) for index in range(3))
    cache = SegmentPageCache(SegmentBudget(3, 1_000_000))
    cache.page_in((99,), records)

    def forbidden_size(_record: SegmentRecord) -> int:
        raise AssertionError("full clean eviction must not recompute sizes")

    monkeypatch.setattr(SegmentRecord, "size_bytes", forbidden_size)
    assert cache.evict_clean(cache.object_count) == 3
    assert cache.object_count == 0
    assert cache.size_bytes == 0
