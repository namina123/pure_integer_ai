from pure_integer_ai.experiments.integer_token_index import (
    build_integer_aggregate_index,
    build_integer_token_index,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.integer_index_store import (
    INDEX_KIND_AGGREGATE,
    INTEGER_INDEX_MEMBER_TABLE,
    INTEGER_INDEX_OCCURRENCE_TABLE,
    IntegerIndexStore,
)


def test_integer_index_store_deduplicates_members_and_replays_occurrences() -> None:
    backend = DictBackend()
    token_index = build_integer_token_index(
        ("甲乙", "甲乙", "丙"), sequence_keys=("a", "b", "c"))
    aggregate_index = build_integer_aggregate_index(
        token_index,
        (("a", (token_index.occurrence_ordinals[0],)),
         ("b", (token_index.occurrence_ordinals[1],)),
         ("c", (token_index.occurrence_ordinals[2],))),
    )
    store = IntegerIndexStore(backend)
    token_binding = store.bind(token_index)
    aggregate_binding = store.bind(token_index, aggregate_index)
    assert aggregate_binding.index_kind == INDEX_KIND_AGGREGATE
    assert token_binding.token_sequence_count == 2
    assert aggregate_binding.aggregate_count == 2
    assert backend.count(INTEGER_INDEX_MEMBER_TABLE) == 8
    assert backend.count(INTEGER_INDEX_OCCURRENCE_TABLE) == 6
    # A resumed run must not append duplicate sequence/aggregate members.
    store.bind(token_index)
    store.bind(token_index, aggregate_index)
    assert backend.count(INTEGER_INDEX_MEMBER_TABLE) == 8
    assert backend.count(INTEGER_INDEX_OCCURRENCE_TABLE) == 6


def test_integer_index_store_replays_from_integer_sqlite_rows() -> None:
    backend = SQLiteBackend(":memory:")
    token_index = build_integer_token_index(
        ("甲乙", "甲乙", "丙"), sequence_keys=("a", "b", "c"))
    aggregate_index = build_integer_aggregate_index(
        token_index,
        (("a", (token_index.occurrence_ordinals[0],)),
         ("b", (token_index.occurrence_ordinals[1],)),
         ("c", (token_index.occurrence_ordinals[2],))),
    )
    store = IntegerIndexStore(backend)
    store.bind(token_index)
    store.bind(token_index, aggregate_index)
    assert tuple(map(chr, store.replay_occurrence(
        token_index.sha256, 1))) == tuple("甲乙")
    assert tuple(map(chr, store.replay_occurrence(
        token_index.sha256, 2, aggregate_index_sha256=aggregate_index.sha256
    ))) == tuple("丙")
    assert store.render_occurrence(
        token_index.sha256, 2,
        aggregate_index_sha256=aggregate_index.sha256) == "丙"
    backend.close()


def test_integer_index_store_replays_nested_aggregate() -> None:
    backend = DictBackend()
    token_index = build_integer_token_index(
        ("甲", "乙"), sequence_keys=("a", "b"))
    aggregate_index = build_integer_aggregate_index(
        token_index,
        (("base", (0,)),
         ("nested", (len(token_index.sequences), 1))),
    )
    store = IntegerIndexStore(backend)
    store.bind(token_index)
    store.bind(token_index, aggregate_index)
    assert tuple(map(chr, store.replay_occurrence(
        token_index.sha256, 1,
        aggregate_index_sha256=aggregate_index.sha256))) == tuple("甲乙")


def test_integer_index_store_roundtrips_with_sqlite_backend(tmp_path) -> None:
    token_index = build_integer_token_index(
        ("甲乙", "甲乙", "丙"), sequence_keys=("a", "b", "c"))
    aggregate_index = build_integer_aggregate_index(
        token_index,
        (("a", (token_index.occurrence_ordinals[0],)),
         ("b", (token_index.occurrence_ordinals[1],)),
         ("c", (token_index.occurrence_ordinals[2],))),
    )
    path = tmp_path / "integer-index.sqlite3"
    first = SQLiteBackend(str(path))
    first_store = IntegerIndexStore(first)
    first_store.bind(token_index)
    first_store.bind(token_index, aggregate_index)
    first.commit()
    first.close()

    reopened = SQLiteBackend(str(path))
    reopened_store = IntegerIndexStore(reopened)
    reopened_store.bind(token_index)
    reopened_store.bind(token_index, aggregate_index)
    assert reopened.count(INTEGER_INDEX_MEMBER_TABLE) == 8
    assert reopened.count(INTEGER_INDEX_OCCURRENCE_TABLE) == 6
    assert reopened.schema_snapshot()["integer_index_header"]["recovery_key"] == (
        "hash_0", "hash_1", "hash_2", "hash_3", "hash_4", "hash_5",
        "hash_6", "hash_7")
    reopened.close()
