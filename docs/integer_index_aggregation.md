# Integer Index Aggregation

Repeated language or structure payloads are represented in three layers:

1. An atom vocabulary maps each visible atom to an integer.
2. A token sequence stores an ordered integer sequence once.
3. An aggregate stores an ordered sequence of token-sequence or earlier aggregate
   references once.

Every occurrence keeps only its source, scope, position, role, and an integer
ordinal into the aggregate. Reading expands the referenced layers only for the
active window. The source and occurrence records are not deduplicated, because
two identical surfaces can have different provenance or meaning.

`pure_integer_ai.experiments.integer_token_index` is the portable exchange
format. `pure_integer_ai.storage.integer_index_store` projects the same data
into any `StorageBackend` using integer-only tables. Both layers use explicit
SHA-256 identities, deterministic ordering, and fail-closed checks for bounds,
forward references, cycles, and identity drift.

The broad-course builder emits both sidecars and the shard/release builders copy
and register both of them. This is content deduplication at the course and
training-index boundary. The graph still records each occurrence and relation;
full graph payload compaction remains a separate measured optimization and must
not be inferred from sidecar deduplication alone.

The training backend stores the vocabulary atom values as Unicode code-point
integers alongside token-sequence members. `IntegerIndexStore.replay_occurrence`
can therefore reconstruct a token or aggregate occurrence from SQLite/DictBackend
rows alone (the caller may convert the returned code points to display text).
Member identities are append-only and content conflicts at the same layer,
parent, and ordinal fail closed.

`render_occurrence` is the convenience boundary for a host that needs display
text; the stored representation remains integer-only.
