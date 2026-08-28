"""SQLite-backed sparse intent index without eager Python object expansion.

The persistent schema stores only integer identifiers, integer counts and the
project's canonical integer-tuple codec.  Querying starts from rare features of
the current turn, bounds the prototype candidate set, then reproduces the v4
integer ranking over those candidates.  The schema is deliberately simple so
another language can consume the same rows without Python object semantics.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

from pure_integer_ai.experiments.conversation_learned_dialogue_response import (
    MAX_INTENT_FEATURE_WEIGHT,
    MIN_INTENT_SHARED_FEATURES,
    LearnedDialogueIntentModel,
    LearnedDialogueResponseError,
    dialogue_intent_features,
    _response_surface_allowed,
    _surface,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


SQLITE_INTENT_SCHEMA = 1
SQLITE_INTENT_FILE = "learned_dialogue_intent_index.sqlite3"
MAX_CURRENT_QUERY_FEATURES = 128
MAX_HISTORY_QUERY_FEATURES = 64
MAX_SEED_FEATURES = 8
MAX_PROTOTYPE_CANDIDATES = 4096
_META_SCHEMA = 1
_META_TRAIN_COUNT = 2
_META_FRAGMENT_COUNT = 3
_META_FEATURE_COUNT = 4
_META_PROTOTYPE_COUNT = 5


def _feature_key(value: tuple[int, ...]) -> bytes:
    return encode_integer_tuple(value)


def _feature_weight(train_count: int, document_count: int) -> int:
    return min(
        MAX_INTENT_FEATURE_WEIGHT,
        max(1, train_count // document_count),
    )


def build_sqlite_learned_dialogue_intent_index(
        path: str | Path, model: LearnedDialogueIntentModel,
        ) -> Path:
    """Publish one deterministic SQLite projection of a learned intent model."""
    if not isinstance(model, LearnedDialogueIntentModel):
        raise TypeError("SQLite intent index 需要 learned intent model")
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"SQLite intent index 已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target))
    try:
        connection.execute("PRAGMA page_size=4096")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript("""
            CREATE TABLE intent_meta(
                key INTEGER PRIMARY KEY,
                value INTEGER NOT NULL
            ) STRICT;
            CREATE TABLE intent_feature(
                feature_key BLOB PRIMARY KEY,
                feature_id INTEGER NOT NULL UNIQUE,
                document_count INTEGER NOT NULL CHECK(document_count > 0)
            ) WITHOUT ROWID, STRICT;
            CREATE TABLE intent_prototype(
                prototype_id INTEGER PRIMARY KEY,
                fragment_id INTEGER NOT NULL,
                occurrence_count INTEGER NOT NULL CHECK(occurrence_count > 0),
                weight_sum INTEGER NOT NULL CHECK(weight_sum > 0),
                current_weight_sum INTEGER NOT NULL CHECK(current_weight_sum > 0)
            ) STRICT;
            CREATE TABLE intent_posting(
                feature_id INTEGER NOT NULL,
                prototype_id INTEGER NOT NULL,
                PRIMARY KEY(feature_id, prototype_id)
            ) WITHOUT ROWID, STRICT;
            CREATE INDEX intent_posting_by_prototype
                ON intent_posting(prototype_id, feature_id);
        """)
        connection.executemany(
            "INSERT INTO intent_meta(key,value) VALUES(?,?)",
            (
                (_META_SCHEMA, SQLITE_INTENT_SCHEMA),
                (_META_TRAIN_COUNT, model.train_count),
                (_META_FRAGMENT_COUNT, model.fragment_count),
                (_META_FEATURE_COUNT, len(model.features)),
                (_META_PROTOTYPE_COUNT, len(model.prototype_features)),
            ),
        )
        connection.executemany(
            "INSERT INTO intent_feature(feature_key,feature_id,document_count) "
            "VALUES(?,?,?)",
            (
                (sqlite3.Binary(_feature_key(value)), ordinal,
                 model.feature_document_counts[ordinal])
                for ordinal, value in enumerate(model.features)
            ),
        )
        prototype_rows = []
        posting_rows = []
        current_prefix = (ord("q"), ord(":"))
        for prototype, features in enumerate(model.prototype_features):
            weight_sum = 0
            current_weight_sum = 0
            for feature in features:
                weight = _feature_weight(
                    model.train_count,
                    model.feature_document_counts[feature])
                weight_sum += weight
                if model.features[feature][:2] == current_prefix:
                    current_weight_sum += weight
                posting_rows.append((feature, prototype))
            prototype_rows.append((
                prototype, model.prototype_fragments[prototype],
                model.prototype_counts[prototype], weight_sum,
                current_weight_sum,
            ))
        connection.executemany(
            "INSERT INTO intent_prototype("
            "prototype_id,fragment_id,occurrence_count,weight_sum,"
            "current_weight_sum) VALUES(?,?,?,?,?)",
            prototype_rows,
        )
        connection.executemany(
            "INSERT INTO intent_posting(feature_id,prototype_id) VALUES(?,?)",
            posting_rows,
        )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    return target


def validate_sqlite_learned_dialogue_intent_index(
        path: str | Path, *, expected_train_count: int | None,
        expected_fragment_count: int,
        ) -> None:
    """Validate schema identity and the model binding without a full scan."""
    target = Path(path).resolve()
    if not target.is_file():
        raise LearnedDialogueResponseError("SQLite intent index 不存在")
    connection = sqlite3.connect(f"{target.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        meta = dict(connection.execute(
            "SELECT key,value FROM intent_meta ORDER BY key"))
        if (set(meta) != {
                _META_SCHEMA, _META_TRAIN_COUNT, _META_FRAGMENT_COUNT,
                _META_FEATURE_COUNT, _META_PROTOTYPE_COUNT}
                or meta[_META_SCHEMA] != SQLITE_INTENT_SCHEMA
                or meta[_META_FRAGMENT_COUNT] != expected_fragment_count
                or expected_train_count is not None
                and meta[_META_TRAIN_COUNT] != expected_train_count):
            raise LearnedDialogueResponseError("SQLite intent meta 漂移")
        if (type(meta[_META_TRAIN_COUNT]) is not int
                or meta[_META_TRAIN_COUNT] <= 0
                or type(meta[_META_FEATURE_COUNT]) is not int
                or meta[_META_FEATURE_COUNT] <= 0
                or type(meta[_META_PROTOTYPE_COUNT]) is not int
                or meta[_META_PROTOTYPE_COUNT] <= 0):
            raise LearnedDialogueResponseError("SQLite intent count 非法")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != {
                "intent_meta", "intent_feature", "intent_prototype",
                "intent_posting"}:
            raise LearnedDialogueResponseError("SQLite intent table 集漂移")
    except sqlite3.DatabaseError as error:
        raise LearnedDialogueResponseError("SQLite intent index 非法") from error
    finally:
        connection.close()


# object-model: derived_cache; representation=runtime; interop=sqlite-intent-v1
class SqliteLearnedDialogueIntentRuntime:
    """Read-only bounded sparse intent runtime over the portable SQLite schema."""

    __slots__ = (
        "path", "fragment_surfaces", "connection", "train_count",
        "fragment_count")

    def __init__(
            self, path: str | Path,
            fragment_surfaces: tuple[tuple[int, ...], ...],
            ) -> None:
        target = Path(path).resolve()
        validate_sqlite_learned_dialogue_intent_index(
            target, expected_train_count=None,
            expected_fragment_count=len(fragment_surfaces))
        self.path = target
        self.fragment_surfaces = fragment_surfaces
        self.connection = sqlite3.connect(
            f"{target.as_uri()}?mode=ro&immutable=1", uri=True)
        self.connection.execute("PRAGMA query_only=ON")
        meta = dict(self.connection.execute(
            "SELECT key,value FROM intent_meta ORDER BY key"))
        self.train_count = int(meta[_META_TRAIN_COUNT])
        self.fragment_count = int(meta[_META_FRAGMENT_COUNT])

    @staticmethod
    def _read_meta(path: Path, key: int) -> int:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            row = connection.execute(
                "SELECT value FROM intent_meta WHERE key=?", (key,)).fetchone()
        except sqlite3.DatabaseError as error:
            raise LearnedDialogueResponseError("SQLite intent meta 非法") from error
        finally:
            connection.close()
        if row is None or type(row[0]) is not int:
            raise LearnedDialogueResponseError("SQLite intent meta 缺失")
        return row[0]

    def close(self) -> None:
        self.connection.close()

    def _matched_features(
            self, prompt: str, history: tuple[tuple[int, str], ...],
            ) -> tuple[tuple[int, int, int, int], ...]:
        query = dialogue_intent_features(prompt, history=history)
        current = frozenset(dialogue_intent_features(prompt))
        current_keys = frozenset(_feature_key(item) for item in current)
        values = tuple(dict.fromkeys(query))
        rows = []
        for start in range(0, len(values), 300):
            chunk = values[start:start + 300]
            placeholders = ",".join("?" for _ in chunk)
            bindings = tuple(sqlite3.Binary(_feature_key(item)) for item in chunk)
            for feature_id, feature_key, document_count in self.connection.execute(
                    "SELECT feature_id,feature_key,document_count "
                    f"FROM intent_feature WHERE feature_key IN ({placeholders})",
                    bindings):
                rows.append((
                    int(feature_id), int(document_count),
                    int(bytes(feature_key) in current_keys),
                    _feature_weight(self.train_count, int(document_count)),
                ))
        current_rows = sorted(
            (item for item in rows if item[2]),
            key=lambda item: (-item[3], item[1], item[0]))[
                :MAX_CURRENT_QUERY_FEATURES]
        history_rows = sorted(
            (item for item in rows if not item[2]),
            key=lambda item: (-item[3], item[1], item[0]))[
                :MAX_HISTORY_QUERY_FEATURES]
        return tuple(current_rows + history_rows)

    def rank(
            self, prompt: str, *, history: tuple[tuple[int, str], ...] = (),
            minimum_similarity_permille: int,
            ) -> tuple[int, int, int, tuple[int, ...]] | None:
        """Rank one bounded candidate set with the v4 pure-integer score."""
        matched = self._matched_features(prompt, history)
        current = tuple(item for item in matched if item[2])
        if len(current) < MIN_INTENT_SHARED_FEATURES:
            return None
        rare_limit = max(16, self.train_count // 64)
        seeds = tuple(item[0] for item in sorted(
            current, key=lambda item: (item[1], -item[3], item[0]))[
                :MAX_SEED_FEATURES])
        if not seeds:
            return None
        q_values = ",".join("(?,?,?,?)" for _ in matched)
        seed_marks = ",".join("?" for _ in seeds)
        bindings: list[int] = []
        for feature_id, document_count, is_current, weight in matched:
            bindings.extend((
                feature_id, weight, is_current,
                int(document_count <= rare_limit),
            ))
        bindings.extend(seeds)
        query = f"""
            WITH q(feature_id,weight,is_current,is_rare) AS (
                VALUES {q_values}
            ), candidate(prototype_id) AS (
                SELECT prototype_id
                FROM intent_posting
                WHERE feature_id IN ({seed_marks})
                GROUP BY prototype_id
                ORDER BY COUNT(*) DESC, prototype_id
                LIMIT {MAX_PROTOTYPE_CANDIDATES}
            )
            SELECT p.prototype_id,p.fragment_id,p.occurrence_count,
                   p.weight_sum,p.current_weight_sum,
                   SUM(q.weight),SUM(q.weight*q.is_current),COUNT(*),
                   SUM(q.is_current),SUM(q.is_rare),
                   SUM(q.is_rare*q.is_current)
            FROM candidate c
            JOIN intent_prototype p ON p.prototype_id=c.prototype_id
            JOIN intent_posting ip ON ip.prototype_id=c.prototype_id
            JOIN q ON q.feature_id=ip.feature_id
            GROUP BY p.prototype_id,p.fragment_id,p.occurrence_count,
                     p.weight_sum,p.current_weight_sum
        """
        query_weight = sum(item[3] for item in matched)
        current_query_weight = sum(item[3] for item in current)
        best_by_fragment: dict[int, tuple[int, ...]] = {}
        for row in self.connection.execute(query, tuple(bindings)):
            (prototype, fragment, occurrence, width, current_width,
             overlap, current_overlap, shared, current_shared, rare,
             current_rare) = (int(item) for item in row)
            if (shared < MIN_INTENT_SHARED_FEATURES or rare <= 0
                    or current_shared < MIN_INTENT_SHARED_FEATURES
                    or current_rare <= 0 or width <= 0
                    or current_width <= 0):
                continue
            surface = _surface(self.fragment_surfaces[fragment])
            if not _response_surface_allowed(surface, prompt):
                continue
            current_score = (
                (2000 * current_overlap) //
                (current_query_weight + current_width))
            context_score = (
                (2000 * overlap) // (query_weight + width))
            current_coverage = (
                (1000 * current_overlap) // current_query_weight)
            coverage = (1000 * overlap) // query_weight
            rank = (
                current_score, context_score, current_coverage, coverage,
                current_overlap, overlap, current_shared, shared,
                current_rare, rare, occurrence, -prototype,
                fragment, prototype,
            )
            prior = best_by_fragment.get(fragment)
            if prior is None or rank > prior:
                best_by_fragment[fragment] = rank
        ranked = sorted(best_by_fragment.values(), reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        if len(ranked) > 1 and ranked[1][:11] == best[:11]:
            return None
        runner_score = ranked[1][0] if len(ranked) > 1 else 0
        relative_confidence = (1000 if runner_score == 0 else
                               (1000 * best[0])
                               // (best[0] + runner_score))
        if (best[0] < minimum_similarity_permille
                or relative_confidence < minimum_similarity_permille):
            return None
        return (
            best[-2], best[0], best[6],
            (5, len(matched), len(current), len(seeds), len(ranked),
             query_weight, current_query_weight, best[0], runner_score,
             relative_confidence, best[1], best[2], best[3], best[4], best[5],
             best[6], best[7], best[8], best[9], best[10], best[-2],
             best[-1]),
        )


__all__ = [
    "SQLITE_INTENT_FILE", "SQLITE_INTENT_SCHEMA",
    "SqliteLearnedDialogueIntentRuntime",
    "build_sqlite_learned_dialogue_intent_index",
    "validate_sqlite_learned_dialogue_intent_index",
]
