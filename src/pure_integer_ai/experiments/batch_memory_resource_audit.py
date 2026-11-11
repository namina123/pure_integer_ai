"""Read completed batch Memory metadata without restoring recursive graph keys.

This is resource evidence only. Budgets must be declared before a run, and
neither a small database nor a short key proves any language capability.
Callers must establish that run owners have exited before final qualification.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3


class MemoryResourceEvidenceError(ValueError):
    """The database does not supply stable, valid integer-size evidence."""


@dataclass(frozen=True, slots=True)
class MemoryResourceBudget:
    max_event_payload_integers: int
    max_identity_key_integers: int
    max_total_event_payload_integers: int

    def __post_init__(self) -> None:
        for value in (self.max_event_payload_integers,
                      self.max_identity_key_integers,
                      self.max_total_event_payload_integers):
            if type(value) is not int or value <= 0:
                raise MemoryResourceEvidenceError("resource budgets require positive integers")


def _schema(db: sqlite3.Connection, table: str, fields: tuple[str, ...]) -> None:
    # These identifiers are private constants; database content is never SQL.
    columns = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
    if not set(fields).issubset(columns):
        raise MemoryResourceEvidenceError(f"missing {table} size evidence; resource state is unknown")


def inspect_batch_memory_resources(
        database: str | Path, *, budget: MemoryResourceBudget | None = None,
        ) -> dict:
    """Inspect header counts only; never read source text or integer-key parts.

    A missing database is an error and is never created. The read transaction
    provides one SQLite snapshot; observed file/version drift rejects a final
    report. This does not establish process liveness or historical peak usage.
    """
    if budget is not None and not isinstance(budget, MemoryResourceBudget):
        raise MemoryResourceEvidenceError("budget requires a frozen MemoryResourceBudget")
    path = Path(database).resolve(strict=True)
    if not path.is_file():
        raise MemoryResourceEvidenceError("Memory database is not a file")
    before = path.stat()
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1)) as db:
            db.execute('PRAGMA query_only=ON')
            version = db.execute('PRAGMA data_version').fetchone()[0]
            db.execute('BEGIN')
            _schema(db, 'memory_event', ('event_kind', 'object_kind', 'payload_size'))
            _schema(db, 'identity_header', ('identity_kind', 'key_size'))
            for table, fields in (
                    ('memory_event', ('event_kind', 'object_kind', 'payload_size')),
                    ('identity_header', ('identity_kind', 'key_size'))):
                invalid = ' OR '.join(f"typeof({field}) <> 'integer' OR {field} <= 0"
                                      for field in fields)
                if db.execute(f'SELECT 1 FROM {table} WHERE {invalid} LIMIT 1').fetchone():
                    raise MemoryResourceEvidenceError(f"invalid integer size metadata: {table}")
            events = db.execute(
                'SELECT event_kind, object_kind, COUNT(*), MIN(payload_size), '
                'MAX(payload_size), SUM(payload_size) FROM memory_event '
                'GROUP BY event_kind, object_kind ORDER BY event_kind, object_kind').fetchall()
            identities = db.execute(
                'SELECT identity_kind, COUNT(*), MAX(key_size), SUM(key_size) '
                'FROM identity_header GROUP BY identity_kind ORDER BY identity_kind').fetchall()
            db.rollback()
            if db.execute('PRAGMA data_version').fetchone()[0] != version:
                raise MemoryResourceEvidenceError("Memory database changed during resource audit")
    except sqlite3.Error as error:
        raise MemoryResourceEvidenceError(f"unreadable Memory resource evidence: {error}") from error
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise MemoryResourceEvidenceError("Memory database file changed during resource audit")
    if not events or not identities:
        raise MemoryResourceEvidenceError("empty Memory resource evidence; resource state is unknown")
    max_event = max(row[4] for row in events)
    max_identity = max(row[2] for row in identities)
    total_event = sum(row[5] for row in events)
    issues = []
    declared = None
    if budget is not None:
        declared = {
            'max_event_payload_integers': budget.max_event_payload_integers,
            'max_identity_key_integers': budget.max_identity_key_integers,
            'max_total_event_payload_integers': budget.max_total_event_payload_integers,
        }
        for label, actual, limit in (
                ('EVENT_PAYLOAD', max_event, budget.max_event_payload_integers),
                ('IDENTITY_KEY', max_identity, budget.max_identity_key_integers),
                ('TOTAL_EVENT_PAYLOAD', total_event, budget.max_total_event_payload_integers)):
            if actual > limit:
                issues.append(f'{label}_INTEGER_BUDGET_EXCEEDED')
    return {
        'format': 'PURE_INTEGER_BATCH_MEMORY_RESOURCE_METADATA_V1',
        'status': ('BUDGET_EXCEEDED' if issues else 'WITHIN_DECLARED_BUDGET')
                  if budget is not None else 'UNKNOWN_NO_DECLARED_BUDGET',
        'database': str(path), 'database_bytes': after.st_size,
        'event_count': sum(row[2] for row in events),
        'max_event_payload_integer_count': max_event,
        'total_event_payload_integer_count': total_event,
        'max_identity_key_integer_count': max_identity,
        'total_identity_key_integer_count': sum(row[3] for row in identities),
        'events': [dict(event_kind=row[0], object_kind=row[1], count=row[2],
                        min_payload_integer_count=row[3], max_payload_integer_count=row[4],
                        total_payload_integer_count=row[5]) for row in events],
        'identities': [dict(identity_kind=row[0], count=row[1],
                            max_key_integer_count=row[2], total_key_integer_count=row[3])
                       for row in identities],
        'declared_budget': declared, 'issues': issues,
        'owners_inactive_qualification': 'UNKNOWN_requires_independent_process_evidence',
        'peak_working_set_bytes': None,
        'query_reads': None, 'context_depth': None,
        'language_qualification': 'UNKNOWN_not_measured_by_resource_metadata',
        'free_dialogue_complete': 0,
    }
