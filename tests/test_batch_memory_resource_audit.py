"""Bounded resource evidence must not create or rewrite Memory artifacts."""
from __future__ import annotations

from contextlib import closing
import hashlib
import sqlite3

import pytest

from pure_integer_ai.experiments.batch_memory_resource_audit import (
    MemoryResourceBudget, MemoryResourceEvidenceError, inspect_batch_memory_resources,
)


def _database(tmp_path):
    path = tmp_path / 'memory.sqlite3'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE memory_event(event_kind INTEGER, object_kind INTEGER, payload_size INTEGER)')
        db.execute('CREATE TABLE identity_header(identity_kind INTEGER, key_size INTEGER)')
        db.executemany('INSERT INTO memory_event VALUES(?,?,?)',
                       [(1, 1, 10), (1, 1, 25), (2, 2, 100), (12, 10, 50)])
        db.executemany('INSERT INTO identity_header VALUES(?,?)', [(1, 37), (91536, 70)])
    return path


def test_header_only_audit_preserves_database_and_does_not_assume_a_budget(tmp_path):
    path = _database(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = inspect_batch_memory_resources(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert report['status'] == 'UNKNOWN_NO_DECLARED_BUDGET'
    assert report['event_count'] == 4
    assert report['max_event_payload_integer_count'] == 100
    assert report['total_event_payload_integer_count'] == 185
    assert report['max_identity_key_integer_count'] == 70
    assert report['owners_inactive_qualification'].startswith('UNKNOWN')
    assert report['peak_working_set_bytes'] is None
    assert report['language_qualification'].startswith('UNKNOWN')
    assert report['free_dialogue_complete'] == 0
    # The fixture deliberately has no source body, part table or graph loader.
    assert {item['event_kind'] for item in report['events']} == {1, 2, 12}


def test_predeclared_limits_report_exact_exceeded_dimensions(tmp_path):
    report = inspect_batch_memory_resources(
        _database(tmp_path), budget=MemoryResourceBudget(90, 60, 180))
    assert report['status'] == 'BUDGET_EXCEEDED'
    assert report['issues'] == ['EVENT_PAYLOAD_INTEGER_BUDGET_EXCEEDED',
                               'IDENTITY_KEY_INTEGER_BUDGET_EXCEEDED',
                               'TOTAL_EVENT_PAYLOAD_INTEGER_BUDGET_EXCEEDED']


def test_budget_boundary_is_inclusive_and_is_only_resource_qualification(tmp_path):
    report = inspect_batch_memory_resources(
        _database(tmp_path), budget=MemoryResourceBudget(100, 70, 185))
    assert report['status'] == 'WITHIN_DECLARED_BUDGET'
    assert report['language_qualification'].startswith('UNKNOWN')


def test_missing_database_or_size_schema_cannot_create_zero_evidence(tmp_path):
    missing = tmp_path / 'missing.sqlite3'
    with pytest.raises(FileNotFoundError):
        inspect_batch_memory_resources(missing)
    assert not missing.exists()
    with closing(sqlite3.connect(missing)) as db, db:
        db.execute('CREATE TABLE unrelated(value INTEGER)')
    with pytest.raises(MemoryResourceEvidenceError, match='missing memory_event'):
        inspect_batch_memory_resources(missing)


@pytest.mark.parametrize('value', [0, -1, 1.5, 'bad'])
def test_invalid_database_integer_metadata_is_unknown(tmp_path, value):
    path = _database(tmp_path)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('UPDATE memory_event SET payload_size=? WHERE event_kind=2', (value,))
    with pytest.raises(MemoryResourceEvidenceError, match='invalid integer'):
        inspect_batch_memory_resources(path)


def test_empty_metadata_and_boolean_budgets_are_not_positive_evidence(tmp_path):
    path = _database(tmp_path)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('DELETE FROM identity_header')
    with pytest.raises(MemoryResourceEvidenceError, match='empty Memory'):
        inspect_batch_memory_resources(path)
    with pytest.raises(MemoryResourceEvidenceError, match='positive integers'):
        MemoryResourceBudget(True, 1, 1)


def test_audit_closes_readonly_connection_on_success_and_schema_error(tmp_path, monkeypatch):
    path = _database(tmp_path)
    invalid = tmp_path / 'invalid.sqlite3'
    with closing(sqlite3.connect(invalid)) as db, db:
        db.execute('CREATE TABLE unrelated(value INTEGER)')
    connect = sqlite3.connect
    opened = []

    def record_connection(*args, **kwargs):
        db = connect(*args, **kwargs)
        opened.append(db)
        return db

    monkeypatch.setattr(sqlite3, 'connect', record_connection)
    inspect_batch_memory_resources(path)
    with pytest.raises(MemoryResourceEvidenceError, match='missing memory_event'):
        inspect_batch_memory_resources(invalid)
    assert len(opened) == 2
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
            db.execute('SELECT 1')
