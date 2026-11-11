"""Wire attribution agrees with real serializers and rejects broken boundaries."""
from contextlib import closing
import hashlib
import sqlite3

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_PROPOSITION, OwnerScope, ParserVersion, SourceRef, TypedRef,
    VersionBundle, VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload, IntakeDerivedBinding, IntakeManifestPayload,
    MemoryLinkedRef, MemoryObjectRef, ObservationPayload,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED, LogicalClockIdentity, LogicalTimestamp, session_scope,
)
from pure_integer_ai.experiments.batch_memory_field_audit import (
    MemoryFieldAuditError, attribute_memory_payload, inspect_memory_field_sizes,
)
from pure_integer_ai.storage.spaces.registry import SpaceIdentity


def _payloads():
    owner = OwnerScope(1, 2, 3, VISIBILITY_SESSION)
    versions = VersionBundle(parser=ParserVersion(1))
    source = SourceRef(1, 1, 1, owner, versions)
    scope = session_scope(3, owner=owner, versions=versions)
    timestamp = LogicalTimestamp(LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED), 1)
    concept = TypedRef(OBJECT_CONCEPT, 1, 1, owner, versions)
    proposition = TypedRef(OBJECT_PROPOSITION, 1, 2, owner, versions)
    leaf = MemoryObjectRef(SpaceIdentity(2, 1, 1), owner, versions, 1, (9, 8, 7))
    observation = ObservationPayload(
        source, MemoryLinkedRef.core(concept), (concept, concept),
        (concept, concept), None, (proposition,), (MemoryLinkedRef.memory(leaf),), timestamp)
    ref = MemoryObjectRef(leaf.memory_space, owner, versions, 1, observation.stable_key())
    episode = EpisodePayload(
        ref, None, (MemoryLinkedRef.memory(leaf),), None,
        MemoryLinkedRef.memory(ref), (leaf,), (), None, 1, scope, timestamp)
    manifest = IntakeManifestPayload(
        source, 1, versions.parser.value, 1,
        (IntakeDerivedBinding(1, (1,), ref),), None, timestamp)
    return [(1, observation.stable_key()), (2, episode.stable_key()),
            (12, manifest.stable_key())], leaf, ref


def test_sizes_match_real_flat_typed_groups_and_full_memory_ref_serializers():
    payloads, leaf, ref = _payloads()
    reports = [attribute_memory_payload(kind, len(key), key.__getitem__)
               for kind, key in payloads]
    assert all(row['conservation'] == 1 for row in reports)
    fields = {row['name']: row for row in reports[0]['fields']}
    assert fields['concept_refs']['stored_integers'] == 1 + 2 * 11
    assert fields['ordered_refs']['stored_integers'] == 1 + 2 * 11
    assert fields['proposition_refs']['stored_integers'] == 1 + 11
    assert fields['relation_occurrences']['embedded_object_key_integers'] == len(leaf.object_key)
    assert reports[1]['embedded_memory_ref_count'] == 4
    assert reports[1]['embedded_object_key_integers'] == 2 * (len(ref.object_key) + len(leaf.object_key))
    assert reports[2]['embedded_object_key_integers'] == len(ref.object_key)


@pytest.mark.parametrize('mutate', [
    lambda key: (2, *key[1:]),
    lambda key: (True, *key[1:]),
    lambda key: key[:-1],
    lambda key: (*key, 1),
    lambda key: (key[0], len(key) + 100, *key[2:]),
])
def test_invalid_versions_lengths_and_tail_are_rejected(mutate):
    key = mutate(_payloads()[0][0][1])
    with pytest.raises(MemoryFieldAuditError):
        attribute_memory_payload(1, len(key), key.__getitem__)


def _database(tmp_path):
    path = tmp_path / 'memory.sqlite3'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE memory_event(event_hash INTEGER, event_kind INTEGER, payload_size INTEGER, timeline_seq INTEGER)')
        columns = ','.join(f'part_{index:02d} INTEGER' for index in range(16))
        db.execute(f'CREATE TABLE memory_event_part(event_hash INTEGER, chunk_index INTEGER, part_size INTEGER,{columns})')
        db.execute('CREATE INDEX event_chunk ON memory_event_part(event_hash, chunk_index)')
        for event_hash, (kind, key) in enumerate(_payloads()[0], start=1):
            db.execute('INSERT INTO memory_event VALUES(?,?,?,?)', (event_hash, kind, len(key), event_hash))
            for start in range(0, len(key), 16):
                part = key[start:start + 16]
                db.execute('INSERT INTO memory_event_part VALUES(' + ','.join('?' * 19) + ')',
                           (event_hash, start // 16, len(part), *part, *([0] * (16 - len(part)))))
    return path


def test_readonly_sparse_audit_preserves_sha_and_claims_only_field_sizes(tmp_path):
    path = _database(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = inspect_memory_field_sizes(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert report['event_count'] == 3
    assert report['domain_objects_restored'] == report['source_text_decoded'] == 0
    assert report['payload_integers_fetched_in_marker_chunks'] < report['total_selected_payload_integers']
    assert report['domain_validity'].startswith('UNKNOWN')
    assert report['runtime_cost_causality'].startswith('UNKNOWN')
    assert report['owners_inactive_qualification'].startswith('UNKNOWN')


def test_missing_chunks_and_work_limits_fail_without_creating_missing_database(tmp_path):
    missing = tmp_path / 'missing.sqlite3'
    with pytest.raises(FileNotFoundError):
        inspect_memory_field_sizes(missing)
    assert not missing.exists()
    path = _database(tmp_path)
    with pytest.raises(MemoryFieldAuditError, match='event inspection limit'):
        inspect_memory_field_sizes(path, max_events=2)
    with pytest.raises(MemoryFieldAuditError, match='chunk inspection limit'):
        inspect_memory_field_sizes(path, max_chunk_reads=1)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('DELETE FROM memory_event_part WHERE event_hash=1 AND chunk_index=1')
    with pytest.raises(MemoryFieldAuditError, match='extent'):
        inspect_memory_field_sizes(path)
