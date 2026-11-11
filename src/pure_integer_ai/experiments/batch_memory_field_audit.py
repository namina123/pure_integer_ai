"""Attribute completed Memory wire sizes without restoring recursive objects.

Only length/count markers and reference headers are inspected. Unread key
contents and domain validity remain unknown. This is a resource diagnostic,
not a query, a graph restore, a resource-budget pass or a language evaluation.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from pathlib import Path
import sqlite3


class MemoryFieldAuditError(ValueError):
    """Missing, drifting or unsupported wire evidence cannot be attributed."""


class _Cursor:
    def __init__(self, word, start: int, end: int):
        self.word = word
        self.position = start
        self.end = end

    def scalar(self) -> int:
        if self.position >= self.end:
            raise MemoryFieldAuditError('truncated wire marker')
        value = self.word(self.position)
        self.position += 1
        if type(value) is not int:
            raise MemoryFieldAuditError('wire marker is not an integer')
        return value

    def sized(self, *, optional: bool = False) -> tuple[int, int]:
        size = self.scalar()
        if size < 0 or (size == 0 and not optional) or self.position + size > self.end:
            raise MemoryFieldAuditError('invalid wire field length')
        start = self.position
        self.position += size
        return start, size

    def count(self) -> int:
        value = self.scalar()
        if value < 0 or value > 4096 or value > self.end - self.position:
            raise MemoryFieldAuditError('invalid or excessive wire group count')
        return value

    def finish(self) -> None:
        if self.position != self.end:
            raise MemoryFieldAuditError('trailing wire fields')


def attribute_memory_payload(event_kind: int, payload_size: int, word) -> dict:
    """Describe V1 O/Episode/manifest fields using random-access integer reads.

    No domain objects, source text or child keys are reconstructed. Counts of
    embedded object-key integers include each stored occurrence separately.
    """
    if type(payload_size) is not int or payload_size <= 0:
        raise MemoryFieldAuditError('invalid payload size')
    raw_word = word

    def word(position: int) -> int:
        value = raw_word(position)
        if type(value) is not int:
            raise MemoryFieldAuditError('wire marker is not an integer')
        return value

    cursor = _Cursor(word, 0, payload_size)
    if cursor.scalar() != 1:
        raise MemoryFieldAuditError('unsupported payload wire version')
    fields = []

    def memory(start: int, size: int, item: dict) -> None:
        if (size < 15 or word(start) != 1 or word(start + 1) != 2
                or type(word(start + 12)) is not int
                or word(start + 12) not in range(1, 12)
                or type(word(start + 13)) is not int
                or word(start + 13) != size - 14):
            raise MemoryFieldAuditError('unsupported MemoryObjectRef wire header')
        item['embedded_memory_ref_count'] += 1
        item['embedded_object_key_integers'] += size - 14
        item['max_embedded_object_key_integers'] = max(
            item['max_embedded_object_key_integers'], size - 14)
        item['memory_object_kinds'][word(start + 12)] += 1

    def link(start: int, size: int, item: dict) -> None:
        if (size < 4 or word(start) != 1 or word(start + 1) not in range(1, 5)
                or word(start + 2) != size - 3):
            raise MemoryFieldAuditError('unsupported MemoryLinkedRef wire header')
        if word(start + 1) == 2:
            memory(start + 3, size - 3, item)

    def binding(start: int, size: int, item: dict) -> None:
        nested = _Cursor(word, start, start + size)
        binding_kind = nested.scalar()
        if binding_kind not in range(1, 6):
            raise MemoryFieldAuditError('unsupported manifest binding kind')
        nested.sized()  # lineage is counted, never decoded
        ref_start, ref_size = nested.sized()
        memory(ref_start, ref_size, item)
        nested.finish()

    def field(name: str, *, parser=None, group=False, optional=False,
              scalar=False, typed=False) -> None:
        initial = cursor.position
        item = {
            'name': name, 'stored_integers': 0, 'member_count': 0,
            'embedded_memory_ref_count': 0, 'embedded_object_key_integers': 0,
            'max_embedded_object_key_integers': 0, 'memory_object_kinds': Counter(),
        }
        if scalar:
            cursor.scalar()
        else:
            count = cursor.count() if group else 1
            item['member_count'] = count
            if typed:
                # V1 TypedRef groups are flat fixed-width keys, not sized members.
                cursor.position += 11 * count
                if cursor.position > cursor.end:
                    raise MemoryFieldAuditError('truncated flat TypedRef group')
            else:
                for _ in range(count):
                    start, size = cursor.sized(optional=optional)
                    if size and parser is not None:
                        parser(start, size, item)
        item['stored_integers'] = cursor.position - initial
        item['memory_object_kinds'] = dict(sorted(item['memory_object_kinds'].items()))
        fields.append(item)

    if event_kind == 1:
        field('source')
        field('context', parser=link)
        field('concept_refs', group=True, typed=True)
        field('ordered_refs', group=True, typed=True)
        field('structure_ref', optional=True)
        field('proposition_refs', group=True, typed=True)
        field('relation_occurrences', parser=link, group=True)
        field('observed_at')
    elif event_kind == 2:
        field('input_observation_ref', parser=memory)
        field('intent_ref', parser=link, optional=True)
        field('candidate_refs', parser=link, group=True)
        field('selected_path_ref', parser=link, optional=True)
        field('output_ref', parser=link, optional=True)
        field('used_memory_refs', parser=memory, group=True)
        field('result_signal_refs', parser=link, group=True)
        field('failure_ref', parser=link, optional=True)
        field('round_seq', scalar=True)
        field('session_ref')
        field('created_at')
    elif event_kind == 12:
        field('source')
        field('batch_id', scalar=True)
        field('parser_version', scalar=True)
        field('outcome_kind', scalar=True)
        field('bindings', parser=binding, group=True)
        field('supersedes_manifest_ref', parser=memory, optional=True)
        field('completed_at')
    else:
        raise MemoryFieldAuditError('unsupported event kind for field attribution')
    cursor.finish()
    return {
        'event_kind': event_kind, 'payload_integers': payload_size,
        'protocol_version_integers': 1, 'fields': fields,
        'embedded_memory_ref_count': sum(item['embedded_memory_ref_count'] for item in fields),
        'embedded_object_key_integers': sum(item['embedded_object_key_integers'] for item in fields),
        'conservation': int(1 + sum(item['stored_integers'] for item in fields) == payload_size),
    }


def inspect_memory_field_sizes(database: str | Path, *, max_events: int = 256,
                               max_chunk_reads: int = 20000) -> dict:
    """Read only marker chunks of a stable SQLite snapshot and return sizes.

    Limits bound diagnostic work; they are not retrospective training budgets.
    The caller must independently establish owner inactivity for final audits.
    """
    if any(type(value) is not int or value <= 0 for value in (max_events, max_chunk_reads)):
        raise MemoryFieldAuditError('diagnostic limits require positive integers')
    path = Path(database).resolve(strict=True)
    before = path.stat()
    chunk_reads = 0
    integers_fetched = 0
    events = []
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1)) as db:
        db.execute('PRAGMA query_only=ON')
        version = db.execute('PRAGMA data_version').fetchone()[0]
        db.execute('BEGIN')
        headers = db.execute(
            'SELECT event_hash, event_kind, payload_size, timeline_seq '
            'FROM memory_event WHERE event_kind IN (1,2,12) '
            'ORDER BY timeline_seq, event_hash LIMIT ?', (max_events + 1,)).fetchall()
        if not headers or len(headers) > max_events:
            raise MemoryFieldAuditError('empty evidence or event inspection limit exceeded')
        columns = ','.join(f'part_{index:02d}' for index in range(16))
        for event_hash, event_kind, size, timeline_seq in headers:
            if (any(type(value) is not int or value <= 0 for value in
                    (event_hash, event_kind, size, timeline_seq))):
                raise MemoryFieldAuditError('invalid event header integers')
            expected_chunks = (size + 15) // 16
            summary = db.execute(
                'SELECT COUNT(*), COUNT(DISTINCT chunk_index), MIN(chunk_index), '
                'MAX(chunk_index), SUM(part_size) FROM memory_event_part WHERE event_hash=?',
                (event_hash,)).fetchone()
            if summary != (expected_chunks, expected_chunks, 0, expected_chunks - 1, size):
                raise MemoryFieldAuditError('payload chunk extent disagrees with envelope')
            cache = {}

            def word(position: int) -> int:
                nonlocal chunk_reads, integers_fetched
                if position < 0 or position >= size:
                    raise MemoryFieldAuditError('marker position outside payload')
                index, offset = divmod(position, 16)
                if index not in cache:
                    if chunk_reads >= max_chunk_reads:
                        raise MemoryFieldAuditError('marker chunk inspection limit exceeded')
                    rows = db.execute(
                        f'SELECT part_size,{columns} FROM memory_event_part '
                        'WHERE event_hash=? AND chunk_index=? LIMIT 2',
                        (event_hash, index)).fetchall()
                    if len(rows) != 1:
                        raise MemoryFieldAuditError('marker chunk missing or duplicated')
                    count, *parts = rows[0]
                    if (count != min(16, size - index * 16)
                            or any(type(value) is not int for value in (count, *parts))
                            or any(value != 0 for value in parts[count:])):
                        raise MemoryFieldAuditError('marker chunk has invalid size/integers/padding')
                    cache[index] = parts
                    chunk_reads += 1
                    integers_fetched += count
                return cache[index][offset]

            event = attribute_memory_payload(event_kind, size, word)
            event.update(event_hash=event_hash, timeline_seq=timeline_seq,
                         marker_chunks_read=len(cache), payload_chunks=expected_chunks)
            events.append(event)
        db.rollback()
        if db.execute('PRAGMA data_version').fetchone()[0] != version:
            raise MemoryFieldAuditError('Memory changed during field attribution')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise MemoryFieldAuditError('Memory file changed during field attribution')
    return {
        'format': 'PURE_INTEGER_MEMORY_FIELD_SIZE_DIAGNOSTIC_V1',
        'status': 'ATTRIBUTED_WIRE_SIZES_ONLY', 'database': str(path),
        'database_bytes': after.st_size, 'events': events,
        'event_count': len(events), 'marker_chunk_reads': chunk_reads,
        'payload_integers_fetched_in_marker_chunks': integers_fetched,
        'total_selected_payload_integers': sum(item['payload_integers'] for item in events),
        'source_text_decoded': 0, 'domain_objects_restored': 0,
        'unread_payload_content_integrity': 'UNKNOWN',
        'domain_validity': 'UNKNOWN_NOT_VALIDATED_BY_SIZE_ATTRIBUTION',
        'owners_inactive_qualification': 'UNKNOWN_REQUIRES_INDEPENDENT_PROCESS_EVIDENCE',
        'runtime_cost_causality': 'UNKNOWN_REQUIRES_PHASE_PROFILE',
        'language_qualification': 'UNKNOWN_NOT_MEASURED', 'free_dialogue_complete': 0,
    }
