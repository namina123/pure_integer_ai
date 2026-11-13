"""Compile explicitly licensed role substitution into native reference frames."""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


REFERENCE_ROLE_LEARNING_VERSION = 94180


def _key(value):
    if (type(value) is not list or not value
            or any(type(item) is not int or not 0 <= item < (1 << 63) for item in value)):
        raise ValueError('reference role instruction requires complete integer identities')
    return tuple(value)


def compile_train_reference_role_record(record, *, trained_frames, role_contracts):
    """Teach one explicit slot using mounted parents and an unchanged type contract.

    This training function neither resolves a referent nor grants a Core fact.
    The returned frame uses the existing runtime syntax and competition gates.
    """
    fields = {'owner', 'split', 'license_id', 'parent_frame_key', 'marker_frame_key',
              'reference_role_key', 'allowed_object_kinds'}
    if (type(record) is not dict or set(record) != fields
            or (record['owner'], record['split'], record['license_id'])
            != ('teacher', 'train', 'CC0-1.0')):
        raise ValueError('reference role learning requires a licensed teacher train instruction')
    parent_key, marker_key, role, kinds = (_key(record[name]) for name in (
        'parent_frame_key', 'marker_frame_key', 'reference_role_key', 'allowed_object_kinds'))
    if kinds != tuple(sorted(set(kinds))):
        raise ValueError('reference role kinds must be a canonical unchanged contract')
    if (type(trained_frames) is not tuple
            or any(not isinstance(item, TrainedRelationQuestionFrame) for item in trained_frames)
            or len({item.stable_key() for item in trained_frames}) != len(trained_frames)):
        raise ValueError('reference role learning requires complete unique mounted frames')
    mounted = {item.stable_key(): item for item in trained_frames}
    parent, marker = mounted.get(parent_key), mounted.get(marker_key)
    if (parent is None or marker is None or parent.is_slot_question
            or not marker.is_reference_slot or role not in parent.roles):
        raise ValueError('reference role learning lacks a complete parent or trained reference marker')
    if any(sum(item.source_case_key == candidate.source_case_key for item in trained_frames) != 1
           for candidate in (parent, marker)):
        raise ValueError('reference role parent source is not uniquely recoverable')
    contracts = tuple(item for item in role_contracts
                      if item[:2] == (parent.predicate_key, role))
    if len(contracts) != 1 or contracts[0][2] != kinds:
        raise ValueError('reference role instruction differs from its mounted type contract')
    values = marker.wh_values
    if any(gap[index:index + len(values)] == values
           for gap in parent.gaps for index in range(len(gap) - len(values) + 1)):
        raise ValueError('reference marker already occurs in another literal interval')
    raw = canonical_json_bytes(record)
    if len(raw) > 1024 * 1024:
        raise ValueError('reference role instruction exceeds its source budget')
    digest = hashlib.sha256(raw).digest()
    digest_key = tuple(int.from_bytes(digest[index:index + 4], 'big') for index in range(0, 32, 4))
    source = pack_record(REFERENCE_ROLE_LEARNING_VERSION, digest_key,
                         parent.source_case_key, marker.source_case_key, role, kinds)
    index = parent.roles.index(role)
    gaps = (*parent.gaps[:index], parent.gaps[index] + values + parent.gaps[index + 1],
            *parent.gaps[index + 2:])
    return TrainedRelationQuestionFrame(source, parent.operation_key, parent.predicate_key,
        parent.roles[:index] + parent.roles[index + 1:], gaps, role, values, SLOT_MODE_REFERENCE)
