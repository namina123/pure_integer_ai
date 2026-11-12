"""由已许可typed教师来源学习程序消费权限，保持程序、语言与真值门不变。"""
from __future__ import annotations

from dataclasses import replace
import hashlib

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef
from pure_integer_ai.cognition.shared.integer_operation_permission import (
    OperationSourcePermission, PERMISSION_SOURCE,
)
from pure_integer_ai.cognition.shared.integer_operation_program import OperationProgramWitness
from pure_integer_ai.cognition.shared.relation_composition import schema_key
from pure_integer_ai.experiments.operation_fact_source import verify_operation_fact_source
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_w06_identity_protocol import W06_IDENTITY_VERSIONS


def compile_train_operation_source_permission_record(row, facts, witness):
    """只从完整teacher/train/CC0课程授权同schema来源；不学习事实或表面。

    教师指令显式命名原始程序及新增课程身份。所有已有增量完整保留，新的
    来源集合只能由已核验课程中与实际程序schema相同的候选派生；调用方不能
    直接传入任意source列表。不同许可证、角色类型或schema一律拒绝。
    """
    if not isinstance(witness, OperationProgramWitness):
        raise TypeError('source permission requires a complete current program witness')
    if (type(row) is not dict or set(row) != {'base_witness_key', 'course_source_identity',
            'owner', 'split', 'license_id'} or row['owner'] != 'teacher'
            or row['split'] != 'train' or row['license_id'] != 'CC0-1.0'
            or type(row['base_witness_key']) is not list
            or type(row['course_source_identity']) is not list
            or any(type(value) is not int or value < 0 or value >= 1 << 63
                   for field in ('base_witness_key', 'course_source_identity') for value in row[field])):
        raise ValueError('source permission requires an explicit CC0 train teacher instruction')
    base = (witness.stable_key() if not witness.source_extensions
            else witness.source_extensions[0].base_witness_key)
    if tuple(row['base_witness_key']) != base:
        raise ValueError('source permission instruction names a different original program')
    adapter, _seeds = verify_operation_fact_source(facts)
    if (tuple(row['course_source_identity']) != facts.course.source_identity
            or facts.course.license_identity != tuple(map(ord, row['license_id']))
            or witness.license_identity != facts.course.license_identity):
        raise ValueError('source permission course identity or license drifted')
    schemas = {atom.schema.relation: schema_key(atom.schema) for atom in witness.program.atoms}
    selected = tuple(candidate for candidate in adapter.candidates
                     if candidate.schema.relation in schemas)
    if not selected or any(schema_key(candidate.schema) != schemas[candidate.schema.relation]
                           for candidate in selected):
        raise ValueError('source permission has no matching complete program schema')
    raw = canonical_json_line(row)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('source permission teacher instruction exceeds its bounded record')
    digest = hashlib.sha256(raw).digest()
    permission = OperationSourcePermission(base,
        SourceRef(PERMISSION_SOURCE, (int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)) or 1,
                  0, GLOBAL_OWNER_SCOPE, W06_IDENTITY_VERSIONS),
        tuple(digest), facts.course.source_identity, facts.course.license_identity,
        tuple(sorted({schema_key(candidate.schema) for candidate in selected})),
        tuple(sorted({candidate.source_ref for candidate in selected}, key=SourceRef.stable_key)))
    extensions = tuple(sorted(set((*witness.source_extensions, permission)),
                              key=OperationSourcePermission.stable_key))
    sources = tuple(sorted(set(witness.authorized_forming_sources) | set(permission.forming_sources),
                           key=SourceRef.stable_key))
    return replace(witness, authorized_forming_sources=sources, source_extensions=extensions)
