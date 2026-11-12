"""来源增量保留完整程序，未授权/伪造/schema漂移仍由实际执行与冷核拒绝。"""
from dataclasses import replace
import hashlib

import pytest

from test_integer_operation_program import context, _source, _witness, _schema, _fact
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.integer_operation_execution import READY, UNKNOWN, execute_integer_operation
from pure_integer_ai.cognition.shared.integer_operation_permission import OperationSourcePermission, PERMISSION_SOURCE
from pure_integer_ai.cognition.shared.integer_operation_program import OperationProgramWitness, WITNESS, SEQUENCE
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.relation_composition import schema_key
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.experiments.trained_operation_program_graph import materialize_operation_program_graph, read_operation_program_graph


def _permission(base, schemas, sources, number=1):
    """固定整数测试指令来源，不包含自然语言或事实目标。"""
    digest = hashlib.sha256(bytes((number,))).digest()
    source = SourceRef(PERMISSION_SOURCE,
        (int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)) or 1,
        0, base.source.owner, base.source.versions)
    return OperationSourcePermission(base.stable_key(), source, tuple(digest), (92890, number),
        base.license_identity, tuple(sorted(map(schema_key, schemas))),
        tuple(sorted(set(sources), key=SourceRef.stable_key)))


def _extended(context):
    """使用原独立证据fixture，第一来源不变，只显式授权第二个前提来源。"""
    *_, first, second, facts, program = context
    base = _witness(program, facts[:1])
    permission = _permission(base, (second,), facts[1].candidate.forming_sources)
    current = replace(base, source_extensions=(permission,), authorized_forming_sources=tuple(sorted(
        set(base.authorized_forming_sources) | set(permission.forming_sources), key=SourceRef.stable_key)))
    return base, permission, current


def test_legacy_bytes_unchanged_and_extended_graph_roundtrip(context, tmp_path):
    """旧六字段逐字节保留，新完整许可通过现役typed图存取而非正文侧库。"""
    base, permission, current = _extended(context)
    fields = unpack_record(base.stable_key(), WITNESS)
    assert len(fields) == 6
    assert base.stable_key() == pack_record(WITNESS, *fields)
    assert OperationProgramWitness.from_stable_key(base.stable_key()) == base
    assert len(unpack_record(current.stable_key(), WITNESS)) == 7
    assert OperationSourcePermission.from_stable_key(permission.stable_key()) == permission
    assert OperationProgramWitness.from_stable_key(current.stable_key()) == current
    database = tmp_path / 'programs.sqlite3'
    materialize_operation_program_graph((current,), database)
    assert read_operation_program_graph(database) == (current,)


def test_permission_changes_actual_execution_without_granting_truth(context):
    """同输入与事实下原程序拒绝，新许可实际闭合；反驳前提仍阻断。"""
    a, b, c, _, _, second, facts, _ = context
    base, _, current = _extended(context)
    variables = ((1, a), (2, b))
    original = execute_integer_operation(base, (92891, 1), variables, facts[:2])
    assert original.status == UNKNOWN
    actual = execute_integer_operation(current, (92891, 1), variables, facts[:2])
    assert actual.status == READY
    verify_integer_operation(actual, facts[:2], (current,))
    with pytest.raises(ValueError):
        verify_integer_operation(actual, facts[:2], (base,))
    unsupported = _fact(2, second.relation, b, c, stance=2)
    assert execute_integer_operation(current, (92891, 1), variables, (facts[0], unsupported)).status != READY


@pytest.mark.parametrize('change', ('extra_source', 'missing_source', 'foreign_program', 'foreign_license', 'foreign_schema', 'duplicate', 'recursive'))
def test_permission_cannot_smuggle_sources_or_change_original_contract(context, change):
    """完整根、schema、来源精确并集和规范成员逐项失败关闭。"""
    base, permission, current = _extended(context)
    with pytest.raises(ValueError):
        if change == 'extra_source':
            replace(current, authorized_forming_sources=tuple(sorted(
                (*current.authorized_forming_sources, _source(9999)), key=SourceRef.stable_key)))
        elif change == 'missing_source':
            replace(current, authorized_forming_sources=base.authorized_forming_sources)
        elif change == 'foreign_program':
            replace(current, program=replace(current.program, operation_key=(92891, 2)))
        elif change == 'foreign_license':
            replace(current, source_extensions=(replace(permission, license_identity=(92892,)),))
        elif change == 'foreign_schema':
            replace(current, source_extensions=(replace(permission, schema_keys=(schema_key(_schema(99)),)),))
        elif change == 'duplicate':
            replace(current, source_extensions=(permission, permission))
        else:
            replace(permission, base_witness_key=current.stable_key())


def test_empty_seventh_field_and_truncated_permission_are_not_canonical(context):
    """空扩展不得为旧witness制造第二种编码，截断不得以缺省许可恢复。"""
    base, permission, _ = _extended(context)
    with pytest.raises(ValueError):
        OperationProgramWitness.from_stable_key(pack_record(WITNESS,
            *unpack_record(base.stable_key(), WITNESS), pack_record(SEQUENCE)))
    with pytest.raises(ValueError):
        OperationSourcePermission.from_stable_key(permission.stable_key()[:-1])


def test_multiple_permissions_preserve_original_root_and_canonical_order(context):
    """不同课程来源可单调累积，原始程序本体和全部先前许可成员保持。"""
    base, first, current = _extended(context)
    _, _, _, _, _, schema, facts, _ = context
    second = _permission(base, (schema,), facts[2].candidate.forming_sources, number=2)
    extensions = tuple(sorted((first, second), key=OperationSourcePermission.stable_key))
    result = replace(current, source_extensions=extensions, authorized_forming_sources=tuple(sorted(
        set(current.authorized_forming_sources) | set(second.forming_sources), key=SourceRef.stable_key)))
    assert result.program == base.program
    assert OperationProgramWitness.from_stable_key(result.stable_key()) == result
    assert set(current.authorized_forming_sources) < set(result.authorized_forming_sources)
    with pytest.raises(ValueError):
        replace(result, source_extensions=tuple(reversed(extensions)))
