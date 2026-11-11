"""typed教师必须通过完整物理来源核验后才能训练程序和自然请求。"""
from dataclasses import replace
import sqlite3

import pytest

from test_authored_relation_sqlite_course import SOURCE_ROOT, _course
from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationAtom, OperationProjection,
)
from pure_integer_ai.experiments.operation_fact_source import (
    SQLITE_OPERATION_SOURCE, read_sqlite_operation_fact_source, verify_operation_fact_source,
)
from pure_integer_ai.experiments.operation_program_course import (
    compile_train_operation_program, compile_train_operation_program_record,
)
from pure_integer_ai.experiments.operation_request_course import (
    compile_train_operation_request, compile_train_operation_request_record,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import authored_relation_role_identity
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _prepared(tmp_path):
    """使用真实许可原型和typed端点，构造不携带常量的单锚点程序。"""
    path, _original = _course(tmp_path)
    facts = read_sqlite_operation_fact_source(path, SOURCE_ROOT)
    adapter, seeds = verify_operation_fact_source(facts)
    schema, = adapter.schemas
    roles = tuple(sorted((slot.role for slot in schema.slots), key=lambda role: role.stable_key()))
    variables = tuple((role, index) for index, role in enumerate(roles, 1))
    program = IntegerOperationProgram((92431, 1), (1,), (OperationAtom(schema, variables),), (),
        (OperationProjection(1, roles),))
    row = dict(program_key=list(program.stable_key()), owner='teacher', split='train', license_id='CC0-1.0')
    return path, facts, seeds[0], program, row


def test_typed_source_reconstructs_complete_course_without_exchange_files(tmp_path):
    """完整source身份、所有W-06成员和角色均从原库恢复，来源目录不新增文件。"""
    path, facts, _seed, _program, _row = _prepared(tmp_path)
    before = tuple(sorted(path.parent.iterdir()))
    adapter, seeds = verify_operation_fact_source(facts)
    assert facts.course.source_identity == (SQLITE_OPERATION_SOURCE, *facts.source_sha256)
    assert len(adapter.candidates) == len(seeds) == len(facts.course.records) == 1
    assert facts == read_sqlite_operation_fact_source(path, SOURCE_ROOT)
    assert tuple(sorted(path.parent.iterdir())) == before


def test_typed_source_compiles_identical_program_and_request_witnesses(tmp_path):
    """两个编译入口完全相等，请求只学习间隔和reader，不保存示例端点。"""
    _path, facts, seed, program, row = _prepared(tmp_path)
    program_path = tmp_path / 'program.exchange'
    program_path.write_bytes(canonical_json_line(row))
    witness = compile_train_operation_program_record(row, facts)
    assert witness == compile_train_operation_program(program_path, facts)
    assert witness.program == program
    role = program.atoms[0].role_variables[0][0]
    binding, = (binding for binding in seed.bindings
                if authored_relation_role_identity(binding.role_kind) == role)
    endpoint, = (endpoint for endpoint in seed.endpoints if endpoint.endpoint_id == binding.endpoint_id)
    request = dict(question='请介绍' + endpoint.surface_fragment + '的关系。',
        readers=[dict(variable=1, seed_id=seed.seed_id, role_key=list(role.stable_key()))],
        owner='teacher', split='train', license_id='CC0-1.0')
    request_path = tmp_path / 'request.exchange'
    request_path.write_bytes(canonical_json_line(request))
    frame = compile_train_operation_request_record(request, facts, witness)
    assert frame == compile_train_operation_request(request_path, facts, witness)
    assert frame.gaps == (tuple(map(ord, '请介绍')), tuple(map(ord, '的关系。')))
    assert frame.source_identity == facts.course.source_identity


@pytest.mark.parametrize('mutation', ('course', 'license', 'prototype', 'owner', 'endpoint'))
def test_typed_operation_source_rejects_forged_or_changed_members(tmp_path, mutation):
    """内存替换和物理来源变动均拒绝，不能凭旧摘要继续训练。"""
    path, facts, _seed, _program, row = _prepared(tmp_path)
    if mutation == 'course':
        facts = replace(facts, course=replace(facts.course, source_identity=(SQLITE_OPERATION_SOURCE, 1)))
    elif mutation == 'license':
        facts = replace(facts, course=replace(facts.course, license_identity=(1,)))
    else:
        with sqlite3.connect(path) as db:
            if mutation == 'prototype':
                db.execute('UPDATE prototype SET sha256=?', (bytes(32),))
            elif mutation == 'owner':
                db.execute("UPDATE course_meta SET value='evaluator' WHERE name='label_owner'")
            else:
                db.execute('DELETE FROM teacher_endpoint WHERE endpoint_ordinal=1')
    with pytest.raises(ValueError):
        compile_train_operation_program_record(row, facts)


def test_typed_operation_request_rejects_wrong_fact_or_program_source(tmp_path):
    """reader与程序必须同完整教师来源，不允许借别的课程或未知seed授权。"""
    _path, facts, _seed, _program, row = _prepared(tmp_path)
    witness = compile_train_operation_program_record(row, facts)
    request = dict(question='请介绍新部件的关系。', readers=[dict(variable=1, seed_id='absent',
        role_key=list(witness.program.atoms[0].role_variables[0][0].stable_key()))],
        owner='teacher', split='train', license_id='CC0-1.0')
    with pytest.raises(ValueError, match='no independent train fact'):
        compile_train_operation_request_record(request, facts, witness)
    with pytest.raises(ValueError, match='licensed train reader'):
        compile_train_operation_request_record(request, facts,
            replace(witness, source_identity=(SQLITE_OPERATION_SOURCE, 1)))
