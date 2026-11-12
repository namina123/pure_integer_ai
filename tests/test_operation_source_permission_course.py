"""现役typed来源和模型图上的显式许可学习，课程与运行数据严格分离。"""
from dataclasses import replace
import sqlite3

import pytest

from test_authored_relation_sqlite_course import SOURCE_ROOT, _course
from test_sqlite_operation_fact_source import _prepared
from pure_integer_ai.experiments.operation_fact_source import read_sqlite_operation_fact_source
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program_record
from pure_integer_ai.experiments.operation_source_permission_course import compile_train_operation_source_permission_record


def _sources(tmp_path):
    """原程序与新增typed来源具有相同schema、不同完整课程和来源身份。"""
    _, facts, _, _, instruction = _prepared(tmp_path)
    base = compile_train_operation_program_record(instruction, facts)
    directory = tmp_path / 'new-source'
    directory.mkdir()
    path, _ = _course(directory)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE domain SET name_utf8=?", (b'new-permitted-course',))
        db.execute('UPDATE teacher_endpoint SET local_id=local_id+100')
    added = read_sqlite_operation_fact_source(path, SOURCE_ROOT)
    row = dict(base_witness_key=list(base.stable_key()), course_source_identity=list(added.course.source_identity),
        owner='teacher', split='train', license_id='CC0-1.0')
    return base, added, row


def test_learning_preserves_program_and_all_prior_permissions(tmp_path):
    """新增来源只通过完整课程验证，程序/原来源/语言身份均不改写。"""
    base, facts, row = _sources(tmp_path)
    current = compile_train_operation_source_permission_record(row, facts, base)
    assert current.program == base.program
    assert current.source == base.source and current.source_identity == base.source_identity
    assert current.license_identity == base.license_identity
    assert set(base.authorized_forming_sources) < set(current.authorized_forming_sources)
    assert len(current.source_extensions) == 1
    assert current.source_extensions[0].course_source_identity == facts.course.source_identity
    assert current.source_extensions[0].base_witness_key == base.stable_key()
    assert compile_train_operation_source_permission_record(row, facts, current) == current


@pytest.mark.parametrize('field,value', (
    ('owner', 'evaluator'), ('split', 'held_out'), ('license_id', 'UNKNOWN'),
    ('course_source_identity', [92430, 1]), ('base_witness_key', [92206, 1]),
))
def test_permission_learning_rejects_foreign_or_nonteacher_instruction(tmp_path, field, value):
    """来源身份、根程序、owner、split和许可都不能被内存记录替换。"""
    base, facts, row = _sources(tmp_path)
    with pytest.raises(ValueError):
        compile_train_operation_source_permission_record(dict(row, **{field: value}), facts, base)


def test_changed_physical_course_is_rejected_before_permission_learning(tmp_path):
    """课程冻结后物理端点或metadata改变时，不可借旧source identity授权。"""
    base, facts, row = _sources(tmp_path)
    with sqlite3.connect(facts.source_path) as db:
        db.execute("UPDATE domain SET name_utf8=?", (b'drifted-course',))
    with pytest.raises(ValueError):
        compile_train_operation_source_permission_record(row, facts, base)


@pytest.mark.parametrize('invalid', (True, 1.0))
def test_teacher_integer_keys_reject_boolean_and_float_aliases(tmp_path, invalid):
    """宿主的True==1或1.0==1不能改变规范纯整数指令。"""
    base, facts, row = _sources(tmp_path)
    key = list(row['base_witness_key'])
    key[key.index(1)] = invalid
    with pytest.raises(ValueError):
        compile_train_operation_source_permission_record(dict(row, base_witness_key=key), facts, base)
