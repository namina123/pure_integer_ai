"""有类型来源入口保持原输出合同身份及teacher/train权限。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.integer_operation_output import OperationOutputContract
from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationProjection,
)
from pure_integer_ai.cognition.shared.semantic_object import entity_identity
from pure_integer_ai.experiments.operation_output_course import (
    compile_train_operation_output, compile_train_operation_output_record,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from test_integer_operation_program import _atom, _fact, _schema, _source, _witness, LEFT, RIGHT


def _course():
    """只构造完整整数许可，不引入来源事实或固定语言答案。"""
    schema, source = _schema(1), _source(92470)
    program = IntegerOperationProgram((92470, 1), (1,), (_atom(schema, (1, 2)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),))
    fact = _fact(92470, schema.relation, entity_identity(source, (1,)), entity_identity(source, (2,)))
    witness = _witness(program, (fact,))
    contract = OperationOutputContract(witness.stable_key(), (1,))
    return witness, dict(contract_key=list(contract.stable_key()), owner='teacher',
                         split='train', license_id='CC0-1.0')


def test_typed_course_retains_the_complete_physical_source_identity(tmp_path):
    """无正文typed课程与原规范记录须得到完全相同的可逆witness。"""
    program, row = _course()
    path = tmp_path/'output.train.json'
    path.write_bytes(canonical_json_line(row))
    assert (compile_train_operation_output_record(row, (program,))
            == compile_train_operation_output(path, (program,)))


@pytest.mark.parametrize('change', ({'owner':'evaluator'}, {'split':'held_out'}, {'license_id':''}))
def test_typed_course_cannot_promote_unauthorized_instructions(change):
    """更换载体不能让评测或未许可指令进入训练图。"""
    program, row = _course()
    with pytest.raises(ValueError, match='CC0 train instruction'):
        compile_train_operation_output_record({**row, **change}, (program,))


def test_typed_course_requires_the_exact_current_program():
    """相同operation键仍须逐项比较实际程序，不以键或摘要借权。"""
    program, row = _course()
    changed = replace(program, program=replace(program.program, realization_key=(92471, 1)))
    with pytest.raises(ValueError, match='exact current program permission'):
        compile_train_operation_output_record(row, (changed,))
