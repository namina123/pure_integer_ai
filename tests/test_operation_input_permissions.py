"""新来源请求可与旧语言并存，但同operation的全部输入仍须同权。"""
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.operation_request_consumer import validate_operation_input_permissions


def _witness():
    """权限检查只消费明确operation及完整整数身份。"""
    return SimpleNamespace(program=SimpleNamespace(operation_key=(7, 1)),
                           source_identity=(8, 2), license_identity=(9, 2))


def test_unrelated_input_course_cannot_impose_its_source_on_new_operation():
    """旧语言图未声明此程序时，新request权限不被错误换成旧来源。"""
    validate_operation_input_permissions(_witness(), (
        ((SimpleNamespace(operation_key=(7, 2)),),
         dict(source_identity=(8, 1), license_identity=(9, 1))),))


@pytest.mark.parametrize('field', ('source_identity', 'license_identity'))
def test_same_operation_rejects_each_permission_dimension(field):
    """同operation的竞争来源不得借已通过的request或其它图放行。"""
    witness = _witness()
    metadata = dict(source_identity=witness.source_identity, license_identity=witness.license_identity)
    metadata[field] = (10, 1)
    with pytest.raises(ValueError, match='permission'):
        validate_operation_input_permissions(witness, (
            ((), None), ((SimpleNamespace(operation_key=witness.program.operation_key),), metadata)))


def test_matching_input_operation_keeps_complete_source_permission():
    """完整来源与许可均一致才通过；不使用裸摘要或集合首成员。"""
    witness = _witness()
    validate_operation_input_permissions(witness, (
        ((SimpleNamespace(operation_key=witness.program.operation_key),),
         dict(source_identity=list(witness.source_identity), license_identity=list(witness.license_identity))),))
