"""语言槽序必须由完整模板恢复，不能依赖canonical reader的存储序。"""
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import structure_concept_identity
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.experiments.operation_realization_contract import (
    native_operation_role_order, require_native_operation_role_order,
)


def _grammar():
    """模板使用逆角色序，reader表仍保持身份序，模拟常见的逆向关系表达。"""
    roles = tuple(role_identity((77, i)) for i in (1, 2))
    scoped = tuple(role_identity((78, i)) for i in (1, 2))
    slots = tuple(structure_concept_identity((20916, 3, 1, 1, 10, i)) for i in (1, 2, 3))
    template = SimpleNamespace(stable_key=lambda: (91,),
        slots=tuple(SimpleNamespace(slot=slot) for slot in reversed(slots)),
        bindings=(SimpleNamespace(slot=slots[0], role=scoped[1]),
                  SimpleNamespace(slot=slots[1], role=None),
                  SimpleNamespace(slot=slots[2], role=scoped[0])))
    witness = SimpleNamespace(template_key=(91,),
        readers=tuple((1, role, scope, 0) for role, scope in zip(roles, scoped)))
    return roles, template, witness


def test_native_template_order_is_distinct_from_canonical_reader_order():
    roles, template, witness = _grammar()
    expected = ((1, roles[1].stable_key()), (1, roles[0].stable_key()))
    assert native_operation_role_order(template, witness) == expected
    require_native_operation_role_order((((81,), tuple(role for _, role in expected)),), template, witness)
    with pytest.raises(ValueError, match="explicit operation projection"):
        require_native_operation_role_order((((81,), tuple(role.stable_key() for role in roles)),), template, witness)
    witness.readers = tuple(reversed(witness.readers))
    template.bindings = tuple(reversed(template.bindings))
    assert native_operation_role_order(template, witness) == expected


@pytest.mark.parametrize('change', ('template', 'reader', 'binding', 'repeated', 'slot'))
def test_incomplete_or_competing_grammar_cannot_authorize_a_projection(change):
    _roles, template, witness = _grammar()
    if change == 'template':
        witness.template_key = (92,)
    elif change == 'reader':
        witness.readers = witness.readers[:1]
    elif change == 'binding':
        template.bindings = template.bindings[:-1]
    elif change == 'repeated':
        template.bindings[-1].role = template.bindings[0].role
    else:
        template.slots = (*template.slots, template.slots[0])
    with pytest.raises(ValueError):
        native_operation_role_order(template, witness)
