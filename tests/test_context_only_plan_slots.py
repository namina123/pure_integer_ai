"""允许语言组织所需的原子槽，同时拒绝可见事实对象或伪报类型。"""
from pure_integer_ai.cognition.shared.identity import OBJECT_LANGUAGE_ATOM, ObjectIdentity
from pure_integer_ai.experiments.context_only_plan_slots import language_only_plan_slots


def test_language_atoms_are_allowed_without_admitting_factual_object_slots():
    language = {'allowed_node_kinds': [OBJECT_LANGUAGE_ATOM],
                'filler': list(ObjectIdentity(OBJECT_LANGUAGE_ATOM, (1, 2)).stable_key())}
    assert language_only_plan_slots({'slot_sequence': [language]})
    assert language_only_plan_slots({'slot_sequence': []})
    factual = dict(language, filler=list(ObjectIdentity(4, (1, 2)).stable_key()))
    assert not language_only_plan_slots({'slot_sequence': [language, factual]})
    assert not language_only_plan_slots({'slot_sequence': [dict(language, allowed_node_kinds=[4, OBJECT_LANGUAGE_ATOM])]})
    assert not language_only_plan_slots({})
    assert not language_only_plan_slots({'slot_sequence': [{'allowed_node_kinds': [OBJECT_LANGUAGE_ATOM]}]})
