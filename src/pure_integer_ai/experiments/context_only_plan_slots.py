"""区分训练语言原子槽与表示事实对象的槽，不按空槽数替代语义资格。"""
from pure_integer_ai.cognition.shared.identity import OBJECT_LANGUAGE_ATOM, ObjectIdentity


def language_only_plan_slots(plan):
    """每个可见槽都必须只允许并实际引用语言原子；实体或类型混入均拒绝。"""
    slots = plan.get('slot_sequence')
    if not isinstance(slots, (tuple, list)):
        return False
    try:
        return all(tuple(slot['allowed_node_kinds']) == (OBJECT_LANGUAGE_ATOM,)
                   and ObjectIdentity.from_stable_key(tuple(slot['filler'])).object_kind == OBJECT_LANGUAGE_ATOM
                   for slot in slots)
    except (KeyError, TypeError, ValueError):
        return False
