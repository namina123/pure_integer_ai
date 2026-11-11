"""从完整原生模板恢复语义reader的语言序，拒绝用身份排序冒充槽序。"""
from __future__ import annotations

from pure_integer_ai.experiments.trained_generation_connector_runtime import _slot_ordinal


def native_operation_role_order(template, witness):
    """逐槽恢复claim/role；每个原reader必须在真实模板里恰好出现一次。"""
    if template.stable_key() != witness.template_key:
        raise ValueError("native role order requires the complete witnessed template")
    readers = {scoped: (claim, role.stable_key())
               for claim, role, scoped, _ordinal in witness.readers}
    if len(readers) != len(witness.readers):
        raise ValueError("native role order has duplicate scoped readers")
    slots = tuple(sorted(template.slots, key=lambda item: _slot_ordinal(item.slot)))
    if len({item.slot for item in slots}) != len(slots):
        raise ValueError("native role order has duplicate template slots")
    bindings = {item.slot: item for item in template.bindings}
    if len(bindings) != len(template.bindings) or set(bindings) != {item.slot for item in slots}:
        raise ValueError("native role order lacks exact slot bindings")
    ordered, used = [], set()
    for slot in slots:
        binding = bindings[slot.slot]
        if binding.role is None:
            continue
        if binding.role not in readers or binding.role in used:
            raise ValueError("native role order has an unknown or repeated reader")
        used.add(binding.role)
        ordered.append(readers[binding.role])
    if used != set(readers):
        raise ValueError("native role order omits an actual reader")
    return tuple(ordered)


def require_native_operation_role_order(outputs, template, witness):
    """完整投影序必须等于模板的真实语言序；集合相等不足以授权。"""
    expected = tuple((claim, role) for claim, (_key, roles) in enumerate(outputs, 1)
                     for role in roles)
    if native_operation_role_order(template, witness) != expected:
        raise ValueError("native composition role order differs from the explicit operation projection")
