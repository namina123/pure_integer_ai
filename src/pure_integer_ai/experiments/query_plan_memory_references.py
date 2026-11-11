"""从原查询成员恢复 Plan 引用的允许集合，并在既有声明预算内完整核验。"""
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import ROOT_EXPANDED, SPACE_MEMORY


PLAN_MEMORY_REFERENCE_PAGE_SIZE = 128


def allowed_plan_memory_references(input_ref, evidence, roots, input_roles, contexts):
    """只从当前完整状态和已冷审上下文恢复引用，不读取待审 Plan 或历史正文。"""
    result = {input_ref}
    candidates = [root.root_key for root in roots
                  if root.owner_space == SPACE_MEMORY and root.status == ROOT_EXPANDED]
    for entry in evidence:
        if entry.space == SPACE_MEMORY:
            candidates.extend((entry.hypothesis_key, entry.evidence_key))
    for key in candidates:
        try:
            ref = MemoryObjectRef.from_stable_key(key)
        except (TypeError, ValueError):
            # Memory 中的一等结构节点仍由原 QueryState 证据和 binding 核验，
            # 它们不是 MemoryObjectRef，不能进入此类型专用的允许集合。
            continue
        result.add(ref)
    for role in input_roles:
        result.update((role.observation, role.hypothesis))
    for context in contexts:
        result.update(context.memory_refs())
    return frozenset(result)


def read_plan_memory_references(value, *, allowed, input_ref, maximum, key_reader, error):
    """逐页核验全部引用；总量受原声明预算约束，拒绝重复、跨域和无归属引用。"""
    if (not isinstance(value, (list, tuple)) or len(value) > maximum):
        raise error("plan Memory is missing or exceeds audit budget")
    refs = set()
    for start in range(0, len(value), PLAN_MEMORY_REFERENCE_PAGE_SIZE):
        for key in value[start:start + PLAN_MEMORY_REFERENCE_PAGE_SIZE]:
            ref = MemoryObjectRef.from_stable_key(key_reader(key, "plan Memory"))
            if (ref.owner != input_ref.owner or ref.versions != input_ref.versions
                    or ref.memory_space != input_ref.memory_space):
                raise error("plan Memory crosses the current owner/version/space")
            if ref in refs:
                raise error("plan Memory reference is duplicated")
            if ref not in allowed:
                raise error("plan Memory lacks a current QueryState or qualified context reference")
            refs.add(ref)
    return refs
