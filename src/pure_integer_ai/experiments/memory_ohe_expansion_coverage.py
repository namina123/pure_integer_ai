"""由当前完整观察根和来源内 O/H/E 成员边派生内容展开覆盖集合。"""
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_EVIDENCE, MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_OBSERVATION,
)
from pure_integer_ai.cognition.shared.query_state import SPACE_MEMORY


def memory_ohe_expansion_coverage(roots, observations, hypotheses, evidence):
    """只覆盖已有精确根可达的完整声明，不改变根、关系、立场或访问状态。

    返回值仅用于避免上下文链接重复调度；O→H→E 原展开仍逐项消费对象。
    同来源和 scope 以完整整数记录相等判定，缺成员或跨来源对象不覆盖。
    """
    rooted = {(root.root_key, root.source_ref, root.scope_key)
              for root in roots if root.owner_space == SPACE_MEMORY}
    covered_observations = {
        item.observation_key: item for item in observations
        if (item.observation_key, item.source_ref, item.scope_key) in rooted}
    covered_hypotheses = {}
    for item in hypotheses:
        parent = covered_observations.get(item.observation_key)
        if (parent is not None and item.hypothesis_key in parent.hypothesis_keys
                and item.source_ref == parent.source_ref
                and item.scope_key == parent.scope_key):
            covered_hypotheses[item.hypothesis_key] = item
    covered = {
        *((MEMORY_OBJECT_OBSERVATION, key) for key in covered_observations),
        *((MEMORY_OBJECT_HYPOTHESIS, key) for key in covered_hypotheses),
    }
    for item in evidence:
        parent = covered_hypotheses.get(item.hypothesis_key)
        if (parent is not None and item.evidence_key in parent.evidence_keys
                and item.source_ref == parent.source_ref
                and item.scope_key == parent.scope_key):
            covered.add((MEMORY_OBJECT_EVIDENCE, item.evidence_key))
    return frozenset(covered)
