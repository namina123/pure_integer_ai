"""把全部受支持历史 occurrence 绑定到同一精确目标，保留 H-00/H-04 全部成员。"""
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED, EPISTEMIC_SUPPORTED, EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT, EVIDENCE_UNKNOWN, LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


REFERENCE_TARGET_SUPPORT_VERSION = 91655


def _key(value):
    """要求完整、非空且可跨语言恢复的规范整数键。"""
    if not value:
        raise ValueError("目标支持成员缺少完整身份")
    pack_record(REFERENCE_TARGET_SUPPORT_VERSION, value)


def _keys(values):
    """成员列表须非空、有序且无重复，不以数量或哈希替代成员。"""
    if type(values) is not tuple or not values or values != tuple(sorted(set(values))):
        raise ValueError("目标支持证据必须是完整规范集合")
    for value in values:
        _key(value)


@dataclass(frozen=True, slots=True)
class ReferenceTargetSupportMember:
    """一个历史候选及其全部图证据、H-00 证据与恢复上下文。"""

    candidate_key: tuple[int, ...]
    antecedent_key: tuple[int, ...]
    hypothesis_key: tuple[int, ...]
    context_key: tuple[int, ...]
    graph_evidence_keys: tuple[tuple[int, ...], ...]
    evidence_record_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        """拒绝缺身份、缺完整证据或非规范成员。"""
        for value in (self.candidate_key, self.antecedent_key,
                      self.hypothesis_key, self.context_key):
            _key(value)
        _keys(self.graph_evidence_keys)
        _keys(self.evidence_record_keys)

    def stable_key(self):
        """按固定字段序和长度边界保存全部整数成员。"""
        return pack_record(REFERENCE_TARGET_SUPPORT_VERSION, (1,),
            self.candidate_key, self.antecedent_key, self.hypothesis_key, self.context_key,
            pack_record(1, *self.graph_evidence_keys), pack_record(1, *self.evidence_record_keys))

    @classmethod
    def from_stable_key(cls, key):
        """无损恢复成员，不接受未知字段或截断记录。"""
        fields = unpack_record(key, REFERENCE_TARGET_SUPPORT_VERSION)
        if len(fields) != 7 or fields[0] != (1,):
            raise ValueError("目标支持成员协议不匹配")
        return cls(*fields[1:5], unpack_record(fields[5], 1), unpack_record(fields[6], 1))


@dataclass(frozen=True, slots=True)
class ReferenceTargetSupport:
    """多个历史候选共同支持一个目标；不指定单 occurrence winner。"""

    candidate_set_key: tuple[int, ...]
    target_key: tuple[int, ...]
    members: tuple[ReferenceTargetSupportMember, ...]
    decision_key: tuple[int, ...]

    def __post_init__(self):
        """要求至少两个独立完整候选，成员身份不得相互覆盖。"""
        for value in (self.candidate_set_key, self.target_key, self.decision_key):
            _key(value)
        if (type(self.members) is not tuple or len(self.members) < 2
                or any(not isinstance(x, ReferenceTargetSupportMember) for x in self.members)
                or self.members != tuple(sorted(set(self.members), key=lambda x: x.stable_key()))
                or any(len({getattr(x, name) for x in self.members}) != len(self.members)
                       for name in ('candidate_key', 'antecedent_key', 'hypothesis_key'))):
            raise ValueError("目标支持必须保留全部独立规范候选")

    def stable_key(self):
        """保存完整候选集、精确目标、成员与 H-04 决策，禁止派生摘要充当本体。"""
        return pack_record(REFERENCE_TARGET_SUPPORT_VERSION, (2,), self.candidate_set_key,
            self.target_key, self.decision_key, *(x.stable_key() for x in self.members))

    @classmethod
    def from_stable_key(cls, key):
        """从规范整数记录恢复全部成员并重新核验结构约束。"""
        fields = unpack_record(key, REFERENCE_TARGET_SUPPORT_VERSION)
        if len(fields) < 6 or fields[0] != (2,):
            raise ValueError("目标支持协议不匹配")
        return cls(fields[1], fields[2], tuple(
            ReferenceTargetSupportMember.from_stable_key(x) for x in fields[4:]), fields[3])


def reference_target_support(candidate_set, candidates, evidence, evidence_records, decision):
    """仅在 H-04 保留全部一致 supported 候选、无冲突且逐项证据完整时形成记录。

    图 evidence 必须已由调用者的当前 Core 读取门筛选；本函数不读取模型、不选择
    来源或最近历史，不把多个不同目标改写成一个目标。
    """
    traces = {x.hypothesis: x.after for x in decision.candidates}
    if (len(traces) != len(candidates) or set(traces) != {x.hypothesis for x in candidates}
            or any(x.epistemic_status == EPISTEMIC_CONFLICTED for x in traces.values())):
        return None
    supported = tuple(x for x in candidates
                      if traces[x.hypothesis].lifecycle == LIFECYCLE_ACTIVE
                      and traces[x.hypothesis].epistemic_status == EPISTEMIC_SUPPORTED)
    if (len(supported) < 2
            or set(decision.adopted_hypotheses) != {x.hypothesis for x in supported}
            or len(decision.adopted_hypotheses) != len(supported)):
        return None
    target_keys, members = set(), []
    for candidate in supported:
        graphs = tuple(x for x in evidence if x.antecedent_key == candidate.occurrence.stable_key())
        records = tuple(x for x in evidence_records if x.hypothesis == candidate.hypothesis)
        decisive = {x.target.stable_key() for x in graphs
                    if x.decisive and x.stance == EVIDENCE_SUPPORT}
        if len(decisive) != 1 or not decisive <= {x.stable_key() for x in candidate.semantic_objects}:
            return None
        snapshot = traces[candidate.hypothesis]
        for stance, ids in ((EVIDENCE_SUPPORT, snapshot.support_evidence_ids),
                            (EVIDENCE_REFUTE, snapshot.refute_evidence_ids),
                            (EVIDENCE_UNKNOWN, snapshot.unknown_evidence_ids)):
            actual = tuple(sorted(x.evidence_id for x in records if x.stance == stance))
            if actual != tuple(sorted(ids)):
                return None
        if any(not any(r.payload == x.stable_key() and r.stance == x.stance
                       and r.source == x.source and r.timestamp_seq == decision.timestamp_seq
                       and r.reason_key == (x.stable_key()[0], x.dimension, x.decisive)
                       for r in records) for x in graphs):
            return None
        target_keys.update(decisive)
        members.append(ReferenceTargetSupportMember(candidate.stable_key(),
            candidate.occurrence.stable_key(), candidate.hypothesis.stable_key(),
            candidate.context_key, tuple(sorted(x.stable_key() for x in graphs)),
            tuple(sorted(x.stable_key() for x in records))))
    if len(target_keys) != 1:
        return None
    return ReferenceTargetSupport(candidate_set.stable_key(), next(iter(target_keys)),
        tuple(sorted(members, key=lambda x: x.stable_key())), decision.stable_key())
