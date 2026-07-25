"""teacher.probe_set — D4 留出探针集隔离 + 版本化（§十一 #4-bis line711·#358 完整实现）。

本模块只提供旧签名集合的精确内容不相交诊断和版本化。该判据不能识别同源改写，不能
单独作为断奶 D4；正式隔离由 experiments.evaluation_protocol 的完整内容、dedup cluster、
provenance cluster 和 EXTERNAL 分账共同判定。

与 D3 互补：D3 防判官偏袒 / D4 防数据泄漏·两道同时过才允许断奶评估。

铁律：纯整数（ConceptRef 元组·集合运算）/ 确定性 bit-identical（版本化·frozenset 确定性）/
  不写死（探针采样质量是 oracle 责任非本模块）/ 几百G不重训（新 run_id 新探针版本）。
诚实边界：本模块只防精确签名重复，不证明来源独立或探针代表性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


@dataclass(frozen=True)
class ProbeSet:
    """留出探针集（纯整·版本化·bit-identical 可复现）。

    version      版本号（每轮同版本·新 run_id 新版本·守几百G不重训红线）
    probe_refs   探针概念 ref 集（frozenset·确定性·纯整元组）
    """

    version: int
    probe_refs: frozenset[ConceptRef]

    def __post_init__(self) -> None:
        assert_int(self.version, _where="ProbeSet.version")
        if not isinstance(self.probe_refs, frozenset):
            object.__setattr__(self, "probe_refs", frozenset(self.probe_refs))


def is_disjoint(probe_set: ProbeSet, training_refs: Iterable[ConceptRef]) -> bool:
    """判断旧合成 ref 是否精确不相交，不据此宣称 V-00 来源隔离。

    非空交集一定是泄漏；空交集仍可能包含同源改写或哈希身份不足。
    """
    return probe_set.probe_refs.isdisjoint(set(training_refs))


def make_probe_set(version: int, refs: Iterable[ConceptRef]) -> ProbeSet:
    """构造探针集（确定性·frozenset·版本化）。caller 保证 refs 是训练期从未喂过/从未被教师评过的输入。"""
    return ProbeSet(version=version, probe_refs=frozenset(refs))


def ref_from_signature(signature: str, space_id: int = 0) -> ConceptRef:
    """W4·CollectedItem 签名 → 确定性 ConceptRef（D4 探针隔离判定用·非 observe struct_ref）。

    纯整 hash·确定性 bit-identical。space_id 默认 0（探针 ref 合成空间·非图 struct_ref·异 observe seed
    "observe.prog.v1"）。同签名→同 ref（泄漏检测正确）·异签名→异 ref（不相交正确）。

    为何不用 observe struct_ref：① 逻辑悖论——拿探针 observe struct_ref 须先 observe 探针=训练=泄漏·违 D4
    ② observe struct_ref 依赖 stage（`__prog_{stage}_`）·不稳定 ③ D4 是 item 级判据·图 struct_ref 语义错配。
    诚实边界：签名 hash 判 item 不相交·非图 struct_ref 不相交（同 item 必同 ref=泄漏检测正确·异 item 未必真无关·D 墙）。
    """
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    assert_int(space_id, _where="ref_from_signature.space_id")
    return (space_id, Hasher("probe.ref").h63(signature))
