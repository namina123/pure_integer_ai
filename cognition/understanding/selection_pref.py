"""cognition.understanding.selection_pref — 选择倾向共现统计 builder（刀5 件5 地基·§十 边约束）。

build_selection_pref_count —— 段内 token 类聚合共现计数·落 selection_pref_count 表（镜像
  cooccurs.build_cooccurs 段内 i<j 配对范式·但第二维上卷到 IS_A 类·且写统计表非 SHADOW 边）。

**设计源头**（doc/概念空间改造方案.md §十·"我吃猫"/"石头追老鼠" 边约束）：抽象骨架不知谁能搭配谁·
  搭配统计约束生成（非硬拒·软偏置）。§9 难点A：动词走 emergent_role 位置桶涌现·不注入词性。

**key 设计**（刀5 第3 fork·写时不标 predicate）：每段内 i<j 配对 (a, b)·双向记录：
  - (a, class_of(b))："a 共现 b 的类"（若 a 是 predicate·此即 a 的论元类偏好）
  - (b, class_of(a))："b 共现 a 的类"（双向·支持 a/b 任一作 predicate 的 read-time 解释）
  class_of(x) = x 的 IS_A 最近祖先（ancestor_map[x] 中最深·nearest_isa_ancestor·无祖先→x 自身·冷启动退化恒等）。
  predicate 写时识别 defer S4（emergent_role 循环性·何位=action 是 per-concept 涌现）。

**反 theater**（"石头追老鼠"）：
  - "狐狸追鸡"/"猫追老鼠"/"狗追猫" 多次 → (追, class_of(狐狸/鸡/猫/老鼠)=动物) sp_tn 高
  - "石头追老鼠" 偶现 → (追, class_of(石头)=石头) sp_tn 低（石头与动物无共同 IS_A 祖先）
  - 验收：read_selection_pref_count(追, 动物).sp_tn >> read_selection_pref_count(追, 石头).sp_tn

**gate SELECTION_PREF_MODE**（默认 OFF·守回归·self-gate 入口返 0）：
  OFF = 不写 sp_tn·等同刀4 后现状（selection_pref_count 表全空）。

铁律：纯整数（ConceptRef/sp_tn 全 int·assert_int 守）/ 确定性 bit-identical（段内 i<j 升序·NodeRef
  升序 tiebreak）/ 不写死（IS_A LCA 结构查询非语义·emergent_role 涌现非词性标签）/ §8.1c（统计表非
  关系边·不涉裸共现直落）/ §8.5（不建边·独立表）/ reward CAUSES-only（observe 写 sp_tn only·
  sp_sn reward feed defer S4）/ 单向依赖（L4 understanding→L4 process.abstraction 纯读·is_a.py:34 先例）。
诚实边界：地基非楼（PR 软加权/sp_sn reward feed/D:13 边/predicate 写时识别全 defer·反 theater 用
  sp_tn count 区分非 PR 偏置·stable≠correct "吃猫"数据见过就高 count 接地墙外）。
"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map, nearest_isa_ancestor
from pure_integer_ai.storage.selection_pref_count import record_selection_pref_cooccur
from pure_integer_ai.cognition.understanding.cooccurs import segment_cooccurrence_pairs

# 选择倾向统计段内配对上限（同 COOCCURS·O(n²) 节流·§训练性能）。
DEFAULT_SELECTION_PREF_PAIR_CAP = 256


def _nearest_isa_ancestor(ancestor_map: dict, ref: ConceptRef) -> ConceptRef:
    """ref 的 IS_A 最近祖先（最深·S4 项2·转调 abstraction.nearest_isa_ancestor·写读一致三处同源）。

    保留 wrapper 守 selection_pref 内调用点（:96/:97）不变·消除 min(ancestors) 字面重复。
    多层 IS_A 真 LCA 落地（替原升序首 min·非最深）·三处同源（graph_view + selection_pref + reward_propagate）。
    """
    return nearest_isa_ancestor(ancestor_map, ref)


def build_selection_pref_count(backend, refs: list[ConceptRef], *,
                               space_id: int, lang: int,
                               pair_cap: int = DEFAULT_SELECTION_PREF_PAIR_CAP,
                               ancestor_map: dict | None = None) -> int:
    """段内 token 类聚合共现计数 → selection_pref_count 表 sp_tn++（刀5 件5 地基 builder）。

    段内 i<j 配对（同 cooccurs.segment_cooccurrence_pairs·cap 节流）·每对 (a, b) 双向记录：
      (a, class_of(b)) 与 (b, class_of(a))·各 sp_tn++。两记录第一维（concept_a）不同（a≠b）·
      必各记一次（支持 a/b 任一作 predicate 的 read-time 解释）。
    class_of(x) = _nearest_isa_ancestor（IS_A 最近祖先最深·转调 nearest_isa_ancestor·无则自身·冷启动退化恒等）。

    **lang 桶守门**（C1 防跨语言污染·同 cooccurs 契约）：lang 标识段所属桶·caller 须传单 lang 段
    （observe 段内天然同桶·中文/英文 token 不混段）。lang 不入表 schema（concept_a/argument_class
    ConceptRef 已 lang-scoped·同 surface 不同 lang 不同 ref·天然分桶）·仅 caller 契约守门 + assert。

    gate SELECTION_PREF_MODE（默认 OFF）→ 返 0（不写·守回归 bit-identical）。
    ancestor_map：caller 可透传（observe hoist 增量 map·#1115 perf·免每段重建）或 None（self-built·既有单测）。
    **#1115 perf（observe hoist 增量透传）**：observe 段循环外 hoist ancestor_map（per space·per-pipeline·
    build_is_a_edges 建新 IS_A 时 apply_isa_edge_to_map 增量·环 fall back 全量）·selection_pref 透传复用·
    免每段 build_isa_ancestor_map 全图重建（cProfile 47.6% 热点·67 次重建）。None→self-built 退化既有。
    返记录数（双向·= 配对数 × 2·cap 限内）。
    """
    if not getattr(gates, "SELECTION_PREF_MODE", False):
        return 0   # gate OFF·守回归（selection_pref_count 表全空·等同刀4 后现状）
    assert_int(space_id, lang, _where="build_selection_pref_count.space_id_lang")
    pairs = segment_cooccurrence_pairs(len(refs), cap=pair_cap)
    if not pairs:
        return 0
    # ancestor_map：透传（observe hoist 增量·#1115）or self-built（None·既有单测退化·bit-identical）
    if ancestor_map is None:
        ancestor_map = build_isa_ancestor_map(backend, space_id=space_id)
    n = 0
    for i, j in pairs:
        a = refs[i]
        b = refs[j]
        if a == b:
            continue
        class_of_b = _nearest_isa_ancestor(ancestor_map, b)
        class_of_a = _nearest_isa_ancestor(ancestor_map, a)
        # 双向记录（第一维 concept_a 不同 a≠b·必各记一次·支持任一作 predicate 的 read-time 解释）
        record_selection_pref_cooccur(backend, ref_a=a, ref_class=class_of_b)   # a 共现 b 的类
        record_selection_pref_cooccur(backend, ref_a=b, ref_class=class_of_a)   # b 共现 a 的类
        n += 2
    return n
