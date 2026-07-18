"""storage.selection_pref_count — 选择倾向共现统计台账（刀5 件5 地基·§十 边约束·避免"我吃猫"/"石头追老鼠"）。

selection_pref_count = (concept_a, argument_class) 类聚合共现计数·选择倾向（selectional preference）
  的记忆侧地基。镜像 experience_count 范式（MUTABLE_MONOTONE 扩展表·core=False·独立表非 concept_node 加列）。

**设计源头**（doc/概念空间改造方案.md §十·用户纠正 2026-06-26）：抽象骨架只知道结构·不知道谁能搭配谁·
  无边约束 → "我吃猫""石头追老鼠"（语法对·语义荒谬）。件5 = 搭配统计边约束·约束生成（非硬拒·软偏置）。

**key 设计**（刀5 第3 fork·用户认·写时不标 predicate·predicate 解释 defer S4 read-time）：
  key = (concept_a ConceptRef, argument_class ConceptRef)。argument_class = concept_b 的 IS_A 最近祖先
  （无祖先→b 自身·冷启动退化恒等）。"猫追老鼠"+"狗追老鼠"→argument_class LCA(猫,狗)=动物 → 同行聚合。
  **写时不标 predicate**（何位=action 是 per-concept 涌现·写时识别循环·§9 难点A）·predicate 解释
  defer S4 PR docking 消费侧（按 emergent_role action 桶判 concept_a 是否 predicate）。

**两源同表**（镜像 experience_count）：
  base_count  通识先验频次（append-only·录放层教师注入·reward 不调·首版=0 占位 defer）
  sp_sn       经验成功数（MUTABLE_MONOTONE·reward>0 episode feed·S4 片3 落·**消费侧 = 生成精查**
              graph_view.selection_pref_score 读 sp_sn + sp_tn·成功搭配加成 boost·镜像 effective_freq
              的 e_sn→promote 消费路径在 selection_pref 的对偶是"生成精查"·PR 粗筛侧 sp_agg 不含 sp_sn）
  sp_tn       共现总数（**两路写**：observe record_selection_pref_cooccur 段内共现 +1 +
              reward record_selection_pref_reward episode 配对 +1·S4 片3 后两路·sp_sn ≤ sp_tn 子集守）

**reward CAUSES-only 防塌柱①**（铁律·刀5 首版守）：selection_pref_count 是统计台账非 edge reward·
  sp_sn reward feed S4 片3 已落（reward_propagate 落点⑥ pair-key feed·独立表不进 edge reward 多头）·
  不进 causes_edges/distributed/record_episode_result·reward_propagate.py:148-150 assert 不动·
  effective_weight:82 assert D:13 不内（不建 D:13 边·守 role_scheme defer）。

**§8.5 边 schema 不预留乘子字段**（铁律）：selection_pref_count 是独立表非边·不挂乘子·
  PR 软加权 dock seed 向量 **S4 落地**（_seed_weight 乘积 dock w_freq·w_sp·sp_agg = sum_base + sum_sp_tn）。
  **纠偏（S4 Plan agent·doc L264）**：seed=struct_ref 是**数据真空非机制阻塞**（struct_ref 段标签非 token·
  _inject_base_freq 不注·eff_freq=0·_seed_weight 退 ONE 是设计正确）·freq 维真生效走 attractor 扩张路径
  （ATTRACTOR_MODE ON·token concept seed 有 eff_freq）·selection_pref 维同（attractor 扩张 token seed 有 sp_agg）·
  乘子 dock 走 _seed_weight 权重缩放（线性性）不动 seed 节点集·不破遍历序·不破 bit-identical。

铁律：纯整数（base_count/sp_sn/sp_tn 全 TYPE_INT·assert_int 守）/ MUTABLE_MONOTONE（表纪律·delta +1 无负）/
  append-only 行级（insert 一次 + sp_tn update·同 edge 表范式·不动 concept_node 不变量）/ 确定性（bit-identical）/
  单向依赖（L0 storage·L4 selection_pref 写·L8 formal_train register·皆向下）/ 不写死（schema 元定义列·
  计数器非语义规则）。
诚实边界：本表是地基非楼（解选择倾向 stats substrate·PR 软加权已落 S4 片2/sp_sn feed+消费已落 S4 片3+
  后续加固/predicate 写时识别/D:13 边仍 defer·反 theater 用 sp_tn count 区分非 PR 偏置·
  stable≠correct "吃猫"数据见过就高 count 接地墙外）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

SELECTION_PREF_COUNT_TABLE = "selection_pref_count"


_SELECTION_PREF_COUNT_COLUMNS = [
    ("space_id_from", TYPE_INT),
    ("local_id_from", TYPE_INT),
    ("space_id_to", TYPE_INT),     # argument_class ConceptRef space（IS_A LCA·非 token 本身）
    ("local_id_to", TYPE_INT),     # argument_class ConceptRef local
    ("base_count", TYPE_INT),      # 通识先验频次（append-only·录放层注入·reward 不调·首版=0 占位 defer）
    ("sp_sn", TYPE_INT),           # 经验成功数（reward>0 episode feed·S4 片3 落·消费侧=生成精查 selection_pref_score）
    ("sp_tn", TYPE_INT),           # 共现总数（observe 段内 + reward episode 两路写·S4 片3 后两路·sp_sn ≤ sp_tn 子集守·reward 路 β_arith 染色）
    ("sp_observe_tn", TYPE_INT),   # observe 路纯计数（sign-agnostic·方案3 tn路 B5 β_arith 修法·record_selection_pref_cooccur 写·reward 不调·consumer SP_OBSERVE_MODE 守读替 sp_tn）
]
_SELECTION_PREF_COUNT_INDEXES = [
    ("space_id_from", "local_id_from", "space_id_to", "local_id_to"),   # 主查询 key
    # perf round3（2026-07-13）：read_selection_pref_agg 查 {space_id_from, local_id_from}（2 键·
    # selection_pref_score/attractor _seed_weight 聚合路径）·4 列索引因 space_id_to/local_id_to ∉ where
    # 不满足 _covering_candidates 全覆盖判据 → 全表扫（profile n=4 实测 4284us/次 vs 4 键 4.6us = 931x）。
    # 加 2 列前缀索引覆盖 agg 读·bit-identical（桶保插入序·agg 按 sum 聚合 + (space_id_to,local_id_to)
    # 重排序无关·MUTABLE_MONOTONE 更新只改 sp_tn 非索引列 dirty 空·桶不 rebuild）。同 def_array 修法。
    ("space_id_from", "local_id_from"),
]


def register_selection_pref_count(backend: StorageBackend) -> None:
    """注册 selection_pref_count 扩展表（core=False·MUTABLE_MONOTONE·启动/用前调·幂等）。"""
    register_extension_table(backend, SELECTION_PREF_COUNT_TABLE,
                             _SELECTION_PREF_COUNT_COLUMNS,
                             disc.DISC_MUTABLE_MONOTONE, _SELECTION_PREF_COUNT_INDEXES)


def read_selection_pref_count(backend: StorageBackend,
                              ref_a: tuple[int, int], ref_class: tuple[int, int],
                              *, observe_mode: bool = False
                              ) -> tuple[int, int, int] | None:
    """读 (concept_a, argument_class) 选择倾向共现 → (base_count, sp_sn, tn) | None。

    tn 列由 observe_mode 选（方案3 tn路·B5 β_arith 修法·gate SP_OBSERVE_MODE 守·caller 传参）：
      - observe_mode=False（gate OFF·既有 bit-identical）→ sp_tn（observe 段内 + reward episode 两路·reward 路 β_arith 染色）。
      - observe_mode=True（gate ON）→ sp_observe_tn（observe 路纯计数·sign-agnostic·独立 episode reward 符号·跨决策分化）。
    返 3-tuple 结构不变（第 3 元语义随 observe_mode 切换）·base_count/sp_sn 行为不变。
    无行=冷启动（未 observe 共现·caller 判 None→count 0）。表未注册→None
    （环境未启 selection_pref_count 台账·向后兼容·同 read_experience_count 范式）。
    """
    sid_a, lid_a = ref_a
    sid_c, lid_c = ref_class
    assert_int(sid_a, lid_a, sid_c, lid_c, _where="read_selection_pref_count.args")
    try:
        rows = backend.select(SELECTION_PREF_COUNT_TABLE, where={
            "space_id_from": sid_a, "local_id_from": lid_a,
            "space_id_to": sid_c, "local_id_to": lid_c,
        }, limit=1)
    except KeyError:
        return None   # 表未注册（caller 未 register_selection_pref_count）·向后兼容
    if not rows:
        return None   # 冷启动（该 (concept_a, arg_class) 对未落过共现）·caller 判 None
    r = rows[0]
    tn_col = r["sp_observe_tn"] if observe_mode else r["sp_tn"]
    return (r["base_count"], r["sp_sn"], tn_col)


def record_selection_pref_cooccur(backend: StorageBackend, *,
                                  ref_a: tuple[int, int],
                                  ref_class: tuple[int, int]) -> None:
    """记一次段内共现（observe 调·sp_tn++ & sp_observe_tn++·镜像 experience_count R1 episode 符号·reward 不调此函数）。

    首次：insert(base_count=0, sp_sn=0, sp_tn=1, sp_observe_tn=1)。
    已存在：sp_tn += 1 & sp_observe_tn += 1（MUTABLE_MONOTONE·delta 固定 +1·无负·表纪律双保险）。
    sp_observe_tn 是 observe 路纯计数（sign-agnostic·方案3 tn路 B5 β_arith 修法·consumer SP_OBSERVE_MODE 守读替 sp_tn·
      reward 路不写 sp_observe_tn·独立 episode reward 符号·跨决策分化避 β_arith）。
    base_count append-only 永不调（observe 路径不碰 base_count·sp_sn 由 record_selection_pref_reward 写·S4 片3）。
    表未注册（bare fixture/未注册场景）→ KeyError 静默 skip（向后兼容·镜像 record_base_freq 范式）。
    """
    sid_a, lid_a = ref_a
    sid_c, lid_c = ref_class
    assert_int(sid_a, lid_a, sid_c, lid_c, _where="record_selection_pref_cooccur.args")
    try:
        existing = backend.select(SELECTION_PREF_COUNT_TABLE, where={
            "space_id_from": sid_a, "local_id_from": lid_a,
            "space_id_to": sid_c, "local_id_to": lid_c,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if not existing:
        backend.insert(SELECTION_PREF_COUNT_TABLE, {
            "space_id_from": sid_a, "local_id_from": lid_a,
            "space_id_to": sid_c, "local_id_to": lid_c,
            "base_count": 0, "sp_sn": 0, "sp_tn": 1, "sp_observe_tn": 1,
        })
        return
    backend.update(SELECTION_PREF_COUNT_TABLE, where={
        "space_id_from": sid_a, "local_id_from": lid_a,
        "space_id_to": sid_c, "local_id_to": lid_c,
    }, set_={"sp_tn": ("+=", 1), "sp_observe_tn": ("+=", 1)})


def read_selection_pref_agg(backend: StorageBackend,
                             ref_a: tuple[int, int], *,
                             observe_mode: bool = False) -> tuple[int, int, int]:
    """读 concept_a 的所有 argument_class 行聚合 → (sum_base, sum_sp_sn, sum_tn)（S4 片1·_seed_weight 乘积 dock 用）。

    sum_tn 列由 observe_mode 选（方案3 tn路·B5 β_arith 修法·gate SP_OBSERVE_MODE 守·caller 传参）：
      - observe_mode=False（gate OFF·既有 bit-identical）→ sum_sp_tn（observe+reward 两路·reward 路 β_arith 染色）。
      - observe_mode=True（gate ON）→ sum_sp_observe_tn（observe 路纯计数·sign-agnostic·独立 episode reward 符号）。
    返 3-tuple 结构不变（第 3 元语义随 observe_mode 切换）·sum_base/sum_sp_sn 行为不变。
    sp_agg（caller 算）= sum_base + sum_tn（总搭配次数·镜像 effective_freq 两源范式 base_freq + e_tn·
      **不含 sp_sn 单独**·sp_sn 是 sp_tn 的成功子集·镜像 eff_freq 不含 e_sn 单独）。

    聚合 concept_a 的所有 (concept_a, *) 行（不同 argument_class·"追"搭配动物/石头/... 各一行）·sum 各列。
    冷启动（无行/表未注册）→ (0, 0, 0)（caller 判 sp_agg=0 → w_sp=ONE·退化 bit-identical）。
    确定性：按 (space_id_to, local_id_to) 升序遍历聚合（NodeRef 升序 tiebreak·bit-identical·
      同 read_sense_candidates:108 范式）。

    铁律：纯整数（sum 三列全 int）/ 确定性（sorted NodeRef 升序）/ 单向依赖（L0 storage·不 import cognition）。
    诚实边界：聚合丢 argument_class 区分（PR 侧粗筛·生成侧精查 read_selection_pref_count pair rate·两层正交·
      S4 决断 2 设计故意·非缺陷）·cold-start IS_A 稀疏 class_of 退 token 时 sum 弱（随刀0 IS_A 积累改善）。
    """
    sid_a, lid_a = ref_a
    assert_int(sid_a, lid_a, _where="read_selection_pref_agg.args")
    try:
        rows = backend.select(SELECTION_PREF_COUNT_TABLE, where={
            "space_id_from": sid_a, "local_id_from": lid_a,
        })
    except KeyError:
        return (0, 0, 0)   # 表未注册（caller 未 register_selection_pref_count）·向后兼容
    if not rows:
        return (0, 0, 0)   # 冷启动（该 concept_a 无任何搭配行）→ sp_agg=0 → w_sp=ONE 退化
    rows_sorted = sorted(rows, key=lambda r: (r["space_id_to"], r["local_id_to"]))
    sum_base = sum(r["base_count"] for r in rows_sorted)
    sum_sn = sum(r["sp_sn"] for r in rows_sorted)
    sum_tn = sum((r["sp_observe_tn"] if observe_mode else r["sp_tn"]) for r in rows_sorted)
    return (sum_base, sum_sn, sum_tn)


def record_selection_pref_reward(backend: StorageBackend, *,
                                  ref_a: tuple[int, int],
                                  ref_class: tuple[int, int],
                                  reward: int) -> None:
    """记一次 reward episode 的 (concept_a, argument_class) 搭配（reward_propagate 第三条腿调·S4 片3）。

    R1 符号（镜像 record_experience_outcome / record_selection_pref_cooccur 范式·reward episode 级）：
      reward>0  → sp_sn += 1 & sp_tn += 1（参与成功 episode·搭配被奖励）
      reward≤0  → sp_tn += 1（judge veto reward==0 / 死路 reward<0·率自然降·不 decrement sp_sn 守 MUTABLE_MONOTONE）
    首次（无行）：insert(base_count=0, sp_sn=1 if reward>0 else 0, sp_tn=1, sp_observe_tn=0)（reward 路不写 sp_observe_tn·显式 0 避 NULL corruption·observe 路独立写）。
    base_count append-only 永不调（reward 路径不碰 base_count·observe record_selection_pref_cooccur 写 sp_tn only）。

    表未注册（bare fixture/未注册场景）→ KeyError 静默 skip（向后兼容·镜像 record_selection_pref_cooccur）。

    铁律：reward CAUSES-only 防塌柱①（本函数是独立表写·不进 causes_edges/distributed/record_episode_result·
      reward_propagate.py:131 assert 不触发·effective_weight:82 assert 不内）。MUTABLE_MONOTONE（delta +1 无负）。
    诚实边界：reward 侧配对是 episode 级粗聚合（concept_targets·非 observe 段内精确）·设计故意·
      stable≠correct（"吃猫"数据见过 reward>0 就高 sp_sn 是接地墙外）。
    """
    sid_a, lid_a = ref_a
    sid_c, lid_c = ref_class
    assert_int(sid_a, lid_a, sid_c, lid_c, reward,
               _where="record_selection_pref_reward.args")
    try:
        existing = backend.select(SELECTION_PREF_COUNT_TABLE, where={
            "space_id_from": sid_a, "local_id_from": lid_a,
            "space_id_to": sid_c, "local_id_to": lid_c,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if not existing:
        backend.insert(SELECTION_PREF_COUNT_TABLE, {
            "space_id_from": sid_a, "local_id_from": lid_a,
            "space_id_to": sid_c, "local_id_to": lid_c,
            "base_count": 0,
            "sp_sn": 1 if reward > 0 else 0,
            "sp_tn": 1,
            "sp_observe_tn": 0,   # reward 路不写 sp_observe_tn·显式 0 避 SQLite 缺列 NULL+1=NULL corruption（方案3 tn路 B5·observe 路独立写）
        })
        return
    set_: dict = {"sp_tn": ("+=", 1)}
    if reward > 0:
        set_["sp_sn"] = ("+=", 1)
    backend.update(SELECTION_PREF_COUNT_TABLE, where={
        "space_id_from": sid_a, "local_id_from": lid_a,
        "space_id_to": sid_c, "local_id_to": lid_c,
    }, set_=set_)
