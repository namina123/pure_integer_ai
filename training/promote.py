"""training.promote — promote 三重（频次/reward/定义·SHADOW→PRIMARY·§十三防塌C4 + §十二阶段4）。

promote_edge(edge_store, node_store, ref, *, teacher=None) -> bool
  三重证据判据（SHADOW→PRIMARY tier flip·MUTABLE_MONOTONE 只升不降）：
    ① 频次 G1：sn+tn ≥ PROMOTE_FREQ_MIN（经验频次达·G1 频次门）
    ② 边级 reward 信号支撑 G5-B：success_rate = sn/(sn+tn) ≥ PROMOTE_REWARD（reward 信号支撑·非裸共现·
       **G5-B 边级 promote·命名借用**·非 §十三 G5-C 记忆项延迟晋升闸·非 §十四 G5-A 自证机门因子·三 G5 同名不同物）
    ③ 定义 G3：有结构锚（IS_A/PROPERTY/REFERS_TO 任一出边·结构锚≥1）∨ 教师确认（teacher.confirm_*）
  三重全达（①∧②∧③）→ tier flip SHADOW→PRIMARY。G4 自相关逃生（结构正则性·§十三 G4）作 defer 登记。

  promote 三重 vs memory consolidate（EXPERIENCE→CONSOLIDATED）：promote 是 tier 晋升（边/节点
  SHADOW→PRIMARY·进默认建模）·consolidate 是记忆空间内状态晋升（带衰减→无衰减·不跨 space·§十三决断4）。
  两者复用晋升闸 G1-G7 范式（频次/冗余/结构锚/自相关/reward/typed 纯度/stale archive）。

promote_concept(edge_store, node_store, ref) —— 节点 tier = max 其边 tier（§十二⑤·复用 edge_store 查边 + node_store.set_tier）。

铁律：纯整数（频次/sn/tn/率×1000·全整）/ MUTABLE_MONOTONE（tier 只升不降·demotion 抛违例）/ 不写死
  （阈值 oracle 标·初值占位·三重判据非硬编码语义规则）/ 外部只启发（教师确认走录放层·给事实判断非边关系真伪）。
诚实边界：promote 判结构证据非语义真伪（接地墙·SHADOW 持有不删·promote 是信任档非真值判定）/
  阈值 oracle 标经验值（能力达标∧依赖度低二维·§十一#4）/ stable≠correct（tier PRIMARY≠语义正确）。
defer：G4 自相关逃生（结构正则性逃生·§十三 G4·首版占位登记）/ G6 typed 纯度（typed 边纯净性·随闭包落）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.compare import cross_ge
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.edge_types import (
    EDGE_IS_A, EDGE_PROPERTY, EDGE_REFERS_TO, EDGE_RELATION_SIGNAL,
)
from pure_integer_ai.storage.experience_count import read_experience_count
from pure_integer_ai.storage.structure_match_count import read_structure_match_count, read_structure_match_per_rel
from pure_integer_ai.storage.composes_attr import read_composes_attrs, ATTR_RELATION_PRIMITIVE
from pure_integer_ai.storage.spaces.memory_space import STATUS_EXPERIENCE
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.config import gates

# ---- promote 三重阈值（oracle 标占位·§十一#4 能力达标∧依赖度低二维调） ----
PROMOTE_FREQ_MIN = 3          # ① 频次 G1：sn+tn ≥ 3（经验频次达·防单次噪声晋 PRIMARY）
PROMOTE_REWARD_NUM = 1        # ② reward G5：success_rate ≥ 1/2（sn/(sn+tn)·reward 信号支撑）
PROMOTE_REWARD_DEN = 2
# ③ 定义 G3：结构锚 = IS_A/PROPERTY/REFERS_TO 任一出边存在（结构锚≥1·非裸共现）

# ---- 刀4 D:11 涌现关系 promote 双轨阈值（experience 主导 + teacher 加分·用户 2026-07-05 拍板） ----
# D:11 边不接 reward（effective_weight:82 assert·sn/tn 恒 0）→ 既有 _reward_ok 永远 False。
# 双轨绕开：experience_count 概念维对偶（e_sn/e_tn·非 edge reward·绕 CAUSES-only 死锁）+
#   teacher/种子 加分（软 ∨ 项·断奶前稳）。两条都构造性（reward 来自 judge 来自教师 GT·
#   非真独立源验证·#479 defer）。
PROMOTE_EXP_FREQ_MIN = 3      # D:11 experience 闸：e_sn+e_tn ≥ 3（镜像 G1·经验频次达）
PROMOTE_EXP_SUCCESS_NUM = 1   # D:11 experience 闸：e_sn/(e_sn+e_tn) ≥ 1/2（镜像 G5·成功率）

# ---- 对应泛化 v2 D:11 结构反推 promote（审2条件1+2·三路分离 + SHADOW 创建·ORACLE_PROMOTE_MODE ON 时唯一轨）----
# 结构反推 tally 闸：W 落 REALIZES-R-skeleton cue slot 的 distinct forming-sample ≥ K（镜像 PROMOTE_EXP_FREQ_MIN）。
# 非循环（R 来自 REALIZES oracle·非 cue）·学全（W 可新词·非 oracle/frozenset）·详见 doc/重来_对应泛化_结构反推_学全。
PROMOTE_STRUCTURE_MATCH_MIN = 3

# ---- #732 G5-C 记忆项延迟晋升闸阈值（§十三:1108/1120 决断4·oracle 标·独立非镜像 G5-B） ----
# G5-C 数据源 = memory_item SEG_EPISODIC 比率门 sum(sc)*1000//sum(count) ≥ θ_rw（caller 侧 by info_ref 聚合·方案 d）。
# **公式数学**（审1 P0-1 纠偏）：record_use(success=True) → sc+=2/count+=1·故 sum(sc)=2×positive·sum(count)=total·
# sum(sc)/sum(count) = 2×(positive 比率)·**非镜像 G5-B**（G5-B sn/(sn+tn)=positive 比率·结构不同·阈值差 4 倍）。
# 阈值 1/1（cross_ge(sum_sc, sum_count, 1, 1)·即 sum_sc ≥ sum_count·即 2×positive ≥ total·即 ≥50% positive 才 consolidate）。
PROMOTE_MEM_FREQ_MIN = 3      # G5-C 频次：sum(count) ≥ 3（同 info_ref ≥3 episode·防单次噪声·镜像 G1）
PROMOTE_MEM_REWARD_NUM = 1    # G5-C 比率门：sum(sc)/sum(count) ≥ 1/1（即 ≥50% positive·sc=2×positive 缩放·独立 oracle 标）
PROMOTE_MEM_REWARD_DEN = 1

_SCALE = 1000


def _edge_row(edge_store: EdgeStore, ref: tuple) -> dict[str, Any] | None:
    """按 EdgeRef 元组 (from_sid,from_lid,to_sid,to_lid,et) 取边行。"""
    fsid, flid, tsid, tlid, et = ref
    return edge_store.get(space_id_from=fsid, local_id_from=flid,
                          space_id_to=tsid, local_id_to=tlid, edge_type=et)


def _freq_ok(row: dict[str, Any]) -> bool:
    """① 频次 G1：sn+tn ≥ PROMOTE_FREQ_MIN。"""
    sn = int(row.get("sn", 0) or 0)
    tn = int(row.get("tn", 0) or 0)
    return (sn + tn) >= PROMOTE_FREQ_MIN


def _reward_ok(row: dict[str, Any]) -> bool:
    """② 边级 reward 信号支撑 G5-B：success_rate = sn/(sn+tn) ≥ PROMOTE_REWARD（无经验=未达·防全 0 假晋）。

    **G5-B 边级 promote·命名借用**（非 §十三 G5-C 记忆项延迟晋升闸·非 §十四 G5-A 自证机门因子·三 G5 同名不同物）。
    """
    sn = int(row.get("sn", 0) or 0)
    tn = int(row.get("tn", 0) or 0)
    total = sn + tn
    if total == 0:
        return False
    # cross_ge(sn, total, num, den) 即 sn/total ≥ num/den
    return cross_ge(sn, total, PROMOTE_REWARD_NUM, PROMOTE_REWARD_DEN)


def _definition_ok(edge_store: EdgeStore, ref: tuple,
                   teacher: Any = None) -> bool:
    """③ 定义 G3：结构锚（IS_A/PROPERTY/REFERS_TO 任一出边）∨ 教师确认（录放层）。"""
    fsid, flid, tsid, tlid, et = ref
    # 结构锚：该 from 节点有 IS_A/PROPERTY/REFERS_TO 出边（结构定义性·非裸共现）
    for anchor_et in (EDGE_IS_A, EDGE_PROPERTY, EDGE_REFERS_TO):
        if edge_store.query_from(fsid, flid, anchor_et):
            return True
    # 教师确认（断奶前·录放层·给事实判断·§8.1c-bis 来源③）
    if teacher is not None:
        confirmed = teacher.confirm_causes((fsid, flid), (tsid, tlid))
        if confirmed:
            return True
    return False


def _experience_ok(backend, ref: tuple[int, int]) -> bool:
    """刀4 D:11 experience 闸（experience_count 概念维对偶·绕 reward CAUSES-only 死锁）。

    D:11 边不接 reward（effective_weight:82 assert·sn/tn 恒 0）→ 既有 _reward_ok 永远 False。
    本闸读 experience_count 表 e_sn/e_tn（概念维·非 edge reward·reward_propagate 落点① feed）判
    e_sn/(e_sn+e_tn) ≥ 1/2 ∧ total ≥ PROMOTE_EXP_FREQ_MIN。无行/全 0 → False。

    **experience 主导**（用户拍板·硬条件）：D:11 promote 须此闸达标·无教师 at promote time·
    与初心"非教师精炼"对齐。构造性（reward 来自 judge 来自教师 GT）≠ 真独立源验证（#479 defer）。

    **审计根治 [严重-2]**：observe_mode 切 total（observe_tn 替 e_tn·避 β_arith·gate FREQ_OBSERVE_MODE 守·
    内部读 gate 同 selection_pref_score 读 SP_OBSERVE_MODE 范式·promote_edge/report 不改签名）。
    observe_tn 决策时写（dag_path add_active·非 episode 后批量）·独立 reward 符号·success_rate = e_sn/(e_sn+observe_tn)
    rate 不恒 1/2·缓解 β_arith（β_arith 病：reward>0 episode 同比 e_sn++/e_tn++ → rate=1/2 恰达标 promote 闸·
    observe_tn 不同比 e_sn 增→rate 降）。**不违 promote 三重②**：_experience_ok 是 D:11 专用闸（D:11 边不接
    reward 走 experience 对偶）·非三重②（_reward_ok 边级 sn/(sn+tn)·非 D:11）·路径1 改 _experience_ok 不动 _reward_ok。
    **诚实边界**：β_arith 缓解非根治·e_sn 仍 reward-feed 染·rate 缓解非恒 1/2（gate OFF 退化既有 bit-identical）。
    """
    observe_mode = bool(getattr(gates, "FREQ_OBSERVE_MODE", False))
    got = read_experience_count(backend, ref, observe_mode=observe_mode)   # (base_freq, e_sn, tn) | None
    if got is None:
        return False
    _, e_sn, tn = got
    # 对抗审 catch：observe_tn=0 假阳性守卫（gate ON 时 tn=observe_tn·observe_tn=0 → rate=e_sn/e_sn=1.0 假阳性 promote）。
    # 对称 selection_pref_score cap 天然防护（min(sp_sn,0)+0=0）·_experience_ok 须显式守卫·observe_tn=0 表无决策数据·不该判率·退 False。
    # 诚实边界：observe_tn < e_sn 时 rate > 1/2 反放宽（concept_targets 扩展非 path 节点·mitigation 退化·
    # β_arith 缓解依赖 observe_tn > e_sn 假设·非根治·gate OFF 退化既有 bit-identical）。
    if observe_mode and tn == 0:
        return False
    total = e_sn + tn
    if total < PROMOTE_EXP_FREQ_MIN:
        return False
    return cross_ge(e_sn, total, PROMOTE_EXP_SUCCESS_NUM, PROMOTE_REWARD_DEN)


def _definition_ok_d11(edge_store: EdgeStore, ref: tuple,
                       teacher: Any = None) -> bool:
    """刀4 D:11 G3 加分（teacher ∨ 同词教师 PRIMARY D:11 种子·软项·非必须）。

    word concept 节点无 IS_A/PROPERTY/REFERS_TO 出边（_definition_ok 结构锚失败）·故 D:11 用：
      ∨ 项 1：from 节点已有 source=SOURCE_TEACHER 的 PRIMARY D:11 边（同词教师种子确认·刀3 _REL_LEXICAL_CUE 范式）
      ∨ 项 2：teacher.confirm_causes（断奶前录放层·给事实判断）

    **软项**（experience 主导）：promote 时作 ∨ 加分项·有教师/种子时更易 promote（experience
    未达标但 teacher 确认也可 promote·断奶前稳）。MVP 反 theater 验收不依赖 teacher（experience 路径）。

    **对抗审 RISK-3 修**：∨ 项 1 限 source=SOURCE_TEACHER（教师种子）·**非**任意 PRIMARY D:11·
    防涌现 promote 的同词兄弟（source=BARE_TEXT）触发级联 promote（守 experience 主导·防未验关系混入 readback）。
    """
    from pure_integer_ai.storage.edge_store import SOURCE_TEACHER
    fsid, flid = ref[0], ref[1]
    # ∨ 项 1：from 节点已有教师种子 PRIMARY D:11 边（source=SOURCE_TEACHER·非涌现 promote 的 BARE_TEXT 兄弟）
    for r in edge_store.query_from(fsid, flid, edge_type=EDGE_RELATION_SIGNAL):
        if r.get("tier") == TIER_PRIMARY and r.get("source") == SOURCE_TEACHER:
            return True
    # ∨ 项 2：teacher 确认（断奶前录放层·给事实判断·§8.1c-bis 来源③）
    if teacher is not None:
        if teacher.confirm_causes((fsid, flid), (ref[2], ref[3])):
            return True
    return False


def _structure_match_ok(backend, word_sid: int, word_lid: int, rel_kind: int) -> bool:
    """对应泛化 v2 结构反推 tally 闸（D:11 promote 唯一证据轨·ORACLE_PROMOTE_MODE ON 时）。

    读 structure_match_count(W,R) distinct forming-sample ≥ PROMOTE_STRUCTURE_MATCH_MIN
    + **specificity gate**（审1 CONDITION 1·学得非写死·W' count(R) 须 > Σ count(other R)·特异主 R·
      非通用连接词·过滤"和/的"误晋）+ **显式守 CUE_CLUSTER_MODE**（审1 C3·OFF→ATTR_CUE_SIG 不写→
      cue slot 无→确定 soft-fail 非静默失效）。

    **非循环**（心脏·§四）：R 来自 REALIZES oracle（内容对命中 ConceptNet·source==CONCEPTNET·非 cue）·
    W 是观察（cue slot 落位）·提升反馈在 REALIZES source filter 断 → 非自证。
    **学全**：W 可新词（不在 oracle/frozenset）·distinct forming-sample 达阈即 promote·真泛化。
    无行/全 0/不特异 → False。gate OFF（caller promote_edge ORACLE_PROMOTE_MODE 分支不调此函数）。

    **specificity 诚实边界**（审1 CONDITION 1）：结构锚降噪非消噪·specificity 在 abstract_sig 异（使/是 skeleton
    LCA 不同）时有效（引发 只匹 使-skeleton→特异→晋）·abstract_sig 同（6a 残余）时弱·靠 held-out false-positive
    实测硬闸兜底（和/的 不晋·超阈 reject·非此处判定·eval 层）。
    """
    if not getattr(gates, "CUE_CLUSTER_MODE", False):
        return False   # 审1 C3：OFF→ATTR_CUE_SIG 不写→cue slot 无→确定 soft-fail（非静默失效）
    word_ref = (word_sid, word_lid)
    count = read_structure_match_count(backend, space_id=word_sid,
                                       word_ref=word_ref, rel_kind=rel_kind)
    if count < PROMOTE_STRUCTURE_MATCH_MIN:
        return False
    # specificity gate（审1 CONDITION 1）：W' count(R) 须 > Σ count(other R)（dominant·纯整数·学得非写死）
    per_rel = read_structure_match_per_rel(backend, space_id=word_sid, word_ref=word_ref)
    other = sum(c for rk, c in per_rel.items() if rk != rel_kind)
    return count > other


def promote_edge(edge_store: EdgeStore, node_store: NodeStore,
                 ref: tuple, *, teacher: Any = None,
                 backend: Any = None) -> bool:
    """promote（SHADOW→PRIMARY tier flip·§十三防塌C4）。

    ref = (from_sid, from_lid, to_sid, to_lid, edge_type)。返 True=已晋/本次晋·False=未达。

    **非 D:11 边**（既有三重）：频次 G1 + reward G5 + 定义 G3 全达 → flip。
    **D:11 EDGE_RELATION_SIGNAL 边**（刀4 双轨·绕 reward CAUSES-only 死锁·D:11 边 sn/tn 恒 0）：
      eligible = _experience_ok（experience 主导·硬·概念维 e_sn/e_tn）∨ _definition_ok_d11（teacher/种子 加分·软）。
      experience 主导 = 无教师 at promote time 亦可 promote（初心"非教师精炼"）·teacher 加分 ∨ 项（断奶前稳）。
      backend 须非 None（D:11 验证读 experience_count·caller _promote_eligible 传 ctx.backend）。
    达则 → edge tier flip + 概念点 tier = max 其边 tier（§十二⑤）。
    """
    fsid, flid, tsid, tlid, et = ref
    assert_int(fsid, flid, tsid, tlid, et, _where="promote_edge.ref")
    row = _edge_row(edge_store, ref)
    if row is None:
        return False
    if row["tier"] >= TIER_PRIMARY:
        return True   # 已 PRIMARY·幂等
    # 刀4 D:11 双轨 vs 既有三重（按 edge_type 分流）
    if et == EDGE_RELATION_SIGNAL:
        if getattr(gates, "ORACLE_PROMOTE_MODE", False):
            # 对应泛化 v2：结构匹配轨（审2条件3·两 gate 共存）。generator 关·D:11 SHADOW→PRIMARY **只认
            # _structure_match_ok**（distinct forming-sample ≥K + specificity + 守 CUE_CLUSTER_MODE）。
            # experience/teacher 轨退场（word 级 reward-feed 染·跨域污染 = 审2 BLOCKER 1 病源）。
            if backend is None:
                return False
            # rel_kind 桥（审2条件4a）：从 to-ref（REL_* 节点）读 ATTR_RELATION_PRIMITIVE int_a（自包含·
            # 无需 rel_primitives 透传·纯整数字典查）。to 非 REL_* primitive（不应发生·防御）→ 不晋。
            rel_kind = read_composes_attrs(backend, (tsid, tlid)).get(
                ATTR_RELATION_PRIMITIVE, (0, 0))[0]
            if rel_kind == 0:
                return False
            if not _structure_match_ok(backend, fsid, flid, rel_kind):
                return False
        else:
            # D:11 边不接 reward（sn/tn 恒 0）→ 既有 _reward_ok 永远 False·走双轨绕开（bit-identical·
            # knife4 旧路径回归守：HYPOTHESIS_MODE 命中此分支·experience∨teacher 既有不变）
            if backend is None:
                return False   # D:11 验证须 backend（读 experience_count）·无则不 promote（safe degradation）
            if not (_experience_ok(backend, (fsid, flid))
                    or _definition_ok_d11(edge_store, ref, teacher)):
                return False
    else:
        # 既有三重判据（非 D:11·bit-identical 不动）
        if not (_freq_ok(row) and _reward_ok(row)
                and _definition_ok(edge_store, ref, teacher)):
            return False
    # tier flip（MUTABLE_MONOTONE·edge_store.set_tier 守只升不降）
    edge_store.set_tier(space_id_from=fsid, local_id_from=flid,
                        space_id_to=tsid, local_id_to=tlid,
                        edge_type=et, new_tier=TIER_PRIMARY)
    # 概念点 tier = max 其边 tier（§十二⑤·两端节点都晋）
    # D:11 flip 改 PRIMARY 边 -> 清 lookup cache（bit-identical·lookup_word_concept run-scoped cache 失效）
    if et == EDGE_RELATION_SIGNAL:
        _cache = getattr(edge_store, "_d11_lookup_cache", None)
        if _cache is not None:
            _cache.clear()
    promote_concept(edge_store, node_store, (fsid, flid))
    promote_concept(edge_store, node_store, (tsid, tlid))
    return True


def promote_concept(edge_store: EdgeStore, node_store: NodeStore,
                    ref: ConceptRef) -> bool:
    """节点 tier = max 其边 tier（§十二⑤·MUTABLE_MONOTONE 只升不降）。

    概念点 tier = max(其出/入边 tier)·无边则保 SHADOW 不冒晋（stub #8 修：旧版无条件 flip PRIMARY
    靠 monotone 守卫防降·契约"max 其边 tier"未真现·无边节点也冒晋）。有 PRIMARY 边则概念点 PRIMARY 可参与建模。
    """
    sid, lid = ref
    cur = node_store.get(sid, lid)
    if cur is None:
        return False
    edge_max = TIER_SHADOW
    for row in edge_store.query_from(sid, lid):
        if row["tier"] > edge_max:
            edge_max = row["tier"]
    for row in edge_store.query_to(sid, lid):
        if row["tier"] > edge_max:
            edge_max = row["tier"]
    if edge_max <= cur["tier"]:
        return False   # 边 max 不高于当前 tier·不升（守 max 其边 tier 契约·不冒晋）
    node_store.set_tier(sid, lid, edge_max)
    return True


def promote_report(edge_store: EdgeStore, ref: tuple,
                   *, teacher: Any = None, backend: Any = None) -> dict[str, bool]:
    """判据诊断报告（不 flip·度量/调试用·promote 率统计·§十二阶段4 OOV 晋升率）。

    **eligible 须忠实预测 promote_edge 返回值**（生产入口 _promote_eligible 以 rep["eligible"]
    预筛后调 promote_edge·两函数判据须逐轨一致·否则预筛误淘汰 = theater）。

    D:11 EDGE_RELATION_SIGNAL 边：
      - ORACLE_PROMOTE_MODE ON（对应泛化 v2·审1 B1）：eligible = structure_match_ok
        （distinct forming-sample ≥K + specificity + 守CUE_CLUSTER + rel_kind 桥·镜像 promote_edge:231-244）。
      - OFF（刀4 双轨）：eligible = experience_ok ∨ definition_d11（experience 主导·加 "experience" 键）。
    D:11 不接 reward·freq/reward 恒 False。非 D:11 边（既有三重）：eligible = freq ∧ reward ∧ definition（bit-identical 不动）。
    """
    row = _edge_row(edge_store, ref)
    if row is None:
        return {"freq": False, "reward": False, "definition": False, "eligible": False}
    et = ref[4]
    if et == EDGE_RELATION_SIGNAL:
        if getattr(gates, "ORACLE_PROMOTE_MODE", False):
            # 对应泛化 v2 结构匹配轨（审1 post-impl B1 修·镜像 promote_edge:231-244）：
            # generator 关后 D:11 SHADOW 唯一来源 = tally hook（record_emergent_relation_signal_shadow）。
            # **promote_report["eligible"] 须忠实预测 promote_edge 返回值**——生产入口 _promote_eligible
            # (formal_train.py) 以 rep["eligible"] 预筛后调 promote_edge；此分支缺失 → tally 建 SHADOW
            # 被 eligible=False 淘汰 → promote_edge 永不调 → SHADOW 永不 flip PRIMARY → 学全 theater。
            # rel_kind 桥（审2条件4a·读 to-ref ATTR_RELATION_PRIMITIVE）+ _structure_match_ok（distinct≥K+specificity+守CUE_CLUSTER）。
            if backend is None:
                return {"freq": False, "reward": False, "definition": False, "eligible": False}
            rel_kind = read_composes_attrs(backend, (ref[2], ref[3])).get(
                ATTR_RELATION_PRIMITIVE, (0, 0))[0]
            sm_ok = (rel_kind != 0 and _structure_match_ok(backend, ref[0], ref[1], rel_kind))
            return {
                "freq": False, "reward": False, "definition": False,   # D:11 不接 reward·N/A
                "structure_match": sm_ok,    # D:11 oracle 轨专用诊断键（OFF 轨无此键）
                "eligible": (row["tier"] >= TIER_PRIMARY or sm_ok),
            }
        # D:11 双轨诊断（experience 主导 + teacher 加分·绕 reward CAUSES-only 死锁）
        # 与 promote_edge 一致：backend None → 不 eligible（safe degradation·D:11 验证须 backend）
        if backend is None:
            return {"freq": False, "reward": False, "definition": False, "eligible": False}
        exp_ok = _experience_ok(backend, (ref[0], ref[1]))
        def_d11 = _definition_ok_d11(edge_store, ref, teacher)
        return {
            "freq": False, "reward": False,   # D:11 不接 reward·sn/tn 恒 0·N/A
            "definition": def_d11,
            "experience": exp_ok,             # D:11 专用诊断键（非 D:11 无此键）
            "eligible": (row["tier"] >= TIER_PRIMARY or (exp_ok or def_d11)),
        }
    return {
        "freq": _freq_ok(row),
        "reward": _reward_ok(row),
        "definition": _definition_ok(edge_store, ref, teacher),
        "eligible": (row["tier"] >= TIER_PRIMARY
                     or (_freq_ok(row) and _reward_ok(row)
                         and _definition_ok(edge_store, ref, teacher))),
    }


def promote_memory_consolidate(backend: Any, memory_read: Any) -> int:
    """G5-C 记忆项延迟晋升闸（§十三:1108/1120 决断4·#732·**记忆项 status flip·非边级 tier flip**）。

    **三个 G5 同名不同物**：G5-A judge.py:167 自证机门因子（已 live·§十四:1245）/ G5-B promote.py:_reward_ok
    边级 promote（已 live·edge sn/tn）/ **G5-C 本函数**（记忆项 EXPERIENCE→CONSOLIDATED·memory_item SEG_EPISODIC 比率门）。

    扫 memory_item by info_ref（where space_id=memory_read.space_id·守两层物理分开 §十五决策1）·
    按 (info_ref_space, info_ref_id) 聚合 EXPERIENCE 态行·sum(count)/sum(success_count) 算比率门：
      sum(count) ≥ PROMOTE_MEM_FREQ_MIN ∧ cross_ge(sum_sc, sum_count, PROMOTE_MEM_REWARD_NUM, PROMOTE_MEM_REWARD_DEN)
      → consolidate flip 所有同 info_ref 的 EXPERIENCE 行 status EXPERIENCE→CONSOLIDATED。

    **公式数学**（审1 P0-1）：record_use(success=True)→sc+=2/count+=1·故 sum(sc)=2×positive·sum(count)=total·
      cross_ge(sum_sc,sum_count,1,1)=sum_sc≥sum_count=2×positive≥total=≥50% positive 才 consolidate。
      **非镜像 G5-B**（G5-B sn/(sn+tn)=positive 比率·结构不同·阈值差 4 倍·独立 oracle 标）。

    返 consolidate_count（flip 的 memory_item 行数）。已 CONSOLIDATED 行跳过（幂等）。

    **诚实边界**：G5-C 是 reward 回溯统计·非真因判定（#479 墙·stable≠correct）。record_use 接线构造性
    （reward 来自 judge 来自教师 GT·非真独立源验证）。G5-C 与 tri_space 中环正交（offline 扫 vs online 检索·非双重 theater）。
    **调用模式偏离** line 1120 同 memory_item 多事件累加——本函数 caller 侧 sum 跨 episode 聚合·数学等价·
    语义偏离（doc §十三 line 1120 已同步修注·#732 实施片）。
    """
    rows = backend.select("memory_item", where={"space_id": memory_read.space_id})
    # 按 (info_ref_space, info_ref_id) 聚合 EXPERIENCE 态行
    by_ref: dict[tuple[int, int], list] = {}
    for row in rows:
        if int(row.get("status", 0)) != STATUS_EXPERIENCE:
            continue   # 已 CONSOLIDATED 跳过（幂等）
        info_key = (int(row.get("info_ref_space", 0) or 0), int(row.get("info_ref_id", 0) or 0))
        if info_key not in by_ref:
            by_ref[info_key] = [0, 0, []]   # [sum_count, sum_sc, local_ids]
        by_ref[info_key][0] += int(row.get("count", 0) or 0)
        by_ref[info_key][1] += int(row.get("success_count", 0) or 0)
        by_ref[info_key][2].append(int(row["local_id"]))
    consolidate_count = 0
    # sorted(info_key) 守 bit-identical（确定迭代序·consolidate 幂等顺序无关但守确定）
    for info_key in sorted(by_ref):
        sum_count, sum_sc, lids = by_ref[info_key]
        if sum_count < PROMOTE_MEM_FREQ_MIN:
            continue
        if not cross_ge(sum_sc, sum_count, PROMOTE_MEM_REWARD_NUM, PROMOTE_MEM_REWARD_DEN):
            continue
        # G5-C 闸判达 → consolidate 所有同 info_ref 的 EXPERIENCE 行
        for lid in lids:
            memory_read.consolidate(lid)
            consolidate_count += 1
    return consolidate_count
