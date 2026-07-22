"""cognition.process.reward_propagate — 模块8 reward 反传通道（CAUSES 头 + 5 落点·R1）。

propagate_reward(path_result, output_words, reward, ctx_tag, intent_type, workmem, *, edge_store, backend)
  只走 CAUSES 头（PRECEDES 永不接 reward·strength 恒=1·AST 真值序不污染）。
  ★reward 多头真墙决断（修正分析五 / P0-2·doc/重来_P0决断集_修正分析十三.md §三）：reward 永走
    CAUSES-only = reward 通道防塌柱①纪律（逐边塌缩有意防·非丢弃）·非因果经验走 experience_count（概念维对偶·非
    reward 多头）。纠"reward 多头=丢弃"误判（=reward 通道铁律·经验学习走 experience_count 非扩多头）。
    code assertion 已落阶段2d（落点① 循环 assert distributed 全 CAUSES·把纪律从 doc 升 code）。
    ★分层（2026-07-09 纠偏轮 round1 lens·非"禁止扩宽"的铁律）：(a) 频次类边（COOCCURS/PRECEDES）
    reward 纪律成立不可放宽·(b) 命题类边（PROPERTY）reward 接入待 P0.3/P0.4 实证·(c) 独立表
    （experience_count/selection_pref_count/sense_candidates）概念维对偶可扩有先例·非 reward 多头。
  reward 符号契约（§十三D-E3）：judge 产 ≥0 / 步进死路产 <0 / propagate 接收可负。

  R1 落盘：sn/tn 判定用 **episode 级 reward 符号**非边级 delta_reward（防新边 delta floor=0
    误判 failure 永久卡 rate=0）：
      reward>0  → 每条选定 CAUSES 边 sn++ & tn++（参与即成功·episode 级）+ strength+=Δ（分功>0 才加）
      reward==0 → 每条选定 CAUSES 边 tn++（judge veto·破永正·非"不调"）
      reward<0  → 每条选定 CAUSES 边 tn++（死路·率自然降·不 decrement sn 守单调）
    delta 分功仍用于 strength 幅度（高率边多涨·仅 reward>0 时对分功>0 边加 Δ）。
  R5 落盘：Σactive 率=0 边界（全脏边·冷启动）→ reward>0 每条 sn++&tn++（首次观测给新边机会·
    不依赖分功·不丢弃成功信号）·reward≤0 tn++。
  A3 多部分反传：按 active 前驱率分功（A3·复用同机制·整数 floor+余数按 ref 序分发）。

  5 落点（§十三D-P2/B3）：
    ① 核心 CAUSES sn/tn+strength（持久学习·首版必做 production 落点）+ 概念维对偶 experience_count
       feed（阶段2·同 episode reward 聚合到概念行 e_sn/e_tn·零新管道·端点来自 causes_edges 守 CAUSES-only）
    ② 记忆 episodic memory_item（reward>0 SEG_EPISODIC / reward<0 SEG_NEGATIVE·**M10 第一刀 11d
       已真写 memory_read 一层**·**#732 G5-C 闸落 code**：record_use 接线 count/sc 累加 + promote_memory_consolidate
       caller 侧 sum 聚合 by info_ref·消费者部分活·tri_space 中环五连断 4/5 仍断·写活+G5-C 读活+中环读 defer）
    ③ COOCCURS 不接 reward（防塌 C1·credit_sink 弃用 COOCCURS reward 落点·只接 Hebbian 共激活）
    ④ 上下文敏感标记台账（C2·reward 标记不定义·marker ledger 随 Stage 6·**首版 defer**）
    ⑤ pass_reward 台账（D-R3 过程反传 default OFF·噪音风险·设计意图）
    ⑥ 选择倾向 pair-key reward feed（**S4 片3 第三条腿**·gate SELECTION_PREF_FEED_MODE default OFF·
       concept_targets 同空间 i<j 配对算 class_of → record_selection_pref_reward sp_sn/sp_tn·独立表守 CAUSES-only·
       不进 distributed/record_episode_result·:131 assert 不触发·镜像落点① 概念维对偶 pair-key 扩展）

gate：闭环核心默认 ON（H1）·落点⑤ pass_reward 台账 defer Stage 6（原 PROCESS_REWARD_PROP_MODE dead stub 已删 P3 #1054·机制未实现）。
铁律：纯整 / MUTABLE_MONOTONE（sn 单调·base_strength append-only 不动）/ append-only（台账/记忆）/
  确定性（整数 floor+余数按 ref 序分发）。
诚实边界：CAUSES OR 真因归因（按 active 前驱率分功是可观测信用分配非真因判定·§十三D）·
  reward 只标记多义不定义（判多义=接地墙·定义权归教师）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.config import gates
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.experience_count import record_experience_outcome, pack_ctx_code
from pure_integer_ai.storage.spaces.memory_space import SEG_EPISODIC, SEG_NEGATIVE
from pure_integer_ai.cognition.shared.types import (
    EdgeRef,
    MODALITY_LANGUAGE,
    PathResult,
    REWARD_LEGITIMATE_DOMAINS,
)
from pure_integer_ai.cognition.process.effective_weight import edge_rate, is_unobserved, RATE_SCALE
from pure_integer_ai.cognition.process.abstraction import nearest_isa_ancestor

# strength 增量默认（delta=1 默认零调参·率自归一化溶解震荡·§十三D）
DELTA_DEFAULT = 1

# 落点⑥ 第三条腿 reward 侧选择倾向配对 concept 上限（S4 片3·防御 cap·episode concept_targets 通常小·
# pathological 大 episode 节流·sorted 确定 truncate·bit-identical）。
_REWARD_SP_MAX_CONCEPTS = 64


def _nearest_isa_ancestor_reward(amap: dict, ref: tuple[int, int]) -> tuple[int, int]:
    """ref 的 IS_A 最近祖先（最深·S4 项2·转调 abstraction.nearest_isa_ancestor·写读一致三处同源）。

    保留 wrapper 守 reward_propagate 内调用点（:248/:249）不变 + 守 process→understanding 单向依赖
    （转调 process.abstraction 同包·非 understanding.selection_pref）。多层 IS_A 真 LCA 落地（替原升序首 min）。
    """
    return nearest_isa_ancestor(amap, ref)


def _fetch_edge_row(edge_store: EdgeStore,
                    edge_ref: EdgeRef) -> dict[str, Any] | None:
    """经 EdgeStore 读取 EdgeRef 对应唯一旧边，禁止奖励路径静默选择首行。"""
    return edge_store.get(
        space_id_from=edge_ref[0], local_id_from=edge_ref[1],
        space_id_to=edge_ref[2], local_id_to=edge_ref[3],
        edge_type=edge_ref[4],
    )


def _distribute_by_rate(items: list[tuple[EdgeRef, int]], reward: int
                        ) -> list[tuple[EdgeRef, int]]:
    """按 active 前驱率分功：reward 单位按 rate 比例分到各边（整数 floor+余数按 ref 序）。

    items：[(edge_ref, rate), ...]（rate = effective_weight 的 rate 部分·0..1000）。
    返 [(edge_ref, delta_reward), ...]。Σ=0（R5）→ 返全 0（调用方按 episode 符号处理）。
    仅 reward>0 时分发有意义（reward≤0 不调 strength·delta 不用）。
    """
    if not items:
        return []
    sum_rate = sum(r for _, r in items)
    if sum_rate <= 0 or reward <= 0:
        return [(ref, 0) for ref, _ in items]
    # floor 分配
    floors = [(ref, (reward * r) // sum_rate) for ref, r in items]
    allocated = sum(f for _, f in floors)
    remainder = reward - allocated
    # 余数按 ref 序分发（确定·bit-identical·ref 升序）
    if remainder > 0:
        order = sorted(range(len(items)), key=lambda i: items[i][0])
        idx = 0
        while remainder > 0 and order:
            i = order[idx % len(order)]
            floors[i] = (floors[i][0], floors[i][1] + 1)
            remainder -= 1
            idx += 1
            if idx >= len(order):
                break
    return floors


def propagate_reward(path_result: PathResult, output_words: list[str],
                     reward: int, ctx_tag: tuple, intent_type: int,
                     workmem: Any, *, edge_store: EdgeStore,
                     backend: Any, memory_read: Any = None) -> None:
    """reward 反传（CAUSES 头 + 5 落点·R1 episode 级符号）。

    path_result：步进产出（模块4·path.edges 选定边集）。
    output_words：生成结果词（落点④ 读·卷三模块1 产出·F6）。
    reward：episode 级 reward 符号（judge ≥0 / 死路 <0 / veto=0）。
    ctx_tag：多维 (domain, modality, task, intent_type)（落点② context_tag·F7）。
    intent_type：意图类型（落点⑤ pass_reward·context_tag 维）。
    """
    assert_int(reward, intent_type, _where="propagate_reward.reward")
    assert_no_float(reward, _where="propagate_reward.reward_float")
    if ctx_tag[1] == MODALITY_LANGUAGE:
        return
    backend = backend or edge_store._b

    # —— 只走 CAUSES 头（PRECEDES 永不接 reward·M2 读 path.edges 选定 CAUSES 边） ——
    causes_edges: list[EdgeRef] = [e for e in path_result.path.edges
                                   if e[4] == EDGE_CAUSES]

    # 取边行 + rate（分功用）
    items: list[tuple[EdgeRef, int]] = []
    rows: dict[EdgeRef, dict[str, Any]] = {}
    for ref in causes_edges:
        row = _fetch_edge_row(edge_store, ref)
        if row is None:
            continue   # 边已不在（冷区脱离）·跳过
        rows[ref] = row
        items.append((ref, edge_rate(row)))

    sum_rate = sum(r for _, r in items)
    distributed = _distribute_by_rate(items, reward) if sum_rate > 0 else \
        [(ref, 0) for ref, _ in items]

    # —— 止血（methodology doc §五·reward 非 frame #1146）：reward-illegitimate 域（语言 DOMAIN_TEXT /
    # bare DOMAIN_BARE·judge G5 vacated 无 correctness 锚·reward 结构性 theater）剔出 CAUSES edge reward 写 ——
    # 语言 episode 全 dead-end（热区 k=2 排 sink / PRECEDES 成环 dag_path:253-254）→ reward≤0 → 落点① CAUSES
    # tn++ 在主动惩罚唯一 reward-active 边（有害·§二铁事实3）。gate CAUSES_REWARD_DOMAIN_FILTER_MODE
    # （default OFF 守 CI bit-identical·OFF → _skip 恒 False → 落点① 逐字现状）。
    # **仅剔 edge 写（sn/tn/strength）**：概念维对偶 experience_count（落点① 对偶段 :168-186）/ memory（落点②）/
    # selection_pref 台账（落点⑥）不受影响——皆概念维 / 独立台账·非 edge reward 多头（docstring :9-11 明示）。
    # 语言模态已在函数入口拒绝 scalar reward 的全部持久写；本域过滤继续保护其他无自证锚的旧域。
    _skip_causes_edge_write = (
        getattr(gates, "CAUSES_REWARD_DOMAIN_FILTER_MODE", False)
        and ctx_tag[0] not in REWARD_LEGITIMATE_DOMAINS
    )

    # —— 落点①：核心 CAUSES sn/tn+strength（持久学习·R1 episode 级符号） ——
    for ref, delta_reward in distributed:
        assert ref[4] == EDGE_CAUSES, (
            f"reward feed 防塌柱①·distributed 须全 CAUSES·got edge_type={ref[4]} "
            "(distributed 来自 causes_edges 过滤·invariant·防未来 edit 静默引入非 CAUSES reward feed)")
        if _skip_causes_edge_write:
            continue   # 止血：reward-illegitimate 域跳过 CAUSES edge sn/tn/strength 写（概念维对偶 feed 仍执行）
        row = rows[ref]
        if reward > 0:
            if sum_rate > 0:
                # 正常分功：参与即成功 + strength+=Δ（分功>0 才加）
                sn_delta, tn_delta = 1, 1
                strength_delta = DELTA_DEFAULT if delta_reward > 0 else 0
            elif is_unobserved(row):
                # R5：未观测边（sn=tn=0）首次观测给机会（sn++&tn++·不依赖分功·不丢弃成功信号）
                # item3 缺漏2：冷启动死锁破——未观测边进 path.edges 后 R5 兜底给首次 sn++
                # 不加 strength（无分功→delta=0·strength 幅度靠后续有观测后的分功）
                sn_delta, tn_delta = 1, 1
                strength_delta = 0
            else:
                # 已失败边（sn=0 tn>0·rate=0·sum_rate==0 因他边也未观测或无 active）
                # 不给 sn++（已失败不该因 reward>0 翻盘）·tn++ 守永正
                sn_delta, tn_delta = 0, 1
                strength_delta = 0
        elif reward < 0:
            # 死路失败：tn++ only（率自然降·不 decrement sn 守单调）
            sn_delta, tn_delta, strength_delta = 0, 1, 0
        else:
            # reward==0：judge veto（破永正·非"不调"·reward 符号契约§十三D-E3）
            sn_delta, tn_delta, strength_delta = 0, 1, 0
        edge_store.record_episode_result(
            space_id_from=ref[0], local_id_from=ref[1],
            space_id_to=ref[2], local_id_to=ref[3], edge_type=EDGE_CAUSES,
            sn_delta=sn_delta, tn_delta=tn_delta,
            strength_delta=strength_delta)
        # 守不变量：base_strength append-only 不动（record_episode_result 不碰 base_strength）
        # sn 单调（sn_delta≥0）·strength 单调（strength_delta≥0）

    # —— 落点① 概念维对偶：reward episode 聚合到概念行（experience_count·阶段2·零新管道）——
    # 主集 = CAUSES 边两端点 + path sink + struct_unit_refs·set 去重（同 episode 同概念只 feed 一次·布尔参与非计数）
    # 端点来自 causes_edges（line 109-110 CAUSES-only 过滤）→ 结构性守 reward CAUSES-only 真墙（2d 数据流守门）
    concept_targets: set = set()
    for _ref in causes_edges:
        concept_targets.add((_ref[0], _ref[1]))   # from 端点 (from_sid, from_lid)
        concept_targets.add((_ref[2], _ref[3]))   # to   端点 (to_sid,   to_lid)
    if path_result.sink is not None:
        concept_targets.add(path_result.sink)
    for _su_ref in path_result.path.struct_unit_refs:
        concept_targets.add(_su_ref)
    # —— 落点① 概念维对偶扩：刀4 子环3 鸡生蛋破解（gate EMERGENT_RELATION_FEED_MODE·default OFF 守回归） ——
    # D:11 SHADOW 候选 word concept（涌现假设·非-cue 词"引发"无 CAUSES 边·不在主 concept_targets）进
    # concept_targets 被经验 feed·子环4 _experience_ok 验证（e_sn/e_tn 达标→promote）。
    # **守 reward CAUSES-only 防塌柱①**：只 word ref 进 concept_targets（experience_count 概念维对偶·
    # 非 edge reward 多头）·causes_edges/distributed/record_episode_result 全不变·D:11 边永不进 edge
    # sn/tn 写·:131 CAUSES-only assert 不动。
    if getattr(gates, "EMERGENT_RELATION_FEED_MODE", False):
        from pure_integer_ai.cognition.understanding.emergent_relation_feed import (
            collect_emergent_word_concepts_for_feed,
        )
        concept_targets.update(collect_emergent_word_concepts_for_feed(backend))
    # 确定性迭代序（sorted ConceptRef tuple·bit-identical·同 episode 同概念只 feed 一次）
    _ctx_code = pack_ctx_code(*ctx_tag)   # ctx_tag 四维 → ctx_code 单维（第二刀启用·防混淆分桶·阶段6）
    for _c_ref in sorted(concept_targets):
        record_experience_outcome(backend, ref=_c_ref, reward=reward, ctx_code=_ctx_code)
        # R1 符号在 record_experience_outcome 内：reward>0→e_sn++&e_tn++ / reward≤0→e_tn++
        # 表未注册（bare fixture）→ record_experience_outcome 内 try/except KeyError 兜底 skip（阶段2 改动点 C）

    # —— 落点⑥ 选择倾向 pair-key reward feed（S4 片3·第三条腿·gate SELECTION_PREF_FEED_MODE·守 CAUSES-only）——
    # concept_targets（已 CAUSES-only 过滤 :168-175·CAUSES 端点 + sink + struct_unit_refs）同空间内 i<j 配对·
    # 双向算 (a, class_of(b)) / (b, class_of(a))·feed record_selection_pref_reward（独立表·**不进
    # causes_edges/distributed/record_episode_result**·:131 assert 在 :130 distributed 循环·本段独立·assert 不触发·
    # 守 reward CAUSES-only 防塌柱①·effective_weight:82 assert 不内只写独立表）。
    # **class_of 自建**（build_isa_ancestor_map process.abstraction 同包向下 + _nearest_isa_ancestor_reward 转调 nearest_isa_ancestor 最深·
    # 不复用 understanding.selection_pref._nearest_isa_ancestor·守 process→understanding 单向依赖）。
    # **配对语义边界**：reward 侧 concept_targets 配对是 **episode 级粗聚合**（非 observe 段内精确）·设计故意
    # （observe build_selection_pref_count 段内精确共现 + reward 粗标 episode 成功·两路同表不同列不同写函数）。
    # **双向 + 无词性过滤**：每对 (a,b) 双向算 (a,class_of(b))+(b,class_of(a))·且 a/b 不限词性（动词×动词同 episode
    # 也配对·如"追"+"跑"同 episode）—— 这是比 observe 段内 builder（单向 concept_a→argument_class）更粗一档的聚合·
    # 生成侧精查 read_selection_pref_count pair rate 兜底区分（两层正交·S4 决断 2）·PR 侧 w_sp 只粗筛不精分。
    # 镜像 EMERGENT_RELATION_FEED_MODE :182-186 先例（gate 守·concept_targets·独立表 feed）。
    # 性能 defer：每 episode 重算 ancestor_map（transitive_closure·IS_A 边数 O(n)·首版 <千级可接受·
    # concept_targets 已 cap _REWARD_SP_MAX_CONCEPTS=64 减闭包计算·hoist 到 episode_loop 外 per-space 共享 defer）。
    if getattr(gates, "SELECTION_PREF_FEED_MODE", False):
        from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map
        from pure_integer_ai.storage.selection_pref_count import record_selection_pref_reward
        _by_space: dict[int, list[tuple[int, int]]] = {}
        for _ref in concept_targets:
            _by_space.setdefault(_ref[0], []).append(_ref)
        for _sid_sp, _refs_sp in _by_space.items():
            _refs_sp_sorted = sorted(_refs_sp)
            if len(_refs_sp_sorted) > _REWARD_SP_MAX_CONCEPTS:
                _refs_sp_sorted = _refs_sp_sorted[:_REWARD_SP_MAX_CONCEPTS]   # 防御 cap·sorted 确定 truncate
            if len(_refs_sp_sorted) < 2:
                continue
            _amap = build_isa_ancestor_map(backend, space_id=_sid_sp)
            for _i in range(len(_refs_sp_sorted)):
                _a = _refs_sp_sorted[_i]
                for _j in range(_i + 1, len(_refs_sp_sorted)):
                    _b = _refs_sp_sorted[_j]
                    if _a == _b:
                        continue
                    _cls_b = _nearest_isa_ancestor_reward(_amap, _b)   # a 共现 b 的类
                    _cls_a = _nearest_isa_ancestor_reward(_amap, _a)   # b 共现 a 的类
                    record_selection_pref_reward(backend, ref_a=_a, ref_class=_cls_b, reward=reward)
                    record_selection_pref_reward(backend, ref_a=_b, ref_class=_cls_a, reward=reward)

    # —— 落点②：记忆 episodic memory_item（reward>0 SEG_EPISODIC / reward<0 SEG_NEGATIVE） ——
    # M10 第一刀 11d：真写 memory_read 一层（训练期建阅读记忆种子·主线 doc §十三"训练期记忆建但不参与"）。
    # 单 sink 两列（info_ref_space/info_ref_id·与 experience_count 落点① 概念聚合正交不重复）·
    # content_hash 留 0 占位（reward_propagate 无 surface·未来 lazy backfill from concept_identity）。
    # 反 theater 诚实边界：写活（no-op→真 insert·行为变）+ **G5-C 闸 #732 落 code**（record_use 接线 +
    # promote_memory_consolidate caller·消费者部分活）+ tri_space 中环五连断 4-5 仍断（B1 layer-defer 范式）。
    # 不伪称闭环——锚是真写+G5-C 真消费+兑现"训练期建阅读种子"·非"中环激活"（中环 4-5 仍 defer）。
    # reward==0 不写（judge veto·非经验信号·守记忆洁净）/ sink=None 不写（无锚）/ memory_read=None 不写（退化）。
    if memory_read is not None and reward != 0 and path_result.sink is not None:
        _seg = SEG_EPISODIC if reward > 0 else SEG_NEGATIVE
        _lid = memory_read.new_local_id()
        memory_read.put(
            _lid, content_hash=0, session_id=None,
            seg_type=_seg,
            info_ref_space=path_result.sink[0], info_ref_id=path_result.sink[1],
            context_tag=pack_ctx_code(*ctx_tag),
            round_id=getattr(workmem, "round_id", 0))
        # #732 G5-C record_use 接线：每 episode 新建行后立即 update 同行 count/sc（if 块内·reward!=0 守·
        # 防若误放 if 外 reward==0 时 _lid 未定义 NameError·审2 P1-2）。守 record_use 函数语义（line 1120
        # success_count += 2 per success）·调用模式偏离（per-row count 恒 1·G5-C caller 侧 sum 聚合·doc line 1120 修注）。
        # **record_use 接线不受 G5_C_CONSOLIDATE_MODE gate 控制**（审2 P1-1 诚实标注）：gate 守 consolidate caller
        # （formal_train _promote_eligible 后）·不守 write-side 累加·否则 gate ON 时 count/sc 恒 0 → G5-C 闸读 0/0
        # 恒 False = theater。gate OFF 时 count/sc 仍累加（行为变化·测试层零回归·test_stage11 不查 count/sc·
        # test_formal_train 不查 memory_item·生产层数值变但 bit-identical 守）。
        memory_read.record_use(_lid, success=reward > 0)

    # —— 落点③：COOCCURS 不接 reward（防塌 C1·credit_sink 弃用 COOCCURS reward 落点） ——
    # COOCCURS 只接 observe 时 Hebbian 共激活 +1（sign-agnostic·卷一模块6）·不接 reward。

    # —— 落点④：上下文敏感标记台账（C2·reward 标记不定义·marker ledger 随 Stage 6 defer） ——
    # for word in output_words: marker_ledger_update(word, ctx_tag, sign(reward))
    # 超阈值→触发教师 re-define 同词第二义（REFERS_TO 多挂·§十一缺口#2·定义权归教师接地墙）
    _ = output_words

    # —— 落点⑤：pass_reward 台账（D-R3 过程反传·defer Stage 6·D8 source/milestone/count 派生） ——
    # 原 PROCESS_REWARD_PROP_MODE gate 已删（P3 #1054 死码清理·dead stub·body=pass·翻 ON no-op·
    # gate 存在暗示机制实为空·违"无死码"）。台账机制未实现·设计意图保留：
    #   pass_reward_ledger_update(intent_type, source=path_result.source,
    #                             milestone=path_result.sink, count=1)
    # 随 Stage 6 落（届时加 gate + 真实现·非空 stub）。
