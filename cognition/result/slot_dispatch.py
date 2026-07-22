"""cognition.result.slot_dispatch — 模块2 逐槽分派（概念填槽 vs 记忆序列回放·target_lang 偏好）。

dispatch_slot(slot, dag_path, graph, workmem, target_lang) -> (word, source)
  记忆序列 → 回放直出（序列上下文已固定词形免逐节点消歧·标 DEF_REPLAY 血统不伪装）
  单概念   → 词形路径（REFERS_TO 反向 activate_candidates 返全部禁取首 → collide_score
            上下文扩散纯整数共现选最佳 → target_lang 硬偏好同 lang → modality_serialize 序化）

  target_lang 硬偏好（§十四C1）：同 lang 词形优先·无同 lang 才回退跨 lang
    （防中文答 apple/for 泄数学·C1 防跨语言污染）。
  逐槽分派非逐路径（真实路径常混合·代码主干组合逻辑+末端函数名回放）。

复用：REFERS_TO 归一链（卷一模块3）/ activate_candidates（graph_view）/ collide_score /
  structure_template（modality_serialize·非 anchored·M11·骨架学得非预制）。
gate GENERATE_MODE：**承重件(逐槽分派·闭环必跑)永远 active·dispatch_slot 函数体不读 gate·gate 装饰性保留位**(无 OFF 态·同 generate.py)。
铁律：纯整数（collide_score 共现计数）/ 不写死（role 涌现自 emergent_role 非硬编码 SVO·M11）/
  最少冗余（回放复用记忆已有序列不重建/填槽复用 structure_template 不新造）。
诚实边界：消歧=拓扑共现非语义接地（共现选词≠语义理解·§十四诚实边界）/ 回放=记忆复现非生成。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.types import (
    RoleSlot, ConceptRef, PathResult, LINEAGE_CONCEPT_FILL, LINEAGE_DEF_REPLAY,
    CUE_SLOT_FILL,
)
from pure_integer_ai.cognition.shared.types import LANG_NONE
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.config import gates

# collide_score ×1000 缩放（共现分→选优权重·纯整·确定性 tiebreak）
SCORE_SCALE = 1000
PR_SLOT_BONUS_CAP = 3   # pronoun 维加成 cap（recency 尺度·同 observe 侧 PR_TN_BONUS_CAP·并入 sp 维联合 _cap_sp 守 999·不颠覆 collide 主轴）
# 对应泛化桥 correspondence bonus（(β) 独立轴·doc/重来_对应泛化_readback_generation_桥·两审定值=1001）：严格胜 collide=1(1000)
# → held-out cue 词(collide=0) 稳定激活（floor 可重复·1000 tie 致 ref 序决 50/50 不可重复）·输 collide=2+(2000+)守共现主轴·
# 学全不颠覆共现。**独立轴**：不入 _cap_sp（审1 CONDITION B·cap 999 下 held-out collide=0 cue 词选不出=未过墙）。
CORR_BONUS = SCORE_SCALE + 1   # =1001


def _cap_sp(sp: int) -> int:
    """sel_pref 维 cap 在 SCORE_SCALE-1（守 collide 主轴：1 个真 token 共现 > 999 个 class 共现）。

    sel_pref 是亚主轴 tiebreak（0..999）·combine = collide×SCORE_SCALE + _cap_sp(sp)·
    保证 collide 维永远 > sel_pref 维（collide=1,sp=0 → 1000 > collide=0,sp=capped999 → 999）。
    **cap 内部组成**（S4 后续加固 sp_sn 维后）：sel_pref = sp_tn + sp_sn（共现总数 + 成功搭配加成）·
    cap 999 可由 sp_tn+sp_sn 联合达成（如 sp_tn=500+sp_sn=499）·非纯 token 共现·但 cap 本质
    "sel_pref 是亚主轴"未变·不影响 collide 主轴优先（1000 > 999 不变量守）。
    设计故意·大 episode sp_tn/sp_sn 累计超 999 时 sel_pref 维饱和退 ref tiebreak（诚实边界·真无 cap defer）。
    """
    return sp if sp < SCORE_SCALE else SCORE_SCALE - 1


def _pronoun_bonus(graph: ConceptGraph, c: ConceptRef,
                   ctx_refs: list[ConceptRef]) -> int:
    """c 作为 pronoun 解析到 ctx_refs 中各 token 的 max pr_tn（cap PR_SLOT_BONUS_CAP·pair-key 对偶 observe 侧）。

    **对抗审 catch [严重-3] 修**：原 pronoun_score(c, slot.ref) 中 slot.ref=struct_ref（generate.py:124
    RoleSlot(ref=unit)·结构概念点）≠ observe 侧写的 antecedent_ref=token concept ref（词概念）→ pair-key
    错位·pr_tn 永远 0·theater。修：改用 ctx_refs（上下文 token·prior_topic_refs + produced_refs·token
    concept ref）·对每个 ctx_ref 查 pronoun_score(c, ctx_ref)·取 max·pair-key (c=pronoun, ctx_ref=antecedent
    token) 匹配 observe 侧写 (pronoun_ref, antecedent_ref)·数据流闭环（refers_occurrence.py:157 同 pair-key）。
    a==b 跳过避自 boost（对称 observe 侧·镜像 selection_pref_score a==b continue）。
    cap PR_SLOT_BONUS_CAP 提前退出（recency 尺度·不颠覆 collide 主轴·联合 _cap_sp 守 999）。
    """
    bonus = 0
    for ctx_ref in ctx_refs:
        if ctx_ref == c:
            continue   # a==b 跳过·避自 boost（对称 observe 侧·镜像 selection_pref_score a==b continue）
        pr = graph.pronoun_score(c, ctx_ref)
        if pr > bonus:
            bonus = pr
        if bonus >= PR_SLOT_BONUS_CAP:
            break   # cap 提前退出
    return min(bonus, PR_SLOT_BONUS_CAP)


def _second_order_bonus(graph: ConceptGraph, c: ConceptRef,
                        ctx_refs: list[ConceptRef]) -> int:
    """c 与 ctx_refs 各 token 的 max 二阶相似（shared-neighbor Jaccard·Phase C §十五-bis C.2）。

    **per-candidate（vary with c）**——审1/审2 承重修 [严重-3] pair-key错位 bug 同类：原设计 `second_order_similarity
    (slot.ref, ctx)` 用 slot.ref=struct_ref（候选循环常量）≠ 候选 c·常量 offset 被 _cap_sp 吸收→零 argmax 效果
    →consumer live-in-code 但 inert-in-effect→C.3 defer 退化为 defer-as-shield。修：per-candidate c·mirror _pronoun_bonus。
    max over ctx_refs·skip c==r 避自 boost·early-exit on SCORE_SCALE-1（cap·联合 _cap_sp 守 999·collide 主轴 1000>999）。
    **真 exposure signal**（COOCCURS observed·确定性 computation over 观察数据·非"学得"·同 collide_score 一阶）。
    """
    bonus = 0
    for ctx_ref in ctx_refs:
        if ctx_ref == c:
            continue   # a==b 跳过避自 boost（镜像 _pronoun_bonus / selection_pref_score a==b continue）
        sim = graph.second_order_similarity(c, ctx_ref)
        if sim > bonus:
            bonus = sim
        if bonus >= SCORE_SCALE - 1:
            break   # cap 提前退出（_cap_sp max 999·联合守 collide 主轴·镜像 _pronoun_bonus PR_SLOT_BONUS_CAP early-exit）
    return bonus


def _correspondence_bonus(graph: ConceptGraph, c: ConceptRef,
                          unit_rel_kind: int, is_cue_slot: bool) -> int:
    """(β) 独立轴 correspondence bonus：cue slot + rel_kind 匹配 → CORR_BONUS（对应桥第 8 路）。

    对应泛化 readback→generation 桥（doc/重来_对应泛化_readback_generation_桥·CORRESPONDENCE_SLOT_MODE）。
    v2 学到的 cue 词（D:11 W→REL_*·source=SOURCE_BARE_TEXT·cue_rel_of(c)）在 R-unit 的 cue slot 匹配
    unit_rel_kind（skeleton REALIZES 的 R）→ CORR_BONUS（1001·严格胜 collide=1·独立轴不入 _cap_sp）。
    让学到的对应驱动生成（学全 BEYOND 共现·非 collide 驱动）。

    **cue-slot-aware（审1 CONDITION D·反 theater 心脏）**：is_cue_slot=False → 返 0（非 cue slot 不触发·
    subject/object 位走 collide·防全 slot 激活同 cue 词=theater）。ATTR_CUE_SIG 天然单 cue 位（审2 证 _cluster_by_cue
    单拆硬约束）→ bonus 当前不可能多 slot 触发。unit_rel_kind=0（非 R-skeleton/无 INSTANTIATES）→ 返 0（robust 退化）。

    **不入 _cap_sp**（独立轴·审1 CONDITION B）：combine = s×SCORE_SCALE + _cap_sp(sub) + correspondence_bonus·
    bonus 在 _cap_sp 外加·held-out cue 词(collide=0,sub=0)=CORR_BONUS=1001 严格胜 collide=1 词(1000)·
    collide=2+(2000+) 仍胜（守共现主轴）。cue-slot-only + source 纯净 + specificity gate 守反 theater。
    gate CORRESPONDENCE_SLOT_MODE caller 守（_corr_gate）·OFF 不调。
    """
    if not is_cue_slot or unit_rel_kind == 0:
        return 0
    return CORR_BONUS if graph.cue_rel_of(c) == unit_rel_kind else 0


def _render_sequence(seq: list[ConceptRef], graph: ConceptGraph) -> str:
    """记忆序列 → 词形串（序列上下文已固定词形免逐节点消歧·回放直出）。

    序列元素经 graph.surface_of 解词形·None 走 ref 字面（诚实占位）。
    序列为空 → 空串（caller 不应进入回放分支·防御）。
    """
    parts: list[str] = []
    for ref in seq:
        assert_int(ref[0], ref[1], _where="_render_sequence.ref")
        s = graph.surface_of(ref)
        parts.append(s if s is not None else f"#{ref[0]}:{ref[1]}")
    return "".join(parts)


def _stable_tiebreak(scored: list[tuple[ConceptRef, int]]) -> ConceptRef:
    """确定性选最佳（-score 升序=高分先·ref 自然序 tiebreak·bit-identical）。"""
    best = min(scored, key=lambda x: (-x[1], x[0]))
    return best[0]


def modality_serialize(ref: ConceptRef, role: int, target_lang: int,
                       graph: ConceptGraph) -> str:
    """模态序化：concept → 词形串（语言 role→词序·代码 role→构造·答案 sink 槽→填之）。

    首版语言模态：surface_of 解词形·沿 role 序化（连接词插补 defer·M11 非 anchored）。
    代码模态序化（OPERATION→算子/OBJECT→操作数·复用 symbol_domain）defer Stage 6 接线。
    答案 sink 槽：ref 即 sink 末端节点·surface_of 填之。
    target_lang 仅作偏好已在上游分派过滤·此处序化不再判 lang。
    """
    assert_int(role, target_lang, _where="modality_serialize")
    s = graph.surface_of(ref)
    return s if s is not None else f"#{ref[0]}:{ref[1]}"


def dispatch_slot(slot: RoleSlot, dag_path: PathResult, graph: ConceptGraph,
                  workmem: Any, target_lang: int) -> tuple[str, int]:
    """逐槽分派。返 (word, source∈{LINEAGE_CONCEPT_FILL, LINEAGE_DEF_REPLAY})。

    workmem.ctx：多维 context_tag（F8）·collide_score 上下文来源（prior_topic_refs + produced_refs）。
    target_lang：LangMarker（LANG_ZH/EN·C1 硬偏好·LANG_NONE=非语言模态无偏好）。
    """
    assert_int(target_lang, _where="dispatch_slot.target_lang")
    # —— 记忆序列槽 → 回放直出（逐槽原语·辅） ——
    if slot.filler_is_memory_sequence:
        seq = graph.read_memory_sequence(slot.ref)
        if seq:
            return _render_sequence(seq, graph), LINEAGE_DEF_REPLAY
        # 序列空→退化填槽（防御·不应发生·caller 标 filler 时保证序列在）

    # —— 单概念 → 词形路径（填槽主） ——
    # cue 位有两级行为：无已学习 relation cue 时保持骨架原 cue 直出；有候选时把原 cue 作为 fallback
    # 与 PRIMARY+BARE_TEXT D:11 cue 同池竞争。这样保留结构活化，同时让学习结果真正参与 winner。
    _is_cue = getattr(workmem, "current_slot_is_cue", False)
    _corr_gate = getattr(gates, "CORRESPONDENCE_SLOT_MODE", False)
    _unit_rk = getattr(workmem, "current_rel_kind", 0)
    _cue_fill_gate = getattr(gates, "CUE_SLOT_FILL_MODE", False)
    _learned_cues = (graph.relation_cue_candidates(_unit_rk, space_id=slot.ref[0])
                     if _is_cue and _corr_gate and _unit_rk != 0 else [])
    _cue_ref: ConceptRef | None = None
    if _is_cue and _cue_fill_gate:
        _cue_sig = getattr(workmem, "current_cue_sig", ())
        _slot_idx = getattr(workmem, "current_slot_idx", 0)
        if _slot_idx < len(_cue_sig) and _cue_sig[_slot_idx] is not None:
            _cue_ref = _cue_sig[_slot_idx]
            _word = graph.surface_of(_cue_ref)
            if _word is not None and not _learned_cues:
                return _word, CUE_SLOT_FILL
    candidates = graph.activate_candidates(slot.ref)
    _seen = set(candidates)
    if _cue_ref is not None and _cue_ref not in _seen:
        candidates.append(_cue_ref)
        _seen.add(_cue_ref)
    for _learned in _learned_cues:
        if _learned not in _seen:
            candidates.append(_learned)
            _seen.add(_learned)
    _cue_lineage_candidates = (
        ({_cue_ref} if _cue_ref is not None else set()) | set(_learned_cues)
        if _cue_fill_gate and _is_cue else set()
    )
    # 维度桥 reader 移至 generate.py（读 unit=struct_ref·非 slot.ref·DISPATCH_TOKEN_CHAIN_MODE ON 时 slot.ref=token
    # 无 binding·审1 MEDIUM-1 修·slot_dispatch 只收 slot.ref 无 unit）。workmem.last_dim_skeleton 由 generate 每 unit 设。
    # P2 断桥 consumer 将在此读 workmem.last_dim_skeleton 做值填充（VALUE_TRANSIT_MODE defer·P1 无消费者·本函数不读）。
    # STEP5 PR4：EDGE_SIMILAR slot-filler 候选扩展（gate SIMILAR_SLOT_MODE·D2 合规非向量·
    # X 像 Y → X 可填 Y 的 slot·双向 similar_candidates·去重·hub filter 前扩·gate OFF 不扩 bit-identical）。
    if getattr(gates, "SIMILAR_SLOT_MODE", False):
        _seen = set(candidates)
        for c in list(candidates):
            for s in graph.similar_candidates(c):
                if s not in _seen:
                    candidates.append(s)
                    _seen.add(s)
    # collide_score 上下文（workmem ctx refs·prior_topic_refs+produced_refs·F8）
    ctx_refs = list(getattr(workmem, "prior_topic_refs", [])) + \
        list(getattr(workmem, "produced_refs", []))
    # 归一化半 A：hub 排除（read-time·gate 守·doc/重来_归一化与功能词排除_设计_2026-07-08.md 决断 A3）。
    # caller 侧排除（非返 0·返 0 时 _stable_tiebreak 仍选 ref 最小 hub + sel_pref 维浮起 combine>0）：
    # candidates 过滤（hub 不进 scored 池）+ ctx_refs 过滤（hub 作上下文是非判别性噪声·移除增判别力）。
    # 空 fallback 守：全 candidates 是 hub（如 slot.ref 自身 hub 且 activate 返 [self]）时保原 candidates
    # （避 _stable_tiebreak([]) crash·degenerate 槽退原 unfiltered·stable≠correct·gate OFF 零行为变）。
    if getattr(gates, "EXCLUDE_FUNCTION_MODE", False) and not _cue_lineage_candidates:
        # perf round5（2026-07-13）：per-ref is_hub（K+M 次 × 2 query）-> 一次 compute_hub_set
        # + membership O(1)（解生产 EXCLUDE_FUNCTION_MODE ON 36 万调 fan-out·cProfile 15.4s）。
        # bit-identical：hub_set 与 is_hub 同 theta + 同 strength 累加 -> `c in hub_set` == is_hub(c)。
        _hub_set = graph.hub_set()
        cand_f = [c for c in candidates if c not in _hub_set]
        if cand_f:
            candidates = cand_f
        ctx_refs = [r for r in ctx_refs if r not in _hub_set]
    # 命门③ 候选 C（slot_lca 抽象约束·doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）：内容词位按 slot IS_A LCA 类过滤候选。
    # mirror hub filter :207-215 范式（filter + 空集 fallback 走 collide·非 bonus·design 决策·"抽象约束"语义纯·无须 cap）。
    # 生产 cue 候选竞争绕过内容词 LCA；CUE_SLOT_FILL_MODE OFF 的实验组合仍保留旧过滤行为。
    # current_slot_lca=() 跳过（无约束位/gate OFF/getattr 默认守·bit-identical）。
    if getattr(gates, "SLOT_LCA_CONSTRAINT_MODE", False) and not _cue_lineage_candidates:
        _slot_lca = getattr(workmem, "current_slot_lca", ())   # getattr 默认守（mirror B 审1 LOW-1·current_slot_lca default ()）
        if _slot_lca != ():    # ()=无约束位->跳过
            cand_f = [c for c in candidates if graph.is_a_descendant_of(c, _slot_lca)]
            if cand_f:    # 空集 fallback 走 collide（未见过该类 token 退化·mirror hub filter `if cand_f:` :213）
                candidates = cand_f
    scored = [(c, graph.collide_score(c, ctx_refs)) for c in candidates]

    # S4 决断 2 两层正交第二腿·生成侧 selection_pref pair-rate 精查（CLASS 级共现 boost）：
    # collide_score 是 token 级（已算 s）·gate ON 时复用 s + sel_pref 亚主轴（cap 999·守 collide 主轴）。
    # combine = s×SCORE_SCALE + _cap_sp(graph.selection_pref_score(c, ctx_refs))·
    # **gate OFF 时整块跳过**（scored 不重算·if 外短路·守 984 测 bit-identical·零 IO·ancestor_map 不 build）。
    # 第4路 selection_pref（gate GENERATE_SELECTION_PREF_MODE）+ 第5路 pronoun（gate PRONOUN_SLOT_MODE）联合 cap 999·守 collide 主轴。
    # 审计根治 [严重-3]：pronoun 加成并入 sp 维联合 _cap_sp·sp+pr 联合 cap 999（collide 主轴优先 1000>999 不变）。
    # 对抗审 catch 修：pronoun 加成用 _pronoun_bonus(graph, c, ctx_refs)（ctx_refs=token concept ref·对偶 observe 侧）·
    # 非 slot.ref（struct_ref·pair-key 错位 theater）。
    # gate OFF 退化（两 gate OFF 不进 if·既有 collide only·bit-identical）。
    _sp_gate = gates.GENERATE_SELECTION_PREF_MODE
    _pr_gate = getattr(gates, "PRONOUN_SLOT_MODE", False)
    _mod_gate = getattr(gates, "MODIFIER_DIRECTION_MODE", False)   # G2 修饰方向A·head 偏好第 6 路
    _so_gate = getattr(gates, "SIMILAR_SECOND_ORDER_MODE", False)   # Phase C 二阶相似·第 7 路 tiebreak（read-side Jaccard）
    if _sp_gate or _pr_gate or _mod_gate or _so_gate or _corr_gate:
        # 对应桥 per-unit/per-slot stash（generate CORRESPONDENCE_SLOT_MODE 块设·getattr default 守 gate OFF）：
        # _unit_rk=skeleton REALIZES R（两跳 reader）·_is_cue=该 slot 是否 cue 位（cue-slot-aware 反 theater）。
        scored = [(c, s * SCORE_SCALE
                   + _cap_sp(
                       (graph.selection_pref_score(c, ctx_refs) if _sp_gate else 0)
                       + (_pronoun_bonus(graph, c, ctx_refs) if _pr_gate else 0)
                       + (graph.head_pref_score(c) if _mod_gate else 0)
                       + (_second_order_bonus(graph, c, ctx_refs) if _so_gate else 0)
                   )
                   + (_correspondence_bonus(graph, c, _unit_rk, _is_cue) if _corr_gate else 0)  # ★ (β) 独立轴·不入 _cap_sp·cue-slot-aware
                  ) for c, s in scored]

    # target_lang 硬偏好（§十四C1·防中文答 apple 泄数学）
    if target_lang != LANG_NONE:
        same_lang = [(c, s) for c, s in scored
                     if graph.lang_of(c) == target_lang]
        pool = same_lang if same_lang else scored   # 无同 lang 才回退跨 lang
    else:
        pool = scored   # 非语言模态无 lang 偏好

    best = _stable_tiebreak(pool)
    word = modality_serialize(best, slot.role, target_lang, graph)
    source = CUE_SLOT_FILL if best in _cue_lineage_candidates else LINEAGE_CONCEPT_FILL
    return word, source
