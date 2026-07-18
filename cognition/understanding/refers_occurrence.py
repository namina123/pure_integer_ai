"""cognition.understanding.refers_occurrence — 模块5 REFERS_TO 性质B 语篇指代（occurrence token）。

性质B = occurrence-bound 语篇指代（他→小明本篇）→ 记忆空间 occurrence token·带时序衰减·不进纯同指闭包。
  - 落**记忆空间**（非核心稳定边·§十一#2-bis 性质B·跨语篇失效·不冒充定义）
  - occurrence token = 记忆空间 REFERS_TO 性质B edge（subtype=OCCURRENCE·memory_time_attach=timestamp_seq）
  - scope 跨句 partial：层2 WorkMemory carry window N=3 FIFO（§十四J4 低②）
  - 层3 记忆衰减回看（#733·§十四:1291）：超 N=3 FIFO 窗口读 pronoun_ref 的 OCCURRENCE 出边历史先行词·
    score=effective_weight 衰减·扩候选覆盖范围非语义消解·stable≠correct·#479 墙
  - 代词特征软兜：人称/数/生命性/性别 作 PROPERTY 边进 PR 种子 e（软兜·硬过滤 defer）
  - 悬空（无候选 / score<θ）→ None → _segment_dangling++ → observe 段末标 struct_ref → J4=0 真碎句（② fix·#733）

无墙钟：time_attach 用 timestamp_seq（audit_event 自增序·§十三A3·非墙钟·非 order_index）。
occurrence-bound：跨语篇失效（他这篇指小明换篇指李华）·不冒充定义。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.config import gates
from pure_integer_ai.storage.edge_store import EdgeStore, SUBTYPE_OCCURRENCE
from pure_integer_ai.storage.node_store import TIER_SHADOW, TIER_PRIMARY
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.pronoun_resolution_count import (
    record_pronoun_resolution_decision, record_pronoun_resolution_dangling,
    read_pronoun_resolution_count,
)
from pure_integer_ai.cognition.shared.edge_types import EDGE_REFERS_TO, EDGE_PROPERTY
from pure_integer_ai.cognition.shared.hub_detect import compute_hub_set
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory, DEFAULT_PRONOUN_WINDOW
from pure_integer_ai.cognition.process.effective_weight import effective_weight   # #733 layer 3·OCCURRENCE 衰减复用（同层 cognition L5·import_direction 允许·无循环）

# 代词特征软兜（元定义出厂硬件·非写死语义规则）：人称/数/生命性/性别
# 首版：pronoun_features 注入（lookup_pronoun_features·元定义枚举）·默认 None。
THETA_PRONOUN_NUM = 1   # θ oracle 标定起点（§十五 B 组初值·oracle 验后调）
THETA_PRONOUN_DEN = 4

# B6 方案3 tn路 consumer 候选分加成上限（PR_TN_BONUS_CAP·recency 尺度·oracle 标定起点）。
# resolve_pronoun_occurrence gate PRONOUN_RESOLVE_COUNT_MODE ON 时读历史 pr_tn(pronoun, candidate)·
# score += min(pr_tn, PR_TN_BONUS_CAP)（recency 1..3 尺度·不颠覆 layer3 effective_weight 1000·保守加成）。
# 冷启动（无历史 pr_tn=0）→ 加成 0 → 候选排序 bit-identical。reward>0 鲁棒（J4 bool veto 只查 dangling）。
PR_TN_BONUS_CAP = 3

# OCCURRENCE 边初始记忆强度（#733 layer 3·oracle 标定起点）。
# effective_weight OCCURRENCE 衰减 = max(0, strength×DECAY_K − logical_age)·strength=本常量·
# 记忆窗口 = OCCURRENCE_STRENGTH seq 步（~100 segments·长文档）·超窗 score=0 → 候选不入 → 悬空（honest forgetting）。
# 旧值=best_score（recency 1..3）·衰减窗口 ~3 seq 步短于 N=3 segments·layer 3 超 N=3 时早归零无用·改常量。
# PR 不读（memory_active=False·a3_pr_wrapper.py:115·生产恒 False）·改 strength 零当前行为变 bit-identical·
# 未来 memory_active=True 时 OCCURRENCE 进 PR 邻接·strength=1000 量级合理（PRECEDES=1/CAUSES≤1000）·oracle 校准。
OCCURRENCE_STRENGTH = 1000


def resolve_pronoun_occurrence(edge_store: EdgeStore, concept_index: ConceptIndex,
                               pronoun_surface: str,
                               *, work_memory: WorkMemory,
                               memory_space_id: int,
                               timestamp_seq: int,
                               pronoun_features: int | None = None,
                               theta_num: int = THETA_PRONOUN_NUM,
                               theta_den: int = THETA_PRONOUN_DEN,
                               backend: StorageBackend | None = None) -> tuple[int, int] | None:
    """性质B pronoun 解析·落记忆 occurrence token。

    返回先行词 ConceptRef·悬空返 None（J4=0 真碎句）。
    memory_space_id：性质B 必落记忆（非核心）。
    timestamp_seq：audit_event 自增序（唯一时间源·无墙钟）。
    backend：B6 方案3 tn+fn 路 pronoun_resolution_count 读写（gate PRONOUN_RESOLVE_COUNT_MODE 守·
      None=bare fixture skip·向后兼容）。observe 扩展（§九.2 病灶"attribute 给谁"=per-occurrence 落 pronoun）：
      决策时写 pr_tn（record_pronoun_resolution_decision·选 best antecedent·sign-agnostic·per-occurrence·避 β_arith）/
      悬空时写 pr_fn（record_pronoun_resolution_dangling·self-loop (pronoun,pronoun)·零教师·per-occurrence）。
      consumer 自消费（gate ON·读历史 pr_tn 加候选分·B0 dim consumer=refers_occurrence·reward>0 鲁棒·J4 bool veto 只查 dangling）。
    """
    # 代词概念点（性质B 落记忆·occurrence token）·layer 3 候选扩源需 pronoun_ref 先建
    pronoun_ref = concept_index.ensure(pronoun_surface, space_id=memory_space_id,
                                       tier=TIER_SHADOW)
    # 候选生成：层1 同段前序 token（factor E·gate 守）+ 层2 FIFO（prior_segments N=3·score=recency）+
    #          层3 OCCURRENCE 边（#733·超 N=3 记忆衰减回看）
    candidates: list[tuple[tuple[int, int], int]] = []  # (ref, score)
    # 归一化半 A：hub 不当代词先行词（gate 守·解"他"→"曾经"语义层污染·审1 Q6 P1·决断 A3）。
    # **perf**（2026-07-08 cProfile 实测）：per-candidate is_hub 每 ref 2 全表扫·pronoun_ref OCCURRENCE 边
    # 跨 round append-only 累积 → layer3 循环膨胀 × 每候选 2 全表扫 = 7194×2 = 276M 行 = 218s（n=5 占 65%）。
    # 改 compute_hub_set **单遍** O(#COOCCURS) 建 degree map·每 resolve 调用 1 次·set 查表 O(1)·根治。
    # gate OFF → hub_set 空 set → `in` 永假 → 不过滤 → bit-identical 现状。
    hub_set = (compute_hub_set(edge_store)
               if getattr(gates, "EXCLUDE_FUNCTION_MODE", False) else set())
    # 层1（factor E·2026-07-09·doc/重来_factorE_层1指代_intra_seg_设计）：同段前序 token（同段前指·"动物...它们"
    # 同段）。judge.py:58 注释声称"层1 单句指代已解析"是 theater——#733 只实施层3+② fix·层1 候选生成从未实现。
    # work_memory._current_segment_refs = pronoun 前已 normalize 的 token ref（pronoun 未入·observe append 在
    # normalize 返回后）。score=k+1（近因·最近先行词 k=n-1 分最高=n·最远 k=0 分=1）。
    # **factor F（hub 交互·实测发现）**：层1 用硬 hub 过滤（同层2/3）→ 主题集中语料（动物/生物/类群 跨段重复）
    # 内容词 COOCCURS degree≥8 → 全归类 hub → 层1 零候选 → dangling → reward=0（hub_detect 已标"小语料 θ 误排
    # 主题中心实词"）。改**软 hub 过滤**：非 hub 优先（正常语料 它→动物·正确）·全 hub 时 fallback 收全（集中语料
    # 它→最近 token·解析胜悬空·trainability > dangling·stable≠correct·它→功能词可能·特征过滤 defer）。
    # gate PRONOUN_INTRASEG_MODE OFF → 跳过 → 候选源仅层2+层3 = current bit-identical。
    if getattr(gates, "PRONOUN_INTRASEG_MODE", False):
        cur_refs = getattr(work_memory, "_current_segment_refs", [])
        _l1_pref: list[tuple[tuple[int, int], int]] = []    # 非 hub（优先·正常语料 它→内容词）
        _l1_fb: list[tuple[tuple[int, int], int]] = []      # hub（全 hub 时 fallback·集中语料 它→最近）
        for _k, _ref in enumerate(cur_refs):
            if _ref in hub_set:
                _l1_fb.append((_ref, _k + 1))
            else:
                _l1_pref.append((_ref, _k + 1))
        candidates.extend(_l1_pref if _l1_pref else _l1_fb)   # 非 hub 优先·空则 fallback 全 hub
    # 层2：沿 WorkMemory prior_segments 回溯前文（跨句 scope N=3 FIFO）
    for seg_seq, refs in work_memory.prior_segments(window=DEFAULT_PRONOUN_WINDOW):
        recency = work_memory.recency_weight(seg_seq)
        for ref in refs:
            if ref in hub_set:
                continue   # hub 不代近因先行词（gate OFF→空集→不过滤）
            # score = recency（近因·纯整数）+ pronoun_features 软兜（PROPERTY 边·卷二读）
            score = recency
            candidates.append((ref, score))
    # 层3（#733·§十四:1291 记忆衰减回看扩候选覆盖范围）：读 pronoun_ref 的 OCCURRENCE REFERS_TO 出边
    # （历史解析先行词·超 N=3 FIFO 窗口）。pronoun_ref 同 surface+space 去重（concept_index.ensure）·
    # 所有历史"他"解析共享一 pronoun_ref·其出边=全部历史先行词。score=effective_weight 复用（同 PR matrix
    # OCCURRENCE 衰减式 max(0, OCCURRENCE_STRENGTH×DECAY_K − logical_age)）·超窗 score=0 不入（honest forgetting）。
    # 诚实边界：扩候选覆盖范围非语义消解·stable≠correct·antecedent quality ≤ prior resolution·#479 墙。
    for e in edge_store.query_from(pronoun_ref[0], pronoun_ref[1],
                                   edge_type=EDGE_REFERS_TO):
        if e.get("subtype") != SUBTYPE_OCCURRENCE:
            continue   # 仅性质B occurrence·非 PURE_ALIAS/METAPHOR（闭包纯净性·S1）
        if (e["space_id_to"], e["local_id_to"]) in hub_set:
            continue   # hub 不代历史先行词（解"他"→"曾经"语义层污染）
        score = effective_weight(e, current_seq=timestamp_seq)
        if score <= 0:
            continue   # 衰减归零·honest forgetting·不入候选
        candidates.append(((e["space_id_to"], e["local_id_to"]), score))
    # 代词特征软兜（PROPERTY 边进 PR 种子 e·防 PR 软排序把"他"指向"苹果"）
    if pronoun_features is not None:
        feat_ref = concept_index.ensure(pronoun_features, space_id=memory_space_id,
                                        tier=TIER_PRIMARY)
        edge_store.add(
            space_id_from=pronoun_ref[0], local_id_from=pronoun_ref[1],
            space_id_to=feat_ref[0], local_id_to=feat_ref[1],
            edge_type=EDGE_PROPERTY, strength=1,
            source=6,  # SOURCE_DERIVED
            epistemic_origin=None, tier=TIER_PRIMARY,
        )
    # 候选排序（近因优先·确定性 tiebreak 按 ref·layer 3 recent 高分 dominate FIFO·设计故意：
    # OCCURRENCE 边是已记录解析·比 FIFO 通用近因 ref 更可靠·代词 anaphora 跨句倾向同先行词）
    if not candidates:
        work_memory._segment_dangling += 1   # ② fix·悬空记·observe 段末标 struct_ref
        # B6 方案3 fn路（失败侧归因·零教师·per-occurrence·§九.2 病灶"attribute 给谁"=per-occurrence 落 pronoun）：
        # 悬空（无候选）→ pr_fn++ self-loop (pronoun,pronoun)·独立 episode 符号·避 β_arith。
        if backend is not None and getattr(gates, "PRONOUN_RESOLVE_COUNT_MODE", False):
            record_pronoun_resolution_dangling(backend, pronoun_ref=pronoun_ref)
        return None
    # B6 方案3 tn路 consumer（gate PRONOUN_RESOLVE_COUNT_MODE·observe 侧自消费·B0 dim consumer=refers_occurrence）：
    # 读历史 pr_tn(pronoun, candidate)·score += min(pr_tn, PR_TN_BONUS_CAP)（recency 尺度加成·不颠覆 layer3 effective_weight 1000）。
    # 冷启动（无历史 pr_tn=0）→ 加成 0 → 候选排序 bit-identical。reward>0 鲁棒（J4 bool veto 只查 dangling 不查 antecedent 质量·
    #   选哪个非悬空 antecedent 都 reward>0）。件 C dispatch_slot pronoun scoring defer STEP6（§十二.5 主面件 C slot 级·生成侧）。
    if backend is not None and getattr(gates, "PRONOUN_RESOLVE_COUNT_MODE", False):
        _adj: list[tuple[tuple[int, int], int]] = []
        for _cand_ref, _cand_score in candidates:
            _row = read_pronoun_resolution_count(backend, pronoun_ref, _cand_ref)
            _pr_tn = _row[1] if _row is not None else 0
            _adj.append((_cand_ref, _cand_score + min(_pr_tn, PR_TN_BONUS_CAP)))
        candidates = _adj
    candidates.sort(key=lambda x: (-x[1], x[0]))
    best_ref, best_score = candidates[0]
    # θ 悬空判定（oracle 标定·起点 1/4）·cross_compare 零误差
    # #733 审2 P2-2：θ=1/4 当前等价 disabled——best_score 整数·FIFO recency max(1,..)≥1·layer 3 score>0 入候选·
    # cross_ge(best_score,1,1,4)=best_score*4≥1·整数 best_score≥1 恒 True·此分支永不执行。待 oracle 调 θ>1 才生效。
    # B6 注：best_score 含 pr_tn 加成（gate ON）·θ disabled 不受影响·oracle 调 θ>1 时加成使高 pr_tn antecedent 更不易悬空（设计正确）。
    from pure_integer_ai.crosscut.integer import compare as cmp
    if not cmp.cross_ge(best_score, 1, theta_num, theta_den):
        work_memory._segment_dangling += 1   # ② fix·θ 不达悬空记
        # B6 方案3 fn路（θ 不达悬空归因·零教师·per-occurrence）：
        if backend is not None and getattr(gates, "PRONOUN_RESOLVE_COUNT_MODE", False):
            record_pronoun_resolution_dangling(backend, pronoun_ref=pronoun_ref)
        return None   # 悬空 → J4=0 真碎句
    # 时序衰减权重 = OCCURRENCE_STRENGTH（#733·recency 1..3 衰减窗口过激改常量·见模块常量注释）
    # effective_weight 检索 max(0, OCCURRENCE_STRENGTH×DECAY_K − logical_age)·layer 3 候选 score 复用此式。
    decay_weight = OCCURRENCE_STRENGTH
    edge_store.add(
        space_id_from=pronoun_ref[0], local_id_from=pronoun_ref[1],
        space_id_to=best_ref[0], local_id_to=best_ref[1],
        edge_type=EDGE_REFERS_TO, subtype=SUBTYPE_OCCURRENCE,
        strength=decay_weight, source=6,  # SOURCE_DERIVED
        epistemic_origin=None,
        memory_time_attach=timestamp_seq,
        order_index=None, role=None, tier=TIER_SHADOW,  # 性质B 不进默认 A1·经 effective_weight 第三分支 F3 进 PR
    )
    # B6 方案3 tn路（observe·决策时·sign-agnostic·per-occurrence·避 β_arith）：
    # 选 best antecedent → pr_tn++ (pronoun, best_ref)·独立 episode 符号·rate=pr_fn/pr_tn 不塞 episode 标量。
    if backend is not None and getattr(gates, "PRONOUN_RESOLVE_COUNT_MODE", False):
        record_pronoun_resolution_decision(backend, pronoun_ref=pronoun_ref,
                                           antecedent_ref=best_ref)
    return best_ref
