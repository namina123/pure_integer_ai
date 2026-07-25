"""cognition.result.graph_view — 卷三读图统一接口（§十四路径填槽/judge 查图）。

ConceptGraph：卷三经此抽象访问持久图·不直接碰 backend schema（守最少耦合·复用既有
edge_store/node_store 读路径）。卷三模块1/2/3 读 role_seq/activate_candidates/
memory_sequence/collide_score 全经此接口。

  read_role_seq(struct_ref) -> list[int]
      role_seq 作结构概念点 def_array 有序属性（卷一模块2 attach_role_seq·
      ref_space_id==0 标记 role 标记非概念 ref·消费方按此判）。
  activate_candidates(ref) -> list[ConceptRef]
      REFERS_TO 反向：concept → 候选词集（query_to EDGE_REFERS_TO 取 from 端·返全部禁取首·§H5/§B12）。
      无候选 → [ref] 自身（概念点本身可作词形·OOV/单挂 fallback）。
  read_memory_sequence(ref) -> list[ConceptRef]
      记忆空间种类3 序列节点（def_array ref_space_id!=0 的概念 ref 序·回放直出）。
      无序列 → []（caller 判 filler_is_memory_sequence=False 走填槽主路）。
  collide_score(c, ctx_refs) -> int
      上下文扩散纯整数共现分（读 Σ strength 累加·总收口 0.1 COOCCURS A'·SHADOW 隔离不影响·§十四消歧=拓扑共现非语义接地）。
      = c 与 ctx_refs 共现 COOCCURS 边的 strength 总和（频次计分·可 >n·gate OFF strength 恒 1=边数·gate ON strength=频次·两态同值·tiebreak 用 ref 序）。
  surface_of(ref) -> str | None
      概念 ref → 词形字符串（surface 文本入伴随库·卷三读图经此解析·
      首版注入式 surface_of callable·默认 None 走 ref 字面·生产接线随伴随库反查）。

铁律：纯整数（role/共现计数全整）/ 确定性（def_array order_index 序·ref 自然序 tiebreak）。
诚实边界：消歧=拓扑共现非语义接地（共现选词≠语义理解·§十四诚实边界·#479 W2 truth 墙不掩盖·**非 D 物理接地**）。

**刀6 片4 sense 选优**：sense_candidates gate ON 时·formal_train `_discover_and_recognize_lang_structures`
  clone aligning_root 逐 sense 候选试骨架对齐·多 sense concept_ref 进 recognize 候选池·collide_score 共现 +
  ATTR_SLOT_ROLE IS_A 共祖选优 = sense 消歧的结构机制（**非语义接地·#479 墙·stable≠correct·共现也无法区分时
  撞墙·同 selection_pref_count 范式**）。
"""
from __future__ import annotations

from typing import Any, Callable

from pure_integer_ai.config import gates
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SUBTYPE_PURE_ALIAS, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import NODE_WORD, TIER_PRIMARY
from pure_integer_ai.storage.edge_types import (
    EDGE_REFERS_TO, EDGE_COOCCURS, EDGE_COMPOSES, EDGE_SIMILAR, EDGE_INSTANTIATES,
    EDGE_REALIZES, EDGE_RELATION_SIGNAL,
)
from pure_integer_ai.storage.composes_attr import (
    read_composes_attrs, ATTR_OPERATOR, ATTR_CTRL_TAG, ATTR_OPERAND,
    ATTR_IMMEDIATE, ATTR_STORE_TARGET, ATTR_PROPOSITION, COMPOSES_ATTR_TABLE,
    ATTR_PROP_SUBJ, ATTR_PROP_ATTR, ATTR_PROP_POLMOD,
    ATTR_RELATION_PRIMITIVE, ATTR_CUE_SIG, ATTR_SLOT_ROLE,
)
from pure_integer_ai.storage.chapter_seq import CHAPTER_SEQ_TABLE
from pure_integer_ai.storage.selection_pref_count import read_selection_pref_count
from pure_integer_ai.storage.pronoun_resolution_count import read_pronoun_resolution_count
from pure_integer_ai.storage.abstract_mark import get_mark, MARK_LANG
from pure_integer_ai.storage.telemetry import record_diagnostic_event
from pure_integer_ai.storage.concept_correspondence import (
    load_correspondence as _load_correspondence, CORR_ORDINAL,
)
from pure_integer_ai.crosscut.integer.unicode_codec import decode as _decode_codepoints
from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map, nearest_isa_ancestor
from pure_integer_ai.cognition.shared.hub_detect import (
    HubDegreeState,
    compute_hub_set as _shared_compute_hub_set,
    is_hub as _shared_is_hub,
)
from pure_integer_ai.cognition.understanding.modification_direction import head_preference, HEAD_PREF_CAP
from pure_integer_ai.cognition.shared.types import ConceptRef

# Phase C §十五-bis：二阶相似 Jaccard 缩放（mirror slot_dispatch.SCORE_SCALE=1000 / a4_align.COVERAGE_SCALE=1000·
# 去环 dedup·graph_view→slot_dispatch 循环 import 故本模块内定义·codebase 既有 dedup 先例 SCORE_SCALE vs COVERAGE_SCALE）。
_SIM_SCALE = 1000

SurfaceResolver = Callable[[ConceptRef], "str | None"]
LangResolver = Callable[[ConceptRef], "int | None"]

_LANG_MISS = object()   # perf round7：lang_of per-ref cache sentinel（区分"未算"与"算得 None"）
_COUNT_MISS = object()   # perf round9：pair-read cache sentinel（区分"未算"与"算得 None"·冷启动 pair 多）
_SURFACE_MISS = object()   # P0a：surface_of per-ref cache sentinel（区分"未算"与"算得 None"·gate ON 才启用）


def ordinal_surface_of(backend: StorageBackend, ref: ConceptRef) -> str | None:
    """概念 ref → ordinal 码点对应 → 文本（surface_of resolver·P0a·读 concept_correspondence）。

    读 concept_correspondence WHERE (sid,lid,kind=CORR_ORDINAL) ORDER BY order_index → decode → 文本。
    **无行 → None（绝不返 ""·承重不变量）**：judge.J2s slot_fill_rate（:99-105）`for w: if w:` 读
    truthiness·"" falsy 会让 bound 不增→reward 变→破 bit-identical。故空对应返 None（caller 退 #ref 占位）·
    非 decode(())→""。表未注册（bare fixture）→ KeyError → None（向后兼容）。
    """
    sid, lid = ref
    codepoints = _load_correspondence(backend, space_id=sid, local_id=lid,
                                      corr_kind=CORR_ORDINAL)
    if not codepoints:
        return None   # 无对应（int surface / 未写 / 表空）→ None（**非 ""**·承重不变量）
    return _decode_codepoints(codepoints)


class ConceptGraph:
    """卷三读图统一接口（经 backend 抽象·卷三唯一图访问点）。"""

    def __init__(self, backend: StorageBackend, *,
                 surface_of: SurfaceResolver | None = None,
                 lang_of: LangResolver | None = None) -> None:
        self._b = backend
        self._edge_store = EdgeStore(backend)   # hub_detect 读 COOCCURS 用（归一化半 A·thin wrapper）
        self._hub_degree_state = HubDegreeState(self._edge_store)
        self._surface_of = surface_of
        self._lang_of = lang_of   # C1 跨语言偏好用·概念 lang 注入式（observe lang 标注·首版 defer）
        # selection_pref_score 的 ancestor_map lazy cache（per space_id·S4 决断 2 生成侧精查）。
        # **生产 ConceptGraph 是 ctx 单例**（make_train_context:211 一次建·存 ctx.concept_graph·跨 round
        # reuse）·observe 增 IS_A 边后 cache 单调陈旧（漏新增祖先→class_of 退 c 自身→under-boost 非乱 boost·
        # IS_A append-only 无删·错方向保守）。gate 默认 OFF 不触发·同 generation 内 IS_A 不变（生成不 observe）
        # → 单次 generation 内 cache 正确。ON 生产接线 caller 须 round 边界清 cache 或重建 ConceptGraph（defer 真修）。
        self._ancestor_map_cache: dict[int, dict] = {}
        # perf round5（2026-07-13）：生成侧只读函数 per-ref 缓存（read_role_seq/activate_candidates/
        # read_memory_sequence·cProfile n=4 read_role_seq 348K 调=每 generate 10875 次·大量同 struct_ref
        # 重复 select def_array/edge 全冗余——一代内 observe 已完·def_array/REFERS_TO 不变）。
        # 同 ancestor_map_cache 范式：生产 ctx 单例·**invalidate_ancestor_map 在 observe 后清**（formal_train:522
        # per-item·post-observe pre-generate）·一代内命中·跨 item 清。bit-identical：缓存的 sorted 列表 ==
        # fresh select 结果（def_array/REFERS_TO 一代内不变·observe 后已清）·纯读无写。
        self._role_seq_cache: dict[ConceptRef, list[int]] = {}
        self._activate_cache: dict[ConceptRef, list[ConceptRef]] = {}
        self._mem_seq_cache: dict[ConceptRef, list[ConceptRef]] = {}
        # P0 #1040：read_token_seq per-struct_ref cache。读 def_array ref_space_id!=0（attach_token_seq 写·observe
        # call-site gate ON）·一代内 generation 不 observe → 稳定 → per-item invalidate（invalidate_ancestor_map 清）够·
        # bit-identical（cache list == fresh select）。仅 gate DISPATCH_TOKEN_CHAIN_MODE ON 时 generate 调（OFF 不调）。
        self._token_seq_cache: dict[ConceptRef, list[ConceptRef]] = {}
        # perf round6（2026-07-13）：collide_score per-ref COOCCURS 邻接 cache。cProfile round5 定
        # collide 是生成侧 select 大头（n=8 805K 调 × 2 select = 1.6M·cand_distinct 仅 3 / (c,ctx)
        # distinct 仅 7·recurrence 26 万倍·_covering_candidates 7.1s self）→ per-ref 邻接 cache 杀 select
        # fan-out。**同 round5 read-cache 纯范式（显式 invalidate·无版本号自失效）**：对抗审证 edge_store
        # 多实例（ctx.edge_store / 本 _edge_store / ObservePipeline.edge_store 每次 observe 新建）→ COOCCURS
        # 写 bump 的是 ObservePipeline 自己的 ES·**本 _edge_store.cooccurs_version 永留 0** → 版本号自失效死
        # 代码（曾试加·对抗审 catch 删）。正确性全靠 invalidate_ancestor_map 显式 clear（formal_train:532 每
        # item post-observe pre-generate）+ 一代内 generation 不 observe（COOCCURS 稳定）。同 round5 read-cache
        # 依赖链。bit-identical：cache (other,strength) 序 == fresh 2 select 同源同序（from 先 to 后·保插入序
        # =原扫描序·整数累加同序·自环 c→c 双边计同）。对抗审 [[round6]] 全 SAFE·0-发散/121584。
        self._cooccur_nbr_cache: dict[ConceptRef, list[tuple[ConceptRef, int]]] = {}
        # perf round7（2026-07-13）：lang_of per-ref cache。cProfile round7 热区 ON 后 generate 68%·
        # lang_of（dispatch_slot:172 target_lang 过滤·每候选 1 次·730224 调 → get_mark → get_marks → select）
        # 是未 cache 的 select 大头之一。MARK_LANG 由 observe set_mark 写（word concept 语言标记·§7.7.1 片3）·
        # reward/judge 写 CAUSES strength 非 mark → **一代内 MARK_LANG 稳定**（同 _cooccur_nbr_cache COOCCURS
        # observe-only 论证）。**同 round5/6 read-cache 纯范式（显式 invalidate·无版本号自失效）**：对抗审证
        # edge_store 多实例 → 版本号自失效死代码·正确性全靠 invalidate_ancestor_map 显式 clear（per-item
        # post-observe pre-generate）+ 一代内 generation 不 observe（MARK_LANG 稳定）。bit-identical：cache 值
        # == fresh get_mark（MARK_LANG·status PROMOTED）·None 亦缓存（concept 无 lang mark·返 None）。
        self._lang_cache: dict[ConceptRef, int | None] = {}
        # P0a（2026-07-14）：surface_of per-ref cache。ordinal 码点对应由 observe ensure 写
        # （append-only·新增概念才写）·一代内 generation 不 observe → 对应稳定 → per-item invalidate
        # （invalidate_ancestor_map 清·formal_train:522 post-observe pre-generate）够·bit-identical。
        # **仅 gate ON 时启用**（surface_of 方法体内 live-read gate·gate OFF 直返 None 不 cache·防 try/finally
        # flip 后旧 None 命中）。cache 值 == fresh ordinal_surface_of（含 None·int 概念无对应）。
        self._surface_cache: dict[ConceptRef, str | None] = {}
        # perf round7（2026-07-13）：similar_candidates per-ref cache。SIMILAR_SLOT_MODE 生产 ON → dispatch_slot
        # 每 dispatch 调（730K 调 → out_edges+in_edges 各 ~730K select）·EDGE_SIMILAR observe-only（不接 reward）→
        # 同 _cooccur_nbr_cache / _lang_cache per-item invalidate 范式·bit-identical（cache sorted == fresh sorted）。
        self._similar_cache: dict[ConceptRef, list[ConceptRef]] = {}
        # Phase C §十五-bis：second_order_similarity per-pair 对称 cache（key=tuple(sorted([a,b]))·Jaccard 对称·
        # 同 round5 read-cache 范式：invalidate_ancestor_map 显式 clear（observe 增 COOCCURS 后清·保 fresh）+
        # generation 内不 observe（COOCCURS 稳定）·bit-identical（cache 值 == fresh Jaccard·strength-invariant 集成员）。
        self._second_order_cache: dict[tuple[ConceptRef, ConceptRef], int] = {}
        # perf round9（2026-07-14）：selection_pref_score / pronoun_score per-pair read cache。
        # cProfile n=60 实测 generate→dispatch_slot→selection_pref_score(8.6M)+_pronoun_bonus(8.6M)→
        # read_selection_pref_count/read_pronoun_resolution_count 各 1 select = **18.6M select（n=656 真墙·非 PR matrix·
        # 前 session 从 matrix.n 分布推断 PR 是墙系未计时·测前勿假设纠正）**。两 read 均纯函数 of (backend 状态, pair)·
        # memo 化返同值 = bit-identical by construction（同 round5/6/7 read-cache 纯范式·ungated always-on·CI gate OFF
        # 不调用 selection_pref_score/pronoun_score → 死代码 → 平凡 bit-identical）。
        # **失效粒度差异（核证写路径）**：
        #   selection_pref_count = observe(record_selection_pref_cooccur)+reward(record_selection_pref_reward) 两路写 →
        #     reward 在 episode 间写（propagate_reward·episode_loop generate 后）→ **per-generate clear**（generate_output
        #     入口 invalidate_generate_read_cache·见 generate.py）+ per-item clear（invalidate_ancestor_map·observe 后）。一代内
        #     表稳定 → cache 正确·跨 generate 清 → bit-identical。
        #   pronoun_resolution_count = observe-only 写（record_pronoun_resolution_decision·"reward 不调此函数"·reward 间不变）→
        #     per-item clear（invalidate_ancestor_map）够·可跨 generate 复用（更高效）。
        # None 亦缓存（冷启动 pair 多·_COUNT_MISS sentinel 区分"未算"与"算得 None"·镜像 _LANG_MISS）。
        self._sp_count_cache: dict[tuple, tuple | None] = {}
        self._pronoun_count_cache: dict[tuple, tuple | None] = {}
        # perf round10：G2 head_pref per-ref read cache（head_preference 读 modification_direction 表·observe-only 写
        # 同 pronoun 生命周期 → per-item invalidate_ancestor_map 清够·可跨 generate 复用）。gate MODIFIER_DIRECTION_MODE
        # OFF → head_pref_score 早返 0 不触 cache → CI bit-identical。0 亦缓存（_COUNT_MISS 区分"未算"与"算得 0"）。
        self._head_pref_cache: dict[ConceptRef, int] = {}
        # 对应泛化桥 cue_rel_of per-ref read cache（D:11 EDGE_RELATION_SIGNAL word→REL_*·doc/重来_对应泛化_readback_generation_桥）。
        # D:11 边 observe/tally/promote 写（generation 不写·审2 核证 cognition/result/ 零 D:11 写点）→ per-item
        # invalidate_ancestor_map 清够（同 _pronoun_count_cache observe-only 生命周期·可跨 generate 复用）。gate
        # CORRESPONDENCE_SLOT_MODE OFF → cue_rel_of 不被调（slot_dispatch _corr_gate=False）→ CI 不 cache·bit-identical。
        # 0 亦缓存（_COUNT_MISS 区分"未算"与"算得 0"·冷启动 v2-learned D:11 多为 0）。cache 值 == fresh query（D:11 一代内稳定）。
        self._cue_rel_cache: dict[ConceptRef, int] = {}
        # 生成 cue 候选反向索引：同 space 的 PRIMARY + BARE_TEXT D:11 按 rel_kind 分组。
        # D:11 只在训练/tally/promote 写，和 cue_rel_of 共用 per-item invalidate 生命周期。
        self._relation_cue_cache: dict[tuple[int, int], tuple[ConceptRef, ...]] = {}

    # ---- role_seq（卷一模块2 attach_role_seq def_array） ----

    def read_role_seq(self, struct_ref: ConceptRef) -> list[int]:
        """结构概念点的 role_seq（def_array ref_space_id==0 的 role 标记·order_index 序）。"""
        cached = self._role_seq_cache.get(struct_ref)
        if cached is None:
            sid, lid = struct_ref
            rows = self._b.select("def_array", where={
                "space_id": sid, "local_id": lid, "ref_space_id": 0,
            })
            rows = sorted(rows, key=lambda r: r["order_index"])
            cached = [r["ref_local_id"] for r in rows]
            self._role_seq_cache[struct_ref] = cached
        return list(cached)   # copy：caller（generate）可能 mutate·避污染 cache

    # ---- activate_candidates（REFERS_TO 反向·返全部禁取首） ----

    def activate_candidates(self, ref: ConceptRef) -> list[ConceptRef]:
        """concept → 候选词集（REFERS_TO 反向·query_to 取 from 端·返全部禁取首）。

        性质A 稳定同指 + 性质B occurrence 都返（消歧在生成侧·§十一#2-bis·H5）。
        无候选 → [ref]（单挂/OOV fallback·概念点本身可作词形）。
        **PURE_ALIAS-gated 自包含（P0b·跨语言桥·node_type 精准门控）**：rows 含 PURE_ALIAS 边且 ref 是
          NODE_WORD（词形·observed token·P0a 说话单元）→ 补 self。语义：从词形派发时（apple↔苹果 从 apple
          派发）·apple 自身亦是合法输出词形·须在候选内（否则 EN 目标只余 苹果·选不出 apple）。
          NODE_CONCEPT（抽象概念）**不补 self**——其 PURE_ALIAS 词形候选代表它（承重既有语义：test_stage5
          C / test_hub_exclude C / test_m5_e2e u 皆 concept+词形→concept·concept 自身 excluded·词形优先）。
          node_type 是干净判据：词形派发 include self / 抽象概念派发 exclude self。
          **bit-identical**：CI 零 PURE_ALIAS 边（lemmatizer dormant 无 caller 注入·无 alias_facts 文件）→
          elif 不进 → 退化 `if not cands` 现状（OCCURRENCE 代词边 subtype≠PURE_ALIAS 亦不进）。
        """
        cached = self._activate_cache.get(ref)
        if cached is None:
            sid, lid = ref
            rows = self._b.select("edge", where={
                "space_id_to": sid, "local_id_to": lid, "edge_type": EDGE_REFERS_TO,
            })
            cands = {(r["space_id_from"], r["local_id_from"]) for r in rows}
            if not cands:
                cands.add(ref)   # 无 REFERS_TO 候选→自身 fallback（单挂/OOV·词形=概念点本身）
            elif any(r.get("subtype") == SUBTYPE_PURE_ALIAS for r in rows) and \
                    self._ref_node_type(ref) == NODE_WORD:
                # PURE_ALIAS 别名在 + ref 是词形(NODE_WORD) → 补 self（dispatch_slot 可选 ref 自身 lang 词形）。
                # NODE_CONCEPT 不补（词形候选代表它·既有语义）。CI 零 PURE_ALIAS→elif 不进→逐字现状·bit-identical。
                cands.add(ref)
            cached = sorted(cands)   # 确定性 tiebreak（ref 自然序）
            self._activate_cache[ref] = cached
        return list(cached)   # copy：dispatch_slot SIMILAR 扩展 append mutate·避污染 cache

    def read_instantiates(self, ref: ConceptRef) -> ConceptRef | None:
        """结构一等化 reader（Phase A §十三-bis A.1·读 EDGE_INSTANTIATES 边 ref→skeleton_ref·DIM_BRIDGE_READ_MODE caller 门控）。

        返 skeleton_ref ConceptRef 或 None（无 INSTANTIATES 边）。纯读·无副作用·caller（generate·读 unit=struct_ref）
        gate 门控。observe 写侧（COMPOSES_COMBINE_MODE）建 EDGE_INSTANTIATES 真边（struct→skeleton·替 ATTR_SKELETON_BINDING
        注解·关联在图中）·本方法读回。确定性首边（min by (to_sid,to_lid)·INSTANTIATES 一 struct 一 skeleton·sorted 守 bit-identical）。
        """
        rows = self._b.select("edge", where={
            "space_id_from": ref[0], "local_id_from": ref[1], "edge_type": EDGE_INSTANTIATES,
        })
        if not rows:
            return None
        r = min(rows, key=lambda row: (row["space_id_to"], row["local_id_to"]))
        return (r["space_id_to"], r["local_id_to"])

    # ---- 对应泛化 readback→generation 桥（CORRESPONDENCE_SLOT_MODE·doc/重来_对应泛化_readback_generation_桥_2026-07-17） ----

    def rel_kind_of_skeleton(self, skeleton_ref: ConceptRef) -> int:
        """skeleton → EDGE_REALIZES → REL_* → ATTR_RELATION_PRIMITIVE int_a（对应桥两跳第二跳·审2 致命-1 修）。

        对应泛化 readback→generation 桥（CORRESPONDENCE_SLOT_MODE）。**两跳 reader 第二跳**：caller（generate.py）
        先 read_instantiates(unit) 得 skeleton_ref·再调本方法读 skeleton→REL_*。挂 skeleton 非 unit（unit=struct_ref 无
        outgoing REALIZES 边·instantiates.py 立法分离·审2 致命-1：v1 一跳 rel_kind_of(unit) 恒返 0→桥白建）。复用
        promote.py:239 同一 read 原语（read_composes_attrs(backend, target).get(ATTR_RELATION_PRIMITIVE,(0,0))[0]·自包含）。

        skeleton→REL_* **一对一或多**（REALIZES labeler 允许同 skeleton 命中 IS_A+CAUSES 两 oracle·罕见）·多时取
        min(by (to_sid,to_lid)) 定序（确定性无 hash 依赖·与 read_instantiates 同范式·审1 LOW-2：cue_rel_of 匹配另一 R→
        bonus=0 强退化·非 crash 非 theater）。无 REALIZES 边 → 0（非 R-skeleton·无 bonus·robust）。

        纯读·无副作用·caller gate 门控（generate CORRESPONDENCE_SLOT_MODE）。确定性（min int tuple·bit-identical）。
        """
        rows = self._b.select("edge", where={
            "space_id_from": skeleton_ref[0], "local_id_from": skeleton_ref[1],
            "edge_type": EDGE_REALIZES,
        })
        if not rows:
            return 0
        r = min(rows, key=lambda row: (row["space_id_to"], row["local_id_to"]))
        attrs = read_composes_attrs(self._b, (r["space_id_to"], r["local_id_to"]))
        return attrs.get(ATTR_RELATION_PRIMITIVE, (0, 0))[0]

    def read_cue_sig(self, skeleton_ref: ConceptRef) -> tuple[ConceptRef | None, ...]:
        """skeleton 子树 DFS 前序 → PARAM slot 序的 ATTR_CUE_SIG tuple（cue slot 重建·对应桥 cue-slot-aware·纯读）。

        对应泛化桥（CORRESPONDENCE_SLOT_MODE）：从 skeleton 重建 cue_sig·generate per-unit stash 算 current_cue_slots
        （cue slot 索引集·runtime length-guard 守 slot 对齐·纯语言 flat skeleton DFS 序=token position 序）·
        dispatch_slot _correspondence_bonus 仅 cue slot 触发（反 theater·审2 证 ATTR_CUE_SIG 天然单 cue 位·_cluster_by_cue
        单拆硬约束）。**镜像 _collect_cue_sig（structure_discover.py:986-1020）逐字逻辑**：read_composes_tree DFS 前序·
        PARAM slot 叶（ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs）逐 slot 读 ATTR_CUE_SIG（None 若无·build
        CONCEPT_LEAF 仅 cue 拆簇拆位非 None 写）。

        **为何 graph_view 复制而非 import structure_discover**：避免 result→process 反向依赖（守单向 L5→L0）·
        ~15 行纯读逻辑全在 graph_view 已有 read 原语（read_composes_tree / read_composes_attrs）上。

        返 ()=无 cue 位 / 全 None（arith/裸 NL/CUE_CLUSTER_MODE OFF）。纯读·确定性（DFS 前序 + visited·bit-identical）。
        """
        children_of = self.read_composes_tree(skeleton_ref)[0]
        parts: list[ConceptRef | None] = []
        visited: set[ConceptRef] = set()

        def _dfs(node: ConceptRef) -> None:
            if node in visited:
                return
            visited.add(node)
            attrs = read_composes_attrs(self._b, node)
            if ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs:
                # PARAM slot leaf（build CONCEPT_LEAF 写 ATTR_OPERAND=make_variable(slot)）·读 ATTR_CUE_SIG
                # （cue 拆簇拆位·build 仅 cue_sig 非 None 写·absence=非 cue 位 None·镜像 _collect_cue_sig）。
                cue = attrs.get(ATTR_CUE_SIG)
                parts.append((cue[0], cue[1]) if cue is not None else None)
            for child in children_of.get(node, []):
                _dfs(child)

        _dfs(skeleton_ref)
        return tuple(parts)

    def read_slot_lcas(self, skeleton_ref: ConceptRef) -> tuple[ConceptRef | None, ...]:
        """skeleton 子树 DFS 前序 -> PARAM slot 序的 ATTR_SLOT_ROLE tuple（slot LCA 重建·候选 C 内容词抽象约束·纯读）。

        命门③ 候选 C（slot_lca 抽象约束）：从 skeleton 重建 slot_lcas·generate per-unit stash 算 current_slot_lca
        （runtime length-guard 守 slot 对齐）·dispatch_slot 内容词位按 c IS_A slot_lca 过滤候选（reflexive-transitive·is_a_descendant_of）。
        **镜像 read_cue_sig 范式 + _collect_slot_lcas（structure_discover.py:939-983）逐字逻辑**：read_composes_tree DFS 前序·
        PARAM slot 叶（ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs）逐 slot 读 ATTR_SLOT_ROLE（None 若无·build
        CONCEPT_LEAF 仅 lca_ref 非 None 写·absence=无类约束）。

        **为何 graph_view 复制而非 import structure_discover**：避免 result->process 反向依赖（守单向 L5->L0·同 read_cue_sig :281-282）。

        返 ()=无 slot / 全 None（无 INSTANTIATES skeleton / _cluster_by_lca 未触发）。纯读·确定性（DFS 前序 + visited·bit-identical）。
        """
        children_of = self.read_composes_tree(skeleton_ref)[0]
        parts: list[ConceptRef | None] = []
        visited: set[ConceptRef] = set()

        def _dfs(node: ConceptRef) -> None:
            if node in visited:
                return
            visited.add(node)
            attrs = read_composes_attrs(self._b, node)
            if ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs:
                # PARAM slot leaf（build CONCEPT_LEAF 写 ATTR_OPERAND=make_variable(slot)）·读 ATTR_SLOT_ROLE
                # （_cluster_by_lca set_lca 簇内 slot 全 token LCA·build 仅 lca_ref 非 None 写·absence=无类约束 None·镜像 _collect_slot_lcas）。
                lca = attrs.get(ATTR_SLOT_ROLE)
                parts.append((lca[0], lca[1]) if lca is not None else None)
            for child in children_of.get(node, []):
                _dfs(child)

        _dfs(skeleton_ref)
        return tuple(parts)

    def is_a_descendant_of(self, c: ConceptRef, ancestor: ConceptRef) -> bool:
        """c IS_A ancestor（reflexive-transitive·c==ancestor or ancestor in ancestors(c)）·命门③ 候选 C slot_lca 过滤判定。

        **非 nearest_isa_ancestor==slot_lca**（过严·c 可深 slot_lca 下多层·nearest 仅返单最深祖先·nearest==slot_lca False 漏合法候选·design 决策·审1/审2 核证）。
        amap 复用 self._ancestor_map_cache（per-space lazy build·mirror selection_pref_score:579-582 范式）·无新 per-ref cache->
        无新 invalidate 需求（既有 invalidate_ancestor_map:613-642 清 _ancestor_map_cache 已够·formal_train:522 per-item post-observe pre-generate 调）。
        """
        if c == ancestor:
            return True
        sid = c[0]
        amap = self._ancestor_map_cache.get(sid)
        if amap is None:
            amap = build_isa_ancestor_map(self._b, space_id=sid)
            self._ancestor_map_cache[sid] = amap
        return ancestor in amap.get(c, frozenset())

    def cue_rel_of(self, word_ref: ConceptRef) -> int:
        """word → D:11 EDGE_RELATION_SIGNAL PRIMARY + SOURCE_BARE_TEXT → REL_* → rel_kind（v2-learned only·反 theater）。

        对应泛化桥 per-candidate readback（dispatch_slot _correspondence_bonus 消费·CORRESPONDENCE_SLOT_MODE）。
        **镜像 lookup_word_concept 但加 source 滤**（审1 CONDITION A）：lookup_word_concept 只 tier_filter·boot STEP5
        PR1 种子（等于/导致/是·source=SOURCE_TEACHER·collide 已高·不需 bridge bonus）会误触发→floor 虚高 theater。
        本方法双特征 `tier==TIER_PRIMARY AND source==SOURCE_BARE_TEXT`·只认 v2 tally→promote 学到的 D:11
        （record_emergent_relation_signal_shadow source=SOURCE_BARE_TEXT·promote set_tier 翻 PRIMARY 不改 source·
        审1/审2 核证 EDGE_RELATION_SIGNAL 仅两写侧·无第三 source 误杀）。

        多 v2-learned D:11（word→多 REL_*·罕见）取 min(by (to_sid,to_lid)) 定序（确定性·镜像 rel_kind_of_skeleton·
        审1 LOW-2 强退化语义：匹配另一 R→bonus=0·非 crash）。返首匹配 rel_kind / 0（无 v2-learned PRIMARY+BARE_TEXT）。
        per-ref cache（_cue_rel_cache·D:11 tally/promote observe-only 写·per-item invalidate_ancestor_map 清够·
        镜像 _pronoun_count_cache·0 亦缓存 _COUNT_MISS 区分）。gate caller 门控（OFF 不调→CI 不 cache）。

        非自证（审2 核证 cognition/result/ 零 D:11 写点）：纯 read·generation 不写 D:11→无自举环。
        """
        cached = self._cue_rel_cache.get(word_ref, _COUNT_MISS)
        if cached is not _COUNT_MISS:
            return cached
        best: tuple[tuple[int, int], int] | None = None   # (to_ref, rel_kind)·min by to_ref 定序
        for r in self._b.select("edge", where={
            "space_id_from": word_ref[0], "local_id_from": word_ref[1],
            "edge_type": EDGE_RELATION_SIGNAL,
        }):
            if r.get("tier") != TIER_PRIMARY:
                continue   # SHADOW（未 promote）不触发（学到的才驱动·审1 CONDITION B）
            if r.get("source") != SOURCE_BARE_TEXT:
                continue   # 审1 CONDITION A：排 boot SOURCE_TEACHER 种子（等于/导致·非 v2-learned）
            to_ref = (r["space_id_to"], r["local_id_to"])
            attrs = read_composes_attrs(self._b, to_ref)
            kind = attrs.get(ATTR_RELATION_PRIMITIVE, (0, 0))[0]
            if kind != 0 and (best is None or to_ref < best[0]):
                best = (to_ref, kind)   # 确定性 min by to_ref（多 v2-learned R·罕见·定序守 bit-identical）
        rel_kind = best[1] if best is not None else 0
        self._cue_rel_cache[word_ref] = rel_kind
        return rel_kind

    def relation_cue_candidates(self, rel_kind: int, *,
                                space_id: int) -> list[ConceptRef]:
        """读取关系类型对应的已学习 cue 词候选。

        只返回同一 space 内 `PRIMARY + SOURCE_BARE_TEXT` 的 D:11 word->REL_* 边。
        教师 boot 词和未晋升 SHADOW 不进入生成候选池；目标节点必须带匹配的
        ATTR_RELATION_PRIMITIVE，避免 D:11 共享边类型的 operator/modal/action 污染。
        """
        assert_int(rel_kind, space_id,
                   _where="ConceptGraph.relation_cue_candidates")
        if rel_kind == 0:
            return []
        key = (space_id, rel_kind)
        cached = self._relation_cue_cache.get(key)
        if cached is None:
            candidates: set[ConceptRef] = set()
            rows = self._b.select("edge", where={
                "space_id_from": space_id,
                "edge_type": EDGE_RELATION_SIGNAL,
                "tier": TIER_PRIMARY,
                "source": SOURCE_BARE_TEXT,
            })
            for row in rows:
                target = (row["space_id_to"], row["local_id_to"])
                attrs = read_composes_attrs(self._b, target)
                if attrs.get(ATTR_RELATION_PRIMITIVE, (0, 0))[0] == rel_kind:
                    candidates.add((row["space_id_from"], row["local_id_from"]))
            cached = tuple(sorted(candidates))
            self._relation_cue_cache[key] = cached
        return list(cached)

    def _ref_node_type(self, ref: ConceptRef) -> int | None:
        """读 ref 的 node_type（concept_node.type·PURE_ALIAS 自包含 node_type 判据用）。

        无行→None（理论不发生·activate_candidates 的 ref 皆是已建节点）。仅在 PURE_ALIAS 分支调（rare）·
        activate_candidates 结果已 per-ref 缓存故每 ref 至多一次读·无独立 cache 必要。
        """
        rows = self._b.select("concept_node",
                              where={"space_id": ref[0], "local_id": ref[1]}, limit=1)
        return rows[0]["type"] if rows else None

    # ---- memory sequence（记忆空间种类3·def_array 概念 ref 序） ----

    def read_memory_sequence(self, ref: ConceptRef) -> list[ConceptRef]:
        """记忆序列节点（def_array ref_space_id!=0 的概念 ref 序·回放直出）。

        记忆序列存储首版经 def_array（§十五决策4 变长序列范式）·无序列 → []。
        """
        cached = self._mem_seq_cache.get(ref)
        if cached is None:
            sid, lid = ref
            rows = self._b.select("def_array", where={
                "space_id": sid, "local_id": lid,
            })
            seq: list[ConceptRef] = []
            for r in sorted(rows, key=lambda r: r["order_index"]):
                if r["ref_space_id"] != 0:   # 概念 ref 非 role 标记
                    seq.append((r["ref_space_id"], r["ref_local_id"]))
            cached = seq
            self._mem_seq_cache[ref] = cached
        return list(cached)   # copy：caller 可能 mutate·避污染 cache

    # ---- token 序（P0 #1040·struct_ref → 段 token concept 序·def_array 存储） ----

    def read_token_seq(self, struct_ref: ConceptRef) -> list[ConceptRef]:
        """struct_ref → 段内 token concept ref 序（def_array ref_space_id!=0·order_index 序·P0 #1040）。

        observe attach_token_seq（gate ON·call-site 门控）存段 resolved token concept ref 序到 struct_ref 的
        def_array（每 position 一行·ref_space_id=token 真实 space·与 role markers ref_space_id==0 共存）。本方法读
        ref_space_id!=0 行按 order_index 序 → token concept ref 序。

        **为何存储非 PRECEDES walk**：walk 按 concept ref dedup·重复 token（功能词"的"跨 position 共享同 concept ref）
        致 walk 丢 position 漏 token·真语料炸。存储每 position 一行 → 完整序列·repeat-safe。

        **与 read_memory_sequence 的关系**：两者皆读 def_array ref_space_id!=0·对 struct_ref 返同序（段 token 序
        既是 dispatch 源亦是潜在 replay 序）。语义分立：本方法=生成侧 dispatch token concept 源（slot.ref=token·
        activate_candidates 取别名/surface_of 产真字）/ read_memory_sequence=回放直出（filler=True·generate 硬 False
        不走·dormant）。独立 cache 避语义耦合。

        **P0/P0b/P0a 接通**：返 token concept ref → generate slot.ref=token → activate_candidates(token) 取
        PURE_ALIAS 别名词形（apple↔苹果 多候选·P0b 活）/ [token] 单挂 → surface_of(token) → P0a 码点真字。
        统计独立产出腿（判据①⑤·非 truth/非 can_ween）。

        无存行（gate OFF observe 不写 / code/arith struct_ref 不走此路径）→ []。仅 gate DISPATCH_TOKEN_CHAIN_MODE
        ON 时 generate 调。确定性（order_index 序）·bit-identical（一代内 def_array 稳定·cache list == fresh select）。
        """
        cached = self._token_seq_cache.get(struct_ref)
        if cached is not None:
            return list(cached)
        sid, lid = struct_ref
        rows = self._b.select("def_array", where={"space_id": sid, "local_id": lid})
        seq: list[ConceptRef] = [
            (r["ref_space_id"], r["ref_local_id"])
            for r in sorted(rows, key=lambda r: r["order_index"])
            if r["ref_space_id"] != 0   # 仅 token concept ref（排 role markers ref_space_id==0）
        ]
        self._token_seq_cache[struct_ref] = seq
        return list(seq)   # copy：caller（generate）不 mutate·守 cache 纯净

    # ---- chapter_seq（篇章结构序·缺口①·独立扩展表·镜像 read_role_seq） ----

    def read_chapter_seq(self, struct_ref: ConceptRef) -> tuple[int, int, int] | None:
        """段 struct_ref 的篇章结构序 → (chapter_seq, section_seq, doc_seq) | None。

        章节标记从独立扩展表 chapter_seq_table 读（observe attach_chapter_seq 落·机器可读结构源
        HTML/Markdown/LaTeX/code AST parse 填 segment.chapter_seq·文学卷章回 defer）。生成 M5 章边界
        分页候选读（generate.py·反 theater 最小消费者）·M5/M6 真分页 defer（Stage 6）。
        无行=该 struct_ref 无章节标记（无标记主流文本·退化同流水账·章节承载 defer 钥匙①）。
        表未注册→None（bare fixture / 未 register_chapter_seq·向后兼容）。
        镜像 read_role_seq 语义但落独立表（不撞 def_array 二元区分 ref_space_id==0/!=0）。
        """
        sid, lid = struct_ref
        try:
            rows = self._b.select(CHAPTER_SEQ_TABLE, where={
                "space_id": sid, "local_id": lid,
            }, limit=1)
        except KeyError:
            return None   # 表未注册（bare fixture）·向后兼容
        if not rows:
            return None   # 该 struct_ref 无章节标记（无标记文本·退化）
        r = rows[0]
        return (r["chapter_seq"], r["section_seq"], r["doc_seq"])

    # ---- collide_score（上下文扩散纯整数共现分） ----

    def collide_score(self, c: ConceptRef, ctx_refs: list[ConceptRef]) -> int:
        """c 与 ctx_refs 的共现分（读 strength 累加·总收口 0.1 COOCCURS A'·纯整数·确定性）。

        = Σ strength（c 与 ctx_refs 共现的 COOCCURS 边·from/to 双向）。
        读 strength 累加（与 hub_degree/_cooccurs_count 协同）：gate OFF（旧 add strength 恒 1）累加=row count
        = 旧语义 bit-identical；gate ON（add_cooccurs_dedup·strength=频次）累加=真共现频次（c 与高频 ctx 共现→高分·
        保持消歧判别力）。ctx_set 去重（同 ctx_ref 不重复计）但**同 c-ctx 多频次计入 strength**（频次语义·非集合基数）。
        COOCCURS SHADOW 隔离不影响（计数用·不进默认 A1 头聚合·§十五C9-bis A12）。无 ctx → 0（消歧退纯 tiebreak）。

        perf round6：邻接经 _cooccur_neighbors per-ref cache（替每调 2 select·cProfile n=8 805K 调 1.6M select
        cand_distinct 仅 3 recurrence 26 万倍·cache 杀 fan-out）。bit-identical：cache (other,strength) 序 ==
        原 2 select 扫描序（from 先 to 后·保插入序）·整数累加同序·自环 c→c 双边计同（from/to 各 append 一次）。
        """
        if not ctx_refs:
            return 0
        ctx_set = set(ctx_refs)
        score = 0
        for other, strength in self._cooccur_neighbors(c):
            if other in ctx_set:
                score += strength
        assert_int(score, _where="ConceptGraph.collide_score")   # 总收口 0.1：读 strength 累加·守纯整数
        return score

    # ---- second_order_similarity（二阶 shared-neighbor Jaccard·read-side·Phase C §十五-bis C.1） ----

    def second_order_similarity(self, a: ConceptRef, b: ConceptRef) -> int:
        """a/b 二阶相似（shared-neighbor Jaccard·read-side·**确定性 computation 非"学得"**·Phase C §十五-bis C.1）。

        Sahlgren refined DH：一阶共现(collide_score)=syntagmatic 组合·**二阶 shared-neighbor=paradigmatic 相似**。
        = `(_SIM_SCALE * |A∩B|) // |A∪B|`（A/B = **hub-滤后** COOCCURS 邻集·scaled-int floor division·
        mirror coverage_overlap a4_align:33·0.._SIM_SCALE·**空并集→0** 守确定性无 ZeroDivision）。

        **不存 EDGE_SIMILAR 边**（饱和=theater·词库类不可学·方法论 §三 matrix·§四 精炼①）·**read-side 算**·
        consumer（slot_dispatch._second_order_bonus）调·phase D/E 对应 bootstrap 亦调（downstream·非 Phase C 交付）。
        **strength-invariant**：邻集取 **ref SET**（忽略 strength）·COOCCURS dedup gate 不影响集成员→bit-identical。
        **hub 滤**（graph.hub_set 排除 hub·剔"的/是"无区分度 token 否则 hub co-occur-with-all 主宰交集塌相似度）。
        per-pair **对称 cache**（key=tuple(sorted([a,b]))·Jaccard(a,b)==Jaccard(b,a)·invalidate_ancestor_map 清）。

        铁律：纯整数（scaled-int //·禁 float）/ 确定性（sorted cache key + set 运算 bit-identical）/
          幂等（纯读 + cache·重复同果）/ bit-identical（strength-invariant·gate OFF/ON 同集）。
        诚实边界：二阶相似是 set-overlap 整数（确定性 computation over observed COOCCURS）·非"学得相似"·
          非语义同义（#479 truth 墙·stable≠correct）·冷启动空邻集→0（无证据·非 theater）。
        """
        key: tuple[ConceptRef, ConceptRef] = tuple(sorted([a, b]))   # 对称 key（Jaccard 对称）
        cached = self._second_order_cache.get(key)
        if cached is not None:
            return cached
        hubs = self.hub_set()
        nbrs_a = {n for n, _s in self._cooccur_neighbors(a) if n not in hubs}
        nbrs_b = {n for n, _s in self._cooccur_neighbors(b) if n not in hubs}
        inter = nbrs_a & nbrs_b
        union = nbrs_a | nbrs_b
        score = (_SIM_SCALE * len(inter)) // len(union) if union else 0   # 空并集→0（冷启动·守确定性）
        assert_int(score, _where="ConceptGraph.second_order_similarity")
        self._second_order_cache[key] = score
        return score

    def _cooccur_neighbors(self, c: ConceptRef) -> list[tuple[ConceptRef, int]]:
        """c 的 COOCCURS 邻接（双向·(other_ref, strength)·perf round6 per-ref cache）。

        返 c 全部 COOCCURS 边对端 + strength（from 端先·to 端后·保插入序 = collide_score 原两 select 扫描序）。
        collide_score 唯一 consumer（slot_dispatch:150·只读迭代不 mutate·返 cache 本体无 copy）。
        **同 round5 read-cache 范式（显式 invalidate·无版本号自失效）**：对抗审证 edge_store 多实例→本
        _edge_store.cooccurs_version 永留 0（observe 写经 ObservePipeline 自己的 ES bump）→版本号自失效死代码。
        正确性靠 invalidate_ancestor_map 显式 clear（formal_train:532 每 item）+ generation 内不 observe
        （COOCCURS 稳定）。bit-identical：与原 collide_score 内联 2 select 同源（edge_type+from/to 端）同序
        （from 先 to 后）同 strength 列·自环 c→c 在 from-loop + to-loop 各 append 一次（=原两边各计一次·双边计保留）。
        """
        cached = self._cooccur_nbr_cache.get(c)
        if cached is None:
            sid, lid = c
            nbrs: list[tuple[ConceptRef, int]] = []
            # COOCCURS 无向：from 或 to 端命中 c 都算·读 strength（gate OFF 恒 1·等价数行；gate ON 频次）
            for r in self._b.select("edge", where={
                "edge_type": EDGE_COOCCURS, "space_id_from": sid, "local_id_from": lid,
            }):
                nbrs.append(((r["space_id_to"], r["local_id_to"]), r["strength"]))
            for r in self._b.select("edge", where={
                "edge_type": EDGE_COOCCURS, "space_id_to": sid, "local_id_to": lid,
            }):
                nbrs.append(((r["space_id_from"], r["local_id_from"]), r["strength"]))
            self._cooccur_nbr_cache[c] = nbrs
            cached = nbrs
        return cached

    # ---- is_hub（hub_degree 统计判据·归一化半 A read-time 过滤·caller gate 守） ----

    def is_hub(self, ref: ConceptRef) -> bool:
        """ref 是否 hub（COOCCURS 关联边总数 ≥ θ·read-time·gate 守由 caller slot_dispatch）。

        委托 cognition/shared/hub_detect.is_hub（fresh compute·零 module cache·避 stale-cache）。
        slot_dispatch collide_score caller 侧排除用（candidates + ctx_refs 过滤·解"分子是曾经"排序污染）。
        冷启动（无 COOCCURS 边）→ False（不排除·bit-identical OFF 退化·无 crash）。
        """
        return _shared_is_hub(ref, self._edge_store)

    def hub_set(self) -> set[ConceptRef]:
        """hub ref 集（上下文派生状态·替 per-ref is_hub 批量调用）。

        委托 cognition/shared/hub_detect.compute_hub_set；首次单遍扫 COOCCURS，后续同一生成轮复用。
        ``invalidate_ancestor_map`` 在 observe 后显式失效，避免本对象独立 EdgeStore 的版本号看不到外部 writer。
        slot_dispatch 批量
        排除用（candidates + ctx_refs 一次 membership 查·替 K+M 次 per-ref is_hub·解 36 万调 fan-out）。
        与 is_hub 同 theta（THETA_HUB_DEGREE）-> `ref in hub_set()` == `is_hub(ref)` bit-identical。
        生成期 COOCCURS 不变（generation 不 observe）-> 缓存跨 slot 稳定·一代一次 fresh 重算。
        """
        return _shared_compute_hub_set(
            self._edge_store,
            state=self._hub_degree_state,
        )

    # ---- selection_pref_score（CLASS 级共现·S4 决断 2 生成侧精查·两层正交第二腿） ----

    def selection_pref_score(self, c: ConceptRef, ctx_refs: list[ConceptRef]) -> int:
        """c 与 ctx_refs 的 selection_pref CLASS 级共现分（pair-rate·纯整数·确定性）。

        = Σ_{r ∈ ctx_refs, r ≠ c} read_selection_pref_count(backend, ref_a=r,
        ref_class=class_of(c)) → (sp_sn + sp_tn)（None→0·sp_sn 成功搭配加成 S4 后续加固）。

        class_of(c) = nearest_isa_ancestor（IS_A 最近祖先·最深非升序首·S4 项2·三处同源
        abstraction.nearest_isa_ancestor·写读一致·守 pair-rate 命中·inline 不 import
        understanding.selection_pref·守 result→understanding 单向依赖·同 reward_propagate 先例）。

        **与 collide_score 非冗余**（两层正交·S4 决断 2 设计故意）：collide_score = **token 级**
        共现（c 与 ctx_refs 的 COOCCURS 边计数）·本方法 = **CLASS 级**共现（c 的 IS_A class 与
        ctx_refs 的 selection_pref_count）·class 级解"未见 token 经 class 共现泛化"（"追"未见配
        "狐狸"但见配"动物"class·狐狸 IS_A 动物 → 仍获 boost）·token 级解"已见 token 共现"。

        ancestor_map lazy build per space（build_isa_ancestor_map·self._ancestor_map_cache·
        生产 ConceptGraph 是 ctx 单例·同 generation 内 IS_A 不变→cache 正确·跨 round observe 增 IS_A 后
        单调陈旧→见 __init__ _ancestor_map_cache 诚实边界）。c ∈ ctx_refs 跳过避自 boost（对称写时
        if a == b: continue）。读 sp_tn + sp_sn（共现总数 + 成功搭配加成·**sp_sn 消费侧落 S4 后续加固·
        反 dead column**：sp_sn 由 reward>0 episode feed（record_selection_pref_reward·reward_propagate 落点⑥）·
        成功搭配比纯共现更可信 → boost·强化 selectional preference 本意·镜像 effective_freq 的 e_sn→promote
        消费路径在 selection_pref 的对偶是"生成精查"·PR 粗筛侧 sp_agg 不含 sp_sn 镜像 eff_freq 不含 e_sn）。
        base_count 不读（首版=0 占位·通识注入 defer·同 sp_agg 不含 base 单独）。表未注册→
        read_selection_pref_count 返 None→0（向后兼容·同 collide_score 范式）。

        诚实边界：消歧=拓扑共现非语义接地（#479 W2 truth 墙·**非 D 物理接地**·“吃猫”数据见过就高 count 不保证语义正确·stable≠correct）。
        """
        if not ctx_refs:
            return 0
        sid, lid = c
        assert_int(sid, lid, _where="ConceptGraph.selection_pref_score.c")
        # ancestor_map lazy build per space（episode-scoped cache·首次调用 build·后续命中）
        amap = self._ancestor_map_cache.get(sid)
        if amap is None:
            amap = build_isa_ancestor_map(self._b, space_id=sid)
            self._ancestor_map_cache[sid] = amap
        # class_of(c) = nearest_isa_ancestor（IS_A 最近祖先·最深非升序首·S4 项2·写读一致三处同源）
        class_of_c = nearest_isa_ancestor(amap, c)
        # Σ_{r ∈ ctx_refs, r ≠ c} (sp_sn + sp_tn)(r, class_of(c))·None→0
        # sp_sn 成功搭配加成（reward>0 episode feed·反 dead column·S4 后续加固 sp_sn 消费侧落生成精查）
        # observe_mode（方案3 tn路 B5 β_arith 修法·gate SP_OBSERVE_MODE 守）：ON→row[2]=sp_observe_tn（避 reward 染色）/
        # OFF→row[2]=sp_tn（既有 bit-identical）。hoist 出循环（一代内 gate 常量·perf round9 cache key 复用）。
        obs_mode = bool(getattr(gates, "SP_OBSERVE_MODE", False))
        score = 0
        for r in ctx_refs:
            if r == c:
                continue   # 对称写时 a==b 跳过·避自 boost
            # perf round9：per-pair read cache（key=(r, class_of_c, obs_mode)·memo read_selection_pref_count·
            # reward 在 episode 间写 selection_pref_count → per-generate clear·见 __init__ + invalidate_generate_read_cache）。
            # None 亦缓存（冷启动 pair 多·_COUNT_MISS sentinel 区分"未算"）。
            key = (r, class_of_c, obs_mode)
            row = self._sp_count_cache.get(key, _COUNT_MISS)
            if row is _COUNT_MISS:
                row = read_selection_pref_count(self._b, r, class_of_c, observe_mode=obs_mode)
                self._sp_count_cache[key] = row
            if row is not None:
                # 审计根治 [严重-2]：gate SP_OBSERVE_MODE ON 时 cap sp_sn ≤ observe_tn（守子集防膨胀）。
                # sp_observe_tn sign-agnostic（决策时写·非 reward episode）·不同比 sp_sn 增·cap sp_sn ≤ observe_tn
                # 防 β_arith 塌缩（β_arith 病：reward>0 episode 同比 sp_sn++/sp_tn++ → 生成侧 score 塌缩·
                # cap 守 sp_sn 不超 observe_tn 子集·保留成功加成但防膨胀）。gate OFF 既有 sp_sn + sp_tn（bit-identical）。
                if obs_mode:
                    score += min(row[1], row[2]) + row[2]   # cap sp_sn ≤ observe_tn·min(sp_sn,observe_tn)+observe_tn
                else:
                    score += row[1] + row[2]   # 既有 sp_sn + sp_tn（row[0]=base 永 0 不读·通识注入 defer）
        return score

    def invalidate_ancestor_map(self, space_id: int | None = None) -> None:
        """清 ancestor_map lazy cache + 生成侧只读 per-ref cache（S4 后续加固·perf round5）。

        生产 ConceptGraph 是 ctx 单例·observe 增 IS_A / def_array / REFERS_TO 后 cache 单调陈旧。caller 须
        在 round 边界（observe 后、生成前）调此方法清 cache·让 selection_pref_score / read_role_seq /
        activate_candidates / read_memory_sequence 下次调用重建（含新边）。formal_train:522 per-item 已调
        （post-observe pre-generate）·一代内命中·跨 item 清。

        space_id=None 清全 cache·指定 sid 清单 space（ancestor_map）·生成侧 per-ref cache 不分 space 全清
        （observe 任意 space 增边均可能影响·保守全清·bit-identical）。无 cache 时 no-op（gate OFF 安全）。
        铁律：纯性能（cache 不影响输出值·幂等重建）/ 确定性 / bit-identical（一代内 def_array/REFERS_TO/IS_A
        不变·observe 后已清·缓存的 sorted 列表 == fresh select）。
        """
        if space_id is None:
            self._ancestor_map_cache.clear()
        else:
            self._ancestor_map_cache.pop(space_id, None)
        self._role_seq_cache.clear()
        self._activate_cache.clear()
        self._mem_seq_cache.clear()
        self._token_seq_cache.clear()   # P0 #1040：read_token_seq def_array 读 cache（observe attach_token_seq 增 token 行后清·保 fresh）
        self._cooccur_nbr_cache.clear()   # perf round6：collide_score 邻接 cache（observe 增 COOCCURS 后清·保 fresh）
        self._hub_degree_state.invalidate()   # 其他 EdgeStore 写 COOCCURS 后重建 hub 度数
        self._lang_cache.clear()   # perf round7：lang_of cache（observe set_mark 后清·保 fresh）
        self._surface_cache.clear()   # P0a：surface_of cache（observe ensure 增概念后清·保 fresh）
        self._similar_cache.clear()   # perf round7：similar_candidates cache（observe 增 EDGE_SIMILAR 后清·保 fresh）
        self._sp_count_cache.clear()   # perf round9：selection_pref pair read cache（observe 增共现后清·reward 写另由 generate 入口清）
        self._pronoun_count_cache.clear()   # perf round9：pronoun pair read cache（observe-only 写·per-item 清够）
        self._head_pref_cache.clear()   # perf round10：G2 head_pref read cache（modification_direction observe-only 写·per-item 清够·镜像 pronoun）
        self._second_order_cache.clear()   # Phase C：second_order_similarity per-pair cache（COOCCURS observe-only 写·per-item 清够·镜像 _cooccur_nbr_cache·保 fresh）
        self._cue_rel_cache.clear()   # 对应桥 cue_rel_of cache（D:11 tally/promote observe-only 写·per-item 清够·保 fresh）
        self._relation_cue_cache.clear()   # relation-driven cue 候选（同 D:11 生命周期）

    def invalidate_generate_read_cache(self) -> None:
        """perf round9：清 selection_pref per-pair read cache（generate_output 入口调）。

        selection_pref_count 由 reward（record_selection_pref_reward·propagate_reward·episode_loop generate 后）
        在 episode 间写 → 跨 generate cache 会陈旧。generate_output 入口清：一代内 selection_pref_count 表稳定
        （generate 不写）→ cache 正确·跨 generate 清 → bit-identical。pronoun observe-only 写不在此清（per-item 够）。
        无 cache 时 no-op（gate OFF / 首次）。
        """
        self._sp_count_cache.clear()

    # ---- 命题节点迭代（G1+#774·G3b 全局扫命题节点 PROPERTY 出边用） ----

    def iter_proposition_nodes(self) -> list[ConceptRef]:
        """全 core space 命题节点（ATTR_PROPOSITION 标记·G3b 全局扫用·G1 reification+#774）。

        读 composes_attr kind=ATTR_PROPOSITION → 命题节点 ref 列表（确定性·ref 自然序 tiebreak·bit-identical）。
        G3b counterfactual_value_check 全局扫这些节点的 PROPERTY 出边·判同(subject,attr_type)多值=结构矛盾
        （层a·fork 分析 §3.2/§3.3·命题身份=(subject,attr_type) 去重→同对多值聚同节点→精确判矛盾无假矛盾）。

        表未注册（bare fixture / 未 register_composes_attr）→ KeyError try/except → []（向后兼容·
        同 read_chapter_seq 范式）。gate PROPOSITION_MODE OFF 时 observe skip build_property_edges →
        无 ATTR_PROPOSITION 行 → 返 [] → G3b 扫空返 1（既有行为 bit-identical·G3b has_value_claim=False 不激活）。
        """
        try:
            rows = self._b.select(COMPOSES_ATTR_TABLE, where={"kind": ATTR_PROPOSITION})
        except KeyError:
            return []   # 表未注册（bare fixture）·向后兼容·同 read_chapter_seq:133
        return sorted({(r["space_id"], r["local_id"]) for r in rows})

    # ---- 命题节点 identity（STEP6 PR3·G3b 模态对当跨节点分组用·解 ref→surface defer） ----

    def iter_proposition_identity(self) -> list[tuple[ConceptRef, ConceptRef, ConceptRef, int, int]]:
        """命题节点 identity 列表 → [(prop_ref, subj_ref, attr_ref, pol, mod), ...]（STEP6 PR3）。

        读 ATTR_PROP_SUBJ/ATTR_PROP_ATTR/ATTR_PROP_POLMOD（build_property_edges record·命题节点 (subj,attr,pol,mod)
        结构存·解 ref→surface defer——node_store 无 surface 列·命题 identity 须结构存于 composes_attr）。
        G3b 模态对当跨节点按 (subj,attr) 分组读此·判模态方阵（□p vs ◇¬p / □p vs ¬p·T 公理层a 形式）。

        返 (prop_ref, subj_ref, attr_ref, pol, mod) per 命题节点。确定性（iter_proposition_nodes 已 sorted·
        bit-identical）。无 identity 记录（PR3 前旧命题节点 / 缺 kind）→ skip（向后兼容·同 iter_proposition_nodes 范式）。
        gate PROPOSITION_MODE OFF → 无命题节点 → iter_proposition_nodes 返 [] → 返 []（既有行为 bit-identical）。
        """
        out: list[tuple[ConceptRef, ConceptRef, ConceptRef, int, int]] = []
        for prop_ref in self.iter_proposition_nodes():
            attrs = read_composes_attrs(self._b, prop_ref)
            subj = attrs.get(ATTR_PROP_SUBJ)
            attr = attrs.get(ATTR_PROP_ATTR)
            polmod = attrs.get(ATTR_PROP_POLMOD)
            if subj is None or attr is None or polmod is None:
                continue   # PR3 前旧命题节点无 identity / 缺 kind → skip（向后兼容）
            out.append((prop_ref, (subj[0], subj[1]), (attr[0], attr[1]), polmod[0], polmod[1]))
        return out

    # ---- 边查询（judge J3/J4/G3b 查图用） ----

    def out_edges(self, ref: ConceptRef, edge_type: int | None = None) -> list[dict[str, Any]]:
        sid, lid = ref
        where: dict[str, Any] = {"space_id_from": sid, "local_id_from": lid}
        if edge_type is not None:
            where["edge_type"] = edge_type
        return self._b.select("edge", where=where)

    def in_edges(self, ref: ConceptRef, edge_type: int | None = None) -> list[dict[str, Any]]:
        sid, lid = ref
        where: dict[str, Any] = {"space_id_to": sid, "local_id_to": lid}
        if edge_type is not None:
            where["edge_type"] = edge_type
        return self._b.select("edge", where=where)

    def similar_candidates(self, ref: ConceptRef) -> list[ConceptRef]:
        """EDGE_SIMILAR 邻居（双向·X~Y·slot-filler 候选扩展用·STEP5 PR4·D2 合规非向量）。

        返 ref 的相似概念 list（去重·双向：ref→X 出边 + X→ref 入边·"X 像 Y" 对称扩展）。
        dispatch_slot（gate SIMILAR_SLOT_MODE）读此扩展 slot 候选（X 像 Y → X 可填 Y 的 slot）。
        EDGE_SIMILAR 二元离散边·非相似度 SCORE·非向量（D2 三维度全不满）。

        perf round7（2026-07-13）：per-ref cache（_similar_cache·mirror _cooccur_nbr_cache / _lang_cache）。
        SIMILAR_SLOT_MODE 生产 ON（formal_train:2245 翻）→ dispatch_slot 每 dispatch 调一次（cProfile n=12
        730K 调·out_edges+in_edges 各 ~730K select·unique ~7729 → 95x 命中）。EDGE_SIMILAR observe-only
        （observe.py:256 不接 reward·strength 恒=1）→ reward 写 CAUSES 不染 SIMILAR → 一代内稳定 → per-item
        invalidate_ancestor_map clear 保 fresh。bit-identical：cache sorted(refs) == fresh sorted（同边集同序）。
        consumer 只读迭代（slot_dispatch:129 for s in ...）·不 mutate → 返 cache 本体无 copy（同 _cooccur_nbr_cache）。
        """
        cached = self._similar_cache.get(ref)
        if cached is not None:
            return cached
        out = self.out_edges(ref, EDGE_SIMILAR)   # ref→X（ref 像 X）
        inc = self.in_edges(ref, EDGE_SIMILAR)    # X→ref（X 像 ref）
        refs: set[ConceptRef] = set()
        for r in out:
            refs.add((r["space_id_to"], r["local_id_to"]))
        for r in inc:
            refs.add((r["space_id_from"], r["local_id_from"]))
        result = sorted(refs)   # sorted 守确定性（bit-identical·镜像 activate_candidates sorted 范式）
        self._similar_cache[ref] = result
        return result

    # ---- pronoun_score（B6 生成侧·审计根治 [严重-3]·pair-key 对偶 observe 侧） ----

    def pronoun_score(self, c: ConceptRef, antecedent_ref: ConceptRef) -> int:
        """c 作为 pronoun 解析到 antecedent_ref 的 pr_tn（pair-key 对偶 observe 侧·审计根治 [严重-3]）。

        读 read_pronoun_resolution_count(backend, pronoun_ref=c, antecedent_ref=antecedent_ref) → pr_tn
        （observe 路·决策时写·sign-agnostic·不染 reward·per-occurrence·避 β_arith）。
        pair-key 对偶 observe 侧：observe 写 (pronoun, best_antecedent)·生成侧读 (candidate=pronoun, slot.ref=antecedent)·
        同一 pair 数据·两侧各用各的（observe 写 pr_tn → 生成侧读 pr_tn 加 slot 候选分）。

        **gate PRONOUN_SLOT_MODE 守**（default OFF·守回归·OFF 返 0 不读·bit-identical）。
        冷启动（无 pair 行 / 表未注册）→ 0（向后兼容·镜像 selection_pref_score 范式）。
        consumer = dispatch_slot 第5路（gate PRONOUN_SLOT_MODE·并入 sp 维联合 _cap_sp cap 999·守 collide 主轴）。

        诚实边界：pr_tn 是决策频次非语义正确（stable≠correct·"它们→最近 token 可能功能词"接地墙外·
        代词消解结构非墙 vs sense 消歧 #479 真墙·§九.7.6 W2 拆分）。
        """
        record_diagnostic_event("hotspot.pronoun")
        if not getattr(gates, "PRONOUN_SLOT_MODE", False):
            return 0
        # perf round9：per-pair read cache（key=(c, antecedent_ref)·memo read_pronoun_resolution_count）。
        # pronoun_resolution_count observe-only 写（record_pronoun_resolution_decision·reward 不写·见其 docstring）→
        # per-item invalidate_ancestor_map 清够·可跨 generate 复用（比 sp_count 更高效）。None 亦缓存（_COUNT_MISS）。
        key = (c, antecedent_ref)
        row = self._pronoun_count_cache.get(key, _COUNT_MISS)
        if row is _COUNT_MISS:
            row = read_pronoun_resolution_count(self._b, c, antecedent_ref)
            self._pronoun_count_cache[key] = row
        if row is None:
            return 0
        return row[1]   # pr_tn（observe 路·sign-agnostic·避 β_arith）

    def head_pref_score(self, c: ConceptRef) -> int:
        """G2 修饰方向A：c 作 head（ 的 后）的偏好 bonus（head_count - mod_count·cap HEAD_PREF_CAP=9）。

        读 modification_direction.head_preference(backend, c) → (head_count, mod_count) →
        max(0, head_count - mod_count) capped HEAD_PREF_CAP（modest 亚轴 tiebreak·守 collide 主轴）。
        **gate MODIFIER_DIRECTION_MODE 守**（default OFF 返 0 不读·bit-identical）。
        冷启动（表未注册 / 无行）→ 0（向后兼容·镜像 pronoun_score 范式·try 守表未注册）。
        consumer = dispatch_slot 第 6 路（gate MODIFIER_DIRECTION_MODE·并入 sp 维联合 _cap_sp cap 999）。

        诚实边界：head/modifier 是 的-位置统计非语义中心语（#479 truth 墙·系统不判"真正 head"·只统计
        的-位置 head_count）。stable≠correct·30-40% 行为表面·语义中心语长尾撞墙。
        """
        if not getattr(gates, "MODIFIER_DIRECTION_MODE", False):
            return 0
        # perf round10：per-ref read cache（镜像 _pronoun_count_cache·head_preference 读 modification_direction
        # observe-only 写 → per-item invalidate_ancestor_map 清够·可跨 generate 复用）。dispatch_slot 每-候选调
        # → 全量 corpus generate 515k 调（调用体量下游于 dag_edges/topo_layers 规模·真根因见 dag_edges hotzone）
        # × head_preference 索引 O(1) 查（modification_direction.py:50 covering index·非全表扫）= 常量因子开销·非独立 O(n²)。
        # memo 后塌缩到 ~unique-ref 次（515k→unique·省 dict hash + int 转换）。gate OFF 早返 0 不触 cache（CI bit-identical）。0 亦缓存（_COUNT_MISS）。
        cached = self._head_pref_cache.get(c, _COUNT_MISS)
        if cached is not _COUNT_MISS:
            return cached
        try:
            head_count, mod_count = head_preference(self._b, c)
        except (KeyError, ValueError):
            score = 0   # 表未注册 / 查询错 → 0（冷启动向后兼容）
        else:
            diff = head_count - mod_count
            score = 0 if diff <= 0 else (diff if diff < HEAD_PREF_CAP else HEAD_PREF_CAP)
        self._head_pref_cache[c] = score
        return score

    # ---- COMPOSES 程序子树读（A3·致命#1·doc/重来_A3_代码域observe设计补充.md） ----

    _COMPOSES_MAX_DEPTH = 64   # 同 graph_compile.DEFAULT_MAX_DEPTH·防病态深递归

    def read_composes_tree(self, root: ConceptRef) -> tuple[
        dict[ConceptRef, list[ConceptRef]],
        dict[ConceptRef, int],
        dict[ConceptRef, int],
        dict[ConceptRef, tuple[int, int]],
        dict[ConceptRef, int],
    ]:
        """读 COMPOSES 子树 → (children_of, operator_of, operand_of, immediate_of, store_target_of)。

        从 root 沿 EDGE_COMPOSES 遍历全子树（DAG·visited 防重复入队·max_depth 防病态）。
        children_of 按 (order_index, NodeRef) 排（确定性 bit-identical）。
        其余 4 dict 从 composes_attr 表读（read_composes_attrs）。
        vm_proof_fn 经此重建 compile_graph 所需 5 dict（A3·致命#1）。
        """
        record_diagnostic_event("hotspot.skeleton_read")
        from collections import deque
        children_of: dict[ConceptRef, list[ConceptRef]] = {}
        operator_of: dict[ConceptRef, int] = {}
        operand_of: dict[ConceptRef, int] = {}
        immediate_of: dict[ConceptRef, tuple[int, int]] = {}
        store_target_of: dict[ConceptRef, int] = {}
        visited: set[ConceptRef] = set()
        depth: dict[ConceptRef, int] = {root: 0}
        queue: deque[ConceptRef] = deque([root])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            # 读节点属性（composes_attr 表）
            attrs = read_composes_attrs(self._b, node)
            if ATTR_CTRL_TAG in attrs:
                operator_of[node] = attrs[ATTR_CTRL_TAG][0]   # 控制流根 CTRL_*
            elif ATTR_OPERATOR in attrs:
                operator_of[node] = attrs[ATTR_OPERATOR][0]   # 普通算子 OPCODE_*
            if ATTR_OPERAND in attrs:
                operand_of[node] = attrs[ATTR_OPERAND][0]
            if ATTR_IMMEDIATE in attrs:
                immediate_of[node] = attrs[ATTR_IMMEDIATE]
            if ATTR_STORE_TARGET in attrs:
                store_target_of[node] = attrs[ATTR_STORE_TARGET][0]
            # 读 COMPOSES 子边（按 order_index + NodeRef 排·确定性）
            if depth[node] >= self._COMPOSES_MAX_DEPTH:
                continue   # 防病态深（compile_graph 自己亦守 max_depth）
            edges = self.out_edges(node, EDGE_COMPOSES)
            if edges:
                ordered = sorted(
                    ((e["order_index"], (e["space_id_to"], e["local_id_to"]))
                     for e in edges),
                    key=lambda t: (t[0] if t[0] is not None else (1 << 30),
                                   t[1][0], t[1][1]),
                )
                child_list = [c for _, c in ordered]
                children_of[node] = child_list
                for c in child_list:
                    if c not in depth:
                        depth[c] = depth[node] + 1
                        queue.append(c)
        return children_of, operator_of, operand_of, immediate_of, store_target_of

    # ---- surface 文本（伴随库反查·注入式） ----

    def surface_of(self, ref: ConceptRef) -> str | None:
        """概念 ref → 词形字符串。

        三路（优先序）：
          1. 注入 SurfaceResolver（test 隔离用）→ override 返（测向后兼容）。
          2. 生产 gate ORDINAL_SURFACE_MODE ON → 读 concept_correspondence 码点 → chr → 文本
             （P0a·live-read gate·try/finally flip 即时生效·镜像 slot_dispatch live-read 范式）。
          3. gate OFF（CI 默认）→ None → 调用方走 ref 字面（`#{ref[0]}:{ref[1]}` 占位·既有行为·bit-identical）。

        **cache 仅 gate ON 时启用**（gate OFF 直返 None 不 cache·防 try/finally flip 后旧 None 命中）。
        **承重不变量**：无对应返 None 非 ""（judge.J2s slot_fill_rate `for w: if w:` 读 truthiness·"" falsy
        → bound 不增 → reward 变 → 破 bit-identical）。ordinal_surface_of 已守此（无行→None）。
        per-item invalidate_ancestor_map 清 cache（observe ensure 后·保 fresh·bit-identical）。
        """
        assert_int(ref[0], ref[1], _where="ConceptGraph.surface_of")
        if self._surface_of is not None:
            return self._surface_of(ref)
        if not getattr(gates, "ORDINAL_SURFACE_MODE", False):
            return None   # gate OFF（CI 默认）→ 退占位·bit-identical·不 cache
        cached = self._surface_cache.get(ref, _SURFACE_MISS)
        if cached is not _SURFACE_MISS:
            return cached
        v = ordinal_surface_of(self._b, ref)
        self._surface_cache[ref] = v
        return v

    def lang_of(self, ref: ConceptRef) -> int | None:
        """概念 ref → lang 标记（C1 跨语言偏好用·§7.7.1 路径 B 片3 接通）。

        Option A 两路（注入 override + abstract_mark fallback）：
          - 注入 LangResolver（test 隔离用·test_stage5 target_lang 测）→ override 返（测向后兼容）。
          - 无注入（生产）→ 读 abstract_mark MARK_LANG（refers_to NODE_WORD set_mark 写·片3 接通）。
        无 mark → None（概念点无 mark ≡ 既有注入式 None·bit-identical·dispatch_slot target_lang 退无偏好）。

        perf round7（2026-07-13）：per-ref cache（_lang_cache·mirror _cooccur_nbr_cache）。MARK_LANG observe-only
        （reward 写 CAUSES 非 mark·一代内稳定）→ per-item invalidate_ancestor_map clear 保 fresh。注入 override
        路径亦 cache（test LangResolver 返值一代内确定）。bit-identical：cache 值 == fresh get_mark·含 None。
        """
        cached = self._lang_cache.get(ref, _LANG_MISS)
        if cached is not _LANG_MISS:
            return cached
        if self._lang_of is not None:
            v = self._lang_of(ref)
        else:
            v = get_mark(self._b, ref=ref, mark_kind=MARK_LANG)
        self._lang_cache[ref] = v
        return v
