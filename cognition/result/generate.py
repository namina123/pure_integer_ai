"""cognition.result.generate — 模块1 路径填槽主框架 + 回放逐槽原语（§十四子问题1）。

generate_output(dag_path, graph, workmem, target_lang) -> OutputResult
  沿 DAG 拓扑序单 pass 遍历（不补 refinement loop·§十四砍）：
    for layer in dag_path.topo_layers:        # 分页 M5 每段 ≤ 热区 cap
      for unit in structure_units(layer):     # 结构单元=汇聚点/分支终点（DAG 拓扑自带）
        role_seq = graph.read_role_seq(unit)
        for slot in role_seq: dispatch_slot(...)   # 逐槽分派（模块2）
        workmem.add_produced(unit)            # carry·M5 长输出分页续接（FIFO cap=PRODUCED_REFS_WINDOW·#729）
  reached_sink = sink 在产出结构单元中（judge G2p 读·F6 落盘·模块1 填）

  路径填槽是主框架·回放直出是填槽内逐槽原语（§十四·非对立是包含）。
  伴随库不参与输出（未验证不发出·§十四三空间在生成的角色）。
  长输出分页 M5：拓扑层分段 + WorkMemory carry 续接（首版 in-memory·真分页 defer）。

  structure_units：首版 = 层内有 role_seq 的节点 ∪ dag_path.path.struct_unit_refs（汇聚点）。
    真实结构单元=汇聚点/分支终点由 DAG 拓扑自带（不外部划定）·role_seq 驱动槽序。

gate GENERATE_MODE：**承重件(路径填槽·闭环必跑)永远 active·generate_output/dispatch_slot 函数体不读 gate·gate 装饰性保留位**(无 OFF 态·生成是闭环核心非可选)。
铁律：纯整数（refs/序/in-degree 全整）/ DAG 无环断言 / 确定性 bit-identical（拓扑序+tiebreak 稳定）/
  最少冗余（role_seq 从图读不复制）。
诚实边界：生成=结构填充非理解表达（填已学骨架不懂所说·结构合法≠语义正确·§十四）/ 回放=逐槽原语非回声兜底主导。
defer：refinement loop 砍（逐槽分派+J1+J3+真负通路全覆盖）/ 长输出真分页 M5（首版 in-memory）/ transient DAG-path。
"""
from __future__ import annotations

from collections import deque
from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import (
    PathResult, OutputResult, OutputPart, RoleSlot, ConceptRef, CUE_SLOT_FILL,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot

# M5 长输出分页：每段最大结构单元数（热区 cap·oracle 标·首版 in-memory 不真分页）
LAYER_UNIT_CAP = 256

# 对话止血②（2026-07-18）：fill loop 段内词数硬上限（gate OUTPUT_LEN_CAP_MODE）。
# ①修幂等后 role_seq≈token_seq 正常不触发·仅①未覆盖累积路径兜底。实测 corpus_zh 单段 token max=120·256=max×2 真余量。
MAX_WORDS_PER_PART = 256


def _path_acyclic(dag_path: PathResult) -> bool:
    """DAG 无环断言（path.edges 选定边集·结构不变量·坏 DAG=坏多部分）。

    Kahn on path.edges：若所有节点可分层则无环。确定性（节点自然序）。
    """
    edges = dag_path.path.edges
    if not edges:
        return True
    nodes: set[ConceptRef] = set()
    adj: dict[ConceptRef, list[ConceptRef]] = {}
    indeg: dict[ConceptRef, int] = {}
    for ref in edges:
        u = (ref[0], ref[1])
        v = (ref[2], ref[3])
        if u == v:
            return False   # 自环
        nodes.add(u)
        nodes.add(v)
        adj.setdefault(u, []).append(v)
        indeg[v] = indeg.get(v, 0) + 1
    # Kahn 环检测（序无关·deque O(1) popleft 替 list.pop(0)·跳 sort·O(V²logV)->O(V+E)·bit-identical）。
    # 结果 visited==len(nodes) 与处理序无关（环内节点永不到达 indeg=0·序不影响环检测）。
    queue = deque(n for n in nodes if indeg.get(n, 0) == 0)
    visited = 0
    while queue:
        n = queue.popleft()
        visited += 1
        for m in adj.get(n, []):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return visited == len(nodes)


def structure_units(layer: list[ConceptRef], dag_path: PathResult,
                    graph: ConceptGraph) -> list[ConceptRef]:
    """层内结构单元（汇聚点/分支终点·DAG 拓扑自带·不外部划定）。

    首版 = 层内有 role_seq 的节点 ∪ dag_path.path.struct_unit_refs（步进标记的汇聚点）。
    确定性：按 layer 序 + ref 自然序去重保序。
    """
    unit_set = set(dag_path.path.struct_unit_refs)
    units: list[ConceptRef] = []
    seen: set[ConceptRef] = set()
    for node in layer:
        is_unit = node in unit_set or bool(graph.read_role_seq(node))
        if is_unit and node not in seen:
            units.append(node)
            seen.add(node)
    # struct_unit_refs 中跨层遗漏的补尾（保序）
    for ref in dag_path.path.struct_unit_refs:
        if ref not in seen:
            units.append(ref)
            seen.add(ref)
    return units


def generate_output(dag_path: PathResult, graph: ConceptGraph, workmem: Any,
                    target_lang: int) -> OutputResult:
    """路径填槽主框架。返 OutputResult{parts, lineage, reached_sink}。

    workmem：produced_refs carry（M5 续接）/ prior_topic_refs（collide_score ctx）。
    target_lang：LangMarker（C1 硬偏好·LANG_NONE=非语言模态无偏好）。
    """
    assert_no_float(target_lang, _where="generate_output.target_lang")
    assert_int(target_lang, _where="generate_output.target_lang_int")
    assert _path_acyclic(dag_path), "DAG 须无环（结构不变量·坏 DAG=坏多部分）"
    # perf round9：清 selection_pref per-pair read cache（reward_propagate 在 episode 间写 selection_pref_count·
    # 不则跨 generate cache 陈旧·bit-identical 守。一代内 generate 不写表→cache 正确。pronoun observe-only 不清）。
    graph.invalidate_generate_read_cache()

    parts: list[OutputPart] = []
    lineage: dict[tuple[ConceptRef, int], int] = {}
    produced: list[ConceptRef] = []
    truncated_units: set[ConceptRef] = set()   # 对话止血②：截断 unit 记录（gate OUTPUT_LEN_CAP_MODE OFF 永空）
    prev_chapter: int | None = None   # 9d 章边界跟踪（chapter_seq 变化点作 M5 分页候选）
    # P0 #1040：slot.ref 派发 token concept（非 struct_ref）+ ctx_refs token 级。live-read（try/finally flip 即时
    # 生效·镜像 slot_dispatch 范式）。OFF（CI 默认）= slot.ref=struct_ref + ctx=unit → 逐字现状 bit-identical。
    _dispatch_tokens = getattr(gates, "DISPATCH_TOKEN_CHAIN_MODE", False)

    for layer in dag_path.topo_layers:
        units = structure_units(layer, dag_path, graph)
        for unit in units:
            role_seq = graph.read_role_seq(unit)
            if not role_seq:
                continue
            # P0 #1040：gate ON 读段 token concept 序（graph.read_token_seq·def_array 存储·repeat-safe）。
            # OFF token_seq 空 → ref 退 unit（逐字现状）。无存行（code/arith struct_ref·gate OFF observe 不写）
            # → [] → 跳（gate ON 新增路径·非 bit-identical 路径）。
            token_seq: list[ConceptRef] = []
            if _dispatch_tokens:
                token_seq = graph.read_token_seq(unit)
                if not token_seq:
                    continue
            # 维度桥 reader（P1 G-PR2·DIM_BRIDGE_READ_MODE ON·读 EDGE_INSTANTIATES on unit·§十三-bis A.1）。
            # **审1 MEDIUM-1 修**：binding 在 __seg_ struct_ref=unit·非 slot.ref（DISPATCH_TOKEN_CHAIN_MODE ON 时
            # slot.ref=token concept 无 binding·读 slot.ref 恒 None）。unit 在 generate 作用域·slot_dispatch 只收
            # slot.ref 无 unit·故 reader 在此（每 unit 读一次·binding 段级非槽级·较 per-slot 更准更省）。
            # 记 workmem.last_dim_skeleton（P2 断桥 consumer stub·P1 write-only 无消费者·值填充 defer·非 observability）。
            # 每 unit 重置（binding 或 ()）避跨 unit 陈旧。gate OFF→不读→last_dim_skeleton 保持 default ()→bit-identical。
            # Phase A §十三-bis A.1：读 EDGE_INSTANTIATES 真边（替 ATTR_SKELETON_BINDING 注解·read_instantiates·关联在图中）。
            if getattr(gates, "DIM_BRIDGE_READ_MODE", False):
                _dim_skel = graph.read_instantiates(unit)
                workmem.last_dim_skeleton = _dim_skel if _dim_skel is not None else ()
            # 对应泛化桥 reader（CORRESPONDENCE_SLOT_MODE·doc/重来_对应泛化_readback_generation_桥_2026-07-17）：
            # 读 unit→INSTANTIATES→skeleton→REALIZES→rel_kind（两跳·审2 致命-1 修·挂 skeleton 非 unit·struct_ref 无 REALIZES 边）+
            # skeleton cue_sig → current_cue_slots（runtime length-guard 守 slot 对齐·纯语言 flat skeleton DFS 序=token position 序）。
            # dispatch_slot _correspondence_bonus（第 8 路·(β) 独立轴·cue-slot-aware）消费 current_rel_kind + current_slot_is_cue。
            # **每 unit 全路径写（4 case 无漏清·审2 核证无 stale state）**：gate OFF 不进→default 守 / 无 INSTANTIATES→else 清双字段 /
            # 有 INSTANTIATES+rel_kind=0→elif 清 cue_slots / 有 INSTANTIATES+rel_kind!=0→设 cue_slots（length-guard·不等→∅）。
            # read_instantiates 带 cache（DIM_BRIDGE 块已调一次·冗余经 cache 吸收·保持两 gate 独立·审2 低-1）。
            if getattr(gates, "CORRESPONDENCE_SLOT_MODE", False):
                _corr_skel = graph.read_instantiates(unit)
                if _corr_skel is None:
                    workmem.current_rel_kind = 0
                    workmem.current_cue_slots = frozenset()
                    workmem.current_cue_sig = ()   # case ② 无 INSTANTIATES->清（命门③ 4 case 无漏清·审1 MED-1）
                else:
                    workmem.current_rel_kind = graph.rel_kind_of_skeleton(_corr_skel)
                    if workmem.current_rel_kind != 0:
                        _cue_sig = graph.read_cue_sig(_corr_skel)
                        # runtime alignment guard：cue_sig length 须 == role_seq length
                        # （纯语言 flat skeleton 两序一致·不等=错位/混合 skeleton→退化无 bonus·sound future-risk 守·§3）。
                        if len(_cue_sig) == len(role_seq):
                            workmem.current_cue_slots = frozenset(
                                i for i, _c in enumerate(_cue_sig) if _c is not None
                            )
                            workmem.current_cue_sig = tuple(_cue_sig)   # case ④ 真值·dispatch_slot cue 位早 return 读
                        else:
                            workmem.current_cue_slots = frozenset()
                            workmem.current_cue_sig = ()   # case ④' length-guard fail->清（避 stale 残留）
                    else:
                        workmem.current_cue_slots = frozenset()
                        workmem.current_cue_sig = ()   # case ③ rel_kind=0->清（非 R-skeleton 无 cue 词）
            # 命门③ 候选 C（slot_lca 抽象约束·doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）：内容词按 slot IS_A LCA 类约束。
            # **独立块·非嵌 CORRESPONDENCE_SLOT_MODE**（design·审1/审2 核）：slot_lca 适用任意 INSTANTIATES 骨架非仅 R-skeleton·rel_kind!=0 非前置·
            # ATTR_SLOT_ROLE 由 _cluster_by_lca 写非 CUE_CLUSTER_MODE 门·独立于 cue 链。4 case 全路径设 current_slot_lcas（无 stale·审1 MED-1）。
            # **perf 诚实边界**（审1 LOW-1·非阻塞）：独立块再调 read_instantiates 一次（既有 CORRESPONDENCE 块:159+DIM_BRIDGE 块:149 各调一次）->每 unit 3 次 select·
            # n=300 时 900 次冗余。read_instantiates 无显式 per-ref cache（审1 实查）·bit-identical 不破·首版接受 perf 成本保设计清晰·gate-gated default OFF CI 不触发。
            if getattr(gates, "SLOT_LCA_CONSTRAINT_MODE", False):
                _lca_skel = graph.read_instantiates(unit)
                if _lca_skel is None:
                    workmem.current_slot_lcas = ()   # case ② 无 INSTANTIATES->清
                else:
                    _slot_lcas = graph.read_slot_lcas(_lca_skel)
                    if len(_slot_lcas) == len(role_seq):   # runtime alignment guard（同 cue_sig length-guard :170）
                        workmem.current_slot_lcas = tuple(_slot_lcas)   # case ④ 真值·dispatch_slot 内容词位读过滤
                    else:
                        workmem.current_slot_lcas = ()   # case ③ length-guard fail->清（避 stale 残留）
            words: list[str] = []
            emitted_tokens: list[ConceptRef] = []
            _len_cap = getattr(gates, "OUTPUT_LEN_CAP_MODE", False)
            _truncated = False
            for slot_idx, role in enumerate(role_seq):
                # 对话止血②（gate OUTPUT_LEN_CAP_MODE·MAX_WORDS_PER_PART）：段内词数硬上限。
                # ①修幂等后 role_seq≈token_seq 正常不触发·仅①未覆盖累积路径兜底。截断记 truncated_units。
                if _len_cap and len(words) >= MAX_WORDS_PER_PART:
                    _truncated = True
                    break
                # gate ON：slot.ref=段 token concept（activate_candidates 取别名/surface_of 产真字·P0b/P0a 接通）。
                # gate OFF：slot.ref=struct_ref（逐字现状·产 __seg_*）。filler_is_memory_sequence 硬编码 False defer。
                # role_seq 长 > token_seq 时（罕见·accumulation）extra slot 退 unit（防御）·但 emitted_tokens
                # 只收真 token（slot_idx < len）·守 token_refs/ctx 不渗 struct_ref（LOW-2 守）。
                has_token = bool(token_seq) and slot_idx < len(token_seq)
                ref = token_seq[slot_idx] if has_token else unit
                # 对应桥 per-slot cue flag（workmem 范式·slot 循环设·_correspondence_bonus 读 is_cue_slot 反 theater）：
                # cue-slot-aware → bonus 仅 cue slot 触发（非 cue slot→current_slot_is_cue=False→bonus=0→走 collide）。
                # gate OFF → current_cue_slots 保持 default ∅ → slot_idx in ∅ 恒 False → bit-identical（审1 LOW-1）。
                workmem.current_slot_is_cue = slot_idx in workmem.current_cue_slots
                workmem.current_slot_idx = slot_idx   # 命门③ HIGH-1 修·dispatch_slot 无 slot_idx 参数->走 workmem
                # 命门③ 候选 C：当前 slot_idx 的 LCA（None 占位位或越界->()·dispatch_slot 内容词位读 is_a_descendant_of 过滤）
                _slot_lcas_t = workmem.current_slot_lcas
                workmem.current_slot_lca = _slot_lcas_t[slot_idx] if (slot_idx < len(_slot_lcas_t) and _slot_lcas_t[slot_idx] is not None) else ()
                slot = RoleSlot(ref=ref, role=role,
                                filler_is_memory_sequence=False)
                word, src = dispatch_slot(slot, dag_path, graph, workmem,
                                          target_lang)
                words.append(word)
                lineage[(unit, slot_idx)] = src
                # 命门③ HIGH-2 修·cue token（src==CUE_SLOT_FILL）不入 emitted_tokens/produced_refs：
                # 它是结构活化（功能词）非内容词·不入 collide ctx 保信号质量·lineage 仍记 CUE_SLOT_FILL=3 标血统。
                # gate OFF src 永 != CUE_SLOT_FILL（走 collide 返 LINEAGE_CONCEPT_FILL=1）->has_token 真 token 仍入·bit-identical。
                if has_token and src != CUE_SLOT_FILL:
                    emitted_tokens.append(ref)
            parts.append(OutputPart(unit=unit, words=words,
                                    token_refs=list(emitted_tokens)))
            if _truncated:
                truncated_units.add(unit)
            produced.append(unit)
            # #729 M5 produced_refs FIFO cap：add_produced 保序去重 + 截断保近期（window=PRODUCED_REFS_WINDOW=48·
            # 硬编码守 bit-identical·复用 legacy_v1 add_produced 范式）。cap > 单段 unit 数守去重语义。
            # P0 #1040：gate ON ctx 用 token（produced_refs 唯一消费方 slot_dispatch·collide/sel_pref/pronoun 皆
            # token 级 COOCCURS/selection_pref/pronoun_count·旧填 unit 致错节点=同一 disconnect 族）·OFF 用 unit。
            if emitted_tokens:
                for _t in emitted_tokens:
                    workmem.add_produced(_t)
            else:
                workmem.add_produced(unit)
            # 9d 章节边界（缺口①·反 theater 最小消费者·与阶段1c 同期·doc/重来_篇章结构层级设计 §八）：
            # chapter_seq 变化点作 M5 分页边界候选（既有 LAYER_UNIT_CAP in-memory 分页的章边界补充）。
            # read_chapter_seq 返 None（无标记文本/表未注册）→ chap_no=0·首段建 prev_chapter·其后 0==0
            # 无 boundary·故现存测试零行为变 bit-identical。章边界→ carry_to_workmem 写 prior_topic_refs
            # （#729 章末 anchor·五环闭环：读章边界→触发 carry→写 prior_topic_refs→slot_dispatch 消化）。
            chap = graph.read_chapter_seq(unit)
            chap_no = chap[0] if chap is not None else 0
            if prev_chapter is not None and chap_no != prev_chapter:
                # #729 决断6 caller 端过滤：排除新章首 unit（已 append 进 parts）·只 carry 前章末子集。
                # chap_filter 按 read_chapter_seq(part.unit)[0] == prev_chapter 过滤·确定性（表内容固定）。
                chap_parts = [p for p in parts
                              if _part_chap_no(graph, p) == prev_chapter]
                carry_to_workmem(workmem, chap_parts)
            prev_chapter = chap_no
        # M5 分页：段满 carry + reload（首版 in-memory·真分页 defer·produced_refs 已在线上 add_produced）
        if len(produced) >= LAYER_UNIT_CAP:
            carry_to_workmem(workmem, parts)   # 段满 carry（prior_topic_refs ctx 快照·produced_refs 已 inline）
            # reload_next_layer：首版 in-memory no-op（热区已全载·真分页 Stage 6 接线）

    reached_sink = (dag_path.sink is not None and dag_path.sink in produced)
    return OutputResult(parts=parts, lineage=lineage, reached_sink=reached_sink,
                        truncated_units=truncated_units)


def _part_chap_no(graph: ConceptGraph, part: OutputPart) -> int:
    """读 part.unit 的 chapter_seq 号（None→0·确定性·#729 决断6 caller 端过滤 helper）。"""
    chap = graph.read_chapter_seq(part.unit)
    return chap[0] if chap is not None else 0


def carry_to_workmem(workmem: Any, parts: list[OutputPart]) -> None:
    """M5 分页 carry：snapshot parts.unit 进 prior_topic_refs（FIFO cap + 去重·#729）。

    章边界路径：caller 端过滤本章 parts（chap_no == prev_chapter·排除新章首 unit·决断6）→
      写 prior_topic_refs 章末 anchor（M7 主题锚 stub 字段激活·slot_dispatch:100 ctx_refs 已读）。
    段满路径：传全量 parts（produced_refs 已在线上 add_produced·此处补 prior_topic_refs ctx 快照）。
    prior_topic_refs cap=PRIOR_TOPIC_REFS_WINDOW=16 FIFO 截断保近期（复用 legacy_v1 add_prior_topic 范式）。
    真分页 reload 时热区刷新重置 e=e₀·carry 保留不丢上下文（卷三:74·本 task reload 维持 no-op·决断1）。
    """
    for p in parts:
        # P0 #1040：gate ON part 携 token_refs → carry token 级 prior_topic（collide ctx 同空间·解错节点）。
        # gate OFF token_refs 空 → carry unit（逐字现状·bit-identical）。
        if p.token_refs:
            for _t in p.token_refs:
                workmem.add_prior_topic(_t)
        else:
            workmem.add_prior_topic(p.unit)
