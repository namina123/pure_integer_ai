"""cognition.process.dag_path — 模块4 DAG-path 步进主控（§十四DAG-path 契约）。

dag_path_step(subgraph_edges, seeds, workmem, intent) -> PathResult
  衔接 A2 拓扑分层（模块1）+ A3 PR 重算（模块2·种子 e）+ A4 结构合（模块3·汇聚点消费）+
  attractor 动态演化（模块5·步进到新节点是否加入 e）+ 死路检测（模块6）+ 终点判定。

  terminal ∈ {REACHED_SINK, DEAD_END}
  path = PathData{steps, edges(选定边集存非派生), struct_unit_refs(沿途汇聚点)}
  topo_layers/convergence/source = FULL 相关子图视图（卷三消费识别结构单元）

  j4_closure_check 占位返 true（A1·卷二步进产完整 path 后一次性交卷三 judge·非交替守
    bit-identical·卷三模块3 J4 做真闭合判定）。
  step_budget = 段内拓扑层数 × SAFETY_FACTOR（M4·非全图固定 STEP_LIMIT·避免长文本假死路·
    环靠 Kahn 检测非 STEP_LIMIT）。
  WorkMemory carry：produced_refs 生成侧写（步进侧不写·M3）·pr_vector 本 episode PR 向量。

gate：闭环核心默认 ON（H1）·attractor 动态扩张读 ATTRACTOR_MODE（OFF=e 固定 e₀ bit-identical）。
铁律：无墙钟（步进用 layer 序/order_index 隐含步序非 wall-clock）/ 有限步终止（step_budget
  兜底·DAG 无环）/ 确定性 bit-identical。
诚实边界：步进不判语义路径正确（沿结构边走非"推理对"·stable≠correct）·J4 闭合判定在卷三。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.storage.experience_count import (
    read_effective_freq, record_experience_observe, preload_effective_freq, DEFAULT_CTX_CODE)
from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, Step, IntentType, ConceptRef,
    INTENT_QUESTION, INTENT_COMMAND,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END,
)
from pure_integer_ai.cognition.process.a2_stepper import a2_layer, a2_layer_oi, BLOCKED
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper
from pure_integer_ai.cognition.process.a4_align import coverage_overlap
from pure_integer_ai.cognition.process.dead_end import is_dead_end
from pure_integer_ai.cognition.process.attractor import maybe_expand_attractor, _build_in_degree_seq_map

NodeRef = ConceptRef

# step_budget 安全系数（oracle 标·段内拓扑层数 × 此值·M4）
SAFETY_FACTOR = 4
# 词终止频次阈值（oracle 标·eff_freq=base_freq+e_tn≥此值→通识离群→skip 出边推进·阶段3）。
# 默认 1000 = 既有 e2e fixture e_tn 上限（≤~6/round）的 ≥150× 安全裕度·oracle 标定起点·
# 真训练 run 前 oracle 校准（标到只砍纯功能词·防误伤 sink 必经前置）。
# bit-identical：dag_path_step 跑时 eff_freq 远低 THETA_FREQ（base_freq 首注=0·_inject_base_freq
#   在 stage 循环后 + e_tn 单 run 跨 round 累积 ≤~6/round）→ 不 fire·既有测试 bit-identical。
THETA_FREQ = 1000
# 步进头序（确定性·PRECEDES 先于 CAUSES）
_HEAD_ORDER = (EDGE_PRECEDES, EDGE_CAUSES)


def _goal_coverage(key_skeleton: list[ConceptRef], active: set[NodeRef]) -> int:
    """目标骨架覆盖率（纯整 ×1000·集合覆盖·attractor 第一本职"目标达成"判据·阶段9 S1）。

    = coverage_overlap(key_ids, [active_ids], ordered=False)·active 无序 set → 集合覆盖（非 LCS 序）。
    local_id 口径同 judge J1（ref[1]·judge.py:255）。复用 a4_align.coverage_overlap（judge J1 同款纯整）。
    active = stepper.active（dag_path 主循环 add_active 维护·:190 注释"active 即已到达"）·
      非 struct_unit_refs（后者只 AND 汇聚点·不含 seed/sink/普通链节点·用它会 break key_skeleton=[sink] 既有 e2e）。
    """
    key_ids = [r[1] for r in key_skeleton]
    active_ids = [r[1] for r in active]
    return coverage_overlap(key_ids, [active_ids], ordered=False)


def j4_closure_check(path: PathData, workmem: Any) -> bool:
    """J4 闭合性检查占位（A1·**永久设计占位·非 vestigial·非 defer**·B3 2026-07-03 reframe）。

    **恒返 True 是设计故意的非交替占位**——J4 闭合判定**非逐步**（交替破 bit-identical·
    cognition/result/__init__.py:18）：卷二步进产完整 path 后一次性交卷三 judge·步进侧不判 J4。
    **真 J4 = check_closure(output, dag_path, graph, workmem)**（judge.py:54·judge.py:213 调→设 G4
    veto）·生产 live（JUDGE_MODE 承重件永远 active·reward=ΠG·G4=J4 闭合否决）。
    本占位让步进先产 path（sink-reached 即 REACHED_SINK·J4 留卷三 judge 一次性判）·
    非"待接"·非"遗漏"——步进侧永远不判 J4（架构契约）。
    """
    return True


def _ctx_override(c: NodeRef, workmem: Any) -> int:
    """语境覆写 hook（概念阻断三 gate 之 gate②·word_terminated:121 调）。首版返 0=不覆写。

    精确 role override（当前主题词 vs 通识助词）依赖钥匙① role 分化（第二刀抽象对撞 IS_A LCA 上卷·
    动词抽象层=role 涌现）·defer。episode 级 WM 热区 override 会误 override 通识助词（破坏概念阻断精确性）·
    故首版返 0·待 role 分化落（修正分析二§三"语境 override"）。
    """
    return 0


def _intent_override(c: NodeRef, intent: IntentType, workmem: Any,
                     *, backend: Any = None, edge_store: Any = None) -> int:
    """操作意图覆写 hook（概念阻断三 gate 之 gate③·word_terminated 调·**B-PR3 接通**·doc §18）。

    **B-PR3 接通**（2026-07-12·doc §18·gate ACTION_INTENT_OVERRIDE_MODE）：命令态（intent.type==INTENT_COMMAND）
    动作词（c 有 D:11 PRIMARY 边到 ACTION_*/COMMAND_MOOD concept）→ 返 1（不终止·留 path→dag_path 导向动作拓扑·§13.3）。
    否则返 0（终止如常）。

    **B-PR1 ATTR_OPERATION_INTENT=23 落地推翻了原"defer S10"理由**——ATTR 旗标正是原 docstring 所待"操作 token 标记
    机制"（node operation_intent attr 候选·原 :100）。机制已活（composes_attr·D:11 lookup_word_action readback）·
    不再 defer S10。原"全 token override 破 gate① freq 精度"顾虑亦解——D:11 **per-word** 查找只影响有动作边的词
    （通识词"的"无 D:11 边仍终止）·非全 token override。

    **D:11 per-word 查找**：dag_path 只走 PRECEDES/CAUSES（_HEAD_ORDER·不走 D:11）→ c 是 word concept·其自身不挂
    ATTR_OPERATION_INTENT（旗标在 ACTION_* concept·D:11 target）·故须 lookup_word_action(backend, edge_store, c,
    tier_filter=TIER_PRIMARY) 查 c 的 D:11 边判定（非读 c 自身 ATTR）。命中任一 ACTION_*（1-4）/COMMAND_MOOD（0）→返 1。

    **intent.type==INTENT_COMMAND 闸**（§18.1 决断1·实施者推翻设计审"不查 intent"）：仅命令态 override。理由四重：
    ①语义（函数名/参数 intent/注释"操作意图覆写"全 intent 中心）②一致性（B-PR2 D3 要求 COMMAND §17.1·gate③ 同属
    动作链）③blast radius 仅 COMMAND（更安全）④§13.3 gate③ 导向动作执行·QUESTION 不执行动作·freq 终止合理。

    gate OFF / backend|edge_store None（bare fixture）/ intent≠COMMAND / 无 D:11 PRIMARY 边 → 返 0（bit-identical）。
    **gate③ 只在 gate① freq 通过后触及**（word_terminated eff_freq≥θ_freq 才到 gate③）·fixture eff_freq≪1000→生产 dormant。
    """
    if not getattr(gates, "ACTION_INTENT_OVERRIDE_MODE", False):
        return 0   # gate OFF bit-identical
    if intent.type != INTENT_COMMAND:
        return 0   # 决断1 intent 闸：仅命令态 override（QUESTION 不路由动作执行·§18.1）
    if backend is None or edge_store is None:
        return 0   # 无 backend/edge_store 退化（bare fixture·caller 未穿）
    from pure_integer_ai.cognition.shared.action_primitives import lookup_word_action
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    hits = lookup_word_action(backend, edge_store, c,
                              space_id=c[0], tier_filter=TIER_PRIMARY)
    return 1 if hits else 0


def word_terminated(c: NodeRef, workmem: Any, backend: Any, *,
                    intent: IntentType,
                    theta_freq: int = THETA_FREQ,
                    ctx_code: int = DEFAULT_CTX_CODE,
                    edge_store: Any = None,
                    eff_freq_lookup: Any = None) -> bool:
    """experience_count 第一消费者·通识词终止判定（三 0/1 gate·全 1=skip·真行为变·阶段3）。

    落 dag_path 主循环（非 attractor entry·与 maybe_expand_attractor 正交分函数·§八-bis）。
    纯读：不写 backend / 不改 stepper.active（dag_path 主控据返回值决定 add_active）。
    True → dag_path 不 stepper.add_active(c) → 后继因前驱缺失 BLOCKED → 遍历后继时 is_dead_end
    主动判 DEAD_END → 通识词出边不进 path.edges。

    三 0/1 gate（修正分析二§三·experience_count 指引§点5/§八-bis）：
      terminate = (effective_freq(c, ctx_code) ≥ θ_freq)    # ①经验频次离群（base 通识 (0,0) 桶 + e_tn 当前 ctx_code 桶·复合 key 第二刀按桶读·防混淆频次）
                × (ctx_override(c) == 0)           # ②语境覆写（hook·首版 0）
                × (intent_override(c) == 0)        # ③操作意图覆写（B-PR3 接通·命令态 D:11 动作词 override 不终止·doc §18）
    sink 保护：c == intent.sink → 永不终止（达 sink 语义·防断路径）。
    诚实边界：PRECEDES AND 汇聚点前驱被 skip → 全图 DEAD_END 是诚实降级（通识词本不参与路径·非 bug·T7 覆盖）。
    gate 状态：gate① freq 默认活（THETA_FREQ=1000）/ gate② ctx_override 首版 0（defer attractor 本职/记忆空间·
      待消费者接）/ gate③ intent_override **B-PR3 接通**（命令态 intent.type==COMMAND + D:11 动作词 override·
      doc §18·gate ACTION_INTENT_OVERRIDE_MODE）。gate OFF / fixture eff_freq≪1000 → gate③ 生产 dormant
      （同 B-PR2 范式·反 theater 须直调 _intent_override 单测验真机制·test_action_intent_override）。
    seed 节点：已 in stepper.active（a2_stepper init·active=seeds）·word_terminated(seed)=True 幂等
      （seed 已 active·不调 add_active 不影响）·后继正常推进（seed→child 在 child 遍历时进 path）。
    theta_freq 前置 >0（=0 会让所有非 sink node skip·整机 DEAD_END·生产默认 1000 不触发·caller 责任）。
    """
    assert_int(c[0], c[1], theta_freq, _where="word_terminated.c")
    if intent.sink is not None and c == intent.sink:
        return False   # sink 保护：达 sink 永不终止
    # perf round3 候选1：dag_path_step 入口预加载 lookup（dict 替 per-node select）·None 回退 live read。
    # lookup 由 caller preload_effective_freq 建（snapshot observe_mode + pre-dag_path_step 状态）·
    # bit-identical 安全见 preload_effective_freq docstring（(sid,lid,ctx,sp) 唯一 + 读先于 own write）。
    if eff_freq_lookup is not None:
        eff_freq = eff_freq_lookup(c)
    elif backend is not None:
        eff_freq = read_effective_freq(backend, c, ctx_code=ctx_code,
                                       observe_mode=bool(getattr(gates, "FREQ_OBSERVE_MODE", False)))
    else:
        eff_freq = 0
    if eff_freq < theta_freq:
        return False   # gate① 经验频次不足
    if _ctx_override(c, workmem) != 0:
        return False   # gate② 语境覆写关停
    if _intent_override(c, intent, workmem,
                        backend=backend, edge_store=edge_store) != 0:
        return False   # gate③ 操作意图覆写关停
    return True


def _step_budget(topo_layers: list[list[NodeRef]], safety_factor: int) -> int:
    """步数上限 = 段内拓扑层数 × SAFETY_FACTOR（M4·非全图固定 STEP_LIMIT）。"""
    return max(1, len(topo_layers)) * safety_factor


def dag_path_step(subgraph_edges: list[dict[str, Any]],
                  seeds: list[NodeRef], workmem: Any,
                  intent: IntentType, *, current_seq: int = 0,
                  memory_active: bool = False,
                  safety_factor: int = SAFETY_FACTOR,
                  backend: Any = None,
                  edge_store: Any = None,
                  theta_freq: int = THETA_FREQ,
                  ctx_code: int = DEFAULT_CTX_CODE,
                  key_skeleton: list[ConceptRef] | None = None,
                  coverage_threshold: int = 0) -> PathResult:
    """DAG-path 步进主控。返 PathResult（terminal ∈ {REACHED_SINK, DEAD_END}）。

    subgraph_edges：热区相关边集（卷一建好·audit_float==0·caller 加载·M5 分页 defer）。
    seeds：attractor 种子集 e₀（语境集 + intent sink·§十一）。
    workmem：WorkMemory（pr_vector 本 episode 填·produced_refs 步进侧不写 M3）。
    intent：意图（sink 终点判定·§十一缺口#2）。
    current_seq：当前 timestamp_seq（OCCURRENCE 衰减用·无墙钟）。
    memory_active：ZERO_AI_MEMORY_ACTIVE（True 时 OCCURRENCE REFERS_TO 进 PR 邻接·Stage 5）。
    backend：可选（attractor node tier 判据用）。
    key_skeleton：目标骨架 ConceptRef 集（attractor 第一本职"目标达成"判据·§十一缺口#2·
      legacy `doc/吸引子动力学与目标达成.md`§二.1"生成过程内在停止条件"·阶段9 S1）。
      非空 + coverage_threshold>0 启用覆盖率停止判据（路径已到达节点覆盖目标骨架≥阈值→REACHED_SINK）。
      None/空/threshold=0 → 退化既有 sink 节点判定（bit-identical·既有测试零 break）。
      不挂 IntentType（职责正交·key_skeleton 输入层属性 types.py:160 / IntentType 意图属性·
      judge 已以 input_payload.key_skeleton 单一来源·塞 intent 造两处来源）。
    coverage_threshold：覆盖率达成阈值（0..1000·0=退化不启用·生产 oracle 标定起点如 500·
      真训练 run 前校准·同 THETA_FREQ 范式）。
    """
    assert_no_float(current_seq, safety_factor, _where="dag_path_step")
    for s in seeds:
        assert_int(s[0], s[1], _where="dag_path_step.seed")

    # —— A2 拓扑分层（模块1） ——
    e_set: set[NodeRef] = set(seeds)
    local_seeds = list(seeds)
    # #728 B 半：dag_path 入口 local_seeds 扩张（replay_candidates 作额外种子·路径选择连贯·footnote4）。
    # gate OFF → workmem.replay 永空 → local_seeds == seeds → bit-identical（replay 空时跳过 subgraph_nodes 遍历·零开销）。
    # 过滤当前 subgraph 节点集（历史 sink 不在当前子图则不加·避孤立 seed 污染 PR·Agent B 判决）。
    # replay 存 info_ref concept ref（tri_space 纠偏 A·过滤 (0,0) 退化行已 in tri_space）。
    # 弱有效：topo_layers 更宽（历史成功 sink 作初始 active → Kahn 分层起点更多）+ PR 偏向（solve 多 seed）→
    # 路径层序变 → generate 输出变。stable≠correct（#479 墙·统计倾向非语义正确）。
    replay = getattr(workmem, "replay_candidates", [])
    if replay:
        _seed_set = set(seeds)
        subgraph_nodes = {(e["space_id_from"], e["local_id_from"]) for e in subgraph_edges}
        subgraph_nodes |= {(e["space_id_to"], e["local_id_to"]) for e in subgraph_edges}
        for r in replay:
            if r not in _seed_set and r in subgraph_nodes:
                local_seeds.append(r)
                e_set.add(r)
    # B-PR4 动作词种子偏向（doc §19·mirror #728 replay 扩张·gate ACTION_SEED_BIAS_MODE·subgraph_nodes 过滤）：
    # formal_train _run_reward_round 预算动作词 word_ref（D:11 源端·_collect_action_seed_candidates 洗净 sn==0 + rate-sort
    # 降序）→ workmem.action_seed_candidates → 此处读 + 过滤当前子图节点集 + append local_seeds/e_set →
    # solve(local_seeds) PR 偏向动作拓扑邻域（§13.3·复用 attractor 不改数学·maybe_expand_attractor 用扩张后 e_set）。
    # **种子=动作词概念（D:11 源端 word concept·语言域 token·在 PR matrix）·非 ACTION_* 元概念**（D:11 目标端 target·
    # 只有 D:11 边→不在 PR matrix→add_seed no-op theater·doc §19.0 矛盾 A）。
    # gate OFF → formal_train 不预算 → action_seed_candidates 空 → `if action_seeds:` 假 → 跳过 → local_seeds==seeds → bit-identical。
    if bool(getattr(gates, "ACTION_SEED_BIAS_MODE", False)):
        action_seeds = getattr(workmem, "action_seed_candidates", [])
        if action_seeds:
            subgraph_nodes = {(e["space_id_from"], e["local_id_from"]) for e in subgraph_edges}
            subgraph_nodes |= {(e["space_id_to"], e["local_id_to"]) for e in subgraph_edges}
            for word_ref in action_seeds:   # 已按率降序（_collect_action_seed_candidates 洗净+sort）
                if word_ref not in e_set and word_ref in subgraph_nodes:
                    local_seeds.append(word_ref)
                    e_set.add(word_ref)
    head_types = {EDGE_PRECEDES, EDGE_CAUSES}
    # S2 dead-end factor C 修（gate PRECEDES_OI_MODE·F2）：language PRECEDES 概念成环·Kahn 丢环节点含 sink
    # -> sink 永不可达 -> DEAD_END -> reward 恒 0。gate ON + 子图有 PRECEDES 边（language）->
    # a2_layer_oi（弃 Kahn·oi-first-occ 序遍历·含环节点）；否则 a2_layer（Kahn·arith/code 无 PRECEDES 边·
    # oi-layering 无意义会堆单层破 CAUSES 前向·须 Kahn·bit-identical）。OI_MODE implies OR（a2_stepper）。
    _has_precedes = any(e["edge_type"] == EDGE_PRECEDES for e in subgraph_edges)
    if bool(getattr(gates, "PRECEDES_OI_MODE", False)) and _has_precedes:
        topo_layers, convergence, stepper = a2_layer_oi(subgraph_edges, e_set, head_types)
    else:
        topo_layers, convergence, stepper = a2_layer(subgraph_edges, e_set, head_types)
    source = topo_layers[0][0] if topo_layers else None

    # —— A3 PR 重算（模块2·种子 e） ——
    pr_wrapper = A3PRWrapper.build(subgraph_edges, current_seq=current_seq,
                                   memory_active=memory_active,
                                   backend=backend, ctx_code=ctx_code,
                                   seeds=local_seeds)
    x = pr_wrapper.solve(local_seeds)   # #728 local_seeds 含 replay 扩张（gate OFF == seeds·bit-identical）
    workmem.pr_vector = x   # 本 episode PR 向量（Episode 聚合/防塌柱③方差读·F5）

    # —— 防塌柱③ proactive probe（EXPLORATION_MODE ON + PR 方差趋平时注入新种子·③最小版·F5） ——
    # 注入落步进侧（pr_wrapper/e_set 局部于此·post-episode caller 取不到·故此 proactive 而非事后）。
    # EXPLORATION_MODE 默认 OFF→不注入 bit-identical·ON 才 proactive（柱③ falsifiable 的真注入半边）。
    exploration_injected = False
    if bool(getattr(gates, "EXPLORATION_MODE", False)):
        from pure_integer_ai.cognition.result.anti_collapse import (
            integer_variance, inject_seeded_exploration, deterministic_seed, THETA_VARIANCE)
        if integer_variance(x) < THETA_VARIANCE:
            seed = deterministic_seed(current_seq, intent.sink)
            if inject_seeded_exploration(pr_wrapper, e_set, seed) is not None:
                exploration_injected = True
                workmem.pr_vector = pr_wrapper.snapshot()   # 注入后 x 变·重设（F5 读扩张后 x）

    path = PathData()
    budget = _step_budget(topo_layers, safety_factor)
    attractor_on = bool(getattr(gates, "ATTRACTOR_MODE", False))
    # F2 acyclic fix：显式 query-local 附加 seed 可能带入有入边节点，初始 active 必须保持无环处理。
    # candidate（struct_ref[i]·i>0·**有 inter-seg in-edge**）。replay seed 进 active 致其 in-edge（last_token[i-1]
    # ->struct_ref[i]）与 out-edge（struct_ref[i]->token_0[i]·后继访时收）组合成环 -> generate.py:109 _path_acyclic
    # crash。OI_MODE ON -> seed 节点跳过 advance（不收 pred 边）·seed 已 active（初始）无须 add_active·其 out-edge
    # 由后继访时收·in-edge 不收 -> 破环。EXPLORATION/ATTRACTOR 加 e_set 不加 stepper.active（stepper :202 构造后
    # copy·后续 e_set 扩张不回传）-> 不入 initial_active -> 不跳过。gate OFF -> initial_active 不用（bit-identical）。
    initial_active = set(stepper.active)
    oi_mode = bool(getattr(gates, "PRECEDES_OI_MODE", False))
    # perf round3 候选1：word_terminated read_effective_freq 批量化——入口预加载 experience_count 两桶
    # → dict lookup（替 per-node 2 select·item 656 时 4-6 万节点 per-node select 是 O(N²) 常数因子）。
    # backend None（单测退化）→ None 回退 live read·bit-identical。安全见 preload_effective_freq docstring。
    _eff_freq_lookup = (preload_effective_freq(
                            backend, ctx_code=ctx_code,
                            observe_mode=bool(getattr(gates, "FREQ_OBSERVE_MODE", False)))
                        if backend is not None else None)
    # perf round3（2026-07-13）：attractor _in_degree_seq 预算 map（解每 node 全扫 O(n²)·cProfile n=4 top self 23.3s/12%）。
    # dag_path_step 建一次传 maybe_expand_attractor·O(1) 查替每调用全扫。attractor_on OFF 不建（零开销·守 bit-identical）。
    _in_degree_map = (_build_in_degree_seq_map(subgraph_edges) if attractor_on else None)

    for layer_idx, layer in enumerate(topo_layers):
        for node in layer:
            # —— 阶段3 第一消费者：通识词终止 skip（eff_freq≥θ_freq 且无覆写→不 add_active→
            #     后继前驱缺失 BLOCKED→DEAD_END·通识词出边不进 path.edges·真行为变） ——
            if word_terminated(node, workmem, backend, intent=intent,
                               theta_freq=theta_freq, ctx_code=ctx_code,
                               edge_store=edge_store,
                               eff_freq_lookup=_eff_freq_lookup):
                continue   # node 级 skip（所有 head 不跑·node 永不 active）
            # F2 acyclic fix：OI_MODE ON + node 是初始 seed（含 replay candidate·有 in-edge）-> 跳过 advance
            # （不收 pred 边·破 replay seed in-edge+out-edge 环）。seed 已 active·其 out-edge 由后继访时收。
            if not (oi_mode and node in initial_active):
                _node_activated = False   # once-per-node observe_tn（避多头成功过计·方案3 tn路）
                # —— 按头步进（模块1 stepper·返选定边集） ——
                for head in _HEAD_ORDER:
                    if head not in head_types:
                        continue
                    sel = stepper.advance(node, head)
                    if sel is BLOCKED:
                        continue   # CAUSES OR 零 active 前驱·正确停滞
                    if sel:   # 非空=有选定边（AND 全到齐 / OR 选高优先）
                        path.steps.append(Step(node, head, list(sel)))
                        path.edges.extend(sel)
                        if (node, head) in convergence and node not in path.struct_unit_refs:
                            # F8 汇聚点 ref 收集（裸 append·非 fold·a4_align 函数已删 2026-07-07·
                            # process-time 汇聚点 consensus 折叠首版无消费场景·fold 由件4 lang_structure_align
                            # discover-time cover·此处仅收集 convergence node ref 供 generate structure_units + reward feed 消费）
                            path.struct_unit_refs.append(node)
                        stepper.add_active(node)   # active 即"已到达"·chains 得以传播
                        _node_activated = True
                        # M3：步进侧不写 produced_refs（语义=已输出·生成侧写·J4 carry 期望"已输出"）
                        # —— attractor 动态演化（模块5·松入严留·新种子 A3_add_seed 在模块5内调·D6 防双加） ——
                        if attractor_on:
                            maybe_expand_attractor(node, e_set, pr_wrapper,
                                                   subgraph_edges, workmem,
                                                   backend=backend,
                                                   in_degree_map=_in_degree_map)
                            # item3 缺漏4：attractor 扩张后 x 变·重设 pr_vector（F5 聚合读扩张后 x·非扩张前）
                            workmem.pr_vector = pr_wrapper.snapshot()
                # 方案3 tn路（B4 β_arith 修法）：决策时 observe_tn 写·once per node per dag_path_step
                # ·sign-agnostic·gate FREQ_OBSERVE_MODE 守·喂 word_terminated consumer（读 path 节点 eff_freq）。
                # 多头成功仅记一次（_node_activated flag·避 PRECEDES+CAUSES 双头过计）·
                # gate OFF → 不写 → bit-identical·e_tn reward feed 路径不变。
                # **时序不变量（perf round3 候选1 lookup 依赖·对抗审1）**：此写须在 word_terminated(node) 读
                # （:307）+ attractor _seed_weight 读（:336 live·同 node）**之后**。word_terminated 走入口预加载
                # snapshot（pre-write）·若此写移到读之前→snapshot 与 live post-write 散开→bit-identical 破。
                # 每 node topo 访一次（Kahn/OI）+ 写 ref=node（本节点行·跨节点不互扰）守此不变量。
                if (_node_activated and backend is not None
                        and bool(getattr(gates, "FREQ_OBSERVE_MODE", False))):
                    record_experience_observe(backend, ref=node, ctx_code=ctx_code)
            # —— 死路检测（模块6·三条件·D7 intent+active） ——
            if is_dead_end(node, subgraph_edges, intent, stepper.active,
                           len(path.steps), budget, stepper=stepper):
                return PathResult(path, TERMINAL_DEAD_END, None,
                                  topo_layers, convergence, source,
                                  exploration_injected=exploration_injected)
            # —— 终点判定（达 sink·attractor 第一本职"目标达成"控制环收敛判据·阶段9 S1） ——
            # 退化（key_skeleton 空 / threshold=0）：达 sink ∧ J4 闭合 → REACHED_SINK（既有·bit-identical）。
            # 启用（key_skeleton 非空 + coverage_threshold>0）：达 sink 时覆盖率主导——
            #   路径已到达节点(stepper.active) 覆盖目标骨架≥阈值 → REACHED_SINK（真达成）·
            #   覆盖不足 → DEAD_END（走到 sink 但骨架未覆盖够=未达成·反 theater 主锚·T2）。
            #   Q/C 活·STATEMENT 不活（陈述不步进取证·types.py:208）。走到 sink 即停（同退化终止点）。
            # why 达 sink 判非每-node 提前停/非层尽：每-node 提前停 source(active={seed}) 过早 50% 触发 +
            #   path 未含 CAUSES 边→reward feed 断；层尽走过头（sink 后节点）触发 attractor K_CAP_SOFT
            #   溢出 min(Rational) 阶段8 latent。达 sink = 既有终止点·path 含到 sink 全步进边（feed 完整）+ 不走过头。
            if intent.sink is not None and node == intent.sink:
                if key_skeleton and coverage_threshold > 0:
                    if intent.type in (INTENT_QUESTION, INTENT_COMMAND) \
                            and _goal_coverage(key_skeleton, stepper.active) >= coverage_threshold:
                        return PathResult(path, TERMINAL_REACHED_SINK, node,
                                          topo_layers, convergence, source,
                                          exploration_injected=exploration_injected)
                    return PathResult(path, TERMINAL_DEAD_END, None,
                                      topo_layers, convergence, source,
                                      exploration_injected=exploration_injected)
                if j4_closure_check(path, workmem):
                    return PathResult(path, TERMINAL_REACHED_SINK, node,
                                      topo_layers, convergence, source,
                                      exploration_injected=exploration_injected)
            # —— 步数上限兜底（M4③·段内层数×安全系数·环靠 Kahn 检测非此） ——
            if len(path.steps) >= budget:
                return PathResult(path, TERMINAL_DEAD_END, None,
                                  topo_layers, convergence, source,
                                  exploration_injected=exploration_injected)
    # 层尽未达 sink = 死路
    return PathResult(path, TERMINAL_DEAD_END, None,
                      topo_layers, convergence, source,
                      exploration_injected=exploration_injected)
