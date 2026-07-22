"""算术结构发现、识别和 held-out 泛化验证 runtime。"""
from __future__ import annotations

from typing import Sequence

from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveredOperator,
    Recognition,
    _collect_cue_sig,
    _collect_slot_lcas,
    _normalize_abstract_sig,
    auto_discover_operators,
    recognize_operators,
    route_samples_for_discovery,
    shape_signature,
)
from pure_integer_ai.cognition.shared.types import ConceptRef, MODALITY_ARITH
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.train_result_types import GeneralizationSummary
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.op_confidence import record_op_outcome
from pure_integer_ai.training.vm_proof import execute_composes_value

_DISC_ROOT_SEED = "formal_train.disc_src"


def _discover_and_recognize_arith_operators(
        ctx: TrainContext,
        corpus: list[CollectedItem],
        *,
        existing_operators: Sequence[DiscoveredOperator] = (),
        ) -> tuple[list[DiscoveredOperator], list[Recognition], GeneralizationSummary]:
    """序列6-min 发现（WRITE）+ 序列3-min 识别（READ）+ 验证半闭环（vm_proof 验泛化）生产触发器：
    算术语料 → 内容哈希独立根 → per-shape 留 held-out → auto_discover_operators + recognize_operators
    + _verify_generalization。

    **序列6-min（§八.6·de-theater 序列1·WRITE）**：discover_skeleton 从"仅 tests 调"升"真生产 caller"。
    绕两个 observe 既有限制（非本步引入·doc §八.6 诚实边界②）：① observe 多 arith item 撞同一 __seg_{stage}_0
    struct_ref；② EdgeStore.add 纯 append-only 不去重。故按 **arith_source 内容哈希**建独立根 __disc_src_{h63}
    （同程序同根·不同程序不同根·跨 run 幂等）·已建有 COMPOSES 出边的根 skip（防重 build 复制边）。

    **序列3-min（§八.3·生产期 READ 消费·"新样本命中已学骨架"）**：per-shape 留 held-out——同形 ≥
    MIN_DISCOVER_SAMPLES+1（≥3）→ 发现首 K 个·识别余（held-out 新实例）；同形 ∈ [K, K+1)（==2）→ 全发现·无
    held-out；同形 <K → 不发现。识别 held-out 新输入（**非发现样本集→真泛化·非循环 theater**：骨架从 {5,6}
    学·识别 {7,8} 新输入·设计 line193 "新样本命中已学骨架" 本意）。
    **序列6-min 进化**：序列6-min 在全语料发现·序列3-min 进化为 per-shape 留 held-out（识别须新输入·全语料
    发现+识别同集=循环 theater·反 §8.7）。2 样本语料（同形==2）行为不变（全发现·序列6-min 既有测仍过）。

    **验证半闭环（§8.7·反 theater + 学到能力证据）**：识别产物 recognitions 不再 terminal——_verify_generalization
    对每个识别做 caller 级 vm_proof（execute_composes_value）：骨架绑识别 params 执行 == held-out 新输入执行值。
    识别=结构对齐·vm_proof=VM 执行比对·两路独立计算——**诚实**：对正确识别同值是构造性预期（结构同构·非惊奇交叉验证）·
    vm_proof 真"牙"是抓获 PARAM 阅读序错位/编译发散/shape 漏判异配（probe SUB 错参 -47≠43 不 verified）·重执行本身即真
    READ+应用消费（非 theater·非死写）。verified/total_held_out = 泛化率（学到的能力覆盖多少 held-out 新输入·直接量化
    "学到能力"·解"不能从语料学到能力"根因侧证）。

    返 (discovered, recognitions, generalization)（formal_train 写 FormalTrainResult.discovered_operators/
    recognitions/generalization·可观测 + 反 theater 锚点）。生产路径：build_composes_from_arith（真 builder）→
    auto_discover_operators（group + discover_skeleton + register·WRITE）→ recognize_operators（held-out 读骨架
    抽 PARAM 绑定·READ）→ _verify_generalization（vm_proof 验骨架绑参复现 held-out 新输入值·消费 recognitions）。
    """
    from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
    from pure_integer_ai.storage.edge_store import SOURCE_MATH
    from pure_integer_ai.storage.node_store import NODE_CONCEPT

    arith_items = [it for it in corpus
                   if it.modality == MODALITY_ARITH and it.arith_source]
    if not arith_items:
        return [], [], GeneralizationSummary()   # 无算术语料 → 无发现/识别/泛化（空汇总·诚实）
    # 序列7 跨 run READ：existing_operators = 已载发现算子（**caller 传**·formal_train resume load_run 后经
    # load_discovered_operators 取·非本函数内查 backend）→ 避免同一 backend 内两次调把首调注册算子误当"载入"
    # （会循环识别重喂发现集=theater）。默认 () → 直调单测（不传）bit-identical 旧为（无跨 run 识别）。
    existing_ops = list(existing_operators)
    # Half B（§八.7②·Finding1 真修）：路由键 (sig, arity) **非 sig-only** —— square(arity1) 载入后 mul(arity2)
    # 同形异 arity 仍须独立发现（原 sig-only 路由把同形全送识别→mul 被 square 拒/坍缩→静默丢）。existing_keys =
    # 载入算子 (sig,arity,abstract_sig)·existing_sigs = fallback（<K 或 probe None 时按 sig 认载入→识别候选）。
    # **B6 Bug 2 修（2026-07-06·聚类前置）**：existing_keys 加 abstract_sig 维——从 op.skeleton_ref 经
    # _collect_slot_lcas 重建（同 LOAD 端 Bug 1 修法）+ _normalize_abstract_sig 归一。解 resume 时同 (sig,arity)
    # 异 abstract_sig 新样本误判"已载"静默丢→新抽象类本轮不发现。arith abstract_sig 恒 ()（无 CONCEPT_LEAF）→
    # 载入算子 abstract_sig 亦 () → (sig,arity,()) 与原 (sig,arity) 等价·bit-identical。
    existing_keys: set[tuple[tuple[int, ...], int, tuple, tuple]] = set()
    existing_sigs: set[tuple[int, ...]] = set()
    for op in existing_ops:
        op_sig = tuple(shape_signature(ctx.concept_graph, op.skeleton_ref))
        op_asig = _normalize_abstract_sig(_collect_slot_lcas(
            ctx.backend, ctx.concept_graph, op.skeleton_ref))
        op_cue = _normalize_abstract_sig(_collect_cue_sig(
            ctx.backend, ctx.concept_graph, op.skeleton_ref))   # §十八 condition 6a-3：cue_sig 第4维（镜像 abstract_sig·gate OFF 全 None→()→bit-identical·gate ON 是/使 异键独立路由）
        existing_keys.add((op_sig, op.arity, op_asig, op_cue))
        existing_sigs.add(op_sig)
    # 内容哈希独立根（同程序同根幂等·不同程序不同根·绕 observe struct_ref 碰撞 + EdgeStore.add 不去重）
    roots: list[ConceptRef] = []
    for item in arith_items:
        h = Hasher(_DISC_ROOT_SEED).h63(item.arith_source)
        root = ctx.concept_index.ensure(
            f"__disc_src_{h}", space_id=ctx.space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 幂等：已建有 COMPOSES 出边 → skip（EdgeStore.add 不去重·防重 build 复制边 corrupt 树）
        if not ctx.edge_store.query_from(root[0], root[1], edge_type=EDGE_COMPOSES):
            build_composes_from_arith(
                item.arith_source, concept_index=ctx.concept_index,
                edge_store=ctx.edge_store, backend=ctx.backend,
                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=root)
        roots.append(root)
    # 路由（聚类前置·B6 Bug 2+3·2026-07-06）：按 (sig,hint) 分组 → 每组 LCA 聚类 → 按簇 abstract_sig 路由
    # discover/recognize（解 existing_keys 缺 abstract_sig 致跨 run 覆盖渐失 + cluster-blind held-out 致混合簇不发现）。
    # arith 首 sample 无 CONCEPT_LEAF → 聚类单簇 None → abstract_sig=() → 路由键 (sig,arity,()) 与原 (sig,arity)
    # 等价·bit-identical。helper 详 structure_discover.route_samples_for_discovery。
    discover_roots, recognize_roots = route_samples_for_discovery(
        ctx.backend, ctx.concept_graph, roots,
        existing_keys=existing_keys, existing_sigs=existing_sigs,
        space_id=ctx.space_id)
    discovered = auto_discover_operators(
        discover_roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=ctx.space_id, source=SOURCE_MATH)
    # 序列7：all_ops = 载入（跨 run）+ 本 run 新发现。识别/验证两路皆对 all_ops（载入算子可识别新输入=跨 run 泛化）。
    all_ops: list[DiscoveredOperator] = list(existing_ops) + discovered
    if not all_ops or not recognize_roots:
        # 无已知算子 / 无 held-out → 无可识别/泛化（诚实·不伪造）。返 discovered（本 run 新发现仍上报）。
        return discovered, [], GeneralizationSummary()
    recognitions = recognize_operators(
        recognize_roots, discovered_operators=all_ops,
        backend=ctx.backend, space_id=ctx.space_id)
    # 验证半闭环（§8.7）：vm_proof 独立验每个识别（骨架绑参复现 held-out 新输入值）→ 泛化汇总。
    # 识别产物 recognitions 在此被真消费（解 terminal 边界·反 theater）。total_held_out=len(recognize_roots)。
    generalization = _verify_generalization(
        ctx, recognitions, all_ops, total_held_out=len(recognize_roots))
    return discovered, recognitions, generalization




def _verify_generalization(ctx: TrainContext, recognitions: list[Recognition],
                           discovered_operators: list[DiscoveredOperator],
                           *, total_held_out: int) -> GeneralizationSummary:
    """序列3-min 验证半闭环：识别 → caller 级 vm_proof 独立验泛化（骨架绑参 == held-out 输入值）→ 汇总。

    **反 theater + 学到能力证据**：发现骨架从发现集学到·识别 held-out 新输入·vm_proof 独立**重新执行**骨架
    （绑识别 params）确认复现新输入值。识别 = 结构对齐（recognize_operators._align_walk）·vm_proof = VM 执行
    比对（execute_composes_value）·两路独立计算。**诚实定位**（对抗审计）：对正确识别·骨架与输入结构同构（_align_walk
    固定位值等门保）→ VM 把 LOAD mv_i(绑值 v) 与 PUSH_IMM v 等同 → 同值是**构造性预期**（非惊奇交叉验证）。
    vm_proof 真"牙"=抓获 PARAM 阅读序错位 / skeleton 编译发散 / shape 漏判结构异配（probe：SUB 错参→-47≠43
    不 verified）→ 非恒真 stub。重执行本身 = 真 READ+应用消费（非 theater·非死写）。verified/total_held_out =
    泛化率（学到的能力覆盖多少 held-out 新输入·量化"学到能力"·解"不能从语料学到能力"根因侧证）。

    caller 级 vm_proof（L8 调 L7 execute_composes_value·守 recognize_operators L5 不调 L7 单向依赖）。
    **operand-input 识别**（rec.is_operand_input=True）：input 含 OPERAND 叶·按 operand_binding 反演 input 探针
    （input_probe[in_slot]=param_values[skel_slot]）执行比对——探针纯从 Recognition 字段反演无须 import 常量。
    诚实定位同 immediate：探针比对构造性（结构同构→同值）·真牙在 recognize_operators 变量同一性判定·探针=重执行消费。
    诚实：vm_proof None（非 COMPOSES 根/StepLimit）→ 不计 verified（不伪造·保留 recognized 计数）。
    """
    assert_int(total_held_out, _where="_verify_generalization.total_held_out")
    graph = ctx.concept_graph
    op_by_name = {op.name: op for op in discovered_operators}
    # 刀2 件6 防双计（doc §5 刀2 点3·caller 责任）：summary recognized/verified 计 **distinct input_root**
    # （同 root 多解析都验算 1 root·非每 rec +1）·守 rate_permille≤1000（recognized≤total_held_out）。
    # **op_confidence 半环 per-op 不双计**（循环体 record_op_outcome 不变·每 rec → distinct op name_ref 各 +sn/+tn）。
    verified_roots: set[ConceptRef] = set()
    for rec in recognitions:
        op = op_by_name.get(rec.operator_name)
        if op is None:
            continue   # 识别指向未知算子（理论不该发生·防御跳过·不计 verified·不写置信度）
        v_skel = execute_composes_value(graph, op.skeleton_ref, rec.param_values)
        if rec.is_operand_input:
            # operand-input 识别（探针值执行比对）：input 含 OPERAND 叶·用 rec.input_probe_values 直接执行 input
            # （连续 slot-序探针·_align_extract 派生·含未用 slot·消除反演洞·对抗审计 F1）·无须 import 探针常量。
            v_input = execute_composes_value(graph, rec.input_root, rec.input_probe_values)
        else:
            v_input = execute_composes_value(graph, rec.input_root, ())   # immediate 输入=nullary（既有路径）
        eq = (v_skel is not None and v_input is not None
              and rational.eq(v_skel, v_input))
        if eq:
            verified_roots.add(rec.input_root)   # distinct root（刀2 防双计·非 verified+=1）·泛化得证（两路独立·反 theater）
        # §8.7-洗 洗净循环反馈半闭环：vm_proof 验结果写算子置信度（op_confidence sn/tn/strength）→
        # recognize_operators 择优读（滤非泛化算子=洗净）·解 recognitions terminal·反 theater 半环。
        # verified→sn++&tn++&strength+=1 / not-verified（mismatch/deadloop None）→tn++ only（R1 符号·sn 单调）。
        # **刀2 多解析**：同 input_root 多 rec（异 op）→ 每 op 各记 op_confidence（per-op·非双计·distinct name_ref）。
        if op.name_ref != (0, 0):
            record_op_outcome(ctx.backend, ref=op.name_ref, verified=eq)
    return GeneralizationSummary(
        total_held_out=total_held_out,
        recognized=len({rec.input_root for rec in recognitions}),   # 刀2 distinct（非 len·防双计）
        verified=len(verified_roots))   # 刀2 distinct verified roots（非 verified 计数）

__all__ = [
    "_discover_and_recognize_arith_operators",
    "_verify_generalization",
]
