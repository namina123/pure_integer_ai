"""floor 端到端下游激活率·纯读测量（断奶 critical path 第 2 件·反 theater 首版机制层预验）。

档 `doc/重来_floor_端到端下游激活率_2026-07-17.md`。承 readback→generation 桥（DONE·四 reader 在 graph_view）。

命门：reward-ratio floor（weaning._floor_ok）测 4 能力指标平台 + oov_promote 域错恒卡阈·**不测端到端 cue↔rel 下游激活**
（floor 的真语义）。桥使生成侧消费学到的 D:11（cue_rel_of 读 v2 tally→promote 的 W→REL_*·source==SOURCE_BARE_TEXT·
非 boot 闭包）→ held-out R-skeleton cue slot 的对应词激活**可测** → 反 theater 真信号替 reward-ratio 假腿。

**本模块 = 读侧后验重导**（不改 generate 写侧·bit-identical）：对 held-out OutputResult.parts 逐 unit 镜像
generate.py:153-171 per-unit stash 逻辑（read_instantiates→rel_kind_of_skeleton→read_cue_sig→cue slot 集）·
读 cue slot 选词的 cue_rel_of 是否 == skeleton rel_kind → activation_rate + false_positive_rate + measured-guard。

**单向依赖（审1 MEDIUM-1）**：本模块**只 import graph_view 四 reader + types**·**不 import cognition/process·
不 import understanding**·守 result→process/L5→L0 单向依赖（同 graph_view.read_cue_sig 复制而非 import structure_discover 范式）。

★ **不变量锁（审2 LOW-2）**：本模块**纯读不写 D:11**（非自证·cognition/result/ 零 D:11 写点）。held-out 学到的 D:11
由训练侧 tally→promote 建（formal_train 训练路径 :3632 caller·auto_discover_operators **不含 tally**）。orchestrator
（formal_train._measure_floor_pass）observe+discover held-out 时**禁止调 tally_cue_slot_matches**——否则 held-out_root
进 structure_match_count → (cue 词, held-out-instance) pairing 不 disjoint → activation 虚高 = theater。

**诚实边界（审2 MEDIUM-1）**：首版测「学到的 W 在 held-out cue slot 正确激活」（D:11 学习链端到端可读 + C-vs-L 真判别）·
**非测「bonus 真驱动 vs collide 驱动」**（后者 closure-only 消融〔CORRESPONDENCE_SLOT_MODE OFF 跑 floor 比 margin〕
defer Phase F §十二 前置⑤）。fixture 手造 held-out 预验机制层（同桥 TC1-8）·非 empirical 泛化率（真 ConceptNet
held-out defer W4）。sample-disjoint 现在可测·vocab-disjoint（新词无 D:11 边）defer Phase E/F。
"""
from __future__ import annotations

from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.shared.types import ConceptRef, FloorActivation, OutputResult

_SCALE = 1000   # permille（纯整·镜像 lang_rate_permille / capability_exam THRESH_RATE_PERMILLE）


def measure_floor_activation(graph: ConceptGraph, output: OutputResult) -> FloorActivation:
    """读侧后验重导 held-out cue slot 激活率（镜像 generate.py:153-171 stash·纯读·不改写侧）。

    对 output.parts（held-out generate 产·每 part.unit=struct_ref·part.token_refs=段真 token concept 序）逐 unit：
      1. read_instantiates(unit)→skeleton_ref（无 INSTANTIATES→skip·非结构一等化 unit）
      2. rel_kind_of_skeleton(skeleton_ref)→rel_kind（REALIZES→REL_*·无→0=非 R-skeleton·skip·无对应可测）
      3. read_cue_sig(skeleton_ref)→cue slot 序（无 cue 位→skip）
      4. **runtime length-guard**（镜像 generate.py:164）：`len(cue_sig)==len(role_seq)` 不等→skip（accumulation/
         混合 skeleton 错位·审1 MEDIUM-2·用 role_seq 非 token_refs：generate role_seq 长·token_refs 只收真 token·
         accumulation 时 len(token_refs)<len(role_seq)·用 token_refs 会错位漏计激活）
      5. 逐 cue slot：cue_rel_of(token_refs[slot])==rel_kind→activated / 否则→false_positive（distractor 误激活）

    **accumulation 守**（generate.py:179）：slot_idx≥len(token_refs) 的 cue slot 退 unit（不入 emitted_tokens·
    无真 token）→ 跳过不计 total（无法测 struct_ref 的 cue_rel_of·非真实成词·不计 false_positive 避误罚）。

    返 FloorActivation{activation_permille, false_positive_permille, measured, total, activated}。
    activation_permille = activated*1000//max(total,1)（纯整）。measured = total>0（measured-guard·空探针/not-run→False）。

    纯读·确定性·无副作用·gate caller 门控（formal_train FLOOR_ACTIVATION_MODE ON·四 reader 由 CORRESPONDENCE/
    DIM_BRIDGE gate 守·CI gate OFF 不调本函数→bit-identical）。
    """
    total = 0
    activated = 0
    false_positive = 0
    for part in output.parts:
        skeleton_ref = graph.read_instantiates(part.unit)
        if skeleton_ref is None:
            continue                              # 非结构一等化 unit（无 INSTANTIATES 边）
        rel_kind = graph.rel_kind_of_skeleton(skeleton_ref)
        if rel_kind == 0:
            continue                              # 非 R-skeleton（无 REALIZES→REL_*·无对应可测）
        cue_sig = graph.read_cue_sig(skeleton_ref)
        if not any(c is not None for c in cue_sig):
            continue                              # 无 cue 位（全 None / ()=非 cue skeleton）
        # runtime length-guard（镜像 generate.py:164·审1 MEDIUM-2）：用 role_seq 非 token_refs。
        role_seq = graph.read_role_seq(part.unit)
        if len(cue_sig) != len(role_seq):
            continue                              # 错位/混合 skeleton→退化不计（sound future-risk 守）
        for slot_idx, cue in enumerate(cue_sig):
            if cue is None:
                continue                          # 非 cue slot（走 collide·非对应可测）
            # accumulation 守（generate.py:179 has_token）：extra slot 退 unit 不入 token_refs→无真 token→跳过。
            if slot_idx >= len(part.token_refs):
                continue
            total += 1
            word_ref: ConceptRef = part.token_refs[slot_idx]
            # slot ref 的 D:11（非「实际选的词」concept_ref·dispatch 可选 alias 致 selected≠slot.ref·但
            # CORR_BONUS=1001 严格胜 collide=1→bonus 触发≈selected·审1 LOW-1）。
            word_rel = graph.cue_rel_of(word_ref)
            if word_rel == rel_kind:
                activated += 1                    # ★ 学到的对应词在 cue slot 正确激活（C-vs-L 真判别）
            else:
                false_positive += 1               # cue slot 选了 cue_rel_of≠rel_kind 或无 D:11 边（distractor 误激活）
    activation_permille = (activated * _SCALE) // max(total, 1)
    false_positive_permille = (false_positive * _SCALE) // max(total, 1)
    measured = total > 0                          # measured-guard（空探针/not-run/无 held-out cue slot→False）
    return FloorActivation(
        activation_permille=activation_permille,
        false_positive_permille=false_positive_permille,
        measured=measured,
        total=total,
        activated=activated,
    )
