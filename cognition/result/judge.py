"""cognition.result.judge — 模块3 judge 四判据合成 reward=ΠG·ΣwJ（§十四子问题2）。

judge(output, dag_path, input, graph, weights, workmem) -> (reward, GMeta)
  reward = G4 · G2p · G3a · G3b · G5 · (w1·J1 + w2·J2s + w3·J3path + w4·J4word)   # §十四合成·R4 补 G3b/G5·#1041 补 J4word
  G门=乘法否决因子（hard 下限定义可行域）·J 加权=加法梯度（soft reward 实体）。
  否决≠替代（旧错=把 constraint 当 reward 本身·§十四核心算术）。

  G_meta 5字段 {G4,G2p,G3a,G3b,G5}_vetoed（D1 落盘·R4 加 G3b/G5 须消费者同改）：
    early return 前写 vetoed=True（否则 Episode 漏记 G3b/G5 veto→pillar1 假塌+pillar2 假不活跃+收敛误报）。
  四判据自锚于输入+拓扑不依赖教师标准（自评命门破解）·权重断奶后冻结（防自评膨胀第一闸）。

  G3b（R4 写回核心·仅"含值主张"意图激活·层a 结构值冲突~30行·防 vacuous-reward 训练污染）。
  G5/C6（R4 写回核心·仅算术/代码域激活·唯一承重正确性件·首版落 harness·Mode A 教师 ground-truth
    断奶前 / Mode B self-consistency 断奶后）。
  H3：结构序推理意图 J3 归零 G3a=1 跳过（纯 PRECEDES/T_STEP 路径无 CAUSES 锚不该罚）。

gate JUDGE_MODE：**承重件(防塌柱①·judge 门否决 G4/G2p/G3a)永远 active·judge() 函数体不读 gate·gate 装饰性保留位**(关 judge=reward 虚高=塌·故无 OFF 态·gate 二分只对可选机制有意义·ATTRACTOR/EXPLORATION/CUE_EXTRACTOR 等)。
铁律：纯整数（G门 0/1·J 加权纯整 ×1000·reward 纯整 ≥0）/ 不写死（权重 env 默认+oracle 标·CAUSES 来源门控）/
  外部只启发（教师经录放层只标定权重非定义判据·断奶后退场）。
诚实边界：judge 不判真因果/真语义对（判结构正确是语义正确代理·§十四）/ J2 验槽被填不验 TRUE /
  J3 连通到结论不证结论 TRUE / stable≠correct。
defer：J4b cycle_walk 代数闭合（辅助子信号）/ G3b 层(b)(c)（语义真对立·#479 truth 关切 W2·**非 W1 D 物理接地墙**·provisional/可废止对立机制 E3 覆写+E4 推理引擎可达·只 definitive truth 验证撞 #479）/ G5 Mode B 深化（C6 re-derivation correctness 自评·
  D 墙+VM 接线 defer（★round2 lens 拆分：stable≠correct=真 #479 W2 truth 墙·断奶后无 ground-truth·非 W1 D 物理接地；VM 交叉验证接线=wiring 缺口非墙·见主线:1260）·minimal=结构门自洽+反馈环已 achieved·G5 vacate pass=1·见 doc/重来_ModeB自洽设计补充.md）。
"""
from __future__ import annotations

from typing import Any, Callable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.types import (
    OutputResult, PathResult, InputPayload, JudgeWeights, GMeta,
    ConceptRef, J3_CAUSES_WEIGHT, J3_PRECEDES_WEIGHT,
    REWARD_LEGITIMATE_DOMAINS, WEANING_PRE, WEANING_POST,
)
from pure_integer_ai.storage.edge_types import EDGE_CAUSES, EDGE_PRECEDES, EDGE_PROPERTY
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.output_measure import output_word_ratio
from pure_integer_ai.cognition.process.a4_align import coverage_overlap, COVERAGE_SCALE
from pure_integer_ai.config import gates

# G5 激活域（算术=DOMAIN_MATH / 代码=DOMAIN_CODE·C6 自证机唯一承重正确性件）= reward-legitimate 域。
# canonical 定义 hoist 到 shared REWARD_LEGITIMATE_DOMAINS（reward_propagate 复用·单一源·止血 #1146）·
# 此处 alias 保留（line 311 `input_payload.domain in _ARITH_DOMAINS` + 多处 doc 引用名不变）。
_ARITH_DOMAINS = REWARD_LEGITIMATE_DOMAINS

# J 加权 ×1000 缩放（coverage_overlap / slot_fill_rate / path_strength 都纯整 0..1000）
J_SCALE = COVERAGE_SCALE

# 自证机 C6 注入式（Mode A 教师 ground-truth 断奶前 / Mode B self-consistency 断奶后）
# 默认 None → harness 未接·G5 域 pass=1 占位（judge 永远 active·self_proof_fn=None 时 G5 占位 pass=1·gate 不读）
# 返 int | None：None=miss/退场（self_proof_check 转 veto 防 miss 占位 pass·stub #3）
SelfProofFn = Callable[[OutputResult, PathResult, ConceptGraph], "int | None"]


# ---- J4 闭合（纯硬否决 G4·J4a 引用解析+槽绑定·主原语） ----

def check_closure(output: OutputResult, dag_path: PathResult,
                  graph: ConceptGraph, workmem: Any) -> bool:
    """J4 闭合性检查（J4a·引用解析+槽绑定·非 cycle_walk·J4b defer）。

    三层分层（§十四B13 低② + #733 纠偏 + factor E 层1 补全）：
      层1 同段前序 token（factor E·gate PRONOUN_INTRASEG_MODE·resolve_pronoun_occurrence 层1 候选块·
            work_memory._current_segment_refs·"动物...它们"同段前指。旧注释"已解析"是 theater·2026-07-09 补全）
      层2 跨句 N=3 carry 窗口（workmem.prior_segments FIFO·transient·resolve 候选源）
      层3 记忆 occurrence token 性质B 带衰减回看（#733·落 observe resolve_pronoun_occurrence 扩候选·非 judge）
    **#733 设计纠偏**：层3 落 observe（扩候选覆盖范围·§十四:1291）非 judge（task doc 原 placement 错·内部矛盾
      "代词消解"=observe vs placement judge）。judge ② 查结果悬空·非扩候选。
    **② fix（#733）**：旧 ② 查 produced_refs 代词悬空是 theater（4 重 kill switch：produced_refs 是 struct_refs
      非代词 / surface_of 生产返 None / out_edges 全类型 / EDGE_REFERS_TO 未 import 破函数）。改读 workmem.dangling_units
      ——observe resolve_pronoun_occurrence 悬空时 _segment_dangling++·段末标 struct_ref 进 dangling_units·
      judge 查 output.parts[*].unit ∈ dangling_units → J4=0 真碎句。绕 4 重 kill switch·非 theater 真闭合判据。
    首版闭合判据（结构层·诚实）：
      ① 槽绑定：每个 OutputPart 有词（无空槽·槽被填）
      ② 引用解析：output unit 不含悬空段（dangling_units·observe 标·judge 查）
    sink 一致由 G2p 守（reached_sink·非 J4 职责）·J4b cycle_walk defer。
    诚实边界：J4 查代词是否解析（input 侧·§十四:751）非查输出含代词（output 侧）·stable≠correct。
    """
    # ① 槽绑定
    for part in output.parts:
        if not part.words:
            return False   # 空槽=未绑定→有洞即破
    # ② 引用解析（#733 ② fix·旧 produced_refs ② theater 4 重 kill switch 删·改读 workmem.dangling_units）
    #    observe resolve_pronoun_occurrence 悬空（返 None）→ _segment_dangling++ → 段末标 struct_ref 进
    #    dangling_units·judge 查输出 unit 是否含悬空段·是→J4=0 真碎句（§十四:751 悬空0真碎句）。
    #    语义从"查 produced_refs output 含代词"转"查 input 解析失败标记"·对齐 §十四:751（查代词是否解析）。
    dangling_units = getattr(workmem, "dangling_units", set())
    for part in output.parts:
        if part.unit in dangling_units:
            return False   # 输出含悬空段=引用未解析→J4=0 真碎句
    return True


# ---- J2 意图（混合·G2p 硬否决+J2s 加权） ----

def slot_fill_rate(output: OutputResult, intent: Any) -> int:
    """J2s secondary 槽填充率（纯整 0..1000）。

    = 1000 × (bound_slots / total_slots)。bound=有词的槽·total=所有 OutputPart 词位。
    单槽退化纯否决（G2p 已守）·多槽给梯度。
    """
    total = 0
    bound = 0
    for part in output.parts:
        for w in part.words:
            total += 1
            if w:
                bound += 1
    if total == 0:
        return 0
    return (J_SCALE * bound) // total


# ---- J3 因果（混合·G3a 硬否决+J3path 加权） ----

def path_strength_weighted(dag_path: PathResult, graph: ConceptGraph) -> int:
    """J3path 路径强度加权（纯整 0..1000·CAUSES 高权/PRECEDES 低权 1:10·B1 占位）。

    = min(1000, Σ_et weight(et) × strength(et))·CAUSES=10·PRECEDES=1（序边贡献一个数量级低·防堆链游戏）。
    H3 结构序推理意图已归零跳过（caller 守·此函数只在因果机制推理意图下调）。
    """
    total = 0
    for ref in dag_path.path.edges:
        et = ref[4]
        strength = _edge_strength(ref, graph)
        if et == EDGE_CAUSES:
            total += J3_CAUSES_WEIGHT * strength
        elif et == EDGE_PRECEDES:
            total += J3_PRECEDES_WEIGHT * strength
    return min(J_SCALE, total)


def _edge_strength(ref: ConceptRef, graph: ConceptGraph) -> int:
    """按 EdgeRef 取边 strength（graph 查行·不在则 0）。"""
    rows = graph.out_edges((ref[0], ref[1]), ref[4])
    for r in rows:
        if (r["space_id_to"], r["local_id_to"]) == (ref[2], ref[3]):
            return r.get("strength", 0)
    return 0


# ---- G3b 反事实层a（R4 写回核心·结构值冲突·仅"含值主张"意图） ----

def counterfactual_value_check(output: OutputResult, dag_path: PathResult,
                                graph: ConceptGraph) -> int:
    """G3b 反事实层a 结构值冲突判断（墙内·CONTRADICTED→0·默认采甲·§十四B11）。

    G1+#774 选 b（设计 doc/重来_G1reification_774PROPERTY_设计_2026-07-09.md §二.2）：全局扫命题节点
    （ATTR_PROPOSITION）的 PROPERTY 出边·任一命题节点同(subject,attr_type)多值=结构矛盾（层a·命题身份
    =(subject,attr_type) 去重→同对多 value 聚同节点→精确判矛盾无假矛盾·fork 分析 §3.2/§3.3）。命题节点是
    **判断层载体**（G3b 读）非**路径层载体**（不进 dag_path/structure_units·零 J1/J2/J3 扰动）·故不读
    output.parts（part.unit=struct_ref 无 PROPERTY 边·旧 part.unit 扫是 theater·fork §一·写回核心~30行·非 defer）。
    output/dag_path 保留签名（G-check 族一致性·output:55 check_closure / self_proof_check:161 同签名·
    层b/c defer 时或用·当前层a 全局扫不读 output/dag_path）。

    **bit-identical**：gate PROPOSITION_MODE OFF → 无命题节点建（observe skip build_property_edges）→
    iter_proposition_nodes 返 [] → 扫空返 1（G3b 仅 has_value_claim=True 时调·gate OFF 时 has_value_claim=False
    不激活·双重守）。层(b) 桥段锚定召回质量 / 层(c) 语义真对立 defer（**#479 truth 关切 W2·非 W1 D 物理接地墙**·provisional/可废止对立 E3 覆写+E4 推理引擎可达·definitive truth 验证撞 #479·judge.py:22 既有 defer注·2026-07-09 纠偏 round1 relabel：旧"D 墙"性质错配）。
    返 1=无冲突 / 0=CONTRADICTED（hard-veto 门因子）。
    """
    for prop_ref in graph.iter_proposition_nodes():
        # 收集该命题节点 PROPERTY 出边指向的值概念集（value·命题节点→value 概念）
        value_targets: set[ConceptRef] = set()
        for e in graph.out_edges(prop_ref, EDGE_PROPERTY):
            value_targets.add((e["space_id_to"], e["local_id_to"]))
        if len(value_targets) > 1:
            return 0   # 同(subject,attr_type)命题节点多值=结构值冲突→CONTRADICTED
    # 层a-extended：模态对当矛盾（STEP6 PR3·T 公理形式层·非 #479 truth·gate MODALITY_MODE）
    # 跨命题节点按 (subj,attr) 分组·判模态方阵（□p vs ◇¬p / □p vs ¬p·T 公理 □p→p + □>◇）
    if getattr(gates, "MODALITY_MODE", False):
        if _modal_contradiction_in_graph(graph):
            return 0   # 模态对当矛盾→CONTRADICTED（层a-extended 形式·T 公理非 truth）
    return 1


def _modal_contradiction_in_graph(graph: ConceptGraph) -> bool:
    """G3b 层a-extended 模态对当矛盾（STEP6 PR3·跨命题节点按 (subj,attr,val) 分组判模态方阵）。

    读 graph.iter_proposition_identity()（命题节点 (subj,attr,pol,mod) 结构存·解 ref→surface defer）+
    每节点 PROPERTY 出边（value）·按 (subj,attr,val) 分组·每组调 _modal_contradiction 判模态方阵矛盾。

    **★value 维度（对抗审 catch·防假阳性）**：分组 key 含 val·□(黑)+¬(白) 异值归不同组→不矛盾（兼容·
    黑蕴涵非白）·□(黑)+¬(黑) 同值归同组→矛盾（T 公理·必然黑+非黑）。无 val 分组会假阳性 over-veto。

    **T 公理形式层墙内**（构造性检查·非 truth·情态比命题多一口气=定理有效性层有形式锚）：
      - 认识 □(mod=1) + 反极性认识 claim(mod∈{0,1,2}) → 矛盾（T 公理 □p→p + □>◇ + □p→¬◇¬p）
      - 道义 道义必然(mod=3) + 反极性道义 claim(mod∈{3,4}) → 矛盾（Ought(p)+May(¬p)·Ought≠Is 故不跨风味）
      - 跨风味（认识 vs 道义）→ 不矛盾；◇p+◇¬p → 不矛盾；断言p+断言¬p → 不矛盾（B1 对立·无 □）
    **守墙**：实质情态真值（认识/规范 W2+动力 W1）defer·T 公理形式层墙内·非 #479 truth。
    """
    groups: dict[tuple[ConceptRef, ConceptRef, ConceptRef], list[tuple[int, int]]] = {}
    for prop_ref, subj_ref, attr_ref, pol, mod in graph.iter_proposition_identity():
        # 收集该命题节点 PROPERTY 出边 value·按 (subj,attr,val) 分组（val 维度防异值假阳性）
        for e in graph.out_edges(prop_ref, EDGE_PROPERTY):
            val_ref = (e["space_id_to"], e["local_id_to"])
            groups.setdefault((subj_ref, attr_ref, val_ref), []).append((pol, mod))
    for claims in groups.values():
        if _modal_contradiction(claims):
            return True
    return False


def _modal_contradiction(claims: list[tuple[int, int]]) -> bool:
    """单组 (subj,attr) 的 (pol,mod) 列表是否有模态对当矛盾（T 公理形式层·step6 PR3）。

    mod 编码：0=实然/1=□必然/2=◇可能/3=道义必然/4=道义可能（P0.3 surface）。
    - 认识风味（mod 0/1/2·含断言 mod=0）：∃ □(mod=1) AND ∃ 反极性认识 claim → 矛盾
      （□p+¬p→T 公理 / □p+◇¬p→□>◇ / □p+□¬p→两 □ 对立·皆矛盾）
    - 道义风味（mod 3/4）：∃ 道义必然(mod=3) AND ∃ 反极性道义 claim → 矛盾
      （道义必然p+道义可能¬p / 道义必然p+道义必然¬p·皆矛盾·Ought(p)→¬May(¬p)）
    - 跨风味 / ◇◇对立 / 断言对立（无 □）→ 不矛盾。
    """
    # 认识风味（mod 0/1/2·断言+epistemic □/◇）
    ep = [(pol, mod) for pol, mod in claims if mod in (0, 1, 2)]
    for pol, mod in ep:
        if mod == 1:   # □ epistemic necessity
            if any(other_pol != pol for other_pol, _ in ep):
                return True   # □p + 反极性认识 claim → 矛盾（T 公理 + □>◇）
    # 道义风味（mod 3/4）
    deo = [(pol, mod) for pol, mod in claims if mod in (3, 4)]
    for pol, mod in deo:
        if mod == 3:   # 道义必然
            if any(other_pol != pol for other_pol, _ in deo):
                return True   # 道义必然p + 反极性道义 claim → 矛盾
    return False


# ---- G5 自证机 C6（R4 写回核心·仅算术/代码域·harness） ----

def self_proof_check(output: OutputResult, dag_path: PathResult,
                     graph: ConceptGraph, *, weaning_phase: int = WEANING_PRE,
                     self_proof_fn: SelfProofFn | None = None) -> int:
    """G5/C6 自证机（算术/代码域唯一承重正确性件·fail→0·§十四C6）。

    **G5-A 自证机门因子**（§十四:1245·已 live·写回 G_meta）·**非 G5-B 边级 promote**（promote.py:_reward_ok
    edge sn/tn·命名借用）·**非 G5-C 记忆项延迟晋升闸**（§十三:1108 memory_item SEG_EPISODIC 比率门·#732
    promote_memory_consolidate·记忆项 status flip）·**三个 G5 同名不同物**。

    3 态路由（R1·doc/重来_VM图灵完备与C6设计补充.md §5.3）——图灵完备 VM 后 StepLimitExceeded 语义
    须定：禁 pass（奖励死循环）/ 禁 veto（误杀合法长迭代）/ 正解 vacate。按 weaning_phase 路由 None：

      fn is None（harness 未接 / TEACHER_MODE OFF）        → 1  skip/vacate（G5 非承重）
      fn 返 1（Mode A VM==教师 GT）                        → 1  verified
      fn 返 0（Mode A 不匹配 / Mode B 结构破/路径不一致）   → 0  mismatch 硬否决
      fn 返 None · WEANING_PRE（教师 miss·stub#3）          → 0  veto（防脏 reward·E4 红线）
      fn 返 None · WEANING_POST（无子图/StepLimit/单路径）  → 1  vacate（G5 非承重·不奖励死循环亦不误杀）

    StepLimitExceeded 由 VM-proof fn 捕获→返 None（守 SelfProofFn 签名 int|None·VM catch 在 proof fn 层·
      judge 不 import VM·解耦）·此处按 phase 路由 None（POST=vacate / PRE=veto）。
    R6（Mode B 多路径同源=theater·§5.2）：多路径交叉自证须**独立来源**（闭式=独立 PROPERTY 槽声明 /
      迭代=COMPOSES 算子树）·同源编译=同错一致=假 reward。单路径诚实下限=返 None vacate（非 pass=1）·
      A3/C6 接线时强制独立源。
    """
    if self_proof_fn is None:
        return 1   # harness 未接（Mode B defer / TEACHER_MODE OFF）·skip/vacate·G5 非承重
    result = self_proof_fn(output, dag_path, graph)
    if result is None:
        # R1：None 按 weaning_phase 消歧——PRE 教师 miss→veto（防占位 pass 脏 reward·E4）/ POST 无GT·StepLimit·单路径→vacate
        return 0 if weaning_phase == WEANING_PRE else 1
    return int(result)


# ---- judge 主合成 ----

def judge(output: OutputResult, dag_path: PathResult, input_payload: InputPayload,
          graph: ConceptGraph, weights: JudgeWeights, workmem: Any, *,
          self_proof_fn: SelfProofFn | None = None) -> tuple[int, GMeta]:
    """judge 四判据合成。返 (reward≥0, GMeta 5字段)。

    reward = G4·G2p·G3a·G3b·G5·(w1·J1 + w2·J2s + w3·J3path + w4·J4word)·负值只来自步进死路非 judge。
    J4word（#1041·gate OUTPUT_WORD_REWARD_MODE 门控）= 产出真词覆盖率·解 truthiness（真词/__seg_* 同分）。
    """
    assert_no_float(weights.w1, weights.w2, weights.w3, weights.w4, _where="judge.weights")
    assert_int(weights.w1, weights.w2, weights.w3, weights.w4, _where="judge.weights_int")
    intent = input_payload.intent
    g = GMeta()   # 5字段全 False 初始化

    # —— J4 闭合（纯硬否决 G4） ——
    j4_closed = check_closure(output, dag_path, graph, workmem)
    if not j4_closed:
        g.G4 = True   # D1 落盘·早退前写 vetoed
        return 0, g
    # G4=0 即否决·此处 closed→G4=1（g.G4 保持 False=未 veto）

    # —— J2 意图（G2p 硬否决） ——
    if not output.reached_sink:
        g.G2p = True
        return 0, g
    j2s = slot_fill_rate(output, intent)

    # —— J3 因果（G3a 硬否决+J3path 加权·H3 结构序归零跳过） ——
    # ★P0-5（doc/重来_P0决断集_修正分析十三.md §六）：G3a 存在≠越界·对 causal-reasoning 意图要求
    # CAUSES 锚=因果域本职。越界①(全局判据)已由此三 bool 门控收回(降级)·门控 dead 是 intent 未填值
    # 的下游症状(formal_train INTENT_QUESTION 硬编码)·非 G3a 本体越界。填值→阶段5b/5e。
    if intent.is_structural_sequence_reasoning or not intent.is_causal_reasoning:
        g.G3a = False   # 归零跳过·G3a=1（未 veto）
        j3path = 0
    else:
        has_causes_anchor = any(ref[4] == EDGE_CAUSES
                                for ref in dag_path.path.edges)
        if not has_causes_anchor:
            g.G3a = True   # D1 落盘·推理意图无 CAUSES 锚=推理无根→veto
            return 0, g
        j3path = path_strength_weighted(dag_path, graph)

    # —— G3b 反事实层a（仅"含值主张"意图激活·R4 写回核心） ——
    if intent.has_value_claim:
        if counterfactual_value_check(output, dag_path, graph) == 0:
            g.G3b = True   # D1 落盘·早退前写 vetoed
            return 0, g

    # —— G5 自证机 C6（仅算术/代码域激活·R4 写回核心） ——
    if input_payload.domain in _ARITH_DOMAINS:
        if self_proof_check(output, dag_path, graph,
                            weaning_phase=input_payload.weaning_phase,
                            self_proof_fn=self_proof_fn) == 0:
            g.G5 = True   # D1 落盘·早退前写 vetoed
            return 0, g

    # —— J1 覆盖（纯加权·量 KEY 骨架非全量防堆量游戏） ——
    key_ids = [ref[1] for ref in input_payload.key_skeleton]
    produced_ids = [part.unit[1] for part in output.parts]
    j1 = coverage_overlap(key_ids, [produced_ids], ordered=True) if key_ids else 0

    # —— J4word 产出真词覆盖率（#1041 构造②·gate OUTPUT_WORD_REWARD_MODE 门控·truthiness 校准） ——
    # review-2 钉死：旧 reward 对真词/__seg_* 同分（slot_fill_rate `if w:` 只判非空字符串）→ 信号假（判据②③）。
    # gate ON：J4word=产出真 token 覆盖率（output_word_ratio 读 OutputPart.token_refs·#1040 携真 token concept 序）
    #   → w4·J4word 进 reward → reward 反映产出真词质量（非 truthiness 非空）。
    # gate OFF：J4word=0（主守）+ DISPATCH_TOKEN_CHAIN_MODE OFF→token_refs 空→ratio=0（次守）→ w4·0=0
    #   → reward = w1·J1+w2·J2s+w3·J3path 逐字现状 bit-identical。w4 默认 1·gate OFF 时无关（H2 标定不动 w4）。
    j4word = output_word_ratio(output) if getattr(gates, "OUTPUT_WORD_REWARD_MODE", False) else 0

    # —— 合成（纯整数·R4 补 G3b·G5 写回门因子·#1041 补 J4word） ——
    g4 = 0 if g.G4 else 1
    g2p = 0 if g.G2p else 1
    g3a = 0 if g.G3a else 1
    g3b = 0 if g.G3b else 1
    g5 = 0 if g.G5 else 1
    reward = g4 * g2p * g3a * g3b * g5 * (
        weights.w1 * j1 + weights.w2 * j2s + weights.w3 * j3path
        + weights.w4 * j4word)
    assert reward >= 0, f"judge 输出约束 reward≥0·got {reward}"
    return reward, g
