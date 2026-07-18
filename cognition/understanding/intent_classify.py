"""cognition.understanding.intent_classify — M1片2 intent 分类（替换 INTENT_QUESTION 硬编码）。

设计：doc/重来_M1片2_intent分类设计_2026-07-08.md（4 路探查 + 2 对抗审·反 theater 收窄）。

M1 = intent producer 缺失：消费者 judge.py:224,236 已接（G3a/G3b 门控）·机制
types.py:225-229 三 bool 字段已落·只差 formal_train.py:366,1448 两处
``IntentType(type=INTENT_QUESTION, ...)`` 硬编码填值。本模块 = classify_intent 填值函数
（reward 阶段 + H2 标定共用）。

反 theater 收窄（原 M1片2 三块经 4 路核证→一块半）：
  - is_causal_reasoning = _has_causes_signal(segments)  ← **核心解 G3a**（j3path 从永 0 到加权）
  - type = INTENT_QUESTION 默认 / INTENT_COMMAND（W7+B-PR1 doc §16：子 gate INTENT_COMMAND_MODE ON +
    _has_action_intent 命中动作意图词→COMMAND。dag_path.py:302 早已 tuple 含 COMMAND·Q/C 等价合法·
    STATEMENT(3) 才 DEAD_END·COMMAND 不违终止态闸·零终止态差异。命令词 OR 动作词 doc §16.4）
  - is_structural_sequence_reasoning = False（语言域设计正确·code/arith 走 verify 绕 judge）
  - has_value_claim = _has_value_claim(segments)  ← **G1+#774 解 G3b**（property_claims 非空·gate PROPOSITION_MODE·
    G3b 反 theater 真活·设计 doc/重来_G1reification_774PROPERTY_设计_2026-07-09.md §三.3）

单向依赖：本模块在 cognition.understanding（L5）·只 import cognition.shared.types（同层）·
不 import experiments（L8）·故 classify_intent 签名无 CollectedItem 形参（审2 A 硬破口修）。
gate live-read 经 pure_integer_ai.config.gates（config 是 L0 跨层公共·无向上耦合·同 cue_extractor 范式）。
"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import (
    ConceptRef, INTENT_COMMAND, INTENT_QUESTION, IntentType, Segment,
)


def _has_causes_signal(segments: list[Segment]) -> bool:
    """输入 segments 含因果对 ⇔ observe 将建 EDGE_CAUSES 边（与 causes.py:38-51 建边同源）。

    observe.py:215-220 调 ``build_causes_edges(structured_pairs=seg.structured_causal_pairs,
    cue_pairs=seg.cue_based_causal_pairs)``·故任一 segment 的两源任一非空 ⇔ observe 建了
    CAUSES 边 ⇔ dag_path 极大概率含 CAUSES 锚（judge.py:228 ``any(ref[4]==EDGE_CAUSES ...)``
    通过·j3path = path_strength_weighted 加权）。

    为何用 segment 信号而非查 graph：classify_intent 在 episode 前调（构造 intent 喂
    dag_path_step）·dag_path 此时未产·不能查 dag_path.path.edges。segment causal pairs 是
    episode 前信号·与 observe 建边条件同源（causes.py builder 按 segment 字段建）。

    边界（判 True 但 dag_path 无锚 → G3a 误 veto·低概率·首版可接受·oracle 后细化）：
      - a==b self-pair（causes.py:57·两侧 token normalize 到同 concept·如"X 导致 X"）→ 不建边
      - from-node 被 word_terminated skip（dag_path.py:227·长跑训练 eff_freq 累积触发
        THETA_FREQ=1000）→ EDGE_CAUSES BLOCKED
      - from-node 不可达（struct_anchor 链断·概率极低）
    典型因果 fixture（"雨导致地湿"·cue 在两异 token 间）净正。
    """
    for seg in segments:
        if seg.cue_based_causal_pairs or seg.structured_causal_pairs:
            return True
    return False


def _has_value_claim(segments: list[Segment]) -> bool:
    """G1+#774：输入 segments 含属性命题 ⇔ observe 将建命题节点 + PROPERTY 边（与 property.py:48 建边同源）。

    observe.py ④-ter 调 ``build_property_edges(property_claims=seg.property_claims)``（gate PROPOSITION_MODE ON 时）·
    故任一 segment 的 property_claims 非空 ⇔ observe 建了命题节点 PROPERTY 出边 ⇔ G3b counterfactual_value_check
    全局扫命题节点有真目标（judge.py:237 ``if intent.has_value_claim:`` 激活 G3b）。

    **gate PROPOSITION_MODE 双守**（反 theater·设计 §三.3）：gate OFF → property_claims 永空（extract_property_claims_gated
    返 []）→ has_value_claim=False → G3b 不激活 → 既有行为零变 bit-identical。gate ON → property_claims 真提 →
    has_value_claim=True → G3b 激活全局扫命题节点（命题节点真有 PROPERTY 出边·非空集永返 1·反 theater）。

    边界（与 _has_causes_signal 同构）：领属模式（attr_type=-1）build_property_edges skip·property_claims 含
    领属 6-tuple 但无命题节点建→G3b 扫不到该命题（首版 defer·诚实·design §六）。典型 的...是 fixture 净正。
    """
    if not getattr(gates, "PROPOSITION_MODE", False):
        return False   # gate OFF 守回归 bit-identical（property_claims 永空·G3b 不激活）
    for seg in segments:
        if seg.property_claims:
            return True
    return False


# ---- W7+B-PR1 动作意图命令判定（doc §16·命令词 OR 动作词·复用 cue_words.is_action_intent_cue 两源） ----
# 命令判定 = 任一 token 命中动作意图词（命令词 帮我/请→COMMAND_MOOD + 动作词 生成/计算→ACTION_*）。
# 复用 cue_words.is_action_intent_cue（frozenset 第一源 action_primitives._ACTION_LEXICAL_CUE + D:11 readback 第二源·
#   镜像 is_negation_cue #940 两源范式·gate ACTION_D11_READBACK_MODE）·不本地存词表（解命令词/动作词穷举不尽·开放变体走 D:11）。
# **doc §16.1 推翻 §15.1 纠正③**：命令词走 D:11（非"不走 D:11"）·命令 mood 概念先天·命令词 alias 开放·D:11 学。


def _has_action_intent(segments: list[Segment], *,
                       backend=None, edge_store=None,
                       space_id: int | None = None, concept_index=None) -> bool:
    """W7+B-PR1：输入 segments 含动作意图词（命令词 OR 动作词）→ type=INTENT_COMMAND（doc §16）。

    扫 segment.tokens·任意 token 命中 is_action_intent_cue→True。命令词（帮我/请·祈使引导词）+ 动作词
    （生成/计算·B-PR1 ACTION_*）都判命令（doc §16.4·一条命令 = 祈使 mood OR 动作内容·覆盖引导词祈使 + 有动作词裸祈使）。

    gate INTENT_COMMAND_MODE OFF → 返 False（守 bit-identical·type 永 QUESTION）。
    is_action_intent_cue 两源：frozenset（gate OFF 基底）+ D:11 readback（gate ACTION_D11_READBACK_MODE·需 backend 等参数）。
    backend 等参数 None → is_action_intent_cue 退化 frozenset（不读 D:11）。

    诚实边界：纯句式祈使（去开门·无引导词无动作词）→ 漏判 QUESTION → 无回写 → 泛化 defer B-PR2 experience_count 扩散（§13.8）。
    误判代价当前为零（COMMAND=QUESTION dag_path:302 行为同）·B-PR2 落地后缓解=D3 三重防巧合。
    """
    if not getattr(gates, "INTENT_COMMAND_MODE", False):
        return False   # gate OFF 守 bit-identical（type 永 INTENT_QUESTION·与当前 M1-ON 一致）
    from pure_integer_ai.cognition.understanding.cue_words import is_action_intent_cue
    for seg in segments:
        for tok in seg.tokens:
            if is_action_intent_cue(tok, seg.lang, backend=backend, edge_store=edge_store,
                                    space_id=space_id, concept_index=concept_index):
                return True
    return False


def classify_intent(sink: ConceptRef | None,
                    segments: list[Segment], *,
                    backend=None, edge_store=None,
                    space_id: int | None = None, concept_index=None) -> IntentType:
    """M1片2：reward 阶段 + H2 标定共用·替换两处 ``IntentType(INTENT_QUESTION)`` 硬编码。

    **签名无 item**（审2 A 硬破口修）：CollectedItem 在 experiments（L8）·本模块在
    understanding（L5）·传 item = L5→L8 向上耦合违 lint import_direction。函数体不用 item
    （_has_causes_signal 只读 Segment）·故删 item 形参。未来层① 需 lang/domain 时传标量
    非 CollectedItem。

    首版（反 theater 收窄 + W7 doc §15）：
      type   = INTENT_QUESTION 默认 / INTENT_COMMAND（W7+B-PR1·doc §16：子 gate INTENT_COMMAND_MODE ON 且
               _has_action_intent 命中动作意图词 命令词 帮我/请 OR 动作词 生成/计算。dag_path.py:302 早已
               tuple 含 COMMAND·Q/C 等价合法终止态·非 STATEMENT(3)→DEAD_END·故 COMMAND 不违终止态闸·零终止态差异）
      **可选 backend/edge_store/space_id/concept_index**（B-PR1 D:11 readback·gate ACTION_D11_READBACK_MODE）：
        None → _has_action_intent 退化 frozenset（is_action_intent_cue 第一源·gate OFF 基底·bit-identical）。
      sink   透传 caller 的 struct_refs[-1]（选项 B·维持 reward 通路·禁 sink=None 杀陈述）
      is_causal_reasoning = _has_causes_signal(segments)  ← 解 G3a 核心
      is_structural_sequence_reasoning = False（语言域设计正确）
      has_value_claim = _has_value_claim(segments)  ← G1+#774 解 G3b（property_claims 非空·gate PROPOSITION_MODE）
    """
    return IntentType(
        type=INTENT_COMMAND if _has_action_intent(
            segments, backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index) else INTENT_QUESTION,
        sink=sink,
        is_causal_reasoning=_has_causes_signal(segments),
        is_structural_sequence_reasoning=False,
        has_value_claim=_has_value_claim(segments),
    )
