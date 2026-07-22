"""cognition.understanding.cue_words — 元定义层指向词/系词集（出厂硬件·§8.1c-bis 来源②锚）。

CAUSES / IS_A 构造器（cue_extractor）的句法锚定词集。**元定义层固化·非语义规则·
reward 不调·断奶前后不变**（§九铁律承认 enum 例外·同 space_routing META_DEFINITION /
模态标记码点·C9-bis §D 注：元定义固化词不进学习型信号边候选池·两子集不重叠）。

三类（按句法方向分· extractor 按此判因/果/child/parent 方向）：
  CAUSES_CUE_FORWARD   前因后果（因在指向词前·果在后）：所以/因此/导致 ...
  CAUSES_CUE_BACKWARD  前果后因（果在前·因在后）：因为/由于 ...
  IS_A_CUE             系词（child 在前·parent 在后）：是一种/属于 ...
  PRECEDES_CUE_FORWARD 时序（刀 A）：A → [cue] → B·A 先于 B：然后/之后/接着 ...
  ARITH_EQUALS_CUE     数值等式声明（刀 B）：EXPR 等于 NUM·左式二目算术：等于/equals ...
  UNIVERSAL_CUE        全称量化（刀 C）：child → [cue] → parent·X 都是 Y（内涵分类子集 X⊆Y）：都是/全是 ...
  EXISTENTIAL_CUE      存在量化：有的 X 是 Y·只标记 A∩B 非空声明，不携带证明：有的/有些 ...
  MEREOLOGY_CUE        部分-整体（T-L1d·客观序 gap）：X 的一部分 Y·part → whole·boot loader 主路径（同 is_a/causes）·
                       解 REL_MEREOLOGY 误路由入 IS_A_CUE·observe-time 提取 defer
（刀 A 时序 + 刀 B 数值 = 构造性检查 SELF_PRODUCED·刀 C 量化 = 构造性验证 EXTERNAL·ConceptNet 外部源对齐·
 三值逻辑 None 守属性全称 G5b #479 墙·同元定义层固化·详 doc/重来_刀C量化cue设计_2026-07-08.md）
数值算子词（加/减/乘·刀 B·_ARITH_OP_WORDS·非 cue_type·arith_op_of 查）是 ARITH_EQUALS_CUE 左式算子识别。

中英按 lang 分（C1 防跨语言·同 COOCCURS 分桶）·不同 lang 的词集不串。

诚实边界：
  - 词集是句法锚定锚点·非"词义→关系"映射（不判"导致"一词的语义·只用作位置锚）。
  - exact token 匹配（caller 须将指向词切为独立 token·首版按空白切语料·
    emergent_role/真 tokenize defer·§十一 6Q）。不命中零 pair（守反统计契约）。
"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_CAUSES, REL_PRECEDES, REL_SUBSET, REL_MEMBER, REL_MEREOLOGY, REL_EQUAL, REL_PROPERTY, REL_SIMILAR,
)
from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD, OPCODE_SUB, OPCODE_MUL
from pure_integer_ai.crosscut.integer.compare import CMP_GT, CMP_LT, CMP_GE, CMP_LE   # 刀 D 比较 OP opcode（canonical ordering home）

# ---- cue 类型（按句法方向·extractor 用） ----
CAUSES_CUE_FORWARD = 1    # 前因后果：因 → [cue] → 果
CAUSES_CUE_BACKWARD = 2   # 前果后因：果 → [cue] → 因
IS_A_CUE = 3              # 系词：child → [cue] → parent
PRECEDES_CUE_FORWARD = 4  # 时序（刀 A）：A → [cue] → B·A 先于 B（然后/之后/接着·单向 FORWARD）
ARITH_EQUALS_CUE = 5      # 数值等式声明（刀 B）：EXPR 等于 NUM ·左式二目算术 NUM OP NUM·右式 NUM·闭包传检查·不入图
UNIVERSAL_CUE = 6         # 全称量化（刀 C）：child → [cue] → parent ·X 都是 Y·内涵分类子集 X⊆Y·
                         # ConceptNet 外部源验·构造性验证 EXTERNAL（刀 A/B SELF_PRODUCED 是检查·刀 C 升验证）·三值逻辑
EXISTENTIAL_CUE = 7       # 存在量化：有的 X 是 Y；cue 只标记声明，真值需要独立 typed Evidence。
MEREOLOGY_CUE = 8         # 部分-整体（T-L1d·客观序 gap 补）：X 的一部分 Y ·part(左) → whole(右)·
                         # 解 REL_MEREOLOGY 误路由入 IS_A_CUE（gate ON 时 部分-整体 被建成 IsA 边=语义错）·
                         # 独立 EDGE_MEREOLOGY=25 typed 边·boot loader mereology_facts 主路径（同 is_a/causes）·
                         # 与 UNIVERSAL/EXISTENTIAL(6/7) 同范式：定义 cue_type 但走独立消费者·extract_cues if/elif
                         # 不产 pair（部分 首源不入 _CUE_WORDS frozenset·gate OFF 仍 None·bit-identical）·
                         # observe-time 提取 defer（boot loader 是主数据路径·镜像 is_a/causes/alias）。

# ---- 刀4 决断5 / 刀 A：REL_* → cue_type 映射（元定义层立法·D:11 readback 用） ----
# 因果 → CAUSES_CUE_FORWARD·时序 → PRECEDES_CUE_FORWARD（刀 A 入手⑥纠偏·原误并因果·时序≠因果）·
# 类属 → IS_A_CUE（child→parent 子集）·mereology → MEREOLOGY_CUE（part→whole·T-L1d 独立单列·
#   解首版"折入 IS_A_CUE"语义误路由·部分-整体≠子集·客观序 gap 补 EDGE_MEREOLOGY=25 + 构造器）。
# REL_EQUAL → ARITH_EQUALS_CUE（STEP5·"等于"类词→数值等式声明锚·D:11 readback 让非 frozenset 等同词
#   经教师晋升被识别·consumer=extract_numeric_claims→numeric_proof_fn）。
# REL_PROPERTY/SIMILAR 无 cue_type 对应（不映射·readback 返 None·走各自独立消费者·STEP5 PR3/PR4）。
_REL_KIND_TO_CUE_TYPE: dict[int, int] = {
    REL_CAUSES: CAUSES_CUE_FORWARD,
    REL_PRECEDES: PRECEDES_CUE_FORWARD,
    REL_SUBSET: IS_A_CUE,
    REL_MEMBER: IS_A_CUE,
    REL_MEREOLOGY: MEREOLOGY_CUE,
    REL_EQUAL: ARITH_EQUALS_CUE,
}

# ---- 元定义层词集（lang → {cue_type: frozenset[word]}) ----
_CUE_WORDS: dict[int, dict[int, frozenset[str]]] = {
    LANG_ZH: {
        CAUSES_CUE_FORWARD: frozenset({
            "所以", "因此", "故", "导致", "使得", "造成", "引起", "从而", "致使",
        }),
        CAUSES_CUE_BACKWARD: frozenset({
            "因为", "由于", "因",
        }),
        IS_A_CUE: frozenset({
            "是一种", "属于", "是一类", "乃",
        }),
        PRECEDES_CUE_FORWARD: frozenset({
            "然后", "之后", "接着", "随后", "后来",  # A → [cue] → B·A 先于 B（单向 FORWARD·"之前"逆向 defer）
        }),
        ARITH_EQUALS_CUE: frozenset({
            "等于",  # NUM OP NUM 等于 NUM ·数值等式声明（刀 B·闭包传检查·构造性检查非验证·Layer0 SELF_PRODUCED）
        }),
        UNIVERSAL_CUE: frozenset({
            "都是", "全是",  # X 都是 Y ·全称量化内涵分类子集 X⊆Y（刀 C·ConceptNet 外部源验·构造性验证 EXTERNAL·三值逻辑 None 守属性全称墙）
        }),
        EXISTENTIAL_CUE: frozenset({
            "有的", "有些",  # closed-class 存在量化词；开放变体由后续学习机制处理。
        }),
    },
    LANG_EN: {
        CAUSES_CUE_FORWARD: frozenset({
            "so", "therefore", "thus", "hence", "causes", "caused",
            "leads", "produces", "brings",
        }),
        CAUSES_CUE_BACKWARD: frozenset({
            "because", "since", "due",
        }),
        IS_A_CUE: frozenset({
            "is_a", "is_a_kind_of", "is_an", "belongs_to",  # 预切短语 token（空白切前须归一）
        }),
        PRECEDES_CUE_FORWARD: frozenset({
            "then", "after", "afterwards", "subsequently", "later",  # A → [cue] → B（before 逆向 defer）
        }),
        ARITH_EQUALS_CUE: frozenset({
            "equals",  # NUM OP NUM equals NUM（刀 B·同 ZH 等于）
        }),
        UNIVERSAL_CUE: frozenset({
            "are_all",  # X are_all Y ·全称量化（刀 C·同 ZH 都是·预切短语 token·caller 空白切前归一）
        }),
    },
}

# ---- 数值等式算子词（刀 B·元定义层固化·非 cue_type·表达式算子识别用） ----
# ARITH_EQUALS_CUE 是等式声明锚（"等于"）·算子词（加/减/乘）是左式二目算术的算子识别·两者分离
# （算子词非 claim 锚·不入 _CUE_WORDS·不参与 cue_type_of 判·extract_numeric_claims 单独查）。
# 仅整数保持算术（+,-,×）·除法 defer（有理结果·须 Rational·首版窄域诚实 scope）。
_ARITH_OP_WORDS: dict[int, dict[str, int]] = {
    LANG_ZH: {
        "加": OPCODE_ADD, "加上": OPCODE_ADD,
        "减": OPCODE_SUB, "减去": OPCODE_SUB,
        "乘": OPCODE_MUL, "乘以": OPCODE_MUL,
    },
    LANG_EN: {
        "plus": OPCODE_ADD, "add": OPCODE_ADD,
        "minus": OPCODE_SUB, "subtract": OPCODE_SUB,
        "times": OPCODE_MUL, "multiplied_by": OPCODE_MUL,
    },
}


def arith_op_of(token: str, lang: int, *,
                backend=None, edge_store=None,
                space_id: int | None = None, concept_index=None) -> int | None:
    """token 是否数值算子词（刀 B·加/减/乘·元定义层固化）·返 OPCODE_* / None。

    **两源**（STEP5 PR2·镜像 cue_type_of 范式）：
      第一源（既有·元定义固化）：_ARITH_OP_WORDS frozenset exact 匹配（返 OPCODE_*·reward 不调·断奶前后不变）。
      第二源（STEP5 PR2 新增·D:11 readback·gate OPERATOR_D11_READBACK_MODE）：
        lookup token → word_ref → lookup_word_operator 读 D:11 PRIMARY 边 → OP_*（过滤算术 OP·OP_ADD/SUB/MUL）
        → _OP_TO_OPCODE → OPCODE_*。反 theater：未验证 SHADOW 不注入（tier_filter=TIER_PRIMARY）。
        冷启动（D:11 OP_* 无教师种子）→ 第二源返 None → 退化 frozenset。

    **bit-identical 守卫**：gate OFF（默认）→ 只走第一源·退化纯 frozenset·回归零翻。
    exact token 匹配（caller 须切算子词为独立 token·同 cue 词纪律）。
    非算子词 / 他 lang 返 None（extract_numeric_claims 据此跳·守反统计契约·不凑配）。
    """
    # 第一源：frozenset exact 匹配（既有·元定义固化）
    op = _ARITH_OP_WORDS.get(lang, {}).get(token)
    if op is not None:
        return op
    # 第二源：D:11 readback（STEP5 PR2·gate 守·反 theater）
    if not getattr(gates, "OPERATOR_D11_READBACK_MODE", False):
        return None
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return None   # 参数不全→退化（不读 D:11）
    return _arith_op_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _arith_op_from_d11_primary(token: str, space_id: int,
                               backend, edge_store, concept_index) -> int | None:
    """STEP5 PR2：D:11 PRIMARY 边 readback → OPCODE_*（算术 OP·反 theater·冷启动返 None）。

    flow：concept_index.lookup(token, space_id) → word_ref | None
      → lookup_word_operator(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → [(op_ref, op_kind), ...] → 过滤 is_arith_op_kind → op_kind_to_opcode → 首命中 OPCODE_* | None。

    只读 TIER_PRIMARY D:11 边（已验证晋升/教师种子·未验证 SHADOW 不注入·反 theater）。
    过滤算术 OP（OP_ADD/SUB/MUL）·非比较 OP（OP_GT/LT/GE/LE）·无交叉污染（comparison_op_of 同范过滤比较 OP）。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return None   # 词未概念化（冷启动·未 observe）·退化
    from pure_integer_ai.cognition.shared.operator_primitives import (
        lookup_word_operator, is_arith_op_kind, op_kind_to_opcode,
    )
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    ops = lookup_word_operator(backend, edge_store, word_ref,
                               space_id=space_id, tier_filter=TIER_PRIMARY)
    for _op_ref, op_kind in ops:
        if not is_arith_op_kind(op_kind):
            continue   # 非算术 OP（比较 OP）·arith_op_of 只认算术·skip
        opcode = op_kind_to_opcode(op_kind)
        if opcode is not None:
            return opcode
    return None


# ---- G1+#774 属性命题 cue（独立于 cue_type_of·不入 _CUE_WORDS·防 是/的 污染 extract_cues 邻居判） ----
# 设计 doc/重来_G1reification_774PROPERTY_设计_2026-07-09.md §三。属性命题 = (subject, attr_type, value) 三元
# 由句法窗口 "X 的 Y 是 Z" 锚定（的=attr marker·是=value copula·两 cue token 固定窗口）。是/的 太通用（是 是
# 汉语最高频字）·若入 _CUE_WORDS 会让 cue_type_of(是) 返非 None → extract_cues 邻居判把 是 当 cue 跳过配对
# → 改变 CAUSES/IS_A/PRECEDES 提取行为 → 非 bit-identical。故属性命题检测走独立 helpers（同 _ARITH_OP_WORDS
# 范式·非 cue_type·extract_property_claims 单独查）·cue_type_of 对 是/的 仍返 None（零行为变）。
_PROPERTY_ATTR_MARKER: dict[int, frozenset[str]] = {
    LANG_ZH: frozenset({"的"}),    # X 的 Y 是 Z ·属性标记（attr marker·subject 与 attr_type 之间）
    # EN "'s" tokenization defer（"X's Y is Z" 须预切 's 独立 token·首版 ZH corpus 优先·EN 的...是 等价 defer）
}
_PROPERTY_VALUE_COPULA: dict[int, frozenset[str]] = {
    LANG_ZH: frozenset({"是"}),    # X 的 Y 是 Z ·值系词（value copula·attr_type 与 value 之间·裸 是 非 是一种/是一类）
    # EN "is" defer（同 's ·须 tokenization·首版 ZH 优先）
}
_PROPERTY_POSSESS_CUE: dict[int, frozenset[str]] = {
    LANG_ZH: frozenset({"具有", "有"}),   # X 具有 Z / X 有 Z ·领属句（attr_type 缺省·首版 defer·build_property_edges skip）
    LANG_EN: frozenset({"has", "have"}),  # X has Z ·领属句（attr_type 缺省·同 ZH 具有 defer）
}


def is_property_attr_marker(token: str, lang: int) -> bool:
    """token 是否属性标记（G1+#774·的·X 的 Y 是 Z·attr marker）·exact 匹配。

    独立于 cue_type_of（不入 _CUE_WORDS·防 是/的 污染 extract_cues·见上注）。
    不命中返 False（extract_property_claims 据此判 的 锚位·守反统计契约·固定窗口不凑配）。
    """
    return token in _PROPERTY_ATTR_MARKER.get(lang, frozenset())


def is_property_value_copula(token: str, lang: int) -> bool:
    """token 是否值系词（G1+#774·是·X 的 Y 是 Z·value copula）·exact 匹配。

    独立于 cue_type_of（同 is_property_attr_marker 注）。裸 是 非 IS_A 的"是一种/是一类"
    （后者多字 token·cue_type_of 走 IS_A_CUE 分支·零冲突）。不命中返 False。
    """
    return token in _PROPERTY_VALUE_COPULA.get(lang, frozenset())


def is_property_possess_cue(token: str, lang: int, *,
                            backend=None, edge_store=None,
                            space_id: int | None = None, concept_index=None) -> bool:
    """token 是否领属句 cue（G1+#774·具有/有/has·X 具有 Z）·exact 匹配 + D:11 readback。

    **两源**（STEP5 PR3·镜像 cue_type_of 范式）：
      第一源（既有·元定义固化）：_PROPERTY_POSSESS_CUE frozenset exact 匹配（具有/有/has/have）。
      第二源（STEP5 PR3 新增·D:11 readback·gate EMERGENT_RELATION_CUE_READBACK_MODE）：
        lookup token → word_ref → lookup_word_concept 读 D:11 PRIMARY 边 → REL_PROPERTY → True。
        反 theater：未验证 SHADOW 不注入（tier_filter=TIER_PRIMARY）。

    **bit-identical 守卫**：gate OFF（默认）→ 只走第一源·退化纯 frozenset·回归零翻。
    STEP5 PR3 possess un-defer：领属句（attr_idx<0）用 REL_PROPERTY 作默认 attr_type·build_property_edges
    建命题节点 (subject,REL_PROPERTY,value)·G3b 消费（非首版 defer skip）。
    独立于 cue_type_of。不命中返 False。
    """
    if token in _PROPERTY_POSSESS_CUE.get(lang, frozenset()):
        return True
    if not getattr(gates, "EMERGENT_RELATION_CUE_READBACK_MODE", False):
        return False
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return False   # 参数不全→退化（不读 D:11）
    return _possess_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _possess_from_d11_primary(token: str, space_id: int,
                              backend, edge_store, concept_index) -> bool:
    """STEP5 PR3：D:11 PRIMARY 边 readback → bool（REL_PROPERTY 命中→True·反 theater·冷启动 False）。

    flow：concept_index.lookup(token, space_id) → word_ref | None
      → lookup_word_concept(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → [(rel_ref, rel_kind), ...] → rel_kind==REL_PROPERTY → True。

    只读 TIER_PRIMARY D:11 边（反 theater）。lookup_word_concept 过滤 ATTR_RELATION_PRIMITIVE（OP_* target skip）。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return False
    from pure_integer_ai.cognition.understanding.word_concept_signal import lookup_word_concept
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    rels = lookup_word_concept(backend, edge_store, word_ref,
                               space_id=space_id, tier_filter=TIER_PRIMARY)
    for _rel_ref, rel_kind in rels:
        if rel_kind == REL_PROPERTY:
            return True
    return False


# ---- STEP5 PR4：REL_SIMILAR 相似 cue（D:11-readback-only·不新增检测 frozenset·D6 更守不写死） ----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §四-bis STEP5 PR4。相似词无既有检测 frozenset·
# is_similar_cue 走 D:11 readback 唯一源（gate EMERGENT_RELATION_CUE_READBACK_MODE·镜像 is_property_possess_cue
# 第二源·但无第一源 frozenset·更守 D6 少一份硬编码词表）。gate OFF→恒 False（bit-identical·无相似检测）。

def is_similar_cue(token: str, lang: int, *,
                   backend=None, edge_store=None,
                   space_id: int | None = None, concept_index=None) -> bool:
    """token 是否相似关系 cue（STEP5 PR4·像/resembles·X 像 Y）·**D:11-readback-only**。

    **单源**（D:11 readback·gate EMERGENT_RELATION_CUE_READBACK_MODE·无 frozenset 第一源）：
      lookup token → word_ref → lookup_word_concept 读 D:11 PRIMARY 边 → REL_SIMILAR → True。
      反 theater：未验证 SHADOW 不注入（tier_filter=TIER_PRIMARY）。

    **bit-identical 守卫**：gate OFF（默认）→ 恒 False（无相似检测·退化现状）。
    冷启动（D:11 REL_SIMILAR 无种子/教师）→ False。
    seeded '像'（_REL_LEXICAL_CUE D:11 种子）→ gate ON True·gate OFF False（行为差可观测·反 theater）。
    独立于 cue_type_of（不入 _CUE_WORDS·防 像 污染 extract_cues 邻居判·同 _ARITH_OP_WORDS 范式）。
    """
    if not getattr(gates, "EMERGENT_RELATION_CUE_READBACK_MODE", False):
        return False
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return False   # 参数不全→退化（不读 D:11）
    return _similar_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _similar_from_d11_primary(token: str, space_id: int,
                              backend, edge_store, concept_index) -> bool:
    """STEP5 PR4：D:11 PRIMARY 边 readback → bool（REL_SIMILAR 命中→True·反 theater·冷启动 False）。

    flow：concept_index.lookup(token, space_id) → word_ref | None
      → lookup_word_concept(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → [(rel_ref, rel_kind), ...] → rel_kind==REL_SIMILAR → True。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return False
    from pure_integer_ai.cognition.understanding.word_concept_signal import lookup_word_concept
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    rels = lookup_word_concept(backend, edge_store, word_ref,
                               space_id=space_id, tier_filter=TIER_PRIMARY)
    for _rel_ref, rel_kind in rels:
        if rel_kind == REL_SIMILAR:
            return True
    return False


# ---- B1 否定 cue（独立于 cue_type_of·不入 _CUE_WORDS·镜像 _PROPERTY_* 范式·P0.3 polarity=1 填值） ----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §七 B1。否定词识别（不/没/非/无 + not/no/never）·
# "X 的 Y 不 是 Z" → polarity=1（P0.3 命题节点扩展·pol 进 surface·B1 cue 抽取填值）。
# 独立 helper（不入 _CUE_WORDS·防 不/没 污染 extract_cues 邻居判·同 _PROPERTY_* 范式）。
# 守墙：结构否定（polarity 标记）墙内·否定语用（言外否定"我不觉得他来了"=他没来）= W2 defer。
# **审计根治 #940**：D6 否定词穷举不尽（未必/绝非/谈不上 开放类）走 D:11 learnable 二源（frozenset 第一源 +
# D:11 readback 第二源·镜像 modal/op 范式）·否定=符号域先天（TYPE_NEGATION=12·同 operator·复用 ATTR_SYMBOL_TYPE=17
# 不挂 abstract_mark·激活 ensure_symbol_types）·开放变体走 D:11 教师晋升有路径。
_NEGATION_CUES: dict[int, frozenset[str]] = {
    LANG_ZH: frozenset({"不", "没", "非", "无"}),   # X 的 Y 不 是 Z（不）/ 没（罕·没是）/ 非文言 / 无文言
    LANG_EN: frozenset({"not", "no", "never"}),     # 英文例：X's Y is not Z / no Y is Z / never
}


def is_negation_cue(token: str, lang: int, *,
                    backend=None, edge_store=None,
                    space_id: int | None = None, concept_index=None) -> bool:
    """token 是否否定词（B1·不/没/非/无 + not/no/never·P0.3 polarity=1 填值）·exact 匹配。

    **两源**（#940·镜像 modal_op_of / arith_op_of / comparison_op_of / is_property_possess_cue 范式）：
      第一源（既有·元定义固化）：_NEGATION_CUES frozenset exact 匹配（返 bool·closed-class 否定词·
        reward 不调·断奶前后不变·gate OFF 基底·bit-identical）。
      第二源（#940 新增·D:11 readback·gate NEGATION_D11_READBACK_MODE）：
        lookup token → word_ref → lookup_word_negation 读 D:11 PRIMARY 边 → 是否指向 TYPE_NEGATION concept
        （否定词文字 alias 可学习·教师晋升新否定词如未必/绝非）。反 theater：未验证 SHADOW 不注入
        （tier_filter=TIER_PRIMARY）。冷启动（D:11 无教师种子）→ 第二源返 False → 退化 frozenset。

    独立于 cue_type_of（不入 _CUE_WORDS·防 不/没 污染 extract_cues·同 is_property_* 范式）。
    extract_property_claims 据此判否定窗口（"X 的 Y 不 是 Z"·不 at j-1·pol=1）·gate NEGATION_MODE 守
    （OFF → negation_on=False → 既有肯定窗口 pol=0·bit-identical）。不命中返 False。

    **bit-identical 守卫**：gate OFF（默认）→ 只走第一源·退化纯 frozenset·回归零翻。
    **守墙**：结构否定（polarity 标记）墙内·否定语用（言外否定"我不觉得他来了"=他没来）= W2 defer。
    **否定=符号域先天**：¬ 概念先天（TYPE_NEGATION）·D:11 readback=文字 alias 可学习（同 operator）·非概念可学（异 modal）。
    """
    # 第一源：_NEGATION_CUES frozenset exact 匹配（既有·元定义固化·gate OFF 基底）
    if token in _NEGATION_CUES.get(lang, frozenset()):
        return True
    # 第二源：D:11 readback（#940·gate 守·反 theater）
    if not getattr(gates, "NEGATION_D11_READBACK_MODE", False):
        return False
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return False   # 参数不全→退化（不读 D:11）
    return _negation_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _negation_from_d11_primary(token: str, space_id: int,
                               backend, edge_store, concept_index) -> bool:
    """#940：D:11 PRIMARY 边 readback → 是否否定词（TYPE_NEGATION target·反 theater·冷启动返 False）。

    flow：concept_index.lookup(token, space_id) → word_ref | None
      → lookup_word_negation(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → bool（word 是否有 D:11 边指向 __TYPE_NEGATION__ concept）。

    只读 TIER_PRIMARY D:11 边（已验证晋升/教师种子·未验证 SHADOW 不注入·反 theater）。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return False   # 词未概念化（冷启动·未 observe）·退化
    from pure_integer_ai.cognition.shared.symbol_types import lookup_word_negation
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    return lookup_word_negation(backend, edge_store, word_ref,
                                space_id=space_id, tier_filter=TIER_PRIMARY)


# ---- B-PR1 动作意图 cue（命令词 帮我/请 + 动作词 生成/计算·W7 命令判定·doc §16·镜像 is_negation_cue #940） ----
# 设计 doc/重来_真生成施工蓝图_2026-07-12.md §16。命令判定 = 命令词 OR 动作词命中任一（§16.4）。
# 命令 mood 词（→INTENT_COMMAND_MOOD·帮我/请·祈使引导词·非动作动词·职责正交）+ 动作词（→ACTION_* 类别·生成/计算·B-PR1）。
# 镜像 is_negation_cue 两源范式（frozenset 第一源 + D:11 readback 第二源）·解命令词/动作词穷举不尽（劳驾/编写/运算 开放变体）。
# **动作意图=符号域先天**（镜像 operator·异 modal·doc §16.3）·D:11 readback=文字 alias 可学习（同否定词/算子词）·非概念可学。
# **覆盖**（doc §16.5）：引导词祈使（帮我生成·命令词+动作词）+ 有动作词裸祈使（生成代码·仅动作词）。
# 纯句式祈使（去开门·无引导词无动作词）冷启动漏判·defer B-PR2 experience_count 回写扩散（结构意图后天学·§13.8）。
def is_action_intent_cue(token: str, lang: int, *,
                         backend=None, edge_store=None,
                         space_id: int | None = None, concept_index=None) -> bool:
    """token 是否动作意图词（B-PR1·命令词 帮我/请 + 动作词 生成/计算·W7 命令判定·doc §16）·exact 匹配。

    **两源**（镜像 is_negation_cue #940 范式）：
      第一源（元定义固化）：action_primitives._ACTION_LEXICAL_CUE exact 匹配（命令词+动作词·closed-class 种子·
        gate OFF 基底·bit-identical）。返 bool（命中任一 ACTION_INTENT_*·COMMAND_MOOD 或 ACTION_*）。
      第二源（D:11 readback·gate ACTION_D11_READBACK_MODE）：lookup token → word_ref → lookup_word_action 读 D:11
        PRIMARY 边 → 是否指向 ACTION_INTENT_* concept（命令词/动作词 alias 可学习·教师晋升劳驾/编写/运算）。
        反 theater：未验证 SHADOW 不注入（tier_filter=TIER_PRIMARY）。冷启动→第二源返 False。

    W7 命令判定（intent_classify._has_action_intent）：任一 token 命中→type=INTENT_COMMAND。

    独立于 cue_type_of（不入 _CUE_WORDS·防 帮我/生成 污染 extract_cues·同 is_negation_cue 范式）。
    **bit-identical 守卫**：gate OFF（默认）→ 只第一源·退化纯 frozenset·回归零翻。
    """
    # 第一源：_ACTION_LEXICAL_CUE frozenset exact 匹配（命令词+动作词·gate OFF 基底）
    from pure_integer_ai.cognition.shared.action_primitives import _ACTION_LEXICAL_CUE
    cues = _ACTION_LEXICAL_CUE.get(lang, {})
    if token in cues:
        return True
    # 第二源：D:11 readback（gate 守·反 theater）
    if not getattr(gates, "ACTION_D11_READBACK_MODE", False):
        return False
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return False   # 参数不全→退化（不读 D:11）
    return _action_intent_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _action_intent_from_d11_primary(token: str, space_id: int,
                                    backend, edge_store, concept_index) -> bool:
    """D:11 PRIMARY 边 readback → 是否动作意图词（ACTION_INTENT_* target·反 theater·冷启动返 False）。

    flow：concept_index.lookup(token, space_id) → word_ref | None
      → lookup_word_action(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → bool（word 是否有 D:11 边指向 INTENT_COMMAND_MOOD 或 ACTION_* concept）。

    只读 TIER_PRIMARY D:11 边（已验证晋升/教师种子·未验证 SHADOW 不注入·反 theater）。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return False   # 词未概念化（冷启动·未 observe）·退化
    from pure_integer_ai.cognition.shared.action_primitives import lookup_word_action
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    hits = lookup_word_action(backend, edge_store, word_ref,
                              space_id=space_id, tier_filter=TIER_PRIMARY)
    return len(hits) > 0


def collect_action_intent_concepts(segments, *, backend, edge_store,
                                   space_id: int, concept_index) -> list[tuple[tuple[int, int], int]]:
    """B-PR2：收集 segments 中命中 D:11 PRIMARY ACTION_* concept 的 distinct refs（doc §17·experience_count feed 用）。

    扫 segments.tokens·``concept_index.lookup(tok, space_id) → word_ref | None``·
    ``lookup_word_action(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY) → [(action_ref, kind)]``·
    distinct by action_ref（同 episode 同 ACTION_* concept 只返一次·镜像 reward_propagate concept_targets set 去重 :208-210）。

    **D:11 readback 单源**（非 is_action_intent_cue 两源·设计审 C CONFIRMED·§17.1 决断3）：B-PR2 须 concept **REF**（写
    experience_count 须 (space_id, local_id) ref）·ref 只从 D:11 边来（lookup_word_action 返 [(action_ref, kind)]）·
    frozenset 第一源给 bool 不给 ref → 单源 D:11 是唯一可用源。boot 种子词（帮我/请/生成/计算/分析/解决）+ 教师晋升 alias
    （劳驾/编写）都有 D:11 PRIMARY 边（bootstrap_action_signals 种·word_concept_signal.py:114 tier=PRIMARY）→ 全命中。

    **tier_filter=TIER_PRIMARY**（反 theater）：未验证 SHADOW alias 不注入（SHADOW=涌现假设·未晋升·不应攒"验证率"）。

    **返 kind**（int_a·0=COMMAND_MOOD / 1-4=ACTION_*）：caller（formal_train hook）写 experience_count 不区分 kind（同 rate 桶）·
    kind 仅供 caller 日志/未来 B-PR3 按类分流感（B-PR3 读 int_a 分流·非本 collector 责）。

    **无 gate**（纯读 collector·gate 在 formal_train hook 守 ACTION_EXPERIENCE_FEED_MODE·gate OFF 不调本函数）。
    纯读（concept_index.lookup + lookup_word_action 均 select/read·无 insert/update·设计审 D CONFIRMED）。
    """
    from pure_integer_ai.cognition.shared.action_primitives import lookup_word_action
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    seen: set = set()
    out: list[tuple[tuple[int, int], int]] = []
    for seg in segments:
        for tok in seg.tokens:
            word_ref = concept_index.lookup(tok, space_id)
            if word_ref is None:
                continue   # 词未概念化（冷启动·未 observe）·skip
            for action_ref, kind in lookup_word_action(
                    backend, edge_store, word_ref,
                    space_id=space_id, tier_filter=TIER_PRIMARY):
                if action_ref in seen:
                    continue   # 同 concept distinct 去重（同 episode 同 ACTION_* 只 feed 一次）
                seen.add(action_ref)
                out.append((action_ref, kind))
    return out


# ---- B2 情态 cue（独立于 cue_type_of·不入 _CUE_WORDS·镜像 _NEGATION_CUES 范式·P0.3 modality 填值） ----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §七 B2。情态词（必然/可能/也许/必须/应该/可以）在"是"前
# （j-1·同否定词槽位·与 negation 互斥·先查 modal）→ extract_property_claims 情态窗口（modality_on 参数·_gated 传
# 本 gate）·modality 填值（0-4）·命题节点建独立 surface 后缀 _0_{mod}（P0.3）·G3b 各判多值·模态对当扩展在 PR3。
# 返 modality 编码：0=实然/1=□必然/2=◇可能/3=道义必然/4=道义可能（P0.3 surface 后缀 _{pol}_{mod}）。
# 独立 helper（不入 _CUE_WORDS·防 必然/可能 污染 extract_cues 邻居判·同 _NEGATION_CUES 范式）。
# 守墙：T 公理形式层墙内（构造性检查·非 truth）·实质情态真值（认识/规范 W2 + 动力 W1）defer。
# **审计根治 [严重-1]**：D6 模态种类归抽象空间后天可学习（D6:60）·走 D:11 learnable 二源（frozenset 第一源 +
# D:11 readback 第二源·镜像 arith_op_of/comparison_op_of/is_property_possess_cue 范式）·开放变体（想必/势必/说不定）
# 走 D:11 教师晋升有路径。建 modal_kind concept（modal_primitives.py）+ ATTR_MODAL_KIND=22 readback +
# abstract_mark MARK_MODAL_KIND=5 D6 归属·不违 STOP（ATTR_* 非 TYPE_*）不违 D6（abstract_mark 归属）。
_MODAL_CUES: dict[int, dict[str, int]] = {
    LANG_ZH: {
        "必然": 1,   # □ 必然（认识·epistemic necessity）
        "可能": 2,   # ◇ 可能（认识·epistemic possibility）
        "也许": 2,   # ◇ 可能（认识·同义）
        "必须": 3,   # 道义必然（deontic necessity·must）
        "应该": 3,   # 道义必然（deontic·should 弱义务·首版归道义必然）
        "可以": 4,   # 道义可能（deontic possibility·permission·can）
    },
    # EN defer（modal 词 must/can/may/should/might·同 property cue ZH-first·EN 情态窗口 defer·须 tokenization）
}


def modal_op_of(token: str, lang: int, *,
                backend=None, edge_store=None,
                space_id: int | None = None, concept_index=None) -> int | None:
    """token 是否情态词（B2·必然/可能/也许/必须/应该/可以）·返 modality 0-4 / None。

    **两源**（审计根治·镜像 arith_op_of / comparison_op_of / is_property_possess_cue 范式）：
      第一源（既有·元定义固化）：_MODAL_CUES dict exact 匹配（返 modality 0-4·closed-class 情态副词·
        reward 不调·断奶前后不变·gate OFF 基底·bit-identical）。
      第二源（审计根治新增·D:11 readback·gate MODAL_D11_READBACK_MODE）：
        lookup token → word_ref → lookup_word_modality 读 D:11 PRIMARY 边 → MODAL_KIND_*（= modality 编码·
        modal_kind 即 modality 值·不需 opcode 映射·比 operator 简单）。反 theater：未验证 SHADOW 不注入
        （tier_filter=TIER_PRIMARY）。冷启动（D:11 MODAL_KIND 无教师种子）→ 第二源返 None → 退化 frozenset。

    返 modality 编码（0=实然/1=□必然/2=◇可能/3=道义必然/4=道义可能·P0.3 surface 后缀 _{pol}_{mod}）。
    独立于 cue_type_of（不入 _CUE_WORDS·防 必然/可能 污染 extract_cues 邻居判·同 is_negation_cue 范式）。
    extract_property_claims 情态窗口据此填 modality 值·gate MODALITY_MODE 守
    （OFF → modality_on=False → 既有肯定窗口 modality=0 bit-identical）。不命中返 None。

    **bit-identical 守卫**：gate OFF（默认）→ 只走第一源·退化纯 frozenset·回归零翻。
    **守墙**：T 公理形式层墙内（构造性检查·非 truth·情态比命题多一口气=定理有效性层有形式锚）·
    实质情态真值（认识/规范 W2 + 动力 W1）defer。
    """
    # 第一源：_MODAL_CUES dict exact 匹配（既有·元定义固化·gate OFF 基底）
    op = _MODAL_CUES.get(lang, {}).get(token)
    if op is not None:
        return op
    # 第二源：D:11 readback（审计根治·gate 守·反 theater）
    if not getattr(gates, "MODAL_D11_READBACK_MODE", False):
        return None
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return None   # 参数不全→退化（不读 D:11）
    return _modal_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _modal_from_d11_primary(token: str, space_id: int,
                            backend, edge_store, concept_index) -> int | None:
    """审计根治：D:11 PRIMARY 边 readback → modality（MODAL_KIND_*·反 theater·冷启动返 None）。

    flow：concept_index.lookup(token, space_id) → word_ref | None
      → lookup_word_modality(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → [(modal_ref, modal_kind), ...] → 首命中 modal_kind（= modality 编码·不需 opcode 映射）| None。

    只读 TIER_PRIMARY D:11 边（已验证晋升/教师种子·未验证 SHADOW 不注入·反 theater）。
    modal_kind 即 modality 值（1-4·与 P0.3 surface modality int 一致）·比 operator 简单（operator 需
    OP_*→opcode 映射·modal 直接返 modal_kind）。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return None   # 词未概念化（冷启动·未 observe）·退化
    from pure_integer_ai.cognition.shared.modal_primitives import lookup_word_modality
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    mods = lookup_word_modality(backend, edge_store, word_ref,
                                space_id=space_id, tier_filter=TIER_PRIMARY)
    for _modal_ref, modal_kind in mods:
        return modal_kind   # 首命中（modal_kind=modality 编码·不需 opcode 映射）
    return None


def is_modal_cue(token: str, lang: int, *,
                 backend=None, edge_store=None,
                 space_id: int | None = None, concept_index=None) -> bool:
    """token 是否情态词（B2·modal_op_of is not None·邻居判用·配对两端不取 modal token）。

    独立于 cue_type_of（不入 _CUE_WORDS·防 必然/可能 污染 extract_cues·同 is_negation_cue 范式）。
    extract_property_claims 据此判情态窗口（"X 的 Y [必然] 是 Z"·modal at j-1·modality 填值）·gate MODALITY_MODE 守
    （OFF → modality_on=False → 既有肯定窗口 modality=0·bit-identical）。不命中返 False。
    **审计根治**：透传 4 参→modal_op_of D:11 readback（gate ON 时非 frozenset 情态词亦判·与主调一致）。
    默认 None→退化纯 frozenset（既有 caller 无 4 参·bit-identical）。
    """
    return modal_op_of(token, lang, backend=backend, edge_store=edge_store,
                       space_id=space_id, concept_index=concept_index) is not None


# ---- 程度 degree cue（#1134·degree 副词→Rational intensity·**file-driven 非 §九 frozenset**·gate DEGREE_MODE） ----
# doc/重来_程度属性器intensity_2026-07-16.md。程度副词（很/非常/极其/较/稍·强度 2/1·3/2·2/5·Rational·非 float）
# = 属性器命题值强度缩放（平行 pol/mod·非独立机制）。**异 modal/negation 的 §九 code frozenset**：程度是语义强度
# （很=2/1 是语义量级·非句法位置锚）·故 cue+intensity **全来自外部 degree_cues_{lang}.txt**（loader resolve_degree_facts）·
# boot populate module cache·core 不 import 文件·生产 default 无文件→空 cache→is_degree_cue 恒 False→bit-identical。
# gate DEGREE_MODE（默认 OFF）·OFF→degree_intensity_of 返 None·既有 property 窗口 intensity 恒 1/1·bit-identical。
_DEGREE_CUES: dict[int, dict[str, tuple[int, int]]] = {}   # lang -> {cue: (num, den)}·boot populate（mutable module cache）


def populate_degree_cues(lang: int, mapping: dict[str, tuple[int, int]]) -> None:
    """boot populate degree cue→intensity cache（formal_train boot 调·file-driven·非 §九 frozenset·#1134）。

    formal_train resolve_degree_facts(lang) → dict[cue]=(num,den) → 本函数喂 _DEGREE_CUES[lang]。
    空 mapping → no-op（无文件/未知 lang·bit-identical 守·不污染 cache）。幂等 update（同 lang 重 boot 合并）。

    铁律：纯整数（num/den int）/ 确定性（dict 内容序无关）/ 不写死（数据来自外部文件·core 不 import）。
    """
    if not mapping:
        return   # 空映射 no-op（生产 default 无文件→不污染 cache→is_degree_cue 恒 False→bit-identical）
    _DEGREE_CUES.setdefault(lang, {}).update(mapping)


def degree_intensity_of(token: str, lang: int) -> tuple[int, int] | None:
    """token 是否程度副词 → (num, den) intensity / None（#1134·gate DEGREE_MODE·file-driven cache）。

    gate DEGREE_MODE OFF（默认）→ 返 None（既有窗口 intensity 恒 1/1·bit-identical 守）。
    ON → ``_DEGREE_CUES[lang].get(token)``（boot populate 的 file-driven cache·空 cache 返 None·冷启动退化）。
    独立于 cue_type_of（不入 _CUE_WORDS·防 很/非常 污染 extract_cues 邻居判·同 is_modal_cue 范式）。
    extract_property_claims degree 窗口据此填 intensity（tokens[val_idx] 是 degree cue→value 后移+intensity）。

    返 (num, den) Rational intensity（很/非常=2/1·较=3/2·稍=2/5·正缩放）/ None（非程度词 或 gate OFF）。
    **诚实边界**：intensity magnitude 暂无消费者（G3b 读 PROPERTY 出边 count·judge 只权 CAUSES/PRECEDES）·
    consumer defer（intensity-aware A1 聚合 / degree-comparison·revisit）·degree wired-but-dormant（gate OFF default）。
    """
    if not getattr(gates, "DEGREE_MODE", False):
        return None   # gate OFF → None（既有窗口不变·bit-identical 守）
    return _DEGREE_CUES.get(lang, {}).get(token)


def is_degree_cue(token: str, lang: int) -> bool:
    """token 是否程度副词（degree_intensity_of is not None·邻居判用·#1134）。

    独立于 cue_type_of（不入 _CUE_WORDS·防程度词污染 extract_cues·同 is_modal_cue 范式）。
    extract_property_claims 据此判 degree 窗口（"X 的 Y 是 [非常] Z"·degree at val_idx→value 后移）·gate DEGREE_MODE 守
    （OFF → degree_intensity_of 返 None → 既有窗口 intensity 恒 1/1·bit-identical）。不命中返 False。
    """
    return degree_intensity_of(token, lang) is not None


# ---- 刀 D 比较 cue（独立于 cue_type_of·不入 _CUE_WORDS·bit-identical-safe·镜像 _ARITH_OP_WORDS 范式） ----
# 设计 doc/重来_刀D比较cue设计_2026-07-09.md §四。比较声明 = NUM 比较OP NUM·比较 OP 词（大于/小于/不小于/不大于）
# 既是声明锚又是序方向。**不入 _CUE_WORDS**（异刀B 等于入 _CUE_WORDS）：大于/小于若入 _CUE_WORDS 会让
# cue_type_of(大于) 返非 None → extract_cues 邻居判把 大于 当 cue 跳过配对 → 改变 CAUSES/IS_A/PRECEDES
# 提取行为 → 非 bit-identical。故比较 OP 检测走独立 helpers（同 _ARITH_OP_WORDS 范式·非 cue_type）·
# cue_type_of 对 大于/小于 仍返 None（零行为变·比刀 B 更 safe）。opcode 在 crosscut/integer/compare.CMP_*。
_COMPARISON_OP_WORDS: dict[int, dict[str, int]] = {
    LANG_ZH: {
        "大于": CMP_GT,   # NUM 大于 NUM ·a > b
        "小于": CMP_LT,   # NUM 小于 NUM ·a < b
        "不小于": CMP_GE,  # NUM 不小于 NUM ·a ≥ b（≥ 的单 token 表·"大于等于"多字 token defer）
        "不大于": CMP_LE,  # NUM 不大于 NUM ·a ≤ b（≤ 的单 token 表·"小于等于"多字 token defer）
        # 注：等于/equal_to **不入** _COMPARISON_OP_WORDS——等于属刀B ARITH_EQUALS_CUE（数值等式声明·
        # _CUE_WORDS·extract_numeric_claims 消费）。code_problem 条件等式经 cue_type_of==ARITH_EQUALS_CUE→CMP_EQ
        # 复用此单源注册（piece 2.1·避双注册：否则 extract_comparison_claims 对"二加三等于五"误抽假比较声明 3==5）。
    },
    LANG_EN: {
        "greater_than": CMP_GT,   # NUM greater_than NUM（whitespace tokenize 须 caller 切·首版窄域）
        "less_than": CMP_LT,      # NUM less_than NUM
        "at_least": CMP_GE,       # NUM at_least NUM ·a ≥ b
        "at_most": CMP_LE,        # NUM at_most NUM ·a ≤ b
        # EN 等式词 equals 亦不入此表（同上·属 ARITH_EQUALS_CUE·code_problem 复用）。
    },
}


def comparison_op_of(token: str, lang: int, *,
                     backend=None, edge_store=None,
                     space_id: int | None = None, concept_index=None) -> int | None:
    """token 是否比较 OP 词（刀 D·大于/小于/不小于/不大于·元定义层固化）·返 CMP_* / None。

    **两源**（STEP5 PR2·镜像 arith_op_of / cue_type_of 范式）：
      第一源（既有·元定义固化）：_COMPARISON_OP_WORDS frozenset exact 匹配（返 CMP_*）。
      第二源（STEP5 PR2 新增·D:11 readback·gate OPERATOR_D11_READBACK_MODE）：
        lookup token → word_ref → lookup_word_operator 读 D:11 PRIMARY 边 → OP_*（过滤比较 OP·OP_GT/LT/GE/LE）
        → _OP_TO_OPCODE → CMP_*。反 theater：未验证 SHADOW 不注入（tier_filter=TIER_PRIMARY）。

    **bit-identical 守卫**：gate OFF（默认）→ 只走第一源·退化纯 frozenset·回归零翻。
    exact token 匹配（caller 须切比较 OP 词为独立 token·同 cue/arith_op 词纪律）。
    非 OP 词 / 他 lang 返 None（extract_comparison_claims 据此跳·守反统计契约·不凑配）。
    独立于 cue_type_of（不入 _CUE_WORDS·防 大于/小于 污染 extract_cues·见上注）。
    过滤比较 OP（OP_GT/LT/GE/LE）·非算术 OP（OP_ADD/SUB/MUL）·无交叉污染（arith_op_of 同范过滤算术 OP）。
    """
    # 第一源：frozenset exact 匹配（既有·元定义固化）
    cmp = _COMPARISON_OP_WORDS.get(lang, {}).get(token)
    if cmp is not None:
        return cmp
    # 第二源：D:11 readback（STEP5 PR2·gate 守·反 theater）
    if not getattr(gates, "OPERATOR_D11_READBACK_MODE", False):
        return None
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return None   # 参数不全→退化（不读 D:11）
    return _comparison_op_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _comparison_op_from_d11_primary(token: str, space_id: int,
                                    backend, edge_store, concept_index) -> int | None:
    """STEP5 PR2：D:11 PRIMARY 边 readback → CMP_*（比较 OP·反 theater·冷启动返 None）。

    flow：concept_index.lookup(token, space_id) → word_ref | None
      → lookup_word_operator(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → [(op_ref, op_kind), ...] → 过滤 is_comparison_op_kind → op_kind_to_opcode → 首命中 CMP_* | None。

    只读 TIER_PRIMARY D:11 边（反 theater）。过滤比较 OP（OP_GT/LT/GE/LE）·非算术 OP·无交叉污染。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return None
    from pure_integer_ai.cognition.shared.operator_primitives import (
        lookup_word_operator, is_comparison_op_kind, op_kind_to_opcode,
    )
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    ops = lookup_word_operator(backend, edge_store, word_ref,
                               space_id=space_id, tier_filter=TIER_PRIMARY)
    for _op_ref, op_kind in ops:
        if not is_comparison_op_kind(op_kind):
            continue   # 非比较 OP（算术 OP）·comparison_op_of 只认比较·skip
        opcode = op_kind_to_opcode(op_kind)
        if opcode is not None:
            return opcode
    return None


def is_comparison_op_token(token: str, lang: int, *,
                           backend=None, edge_store=None,
                           space_id: int | None = None, concept_index=None) -> bool:
    """token 是否任一比较 OP（exact·守反统计·配对两端不取 OP token·同 extract_cues:66 邻居判）。
    STEP5 PR2：透传 4 参→comparison_op_of D:11 readback（gate ON 时非 frozenset OP 词亦判·与主调一致）。
    默认 None→退化纯 frozenset（extract_cues 既有 caller 无 4 参·bit-identical）。
    """
    return comparison_op_of(token, lang, backend=backend, edge_store=edge_store,
                            space_id=space_id, concept_index=concept_index) is not None


# ---- 条件结构 cue（language→code piece 2·closed-class 句法锚·元定义层固化·§九例外） ----
# 设计 doc/重来_语言通用接地_2026-07-16 §七-bis。条件结构词（如果/那么/否则·if/then/else）= 控流保留字·
# **闭类句法锚**（有限基数·标结构槽·无指称内容·同 加→ADD / 大于→CMP_GT）→ §九元定义 frozenset 种子
# （非外部数据文件·构造非值·异数字词 number_facts·非写死行为·非语义关联）。
# **独立 _CUE_WORDS**（不入 extract_cues 邻居判·bit-identical-safe·镜像 _COMPARISON_OP_WORDS 范式）：
# cue_type_of(如果/那么/否则) 仍返 None（零行为变）→ CAUSES/IS_A/PRECEDES 提取不变。
# 唯一消费者 = code_problem.code_problem_value（无生产 caller·CI 零调用→bit-identical）。
_COND_IF = 1     # 如果 / if
_COND_THEN = 2   # 那么 / then
_COND_ELSE = 3   # 否则 / else
_COND_KEYWORDS: dict[int, dict[str, int]] = {
    LANG_ZH: {"如果": _COND_IF, "那么": _COND_THEN, "否则": _COND_ELSE},
    LANG_EN: {"if": _COND_IF, "then": _COND_THEN, "else": _COND_ELSE},
}


def cond_keyword_of(token: str, lang: int) -> int | None:
    """token 是否条件结构词（如果/那么/否则·元定义层固化·closed-class 句法锚·§九例外）·返 _COND_* / None。

    **第一源（元定义 frozenset exact 匹配·本函数仅此源）**：_COND_KEYWORDS（加/大于/如果 同为 closed-class
    句法锚·§九元定义种子·非外部数据·非写死行为）。
    **第二源 D:11 readback defer**：异 comparison_op_of/arith_op_of 两源——条件结构 cue 无对应 COND_* D:11
    原语类型（须新建 control-flow-keyword 原语·piece 2.x+ defer·非本版 scope）。故本函数单源·教师无法经
    D:11 注新条件词（倘/假使 defer）·诚实单源（非镜像 comparison_op_of 两源机制·仅镜像其 frozenset 第一源）。

    独立于 cue_type_of（不入 _CUE_WORDS·防 如果/那么/否则 污染 extract_cues·bit-identical·镜像 _COMPARISON_OP_WORDS）。
    非条件结构词 / 他 lang 返 None（code_problem_value 据此跳·守反统计契约·不凑配）。
    """
    return _COND_KEYWORDS.get(lang, {}).get(token)


def cue_type_of(token: str, lang: int, *,
                backend=None, edge_store=None,
                space_id: int | None = None, concept_index=None) -> int | None:
    """token 是否是某类 cue（exact 匹配）·返 cue_type / None。

    exact token 匹配（caller 须将指向词切为独立 token·首版纪律）。
    不命中返 None·extractor 据此跳（守反统计契约·不凑配）。

    **两源**（刀4 决断5）：
      第一源（既有·元定义固化）：_CUE_WORDS frozenset exact 匹配（reward 不调·断奶前后不变）。
      第二源（刀4 新增·D:11 readback·gate EMERGENT_RELATION_CUE_READBACK_MODE）：
        lookup token → word_ref（concept_index.lookup）→ lookup_word_concept 读 D:11 PRIMARY 边
        → REL_* → _REL_KIND_TO_CUE_TYPE 映射。**反 theater 关键**：涌现学习成果（promote PRIMARY）
        反馈到 cue 识别·冷启动 frozenset 不含的词（如"引发"）经涌现晋升后第二轮返非 None。

    **bit-identical 守卫**：gate OFF（默认）→ 只走第一源·退化纯 frozenset·回归零翻。
      冷启动（D:11 全 SHADOW 未 promote）→ 第二源返 None → 退化 frozenset。
      第二源只读 TIER_PRIMARY D:11 边（lookup_word_concept tier_filter=TIER_PRIMARY·未验证 SHADOW 不注入）。
    """
    # 第一源：frozenset exact 匹配（既有·元定义固化）
    lang_set = _CUE_WORDS.get(lang)
    if lang_set is not None:
        for cue_type, words in lang_set.items():
            if token in words:
                return cue_type
    # 第二源：D:11 readback（刀4·gate 守·反 theater）
    if not getattr(gates, "EMERGENT_RELATION_CUE_READBACK_MODE", False):
        return None
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return None   # 参数不全→退化（不读 D:11）
    return _cue_type_from_d11_primary(token, lang, space_id, backend,
                                       edge_store, concept_index)


def _cue_type_from_d11_primary(token: str, lang: int, space_id: int,
                               backend, edge_store, concept_index) -> int | None:
    """刀4 决断5：D:11 PRIMARY 边 readback → cue_type（反 theater 关键·冷启动返 None）。

    flow：旧概念索引或词形索引 lookup(token, lang, space_id) → word_ref | None（词未概念化→None）
      → lookup_word_concept(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY)
      → [(rel_ref, rel_kind), ...] → _REL_KIND_TO_CUE_TYPE.get(rel_kind) → 首命中 cue_type | None。

    只读 TIER_PRIMARY D:11 边（已验证晋升·未验证 SHADOW 不注入·反 theater）。
    词未概念化（concept_index.lookup 返 None）/ 无 D:11 PRIMARY 边 / rel_kind 无映射 → None。
    """
    # 先查旧的 surface→概念身份，再查词形域的类型化身份；两者可并存。
    word_refs = []
    concept_ref = concept_index.lookup(token, space_id)
    if concept_ref is not None:
        word_refs.append(concept_ref)
    from pure_integer_ai.cognition.understanding.word_form_index import WordFormIndex
    word_ref = WordFormIndex(backend, concept_index).lookup(
        token, language=lang, space_id=space_id)
    if word_ref is not None and word_ref not in word_refs:
        word_refs.append(word_ref)
    if not word_refs:
        return None   # 词未概念化（冷启动·未 observe）·退化
    from pure_integer_ai.cognition.understanding.word_concept_signal import lookup_word_concept
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    for word_ref in word_refs:
        rels = lookup_word_concept(backend, edge_store, word_ref,
                                   space_id=space_id, tier_filter=TIER_PRIMARY)
        for _rel_ref, rel_kind in rels:
            cue_type = _REL_KIND_TO_CUE_TYPE.get(rel_kind)
            if cue_type is not None:
                return cue_type
    return None


def is_cue_token(token: str, lang: int) -> bool:
    """token 是否任一类 cue（exact）。"""
    return cue_type_of(token, lang) is not None
