"""cognition.understanding.cue_words — 来源化语言信号主读与旧词表兼容投影。

自然语言表示的正式作用由 U-04 LanguageAtom/Representation/MinimalInstruction 图和调用方
绑定决定。模块内 Python 词表及旧 D:11 只服务显式 compatibility，不能计 readiness。

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

# ---- 迁移兼容词集（lang → {cue_type: frozenset[word]}) ----
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

# ---- 数值等式算子兼容词表（刀 B·非 cue_type·表达式算子识别用） ----
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


def _graph_bound_integer(
        token: str, lang: int, *, language_signal_runtime,
        instruction_bindings: tuple[tuple[tuple[int, ...], int], ...],
        label: str) -> tuple[bool, int | None]:
    """把一致图指令解析为调用方整数作用，并保留冲突与无证据的区别。"""
    resolution = language_signal_runtime.resolve_instruction(
        token, language=lang)
    if not resolution.has_evidence:
        return False, None
    if resolution.instruction_key is None:
        return True, None
    binding_map: dict[tuple[int, ...], int] = {}
    for instruction_key, value in instruction_bindings:
        if (not isinstance(instruction_key, tuple)
                or not instruction_key
                or any(type(item) is not int for item in instruction_key)):
            raise TypeError(f"{label} instruction_key 必须是非空严格整数 tuple")
        if type(value) is not int:
            raise TypeError(f"{label} 作用值必须是严格整数")
        if instruction_key in binding_map:
            raise ValueError(f"{label} instruction_key 绑定重复")
        binding_map[instruction_key] = value
    return True, binding_map.get(resolution.instruction_key)


def _graph_matches_instruction(
        token: str, lang: int, *, language_signal_runtime,
        instruction_key: tuple[int, ...] | None) -> bool | None:
    """保留图无证据状态，并让缺失目标绑定在有证据时 fail closed。"""
    resolution = language_signal_runtime.resolve_instruction(
        token, language=lang)
    if not resolution.has_evidence:
        return None
    if instruction_key is None or resolution.instruction_key is None:
        return False
    if (not isinstance(instruction_key, tuple) or not instruction_key
            or any(type(item) is not int for item in instruction_key)):
        raise TypeError("instruction_key 必须是非空严格整数 tuple")
    return resolution.instruction_key == instruction_key


def arith_op_of(token: str, lang: int, *,
                backend=None, edge_store=None,
                space_id: int | None = None, concept_index=None,
                language_signal_runtime=None,
                arithmetic_instruction_bindings: tuple[
                    tuple[tuple[int, ...], int], ...] = (),
                language_signal_compatibility_enabled: bool = True,
                ) -> int | None:
    """按一致图指令和调用方绑定解析算术作用，必要时读取迁移兼容源。

    图中存在候选但冲突或未绑定时返回 ``None``，旧词表和 D:11 不得覆盖。
    """
    if language_signal_runtime is not None:
        has_evidence, value = _graph_bound_integer(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_bindings=arithmetic_instruction_bindings,
            label="算术",
        )
        if has_evidence:
            return value
    if not language_signal_compatibility_enabled:
        return None
    # 迁移期第一兼容源：Python 字面词表。
    op = _ARITH_OP_WORDS.get(lang, {}).get(token)
    if op is not None:
        return op
    # 迁移期第二兼容源：D:11 PRIMARY readback。
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


def is_property_attr_marker(
        token: str, lang: int, *, language_signal_runtime=None,
        property_attr_instruction_key: tuple[int, ...] | None = None,
        language_signal_compatibility_enabled: bool = True) -> bool:
    """按图候选优先、旧词表显式兼容的顺序判断属性标记。

    图中一致候选只有匹配调用方注入的属性标记指令键才为真；混合候选或明确的
    其他指令直接为假。只有图中完全无候选且兼容开关开启时才读取旧字面词表。
    """
    if language_signal_runtime is not None:
        graph_result = _graph_matches_instruction(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_key=property_attr_instruction_key,
        )
        if graph_result is not None:
            return graph_result
    if not language_signal_compatibility_enabled:
        return False
    return token in _PROPERTY_ATTR_MARKER.get(lang, frozenset())


def is_property_value_copula(
        token: str, lang: int, *, language_signal_runtime=None,
        property_value_instruction_key: tuple[int, ...] | None = None,
        language_signal_compatibility_enabled: bool = True) -> bool:
    """按图候选优先、旧词表显式兼容的顺序判断属性值系词。

    裸系词与 IS_A 多字表示保持不同作用。图中冲突、未匹配目标或兼容关闭时均
    fail closed，避免旧字面“是”覆盖来源化证据。
    """
    if language_signal_runtime is not None:
        graph_result = _graph_matches_instruction(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_key=property_value_instruction_key,
        )
        if graph_result is not None:
            return graph_result
    if not language_signal_compatibility_enabled:
        return False
    return token in _PROPERTY_VALUE_COPULA.get(lang, frozenset())


def is_property_possess_cue(token: str, lang: int, *,
                            backend=None, edge_store=None,
                            space_id: int | None = None, concept_index=None,
                            language_signal_runtime=None,
                            property_possess_instruction_key: tuple[int, ...] | None = None,
                            language_signal_compatibility_enabled: bool = True) -> bool:
    """按来源化图候选判断领属 cue，并显式隔离迁移兼容源。

    图中一致候选只有匹配调用方注入的领属指令键才为真；冲突或其他指令直接为
    假。图完全无证据且兼容开启时，才依次读取 Python 字面和 D:11 PRIMARY；
    D:11 仍只接受已晋升的 REL_PROPERTY，不得计作 U-04 readiness。
    """
    if language_signal_runtime is not None:
        graph_result = _graph_matches_instruction(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_key=property_possess_instruction_key,
        )
        if graph_result is not None:
            return graph_result
    if not language_signal_compatibility_enabled:
        return False
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


# ---- STEP5 PR4：REL_SIMILAR 相似 cue（U-04 图主读，D:11 仅兼容） ----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §四-bis STEP5 PR4。
# 来源化图负责表示到 MinimalInstruction 的关联，调用方注入相似作用键；旧 D:11 只在图无证据且
# compatibility 开启时读取，不能计入 U-04 readiness。

def is_similar_cue(token: str, lang: int, *,
                   backend=None, edge_store=None,
                   space_id: int | None = None, concept_index=None,
                   language_signal_runtime=None,
                   similar_instruction_key: tuple[int, ...] | None = None,
                   language_signal_compatibility_enabled: bool = True) -> bool:
    """按来源化图候选判断相似 cue，并显式隔离旧 D:11 兼容源。

    图中一致候选只有匹配调用方注入的相似作用键才为真；冲突或未绑定直接为假。
    只有图完全无证据且兼容开启时才读取已晋升的 D:11 REL_SIMILAR。
    """
    if language_signal_runtime is not None:
        graph_result = _graph_matches_instruction(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_key=similar_instruction_key,
        )
        if graph_result is not None:
            return graph_result
    if not language_signal_compatibility_enabled:
        return False
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
# U-04 迁移后，来源化 LanguageAtom/Representation/MinimalInstruction 图是主读。
# 下列词表和 D:11 只服务显式开启的迁移兼容路径，不得计 readiness；关闭兼容后缺图证据 fail closed。
_NEGATION_CUES: dict[int, frozenset[str]] = {
    LANG_ZH: frozenset({"不", "没", "非", "无"}),   # X 的 Y 不 是 Z（不）/ 没（罕·没是）/ 非文言 / 无文言
    LANG_EN: frozenset({"not", "no", "never"}),     # 英文例：X's Y is not Z / no Y is Z / never
}


def is_negation_cue(token: str, lang: int, *,
                    backend=None, edge_store=None,
                    space_id: int | None = None, concept_index=None,
                    language_signal_runtime=None,
                    negation_instruction_key: tuple[int, ...] | None = None,
                    language_signal_compatibility_enabled: bool = True) -> bool:
    """按图候选优先、旧源显式兼容的顺序判断 token 是否表达否定。

    调用方同时注入语言信号 runtime 和否定指令键时，先读取全部图候选：一致命中
    才为真，明确非目标或混合候选直接为假，旧词表不得覆盖冲突。只有图中完全
    无候选且 compatibility 开启时，才读取 Python 词表和 D:11 PRIMARY 旧源。
    compatibility 关闭后缺图证据一律 fail closed。
    """
    if language_signal_runtime is not None:
        graph_result = _graph_matches_instruction(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_key=negation_instruction_key,
        )
        if graph_result is not None:
            return graph_result
    if not language_signal_compatibility_enabled:
        return False
    # 迁移期第一兼容源：Python 字面词表。
    if token in _NEGATION_CUES.get(lang, frozenset()):
        return True
    # 迁移期第二兼容源：D:11 PRIMARY readback。
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
# U-04 后来源化图为主读；下列 Python/D:11 只作显式迁移兼容，不能计 readiness。
# **动作意图=符号域先天**（镜像 operator·异 modal·doc §16.3）·D:11 readback=文字 alias 可学习（同否定词/算子词）·非概念可学。
# **覆盖**（doc §16.5）：引导词祈使（帮我生成·命令词+动作词）+ 有动作词裸祈使（生成代码·仅动作词）。
# 纯句式祈使（去开门·无引导词无动作词）冷启动漏判·defer B-PR2 experience_count 回写扩散（结构意图后天学·§13.8）。
def is_action_intent_cue(token: str, lang: int, *,
                         backend=None, edge_store=None,
                         space_id: int | None = None, concept_index=None,
                         language_signal_runtime=None,
                         action_instruction_bindings: tuple[
                             tuple[tuple[int, ...], int], ...] = (),
                         language_signal_compatibility_enabled: bool = True) -> bool:
    """按图中动作 kind 绑定判断命令或动作 cue，旧源仅作显式兼容。

    图中有候选时必须得到唯一且合法的调用方动作 kind；混合、未绑定或非法候选
    均为假，并压住 Python/D:11。图完全无证据时才按兼容开关读取旧词表和
    PRIMARY D:11。该布尔入口只服务 intent 判断，ConceptRef 消费走
    ``action_intent_targets_of``，不能从布尔值伪造对象身份。
    """
    if language_signal_runtime is not None:
        has_evidence, action_kind = _graph_action_kind(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            action_instruction_bindings=action_instruction_bindings,
        )
        if has_evidence:
            return action_kind is not None
    if not language_signal_compatibility_enabled:
        return False
    from pure_integer_ai.cognition.shared.action_primitives import _ACTION_LEXICAL_CUE
    if token in _ACTION_LEXICAL_CUE.get(lang, {}):
        return True
    if not getattr(gates, "ACTION_D11_READBACK_MODE", False):
        return False
    if backend is None or edge_store is None or space_id is None or concept_index is None:
        return False   # 参数不全→退化（不读 D:11）
    return _action_intent_from_d11_primary(token, space_id, backend, edge_store, concept_index)


def _graph_action_kind(
        token: str, lang: int, *, language_signal_runtime,
        action_instruction_bindings: tuple[
            tuple[tuple[int, ...], int], ...]) -> tuple[bool, int | None]:
    """把一致图指令解析为合法动作 kind，并保留无证据与拒绝状态。"""
    has_evidence, action_kind = _graph_bound_integer(
        token, lang,
        language_signal_runtime=language_signal_runtime,
        instruction_bindings=action_instruction_bindings,
        label="动作意图",
    )
    if not has_evidence or action_kind is None:
        return has_evidence, None
    from pure_integer_ai.cognition.shared.action_primitives import (
        is_action_class_kind,
        is_command_mood_kind,
    )
    if not (is_command_mood_kind(action_kind)
            or is_action_class_kind(action_kind)):
        raise ValueError("动作意图绑定值不属于已定义的动作 kind")
    return True, action_kind


def action_intent_of(
        token: str, lang: int, *, backend=None, edge_store=None,
        space_id: int | None = None, concept_index=None,
        language_signal_runtime=None,
        action_instruction_bindings: tuple[
            tuple[tuple[int, ...], int], ...] = (),
        language_signal_compatibility_enabled: bool = True) -> int | None:
    """返回图优先的动作 kind；兼容 D:11 只有唯一 kind 时才投影为单值。"""
    if language_signal_runtime is not None:
        has_evidence, action_kind = _graph_action_kind(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            action_instruction_bindings=action_instruction_bindings,
        )
        if has_evidence:
            return action_kind
    if not language_signal_compatibility_enabled:
        return None
    from pure_integer_ai.cognition.shared.action_primitives import _ACTION_LEXICAL_CUE
    lexical_kind = _ACTION_LEXICAL_CUE.get(lang, {}).get(token)
    if lexical_kind is not None:
        return lexical_kind
    if (not getattr(gates, "ACTION_D11_READBACK_MODE", False)
            or backend is None or edge_store is None
            or space_id is None or concept_index is None):
        return None
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return None
    from pure_integer_ai.cognition.shared.action_primitives import lookup_word_action
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    kinds = {
        kind for _action_ref, kind in lookup_word_action(
            backend, edge_store, word_ref,
            space_id=space_id, tier_filter=TIER_PRIMARY)
    }
    return next(iter(kinds)) if len(kinds) == 1 else None


def action_intent_targets_of(
        token: str, lang: int, *, backend, edge_store,
        space_id: int, concept_index,
        language_signal_runtime=None,
        action_instruction_bindings: tuple[
            tuple[tuple[int, ...], int], ...] = (),
        action_primitive_refs: tuple[
            tuple[int, tuple[int, int]], ...] = (),
        language_signal_compatibility_enabled: bool = True,
        ) -> tuple[tuple[tuple[int, int], int], ...]:
    """返回动作词对应的一等动作 ConceptRef 与 kind，不从布尔 cue 伪造 ref。"""
    if language_signal_runtime is not None:
        has_evidence, action_kind = _graph_action_kind(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            action_instruction_bindings=action_instruction_bindings,
        )
        if has_evidence:
            if action_kind is None:
                return ()
            primitive_map = dict(action_primitive_refs)
            action_ref = primitive_map.get(action_kind)
            return () if action_ref is None else ((action_ref, action_kind),)
    if not language_signal_compatibility_enabled:
        return ()
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return ()
    from pure_integer_ai.cognition.shared.action_primitives import lookup_word_action
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    return tuple(sorted(
        lookup_word_action(
            backend, edge_store, word_ref,
            space_id=space_id, tier_filter=TIER_PRIMARY),
        key=lambda item: (item[0][0], item[0][1], item[1]),
    ))


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


def collect_action_intent_concepts(
        segments, *, backend, edge_store, space_id: int, concept_index,
        language_signal_runtime=None,
        action_instruction_bindings: tuple[
            tuple[tuple[int, ...], int], ...] = (),
        action_primitive_refs: tuple[
            tuple[int, tuple[int, int]], ...] = (),
        language_signal_compatibility_enabled: bool = True,
        ) -> list[tuple[tuple[int, int], int]]:
    """B-PR2：收集 segments 中命中 D:11 PRIMARY ACTION_* concept 的 distinct refs（doc §17·experience_count feed 用）。

    扫 segments.tokens·``concept_index.lookup(tok, space_id) → word_ref | None``·
    ``lookup_word_action(backend, edge_store, word_ref, space_id, tier_filter=TIER_PRIMARY) → [(action_ref, kind)]``·
    distinct by action_ref（同 episode 同 ACTION_* concept 只返一次·镜像 reward_propagate concept_targets set 去重 :208-210）。

    U-04 图路径先把 MinimalInstruction 映射到调用方 action kind，再由当前 context 的
    ``action_primitive_refs`` 映射到一等动作 ConceptRef；缺 ref 时不得从布尔 cue 伪造。
    图无证据且兼容开启时才保留旧 D:11 PRIMARY ConceptRef 路径。

    兼容 D:11 仍固定 ``TIER_PRIMARY``，未验证 SHADOW alias 不注入验证率。

    **返 kind**（int_a·0=COMMAND_MOOD / 1-4=ACTION_*）：caller（formal_train hook）写 experience_count 不区分 kind（同 rate 桶）·
    kind 仅供 caller 日志/未来 B-PR3 按类分流感（B-PR3 读 int_a 分流·非本 collector 责）。

    **无 gate**（纯读 collector·gate 在 formal_train hook 守 ACTION_EXPERIENCE_FEED_MODE·gate OFF 不调本函数）。
    纯读（concept_index.lookup + lookup_word_action 均 select/read·无 insert/update·设计审 D CONFIRMED）。
    """
    seen: set = set()
    out: list[tuple[tuple[int, int], int]] = []
    for seg in segments:
        for tok in seg.tokens:
            for action_ref, kind in action_intent_targets_of(
                    tok, seg.lang,
                    backend=backend, edge_store=edge_store,
                    space_id=space_id, concept_index=concept_index,
                    language_signal_runtime=language_signal_runtime,
                    action_instruction_bindings=action_instruction_bindings,
                    action_primitive_refs=action_primitive_refs,
                    language_signal_compatibility_enabled=(
                        language_signal_compatibility_enabled)):
                if action_ref in seen:
                    continue   # 同 concept distinct 去重（同 episode 同 ACTION_* 只 feed 一次）
                seen.add(action_ref)
                out.append((action_ref, kind))
    return out


def collect_action_intent_word_decisions(
        segments, *, space_id: int, concept_index,
        language_signal_runtime=None,
        action_instruction_bindings: tuple[
            tuple[tuple[int, ...], int], ...] = (),
        language_signal_compatibility_enabled: bool = True,
        ) -> dict[tuple[int, int], bool]:
    """为 dag_path 记录图已裁决的词概念，阻止后续 D:11 覆盖冲突。"""
    decisions: dict[tuple[int, int], bool] = {}
    if language_signal_runtime is None:
        return decisions
    for seg in segments:
        for token in seg.tokens:
            word_ref = concept_index.lookup(token, space_id)
            if word_ref is None:
                continue
            has_evidence, action_kind = _graph_action_kind(
                token, seg.lang,
                language_signal_runtime=language_signal_runtime,
                action_instruction_bindings=action_instruction_bindings,
            )
            if not has_evidence and language_signal_compatibility_enabled:
                continue
            decision = action_kind is not None if has_evidence else False
            previous = decisions.get(word_ref)
            decisions[word_ref] = (
                decision if previous is None else previous and decision)
    return decisions


# ---- B2 情态 cue（独立于 cue_type_of·不入 _CUE_WORDS·镜像 _NEGATION_CUES 范式·P0.3 modality 填值） ----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §七 B2。情态词（必然/可能/也许/必须/应该/可以）在"是"前
# （j-1·同否定词槽位·与 negation 互斥·先查 modal）→ extract_property_claims 情态窗口（modality_on 参数·_gated 传
# 本 gate）·modality 填值（0-4）·命题节点建独立 surface 后缀 _0_{mod}（P0.3）·G3b 各判多值·模态对当扩展在 PR3。
# 返 modality 编码：0=实然/1=□必然/2=◇可能/3=道义必然/4=道义可能（P0.3 surface 后缀 _{pol}_{mod}）。
# 独立 helper（不入 _CUE_WORDS·防 必然/可能 污染 extract_cues 邻居判·同 _NEGATION_CUES 范式）。
# 守墙：T 公理形式层墙内（构造性检查·非 truth）·实质情态真值（认识/规范 W2 + 动力 W1）defer。
# U-04 迁移后，来源化语言信号图是情态主读，指令到 modality 值的作用由调用方注入。
# 下列词表和 D:11 只服务显式开启的迁移兼容路径，不得计 readiness。
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
                space_id: int | None = None, concept_index=None,
                language_signal_runtime=None,
                modality_instruction_bindings: tuple[
                    tuple[tuple[int, ...], int], ...] = (),
                language_signal_compatibility_enabled: bool = True,
                ) -> int | None:
    """按图中一致指令和调用方绑定解析情态作用，必要时读取迁移兼容源。

    图中存在混合指令或一致指令未出现在绑定中时返回 ``None``，不得回退旧源。
    只有图无候选时，compatibility 开关才允许读取 Python 词表和 D:11 PRIMARY。
    T 公理等形式层处理只提供构造性机制，不证明实质情态真值；认识和规范真值
    仍受 W2 限制，动力情态的物理接地仍受 W1 限制并保持 defer。
    """
    if language_signal_runtime is not None:
        has_evidence, value = _graph_bound_integer(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_bindings=modality_instruction_bindings,
            label="情态",
        )
        if has_evidence:
            if value is not None and value <= 0:
                raise ValueError("情态作用值必须是严格正整数")
            return value
    if not language_signal_compatibility_enabled:
        return None
    # 迁移期第一兼容源：Python 字面词表。
    op = _MODAL_CUES.get(lang, {}).get(token)
    if op is not None:
        return op
    # 迁移期第二兼容源：D:11 PRIMARY readback。
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
                 space_id: int | None = None, concept_index=None,
                 language_signal_runtime=None,
                 modality_instruction_bindings: tuple[
                     tuple[tuple[int, ...], int], ...] = (),
                 language_signal_compatibility_enabled: bool = True) -> bool:
    """复用 modal_op_of 的图优先裁决，判断 token 是否有确定情态作用。"""
    return modal_op_of(token, lang, backend=backend, edge_store=edge_store,
                       space_id=space_id, concept_index=concept_index,
                       language_signal_runtime=language_signal_runtime,
                       modality_instruction_bindings=(
                           modality_instruction_bindings),
                       language_signal_compatibility_enabled=(
                           language_signal_compatibility_enabled)) is not None


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


# ---- 刀 D 比较 cue 兼容词表（独立于 cue_type_of·不入 _CUE_WORDS） ----
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
                     space_id: int | None = None, concept_index=None,
                     language_signal_runtime=None,
                     comparison_instruction_bindings: tuple[
                         tuple[tuple[int, ...], int], ...] = (),
                     language_signal_compatibility_enabled: bool = True,
                     ) -> int | None:
    """按一致图指令和调用方绑定解析比较作用，必要时读取迁移兼容源。

    图中存在候选但冲突或未绑定时返回 ``None``，旧词表和 D:11 不得覆盖。
    """
    if language_signal_runtime is not None:
        has_evidence, value = _graph_bound_integer(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_bindings=comparison_instruction_bindings,
            label="比较",
        )
        if has_evidence:
            return value
    if not language_signal_compatibility_enabled:
        return None
    # 迁移期第一兼容源：Python 字面词表。
    cmp = _COMPARISON_OP_WORDS.get(lang, {}).get(token)
    if cmp is not None:
        return cmp
    # 迁移期第二兼容源：D:11 PRIMARY readback。
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
                           space_id: int | None = None, concept_index=None,
                           language_signal_runtime=None,
                           comparison_instruction_bindings: tuple[
                               tuple[tuple[int, ...], int], ...] = (),
                           language_signal_compatibility_enabled: bool = True,
                           ) -> bool:
    """token 是否任一比较 OP（exact·守反统计·配对两端不取 OP token·同 extract_cues:66 邻居判）。
    STEP5 PR2：透传 4 参→comparison_op_of D:11 readback（gate ON 时非 frozenset OP 词亦判·与主调一致）。
    默认 None→退化纯 frozenset（extract_cues 既有 caller 无 4 参·bit-identical）。
    """
    return comparison_op_of(token, lang, backend=backend, edge_store=edge_store,
                            space_id=space_id, concept_index=concept_index,
                            language_signal_runtime=language_signal_runtime,
                            comparison_instruction_bindings=(
                                comparison_instruction_bindings),
                            language_signal_compatibility_enabled=(
                                language_signal_compatibility_enabled)) is not None


# ---- 条件结构 cue 兼容词表（language→code piece 2） ----
# 设计 doc/重来_语言通用接地_2026-07-16 §七-bis。条件结构词（如果/那么/否则·if/then/else）= 控流保留字·
# 正式结构槽由来源化图指令和调用方作用绑定决定；下表仅供迁移兼容。
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


def cond_keyword_of(
        token: str, lang: int, *, language_signal_runtime=None,
        condition_instruction_bindings: tuple[
            tuple[tuple[int, ...], int], ...] = (),
        language_signal_compatibility_enabled: bool = True) -> int | None:
    """按一致图指令和调用方绑定解析条件槽作用，必要时读取旧字面兼容源。

    图中存在候选但冲突或未绑定时返回 ``None``，compatibility 关闭后不读取
    Python 条件词表。条件作用只描述结构槽，不宣称条件内容为真。
    """
    if language_signal_runtime is not None:
        has_evidence, value = _graph_bound_integer(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_bindings=condition_instruction_bindings,
            label="条件",
        )
        if has_evidence:
            return value
    if not language_signal_compatibility_enabled:
        return None
    return _COND_KEYWORDS.get(lang, {}).get(token)


def cue_type_of(token: str, lang: int, *,
                backend=None, edge_store=None,
                space_id: int | None = None, concept_index=None,
                language_signal_runtime=None,
                cue_instruction_bindings: tuple[
                    tuple[tuple[int, ...], int], ...] = (),
                language_signal_compatibility_enabled: bool = True,
                ) -> int | None:
    """按一致图指令和调用方绑定解析关系/量化 cue 类型。

    图中存在候选但冲突或未绑定时返回 ``None``，不得由 Python 词表或 D:11
    覆盖。只有图无证据且 compatibility 开启时才读取迁移期旧源。
    """
    if language_signal_runtime is not None:
        has_evidence, value = _graph_bound_integer(
            token, lang,
            language_signal_runtime=language_signal_runtime,
            instruction_bindings=cue_instruction_bindings,
            label="关系 cue",
        )
        if has_evidence:
            return value
    if not language_signal_compatibility_enabled:
        return None
    # 迁移期第一兼容源：Python 字面词表。
    lang_set = _CUE_WORDS.get(lang)
    if lang_set is not None:
        for cue_type, words in lang_set.items():
            if token in words:
                return cue_type
    # 迁移期第二兼容源：D:11 PRIMARY readback。
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


def is_cue_token(
        token: str, lang: int, *, backend=None, edge_store=None,
        space_id: int | None = None, concept_index=None,
        language_signal_runtime=None,
        cue_instruction_bindings: tuple[
            tuple[tuple[int, ...], int], ...] = (),
        language_signal_compatibility_enabled: bool = True) -> bool:
    """复用 cue_type_of 的图优先裁决，判断 token 是否有确定 cue 作用。"""
    return cue_type_of(
        token, lang,
        backend=backend, edge_store=edge_store,
        space_id=space_id, concept_index=concept_index,
        language_signal_runtime=language_signal_runtime,
        cue_instruction_bindings=cue_instruction_bindings,
        language_signal_compatibility_enabled=(
            language_signal_compatibility_enabled)) is not None
