"""cognition.understanding.cue_extractor — 指向词/系词句法锚定提取（CAUSES 来源② / IS_A 来源②）。

**反统计契约（最关键·§8.1c-bis §7 同构禁令）**：
  - 只在指向词/系词命中时产 pair·不命中零 pair。
  - **绝无**"句中任意两概念 + 附近有指向词→共现式 N×N 配对"（否则滑统计·违排除法死刑同构）。
  - 一个 cue token 最多产一对（用其紧邻左右 token·锚定非穷举）。
  - pair 两端 = cue token 的紧邻非-cue token（首版·确定性·非语义判"中心词"·emergent_role defer）。

句法方向（cue_words.cue_type_of）：
  CAUSES_CUE_FORWARD  因 → [cue] → 果   pair = (left, right)
  CAUSES_CUE_BACKWARD 果 → [cue] → 因   pair = (right, left)   # 因在右
  IS_A_CUE            child → [cue] → parent  pair = (left, right)

**首版诚实边界**：
  - exact token 匹配（caller 须切 cue 为独立 token·空白切语料 caller-fill）。
  - 紧邻左右 token 作代表（非语义中心词）·tokenize 质量决定命中·真 tokenizer defer。
  - 边界 cue（句首/句末无左/右）跳·不凑配（守反统计）。
  - gate CUE_EXTRACTOR_MODE 默认 OFF（守回归 bit-identical·ON 启裸文本自产）。

落点：formal_train._split_item_to_segments 每句段调·填 Segment.cue_based_causal_pairs /
  Segment.is_a_pairs（段内 token index·切片后已重映射）。

**刀 B 数值等式声明**（extract_numeric_claims / extract_numeric_claims_gated·独立函数·不改 extract_cues
3-tuple 签名·隔离改动·降风险）：NUM OP NUM 等于 NUM 窗口扫描·填 Segment.numeric_claims（纯整数 4-tuple）·
构造性检查 SELF_PRODUCED（数 single-source·同刀 A 时序定位·闭包传 numeric_proof_fn 不入图）。

**刀 C 全称量化声明**（extract_universal_claims / extract_universal_claims_gated·独立函数·同隔离范式）：
X 都是 Y·UNIVERSAL_CUE 紧邻 pair (child_idx, parent_idx)·填 Segment.universal_claims·
构造性**验证** EXTERNAL（IS_A 来自 ConceptNet 外部源·build_isa_ancestor_map_external·非 cue 自产·
反 single-source theater·三值逻辑 None 守属性全称 G5b #479 墙·详 doc/重来_刀C量化cue设计_2026-07-08.md）。

**G1+#774 属性命题声明**（extract_property_claims / extract_property_claims_gated·独立函数·入图非闭包传·
异刀A/B/C）：X 的 Y 是 Z / X 具有 Z·固定窗口 4-tuple (subject_idx, attr_type_idx, value_idx, 0)·
填 Segment.property_claims·observe build_property_edges 建命题节点（ATTR_PROPOSITION）+PROPERTY 出边（value）·
G3b 全局扫命题节点判同(subject,attr_type)多值结构矛盾（层a·reification 表达力非验证力·truth=#479 墙·
详 doc/重来_G1reification_774PROPERTY_设计_2026-07-09.md）。独立 helpers（is_property_*·非 cue_type_of·
防 是/的 污染 extract_cues 邻居判）·gate PROPOSITION_MODE（异刀B/刀C 的 CUE_EXTRACTOR_MODE）。

**刀 D 比较声明**（extract_comparison_claims / extract_comparison_claims_gated·独立函数·闭包传·同刀A/B）：
NUM 比较OP NUM（大于/小于/不小于/不大于）·紧邻 3-token 窗口 3-tuple (left_num, cmp_opcode, right_num)·
填 Segment.comparison_claims·闭包传 comparison_proof_fn（cross_compare 交叉积验序·不入图）·构造性检查
SELF_PRODUCED（数 single-source·同刀 A/B）。独立 helpers（comparison_op_of·非 cue_type_of·不入 _CUE_WORDS·
防 大于/小于 污染 extract_cues·比刀 B 等于入 _CUE_WORDS 更 safe）·gate CUE_EXTRACTOR_MODE（同刀B/刀C）。
doc "命题值比序"(B) defer（须 ref→surface 基建·concept_index 无反查）·首刀做 (A) 字面数值比序。
"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, CAUSES_CUE_FORWARD, CAUSES_CUE_BACKWARD, IS_A_CUE, PRECEDES_CUE_FORWARD,
    ARITH_EQUALS_CUE, arith_op_of, UNIVERSAL_CUE, EXISTENTIAL_CUE,
    is_property_attr_marker, is_property_value_copula, is_property_possess_cue,
    is_negation_cue,
    is_modal_cue, modal_op_of,
    comparison_op_of, is_comparison_op_token,
    is_similar_cue,
    degree_intensity_of,
)


def extract_cues(tokens: list[str], *, lang: int,
                 backend=None, edge_store=None,
                 space_id: int | None = None, concept_index=None
                 ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """段 tokens → (cue_based_causal_pairs, is_a_pairs, precedes_pairs)（token index 对）。

    每个 cue token 最多产一对·用紧邻左右 token·边界 cue 跳。
    返三 list[(因index, 果index)] / [(child_index, parent_index)] / [(A_index, B_index)·A 先于 B]。

    **刀5 件8 透传**（close 刀4 生产 gap）：4 可选参透传给 cue_type_of 第二源
    （D:11 readback·gate EMERGENT_RELATION_CUE_READBACK_MODE·冷启动退化纯 frozenset）。
    默认全 None → cue_type_of:99 退化纯 frozenset → 现状零行为变（bit-identical）。
    生产 caller（formal_train._split_item_to_segments）透传 ctx.backend/edge_store/space_id/concept_index。
    """
    cue_pairs: list[tuple[int, int]] = []
    is_a_pairs: list[tuple[int, int]] = []
    precedes_pairs: list[tuple[int, int]] = []   # 刀 A 时序 cue（PRECEDES_CUE_FORWARD·A 先于 B·闭包传验序器·不入图）
    if not tokens:
        return cue_pairs, is_a_pairs, precedes_pairs
    n = len(tokens)
    for i, tok in enumerate(tokens):
        ct = cue_type_of(tok, lang, backend=backend, edge_store=edge_store,
                         space_id=space_id, concept_index=concept_index)
        if ct is None:
            continue
        left = i - 1
        right = i + 1
        # 紧邻左右须存在·且非自身是 cue（跳过相邻连用 cue·守锚定单义）
        if left < 0 or right >= n:
            continue   # 边界 cue 无左/右·跳（守反统计·不凑配）
        if cue_type_of(tokens[left], lang, backend=backend, edge_store=edge_store,
                       space_id=space_id, concept_index=concept_index) is not None:
            continue   # 左邻也是 cue·跳（连用指向词·锚定歧义·首版保守跳）
        if cue_type_of(tokens[right], lang, backend=backend, edge_store=edge_store,
                       space_id=space_id, concept_index=concept_index) is not None:
            continue
        if ct == CAUSES_CUE_FORWARD:
            cue_pairs.append((left, right))            # 因(左) → 果(右)
        elif ct == CAUSES_CUE_BACKWARD:
            cue_pairs.append((right, left))            # 因(右) → 果(左)
        elif ct == IS_A_CUE:
            is_a_pairs.append((left, right))           # child(左) → parent(右)
        elif ct == PRECEDES_CUE_FORWARD:
            precedes_pairs.append((left, right))       # A(左) 先于 B(右)·时序 cue（刀 A·闭包传验序器·不入图）
    return cue_pairs, is_a_pairs, precedes_pairs


def extract_cues_gated(tokens: list[str], *, lang: int,
                       backend=None, edge_store=None,
                       space_id: int | None = None, concept_index=None
                       ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """gate 守门版（CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归）。

    formal_train 调此版（非 extract_cues）·gate OFF 时等同现状（三字段空）。
    **刀5 件8**：透传 4 可选参给 extract_cues（→ cue_type_of 第二源 D:11 readback）。
    两层 gate 独立：CUE_EXTRACTOR_MODE 守整体（本函数）·
    EMERGENT_RELATION_CUE_READBACK_MODE 守第二源（cue_words.cue_type_of:97）。
    **刀 A**：返 3-tuple（加 precedes_pairs·时序 cue·闭包传验序器·不入图）。
    """
    if not getattr(gates, "CUE_EXTRACTOR_MODE", False):
        return [], [], []
    return extract_cues(tokens, lang=lang, backend=backend, edge_store=edge_store,
                        space_id=space_id, concept_index=concept_index)


# ============ 刀 B：数值等式声明提取（NUM OP NUM 等于 NUM·独立函数·不改 extract_cues 签名） ============

def _parse_int_token(token: str) -> int | None:
    """token → int（ASCII 数字·可选前导负号·刀 B 数值 cue 用）。

    仅 ASCII 数字（isascii 排全角·isdigit 排字母）·空/仅"−"/非数字返 None（守反统计·不凑配）。
    中文数字（一二三）defer（首版窄域·须专用解析·非本刀 scope）。纯整数铁律：返 int·零浮点。
    非 str 输入（None 等·对抗审 P2-4）→ None（与 cue_type_of 对 None 静默跳一致·统一健壮性·caller 违约 fail-soft）。
    """
    if not isinstance(token, str):   # None / 非 str → None（防 AttributeError·同 cue_type_of 健壮性）
        return None
    s = token.strip()
    if not s:
        return None
    neg = s.startswith("-")
    digits = s[1:] if neg else s
    # 仅 ASCII 数字（isascii 排全角 １２３·isdigit 排字母）·空 digits（仅 "-"）→ None
    if digits and digits.isascii() and digits.isdigit():
        return -int(digits) if neg else int(digits)
    return None


def extract_numeric_claims(tokens: list[str], *, lang: int,
                           backend=None, edge_store=None,
                           space_id: int | None = None, concept_index=None
                           ) -> list[tuple[int, int, int, int]]:
    """刀 B：段 tokens → 数值等式声明列表 [(left_num, op_opcode, right_num, result_num), ...]。

    模式：``NUM OP NUM 等于 NUM``（cue=ARITH_EQUALS_CUE 在 index i·左式 tokens[i-3..i-1] = NUM OP NUM·
    右式 tokens[i+1] = NUM）。cue token 处固定窗口扫描（确定性·非穷举·守反统计契约·一个等于锚最多一声明）。

    不匹配模式（左/右非数字 / 算子词未命中 / 边界）→ 跳（不凑配·守反统计·同 extract_cues 反统计契约）。
    仅整数保持算术 +,-,×（OPCODE_ADD/SUB/MUL·arith_op_of 识别加/减/乘词）·除法 defer（有理结果须 Rational·首版窄域）。

    **构造性检查 ≠ 构造性验证**：左式/右式数均来自文本（single-source·系统从 cue 锚/token 读非 R6 独立源）
    → 构造性检查·非验证·Layer0 标 SELF_PRODUCED（全自产不准停）。"3+5=8" 算术真独立于来源·但数据
    single-source 故纪律标检查非验证（同刀 A 时序·镜像其 single-source 定位）。

    返纯整数 4-tuple list（left_num, op, right_num, result_num·全 int·assert_int 守）·填 Segment.numeric_claims·
    formal_train._run_numeric_verify_round 闭包传 numeric_proof_fn 检查（不入图·镜像刀 A 时序边不入图）。

    **独立于 extract_cues**（刀 B 决断）：不改 extract_cues 3-tuple 签名（避免 14 处 unpack 改·降风险）·
    数值提取是不同模式（固定窗口扫描·非紧邻 pair）·独立函数隔离改动。生产 caller 透传同 4 参（D:11 readback
    一致·STEP5 PR1 加 REL_EQUAL→ARITH_EQUALS_CUE 映射·'等于'类词经 D:11 readback 可作等式锚·gate OFF 退化 frozenset）。
    STEP5 PR2：arith_op_of 亦透传 4 参（operator D:11 readback 第二源·'相加'类词经 D:11→OP_ADD→OPCODE_ADD）。
    """
    claims: list[tuple[int, int, int, int]] = []
    if not tokens:
        return claims
    n = len(tokens)
    for i, tok in enumerate(tokens):
        ct = cue_type_of(tok, lang, backend=backend, edge_store=edge_store,
                         space_id=space_id, concept_index=concept_index)
        if ct != ARITH_EQUALS_CUE:
            continue
        # 模式 NUM OP NUM 等于 NUM：cue 在 i·须 i-3..i-1（左式 3 token）+ i+1（右式 1 token）全在界
        if i - 3 < 0 or i + 1 >= n:
            continue   # 边界·左式 3 token / 右式 1 token 不足·跳（守反统计·不凑配）
        left_tok = tokens[i - 3]
        op_tok = tokens[i - 2]
        right_tok = tokens[i - 1]
        result_tok = tokens[i + 1]
        left_num = _parse_int_token(left_tok)
        right_num = _parse_int_token(right_tok)
        result_num = _parse_int_token(result_tok)
        op = arith_op_of(op_tok, lang, backend=backend, edge_store=edge_store,
                         space_id=space_id, concept_index=concept_index)
        if left_num is None or right_num is None or result_num is None or op is None:
            continue   # 不匹配模式（非数字 / 非算子词）·跳（守反统计契约·不凑配）
        claims.append((left_num, op, right_num, result_num))
    return claims


def extract_numeric_claims_gated(tokens: list[str], *, lang: int,
                                 backend=None, edge_store=None,
                                 space_id: int | None = None, concept_index=None
                                 ) -> list[tuple[int, int, int, int]]:
    """gate 守门版（CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归·同 extract_cues_gated 范式）。

    formal_train._split_item_to_segments 调此版（gate OFF 时返空·现状零行为变）。
    **独立于 extract_cues_gated**（刀 B·不改 3-tuple 签名·隔离改动）·同 CUE_EXTRACTOR_MODE gate 守门
    （数值提取是 cue 提取一部分·同 gate 合理·非新 gate）。
    """
    if not getattr(gates, "CUE_EXTRACTOR_MODE", False):
        return []
    return extract_numeric_claims(tokens, lang=lang, backend=backend, edge_store=edge_store,
                                  space_id=space_id, concept_index=concept_index)


# ============ 刀 C：全称量化声明提取（X 都是 Y·独立函数·不改 extract_cues 签名） ============

def extract_universal_claims(tokens: list[str], *, lang: int,
                             backend=None, edge_store=None,
                             space_id: int | None = None, concept_index=None
                             ) -> list[tuple[int, int]]:
    """刀 C：段 tokens → 全称量化声明列表 [(child_idx, parent_idx), ...]（token index 对）。

    模式：``X [都是] Y``（cue=UNIVERSAL_CUE 在 index i·child=tokens[i-1]·parent=tokens[i+1]·
    紧邻 pair·镜像 IS_A_CUE 分支但全称 force）。cue token 处紧邻左右 token 作 child/parent·
    一个 cue 最多一对（锚定非穷举·守反统计契约·同 extract_cues）。

    不匹配模式（边界 cue 无左/右·左/右邻也是 cue·连用全称系词）→ 跳（守反统计·不凑配·同 extract_cues
    反统计契约）。返 token index 对（child_idx, parent_idx）·resolve 在验序器（token→ConceptRef·
    concept_index.lookup·非构造器·隔离改动降风险·同刀 A precedes_pairs 范式）。

    **构造性验证 ≠ 构造性检查**（刀 C 升验证·刀 A/B 是检查）：child/parent 概念须在**外部 ConceptNet 祖先图**
    （build_isa_ancestor_map_external·source=SOURCE_CONCEPTNET·非 cue 自产）才 verified/falsified·
    否则 can't-verify（None·诚实弃权·守属性全称 G5b #479 墙·详 doc/重来_刀C量化cue设计_2026-07-08.md §六b）。
    本函数只提 token index 对·外部源验在 universal_proof_fn_factory + _run_universal_verify_round。

    **首版诚实 scope**：紧邻单 token child/parent（多 token "会飞 的 鸟" defer·tokenize 质量依赖·同刀 A/B）·
    不要求限定词"所有/每个"前置（"都是"自含全称 force·非分类声明由三值 None 过滤·非机制判别）。
    """
    claims: list[tuple[int, int]] = []
    if not tokens:
        return claims
    n = len(tokens)
    for i, tok in enumerate(tokens):
        ct = cue_type_of(tok, lang, backend=backend, edge_store=edge_store,
                         space_id=space_id, concept_index=concept_index)
        if ct != UNIVERSAL_CUE:
            continue
        left = i - 1
        right = i + 1
        if left < 0 or right >= n:
            continue   # 边界 cue 无左/右·跳（守反统计·不凑配·同 extract_cues:64）
        if cue_type_of(tokens[left], lang, backend=backend, edge_store=edge_store,
                       space_id=space_id, concept_index=concept_index) is not None:
            continue   # 左邻也是 cue·跳（连用全称系词·锚定歧义·首版保守跳·同 extract_cues:66）
        if cue_type_of(tokens[right], lang, backend=backend, edge_store=edge_store,
                       space_id=space_id, concept_index=concept_index) is not None:
            continue   # 右邻也是 cue·跳
        claims.append((left, right))   # (child_idx, parent_idx)·X 都是 Y·child⊆parent
    return claims


def extract_universal_claims_gated(tokens: list[str], *, lang: int,
                                   backend=None, edge_store=None,
                                   space_id: int | None = None, concept_index=None
                                   ) -> list[tuple[int, int]]:
    """gate 守门版（CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归·同 extract_numeric_claims_gated 范式）。

    formal_train._split_item_to_segments 调此版（gate OFF 时返空·现状零行为变）。
    **独立于 extract_cues_gated**（刀 C·不改 3-tuple 签名·隔离改动）·同 CUE_EXTRACTOR_MODE gate 守门
    （全称量化是 cue 提取一部分·同 gate 合理·非新 gate）。
    """
    if not getattr(gates, "CUE_EXTRACTOR_MODE", False):
        return []
    return extract_universal_claims(tokens, lang=lang, backend=backend, edge_store=edge_store,
                                    space_id=space_id, concept_index=concept_index)


# ============ A1·STEP6：存在量化声明提取（有的 X 是 Y·镜像刀 C·双向祖先·独立函数） ============

def extract_existential_claims(tokens: list[str], *, lang: int,
                               backend=None, edge_store=None,
                               space_id: int | None = None, concept_index=None
                               ) -> list[tuple[int, int]]:
    """A1·STEP6：段 tokens → 存在量化声明列表 [(child_idx, parent_idx), ...]（token index 对）。

    模式：``有的 X 是 Y``（cue=EXISTENTIAL_CUE 在 index i·child=tokens[i+1]·是=value copula 须在 i+2·
    parent=tokens[i+3]·4-token **起始 cue** 窗口·镜像 extract_universal_claims 但 ∃ 起始 cue + 是 锚定·
    非 ∀ 中间 cue 紧邻 pair·∃ 自然句法量化词在前）。cue token 处窗口扫描（确定性·非穷举·守反统计契约·
    一个 ∃ cue 最多一声明·同 extract_universal_claims 窗口范式）。

    不匹配模式（边界 child/是/parent 不足·i+2 非 是·child/parent 自身是 cue token）→ 跳
    （守反统计·不凑配·同 extract_universal_claims 反统计契约）。返 token index 对（child_idx, parent_idx）·
    resolve 在验序器（token→ConceptRef·concept_index.lookup·非构造器·隔离改动降风险·同 ∀ 范式）。

    **构造性验证 ≠ 构造性检查**（同 ∀·A1 镜像）：child/parent 概念须在**外部 ConceptNet 祖先图**
    （build_isa_ancestor_map_external·source=SOURCE_CONCEPTNET·非 cue 自产）才 verified/falsified·
    否则 can't-verify（None·诚实弃权·守属性 ∃ #479 墙）。本函数只提 token index 对·外部源验在
    existential_proof_fn_factory + _run_existential_verify_round。

    **★双向祖先（关键·非单纯 reversed）**：∃ "有的 X 是 Y"=∃x:X(x)∧Y(x)=X∩Y≠∅·类层真 iff X⊆Y OR Y⊆X
    （其一为子集→小类实例即 X∩Y 样本）。"有的鸟是企鹅"（企鹅⊆鸟→reversed 命中）/ "有的鸟是动物"
    （鸟⊆动物→forward 命中·同 ∀）。**单向 reversed 会误证伪"有的鸟是动物"**（鸟∉ancestors(动物)）→
    须双向 OR：`parent ∈ ancestors(child) OR child ∈ ancestors(parent)`·双向皆不命中+两分类→falsified。

    **首版诚实 scope**：紧邻单 token child/parent（多 token defer·同 ∀）·要求 是 锚定（ZH·EN defer·
    同 property cue ZH-first）·"有的"自含 ∃ force（非分类声明由三值 None 过滤·非机制判别）。
    """
    claims: list[tuple[int, int]] = []
    if not tokens:
        return claims
    n = len(tokens)
    for i, tok in enumerate(tokens):
        ct = cue_type_of(tok, lang, backend=backend, edge_store=edge_store,
                         space_id=space_id, concept_index=concept_index)
        if ct != EXISTENTIAL_CUE:
            continue
        # 模式 有的 X 是 Y：cue 在 i·须 i+1（child）+ i+2（是·value copula）+ i+3（parent）全在界
        if i + 3 >= n:
            continue   # 边界·child/是/parent 不足·跳（守反统计·不凑配·同 extract_universal:248）
        child_idx = i + 1
        copula_idx = i + 2
        parent_idx = i + 3
        if not is_property_value_copula(tokens[copula_idx], lang):
            continue   # i+2 非 是·非存在量化窗口（"有的 X Y" 非法·跳·守 是 锚定·同 property 固定窗口范式）
        if (cue_type_of(tokens[child_idx], lang, backend=backend, edge_store=edge_store,
                        space_id=space_id, concept_index=concept_index) is not None
                or cue_type_of(tokens[parent_idx], lang, backend=backend, edge_store=edge_store,
                               space_id=space_id, concept_index=concept_index) is not None):
            continue   # child/parent 自身是 cue·跳（连用 cue·锚定歧义·首版保守跳·同 extract_universal:250）
        claims.append((child_idx, parent_idx))   # (child_idx, parent_idx)·有的 X 是 Y·∃x∈X∧x∈Y
    return claims


def extract_existential_claims_gated(tokens: list[str], *, lang: int,
                                     backend=None, edge_store=None,
                                     space_id: int | None = None, concept_index=None
                                     ) -> list[tuple[int, int]]:
    """gate 守门版（CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归·同 extract_universal_claims_gated 范式）。

    formal_train._split_item_to_segments 调此版（gate OFF 时返空·现状零行为变）。
    同 CUE_EXTRACTOR_MODE gate 守门（存在量化是 cue 提取一部分·同 gate 合理·非新 gate·同刀 B/C/D）。
    """
    if not getattr(gates, "CUE_EXTRACTOR_MODE", False):
        return []
    return extract_existential_claims(tokens, lang=lang, backend=backend, edge_store=edge_store,
                                      space_id=space_id, concept_index=concept_index)


# ============ G1+#774：属性命题声明提取（X 的 Y 是 Z / X 具有 Z·独立函数·入图非闭包传） ============

def _is_property_cue_token(token: str, lang: int, *,
                           backend=None, edge_store=None,
                           space_id: int | None = None, concept_index=None) -> bool:
    """token 是否任一属性/否定/情态 cue（的/是/具有/有/has + 不/没/非/无/not/no + 必然/可能/必须/应该/可以·守反统计·配对两端不取 cue token）。

    B1 扩含否定词（防"不"作 subject/attr/value·守反统计·同 extract_cues:66 邻居判）。
    B2 扩含情态词（防"必然/可能"作 subject/attr/value·守反统计·同 B1 范式）。
    STEP5 PR3：透传 4 参→is_property_possess_cue D:11 readback（gate ON 时非 frozenset 领属词亦判·与主调一致）。
    默认 None→退化纯 frozenset（bit-identical）。"""
    return (is_property_attr_marker(token, lang)
            or is_property_value_copula(token, lang)
            or is_property_possess_cue(token, lang, backend=backend, edge_store=edge_store,
                                       space_id=space_id, concept_index=concept_index)
            or is_negation_cue(token, lang, backend=backend, edge_store=edge_store,
                                       space_id=space_id, concept_index=concept_index)
            or is_modal_cue(token, lang, backend=backend, edge_store=edge_store,
                                       space_id=space_id, concept_index=concept_index))


def extract_property_claims(tokens: list[str], *, lang: int,
                            negation_on: bool = False,
                            modality_on: bool = False,
                            degree_on: bool = False,
                            backend=None, edge_store=None,
                            space_id: int | None = None, concept_index=None
                            ) -> list[tuple[int, int, int, int, int, int, int, int]]:
    """G1+#774：段 tokens → 属性命题声明列表 [(subject_idx, attr_type_idx, value_idx, 0, polarity, modality, intensity_num, intensity_den), ...]（8-int tuple·P0.3 pol/mod + #1134 intensity·default 0/1·B1 否定/B2 情态/#1134 程度填值）。

    两模式（固定窗口扫描·确定性·非穷举·守反统计契约·一个 cue 锚最多一声明·同 extract_numeric_claims 窗口范式）：
      的...是：``X 的 Y 是 Z``（是=value copula 在 index j·的=attr marker 须在 j-2）·
               subject=tokens[j-3]·attr_type=tokens[j-1]·value=tokens[j+1]
      领属：  ``X 具有 Z``（具有/有=possess cue 在 index j）·
               subject=tokens[j-1]·attr_type=-1（缺省·首版 defer·build_property_edges skip）·value=tokens[j+1]

    不匹配模式（边界 subject/attr/value 不足·j-2 非 的·subject/attr/value 自身是 cue token）→ 跳
    （守反统计·不凑配·同 extract_cues 反统计契约）。返纯整数 4-tuple（attr_type_idx<0=无 attr_type·build skip）·
    填 Segment.property_claims·observe build_property_edges 建命题节点+PROPERTY 出边（入图·异刀A/B/C 闭包传）。

    **G1+#774 诚实边界**：reification 给**表达力非验证力**（命题 truth=#479 墙·G3b 只判结构矛盾层a·
    同(subject,attr_type)多值=CONTRADICTED·语义真对立层b/c #479 truth 关切 W2·**非 W1 D 物理接地墙**·provisional/可废止对立 E3 覆写+E4 推理引擎可达·只 definitive truth 撞 #479·defer）。cue 提取是 single-source（文本）·
    命题节点 truth 不验·只承载三元供 G3b 读判。stable≠correct（"猫的颜色是黑" 是否真=语义层·#479 墙）。
    独立 helpers（is_property_*·非 cue_type_of·防 是/的 污染 extract_cues 邻居判·见 cue_words 注）。
    """
    claims: list[tuple[int, int, int, int, int, int, int, int]] = []
    if not tokens:
        return claims
    n = len(tokens)
    for j, tok in enumerate(tokens):
        # 模式 的...是（X 的 Y 是 Z·肯定 pol=0/mod=0 / X 的 Y 不 是 Z·否定 pol=1·B1 / X 的 Y [必然] 是 Z·情态 mod>0·B2）
        if is_property_value_copula(tok, lang):
            # B2 情态检查：modality_on + tokens[j-1] 情态词 → "X 的 Y [必然] 是 Z"·modality 填值·窗口偏移 1（同否定几何）
            # modal 与 negation 同槽 j-1·互斥（情态词≠否定词·不同词集）·先查 modal（情态优先于否定·首版 modal-only 窗口·复合 defer）
            is_modal = (modality_on and j - 1 >= 0
                        and is_modal_cue(tokens[j - 1], lang, backend=backend, edge_store=edge_store,
                                          space_id=space_id, concept_index=concept_index))
            if is_modal:
                # 情态窗口：modal at j-1·attr_type at j-2·的 at j-3·subject at j-4·value at j+1（同否定 offset+1 几何）
                if j - 4 < 0 or j + 1 >= n:
                    continue   # 边界·subject/value 不足·跳（守反统计·不凑配）
                if not is_property_attr_marker(tokens[j - 3], lang):
                    continue   # j-3 非 的·非属性窗口·跳
                subj_idx, attr_idx, val_idx = j - 4, j - 2, j + 1
                polarity = 0
                modality = modal_op_of(tokens[j - 1], lang, backend=backend, edge_store=edge_store,
                                          space_id=space_id, concept_index=concept_index)
            elif (negation_on and j - 1 >= 0
                      and is_negation_cue(tokens[j - 1], lang,
                              backend=backend, edge_store=edge_store,
                              space_id=space_id, concept_index=concept_index)):
                # B1 否定窗口：不 at j-1·attr_type at j-2·的 at j-3·subject at j-4·value at j+1·pol=1
                if j - 4 < 0 or j + 1 >= n:
                    continue   # 边界·subject/value 不足·跳（守反统计·不凑配）
                if not is_property_attr_marker(tokens[j - 3], lang):
                    continue   # j-3 非 的·非属性窗口·跳
                subj_idx, attr_idx, val_idx = j - 4, j - 2, j + 1
                polarity = 1
                modality = 0
            else:
                # 既有肯定窗口：的 at j-2·subject j-3·attr_type j-1·value j+1·pol=0/mod=0
                if j - 3 < 0 or j + 1 >= n:
                    continue   # 边界·subject/value 不足·跳（守反统计·不凑配）
                if not is_property_attr_marker(tokens[j - 2], lang):
                    continue   # j-2 非 的·非属性窗口（是 可能是 IS_A/其他用法）·跳（守 的...是 固定窗口）
                subj_idx, attr_idx, val_idx = j - 3, j - 1, j + 1
                polarity = 0
                modality = 0
            # #1134 degree intensity（degree_on·copula 是 与 value 间 degree 副词·tokens[val_idx] 是 degree cue → value 后移+intensity·三 是-window 共用此解析）
            deg = degree_intensity_of(tokens[val_idx], lang) if degree_on else None
            if deg is not None:
                if val_idx + 1 >= n:
                    continue   # value 被 degree 占位·真 value 越界·跳（守反统计·不凑配）
                val_idx = val_idx + 1   # degree 占原 val_idx·真 value 后移一位（"X 的 Y 是 非常 Z"·非常=j+1·Z=j+2）
                intensity_num, intensity_den = deg
            else:
                intensity_num, intensity_den = 1, 1
            if (_is_property_cue_token(tokens[subj_idx], lang, backend=backend, edge_store=edge_store,
                                        space_id=space_id, concept_index=concept_index)
                    or _is_property_cue_token(tokens[attr_idx], lang, backend=backend, edge_store=edge_store,
                                               space_id=space_id, concept_index=concept_index)
                    or _is_property_cue_token(tokens[val_idx], lang, backend=backend, edge_store=edge_store,
                                               space_id=space_id, concept_index=concept_index)):
                continue   # subject/attr/value 自身是 cue·跳（守反统计·不配 cue token·同 extract_cues:66）
            claims.append((subj_idx, attr_idx, val_idx, 0, polarity, modality, intensity_num, intensity_den))
            continue
        # 模式 领属（X 具有 Z / X 有 Z / X has Z·attr_type 缺省·STEP5 PR3 possess un-defer·REL_PROPERTY 作默认 attr_type）
        if is_property_possess_cue(tok, lang, backend=backend, edge_store=edge_store,
                                   space_id=space_id, concept_index=concept_index):
            if j - 1 < 0 or j + 1 >= n:
                continue   # 边界·subject/value 不足·跳
            subj_idx, val_idx = j - 1, j + 1
            # #1134 degree intensity（possess 窗口同 degree 几何·tokens[val_idx] 是 degree cue → value 后移）
            deg = degree_intensity_of(tokens[val_idx], lang) if degree_on else None
            if deg is not None:
                if val_idx + 1 >= n:
                    continue   # value 被 degree 占位·真 value 越界·跳
                val_idx = val_idx + 1
                p_num, p_den = deg
            else:
                p_num, p_den = 1, 1
            if (_is_property_cue_token(tokens[subj_idx], lang, backend=backend, edge_store=edge_store,
                                        space_id=space_id, concept_index=concept_index)
                    or _is_property_cue_token(tokens[val_idx], lang, backend=backend, edge_store=edge_store,
                                               space_id=space_id, concept_index=concept_index)):
                continue   # subject/value 自身是 cue·跳（守反统计）
            claims.append((subj_idx, -1, val_idx, 0, 0, 0, p_num, p_den))   # attr_type=-1·STEP5 PR3 default_attr_ref=REL_PROPERTY 补身份·P0.3 pol/mod=0·#1134 intensity=p_num/p_den
    return claims


def extract_property_claims_gated(tokens: list[str], *, lang: int,
                                  backend=None, edge_store=None,
                                  space_id: int | None = None, concept_index=None
                                  ) -> list[tuple[int, int, int, int, int, int, int, int]]:
    """gate 守门版（PROPOSITION_MODE OFF → 返空·bit-identical 守回归）。

    formal_train._split_item_to_segments 调此版（gate OFF 时返空·现状零行为变）。
    **gate = PROPOSITION_MODE**（异刀B/刀C 用 CUE_EXTRACTOR_MODE·属性命题是独立 G1+#774 特性·单 gate
    守 extraction→build→intent→G3b 全链·cleaner bit-identical·设计 §四）。STEP5 PR3：加 4 可选参
    （is_property_possess_cue D:11 readback 第二源·'拥有'类词经 D:11→REL_PROPERTY→True·gate
    EMERGENT_RELATION_CUE_READBACK_MODE OFF 退化纯 frozenset·bit-identical）。
    **#1134**：传 degree_on=DEGREE_MODE（程度窗口·boot 先 populate_degree_cues 喂 cache·
    OFF→degree_intensity_of 返 None→intensity 恒 1/1→bit-identical）。
    """
    if not getattr(gates, "PROPOSITION_MODE", False):
        return []
    return extract_property_claims(tokens, lang=lang,
                                   negation_on=getattr(gates, "NEGATION_MODE", False),
                                   modality_on=getattr(gates, "MODALITY_MODE", False),
                                   degree_on=getattr(gates, "DEGREE_MODE", False),
                                   backend=backend, edge_store=edge_store,
                                   space_id=space_id, concept_index=concept_index)


# ============ 刀 D：比较声明提取（NUM 比较OP NUM·独立函数·不改 extract_cues 签名） ============

def extract_comparison_claims(tokens: list[str], *, lang: int,
                              backend=None, edge_store=None,
                              space_id: int | None = None, concept_index=None
                              ) -> list[tuple[int, int, int]]:
    """刀 D：段 tokens → 比较声明列表 [(left_num, cmp_opcode, right_num), ...]（纯整数 3-tuple）。

    模式：``NUM 比较OP NUM``（比较 OP 词在 index i·comparison_op_of(tok[i]) 非 None·左式 tokens[i-1]=NUM·
    右式 tokens[i+1]=NUM·紧邻 3-token 窗口）。比较 OP 词既是声明锚又是序方向（大于→CMP_GT/小于→CMP_LT/
    不小于→CMP_GE/不大于→CMP_LE·cue_words.comparison_op_of 识别·**独立 helpers·非 cue_type·不入 _CUE_WORDS**）。
    cue token 处固定窗口扫描（确定性·非穷举·守反统计契约·一个 OP 锚最多一声明·同 extract_numeric_claims 窗口范式）。

    不匹配模式（左/右非数字 / 边界 / 左/右邻也是比较 OP）→ 跳（守反统计·不凑配·同 extract_cues 反统计契约）。
    仅整数 operand（_parse_int_token·同刀 B·ASCII 数字·中文数字 defer）·分数 operand（num/den）defer。

    **构造性检查 ≠ 构造性验证**（同刀 A 时序 / 刀 B 数值）：左/右式数均 single-source（文本 cue 锚·非 R6 独立源）
    → 构造性检查·非验证·Layer0 标 SELF_PRODUCED（全自产不准停）。"5 大于 3" 比序真独立于来源·但数据
    single-source 故纪律标检查非验证（同刀 A/B 定位）。

    返纯整数 3-tuple list（left_num, cmp_opcode, right_num·全 int·cmp_opcode ∈ CMP_GT/LT/GE/LE）·
    填 Segment.comparison_claims·formal_train._run_comparison_verify_round 闭包传 comparison_proof_fn
    检查（cross_compare 交叉积验序·不入图·镜像刀 A/B）。

    **独立于 extract_cues**（刀 D 决断·同刀 B/C）：不改 extract_cues 3-tuple 签名（避免 unpack 改·降风险）·
    比较提取是不同模式（3-token 窗口·非紧邻 pair）·独立函数隔离改动。比较 OP 词不入 _CUE_WORDS（异刀B 等于入
    _CUE_WORDS）→ cue_type_of(大于/小于) 仍返 None → extract_cues 邻居判零变 → bit-identical（比刀 B 更 safe）。
    无 4 可选参 → STEP5 PR2 加 4 可选参（comparison_op_of D:11 readback 第二源·'超过'类词经 D:11→OP_GT→
    CMP_GT·gate OPERATOR_D11_READBACK_MODE OFF 退化纯 frozenset _COMPARISON_OP_WORDS·bit-identical）。
    """
    claims: list[tuple[int, int, int]] = []
    if not tokens:
        return claims
    n = len(tokens)
    for i, tok in enumerate(tokens):
        cmp = comparison_op_of(tok, lang, backend=backend, edge_store=edge_store,
                               space_id=space_id, concept_index=concept_index)
        if cmp is None:
            continue   # 非比较 OP 词·跳
        # 模式 NUM 比较OP NUM：OP 在 i·须 i-1（左式）+ i+1（右式）全在界
        if i - 1 < 0 or i + 1 >= n:
            continue   # 边界·左/右式不足·跳（守反统计·不凑配·同 extract_numeric_claims:169）
        left_tok = tokens[i - 1]
        right_tok = tokens[i + 1]
        left_num = _parse_int_token(left_tok)
        right_num = _parse_int_token(right_tok)
        if left_num is None or right_num is None:
            continue   # 不匹配模式（非数字）·跳（守反统计契约·不凑配）
        if is_comparison_op_token(left_tok, lang, backend=backend, edge_store=edge_store,
                                  space_id=space_id, concept_index=concept_index) \
           or is_comparison_op_token(right_tok, lang, backend=backend, edge_store=edge_store,
                                     space_id=space_id, concept_index=concept_index):
            continue   # 左/右邻也是比较 OP·跳（连用 OP·锚定歧义·首版保守跳·同 extract_cues:66）
        claims.append((left_num, cmp, right_num))
    return claims


def extract_comparison_claims_gated(tokens: list[str], *, lang: int,
                                    backend=None, edge_store=None,
                                    space_id: int | None = None, concept_index=None
                                    ) -> list[tuple[int, int, int]]:
    """gate 守门版（CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归·同 extract_numeric/universal_claims_gated 范式）。

    formal_train._split_item_to_segments 调此版（gate OFF 时返空·现状零行为变）。
    **独立于 extract_cues_gated**（刀 D·不改 3-tuple 签名·隔离改动）·同 CUE_EXTRACTOR_MODE gate 守门
    （比较提取是 cue 提取一部分·同 gate 合理·非新 gate·同刀 B/刀C）。
    """
    if not getattr(gates, "CUE_EXTRACTOR_MODE", False):
        return []
    return extract_comparison_claims(tokens, lang=lang, backend=backend, edge_store=edge_store,
                                     space_id=space_id, concept_index=concept_index)


# ============ STEP5 PR4：相似声明提取（X 像 Y·EDGE_SIMILAR slot-filler·独立函数·D2 合规非向量） ============

def extract_similar_claims(tokens: list[str], *, lang: int,
                           backend=None, edge_store=None,
                           space_id: int | None = None, concept_index=None
                           ) -> list[tuple[int, int]]:
    """STEP5 PR4：段 tokens → 相似声明列表 [(left_idx, right_idx), ...]（2-int tuple·"X 像 Y" 提取）。

    模式：``X 像 Y``（相似 cue 在 index i·is_similar_cue(tok[i]) True·左式 tokens[i-1]=X·
    右式 tokens[i+1]=Y·紧邻 3-token 窗口·同 extract_comparison_claims 窗口范式）。
    is_similar_cue **D:11-readback-only**（无 frozenset 第一源·gate EMERGENT_RELATION_CUE_READBACK_MODE）·
    冷启动（D:11 REL_SIMILAR 无种子）→ False → 无 similar_claims。

    返纯整数 2-tuple list（left_idx, right_idx）·填 Segment.similar_claims·observe build_similar_edges
    建 EDGE_SIMILAR 边（X→Y·TIER_SHADOW·strength=1·不入 effective_weight·不接 reward·D2 合规非向量）。
    dispatch_slot（gate SIMILAR_SLOT_MODE）读 EDGE_SIMILAR 双向扩展 slot 候选（slot-filler）。

    **D2 合规**：EDGE_SIMILAR 二元离散边（非向量·非相似度 SCORE）·确定性文本提取（非学习型）·
    结构关系 slot-filler 扩展（非语义承载）·三维度全不满→非向量→合法（同 EDGE_IS_A 范式）。
    """
    claims: list[tuple[int, int]] = []
    if not tokens:
        return claims
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if not is_similar_cue(tok, lang, backend=backend, edge_store=edge_store,
                              space_id=space_id, concept_index=concept_index):
            continue   # 非相似 cue·跳
        if i - 1 < 0 or i + 1 >= n:
            continue   # 边界·左/右式不足·跳（守反统计·不凑配）
        left_idx = i - 1
        right_idx = i + 1
        # 左/右邻也是 cue token → 跳（连用 cue·锚定歧义·首版保守·同 extract_cues:66 邻居判）
        if (is_similar_cue(tokens[left_idx], lang, backend=backend, edge_store=edge_store,
                            space_id=space_id, concept_index=concept_index)
                or is_similar_cue(tokens[right_idx], lang, backend=backend, edge_store=edge_store,
                                   space_id=space_id, concept_index=concept_index)):
            continue
        claims.append((left_idx, right_idx))
    return claims


def extract_similar_claims_gated(tokens: list[str], *, lang: int,
                                 backend=None, edge_store=None,
                                 space_id: int | None = None, concept_index=None
                                 ) -> list[tuple[int, int]]:
    """gate 守门版（CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归·同 extract_numeric/comparison_claims_gated 范式）。

    formal_train._split_item_to_segments 调此版（gate OFF 时返空·现状零行为变）。
    同 CUE_EXTRACTOR_MODE gate 守门（相似提取是 cue 提取一部分·同 gate 合理·非新 gate·同刀 B/C/D）。
    """
    if not getattr(gates, "CUE_EXTRACTOR_MODE", False):
        return []
    return extract_similar_claims(tokens, lang=lang, backend=backend, edge_store=edge_store,
                                  space_id=space_id, concept_index=concept_index)
