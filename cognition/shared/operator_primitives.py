"""cognition.shared.operator_primitives — 算子/比较原语 first-class NODE_CONCEPT（L0 元定义层·STEP5 PR2·D6 对齐）。

算子（+/-/×）+ 比较（>/</≥/≤）作 first-class NODE_CONCEPT 节点·D:11 EDGE_RELATION_SIGNAL 词→算子概念边
的 typed target。镜像 relation_primitives.py 范式（REL_* 关系原语）·OP_* 是算子/比较原语的元定义命名空间
（逻辑/算子原语·§九铁律承认 enum 例外·同 REL_* / OPCODE_* / ORIGIN_*·reward 不调·断奶前后不变）。

**符号空间 vs 抽象空间**（D6·AGENT.md:54-77·勿混）：算子/比较 = 符号域**先天·冻结·元定义·不学**
（"节点**是**什么"·参照 UD closed-class + 形式逻辑原语 + Spelke 核心知识）。文字 alias（加/大于/等于）
是开放类 surface·走 D:11 learnable 二源（frozenset 冷启动种子 + D:11 readback 教师晋升）·非硬编码穷举。

**与 OPCODE_*/CMP_* 的关系**：OPCODE_*（numeric/symbol_domain·KIND_OPCODE·axis_symbol_id 大整数）+
CMP_*（crosscut/integer/compare·1-4）是 VM/crosscut 的 opcode 整数（dispatch 用·非 ConceptRef）。
OP_* 是概念图 first-class NODE_CONCEPT（D:11 边端点须 ConceptRef）·_OP_TO_OPCODE 桥 OP_*→opcode
（readback 读 D:11→OP_* concept→ATTR_OPERATOR_PRIMITIVE int_a=OP_*→_OP_TO_OPCODE→opcode）。
OPCODE_*≈2^60+n·CMP_*=1-4·值域不重叠·_OP_TO_OPCODE 双射安全。

**ATTR_OPERATOR_PRIMITIVE=18**（composes_attr·非结构 kind·_STRUCTURAL_KINDS 不含·read_composes_tree
忽略·inline 不传播·镜像 ATTR_RELATION_PRIMITIVE=10）。**勿复用 ATTR_OPERATOR=1**（结构 kind·VM COMPOSES
树专用·复用污染 5-dict 重建）。

**D:11 共享边类型隔离**：EDGE_RELATION_SIGNAL 边 word→any concept。REL_* target 挂 ATTR_RELATION_PRIMITIVE·
OP_* target 挂 ATTR_OPERATOR_PRIMITIVE。lookup_word_concept 过滤 ATTR_RELATION_PRIMITIVE（OP_* target
kind=0→skip）·lookup_word_operator 过滤 ATTR_OPERATOR_PRIMITIVE（REL_* target kind=0→skip）·无交叉污染。

位于 cognition/shared（L0）·import storage 向下合规·非 re-export（同 relation_primitives.py 范式）。

铁律：纯整数（OP_*/opcode/ConceptRef 全 int·assert_int 守）/ 确定性（stable surface hash bit-identical）/
  不写死（OP_* enum=meta定义例外·非语义规则·closed-class 种子·开放变体走 D:11 教师晋升）/ 单向依赖
  （L0 依赖 storage 向下）/ D:11 不接 reward（effective_weight.py:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_OPERATOR_PRIMITIVE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.composes_attr import read_composes_attrs
from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD, OPCODE_SUB, OPCODE_MUL
from pure_integer_ai.crosscut.integer.compare import CMP_GT, CMP_LT, CMP_GE, CMP_LE
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN

# ---- OP_* 枚举（算子/比较原语类型·meta定义·STEP5 PR2·D6 对齐·非抽象空间 abstract_mark） ----
# 元定义层固化·非语义规则（同 REL_* / OPCODE_* / CMP_*·reward 不调·断奶前后不变）
OP_ADD = 1   # 算术加 +（OPCODE_ADD）
OP_SUB = 2   # 算术减 -（OPCODE_SUB）
OP_MUL = 3   # 算术乘 ×（OPCODE_MUL）
OP_GT = 4    # 比较 > （CMP_GT）
OP_LT = 5    # 比较 < （CMP_LT）
OP_GE = 6    # 比较 ≥ （CMP_GE）
OP_LE = 7    # 比较 ≤ （CMP_LE）

# OP_* → opcode 映射（readback 用·D:11→OP_* concept→_OP_TO_OPCODE→opcode）
# OPCODE_*=OPCODE_BASE|n≈2^60+n 大整数·CMP_*=1-4·值域不重叠·双射安全
_OP_TO_OPCODE: dict[int, int] = {
    OP_ADD: OPCODE_ADD, OP_SUB: OPCODE_SUB, OP_MUL: OPCODE_MUL,
    OP_GT: CMP_GT, OP_LT: CMP_LT, OP_GE: CMP_GE, OP_LE: CMP_LE,
}

# 算术 OP_* 集合（arith_op_of readback 过滤用·只认算术 OP·非比较 OP）
_ARITH_OPS = frozenset({OP_ADD, OP_SUB, OP_MUL})
# 比较 OP_* 集合（comparison_op_of readback 过滤用·只认比较 OP·非算术 OP）
_COMPARISON_OPS = frozenset({OP_GT, OP_LT, OP_GE, OP_LE})

# 稳定 surface（content_hash dedup·跨 run identity·bit-identical）
_OP_SURFACE: dict[int, str] = {
    OP_ADD: "__OP_ADD__", OP_SUB: "__OP_SUB__", OP_MUL: "__OP_MUL__",
    OP_GT: "__OP_GT__", OP_LT: "__OP_LT__", OP_GE: "__OP_GE__", OP_LE: "__OP_LE__",
}

# ---- 元定义层种子词（lang → {word: op_kind}·镜像 cue_words._ARITH_OP_WORDS + _COMPARISON_OP_WORDS
# closed-class 核心·D6·开放变体（相加/超过/增加 等穷举不尽）走 D:11 教师晋升非硬编码） ----
# 与 cue_words._ARITH_OP_WORDS/_COMPARISON_OP_WORDS 重叠是自然——两者都识别算子/比较词·
# _ARITH_OP_WORDS=cue 检测 frozenset（第一源·返 OPCODE_*）·_OP_LEXICAL_CUE=D:11 boot 种子（第二源·
# 建 word→OP_* concept D:11 边·readback 返 OP_*→opcode）·加二源非替换（D6 关键区分4）·互补非冲突。
_OP_LEXICAL_CUE: dict[int, dict[str, int]] = {
    LANG_ZH: {
        "加": OP_ADD, "加上": OP_ADD,
        "减": OP_SUB, "减去": OP_SUB,
        "乘": OP_MUL, "乘以": OP_MUL,
        "大于": OP_GT, "小于": OP_LT, "不小于": OP_GE, "不大于": OP_LE,
    },
    LANG_EN: {
        "plus": OP_ADD, "add": OP_ADD,
        "minus": OP_SUB, "subtract": OP_SUB,
        "times": OP_MUL, "multiplied_by": OP_MUL,
        "greater_than": OP_GT, "less_than": OP_LT, "at_least": OP_GE, "at_most": OP_LE,
    },
}


def ensure_operator_primitives(concept_index, backend: StorageBackend, *,
                               space_id: int) -> dict[int, tuple[int, int]]:
    """ensure 全部 OP_* first-class NODE_CONCEPT 节点 + ATTR_OPERATOR_PRIMITIVE=18 标记。

    镜像 ensure_relation_primitives。每 OP_*：
    concept_index.ensure(_OP_SURFACE[kind], NODE_CONCEPT, TIER_PRIMARY) → ref
    + record_composes_attr(backend, ref, kind=ATTR_OPERATOR_PRIMITIVE, int_a=kind)。
    返 {op_kind: ConceptRef}（caller bootstrap_operator_signals 用·D:11 target 解析）。

    **幂等**（ConceptIndex.ensure 同 hash 返既有 tier 单调升 + record_composes_attr 同 (ref,kind) skip）→
    每 boot 调安全（resume 跨 run / 重复 boot 不 corrupt）。

    无条件 ensure 全部 OP_*（元定义层常驻·类 REL_*·boot 种 D:11 边前先建 target）。

    backend 显式传（镜像 ensure_relation_primitives·record_composes_attr 需 backend·不触 ConceptIndex 私有 _b）。
    """
    assert_int(space_id, _where="ensure_operator_primitives.space_id")
    out: dict[int, tuple[int, int]] = {}
    for kind, surface in _OP_SURFACE.items():
        ref = concept_index.ensure(surface, space_id=space_id,
                                   tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        record_composes_attr(backend, ref=ref,
                             kind=ATTR_OPERATOR_PRIMITIVE, int_a=kind)
        out[kind] = ref
    return out


def lookup_word_operator(backend: StorageBackend, edge_store: EdgeStore,
                         word_ref: tuple[int, int], *, space_id: int,
                         tier_filter: int | None = None,
                         ) -> list[tuple[tuple[int, int], int]]:
    """读 word_ref 的 D:11 边 → [(op_ref, op_kind), ...]（operator D:11 readback API·镜像 lookup_word_concept）。

    query_from(word_ref, D:11) → 每 target 读 read_composes_attrs 得 ATTR_OPERATOR_PRIMITIVE int_a=op_kind。
    kind==0 skip（非 OP_* 节点·如 REL_* target 挂 ATTR_RELATION_PRIMITIVE·过滤隔离无交叉污染）。

    **tier_filter**（反 theater）：传 TIER_PRIMARY 只返 PRIMARY 边（已验证晋升/教师种子）·
    None（默认）返全 tier（含 SHADOW·bit-identical）。caller（_arith/_comparison_op_from_d11_primary）
    传 TIER_PRIMARY（未验证 SHADOW 不注入 readback·反 theater）。
    """
    if word_ref is None:
        return []
    rows = edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    out: list[tuple[tuple[int, int], int]] = []
    for r in rows:
        if tier_filter is not None and r.get("tier") != tier_filter:
            continue   # tier 过滤（反 theater：未验证 SHADOW 不注入 readback）
        op_ref = (r["space_id_to"], r["local_id_to"])
        attrs = read_composes_attrs(backend, op_ref)
        kind = attrs.get(ATTR_OPERATOR_PRIMITIVE, (0, 0))[0]
        if kind == 0:
            # 防御：D:11 target 无 ATTR_OPERATOR_PRIMITIVE（如 REL_* target·挂 ATTR_RELATION_PRIMITIVE）
            # → kind=0 非合法 OP_*（enum 1-7）·skip 不返（无交叉污染·lookup_word_concept 同范式过滤 REL_*）
            continue
        out.append((op_ref, kind))
    return out


def op_kind_to_opcode(op_kind: int) -> int | None:
    """OP_* → opcode（arith_op_of/comparison_op_of readback 用·_OP_TO_OPCODE 查表）。
    非 OP_* → None（防御）。"""
    return _OP_TO_OPCODE.get(op_kind)


def is_arith_op_kind(op_kind: int) -> bool:
    """op_kind 是否算术 OP（OP_ADD/SUB/MUL）·arith_op_of readback 过滤用（非比较 OP）。"""
    return op_kind in _ARITH_OPS


def is_comparison_op_kind(op_kind: int) -> bool:
    """op_kind 是否比较 OP（OP_GT/LT/GE/LE）·comparison_op_of readback 过滤用（非算术 OP）。"""
    return op_kind in _COMPARISON_OPS
