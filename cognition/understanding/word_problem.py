"""cognition.understanding.word_problem — 语言应用题 → 算数估值（language→arith 端到端·piece 1.2）。

word_problem_value(tokens, ...) -> Rational | None
  tokens（["三","加","二"]）→ 数字词接地（**走图**：PURE_ALIAS→整数概念→CORR_NUMERIC）+ 算子 cue
  （arith_op_of·加→+）→ arith 树（build_composes_from_arith）→ **vm_proof execute** → 估值。
  例：["三","加","二"] → 5 · ["十","减","三"] → 7 · ["二","乘","三"] → 6。

**language→arith 桥首刀（语言域主攻·语言嵌入算数·doc/重来_语言通用接地 §六 piece 1）**：
- 结构→结构（语言表达式有可恢复结构 [num op num] + 形式域可执行验）·**不撞 D墙**（架构审视 §5·非"自然语言需求→结构"）。
- 复用既有解决（closure-falsified 从语料学机制）：数字接地（number_grounding·bootstrap 种图边）+ 算子 cue（arith_op_of·元定义+D:11）+ arith builder（arith_observe）+ vm_proof（execute·ground truth）。
- 反 theater：vm_proof 真执行验值（非 parser 输出当答案）·数字值经图遍历取得（关联在图中）。

**不写死 + 关联在图中**：数字词→整数经图边（resolve_number_word·PURE_ALIAS→CORR_NUMERIC·非旁侧表）·
算子词→opcode 经 cue（元定义种子+D:11 图）·代码只解析结构（[num op num]）非硬编关联。

首版 scope：简单二元 NUM OP NUM（左操作数 算子词 右操作数）·无优先级/多步/括号（piece 1.2+）。
诚实边界：piece 1 = 数据接地（number_facts 种子·非学习）·纯语义 word-problem（无可恢复结构·意图推理）= D墙搁置。

铁律：纯整数（数字/opcode 全 int·Rational 返）/ 确定性（图遍历 + arith 执行 bit-identical）/ 单向依赖
（L4 understanding → L5 graph_view/storage + L7 vm·向下）/ 不写死（关联经图/cue·代码只结构解析）/ 反 theater（vm_proof 验）。
"""
from __future__ import annotations

from itertools import count

from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD, OPCODE_SUB, OPCODE_MUL
from pure_integer_ai.crosscut.integer.rational import Rational
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_store import SOURCE_MATH, EdgeStore
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.cognition.understanding.cue_words import arith_op_of
from pure_integer_ai.cognition.understanding.number_grounding import resolve_number_word
from pure_integer_ai.vm.graph_compile import compile_graph
from pure_integer_ai.vm.vm_core import execute

# OPCODE → DSL 算子符号（arith DSL builder 识别·+ − ×·除 defer 有理）
_OPCODE_TO_SYM: dict[int, str] = {
    OPCODE_ADD: "+",
    OPCODE_SUB: "-",
    OPCODE_MUL: "*",
}

# 每次估值用唯一 arith 根（deterministic counter·非 time/random·itertools.count 确定性·进程内单调）：
# 复用同根会累积多 COMPOSES 树（APPEND_ONLY）→ read_composes_tree 混读错值（首次对·后续错）。
# **诚实边界（2 对抗审 LOW·latent·当前不可达）**：进程级计数器·dump-then-resume 后重置为 1·而 __wp_exec_1 经
# content_hash dedup 命中既有节点→会再挂第 2 个 COMPOSES 子树（同混读）。当前无生产 caller（仅测试·fresh backend/进程）
# 故不可达；若未来接 persisted-run caller·须改用 backend-backed 单调计数（如 node_store.next_id）或 root-occupied guard。
_WP_EXEC_SEQ = count(1)


def word_problem_value(tokens: list[str], *, concept_index: ConceptIndex,
                       edge_store: EdgeStore, backend, space_id: int,
                       lang: int) -> Rational | None:
    """语言应用题 tokens → arith 估值（vm_proof）·简单二元 [num, op, num]·返 Rational | None。

    tokens : 已切词的 token 列表（caller 须切数字词/算子词为独立 token·同 cue 纪律）·首版须恰好 3 个 [num, op, num]。
    lang   : 语言码（LANG_ZH/EN·arith_op_of 第一源 frozenset + 第二源 D:11 按 lang 分）。
    返 Rational（vm_proof 真执行估值）| None（非二元 / 数字词未接地 / 算子未识别 / DSL 不支持→诚实不伪造）。

    flow：resolve_number_word(左) + arith_op_of(中) + resolve_number_word(右) → 三段非 None
      → DSL `lambda: {n1} {op_sym} {n2}`（arity-0 常量式）→ build_composes_from_arith → compile_graph → execute。
    **反 theater**：vm_proof 真执行（非 parser 算）·数字值经图遍历（resolve_number_word·关联在图中）。

    诚实边界：首版简单二元·无优先级/多步/括号（piece 1.2+）·除法 defer（有理·DSL // 不支持）。
    """
    if len(tokens) != 3:
        return None   # 首版只支持简单二元 NUM OP NUM（多步/优先级 = piece 1.2+）
    n1 = resolve_number_word(concept_index, edge_store, backend, tokens[0], space_id=space_id)
    op = arith_op_of(tokens[1], lang, backend=backend, edge_store=edge_store,
                     space_id=space_id, concept_index=concept_index)
    n2 = resolve_number_word(concept_index, edge_store, backend, tokens[2], space_id=space_id)
    if n1 is None or op is None or n2 is None:
        return None   # 数字词未接地 / 算子未识别 → 诚实不伪造（守反统计契约·非凑配）
    op_sym = _OPCODE_TO_SYM.get(op)
    if op_sym is None:
        return None   # 仅 + − × ·除法 defer
    # 建 arith 树（arity-0 常量式 lambda: n1 op n2）+ vm_proof execute（反 theater）
    register_composes_attr(backend)   # 幂等（caller 已 register 亦无妨）
    # 唯一根/调用（deterministic counter·非 time/random）：复用同根会累积多 COMPOSES 树致 read 混读错值
    root_ref = concept_index.ensure(f"__wp_exec_{next(_WP_EXEC_SEQ)}", space_id=space_id,
                                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(f"lambda: {n1} {op_sym} {n2}",
                              concept_index=concept_index, edge_store=edge_store,
                              backend=backend, space_id=space_id,
                              source=SOURCE_MATH, root_ref=root_ref)
    g = ConceptGraph(backend)
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        g.read_composes_tree(root_ref)
    instrs = compile_graph(root_ref, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    return execute(instrs, {})   # arity-0 → env 空·vm_proof 真执行
