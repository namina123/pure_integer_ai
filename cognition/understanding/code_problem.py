"""cognition.understanding.code_problem — 语言条件句 → 可执行 code（language→code 端到端·piece 2）。

code_problem_value(tokens, ...) -> Rational | None
  tokens（["如果","三","大于","二","那么","一","否则","零"]）→ 条件结构 cue + 数字词接地（**走图**：
  PURE_ALIAS→整数概念→CORR_NUMERIC）+ 比较 cue（comparison_op_of·大于→CMP_GT）→ Python 源码模板
  （if/Compare/Return）→ build_composes_from_source → **vm_proof execute** → 估值。
  例：["如果","三","大于","二","那么","一","否则","零"] → 1（3>2 真→返 1）·
      ["如果","二","大于","三","那么","一","否则","零"] → 0（2>3 假→返 0）。

**language→code 桥第二刀（语言域主攻·语言嵌入代码·doc/重来_语言通用接地 §七-bis piece 2）**：
- **泛化 piece 1 模式到控制流**（非纯表达式 [num op num]）·结构→结构·**不撞 D墙**（架构审视 §5·
  非"自然语言需求→结构"·本句有可恢复结构 [如果 n cmp n 那么 a 否则 b] + 形式域可执行验）。
- 复用既有解决（closure-falsified 从语料学机制 + piece 1）：数字接地（number_grounding·图边）+
  比较 cue（comparison_op_of·元定义+D:11）+ code 建造者（code_observe.build_composes_from_source）+ vm_proof（execute）。
- 反 theater：vm_proof 真执行 if/Compare/Return（非 parser 算）·数字值经图遍历取得（关联在图中）。

**不写死 + 关联在图中**：数字词→整数经图边（resolve_number_word·PURE_ALIAS→CORR_NUMERIC·非旁侧表）·
比较词→CMP_* 经 cue（元定义种子+D:11）·条件结构词（如果/那么/否则）= **closed-class 句法锚** →
**元定义 dict 种子 cue_words._COND_KEYWORDS**（§九例外·元定义层·独立 _CUE_WORDS·bit-identical·镜像
_COMPARISON_OP_WORDS 范式·cond_keyword_of 单源 D:11 readback defer·无 COND_* D:11 原语）·
代码只结构解析+源码模板（非硬编语义关联）。

**CMP_*→源码符 而非 CMP_*→OPCODE_***：走 Python 源码模板（code_observe FunctionDef 入口）·
ast 自动把 `>`/`<` 产 OPCODE_GT/LT·**免手 CMP→OPCODE 映射**（最净·避两常量族相异坑）。

首版 scope：单层 if-else + 立即数返回（8-token 固定结构 [如果,n1,cmp,n2,那么,a,否则,b]）·
无嵌套/while/函数参数/变量赋值/多步（piece 2.1+）。诚实边界：piece 2 = 数据接地（数字）+ 元定义 cue
（比较/条件）·**非学习**·可学条件接地 defer #479。

铁律：纯整数（n1/n2/a/b 经 number_grounding 返 int·源码模板插 int·vm 执行产 Rational·零浮点）/
确定性（图遍历 + code 执行 bit-identical）/ 单向依赖（L4 understanding → vm L7 + storage L0 向下·
镜像 word_problem·不环）/ 不写死（数字经图·比较/条件 cue 元定义·代码只结构解析+模板）/ 反 theater（vm_proof 验）。
"""
from __future__ import annotations

from itertools import count

from pure_integer_ai.crosscut.integer.compare import CMP_GT, CMP_LT, CMP_EQ
from pure_integer_ai.crosscut.integer.rational import Rational
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_store import SOURCE_CODE, EdgeStore
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.code_observe import build_composes_from_source
from pure_integer_ai.cognition.understanding.cue_words import (
    comparison_op_of, cond_keyword_of, cue_type_of, ARITH_EQUALS_CUE,
    _COND_IF, _COND_THEN, _COND_ELSE)
from pure_integer_ai.cognition.understanding.number_grounding import resolve_number_word
from pure_integer_ai.vm.graph_compile import compile_graph
from pure_integer_ai.vm.vm_core import execute

# 条件结构 cue（如果/那么/否则·closed-class 句法锚·§九元定义种子）→ **移居元定义层 cue_words**
# （_COND_KEYWORDS + cond_keyword_of·2 对抗审 A1 修：元定义 cue 归 cue_words.py·与 _COMPARISON_OP_WORDS/
# _ARITH_OP_WORDS/_NEGATION_CUES/_MODAL_CUES 同层·非 consumer-local）。本模块经 import 复用（见 import 块）。
# cond_keyword_of **单源**（元定义 frozenset 第一源·D:11 readback 第二源 defer·无 COND_* D:11 原语·异 comparison_op_of 两源）。

# CMP_*（crosscut/integer/compare·构造性检查通道）→ Python 源码比较符（code_observe 经 ast.parse 自动产
# OPCODE_GT/LT/EQ·symbol_domain·vm 执行·**免手 CMP→OPCODE 映射**·避两常量族相异坑）。
# GT/LT/EQ：code_observe 支持 Compare Eq/Lt/Gt·**不支持 Ge/Le/NotEq**（GE/LE/NEQ→None defer·code_observe 限制·piece 2.2+）。
_CMP_TO_SYM: dict[int, str] = {CMP_GT: ">", CMP_LT: "<", CMP_EQ: "=="}

# 每次估值用唯一 code 根（deterministic counter·非 time/random·itertools.count 确定性·进程内单调）：
# 复用同根会累积多 COMPOSES 树（APPEND_ONLY）→ read_composes_tree 混读错值（首次对·后续错·镜像 word_problem _WP_EXEC_SEQ）。
# **诚实边界（同 word_problem·latent·当前不可达）**：进程级计数器·dump-resume 后重置→既有根 content_hash dedup
# 命中→再挂第 2 子树混读。当前无生产 caller（仅测试 fresh backend/进程）故不可达；未来接 persisted-run caller 须改 backend-backed 计数器。
_CP_EXEC_SEQ = count(1)


def code_problem_value(tokens: list[str], *, concept_index: ConceptIndex,
                       edge_store: EdgeStore, backend, space_id: int,
                       lang: int) -> Rational | None:
    """语言条件句 tokens → code 估值（vm_proof）·单层 if-else 立即数·返 Rational | None。

    tokens : 已切词的 token 列表·首版须恰好 8 个：[如果, n1, cmp, n2, 那么, a, 否则, b]。
    lang   : 语言码（LANG_ZH/EN·_COND_KEYWORDS + comparison_op_of 按 lang 分）。
    返 Rational（vm_proof 真执行 if/Compare/Return 的 HALT 栈顶值）| None（非 8 元 / 结构 cue 缺 /
      数字词未接地 / 比较 cue 未识别 / DSL 不支持→诚实不伪造）。

    flow：cond_keyword_of(如果/那么/否则) 结构校验 + resolve_number_word(n1/n2/a/b·**图遍历**) +
      comparison_op_of(cmp)→CMP_GT/LT（或 等于 经 cue_type_of==ARITH_EQUALS_CUE→CMP_EQ·单源复用刀B 注册）→
      sym `>`/`<`/`==` → 源码模板
      `def f(): if n1 sym n2: return a else: return b` → build_composes_from_source → compile_graph → execute。
    **反 theater**：vm_proof 真执行 if/Compare/Return（非 parser 算）·数字值经图遍历（resolve_number_word·关联在图中）。

    诚实边界：首版单层 if-else + 立即数返回·无嵌套/while/函数参数/变量赋值/多步（piece 2.1+）·
      GE/LE defer（code_observe 不支持 Ge/Le·NEQ 同 defer·**EQ 已支持** 经 cue_type_of==ARITH_EQUALS_CUE→CMP_EQ→`==`·见上·piece 2.1）。
    """
    if len(tokens) != 8:
        return None   # 首版只支持 8-token 单层 if-else（[如果,n1,cmp,n2,那么,a,否则,b]·多结构 piece 2.1+）
    if cond_keyword_of(tokens[0], lang) != _COND_IF:
        return None   # 结构 cue 缺（非如果起·守反统计契约·非凑配）
    if cond_keyword_of(tokens[4], lang) != _COND_THEN:
        return None
    if cond_keyword_of(tokens[6], lang) != _COND_ELSE:
        return None
    n1 = resolve_number_word(concept_index, edge_store, backend, tokens[1], space_id=space_id)
    cmp = comparison_op_of(tokens[2], lang, backend=backend, edge_store=edge_store,
                           space_id=space_id, concept_index=concept_index)
    if cmp is None and cue_type_of(tokens[2], lang, backend=backend, edge_store=edge_store,
                                   space_id=space_id, concept_index=concept_index) == ARITH_EQUALS_CUE:
        # 等于/equals → 等式 CMP_EQ（**单源**·复用刀B ARITH_EQUALS_CUE 注册·**非** _COMPARISON_OP_WORDS·
        # 避双注册冲突：否则 extract_comparison_claims 对"二加三等于五"误抽假比较声明 3==5·与 extract_numeric_claims 冲突）。
        cmp = CMP_EQ
    n2 = resolve_number_word(concept_index, edge_store, backend, tokens[3], space_id=space_id)
    a = resolve_number_word(concept_index, edge_store, backend, tokens[5], space_id=space_id)
    b = resolve_number_word(concept_index, edge_store, backend, tokens[7], space_id=space_id)
    if n1 is None or n2 is None or a is None or b is None or cmp is None:
        return None   # 数字词未接地 / 比较 cue 未识别 → 诚实不伪造（守反统计契约·非凑配）
    sym = _CMP_TO_SYM.get(cmp)
    if sym is None:
        return None   # GT/LT/EQ 有 sym·GE/LE→None defer（code_observe 不支持 Ge/Le·NEQ 同 defer·piece 2.2+）
    # Python 源码模板（FunctionDef·code_observe 入口·ast 自动产 OPCODE_GT/LT 免手映射）+ vm_proof execute（反 theater）
    register_composes_attr(backend)   # 幂等（caller 已 register 亦无妨）
    # 唯一根/调用（deterministic counter·非 time/random）：复用同根累积多 COMPOSES 树致 read 混读错值（镜像 word_problem）
    root_ref = concept_index.ensure(f"__cp_exec_{next(_CP_EXEC_SEQ)}", space_id=space_id,
                                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    code_source = (
        f"def f():\n"
        f"    if {n1} {sym} {n2}:\n"
        f"        return {a}\n"
        f"    else:\n"
        f"        return {b}\n"
    )
    build_composes_from_source(code_source,
                               concept_index=concept_index, edge_store=edge_store,
                               backend=backend, space_id=space_id,
                               source=SOURCE_CODE, root_ref=root_ref)
    g = ConceptGraph(backend)
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        g.read_composes_tree(root_ref)
    instrs = compile_graph(root_ref, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    return execute(instrs, {})   # arity-0 → env 空·vm_proof 真执行 if/Compare/Return → HALT 栈顶
