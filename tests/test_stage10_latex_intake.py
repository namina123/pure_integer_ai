"""Stage 10 验收门测试：LaTeX 数学记号 intake → arith DSL（换皮·复用 _ArithBuilder）。

doc/重来_符号算子与一种最根本表达设计补充.md §五（intake 分流：LaTeX=字面文法承载→墙内 parser）。
LaTeX 子集 → Python lambda DSL 字符串 → build_composes_from_arith（_ArithBuilder 全复用·零新
builder/边/opcode·换皮=好典型）。本回合 2 对抗智能体验证落地（前提/边界 + 文法/确定性·5 致命漏洞
+ 2 架构决断全落：EQ token / 复合 body 须括号 / 自由变量 sorted / 撞魔法名 fail-loud / codegen 强制括号 /
不映射 Recur / 返回 (dsl, param_order)）。

覆盖：
  T1 LaTeX→DSL 翻译正确性（子集映射 + index 名重写 + 自由变量收集 sorted + param_order 语义契约）
  T2 e2e 执行正确（LaTeX→DSL→build→execute 算术正确·含精确有理除 + 嵌套和 + nullary）
  T3 vm_proof 闭环（reward=1 + mismatch→0·反 theater）
  T4 bit-identical（两独立 backend 同果·确定性）
  T5 生产闭环（LaTeX→DSL→item→run_round_full reward==1·非仅单测可达）
  T6 fail-loud 拒清单（C7 超越/极限/积分 + 语法不支持 + DSL 单魔法名限制）

铁律：纯整数 / 确定性 bit-identical / fail-loud / 依赖单向向下 / 复用 _ArithBuilder / 反 theater。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    ConceptRef, CodeSpec, MODALITY_ARITH, DOMAIN_MATH, LANG_NONE,
    TERMINAL_REACHED_SINK,
)
from pure_integer_ai.cognition.understanding.arith_observe import UnsupportedConstruct
from pure_integer_ai.cognition.understanding.latex_intake import (
    latex_to_arith_dsl, build_composes_from_latex,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.vm.graph_compile import compile_graph
from pure_integer_ai.vm.vm_core import execute
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.crosscut.integer.rational import make
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import make_train_context, DefaultRoundRunner
from pure_integer_ai.training.stages import STAGE3_REWARD


# ---- fixtures ----

@pytest.fixture
def arith_env():
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    yield b, sid, es, ci, g
    b.close()


def _build_latex(arith_env, latex: str, *, seg_label: str = "__seg_lat_0") -> ConceptRef:
    """LaTeX → COMPOSES 树·返 root=struct_ref。"""
    b, sid, es, ci, _ = arith_env
    root_ref = ci.ensure(seg_label, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_latex(latex, concept_index=ci, edge_store=es, backend=b,
                              space_id=sid, source=SOURCE_MATH, root_ref=root_ref)
    return root_ref


def _run(arith_env, root: ConceptRef, input_args: tuple[int, ...]):
    """read_composes_tree → compile → execute（预载 input_args·按 param_order 序）·返 Rational。"""
    _, _, _, _, g = arith_env
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        g.read_composes_tree(root)
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    env = {make_variable(i): make(int(a), 1) for i, a in enumerate(input_args)}
    return execute(instrs, env)


def _latex_item(dsl_src: str, specs) -> CollectedItem:
    """造算术域 CollectedItem（LaTeX 翻译后的 DSL 串·一段一 lambda + 多 spec）。"""
    return CollectedItem(
        modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
        source=SOURCE_MATH, arith_source=dsl_src, arith_specs=tuple(specs),
    )


# ============ T1 LaTeX→DSL 翻译正确性 ============

def test_latex_sum_translation():
    """\\sum_{i=1}^{n}{i} → lambda n: Sigma(1, n, i)·param_order=(n,)。"""
    dsl, params = latex_to_arith_dsl(r"\sum_{i=1}^{n}{i}")
    assert dsl == "lambda n: Sigma(1, n, i)", f"翻译错·得 {dsl}"
    assert params == ("n",)


def test_latex_index_name_rename_to_magic_i():
    """LaTeX index 名 k → DSL 魔法名 i（DSL hardwired·局部重写·非全局替换）。"""
    dsl, params = latex_to_arith_dsl(r"\sum_{k=1}^{n}{k}")
    assert dsl == "lambda n: Sigma(1, n, i)", f"k 须重写为 i·得 {dsl}"
    assert params == ("n",)


def test_latex_frac_translation():
    """\\frac{A}{B} → (A/B)（codegen 强制括号·避 1/2/3 左结合歧义）。"""
    dsl, _ = latex_to_arith_dsl(r"\frac{n}{2}")
    assert dsl == "lambda n: (n/2)", f"\\frac 翻译错·得 {dsl}"


def test_latex_pow_translation():
    """A^{K} / A^K → (A**K)（字面非负 int 指数·codegen 强制括号）。"""
    dsl, _ = latex_to_arith_dsl(r"n^{2}")
    assert dsl == "lambda n: (n**2)", f"翻译错·得 {dsl}"
    dsl2, _ = latex_to_arith_dsl(r"n^2")    # 单字符指数
    assert dsl2 == "lambda n: (n**2)"


def test_latex_two_free_vars_sorted():
    """多自由变量 sorted 字典序（bit-identical + vm_proof input_args 语义契约）。"""
    dsl, params = latex_to_arith_dsl(r"y^{2} + x^{2}")
    assert params == ("x", "y"), f"自由变量须 sorted·得 {params}"
    assert dsl == "lambda x, y: ((y**2)+(x**2))", f"翻译错·得 {dsl}"


def test_latex_nullary_no_free_var():
    """\\sum_{i=1}^{5}{i} 无自由变量 → nullary lambda:（hi 字面 5·i 是 index 非 free）。"""
    dsl, params = latex_to_arith_dsl(r"\sum_{i=1}^{5}{i}")
    assert params == (), f"无自由变量·得 {params}"
    assert dsl == "lambda: Sigma(1, 5, i)"


def test_latex_nested_sum_translation():
    """嵌套和 \\sum_{k=1}^{n}\\sum_{j=1}^{k}{j} → Sigma(1, n, Sigma(1, i, i))。

    两 index 名 k/j 都重写为 i（DSL 单魔法名）·内层 hi k=外层 index→i（DSL scope 栈解析为外层）·
    内层 body j=内层 index→i（scope 栈解析为本层）。语义保真。
    """
    dsl, params = latex_to_arith_dsl(r"\sum_{k=1}^{n}\sum_{j=1}^{k}{j}")
    assert dsl == "lambda n: Sigma(1, n, Sigma(1, i, i))", f"嵌套翻译错·得 {dsl}"
    assert params == ("n",)


# ============ T2 e2e 执行正确 ============

def test_latex_sum_e2e(arith_env):
    """\\sum_{i=1}^{n}{i} 执行 = 1+..+n·n=5→15·n=10→55。"""
    root = _build_latex(arith_env, r"\sum_{i=1}^{n}{i}")
    assert rational.eq(_run(arith_env, root, (5,)), make(15, 1))
    assert rational.eq(_run(arith_env, root, (10,)), make(55, 1))


def test_latex_prod_e2e(arith_env):
    """\\prod_{i=1}^{n}{i} = n!·n=5→120。"""
    root = _build_latex(arith_env, r"\prod_{i=1}^{n}{i}")
    assert rational.eq(_run(arith_env, root, (5,)), make(120, 1))


def test_latex_frac_exact_rational_e2e(arith_env):
    """\\frac{n}{2} = 精确有理除·n=3→3/2（非 float）·n=4→2。"""
    root = _build_latex(arith_env, r"\frac{n}{2}")
    assert _run(arith_env, root, (3,)) == make(3, 2), "须精确有理 Rational(3,2)"
    assert rational.eq(_run(arith_env, root, (4,)), make(2, 1))


def test_latex_cdot_e2e(arith_env):
    """n \\cdot (n+1) / 2 闭式·n=5→15。"""
    root = _build_latex(arith_env, r"\frac{n \cdot (n+1)}{2}")
    assert rational.eq(_run(arith_env, root, (5,)), make(15, 1))


def test_latex_pow_e2e(arith_env):
    """n^{2} → n**2·n=5→25。"""
    root = _build_latex(arith_env, r"n^{2}")
    assert rational.eq(_run(arith_env, root, (5,)), make(25, 1))


def test_latex_sum_pow_body_e2e(arith_env):
    """\\sum_{i=1}^{n}{i^{2}} = 1+4+9+...·n=3→14。"""
    root = _build_latex(arith_env, r"\sum_{i=1}^{n}{i^{2}}")
    assert rational.eq(_run(arith_env, root, (3,)), make(14, 1))


def test_latex_two_vars_e2e(arith_env):
    """x^{2} + y^{2} → x=3,y=4→25（input_args 按 sorted param_order=(x,y) 序传）。"""
    root = _build_latex(arith_env, r"x^{2} + y^{2}")
    assert rational.eq(_run(arith_env, root, (3, 4)), make(25, 1))


def test_latex_nested_sum_e2e(arith_env):
    """\\sum_{k=1}^{n}\\sum_{j=1}^{k}{j} → n=3→1+(1+2)+(1+2+3)=10。"""
    root = _build_latex(arith_env, r"\sum_{k=1}^{n}\sum_{j=1}^{k}{j}")
    assert rational.eq(_run(arith_env, root, (3,)), make(10, 1))


def test_latex_nullary_e2e(arith_env):
    """\\sum_{i=1}^{5}{i} nullary → 15（无 lambda 参数·input_args=()）。"""
    root = _build_latex(arith_env, r"\sum_{i=1}^{5}{i}")
    assert rational.eq(_run(arith_env, root, ()), make(15, 1))


def test_latex_left_right_paren_e2e(arith_env):
    """\\left( n + 1 \\right) → n=5→6（\\left/\\right 括号对）。"""
    root = _build_latex(arith_env, r"\left( n + 1 \right)")
    assert rational.eq(_run(arith_env, root, (5,)), make(6, 1))


# ============ T3 vm_proof 闭环（反 theater） ============

def test_latex_vm_proof_closed_loop_e2e(arith_env):
    """vm_proof 闭环：LaTeX 引用段经 vm_proof_fn 执行验对错·reward=1·mismatch→0（反 theater）。"""
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult
    _, _, _, _, g = arith_env
    root_ref = _build_latex(arith_env, r"\sum_{i=1}^{n}{i}", seg_label="__seg_lat_proof")
    dag = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_ref)
    # pass：n=5 → 15
    assert vm_proof_fn_factory(input_args=(5,), expected=(15, 1))(OutputResult(), dag, g) == 1
    # 反 theater：mismatch expected=14 → 0（证 reward 来自真 VM 执行·非 stub）
    assert vm_proof_fn_factory(input_args=(5,), expected=(14, 1))(OutputResult(), dag, g) == 0


# ============ T4 bit-identical ============

def test_latex_bit_identical():
    """同 LaTeX 两独立 backend 跑 bit-identical（执行结果一致·确定性）。"""
    latex = r"\sum_{i=1}^{n}{i}"

    def _build_and_run():
        b = DictBackend(); bootstrap(b); register_composes_attr(b)
        reg = SpaceRegistry(b); sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b); ci = ConceptIndex(b)
        root = ci.ensure("__seg_lat_0", space_id=sp.space_id,
                         tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_latex(latex, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sp.space_id, source=SOURCE_MATH, root_ref=root)
        g = ConceptGraph(b)
        ch, op, nd, imm, st = g.read_composes_tree(root)
        instrs = compile_graph(root, ch, op, nd,
                               immediate_of=imm or None, store_target_of=st or None)
        return execute(instrs, {make_variable(0): make(5, 1)})

    assert _build_and_run() == _build_and_run() == make(15, 1)


# ============ T5 生产闭环（非仅单测可达） ============

def test_latex_production_closed_loop_e2e():
    """生产闭环：LaTeX→DSL→item→run_round_full reward==1（经真实训练管线可达）。

    LaTeX 经 latex_to_arith_dsl 翻译为 DSL 串→CollectedItem.arith_source→observe→
    build_composes_from_arith→vm_proof。证 LaTeX intake 经生产路径可达·非仅单测。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    dsl, params = latex_to_arith_dsl(r"\sum_{i=1}^{n}{i}")
    assert params == ("n",)
    r = DefaultRoundRunner()
    item = _latex_item(dsl, [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1
    assert res.dag_path is not None and res.dag_path.sink == res.episode.ref


# ============ T6 fail-loud 拒清单 ============

def test_latex_sqrt_reject():
    """\\sqrt 拒（C7 超越值墙·查表双违·非静默）。"""
    with pytest.raises(UnsupportedConstruct, match="\\\\sqrt"):
        latex_to_arith_dsl(r"\sqrt{x}")


def test_latex_transcendental_reject():
    """超越函数 \\sin/\\cos/\\log/\\exp 全拒（C7 超越值墙·查表=硬编码数学词典双违）。"""
    for cmd in (r"\sin", r"\cos", r"\log", r"\exp", r"\ln", r"\tan"):
        with pytest.raises(UnsupportedConstruct):
            latex_to_arith_dsl(cmd + r"{x}")


def test_latex_lim_int_reject():
    """\\lim/\\int 拒（极限/积分值计算 defer·CAS·C7·建树空壳 theater）。"""
    with pytest.raises(UnsupportedConstruct, match="\\\\lim"):
        latex_to_arith_dsl(r"\lim_{x \to 0}{x}")
    with pytest.raises(UnsupportedConstruct, match="\\\\int"):
        latex_to_arith_dsl(r"\int_{0}^{1}{x}")


def test_latex_negative_exponent_reject():
    """负指数 x^{-1} 拒（须用 \\frac·DSL Pow 仅字面非负 int）。"""
    with pytest.raises(UnsupportedConstruct, match="指数须字面非负"):
        latex_to_arith_dsl(r"x^{-1}")


def test_latex_variable_exponent_reject():
    """变量指数 x^{n} 拒（须用 Recur·DSL Pow 仅字面非负 int）。"""
    with pytest.raises(UnsupportedConstruct, match="指数须字面非负"):
        latex_to_arith_dsl(r"x^{n}")


def test_latex_implicit_multiplication_reject():
    """隐式乘法 n(n+1) 拒（须显式 \\cdot 或 *·避 f(x) 函数调用歧义）。"""
    with pytest.raises(UnsupportedConstruct, match="多余 token"):
        latex_to_arith_dsl(r"n(n+1)")


def test_latex_decimal_reject():
    """小数点 1.5 拒（纯整数铁律·须用 \\frac{3}{2}）。"""
    with pytest.raises(UnsupportedConstruct, match="小数点"):
        latex_to_arith_dsl(r"x + 1.5")


def test_latex_compound_body_without_braces_reject():
    """复合 body 须括号：\\sum_{..}^{..} i \\cdot i（裸含 \\cdot）→ body=i 后残留 \\cdot i → fail-loud。

    复合 body 须 {..}/(..) 显式包裹（确定性·避 body 边界歧义）。
    """
    with pytest.raises(UnsupportedConstruct):
        latex_to_arith_dsl(r"\sum_{i=1}^{n} i \cdot i")


def test_latex_bare_i_outside_index_scope_reject():
    """裸 i 在 index 作用域外 \\sum_{k=1}^{n}{k} + i → trailing i 是自由变量→撞魔法名 i→fail-loud。"""
    with pytest.raises(UnsupportedConstruct, match="撞 DSL 魔法名"):
        latex_to_arith_dsl(r"\sum_{k=1}^{n}{k} + i")


def test_latex_free_var_named_a_reject():
    """自由变量名 a 撞 DSL 累加器魔法名 a → fail-loud（a 是 Recur 累加器·非 lambda 参数）。"""
    with pytest.raises(UnsupportedConstruct, match="撞 DSL 魔法名"):
        latex_to_arith_dsl(r"a + 1")


def test_latex_inner_body_references_outer_index_reject():
    """嵌套内层 body 引用外层 index \\sum_{k=1}^{n}\\sum_{j=1}^{k}{j+k} → DSL i 仅达本层·不可表达→fail-loud。"""
    with pytest.raises(UnsupportedConstruct, match="内层 body 引用外层 index"):
        latex_to_arith_dsl(r"\sum_{k=1}^{n}\sum_{j=1}^{k}{j + k}")


def test_latex_empty_reject():
    """空输入 → fail-loud。"""
    with pytest.raises(UnsupportedConstruct, match="为空"):
        latex_to_arith_dsl("")


def test_latex_unknown_command_reject():
    """未知命令 \\foo → fail-loud（非静默·显式拒）。"""
    with pytest.raises(UnsupportedConstruct):
        latex_to_arith_dsl(r"\foo{x}")
