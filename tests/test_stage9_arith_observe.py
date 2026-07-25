"""Stage 9 验收门测试：算术域 observe 建 COMPOSES 程序（A3 兄弟件·doc/重来_算术域observe设计补充.md）。

覆盖 6 层：
  L1 DSL→COMPOSES 映射单元（Sigma/Prod/Recur/闭式/`/`精确有理/Pow desugar/fail-loud 拒绝/bit-identical）
  L2 持久化往返（build→read_composes_tree→compile→execute 端到端算术正确·含精确有理除 + 嵌套和）
  L3 observe 集成（MODALITY_ARITH 段→EDGE_COMPOSES 落盘 + struct_ref=COMPOSES 根 + 不建 PRECEDES）
  L4 vm_proof_fn 闭环（arith item→vm_proof_fn 验 vs CodeSpec·modality-agnostic）
  L5 生产闭环（run_round_full MODALITY_ARITH·reward>0 涌现 + 反 theater 锚点：mismatch/deadloop/POST→0）
  L6 Mode B 演示（闭式 vs 迭代交叉自证·R6 非 theater·expected 手编硬编码）
  L7 inline-on-reference（符号算子"一种最根本表达"·承重第一步·Name→已学 nullary 结论 deep-copy 嫁接）
  L8 参数化 inline + β-归约（L1.5·承重第二步·Call→已学参数化算子→inline + β-归约·param i↔make_variable(i)）
  L9 归约规则 = 教师注入结论即程序（L3 注入层·学习层 theater defer·分配律/求导规则经 register_arith_operator inject+inline）

核心架构（doc §一/§九）：vm_proof_fn modality-agnostic·算术域 builder（lambda DSL→COMPOSES）+ MODALITY_ARITH
路由即闭环。`/`=OPCODE_DIV 精确有理（vs code_observe 拒 `/`）。词法 scope 栈解魔法名 i/a + 嵌套和。
铁律：纯整数 / 确定性 bit-identical / fail-loud / 依赖单向向下 / 反 theater（reward>0 测试配 reward=0 锚点）。
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
    TERMINAL_REACHED_SINK, WEANING_PRE, WEANING_POST,
)
from pure_integer_ai.cognition.understanding.arith_observe import (
    build_composes_from_arith, register_arith_operator, UnsupportedConstruct,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.vm.graph_compile import compile_graph, CTRL_WHILE
from pure_integer_ai.vm.vm_core import execute, StepLimitExceeded
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.crosscut.integer.rational import make
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import (
    make_train_context, DefaultRoundRunner, _h2_calibrate,
)
from pure_integer_ai.training.stages import STAGE2_CAUSES_ABS, STAGE3_REWARD


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


def _build(arith_env, src: str, *, seg_label: str = "__seg_1_0") -> ConceptRef:
    """建 COMPOSES 树·返 root=struct_ref。"""
    b, sid, es, ci, _ = arith_env
    root_ref = ci.ensure(seg_label, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(
        src, concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_MATH, root_ref=root_ref)
    return root_ref


def _run(arith_env, root: ConceptRef, input_args: tuple[int, ...]):
    """read_composes_tree → compile → execute（预载 input_args）·返 Rational。"""
    _, _, _, _, g = arith_env
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        g.read_composes_tree(root)
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    env = {make_variable(i): make(int(a), 1) for i, a in enumerate(input_args)}
    return execute(instrs, env)


def _arith_item(src: str, specs, *, source: int = SOURCE_MATH) -> CollectedItem:
    """造算术域 CollectedItem（一段一 lambda 记号 + 多测试用例 spec）。"""
    return CollectedItem(
        modality=MODALITY_ARITH,
        domain=DOMAIN_MATH,
        lang=LANG_NONE,
        source=source,
        arith_source=src,
        arith_specs=tuple(specs),
    )


# ============ L1 DSL→COMPOSES 映射单元 ============

def test_sigma_builds_ctrl_while(arith_env):
    """Sigma → ITER SEQ NOP 块·内含 CTRL_WHILE 控制流根。"""
    b, _, _, _, _ = arith_env
    root = _build(arith_env, "lambda n: Sigma(1, n, i)")
    g = ConceptGraph(b)
    _, operator_of, _, _, _ = g.read_composes_tree(root)
    # 子树中存在 CTRL_WHILE 控制流根
    ctrl_count = sum(1 for op in operator_of.values() if op == CTRL_WHILE)
    assert ctrl_count == 1, "Sigma 须建 1 个 CTRL_WHILE"


def test_div_maps_to_exact_rational(arith_env):
    """BinOp Div → OPCODE_DIV·执行得精确有理（n/2 n=1 → Rational(1,2)·非 float）。"""
    root = _build(arith_env, "lambda n: n / 2")
    result = _run(arith_env, root, (1,))
    assert result == make(1, 2), f"`/` 须精确有理除·得 {result}·期 Rational(1,2)"


def test_pow_desugar_to_repeated_mul(arith_env):
    """Pow 字面非负指数 → 重复 MUL（n**3 n=2 → 8）。"""
    root = _build(arith_env, "lambda n: n ** 3")
    assert rational.eq(_run(arith_env, root, (2,)), make(8, 1))
    # a**0 = 1（VM 约定）
    root0 = _build(arith_env, "lambda n: n ** 0")
    assert rational.eq(_run(arith_env, root0, (5,)), make(1, 1))


def test_lim_fail_loud_defer(arith_env):
    """Lim → fail-loud defer（CAS·C7 超越数墙外·非静默丢）。"""
    with pytest.raises(UnsupportedConstruct, match="Lim"):
        _build(arith_env, "lambda n: Lim(n, 1 / n)")


def test_compare_reject(arith_env):
    """Compare 全拒绝（DSL 用户表达式无比较·cond 由 builder 内部生成为 LT）。"""
    with pytest.raises(UnsupportedConstruct):
        _build(arith_env, "lambda n: n < 5")


def test_negative_exponent_reject(arith_env):
    """Pow 负指数（=UnaryOp）→ reject（须用 Recur 显式表达）。"""
    with pytest.raises(UnsupportedConstruct):
        _build(arith_env, "lambda n: n ** -1")


def test_reserved_lambda_arg_reject(arith_env):
    """lambda 参数禁用魔法名 i/a（fail-loud 避撞·doc §三必改#1）。"""
    with pytest.raises(UnsupportedConstruct):
        _build(arith_env, "lambda i: i")
    with pytest.raises(UnsupportedConstruct):
        _build(arith_env, "lambda a: a")


def test_unbound_var_fail_loud(arith_env):
    """未绑定变量（body 用了未声明的名）→ fail-loud（无静默错值）。"""
    with pytest.raises(UnsupportedConstruct, match="未绑定"):
        _build(arith_env, "lambda n: n + x")   # x 未声明


def test_entry_rejects_non_lambda(arith_env):
    """入口须 lambda（非 FunctionDef/非表达式）→ fail-loud（doc §三必改#2）。"""
    with pytest.raises(UnsupportedConstruct):
        _build(arith_env, "def f(n): return n")     # FunctionDef 非 lambda
    with pytest.raises(UnsupportedConstruct):
        _build(arith_env, "n + 1")                   # 非 lambda 表达式


def test_bit_identical():
    """同 arith_source 两独立 backend 跑 bit-identical（执行结果一致·确定性）。

    两独立 backend（不同 space_id）·同源码同遍历序→同 surface 结构→同执行结果。
    ConceptRef 跨 backend 不可比（space_id 异）·bit-identical 判执行结果一致性。
    """
    src = "lambda n: Sigma(1, n, i)"

    def _build_and_run():
        b = DictBackend(); bootstrap(b); register_composes_attr(b)
        reg = SpaceRegistry(b); sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b); ci = ConceptIndex(b)
        root = ci.ensure("__seg_1_0", space_id=sp.space_id,
                         tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith(src, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sp.space_id, source=SOURCE_MATH, root_ref=root)
        g = ConceptGraph(b)
        ch, op, nd, imm, st = g.read_composes_tree(root)
        instrs = compile_graph(root, ch, op, nd,
                               immediate_of=imm or None, store_target_of=st or None)
        return execute(instrs, {make_variable(0): make(5, 1)})

    assert _build_and_run() == _build_and_run() == make(15, 1)


# ============ L2 持久化往返（端到端算术正确） ============

def test_sigma_sum_e2e(arith_env):
    """Sigma(1,n,i) = 1+..+n·n=5→15·n=10→55。"""
    root = _build(arith_env, "lambda n: Sigma(1, n, i)")
    assert rational.eq(_run(arith_env, root, (5,)), make(15, 1))
    assert rational.eq(_run(arith_env, root, (10,)), make(55, 1))


def test_prod_factorial_e2e(arith_env):
    """Prod(1,n,i) = n!·n=5→120。"""
    root = _build(arith_env, "lambda n: Prod(1, n, i)")
    assert rational.eq(_run(arith_env, root, (5,)), make(120, 1))


def test_recur_factorial_e2e(arith_env):
    """Recur(1,n,a*i) = n!（单状态递推·a=累加器 i=索引）·n=5→120。"""
    root = _build(arith_env, "lambda n: Recur(1, n, a * i)")
    assert rational.eq(_run(arith_env, root, (5,)), make(120, 1))


def test_recur_equals_sigma(arith_env):
    """Recur(0,n,a+i) ≡ Sigma(1,n,i)（递推泛化验证）·n=5→15。

    两树用不同 seg_label（不同 struct_ref root_lid）·避 surface 冲突（同 args_sig+seq+type·
    须靠 root_lid 隔离·doc §三必改#2·镜像 code_observe root_lid 隔离）。
    """
    root_sigma = _build(arith_env, "lambda n: Sigma(1, n, i)", seg_label="__seg_sigma")
    root_recur = _build(arith_env, "lambda n: Recur(0, n, a + i)", seg_label="__seg_recur")
    assert _run(arith_env, root_sigma, (5,)) == _run(arith_env, root_recur, (5,)) == make(15, 1)


def test_closed_form_exact_rational_division_e2e(arith_env):
    """闭式 n*(n+1)/2 精确有理除·n=5→15（非 float）。"""
    root = _build(arith_env, "lambda n: n * (n + 1) / 2")
    assert rational.eq(_run(arith_env, root, (5,)), make(15, 1))
    assert rational.eq(_run(arith_env, root, (10,)), make(55, 1))


def test_nested_sigma_lexical_scope_e2e(arith_env):
    """嵌套和 Sigma(1,n,Sigma(1,i,i)) = Σ_{i=1..n} Σ_{j=1..i} j·词法 scope 栈·n=3→10。

    内层 lo/hi 的 i=外层索引·内层 body i=内层索引（doc §五 scope 时序）。
    """
    root = _build(arith_env, "lambda n: Sigma(1, n, Sigma(1, i, i))")
    # i=1:Σ_{j=1}^1 j=1 / i=2:1+2=3 / i=3:1+2+3=6 → 1+3+6=10
    assert rational.eq(_run(arith_env, root, (3,)), make(10, 1))


# ============ L3 observe 集成 ============

def test_observe_arith_segment_builds_composes_no_precedes():
    """MODALITY_ARITH 段→observe 建 COMPOSES 树·root=struct_ref·不建 PRECEDES（镜像代码域）。"""
    from pure_integer_ai.cognition.understanding.observe import observe
    from pure_integer_ai.cognition.shared.types import InputPayload, Segment, SpaceContext
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    core = AbstractSpace.create(reg, "core")
    sctx = SpaceContext(core=core, memory_read=None, memory_interact=None,
                        companion=None, stage=1)
    seg = Segment(seg_id=0, modality=MODALITY_ARITH, lang=LANG_NONE,
                  domain=DOMAIN_MATH, arith_source="lambda n: Sigma(1, n, i)")
    raw = InputPayload(segments=[seg], source=SOURCE_MATH, stage=1,
                       modality=MODALITY_ARITH, lang=LANG_NONE, domain=DOMAIN_MATH)
    ci = ConceptIndex(b)
    from pure_integer_ai.cognition.shared.work_memory import WorkMemory
    obs = observe(raw, sctx, concept_index=ci, work_memory=WorkMemory())
    # struct_ref 落 obs.struct_refs
    assert len(obs.struct_refs) == 1
    # EDGE_COMPOSES 边真落盘（observe MODALITY_ARITH gate 建树）
    composes = b.select("edge", where={"edge_type": EDGE_COMPOSES})
    assert len(composes) >= 2


# ============ Task #477 observe 多程序去重（struct_ref 内容哈希唯一化 + 幂等） ============


def _observe_arith_setup():
    """observe 直接调用环境（bootstrap + composes_attr + core space + sctx + ci）。"""
    from pure_integer_ai.cognition.shared.types import SpaceContext
    from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
    b = DictBackend(); bootstrap(b); register_composes_attr(b)
    reg = SpaceRegistry(b); core = AbstractSpace.create(reg, "core")
    sctx = SpaceContext(core=core, memory_read=None, memory_interact=None,
                        companion=None, stage=1)
    return b, sctx, ConceptIndex(b)


def _observe_one(sctx, ci, src: str):
    """observe 一段算术程序·返 obs（struct_refs[0]=struct_ref）。"""
    from pure_integer_ai.cognition.understanding.observe import observe
    from pure_integer_ai.cognition.shared.types import InputPayload, Segment
    from pure_integer_ai.cognition.shared.work_memory import WorkMemory
    seg = Segment(seg_id=0, modality=MODALITY_ARITH, lang=LANG_NONE,
                  domain=DOMAIN_MATH, arith_source=src)
    raw = InputPayload(segments=[seg], source=SOURCE_MATH, stage=1,
                       modality=MODALITY_ARITH, lang=LANG_NONE, domain=DOMAIN_MATH)
    return observe(raw, sctx, concept_index=ci, work_memory=WorkMemory())


def test_observe_multi_program_unique_struct_refs():
    """Task #477：两不同算术程序经 observe → 不同 struct_ref（内容哈希·无碰撞·多程序去重）。

    无 fix：__seg_{stage}_0·seg_idx 每 observe 重置→两程序撞同 struct_ref→重 build corrupt。
    """
    b, sctx, ci = _observe_arith_setup()
    ref1 = _observe_one(sctx, ci, "lambda: 5 * 5").struct_refs[0]
    ref2 = _observe_one(sctx, ci, "lambda: 6 * 6").struct_refs[0]
    assert ref1 != ref2, f"两不同程序须不同 struct_ref（内容哈希无碰撞）·得 {ref1} == {ref2}"


def test_observe_same_program_idempotent_no_duplicate_edges():
    """Task #477 幂等：同程序 observe 两次 → 同 struct_ref + COMPOSES 边不复制（已建 skip）。

    无 fix：seg_idx 重置→同 struct_ref·重 build→EdgeStore.add 复制边 corrupt 树（多轮训练 rounds_per_stage>1 触发）。
    """
    b, sctx, ci = _observe_arith_setup()
    _observe_one(sctx, ci, "lambda: 5 * 5")
    edges_after_1 = len(b.select("edge", where={"edge_type": EDGE_COMPOSES}))
    obs2 = _observe_one(sctx, ci, "lambda: 5 * 5")   # 同程序再 observe（模拟多轮）
    edges_after_2 = len(b.select("edge", where={"edge_type": EDGE_COMPOSES}))
    assert obs2.struct_refs[0] == _observe_one(sctx, ci, "lambda: 5 * 5").struct_refs[0], (
        "同程序→同 struct_ref（内容哈希）")
    assert edges_after_2 == edges_after_1, (
        f"同程序重 observe 须幂等（COMPOSES 边不复制）·{edges_after_1}→{edges_after_2}")


def test_observe_multi_program_trees_not_corrupted():
    """Task #477：两不同程序 observe → 各自 COMPOSES 树完整不互相 corrupt（碰撞致 corrupt 的反证）。

    无 fix：两程序撞同 struct_ref→边叠加 corrupt·vm_proof 执行错。修后：各自独立树·vm_proof 各自正确。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sctx, ci = _observe_arith_setup()
    ref1 = _observe_one(sctx, ci, "lambda: 5 * 5").struct_refs[0]
    ref2 = _observe_one(sctx, ci, "lambda: 100 + 7").struct_refs[0]
    g = ConceptGraph(b)
    # 各自树完整·vm_proof 执行对（碰撞 corrupt 会发散）
    v1 = execute_composes_value(g, ref1, ())
    v2 = execute_composes_value(g, ref2, ())
    assert v1 is not None and rational.eq(v1, rational.make(25, 1)), f"5*5=25·得 {v1}"
    assert v2 is not None and rational.eq(v2, rational.make(107, 1)), f"100+7=107·得 {v2}"


# ============ L4 vm_proof_fn 闭环（modality-agnostic） ============

def test_vm_proof_sigma_pass_e2e(arith_env):
    """arith item → vm_proof_fn 验 Sigma 执行 vs CodeSpec → 1（modality-agnostic·读 COMPOSES 树）。"""
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult
    b, _, _, _, _ = arith_env
    root = _build(arith_env, "lambda n: Sigma(1, n, i)")
    fn = vm_proof_fn_factory(input_args=(5,), expected=(15, 1))
    dag = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root)
    g = ConceptGraph(b)
    assert fn(OutputResult(), dag, g) == 1


def test_vm_proof_closed_form_pass_e2e(arith_env):
    """vm_proof_fn 验闭式 n*(n+1)/2 vs CodeSpec → 1。"""
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult
    b, _, _, _, _ = arith_env
    root = _build(arith_env, "lambda n: n * (n + 1) / 2")
    fn = vm_proof_fn_factory(input_args=(5,), expected=(15, 1))
    dag = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root)
    g = ConceptGraph(b)
    assert fn(OutputResult(), dag, g) == 1


# ============ L5 生产闭环（反 theater 锚点） ============

def test_arith_round_sigma_emerges_reward_positive_e2e():
    """涌现门：arith item（Sigma + spec f(5)=15）→ reward=1·G5 承重·ref=root·sink=root。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1
    assert res.episode.terminal == TERMINAL_REACHED_SINK
    assert res.episode.judge_G5_active is True
    assert res.episode.judge_veto_count == 0
    assert res.dag_path is not None and res.dag_path.sink == res.episode.ref


def test_arith_round_closed_form_reward_positive_e2e():
    """闭式 n*(n+1)/2 + spec f(5)=15 → reward=1。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: n * (n + 1) / 2", [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1


def test_arith_round_multi_spec_all_pass_e2e():
    """多 spec all-pass：Sigma f(5)=15 + f(10)=55 → reward=1。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)",
                       [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1


def test_arith_round_mismatch_reward_zero_e2e():
    """反 theater 锚点①：mismatch（Sigma spec f(5)=14 错·真值 15）→ reward=0。

    证 reward 来自真 VM 执行 vs spec·非 stub 返 1。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((5,), (14, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0
    assert res.episode.judge_veto_count == 1


def test_arith_round_deadloop_reward_zero_pre_e2e():
    """反 theater 锚点②：超大 Sigma（n=10^7·爆 step_limit）→ StepLimit→None→reward=0（PRE 非 vacate）。

    算术 DSL 构造都是有界循环（Sigma/Prod/Recur bounded）·无真死循环·用 step_limit 超界作诚实锚点
    （不验过不给 reward·证 reward 来自真 VM 执行 vs spec·非 stub）。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)",
                       [CodeSpec((10_000_000,), (999, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0, "超大 Sigma 爆 step_limit→None→PRE reward=0（非 vacate·反 theater）"
    assert res.episode.judge_G5_active is True   # PRE 仍 active（fail=veto）


def test_arith_round_mode_b_post_reward_zero_no_vacuous_e2e():
    """反 theater 锚点③：Mode B POST → reward=0·G5 不 active·不调 vm_proof（防 vacuous reward=1）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0
    assert res.episode.judge_G5_active is False


def test_arith_round_observe_only_stage_no_episode():
    """observe-only 阶段（STAGE2）：无 episode·EDGE_COMPOSES 边已落盘。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((5,), (15, 1))])
    ep = r.run_round(ctx, item, STAGE2_CAUSES_ABS, 0)
    assert ep is None   # observe-only·无 episode
    composes = b.select("edge", where={"edge_type": EDGE_COMPOSES})
    assert len(composes) >= 2


def test_arith_round_no_spec_honest_skip():
    """无 spec 诚实跳过：arith_source 有但 arith_specs=() → RoundResult()（无 episode·不伪造 reward=0）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)", [])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is None
    assert len(b.select("edge", where={"edge_type": EDGE_COMPOSES})) >= 2


def test_arith_round_no_source_honest_skip():
    """无 arith_source：MODALITY_ARITH 段无记号→_split 返空→RoundResult()（observe 无可建）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                         source=SOURCE_MATH, arith_source=None,
                         arith_specs=(CodeSpec((1,), (1, 1)),))
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is None


def test_h2_calibrate_skips_arith_items():
    """H2 标定排 arith：arith-only corpus → _h2_calibrate 跳 arith → 无 sample → 返 ctx.weights。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    arith_corpus = [_arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((5,), (15, 1))])]
    result = _h2_calibrate(ctx, arith_corpus, r)
    assert result is ctx.weights   # arith 全跳→samples 空→返 ctx.weights 原对象


def test_pre_flight_arith_corpus_compatible():
    """pre_flight 兼容：arith corpus → has_pos=True·metrics_signal=True·anti_collapse verified==0。"""
    from pure_integer_ai.experiments.formal_train import pre_flight
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_arith_item("lambda n: Sigma(1, n, i)",
                          [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])]
    rep = pre_flight(ctx, corpus, rounds=2)
    assert rep.metrics_signal is True
    assert rep.detail["graph_size"] > 0
    assert rep.reward_gate_ok is True
    assert rep.detail["has_pos_reward"] is True
    assert rep.detail["anti_collapse"]["verified"] == 0   # arith pr_vector 空→跳过（诚实）


def test_arith_round_bit_identical():
    """同 arith item 两跑 bit-identical（reward/ref/terminal 一致）。"""
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)",
                       [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])
    b1 = DictBackend(); ctx1 = make_train_context(b1)
    res1 = r.run_round_full(ctx1, item, STAGE3_REWARD, 0)
    b2 = DictBackend(); ctx2 = make_train_context(b2)
    res2 = r.run_round_full(ctx2, item, STAGE3_REWARD, 0)
    assert res1.episode.reward == res2.episode.reward
    assert res1.episode.ref == res2.episode.ref
    assert res1.episode.terminal == res2.episode.terminal
    assert res1.dag_path.sink == res2.dag_path.sink


# ============ L6 Mode B 演示（闭式 vs 迭代交叉自证·R6 非 theater） ============

def test_mode_b_closed_form_vs_iterative_cross_verify_r6_non_theater():
    """R6 非 theater：迭代 Σ 与闭式 n(n+1)/2 两独立句法源·各自 vm_proof 验 vs 手编 expected→一致。

    expected 手编硬编码（(15,1)/(55,1)·非 Python n*(n+1)//2 算出·否则与闭式 COMPOSES 同源=theater）。
    4 断言（doc §三必改#5）：
      ①迭代 COMPOSES vm_proof pass  ②闭式 COMPOSES vm_proof pass
      ③两 expected 独立手编却相等  ④换 input 仍各 pass
    证两独立句法源（Call Sigma vs BinOp）交叉吻合·R6 机制可行（生产 Mode B re-derivation defer）。
    """
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    # 手编硬编码 expected（独立源·非程序算出·n=5→15·n=10→55）
    hand_expected = {5: (15, 1), 10: (55, 1)}
    # 两独立句法源：迭代 Σ（CTRL_WHILE COMPOSES）/ 闭式（直线 BinOp COMPOSES）
    root_iter = ci.ensure("__seg_iter", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda n: Sigma(1, n, i)", concept_index=ci, edge_store=es,
                              backend=b, space_id=sid, source=SOURCE_MATH, root_ref=root_iter)
    root_closed = ci.ensure("__seg_closed", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda n: n * (n + 1) / 2", concept_index=ci, edge_store=es,
                              backend=b, space_id=sid, source=SOURCE_MATH, root_ref=root_closed)
    for n_val, exp in hand_expected.items():
        dag_i = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_iter)
        dag_c = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_closed)
        fn_i = vm_proof_fn_factory(input_args=(n_val,), expected=exp)
        fn_c = vm_proof_fn_factory(input_args=(n_val,), expected=exp)
        ri = fn_i(OutputResult(), dag_i, g)
        rc = fn_c(OutputResult(), dag_c, g)
        # ①迭代 pass ②闭式 pass ③④两独立源各匹配同一手编 expected（换 input 仍 pass）
        assert ri == 1, f"迭代 Σ n={n_val} 须 vm_proof pass"
        assert rc == 1, f"闭式 n={n_val} 须 vm_proof pass"


# ============ L7 inline-on-reference（符号算子"一种最根本表达"·承重第一步） ============
# doc/重来_符号算子与一种最根本表达设计补充.md。Name→已注册算子→deep-copy 其 struct_ref COMPOSES
# 子树(alpha-renaming)·建造期嫁接·read_composes_tree/graph_compile/vm_proof_fn 全零改。
# 注册由测试显式调 register_arith_operator(L1·observe 自动命名算子=L1.5+ defer)。

def test_inline_semantic_equivalence_e2e(arith_env):
    """inline 语义等价：引用段(S100+1) == 直接段(Sigma(1,100,i)+1)·VM 执行同果。

    已学 nullary 结论 S100=Σ_{1..100} i=5050·引用段 S100+1→inline 嫁接→执行=5051·与直接段同。
    证 inline 后展开树 VM 可执行·语义同直接建树(核心验证点)。
    """
    b, _, _, ci, _ = arith_env
    root_learned = _build(arith_env, "lambda: Sigma(1, 100, i)", seg_label="__seg_s100")
    register_arith_operator(b, ci, "S100", root_learned)
    root_ref = _build(arith_env, "lambda: S100 + 1", seg_label="__seg_ref")
    root_direct = _build(arith_env, "lambda: Sigma(1, 100, i) + 1", seg_label="__seg_direct")
    assert _run(arith_env, root_ref, ()) == _run(arith_env, root_direct, ()) == make(5051, 1)


def test_inline_alpha_renaming_no_capture_e2e(arith_env):
    """alpha-renaming 防 capture：inline 子树 internal sid 重分配·不撞引用段的 lambda arg sid。

    S5=Sigma(1,5,i)=15。引用段 lambda n: S5 + n 若 alpha-renaming 失败（保留原 sid 0,1）·
    则 Sigma 的 STORE 会覆盖 n（sid 0）→结果错。alpha 生效→S5 内部用 fresh sid·n 不被撞→n=10 得 25。
    两次 inline（lambda n: S5 + S5 + n）→两副本 sid 互异 + 与 n 异→得 40。

    注：跨段 sid 数值相等无害（独立子树·vm_proof 只执行 root_ref 子树·sid 是子树局部标签）·
    alpha 真不变量=引用段内不撞（副本 sid 不撞段自己的 lambda arg/其他副本）·靠 _alloc_internal
    从引用段 counter 取号保证。本测试用带 lambda arg 的执行正确性直接证此不变量。
    """
    b, _, _, ci, _ = arith_env
    root_learned = _build(arith_env, "lambda: Sigma(1, 5, i)", seg_label="__seg_learned")
    register_arith_operator(b, ci, "S5", root_learned)
    # 单 inline + lambda arg：alpha 失败则 Sigma STORE 撞 n→错值
    root1 = _build(arith_env, "lambda n: S5 + n", seg_label="__seg_r1")
    assert rational.eq(_run(arith_env, root1, (10,)), make(25, 1)), (
        "alpha-renaming 失败：n 被 inline STORE 撞（应 fresh 重分配）")
    # 两 inline + lambda arg：两副本须 sid 互异 + 与 n 异
    root2 = _build(arith_env, "lambda n: S5 + S5 + n", seg_label="__seg_r2")
    assert rational.eq(_run(arith_env, root2, (10,)), make(40, 1)), (
        "两次 inline sid 互撞或撞 n（应各 fresh 重分配）")


def test_inline_unregistered_name_fails_loud(arith_env):
    """未注册名→fail-loud(回退 _resolve_name_or_inline raise·非静默当 0/空)。"""
    with pytest.raises(UnsupportedConstruct, match="未绑定"):
        _build(arith_env, "lambda: UNKNOWN_OP + 1")


def test_inline_free_variable_fails_loud(arith_env):
    """非 nullary 结论(含 lambda arg 自由变量)inline→fail-loud(L1·参数化算子=Call 路径=L1.5 defer)。

    注册 F=lambda n: Sigma(1,n,i)(n 是自由变量)·引用 lambda: F→inline 校验 n 非子树 internal
    store_target→自由变量 fail-loud。
    """
    b, _, _, ci, _ = arith_env
    root_fn = _build(arith_env, "lambda n: Sigma(1, n, i)", seg_label="__seg_fn")
    register_arith_operator(b, ci, "F", root_fn)
    with pytest.raises(UnsupportedConstruct, match="自由变量"):
        _build(arith_env, "lambda: F", seg_label="__seg_call")


def test_inline_empty_struct_ref_fails_loud(arith_env):
    """struct_ref 无 COMPOSES 子树(空壳)inline→fail-loud(不可 inline 未建树的名)。"""
    b, sid, _, ci, _ = arith_env
    # 建一个纯概念点作 struct_ref(无 COMPOSES 子树)·注册成名后试图 inline
    empty_ref = ci.ensure("__seg_empty", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    register_arith_operator(b, ci, "EMPTY", empty_ref)
    with pytest.raises(UnsupportedConstruct, match="无 COMPOSES 子树"):
        _build(arith_env, "lambda: EMPTY", seg_label="__seg_call_empty")


def test_register_duplicate_name_conflict_fails_loud(arith_env):
    """同算子名映射不同 struct_ref→fail-loud 拒歧义(同 name 同 struct_ref 幂等)。"""
    b, _, _, ci, _ = arith_env
    root_a = _build(arith_env, "lambda: Sigma(1, 5, i)", seg_label="__seg_a")
    root_b = _build(arith_env, "lambda: Sigma(1, 6, i)", seg_label="__seg_b")
    register_arith_operator(b, ci, "OP", root_a)
    # 同 name 不同 struct_ref → 冲突 fail-loud
    with pytest.raises(UnsupportedConstruct, match="重名冲突"):
        register_arith_operator(b, ci, "OP", root_b)
    # 同 name 同 struct_ref → 幂等(不抛)
    register_arith_operator(b, ci, "OP", root_a)


def test_inline_bit_identical():
    """同注册+引用两独立 backend 跑 bit-identical(inline 结果一致·确定性)。"""
    def _build_and_run():
        b = DictBackend(); bootstrap(b); register_composes_attr(b)
        reg = SpaceRegistry(b); sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b); ci = ConceptIndex(b)
        root_l = ci.ensure("__seg_l", space_id=sp.space_id,
                           tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith("lambda: Sigma(1, 5, i)", concept_index=ci, edge_store=es,
                                  backend=b, space_id=sp.space_id, source=SOURCE_MATH,
                                  root_ref=root_l)
        register_arith_operator(b, ci, "S5", root_l)
        root_r = ci.ensure("__seg_r", space_id=sp.space_id,
                           tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith("lambda: S5 + 1", concept_index=ci, edge_store=es,
                                  backend=b, space_id=sp.space_id, source=SOURCE_MATH,
                                  root_ref=root_r)
        g = ConceptGraph(b)
        ch, op, nd, imm, st = g.read_composes_tree(root_r)
        instrs = compile_graph(root_r, ch, op, nd, immediate_of=imm or None,
                               store_target_of=st or None)
        return execute(instrs, {})

    assert _build_and_run() == _build_and_run() == make(16, 1)


def test_inline_vm_proof_closed_loop_e2e(arith_env):
    """vm_proof 闭环：inline 引用段经 vm_proof_fn 执行验对错·reward=1·mismatch→0(反 theater)。"""
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult
    b, _, _, ci, g = arith_env
    root_learned = _build(arith_env, "lambda: Sigma(1, 5, i)", seg_label="__seg_l")
    register_arith_operator(b, ci, "S5", root_learned)
    root_ref = _build(arith_env, "lambda: S5", seg_label="__seg_ref")
    dag = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_ref)
    # pass：S5=15
    assert vm_proof_fn_factory(input_args=(), expected=(15, 1))(OutputResult(), dag, g) == 1
    # 反 theater：mismatch expected=14→0(证 reward 来自真 VM 执行·非 stub)
    assert vm_proof_fn_factory(input_args=(), expected=(14, 1))(OutputResult(), dag, g) == 0


def test_inline_production_closed_loop_e2e():
    """生产闭环：ctx 注册已学结论→引用段 run_round_full→observe(复用 ctx.concept_index)→inline→vm_proof reward==1。

    证 inline 经真实训练管线(observe→build_composes_from_arith→Name 分支→_try_inline_learned)
    可达·非仅单测可达。注册与 observe 共用 ctx.concept_index(ConceptIndex.lookup 纯内存)。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    # 在 ctx.backend + ctx.concept_index 预建已学结论(observe 复用 ctx.concept_index·inline 命中)
    learned_root = ctx.concept_index.ensure(
        "__seg_learned", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda: Sigma(1, 5, i)", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=b,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=learned_root)
    register_arith_operator(b, ctx.concept_index, "S5", learned_root)
    # 引用段经生产路径(observe 用 ctx.concept_index→Name S5→inline 命中)
    r = DefaultRoundRunner()
    item = _arith_item("lambda: S5", [CodeSpec((), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1
    assert res.dag_path is not None and res.dag_path.sink == res.episode.ref


# ============ L8 参数化 inline + β-归约（L1.5·承重第二步） ============
# doc/重来_符号算子与一种最根本表达设计补充.md §三5。Call→已注册参数化算子→inline + β-归约
# （param i↔make_variable(i)·实参子树每用 fresh 嫁接 + arg internal alpha）。本回合 2 对抗智能体验证：
# Agent1（正确性·16 trace 不破）+ Agent2（架构·ATTR_ARITY 挂 name 节点非 root 消复制循环 latent bug）。
# 注册由测试显式调 register_arith_operator(name, struct_ref, arity=N)。

def test_call_inline_semantic_equivalence_e2e(arith_env):
    """Call 语义等价：square=λx.x*x(arity 1)·λn: square(n) == λn: n*n·VM 执行同果。[capture C1]

    square 形参 x=make_variable(0)·caller 形参 n=make_variable(0)·同 sid 值。β-归约移除 square 形参
    （替换为 caller n 子树）→ param/caller sid 数值相撞无害（对抗 Agent1 核证）。n=5→25·n=7→49。
    """
    b, _, _, ci, _ = arith_env
    root_sq = _build(arith_env, "lambda x: x * x", seg_label="__seg_sq")
    register_arith_operator(b, ci, "square", root_sq, arity=1)
    root_ref = _build(arith_env, "lambda n: square(n)", seg_label="__seg_ref")
    root_direct = _build(arith_env, "lambda n: n * n", seg_label="__seg_direct")
    assert _run(arith_env, root_ref, (5,)) == _run(arith_env, root_direct, (5,)) == make(25, 1)
    assert rational.eq(_run(arith_env, root_ref, (7,)), make(49, 1))


def test_call_inline_beta_arg_subtree_e2e(arith_env):
    """β 实参子树：λn: square(n+1) == (n+1)²·实参是复杂子树（非单变量）·每用 fresh 拷贝。n=3→16。"""
    b, _, _, ci, _ = arith_env
    root_sq = _build(arith_env, "lambda x: x * x", seg_label="__seg_sq")
    register_arith_operator(b, ci, "square", root_sq, arity=1)
    root_ref = _build(arith_env, "lambda n: square(n + 1)", seg_label="__seg_ref")
    root_direct = _build(arith_env, "lambda n: (n + 1) * (n + 1)", seg_label="__seg_direct")
    assert _run(arith_env, root_ref, (3,)) == _run(arith_env, root_direct, (3,)) == make(16, 1)


def test_call_inline_multi_use_arg_no_alias_e2e(arith_env):
    """多用实参无别名（对抗 Agent1 Finding4 关键）：double=λx.x+x·λ: double(Sigma(1,3,i)) == 6+6=12。

    实参 Sigma(1,3,i)=6 含 STORE（acc/idx）。double 体用 x 两次→两次 fresh 拷贝实参子树·
    且 arg 拷贝 alpha 自己的 internal STORE sid（不共享）→ 无别名。若共享同一 Sigma 的 STORE→错值。
    对比直接段（两独立 Sigma）同果=12。
    """
    b, _, _, ci, _ = arith_env
    rootdbl = _build(arith_env, "lambda x: x + x", seg_label="__seg_dbl")
    register_arith_operator(b, ci, "double", rootdbl, arity=1)
    root_ref = _build(arith_env, "lambda: double(Sigma(1, 3, i))", seg_label="__seg_ref")
    root_direct = _build(arith_env, "lambda: Sigma(1, 3, i) + Sigma(1, 3, i)",
                         seg_label="__seg_direct")
    assert _run(arith_env, root_ref, ()) == _run(arith_env, root_direct, ()) == make(12, 1)


def test_call_inline_operator_with_internal_e2e(arith_env):
    """算子含 internal（param + Sigma acc/idx 交互）：sumto=λx: Sigma(1,x,i)·λn: sumto(n) == Σ_{1..n} i。

    sumto 形参 x=mv0·Sigma acc/idx=internal·alpha-renamed·x 经 β-归约替换为 caller n。n=5→15。
    """
    b, _, _, ci, _ = arith_env
    root_st = _build(arith_env, "lambda x: Sigma(1, x, i)", seg_label="__seg_sumto")
    register_arith_operator(b, ci, "sumto", root_st, arity=1)
    root_ref = _build(arith_env, "lambda n: sumto(n)", seg_label="__seg_ref")
    root_direct = _build(arith_env, "lambda n: Sigma(1, n, i)", seg_label="__seg_direct")
    assert _run(arith_env, root_ref, (5,)) == _run(arith_env, root_direct, (5,)) == make(15, 1)


def test_call_inline_nested_operators_e2e(arith_env):
    """嵌套算子（对抗 Agent1 Finding3）：A=λx:x+1·B=λx:A(x)*2·λn: B(n) == (n+1)*2·n=5→12。

    A 在 B 建造期已 inline 进 B 树（注册序保证·否则 B 建造 fail-loud）·inline B 复制预展开树
    （含已 inline 的 A）·caller inline 期不递归解析算子。B 形参 x 经 β-归约替换为 caller n。
    """
    b, _, _, ci, _ = arith_env
    root_a = _build(arith_env, "lambda x: x + 1", seg_label="__seg_a")
    register_arith_operator(b, ci, "A", root_a, arity=1)
    root_b = _build(arith_env, "lambda x: A(x) * 2", seg_label="__seg_b")
    register_arith_operator(b, ci, "B", root_b, arity=1)
    root_ref = _build(arith_env, "lambda n: B(n)", seg_label="__seg_ref")
    root_direct = _build(arith_env, "lambda n: (n + 1) * 2", seg_label="__seg_direct")
    assert _run(arith_env, root_ref, (5,)) == _run(arith_env, root_direct, (5,)) == make(12, 1)


def test_call_inline_two_arg_operator_e2e(arith_env):
    """两参算子：add=λx,y:x+y(arity 2)·λn: add(n, n*2) == n + n*2 == 3n·n=3→9。

    两形参 x=mv0·y=mv1·各 β-归约替换为对应实参子树（n / n*2）。param i↔make_variable(i) 位置映射。
    """
    b, _, _, ci, _ = arith_env
    root_add = _build(arith_env, "lambda x, y: x + y", seg_label="__seg_add")
    register_arith_operator(b, ci, "add", root_add, arity=2)
    root_ref = _build(arith_env, "lambda n: add(n, n * 2)", seg_label="__seg_ref")
    root_direct = _build(arith_env, "lambda n: n + n * 2", seg_label="__seg_direct")
    assert _run(arith_env, root_ref, (3,)) == _run(arith_env, root_direct, (3,)) == make(9, 1)


def test_call_arity_mismatch_fails_loud(arith_env):
    """Call arity 不匹配：square(1,2)（arity 1·给 2 参）→ fail-loud（拒静默丢参/越界）。"""
    b, _, _, ci, _ = arith_env
    root_sq = _build(arith_env, "lambda x: x * x", seg_label="__seg_sq")
    register_arith_operator(b, ci, "square", root_sq, arity=1)
    with pytest.raises(UnsupportedConstruct, match="arity 不匹配"):
        _build(arith_env, "lambda: square(1, 2)", seg_label="__seg_ref")


def test_call_unregistered_name_fails_loud(arith_env):
    """Call 未注册名：unknown(5)→fail-loud（回退 _build_call raise·非静默当 0/空）。"""
    with pytest.raises(UnsupportedConstruct, match="Call 不支持"):
        _build(arith_env, "lambda: unknown(5)")


def test_parameterized_name_reference_fails_loud(arith_env):
    """参数化算子裸名引用：square(arity 1)·lambda: square（无 Call）→ fail-loud（须 Call·无函数值）。"""
    b, _, _, ci, _ = arith_env
    root_sq = _build(arith_env, "lambda x: x * x", seg_label="__seg_sq")
    register_arith_operator(b, ci, "square", root_sq, arity=1)
    with pytest.raises(UnsupportedConstruct, match="须 Call"):
        _build(arith_env, "lambda: square", seg_label="__seg_ref")


def test_call_inline_free_variable_via_misregistration_fails_loud(arith_env):
    """误注册（arity 过低）→ inline 撞自由变量：add=λx,y:x+y 实际 arity 2·误注册 arity=1→add(5) 撞 y 自由变量。

    arg_subst 仅含 mv0（x）·body 的 mv1（y）非 internal 非 param→fail_on_external fail-loud。
    （守闭项·防御性：build 期已解析一切 Name·自由变量仅误注册可触发。）
    """
    b, _, _, ci, _ = arith_env
    root_add = _build(arith_env, "lambda x, y: x + y", seg_label="__seg_add")
    register_arith_operator(b, ci, "add", root_add, arity=1)   # 误注册（实际 arity 2）
    with pytest.raises(UnsupportedConstruct, match="自由变量"):
        _build(arith_env, "lambda: add(5)", seg_label="__seg_ref")


def test_call_inline_bit_identical():
    """同注册+Call 引用两独立 backend 跑 bit-identical（参数化 inline 结果一致·确定性）。"""
    def _build_and_run():
        b = DictBackend(); bootstrap(b); register_composes_attr(b)
        reg = SpaceRegistry(b); sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b); ci = ConceptIndex(b)
        root_sq = ci.ensure("__seg_sq", space_id=sp.space_id,
                            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith("lambda x: x * x", concept_index=ci, edge_store=es,
                                  backend=b, space_id=sp.space_id, source=SOURCE_MATH,
                                  root_ref=root_sq)
        register_arith_operator(b, ci, "square", root_sq, arity=1)
        root_r = ci.ensure("__seg_r", space_id=sp.space_id,
                           tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith("lambda n: square(n) + 1", concept_index=ci, edge_store=es,
                                  backend=b, space_id=sp.space_id, source=SOURCE_MATH,
                                  root_ref=root_r)
        g = ConceptGraph(b)
        ch, op, nd, imm, st = g.read_composes_tree(root_r)
        instrs = compile_graph(root_r, ch, op, nd, immediate_of=imm or None,
                               store_target_of=st or None)
        return execute(instrs, {make_variable(0): make(5, 1)})

    assert _build_and_run() == _build_and_run() == make(26, 1)   # square(5)+1 = 25+1


def test_call_inline_vm_proof_closed_loop_e2e(arith_env):
    """vm_proof 闭环：参数化 inline 引用段经 vm_proof_fn 执行验对错·reward=1·mismatch→0（反 theater）。"""
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult
    b, _, _, ci, g = arith_env
    root_sq = _build(arith_env, "lambda x: x * x", seg_label="__seg_sq")
    register_arith_operator(b, ci, "square", root_sq, arity=1)
    root_ref = _build(arith_env, "lambda n: square(n)", seg_label="__seg_ref")
    dag = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_ref)
    # pass：square(5)=25
    assert vm_proof_fn_factory(input_args=(5,), expected=(25, 1))(OutputResult(), dag, g) == 1
    # 反 theater：mismatch expected=26→0（证 reward 来自真 VM 执行·非 stub）
    assert vm_proof_fn_factory(input_args=(5,), expected=(26, 1))(OutputResult(), dag, g) == 0


def test_call_inline_production_closed_loop_e2e():
    """生产闭环：ctx 注册参数化算子→引用段 run_round_full→observe(复用 ctx.concept_index)→Call inline→vm_proof reward==1。

    证参数化 inline 经真实训练管线（observe→build_composes_from_arith→_build_call→_try_inline_learned）
    可达·非仅单测可达。注册与 observe 共用 ctx.concept_index（ConceptIndex.lookup 纯内存）。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    # 在 ctx.backend + ctx.concept_index 预建参数化算子 square（observe 复用 ctx.concept_index·Call 命中）
    sq_root = ctx.concept_index.ensure(
        "__seg_square", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda x: x * x", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=b,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=sq_root)
    register_arith_operator(b, ctx.concept_index, "square", sq_root, arity=1)
    # 引用段经生产路径（observe 用 ctx.concept_index→Call square→inline + β）
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: square(n)", [CodeSpec((5,), (25, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1
    assert res.dag_path is not None and res.dag_path.sink == res.episode.ref


# ============ L9 归约规则 = 教师注入结论即程序（L3 注入层·学习层 theater defer） ============
# doc/重来_符号算子与一种最根本表达设计补充.md §七 row5 + §三6：归约规则"学来"= theater（COMPOSES 边
# reward=否·无反传→断奶后才能学）。但**注入层**（断奶前教师注入 reduction-rule 元定义为注册算子）现在可行·
# 经 register_arith_operator（L1/L1.5 已落）inject + inline。**机制已存在**（结论即程序·struct_ref+COMPOSES+inline）·
# 本段 = demo 证既有原语的 reach 覆盖归约规则（分配律恒等 / 求导规则）·非新建大机制。学习层（系统自学归约）defer。

def test_reduction_distributive_identity_e2e(arith_env):
    """分配律 a*(b+c) = a*b+a*c 作为教师注入算子 distrib=λa,b,c:a*b+a*c（arity 3）·引用 inline == 直算。

    注入 reduction-rule 元定义（分配律恒等）·经 register_arith_operator(arity=3)·Call 引用 β-归约。
    distrib(2,3,4) = 2*3+2*4 = 14 == 直算 2*(3+4) = 14（值等价证恒等作为程序成立）。arity 3 = L8 未覆盖。
    """
    b, _, _, ci, _ = arith_env
    root_distrib = _build(arith_env, "lambda x, y, z: x * y + x * z", seg_label="__seg_distrib")
    register_arith_operator(b, ci, "distrib", root_distrib, arity=3)
    root_ref = _build(arith_env, "lambda n: distrib(n, n + 1, n + 2)", seg_label="__seg_ref")
    # distrib(n, n+1, n+2) = n*(n+1) + n*(n+2)·n=2 → 2*3 + 2*4 = 6+8 = 14 == 2*(3+4) = 14
    assert rational.eq(_run(arith_env, root_ref, (2,)), make(14, 1))
    # 值等价：直算 n*(n+1) + n*(n+2) 同果
    root_direct = _build(arith_env, "lambda n: n * (n + 1) + n * (n + 2)", seg_label="__seg_direct")
    assert _run(arith_env, root_ref, (5,)) == _run(arith_env, root_direct, (5,))


def test_reduction_derivative_rule_e2e(arith_env):
    """求导规则 d/dx(x²)=2x 作为教师注入算子 deriv_xsq=λx:2*x·引用 inline == 2*x（结论即程序）。

    求导 = 背表 + 组合机械（memory lim 机制修正认知）·d/dx(x²)=2x 作为【关系】结论（非【值】计算）经
    算子 inline 表达（C7 拦超越值不拦变换关系）。注入 layer：教师注入求导规则元定义·系统引用即用。
    deriv_xsq(5) = 2*5 = 10·deriv_xsq(7) = 14。
    """
    b, _, _, ci, _ = arith_env
    root_deriv = _build(arith_env, "lambda x: 2 * x", seg_label="__seg_deriv")
    register_arith_operator(b, ci, "deriv_xsq", root_deriv, arity=1)
    root_ref = _build(arith_env, "lambda n: deriv_xsq(n)", seg_label="__seg_ref")
    assert rational.eq(_run(arith_env, root_ref, (5,)), make(10, 1))
    assert rational.eq(_run(arith_env, root_ref, (7,)), make(14, 1))


def test_reduction_rule_composition_e2e(arith_env):
    """归约规则组合：sumsq=λn:n*n（平方）·deriv_chain=λn:2*sumsq(n)·引用 inline 嵌套展开。

    一个归约规则引用另一个（嵌套 inline·A 体引 B·B 在 A 建造期已 inline 进 A 树）。
    deriv_chain(n) = 2*sumsq(n) = 2*n*n·n=3 → 2*9 = 18。证归约规则可组合（结论即程序链）。
    """
    b, _, _, ci, _ = arith_env
    root_sumsq = _build(arith_env, "lambda n: n * n", seg_label="__seg_sumsq")
    register_arith_operator(b, ci, "sumsq", root_sumsq, arity=1)
    root_chain = _build(arith_env, "lambda n: 2 * sumsq(n)", seg_label="__seg_chain")
    register_arith_operator(b, ci, "deriv_chain", root_chain, arity=1)
    root_ref = _build(arith_env, "lambda n: deriv_chain(n)", seg_label="__seg_ref")
    assert rational.eq(_run(arith_env, root_ref, (3,)), make(18, 1))


def test_reduction_vm_proof_closed_loop_e2e(arith_env):
    """归约规则 vm_proof 闭环：注入求导规则·引用段 vm_proof_fn 验·reward=1·mismatch→0（反 theater）。"""
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult
    b, _, _, ci, g = arith_env
    root_deriv = _build(arith_env, "lambda x: 2 * x", seg_label="__seg_deriv_proof")
    register_arith_operator(b, ci, "deriv_xsq", root_deriv, arity=1)
    root_ref = _build(arith_env, "lambda n: deriv_xsq(n)", seg_label="__seg_ref_proof")
    dag = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_ref)
    # pass：n=5 → 2*5 = 10
    assert vm_proof_fn_factory(input_args=(5,), expected=(10, 1))(OutputResult(), dag, g) == 1
    # 反 theater：mismatch expected=11 → 0
    assert vm_proof_fn_factory(input_args=(5,), expected=(11, 1))(OutputResult(), dag, g) == 0
