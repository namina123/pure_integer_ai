# tests/test_discover_iteration.py — S3 discover-迭代骨架测试（doc/重来_S3S4迭代机制设计 §三-bis）
"""S3 ctrl/store-迭代骨架（Sigma/Prod/Recur·CTRL_WHILE+STORE）discover-泛化机制测试。

权威设计 = doc/重来_S3S4迭代机制设计_2026-07-16.md §三-bis（4 改点加性扩展·直线 discover 零回归）。
覆盖：
  (a) ≥2 异参名 Sigma(1,n,i) 样本 → discover_skeleton 非 None + arity 正确
  (b) discover 产迭代骨架 → register_arith_operator → vm_proof sum_1_to_N(7)=28 真**反 theater**
  (c) Prod(1,n,i) n! 迭代同（+ Recur factorial）
  (d) internal sid 不撞 PARAM（vm_proof 多 n 验值正确）
  (e) 异 body（Sigma i vs i*i）分异（不混同→None）
  (f) held-out 识别③（discover 2 + recognize 第3 新参名·**alpha-实例识别**·非跨结构泛化·真泛化=参数化骨架对任意 n 值 测(d)）
  (g) run_capability_exam ③计算 permille>0（核心反 theater 信号·现 0·扩后>0）

铁律：纯整数 / 确定性（sorted 锁 sid_remap）/ 加性（直线零回归）/ fail-loud / 反 theater（vm_proof 真验值）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.process.structure_discover import (
    discover_skeleton, probe_arity, auto_discover_operators, recognize_operators)
from pure_integer_ai.vm.graph_compile import compile_graph
from pure_integer_ai.vm.vm_core import execute
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.crosscut.integer.rational import make
from pure_integer_ai.crosscut.integer import rational


# ---- fixtures（镜像 test_stage9 disc_env 范式） ----

@pytest.fixture
def disc_env():
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


def _build(disc_env, src: str, *, seg_label: str):
    b, sid, es, ci, _ = disc_env
    root_ref = ci.ensure(seg_label, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(
        src, concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_MATH, root_ref=root_ref)
    return root_ref


def _run(disc_env, root, input_args: tuple[int, ...]):
    """read_composes_tree → compile → execute（预载 input_args）·返 Rational。"""
    _, _, _, _, g = disc_env
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        g.read_composes_tree(root)
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    env = {make_variable(i): make(int(a), 1) for i, a in enumerate(input_args)}
    return execute(instrs, env)


# ============ (a)(b) Sigma discover + vm_proof 反 theater ============

def test_sigma_discovers_and_vm_proof_sum(disc_env):
    """(a)(b) ≥2 异参名 Sigma(1,n,i) → discover 成功 arity=1 + vm_proof sum_1_to_N(7)=28 反 theater。

    反 theater：discover 产迭代骨架·vm_proof 直接 execute 验值（非"建了=活"）。
    internal sid（acc/idx）alpha 到 make_variable(arity+k)·避 PARAM 区 mv0·vm_proof 绑 mv0=input n·STORE/LOAD 管 internal。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__sig1")
    s2 = _build(disc_env, "lambda m: Sigma(1, m, i)", seg_label="__sig2")
    # drift 防线：probe_arity == discover_skeleton.arity
    assert probe_arity(b, [s1, s2]) == 1, "probe_arity Sigma → 1（n=PARAM）"
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="sigma")
    assert res is not None, "Sigma discover 须成功（ctrl/store 已支持·非 defer）"
    assert res.arity == 1, "Sigma(1,n,i) arity=1（n=PARAM·acc/idx=internal 非 PARAM）"
    # 反 theater vm_proof：sum_1_to_N(7)=28·sum_1_to_N(10)=55·sum_1_to_N(100)=5050
    assert rational.eq(_run(disc_env, res.skeleton_ref, (7,)), make(28, 1)), "sum_1_to_7=28"
    assert rational.eq(_run(disc_env, res.skeleton_ref, (10,)), make(55, 1)), "sum_1_to_10=55"
    assert rational.eq(_run(disc_env, res.skeleton_ref, (100,)), make(5050, 1)), "sum_1_to_100=5050"


def test_sigma_register_inline_consumes(disc_env):
    """register_arith_operator 注册迭代骨架 → inline Call 嫁接消费（下游消费非摆设·反 theater）。

    sumfunc(n) = Sigma(1,n,i) 骨架注册后·lambda k: sumfunc(k) inline+β-归约 → 执行 == 直接 Sigma(k)。
    """
    from pure_integer_ai.cognition.understanding.arith_observe import register_arith_operator
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__sig1")
    s2 = _build(disc_env, "lambda m: Sigma(1, m, i)", seg_label="__sig2")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="sigma")
    assert res is not None
    register_arith_operator(b, ci, "sum1toN", res.skeleton_ref, arity=res.arity)
    use = _build(disc_env, "lambda k: sum1toN(k)", seg_label="__use")
    direct = _build(disc_env, "lambda k: Sigma(1, k, i)", seg_label="__direct")
    assert rational.eq(_run(disc_env, use, (7,)), make(28, 1)), "inline sum1toN(7)=28"
    assert _run(disc_env, use, (6,)) == _run(disc_env, direct, (6,)) == make(21, 1), (
        "inline β 嫁接骨架 == 直接 Sigma（下游消费真活·反 theater）")


# ============ (c) Prod / Recur 迭代 ============

def test_prod_discovers_factorial(disc_env):
    """(c) Prod(1,n,i) n! 迭代 → discover 成功 + vm_proof 5!=120。"""
    b, sid, es, ci, _ = disc_env
    p1 = _build(disc_env, "lambda n: Prod(1, n, i)", seg_label="__prod1")
    p2 = _build(disc_env, "lambda m: Prod(1, m, i)", seg_label="__prod2")
    assert probe_arity(b, [p1, p2]) == 1
    res = discover_skeleton([p1, p2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="prod")
    assert res is not None and res.arity == 1, "Prod discover arity=1"
    # 5! = 120·6! = 720·3! = 6
    assert rational.eq(_run(disc_env, res.skeleton_ref, (5,)), make(120, 1)), "5!=120"
    assert rational.eq(_run(disc_env, res.skeleton_ref, (6,)), make(720, 1)), "6!=720"
    assert rational.eq(_run(disc_env, res.skeleton_ref, (3,)), make(6, 1)), "3!=6"


def test_recur_discovers_factorial(disc_env):
    """(c) Recur(1,n,a*i) 递推累乘（a=累加器·Recur 第三迭代 builder）→ discover + vm_proof 5!=120。"""
    b, sid, es, ci, _ = disc_env
    r1 = _build(disc_env, "lambda n: Recur(1, n, a*i)", seg_label="__recur1")
    r2 = _build(disc_env, "lambda m: Recur(1, m, a*i)", seg_label="__recur2")
    assert probe_arity(b, [r1, r2]) == 1
    res = discover_skeleton([r1, r2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="recur")
    assert res is not None and res.arity == 1, "Recur discover arity=1"
    assert rational.eq(_run(disc_env, res.skeleton_ref, (5,)), make(120, 1)), "Recur 5!=120"
    assert rational.eq(_run(disc_env, res.skeleton_ref, (4,)), make(24, 1)), "Recur 4!=24"


# ============ (d) internal sid 不撞 PARAM（多 n 验值） ============

def test_internal_sid_no_param_collision_multi_n(disc_env):
    """(d) internal sid alpha 到 make_variable(arity+k)·与 PARAM mv0 区不撞·vm_proof 多 n 验值正确。

    命门验证：若 internal sid 撞 PARAM（如 acc 错占 mv0）·vm_proof 会静默错值（LOAD 到 input n 而非 acc）。
    多 n 值（1,2,...,20）逐验 sum_1_to_n = n*(n+1)/2·全对 = internal/PARAM 区精确不撞。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__sig1")
    s2 = _build(disc_env, "lambda m: Sigma(1, m, i)", seg_label="__sig2")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="sigma")
    assert res is not None
    for n in range(1, 21):
        expected = make(n * (n + 1) // 2, 1) if (n * (n + 1)) % 2 == 0 else make(n * (n + 1), 2)
        got = _run(disc_env, res.skeleton_ref, (n,))
        assert rational.eq(got, expected), f"sum_1_to_{n}={expected} got {got}（internal/PARAM 撞→静默错值）"


def test_nested_sigma_4_internal_sids(disc_env):
    """嵌套迭代回归测（设计 §三-bis 保留项）：Sigma(1,n,Sigma(1,n,i)) = n*(n(n+1)/2)·4 internal sids。

    嵌套 Sigma 有 4 internal sids（外 acc/idx + 内 acc/idx）·sorted 锁对应须保分配序（外先内后）·
    alpha-重命名到 make_variable(arity+0..3) 不撞 PARAM mv0。vm_proof 多 n 验值锁死（防 sorted 对应错位）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda n: Sigma(1, n, Sigma(1, n, i))", seg_label="__nest1")
    s2 = _build(disc_env, "lambda m: Sigma(1, m, Sigma(1, m, i))", seg_label="__nest2")
    assert probe_arity(b, [s1, s2]) == 1
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="nest")
    assert res is not None and res.arity == 1, "嵌套 Sigma discover arity=1"
    for n in (2, 3, 4, 5):
        expected = n * (n * (n + 1) // 2)
        got = _run(disc_env, res.skeleton_ref, (n,))
        assert rational.eq(got, make(expected, 1)), (
            f"嵌套 Sigma n={n}={expected} got {got}（4 internal sids sorted 对应错位→错值）")


# ============ (e) 异 body 不混同 ============

def test_different_body_not_aligned(disc_env):
    """(e) Sigma(1,n,i) vs Sigma(1,n,i*i) → None（body 位 OPERAND vs OPERATOR 异构→不混同）。

    反 theater 牙：不同 body（Σi vs Σi²）须分异组·不能强行对齐成同骨架（否则认 Σi 为 Σi²·vm_proof 错值）。
    """
    b, sid, es, ci, _ = disc_env
    a1 = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__body_i")
    a2 = _build(disc_env, "lambda n: Sigma(1, n, i*i)", seg_label="__body_ii")
    res = discover_skeleton([a1, a2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="mixbody")
    assert res is None, "异 body（Σi vs Σi²）须不混同→None（body 位 OPERAND vs MUL 异构）"


def test_prod_vs_sigma_not_aligned(disc_env):
    """Sigma vs Prod 同形（迭代 scaffolding 同·init/update opcode 异）→ None（不混同）。

    Sigma init=0/update=ADD·Prod init=1/update=MUL·init IMM 值异（0 vs 1）→ 骨架 PARAM 化该位
    但 update opcode ADD vs MUL 异构 → 无共性骨架。反 theater：Σ 与 Π 须分异。
    """
    b, sid, es, ci, _ = disc_env
    s = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__sig")
    p = _build(disc_env, "lambda n: Prod(1, n, i)", seg_label="__prod")
    res = discover_skeleton([s, p], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="sigprod")
    # update opcode ADD vs MUL 异构 → _NoSkeleton（init IMM 0 vs 1 也异·但 opcode 先判）
    assert res is None, "Sigma vs Prod（ADD vs MUL update）须不混同→None"


# ============ (f) held-out 识别③（discover 2 + recognize 第3 新参名） ============

def test_sigma_held_out_recognition(disc_env):
    """(f) discover 从 2 异参名 Sigma 学骨架 → recognize held-out 第3 新参名 Sigma（alpha-实例识别·非跨结构泛化）。

    ③识别 = READ 消费：骨架被 read_composes_tree 真读 + _align_walk 对齐新输入（含 ctrl/store 节点）→ operand_binding。
    held-out `lambda k: Sigma(1,k,i)`（k 非发现集）被识别 = **alpha-实例识别**（k/n/m 经 DSL 形参归一为同结构树·结构同构匹配）。
    **真泛化 = 参数化骨架对任意输入值**（n=1..20 验·测 (d)·骨架算 sum_1_to_N 对未见过 N 正确）·非"recognize 跨结构不同输入"
    （循环骨架不泛化成非循环闭式·设计·结构对齐本质）。审2 MED-1 诚实定位。
    """
    b, sid, es, ci, _ = disc_env
    d1 = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__learn1")
    d2 = _build(disc_env, "lambda m: Sigma(1, m, i)", seg_label="__learn2")
    held = _build(disc_env, "lambda k: Sigma(1, k, i)", seg_label="__held")
    ops = auto_discover_operators([d1, d2], concept_index=ci, edge_store=es,
                                  backend=b, space_id=sid, source=SOURCE_MATH)
    assert len(ops) >= 1 and ops[0].arity == 1, "auto_discover 须产 Sigma 算子 arity=1"
    recs = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)
    assert len(recs) >= 1, "held-out Sigma 须被识别（ctrl/store 骨架 READ 消费）"
    rec = recs[0]
    assert rec.is_operand_input, "Sigma input 含 OPERAND 叶（n=参数）→ operand-input 识别"
    # vm_proof 独立验：识别绑定执行 == held-out 执行（两路独立·反 theater）
    probe = rec.input_probe_values
    held_val = _run(disc_env, held, (9,))   # sum_1_to_9 = 45
    assert rational.eq(held_val, make(45, 1)), "held-out Sigma(9)=45"
    # 骨架绑探针执行（验证识别产物可消费）
    skel_probe_val = _run(disc_env, ops[0].skeleton_ref, (probe[0][0],))
    assert rational.eq(skel_probe_val, _run(disc_env, held, (probe[0][0],))), (
        "骨架绑探针 == held-out 绑同探针（识别绑定结构正确）")


# ============ (h) internal-LOAD 变量同一性牙（审2 MED-2 修·拒 Sigma(1,z,z) 假阳性） ============

def test_sigma_zz_body_not_recognized_as_sigma(disc_env):
    """(h) recognize 变量同一性牙：Sigma(1,z,z)（body=z=PARAM·语义 z²）**不**误识为 Sigma（=三角数）。

    审2 MED-2 修：原 _align_walk internal-LOAD 分支只查"input 有 OPERAND"→ z=PARAM 误过→假阳性（held(7)=49 vs skel(7)=28）。
    修后：skeleton internal LOAD 须对齐 input 对应 internal sid（sorted-position·acc/idx）·body=z=PARAM 非 internal→对齐失败→不识别。
    下游两路独立 vm_proof 兜底（formal_train _verify_generalization 滤值不等）**外**·recognize_operators 单函数亦拒（直调 caller 不吃假阳性）。
    """
    b, sid, es, ci, _ = disc_env
    d1 = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__learn1")
    d2 = _build(disc_env, "lambda m: Sigma(1, m, i)", seg_label="__learn2")
    zz = _build(disc_env, "lambda z: Sigma(1, z, z)", seg_label="__zz")  # body=z=PARAM·非 loop idx
    ops = auto_discover_operators([d1, d2], concept_index=ci, edge_store=es,
                                  backend=b, space_id=sid, source=SOURCE_MATH)
    assert len(ops) >= 1, "Sigma 发现须产算子"
    recs = recognize_operators([zz], discovered_operators=ops, backend=b, space_id=sid)
    assert recs == [], (
        "Sigma(1,z,z) body=z=PARAM·非 internal·须**不**误识为 Sigma（变量同一性牙·审2 MED-2·"
        f"got {[r.operator_name for r in recs]}）")
    # 反向确认：valid held-out Sigma(1,k,i) 仍识别（变量同一性牙不误伤合法输入·body=i=internal idx）
    held = _build(disc_env, "lambda k: Sigma(1, k, i)", seg_label="__held_ok")
    recs_ok = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)
    assert len(recs_ok) >= 1, "valid held-out Sigma(1,k,i) 须仍识别（变量同一性牙不误伤合法输入）"


# ============ (g) run_capability_exam ③计算 permille>0（核心反 theater） ============

def test_sigma_capability_exam_permille_gt0(tmp_path):
    """(g) Sigma(1,n,i) 异参名语料（≥3 同形）→ run_capability_exam ③计算 permille>0（核心反 theater）。

    扩前：discover 排除 ctrl/store → ③=0（现 MEMORY 记 S3 smoke ③=FAIL 0）。
    扩后：discover 产迭代骨架 + recognize held-out + vm_proof 验 → ③ permille>0。
    镜像 test_arith_s2_corpus 范式（load_arith_s2_corpus 风格 inline 自造语料）。
    """
    from pure_integer_ai.experiments.formal_train import FormalTrainConfig
    from pure_integer_ai.experiments.capability_exam import run_capability_exam
    from pure_integer_ai.experiments.collection import (
        CollectedItem, CodeSpec, MODALITY_ARITH, DOMAIN_MATH, LANG_NONE, SOURCE_MATH)

    # ≥3 同形异参名 Sigma（首2 发现·第3+ held-out 识别·真泛化）
    names = ["n", "m", "k", "t"]
    vals = [7, 10, 12, 20]
    corpus = []
    for nm, v in zip(names, vals):
        corpus.append(CollectedItem(
            modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE, source=SOURCE_MATH,
            arith_source=f"lambda {nm}: Sigma(1, {nm}, i)",
            arith_specs=(CodeSpec((v,), (v * (v + 1) // 2, 1)),)))
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id="s3_sigma_test", collect_episodes=True)
    report = run_capability_exam(cfg, corpus, backend=backend,
                                 training_mode=True, flat_floors=True)
    dim3 = report.to_json()["dimensions"]["③计算"]
    assert dim3["permille"] > 0, (
        f"Sigma ③ 泛化 permille={dim3['permille']}（须>0·ctrl/store 迭代 discover 未通→theater）")


def test_prod_capability_exam_permille_gt0(tmp_path):
    """(g) Prod(1,n,i) n! 异参名语料 → run_capability_exam ③计算 permille>0（Prod 迭代 builder 同证）。"""
    from pure_integer_ai.experiments.formal_train import FormalTrainConfig
    from pure_integer_ai.experiments.capability_exam import run_capability_exam
    from pure_integer_ai.experiments.collection import (
        CollectedItem, CodeSpec, MODALITY_ARITH, DOMAIN_MATH, LANG_NONE, SOURCE_MATH)

    import math
    names = ["n", "m", "k", "t"]
    vals = [3, 4, 5, 6]
    corpus = []
    for nm, v in zip(names, vals):
        corpus.append(CollectedItem(
            modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE, source=SOURCE_MATH,
            arith_source=f"lambda {nm}: Prod(1, {nm}, i)",
            arith_specs=(CodeSpec((v,), (math.factorial(v), 1)),)))
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id="s3_prod_test", collect_episodes=True)
    report = run_capability_exam(cfg, corpus, backend=backend,
                                 training_mode=True, flat_floors=True)
    dim3 = report.to_json()["dimensions"]["③计算"]
    assert dim3["permille"] > 0, (
        f"Prod ③ 泛化 permille={dim3['permille']}（须>0·Prod 迭代 discover 未通→theater）")
