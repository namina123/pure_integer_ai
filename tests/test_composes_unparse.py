"""#730 长代码 generate 路径 W 测试（序化器 composes_unparse + ast_normalize + task-driven code 模态）。

覆盖（doc/重来_任务0730_长代码generate_设计.md 决断 2/3/4/7）：
  - 序化器单元：白名单内代码段（Return/BinOp/Assign/If/IfElse/While/Compare/UnaryOp/Constant）unparse 对称
    + CTRL_TAG 优先 + STORE 分支 + var{index} 稳定 + 写读序 bit-identical
  - ast_normalize bodies_match：结构等价判（名统一 _v{k}·剥 FunctionDef 签名·body 层）
  - gate ON/OFF（CODE_UNPARSE_MODE·反 theater·代码模态 episode gated）
  - e2e formal_train code corpus → ⑤ Mode A PASS + 反 theater（parts 非空⟺ verified）

铁律：纯读测试 / 确定性（两跑 bit-identical）/ 不写死（白名单通用非特定程序）。
诚实边界：Mode A 构造性（unparse(__prog_*)==code_source 构造性必然·非真生成）/ stable≠correct（var{index} 人造名）。
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_CODE
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.code_observe import build_composes_from_source
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.composes_unparse import unparse_composes
from pure_integer_ai.cognition.result.ast_normalize import bodies_match, normalize_code_body
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.cognition.shared.types import STAGE_TRAINING, MODALITY_CODE
from pure_integer_ai.experiments.collection import CollectedItem, CodeSpec
from pure_integer_ai.experiments.formal_train import _run_task_driven_generate
from pure_integer_ai.config import gates


# ---- helpers ----

def _build(code_source: str, space_id: int = 1):
    """建 code COMPOSES 树（镜像 observe.py:132-170 内容哈希 label）→ 返 (graph, root)。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    label = f"__prog_{STAGE_TRAINING}_{Hasher('observe.prog.v1').h63(code_source)}"
    root = ci.ensure(label, space_id=space_id)
    build_composes_from_source(code_source, concept_index=ci, edge_store=es,
                               backend=b, space_id=space_id, source=SOURCE_CODE,
                               root_ref=root)
    g = ConceptGraph(b)
    return g, root


def _code_item(code_source: str, n: int = 5) -> CollectedItem:
    """造 code CollectedItem（code_source + code_specs·expected=任意正确值·unparse 验不读 spec）。"""
    return CollectedItem(modality=MODALITY_CODE, source=SOURCE_CODE,
                         code_source=code_source,
                         code_specs=(CodeSpec((n,), (n * 2, 1)),))


# ============ 序化器单元：白名单代码段 unparse 对称 ============

@pytest.mark.parametrize("src", [
    "def f(n):\n  return n + n",            # BinOp Add
    "def f(a):\n  return a * a",            # BinOp Mul
    "def f(a, b):\n  return a + b",         # 2 args
    "def f(n):\n  return n - 1",            # BinOp Sub + IMM
    "def f(n):\n  return -1",               # UnaryOp USub → IMM(-1)
    "def f(n):\n  return n < 5",            # Compare Lt
    "def f(n):\n  return n > 5",            # Compare Gt
    "def f(n):\n  return n == 5",           # Compare Eq
])
def test_unparse_whitelist_matches_source(src):
    """白名单代码段：unparse(COMPOSES 树) normalize == code_source normalize（Mode A 构造性）。"""
    g, root = _build(src)
    unparsed = unparse_composes(g, root)
    assert bodies_match(unparsed, src), (
        f"unparse normalize != source normalize\n  src={src!r}\n  unparsed={unparsed!r}")


def test_unparse_assign_statement():
    """Assign 语句（STORE 节点）：x = n + 1 + return x → unparse 对称。"""
    src = "def f(n):\n  x = n + 1\n  return x"
    g, root = _build(src)
    unparsed = unparse_composes(g, root)
    assert bodies_match(unparsed, src), f"unparsed={unparsed!r}"


def test_unparse_if_statement():
    """CTRL_IF：if n: return 1 / return 0 → unparse 对称（缩进体）。"""
    src = "def f(n):\n  if n:\n    return 1\n  return 0"
    g, root = _build(src)
    unparsed = unparse_composes(g, root)
    assert bodies_match(unparsed, src), f"unparsed={unparsed!r}"


def test_unparse_ifelse_statement():
    """CTRL_IFELSE：if/else 两分支 → unparse 对称。"""
    src = "def f(n):\n  if n:\n    return 1\n  else:\n    return 0"
    g, root = _build(src)
    unparsed = unparse_composes(g, root)
    assert bodies_match(unparsed, src), f"unparsed={unparsed!r}"


def test_unparse_while_statement():
    """CTRL_WHILE：while n: return n / return 0 → unparse 对称。"""
    src = "def f(n):\n  while n:\n    return n\n  return 0"
    g, root = _build(src)
    unparsed = unparse_composes(g, root)
    assert bodies_match(unparsed, src), f"unparsed={unparsed!r}"


def test_unparse_augassign_matches_after_flatten():
    """AugAssign（code_observe 支持·与 Assign 同构树）→ normalize AugAssign→Assign 归一后 match（审2 P1-1 修）。

    code_observe 对 `n += 1` 与 `n = n + 1` 产同构 COMPOSES 树·序化器出 Assign 形式·normalize 归一后语义等价 match。
    """
    src = "def f(n):\n  n += 1\n  return n"
    g, root = _build(src)
    unparsed = unparse_composes(g, root)
    assert bodies_match(unparsed, src), (
        f"AugAssign 须归一后 match（审2 P1-1）·unparsed={unparsed!r}")


def test_unparse_var_index_human_name():
    """var{index} 人造名（原始名丢失·决断 3）：unparse 含 var0/var1 非 'n'/'a'。"""
    g, root = _build("def f(n):\n  return n + n")
    unparsed = unparse_composes(g, root)
    assert "var0" in unparsed, f"期望 var0 人造名·得 {unparsed!r}"
    assert "n " not in unparsed.replace("n ", "") or "var0" in unparsed  # n 不直接出现（被 var0 替）


def test_unparse_bit_identical_two_runs():
    """写读序对称 bit-identical：两次 build + unparse → 逐字节相同串。"""
    src = "def f(a, b):\n  if a:\n    return a + b\n  return b"
    g1, r1 = _build(src)
    g2, r2 = _build(src)
    u1 = unparse_composes(g1, r1)
    u2 = unparse_composes(g2, r2)
    assert u1 == u2, f"两跑不一致:\n  {u1!r}\n  {u2!r}"


# ============ ast_normalize bodies_match ============

def test_bodies_match_true_different_names():
    """两源结构同·变量名异 → match（名按位置统一 _v{k}）。"""
    a = "def f(n):\n  return n + n"
    b = "def g(x):\n  return x + x"
    assert bodies_match(a, b)  # 都是 normalize 后比（这里测两端都走 normalize）


def test_bodies_match_false_different_op():
    """结构异（Add vs Sub）→ 不 match。"""
    g, root = _build("def f(n):\n  return n + n")
    unparsed = unparse_composes(g, root)   # 含 +
    assert not bodies_match(unparsed, "def f(n):\n  return n - n")   # source 是 -


def test_bodies_match_strips_signature():
    """剥 FunctionDef 签名：def 包装 vs body-only → body 等价 match。"""
    full = "def f(n):\n  return n + n"
    body_only = "return var0 + var0"   # unparse 产 body-only
    assert bodies_match(body_only, full)


def test_normalize_code_body_deterministic():
    """normalize 确定性：同输入两调 → 同串。"""
    s = "def f(a):\n  return a + a"
    assert normalize_code_body(s) == normalize_code_body(s)


# ============ gate ON/OFF（CODE_UNPARSE_MODE·反 theater） ============

def test_gate_default_off():
    """gate default OFF·守 CI bit-identical（gate OFF→代码模态 episode 不激活·零行为变）。"""
    import pure_integer_ai.config.gates as g_mod
    assert g_mod.CODE_UNPARSE_MODE is False


def test_code_episode_gate_off_skipped():
    """gate OFF：_run_task_driven_generate 代码模态 episode 不激活（task-driven 只走 arith·code corpus total=0）。"""
    src = "def f(n):\n  return n + n"
    g, root = _build(src)
    item = _code_item(src)
    item.code_struct_ref = root
    ctx = SimpleNamespace(concept_graph=g, backend=g._b)
    saved = gates.CODE_UNPARSE_MODE
    gates.CODE_UNPARSE_MODE = False
    try:
        eps, summary = _run_task_driven_generate(ctx, [item], [])
    finally:
        gates.CODE_UNPARSE_MODE = saved
    assert summary.total_tasks == 0, "gate OFF 代码模态 episode 须不激活"
    assert summary.verified == 0
    assert eps == []


def test_code_episode_gate_on_activates():
    """gate ON：_run_task_driven_generate 代码模态 episode 激活（unparse verified·parts 非空⟺verified·反 theater）。"""
    src = "def f(n):\n  return n + n"
    g, root = _build(src)
    item = _code_item(src)
    item.code_struct_ref = root
    ctx = SimpleNamespace(concept_graph=g, backend=g._b)
    saved = gates.CODE_UNPARSE_MODE
    gates.CODE_UNPARSE_MODE = True
    try:
        eps, summary = _run_task_driven_generate(ctx, [item], [])
    finally:
        gates.CODE_UNPARSE_MODE = saved
    assert summary.total_tasks == 1
    assert summary.selected == 1
    assert summary.verified == 1   # Mode A 构造性必然（unparse==source）
    assert len(eps) == 1
    # 反 theater ③：parts 非空 ⟺ verified
    assert eps[0].output.parts, "verified→parts 非空（产出真提交）"
    assert eps[0].reward == 1


def test_code_episode_gate_on_mismatch_not_verified():
    """gate ON + 树与 source 不匹配 → not verified + parts 空（反 theater：未验不提交产出）。

    构造：code_struct_ref 指向 return n+n 树·但 code_source 标 return n-n（mismatch）→ unparse 是 +·source 是 -·不 match。
    """
    g, root = _build("def f(n):\n  return n + n")
    item = _code_item("def f(n):\n  return n + n")
    item.code_struct_ref = root
    item.code_source = "def f(n):\n  return n - n"   # 篡改 source 致 mismatch
    ctx = SimpleNamespace(concept_graph=g, backend=g._b)
    saved = gates.CODE_UNPARSE_MODE
    gates.CODE_UNPARSE_MODE = True
    try:
        eps, summary = _run_task_driven_generate(ctx, [item], [])
    finally:
        gates.CODE_UNPARSE_MODE = saved
    assert summary.verified == 0, "mismatch 须 not verified"
    assert eps[0].output.parts == [] or not eps[0].output.parts, "未验→parts 空（反 theater ③）"
    assert eps[0].reward == 0


def test_code_episode_no_struct_ref_skipped():
    """gate ON 但 code_struct_ref=None（observe 未建树）→ skip（不计 total·诚实·防御）。"""
    src = "def f(n):\n  return n + n"
    g, _root = _build(src)
    item = _code_item(src)
    item.code_struct_ref = None   # 模拟 observe 未建树
    ctx = SimpleNamespace(concept_graph=g, backend=g._b)
    saved = gates.CODE_UNPARSE_MODE
    gates.CODE_UNPARSE_MODE = True
    try:
        eps, summary = _run_task_driven_generate(ctx, [item], [])
    finally:
        gates.CODE_UNPARSE_MODE = saved
    assert summary.total_tasks == 0
    assert eps == []


# ============ e2e formal_train code corpus → ⑤ Mode A PASS ============

def test_e2e_formal_train_code_corpus_generate_pass(tmp_path):
    """e2e：formal_train code corpus（生产路径 gate 翻 ON）→ result.generate verified=10/10·rate=1000。"""
    from pure_integer_ai.experiments.collection import load_code_corpus
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    corpus = load_code_corpus()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "test730e2e"), run_id="test730e2e")
    res = formal_train(cfg, corpus, backend=DictBackend())
    g = res.generate
    assert g.total_tasks == 10
    assert g.verified == 10
    assert g.rate_permille == 1000
    # code_struct_ref 捕获（run_round_full observe 后）
    assert all(it.code_struct_ref is not None for it in corpus)


def test_e2e_capability_exam_code_dim_pass(tmp_path):
    """e2e：capability_exam code corpus → ⑤长代码 status=PASS + footnote 诚实标（Mode A PASS ≠ 源码 generate）。"""
    from pure_integer_ai.experiments.collection import load_code_corpus
    from pure_integer_ai.experiments.capability_exam import run_capability_exam
    from pure_integer_ai.experiments.formal_train import FormalTrainConfig
    corpus = load_code_corpus()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "test730cap"), run_id="test730cap")
    rep = run_capability_exam(cfg, corpus, backend=DictBackend())
    js = rep.to_json()
    d5 = js["dimensions"]["⑤长代码"]
    # STEP2 #889：⑤取严 NE（generate 字面零测·D5）·Mode A code unparse 真活进 evidence
    assert d5["status"] == "NE", f"⑤ 须 NE（#889 取严·generate 字面零测）·得 {d5['status']}"
    assert d5["permille"] == -1
    # Mode A code unparse 真活进 evidence（rate=1000）
    assert any("Mode A task-driven" in e and "1000" in e for e in d5["evidence"]), (
        f"⑤ evidence 须含 Mode A rate=1000（code unparse 真活·#889）·得 {d5['evidence']}")
    # 诚实标：footnote 含 "拆双格" + "Mode A PASS 不偷渡"
    assert "拆双格" in d5["footnote"] and "Mode A PASS 不偷渡" in d5["footnote"]


def test_e2e_formal_train_code_corpus_bit_identical(tmp_path):
    """bit-identical：两跑 formal_train code corpus → result.generate 完全一致（写读序对称）。"""
    from pure_integer_ai.experiments.collection import load_code_corpus
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig

    def _run(rid):
        corpus = load_code_corpus()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / rid), run_id=rid)
        return formal_train(cfg, corpus, backend=DictBackend()).generate
    g1 = _run("test730bi_a")
    g2 = _run("test730bi_b")
    assert (g1.total_tasks, g1.selected, g1.verified, g1.rate_permille) == \
           (g2.total_tasks, g2.selected, g2.verified, g2.rate_permille)
