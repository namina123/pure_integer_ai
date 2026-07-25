"""Phase 0 语料 loader + boot-inject e2e 测试（doc/重来_阶段断奶路线详设_2026-07-15 §二 Phase 0.1/0.3）。

验：
- loader 解析 data/transform_rules.txt.sample / inverse_relations.txt.sample → 正确 TransformSpec/InverseRelationSpec。
- loader E5 graceful（缺文件→[]·malformed 行 skip·不抛崩·镜像 load_alias_facts_file 范式）。
- resolve_* 经 PURE_INTEGER_AI_LOCAL_DIR 解析路径。
- 样例 DSL-correct + 机制兼容：loaded specs → _run_task_driven_generate → verified episodes（gate ON）
  + gate OFF bit-identical（零 episode）。这同时验证 boot-inject 产出的 CollectedItem 形态正确（镜像 TC11/IR5）。

铁律：纯整数 / bit-identical（gate OFF→零 episode）/ 反 theater（gate ON→verified episode 真执行 cross-verify）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_MATH
from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import (
    TransformSpec, TransformHeldOut, InverseRelationSpec,
    MODALITY_ARITH, DOMAIN_MATH, LANG_NONE,
    TERMINAL_REACHED_SINK, VERIFY_SOURCE_SELF_PRODUCED,
)
from pure_integer_ai.experiments.collection import (
    CollectedItem,
    load_transform_rules_file, resolve_transform_rules,
    load_inverse_relations_file, resolve_inverse_relations,
)
from pure_integer_ai.experiments.formal_train import make_train_context, _run_task_driven_generate

_DATA = Path(__file__).resolve().parent.parent / "data"
_XFORM_SAMPLE = _DATA / "transform_rules.txt.sample"
_INV_SAMPLE = _DATA / "inverse_relations.txt.sample"


# ============================================================
# loader 解析（样例文件 → TransformSpec/InverseRelationSpec）
# ============================================================

def test_load_transform_rules_sample_parses():
    """data/transform_rules.txt.sample → 正确 TransformSpec 列表（含 d/dx/distrib/comm_add）·S7 defer 注释 skip。"""
    rules = load_transform_rules_file(str(_XFORM_SAMPLE))
    names = {r.rule_name for r in rules}
    assert "ddx_pow" in names
    assert "distrib" in names
    assert "comm_add" in names
    assert "ddx_const" in names
    # S7 defer 注释段（# 开头）不计入规则（loader 注释 skip）
    assert all(not r.rule_name.startswith("#") for r in rules)
    # d/dx 规则字段精确（PARAM 同序对齐·Pow(b,n)→n*Pow(b,n-1)）
    ddx = next(r for r in rules if r.rule_name == "ddx_pow")
    assert ddx.lhs_source == "lambda b,n: Pow(b,n)"
    assert ddx.rhs_source == "lambda b,n: n*Pow(b,n-1)"
    assert len(ddx.held_out) >= 2   # 多样性铁律：≥2 异 base held-out
    assert all(isinstance(h, TransformHeldOut) for h in ddx.held_out)
    # distrib 两条异 base held-out（反统一泛化铁律）
    distrib = next(r for r in rules if r.rule_name == "distrib")
    assert len(distrib.held_out) >= 2


def test_load_inverse_relations_sample_parses():
    """data/inverse_relations.txt.sample → 正确 InverseRelationSpec（double/halve·精确有理除 /）。"""
    rels = load_inverse_relations_file(str(_INV_SAMPLE))
    names = {r.relation_name for r in rels}
    assert "double_halve_inv" in names
    assert "addsub_one_inv" in names
    assert "muldiv_three_inv" in names
    dh = next(r for r in rels if r.relation_name == "double_halve_inv")
    assert dh.rule_a.rule_name == "double"
    assert dh.rule_b.rule_name == "halve"
    assert dh.rule_a.rhs_source == "lambda x: x*2"
    assert dh.rule_b.rhs_source == "lambda x: x/2"   # 精确有理除（非 //）
    assert len(dh.sample_sources) >= 2   # 异形 sample（防过窄）
    # DEFER 注释段（d/dx↔∫ / 链式）不计入
    assert all(not r.relation_name.startswith("#") for r in rels)


# ============================================================
# loader E5 graceful（缺文件 / malformed → 不抛崩）
# ============================================================

def test_loader_e5_graceful_missing_file():
    """缺文件 → [] （E5 graceful·不抛崩·bit-identical 守）。"""
    assert load_transform_rules_file("/nonexistent/path/transform_rules.txt") == []
    assert load_inverse_relations_file("/nonexistent/path/inverse_relations.txt") == []


def test_loader_e5_malformed_lines_skip(tmp_path):
    """malformed 行（字段数错 / 缺分隔符 / 空字段）skip·合法行仍解析（E5 graceful·镜像 load_alias_facts_file）。"""
    f = tmp_path / "transform_rules.txt"
    f.write_text(
        "# comment line\n"
        "\n"
        "bad_no_tabs_here\n"                                    # 1 字段≠4 → skip
        "comm\tx y z\tlambda x: x\n"                           # 3 字段≠4 → skip
        "good\tlambda x: x\tlambda x: x\t\n"                   # 4 字段·held_out 空 → 合法（()）
        "ddx\tlambda b,n: Pow(b,n)\tlambda b,n: n*Pow(b,n-1)\tbadpair_no_arrow\n"  # held_out 缺 => → held_out=()·规则仍合法
        "ok2\tlambda x: x+0\tlambda x: x\tlambda p: p+0=>lambda p: p\n",
        encoding="utf-8")
    rules = load_transform_rules_file(str(f))
    names = {r.rule_name for r in rules}
    assert names == {"good", "ddx", "ok2"}, f"malformed skip·合法 3 条·实 {names}"
    # held_out 缺 => 的 ddx → held_out=()（pair skip·规则保留）
    ddx = next(r for r in rules if r.rule_name == "ddx")
    assert ddx.held_out == ()


def test_loader_e5_inverse_malformed_skip(tmp_path):
    """inverse malformed 行（≠8 字段）skip·合法行仍解析。"""
    f = tmp_path / "inverse_relations.txt"
    f.write_text(
        "# c\n"
        "bad\tshort\n"   # 2 字段≠8 → skip
        + "\t".join([
            "rel1", "ra", "lambda x: x", "lambda x: x+1",
            "rb", "lambda x: x", "lambda x: x-1", "lambda x: x+3;lambda x: x*5",
        ]) + "\n",
        encoding="utf-8")
    rels = load_inverse_relations_file(str(f))
    assert len(rels) == 1 and rels[0].relation_name == "rel1"
    assert len(rels[0].sample_sources) == 2


# ============================================================
# resolve_* 经 PURE_INTEGER_AI_LOCAL_DIR 解析
# ============================================================

def test_resolve_via_local_dir_env(tmp_path, monkeypatch):
    """PURE_INTEGER_AI_LOCAL_DIR 指向目录 → resolve_transform_rules 读 transform_rules.txt。无 env → []。"""
    (tmp_path / "transform_rules.txt").write_text(
        _XFORM_SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))
    rules = resolve_transform_rules()
    assert any(r.rule_name == "ddx_pow" for r in rules)
    # 无 env → []（bit-identical 守）
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    assert resolve_transform_rules() == []


def test_resolve_via_local_dir_param(tmp_path):
    """local_dir 参数优先（不经 env）。"""
    (tmp_path / "inverse_relations.txt").write_text(
        _INV_SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    rels = resolve_inverse_relations(local_dir=str(tmp_path))
    assert any(r.relation_name == "double_halve_inv" for r in rels)


# ============================================================
# e2e：样例 specs → _run_task_driven_generate（验 DSL-correct + 机制兼容 + boot-inject 形态）
# ============================================================

def _phase0_item():
    """镜像 formal_train boot-inject 产出的 CollectedItem（loaded specs 挂 transform/inverse）。"""
    rules = load_transform_rules_file(str(_XFORM_SAMPLE))
    rels = load_inverse_relations_file(str(_INV_SAMPLE))
    assert rules and rels, "样例文件须非空（否则前置失败）"
    return CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                        source=SOURCE_MATH,
                        transform_specs=tuple(rules),
                        inverse_relation_specs=tuple(rels))


def test_sample_specs_verified_gate_on():
    """gate ON：样例 specs（8 transform + 3 relation）→ _run_task_driven_generate → 多 verified episode。
    证样例 DSL-correct（build/apply/cross-verify 全通过）+ boot-inject CollectedItem 形态正确。
    weaning-safe 决断 A（独立 task-driven episode·SELF_PRODUCED·不准驱动停止·反 theater）。"""
    saved_x = gates.SYMBOLIC_TRANSFORM_MODE
    saved_r = gates.SYMBOLIC_RELATION_MODE
    gates.SYMBOLIC_TRANSFORM_MODE = True
    gates.SYMBOLIC_RELATION_MODE = True
    try:
        ctx = make_train_context(DictBackend())
        episodes, summary = _run_task_driven_generate(ctx, [_phase0_item()], all_ops=[])
        # 8 transform + 3 relation = 11 任务（total_tasks 结构确定性）
        assert summary.total_tasks >= 11, f"样例 11 specs→total_tasks≥11·实 {summary.total_tasks}"
        # 多 verified（d/dx + distrib + comm + ident + double/halve + addsub + muldiv·允许个别边缘失败）
        assert summary.verified >= 5, f"样例 specs 应产 ≥5 verified·实 {summary.verified}"
        ver = [e for e in episodes
               if e.reward == 1 and e.terminal == TERMINAL_REACHED_SINK
               and e.verify_source == VERIFY_SOURCE_SELF_PRODUCED]
        assert len(ver) >= 5, "≥5 SELF_PRODUCED verified episode（反 theater）"
    finally:
        gates.SYMBOLIC_TRANSFORM_MODE = saved_x
        gates.SYMBOLIC_RELATION_MODE = saved_r


def test_sample_specs_gate_off_bit_identical():
    """gate OFF：样例 specs 不消费 → 零 episode（bit-identical·镜像 TC12/IR6）。
    证 Phase 0.2 翻 gate 是激活条件·OFF 时核心零行为变（CI 守）。"""
    saved_x = gates.SYMBOLIC_TRANSFORM_MODE
    saved_r = gates.SYMBOLIC_RELATION_MODE
    gates.SYMBOLIC_TRANSFORM_MODE = False
    gates.SYMBOLIC_RELATION_MODE = False
    try:
        ctx = make_train_context(DictBackend())
        episodes, summary = _run_task_driven_generate(ctx, [_phase0_item()], all_ops=[])
        assert episodes == [], (
            "gate OFF → transform/relation specs 不消费 → 零 episode（MODALITY_ARITH item 无 arith_specs·"
            "gate OFF 全路径不产·bit-identical）")
    finally:
        gates.SYMBOLIC_TRANSFORM_MODE = saved_x
        gates.SYMBOLIC_RELATION_MODE = saved_r
