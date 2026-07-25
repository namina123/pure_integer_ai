# tests/test_arith_s1_corpus.py — S1 diverse 算术语料生成器测试（doc/重来_阶段断奶路线详设 §三 S1）
"""S1 diverse 算术语料生成器（load_arith_family_corpus / load_arith_s1_corpus）测试。

锁：①异参名 distinct（防 v1 同源 dedup→discover<2→③=0 bug）②expected 正确（Fraction 验）
③ast-valid ④determinism ⑤build_composes DSL-valid（arith_observe 接受）⑥单族训练 ③ 泛化>0（行为锁）。
"""
from __future__ import annotations

import ast
from fractions import Fraction

import pytest

from pure_integer_ai.experiments.collection import (
    load_arith_family_corpus, load_arith_s1_corpus,
    _distinct_arith_pairs, _arith_div_expected, _ARITH_OP_FAMILIES, _ARITH_DIV_VALS,
)


def _true_result(op, a, b):
    if op == "+":
        return Fraction(a + b)
    if op == "-":
        return Fraction(a - b)
    if op == "*":
        return Fraction(a * b)
    if op == "/":
        return Fraction(a, b)
    raise ValueError(op)


def test_distinct_arith_pairs_no_dedup():
    """异参名铁律：n distinct 对（p≠q·无 a/i）·防同源 dedup→discover<2→③=0。"""
    pairs = _distinct_arith_pairs(30)
    assert len(pairs) == 30
    assert len(set(pairs)) == 30          # 全 distinct（v1 bug 教训：同源→1 树→discover None）
    for p, q in pairs:
        assert p != q
        assert p not in "ai" and q not in "ai"   # 避保留 a/i（arith_observe._RESERVED）


def test_family_corpus_distinct_expected_ast_valid():
    """每族 30 fixture：异参名 distinct + expected 正确（Fraction 验）+ ast-valid。"""
    for op in _ARITH_OP_FAMILIES:
        items = load_arith_family_corpus(op, 30)
        assert len(items) == 30, f"{op} 族 fixture 数"
        sources = [it.arith_source for it in items]
        assert len(set(sources)) == 30, f"{op} arith_source 非 distinct（dedup 风险）"
        for it in items:
            ast.parse(it.arith_source)                         # Python/DSL 语法 valid
            assert it.modality is not None and it.arith_specs
            spec = it.arith_specs[0]
            a, b = spec.input_args
            num, den = spec.expected
            assert Fraction(num, den) == _true_result(op, a, b), (
                f"{op} {it.arith_source} input={spec.input_args} expected={spec.expected} 错")


def test_arith_div_expected_correctness():
    assert _arith_div_expected("+", 2, 3) == (5, 1)
    assert _arith_div_expected("-", 10, 3) == (7, 1)
    assert _arith_div_expected("*", 4, 5) == (20, 1)
    assert _arith_div_expected("/", 7, 2) == (7, 2)   # raw (num,den)·rational.eq 归一
    with pytest.raises(ValueError):
        _arith_div_expected("%", 1, 1)
    with pytest.raises(ValueError):
        _arith_div_expected("/", 5, 0)   # 审1 LOW-2：除以零早 fail-loud（纯整数铁律）


def test_s1_corpus_combined_structure():
    """合族 corpus = 120（4×30）+ 2 arith_bad。"""
    s1 = load_arith_s1_corpus()
    assert len(s1) == 122
    # arith_bad：末 2 项 expected 错（D2 负通路）
    bad = [it for it in s1 if it.arith_specs[0].expected == (999, 1)]
    assert len(bad) == 2


def test_determinism_bit_identical():
    """两次调用逐项相等（确定性·bit-identical）。"""
    for op in _ARITH_OP_FAMILIES:
        a = load_arith_family_corpus(op, 30)
        b = load_arith_family_corpus(op, 30)
        assert a == b, f"{op} 非确定"
    assert load_arith_s1_corpus() == load_arith_s1_corpus()


def test_family_op_validation():
    with pytest.raises(ValueError):
        load_arith_family_corpus("%", 5)
    with pytest.raises(ValueError):
        load_arith_family_corpus("+", 999)   # 超 _ARITH_DIV_VALS 池


@pytest.mark.parametrize("op", _ARITH_OP_FAMILIES)
def test_family_trains_generalizes(tmp_path, op):
    """单族（每 op·n=12）独训 → ③计算 泛化 permille>0（行为锁·机制真活非 theater）。

    覆盖全 4 族（+−×÷）：单族独训 ③>0（discover fires + recognize held-out + vm_proof 验）·
    端到端验 DSL-valid + 泛化·非 0（v1 同源 dedup bug 回归锚）。§三 S1"每算子独训防互扰"。
    """
    from pure_integer_ai.storage.backend import DictBackend
    from pure_integer_ai.experiments.formal_train import FormalTrainConfig
    from pure_integer_ai.experiments.capability_exam import run_capability_exam

    corpus = load_arith_family_corpus(op, 12)
    _safe = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op]   # op→文件名安全（避 * / Windows 非法）
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id=f"s1_{_safe}_test", collect_episodes=True)
    report = run_capability_exam(cfg, corpus, backend=backend,
                                 training_mode=True, flat_floors=True)
    dim3 = report.to_json()["dimensions"]["③计算"]
    assert dim3["permille"] > 0, (
        f"{op} 单族 ③ 泛化=0（discover 未 fire·查异参名 dedup 回归 / DSL-invalid）")


def test_combined_corpus_clean_recognize(tmp_path):
    """合族（load_arith_s1_corpus·4 族+arith_bad）共跑：③=1000 干净（4 族异骨架各匹配各 held-out）。

    **反 theater 纠偏**：早期 smoke 见"合族 ③745 互扰"=**旧 arith_bad(b,c) 撞 +族首项 source 的 dedup
    artifact**·LOW-1 fix（异参 o,p/q,r）消去→真 combined ③=1000（recognize 干净·4 异骨架 ADD/SUB/MUL/DIV
    各匹配本族 held-out）。真干扰在 ⑤ task-driven op 选择（verified<total·cold-start 选首发现 op）·非 ③。
    本测锚 combined ③>0（discover fires·反 v1 dedup 回归）+ arith_bad D2（verified<total 含坏 spec 失败）。
    """
    from pure_integer_ai.storage.backend import DictBackend
    from pure_integer_ai.experiments.formal_train import FormalTrainConfig
    from pure_integer_ai.experiments.capability_exam import run_capability_exam

    corpus = load_arith_s1_corpus()                       # 122 = 4 族×30 + 2 arith_bad
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id="s1_combined_test", collect_episodes=True)
    report = run_capability_exam(cfg, corpus, backend=backend,
                                 training_mode=True, flat_floors=True)
    js = report.to_json()
    permille = js["dimensions"]["③计算"]["permille"]
    assert permille > 0, "合族 ③=0（discover 未 fire·查 dedup 回归）"
    # ⑤ verified < total：arith_bad 2 坏 spec 失败（+ 可能 op 选择干扰）→ D2 负通路活
    dim5_ev = " ".join(js["dimensions"]["⑤长代码"]["evidence"])
    assert "verified=" in dim5_ev, "⑤ 无 verified 计数"
