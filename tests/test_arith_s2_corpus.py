# tests/test_arith_s2_corpus.py — S2 复合表达式语料生成器测试（doc §三 S2）
"""S2 复合表达式 corpus（load_arith_s2_corpus）测试·镜像 test_arith_s1_corpus 范式。

锁：①异参名 distinct triple（防同源 dedup）②expected 正确（Fraction 验）③ast-valid ④determinism
⑤per-shape 训练 ③ 泛化>0（discover 处理嵌套 BinOp·smoke 已验 addmul/submul ③=1000）。
"""
from __future__ import annotations

import ast
from fractions import Fraction

import pytest

from pure_integer_ai.experiments.collection import (
    load_arith_s2_corpus, _ARITH_S2_SHAPES, _distinct_arith_triples, _ARITH_S2_VALS)


def test_distinct_arith_triples():
    """异参名铁律：n distinct ordered (p,q,r)·p/q/r 全异·无 a/i。"""
    triples = _distinct_arith_triples(7)
    assert len(triples) == 7
    assert len(set(triples)) == 7
    for p, q, r in triples:
        assert len({p, q, r}) == 3          # 全异
        assert not ({p, q, r} & {"a", "i"})  # 避保留 a/i


def test_s2_corpus_distinct_expected_ast_valid():
    """6 shape ×7：异参名 distinct + expected 正确 + ast-valid。"""
    items = load_arith_s2_corpus()
    assert len(items) == len(_ARITH_S2_SHAPES) * 7
    sources = [it.arith_source for it in items]
    assert len(set(sources)) == len(items), "arith_source 非 distinct（dedup 风险）"
    for it in items:
        ast.parse(it.arith_source)
        spec = it.arith_specs[0]
        a, b, c = spec.input_args
        # expected 正确性：exec source（DSL=合法 Python 子集）vs expected (num,den)
        fn = eval(it.arith_source)   # lambda（S2 仅 +−* 复合·eval 安全确定）
        assert Fraction(fn(a, b, c)) == Fraction(spec.expected[0], spec.expected[1]), (
            f"{it.arith_source} {(a,b,c)} expected={spec.expected} 错")


def test_s2_determinism():
    """两次调用逐项相等（确定性·bit-identical）。"""
    assert load_arith_s2_corpus() == load_arith_s2_corpus()


def test_s2_op_validation():
    with pytest.raises(ValueError):
        load_arith_s2_corpus(per_shape=999)   # 超 triple 池 / _ARITH_S2_VALS


@pytest.mark.parametrize("shape_idx", range(len(_ARITH_S2_SHAPES)))
def test_s2_shape_trains_generalizes(tmp_path, shape_idx):
    """per-shape（每 compound shape·n=7）独训 → ③计算 泛化 permille>0（discover 处理嵌套 BinOp）。

    smoke 已验 addmul/submul ③=1000·本测 parametrize 全 6 shape 锁行为。
    """
    from pure_integer_ai.storage.backend import DictBackend
    from pure_integer_ai.experiments.formal_train import FormalTrainConfig
    from pure_integer_ai.experiments.capability_exam import run_capability_exam
    from pure_integer_ai.experiments.collection import CollectedItem, CodeSpec, MODALITY_ARITH, DOMAIN_MATH, LANG_NONE, SOURCE_MATH

    name, tmpl, exp_fn = _ARITH_S2_SHAPES[shape_idx]
    triples = _distinct_arith_triples(7)
    corpus = []
    for (p, q, r), (a, b, c) in zip(triples, _ARITH_S2_VALS[:7]):
        corpus.append(CollectedItem(
            modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE, source=SOURCE_MATH,
            arith_source=f"lambda {p},{q},{r}: {tmpl(p, q, r)}",
            arith_specs=(CodeSpec((a, b, c), exp_fn(a, b, c)),)))
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id=f"s2_{name}_test", collect_episodes=True)
    report = run_capability_exam(cfg, corpus, backend=backend,
                                 training_mode=True, flat_floors=True)
    dim3 = report.to_json()["dimensions"]["③计算"]
    assert dim3["permille"] > 0, (
        f"{name} shape ③ 泛化=0（discover 未处理嵌套 BinOp·查 dedup / DSL-invalid）")
