"""S3 discover-迭代【学习】corpus 测试（doc/重来_S3S4迭代机制设计 §三-bis/§五·discover learn leg）。

S3 = 多异参名 examples/shape（distinct source string → distinct COMPOSES root）→ formal_train 内部
auto_discover_operators 抽迭代骨架 + recognize held-out + _verify_generalization vm_proof 验。
4 shape 覆盖**三迭代 builder**：Sigma（sum/sq）+ Prod（factorial）+ Recur（factorial）。

反 theater：片5 每 shape run_capability_exam ③ permille>0（discover 真学迭代骨架 + recognize held-out + vm_proof·非"建了=活"）。
诚实边界：held-out recognize = alpha-实例识别（同结构树）·真泛化 = 参数化骨架对任意输入值·③ permille 构造性（#479 墙内统计）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.collection import load_arith_s3_corpus, load_arith_s3_shape
from pure_integer_ai.config import gates

_SHAPES = ("sigma_sum", "sigma_sq", "prod_fact", "recur_fact")


# ============ 片1 corpus 结构 ============

def test_s3_corpus_structure():
    """load_arith_s3_corpus = 4 shape × 4 异参名 = 16 items·每 item lambda + specs（≥3 探针）。"""
    items = load_arith_s3_corpus()
    assert len(items) == 16, f"4 shape × 4 = 16·got {len(items)}"
    for it in items:
        assert it.arith_source.startswith("lambda "), f"须 lambda DSL: {it.arith_source!r}"
        assert it.arith_source_b is None, "S3 = learn leg·无 source_b（S4 才有）"
        assert len(it.arith_specs) >= 3, "每 item ≥3 探针"
        for spec in it.arith_specs:
            assert len(spec.input_args) == 1, "arity-1（单 param）"


def test_s3_shape_load_and_unknown():
    """load_arith_s3_shape 单 shape 返 n examples·未知 shape raise ValueError。"""
    for key in _SHAPES:
        items = load_arith_s3_shape(key, 4)
        assert len(items) == 4
    with pytest.raises(ValueError):
        load_arith_s3_shape("bogus", 4)


# ============ 片2 异参名铁律 + distinct source string ============

def test_s3_distinct_params_no_reserved():
    """异参名铁律：每 shape 4 examples 各异参名·无 reserved a/i。"""
    for key in _SHAPES:
        items = load_arith_s3_shape(key, 4)
        params = [it.arith_source.split("lambda", 1)[1].split(":", 1)[0].strip() for it in items]
        for p in params:
            assert p not in ("a", "i"), f"param 禁 reserved a/i: {p!r}"
        assert len(set(params)) == 4, f"{key}: 4 examples 须各异参名: {params}"


def test_s3_distinct_source_strings():
    """★ 防同源 dedup：每 shape 4 examples 须 distinct arith_source（→ distinct COMPOSES root → discover ≥2）。

    若 dedup 成 1 → discover <MIN_DISCOVER_SAMPLES → ③=0（v1 S1 square-only smoke bug 同根因）。
    """
    for key in _SHAPES:
        items = load_arith_s3_shape(key, 4)
        sources = [it.arith_source for it in items]
        assert len(set(sources)) == 4, f"{key}: 4 examples 须 distinct source（防同源 dedup）: {sources}"


# ============ 片3 期望值正确性（host 独立重算·反同源 theater） ============

def test_s3_expected_correct():
    """每 shape 探针 expected = 真值（host 独立重算·非依赖 _s4_* 以防同源 theater）。"""
    def real_sum(n): return sum(range(1, n + 1))
    def real_sq(n): return sum(k * k for k in range(1, n + 1))
    def real_fact(n):
        f = 1
        for k in range(2, n + 1):
            f *= k
        return f
    real = {"sigma_sum": real_sum, "sigma_sq": real_sq,
            "prod_fact": real_fact, "recur_fact": real_fact}
    for key in _SHAPES:
        items = load_arith_s3_shape(key, 4)
        fn = real[key]
        for it in items:
            for spec in it.arith_specs:
                n = spec.input_args[0]
                assert spec.expected == (fn(n), 1), \
                    f"{key} n={n}: expected={spec.expected} 真值={(fn(n), 1)}"


# ============ 片4 确定性 + bit-identical ============

def test_s3_corpus_deterministic():
    """load 两次完全相同（确定性 in-memory·纯数据·无随机）。"""
    a = load_arith_s3_corpus()
    b = load_arith_s3_corpus()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.arith_source == y.arith_source
        assert x.arith_specs == y.arith_specs
        assert x.arith_source_b is y.arith_source_b is None


def test_s3_bit_identical_defaults():
    """bit-identical 守：S3 item 显式 arith_source_b=None·gate 默认 OFF。"""
    for it in load_arith_s3_corpus():
        assert it.arith_source_b is None
    assert gates.MODE_B_CROSS_VERIFY_MODE is False, "gate 默认 OFF（S3 learn leg·不走 Mode B）"


# ============ 片5 ★反 theater 核心：每 shape run_capability_exam ③ permille>0 ============

@pytest.mark.parametrize("shape_key", _SHAPES)
def test_s3_per_shape_training_permille_gt0(shape_key, tmp_path):
    """★ 反 theater：每 shape 4 异参名 → run_capability_exam ③ permille>0（discover 真学迭代骨架 + recognize held-out）。

    生产 formal_train discover 路径：auto_discover_operators（首2 examples 抽骨架）+ recognize_operators（held-out 2 新参名）
    + _verify_generalization（vm_proof 任意输入值验）。③>0 = 真反 theater（扩前 S3 smoke ③=FAIL 0·ctrl/store 排除）。
    镜像 test_discover_iteration.test_sigma_capability_exam_permille_gt0·扩到 4 shape × 三迭代 builder。
    """
    from pure_integer_ai.storage.backend import DictBackend
    from pure_integer_ai.experiments.formal_train import FormalTrainConfig
    from pure_integer_ai.experiments.capability_exam import run_capability_exam

    corpus = load_arith_s3_shape(shape_key, 4)
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id=f"s3_{shape_key}_test", collect_episodes=True)
    report = run_capability_exam(cfg, corpus, backend=backend,
                                 training_mode=True, flat_floors=True)
    dim3 = report.to_json()["dimensions"]["③计算"]
    assert dim3["permille"] > 0, (
        f"{shape_key} ③ 泛化 permille={dim3['permille']}（须>0·discover 迭代未通→theater）")
