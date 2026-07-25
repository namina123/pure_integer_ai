# tests/test_math_measures.py — #1124 S5-S8 symbolic 度量（CapabilityReport.math_measures）测试
"""math_measures（additive observability·project_symbolic_measures）测试。

锁：①GenerateSummary xform_verified/inv_verified 默认 0（CI 无 symbolic specs bit-identical）
②project_symbolic_measures 读 result.generate（含 None 防御）③CapabilityReport.to_json 含 math_measures
④symbolic run（d/dx transform_spec）→ math_measures.xform_verified>0（反 theater：symbolic 学习可见非 invisible）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.formal_train import GenerateSummary, FormalTrainConfig
from pure_integer_ai.experiments.capability_exam import (
    CapabilityReport, project_symbolic_measures, run_capability_exam)
from pure_integer_ai.experiments.collection import (
    CollectedItem, TransformSpec, TransformHeldOut, InverseRelationSpec)
from pure_integer_ai.cognition.shared.types import MODALITY_ARITH, DOMAIN_MATH, LANG_NONE
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_MATH


def test_generate_summary_symbolic_defaults():
    """GenerateSummary xform_verified/inv_verified 默认 0（additive·CI 无 symbolic→bit-identical）。"""
    gs = GenerateSummary()
    assert gs.xform_verified == 0 and gs.inv_verified == 0
    gs2 = GenerateSummary(xform_verified=3, inv_verified=2)
    assert gs2.xform_verified == 3 and gs2.inv_verified == 2


def test_project_symbolic_measures_reads_generate():
    """project_symbolic_measures 读 result.generate 子计数 + 分母 + rate permille（M-1·None 防御）。"""
    class _R:
        generate = GenerateSummary(xform_verified=5, inv_verified=4, xform_total=6, inv_total=5)
    m = project_symbolic_measures(_R())
    assert m["xform_verified"] == 5 and m["xform_total"] == 6
    assert m["xform_rate_permille"] == 833   # 5*1000//6
    assert m["inv_verified"] == 4 and m["inv_total"] == 5
    assert m["inv_rate_permille"] == 800     # 4*1000//5

    class _RNone:
        generate = None
    m0 = project_symbolic_measures(_RNone())
    assert all(v == 0 for v in m0.values()), "CI 无 symbolic→全 0（bit-identical）"


def test_capability_report_to_json_has_math_measures():
    """CapabilityReport.to_json 含 math_measures（additive·不改既有 keys）。"""
    rep = CapabilityReport(run_id="t", math_measures={"xform_verified": 7, "inv_verified": 1})
    js = rep.to_json()
    assert js["math_measures"] == {"inv_verified": 1, "xform_verified": 7}
    assert "lang_measures" in js   # 既有 key 不破


def _ddx_item():
    """CollectedItem + d/dx transform_spec（镜像 test_symbolic_transform._ddx_item·2 held-out 验证对）。"""
    spec = TransformSpec(
        rule_name="ddx_pow_mm",
        lhs_source="lambda b,n: Pow(b,n)",
        rhs_source="lambda b,n: n * Pow(b, n-1)",
        held_out=(TransformHeldOut("lambda x: Pow(x,2)", "lambda x: 2*x"),
                  TransformHeldOut("lambda y: Pow(y,3)", "lambda y: 3*y*y")),
    )
    return CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                         source=SOURCE_MATH, transform_specs=(spec,))


def test_symbolic_run_populates_math_measures(tmp_path):
    """symbolic run（d/dx transform_spec）→ math_measures.xform_verified>0（反 theater·学习可见）。

    run_capability_exam → formal_train 生产 try/finally 翻 SYMBOLIC_TRANSFORM_MODE ON → transform block
    register+apply+cross-verify → xform_verified+=1 → project_symbolic_measures → math_measures。
    """
    import os
    os.environ.setdefault("PYTHONHASHSEED", "0")
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id="mm_ddx_test", collect_episodes=True)
    report = run_capability_exam(cfg, [_ddx_item()], backend=backend,
                                 training_mode=True, flat_floors=True)
    js = report.to_json()
    assert js["math_measures"]["xform_verified"] > 0, (
        "symbolic transform verified=0（math_measures 未 fire·查 SYMBOLIC_TRANSFORM_MODE flip / cross-verify）")


def _inv_item():
    """CollectedItem + double/halve inverse_relation_spec（镜像 test_symbolic_relation._inv_item·M-3 inv 端到端）。"""
    spec = InverseRelationSpec(
        relation_name="dbl_hlv_mm",
        rule_a=TransformSpec("dbl_mm", "lambda p: p", "lambda p: 2*p"),
        rule_b=TransformSpec("hlv_mm", "lambda p: 2*p", "lambda p: p"),
        sample_sources=("lambda x: x+3", "lambda x: x*5", "lambda x: x-7"))
    return CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                         source=SOURCE_MATH, inverse_relation_specs=(spec,))


def test_inverse_run_populates_math_measures(tmp_path):
    """symbolic run（double/halve inverse_relation_spec）→ math_measures.inv_verified>0（M-3 inv 端到端）。

    补审2 M-3 缺口：inv_verified 的 formal_train→GenerateSummary→project_symbolic_measures→
    CapabilityReport.math_measures 链端到端验（_ddx_item 仅 transform·inv 链原零覆盖）。
    """
    import os
    os.environ.setdefault("PYTHONHASHSEED", "0")
    backend = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path), run_id="mm_inv_test", collect_episodes=True)
    report = run_capability_exam(cfg, [_inv_item()], backend=backend,
                                 training_mode=True, flat_floors=True)
    js = report.to_json()
    assert js["math_measures"]["inv_verified"] > 0, (
        "symbolic inverse verified=0（inv_verified 未 fire·查 SYMBOLIC_RELATION_MODE flip / B∘A 还原）")
