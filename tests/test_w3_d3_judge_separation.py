"""tests/test_w3_d3_judge_separation — W3 D3 独立裁判分离（工程闸门·墙内）。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W3）：
1. e2e：caller 传独立 judge_source_id（99·与 teacher.source_id=42 不相交）→ build_judge_fn :463
   算 sources_disjoint({99},{42})=True 设 ctx.judge_source_independent → stage4 路径 B :2018 读 ctx
   → weaning_check D3 过（连通死属性·单一真相源·补 test_weaning_gates:191 盲区）
2. judge_source_independent_arith 算术域判定接口（vm_proof 自锚·架构保证·非 sources_disjoint）
3. 默认 judge_source_id=None → D3 False（bit-identical·既有测零翻）
4. 算术域 vm_proof 自锚非教师本尊（反同源偷渡·风险②·teacher=None judge_fn 不构建）

**W3 = 通用机制修正（路径 B 死属性 + ctx track + caller 传独立 sid）+ 算术域 D3 判定接口**。
general-purpose agent 核证发现两条断开 D3 路径：路径 A（stages.py:163 挂 judge_fn.judge_source_independent
死属性）vs 路径 B（formal_train.py:2018 硬编 {teacher_sid},{teacher_sid} 永远 False·weaning_check 实读）。
W3 ctx track 连通（caller :463 设 ctx.judge_source_independent·路径 B 读 ctx）·单一真相源。

**诚实边界**：W3 通用机制让 D3 通用路径能过（caller 传独立 sid）·但算术域 run（teacher=None）D3 仍 False
（通用路径同源）·W7 才接算术域 D3 全（路径 B 读 judge_source_independent_arith）。self_proof_fn 独立 GT
来源 defer W8（语言域独立 LLM 录制·#731）·W3 不改 self_proof_fn 来源（stages.py:158 绑 teacher.judge_ground_truth 保留）。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.teacher.recordable_teacher import (
    RecordableLLMTeacher, register_recording_table,
    MODE_RECORD, CONTENT_META_DEFINITION, KIND_DEFINE, GT_PASS,
)
from pure_integer_ai.teacher.weaning import weaning_check, WeaningMetrics
from pure_integer_ai.teacher.source_independence import judge_source_independent_arith
from pure_integer_ai.training import stages
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.formal_train import (
    formal_train, FormalTrainConfig, DefaultRoundRunner,
)
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES
from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith
from tests.boundary_fixtures import attach_boundary_fixture


# ---- helpers（镜像 test_experiments._corpus/_teacher·自造免跨测试 import） ----

def _corpus(n=4):
    """多句段语言域语料（句末标点切 ≥2 段→obs.struct_refs≥2→走 :465 build_judge_fn 非 :446 早返）。

    单段 bare-text（role_seq=[1,2,3]）只产 1 struct_ref→:446 len<2 早返→:465 build_judge_fn 不执行
    →ctx.judge_source_independent 保持默认 False→路径 B 读 False→D3 blocker（首版 Fail 根因）。
    多句段（句末。切 ≥2 段）→struct_refs≥2→:465 执行设 ctx.judge_source_independent→D3 过。
    镜像 test_experiments._multi_sent_item 范式（端到端 reward>0 门用多句段）。
    """
    return [attach_boundary_fixture(CollectedItem(
                          tokens=[f"a{i}", f"b{i}。", f"c{i}", f"d{i}。"],
                          role_seq=[1, 1, 1, 1],
                          collect_type=COLLECT_PRECEDES,
                          source=SOURCE_BARE_TEXT),
                          cut_after=(2,))
            for i in range(n)]


def _teacher_with_source(backend, source_id):
    """假教师（录放层·确定性·带 source_id·D3 比对用）。"""
    def llm_call(kind, args):
        if kind == KIND_DEFINE:
            return {"kind": KIND_DEFINE, "content_type": CONTENT_META_DEFINITION,
                    "text": f"def_{args[2]}", "response_int": 0}
        return {"kind": kind, "content_type": CONTENT_META_DEFINITION,
                "text": None, "response_int": GT_PASS}
    return RecordableLLMTeacher(backend, mode=MODE_RECORD, llm_call=llm_call,
                                source_id=source_id)


def _m(rounds, *, cond=0, real=0, judge=0, oov=0, interv=0, reten=0, dep=0):
    return WeaningMetrics(rounds=rounds, conduction_rate=cond, realizes_rate=real,
                          judge_self_rate=judge, oov_promote_rate=oov,
                          intervention_rate=interv, holdout_retention=reten,
                          dependency=dep)


def _gates_true():
    return dict(neg_pathway_active=True, judge_source_independent=True,
                probe_set_disjoint=True, mode_b_prevalidated=True, e2_passed=True)


def test_w3_independent_judge_source_passes_d3_e2e(tmp_path):
    """★e2e：caller 传独立 judge_source_id=99（teacher source_id=42 不相交）→ D3 通用路径过。

    复用 H2 范式（test_experiments:1199·DefaultRoundRunner + _corpus(4) bare-text + TEACHER_MODE ON
    + teacher 在位 + flat_floors）。_corpus(4) 非 verify modality → 走 :463 judge_fn（非 :374 早返）
    → build_judge_fn 算 sources_disjoint({99},{42})=True 设 ctx.judge_source_independent →
    stage4 路径 B :2018 读 ctx → weaning_check D3 过。补 test_weaning_gates:191 盲区（死属性→端到端）。
    weaning_ready 仍 False（D4/D5/E2 defer·只 D3 单闸门过）。
    """
    b = DictBackend()
    bootstrap(b)
    register_recording_table(b)
    saved_t = gates.TEACHER_MODE
    saved_train = gates.TRAINING_MODE
    saved_floors = (stages.FLOOR_GRAPH_SIZE_S1, stages.FLOOR_CAUSES_COV_S2,
                    stages.FLOOR_CONDUCTION_S3, stages.FLOOR_PROMOTE_S4)
    gates.TEACHER_MODE = True
    gates.TRAINING_MODE = True
    try:
        stages.FLOOR_GRAPH_SIZE_S1 = 0
        stages.FLOOR_CAUSES_COV_S2 = 0
        stages.FLOOR_CONDUCTION_S3 = 0
        stages.FLOOR_PROMOTE_S4 = 0
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "w3e2e"), run_id="w3e2e",
                                rounds_per_stage=2, judge_source_id=99)
        teacher = _teacher_with_source(b, source_id=42)
        result = formal_train(cfg, _corpus(4), backend=b,
                              runner=DefaultRoundRunner(), teacher=teacher)
        # D3 通用路径过（99 与 42 不相交→sources_disjoint True→ctx.judge_source_independent True→路径 B 读 True）
        assert "D3_judge_source_independent" not in result.weaning_blockers, (
            f"caller 传独立 judge_source_id=99（teacher source_id=42 不相交）须 D3 通用路径过·"
            f"blockers={result.weaning_blockers}")
        # weaning_ready 仍 False（D4/D5/E2 defer·只 D3 单闸门过·诚实非真断奶）
        assert not result.weaning_ready
    finally:
        gates.TEACHER_MODE = saved_t
        gates.TRAINING_MODE = saved_train
        (stages.FLOOR_GRAPH_SIZE_S1, stages.FLOOR_CAUSES_COV_S2,
         stages.FLOOR_CONDUCTION_S3, stages.FLOOR_PROMOTE_S4) = saved_floors


def test_w3_judge_source_independent_arith_true():
    """★算术域 D3 判定接口（vm_proof 自锚·架构保证·非 sources_disjoint）。

    判据：verify_uses_vm_proof（算术域 verify 走 vm_proof·VM 执行值自锚非教师 GT）+
    teacher_not_judge（算术域绕 judge·_run_verify_round:374 早返·teacher=None judge_fn 不构建）。
    算术域裁判源=VM 执行值（R6 外部锚·非教师 source_id）→ 天然独立。
    W3 只建判定 + 算术域能 True·weaning_check D3 仍 False（通用路径 teacher=None 同源）·W7 才接全。
    """
    # 算术域 D3 就位：vm_proof 自锚 + 绕 judge → True
    assert judge_source_independent_arith(verify_uses_vm_proof=True,
                                          teacher_not_judge=True) is True
    # 反 theater：vm_proof 未用 / teacher 在 judge 位 → False
    assert judge_source_independent_arith(verify_uses_vm_proof=False,
                                          teacher_not_judge=True) is False
    assert judge_source_independent_arith(verify_uses_vm_proof=True,
                                          teacher_not_judge=False) is False
    # 算术域 weaning_check D3 仍 False（通用路径 teacher=None 同源·W7 才接 judge_source_independent_arith）
    hist = [_m(1, cond=600, real=400, judge=600, oov=100, interv=300, reten=800, dep=100),
            _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=750, dep=100),
            _m(3, cond=600, real=400, judge=600, oov=100, interv=200, reten=720, dep=100),
            _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100)]
    gates_kw = _gates_true()
    gates_kw["judge_source_independent"] = False   # 算术域通用路径仍 False（teacher=None 同源）
    rep = weaning_check(hist, **gates_kw)
    assert rep.ready is False   # D3 通用路径挡·W7 才接算术域判定


def test_w3_gate_off_bit_identical(tmp_path):
    """★默认 judge_source_id=None → D3 False（bit-identical·既有测零翻）。

    run_weaning_arith 算术域默认 run（不传 judge_source_id·config 默认 None）→ 算术域 :376
    _is_verify_modality 早返绕 build_judge_fn → ctx.judge_source_independent 保持 TrainContext
    默认 False（:167·非 build_judge_fn 算出·结果等价同源 False）→ 路径 B 读 False（同原硬编·零翻）。
    D3_judge_source_independent in blockers（算术域 D3 仍 False·诚实·W7 才接 judge_source_independent_arith）。
    """
    result = run_weaning_arith(rounds_per_stage=1, training_mode=True,
                               flat_floors=True, run_dir=str(tmp_path / "w3pre"),
                               return_backend=False)
    # 默认 None → D3 False（bit-identical·算术域通用路径 teacher=None 同源）
    assert "D3_judge_source_independent" in result.weaning_blockers, (
        f"默认 judge_source_id=None 须 D3 False（bit-identical·算术域 teacher=None 同源）·"
        f"blockers={result.weaning_blockers}")
    assert not result.weaning_ready


def test_w3_arith_judge_not_teacher_alias(tmp_path):
    """★算术域 vm_proof 自锚非教师本尊（反同源偷渡·风险②）。

    算术域 run_weaning_arith 不传 teacher（teacher=None·formal_train :110 默认 None）→
    _run_verify_round:374 _is_verify_modality 早返绕 judge → judge_fn 不构建（无教师本尊可偷渡）→
    vm_proof_fn 读 spec.expected（corpus fixture·非 teacher.judge_ground_truth）→ 裁判源=VM 执行值自锚。
    judge_source_independent_arith(verify_uses_vm_proof=True, teacher_not_judge=True)=True
    （架构保证独立·非教师本尊别名·反风险②同源偷渡）。
    """
    result, _backend = run_weaning_arith(rounds_per_stage=1, training_mode=True,
                                         flat_floors=True,
                                         run_dir=str(tmp_path / "w3alias"),
                                         return_backend=True)
    # 算术域 vm_proof 自锚 reward>0（W0 既有·裁判源=VM 执行值非教师 GT）
    assert result.final_metrics.conduction_rate > 0, (
        "算术域 vm_proof 自锚须 conduction_rate>0（裁判源=VM 执行值·非教师本尊）")
    # 算术域 D3 判定接口 True（架构保证：vm_proof 自锚 + 绕 judge·非教师本尊别名）
    assert judge_source_independent_arith(verify_uses_vm_proof=True,
                                          teacher_not_judge=True) is True, (
        "算术域裁判源=vm_proof 自锚（非教师本尊）→ judge_source_independent_arith 须 True")
    # 反同源偷渡（风险②）：算术域 teacher=None（run_weaning_arith 不传 teacher）·
    # judge_fn 不构建（_run_verify_round:374 早返）·无教师本尊可偷渡·裁判源天然独立
    # （通用路径 D3 仍 False·因 teacher=None 同源 sid=0·但算术域裁判是 vm_proof 非通用 judge·W7 接全）
    assert "D3_judge_source_independent" in result.weaning_blockers, (
        "算术域通用路径 D3 仍 False（teacher=None 同源·W7 才接 judge_source_independent_arith）·"
        f"blockers={result.weaning_blockers}")
