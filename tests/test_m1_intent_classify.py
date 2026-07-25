"""M1片2 intent 分类测试（doc/重来_M1片2_intent分类设计_2026-07-08.md）。

M1 = intent producer 缺失：消费者 judge.py:224,236 接好·机制 types.py 三 bool 落好·
只差 formal_train.py:366,1448 两处 IntentType(INTENT_QUESTION) 硬编码填值。本测覆盖：

  - _has_causes_signal 单元（与 causes.py:38-51 / observe.py:215-220 建边同源·各边界）
  - classify_intent 单元（type 硬要求 INTENT_QUESTION / sink 透传含 None / 三 bool 取值 / 反 theater）
  - gate M1_INTENT_CLASSIFY_MODE default OFF（守 CI bit-identical）
  - run_round_full gate OFF → classify_intent 不调（硬编码 else 分支·bit-identical 证明）
  - run_round_full gate ON（+ CUE ON）→ classify_intent 调 + 因果 fixture is_causal=True
  - formal_train 生产 try/finally 翻 ON → classify_intent 真调（审1 P0·否则 theater）
  - formal_train 因果 corpus → observe 自产 EDGE_CAUSES（dag_path 含锚前置·G3a else 分支不误 veto）
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    Segment, IntentType, INTENT_COMMAND, INTENT_QUESTION,
    MODALITY_LANGUAGE, LANG_ZH, DOMAIN_TEXT,
)
from pure_integer_ai.cognition.shared.edge_types import EDGE_CAUSES
from pure_integer_ai.config import gates
from pure_integer_ai.cognition.understanding import intent_classify as ic
from pure_integer_ai.experiments.collection import CollectedItem
from tests.boundary_fixtures import attach_boundary_fixture


# ---- helpers ----

def _seg(tokens, **kw):
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens, **kw)


def _causal_item() -> CollectedItem:
    """因果多句 item："雨导致地湿。猫追老鼠" → 2 段·段1 cue 导致 → cue_based_causal_pairs 非空。

    两异 token（雨≠地湿）避 a==b self-pair（causes.py:57 不建边）。cue 在两异 token 间·
    CUE_EXTRACTOR_MODE ON 时 extract_cues_gated 产 [(0,2)]。
    """
    return attach_boundary_fixture(CollectedItem(
        tokens=["雨", "导致", "地湿", "。", "猫", "追", "老鼠"]),
        cut_after=(4,),
    )


def _non_causal_item() -> CollectedItem:
    """非因果多句 item："狗追兔子。太阳升起" → 2 段·无 cue 词 → cue_based_causal_pairs 空。"""
    return attach_boundary_fixture(CollectedItem(
        tokens=["狗", "追", "兔子", "。", "太阳", "升起"]),
        cut_after=(4,),
    )


# ============ _has_causes_signal 单元（与 observe 建边同源）============

def test_has_causes_signal_empty_segments():
    """空 segments → False（无因果对·observe 不建 CAUSES）。"""
    assert ic._has_causes_signal([]) is False


def test_has_causes_signal_segment_no_pairs():
    """segment 两源皆空 → False。"""
    seg = _seg(["猫", "追", "老鼠"])
    assert ic._has_causes_signal([seg]) is False


def test_has_causes_signal_structured_pairs():
    """structured_causal_pairs 非空 → True（来源①·EPI_STRUCTURED·与 causes.py:39 同源）。"""
    seg = _seg(["雨", "导致", "地湿"], structured_causal_pairs=[(0, 2)])
    assert ic._has_causes_signal([seg]) is True


def test_has_causes_signal_cue_pairs():
    """cue_based_causal_pairs 非空 → True（来源②·EPI_CUE·与 causes.py:43 同源）。"""
    seg = _seg(["雨", "导致", "地湿"], cue_based_causal_pairs=[(0, 2)])
    assert ic._has_causes_signal([seg]) is True


def test_has_causes_signal_multi_segment_one_has_pairs():
    """多 segment·任一非空 → True（首段空·次段有 pair）。"""
    a = _seg(["狗", "追", "兔子"])
    b = _seg(["雨", "导致", "地湿"], cue_based_causal_pairs=[(0, 2)])
    assert ic._has_causes_signal([a, b]) is True


# ============ classify_intent 单元 ============

def test_classify_intent_type_question_when_command_gate_off():
    """W7+B-PR1 doc §16：INTENT_COMMAND_MODE gate OFF（默认）→ type 永 INTENT_QUESTION（即使含动作意图词）。

    守 bit-identical（默认 OFF·_has_action_intent 返 False·type=QUESTION）。
    """
    intent = ic.classify_intent(sink=(1, 100), segments=[_seg(["帮我", "生成", "代码"])])
    assert intent.type == INTENT_QUESTION


def test_classify_intent_type_question_no_action_intent_gate_on(monkeypatch):
    """W7+B-PR1 doc §16：gate ON + 无动作意图词（命令词 OR 动作词）→ type=INTENT_QUESTION。"""
    monkeypatch.setattr(gates, "INTENT_COMMAND_MODE", True)
    intent = ic.classify_intent(sink=(1, 100), segments=[_seg(["雨", "导致", "地湿"],
                                                                cue_based_causal_pairs=[(0, 2)])])
    assert intent.type == INTENT_QUESTION


def test_classify_intent_type_command_action_intent_gate_on(monkeypatch):
    """W7+B-PR1 doc §16：gate ON + 含动作意图词（命令词 帮我/请 OR 动作词 生成/计算）→ type=INTENT_COMMAND。

    命令判定 = 命令词 OR 动作词命中（doc §16.4·覆盖引导词祈使 + 有动作词裸祈使）。
    dag_path.py:302 早已 tuple 含 COMMAND·Q/C 等价合法终止态·零终止态差异。
    "帮我生成代码" = 帮我(COMMAND_MOOD) + 生成(ACTION_GENERATE)·命中→COMMAND。
    """
    monkeypatch.setattr(gates, "INTENT_COMMAND_MODE", True)
    intent = ic.classify_intent(sink=(1, 100), segments=[_seg(["帮我", "生成", "代码"])])
    assert intent.type == INTENT_COMMAND


def test_classify_intent_type_command_action_verb_only_gate_on(monkeypatch):
    """W7+B-PR1 doc §16：gate ON + 仅动作词（裸祈使"生成代码"·无引导词）→ type=INTENT_COMMAND。

    动作词（生成/计算·ACTION_*）也判命令（doc §16.4·一条命令 = 祈使 mood OR 动作内容）。
    覆盖有动作词裸祈使·纯句式（去开门·无引导词无动作词）仍漏 defer B-PR2。
    """
    monkeypatch.setattr(gates, "INTENT_COMMAND_MODE", True)
    intent = ic.classify_intent(sink=(1, 100), segments=[_seg(["生成", "代码"])])
    assert intent.type == INTENT_COMMAND


def test_classify_intent_sink_passthrough_including_none():
    """sink 透传 caller struct_refs[-1]（选项 B·维持 reward 通路）·None 也透传（不伪造）。"""
    ref = (2, 55)
    assert ic.classify_intent(sink=ref, segments=[]).sink == ref
    assert ic.classify_intent(sink=None, segments=[]).sink is None


def test_classify_intent_is_causal_matches_signal():
    """is_causal_reasoning = _has_causes_signal(segments)（因果 True / 非因果 False）。"""
    causal = [_seg(["雨", "导致", "地湿"], cue_based_causal_pairs=[(0, 2)])]
    non_causal = [_seg(["猫", "追", "老鼠"])]
    assert ic.classify_intent(sink=None, segments=causal).is_causal_reasoning is True
    assert ic.classify_intent(sink=None, segments=non_causal).is_causal_reasoning is False


def test_classify_intent_structural_and_value_always_false():
    """反 theater：is_structural_sequence_reasoning / has_value_claim 首版永 False。

    is_structural 语言域设计正确（code/arith 走 verify 绕 judge）·
    has_value_claim 双重 theater defer（PROPERTY builder 拆新 task #774）。
    无论 segments 是否含因果对·两 bool 不动。
    """
    causal = [_seg(["雨", "导致", "地湿"], cue_based_causal_pairs=[(0, 2)])]
    intent = ic.classify_intent(sink=None, segments=causal)
    assert intent.is_structural_sequence_reasoning is False
    assert intent.has_value_claim is False


# ============ _has_action_intent 单元（W7+B-PR1 doc §16·命令词 OR 动作词）============

def test_has_action_intent_gate_off_returns_false():
    """gate OFF（默认）→ 返 False（守 bit-identical·type 永 QUESTION·_has_action_intent gate OFF 早返）。"""
    assert ic._has_action_intent([_seg(["帮我", "生成"])]) is False


def test_has_action_intent_no_cue_gate_on(monkeypatch):
    """gate ON + 无命令词无动作词 → False（"导致"非动作意图·雨导致地湿 非命令）。"""
    monkeypatch.setattr(gates, "INTENT_COMMAND_MODE", True)
    assert ic._has_action_intent([_seg(["雨", "导致", "地湿"])]) is False


def test_has_action_intent_command_word_gate_on(monkeypatch):
    """gate ON + 命令词（帮我/请/给我/能不能/可不可以/麻烦·→COMMAND_MOOD）→ True。"""
    monkeypatch.setattr(gates, "INTENT_COMMAND_MODE", True)
    for marker in ["帮我", "请", "给我", "能不能", "可不可以", "麻烦"]:
        assert ic._has_action_intent([_seg([marker, "生成", "代码"])]) is True


def test_has_action_intent_action_verb_hits_gate_on(monkeypatch):
    """W7+B-PR1 doc §16.4：动作动词（生成/计算/分析/解决·→ACTION_*）**命中**判命令（裸祈使"生成代码"）。

    异 W7 第一版（§15·动作词不命中）——合并后动作词也判命令（OR·非职责分离）。
    纯句式（去开门·无动作词）仍漏 defer B-PR2。
    """
    monkeypatch.setattr(gates, "INTENT_COMMAND_MODE", True)
    for verb in ["生成", "计算", "分析", "解决"]:
        assert ic._has_action_intent([_seg([verb, "结果"])]) is True


# ============ gate default ============

def test_m1_gate_default_off():
    """M1_INTENT_CLASSIFY_MODE default OFF（守 CI 回归·OFF = 硬编码 INTENT_QUESTION bit-identical）。

    测 _flag 默认语义（env 未设→False）·不 reload 模块（防扰动同 session 其他 gate 运行态）。
    """
    import os
    from pure_integer_ai.config.gates import _flag
    saved = os.environ.pop("PURE_INTEGER_AI_M1_INTENT_CLASSIFY_MODE", None)
    try:
        assert _flag("PURE_INTEGER_AI_M1_INTENT_CLASSIFY_MODE", False) is False
    finally:
        if saved is not None:
            os.environ["PURE_INTEGER_AI_M1_INTENT_CLASSIFY_MODE"] = saved


def test_intent_command_gate_default_off():
    """INTENT_COMMAND_MODE default OFF（W7 doc §15·守 CI 回归·OFF = type 永 QUESTION bit-identical）。

    测 _flag 默认语义（env 未设→False）·不 reload 模块（防扰动同 session 其他 gate 运行态）。
    """
    import os
    from pure_integer_ai.config.gates import _flag
    saved = os.environ.pop("PURE_INTEGER_AI_INTENT_COMMAND_MODE", None)
    try:
        assert _flag("PURE_INTEGER_AI_INTENT_COMMAND_MODE", False) is False
    finally:
        if saved is not None:
            os.environ["PURE_INTEGER_AI_INTENT_COMMAND_MODE"] = saved


# ============ run_round_full 接线（gate OFF bit-identical + gate ON wiring）============

def _make_ctx():
    """建 TrainContext（make_train_context·run_round_full 直调用）。"""
    from pure_integer_ai.experiments.formal_train import make_train_context
    b = DictBackend()
    bootstrap(b)
    return make_train_context(b), b


def test_run_round_full_gate_off_does_not_call_classify(monkeypatch):
    """gate OFF → run_round_full 走硬编码 else 分支·classify_intent 不调（bit-identical OFF 证明）。

    run_round_full 自身不翻 M1 gate（仅 formal_train 生产入口翻）·故直调时 gate 保持 OFF。
    """
    calls = []
    def _spy(sink, segments, **kwargs):
        calls.append(sink)   # gate OFF 不应走到此·若走到说明 wiring 错
        return IntentType(type=INTENT_QUESTION, sink=sink)
    monkeypatch.setattr(ic, "classify_intent", _spy)
    monkeypatch.setattr(gates, "M1_INTENT_CLASSIFY_MODE", False)

    from pure_integer_ai.experiments.formal_train import DefaultRoundRunner
    from pure_integer_ai.training.stages import STAGE3_REWARD
    (ctx, _b) = _make_ctx()
    runner = DefaultRoundRunner()
    runner.run_round_full(ctx, _causal_item(), STAGE3_REWARD, 0)
    assert calls == [], "gate OFF → 须走硬编码 else·classify_intent 不应被调"


def test_run_round_full_gate_on_causal_calls_classify_is_causal_true(monkeypatch):
    """gate ON（M1 + CUE 同翻·镜像 formal_train 生产）→ classify_intent 调·因果 item is_causal=True。

    CUE_EXTRACTOR_MODE 须同翻（cue_extractor 在 _split_item_to_segments 读·gate OFF 则
    cue_based_causal_pairs 空 → is_causal False·与生产不一致）。M1 gate 在 run_round_full :366 读。
    """
    calls: list[IntentType] = []
    real = ic.classify_intent
    def _spy(sink, segments, **kwargs):
        r = real(sink, segments, **kwargs)
        calls.append(r)
        return r
    monkeypatch.setattr(ic, "classify_intent", _spy)
    monkeypatch.setattr(gates, "M1_INTENT_CLASSIFY_MODE", True)
    monkeypatch.setattr(gates, "CUE_EXTRACTOR_MODE", True)

    from pure_integer_ai.experiments.formal_train import DefaultRoundRunner
    from pure_integer_ai.training.stages import STAGE3_REWARD
    (ctx, _b) = _make_ctx()
    runner = DefaultRoundRunner()
    runner.run_round_full(ctx, _causal_item(), STAGE3_REWARD, 0)
    assert len(calls) >= 1, "gate ON → reward 阶段 :366 须调 classify_intent"
    assert any(it.is_causal_reasoning for it in calls), \
        "因果 item（雨导致地湿·cue ON 自产 CAUSES pair）→ is_causal_reasoning 须 True"


# ============ formal_train 生产 try/finally 翻 ON（审1 P0·核心反 theater）============

def test_formal_train_prod_tryfinally_flips_m1_gate_on(tmp_path, monkeypatch):
    """formal_train 生产入口 try/finally 翻 M1_INTENT_CLASSIFY_MODE ON（审1 P0·核心反 theater）。

    审1 P0：若 gate 非 try/finally 翻 ON·classify_intent 永不调 → is_causal 永假 →
    reward 退化核心病灶未修 = theater。本测在 run_round_full 入口采 gates.M1_INTENT_CLASSIFY_MODE·
    断言生产路径运行期间 gate 曾为 True（:1247 flip 生效·:1426 finally 复位）。

    注：reward 阶段 :366 真调 classify_intent 由 test_run_round_full_gate_on_causal_... 直证
    （run_round_full + STAGE3_REWARD + gate ON）。完整 formal_train 跑到 reward 阶段须语料过
    stage_metric_gate（:1345·小语料 stage2 即 break·非 M1 范围）·故本测聚焦"gate 生产翻 ON"
    这个审1 P0 关键点，不依赖 reward 阶段实际执行。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)   # 生产路径（stage_active_gates 读）
    seen: list[bool] = []
    from pure_integer_ai.experiments.formal_train import DefaultRoundRunner
    orig = DefaultRoundRunner.run_round_full
    def wrap(self, ctx, item, stage, rid):
        seen.append(bool(gates.M1_INTENT_CLASSIFY_MODE))
        return orig(self, ctx, item, stage, rid)
    monkeypatch.setattr(DefaultRoundRunner, "run_round_full", wrap)

    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    corpus = [_causal_item(), _non_causal_item()]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="m1_prod_flip", rounds_per_stage=1)
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    assert seen, "formal_train 须至少调一次 run_round_full（采 gate 状态）"
    assert any(seen), "生产 try/finally 须翻 M1_INTENT_CLASSIFY_MODE ON（审1 P0·否则 theater）"
    # finally 复位（防 gate 泄漏·守后续测 bit-identical）
    assert gates.M1_INTENT_CLASSIFY_MODE is False, "finally 须复位 M1 gate（saved_m1_intent）"


def test_h2_calibrate_fires_classify_intent_gate_on(monkeypatch):
    """H2 标定 site（formal_train.py:1456）gate ON 时真调 classify_intent（正确性审 J 补强）。

    _h2_calibrate 对每 item 调 run_round_full（产 res.dag_path）→ 建 CalibrationSample 时
    h2_intent = classify_intent(res.dag_path.sink, h2_segments)。_split_item_to_segments 重切
    与 reward :320 同源·res.dag_path.sink 派生（judge/calibrate_weights 不读 sink·值无害·审 D）。
    直调 _h2_calibrate（绕 formal_train stage_metric_gate·ctx.teacher=None 退化 GT=pass·审 E）。
    """
    monkeypatch.setattr(gates, "M1_INTENT_CLASSIFY_MODE", True)
    monkeypatch.setattr(gates, "CUE_EXTRACTOR_MODE", True)   # cue 自产（与 reward 测同）
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    calls: list[IntentType] = []
    real = ic.classify_intent
    def _spy(sink, segments, **kwargs):
        r = real(sink, segments, **kwargs)
        calls.append(r)
        return r
    monkeypatch.setattr(ic, "classify_intent", _spy)

    from pure_integer_ai.experiments.formal_train import DefaultRoundRunner, _h2_calibrate
    (ctx, _b) = _make_ctx()
    _h2_calibrate(ctx, [_causal_item()], DefaultRoundRunner())
    # reward site（run_round_full 内 :368）+ H2 site（:1456）各调一次 → ≥2·证 H2 接线真活非 theater
    assert len(calls) >= 2, "H2 site 须调 classify_intent（reward site + H2 site 各一次）"
    assert any(it.is_causal_reasoning for it in calls), "因果 item → is_causal True"
    assert all(not it.has_value_claim for it in calls), "has_value_claim theater defer（#774）"


def test_formal_train_causal_corpus_builds_causes_anchor(tmp_path, monkeypatch):
    """formal_train 因果 corpus → observe 自产 EDGE_CAUSES（CUE_EXTRACTOR 生产 ON）。

    dag_path 含 CAUSES 锚前置（judge.py:228 has_causes_anchor 通过·G3a else 分支不误 veto）。
    与 is_causal 同源（causes.py builder 按 segment 字段建）·证全链：cue 自产 → CAUSES 边 → 锚可命中。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)   # 生产条件（reward 阶段·与 spy 测同）
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    corpus = [_causal_item()]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="m1_causes_anchor", rounds_per_stage=1)
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())
    causes = [r for r in b.select("edge", where={"edge_type": EDGE_CAUSES})]
    assert len(causes) >= 1, \
        "因果 item（雨导致地湿·CUE ON）→ observe 须建 EDGE_CAUSES（dag_path CAUSES 锚前置）"
