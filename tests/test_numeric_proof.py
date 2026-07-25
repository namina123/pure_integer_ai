"""tests.test_numeric_proof — 刀 B 数值/算术 cue 测试套件（形式 cue 扩展第二刀·语言域 LIVE form_proof_fn）。

测数值 cue 4 件齐（Option A·数值声明不入图·闭包传·同刀 A）：
  ① 词表/类型（ARITH_EQUALS_CUE·arith_op_of 加/减/乘·_parse_int_token）
  ② 构造器（extract_numeric_claims·NUM OP NUM 等于 NUM 窗口扫描·独立函数·不改 extract_cues 3-tuple）
  ③ 消费者（numeric_proof_fn_factory·全声明一致→1/任一违反→0/空→None·构造性检查层·直接整数算术）
  ④ e2e（_run_numeric_verify_round·NUMERIC_PROOF_MODE·SELF_PRODUCED）+ capability_exam 消费 + Layer0 反 theater

诚实边界：构造性检查 ≠ 构造性验证（左式/右式数 single-source·Layer0 标 SELF_PRODUCED·全自产不准停·同刀 A）·
数值声明不入图（Option A·闭包传·同刀 A 防结构发现污染）·为何直接整数算术非 execute_composes_value（平坦表达式直接算术即可·
建 COMPOSES 树反图污染+无验证增益·详 numeric_proof.py docstring）。
"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.cognition.shared.types import (
    LANG_ZH, LANG_EN, MODALITY_LANGUAGE, VERIFY_SOURCE_SELF_PRODUCED,
)
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_CAUSES
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, DefaultRoundRunner, make_train_context,
)
from pure_integer_ai.experiments.capability_exam import (
    run_capability_exam, CapabilityReport,
)
from pure_integer_ai.cognition.result.layer0_anchor import (
    is_constructive_verification, external_anchor_satisfied,
)
from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD, OPCODE_SUB, OPCODE_MUL
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, ARITH_EQUALS_CUE, arith_op_of, CAUSES_CUE_FORWARD, IS_A_CUE,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_numeric_claims, extract_numeric_claims_gated, _parse_int_token,
)
from pure_integer_ai.training.numeric_proof import numeric_proof_fn_factory


def _numeric_item(tokens=None, *, claim_true: bool = True, lang: int = LANG_ZH):
    """建数值声明语言 item（MODALITY_LANGUAGE·tokens 含 NUM OP NUM 等于 NUM）。

    claim_true=True → "3 加 5 等于 8"（3+5=8 真·reward=1）；False → "3 加 5 等于 9"（假·reward=0）。
    """
    if tokens is not None:
        toks = list(tokens)
    else:
        toks = ["3", "加", "5", "等于", "8"] if claim_true else ["3", "加", "5", "等于", "9"]
    return CollectedItem(
        tokens=toks,
        role_seq=[1] * len(toks),
        collect_type=COLLECT_CAUSES,
        source=SOURCE_BARE_TEXT,
        lang=lang,
    )


# ============ 件① 词表/类型（cue_type_of + arith_op_of + _parse_int_token） ============

def test_cue_type_of_equals_zh():
    """ZH 等式词（等于）→ ARITH_EQUALS_CUE（exact 匹配·数值等式声明锚）。"""
    assert cue_type_of("等于", LANG_ZH) == ARITH_EQUALS_CUE


def test_cue_type_of_equals_en():
    """EN 等式词（equals）→ ARITH_EQUALS_CUE。"""
    assert cue_type_of("equals", LANG_EN) == ARITH_EQUALS_CUE


def test_cue_type_of_non_arith_unchanged():
    """因果/系词/时序 cue 不受数值扩展影响（守既有语义·零行为变）。"""
    assert cue_type_of("导致", LANG_ZH) == CAUSES_CUE_FORWARD
    assert cue_type_of("是一种", LANG_ZH) == IS_A_CUE


def test_arith_op_of_zh():
    """ZH 算子词（加/加上/减/减去/乘/乘以）→ OPCODE_ADD/SUB/MUL。"""
    assert arith_op_of("加", LANG_ZH) == OPCODE_ADD
    assert arith_op_of("加上", LANG_ZH) == OPCODE_ADD
    assert arith_op_of("减", LANG_ZH) == OPCODE_SUB
    assert arith_op_of("减去", LANG_ZH) == OPCODE_SUB
    assert arith_op_of("乘", LANG_ZH) == OPCODE_MUL
    assert arith_op_of("乘以", LANG_ZH) == OPCODE_MUL


def test_arith_op_of_en():
    """EN 算子词（plus/minus/times）→ OPCODE_ADD/SUB/MUL。"""
    assert arith_op_of("plus", LANG_EN) == OPCODE_ADD
    assert arith_op_of("minus", LANG_EN) == OPCODE_SUB
    assert arith_op_of("times", LANG_EN) == OPCODE_MUL


def test_arith_op_of_non_op_returns_none():
    """非算子词 → None（守反统计契约·不凑配）。"""
    assert arith_op_of("等于", LANG_ZH) is None   # 等式词非算子词
    assert arith_op_of("3", LANG_ZH) is None      # 数字非算子词
    assert arith_op_of("加", LANG_EN) is None      # ZH 词在 EN lang 不命中（lang 隔离）


def test_parse_int_token_valid():
    """_parse_int_token：ASCII 数字（含负号）→ int·纯整数。"""
    assert _parse_int_token("3") == 3
    assert _parse_int_token("0") == 0
    assert _parse_int_token("-5") == -5
    assert _parse_int_token("42") == 42


def test_parse_int_token_invalid():
    """_parse_int_token：float/字母/中文数字/空 → None（守反统计·禁浮点·纯整数铁律）。"""
    assert _parse_int_token("3.5") is None     # float 拒
    assert _parse_int_token("abc") is None     # 字母拒
    assert _parse_int_token("三") is None       # 中文数字 defer（首版窄域）
    assert _parse_int_token("") is None         # 空拒
    assert _parse_int_token("-") is None        # 仅负号拒
    assert _parse_int_token("１２") is None     # 全角数字拒（isascii 守）


# ============ 件② 构造器（extract_numeric_claims·独立函数） ============

def test_extract_numeric_claims_add():
    """3 加 5 等于 8 → [(3, OPCODE_ADD, 5, 8)]（窗口扫描·cue=等于 在 index 3）。"""
    claims = extract_numeric_claims(["3", "加", "5", "等于", "8"], lang=LANG_ZH)
    assert claims == [(3, OPCODE_ADD, 5, 8)], f"got {claims}"


def test_extract_numeric_claims_sub():
    """10 减 3 等于 7 → [(10, OPCODE_SUB, 3, 7)]。"""
    claims = extract_numeric_claims(["10", "减", "3", "等于", "7"], lang=LANG_ZH)
    assert claims == [(10, OPCODE_SUB, 3, 7)]


def test_extract_numeric_claims_mul():
    """4 乘 6 等于 24 → [(4, OPCODE_MUL, 6, 24)]。"""
    claims = extract_numeric_claims(["4", "乘", "6", "等于", "24"], lang=LANG_ZH)
    assert claims == [(4, OPCODE_MUL, 6, 24)]


def test_extract_numeric_claims_en():
    """EN：3 plus 5 equals 8 → [(3, OPCODE_ADD, 5, 8)]（lang 隔离·EN 词表命中）。"""
    claims = extract_numeric_claims(["3", "plus", "5", "equals", "8"], lang=LANG_EN)
    assert claims == [(3, OPCODE_ADD, 5, 8)]


def test_extract_numeric_claims_boundary_insufficient():
    """边界：左式 3 token 不足（等于 在 index<3）→ 跳·守反统计（不凑配）。"""
    # 等于 在 index 2（左式仅 2 token "加 5"）→ 不足 → 空
    claims = extract_numeric_claims(["加", "5", "等于", "8"], lang=LANG_ZH)
    assert claims == [], "左式 3 token 不足 → 跳"
    # 等于 在末尾（无右式）→ 跳
    claims = extract_numeric_claims(["3", "加", "5", "等于"], lang=LANG_ZH)
    assert claims == [], "无右式 → 跳"


def test_extract_numeric_claims_non_match_skipped():
    """不匹配模式（非数字/非算子词）→ 跳·守反统计契约。"""
    # 左式非数字（"苹果" 非数字）
    claims = extract_numeric_claims(["苹果", "加", "5", "等于", "8"], lang=LANG_ZH)
    assert claims == [], "左式非数字 → 跳"
    # 算子位非算子词（"和" 非算子词）
    claims = extract_numeric_claims(["3", "和", "5", "等于", "8"], lang=LANG_ZH)
    assert claims == [], "算子位非算子词 → 跳"
    # 右式非数字
    claims = extract_numeric_claims(["3", "加", "5", "等于", "八"], lang=LANG_ZH)
    assert claims == [], "右式中文数字 → 跳"


def test_extract_numeric_claims_multi_claims():
    """多声明同段：两个等于 cue → 两声明（逐 cue 独立·确定性序）。"""
    # "3 加 5 等于 8 减 3 等于 5" → 两个等于锚·两声明
    claims = extract_numeric_claims(["3", "加", "5", "等于", "8", "减", "3", "等于", "5"], lang=LANG_ZH)
    # 第一个等于在 index 3：3 加 5 等于 8 → (3, ADD, 5, 8)
    # 第二个等于在 index 7：8 减 3 等于 5 → (8, SUB, 3, 5)
    assert claims == [(3, OPCODE_ADD, 5, 8), (8, OPCODE_SUB, 3, 5)], f"got {claims}"


def test_extract_numeric_claims_no_equals_empty():
    """无等于 cue → 空（无数值声明·守反统计）。"""
    claims = extract_numeric_claims(["3", "加", "5"], lang=LANG_ZH)
    assert claims == []


def test_extract_numeric_claims_empty_tokens():
    """空 tokens → 空（确定性·不崩）。"""
    assert extract_numeric_claims([], lang=LANG_ZH) == []


def test_extract_numeric_claims_gated_off_empty():
    """CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归（同 extract_cues_gated 范式）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = False
    try:
        claims = extract_numeric_claims_gated(["3", "加", "5", "等于", "8"], lang=LANG_ZH)
        assert claims == [], "gate OFF 返空（bit-identical）"
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_extract_numeric_claims_gated_on():
    """CUE_EXTRACTOR_MODE ON → 提取非空（生产路径·frozenset 第一源命中）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        claims = extract_numeric_claims_gated(["3", "加", "5", "等于", "8"], lang=LANG_ZH)
        assert claims == [(3, OPCODE_ADD, 5, 8)]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_extract_numeric_claims_pure_int():
    """claims 全纯整数（assert_int 守·无浮点·op_opcode 亦纯整 symbol_id）。"""
    claims = extract_numeric_claims(["3", "加", "5", "等于", "8"], lang=LANG_ZH)
    for (ln, op, rn, res) in claims:
        assert isinstance(ln, int) and not isinstance(ln, bool)
        assert isinstance(op, int) and not isinstance(op, bool)
        assert isinstance(rn, int) and not isinstance(rn, bool)
        assert isinstance(res, int) and not isinstance(res, bool)


# ============ 件③ 消费者（numeric_proof_fn_factory·构造性检查层） ============

def test_numeric_proof_fn_all_hold_returns_1():
    """全声明算术一致 → 1（verified·构造性检查通过·3+5=8 真）。"""
    fn = numeric_proof_fn_factory(claims=[(3, OPCODE_ADD, 5, 8)])
    assert fn(None, None, None) == 1


def test_numeric_proof_fn_violation_returns_0():
    """任一声明违反 → 0（mismatch·3+5=9 假·构造性检查未过）。"""
    fn = numeric_proof_fn_factory(claims=[(3, OPCODE_ADD, 5, 9)])
    assert fn(None, None, None) == 0


def test_numeric_proof_fn_sub_mul():
    """减/乘算术一致 → 1（10-3=7 / 4×6=24）。"""
    fn_sub = numeric_proof_fn_factory(claims=[(10, OPCODE_SUB, 3, 7)])
    assert fn_sub(None, None, None) == 1
    fn_mul = numeric_proof_fn_factory(claims=[(4, OPCODE_MUL, 6, 24)])
    assert fn_mul(None, None, None) == 1


def test_numeric_proof_fn_sub_violation():
    """减违反 → 0（10-3=8 假）。"""
    fn = numeric_proof_fn_factory(claims=[(10, OPCODE_SUB, 3, 8)])
    assert fn(None, None, None) == 0


def test_numeric_proof_fn_multi_mixed():
    """多声明混合：一真一假 → 0（任一违反即 mismatch·短路语义）。"""
    fn = numeric_proof_fn_factory(claims=[(3, OPCODE_ADD, 5, 8), (4, OPCODE_MUL, 6, 25)])
    # 第一声明 3+5=8 真·第二 4×6=25 假 → 0
    assert fn(None, None, None) == 0


def test_numeric_proof_fn_multi_all_hold():
    """多声明全真 → 1。"""
    fn = numeric_proof_fn_factory(claims=[(3, OPCODE_ADD, 5, 8), (10, OPCODE_SUB, 3, 7)])
    assert fn(None, None, None) == 1


def test_numeric_proof_fn_empty_returns_none():
    """claims 空 → None（vacate·无数值声明可验·诚实退场·非 pass·非 theater）。"""
    fn = numeric_proof_fn_factory(claims=[])
    assert fn(None, None, None) is None


def test_numeric_proof_fn_deterministic():
    """确定性：同输入同输出（整数算术·bit-identical）。"""
    fn = numeric_proof_fn_factory(claims=[(3, OPCODE_ADD, 5, 8)])
    r1 = fn(None, None, None)
    r2 = fn(None, None, None)
    assert r1 == r2 == 1


def test_numeric_proof_fn_negative_numbers():
    """负数声明：-3 加 5 等于 2 → 1（_parse_int_token 解析负号·算术正确）。"""
    fn = numeric_proof_fn_factory(claims=[(-3, OPCODE_ADD, 5, 2)])
    assert fn(None, None, None) == 1


def test_numeric_proof_fn_defensive_copy():
    """factory 防御性拷贝 claims（对抗审 P2-3）：caller 后续 mutation 不改已造 fn 行为。"""
    claims = [(3, OPCODE_ADD, 5, 8)]
    fn = numeric_proof_fn_factory(claims=claims)
    # caller 后续 append（违反声明）·不应影响已捕获的 fn（闭包持拷贝非引用）
    claims.append((4, OPCODE_MUL, 6, 25))   # 假声明·若 fn 持引用会致 fn()→0
    assert fn(None, None, None) == 1, "factory 须拷贝·caller mutation 不影响 fn（防 alias bug）"


def test_parse_int_token_none_safe():
    """_parse_int_token(None) / 非 str → None（对抗审 P2-4·不 crash·与 cue_type_of 一致）。"""
    assert _parse_int_token(None) is None
    assert _parse_int_token(123) is None      # 非 str（int）→ None
    assert _parse_int_token(["3"]) is None    # 非 str（list）→ None


# ============ 件③ e2e（_run_numeric_verify_round·NUMERIC_PROOF_MODE·SELF_PRODUCED） ============

def test_run_numeric_verify_round_reward_1_self_produced():
    """e2e：数值声明 item（3+5=8）+ CUE+NUMERIC gates ON → reward=1 + verify_source=SELF_PRODUCED。

    镜像 occurrence-order SELF_PRODUCED 范式·但数值声明 self-contained
    （无需 token→ConceptRef resolve·无需 EDGE_PRECEDES query·比时序简）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_num = gates.NUMERIC_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.NUMERIC_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        item = _numeric_item(claim_true=True)   # 3 加 5 等于 8
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is not None, "数值 verify round 须产 episode（gates ON + 数值 cue·路由应走 _run_numeric_verify_round）"
        assert res.episode.reward == 1, "3+5=8 → reward=1（数值声明算术一致·构造性检查通过）"
        assert res.episode.verify_source == VERIFY_SOURCE_SELF_PRODUCED, (
            f"_run_numeric_verify_round 须标 SELF_PRODUCED（数 single-source·构造性检查非验证）"
            f"·got verify_source={res.episode.verify_source}")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.NUMERIC_PROOF_MODE = saved_num


def test_run_numeric_verify_round_reward_0_on_violation():
    """e2e：假声明 item（3+5=9）+ gates ON → reward=0（违反·构造性检查未过·veto）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_num = gates.NUMERIC_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.NUMERIC_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        item = _numeric_item(claim_true=False)   # 3 加 5 等于 9（假）
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is not None
        assert res.episode.reward == 0, "3+5=9 假 → reward=0（违反·veto）"
        assert res.episode.verify_source == VERIFY_SOURCE_SELF_PRODUCED
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.NUMERIC_PROOF_MODE = saved_num


def test_run_numeric_verify_round_gate_off_no_episode():
    """NUMERIC_PROOF_MODE OFF → 路由不走·数值 item 走正常语言 episode_loop（bit-identical·零行为变）。

    数值 item 单段 struct_ref 孤立（len<2）→ 正常路径返空 RoundResult（无 numeric verify episode）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_num = gates.NUMERIC_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.NUMERIC_PROOF_MODE = False   # OFF
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        item = _numeric_item(claim_true=True)
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        # gate OFF → 不走 numeric verify·单段 struct_ref 孤立 → 正常路径 RoundResult() 空
        # （不伪造 numeric verify episode·bit-identical）
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_SELF_PRODUCED, (
            "NUMERIC_PROOF_MODE OFF → 不产 SELF_PRODUCED numeric verify episode（bit-identical）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.NUMERIC_PROOF_MODE = saved_num


# ============ 件④ capability_exam 消费 + Layer0 反 theater ============

def test_capability_exam_runs_with_numeric_item(tmp_path):
    """capability_exam + 数值声明 item → 产 report（不崩·layer0 6 key 齐·additive 字段）。

    **诚实 scope**：capability_exam 默认 teacher=None → reward 阶段不激活 → 语言域 numeric verify episode
    不被 formal_train stage loop 收集（同 arith 单 item·task-driven 才走独立收集路径）。故此处只验
    capability_exam 不崩 + layer0 字段齐·**不**断 self_produced>0（收集须 teacher·生产 weaning-pre 有）。
    numeric verify episode 生产 + SELF_PRODUCED 标记由直调 e2e（test_run_numeric_verify_round_*）守。
    """
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "num"), run_id="num_1")
        report = run_capability_exam(
            cfg, [_numeric_item(claim_true=True)],
            backend=b, runner=DefaultRoundRunner())
        assert isinstance(report, CapabilityReport)
        # layer0 6 key 齐（additive 字段·project_layer0 消费 result.episodes·空时全 0·确定性）
        assert set(report.layer0_attribution.keys()) == {
            "external_verified", "self_produced_check_passed", "self_produced_check_failed",
            "anchor_satisfied", "anchor_violated", "total",
        }
    finally:
        gates.TRAINING_MODE = saved


def test_project_layer0_counts_numeric_self_produced():
    """capability_exam 消费者（project_layer0）正确计数值值 SELF_PRODUCED episode（构造 episode·直测消费）。

    构造 result.episodes 含 numeric verify SELF_PRODUCED reward=1 episode → project_layer0 计
    self_produced_check_passed（非 external_verified·反 theater）。这是刀 B 经 Layer0 的真实消费链
    （project_layer0 → count_layer0·is_constructive_verification 排除 SELF_PRODUCED）。
    """
    from types import SimpleNamespace
    from pure_integer_ai.cognition.shared.types import Episode, VERIFY_SOURCE_EXTERNAL
    from pure_integer_ai.experiments.capability_exam import project_layer0
    # 构造：1 numeric SELF_PRODUCED (reward=1·检查通过) + 1 EXTERNAL (reward=1·真验证·对照)
    result = SimpleNamespace(episodes=[
        Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED),   # numeric verify episode
        Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL),        # arith vm_proof 对照
    ])
    counts = project_layer0(result)
    assert counts["self_produced_check_passed"] == 1, "numeric SELF_PRODUCED reward=1 计检查通过"
    assert counts["external_verified"] == 1, "仅 EXTERNAL 计构造性验证（反 theater·SELF_PRODUCED 不计）"
    assert counts["anchor_violated"] == 1, "SELF_PRODUCED 违外部锚门（全自产不准停）"
    assert counts["anchor_satisfied"] == 1, "EXTERNAL 满足锚门"
    assert counts["total"] == 2


def test_numeric_episode_anti_theater_not_external_verified():
    """**反 theater 端到端**：数值 verify episode reward=1（声明一致）但非构造性验证 + 违锚门。

    SELF_PRODUCED reward=1（3+5=8 算术一致·检查通过）≠ 构造性验证（数 single-source）。
    对照：is_constructive_verification=False + external_anchor_satisfied=False（全自产不准停）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_num = gates.NUMERIC_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.NUMERIC_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        res = r.run_round_full(ctx, _numeric_item(claim_true=True), STAGE3_REWARD, 0)
        ep = res.episode
        assert ep is not None and ep.reward == 1
        # 反 theater 核心：reward=1 但非构造性验证 + 违外部锚门（全自产不准停）
        assert is_constructive_verification(ep) is False, (
            "SELF_PRODUCED 不计构造性验证（反 theater）·即使 reward=1（检查通过非验证）")
        assert external_anchor_satisfied(ep) is False, (
            "SELF_PRODUCED 全自产不准停（违外部锚门）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.NUMERIC_PROOF_MODE = saved_num


def test_numeric_verify_bit_identical():
    """project_layer0 两构造同输入 → 一致（bit-identical·count_layer0 确定性·numeric SELF_PRODUCED 分桶稳定）。"""
    import json
    from types import SimpleNamespace
    from pure_integer_ai.cognition.shared.types import Episode
    from pure_integer_ai.experiments.capability_exam import project_layer0

    def build():
        return SimpleNamespace(episodes=[
            Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED),
            Episode(reward=0, verify_source=VERIFY_SOURCE_SELF_PRODUCED),
        ])

    l1 = project_layer0(build())
    l2 = project_layer0(build())
    assert json.dumps(l1, sort_keys=True) == json.dumps(l2, sort_keys=True), "两构造 layer0 不一致·违 bit-identical"
    assert l1["self_produced_check_passed"] == 1 and l1["self_produced_check_failed"] == 1


# ============ 诚实边界（docstring 标注） ============

def test_numeric_proof_is_constructive_check_not_verification():
    """诚实标注：numeric_proof 是构造性检查层（整数算术确定性）·非构造性验证（须 R6·single-source）。"""
    import pure_integer_ai.training.numeric_proof as mod
    docstring = mod.__doc__ or ""
    assert "构造性检查" in docstring, "模块 docstring 须标构造性检查层"
    assert "构造性验证" in docstring, "须诚实标非构造性验证（数 single-source·须 R6 升验证）"
    assert "SELF_PRODUCED" in docstring or "Layer0" in docstring, "须标 Layer0 SELF_PRODUCED"


def test_numeric_proof_option_a_no_persist():
    """Option A 诚实标注：数值声明不入图（闭包传·同刀 A 防结构发现污染 + emergence 干扰）。"""
    import pure_integer_ai.training.numeric_proof as mod
    docstring = mod.__doc__ or ""
    assert "不入图" in docstring or "闭包" in docstring, "须标 Option A（数值声明不入图·闭包传）"


def test_numeric_proof_no_execute_composes_value_rationale():
    """诚实标注：为何直接整数算术非 execute_composes_value（防图污染 + 无验证增益）。"""
    import pure_integer_ai.training.numeric_proof as mod
    docstring = mod.__doc__ or ""
    assert "execute_composes_value" in docstring, "须标 execute_composes_value 取舍理由"


def test_numeric_proof_never_reward():
    """数值验序永不接 reward（self_proof_fn 通道·reward 通道严格不动）。"""
    import pure_integer_ai.training.numeric_proof as mod
    docstring = mod.__doc__ or ""
    assert "reward" in docstring.lower(), "须标永不接 reward"


# ============ gate 默认 OFF（bit-identical 守 CI=生产） ============

def test_numeric_proof_mode_default_off():
    """NUMERIC_PROOF_MODE 默认 OFF·守 CI 回归 bit-identical（路由不走·既有语言域 episode_loop 不变）。"""
    import importlib
    import pure_integer_ai.config.gates as g
    importlib.reload(g)
    assert g.NUMERIC_PROOF_MODE is False, "NUMERIC_PROOF_MODE 默认 OFF 守 bit-identical"
