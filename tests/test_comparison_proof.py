"""tests.test_comparison_proof — 刀 D 比较 cue 测试套件（形式 cue 扩展第四刀·语言域 LIVE form_proof_fn）。

测比较 cue 4 件齐（Option A·比较声明不入图·闭包传·同刀 A/B）：
  ① 词表/类型（comparison_op_of 大于/小于/不小于/不大于·CMP_GT/LT/GE/LE·_parse_int_token 复用刀 B）
  ② 构造器（extract_comparison_claims·NUM 比较OP NUM 紧邻 3-token 窗口·独立函数·不改 extract_cues 3-tuple）
  ③ 消费者（comparison_proof_fn_factory·cross_compare 交叉积·全声明序一致→1/任一违反→0/空→None·构造性检查层）
  ④ e2e（_run_comparison_verify_round·COMPARISON_PROOF_MODE·SELF_PRODUCED）+ capability_exam 消费 + Layer0 反 theater

诚实边界：构造性检查 ≠ 构造性验证（左/右式数 single-source·Layer0 标 SELF_PRODUCED·全自产不准停·同刀 A/B）·
比较声明不入图（Option A·闭包传·同刀 A/B 防污染）·为何用 cross_compare 非裸 sign(left−right)（compare.py 铁律
"任何比序强制走本模块"·零误差·给 cross_compare 首个真比较消费者·反 theater）·doc"命题值比序"(B) defer
（须 ref→surface 基建·concept_index 无反查·本刀做 (A) 字面数值比序）。
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
from pure_integer_ai.crosscut.integer.compare import CMP_GT, CMP_LT, CMP_GE, CMP_LE
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, comparison_op_of, is_comparison_op_token,
    CAUSES_CUE_FORWARD, IS_A_CUE,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_comparison_claims, extract_comparison_claims_gated,
)
from pure_integer_ai.training.comparison_proof import comparison_proof_fn_factory


def _comparison_item(tokens=None, *, claim_true: bool = True, lang: int = LANG_ZH):
    """建比较声明语言 item（MODALITY_LANGUAGE·tokens 含 NUM 比较OP NUM）。

    claim_true=True → "5 大于 3"（5>3 真·reward=1）；False → "3 大于 5"（3>5 假·reward=0）。
    """
    if tokens is not None:
        toks = list(tokens)
    else:
        toks = ["5", "大于", "3"] if claim_true else ["3", "大于", "5"]
    return CollectedItem(
        tokens=toks,
        role_seq=[1] * len(toks),
        collect_type=COLLECT_CAUSES,
        source=SOURCE_BARE_TEXT,
        lang=lang,
    )


# ============ 件① 词表/类型（comparison_op_of + is_comparison_op_token + bit-identical 邻居不污染） ============

def test_comparison_op_of_zh():
    """ZH 比较 OP 词（大于/小于/不小于/不大于）→ CMP_GT/LT/GE/LE。"""
    assert comparison_op_of("大于", LANG_ZH) == CMP_GT
    assert comparison_op_of("小于", LANG_ZH) == CMP_LT
    assert comparison_op_of("不小于", LANG_ZH) == CMP_GE
    assert comparison_op_of("不大于", LANG_ZH) == CMP_LE


def test_comparison_op_of_en():
    """EN 比较 OP 词（greater_than/less_than/at_least/at_most）→ CMP_GT/LT/GE/LE。"""
    assert comparison_op_of("greater_than", LANG_EN) == CMP_GT
    assert comparison_op_of("less_than", LANG_EN) == CMP_LT
    assert comparison_op_of("at_least", LANG_EN) == CMP_GE
    assert comparison_op_of("at_most", LANG_EN) == CMP_LE


def test_comparison_op_of_non_op_returns_none():
    """非比较 OP 词 → None（守反统计契约·不凑配）。"""
    assert comparison_op_of("等于", LANG_ZH) is None     # 等式词非比较 OP（刀 B）
    assert comparison_op_of("3", LANG_ZH) is None        # 数字非 OP
    assert comparison_op_of("大于", LANG_EN) is None      # ZH 词在 EN lang 不命中（lang 隔离）
    assert comparison_op_of("大于等于", LANG_ZH) is None   # 多字 token defer（首版仅 大于/小于/不大于/不小于）


def test_is_comparison_op_token():
    """is_comparison_op_token：exact 匹配四 OP·非 OP 返 False（extract 邻居判·反统计）。"""
    assert is_comparison_op_token("大于", LANG_ZH) is True
    assert is_comparison_op_token("不小于", LANG_ZH) is True
    assert is_comparison_op_token("大于", LANG_EN) is False   # lang 隔离
    assert is_comparison_op_token("5", LANG_ZH) is False
    assert is_comparison_op_token("导致", LANG_ZH) is False


def test_cue_type_of_not_polluted_by_comparison_op():
    """**bit-identical 硬守**：比较 OP 词不入 _CUE_WORDS → cue_type_of(大于/小于) 仍返 None。

    防 大于/小于 污染 extract_cues 邻居判（若入 _CUE_WORDS·cue_type_of(大于) 返非 None → extract_cues
    把 大于 当 cue 跳过配对 → 改变 CAUSES/IS_A/PRECEDES 提取行为 → 非 bit-identical）。异刀 B 等于入 _CUE_WORDS
    （刀 B 等于微改"X 所以 等于 Y"模式·刀 D 比较 OP 不入 _CUE_WORDS 更 safe·零行为变）。
    """
    assert cue_type_of("大于", LANG_ZH) is None, "大于 不入 _CUE_WORDS·cue_type_of 返 None（bit-identical）"
    assert cue_type_of("小于", LANG_ZH) is None
    assert cue_type_of("不小于", LANG_ZH) is None
    assert cue_type_of("greater_than", LANG_EN) is None


def test_cue_type_of_existing_cues_unchanged():
    """因果/系词 cue 不受比较扩展影响（守既有语义·零行为变）。"""
    assert cue_type_of("导致", LANG_ZH) == CAUSES_CUE_FORWARD
    assert cue_type_of("是一种", LANG_ZH) == IS_A_CUE


# ============ 件② 构造器（extract_comparison_claims·独立函数） ============

def test_extract_comparison_claims_gt():
    """5 大于 3 → [(5, CMP_GT, 3)]（3-token 窗口·OP=大于 在 index 1）。"""
    claims = extract_comparison_claims(["5", "大于", "3"], lang=LANG_ZH)
    assert claims == [(5, CMP_GT, 3)], f"got {claims}"


def test_extract_comparison_claims_lt():
    """3 小于 5 → [(3, CMP_LT, 5)]。"""
    claims = extract_comparison_claims(["3", "小于", "5"], lang=LANG_ZH)
    assert claims == [(3, CMP_LT, 5)]


def test_extract_comparison_claims_ge_le():
    """5 不小于 3 / 3 不大于 5 → CMP_GE / CMP_LE。"""
    assert extract_comparison_claims(["5", "不小于", "3"], lang=LANG_ZH) == [(5, CMP_GE, 3)]
    assert extract_comparison_claims(["3", "不大于", "5"], lang=LANG_ZH) == [(3, CMP_LE, 5)]


def test_extract_comparison_claims_equals_not_comparison_op():
    """**piece 2.1 反回归锁**：等于/equals **不**入 _COMPARISON_OP_WORDS（属刀B ARITH_EQUALS_CUE·数值等式声明）。

    故 extract_comparison_claims 不对 等于 抽比较声明——否则 "3 等于 5"（或 numeric "2 加 3 等于 5" 的
    [3,等于,5] 窗口）被误抽假比较声明 3==5·与 extract_numeric_claims 真数值等式 2+3=5 冲突（双注册坑·
    bit-identical 回归）。code_problem 条件等式 "如果X等于Y" 经 cue_type_of==ARITH_EQUALS_CUE **单源**复用
    → CMP_EQ·不经 _COMPARISON_OP_WORDS（避坑·test_code_problem_equality 守 code 侧）。
    """
    # 等于 非比较 OP：[3, 等于, 5] 窗口不抽（若 等于 误入比较族·会抽 (3,CMP_EQ,5) 假声明）
    assert extract_comparison_claims(["3", "等于", "5"], lang=LANG_ZH) == [], (
        "等于 非比较 OP·不抽比较声明（避与 numeric 等式双注册冲突）")
    # comparison_op_of(等于/equals) 仍返 None（未入 _COMPARISON_OP_WORDS·单源归刀B ARITH_EQUALS_CUE）
    assert comparison_op_of("等于", LANG_ZH) is None
    assert comparison_op_of("equals", LANG_EN) is None


def test_extract_comparison_claims_en():
    """EN：5 greater_than 3 → [(5, CMP_GT, 3)]（lang 隔离·EN 词表命中）。"""
    claims = extract_comparison_claims(["5", "greater_than", "3"], lang=LANG_EN)
    assert claims == [(5, CMP_GT, 3)]


def test_extract_comparison_claims_boundary_insufficient():
    """边界：左/右式不足（OP 在首/末）→ 跳·守反统计（不凑配）。"""
    # OP 在 index 0（无左式）→ 跳
    assert extract_comparison_claims(["大于", "5"], lang=LANG_ZH) == [], "无左式 → 跳"
    # OP 在末尾（无右式）→ 跳
    assert extract_comparison_claims(["5", "大于"], lang=LANG_ZH) == [], "无右式 → 跳"


def test_extract_comparison_claims_non_match_skipped():
    """不匹配模式（非数字）→ 跳·守反统计契约。"""
    # 左式非数字
    assert extract_comparison_claims(["苹果", "大于", "5"], lang=LANG_ZH) == [], "左式非数字 → 跳"
    # 右式非数字
    assert extract_comparison_claims(["5", "大于", "猫"], lang=LANG_ZH) == [], "右式非数字 → 跳"
    # 中文数字 defer
    assert extract_comparison_claims(["三", "大于", "五"], lang=LANG_ZH) == [], "中文数字 defer → 跳"


def test_extract_comparison_claims_negative_numbers():
    """负数 operand：-5 大于 -3 → [(-5, CMP_GT, -3)]（_parse_int_token 解析负号）。"""
    claims = extract_comparison_claims(["-5", "大于", "-3"], lang=LANG_ZH)
    assert claims == [(-5, CMP_GT, -3)], f"got {claims}"


def test_extract_comparison_claims_neighbor_op_skipped():
    """左/右邻也是比较 OP → 跳（连用 OP·锚定歧义·首版保守跳·同 extract_cues:66）。"""
    # "大于 大于 5"（左邻是 OP）→ 跳
    assert extract_comparison_claims(["大于", "大于", "5"], lang=LANG_ZH) == [], "左邻 OP → 跳"
    # "5 大于 小于"（右邻是 OP）→ 跳
    assert extract_comparison_claims(["5", "大于", "小于"], lang=LANG_ZH) == [], "右邻 OP → 跳"


def test_extract_comparison_claims_multi_claims():
    """多声明同段：两个比较 OP → 两声明（逐 OP 独立·确定性序）。"""
    # "5 大于 3 小于 8" → 大于在 index1：(5,GT,3)；小于在 index3：(3,LT,8)
    claims = extract_comparison_claims(["5", "大于", "3", "小于", "8"], lang=LANG_ZH)
    assert claims == [(5, CMP_GT, 3), (3, CMP_LT, 8)], f"got {claims}"


def test_extract_comparison_claims_no_op_empty():
    """无比较 OP → 空（无比较声明·守反统计）。"""
    assert extract_comparison_claims(["5", "3"], lang=LANG_ZH) == []


def test_extract_comparison_claims_empty_tokens():
    """空 tokens → 空（确定性·不崩）。"""
    assert extract_comparison_claims([], lang=LANG_ZH) == []


def test_extract_comparison_claims_gated_off_empty():
    """CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归（同 extract_numeric_claims_gated 范式）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = False
    try:
        claims = extract_comparison_claims_gated(["5", "大于", "3"], lang=LANG_ZH)
        assert claims == [], "gate OFF 返空（bit-identical）"
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_extract_comparison_claims_gated_on():
    """CUE_EXTRACTOR_MODE ON → 提取非空（生产路径）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        claims = extract_comparison_claims_gated(["5", "大于", "3"], lang=LANG_ZH)
        assert claims == [(5, CMP_GT, 3)]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_extract_comparison_claims_pure_int():
    """claims 全纯整数（assert_int 守·无浮点·cmp_opcode 亦纯整）。"""
    claims = extract_comparison_claims(["5", "大于", "3"], lang=LANG_ZH)
    for (ln, cmp, rn) in claims:
        assert isinstance(ln, int) and not isinstance(ln, bool)
        assert isinstance(cmp, int) and not isinstance(cmp, bool)
        assert isinstance(rn, int) and not isinstance(rn, bool)


# ============ 件③ 消费者（comparison_proof_fn_factory·构造性检查层·cross_compare） ============

def test_comparison_proof_fn_gt_hold_returns_1():
    """5 大于 3 → cross_compare(5,1,3,1)=1>0 → 1（verified·构造性检查通过）。"""
    fn = comparison_proof_fn_factory(claims=[(5, CMP_GT, 3)])
    assert fn(None, None, None) == 1


def test_comparison_proof_fn_gt_violation_returns_0():
    """3 大于 5 → cross_compare=−1·GT 须 >0 → 违反 → 0（mismatch）。"""
    fn = comparison_proof_fn_factory(claims=[(3, CMP_GT, 5)])
    assert fn(None, None, None) == 0


def test_comparison_proof_fn_lt_paths():
    """小于：3 小于 5 → 1（sign=−1<0 ✓）；5 小于 3 → 0（sign=1·LT 须<0 违）。"""
    assert comparison_proof_fn_factory(claims=[(3, CMP_LT, 5)])(None, None, None) == 1
    assert comparison_proof_fn_factory(claims=[(5, CMP_LT, 3)])(None, None, None) == 0


def test_comparison_proof_fn_ge_paths():
    """不小于（≥）：5 不小于 3 → 1（sign=1≥0 ✓）；5 不小于 5 → 1（sign=0≥0 ✓·等值满足 GE）；3 不小于 5 → 0。"""
    assert comparison_proof_fn_factory(claims=[(5, CMP_GE, 3)])(None, None, None) == 1
    assert comparison_proof_fn_factory(claims=[(5, CMP_GE, 5)])(None, None, None) == 1, "等值满足 ≥"
    assert comparison_proof_fn_factory(claims=[(3, CMP_GE, 5)])(None, None, None) == 0


def test_comparison_proof_fn_le_paths():
    """不大于（≤）：3 不大于 5 → 1（sign=−1≤0 ✓）；5 不大于 5 → 1（等值满足 LE）；5 不大于 3 → 0。"""
    assert comparison_proof_fn_factory(claims=[(3, CMP_LE, 5)])(None, None, None) == 1
    assert comparison_proof_fn_factory(claims=[(5, CMP_LE, 5)])(None, None, None) == 1, "等值满足 ≤"
    assert comparison_proof_fn_factory(claims=[(5, CMP_LE, 3)])(None, None, None) == 0


def test_comparison_proof_fn_multi_mixed():
    """多声明混合：一真一假 → 0（任一违反即 mismatch·短路语义·镜像 numeric）。"""
    fn = comparison_proof_fn_factory(claims=[(5, CMP_GT, 3), (3, CMP_GT, 5)])
    # 第一 5>3 真·第二 3>5 假 → 0
    assert fn(None, None, None) == 0


def test_comparison_proof_fn_multi_all_hold():
    """多声明全真 → 1。"""
    fn = comparison_proof_fn_factory(claims=[(5, CMP_GT, 3), (3, CMP_LT, 8)])
    assert fn(None, None, None) == 1


def test_comparison_proof_fn_empty_returns_none():
    """claims 空 → None（vacate·无比较声明可验·诚实退场·非 pass·非 theater）。"""
    fn = comparison_proof_fn_factory(claims=[])
    assert fn(None, None, None) is None


def test_comparison_proof_fn_deterministic():
    """确定性：同输入同输出（cross_compare 交叉积·bit-identical）。"""
    fn = comparison_proof_fn_factory(claims=[(5, CMP_GT, 3)])
    r1 = fn(None, None, None)
    r2 = fn(None, None, None)
    assert r1 == r2 == 1


def test_comparison_proof_fn_negative_numbers():
    """负数：-3 大于 -5 → cross_compare(-3,1,-5,1)=sign(−3+5)=sign(2)=1>0 → 1。"""
    fn = comparison_proof_fn_factory(claims=[(-3, CMP_GT, -5)])
    assert fn(None, None, None) == 1


def test_comparison_proof_fn_defensive_copy():
    """factory 防御性拷贝 claims（镜像 numeric P2-3）：caller 后续 mutation 不改已造 fn 行为。"""
    claims = [(5, CMP_GT, 3)]
    fn = comparison_proof_fn_factory(claims=claims)
    claims.append((3, CMP_GT, 5))   # 假声明·若 fn 持引用会致 fn()→0
    assert fn(None, None, None) == 1, "factory 须拷贝·caller mutation 不影响 fn（防 alias bug）"


def test_comparison_proof_uses_cross_compare():
    """cross_compare 真比较消费者：proof_fn 行为与 cross_compare 一致（反 theater·机制获真消费者）。"""
    from pure_integer_ai.crosscut.integer.compare import cross_compare
    # 5 大于 3：cross_compare(5,1,3,1)=1·GT 须>0 → 一致
    assert cross_compare(5, 1, 3, 1) == 1
    assert comparison_proof_fn_factory(claims=[(5, CMP_GT, 3)])(None, None, None) == 1
    # 3 不小于 5：cross_compare(3,1,5,1)=−1·GE 须≥0 → 违反一致
    assert cross_compare(3, 1, 5, 1) == -1
    assert comparison_proof_fn_factory(claims=[(3, CMP_GE, 5)])(None, None, None) == 0


# ============ 件③ e2e（_run_comparison_verify_round·COMPARISON_PROOF_MODE·SELF_PRODUCED） ============

def test_run_comparison_verify_round_reward_1_self_produced():
    """e2e：比较声明 item（5>3）+ CUE+COMPARISON gates ON → reward=1 + verify_source=SELF_PRODUCED。

    镜像 test_run_numeric_verify_round_reward_1_self_produced 范式·比较声明 self-contained
    （无需 token→ConceptRef resolve·无需 backend query·比数值更简）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_cmp = gates.COMPARISON_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.COMPARISON_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        item = _comparison_item(claim_true=True)   # 5 大于 3
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is not None, "比较 verify round 须产 episode（gates ON + 比较 cue·路由应走 _run_comparison_verify_round）"
        assert res.episode.reward == 1, "5>3 → reward=1（比较声明序一致·构造性检查通过）"
        assert res.episode.verify_source == VERIFY_SOURCE_SELF_PRODUCED, (
            f"_run_comparison_verify_round 须标 SELF_PRODUCED（数 single-source·构造性检查非验证）"
            f"·got verify_source={res.episode.verify_source}")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.COMPARISON_PROOF_MODE = saved_cmp


def test_run_comparison_verify_round_reward_0_on_violation():
    """e2e：假声明 item（3>5）+ gates ON → reward=0（违反·构造性检查未过·veto）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_cmp = gates.COMPARISON_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.COMPARISON_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        item = _comparison_item(claim_true=False)   # 3 大于 5（假）
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is not None
        assert res.episode.reward == 0, "3>5 假 → reward=0（违反·veto）"
        assert res.episode.verify_source == VERIFY_SOURCE_SELF_PRODUCED
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.COMPARISON_PROOF_MODE = saved_cmp


def test_run_comparison_verify_round_gate_off_no_episode():
    """COMPARISON_PROOF_MODE OFF → 路由不走·比较 item 走正常语言 episode_loop（bit-identical·零行为变）。

    比较 item 单段 struct_ref 孤立（len<2）→ 正常路径返空 RoundResult（无 comparison verify episode）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_cmp = gates.COMPARISON_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.COMPARISON_PROOF_MODE = False   # OFF
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        item = _comparison_item(claim_true=True)
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_SELF_PRODUCED, (
            "COMPARISON_PROOF_MODE OFF → 不产 SELF_PRODUCED comparison verify episode（bit-identical）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.COMPARISON_PROOF_MODE = saved_cmp


def test_run_comparison_verify_round_le_claim():
    """e2e：不大于声明（5 不大于 5·等值满足 LE）→ reward=1（GE/LE 等值路径）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_cmp = gates.COMPARISON_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.COMPARISON_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        item = _comparison_item(tokens=["5", "不大于", "5"], claim_true=True)
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is not None
        assert res.episode.reward == 1, "5 不大于 5（等值满足 LE）→ reward=1"
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.COMPARISON_PROOF_MODE = saved_cmp


# ============ 件④ capability_exam 消费 + Layer0 反 theater ============

def test_capability_exam_runs_with_comparison_item(tmp_path):
    """capability_exam + 比较声明 item → 产 report（不崩·layer0 6 key 齐·additive 字段）。

    **诚实 scope**（同 test_capability_exam_runs_with_numeric_item）：capability_exam 默认 teacher=None →
    reward 阶段不激活 → 语言域 comparison verify episode 不被 formal_train stage loop 收集。故此处只验
    capability_exam 不崩 + layer0 字段齐·**不**断 self_produced>0。comparison verify episode 生产 +
    SELF_PRODUCED 标记由直调 e2e（test_run_comparison_verify_round_*）守。
    """
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "cmp"), run_id="cmp_1")
        report = run_capability_exam(
            cfg, [_comparison_item(claim_true=True)],
            backend=b, runner=DefaultRoundRunner())
        assert isinstance(report, CapabilityReport)
        assert set(report.layer0_attribution.keys()) == {
            "external_verified", "self_produced_check_passed", "self_produced_check_failed",
            "anchor_satisfied", "anchor_violated", "total",
        }
    finally:
        gates.TRAINING_MODE = saved


def test_project_layer0_counts_comparison_self_produced():
    """capability_exam 消费者（project_layer0）正确计数比较 SELF_PRODUCED episode（构造 episode·直测消费）。

    构造 result.episodes 含 comparison verify SELF_PRODUCED reward=1 episode → project_layer0 计
    self_produced_check_passed（非 external_verified·反 theater）。
    """
    from types import SimpleNamespace
    from pure_integer_ai.cognition.shared.types import Episode, VERIFY_SOURCE_EXTERNAL
    from pure_integer_ai.experiments.capability_exam import project_layer0
    result = SimpleNamespace(episodes=[
        Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED),   # comparison verify episode
        Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL),        # arith vm_proof 对照
    ])
    counts = project_layer0(result)
    assert counts["self_produced_check_passed"] == 1, "comparison SELF_PRODUCED reward=1 计检查通过"
    assert counts["external_verified"] == 1, "仅 EXTERNAL 计构造性验证（反 theater·SELF_PRODUCED 不计）"
    assert counts["anchor_violated"] == 1, "SELF_PRODUCED 违外部锚门（全自产不准停）"
    assert counts["anchor_satisfied"] == 1, "EXTERNAL 满足锚门"
    assert counts["total"] == 2


def test_comparison_episode_anti_theater_not_external_verified():
    """**反 theater 端到端**：比较 verify episode reward=1（序一致）但非构造性验证 + 违锚门。

    SELF_PRODUCED reward=1（5>3 序一致·检查通过）≠ 构造性验证（数 single-source）。
    对照：is_constructive_verification=False + external_anchor_satisfied=False（全自产不准停）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_cmp = gates.COMPARISON_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.COMPARISON_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        res = r.run_round_full(ctx, _comparison_item(claim_true=True), STAGE3_REWARD, 0)
        ep = res.episode
        assert ep is not None and ep.reward == 1
        assert is_constructive_verification(ep) is False, (
            "SELF_PRODUCED 不计构造性验证（反 theater）·即使 reward=1（检查通过非验证）")
        assert external_anchor_satisfied(ep) is False, (
            "SELF_PRODUCED 全自产不准停（违外部锚门）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.COMPARISON_PROOF_MODE = saved_cmp


# ============ 诚实边界（docstring 标注） ============

def test_comparison_proof_is_constructive_check_not_verification():
    """诚实标注：comparison_proof 是构造性检查层（cross_compare 确定性）·非构造性验证（须 R6·single-source）。"""
    import pure_integer_ai.training.comparison_proof as mod
    docstring = mod.__doc__ or ""
    assert "构造性检查" in docstring, "模块 docstring 须标构造性检查层"
    assert "构造性验证" in docstring, "须诚实标非构造性验证（数 single-source·须 R6 升验证）"
    assert "SELF_PRODUCED" in docstring or "Layer0" in docstring, "须标 Layer0 SELF_PRODUCED"


def test_comparison_proof_uses_cross_compare_rationale():
    """诚实标注：为何用 cross_compare 非裸 sign(left−right)（compare.py 铁律·零误差·真比较消费者）。"""
    import pure_integer_ai.training.comparison_proof as mod
    docstring = mod.__doc__ or ""
    assert "cross_compare" in docstring, "须标用 cross_compare（比序唯一零误差路径）"


def test_comparison_proof_option_a_no_persist():
    """Option A 诚实标注：比较声明不入图（闭包传·同刀 A/B 防结构发现污染 + emergence 干扰）。"""
    import pure_integer_ai.training.comparison_proof as mod
    docstring = mod.__doc__ or ""
    assert "不入图" in docstring or "闭包" in docstring, "须标 Option A（比较声明不入图·闭包传）"


def test_comparison_proof_proposition_value_defer():
    """诚实标注：doc"命题值比序"(B) defer（须 ref→surface 基建·concept_index 无反查·本刀做 (A) 字面数值比序）。"""
    import pure_integer_ai.training.comparison_proof as mod
    docstring = mod.__doc__ or ""
    assert "命题值" in docstring or "(B)" in docstring or "defer" in docstring, (
        "须标 (B) 命题值比序 defer 边界")


# ============ gate 默认 OFF（bit-identical 守 CI=生产） ============

def test_comparison_proof_mode_default_off():
    """COMPARISON_PROOF_MODE 默认 OFF·守 CI 回归 bit-identical（路由不走·既有语言域 episode_loop 不变）。"""
    import importlib
    import pure_integer_ai.config.gates as g
    importlib.reload(g)
    assert g.COMPARISON_PROOF_MODE is False, "COMPARISON_PROOF_MODE 默认 OFF 守 bit-identical"
