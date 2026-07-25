"""tests.test_layer0_anchor — Layer0 外部锚门测试套件（停止决策守门·防 cue 自产边 theater）。

测 Layer0 三件 + 一消费者（分层墙 §八b "找到就停纪律"·构造性检查≠构造性验证的系统级强制）：
  ① 溯源枚举 + Episode.verify_source 字段（default NONE·向后兼容）
  ② 守门函数（external_anchor_satisfied / is_constructive_verification / count_layer0）
  ③ 通道接线（_run_verify_round→EXTERNAL·occurrence-order adapter→SELF_PRODUCED）
  ④ capability_exam 消费者（project_layer0 + report.layer0_attribution + to_json + bit-identical）

反 theater 核心（端到端）：SELF_PRODUCED reward=1（时序无环检查通过）不被计构造性验证 +
external_anchor_satisfied=False（全自产不准停）·对照 EXTERNAL reward=1 → 双 True。

诚实边界：Layer0 只标记+守门·不提供 R6（时序升验证 defer）·stable≠correct·不碰 #355（标记在 Episode 非边）。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.cognition.shared.types import (
    Episode, VERIFY_SOURCE_NONE, VERIFY_SOURCE_EXTERNAL, VERIFY_SOURCE_SELF_PRODUCED,
    CodeSpec, MODALITY_ARITH, DOMAIN_MATH, LANG_NONE, LANG_ZH,
)
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, SOURCE_MATH
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, DefaultRoundRunner, make_train_context,
)
from pure_integer_ai.experiments.capability_exam import (
    run_capability_exam, CapabilityReport, project_layer0,
)
from pure_integer_ai.cognition.result.layer0_anchor import (
    external_anchor_satisfied, is_constructive_verification, count_layer0,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderProtocol,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from tests.test_experiments import _causal_multi_sent_item, flat_floors
from tests.test_stage9_arith_observe import _arith_item


# ============ 件① 溯源枚举 + Episode 字段（向后兼容） ============

def test_verify_source_enum_values():
    """溯源枚举值固定（NONE=0 / EXTERNAL=1 / SELF_PRODUCED=2·纯整数·bit-identical 基）。"""
    assert (VERIFY_SOURCE_NONE, VERIFY_SOURCE_EXTERNAL, VERIFY_SOURCE_SELF_PRODUCED) == (0, 1, 2)


def test_episode_verify_source_default_none():
    """Episode.verify_source default NONE（向后兼容·既有 Episode 构造零改·reward 通道不声称构造性）。"""
    ep = Episode()
    assert ep.verify_source == VERIFY_SOURCE_NONE


def test_episode_verify_source_settable():
    """Episode.verify_source 可设（通道接线用·formal_train 填 EXTERNAL/SELF_PRODUCED）。"""
    assert Episode(verify_source=VERIFY_SOURCE_EXTERNAL).verify_source == VERIFY_SOURCE_EXTERNAL
    assert Episode(verify_source=VERIFY_SOURCE_SELF_PRODUCED).verify_source == VERIFY_SOURCE_SELF_PRODUCED


# ============ 件② 守门函数（external_anchor_satisfied 三态） ============

def test_external_anchor_satisfied_none_passes():
    """NONE（reward 通道·judge 产）→ True（不声称构造性·锚门对其放行·无意义）。"""
    ep = Episode(reward=1, verify_source=VERIFY_SOURCE_NONE)
    assert external_anchor_satisfied(ep) is True


def test_external_anchor_satisfied_external_passes():
    """EXTERNAL（vm_proof R6 独立源）→ True（满足外部锚门·可驱动停止决策）。"""
    ep = Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL)
    assert external_anchor_satisfied(ep) is True


def test_external_anchor_satisfied_self_produced_blocks():
    """SELF_PRODUCED（time_seq 自产）→ False（全自产不准停·防 cue 自产边 theater·§八b 核心）。"""
    ep = Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED)
    assert external_anchor_satisfied(ep) is False


def test_external_anchor_satisfied_self_produced_blocks_even_on_fail():
    """SELF_PRODUCED reward==0（检查未过·有环）→ 仍 False（全自产即违锚门·不问检查过否不准停）。"""
    ep = Episode(reward=0, verify_source=VERIFY_SOURCE_SELF_PRODUCED)
    assert external_anchor_satisfied(ep) is False


def test_external_anchor_satisfied_getattr_fallback():
    """getattr 防 fake Episode（无 verify_source 属性→退化 NONE→放行·镜像 _door_vetoed getattr 范式）。"""
    fake = SimpleNamespace(reward=1)   # 无 verify_source
    assert external_anchor_satisfied(fake) is True


# ============ 件② 守门函数（is_constructive_verification 三态） ============

def test_is_constructive_verification_external_reward_positive():
    """EXTERNAL + reward>0 → True（真构造性验证·构造性检查 + R6 外部源两齐）。"""
    ep = Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL)
    assert is_constructive_verification(ep) is True


def test_is_constructive_verification_self_produced_not_verification():
    """**反 theater 核心**：SELF_PRODUCED + reward>0（时序检查通过）→ False（构造性检查·非验证）。
    聚合计数构造性验证须用此·非 reward>0（reward>0 含本类·计入即 theater）。"""
    ep = Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED)
    assert is_constructive_verification(ep) is False


def test_is_constructive_verification_external_reward_zero_false():
    """EXTERNAL + reward==0（验证失败）→ False（无论来源·reward==0 即非验证）。"""
    ep = Episode(reward=0, verify_source=VERIFY_SOURCE_EXTERNAL)
    assert is_constructive_verification(ep) is False


def test_is_constructive_verification_none_not_verification():
    """NONE + reward>0（reward 通道·经验统计）→ False（不声称构造性）。"""
    ep = Episode(reward=1, verify_source=VERIFY_SOURCE_NONE)
    assert is_constructive_verification(ep) is False


# ============ 件② count_layer0（汇总计数·确定性·纯整数） ============

def test_count_layer0_mixed_classification():
    """混合 episodes → count_layer0 正确分类计数（反 theater：SELF_PRODUCED reward>1 不入 external_verified）。"""
    episodes = [
        Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL),           # external_verified
        Episode(reward=0, verify_source=VERIFY_SOURCE_EXTERNAL),           # EXTERNAL 验证失败
        Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED),      # self_produced_check_passed
        Episode(reward=0, verify_source=VERIFY_SOURCE_SELF_PRODUCED),      # self_produced_check_failed
        Episode(reward=1, verify_source=VERIFY_SOURCE_NONE),               # reward 通道
    ]
    counts = count_layer0(episodes)
    assert counts["external_verified"] == 1, "仅 EXTERNAL+reward>0 计构造性验证"
    assert counts["self_produced_check_passed"] == 1, "SELF_PRODUCED+reward>0 计检查通过（非验证）"
    assert counts["self_produced_check_failed"] == 1, "SELF_PRODUCED+reward==0 计检查未过"
    assert counts["anchor_satisfied"] == 3, "NONE(1)+EXTERNAL(2·reward 不问)=3 满足锚门"
    assert counts["anchor_violated"] == 2, "SELF_PRODUCED(2) 全违锚门（不问 reward）"
    assert counts["total"] == 5


def test_count_layer0_empty():
    """空 episodes → 全 0 计数（确定性·capability_exam 无 episode 时不崩）。"""
    counts = count_layer0([])
    assert counts == {
        "external_verified": 0, "self_produced_check_passed": 0,
        "self_produced_check_failed": 0, "anchor_satisfied": 0,
        "anchor_violated": 0, "total": 0,
    }


def test_count_layer0_keys_fixed_order():
    """count_layer0 返 dict key 固定序（bit-identical·to_json sort_keys 双保险）。"""
    counts = count_layer0([Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL)])
    assert list(counts.keys()) == [
        "external_verified", "self_produced_check_passed", "self_produced_check_failed",
        "anchor_satisfied", "anchor_violated", "total",
    ]


def test_count_layer0_pure_int():
    """计数全纯整数（assert_int 守·无浮点）。"""
    counts = count_layer0([
        Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL),
        Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED),
    ])
    for v in counts.values():
        assert isinstance(v, int) and not isinstance(v, bool)


# ============ 件③ 通道接线（EXTERNAL arith e2e） ============

def test_run_verify_round_sets_external_source():
    """_run_verify_round（arith·vm_proof 用 spec.expected R6）→ Episode.verify_source=EXTERNAL。

    e2e：arith item（Sigma + spec f(5)=15）→ run_round_full → verify round → reward=1 +
    verify_source=EXTERNAL（真构造性验证·R6 独立源·可驱动停止决策）。镜像 test_stage9 e2e 范式。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1, "Sigma f(5)=15 → reward=1（vm_proof 验过）"
    assert res.episode.verify_source == VERIFY_SOURCE_EXTERNAL, \
        "_run_verify_round 须标 EXTERNAL（vm_proof R6 独立源·真构造性验证）"
    # 反 theater：该 episode 过 Layer0 守门（可计构造性验证 + 可驱动停止决策）
    assert is_constructive_verification(res.episode) is True
    assert external_anchor_satisfied(res.episode) is True


# ============ 件③ 通道接线（EXTERNAL task-driven generate e2e·P1-1 修） ============

def test_task_driven_generate_sets_external_source(tmp_path, flat_floors):
    """task-driven generate episode（arith·execute vs spec.expected R6）→ EXTERNAL·计 external_verified。

    e2e（2 审 P1-1 修验）：2 arith items（同 Sigma shape·不同 spec）→ capability_exam → discovery +
    task-driven generate → episode verify_source=EXTERNAL（同 _run_verify_round·spec.expected R6 外部源·
    Mode A 构造性验证）→ external_verified > 0（task-driven 验过数计入·不再少计）。
    """
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "td"), run_id="td_1")
        report = run_capability_exam(
            cfg, [
                _arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((5,), (15, 1))]),
                _arith_item("lambda n: Sigma(1, n, i)", [CodeSpec((10,), (55, 1))]),
            ],
            backend=b, runner=DefaultRoundRunner())
        ext = report.layer0_attribution["external_verified"]
        # P1-1 修：task-driven 标 EXTERNAL → external_verified 含 task-driven 验过数（不再少计归 NONE）
        assert ext > 0, (
            f"task-driven generate 须产 EXTERNAL episode 计 external_verified（P1-1 修）·got layer0={report.layer0_attribution}")
    finally:
        gates.TRAINING_MODE = saved


# ============ 件③ 通道接线（SELF_PRODUCED 语言时序 e2e） ============

def test_occurrence_order_verify_round_sets_self_produced_source():
    """occurrence 顺序 cue 核验只产生 SELF_PRODUCED 证据。

    e2e：语言 item（"a 然后 b"·时序 cue）+ CUE_EXTRACTOR_MODE+TIME_SEQ_PROOF_MODE ON →
    run_round_full → 时序 verify round → reward=1（Kahn 无环）+ verify_source=SELF_PRODUCED。
    **反 theater**：reward=1 但 is_constructive_verification=False + external_anchor_satisfied=False
    （全自产不准停·构造性检查非验证·防 cue 自产边 theater）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_ts = gates.TIME_SEQ_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.TIME_SEQ_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        install_language_graph_protocols(
            ctx,
            occurrence_protocol=OccurrenceProtocol((99001,)),
            occurrence_order_protocol=OccurrenceOrderProtocol((99002,)),
        )
        r = DefaultRoundRunner()
        item = CollectedItem(
            tokens=["a", "然后", "b"],
            raw_text="a然后b",
            role_seq=[1, 1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
            lang=LANG_ZH,
        )
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is not None, (
            "occurrence-order verify round 须产 episode")
        assert res.episode.verify_source == VERIFY_SOURCE_SELF_PRODUCED, (
            f"occurrence-order adapter 须标 SELF_PRODUCED（cue 对 single-source·"
            f"构造性检查非验证）·got verify_source={res.episode.verify_source}")
        # 反 theater 核心：reward 可能 1（无环检查通过）但非构造性验证 + 违锚门
        assert is_constructive_verification(res.episode) is False, (
            "SELF_PRODUCED 不计构造性验证（反 theater）·即使 reward>0（检查通过非验证）")
        assert external_anchor_satisfied(res.episode) is False, (
            "SELF_PRODUCED 全自产不准停（违外部锚门）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.TIME_SEQ_PROOF_MODE = saved_ts


# ============ 件④ capability_exam 消费者（project_layer0） ============

def test_project_layer0_reads_episodes():
    """project_layer0 读 result.episodes → count_layer0（确定性·纯整数）。"""
    result = SimpleNamespace(episodes=[
        Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL),
        Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED),
    ])
    counts = project_layer0(result)
    assert counts["external_verified"] == 1
    assert counts["self_produced_check_passed"] == 1
    assert counts["anchor_violated"] == 1


def test_project_layer0_no_episodes_field():
    """result 无 episodes 属性 → 退化空 [] → 全 0 计数（getattr 守·不崩）。"""
    result = SimpleNamespace()
    counts = project_layer0(result)
    assert counts["total"] == 0


def test_capability_exam_populates_layer0_attribution(tmp_path, flat_floors):
    """run_capability_exam → report.layer0_attribution 填（additive 字段·project_layer0 消费 result.episodes）。"""
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "l0"), run_id="l0_1")
        report = run_capability_exam(
            cfg, [_causal_multi_sent_item()],
            backend=b, runner=DefaultRoundRunner())
        # layer0_attribution 填（6 key 齐·纯整数）
        assert isinstance(report, CapabilityReport)
        assert set(report.layer0_attribution.keys()) == {
            "external_verified", "self_produced_check_passed", "self_produced_check_failed",
            "anchor_satisfied", "anchor_violated", "total",
        }
        # 语言因果语料 + TIME_SEQ_PROOF_MODE OFF（默认）→ 无 SELF_PRODUCED episode
        assert report.layer0_attribution["self_produced_check_passed"] == 0
        assert report.layer0_attribution["anchor_violated"] == 0
        # total 与 episode 数一致（collect_episodes=True 强制）
        assert report.layer0_attribution["total"] >= 0
    finally:
        gates.TRAINING_MODE = saved


def test_capability_exam_to_json_contains_layer0(tmp_path, flat_floors):
    """CapabilityReport.to_json 含 layer0_attribution（additive·bit-identical·key sort）。"""
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "l0j"), run_id="l0j_1")
        report = run_capability_exam(
            cfg, [_causal_multi_sent_item()],
            backend=b, runner=DefaultRoundRunner())
        j = report.to_json()
        assert "layer0_attribution" in j
        assert isinstance(j["layer0_attribution"], dict)
        assert "external_verified" in j["layer0_attribution"]
    finally:
        gates.TRAINING_MODE = saved


def test_capability_exam_layer0_bit_identical(tmp_path, flat_floors):
    """两跑 → layer0_attribution 一致（bit-identical·count_layer0 确定性）。"""
    saved = gates.TRAINING_MODE

    def run_once(run_id: str):
        gates.TRAINING_MODE = True
        try:
            b = DictBackend()
            cfg = FormalTrainConfig(run_dir=str(tmp_path / run_id), run_id=run_id)
            report = run_capability_exam(
                cfg, [_causal_multi_sent_item()],
                backend=b, runner=DefaultRoundRunner())
            return report.to_json()["layer0_attribution"]
        finally:
            gates.TRAINING_MODE = saved

    l1 = run_once("l0bi1")
    l2 = run_once("l0bi2")
    assert json.dumps(l1, sort_keys=True) == json.dumps(l2, sort_keys=True), (
        "两跑 layer0_attribution 不一致·违 bit-identical")


# ============ 反 theater 端到端（SELF_PRODUCED vs EXTERNAL 对照） ============

def test_anti_theater_self_produced_not_counted_as_verification():
    """**反 theater 端到端**：时序无环检查通过（reward=1 SELF_PRODUCED）≠ 构造性验证。

    对照：
      - SELF_PRODUCED reward=1（time_seq Kahn 无环·检查通过）：is_constructive_verification=False +
        external_anchor_satisfied=False（全自产不准停·防 cue 自产边 theater）。
      - EXTERNAL reward=1（vm_proof R6·真验证）：双 True。
    count_layer0 把前者归 self_produced_check_passed（非 external_verified）·聚合计数守。
    """
    self_produced_pass = Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED)
    external_pass = Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL)
    # 对照
    assert is_constructive_verification(self_produced_pass) is False
    assert external_anchor_satisfied(self_produced_pass) is False
    assert is_constructive_verification(external_pass) is True
    assert external_anchor_satisfied(external_pass) is True
    # count_layer0 分桶正确（SELF_PRODUCED 不入 external_verified）
    counts = count_layer0([self_produced_pass, external_pass])
    assert counts["external_verified"] == 1, "仅 EXTERNAL 计构造性验证"
    assert counts["self_produced_check_passed"] == 1, "SELF_PRODUCED 归检查通过（非验证）"


# ============ 诚实边界（docstring 标注） ============

def test_layer0_anchor_docstring_honest_boundary():
    """诚实标注：构造性检查≠构造性验证 / 不提供 R6 / stable≠correct（模块 docstring 守）。"""
    import pure_integer_ai.cognition.result.layer0_anchor as mod
    doc = mod.__doc__ or ""
    assert "构造性检查" in doc and "构造性验证" in doc, "须标构造性检查≠构造性验证"
    assert "不提供 R6" in doc or "不提供" in doc, "须诚实标不提供 R6（只标+守门）"
    assert "stable" in doc.lower(), "须标 stable≠correct"


def test_layer0_anchor_no_reward_mutation():
    """Layer0 守门函数只读 Episode（reward / verify_source）·不写·不碰 reward 通道。"""
    ep = Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED)
    snapshot_reward = ep.reward
    snapshot_src = ep.verify_source
    external_anchor_satisfied(ep)
    is_constructive_verification(ep)
    count_layer0([ep])
    assert ep.reward == snapshot_reward, "守门函数不动 reward"
    assert ep.verify_source == snapshot_src, "守门函数不动 verify_source"
