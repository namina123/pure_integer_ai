"""全称量化 cue、开放世界 proof、external caller 和 Layer0 测试。"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.cognition.shared.types import (
    LANG_ZH, LANG_EN, MODALITY_LANGUAGE, VERIFY_SOURCE_EXTERNAL, VERIFY_SOURCE_SELF_PRODUCED,
)
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.storage.edge_store import (
    SOURCE_BARE_TEXT, SOURCE_CONCEPTNET, EPI_STRUCTURED, EPI_CUE,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_CAUSES
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, DefaultRoundRunner, make_train_context,
)
from pure_integer_ai.experiments.capability_exam import (
    run_capability_exam, CapabilityReport, project_layer0,
)
from pure_integer_ai.cognition.result.layer0_anchor import (
    is_constructive_verification, external_anchor_satisfied,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, UNIVERSAL_CUE, IS_A_CUE, CAUSES_CUE_FORWARD,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_universal_claims, extract_universal_claims_gated,
)
from pure_integer_ai.cognition.understanding.is_a import build_is_a_edge, bootstrap_is_a_edges
from pure_integer_ai.cognition.process.abstraction import (
    build_isa_ancestor_map, build_isa_ancestor_map_external,
)
from pure_integer_ai.training.universal_proof import universal_proof_fn_factory


def _universal_item(tokens=None, *, lang: int = LANG_ZH):
    """建全称量化语言 item（MODALITY_LANGUAGE·tokens 含 X 都是 Y）。"""
    toks = list(tokens) if tokens is not None else ["鸟", "都是", "动物"]
    return CollectedItem(
        tokens=toks,
        role_seq=[1] * len(toks),
        collect_type=COLLECT_CAUSES,
        source=SOURCE_BARE_TEXT,
        lang=lang,
    )


def _ensure_concept(ctx, surface: str):
    """concept_index.ensure 一个概念节点（返 ConceptRef·测试确定性 concept resolution 用）。"""
    return ctx.concept_index.ensure(
        surface, space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


# ============ 件① 词表/类型（cue_type_of·UNIVERSAL_CUE） ============

def test_cue_type_of_doushi_zh():
    """ZH 全称系词（都是）→ UNIVERSAL_CUE（exact 匹配·全称量化内涵分类子集锚）。"""
    assert cue_type_of("都是", LANG_ZH) == UNIVERSAL_CUE
    assert cue_type_of("全是", LANG_ZH) == UNIVERSAL_CUE


def test_cue_type_of_are_all_en():
    """EN 全称系词（are_all）→ UNIVERSAL_CUE。"""
    assert cue_type_of("are_all", LANG_EN) == UNIVERSAL_CUE


def test_cue_type_of_universal_distinct_from_is_a():
    """UNIVERSAL_CUE（都是）≠ IS_A_CUE（是一种/属于）·两独立 cue_type·零行为变（守既有语义）。"""
    assert cue_type_of("都是", LANG_ZH) == UNIVERSAL_CUE
    assert cue_type_of("是一种", LANG_ZH) == IS_A_CUE
    assert cue_type_of("属于", LANG_ZH) == IS_A_CUE   # 单数系词·非全称
    assert cue_type_of("导致", LANG_ZH) == CAUSES_CUE_FORWARD


def test_cue_type_of_universal_non_word():
    """非全称词 → None（守反统计契约·不凑配）。"""
    assert cue_type_of("鸟", LANG_ZH) is None
    assert cue_type_of("都是", LANG_EN) is None   # ZH 词在 EN lang 不命中（lang 隔离）


# ============ 件② 构造器（extract_universal_claims·独立函数） ============

def test_extract_universal_claims_basic():
    """鸟 都是 动物 → [(0, 2)]（紧邻 pair·child=tokens[0]·parent=tokens[2]·cue=都是 在 index 1）。"""
    claims = extract_universal_claims(["鸟", "都是", "动物"], lang=LANG_ZH)
    assert claims == [(0, 2)], f"got {claims}"


def test_extract_universal_claims_determiner_optional():
    """限定词"所有"前置不要求：所有 鸟 都是 动物 → child=鸟(idx1) parent=动物(idx3)（都是自含全称 force）。"""
    claims = extract_universal_claims(["所有", "鸟", "都是", "动物"], lang=LANG_ZH)
    assert claims == [(1, 3)], f"got {claims}"


def test_extract_universal_claims_quan_shi():
    """全是 系词：猫 全是 动物 → [(0, 2)]。"""
    claims = extract_universal_claims(["猫", "全是", "动物"], lang=LANG_ZH)
    assert claims == [(0, 2)]


def test_extract_universal_claims_en():
    """EN：birds are_all animals → [(0, 2)]（lang 隔离·EN 词表命中）。"""
    claims = extract_universal_claims(["birds", "are_all", "animals"], lang=LANG_EN)
    assert claims == [(0, 2)]


def test_extract_universal_claims_boundary():
    """边界 cue（无左/右）→ 跳·守反统计（不凑配）。"""
    # 都是 在 index 0（无左）→ 跳
    assert extract_universal_claims(["都是", "动物"], lang=LANG_ZH) == []
    # 都是 在末尾（无右）→ 跳
    assert extract_universal_claims(["鸟", "都是"], lang=LANG_ZH) == []


def test_extract_universal_claims_cue_neighbor_skipped():
    """左/右邻也是 cue → 跳（连用全称系词·锚定歧义·首版保守跳·同 extract_cues 反统计契约）。"""
    # 左邻是 cue（都是 都是 动物·左邻"都是"也是 cue）
    assert extract_universal_claims(["都是", "都是", "动物"], lang=LANG_ZH) == []


def test_extract_universal_claims_multi_claims():
    """多声明同段：两 都是 cue → 两 pair（逐 cue 独立·确定性序）。"""
    # 鸟 都是 动物 猫 都是 动物 → 两声明
    claims = extract_universal_claims(
        ["鸟", "都是", "动物", "猫", "都是", "动物"], lang=LANG_ZH)
    assert claims == [(0, 2), (3, 5)], f"got {claims}"


def test_extract_universal_claims_no_universal_empty():
    """无全称 cue → 空（守反统计）。"""
    assert extract_universal_claims(["鸟", "是一种", "动物"], lang=LANG_ZH) == []   # IS_A_CUE 非 UNIVERSAL


def test_extract_universal_claims_empty_tokens():
    """空 tokens → 空（确定性·不崩）。"""
    assert extract_universal_claims([], lang=LANG_ZH) == []


def test_extract_universal_claims_gated_off_empty():
    """CUE_EXTRACTOR_MODE OFF → 返空·bit-identical 守回归（同 extract_cues_gated 范式）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = False
    try:
        claims = extract_universal_claims_gated(["鸟", "都是", "动物"], lang=LANG_ZH)
        assert claims == [], "gate OFF 返空（bit-identical）"
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_extract_universal_claims_gated_on():
    """CUE_EXTRACTOR_MODE ON → 提取非空（生产路径·frozenset 第一源命中）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        claims = extract_universal_claims_gated(["鸟", "都是", "动物"], lang=LANG_ZH)
        assert claims == [(0, 2)]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


# ============ 件③ build_isa_ancestor_map_external（source filter·反 theater 核心） ============

def test_build_isa_ancestor_map_external_filters_source():
    """外部图只含 ConceptNet 边（source=CONCEPTNET+epistemic=STRUCTURED）·cue 边被滤·反 single-source theater。

    核心反 theater：若 cue 边混入外部图·"所有 X 都是 Y" 系词自产 (X,Y) → 验证平凡通过 = 自证闭环。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    bird = _ensure_concept(ctx, "鸟")
    animal = _ensure_concept(ctx, "动物")
    cat = _ensure_concept(ctx, "猫")
    # ConceptNet 外部边：鸟 IsA 动物（source=CONCEPTNET·epistemic=STRUCTURED）
    build_is_a_edge(ctx.edge_store, bird, animal,
                    source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=ctx.space_id)
    # cue 自产边：猫 IsA 动物（source=BARE_TEXT·epistemic=CUE·来源②系词提取）
    build_is_a_edge(ctx.edge_store, cat, animal,
                    source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=ctx.space_id)
    # 外部图：只 鸟（ConceptNet）·不含 猫（cue 滤掉）
    ext = build_isa_ancestor_map_external(ctx.backend, space_id=ctx.space_id)
    assert animal in ext.get(bird, set()), "ConceptNet 鸟→动物 须在外部图"
    assert cat not in ext, "cue 自产边（猫·EPI_CUE）须被 source filter 滤掉·反 theater"
    # 对照：混图（build_isa_ancestor_map）含两者
    mixed = build_isa_ancestor_map(ctx.backend, space_id=ctx.space_id)
    assert animal in mixed.get(bird, set()) and animal in mixed.get(cat, set()), "混图含全部 IS_A"


def test_build_isa_ancestor_map_external_empty_when_no_conceptnet():
    """无 ConceptNet 边（CI/生产 default）→ 外部图空 dict → 全 can't-verify（诚实降级·非 theater）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    bird = _ensure_concept(ctx, "鸟")
    animal = _ensure_concept(ctx, "动物")
    # 仅 cue 边（无 ConceptNet）
    build_is_a_edge(ctx.edge_store, bird, animal,
                    source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=ctx.space_id)
    ext = build_isa_ancestor_map_external(ctx.backend, space_id=ctx.space_id)
    assert ext == {}, "无 ConceptNet 边 → 外部图空（CI default·全 can't-verify·非平凡通过）"


# ============ 件③ 消费者（universal_proof_fn_factory·三值逻辑·构造性验证层） ============

def _ref(n: int):
    """造测试 ConceptRef (1, n)。"""
    return (1, n)


def test_universal_proof_fn_verified_returns_1():
    """parent ∈ ancestors(child) → 1（ConceptNet 外部断言 X⊆Y·真构造性验证通过）。"""
    ancestor_map = {_ref(1): {_ref(2)}}   # 鸟(1) IsA 动物(2)
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2))])
    assert fn(None, None, None) == 1


def test_universal_proof_fn_missing_path_returns_none():
    """开放世界中两个已知概念之间缺路径仍是未知，不能当作外部反证。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(4)}}
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(3))])
    assert fn(None, None, None) is None


def test_universal_proof_fn_ancestor_only_missing_path_returns_none():
    """parent 即使在其他祖先集中出现，缺少目标路径也不构成反证。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(4)}}
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(4))])
    assert fn(None, None, None) is None



def test_universal_proof_fn_cant_verify_property_returns_none():
    """祖先图没有声明目标路径时保持未知，不区分词面类别。"""
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(5))])
    assert fn(None, None, None) is None, "parent 非分类概念 → can't-verify（None·守属性全称 #479 墙）"


def test_universal_proof_fn_cant_verify_unknown_child_returns_none():
    """child 非分类概念（未在 ConceptNet）→ None（外部源不足·诚实降级·非证伪）。"""
    ancestor_map = {_ref(1): {_ref(2)}}   # 仅 鸟→动物·石头(6) 不在
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(6), _ref(2))])
    assert fn(None, None, None) is None


def test_universal_proof_fn_multi_all_verified():
    """多声明全 verified → 1。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(4)}}
    fn = universal_proof_fn_factory(
        ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2)), (_ref(3), _ref(4))])
    assert fn(None, None, None) == 1


def test_universal_proof_fn_multi_one_falsified_short_circuits():
    """多声明中任一显式反驳使合取为 0。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(4)}}
    fn = universal_proof_fn_factory(
        ancestor_map=ancestor_map,
        claims=[(_ref(1), _ref(2)), (_ref(1), _ref(3))],
        refuted_claims=[(_ref(1), _ref(3))],
    )
    assert fn(None, None, None) == 0


def test_universal_proof_fn_conflicted_claim_returns_none():
    """同一全称声明同时被支持和反驳时保持冲突未知。"""
    claim = (_ref(1), _ref(2))
    fn = universal_proof_fn_factory(
        ancestor_map={_ref(1): {_ref(2)}},
        claims=[claim],
        refuted_claims=[claim],
    )
    assert fn(None, None, None) is None


def test_universal_proof_fn_multi_cant_verify_no_falsify_returns_none():
    """多声明·有 can't-verify 无 falsified → None（最弱约束诚实·弃权非证伪）。"""
    # claim1 verified (鸟⊆动物)·claim2 can't-verify (鸟⊆飞·飞非分类)
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = universal_proof_fn_factory(
        ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2)), (_ref(1), _ref(5))])
    assert fn(None, None, None) is None


def test_universal_proof_fn_empty_returns_none():
    """claims 空 → None（vacate·诚实退场·非 pass·非 theater）。"""
    fn = universal_proof_fn_factory(ancestor_map={}, claims=[])
    assert fn(None, None, None) is None


def test_universal_proof_fn_deterministic():
    """确定性：同输入同输出（集合查询·bit-identical）。"""
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2))])
    assert fn(None, None, None) == fn(None, None, None) == 1


def test_universal_proof_fn_defensive_copy_claims():
    """factory 防御拷贝 claims（caller 后续 mutation 不改已造 fn 行为·同 numeric P2-3）。"""
    claims = [(_ref(1), _ref(2))]
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=claims)
    claims.append((_ref(3), _ref(4)))   # caller mutation·若持引用会改 fn
    assert fn(None, None, None) == 1, "factory 须拷贝 claims·caller mutation 不影响 fn"


def test_universal_proof_fn_defensive_copy_ancestor_map():
    """factory 防御拷贝 ancestor_map（caller 后续改 map 不改已造 fn·dict+set 深拷贝值集）。"""
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = universal_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2))])
    ancestor_map[_ref(1)].add(_ref(3))   # caller mutation·若持引用会改 fn 的 ancestors
    ancestor_map[_ref(7)] = {_ref(8)}
    assert fn(None, None, None) == 1, "factory 须拷贝 ancestor_map·caller mutation 不影响 fn"


# ============ 件③ e2e（_run_universal_verify_round·UNIVERSAL_PROOF_MODE·EXTERNAL） ============

def test_run_universal_verify_round_reward_1_external():
    """e2e：全称 item（鸟 都是 动物）+ ConceptNet 鸟 IsA 动物 + gates ON → reward=1 + verify_source=EXTERNAL。

    **★首个语言域真构造性验证 episode**（刀 A/B SELF_PRODUCED 是检查·刀 C 升验证·Layer0 external_verified 计入）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_uni = gates.UNIVERSAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.UNIVERSAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 种 ConceptNet 外部边：鸟 IsA 动物（source=CONCEPTNET·epistemic=STRUCTURED）
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _universal_item(["鸟", "都是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is not None, "全称 verify round 须产 episode（ConceptNet 验证通过）"
        assert res.episode.reward == 1, "鸟⊆动物（ConceptNet 外部断言）→ reward=1（构造性验证通过）"
        assert res.episode.verify_source == VERIFY_SOURCE_EXTERNAL, (
            f"_run_universal_verify_round 须标 EXTERNAL（ConceptNet 外部 R6 源·首个语言域真构造性验证）"
            f"·got verify_source={res.episode.verify_source}")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.UNIVERSAL_PROOF_MODE = saved_uni


def test_run_universal_verify_round_missing_path_no_episode():
    """e2e：formal runtime 未注入显式反证时，ConceptNet 缺边必须弃权。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_uni = gates.UNIVERSAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.UNIVERSAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 种 ConceptNet：鸟 IsA 动物 / 植物 IsA 生物（两分类概念·鸟⊄植物）
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store,
                             [("鸟", "动物"), ("植物", "生物")], space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _universal_item(["鸟", "都是", "植物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.UNIVERSAL_PROOF_MODE = saved_uni


def test_run_universal_verify_round_cant_verify_no_episode():
    """e2e：没有目标祖先路径时返回未知且不产生 EXTERNAL episode。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_uni = gates.UNIVERSAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.UNIVERSAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 种 ConceptNet：鸟 IsA 动物（石头 不种·非分类概念）
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        _ensure_concept(ctx, "石头")   # ensure 石头 被概念化（observe 也能·此处确定性保证 lookup 命中）
        r = DefaultRoundRunner()
        item = _universal_item(["石头", "都是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        # can't-verify → 弃权·无 EXTERNAL verify episode（res.episode None 或非 EXTERNAL universal）
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL, (
            "child 非分类概念 → can't-verify → 弃权无 EXTERNAL episode（守 #479 墙·非证伪）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.UNIVERSAL_PROOF_MODE = saved_uni


def test_run_universal_verify_round_gate_off_no_episode():
    """UNIVERSAL_PROOF_MODE OFF → 路由不走·全称 item 走正常语言 episode_loop（bit-identical·零行为变）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_uni = gates.UNIVERSAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.UNIVERSAL_PROOF_MODE = False   # OFF
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _universal_item(["鸟", "都是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        # gate OFF → 不走 universal verify（单段 struct_ref 孤立 → 正常路径 RoundResult() 空）
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL, (
            "UNIVERSAL_PROOF_MODE OFF → 不产 EXTERNAL universal verify episode（bit-identical）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.UNIVERSAL_PROOF_MODE = saved_uni


def test_run_universal_verify_round_no_conceptnet_can_verify_honest():
    """e2e：无 ConceptNet 文件（CI default·ext_map 空）→ 全 can't-verify → 无 reward（诚实降级·非 theater·非平凡通过）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_uni = gates.UNIVERSAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.UNIVERSAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 不种 ConceptNet 边（CI default·ext_map 空）
        r = DefaultRoundRunner()
        item = _universal_item(["鸟", "都是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        # ext_map 空 → 无支持路径，必须弃权且不得产生 reward=1。
        assert not (res.episode is not None and res.episode.reward == 1
                    and res.episode.verify_source == VERIFY_SOURCE_EXTERNAL), (
            "无 ConceptNet 外部源 → 不准 reward=1 EXTERNAL（诚实降级·非平凡通过·反 theater）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.UNIVERSAL_PROOF_MODE = saved_uni


# ============ 件④ capability_exam 消费 + Layer0 反 theater（EXTERNAL 真验证） ============

def test_capability_exam_runs_with_universal_item(tmp_path):
    """capability_exam + 全称 item → 产 report（不崩·layer0 6 key 齐·additive 字段）。"""
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "uni"), run_id="uni_1")
        report = run_capability_exam(
            cfg, [_universal_item(["鸟", "都是", "动物"])],
            backend=b, runner=DefaultRoundRunner())
        assert isinstance(report, CapabilityReport)
        assert set(report.layer0_attribution.keys()) == {
            "external_verified", "self_produced_check_passed", "self_produced_check_failed",
            "anchor_satisfied", "anchor_violated", "total",
        }
    finally:
        gates.TRAINING_MODE = saved


def test_project_layer0_counts_universal_external():
    """capability_exam 消费者（project_layer0）正确计数全称 EXTERNAL episode（构造 episode·直测消费）。

    构造 result.episodes 含 universal verify EXTERNAL reward=1 episode → project_layer0 计
    external_verified（**真构造性验证**·反 SELF_PRODUCED 不计）。对照 SELF_PRODUCED reward=1 不计 external_verified。
    """
    from types import SimpleNamespace
    from pure_integer_ai.cognition.shared.types import Episode
    # 构造：1 universal EXTERNAL (reward=1·真验证) + 1 SELF_PRODUCED (reward=1·检查·对照)
    result = SimpleNamespace(episodes=[
        Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL),       # universal verify episode
        Episode(reward=1, verify_source=VERIFY_SOURCE_SELF_PRODUCED),  # time/numeric 对照
    ])
    counts = project_layer0(result)
    assert counts["external_verified"] == 1, "universal EXTERNAL reward=1 计构造性验证（首个语言域）"
    assert counts["self_produced_check_passed"] == 1, "SELF_PRODUCED reward=1 计检查通过"
    assert counts["anchor_satisfied"] == 1, "EXTERNAL 满足锚门（可驱动停止决策）"
    assert counts["anchor_violated"] == 1, "SELF_PRODUCED 违锚门（全自产不准停）"
    assert counts["total"] == 2


def test_universal_episode_is_constructive_verification():
    """**反 theater 端到端·首个真验证**：universal verify episode reward=1（ConceptNet 验证）= 构造性验证 + 满足锚门。

    对照刀 A/B SELF_PRODUCED（reward=1 但非构造性验证）·刀 C EXTERNAL（reward=1 且 is_constructive_verification=True）。
    这是分层墙认知更正 §六D#1"形式子结构层须 R6 加固"的 LIVE 落地（ConceptNet 即 R6·首个语言域 EXTERNAL）。
    """
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_uni = gates.UNIVERSAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.UNIVERSAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        r = DefaultRoundRunner()
        res = r.run_round_full(ctx, _universal_item(["鸟", "都是", "动物"]), STAGE3_REWARD, 0)
        ep = res.episode
        assert ep is not None and ep.reward == 1
        # ★反 theater 核心（刀 C 与刀 A/B 的本质差异）：EXTERNAL reward=1 = 构造性验证 + 满足锚门
        assert is_constructive_verification(ep) is True, (
            "EXTERNAL + reward=1 计构造性验证（首个语言域真验证·ConceptNet R6 外部源）")
        assert external_anchor_satisfied(ep) is True, (
            "EXTERNAL 满足外部锚门（可驱动停止决策·反 SELF_PRODUCED 全自产不准停）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.UNIVERSAL_PROOF_MODE = saved_uni


def test_universal_verify_bit_identical():
    """project_layer0 两构造同输入 → 一致（bit-identical·count_layer0 确定性·universal EXTERNAL 分桶稳定）。"""
    import json
    from types import SimpleNamespace
    from pure_integer_ai.cognition.shared.types import Episode

    def build():
        return SimpleNamespace(episodes=[
            Episode(reward=1, verify_source=VERIFY_SOURCE_EXTERNAL),
            Episode(reward=0, verify_source=VERIFY_SOURCE_EXTERNAL),
        ])

    l1 = project_layer0(build())
    l2 = project_layer0(build())
    assert json.dumps(l1, sort_keys=True) == json.dumps(l2, sort_keys=True), "两构造 layer0 不一致·违 bit-identical"
    assert l1["external_verified"] == 1   # reward=1 EXTERNAL


# ============ 诚实边界（docstring 标注） ============

def test_universal_proof_documents_open_world_boundary():
    """模块说明必须明确缺边不是反证以及冲突证据返回未知。"""
    import pure_integer_ai.training.universal_proof as mod
    docstring = mod.__doc__ or ""
    assert "缺少路径不是" in docstring
    assert "冲突" in docstring and "None" in docstring


def test_universal_proof_option_a_no_persist():
    """proof 模块只消费注入证据，不导入存储 writer。"""
    import pure_integer_ai.training.universal_proof as mod
    source = __import__("inspect").getsource(mod)
    assert "pure_integer_ai.storage" not in source


def test_universal_proof_has_explicit_refutation_input():
    """反证必须由调用方显式注入，不能由关系名称或缺边推断。"""
    import pure_integer_ai.training.universal_proof as mod
    signature = __import__("inspect").signature(mod.universal_proof_fn_factory)
    assert "refuted_claims" in signature.parameters


def test_universal_proof_does_not_hardcode_evidence_source():
    """通用 proof 不写死 ConceptNet 或其他具体外部来源。"""
    import pure_integer_ai.training.universal_proof as mod
    source = __import__("inspect").getsource(mod)
    assert "ConceptNet" not in source


def test_universal_proof_never_reward():
    """proof 只返回三值结果，不直接写 episode 或 reward。"""
    import pure_integer_ai.training.universal_proof as mod
    source = __import__("inspect").getsource(mod)
    assert "Episode(" not in source and "reward=" not in source


# ============ gate 默认 OFF（bit-identical 守 CI=生产） ============

def test_universal_proof_mode_default_off():
    """UNIVERSAL_PROOF_MODE 默认 OFF·守 CI 回归 bit-identical（路由不走·既有语言域 episode_loop 不变）。"""
    import importlib
    import pure_integer_ai.config.gates as g
    importlib.reload(g)
    assert g.UNIVERSAL_PROOF_MODE is False, "UNIVERSAL_PROOF_MODE 默认 OFF 守 bit-identical"
