"""存在量化 cue、开放世界 proof 和 formal runtime 诚实退场测试。"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.cognition.shared.types import (
    LANG_ZH, LANG_EN, MODALITY_LANGUAGE, VERIFY_SOURCE_EXTERNAL, VERIFY_SOURCE_SELF_PRODUCED,
)
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, SOURCE_CONCEPTNET, EPI_STRUCTURED, EPI_CUE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_CAUSES
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, DefaultRoundRunner, make_train_context,
)
from pure_integer_ai.cognition.result.layer0_anchor import (
    is_constructive_verification, external_anchor_satisfied,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, EXISTENTIAL_CUE, UNIVERSAL_CUE,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_existential_claims, extract_existential_claims_gated,
)
from pure_integer_ai.cognition.understanding.is_a import bootstrap_is_a_edges
from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map_external
from pure_integer_ai.training.existential_proof import existential_proof_fn_factory


def _ref(n: int):
    """测试 ConceptRef（space_id=1·local_id=n·同 test_universal_proof 范式）。"""
    return (1, n)


def _existential_item(tokens=None, *, lang: int = LANG_ZH):
    """建存在量化语言 item（MODALITY_LANGUAGE·tokens 含 有的 X 是 Y）。"""
    toks = list(tokens) if tokens is not None else ["有的", "鸟", "是", "动物"]
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


# ============ 件① 词表/类型（cue_type_of·EXISTENTIAL_CUE） ============

def test_cue_type_of_youda_zh():
    """ZH 存在量化词（有的）→ EXISTENTIAL_CUE（exact 匹配·存在量化 ∃x∈X∧x∈Y 锚）。"""
    assert cue_type_of("有的", LANG_ZH) == EXISTENTIAL_CUE
    assert cue_type_of("有些", LANG_ZH) == EXISTENTIAL_CUE


def test_cue_type_of_existential_not_universal():
    """EXISTENTIAL_CUE ≠ UNIVERSAL_CUE（∃ ≠ ∀·不同 cue_type·不同验序器）。"""
    assert EXISTENTIAL_CUE != UNIVERSAL_CUE


def test_cue_type_of_non_modal_word():
    """非存在量化词 → None（exact 匹配·不命中零 pair·守反统计）。"""
    assert cue_type_of("鸟", LANG_ZH) is None
    assert cue_type_of("是", LANG_ZH) is None
    assert cue_type_of("某些", LANG_ZH) is None   # 开放变体 defer·不在 frozenset（D6 closed-class only）


# ============ 件② 构造器（extract_existential_claims·有的 X 是 Y 起始 cue 窗口） ============

def test_extract_existential_claims_basic_window():
    """有的 X 是 Y → [(child_idx=1, parent_idx=3)]（起始 cue·是 at 2·parent at 3）。"""
    tokens = ["有的", "鸟", "是", "动物"]
    claims = extract_existential_claims(tokens, lang=LANG_ZH)
    assert claims == [(1, 3)], f"有的 鸟 是 动物 → [(1,3)]·got {claims}"


def test_extract_existential_claims_youxie():
    """有些 X 是 Y 同窗口（closed-class 同义·D6）。"""
    tokens = ["有些", "鸟", "是", "企鹅"]
    claims = extract_existential_claims(tokens, lang=LANG_ZH)
    assert claims == [(1, 3)]


def test_extract_existential_claims_requires_copula():
    """无 是（"有的 X Y" 非法）→ []（守 是 锚定·不凑配·同 property 固定窗口范式）。"""
    tokens = ["有的", "鸟", "动物"]   # 无 是·非存在量化窗口
    claims = extract_existential_claims(tokens, lang=LANG_ZH)
    assert claims == [], "无 是 锚定 → 不产 claim（守反统计·不凑配）"


def test_extract_existential_claims_boundary_parent_missing():
    """边界·parent 不足（cue 在末尾）→ []（守反统计·不凑配）。"""
    tokens = ["鸟", "是", "有的"]   # cue at 2·i+3 越界
    claims = extract_existential_claims(tokens, lang=LANG_ZH)
    assert claims == []


def test_extract_existential_claims_child_is_cue_skip():
    """child 自身是 cue → 跳（连用 cue·锚定歧义·首版保守跳·同 ∀:250）。"""
    tokens = ["有的", "都是", "是", "动物"]   # child=都是 是 cue
    claims = extract_existential_claims(tokens, lang=LANG_ZH)
    assert claims == []


def test_extract_existential_claims_empty_tokens():
    """空 tokens → []（守边界·不崩）。"""
    assert extract_existential_claims([], lang=LANG_ZH) == []


def test_extract_existential_claims_gated_off_returns_empty():
    """CUE_EXTRACTOR_MODE OFF → _gated 返 []（bit-identical 守回归·同 ∀ 范式）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = False
    try:
        tokens = ["有的", "鸟", "是", "动物"]
        assert extract_existential_claims_gated(tokens, lang=LANG_ZH) == []
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_extract_existential_claims_gated_on_passthrough():
    """CUE_EXTRACTOR_MODE ON → _gated 透传 extract_existential_claims（同结果）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        tokens = ["有的", "鸟", "是", "动物"]
        assert extract_existential_claims_gated(tokens, lang=LANG_ZH) == [(1, 3)]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


# ============ 件③ 消费者（existential_proof_fn_factory·三值证据） ============

def test_existential_proof_fn_forward_subset_without_nonempty_returns_none():
    """仅有鸟⊆动物而不知道鸟非空时，存在声明保持未知。"""
    ancestor_map = {_ref(1): {_ref(2)}}   # 鸟(1) IsA 动物(2)
    fn = existential_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2))])
    assert fn(None, None, None) is None


def test_existential_proof_fn_reversed_subset_without_nonempty_returns_none():
    """仅有企鹅⊆鸟而不知道企鹅非空时，反向存在声明也保持未知。"""
    ancestor_map = {_ref(3): {_ref(1)}}   # 企鹅(3) IsA 鸟(1)
    fn = existential_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(3))])
    assert fn(None, None, None) is None


def test_existential_proof_fn_known_nonempty_subclass_returns_1():
    """已知非空的小类同时包含于两侧时，存在声明得到正证。"""
    ancestor_map = {_ref(3): {_ref(1), _ref(2)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map,
        claims=[(_ref(1), _ref(2))],
        known_nonempty=[_ref(3)],
    )
    assert fn(None, None, None) == 1


def test_existential_proof_fn_explicit_overlap_returns_1():
    """共同 MEMBER 见证归约出的 overlap 集合对可直接支持存在声明。"""
    fn = existential_proof_fn_factory(
        ancestor_map={},
        claims=[(_ref(1), _ref(2))],
        overlap_witnesses=[(_ref(2), _ref(1))],
    )
    assert fn(None, None, None) == 1


def test_existential_proof_fn_falsified_returns_0():
    """只有显式 DISJOINT 证据才能反驳集合相交声明。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(4)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map,
        claims=[(_ref(1), _ref(3))],
        disjoint_pairs=[(_ref(3), _ref(1))],
    )
    assert fn(None, None, None) == 0


def test_existential_proof_fn_missing_paths_return_none():
    """两个分类概念间没有任一子集路径时也不能推出空交集。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(4)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map, claims=[(_ref(1), _ref(3))])
    assert fn(None, None, None) is None


def test_existential_proof_fn_cant_verify_property_returns_none():
    """没有 overlap、非空或 DISJOINT 证据的属性声明保持未知。"""
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = existential_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(1), _ref(5))])
    assert fn(None, None, None) is None, "parent 非分类概念 → can't-verify（None·守属性 ∃ #479 墙）"


def test_existential_proof_fn_cant_verify_unknown_child_returns_none():
    """child 非分类概念（未在 ConceptNet）→ None（外部源不足·诚实降级·非证伪）。"""
    ancestor_map = {_ref(1): {_ref(2)}}   # 仅 鸟→动物·石头(6) 不在
    fn = existential_proof_fn_factory(ancestor_map=ancestor_map, claims=[(_ref(6), _ref(2))])
    assert fn(None, None, None) is None


def test_existential_proof_fn_multi_all_verified():
    """多声明均具备非空子类证据时返回 1。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(1)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map,
        claims=[(_ref(1), _ref(2)), (_ref(1), _ref(3))],
        known_nonempty=[_ref(1), _ref(3)],
    )
    assert fn(None, None, None) == 1


def test_existential_proof_fn_multi_one_falsified_short_circuits():
    """多声明中任一显式 DISJOINT 使合取为 0。"""
    ancestor_map = {_ref(1): {_ref(2)}, _ref(3): {_ref(4)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map,
        claims=[(_ref(1), _ref(2)), (_ref(1), _ref(3))],
        known_nonempty=[_ref(1)],
        disjoint_pairs=[(_ref(1), _ref(3))],
    )
    assert fn(None, None, None) == 0


def test_existential_proof_fn_conflicted_claim_returns_none():
    """同一集合对同时有 overlap 与 DISJOINT 时保持冲突未知。"""
    pair = (_ref(1), _ref(2))
    fn = existential_proof_fn_factory(
        ancestor_map={}, claims=[pair],
        overlap_witnesses=[pair], disjoint_pairs=[pair],
    )
    assert fn(None, None, None) is None


def test_existential_proof_fn_multi_cant_verify_no_falsify_returns_none():
    """多声明·有 can't-verify 无 falsified → None（最弱约束诚实·弃权非证伪）。"""
    # claim1 verified (鸟⊆动物 forward)·claim2 can't-verify (鸟∩飞·飞非分类)
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2)), (_ref(1), _ref(5))])
    assert fn(None, None, None) is None


def test_existential_proof_fn_empty_returns_none():
    """claims 空 → None（vacate·诚实退场·非 pass·非 theater）。"""
    fn = existential_proof_fn_factory(ancestor_map={}, claims=[])
    assert fn(None, None, None) is None


def test_existential_proof_fn_deterministic():
    """确定性：同输入同输出（集合查询·bit-identical）。"""
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2))],
        known_nonempty=[_ref(1)])
    assert fn(None, None, None) == fn(None, None, None) == 1


def test_existential_proof_fn_defensive_copy_claims():
    """factory 防御拷贝 claims（caller 后续 mutation 不改已造 fn 行为·同 ∀ P2-3）。"""
    claims = [(_ref(1), _ref(2))]
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map, claims=claims, known_nonempty=[_ref(1)])
    claims.append((_ref(3), _ref(4)))   # caller mutation·若持引用会改 fn
    assert fn(None, None, None) == 1, "factory 须拷贝 claims·caller mutation 不影响 fn"


def test_existential_proof_fn_defensive_copy_ancestor_map():
    """factory 防御拷贝 ancestor_map（caller 后续改 map 不改已造 fn·dict+set 深拷贝值集）。"""
    ancestor_map = {_ref(1): {_ref(2)}}
    fn = existential_proof_fn_factory(
        ancestor_map=ancestor_map, claims=[(_ref(1), _ref(2))],
        known_nonempty=[_ref(1)])
    ancestor_map[_ref(1)].add(_ref(3))   # caller mutation·若持引用会改 fn 的 ancestors
    ancestor_map[_ref(7)] = {_ref(8)}
    assert fn(None, None, None) == 1, "factory 须拷贝 ancestor_map·caller mutation 不影响 fn"


# ============ 件③ 外部祖先图 source filter ============

def test_build_isa_ancestor_map_external_filters_source_existential():
    """反 theater 第一路：build_isa_ancestor_map_external 双 filter 排除 cue 自产边（∃ 复用 ∀ 的 ext_map）。

    ConceptNet 边（鸟→动物·source=CONCEPTNET）保留·cue 边（猫→动物·source=BARE_TEXT·epistemic=CUE）滤掉。
    """
    from pure_integer_ai.cognition.understanding.is_a import build_is_a_edge
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
    assert cat not in ext, "cue 自产边（猫·EPI_CUE）须被 source filter 滤掉·反 single-source theater"


# ============ 件④ e2e（_run_existential_verify_round·EXISTENTIAL_PROOF_MODE·EXTERNAL） ============

def test_run_existential_verify_round_subset_without_nonempty_no_episode():
    """e2e：formal runtime 未注入非空证据时，子集路径不能产生存在奖励。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_exist = gates.EXISTENTIAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.EXISTENTIAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _existential_item(["有的", "鸟", "是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.EXISTENTIAL_PROOF_MODE = saved_exist


def test_run_existential_verify_round_reversed_without_nonempty_no_episode():
    """e2e：反向子集同样需要较小类非空证据，不能单独证明存在。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_exist = gates.EXISTENTIAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.EXISTENTIAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 种 ConceptNet：企鹅 IsA 鸟（reversed：鸟∈ancestors(企鹅)）
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("企鹅", "鸟")],
                             space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _existential_item(["有的", "鸟", "是", "企鹅"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.EXISTENTIAL_PROOF_MODE = saved_exist


def test_run_existential_verify_round_missing_paths_no_episode():
    """e2e：formal runtime 未注入 DISJOINT 时，双向缺路径必须弃权。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_exist = gates.EXISTENTIAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.EXISTENTIAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 只有两条互不相连的分类边，不能据此推出鸟与植物不相交。
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store,
                             [("鸟", "动物"), ("植物", "生物")], space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _existential_item(["有的", "鸟", "是", "植物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.EXISTENTIAL_PROOF_MODE = saved_exist


def test_run_existential_verify_round_cant_verify_no_episode():
    """e2e：child 非分类概念（石头 不在 ConceptNet）→ can't-verify → 无 episode（弃权·守 #479 墙）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_exist = gates.EXISTENTIAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.EXISTENTIAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        _ensure_concept(ctx, "石头")   # ensure 石头 被概念化（lookup 命中但非分类概念）
        r = DefaultRoundRunner()
        item = _existential_item(["有的", "石头", "是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL, (
            "child 非分类概念 → can't-verify → 弃权无 EXTERNAL episode（守 #479 墙·非证伪）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.EXISTENTIAL_PROOF_MODE = saved_exist


def test_run_existential_verify_round_gate_off_no_episode():
    """EXISTENTIAL_PROOF_MODE OFF → 路由不走·存在 item 走正常语言 episode_loop（bit-identical·零行为变）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_exist = gates.EXISTENTIAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.EXISTENTIAL_PROOF_MODE = False   # OFF
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _existential_item(["有的", "鸟", "是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or res.episode.verify_source != VERIFY_SOURCE_EXTERNAL, (
            "EXISTENTIAL_PROOF_MODE OFF → 不产 EXTERNAL 存在 verify episode（bit-identical）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.EXISTENTIAL_PROOF_MODE = saved_exist


def test_run_existential_verify_round_no_conceptnet_honest():
    """e2e：无 ConceptNet 文件（CI default·ext_map 空）→ 全 can't-verify → 无 reward=1（诚实降级·非 theater）。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_exist = gates.EXISTENTIAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.EXISTENTIAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 不种 ConceptNet 边（CI default·ext_map 空）
        r = DefaultRoundRunner()
        item = _existential_item(["有的", "鸟", "是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert not (res.episode is not None and res.episode.reward == 1
                    and res.episode.verify_source == VERIFY_SOURCE_EXTERNAL), (
            "无 ConceptNet 外部源 → 不准 reward=1 EXTERNAL（诚实降级·非平凡通过·反 theater）")
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.EXISTENTIAL_PROOF_MODE = saved_exist


# ============ 件④ Layer0 反 theater（EXTERNAL 真验证·is_constructive_verification） ============

def test_existential_runtime_does_not_claim_constructive_verification_without_evidence():
    """runtime 尚无 MEMBER/nonempty/overlap 适配器时不得制造 EXTERNAL 验证。"""
    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_exist = gates.EXISTENTIAL_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.EXISTENTIAL_PROOF_MODE = True
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("鸟", "动物")],
                             space_id=ctx.space_id)
        r = DefaultRoundRunner()
        item = _existential_item(["有的", "鸟", "是", "动物"])
        res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
        assert res.episode is None or not (
            is_constructive_verification(res.episode)
            and external_anchor_satisfied(res.episode)
        )
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.EXISTENTIAL_PROOF_MODE = saved_exist


def test_d6_no_hardcode_open_variants():
    """D6 守卫：_CUE_WORDS EXISTENTIAL_CUE 只 closed-class {有的,有些}·开放变体（某些/部分/存在着）零硬编码。"""
    from pure_integer_ai.cognition.understanding.cue_words import _CUE_WORDS
    zh_existential = _CUE_WORDS[LANG_ZH].get(7, frozenset())   # EXISTENTIAL_CUE=7
    assert zh_existential == frozenset({"有的", "有些"}), (
        f"∃ closed-class 只 {有的,有些}·got {zh_existential}")
    # 开放变体不在 frozenset（走未来 D:11 教师晋升·无 REL_EXISTENTIAL 故 D:11 ∃ defer）
    for open_word in ("某些", "部分", "存在着", "有的个"):
        assert open_word not in zh_existential, f"开放变体 {open_word} 须零硬编码（D6）"
