"""刀6 件7 sense 多义管线修通测试（学习放开 6 刀第七刀·任务 #631-#638·doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀6）。

刀6 = 件7 sense 消歧。设计原定 #479 墙（教师定义权·真消歧断奶后内生判据缺）。侦察核证 MultiRef 管线
  实现上完全断裂（observe:111 塌缩 + multi_refs 死列表 + 生成侧 activate_candidates 是 concept→词形非选 sense +
  recognize grep sense 零命中 + sense_lookup 生产永远 None）= 纸面闭合。**用户拍板 Option B**（修通 MultiRef 域内管线）。

本测覆盖：
  - 片1 表 unit（sense_candidates MUTABLE_MONOTONE·key=surface_hash·register/read/record/bootstrap）+ bit-identical
  - 片3 摄入侧（observe 写 sc_tn + sense_lookup hook·gate 守）— 片3 加
  - 片6 反 theater e2e（"猫追老鼠" held-out clone 动物老鼠/鼠标·动物类骨架命中动物老鼠·不命中鼠标·IS_A 共祖选优非语义消歧·#479 墙）

**诚实边界**（反 theater·IS_A 共祖 + collide_score 结构选优·非语义消歧）：
  - "老鼠"两 sense（动物老鼠/鼠标）·训练"猫追老鼠"建动物类骨架（IS_A 上卷）·held-out clone 动物老鼠 root
    命中（IS_A 共祖）·鼠标 root 不命中（IS_A 物品非动物）→ recognized==1 distinct origin。
  - 共现无法区分时撞 #479 墙（结构选优≠语义消歧·stable≠correct）。

铁律：纯整数 / 确定性 bit-identical（SENSE_LOOKUP_MODE 默认 OFF·sorted/NodeRef 升序/Hasher 固定种子）/
  不写死（sense 词典外部数据源·emergent_role 涌现非词性）/ §8.1c（统计表非关系边）/ §8.5（不建边不预留乘子）/
  reward CAUSES-only（sense 管线不接 reward·sc_sn defer）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.sense_candidates import (
    register_sense_candidates, read_sense_candidates, record_sense_token_seen,
    bootstrap_sense_candidates, sense_surface_hash, SENSE_CANDIDATES_TABLE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex


# ============ 片1：表 unit ============

@pytest.fixture
def sc_env():
    """sense_candidates 单测环境（dict backend + core space + sense_candidates 注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_sense_candidates(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, ns, ci
    b.close()


def _ensure(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


def test_sense_surface_hash_deterministic():
    """sense_surface_hash 固定种子·跨调用跨 run 确定（bit-identical·Hasher("sense_candidates.surface")）。"""
    h1 = sense_surface_hash("老鼠")
    h2 = sense_surface_hash("老鼠")
    assert h1 == h2, "同 surface 同 hash（确定性）"
    assert isinstance(h1, int), "hash 是 int（纯整数铁律）"
    assert sense_surface_hash("老鼠") != sense_surface_hash("动物老鼠"), "不同 surface 不同 hash"


def test_record_inserts_and_increments(sc_env):
    """record_sense_token_seen：首次 insert(sc_tn=1)·再次 sc_tn++。base_count/sc_sn 守 0（observe 路径 defer）。"""
    b, sid, ns, ci = sc_env
    sh = sense_surface_hash("老鼠")
    sense1 = _ensure(ci, sid, "动物老鼠")
    assert read_sense_candidates(b, sid, sh) == [], "首次前无行"
    record_sense_token_seen(b, sid, sh, sense1)
    record_sense_token_seen(b, sid, sh, sense1)
    got = read_sense_candidates(b, sid, sh)
    assert len(got) == 1, "一 sense 一行"
    assert got[0][0] == sense1, "sense ref 对"
    assert got[0][1:] == (0, 0, 2), "base_count/sc_sn 守 0·sc_tn=2（observe 写·reward defer）"


def test_read_multiple_senses_sorted_noderef(sc_env):
    """一 token 多 sense → 多行·read 按 sense_ref NodeRef 升序（确定性 tiebreak·bit-identical）。"""
    b, sid, ns, ci = sc_env
    sh = sense_surface_hash("老鼠")
    sense_a = _ensure(ci, sid, "动物老鼠")
    sense_b = _ensure(ci, sid, "鼠标")
    record_sense_token_seen(b, sid, sh, sense_a)
    record_sense_token_seen(b, sid, sh, sense_b)
    got = read_sense_candidates(b, sid, sh)
    assert len(got) == 2, "两 sense 两行（一 token 多义）"
    refs = [g[0] for g in got]
    assert refs == sorted(refs), "按 sense_ref NodeRef 升序（确定性）"


def test_read_unregistered_table_returns_empty():
    """表未注册 → read 返 []（向后兼容·同 read_experience_count/read_selection_pref_count 范式）。"""
    b = DictBackend()
    bootstrap(b)   # 不调 register_sense_candidates
    sh = sense_surface_hash("老鼠")
    assert read_sense_candidates(b, 1, sh) == [], "表未注册→空列表"


def test_record_unregistered_table_skips_silently():
    """表未注册 → record 静默 skip（向后兼容·不抛）。"""
    b = DictBackend()
    bootstrap(b)
    sh = sense_surface_hash("老鼠")
    # 不抛（bare fixture 向后兼容）
    record_sense_token_seen(b, 1, sh, (1, 1))


def test_surface_hash_key_decouples_token_from_sense_ref(sc_env):
    """key=surface_hash 解 N10：两 sense 用同 token surface_hash key·但不同 sense ref（不同 surface ensure）。

    concept_index.ensure("老鼠") 幂等返同 ref·两 sense 不能用同 ref·用 surface_hash 解耦 token 标识与 sense ref。
    """
    b, sid, ns, ci = sc_env
    sh = sense_surface_hash("老鼠")   # token 标识
    sense_a = _ensure(ci, sid, "动物老鼠")   # 不同 surface → 不同 ref
    sense_b = _ensure(ci, sid, "鼠标")
    assert sense_a != sense_b, "两 sense 不同 surface 不同 ref（解 ensure 幂等撞 ref）"
    record_sense_token_seen(b, sid, sh, sense_a)
    record_sense_token_seen(b, sid, sh, sense_b)
    got = read_sense_candidates(b, sid, sh)
    sense_refs = {g[0] for g in got}
    assert sense_refs == {sense_a, sense_b}, "同 token(surface_hash) 两 sense·各一行"


# ============ 片1：bootstrap unit ============

def test_bootstrap_empty_pairs_zero_side_effect(sc_env):
    """bootstrap_sense_candidates 空 sense_pairs → 立即返 0·绝不调 ensure/select/insert（bit-identical P0·镜像 bootstrap_is_a_edges）。

    无 PURE_INTEGER_AI_LOCAL_DIR → resolve_sense_facts 返 [] → bootstrap 空 short-circuit → 表空 → 退化链 5 步守。
    """
    b, sid, ns, ci = sc_env
    n = bootstrap_sense_candidates(b, ci, [], space_id=sid)
    assert n == 0, "空 pairs → 0"
    rows = b.select(SENSE_CANDIDATES_TABLE)
    assert len(rows) == 0, "空 pairs → 表无新行（零副作用）"


def test_bootstrap_seeds_base_count(sc_env):
    """bootstrap 非空：每 (word, [senses]) 种 base_count=1·sc_sn=0·sc_tn=0（boot 先验·observe 自写 sc_tn）。"""
    b, sid, ns, ci = sc_env
    pairs = [("老鼠", ["动物老鼠", "鼠标"]), ("猫", ["动物猫"])]
    n = bootstrap_sense_candidates(b, ci, pairs, space_id=sid)
    assert n == 3, "3 sense 行（老鼠 2 + 猫 1）"
    sh_mouse = sense_surface_hash("老鼠")
    got = read_sense_candidates(b, sid, sh_mouse)
    assert len(got) == 2, "老鼠 2 sense"
    for sense_ref, base, sc_sn, sc_tn in got:
        assert base == 1, "boot 种 base_count=1"
        assert sc_sn == 0, "sc_sn 守 0（reward feed defer）"
        assert sc_tn == 0, "sc_tn 守 0（observe 自写·boot 不碰）"


def test_bootstrap_idempotent(sc_env):
    """bootstrap 幂等：重复 boot 同 pairs → 不 corrupt（first-write-wins·base_count append-only 不重写）。"""
    b, sid, ns, ci = sc_env
    pairs = [("老鼠", ["动物老鼠", "鼠标"])]
    n1 = bootstrap_sense_candidates(b, ci, pairs, space_id=sid)
    n2 = bootstrap_sense_candidates(b, ci, pairs, space_id=sid)   # 重复 boot（resume 跨 run）
    assert n1 == 2, "首 boot 种 2 行"
    assert n2 == 0, "重复 boot 幂等 skip（first-write-wins）"
    sh = sense_surface_hash("老鼠")
    got = read_sense_candidates(b, sid, sh)
    assert len(got) == 2, "重复 boot 不 corrupt·仍 2 行"
    for _, base, _, _ in got:
        assert base == 1, "base_count 不重写（append-only）"


def test_bootstrap_empty_senses_skipped(sc_env):
    """bootstrap：word 无 sense 候选（空 list）→ 跳（守确定性·不种空 word 行）。"""
    b, sid, ns, ci = sc_env
    pairs = [("老鼠", []), ("猫", ["动物猫"])]
    n = bootstrap_sense_candidates(b, ci, pairs, space_id=sid)
    assert n == 1, "空 senses 的 word 跳·只种猫 1 行"


def test_bootstrap_observe_separate_sources(sc_env):
    """boot 写 base_count + observe 写 sc_tn·两源分离（base_count=1 boot·sc_tn observe 自增·不互扰）。"""
    b, sid, ns, ci = sc_env
    pairs = [("老鼠", ["动物老鼠"])]
    bootstrap_sense_candidates(b, ci, pairs, space_id=sid)
    sh = sense_surface_hash("老鼠")
    sense = _ensure(ci, sid, "动物老鼠")
    # observe 写 sc_tn（boot 后）
    record_sense_token_seen(b, sid, sh, sense)
    record_sense_token_seen(b, sid, sh, sense)
    got = read_sense_candidates(b, sid, sh)
    assert len(got) == 1
    _, base, sc_sn, sc_tn = got[0]
    assert base == 1, "boot base_count 守 1（observe 不碰 base）"
    assert sc_tn == 2, "observe 写 sc_tn=2"
    assert sc_sn == 0, "sc_sn 守 0"


# ============ 片1：bit-identical（make_train_context 注册 + dump 含表·gate OFF 表空） ============

def test_make_train_context_registers_sense_candidates(monkeypatch):
    """bit-identical：make_train_context 注册 sense_candidates 表（select 不抛）·gate OFF + 无文件 → 表空。

    退化链 5 步（plan 决断 5）：无 PURE_INTEGER_AI_LOCAL_DIR → resolve_sense_facts 返 [] → bootstrap 空
    short-circuit → 表空。dump_tables 含表名验证留片6 反 theater e2e（formal_train 跑触发 dump 间接验）。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    from pure_integer_ai.experiments.formal_train import make_train_context
    b = DictBackend()
    ctx = make_train_context(b)
    # 表已注册（register_sense_candidates 在 make_train_context 调·select 不抛 KeyError）
    rows = b.select(SENSE_CANDIDATES_TABLE)
    assert rows == [], "gate OFF + 无文件 → 表空（bit-identical 退化·零行为变）"


# ============ 片2：sense_facts loader unit ============

from pure_integer_ai.experiments.collection import (
    load_sense_facts_file, resolve_sense_facts,
)
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN, LANG_NONE


def test_load_sense_facts_file_parses(tmp_path):
    """load_sense_facts_file：每行 "word sense1 sense2 ..."→ (word, [senses])·首段 word·其余 senses。"""
    p = tmp_path / "sense_facts_test.txt"
    p.write_text(
        "# 注释\n"
        "老鼠 动物老鼠 鼠标\n"
        "苹果 水果苹果 公司苹果\n",
        encoding="utf-8",
    )
    pairs = load_sense_facts_file(str(p))
    assert pairs == [("老鼠", ["动物老鼠", "鼠标"]),
                     ("苹果", ["水果苹果", "公司苹果"])], "解析 word→[senses]"


def test_load_sense_facts_file_graceful(tmp_path):
    """E5 graceful：注释/空行/<2 段/自环 sense 全 skip + 不抛崩（镜像 load_is_a_facts_file）。"""
    p = tmp_path / "sense_facts_graceful.txt"
    p.write_text(
        "# 注释行 skip\n"
        "\n"
        "老鼠 动物老鼠 鼠标\n"
        "猫\n"             # <2 段（只 word 无 sense）skip
        "狗 狗\n"          # 自环 sense（sense==word）早跳→senses 空→skip
        "鱼 鱼鱼 鲤鱼\n",
        encoding="utf-8",
    )
    pairs = load_sense_facts_file(str(p))
    # "狗 狗" 自环→senses 空→skip；其余有效
    assert ("老鼠", ["动物老鼠", "鼠标"]) in pairs
    assert ("鱼", ["鱼鱼", "鲤鱼"]) in pairs
    assert not any(w == "猫" for w, _ in pairs), "<2 段 skip"
    assert not any(w == "狗" for w, _ in pairs), "自环 sense 全 skip"
    assert len(pairs) == 2


def test_load_sense_facts_file_oserror_returns_empty(tmp_path):
    """文件读错/不存在 → 返 []（E5 graceful·不抛崩）。"""
    assert load_sense_facts_file(str(tmp_path / "nonexistent.txt")) == []


def test_resolve_sense_facts_no_local_dir_returns_empty(monkeypatch):
    """无 PURE_INTEGER_AI_LOCAL_DIR → resolve 返 []（生产 default bit-identical 守·镜像 resolve_is_a_facts）。"""
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    assert resolve_sense_facts(LANG_ZH) == []
    assert resolve_sense_facts(LANG_ZH, local_dir=None) == []


def test_resolve_sense_facts_lang_none_returns_empty(monkeypatch, tmp_path):
    """LANG_NONE 无文件后缀映射 → resolve 返 []（非语言模态不种 sense）。"""
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))
    assert resolve_sense_facts(LANG_NONE) == []


def test_resolve_sense_facts_reads_local_dir(tmp_path, monkeypatch):
    """PURE_INTEGER_AI_LOCAL_DIR + sense_facts_{suffix}.txt 存在 → resolve 读文件返 pairs。"""
    d = tmp_path / "data"
    d.mkdir()
    (d / "sense_facts_zh.txt").write_text("老鼠 动物老鼠 鼠标\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(d))
    pairs = resolve_sense_facts(LANG_ZH)
    assert pairs == [("老鼠", ["动物老鼠", "鼠标"])]
    # EN 文件不存在 → 空（lang 隔离）
    assert resolve_sense_facts(LANG_EN) == []


# ============ 片3：sense_lookup hook + observe 写 sc_tn e2e ============

from pure_integer_ai.cognition.understanding.sense_lookup_hook import make_sense_lookup
from pure_integer_ai.config import gates


def test_make_sense_lookup_gate_off_returns_none(sc_env):
    """gate OFF → make_sense_lookup 返 None（observe sense_lookup=None·MultiRef 不产·退化 bit-identical）。"""
    b, sid, ns, ci = sc_env
    saved = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = False
    try:
        assert make_sense_lookup(b, sid) is None
    finally:
        gates.SENSE_LOOKUP_MODE = saved


def test_make_sense_lookup_gate_on_returns_base_count_positive(sc_env):
    """gate ON + boot 种 → hook 返 base_count>0 候选（NodeRef 升序·确定性·不读 sc_tn 防循环）。"""
    b, sid, ns, ci = sc_env
    pairs = [("老鼠", ["动物老鼠", "鼠标"])]
    bootstrap_sense_candidates(b, ci, pairs, space_id=sid)
    # observe 写 sc_tn（base_count 不动·hook 仍读 base_count>0）
    sh = sense_surface_hash("老鼠")
    record_sense_token_seen(b, sid, sh, _ensure(ci, sid, "动物老鼠"))
    saved = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = True
    try:
        hook = make_sense_lookup(b, sid)
        assert hook is not None
        got = hook("老鼠")
        assert len(got) == 2, "两 sense base_count>0"
        assert got == sorted(got), "NodeRef 升序（确定性）"
    finally:
        gates.SENSE_LOOKUP_MODE = saved


def test_make_sense_lookup_no_candidates_returns_empty(sc_env):
    """gate ON 但 token 无 sense 候选（表空）→ hook 返 []（observe 走 OOV·退化）。"""
    b, sid, ns, ci = sc_env
    saved = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = True
    try:
        hook = make_sense_lookup(b, sid)
        assert hook is not None
        assert hook("未知词") == [], "无候选→空（observe 走 OOV）"
    finally:
        gates.SENSE_LOOKUP_MODE = saved


def test_make_sense_lookup_does_not_read_sc_tn(sc_env):
    """hook 只读 base_count·不读 sc_tn（防 observe 写→hook 读→MultiRef→observe 又写 循环·决断 7）。

    base_count=0 的行 → hook 不返（即使 sc_tn 高）·守防循环。
    """
    b, sid, ns, ci = sc_env
    sh = sense_surface_hash("test")
    sense = _ensure(ci, sid, "testsense")
    b.insert(SENSE_CANDIDATES_TABLE, {
        "space_id": sid, "surface_hash": sh, "sense_sid": sense[0], "sense_lid": sense[1],
        "base_count": 0, "sc_sn": 0, "sc_tn": 5,   # sc_tn 高但 base_count=0
    })
    saved = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = True
    try:
        hook = make_sense_lookup(b, sid)
        assert hook("test") == [], "base_count=0 不返（hook 只读 base_count·不读 sc_tn·防循环）"
    finally:
        gates.SENSE_LOOKUP_MODE = saved


def test_observe_writes_sc_tn_via_sense_lookup(tmp_path, monkeypatch):
    """e2e：boot 种 sense_facts + observe 跑 → MultiRef 产生 → record sc_tn（摄入侧写真）。

    run_round_full 不跑 boot 段（boot 在 run_train 主流程）也不翻 gate·故测里手动 boot 种（resolve_sense_facts
    + bootstrap_sense_candidates·镜像 formal_train boot 段）+ 手动翻 SENSE_LOOKUP_MODE（同 test_knife5 范式）。
    反 theater 牙：observe 真写 sc_tn（非死列表·非空表 theater·刀6 修通 MultiRef 摄入侧）。
    """
    d = tmp_path / "data"
    d.mkdir()
    (d / "sense_facts_zh.txt").write_text("老鼠 动物老鼠 鼠标\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(d))
    from pure_integer_ai.experiments.formal_train import make_train_context, DefaultRoundRunner
    from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES, resolve_sense_facts
    from pure_integer_ai.training.stages import STAGE3_REWARD
    b = DictBackend()
    ctx = make_train_context(b)
    # 手动 boot 种（run_round_full 不跑 boot·boot 在 run_train；镜像 formal_train boot 段）
    pairs = resolve_sense_facts(LANG_ZH)
    bootstrap_sense_candidates(b, ctx.concept_index, pairs, space_id=ctx.space_id)
    runner = DefaultRoundRunner()
    saved = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = True   # run_round_full 不翻 gate·手动翻（observe sense_lookup hook 命中）
    try:
        item = CollectedItem(tokens=["猫", "追", "老鼠"], collect_type=COLLECT_PRECEDES, lang=LANG_ZH)
        runner.run_round(ctx, item, STAGE3_REWARD, 0)
    finally:
        gates.SENSE_LOOKUP_MODE = saved
    sh = sense_surface_hash("老鼠")
    rows = b.select(SENSE_CANDIDATES_TABLE, where={"surface_hash": sh, "space_id": ctx.space_id})
    assert len(rows) == 2, "老鼠 两 sense（boot 种 base_count + observe 写 sc_tn）"
    for r in rows:
        assert int(r["base_count"]) == 1, "boot 种 base_count=1"
        assert int(r["sc_tn"]) >= 1, "observe 写 sc_tn（MultiRef record·摄入侧写真）"
