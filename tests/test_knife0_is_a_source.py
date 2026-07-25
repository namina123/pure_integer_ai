"""刀0 接 IS_A 源测试（学习放开 6 刀·任务 #591·doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀0）。

刀0 = boot 时种 EDGE_ISA 边（formal_train make_train_context 后·lang 发现前）·让 ancestor_map 非空 →
S3 第二刀 Interp2 LCA 聚类真火（机制活数据空→数据非空）。来源① ConceptNet 本地文件（E10·EPI_STRUCTURED·
外部数据非 core enum·守「不写死」）。

**刀0 验收判据**（关键·非 recognize 命中/拒）：
  - WITH IS_A 文件 → boot 种 EDGE_ISA → discovered lang 骨架 ATTR_SLOT_ROLE 非 None（Interp2 聚类写 slot LCA）。
  - WITHOUT 文件（CI/生产 default）→ boot 零副作用 → discovered lang 骨架 ATTR_SLOT_ROLE 全 None（Interp2 跳聚类）。
  - contrast 证刀0 boot 接线让生产 caller ancestor_map 非空（机制测 test_stage12:809 用 _inject_isa 绕 formal_train·
    本测走真 formal_train 全 main 流程·验 boot 接线层）。

注意：_align_walk 无 ATTR_SLOT_ROLE 时 PARAM slot 接受任何 token（structure_discover.py:1124）→ recognize 命中
非刀0 判据（WITH/WITHOUT 都命中）·ATTR_SLOT_ROLE 有/无才是。

铁律：纯整数 / 确定性 bit-identical / 不写死（外部文件·core 不 import）/ E5 graceful（错行/缺文件不崩）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_CONCEPTNET, SOURCE_BARE_TEXT, EPI_CUE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_IS_A
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs, ATTR_OPERAND, ATTR_OPERATOR, ATTR_SLOT_ROLE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    ConceptRef, LANG_ZH, LANG_EN, MODALITY_LANGUAGE,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.is_a import (
    bootstrap_is_a_edges, IS_A_STRENGTH_EMPIRICAL,
)
from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map
from pure_integer_ai.experiments.collection import (
    load_is_a_facts_file, resolve_is_a_facts, CollectedItem, COLLECT_PRECEDES,
)


# ---- fixtures ----

@pytest.fixture
def isa_env():
    """bootstrap_is_a_edges 单测环境（dict backend·core space·composes_attr 注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ci
    b.close()


def _read_slot_roles(b: ConceptGraph, skeleton_ref: ConceptRef) -> list[tuple[int, int] | None]:
    """读 skeleton 全 CONCEPT slot 的 ATTR_SLOT_ROLE（DFS 阅读序·None=无类约束·镜像 test_stage12:_read_slot_roles）。"""
    g = ConceptGraph(b)
    children_of = g.read_composes_tree(skeleton_ref)[0]
    roles: list[tuple[int, int] | None] = []
    visited: set[ConceptRef] = set()

    def _dfs(node: ConceptRef) -> None:
        if node in visited:
            return
        visited.add(node)
        attrs = read_composes_attrs(b, node)
        if ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs:
            if ATTR_SLOT_ROLE in attrs:
                r = attrs[ATTR_SLOT_ROLE]
                roles.append((r[0], r[1]))
            else:
                roles.append(None)
        for child in children_of.get(node, []):
            _dfs(child)

    _dfs(skeleton_ref)
    return roles


# ============ unit：load_is_a_facts_file（E10 纯本地读·E5 graceful） ============

def test_load_is_a_facts_file_parses_pairs(tmp_path):
    """正常文件：每行 child parent·#注释/空行 skip·中段忽略（容错）。"""
    f = tmp_path / "is_a_facts_zh.txt"
    f.write_text(
        "# 注释行 skip\n"
        "猫 动物\n"
        "\n"   # 空行 skip
        "狗 是一种 动物\n"   # 中段系词忽略·首段 child 末段 parent
        "狐狸 动物\n",
        encoding="utf-8")
    pairs = load_is_a_facts_file(str(f))
    assert pairs == [("猫", "动物"), ("狗", "动物"), ("狐狸", "动物")], \
        "loader 解析 child/parent 对·注释/空行 skip·中段忽略"


def test_load_is_a_facts_file_malformed_lines_skipped(tmp_path):
    """E5 graceful：格式错行（<2 段）skip + 不抛崩·自环 skip。"""
    f = tmp_path / "is_a_facts_zh.txt"
    f.write_text(
        "猫 动物\n"
        "孤立词\n"   # 格式错（<2 段）skip
        "动物 动物\n"   # 自环 skip
        "狗 动物\n",
        encoding="utf-8")
    pairs = load_is_a_facts_file(str(f))
    assert pairs == [("猫", "动物"), ("狗", "动物")], \
        "格式错行 + 自环 skip·不抛崩（E5 graceful）"


def test_load_is_a_facts_file_missing_returns_empty(tmp_path):
    """E5 graceful：文件不存在 → 返 []（不抛崩）。"""
    assert load_is_a_facts_file(str(tmp_path / "nonexistent.txt")) == [], \
        "文件不存在 → 返空 list（E5 graceful·不抛崩）"


def test_load_is_a_facts_file_strips_bom(tmp_path):
    """utf-8-sig 自动 strip BOM（对抗审点 2）：BOM 文件首行注释正确识别·不污染概念点。"""
    f = tmp_path / "is_a_facts_zh.txt"
    # 写带 BOM 的 UTF-8·首行注释（BOM 前置·utf-8 读时 startswith("#") 不命中→utf-8-sig strip 修复）
    content = "# comment line\n猫 动物\n".encode("utf-8")
    f.write_bytes(b"\xef\xbb\xbf" + content)
    pairs = load_is_a_facts_file(str(f))
    assert pairs == [("猫", "动物")], \
        "BOM 文件首行注释须正确 skip（utf-8-sig strip BOM·不把 BOM+# 当数据行污染）"


# ============ unit：resolve_is_a_facts（env PURE_INTEGER_AI_LOCAL_DIR·缺文件空·bit-identical 守） ============

def test_resolve_is_a_facts_reads_local_dir(tmp_path, monkeypatch):
    """resolve 按 lang 读 PURE_INTEGER_AI_LOCAL_DIR/is_a_facts_{lang}.txt。"""
    f = tmp_path / "is_a_facts_zh.txt"
    f.write_text("猫 动物\n狗 动物\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))
    assert resolve_is_a_facts(LANG_ZH) == [("猫", "动物"), ("狗", "动物")], \
        "resolve_is_a_facts 按 lang 读 is_a_facts_zh.txt"


def test_resolve_is_a_facts_missing_file_empty(tmp_path, monkeypatch):
    """bit-identical 守：文件不存在 → 返 []（CI/生产 default）。"""
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))   # 目录存在但无 is_a_facts_zh.txt
    assert resolve_is_a_facts(LANG_ZH) == [], "文件不存在 → 返空（bit-identical 守）"


def test_resolve_is_a_facts_no_local_dir_empty(monkeypatch):
    """bit-identical 守：无 PURE_INTEGER_AI_LOCAL_DIR → 返 []。"""
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    assert resolve_is_a_facts(LANG_ZH) == [], "无 PURE_INTEGER_AI_LOCAL_DIR → 返空"
    assert resolve_is_a_facts(LANG_EN) == [], "无 PURE_INTEGER_AI_LOCAL_DIR → 返空（en 同）"


def test_resolve_is_a_facts_unmapped_lang_empty(tmp_path, monkeypatch):
    """无映射的 lang（LANG_NONE 等）→ 返 []。"""
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))
    assert resolve_is_a_facts(0) == [], "LANG_NONE 无文件映射 → 返空"


# ============ unit：bootstrap_is_a_edges（P0 空零副作用 + query_from 按源幂等 skip） ============

def test_bootstrap_is_a_edges_empty_no_side_effect(isa_env):
    """P0 bit-identical 硬守：空 pairs → return 0 + 零写盘（不调 ensure/query_from/build）。"""
    b, sid, es, ci = isa_env
    concept_before = len(b.select("concept_node", where=None))
    edge_before = len(b.select("edge", where=None))
    n = bootstrap_is_a_edges(ci, es, [], space_id=sid)
    assert n == 0, "空 pairs → return 0"
    assert len(b.select("concept_node", where=None)) == concept_before, "空 pairs 零概念点副作用"
    assert len(b.select("edge", where=None)) == edge_before, "空 pairs 零边副作用"


def test_bootstrap_is_a_edges_builds_edges_and_ancestor_map(isa_env):
    """种 pairs → EDGE_ISA 边建 + ancestor_map 非空（解锁 Interp2）。"""
    b, sid, es, ci = isa_env
    n = bootstrap_is_a_edges(ci, es, [("猫", "动物"), ("狗", "动物")], space_id=sid)
    assert n == 2, "两对 IS_A → 两条 EDGE_ISA 边"
    isa_edges = [r for r in b.select("edge", where={"edge_type": EDGE_IS_A})]
    assert len(isa_edges) == 2, "EDGE_ISA 边数 = 2"
    assert all(r["source"] == SOURCE_CONCEPTNET for r in isa_edges), "来源① SOURCE_CONCEPTNET"
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert amap, "ancestor_map 非空（boot 种边后 Interp2 可读）"
    动物 = ci.lookup("动物", sid)
    assert 动物 in amap.get(ci.lookup("猫", sid), set()), "猫 的祖先集含动物"


def test_bootstrap_is_a_edges_idempotent_same_source_skip(isa_env):
    """query_from 按源幂等 skip：同源(SOURCE_CONCEPTNET)同三元组重种 → skip（resume 跨 run 不 corrupt）。"""
    b, sid, es, ci = isa_env
    n1 = bootstrap_is_a_edges(ci, es, [("猫", "动物")], space_id=sid)
    n2 = bootstrap_is_a_edges(ci, es, [("猫", "动物")], space_id=sid)   # 同源同三元组重种
    assert n1 == 1 and n2 == 0, "同源同三元组第二次种 → skip（query_from 幂等）"
    isa_edges = [r for r in b.select("edge", where={"edge_type": EDGE_IS_A})]
    assert len(isa_edges) == 1, "重种不增边（幂等·EdgeStore.add 不去重·须 query_from skip 守）"


def test_bootstrap_is_a_edges_does_not_block_other_source(isa_env):
    """query_from 按源细化：CONCEPTNET 边不挡 observe EPI_CUE 路径（异源同三元组并存·风险4）。"""
    b, sid, es, ci = isa_env
    # boot 种 SOURCE_CONCEPTNET 边
    bootstrap_is_a_edges(ci, es, [("猫", "动物")], space_id=sid)
    # observe 路径（build_is_a_edges 硬编码 EPI_CUE/SOURCE_BARE_TEXT）种同三元组异源边
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    from pure_integer_ai.cognition.understanding.is_a import build_is_a_edge
    build_is_a_edge(es, 猫, 动物, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)
    isa_edges = [r for r in b.select("edge", where={"edge_type": EDGE_IS_A})]
    assert len(isa_edges) == 2, "异源同三元组并存（CONCEPTNET 不挡 BARE_TEXT/EPI_CUE observe 路径）"
    sources = sorted(r["source"] for r in isa_edges)
    assert sources == sorted([SOURCE_CONCEPTNET, SOURCE_BARE_TEXT]), "两条边异源"


# ============ e2e：formal_train 全 main 流程（刀0 真验收·boot 接线层） ============

def _lang_item(tokens: list[str]) -> CollectedItem:
    """语言 corpus item（MODALITY_LANGUAGE·LANG_ZH·空白已切 token）。"""
    return CollectedItem(tokens=tokens, collect_type=COLLECT_PRECEDES)


def test_knife0_formal_train_boot_isa_fires_interp2(tmp_path, monkeypatch):
    """刀0 真验收（WITH IS_A 文件）：formal_train boot 种 EDGE_ISA → Interp2 聚类写 ATTR_SLOT_ROLE。

    走真 formal_train 全 main 流程（非 _inject_isa）·证 boot 接线层让生产 caller ancestor_map 非空。
    断言：① boot EDGE_ISA 边建 ② ancestor_map 非空 ③ discovered lang 骨架 ATTR_SLOT_ROLE 非 None。
    """
    # tmp dir 写 is_a_facts_zh.txt（猫/狗/老鼠/兔子 ⊂ 动物·slot0+slot2 都须共同祖先才 joinable）
    local = tmp_path / "local"
    local.mkdir()
    (local / "is_a_facts_zh.txt").write_text(
        "猫 动物\n狗 动物\n老鼠 动物\n兔子 动物\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(local))

    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    corpus = [_lang_item(["猫", "追", "老鼠"]), _lang_item(["狗", "追", "兔子"])]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="knife0_with", rounds_per_stage=1)
    result = formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    # ① boot 种了 EDGE_ISA 边（formal_train boot 段跑·从 local/is_a_facts_zh.txt）
    isa_edges = [r for r in b.select("edge", where={"edge_type": EDGE_IS_A})]
    assert any(r["source"] == SOURCE_CONCEPTNET for r in isa_edges), \
        "boot 段须种 SOURCE_CONCEPTNET EDGE_ISA 边（formal_train boot 接线）"
    # ② ancestor_map 非空（lang 发现可读·build_isa_ancestor_map 内部建）
    amap = build_isa_ancestor_map(b, space_id=result.discovered_operators[0].skeleton_ref[0]
                                  if result.discovered_operators else 1)
    # ③ discovered lang 骨架 ATTR_SLOT_ROLE 非 None（Interp2 聚类写 slot LCA = 动物）
    lang_ops = [op for op in result.discovered_operators]
    assert lang_ops, "formal_train 须发现 ≥1 语言骨架"
    slot_roles_all = []
    for op in lang_ops:
        slot_roles_all.extend(_read_slot_roles(b, op.skeleton_ref))
    assert any(r is not None for r in slot_roles_all), \
        "WITH IS_A 文件 → Interp2 聚类写 ATTR_SLOT_ROLE（slot LCA=动物·非 None）"


def test_knife0_no_file_bit_identical_interp2_skipped(tmp_path, monkeypatch):
    """bit-identical（WITHOUT IS_A 文件）：boot 零副作用 → discovered 骨架 ATTR_SLOT_ROLE 全 None（Interp2 跳聚类）。

    无 PURE_INTEGER_AI_LOCAL_DIR（CI/生产 default）→ resolve 返 [] → bootstrap 返 0 → ancestor_map 空 →
    Interp2 has_isa=False → 跳 LCA 聚类 → 骨架全 PARAM 无 ATTR_SLOT_ROLE（既有行为 bit-identical）。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)   # 模拟 CI/生产 default 无文件
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    corpus = [_lang_item(["猫", "追", "老鼠"]), _lang_item(["狗", "追", "兔子"])]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="knife0_without", rounds_per_stage=1)
    result = formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    # 无文件 → boot 不种 SOURCE_CONCEPTNET 边（零副作用）
    isa_edges = [r for r in b.select("edge", where={"edge_type": EDGE_IS_A})]
    assert not any(r["source"] == SOURCE_CONCEPTNET for r in isa_edges), \
        "无 IS_A 文件 → boot 零副作用（无 SOURCE_CONCEPTNET EDGE_ISA 边）"
    # discovered 骨架 ATTR_SLOT_ROLE 全 None（Interp2 跳聚类·既有行为 bit-identical）
    lang_ops = [op for op in result.discovered_operators]
    assert lang_ops, "formal_train 须发现 ≥1 语言骨架（Interp2 退化·仍发现全 PARAM 骨架）"
    slot_roles_all = []
    for op in lang_ops:
        slot_roles_all.extend(_read_slot_roles(b, op.skeleton_ref))
    assert all(r is None for r in slot_roles_all), \
        "WITHOUT IS_A 文件 → Interp2 跳聚类·ATTR_SLOT_ROLE 全 None（bit-identical 既有行为）"


def test_knife0_held_out_abstract_recognize_via_file(tmp_path, monkeypatch):
    """WITH IS_A 文件·recognize 抽象匹配 READ 消费 held-out token（狐狸追鸡 concept_binding[0]=狐狸）。

    discovered 动物类骨架 slot0 ATTR_SLOT_ROLE=动物（Interp2 fired·前测验）·held-out 狐狸追鸡 input token
    沿 IS_A 可达动物 → 抽象命中（structure_discover.py:1116-1121）·concept_binding[0]=狐狸。

    **诚实边界（对抗审点 1）**：PARAM slot 无 ATTR_SLOT_ROLE 时也接受任何 token（:1124）·故 concept_binding[0]=狐狸
    在 WITHOUT-file 时也成立（PARAM 兜底）·**非「刀0 抽象路径独有」证明**。刀0 独有性判据是 ATTR_SLOT_ROLE 写盘
    （test_knife0_formal_train_boot_isa_fires_interp2 已验）。本测验证：① boot 种 held-out token（狐狸/鸡）概念点
    ② recognize READ 消费 input 真跑（concept_binding 真记录 held-out token ref）·非 PARAM 兜底的纸面 hit。
    """
    local = tmp_path / "local"
    local.mkdir()
    (local / "is_a_facts_zh.txt").write_text(
        "猫 动物\n狗 动物\n老鼠 动物\n兔子 动物\n狐狸 动物\n鸡 动物\n",
        encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(local))

    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    # 发现集：猫追老鼠/狗追兔子（动物类·≥K=2 发现）·held-out：狐狸追鸡（recognize）
    corpus = [
        _lang_item(["猫", "追", "老鼠"]),
        _lang_item(["狗", "追", "兔子"]),
        _lang_item(["狐狸", "追", "鸡"]),   # held-out
    ]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="knife0_heldout", rounds_per_stage=1)
    result = formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    # discovered 动物类骨架 slot0 ATTR_SLOT_ROLE=动物（Interp2 聚类 fired·slot LCA 写盘）
    lang_ops = [op for op in result.discovered_operators]
    assert lang_ops, "formal_train 须发现 ≥1 语言骨架"
    slot_roles_all = []
    for op in lang_ops:
        slot_roles_all.extend(_read_slot_roles(b, op.skeleton_ref))
    assert any(r is not None for r in slot_roles_all), \
        "WITH file → discovered 骨架 ATTR_SLOT_ROLE 非 None（Interp2 fired）"
    # recognize 狐狸追鸡命中·concept_binding[0]=狐狸（抽象匹配·held-out token ref 真记录·READ 消费）
    assert result.recognitions, "recognize 须产命中（held-out 狐狸追鸡）"
    ci_post = ConceptIndex(b)
    sid = lang_ops[0].skeleton_ref[0]
    狐狸_ref = ci_post.lookup("狐狸", sid)
    assert 狐狸_ref is not None, "boot 种边后 狐狸 概念点须存在"
    found_fox_binding = False
    for rec in result.recognitions:
        cb = getattr(rec, "concept_binding", None)
        if not cb:
            continue
        for ref in cb:
            r = tuple(ref) if isinstance(ref, (list, tuple)) else ref
            if r == 狐狸_ref:
                found_fox_binding = True
                break
        if found_fox_binding:
            break
    assert found_fox_binding, \
        "recognize concept_binding 须含 狐狸（READ 消费·held-out token ref 真记录·非纸面 hit）"
