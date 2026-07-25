"""入手④ ConceptNet Causes loader 测试（总收口 §三簇1入手④·镜像刀0 IS_A loader 范式）。

入手④ = boot 时种 EDGE_CAUSES 边（formal_train make_train_context 后·observe 前）·给 CAUSES 外部 R6 独立源
（非仅 cue 自产/LLM）。来源① ConceptNet Causes 有向三元组（cause Causes effect·照搬不反转·M1·§8.1c-bis·
EPI_STRUCTURED·外部数据非 core enum·守「不写死」）。

**入手④ 验收判据**（镜像刀0）：
  - WITH causes_facts 文件 → boot 种 SOURCE_CONCEPTNET EDGE_CAUSES 边（cause→effect·EPI_STRUCTURED）。
  - WITHOUT 文件（CI/生产 default）→ boot 零副作用 → 无 SOURCE_CONCEPTNET EDGE_CAUSES 边（bit-identical）。
  - CAUSES 接 reward 反传（异 IS_A）·active 文件改变训练 reward（预期·R6 外部因果信号）·本测验 boot 接线层。

铁律：纯整数 / 确定性 bit-identical / 不写死（外部文件·core 不 import）/ E5 graceful（错行/缺文件不崩）。
诚实边界：因果真伪/方向 = 外部数据责任（ConceptNet 可错·stable≠correct·#479 墙·照搬不反转不校验）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import (
    EdgeStore, SOURCE_CONCEPTNET, SOURCE_BARE_TEXT, EPI_STRUCTURED, EPI_CUE,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.understanding.causes import bootstrap_causes_edges
from pure_integer_ai.experiments.collection import (
    load_causes_facts_file, resolve_causes_facts, CollectedItem, COLLECT_PRECEDES,
)


# ---- fixtures ----

@pytest.fixture
def causes_env():
    """bootstrap_causes_edges 单测环境（dict backend·core space）。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ci
    b.close()


# ============ unit：load_causes_facts_file（E10 纯本地读·E5 graceful） ============

def test_load_causes_facts_file_parses_pairs(tmp_path):
    """正常文件：每行 cause effect·#注释/空行 skip·中段忽略（容错·支持带指向词）。"""
    f = tmp_path / "causes_facts_zh.txt"
    f.write_text(
        "# 注释行 skip\n"
        "雨 地湿\n"
        "\n"   # 空行 skip
        "太阳 导致 升温\n"   # 中段指向词忽略·首段 cause 末段 effect
        "火 烟\n",
        encoding="utf-8")
    pairs = load_causes_facts_file(str(f))
    assert pairs == [("雨", "地湿"), ("太阳", "升温"), ("火", "烟")], \
        "loader 解析 cause/effect 对·注释/空行 skip·中段忽略"


def test_load_causes_facts_file_malformed_lines_skipped(tmp_path):
    """E5 graceful：格式错行（<2 段）skip + 不抛崩·自环 skip。"""
    f = tmp_path / "causes_facts_zh.txt"
    f.write_text(
        "雨 地湿\n"
        "孤立词\n"   # 格式错（<2 段）skip
        "火 火\n"   # 自环 skip
        "太阳 升温\n",
        encoding="utf-8")
    pairs = load_causes_facts_file(str(f))
    assert pairs == [("雨", "地湿"), ("太阳", "升温")], \
        "格式错行 + 自环 skip·不抛崩（E5 graceful）"


def test_load_causes_facts_file_missing_returns_empty(tmp_path):
    """E5 graceful：文件不存在 → 返 []（不抛崩）。"""
    assert load_causes_facts_file(str(tmp_path / "nonexistent.txt")) == [], \
        "文件不存在 → 返空 list（E5 graceful·不抛崩）"


def test_load_causes_facts_file_strips_bom(tmp_path):
    """utf-8-sig 自动 strip BOM（镜像 IS_A 对抗审点 2）：BOM 文件首行注释正确识别。"""
    f = tmp_path / "causes_facts_zh.txt"
    content = "# comment line\n雨 地湿\n".encode("utf-8")
    f.write_bytes(b"\xef\xbb\xbf" + content)
    pairs = load_causes_facts_file(str(f))
    assert pairs == [("雨", "地湿")], \
        "BOM 文件首行注释须正确 skip（utf-8-sig strip BOM·不把 BOM+# 当数据行污染）"


# ============ unit：resolve_causes_facts（env PURE_INTEGER_AI_LOCAL_DIR·缺文件空·bit-identical 守） ============

def test_resolve_causes_facts_reads_local_dir(tmp_path, monkeypatch):
    """resolve 按 lang 读 PURE_INTEGER_AI_LOCAL_DIR/causes_facts_{lang}.txt。"""
    f = tmp_path / "causes_facts_zh.txt"
    f.write_text("雨 地湿\n太阳 升温\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))
    assert resolve_causes_facts(LANG_ZH) == [("雨", "地湿"), ("太阳", "升温")], \
        "resolve_causes_facts 按 lang 读 causes_facts_zh.txt"


def test_resolve_causes_facts_missing_file_empty(tmp_path, monkeypatch):
    """bit-identical 守：文件不存在 → 返 []（CI/生产 default）。"""
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))   # 目录存在但无 causes_facts_zh.txt
    assert resolve_causes_facts(LANG_ZH) == [], "文件不存在 → 返空（bit-identical 守）"


def test_resolve_causes_facts_no_local_dir_empty(monkeypatch):
    """bit-identical 守：无 PURE_INTEGER_AI_LOCAL_DIR → 返 []。"""
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    assert resolve_causes_facts(LANG_ZH) == [], "无 PURE_INTEGER_AI_LOCAL_DIR → 返空"
    assert resolve_causes_facts(LANG_EN) == [], "无 PURE_INTEGER_AI_LOCAL_DIR → 返空（en 同）"


def test_resolve_causes_facts_unmapped_lang_empty(tmp_path, monkeypatch):
    """无映射的 lang（LANG_NONE 等）→ 返 []。"""
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))
    assert resolve_causes_facts(0) == [], "LANG_NONE 无文件映射 → 返空"


# ============ unit：bootstrap_causes_edges（P0 空零副作用 + query_from 按源幂等 skip） ============

def test_bootstrap_causes_edges_empty_no_side_effect(causes_env):
    """P0 bit-identical 硬守：空 pairs → return 0 + 零写盘（不调 ensure/query_from/build）。"""
    b, sid, es, ci = causes_env
    concept_before = len(b.select("concept_node", where=None))
    edge_before = len(b.select("edge", where=None))
    n = bootstrap_causes_edges(ci, es, [], space_id=sid)
    assert n == 0, "空 pairs → return 0"
    assert len(b.select("concept_node", where=None)) == concept_before, "空 pairs 零概念点副作用"
    assert len(b.select("edge", where=None)) == edge_before, "空 pairs 零边副作用"


def test_bootstrap_causes_edges_builds_edges(causes_env):
    """种 pairs → EDGE_CAUSES 边建（cause→effect·SOURCE_CONCEPTNET·EPI_STRUCTURED）。"""
    b, sid, es, ci = causes_env
    n = bootstrap_causes_edges(ci, es, [("雨", "地湿"), ("太阳", "升温")], space_id=sid)
    assert n == 2, "两对 CAUSES → 两条 EDGE_CAUSES 边"
    causes_edges = [r for r in b.select("edge", where={"edge_type": EDGE_CAUSES})]
    assert len(causes_edges) == 2, "EDGE_CAUSES 边数 = 2"
    assert all(r["source"] == SOURCE_CONCEPTNET for r in causes_edges), "来源① SOURCE_CONCEPTNET"
    assert all(r["epistemic_origin"] == EPI_STRUCTURED for r in causes_edges), "EPI_STRUCTURED（§8.1c-bis 来源①）"
    # 方向：from=cause to=effect（ConceptNet Causes 照搬不反转·M1）
    雨 = ci.lookup("雨", sid)
    地湿 = ci.lookup("地湿", sid)
    assert 雨 is not None and 地湿 is not None, "cause/effect 概念点须 ensure 建"
    dirs = {(r["space_id_from"], r["local_id_from"], r["space_id_to"], r["local_id_to"])
            for r in causes_edges}
    assert (雨[0], 雨[1], 地湿[0], 地湿[1]) in dirs, "方向 from=cause(雨) to=effect(地湿)·照搬不反转"


def test_bootstrap_causes_edges_idempotent_same_source_skip(causes_env):
    """query_from 按源幂等 skip：同源(SOURCE_CONCEPTNET)同三元组重种 → skip（resume 跨 run 不 corrupt）。"""
    b, sid, es, ci = causes_env
    n1 = bootstrap_causes_edges(ci, es, [("雨", "地湿")], space_id=sid)
    n2 = bootstrap_causes_edges(ci, es, [("雨", "地湿")], space_id=sid)   # 同源同三元组重种
    assert n1 == 1 and n2 == 0, "同源同三元组第二次种 → skip（query_from 幂等）"
    causes_edges = [r for r in b.select("edge", where={"edge_type": EDGE_CAUSES})]
    assert len(causes_edges) == 1, "重种不增边（幂等·EdgeStore.add 不去重·须 query_from skip 守）"


def test_bootstrap_causes_edges_does_not_block_other_source(causes_env):
    """query_from 按源细化：CONCEPTNET 边不挡 observe EPI_CUE 路径（异源同三元组并存·镜像 IS_A 风险4）。"""
    b, sid, es, ci = causes_env
    # boot 种 SOURCE_CONCEPTNET 边
    bootstrap_causes_edges(ci, es, [("雨", "地湿")], space_id=sid)
    # observe 路径（build_causes_edges·cue 来源 EPI_CUE/SOURCE_BARE_TEXT）种同三元组异源边
    雨 = ci.ensure("雨", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    地湿 = ci.ensure("地湿", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    from pure_integer_ai.cognition.understanding.causes import build_causes_edges
    build_causes_edges(es, [雨, 地湿], structured_pairs=[(0, 1)],
                       cue_pairs=[], source=SOURCE_BARE_TEXT, space_id=sid)
    causes_edges = [r for r in b.select("edge", where={"edge_type": EDGE_CAUSES})]
    assert len(causes_edges) == 2, "异源同三元组并存（CONCEPTNET 不挡 BARE_TEXT/EPI_CUE observe 路径）"
    sources = sorted(r["source"] for r in causes_edges)
    assert sources == sorted([SOURCE_CONCEPTNET, SOURCE_BARE_TEXT]), "两条边异源"


def test_bootstrap_causes_edges_self_loop_skipped(causes_env):
    """自环（cause==effect）不建边（_insert_causes:57 a==b 守·loader 早跳·此处验 builder 层）。"""
    b, sid, es, ci = causes_env
    n = bootstrap_causes_edges(ci, es, [("雨", "雨")], space_id=sid)
    assert n == 0, "自环不建边（_insert_causes a==b 守）"
    causes_edges = [r for r in b.select("edge", where={"edge_type": EDGE_CAUSES})]
    assert len(causes_edges) == 0, "自环零边"


# ============ e2e：formal_train 全 main 流程（入手④ 真验收·boot 接线层） ============

def _lang_item(tokens: list[str]) -> CollectedItem:
    """语言 corpus item（MODALITY_LANGUAGE·LANG_ZH·空白已切 token）。"""
    return CollectedItem(tokens=tokens, collect_type=COLLECT_PRECEDES)


def test_causes_loader_formal_train_boot_seeds_edges(tmp_path, monkeypatch):
    """入手④ 真验收（WITH causes_facts 文件）：formal_train boot 种 SOURCE_CONCEPTNET EDGE_CAUSES 边。

    走真 formal_train 全 main 流程·证 boot 接线层（make_train_context 后·observe 前）种外部 CAUSES 边。
    断言：① boot SOURCE_CONCEPTNET EDGE_CAUSES 边建 ② 方向 cause→effect ③ EPI_STRUCTURED。
    """
    local = tmp_path / "local"
    local.mkdir()
    (local / "causes_facts_zh.txt").write_text(
        "雨 地湿\n太阳 升温\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(local))

    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    corpus = [_lang_item(["猫", "追", "老鼠"]), _lang_item(["狗", "追", "兔子"])]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="causes_with", rounds_per_stage=1)
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    # ① boot 段种了 SOURCE_CONCEPTNET EDGE_CAUSES 边（formal_train boot 接线）
    causes_edges = [r for r in b.select("edge", where={"edge_type": EDGE_CAUSES})
                    if r["source"] == SOURCE_CONCEPTNET]
    assert len(causes_edges) >= 2, \
        "boot 段须种 ≥2 SOURCE_CONCEPTNET EDGE_CAUSES 边（雨→地湿 + 太阳→升温）"
    # ② EPI_STRUCTURED（§8.1c-bis 来源①）
    assert all(r["epistemic_origin"] == EPI_STRUCTURED for r in causes_edges), \
        "boot CAUSES 边须 EPI_STRUCTURED"


def test_causes_loader_no_file_bit_identical(tmp_path, monkeypatch):
    """bit-identical（WITHOUT causes_facts 文件）：boot 零副作用 → 无 SOURCE_CONCEPTNET EDGE_CAUSES 边。

    无 PURE_INTEGER_AI_LOCAL_DIR（CI/生产 default）→ resolve 返 [] → bootstrap 返 0 → 零 SOURCE_CONCEPTNET CAUSES 边
    （既有行为 bit-identical）。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)   # 模拟 CI/生产 default 无文件
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    corpus = [_lang_item(["猫", "追", "老鼠"]), _lang_item(["狗", "追", "兔子"])]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="causes_without", rounds_per_stage=1)
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    # 无文件 → boot 不种 SOURCE_CONCEPTNET CAUSES 边（零副作用）
    causes_edges = [r for r in b.select("edge", where={"edge_type": EDGE_CAUSES})
                    if r["source"] == SOURCE_CONCEPTNET]
    assert len(causes_edges) == 0, \
        "无 causes_facts 文件 → boot 零副作用（无 SOURCE_CONCEPTNET EDGE_CAUSES 边·bit-identical）"
