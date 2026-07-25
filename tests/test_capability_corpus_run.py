"""#727 算术+代码 corpus 跑测试（片4a·验证片）。

doc/重来_任务0727_corpus跑.md 决断1-5 + 反 theater 坏 corpus + fixture_size_note 守。

覆盖：
  - 算术 corpus 跑（验③）：load_arith_corpus → ③ PASS permille=1000 + ③ × G5 total≥10 active=0
  - 代码 corpus 跑（验⑤⑥）：load_code_corpus → ⑤ × G5 total≥10（Mode A verify·非 Mode B）+ ⑥ G3a/G3b ALIVE 一致性（#889 stale 修正）
  - 反 theater 坏算术 corpus：load_arith_bad_corpus → ③ FAIL permille=0 + ③ × G5 active=total（全 veto）
  - 反 theater 对比：good vs bad ③ status 不同（PASS vs FAIL）·证 ③ 判据 corpus-sensitive 非 theater
  - fixture_size_note 守：tiny corpus（total<10）→ STATISTICAL_NOISE + footnote 标
  - bit-identical：同 corpus 同 seed 两跑 → to_json 一致（含 fixture_size_note + g_attribution）

诚实边界：
  - Mode A verify（PRE·weaning_phase=WEANING_PRE 默认）·非 Mode B cross-verify（POST·defer 独立 session）
  - flat_floors（FLOOR_*=0）绕 stage 进度门（arith/code 无 CAUSES → causes_coverage=0 自然挡 STAGE3·
    flat_floors 让 verify path 跑·非真实端到端进度·harness 探针非生产）
  - 玩具 PASS 不覆盖既有 FAIL：换 corpus 是补证据非翻案（决断5）
  - stable≠correct：corpus 跑验机制活 + 统计 permille·非语义正确（#479 墙）

铁律：纯整数 / bit-identical / 反 theater（坏 corpus 必异·good vs bad status 不同）/ 不纸面闭合。
"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, DefaultRoundRunner,
)
from pure_integer_ai.experiments.capability_exam import (
    run_capability_exam, CapabilityReport,
    DIM_COMPUTE, DIM_LONG_CODE, DIM_THREE_RING,
    STATUS_PASS, STATUS_FAIL, STATUS_NE,
    G_DOOR_G3A, G_DOOR_G3B, G_DOOR_G5,
    G_DEAD_LEAK, G_DEAD_DESIGN, G_ALIVE,
    FIXTURE_SIZE_MIN, FIXTURE_NOTE_OK, FIXTURE_NOTE_NOISE,
)
from pure_integer_ai.experiments.collection import (
    load_arith_corpus, load_arith_bad_corpus, load_code_corpus, load_corpus,
    CORPUS_KIND_ARITH, CORPUS_KIND_ARITH_BAD, CORPUS_KIND_CODE, CORPUS_KIND_LANG,
    CollectedItem,
)


# ---- flat_floors：arith/code 无 CAUSES → causes_coverage=0 自然挡 STAGE3·置 0 让 verify path 跑 ----
# 镜像 test_capability_g_attribution.py:408-412 + test_experiments.py:129-135 范式（harness 探针·非生产）

def _flat_floors(monkeypatch):
    """arith/code corpus 跑前置：FLOOR_*=0 绕 stage 进度门（verify path 探针）。"""
    from pure_integer_ai.config import gates
    from pure_integer_ai.training import stages as _st
    monkeypatch.setattr(_st, "FLOOR_GRAPH_SIZE_S1", 0)
    monkeypatch.setattr(_st, "FLOOR_CAUSES_COV_S2", 0)
    monkeypatch.setattr(_st, "FLOOR_CONDUCTION_S3", 0)
    monkeypatch.setattr(_st, "FLOOR_PROMOTE_S4", 0)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)


# ============ loader 单元 ============

def test_load_corpus_loaders_return_distinct_items():
    """loader 返非空 list[CollectedItem]·确定性·bit-identical（同调两跑同结果）。"""
    arith = load_arith_corpus()
    bad = load_arith_bad_corpus()
    code = load_code_corpus()
    assert len(arith) == 12, f"arith corpus 须 12 样本·得 {len(arith)}"
    assert len(bad) == 2, f"bad arith 须 2 样本·得 {len(bad)}"
    assert len(code) == 10, f"code corpus 须 10 样本·得 {len(code)}"
    # 确定性：两调同结果（in-memory 种子·bit-identical）
    assert load_arith_corpus() == arith
    assert load_code_corpus() == code
    # modality 正确
    assert all(it.modality == 7 for it in arith)   # MODALITY_ARITH=7
    assert all(it.modality == 6 for it in code)    # MODALITY_CODE=6
    # arith 有 specs（Mode A verify + task-driven 触发）
    assert all(it.arith_specs for it in arith)
    assert all(it.code_specs for it in code)


def test_load_corpus_by_kind_dispatch():
    """load_corpus(kind) 分发正确·未知 kind raise。"""
    assert load_corpus(CORPUS_KIND_ARITH) == load_arith_corpus()
    assert load_corpus(CORPUS_KIND_ARITH_BAD) == load_arith_bad_corpus()
    assert load_corpus(CORPUS_KIND_CODE) == load_code_corpus()
    assert load_corpus(CORPUS_KIND_LANG) == []   # lang 既有 _causal_multi_sent_item 覆盖
    with pytest.raises(ValueError, match="未知 corpus kind"):
        load_corpus("nonsense")


# ============ 算术 corpus 跑（验③）============

def test_arith_corpus_compute_dim_pass(tmp_path, monkeypatch):
    """算术 corpus 跑 → ③计算 PASS permille=1000（generalization·discover 2 + recognize 10 + vm_proof 全验）。

    决断2：PASS 且 rate_permille>0（非死写默认）。③ × G5 attribution total≥10 active=0（Mode A verify 全 pass）。
    附验：⑤ Mode A task-driven（arith_specs 触发·square arity=1 匹配 spec arity=1）→ ⑤ PASS permille=1000
    （collection.py load_arith_corpus docstring 声称·此处验非纸面闭合）。fixture_size_note=OK（total≥10）。
    """
    _flat_floors(monkeypatch)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap_arith"), run_id="cap_arith")
    rep = run_capability_exam(cfg, load_arith_corpus(), backend=b, runner=DefaultRoundRunner())

    assert isinstance(rep, CapabilityReport)
    # ③ PASS permille=1000
    d3 = rep.dimensions[DIM_COMPUTE]
    assert d3.status == STATUS_PASS, f"③ 须 PASS·得 {d3.status}（permille={d3.permille}）"
    assert d3.permille == 1000, f"③ permille 须 1000（全验）·得 {d3.permille}"
    assert d3.permille > 0, "③ 须 permille>0（非死写 0）"
    # ③ × G5 attribution：Mode A verify episodes total≥10 active=0（全 pass·无 veto）
    cell = rep.g_attribution[DIM_COMPUTE][G_DOOR_G5]
    assert cell["total"] >= 10, f"③ × G5 total 须≥10·得 {cell['total']}"
    assert cell["active"] == 0, f"③ × G5 active 须 0（全 pass）·得 {cell['active']}"
    # ⑤ Mode A task-driven（STEP2 #889：⑤取严 NE·generate 字面零测·Mode A rate 进 evidence）
    # 验 load_arith_corpus docstring 声称非纸面闭合（审2 P2-3 修·Mode A 真活进 evidence 非 status）
    d5 = rep.dimensions[DIM_LONG_CODE]
    assert d5.status == STATUS_NE, f"⑤ 须 NE（#889 取严·generate 字面零测）·得 {d5.status}"
    assert d5.permille == -1, f"⑤ permille 须 -1（NE·#889 取严）·得 {d5.permille}"
    # Mode A task-driven 真活进 evidence（rate=1000·verified=12）
    assert any("Mode A task-driven" in e and "1000" in e for e in d5.evidence), (
        f"⑤ evidence 须含 Mode A rate=1000（真活·#889 Mode A 进 evidence）·得 {d5.evidence}")
    # fixture_size_note=OK
    assert rep.fixture_size_note == FIXTURE_NOTE_OK


# ============ 反 theater 坏算术 corpus ============

def test_arith_bad_corpus_compute_dim_fail(tmp_path, monkeypatch):
    """坏算术 corpus 跑 → ③计算 FAIL permille=0（held_out=0·rate=0）+ ③ × G5 active=total（全 veto）。

    决断2 反 theater：坏 corpus（全错 spec）→ ③必 FAIL。若 harness 报 PASS → abort（theater）。
    ③ × G5 attribution：active=total（全 veto·reward=0）·permille=1000（veto 率 100%）。
    """
    _flat_floors(monkeypatch)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap_bad"), run_id="cap_bad")
    rep = run_capability_exam(cfg, load_arith_bad_corpus(), backend=b, runner=DefaultRoundRunner())

    # ③ FAIL permille=0
    d3 = rep.dimensions[DIM_COMPUTE]
    assert d3.status == STATUS_FAIL, f"坏 corpus ③ 须 FAIL·得 {d3.status}（反 theater）"
    assert d3.permille == 0, f"③ permille 须 0（held_out=0）·得 {d3.permille}"
    # ③ × G5：active=total（全 veto·wrong spec → reward=0）
    cell = rep.g_attribution[DIM_COMPUTE][G_DOOR_G5]
    assert cell["total"] > 0, "坏 corpus ③ × G5 须有 verify episodes"
    assert cell["active"] == cell["total"], (
        f"坏 corpus ③ × G5 active 须=total（全 veto）·active={cell['active']} total={cell['total']}")
    # evidence_eps 溯源（active 格带 episode index）
    assert len(cell["evidence_eps"]) == cell["active"], (
        f"evidence_eps 须=active 数·得 {len(cell['evidence_eps'])}")


def test_arith_corpus_good_vs_bad_status_differs(tmp_path, monkeypatch):
    """反 theater 核心：good vs bad corpus ③ status 不同（PASS vs FAIL）·证 ③ 判据 corpus-sensitive。

    决断2：若 ③ 判据是 theater（死写 PASS）·good/bad 都 PASS → 测试抓到。good=PASS + bad=FAIL → 非 theater。
    """
    _flat_floors(monkeypatch)

    b1 = DictBackend()
    cfg1 = FormalTrainConfig(run_dir=str(tmp_path / "gv_good"), run_id="gv_good")
    rep_good = run_capability_exam(cfg1, load_arith_corpus(), backend=b1, runner=DefaultRoundRunner())

    b2 = DictBackend()
    cfg2 = FormalTrainConfig(run_dir=str(tmp_path / "gv_bad"), run_id="gv_bad")
    rep_bad = run_capability_exam(cfg2, load_arith_bad_corpus(), backend=b2, runner=DefaultRoundRunner())

    s_good = rep_good.dimensions[DIM_COMPUTE].status
    s_bad = rep_bad.dimensions[DIM_COMPUTE].status
    assert s_good == STATUS_PASS and s_bad == STATUS_FAIL, (
        f"good/bad ③ status 须不同（PASS vs FAIL·反 theater）·得 good={s_good} bad={s_bad}")


# ============ 代码 corpus 跑（验⑤⑥）============

def test_code_corpus_g5_attribution_and_dead_leak(tmp_path, monkeypatch):
    """代码 corpus 跑 → ⑤ × G5 attribution total≥10（Mode A verify·非 Mode B）+ ⑥ G3a/G3b ALIVE 一致性（#889 stale 修正）。

    决断3 ⑤：Mode A verify（PRE）真实命中率（total≥10·active=0 全 pass）·非 Mode B cross-verify（POST defer）。
    决断3 ⑥：G3a/G3b 在代码域 ALIVE（#889 stale 修正·classify_intent 真填·与 language 域一致·#723 归因表一致性验证）。
    """
    _flat_floors(monkeypatch)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap_code"), run_id="cap_code")
    rep = run_capability_exam(cfg, load_code_corpus(), backend=b, runner=DefaultRoundRunner())

    # ⑤ × G5 attribution：Mode A verify total≥10 active=0
    cell5 = rep.g_attribution[DIM_LONG_CODE][G_DOOR_G5]
    assert cell5["total"] >= 10, f"⑤ × G5 total 须≥10（Mode A verify）·得 {cell5['total']}"
    assert cell5["active"] == 0, f"⑤ × G5 active 须 0（全 pass）·得 {cell5['active']}"
    assert cell5["dead_state"] == G_ALIVE, "⑤ × G5 须 ALIVE（verify path 真承重）"

# ⑥ G3a/G3b ALIVE 一致性（#889 stale 修正·与 language 域同·classify_intent 真填不区分模态）
    t6 = rep.g_attribution[DIM_THREE_RING]
    assert t6[G_DOOR_G3A]["dead_state"] == G_ALIVE, "⑥ G3a 须 ALIVE（#889 stale 修正·代码域一致性）"
    assert t6[G_DOOR_G3B]["dead_state"] == G_ALIVE, "⑥ G3b 须 ALIVE（#889 stale 修正·代码域一致性）"
    # ⑥ G5 DEAD_DESIGN（code 不在 _ARITH_DOMAINS judge path·与 language 同）
    assert t6[G_DOOR_G5]["dead_state"] == G_DEAD_DESIGN, "⑥ G5 须 DEAD_DESIGN（代码域一致性）"

    # footnotes 含 G5 DEAD_DESIGN 标注（G3a/G3b ALIVE 后不标 DEAD_LEAK·#889 stale 修正）
    assert any("G5" in fn and "DEAD_DESIGN" in fn for fn in rep.footnotes), "footnote 须标 G5 DEAD_DESIGN"


# ============ fixture_size_note 守（决断5·防误读）============

def test_fixture_size_note_noise_for_tiny_corpus(tmp_path, monkeypatch):
    """tiny corpus（total episode<10）→ fixture_size_note=STATISTICAL_NOISE + footnote 标。

    决断5：total<10 强制标 STATISTICAL_NOISE·不计 PASS/FAIL 为定论。
    用 1-item arith corpus + rounds_per_stage=1 → total<10 → STATISTICAL_NOISE。
    """
    _flat_floors(monkeypatch)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap_tiny"), run_id="cap_tiny",
                            rounds_per_stage=1)
    tiny_corpus = load_arith_bad_corpus()[:1]   # 1 item → 少 episode
    rep = run_capability_exam(cfg, tiny_corpus, backend=b, runner=DefaultRoundRunner())

    assert rep.fixture_size_note == FIXTURE_NOTE_NOISE, (
        f"tiny corpus 须 STATISTICAL_NOISE·得 {rep.fixture_size_note}")
    # footnote 标 STATISTICAL_NOISE
    assert any("STATISTICAL_NOISE" in fn for fn in rep.footnotes), (
        "footnote 须含 STATISTICAL_NOISE 标注")
    # to_json 含 fixture_size_note
    j = rep.to_json()
    assert j["fixture_size_note"] == FIXTURE_NOTE_NOISE


def test_fixture_size_note_ok_for_adequate_corpus(tmp_path, monkeypatch):
    """adequate corpus（total≥10）→ fixture_size_note=OK（无 STATISTICAL_NOISE footnote）。"""
    _flat_floors(monkeypatch)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap_ok"), run_id="cap_ok")
    rep = run_capability_exam(cfg, load_arith_corpus(), backend=b, runner=DefaultRoundRunner())

    assert rep.fixture_size_note == FIXTURE_NOTE_OK, (
        f"adequate corpus 须 OK·得 {rep.fixture_size_note}")
    # 无 STATISTICAL_NOISE footnote
    assert not any("STATISTICAL_NOISE" in fn for fn in rep.footnotes), (
        "adequate corpus 不应标 STATISTICAL_NOISE")


# ============ bit-identical（同 corpus 同 seed 两跑 → to_json 一致）============

def test_corpus_run_bit_identical(tmp_path, monkeypatch):
    """同 corpus 两独立 backend 跑 → CapabilityReport.to_json 一致（决断4·bit-identical）。

    归一化 run_id（两跑 run_id 必不同·非核心）。fixture_size_note + g_attribution + dimensions 全一致。
    """
    _flat_floors(monkeypatch)

    def run_once(run_id: str):
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / run_id), run_id=run_id)
        rep = run_capability_exam(cfg, load_arith_corpus(), backend=b, runner=DefaultRoundRunner())
        j = rep.to_json()
        j["run_id"] = "NORMALIZED"   # 归一化 run_id
        return j

    j1 = run_once("bi_1")
    j2 = run_once("bi_2")
    assert j1 == j2, (
        "同 corpus 两跑 to_json 须 bit-identical（归一化 run_id 后）·决断4 守")


def test_corpus_run_code_bit_identical(tmp_path, monkeypatch):
    """代码 corpus 同样 bit-identical（决断4·两独立 backend 跑）。"""
    _flat_floors(monkeypatch)

    def run_once(run_id: str):
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / run_id), run_id=run_id)
        rep = run_capability_exam(cfg, load_code_corpus(), backend=b, runner=DefaultRoundRunner())
        j = rep.to_json()
        j["run_id"] = "NORMALIZED"
        return j

    j1 = run_once("cbi_1")
    j2 = run_once("cbi_2")
    assert j1 == j2, "代码 corpus 两跑 to_json 须 bit-identical"


# ============ 反 theater：玩具 PASS 不覆盖既有 FAIL（决断5）============

def test_corpus_run_does_not_override_existing_fail_judgment(tmp_path, monkeypatch):
    """决断5 反 theater：换 corpus 是补证据非翻案·既有 FAIL（language 域跑出）不被 arith PASS 覆盖。

    arith corpus 跑出 ③ PASS（generalization）·但这不改变"language 域 ③ FAIL"的既有判定。
    验：arith report 自洽（③ PASS on arith）·但不声称修复 language 域 ③（不同 corpus 不同判据·补证据非翻案）。
    footnotes 含 stable≠correct（验机制活非语义正确）+ #479 墙（非真独立源）。
    """
    _flat_floors(monkeypatch)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap_noov"), run_id="cap_noov")
    rep = run_capability_exam(cfg, load_arith_corpus(), backend=b, runner=DefaultRoundRunner())

    # arith ③ PASS（自洽）
    assert rep.dimensions[DIM_COMPUTE].status == STATUS_PASS
    # 但 footnotes 诚实标 stable≠correct + #479（PASS 非语义正确·非真墙突破）
    assert any("stable≠correct" in fn for fn in rep.footnotes), "须标 stable≠correct（PASS 非语义正确）"
    assert any("#479" in fn for fn in rep.footnotes), "须标 #479 墙（非真独立源验证）"
