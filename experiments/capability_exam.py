"""experiments.capability_exam — 全方位能力考核 harness（片1 MVP·2026-07-07）。

正式训练前的**终极 gate**（doc/重来_全方位能力考核设计_2026-07-07.md）。
回答纠偏初心：现在能不能从语言中学到完备能力了？

  run_capability_exam(config, corpus, *, backend, teacher, runner) -> CapabilityReport

复用 formal_train()（不重造训练循环）+ 训练前后 strength snapshot（§D 反 theater 第一层·学到代理）
+ FormalTrainResult 字段投影到 8 能力维度（§三.1 阈值 pre-registration）+ 墙边界 footnote。

**§三.1 阈值 pre-registration**（实施前定·跑完不许调·从既有测断言回填·防"为通过降阈值"）：
  ①概念：graph_size>0 阈值=1·件1/3/6 仅机制测 → NE
  ②结构：causes_coverage ≥ 500
  ③计算：generalization.rate_permille ≥ 500（若 None → NE）
  ④长文本：NE（零测·threshold=-1）
  ⑤长代码：generate=NE + Mode B（generate.rate_permille ≥ 500·若有）·混合 status
  ⑥三环：collapse 三柱全 ok + strength_delta_total>0 → MECHANISM_LIVE（机制接通非能力达成·#479 墙）·G3a/G3b ALIVE（STEP2 #889·intent 已动态化）
  ⑦初心：strength_delta_total>0 → MECHANISM_LIVE（reward 通路活代理·非真学到·#479 墙）
  ⑧记忆：memory_item 写入>0 + 消费者触发=0 → **FAIL**（非缺口美化·审 2 P0-3）

**§D 反 theater 三层判据**（替代 ON/OFF delta·避开 TRAINING_MODE 陷阱）：
  1. strength 训练前后 delta（学到代理·片1 落）— H4 闭环真训练增量·不依赖 TRAINING_MODE gate
  2. 反 theater e2e 锚点自检（偏真·片2 落·TODO）
  3. 反向回归必做（可证伪·片2 落·TODO）

**诚实边界（必须在 footnotes 标）**：
  - strength_delta 是"reward 通路活"代理·**非"真独立源验证学到"**（#479 墙·experience feed 构造性）
  - ④⑤ generate 零测 = NE ≠ PASS
  - ⑧记忆消费者=0 → FAIL（非缺口美化）
  - language 域 G3a/G3b ALIVE（STEP2 #889·intent 已动态化·classify_intent 真填）·⑥⑦ 机制接通 → MECHANISM_LIVE 非 PASS（能力达成未确认·#479 墙）
  - 断奶：can_ween(语言域)永False 是决策层 truth（E2 三路堵·非'统计层不能学'）·统计层持续学习就绪判据另建（非 can_ween·连 A/C 编码接地）·算术域 W7 can_ween=True（机制接通·真泛化 defer W8）
  - stable≠correct（考核验统计完备非语义正确）
  - 物种差异：离散图 vs 连续自回归·统计表现接近非等同

铁律：纯整数（assert_no_float 守）/ bit-identical（to_json sort_keys）/ 单向依赖（experiments →
cognition/storage/training·lint import_direction 守）/ 复用 formal_train（不重造）/ 不纸面闭合（NE 真标
NE·FAIL 真标 FAIL·不为考过降阈值）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, SOURCE_MATH
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.layer0_anchor import count_layer0
from pure_integer_ai.cognition.shared.types import (
    CodeSpec, Episode, MODALITY_ARITH, MODALITY_CODE, MODALITY_LANGUAGE,
    DOMAIN_MATH, LANG_NONE,
)
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES, COLLECT_CAUSES
from pure_integer_ai.experiments.formal_train import (
    formal_train, FormalTrainConfig, FormalTrainResult,
    GeneralizationSummary, GenerateSummary, DefaultRoundRunner,
)
from pure_integer_ai.config import gates
from pure_integer_ai.training import stages
from pure_integer_ai.training.stages import StageMetrics


# ---- 维度名（§三.1 8 维度·固定序） ----
DIM_CONCEPT = "①概念"          # 概念一等公民（种概念/IS_A/REL_*8）
DIM_STRUCTURE = "②结构"        # 结构一等公民（typed edges/DAG/汇聚）
DIM_COMPUTE = "③计算"          # 计算一等公民（VM/递归/Mode B·最强）
DIM_LONG_TEXT = "④长文本"      # 多段多章连贯生成（零测·NE）
DIM_LONG_CODE = "⑤长代码"      # 代码 generate（零测·NE）+ Mode B
DIM_THREE_RING = "⑥三环"       # 三大建模闭环（observe/dag_path/judge+reward）
DIM_INTENT = "⑦初心"           # 初心判据（8 件愿景"学到"）
DIM_MEMORY = "⑧记忆"           # 记忆动力学（写入+消费）

DIM_ORDER: tuple[str, ...] = (
    DIM_CONCEPT, DIM_STRUCTURE, DIM_COMPUTE,
    DIM_LONG_TEXT, DIM_LONG_CODE, DIM_THREE_RING,
    DIM_INTENT, DIM_MEMORY,
)

# ---- 阈值 pre-registration（§三.1·实施前定·跑完不许调） ----
THRESH_GRAPH_SIZE = 1            # ①概念：graph_size>0（机制测·件1/3/6 仅机制测 = NE）
THRESH_CAUSES_COV = 500          # ②结构：causes_coverage ≥ COVERAGE_THRESHOLD（dag_path.py:51）
THRESH_RATE_PERMILLE = 500       # ③计算/⑤Mode B：rate_permille ≥ 既有 e2e 断言
THRESH_STRENGTH_DELTA = 1        # ⑥三环/⑦初心：strength_delta_total>0

# ---- #727 fixture 限制守（决断5·防误读） ----
# total episode < FIXTURE_SIZE_MIN → STATISTICAL_NOISE（fixture 不足·不计 PASS/FAIL 为定论）。
# 跑出 FAIL 且 total<10 → "无法判定"（噪声）·footnote 标 STATISTICAL_NOISE。
# 跑出 FAIL 且 total≥10 且 permille=0 → "真缺口"（机制死或 corpus 全错）。
FIXTURE_SIZE_MIN = 10
FIXTURE_NOTE_OK = "OK"
FIXTURE_NOTE_NOISE = "STATISTICAL_NOISE"

# ---- status 枚举（字符串·jsonl 友好） ----
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NE = "NE"        # Not Examined（零测·≠ PASS·≠ FAIL）
STATUS_ABORT = "ABORT"  # 反 theater 锚点自检 fail（片2 落·片1 不产 ABORT）
# 机制接通（reward 通路活/三柱 ok）但能力达成未确认（#479 墙·strength_delta 是 reward 通路活代理非真学到）。
# 第三态：≠ PASS（非能力达成·禁偷渡）·≠ NE（已考核非零测·strength_delta/pillars 真测了）·≠ FAIL（机制活非缺陷·delta=0 才 FAIL）。
# ⑥⑦ 用（⑤ generate 字面零测仍 NE·③ rate 达阈仍 PASS·② cov 达阈仍 PASS）。
STATUS_MECHANISM_LIVE = "MECHANISM_LIVE"

# ---- #723 G 归因：G 门名 + dead_state 枚举（§D 判据偏真闭合最后一环） ----
# G 门 5 扇（judge.py:200-268·ΠG 合成·各 True=该门 veto reward=0）：
#   G4  J4 闭合否决（output 未闭合 sink → veto·judge.py:215）
#   G2p J2 意图否决（output 未 reached_sink → veto·judge.py:221）
#   G3a J3 因果否决（causal-reasoning 意图无 CAUSES 锚 → veto·judge.py:236）
#   G3b 反事实否决（含值主张意图 counterfactual 失败 → veto·judge.py:243）
#   G5  自证机否决（arith/code 域 self_proof 失败 → veto·judge.py:251）
G_DOOR_G4 = "G4"
G_DOOR_G2P = "G2p"
G_DOOR_G3A = "G3a"
G_DOOR_G3B = "G3b"
G_DOOR_G5 = "G5"
_G_DOORS: tuple[str, ...] = (G_DOOR_G4, G_DOOR_G2P, G_DOOR_G3A, G_DOOR_G3B, G_DOOR_G5)

# dead_state 枚举（决断4·防美化核心）：
G_ALIVE = "ALIVE"                  # 门活：生产 episode 真走该门 veto 路径·active 可 >0
G_DEAD_DESIGN = "DEAD_DESIGN"      # 设计性 dead：消费者根本不存在/按架构不应有·不计 fail·登记册标
G_DEAD_LEAK = "DEAD_LEAK"          # 漏洞性 dead：消费者存在但硬编码/漏写短路·计隐性 fail·独立 issue
G_NA = "N/A"                       # 维度不涉该 G 门（如 ①概念 不涉任何 G 门）

# Episode 字段名 → G 门名映射（types.py:301-305 judge_G{X}_active）
_DOOR_FIELD: dict[str, str] = {
    G_DOOR_G4: "judge_G4_active",
    G_DOOR_G2P: "judge_G2p_active",
    G_DOOR_G3A: "judge_G3a_active",
    G_DOOR_G3B: "judge_G3b_active",
    G_DOOR_G5: "judge_G5_active",
}

# episode 生产路径 → 该路径真走的 G 门（语义锚·非任意交叉·决断3 真相版）：
#   judge  : language 域 run_round_full→episode_loop→judge()·5 门全走（judge.py:200-268）
#   verify : ARITH/CODE 域 _run_verify_round·只走 G5（formal_train.py:519·Mode B cross_verify）
#   mode_a : _run_task_driven_generate·G5=False 硬编码（formal_train.py:1966·外真验非 G5 门）·不走任何 G 门
_PATH_DOORS: dict[str, tuple[str, ...]] = {
    "judge": (G_DOOR_G4, G_DOOR_G2P, G_DOOR_G3A, G_DOOR_G3B, G_DOOR_G5),
    "verify": (G_DOOR_G5,),
    "mode_a": (),
}

# 维度 × G门 → dead_state 静态表（基于 code 真相·非 doc 决断3 误标"ARITH/G4 G2p 真走 judge"）：
# **doc 决断3 偏差纠偏**：doc 原 map "DIM_ARITH/G4 G2p 真走 judge → 真填" 错——ARITH/CODE episode 走
# _run_verify_round（formal_train.py:510-523·只设 G5）非 judge()·G4/G2p 对 ③⑤ = N/A（verify 不经 judge）。
# ⑥三环 language judge：G4/G2p/G3a/G3b ALIVE·G5 DEAD_DESIGN（language 不在 _ARITH_DOMAINS·judge.py:247 设计性排除）。
# **STEP2 #889 修正 stale**（M1片2+G1+#774 落地·2026-07-10）：原标 G3a/G3b DEAD_LEAK（"intent 三 bool 硬编码 False"）
# 已过期--classify_intent（intent_classify.py:94-100）真填 is_causal/has_value_claim·formal_train.py:432-434 gate ON
# 走 classify_intent（:435-436 gate OFF 回归态硬编码 IntentType 三 bool 永 False）·:1646-1647 生产翻 M1_INTENT_CLASSIFY_MODE ON。
# e2e 核证（TM=ON+flat_floors+_causal_multi_sent_item）：8 ep 全 is_causal=True·G3a/G3b active=0（不 veto·有 CAUSES 锚·
# 非硬编码短路）-> G3a/G3b 真活（ALIVE）·active=0 因不 veto 非 dead。dead-G门前置（_three_ring_status）保留防退化（D5-enforcing）。
# ③计算 ARITH verify：G5 ALIVE·其余 N/A。⑤长代码 CODE verify：G5 ALIVE·其余 N/A（Mode A G5 不计入·mode_a 路径不走门）。
# ①②④⑦⑧ 全 N/A（无 episode judge 路径 / 判据不同源·⑦ strength_delta 非 G 门）。
_DIM_G_DEAD: dict[str, dict[str, str]] = {
    DIM_THREE_RING: {
        G_DOOR_G4: G_ALIVE, G_DOOR_G2P: G_ALIVE,
        G_DOOR_G3A: G_ALIVE, G_DOOR_G3B: G_ALIVE,
        G_DOOR_G5: G_DEAD_DESIGN,
    },
    DIM_COMPUTE: {
        G_DOOR_G4: G_NA, G_DOOR_G2P: G_NA, G_DOOR_G3A: G_NA, G_DOOR_G3B: G_NA,
        G_DOOR_G5: G_ALIVE,
    },
    DIM_LONG_CODE: {
        G_DOOR_G4: G_NA, G_DOOR_G2P: G_NA, G_DOOR_G3A: G_NA, G_DOOR_G3B: G_NA,
        G_DOOR_G5: G_ALIVE,
    },
}

# ---- 墙边界 footnote（总注·固定） ----
FOOTNOTE_WALL_479 = ("#479 墙：初心判据 experience feed 构造性·strength_delta 来源不独立——observe 建新 CAUSES 边 "
                     "base_strength=1（结构性）+ reward 反传 add_strength（学习性·须 episode_loop）。capability_exam "
                     "training_mode opt-in（生产 caller 翻 True → reward 环路触发·CI 默认 False → 零触发·delta 全结构性建边）·"
                     "⑦ strength_delta>0 → MECHANISM_LIVE 非 PASS（机制接通非能力达成·机制活冒充学到=偷渡·"
                     "D5 撞墙保 MECHANISM_LIVE/FAIL 禁偷渡 PASS）")
FOOTNOTE_WEANING = ("断奶：can_ween(语言域)永False 是决策层 truth（E2 第三条件 produced_without_teacher_anchor "
                    "三路堵·#479 truth墙/single-source 自产自验墙/外部对齐非自锚·非'统计层不能学'）·"
                    "统计层持续学习就绪判据另建（非 can_ween·连 A/C 编码接地·未建 defer）·"
                    "算术域 W7 can_ween=True（六闸门机制接通·真泛化 defer W8·考核验机制接通非断奶后稳态）")
FOOTNOTE_STABLE = "stable≠correct：考核验统计完备非语义正确"
FOOTNOTE_SPECIES = "物种差异：离散图 vs 连续自回归·统计表现接近非等同"
FOOTNOTE_LANG_MEASURES = ("lang_measures：语言域统计层产出度量（#1041 构造③④·判据④泛化率 lang_rate_permille + 判据⑤跨语言汇聚 "
                          "pure_alias_edges_seeded）= observability 信号**非断奶判据**（thresholds/真语料 defer·全统计层）·"
                          "lang_rate 相1 recognize 结构对齐口径（语言不可 vm_proof·渐近判据·observability 报告消费者非闭环）·"
                          "pure_alias_edges_seeded 度量**桥执行**（boot 种双向 PURE_ALIAS 边数·输入侧 precondition·无 edges→无汇聚可能）·"
                          "activate_candidates 收敛对称性由 P0b 桥结构保证（双向 seed + 自包含 fix）+ test_alias_bridge AB3 回归测（此处不重复测·冗余）·"
                          "生成侧真收敛（generate 按 target_lang 产 apple/苹果）**机制层已证**（test_dispatch_token_chain TC5：apple↔苹果 PURE_ALIAS→generate_output 端到端 ZH 产苹果/EN 产 apple 真字·2026-07-15 核证·2068 测零回归实证·非同义反复）·"
                          "defer 仅**规模锻炼**（76k corpus + EN-target 训练 episode·非机制 gap·课程轴B）·"
                          "CI 无 alias_facts→edges=0→无信号")

FOOTNOTE_MATH_MEASURES = ("math_measures：S5-S8 符号数学统计层产出度量（#1124·镜像 lang_measures·xform_verified/total + "
                          "xform_rate_permille·inv_verified/total + inv_rate_permille）= observability 信号**非断奶判据**·"
                          "**signal 非 criterion**（thresholds defer·全统计层·非 can_ween/truth 决策层）·"
                          "count = verified 数（cross-verify/B∘A 还原 verified·Mode A 构造性 SELF_PRODUCED·single-source 自产自验）·"
                          "rate = verified/total ×1000（最接近泛化信号·镜像 lang_rate_permille·**stable≠correct**：cross-verify 统计非 truth·采样一致非证明·#479 守）·"
                          "CI 无 symbolic specs（无 ZERO_AI_LOCAL_DIR transform/inverse 文件）→ 全 0·无信号（诚实非伪装）")


@dataclass
class DimScore:
    """单维度评分（纯整数 × 1000·NE=-1·ABORT=-2）。

    status 派生规则（严格·不允许独立写死 PASS）：
      permille ≥ threshold → PASS（能力达成）
      permille < threshold → FAIL
      零测维度 → NE（permille=-1·threshold=-1）
      机制接通（reward 通路活/三柱 ok）但能力达成未确认 → MECHANISM_LIVE（⑥⑦·#479 墙·strength_delta 是通路活代理非真学到）
      ⑥三环 G3a/G3b ALIVE（STEP2 #889·intent 已动态化）→ 按 collapse+strength_delta 判 MECHANISM_LIVE/FAIL
    """

    dim: str               # 维度名（DIM_*）
    status: str            # STATUS_PASS / STATUS_FAIL / STATUS_NE / STATUS_ABORT
    permille: int          # 量化分 0..1000（NE=-1·status=ABORT 时 -2）
    threshold: int         # 预注册阈值（§三.1·NE 时 = -1）
    evidence: list[str] = field(default_factory=list)   # 锚点 + 解释
    footnote: str = ""     # 墙边界（如 #479 / 断奶 / DEAD G 门（DEAD_DESIGN/DEAD_LEAK）/ ""）

    def __post_init__(self) -> None:
        assert_int(self.permille, self.threshold, _where="DimScore.permille/threshold")
        assert_no_float(self.permille, self.threshold,
                        _where="DimScore.permille/threshold")


@dataclass
class CapabilityReport:
    """全方位能力考核报告（jsonl 友好·bit-identical·sort_keys）。

    summary 强制 "X/8 examined·Z NE·W PASS·M MECHANISM_LIVE·V FAIL"（NE 不混入 PASS·MECHANISM_LIVE 不混入 PASS·防"机制接通被读成能力达成"·防"零测被读成考过"）。
    """

    run_id: str
    dimensions: dict[str, DimScore] = field(default_factory=dict)   # 8 维度
    strength_delta: dict[str, int] = field(default_factory=dict)    # CAUSES 边 strength delta（edge_key → delta·学到代理）
    strength_delta_total: int = 0                                   # sum delta（>0 = reward 通路活）
    anti_theater_passed: bool = True                                # 顶层旗标（任一锚点/反向回归失败→False·生产 gate 读此·P0-4）
    anti_theater_anchor: list = field(default_factory=list)         # 反 theater 锚点自检（list[dict]·anti_theater=True 填 / False 时 placeholder·P0-5）
    reverse_regression: list = field(default_factory=list)          # 反向回归结果（list[dict]·同上）
    # #723 G 归因：维度 × G门 交叉表（dict[dim][door] = {active,total,permille,dead_state,evidence_eps}）。
    # collect_episodes=True 时填（run_capability_exam 强制开）·to_json 序列化（cell dict key sort·bit-identical）。
    # **不含 episodes 列表**（决断2·Episode 嵌套 sort_keys 不确定·守 bit-identical）。
    g_attribution: dict = field(default_factory=dict)
    # Layer0 外部锚门归因（构造性检查≠构造性验证·防 cue 自产边 theater·分层墙 §八b）：
    # project_layer0 产（collect_episodes=True 时填）·to_json 序列化（key sort·bit-identical）。
    # additive 字段·不改 dimensions/g_attribution·守既有测零回归。
    layer0_attribution: dict = field(default_factory=dict)
    # #1041 构造③④：语言域统计层产出度量（判据④ lang_rate_permille + 判据⑤ pure_alias_edges_seeded）。
    # additive observability 字段（同 layer0_attribution 范式）·不改 dimensions/summary/g_attribution·守既有测零回归。
    # **signal 非 criterion**（thresholds defer·FOOTNOTE_LANG_MEASURES）·全统计层（非 can_ween/truth 决策层）。
    lang_measures: dict = field(default_factory=dict)
    # #1124 S5-S8 symbolic 统计层产出度量（transform rules verified + inverse relations verified）。
    # additive observability 字段（同 lang_measures 范式）·读 result.generate.xform_verified/inv_verified·不改 dimensions/
    # summary/g_attribution/lang_measures·守既有测零回归。**signal 非 criterion**（symbolic learning 可见·反 theater）。
    # CI 无 symbolic specs（无 ZERO_AI_LOCAL_DIR transform/inverse）→ 0/0 → bit-identical。
    math_measures: dict = field(default_factory=dict)
    summary: str = ""
    footnotes: list[str] = field(default_factory=list)
    # #727 fixture 限制守（决断5）：total episode<FIXTURE_SIZE_MIN → "STATISTICAL_NOISE"（fixture 不足·
    # 不计 PASS/FAIL 为定论）·否则 "OK"。to_json 序列化（str·bit-identical）。
    fixture_size_note: str = FIXTURE_NOTE_OK

    def to_json(self) -> dict[str, Any]:
        """jsonl 行（sort_keys 确定性·bit-identical）。

        DimScore 拍平为 dict（dim/status/permille/threshold/evidence/footnote）。
        anti_theater_anchor/reverse_regression 是 list[dict]（#726 片2 P0-5）—— top-level key sort·
        evidence 在 AnchorCheck.to_dict / ReverseRegressionCase.to_dict 时已 sort（list 序 json.dumps 不 sort·故预 sort）。
        g_attribution（#723）：dict[dim][door] = cell·dim 按 DIM_ORDER 序 / door 按 _G_DOORS 序 / cell key sort·
        evidence_eps 预 sort（index 升序）·**不含 episodes 列表**（决断2守 bit-identical）。
        """
        assert_int(self.strength_delta_total, _where="CapabilityReport.strength_delta_total")
        assert_no_float(self.strength_delta_total,
                        _where="CapabilityReport.strength_delta_total")
        # g_attribution 序列化（dim 按 DIM_ORDER / door 按 _G_DOORS / cell key sort·bit-identical）
        g_out: dict[str, dict[str, dict[str, Any]]] = {}
        for dim in DIM_ORDER:
            dim_cells = self.g_attribution.get(dim, {})
            g_out[dim] = {}
            for door in _G_DOORS:
                cell = dim_cells.get(door, {})
                g_out[dim][door] = {
                    "active": cell.get("active", 0),
                    "dead_state": cell.get("dead_state", G_NA),
                    "evidence_eps": sorted(cell.get("evidence_eps", [])),
                    "permille": cell.get("permille", -1),
                    "total": cell.get("total", 0),
                }
        return {
            "run_id": self.run_id,
            "summary": self.summary,
            "strength_delta_total": self.strength_delta_total,
            "anti_theater_passed": self.anti_theater_passed,
            "dimensions": {
                d.dim: {
                    "dim": d.dim,
                    "status": d.status,
                    "permille": d.permille,
                    "threshold": d.threshold,
                    "evidence": list(d.evidence),
                    "footnote": d.footnote,
                }
                for d in self.dimensions.values()
            },
            "strength_delta": dict(sorted(self.strength_delta.items())),
            "anti_theater_anchor": [dict(sorted(d.items())) for d in self.anti_theater_anchor],
            "reverse_regression": [dict(sorted(d.items())) for d in self.reverse_regression],
            "g_attribution": g_out,
            "layer0_attribution": dict(sorted(self.layer0_attribution.items())),
            "lang_measures": dict(sorted(self.lang_measures.items())),
            "math_measures": dict(sorted(self.math_measures.items())),
            "fixture_size_note": self.fixture_size_note,
            "footnotes": list(self.footnotes),
        }


# ---- strength snapshot（§D 反 theater 第一层·学到代理） ----

def snapshot_strengths(backend: Any, graph: ConceptGraph) -> dict[str, int]:
    """遍历所有概念节点的 EDGE_CAUSES 出边 → {edge_key: strength}（纯整数读）。

    edge_key 格式 "sf:lf->st:lt"（不含 edge_type·已按 EDGE_CAUSES 过滤）。
    遍历用 backend.select("concept_node") 全表扫（每节点 out_edges(EDGE_CAUSES)）·确定性序
    （concept_node 表按 PK 序·edge 行按 PK 序·同输入两跑一致）。

    **纯整数读**：strength 列 TYPE_INT·assert_int 守。无浮点。
    **bit-identical**：dict 构造序确定（concept_node/edge 表 PK 序·sort 后写 dict）。
    """
    out: dict[str, int] = {}
    nodes = backend.select("concept_node", where=None)
    for n in nodes:
        sid = n["space_id"]
        lid = n["local_id"]
        edges = graph.out_edges((sid, lid), edge_type=EDGE_CAUSES)
        for e in edges:
            sf = e["space_id_from"]
            lf = e["local_id_from"]
            st = e["space_id_to"]
            lt = e["local_id_to"]
            key = f"{sf}:{lf}->{st}:{lt}"
            strength = e["strength"]
            assert_int(strength, _where="snapshot_strengths.strength")
            out[key] = strength
    return dict(sorted(out.items()))


# ---- 维度 status 派生 helper（#726 片2·mutation 测 monkeypatch 点·bit-identical 纯重构） ----
# 提取自 project_dimensions 内联派生·逻辑零变·为 mutation 敏感性测（mut4/mut5）提供 monkeypatch 钩。
# ①④ 当前恒 NE（零测·#479 墙外）·⑥ 联立判据（pillars_all_ok AND delta>0）。
# 反 theater 牙：monkeypatch 这些 helper 模拟"判据失效" → harness 必须抓到 status 漂移（#726 片2 陷阱3）。

def _concept_status(graph_size: int) -> str:
    """①概念维度 status 派生（NE 守恒·件1/3/6 仅机制测）。

    当前恒 STATUS_NE（学到测全在 #479 墙外·关系节点 first-class 被消费未测）。
    graph_size 参数为未来 ① 落学到测时签名就位·当前忽略。
    **mutation 测 monkeypatch 此 helper**（mut4：返 STATUS_PASS）验 NE 守恒——
    若未来偷偷改 ① 读 graph_size 判 PASS/FAIL·NE 守恒测（注入 graph_size>0 诱因·#726 反向回归①）会抓到。
    """
    return STATUS_NE


def _long_text_status() -> str:
    """④长文本维度 status 派生（NE 守恒·多段连贯生成零测 + M5 分页仅机制测）。

    当前恒 STATUS_NE。**mutation 测 monkeypatch**（mut4）验 NE 守恒。
    """
    return STATUS_NE


def _long_code_status(generate_literal_ne: bool, mode_a_status: str) -> str:
    """⑤长代码维度 status 派生（拆双格·D5-enforcing·自盲修·2026-07-10·#889）。

    双格：
      - generate 字面路径（路径 X STRUCT_BIND reader 维度桥 defer·#478/#730）-> 恒 NE（零测）
      - Mode A task-driven（arith execute + #730 code unparse 重建·有阈值）-> PASS/FAIL
    **拆双格**：top-level status 取严--generate 字面 NE 子格不许被 Mode A PASS 吞（D5·撞墙保 NE）。
    即 generate_literal_ne -> ⑤ status=NE（即使 Mode A PASS）·Mode A 进 evidence/permille 不进 status。
    M14 反 theater 自盲：原 ⑤ status 取 Mode A·top-level 吞 generate 字面 NE 子格·误传"⑤ 能力达成"。
    **mutation 测 monkeypatch**（mut7：删 generate_literal_ne 前置）验 NE 守恒行敏感。
    """
    if generate_literal_ne:
        return STATUS_NE
    return mode_a_status


def _three_ring_status(pillars_all_ok: bool, strength_delta_total: int,
                       dead_states: tuple[str, ...] = ()) -> str:
    """⑥三环维度 status 派生（collapse 三柱全 ok AND strength_delta_total>0 -> MECHANISM_LIVE）。

    联立判据·两腿 AND（pillars_all_ok + delta>0）。**mutation 测 monkeypatch**（mut5：改 and 为 or）
    验判据行敏感--若判据被改·反向回归⑥（三柱 ok + delta=0·#726）会抓到。

    **MECHANISM_LIVE 非 PASS（2026-07-11·算数域病灶以小见大）**：strength_delta 是 reward 通路活代理
    （#479 墙·非真独立源验证学到）·三柱 ok + delta>0 = 机制接通非能力达成。旧标 PASS = "机制接通冒充能力达成"
    偷渡（同算数域"参与计算误成算出精确值"病灶）·改 MECHANISM_LIVE 第三态（≠ PASS/NE/FAIL·已考核非零测·
    机制活非缺陷·能力达成未确认）。delta=0 / pillars 缺 -> FAIL（机制未接通）。

    **STEP2 dead-G门 ALIVE 前置（D5-enforcing·自盲修·2026-07-10·#889）**：⑥三环 G3a/G3b DEAD_LEAK
    （漏洞性死·intent 三 bool 硬编码·formal_train.py:361-362）-> ⑥ 不许偷渡 MECHANISM_LIVE/PASS·保持 FAIL。
    D5 纪律：撞墙维度保 MECHANISM_LIVE/FAIL 禁偷渡 PASS。dead_states 任一 DEAD_LEAK -> FAIL（前置·覆盖联立判据）。
    DEAD_DESIGN（设计性死·如 G5 language 不在 _ARITH_DOMAINS）不计 fail·登记册标·不触发前置。
    现 ⑥ _DIM_G_DEAD G3a/G3b ALIVE（STEP2 #889·intent 已动态化）·前置防未来退化（DEAD_LEAK 注入测 mut6 验）。
    **mutation 测 monkeypatch**（mut6：删 dead-G门前置）验前置行敏感--若前置被删·mut6（三柱 ok + delta>0
    + dead G门·#889 STEP2）会抓到。
    """
    if any(s == G_DEAD_LEAK for s in dead_states):
        return STATUS_FAIL
    return STATUS_MECHANISM_LIVE if (pillars_all_ok and strength_delta_total > 0) else STATUS_FAIL


def _intent_status(strength_delta_total: int, reward_pos: int) -> str:
    """⑦初心维度 status 派生（strength_delta_total>0 AND reward_pos>0 -> MECHANISM_LIVE·#479 墙代理）。

    双腿判据（delta>0 AND reward_pos>0）。**MECHANISM_LIVE 非 PASS**：双腿全活 = reward 学习机制接通
    非能力达成·旧标 PASS = 偷渡·改 MECHANISM_LIVE（已考核非零测·机制活非缺陷·能力达成未确认）。

    **reward_pos 腿深修（2026-07-14·I-新·反 theater·落地旧 defer）**：旧单腿判据（delta>0）在 reward
    零触发时 delta 全为建边 base_strength 仍标 MECHANISM_LIVE = 建图机制活冒充 reward 学习活（偷渡）。
    strength_delta 来源有二--(a) observe/bootstrap 建新 CAUSES 边 base_strength=1（结构性·每新边 +1）·
    (b) reward 反传 add_strength（学习性·须 episode_loop->propagate_reward->record_episode_result·reward>0 才触发）。
    reward_pos=0 -> delta 全为 (a) 结构性建边 = 建图机制活非 reward 学习活 -> ⑦ FAIL（非 MECHANISM_LIVE）。
    reward_pos>0 -> reward 环路触发·delta 含 (b) reward 学习贡献 -> ⑦ MECHANISM_LIVE（reward 学习机制活）。
    capability_exam training_mode opt-in（生产 True / CI 默认 False）·CI 默认 reward_pos=0 -> ⑦ FAIL
    （诚实：CI 路径 reward 学习未触发·禁标 MECHANISM_LIVE 偷渡）·生产 training_mode=True -> reward_pos>0
    -> ⑦ MECHANISM_LIVE。evidence 注 reward_pos/episode_count 实测值让读者辨来源。

    **mutation 测 monkeypatch**（mut3：改判据为 delta>=0）验判据行敏感--若判据被改·反向回归⑦会抓到。
    ⑦与⑥的 delta 腿同源·⑥多一个 pillars 腿·⑥不要求 reward_pos（⑥=三环 collapse·⑦=初心 reward 学习·两维不同）。

    **mutation 测 monkeypatch**（mut3：改 >0 为 >=0）验判据行敏感--
    若判据被改·反向回归⑦（delta=0·#726）会抓到。⑦与⑥的 delta 腿同源·⑥多一个 pillars 腿。
    """

    return STATUS_MECHANISM_LIVE if (strength_delta_total > 0 and reward_pos > 0) else STATUS_FAIL


# ---- 反 theater 锚点 + 反向回归（#726 片2·§D 第二三层） ----

@dataclass
class AnchorCheck:
    """反 theater e2e 锚点自检（§D 第二层·#726 片2）。

    锚点走 corpus 层注入（陷阱1 正路）—— run_capability_exam 跑反例 corpus → formal_train 主路径
    自己产出 metrics → harness 投影 → 断言期望维度判 FAIL（非死写 PASS）。任一 passed=False → 维度升 ABORT。
    passed 是 bool·int_blocker:25-26 显式放行 bool·不 assert_int（P1-5）。
    **诚实边界**：反 theater 验判据可证伪·非验语义正确（stable≠correct·顶层 CapabilityReport.footnotes 总注·#479 墙）。
    """
    name: str               # "anchor_arith_no_heldout" 等
    injected: str           # corpus 层注入描述
    expected_status: str    # 期望该维度 status（通常 FAIL）
    actual_status: str      # harness 实际判
    passed: bool            # actual == expected
    evidence: list[str] = field(default_factory=list)   # 锚点 mismatch 证据（含 metrics 实测值）
    footnote: str = ""      # 诚实边界（如 stable≠correct / 真牙 defer）

    def __post_init__(self) -> None:
        # passed 是 bool（int_blocker 显式放行 bool:25-26）·不 assert_int（防误加·P1-5）
        # evidence 含 metrics f-string（str·非数值字段·不触发 float_guard）。
        assert isinstance(self.passed, bool), "AnchorCheck.passed 须 bool"

    def to_dict(self) -> dict[str, Any]:
        """序列化 dict（to_json 用·key 固定序·evidence 预 sort 保 bit-identical·#726 P0-5）。"""
        return {
            "name": self.name,
            "injected": self.injected,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "passed": self.passed,
            "evidence": sorted(self.evidence),
            "footnote": self.footnote,
        }


@dataclass
class ReverseRegressionCase:
    """反向回归（§D 第三层·#726 片2）—— project_dimensions 直调 + fake_result 注入。

    逐维度验判据可证伪（regressable）或 NE 守恒（ne_conservation）。分类不许混（陷阱2/3）。
    NE 守恒①④：注入非 NE 诱因（graph_size>0）→ 断言仍 NE（守未来偷偷塞判据·P0-3·非同义反复）。
    ⑥精确联立（三柱全 ok + delta=0 → FAIL）：测 strength_delta>0 那条腿（P0-1/纠正2·投影反例·非 e2e）。
    **诚实边界**：反 theater 验判据可证伪·非验语义正确（stable≠correct·顶层 CapabilityReport.footnotes 总注）。
    """
    dim: str                # DIM_STRUCTURE 等
    category: str           # "regressable" / "ne_conservation"（分类·不许混·陷阱2）
    bad_fixture: str        # bad fixture 描述（注入了什么）
    expected_status: str    # STATUS_FAIL / STATUS_NE
    actual_status: str
    passed: bool
    evidence: list[str] = field(default_factory=list)
    footnote: str = ""

    def __post_init__(self) -> None:
        assert isinstance(self.passed, bool), "ReverseRegressionCase.passed 须 bool"
        assert self.category in ("regressable", "ne_conservation"), (
            f"ReverseRegressionCase.category 须 regressable/ne_conservation·不许混·got {self.category}")

    def to_dict(self) -> dict[str, Any]:
        """序列化 dict（to_json 用·key 固定序·evidence 预 sort 保 bit-identical·#726 P0-5）。"""
        return {
            "dim": self.dim,
            "category": self.category,
            "bad_fixture": self.bad_fixture,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "passed": self.passed,
            "evidence": sorted(self.evidence),
            "footnote": self.footnote,
        }


# ---- 8 维度投影（核心·按 §三.1 阈值表·不许作弊） ----

def project_dimensions(result: FormalTrainResult,
                       strength_delta_total: int,
                       *, backend: Any | None = None,
                       g_dead_override: dict[str, dict[str, str]] | None = None
                       ) -> dict[str, DimScore]:
    """投影 FormalTrainResult → 8 维度 DimScore（按 §三.1 阈值·pre-registered·不许跑完调）。

    backend 可选（⑧记忆查 memory_item 行数）·None 时 ⑧退化 FAIL（无法验写入侧）。
    **判据可证伪**：注入低 metrics（如 strength_delta_total=0）→ ⑦初心判 FAIL（证非死写 PASS）。
    """
    dims: dict[str, DimScore] = {}
    # STEP2 (a) #889：g_dead 可注入（测试用·默认 _DIM_G_DEAD 静态表）·_three_ring_status dead-G门前置读此
    _g_dead = g_dead_override if g_dead_override is not None else _DIM_G_DEAD
    metrics = result.final_metrics
    graph_size = metrics.graph_size
    causes_cov = metrics.causes_coverage

    # ---- ①概念：graph_size>0 阈值=1·件1/3/6 仅机制测 → NE ----
    # 既有测：test_knife0/3 + test_stage16（机制测·无学到测·doc §三.1 标 NE）
    dims[DIM_CONCEPT] = DimScore(
        dim=DIM_CONCEPT,
        status=_concept_status(graph_size),
        permille=-1,
        threshold=-1,
        evidence=[f"graph_size={graph_size}·件1/3/6 仅机制测（test_knife0/3+test_stage16）·无学到测"],
        footnote="概念维度仅机制测·学到测全在 #479 墙外（关系节点 first-class 被消费未测）",
    )

    # ---- ②结构：causes_coverage ≥ 500 ----
    # 既有测：test_stage9（COVERAGE_THRESHOLD=500·dag_path.py:51）
    # **scale-calibration 诚实注（2026-07-15 #1120 验收核证）**：causes_coverage = _causes_coverage
    # （有 CAUSES 出边节点数/总节点 ×1000·formal_train:1702）。阈 500 permille(50%) 对 toy fixture
    # （graph_size~100·test 用 causes_coverage=600）校准·corpus-scale（82k 节点·多非 cause 词形/token/
    # alias/math·max≈30 permille·分母主导）**结构性不可达** → FAIL at scale ≠ 结构破裂（图 252k 边·
    # 47k COMPOSES/80k REFERS_TO/46k PRECEDES/7k IS_A 健康）·非学习缺陷·非课程可提（分母 capped）。
    s2 = STATUS_PASS if causes_cov >= THRESH_CAUSES_COV else STATUS_FAIL
    dims[DIM_STRUCTURE] = DimScore(
        dim=DIM_STRUCTURE,
        status=s2,
        permille=causes_cov,
        threshold=THRESH_CAUSES_COV,
        evidence=[f"causes_coverage={causes_cov}{'≥' if s2 == STATUS_PASS else '<'}500·test_stage9 断言回填·dag_path.py:51 COVERAGE_THRESHOLD"],
        footnote="多前驱>2 深汇聚 + 汇聚后 LCA 折叠组合未测（doc §三.1 缺口）·⚠ 阈 500 permille(50%) fixture 校准·corpus-scale 分母主导不可达（max≈30 permille）·FAIL≠结构破裂",
    )

    # ---- ③计算：generalization.rate_permille ≥ 500（若 None → NE） ----
    # 既有测：test_stage9_arith_observe（rate_permille·vm_proof 真 oracle）
    if result.generalization is not None:
        rp = result.generalization.rate_permille
        s3 = STATUS_PASS if rp >= THRESH_RATE_PERMILLE else STATUS_FAIL
        dims[DIM_COMPUTE] = DimScore(
            dim=DIM_COMPUTE,
            status=s3,
            permille=rp,
            threshold=THRESH_RATE_PERMILLE,
            evidence=[f"generalization.rate_permille={rp}{'≥' if s3 == STATUS_PASS else '<'}500·test_stage9_arith_observe 断言回填·vm_proof 真 oracle"],
            footnote="最厚维度·基本无缺口·Mode B cross-verify 已 live（test_mode_b_cross_verify）",
        )
    else:
        dims[DIM_COMPUTE] = DimScore(
            dim=DIM_COMPUTE,
            status=STATUS_NE,
            permille=-1,
            threshold=-1,
            evidence=["generalization=None（非算术域 corpus·无 vm_proof oracle）"],
            footnote="corpus 非算术域 → 计算维度零测·NE ≠ PASS",
        )

    # ---- ④长文本：NE（零测） ----
    # doc §三.1：多段连贯生成零测 + M5 分页仅机制测
    dims[DIM_LONG_TEXT] = DimScore(
        dim=DIM_LONG_TEXT,
        status=_long_text_status(),
        permille=-1,
        threshold=-1,
        evidence=["多段多章连贯生成零测·M5 分页仅机制测（不测真分页生成）"],
        footnote="零测 ≠ PASS·顶部计 Z NE",
    )

# ---- ⑤长代码：拆双格 generate 字面 NE + Mode A task-driven（D5-enforcing·#889 STEP2 自盲修） ----
    # doc §三.1：代码 generate 字面路径零测（generate.py STRUCT_BIND defer·#478/#730·**路径 X 维度桥阻断**）= NE 子格。
    # result.generate（若非 None）来自 _run_task_driven_generate（formal_train.py·**Mode A task-driven L8 episode**）·
    # **两模态**：arith（execute skeleton(新 args)==expected）+ code（#730 路径 W·unparse COMPOSES->源码串 normalize==code_source·
    # 构造性重建）·**非 Mode B cross-verify**（POST reward 不进 result.generate）。#726 片2 P0-2 纠偏：原 evidence 误标
    # "Mode B ... test_mode_b_cross_verify"撒谎·改正。#730 子决断 1-bis：Mode A PASS 仅覆盖 task-driven（arith execute +
    # code unparse 重建）·**非源码 generate**。
    # **STEP2 (b) #889 拆双格**：原 status 取 Mode A·top-level 吞 generate 字面 NE 子格（M14 反 theater 自盲·误传"⑤ 达成"）。
    # 改：status 取严--generate 字面 NE 子格 -> ⑤ status=NE（即使 Mode A PASS·D5 撞墙保 NE 禁偷渡）。
    # Mode A rate 进 evidence（permille 取严 NE 时 -1·Mode A rate 在 evidence 字符串·守 DimScore NE=-1 约定）。
    gen_evidence = ["代码 generate 字面路径零测（generate.py STRUCT_BIND reader·路径 X 维度桥阻断 defer·#478/#730）·NE ≠ PASS"]
    mode_a_status = STATUS_NE
    mode_a_permille = -1
    mode_a_threshold = -1
    if result.generate is not None:
        # Mode A task-driven 泛化率（arith execute 探针 + #730 code unparse 重建·两模态合计 verified/total）
        grp = result.generate.rate_permille
        mode_a_status = STATUS_PASS if grp >= THRESH_RATE_PERMILLE else STATUS_FAIL
        mode_a_permille = grp
        mode_a_threshold = THRESH_RATE_PERMILLE
        gen_evidence.append(
            f"Mode A task-driven generate.rate_permille={grp}{'≥' if mode_a_status == STATUS_PASS else '<'}500·"
            f"_run_task_driven_generate·verified={result.generate.verified}/{result.generate.total_tasks}"
            f"（arith execute + #730 code unparse 两模态合计）"
        )
        gen_evidence.append("**非 Mode B cross-verify**（POST reward 不进 result.generate·test_mode_b_cross_verify 独立线）")
        gen_evidence.append("**Mode A PASS ≠ 源码 generate**（code unparse 是构造性重建·skeleton 派生自 code_source·"
                            "非真生成·真源码 generate 须路径 X 跨模态维度桥 defer）")
    # 拆双格：generate 字面恒 NE（路径 X defer·#478/#730·未变）·status 取严 NE（D5·禁 Mode A PASS 偷渡）
    _gen_literal_ne = True   # generate 字面路径恒 NE（路径 X defer·未来维度桥打通改 False）
    gen_status = _long_code_status(_gen_literal_ne, mode_a_status)
    # permille/threshold 取严：status NE -> -1（Mode A rate 在 evidence·不进 permille·守 DimScore NE=-1 约定）
    gen_permille = -1 if gen_status == STATUS_NE else mode_a_permille
    gen_threshold = -1 if gen_status == STATUS_NE else mode_a_threshold
    gen_footnote = ("拆双格·generate 字面 NE 子格 + Mode A task-driven 子格·status 取严 NE（D5·#889 STEP2·"
                    "generate 字面零测路径 X defer·Mode A PASS 不偷渡⑤ status）·Mode A rate 进 evidence")
    dims[DIM_LONG_CODE] = DimScore(
        dim=DIM_LONG_CODE,
        status=gen_status,
        permille=gen_permille,
        threshold=gen_threshold,
        evidence=gen_evidence,
        footnote=gen_footnote,
    )

    # ---- ⑥三环：collapse 三柱全 ok + strength_delta_total>0 → MECHANISM_LIVE（机制接通非能力达成·#479 墙）----
    # language 域 G3a/G3b ALIVE（STEP2 #889·intent 已动态化·classify_intent 真填）·旧"永死"stale 已纠
    # 既有测：test_stage5（collapse 三柱）+ test_experiments（formal_train e2e）
    cs = result.collapse_summary or {}
    p1_ok = bool(cs.get("pillar1_ok", 0))
    p2_ok = bool(cs.get("pillar2_ok", 0))
    p3_ok = bool(cs.get("pillar3_ok", 0))
    pillars_all_ok = p1_ok and p2_ok and p3_ok
    _dead_states_6 = tuple(_g_dead.get(DIM_THREE_RING, {}).values())
    s6 = _three_ring_status(pillars_all_ok, strength_delta_total, _dead_states_6)
    # permille = strength_delta_total（>0 即 MECHANISM_LIVE·代理 reward 通路活·非能力达成）·threshold=1
    dims[DIM_THREE_RING] = DimScore(
        dim=DIM_THREE_RING,
        status=s6,
        permille=strength_delta_total,
        threshold=THRESH_STRENGTH_DELTA,
        evidence=[
            f"collapse 三柱 pillar1_ok={p1_ok}/pillar2_ok={p2_ok}/pillar3_ok={p3_ok}（全 ok={pillars_all_ok}）",
            f"strength_delta_total={strength_delta_total}{'>0' if strength_delta_total > 0 else '≤0'}·test_stage5+test_experiments 断言回填",
        ],
        footnote="language 域 G3a/G3b ALIVE（STEP2 #889·intent 已动态化）·⑥机制接通→MECHANISM_LIVE 非 PASS（#479 墙·strength_delta 是 reward 通路活代理非真学到）",
    )

    # ---- ⑦初心：strength_delta_total>0 AND reward_pos>0 -> MECHANISM_LIVE（#479 墙 footnote） ----
    # 既有测：test_formal_train_strength_changes_e2e（H4 闭环·strength 变）。
    # **reward_pos 腿深修（I-新·2026-07-14·落地 2026-07-12 反 theater 核证旧 defer）**：strength_delta 含
    # (a) observe 建新 CAUSES 边 base_strength=1（结构性）+ (b) reward 反传 add_strength（学习性·须 episode_loop）。
    # 旧单腿（delta>0）reward 零触发时仍标 MECHANISM_LIVE = 建图机制活冒充 reward 学习活（偷渡）->
    # 加 reward_pos 腿（delta>0 AND reward_pos>0 才 MECHANISM_LIVE）·reward_pos=0 -> FAIL（CI 默认路径诚实）。
    # training_mode opt-in（生产 True / CI 默认 False）·reward_pos 从 result.episodes 算（不在 final_metrics）。

    _eps = _legacy_result_episodes(result)
    _ec = len(_eps)
    _rp = sum(1 for e in _eps if getattr(e, "reward", 0) > 0)
    # reward_pos 腿深修（I-新·2026-07-14 落地旧 defer）：⑦ MECHANISM_LIVE 须 delta>0 AND reward_pos>0。
    # reward_pos=0（CI 默认 training_mode False->reward 环路零触发）-> ⑦ FAIL（建图机制活非 reward 学习活·禁偷渡）。
    # reward_pos>0（生产 training_mode True->reward 环路触发）-> ⑦ MECHANISM_LIVE（reward 学习机制活）。
    s7 = _intent_status(strength_delta_total, _rp)
    _src = ("reward 环路零触发->delta 全为结构性建边·建图机制活非 reward 学习->⑦ FAIL（I-新 reward_pos 腿）"
            if _rp == 0 else f"reward 环路已触发（reward_pos={_rp}>0）->delta 含 reward 学习贡献->⑦ MECHANISM_LIVE")
    dims[DIM_INTENT] = DimScore(
        dim=DIM_INTENT,
        status=s7,
        permille=strength_delta_total,
        threshold=THRESH_STRENGTH_DELTA,
        evidence=[
            f"strength_delta_total={strength_delta_total}{'>0' if strength_delta_total > 0 else '≤0'}·CAUSES 边 strength 训练前后 delta 之和",
            f"⚠ delta 来源不独立：observe 建新 CAUSES 边 base_strength=1（结构性）+ reward 反传 add_strength（学习性·须 episode_loop→record_episode_result）",
            f"capability_exam reward_pos={_rp}/episode_count={_ec}（training_mode opt-in·CI 默认 False=零触发 / 生产 True=触发）·{_src}",
        ],
        footnote=FOOTNOTE_WALL_479,
    )

    # ---- ⑧记忆：写入（memory_item 行数>0）+ 消费者触发（G5-C consolidate flip 行数）----
    # 既有测：test_stage11_memory_space（11d 写活 + #732 G5-C 闸落 code·消费者部分活）
    # 审 2 P0-3：消费者 dead 却触发 → abort·消费者=0 → fail（非缺口美化）
    # #732 G5-C 闸落 code：consumer_triggers 实查 CONSOLIDATED 行数（gate ON 时 consolidate flip >0）
    memory_rows = 0
    consumer_triggers = 0
    if backend is not None:
        try:
            memory_rows = len(backend.select("memory_item", where=None))
        except Exception:
            memory_rows = 0   # 表未注册 / bare fixture → 退化 0（写入侧亦无）
        try:
            from pure_integer_ai.storage.spaces.memory_space import STATUS_CONSOLIDATED
            # 审1/审2 P2-1：当前不带 space_id 过滤·生产 consolidate caller 只扫 memory_read.space_id
            # （promote_memory_consolidate where space_id=memory_read.space_id）·CONSOLIDATED 行只在 memory_read
            # space·当前无混计风险。#728 落地若 memory_interact 加 consolidate caller·须带 space_id 过滤防混计。
            consumer_triggers = len(backend.select(
                "memory_item", where={"status": STATUS_CONSOLIDATED}))
        except Exception:
            consumer_triggers = 0   # 表未注册 / bare fixture → 退化 0
    # ⑧ = ⑧a AND ⑧b（#732 引入设计选择·非权威设计定义·记忆真消费须 consolidate G5-C + 检索 J4/tri_space
    # 双通道活·单通道=半拉子）。⑧a 训练侧（G5-C consolidate·consumer_triggers>0 = gate ON 真消费）·
    # ⑧b 消费者侧（#728 generate 读 memory_space + J4 #733 + tri_space 中环 4-5）仍 defer。
    # G5-C code live（gate default OFF·capability_exam 默认跑不触发 consolidate·consumer_triggers=0）·
    # ⑧反向回归 gate ON 可触发 consolidate（consumer_triggers>0）·⑧整体仍 FAIL（⑧b defer·非半拉子美化）。
    s8 = STATUS_FAIL  # ⑧b 仍 defer → ⑧ 整体仍 FAIL（⑧a 活但 ⑧b 断·⑧=⑧a AND ⑧b）
    # permille：写入行数（>0 证写入侧活·threshold=1）·但 status 仍 FAIL（⑧b 消费者断）
    dims[DIM_MEMORY] = DimScore(
        dim=DIM_MEMORY,
        status=s8,
        permille=memory_rows,
        threshold=THRESH_STRENGTH_DELTA,   # 写入侧 threshold=1·status 由 ⑧b 消费者决定
        evidence=[
            f"memory_item 写入行数={memory_rows}{'>0' if memory_rows > 0 else '≤0'}·test_stage11_memory_space 11d 写活",
            f"⑧a G5-C consumer_triggers={consumer_triggers}（CONSOLIDATED 行数·gate ON 时 >0·OFF 时 =0·#732 落 code）",
            f"⑧b 消费者侧（#728 generate 读 memory_space + J4 #733 + tri_space 中环 4-5）仍 defer",
        ],
        footnote="G5-C code live（#732·gate default OFF·capability_exam 默认跑 consumer_triggers=0）·⑧反向回归 gate ON 可触发·⑧整体仍 FAIL（⑧b defer·⑧=⑧a AND ⑧b·#732 引入设计选择·非半拉子美化）",
    )

    return dims


# ---- #723 G 归因：维度 × G门 交叉表（判据偏真闭合最后一环） ----

def _classify_episode(ep: Any) -> tuple[str, str] | None:
    """episode → (dim, path) 分类（G 归因表用·语义锚非任意交叉）。

    三路 episode（formal_train 产出）：
      - language judge（input.modality=LANGUAGE + pr_vector 非空）→ (⑥三环, "judge")·5 门全走
      - ARITH/CODE verify（input.modality=ARITH/CODE + pr_vector 空）→ (③/⑤, "verify")·只走 G5
      - Mode A task-driven（input=None）→ (⑤长代码, "mode_a")·不走任何 G 门（G5=False 硬编码·外真验）
    其余 → None（不贡任何格分母·如 language 但 pr_vector 空 = observe-only 阶段无 dag_path）。

    **诚实边界**：分类按 episode 真实字段（modality + pr_vector + input）·非死写维度。
    pr_vector 空判 verify 路径（_run_verify_round:518 / _run_task_driven_generate:1965 都置空）。
    """
    if not isinstance(ep, Episode):
        return None
    inp = ep.input
    if inp is None:
        return (DIM_LONG_CODE, "mode_a")   # Mode A task-driven（input=None·formal_train.py:1961）
    mod = getattr(inp, "modality", None)
    has_pr = bool(getattr(ep, "pr_vector", None))
    if mod == MODALITY_LANGUAGE and has_pr:
        return (DIM_THREE_RING, "judge")
    if mod == MODALITY_ARITH and not has_pr:
        return (DIM_COMPUTE, "verify")
    if mod == MODALITY_CODE and not has_pr:
        return (DIM_LONG_CODE, "verify")
    return None


def _legacy_result_episodes(result: FormalTrainResult) -> list[Episode]:
    """返回旧能力考核可消费的标量 reward episode，隔离 typed 协议。"""
    return [
        item for item in (getattr(result, "episodes", None) or [])
        if isinstance(item, Episode)
    ]


def _door_vetoed(ep: Any, path: str, door: str) -> bool:
    """该 G 门在该 episode 是否真 veto（非仅承重）·#723 P0 修·对抗审1 抓。

    **judge path**：judge_G{door}_active=True 即该门 veto（judge.py:215/221/236/243/251 veto 时
    return 0 并设 g.G{X}=True·episode.vetoed=(reward==0)=True·故 judge_G{X}_active ⟺ 该门 veto）。
    **verify path**：judge_G5_active=**承重门**（formal_train.py:506/519·Mode B cross-verify 跑即 True·
    pass=reward=1 / fail=reward=0 都设 True）·非 veto。真 G5 veto = 承重 AND episode vetoed（reward=0·
    cross_verify disagree）。非 Mode B（g5_active=False·reward=0 占位）不算 G5 veto（G5 未承重）。
    """
    bearing = bool(getattr(ep, _DOOR_FIELD[door], False))
    if path == "verify":
        # verify path G5：承重 AND episode vetoed（防 bearing=True 但 pass 的 episode 误计 veto）
        return bearing and bool(getattr(ep, "vetoed", False))
    # judge path：judge_G{door}_active=True 即该门 veto（per-door 归因·非整体 vetoed）
    return bearing


def project_g_attribution(result: FormalTrainResult) -> dict[str, dict[str, dict[str, Any]]]:
    """投影 result.episodes → 维度 × G门 交叉表（§D 判据偏真闭合最后一环·#723）。

    每格 cell = {active, total, permille, dead_state, evidence_eps}：
      - total   : 该维度该路径 episode 数（分母·仅 _PATH_DOORS[path] 内的门计）
      - active  : 该门真 veto 次数（judge path: judge_G{X}_active / verify path: 承重 AND vetoed·#723 P0 修）
      - permille: active*1000 // max(total,1)（N/A 格 = -1·与 DimScore NE 范式一致）
      - dead_state: ALIVE / DEAD_DESIGN / DEAD_LEAK / N/A（静态 _DIM_G_DEAD·基于 code 真相）
      - evidence_eps: active 的 episode 在 result.episodes 中的 index 列表（溯源·反 theater 层2）

    **反 theater**（决断4·防美化核心）：
      - dead 分类静态（非代码自报）·_DIM_G_DEAD 基于 code 结构（formal_train.py:362 / judge.py:247,229,241）。
      - dead 门 active 须=0（生产永不触发）·反 theater 测试注入"应触发"fixture 验仍=0（证非"恰好没触发"）。
      - active 格必带 evidence_eps（溯源可抽样核）。
      - 坏 fixture（G 门 veto 多）→ 表必异于正常 fixture（非 theater·层1）。

    **诚实边界**：table 仅 ③⑤⑥ 有真分母（judge/verify 路径）·①②④⑦⑧ 全 N/A（无 episode judge 路径）。
    Mode A task-driven episode（input=None）归 ⑤ 但 mode_a 路径不走门·不计 ⑤ G5 分母（G5=False 硬编码·外真验非 G5 门）。
    stable≠correct（考核验统计完备非语义正确·顶层 footnotes 总注）。
    """
    # init table: 8 dims × 5 doors·全 N/A / 0
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in DIM_ORDER:
        table[dim] = {}
        dead_map = _DIM_G_DEAD.get(dim, {})
        for door in _G_DOORS:
            dead = dead_map.get(door, G_NA)
            table[dim][door] = {
                "active": 0,
                "total": 0,
                "permille": -1 if dead == G_NA else 0,
                "dead_state": dead,
                "evidence_eps": [],
            }

    # classify + count（仅 result.episodes 真活时·collect_episodes=True 填）
    for idx, ep in enumerate(result.episodes):
        cls = _classify_episode(ep)
        if cls is None:
            continue
        dim, path = cls
        for door in _PATH_DOORS[path]:
            cell = table[dim][door]
            cell["total"] += 1
            if _door_vetoed(ep, path, door):   # #723 P0 修·verify path 须承重 AND vetoed
                cell["active"] += 1
                cell["evidence_eps"].append(idx)

    # permille + sort evidence_eps + 纯整数守（index 升序·bit-identical·对抗审2 P2-2）
    for dim in table:
        for door in table[dim]:
            cell = table[dim][door]
            if cell["dead_state"] == G_NA:
                cell["permille"] = -1
                continue
            total = cell["total"]
            cell["permille"] = (cell["active"] * 1000) // max(total, 1) if total > 0 else 0
            cell["evidence_eps"] = sorted(cell["evidence_eps"])
            assert_int(cell["active"], cell["total"], cell["permille"],
                        _where=f"project_g_attribution cell {dim}/{door}")
            assert_no_float(cell["active"], cell["total"], cell["permille"],
                            _where=f"project_g_attribution cell {dim}/{door}")

    return table


def project_layer0(result: FormalTrainResult) -> dict[str, int]:
    """Layer0 外部锚门归因（构造性检查≠构造性验证·防 cue 自产边 theater·分层墙 §八b）。

    读 result.episodes → count_layer0 → 返 Layer0 分类计数 dict（纯整数·key 固定序·bit-identical）。
    与 project_g_attribution 平行（G 门归因）·**职责分离**：G 归因查"门 veto"·Layer0 查"验证来源溯源"。
    **不碰 _classify_episode**（time_seq → None 在 G 表正确·language G5=DEAD·时序可见性归本 summary）。

    反 theater（停止决策守门·§八b "全自产不准停"）：
      - self_produced_check_passed>0（时序检查通过·非验证）必不入 external_verified·防 cue 自产边被计构造性验证。
      - anchor_violated>0（全自产 episode 存在）= 有 episode 不准驱动停止决策。

    诚实边界：本投影只标+汇总·不提供 R6（R6 加固属刀G ConceptNet Causes loader / 时序升验证 defer）。
    stable≠correct（外部锚门满足≠语义正确·#479 墙）。
    """
    return count_layer0(_legacy_result_episodes(result))


def project_lang_measures(result: FormalTrainResult) -> dict[str, int]:
    """语言域统计层产出度量（#1041 构造③④·判据④泛化 + 判据⑤跨语言汇聚·observability 信号非 criterion）。

    **③判据④（lang generalization rate）** = result.lang_generalization.lang_rate_permille（S7 相1 recognize
      结构对齐口径·语言不可 vm_proof·渐近判据·formal_train:1962）。**前 observability-only**（formal_train:1967
      自标"非闭环消费者"）·本函数是 deferred **observability 报告消费者**（单向 tap：值→报告→止·无 decision/
      threshold/feedback·非闭环·signal 非 criterion·thresholds defer 真语料）。
    **④判据⑤（cross-lang convergence）** = result.alias_edges_seeded（P0b 桥 boot 种双向 PURE_ALIAS 边数·
      formal_train boot 捕获）。**edges_seeded 度量桥执行**（输入侧 precondition·无 edges→无汇聚可能）·**非**测
      activate_candidates 收敛对称性——后者由 P0b 桥结构保证（双向 seed a→b+b→a + activate_candidates
      PURE_ALIAS-gated 自包含 fix·graph_view.py:198）+ test_alias_bridge AB3 回归测·此处不重复测（冗余）。
      **生成侧真收敛**（generate 按 target_lang 产 apple/苹果）= **机制层已证**·判据⑤机制层满足（test_dispatch_token_chain
      TC5：apple↔苹果 PURE_ALIAS→generate_output 端到端 ZH 产苹果/EN 产 apple 真字·2026-07-15 #1120 验收核证·
      非同义反复·旧标"未证"stale 过保守已纠）。defer 仅规模锻炼（76k corpus + EN-target 训练 episode·非机制 gap·课程轴B）。

    返 dict[str,int]（key sort·纯整数·bit-identical）。lang_generalization None→③全 -1/0（理论不发生·
      formal_train:2216 总赋值）。getattr alias_edges_seeded（默认 0·纵深防御旧 result 兼容）。全统计层
      （判据④⑤·FOOTNOTE_LANG_MEASURES）。
    """
    out: dict[str, int] = {}
    lg = result.lang_generalization
    if lg is not None:
        out["lang_rate_permille"] = lg.lang_rate_permille
        out["lang_total_held_out"] = lg.total_held_out
        out["lang_recognized"] = lg.recognized
    else:
        out["lang_rate_permille"] = -1
        out["lang_total_held_out"] = 0
        out["lang_recognized"] = 0
    # ④判据⑤：跨语言 PURE_ALIAS 桥种边数（P0b boot·getattr 守旧 result 兼容）
    out["pure_alias_edges_seeded"] = getattr(result, "alias_edges_seeded", 0)
    for _v in out.values():
        assert_int(_v, _where="project_lang_measures")
    return dict(sorted(out.items()))


def project_symbolic_measures(result: FormalTrainResult) -> dict[str, int]:
    """S5-S8 符号数学统计层产出度量（#1124·additive observability·镜像 project_lang_measures 范式）。

    读 result.generate.xform_verified / inv_verified（formal_train _run_task_driven_generate 的 symbolic 子计数·
    S5-S7 transform 规则 cross-verify verified + S8 inverse 关系 B∘A 还原 verified）+ xform_total / inv_total 分母
    → xform_rate_permille / inv_rate_permille（spec §四 4.2 symbolic_cross_verify_rate·镜像 lang_rate_permille·最接近泛化信号）。

    **signal 非 criterion**（全统计层·非 can_ween/truth 决策层·同 lang_measures）：symbolic 路径已活（SYMBOLIC_TRANSFORM/
    RELATION_MODE 生产 try/finally 翻 ON·transform_rules+inverse_relations 语料 boot-inject）但**学习无度量=不可见=theater 风险**
    → 本函数是反 theater observability 消费者（symbolic verified 可见·非 invisible）。thresholds defer（symbolic stable≠correct·
    cross-verify 统计非 truth·#479 守）。**count = verified 数·rate = verified/total（M-1 加分母·非偷渡·stable≠correct）**。

    CI 无 symbolic specs（无 ZERO_AI_LOCAL_DIR transform/inverse 文件）→ generate=None 或全 0 → 返 0 分母 0 rate
    （bit-identical·additive 不改 dimensions/summary）。result.generate None→getattr 守（防御旧 result）。
    """
    gen = getattr(result, "generate", None)
    xv = getattr(gen, "xform_verified", 0) if gen is not None else 0
    iv = getattr(gen, "inv_verified", 0) if gen is not None else 0
    xt = getattr(gen, "xform_total", 0) if gen is not None else 0
    it = getattr(gen, "inv_total", 0) if gen is not None else 0
    out = {
        "xform_verified": xv, "xform_total": xt,
        "xform_rate_permille": (xv * 1000) // max(xt, 1),   # verified/total ×1000（total=0→0·镜像 lang_rate）
        "inv_verified": iv, "inv_total": it,
        "inv_rate_permille": (iv * 1000) // max(it, 1),
    }
    for _v in out.values():
        assert_int(_v, _where="project_symbolic_measures")
    return dict(sorted(out.items()))


def _g_dead_summary(table: dict[str, dict[str, dict[str, Any]]]) -> tuple[int, int]:
    """数 dead 门（cross-table 全格扫）→ (dead_design_count, dead_leak_count)。

    summary 强制格式 "K DEAD_DESIGN·J DEAD_LEAK"（决断4·dead 不混 PASS/FAIL 但必出现 summary·防美化）。
    只数 dead_state 为 DEAD_DESIGN/DEAD_LEAK 的格（N/A/ALIVE 不计）。
    """
    dd = dl = 0
    for dim in table:
        for door in table[dim]:
            state = table[dim][door]["dead_state"]
            if state == G_DEAD_DESIGN:
                dd += 1
            elif state == G_DEAD_LEAK:
                dl += 1
    return dd, dl


def _g_dead_footnotes(table: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """dead 门一一列出 footnote（决断4·强制顶层 footnotes 复述·防美化）。

    DEAD_LEAK 标源行号（formal_train.py:362 intent 三 bool 硬编码）·计隐性 fail。
    DEAD_DESIGN 标设计性排除（judge.py:247 language 不在 _ARITH_DOMAINS）·不计 fail。
    """
    notes: list[str] = []
    for dim in DIM_ORDER:
        for door in _G_DOORS:
            cell = table.get(dim, {}).get(door, {})
            state = cell.get("dead_state", G_NA)
            if state == G_DEAD_LEAK:
                notes.append(
                    f"G 门 {door}（{dim}）DEAD_LEAK：消费者存在但 intent 三 bool 硬编码 False"
                    f"（formal_train.py:362）→ judge.py 该门永不触发·漏洞性 dead·计隐性 fail")
            elif state == G_DEAD_DESIGN:
                notes.append(
                    f"G 门 {door}（{dim}）DEAD_DESIGN：language 域不在 _ARITH_DOMAINS"
                    f"（judge.py:247）→ 设计性排除·不计 fail·消费者（language self_proof）按架构不应有")
    return notes


# ---- harness 主入口 ----

def _run_capability_exam_impl(config: FormalTrainConfig,
                              corpus: list, *,
                              backend: Any,
                              teacher: Any = None,
                              runner: Any = None,
                              anti_theater: bool = False,
                              backend_factory: Callable[[], Any] | None = None,
                              training_mode: bool = False,
                              flat_floors: bool = False) -> CapabilityReport:
    """跑一轮 formal_train + 训练前后 strength snapshot + 投影 8 维度 → CapabilityReport。

    步骤：
      1. 训练前：ConceptGraph(backend) → snapshot_strengths → pre
      2. result = formal_train(config, corpus, backend, teacher, runner)
      3. 训练后：重建 ConceptGraph(backend)（observe 增边后 cache 新）→ snapshot_strengths → post
      4. strength_delta = {k: post[k] - pre.get(k, 0) for k in post}
         （新边 delta=post_strength·既有边 delta=差·MUTABLE_MONOTONE 只增不降）
      5. strength_delta_total = sum(delta.values())
      6. dimensions = project_dimensions(result, strength_delta_total, backend=backend)
      7. anti_theater（#726 片2·anti_theater=True 时跑锚点+反向回归·失败维度升 ABORT·P0-4）
      8. summary = "X/8 examined·Z NE·W PASS·V FAIL"（+ ABORT 计数·anti_theater=True 时）
      9. footnotes = [#479, 断奶 can_ween决策层truth/统计层判据另建, stable≠correct, 物种差异]
     10. anti_theater_anchor/reverse_regression = list[dict]（anti_theater=True 真跑 / False placeholder）

    **anti_theater 默认 False**：保片1 零回归（既有 8 测不期待锚点跑·placeholder list[dict]）。
    生产 gate 调用时传 anti_theater=True + backend_factory（每锚点独立 backend）。

    返 CapabilityReport（to_json sort_keys bit-identical）。
    """
    # 1. 训练前 snapshot（ConceptGraph 单例·纯读不建边）
    # bootstrap 后端表（formal_train 内部也调·此处先调保 pre-snapshot 表已注册·幂等）。
    # 不重造 formal_train：bootstrap 是 read lifecycle 设置（register_table 幂等）·非训练逻辑。
    from pure_integer_ai.storage import bootstrap as _bootstrap
    _bootstrap(backend)
    pre_graph = ConceptGraph(backend)
    pre = snapshot_strengths(backend, pre_graph)

    # 2. 跑 formal_train（复用·不重造）·#723 强制 collect_episodes=True（harness 须读 G_meta 5 字段建归因表）
    # dataclasses.replace 非_mutating caller config·仅翻 collect_episodes（其余 run_dir/run_id 等不变）。
    from dataclasses import replace as _dc_replace
    _exam_config = _dc_replace(config, collect_episodes=True) if not config.collect_episodes else config
    # training_mode opt-in（默认 False 守既有测 bit-identical·镜像 run_weaning_arith:151-171）
    # 生产 caller 传 True：翻 TRAINING_MODE → reward 环路触发（gap1 修复·eff_stage=STAGE3·5 verify 通道可达）
    # flat_floors 绕 stage 门控（cap_exam 职责=8 维考核非门控标定·门控 defer D5·既有 e2e 同范式）
    gate_token = gates.push_gate_overrides({
        "TRAINING_MODE": True if training_mode else gates.TRAINING_MODE,
    })
    floor_token = stages.push_stage_floor_overrides({
        "FLOOR_GRAPH_SIZE_S1": 0,
        "FLOOR_CAUSES_COV_S2": 0,
        "FLOOR_CONDUCTION_S3": 0,
        "FLOOR_PROMOTE_S4": 0,
    } if flat_floors else {})
    try:
        result: FormalTrainResult = formal_train(
            _exam_config, corpus, backend=backend, teacher=teacher, runner=runner)
    finally:
        stages.reset_stage_floor_overrides(floor_token)
        gates.reset_gate_overrides(gate_token)

    # 3. 训练后 snapshot（重建 ConceptGraph·observe 增边后图新鲜）
    post_graph = ConceptGraph(backend)
    post = snapshot_strengths(backend, post_graph)

    # 4. strength delta（MUTABLE_MONOTONE 只增不降·delta≥0 守·新边 delta=post_strength）
    strength_delta: dict[str, int] = {}
    for k, v in post.items():
        delta = v - pre.get(k, 0)
        assert_int(delta, _where="run_capability_exam.delta")
        # MUTABLE_MONOTONE 守：delta 应 ≥ 0（reward 反传 add_strength 只 += 正·base_strength 同）
        # 若 delta<0 → 数据 corrupt·不静默吞·harness 抛（反 theater·证 snapshot 一致性）
        if delta < 0:
            raise RuntimeError(
                f"strength delta<0（edge={k} delta={delta}）·违 MUTABLE_MONOTONE·数据 corrupt")
        strength_delta[k] = delta
    strength_delta = dict(sorted(strength_delta.items()))

    # 5. total
    strength_delta_total = sum(strength_delta.values())
    assert_int(strength_delta_total, _where="run_capability_exam.strength_delta_total")
    assert_no_float(strength_delta_total,
                    _where="run_capability_exam.strength_delta_total")

    # 6. 投影 8 维度
    dimensions = project_dimensions(result, strength_delta_total, backend=backend)

    # 6b. #723 G 归因交叉表（维度 × G门·collect_episodes=True 已强制·result.episodes 真活）
    g_attribution = project_g_attribution(result)
    dead_design_count, dead_leak_count = _g_dead_summary(g_attribution)
    # 6c. Layer0 外部锚门归因（构造性检查≠构造性验证·防 cue 自产边 theater·分层墙 §八b）。
    # 与 G 归因平行·查验证来源溯源（EXTERNAL R6 vs SELF_PRODUCED 自产）·additive 不改 G 表。
    layer0_attribution = project_layer0(result)
    # 6d. #1041 构造③④：语言域统计层产出度量（判据④ lang_rate + 判据⑤ pure_alias·observability 信号非 criterion）。
    # additive 不改 dimensions/summary/g_attribution/layer0·守既有测零回归（同 layer0_attribution 范式）。
    lang_measures = project_lang_measures(result)
    math_measures = project_symbolic_measures(result)   # #1124 S5-S8 symbolic verified 度量（反 theater 可见）

    # 7. 反 theater 第二三层（#726 片2·anti_theater=True 时跑·默认 False 保片1 零回归）
    if anti_theater:
        if backend_factory is None:
            raise ValueError(
                "anti_theater=True 须提供 backend_factory（每锚点独立 backend·防 cross-contamination）")
        anchors = run_anti_theater_anchor(config, backend_factory, runner=runner)
        regressions = run_reverse_regression()
        anti_theater_passed = all(a.passed for a in anchors) and all(r.passed for r in regressions)
        # ABORT 只升失败锚点/反向回归对应的维度（P0-4·非全维度升·保诊断信息）
        abort_dims: set[str] = set()
        for a in anchors:
            if not a.passed:
                abort_dims.update(_ANCHOR_DIMS.get(a.name, ()))
        for r in regressions:
            if not r.passed:
                abort_dims.add(r.dim)
        for dim in abort_dims:
            if dim in dimensions:
                old = dimensions[dim]
                dimensions[dim] = DimScore(
                    dim=old.dim,
                    status=STATUS_ABORT,
                    permille=-2,
                    threshold=-2,
                    evidence=list(old.evidence) + [f"anti_theater ABORT：维度 {dim} 锚点/反向回归 passed=False"],
                    footnote=old.footnote,
                )
        anti_theater_anchor_out: list = [a.to_dict() for a in sorted(anchors, key=lambda a: a.name)]
        reverse_regression_out: list = [r.to_dict() for r in sorted(regressions, key=lambda r: r.dim)]
    else:
        anti_theater_passed = True
        anti_theater_anchor_out = [{
            "_status": "anti_theater=False（默认·保片1 零回归）",
            "_note": "反 theater §D 第二三层未触发（#726 片2·anti_theater=True 时跑锚点+反向回归）",
        }]
        reverse_regression_out = [{
            "_status": "anti_theater=False（默认）",
            "_note": "反向回归未触发（anti_theater=True 时跑 8 维度判据可证伪 + NE 守恒）",
        }]

    # 8. summary（NE 不混入 PASS·MECHANISM_LIVE 单独计非 PASS·ABORT 维度单独计·P0-4·#723 dead 门计数强制出现·决断4）
    ne_count = sum(1 for d in dimensions.values() if d.status == STATUS_NE)
    pass_count = sum(1 for d in dimensions.values() if d.status == STATUS_PASS)
    mech_live_count = sum(1 for d in dimensions.values() if d.status == STATUS_MECHANISM_LIVE)
    fail_count = sum(1 for d in dimensions.values() if d.status == STATUS_FAIL)
    abort_count = sum(1 for d in dimensions.values() if d.status == STATUS_ABORT)
    examined = len(dimensions) - ne_count   # examined = PASS + MECHANISM_LIVE + FAIL + ABORT（NE 不计·MECHANISM_LIVE 已考核计入）
    abort_prefix = ""
    if abort_count > 0:
        # ABORT 维度名按 DIM_ORDER 序拼接（bit-identical）
        abort_names = [d for d in DIM_ORDER if dimensions.get(d) and dimensions[d].status == STATUS_ABORT]
        abort_prefix = f"ABORT[{'/'.join(abort_names)}]: "
    summary = (f"{abort_prefix}{examined}/{len(dimensions)} examined·"
               f"{ne_count} NE·{pass_count} PASS·{mech_live_count} MECHANISM_LIVE·{fail_count} FAIL"
               + (f"·{abort_count} ABORT" if abort_count > 0 else "")
               + (f"·{dead_design_count} DEAD_DESIGN" if dead_design_count > 0 else "")
               + (f"·{dead_leak_count} DEAD_LEAK" if dead_leak_count > 0 else ""))

    # 9. footnotes（总注·固定·sort bit-identical·#723 dead 门一一列出·决断4 强制复述）
    footnotes = [
        FOOTNOTE_WALL_479,
        FOOTNOTE_WEANING,
        FOOTNOTE_STABLE,
        FOOTNOTE_SPECIES,
    ]
    footnotes.extend(_g_dead_footnotes(g_attribution))
    # #1041 构造③④：lang_measures 信号非 criterion 守（反 theater·防 observability 度量被读成断奶判据）
    footnotes.append(FOOTNOTE_LANG_MEASURES)
    # #1124 M-2：math_measures 信号非 criterion 对等 disclaim（反 theater·报告级诚实对称 lang_measures）
    footnotes.append(FOOTNOTE_MATH_MEASURES)

    # 9b. #727 fixture 限制守（决断5·防误读）：total episode < FIXTURE_SIZE_MIN → STATISTICAL_NOISE。
    # total = result.episodes 数（collect_episodes=True 已强制·#723）·含 language judge + Mode A verify（PRE·
    # #727 决断3 纠偏·非 Mode B POST·后者 gate OFF + WEANING_PRE + 无 source_b 不激活·defer 独立 session）+
    # Mode A task-driven 三路 episode。空 corpus（lang 玩具 1 item）total 可能 < 10 → 诚实标噪声。
    total_episodes = len(_legacy_result_episodes(result))
    assert_int(total_episodes, _where="run_capability_exam.total_episodes")
    if total_episodes < FIXTURE_SIZE_MIN:
        fixture_size_note = FIXTURE_NOTE_NOISE
        footnotes.append(
            f"STATISTICAL_NOISE：total episodes={total_episodes}<{FIXTURE_SIZE_MIN}·"
            f"fixture 不足·PASS/FAIL 非定论（决断5·防误读·须 total≥10 才判真缺口）")
    else:
        fixture_size_note = FIXTURE_NOTE_OK

    return CapabilityReport(
        run_id=result.run_id,
        dimensions=dimensions,
        strength_delta=strength_delta,
        strength_delta_total=strength_delta_total,
        anti_theater_passed=anti_theater_passed,
        anti_theater_anchor=anti_theater_anchor_out,
        reverse_regression=reverse_regression_out,
        g_attribution=g_attribution,
        layer0_attribution=layer0_attribution,
        lang_measures=lang_measures,
        math_measures=math_measures,
        summary=summary,
        footnotes=footnotes,
        fixture_size_note=fixture_size_note,
    )


def run_capability_exam(config: FormalTrainConfig,
                        corpus: list, *,
                        backend: Any,
                        teacher: Any = None,
                        runner: Any = None,
                        anti_theater: bool = False,
                        backend_factory: Callable[[], Any] | None = None,
                        training_mode: bool = False,
                        flat_floors: bool = False) -> CapabilityReport:
    """在独立 backend 中运行统一能力考核，只返回多维报告而不提交训练副作用。"""
    from dataclasses import replace as _dc_replace
    from pure_integer_ai.experiments.evaluation_isolation import (
        isolated_backend_evaluation,
    )

    eval_config = _dc_replace(config, persist_graph_dump=False)
    with isolated_backend_evaluation(backend, teacher=teacher) as isolated:
        eval_backend, eval_teacher = isolated
        return _run_capability_exam_impl(
            eval_config,
            corpus,
            backend=eval_backend,
            teacher=eval_teacher,
            runner=runner,
            anti_theater=anti_theater,
            backend_factory=backend_factory,
            training_mode=training_mode,
            flat_floors=flat_floors,
        )


# ---- 反 theater 锚点 → 期望维度映射（P0-4·ABORT 只升失败锚点对应维度） ----
_ANCHOR_DIMS: dict[str, tuple[str, ...]] = {
    "anchor_arith_no_heldout": (DIM_COMPUTE,),
    "anchor_arith_all_wrong": (DIM_LONG_CODE,),
    "anchor_no_causes_lang": (DIM_THREE_RING, DIM_INTENT),
}


# ---- 锚点 corpus（陷阱1·corpus 层注入·非 metrics 层灌） ----

def _anchor_arith_no_heldout_corpus() -> list[CollectedItem]:
    """锚点1 corpus：nullary 算术 2 样本（mul 同形）→ held_out=0 → rate=0 → ③FAIL（P0-1 重设计）。

    nullary lambda（arity=0）·2 样本 ≥ MIN_DISCOVER_SAMPLES=2 → 全 discover 前 2 + recognize 余 0
    → held_out=0 → rate_permille=(0*1000)//max(0,1)=0 → ③ FAIL（< THRESH_RATE_PERMILLE=500）。
    **诚实边界**：验③判据对 rate_permille 敏感·非验 vm_proof 抓错算子（后者单测覆盖·真错算子 e2e defer #727）。
    """
    return [
        CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                      source=SOURCE_MATH, arith_source="lambda: 5 * 5"),
        CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                      source=SOURCE_MATH, arith_source="lambda: 6 * 6"),
    ]


def _anchor_arith_all_wrong_corpus() -> list[CollectedItem]:
    """锚点2 corpus：arith_specs expected 全错 → Mode A task-driven verified=0 → ⑤FAIL（纠正3 降级）。

    nullary mul 2 样本（discover 触发·学到 mul 算子）+ arith_specs expected 全错（真值 10 标 999）。
    _run_task_driven_generate 消费 arith_specs → skeleton(5*2)=10 ≠ expected 999 → verified=0 → rate=0 → ⑤FAIL。
    """
    return [
        CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                      source=SOURCE_MATH, arith_source="lambda: 5 * 2",
                      arith_specs=(CodeSpec((), (999, 1)),)),   # 真值 10 标 999·全错
        CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                      source=SOURCE_MATH, arith_source="lambda: 6 * 2",
                      arith_specs=(CodeSpec((), (999, 1)),)),   # 真值 12 标 999·全错
    ]


def _anchor_no_causes_corpus() -> list[CollectedItem]:
    """锚点3 corpus：_multi_sent_item 范式·无 cue 词无 causal_pairs → 零 CAUSES 边 → ⑥⑦FAIL（纠正2 e2e 半）。

    tokens=["a","b。","c","d。"]·_CUE_WORDS[LANG_ZH] 不含 a/b/c/d → 无 cue → observe 零 CAUSES 边
    （无论 CUE_EXTRACTOR_MODE ON/OFF·P1-3·formal_train 内 gate 强制 ON 但词不命中）→
    snapshot pre==post → strength_delta_total=0 → collapse_summary verified=0（无 pr_vector episode 跳过）
    → 三柱全 0 → ⑥ FAIL（pillars_all_ok=False）+ ⑦ FAIL（delta=0）。
    """
    return [
        CollectedItem(
            tokens=["a", "b。", "c", "d。"],
            role_seq=[1, 1, 1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
        ),
    ]


# ---- 反 theater 第二层：e2e 锚点自检（#726 片2） ----

def run_anti_theater_anchor(config: FormalTrainConfig,
                            backend_factory: Callable[[], Any],
                            runner: Any = None) -> list[AnchorCheck]:
    """反 theater e2e 锚点自检（§D 第二层·#726 片2）。

    3 锚点走 corpus 层注入（陷阱1 正路）·每个锚点独立 backend + 跑 run_capability_exam
    （anti_theater=False·防递归）→ formal_train 主路径自己产出 metrics → harness 投影
    → 断言期望维度判 FAIL（非死写 PASS）。任一 passed=False → 调用方升该维度 ABORT（P0-4）。
    """
    checks: list[AnchorCheck] = []
    r = runner or DefaultRoundRunner()

    anchors_spec: list[tuple] = [
        ("anchor_arith_no_heldout", _anchor_arith_no_heldout_corpus(),
         "nullary 算术 2 样本（lambda:5*5 + lambda:6*6）→ held_out=0 → rate_permille=0",
         "验③判据对 rate_permille 敏感·非验 vm_proof 抓错算子（test_stage9_arith_observe:409 + test_stage9_structure_discover:800 单测覆盖·真错算子 e2e defer #727）"),
        ("anchor_arith_all_wrong", _anchor_arith_all_wrong_corpus(),
         "算术 corpus arith_specs expected 全错（lambda:5*2 真值 10 标 999）→ task-driven verified=0",
         "Mode A task-driven（result.generate）·非 Mode B cross-verify·非 selection_pref（真 sel_pref collide defer #730·#730 须先补输出度量）"),
        ("anchor_no_causes_lang", _anchor_no_causes_corpus(),
         "_multi_sent_item 无 cue 词无 causal_pairs（a/b/c/d 不命中 _CUE_WORDS）→ observe 零 CAUSES 边",
         "_multi_sent_item 无 cue 词·无论 CUE_EXTRACTOR_MODE ON/OFF 都不建 CAUSES（formal_train 内 gate 强制 ON·P1-3）"),
    ]

    for name, corpus, injected, footnote in anchors_spec:
        b = backend_factory()
        cfg = FormalTrainConfig(
            run_dir=os.path.join(config.run_dir, name),
            run_id=name,
        )
        rep = run_capability_exam(cfg, corpus, backend=b, runner=r, anti_theater=False,
                                  training_mode=True, flat_floors=True)
        dims = rep.dimensions
        if name == "anchor_no_causes_lang":
            # 验 ⑥+⑦ 双 FAIL
            d6 = dims[DIM_THREE_RING]
            d7 = dims[DIM_INTENT]
            passed = (d6.status == STATUS_FAIL and d7.status == STATUS_FAIL)
            actual = f"⑥{d6.status}/⑦{d7.status}"
            evidence = [
                f"⑥三环 status={d6.status}·permille={d6.permille}",
                f"⑦初心 status={d7.status}·permille={d7.permille}",
                f"strength_delta_total={rep.strength_delta_total}（零 CAUSES 边→delta=0）",
            ]
        elif name == "anchor_arith_no_heldout":
            d3 = dims[DIM_COMPUTE]
            passed = d3.status == STATUS_FAIL
            actual = d3.status
            evidence = [
                f"③计算 status={d3.status}·permille={d3.permille}·threshold={d3.threshold}",
                f"rate_permille=0<500 → FAIL（held_out=0·nullary 2 样本全发现无留出）",
            ]
        else:   # anchor_arith_all_wrong
            d5 = dims[DIM_LONG_CODE]
            # STEP2 #889：⑤ status 取严 NE（generate 字面零测·D5）·锚点改验 Mode A FAIL（evidence）
            # 反 theater：坏 corpus -> Mode A verified=0 -> rate=0 -> Mode A FAIL（evidence 标）·非死写 PASS
            mode_a_fail = any("Mode A task-driven" in e and ("<500" in e or "FAIL" in e) for e in d5.evidence)
            passed = (d5.status == STATUS_NE and mode_a_fail)
            actual = d5.status
            evidence = [
                f"⑤长代码 status={d5.status}·permille={d5.permille}·threshold={d5.threshold}",
                f"Mode A task-driven generate.rate_permille=0<500 -> FAIL（verified=0·expected 全错）",
                f"⑤取严 NE（#889 D5·generate 字面零测）·Mode A FAIL 验坏 corpus 反 theater",
            ]
        # STEP2 #889：anchor_arith_all_wrong expected=NE（⑤取严）·其余锚点 expected=FAIL
        _expected = STATUS_NE if name == "anchor_arith_all_wrong" else STATUS_FAIL
        checks.append(AnchorCheck(
            name=name,
            injected=injected,
            expected_status=_expected,
            actual_status=actual,
            passed=passed,
            evidence=evidence,
            footnote=footnote,
        ))

    return checks


# ---- 反 theater 第三层：反向回归（#726 片2·project_dimensions 直调） ----

def _assert_rr_int(*vals: int) -> None:
    """run_reverse_regression fake_result 数值字段纯整数守（#726 P1-2·防手滑 0.0）。

    7 个 fake fixture 全字段守（对抗审 2 P1-a：原只 fake_ne 守 2 字段·与 docstring 自承不符）。
    StageMetrics/GeneralizationSummary/GenerateSummary 无 __post_init__·靠此 helper + 字面量诚实双保险。
    """
    assert_int(*vals, _where="run_reverse_regression fake metrics")
    assert_no_float(*vals, _where="run_reverse_regression fake metrics")


def run_reverse_regression() -> list[ReverseRegressionCase]:
    """反向回归（§D 第三层·#726 片2）—— project_dimensions 直调 + fake_result 注入。

    逐维度验判据可证伪（regressable·②③⑤⑥⑦⑧）或 NE 守恒（ne_conservation·①④）。
    NE 守恒①④：注入非 NE 诱因（graph_size>0·看似全 PASS）→ 断言仍 NE（P0-3·守未来偷偷塞判据）。
    ⑥精确联立（三柱全 ok + delta=0 → FAIL）：测 strength_delta>0 那条腿（纠正2·投影反例·非 e2e）。
    """
    cases: list[ReverseRegressionCase] = []

    # ---- NE 守恒 fixture：注入非 NE 诱因（看似全 PASS）→ ①④ 须仍 NE ----
    fake_ne = FormalTrainResult(
        run_id="rr_ne",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
        generalization=GeneralizationSummary(total_held_out=2, recognized=2, verified=2),   # rate=1000 看似 PASS
    )
    _assert_rr_int(
        fake_ne.final_metrics.graph_size, fake_ne.final_metrics.causes_coverage,
        fake_ne.collapse_summary["pillar1_ok"], fake_ne.collapse_summary["pillar2_ok"],
        fake_ne.collapse_summary["pillar3_ok"],
        fake_ne.generalization.total_held_out, fake_ne.generalization.recognized,
        fake_ne.generalization.verified,
    )   # P1-2 纯整数守（全字段·对抗审 2 P1-a）
    dims_ne = project_dimensions(fake_ne, strength_delta_total=5, backend=None)

    # ①概念 NE 守恒
    d1 = dims_ne[DIM_CONCEPT]
    cases.append(ReverseRegressionCase(
        dim=DIM_CONCEPT,
        category="ne_conservation",
        bad_fixture="注入非 NE 诱因（graph_size=100+causes_cov=600+三柱 ok+delta=5+generalization rate=1000·看似全 PASS）",
        expected_status=STATUS_NE,
        actual_status=d1.status,
        passed=d1.status == STATUS_NE,
        evidence=[f"①概念 status={d1.status}（注入诱因后仍 NE = 守恒）",
                  "守未来偷偷塞判据：若改①读 graph_size 判 PASS·此测抓到（P0-3）"],
        footnote="NE 守恒非坏 fixture（陷阱2）·注入诱因验不许偷偷塞判据",
    ))
    # ④长文本 NE 守恒
    d4 = dims_ne[DIM_LONG_TEXT]
    cases.append(ReverseRegressionCase(
        dim=DIM_LONG_TEXT,
        category="ne_conservation",
        bad_fixture="注入非 NE 诱因（同①·看似全 PASS）",
        expected_status=STATUS_NE,
        actual_status=d4.status,
        passed=d4.status == STATUS_NE,
        evidence=[f"④长文本 status={d4.status}（注入诱因后仍 NE = 守恒）"],
        footnote="NE 守恒·零测维度不许偷偷塞判据",
    ))

    # ---- regressable fixtures ----
    # ②结构：causes_coverage=0 → FAIL
    fake2 = FormalTrainResult(
        run_id="rr2",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=0),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
    )
    _assert_rr_int(fake2.final_metrics.graph_size, fake2.final_metrics.causes_coverage,
                   fake2.collapse_summary["pillar1_ok"], fake2.collapse_summary["pillar2_ok"],
                   fake2.collapse_summary["pillar3_ok"])
    dims2 = project_dimensions(fake2, strength_delta_total=5, backend=None)
    d_s2 = dims2[DIM_STRUCTURE]
    cases.append(ReverseRegressionCase(
        dim=DIM_STRUCTURE,
        category="regressable",
        bad_fixture="causes_coverage=0（< THRESH_CAUSES_COV=500）",
        expected_status=STATUS_FAIL,
        actual_status=d_s2.status,
        passed=d_s2.status == STATUS_FAIL,
        evidence=[f"②结构 status={d_s2.status}·permille={d_s2.permille}·cov=0<500"],
    ))

    # ③计算：generalization 非 None + verified=0/total=2 → rate=0 → FAIL
    fake3 = FormalTrainResult(
        run_id="rr3",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
        generalization=GeneralizationSummary(total_held_out=2, recognized=0, verified=0),
    )
    _assert_rr_int(fake3.final_metrics.graph_size, fake3.final_metrics.causes_coverage,
                   fake3.collapse_summary["pillar1_ok"], fake3.collapse_summary["pillar2_ok"],
                   fake3.collapse_summary["pillar3_ok"],
                   fake3.generalization.total_held_out, fake3.generalization.recognized,
                   fake3.generalization.verified)
    dims3 = project_dimensions(fake3, strength_delta_total=5, backend=None)
    d_s3 = dims3[DIM_COMPUTE]
    cases.append(ReverseRegressionCase(
        dim=DIM_COMPUTE,
        category="regressable",
        bad_fixture="generalization 非 None + verified=0/total=2 → rate_permille=0",
        expected_status=STATUS_FAIL,
        actual_status=d_s3.status,
        passed=d_s3.status == STATUS_FAIL,
        evidence=[f"③计算 status={d_s3.status}·permille={d_s3.permille}·rate=0<500"],
    ))

    # ⑤长代码 NE 守恒（STEP2 #889 取严 NE·D5-enforcing）：generate 字面零测 -> ⑤ 恒 NE
    # 注入 Mode A 诱因（generate 非 None + verified=0 -> Mode A rate=0·看似该 FAIL）-> ⑤ 仍 NE（取严·禁 Mode A 偷渡 status）
    fake5 = FormalTrainResult(
        run_id="rr5",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
        generate=GenerateSummary(total_tasks=2, selected=2, verified=0),
    )
    _assert_rr_int(fake5.final_metrics.graph_size, fake5.final_metrics.causes_coverage,
                   fake5.collapse_summary["pillar1_ok"], fake5.collapse_summary["pillar2_ok"],
                   fake5.collapse_summary["pillar3_ok"],
                   fake5.generate.total_tasks, fake5.generate.selected, fake5.generate.verified)
    dims5 = project_dimensions(fake5, strength_delta_total=5, backend=None)
    d_s5 = dims5[DIM_LONG_CODE]
    cases.append(ReverseRegressionCase(
        dim=DIM_LONG_CODE,
        category="ne_conservation",
        bad_fixture="generate 非 None + verified=0/total=2 -> Mode A rate=0（看似该 FAIL）·⑤取严 NE 不被 Mode A 偷渡",
        expected_status=STATUS_NE,
        actual_status=d_s5.status,
        passed=d_s5.status == STATUS_NE,
        evidence=[f"⑤长代码 status={d_s5.status}·permille={d_s5.permille}·Mode A rate=0<500（evidence 标·不进 status）"],
        footnote="⑤取严 NE（#889 D5·generate 字面零测）·Mode A 进 evidence 不偷渡 status",
    ))

    # ⑥三环精确联立：三柱全 ok + delta=0 → FAIL（测 strength_delta>0 那条腿·纠正2）
    fake6 = FormalTrainResult(
        run_id="rr6",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},   # 三柱全 ok
    )
    _assert_rr_int(fake6.final_metrics.graph_size, fake6.final_metrics.causes_coverage,
                   fake6.collapse_summary["pillar1_ok"], fake6.collapse_summary["pillar2_ok"],
                   fake6.collapse_summary["pillar3_ok"])
    dims6 = project_dimensions(fake6, strength_delta_total=0, backend=None)   # delta=0
    d_s6 = dims6[DIM_THREE_RING]
    cases.append(ReverseRegressionCase(
        dim=DIM_THREE_RING,
        category="regressable",
        bad_fixture="collapse 三柱全 ok + strength_delta_total=0（联立 strength_delta>0 那条腿）",
        expected_status=STATUS_FAIL,
        actual_status=d_s6.status,
        passed=d_s6.status == STATUS_FAIL,
        evidence=[f"⑥三环 status={d_s6.status}·三柱 ok 但 delta=0 → 联立破在 delta 腿",
                  "投影反例·非 e2e（e2e 造不出·CAUSES 边同时撑三柱+delta·纠正2）"],
        footnote="测 project_dimensions 联立判据·非 corpus 反例（陷阱1 边界·诚实标注）",
    ))

    # ⑦初心：delta=0 → FAIL（同 fake6）
    d_s7 = dims6[DIM_INTENT]
    cases.append(ReverseRegressionCase(
        dim=DIM_INTENT,
        category="regressable",
        bad_fixture="strength_delta_total=0（≤ THRESH_STRENGTH_DELTA=1）",
        expected_status=STATUS_FAIL,
        actual_status=d_s7.status,
        passed=d_s7.status == STATUS_FAIL,
        evidence=[f"⑦初心 status={d_s7.status}·delta=0≤0"],
        footnote=FOOTNOTE_WALL_479,
    ))

    # ⑧记忆：memory_rows=0（backend=None）+ G5-C consumer=0 → FAIL（⑧整体仍 FAIL·⑧b defer·验不许美化成 PASS）
    # #732 G5-C 落后 ⑧a 可活（gate ON 时 consumer_triggers>0）·但 ⑧=⑧a AND ⑧b·⑧b 仍 defer → ⑧ 整体仍 FAIL。
    # ⑧ regressable：bad case（backend=None·consumer=0）FAIL·good case（gate ON·consumer>0）仍 FAIL（⑧b defer）·
    # 但 evidence 变化（consumer_triggers 0→>0）·非 NE 守恒（⑧ 始终 FAIL 非 NE·bad/good evidence 可区分）。
    d_s8 = dims6[DIM_MEMORY]
    cases.append(ReverseRegressionCase(
        dim=DIM_MEMORY,
        category="regressable",
        bad_fixture="memory_rows=0（backend=None）+ G5-C consumer=0（gate ON 亦无 consolidate）",
        expected_status=STATUS_FAIL,
        actual_status=d_s8.status,
        passed=d_s8.status == STATUS_FAIL,
        evidence=[f"⑧记忆 status={d_s8.status}·memory_rows={d_s8.permille}·⑧a consumer={d_s8.evidence[1] if len(d_s8.evidence) > 1 else 'N/A'}",
                  "⑧整体仍 FAIL（⑧b defer·⑧=⑧a AND ⑧b·#732 G5-C 落 code 解⑧a 但 ⑧b 仍断·验不许美化成 PASS）"],
    ))

    return cases
