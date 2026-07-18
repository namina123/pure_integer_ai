"""experiments.formal_train — 正式训练驱动（§十二五阶段 + --resume 续训 + E7 pre-flight 放量门）。

  formal_train(config, corpus, *, backend) -> FormalTrainResult   五阶段编排主入口
  pre_flight(ctx, corpus, *, rounds) -> PreFlightReport           E7 放量门（大规模放量前守卫）

**五阶段编排**（§十二最优路径·能力是地基涌现非单独训练）：
  STAGE1_SKELETON     observe only 建结构骨架（冷启动·教师元定义补 PRIMARY）
  STAGE2_CAUSES_ABS   observe only 因果+抽象并行（CAUSES 结构化源① + cue 自产源② + 断奶前教师③·A1 接线后生产 observe 自产②）
  STAGE3_REWARD       H2 小批量标定权重 → 开全量 reward 闭环（judge+反传）
  STAGE4_PROMOTE_WEAN promote 三重 + 断奶双曲线趋势 D1（LLM 退场）
  STAGE5_MULTIMODAL   defer（非训练·机制骨架随模态扩展）
  observe 先 reward 后破死锁（阶段1-2 observe only·阶段3 才开 reward）。
  度量门控合格才进下阶段（stage_metric_gate·防缺防超喂）。

**--resume 续训（E1/E4/E8·几百G不重训红线）**：
  load_run(base_run_id 终 dump) → cursor_resume stage-skip（跳已完成 skippable·E8）→
  check_replay_coverage（E4·未达标禁续训防 miss→None 静默降级）→ 增量喂新语料。
  阶段3 reward 闭环须重标定权重 H2 → non-skippable（不跳）。
  每正式 run 新 run_id·终 dump = 跨 run 续训权威 base（E1）。

**H2 鸡生蛋破解（§十四 line1030）**：阶段3 先小批量离线标定权重（教师 ground-truth 经录放层·
  纯整数网格最大化 agreement）→ 权重定后开全量 reward 反传。默认权重期 reward 不落 strength（防污染）。

**E7 pre-flight 放量门（守几百G不重训红线）**：配置错/度量盲直接放量=白训。pre-flight = 小规模试跑
  → 6 验收项全过才放量：①度量真有信号 ②内存峰值<mem_hard_pct ③reward gate 实际生效
  ④replay 覆盖率≥阈值 ⑤cursor resume 能跳 ⑥防塌柱③ 探索压力（S12 collapse_ok）。
  config.pre_flight=True 接通 formal_train() 生产主入口（S12 follow-up·破纸面闭合：6 项 passed 生产真读）·
  默认 OFF 守 bit-identical（DictBackend + SQLiteBackend 均支持 snapshot/rollback·§施工序 4.2-1）。trial 副作用
  snapshot/rollback 5 状态隔离（backend snapshot + _id_pool + ConceptIndex._index/_loaded_spaces + work_memory）
  ·stage loop 不变。pre-flight 失败=禁放量（raise RuntimeError·修配置重试）非"继续跑看看"。

**RoundRunner 协议（可换·experiments 是可换层）**：per-round 认知执行（observe+episode）注入式·
  DefaultRoundRunner 走真接线（observe + episode_loop + build_judge_fn）·用户可换自定义预处理/检索。

铁律：纯整数（度量×1000·round_id 整序无墙钟）/ 确定性（per-round 序确定·bit-identical·
  终 dump 同 cursor.dump_run）/ gate 二分（TRAINING_MODE/TEACHER_MODE 默认 OFF·live-read）/
  几百G不重训（新 run_id·终 dump base·cursor stage-skip·度量门控合格才进下）。
诚实边界：formal_train 是编排非认知（能力涌现非训练保证·地基建好≠能力必现·D 墙）/ pre-flight 是
  软守卫非硬保证（经验阈值 oracle 标）/ 默认权重期 reward 不落 strength（H2·防错权重污染图）/
  续训跨 run 须重标 reward 权重 H2（非 skippable）/ stable≠correct。
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, TYPE_CHECKING, runtime_checkable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend, DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_CONCEPTNET, SOURCE_CHINESE_KB
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY
from pure_integer_ai.storage.composes_attr import COMPOSES_ATTR_TABLE
from pure_integer_ai.storage.op_confidence import (
    OP_CONFIDENCE_TABLE, record_op_outcome, read_op_confidence)
from pure_integer_ai.storage.concept_identity import CONCEPT_IDENTITY_TABLE
from pure_integer_ai.storage.concept_correspondence import CONCEPT_CORRESPONDENCE_TABLE
from pure_integer_ai.storage.experience_count import EXPERIENCE_COUNT_TABLE, pack_ctx_code
from pure_integer_ai.storage.chapter_seq import CHAPTER_SEQ_TABLE
from pure_integer_ai.storage.selection_pref_count import SELECTION_PREF_COUNT_TABLE
from pure_integer_ai.storage.pronoun_resolution_count import PRONOUN_RESOLUTION_COUNT_TABLE
from pure_integer_ai.storage.sense_candidates import SENSE_CANDIDATES_TABLE
from pure_integer_ai.storage.abstract_mark import ABSTRACT_MARK_TABLE
from pure_integer_ai.cognition.understanding.emergent_role import POSITION_HIST_TABLE
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.cognition.shared.types import (
    InputPayload, Segment, IntentType, Episode, JudgeWeights, ConceptRef,
    PathResult, PathData, OutputResult, OutputPart, CodeSpec,
    FloorActivation,
    STAGE_TRAINING, MODALITY_LANGUAGE, MODALITY_CODE, MODALITY_ARITH, LANG_ZH, LANG_NONE,
    DOMAIN_TEXT, DOMAIN_CODE, DOMAIN_MATH, INTENT_QUESTION, INTENT_COMMAND, WEANING_PRE, WEANING_POST,
    TERMINAL_REACHED_SINK,
    VERIFY_SOURCE_EXTERNAL, VERIFY_SOURCE_SELF_PRODUCED,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.observe import observe
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_cues_gated, extract_numeric_claims_gated, extract_universal_claims_gated,
    extract_existential_claims_gated,
    extract_property_claims_gated, extract_comparison_claims_gated, extract_similar_claims_gated,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.process.episode import episode_loop, _ctx_tag
from pure_integer_ai.cognition.process.structure_discover import (
    auto_discover_operators, recognize_operators, shape_signature,
    load_discovered_operators, probe_arity,
    route_samples_for_discovery, _collect_slot_lcas, _collect_cue_sig,
    _normalize_abstract_sig,
    DiscoveredOperator, Recognition, MIN_DISCOVER_SAMPLES,
    _OP_CONF_RATE_SCALE,
)
from pure_integer_ai.config import gates
from pure_integer_ai.training.stages import (
    STAGES, STAGE1_SKELETON, STAGE2_CAUSES_ABS, STAGE3_REWARD,
    STAGE4_PROMOTE_WEAN, STAGE5_MULTIMODAL, SKIPPABLE_STAGES,
    stage_gate_config, stage_metric_gate, stage_active_gates,
    is_skippable, build_judge_fn, StageMetrics, StageGateConfig,
)
from pure_integer_ai.training.oracle import calibrate_weights, validate_b1_b4
from pure_integer_ai.training.vm_proof import vm_proof_fn_factory, execute_composes_value
from pure_integer_ai.training.time_seq_proof import time_seq_proof_fn_factory
from pure_integer_ai.training.numeric_proof import numeric_proof_fn_factory
from pure_integer_ai.training.universal_proof import universal_proof_fn_factory
from pure_integer_ai.training.existential_proof import existential_proof_fn_factory
from pure_integer_ai.training.comparison_proof import comparison_proof_fn_factory
from pure_integer_ai.cognition.shared.edge_types import EDGE_PRECEDES, EDGE_CAUSES, EDGE_T_STEP
from pure_integer_ai.training.mode_b_cross_verify import cross_verify_pair
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.training.promote import promote_edge, promote_report, promote_memory_consolidate
from pure_integer_ai.training.cursor import (
    dump_run, load_run, CursorState, cursor_resume, check_replay_coverage,
    mark_completed, DUMP_TABLES,
)
from pure_integer_ai.teacher.weaning import weaning_check
from pure_integer_ai.teacher.recordable_teacher import MODE_OFF
from pure_integer_ai.experiments.collection import (
    CollectedItem, CollectionReport, COLLECT_PRECEDES, resolve_is_a_facts, resolve_abstract_facts, resolve_sense_facts,
    resolve_struct_bind_pairs, resolve_causes_facts, resolve_alias_facts,
    resolve_number_facts, resolve_degree_facts,
    resolve_transform_rules, resolve_inverse_relations, resolve_mereology_facts, resolve_antonym_facts, resolve_similar_facts,
    corpus_relevant_vocab, filter_pairs_to_vocab,
)
from pure_integer_ai.cognition.understanding.is_a import bootstrap_is_a_edges
from pure_integer_ai.cognition.understanding.causes import bootstrap_causes_edges
from pure_integer_ai.cognition.understanding.mereology import bootstrap_mereology_edges
from pure_integer_ai.cognition.understanding.antonym import bootstrap_antonym_edges
from pure_integer_ai.cognition.understanding.similar import bootstrap_similar_edges
from pure_integer_ai.cognition.understanding.cue_words import populate_degree_cues   # #1134 degree cue→intensity cache boot populate
from pure_integer_ai.cognition.understanding.alias_bridge import bootstrap_alias_edges
from pure_integer_ai.cognition.understanding.number_grounding import bootstrap_number_grounding
from pure_integer_ai.cognition.process.struct_bind import (
    bootstrap_struct_bind_edges, collect_skeleton_slot_refs,
)
from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map_external
from pure_integer_ai.cognition.understanding.sense_lookup_hook import make_sense_lookup
from pure_integer_ai.storage.sense_candidates import bootstrap_sense_candidates
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_word_concept_signals, bootstrap_operator_signals, bootstrap_modal_signals,
    bootstrap_negation_signals, bootstrap_action_signals,
)
from pure_integer_ai.experiments.metrics import MetricsCollector

if TYPE_CHECKING:
    from pure_integer_ai.teacher.probe_set import ProbeSet   # W4 D4·TrainContext.probe_set 注解（运行时 lazy·避上向 import）

# E7 pre-flight 试跑轮次经验初值（oracle 可调·§十二 line904）
PRE_FLIGHT_ROUNDS = 50000
# E7 pre-flight ② 内存预算：每 trial round 允许的概念点+边增长上限（stub ② 修·防超线性膨胀 OOM·
# 真 OS 级 mem_hard_pct 监控 defer 工程层·此为纯整 in-process 代理·oracle 标）
PRE_FLIGHT_MEM_BUDGET_PER_ROUND = 4096
# H2 小批量标定集大小（阶段3 开全量 reward 前先标权重·§十四 H2）
H2_CALIB_BATCH = 16


# ---- 训练上下文 ----

@dataclass
class TrainContext:
    """训练上下文（持 backend/图接口/教师/权重/work_memory·per-round 消费）。

    core_space    核心概念空间（训练期增长·训练后固化）。
    concept_graph 卷三读图接口（generate/judge 消费）。
    teacher       RecordableLLMTeacher | None（断奶前在位·断奶后退场）。
    weights       JudgeWeights（H2 标定后设·默认 (1,1,1)·断奶后冻结）。
    """

    backend: StorageBackend
    core_space: AbstractSpace
    edge_store: EdgeStore
    node_store: NodeStore
    concept_index: ConceptIndex
    concept_graph: ConceptGraph
    teacher: Any = None
    weights: JudgeWeights = field(default_factory=JudgeWeights)
    work_memory: WorkMemory = field(default_factory=WorkMemory)
    weaning_phase: int = WEANING_PRE
    judge_source_id: int | None = None      # W3 D3·独立裁判 source_id（config 注入·:463 caller 传 build_judge_fn）
    judge_source_independent: bool = False  # W3 D3·build_judge_fn 算结果（:463 设·路径 B :2018 读·默认 False 守 bit-identical）
    probe_set: ProbeSet | None = None       # W4 D4·留出探针集（formal_train 主入口切分建·默认 None 守 bit-identical）
    probe_corpus: list = field(default_factory=list)  # W4 D4·探针 CollectedItem（held-out·W6 模拟退场 eval 读·默认空）
    probe_set_disjoint: bool = False        # W4 D4·探针∩训练集=∅（主入口 is_disjoint 算·路径 B 读·默认 False 守 bit-identical·镜像 W3 judge_source_independent）
    holdout_retention: int = 0              # W4 D4·探针保持率 track（默认 0·真值 W6 模拟退场 eval 采·D1 曲线②·record_round 传）
    e2_eval_passed: bool = False            # W6 E2·模拟退场 eval 结果（算术域三条件 and·_run_simulated_offline_eval 设·路径 B 读·默认 False 守 bit-identical）
    memory_read: Any = None       # M10 第一刀 11a·记忆一层阅读（训练期实例化·SpaceContext 训练期守 None·11d 落点② 写）
    memory_interact: Any = None   # M10 第一刀 11a·记忆二层交互

    @property
    def space_id(self) -> int:
        return self.core_space.space_id


def make_train_context(backend: StorageBackend, *,
                       teacher: Any = None,
                       weights: JudgeWeights | None = None,
                       companion: bool = False) -> TrainContext:
    """构造 TrainContext（bootstrap + 核心空间 + 图接口·formal_train/pre_flight 用）。

    companion=True 建伴随库（surface 文本留档）·False 则 None（observe 仍可跑·surface 不留档）。
    """
    bootstrap(backend)
    # cognition 扩展表（position_hist·缺口#1 emergent_role 主导度闸专用表·doc line580①·
    # storage.bootstrap 只注册核心表·cognition 扩展表在用前注册·守依赖单向向下）
    from pure_integer_ai.cognition.understanding.emergent_role import register_position_hist
    register_position_hist(backend)
    # teacher 扩展表（D5 预验台账·断奶前 Mode A vs B 并行标定·§十一 #4-bis line712·守依赖单向向下）
    from pure_integer_ai.teacher.weaning_calibration import register_weaning_calibration
    register_weaning_calibration(backend)
    # COMPOSES 程序属性表（A3·代码域 AST→COMPOSES 节点算子/操作数/立即数/STORE 目标持久化·
    # vm_proof_fn 经 ConceptGraph.read_composes_tree 读回·守依赖单向向下）
    from pure_integer_ai.storage.composes_attr import register_composes_attr
    register_composes_attr(backend)
    # 算子置信度台账（§8.7-洗·洗净循环反馈半闭环·op_confidence MUTABLE_MONOTONE·
    # _verify_generalization 写 vm_proof 验结果·recognize_operators 择优读·守依赖单向向下）
    from pure_integer_ai.storage.op_confidence import register_op_confidence
    register_op_confidence(backend)
    # 概念身份索引持久化（Task #475·§8.7-idx·跨 run _index 重建·concept_identity APPEND_ONLY·
    # ConceptIndex lazy 重建 + ensure 写·守依赖单向向下·载入算子可 inline + observe 续训 dedup）
    from pure_integer_ai.storage.concept_identity import register_concept_identity
    register_concept_identity(backend)
    # P0a·概念↔对应台账（码点 ordinal·surface=str 时 ensure 写·surface_of resolver 读产文本·解 A 偏离）。
    # concept_correspondence APPEND_ONLY·独立 core=False 表（决策3 concept_node 不动·决策4 序列符号范式）·
    # 一个抽象概念 ↔ 一条 ordinal 对应·跨 run load 还原·dump 含以续训保留文本·守依赖单向向下。
    from pure_integer_ai.storage.concept_correspondence import register_concept_correspondence
    register_concept_correspondence(backend)
    # 概念维经验计数台账（阶段1地基层·三把钥匙公共 0→1 根因·experience_count MUTABLE_MONOTONE·
    # 边级 sn/tn 的概念维对偶·阶段2 reward feed 写 + 阶段3 attractor 词终止读 effective_freq·守依赖单向向下）
    from pure_integer_ai.storage.experience_count import register_experience_count
    register_experience_count(backend)
    # 对应泛化 v2：结构反推 tally 台账（structure_match_count APPEND_ONLY·distinct forming-sample 计数·
    # relation-specific + structure-grounded·tally_cue_slot_matches 写 + _structure_match_ok 读·守依赖单向向下）
    from pure_integer_ai.storage.structure_match_count import register_structure_match_count
    register_structure_match_count(backend)
    # 篇章结构序承载（阶段1c·缺口①·修正分析九v2·chapter_seq_table APPEND_ONLY·段 struct_ref 章节标记·
    # observe 写 + generate M5 章边界分页候选读·守依赖单向向下·反 theater 最小消费者同期落）
    from pure_integer_ai.storage.chapter_seq import register_chapter_seq
    register_chapter_seq(backend)
    # 刀5 件5 选择倾向共现统计台账（§十 边约束·selection_pref_count MUTABLE_MONOTONE·
    # (concept_a, argument_class IS_A LCA) 类聚合共现 count·observe 写 sp_tn·predicate 写时识别 defer S4·
    # PR 软加权 dock seed defer S4·守 reward CAUSES-only 不接 edge reward·守依赖单向向下）
    from pure_integer_ai.storage.selection_pref_count import register_selection_pref_count
    register_selection_pref_count(backend)
    # 刀6 件7 sense 多义候选台账（sense_candidates MUTABLE_MONOTONE·key=surface_hash·
    # (token surface_hash, sense ConceptRef) 1:N 多义候选·observe 写 sc_tn + boot 种 base_count·
    # 理解侧 recognize clone 选 sense 候选源·守 reward CAUSES-only 不接 edge reward·守依赖单向向下·#479 墙不破）
    from pure_integer_ai.storage.sense_candidates import register_sense_candidates
    register_sense_candidates(backend)
    # B6 指代维 方案3 tn+fn 路 pronoun_resolution_count 台账（MUTABLE_MONOTONE·(pronoun,antecedent) pair-key·
    # 悬空 self-loop·pr_tn=observe 决策时 / pr_fn=失败侧悬空 per-occurrence / pr_sn=教师 P2 defer·
    # resolve_pronoun_occurrence per-occurrence 写·自消费读 pr_tn 加候选分·守 reward CAUSES-only 不接 edge reward·
    # §九.2 病灶"attribute 给谁"=per-occurrence 落 pronoun·守依赖单向向下·集中注册守 pre_flight snapshot 对称）
    from pure_integer_ai.storage.pronoun_resolution_count import register_pronoun_resolution_count
    register_pronoun_resolution_count(backend)
    # G2 修饰方向A：modification_hist 表（ 的-cue head/modifier 统计·source write gate-independent·read gated）。
    from pure_integer_ai.cognition.understanding.modification_direction import register_modification_hist
    register_modification_hist(backend)
    # §七实现层 modality_subspace（§7.4 L212 + §7.7.1 路径 B·abstract_mark DISC_NONE·节点抽象归属标记
    # 多维 modality/lang/domain/topo·set_mark 幂等 upsert + query_intersection 多维相交·
    # modality_marker 节点列迁 MARK_MODALITY + lang/domain 挂词形 NODE_WORD·守不污染节点列·依赖单向向下）
    from pure_integer_ai.storage.abstract_mark import register_abstract_mark
    register_abstract_mark(backend)
    reg = SpaceRegistry(backend)
    core = AbstractSpace.create(reg, "core")
    comp = CompanionSpace.create(reg, "companion") if companion else None
    # M10 第一刀 11a·记忆空间两层实例化（记忆空间是三重缠绕的家·经验计数/概念阻断/静默学习的宿主）。
    # 挂 TrainContext·训练期 SpaceContext.memory_read 守 None（守 observe bit-identical·训练期核心养洁净）·
    # 11d reward_propagate 落点② 写 memory_read（训练期建阅读记忆种子·line998-1001）·memory_active 守 False。
    # register_memory_table 已在 bootstrap 调（core=True）·cursor.DUMP_TABLES 已含 memory_item·不重注不改 dump。
    from pure_integer_ai.storage.spaces.memory_space import MemorySpace
    mem_read = MemorySpace.create(reg, "memory_read")
    mem_interact = MemorySpace.create(reg, "memory_interact")
    es = EdgeStore(backend)
    ns = NodeStore(backend)
    ci = ConceptIndex(backend, comp)
    g = ConceptGraph(backend)
    return TrainContext(
        backend=backend, core_space=core, edge_store=es, node_store=ns,
        concept_index=ci, concept_graph=g, teacher=teacher,
        weights=weights or JudgeWeights(), work_memory=WorkMemory(),
        memory_read=mem_read, memory_interact=mem_interact,
    )


# ---- 校准样本（H2） ----

@dataclass
class CalibrationSample:
    """H2 标定样本（小批量·教师 ground-truth 经录放层·judge 对齐 GT）。

    judge_fn(sample, weights=...) -> (reward, GMeta)（calibrate_weights 契约）。
    teacher_gt(sample) -> int（GT_PASS/GT_FAIL·教师 ground-truth 经录放层零 LLM）。
    """

    output: Any
    dag_path: Any
    input_payload: InputPayload
    graph: ConceptGraph
    workmem: WorkMemory


def _make_calib_judge_fn(teacher: Any, weaning_phase: int
                         ) -> tuple[Any, Any]:
    """造 H2 标定用 judge_fn + teacher_gt（calibrate_weights 契约·教师 GT 经录放层）。

    judge_fn(sample, weights=weights) -> (reward, GMeta)·self_proof_fn=teacher.judge_ground_truth（weaning pre）。
    teacher_gt(sample) -> int（GT_PASS/GT_FAIL）。
    """
    from pure_integer_ai.cognition.result.judge import judge as _judge

    def judge_fn(sample: CalibrationSample, *,
                 weights: JudgeWeights = JudgeWeights()) -> tuple[int, Any]:
        spf = teacher.judge_ground_truth if (teacher is not None
                                             and weaning_phase == WEANING_PRE) else None
        return _judge(sample.output, sample.dag_path, sample.input_payload,
                      sample.graph, weights, sample.workmem,
                      self_proof_fn=spf)

    def teacher_gt(sample: CalibrationSample) -> int:
        if teacher is None:
            return 1   # 无教师→默认 GT pass（标定退化为默认权重·oracle 验后调）
        return teacher.judge_ground_truth(sample.output, sample.dag_path,
                                          sample.graph)

    return judge_fn, teacher_gt


# ---- RoundRunner 协议（可换·experiments 可换层） ----

@runtime_checkable
class RoundRunner(Protocol):
    """per-round 认知执行协议（observe + episode·可换预处理/检索）。

    run_round(ctx, item, stage, round_id) -> Episode | None
      observe-only 阶段（stage1/2）可返 None（无 episode·只建图）。
      reward 阶段（stage3+）返 Episode（供 metrics/防塌/收敛消费）。
    """

    def run_round(self, ctx: TrainContext, item: CollectedItem,
                  stage: int, round_id: int) -> Episode | None: ...


@dataclass
class RoundResult:
    """per-round 产出（episode + 重建 path/output 供 H2 标定·确定性重算 bit-identical）。"""

    episode: Episode | None = None
    output: Any = None
    dag_path: Any = None


class DefaultRoundRunner:
    """默认 per-round 执行（真接线：observe → episode_loop + build_judge_fn）。

    observe 阶段（stage1/2）：CollectedItem 切多句段 → observe 建图·返 None（无 episode）。
    reward 阶段（stage3+）：observe + episode_loop（build_judge_fn 绑教师 GT·weaning pre）。
      CollectedItem 切多句段 observe 串 struct_ref 链（inter-segment PRECEDES·设计原意·破致命6）·
      seed=首 struct_ref / sink=末 struct_ref（≠seed·破 intent 退化致命1）·
      key_skeleton=struct_refs（J1 覆盖目标）。单段→跳过（emergent_role defer·bare-text 只骨架）。
    语言首版最小检索（全 edges·小图 OK·真热区裁剪 Stage 6+ defer）。
    run_round_full 额外返 (output, dag_path) 供 H2 标定（确定性重算·同 episode 内部 bit-identical）。
    """

    def run_round(self, ctx: TrainContext, item: CollectedItem,
                  stage: int, round_id: int) -> Episode | None:
        return self.run_round_full(ctx, item, stage, round_id).episode

    def run_round_full(self, ctx: TrainContext, item: CollectedItem,
                       stage: int, round_id: int) -> RoundResult:
        assert_int(stage, round_id, _where="DefaultRoundRunner.run_round_full")
        # 多段 observe：CollectedItem 段落 → 句段 Segment 列表·串 struct_ref 链（设计原意·破致命6）
        # 刀5 件8：透传 ctx 4 参 → cue_extractor → cue_type_of 第二源 D:11 readback（生产路径·close 刀4 gap）
        segments = _split_item_to_segments(
            item, backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index)
        if not segments:
            return RoundResult()
        raw = InputPayload(
            segments=segments, source=item.source, stage=STAGE_TRAINING,
            modality=item.modality, lang=item.lang, domain=item.domain,
            weaning_phase=ctx.weaning_phase,
            item_key=id(item),
        )
        # observe 建图（卷一 observe 总控·三空间分流·回传 struct_refs）
        sctx = _build_space_ctx(ctx)
        # B5：pronoun_feature_lookup 注入（元定义出厂硬件·性质B 软兜·防 PR 软排序把"他"指向"苹果"非人称）
        from pure_integer_ai.cognition.understanding.pronoun_features import lookup_pronoun_features
        obs = observe(raw, sctx, concept_index=ctx.concept_index,
                      work_memory=ctx.work_memory,
                      pronoun_feature_lookup=lookup_pronoun_features,
                      sense_lookup=make_sense_lookup(ctx.backend, ctx.space_id))

        # #730 路径 W：捕获 code item 的 observe-built COMPOSES 根（__prog_* = obs.struct_refs[0]）供 task-driven
        # 代码模态 unparse（候选 A·observe 期建树一次·task-driven 纯读·幂等守 bit-identical）。code struct_ref
        # 确定性（=__prog_{stage}_{h63(code_source)}·跨 round 稳定·observe guard 防重 build）·每 round 重设幂等。
        # 非 code 模态 / observe 未建 struct_ref → 不设（保 None·task-driven 跳过）。observe-only 阶段也捕获（树
        # 在 stage1 已建·struct_ref 可用·无副作用）。
        if item.modality == MODALITY_CODE and obs.struct_refs:
            item.code_struct_ref = obs.struct_refs[0]

        # 断桥 Phase B 片1 数据桥（doc/重来_断桥设计refinement_2026-07-15 §Phase B 片1）：flatten raw.segments[*].numeric_claims
        # → CollectedItem.numeric_claims_flat·供 _run_task_driven_generate Phase B block 读（mirror :386 code_struct_ref
        # 捕获范式·observe 期填一次·task-driven 纯读·幂等守 bit-identical）。ungated 纯缓存（NUMERIC_PROOF_MODE OFF→
        # seg.numeric_claims 空→flat 空→Phase B 不进→bit-identical·数据已存 segments·拷贝到 CollectedItem 零行为变）。
        # 每 round 重设幂等（numeric_claims observe 期确定性重填·flatten 序确定·bit-identical）。
        if raw.segments:
            item.numeric_claims_flat = tuple(
                c for seg in raw.segments if seg.numeric_claims for c in seg.numeric_claims)

        if stage < STAGE3_REWARD:
            return RoundResult()   # observe only 阶段·无 episode

        # 代码域/算术域独立 episode 路径（verify-driven COMPOSES·doc/重来_A3_代码域observe设计补充.md
        # §二致命#2 + doc/重来_算术域observe设计补充.md §九）：vm_proof_fn 直调绕 judge/generate/propagate
        # （judge G5 路由坏：G2p 在 not reached_sink veto + J-sum 恒 0·verify 模态无词生成/无 key_skeleton/无 CAUSES）。
        if _is_verify_modality(item.modality):
            return self._run_verify_round(ctx, item, raw, obs, round_id)

        # 刀 B：语言域数值等式 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 A 时序
        # 但验数值等式声明算术一致·直接整数算术·非 PRECEDES DAG·非 COMPOSES 执行·构造性检查层·永不接 reward）。
        # 数值声明不入图（Option A·闭包传·同刀 A 防污染）·gate OFF 路由不走 bit-identical。
        # **numeric priority over precedes**：同 item 含数值+时序 cue 时走数值分支（首版诚实 scope·混合罕见·
        # documented·两 gate 独立·bit-identical OFF 不影响）。构造性检查 SELF_PRODUCED（数 single-source·Layer0）。
        if (item.modality == MODALITY_LANGUAGE
                and getattr(gates, "NUMERIC_PROOF_MODE", False)
                and any(seg.numeric_claims for seg in segments)):
            return self._run_numeric_verify_round(ctx, item, raw, obs, round_id)

        # 刀 D：语言域比较 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 A/B/C
        # 但验比较声明算术序·cross_compare 交叉积·非整数等式算术·非 PRECEDES DAG·非 COMPOSES 执行·
        # 构造性检查层 SELF_PRODUCED·第 4 个 LIVE form_proof_fn·给 cross_compare 首个真比较消费者）。
        # 比较声明不入图（Option A·闭包传·同刀 A/B 防污染）·gate OFF 路由不走 bit-identical。
        # **numeric>comparison>universal>precedes 序**（同 item 多形式 cue 时首版走 numeric·documented·
        # 两 gate 独立·比较邻数值同属算术形式域·bit-identical OFF 不影响）。构造性检查 SELF_PRODUCED（数 single-source·Layer0）。
        if (item.modality == MODALITY_LANGUAGE
                and getattr(gates, "COMPARISON_PROOF_MODE", False)
                and any(seg.comparison_claims for seg in segments)):
            return self._run_comparison_verify_round(ctx, item, raw, obs, round_id)

        # 刀 C：语言域全称量化 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 A/B
        # 但验全称量化内涵分类子集 X⊆Y·ConceptNet 外部祖先图·**构造性验证层·首个 EXTERNAL**·刀 A/B 是检查
        # SELF_PRODUCED·刀 C 升验证·Layer0 external_verified 首个语言域计入）。IS_A 须 ConceptNet 外部源
        # （build_isa_ancestor_map_external·source filter·非 cue 自产·反 single-source theater）·gate OFF 路由不走
        # bit-identical。**numeric>universal>precedes 序**（同 item 多形式 cue 时首版走 numeric·documented·两 gate 独立）。
        # 三值逻辑：verified→reward=1 EXTERNAL / falsified→reward=0 EXTERNAL / can't-verify→None 弃权无 episode
        # （守属性全称 G5b #479 墙·child/parent 非分类概念诚实降级·非 theater）。
        if (item.modality == MODALITY_LANGUAGE
                and getattr(gates, "UNIVERSAL_PROOF_MODE", False)
                and any(seg.universal_claims for seg in segments)):
            return self._run_universal_verify_round(ctx, item, raw, obs, round_id)

        # A1·STEP6：语言域存在量化 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 C
        # 但验存在量化 X∩Y≠∅·**双向祖先** X⊆Y OR Y⊆X·ConceptNet 外部祖先图·构造性验证层 EXTERNAL·同刀 C）。
        # 存在量化声明不入图（Option A·闭包传·同刀 C 防污染）·gate OFF 路由不走 bit-identical。
        # **numeric>comparison>universal>existential>precedes 序**（同 item 多形式 cue 时首版走 numeric·documented·
        # 两 gate 独立·bit-identical OFF 不影响）。三值逻辑：verified→reward=1 EXTERNAL / falsified→reward=0
        # EXTERNAL / can't-verify→None 弃权无 episode（守属性 ∃ #479 墙·双向皆不命中+两分类才 falsified）。
        if (item.modality == MODALITY_LANGUAGE
                and getattr(gates, "EXISTENTIAL_PROOF_MODE", False)
                and any(seg.existential_claims for seg in segments)):
            return self._run_existential_verify_round(ctx, item, raw, obs, round_id)

        # 刀 A：语言域时序 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像 _run_verify_round
        # 但验 PRECEDES DAG Kahn 无环·非 COMPOSES 执行·构造性检查层·永不接 reward·Layer0 下 session升验证）。
        # 语言域 G5=DEAD_DESIGN（judge.py:41 _ARITH_DOMAINS 不含语言域）→ 走独立 episode 绕 judge·非挂 G5。
        # 时序 cue 对不入图（Option A·闭包传·防 #355 provenance 冲突 + emergence 污染）·gate OFF 路由不走 bit-identical。
        # **设计取舍（对抗审 P1-3）**：路由分流致整语言 episode 绕 episode_loop·混合因果+时序项的 CAUSES 边（observe 建）
        # 不获 reward 强化（propagate 不跑·边仍存·sn/tn/strength 不涨）。刀 A 接受（诚实基建首刀·时序 verify 优先·
        # CAUSES 强化靠非时序项）·Layer0 session 复评（届时时序边入图 + R6·CAUSES 强化路径重审）。
        if (item.modality == MODALITY_LANGUAGE
                and getattr(gates, "TIME_SEQ_PROOF_MODE", False)
                and any(seg.precedes_pairs for seg in segments)):
            return self._run_time_seq_verify_round(ctx, item, raw, obs, round_id)

        # 刀4 子环1+2：涌现假设生成 + D:11 SHADOW 落边（gate HYPOTHESIS_MODE·reward 阶段 observe 后
        # episode 前·D:11 SHADOW 边须先存在·下方 episode_loop→propagate_reward concept_targets 扩展才能
        # feed 这些候选词·子环3 鸡生蛋破解）。gate OFF（默认）跳过·CI 回归零翻（镜像 ATTRACTOR/EXPLORATION 范式）。
        # **对应泛化 v2（审2条件3·两 gate 共存）**：ORACLE_PROMOTE_MODE ON 时跳 generator（结构反推 tally 轨
        # 替代——tally_cue_slot_matches 在 recognize 后建 SHADOW·D:11 只认 _structure_match_ok·generator 的
        # PRECEDES-connector 浅共现"和"误产 REL_CAUSES 病源退场）。OFF 时既有 HYPOTHESIS_MODE 路径照旧（bit-identical）。
        if (getattr(gates, "EMERGENT_RELATION_HYPOTHESIS_MODE", False)
                and not getattr(gates, "ORACLE_PROMOTE_MODE", False)):
            _run_emergence_hook(ctx, item.lang)

        # reward 阶段：struct_ref 链 = episode 遍历目标（多段 observe 串 inter-segment PRECEDES·设计原意）
        # seed=首 struct_ref / sink=末 struct_ref（≠seed·破 intent 退化致命1）·
        # key_skeleton=struct_refs（J1 覆盖目标·破 J1 恒 0）。单段 struct_ref 孤立→产不出 part→跳过（诚实·emergent_role defer）
        struct_refs = obs.struct_refs
        if len(struct_refs) < 2:
            return RoundResult()
        seed = struct_refs[0]
        sink = struct_refs[-1]
        # ★M1片2（doc/重来_M1片2_intent分类设计_2026-07-08.md）：classify_intent 替换硬编码
        # INTENT_QUESTION（gate M1_INTENT_CLASSIFY_MODE·生产 try/finally 翻 ON 见 :1238 段）。is_causal_reasoning
        # = _has_causes_signal(raw.segments)（与 causes.py:38-51 建边同源·解 judge.py:224 G3a 死门·j3path 从永 0
        # 到加权）。type 默认 INTENT_QUESTION / COMMAND（W7 doc §15：子 gate INTENT_COMMAND_MODE ON 且 mood 命中
        # →COMMAND·dag_path:302 早已 tuple 含 COMMAND·Q/C 等价·STATEMENT(3) 才 DEAD_END·零行为差异）·
        # sink 透传 struct_refs[-1]（选项 B·维持 reward 通路）·is_structural/has_value_claim 维持 False
        # （语言域设计正确 / theater defer #774）。gate OFF 走原硬编码（三 bool 永 False·G3a/G3b dead·bit-identical）。
        if getattr(gates, "M1_INTENT_CLASSIFY_MODE", False):
            from pure_integer_ai.cognition.understanding.intent_classify import classify_intent
            intent = classify_intent(
                sink, raw.segments,
                backend=ctx.backend, edge_store=ctx.edge_store,
                space_id=ctx.space_id, concept_index=ctx.concept_index)
        else:
            intent = IntentType(type=INTENT_QUESTION, sink=sink)
        raw.intent = intent
        raw.key_skeleton = list(struct_refs)
        # perf round5：dag_path 仅消费 {PRECEDES, CAUSES, T_STEP}（dag_path_step 8 消费者全核证·见 a2_stepper/
        # a3_pr_wrapper/attractor/dead_end）·select(where=None) 全边 dict-copy（n=656 ~195K 行·~90% COOCCURS 死载
        # copy 后 dag_path 零读）→ 按型三 select 仅 copy 需要的 ~20K。COOCCURS/SIMILAR/IS_A/D:11/REFERS_TO(non-occ)
        # 从 subgraph_edges 零消费（生成侧 collide/selection_pref 读 backend 非 dag_edges）。
        # bit-identical by construction（always-on·同 ancestor_map cache 纯 perf 范式·非 gate）：dag_path 输出与
        # 边迭代序无关——Kahn 自然序队列 + _build_pred_index 排序副本 + convergence 计数 + build_matrix 可交换
        # 累加(acc[iu][iv]+=w·sorted col) + _build_in_degree_seq_map 计数。select 保插入序→各型子表内序不变→
        # _build_pred_index 同序 double-safe。T_STEP 必含（attractor._build_in_degree_seq_map 计 {T_STEP,PRECEDES}）。
        # perf round6 hotzone（gate DAG_PATH_REACHABLE_SCOPE_MODE·设计主线 line 530/978/1224 热区子图非全图）：
        # gate ON → dag_edges 在 episode_loop 前（seed 已知）由 _hotzone_dag_edges query_from k-hop 算·此处跳过全 select。
        # gate OFF → 全 dag_edges select（既有 bit-identical·round5 edge filter 注释见上）。
        dag_edges: list | None = None
        if not getattr(gates, "DAG_PATH_REACHABLE_SCOPE_MODE", False):
            dag_edges = (ctx.backend.select("edge", where={"edge_type": EDGE_PRECEDES})
                         + ctx.backend.select("edge", where={"edge_type": EDGE_CAUSES})
                         + ctx.backend.select("edge", where={"edge_type": EDGE_T_STEP}))
        generate_fn = lambda pr, w, i: generate_output(
            pr, ctx.concept_graph, w, item.lang)
        judge_fn = build_judge_fn(ctx.concept_graph, ctx.weights,
                                  teacher=ctx.teacher,
                                  weaning_phase=ctx.weaning_phase,
                                  judge_source_id=ctx.judge_source_id)
        ctx.judge_source_independent = judge_fn.judge_source_independent   # W3 路径 B :2018 读此（连通死属性·单一真相源·默认 False 守 bit-identical）
        # item3 缺漏4：reward 阶段置 ATTRACTOR_MODE ON（反馈腿输出半边闭合·attractor 扩张 e 响应 reward）
        # 默认 OFF 守单测 bit-identical·生产训练 reward 阶段 ON·try/finally 守回归
        # A2：reward 阶段同置 EXPLORATION_MODE ON（防塌柱③ proactive probe·dag_path:96 PR 方差趋平时注入新种子
        # 解 anti_collapse 柱③ EXPLORATION 生产永关·方差趋平时柱③ 失守）。同 ATTRACTOR 范式 try/finally 守回归。
        saved_attractor = gates.ATTRACTOR_MODE
        saved_exploration = gates.EXPLORATION_MODE
        saved_sp_dock = gates.SELECTION_PREF_DOCK_MODE
        saved_sp_feed = gates.SELECTION_PREF_FEED_MODE
        saved_sp_gen = gates.GENERATE_SELECTION_PREF_MODE
        saved_replay = gates.MEMORY_REPLAY_MODE
        saved_freq_observe = gates.FREQ_OBSERVE_MODE
        saved_sp_observe = gates.SP_OBSERVE_MODE
        gates.ATTRACTOR_MODE = True
        gates.EXPLORATION_MODE = True
        # S4 三乘子进 PR：selection_pref 维 dock PR seed（_seed_weight 乘积·attractor 扩张路径 token seed 真生效）
        # + sp_sn reward feed 第三条腿（reward_propagate 落点⑥·concept_targets 配对 feed）。同 ATTRACTOR try/finally 守回归。
        gates.SELECTION_PREF_DOCK_MODE = True
        gates.SELECTION_PREF_FEED_MODE = True
        # S4 后续加固·项1：生成侧 selection_pref pair-rate 精查接线（GENERATE_SELECTION_PREF_MODE 生产 ON）。
        # observe+reward 写 selection_pref_count·生成侧 slot_dispatch:105 读 selection_pref_score·gate 不接则写了不读=theater。
        # 同 ATTRACTOR try/finally 守回归（CI gate OFF 零翻·生产 ON 生成侧真活）。
        gates.GENERATE_SELECTION_PREF_MODE = True
        # #728 输出侧 memory_space 融通：MEMORY_REPLAY_MODE 生产 ON（tri_space caller 接线 + dag_path local_seeds 扩张）。
        # episode_loop 末尾调 tri_space_coordination（gate ON + memory_read=ctx.memory_read 传入 → query memory → 写
        # workmem.replay·info_ref concept ref·每 episode 清 fresh）→ 下 episode dag_path local_seeds 扩张 replay_candidates。
        # CI gate OFF → tri_space early-return → workmem.replay 永空 → dag_path local_seeds == seeds → bit-identical。
        # 同 ATTRACTOR try/finally 守回归。verify round 不调 episode_loop 不翻（_run_verify_round 绕 episode_loop）。
        gates.MEMORY_REPLAY_MODE = True
        # 方案3 tn路（B4 β_arith 修法）：FREQ_OBSERVE_MODE 生产 ON（dag_path add_active + attractor add_seed 写
        # observe_tn·read_effective_freq observe_mode=True 读 base+observe_tn·解 β_arith rate 塌缩 w_freq 塌缩）。
        # 同 ATTRACTOR try/finally 守回归（CI gate OFF 零翻·生产 ON observe_tn 真写·否则 = theater）。
        gates.FREQ_OBSERVE_MODE = True
        # 方案3 tn路（B5 β_arith 修法）：SP_OBSERVE_MODE 生产 ON（selection_pref 维 consumer 读 sp_observe_tn 替 sp_tn·
        # 解 β_arith rate 塌缩 w_sp 塌缩·sp_observe_tn 由 record_selection_pref_cooccur 写·SELECTION_PREF_MODE 守写）。
        # 同 FREQ_OBSERVE_MODE try/finally 守回归（CI gate OFF 零翻·生产 ON sp_observe_tn 真读·否则 = theater）。
        gates.SP_OBSERVE_MODE = True
        # perf round6 hotzone（DAG_PATH_REACHABLE_SCOPE_MODE）：**env 控制·非 try/finally 翻**——热区改 dag_path 行为
        # （非 bit-identical）·try/finally 强翻会破测试。断奶/perf 跑设 ZERO_AI_DAG_PATH_REACHABLE_SCOPE_MODE=1 + ZERO_AI_DAG_HOPS=2
        # （_flag gate env 读·默 2 hop）。CI/测试 unset=bit-identical（全 dag_edges·既有）。
        try:
            # 项1 cache invalidate：observe 增 IS_A 后清 ancestor_map cache·生成侧 selection_pref_score 重建含新 IS_A。
            # 生产 ConceptGraph 是 ctx 单例·跨 round reuse·不清则 cache 单调陈旧（漏新祖先→under-boost）。
            ctx.concept_graph.invalidate_ancestor_map()
            # —— B-PR4 动作词种子偏向预算（doc §19·episode_loop 前填 workmem.action_seed_candidates·dag_path 入口读） ——
            # gate ACTION_SEED_BIAS_MODE 时预算（_collect_action_seed_candidates 守 intent==INTENT_COMMAND·mirror B-PR2
            # caller 守 gate / helper 守 intent 范式）：扫 segments 动作词 D:11 PRIMARY → read_experience_count 洗净 sn==0
            # 滤除 + rate-sort 降序 → 写 workmem。dag_path_step 入口读 + subgraph_nodes 过滤 + append local_seeds/e_set
            # （mirror #728 replay 扩张·PR 偏向动作拓扑）。
            # gate OFF → 写 []（fresh each round·无 stale 携带）→ dag_path `if candidates:` 假 → 跳过 → bit-identical。
            # gate ON + QUESTION → helper intent 守返 [] → 同 OFF（bit-identical·intent 守可单测 test_collect_question_no_budget）。
            # ctx_code 同 episode_loop :82 / B-PR2 写桶（pack_ctx_code(domain,modality,0,intent)·COMMAND 桶读 COMMAND 桶写）。
            if getattr(gates, "ACTION_SEED_BIAS_MODE", False):
                ctx.work_memory.action_seed_candidates = _collect_action_seed_candidates(
                    segments=raw.segments, backend=ctx.backend, edge_store=ctx.edge_store,
                    space_id=ctx.space_id, concept_index=ctx.concept_index,
                    intent_type=intent.type,
                    ctx_code=pack_ctx_code(raw.domain, raw.modality, 0, intent.type))
            else:
                ctx.work_memory.action_seed_candidates = []   # fresh each round（gate OFF → 空 → bit-identical）
            # perf round6：dag_edges scope 到 seed k-hop 热区子图（gate DAG_PATH_REACHABLE_SCOPE_MODE·O(n²)→O(n) 杠杆）。
            # gate OFF → 全 dag_edges（既有 bit-identical）。gate ON → _hotzone_dag_edges query_from k-hop（ZERO_AI_DAG_HOPS
            # =k·默 2·镜像 PR HOTZONE·设计主线 line 530/978 热区非全图）·O(k-hop 邻域) 非 O(全累积)·保 H4 闭环。
            if getattr(gates, "DAG_PATH_REACHABLE_SCOPE_MODE", False):
                _h = os.environ.get("ZERO_AI_DAG_HOPS")
                _hops = int(_h) if _h else 2
                dag_edges = _hotzone_dag_edges(ctx.edge_store, [seed], max_hops=_hops)
            _scope_edges = dag_edges
            out, ep = episode_loop(
                raw, _scope_edges, [seed], ctx.work_memory, intent,
                generate_fn=generate_fn, judge_fn=judge_fn,
                edge_store=ctx.edge_store, backend=ctx.backend,
                current_seq=round_id,
                memory_active=False,
                coverage_threshold=COVERAGE_THRESHOLD,
                memory_read=ctx.memory_read,
            )
            # 重建 path/output 供 H2 标定（确定性重算·同 episode 内部 bit-identical·attractor ON 下重算一致）。
            # S4 片4：ctx_code 同 episode_loop :82 算法（_ctx_tag + pack_ctx_code）·H2 重算须同桶（freq/sp dock
            # attractor 扩张路径 token seed 读 ctx 桶·不同桶 bit-identical 失·stage8 latent 修）。
            _h2_ctx_code = pack_ctx_code(*_ctx_tag(raw, intent))
            path_result = _rebuild_path(ctx, _scope_edges, [seed], intent, round_id,
                                        key_skeleton=raw.key_skeleton,
                                        coverage_threshold=COVERAGE_THRESHOLD,
                                        ctx_code=_h2_ctx_code)
            output = generate_fn(path_result, ctx.work_memory, raw)
        finally:
            gates.ATTRACTOR_MODE = saved_attractor
            gates.EXPLORATION_MODE = saved_exploration
            gates.SELECTION_PREF_DOCK_MODE = saved_sp_dock
            gates.SELECTION_PREF_FEED_MODE = saved_sp_feed
            gates.GENERATE_SELECTION_PREF_MODE = saved_sp_gen
            gates.MEMORY_REPLAY_MODE = saved_replay
            gates.FREQ_OBSERVE_MODE = saved_freq_observe
            gates.SP_OBSERVE_MODE = saved_sp_observe
        # —— B-PR2 动作意图经验回写（doc §17·ACTION_* concept 动作验证率·对偶 op_confidence·经验层第三层） ——
        # D3（type==INTENT_COMMAND + terminal==REACHED_SINK）激活 → collect_action_intent_concepts 扫 segments D:11 PRIMARY
        # → distinct ACTION_* refs → record_experience_outcome 写 experience_count（reward 驱动 R1·ctx_code 自动落 COMMAND 桶）。
        # reward>0 = R1 成功臂（e_sn++&e_tn++）非排除闸·reward==0 veto→e_tn++ only→率<1 有判别力（设计审 B CONFIRMED·硬排除→率恒1 β_arith 病）。
        # gate OFF → helper 早返→experience_count 零新增→bit-identical。依赖 INTENT_COMMAND_MODE（OFF→type 永 QUESTION→D3 永假→零写）。
        if getattr(gates, "ACTION_EXPERIENCE_FEED_MODE", False):
            _feed_action_experience(
                backend=ctx.backend, edge_store=ctx.edge_store,
                space_id=ctx.space_id, concept_index=ctx.concept_index,
                segments=raw.segments, domain=raw.domain, modality=raw.modality,
                intent_type=intent.type, reward=ep.reward, terminal=ep.terminal)
        return RoundResult(episode=ep, output=output, dag_path=path_result)

    def _run_verify_round(self, ctx: TrainContext, item: CollectedItem,
                          raw: InputPayload, obs: Any,
                          round_id: int) -> RoundResult:
        """verify-driven COMPOSES 独立 episode 路径（代码域 C6 PRE + 算术域·doc/重来_算术域observe设计补充.md §九）。

        observe 已建 COMPOSES 树（root=struct_ref·obs.struct_refs[0]）·本方法只做验证：
          PRE：逐 spec vm_proof_fn 执行学生 COMPOSES vs 独立 expected → reward = 1 iff 全 spec pass。
          POST（Mode B defer）：reward=0·不调 vm_proof（防 self_proof_check(POST,None)→1 vacate theater）。

        直调 vm_proof_fn·不经 self_proof_check（verify 模态里 vm_proof 是整个 reward 非 G5 一门·避免 None 3 态路由耦合）。
        vm_proof_fn 只读 dag_path.sink·minimal PathResult(sink=root, terminal=REACHED_SINK) 够（不跑 dag_path_step·verify 模态无 PRECEDES/CAUSES 链可遍历）。
        reward 不落边 strength（verify propagate 永久 no-op·COMPOSES 边 inert·架构真差异·doc §4.5）·信号进 Episode（metrics conduction_rate + 反 theater 锚点）。

        specs 按模态选：CODE→code_specs / ARITH→arith_specs（vm_proof_fn modality-agnostic·读 COMPOSES 树执行不问来源）。
        诚实边界：无 spec 不能验证→返空 RoundResult（observe-only·不伪造 reward=0）。Mode B re-derivation defer。
        """
        assert_int(round_id, _where="_run_verify_round.round_id")
        struct_refs = obs.struct_refs
        if not struct_refs:
            return RoundResult()   # observe 未建 struct_ref（code_source/arith_source 空 etc）·诚实跳过
        root = struct_refs[0]   # 单函数/单记号=单 struct_ref=COMPOSES 根（observe MODALITY_CODE/ARITH gate）
        specs = item.code_specs if item.modality == MODALITY_CODE else item.arith_specs
        if not specs:
            return RoundResult()   # 无 spec 不能验证·诚实 observe-only·不伪造 reward=0
        # minimal dag_path：vm_proof_fn 只读 sink·代码域无链可遍历不跑 dag_path_step
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 代码域无词生成（代码是输入非生成·generate defer）
        weaning_pre = (ctx.weaning_phase == WEANING_PRE)
        if weaning_pre:
            if getattr(gates, "VALUE_SYNTHESIZE_MODE", False):
                # 相1 G-PR1 算术归纳合成（doc §二十）：行为匹配搜索骨架池·非构造性 verify。
                # synthesize_value 内部 execute_composes_value 验全 specs·命中=归纳合成成功（跨 item 骨架=泛化信号）。
                # 跳过 vm_proof_fn 循环（synthesize_value 已行为匹配·非冗余双执行）。
                # sink 重指搜索产物（§20.0 核证：caller 设 sink·dag_path.py 不设·不违 observe 契约）。
                from pure_integer_ai.training.value_synthesize import synthesize_value
                _pool = load_discovered_operators(ctx.backend, space_id=ctx.space_id)
                _matches = synthesize_value(ctx.concept_graph, _pool, specs)
                if _matches:
                    _synth_root = _matches[0][0]   # skeleton_ref（pool 升序·首命中 bit-identical）
                    dag_path = PathResult(
                        path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=_synth_root,
                        topo_layers=[], convergence={}, source=None)
                    root = _synth_root
                    reward = 1   # 归纳合成成功（行为匹配全 specs）
                else:
                    root = struct_refs[0]   # ref 有效性 fallback（合成无匹配）
                    reward = 0   # DISAGREE 牙：pool 无行为匹配·诚实 0（非伪造）
                g5_active = True   # PRE 有 spec·G5 active（合成成功=reward=1 / 无匹配=veto=reward=0）
            else:
                # PRE（Mode A）：逐 spec 直调 vm_proof_fn·all-pass 才 reward=1（None→0·deadloop/mismatch 诚实 0）
                all_pass = True
                for spec in specs:
                    fn = vm_proof_fn_factory(input_args=spec.input_args,
                                             expected=spec.expected)
                    r = fn(output, dag_path, ctx.concept_graph)
                    if r != 1:   # 0 mismatch / None deadloop（R1 PRE→0 诚实·非 vacate）
                        all_pass = False
                        break   # 任一 fail 即 reward=0（短路·确定性·后续 spec 不跑）
                reward = 1 if all_pass else 0
                g5_active = True   # PRE 有 spec·G5 承重门 active（pass=reward=1 / fail=veto=reward=0）
        else:
            # POST（Mode B）：教师退场无 expected 独立源·cross-verify（异算法统计一致·#479 加强腿）激活时
            # 两路独立 execute_composes_value + rational.eq（Mechanism Y·无 oracle·"另一棵树"即独立源·POST 可用）·
            # 否则 reward=0（防 vacuous reward=1 theater）。gate OFF + source_b 缺 → 双 False 短路 bit-identical。
            # **模态对称**（§施工序 1.2）：ARITH 用 arith_source_b / build_composes_from_arith · CODE 用 code_source_b /
            # build_composes_from_source · execute_composes_value + rational.eq 模态无关（两域都返 Rational）·corpus-agnostic。
            # 用户哲学重定向（2026-07-06）：不追求 correctness 真墙（Rice·须墙外 #478/#493）·只求统计学内一致——
            # cross-verify 是 on-target 的统计一致性机制（agreement 非 identity·声称守·doc/重来_ModeB自洽设计补充.md §七）。
            from pure_integer_ai.storage.edge_store import SOURCE_MATH, SOURCE_CODE
            if item.modality == MODALITY_ARITH:
                source_b = item.arith_source_b
                specs_b = item.arith_specs
                hash_seed = 'xver.b.v1'
                source_tag = SOURCE_MATH
            else:   # MODALITY_CODE（_is_verify_modality 守只 CODE/ARITH 进 _run_verify_round）
                source_b = item.code_source_b
                specs_b = item.code_specs
                hash_seed = 'xver.b.code.v1'
                source_tag = SOURCE_CODE
            if gates.MODE_B_CROSS_VERIFY_MODE and source_b:
                from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
                from pure_integer_ai.cognition.understanding.code_observe import build_composes_from_source
                from pure_integer_ai.crosscut.determinism.hasher import Hasher
                from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
                from pure_integer_ai.storage.node_store import NODE_CONCEPT
                # 参树 builder 按模态选（ARITH arith_observe / CODE code_observe·corpus-agnostic·两 builder 签名同构）
                build_fn = (build_composes_from_arith if item.modality == MODALITY_ARITH
                            else build_composes_from_source)
                # 参树 root_b：内容哈希独立根（hash_seed 按模态分·ARITH 'xver.b.v1' / CODE 'xver.b.code.v1'·
                # 避 observe hash 空间纠缠 + 模态间参树 hash 空间分离·漏洞 3 修）·同 space
                h_b = Hasher(hash_seed).h63(source_b)
                root_b = ctx.concept_index.ensure(
                    f"__xver_b_{h_b}", space_id=ctx.space_id,
                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                # 幂等：已建有 COMPOSES 出边 → skip（EdgeStore.add 不去重·防 H2 重算/resume 重 build 复制边 corrupt 树·同 :1477 范式·漏洞 4 修）
                if not ctx.edge_store.query_from(root_b[0], root_b[1], edge_type=EDGE_COMPOSES):
                    build_fn(source_b, concept_index=ctx.concept_index,
                             edge_store=ctx.edge_store, backend=ctx.backend,
                             space_id=ctx.space_id, source=source_tag, root_ref=root_b)
                # 探针 = spec.input_args（复用既有测试输入·丢 expected·避教师 oracle·守 #479 不破·按模态 specs）
                probes = tuple(spec.input_args for spec in specs_b)
                cv = cross_verify_pair(ctx.concept_graph, root, root_b, probes)
                reward = 1 if cv.all_agree else 0
                g5_active = True
            else:
                reward = 0
                g5_active = False
        ep = Episode(
            episode_id=round_id,
            run_id=round_id,
            input=raw,
            output=output,
            reward=reward,
            ref=root,
            terminal=TERMINAL_REACHED_SINK,
            pr_vector={},   # 代码域不跑 dag_path_step·无 PR 向量（anti_collapse 跳过空 pr_vector·诚实）
            judge_G5_active=g5_active,
            judge_veto_count=1 if reward == 0 else 0,
            dead_end_count=0,
            vetoed=(reward == 0),
            verify_source=VERIFY_SOURCE_EXTERNAL,   # Layer0 外部锚门：verify 通道用 R6 外部源（PRE: spec.expected / POST: source_b）·POST gate-off 时 reward=0 占位未真验·EXTERNAL 标通道源类（reward=0 不计 external_verified·无害·2 审 P2）
        )
        return RoundResult(episode=ep, output=output, dag_path=dag_path)

    def _run_time_seq_verify_round(self, ctx: TrainContext, item: CollectedItem,
                                   raw: InputPayload, obs: Any,
                                   round_id: int) -> RoundResult:
        """刀 A：语言域时序 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像 _run_verify_round :446-550）。

        验 **PRECEDES DAG Kahn 无环**（构造性检查）·非 COMPOSES 执行值。语言域 G5=DEAD_DESIGN（judge.py:41
        _ARITH_DOMAINS 不含语言域）→ 走独立 episode 绕 judge（镜像 _run_verify_round）·非挂 G5。reward=1 iff
        Kahn 无环·不落 strength（verify propagate no-op·镜像 :457）·**永不接 reward**（PRECEDES strength 恒 1）。

        时序 cue 对（segments.precedes_pairs·resolve 段内 token index → ConceptRef·跨 cue 词 shortcut A→B）+
        intra-space EDGE_PRECEDES 边集（role_precedes 三建边器建·token 序/句间序/struct 锚）→
        time_seq_proof_fn_factory 闭包 → Kahn 合并验序（cue 对 + PRECEDES 边合并集无环 = 时序一致）。

        **Option A**：时序 cue 对不入图（闭包传·防 #355 provenance 冲突 + emergence 污染）·持久化 defer Layer0 session。
        **构造性检查 ≠ 构造性验证**：Kahn 验 DAG 无环（确定性可执行）·非 R6 独立源验证·Layer0 下 session升验证。
        **stable≠correct**：DAG 无环 ≠ 语义时序正确（#479 墙·语言命题无执行值）。
        """
        assert_int(round_id, _where="_run_time_seq_verify_round.round_id")
        segments = raw.segments
        struct_refs = obs.struct_refs
        if not struct_refs:
            return RoundResult()   # observe 未建 struct_ref·诚实跳过（不伪造 reward）
        root = struct_refs[0]
        space_id = ctx.space_id
        # 1. resolve 时序 cue 对（段内 token index → ConceptRef·concept_index.lookup·未概念化/自环跳·守反统计）
        cue_pair_edges: list = []
        for seg in segments:
            if not seg.precedes_pairs:
                continue
            for (i, j) in seg.precedes_pairs:
                if i >= len(seg.tokens) or j >= len(seg.tokens):
                    continue
                a = ctx.concept_index.lookup(seg.tokens[i], space_id)
                b = ctx.concept_index.lookup(seg.tokens[j], space_id)
                if a is None or b is None:
                    continue   # token 未概念化→跳（诚实·反统计·不凑配）
                if a == b:
                    continue   # 自环不计（PRECEDES 非自反·镜像 role_precedes:37）
                cue_pair_edges.append((a, b))
        if not cue_pair_edges:
            return RoundResult()   # 无可 resolve 时序 cue 对·诚实 observe-only·不伪造 reward=0
        # 2. query intra-space EDGE_PRECEDES 边集（镜像 emergent_relation_signal.py:122-132·role_precedes 建的 token 序/句间序/struct 锚）
        # **累积跨项**（对抗审 P2-2）：backend.select 返本 space 全部 EDGE_PRECEDES（跨所有 observe 项·非仅当前项）·
        # Kahn 验全局时序一致性·reward 可能受先前项 PRECEDES 影响（feature 非 bug·全局时序矛盾即矛盾）。
        precedes_edges: list = []
        try:
            rows = ctx.backend.select("edge", where={"edge_type": EDGE_PRECEDES})
        except KeyError:
            rows = []   # edge 表未注册（bare fixture）·向后兼容
        for r in rows:
            if r["space_id_from"] != space_id or r["space_id_to"] != space_id:
                continue   # 仅本 space（镜像 emergent intra-space 过滤·防跨 space struct_ref 干扰）
            a = (r["space_id_from"], r["local_id_from"])
            b = (r["space_id_to"], r["local_id_to"])
            precedes_edges.append((a, b))
        # 3. factory 闭包 → fn → Kahn 合并验（cue 对 + PRECEDES 边·合并集无环 = 时序一致·构造性检查）
        fn = time_seq_proof_fn_factory(cue_pair_edges=cue_pair_edges, precedes_edges=precedes_edges)
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 时序 verify 无词生成（语言域 episode 不 generate·镜像 _run_verify_round :475）
        r = fn(output, dag_path, ctx.concept_graph)
        reward = 1 if r == 1 else 0   # 1 无环 / 0 有环 / None 边集空（路由守 cue 对非空·None 不应达·降级 reward=0）
        g5_active = True   # 时序 verify 承重门 active（pass=reward=1 / 有环=veto=reward=0·构造性检查层·诚实标）
        ep = Episode(
            episode_id=round_id,
            run_id=round_id,
            input=raw,
            output=output,
            reward=reward,
            ref=root,
            terminal=TERMINAL_REACHED_SINK,
            pr_vector={},   # 时序 verify 不跑 dag_path_step·无 PR 向量（镜像 _run_verify_round :544·anti_collapse 跳过空 pr_vector）
            judge_G5_active=g5_active,   # 字段重载：语言域 G5=DEAD_DESIGN·非真 G5·"verify 门 active"标志（对抗审 P2-1·anti_collapse skip 空 pr_vector 无害·capability_exam G 归因可能误计）
            judge_veto_count=1 if reward == 0 else 0,
            dead_end_count=0,
            vetoed=(reward == 0),
            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # Layer0 外部锚门：cue 对+token 序 single-source·构造性检查·非验证·全自产不准停
        )
        return RoundResult(episode=ep, output=output, dag_path=dag_path)

    def _run_numeric_verify_round(self, ctx: TrainContext, item: CollectedItem,
                                  raw: InputPayload, obs: Any,
                                  round_id: int) -> RoundResult:
        """刀 B：语言域数值等式 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像 _run_time_seq_verify_round）。

        验 **数值等式声明的算术一致**（构造性检查·直接整数算术）·非 PRECEDES DAG·非 COMPOSES 执行。
        语言域 G5=DEAD_DESIGN → 走独立 episode 绕 judge（镜像 _run_time_seq_verify_round / _run_verify_round）·非挂 G5。
        reward=1 iff 全数值声明算术一致·不落 strength（verify propagate no-op·镜像 _run_time_seq_verify_round）·
        **永不接 reward**（数值声明不入图·闭包传）。

        数值声明（segments.numeric_claims·extract_numeric_claims 已解析纯整数 4-tuple·self-contained）→
        numeric_proof_fn_factory 闭包 → 逐声明整数算术（ADD/SUB/MUL）算 left op right·比对 result_num。
        **多段语义**（对抗审 P2-2）：跨段 flatten 到单 claims list·**全段全声明一致才 reward=1**（任一段任一声明
        违反即 reward=0·短路·确定性·镜像 proof_fn "任一违反→0"）。**比时序简**：claims 已是纯整数·无需
        token→ConceptRef resolve（concept_index.lookup）·无需 backend query（EDGE_PRECEDES）·数值声明 self-contained
        （数+算子从 cue 窗口解析·非图边）。

        **Option A**：数值声明不入图（闭包传·同刀 A 时序边不入图·防结构发现污染 + emergence 干扰）。
        **构造性检查 ≠ 构造性验证**：左式/右式数 single-source（来自文本 cue 锚·非 R6 独立源）→ 非构造性验证·
        Layer0 标 SELF_PRODUCED（全自产不准停·同刀 A 时序定位）。
        **stable≠correct**：算术一致 ≠ 命题真（"3 加 5 等于 8" 算术对·文本是否真陈述此算式是语义层·#479 墙）。
        """
        assert_int(round_id, _where="_run_numeric_verify_round.round_id")
        segments = raw.segments
        struct_refs = obs.struct_refs
        if not struct_refs:
            return RoundResult()   # observe 未建 struct_ref·诚实跳过（不伪造 reward）
        root = struct_refs[0]
        # 1. 收集数值声明（已解析纯整数 4-tuple·self-contained·无需 resolve·镜像 _run_time_seq_verify_round cue_pair_edges 但更简）
        claims: list[tuple[int, int, int, int]] = []
        for seg in segments:
            if not seg.numeric_claims:
                continue
            # 拷贝防 alias（Segment.numeric_claims 是 list·闭包捕获前 flatten 到单 claims list）
            claims.extend(seg.numeric_claims)
        if not claims:
            return RoundResult()   # 无数值声明·诚实 observe-only·不伪造 reward=0
        # 2. factory 闭包 → fn → 逐声明整数算术验（claims 已纯整数·闭包传·不入图·镜像 time_seq_proof_fn_factory）
        fn = numeric_proof_fn_factory(claims=claims)
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 数值 verify 无词生成（语言域 episode 不 generate·镜像 _run_time_seq_verify_round :629）
        r = fn(output, dag_path, ctx.concept_graph)
        reward = 1 if r == 1 else 0   # 1 全声明一致 / 0 任一违反 / None claims 空（路由守非空·None 不应达·降级 reward=0）
        g5_active = True   # 数值 verify 承重门 active（pass=reward=1 / 违反=veto=reward=0·构造性检查层·诚实标·同 time_seq）
        ep = Episode(
            episode_id=round_id,
            run_id=round_id,
            input=raw,
            output=output,
            reward=reward,
            ref=root,
            terminal=TERMINAL_REACHED_SINK,
            pr_vector={},   # 数值 verify 不跑 dag_path_step·无 PR 向量（镜像 _run_verify_round / time_seq :641·anti_collapse skip 空 pr_vector）
            judge_G5_active=g5_active,   # 字段重载：语言域 G5=DEAD_DESIGN·非真 G5·"verify 门 active"标志（同 time_seq :642·capability_exam G 归因 time_seq→None 正确·Layer0 summary 处理）
            judge_veto_count=1 if reward == 0 else 0,
            dead_end_count=0,
            vetoed=(reward == 0),
            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # Layer0 外部锚门：左式/右式数 single-source·构造性检查·非验证·全自产不准停（同刀 A 时序）
        )
        return RoundResult(episode=ep, output=output, dag_path=dag_path)

    def _run_comparison_verify_round(self, ctx: TrainContext, item: CollectedItem,
                                     raw: InputPayload, obs: Any,
                                     round_id: int) -> RoundResult:
        """刀 D：语言域比较 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像 _run_numeric_verify_round）。

        验 **比较声明的算术序一致性**（构造性检查·cross_compare 交叉积·比序唯一零误差路径）·非整数等式算术·
        非 PRECEDES DAG·非 COMPOSES 执行。语言域 G5=DEAD_DESIGN → 走独立 episode 绕 judge（镜像 _run_numeric /
        time_seq_verify_round / _run_verify_round）·非挂 G5。reward=1 iff 全比较声明比序一致·不落 strength
        （verify propagate no-op·镜像 _run_numeric_verify_round）·**永不接 reward**（比较声明不入图·闭包传）。

        比较声明（segments.comparison_claims·extract_comparison_claims 已解析纯整数 3-tuple·self-contained）→
        comparison_proof_fn_factory 闭包 → 逐声明 cross_compare(left,1,right,1)=sign(left−right)·比对 cmp_opcode。
        **多段语义**（同 numeric P2-2）：跨段 flatten 到单 claims list·**全段全声明一致才 reward=1**（任一段任一声明
        违反即 reward=0·短路·确定性·镜像 numeric "任一违反→0"）。**比数值更简**：claims 已纯整数·无需
        token→ConceptRef resolve·无需 backend query·比较声明 self-contained（数+比较 OP 从 cue 窗口解析·非图边）。

        **第 4 个 LIVE form_proof_fn**（刀 A 时序 / 刀 B 数值 / 刀 C 量化 / 刀 D 比较）·给 cross_compare 首个真**比较**
        消费者（既有 1 caller 非比较用途·分层墙 §四缝1·反 theater：机制获真消费者）。

        **Option A**：比较声明不入图（闭包传·同刀 A/B·防结构发现污染 + emergence 干扰）。
        **构造性检查 ≠ 构造性验证**：左/右式数 single-source（来自文本 cue 锚·非 R6 独立源）→ 非构造性验证·
        Layer0 标 SELF_PRODUCED（全自产不准停·同刀 A 时序 / 刀 B 数值定位）。
        **doc "命题值比序"(B) defer**（须 ref→surface 基建·concept_index 无反查·设计 §三）：本刀做 (A) 字面数值比序。
        **stable≠correct**：比序一致 ≠ 命题真（"5 大于 3" 算术对·文本是否真陈述此比较是语义层·#479 墙）。
        """
        assert_int(round_id, _where="_run_comparison_verify_round.round_id")
        segments = raw.segments
        struct_refs = obs.struct_refs
        if not struct_refs:
            return RoundResult()   # observe 未建 struct_ref·诚实跳过（不伪造 reward）
        root = struct_refs[0]
        # 1. 收集比较声明（已解析纯整数 3-tuple·self-contained·无需 resolve·镜像 _run_numeric_verify_round claims 收集）
        claims: list[tuple[int, int, int]] = []
        for seg in segments:
            if not seg.comparison_claims:
                continue
            # 拷贝防 alias（Segment.comparison_claims 是 list·闭包捕获前 flatten 到单 claims list）
            claims.extend(seg.comparison_claims)
        if not claims:
            return RoundResult()   # 无比较声明·诚实 observe-only·不伪造 reward=0
        # 2. factory 闭包 → fn → 逐声明 cross_compare 验序（claims 已纯整数·闭包传·不入图·镜像 numeric_proof_fn_factory）
        fn = comparison_proof_fn_factory(claims=claims)
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 比较 verify 无词生成（语言域 episode 不 generate·镜像 numeric:722/time_seq:629）
        r = fn(output, dag_path, ctx.concept_graph)
        reward = 1 if r == 1 else 0   # 1 全声明比序一致 / 0 任一违反 / None claims 空（路由守非空·None 不应达·降级 reward=0）
        g5_active = True   # 比较 verify 承重门 active（pass=reward=1 / 违反=veto=reward=0·构造性检查层·诚实标·同 numeric/time_seq）
        ep = Episode(
            episode_id=round_id,
            run_id=round_id,
            input=raw,
            output=output,
            reward=reward,
            ref=root,
            terminal=TERMINAL_REACHED_SINK,
            pr_vector={},   # 比较 verify 不跑 dag_path_step·无 PR 向量（镜像 numeric:734/time_seq:641·anti_collapse skip 空 pr_vector）
            judge_G5_active=g5_active,   # 字段重载：语言域 G5=DEAD_DESIGN·非真 G5·"verify 门 active"标志（同 numeric/time_seq·capability_exam G 归因→None 正确·Layer0 summary 处理）
            judge_veto_count=1 if reward == 0 else 0,
            dead_end_count=0,
            vetoed=(reward == 0),
            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # Layer0 外部锚门：左/右式数 single-source·构造性检查·非验证·全自产不准停（同刀 A 时序 / 刀 B 数值）
        )
        return RoundResult(episode=ep, output=output, dag_path=dag_path)

    def _run_universal_verify_round(self, ctx: TrainContext, item: CollectedItem,
                                    raw: InputPayload, obs: Any,
                                    round_id: int) -> RoundResult:
        """刀 C：语言域全称量化 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 A/B verify round）。

        验 **全称量化内涵分类子集 X⊆Y**（构造性**验证**·ConceptNet 外部祖先图）·非 PRECEDES DAG·非数值算术·
        非 COMPOSES 执行。语言域 G5=DEAD_DESIGN → 走独立 episode 绕 judge（镜像 _run_time_seq/numeric_verify_round）。
        reward=1 iff 全声明 verified（ConceptNet 外部断言 child⊆parent）·不落 strength（verify propagate no-op）·
        **永不接 reward**（量化声明不入图·闭包传外部祖先图·Option A·同刀 A/B 防污染）。

        **★构造性验证层·首个 EXTERNAL**（刀 A 时序/刀 B 数值 SELF_PRODUCED 是构造性检查·刀 C 升验证·Layer0
        external_verified 首个语言域 episode 计入·可驱动停止决策·反 SELF_PRODUCED 全自产不准停）。
        resolve 段 token→ConceptRef（concept_index.lookup·镜像 time_seq:603-617·未概念化/自环跳）+
        **外部 ConceptNet 祖先图**（build_isa_ancestor_map_external·source=SOURCE_CONCEPTNET+epistemic=EPI_STRUCTURED
        双 filter·非 cue 自产·**反 single-source theater 核心**）→ universal_proof_fn_factory 闭包 → 三值判定。

        **三值诚实逻辑**（universal_proof_fn·守属性全称 G5b #479 墙）：
          r=1（全 verified·ConceptNet 确认 child⊆parent）→ reward=1·verify_source=EXTERNAL·产 episode
          r=0（任一 falsified·两分类概念 ConceptNet 否认子集）→ reward=0·verify_source=EXTERNAL·产 episode（外部证伪）
          r=None（任一 can't-verify·child/parent 非分类概念如"会飞"·外部源不足）→ **弃权·返空 RoundResult 无 episode**
          （诚实降级·非证伪·非 theater·守 G5b 属性全称 #479 墙）

        **Option A**：量化声明不入图（闭包传外部图·同刀 A/B）·标记在 Episode（verify_source）非边·#355 维持。
        **构造性验证 ≠ truth**：ConceptNet 外部源对齐非命题真（ConceptNet 可错·stable≠correct·#479 墙）。
        **刀 C ≠ G5b 实现**：验内涵分类子集（"所有鸟都是动物"）·属性全称子域（"所有鸟都会飞"）三值 None 弃权守墙。
        详 doc/重来_刀C量化cue设计_2026-07-08.md §六b。
        """
        assert_int(round_id, _where="_run_universal_verify_round.round_id")
        segments = raw.segments
        struct_refs = obs.struct_refs
        if not struct_refs:
            return RoundResult()   # observe 未建 struct_ref·诚实跳过（不伪造 reward）
        root = struct_refs[0]
        space_id = ctx.space_id
        # 1. resolve 全称量化声明（段内 token index → ConceptRef·concept_index.lookup·镜像 time_seq:603-617）
        # 收集 (child_ref, parent_ref) 对·未概念化/自环跳（守反统计·不凑配·同 time_seq cue_pair_edges）
        resolved_claims: list = []
        for seg in segments:
            if not seg.universal_claims:
                continue
            for (ci, pi) in seg.universal_claims:
                if ci >= len(seg.tokens) or pi >= len(seg.tokens):
                    continue
                child = ctx.concept_index.lookup(seg.tokens[ci], space_id)
                parent = ctx.concept_index.lookup(seg.tokens[pi], space_id)
                if child is None or parent is None:
                    continue   # token 未概念化→跳（诚实·反统计·不凑配·同 time_seq:613）
                if child == parent:
                    continue   # 自环不计（X⊆X 平凡·非全称量化声明·跳·镜像 time_seq:615）
                resolved_claims.append((child, parent))
        if not resolved_claims:
            return RoundResult()   # 无可 resolve 全称声明·诚实 observe-only·不伪造 reward=0
        # 2. 构建**外部 ConceptNet 祖先图**（仅 source=SOURCE_CONCEPTNET+epistemic=EPI_STRUCTURED·反 single-source
        # theater·每 round 调一次 run-scoped·CI 无文件→空 dict→全 can't-verify→None→弃权·非 theater）。perf cache defer。
        ext_map = build_isa_ancestor_map_external(ctx.backend, space_id=ctx.space_id)
        # 3. factory 闭包 → fn → 三值判定（verified/falsified/can't-verify·ext_concepts 判别·守属性全称墙）
        fn = universal_proof_fn_factory(ancestor_map=ext_map, claims=resolved_claims)
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 全称 verify 无词生成（语言域 episode 不 generate·镜像 time_seq:629/numeric:701）
        r = fn(output, dag_path, ctx.concept_graph)
        if r is None:
            return RoundResult()   # can't-verify（child/parent 非分类概念·外部源不足）→ 弃权·无 episode（诚实降级·非证伪·守 #479 墙）
        reward = 1 if r == 1 else 0   # 1 全 verified（ConceptNet 确认 X⊆Y）/ 0 任一 falsified（ConceptNet 证伪）
        g5_active = True   # 全称 verify 承重门 active（pass=reward=1 / 证伪=reward=0·构造性验证层·诚实标·同 time_seq/numeric）
        ep = Episode(
            episode_id=round_id,
            run_id=round_id,
            input=raw,
            output=output,
            reward=reward,
            ref=root,
            terminal=TERMINAL_REACHED_SINK,
            pr_vector={},   # 全称 verify 不跑 dag_path_step·无 PR 向量（镜像 time_seq:641/numeric:711·anti_collapse skip 空 pr_vector）
            judge_G5_active=g5_active,   # 字段重载：语言域 G5=DEAD_DESIGN·非真 G5·"verify 门 active"标志（同 time_seq/numeric·Layer0 summary 处理）
            judge_veto_count=1 if reward == 0 else 0,
            dead_end_count=0,
            vetoed=(reward == 0),
            verify_source=VERIFY_SOURCE_EXTERNAL,   # ★Layer0 首个语言域 EXTERNAL：ConceptNet 外部 R6 源·真构造性验证（reward>0 时 is_constructive_verification=True·external_verified 计入·可驱动停止决策）
        )
        return RoundResult(episode=ep, output=output, dag_path=dag_path)


    def _run_existential_verify_round(self, ctx: TrainContext, item: CollectedItem,
                                       raw: InputPayload, obs: Any,
                                       round_id: int) -> RoundResult:
        """A1·STEP6：语言域存在量化 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 C）。

        验 **存在量化 X∩Y≠∅**（构造性**验证**·ConceptNet 外部祖先图·**双向祖先** X⊆Y OR Y⊆X）·非 PRECEDES DAG·
        非数值算术·非 COMPOSES 执行。语言域 G5=DEAD_DESIGN → 走独立 episode 绕 judge（镜像 _run_universal_verify_round）。
        reward=1 iff 全声明 verified（ConceptNet 外部断言 X∩Y≠∅·双向其一子集）·不落 strength·
        **永不接 reward**（量化声明不入图·闭包传外部祖先图·Option A·同刀 C 防污染）。

        **★构造性验证层 EXTERNAL**（同刀 C·Layer0 external_verified 计入·反 SELF_PRODUCED 全自产不准停）。
        resolve 段 token→ConceptRef（concept_index.lookup·镜像 universal:876-891·未概念化/自环跳）+
        **外部 ConceptNet 祖先图**（build_isa_ancestor_map_external·复用 ∀ 的 ext_map·同 ConceptNet 源·
        source=SOURCE_CONCEPTNET+epistemic=EPI_STRUCTURED 双 filter·反 single-source theater）→
        existential_proof_fn_factory 闭包 → 三值判定。

        **三值诚实逻辑**（existential_proof_fn·守属性 ∃ #479 墙·同 ∀）：
          r=1（全 verified·ConceptNet 确认 X∩Y≠∅·双向其一子集）→ reward=1·verify_source=EXTERNAL·产 episode
          r=0（任一 falsified·两分类概念 ConceptNet 否认 X∩Y·双向皆不命中）→ reward=0·verify_source=EXTERNAL·产 episode（外部证伪）
          r=None（任一 can't-verify·child/parent 非分类概念如"会飞"·外部源不足）→ **弃权·返空 RoundResult 无 episode**
          （诚实降级·非证伪·非 theater·守属性 ∃ #479 墙）

        **Option A**：量化声明不入图（闭包传外部图·同刀 C）·标记在 Episode（verify_source）非边·#355 维持。
        **构造性验证 ≠ truth**：ConceptNet 外部源对齐非命题真（ConceptNet 可错·stable≠correct·#479 墙）。
        **严格实例存在 defer**：双向祖先验类层 X∩Y≠∅·实例非空=世界态 #479 墙 defer。
        """
        assert_int(round_id, _where="_run_existential_verify_round.round_id")
        segments = raw.segments
        struct_refs = obs.struct_refs
        if not struct_refs:
            return RoundResult()   # observe 未建 struct_ref·诚实跳过（不伪造 reward）
        root = struct_refs[0]
        space_id = ctx.space_id
        # 1. resolve 存在量化声明（段内 token index → ConceptRef·concept_index.lookup·镜像 universal:876-891）
        # 收集 (child_ref, parent_ref) 对·未概念化/自环跳（守反统计·不凑配·同 universal cue_pair_edges）
        resolved_claims: list = []
        for seg in segments:
            if not seg.existential_claims:
                continue
            for (ci, pi) in seg.existential_claims:
                if ci >= len(seg.tokens) or pi >= len(seg.tokens):
                    continue
                child = ctx.concept_index.lookup(seg.tokens[ci], space_id)
                parent = ctx.concept_index.lookup(seg.tokens[pi], space_id)
                if child is None or parent is None:
                    continue   # token 未概念化→跳（诚实·反统计·不凑配·同 universal:888）
                if child == parent:
                    continue   # 自环不计（X∩X=X 平凡·非存在量化声明·跳·镜像 universal:890）
                resolved_claims.append((child, parent))
        if not resolved_claims:
            return RoundResult()   # 无可 resolve 存在声明·诚实 observe-only·不伪造 reward=0
        # 2. 构建**外部 ConceptNet 祖先图**（复用 ∀ 的 build_isa_ancestor_map_external·同 ConceptNet 源·
        # 反 single-source theater·每 round 调一次 run-scoped·CI 无文件→空 dict→全 can't-verify→None→弃权·非 theater）
        ext_map = build_isa_ancestor_map_external(ctx.backend, space_id=ctx.space_id)
        # 3. factory 闭包 → fn → 三值判定（双向祖先 verified/falsified/can't-verify·守属性 ∃ 墙）
        fn = existential_proof_fn_factory(ancestor_map=ext_map, claims=resolved_claims)
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 存在 verify 无词生成（语言域 episode 不 generate·镜像 universal:903）
        r = fn(output, dag_path, ctx.concept_graph)
        if r is None:
            return RoundResult()   # can't-verify（child/parent 非分类概念·外部源不足）→ 弃权·无 episode（诚实降级·非证伪·守 #479 墙）
        reward = 1 if r == 1 else 0   # 1 全 verified（ConceptNet 确认 X∩Y≠∅）/ 0 任一 falsified（ConceptNet 证伪）
        g5_active = True   # 存在 verify 承重门 active（pass=reward=1 / 证伪=reward=0·构造性验证层·诚实标·同 universal）
        ep = Episode(
            episode_id=round_id,
            run_id=round_id,
            input=raw,
            output=output,
            reward=reward,
            ref=root,
            terminal=TERMINAL_REACHED_SINK,
            pr_vector={},   # 存在 verify 不跑 dag_path_step·无 PR 向量（镜像 universal:917·anti_collapse skip 空 pr_vector）
            judge_G5_active=g5_active,   # 字段重载：语言域 G5=DEAD_DESIGN·非真 G5·"verify 门 active"标志（同 universal·Layer0 summary 处理）
            judge_veto_count=1 if reward == 0 else 0,
            dead_end_count=0,
            vetoed=(reward == 0),
            verify_source=VERIFY_SOURCE_EXTERNAL,   # ★构造性验证 EXTERNAL（同 ∀·ConceptNet 外部 R6 源·reward>0 时 is_constructive_verification=True·external_verified 计入·可驱动停止决策）
        )
        return RoundResult(episode=ep, output=output, dag_path=dag_path)


# 句末标点（多段切分·设计原意：CollectedItem 段落 → 句段 Segment 串 struct_ref 链）
_SENT_END_CHARS = frozenset("。.!?！？;；\n")


def _sentence_bounds(tokens) -> list[tuple[int, int]]:
    """句末标点切句边界（_SENT_END_CHARS·token 末字符 ∈ 句末标点 → 该 token 后切）。

    返回 [(start, end)] token 半开区间列表（保序·含句末标点 token）。**与 observe `_split_item_to_segments`
    同源切法**（该片 refactor 后两者共用本函数）→ discovery scope B 句级根的 seg_idx 与 observe segment 序
    **逐一对齐**（lang_skeleton_by_item[(id(item),seg_idx)] 键对齐保证·非脆弱约定）。

    scope B（断奶 critical path ④·doc/重来_语料聚簇规模_2026-07-17）：语言 shape_signature 坍缩为词数·整段
    当聚簇单元→每段独份签名→零骨架。切到句级→同长句聚簇→骨架+cue槽涌现。observe 本就按句切段（多段 struct_ref）·
    故切句**不增 observe 成本**（unit 早已句级）·只让 discovery 从段级根改句级根 + map 键升 (id(item),seg_idx)。

    空入参→[]。无句末标点→[(0,len)]（整段当一句·同原行为）。
    """
    cuts: list[int] = []
    for i, tok in enumerate(tokens):
        if tok and tok[-1] in _SENT_END_CHARS:
            cuts.append(i + 1)
    if not cuts or cuts[-1] < len(tokens):
        cuts.append(len(tokens))
    spans: list[tuple[int, int]] = []
    start = 0
    for end in cuts:
        if end > start:
            spans.append((start, end))
            start = end
    return spans


def _is_verify_modality(modality: int) -> bool:
    """是否 verify-driven COMPOSES 模态（代码/算术·vm_proof_fn 验执行 vs spec·绕 judge/generate/propagate）。

    路由（run_round_full）+ H2 排除（_h2_calibrate）共用·防漏排（doc/重来_算术域observe设计补充.md §九）。
    code/arith reward 经 vm_proof_fn 不用 JudgeWeights·进 language judge()→G2p veto reward=0
    对齐 GT=1=垃圾标定污染 JudgeWeights。
    """
    return modality in (MODALITY_CODE, MODALITY_ARITH)


def _split_item_to_segments(item: CollectedItem, *,
                            backend=None, edge_store=None,
                            space_id: int | None = None,
                            concept_index=None) -> list[Segment]:
    """CollectedItem 段落 → 句段 Segment 列表（多段 observe 串 struct_ref 链·设计原意·破致命6）。

    按句末标点切（。.!?；换行）·句段内 token 序保 PRECEDES 骨架。单句段落 → 1 段
    （struct_ref 孤立·reward 阶段 caller 跳过 episode·诚实·emergent_role defer）。
    role_seq/causal_pairs/alias_cue_pairs 按 token 切片 + 段内 index 重映射（确定性）。

    **刀5 件8 透传**（close 刀4 生产 gap）：4 可选参透传给 extract_cues_gated → cue_type_of
    第二源 D:11 readback。生产 caller run_round_full 传 ctx.backend/edge_store/space_id/concept_index。
    默认全 None → cue_type_of 退化纯 frozenset → 现状零行为变（bit-identical）。space_id 是
    语言 token 概念化所在 core space（ctx.space_id·concept_index.lookup(surface, ctx.space_id)）。

    **代码域分支**（C6 生产闭环·doc/重来_A3_代码域observe设计补充.md §二致命#2）：MODALITY_CODE
    段不按句末标点切（代码 `;`/`\n`/`.` 会切碎函数体）·一段一函数·Segment 带 code_source。
    """
    # 代码域：一段一函数·不按句切（代码标点会碎函数体·observe MODALITY_CODE gate 建 COMPOSES 树）
    if item.modality == MODALITY_CODE:
        if not item.code_source:
            return []   # 无源码→observe 无可建·诚实返空（不伪造段）
        return [Segment(
            seg_id=0, modality=MODALITY_CODE, lang=LANG_NONE,
            domain=item.domain if item.domain == DOMAIN_CODE else DOMAIN_CODE,
            code_source=item.code_source,
        )]
    # 算术域：一段一 lambda 记号·不按句切（observe MODALITY_ARITH gate 建 COMPOSES 树·doc §九）
    if item.modality == MODALITY_ARITH:
        if not item.arith_source:
            return []   # 无记号→observe 无可建·诚实返空（不伪造段）
        return [Segment(
            seg_id=0, modality=MODALITY_ARITH, lang=LANG_NONE,
            domain=item.domain if item.domain == DOMAIN_MATH else DOMAIN_MATH,
            arith_source=item.arith_source,
        )]
    tokens = list(item.tokens)
    if not tokens:
        return []
    # 句边界：_sentence_bounds（_SENT_END_CHARS·与 discovery scope B 同源切法·seg_idx 对齐保证）
    segs: list[Segment] = []
    for start, end in _sentence_bounds(tokens):
        rseq = list(item.role_seq[start:end]) if item.role_seq else []
        cpairs = [(a - start, b - start) for (a, b) in item.causal_pairs
                  if start <= a < end and start <= b < end]
        apairs = [(a - start, b - start) for (a, b) in item.alias_cue_pairs
                  if start <= a < end and start <= b < end]
        seg_tokens = tokens[start:end]
        # 指向词/系词提取（致命3 来源②·CUE_EXTRACTOR_MODE OFF → 空·守回归 bit-identical）
        # 刀5 件8：透传 ctx 4 参 → cue_type_of 第二源 D:11 readback（gate CUE_READBACK_MODE）
        cue_pairs, is_a_seg_pairs, precedes_seg_pairs = extract_cues_gated(
            seg_tokens, lang=item.lang,
            backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index)
        # 刀 B：数值等式声明提取（独立函数·不改 extract_cues 3-tuple·同 CUE_EXTRACTOR_MODE gate·
        # NUM OP NUM 等于 NUM 窗口扫描·闭包传 numeric_proof_fn 检查·构造性检查 SELF_PRODUCED）
        numeric_seg_claims = extract_numeric_claims_gated(
            seg_tokens, lang=item.lang,
            backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index)
        # 刀 C：全称量化声明提取（独立函数·同 CUE_EXTRACTOR_MODE gate·X 都是 Y 紧邻 pair·
        # resolve 在验序器·ConceptNet 外部源验·构造性验证 EXTERNAL·三值逻辑守属性全称墙）
        universal_seg_claims = extract_universal_claims_gated(
            seg_tokens, lang=item.lang,
            backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index)
        # A1·STEP6：存在量化声明提取（独立函数·同 CUE_EXTRACTOR_MODE gate·有的 X 是 Y 起始 cue 窗口·
        # resolve 在验序器·ConceptNet 外部源验·构造性验证 EXTERNAL·双向祖先 X⊆Y OR Y⊆X·三值逻辑守属性 ∃ 墙）
        existential_seg_claims = extract_existential_claims_gated(
            seg_tokens, lang=item.lang,
            backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index)
        # G1+#774：属性命题声明提取（独立函数·gate PROPOSITION_MODE·X 的 Y 是 Z 固定窗口 6-tuple·P0.3 扩 pol/mod·
        # observe build_property_edges 建命题节点+PROPERTY 出边·G3b 读判同(subject,attr_type)多值矛盾·入图非闭包传）
        property_seg_claims = extract_property_claims_gated(
            seg_tokens, lang=item.lang,
            backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index)
        # 刀 D：比较声明提取（独立函数·gate CUE_EXTRACTOR_MODE·NUM 比较OP NUM 紧邻 3-token 窗口·
        # 闭包传 comparison_proof_fn 检查（cross_compare）·不入图·构造性检查 SELF_PRODUCED·同刀 B 范式）
        comparison_seg_claims = extract_comparison_claims_gated(
            seg_tokens, lang=item.lang,
            backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index)
        # STEP5 PR4：相似声明提取（X 像 Y·EDGE_SIMILAR slot-filler 候选扩展·D2 合规非向量·gate CUE_EXTRACTOR_MODE）
        similar_seg_claims = extract_similar_claims_gated(
            seg_tokens, lang=item.lang,
            backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index)
        segs.append(Segment(
            seg_id=len(segs),
            modality=item.modality, lang=item.lang, domain=item.domain,
            tokens=seg_tokens, role_seq=rseq,
            structured_causal_pairs=cpairs,
            cue_based_causal_pairs=cue_pairs,
            is_a_pairs=is_a_seg_pairs,
            precedes_pairs=precedes_seg_pairs,
            numeric_claims=numeric_seg_claims,
            universal_claims=universal_seg_claims,
            existential_claims=existential_seg_claims,
            property_claims=property_seg_claims,
            comparison_claims=comparison_seg_claims,
            similar_claims=similar_seg_claims,
            alias_cue_pairs=apairs,
        ))
    return segs


# 目标达成覆盖率阈值（attractor 第一本职"目标达成"判据·阶段9 S1·纯整 0..1000）。
# 500 = oracle 标定起点（满分 1000 过半·"路径覆盖目标骨架过半算达成"·真训练 run 前校准·同 THETA_FREQ 范式）。
# episode_loop/_rebuild_path 生产路径透传·dag_path_step 默认 0 守 bit-identical（既有测试不传→退化）。
COVERAGE_THRESHOLD = 500


def _hotzone_dag_edges(edge_store, seeds: list[ConceptRef],
                      max_hops: int = 2) -> list:
    """perf round6：k-hop 热区 BFS via edge_store.query_from（covering-indexed·O(k-hop neighborhood)·非全扫）。

    设计主线 line 530/978/1224 "热区子图非全图·已有·复用"·镜像 PR HOTZONE（a3_pr_wrapper k-hop ≤2048）。
    从 seeds 沿 {PRECEDES,CAUSES,T_STEP} 出边 BFS ≤max_hops 跳·query_from 用 backend covering 索引（O(degree)/
    concept）→ 总 O(k-hop 邻域) 非 O(全 dag_edges)。保 H4 闭环（item seed 邻域 CAUSES 进 path.edges）·切热区外
    无关累积边（设计本就不让进热区）。n=8 k=2 实测 3.4x 加速·capability status 全保（②PASS 856=856）·非大退化。
    """
    _TYPES = (EDGE_PRECEDES, EDGE_CAUSES, EDGE_T_STEP)
    reachable: set[ConceptRef] = set(seeds)
    frontier = list(seeds)
    out: list = []
    hops = 0
    while frontier and hops < max_hops:
        nxt: list[ConceptRef] = []
        for sid, lid in frontier:
            for et in _TYPES:
                for r in edge_store.query_from(sid, lid, edge_type=et):
                    out.append(r)
                    v = (r["space_id_to"], r["local_id_to"])
                    if v not in reachable:
                        reachable.add(v)
                        nxt.append(v)
        frontier = nxt
        hops += 1
    return out


def _reachable_dag_edges(dag_edges: list, seeds: list[ConceptRef],
                         max_hops: int | None = None) -> list:
    """perf round6：从 seeds 前向 BFS·dag_edges 子集（gate DAG_PATH_REACHABLE_SCOPE_MODE）。

    **k-hop bounded（镜像 PR HOTZONE·设计主线 line 530/978/1224 "热区子图非全图"）**·非 unbounded reachable
    （密集图 reachable≈full 失效）。max_hops=None 全可达·=int k 跳内（PR HOTZONE k=2 范式）。path 通常在 k-hop
    内（seed→sink 短路径）→ k-hop 子集可能 bit-identical（AB 测）。O(dag_edges) 建 adj + O(k-hop) BFS。
    保插入序（dag_edges 序内子集）·dag_path 序无关 consumer（round5 edge filter 证）。
    """
    adj: dict[ConceptRef, list] = {}
    for e in dag_edges:
        u = (e["space_id_from"], e["local_id_from"])
        adj.setdefault(u, []).append(e)
    reachable: set[ConceptRef] = set(seeds)
    frontier = list(seeds)
    out: list = []
    hops = 0
    while frontier and (max_hops is None or hops < max_hops):
        nxt: list[ConceptRef] = []
        for u in frontier:
            for e in adj.get(u, ()):
                out.append(e)
                v = (e["space_id_to"], e["local_id_to"])
                if v not in reachable:
                    reachable.add(v)
                    nxt.append(v)
        frontier = nxt
        hops += 1
    return out


def _rebuild_path(ctx: TrainContext, dag_edges: list, seeds: list[ConceptRef],
                  intent: IntentType, round_id: int, *,
                  key_skeleton: list[ConceptRef] | None = None,
                  coverage_threshold: int = 0,
                  ctx_code: int = 0):
    """确定性重建 path_result（H2 标定用·同 episode_loop 内部 dag_path_step·bit-identical）。

    key_skeleton/coverage_threshold/ctx_code 须与 episode_loop 生产调用同传（H2 重算 path 与生产 bit-identical）。
    **S4 片4**：ctx_code 透传（stage8 latent 修·_seed_weight freq+selection_pref dock 后 attractor 扩张路径
    token seed eff_freq>0 读 ctx 桶·生产 episode_loop :82 / H2 _rebuild_path 须同桶·否则 bit-identical 失）。
    **B-PR3**：edge_store 透传（gate③ _intent_override D:11 查找需 edge_store·生产 episode_loop 穿 →
    H2 _rebuild_path 须同穿·否则 gate ON 时 gate③ override 分叉 → H2 标定 path ≠ 生产 path → bit-identical 失）。
    """
    from pure_integer_ai.cognition.process.dag_path import dag_path_step
    return dag_path_step(dag_edges, seeds, ctx.work_memory, intent,
                         current_seq=round_id, memory_active=False,
                         backend=ctx.backend,
                         edge_store=ctx.edge_store,
                         ctx_code=ctx_code,
                         key_skeleton=key_skeleton,
                         coverage_threshold=coverage_threshold)


def _build_space_ctx(ctx: TrainContext):
    """造 observe 用的 SpaceContext（核心空间·memory/companion 首版 None·训练期 memory_active=False）。

    M10 第一刀 11a：memory_read/memory_interact 实例化挂 TrainContext（make_train_context）·
    SpaceContext 训练期守 None（守 observe:93 bit-identical + 训练期核心养洁净）·11d 落点② 写
    用 TrainContext.memory_read（episode_loop memory_read 参数·不经 SpaceContext）。
    """
    from pure_integer_ai.cognition.shared.types import SpaceContext
    return SpaceContext(
        core=ctx.core_space, memory_read=None, memory_interact=None,
        companion=None, stage=STAGE_TRAINING, memory_active=False,
        weaning_phase=ctx.weaning_phase,
    )


def _run_calibration_phase(ctx: TrainContext, corpus: list,
                           backend: StorageBackend) -> None:
    """W5 D5 Mode B 预验台账：stage4 末并行 Mode A vs B 评估（config.calibrate_mode_b=True 触发）。

    calibration_set = training_corpus 算术 item（已 observe 学过·非 held-out probe·D5 验两路在学过的
    东西上一致=stable≠correct 墙内弱自洽验证·非保持率度量）。WEANING_WINDOW_ROUNDS 轮独立 calibration
    （round_id 高位偏移 10M 避 training round_id 碰撞·与 rounds_per_stage 解耦·4 distinct rid 填窗）·
    每轮 per arith item：
      Mode A = vm_proof_fn_factory execute 学树 vs spec.expected（静态整数·非 live teacher·镜像 PRE :581-587）
      Mode B = cross_verify_pair（root_a observe 学树 × root_b build_composes_from_arith 二次独立建·
               异 shape R6 真守·纯 VM execute + rational.eq·零 judge/teacher·镜像 POST :619-630）
    → record_calibration(backend, round_id, mode_a_pass, mode_b_pass) 真写台账。

    纯评估：不写 episode / 不染 op_confidence / 不进 conduction_rate（calibration 独立台账·职责分离）。
    幂等：observe guard skip 已建树（observe.py:195）·build root_b guard skip 已建参树（:624 范式）·
    确定性 bit-identical。D5 域无关（Mode A/B 零 judge/teacher）→ 算术域走通用 track·无须判定接口（同 D4）。

    诚实边界：flat trend=不回升=通过（MUTABLE_MONOTONE·学树 stage4 静态→4 轮同 rate·FLOOR_MODE_B=500
    守低通过率平台）·stable≠correct（两路径一致≠对错·D5 既有边界）·single-source 脆弱（两 DSL 同出
    corpus·系统性中毒 agree wrong 无法检出·weaning_calibration.py:23·非 W5 须修的 bug）。
    """
    from pure_integer_ai.teacher.weaning_calibration import record_calibration
    from pure_integer_ai.teacher.weaning import WEANING_WINDOW_ROUNDS
    from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
    from pure_integer_ai.cognition.understanding.pronoun_features import lookup_pronoun_features
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
    from pure_integer_ai.storage.node_store import NODE_CONCEPT
    from pure_integer_ai.storage.edge_store import SOURCE_MATH

    # calibration_set = training_corpus 算术 item（须有 arith_source + arith_specs + arith_source_b）
    cal_items = [it for it in corpus
                 if it.modality == MODALITY_ARITH and it.arith_source
                 and it.arith_specs and it.arith_source_b]
    if not cal_items:
        return   # 无 calibration 候选（arith_source_b 缺）→ 台账空 → mode_b_prevalidated=False（诚实）

    sctx = _build_space_ctx(ctx)
    # Phase 1：observe（幂等·guard skip 已建树）拿 root_a + build root_b（幂等 guard）·缓存复用（学树 stage4 静态）
    cal_data: list[tuple[Any, Any, Any]] = []   # (item, root_a, root_b)
    for item in cal_items:
        segments = _split_item_to_segments(
            item, backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index)
        if not segments:
            continue
        raw = InputPayload(
            segments=segments, source=item.source, stage=STAGE_TRAINING,
            modality=item.modality, lang=item.lang, domain=item.domain,
            weaning_phase=ctx.weaning_phase,
            item_key=id(item),
        )
        obs = observe(raw, sctx, concept_index=ctx.concept_index,
                      work_memory=ctx.work_memory,
                      pronoun_feature_lookup=lookup_pronoun_features,
                      sense_lookup=make_sense_lookup(ctx.backend, ctx.space_id))
        if not obs.struct_refs:
            continue
        root_a = obs.struct_refs[0]   # observe 学树 COMPOSES 根（单函数单 struct_ref·镜像 :566）
        # root_b：参树（异 shape·build_composes_from_arith 二次独立建·镜像 POST :619-627·R6 真守）
        h_b = Hasher('xver.b.v1').h63(item.arith_source_b)
        root_b = ctx.concept_index.ensure(
            f"__xver_b_{h_b}", space_id=ctx.space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        if not ctx.edge_store.query_from(root_b[0], root_b[1], edge_type=EDGE_COMPOSES):
            build_composes_from_arith(
                item.arith_source_b, concept_index=ctx.concept_index,
                edge_store=ctx.edge_store, backend=ctx.backend,
                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=root_b)
        cal_data.append((item, root_a, root_b))

    if not cal_data:
        return   # 全 observe 失败 → 台账空 → mode_b_prevalidated=False（诚实）

    # Phase 2：WEANING_WINDOW_ROUNDS 轮 calibration（复用缓存 root_a/root_b·学树 stage4 静态）
    output = OutputResult()   # Mode A vm_proof_fn 只读 dag_path.sink·output 共用空壳（代码域无词生成）
    for cal_round in range(WEANING_WINDOW_ROUNDS):
        cal_rid = 10_000_000 + cal_round   # 高位偏移避 training round_id 碰撞（mode_b_pass_series 按 rid 分组）
        for item, root_a, root_b in cal_data:
            # Mode A：vm_proof vs spec.expected（镜像 PRE :581-587·静态整数非 live teacher）
            dag_path = PathResult(
                path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_a,
                topo_layers=[], convergence={}, source=None,
            )
            mode_a_pass = 1
            for spec in item.arith_specs:
                fn = vm_proof_fn_factory(input_args=spec.input_args,
                                         expected=spec.expected)
                r = fn(output, dag_path, ctx.concept_graph)
                if r != 1:   # 0 mismatch / None deadloop（R1 PRE→0 诚实·非 vacate）
                    mode_a_pass = 0
                    break
            # Mode B：cross_verify_pair（镜像 POST :630·root_a × root_b 独立建·纯 VM 零 judge/teacher）
            probes = tuple(spec.input_args for spec in item.arith_specs)
            cv = cross_verify_pair(ctx.concept_graph, root_a, root_b, probes)
            mode_b_pass = 1 if cv.all_agree else 0
            record_calibration(backend, round_id=cal_rid,
                               mode_a_pass=mode_a_pass, mode_b_pass=mode_b_pass)


def _run_simulated_offline_eval(ctx: TrainContext, corpus: list,
                                backend: StorageBackend) -> None:
    """W6 E2 模拟退场验证子阶段（stage4 末·weaning_check 之前·解 teacher_offline 循环依赖·预验非后验）。

    读 ctx.probe_corpus（W4 held-out 探针·**W6 首个 reader**·从未进训练 observe/boot/discovery/H2/generate/
    base_freq）→ per probe item：observe 探针建学树 root_a（eval 时首次 observe·非训练时·probe 是教师从未评过的
    新输入·D4 守）+ build root_b 参树（镜像 _run_calibration_phase :1266-1275·异 shape R6 守）→ cross_verify_pair
    （root_a × root_b·零教师·VM 执行值自锚·mode_b_cross_verify.py:64-97）。

    三条件采（镜像 e2_execution_ready_arith 入参）：
      teacher_offline = (ctx.teacher is None)    算术域 teacher=None→真退场（架构事实·反 theater：无 recording/replay/GT）
      probe_input_novel = ctx.probe_set_disjoint 探针集隔离（W4 track·held-out 不相交训练集）
      produced_without_teacher_anchor = produced_without_teacher_anchor_arith(cross_verify_ran, cv_all_agree)
    → ctx.e2_eval_passed = e2_execution_ready_arith(三条件 and)  路径 B 读
    → ctx.holdout_retention = cross_verify 通过率 ×1000  W4 defer 的 track 填真值（默认 0·eval 驱动真值）

    纯评估：不写 episode / 不染 op_confidence / 不进 conduction_rate / 不调 record_round（镜像 calibration·
    职责分离）。eval 在 stage loop 外所有 stage_metric_gate + record_round 之后→零偷渡 D1-D5。

    算术域（teacher=None）跑 eval。语言域（teacher≠None）no-op return（真翻 MODE_OFF + 恢复 defer W8）。

    诚实边界：
      - 算术域 teacher=None 是架构事实非"模拟退场"（语言域真翻 MODE_OFF + 恢复 MODE_REPLAY defer W8）
      - 算术域 fixture 同源 trivial（probe/training 都 square n²→cross_verify 恒 agree→retention 恒 1000·真泛化 defer W8）
      - eval observe 探针建学树是 fresh-compile（从 arith_source 编译）·非 recognize_operators（回忆已学骨架）·
        验"自锚产出非教师锚"（E2 核心）·非"保持率泛化"（retention 真泛化 defer W8）
      - probe 学树 root_a + root_b 参树留 backend（eval 后不清理·镜像 calibration root_b 既有范式·下游不扫图节点·无功能影响）
    """
    # 语言域 no-op（teacher≠None·真翻 MODE_OFF defer W8·算术域 eval 只在 teacher=None 时跑）
    if ctx.teacher is not None:
        return

    # 读 ctx.probe_corpus（W4 held-out 探针·W6 首个 reader）
    probe_items = [it for it in ctx.probe_corpus
                   if it.modality == MODALITY_ARITH and it.arith_source
                   and it.arith_specs and it.arith_source_b]
    if not probe_items:
        return   # 无 held-out 探针（probe_holdout=0 或无 arith_source_b）→ eval no-op → ctx.e2_eval_passed 默认 False

    # 算术域 teacher_offline：teacher=None→真退场（上面 guard 已守·恒 True·反 theater① 机制保证）
    teacher_offline = (ctx.teacher is None)
    # probe_input_novel：探针集隔离（W4 ctx track·caller 传 probe_holdout>0 切分验 disjoint）
    probe_input_novel = ctx.probe_set_disjoint

    from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
    from pure_integer_ai.cognition.understanding.pronoun_features import lookup_pronoun_features
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
    from pure_integer_ai.storage.node_store import NODE_CONCEPT
    from pure_integer_ai.storage.edge_store import SOURCE_MATH

    sctx = _build_space_ctx(ctx)
    total_valid = 0
    total_agree = 0
    cv_all_agree_global = True
    cross_verify_ran = False

    for item in probe_items:
        # observe 探针建学树 root_a（eval 时首次 observe·probe 从未进训练 observe·D4 守·镜像 calibration :1248-1265）
        segments = _split_item_to_segments(
            item, backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index)
        if not segments:
            continue
        raw = InputPayload(
            segments=segments, source=item.source, stage=STAGE_TRAINING,
            modality=item.modality, lang=item.lang, domain=item.domain,
            weaning_phase=ctx.weaning_phase,
            item_key=id(item),
        )
        obs = observe(raw, sctx, concept_index=ctx.concept_index,
                      work_memory=ctx.work_memory,
                      pronoun_feature_lookup=lookup_pronoun_features,
                      sense_lookup=make_sense_lookup(ctx.backend, ctx.space_id))
        if not obs.struct_refs:
            continue
        root_a = obs.struct_refs[0]   # observe 学树 COMPOSES 根（单函数单 struct_ref·镜像 :1265）
        # build root_b 参树（镜像 calibration :1266-1275·异 shape R6 真守）
        h_b = Hasher('xver.b.v1').h63(item.arith_source_b)
        root_b = ctx.concept_index.ensure(
            f"__xver_b_{h_b}", space_id=ctx.space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        if not ctx.edge_store.query_from(root_b[0], root_b[1], edge_type=EDGE_COMPOSES):
            build_composes_from_arith(
                item.arith_source_b, concept_index=ctx.concept_index,
                edge_store=ctx.edge_store, backend=ctx.backend,
                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=root_b)

        # cross_verify_pair（root_a × root_b·零教师·VM 执行值自锚·mode_b_cross_verify.py:64-97）
        probes = tuple(spec.input_args for spec in item.arith_specs)
        cv = cross_verify_pair(ctx.concept_graph, root_a, root_b, probes)
        cross_verify_ran = True
        total_valid += cv.n_valid
        total_agree += cv.n_agree
        if not cv.all_agree:
            cv_all_agree_global = False

    if not cross_verify_ran:
        return   # 全 observe 失败 → eval no-op → ctx.e2_eval_passed 默认 False

    # 采 holdout_retention 真值（cross_verify 通过率 ×1000·纯整禁浮点·W4 defer 的 track 填真值）
    ctx.holdout_retention = (total_agree * 1000) // max(total_valid, 1)

    # 验产出非教师锚（produced_without_teacher_anchor_arith·W2 已建·复用）
    from pure_integer_ai.teacher.weaning_e2 import (
        produced_without_teacher_anchor_arith, e2_execution_ready_arith)
    produced_without_teacher = produced_without_teacher_anchor_arith(
        cross_verify_ran=cross_verify_ran,
        cv_all_agree=cv_all_agree_global)

    # E2 算术域三条件 and → ctx.e2_eval_passed（路径 B 读·解循环依赖：预验非后验）
    ctx.e2_eval_passed = e2_execution_ready_arith(
        teacher_offline=teacher_offline,
        probe_input_novel=probe_input_novel,
        produced_without_teacher_anchor=produced_without_teacher)


# ---- floor 端到端下游激活率 orchestrator（断奶 critical path 第 3 件 piece 3·doc/重来_floor_orchestrator_piece3_2026-07-17）----
#
# 承 piece 2 measure 机制（floor_measure.py 纯读·DONE）+ readback→generation 桥（DONE）+ 方法论 PIVOT（reward 非 frame·
# floor = 端到端下游激活率）。piece 3 = 生产胶水：把 held-out（probe_corpus）项喂过「tally-free discovery → observe →
# minimal OutputResult → measure_floor_activation」产 FloorActivation·wire 进 language_statistical_weaning_check。
#
# **2 对抗审简化（post-review·§9 锁定）**：
#   - **drop generate**（审2 MEDIUM-1）：trace 证 `OutputPart.token_refs` = observed input token（generate.py:180,191 存
#     observed token 非 dispatch winner）→ measure_floor_activation 读 cue_rel_of(OBSERVED cue 位 token)·**与 dispatch_slot
#     选择 + generate 产词无关**。故 dag_path / generate_output / _rebuild_path 全 vestigial·从 observed struct_ref 直接算。
#     解 M2（无 generate→无 work_memory 变异→无 task-driven-generate 污染）+ MEDIUM-2（无 dispatch→collide/sub 膨胀
#     irrelevant·floor 读 D:11 on observed token·structurally immune to collide/sub inflation）。
#   - **诚实 framing**（审2 MEDIUM-1 + scope elevation）：首版 floor = **RECOGNITION 测**「D:11 学习链（observe→tally→promote）
#     fired + held-out cue 位 observed token 携 training-learned D:11 匹配 skeleton rel」= 机制层预验。**非测**「generation
#     activates / bonus 真驱动 / 学全 / 新词反推 / vocab-disjoint」（后者 defer Phase E/F）。production/bonus-driven test = Phase F。
#
# **3 隔离锁（反 theater）**：
#   - **tally-free discovery 锁**：held-out discovery pass **禁调 tally_cue_slot_matches**（否则 held-out_root 进
#     structure_match_count → (cue 词,held-out-instance) pairing 不 disjoint → activation 虚高 theater）。
#     auto_discover_operators 天然 tally-free（结构发现+注册·无 tally caller）。本 pass docstring 显式锁此不变量。
#   - **sample-disjoint 锁**：probe_corpus 隔离（held-out 从未进 training tally·W4 shadow）→ (cue 词,held-out-instance)
#     pairing 自动 disjoint。probe_set_disjoint 早返（审2 LOW-3·defense-in-depth）。
#   - **post-training 隔离锁**：observe 跑 stage loop 后（其余 verdict 锚已算完·held-out 边惰性·mirror W6）。


def _held_out_discovery_tally_free(
        ctx: TrainContext,
        backend: StorageBackend,
        lang_probe: list,
        training_lang_ops: Sequence[DiscoveredOperator],
        ) -> tuple[list[ConceptRef], dict[int, ConceptRef]]:
    """held-out tally-free discovery 子集（mirror `_discover_and_recognize_lang_structures:3481-3579` SKIP tally + routing）。

    建 `__disc_lang_{h63(tokens)}` 内容哈希独立根 + COMPOSES pre-build（mirror :3481-3513·auto_discover 需 shape_sig·
    idempotent query_from skip）+ variable-length align（mirror :3521-3553·异 token 数 roots 换 __disc_lang_align_*）+
    `auto_discover_operators`（mirror :3562·**body 天然 tally-free**·建 skeleton + cue_sig）+ `label_realizes_is_a`/
    `label_realizes_causes`（mirror :3569-3579·gated REALIZES_MODE·外源 ConceptNet oracle·sound 无 D:11 写点）。

    **S1 fix（审1 严重·load-bearing）**：sample-disjoint held-out 与训练同 shape_signature → auto_discover 幂等 skip
    （structure_discover.py:1366-1368·lookup+ATTR_OPERATOR_DEF）→ held-out root **不在 this-call `discovered`** →
    返空 → 若仅用 this-call discovered 建 lang_skeleton_by_item map → map 不填 → observe :204-208 不 fire INSTANTIATES
    → read_instantiates=None → measured=False silent veto。**修**：build `_skel_by_sig` 搜 `all_ops = training_lang_ops
    + newly_discovered`（非仅 this-call discovered·sample-disjoint held-out skeleton = 训练已注册·lookup 命中）。

    ★ **不变量锁**（审2 LOW-1 + piece 2 floor_measure.py:16-19）：本函数 **禁调 tally_cue_slot_matches**
    （:3647 SKIP·tally 须仅训练路径 caller·held-out 测量侧禁污染 D:11·防 held-out_root 进 structure_match_count →
    pairing 不 disjoint → activation 虚高 theater）。auto_discover_operators 天然 tally-free（structure_discover.py:1540
    caller 仅训练路径 :3647·body 无 tally）。本函数 SKIP routing/recognize/expected_verified（idempotency skip + S1 all_ops
    search 已守·mirror :3558/:3633/:3658 SKIP）。

    Returns (roots, skel_by_item_new)。roots 是 held-out 内容哈希根（scope B 句级·变长 align 后可能换 align_root·用于 orchestrator
    诊断/log）·skel_by_item_new = {(id(item), seg_idx): skeleton_ref}（scope B·S1 all_ops 搜结果·caller 写 work_memory.lang_skeleton_by_item）。
    """
    from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_OPERATOR
    from pure_integer_ai.numeric.symbol_domain import OPCODE_NOP
    from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, EPI_STRUCTURED
    from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
    from pure_integer_ai.storage.node_store import NODE_CONCEPT
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    from pure_integer_ai.cognition.process.structure_discover import (
        auto_discover_operators, shape_signature as _dim_shape_sig,
        label_realizes_is_a, label_realizes_causes,
    )

    roots: list[ConceptRef] = []
    root_keys: list[tuple[int, int]] = []   # scope B：(id(item), seg_idx) per root·observe (item_key,seg_idx) 对齐
    # scope B（同训练 _discover_and_recognize_lang_structures·断奶 critical path ④）：gate COMPOSES_COMBINE_MODE ON→
    # 按句切建根（_sentence_bounds·与 observe 同切法·seg_idx 对齐）·OFF→整段单 span=原段级行为（bit-identical）。
    _scope_b_split = bool(getattr(gates, "COMPOSES_COMBINE_MODE", False))
    _flat_units: list[tuple[CollectedItem, int, list[str]]] = []
    for item in lang_probe:
        _spans = _sentence_bounds(item.tokens) if _scope_b_split else [(0, len(item.tokens))]
        for seg_idx, (_s, _e) in enumerate(_spans):
            _toks = list(item.tokens[_s:_e])
            if _toks:
                _flat_units.append((item, seg_idx, _toks))
    for item, seg_idx, tokens in _flat_units:
        h = Hasher(_DISC_LANG_SEED).h63("\x1f".join(tokens))
        root = ctx.concept_index.ensure(
            f"__disc_lang_{h}", space_id=ctx.space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # COMPOSES pre-build (审1 M1·mirror :3502-3511)：auto_discover skip 空 shape_signature roots
        # （structure_discover.py:1306-1307·shape_sig 须 EDGE_COMPOSES children）·idempotent query_from skip。
        if not ctx.edge_store.query_from(root[0], root[1], edge_type=EDGE_COMPOSES):
            record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
            for ti, tok in enumerate(tokens):
                tok_ref = ctx.concept_index.ensure(
                    tok, space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                ctx.edge_store.add(
                    space_id_from=root[0], local_id_from=root[1],
                    space_id_to=tok_ref[0], local_id_to=tok_ref[1],
                    edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
                    epistemic_origin=EPI_STRUCTURED, order_index=ti)
        roots.append(root)
        root_keys.append((id(item), seg_idx))

    # variable-length align (审1 M1·mirror :3521-3553)：异 token 数 roots → pairwise_fold consensus → 等长对齐根
    # （破同子数门 structure_discover.py:339）。同长（length set 单一）→ roots 不变走原路径（bit-identical）。
    # 退化（consensus 空/未全匹配/<K）→ roots 不变（变长不发现·诚实不纸面闭合·非 theater）。
    root_token_lens = [len(ctx.concept_graph.read_composes_tree(r)[0].get(r, []))
                       for r in roots]
    if len(set(root_token_lens)) > 1:
        from pure_integer_ai.cognition.process.lang_structure_align import (
            align_variable_lang_sequences)
        aligned_seqs = align_variable_lang_sequences(
            ctx.concept_graph, roots,
            concept_index=ctx.concept_index, space_id=ctx.space_id)
        if aligned_seqs is not None:
            new_roots: list[ConceptRef] = []
            for seq, _orig_root in zip(aligned_seqs, roots):
                h = Hasher(_DISC_LANG_ALIGN_SEED).h63(
                    "\x1f".join(f"{r[0]}:{r[1]}" for r in seq))
                align_root = ctx.concept_index.ensure(
                    f"__disc_lang_align_{h}", space_id=ctx.space_id,
                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                if not ctx.edge_store.query_from(
                        align_root[0], align_root[1], edge_type=EDGE_COMPOSES):
                    record_composes_attr(ctx.backend, ref=align_root,
                                         kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
                    for ti, ref in enumerate(seq):
                        ctx.edge_store.add(
                            space_id_from=align_root[0], local_id_from=align_root[1],
                            space_id_to=ref[0], local_id_to=ref[1],
                            edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
                            epistemic_origin=EPI_STRUCTURED, order_index=ti)
                new_roots.append(align_root)
            roots = new_roots
        # else: consensus 退化·roots 不变（变长不发现·诚实）

    # auto_discover_operators（mirror :3562·**body 天然 tally-free**·建 skeleton + cue_sig）。
    # sample-disjoint held-out 同 shape → 幂等 skip（training 已注册）→ newly_discovered 多为空（预期·S1 all_ops 搜补）。
    newly_discovered = auto_discover_operators(
        roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=ctx.space_id, source=SOURCE_BARE_TEXT)

    # label_realizes on newly discovered（mirror :3569-3579·gated REALIZES_MODE·外源 ConceptNet oracle·sound 无 D:11 写）。
    # sample-disjoint held-out → newly_discovered 空 → no-op（REALIZES 边训练期已建·idempotent）。novel-shape held-out
    # → newly_discovered 非空 → label_realizes 在新 skeleton 上建 REALIZES→REL_*（gated REALIZES_MODE 生产已翻）。
    if getattr(gates, "REALIZES_MODE", False) and newly_discovered:
        from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
        _rel_prims = ensure_relation_primitives(
            ctx.concept_index, ctx.backend, space_id=ctx.space_id)   # 幂等·确保 __REL_SUBSET__/__REL_CAUSES__ 存在
        label_realizes_is_a(newly_discovered, graph=ctx.concept_graph, edge_store=ctx.edge_store,
                            rel_primitives=_rel_prims, space_id=ctx.space_id)
        label_realizes_causes(newly_discovered, graph=ctx.concept_graph, edge_store=ctx.edge_store,
                              rel_primitives=_rel_prims, space_id=ctx.space_id)

    # ★ S1 fix（审1 严重·load-bearing）：build `_skel_by_sig` 搜 all_ops = training_lang_ops + newly_discovered。
    # sample-disjoint held-out skeleton = 训练已注册（lookup 命中）→ auto_discover 幂等 skip → newly_discovered 空·
    # 但 training_lang_ops 含训练期注册的 skeleton_ref（shape 匹配 held-out root）→ _skel_by_sig 命中 → map 填充。
    # gate COMPOSES_COMBINE_MODE OFF → 不建 map（lang_skeleton_by_item 不填 → INSTANTIATES 不 fire → measured=False·守 bit-identical）。
    skel_by_item_new: dict[tuple[int, int], ConceptRef] = {}
    if getattr(gates, "COMPOSES_COMBINE_MODE", False):
        all_ops: list[DiscoveredOperator] = list(training_lang_ops) + list(newly_discovered)
        # ★ Bug C2 修法（_skel_by_sig 键扩展·2审 APPROVE-WITH-CONDITIONS）：
        # 旧 _skel_by_sig.setdefault(shape, skel) 单值 first-wins·shape 对语言坍缩为长度·是/使/异 cue 同 shape 误绑。
        # 新 _ops_by_sig dict[shape, list[ConceptRef]]（C3）·多 skeleton 同 shape 收集 candidate list·
        # per-root 调 _aligns_to_skeleton（cue+LCA 维对齐）first-match（C5 tiebreak = all_ops 顺序·确定性）。
        from pure_integer_ai.cognition.process.structure_discover import _aligns_to_skeleton
        _ops_by_sig: dict[tuple[int, ...], list[ConceptRef]] = {}
        for _op in all_ops:
            _ops_by_sig.setdefault(
                tuple(_dim_shape_sig(ctx.concept_graph, _op.skeleton_ref)), []).append(_op.skeleton_ref)
        for _key, _root in zip(root_keys, roots):
            _sig = tuple(_dim_shape_sig(ctx.concept_graph, _root))
            _chosen: ConceptRef | None = None
            for _skel in _ops_by_sig.get(_sig, []):
                if _aligns_to_skeleton(ctx.backend, ctx.concept_graph, _root, _skel):
                    _chosen = _skel
                    break   # first-match（per-root cue+LCA 对齐·破 是/使 误绑）
            if _chosen is not None:
                skel_by_item_new[_key] = _chosen
            else:
                # S1 defensive surface（审1 严重 foot-gun·防 silent-veto）：held-out root 形状无匹配 skeleton。
                # 不 raise（novel-shape held-out 是合法场景·此时 newly_discovered 应含新 skeleton·若仍空=问题）。
                # log 到 stderr（不污染返参·caller observe 后再核 read_instantiates）。
                import sys
                print(f"[floor_orchestrator] WARN S1: held-out unit key={_key} "
                      f"root={_root} sig={_sig} 无匹配 skeleton（silent-veto risk·training_lang_ops="
                      f"{len(training_lang_ops)} newly_discovered={len(newly_discovered)}）",
                      file=sys.stderr)
    return roots, skel_by_item_new


def _measure_floor_pass(ctx: TrainContext, backend: StorageBackend,
                        graph: ConceptGraph) -> FloorActivation:
    """floor 端到端下游激活率 orchestrator（piece 3·mirror W6 `_run_simulated_offline_eval:1489` 隔离范式·drop generate·§9）。

    读 ctx.probe_corpus（W4 held-out 语言探针·从未进训练 observe/boot/discovery/H2/generate/base_freq·**W4 shadow 守
    sample-disjoint**）→ per probe item：(1) tally-free discovery 建/查 lang skeleton（S1 all_ops 搜·防 silent veto）
    → (2) W6-隔离 observe 建学树 struct_ref（observe fire INSTANTIATES via work_memory.lang_skeleton_by_item·建 token_seq
    observed input tokens）→ (3) build minimal OutputResult（parts=observed struct_refs·token_refs=graph.read_token_seq
    observed input token·**NOT** dispatch winner·**NOT** generate 输出·审2 MEDIUM-1 recognition measure）→ (4) piece 2
    measure_floor_activation 纯读（accumulation across probe items·审2 LOW-2）。

    **诚实 framing**：本函数测 **RECOGNITION** =「D:11 学习链 fired + held-out cue 位 observed input token 携
    training-learned D:11 匹配 skeleton rel」= 机制层预验（C-vs-L 真判别 + sample-disjoint + measured-guard·非 vacuous）。
    **非测**「generation activates / bonus 真驱动 / 学全 / 新词反推 / vocab-disjoint」（defer Phase E/F）·**非** empirical
    泛化率（真 ConceptNet held-out defer W4）。verdict 强度 = 「correspondence 学习机制在 held-out 产一致 D:11」·
    weaker-than-can_ween·机制层非 empirical 泛化。

    ★ **不变量锁**（审2 LOW-1 + piece 2 floor_measure.py:16-19）：本 orchestrator **禁止调 tally_cue_slot_matches**
    on held-out（_held_out_discovery_tally_free 天然 tally-free·auto_discover_operators body 无 tally caller）——
    否则 held-out_root 进 structure_match_count → (cue 词, held-out-instance) pairing 不 disjoint → activation 虚高 = theater。
    _run_emergence_hook（:485-487）training-only + ORACLE_PROMOTE_MODE OFF 时 + SHADOW unread by cue_rel_of PRIMARY 滤 → inert。

    **隔离论证（核心·mirror W6）**：observe 进同一 live graph（非拷贝·W6 :1515 probe 学树留 backend 既有范式）。
    held-out observe 写 COOCCURS/IS_A 等（observe 建边）·**但不写 D:11**（tally 省略·EDGE_RELATION_SIGNAL 仅 tally/promote
    写）。floor 读 D:11（training-learned）非 COOCCURS/IS_A。observe 跑 stage loop 后（training 已完·无前序训练污染·
    其余 verdict 锚已算完·held-out 边惰性·同 W6）。

    Returns FloorActivation（accumulation 聚合·审2 LOW-2·permille = Σactivated*1000//max(Σtotal,1)·mirror W6 :1581-1582）。
    早返 FloorActivation(measured=False) if (无 probe_corpus) OR (probe_set_disjoint=False·审2 LOW-3 defense-in-depth)
    OR (无 lang_probe)。gate caller 守 FLOOR_ACTIVATION_MODE（env-gated·默认 OFF→本函数不调→零副作用 bit-identical）。
    """
    # 审2 LOW-3：probe_set_disjoint 早返（defense-in-depth·非 blocker·anchor_heldout 已守 sample-disjoint）。
    if not ctx.probe_corpus or not getattr(ctx, "probe_set_disjoint", False):
        return FloorActivation(measured=False)
    lang_probe = [it for it in ctx.probe_corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens]
    if not lang_probe:
        return FloorActivation(measured=False)

    # S1 fix：training_lang_ops 源（load_discovered_operators 载全注册算子·含训练期发现的语言 skeleton·
    # load-bearing：sample-disjoint held-out 的 skeleton 由训练期 auto_discover 注册·此查 load 它）。
    training_lang_ops = load_discovered_operators(backend, space_id=ctx.space_id)

    # Step 1：held-out tally-free discovery + S1 all_ops map（_held_out_discovery_tally_free）。
    _roots, skel_by_item_new = _held_out_discovery_tally_free(
        ctx, backend, lang_probe, training_lang_ops)
    # S1 fix：populate work_memory.lang_skeleton_by_item（mirror training·caller observe 前 fill·
    # observe.py:204-208 读 work_memory 此 map → build_instantiates_edge fire）。setdefault first-wins（同 training）。
    # scope B：键 = (id(item), seg_idx) tuple（句级·observe (item_key,seg_idx) 对齐）。
    for _key, _skel in skel_by_item_new.items():
        ctx.work_memory.lang_skeleton_by_item.setdefault(_key, _skel)

    # Step 2：held-out observe（mirror W6 `_run_simulated_offline_eval:1548-1562`·observe 进 live graph·post-training 惰性）。
    # observe fire build_instantiates_edge（:204-208·COMPOSES_COMBINE_MODE ON + lang_skeleton_by_item 填）→ __seg_→skeleton
    # INSTANTIATES 真边·observe 建 token_seq（attach_token_seq :285·observed input token·gate DISPATCH_TOKEN_CHAIN_MODE ON）。
    from pure_integer_ai.cognition.understanding.observe import observe
    from pure_integer_ai.cognition.understanding.pronoun_features import lookup_pronoun_features
    from pure_integer_ai.cognition.understanding.sense_lookup_hook import make_sense_lookup
    sctx = _build_space_ctx(ctx)
    held_out_struct_refs: list[ConceptRef] = []
    # scope B：roots 现为句级（len > lang_probe）·故按 item 迭代（observe 内部按句 segment 产 struct_ref·
    # 逐 segment 读 lang_skeleton_by_item[(item_key,seg_idx)]）。_roots 仅诊断用（不再 zip item）。
    for item in lang_probe:
        # S1 defensive surface（审1 严重 foot-gun）：observe 前核 map 已填·否则 INSTANTIATES 全不 fire → silent veto。
        # scope B：键=(id(item),seg_idx)·核该 item 任一句 seg 进 map（observe 后逐 seg 读 (id,seg_idx)）。
        _nseg = len(_sentence_bounds(item.tokens))
        if not any((id(item), _si) in ctx.work_memory.lang_skeleton_by_item
                   for _si in range(_nseg)):
            import sys
            print(f"[floor_orchestrator] WARN S1: held-out item id={id(item)} "
                  f"无句 seg 进 lang_skeleton_by_item → INSTANTIATES 不 fire → read_instantiates=None "
                  f"(silent-veto·measured=False 风险)", file=sys.stderr)
        segments = _split_item_to_segments(
            item, backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index)
        if not segments:
            continue
        raw = InputPayload(
            segments=segments, source=item.source, stage=STAGE_TRAINING,
            modality=item.modality, lang=item.lang, domain=item.domain,
            weaning_phase=ctx.weaning_phase,
            item_key=id(item),
        )
        obs = observe(raw, sctx, concept_index=ctx.concept_index,
                      work_memory=ctx.work_memory,
                      pronoun_feature_lookup=lookup_pronoun_features,
                      sense_lookup=make_sense_lookup(ctx.backend, ctx.space_id))
        held_out_struct_refs.extend(obs.struct_refs)

    if not held_out_struct_refs:
        return FloorActivation(measured=False)

    # Step 3：build minimal OutputResult from observed struct_refs（审2 MEDIUM-1 drop generate·recognition measure）。
    # token_refs = observed input tokens（graph.read_token_seq(struct_ref)·observe attach_token_seq 写·gate ON）·
    # 非 dispatch winner·非 generate 输出·floor=recognition 诚实。
    parts = [OutputPart(unit=ref, token_refs=list(graph.read_token_seq(ref)))
             for ref in held_out_struct_refs]
    _floor_output = OutputResult(parts=parts)

    # Step 4：measure（piece 2 纯读·accumulation across probe items·审2 LOW-2·mirror W6 :1581-1582）。
    # lazy import（gate OFF 时 orchestrator 不调·零 import 副作用·守 bit-identical）。
    from pure_integer_ai.cognition.result.floor_measure import measure_floor_activation
    return measure_floor_activation(graph, _floor_output)



# ---- per-round 集合 runner（一轮多 item·收集 episodes） ----

def _run_round_batch(ctx: TrainContext, runner: RoundRunner,
                     items: list[CollectedItem], stage: int,
                     round_id: int) -> list[Episode]:
    """跑一轮（多 item·收集非 None episodes·observe-only 阶段返空 list）。"""
    eps: list[Episode] = []
    for i, item in enumerate(items):
        ep = runner.run_round(ctx, item, stage, round_id * 1000 + i)
        if ep is not None:
            eps.append(ep)
    return eps


# ---- 图度量（metrics 同源 D2 输入） ----

def _graph_size(ctx: TrainContext) -> int:
    return ctx.backend.count("concept_node")


def _edge_count(ctx: TrainContext) -> int:
    """边总数（pre_flight ② 内存代理·概念点+边总数=图资源·防超线性膨胀）。"""
    return ctx.backend.count("edge")


def _inject_base_freq(ctx: TrainContext, corpus: list[CollectedItem]) -> None:
    """阶段2 通识 base_freq 注入（experience_count·断奶前·observe 后 lookup 命中 token concept）。

    镜像 edge_store base_strength 范式（建概念时一次性写通识先验·reward 不调·断奶后退场镜像
    EPI_LLM_CONFIRM）。只注 token concept（surface=token 文本）·不注段 struct_ref（observe seg_label
    `__seg_/__prog_` 结构标签非 token·无频次语义）/ code / arith source。不改 observe /
    normalize_to_concept / ConceptIndex.ensure（保 observe 4 入口公共原语契约 §十一缺口#3）。
    Counter 返 int 纯整数合规·first-write-wins 幂等·sorted 确定序 bit-identical。
    """
    from collections import Counter
    from pure_integer_ai.storage.experience_count import record_base_freq
    if ctx.weaning_phase != WEANING_PRE:
        return   # 断奶后退场（镜像 EPI_LLM_CONFIRM·断奶后新概念无 base_freq 只靠 exp 自积累）
    surface_freq = Counter()
    for item in corpus:
        if item.modality in (MODALITY_CODE, MODALITY_ARITH) or not item.tokens:
            continue   # 代码/算术非 token concept / 空 tokens 跳过
        for tok in item.tokens:
            if tok:
                surface_freq[tok] += 1
    for surface, freq in sorted(surface_freq.items()):   # sorted bit-identical
        ref = ctx.concept_index.lookup(surface, ctx.space_id)
        if ref is None:
            continue   # lookup miss：未建 concept（结构标签/未 observe token）·不建·诚实 skip
        record_base_freq(ctx.backend, ref=ref, base_freq=int(freq))


def _anti_collapse_summary(eps: list[Episode]) -> dict[str, Any]:
    """防塌三柱验收汇总（致命5：anti_collapse_verify 生产 caller·pre_flight/主循环调·非 theater）。
    P0-3 决断（doc/重来_P0决断集_修正分析十三.md §四）：anti_collapse 已接此处·credit_sink 弃
    COOCCURS reward 落点=防塌柱①有意断(见 reward_propagate 落点③)·均非缺口。

    对非空 pr_vector 的 episode 跑 anti_collapse_verify（柱①②③ falsifiable）·汇总各柱通过率。
    空 pr_vector episode 跳过（dag_path 未跑·无 PR 向量可验·层1 闭合前提缺）。
    返 {verified, total, pillar1_ok, pillar2_ok, pillar3_ok, low_variance}。
    """
    from pure_integer_ai.cognition.result.anti_collapse import (
        anti_collapse_verify, integer_variance, THETA_VARIANCE)
    verified = 0
    p1 = p2 = p3 = low_var = 0
    for ep in eps:
        if not ep.pr_vector:
            continue   # 无 PR 向量·跳过（层1 闭合前提缺·不验）
        rep = anti_collapse_verify(ep)
        verified += 1
        p1 += int(rep.pillar1_ok)
        p2 += int(rep.pillar2_ok)
        p3 += int(rep.pillar3_ok)
        if integer_variance(ep.pr_vector) < THETA_VARIANCE:
            low_var += 1
    return {"verified": verified, "total": len(eps),
            "pillar1_ok": p1, "pillar2_ok": p2, "pillar3_ok": p3,
            "low_variance": low_var}


def _weaning_blockers(rep: Any) -> list[str]:
    """D1-D5/E2 未过闸门清单（诚实标注·不静默·§十一 #4-bis）。

    weaning_ready=False 时列出未过闸门·进训练日志（run_id 下）·驱动诊断与续训决策。
    """
    blockers: list[str] = []
    # W7 修：not rep.plateaued 是 dict truthiness（非空=True→not=False·恒漏 plateau 失败）·须 all(values)。
    # 既有 bug：D1 plateau 失败但 floors_met 过时 blockers 不标（不诚实）·W7 round_series=False 场景暴露。
    if not all(rep.plateaued.values()) or not rep.floors_met:
        blockers.append("D1_capability_plateau")       # 4 能力指标平台/下限
    if not rep.intervention_decreasing:
        blockers.append("D1_intervention_decreasing")  # 曲线① 方向性
    if not rep.retention_stable:
        blockers.append("D1_retention_stable")         # 曲线② 方向性
    if not rep.dependency_low:
        blockers.append("D1_dependency_low")           # 依赖度
    if not rep.neg_pathway_active:
        blockers.append("D2_neg_pathway_active")       # 负通路活跃
    if not rep.judge_source_independent:
        blockers.append("D3_judge_source_independent")  # 裁判源独立
    if not rep.probe_set_disjoint:
        blockers.append("D4_probe_set_disjoint")       # 探针集隔离
    if not rep.mode_b_prevalidated:
        blockers.append("D5_mode_b_prevalidated")      # Mode B 预验
    if not rep.e2_passed:
        blockers.append("E2_independent_production")   # 教师下线独立产出（最硬·当前永未过）
    return blockers


def _causes_coverage(ctx: TrainContext) -> int:
    """CAUSES 覆盖率（有 CAUSES 出边节点占比 ×1000·阶段2 门控）。"""
    from pure_integer_ai.storage.edge_types import EDGE_CAUSES
    nodes = ctx.backend.select("concept_node", where=None)
    if not nodes:
        return 0
    causes_from = {(r["space_id_from"], r["local_id_from"])
                   for r in ctx.backend.select("edge", where={"edge_type": EDGE_CAUSES})}
    covered = sum(1 for n in nodes
                  if (n["space_id"], n["local_id"]) in causes_from)
    return (covered * 1000) // len(nodes)


# ---- E7 pre-flight 放量门 ----

@dataclass
class PreFlightReport:
    """E7 pre-flight 验收报告（6 项全过才放量·守几百G不重训红线）。"""

    metrics_signal: bool = False       # ① 度量真有信号（图/CAUSES/导通率非全0非盲）
    mem_ok: bool = False               # ② 内存峰值<mem_hard_pct（轻量代理·真 mem 工程层 defer）
    reward_gate_ok: bool = False       # ③ reward gate 实际生效（judge 门否决/反传只 CAUSES）
    replay_coverage_ok: bool = False   # ④ replay 覆盖率≥阈值（E4 续训前置）
    cursor_resume_ok: bool = False     # ⑤ cursor resume 能跳已完成阶段（E8 续训机制）
    # ⑥ 防塌柱③ 探索压力（S12·9a·闭环证伪剩 D 墙前置）。柱③ 是唯一在"无显式失败"时 active 的柱
    # （① 结构 judge / ② 真负通路 在试跑 happy path 上 dormant·不算塌·anti_collapse.py:12）。
    # 故 collapse_ok 口径=柱③ 无失守（有 PR 的 episode 全柱③ OK·方差够 dormant OR 注入缓解）·
    # 非三柱全过（避 happy path 误报塌）。verified=0（全空 PR·dag_path 未跑）→ 退化放行 +
    # detail["collapse_degraded"]=True 诚实标（无 PR 可验·非趋平退化信号·由 ①/③ 门先拦）。
    collapse_ok: bool = False
    anti_theater_ok: bool = True       # ⑦ 反 theater 自我考核（#726 片2·层2锚点+层3反向回归·默认 True=未触发 passthrough·生产 caller 传 config+backend_factory 触发后真判·I-新闭环"旗标对自身失效残留"）
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (self.metrics_signal and self.mem_ok and self.reward_gate_ok
                and self.replay_coverage_ok and self.cursor_resume_ok
                and self.collapse_ok and self.anti_theater_ok)


def pre_flight(ctx: TrainContext, corpus: list[CollectedItem], *,
               rounds: int = PRE_FLIGHT_ROUNDS,
               runner: RoundRunner | None = None,
               replay_needed: list[tuple[int, tuple]] | None = None,
               config: "FormalTrainConfig | None" = None,
               backend_factory: "Callable[[], Any] | None" = None) -> PreFlightReport:
    """E7 pre-flight 放量门（小规模试跑 → 5 验收项·全过才放量·§十二 line903）。

    rounds 经验初值 50000（oracle 可调）·试跑 min(rounds, len(corpus)) 轮。
    失败=禁放量（修配置重试小规模）非"继续跑看看"。
    """
    validate_b1_b4()   # B1-B4 占位校验前置（防漂移）
    r = runner or DefaultRoundRunner()
    trial = corpus[:min(rounds, len(corpus))] if corpus else []
    eps: list[Episode] = []
    # A1：生产试跑 observe 须自产 CAUSES（同 formal_train·CUE_EXTRACTOR_MODE ON·致命3 残留·断奶后语言域源）。
    # 翻在此非 run_round_full（cue 在 split+observe 被读·run_round_full 保 gate-respecting 可测单元）·详见 formal_train A1 块。
    saved_cue = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        for rid, item in enumerate(trial):
            ep = r.run_round(ctx, item, STAGE3_REWARD, rid)
            if ep is not None:
                eps.append(ep)
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue

    rep = PreFlightReport()
    # ① 度量真有信号
    gsize = _graph_size(ctx)
    ccov = _causes_coverage(ctx)
    cond = (sum(1 for e in eps if e.reward > 0) * 1000 // max(len(eps), 1)) if eps else 0
    rep.metrics_signal = gsize > 0 or ccov > 0 or cond > 0
    rep.detail["graph_size"] = gsize
    rep.detail["causes_coverage"] = ccov
    rep.detail["conduction_rate"] = cond

    # ② 内存峰值（轻量代理：试跑后概念点+边总数 ≤ 每 round 预算×rounds·防超线性膨胀 OOM·
    #    stub ② 修：旧版硬编码 True·真 OS 级 mem_hard_pct 监控 defer 工程层·此为纯整 in-process 代理 falsifiable）
    peak_resource = _graph_size(ctx) + _edge_count(ctx)
    mem_budget = PRE_FLIGHT_MEM_BUDGET_PER_ROUND * max(len(trial), 1)
    rep.mem_ok = peak_resource <= mem_budget
    rep.detail["peak_resource"] = peak_resource
    rep.detail["mem_budget"] = mem_budget
    rep.detail["trial_rounds"] = len(trial)

    # ③ reward gate 实际生效（judge 产 veto 或 reward>0·非全 0 盲·stub ③ 修：删 cond>=0 恒真尾·须真产信号）
    has_veto = any(e.judge_veto_count > 0 or e.dead_end_count > 0 for e in eps)
    has_pos = any(e.reward > 0 for e in eps)
    rep.reward_gate_ok = has_veto or has_pos
    rep.detail["has_veto"] = has_veto
    rep.detail["has_pos_reward"] = has_pos

    # ④ replay 覆盖率（E4·教师续训前置）
    if ctx.teacher is not None and replay_needed:
        rep.replay_coverage_ok = check_replay_coverage(ctx.teacher, replay_needed)
    else:
        rep.replay_coverage_ok = True   # 无教师/无 needed → 放行（非续训场景）
    rep.detail["replay_needed_count"] = len(replay_needed or [])

    # ⑤ cursor resume 能跳（E8·机制验·跳已完成 skippable）
    st = CursorState(base_run_id="preflight", run_id="preflight")
    mark_completed(st, STAGE1_SKELETON, skippable=True)
    todo = cursor_resume(st, list(STAGES), skippable=SKIPPABLE_STAGES)
    rep.cursor_resume_ok = STAGE1_SKELETON not in todo and STAGE3_REWARD in todo
    rep.detail["cursor_todo"] = todo

    # ⑥ 防塌柱③ 探索压力（S12·9a·闭环证伪剩 D 墙前置：collapse_ok 进 passed 阻塞门）。
    # _anti_collapse_summary 对非空 pr_vector 的 episode 跑 anti_collapse_verify·汇总柱①②③ 计数。
    # collapse_ok 口径=柱③ 无失守（有 PR 的 episode 全柱③ OK）·非三柱全过——柱③ 是唯一在"无显式
    # 失败"时 active 的柱（①② 在试跑 happy path 上 dormant 不算塌·anti_collapse.py:12）·故只用柱③
    # 做放量阻塞判据。reward 阶段 EXPLORATION_MODE ON（run_round_full :339）→ dag_path 内注入 →
    # 趋平时柱③ OK；柱③ 失守（注入失败/EXPLORATION_MODE 关且趋平）= 趋平退化信号 → 禁放量。
    # verified=0（全空 PR·dag_path 未跑·如代码域语料）→ 退化放行 + 诚实标（无 PR 可验·非趋平信号）。
    ac = _anti_collapse_summary(eps)
    rep.detail["anti_collapse"] = ac
    if ac["verified"] == 0:
        rep.collapse_ok = True
        rep.detail["collapse_degraded"] = True    # 退化放行（无 PR 可验·由 ①/③ 门先拦）
    else:
        rep.collapse_ok = (ac["pillar3_ok"] == ac["verified"])
        rep.detail["collapse_degraded"] = False

    # ⑦ 反 theater 自我考核（#726 片2·层2锚点+层3反向回归·I-新闭环"旗标对自身失效残留"）。
    # 机制在测试中真活（test_capability_anti_theater 验锚点真造 FAIL + mutation 验判据敏感·非死 theater）·
    # 但生产 caller 从未传 anti_theater=True -> 旗标对自身失效残留（D6 病更深层）。pre_flight 是放量门·
    # 天然反 theater 自检点：caller 传 config+backend_factory -> 跑锚点（corpus 层注入坏语料·独立 backend·
    # 验期望维度判 FAIL 非 dead theater）+ 反向回归（8 维判据可证伪+NE 守恒）-> anti_theater_ok 真判。
    # 默认 None（既有 9+ caller 不传）-> skip·anti_theater_ok=True passthrough·守 bit-identical（既有测零翻）。
    # 主入口 :2398 传 config+lambda:DictBackend() -> 生产 pre_flight 触发自检·fail 禁放量（passed 含 anti_theater_ok）。
    if config is not None and backend_factory is not None:
        from pure_integer_ai.experiments.capability_exam import (
            run_anti_theater_anchor, run_reverse_regression)
        _pf_runner = runner or DefaultRoundRunner()
        _anchors = run_anti_theater_anchor(config, backend_factory, runner=_pf_runner)
        _regressions = run_reverse_regression()
        rep.anti_theater_ok = (all(a.passed for a in _anchors)
                               and all(r.passed for r in _regressions))
        rep.detail["anti_theater_triggered"] = True
        rep.detail["anti_theater_anchor"] = [
            a.to_dict() for a in sorted(_anchors, key=lambda a: a.name)]
        rep.detail["anti_theater_regression"] = [
            r.to_dict() for r in sorted(_regressions, key=lambda r: r.dim)]
    else:
        rep.detail["anti_theater_triggered"] = False
        rep.detail["anti_theater_note"] = (
            "反 theater 自我考核未触发（caller 未传 config/backend_factory·#726 片2 生产 caller opt-in·"
            "机制在 test_capability_anti_theater 真活·此 caller 跳过自检）")
    return rep


# ---- formal_train 主入口 ----

@dataclass
class FormalTrainConfig:
    """formal_train 配置（run 标识 + 续训 + 轮次 + dump）。"""

    run_dir: str
    run_id: str
    rounds_per_stage: int = 4
    resume: bool = False
    base_run_id: str | None = None      # 续训 base 终 dump run_id（E1）
    metrics_path: str | None = None     # None → <run_dir>/<run_id>/metrics.jsonl
    replay_needed: list[tuple[int, tuple]] | None = None   # 续训 replay 覆盖率前置（E4）
    # E7 pre-flight 放量门接通生产主入口（S12 follow-up·破纸面闭合：collapse_ok 已进 PreFlightReport.passed
    # 但 caller 已接通（主入口 :2386 if config.pre_flight -> pre_flight trial·非"零调 theater"·旧注 stale 已纠）。
    # 默认 OFF 守 bit-identical（既有测零翻·pre_flight 块 if 外跳过）。ON 时 boot+discovery 后 snapshot
    # 生产 ctx → pre_flight trial（STAGE3 reward 路径·representative）→ fail raise 禁放量 → rollback 5 状态
    # （backend._data + _id_pool + ConceptIndex._index/_loaded_spaces + work_memory）→ stage loop 不变。
    # DictBackend + SQLiteBackend 均支持（snapshot/load_snapshot·§施工序 4.2-1 解 DictBackend-only）。
    pre_flight: bool = False
    pre_flight_rounds: int | None = None   # None → PRE_FLIGHT_ROUNDS（50000 经验值·oracle 可调·生产 caller 须自标小值）
    # 序列7：dump 含 composes_attr（算子 payload·ATTR_OPERATOR_DEF/ARITY/ORIGIN + 结构 opcode/operand/
    # immediate/store_target·跨 run load 后 read_composes_tree 重建 5 dict 须它）。cursor.DUMP_TABLES
    # 保持纯核心表不动（守 core/extension 分离·test_cursor_dump_tables_cover_core 仍过）·训练驱动层
    # 自决 round-trip 表集。load_run table-agnostic 自动还原（无须配套）。
    # §8.7-洗：dump 含 op_confidence（算子置信度 sn/tn/strength·跨 run 累积·recognize 择优读）。
    # 阶段1：dump 含 experience_count（概念维经验计数 base_freq/e_sn/e_tn·跨 run 累积·attractor 词终止
    # 读 effective_freq·阶段2 feed 写）+ chapter_seq_table（篇章结构序·段 struct_ref 章节标记·跨 run 还原）。
    # 刀5 件5：dump 含 selection_pref_count（选择倾向共现统计·跨 run 累积·observe 写 sp_tn·S4 PR 读）。
    # B6 指代维：dump 含 pronoun_resolution_count（指代消解 pair-key 台账·跨 run 累积·resolve 自消费读 pr_tn
    #   加候选分·H-新对称性修回：selection_pref_count 已 dump 而 pronoun_resolution_count 漏->续训后 pr_tn/pr_fn
    #   归零->resolve 自消费读冷启动不加候选分->跨 run 累积意图破·镜像 selection_pref_count 加表修对称）。
    # 缺口#1 主导度闸：dump 含 position_hist（位置直方图·UNGATED 每 token 写·consumer dominant_role->
    #   position_histogram·H-新对称性修回：续训后归零->主导度闸退化全 SUBJECT 兜底->"去 SVO 写死"续训后失效）。
    # 刀6 件7：dump 含 sense_candidates（多义 sense 候选台账·跨 run 累积·boot 种 base_count + observe 写 sc_tn·
    #   理解侧 recognize clone 读·断奶后 #479 墙突破时消费）。
    # §七 modality_subspace：dump 含 abstract_mark（节点抽象归属标记 modality/lang/domain/topo·
    #   observe 写 MARK_MODALITY + refers_to 写 MARK_LANG/MARK_DOMAIN·跨 run 还原·path B 迁移·守不污染节点列）。
    dump_tables: tuple[str, ...] = DUMP_TABLES + (COMPOSES_ATTR_TABLE, OP_CONFIDENCE_TABLE,
                                                   CONCEPT_IDENTITY_TABLE,
                                                   EXPERIENCE_COUNT_TABLE, CHAPTER_SEQ_TABLE,
                                                   SELECTION_PREF_COUNT_TABLE,
                                                   PRONOUN_RESOLUTION_COUNT_TABLE,
                                                   POSITION_HIST_TABLE,
                                                   SENSE_CANDIDATES_TABLE,
                                                   ABSTRACT_MARK_TABLE,
                                                   CONCEPT_CORRESPONDENCE_TABLE)   # P0a：码点对应（续训保留文本）
    # #723 G 归因：collect_episodes=True 时主循环收集全 episode 列表挂 result.episodes（G_meta 5 字段活·
    # judge/verify/task-driven 三路）。harness 跑考核时开·生产训练默认 OFF（防大 corpus episodes 爆内存）。
    # 默认 False 守 bit-identical（既有测零感知·result.episodes default 空 list·to_json 不序列化 episodes）。
    collect_episodes: bool = False
    # W2 断奶训练 mock POST 注入：weaning_phase 默认 WEANING_PRE 守 bit-identical（既有测零翻）。
    # WEANING_POST 时 _run_verify_round 走 cross-verify POST 分支（:578·Mode B 自锚·§7.6）。
    # 镜像 pre_flight/collect_episodes 范式（config 层注入·:1592 后设 ctx.weaning_phase）。
    # 诚实：mock POST 非真断奶（weaning_ready 仍 False·非 rep.ready 切换 :2034·E2 整体仍 False·
    # teacher_offline defer W6 / probe_input_novel defer W4·W2 只验第三条件算术域就位）。
    weaning_phase: int = WEANING_PRE
    # W3 D3 独立裁判分离：judge_source_id 默认 None 守 bit-identical（既有测零翻·build_judge_fn j_sid
    # 回落 teacher_sid→sources_disjoint 同源 False）。caller 传独立 source_id（与 teacher.source_id 不相交）
    # → D3 通用路径过（为 W8 语言域独立裁判铺路）。镜像 weaning_phase 范式（config 层注入·:1599 后设 ctx）。
    # 诚实：算术域绕 judge（_run_verify_round:374）不走通用路径·算术域 D3 走 judge_source_independent_arith
    # 判定接口（W7 接全）·W3 只建通用机制 + 算术域判定接口·weaning_check D3 算术域仍 False。
    judge_source_id: int | None = None
    # W4 D4 探针集采样：probe_holdout 默认 0 不切 corpus（bit-identical·既有测零翻）。
    # caller 传 >0 → formal_train 主入口 :1611 切 corpus 末尾 N 作 held-out probe（不喂 boot/discovery/H2/
    # stage/generate/base_freq 全部下游·shadow corpus=training subset）→ ctx.probe_set_disjoint=True（D4 过）。
    # probe_version 探针版本号（默认 0·caller 派生自 run_id 或显式传·守几百 G 不重训·bit-identical 可复现）。
    # 诚实：D4 域无关（corpus 切分·不依赖 judge_fn/teacher）→ 算术域走通用 ctx track 过 D4·不似 D3 须判定接口。
    # holdout_retention 真度量 defer W6（模拟退场 eval·naive fresh-compile 恒 1000 theater 故 defer）。
    probe_holdout: int = 0
    probe_version: int = 0
    # W5 D5 Mode B 预验台账：calibrate_mode_b 默认 False 守 bit-identical（既有测零翻·台账空→
    # mode_b_prevalidated(backend) 返 False→D5 blocker·同既有行为）。caller 传 True 时 stage4 末
    # （:2052 所有 D-check 前）跑 _run_calibration_phase：WEANING_WINDOW_ROUNDS 轮并行 Mode A vs B 评估
    # → record_calibration 真写台账 → :2068 mode_b_prevalidated(backend) 接非空台账真判定。
    # Mode A = vm_proof execute 学树 vs spec.expected（静态整数·非 live teacher）/ Mode B = cross_verify_pair
    # （root_a observe 学树 × root_b build_composes_from_arith 二次独立建·异 shape R6 守·纯 VM 零 judge/teacher）。
    # 诚实：D5 域无关（同 D4·不似 D3·无须判定接口）·calibration 独立台账（不写 episode/不染 op_confidence/
    # 不进 conduction_rate）·flat trend=不回升=通过（MUTABLE_MONOTONE·FLOOR_MODE_B=500 守低通过率平台）·
    # stable≠correct（墙内弱·两路径一致≠对错·D5 既有边界）。
    calibrate_mode_b: bool = False
    # W6 E2 模拟退场 eval：simulate_offline_eval 默认 False 守 bit-identical（既有测零翻）。caller 传 True 时
    # stage4 末（weaning_check 之前·预验非后验·解 teacher_offline 循环依赖）跑 _run_simulated_offline_eval：
    # 读 ctx.probe_corpus（W4 held-out 探针·W6 首个 reader·须配 probe_holdout>0）→ observe 探针建学树 root_a +
    # build root_b 参树 → cross_verify_pair（零教师自锚·VM 执行值）→ 采 holdout_retention 真值 + 设 ctx.e2_eval_passed
    # （算术域三条件 and）→ 路径 B（:2185 域特化分支·teacher=None 读 ctx / else 读 e2_execution_ready() defer W8）→ E2 算术域过。
    # 算术域 teacher=None 天然退场（架构事实·无 recording/replay/GT·无 mode 可翻）·语言域真翻 MODE_OFF + 恢复 defer W8。
    # 诚实：E2 单闸门过非真断奶（weaning_ready 仍 False·D1-D5 defer）·算术域 fixture 同源 trivial
    # （probe/training 都 square n²→cross_verify 恒 agree→retention 恒 1000·真泛化保持 defer W8 真语料）。
    simulate_offline_eval: bool = False
    # W7 断点6 per-round weaning_series recording：weaning.py:173 docstring"window_rounds=4 runs"= 设计 per-run·
    # 当前 record_round per-stage 调（observe-only stages 产 0 混入）→ [0,0,X,X] 假跳变→plateau 永假（实现 bug）。
    # True 时 record_round 移内循环 per-round（每 batch_eps 一次·promote/oov 仅 stage4 末轮）→ 8 entries·
    # 末4 全 stage3/4 verify 同值 flat→_max_recent_increment=0<THETA_PLATEAU→plateau True。默认 False per-stage bit-identical。
    weaning_round_series: bool = False
    curriculum_active_relations: frozenset[str] | None = None   # #1143 课程增量 boot：None=全 load（bit-identical·既有行为）·frozenset=只该集关系 boot（T-L0~L6 stage-by-stage 有序学习·镜像 arith S1-S8）


@dataclass
class GeneralizationSummary:
    """序列3-min 验证半闭环汇总（识别 → vm_proof 验泛化·§8.7 反 theater + 学到能力证据）。

    发现骨架从发现集学到 → 识别 held-out 新输入（READ）→ vm_proof 独立验骨架绑参复现新输入值。
    verified/total_held_out = 泛化率（×1000·学到的能力覆盖多少 held-out 新输入·直接量化"学到能力"）。

    反 theater：识别 = 结构对齐（_align_walk）·vm_proof = VM 执行比对（execute_composes_value）·两路独立
    计算。**诚实定位**（对抗审计·勿过判）：对正确识别·骨架与输入结构同构 → 同值是构造性预期（非惊奇交叉验证）·
    vm_proof 真"牙"=抓获 PARAM 阅读序错位 / 编译发散 / shape 漏判结构异配（probe：SUB 错参 -47≠43 不 verified）。
    重执行本身 = 真 READ+应用消费（非 theater·非死写·识别产物 recognitions 现被 vm_proof 真消费·解 terminal 边界）。
    生成侧洗净循环反馈半闭环（§8.7-洗·2026-07-03 done）：本函数验结果现写算子置信度（op_confidence
    sn/tn/strength）→ recognize_operators 择优读（滤非泛化算子=洗净）·解 recognitions terminal·反 theater 半环。
    生成侧全环（generate.py 读置信度·OutputModel 路径填槽消费骨架）= 独立大切片 defer（须独立设计 pass）。
    **【2026-06-30 证伪·3 对抗智能体】** 字面机制当前架构不可达（generate.py L6→execute_composes_value L7
    向上违单向 + STRUCT_BIND 跨模态桥 VF defer 零 caller + 算术骨架无语言 surface）·算子域闭合环已由本半环
    （recognize↔verify+vm_proof+op_confidence）完成·vm_proof 是骨架执行值真消费者。"生成侧全环"对算术模态
    是伪需求·须 STRUCT_BIND VF 落地后才有意义。详见 doc/重来_结构发现设计补充.md §8.7-洗-证伪。
    """
    total_held_out: int = 0   # 留出的 held-out 新输入总数（识别候选池·= len(recognize_roots)）
    recognized: int = 0       # 命中已学骨架的 held-out 数（recognize_operators 产）
    verified: int = 0         # 命中中 vm_proof 验过（骨架绑参==输入值·两路独立）的数
    expected_verified: int = 0   # S7 相0 钥匙③：教师标定比对命中数（recognize 命中骨架 ref==item.expected_skeleton·断奶前教师路径·POST 退场·非 vm_proof）

    @property
    def rate_permille(self) -> int:
        """泛化率 ×1000（verified / total_held_out·算术域 vm_proof 口径·total=0→0·纯整数无浮点）。"""
        return (self.verified * 1000) // max(self.total_held_out, 1)

    @property
    def lang_rate_permille(self) -> int:
        """S7 相1 钥匙③：语言含义命中率 ×1000（recognized / total_held_out·渐近判据·total=0→0·纯整数）。

        区别算术域 rate_permille（verified vm_proof 口径）：语言不可 vm_proof（钥匙③墙）·相1 用 recognized
        （recognize 结构对齐命中数）。渐近判据·非闭式真理。
        **消费者诚实边界**：相1 测试断言 + result.lang_generalization 字段暴露 + capability_exam.project_lang_measures
        observability 报告读取（#1041 构造③·单向 tap·**非闭环消费者**·无 decision/threshold/feedback）。recognize 择优读
        op_confidence（rate·已落）属**相0 半环**消费者·非本 property 消费者。本 property 与相0 op_confidence 半环当前
        **无机制耦合**。下游闭环消费者（metrics 显式读取 / weaning_ready 判据接线）仍 defer（相1 计算器先行·非纸面闭合）。
        """
        return (self.recognized * 1000) // max(self.total_held_out, 1)


@dataclass
class GenerateSummary:
    """§8.7-全 生成侧 task-driven L8 episode 汇总（任务→选算子→执行骨架→验·§8.7-全）。

    **认识论增量（审计降级后·Mode A 构造性·非"外真非传递"）**：半环（§8.7-洗 done）测**自洽**——
    skeleton(recognized_params)==input()（骨架复现输入程序值·学生==学生·传递必然对正确算子）。
    task-driven 测**新 args 泛化探针 + 生成侧函数应用姿态**——skeleton(新 task args)==expected（骨架对新
    任务输入产答案·args 非学习输入→产新值非记忆复现·生成姿态 call 算子为函数 vs 半环识别姿态 align 程序）。
    **Mode A 构造性**：skeleton 派生自输入程序·故 expected=正确答案时 skeleton(args)==expected 构造性必然
    （传递经 skeleton 起源·同半环牙·抓获 PARAM 序错/编译发散/shape 异配·无新牙）·**非"外真非传递"**。
    真非传递外真 = Mode B（异算法闭式 O(1) vs 迭代 O(n)·断奶后 defer·§8.7-全 前置/范围）。

    total_tasks : 任务总数（arith_specs 总数·每 CodeSpec=(input_args,expected) 一任务）。
    selected    : 选到算子的任务数（arity 匹配候选·滤 tested-never-verified 后有候选）。
    verified    : 验过数（skeleton(input_args)==expected·Mode A 构造性·output.parts 非空⟺verified）。

    反 theater 4 锚点（审计后·①④ PASS·②③ 降级/修）：①行为真变（多候选读置信度择优·置 sn=0 产出变）②
    新消费（skeleton(新 args) vs expected·新 args 非学习输入·**Mode A 构造性·真非传递留 Mode B**）③下游读者
    （OutputResult.parts→metrics generate_verified 计数真读·**审计必修：metrics 读 parts 非 reward·parts 非空⟺
    verified·否则=(Z) theater 重引入**）④拒坏选好（不 fit task→tn++·fit→sn++·择优选 fit）。
    诚实边界：①单候选构造性必过（信号薄·同半环）·多候选信号实·信号强度依赖语料覆盖非墙；
    ②mul/square 不可分（置信度正交于变量同一性判别器·Half B arity 区分 arity·同 arity 同形仍不可分）；
    ③stable≠correct（VM 跑通≠语义对·结构正确非语义理解·接地墙算术版·不阻塞）；
    ④防双计=caller 责任（task 输入须 ≠ recognize held-out 程序·机制不强制·rate 不受影响·§8.7-全 F3）；
    ⑤闭环范围=生成侧 task-driven 探针（Mode A 构造性）+ 半环（自洽）= 生成侧 wash 环（对本模态·Mode A）·
    **真外真全环须 Mode B**·generate.py 字面路径仍须 STRUCT_BIND(VF)·独立后续。
    """
    total_tasks: int = 0
    selected: int = 0
    verified: int = 0
    # #1124 symbolic 子计数（additive·0=CI 无 symbolic specs bit-identical·formal_train _run_task_driven_generate 传）：
    # xform_verified=transform 规则 cross-verify verified 数（S5-S7）·inv_verified=inverse 关系 B∘A 还原 verified 数（S8）。
    # capability_exam.project_symbolic_measures → CapabilityReport.math_measures（反 theater：symbolic 学习可见非 invisible）。
    xform_verified: int = 0
    inv_verified: int = 0
    # #1124 M-1：symbolic 分母（spec §四 4.2 symbolic_cross_verify_rate permille 用）。
    xform_total: int = 0
    inv_total: int = 0

    @property
    def rate_permille(self) -> int:
        """验过率 ×1000（verified / total_tasks·total=0→0·纯整数无浮点）。"""
        return (self.verified * 1000) // max(self.total_tasks, 1)


@dataclass
class FormalTrainResult:
    """formal_train 产出（阶段完成集 + 最终度量 + dump spaces + 断奶报告）。"""

    run_id: str
    stages_completed: list[int] = field(default_factory=list)
    stages_skipped: list[int] = field(default_factory=list)
    final_metrics: StageMetrics = field(default_factory=StageMetrics)
    dump_spaces: list[int] = field(default_factory=list)
    weaning_ready: bool = False
    weaning_blockers: list[str] = field(default_factory=list)   # D1-D5/E2 未过闸门（诚实标注·不静默）
    # #1143 语言域统计层断奶（5判据+5锚点·非 can_ween·weaker-than-can_ween·与 weaning_ready=can_ween 并列独立 verdict）。
    # gate STATISTICAL_WEANING_MODE（默认 OFF·CI bit-identical）·excluded_gates 复述 E2/D5/D3 防 theater。
    statistical_weaning_ready: bool = False
    statistical_weaning_report: Any = None    # StatisticalWeaningReport（measured-guard 守·未建测量 fadeout/heldout 不过）
    weights: JudgeWeights = field(default_factory=JudgeWeights)
    collapse_summary: dict[str, Any] = field(default_factory=dict)   # 防塌三柱验收汇总（致命5 生产 caller）
    discovered_operators: list = field(default_factory=list)   # 序列6-min 发现算子（DiscoveredOperator·可观测+反theater锚点·§八.6）
    recognitions: list = field(default_factory=list)   # 序列3-min 识别（Recognition·held-out 新输入命中已学骨架·生产期READ消费·§八.3）
    generalization: GeneralizationSummary | None = None   # 序列3-min 验证半闭环（vm_proof 验泛化·识别产物真消费·反theater·§8.7·算术域）
    lang_generalization: GeneralizationSummary | None = None   # 钥匙①语言结构发现（S3·verified=0 语言不可 vm_proof 钥匙③墙·recognized=命中骨架数·concept_binding 件5）
    alias_edges_seeded: int = 0   # #1041 构造④：P0b 跨语言 PURE_ALIAS 桥 boot 种边数（bootstrap_alias_edges 返·0=无 alias_facts→CI bit-identical·判据⑤跨语言汇聚 observability 信号）
    number_edges_seeded: int = 0   # language-grounding piece 1：数字词接地 boot 种边数（bootstrap_number_grounding 返·0=无 number_facts→CI bit-identical·对称 alias_edges_seeded·W8 语言域断奶 observability 信号·消费者待接 #941）
    mereology_edges_seeded: int = 0   # #1119 数据补全：T-L1d 部分-整体 EDGE_MEREOLOGY boot 种边数（bootstrap_mereology_edges 返·0=无 mereology_facts→CI bit-identical·对称 alias/number·W8 observability 信号·消费者待接 #941·reader 待接线）
    antonym_edges_seeded: int = 0   # #1119 数据补全：T-L1e 反义 EDGE_ANTONYM boot 种边数（bootstrap_antonym_edges 返·0=无 antonym_facts→CI bit-identical·对称 mereology/alias/number·reader 待接线·W8 observability 信号）
    similar_edges_seeded: int = 0   # #1132 数据补全：T-L1c 近义 EDGE_SIMILAR boot 种边数（bootstrap_similar_edges 返·0=无 similar_facts→CI bit-identical·对称 antonym/mereology/alias/number·机制全在 #898·reader dispatch_slot 在·W8 observability 信号）
    abstract_is_a_edges_seeded: int = 0   # #1133 数据补全：抽象→EDGE_IS_A 泛化 boot 种边数（bootstrap_is_a_edges source=SOURCE_CHINESE_KB 返·0=无 abstract_facts→CI bit-identical·异 ConceptNet is_a 刀0·抽象=IS_A 从始至今核心·abstraction.py LCA/祖先图 enrich·W8 observability 信号）
    generate: GenerateSummary | None = None   # §8.7-全 生成侧全环·task-driven L8 episode（外真半·任务→选算子→验·反theater·§8.7-全）
    pre_flight_report: PreFlightReport | None = None   # E7 放量门报告（config.pre_flight=True 时填·observability 非阻塞判据·S12 follow-up·破纸面闭合）
    # #723 G 归因：collect_episodes=True 时主循环收集全 episode 列表（language judge + Mode B verify +
    # Mode A task-driven 三路·G_meta 5 字段活·harness 跑考核时开）。**to_json 不序列化**（capability_exam
    # CapabilityReport.to_json 守·Episode 嵌套 dict/list sort_keys 不确定·守 bit-identical）。生产训练默认
    # OFF（config.collect_episodes=False·防大 corpus episodes 爆内存）+ 既有测零感知（default 空 list）。
    episodes: list = field(default_factory=list)
    probe_set: ProbeSet | None = None   # W4 D4 留出探针集（config.probe_holdout>0 时 formal_train 主入口建·版本化·W6/caller/test 可查·默认 None）
    holdout_retention: int = 0   # W6 E2 模拟退场 eval 采的探针保持率真值（默认 0 bit-identical·cross_verify 通过率×1000·D1 曲线②度量·真泛化 defer W8）


def formal_train(config: FormalTrainConfig,
                 corpus: list[CollectedItem], *,
                 backend: StorageBackend,
                 teacher: Any = None,
                 runner: RoundRunner | None = None,
                 weights: JudgeWeights | None = None,
                 metrics: MetricsCollector | None = None) -> FormalTrainResult:
    """五阶段正式训练主入口（§十二最优路径 + --resume 续训 + H2 + 终 dump）。

  度量门控合格才进下阶段（stage_metric_gate·防缺防超喂）。
  --resume：load_run(base) + cursor_resume stage-skip + check_replay_coverage（E1/E4/E8）。
  阶段3：H2 小批量标定权重 → 开全量 reward。阶段4：promote 三重 + 断奶判据。
  终 dump（dump_run·per-space·新 run_id·E1 权威 base）。
    """
    assert_no_float(config.rounds_per_stage, _where="formal_train.rounds_per_stage")
    ctx = make_train_context(backend, teacher=teacher, weights=weights)
    ctx.weaning_phase = config.weaning_phase   # W2 mock POST 注入（默认 PRE 守 bit-identical·WEANING_POST 走 cross-verify :578）
    ctx.judge_source_id = config.judge_source_id   # W3 D3 独立裁判注入（默认 None 守 bit-identical·caller :463 传 build_judge_fn）
    # W4 D4 探针留出 + 隔离判定 + 版本化（默认 probe_holdout=0 不切·bit-identical）。插入点在 boot/discovery/H2/
    # stage/generate/base_freq 全部下游之前——shadow corpus 覆盖全部下游消费（probe 从 boot seeding + discovery
    # held-out 池 + H2 + observe + generate + base_freq 全排除·正确·非仅 observe）。D4 域无关（corpus 切分·不依赖
    # judge_fn/teacher）→ 算术域走通用 ctx track 过 D4·不似 D3 须 judge_source_independent_arith 判定接口。
    training_corpus, probe_corpus = _split_holdout(corpus, config.probe_holdout)
    if probe_corpus:
        from pure_integer_ai.teacher.probe_set import make_probe_set, is_disjoint, ref_from_signature
        probe_refs = frozenset(ref_from_signature(_item_sig(it)) for it in probe_corpus)
        training_refs = [ref_from_signature(_item_sig(it)) for it in training_corpus]
        ctx.probe_set = make_probe_set(config.probe_version, probe_refs)
        ctx.probe_corpus = probe_corpus              # W6 模拟退场 eval 读（held-out 不喂任何训练消费）
        ctx.probe_set_disjoint = is_disjoint(ctx.probe_set, training_refs)
        corpus = training_corpus                      # shadow：全部下游消费 training subset（probe 不进 observe/boot/discovery/H2/generate/base_freq）
    r = runner or DefaultRoundRunner()
    mpath = config.metrics_path or os.path.join(
        config.run_dir, config.run_id, "metrics.jsonl")
    own_metrics = metrics is None
    mc = metrics or MetricsCollector(mpath)

    # ---- --resume 续训（E1/E4/E8） ----
    state = CursorState(
        base_run_id=config.base_run_id or config.run_id,
        run_id=config.run_id,
    )
    todo_stages = list(STAGES)
    if config.resume and config.base_run_id is not None:
        load_run(backend, config.run_dir, config.base_run_id)   # E1 终 dump base
        # E8：载入 base run 的 cursor state（已完成阶段集·skippable 跳过）
        base_state = _load_cursor(config.run_dir, config.base_run_id)
        if base_state is not None:
            state.completed = set(base_state.completed)
            state.non_skippable = set(base_state.non_skippable)
        # E4 replay 覆盖率前置（未达标禁续训防 miss→None 静默降级）
        if teacher is not None and config.replay_needed:
            if not check_replay_coverage(teacher, config.replay_needed):
                raise RuntimeError(
                    "续训前置失败：teacher replay 覆盖率未达标（E4·禁续训防静默降级）")
        # E8 stage-skip（跳已完成 skippable·非 skippable 保留须重标 H2）
        todo_stages = cursor_resume(state, list(STAGES), skippable=SKIPPABLE_STAGES)

    result = FormalTrainResult(run_id=config.run_id, weights=ctx.weights)
    result.probe_set = ctx.probe_set   # W4 D4 探针集 expose（config.probe_holdout>0 时主入口建·版本化·W6/caller/test 可查）
    # 语料相关 KB vocab（perf fix·doc/重来_语料相关KB过滤_2026-07-16）：KB vocab-edge（is_a/abstract/mereology/
    # antonym/similar/alias）resolve 后过滤·只留 ≥1 surface 在语料 vocab 的 pair。全量 KB 对 656-paragraph 语料
    # 84-99.5% out-of-corpus ballast → boot 95s + 训练图 660k 边 7x 慢。语料相关过滤随语料 scale（非 hack·非截断）。
    # causes_coverage ②结构 permille 分母稀释是 capability_exam:545 文档已知（FAIL≠结构破裂·reward delta 不变）。
    # **bit-identical**：CI 无 ZERO_AI_LOCAL_DIR→resolve_*_facts 返 []→filter 空 list 返空（filter_pairs_to_vocab
    # `not pairs` 短路）·零行为变。空语料 vocab→filter 返原 pairs（短路·守 bit-identical）。
    # #1143 课程增量 boot helper：curriculum_active_relations None=全 load（bit-identical·既有行为）·
    # frozenset=只该集关系 boot（stage-by-stage 有序学习·镜像 arith S1-S8）。
    def _rel_on(rel: str) -> bool:
        return config.curriculum_active_relations is None or rel in config.curriculum_active_relations
    _kb_vocab = corpus_relevant_vocab(corpus)
    # 刀0 接 IS_A 源（boot 时种 EDGE_ISA·解锁 Interp2 生产路径·doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀0）：
    # ancestor_map 在 lang 发现（下文 :952 _discover_and_recognize_lang_structures）内部自建·读「调用时刻」
    # space 全部 EDGE_ISA 边·**早于** stage loop observe（:978+）·故 observe 经 CUE（来源②）建的 IS_A 边在本
    # run lang 发现时刻还不存在。须 boot 时种（来源① ConceptNet 本地文件·E10·EPI_STRUCTURED·外部数据非 core enum·
    # 守「不写死」）。按 corpus 唯一 lang 集循环 resolve_is_a_facts（env ZERO_AI_LOCAL_DIR/is_a_facts_{lang}.txt）。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 零副作用返 0 → ancestor_map 空
    # → bit-identical 守回归**（bootstrap_is_a_edges 空 pairs 首行短路·绝不调 ensure/query_from/build）。
    # resume 路径：load_run（:923）已还原 base run 边 → boot 再种走 query_from 幂等 skip（同源同三元组）·不 corrupt。
    for _lang in {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}:
        _pairs = filter_pairs_to_vocab(resolve_is_a_facts(_lang), _kb_vocab) if _rel_on("is_a") else ()
        if _pairs:
            bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, _pairs,
                                 space_id=ctx.space_id)
    # #1133 抽象→IS_A 泛化 boot 种（ChineseSemanticKB 抽象关系库·specific→general·抽象=IS_A 非新 edge·纠偏撤回 EDGE_ABSTRACT）：
    # abstraction.py(build_isa_ancestor_map/LCA/set_lca)+EDGE_IS_A+bootstrap_is_a_edges 全在（从始至今核心）·本块补第二 IS_A 源（异 ConceptNet 刀0）。
    # **source=SOURCE_CHINESE_KB provenance 准**（异 ConceptNet is_a_facts 的 SOURCE_CONCEPTNET·避 build_isa_ancestor_map_external 刀C 验证污染·MED-1 式防御）。
    # 按 corpus 唯一 lang 集循环 resolve_abstract_facts（env ZERO_AI_LOCAL_DIR/abstract_facts_{lang}.txt）。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 零副作用返 0 → bit-identical 守回归**
    # （bootstrap_is_a_edges 空 pairs 首行短路·绝不调 ensure/query_from/build）。resume 走 query_from 幂等 skip（同源同三元组）。
    # **IS_A 不接 reward 反传**（effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·IS_A 不内·静态 strength·IS_A_STRENGTH_EMPIRICAL）·
    # abstract IS_A 边 enrich ancestor_map/LCA 聚类（abstraction.py）→ 影响 selection_pref 生成（预期非 bit-identical·gate ON 生产）。
    for _lang in {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}:
        _abs_pairs = filter_pairs_to_vocab(resolve_abstract_facts(_lang), _kb_vocab) if _rel_on("abstract") else ()
        if _abs_pairs:
            # #1133：捕获 boot 种边数 → result.abstract_is_a_edges_seeded（异 ConceptNet is_a·source=CHINESE_KB·对称 similar/antonym·W8 observability）。
            # 空 pairs（CI 无文件）短路不进此分支 → abstract_is_a_edges_seeded 落默认 0 → bit-identical。
            result.abstract_is_a_edges_seeded += bootstrap_is_a_edges(
                ctx.concept_index, ctx.edge_store, _abs_pairs,
                space_id=ctx.space_id, source=SOURCE_CHINESE_KB)
    # 刀6 件7 sense 多义候选 boot 种（boot 时种 sense_candidates base_count·解锁理解侧 recognize clone 选 sense·
    # doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀6）。sense_facts 本地文件（来源① 结构化·E10·EPI_STRUCTURED·
    # 外部多义词典非 core enum·守「不写死」·类比刀0 IS_A loader）。按 corpus 唯一 lang 集循环 resolve_sense_facts
    # （env ZERO_AI_LOCAL_DIR/sense_facts_{lang}.txt）→ bootstrap_sense_candidates 种 (word→[senses]) base_count=1。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 空 short-circuit 返 0 →
    # sense_candidates 表空 → bit-identical 守回归**（bootstrap_sense_candidates 空 pairs 首行短路·绝不调
    # ensure/select/insert·退化链 5 步·plan 决断 5）。resume：load_run 已还原 → boot 再种 first-write-wins 幂等。
    # **gate SENSE_LOOKUP_MODE 守**：boot 种 base_count 无条件（同 IS_A·有 lang corpus 才种·sense_facts 文件
    # 存在才非空）·消费侧（observe 写 sc_tn / recognize clone）gate OFF 时退化零行为变。
    for _lang in {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}:
        _sense_pairs = resolve_sense_facts(_lang) if _rel_on("sense") else ()
        if _sense_pairs:
            bootstrap_sense_candidates(ctx.backend, ctx.concept_index, _sense_pairs,
                                       space_id=ctx.space_id)
    # 入手④ 接 CAUSES 源（boot 时种 EDGE_CAUSES·给 CAUSES 外部 R6 独立源·总收口 §三簇1入手④·镜像刀0 IS_A boot）：
    # ConceptNet Causes 有向三元组（cause Causes effect·照搬不反转·M1·§8.1c-bis 来源① EPI_STRUCTURED·外部数据非
    # core enum·守「不写死」）。按 corpus 唯一 lang 集循环 resolve_causes_facts（env ZERO_AI_LOCAL_DIR/causes_facts_{lang}.txt）。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 零副作用返 0 → bit-identical 守回归**
    # （bootstrap_causes_edges 空 pairs 首行短路·绝不调 ensure/query_from/build·镜像 bootstrap_is_a_edges:119-120）。
    # resume 路径：load_run 已还原 base run 边 → boot 再种走 query_from 幂等 skip（同源同三元组）·不 corrupt。
    # **CAUSES 接 reward 反传**（异 IS_A 不接）·故 active causes_facts 文件改变训练 reward（预期·R6 外部因果信号）。
    for _lang in {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}:
        _cause_pairs = resolve_causes_facts(_lang) if _rel_on("causes") else ()
        if _cause_pairs:
            bootstrap_causes_edges(ctx.concept_index, ctx.edge_store, _cause_pairs,
                                   space_id=ctx.space_id)
    # T-L1d mereology 部分-整体 boot 种（boot 时种 EDGE_MEREOLOGY 边·解 cue_words REL_MEREOLOGY 误路由入 IS_A_CUE·
    # 客观序 gap 补·doc/重来_语言域断奶客观序_2026-07-15 §三 T-L1d·镜像刀0 IS_A / 入手④ CAUSES boot）：
    # ConceptNet PartOf / WordNet meronym 有向三元组（part part-of whole·照搬不反转·M1·§8.1c-bis 来源① EPI_STRUCTURED·
    # 外部数据非 core enum·守「不写死」·**MEREOLOGY≠IS_A** 部分-整体≠子集·语义正交）。按 corpus 唯一 lang 集循环
    # resolve_mereology_facts（env ZERO_AI_LOCAL_DIR/mereology_facts_{lang}.txt）。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 零副作用返 0 → bit-identical 守回归**
    # （bootstrap_mereology_edges 空 pairs 首行短路·绝不调 ensure/query_from/build·镜像 bootstrap_is_a_edges:119-120）。
    # resume 路径：load_run 已还原 base run 边 → boot 再种走 query_from 幂等 skip（同源同三元组）·不 corrupt。
    # **MEREOLOGY 不接 reward 反传**（异 CAUSES 接 / 同 IS_A 不接）·effective_weight:82 assert 只认
    # {PRECEDES,CAUSES,REFERS_TO}·MEREOLOGY 不内·静态 base_strength·故 mereology_facts 文件不改变训练 reward（仅扩图）。
    for _lang in {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}:
        _mere_pairs = filter_pairs_to_vocab(resolve_mereology_facts(_lang), _kb_vocab) if _rel_on("mereology") else ()
        if _mere_pairs:
            # #1119：捕获 boot 种边数 → result.mereology_edges_seeded（对称 alias #2220 / number #2236·W8 observability 信号·消费者待接 #941）。
            # 空 pairs（CI 无文件）短路不进此分支 → mereology_edges_seeded 落默认 0 → bit-identical。
            result.mereology_edges_seeded += bootstrap_mereology_edges(
                ctx.concept_index, ctx.edge_store, _mere_pairs, space_id=ctx.space_id)
    # T-L1e antonym 反义对称 boot 种（boot 时种 EDGE_ANTONYM 边·客观序 gap 补·
    # doc/重来_语言域断奶客观序_2026-07-15 §三 T-L1e·镜像 T-L1d mereology / 刀0 IS_A boot）：
    # ConceptNet /r/Antonym / WordNet antonym 对（无序对称·照搬·§8.1c-bis 来源① EPI_STRUCTURED·外部数据非 core enum·守「不写死」·
    # **语言反义 concept↔concept 1 阶·非 verify_inverse**（代数逆 transform↔transform T-L4·verify_inverse 只验数学·#479 外部 seed 非 verify））。
    # 按 corpus 唯一 lang 集循环 resolve_antonym_facts（env ZERO_AI_LOCAL_DIR/antonym_facts_{lang}.txt）。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 零副作用返 0 → bit-identical 守回归**
    # （bootstrap_antonym_edges 空 pairs 首行短路·绝不调 ensure/query_from/build·镜像 bootstrap_mereology_edges:97-98）。
    # resume 路径：load_run 已还原 → boot 再种走 query_from 幂等 skip（同源同三元组）·不 corrupt。
    # **ANTONYM 不接 reward 反传**（同 SIMILAR/IS_A/MEREOLOGY·effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·ANTONYM 不内·
    # 静态 strength 恒=1·故 antonym_facts 文件不改变训练 reward·仅扩图·对称单边 a→b·reader 双向查镜像 similar_candidates）。
    for _lang in {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}:
        _ant_pairs = filter_pairs_to_vocab(resolve_antonym_facts(_lang), _kb_vocab) if _rel_on("antonym") else ()
        if _ant_pairs:
            # #1119 MED-1 fix：antonym 按 lang 路由 source provenance——zh=ChineseSemanticKB（反义关系库·SOURCE_CHINESE_KB）·
            # en=ConceptNet（/r/Antonym·SOURCE_CONCEPTNET）。避免 zh 反义（ChineseSemanticKB 源）误盖 SOURCE_CONCEPTNET
            # provenance（boot 按源细化幂等 dedup 须准·#1119 审2 MED-1）。CI 无文件→resolve []→不进此分支→bit-identical。
            _ant_src = SOURCE_CHINESE_KB if _lang == LANG_ZH else SOURCE_CONCEPTNET
            # #1119：捕获 boot 种边数 → result.antonym_edges_seeded（对称 mereology/alias/number·reader 待接线·W8 observability 信号）。
            # 空 pairs（CI 无文件）短路不进此分支 → antonym_edges_seeded 落默认 0 → bit-identical。
            result.antonym_edges_seeded += bootstrap_antonym_edges(
                ctx.concept_index, ctx.edge_store, _ant_pairs, space_id=ctx.space_id, source=_ant_src)
    # T-L1c similar 近义对称 boot 种（boot 时种 EDGE_SIMILAR 边·客观序 gap 补·镜像 T-L1e antonym·doc/重来_语言域断奶客观序 §三）：
    # ChineseSemanticKB 同义关系库 / ConceptNet /r/Synonym（无序对称·照搬·§8.1c-bis 来源① EPI_STRUCTURED·外部数据非 core enum·
    # 守「不写死」·**语言近义 concept↔concept 1 阶·非 verify**·EDGE_SIMILAR 机制全在 #898·boot-side loader 镜像 antonym·#1132）。
    # 按 corpus 唯一 lang 集循环 resolve_similar_facts（env ZERO_AI_LOCAL_DIR/similar_facts_{lang}.txt）。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 零副作用返 0 → bit-identical 守回归**
    # （bootstrap_similar_edges 空 pairs 首行短路·绝不调 ensure/query_from/build·镜像 bootstrap_antonym_edges:99-100）。
    # resume 路径：load_run 已还原 → boot 再种走 query_from 幂等 skip（同源同三元组）·不 corrupt。
    # **SIMILAR 不接 reward 反传**（同 ANTONYM/IS_A/MEREOLOGY·effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·
    # SIMILAR 不内·静态 strength 恒=1·故 similar_facts 文件不改变训练 reward·仅扩图·对称单边 a→b·reader 双向查）。
    for _lang in {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}:
        _sim_pairs = filter_pairs_to_vocab(resolve_similar_facts(_lang), _kb_vocab) if _rel_on("similar") else ()
        if _sim_pairs:
            # similar 按 lang 路由 source provenance（镜像 antonym #1119 MED-1）：zh=ChineseSemanticKB（同义关系库·SOURCE_CHINESE_KB）·
            # en=ConceptNet（/r/Synonym·SOURCE_CONCEPTNET）。CI 无文件→resolve []→不进此分支→bit-identical。
            _sim_src = SOURCE_CHINESE_KB if _lang == LANG_ZH else SOURCE_CONCEPTNET
            # #1132：捕获 boot 种边数 → result.similar_edges_seeded（对称 antonym/mereology/alias/number·机制全在 #898·**reader dispatch_slot 在**（slot_dispatch SIMILAR_SLOT_MODE 双向查·#898·异 antonym/mereology reader 待接线）·W8 observability 信号·唯一有 live 消费者的关系 boot 边）。
            # 空 pairs（CI 无文件）短路不进此分支 → similar_edges_seeded 落默认 0 → bit-identical。
            result.similar_edges_seeded += bootstrap_similar_edges(
                ctx.concept_index, ctx.edge_store, _sim_pairs, space_id=ctx.space_id, source=_sim_src)
    # P0b 跨语言/同义 PURE_ALIAS 桥 boot 种（boot 时种双向 REFERS_TO PURE_ALIAS 边·解 C 偏离跨语言汇聚·
    # §3.1/§7.4 "苹果/apple 同节点"=REFERS_TO 等价类非同 local_id·铁律"永不合并节点"·身份 Model A 边连接）。
    # alias_facts 本地文件（来源① 结构化·Wikidata QID 翻译等价·E10·EPI_STRUCTURED·外部跨语言词典非 core enum·
    # 守「不写死」·**异 IS_A/sense/causes 单 lang 文件**：aliases 横跨语言·单一 alias_facts.txt 无 lang 后缀·
    # 镜像 struct_bind_pairs 范式·caller 不带 lang）。resolve_alias_facts（env ZERO_AI_LOCAL_DIR/alias_facts.txt）
    # → bootstrap_alias_edges 种 (surface_a,lang_a,surface_b,lang_b) 双向 PURE_ALIAS + 各自 MARK_LANG + NODE_WORD。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 空 short-circuit 返 0 →
    # 核心空间零 PURE_ALIAS 边 → activate_candidates PURE_ALIAS-gated 自包含退化现状 → bit-identical 守回归**
    # （bootstrap_alias_edges 空 pairs 首行短路·绝不调 ensure/set_mark/build·退化链·镜像 bootstrap_is_a_edges:119-120）。
    # resume：load_run 已还原 PURE_ALIAS 边 → boot 再种走 query_from 幂等 skip（_pure_alias_exists·同源同三元组）·不 corrupt。
    # **PURE_ALIAS 不动拓扑**：effective_weight return 0（不进 PR）·dag_path 不遍历 REFERS_TO·observe/reward/judge 不读
    # PURE_ALIAS 边·故 gate ON 只让 generate 输出按 target_lang 选对词形（预期非 bit-identical·审计 L278"仅桥活时触发"）。
    _alias_pairs = filter_pairs_to_vocab(resolve_alias_facts(), _kb_vocab, 0, 2) if _rel_on("alias") else ()
    if _alias_pairs:
        # #1041 构造④：捕获 boot 种边数 → result.alias_edges_seeded（capability_exam 判据⑤跨语言汇聚 observability 消费者）。
        # 空 pairs（CI 无文件）短路不进此分支 → alias_edges_seeded 落默认 0 → bit-identical。
        result.alias_edges_seeded = bootstrap_alias_edges(
            ctx.concept_index, ctx.edge_store, ctx.backend,
            _alias_pairs, space_id=ctx.space_id)
    # language-grounding piece 1：数字词接地 boot（language→arith 桥·语言域主攻首刀·doc/重来_语言通用接地 §七）。
    # number_facts 本地文件（ZERO_AI_LOCAL_DIR/number_facts.txt·来源① 结构化·EPI_STRUCTURED·**守「不写死」**：
    # 三↔3 来自外部数据非代码·**关联在图中**：bootstrap 种 整数概念 __int_{value} + CORR_NUMERIC 值 + 词 PURE_ALIAS 图边）。
    # **无文件（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → bootstrap 空 short-circuit →
    # 核心空间零整数概念/PURE_ALIAS 数字边 → bit-identical 守回归**（空 facts 首行短路·绝不调 ensure/record/set_mark/build·
    # 退化链·镜像 bootstrap_alias_edges）。resume：PURE_ALIAS 数字边已还原 → boot 再种 query_from 幂等 skip·不 corrupt。
    # PURE_ALIAS 数字边不动拓扑（effective_weight return 0·dag_path 不遍历 REFERS_TO）·同 alias·gate ON 只让 generate 产词。
    _number_facts = resolve_number_facts() if _rel_on("number") else ()
    if _number_facts:
        # 捕获 boot 种边数 → result.number_edges_seeded（对称 alias_edges_seeded #2219·W8 语言域断奶 observability 消费者待接 #941·2 对抗审 LOW 修）。
        # 空 facts（CI 无文件）短路不进此分支 → number_edges_seeded 落默认 0 → bit-identical。
        result.number_edges_seeded = bootstrap_number_grounding(
            ctx.concept_index, ctx.edge_store, ctx.backend,
            _number_facts, space_id=ctx.space_id)
    # 刀3 件1 种概念（boot 时种 REL_* first-class NODE_CONCEPT + D:11 EDGE_RELATION_SIGNAL 词→关系概念边·
    # 解锁件2/5/8 晋升目标框架·doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀3）：
    # 元定义层 frozenset 种子（word_concept_signal._REL_LEXICAL_CUE·同 cue_words 范式·非外部文件·
    # Plan agent 路线决断 ii·doc §5 刀3"元定义层"忠实实现）。D:11 已注册零产消者（edge_types.py:74）·
    # 本块激活产消者。D:11 不接 reward（effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·
    # D:11 不内）·不进 PR/closure（PR 邻接只 {PRECEDES,CAUSES,REFERS_TO}·D:11 不内）·§8.1c-bis 合规。
    # **bit-identical**：frozenset 内置·无条件种（有语言 corpus 才种）·CI===生产·870 测零翻
    # （既有断言全 type-filtered 正交 D:11/REL_* 节点·grep 全 tests/ CONFIRMED 无 formal_train 总数断言）。
    _langs_rel = {it.lang for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens}
    if _langs_rel:   # 有语言 corpus 才种（arith-only run 不种·守最小副作用）
        bootstrap_word_concept_signals(ctx.concept_index, ctx.edge_store, ctx.backend,
                                       space_id=ctx.space_id, langs=_langs_rel)
        # STEP5 PR2：operator-level D:11 种子（加/减/乘/大于/小于 closed-class 核心→OP_* concept D:11 边）。
        # 镜像 bootstrap_word_concept_signals·D:11 共享边类型·OP_* target 挂 ATTR_OPERATOR_PRIMITIVE 与 REL_* 隔离。
        # 加二源非替换（D6）：frozenset _ARITH_OP_WORDS/_COMPARISON_OP_WORDS 检测第一源保留不变·D:11 为 gate-ON 二源。
        bootstrap_operator_signals(ctx.concept_index, ctx.edge_store, ctx.backend,
                                   space_id=ctx.space_id, langs=_langs_rel)
        # 审计根治 [严重-1]：modal-level D:11 种子（必然/可能/必须/应该/可以 closed-class 核心→MODAL_KIND concept D:11 边）。
        # 镜像 bootstrap_operator_signals·D:11 共享边类型·MODAL_KIND target 挂 ATTR_MODAL_KIND=22 与 REL_*/OP_* 隔离。
        # 加二源非替换（D6）：frozenset _MODAL_CUES 检测第一源保留不变·D:11 为 gate-ON 二源·解 _MODAL_CUES 换名字写死。
        bootstrap_modal_signals(ctx.concept_index, ctx.edge_store, ctx.backend,
                                space_id=ctx.space_id, langs=_langs_rel)
        # #940 否定词 D:11 种子（不/没/非/无 closed-class 核心→TYPE_NEGATION concept D:11 边）。
        # 镜像 bootstrap_modal_signals·D:11 共享边类型·TYPE_NEGATION target 挂 ATTR_SYMBOL_TYPE=17 与 REL_*/OP_*/MODAL 隔离。
        # 加二源非替换（D6）：frozenset _NEGATION_CUES 检测第一源保留不变·D:11 为 gate-ON 二源·解否定词穷举不尽。
        # 否定=符号域先天（同 operator·异 modal）·激活 ensure_symbol_types（shadow→活）。
        bootstrap_negation_signals(ctx.concept_index, ctx.edge_store, ctx.backend,
                                   space_id=ctx.space_id, langs=_langs_rel)
        # #1134 程度→属性器 intensity：degree 副词→Rational intensity（很/非常=2/1·较=3/2·稍=2/5·file-driven·非 §九 frozenset）。
        # resolve_degree_facts 读 ZERO_AI_LOCAL_DIR/degree_cues_{lang}.txt → populate_degree_cues 喂 cue_words module cache
        # （degree_intensity_of/is_degree_cue 读·gate DEGREE_MODE 守）。CI/生产 default 无文件 → {} → 不污染 cache
        # → is_degree_cue 恒 False → intensity 恒 1/1 → bit-identical（空 mapping no-op·populate_degree_cues 早返）。
        for _lang in _langs_rel:
            populate_degree_cues(_lang, resolve_degree_facts(_lang))
        # B-PR1（doc §16）：action D:11 种子（帮我/请/生成/计算 closed-class 核心→INTENT_COMMAND_MOOD + ACTION_* concept D:11 边）。
        # 镜像 bootstrap_negation_signals·D:11 共享边类型·动作意图 target 挂 ATTR_OPERATION_INTENT=23 与 REL_*/OP_*/MODAL/NEGATION 隔离。
        # 加二源非替换（D6）：frozenset 命令词/动作词检测第一源（action_primitives._ACTION_LEXICAL_CUE + cue_words.is_action_intent_cue）保留·
        # D:11 为 gate-ON 二源·解命令词/动作词穷举不尽。动作意图=符号域先天（镜像 operator·异 modal）·
        # 命令词+动作词同基建（W7 命令判定 = 命令词 OR 动作词命中任一·doc §16.4）。
        bootstrap_action_signals(ctx.concept_index, ctx.edge_store, ctx.backend,
                                 space_id=ctx.space_id, langs=_langs_rel)
    # 序列6-min 生产触发（de-theater 序列1·§八.6）+ 序列3-min 生产 READ 识别（§八.3）+ 验证半闭环（§8.7）+
    # 序列7 跨 run READ（resume load_run 后识别**全新** held-out 命中载入算子·§八.7）：
    # 算术语料 → per-shape 留 held-out → auto_discover_operators（discover_skeleton 真生产 caller·WRITE）+
    # recognize_operators（held-out 新输入命中已学骨架·READ 消费·非循环 theater）+ _verify_generalization
    # （vm_proof 独立验骨架绑参复现 held-out 新输入值·识别产物 recognitions 真消费·解 terminal 边界·反 theater）。
    # resume load_run 后调（跨 run 幂等·lookup 门 + 内容哈希根 + 纯读识别）。非 reward 驱动（observe/抽象层·stage loop 前）。
    # 序列7：existing_operators = 已载发现算子（fresh run 空·resume 后 load_run 已还原 composes_attr → 取到 run N 算子）。
    _existing_ops = load_discovered_operators(ctx.backend, space_id=ctx.space_id)
    result.discovered_operators, result.recognitions, result.generalization = \
        _discover_and_recognize_arith_operators(ctx, corpus, existing_operators=_existing_ops)
    # 刀6 件7：SENSE_LOOKUP_MODE 须在 _discover_and_recognize_lang_structures（clone 选 sense）之前翻——
    # clone 段（caller 建 COMPOSES 首 sense + recognize_roots clone aligning_root）读 gate·若在 stage loop 前
    # 才翻（旧位 :1080）则 clone 段读到 OFF → 生产路径 clone 永不触发（反 theater 牙失效·纸面闭合·对抗审 P0-1）。
    # stage loop observe（sense_lookup hook → MultiRef → record sc_tn）也用此 gate·同一翻覆盖两处。
    # finally（:1203）复位 saved_sense_lookup。
    saved_sense_lookup = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = True
    # 钥匙①语言结构发现（S3 第二片·件1 落点修正·caller 建独立根 __disc_lang_·反 theater·observe 不建 A6 不冲突）：
    # 语言语料 → 内容哈希独立根 + 建语言 COMPOSES 序（NOP+token 叶·caller 建）→ auto_discover + recognize。
    # vm_proof 跳过（语言骨架不可 VM·钥匙③墙·verified=0 诚实）·concept_binding（件5）= 语言识别产物。
    # 刀6 片4：caller 建 COMPOSES 首 sense ref（gate ON）+ recognize_roots clone aligning_root 逐 sense 试 +
    # distinct origin（反 theater 牙：动物类骨架选动物老鼠不选鼠标·IS_A 共祖结构选优·非语义消歧·#479 墙）。
    # ★ Bug B 修（doc/重来_语言域建模推进设计_2026-07-18 §2.1）：_discover_and_recognize_lang_structures 读
    # COMPOSES（scope B 切句 + dim 桥 map）/REALIZES（label skeleton->R）/CUE_CLUSTER（cue 拆 ATTR_CUE_SIG）
    # /ORACLE_PROMOTE（tally_cue_slot_matches 建 D:11 SHADOW·measured 路径必需）·须在调用前翻 ON。旧位 7-gate flip
    # 在 discovery 后->gate OFF 跑->map 空/无 REALIZES/无 cue 拆/无 tally SHADOW->零 INSTANTIATES 边->measured=False。
    # 镜像 sense_lookup 先翻范式（saved 早存·finally 复位）。ORACLE_PROMOTE 是 ground-truth discovery_only_smoke
    # 漏翻的第 4 gate（smoke 只翻 3 证 cue 涌现·measured 还需 tally SHADOW·故 4 gate 全翻）。pre-flip finally 立即复位·
    # 生产 7-gate flip（:2825）接管 stage loop/generate（含 CORR_SLOT/CUE_SLOT_FILL/SLOT_LCA 生成侧 3 gate）。
    # saved_* 用 _disc_ 前缀避撞生产 saved_*（:2790）。4 gate = discovery 实读集合（生成侧 3 gate discovery 不读·留给生产 flip）。
    _disc_saved_compose = gates.COMPOSES_COMBINE_MODE
    _disc_saved_realizes = gates.REALIZES_MODE
    _disc_saved_cue_cluster = gates.CUE_CLUSTER_MODE
    _disc_saved_oracle_promote = gates.ORACLE_PROMOTE_MODE
    gates.COMPOSES_COMBINE_MODE = True
    gates.REALIZES_MODE = True
    gates.CUE_CLUSTER_MODE = True
    gates.ORACLE_PROMOTE_MODE = True
    try:
        _lang_disc, _lang_rec, result.lang_generalization = _discover_and_recognize_lang_structures(
            ctx, corpus, existing_operators=_existing_ops)
    finally:
        gates.COMPOSES_COMBINE_MODE = _disc_saved_compose
        gates.REALIZES_MODE = _disc_saved_realizes
        gates.CUE_CLUSTER_MODE = _disc_saved_cue_cluster
        gates.ORACLE_PROMOTE_MODE = _disc_saved_oracle_promote
    result.discovered_operators.extend(_lang_disc)
    result.recognitions.extend(_lang_rec)
    # #478 STRUCT_BIND 跨模态槽位级绑定 boot 种边（lang discover 后·两模态 skeleton ref 就位·参照刀0 IS_A 范式·
    # doc/重来_任务0478_STRUCT_BIND_设计.md 决断 4）：
    # gate STRUCT_BIND_BOOT_MODE default OFF（boot-side gate-controlled deferred activation·首例 boot-side gate·
    # IS_A/sense/word_concept boot 无 gate 因 lang-keyed + 空 pairs 短路已足·STRUCT_BIND 跨模态非 lang-keyed 单文件·
    # gate 为显式 bit-identical 开关·决断 8）。**不在生产入口 try/finally 翻 ON**（无真实教师 corpus·#731 defer）。
    # OFF → 完全跳过（不调 resolve/bootstrap·CI===生产 default bit-identical）。
    # ON + 无文件 → resolve_struct_bind_pairs 返 [] → bootstrap 空 pairs 短路（双层守·gate + 空 pairs）。
    # ON + 有文件 → resolve_struct_bind_pairs 读 ZERO_AI_LOCAL_DIR/struct_bind_pairs.txt（来源 a 教师标注·
    #   非 core 数据·守「不写死」）→ discovered_operators 索引 name→skeleton_ref（决断 4"name 映射机制"·cross-run
    #   _existing_ops + 本 run result.discovered_operators）+ collect_skeleton_slot_refs 解析 slot ref →
    #   bootstrap_struct_bind_edges 建 EDGE_STRUCT_BIND 槽位级边（from=A 槽 ref·to=B 槽 ref·order_index=槽序）。
    # name 未命中（跨 run mismatch / 教师 typo）→ 该对 skip（E5 graceful·不抛崩）。slot idx 越界 → 该 slot 对 skip。
    # **消费侧 reader 落 #730**（generate 读 STRUCT_BIND 填语言槽·反 theater 锚闭环）·本 boot 只建边（form 2 风险·
    # #730 跟进后合法·反 theater 锚·决断 7）。STRUCT_BIND 边入主 edge 宽表·dump_tables（既有）已含·无须新表。
    if gates.STRUCT_BIND_BOOT_MODE:
        _bind_name_pairs = resolve_struct_bind_pairs()
        if _bind_name_pairs:
            # name→skeleton_ref 索引·dict 字面量 latter-wins（本 run 新发现覆盖历史 _existing_ops）·
            # 决断 4 优先序：教师 file 同名时绑本 run 新 ref（合理默认·本 run discover 产物为 caller 当前可见 ref）·
            # 跨 run 同 (sig,arity,abstract_sig) 同名算子经 _shape_name hash 确定性·resume _existing_ops 跨 run 同名同 ref。
            _ops_by_name = {op.name: op.skeleton_ref
                            for op in list(_existing_ops) + list(result.discovered_operators)}
            _bind_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for _na, _nb, _slot_map in _bind_name_pairs:
                _sa = _ops_by_name.get(_na)
                _sb = _ops_by_name.get(_nb)
                if _sa is None or _sb is None:
                    continue   # name 未命中（跨 run mismatch / 教师 typo）skip（E5 graceful·不抛崩）
                _slots_a = collect_skeleton_slot_refs(ctx.backend, ctx.concept_graph, _sa)
                _slots_b = collect_skeleton_slot_refs(ctx.backend, ctx.concept_graph, _sb)
                for _ai, _bi in _slot_map:
                    if _ai < len(_slots_a) and _bi < len(_slots_b):
                        _bind_edges.append((_slots_a[_ai], _slots_b[_bi]))
                    # slot idx 越界（slot_map 标 idx > skeleton 实际 arity）→ 该 slot 对 skip（容错·不抛崩）
            bootstrap_struct_bind_edges(ctx.edge_store, _bind_edges, space_id=ctx.space_id)
    # §8.7-全 生成侧全环·task-driven L8 episode（外真半·补半环缺的"外真半"·墙内现可达）：
    # 任务(arith item 的 arith_specs·(input_args,expected)) → 选算子(arity+置信度择优) → 执行骨架
    # → 外真验 vs expected(R6 独立源·非传递) → 写 op_confidence → 打包 OutputResult → metrics generate_verified。
    # 不碰 generate.py L6·不依赖 STRUCT_BIND·守单向 L8→L7 execute/L0 op_confidence 向下。
    # all_ops = 载入(跨 run)+本 run 新发现（同 recognize/verify 的 all_ops·task-driven 消费全部已学算子）。
    _all_ops = list(_existing_ops) + list(result.discovered_operators)
    # task-driven L8 episode 移 stage loop 后调（#730：code COMPOSES 树在 stage loop observe 建·task-driven 须
    # 后置才能读 item.code_struct_ref·_gen_eps/result.generate 在 :1423+ 消费）。_all_ops 此处算（discover 产物·
    # stage loop 不改 discovered_operators·arith op_confidence stage loop _run_verify_round 不写·值稳定）。
    round_id = 0
    # #723 G 归因：跨 stage + Mode A task-driven 全 episode 累加器（collect_episodes=True 时挂 result.episodes）。
    # 三路 episode：language judge（G4/G2p/G3a/G3b/G5·pr_vector 非空）/ Mode B verify（G5 only·pr_vector 空）/
    # Mode A task-driven（_gen_eps·G5=False 硬编码·input=None）。harness 投影 G 归因交叉表读此。
    all_eps: list[Episode] = []
    # 防塌三柱验收累加器（致命5 生产 caller·主循环每 stage 汇总·终写 result.collapse_summary）
    collapse_acc: dict[str, Any] = {"verified": 0, "total": 0, "pillar1_ok": 0,
                                    "pillar2_ok": 0, "pillar3_ok": 0, "low_variance": 0}
    # A1（CUE_EXTRACTOR_MODE 接线·致命3 残留·断奶后语言域 CAUSES 自产源）：
    # 生产入口 observe 须自产 CAUSES（cue_extractor 纯元定义·cue_words 中英 lang 出厂硬件·非接地墙·
    # 断奶后教师退场无手注→必须自产否则 J3 veto reward 锁死）。默认 OFF 守单测 bit-identical·
    # formal_train 生产入口 ON·try/finally 守回归。**翻在此（生产入口）非 run_round_full**：
    # cue 在 _split_item_to_segments（line 452 填 Segment）+ observe（建边）被读·二者在 run_round_full 前段·
    # reward 阶段 ATTRACTOR 翻（line 307）在 observe 之后→翻 reward 阶段太晚；且 run_round_full 须保
    # gate-respecting 可测单元（test_cue_extractor_off_..._e2e 直调 run_round_full+gate OFF 验回归）。
    # 详见 doc/重来_清查后待做整合与执行序.md A1。
    saved_cue = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    # P0a·ordinal 码点 surface resolver（生产入口翻 ON·镜像 CUE_EXTRACTOR 范式·默认 OFF 守回归·try/finally 守）。
    # surface_of live-read 此 gate -> ON 时 generate 读 concept_correspondence 码点产真实文本（解 A 偏离 #1:42 占位）·
    # OFF 退 None -> 占位（CI bit-identical）。reward 阶段（run_round_full ATTRACTOR 内层）继承此 flip。
    saved_ordinal_surface = gates.ORDINAL_SURFACE_MODE
    gates.ORDINAL_SURFACE_MODE = True
    # P0 #1040：generate-dispatch 主缺口修复——slot.ref 派发 token concept（graph.read_token_seq·def_array 存储·
    # repeat-safe）+ ctx_refs token 级（produced_refs/prior_topic_refs）。与 ORDINAL_SURFACE_MODE 同翻（外层 try/finally·
    # observe 在内层前跑·两 gate 同翻：dispatch 出 token concept·surface_of 出真字·系统产真语言非 __seg_* label）。
    # Path C 存储（非 PRECEDES walk）解 reward=0（walk node_type filter 误滤 NODE_CONCEPT token 致空产 → reached_sink
    # False → G2p veto·Path C 直读 def_array → 非空 → reward>0）。CI default OFF bit-identical。
    saved_dispatch_tokens = gates.DISPATCH_TOKEN_CHAIN_MODE
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    # P0 #1041：reward 信号 truthiness 校准——judge J4word 项（产出真词覆盖率·读 token_refs）。
    # 三 gate 同翻：dispatch 出 token concept + surface_of 出真字 + reward 反映真词质量（判据②③信号质量·
    # 解 review-2 钉死：旧 reward 对真词/__seg_* 同分）。CI default OFF→J4word=0→reward 逐字现状 bit-identical。
    saved_output_word_reward = gates.OUTPUT_WORD_REWARD_MODE
    gates.OUTPUT_WORD_REWARD_MODE = True
    # G5-C memory consolidate（生产入口翻 ON·#732 落地 dormant→P1 激活·审计 §6 P1）。
    # STAGE4_PROMOTE_WEAN 末 _promote_eligible 后扫 memory_item by info_ref·G5-C 闸判达 → consolidate flip。
    # ctx.memory_read is not None 时 fire（生产训练期实例化）。判据④记忆层晋升轴。
    saved_g5c = gates.G5_C_CONSOLIDATE_MODE
    gates.G5_C_CONSOLIDATE_MODE = True
    # 刀4 涌现关系学习（生产入口翻 gate·镜像 CUE_EXTRACTOR 范式·默认 OFF 守回归·try/finally 守）：
    # HYPOTHESIS_MODE = 涌现假设生成 + D:11 SHADOW 落边（reward 阶段 observe 后 episode 前·_run_emergence_hook）。
    # FEED_MODE = reward_propagate concept_targets 扩展（D:11 SHADOW 候选进 experience_count feed·子环3 鸡生蛋破解）。
    # 刀5 件8：CUE_READBACK_MODE 在此翻（兑现刀4 defer 注释·cue_extractor 生产透传已落 :288/:496）。
    # readback = cue_type_of 第二源读 D:11 PRIMARY 边（"引发"类涌现词经 promote 后第二轮返非 None·反 theater）。
    saved_emerg_hyp = gates.EMERGENT_RELATION_HYPOTHESIS_MODE
    saved_emerg_feed = gates.EMERGENT_RELATION_FEED_MODE
    saved_emerg_readback = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_HYPOTHESIS_MODE = True
    gates.EMERGENT_RELATION_FEED_MODE = True
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    # 对应泛化 v2（生产激活·三 gate 共翻·结构反推机制 live·doc/重来_对应泛化_结构反推_学全 §六 片2）：
    # REALIZES 标 skeleton→R（oracle grounded·内容对命中 ConceptNet）+ CUE_CLUSTER 拆 是/使 异名骨架（ATTR_CUE_SIG 落盘·
    # cue slot 可位）+ ORACLE_PROMOTE tally→promote 结构匹配轨（D:11 删∨·generator 关·tally 建 SHADOW）。
    # 三者共构结构反推·缺一则机制断（无 REALIZES 无 exemplar / 无 CUE_CLUSTER 无 cue slot / 无 ORACLE_PROMOTE 无 promote）。
    # CI default OFF→零 tally→零 D:11 翻→bit-identical（生产 try/finally 翻 ON = 机制 live·非 CI 行为）。
    saved_realizes = gates.REALIZES_MODE
    saved_cue_cluster = gates.CUE_CLUSTER_MODE
    saved_oracle_promote = gates.ORACLE_PROMOTE_MODE
    # 对应桥第 4 gate（readback→generation·学到的 cue 词流入生成·doc/重来_对应泛化_readback_generation_桥 §2.4）：
    # 消费 v2 三 gate 产物（REALIZES exemplar + CUE_CLUSTER cue slot + ORACLE_PROMOTE D:11）→ dispatch_slot 第 8 路 correspondence bonus。
    # 缺此 gate → 桥机制就位但生产 generate 不读 → 学到的对应只识别不产出=白学（设计 §1 命门·post-impl 审严重-1）。
    saved_corr_slot = gates.CORRESPONDENCE_SLOT_MODE
    # 对应桥写侧第 5 gate（COMPOSES_COMBINE_MODE·observe 建 EDGE_INSTANTIATES 真边 on __seg_ struct_ref→skeleton_ref·
    # doc/重来_对应机制生产激活_2026-07-17）：桥读侧（CORRESPONDENCE_SLOT_MODE·上）已生产 flip 但写侧 dormant→读侧恒走空分支
    # （无 INSTANTIATES 边）→ dispatch 第 8 路从不 fire。翻写侧 ON = 完成桥（让造句真用学到的 D:11·解"白学"）= 当初 Phase A.3
    # consumer 激活（翻 REJECT·learned 路径已就位：REALIZES 外源 + tally→promote D:11 + CUE_CLUSTER cue·2 对抗审
    # APPROVE-WITH-CONDITIONS 确认非 theater·§4.0 用户原则：底子合法+学习真+泛化）。gate OFF→无 INSTANTIATES 边→
    # generate.py:154 走空分支→bit-identical（FC12 直守·monkeypatch OFF+≥K lang corpus→零 INSTANTIATES）。
    saved_compose = gates.COMPOSES_COMBINE_MODE
    # 命门③ 候选 B 第 6 gate（CUE_SLOT_FILL_MODE·doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）：
    # dispatch_slot cue 位早 return 直出 cue token 功能词·绕 collide 全下游。消费前 5 gate 产物（REALIZES exemplar
    # + CUE_CLUSTER cue_sig + ORACLE_PROMOTE D:11 + CORRESPONDENCE_SLOT stash current_cue_sig/slot_idx + COMPOSES_COMBINE
    # INSTANTIATES 边 + ORDINAL_SURFACE surface_of）-> cue 位 cue token 直出。缺此 gate -> 6 gate 链断在末端·
    # cue slot 走 collide 选内容词而非直出功能词 -> 结构活化不活 = theater（post-impl 审 §8 真 bug·同范式漏翻 CORRESPONDENCE_SLOT_MODE）。
    # gate OFF->dispatch_slot:179 双 getattr False 短路->走 collide 返 LINEAGE_CONCEPT_FILL=1->bit-identical（2363 零回归守）。
    saved_cue_slot_fill = gates.CUE_SLOT_FILL_MODE
    # 命门③ 候选 C 第 7 gate（SLOT_LCA_CONSTRAINT_MODE·doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）：内容词位按 slot IS_A LCA 类过滤候选（抽象活化）。
    # 消费 COMPOSES_COMBINE_MODE INSTANTIATES 边（read_instantiates 非 None）+ ATTR_SLOT_ROLE（_cluster_by_lca 写·已 live 零消费者）-> read_slot_lcas 重建 slot_lcas
    # -> dispatch_slot 内容词位 is_a_descendant_of(c, slot_lca) 过滤（reflexive-transitive）。**独立 gate 链**（2 gate·不依赖 cue 链 6 gate·C 独立 2 gate 也可活·生产 B+C 共翻 7 gate 最完整）。
    # 缺此 gate -> 内容词无抽象类约束仍非语义连贯句（B 中间态非终态）= 结构活化无抽象活化 = theater 风险（design §三必需 follow-up·非可选 defer）。
    # gate OFF->dispatch_slot:216 getattr False 短路->candidates 不变->走 collide 返 LINEAGE_CONCEPT_FILL=1->bit-identical。
    saved_slot_lca_constraint = gates.SLOT_LCA_CONSTRAINT_MODE
    # NOTE：FLOOR_ACTIVATION_MODE（floor 端到端激活率·doc/重来_floor_端到端下游激活率_2026-07-17）**不在此 try/finally 翻**
    # ——它是 eval/measurement gate（同 STATISTICAL_WEANING_MODE·env-gated·getattr 读·:2877）·非核心训练变换。
    # 生产 orchestrator `_measure_floor_pass`（observe〔probe_corpus held-out〕+auto_discover〔**不调 tally**〕+generate→
    # measure_floor_activation 读侧后验重导）**defer 课程相位（piece 3）**：须配真 held-out split（probe_holdout>0·课程 run）
    # + 真 curriculum run 才能 e2e 验（同桥生产路径 smoke 验范式）。piece 2 只交付 measure 机制 + gate-gated verdict
    # 接线（weaning.py floor_conjunct·gate OFF→True→bit-identical·审1 严重-1）+ FC1-8 fixture 预验。orchestrator 设计
    # 复杂（held-out INSTANTIATES 桥非 observe 自建 / auto_discover 取 __disc_lang_* 非 __seg_ / REALIZES 独立 gate pass）
    # 见设计档 §9·piece 3 商讨落地。
    gates.REALIZES_MODE = True
    gates.CUE_CLUSTER_MODE = True
    gates.ORACLE_PROMOTE_MODE = True
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.COMPOSES_COMBINE_MODE = True   # 对应桥写侧激活（完成桥·翻 A.3 REJECT·见上 saved_compose 注）
    gates.CUE_SLOT_FILL_MODE = True   # 命门③ 候选 B 第 6 gate·cue 位 cue token 直出（6 gate 同翻末端·见 saved_cue_slot_fill 注）
    gates.SLOT_LCA_CONSTRAINT_MODE = True   # 命门③ 候选 C 第 7 gate·内容词按 slot IS_A LCA 类过滤（抽象活化·见 saved_slot_lca_constraint 注·独立 gate 链末端）
    # STEP5 PR2：operator D:11 readback 第二源（arith_op_of/comparison_op_of 读 D:11 PRIMARY→OP_*→opcode·
    # 镜像 EMERGENT_RELATION_CUE_READBACK_MODE 两源范式·gate OFF 退化纯 frozenset bit-identical）。
    saved_op_readback = gates.OPERATOR_D11_READBACK_MODE
    gates.OPERATOR_D11_READBACK_MODE = True
    # 审计根治 [严重-1]：modal D:11 readback 第二源（modal_op_of/is_modal_cue 读 D:11 PRIMARY→MODAL_KIND→modality·
    # 镜像 OPERATOR_D11_READBACK_MODE 两源范式·gate OFF 退化纯 frozenset _MODAL_CUES bit-identical）。
    saved_modal_readback = gates.MODAL_D11_READBACK_MODE
    gates.MODAL_D11_READBACK_MODE = True
    # #940 否定词 D:11 readback 第二源（is_negation_cue 读 D:11 PRIMARY→TYPE_NEGATION concept·
    # 镜像 MODAL_D11_READBACK_MODE 两源范式·gate OFF 退化纯 frozenset _NEGATION_CUES bit-identical）。
    saved_negation_readback = gates.NEGATION_D11_READBACK_MODE
    gates.NEGATION_D11_READBACK_MODE = True
    # 止血 #1146（methodology §五·reward 非 frame）：CAUSES edge reward 写按域过滤——语言/bare 域剔出
    # reward_propagate 落点① edge 写（reward 结构性 theater·dead-end/veto→tn++ 惩罚唯一 reward-active 边·有害）。
    # 生产翻 ON（语言域 reward 退场·CAUSES 掌握走刀 constructive-check 不接 strength）·CI default OFF 退化现状
    # bit-identical（OFF → reward_propagate 落点① 逐字现状·判据 = shared.REWARD_LEGITIMATE_DOMAINS 与 judge G5 同源）。
    saved_causes_filter = gates.CAUSES_REWARD_DOMAIN_FILTER_MODE
    gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = True
    # STEP5 PR4：EDGE_SIMILAR slot-filler 消费者（dispatch_slot 读 EDGE_SIMILAR 扩展 slot 候选·
    # gate OFF 不扩展 bit-identical·D2 合规非向量·不接 reward）。
    saved_similar_slot = gates.SIMILAR_SLOT_MODE
    gates.SIMILAR_SLOT_MODE = True
    # 审计根治 [严重-3]：B6 生成侧 dispatch_slot pronoun scoring（读 pr_tn 加 slot 候选分·
    # gate OFF 不读 bit-identical·pair-key 对偶 observe 侧·不接 reward·pr_tn sign-agnostic）。
    saved_pronoun_slot = gates.PRONOUN_SLOT_MODE
    gates.PRONOUN_SLOT_MODE = True
    # 刀5 件5：SELECTION_PREF_MODE 在此翻（选择倾向统计 builder·observe 段内共现写 sp_tn·§十 边约束）。
    # 生产 ON 防纸面闭合（selection_pref_count 表生产写真写·非空表 theater）·PR 软加权 dock seed defer S4。
    saved_sel_pref = gates.SELECTION_PREF_MODE
    gates.SELECTION_PREF_MODE = True
    # M1片2：intent 分类替换硬编码（classify_intent·解 G3a 死门·doc/重来_M1片2_intent分类设计_2026-07-08.md）。
    # 生产 ON 防纸面闭合（否则 gate 永不活 → classify_intent 永不调 → is_causal 永假 → reward 退化核心病灶未修 = theater）。
    # 同 CUE/EMERGENT try/finally 守回归（CI gate OFF → reward :366 + H2 :1448 两处硬编码走原路径 bit-identical）。
    saved_m1_intent = gates.M1_INTENT_CLASSIFY_MODE
    gates.M1_INTENT_CLASSIFY_MODE = True
    # 性能修复（2026-07-08 训练测试探索）：COOCCURS 段内配对窗口化 O(L²)→O(L·K)·解训练 scaling 爆炸。
    # 生产 ON 防纸面闭合（否则长文本/真语料训练 71s/n=5 段跑不动）。同 CUE/EMERGENT try/finally 守回归
    # （CI gate OFF → cooccurs.segment_cooccurrence_pairs i<j 全配对 bit-identical 现状）。镜像范式 :1226。
    saved_cooccur_win = gates.COOCCURS_WINDOW_MODE
    gates.COOCCURS_WINDOW_MODE = True
    # 总收口 0.1：跨段去重（COOCCURS_DEDUP_MODE·add_cooccurs_dedup·解 append-only 堆叠 LIVE 病灶①·阻塞 #734）。
    # 生产 ON（镜像 COOCCURS_WINDOW·observe build_cooccurs flip 后走 dedup·reader 读 strength 协同）。
    # gate OFF 时 reader 读 strength 恒 1 等价数行·ON 时 strength=频次；bit-identical 由 gate OFF 单测守。
    saved_cooccur_dedup = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    # S2 dead-end 根因 §10.3：PRECEDES 跨 round 去重（PRECEDES_DEDUP_MODE·add_precedes_dedup·
    # mirror COOCCURS_DEDUP_MODE·解 observe 跨 round 16× 重复·2256→153 distinct）。生产 ON 防纸面闭合
    # （否则 3 builder 走旧 add 堆叠 = 16× 重复未修 = theater）。gate OFF 单测守回归（additive·bit-identical）。
    # 诚实边界：dedup 确定性 perf 16× + 数据卫生赢·未必解 dead-end（AND 语义另议）·dedup 后重测定。
    saved_precedes_dedup = gates.PRECEDES_DEDUP_MODE
    gates.PRECEDES_DEDUP_MODE = True
    # CAUSES 跨 round 去重（CAUSES_DEDUP_MODE·add_causes_dedup·mirror PRECEDES_DEDUP_MODE·解 observe 16× 重复
    # 边膨胀·56% 墙大头）。生产 ON 防纸面闭合（否则 _insert_causes 走旧 add 堆叠 16× = theater）。reward 影响
    # 零核证（snapshot_strengths 覆写去重·16x 与 1x 同 delta）·修假汇聚 bug + 消边膨胀。gate OFF 单测守回归。
    saved_causes_dedup = gates.CAUSES_DEDUP_MODE
    gates.CAUSES_DEDUP_MODE = True
    # PR 热区过滤（HOTZONE_MODE·A3PRWrapper.build BFS k-hop·mirror CAUSES_DEDUP_MODE·解全图 8677²(n=656)）。
    # 生产 ON 防纸面闭合（否则 PR 全图=defer 意外态=theater·设计本意卷二:110 hotzone）。reward 影响零
    # （resolver 证 PR 不回流 path）·仅变 PR 诊断。k=2/PR_MAX_NODES=2048·须配合 PR_B2_LARGE_N_MODE。
    # gate OFF 单测守回归（全图 bit-identical）。
    saved_hotzone = gates.HOTZONE_MODE
    gates.HOTZONE_MODE = True
    # PR_B2_LARGE_N_MODE（生产大 n>512 走 B2 迭代替 B1 O(n³)·CI n<512 走 B1 bit-identical·gate check `ON and n>阈值`）：
    # HOTZONE_MODE ON 缩 matrix 但 k-hop 密集图仍可能 >512·须 B2 防 B1 炸（B2 audit1 LATENT-BUG 修·n<512 时 gate ON
    # 但 n<=阈值->仍 B1 bit-identical）。生产 ON（镜像 HOTZONE_MODE·prior n=656 经 env 设·现落 code 显式）。
    saved_pr_b2 = gates.PR_B2_LARGE_N_MODE
    gates.PR_B2_LARGE_N_MODE = True
    # S2 dead-end factor A：PRECEDES AND→OR（PRECEDES_OR_MODE·a2_stepper 推进语义·dag_path language-only
    # 故 production 安全·解重复词概念多前驱 AND 全 active 永不满足致 dead-end）。生产 ON 防纸面闭合·
    # 外层 try 覆盖 episode_loop + _rebuild_path + H2 标定（同生产语义）。gate OFF 单测守回归（additive）。
    # 诚实：A 修不保证 REACHED_SINK（factor B COVERAGE_THRESHOLD 可能仍挡·A 修后须测）。
    saved_precedes_or = gates.PRECEDES_OR_MODE
    gates.PRECEDES_OR_MODE = True
    # S2 dead-end factor C：PRECEDES oi-first-occ 序遍历（PRECEDES_OI_MODE·F2·doc/重来_F2_PRECEDES_oi遍历_设计_2026-07-09·v3）。
    # factor C = language PRECEDES 概念成环·Kahn 丢环节点含 sink -> sink 永不可达 -> reward 恒 0 -> language 零学习。
    # 生产 ON 防纸面闭合（否则 a2_layer_oi 永不调 = F2 孤儿 = theater·镜像 PRECEDES_OR_MODE 范式）。
    # OI_MODE ON 时 DEDUP+OR 也 ON（三者叠加·解 L1 重复+L2 AND/OR+L3 factor C）。
    # 诚实边界：末段 tokens 在 sink 后不访（1 段损失）+ backward CAUSES 丢 + OR 死锁依赖 ATTRACTOR/EXPLORATION（生产 ON）。
    # gate OFF 单测守回归（a2_layer Kahn·bit-identical）。
    saved_precedes_oi = gates.PRECEDES_OI_MODE
    gates.PRECEDES_OI_MODE = True
    # factor E：层1 同段指代候选（PRONOUN_INTRASEG_MODE·doc/重来_factorE_层1指代_intra_seg_设计_2026-07-09）。
    # factor E = 同段前指代词（"动物...它们"同段）无候选 → dangling → J4 ② fire → G4 veto → reward=0。
    # 生产 ON 防纸面闭合（否则层1 块永不执行 = judge.py:58 注释"层1 已解析"仍是 theater·镜像 PRECEDES_OI_MODE 范式）。
    # 诚实边界：层1 启发式近因非语义消解（stable≠correct）·reward>0=可训练非语义正确。
    saved_pronoun_intraseg = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = True
    # G2 修饰方向A：head 偏好 read-time 加权（MODIFIER_DIRECTION_MODE·dispatch_slot 第 6 路 head_pref_score）。
    # 生产 ON 防纸面闭合（否则 source 写 modification_hist 但 read 不用 = theater）。source write gate-independent。
    saved_modifier_direction = gates.MODIFIER_DIRECTION_MODE
    gates.MODIFIER_DIRECTION_MODE = True
    # B6 指代维 方案3 tn+fn 路（PRONOUN_RESOLVE_COUNT_MODE·count 写在 observe 阶段 resolve_pronoun_occurrence·
    # 非 episode_loop·故在此大 try/finally 翻 ON 盖 observe·同 PRONOUN_INTRASEG_MODE 范式·非 run_round_full 小 try/finally）。
    # 生产 ON 防纸面闭合（否则 resolve 不写 pr_tn/pr_fn + consumer 不读 pr_tn = theater·§九.2 病灶"attribute 给谁"未解）。
    # 诚实边界：指代维 reward=J4 bool veto（非 graded）·consumer 自消费 reward>0 鲁棒·pr_sn 教师 P2 defer·per-occurrence 避 β_arith。
    saved_pronoun_resolve = gates.PRONOUN_RESOLVE_COUNT_MODE
    gates.PRONOUN_RESOLVE_COUNT_MODE = True
    # 归一化半 A：功能词/hub 排除（EXCLUDE_FUNCTION_MODE·read-time hub_degree 过滤 3 点·
    # doc/重来_归一化与功能词排除_设计_2026-07-08.md·对抗挖修正 §二序2）。生产 ON 防纸面闭合（否则 gate
    # 永不活 → 3 点不过滤 → hub 污染 collide_score/_cooccurs_count/refers_occurrence 未修 = theater）。
    # 无条件翻 ON（镜像 M1/COOCCURS_WINDOW·与决断 A4 一致·observe 在 stage loop 内 flip 后才跑·is_hub
    # 调时读 COOCCURS·表未注册 try/except KeyError→False 退化 bit-identical OFF·无除法无 crash）。
    saved_exclude_func = gates.EXCLUDE_FUNCTION_MODE
    gates.EXCLUDE_FUNCTION_MODE = True
    # 刀 A：语言域时序 cue verify episode（TIME_SEQ_PROOF_MODE·run_round_full 语言域路由分支）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 时序验序器孤儿 = theater·镜像 COOCCURS/M1 范式）。
    saved_time_seq = gates.TIME_SEQ_PROOF_MODE
    gates.TIME_SEQ_PROOF_MODE = True
    # 刀 B：语言域数值等式 cue verify episode（NUMERIC_PROOF_MODE·run_round_full 语言域路由分支·numeric priority）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 数值验序器孤儿 = theater·镜像 TIME_SEQ_PROOF_MODE 范式）。
    saved_numeric = gates.NUMERIC_PROOF_MODE
    gates.NUMERIC_PROOF_MODE = True
    # 刀 C：语言域全称量化 cue verify episode（UNIVERSAL_PROOF_MODE·run_round_full 语言域路由分支·numeric>universal>precedes）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 全称验序器孤儿 = theater·镜像 TIME_SEQ/NUMERIC 范式）。
    saved_universal = gates.UNIVERSAL_PROOF_MODE
    gates.UNIVERSAL_PROOF_MODE = True
    # A1·STEP6：语言域存在量化 cue verify episode（EXISTENTIAL_PROOF_MODE·run_round_full 语言域路由分支·
    # numeric>comparison>universal>existential>precedes 序）。生产 ON 防纸面闭合（否则路由永不走 = 存在验序器
    # 孤儿 = theater·镜像 UNIVERSAL 范式·复用 ∀ 的 ConceptNet ext_map·双向祖先 X⊆Y OR Y⊆X）。
    saved_existential = gates.EXISTENTIAL_PROOF_MODE
    gates.EXISTENTIAL_PROOF_MODE = True
    # 刀 D：语言域比较 cue verify episode（COMPARISON_PROOF_MODE·run_round_full 语言域路由分支·numeric>comparison>universal>precedes）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 比较验序器孤儿 = theater·镜像 TIME_SEQ/NUMERIC/UNIVERSAL 范式）。
    saved_comparison = gates.COMPARISON_PROOF_MODE
    gates.COMPARISON_PROOF_MODE = True
    # G1+#774：属性命题 reification（PROPOSITION_MODE·observe build_property_edges 建命题节点+PROPERTY 出边·
    # G3b 全局扫命题节点判矛盾·has_value_claim 真活激活 G3b·反 theater）。生产 ON 防纸面闭合（否则
    # extract_property_claims_gated 永返 [] → property_claims 空 → has_value_claim 永 False → G3b 不激活 =
    # judge.py:236 has_value_claim 门 dead = theater·镜像 TIME_SEQ/NUMERIC/UNIVERSAL 范式）。
    saved_proposition = gates.PROPOSITION_MODE
    gates.PROPOSITION_MODE = True
    # B1 否定收口（#888·STEP1）：NEGATION_MODE ON·否定窗口激活·pol=1 命题节点建独立 surface。
    # 镜像 PROPOSITION_MODE·生产 ON 防纸面闭合（否则 negation_on=False·"不是"错位 skip·否定命题永不建 = theater）。
    # CI corpus 不含"不是" -> 否定窗口不触发 -> 既有测零回归（bit-identical）。
    saved_negation = gates.NEGATION_MODE
    gates.NEGATION_MODE = True
    # B2 情态收口（STEP6 PR2）：MODALITY_MODE ON·情态窗口激活·modality 填值（0-4）·命题节点建独立 surface 后缀 _0_{mod}。
    # 镜像 NEGATION_MODE·生产 ON 防纸面闭合（否则 modality_on=False·"必然是"错位 skip·情态命题永不建 = theater）。
    # CI corpus 不含"必然/可能" -> 情态窗口不触发 -> 既有测零回归（bit-identical）。
    saved_modality = gates.MODALITY_MODE
    gates.MODALITY_MODE = True
    # #1134 程度 intensity 收口：DEGREE_MODE ON·degree 窗口激活（tokens[val_idx] degree cue → value 后移+intensity 填值）·
    # 命题节点建独立 surface 后缀 _i{num}_{den} + ATTR_PROP_INTENSITY=30。镜像 MODALITY_MODE·生产 ON 防纸面闭合
    # （boot 已 populate_degree_cues 喂 cache·OFF 则 is_degree_cue 恒 False·intensity 恒 1/1 = 既有 bit-identical）。
    # **诚实边界**：intensity magnitude 暂无消费者（G3b 读 count·judge 只权 CAUSES/PRECEDES）·consumer defer·dormant 非 theater。
    saved_degree = gates.DEGREE_MODE
    gates.DEGREE_MODE = True
    # 刀6 件7 SENSE_LOOKUP_MODE 已在 _discover_and_recognize_lang_structures（:1035）前翻（clone 段 + observe 共用）。
    # saved_sense_lookup 在 :1032 前·finally（:1203）复位。
    try:
        # E7 pre-flight 放量门接通生产主入口（S12 follow-up·破纸面闭合·config.pre_flight 守）：
        # boot+discovery 完成 + 生产 gates 已翻（CUE/EMERGENT/SELECTION_PREF/SENSE_LOOKUP）→ snapshot 5 状态 →
        # pre_flight trial（STAGE3 reward 路径·representative·内部 try/finally 自管 CUE_EXTRACTOR）→
        # fail raise 禁放量（外层 finally :1231 复位 gates·防泄漏）→ rollback 5 状态（trial 副作用清零·
        # stage loop 从 round_id=0 STAGE1 起 bit-identical）→ result.pre_flight_report=rep（observability）。
        # 默认 OFF（config.pre_flight=False）→ 块整段跳过 → 既有测零改。DictBackend + SQLiteBackend 均支持
        # （snapshot/load_snapshot 接口齐·§施工序 4.2-1·hasattr 守通过·不再 NotImplementedError）。
        # weights 不 rollback（run_round_full 不改 ctx.weights·只 stage loop _h2_calibrate 改·
        # 已核证 cognition/ 零 weights 写·pre_flight 路径无 weights 写）。
        if config.pre_flight:
            if not hasattr(backend, "snapshot"):
                raise NotImplementedError(
                    "backend 缺 snapshot/load_snapshot 接口（pre_flight rollback 须 backend 内部恢复能力·"
                    "DictBackend/SQLiteBackend 已支持·第三方 backend 须自实现·诚实边界·非静默降级）")
            snap_backend = backend.snapshot()
            snap_idpool = dict(getattr(backend, "_id_pool", {}))
            snap_idx = {k: dict(v) for k, v in ctx.concept_index._index.items()}
            snap_loaded = set(ctx.concept_index._loaded_spaces)
            snap_wm = copy.deepcopy(ctx.work_memory)
            pf_rounds = (config.pre_flight_rounds if config.pre_flight_rounds is not None
                         else PRE_FLIGHT_ROUNDS)
            rep = pre_flight(ctx, corpus, rounds=pf_rounds,
                             runner=r, replay_needed=config.replay_needed,
                             config=config, backend_factory=lambda: DictBackend())
            if not rep.passed:
                # fail=禁放量（守几百G不重训红线）。raise 前 backend/ctx 已被 trial 污染（trial 跑建图+
                # reward 写计数器）·**未 rollback**（rollback 仅 pass 路径跑）。caller 须弃 backend 重建·
                # 勿复用（trial 副作用残留）。诚实边界：与 pass 路径 snapshot/rollback 不对称是设计意图
                #（fail 即停·无 stage loop·caller 弃图重试）。
                raise RuntimeError(
                    "pre_flight 放量门失败（禁放量·守几百G不重训红线）：" + str(rep.detail))
            # rollback 5 状态（trial 副作用清零·stage loop bit-identical）
            backend.load_snapshot(snap_backend)
            if hasattr(backend, "_id_pool"):
                backend._id_pool = dict(snap_idpool)
            ctx.concept_index._index = {k: dict(v) for k, v in snap_idx.items()}
            ctx.concept_index._loaded_spaces = set(snap_loaded)
            ctx.work_memory = copy.deepcopy(snap_wm)
            result.pre_flight_report = rep
        for stage in todo_stages:
            if stage == STAGE5_MULTIMODAL:
                # defer·非训练（机制骨架随模态扩展·§十二阶段5）
                result.stages_completed.append(stage)
                continue
            cfg = stage_gate_config(stage)
            active = stage_active_gates(cfg)
            # gate 二分：TRAINING_MODE OFF → reward/promote 不生效·降为 observe-only（bit-identical）
            reward_active = active["reward"]
            # runner 按有效阶段跑（reward 未激活→observe-only·stage<STAGE3）
            eff_stage = stage if reward_active else STAGE2_CAUSES_ABS

            # 阶段3 H2：先小批量标定权重 → 开全量 reward（§十四 H2·鸡生蛋破解）
            if stage == STAGE3_REWARD and reward_active and ctx.teacher is not None:
                ctx.weights = _h2_calibrate(ctx, corpus, r)

            # per-round 执行（reward 未激活→observe-only·runner 返 None）
            items = _stage_items(corpus, stage, config.rounds_per_stage)
            eps: list[Episode] = []
            per_round = config.weaning_round_series   # W7 断点6：per-round series（设计 per-run·解 observe-only 0 混入 bug）
            # #1143 统计层断奶·教师干预测量（fadeout 锚点 data·gate STATISTICAL_WEANING_MODE 守 bit-identical）：
            # _sw_int ON 时 snapshot teacher.call_count 轮边界 delta → intervention_rate/dependency（真测非 stub-0）。
            # OFF（默认 CI）→ delta=0 → intervention_rate=0（既有 bit-identical·2审 HIGH-1 修：fadeout 不再 vacuous）。
            _sw_int = getattr(gates, "STATISTICAL_WEANING_MODE", False)
            _cc_stage_before = (getattr(ctx.teacher, "call_count", 0) if _sw_int else 0)
            for r_idx in range(config.rounds_per_stage):
                _cc_before = (getattr(ctx.teacher, "call_count", 0) if _sw_int else 0)
                batch_eps = _run_round_batch(ctx, r, items, eff_stage, round_id)
                eps.extend(batch_eps)
                _cc_delta = ((getattr(ctx.teacher, "call_count", 0) - _cc_before) if _sw_int else 0)
                _intervention_rate = (_cc_delta * 1000) // max(len(batch_eps), 1)
                if per_round:
                    # W7 断点6：per-round record（每 batch_eps 一次·末4 窗口全 stage3/4 verify flat → plateau True）。
                    # promote/oov 仅 stage4 末轮（_promote_eligible 副作用 tier flip 须一次·stage4 末轮等价既有 stage loop 末）。
                    is_last_r4 = (stage == STAGE4_PROMOTE_WEAN and active["promote"]
                                  and r_idx == config.rounds_per_stage - 1)
                    if is_last_r4:
                        promote_count, oov_promote = _promote_eligible(ctx, teacher)
                        if getattr(gates, "G5_C_CONSOLIDATE_MODE", False) and ctx.memory_read is not None:
                            promote_memory_consolidate(ctx.backend, ctx.memory_read)
                    else:
                        promote_count, oov_promote = 0, 0
                    mc.record_round(
                        round_id, stage, batch_eps,
                        graph_size=_graph_size(ctx),
                        causes_coverage=_causes_coverage(ctx),
                        promote_count=promote_count,
                        oov_promote_count=oov_promote,
                        intervention_rate=_intervention_rate,   # #1143 fadeout 锚点（教师/episode×1000·gate ON 真测）
                        dependency=_intervention_rate,          # #1143 教师依赖 proxy（同源·P2.5 refine）
                        holdout_retention=ctx.holdout_retention,
                        count_g5_self_assess=(ctx.teacher is None and config.simulate_offline_eval),
                    )
                round_id += 1

            # #723 G 归因：跨 stage 累加全 episode（collect_episodes 守·harness 考核时开）
            if config.collect_episodes:
                all_eps.extend(eps)

            # 防塌三柱验收（致命5 生产 caller·每 stage 汇总累加）
            cs = _anti_collapse_summary(eps)
            for k in collapse_acc:
                collapse_acc[k] = collapse_acc.get(k, 0) + cs.get(k, 0)

            # 既有 per-stage 度量记录（weaning_round_series=False bit-identical·同 W6）。
            # per-round 时跳过（内循环已 per-round record）。
            if not per_round:
                # 阶段4 promote 三重（SHADOW→PRIMARY·promote 激活才跑）
                promote_count = 0
                oov_promote = 0
                if stage == STAGE4_PROMOTE_WEAN and active["promote"]:
                    promote_count, oov_promote = _promote_eligible(ctx, teacher)
                    # #732 G5-C 记忆项延迟晋升闸（gate ON 时·STAGE4_PROMOTE_WEAN 末 offline 扫 memory_item by info_ref·
                    # sum(sc)/sum(count) 比率门判达 → consolidate flip status EXPERIENCE→CONSOLIDATED·消费者部分活·解⑧a 训练侧）。
                    # **三个 G5 同名不同物**：G5-C 本 caller（记忆项 status flip）/ G5-B _promote_eligible（边级 tier flip）/ G5-A judge 自证机。
                    # gate default OFF 守 bit-identical·capability_exam ⑧ consumer_triggers 实查 CONSOLIDATED 行数（不依赖返值）。
                    if getattr(gates, "G5_C_CONSOLIDATE_MODE", False) and ctx.memory_read is not None:
                        promote_memory_consolidate(ctx.backend, ctx.memory_read)
                # #1143 统计层断奶·stage 级教师干预测量（gate ON·镜像 per-round·_cc_stage_before 在 loop 前 snapshot）
                _cc_stage_delta = ((getattr(ctx.teacher, "call_count", 0) - _cc_stage_before) if _sw_int else 0)
                _intervention_rate_stage = (_cc_stage_delta * 1000) // max(len(eps), 1)
                mc.record_round(
                    round_id, stage, eps,
                    graph_size=_graph_size(ctx),
                    causes_coverage=_causes_coverage(ctx),
                    promote_count=promote_count,
                    oov_promote_count=oov_promote,
                    intervention_rate=_intervention_rate_stage,   # #1143 fadeout 锚点（gate ON 真测）
                    dependency=_intervention_rate_stage,          # #1143 教师依赖 proxy
                    holdout_retention=ctx.holdout_retention,   # W4 D4 track（默认 0 bit-identical·W6 模拟退场 eval 采真值·D1 曲线②）
                    count_g5_self_assess=(ctx.teacher is None and config.simulate_offline_eval),
                )

            # 度量门控（合格才进下阶段·防缺防超喂）
            snap = mc.snapshot()
            if stage_metric_gate(stage, snap):
                mark_completed(state, stage, skippable=is_skippable(stage))
                result.stages_completed.append(stage)
            else:
                # 未达标：停留本阶段（不进下·守几百G不重训·度量门控诚实）
                result.stages_completed.append(stage)   # 已跑（未达标记完成轮次·不进下）
                break

        # 阶段4 断奶判据（D1-D5/E2 六闸门·#358 完整实现·非布尔阈值·非只看 4 能力指标平台）
        if STAGE4_PROMOTE_WEAN in result.stages_completed:
            # W5 D5 Mode B 预验台账：stage4 末并行 Mode A vs B 评估·写 calibration 台账
            # （config.calibrate_mode_b=True 触发·默认 False 守 bit-identical·台账空→mode_b_prevalidated=False）。
            # 须在 :2068 mode_b_prevalidated(backend) 读台账前填·故置所有 D-check 前。
            # calibration_set = corpus（training subset·W4 split 后·已 observe 学过·非 held-out probe）。
            if config.calibrate_mode_b:
                _run_calibration_phase(ctx, corpus, backend)
            # W6 E2 模拟退场 eval：stage4 末·weaning_check 之前·解 teacher_offline 循环依赖（预验非后验）。
            # 读 ctx.probe_corpus（W4 held-out 探针·须配 probe_holdout>0）→ observe 探针建学树 + cross_verify_pair
            # （零教师自锚）→ 采 holdout_retention 真值 + 设 ctx.e2_eval_passed（算术域三条件 and）。
            # 默认 simulate_offline_eval=False → 不跑 → ctx.e2_eval_passed 默认 False → E2 blocker（同既有·零翻）。
            # 算术域（teacher=None）跑 eval·语言域（teacher≠None）eval no-op（defer W8 真翻 MODE_OFF）。
            if config.simulate_offline_eval:
                _run_simulated_offline_eval(ctx, corpus, backend)
                # W7 断点1：retention backfill（eval 在 record_round 后跑·series baked 0·回填真值解 D1 曲线②
                # _retention_stable 读全 0 永不过 FLOOR_RETENTION·W6 memory 标的风险·W7 修）。
                # 仅 ctx.holdout_retention>0（eval 采到真值·默认 OFF 不调·bit-identical）。
                if ctx.holdout_retention > 0:
                    mc.backfill_retention(ctx.holdout_retention)
            # W7 断点3：D3 路径 B 域特化分支（镜像 E2·算术域 teacher=None 绕 judge→ctx.judge_source_independent
            # 永默认 False·source_independence.py:34 判定接口 W3 建生产零调用）。simulate_offline_eval=True 时接
            # judge_source_independent_arith（verify_uses_vm_proof=True·teacher_not_judge=True）→ D3 算术域过。
            # 默认 OFF 不接→ctx.judge_source_independent 默认 False→D3 blocker（同 W6·bit-identical）。
            if ctx.teacher is None and config.simulate_offline_eval:
                from pure_integer_ai.teacher.source_independence import judge_source_independent_arith
                ctx.judge_source_independent = judge_source_independent_arith(
                    verify_uses_vm_proof=True, teacher_not_judge=True)
            # D2 负通路活跃：调 convergence.neg_pathway_active_from（单点源·B2 接·消内联重复·
            # 与 convergence_check.neg_pathway_active 同源 failure_count>0 M7 同口径·不重复实现）
            from pure_integer_ai.cognition.result.convergence import neg_pathway_active_from
            neg_pathway_active = neg_pathway_active_from(eps)
            # D3 裁判源独立：W3 ctx track（build_judge_fn :463 算 sources_disjoint({j_sid},{teacher_sid}) 设此·
            # 单一真相源·非硬编同源重算）。默认 None→j_sid 回落 teacher_sid→同源 False（bit-identical）。
            # caller 传独立 judge_source_id（与 teacher.source_id 不相交）→ True（D3 通用路径过·W8 语言域独立裁判）。
            # 算术域绕 judge（_run_verify_round:374）judge_fn 不构建→ctx.judge_source_independent=False→D3 仍 False（W7 接 judge_source_independent_arith）。
            judge_source_independent = ctx.judge_source_independent
            # D4 探针集隔离：W4 ctx track（主入口 :1611 切 corpus + is_disjoint 算此·单一真相源·非硬编）。
            # 默认 probe_holdout=0→不切→ctx.probe_set_disjoint=False（bit-identical·同原硬编）。
            # caller 传 probe_holdout>0→切末尾 N held-out→probe_refs∩training_refs=∅→True（D4 通用路径过）。
            # D4 域无关（corpus 切分）→ 算术域走通用 ctx track 过·不似 D3 须 judge_source_independent_arith 判定接口。
            probe_set_disjoint = ctx.probe_set_disjoint
            # D5 Mode B 预验：当前无 calibration 记录（Mode A vs B 并行预验待真训练 run）→ False·诚实
            from pure_integer_ai.teacher.weaning_calibration import mode_b_prevalidated
            mode_b_ok = mode_b_prevalidated(backend)
            # E2 教师下线独立产出：最硬闸门·域特化分支。
            # 算术域（teacher=None）：读模拟退场 eval 结果（ctx.e2_eval_passed·_run_simulated_offline_eval 设·
            #   镜像 D3/D4 ctx track 范式·单一真相源·非硬编）。eval 在上方跑（weaning_check 之前·预验·解循环依赖）。
            # 语言域（teacher≠None）：通用 e2_execution_ready()（仍 False·真翻 MODE_OFF defer W8）。
            if ctx.teacher is None:
                e2_passed = ctx.e2_eval_passed
            else:
                from pure_integer_ai.teacher.weaning_e2 import e2_execution_ready
                e2_passed = e2_execution_ready()

            # W7 断点7：算术域 floor override（COMPOSES 直接 PRIMARY·oov_promote 恒 0·FLOOR_OOV_PROMOTE 不适用·
            # 架构事实非 vacuous·非 SHADOW→promote 流程）。语言域（teacher≠None）走 SHADOW→promote·None 原值 bit-identical。
            # 非 flag-gated（基于 ctx.teacher is None 域判定·非 W7 feature toggle）·默认 OFF 时 inert
            # （judge_self=0<FLOOR_JUDGE_SELF 恒败·floor override 不改 floors_met）·e2e 全 flag ON 时 critical。
            from pure_integer_ai.teacher.weaning import METRIC_OOV_PROMOTE
            floor_ov = ({METRIC_OOV_PROMOTE: 0} if ctx.teacher is None else None)
            rep = weaning_check(
                mc.weaning_series(),
                neg_pathway_active=neg_pathway_active,
                judge_source_independent=judge_source_independent,
                probe_set_disjoint=probe_set_disjoint,
                mode_b_prevalidated=mode_b_ok,
                e2_passed=e2_passed,
                floor_overrides=floor_ov,
            )
            result.weaning_ready = rep.ready
            # D2 断奶点真驱动消费（去 theatrical·非算完即弃）：weaning_ready=True 驱动断奶切换
            # （teacher MODE_OFF + weaning_phase POST·下 run 新 run_id 退场）·False 显式标注未断奶原因
            # 进训练日志（run_id 下·不静默）。当前 gate 全 OFF + E2 未就位→weaning_ready 永 False·
            # 驱动路径接通但执行条件未就位（无真训练 run）·诚实标注。
            if rep.ready:
                ctx.weaning_phase = WEANING_POST   # 断奶切换（下 run 退场·自锚 J1-J4 + C6 Mode B）
                if ctx.teacher is not None:
                    ctx.teacher._mode = MODE_OFF   # 教师退场（断奶后·D 墙·新遇未知靠 SHADOW+晋升闸）
            result.weaning_blockers = _weaning_blockers(rep)   # 诚实标注未断奶原因（不静默）
            # #1143 语言域统计层断奶判定（5判据+5锚点·非 can_ween·gate STATISTICAL_WEANING_MODE·另建 verdict）：
            # weaning_ready=can_ween 须 E2（语言域永 False·正交）·此为独立第三层 verdict（weaker-than-can_ween）。
            # 复用 mc.weaning_series() + 既有 ctx/eps/result 信号。**measured-guard**（2审 HIGH-1/3）：
            # fadeout_measured=False（P2 未建 intervention 聚合·诚实·锚4 不过）·heldout 来自 result.lang_generalization
            # （已算·lang_rate=held-out 识别率·非 arith-only holdout_retention field）。gate OFF → 字段默认·CI bit-identical。
            if getattr(gates, "STATISTICAL_WEANING_MODE", False):
                from pure_integer_ai.teacher.weaning import language_statistical_weaning_check
                _lg = getattr(result, "lang_generalization", None)
                _heldout_measured = _lg is not None
                _heldout_rate = _lg.lang_rate_permille if _lg is not None else 0
                # floor_overrides bug 修（设计 §1 命门3 + §2.4·审1 MEDIUM-3）：语言域 oov_promote 结构性低
                # （止血① 后 SHADOW→promote 计入 oov_promote_count·promote.py:230-244·但 rate=count/max(graph_size,1)·
                # metrics.py:157·graph_size 大→rate 低·恒卡 FLOOR_OOV_PROMOTE=100）→ floors_met 恒 False。
                # domain-aware override 镜像 arith {OOV:0}（arith teacher=None 已有 floor_ov :2845·语言 teacher≠None 须此）。
                # downstream_activation 真信号替代（floor 机制·orchestrator defer piece 3）。
                _floor_ov_lang = ({METRIC_OOV_PROMOTE: 0} if ctx.teacher is not None else None)
                # floor 端到端下游激活率 orchestrator（piece 3·FLOOR_ACTIVATION_MODE env-gated·非 try/finally 翻·
                # 同 STATISTICAL_WEANING_MODE·eval/measurement gate 非核心训练变换）。**gate OFF（CI default）→ 不调 →
                # _floor=default FloorActivation(measured=False/0/0) → weaning.py floor_conjunct gate-gated True →
                # bit-identical（守 SW2/9/16·审1 严重-1·piece 2 FC7 验）**。gate ON（生产）→ orchestrator 真跑 →
                # 3 参 wire floor_conjunct（measured + activation≥阈 + fp≤阈）。**recognition measure**（审2 MEDIUM-1）·
                # 非 generation activation（后者 defer Phase F）。
                _floor = FloorActivation()   # default measured=False/0/0（gate OFF / probe_holdout=0 早返路径）
                if getattr(gates, "FLOOR_ACTIVATION_MODE", False):
                    _floor = _measure_floor_pass(ctx, backend, ctx.concept_graph)
                _sw_rep = language_statistical_weaning_check(
                    mc.weaning_series(),
                    encoding_grounded=True,                              # ①前置（码点接地 P0a done·语言域 foundation·TC2 验）
                    crosslingual_seeded=(getattr(result, "alias_edges_seeded", 0) > 0),  # ⑤前置（PURE_ALIAS 桥种边）
                    probe_set_disjoint=probe_set_disjoint,              # D4 held-out split（ctx track·已算 :2771）
                    neg_pathway_active=neg_pathway_active,              # 锚3 D2 硬 gate（已算 :2761·closes permissive）
                    teacher_present=(ctx.teacher is not None),          # 锚4：教师在场？（None=无教师=结构性独立·有教师=须 measured fadeout）
                    fadeout_measured=(ctx.teacher is not None),         # 锚4 measured（教师在场→intervention 已聚合·gate ON 真测非 stub·2审 HIGH-1 修）
                    heldout_measured=_heldout_measured,                 # 锚5 measured（lang_generalization 算了即 True·lang_rate）
                    heldout_generalization_permille=_heldout_rate,
                    floor_overrides=_floor_ov_lang,                     # 语言域 oov_promote 结构性低 override（审1 MEDIUM-3）
                    floor_measured=_floor.measured,                     # 锚6 floor recognition measure（piece 3 orchestrator·gate ON 真测）
                    floor_activation_permille=_floor.activation_permille,
                    floor_false_positive_permille=_floor.false_positive_permille,
                )
                result.statistical_weaning_ready = _sw_rep.statistical_ready
                result.statistical_weaning_report = _sw_rep

        # §8.7-全 生成侧全环·task-driven L8 episode（stage loop 后·code 树已 observe 建·code_struct_ref 就位）：
        # arith execute（op_confidence 择优·skeleton(新 args)==expected）+ #730 路径 W code unparse（gate
        # CODE_UNPARSE_MODE·unparse COMPOSES→源码串 normalize==code_source·反 theater·序化器真消费者）·两模态合计
        # result.generate。CODE_UNPARSE_MODE 生产翻 ON（反 theater·否则 unparse_composes 孤儿·审 P0·try/finally
        # 守回归·CI 默认 OFF·formal_train 生产路径无条件翻 ON·同 ATTRACTOR/M1/归一化 范式）。
        # Phase 0.1 boot-inject：符号数学语料（transform_rules.txt + inverse_relations.txt·local_dir loader）
        # → 包 CollectedItem 挂 transform_specs/inverse_relation_specs → 注入 task-driven corpus（非 observe 数据·
        # task-driven 直读 specs·不进 observe 建图·镜像 TC11/IR5 测试 CollectedItem 构造范式）。
        # 文件不存在（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve 返 [] → 不 append → bit-identical。
        # doc/重来_阶段断奶路线详设_2026-07-15 §二 Phase 0.1。非 lang-keyed 单文件（同 alias_facts）。
        _phase0_xform = resolve_transform_rules()
        _phase0_inv = resolve_inverse_relations()
        if _phase0_xform or _phase0_inv:
            from pure_integer_ai.cognition.shared.types import MODALITY_ARITH, DOMAIN_MATH, LANG_NONE
            from pure_integer_ai.storage.edge_store import SOURCE_MATH
            corpus.append(CollectedItem(
                modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                source=SOURCE_MATH,
                transform_specs=tuple(_phase0_xform),
                inverse_relation_specs=tuple(_phase0_inv)))
        # Phase 0.2：符号数学 S5-S8 gate 生产翻 ON（task-driven 消费·同 code_unparse 子 try·doc §二 Phase 0.2）。
        # CI 无 specs（boot-inject 不进·无 ZERO_AI_LOCAL_DIR）→ block `_xform_items`/`_rel_items` 空 → no-op → bit-identical。
        # 既有 symbolic 测试（TC11/TC12/IR5/IR6）直调 _run_task_driven_generate 显式控 gate·不受此生产 flip 影响。
        saved_code_unparse = gates.CODE_UNPARSE_MODE
        gates.CODE_UNPARSE_MODE = True
        saved_sym_xform = gates.SYMBOLIC_TRANSFORM_MODE
        gates.SYMBOLIC_TRANSFORM_MODE = True
        saved_sym_rel = gates.SYMBOLIC_RELATION_MODE
        gates.SYMBOLIC_RELATION_MODE = True
        try:
            _gen_eps, result.generate = _run_task_driven_generate(ctx, corpus, _all_ops)
        finally:
            gates.CODE_UNPARSE_MODE = saved_code_unparse
            gates.SYMBOLIC_TRANSFORM_MODE = saved_sym_xform
            gates.SYMBOLIC_RELATION_MODE = saved_sym_rel
        # §8.7-全 生成侧全环度量（反 theater ③下游读者锚·OutputResult.parts→metrics generate_verified 计数真读）：
        # task-driven generate episodes → metrics generate_verified/generate_total 独立 jsonl 行（stage=0）。
        # 不入 weaning_series（非断奶曲线）/ 不更新 _last snapshot（非阶段门控）·纯观测信号行。
        # stage loop 后记（round_id = 已跑轮次·generate 行最后·与阶段行 disambiguate·非断奶曲线不入 series）。
        if _gen_eps:
            mc.record_generate_round(round_id, _gen_eps)
        # #723 G 归因：Mode A task-driven episodes 累加（input=None 路径·G5=False 硬编码·归 ⑤ Mode A 部分）
        if config.collect_episodes:
            all_eps.extend(_gen_eps)

        # 阶段2 通识 base_freq 注入（experience_count·断奶前教师录制期·observe 全语料后 lookup 命中 token concept）
        # 镜像 EPI_LLM_CONFIRM 断奶后退场·断奶后新概念无 base_freq 只靠 exp 自积累·first-write-wins 幂等
        _inject_base_freq(ctx, corpus)

        # 终 dump（per-space·新 run_id·E1 权威 base）
        result.dump_spaces = dump_run(
            backend, config.run_dir, config.run_id,
            spaces=[ctx.space_id], tables=config.dump_tables)
        # E8：持久化 cursor state（已完成阶段集·供下次续训 stage-skip）
        _save_cursor(state, config.run_dir, config.run_id)
        result.final_metrics = mc.snapshot()
        result.holdout_retention = ctx.holdout_retention   # W6 E2 模拟退场 eval 采的保持率真值（默认 0 bit-identical）
        result.weights = ctx.weights
        result.collapse_summary = collapse_acc
        # #723 G 归因：挂全 episode 列表（collect_episodes=True 时·harness 考核读 G_meta 5 字段建交叉表）
        if config.collect_episodes:
            result.episodes = all_eps
        # 续训 cursor state 记 skipped（resume 时跳过）
        result.stages_skipped = [s for s in STAGES if s not in todo_stages]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.ORDINAL_SURFACE_MODE = saved_ordinal_surface
        gates.DISPATCH_TOKEN_CHAIN_MODE = saved_dispatch_tokens   # P0 #1040 generate-dispatch 复位
        gates.OUTPUT_WORD_REWARD_MODE = saved_output_word_reward   # P0 #1041 reward truthiness 校准复位
        gates.G5_C_CONSOLIDATE_MODE = saved_g5c   # G5-C memory consolidate 复位（P1 激活）
        gates.EMERGENT_RELATION_HYPOTHESIS_MODE = saved_emerg_hyp
        gates.EMERGENT_RELATION_FEED_MODE = saved_emerg_feed
        gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = saved_causes_filter   # 止血 #1146 复位
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved_emerg_readback
        gates.REALIZES_MODE = saved_realizes   # 对应泛化 v2 三 gate 复位
        gates.CUE_CLUSTER_MODE = saved_cue_cluster
        gates.ORACLE_PROMOTE_MODE = saved_oracle_promote
        gates.CORRESPONDENCE_SLOT_MODE = saved_corr_slot   # 对应桥第 4 gate 复位（v2 四 gate 共翻）
        gates.COMPOSES_COMBINE_MODE = saved_compose   # 对应桥写侧第 5 gate 复位（完成桥·翻 A.3 REJECT·见上 saved_compose 注）
        gates.CUE_SLOT_FILL_MODE = saved_cue_slot_fill   # 命门③ 候选 B 第 6 gate 复位
        gates.SLOT_LCA_CONSTRAINT_MODE = saved_slot_lca_constraint   # 命门③ 候选 C 第 7 gate 复位
        gates.OPERATOR_D11_READBACK_MODE = saved_op_readback
        gates.MODAL_D11_READBACK_MODE = saved_modal_readback
        gates.NEGATION_D11_READBACK_MODE = saved_negation_readback
        gates.SIMILAR_SLOT_MODE = saved_similar_slot
        gates.PRONOUN_SLOT_MODE = saved_pronoun_slot
        gates.SELECTION_PREF_MODE = saved_sel_pref
        gates.M1_INTENT_CLASSIFY_MODE = saved_m1_intent
        gates.COOCCURS_WINDOW_MODE = saved_cooccur_win
        gates.COOCCURS_DEDUP_MODE = saved_cooccur_dedup
        gates.PRECEDES_DEDUP_MODE = saved_precedes_dedup
        gates.CAUSES_DEDUP_MODE = saved_causes_dedup
        gates.HOTZONE_MODE = saved_hotzone
        gates.PR_B2_LARGE_N_MODE = saved_pr_b2
        gates.PRECEDES_OR_MODE = saved_precedes_or
        gates.PRECEDES_OI_MODE = saved_precedes_oi
        gates.PRONOUN_INTRASEG_MODE = saved_pronoun_intraseg
        gates.MODIFIER_DIRECTION_MODE = saved_modifier_direction
        gates.PRONOUN_RESOLVE_COUNT_MODE = saved_pronoun_resolve
        gates.EXCLUDE_FUNCTION_MODE = saved_exclude_func
        gates.TIME_SEQ_PROOF_MODE = saved_time_seq
        gates.NUMERIC_PROOF_MODE = saved_numeric
        gates.UNIVERSAL_PROOF_MODE = saved_universal
        gates.EXISTENTIAL_PROOF_MODE = saved_existential
        gates.COMPARISON_PROOF_MODE = saved_comparison
        gates.PROPOSITION_MODE = saved_proposition
        gates.NEGATION_MODE = saved_negation
        gates.MODALITY_MODE = saved_modality
        gates.DEGREE_MODE = saved_degree
        gates.SENSE_LOOKUP_MODE = saved_sense_lookup
        if own_metrics:
            mc.close()
    return result


# ---- H2 小批量标定 ----

def _h2_calibrate(ctx: TrainContext, corpus: list[CollectedItem],
                  runner: RoundRunner) -> JudgeWeights:
    """阶段3 H2：小批量离线标定 judge 权重（教师 GT 经录放层·§十四 H2）。

    小批量跑 observe + episode（默认权重）→ 收集 CalibrationSample → calibrate_weights。
    标定用录放层 ground-truth 零 LLM（MODE_REPLAY）·运行时判据自锚输入（两时相不同非矛盾）。
    """
    # 刀 A：H2 标定期间翻 TIME_SEQ_PROOF_MODE OFF（时序 verify 分流的语言项产空 output·judge G2p veto
    # reward=0 对齐 GT=1 不一致·污染 JudgeWeights 标定·对抗审 P1-1）。H2 是 judge 标定·时序 verify 绕 judge
    # 不参与标定·翻 OFF 让语言项正常 episode_loop。镜像 _is_verify_modality :1584 排除防御。
    # 复位在 :1615 return 前·外层 :1469 finally 兜底（异常时）。
    saved_time_seq = gates.TIME_SEQ_PROOF_MODE
    gates.TIME_SEQ_PROOF_MODE = False
    # 刀 B：同上·数值 verify 分流亦绕 judge·H2 标定翻 NUMERIC_PROOF_MODE OFF（镜像时序·防标定污染）。
    saved_numeric = gates.NUMERIC_PROOF_MODE
    gates.NUMERIC_PROOF_MODE = False
    # 刀 C：同上·全称 verify 分流亦绕 judge·H2 标定翻 UNIVERSAL_PROOF_MODE OFF（镜像时序/数值·防标定污染）。
    saved_universal = gates.UNIVERSAL_PROOF_MODE
    gates.UNIVERSAL_PROOF_MODE = False
    # A1·STEP6：同上·存在 verify 分流亦绕 judge·H2 标定翻 EXISTENTIAL_PROOF_MODE OFF（镜像 universal·防标定污染）。
    saved_existential = gates.EXISTENTIAL_PROOF_MODE
    gates.EXISTENTIAL_PROOF_MODE = False
    # 刀 D：同上·比较 verify 分流亦绕 judge·H2 标定翻 COMPARISON_PROOF_MODE OFF（镜像时序/数值/全称·防标定污染）。
    saved_comparison = gates.COMPARISON_PROOF_MODE
    gates.COMPARISON_PROOF_MODE = False
    # G1+#774 PROPOSITION_MODE **不在此翻 OFF**（异 TIME_SEQ/NUMERIC/UNIVERSAL 三刀·对抗审2 发现 A）：
    # 三刀是 verify 路由分流（绕 judge·产空 output·G2p veto reward=0 对齐 GT=1 不一致污染 JudgeWeights 标定）·
    # 须翻 OFF 让语言项正常 episode_loop 进 judge 标定。PROPOSITION_MODE 不路由绕 judge——它只在 observe 建命题
    # 节点 + judge 内部激活 G3b hard veto 乘法门（不进 J 加权）·H2 标定**应让 G3b 参与 judge**（权重适应真实 judge
    # 行为·含 G3b veto）。教师 GT=1 + G3b veto reward=0 是教师与机制分歧·calibrate_weights 网格搜索时此类样本
    # agreement=False·与 J 权重无关·属噪声样本不通过调 J 修复。故 PROPOSITION_MODE 保持 ON（生产态）标定。
    batch = corpus[:H2_CALIB_BATCH]
    samples: list[CalibrationSample] = []
    for rid, item in enumerate(batch):
        # verify-driven COMPOSES 模态（code/arith）排除 H2 标定——reward 经 vm_proof_fn 不用 JudgeWeights·
        # 且进 language judge()→G2p veto reward=0 对齐 GT=1=垃圾标定污染 JudgeWeights（doc §九·_is_verify_modality）。
        if _is_verify_modality(item.modality):
            continue
        res = runner.run_round_full(ctx, item, STAGE3_REWARD, rid) \
            if hasattr(runner, "run_round_full") else None
        if res is None or res.episode is None or res.dag_path is None:
            continue
        # ★M1片2：intent 从硬编码 INTENT_QUESTION 升级为 classify_intent（gate ON 时）。segments 从
        # _split_item_to_segments 重切（与 reward 阶段 :320 同源·确定性·纯读不建图）·sink 从 res.dag_path
        # 派生（calibrate_weights/judge 不读 sink·值无害）。gate OFF 走原硬编码（bit-identical）。
        if getattr(gates, "M1_INTENT_CLASSIFY_MODE", False):
            from pure_integer_ai.cognition.understanding.intent_classify import classify_intent
            h2_segments = _split_item_to_segments(
                item, backend=ctx.backend, edge_store=ctx.edge_store,
                space_id=ctx.space_id, concept_index=ctx.concept_index)
            h2_intent = classify_intent(
                res.dag_path.sink, h2_segments,
                backend=ctx.backend, edge_store=ctx.edge_store,
                space_id=ctx.space_id, concept_index=ctx.concept_index)
        else:
            h2_intent = IntentType(type=INTENT_QUESTION)
        samples.append(CalibrationSample(
            output=res.output,
            dag_path=res.dag_path,
            input_payload=InputPayload(
                segments=[], source=0, stage=STAGE_TRAINING,
                weaning_phase=ctx.weaning_phase,
                intent=h2_intent,
            ),
            graph=ctx.concept_graph,
            workmem=ctx.work_memory,
        ))
    gates.TIME_SEQ_PROOF_MODE = saved_time_seq   # 复位（H2 后生产 stage 继续用·外层 :1469 finally 兜底异常）
    gates.NUMERIC_PROOF_MODE = saved_numeric    # 复位（刀 B·同 time_seq·H2 后生产 stage 继续用）
    gates.UNIVERSAL_PROOF_MODE = saved_universal  # 复位（刀 C·同 time_seq/numeric·H2 后生产 stage 继续用）
    gates.EXISTENTIAL_PROOF_MODE = saved_existential  # 复位（A1·STEP6·同 universal·H2 后生产 stage 继续用）
    gates.COMPARISON_PROOF_MODE = saved_comparison  # 复位（刀 D·同 time_seq/numeric/universal·H2 后生产 stage 继续用）
    if not samples:
        return ctx.weights
    judge_fn, teacher_gt = _make_calib_judge_fn(ctx.teacher, ctx.weaning_phase)
    return calibrate_weights(samples, judge_fn, teacher_gt)


# ---- 刀4 涌现关系学习（子环1+2 涌现钩子·reward 阶段 observe 后 episode 前） ----

def _resolve_emergent_excluded_refs(ctx: TrainContext, lang: int) -> set:
    """刀4：C9-bis §D 候选池排除清单 = `_CUE_WORDS ∪ _REL_LEXICAL_CUE` surfaces（全 lang）→ ConceptRefs。

    防 reward 调固化件（"导致"等已种词不涌为新关系）。surfaces 静态（元定义固化）·ConceptRefs
    按 observe 后已概念化的词 lookup（未 observe 的 cue 词 lookup None·skip·诚实）。

    **对抗审 RISK-1/4 修**：扫全 lang 集（非仅当前 lang）·因 generate_emergent_hypotheses 扫全 space
    PRECEDES（跨 lang 段）·单 lang 漏排他 lang cue 词（混合 lang 语料风险）。
    """
    from pure_integer_ai.cognition.understanding.cue_words import _CUE_WORDS
    from pure_integer_ai.cognition.understanding.word_concept_signal import _REL_LEXICAL_CUE
    surfaces: set[str] = set()
    for _lang_words in _CUE_WORDS.values():
        for _words in _lang_words.values():
            surfaces.update(_words)
    for _lang_words in _REL_LEXICAL_CUE.values():
        surfaces.update(_lang_words.keys())
    excluded: set = set()
    for s in surfaces:
        ref = ctx.concept_index.lookup(s, ctx.space_id)
        if ref is not None:
            excluded.add(ref)
    return excluded


def _run_emergence_hook(ctx: TrainContext, lang: int) -> None:
    """刀4 子环1+2：涌现假设生成（PRECEDES 链 connector 定位）+ D:11 SHADOW 落边。

    generate_emergent_hypotheses 扫 PRECEDES/COOCCURS/CAUSES 签名 → 候选词 w（非-cue·connector）
    → record_emergent_relation_signal_shadow 落 D:11 SHADOW 边（sign=0·staging·§8.1c-bis 合法通路）。
    后续 episode_loop→propagate_reward（gate FEED_MODE ON）feed 候选词 experience_count·
    stage4 _promote_eligible 双轨 promote（experience 主导）。所有步骤幂等（record_shadow query_from skip）。
    """
    from pure_integer_ai.cognition.understanding.emergent_relation_signal import (
        generate_emergent_hypotheses, record_emergent_relation_signal_shadow,
    )
    excluded = _resolve_emergent_excluded_refs(ctx, lang)
    hyps = generate_emergent_hypotheses(
        ctx.backend, ctx.edge_store, ctx.concept_index,
        space_id=ctx.space_id, excluded_word_refs=excluded)
    for w_ref, _rel_kind, rel_ref in hyps:
        record_emergent_relation_signal_shadow(
            ctx.edge_store, w_ref, rel_ref, space_id=ctx.space_id)


def _feed_action_experience(*, backend, edge_store, space_id: int, concept_index,
                            segments, domain: int, modality: int,
                            intent_type: int, reward: int, terminal: int) -> None:
    """B-PR2 动作意图经验回写（doc §17·ACTION_* concept 动作验证率·reward round episode_loop 后调·对偶 op_confidence）。

    D3 激活（intent_type==INTENT_COMMAND + terminal==TERMINAL_REACHED_SINK）→ collect_action_intent_concepts 扫
    segments D:11 PRIMARY → distinct ACTION_* refs → record_experience_outcome 写 experience_count（R1 符号）。
    ctx_code = pack_ctx_code(domain, modality, 0, intent_type)（task=0 defer·同 _ctx_tag·写桶==episode_loop :82 读桶·设计审 F）。

    **reward>0 = R1 成功臂非排除闸**（设计审 B CONFIRMED·§17.1 决断2）：reward>0→e_sn++&e_tn++ / reward==0 veto→e_tn++ only →
    率<1 有判别力（硬排除→率恒1 β_arith 病→B-PR2 无意义）。

    **gate ACTION_EXPERIENCE_FEED_MODE**：OFF → 早返零写（caller 已 if 守·本 helper 内再守·双保险 bit-identical）。
    **依赖 INTENT_COMMAND_MODE**：OFF → intent_type 永 INTENT_QUESTION → D3 永假 → 零写（B-PR2 单测须同翻两 gate）。

    **原语参数**（非 ctx/raw/ep 对象）→ 单测直调免跑全 episode（D3 + R1 + 桶隔离 反 theater 测）。
    """
    if not getattr(gates, "ACTION_EXPERIENCE_FEED_MODE", False):
        return   # 双保险（caller 已 if 守）
    if intent_type != INTENT_COMMAND or terminal != TERMINAL_REACHED_SINK:
        return   # D3 失败·早返（不计 ctx_code·零开销）
    from pure_integer_ai.cognition.understanding.cue_words import collect_action_intent_concepts
    from pure_integer_ai.storage.experience_count import record_experience_outcome
    _ctx_code = pack_ctx_code(domain, modality, 0, intent_type)   # task=0 defer·同 _ctx_tag 写桶==读桶
    for _act_ref, _act_kind in collect_action_intent_concepts(
            segments, backend=backend, edge_store=edge_store,
            space_id=space_id, concept_index=concept_index):
        record_experience_outcome(backend, ref=_act_ref, reward=reward, ctx_code=_ctx_code)


def _collect_action_seed_candidates(*, segments, backend, edge_store, space_id: int,
                                     concept_index, intent_type: int,
                                     ctx_code: int) -> list[ConceptRef]:
    """B-PR4 动作词种子候选预算（doc §19·_run_reward_round episode_loop 前调·写 workmem.action_seed_candidates）。

    扫 segments tokens → concept_index.lookup → lookup_word_action(D:11 PRIMARY) → per action_ref
    read_experience_count(ctx_code) → **洗净 filter（sn==0 tested-never-verified 滤除·ACTIVE 率消费者·非 theater）**
    + rate（None 冷启动→0 给机会 / sn>0→sn×1000//(sn+tn)）→ 收 (rate, word_ref) → stable sort 率降序 → 返 word_ref list。

    **种子=动作词概念（D:11 源端 word concept·语言域 token·有 PRECEDES/CAUSES 边·在 PR matrix）·非 ACTION_* 元概念**
    （D:11 目标端 target·只有 D:11 边→不在 PR matrix→add_seed no-op theater·doc §19.0 矛盾 A）。
    率挂 action_ref（B-PR2 _feed_action_experience 写·矛盾 B）·本 helper 内 D:11 桥接 word_ref→action_ref→read rate。

    **洗净 filter = structure_discover:1145 镜像**（conf 非 None 且 sn==0=验过皆败→滤除·cold-start None→给机会注入）：
    使 experience_count 读取非 theater（sn==0 真排除坏动作词·与 gate③ D:11 存在性正交：gate③=边存在·B-PR4=经验质量）。
    **rate-sort survivors = structure_discover:1154 stable sort 镜像**（dormant ordering·PR solve=Σx_s 交换律致纯排序
    行为惰性·defer future top-K cap·当前注入全部 survivors·doc §19.1 决断2 推翻设计审纯 C1 commutative-theater 修复）。

    **率读 observe_mode=False**（默认·e_sn/e_tn reward 驱动 success rate·即"动作验证率"·非 observe_tn 决策计数）。
    **一词映射多 ACTION_***（现实一词一类·边界场景）：取最高率·全 action sn==0 才整词滤除。
    **intent 守**（mirror B-PR2 _feed_action_experience:2654·caller 守 gate·helper 守 intent·单测可直验）：
    intent_type != INTENT_COMMAND → 返 []（QUESTION 不路由动作执行·不预算动作种子·§13.3）。
    返 list（可能空：QUESTION intent / 无动作词 / 全 sn==0 洗净 / 冷启动无 experience_count 行→全 None→全给机会注入率 0）。
    """
    if intent_type != INTENT_COMMAND:
        return []   # intent 守（mirror B-PR2·caller 守 gate·helper 守 intent·QUESTION 不预算）
    from pure_integer_ai.cognition.shared.action_primitives import lookup_word_action
    from pure_integer_ai.storage.node_store import TIER_PRIMARY
    from pure_integer_ai.storage.experience_count import read_experience_count
    # 率 ×1000 缩放（sn/(sn+tn)→rate·同 structure_discover _OP_CONF_RATE_SCALE 既有约定·pure int）
    _RATE_SCALE = 1000
    seen_words: set = set()
    scored: list[tuple[int, ConceptRef]] = []   # (rate, word_ref)·stable sort 率降序
    for seg in segments:
        for tok in seg.tokens:
            word_ref = concept_index.lookup(tok, space_id)
            if word_ref is None or word_ref in seen_words:
                continue   # 词未概念化 / 已收（distinct by word token）
            actions = lookup_word_action(backend, edge_store, word_ref,
                                         space_id=space_id, tier_filter=TIER_PRIMARY)
            if not actions:
                continue   # 非动作词（无 D:11 PRIMARY ACTION_* 边）
            seen_words.add(word_ref)
            best_rate = -1
            all_bad = True   # 全 action sn==0 → 整词洗净滤除
            for action_ref, _kind in actions:
                conf = read_experience_count(backend, action_ref, ctx_code=ctx_code)
                if conf is not None and conf[1] == 0:
                    continue   # 该 action tested-never-verified（sn==0）·不采纳·别的 action 可能好
                all_bad = False
                if conf is None:
                    rate = 0   # 冷启动·给机会（排末·mirror structure_discover:1148）
                else:
                    rate = conf[1] * _RATE_SCALE // max(conf[1] + conf[2], 1)
                if rate > best_rate:
                    best_rate = rate
            if all_bad:
                continue   # 洗净滤除（ACTIVE 率消费者·非 theater）
            scored.append((best_rate, word_ref))
    # stable sort 率降序·同率保 segments 遍历序（bit-identical·mirror structure_discover:1154 reverse=True 稳定）
    scored.sort(key=lambda x: x[0], reverse=True)
    return [word_ref for _rate, word_ref in scored]


# ---- 阶段4 promote ----

def _promote_eligible(ctx: TrainContext, teacher: Any) -> tuple[int, int]:
    """阶段4 promote 三重（SHADOW→PRIMARY·频次/reward/定义·§十三防塌C4）。

    扫 SHADOW 边 → promote_report 判 eligible → promote_edge flip tier。
    返 (promote_count, oov_promote_count)。oov_promote = 概念点新晋 PRIMARY 计数。
    """
    from pure_integer_ai.storage.node_store import TIER_SHADOW
    promote_count = 0
    oov_promote = 0
    promoted_concepts: set[ConceptRef] = set()
    rows = ctx.backend.select("edge", where={"tier": TIER_SHADOW})
    for row in rows:
        ref = (row["space_id_from"], row["local_id_from"],
               row["space_id_to"], row["local_id_to"], row["edge_type"])
        rep = promote_report(ctx.edge_store, ref, teacher=teacher, backend=ctx.backend)
        if not rep["eligible"]:
            continue
        if promote_edge(ctx.edge_store, ctx.node_store, ref, teacher=teacher,
                        backend=ctx.backend):
            promote_count += 1
            for c in ((ref[0], ref[1]), (ref[2], ref[3])):
                if c not in promoted_concepts:
                    promoted_concepts.add(c)
                    oov_promote += 1
    return promote_count, oov_promote


def _split_holdout(corpus: list[CollectedItem], probe_holdout: int,
                   ) -> tuple[list[CollectedItem], list[CollectedItem]]:
    """W4 D4·确定性切分 corpus → (training, probe)。probe_holdout=0 → (corpus, []) bit-identical。

    取末尾 N 个作 probe（纯整序·确定性·bit-identical）。守 ≥1 训练样本（min 钳位）。
    依赖 corpus 序确定性（caller 保证序有意义·load_arith_corpus params="bcdefghjklmn" 确定序）。
    诚实：探针采样质量是 oracle 责任（末尾 N 非语义择优·真 corpus 须 caller 保证序有意义）。
    """
    if probe_holdout <= 0 or len(corpus) < 2:
        return list(corpus), []
    n = min(probe_holdout, len(corpus) - 1)
    return list(corpus[:-n]), list(corpus[-n:])


def _item_sig(item: CollectedItem) -> str:
    """W4 D4·CollectedItem → 确定性签名 str（probe ref 用·非 observe struct_ref）。

    算术域 arith_source / 代码域 code_source / 语言域 tokens。全空 raise（fail-loud·反静默泄漏检测失败）。
    同 item→同签名→同 ref（泄漏检测正确）·异 item→异签名→异 ref（不相交正确·D 墙：异 item 未必真无关）。
    """
    if item.arith_source:
        return item.arith_source
    if item.code_source:
        return item.code_source
    if item.tokens:
        return "\x00".join(item.tokens)
    raise ValueError(
        "_item_sig: CollectedItem 全空（arith_source/code_source/tokens 皆无）·无法判 probe 隔离")


def _stage_items(corpus: list[CollectedItem], stage: int,
                 rounds: int) -> list[CollectedItem]:
    """该阶段喂的 items（首版全语料·按阶段门控防缺防超喂·真按阶段配比 §十二 defer）。"""
    if not corpus:
        return []
    return list(corpus)


# ---- 序列6-min 生产触发器（de-theater 序列1·§八.6·算子自动发现） ----

# discovery 独立根的 Hasher 种子（固定·跨 run bit-identical·绕 observe 多程序撞 __seg_{stage}_0）
_DISC_ROOT_SEED = "formal_train.disc_src"
_DISC_LANG_SEED = "formal_train.disc_lang"   # 语言发现独立根种子（S3·tokens 内容哈希·绕 observe __seg_ 碰撞 + episode 结构独立）
_DISC_LANG_ALIGN_SEED = "formal_train.disc_lang_align"   # 件4 变长对齐独立根种子（consensus 对齐序列内容哈希）
_DISC_LANG_SENSE_SEED = "formal_train.disc_lang_sense"   # 刀6 片4：sense clone aligning_root 种子（原 root + sense ref 哈希·逐 sense 试骨架对齐）


def _discover_and_recognize_arith_operators(
        ctx: TrainContext,
        corpus: list[CollectedItem],
        *,
        existing_operators: Sequence[DiscoveredOperator] = (),
        ) -> tuple[list[DiscoveredOperator], list[Recognition], GeneralizationSummary]:
    """序列6-min 发现（WRITE）+ 序列3-min 识别（READ）+ 验证半闭环（vm_proof 验泛化）生产触发器：
    算术语料 → 内容哈希独立根 → per-shape 留 held-out → auto_discover_operators + recognize_operators
    + _verify_generalization。

    **序列6-min（§八.6·de-theater 序列1·WRITE）**：discover_skeleton 从"仅 tests 调"升"真生产 caller"。
    绕两个 observe 既有限制（非本步引入·doc §八.6 诚实边界②）：① observe 多 arith item 撞同一 __seg_{stage}_0
    struct_ref；② EdgeStore.add 纯 append-only 不去重。故按 **arith_source 内容哈希**建独立根 __disc_src_{h63}
    （同程序同根·不同程序不同根·跨 run 幂等）·已建有 COMPOSES 出边的根 skip（防重 build 复制边）。

    **序列3-min（§八.3·生产期 READ 消费·"新样本命中已学骨架"）**：per-shape 留 held-out——同形 ≥
    MIN_DISCOVER_SAMPLES+1（≥3）→ 发现首 K 个·识别余（held-out 新实例）；同形 ∈ [K, K+1)（==2）→ 全发现·无
    held-out；同形 <K → 不发现。识别 held-out 新输入（**非发现样本集→真泛化·非循环 theater**：骨架从 {5,6}
    学·识别 {7,8} 新输入·设计 line193 "新样本命中已学骨架" 本意）。
    **序列6-min 进化**：序列6-min 在全语料发现·序列3-min 进化为 per-shape 留 held-out（识别须新输入·全语料
    发现+识别同集=循环 theater·反 §8.7）。2 样本语料（同形==2）行为不变（全发现·序列6-min 既有测仍过）。

    **验证半闭环（§8.7·反 theater + 学到能力证据）**：识别产物 recognitions 不再 terminal——_verify_generalization
    对每个识别做 caller 级 vm_proof（execute_composes_value）：骨架绑识别 params 执行 == held-out 新输入执行值。
    识别=结构对齐·vm_proof=VM 执行比对·两路独立计算——**诚实**：对正确识别同值是构造性预期（结构同构·非惊奇交叉验证）·
    vm_proof 真"牙"是抓获 PARAM 阅读序错位/编译发散/shape 漏判异配（probe SUB 错参 -47≠43 不 verified）·重执行本身即真
    READ+应用消费（非 theater·非死写）。verified/total_held_out = 泛化率（学到的能力覆盖多少 held-out 新输入·直接量化
    "学到能力"·解"不能从语料学到能力"根因侧证）。

    返 (discovered, recognitions, generalization)（formal_train 写 FormalTrainResult.discovered_operators/
    recognitions/generalization·可观测 + 反 theater 锚点）。生产路径：build_composes_from_arith（真 builder）→
    auto_discover_operators（group + discover_skeleton + register·WRITE）→ recognize_operators（held-out 读骨架
    抽 PARAM 绑定·READ）→ _verify_generalization（vm_proof 验骨架绑参复现 held-out 新输入值·消费 recognitions）。
    """
    from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
    from pure_integer_ai.storage.edge_store import SOURCE_MATH
    from pure_integer_ai.storage.node_store import NODE_CONCEPT

    arith_items = [it for it in corpus
                   if it.modality == MODALITY_ARITH and it.arith_source]
    if not arith_items:
        return [], [], GeneralizationSummary()   # 无算术语料 → 无发现/识别/泛化（空汇总·诚实）
    # 序列7 跨 run READ：existing_operators = 已载发现算子（**caller 传**·formal_train resume load_run 后经
    # load_discovered_operators 取·非本函数内查 backend）→ 避免同一 backend 内两次调把首调注册算子误当"载入"
    # （会循环识别重喂发现集=theater）。默认 () → 直调单测（不传）bit-identical 旧为（无跨 run 识别）。
    existing_ops = list(existing_operators)
    # Half B（§八.7②·Finding1 真修）：路由键 (sig, arity) **非 sig-only** —— square(arity1) 载入后 mul(arity2)
    # 同形异 arity 仍须独立发现（原 sig-only 路由把同形全送识别→mul 被 square 拒/坍缩→静默丢）。existing_keys =
    # 载入算子 (sig,arity,abstract_sig)·existing_sigs = fallback（<K 或 probe None 时按 sig 认载入→识别候选）。
    # **B6 Bug 2 修（2026-07-06·聚类前置）**：existing_keys 加 abstract_sig 维——从 op.skeleton_ref 经
    # _collect_slot_lcas 重建（同 LOAD 端 Bug 1 修法）+ _normalize_abstract_sig 归一。解 resume 时同 (sig,arity)
    # 异 abstract_sig 新样本误判"已载"静默丢→新抽象类本轮不发现。arith abstract_sig 恒 ()（无 CONCEPT_LEAF）→
    # 载入算子 abstract_sig 亦 () → (sig,arity,()) 与原 (sig,arity) 等价·bit-identical。
    existing_keys: set[tuple[tuple[int, ...], int, tuple, tuple]] = set()
    existing_sigs: set[tuple[int, ...]] = set()
    for op in existing_ops:
        op_sig = tuple(shape_signature(ctx.concept_graph, op.skeleton_ref))
        op_asig = _normalize_abstract_sig(_collect_slot_lcas(
            ctx.backend, ctx.concept_graph, op.skeleton_ref))
        op_cue = _normalize_abstract_sig(_collect_cue_sig(
            ctx.backend, ctx.concept_graph, op.skeleton_ref))   # §十八 condition 6a-3：cue_sig 第4维（镜像 abstract_sig·gate OFF 全 None→()→bit-identical·gate ON 是/使 异键独立路由）
        existing_keys.add((op_sig, op.arity, op_asig, op_cue))
        existing_sigs.add(op_sig)
    # 内容哈希独立根（同程序同根幂等·不同程序不同根·绕 observe struct_ref 碰撞 + EdgeStore.add 不去重）
    roots: list[ConceptRef] = []
    for item in arith_items:
        h = Hasher(_DISC_ROOT_SEED).h63(item.arith_source)
        root = ctx.concept_index.ensure(
            f"__disc_src_{h}", space_id=ctx.space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 幂等：已建有 COMPOSES 出边 → skip（EdgeStore.add 不去重·防重 build 复制边 corrupt 树）
        if not ctx.edge_store.query_from(root[0], root[1], edge_type=EDGE_COMPOSES):
            build_composes_from_arith(
                item.arith_source, concept_index=ctx.concept_index,
                edge_store=ctx.edge_store, backend=ctx.backend,
                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=root)
        roots.append(root)
    # 路由（聚类前置·B6 Bug 2+3·2026-07-06）：按 (sig,hint) 分组 → 每组 LCA 聚类 → 按簇 abstract_sig 路由
    # discover/recognize（解 existing_keys 缺 abstract_sig 致跨 run 覆盖渐失 + cluster-blind held-out 致混合簇不发现）。
    # arith 首 sample 无 CONCEPT_LEAF → 聚类单簇 None → abstract_sig=() → 路由键 (sig,arity,()) 与原 (sig,arity)
    # 等价·bit-identical。helper 详 structure_discover.route_samples_for_discovery。
    discover_roots, recognize_roots = route_samples_for_discovery(
        ctx.backend, ctx.concept_graph, roots,
        existing_keys=existing_keys, existing_sigs=existing_sigs,
        space_id=ctx.space_id)
    discovered = auto_discover_operators(
        discover_roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=ctx.space_id, source=SOURCE_MATH)
    # 序列7：all_ops = 载入（跨 run）+ 本 run 新发现。识别/验证两路皆对 all_ops（载入算子可识别新输入=跨 run 泛化）。
    all_ops: list[DiscoveredOperator] = list(existing_ops) + discovered
    if not all_ops or not recognize_roots:
        # 无已知算子 / 无 held-out → 无可识别/泛化（诚实·不伪造）。返 discovered（本 run 新发现仍上报）。
        return discovered, [], GeneralizationSummary()
    recognitions = recognize_operators(
        recognize_roots, discovered_operators=all_ops,
        backend=ctx.backend, space_id=ctx.space_id)
    # 验证半闭环（§8.7）：vm_proof 独立验每个识别（骨架绑参复现 held-out 新输入值）→ 泛化汇总。
    # 识别产物 recognitions 在此被真消费（解 terminal 边界·反 theater）。total_held_out=len(recognize_roots)。
    generalization = _verify_generalization(
        ctx, recognitions, all_ops, total_held_out=len(recognize_roots))
    return discovered, recognitions, generalization


def _discover_and_recognize_lang_structures(
        ctx: TrainContext,
        corpus: list[CollectedItem],
        *,
        existing_operators: Sequence[DiscoveredOperator] = (),
        ) -> tuple[list[DiscoveredOperator], list[Recognition], GeneralizationSummary]:
    """钥匙①语言结构发现·分片第二片（S3·doc/重来_钥匙①语言结构发现机制设计_修正分析七.md）：
    语言语料 → 内容哈希独立根 __disc_lang_{h63(tokens)} → 建语言 COMPOSES 序（NOP SEQ root + token 叶·
    **caller 建·件1 落点修正·非 observe·反 theater**）→ per-(shape,hint) 留 held-out → auto_discover_operators
    + recognize_operators（concept_binding·件5 _align_walk 语言分支）。

    **件1 落点修正（反 theater·2026-07-05·S3 第二片）**：件1 原在 observe 建 COMPOSES·经对抗审复查发现零
    生产消费（dag_path/attractor/judge/vm_proof 不读 EDGE_COMPOSES·发现用独立根 __disc_src_）→ theater。
    改 caller 建独立根 __disc_lang_{h63(tokens)}（同 _run_arith __disc_src_ 范式·绕 observe __seg_ 碰撞 +
    episode 结构独立）·发现真消费。observe 不建→A6 不冲突（主线 §8.8 撤销 A6 推翻决断·A6 维持原禁令）。

    **vm_proof 跳过（钥匙③墙·诚实）**：语言骨架 NOP+PARAM 不可 VM 执行（PARAM 绑 token concept_ref 非
    Rational·compile_graph 编译 LOAD operand 但 vm 无 token 值）·_verify_generalization 不调·verified=0
    （语言泛化率不可 vm_proof 量化·钥匙③相0 vm_proof 降级对偶 defer·§七诚实边界）。识别 concept_binding
    （件5 _align_walk 语言分支）= 语言识别产物·**刀2 多解析**：同 input_root 可命中多骨架（类级抽象 + 词例级
    loose·都返 Recognition）·recognized 计 **distinct input_root**（防双计·非 len(recognitions)）。

    返 (discovered, recognitions, GeneralizationSummary(total_held_out, recognized, verified=0))。
    生产路径：内容哈希独立根（caller 建 COMPOSES）→ auto_discover_operators（group+discover_skeleton+
    register·WRITE）→ recognize_operators（held-out 读骨架抽 concept_binding·READ）。
    """
    from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_OPERATOR
    from pure_integer_ai.numeric.symbol_domain import OPCODE_NOP
    from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, EPI_STRUCTURED
    from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
    from pure_integer_ai.storage.node_store import NODE_CONCEPT
    from pure_integer_ai.crosscut.determinism.hasher import Hasher

    lang_items = [it for it in corpus
                  if it.modality == MODALITY_LANGUAGE and it.tokens]
    if not lang_items:
        return [], [], GeneralizationSummary()
    existing_ops = list(existing_operators)
    # Half B (sig,arity) 路由（同 _run_arith·载入算子预过滤防循环识别）+ B6 Bug 2+3 修（聚类前置·2026-07-06）。
    # **原 S3 第二刀 Interp2 已知边界（resume 渐失覆盖）已解**：existing_keys 加 abstract_sig 维（从 op.skeleton_ref
    # 经 _collect_slot_lcas 重建·同 LOAD 端 Bug 1 修法）+ 路由改调 route_samples_for_discovery（聚类前置·per-cluster
    # held-out）。同 (sig,arity) 异 abstract_sig 新样本（动物类载入后非生物类新输入）按簇独立路由→新抽象类本轮发现·
    # 不再全送 recognize 静默丢。fresh run（existing 空）零影响。auto_discover 幂等门（name 含 abstract_sig）+ 路由
    # abstract_sig 维双重守·bit-identical 不破（裸 NL abstract_sig 恒 () → (sig,arity,()) 与原 (sig,arity) 等价）。
    existing_keys: set[tuple[tuple[int, ...], int, tuple, tuple]] = set()
    existing_sigs: set[tuple[int, ...]] = set()
    for op in existing_ops:
        op_sig = tuple(shape_signature(ctx.concept_graph, op.skeleton_ref))
        op_asig = _normalize_abstract_sig(_collect_slot_lcas(
            ctx.backend, ctx.concept_graph, op.skeleton_ref))
        op_cue = _normalize_abstract_sig(_collect_cue_sig(
            ctx.backend, ctx.concept_graph, op.skeleton_ref))   # §十八 condition 6a-3：cue_sig 第4维（镜像 abstract_sig·gate OFF 全 None→()→bit-identical·gate ON 是/使 异键独立路由）
        existing_keys.add((op_sig, op.arity, op_asig, op_cue))
        existing_sigs.add(op_sig)
    # 内容哈希独立根 + 建语言 COMPOSES 序（caller 建·件1 落点修正·反 theater·绕 observe __seg_ 碰撞）
    roots: list[ConceptRef] = []
    root_keys: list[tuple[int, int]] = []   # scope B：每 root 的 (id(item), seg_idx)·observe (item_key,seg_idx) 对齐
    root_expected: dict[ConceptRef, ConceptRef | None] = {}   # S7 相0：root → item.expected_skeleton（教师标定比对）
    # 刀6 片4：root → 各 token 位 (ti, tok_text, sense_candidates, leaf_ref)·供 recognize_roots clone 逐 sense 试。
    # sense_candidates = read_sense_candidates base_count>0（boot 种先验）·leaf_ref = 该位 token 叶 ConceptRef
    # （gate ON + 有 sense → 首 sense ref NodeRef 升序首·有 IS_A 上卷·discover slot_lca 真火；否则 ensure(tok) 原路径）。
    root_token_entries: dict[ConceptRef, list[tuple[int, str, list[tuple[int, int]], tuple[int, int]]]] = {}
    _sense_gate_on = bool(getattr(gates, "SENSE_LOOKUP_MODE", False))
    from pure_integer_ai.storage.sense_candidates import read_sense_candidates, sense_surface_hash
    # scope B（断奶 critical path ④·doc/重来_语料聚簇规模_2026-07-17）：gate COMPOSES_COMBINE_MODE ON→按句切建根
    # （_sentence_bounds·与 observe _split_item_to_segments 同源切法·seg_idx 逐一对齐）·OFF→整段单 span=原段级根行为
    # （bit-identical）。observe 本就按句切段（多段 struct_ref·unit 早已句级）→ 切句**不增 observe 成本**·只让
    # discovery 从段级根改句级根 → 同长句聚簇 → 骨架+cue槽涌现（解整段签名塌词数→零骨架根因）。
    _scope_b_split = bool(getattr(gates, "COMPOSES_COMBINE_MODE", False))
    _flat_units: list[tuple[CollectedItem, int, list[str]]] = []   # (item, seg_idx, sentence_tokens)
    for item in lang_items:
        _spans = _sentence_bounds(item.tokens) if _scope_b_split else [(0, len(item.tokens))]
        for seg_idx, (_s, _e) in enumerate(_spans):
            _toks = list(item.tokens[_s:_e])
            if _toks:
                _flat_units.append((item, seg_idx, _toks))
    for item, seg_idx, tokens in _flat_units:
        h = Hasher(_DISC_LANG_SEED).h63("\x1f".join(tokens))
        root = ctx.concept_index.ensure(
            f"__disc_lang_{h}", space_id=ctx.space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 先填 entries（不论是否已建 COMPOSES·clone 段须用·fresh + resume 统一）
        entries: list[tuple[int, str, list[tuple[int, int]], tuple[int, int]]] = []
        for ti, tok in enumerate(tokens):
            sense_cands: list[tuple[int, int]] = []
            if _sense_gate_on:
                _sh = sense_surface_hash(tok)
                sense_cands = [sr for sr, base, _sn, _tn
                               in read_sense_candidates(ctx.backend, ctx.space_id, _sh) if base > 0]
            # gate ON + 有 sense → 首 sense ref（NodeRef 升序首·boot ensure 的·有 IS_A·discover slot_lca 上卷）。
            # gate OFF 或无 sense → ensure(tok)（**原路径 bit-identical**·退化链 5 步·plan 决断 5）。
            tok_ref = sense_cands[0] if sense_cands else ctx.concept_index.ensure(
                tok, space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
            entries.append((ti, tok, sense_cands, tok_ref))
        root_token_entries[root] = entries
        # 幂等：已建有 COMPOSES 出边 → skip（EdgeStore.add 不去重·防重 build 复制边 corrupt）
        if not ctx.edge_store.query_from(root[0], root[1], edge_type=EDGE_COMPOSES):
            # 建语言 COMPOSES 序：NOP SEQ root（OPCODE_NOP）+ token 叶（边 to 端 concept_ref·不挂 attr·
            # 件2 _is_concept_leaf 无属性叶判定）。镜像 code_observe:86 SEQ NOP + observe 件1 原设计·但 caller 建。
            record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
            for ti, tok, _cands, tok_ref in entries:
                ctx.edge_store.add(
                    space_id_from=root[0], local_id_from=root[1],
                    space_id_to=tok_ref[0], local_id_to=tok_ref[1],
                    edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
                    epistemic_origin=EPI_STRUCTURED, order_index=ti)
        root_expected[root] = item.expected_skeleton
        roots.append(root)
        root_keys.append((id(item), seg_idx))
    # 件4 变长 LCS 对齐（doc/重来_钥匙①语言结构发现机制设计_修正分析七.md §三件4）：
    # 变长 roots（length set 多值·shape 各异 → 分散各组 <K → 原路径永不发现变长结构）→ pairwise_fold
    # consensus 锚位 → 等长对齐根（破同子数门 structure_discover.py:339）·替代变长原 roots。
    # **同长（length set 单一）→ roots 不变走原路径（bit-identical·既有同长语料零改）**。
    # **退化（consensus 空/未全匹配/<K）→ roots 不变（变长不发现·诚实不纸面闭合·非 theater）**。
    # 反 theater：wrapper 生产 caller（_run_lang 接入）+ e2e（变长语料真发现骨架·test_stage12 件4）。
    # 段 slot=段首 token concept_ref（cross-sample 异词同槽=PARAM 泛化牙·件2 D2 弱化门）·空段占位。
    root_token_lens = [len(ctx.concept_graph.read_composes_tree(r)[0].get(r, []))
                       for r in roots]
    if len(set(root_token_lens)) > 1:
        from pure_integer_ai.cognition.process.lang_structure_align import (
            align_variable_lang_sequences)
        aligned_seqs = align_variable_lang_sequences(
            ctx.concept_graph, roots,
            concept_index=ctx.concept_index, space_id=ctx.space_id)
        if aligned_seqs is not None:
            # 建对齐独立根（同 __disc_lang_ 范式·NOP SEQ + 对齐 token 叶）·替代变长原 roots
            new_roots: list[ConceptRef] = []
            new_expected: dict[ConceptRef, ConceptRef | None] = {}
            for seq, orig_root in zip(aligned_seqs, roots):
                h = Hasher(_DISC_LANG_ALIGN_SEED).h63(
                    "\x1f".join(f"{r[0]}:{r[1]}" for r in seq))
                align_root = ctx.concept_index.ensure(
                    f"__disc_lang_align_{h}", space_id=ctx.space_id,
                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                if not ctx.edge_store.query_from(
                        align_root[0], align_root[1], edge_type=EDGE_COMPOSES):
                    record_composes_attr(ctx.backend, ref=align_root,
                                         kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
                    for ti, ref in enumerate(seq):
                        ctx.edge_store.add(
                            space_id_from=align_root[0], local_id_from=align_root[1],
                            space_id_to=ref[0], local_id_to=ref[1],
                            edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
                            epistemic_origin=EPI_STRUCTURED, order_index=ti)
                new_roots.append(align_root)
                new_expected[align_root] = root_expected.get(orig_root)
            roots = new_roots
            root_expected = new_expected
        # else: consensus 退化·roots 不变（变长不发现·诚实）
    # 路由（聚类前置·B6 Bug 2+3·2026-07-06·同 _run_arith）：按 (sig,hint) 分组 → LCA 聚类 → 按簇 abstract_sig
    # 路由 discover/recognize。解 existing_keys 缺 abstract_sig（跨 run 覆盖渐失）+ cluster-blind held-out（混合簇
    # 前 K 横跨簇致每簇 <K 不发现）。裸 NL（has_isa=False）单簇 None → abstract_sig=() → bit-identical。helper 详
    # structure_discover.route_samples_for_discovery。
    discover_roots, recognize_roots = route_samples_for_discovery(
        ctx.backend, ctx.concept_graph, roots,
        existing_keys=existing_keys, existing_sigs=existing_sigs,
        space_id=ctx.space_id)
    discovered = auto_discover_operators(
        discover_roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=ctx.space_id, source=SOURCE_BARE_TEXT)
    # Phase D §十六-bis D.1：REALIZES labeled bed（option-b oracle-pair-match·skeleton→__REL_SUBSET__）。
    # skeleton 通用文本独立发现 + forming-sample token-pair 命中外源 EDGE_IS_A → REALIZES（oracle 定 IS_A 非读 Cue）。
    # **gate REALIZES_MODE default OFF→整块跳过→bit-identical**（含 ensure_relation_primitives 调用·守 CI 零副作用）。
    # labeled bed（同 boot IS_A）·学习 claim 严禁前置·验 floor Phase F·consumer Phase E·D.1 ship ≠ Phase D done。
    if getattr(gates, "REALIZES_MODE", False) and discovered:
        from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
        from pure_integer_ai.cognition.process.structure_discover import label_realizes_is_a, label_realizes_causes
        _rel_prims = ensure_relation_primitives(
            ctx.concept_index, ctx.backend, space_id=ctx.space_id)   # 幂等·确保 __REL_SUBSET__/__REL_CAUSES__ 存在
        label_realizes_is_a(discovered, graph=ctx.concept_graph, edge_store=ctx.edge_store,
                            rel_primitives=_rel_prims, space_id=ctx.space_id)
        # Phase D §十六-bis D.1 CAUSES labeled bed（镜像 IS_A·oracle=外源 ConceptNet CAUSES·REALIZES→__REL_CAUSES__）。
        # condition-6 配套：使-skeleton 走外源 CAUSES oracle 标（非 使 cue·anti-self-proving·避循环）。
        label_realizes_causes(discovered, graph=ctx.concept_graph, edge_store=ctx.edge_store,
                              rel_primitives=_rel_prims, space_id=ctx.space_id)
    # 维度桥 item→skeleton map（P1 G-PR2·COMPOSES_COMBINE_MODE ON·shape_signature 匹配 root→discovered skeleton·
    # 存 ctx.work_memory.lang_skeleton_by_item[(id(item), seg_idx)]（scope B 句级键·observe 读建 EDGE_INSTANTIATES 边·§十三-bis A.1）。gate OFF 不建→bit-identical。
    # 诚实边界（审2 LOW-1）：shape_signature 对语言坍缩为长度（NOP+leaves·忽略 abstract_sig/LCA 子簇）·非"结构精确"。
    # setdefault first-wins·多 LCA 簇同长可能误绑·精确 LCA 匹配 defer 断桥/相1。dormant（gate 不在生产 flip 列表）误绑无后果。
    if getattr(gates, "COMPOSES_COMBINE_MODE", False) and discovered:
        # ★ Bug C2 修法（维度桥同步 floor S1·2审 APPROVE-WITH-CONDITIONS）：
        # 旧双重缺陷：(1) _skel_by_sig 单值 shape 键（坍缩误绑·同 floor S1 病根）；
        # (2) 仅搜 discovered（this-call）漏 existing_ops（跨 run resume 时 held-out 训练 root 无法绑历史 skeleton）。
        # 新 _dim_all_ops = existing_ops + discovered（与 floor S1 all_ops 同源·mirror recognize:3984）+
        # _ops_by_sig dict[shape,list]（C3）+ per-root _aligns_to_skeleton first-match。两处键语义一致（共享 helper）。
        from pure_integer_ai.cognition.process.structure_discover import (
            shape_signature as _dim_shape_sig, _aligns_to_skeleton)
        _dim_all_ops: list[DiscoveredOperator] = list(existing_ops) + list(discovered)
        _ops_by_sig: dict[tuple[int, ...], list[ConceptRef]] = {}
        for _op in _dim_all_ops:
            _ops_by_sig.setdefault(
                tuple(_dim_shape_sig(ctx.concept_graph, _op.skeleton_ref)), []).append(_op.skeleton_ref)
        for _key, _root in zip(root_keys, roots):
            _sig = tuple(_dim_shape_sig(ctx.concept_graph, _root))
            for _skel in _ops_by_sig.get(_sig, []):
                if _aligns_to_skeleton(ctx.backend, ctx.concept_graph, _root, _skel):
                    ctx.work_memory.lang_skeleton_by_item[_key] = _skel
                    break   # first-match（C5 tiebreak = _dim_all_ops 顺序·确定性）
    all_ops: list[DiscoveredOperator] = list(existing_ops) + discovered
    # 刀6 片4：recognize_roots clone aligning_root 逐 sense 试（首个多 sense token 位·每 sense 一 clone）。
    # clone root = __disc_lang_sense_{原root hash + sense ref}（确定性·幂等 ensure）·建 COMPOSES（原 root 各位叶·
    # 该位换 sense ref·其余同首 sense·守 within-sample 同一性 _align_walk 变量同一性）。
    # recognize_operators 喂 [原 root + clones]·逐个试骨架对齐（_align_walk ATTR_SLOT_ROLE IS_A 共祖选优·
    # 动物老鼠 sense 命中动物类骨架·鼠标 sense 不命中·反 theater 牙·#479 墙·结构选优非语义消歧·stable≠correct）。
    # origin 映射：clone root → 原 root（distinct 防双计·recognized 计 distinct origin·守 lang_rate_permille≤1000）。
    # **bit-identical**：gate OFF 或无 sense → sense_cands=[] → 无 clone → sense_recognize_inputs=[(rr,rr)...]=原 recognize_roots → 等同现状零行为变。
    sense_recognize_inputs: list[tuple[ConceptRef, ConceptRef]] = []  # (aligning_root, origin_root)
    for _rr in recognize_roots:
        sense_recognize_inputs.append((_rr, _rr))   # 原 root（首 sense·可能命中）
        _entries = root_token_entries.get(_rr, [])
        # 找首个多 sense token 位（首版简化·单 token 处理·多 token 多 sense 笛卡尔积 defer）
        for _ti, _tok, _cands, _leaf in _entries:
            if len(_cands) > 1:
                # 该位 N sense·首 sense 已是原 root 叶·clone 其余 N-1 sense（每 sense 一 aligning_root）
                for _sense_ref in _cands[1:]:
                    _clone_h = Hasher(_DISC_LANG_SENSE_SEED).h63(
                        f"{_rr[0]}:{_rr[1]}:{_sense_ref[0]}:{_sense_ref[1]}")
                    _clone_root = ctx.concept_index.ensure(
                        f"__disc_lang_sense_{_clone_h}", space_id=ctx.space_id,
                        tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                    if not ctx.edge_store.query_from(_clone_root[0], _clone_root[1], edge_type=EDGE_COMPOSES):
                        record_composes_attr(ctx.backend, ref=_clone_root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
                        for _ti2, _tok2, _cands2, _leaf2 in _entries:
                            # 该位换 _sense_ref·其余用原 leaf（首 sense·同骨架 IS_A 上卷·within-sample 同一性）
                            _clone_leaf = _sense_ref if _ti2 == _ti else _leaf2
                            if _clone_leaf is None:
                                continue   # 防御（resume leaf 缺·不应发生·entries 统一填）
                            ctx.edge_store.add(
                                space_id_from=_clone_root[0], local_id_from=_clone_root[1],
                                space_id_to=_clone_leaf[0], local_id_to=_clone_leaf[1],
                                edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
                                epistemic_origin=EPI_STRUCTURED, order_index=_ti2)
                    sense_recognize_inputs.append((_clone_root, _rr))
                break   # 首版只处理首个多 sense token 位（多 token 笛卡尔积 defer）
    _aligning_roots = [_ar for _ar, _orig in sense_recognize_inputs]
    if not all_ops or not _aligning_roots:
        return discovered, [], GeneralizationSummary()
    recognitions = recognize_operators(
        _aligning_roots, discovered_operators=all_ops,
        backend=ctx.backend, space_id=ctx.space_id)
    # 对应泛化 v2：结构反推 tally（审1C3/审2条件1+2·三路分离 + SHADOW 创建）。新词 W 落 REALIZES-R-skeleton
    # cue slot（cue-blind tally·独立于 recognize 精确匹配轨）→ tally (W,R) distinct forming-sample → 首次建 D:11
    # SHADOW（generator 关后唯一创建者）→ promote W→R D:11 PRIMARY（_structure_match_ok 唯一证据轨）。
    # **gate ORACLE_PROMOTE_MODE**（default OFF·bit-identical·OFF 不调→零 tally→零 D:11 翻→逐字现状）。
    # REALIZES_MODE + CUE_CLUSTER_MODE 须同 ON（REALIZES-skeleton 存在 + ATTR_CUE_SIG 落盘·caller 生产 try/finally 共翻）。
    # **非循环**：R 来自 REALIZES oracle（source==CONCEPTNET·非 cue）·W 观察·反馈在 source filter 断（§四）。
    if getattr(gates, "ORACLE_PROMOTE_MODE", False) and all_ops and _aligning_roots:
        from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
        from pure_integer_ai.cognition.process.structure_discover import tally_cue_slot_matches
        _rel_prims = ensure_relation_primitives(
            ctx.concept_index, ctx.backend, space_id=ctx.space_id)   # 幂等·tally rel_ref→rel_kind 反查用
        tally_cue_slot_matches(_aligning_roots, discovered_operators=all_ops,
                               graph=ctx.concept_graph, edge_store=ctx.edge_store,
                               backend=ctx.backend, space_id=ctx.space_id,
                               rel_primitives=_rel_prims)
    # S7 相0 钥匙③ vm_proof 降级对偶（教师标定比对·断奶前教师路径·POST 退场·镜像 vm_proof :376）：
    # recognize 命中骨架 ref == item.expected_skeleton → verified（op_confidence sn++·语言算子 name_ref 自然分桶）。
    # concept_binding（S3 第二片）已是结构对齐比对器·无须新写。断奶后教师退场（POST）·相0 不调（防 vacuous 命中 theater）。
    # **教师天花板诚实边界**：expected_skeleton 教师主观标定·非闭式真理——(a) 可能多解（同 input 命中多骨架·取教师标）
    # (b) 可能错标（标错→sn=0 退场·机制检出但首次 run 污染 tn·拉低 rate·反 theater 择优降权）
    # (c) 无第二独立源（vm_proof Mode B 才有·断奶后须相2/E1 接力·钥匙③墙≡#479）。相0 = 结构身份匹配·非真 vm_proof 等价。
    _origin_of = dict(sense_recognize_inputs)   # aligning_root → origin_root（clone distinct 防双计）
    expected_verified = 0
    if ctx.weaning_phase == WEANING_PRE:
        from pure_integer_ai.storage.op_confidence import record_op_outcome
        op_by_name = {op.name: op for op in all_ops}
        # 刀6 片4 对抗审 P1-1：expected_verified + record_op_outcome 须按 (origin, op) distinct
        # 防双计（同 origin 的原 root + clone root 都命中同 op → 不重复 sn++/tn++/expected_verified++）。
        # 与 recognized distinct origin 同理（plan 决断 5）·守 op_confidence 半环单调正确。
        _verified_origin_ops: set[tuple[ConceptRef, str]] = set()
        for rec in recognitions:
            op = op_by_name.get(rec.operator_name)
            _origin = _origin_of.get(rec.input_root, rec.input_root)
            _key = (_origin, rec.operator_name)
            if _key in _verified_origin_ops:
                continue   # 同 (origin, op) 已计·skip（防 clone 双计 sn/tn/expected_verified）
            # clone root 的 expected 取原 root 的 expected（origin 映射·clone 同源 input）
            expected = root_expected.get(_origin)
            if op is not None and expected is not None:
                hit = (op.skeleton_ref == expected)
                record_op_outcome(ctx.backend, ref=op.name_ref, verified=hit)
                _verified_origin_ops.add(_key)
                if hit:
                    expected_verified += 1
    # vm_proof 跳过（语言骨架不可 VM·钥匙③墙·诚实 verified=0·非偷懒·相0 vm_proof 降级对偶 = expected_verified）
    # 刀2 件6 防双计 + 刀6 片4 clone distinct：recognized 计 distinct origin（clone root → 原 root·非 clone ref）·
    # 守 lang_rate_permille≤1000。total_held_out=原 recognize_roots（clone 是 sense 候选扩展·不增 held-out 计数）。
    return discovered, recognitions, GeneralizationSummary(
        total_held_out=len(recognize_roots),
        recognized=len({_origin_of.get(rec.input_root, rec.input_root) for rec in recognitions}),
        verified=0, expected_verified=expected_verified)


def _verify_generalization(ctx: TrainContext, recognitions: list[Recognition],
                           discovered_operators: list[DiscoveredOperator],
                           *, total_held_out: int) -> GeneralizationSummary:
    """序列3-min 验证半闭环：识别 → caller 级 vm_proof 独立验泛化（骨架绑参 == held-out 输入值）→ 汇总。

    **反 theater + 学到能力证据**：发现骨架从发现集学到·识别 held-out 新输入·vm_proof 独立**重新执行**骨架
    （绑识别 params）确认复现新输入值。识别 = 结构对齐（recognize_operators._align_walk）·vm_proof = VM 执行
    比对（execute_composes_value）·两路独立计算。**诚实定位**（对抗审计）：对正确识别·骨架与输入结构同构（_align_walk
    固定位值等门保）→ VM 把 LOAD mv_i(绑值 v) 与 PUSH_IMM v 等同 → 同值是**构造性预期**（非惊奇交叉验证）。
    vm_proof 真"牙"=抓获 PARAM 阅读序错位 / skeleton 编译发散 / shape 漏判结构异配（probe：SUB 错参→-47≠43
    不 verified）→ 非恒真 stub。重执行本身 = 真 READ+应用消费（非 theater·非死写）。verified/total_held_out =
    泛化率（学到的能力覆盖多少 held-out 新输入·量化"学到能力"·解"不能从语料学到能力"根因侧证）。

    caller 级 vm_proof（L8 调 L7 execute_composes_value·守 recognize_operators L5 不调 L7 单向依赖）。
    **operand-input 识别**（rec.is_operand_input=True）：input 含 OPERAND 叶·按 operand_binding 反演 input 探针
    （input_probe[in_slot]=param_values[skel_slot]）执行比对——探针纯从 Recognition 字段反演无须 import 常量。
    诚实定位同 immediate：探针比对构造性（结构同构→同值）·真牙在 recognize_operators 变量同一性判定·探针=重执行消费。
    诚实：vm_proof None（非 COMPOSES 根/StepLimit）→ 不计 verified（不伪造·保留 recognized 计数）。
    """
    assert_int(total_held_out, _where="_verify_generalization.total_held_out")
    graph = ctx.concept_graph
    op_by_name = {op.name: op for op in discovered_operators}
    # 刀2 件6 防双计（doc §5 刀2 点3·caller 责任）：summary recognized/verified 计 **distinct input_root**
    # （同 root 多解析都验算 1 root·非每 rec +1）·守 rate_permille≤1000（recognized≤total_held_out）。
    # **op_confidence 半环 per-op 不双计**（循环体 record_op_outcome 不变·每 rec → distinct op name_ref 各 +sn/+tn）。
    verified_roots: set[ConceptRef] = set()
    for rec in recognitions:
        op = op_by_name.get(rec.operator_name)
        if op is None:
            continue   # 识别指向未知算子（理论不该发生·防御跳过·不计 verified·不写置信度）
        v_skel = execute_composes_value(graph, op.skeleton_ref, rec.param_values)
        if rec.is_operand_input:
            # operand-input 识别（探针值执行比对）：input 含 OPERAND 叶·用 rec.input_probe_values 直接执行 input
            # （连续 slot-序探针·_align_extract 派生·含未用 slot·消除反演洞·对抗审计 F1）·无须 import 探针常量。
            v_input = execute_composes_value(graph, rec.input_root, rec.input_probe_values)
        else:
            v_input = execute_composes_value(graph, rec.input_root, ())   # immediate 输入=nullary（既有路径）
        eq = (v_skel is not None and v_input is not None
              and rational.eq(v_skel, v_input))
        if eq:
            verified_roots.add(rec.input_root)   # distinct root（刀2 防双计·非 verified+=1）·泛化得证（两路独立·反 theater）
        # §8.7-洗 洗净循环反馈半闭环：vm_proof 验结果写算子置信度（op_confidence sn/tn/strength）→
        # recognize_operators 择优读（滤非泛化算子=洗净）·解 recognitions terminal·反 theater 半环。
        # verified→sn++&tn++&strength+=1 / not-verified（mismatch/deadloop None）→tn++ only（R1 符号·sn 单调）。
        # **刀2 多解析**：同 input_root 多 rec（异 op）→ 每 op 各记 op_confidence（per-op·非双计·distinct name_ref）。
        if op.name_ref != (0, 0):
            record_op_outcome(ctx.backend, ref=op.name_ref, verified=eq)
    return GeneralizationSummary(
        total_held_out=total_held_out,
        recognized=len({rec.input_root for rec in recognitions}),   # 刀2 distinct（非 len·防双计）
        verified=len(verified_roots))   # 刀2 distinct verified roots（非 verified 计数）


# ---- §8.7-全 生成侧全环·task-driven L8 episode（外真半·补半环缺·墙内现可达）----

def _run_task_driven_generate(ctx: TrainContext, corpus: list[CollectedItem],
                              all_ops: Sequence[DiscoveredOperator]
                              ) -> tuple[list[Episode], GenerateSummary]:
    """§8.7-全 生成侧全环·task-driven L8 episode：任务(input_args,expected) → 选算子 → 执行骨架
    → 外真验 vs expected → 写 op_confidence → 打包 OutputResult（6 步·不碰 generate.py L6·守单向 L8→L7/L0 向下）。

    **为何非 theater（审计降级后·Mode A 构造性·非"外真非传递"）**：半环（§8.7-洗 done）测自洽
    skeleton(recognized_params)==input()（学生==学生·传递必然对正确算子）·本函数测**新 args 泛化探针 +
    生成侧函数应用姿态**——skeleton(新 task args)==expected（生成姿态 call 算子为函数产答案·vs 半环识别姿态
    align 程序·args 非学习输入→产新值非记忆复现）。**Mode A 构造性**：skeleton 派生自输入程序·故 expected=正确答案时
    skeleton(args)==expected 构造性必然（传递经 skeleton 起源·同半环牙·无新牙）·**非"外真非传递"**。
    真非传递外真 = Mode B（异算法·断奶后 defer·§8.7-全 前置/范围）。task-driven 真增量 = 新 args 探针 + 生成姿态 +
    多候选 wash（①④真）·非伪需求·但认识论声称须诚实（Mode A 构造性·非外真）。

    **6 步机制**（§8.7-全）：
      1. task 输入 = CodeSpec(input_args, expected)（教师 Mode A 独立源·断奶前·vm_proof.py:91 R6）。
         任务来源 = arith item 的 arith_specs（同 item specs·设计决策点 (a)·不建输入程序选算子执行）。
      2. 选算子：候选 = arity==len(input_args) 的已发现算子·读 read_op_confidence → rate=sn*1000//max(tn,1)
         → 滤 tested-never-verified(sn==0·同 recognize_operators:730-732) → 稳定排序择优（同率保 BFS 序·bit-identical）。
      3. 执行：execute_composes_value(graph, op.skeleton_ref, input_args) → v_skel（L8→L7 OK·复用 :1108 既有）。
      4. 验（Mode A 构造性）：rational.eq(v_skel, make(*expected)) → verified（外比 expected·Mode A 构造性必然对正确算子）。
         None→not verified（诚实）。
      5. 写置信度：record_op_outcome(ref=op.name_ref, verified=eq)（L8→L0 OK·复用 :1159 既有·与 recognize 半环同表累积）。
      6. 打包 OutputResult（**审计必修·parts 非空⟺verified**·未验→parts=[]）→ Episode.output → metrics
         record_generate_round 读 e.output.parts 计数 generate_verified（真消费·非死写）。

    **反 theater 4 锚点**（审计后·①④ PASS·②③ 降级/修）：①行为真变（多候选读置信度择优·置 sn=0→选他者产出变）②
    新消费（skeleton(新 args) vs expected·新 args 非学习输入·**Mode A 构造性·真非传递留 Mode B**）③下游读者
    （OutputResult.parts→metrics generate_verified 计数真读·**metrics 读 parts 非 reward·parts 非空⟺verified·
    审计必修·否则=(Z) theater 重引入**）④拒坏选好（不 fit→tn++·fit→sn++·择优选 fit）。

    **防双计**（诚实边界④·审计 F3·**caller 责任·机制不强制**）：task_items 取所有有 arith_specs 的 arith item·
    若 held-out 程序项同携匹配 specs（input_args==程序 args）则同值同试验 sn/tn 双计·**但 rate=sn/tn 同比 inflate→
    rate 不变→选择不变·仅绝对计数虚增**。机制不强制 task 输入≠held-out（强制须跳过 recognize_roots 项的 specs·
    独立 follow-up）·caller 须确保 task specs 用 args ≠ held-out 程序。

    铁律：纯整数（input_args/expected/v_skel num/den 全 int·assert_int·Rational 经 make）/ bit-identical
    （rate 排序稳定 tiebreak BFS 序·同 recognize）/ 单向依赖（L8→L7 execute/L0 op_confidence 向下·不碰 L6 generate.py）/
    MUTABLE_MONOTONE（record_op_outcome 同表同键 UPDATE·R1 符号 sn 单调）/ 不写死（选算子=arity 结构匹配·rate 比较非硬编码）/
    核心无墙钟 / 不走外挂 LLM（断奶前教师 Mode A）/ 危险词禁 / IS_A 不涉。
    诚实边界：①单候选构造性必过（仅 mul 时 skeleton(args)==expected 构造性必过·信号薄·同半环）·多候选信号实；
    ②mul/square 不可分（同 task 两都 verified·置信度正交于变量同一性·Half B arity 区分）；③stable≠correct；
    ④防双计 caller 责任（见上·机制不强制·rate 不受影响）；⑤闭环=生成侧 task-driven 探针（Mode A）+ 半环（自洽）=
    生成侧 wash 环（对本模态·Mode A）·**真外真全环须 Mode B**·generate.py 字面须 STRUCT_BIND(VF)·独立后续。
    """
    assert isinstance(all_ops, Sequence), "_run_task_driven_generate.all_ops 须 Sequence[DiscoveredOperator]"
    # 候选按 arity 索引（len(input_args)==op.arity·结构匹配·非语义规则）
    by_arity: dict[int, list[DiscoveredOperator]] = {}
    for op in all_ops:
        by_arity.setdefault(op.arity, []).append(op)
    graph = ctx.concept_graph
    total_tasks = 0
    selected = 0
    verified = 0
    # S5-S8 symbolic 子计数（#1124·additive·镜像 GenerateSummary·0=CI 无 symbolic specs bit-identical）：
    # transform 规则 verified 数（S5-S7 SYMBOLIC_TRANSFORM 块）+ inverse 关系 verified 数（S8 SYMBOLIC_RELATION 块）。
    # 用于 capability_exam.project_symbolic_measures → CapabilityReport.math_measures（反 theater：symbolic 学习可见）。
    xform_verified = 0
    inv_verified = 0
    # #1124 M-1：symbolic 分母（spec §四 4.2 要求 rate permille·镜像 lang_rate_permille）。
    # xform_total=transform 规则总数（含 malformed/cross-verify 失败·分母）·inv_total=inverse 关系总数。
    xform_total = 0
    inv_total = 0
    episodes: list[Episode] = []
    # 任务来源 = arith item 的 arith_specs（同 item specs·设计决策点 (a)）·非 arith_source（不建输入程序）
    task_items = [it for it in corpus
                  if it.modality == MODALITY_ARITH and it.arith_specs]
    ep_id = 0
    for item in task_items:
        for spec in item.arith_specs:
            total_tasks += 1
            n_args = len(spec.input_args)   # CodeSpec.__post_init__ 已 assert_int 守 input_args/expected
            candidates = by_arity.get(n_args, [])
            # 洗净：滤 tested-never-verified (sn==0)·同 recognize_operators:730-732·cold-start(None)给机会不滤
            viable: list[tuple[int, DiscoveredOperator]] = []
            for op in candidates:
                conf = read_op_confidence(ctx.backend, op.name_ref)   # (sn,tn,strength)|None·纯读 L8→L0
                if conf is not None and conf[0] == 0:
                    continue   # tested-never-verified（sn==0=验过皆败）滤除·cold-start(None)给机会
                rate = ((conf[0] * _OP_CONF_RATE_SCALE // max(conf[1], 1))
                        if conf is not None else 0)   # cold-start→0（给机会但排序末位·verified 率高优先）
                viable.append((rate, op))
            if not viable:
                continue   # 无 arity 匹配候选 / 全滤 → 诚实跳过（不伪造·不计 selected/verified）
            # 稳定排序·同率保候选 BFS 序（=发现序·bit-identical）·reverse=True 不破稳定性（equal 保输入序）
            viable.sort(key=lambda x: x[0], reverse=True)
            op = viable[0][1]
            selected += 1
            # 执行骨架（L8→L7 execute_composes_value·复用 _verify_generalization:1108 既有调用·input_args int→(arg,1) Rational）
            v_skel = execute_composes_value(
                graph, op.skeleton_ref, tuple((a, 1) for a in spec.input_args))
            # 验（Mode A 构造性·expected=正确答案时 skeleton(args)==expected 构造性必然·同半环牙·
            # 真非传递外真留 Mode B·§8.7-全 决断标注）。rational.eq 外比（expected=教师 Mode A 独立源）。
            eq = (v_skel is not None
                  and rational.eq(v_skel, rational.make(spec.expected[0], spec.expected[1])))
            if eq:
                verified += 1
            # 写置信度（L8→L0·复用 _verify_generalization:1159 既有·与 recognize 半环同表累积·R1 符号 sn 单调）
            if op.name_ref != (0, 0):
                record_op_outcome(ctx.backend, ref=op.name_ref, verified=eq)
            # 打包 OutputResult（**反 theater ③下游读者锚·审计必修**：parts 非空 ⟺ verified·
            # 未验过/编译发散→parts=[] 不提交产出·metrics record_generate_round 读 e.output.parts 计数
            # generate_verified（真消费·非死写·否则=§8.7-洗-证伪 candidate (Z) theater 重引入）。
            # verified→parts=[算子名 ref + 产出值]·未验→parts=[]（产出值未提交·失败计数在 op_confidence tn）。
            if eq and v_skel is not None:
                output = OutputResult(parts=[OutputPart(
                    unit=op.name_ref, words=[f"{v_skel.num}/{v_skel.den}"])])
            else:
                output = OutputResult(parts=[])   # 未验过/None→不提交产出（诚实·metrics 读 parts 计数）
            ep = Episode(
                episode_id=ep_id, run_id=ep_id,
                input=None, output=output,
                reward=1 if eq else 0,
                ref=op.name_ref,
                terminal=TERMINAL_REACHED_SINK,
                pr_vector={},   # task-driven 不跑 dag_path_step·无 PR 向量（诚实）
                judge_G5_active=False,   # task-driven 不经 judge G5（外真验 = execute vs expected·非 G5 门）
                judge_veto_count=0 if eq else 1,
                dead_end_count=0,
                vetoed=(not eq),
                verify_source=VERIFY_SOURCE_EXTERNAL,   # Layer0：execute vs spec.expected R6 外部源（同 _run_verify_round·Mode A 构造性验证·可计 external_verified·2 审 P1-1 修）
            )
            episodes.append(ep)
            ep_id += 1
    # 断桥 Phase A（P2 G-PR2/3 cross-path·ACTION_BRIDGE_MODE ON·doc/重来_断桥设计refinement_2026-07-15）：
    # CollectedItem.action_specs（教师标 I/O 例·数据驱动**非硬编码**）→ synthesize_value **联合匹配**全 specs（PbE·
    # 一动作多 I/O 例共定一骨架·反 per-spec 碎·审2 F4）→ 独立 task-driven episode。断桥 cross-path：language/action
    # item 经 action_specs 跨路径喂 arith 骨架池合成（**spec→synthesis**·intent 分类=Phase B 动态构造器·Phase A 教师
    # 标 specs 已含 intent 语义·审2 F1/F2/F3 修回：design 原 dict[action_ref]+classify_intent 移 Phase B）。
    # **weaning-safe 决断 A**：独立 episode·**不替换 vm_proof verify round·不碎 W7**（反 VALUE_SYNTHESIZE 翻 ON 教训）。
    # gate OFF 或无 action_specs → 不进 → bit-identical。无匹配骨架 → 诚实 continue（同 arith no-viable :3512-3513）。
    # **不写 op_confidence**（断桥 cross-path 独立度量轴·teacher Mode A specs 构造性·reinforce 会 inflate·Phase B
    # held-out 真泛化后接·同 code-unparse :3590 范式）·故 unit/ref=skeleton_ref（合成骨架·非 arith name_ref·无
    # op_confidence 消费者·Phase B 接 op_confidence 时统一 name_ref·审1 MEDIUM-2/3）。selected==verified（synthesis
    # 返匹配皆已验·match=verified 内禀·非 arith 两段式·审1 LOW-3）。pool=load_discovered_operators（同 verify round
    # :653·persisted 发现骨架升序·all_ops 含本 run 非 persisted 不用·断桥/verify 匹配 persisted 发现骨架·审2 L2）。
    # Phase B（动态 intent→spec 构造器 + dispatch 桥 CHANNEL_*→VM/serializer/judge）defer。
    if getattr(gates, "ACTION_BRIDGE_MODE", False):
        _bridge_items = [it for it in corpus if it.action_specs]   # action_specs 断桥专属（language/action·arith 用 arith_specs·无 overlap·审1 LOW-1）
        if _bridge_items:   # 无 action_specs → 不 load pool（避空载·审1 LOW-4）
            from pure_integer_ai.training.value_synthesize import synthesize_value
            _bridge_pool = load_discovered_operators(ctx.backend, space_id=ctx.space_id)
            for _bitem in _bridge_items:
                total_tasks += 1   # 一 item = 一动作（联合 specs 共定·非 per-spec·审2 F4）
                _bmatches = synthesize_value(graph, _bridge_pool, tuple(_bitem.action_specs))
                if not _bmatches:
                    continue   # 无行为匹配骨架 → 诚实跳过（不伪造 episode·同 arith no-viable）
                _bsynth_root, _bbinding = _bmatches[0]   # 首匹配（pool 升序·bit-identical·多匹配 defer Phase B 排序·同相1 :656）
                selected += 1
                verified += 1   # match=verified 内禀（synthesize_value 已 execute+eq·审2 H4）
                # 实际产出值（re-execute spec[0]·非 expected·守 parts=actual 不变量·同 arith :3519/3536·审1 MEDIUM-1）：
                # match 保证 actual==spec[0].expected·re-execute 守未来 synthesize bug 不静默（_binding_param_values:67 镜像）。
                _bfirst = _bitem.action_specs[0]
                _bparam_vals = tuple((_bfirst.input_args[_bbinding[i]], 1)
                                     for i in range(len(_bbinding)))
                _bactual = execute_composes_value(graph, _bsynth_root, _bparam_vals)
                _boutput = OutputResult(parts=[OutputPart(
                    unit=_bsynth_root, words=[f"{_bactual.num}/{_bactual.den}"])])
                episodes.append(Episode(
                    episode_id=ep_id, run_id=ep_id,
                    input=None, output=_boutput,
                    reward=1, ref=_bsynth_root,
                    terminal=TERMINAL_REACHED_SINK,
                    pr_vector={}, judge_G5_active=False,
                    judge_veto_count=0, dead_end_count=0, vetoed=False,
                    verify_source=VERIFY_SOURCE_EXTERNAL,   # action_spec expected = 教师 Mode A 外部源（同 arith :3550）
                ))
                ep_id += 1
    # 断桥 Phase B 片1（P2 动态构造器·ACTION_BRIDGE_CUE_MODE ON·doc/重来_断桥设计refinement_2026-07-15 §Phase B 片1）：
    # 无教师 action_specs 时·从 language text cues 动态构造 spec：CollectedItem.numeric_claims_flat（刀B observe 期
    # flatten·4-tuple `(left,op,right,result)`）→ CodeSpec 隐 op（input_args=(left,right)·expected=(result,1)·**op 隐藏**
    # =synthesize 找算子非刀B 验算子·真合成）→ synthesize_value 联合匹配（同算子多 claim 共定一骨架·PbE·混算子无匹配
    # 诚实 skip）→ 独立 task-driven episode。input source = text cues（非 teacher·非 held-out·非硬编码·解 Phase B
    # "撞 held-out"之谜：held-out 仅泛化验证须·构造+合成 NOW）。
    # **刀B 无冲突**：刀B reward round 验证轴（用 op·:406 路由）·断桥 generate stage 合成轴（隐 op）·两 stage 分离·
    # 同 numeric item 得两 episode 不同轴（同 Phase A 与他处理共存范式）。断桥合成 ≠ 刀B 验证（合成产骨架学算子）。
    # **weaning-safe 决断 A**：独立 task-driven episode·不替换 vm_proof verify round·不碎 W7（同 Phase A）。
    # gate OFF 或无 numeric_claims_flat → 不进 → bit-identical。无匹配骨架 → 诚实 continue（同 Phase A :3576）。
    # **不写 op_confidence**（同 Phase A·独立度量轴·Phase B held-out 真泛化后接）。
    if getattr(gates, "ACTION_BRIDGE_CUE_MODE", False):
        _cue_items = [it for it in corpus if it.numeric_claims_flat]
        if _cue_items:   # 无 numeric_claims_flat → 不 load pool（避空载·同 Phase A :3569）
            from pure_integer_ai.training.value_synthesize import synthesize_value
            _cue_pool = load_discovered_operators(ctx.backend, space_id=ctx.space_id)
            for _cueitem in _cue_items:
                # cue→spec：numeric_claim (left,op,right,result) → CodeSpec((left,right),(result,1))·op 隐藏（synthesize 找算子）
                _cue_specs = tuple(
                    CodeSpec(input_args=(c[0], c[2]), expected=(c[3], 1))
                    for c in _cueitem.numeric_claims_flat)
                total_tasks += 1   # 一 item = 一合成任务（联合 specs 共定·同 Phase A·审2 F4 范式）
                _cuematches = synthesize_value(graph, _cue_pool, _cue_specs)
                if not _cuematches:
                    continue   # 无行为匹配骨架 → 诚实跳过（同 Phase A·不伪造 episode）
                _csynth_root, _cbinding = _cuematches[0]   # 首匹配（pool 升序·bit-identical·多匹配排序 defer 片2）
                selected += 1
                verified += 1   # match=verified 内禀（synthesize_value 已 execute+eq·同 Phase A·审2 H4）
                # 实际产出值（re-execute spec[0]·非 expected·守 parts=actual 不变量·同 Phase A·审1 MEDIUM-1）：
                _cfirst = _cue_specs[0]
                _cparam_vals = tuple((_cfirst.input_args[_cbinding[i]], 1)
                                     for i in range(len(_cbinding)))
                _cactual = execute_composes_value(graph, _csynth_root, _cparam_vals)
                _coutput = OutputResult(parts=[OutputPart(
                    unit=_csynth_root, words=[f"{_cactual.num}/{_cactual.den}"])])
                episodes.append(Episode(
                    episode_id=ep_id, run_id=ep_id,
                    input=None, output=_coutput,
                    reward=1, ref=_csynth_root,
                    terminal=TERMINAL_REACHED_SINK,
                    pr_vector={}, judge_G5_active=False,
                    judge_veto_count=0, dead_end_count=0, vetoed=False,
                    verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # 审2 LOW-1 修：cue-derived spec.expected 来自 text cues
                    # （numeric_claims_flat·single-source·**非 R6 外部源**·同刀B SELF_PRODUCED :893·非 Phase A action_spec
                    # 教师标 R6 外部）→ SELF_PRODUCED 守"全自产不准停"（layer0_anchor.py:72 EXTERNAL 才计 external_verified
                    # 驱动停止决策·text-derived synthesis 不准驱动停止·反 theater）。
                ))
                ep_id += 1
    # 符号数学扩展 Phase 3（SYMBOLIC_TRANSFORM_MODE ON·doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis.7）：
    # CollectedItem.transform_specs（教师陈述符号变换规则·数据驱动**非硬编码**·humans 学法：从教师/课本学规则
    # 陈述+验证+应用+关联）→ register_transform_rule + apply held-out input + cross-verify 执行 output==expected
    # → 独立 task-driven episode。**weaning-safe 决断 A**：独立 task-driven episode·不替换 vm_proof verify round·
    # 不碎 W7（同断桥 Phase A/B）。**verify_source=SELF_PRODUCED**：规则应用+cross-verify single-source 自产自验
    # （非 R6 外部源·规则+held-out 来自 corpus 但验证是自产执行比对）·守"全自产不准停"（同断桥 Phase B 片1·反 theater）。
    # gate OFF 或无 transform_specs → 不进 → bit-identical。LHS 不匹配/cross-verify 失败 → 诚实 continue（不伪造 episode）。
    # **诚实边界**：cross-verify 单点采样（小素数探针 per arity·多采样点 refinement defer）·stable≠correct（#479 守）。
    if getattr(gates, "SYMBOLIC_TRANSFORM_MODE", False):
        _xform_items = [it for it in corpus if it.transform_specs]
        if _xform_items:
            import ast as _ast
            from pure_integer_ai.training.symbolic_transform import register_transform_rule, apply_transform
            from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith, UnsupportedConstruct
            from pure_integer_ai.storage.edge_store import SOURCE_MATH
            from pure_integer_ai.storage.node_store import NODE_CONCEPT
            _XFORM_PROBES = (2, 3, 5, 7, 11, 13)   # 小素数探针（cross-verify 采样点·per-slot·arity≤6）
            for _xitem in _xform_items:
                for _spec in _xitem.transform_specs:
                    total_tasks += 1   # 一规则 = 一学习任务（register + held-out 验）
                    xform_total += 1   # #1124 M-1 symbolic transform 分母（含失败/malformed）
                    # try/except 守（对抗审 Finding 1·mirror code_unparse :3753-3759）：malformed spec
                    # （build/apply/execute raise UnsupportedConstruct/ValueError/KeyError·如 Pow(x,0)→n-1=-1 负指数
                    # / DSL 解析错 / arity 不匹配）→ 诚实 skip 此 spec（不 abort 整个 run·守"诚实 continue"意图）。
                    try:
                        # build + register 规则（lhs/rhs lambda DSL → COMPOSES·教师陈述模板·非硬编码）
                        _lhs_ref = ctx.concept_index.ensure(
                            f"__xform_lhs_{_spec.rule_name}", space_id=ctx.space_id,
                            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                        build_composes_from_arith(_spec.lhs_source, concept_index=ctx.concept_index,
                            edge_store=ctx.edge_store, backend=ctx.backend,
                            space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_lhs_ref)
                        _rhs_ref = ctx.concept_index.ensure(
                            f"__xform_rhs_{_spec.rule_name}", space_id=ctx.space_id,
                            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                        build_composes_from_arith(_spec.rhs_source, concept_index=ctx.concept_index,
                            edge_store=ctx.edge_store, backend=ctx.backend,
                            space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_rhs_ref)
                        _rule_ref = register_transform_rule(
                            ctx.backend, ctx.concept_index, _spec.rule_name,
                            _lhs_ref, _rhs_ref, space_id=ctx.space_id)
                        # held-out cross-verify（apply 规则到 input·执行 output==expected·统计验规则应用正确）
                        _all_pass = bool(_spec.held_out)   # 无 held-out → 不验→不产 episode（反 theater）
                        _last_words = ["1/1"]
                        for _ho in _spec.held_out:
                            _in_ref = ctx.concept_index.ensure(
                                f"__xform_in_{_spec.rule_name}_{_ho.input_source}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_ho.input_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_in_ref)
                            _exp_ref = ctx.concept_index.ensure(
                                f"__xform_exp_{_spec.rule_name}_{_ho.expected_source}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_ho.expected_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_exp_ref)
                            _out_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                                space_id=ctx.space_id, source=SOURCE_MATH,
                                rule_name_ref=_rule_ref, input_ref=_in_ref)
                            if _out_ref is None:
                                _all_pass = False; break   # LHS 不匹配 → 诚实 skip（非 theater）
                            # cross-verify：arity 探针·执行 output==expected（stable≠correct·统计验非 truth）
                            _arity = len(_ast.parse(_ho.expected_source).body[0].value.args.args)
                            _probes = tuple((_XFORM_PROBES[i], 1) for i in range(_arity))
                            _vo = execute_composes_value(graph, _out_ref, _probes)
                            _ve = execute_composes_value(graph, _exp_ref, _probes)
                            if _vo is None or _ve is None or not rational.eq(_vo, _ve):
                                _all_pass = False; break   # cross-verify 失败 → 诚实 skip（反 theater·不伪造 verified）
                            _last_words = [f"{_vo.num}/{_vo.den}"]
                        if not _all_pass:
                            continue
                        selected += 1
                        verified += 1
                        xform_verified += 1   # #1124 symbolic transform 子计数（S5-S7）
                        episodes.append(Episode(
                            episode_id=ep_id, run_id=ep_id,
                            input=None, output=OutputResult(parts=[OutputPart(unit=_rule_ref, words=_last_words)]),
                            reward=1, ref=_rule_ref, terminal=TERMINAL_REACHED_SINK,
                            pr_vector={}, judge_G5_active=False,
                            judge_veto_count=0, dead_end_count=0, vetoed=False,
                            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # single-source 自产自验·不准驱动停止·反 theater
                        ))
                        ep_id += 1
                    except (UnsupportedConstruct, ValueError, KeyError):
                        continue   # malformed spec（build/apply/cross-verify raise）→ 诚实 skip·不 abort run
    # S8 符号间运算关联 Phase 1（SYMBOLIC_RELATION_MODE ON·doc/重来_S8符号间关联机制设计_2026-07-15 §七）：
    # CollectedItem.inverse_relation_specs（教师陈述逆关系·两条独立变换规则 A↔B 互逆·数据驱动**非硬编码**·humans 学法：
    # 从教师/课本学"两规则互逆"+构造验证·非纯归纳发现 research-grade defer）→ register rule_a + register rule_b +
    # register_inverse_relation + verify_inverse_relation（B∘A 还原 @ 采样·三值 True/False/None）→ verified 则独立
    # task-driven episode。**weaning-safe 决断 A**（同 SYMBOLIC_TRANSFORM 块·独立 episode·不替换 vm_proof·不碎 W7）。
    # **verify_source=SELF_PRODUCED**：两规则 single-source 教师·逆验证 self-consistency（非 R6 两源·非 truth）·
    # 守"全自产不准停"（同 transform_specs·反 theater）。gate OFF 或无 inverse_relation_specs → 不进 → bit-identical。
    # can't-verify(None)/falsified(False) → 诚实 continue（不伪造 episode·反 theater ③下游读者锚 parts 非空⟺verified）。
    # **诚实边界**：逆验证=统计非证明（采样还原 ≠ 数学逆·#479 守）·可复合约束（B LHS 须匹配 A 输出 shape·否则 can't-verify）。
    if getattr(gates, "SYMBOLIC_RELATION_MODE", False):
        _rel_items = [it for it in corpus if it.inverse_relation_specs]
        if _rel_items:
            from pure_integer_ai.training.symbolic_transform import register_transform_rule, load_transform_rule as _load_rule
            from pure_integer_ai.training.symbolic_relation import (
                register_inverse_relation, verify_inverse_relation, RELATION_KIND_INVERSE)
            from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith, UnsupportedConstruct
            from pure_integer_ai.storage.edge_store import SOURCE_MATH
            from pure_integer_ai.storage.node_store import NODE_CONCEPT
            for _ritem in _rel_items:
                for _rspec in _ritem.inverse_relation_specs:
                    total_tasks += 1   # 一逆关系 = 一学习任务（register 两规则 + 逆验证）
                    inv_total += 1   # #1124 M-1 symbolic inverse 分母（含失败/malformed）
                    # try/except 守（同 SYMBOLIC_TRANSFORM 块·mirror code_unparse :3753-3759）：malformed spec
                    # （build/register/verify raise UnsupportedConstruct/ValueError/KeyError·如 DSL 解析错 / arity 不匹配 /
                    # Pow 负指数）→ 诚实 skip 此 spec（不 abort 整个 run）。
                    try:
                        # build + register rule_a / rule_b（两变换规则·逆关系须先存在·镜像 SYMBOLIC_TRANSFORM 块建规则）。
                        # surface 键用 **rule_name**（非 relation_name·对抗审 MEDIUM）：register_transform_rule 按裸 rule_name
                        # 去重·同 rule_name 跨关系须映射同 lhs/rhs ConceptRef→幂等。**重 build 守**（对抗审 MEDIUM 深因）：
                        # build_composes_from_arith 重 build 进已填充 root 会破坏树（duplicate children）→ 共享 rule_name 的
                        # 第二关系重 build 致 apply 失败→不 verified。故 build 前查规则已注册（load_transform_rule）→ 复用跳 build。
                        from pure_integer_ai.training.symbolic_transform import load_transform_rule as _load_rule
                        _rule_refs = {}
                        for _rspec_rule in (_rspec.rule_a, _rspec.rule_b):
                            _rname_ref = ctx.concept_index.ensure(
                                _rspec_rule.rule_name, space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            if _load_rule(ctx.backend, _rname_ref) is not None:
                                _rule_refs[_rspec_rule.rule_name] = _rname_ref   # 已注册·幂等复用·跳 build（防重 build 树破坏）
                                continue
                            _rlhs = ctx.concept_index.ensure(
                                f"__rel_lhs_{_rspec_rule.rule_name}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_rspec_rule.lhs_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_rlhs)
                            _rrhs = ctx.concept_index.ensure(
                                f"__rel_rhs_{_rspec_rule.rule_name}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_rspec_rule.rhs_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_rrhs)
                            _rule_refs[_rspec_rule.rule_name] = register_transform_rule(
                                ctx.backend, ctx.concept_index, _rspec_rule.rule_name,
                                _rlhs, _rrhs, space_id=ctx.space_id)
                        # register 逆关系（KIND=INVERSE·挂关系名 concept + RULE_A + RULE_B）
                        _rel_ref = register_inverse_relation(
                            ctx.backend, ctx.concept_index, space_id=ctx.space_id,
                            name=_rspec.relation_name, kind=RELATION_KIND_INVERSE,
                            rule_a_ref=_rule_refs[_rspec.rule_a.rule_name],
                            rule_b_ref=_rule_refs[_rspec.rule_b.rule_name])
                        # build sample inputs e（B∘A 须还原这些 @ 探针）
                        _sample_refs = []
                        for _ss in _rspec.sample_sources:
                            _se = ctx.concept_index.ensure(
                                f"__rel_sample_{_rspec.relation_name}_{_ss}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_ss, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_se)
                            _sample_refs.append(_se)
                        # 构造验证 B∘A 还原（三值·反 theater 心脏）
                        _verified = verify_inverse_relation(
                            ctx.backend, ctx.concept_index, ctx.edge_store,
                            space_id=ctx.space_id, source=SOURCE_MATH,
                            relation_ref=_rel_ref, sample_inputs=_sample_refs)
                        if _verified is not True:
                            continue   # can't-verify(None)/falsified(False) → 诚实 skip（反 theater·不伪造 verified episode）
                        selected += 1
                        verified += 1
                        inv_verified += 1   # #1124 symbolic inverse 子计数（S8）
                        episodes.append(Episode(
                            episode_id=ep_id, run_id=ep_id,
                            input=None,
                            output=OutputResult(parts=[OutputPart(unit=_rel_ref, words=["inv:verified"])]),
                            reward=1, ref=_rel_ref, terminal=TERMINAL_REACHED_SINK,
                            pr_vector={}, judge_G5_active=False,
                            judge_veto_count=0, dead_end_count=0, vetoed=False,
                            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # 两规则 single-source 自产自验·不准驱动停止·反 theater
                        ))
                        ep_id += 1
                    except (UnsupportedConstruct, ValueError, KeyError):
                        continue   # malformed spec（build/register/verify raise）→ 诚实 skip·不 abort run
    # #730 路径 W：代码模态 task-driven episode（unparse COMPOSES→源码串·Mode A 构造性·gate CODE_UNPARSE_MODE）。
    # 与 arith execute 对称但走 unparse（非 execute·L8→L5 向下·不调 L7）：读 item.code_struct_ref（observe 期
    # __prog_* 根·候选 A·run_round_full observe 后捕获）→ unparse_composes 序化 → ast bodies_match normalize
    # == code_source normalize → verified（构造性必然·skeleton 派生自 code_source·同 arith skeleton(args)==expected·
    # formal_train.py:1944 范式·stable≠correct·非真生成·真生成须路径 X 跨模态 defer）。
    # 无算子选择（用 item 自身 observe 树·非 discover skeleton 抽象 PARAM 占位·unparse 得 PARAM 词非源码）·
    # 不写 op_confidence（非 operator-level·信号在 Episode reward + OutputResult.parts·metrics generate_verified 读）。
    # **反 theater ③下游读者锚**：parts 非空 ⟺ verified（未验/树缺→parts=[] 不提交产出·同 arith :2066-2070）。
    if getattr(gates, "CODE_UNPARSE_MODE", False):
        from pure_integer_ai.cognition.result.composes_unparse import unparse_composes
        from pure_integer_ai.cognition.result.ast_normalize import bodies_match
        code_items = [it for it in corpus
                      if it.modality == MODALITY_CODE and it.code_source]
        for item in code_items:
            if item.code_struct_ref is None:
                continue   # observe 未建树（理论不发生·防御）·诚实 skip（不计 total/selected/verified）
            total_tasks += 1
            selected += 1   # 树已建 = 选定（无算子择优·直接用 item observe 树·selected 语义=有可 unparse 树）
            # per-item try/except（审2 P1-2·镜像 arith v_skel=None 容错·单 item 序化异常不崩整个 run）：
            # 序化器 raise LoopClosureDefect(病态深/环) / ValueError(STORE 须 1 子·非支持形态)→诚实降级 not verified。
            try:
                unparsed = unparse_composes(graph, item.code_struct_ref)
                eq = bodies_match(unparsed, item.code_source)   # AST normalize 结构等价（Mode A 构造性验证）
            except (ValueError, RuntimeError) as _unparse_err:
                # LoopClosureDefect(RuntimeError 子) / ValueError → 序化异常·诚实 not verified（parts 空·reward=0）
                unparsed = "<unparse-failed>"
                eq = False
            if eq:
                verified += 1
            # 打包 OutputResult（反 theater ③：parts 非空 ⟺ verified·未验→parts=[]·metrics 读 parts 计 generate_verified）
            if eq:
                output = OutputResult(parts=[OutputPart(
                    unit=item.code_struct_ref, words=[unparsed])])
            else:
                output = OutputResult(parts=[])
            ep = Episode(
                episode_id=ep_id, run_id=ep_id,
                input=None, output=output,
                reward=1 if eq else 0,
                ref=item.code_struct_ref,
                terminal=TERMINAL_REACHED_SINK,
                pr_vector={},   # 代码模态 task-driven 不跑 dag_path_step·无 PR 向量（诚实·同 arith）
                judge_G5_active=False,   # unparse 验证非 judge G5 门（构造性重建 vs source·非 G5 门因子）
                judge_veto_count=0 if eq else 1,
                dead_end_count=0,
                vetoed=(not eq),
                verify_source=VERIFY_SOURCE_EXTERNAL,   # Layer0：unparse vs code_source R6 外部源（同 _run_verify_round·Mode A 构造性重建·可计 external_verified·2 审 P1-1 修）
            )
            episodes.append(ep)
            ep_id += 1
    summary = GenerateSummary(
        total_tasks=total_tasks, selected=selected, verified=verified,
        xform_verified=xform_verified, inv_verified=inv_verified,
        xform_total=xform_total, inv_total=inv_total)   # #1124 symbolic 子计数 + 分母（M-1）
    return episodes, summary


# ---- E8 cursor state 持久化（续训 stage-skip 用） ----

def _cursor_path(run_dir: str, run_id: str) -> str:
    return os.path.join(run_dir, run_id, "cursor.json")


def _save_cursor(state: CursorState, run_dir: str, run_id: str) -> str:
    """持久化 cursor state（已完成/非 skippable 阶段集·供下次续训 E8 stage-skip）。"""
    import json as _json
    path = _cursor_path(run_dir, run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "base_run_id": state.base_run_id,
        "run_id": state.run_id,
        "completed": sorted(state.completed),
        "non_skippable": sorted(state.non_skippable),
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(_json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return path


def _load_cursor(run_dir: str, run_id: str) -> CursorState | None:
    """载入 base run 的 cursor state（续训 E8 stage-skip 用·无文件返 None）。"""
    import json as _json
    path = _cursor_path(run_dir, run_id)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)
    return CursorState(
        base_run_id=payload.get("base_run_id", run_id),
        run_id=payload.get("run_id", run_id),
        completed=set(payload.get("completed", [])),
        non_skippable=set(payload.get("non_skippable", [])),
    )
