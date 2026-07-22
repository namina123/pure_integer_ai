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

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, Sequence, TYPE_CHECKING, runtime_checkable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage.backend import StorageBackend, DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import (
    EdgeStore, SOURCE_CONCEPTNET, SOURCE_CHINESE_KB, SUBTYPE_OCCURRENCE,
)
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
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
from pure_integer_ai.storage.sense_candidates import (
    SENSE_CANDIDATES_TABLE,
    SENSE_LEGACY_BRIDGE_TABLE,
)
from pure_integer_ai.storage.abstract_mark import ABSTRACT_MARK_TABLE
from pure_integer_ai.storage.word_form_index import (
    WORD_FORM_INDEX_TABLE,
    WORD_FORM_LEGACY_BRIDGE_TABLE,
    register_word_form_index,
)
from pure_integer_ai.cognition.understanding.emergent_role import POSITION_HIST_TABLE
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_CORE, SpaceRegistry
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
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope as source_document_scope,
    episode_scope,
    make_scope,
    session_scope,
    SCOPE_DOCUMENT,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.observe import observe
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.experiments.lang_structure_metrics import (
    LanguageStructureStateStats, measure_language_structure_state,
)
from pure_integer_ai.experiments.train_execution import (
    FormalTrainExecutionStats,
    ExecutionPhaseRecorder,
    TelemetryClock,
    item_candidate_total,
    run_with_execution_telemetry,
    save_execution_metrics,
)
from pure_integer_ai.experiments.train_gate_profile import (
    push_production_training_gates,
    reset_production_training_gates,
)
from pure_integer_ai.experiments.train_scope import resolve_train_scope
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    _item_document_identity,
    _item_observation_identity,
    _item_occurrence_scope,
    make_train_context,
)
from pure_integer_ai.experiments.language_observation import (
    _apply_word_form_providers,
    _item_sentence_bounds,
    _item_token_source_spans,
    _materialize_item_spans,
    _prepare_item_boundaries,
    _prepare_item_boundary,
    _split_item_to_segments,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.language_candidate_runtime import (
    install_language_candidate_runtime,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.language_generation_h2 import (
    run_typed_language_h2,
)
from pure_integer_ai.experiments.language_generation_floor import (
    run_typed_language_floor,
)
from pure_integer_ai.experiments.train_result_types import (
    GeneralizationSummary,
)
from pure_integer_ai.experiments.language_structure_runtime import (
    _discover_and_recognize_lang_structures,
)
from pure_integer_ai.experiments.verification_dispatch import (
    is_verify_modality as _is_verify_modality,
)
from pure_integer_ai.experiments.round_runtime import (
    COVERAGE_THRESHOLD,
    DefaultRoundRunner,
    RoundResult,
    RoundRunner,
    _build_space_ctx,
    _collect_action_seed_candidates,
    _feed_action_experience,
    _hotzone_dag_edges,
    _reachable_dag_edges,
    _rebuild_path,
    _resolve_emergent_excluded_refs,
    _run_emergence_hook,
    _run_round_batch,
)
from pure_integer_ai.experiments.task_generation_runtime import (
    GenerateSummary,
    _run_task_driven_generate,
)
from pure_integer_ai.experiments.arithmetic_structure_runtime import (
    _discover_and_recognize_arith_operators,
    _verify_generalization,
)
from pure_integer_ai.experiments.evaluation_runtime import (
    H2_CALIB_BATCH,
    CalibrationSample,
    _held_out_discovery_tally_free,
    _h2_calibrate,
    _h2_calibrate_impl,
    _make_calib_judge_fn,
    _measure_floor_pass,
    _measure_floor_pass_impl,
    _observe_eval_item,
    _run_calibration_phase,
    _run_calibration_phase_impl,
    _run_simulated_offline_eval,
    _run_simulated_offline_eval_impl,
    run_evaluation_plan,
)
from pure_integer_ai.experiments.preflight_runtime import (
    PRE_FLIGHT_MEM_BUDGET_PER_ROUND,
    PRE_FLIGHT_ROUNDS,
    PreFlightReport,
    _pre_flight_impl,
    pre_flight,
)
from pure_integer_ai.experiments.stage_learning_runtime import (
    _inject_base_freq,
    _promote_eligible,
)
from pure_integer_ai.experiments.train_diagnostics import (
    _anti_collapse_summary,
    _causes_coverage,
    _edge_count,
    _graph_size,
    _weaning_blockers,
)
from pure_integer_ai.experiments.corpus_identity import (
    assign_corpus_source_refs,
    ensure_item_scope,
)
from pure_integer_ai.experiments.chinese_semantic_kb_curriculum import (
    SPLIT_TRAIN,
)
from pure_integer_ai.cognition.process.episode import episode_loop, _ctx_tag
from pure_integer_ai.cognition.process.structure_discover import (
    auto_discover_operators, recognize_operators, shape_signature,
    load_discovered_operators, probe_arity,
    route_samples_for_discovery, _collect_slot_lcas, _collect_cue_sig,
    _normalize_abstract_sig,
    DiscoveredOperator, Recognition, MIN_DISCOVER_SAMPLES,
    _OP_CONF_RATE_SCALE, DiscoveryRouteStats, StructureTallyStats,
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
from pure_integer_ai.storage.telemetry import (
    push_telemetry_scope,
    reset_telemetry_scope,
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
    from pure_integer_ai.experiments.evaluation_protocol import EvaluationPlan
    from pure_integer_ai.experiments.verification_orchestration import (
        VerificationReport,
    )
    from pure_integer_ai.teacher.probe_set import ProbeSet   # W4 D4·TrainContext.probe_set 注解（运行时 lazy·避上向 import）
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
    # 默认 OFF 守 bit-identical。ON 时在 boot+discovery 后克隆独立 backend/context 跑代表性 trial；
    # 通过只回传报告，失败抛错，正式图、身份、水位和 WorkMemory 在两条路径均保持不变。
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
                                                   SENSE_LEGACY_BRIDGE_TABLE,
                                                   ABSTRACT_MARK_TABLE,
                                                   CONCEPT_CORRESPONDENCE_TABLE,
                                                   WORD_FORM_INDEX_TABLE,
                                                   WORD_FORM_LEGACY_BRIDGE_TABLE)   # 词形 legacy 目录及显式迁移桥
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
    # W4 兼容探针采样：probe_holdout 默认 0 不切 corpus。
    # caller 传 >0 → formal_train 主入口 :1611 切 corpus 末尾 N 作 held-out probe（不喂 boot/discovery/H2/
    # stage/generate/base_freq）。它只给精确内容诊断，缺 provenance，V-00 审计后不再解锁 D4。
    # probe_version 探针版本号（默认 0·caller 派生自 run_id 或显式传·守几百 G 不重训·bit-identical 可复现）。
    # 严格 D4 只能由 evaluation_plan 的完整内容、dedup/provenance cluster 和 EXTERNAL 分账置真。
    probe_holdout: int = 0
    probe_version: int = 0
    # V-00 严格入口由数据 manifest/caller 注入完整 split ledger；与旧尾切兼容路径互斥。
    evaluation_plan: EvaluationPlan | None = None
    # evaluator 只接收已隔离 eval context 和单个计划项；None 时只训练并保留 manifest，不执行探针。
    evaluation_probe_evaluator: Any = None
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
    curriculum_boot_relations: frozenset[str] | None = None   # None=沿用 active scope；显式集合=本调用只 boot 该 delta，backend 保留既有关系。
    active_training_stages: tuple[int, ...] | None = None   # None=完整五阶段；课程 runner 可显式只跑当前训练相位，避免每个关系阶段嵌套完整 formal_train。
    persist_graph_dump: bool = True   # 课程中间相位可关闭昂贵 dump/cursor；最终相位仍持久化权威图。
    telemetry_clock_ns: Callable[[], int] | None = None   # 核心默认无墙钟；实验 runner 可从外部注入单调 ns 时钟。
    telemetry_enabled: bool = False   # V-01 外层诊断；关闭时不扫描表、不采工作集且不改变 canonical 状态。
    telemetry_working_set_bytes: Callable[[], int] | None = None   # 可注入确定性测试源；生产默认读宿主进程工作集。
    # U-02 Unicode sequence/UCD 显式入口：两项必须同时提供；默认不加载外部文件且旧训练 bit-identical。
    # 启用后 token 只物化为 Representation 并接 UCD 外部属性，不创建 LanguageAtom 或最终结构作用。
    unicode_raw_root: str | None = None
    unicode_manifest_path: str | None = None
    # L-01 课程词形正式入口：三项必须同时配置；默认关闭保持旧 corpus token 行为。
    # visible_splits 由调用方显式声明并写在 D-01 manifest 中，正式训练通常只读 train。
    language_course_root: str | None = None
    language_source_manifest_path: str | None = None
    language_course_runtime_language: int | None = None
    language_course_visible_splits: tuple[int, ...] = (SPLIT_TRAIN,)
    # L-02 候选协议由调用方注入完整 MinimalInstruction 键和预算；None 保留 L-01 单 FMM 兼容路径。
    language_segmentation_protocol: Any = None
    # L-03 occurrence 图关系协议由调用方注入；None 保留旧共享概念位置兼容路径。
    language_occurrence_protocol: Any = None
    # L-06 来源 occurrence 位置序 predicate 由调用方注入；必须与 L-03 同时启用。
    language_occurrence_order_protocol: Any = None
    # L-04 递归 Span 协议由调用方注入；正式分词接线必须与 L-03 occurrence 同时启用。
    language_span_protocol: Any = None
    # U-03 句界候选协议由调用方注入；只消费显式 Evidence 和已恢复图选择。
    language_boundary_protocol: Any = None
    # H-01 条件预测目标由调用方注入；必须与 L-03 occurrence 同时启用，默认不产生预测开销。
    language_prediction_protocol: Any = None
    # H-05 structure/Sense 候选全链协议；必须与 occurrence/span 同时启用。
    language_candidate_protocol: Any = None
    # S-02/S-03 正式语义课程；只消费 typed Span/Occurrence/Sense，不读取旧词面顺序。
    language_semantic_course_protocol: Any = None
    # L-05B2B 只读语义恢复；只能从 S-02 图和 active H-00 Evidence 重建请求。
    language_semantic_query_protocol: Any = None
    # R-06 顺序闭环协议与课程必须成对注入；宿主不提供语言、Role 或阈值默认值。
    language_precedence_protocol: Any = None
    language_precedence_course: Any = None
    # R-07 因果闭环协议与课程必须成对注入；课程只提交 typed 请求，宿主不解释 cue 或旧边。
    language_causal_protocol: Any = None
    language_causal_course: Any = None
    # L-05B2A typed formal generation owner factory；None 保留版本化 legacy 兼容链。
    # factory 必须从当前 TrainContext 的真实 S-02/S-07/R-01 owner 装配请求 mapper、planner 和 renderer。
    language_generation_runtime_factory: Any = None
    # L-05B2B 默认课程入口：loader 与 component factory 必须成对提供，且与直接 factory 互斥。
    # generation loader 加载 connector 理论；可选 relation loader 加载版本化 R-01 Core 课程。
    # component factory 只需提供辅助组件；未配置 relation loader 时保留旧 alias 注入兼容路径。
    language_generation_course_loader: Any = None
    language_alias_relation_course_loader: Any = None
    language_generation_component_factory: Any = None
    # typed 阶段4 owner 必须由同一 generation factory installation 提供。
    # L-05B2B typed H2 只使用 V-00 development split 的注入式分维期望；不得读取 held-out 调参。
    language_generation_h2_protocol: Any = None
    # L-05B2B typed floor 只使用 V-00 held-out split，并按注入阈值逐维验收。
    language_generation_floor_protocol: Any = None
@dataclass
class FormalTrainResult:
    """formal_train 产出（阶段完成集 + 最终度量 + dump spaces + 断奶报告）。"""

    run_id: str
    stages_completed: list[int] = field(default_factory=list)
    stages_skipped: list[int] = field(default_factory=list)
    stages_requested: list[int] = field(default_factory=list)
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
    evaluation_plan: EvaluationPlan | None = None
    evaluation_plan_path: str | None = None
    evaluation_plan_sha256: str | None = None
    evaluation_strictly_isolated: bool = False
    evaluation_report: Any = None
    probe_set: ProbeSet | None = None   # W4 D4 留出探针集（config.probe_holdout>0 时 formal_train 主入口建·版本化·W6/caller/test 可查·默认 None）
    holdout_retention: int = 0   # W6 E2 模拟退场 eval 采的探针保持率真值（默认 0 bit-identical·cross_verify 通过率×1000·D1 曲线②度量·真泛化 defer W8）
    word_form_course_report: Any = None   # L-01 课程 manifest、可见 split 和去重计数；未配置 provider 时为 None
    alias_relation_course_report: Any = None
    language_generation_course_report: Any = None
    typed_language_h2_report: Any = None
    typed_language_floor_report: Any = None
    typed_language_stage4_report: Any = None
    occurrence_count: int = 0
    source_record_count: int = 0
    occurrence_order_fact_count: int = 0
    precedence_evidence_count: int = 0
    precedence_relation_reports: tuple[Any, ...] = ()
    causal_relation_reports: tuple[Any, ...] = ()
    span_count: int = 0
    span_candidate_fact_count: int = 0
    prediction_observation_count: int = 0
    prediction_evidence_count: int = 0
    prediction_report_count: int = 0
    structure_candidate_reports: tuple[Any, ...] = ()
    structure_boundary_report: Any = None
    sense_candidate_report: Any = None
    verification_reports: tuple[VerificationReport, ...] = ()
    execution: FormalTrainExecutionStats = field(default_factory=FormalTrainExecutionStats)


def formal_train(config: FormalTrainConfig,
                 corpus: list[CollectedItem], *,
                 backend: StorageBackend,
                 teacher: Any = None,
                 runner: RoundRunner | None = None,
                 weights: JudgeWeights | None = None,
                 metrics: MetricsCollector | None = None,
                 existing_operators: Sequence[DiscoveredOperator] = (),
                 ) -> FormalTrainResult:
    """建立可选外层遥测后执行正式训练，并统一写独立 execution 报告。"""
    result = run_with_execution_telemetry(
        enabled=config.telemetry_enabled,
        backend=backend,
        working_set_source=config.telemetry_working_set_bytes,
        operation=lambda: _formal_train_impl(
            config,
            corpus,
            backend=backend,
            teacher=teacher,
            runner=runner,
            weights=weights,
            metrics=metrics,
            existing_operators=existing_operators,
        ),
    )
    save_execution_metrics(
        run_dir=config.run_dir,
        run_id=config.run_id,
        execution=result.execution,
        language_summary=result.lang_generalization,
    )
    return result


def _formal_train_impl(config: FormalTrainConfig,
                       corpus: list[CollectedItem], *,
                       backend: StorageBackend,
                       teacher: Any = None,
                       runner: RoundRunner | None = None,
                       weights: JudgeWeights | None = None,
                       metrics: MetricsCollector | None = None,
                       existing_operators: Sequence[DiscoveredOperator] = (),
                       ) -> FormalTrainResult:
    """五阶段正式训练主入口（§十二最优路径 + --resume 续训 + H2 + 终 dump）。

  度量门控合格才进下阶段（stage_metric_gate·防缺防超喂）。
  --resume：load_run(base) + cursor_resume stage-skip + check_replay_coverage（E1/E4/E8）。
  阶段3：H2 小批量标定权重 → 开全量 reward。阶段4：promote 三重 + 断奶判据。
  终 dump（dump_run·per-space·新 run_id·E1 权威 base）。
    """
    telemetry_clock = TelemetryClock(config.telemetry_clock_ns)
    _total_started_ns = telemetry_clock.now_ns()
    _input_item_count = len(corpus)
    assign_corpus_source_refs(corpus)
    assert_no_float(config.rounds_per_stage, _where="formal_train.rounds_per_stage")
    train_scope = resolve_train_scope(
        known_stages=STAGES,
        requested_stages=config.active_training_stages,
        active_relations=config.curriculum_active_relations,
        boot_relations=config.curriculum_boot_relations,
    )
    requested_stages = list(train_scope.training_stages)
    ctx = make_train_context(backend, teacher=teacher, weights=weights)
    language_course_configured = (
        config.language_course_root is not None,
        config.language_source_manifest_path is not None,
        config.language_course_runtime_language is not None,
    )
    if any(language_course_configured) and not all(language_course_configured):
        raise ValueError(
            "语言课程 root、源 manifest 和运行期语言键必须同时配置")
    unicode_configured = (
        config.unicode_raw_root is not None,
        config.unicode_manifest_path is not None,
    )
    if unicode_configured[0] != unicode_configured[1]:
        raise ValueError("Unicode raw root 与 manifest path 必须同时配置")
    if unicode_configured[0]:
        from pure_integer_ai.experiments.data_manifest import read_manifest
        from pure_integer_ai.experiments.ucd_adapter import UcdReadOnlyAdapter
        from pure_integer_ai.experiments.unicode_intake import UnicodeIntake
        unicode_manifest = read_manifest(config.unicode_manifest_path)
        unicode_adapter = UcdReadOnlyAdapter(
            config.unicode_raw_root, unicode_manifest)
        ctx.unicode_intake = UnicodeIntake(
            ctx.graph_ontology, unicode_adapter)
    ctx.weaning_phase = config.weaning_phase   # W2 mock POST 注入（默认 PRE 守 bit-identical·WEANING_POST 走 cross-verify :578）
    ctx.judge_source_id = config.judge_source_id   # W3 D3 独立裁判注入（默认 None 守 bit-identical·caller :463 传 build_judge_fn）

    # ---- --resume 续训（E1/E4/E8） ----
    state = CursorState(
        base_run_id=config.base_run_id or config.run_id,
        run_id=config.run_id,
    )
    todo_stages = list(requested_stages)
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
        todo_stages = cursor_resume(state, requested_stages, skippable=SKIPPABLE_STAGES)

    if config.evaluation_plan is not None and config.probe_holdout > 0:
        raise ValueError("V-00 evaluation_plan 与旧 probe_holdout 尾切路径互斥")
    if (config.evaluation_probe_evaluator is not None
            and config.evaluation_plan is None):
        raise ValueError("正式 evaluation probe evaluator 必须配套 V-00 evaluation_plan")
    if (config.language_generation_h2_protocol is not None
            and config.evaluation_plan is None):
        raise ValueError("typed language H2 必须配套 V-00 evaluation_plan")
    default_generation_configured = (
        config.language_generation_course_loader is not None,
        config.language_generation_component_factory is not None,
    )
    if any(default_generation_configured) and not all(
            default_generation_configured):
        raise ValueError(
            "默认 connector course loader 与 component factory 必须成对配置")
    if (config.language_generation_runtime_factory is not None
            and any(default_generation_configured)):
        raise ValueError("直接 generation factory 与默认 connector 课程入口互斥")
    if (config.language_alias_relation_course_loader is not None
            and not all(default_generation_configured)):
        raise ValueError("R-01 课程 loader 必须配套默认 connector 课程入口")
    generation_owner_configured = (
        config.language_generation_runtime_factory is not None
        or all(default_generation_configured)
    )
    if (all(default_generation_configured)
            and config.language_semantic_course_protocol is None):
        raise ValueError("默认 connector 课程需要正式 semantic course runtime")
    if (all(default_generation_configured)
            and config.language_precedence_protocol is None):
        raise ValueError("默认 connector 课程需要正式 R-06/S-07 runtime")
    if (config.language_generation_h2_protocol is not None
            and not generation_owner_configured):
        raise ValueError("typed language H2 必须配套 typed generation owner")
    if (config.language_generation_floor_protocol is not None
            and not generation_owner_configured):
        raise ValueError("typed language floor 必须配套 typed generation owner")
    if (config.language_generation_floor_protocol is not None
            and config.evaluation_plan is None):
        raise ValueError("typed language floor 必须配套 V-00 evaluation_plan")
    if (generation_owner_configured
            and STAGE3_REWARD in requested_stages
            and config.language_generation_h2_protocol is None):
        raise ValueError("正式 typed language 阶段3 必须配置分维 H2")
    if (generation_owner_configured
            and STAGE3_REWARD in requested_stages
            and config.language_generation_floor_protocol is None):
        raise ValueError("正式 typed language 阶段3 必须配置 held-out 分维 floor")
    preview_only = config.evaluation_plan is not None
    # L-01 必须在恢复图之后装配：LanguageBranch 和惰性 Representation 都属于当前
    # 权威图，若先装配再 load，运行时编址和恢复身份可能互相覆盖。
    if all(language_course_configured):
        from pure_integer_ai.experiments.language_course_intake import (
            build_word_form_providers,
        )
        providers, course_report = build_word_form_providers(
            backend=ctx.backend,
            concept_index=ctx.concept_index,
            ontology=ctx.graph_ontology,
            course_root=config.language_course_root,
            source_manifest_path=config.language_source_manifest_path,
            runtime_language=config.language_course_runtime_language,
            visible_splits=config.language_course_visible_splits,
            segmentation_protocol=config.language_segmentation_protocol,
        )
        ctx.word_form_providers = providers
        preview_only = preview_only or config.probe_holdout > 0
        retokenized_items = _apply_word_form_providers(
            corpus,
            providers,
            commit_evidence=not preview_only,
        )
        ctx.word_form_course_report = replace(
            course_report,
            retokenized_items=retokenized_items,
        )
    install_language_graph_protocols(
        ctx,
        occurrence_protocol=config.language_occurrence_protocol,
        occurrence_order_protocol=config.language_occurrence_order_protocol,
        span_protocol=config.language_span_protocol,
        boundary_protocol=config.language_boundary_protocol,
        prediction_protocol=config.language_prediction_protocol,
    )
    if config.language_candidate_protocol is not None:
        install_language_candidate_runtime(
            ctx,
            config.language_candidate_protocol,
        )
    if (config.language_semantic_query_protocol is not None
            and config.language_semantic_course_protocol is None):
        raise ValueError("semantic query protocol 需要 semantic course protocol")
    if config.language_semantic_course_protocol is not None:
        from pure_integer_ai.experiments.language_semantic_runtime import (
            install_language_semantic_course_runtime,
        )
        install_language_semantic_course_runtime(
            ctx,
            config.language_semantic_course_protocol,
            config.language_semantic_query_protocol,
        )
    precedence_configured = (
        config.language_precedence_protocol is not None,
        config.language_precedence_course is not None,
    )
    if any(precedence_configured) and not all(precedence_configured):
        raise ValueError("R-06 precedence protocol 与 course 必须成对配置")
    if all(precedence_configured):
        from pure_integer_ai.experiments.precedence_relation_runtime import (
            install_precedence_relation_runtime,
        )
        install_precedence_relation_runtime(
            ctx,
            config.language_precedence_protocol,
            config.language_precedence_course,
        )
    causal_configured = (
        config.language_causal_protocol is not None,
        config.language_causal_course is not None,
    )
    if any(causal_configured) and not all(causal_configured):
        raise ValueError("R-07 causal protocol 与 course 必须成对配置")
    if all(causal_configured):
        from pure_integer_ai.experiments.causal_relation_course import (
            install_causal_relation_runtime,
        )
        install_causal_relation_runtime(
            ctx,
            config.language_causal_protocol,
            config.language_causal_course,
        )
    generation_factory = config.language_generation_runtime_factory
    if all(default_generation_configured):
        from pure_integer_ai.experiments.language_generation_connector_factory import (
            DefaultLanguageConnectorProductionRuntimeBuilder,
            LanguageConnectorProductionFactory,
        )
        relation_factory = None
        if config.language_alias_relation_course_loader is not None:
            loaded_relation = config.language_alias_relation_course_loader.load(ctx)
            ctx.alias_relation_course_report = loaded_relation.report
            relation_factory = loaded_relation.factory
        loaded_course = config.language_generation_course_loader.load(ctx)
        ctx.language_generation_course_report = loaded_course.report
        generation_factory = LanguageConnectorProductionFactory(
            loaded_course.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                config.language_generation_component_factory,
                relation_factory,
            ),
            loaded_course.stage4_policy,
        )
    if generation_factory is not None:
        from pure_integer_ai.experiments.generation_production_runtime import (
            install_production_generation_runtime,
        )
        install_production_generation_runtime(
            ctx,
            generation_factory,
        )
        if (STAGE4_PROMOTE_WEAN in requested_stages
                and ctx.language_generation_stage4_runtime is None):
            raise RuntimeError(
                "typed language 阶段4 尚无候选生命周期和晋升报告，"
                "禁止回退 legacy promote")

    # V-00 严格计划与 W4 兼容尾切均在分词后执行；所有非训练 split 从全部训练消费者排除。
    evaluation_partition = None
    if config.evaluation_plan is not None:
        evaluation_partition = config.evaluation_plan.partition(corpus)
        protocol = config.evaluation_plan.protocol
        training_corpus = list(evaluation_partition.items(
            protocol.training_split))
        probe_corpus = list(evaluation_partition.items(
            protocol.held_out_split))
        withheld_corpus = list(evaluation_partition.non_training_items())
        ctx.evaluation_plan = config.evaluation_plan
        ctx.evaluation_corpora = evaluation_partition.as_dict()
    else:
        training_corpus, probe_corpus = _split_holdout(
            corpus, config.probe_holdout)
        withheld_corpus = list(probe_corpus)
    has_candidate_preview = any(
        item.word_form_parse is not None for item in training_corpus)
    if preview_only and has_candidate_preview:
        # 全量候选预览未写正式 ledger；切分后只对 training 子集提交 H-00 Evidence。
        _apply_word_form_providers(
            training_corpus,
            ctx.word_form_providers,
            commit_evidence=True,
        )
    if ctx.boundary_hypothesis_engine is not None:
        _prepare_item_boundaries(
            ctx,
            training_corpus,
            commit_evidence=True,
            persist_graph=True,
        )
        _prepare_item_boundaries(
            ctx,
            withheld_corpus,
            commit_evidence=False,
            persist_graph=False,
        )
    if evaluation_partition is not None:
        from pure_integer_ai.teacher.probe_set import (
            is_disjoint,
            make_probe_set,
        )
        all_payloads = sorted(
            assignment.identity.content.payload
            for assignment in config.evaluation_plan.assignments
        )
        content_refs = {
            payload: (0, index + 1)
            for index, payload in enumerate(all_payloads)
        }
        training_refs = [
            content_refs[assignment.identity.content.payload]
            for assignment in config.evaluation_plan.assignments
            if assignment.split == config.evaluation_plan.protocol.training_split
        ]
        probe_refs = [
            content_refs[assignment.identity.content.payload]
            for assignment in config.evaluation_plan.assignments
            if assignment.split == config.evaluation_plan.protocol.held_out_split
        ]
        ctx.probe_set = make_probe_set(
            config.evaluation_plan.protocol.version, probe_refs)
        ctx.probe_corpus = probe_corpus
        ctx.probe_content_disjoint = is_disjoint(
            ctx.probe_set, training_refs)
        ctx.evaluation_strictly_isolated = ctx.probe_content_disjoint
        ctx.probe_set_disjoint = ctx.probe_content_disjoint
        if not ctx.evaluation_strictly_isolated:
            raise RuntimeError("V-00 严格计划的兼容 probe 投影发生身份冲突")
        corpus = training_corpus
    elif probe_corpus:
        from pure_integer_ai.teacher.probe_set import (
            is_disjoint,
            make_probe_set,
            ref_from_signature,
        )
        probe_refs = frozenset(
            ref_from_signature(_item_sig(it)) for it in probe_corpus)
        training_refs = [
            ref_from_signature(_item_sig(it)) for it in training_corpus]
        ctx.probe_set = make_probe_set(config.probe_version, probe_refs)
        ctx.probe_corpus = probe_corpus
        ctx.probe_content_disjoint = is_disjoint(
            ctx.probe_set, training_refs)
        # 旧尾切只证明精确签名不重复，没有 provenance cluster，禁止解锁断奶 D4。
        ctx.probe_set_disjoint = ctx.probe_content_disjoint
        corpus = training_corpus

    r = runner or DefaultRoundRunner()
    mpath = config.metrics_path or os.path.join(
        config.run_dir, config.run_id, "metrics.jsonl")
    own_metrics = metrics is None
    mc = metrics or MetricsCollector(mpath)

    _effective_boot_relations = train_scope.boot_relations
    result = FormalTrainResult(run_id=config.run_id, weights=ctx.weights)
    result.stages_requested = list(requested_stages)
    result.evaluation_plan = ctx.evaluation_plan
    result.evaluation_strictly_isolated = ctx.evaluation_strictly_isolated
    if ctx.evaluation_plan is not None:
        from pure_integer_ai.experiments.evaluation_protocol import (
            write_evaluation_plan,
        )
        evaluation_plan_path = os.path.join(
            config.run_dir,
            config.run_id,
            "evaluation-plan.json",
        )
        result.evaluation_plan_path = str(write_evaluation_plan(
            ctx.evaluation_plan, evaluation_plan_path))
        result.evaluation_plan_sha256 = ctx.evaluation_plan.sha256()
    result.probe_set = ctx.probe_set   # W4 D4 探针集 expose（config.probe_holdout>0 时主入口建·版本化·W6/caller/test 可查）
    result.word_form_course_report = ctx.word_form_course_report
    result.alias_relation_course_report = ctx.alias_relation_course_report
    result.language_generation_course_report = (
        ctx.language_generation_course_report)
    result.execution = FormalTrainExecutionStats(
        input_items=_input_item_count,
        training_items=len(corpus),
        probe_items=len(withheld_corpus),
        active_relations=(None if train_scope.active_relations is None
                          else tuple(sorted(train_scope.active_relations))),
        boot_relations=(None if _effective_boot_relations is None
                        else tuple(sorted(_effective_boot_relations))),
        training_stages=tuple(requested_stages),
    )
    execution_recorder = ExecutionPhaseRecorder(
        enabled=config.telemetry_enabled,
        backend=backend,
        execution=result.execution,
        working_set_source=config.telemetry_working_set_bytes,
    )

    bootstrap_scope_token = (
        push_telemetry_scope(query="bootstrap")
        if config.telemetry_enabled else None)
    bootstrap_snapshot = execution_recorder.snapshot()
    _bootstrap_started_ns = telemetry_clock.now_ns()
    # 语料相关 KB vocab（perf fix·doc/重来_语料相关KB过滤_2026-07-16）：KB vocab-edge（is_a/abstract/mereology/
    # antonym/similar/alias）resolve 后过滤·只留 ≥1 surface 在语料 vocab 的 pair。全量 KB 对 656-paragraph 语料
    # 84-99.5% out-of-corpus ballast → boot 95s + 训练图 660k 边 7x 慢。语料相关过滤随语料 scale（非 hack·非截断）。
    # causes_coverage ②结构 permille 分母稀释是 capability_exam:545 文档已知（FAIL≠结构破裂·reward delta 不变）。
    # **bit-identical**：CI 无 ZERO_AI_LOCAL_DIR→resolve_*_facts 返 []→filter 空 list 返空（filter_pairs_to_vocab
    # `not pairs` 短路）·零行为变。空语料 vocab→filter 返原 pairs（短路·守 bit-identical）。
    # #1143 课程增量 boot helper：curriculum_active_relations None=全 load（bit-identical·既有行为）·
    # frozenset=只该集关系 boot（stage-by-stage 有序学习·镜像 arith S1-S8）。
    def _rel_on(rel: str) -> bool:
        return train_scope.relation_enabled(rel)
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
            if ctx.sense_candidate_course_runtime is not None:
                for _surface, _senses in _sense_pairs:
                    ctx.sense_candidate_course_runtime.form_legacy(
                        ctx,
                        runtime_language=_lang,
                        surface=_surface,
                    )
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
    result.execution.bootstrap_elapsed_ns = telemetry_clock.now_ns() - _bootstrap_started_ns
    execution_recorder.finish(
        "bootstrap",
        bootstrap_snapshot,
        elapsed_ns=result.execution.bootstrap_elapsed_ns,
        item_count=len(corpus),
    )
    if bootstrap_scope_token is not None:
        reset_telemetry_scope(bootstrap_scope_token)
    discovery_scope_token = (
        push_telemetry_scope(query="discovery")
        if config.telemetry_enabled else None)
    discovery_snapshot = execution_recorder.snapshot()
    _discovery_started_ns = telemetry_clock.now_ns()
    _loaded_ops = load_discovered_operators(ctx.backend, space_id=ctx.space_id)
    _existing_ops: list[DiscoveredOperator] = []
    _seen_existing_refs: set[ConceptRef] = set()
    _lang_operator_pool: tuple[DiscoveredOperator, ...] = ()
    _lang_new_operator_refs: frozenset[ConceptRef] = frozenset()
    # 进程内课程算子保留 forming_roots，供后续 REALIZES 标注使用。
    for _op in list(existing_operators) + _loaded_ops:
        if _op.skeleton_ref in _seen_existing_refs:
            continue
        _seen_existing_refs.add(_op.skeleton_ref)
        _existing_ops.append(_op)
    result.discovered_operators, result.recognitions, result.generalization = \
        _discover_and_recognize_arith_operators(ctx, corpus, existing_operators=_existing_ops)
    # 刀6 件7：SENSE_LOOKUP_MODE 须在 _discover_and_recognize_lang_structures（clone 选 sense）之前翻——
    # clone 段（caller 建 COMPOSES 首 sense + recognize_roots clone aligning_root）读 gate·若在 stage loop 前
    # 才翻（旧位 :1080）则 clone 段读到 OFF → 生产路径 clone 永不触发（反 theater 牙失效·纸面闭合·对抗审 P0-1）。
    # stage loop observe（sense_lookup hook → MultiRef → record sc_tn）也用此 gate·同一翻覆盖两处。
    # discovery 和 stage 分别用 context-local token 覆盖，异常时精确复位。
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
    _lang_disc: list[DiscoveredOperator] = []
    _lang_rec: list = []
    if (ctx.language_generation_runtime is None
            or ctx.structure_candidate_runtime is not None):
        discovery_gate_token = gates.push_gate_overrides({
            "SENSE_LOOKUP_MODE": True,
            "COMPOSES_COMBINE_MODE": True,
            "REALIZES_MODE": True,
            "CUE_CLUSTER_MODE": True,
            "ORACLE_PROMOTE_MODE": True,
        })
        try:
            (_lang_disc, _lang_rec,
             result.lang_generalization) = (
                _discover_and_recognize_lang_structures(
                    ctx,
                    corpus,
                    existing_operators=_existing_ops,
                ))
        finally:
            gates.reset_gate_overrides(discovery_gate_token)
    result.discovered_operators.extend(_lang_disc)
    result.recognitions.extend(_lang_rec)
    _lang_ops_by_ref: dict[ConceptRef, DiscoveredOperator] = {}
    for _op in list(_existing_ops) + list(_lang_disc):
        _lang_ops_by_ref.setdefault(_op.skeleton_ref, _op)
    _lang_operator_pool = tuple(_lang_ops_by_ref.values())
    _lang_new_operator_refs = frozenset(op.skeleton_ref for op in _lang_disc)
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
    result.execution.discovery_elapsed_ns = telemetry_clock.now_ns() - _discovery_started_ns
    execution_recorder.finish(
        "discovery",
        discovery_snapshot,
        elapsed_ns=result.execution.discovery_elapsed_ns,
        item_count=len(corpus),
        candidate_count=(
            len(result.discovered_operators) + len(result.recognitions)),
    )
    if discovery_scope_token is not None:
        reset_telemetry_scope(discovery_scope_token)
    stage_loop_scope_token = (
        push_telemetry_scope(query="stage_loop")
        if config.telemetry_enabled else None)
    stage_loop_snapshot = execution_recorder.snapshot()
    _stage_loop_started_ns = telemetry_clock.now_ns()
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
    all_eps: list[Episode | TypedLanguageEpisode] = []
    # 防塌三柱验收累加器（致命5 生产 caller·主循环每 stage 汇总·终写 result.collapse_summary）
    collapse_acc: dict[str, Any] = {"verified": 0, "total": 0, "pillar1_ok": 0,
                                    "pillar2_ok": 0, "pillar3_ok": 0, "low_variance": 0}
    training_gate_token = push_production_training_gates()
    # A1（CUE_EXTRACTOR_MODE 接线·致命3 残留·断奶后语言域 CAUSES 自产源）：
    # 生产入口 observe 须自产 CAUSES（cue_extractor 纯元定义·cue_words 中英 lang 出厂硬件·非接地墙·
    # 断奶后教师退场无手注→必须自产否则 J3 veto reward 锁死）。默认 OFF 守单测 bit-identical·
    # formal_train 生产入口 ON·try/finally 守回归。**翻在此（生产入口）非 run_round_full**：
    # cue 在 _split_item_to_segments（line 452 填 Segment）+ observe（建边）被读·二者在 run_round_full 前段·
    # reward 阶段 ATTRACTOR 翻（line 307）在 observe 之后→翻 reward 阶段太晚；且 run_round_full 须保
    # gate-respecting 可测单元（test_cue_extractor_off_..._e2e 直调 run_round_full+gate OFF 验回归）。
    # 详见 doc/重来_清查后待做整合与执行序.md A1。
    # P0a·ordinal 码点 surface resolver（生产入口翻 ON·镜像 CUE_EXTRACTOR 范式·默认 OFF 守回归·try/finally 守）。
    # surface_of live-read 此 gate -> ON 时 generate 读 concept_correspondence 码点产真实文本（解 A 偏离 #1:42 占位）·
    # OFF 退 None -> 占位（CI bit-identical）。reward 阶段（run_round_full ATTRACTOR 内层）继承此 flip。
    # P0 #1040：generate-dispatch 主缺口修复——slot.ref 派发 token concept（graph.read_token_seq·def_array 存储·
    # repeat-safe）+ ctx_refs token 级（produced_refs/prior_topic_refs）。与 ORDINAL_SURFACE_MODE 同翻（外层 try/finally·
    # observe 在内层前跑·两 gate 同翻：dispatch 出 token concept·surface_of 出真字·系统产真语言非 __seg_* label）。
    # Path C 存储（非 PRECEDES walk）解 reward=0（walk node_type filter 误滤 NODE_CONCEPT token 致空产 → reached_sink
    # False → G2p veto·Path C 直读 def_array → 非空 → reward>0）。CI default OFF bit-identical。
    # P0 #1041：reward 信号 truthiness 校准——judge J4word 项（产出真词覆盖率·读 token_refs）。
    # 三 gate 同翻：dispatch 出 token concept + surface_of 出真字 + reward 反映真词质量（判据②③信号质量·
    # 解 review-2 钉死：旧 reward 对真词/__seg_* 同分）。CI default OFF→J4word=0→reward 逐字现状 bit-identical。
    # G5-C memory consolidate（生产入口翻 ON·#732 落地 dormant→P1 激活·审计 §6 P1）。
    # STAGE4_PROMOTE_WEAN 末 _promote_eligible 后扫 memory_item by info_ref·G5-C 闸判达 → consolidate flip。
    # ctx.memory_read is not None 时 fire（生产训练期实例化）。判据④记忆层晋升轴。
    # 刀4 涌现关系学习（生产入口翻 gate·镜像 CUE_EXTRACTOR 范式·默认 OFF 守回归·try/finally 守）：
    # HYPOTHESIS_MODE = 涌现假设生成 + D:11 SHADOW 落边（reward 阶段 observe 后 episode 前·_run_emergence_hook）。
    # FEED_MODE = reward_propagate concept_targets 扩展（D:11 SHADOW 候选进 experience_count feed·子环3 鸡生蛋破解）。
    # 刀5 件8：CUE_READBACK_MODE 在此翻（兑现刀4 defer 注释·cue_extractor 生产透传已落 :288/:496）。
    # readback = cue_type_of 第二源读 D:11 PRIMARY 边（"引发"类涌现词经 promote 后第二轮返非 None·反 theater）。
    # 对应泛化 v2（生产激活·三 gate 共翻·结构反推机制 live·doc/重来_对应泛化_结构反推_学全 §六 片2）：
    # REALIZES 标 skeleton→R（oracle grounded·内容对命中 ConceptNet）+ CUE_CLUSTER 拆 是/使 异名骨架（ATTR_CUE_SIG 落盘·
    # cue slot 可位）+ ORACLE_PROMOTE tally→promote 结构匹配轨（D:11 删∨·generator 关·tally 建 SHADOW）。
    # 三者共构结构反推·缺一则机制断（无 REALIZES 无 exemplar / 无 CUE_CLUSTER 无 cue slot / 无 ORACLE_PROMOTE 无 promote）。
    # CI default OFF→零 tally→零 D:11 翻→bit-identical（生产 try/finally 翻 ON = 机制 live·非 CI 行为）。
    # 对应桥第 4 gate（readback→generation·学到的 cue 词流入生成·doc/重来_对应泛化_readback_generation_桥 §2.4）：
    # 消费 v2 三 gate 产物（REALIZES exemplar + CUE_CLUSTER cue slot + ORACLE_PROMOTE D:11）→ dispatch_slot 第 8 路 correspondence bonus。
    # 缺此 gate → 桥机制就位但生产 generate 不读 → 学到的对应只识别不产出=白学（设计 §1 命门·post-impl 审严重-1）。
    # 对应桥写侧第 5 gate（COMPOSES_COMBINE_MODE·observe 建 EDGE_INSTANTIATES 真边 on __seg_ struct_ref→skeleton_ref·
    # doc/重来_对应机制生产激活_2026-07-17）：桥读侧（CORRESPONDENCE_SLOT_MODE·上）已生产 flip 但写侧 dormant→读侧恒走空分支
    # （无 INSTANTIATES 边）→ dispatch 第 8 路从不 fire。翻写侧 ON = 完成桥（让造句真用学到的 D:11·解"白学"）= 当初 Phase A.3
    # consumer 激活（翻 REJECT·learned 路径已就位：REALIZES 外源 + tally→promote D:11 + CUE_CLUSTER cue·2 对抗审
    # APPROVE-WITH-CONDITIONS 确认非 theater·§4.0 用户原则：底子合法+学习真+泛化）。gate OFF→无 INSTANTIATES 边→
    # generate.py:154 走空分支→bit-identical（FC12 直守·monkeypatch OFF+≥K lang corpus→零 INSTANTIATES）。
    # 命门③ 候选 B 第 6 gate（CUE_SLOT_FILL_MODE）：无 learned relation cue 时直出骨架 cue；有 D:11
    # learned cue 时把骨架 cue 作为 fallback 送入 correspondence 竞争，winner 仍记 CUE_SLOT_FILL 血统。
    # gate OFF->dispatch_slot:179 双 getattr False 短路->走 collide 返 LINEAGE_CONCEPT_FILL=1->bit-identical（2363 零回归守）。
    # 命门③ 候选 C 第 7 gate（SLOT_LCA_CONSTRAINT_MODE·doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）：内容词位按 slot IS_A LCA 类过滤候选（抽象活化）。
    # 消费 COMPOSES_COMBINE_MODE INSTANTIATES 边（read_instantiates 非 None）+ ATTR_SLOT_ROLE（_cluster_by_lca 写·已 live 零消费者）-> read_slot_lcas 重建 slot_lcas
    # -> dispatch_slot 内容词位 is_a_descendant_of(c, slot_lca) 过滤（reflexive-transitive）。**独立 gate 链**（2 gate·不依赖 cue 链 6 gate·C 独立 2 gate 也可活·生产 B+C 共翻 7 gate 最完整）。
    # 缺此 gate -> 内容词无抽象类约束仍非语义连贯句（B 中间态非终态）= 结构活化无抽象活化 = theater 风险（design §三必需 follow-up·非可选 defer）。
    # gate OFF->dispatch_slot:216 getattr False 短路->candidates 不变->走 collide 返 LINEAGE_CONCEPT_FILL=1->bit-identical。
    # NOTE：FLOOR_ACTIVATION_MODE（floor 端到端激活率·doc/重来_floor_端到端下游激活率_2026-07-17）**不在此 try/finally 翻**
    # ——它是 eval/measurement gate（同 STATISTICAL_WEANING_MODE·env-gated·getattr 读·:2877）·非核心训练变换。
    # 生产 orchestrator `_measure_floor_pass`（observe〔probe_corpus held-out〕+auto_discover〔**不调 tally**〕+generate→
    # measure_floor_activation 读侧后验重导）**defer 课程相位（piece 3）**：须配真 held-out split（probe_holdout>0·课程 run）
    # + 真 curriculum run 才能 e2e 验（同桥生产路径 smoke 验范式）。piece 2 只交付 measure 机制 + gate-gated verdict
    # 接线（weaning.py floor_conjunct·gate OFF→True→bit-identical·审1 严重-1）+ FC1-8 fixture 预验。orchestrator 设计
    # 复杂（held-out INSTANTIATES 桥非 observe 自建 / auto_discover 取 __disc_lang_* 非 __seg_ / REALIZES 独立 gate pass）
    # 见设计档 §9·piece 3 商讨落地。
    # STEP5 PR2：operator D:11 readback 第二源（arith_op_of/comparison_op_of 读 D:11 PRIMARY→OP_*→opcode·
    # 镜像 EMERGENT_RELATION_CUE_READBACK_MODE 两源范式·gate OFF 退化纯 frozenset bit-identical）。
    # 审计根治 [严重-1]：modal D:11 readback 第二源（modal_op_of/is_modal_cue 读 D:11 PRIMARY→MODAL_KIND→modality·
    # 镜像 OPERATOR_D11_READBACK_MODE 两源范式·gate OFF 退化纯 frozenset _MODAL_CUES bit-identical）。
    # #940 否定词 D:11 readback 第二源（is_negation_cue 读 D:11 PRIMARY→TYPE_NEGATION concept·
    # 镜像 MODAL_D11_READBACK_MODE 两源范式·gate OFF 退化纯 frozenset _NEGATION_CUES bit-identical）。
    # 止血 #1146（methodology §五·reward 非 frame）：CAUSES edge reward 写按域过滤——语言/bare 域剔出
    # reward_propagate 落点① edge 写（reward 结构性 theater·dead-end/veto→tn++ 惩罚唯一 reward-active 边·有害）。
    # 生产翻 ON（语言域 reward 退场·CAUSES 掌握走刀 constructive-check 不接 strength）·CI default OFF 退化现状
    # bit-identical（OFF → reward_propagate 落点① 逐字现状·判据 = shared.REWARD_LEGITIMATE_DOMAINS 与 judge G5 同源）。
    # STEP5 PR4：EDGE_SIMILAR slot-filler 消费者（dispatch_slot 读 EDGE_SIMILAR 扩展 slot 候选·
    # gate OFF 不扩展 bit-identical·D2 合规非向量·不接 reward）。
    # 审计根治 [严重-3]：B6 生成侧 dispatch_slot pronoun scoring（读 pr_tn 加 slot 候选分·
    # gate OFF 不读 bit-identical·pair-key 对偶 observe 侧·不接 reward·pr_tn sign-agnostic）。
    # 刀5 件5：SELECTION_PREF_MODE 在此翻（选择倾向统计 builder·observe 段内共现写 sp_tn·§十 边约束）。
    # 生产 ON 防纸面闭合（selection_pref_count 表生产写真写·非空表 theater）·PR 软加权 dock seed defer S4。
    # M1片2：intent 分类替换硬编码（classify_intent·解 G3a 死门·doc/重来_M1片2_intent分类设计_2026-07-08.md）。
    # 生产 ON 防纸面闭合（否则 gate 永不活 → classify_intent 永不调 → is_causal 永假 → reward 退化核心病灶未修 = theater）。
    # 同 CUE/EMERGENT try/finally 守回归（CI gate OFF → reward :366 + H2 :1448 两处硬编码走原路径 bit-identical）。
    # 性能修复（2026-07-08 训练测试探索）：COOCCURS 段内配对窗口化 O(L²)→O(L·K)·解训练 scaling 爆炸。
    # 生产 ON 防纸面闭合（否则长文本/真语料训练 71s/n=5 段跑不动）。同 CUE/EMERGENT try/finally 守回归
    # （CI gate OFF → cooccurs.segment_cooccurrence_pairs i<j 全配对 bit-identical 现状）。镜像范式 :1226。
    # 总收口 0.1：跨段去重（COOCCURS_DEDUP_MODE·add_cooccurs_dedup·解 append-only 堆叠 LIVE 病灶①·阻塞 #734）。
    # 生产 ON（镜像 COOCCURS_WINDOW·observe build_cooccurs flip 后走 dedup·reader 读 strength 协同）。
    # gate OFF 时 reader 读 strength 恒 1 等价数行·ON 时 strength=频次；bit-identical 由 gate OFF 单测守。
    # S2 dead-end 根因 §10.3：PRECEDES 跨 round 去重（PRECEDES_DEDUP_MODE·add_precedes_dedup·
    # mirror COOCCURS_DEDUP_MODE·解 observe 跨 round 16× 重复·2256→153 distinct）。生产 ON 防纸面闭合
    # （否则 3 builder 走旧 add 堆叠 = 16× 重复未修 = theater）。gate OFF 单测守回归（additive·bit-identical）。
    # 诚实边界：dedup 确定性 perf 16× + 数据卫生赢·未必解 dead-end（AND 语义另议）·dedup 后重测定。
    # CAUSES 跨 round 去重（CAUSES_DEDUP_MODE·add_causes_dedup·mirror PRECEDES_DEDUP_MODE·解 observe 16× 重复
    # 边膨胀·56% 墙大头）。生产 ON 防纸面闭合（否则 _insert_causes 走旧 add 堆叠 16× = theater）。reward 影响
    # 零核证（snapshot_strengths 覆写去重·16x 与 1x 同 delta）·修假汇聚 bug + 消边膨胀。gate OFF 单测守回归。
    # PR 热区过滤（HOTZONE_MODE·A3PRWrapper.build BFS k-hop·mirror CAUSES_DEDUP_MODE·解全图 8677²(n=656)）。
    # 生产 ON 防纸面闭合（否则 PR 全图=defer 意外态=theater·设计本意卷二:110 hotzone）。reward 影响零
    # （resolver 证 PR 不回流 path）·仅变 PR 诊断。k=2/PR_MAX_NODES=2048·须配合 PR_B2_LARGE_N_MODE。
    # gate OFF 单测守回归（全图 bit-identical）。
    # PR_B2_LARGE_N_MODE（生产大 n>512 走 B2 迭代替 B1 O(n³)·CI n<512 走 B1 bit-identical·gate check `ON and n>阈值`）：
    # HOTZONE_MODE ON 缩 matrix 但 k-hop 密集图仍可能 >512·须 B2 防 B1 炸（B2 audit1 LATENT-BUG 修·n<512 时 gate ON
    # 但 n<=阈值->仍 B1 bit-identical）。生产 ON（镜像 HOTZONE_MODE·prior n=656 经 env 设·现落 code 显式）。
    # S2 dead-end factor A：PRECEDES AND→OR（PRECEDES_OR_MODE·a2_stepper 推进语义·dag_path language-only
    # 故 production 安全·解重复词概念多前驱 AND 全 active 永不满足致 dead-end）。生产 ON 防纸面闭合·
    # 外层 try 覆盖 episode_loop + _rebuild_path + H2 标定（同生产语义）。gate OFF 单测守回归（additive）。
    # 诚实：A 修不保证 REACHED_SINK（factor B COVERAGE_THRESHOLD 可能仍挡·A 修后须测）。
    # S2 dead-end factor C：PRECEDES oi-first-occ 序遍历（PRECEDES_OI_MODE·F2·doc/重来_F2_PRECEDES_oi遍历_设计_2026-07-09·v3）。
    # factor C = language PRECEDES 概念成环·Kahn 丢环节点含 sink -> sink 永不可达 -> reward 恒 0 -> language 零学习。
    # 生产 ON 防纸面闭合（否则 a2_layer_oi 永不调 = F2 孤儿 = theater·镜像 PRECEDES_OR_MODE 范式）。
    # OI_MODE ON 时 DEDUP+OR 也 ON（三者叠加·解 L1 重复+L2 AND/OR+L3 factor C）。
    # 诚实边界：末段 tokens 在 sink 后不访（1 段损失）+ backward CAUSES 丢 + OR 死锁依赖 ATTRACTOR/EXPLORATION（生产 ON）。
    # gate OFF 单测守回归（a2_layer Kahn·bit-identical）。
    # factor E：层1 同段指代候选（PRONOUN_INTRASEG_MODE·doc/重来_factorE_层1指代_intra_seg_设计_2026-07-09）。
    # factor E = 同段前指代词（"动物...它们"同段）无候选 → dangling → J4 ② fire → G4 veto → reward=0。
    # 生产 ON 防纸面闭合（否则层1 块永不执行 = judge.py:58 注释"层1 已解析"仍是 theater·镜像 PRECEDES_OI_MODE 范式）。
    # 诚实边界：层1 启发式近因非语义消解（stable≠correct）·reward>0=可训练非语义正确。
    # G2 修饰方向A：head 偏好 read-time 加权（MODIFIER_DIRECTION_MODE·dispatch_slot 第 6 路 head_pref_score）。
    # 生产 ON 防纸面闭合（否则 source 写 modification_hist 但 read 不用 = theater）。source write gate-independent。
    # B6 指代维 方案3 tn+fn 路（PRONOUN_RESOLVE_COUNT_MODE·count 写在 observe 阶段 resolve_pronoun_occurrence·
    # 非 episode_loop·故在此大 try/finally 翻 ON 盖 observe·同 PRONOUN_INTRASEG_MODE 范式·非 run_round_full 小 try/finally）。
    # 生产 ON 防纸面闭合（否则 resolve 不写 pr_tn/pr_fn + consumer 不读 pr_tn = theater·§九.2 病灶"attribute 给谁"未解）。
    # 诚实边界：指代维 reward=J4 bool veto（非 graded）·consumer 自消费 reward>0 鲁棒·pr_sn 教师 P2 defer·per-occurrence 避 β_arith。
    # 归一化半 A：功能词/hub 排除（EXCLUDE_FUNCTION_MODE·read-time hub_degree 过滤 3 点·
    # doc/重来_归一化与功能词排除_设计_2026-07-08.md·对抗挖修正 §二序2）。生产 ON 防纸面闭合（否则 gate
    # 永不活 → 3 点不过滤 → hub 污染 collide_score/_cooccurs_count/refers_occurrence 未修 = theater）。
    # 无条件翻 ON（镜像 M1/COOCCURS_WINDOW·与决断 A4 一致·observe 在 stage loop 内 flip 后才跑·is_hub
    # 调时读 COOCCURS·表未注册 try/except KeyError→False 退化 bit-identical OFF·无除法无 crash）。
    # 刀 A：语言域时序 cue verify episode（TIME_SEQ_PROOF_MODE·run_round_full 语言域路由分支）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 时序验序器孤儿 = theater·镜像 COOCCURS/M1 范式）。
    # 刀 B：语言域数值等式 cue verify episode（NUMERIC_PROOF_MODE·run_round_full 语言域路由分支·numeric priority）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 数值验序器孤儿 = theater·镜像 TIME_SEQ_PROOF_MODE 范式）。
    # 刀 C：语言域全称量化 cue verify episode（UNIVERSAL_PROOF_MODE·run_round_full 语言域路由分支·numeric>universal>precedes）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 全称验序器孤儿 = theater·镜像 TIME_SEQ/NUMERIC 范式）。
    # A1·STEP6：语言域存在量化 cue verify episode（EXISTENTIAL_PROOF_MODE·run_round_full 语言域路由分支·
    # numeric>comparison>universal>existential>precedes 序）。入口保留用于 typed Evidence adapter 接入；
    # 当前只有 SUBSET_EQ 祖先图，没有 MEMBER/nonempty/overlap/DISJOINT，必须诚实弃权。
    # 刀 D：语言域比较 cue verify episode（COMPARISON_PROOF_MODE·run_round_full 语言域路由分支·numeric>comparison>universal>precedes）。
    # 生产 ON 防纸面闭合（否则路由永不走 = 比较验序器孤儿 = theater·镜像 TIME_SEQ/NUMERIC/UNIVERSAL 范式）。
    # G1+#774：属性命题 reification（PROPOSITION_MODE·observe build_property_edges 建命题节点+PROPERTY 出边·
    # G3b 全局扫命题节点判矛盾·has_value_claim 真活激活 G3b·反 theater）。生产 ON 防纸面闭合（否则
    # extract_property_claims_gated 永返 [] → property_claims 空 → has_value_claim 永 False → G3b 不激活 =
    # judge.py:236 has_value_claim 门 dead = theater·镜像 TIME_SEQ/NUMERIC/UNIVERSAL 范式）。
    # B1 否定收口（#888·STEP1）：NEGATION_MODE ON·否定窗口激活·pol=1 命题节点建独立 surface。
    # 镜像 PROPOSITION_MODE·生产 ON 防纸面闭合（否则 negation_on=False·"不是"错位 skip·否定命题永不建 = theater）。
    # CI corpus 不含"不是" -> 否定窗口不触发 -> 既有测零回归（bit-identical）。
    # B2 情态收口（STEP6 PR2）：MODALITY_MODE ON·情态窗口激活·modality 填值（0-4）·命题节点建独立 surface 后缀 _0_{mod}。
    # 镜像 NEGATION_MODE·生产 ON 防纸面闭合（否则 modality_on=False·"必然是"错位 skip·情态命题永不建 = theater）。
    # CI corpus 不含"必然/可能" -> 情态窗口不触发 -> 既有测零回归（bit-identical）。
    # #1134 程度 intensity 收口：DEGREE_MODE ON·degree 窗口激活（tokens[val_idx] degree cue → value 后移+intensity 填值）·
    # 命题节点建独立 surface 后缀 _i{num}_{den} + ATTR_PROP_INTENSITY=30。镜像 MODALITY_MODE·生产 ON 防纸面闭合
    # （boot 已 populate_degree_cues 喂 cache·OFF 则 is_degree_cue 恒 False·intensity 恒 1/1 = 既有 bit-identical）。
    # **诚实边界**：intensity magnitude 暂无消费者（G3b 读 count·judge 只权 CAUSES/PRECEDES）·consumer defer·dormant 非 theater。
    # 刀6 件7 SENSE_LOOKUP_MODE 已在 _discover_and_recognize_lang_structures（:1035）前翻（clone 段 + observe 共用）。
    # SENSE_LOOKUP_MODE 已包含在当前 stage 的 context-local override 中。
    try:
        # E7 pre-flight 放量门接通生产主入口（S12 follow-up·破纸面闭合·config.pre_flight 守）：
        # boot+discovery 完成 + 生产 gates 已翻后，pre_flight 在 V-06 独立 backend/session 中试跑；
        # 无论通过、失败或异常，正式 backend、身份缓存、WorkMemory 和逻辑水位均不改变。
        if config.pre_flight:
            pf_rounds = (config.pre_flight_rounds if config.pre_flight_rounds is not None
                         else PRE_FLIGHT_ROUNDS)
            result.execution.preflight_item_runs += min(pf_rounds, len(corpus))
            rep = pre_flight(ctx, corpus, rounds=pf_rounds,
                             runner=r, replay_needed=config.replay_needed,
                             config=config, backend_factory=lambda: DictBackend())
            if not rep.passed:
                raise RuntimeError(
                    "pre_flight 放量门失败（禁放量·守几百G不重训红线）：" + str(rep.detail))
            result.pre_flight_report = rep
        for stage in todo_stages:
            stage_scope_token = (
                push_telemetry_scope(query="training_stage", stage=stage)
                if config.telemetry_enabled else None)
            stage_snapshot = execution_recorder.snapshot()
            stage_started_ns = telemetry_clock.now_ns()
            if stage == STAGE5_MULTIMODAL:
                # defer·非训练（机制骨架随模态扩展·§十二阶段5）
                result.stages_completed.append(stage)
                execution_recorder.finish(
                    "training_stage",
                    stage_snapshot,
                    elapsed_ns=telemetry_clock.now_ns() - stage_started_ns,
                    stage=stage,
                )
                if stage_scope_token is not None:
                    reset_telemetry_scope(stage_scope_token)
                continue
            cfg = stage_gate_config(stage)
            active = stage_active_gates(cfg)
            if (stage == STAGE4_PROMOTE_WEAN
                    and ctx.language_generation_runtime is not None
                    and ctx.language_generation_stage4_runtime is None):
                raise RuntimeError(
                    "typed language 阶段4 尚无候选生命周期和晋升报告，"
                    "禁止回退 legacy promote")
            # gate 二分：TRAINING_MODE OFF → reward/promote 不生效·降为 observe-only（bit-identical）
            reward_active = active["reward"]
            # runner 按有效阶段跑（reward 未激活→observe-only·stage<STAGE3）
            eff_stage = stage if reward_active else STAGE2_CAUSES_ABS

            # 阶段3 H2：先小批量标定权重 → 开全量 reward（§十四 H2·鸡生蛋破解）
            if stage == STAGE3_REWARD and reward_active:
                if ctx.language_generation_runtime is not None:
                    if config.language_generation_h2_protocol is not None:
                        result.typed_language_h2_report = run_typed_language_h2(
                            ctx,
                            r,
                            config.language_generation_h2_protocol,
                        )
                        if result.typed_language_h2_report.complete is not True:
                            raise RuntimeError(
                                "typed language H2 分维校准未通过，禁止进入全量 reward")
                elif ctx.teacher is not None:
                    ctx.weights = _h2_calibrate(
                        ctx, corpus, r, execution=result.execution)

            # per-round 执行（reward 未激活→observe-only·runner 返 None）
            items = _stage_items(corpus, stage, config.rounds_per_stage)
            stage_candidate_count = (
                0 if not config.telemetry_enabled
                else sum(item_candidate_total(item) for item in items)
                * config.rounds_per_stage)
            eps: list[Episode | TypedLanguageEpisode] = []
            per_round = config.weaning_round_series   # W7 断点6：per-round series（设计 per-run·解 observe-only 0 混入 bug）
            # #1143 统计层断奶·教师干预测量（fadeout 锚点 data·gate STATISTICAL_WEANING_MODE 守 bit-identical）：
            # _sw_int ON 时 snapshot teacher.call_count 轮边界 delta → intervention_rate/dependency（真测非 stub-0）。
            # OFF（默认 CI）→ delta=0 → intervention_rate=0（既有 bit-identical·2审 HIGH-1 修：fadeout 不再 vacuous）。
            _sw_int = getattr(gates, "STATISTICAL_WEANING_MODE", False)
            _cc_stage_before = (getattr(ctx.teacher, "call_count", 0) if _sw_int else 0)
            for r_idx in range(config.rounds_per_stage):
                _cc_before = (getattr(ctx.teacher, "call_count", 0) if _sw_int else 0)
                result.execution.stage_batch_calls += 1
                result.execution.stage_item_runs += len(items)
                batch_eps = _run_round_batch(ctx, r, items, eff_stage, round_id)
                eps.extend(batch_eps)
                _cc_delta = ((getattr(ctx.teacher, "call_count", 0) - _cc_before) if _sw_int else 0)
                _intervention_rate = (_cc_delta * 1000) // max(len(batch_eps), 1)
                if per_round:
                    # W7 断点6：per-round record（每 batch_eps 一次·末4 窗口全 stage3/4 verify flat → plateau True）。
                    # promote/oov 仅 stage4 末轮（_promote_eligible 副作用 tier flip 须一次·stage4 末轮等价既有 stage loop 末）。
                    is_last_r4 = (stage == STAGE4_PROMOTE_WEAN and active["promote"]
                                  and ctx.language_generation_runtime is None
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

            typed_stage4_report = None
            if (stage == STAGE4_PROMOTE_WEAN
                    and ctx.language_generation_runtime is not None):
                typed_episodes = tuple(
                    item for item in eps
                    if isinstance(item, TypedLanguageEpisode)
                )
                if not typed_episodes or len(typed_episodes) != len(eps):
                    raise RuntimeError(
                        "typed language 阶段4必须只消费本阶段非空 typed episode")
                typed_stage4_report = (
                    ctx.language_generation_stage4_runtime.apply(
                        typed_episodes))
                result.typed_language_stage4_report = typed_stage4_report

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
                if (stage == STAGE4_PROMOTE_WEAN and active["promote"]
                        and ctx.language_generation_runtime is None):
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
            if (stage == STAGE4_PROMOTE_WEAN
                    and ctx.language_generation_runtime is not None):
                stage_gate_passed = (
                    typed_stage4_report is not None
                    and typed_stage4_report.complete is True)
            elif (stage == STAGE3_REWARD
                    and ctx.language_generation_runtime is not None):
                result.typed_language_floor_report = run_typed_language_floor(
                    ctx,
                    r,
                    config.language_generation_floor_protocol,
                )
                stage_gate_passed = (
                    result.typed_language_floor_report.complete is True)
            else:
                stage_gate_passed = stage_metric_gate(stage, snap)
            if stage_gate_passed:
                mark_completed(state, stage, skippable=is_skippable(stage))
                result.stages_completed.append(stage)
            else:
                # typed 阶段4报告是正式完成协议，失败只能保留在 requested，不能伪记 completed。
                # legacy 阶段暂保留“已跑完轮次”的历史结果语义，等待对应课程门迁移时再拆分。
                if not (stage == STAGE4_PROMOTE_WEAN
                        and ctx.language_generation_runtime is not None):
                    result.stages_completed.append(stage)
            execution_recorder.finish(
                "training_stage",
                stage_snapshot,
                elapsed_ns=telemetry_clock.now_ns() - stage_started_ns,
                stage=stage,
                item_count=len(items) * config.rounds_per_stage,
                candidate_count=stage_candidate_count,
            )
            if stage_scope_token is not None:
                reset_telemetry_scope(stage_scope_token)
            if not stage_gate_passed:
                break

        # typed stage4 只完成候选 lifecycle；W-09 接线前不得回退旧标量断奶、旧 floor 或退场模拟。
        if (STAGE4_PROMOTE_WEAN in result.stages_completed
                and ctx.language_generation_runtime is not None):
            result.weaning_blockers = [
                "W-09_typed_weaning_protocol_missing",
            ]
        # 阶段4 断奶判据（D1-D5/E2 六闸门·#358 完整实现·非布尔阈值·非只看 4 能力指标平台）
        elif STAGE4_PROMOTE_WEAN in result.stages_completed:
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
                ctx.holdout_retention, ctx.e2_eval_passed = _run_simulated_offline_eval(
                    ctx, corpus, backend)
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
            # D2 是旧标量 Episode 的断奶判据；typed 多维信号由阶段4报告验收，不能伪装成旧 veto/dead_end。
            # 混合迁移期只把旧 Episode 交给既有单点实现，纯 typed run 因无旧证据保持未通过。
            from pure_integer_ai.cognition.result.convergence import neg_pathway_active_from
            legacy_eps = [item for item in eps if isinstance(item, Episode)]
            neg_pathway_active = neg_pathway_active_from(legacy_eps)
            # D3 裁判源独立：W3 ctx track（build_judge_fn :463 算 sources_disjoint({j_sid},{teacher_sid}) 设此·
            # 单一真相源·非硬编同源重算）。默认 None→j_sid 回落 teacher_sid→同源 False（bit-identical）。
            # caller 传独立 judge_source_id（与 teacher.source_id 不相交）→ True（D3 通用路径过·W8 语言域独立裁判）。
            # 算术域绕 judge（_run_verify_round:374）judge_fn 不构建→ctx.judge_source_independent=False→D3 仍 False（W7 接 judge_source_independent_arith）。
            judge_source_independent = ctx.judge_source_independent
            # D4 只读取 V-00 完整 ledger；旧尾切的精确内容诊断不得替代 provenance 隔离。
            probe_set_disjoint = ctx.evaluation_strictly_isolated
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

        result.execution.stage_loop_elapsed_ns = telemetry_clock.now_ns() - _stage_loop_started_ns
        execution_recorder.finish(
            "stage_loop",
            stage_loop_snapshot,
            elapsed_ns=result.execution.stage_loop_elapsed_ns,
            item_count=result.execution.stage_item_runs,
        )
        if stage_loop_scope_token is not None:
            reset_telemetry_scope(stage_loop_scope_token)
        finalize_scope_token = (
            push_telemetry_scope(query="finalize")
            if config.telemetry_enabled else None)
        finalize_snapshot = execution_recorder.snapshot()
        _finalize_started_ns = telemetry_clock.now_ns()
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
        generate_gate_token = gates.push_gate_overrides({
            "CODE_UNPARSE_MODE": True,
            "SYMBOLIC_TRANSFORM_MODE": True,
            "SYMBOLIC_RELATION_MODE": True,
        })
        try:
            _gen_eps, result.generate = _run_task_driven_generate(ctx, corpus, _all_ops)
        finally:
            gates.reset_gate_overrides(generate_gate_token)
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

        if config.evaluation_probe_evaluator is not None:
            result.evaluation_report = run_evaluation_plan(
                ctx,
                config.evaluation_probe_evaluator,
            )

        # 终 dump（per-space·新 run_id·E1 权威 base）
        if config.persist_graph_dump:
            dump_scope_token = (
                push_telemetry_scope(query="dump")
                if config.telemetry_enabled else None)
            dump_snapshot = execution_recorder.snapshot()
            dump_started_ns = telemetry_clock.now_ns()
            result.dump_spaces = dump_run(
                backend, config.run_dir, config.run_id,
                spaces=[ctx.space_id], tables=config.dump_tables)
            result.execution.graph_dump_calls += 1
            # E8：持久化 cursor state（已完成阶段集·供下次续训 stage-skip）
            _save_cursor(state, config.run_dir, config.run_id)
            execution_recorder.finish(
                "dump",
                dump_snapshot,
                elapsed_ns=telemetry_clock.now_ns() - dump_started_ns,
                dump_calls=1,
            )
            if dump_scope_token is not None:
                reset_telemetry_scope(dump_scope_token)
        result.final_metrics = mc.snapshot()
        result.holdout_retention = ctx.holdout_retention   # W6 E2 模拟退场 eval 采的保持率真值（默认 0 bit-identical）
        result.weights = ctx.weights
        if result.lang_generalization is not None:
            result.lang_generalization.structure_state = measure_language_structure_state(
                ctx.backend, ctx.concept_graph, _lang_operator_pool,
                new_operator_refs=_lang_new_operator_refs,
                space_id=ctx.space_id)
        result.collapse_summary = collapse_acc
        # #723 G 归因：挂全 episode 列表（collect_episodes=True 时·harness 考核读 G_meta 5 字段建交叉表）
        if config.collect_episodes:
            result.episodes = all_eps
        # 续训 cursor state 记 skipped（resume 时跳过）
        result.stages_skipped = [s for s in requested_stages if s not in todo_stages]
        result.execution.finalize_elapsed_ns = telemetry_clock.now_ns() - _finalize_started_ns
        result.execution.total_elapsed_ns = telemetry_clock.now_ns() - _total_started_ns
        execution_recorder.finish(
            "finalize",
            finalize_snapshot,
            elapsed_ns=result.execution.finalize_elapsed_ns,
        )
        if finalize_scope_token is not None:
            reset_telemetry_scope(finalize_scope_token)
    finally:
        reset_production_training_gates(training_gate_token)
        # 正式训练是一个 session；即使阶段中途异常，也不能把活动 scope 留给下一次运行。
        if ctx.work_memory.active_episode_scope is not None:
            ctx.work_memory.abort_episode()
        if ctx.work_memory.active_document_scope is not None:
            ctx.work_memory.end_document()
        if ctx.work_memory.active_session_scope is not None:
            ctx.work_memory.end_session()
        if own_metrics:
            mc.close()
    if ctx.occurrence_index is not None:
        result.occurrence_count = ctx.occurrence_index.occurrence_count()
        result.source_record_count = ctx.occurrence_index.source_count()
    if ctx.occurrence_order_writer is not None:
        result.occurrence_order_fact_count = (
            ctx.occurrence_order_writer.count())
    if ctx.precedence_relation_runtime is not None:
        result.precedence_evidence_count = (
            ctx.precedence_relation_runtime.evidence_count())
        result.precedence_relation_reports = tuple(
            ctx.precedence_relation_reports)
    if ctx.causal_relation_runtime is not None:
        result.causal_relation_reports = tuple(ctx.causal_relation_reports)
    if ctx.span_index is not None:
        result.span_count = ctx.span_index.span_count()
        result.span_candidate_fact_count = len(
            ctx.span_index.candidate_statements())
    if ctx.language_prediction_runtime is not None:
        prediction = ctx.language_prediction_runtime
        result.prediction_observation_count = (
            prediction.engine.model.observation_count())
        result.prediction_evidence_count = prediction.evidence_count()
        result.prediction_report_count = prediction.report_count()
    result.structure_candidate_reports = tuple(
        ctx.structure_candidate_reports)
    result.structure_boundary_report = ctx.structure_boundary_report
    result.verification_reports = tuple(ctx.verification_reports)
    if ctx.sense_candidate_course_runtime is not None:
        result.sense_candidate_report = (
            ctx.sense_candidate_course_runtime.report())
    return result
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
