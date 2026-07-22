"""训练校准、held-out、floor 和离线评测 runtime。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveredOperator,
    load_discovered_operators,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.shared.types import (
    ConceptRef,
    FloorActivation,
    InputPayload,
    IntentType,
    INTENT_QUESTION,
    JudgeWeights,
    MODALITY_ARITH,
    MODALITY_LANGUAGE,
    OutputPart,
    OutputResult,
    PathData,
    PathResult,
    STAGE_TRAINING,
    TERMINAL_REACHED_SINK,
    WEANING_PRE,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.observe import observe
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_protocol import (
    EvaluationAssignment,
    EvaluationProtocolError,
    EvaluationReport,
    ProbeOutcome,
    ProtocolKey,
    build_evaluation_report,
    collected_item_content_identity,
    evaluate_probe,
)
from pure_integer_ai.experiments.language_observation import (
    _item_sentence_bounds,
    _materialize_item_spans,
    _prepare_item_boundary,
    _run_item_predictions,
    _run_item_sense_candidates,
    _split_item_to_segments,
)
from pure_integer_ai.experiments.language_sense_candidate_runtime import (
    observe_sense_lookup,
)
from pure_integer_ai.experiments.round_runtime import (
    RoundRunner,
    _build_space_ctx,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    _item_document_identity,
    _item_observation_identity,
    _item_occurrence_scope,
)
from pure_integer_ai.experiments.train_execution import FormalTrainExecutionStats
from pure_integer_ai.experiments.verification_dispatch import (
    is_verify_modality as _is_verify_modality,
)
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.training.mode_b_cross_verify import cross_verify_pair
from pure_integer_ai.training.oracle import calibrate_weights
from pure_integer_ai.training.stages import STAGE3_REWARD, STAGE4_PROMOTE_WEAN
from pure_integer_ai.training.vm_proof import vm_proof_fn_factory

H2_CALIB_BATCH = 16
_DISC_LANG_SEED = "formal_train.disc_lang"
_DISC_LANG_ALIGN_SEED = "formal_train.disc_lang_align"

EvaluationProbeEvaluator = Callable[[
    TrainContext,
    CollectedItem,
    EvaluationAssignment,
    ProtocolKey,
], ProbeOutcome]


def _evaluation_item_for(
        ctx: TrainContext,
        assignment: EvaluationAssignment,
        ) -> CollectedItem:
    """按完整来源和内容从已隔离 split 中读取唯一原语料项。"""
    items = ctx.evaluation_corpora.get(assignment.split, ())
    matches = [
        item for item in items
        if item.source_ref is not None
        and item.source_ref.stable_key()
        == assignment.identity.source_ref.stable_key()
        and collected_item_content_identity(item)
        == assignment.identity.content
    ]
    if len(matches) != 1:
        raise EvaluationProtocolError("评测 split 中的完整数据身份缺失或不唯一")
    return matches[0]


def run_evaluation_plan(
        ctx: TrainContext,
        evaluator: EvaluationProbeEvaluator,
        ) -> EvaluationReport:
    """逐探针建立独立沙箱，并返回不含综合分数的分维评测报告。"""
    from pure_integer_ai.experiments.evaluation_isolation import (
        isolated_evaluation,
    )

    if ctx.evaluation_plan is None or not ctx.evaluation_strictly_isolated:
        raise EvaluationProtocolError("正式 probe runner 要求已通过 V-00 的严格计划")
    if not callable(evaluator):
        raise TypeError("evaluation probe evaluator 必须可调用")
    plan = ctx.evaluation_plan
    observations = []
    probe_index = 0
    for assignment in plan.assignments:
        if assignment.split == plan.protocol.training_split:
            continue
        item = _evaluation_item_for(ctx, assignment)
        for dimension in assignment.dimensions:
            label = f"v00-probe-{plan.protocol.version}-{probe_index}"
            with isolated_evaluation(ctx, label=label) as eval_ctx:
                local_item = copy.deepcopy(item)
                observation = evaluate_probe(
                    plan,
                    assignment,
                    dimension,
                    lambda: evaluator(
                        eval_ctx,
                        local_item,
                        assignment,
                        dimension,
                    ),
                )
            observations.append(observation)
            probe_index += 1
    return build_evaluation_report(plan, observations)

def _observe_eval_item(ctx: TrainContext, item: CollectedItem, *,
                       stage: int, round_id: int, space_ctx: Any) -> Any:
    """在评测专属 session/document/episode 中观察一个样本。"""
    if ctx.boundary_hypothesis_engine is not None and item.boundary_parse is None:
        _prepare_item_boundary(
            ctx,
            item,
            commit_evidence=False,
            persist_graph=False,
        )
    segments = _split_item_to_segments(
        item,
        backend=ctx.backend,
        edge_store=ctx.edge_store,
        space_id=ctx.space_id,
        concept_index=ctx.concept_index,
    )
    if not segments:
        return None
    item_key, observation_scope = _item_observation_identity(
        ctx, item, stage=stage, round_id=round_id)
    document_scope = observation_scope.parent
    if document_scope is None:
        raise RuntimeError("评测 episode scope 缺少 document parent")
    work_memory = ctx.work_memory
    if work_memory.active_session_scope is None:
        work_memory.begin_session(session_scope(
            ctx.space_id,
            owner=document_scope.owner,
            versions=document_scope.versions,
        ))
    work_memory.begin_document(document_scope)
    work_memory.begin_episode(observation_scope, round_id=round_id)
    raw = InputPayload(
        segments=segments,
        source=item.source,
        stage=STAGE_TRAINING,
        modality=item.modality,
        lang=item.lang,
        domain=item.domain,
        weaning_phase=ctx.weaning_phase,
        item_key=item_key,
        scope_identity=observation_scope,
        source_ref=item.source_ref,
        occurrence_scope_identity=_item_occurrence_scope(ctx, item),
        raw_text=item.raw_text,
        speaker_identity=item.speaker_identity,
    )
    try:
        from pure_integer_ai.cognition.understanding.pronoun_features import (
            lookup_pronoun_features,
        )
        sense_lookup, record_legacy_sense_counts = observe_sense_lookup(
            ctx, runtime_language=item.lang)
        observed = observe(
            raw,
            space_ctx,
            concept_index=ctx.concept_index,
            work_memory=work_memory,
            pronoun_feature_lookup=lookup_pronoun_features,
            sense_lookup=sense_lookup,
            record_legacy_sense_counts=record_legacy_sense_counts,
            word_form_providers=ctx.word_form_providers,
            occurrence_index=ctx.occurrence_index,
            occurrence_order_writer=ctx.occurrence_order_writer,
        )
        _materialize_item_spans(ctx, item, observed)
        _run_item_predictions(ctx, item, observed)
        _run_item_sense_candidates(ctx, item, observed)
        return observed
    except BaseException:
        work_memory.abort_episode()
        raise
    finally:
        if work_memory.active_episode_scope is not None:
            work_memory.end_episode()
        if work_memory.active_document_scope is not None:
            work_memory.end_document()


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
def _run_calibration_phase(ctx: TrainContext, corpus: list,
                           backend: StorageBackend) -> list[dict[str, int]]:
    """在评测沙箱运行 D5 校准，并只回传校准专属台账行。"""
    from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
    from pure_integer_ai.teacher.weaning_calibration import (
        CALIBRATION_TABLE,
        record_calibration,
    )

    with isolated_evaluation(ctx, label="calibration") as eval_ctx:
        before = eval_ctx.backend.select(CALIBRATION_TABLE)
        _run_calibration_phase_impl(eval_ctx, corpus, eval_ctx.backend)
        after = eval_ctx.backend.select(CALIBRATION_TABLE)
        new_rows = after[len(before):]
    for row in new_rows:
        record_calibration(
            backend,
            round_id=int(row["round_id"]),
            mode_a_pass=int(row["mode_a_pass"]),
            mode_b_pass=int(row["mode_b_pass"]),
        )
    return [dict(row) for row in new_rows]


def _run_calibration_phase_impl(ctx: TrainContext, corpus: list,
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
    for item_index, item in enumerate(cal_items):
        obs = _observe_eval_item(
            ctx,
            item,
            stage=STAGE4_PROMOTE_WEAN,
            round_id=10_000_000 + item_index,
            space_ctx=sctx,
        )
        if obs is None:
            continue
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
                                backend: StorageBackend) -> tuple[int, bool]:
    """在独立评测沙箱运行 E2，并只回传保持率和通过标记。"""
    from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation

    with isolated_evaluation(ctx, label="offline_eval") as eval_ctx:
        _run_simulated_offline_eval_impl(eval_ctx, corpus, eval_ctx.backend)
        result = (eval_ctx.holdout_retention, eval_ctx.e2_eval_passed)
    return result


def _run_simulated_offline_eval_impl(ctx: TrainContext, corpus: list,
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
      - probe 学树 root_a + root_b 参树只留在独立评测 backend，正式训练图保持 bit-identical
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

    for probe_index, item in enumerate(probe_items):
        # observe 探针建学树 root_a（eval 时首次 observe·probe 从未进训练 observe·D4 守·镜像 calibration :1248-1265）
        obs = _observe_eval_item(
            ctx,
            item,
            stage=STAGE4_PROMOTE_WEAN,
            round_id=20_000_000 + probe_index,
            space_ctx=sctx,
        )
        if obs is None:
            continue
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
    诊断/log）·skel_by_item_new = {(document_scope_hash, seg_idx): skeleton_ref}（scope B·S1 all_ops 搜结果·caller 写 work_memory.lang_skeleton_by_item）。
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
    root_keys: list[tuple[int, int]] = []   # scope B：(document_scope_hash, seg_idx) 与 observe 对齐
    # scope B（同训练 _discover_and_recognize_lang_structures·断奶 critical path ④）：gate COMPOSES_COMBINE_MODE ON→
    # 按已决句界建根（与 observe 同源·seg_idx 对齐）·OFF→整段单 span=原段级行为。
    _scope_b_split = bool(getattr(gates, "COMPOSES_COMBINE_MODE", False))
    _flat_units: list[tuple[CollectedItem, int, int, list[str]]] = []
    for item in lang_probe:
        item_key, _document_scope = _item_document_identity(ctx, item)
        _spans = _item_sentence_bounds(item) if _scope_b_split else [(0, len(item.tokens))]
        for seg_idx, (_s, _e) in enumerate(_spans):
            _toks = list(item.tokens[_s:_e])
            if _toks:
                _flat_units.append((item, item_key, seg_idx, _toks))
    for item, item_key, seg_idx, tokens in _flat_units:
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
        root_keys.append((item_key, seg_idx))

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
    """在独立评测沙箱测量 floor，禁止 held-out 写入正式图和 WorkMemory。"""
    from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation

    with isolated_evaluation(ctx, label="floor") as eval_ctx:
        return _measure_floor_pass_impl(
            eval_ctx, eval_ctx.backend, eval_ctx.concept_graph)


def _measure_floor_pass_impl(ctx: TrainContext, backend: StorageBackend,
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

    **隔离论证（核心）**：observe 只写独立评测 backend；正式 backend、身份缓存和 WorkMemory 在退出时核验
    bit-identical。held-out 路径仍禁写 D:11 tally，floor 只读从训练快照克隆来的 D:11。

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

    # 第 1 步：held-out 不计 tally 的发现，以及 S1 all_ops 映射（_held_out_discovery_tally_free）。
    _roots, skel_by_item_new = _held_out_discovery_tally_free(
        ctx, backend, lang_probe, training_lang_ops)
    # S1 fix：populate work_memory.lang_skeleton_by_item（mirror training·caller observe 前 fill·
    # observe.py:204-208 读 work_memory 此 map → build_instantiates_edge fire）。setdefault first-wins（同 training）。
    # scope B：键 = (document_scope_hash, seg_idx)（句级·observe 同键对齐）。
    for _key, _skel in skel_by_item_new.items():
        ctx.work_memory.lang_skeleton_by_item.setdefault(_key, _skel)

    # Step 2：held-out observe（只进入评测沙箱·post-training 惰性）。
    # observe fire build_instantiates_edge（:204-208·COMPOSES_COMBINE_MODE ON + lang_skeleton_by_item 填）→ __seg_→skeleton
    # INSTANTIATES 真边·observe 建 token_seq（attach_token_seq :285·observed input token·gate DISPATCH_TOKEN_CHAIN_MODE ON）。
    sctx = _build_space_ctx(ctx)
    held_out_struct_refs: list[ConceptRef] = []
    # scope B：roots 现为句级（len > lang_probe）·故按 item 迭代（observe 内部按句 segment 产 struct_ref·
    # 逐 segment 读 lang_skeleton_by_item[(item_key,seg_idx)]）。_roots 仅诊断用（不再 zip item）。
    for probe_index, item in enumerate(lang_probe):
        # S1 defensive surface（审1 严重 foot-gun）：observe 前核 map 已填·否则 INSTANTIATES 全不 fire → silent veto。
        # scope B：核该稳定 document scope 的任一句 seg 已进入 map。
        item_key, _document_scope = _item_document_identity(ctx, item)
        _nseg = len(_item_sentence_bounds(item))
        if not any((item_key, _si) in ctx.work_memory.lang_skeleton_by_item
                   for _si in range(_nseg)):
            import sys
            print(f"[floor_orchestrator] WARN S1: held-out item scope={item_key} "
                  f"无句 seg 进 lang_skeleton_by_item → INSTANTIATES 不 fire → read_instantiates=None "
                  f"(silent-veto·measured=False 风险)", file=sys.stderr)
        obs = _observe_eval_item(
            ctx,
            item,
            stage=STAGE4_PROMOTE_WEAN,
            round_id=30_000_000 + probe_index,
            space_ctx=sctx,
        )
        if obs is None:
            continue
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
def _h2_calibrate(ctx: TrainContext, corpus: list[CollectedItem],
                  runner: RoundRunner, *,
                  execution: FormalTrainExecutionStats | None = None) -> JudgeWeights:
    """在独立评测沙箱完成 H2，并只把标定后的权重带回正式训练。"""
    from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation

    with isolated_evaluation(ctx, label="h2") as eval_ctx:
        calibrated = _h2_calibrate_impl(
            eval_ctx, corpus, runner, execution=execution)
    # 无样本或标定结果未变时保留既有 no-op 对象身份契约。
    return ctx.weights if calibrated == ctx.weights else calibrated


def _h2_calibrate_impl(ctx: TrainContext, corpus: list[CollectedItem],
                       runner: RoundRunner, *,
                       execution: FormalTrainExecutionStats | None = None) -> JudgeWeights:
    """阶段3 H2：小批量离线标定 judge 权重（教师 GT 经录放层·§十四 H2）。

    小批量跑 observe + episode（默认权重）→ 收集 CalibrationSample → calibrate_weights。
    标定用录放层 ground-truth 零 LLM（MODE_REPLAY）·运行时判据自锚输入（两时相不同非矛盾）。
    """
    # 刀 A：H2 标定期间翻 TIME_SEQ_PROOF_MODE OFF（时序 verify 分流的语言项产空 output·judge G2p veto
    # reward=0 对齐 GT=1 不一致·污染 JudgeWeights 标定·对抗审 P1-1）。H2 是 judge 标定·时序 verify 绕 judge
    # 不参与标定·翻 OFF 让语言项正常 episode_loop。镜像 _is_verify_modality :1584 排除防御。
    # 复位在 :1615 return 前·外层 :1469 finally 兜底（异常时）。
    h2_gate_token = gates.push_gate_overrides({
        "TIME_SEQ_PROOF_MODE": False,
        "NUMERIC_PROOF_MODE": False,
        "UNIVERSAL_PROOF_MODE": False,
        "EXISTENTIAL_PROOF_MODE": False,
        "COMPARISON_PROOF_MODE": False,
    })
    # 刀 B：同上·数值 verify 分流亦绕 judge·H2 标定翻 NUMERIC_PROOF_MODE OFF（镜像时序·防标定污染）。
    # 刀 C：同上·全称 verify 分流亦绕 judge·H2 标定翻 UNIVERSAL_PROOF_MODE OFF（镜像时序/数值·防标定污染）。
    # A1·STEP6：同上·存在 verify 分流亦绕 judge·H2 标定翻 EXISTENTIAL_PROOF_MODE OFF（镜像 universal·防标定污染）。
    # 刀 D：同上·比较 verify 分流亦绕 judge·H2 标定翻 COMPARISON_PROOF_MODE OFF（镜像时序/数值/全称·防标定污染）。
    # G1+#774 PROPOSITION_MODE **不在此翻 OFF**（异 TIME_SEQ/NUMERIC/UNIVERSAL 三刀·对抗审2 发现 A）：
    # 三刀是 verify 路由分流（绕 judge·产空 output·G2p veto reward=0 对齐 GT=1 不一致污染 JudgeWeights 标定）·
    # 须翻 OFF 让语言项正常 episode_loop 进 judge 标定。PROPOSITION_MODE 不路由绕 judge——它只在 observe 建命题
    # 节点 + judge 内部激活 G3b hard veto 乘法门（不进 J 加权）·H2 标定**应让 G3b 参与 judge**（权重适应真实 judge
    # 行为·含 G3b veto）。教师 GT=1 + G3b veto reward=0 是教师与机制分歧·calibrate_weights 网格搜索时此类样本
    # agreement=False·与 J 权重无关·属噪声样本不通过调 J 修复。故 PROPOSITION_MODE 保持 ON（生产态）标定。
    batch = corpus[:H2_CALIB_BATCH]
    samples: list[CalibrationSample] = []
    try:
        for rid, item in enumerate(batch):
            # verify-driven COMPOSES 模态（code/arith）排除 H2 标定——reward 经 vm_proof_fn 不用 JudgeWeights·
            # 且进 language judge()→G2p veto reward=0 对齐 GT=1=垃圾标定污染 JudgeWeights（doc §九·_is_verify_modality）。
            if _is_verify_modality(item.modality):
                continue
            if execution is not None:
                execution.h2_item_runs += 1
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
    finally:
        gates.reset_gate_overrides(h2_gate_token)
    if not samples:
        return ctx.weights
    judge_fn, teacher_gt = _make_calib_judge_fn(ctx.teacher, ctx.weaning_phase)
    return calibrate_weights(samples, judge_fn, teacher_gt)
# ---- 阶段4 promote ----

__all__ = [
    "CalibrationSample",
    "EvaluationProbeEvaluator",
    "_held_out_discovery_tally_free",
    "_h2_calibrate",
    "_h2_calibrate_impl",
    "_make_calib_judge_fn",
    "_measure_floor_pass",
    "_measure_floor_pass_impl",
    "_observe_eval_item",
    "_run_calibration_phase",
    "_run_calibration_phase_impl",
    "_run_simulated_offline_eval",
    "_run_simulated_offline_eval_impl",
    "run_evaluation_plan",
]
