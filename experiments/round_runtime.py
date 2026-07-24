"""单轮训练的观察、验证、路径执行和 reward runtime。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pure_integer_ai.cognition.process.abstraction import (
    build_isa_ancestor_map_external,
)
from pure_integer_ai.cognition.process.episode import episode_loop, _ctx_tag
from pure_integer_ai.cognition.process.structure_discover import (
    load_discovered_operators,
)
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.shared.edge_types import (
    EDGE_CAUSES,
    EDGE_PRECEDES,
    EDGE_T_STEP,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.types import (
    ConceptRef,
    Episode,
    InputPayload,
    IntentType,
    MODALITY_ARITH,
    MODALITY_CODE,
    MODALITY_LANGUAGE,
    INTENT_COMMAND,
    INTENT_QUESTION,
    OutputResult,
    PathData,
    PathResult,
    STAGE_TRAINING,
    TERMINAL_REACHED_SINK,
    VERIFY_SOURCE_EXTERNAL,
    VERIFY_SOURCE_SELF_PRODUCED,
    WEANING_PRE,
)
from pure_integer_ai.cognition.understanding.observe import observe
from pure_integer_ai.config import gates
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.language_observation import (
    _materialize_item_spans,
    _prepare_item_boundary,
    _run_item_predictions,
    _run_item_semantic_course,
    _run_item_sense_candidates,
    _split_item_to_segments,
)
from pure_integer_ai.experiments.language_sense_candidate_runtime import (
    observe_sense_lookup,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    _item_document_identity,
    _item_observation_identity,
    _item_occurrence_scope,
)
from pure_integer_ai.experiments.train_execution import item_candidate_counts
from pure_integer_ai.experiments.verification_dispatch import (
    VERIFY_ROUTE_COMPARISON,
    VERIFY_ROUTE_COMPOSES,
    VERIFY_ROUTE_EXISTENTIAL,
    VERIFY_ROUTE_NUMERIC,
    VERIFY_ROUTE_OCCURRENCE_ORDER,
    VERIFY_ROUTE_UNIVERSAL,
    select_verification_routes,
    verification_dimension_key,
    verification_verifier_key,
)
from pure_integer_ai.experiments.verification_orchestration import (
    MultiVerifierOrchestrator,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationEvaluation,
    VerificationReport,
    VerifierRegistration,
)
from pure_integer_ai.storage.experience_count import pack_ctx_code
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.telemetry import (
    active_backend_telemetry,
    record_candidate_count,
    telemetry_scope,
)
from pure_integer_ai.training.comparison_proof import (
    comparison_proof_fn_factory,
)
from pure_integer_ai.training.existential_proof import (
    existential_proof_fn_factory,
)
from pure_integer_ai.training.mode_b_cross_verify import cross_verify_pair
from pure_integer_ai.training.numeric_proof import numeric_proof_fn_factory
from pure_integer_ai.training.stages import STAGE3_REWARD, build_judge_fn
from pure_integer_ai.training.time_seq_proof import time_seq_proof_fn_factory
from pure_integer_ai.training.universal_proof import universal_proof_fn_factory
from pure_integer_ai.training.vm_proof import vm_proof_fn_factory

# 目标达成覆盖率阈值；保持在 round runtime 的路径调度职责内。
COVERAGE_THRESHOLD = 500

@runtime_checkable
class RoundRunner(Protocol):
    """per-round 认知执行协议（observe + episode·可换预处理/检索）。

    run_round(ctx, item, stage, round_id) -> Episode | None
      observe-only 阶段（stage1/2）可返 None（无 episode·只建图）。
      reward 阶段（stage3+）返 Episode（供 metrics/防塌/收敛消费）。
    """

    def run_round(self, ctx: TrainContext, item: CollectedItem,
                  stage: int, round_id: int
                  ) -> Episode | TypedLanguageEpisode | None: ...


@dataclass
class RoundResult:
    """per-round 产出（episode + 重建 path/output 供 H2 标定·确定性重算 bit-identical）。"""

    episode: Episode | None = None
    output: Any = None
    dag_path: Any = None
    verification_report: VerificationReport | None = None
    typed_episode: TypedLanguageEpisode | None = None

    def episodes(self) -> tuple[Episode | TypedLanguageEpisode, ...]:
        """返回 legacy/typed 或分维 episode，不把不同 reward 协议标量合并。"""
        if self.typed_episode is not None:
            if self.episode is not None:
                raise ValueError("RoundResult 不得混合 typed 与 legacy 主 episode")
            return (self.typed_episode,)
        if self.verification_report is None:
            result = []
            if self.episode is not None:
                result.append(self.episode)
            return tuple(result)
        episodes = []
        for result in self.verification_report.applicable_results():
            artifact = result.artifact
            if isinstance(artifact, RoundResult):
                episodes.extend(artifact.episodes())
        return tuple(episodes)


class MultipleVerificationEpisodesError(RuntimeError):
    """旧单 episode 接口遇到多维结果，要求调用方改用 run_round_many。"""


def _uses_typed_language_pipeline(
        ctx: TrainContext,
        item: CollectedItem,
        ) -> bool:
    """判断当前语言项是否已进入 S-02 或 typed generation 管线。"""
    if item.modality != MODALITY_LANGUAGE:
        return False
    return (
        ctx.language_semantic_course_runtime is not None
        or ctx.language_generation_runtime is not None
    )


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
                  stage: int, round_id: int
                  ) -> Episode | TypedLanguageEpisode | None:
        episodes = self.run_round_many(ctx, item, stage, round_id)
        if len(episodes) > 1:
            raise MultipleVerificationEpisodesError(
                "多维 verifier 结果不能压成旧单 episode 接口")
        return None if not episodes else episodes[0]

    def run_round_many(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            stage: int,
            round_id: int,
            ) -> tuple[Episode | TypedLanguageEpisode, ...]:
        """执行一轮并返回全部独立 episode，不跨协议合成 reward。"""
        return self.run_round_full(ctx, item, stage, round_id).episodes()

    def run_round_full(self, ctx: TrainContext, item: CollectedItem,
                       stage: int, round_id: int) -> RoundResult:
        """在统一 session/document/episode 边界内执行一轮训练。"""
        assert_int(stage, round_id, _where="DefaultRoundRunner.run_round_full")
        if (stage >= STAGE3_REWARD
                and _uses_typed_language_pipeline(ctx, item)
                and ctx.language_generation_runtime is None):
            raise RuntimeError(
                "S-02 语义课程已启用，但 reward stage 缺少 typed generation owner")
        if (ctx.boundary_hypothesis_engine is not None
                and item.boundary_parse is None):
            _prepare_item_boundary(
                ctx,
                item,
                commit_evidence=True,
                persist_graph=True,
            )
        item_key, document_scope = _item_document_identity(ctx, item)
        episode_local_id = Hasher("formal_train.episode_scope.v1").h63(
            (stage, round_id))
        if episode_local_id == 0:
            episode_local_id = 1
        observation_scope = episode_scope(episode_local_id, parent=document_scope)
        ctx.scoped_identity_store.register_scope(observation_scope)

        work_memory = ctx.work_memory
        if work_memory.active_session_scope is None:
            session_identity = session_scope(
                ctx.space_id,
                owner=document_scope.owner,
                versions=document_scope.versions,
            )
            work_memory.begin_session(session_identity)
        work_memory.begin_document(document_scope)
        work_memory.begin_episode(observation_scope, round_id=round_id)
        try:
            return self._run_round_full_unscoped(ctx, item, stage, round_id)
        except BaseException:
            work_memory.abort_episode()
            raise
        finally:
            # 正常路径由 episode_loop 关闭 query；异常路径由 abort_episode 幂等清理。
            if work_memory.active_episode_scope is not None:
                work_memory.end_episode()
            if work_memory.active_document_scope is not None:
                work_memory.end_document()

    def _run_round_full_unscoped(self, ctx: TrainContext, item: CollectedItem,
                                 stage: int, round_id: int) -> RoundResult:
        """执行已由外层生命周期包住的旧 round 逻辑。"""
        assert_int(stage, round_id, _where="DefaultRoundRunner.run_round_full")
        # 多段 observe：CollectedItem 段落 → 句段 Segment 列表·串 struct_ref 链（设计原意·破致命6）
        # 刀5 件8：透传 ctx 4 参 → cue_extractor → cue_type_of 第二源 D:11 readback（生产路径·close 刀4 gap）
        segments = _split_item_to_segments(
            item, backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index)
        if not segments:
            return RoundResult()
        if ctx.unicode_intake is not None:
            ctx.unicode_intake.observe_segments(segments)
        item_key, observation_scope = _item_observation_identity(
            ctx, item, stage=stage, round_id=round_id)
        raw = InputPayload(
            segments=segments, source=item.source, stage=STAGE_TRAINING,
            modality=item.modality, lang=item.lang, domain=item.domain,
            weaning_phase=ctx.weaning_phase,
            item_key=item_key,
            scope_identity=observation_scope,
            source_ref=item.source_ref,
            occurrence_scope_identity=_item_occurrence_scope(ctx, item),
            raw_text=item.raw_text,
            speaker_identity=item.speaker_identity,
        )
        # observe 建图（卷一 observe 总控·三空间分流·回传 struct_refs）
        sctx = _build_space_ctx(ctx)
        # B5：pronoun_feature_lookup 注入（元定义出厂硬件·性质B 软兜·防 PR 软排序把"他"指向"苹果"非人称）
        from pure_integer_ai.cognition.understanding.pronoun_features import lookup_pronoun_features
        sense_lookup, record_legacy_sense_counts = observe_sense_lookup(
            ctx, runtime_language=item.lang)
        obs = observe(raw, sctx, concept_index=ctx.concept_index,
                      work_memory=ctx.work_memory,
                      pronoun_feature_lookup=lookup_pronoun_features,
                      sense_lookup=sense_lookup,
                      record_legacy_sense_counts=record_legacy_sense_counts,
                      word_form_providers=ctx.word_form_providers,
                      occurrence_index=ctx.occurrence_index,
                      occurrence_order_writer=ctx.occurrence_order_writer,
                      position_histogram_state=ctx.position_histogram_state,
                      hub_degree_state=ctx.hub_degree_state,
                      write_legacy_language_sequences=(
                          not _uses_typed_language_pipeline(ctx, item)))
        _materialize_item_spans(ctx, item, obs)
        _run_item_predictions(ctx, item, obs)
        _run_item_sense_candidates(ctx, item, obs)
        _run_item_semantic_course(ctx, item, raw, obs)
        if ctx.precedence_relation_runtime is not None:
            if raw.occurrence_scope_identity is None:
                raise ValueError("R-06 runtime 缺少来源 occurrence scope")
            precedence_report = ctx.precedence_relation_runtime.process(
                raw.occurrence_scope_identity,
                read_only=ctx.scope_owner is not None,
            )
            ctx.precedence_relation_reports.append(precedence_report)
        if ctx.causal_relation_runtime is not None:
            if raw.occurrence_scope_identity is None:
                raise ValueError("R-07 runtime 缺少来源 occurrence scope")
            causal_report = ctx.causal_relation_runtime.process(
                raw.occurrence_scope_identity,
                read_only=ctx.scope_owner is not None,
            )
            ctx.causal_relation_reports.append(causal_report)
        if ctx.set_relation_runtime is not None:
            if raw.occurrence_scope_identity is None:
                raise ValueError("R-02 runtime 缺少来源 occurrence scope")
            set_relation_report = ctx.set_relation_runtime.process(
                raw.occurrence_scope_identity,
                read_only=ctx.scope_owner is not None,
            )
            ctx.set_relation_reports.append(set_relation_report)
        if ctx.property_relation_runtime is not None:
            if raw.occurrence_scope_identity is None:
                raise ValueError("R-03 runtime 缺少来源 occurrence scope")
            property_relation_report = ctx.property_relation_runtime.process(
                raw.occurrence_scope_identity,
                read_only=ctx.scope_owner is not None,
            )
            ctx.property_relation_reports.append(property_relation_report)
        if ctx.mereology_relation_runtime is not None:
            if raw.occurrence_scope_identity is None:
                raise ValueError("R-04 runtime 缺少来源 occurrence scope")
            mereology_relation_report = ctx.mereology_relation_runtime.process(
                raw.occurrence_scope_identity,
                read_only=ctx.scope_owner is not None,
            )
            ctx.mereology_relation_reports.append(mereology_relation_report)
        if ctx.semantic_pair_runtime is not None:
            if raw.occurrence_scope_identity is None:
                raise ValueError("R-05 runtime 缺少来源 occurrence scope")
            semantic_pair_report = ctx.semantic_pair_runtime.process(
                raw.occurrence_scope_identity,
                read_only=ctx.scope_owner is not None,
            )
            ctx.semantic_pair_reports.append(semantic_pair_report)

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

        supplemental_verification = None
        verification_routes = select_verification_routes(item, segments)
        if verification_routes:
            if (item.modality == MODALITY_LANGUAGE
                    and ctx.language_generation_runtime is not None):
                verification = self._run_verification_routes(
                    ctx,
                    item,
                    raw,
                    obs,
                    round_id,
                    verification_routes,
                )
                supplemental_verification = verification.verification_report
                if supplemental_verification is None:
                    raise RuntimeError("typed language verifier 未返回分维报告")
            else:
                return self._run_verification_routes(
                    ctx,
                    item,
                    raw,
                    obs,
                    round_id,
                    verification_routes,
                )
        if (item.modality == MODALITY_LANGUAGE
                and ctx.language_generation_runtime is not None):
            production = ctx.language_generation_runtime.run(
                ctx,
                item,
                raw,
                obs,
            )
            if raw.source_ref is None or raw.scope_identity is None:
                raise ValueError("typed language episode 缺来源或 episode scope")
            typed_episode = TypedLanguageEpisode.from_production(
                round_id,
                raw.source_ref,
                raw.scope_identity,
                production,
                read_only=ctx.scope_owner is not None,
                supplemental_verification=supplemental_verification,
            )
            return RoundResult(
                output=production,
                verification_report=supplemental_verification,
                typed_episode=typed_episode,
            )
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
        reward_gate_token = gates.push_gate_overrides({
            "ATTRACTOR_MODE": True,
            "EXPLORATION_MODE": True,
            "SELECTION_PREF_DOCK_MODE": True,
            "SELECTION_PREF_FEED_MODE": True,
            "GENERATE_SELECTION_PREF_MODE": True,
            "FREQ_OBSERVE_MODE": True,
            "SP_OBSERVE_MODE": True,
        })
        # S4 三乘子进 PR：selection_pref 维 dock PR seed（_seed_weight 乘积·attractor 扩张路径 token seed 真生效）
        # + sp_sn reward feed 第三条腿（reward_propagate 落点⑥·concept_targets 配对 feed）。同 ATTRACTOR try/finally 守回归。
        # S4 后续加固·项1：生成侧 selection_pref pair-rate 精查接线（GENERATE_SELECTION_PREF_MODE 生产 ON）。
        # observe+reward 写 selection_pref_count·生成侧 slot_dispatch:105 读 selection_pref_score·gate 不接则写了不读=theater。
        # 同 ATTRACTOR try/finally 守回归（CI gate OFF 零翻·生产 ON 生成侧真活）。
        # 方案3 tn路（B4 β_arith 修法）：FREQ_OBSERVE_MODE 生产 ON（dag_path add_active + attractor add_seed 写
        # observe_tn·read_effective_freq observe_mode=True 读 base+observe_tn·解 β_arith rate 塌缩 w_freq 塌缩）。
        # 同 ATTRACTOR try/finally 守回归（CI gate OFF 零翻·生产 ON observe_tn 真写·否则 = theater）。
        # 方案3 tn路（B5 β_arith 修法）：SP_OBSERVE_MODE 生产 ON（selection_pref 维 consumer 读 sp_observe_tn 替 sp_tn·
        # 解 β_arith rate 塌缩 w_sp 塌缩·sp_observe_tn 由 record_selection_pref_cooccur 写·SELECTION_PREF_MODE 守写）。
        # 同 FREQ_OBSERVE_MODE try/finally 守回归（CI gate OFF 零翻·生产 ON sp_observe_tn 真读·否则 = theater）。
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
            gates.reset_gate_overrides(reward_gate_token)
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

    def _verification_claim_keys(
            self,
            route: int,
            item: CollectedItem,
            segments: list[Segment],
            ) -> tuple[tuple[int, ...], ...]:
        """保存 legacy 声明的完整整数内容及其段内位置。"""
        if route == VERIFY_ROUTE_COMPOSES:
            specs = (
                item.code_specs
                if item.modality == MODALITY_CODE
                else item.arith_specs
            )
            return tuple((route, index) for index, _spec in enumerate(specs))
        attribute_by_route = {
            VERIFY_ROUTE_NUMERIC: "numeric_claims",
            VERIFY_ROUTE_COMPARISON: "comparison_claims",
            VERIFY_ROUTE_UNIVERSAL: "universal_claims",
            VERIFY_ROUTE_EXISTENTIAL: "existential_claims",
            VERIFY_ROUTE_OCCURRENCE_ORDER: "precedes_pairs",
        }
        attribute = attribute_by_route.get(route)
        if attribute is None:
            raise ValueError("未知 legacy verification route")
        keys = []
        for segment_index, segment in enumerate(segments):
            claims = getattr(segment, attribute)
            for claim_index, claim in enumerate(claims):
                claim_tuple = tuple(claim)
                assert_int(*claim_tuple, _where="verification claim")
                keys.append((
                    route,
                    segment_index,
                    claim_index,
                    len(claim_tuple),
                    *claim_tuple,
                ))
        return tuple(keys)

    def _run_verification_route(
            self,
            route: int,
            ctx: TrainContext,
            item: CollectedItem,
            raw: InputPayload,
            obs: Any,
            round_id: int,
            ) -> RoundResult:
        """调用一个 legacy adapter；分派只选实现，不决定优先级。"""
        if route == VERIFY_ROUTE_COMPOSES:
            return self._run_verify_round(ctx, item, raw, obs, round_id)
        if route == VERIFY_ROUTE_NUMERIC:
            return self._run_numeric_verify_round(
                ctx, item, raw, obs, round_id)
        if route == VERIFY_ROUTE_COMPARISON:
            return self._run_comparison_verify_round(
                ctx, item, raw, obs, round_id)
        if route == VERIFY_ROUTE_UNIVERSAL:
            return self._run_universal_verify_round(
                ctx, item, raw, obs, round_id)
        if route == VERIFY_ROUTE_EXISTENTIAL:
            return self._run_existential_verify_round(
                ctx, item, raw, obs, round_id)
        if route == VERIFY_ROUTE_OCCURRENCE_ORDER:
            return self._run_occurrence_order_verify_round(
                ctx, item, raw, obs, round_id)
        raise ValueError("未知 legacy verification route")

    def _verification_evaluation(
            self,
            route: int,
            ctx: TrainContext,
            item: CollectedItem,
            raw: InputPayload,
            obs: Any,
            round_id: int,
            claim_keys: tuple[tuple[int, ...], ...],
            ) -> VerificationEvaluation:
        """把一个 legacy RoundResult 转为不含综合标量的分维结果。"""
        artifact = self._run_verification_route(
            route,
            ctx,
            item,
            raw,
            obs,
            round_id,
        )
        if artifact.episode is None:
            verdict = VERDICT_UNKNOWN
            detail = (len(claim_keys),)
        elif artifact.episode.reward == 1:
            verdict = VERDICT_SUPPORT
            detail = (len(claim_keys), 1)
        elif artifact.episode.reward == 0:
            verdict = VERDICT_REFUTE
            detail = (len(claim_keys), 0)
        else:
            verdict = VERDICT_UNKNOWN
            detail = (len(claim_keys), artifact.episode.reward)
        return VerificationEvaluation(
            verdict=verdict,
            claim_keys=claim_keys,
            detail=detail,
            source=raw.source_ref,
            scope=raw.scope_identity,
            artifact=artifact,
        )

    def _run_verification_routes(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            raw: InputPayload,
            obs: Any,
            round_id: int,
            routes: tuple[int, ...],
            ) -> RoundResult:
        """执行全部适用 legacy adapter，并保存顺序无关的分维报告。"""
        registrations = []
        multiple = len(routes) > 1
        for route in routes:
            claim_keys = self._verification_claim_keys(
                route, item, raw.segments)
            route_round_id = round_id
            if multiple:
                route_round_id = Hasher(
                    "verification.episode.v1",
                ).h63((
                    round_id,
                    *verification_dimension_key(route).stable_key(),
                ))
                if route_round_id == 0:
                    route_round_id = 1
            registrations.append(VerifierRegistration(
                dimension=verification_dimension_key(route),
                verifier=verification_verifier_key(route),
                applies=lambda _request: True,
                evaluate=lambda _request, selected_route=route,
                        selected_round_id=route_round_id,
                        selected_claim_keys=claim_keys:
                    self._verification_evaluation(
                        selected_route,
                        ctx,
                        item,
                        raw,
                        obs,
                        selected_round_id,
                        selected_claim_keys,
                    ),
            ))
        report = MultiVerifierOrchestrator().run(
            None,
            tuple(registrations),
            read_only=ctx.scope_owner is not None,
        )
        ctx.verification_reports.append(report.detached())
        applicable = report.applicable_results()
        if len(applicable) == 1:
            artifact = applicable[0].artifact
            if isinstance(artifact, RoundResult):
                return RoundResult(
                    episode=artifact.episode,
                    output=artifact.output,
                    dag_path=artifact.dag_path,
                    verification_report=report,
                )
        return RoundResult(verification_report=report)

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

    def _run_occurrence_order_verify_round(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            raw: InputPayload,
            obs: Any,
            round_id: int,
            ) -> RoundResult:
        """核验当前样本 occurrence cue 顺序图无环，不解释事件时间或因果。

        cue 对先解析到来源化 occurrence 端点，再交给既有整数 Kahn 检查。
        本 adapter 不读取事件时间事实、不写 PRECEDES/CAUSES，也不把无环提升为
        scope 外的语义正确；事件时间与因果必须由独立 typed verifier 接入。
        """
        assert_int(
            round_id,
            _where="_run_occurrence_order_verify_round.round_id",
        )
        segments = raw.segments
        struct_refs = obs.struct_refs
        if not struct_refs:
            return RoundResult()   # observe 未建 struct_ref·诚实跳过（不伪造 reward）
        root = struct_refs[0]
        segment_occurrences = obs.segment_occurrence_refs
        if len(segment_occurrences) != len(segments):
            return RoundResult()
        if (ctx.occurrence_order_reader is None
                or raw.occurrence_scope_identity is None):
            return RoundResult()
        occurrence_chain = ctx.occurrence_order_reader.read_chain(
            raw.occurrence_scope_identity)
        observed_occurrences = tuple(obs.occurrence_refs)
        if occurrence_chain.occurrences != observed_occurrences:
            from pure_integer_ai.cognition.understanding.occurrence_order import (
                OccurrenceOrderIntegrityError,
            )
            raise OccurrenceOrderIntegrityError(
                "occurrence-order verifier 端点与来源顺序事实不一致")
        # 解析顺序 cue 对到 occurrence 图端点；未物化、越界和自环不形成声明。
        cue_pair_edges: list = []
        for seg, occurrence_refs in zip(segments, segment_occurrences):
            if not seg.precedes_pairs:
                continue
            if len(occurrence_refs) != len(seg.tokens):
                continue
            for (i, j) in seg.precedes_pairs:
                if i >= len(seg.tokens) or j >= len(seg.tokens):
                    continue
                a = occurrence_refs[i].node_ref()
                b = occurrence_refs[j].node_ref()
                if a == b:
                    continue
                cue_pair_edges.append((a, b))
        if not cue_pair_edges:
            return RoundResult()
        # 当前只验 cue 图；事件时间必须走独立维度，绝不回退旧 PRECEDES。
        fn = time_seq_proof_fn_factory(
            cue_pair_edges=cue_pair_edges,
            event_time_edges=[],
        )
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
            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # Layer0 外部锚门：cue 对 single-source·构造性检查·非验证·全自产不准停
        )
        return RoundResult(episode=ep, output=output, dag_path=dag_path)

    def _run_time_seq_verify_round(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            raw: InputPayload,
            obs: Any,
            round_id: int,
            ) -> RoundResult:
        """兼容旧调用名，权威语义仅为 occurrence cue 顺序核验。"""
        return self._run_occurrence_order_verify_round(
            ctx, item, raw, obs, round_id)

    def _run_numeric_verify_round(self, ctx: TrainContext, item: CollectedItem,
                                  raw: InputPayload, obs: Any,
                                  round_id: int) -> RoundResult:
        """刀 B：语言域数值等式 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像 occurrence-order adapter）。

        验 **数值等式声明的算术一致**（构造性检查·直接整数算术）·非 PRECEDES DAG·非 COMPOSES 执行。
        语言域 G5=DEAD_DESIGN → 走独立 episode 绕 judge（镜像 occurrence-order adapter / _run_verify_round）·非挂 G5。
        reward=1 iff 全数值声明算术一致·不落 strength（verify propagate no-op·镜像 occurrence-order adapter）·
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
        # 1. 收集数值声明（已解析纯整数 4-tuple·self-contained·无需 resolve）。
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
        output = OutputResult()   # 数值 verify 无词生成（语言域 episode 不 generate）。
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

        **三值诚实逻辑**（开放世界）：
          r=1（全 verified·ConceptNet 确认 child⊆parent）→ reward=1·verify_source=EXTERNAL·产 episode
          r=0（任一声明有独立显式反证）→ reward=0·verify_source=EXTERNAL·产 episode
          r=None（缺路径或证据冲突）→ **弃权·返空 RoundResult 无 episode**

        当前 caller 尚未接入 typed 反例 adapter，因此只能产生正证或未知，不能把 ConceptNet 缺边当证伪。

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
        # 3. factory 闭包 → fn；缺路径保持未知，显式反证 adapter 尚未接线。
        fn = universal_proof_fn_factory(ancestor_map=ext_map, claims=resolved_claims)
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 全称 verify 无词生成（语言域 episode 不 generate·镜像 time_seq:629/numeric:701）
        r = fn(output, dag_path, ctx.concept_graph)
        if r is None:
            return RoundResult()   # 缺路径或证据冲突 → 弃权，无 episode。
        reward = 1 if r == 1 else 0   # 0 仅保留给未来注入的独立显式反证。
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
        """验证存在量化声明，并在证据不足时返回空结果。

        正证需要共同 MEMBER 见证、显式 overlap，或已知非空子类同时包含于两侧；反证需要显式
        DISJOINT。当前运行入口只接入外部 SUBSET_EQ 祖先图，尚无上述 typed Evidence adapter，
        因此普通声明必须返回未知且不能形成 EXTERNAL episode。保留此入口用于后续接入，不把缺边
        或单纯双向子集路径误当世界事实。
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
        # 3. 当前只传子集图；未注入非空、overlap 或 DISJOINT 时 proof 必须返回未知。
        fn = existential_proof_fn_factory(ancestor_map=ext_map, claims=resolved_claims)
        dag_path = PathResult(
            path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root,
            topo_layers=[], convergence={}, source=None,
        )
        output = OutputResult()   # 存在 verify 无词生成（语言域 episode 不 generate·镜像 universal:903）
        r = fn(output, dag_path, ctx.concept_graph)
        if r is None:
            return RoundResult()   # typed 存在证据不足或冲突 → 弃权，无 episode。
        reward = 1 if r == 1 else 0   # 仅未来 Evidence adapter 注入后可达。
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
        space_id=ctx.space_id, excluded_word_refs=excluded,
        hub_degree_state=ctx.hub_degree_state)
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


def _run_runner_episodes(
        ctx: TrainContext,
        runner: RoundRunner,
        item: CollectedItem,
        stage: int,
        round_id: int,
        ) -> tuple[Episode | TypedLanguageEpisode, ...]:
    """优先使用多 episode 协议，并兼容尚未迁移的外部 runner。"""
    run_many = getattr(runner, "run_round_many", None)
    if callable(run_many):
        result = tuple(run_many(ctx, item, stage, round_id))
        if any(not isinstance(episode, (Episode, TypedLanguageEpisode))
               for episode in result):
            raise TypeError("run_round_many 返回了未注册 episode 协议")
        return result
    episode = runner.run_round(ctx, item, stage, round_id)
    return () if episode is None else (episode,)


def _run_round_batch(ctx: TrainContext, runner: RoundRunner,
                     items: list[CollectedItem], stage: int,
                     round_id: int) -> list[Episode | TypedLanguageEpisode]:
    """按稳定 item 顺序执行一批 round，并只收集实际产生的 episode。"""
    episodes: list[Episode | TypedLanguageEpisode] = []
    for item_index, item in enumerate(items):
        item_round_id = round_id * 1000 + item_index
        if active_backend_telemetry() is None:
            item_episodes = _run_runner_episodes(
                ctx, runner, item, stage, item_round_id)
        else:
            source_key = (
                None if item.source_ref is None
                else item.source_ref.stable_key())
            scope_key = (
                None if item.source_ref is None
                else document_scope(item.source_ref).stable_key())
            with telemetry_scope(
                    query="round_item",
                    source_key=source_key,
                    scope_key=scope_key,
                    stage=stage,
                    round_id=item_round_id,
                    item_index=item_index):
                for kind, count in item_candidate_counts(item).items():
                    record_candidate_count(kind, count)
                item_episodes = _run_runner_episodes(
                    ctx, runner, item, stage, item_round_id)
        episodes.extend(item_episodes)
    return episodes


__all__ = [
    "COVERAGE_THRESHOLD",
    "DefaultRoundRunner",
    "MultipleVerificationEpisodesError",
    "RoundResult",
    "RoundRunner",
    "_build_space_ctx",
    "_collect_action_seed_candidates",
    "_feed_action_experience",
    "_hotzone_dag_edges",
    "_reachable_dag_edges",
    "_rebuild_path",
    "_resolve_emergent_excluded_refs",
    "_run_emergence_hook",
    "_run_runner_episodes",
    "_run_round_batch",
]
