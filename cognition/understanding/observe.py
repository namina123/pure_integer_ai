"""cognition.understanding.observe — 模块1 observe 流水线总控（6Q 调度 + 落点分流）。

observe(raw_input, ctx) -> ObserveResult。段级循环·每段：
  ① parse_segment 按 modality 分发（语言首版实·非语言骨架 defer·I1 §7.4.2）
  ② normalize_to_concept 每 token 归一（模块3·性质A/B 分发）
  ③ build_precedes（模块2·PRECEDES strength=1）+ attach_role_seq
  ④ build_causes（模块2-bis·独立 CONDITION 守§8.1c）+ build_condition
  ⑤ build_cooccurs（模块6·分桶 SHADOW）
  ⑥ build_spatial_adj（模块2-ter·空间模态·语言 only 时空骨架）
  ⑦ build_refers_stable（模块4·性质A）/ build_refers_occurrence（模块5·性质B）
  ⑧ route_to_space（模块8·按 stage 落点）

铁律：纯整数（audit_float==0）/ 无墙钟（时序用 order_index·time_attach 用 timestamp_seq）/
  append-only / 确定性 bit-identical（确定性 hasher + 排序·禁随机）。

**observer-source 不变量（审1 C1 / 审2 F-2·v2 非循环心脏）**：observe 永不接 source=SOURCE_CONCEPTNET
  输入——ConceptNet oracle 须 bypass observe（boot 直注 build_isa/causes edges / labeler 直建 REALIZES edge）。
  observe 建 CONCEPTNET-source 边会过 REALIZES `_has_external_*` filter·破 v2 非循环（exemplar 集染自派生）。
  observe 入口 assert raw.source != SOURCE_CONCEPTNET 守此不变量（future-regression guard·当前恒满足）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage.edge_store import EdgeStore, EPI_STRUCTURED, EPI_CUE, SOURCE_CONCEPTNET
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.cognition.shared.types import (
    InputPayload, Segment, ObserveResult, SpaceContext, ConceptRef, MultiRef,
    STAGE_EXTERNAL_DEFINE, STAGE_TRAINING,
    MODALITY_LANGUAGE, MODALITY_AUDIO, MODALITY_ANIMATION,
    MODALITY_2D, MODALITY_3D, MODALITY_CODE, MODALITY_ARITH,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.hub_detect import HubDegreeState
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.role_precedes import (
    build_precedes_edges, build_inter_segment_precedes, build_struct_anchor,
    attach_role_seq, attach_token_seq,
)
from pure_integer_ai.cognition.understanding.emergent_role import (
    PositionHistogramState, register_position_hist,
)
from pure_integer_ai.cognition.understanding.causes import build_causes_edges
from pure_integer_ai.cognition.understanding.is_a import build_is_a_edges, has_reverse_isa_edge
from pure_integer_ai.cognition.understanding.instantiates import build_instantiates_edge
from pure_integer_ai.cognition.understanding.property import build_property_edges
from pure_integer_ai.cognition.understanding.similar import build_similar_edges
from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives, REL_PROPERTY
from pure_integer_ai.cognition.understanding.spatial_adj import build_spatial_adj
from pure_integer_ai.cognition.understanding.refers_to import normalize_to_concept
from pure_integer_ai.cognition.understanding.refers_stable import build_refers_stable_edge
from pure_integer_ai.cognition.understanding.refers_occurrence import resolve_pronoun_occurrence
from pure_integer_ai.cognition.understanding.cooccurs import build_cooccurs
from pure_integer_ai.cognition.understanding.selection_pref import build_selection_pref_count
from pure_integer_ai.cognition.process.abstraction import (
    apply_isa_edge_to_map, build_isa_ancestor_map_with_index)
from pure_integer_ai.config import gates
from pure_integer_ai.storage.sense_candidates import record_sense_token_seen, sense_surface_hash
from pure_integer_ai.storage.pronoun_resolution_count import register_pronoun_resolution_count
from pure_integer_ai.cognition.understanding.space_routing import target_space_id, route_to_space
from pure_integer_ai.cognition.understanding.refers_to import is_pronoun
from pure_integer_ai.cognition.understanding.cue_words import is_property_attr_marker
from pure_integer_ai.cognition.understanding.modification_direction import (
    observe_modification, register_modification_hist,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_OBSERVATION,
    SCOPE_DOCUMENT,
    LogicalClock,
    LogicalClockIdentity,
    LogicalTimestamp,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.work_memory import WorkMemoryScopeError


class ObservePipeline:
    """observe 流水线（持 backend/edge_store/concept_index/work_memory + 外部资源 hook）。

    外部资源（lemmatizer/sense_lookup/pronoun_feature_lookup）注入式·默认 None→自足 fallback。
    timestamp_seq：audit_event 自增序（性质B occurrence time_attach 用·无墙钟）。
    """

    def __init__(self, ctx: SpaceContext, *,
                 concept_index: ConceptIndex | None = None,
                 work_memory: WorkMemory | None = None,
                 lemmatizer=None, sense_lookup=None,
                 record_legacy_sense_counts: bool = True,
                 pronoun_feature_lookup=None,
                 word_form_providers=None,
                 occurrence_index=None,
                 source_intake=None,
                 occurrence_order_writer=None,
                 position_histogram_state=None,
                 hub_degree_state=None,
                 write_legacy_language_sequences: bool = True) -> None:
        self.ctx = ctx
        self.backend = ctx.core.backend
        # 缺口#1：position_hist 表用前注册（幂等·cognition 扩展表·守依赖单向向下）
        register_position_hist(self.backend)
        self.position_histogram_state = (
            position_histogram_state or PositionHistogramState(self.backend)
        )
        # B6 指代维：pronoun_resolution_count 表用前注册（幂等·方案3 tn+fn 路·gate 守写读·守依赖单向向下）
        register_pronoun_resolution_count(self.backend)
        # G2 修饰方向A：modification_hist 表用前注册（幂等· 的-cue head/modifier 统计·source write gate-independent·守依赖单向向下）
        register_modification_hist(self.backend)
        self.edge_store = EdgeStore(self.backend)
        self.hub_degree_state = (
            hub_degree_state or HubDegreeState(self.edge_store)
        )
        self.concept_index = concept_index or ConceptIndex(self.backend, ctx.companion)
        self.work_memory = work_memory or WorkMemory()
        self.lemmatizer = lemmatizer
        self.sense_lookup = sense_lookup
        if type(record_legacy_sense_counts) is not bool:
            raise TypeError("record_legacy_sense_counts 必须是 bool")
        self.record_legacy_sense_counts = record_legacy_sense_counts
        self.pronoun_feature_lookup = pronoun_feature_lookup
        self.word_form_providers = word_form_providers
        self.occurrence_index = occurrence_index
        if source_intake is None and occurrence_index is not None and ctx.companion is not None:
            from pure_integer_ai.cognition.understanding.source_intake import SourceIntake
            source_intake = SourceIntake(
                occurrence_index.source_repository, ctx.companion)
        self.source_intake = source_intake
        self.occurrence_order_writer = occurrence_order_writer
        if occurrence_order_writer is not None and occurrence_index is None:
            raise ValueError("occurrence 顺序 writer 必须与 OccurrenceIndex 同时装配")
        if type(write_legacy_language_sequences) is not bool:
            raise TypeError("write_legacy_language_sequences 必须是 bool")
        self.write_legacy_language_sequences = write_legacy_language_sequences
        self.scoped_identity_store = ScopedIdentityStore(self.backend)

    def _next_timestamp(self, clock: LogicalClock | None) -> int:
        """推进 scoped 时钟；无 scope 的旧调用只走运行期兼容序号。"""
        if clock is not None:
            timestamp = clock.advance()
            return timestamp.seq
        legacy_seq = getattr(self.backend, "_legacy_observe_timestamp_seq", 0) + 1
        self.backend._legacy_observe_timestamp_seq = legacy_seq
        return legacy_seq

    def _get_isa_ancestor_hoist(self, space_id: int):
        """#1115：IS_A ancestor_map 增量 hoist（per space·per-run·lazy 首建·挂 backend）。

        返 (ancestor_map, desc_index) | None（None = gate OFF / backend 不支持动态属性 / 该 space 有环禁增量）。
        首次调用 lazy 建（build_isa_ancestor_map_with_index·fresh copy + desc_index·含到此时全 IS_A）·
        observe 期间 build_isa_edges 建新 IS_A 后 apply_isa_edge_to_map 增量更新（环→标记 None 禁增量）。

        **挂 backend 非 self**（cProfile 证）：observe() 便捷入口（observe.py:429）每 call 新建 ObservePipeline·
        挂 self 会 per-observe-call 重建（cProfile 实测 90×0.4s=36s 更差）。backend per-run（formal_train 单 backend）·
        跨 observe/round 复用·镜像既有 backend._isa_ancestor_cache（gen-cache·abstraction.py:87）范式。
        详见 doc/重来_observe性能_#1115_修法设计_2026-07-18.md §7/§13。
        """
        if not getattr(gates, "SELECTION_PREF_MODE", False):
            return None   # gate OFF·不 hoist（逐字现状·CI bit-identical）
        cache = getattr(self.backend, "_isa_ancestor_hoist", None)
        if cache is None:
            cache = {}
            try:
                self.backend._isa_ancestor_hoist = cache
            except (AttributeError, TypeError):
                return None   # backend 不支持动态属性 → 不 hoist（退化 self-built）
        if space_id in cache:
            return cache[space_id]   # (amap, didx) 或 None(环禁增量)
        amap, didx = build_isa_ancestor_map_with_index(self.backend, space_id=space_id)
        cache[space_id] = (amap, didx)
        return (amap, didx)

    def observe(self, raw: InputPayload) -> ObserveResult:
        """在当前 episode 生命周期内执行 observe，异常时丢弃未完成边界。"""
        scoped = self.work_memory.episode_active
        if (not scoped and raw.scope_identity is not None
                and raw.scope_identity.scope_kind != SCOPE_DOCUMENT):
            raise WorkMemoryScopeError(
                "带 episode/query/generation scope 的 observe 必须先打开 WorkMemory 生命周期")
        if scoped:
            self.work_memory.assert_episode_scope(raw.scope_identity)
        try:
            return self._observe_impl(raw)
        except BaseException:
            if scoped:
                self.work_memory.abort_episode()
            raise

    def _observe_impl(self, raw: InputPayload) -> ObserveResult:
        """执行段级建图；生命周期边界由外层 observe 入口统一管理。"""
        assert_no_float(self.ctx.stage, raw.source, raw.stage, _where="observe")
        # observer-source 不变量（审1 C1 / 审2 F-2·v2 非循环心脏）：observe 永不接 source=SOURCE_CONCEPTNET——
        # ConceptNet oracle 须 bypass observe（boot 直注 / labeler 直建 edge）。observe 建 CONCEPTNET-source 边
        # 会过 REALIZES `_has_external_*` filter·破 v2 非循环（doc/重来_对应泛化_结构反推 §四）。
        assert raw.source != SOURCE_CONCEPTNET, (
            "observer-source invariant (审1 C1 / 审2 F-2): ConceptNet oracle must bypass observe "
            "(boot direct-inject / labeler direct-build edge). An observe-built CONCEPTNET-source edge "
            "would pass REALIZES _has_external_* filter and break v2 non-circular heart.")
        # 路由依赖必须先于 Companion、来源表和图写入核验，失败的 Observation 不得留下半摄入来源。
        observation_space_id = target_space_id(raw.stage, self.ctx)
        result = ObserveResult()
        observation_clock = None
        observation_clock_start = 0
        if raw.scope_identity is not None:
            observation_clock = self.scoped_identity_store.resume_clock(
                LogicalClockIdentity(raw.scope_identity, CLOCK_OBSERVATION))
            observation_clock_start = observation_clock.current_seq
        struct_refs: list[ConceptRef] = []
        language_struct_refs: list[ConceptRef] = []   # 语言段 struct_ref（inter-seg PRECEDES 串链·代码域不参与）
        last_tokens: list[ConceptRef] = []   # item3 缺漏5：每段末 token（inter-seg PRECEDES 边 from）
        order_base = 0   # 全局 token 序（C4·跨段递增）
        seg_idx = 0
        previous_occurrence = None
        previous_document_index = None
        # ② fix（#733·J4 指代层3）：per-round reset 悬空段集·observe 段末标 struct_ref 进 dangling_units·judge ② 查
        scoped = self.work_memory.episode_active
        if scoped:
            self.work_memory.begin_observation_state()
        else:
            # 旧直接 observe fixture 没有生命周期身份，保留兼容路径但不把它当作生产契约。
            self.work_memory.dangling_units.clear()
            self.work_memory._segment_dangling = 0
        occurrence_enabled = (
            self.occurrence_index is not None
            and raw.source_ref is not None
            and raw.raw_text is not None
        )
        if raw.stage != STAGE_TRAINING:
            if raw.source_ref is None or raw.raw_text is None:
                raise ValueError("断奶后 Observation 必须携带 SourceRef 和原文")
            if self.source_intake is None:
                raise RuntimeError("断奶后 Observation 缺少 Companion 来源入口")
            if raw.source_license_id is None or raw.source_batch_id is None:
                raise ValueError("断奶后 Observation 缺少许可或 batch")
            self.source_intake.ensure(
                raw.source_ref,
                raw.raw_text,
                license_id=raw.source_license_id,
                batch_id=raw.source_batch_id,
            )
        if occurrence_enabled:
            if raw.occurrence_scope_identity is None:
                raise ValueError("L-03 occurrence 写入必须携带稳定来源 scope")
            self.occurrence_index.ensure_source(raw.source_ref, raw.raw_text)

        for seg in raw.segments:
            if scoped:
                self.work_memory.begin_segment(seg_idx)
            # parse_segment 按 modality 分发（I1·语言首版实·非语言骨架 defer）
            parsed = self._parse_segment(seg)
            # 落点 space（M4·按 stage·非硬编码 CORE）
            space_id = observation_space_id
            memory_space_id = (self.ctx.memory_read.space_id
                               if self.ctx.memory_read is not None else space_id)

            # ② fix（#733）：段内悬空计数 reset·resolve_pronoun_occurrence 悬空时 _segment_dangling++·段末归 dangling_units
            if not scoped:
                self.work_memory._segment_dangling = 0
            # 层1 同段指代（factor E·2026-07-09）：段首清当前段前序 token ref 列表·token loop 内 append·
            # resolve_pronoun_occurrence gate PRONOUN_INTRASEG_MODE ON 时读此作同段前指候选源。
            if not scoped:
                self.work_memory._current_segment_refs.clear()
            # G2 修饰方向A（2-token lookback·source write gate-independent·镜像 factor E）：
            # 当 prev1 token 是 的（is_property_attr_marker）→ 当前 token=head·prev2=modifier → observe_modification。
            _md_prev1_tok: str | None = None
            _md_prev1_ref: ConceptRef | None = None
            _md_prev2_ref: ConceptRef | None = None
            # ② normalize 每 token 归一（模块3）
            resolved: list[ConceptRef] = []
            segment_occurrences = []
            if occurrence_enabled:
                lengths = (
                    len(seg.token_spans),
                    len(seg.document_token_indices),
                    len(seg.occurrence_ordinals),
                )
                if any(length != len(parsed.tokens) for length in lengths):
                    raise ValueError("L-03 occurrence 元数据未与 segment tokens 对齐")
            for ti, tok in enumerate(parsed.tokens):
                if scoped:
                    self.work_memory.next_occurrence_ordinal()
                ref = normalize_to_concept(
                    tok, concept_index=self.concept_index, edge_store=self.edge_store,
                    space_id=space_id, source=raw.source,
                    work_memory=self.work_memory, memory_space_id=memory_space_id,
                    timestamp_seq=self._next_timestamp(observation_clock),
                    lemmatizer=self.lemmatizer, sense_lookup=self.sense_lookup,
                    pronoun_feature_lookup=self.pronoun_feature_lookup,
                    backend=self.backend, lang=raw.lang,
                    hub_degree_state=self.hub_degree_state,
                )
                representation_candidate = None
                if self.word_form_providers is not None:
                    representation_candidate = self.word_form_providers.observe_surface(
                        tok,
                        runtime_language=raw.lang,
                        space_id=space_id,
                    )
                if isinstance(ref, MultiRef):
                    # 单值 caller 才可进入此兼容投影；typed caller 在上游已严格拒绝多 Sense。
                    # legacy MultiRef 各 sense 写 sense_candidates sc_tn；typed 路径不得回写旧目录。
                    # gate SENSE_LOOKUP_MODE OFF → sense_lookup=None → MultiRef 不产 → 此分支不进·退化 bit-identical。
                    resolved.append(ref.refs[0])
                    if (self.record_legacy_sense_counts
                            and getattr(gates, "SENSE_LOOKUP_MODE", False)):
                        _sh = sense_surface_hash(tok)
                        for _sense_ref in ref.refs:
                            record_sense_token_seen(self.backend, space_id, _sh, _sense_ref)
                else:
                    resolved.append(ref)
                if occurrence_enabled:
                    normalized_refs = (
                        tuple(ref.refs) if isinstance(ref, MultiRef) else (ref,))
                    typed_candidates = []
                    legacy_candidates = []
                    for normalized_ref in normalized_refs:
                        typed = self.occurrence_index.typed_candidate_for_node(
                            normalized_ref)
                        if typed is None:
                            legacy_candidates.append(normalized_ref)
                        else:
                            typed_candidates.append(typed)
                    if representation_candidate is not None:
                        typed_candidates.append(representation_candidate)
                    start, end = seg.token_spans[ti]
                    occurrence = self.occurrence_index.record(
                        source=raw.source_ref,
                        raw_text=raw.raw_text,
                        scope=raw.occurrence_scope_identity,
                        start=start,
                        end=end,
                        ordinal=seg.occurrence_ordinals[ti],
                        segment_index=seg.seg_id,
                        local_index=ti,
                        document_index=seg.document_token_indices[ti],
                        speaker=raw.speaker_identity,
                        typed_candidates=tuple(typed_candidates),
                        legacy_candidates=tuple(legacy_candidates),
                    )
                    result.occurrence_refs.append(occurrence.occurrence)
                    segment_occurrences.append(occurrence.occurrence)
                    if (self.occurrence_order_writer is not None
                            and previous_occurrence is not None
                            and previous_document_index is not None):
                        order_fact = self.occurrence_order_writer.record_adjacent(
                            previous_occurrence,
                            occurrence.occurrence,
                            source=raw.source_ref,
                            scope=raw.occurrence_scope_identity,
                            previous_position=previous_document_index,
                            current_position=seg.document_token_indices[ti],
                        )
                        result.order_fact_assertion_hashes.append(
                            order_fact.assertion_hash)
                    previous_occurrence = occurrence.occurrence
                    previous_document_index = seg.document_token_indices[ti]
                # 层1 同段指代（factor E）：append 本段已 normalize 的 token ref（取 resolved[-1]·两分支统一·
                # MultiRef 取 refs[0] 同 resolved）。pronoun normalize 时此列表含前序不含自身（append 在 normalize 返回后）。
                # 对抗审 Bug#1：代词永不作先行词（它→他 pronoun→pronoun 污染 OCCURRENCE 边·未解析代词 SHADOW ref
                # 入候选致后代词误解析到代词）。代词的先行词（若解析）已是段内内容词 token·自身入候选·跳过代词不丢 anaphora 链。
                if not is_pronoun(tok):
                    self.work_memory._current_segment_refs.append(resolved[-1])
                # G2 修饰方向A： 的-cue 2-token lookback（prev1=的 → cur=head·prev2=modifier·source write
                # gate-independent·唯一读 head_pref_score gated·gate OFF 表 inert→bit-identical）。
                if _md_prev1_tok is not None and is_property_attr_marker(_md_prev1_tok, raw.lang) \
                        and _md_prev2_ref is not None:
                    observe_modification(self.backend, head_ref=resolved[-1],
                                         modifier_ref=_md_prev2_ref)
                _md_prev2_ref = _md_prev1_ref
                _md_prev1_ref = resolved[-1]
                _md_prev1_tok = tok
                result.built_concepts += 1

            result.segment_occurrence_refs.append(segment_occurrences)

            # 段结构概念点（一句/一步一概念·承载 role_seq 属性）。
            # code/arith 用**内容哈希** struct_ref（多程序去重·Task #477·__seg_{stage}_{seg_idx} 原 seg_idx
            # 每 observe 重置→多程序/多轮撞同 struct_ref→重 build 复制边 corrupt 树）·语言原用 seg_idx（段序）。
            # **perf round8**：语言 seg_idx 跨 item 同 (stage,seg_idx) 撞同 struct_ref（concept_index dedup）
            # → attach_role_seq 在同 struct_ref 反复 insert（无幂等·role_precedes:110）→ 千段累积 ~4564 role
            # → generate read_role_seq 返全长 → dispatch_slot 爆炸（O(n^1.5) 真根因·n=12 实测 730K dispatch）。
            # gate STRUCT_REF_CONTENT_HASH_MODE ON → 语言分支 h63(seg.tokens)（镜像 code/arith 内容哈希机制）。
            # **诚实边界（2审 FINDING-A）**：哈希机制镜像成立·够用程度不成立——code/arith continue 在 attach_role_seq
            # 前（:195/212·不承载 role_seq）·语言会 attach（:223）·故内容哈希只挡**跨 item 跨句**碰撞（主因~960×）·
            # 同句/跨轮 re-observe 仍撞同 struct_ref 仍累积（~16× 残留·baseline 既有·非本修引入）。真根因=attach 无幂等·
            # defer（first-write-wins guard·round8b）。default OFF 守 CI bit-identical·断奶/perf env 翻 ON。
            if seg.modality == MODALITY_CODE and seg.code_source:
                seg_label = f"__prog_{raw.stage}_{Hasher('observe.prog.v1').h63(seg.code_source)}"
            elif seg.modality == MODALITY_ARITH and seg.arith_source:
                seg_label = f"__prog_{raw.stage}_{Hasher('observe.prog.v1').h63(seg.arith_source)}"
            elif getattr(gates, "STRUCT_REF_CONTENT_HASH_MODE", False) and seg.tokens:
                seg_label = f"__seg_{raw.stage}_{Hasher('observe.seg.v1').h63(seg.tokens)}"
            else:
                seg_label = f"__seg_{raw.stage}_{seg_idx}"
            struct_ref = self.concept_index.ensure(
                seg_label, space_id=space_id,
                tier=TIER_PRIMARY, node_type=NODE_CONCEPT,
            )
            # 维度桥 writer（P1 G-PR2·COMPOSES_COMBINE_MODE ON·建 EDGE_INSTANTIATES 真边 on __seg_ struct_ref→skeleton_ref）。
            # 读 work_memory.lang_skeleton_by_item[(raw.item_key, seg_idx)]（discovery scope B 句级建·(item,seg_idx) 键·
            # seg_idx 与 discovery _sentence_bounds 同源切法对齐）。命中→build_instantiates_edge。
            # gate OFF 或 raw.item_key=0 或无 map→不建→bit-identical。幂等（同 (struct→skeleton) skip·Phase A §十三-bis A.1）。
            # 真边替 ATTR_SKELETON_BINDING 注解（关联在图中·kind=24 effect-dormant 删）·honest EPI_STRUCTURED 纯结构绑定（审2 APPROVE）。
            if getattr(gates, "COMPOSES_COMBINE_MODE", False) and raw.item_key:
                _dim_skel = self.work_memory.lang_skeleton_by_item.get((raw.item_key, seg_idx))
                if _dim_skel is not None:
                    build_instantiates_edge(self.edge_store, struct_ref, _dim_skel,
                                            space_id=space_id)
            # ② fix（#733）：段内若有悬空代词（resolve_pronoun_occurrence 返 None）·标 struct_ref 进
            # dangling_units·judge check_closure ② 查 output.parts[*].unit ∈ dangling_units → J4=0 真碎句。
            # 代码/算术段无代词（pronoun 语言专属）·_segment_dangling 恒 0·no-op·bit-identical。
            if self.work_memory._segment_dangling > 0:
                self.work_memory.dangling_units.add(struct_ref)
            # 篇章结构序（缺口①·修正分析九v2·chapter_seq_table 独立扩展表·struct_ref 章节标记）。
            # observe 创建 struct_ref 后落表（modality 无关·语言/代码/算术段都可承载章节结构）·
            # 生成 M5 章边界分页候选读（generate.py·反 theater 最小消费者）。segment.chapter_seq 由
            # 机器可读结构源 parse 填（HTML/Markdown/LaTeX/code AST·输入层 parser·defer 独立项）·
            # 默认 0=无章节标记→不调 attach→现存测试零行为变 bit-identical（attach try/except 向后兼容）。
            if seg.chapter_seq or seg.section_seq:
                from pure_integer_ai.storage.chapter_seq import attach_chapter_seq
                attach_chapter_seq(self.backend, ref=struct_ref,
                                   chapter_seq=seg.chapter_seq,
                                   section_seq=seg.section_seq)
            # 代码域 modality gate（A3·致命#3·doc/重来_A3_代码域observe设计补充.md §四）：
            # MODALITY_CODE 段建 COMPOSES 树（root=struct_ref）·跳过语言建边·不串 PRECEDES 序链。
            # struct_ref 既是 episode seed/sink 锚点又是 COMPOSES 根（dag_path.sink=struct_ref=root·vm_proof_fn 可定位）。
            if seg.modality == MODALITY_CODE:
                if seg.code_source:
                    # 幂等：已建 COMPOSES 出边 → skip（Task #477·多程序去重·防重 observe 重 build 复制边 corrupt）
                    if not self.edge_store.query_from(struct_ref[0], struct_ref[1],
                                                     edge_type=EDGE_COMPOSES):
                        from pure_integer_ai.cognition.understanding.code_observe import (
                            build_composes_from_source)
                        build_composes_from_source(
                            seg.code_source, concept_index=self.concept_index,
                            edge_store=self.edge_store, backend=self.backend,
                            space_id=space_id, source=raw.source, root_ref=struct_ref)
                        result.built_concepts += 1
                struct_refs.append(struct_ref)   # episode seed/sink=COMPOSES 根（致命#3）
                if scoped:
                    self.work_memory.end_segment(resolved)
                seg_idx += 1
                continue   # 跳过 build_precedes/causes/is_a/cooccurs/attach_role_seq（代码域无句间序）
            # 算术域 modality gate（A3 兄弟件·doc/重来_算术域observe设计补充.md §九）：
            # MODALITY_ARITH 段建 COMPOSES 树（root=struct_ref）·跳过语言建边·不串 PRECEDES 序链。
            if seg.modality == MODALITY_ARITH:
                if seg.arith_source:
                    # 幂等：已建 COMPOSES 出边 → skip（Task #477·多程序去重·防重 observe 重 build 复制边 corrupt）
                    if not self.edge_store.query_from(struct_ref[0], struct_ref[1],
                                                     edge_type=EDGE_COMPOSES):
                        from pure_integer_ai.cognition.understanding.arith_observe import (
                            build_composes_from_arith)
                        build_composes_from_arith(
                            seg.arith_source, concept_index=self.concept_index,
                            edge_store=self.edge_store, backend=self.backend,
                            space_id=space_id, source=raw.source, root_ref=struct_ref)
                        result.built_concepts += 1
                struct_refs.append(struct_ref)   # episode seed/sink=COMPOSES 根（致命#3）
                if scoped:
                    self.work_memory.end_segment(resolved)
                seg_idx += 1
                continue   # 跳过语言建边（算术域无句间序·同代码域）
            # L-05B2A typed formal generation 逐调用关闭旧宽序；未安装 typed owner 的兼容路径保持原行为。
            if self.write_legacy_language_sequences:
                result.built_edges += build_precedes_edges(
                    self.edge_store, resolved, source=raw.source,
                    space_id=space_id, order_base=order_base)
                if resolved:
                    result.built_edges += build_struct_anchor(
                        self.edge_store, struct_ref, resolved[0],
                        source=raw.source, space_id=space_id,
                        order_base=order_base)
                if parsed.role_seq:
                    attach_role_seq(self.backend, struct_ref, parsed.role_seq,
                                    order_base=order_base)
                elif resolved:
                    rseq = self.position_histogram_state.roles_for_tokens(resolved)
                    attach_role_seq(self.backend, struct_ref, rseq,
                                    order_base=order_base)
                    for ti, tok_ref in enumerate(resolved):
                        self.position_histogram_state.observe(tok_ref, ti)
                if (getattr(gates, "DISPATCH_TOKEN_CHAIN_MODE", False)
                        and resolved):
                    attach_token_seq(self.backend, struct_ref, resolved,
                                     order_base=order_base)
            # ④ CAUSES（模块2-bis·§8.1c 硬边界·reward 反传唯一落点）
            result.built_edges += build_causes_edges(
                self.edge_store, resolved,
                structured_pairs=parsed.structured_causal_pairs,
                cue_pairs=parsed.cue_based_causal_pairs,
                source=raw.source, space_id=space_id,
                weaning_phase=self.ctx.weaning_phase,
                assertion_scope=raw.scope_identity,
                assertion_store=(self.scoped_identity_store
                                 if raw.scope_identity is not None else None),
                qualifier_prefix=(seg_idx,))
            # ④-bis IS_A（§8.1b proper subset·致命3 来源② 系词提取·Segment.is_a_pairs）
            result.built_edges += build_is_a_edges(
                self.edge_store, resolved,
                is_a_pairs=parsed.is_a_pairs,
                source=raw.source, space_id=space_id)
            # #1115 perf：增量更新 IS_A ancestor_map hoist（build_is_a_edges 建新 IS_A 后 apply·免 selection_pref 全图重建）。
            # _get lazy 首建（首次含到此时全 IS_A·apply 幂等）·后续命中 apply 新边增量。
            # 环边 apply 返 False → 标记该 space None 禁增量（selection_pref 退化 self-built 全量·SCC 正确处理环）。
            # gate SELECTION_PREF_MODE OFF → _get 返 None → 不 apply（逐字现状·bit-identical）。
            _h_apply = self._get_isa_ancestor_hoist(space_id)
            if _h_apply is not None and parsed.is_a_pairs:
                _amap_h, _didx_h = _h_apply
                for _ci, _pi in parsed.is_a_pairs:
                    if (0 <= _ci < len(resolved) and 0 <= _pi < len(resolved)
                            and resolved[_ci] != resolved[_pi]):
                        _child_h, _parent_h = resolved[_ci], resolved[_pi]
                        # #1115 §14 sync：与 build_is_a_edges 同源去重——apply 前查反向边·有则 skip（不入 hoist）。
                        # 否则 build_is_a_edges skip 了 backend 边·apply 仍用 parsed.is_a_pairs 原始对 → 环检测
                        # 命中 → hoist None（实测 n=50 130s 复发根因·build skip 无效因 apply 不同步）。
                        if has_reverse_isa_edge(self.edge_store, _child_h, _parent_h):
                            continue   # 反向 (parent IsA child) 已存在 → 双向矛盾·skip（mirror build_is_a_edges）
                        if not apply_isa_edge_to_map(_amap_h, _didx_h, _child_h, _parent_h):
                            # #1115 §14.9：环边（apply 环检测命中·直接双向 / 多跳闭环）→ skip 不入 hoist。
                            # ancestor_map 不含环边（DAG 守·无自环·审1 HIGH-1 环检测仍在 apply_isa_edge_to_map·
                            # 仅 caller 选 skip 非 invalidate）·hoist 保留（增量全程生效·解 n=50 127s）。
                            # bit-identical：CI gate OFF observe 不建边（环边无）·生产 gate ON hoist skip 环边
                            # （环边不入 ancestor_map）→ CI/生产 ancestor_map 都无 observe 环边·一致。
                            continue
            # ④-ter G1+#774 属性命题（reification·命题节点承载 subject/attr_type/value 三元·Segment.property_claims·
            # "X 的 Y 是 Z" 提取·build_property_edges 建命题节点 ATTR_PROPOSITION + PROPERTY value 出边·G3b 读判矛盾·
            # gate PROPOSITION_MODE·OFF 返 0 守回归 bit-identical·命题节点不进 dag_path·判断层载体非路径层载体）。
            if getattr(gates, "PROPOSITION_MODE", False) and parsed.property_claims:
                # STEP5 PR3：possess un-defer·领属命题（"X 具有 Z"·attr_idx<0）用 REL_PROPERTY ConceptRef
                # 作默认 attr_type·补命题身份 (subject,REL_PROPERTY,value)·G3b 消费。ensure 幂等（boot 已建）。
                # default_attr_ref=None→既有 skip（bit-identical·PROPOSITION_MODE OFF 时本块不入）。
                _rel_refs = ensure_relation_primitives(self.concept_index, self.backend,
                                                       space_id=space_id)
                result.built_edges += build_property_edges(
                    self.edge_store, self.concept_index, self.backend, resolved,
                    property_claims=parsed.property_claims,
                    source=raw.source, space_id=space_id,
                    default_attr_ref=_rel_refs.get(REL_PROPERTY))
            # ④-quater STEP5 PR4 EDGE_SIMILAR 相似关系（"X 像 Y"·离散符号关系·D2 合规非向量·
            # slot-filler 候选扩展·dispatch_slot 读消费·gate SIMILAR_SLOT_MODE·不接 reward·strength 恒=1）
            if parsed.similar_claims:
                result.built_edges += build_similar_edges(
                    self.edge_store, resolved,
                    similar_claims=parsed.similar_claims,
                    source=raw.source, space_id=space_id)
            # CONDITION（§8.1c 条件包含≠因果）写侧 2026-07-09 删（YAGNI·总收口 §五1.2·
            # 死写侧无 parser 设 has_condition+零读侧消费者·EDGE_CONDITION=7 保留注册登记但不激活）。
            # ⑤ COOCCURS（模块6·分桶 SHADOW）
            result.built_edges += build_cooccurs(
                self.edge_store, resolved, lang=raw.lang, domain=raw.domain,
                source=raw.source, space_id=space_id,
                hub_degree_state=self.hub_degree_state)
            # ⑤-bis 选择倾向共现统计（刀5 件5 地基·§十 边约束·selection_pref_count 表 sp_tn 写）
            # 段内 token 类聚合共现·双向记录 (a, class_of(b))/(b, class_of(a))·predicate 写时识别 defer S4。
            # gate SELECTION_PREF_MODE·self-gate·OFF 返 0·守回归 bit-identical（self.backend 同 observe 既有）。
            # lang 桶守门（C1 防跨语言污染·同 cooccurs 契约·段内单 lang）。
            build_selection_pref_count(self.backend, resolved, space_id=space_id, lang=raw.lang,
                                       ancestor_map=(lambda _h: _h[0] if _h is not None else None)(
                                           self._get_isa_ancestor_hoist(space_id)))
            # ⑥ SPATIAL_ADJ（模块2-ter·空间模态·语言 only no-op）
            # 概念点模态标走 abstract_mark MARK_MODALITY（§7.7.1 路径 B·非 build_spatial_adj 参数·
            # 首版概念点零模态标·bit-identical·未来空间模态落地时 caller ensure 后 set_mark）。
            if raw.modality in (MODALITY_2D, MODALITY_3D, MODALITY_ANIMATION):
                result.built_edges += build_spatial_adj(
                    self.edge_store, parsed.spatial_primitives,
                    source=raw.source, space_id=space_id)
            # ⑦ REFERS_TO 性质A（线索命中·模块4）/ 性质B（代词·模块5 在 normalize 内已调）
            # alias_cue_pairs = 来源② cue 派生（同位语/又名·observe 时句法提取）→ EPI_CUE（Phase B §十四-bis 纠 pre-existing
            # 误标：原 EPI_STRUCTURED 违 refers_stable 来源② taxonomy·is_a.py:85/causes.py:47 同类 cue 路径都标 EPI_CUE）。
            # 纠后 PURE_ALIAS 闭包(Phase B)epistemic==EPI_STRUCTURED 滤正确排除 cue 自证 alias（anti-self-proving）。
            for ai, bi in parsed.alias_cue_pairs:
                a, b = resolved[ai], resolved[bi]
                result.built_edges += build_refers_stable_edge(
                    self.edge_store, self.concept_index, a, b,
                    epistemic=EPI_CUE, space_id=space_id)

            # 段入 WorkMemory FIFO（供后续段 pronoun 回溯·性质B 跨句 partial）
            if scoped:
                self.work_memory.end_segment(resolved)
            else:
                self.work_memory.push_segment(seg_idx, resolved)
            struct_refs.append(struct_ref)
            language_struct_refs.append(struct_ref)   # 语言段才串 inter-seg PRECEDES（代码域 continue 不入）
            last_tokens.append(resolved[-1] if resolved else struct_ref)   # item3 缺漏5：段末 token（inter-seg 边 from）
            order_base += max(len(parsed.tokens), 1)
            seg_idx += 1

        # 句间序 PRECEDES（替旧 TYPE_SENTENCE_TRANSITION·§十五E·item3 缺漏5 d-1：段末token→下段struct_ref）
        # 仅语言段串序链（代码域无句间序·language_struct_refs 不含代码段·A3）
        if (self.write_legacy_language_sequences
                and len(language_struct_refs) > 1):
            result.built_edges += build_inter_segment_precedes(
                self.edge_store, language_struct_refs, last_tokens, source=raw.source,
                space_id=observation_space_id,
                seg_order_base=order_base)

        # ⑧ 落点分流（模块8·审计记录）
        route = route_to_space(raw.stage, self.ctx)
        result.deferred.append(f"route:{route}")
        result.struct_refs = struct_refs   # 段结构概念 ref 序（formal_train episode seed/sink/key_skeleton 用）
        if (observation_clock is not None
                and observation_clock.current_seq > observation_clock_start):
            self.scoped_identity_store.register_timestamp(LogicalTimestamp(
                observation_clock.identity,
                observation_clock.current_seq,
            ))
        return result

    def _parse_segment(self, seg: Segment) -> Segment:
        """parse_segment 按 modality 分发（I1·§7.4.2）。

        语言首版：Segment 已预处理（tokens/role_seq/因果对/代词/线索填）·直接返。
        非语言：骨架 defer（声=帧序列/2D=图元ℚ²/3D=体素/动画=帧+空间）·空 token 流 + spatial_primitives。
        """
        if seg.modality == MODALITY_LANGUAGE:
            return seg   # 语言已预处理（tokens 等·预处理层填·observe 边界）
        # 非语言骨架（I1·消解"声称非defer交付defer"·伪代码层兑现骨架）
        if seg.modality in (MODALITY_AUDIO, MODALITY_ANIMATION):
            # 时间模态：PRECEDES-time 帧序（复用 build_precedes·模态标走 abstract_mark MARK_MODALITY·§7.7.1 路径 B）
            return seg   # tokens = 帧序列（预处理填·骨架 defer 空流）
        if seg.modality in (MODALITY_2D, MODALITY_3D):
            return seg   # spatial_primitives = 图元（预处理填·骨架 defer 空）
        return seg


def observe(raw: InputPayload, ctx: SpaceContext, **kwargs) -> ObserveResult:
    """observe 便捷入口（构造 ObservePipeline 并跑）。"""
    return ObservePipeline(ctx, **kwargs).observe(raw)
