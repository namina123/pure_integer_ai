"""语言 StructureConcept 的发现、识别、证据 tally 与结果汇总。"""
from __future__ import annotations

from typing import Sequence

from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveryRouteStats,
    DiscoveredOperator,
    Recognition,
    StructureTallyStats,
    _collect_cue_sig,
    _collect_slot_lcas,
    _normalize_abstract_sig,
    auto_discover_operators,
    recognize_operators,
    route_samples_for_discovery,
    shape_signature,
)
from pure_integer_ai.cognition.shared.types import (
    ConceptRef,
    MODALITY_LANGUAGE,
    WEANING_PRE,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
)
from pure_integer_ai.config import gates
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.language_observation import (
    _item_sentence_bounds,
    _item_token_source_spans,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    _item_document_identity,
)
from pure_integer_ai.experiments.train_result_types import (
    GeneralizationSummary,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY

_DISC_LANG_SEED = "formal_train.disc_lang"
_DISC_LANG_ALIGN_SEED = "formal_train.disc_lang_align"
_DISC_LANG_SENSE_SEED = "formal_train.disc_lang_sense"


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
    root_sources: dict[ConceptRef, SourceRef] = {}
    root_keys: list[tuple[int, int]] = []   # scope B：每 root 的稳定 document scope 索引和 seg_idx
    root_expected: dict[ConceptRef, ConceptRef | None] = {}   # S7 相0：root → item.expected_skeleton（教师标定比对）
    # 刀6 片4：root → 各 token 位 (ti, tok_text, sense_candidates, leaf_ref)·供 recognize_roots clone 逐 sense 试。
    # sense_candidates = read_sense_candidates base_count>0（boot 种先验）·leaf_ref = 该位 token 叶 ConceptRef
    # （gate ON + 有 sense → 首 sense ref NodeRef 升序首·有 IS_A 上卷·discover slot_lca 真火；否则 ensure(tok) 原路径）。
    root_token_entries: dict[ConceptRef, list[tuple[int, str, list[tuple[int, int]], tuple[int, int]]]] = {}
    _sense_gate_on = bool(
        getattr(gates, "SENSE_LOOKUP_MODE", False)
        and ctx.sense_candidate_course_runtime is None)
    from pure_integer_ai.storage.sense_candidates import read_sense_candidates, sense_surface_hash
    # scope B（断奶 critical path ④·doc/重来_语料聚簇规模_2026-07-17）：gate COMPOSES_COMBINE_MODE ON→按句切建根
    # （_sentence_bounds·与 observe _split_item_to_segments 同源切法·seg_idx 逐一对齐）·OFF→整段单 span=原段级根行为
    # （bit-identical）。observe 本就按句切段（多段 struct_ref·unit 早已句级）→ 切句**不增 observe 成本**·只让
    # discovery 从段级根改句级根 → 同长句聚簇 → 骨架+cue槽涌现（解整段签名塌词数→零骨架根因）。
    _scope_b_split = bool(getattr(gates, "COMPOSES_COMBINE_MODE", False))
    _flat_units: list[
        tuple[CollectedItem, int, int, int, int, list[str]]
    ] = []
    for item in lang_items:
        item_key, _document_scope = _item_document_identity(ctx, item)
        _spans = _item_sentence_bounds(item) if _scope_b_split else [(0, len(item.tokens))]
        for seg_idx, (_s, _e) in enumerate(_spans):
            _toks = list(item.tokens[_s:_e])
            if _toks:
                _flat_units.append(
                    (item, item_key, seg_idx, _s, _e, _toks))
    root_occurrences: dict[ConceptRef, tuple[ObjectIdentity, ...]] = {}
    root_token_spans: dict[
        ConceptRef, tuple[tuple[int, int, int], ...]] = {}
    for item, item_key, seg_idx, token_start, token_end, tokens in _flat_units:
        h = Hasher(_DISC_LANG_SEED).h63("\x1f".join(tokens))
        root = ctx.concept_index.ensure(
            f"__disc_lang_{h}", space_id=ctx.space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 先填 entries（不论是否已建 COMPOSES·clone 段须用·fresh + resume 统一）
        entries: list[tuple[int, str, list[tuple[int, int]], tuple[int, int]]] = []
        for ti, tok in enumerate(tokens):
            sense_cands: list[tuple[int, int]] = []
            if ctx.sense_candidate_course_runtime is not None:
                sense_cands = list(
                    ctx.sense_candidate_course_runtime.active_concept_refs(
                        ctx,
                        runtime_language=item.lang,
                        surface=tok,
                    ))
            elif _sense_gate_on:
                _sh = sense_surface_hash(tok)
                sense_cands = [sr for sr, base, _sn, _tn
                               in read_sense_candidates(ctx.backend, ctx.space_id, _sh) if base > 0]
            # H-05 只在唯一 active typed Sense 时给结构 caller 单值 Concept；多解不得按稳定序私选。
            # 未启 H-05 时保留旧目录首项兼容路径，待迁移窗口结束后删除。
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
        if item.source_ref is None:
            raise RuntimeError("语言结构 root 缺少 SourceRef")
        root_sources[root] = item.source_ref
        item_spans = _item_token_source_spans(item)
        if item_spans is not None:
            selected_spans = item_spans[token_start:token_end]
            root_token_spans[root] = selected_spans
            root_occurrences[root] = tuple(
                occurrence_identity(
                    item.source_ref,
                    start=start,
                    end=end,
                    ordinal=ordinal,
                )
                for start, end, ordinal in selected_spans
            )
        roots.append(root)
        root_keys.append((item_key, seg_idx))
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
                root_sources[align_root] = root_sources[orig_root]
                if orig_root in root_occurrences:
                    root_occurrences[align_root] = root_occurrences[orig_root]
                    root_token_spans[align_root] = root_token_spans[orig_root]
            roots = new_roots
            root_expected = new_expected
        # else: consensus 退化·roots 不变（变长不发现·诚实）
    # 路由（聚类前置·B6 Bug 2+3·2026-07-06·同 _run_arith）：按 (sig,hint) 分组 → LCA 聚类 → 按簇 abstract_sig
    # 路由 discover/recognize。解 existing_keys 缺 abstract_sig（跨 run 覆盖渐失）+ cluster-blind held-out（混合簇
    # 前 K 横跨簇致每簇 <K 不发现）。裸 NL（has_isa=False）单簇 None → abstract_sig=() → bit-identical。helper 详
    # structure_discover.route_samples_for_discovery。
    _route_stats = DiscoveryRouteStats()
    discover_roots, recognize_roots = route_samples_for_discovery(
        ctx.backend, ctx.concept_graph, roots,
        existing_keys=existing_keys, existing_sigs=existing_sigs,
        space_id=ctx.space_id, stats=_route_stats)
    discovered = auto_discover_operators(
        discover_roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=ctx.space_id, source=SOURCE_BARE_TEXT)
    _typed_formation_inputs = 0
    _typed_candidates_registered = 0
    _typed_candidates_recovered = 0
    if ctx.structure_candidate_runtime is not None:
        if ctx.structure_candidate_mapper is None:
            raise RuntimeError("H-05 structure runtime 缺少课程 mapper")
        from pure_integer_ai.experiments.language_structure_candidate_runtime import (
            register_structure_candidates,
        )
        (_typed_formation_inputs, _typed_candidates_registered,
         _typed_candidates_recovered) = (
            register_structure_candidates(
                ctx,
                tuple(discovered),
                root_sources=root_sources,
                mapper=ctx.structure_candidate_mapper,
            ))
    # Phase D §十六-bis D.1：REALIZES labeled bed（option-b oracle-pair-match·skeleton→__REL_SUBSET__）。
    # skeleton 通用文本独立发现 + forming-sample token-pair 命中外源 EDGE_IS_A → REALIZES（oracle 定 IS_A 非读 Cue）。
    # **gate REALIZES_MODE default OFF→整块跳过→bit-identical**（含 ensure_relation_primitives 调用·守 CI 零副作用）。
    # labeled bed（同 boot IS_A）·学习 claim 严禁前置·验 floor Phase F·consumer Phase E·D.1 ship ≠ Phase D done。
    _label_ops = list(existing_ops) + list(discovered)
    if (ctx.structure_candidate_runtime is None
            and getattr(gates, "REALIZES_MODE", False) and _label_ops):
        from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
        from pure_integer_ai.cognition.process.structure_discover import label_realizes_is_a, label_realizes_causes
        _rel_prims = ensure_relation_primitives(
            ctx.concept_index, ctx.backend, space_id=ctx.space_id)   # 幂等·确保 __REL_SUBSET__/__REL_CAUSES__ 存在
        label_realizes_is_a(_label_ops, graph=ctx.concept_graph, edge_store=ctx.edge_store,
                            rel_primitives=_rel_prims, space_id=ctx.space_id)
        # Phase D §十六-bis D.1 CAUSES labeled bed（镜像 IS_A·oracle=外源 ConceptNet CAUSES·REALIZES→__REL_CAUSES__）。
        # condition-6 配套：使-skeleton 走外源 CAUSES oracle 标（非 使 cue·anti-self-proving·避循环）。
        label_realizes_causes(_label_ops, graph=ctx.concept_graph, edge_store=ctx.edge_store,
                              rel_primitives=_rel_prims, space_id=ctx.space_id)
    # 维度桥 item→skeleton map（P1 G-PR2·COMPOSES_COMBINE_MODE ON·shape_signature 匹配 root→discovered skeleton·
    # 存 ctx.work_memory.lang_skeleton_by_item[(document_scope_hash, seg_idx)]（scope B 句级键·observe 读建 EDGE_INSTANTIATES 边）。
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
                    root_sources[_clone_root] = root_sources[_rr]
                    if _rr in root_occurrences:
                        root_occurrences[_clone_root] = root_occurrences[_rr]
                        root_token_spans[_clone_root] = root_token_spans[_rr]
                break   # 首版只处理首个多 sense token 位（多 token 笛卡尔积 defer）
    _aligning_roots = [_ar for _ar, _orig in sense_recognize_inputs]
    _origin_of = dict(sense_recognize_inputs)
    _tally_stats = StructureTallyStats()
    if not all_ops or not _aligning_roots:
        if ctx.structure_candidate_runtime is not None:
            from pure_integer_ai.experiments.language_structure_candidate_runtime import (
                integration_report,
            )
            ctx.structure_candidate_reports.append(integration_report(
                formation_inputs=_typed_formation_inputs,
                candidates_registered=_typed_candidates_registered,
                candidates_recovered=_typed_candidates_recovered,
                recognition_inputs=0,
                traces=(),
            ))
            if ctx.structure_boundary_evidence_mapper is not None:
                from pure_integer_ai.experiments.language_structure_boundary_runtime import (
                    apply_structure_boundary_evidence,
                )
                ctx.structure_boundary_report = (
                    apply_structure_boundary_evidence(
                        ctx,
                        corpus,
                        (),
                        ctx.structure_boundary_evidence_mapper,
                    ))
        return discovered, [], GeneralizationSummary(
            total_held_out=len(recognize_roots),
            routing_stats=_route_stats, tally_stats=_tally_stats)
    recognitions = recognize_operators(
        _aligning_roots, discovered_operators=all_ops,
        backend=ctx.backend, space_id=ctx.space_id)
    _typed_traces = ()
    if ctx.structure_candidate_runtime is not None:
        from pure_integer_ai.experiments.language_structure_candidate_runtime import (
            integration_report,
            recognize_structure_candidates,
        )
        _typed_traces = recognize_structure_candidates(
            ctx,
            tuple(recognitions),
            origin_sources=root_sources,
            origin_of=_origin_of,
            origin_occurrences=root_occurrences,
            origin_token_spans=root_token_spans,
            mapper=ctx.structure_candidate_mapper,
        )
        ctx.structure_candidate_reports.append(integration_report(
            formation_inputs=_typed_formation_inputs,
            candidates_registered=_typed_candidates_registered,
            candidates_recovered=_typed_candidates_recovered,
            recognition_inputs=len(recognitions),
            traces=_typed_traces,
        ))
        if ctx.structure_boundary_evidence_mapper is not None:
            from pure_integer_ai.experiments.language_structure_boundary_runtime import (
                apply_structure_boundary_evidence,
            )
            ctx.structure_boundary_report = apply_structure_boundary_evidence(
                ctx,
                corpus,
                _typed_traces,
                ctx.structure_boundary_evidence_mapper,
            )
        adopted_recognitions = {
            trace.recognition for trace in _typed_traces if trace.adopted
        }
        recognitions = [
            recognition for recognition in recognitions
            if recognition in adopted_recognitions
        ]
    # 对应泛化 v2：结构反推 tally（审1C3/审2条件1+2·三路分离 + SHADOW 创建）。新词 W 落 REALIZES-R-skeleton
    # cue slot（cue-blind tally·独立于 recognize 精确匹配轨）→ tally (W,R) distinct recognition-routed input → 首次建 D:11
    # SHADOW（generator 关后唯一创建者）→ promote W→R D:11 PRIMARY（_structure_match_ok 唯一证据轨）。
    # **gate ORACLE_PROMOTE_MODE**（default OFF·bit-identical·OFF 不调→零 tally→零 D:11 翻→逐字现状）。
    # REALIZES_MODE + CUE_CLUSTER_MODE 须同 ON（REALIZES-skeleton 存在 + ATTR_CUE_SIG 落盘·caller 生产 try/finally 共翻）。
    # **非循环**：R 来自 REALIZES oracle（source==CONCEPTNET·非 cue）·W 观察·反馈在 source filter 断（§四）。
    if (ctx.structure_candidate_runtime is None
            and getattr(gates, "ORACLE_PROMOTE_MODE", False)
            and all_ops and _aligning_roots):
        from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
        from pure_integer_ai.cognition.process.structure_discover import tally_cue_slot_matches
        _rel_prims = ensure_relation_primitives(
            ctx.concept_index, ctx.backend, space_id=ctx.space_id)   # 幂等·tally rel_ref→rel_kind 反查用
        tally_cue_slot_matches(
            _aligning_roots, discovered_operators=all_ops,
            graph=ctx.concept_graph, edge_store=ctx.edge_store,
            backend=ctx.backend, space_id=ctx.space_id,
            rel_primitives=_rel_prims, stats=_tally_stats)
    # S7 相0 钥匙③ vm_proof 降级对偶（教师标定比对·断奶前教师路径·POST 退场·镜像 vm_proof :376）：
    # recognize 命中骨架 ref == item.expected_skeleton → verified（op_confidence sn++·语言算子 name_ref 自然分桶）。
    # concept_binding（S3 第二片）已是结构对齐比对器·无须新写。断奶后教师退场（POST）·相0 不调（防 vacuous 命中 theater）。
    # **教师天花板诚实边界**：expected_skeleton 教师主观标定·非闭式真理——(a) 可能多解（同 input 命中多骨架·取教师标）
    # (b) 可能错标（标错→sn=0 退场·机制检出但首次 run 污染 tn·拉低 rate·反 theater 择优降权）
    # (c) 无第二独立源（vm_proof Mode B 才有·断奶后须相2/E1 接力·钥匙③墙≡#479）。相0 = 结构身份匹配·非真 vm_proof 等价。
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
        verified=0, expected_verified=expected_verified,
        routing_stats=_route_stats, tally_stats=_tally_stats)


__all__ = ["_discover_and_recognize_lang_structures"]
