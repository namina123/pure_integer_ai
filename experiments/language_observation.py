"""语言样本从词形候选到来源化 Segment 的运行期编排。"""
from __future__ import annotations

from typing import Sequence

from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope as source_document_scope,
)
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_CODE,
    DOMAIN_MATH,
    LANG_NONE,
    MODALITY_ARITH,
    MODALITY_CODE,
    MODALITY_LANGUAGE,
    Segment,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_comparison_claims_gated,
    extract_cues_gated,
    extract_existential_claims_gated,
    extract_numeric_claims_gated,
    extract_property_claims_gated,
    extract_similar_claims_gated,
    extract_universal_claims_gated,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.text_segments import sentence_bounds
from pure_integer_ai.experiments.train_context import TrainContext


def _boundary_language_key(ctx: TrainContext,
                           item: CollectedItem) -> tuple[int, ...]:
    """返回当前语言分支的稳定图身份；无 provider 时保留运行期注入键。"""
    providers = ctx.word_form_providers
    if providers is not None:
        provider = providers.provider(item.lang)
        if provider is not None:
            return ctx.graph_ontology.identity_of(provider.branch).stable_key()
    return (item.lang,)


def _prepare_item_boundary(
        ctx: TrainContext, item: CollectedItem, *,
        commit_evidence: bool, persist_graph: bool) -> None:
    """解析一个 item 的显式句界 Evidence，并可写入训练图选择。"""
    engine = ctx.boundary_hypothesis_engine
    materializer = ctx.boundary_span_materializer
    if engine is None or materializer is None:
        item.boundary_parse = None
        item.boundary_decision = None
        return
    if item.modality != MODALITY_LANGUAGE:
        item.boundary_parse = None
        item.boundary_decision = None
        return
    if item.raw_text is None:
        if item.boundary_profile is not None:
            raise ValueError("U-03 句界 Evidence 必须绑定可回读 raw_text")
        item.boundary_parse = None
        item.boundary_decision = None
        return
    if item.source_ref is None:
        raise ValueError("U-03 句界解析前必须分配 SourceRef")
    token_spans = _item_token_source_spans(item)
    if token_spans is None:
        raise RuntimeError("U-03 无法取得来源化 winner token span")
    result = engine.resolve(
        item.raw_text,
        observation=item.source_ref,
        scope=source_document_scope(item.source_ref),
        language_key=_boundary_language_key(ctx, item),
        profile=item.boundary_profile,
        commit=commit_evidence,
    )
    item.boundary_parse = result
    if persist_graph:
        materialized = materializer.materialize(
            result,
            token_spans=token_spans,
        )
        item.boundary_decision = materialized.decision
    else:
        item.boundary_decision = result.decision()


def _prepare_item_boundaries(
        ctx: TrainContext, corpus: Sequence[CollectedItem], *,
        commit_evidence: bool, persist_graph: bool) -> int:
    """批量准备句界决定，并返回实际参与解析的语言来源数量。"""
    prepared = 0
    for item in corpus:
        _prepare_item_boundary(
            ctx,
            item,
            commit_evidence=commit_evidence,
            persist_graph=persist_graph,
        )
        if item.boundary_parse is not None:
            prepared += 1
    return prepared


def _apply_word_form_providers(
        corpus: list[CollectedItem], providers, *,
        commit_evidence: bool = True) -> int:
    """在所有训练消费者之前生成词形候选并投影当前兼容 winner token。"""
    if providers is None:
        return 0
    changed = 0
    for item in corpus:
        if (item.modality != MODALITY_LANGUAGE
                or item.raw_text is None
                or not providers.supports(item.lang)):
            continue
        index_bound = any((
            item.role_seq,
            item.causal_pairs,
            item.is_a_pairs,
            item.alias_cue_pairs,
        ))
        if index_bound:
            raise ValueError(
                "带 token 索引标注的 CollectedItem 不得在 formal_train 内重新分词")
        if item.source_ref is None:
            raise ValueError("raw_text 课程分词前必须先分配 SourceRef")
        parse_result = providers.parse_text(
            item.raw_text,
            runtime_language=item.lang,
            observation=item.source_ref,
            scope=source_document_scope(item.source_ref),
            commit_evidence=commit_evidence,
        )
        if parse_result is None:
            item.word_form_parse = None
            item.tokens = providers.segment_text(
                item.raw_text, runtime_language=item.lang)
        else:
            item.word_form_parse = parse_result
            item.tokens = list(parse_result.tokens)
        changed += 1
    return changed


def _materialize_item_spans(ctx: TrainContext, item: CollectedItem,
                            observed) -> None:
    """把分词 lattice 和句界选择接到共享 L-04 Span 图。"""
    segmentation_materializer = ctx.segmentation_span_materializer
    if segmentation_materializer is not None and item.word_form_parse is not None:
        if ctx.occurrence_index is None:
            raise RuntimeError("L-04 正式接线必须同时启用 L-03 occurrence")
        materialized = segmentation_materializer.materialize(
            item.word_form_parse,
            occurrence_refs=tuple(observed.occurrence_refs),
        )
        observed.span_refs.extend(materialized.span_refs)
        observed.span_statement_assertion_hashes.extend(
            materialized.statement_hashes)

    boundary_materializer = ctx.boundary_span_materializer
    if boundary_materializer is None or item.boundary_parse is None:
        return
    token_spans = _item_token_source_spans(item)
    if token_spans is None:
        raise RuntimeError("U-03 句界 occurrence 锚定缺少来源 token span")
    boundary = boundary_materializer.materialize(
        item.boundary_parse,
        token_spans=token_spans,
        occurrence_refs=tuple(observed.occurrence_refs),
    )
    item.boundary_decision = boundary.decision
    observed.span_refs.extend(boundary.span_refs)
    observed.span_statement_assertion_hashes.extend(
        boundary.statement_hashes)


def _run_item_predictions(ctx: TrainContext, item: CollectedItem,
                          observed) -> None:
    """对当前来源 occurrence 运行 H-01 先预测后学习，并暴露分维结果。"""
    runtime = ctx.language_prediction_runtime
    if runtime is None or item.modality != MODALITY_LANGUAGE:
        return
    occurrence_refs = tuple(observed.occurrence_refs)
    if not occurrence_refs:
        return
    report = runtime.observe_document(occurrence_refs)
    observed.prediction_results.extend(report.evaluations)
    if report not in ctx.language_prediction_reports:
        ctx.language_prediction_reports.append(report)


def _run_item_sense_candidates(ctx: TrainContext, item: CollectedItem,
                               observed) -> tuple:
    """在 occurrence/span 物化后运行 typed Sense prediction 和独立揭示。"""
    runtime = ctx.sense_candidate_course_runtime
    if runtime is None or item.modality != MODALITY_LANGUAGE:
        return ()
    if ctx.occurrence_index is None or ctx.span_index is None:
        raise RuntimeError("H-05 Sense recognition 缺少 occurrence/span 地基")
    traces = runtime.observe_item(ctx, item, observed)
    observed.sense_candidate_traces.extend(traces)
    report = runtime.report()
    if not ctx.sense_candidate_reports or ctx.sense_candidate_reports[-1] != report:
        ctx.sense_candidate_reports.append(report)
    return traces


def _run_item_semantic_course(
        ctx: TrainContext, item: CollectedItem, input_payload, observed):
    """在 typed Span/Sense 就绪后执行正式语义课程，并保存同次请求产物。"""
    runtime = ctx.language_semantic_course_runtime
    if runtime is None or item.modality != MODALITY_LANGUAGE:
        return None
    run = runtime.process(ctx, item, input_payload, observed)
    observed.semantic_course_run = run
    ctx.language_semantic_course_reports.append(run)
    return run


def _item_token_source_spans(
        item: CollectedItem) -> tuple[tuple[int, int, int], ...] | None:
    """把当前 token 投影回原文码点 span，并为同位 occurrence 分配稳定 ordinal。"""
    if item.raw_text is None:
        return None
    raw_spans: list[tuple[int, int]] = []
    if item.word_form_parse is not None:
        selected = item.word_form_parse.selected
        if selected is None:
            if item.tokens:
                raise ValueError("无 winner 的词形解析不得携带非空 token")
            return ()
        parts = selected.segmentation.parts
        if tuple(part.surface for part in parts) != tuple(item.tokens):
            raise ValueError("词形 winner 与 CollectedItem.tokens 不一致")
        raw_spans = [(part.start, part.end) for part in parts]
    else:
        cursor = 0
        for token in item.tokens:
            if not isinstance(token, str):
                raise TypeError("CollectedItem.tokens 必须是字符串序列")
            start = item.raw_text.find(token, cursor)
            if start < 0:
                raise ValueError("token 无法按序对齐回 raw_text")
            end = start + len(token)
            raw_spans.append((start, end))
            cursor = end
    ordinals: dict[tuple[int, int], int] = {}
    out: list[tuple[int, int, int]] = []
    for start, end in raw_spans:
        key = (start, end)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        out.append((start, end, ordinal))
    return tuple(out)


def _item_sentence_bounds(item: CollectedItem) -> list[tuple[int, int]]:
    """把当前 active 句界决定投影为所有消费者共享的 token 半开区间。"""
    token_count = len(item.tokens)
    decision = item.boundary_decision
    if decision is None:
        return sentence_bounds(token_count)
    token_spans = _item_token_source_spans(item)
    if token_spans is None:
        raise RuntimeError("已决句界缺少来源化 token span")
    return sentence_bounds(
        token_count,
        cut_after=decision.token_cuts(token_spans),
    )


def _split_item_to_segments(item: CollectedItem, *,
                            backend=None, edge_store=None,
                            space_id: int | None = None,
                            concept_index=None) -> list[Segment]:
    """CollectedItem 段落 → 已决边界 Segment 列表。

    语言分段只消费 U-03 来源化 active 决定；无证据时整段保留。句段内 token 序保
    PRECEDES 骨架。单段输入只产一个结构根，reward caller 仍按既有规则处理。
    role_seq/causal_pairs/alias_cue_pairs 按 token 切片 + 段内 index 重映射（确定性）。

    **刀5 件8 透传**（close 刀4 生产 gap）：4 可选参透传给 extract_cues_gated → cue_type_of
    第二源 D:11 readback。生产 caller run_round_full 传 ctx.backend/edge_store/space_id/concept_index。
    默认全 None → cue_type_of 退化纯 frozenset → 现状零行为变（bit-identical）。space_id 是
    语言 token 概念化所在 core space（ctx.space_id·concept_index.lookup(surface, ctx.space_id)）。

    **代码域分支**（C6 生产闭环·doc/重来_A3_代码域observe设计补充.md §二致命#2）：MODALITY_CODE
    不消费语言句界候选，一段一函数，Segment 带 code_source。
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
    source_spans = _item_token_source_spans(item)
    if source_spans is not None and len(source_spans) != len(tokens):
        raise ValueError("token span 数量与 CollectedItem.tokens 不一致")
    # 句界：observe、discovery 和 floor 统一消费同一来源化 active 决定。
    segs: list[Segment] = []
    for start, end in _item_sentence_bounds(item):
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
        # A1·STEP6：存在量化声明提取（独立函数·同 CUE_EXTRACTOR_MODE gate·有的 X 是 Y 起始 cue 窗口；
        # 这里只记录声明，MEMBER/nonempty/overlap/DISJOINT 证据由后续 typed adapter 提供）。
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
            token_spans=([] if source_spans is None else [
                (source_spans[index][0], source_spans[index][1])
                for index in range(start, end)
            ]),
            document_token_indices=list(range(start, end)),
            occurrence_ordinals=([] if source_spans is None else [
                source_spans[index][2] for index in range(start, end)
            ]),
        ))
    return segs


__all__ = [
    "_apply_word_form_providers",
    "_item_sentence_bounds",
    "_item_token_source_spans",
    "_materialize_item_spans",
    "_prepare_item_boundaries",
    "_run_item_sense_candidates",
    "_prepare_item_boundary",
    "_run_item_predictions",
    "_split_item_to_segments",
]
