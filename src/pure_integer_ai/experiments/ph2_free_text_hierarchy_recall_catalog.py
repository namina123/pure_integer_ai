"""P3-Ia T0 authored sample 的确定构建、读取与 cluster 审计。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_USER,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    StableRecordKey,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    AbsoluteSpan,
    CandidateEvidence,
    CenterCandidate,
    EvaluatorLabel,
    FreeTextHierarchyRecallCase,
    FreeTextHierarchyRecallContractError,
    FreeTextQuery,
    HierarchyCandidate,
    OwnerIsolation,
    RecallBudget,
    RecallCitation,
    RecallObligation,
    RecallReceipt,
    SourceDocument,
    SplitIdentity,
)


SAMPLE_RELATIVE_PATH = "data/ph2/authored_free_text_hierarchy_recall_seed_v1.jsonl.sample"


def _key(domain: int, ordinal: int, item: int = 1) -> StableRecordKey:
    return StableRecordKey((domain, ordinal, item))


def _source(ordinal: int, *, query: bool = False) -> SourceRef:
    return SourceRef(
        312 if query else 311,
        7000 + ordinal,
        ordinal,
        OwnerScope(91, 17, 0, VISIBILITY_USER),
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(1),
            PrimitiveVersion(1),
            CurriculumVersion(1),
        ),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _span(
        ordinal: int,
        item: int,
        source: SourceRef,
        start: int,
        end: int,
        ) -> AbsoluteSpan:
    return AbsoluteSpan(_key(20, ordinal, item), source, start, end)


def _build_case(
        ordinal: int,
        *,
        split: str,
        raw_text: str,
        query_text: str,
        section_markers: tuple[str, str],
        paragraph_surfaces: tuple[str, str],
        ) -> FreeTextHierarchyRecallCase:
    """建立一个窄 T0 fixture；它不是生产 parser 或语义规则。"""
    source = _source(ordinal)
    query_source = _source(ordinal, query=True)
    document = SourceDocument(
        source,
        _key(10, ordinal, 1),
        _key(11, ordinal, 1),
        _key(12, ordinal, 1),
        raw_text,
        _sha(raw_text),
    )
    first_section_start = raw_text.index(section_markers[0])
    second_section_start = raw_text.index(section_markers[1])
    first_paragraph_start = raw_text.index(paragraph_surfaces[0])
    second_paragraph_start = raw_text.index(paragraph_surfaces[1])
    spans = (
        _span(ordinal, 1, source, first_section_start, second_section_start),
        _span(ordinal, 2, source, second_section_start, len(raw_text)),
        _span(
            ordinal, 3, source, first_paragraph_start,
            first_paragraph_start + len(paragraph_surfaces[0])),
        _span(
            ordinal, 4, source, second_paragraph_start,
            second_paragraph_start + len(paragraph_surfaces[1])),
        _span(
            ordinal, 5, source, first_paragraph_start,
            first_paragraph_start + len(paragraph_surfaces[0])),
        _span(
            ordinal, 6, source, second_paragraph_start,
            second_paragraph_start + len(paragraph_surfaces[1])),
    )
    candidate_keys = tuple(_key(30, ordinal, item) for item in range(1, 7))
    evidence = tuple(CandidateEvidence(
        _key(40, ordinal, item),
        "STRUCTURE" if item <= 4 else ("PARAPHRASE" if item == 6 else "RAW_SPAN"),
        "SOURCE" if item <= 5 else "HISTORY",
        source,
        spans[item - 1].span_key,
        candidate_keys[item - 1],
    ) for item in range(1, 7))
    hierarchy = (
        HierarchyCandidate(
            candidate_keys[0], "SECTION", spans[0], None, 1, "SUPPORTED",
            (evidence[0].evidence_key,)),
        HierarchyCandidate(
            candidate_keys[1], "SECTION", spans[1], None, 2, "SUPPORTED",
            (evidence[1].evidence_key,)),
        HierarchyCandidate(
            candidate_keys[2], "PARAGRAPH", spans[2], candidate_keys[0], 1,
            "SUPPORTED", (evidence[2].evidence_key,)),
        HierarchyCandidate(
            candidate_keys[3], "PARAGRAPH", spans[3], candidate_keys[1], 1,
            "SUPPORTED", (evidence[3].evidence_key,)),
        HierarchyCandidate(
            candidate_keys[4], "PROPOSITION", spans[4], candidate_keys[2], 1,
            "SUPPORTED", (evidence[4].evidence_key,)),
        HierarchyCandidate(
            candidate_keys[5], "PROPOSITION", spans[5], candidate_keys[3], 1,
            "SUPPORTED", (evidence[5].evidence_key,)),
    )
    query = FreeTextQuery(
        _key(50, ordinal, 1),
        query_source,
        _key(51, ordinal, 1),
        _key(52, ordinal, 1),
        _key(53, ordinal, 1),
        query_text,
        _sha(query_text),
    )
    center = CenterCandidate(
        _key(60, ordinal, 1),
        query.query_key,
        query.situation_key,
        "HIERARCHY_CANDIDATE",
        candidate_keys[5],
        "ACTIVE",
        (evidence[5].evidence_key,),
        1,
        0,
    )
    budget = RecallBudget(2, 1, 4096, 1)
    obligation = RecallObligation(
        _key(70, ordinal, 1),
        center.center_key,
        query.query_key,
        "PROPOSITION",
        candidate_keys[5],
        source,
        document.scope_key,
        document.acl_key,
        budget,
        "NONE",
    )
    citation = RecallCitation(
        _key(80, ordinal, 1),
        _key(81, ordinal, 1),
        source,
        spans[5],
    )
    receipt = RecallReceipt(
        obligation.obligation_key,
        (_key(82, ordinal, 1),),
        (_key(83, ordinal, 1),),
        1,
        1,
        len(paragraph_surfaces[1].encode("utf-8")),
        (candidate_keys[5],),
        (citation,),
        "RESOLVED",
        0,
    )
    owner = OwnerIsolation(
        _key(90, ordinal, 1),
        _key(90, ordinal, 2),
        _key(90, ordinal, 3),
        _key(90, ordinal, 4),
        f"packs/p3i/{ordinal}/source.jsonl",
        f"packs/p3i/{ordinal}/candidate.jsonl",
        f"packs/p3i/{ordinal}/teacher.jsonl",
        f"packs/p3i/{ordinal}/evaluator.jsonl",
        ("CANDIDATE", "HISTORY", "SOURCE"),
        tuple(item.evidence_key for item in evidence),
        0,
    )
    split_identity = SplitIdentity(
        split,
        _key(100, ordinal, 1),
        _key(101, ordinal, 1),
        _key(102, ordinal, 1),
        _key(103, ordinal, 1),
        _key(104, ordinal, 1),
    )
    label = EvaluatorLabel(
        _key(110, ordinal, 1),
        owner.evaluator_owner_key,
        tuple(item.candidate_key for item in hierarchy),
        (center.center_key,),
        candidate_keys[5],
        (citation.citation_key,),
    )
    return FreeTextHierarchyRecallCase(
        1,
        _key(1, ordinal, 1),
        document,
        query,
        evidence,
        hierarchy,
        (center,),
        obligation,
        receipt,
        owner,
        split_identity,
        label,
    )


def build_t0_cases() -> tuple[FreeTextHierarchyRecallCase, ...]:
    """返回一对 cluster 物理分离的 train/held-out T0 case。"""
    return (
        _build_case(
            1,
            split="train",
            raw_text=(
                "项目记录\n第一节\n玄衡计划的备用站位于北川。\n"
                "第二节\n后续纪要确认备用站已经迁至南岭。"),
            query_text="玄衡计划的备用站后来搬到哪里？",
            section_markers=("第一节", "第二节"),
            paragraph_surfaces=(
                "玄衡计划的备用站位于北川。",
                "后续纪要确认备用站已经迁至南岭。"),
        ),
        _build_case(
            2,
            split="held_out",
            raw_text=(
                "维护日志\n上篇\n苍梧节点的校验端口设在东塔。\n"
                "下篇\n更新记录说明校验端口已改到西台。"),
            query_text="苍梧节点现在从哪一处校验？",
            section_markers=("上篇", "下篇"),
            paragraph_surfaces=(
                "苍梧节点的校验端口设在东塔。",
                "更新记录说明校验端口已改到西台。"),
        ),
    )


def validate_case_catalog(
        cases: tuple[FreeTextHierarchyRecallCase, ...],
        ) -> tuple[FreeTextHierarchyRecallCase, ...]:
    """拒绝 case 重复或任一 cluster 跨 split 泄漏。"""
    if (not isinstance(cases, tuple) or not cases
            or any(not isinstance(item, FreeTextHierarchyRecallCase)
                   for item in cases)):
        raise FreeTextHierarchyRecallContractError("catalog cases 非法")
    if cases != tuple(sorted(cases, key=lambda item: item.case_key)):
        raise FreeTextHierarchyRecallContractError("catalog case 必须规范排序")
    if len({item.case_key for item in cases}) != len(cases):
        raise FreeTextHierarchyRecallContractError("catalog case identity 重复")
    cluster_splits: dict[tuple[str, StableRecordKey], str] = {}
    owner_paths: set[str] = set()
    for case in cases:
        for name, cluster in case.split_identity.cluster_items():
            key = name, cluster
            previous = cluster_splits.setdefault(key, case.split_identity.split)
            if previous != case.split_identity.split:
                raise FreeTextHierarchyRecallContractError("cluster 跨 split 泄漏")
        isolation = case.owner_isolation
        for path in (
                isolation.source_relative_path,
                isolation.candidate_relative_path,
                isolation.teacher_relative_path,
                isolation.evaluator_relative_path):
            if path in owner_paths:
                raise FreeTextHierarchyRecallContractError("owner path 跨 case 复用")
            owner_paths.add(path)
    return cases


def build_t0_sample_bytes() -> bytes:
    """把 T0 case 编码为规范、确定的 authored JSONL。"""
    return b"".join(
        canonical_json_line(item.to_dict())
        for item in validate_case_catalog(build_t0_cases())
    )


def load_t0_sample(
        path: str | Path,
        ) -> tuple[FreeTextHierarchyRecallCase, ...]:
    """严格读取规范 JSONL 并重跑全部合同和 cluster 审计。"""
    payload = Path(path).read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise FreeTextHierarchyRecallContractError("T0 sample 必须非空且换行结束")
    lines = payload.splitlines()
    cases = tuple(FreeTextHierarchyRecallCase.from_dict(
        parse_canonical_json_bytes(line, require_object=True)) for line in lines)
    return validate_case_catalog(cases)


__all__ = [
    "SAMPLE_RELATIVE_PATH",
    "build_t0_cases",
    "build_t0_sample_bytes",
    "load_t0_sample",
    "validate_case_catalog",
]
