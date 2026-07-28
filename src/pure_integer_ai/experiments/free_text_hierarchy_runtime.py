"""从 raw 文本布局形成来源绑定的 section/paragraph/proposition 候选。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    AbsoluteSpan,
    CandidateEvidence,
    HierarchyCandidate,
    SourceDocument,
)


_TERMINATORS = frozenset("。！？!?；;")


class FreeTextHierarchyRuntimeError(RuntimeError):
    """raw 文本布局或形成后的层级不满足来源和包含关系。"""


def _key(domain: int, *parts: int) -> StableRecordKey:
    """把完整严格整数组成压到稳定正整数身份。"""
    if any(type(value) is not int for value in parts):
        raise TypeError("hierarchy key parts 必须是严格整数")
    payload = ":".join(str(value) for value in (domain, *parts)).encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((domain, value if value else 1))


def _line_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """返回所有非空物理行去除换行后的绝对半开范围。"""
    result = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        surface = line.rstrip("\r\n")
        if surface:
            result.append((cursor, cursor + len(surface)))
        cursor += len(line)
    if cursor < len(text):
        result.append((cursor, len(text)))
    return tuple(result)


def _proposition_ranges(
        text: str, start: int, end: int,
        ) -> tuple[tuple[int, int], ...]:
    """仅按显式句末符机械切分行内 proposition，不解释词义。"""
    result = []
    cursor = start
    for index in range(start, end):
        if text[index] in _TERMINATORS:
            if cursor < index + 1:
                result.append((cursor, index + 1))
            cursor = index + 1
    if cursor < end:
        result.append((cursor, end))
    return tuple(result)


@dataclass(frozen=True)
class FormedTextHierarchy:
    """一次机械层级形成的候选、来源 Evidence 与零标签读取事实。"""

    document: SourceDocument
    candidates: tuple[HierarchyCandidate, ...]
    evidence: tuple[CandidateEvidence, ...]
    private_label_read_count: int

    def __post_init__(self) -> None:
        """核验结果非空、身份唯一且没有私有标签读取。"""
        if not isinstance(self.document, SourceDocument):
            raise TypeError("formed hierarchy document 类型错误")
        if (not isinstance(self.candidates, tuple) or not self.candidates
                or any(not isinstance(item, HierarchyCandidate)
                       for item in self.candidates)):
            raise TypeError("formed hierarchy candidates 类型错误")
        if (not isinstance(self.evidence, tuple) or not self.evidence
                or any(not isinstance(item, CandidateEvidence)
                       for item in self.evidence)):
            raise TypeError("formed hierarchy evidence 类型错误")
        if len({item.candidate_key for item in self.candidates}) != len(
                self.candidates):
            raise FreeTextHierarchyRuntimeError("hierarchy candidate identity 重复")
        if len({item.evidence_key for item in self.evidence}) != len(self.evidence):
            raise FreeTextHierarchyRuntimeError("hierarchy Evidence identity 重复")
        if self.private_label_read_count != 0:
            raise FreeTextHierarchyRuntimeError("hierarchy former 不得读取私有标签")

    def ranges(self) -> tuple[tuple[str, int, int], ...]:
        """返回 evaluator 可比较的 kind 与绝对范围，不返回原文副本。"""
        return tuple(
            (item.candidate_kind, item.span.start, item.span.end)
            for item in self.candidates
        )


class MechanicalTextHierarchyFormer:
    """使用物理行和显式句末符形成层级，不携带语言词表或答案规则。"""

    def form(self, document: SourceDocument) -> FormedTextHierarchy:
        """从 raw SourceDocument 形成一节、多段和行内 proposition 候选。"""
        if not isinstance(document, SourceDocument):
            raise TypeError("hierarchy former 需要 SourceDocument")
        text = document.raw_text
        line_ranges = _line_ranges(text)
        if not line_ranges:
            raise FreeTextHierarchyRuntimeError("raw 文本没有非空物理行")
        section_key = _key(9101, *document.document_key.components)
        section_span = AbsoluteSpan(
            _key(9102, *section_key.components),
            document.source_ref,
            0,
            len(text),
        )
        section_evidence = CandidateEvidence(
            _key(9103, *section_key.components),
            "STRUCTURE",
            "SOURCE",
            document.source_ref,
            section_span.span_key,
            section_key,
        )
        candidates = [HierarchyCandidate(
            section_key,
            "SECTION",
            section_span,
            None,
            1,
            "SUPPORTED",
            (section_evidence.evidence_key,),
        )]
        evidence = [section_evidence]
        for paragraph_ordinal, (start, end) in enumerate(line_ranges, start=1):
            paragraph_key = _key(
                9111, *document.document_key.components, start, end)
            paragraph_span = AbsoluteSpan(
                _key(9112, *paragraph_key.components),
                document.source_ref,
                start,
                end,
            )
            paragraph_evidence = CandidateEvidence(
                _key(9113, *paragraph_key.components),
                "STRUCTURE",
                "SOURCE",
                document.source_ref,
                paragraph_span.span_key,
                paragraph_key,
            )
            candidates.append(HierarchyCandidate(
                paragraph_key,
                "PARAGRAPH",
                paragraph_span,
                section_key,
                paragraph_ordinal,
                "SUPPORTED",
                (paragraph_evidence.evidence_key,),
            ))
            evidence.append(paragraph_evidence)
            for proposition_ordinal, (left, right) in enumerate(
                    _proposition_ranges(text, start, end), start=1):
                proposition_key = _key(
                    9121, *document.document_key.components, left, right)
                proposition_span = AbsoluteSpan(
                    _key(9122, *proposition_key.components),
                    document.source_ref,
                    left,
                    right,
                )
                proposition_evidence = CandidateEvidence(
                    _key(9123, *proposition_key.components),
                    "RAW_SPAN",
                    "SOURCE",
                    document.source_ref,
                    proposition_span.span_key,
                    proposition_key,
                )
                candidates.append(HierarchyCandidate(
                    proposition_key,
                    "PROPOSITION",
                    proposition_span,
                    paragraph_key,
                    proposition_ordinal,
                    "SUPPORTED",
                    (proposition_evidence.evidence_key,),
                ))
                evidence.append(proposition_evidence)
        return FormedTextHierarchy(
            document,
            tuple(candidates),
            tuple(evidence),
            0,
        )


__all__ = [
    "FormedTextHierarchy",
    "FreeTextHierarchyRuntimeError",
    "MechanicalTextHierarchyFormer",
]
