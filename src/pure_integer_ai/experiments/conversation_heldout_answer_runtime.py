"""DLG-05 独立 ANSWER 输入的来源化 claim surface 边界。

本模块只把同次实际 ``GenerationCandidate`` 的来源记录投影成
``GroundedAnswerClaimInput``，不读取课程 episode、expected answer 或 evaluator
label。完整 G-00 至 G-04 runtime 仍由调用方注入的 run-local factory 装配。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerClaimInput,
)
from pure_integer_ai.storage.source_record import SourceRecordRepository


class ConversationHeldOutAnswerInputError(ValueError):
    """独立 ANSWER claim 输入无法由同次来源闭合。"""


def claim_input_from_candidate(
        candidate: GenerationCandidate,
        source_records: SourceRecordRepository,
        ) -> GroundedAnswerClaimInput:
    """从同次候选的完整来源记录形成无标签 claim input。

    Core Evidence 与 Memory Evidence 均必须能回查唯一 SourceRecord。多来源
    候选只有在来源原文一致时才允许形成单 claim surface；冲突来源直接
    fail closed，交由 CONFLICT/CLARIFY 路径处理。
    """
    if not isinstance(candidate, GenerationCandidate):
        raise TypeError("held-out answer candidate 类型错误")
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("held-out answer SourceRecord 类型错误")
    traces: list[object] = []
    traces.extend(item.source for item in candidate.evidence)
    for evidence in candidate.memory_evidence:
        traces.extend(item.trace.source for item in evidence.sources)
    if not traces:
        raise ConversationHeldOutAnswerInputError(
            "独立 ANSWER candidate 缺少可引用来源")
    texts = []
    seen = set()
    for source in traces:
        key = source.stable_key()
        if key in seen:
            continue
        seen.add(key)
        record = source_records.find(key)
        if record is None:
            raise ConversationHeldOutAnswerInputError(
                "独立 ANSWER candidate 缺少 SourceRecord")
        if not record.raw_text:
            raise ConversationHeldOutAnswerInputError(
                "独立 ANSWER SourceRecord 原文为空")
        texts.append(record.raw_text)
    if len(set(texts)) != 1:
        raise ConversationHeldOutAnswerInputError(
            "独立 ANSWER 多来源原文不一致，不能形成单 claim surface")
    return GroundedAnswerClaimInput(texts[0])


__all__ = [
    "ConversationHeldOutAnswerInputError",
    "claim_input_from_candidate",
]
