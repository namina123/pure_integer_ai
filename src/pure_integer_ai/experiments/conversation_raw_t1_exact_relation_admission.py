"""T1-G27：exact candidate evidence 接入 G1/G2/G3 的只读 admission probe。

该模块只消费已经通过 G26 的 exact lexical adapter，再复用现有 proposition bind 与
qualification consumer。它不从文本猜命题，也不把 candidate evidence 删除；原始 evidence
ID 会保留在 admission trace 中。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerResult,
    RawPropositionQualification,
    consume_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationBinding,
    RawPropositionRelationEvidence,
    bind_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
)
from pure_integer_ai.experiments.conversation_raw_t1_candidate_granularity import (
    RawT1CandidateGranularityAudit,
)
from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import (
    RawT1ConsensusCandidateEvidence,
)
from pure_integer_ai.experiments.conversation_raw_t1_exact_lexical_adapter import (
    ADAPTER_ACCEPTED,
    RawT1ExactLexicalAdapterResult,
    adapt_exact_candidate_to_lexical_evidence,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_T1_EXACT_RELATION_ADMISSION_PROTOCOL_V1 = 1
EXACT_RELATION_ADMISSION_ACCEPTED = 1


class RawT1ExactRelationAdmissionError(ValueError):
    """exact lexical 到 proposition/qualification admission 失败。"""


# object-model: value; representation=struct; interop=T1-G27
@dataclass(frozen=True, slots=True)
class RawT1ExactRelationAdmission:
    """G1/G2/G3 闭合结果与 candidate evidence 身份。"""

    status: int
    candidate_evidence_ids: tuple[str, ...]
    lexical_evidence: tuple[RawLexicalEvidence, ...]
    binding: RawPropositionRelationBinding
    consumer: RawPropositionConsumerResult

    def __post_init__(self) -> None:
        if self.status != EXACT_RELATION_ADMISSION_ACCEPTED:
            raise RawT1ExactRelationAdmissionError("admission status 未注册")
        if not self.candidate_evidence_ids:
            raise RawT1ExactRelationAdmissionError("candidate evidence ids 不能为空")
        if tuple(sorted(set(self.candidate_evidence_ids))) != self.candidate_evidence_ids:
            raise RawT1ExactRelationAdmissionError("candidate evidence ids 必须排序且唯一")
        if not isinstance(self.lexical_evidence, tuple) or not self.lexical_evidence:
            raise RawT1ExactRelationAdmissionError("lexical evidence 不能为空")
        if type(self.binding) is not RawPropositionRelationBinding:
            raise TypeError("admission binding 类型错误")
        if type(self.consumer) is not RawPropositionConsumerResult:
            raise TypeError("admission consumer 类型错误")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_EXACT_RELATION_ADMISSION_PROTOCOL_V1, self.status,
                  len(self.candidate_evidence_ids)]
        for value in self.candidate_evidence_ids:
            result.extend((len(value), *(ord(item) for item in value)))
        for item in self.lexical_evidence:
            record = item.canonical_record()
            result.extend((len(record), *record))
        binding = self.binding.canonical_record
        consumer = self.consumer.integer_record
        result.extend((len(binding), *binding, len(consumer), *consumer))
        return tuple(result)


def admit_exact_candidate_relation(
        observation: RawTextObservation,
        candidate_evidence: tuple[RawT1ConsensusCandidateEvidence, ...],
        granularity_audits: tuple[RawT1CandidateGranularityAudit, ...],
        proposition: RawPropositionRelationEvidence,
        qualification: RawPropositionQualification,
        *,
        authority: str,
        ) -> RawT1ExactRelationAdmission:
    """执行 G26→G1→G2→G3 闭合；任一层失败均归一为 admission error。"""
    if type(observation) is not RawTextObservation:
        raise TypeError("exact relation admission 需要 observation")
    if not isinstance(candidate_evidence, tuple) or not candidate_evidence:
        raise RawT1ExactRelationAdmissionError("candidate evidence 不能为空")
    if not isinstance(granularity_audits, tuple):
        raise TypeError("granularity audits 必须是 tuple")
    by_id = {item.evidence_id: item for item in granularity_audits}
    if len(by_id) != len(granularity_audits):
        raise RawT1ExactRelationAdmissionError("granularity evidence_id 不得重复")
    lexical: list[RawLexicalEvidence] = []
    adapter_results: list[RawT1ExactLexicalAdapterResult] = []
    for evidence in candidate_evidence:
        if evidence.evidence_id not in by_id:
            raise RawT1ExactRelationAdmissionError("candidate 缺 granularity audit")
        result = adapt_exact_candidate_to_lexical_evidence(
            evidence, by_id[evidence.evidence_id], observation, authority=authority)
        adapter_results.append(result)
        if result.status != ADAPTER_ACCEPTED or result.lexical_evidence is None:
            raise RawT1ExactRelationAdmissionError("candidate 未形成 exact lexical evidence")
        lexical.append(result.lexical_evidence)
    if len({item.evidence_id for item in lexical}) != len(lexical):
        raise RawT1ExactRelationAdmissionError("lexical evidence_id 不得重复")
    try:
        binding = bind_raw_proposition_relation(observation, tuple(lexical), proposition)
        consumer = consume_raw_proposition_relation(binding, qualification)
    except (TypeError, ValueError) as error:
        raise RawT1ExactRelationAdmissionError("G1/G2/G3 闭合失败") from error
    return RawT1ExactRelationAdmission(
        EXACT_RELATION_ADMISSION_ACCEPTED,
        tuple(sorted(item.evidence_id for item in candidate_evidence)),
        tuple(lexical), binding, consumer,
    )


__all__ = [
    "EXACT_RELATION_ADMISSION_ACCEPTED",
    "RAW_T1_EXACT_RELATION_ADMISSION_PROTOCOL_V1",
    "RawT1ExactRelationAdmission", "RawT1ExactRelationAdmissionError",
    "admit_exact_candidate_relation",
]
