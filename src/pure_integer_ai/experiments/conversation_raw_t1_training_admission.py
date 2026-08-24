"""T1-G20：候选物理闸门之后的 raw T1 training pack admission。

该入口先运行 G19 candidate promotion audit，再调用 G14 pack assembler。它不生成 annotation，
不把 mechanical candidate 当作词义，也不写入任何训练/记忆介质；其作用是阻止绕过物理边界
直接装配训练输入。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    load_raw_lexical_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_t1_training_pack import (
    RawT1TrainingPack,
    RawT1TrainingPackError,
    assemble_raw_t1_training_pack,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_promotion_audit import (
    PROMOTION_ELIGIBLE_FOR_REVIEW,
    PROMOTION_NEGATIVE_WITNESS_ONLY,
    RawTextCandidatePromotionAudit,
    RawTextCandidatePromotionError,
    audit_raw_text_candidate_promotion,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_evidence import (
    RawTextCandidateEvidenceError,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_coverage import (
    RawTextCandidateCoverageError,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)


RAW_T1_TRAINING_ADMISSION_PROTOCOL_V1 = 1
ADMISSION_ACCEPTED = 1


class RawT1TrainingAdmissionError(ValueError):
    """训练 pack 未通过候选物理 admission。"""


# object-model: value; representation=struct; interop=T1-G20
@dataclass(frozen=True, slots=True)
class RawT1TrainingAdmission:
    """已通过 G19 的 pack 与其只读 admission audits。"""

    status: int
    pack: RawT1TrainingPack
    audits: tuple[RawTextCandidatePromotionAudit, ...]

    def __post_init__(self) -> None:
        if self.status != ADMISSION_ACCEPTED:
            raise RawT1TrainingAdmissionError("admission status 未注册")
        if type(self.pack) is not RawT1TrainingPack:
            raise TypeError("admission pack 类型错误")
        if not isinstance(self.audits, tuple) or not self.audits:
            raise RawT1TrainingAdmissionError("admission audits 不能为空")
        if any(type(item) is not RawTextCandidatePromotionAudit
               for item in self.audits):
            raise TypeError("admission audit 类型错误")
        valid = tuple(item for item in self.audits
                      if item.split in {"train", "heldout"})
        negatives = tuple(item for item in self.audits if item.split == "negative")
        if not valid or not negatives:
            raise RawT1TrainingAdmissionError("admission 缺 valid 或 negative audit")
        if any(item.status != PROMOTION_ELIGIBLE_FOR_REVIEW for item in valid):
            raise RawT1TrainingAdmissionError("valid audit 未通过 G19")
        if any(item.status != PROMOTION_NEGATIVE_WITNESS_ONLY for item in negatives):
            raise RawT1TrainingAdmissionError("negative audit 不得进入 valid admission")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_TRAINING_ADMISSION_PROTOCOL_V1, self.status,
                  len(self.audits)]
        for item in self.audits:
            record = item.canonical_record()
            result.extend((len(record), *record))
        pack = self.pack.canonical_record()
        result.extend((len(pack), *pack))
        return tuple(result)


def admit_raw_t1_training_pack(
        observation_payload: bytes,
        lexical_payload: bytes,
        proposition_payload: bytes,
        qualification_payload: bytes,
        ) -> RawT1TrainingAdmission:
    """先审计 G19，再装配 G14；任一候选边界失败即拒绝。"""
    observations = load_raw_text_observation_jsonl(observation_payload)
    lexical = load_raw_lexical_evidence_jsonl(lexical_payload)
    audits = []
    for observation in observations:
        selected = tuple(item for item in lexical
                         if item.observation_id == observation.observation_id)
        try:
            audit = audit_raw_text_candidate_promotion(
                extract_raw_text_candidate_spans(observation.raw_bytes),
                observation, selected)
        except (RawTextCandidatePromotionError, RawTextCandidateEvidenceError,
                RawTextCandidateCoverageError, TypeError) as error:
            raise RawT1TrainingAdmissionError(
                f"{observation.observation_id} candidate admission 失败") from error
        audits.append(audit)
    try:
        pack = assemble_raw_t1_training_pack(
            observation_payload, lexical_payload,
            proposition_payload, qualification_payload)
    except RawT1TrainingPackError as error:
        raise RawT1TrainingAdmissionError("G14 pack assembly 失败") from error
    try:
        return RawT1TrainingAdmission(ADMISSION_ACCEPTED, pack, tuple(audits))
    except (RawT1TrainingAdmissionError, TypeError) as error:
        raise RawT1TrainingAdmissionError("G20 admission audit 失败") from error


__all__ = [
    "ADMISSION_ACCEPTED",
    "RAW_T1_TRAINING_ADMISSION_PROTOCOL_V1",
    "RawT1TrainingAdmission",
    "RawT1TrainingAdmissionError",
    "admit_raw_t1_training_pack",
]
