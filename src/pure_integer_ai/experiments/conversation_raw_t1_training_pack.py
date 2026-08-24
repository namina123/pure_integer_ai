"""T1-G14：raw observation 到 qualification 的统一训练包装配。

该模块不创造 annotation，也不从文本猜测结构。它把 G0/G1/G2/G3 的公开 JSONL
边界装配为一个只读纯值 pack：train/heldout 进入可消费 case，negative 必须在
闭包阶段按注册失败边界被拒绝。JSONL 只是输入载体，canonical record 才是跨语言
可复现的 pack projection。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
    RawLexicalEvidenceError,
    load_raw_lexical_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerError,
    RawPropositionConsumerResult,
    RawPropositionQualification,
    consume_raw_proposition_relation,
    load_raw_qualification_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionEvidenceError,
    RawPropositionRelationBinding,
    RawPropositionRelationEvidence,
    bind_raw_proposition_relation,
    load_raw_proposition_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RAW_TEXT_OBSERVATION_LICENSE,
    RawTextObservation,
    load_raw_text_observation_jsonl,
)


RAW_T1_TRAINING_PACK_PROTOCOL_V1 = 1
RAW_T1_PACK_CASE_RECORD_V1 = 1
RAW_T1_PACK_NEGATIVE_RECORD_V1 = 2
RAW_T1_NEGATIVE_BINDING_REJECTED = 1
RAW_T1_NEGATIVE_CONSUMER_REJECTED = 2
RAW_T1_TRAINING_SPLITS = frozenset({"train", "heldout"})


class RawT1TrainingPackError(ValueError):
    """G0-G3 constituent pack 无法形成闭合、隔离的训练值。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawT1TrainingPackError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _pack(values: tuple[int, ...], where: str) -> tuple[int, ...]:
    if any(type(item) is not int or item < 0 for item in values):
        raise RawT1TrainingPackError(f"{where} 含非法整数")
    return (len(values), *values)


def _pack_text(value: str, where: str) -> tuple[int, ...]:
    value = _text(value, where)
    return _pack(tuple(ord(item) for item in value), where)


def _pack_records(records: tuple[tuple[int, ...], ...], where: str) -> tuple[int, ...]:
    result = [len(records)]
    for index, record in enumerate(records):
        packed = _pack(record, f"{where}[{index}]")
        result.extend(packed)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RawT1TrainingCase:
    """一个 train/heldout 的完整 G0-G3 可消费闭包。"""

    observation: RawTextObservation
    lexical_evidence: tuple[RawLexicalEvidence, ...]
    proposition: RawPropositionRelationEvidence
    binding: RawPropositionRelationBinding
    qualification: RawPropositionQualification
    consumer: RawPropositionConsumerResult

    def __post_init__(self) -> None:
        if type(self.observation) is not RawTextObservation:
            raise TypeError("pack case observation 类型错误")
        if (not isinstance(self.lexical_evidence, tuple)
                or not self.lexical_evidence
                or any(type(item) is not RawLexicalEvidence for item in self.lexical_evidence)):
            raise TypeError("pack case lexical evidence 类型错误")
        if type(self.proposition) is not RawPropositionRelationEvidence:
            raise TypeError("pack case proposition 类型错误")
        if type(self.binding) is not RawPropositionRelationBinding:
            raise TypeError("pack case binding 类型错误")
        if type(self.qualification) is not RawPropositionQualification:
            raise TypeError("pack case qualification 类型错误")
        if type(self.consumer) is not RawPropositionConsumerResult:
            raise TypeError("pack case consumer 类型错误")
        if self.observation.split not in RAW_T1_TRAINING_SPLITS:
            raise RawT1TrainingPackError("valid pack case 只能是 train/heldout")
        if self.proposition.observation_id != self.observation.observation_id:
            raise RawT1TrainingPackError("case proposition/observation 漂移")
        if self.binding.proposition_id != self.proposition.proposition_id:
            raise RawT1TrainingPackError("case binding/proposition 漂移")
        if self.qualification.proposition_id != self.proposition.proposition_id:
            raise RawT1TrainingPackError("case qualification/proposition 漂移")
        if self.consumer.qualification_id != self.qualification.qualification_id:
            raise RawT1TrainingPackError("case consumer/qualification 漂移")

    @property
    def observation_id(self) -> str:
        return self.observation.observation_id

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_PACK_CASE_RECORD_V1, *self.observation.canonical_record()]
        result.extend(_pack_records(
            tuple(item.canonical_record() for item in self.lexical_evidence),
            "case.lexical",
        ))
        result.extend(_pack(self.proposition.canonical_record(), "case.proposition"))
        result.extend(_pack(self.binding.canonical_record, "case.binding"))
        result.extend(_pack(self.qualification.canonical_record(), "case.qualification"))
        result.extend(_pack(self.consumer.integer_record, "case.consumer"))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawT1NegativeWitness:
    """一个已按注册失败阶段拒绝的 negative constituent case。"""

    observation: RawTextObservation
    lexical_evidence: tuple[RawLexicalEvidence, ...]
    proposition: RawPropositionRelationEvidence
    qualification: RawPropositionQualification
    failure_stage: int

    def __post_init__(self) -> None:
        if type(self.observation) is not RawTextObservation:
            raise TypeError("negative observation 类型错误")
        if self.observation.split != "negative":
            raise RawT1TrainingPackError("negative witness split 必须为 negative")
        if not isinstance(self.lexical_evidence, tuple):
            raise TypeError("negative lexical evidence 类型错误")
        if type(self.proposition) is not RawPropositionRelationEvidence:
            raise TypeError("negative proposition 类型错误")
        if type(self.qualification) is not RawPropositionQualification:
            raise TypeError("negative qualification 类型错误")
        if self.failure_stage not in {
                RAW_T1_NEGATIVE_BINDING_REJECTED,
                RAW_T1_NEGATIVE_CONSUMER_REJECTED,
        }:
            raise RawT1TrainingPackError("negative failure stage 未注册")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_PACK_NEGATIVE_RECORD_V1, self.failure_stage,
                  *self.observation.canonical_record()]
        result.extend(_pack_records(
            tuple(item.canonical_record() for item in self.lexical_evidence),
            "negative.lexical",
        ))
        result.extend(_pack(self.proposition.canonical_record(), "negative.proposition"))
        result.extend(_pack(self.qualification.canonical_record(), "negative.qualification"))
        return tuple(result)


# object-model: value; representation=struct; interop=T1-G14
@dataclass(frozen=True, slots=True)
class RawT1TrainingPack:
    """G0-G3 的完整只读训练 pack 与 split/negative 审计。"""

    source_namespace: str
    license_id: str
    cases: tuple[RawT1TrainingCase, ...]
    negatives: tuple[RawT1NegativeWitness, ...]

    def __post_init__(self) -> None:
        _text(self.source_namespace, "pack.source_namespace")
        if self.license_id != RAW_TEXT_OBSERVATION_LICENSE:
            raise RawT1TrainingPackError("pack license 必须是公开 observation license")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise RawT1TrainingPackError("pack cases 不能为空")
        if not isinstance(self.negatives, tuple) or not self.negatives:
            raise RawT1TrainingPackError("pack 必须保留至少一个 negative witness")
        if any(type(item) is not RawT1TrainingCase for item in self.cases):
            raise TypeError("pack cases 类型错误")
        if any(type(item) is not RawT1NegativeWitness for item in self.negatives):
            raise TypeError("pack negatives 类型错误")
        if tuple(item.observation_id for item in self.cases) != tuple(
                sorted(item.observation_id for item in self.cases)):
            raise RawT1TrainingPackError("pack cases 必须按 observation_id 排序")
        if len({item.observation_id for item in self.cases}) != len(self.cases):
            raise RawT1TrainingPackError("pack cases observation_id 不得重复")
        if not {item.observation.split for item in self.cases} >= {"train", "heldout"}:
            raise RawT1TrainingPackError("pack 必须同时包含 train 与 heldout")

    @property
    def split_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple((split, sum(item.observation.split == split for item in self.cases))
                     for split in ("train", "heldout"))

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_TRAINING_PACK_PROTOCOL_V1]
        result.extend(_pack_text(self.source_namespace, "pack.namespace"))
        result.extend(_pack_text(self.license_id, "pack.license"))
        result.extend(_pack_records(
            tuple(item.canonical_record() for item in self.cases), "pack.cases"))
        result.extend(_pack_records(
            tuple(item.canonical_record() for item in self.negatives), "pack.negatives"))
        return tuple(result)


def assemble_raw_t1_training_pack(
        observation_payload: bytes,
        lexical_payload: bytes,
        proposition_payload: bytes,
        qualification_payload: bytes,
        ) -> RawT1TrainingPack:
    """装配四个公开 JSONL constituent，验证 valid/negative 的闭包边界。"""
    observations = load_raw_text_observation_jsonl(observation_payload)
    lexical = load_raw_lexical_evidence_jsonl(lexical_payload)
    propositions = load_raw_proposition_jsonl(proposition_payload)
    qualifications = load_raw_qualification_jsonl(qualification_payload)
    by_observation = {item.observation_id: item for item in observations}
    if len(by_observation) != len(observations):
        raise RawT1TrainingPackError("observation_id 不得重复")
    by_proposition = {item.observation_id: item for item in propositions}
    by_qualification = {item.observation_id: item for item in qualifications}
    if (len(by_proposition) != len(propositions)
            or len(by_qualification) != len(qualifications)):
        raise RawT1TrainingPackError("proposition/qualification observation_id 不得重复")
    lexical_by_observation: dict[str, list[RawLexicalEvidence]] = {}
    for item in lexical:
        if item.observation_id not in by_observation:
            raise RawT1TrainingPackError("lexical evidence 指向未知 observation")
        lexical_by_observation.setdefault(item.observation_id, []).append(item)
    cases: list[RawT1TrainingCase] = []
    negatives: list[RawT1NegativeWitness] = []
    namespace = None
    for observation in observations:
        obs_id = observation.observation_id
        proposition = by_proposition.get(obs_id)
        qualification = by_qualification.get(obs_id)
        if proposition is None or qualification is None:
            raise RawT1TrainingPackError("observation 缺 proposition 或 qualification")
        lexical_items = tuple(lexical_by_observation.get(obs_id, ()))
        if namespace is None:
            namespace = observation.source_namespace
        if observation.source_namespace != namespace:
            raise RawT1TrainingPackError("pack source_namespace 漂移")
        if observation.split == "negative":
            try:
                binding = bind_raw_proposition_relation(
                    observation, lexical_items, proposition)
            except (RawPropositionEvidenceError, RawLexicalEvidenceError):
                negatives.append(RawT1NegativeWitness(
                    observation, lexical_items, proposition, qualification,
                    RAW_T1_NEGATIVE_BINDING_REJECTED,
                ))
                continue
            try:
                consume_raw_proposition_relation(binding, qualification)
            except RawPropositionConsumerError:
                negatives.append(RawT1NegativeWitness(
                    observation, lexical_items, proposition, qualification,
                    RAW_T1_NEGATIVE_CONSUMER_REJECTED,
                ))
                continue
            raise RawT1TrainingPackError(
                "negative constituent 意外闭合，不能冒充 negative witness")
        if observation.split not in RAW_T1_TRAINING_SPLITS:
            raise RawT1TrainingPackError("observation split 未注册")
        try:
            binding = bind_raw_proposition_relation(
                observation, lexical_items, proposition)
            consumer = consume_raw_proposition_relation(binding, qualification)
        except (RawPropositionEvidenceError, RawLexicalEvidenceError,
                RawPropositionConsumerError) as error:
            raise RawT1TrainingPackError(
                f"{observation.split} constituent 闭包失败") from error
        cases.append(RawT1TrainingCase(
            observation, lexical_items, proposition, binding, qualification, consumer))
    if namespace is None:
        raise RawT1TrainingPackError("pack observation 不能为空")
    if not negatives:
        raise RawT1TrainingPackError("pack 缺 negative witness")
    return RawT1TrainingPack(
        namespace, RAW_TEXT_OBSERVATION_LICENSE,
        tuple(sorted(cases, key=lambda item: item.observation_id)),
        tuple(sorted(negatives, key=lambda item: item.observation.observation_id)),
    )


__all__ = [
    "RAW_T1_NEGATIVE_BINDING_REJECTED", "RAW_T1_NEGATIVE_CONSUMER_REJECTED",
    "RAW_T1_PACK_CASE_RECORD_V1", "RAW_T1_PACK_NEGATIVE_RECORD_V1",
    "RAW_T1_TRAINING_PACK_PROTOCOL_V1", "RawT1NegativeWitness",
    "RawT1TrainingCase", "RawT1TrainingPack", "RawT1TrainingPackError",
    "assemble_raw_t1_training_pack",
]
