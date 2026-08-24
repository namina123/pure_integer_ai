"""T1-G3：命题关系的 source/evidence qualification 与只读消费边界。

本模块不生成事实文本，也不写入会话或记忆。它只把已闭合的 proposition relation 与
显式 qualification 对齐，并将 SUPPORTED/UNKNOWN/CONFLICT 映射到 ANSWER/UNKNOWN/
CLARIFY 的上游 response-act 义务，供后续表层组织层消费。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionEvidenceError,
    RawPropositionRelationBinding,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RAW_TEXT_OBSERVATION_LICENSE,
    RAW_TEXT_OBSERVATION_SPLITS,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


RAW_PROPOSITION_CONSUMER_PROTOCOL_V1 = 1
RAW_PROPOSITION_QUALIFICATION_RECORD_KIND = (
    "DLG_RAW_PROPOSITION_QUALIFICATION_V1")
RAW_PROPOSITION_CONSUMER_RECORD_KIND = "DLG_RAW_PROPOSITION_CONSUMER_V1"
RAW_PROPOSITION_CONSUMER_LICENSE = RAW_TEXT_OBSERVATION_LICENSE
QUALIFICATION_STATES = frozenset({"SUPPORTED", "UNKNOWN", "CONFLICT"})
_QUALIFICATION_FIELDS = frozenset({
    "authority", "context_id", "evidence_ids", "family_id", "license_id",
    "observation_id", "proposition_id", "protocol_version", "qualification_id",
    "record_kind", "reason_id", "source_id", "source_namespace", "split", "state",
})


class RawPropositionConsumerError(ValueError):
    """qualification 或 proposition 消费越过只读 response-act 边界。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawPropositionConsumerError(f"{where} 必须是无首尾空白的非空字符串")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise RawPropositionConsumerError(f"{where} 含非 Unicode scalar")
    return value


@dataclass(frozen=True, slots=True)
class RawPropositionQualification:
    """一条由 source/evidence 侧显式给出的资格状态。"""

    qualification_id: str
    proposition_id: str
    observation_id: str
    source_id: str
    context_id: str
    family_id: str
    source_namespace: str
    split: str
    state: str
    reason_id: str
    evidence_ids: tuple[str, ...]
    authority: str

    def __post_init__(self) -> None:
        for name in (
                "qualification_id", "proposition_id", "observation_id", "source_id",
                "context_id", "family_id", "source_namespace", "reason_id", "authority"):
            _text(getattr(self, name), f"qualification.{name}")
        if self.split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawPropositionConsumerError("qualification.split 未注册")
        if self.state not in QUALIFICATION_STATES:
            raise RawPropositionConsumerError("qualification.state 未注册")
        if not isinstance(self.evidence_ids, tuple) or not self.evidence_ids:
            raise RawPropositionConsumerError("qualification.evidence_ids 不能为空")
        if any(not isinstance(item, str) or not item or item.strip() != item
               for item in self.evidence_ids):
            raise RawPropositionConsumerError("qualification.evidence_ids 含非法 id")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise RawPropositionConsumerError("qualification.evidence_ids 不得重复")
        if self.state == "CONFLICT" and len(self.evidence_ids) < 2:
            raise RawPropositionConsumerError("CONFLICT 至少需要两个独立 evidence")

    def canonical_record(self) -> tuple[int, ...]:
        """返回不承载事实文本的纯整数 qualification record。"""
        result = [RAW_PROPOSITION_CONSUMER_PROTOCOL_V1]
        for value in (
                self.qualification_id, self.proposition_id, self.observation_id,
                self.source_id, self.context_id, self.family_id, self.source_namespace,
                self.split, self.state, self.reason_id, self.authority):
            scalars = tuple(ord(item) for item in value)
            result.extend((len(scalars), *scalars))
        result.append(len(self.evidence_ids))
        for value in self.evidence_ids:
            scalars = tuple(ord(item) for item in value)
            result.extend((len(scalars), *scalars))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawPropositionConsumerResult:
    """只读 response-act obligation，不含 surface 或 claim 文本。"""

    proposition_id: str
    qualification_id: str
    observation_id: str
    source_id: str
    context_id: str
    family_id: str
    state: str
    response_act: str
    evidence_ids: tuple[str, ...]
    integer_record: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.proposition_id, "consumer.proposition_id")
        _text(self.qualification_id, "consumer.qualification_id")
        _text(self.observation_id, "consumer.observation_id")
        _text(self.source_id, "consumer.source_id")
        _text(self.context_id, "consumer.context_id")
        _text(self.family_id, "consumer.family_id")
        if self.state not in QUALIFICATION_STATES:
            raise RawPropositionConsumerError("consumer.state 未注册")
        expected = {"SUPPORTED": "ANSWER", "UNKNOWN": "UNKNOWN",
                    "CONFLICT": "CLARIFY"}[self.state]
        if self.response_act != expected:
            raise RawPropositionConsumerError("consumer response_act 与 state 不一致")
        if (not isinstance(self.evidence_ids, tuple) or not self.evidence_ids
                or any(not isinstance(item, str) or not item for item in self.evidence_ids)):
            raise RawPropositionConsumerError("consumer.evidence_ids 非法")
        if (not isinstance(self.integer_record, tuple)
                or any(type(item) is not int for item in self.integer_record)):
            raise RawPropositionConsumerError("consumer.integer_record 必须是整数 tuple")


def raw_qualification_to_json_object(
        qualification: RawPropositionQualification,
        ) -> dict[str, Any]:
    """把 qualification 投影为规范 JSON object。"""
    if not isinstance(qualification, RawPropositionQualification):
        raise TypeError("需要 RawPropositionQualification")
    return {
        "authority": qualification.authority,
        "context_id": qualification.context_id,
        "evidence_ids": list(qualification.evidence_ids),
        "family_id": qualification.family_id,
        "license_id": RAW_PROPOSITION_CONSUMER_LICENSE,
        "observation_id": qualification.observation_id,
        "proposition_id": qualification.proposition_id,
        "protocol_version": RAW_PROPOSITION_CONSUMER_PROTOCOL_V1,
        "qualification_id": qualification.qualification_id,
        "record_kind": RAW_PROPOSITION_QUALIFICATION_RECORD_KIND,
        "reason_id": qualification.reason_id,
        "source_id": qualification.source_id,
        "source_namespace": qualification.source_namespace,
        "split": qualification.split,
        "state": qualification.state,
    }


def compile_raw_qualification_json(
        qualification: RawPropositionQualification,
        ) -> bytes:
    """返回单条 qualification 规范 JSONL。"""
    return canonical_json_line(raw_qualification_to_json_object(qualification))


def parse_raw_qualification_record(value: Any) -> RawPropositionQualification:
    """严格恢复 qualification，拒绝字段、许可或版本漂移。"""
    if not isinstance(value, dict) or set(value) != _QUALIFICATION_FIELDS:
        raise RawPropositionConsumerError("qualification 字段集合漂移")
    if value["record_kind"] != RAW_PROPOSITION_QUALIFICATION_RECORD_KIND:
        raise RawPropositionConsumerError("qualification record kind 不匹配")
    if value["protocol_version"] != RAW_PROPOSITION_CONSUMER_PROTOCOL_V1:
        raise RawPropositionConsumerError("qualification protocol version 不匹配")
    if value["license_id"] != RAW_PROPOSITION_CONSUMER_LICENSE:
        raise RawPropositionConsumerError("qualification 许可不匹配")
    evidence_ids = value["evidence_ids"]
    if not isinstance(evidence_ids, list):
        raise RawPropositionConsumerError("qualification.evidence_ids 必须是 list")
    qualification = RawPropositionQualification(
        _text(value["qualification_id"], "record.qualification_id"),
        _text(value["proposition_id"], "record.proposition_id"),
        _text(value["observation_id"], "record.observation_id"),
        _text(value["source_id"], "record.source_id"),
        _text(value["context_id"], "record.context_id"),
        _text(value["family_id"], "record.family_id"),
        _text(value["source_namespace"], "record.source_namespace"),
        _text(value["split"], "record.split"),
        _text(value["state"], "record.state"),
        _text(value["reason_id"], "record.reason_id"),
        tuple(_text(item, "record.evidence_ids[]") for item in evidence_ids),
        _text(value["authority"], "record.authority"),
    )
    if raw_qualification_to_json_object(qualification) != value:
        raise RawPropositionConsumerError("qualification 规范回读漂移")
    return qualification


def load_raw_qualification_jsonl(
        payload: bytes,
        *,
        expected_split: str | None = None,
        ) -> tuple[RawPropositionQualification, ...]:
    """严格回读 qualification JSONL，并可锁定一个 split。"""
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise RawPropositionConsumerError("qualification JSONL 必须以换行结束")
    rows = payload.splitlines(keepends=True)
    if not rows:
        raise RawPropositionConsumerError("qualification JSONL 不能为空")
    result = []
    for index, line in enumerate(rows):
        if line == b"\n" or not line.endswith(b"\n"):
            raise RawPropositionConsumerError(f"qualification JSONL 第 {index} 行无效")
        value = parse_canonical_json_bytes(line[:-1], require_object=True)
        if canonical_json_line(value) != line:
            raise RawPropositionConsumerError(f"qualification JSONL 第 {index} 行非规范")
        result.append(parse_raw_qualification_record(value))
    if len({item.qualification_id for item in result}) != len(result):
        raise RawPropositionConsumerError("qualification_id 不得重复")
    if expected_split is not None:
        if expected_split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawPropositionConsumerError("expected_split 未注册")
        if any(item.split != expected_split for item in result):
            raise RawPropositionConsumerError("qualification JSONL 混入了不允许的 split")
    return tuple(result)


def consume_raw_proposition_relation(
        binding: RawPropositionRelationBinding,
        qualification: RawPropositionQualification,
        ) -> RawPropositionConsumerResult:
    """只读消费 qualified relation，状态不足时选择 UNKNOWN/CLARIFY。"""
    if not isinstance(binding, RawPropositionRelationBinding):
        raise TypeError("需要 RawPropositionRelationBinding")
    if not isinstance(qualification, RawPropositionQualification):
        raise TypeError("需要 RawPropositionQualification")
    if qualification.proposition_id != binding.proposition_id:
        raise RawPropositionConsumerError("qualification/proposition identity 漂移")
    expected_evidence = tuple(item.evidence_id for item in binding.arguments)
    if qualification.evidence_ids != expected_evidence:
        raise RawPropositionConsumerError("qualification evidence 链与 relation 参数不一致")
    response_act = {
        "SUPPORTED": "ANSWER", "UNKNOWN": "UNKNOWN", "CONFLICT": "CLARIFY",
    }[qualification.state]
    integer_record = (
        RAW_PROPOSITION_CONSUMER_PROTOCOL_V1,
        *qualification.canonical_record(),
        *binding.canonical_record,
    )
    return RawPropositionConsumerResult(
        binding.proposition_id, qualification.qualification_id,
        qualification.observation_id,
        qualification.source_id, qualification.context_id, qualification.family_id,
        qualification.state, response_act, qualification.evidence_ids,
        integer_record,
    )


__all__ = [
    "QUALIFICATION_STATES", "RAW_PROPOSITION_CONSUMER_LICENSE",
    "RAW_PROPOSITION_CONSUMER_PROTOCOL_V1", "RAW_PROPOSITION_CONSUMER_RECORD_KIND",
    "RAW_PROPOSITION_QUALIFICATION_RECORD_KIND", "RawPropositionConsumerError",
    "RawPropositionConsumerResult", "RawPropositionQualification",
    "compile_raw_qualification_json", "consume_raw_proposition_relation",
    "load_raw_qualification_jsonl", "parse_raw_qualification_record",
    "raw_qualification_to_json_object",
]
