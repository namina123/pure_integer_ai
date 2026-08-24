"""T1-G15：raw T1 training pack 的只读规模与失败边界盘点。

本模块不学习、不写盘、不改变 pack。它把 constituent 数量、split、negative 失败阶段和
canonical integer pack 的 SHA-256 投影为一个稳定 census，供训练切片扩展前审计规模与重放身份。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.conversation_raw_t1_training_pack import (
    RAW_T1_NEGATIVE_BINDING_REJECTED,
    RAW_T1_NEGATIVE_CONSUMER_REJECTED,
    RawT1TrainingPack,
)


RAW_T1_TRAINING_CENSUS_PROTOCOL_V1 = 1


class RawT1TrainingCensusError(ValueError):
    """训练 pack census 输入或摘要不满足冻结边界。"""


def _u64(value: int, where: str) -> bytes:
    if type(value) is not int or value < 0 or value >= 1 << 64:
        raise RawT1TrainingCensusError(f"{where} 必须是 0..2^64-1 的严格整数")
    return value.to_bytes(8, "big")


def _hash_integer_record(record: tuple[int, ...]) -> tuple[int, ...]:
    if any(type(value) is not int or value < 0 for value in record):
        raise RawT1TrainingCensusError("pack canonical record 含非法整数")
    payload = b"".join(_u64(value, "pack canonical record") for value in record)
    return tuple(hashlib.sha256(payload).digest())


# object-model: value; representation=struct; interop=T1-G15
@dataclass(frozen=True, slots=True)
class RawT1TrainingCensus:
    """一个不读取外部状态的 pack 规模/身份摘要。"""

    source_namespace: str
    license_id: str
    case_count: int
    negative_count: int
    lexical_evidence_count: int
    split_counts: tuple[tuple[str, int], ...]
    negative_stage_counts: tuple[tuple[int, int], ...]
    canonical_sha256_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.source_namespace) is not str or not self.source_namespace:
            raise RawT1TrainingCensusError("source_namespace 非法")
        if type(self.license_id) is not str or not self.license_id:
            raise RawT1TrainingCensusError("license_id 非法")
        for name in ("case_count", "negative_count", "lexical_evidence_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise RawT1TrainingCensusError(f"{name} 必须是非负严格整数")
        if not self.split_counts or not self.negative_stage_counts:
            raise RawT1TrainingCensusError("census counts 不能为空")
        if len(self.canonical_sha256_u8) != 32 or any(
                type(value) is not int or not 0 <= value <= 255
                for value in self.canonical_sha256_u8):
            raise RawT1TrainingCensusError("canonical SHA-256 必须是 32 个 u8")

    @property
    def canonical_integer_record(self) -> tuple[int, ...]:
        result = [RAW_T1_TRAINING_CENSUS_PROTOCOL_V1,
                  self.case_count, self.negative_count,
                  self.lexical_evidence_count]
        result.append(len(self.split_counts))
        for split, count in self.split_counts:
            result.extend((len(split), *(ord(item) for item in split), count))
        result.append(len(self.negative_stage_counts))
        for stage, count in self.negative_stage_counts:
            result.extend((stage, count))
        result.extend(self.canonical_sha256_u8)
        return tuple(result)


def build_raw_t1_training_census(pack: RawT1TrainingPack) -> RawT1TrainingCensus:
    """对已闭合 pack 生成只读 census；不重新解析 constituent 或写入任何介质。"""
    if type(pack) is not RawT1TrainingPack:
        raise TypeError("需要 RawT1TrainingPack")
    stage_counts = {
        RAW_T1_NEGATIVE_BINDING_REJECTED: 0,
        RAW_T1_NEGATIVE_CONSUMER_REJECTED: 0,
    }
    for item in pack.negatives:
        stage_counts[item.failure_stage] += 1
    return RawT1TrainingCensus(
        source_namespace=pack.source_namespace,
        license_id=pack.license_id,
        case_count=len(pack.cases),
        negative_count=len(pack.negatives),
        lexical_evidence_count=sum(
            len(item.lexical_evidence) for item in pack.cases
        ) + sum(len(item.lexical_evidence) for item in pack.negatives),
        split_counts=pack.split_counts,
        negative_stage_counts=tuple(
            (stage, stage_counts[stage])
            for stage in (RAW_T1_NEGATIVE_BINDING_REJECTED,
                          RAW_T1_NEGATIVE_CONSUMER_REJECTED)
        ),
        canonical_sha256_u8=_hash_integer_record(pack.canonical_record()),
    )


__all__ = [
    "RAW_T1_TRAINING_CENSUS_PROTOCOL_V1",
    "RawT1TrainingCensus",
    "RawT1TrainingCensusError",
    "build_raw_t1_training_census",
]
