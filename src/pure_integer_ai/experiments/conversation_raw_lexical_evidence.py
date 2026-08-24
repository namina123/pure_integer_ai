"""T1-G1：把原始文本 observation 绑定到独立 lexical/structural evidence。

本模块只验证上游显式标注与物理 span 的一致性。``unit_kind`` 是可扩展的结构标签，
不在这里被解释成词义、命题、关系、来源真值或对话行为；没有 evidence 的 span 不进入
后续学习闭环。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RAW_TEXT_OBSERVATION_LICENSE,
    RAW_TEXT_OBSERVATION_SPLITS,
    RawTextObservation,
    RawTextObservationError,
    RawTextSpanUnit,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


RAW_LEXICAL_EVIDENCE_PROTOCOL_V1 = 1
RAW_LEXICAL_EVIDENCE_RECORD_KIND = "DLG_RAW_LEXICAL_EVIDENCE_V1"
RAW_LEXICAL_EVIDENCE_LICENSE = RAW_TEXT_OBSERVATION_LICENSE
_EVIDENCE_FIELDS = frozenset({
    "authority", "context_id", "end_byte", "end_scalar", "evidence_id",
    "family_id", "license_id", "observation_id", "protocol_version",
    "record_kind", "source_id", "source_namespace", "split", "start_byte",
    "start_scalar", "unit_id", "unit_kind",
})


class RawLexicalEvidenceError(ValueError):
    """lexical/structural evidence 或其 observation 绑定越过合同。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawLexicalEvidenceError(f"{where} 必须是无首尾空白的非空字符串")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise RawLexicalEvidenceError(f"{where} 含非 Unicode scalar")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise RawLexicalEvidenceError(f"{where} 必须是非负严格整数")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise RawLexicalEvidenceError(f"{where} 必须是正严格整数")
    return value


@dataclass(frozen=True, slots=True)
class RawLexicalEvidence:
    """一条由上游显式给出的 lexical/structural span evidence。"""

    evidence_id: str
    observation_id: str
    source_id: str
    context_id: str
    family_id: str
    source_namespace: str
    split: str
    unit_id: str
    unit_kind: str
    authority: str
    start_scalar: int
    end_scalar: int
    start_byte: int
    end_byte: int

    def __post_init__(self) -> None:
        for name in (
                "evidence_id", "observation_id", "source_id", "context_id",
                "family_id", "source_namespace", "unit_id", "unit_kind",
                "authority"):
            _text(getattr(self, name), f"evidence.{name}")
        if self.split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawLexicalEvidenceError("evidence.split 未注册")
        _nonnegative(self.start_scalar, "evidence.start_scalar")
        _positive(self.end_scalar, "evidence.end_scalar")
        _nonnegative(self.start_byte, "evidence.start_byte")
        _positive(self.end_byte, "evidence.end_byte")
        if self.end_scalar <= self.start_scalar:
            raise RawLexicalEvidenceError("evidence scalar span 必须非空")
        if self.end_byte <= self.start_byte:
            raise RawLexicalEvidenceError("evidence byte span 必须非空")

    def canonical_record(self) -> tuple[int, ...]:
        """返回只含整数的可迁移 evidence record。"""
        result = [RAW_LEXICAL_EVIDENCE_PROTOCOL_V1]
        for value in (
                self.evidence_id, self.observation_id, self.source_id,
                self.context_id, self.family_id, self.source_namespace,
                self.split, self.unit_id, self.unit_kind, self.authority):
            scalars = tuple(ord(item) for item in value)
            result.extend((len(scalars), *scalars))
        result.extend((self.start_scalar, self.end_scalar,
                       self.start_byte, self.end_byte))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawLexicalEvidenceBinding:
    """evidence 与一个 raw observation unit 的只读绑定结果。"""

    observation_id: str
    evidence_id: str
    unit_id: str
    unit_kind: str
    unit_scalars: tuple[int, ...]
    unit_bytes: tuple[int, ...]
    evidence_record: tuple[int, ...]


def bind_raw_lexical_evidence(
        observation: RawTextObservation,
        evidence: RawLexicalEvidence,
        ) -> RawLexicalEvidenceBinding:
    """只按 identity/span 绑定 evidence，不从 raw surface 推断标签。"""
    if not isinstance(observation, RawTextObservation):
        raise TypeError("需要 RawTextObservation")
    if not isinstance(evidence, RawLexicalEvidence):
        raise TypeError("需要 RawLexicalEvidence")
    if (evidence.observation_id != observation.observation_id
            or evidence.source_id != observation.source_id
            or evidence.context_id != observation.context_id
            or evidence.family_id != observation.family_id
            or evidence.source_namespace != observation.source_namespace
            or evidence.split != observation.split):
        raise RawLexicalEvidenceError("evidence 与 observation identity 不一致")
    unit = next((item for item in observation.units
                 if item.unit_id == evidence.unit_id), None)
    if unit is None:
        raise RawLexicalEvidenceError("evidence.unit_id 未在 observation 中声明")
    if (evidence.start_scalar, evidence.end_scalar,
            evidence.start_byte, evidence.end_byte) != (
                unit.start_scalar, unit.end_scalar,
                unit.start_byte, unit.end_byte):
        raise RawLexicalEvidenceError("evidence span 与 observation unit 不一致")
    return RawLexicalEvidenceBinding(
        observation.observation_id, evidence.evidence_id, unit.unit_id, evidence.unit_kind,
        observation.unit_scalars(unit), observation.unit_bytes(unit),
        evidence.canonical_record(),
    )


def raw_lexical_evidence_to_json_object(
        evidence: RawLexicalEvidence,
        ) -> dict[str, Any]:
    """把 evidence 投影为无浮点、可规范回读的 JSON object。"""
    if not isinstance(evidence, RawLexicalEvidence):
        raise TypeError("需要 RawLexicalEvidence")
    return {
        "authority": evidence.authority,
        "context_id": evidence.context_id,
        "end_byte": evidence.end_byte,
        "end_scalar": evidence.end_scalar,
        "evidence_id": evidence.evidence_id,
        "family_id": evidence.family_id,
        "license_id": RAW_LEXICAL_EVIDENCE_LICENSE,
        "observation_id": evidence.observation_id,
        "protocol_version": RAW_LEXICAL_EVIDENCE_PROTOCOL_V1,
        "record_kind": RAW_LEXICAL_EVIDENCE_RECORD_KIND,
        "source_id": evidence.source_id,
        "source_namespace": evidence.source_namespace,
        "split": evidence.split,
        "start_byte": evidence.start_byte,
        "start_scalar": evidence.start_scalar,
        "unit_id": evidence.unit_id,
        "unit_kind": evidence.unit_kind,
    }


def compile_raw_lexical_evidence_json(evidence: RawLexicalEvidence) -> bytes:
    """返回单条以换行结束的规范 evidence JSONL。"""
    return canonical_json_line(raw_lexical_evidence_to_json_object(evidence))


def parse_raw_lexical_evidence_record(value: Any) -> RawLexicalEvidence:
    """严格恢复单条 evidence，拒绝字段或版本漂移。"""
    if not isinstance(value, dict) or set(value) != _EVIDENCE_FIELDS:
        raise RawLexicalEvidenceError("lexical evidence 字段集合漂移")
    if value["record_kind"] != RAW_LEXICAL_EVIDENCE_RECORD_KIND:
        raise RawLexicalEvidenceError("lexical evidence record kind 不匹配")
    if value["protocol_version"] != RAW_LEXICAL_EVIDENCE_PROTOCOL_V1:
        raise RawLexicalEvidenceError("lexical evidence protocol version 不匹配")
    if value["license_id"] != RAW_LEXICAL_EVIDENCE_LICENSE:
        raise RawLexicalEvidenceError("lexical evidence 许可不匹配")
    evidence = RawLexicalEvidence(
        _text(value["evidence_id"], "record.evidence_id"),
        _text(value["observation_id"], "record.observation_id"),
        _text(value["source_id"], "record.source_id"),
        _text(value["context_id"], "record.context_id"),
        _text(value["family_id"], "record.family_id"),
        _text(value["source_namespace"], "record.source_namespace"),
        _text(value["split"], "record.split"),
        _text(value["unit_id"], "record.unit_id"),
        _text(value["unit_kind"], "record.unit_kind"),
        _text(value["authority"], "record.authority"),
        _nonnegative(value["start_scalar"], "record.start_scalar"),
        _positive(value["end_scalar"], "record.end_scalar"),
        _nonnegative(value["start_byte"], "record.start_byte"),
        _positive(value["end_byte"], "record.end_byte"),
    )
    if raw_lexical_evidence_to_json_object(evidence) != value:
        raise RawLexicalEvidenceError("lexical evidence 规范回读漂移")
    return evidence


def load_raw_lexical_evidence_jsonl(
        payload: bytes,
        *,
        expected_split: str | None = None,
        ) -> tuple[RawLexicalEvidence, ...]:
    """严格回读 evidence JSONL，并可锁定单一 split。"""
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise RawLexicalEvidenceError("lexical evidence JSONL 必须以换行结束")
    rows = payload.splitlines(keepends=True)
    if not rows:
        raise RawLexicalEvidenceError("lexical evidence JSONL 不能为空")
    result = []
    for index, line in enumerate(rows):
        if line == b"\n" or not line.endswith(b"\n"):
            raise RawLexicalEvidenceError(f"lexical evidence JSONL 第 {index} 行无效")
        value = parse_canonical_json_bytes(line[:-1], require_object=True)
        if canonical_json_line(value) != line:
            raise RawLexicalEvidenceError(f"lexical evidence JSONL 第 {index} 行非规范")
        result.append(parse_raw_lexical_evidence_record(value))
    if len({item.evidence_id for item in result}) != len(result):
        raise RawLexicalEvidenceError("evidence_id 不得重复")
    if expected_split is not None:
        if expected_split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawLexicalEvidenceError("expected_split 未注册")
        if any(item.split != expected_split for item in result):
            raise RawLexicalEvidenceError("evidence JSONL 混入了不允许的 split")
    return tuple(result)


def compile_raw_lexical_evidence_pack(
        observations: Iterable[RawTextObservation],
        evidence: Iterable[RawLexicalEvidence],
        *,
        expected_split: str | None = None,
        ) -> tuple[RawLexicalEvidenceBinding, ...]:
    """绑定一个 evidence pack，并要求每个 observation unit 恰好一次。"""
    observations_tuple = tuple(observations)
    evidence_tuple = tuple(evidence)
    if not observations_tuple or not evidence_tuple:
        raise RawLexicalEvidenceError("observation/evidence pack 不能为空")
    by_id = {item.observation_id: item for item in observations_tuple}
    if len(by_id) != len(observations_tuple):
        raise RawLexicalEvidenceError("observation_id 不得重复")
    if expected_split is not None:
        if expected_split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawLexicalEvidenceError("expected_split 未注册")
        if any(item.split != expected_split for item in observations_tuple):
            raise RawLexicalEvidenceError("observation pack split 不一致")
    seen_units: set[tuple[str, str]] = set()
    bindings = []
    for item in evidence_tuple:
        observation = by_id.get(item.observation_id)
        if observation is None:
            raise RawLexicalEvidenceError("evidence 指向未知 observation")
        key = (item.observation_id, item.unit_id)
        if key in seen_units:
            raise RawLexicalEvidenceError("同一 unit 不得重复标注")
        seen_units.add(key)
        bindings.append(bind_raw_lexical_evidence(observation, item))
    expected_units = {
        (observation.observation_id, unit.unit_id)
        for observation in observations_tuple
        for unit in observation.units
    }
    if seen_units != expected_units:
        raise RawLexicalEvidenceError("evidence 必须覆盖且仅覆盖每个显式 unit")
    return tuple(bindings)


__all__ = [
    "RAW_LEXICAL_EVIDENCE_LICENSE", "RAW_LEXICAL_EVIDENCE_PROTOCOL_V1",
    "RAW_LEXICAL_EVIDENCE_RECORD_KIND", "RawLexicalEvidence",
    "RawLexicalEvidenceBinding", "RawLexicalEvidenceError",
    "bind_raw_lexical_evidence", "compile_raw_lexical_evidence_json",
    "compile_raw_lexical_evidence_pack", "load_raw_lexical_evidence_jsonl",
    "parse_raw_lexical_evidence_record", "raw_lexical_evidence_to_json_object",
]
