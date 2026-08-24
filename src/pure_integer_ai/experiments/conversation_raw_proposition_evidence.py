"""T1-G2：由显式 lexical evidence 授权的 proposition/relation annotation。

此模块只验证关系标签、参数顺序和 evidence 链是否真实绑定到 T1-G1 的物理 units。
relation_kind 是上游合同中的开放标签；本模块不从中文字符串、共现或模板猜测命题。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
    RawLexicalEvidenceBinding,
    RawLexicalEvidenceError,
    bind_raw_lexical_evidence,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RAW_TEXT_OBSERVATION_LICENSE,
    RAW_TEXT_OBSERVATION_SPLITS,
    RawTextObservation,
    RawTextObservationError,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


RAW_PROPOSITION_EVIDENCE_PROTOCOL_V1 = 1
RAW_PROPOSITION_EVIDENCE_RECORD_KIND = "DLG_RAW_PROPOSITION_RELATION_EVIDENCE_V1"
RAW_PROPOSITION_EVIDENCE_LICENSE = RAW_TEXT_OBSERVATION_LICENSE
_RECORD_FIELDS = frozenset({
    "arguments", "authority", "context_id", "family_id", "license_id",
    "observation_id", "proposition_id", "protocol_version", "record_kind",
    "relation_kind", "source_id", "source_namespace", "split",
})
_ARGUMENT_FIELDS = frozenset({"evidence_id", "order", "role", "unit_id"})


class RawPropositionEvidenceError(ValueError):
    """命题/关系 annotation 或 evidence 链不满足合同。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawPropositionEvidenceError(f"{where} 必须是无首尾空白的非空字符串")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise RawPropositionEvidenceError(f"{where} 含非 Unicode scalar")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise RawPropositionEvidenceError(f"{where} 必须是正严格整数")
    return value


@dataclass(frozen=True, slots=True)
class RawRelationArgument:
    """命题关系中的一个有序参数，仅引用已存在的 evidence/unit。"""

    evidence_id: str
    unit_id: str
    role: str
    order: int

    def __post_init__(self) -> None:
        _text(self.evidence_id, "argument.evidence_id")
        _text(self.unit_id, "argument.unit_id")
        _text(self.role, "argument.role")
        _positive(self.order, "argument.order")

    def canonical_record(self) -> tuple[int, ...]:
        """返回参数的纯整数 record。"""
        result: list[int] = []
        for value in (self.evidence_id, self.unit_id, self.role):
            scalars = tuple(ord(item) for item in value)
            result.extend((len(scalars), *scalars))
        result.append(self.order)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawPropositionRelationEvidence:
    """一条显式 proposition/relation annotation。"""

    proposition_id: str
    observation_id: str
    source_id: str
    context_id: str
    family_id: str
    source_namespace: str
    split: str
    relation_kind: str
    authority: str
    arguments: tuple[RawRelationArgument, ...]

    def __post_init__(self) -> None:
        for name in (
                "proposition_id", "observation_id", "source_id", "context_id",
                "family_id", "source_namespace", "relation_kind", "authority"):
            _text(getattr(self, name), f"proposition.{name}")
        if self.split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawPropositionEvidenceError("proposition.split 未注册")
        if not isinstance(self.arguments, tuple) or not self.arguments:
            raise RawPropositionEvidenceError("proposition.arguments 不能为空")
        if any(type(item) is not RawRelationArgument for item in self.arguments):
            raise TypeError("proposition.arguments 类型错误")
        if tuple(item.order for item in self.arguments) != tuple(
                range(1, len(self.arguments) + 1)):
            raise RawPropositionEvidenceError("arguments.order 必须从 1 连续递增")
        if len({item.evidence_id for item in self.arguments}) != len(self.arguments):
            raise RawPropositionEvidenceError("同一 evidence 不得重复作为参数")
        if len({item.unit_id for item in self.arguments}) != len(self.arguments):
            raise RawPropositionEvidenceError("同一 unit 不得重复作为参数")

    def canonical_record(self) -> tuple[int, ...]:
        """返回不携带宿主对象身份的纯整数关系 record。"""
        result = [RAW_PROPOSITION_EVIDENCE_PROTOCOL_V1]
        for value in (
                self.proposition_id, self.observation_id, self.source_id,
                self.context_id, self.family_id, self.source_namespace,
                self.split, self.relation_kind, self.authority):
            scalars = tuple(ord(item) for item in value)
            result.extend((len(scalars), *scalars))
        for argument in self.arguments:
            record = argument.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawPropositionRelationBinding:
    """关系 annotation 与 lexical bindings 的只读闭包。"""

    proposition_id: str
    relation_kind: str
    arguments: tuple[RawLexicalEvidenceBinding, ...]
    canonical_record: tuple[int, ...]


def bind_raw_proposition_relation(
        observation: RawTextObservation,
        lexical_evidence: Iterable[RawLexicalEvidence],
        proposition: RawPropositionRelationEvidence,
        ) -> RawPropositionRelationBinding:
    """验证命题参数逐一引用同一 observation 的显式 lexical evidence。"""
    if not isinstance(observation, RawTextObservation):
        raise TypeError("需要 RawTextObservation")
    if not isinstance(proposition, RawPropositionRelationEvidence):
        raise TypeError("需要 RawPropositionRelationEvidence")
    if (proposition.observation_id != observation.observation_id
            or proposition.source_id != observation.source_id
            or proposition.context_id != observation.context_id
            or proposition.family_id != observation.family_id
            or proposition.source_namespace != observation.source_namespace
            or proposition.split != observation.split):
        raise RawPropositionEvidenceError("proposition 与 observation identity 不一致")
    evidence_by_id = {}
    for evidence in lexical_evidence:
        if evidence.evidence_id in evidence_by_id:
            raise RawPropositionEvidenceError("evidence_id 不得重复")
        evidence_by_id[evidence.evidence_id] = bind_raw_lexical_evidence(
            observation, evidence)
    if not evidence_by_id:
        raise RawPropositionEvidenceError("proposition 缺少 lexical evidence")
    bindings = []
    for argument in proposition.arguments:
        binding = evidence_by_id.get(argument.evidence_id)
        if binding is None:
            raise RawPropositionEvidenceError("argument 引用了未知 evidence")
        if binding.unit_id != argument.unit_id:
            raise RawPropositionEvidenceError("argument evidence/unit identity 漂移")
        bindings.append(binding)
    return RawPropositionRelationBinding(
        proposition.proposition_id, proposition.relation_kind, tuple(bindings),
        proposition.canonical_record(),
    )


def raw_proposition_to_json_object(
        proposition: RawPropositionRelationEvidence,
        ) -> dict[str, Any]:
    """把 proposition annotation 投影为规范 JSON object。"""
    if not isinstance(proposition, RawPropositionRelationEvidence):
        raise TypeError("需要 RawPropositionRelationEvidence")
    return {
        "arguments": [
            {"evidence_id": item.evidence_id, "order": item.order,
             "role": item.role, "unit_id": item.unit_id}
            for item in proposition.arguments
        ],
        "authority": proposition.authority,
        "context_id": proposition.context_id,
        "family_id": proposition.family_id,
        "license_id": RAW_PROPOSITION_EVIDENCE_LICENSE,
        "observation_id": proposition.observation_id,
        "proposition_id": proposition.proposition_id,
        "protocol_version": RAW_PROPOSITION_EVIDENCE_PROTOCOL_V1,
        "record_kind": RAW_PROPOSITION_EVIDENCE_RECORD_KIND,
        "relation_kind": proposition.relation_kind,
        "source_id": proposition.source_id,
        "source_namespace": proposition.source_namespace,
        "split": proposition.split,
    }


def compile_raw_proposition_json(
        proposition: RawPropositionRelationEvidence,
        ) -> bytes:
    """返回单条规范 proposition/relation JSONL。"""
    return canonical_json_line(raw_proposition_to_json_object(proposition))


def parse_raw_proposition_record(value: Any) -> RawPropositionRelationEvidence:
    """严格恢复 proposition annotation，拒绝字段和版本漂移。"""
    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        raise RawPropositionEvidenceError("proposition evidence 字段集合漂移")
    if value["record_kind"] != RAW_PROPOSITION_EVIDENCE_RECORD_KIND:
        raise RawPropositionEvidenceError("proposition evidence record kind 不匹配")
    if value["protocol_version"] != RAW_PROPOSITION_EVIDENCE_PROTOCOL_V1:
        raise RawPropositionEvidenceError("proposition evidence protocol version 不匹配")
    if value["license_id"] != RAW_PROPOSITION_EVIDENCE_LICENSE:
        raise RawPropositionEvidenceError("proposition evidence 许可不匹配")
    raw_arguments = value["arguments"]
    if not isinstance(raw_arguments, list) or not raw_arguments:
        raise RawPropositionEvidenceError("record.arguments 必须是非空列表")
    arguments = []
    for index, item in enumerate(raw_arguments):
        if not isinstance(item, dict) or set(item) != _ARGUMENT_FIELDS:
            raise RawPropositionEvidenceError(
                f"record.arguments[{index}] 字段集合漂移")
        arguments.append(RawRelationArgument(
            _text(item["evidence_id"], f"argument[{index}].evidence_id"),
            _text(item["unit_id"], f"argument[{index}].unit_id"),
            _text(item["role"], f"argument[{index}].role"),
            _positive(item["order"], f"argument[{index}].order"),
        ))
    proposition = RawPropositionRelationEvidence(
        _text(value["proposition_id"], "record.proposition_id"),
        _text(value["observation_id"], "record.observation_id"),
        _text(value["source_id"], "record.source_id"),
        _text(value["context_id"], "record.context_id"),
        _text(value["family_id"], "record.family_id"),
        _text(value["source_namespace"], "record.source_namespace"),
        _text(value["split"], "record.split"),
        _text(value["relation_kind"], "record.relation_kind"),
        _text(value["authority"], "record.authority"), tuple(arguments),
    )
    if raw_proposition_to_json_object(proposition) != value:
        raise RawPropositionEvidenceError("proposition 规范回读漂移")
    return proposition


def load_raw_proposition_jsonl(
        payload: bytes,
        *,
        expected_split: str | None = None,
        ) -> tuple[RawPropositionRelationEvidence, ...]:
    """严格回读 proposition JSONL，并可锁定一个 split。"""
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise RawPropositionEvidenceError("proposition JSONL 必须以换行结束")
    rows = payload.splitlines(keepends=True)
    if not rows:
        raise RawPropositionEvidenceError("proposition JSONL 不能为空")
    result = []
    for index, line in enumerate(rows):
        if line == b"\n" or not line.endswith(b"\n"):
            raise RawPropositionEvidenceError(f"proposition JSONL 第 {index} 行无效")
        value = parse_canonical_json_bytes(line[:-1], require_object=True)
        if canonical_json_line(value) != line:
            raise RawPropositionEvidenceError(f"proposition JSONL 第 {index} 行非规范")
        result.append(parse_raw_proposition_record(value))
    if len({item.proposition_id for item in result}) != len(result):
        raise RawPropositionEvidenceError("proposition_id 不得重复")
    if expected_split is not None:
        if expected_split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawPropositionEvidenceError("expected_split 未注册")
        if any(item.split != expected_split for item in result):
            raise RawPropositionEvidenceError("proposition JSONL 混入了不允许的 split")
    return tuple(result)


__all__ = [
    "RAW_PROPOSITION_EVIDENCE_LICENSE", "RAW_PROPOSITION_EVIDENCE_PROTOCOL_V1",
    "RAW_PROPOSITION_EVIDENCE_RECORD_KIND", "RawPropositionEvidenceError",
    "RawPropositionRelationBinding", "RawPropositionRelationEvidence",
    "RawRelationArgument", "bind_raw_proposition_relation",
    "compile_raw_proposition_json", "load_raw_proposition_jsonl",
    "parse_raw_proposition_record", "raw_proposition_to_json_object",
]
