"""P3-Ia 自由文本层级、中心形成与长期召回的纯合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey


FORMAT_VERSION = 1
HIERARCHY_KINDS = ("PARAGRAPH", "PROPOSITION", "SECTION")
CANDIDATE_STATES = (
    "CONFLICT", "PROPOSED", "REFUTED", "SUPPORTED", "UNKNOWN")
EVIDENCE_KINDS = (
    "ALIAS", "CITATION", "HISTORY", "PARAPHRASE", "QUERY",
    "RAW_SPAN", "REFERS", "SITUATION", "STRUCTURE",
)
EVIDENCE_OWNER_KINDS = ("CANDIDATE", "HISTORY", "SOURCE")
CENTER_STATES = ("ACTIVE", "AMBIGUOUS", "CLARIFY", "PROPOSED", "REJECTED")
CENTER_TARGET_KINDS = (
    "HIERARCHY_CANDIDATE", "PROPOSITION", "SOURCE_REF")
RECALL_TARGET_KINDS = ("CITATION", "PROPOSITION", "SOURCE_REF")
RECALL_FAILURE_STATES = (
    "AMBIGUOUS", "BUDGET_EXHAUSTED", "CONFLICT", "NONE", "NOT_FOUND",
    "SOURCE_VERSION_MISMATCH", "UNAUTHORIZED",
)
RECALL_STOP_REASONS = (
    "AMBIGUOUS", "BUDGET_EXHAUSTED", "CONFLICT", "NOT_FOUND", "RESOLVED",
    "SOURCE_VERSION_MISMATCH", "UNAUTHORIZED",
)
SPLITS = ("adversarial", "dev", "held_out", "train", "wall")
VISIBLE_OWNER_KINDS = ("CANDIDATE", "HISTORY", "SOURCE")


class FreeTextHierarchyRecallContractError(RuntimeError):
    """自由文本合同出现身份、层级、权限、预算或 owner 泄漏。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise FreeTextHierarchyRecallContractError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise FreeTextHierarchyRecallContractError(f"{where} 不能为空")
    return value


def _enum(value: Any, allowed: tuple[str, ...], *, where: str) -> str:
    result = _text(value, where=where)
    if result not in allowed:
        raise FreeTextHierarchyRecallContractError(f"{where} 枚举非法")
    return result


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是非负严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是 0/1")
    return value


def _sha256(value: Any, *, where: str) -> str:
    result = _text(value, where=where)
    if (len(result) != 64
            or any(char not in "0123456789abcdef" for char in result)):
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是小写 SHA-256")
    return result


def _key(value: Any, *, where: str) -> StableRecordKey:
    if not isinstance(value, StableRecordKey):
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是 StableRecordKey")
    return value


def _key_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[StableRecordKey, ...]:
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是稳定键 tuple")
    if any(not isinstance(item, StableRecordKey) for item in value):
        raise FreeTextHierarchyRecallContractError(f"{where} 含非法稳定键")
    if value != tuple(sorted(set(value))):
        raise FreeTextHierarchyRecallContractError(f"{where} 必须排序去重")
    return value


def _source(value: Any, *, where: str) -> SourceRef:
    if not isinstance(value, SourceRef):
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是 SourceRef")
    return value


def _source_to_value(value: SourceRef) -> list[int]:
    return list(value.stable_key())


def _source_from_value(value: Any, *, where: str) -> SourceRef:
    if (not isinstance(value, list) or len(value) != 11
            or any(type(item) is not int for item in value)):
        raise FreeTextHierarchyRecallContractError(f"{where} SourceRef 键非法")
    try:
        return SourceRef.from_stable_key(tuple(value))
    except (TypeError, ValueError) as error:
        raise FreeTextHierarchyRecallContractError(
            f"{where} SourceRef 键非法") from error


def _key_from_value(value: Any, *, where: str) -> StableRecordKey:
    try:
        return StableRecordKey.from_value(value, where=where)
    except ValueError as error:
        raise FreeTextHierarchyRecallContractError(f"{where} 稳定键非法") from error


def _keys_to_value(values: tuple[StableRecordKey, ...]) -> list[list[int]]:
    return [item.to_list() for item in values]


def _keys_from_value(value: Any, *, where: str) -> tuple[StableRecordKey, ...]:
    if not isinstance(value, list):
        raise FreeTextHierarchyRecallContractError(f"{where} 必须是列表")
    return tuple(_key_from_value(item, where=where) for item in value)


def _optional_key_to_value(value: StableRecordKey | None) -> list[int] | None:
    return None if value is None else value.to_list()


def _optional_key_from_value(value: Any, *, where: str) -> StableRecordKey | None:
    return None if value is None else _key_from_value(value, where=where)


def _relative_path(value: Any, *, where: str) -> str:
    result = _text(value, where=where)
    path = PurePosixPath(result)
    if (path.is_absolute() or ".." in path.parts or "\\" in result
            or path.as_posix() != result):
        raise FreeTextHierarchyRecallContractError(
            f"{where} 必须是安全 POSIX 相对路径")
    return result


@dataclass(frozen=True)
class SourceDocument:
    """一个原始文档及其 source/version/scope/ACL 边界。"""

    source_ref: SourceRef
    document_key: StableRecordKey
    scope_key: StableRecordKey
    acl_key: StableRecordKey
    raw_text: str
    raw_sha256: str

    def __post_init__(self) -> None:
        _source(self.source_ref, where="document source_ref")
        for name in ("document_key", "scope_key", "acl_key"):
            _key(getattr(self, name), where=f"document {name}")
        _text(self.raw_text, where="document raw_text")
        _sha256(self.raw_sha256, where="document raw_sha256")
        actual = hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()
        if actual != self.raw_sha256:
            raise FreeTextHierarchyRecallContractError("raw_text 与 raw_sha256 不一致")

    def to_dict(self) -> dict[str, Any]:
        return {
            "acl_key": self.acl_key.to_list(),
            "document_key": self.document_key.to_list(),
            "raw_sha256": self.raw_sha256,
            "raw_text": self.raw_text,
            "scope_key": self.scope_key.to_list(),
            "source_ref_key": _source_to_value(self.source_ref),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceDocument":
        raw = _exact(value, {
            "acl_key", "document_key", "raw_sha256", "raw_text", "scope_key",
            "source_ref_key",
        }, where="SourceDocument")
        return cls(
            _source_from_value(raw["source_ref_key"], where="source_ref_key"),
            _key_from_value(raw["document_key"], where="document_key"),
            _key_from_value(raw["scope_key"], where="scope_key"),
            _key_from_value(raw["acl_key"], where="acl_key"),
            raw["raw_text"],
            raw["raw_sha256"],
        )


@dataclass(frozen=True, order=True)
class AbsoluteSpan:
    """绑定完整 SourceRef 的半开绝对字符范围。"""

    span_key: StableRecordKey
    source_ref: SourceRef
    start: int
    end: int

    def __post_init__(self) -> None:
        _key(self.span_key, where="span key")
        _source(self.source_ref, where="span source_ref")
        _nonnegative(self.start, where="span start")
        _positive(self.end, where="span end")
        if self.start >= self.end:
            raise FreeTextHierarchyRecallContractError("span 必须是非空半开范围")

    def validate_document(self, document: SourceDocument) -> None:
        if self.source_ref != document.source_ref:
            raise FreeTextHierarchyRecallContractError("span 跨 source/version 继承")
        if self.end > len(document.raw_text):
            raise FreeTextHierarchyRecallContractError("span 超出 raw 文档")

    def contains(self, other: "AbsoluteSpan") -> bool:
        return (self.source_ref == other.source_ref
                and self.start <= other.start and other.end <= self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end,
            "source_ref_key": _source_to_value(self.source_ref),
            "span_key": self.span_key.to_list(),
            "start": self.start,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AbsoluteSpan":
        raw = _exact(value, {
            "end", "source_ref_key", "span_key", "start",
        }, where="AbsoluteSpan")
        return cls(
            _key_from_value(raw["span_key"], where="span_key"),
            _source_from_value(raw["source_ref_key"], where="source_ref_key"),
            raw["start"],
            raw["end"],
        )


@dataclass(frozen=True, order=True)
class CandidateEvidence:
    """候选侧允许读取的一等 Evidence identity。"""

    evidence_key: StableRecordKey
    evidence_kind: str
    owner_kind: str
    source_ref: SourceRef
    span_key: StableRecordKey | None
    subject_key: StableRecordKey

    def __post_init__(self) -> None:
        _key(self.evidence_key, where="evidence key")
        _enum(self.evidence_kind, EVIDENCE_KINDS, where="evidence kind")
        _enum(self.owner_kind, EVIDENCE_OWNER_KINDS, where="evidence owner kind")
        _source(self.source_ref, where="evidence source_ref")
        if self.span_key is not None:
            _key(self.span_key, where="evidence span key")
        _key(self.subject_key, where="evidence subject key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_key": self.evidence_key.to_list(),
            "evidence_kind": self.evidence_kind,
            "owner_kind": self.owner_kind,
            "source_ref_key": _source_to_value(self.source_ref),
            "span_key": _optional_key_to_value(self.span_key),
            "subject_key": self.subject_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateEvidence":
        raw = _exact(value, {
            "evidence_key", "evidence_kind", "owner_kind", "source_ref_key",
            "span_key", "subject_key",
        }, where="CandidateEvidence")
        return cls(
            _key_from_value(raw["evidence_key"], where="evidence_key"),
            raw["evidence_kind"],
            raw["owner_kind"],
            _source_from_value(raw["source_ref_key"], where="source_ref_key"),
            _optional_key_from_value(raw["span_key"], where="span_key"),
            _key_from_value(raw["subject_key"], where="subject_key"),
        )


@dataclass(frozen=True, order=True)
class HierarchyCandidate:
    """section、paragraph 或 proposition 的候选层级节点。"""

    candidate_key: StableRecordKey
    candidate_kind: str
    span: AbsoluteSpan
    parent_key: StableRecordKey | None
    ordinal: int
    state: str
    evidence_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        _key(self.candidate_key, where="candidate key")
        _enum(self.candidate_kind, HIERARCHY_KINDS, where="candidate kind")
        if not isinstance(self.span, AbsoluteSpan):
            raise FreeTextHierarchyRecallContractError("candidate span 类型错误")
        if self.parent_key is not None:
            _key(self.parent_key, where="candidate parent key")
        _positive(self.ordinal, where="candidate ordinal")
        _enum(self.state, CANDIDATE_STATES, where="candidate state")
        _key_tuple(self.evidence_keys, where="candidate evidence keys")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key.to_list(),
            "candidate_kind": self.candidate_kind,
            "evidence_keys": _keys_to_value(self.evidence_keys),
            "ordinal": self.ordinal,
            "parent_key": _optional_key_to_value(self.parent_key),
            "span": self.span.to_dict(),
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HierarchyCandidate":
        raw = _exact(value, {
            "candidate_key", "candidate_kind", "evidence_keys", "ordinal",
            "parent_key", "span", "state",
        }, where="HierarchyCandidate")
        return cls(
            _key_from_value(raw["candidate_key"], where="candidate_key"),
            raw["candidate_kind"],
            AbsoluteSpan.from_dict(raw["span"]),
            _optional_key_from_value(raw["parent_key"], where="parent_key"),
            raw["ordinal"],
            raw["state"],
            _keys_from_value(raw["evidence_keys"], where="evidence_keys"),
        )


@dataclass(frozen=True)
class FreeTextQuery:
    """候选侧可见的自由文本 query 和 current-situation 边界。"""

    query_key: StableRecordKey
    source_ref: SourceRef
    scope_key: StableRecordKey
    acl_key: StableRecordKey
    situation_key: StableRecordKey
    raw_text: str
    raw_sha256: str

    def __post_init__(self) -> None:
        _key(self.query_key, where="query key")
        _source(self.source_ref, where="query source_ref")
        for name in ("scope_key", "acl_key", "situation_key"):
            _key(getattr(self, name), where=f"query {name}")
        _text(self.raw_text, where="query raw_text")
        _sha256(self.raw_sha256, where="query raw_sha256")
        if hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest() != self.raw_sha256:
            raise FreeTextHierarchyRecallContractError("query 与 raw_sha256 不一致")

    def to_dict(self) -> dict[str, Any]:
        return {
            "acl_key": self.acl_key.to_list(),
            "query_key": self.query_key.to_list(),
            "raw_sha256": self.raw_sha256,
            "raw_text": self.raw_text,
            "scope_key": self.scope_key.to_list(),
            "situation_key": self.situation_key.to_list(),
            "source_ref_key": _source_to_value(self.source_ref),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FreeTextQuery":
        raw = _exact(value, {
            "acl_key", "query_key", "raw_sha256", "raw_text", "scope_key",
            "situation_key", "source_ref_key",
        }, where="FreeTextQuery")
        return cls(
            _key_from_value(raw["query_key"], where="query_key"),
            _source_from_value(raw["source_ref_key"], where="source_ref_key"),
            _key_from_value(raw["scope_key"], where="scope_key"),
            _key_from_value(raw["acl_key"], where="acl_key"),
            _key_from_value(raw["situation_key"], where="situation_key"),
            raw["raw_text"],
            raw["raw_sha256"],
        )


@dataclass(frozen=True, order=True)
class CenterCandidate:
    """由 query/situation 提出的 activation-only center candidate。"""

    center_key: StableRecordKey
    query_key: StableRecordKey
    situation_key: StableRecordKey
    target_kind: str
    target_key: StableRecordKey
    state: str
    evidence_keys: tuple[StableRecordKey, ...]
    activation_only: int
    adopted: int

    def __post_init__(self) -> None:
        for name in ("center_key", "query_key", "situation_key", "target_key"):
            _key(getattr(self, name), where=f"center {name}")
        _enum(self.target_kind, CENTER_TARGET_KINDS, where="center target kind")
        _enum(self.state, CENTER_STATES, where="center state")
        _key_tuple(self.evidence_keys, where="center evidence keys")
        if _flag(self.activation_only, where="center activation_only") != 1:
            raise FreeTextHierarchyRecallContractError("center 必须 activation-only")
        if _flag(self.adopted, where="center adopted") != 0:
            raise FreeTextHierarchyRecallContractError("activation 不得等于 adoption")

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_only": self.activation_only,
            "adopted": self.adopted,
            "center_key": self.center_key.to_list(),
            "evidence_keys": _keys_to_value(self.evidence_keys),
            "query_key": self.query_key.to_list(),
            "situation_key": self.situation_key.to_list(),
            "state": self.state,
            "target_key": self.target_key.to_list(),
            "target_kind": self.target_kind,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CenterCandidate":
        raw = _exact(value, {
            "activation_only", "adopted", "center_key", "evidence_keys",
            "query_key", "situation_key", "state", "target_key", "target_kind",
        }, where="CenterCandidate")
        return cls(
            _key_from_value(raw["center_key"], where="center_key"),
            _key_from_value(raw["query_key"], where="query_key"),
            _key_from_value(raw["situation_key"], where="situation_key"),
            raw["target_kind"],
            _key_from_value(raw["target_key"], where="target_key"),
            raw["state"],
            _keys_from_value(raw["evidence_keys"], where="evidence_keys"),
            raw["activation_only"],
            raw["adopted"],
        )


@dataclass(frozen=True)
class RecallBudget:
    """一次 recall obligation 的硬读取和返回上限。"""

    max_index_gets: int
    max_segment_payload_gets: int
    max_segment_payload_bytes: int
    max_results: int

    def __post_init__(self) -> None:
        for name in (
                "max_index_gets", "max_segment_payload_gets",
                "max_segment_payload_bytes", "max_results"):
            _positive(getattr(self, name), where=f"recall budget {name}")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_index_gets": self.max_index_gets,
            "max_results": self.max_results,
            "max_segment_payload_bytes": self.max_segment_payload_bytes,
            "max_segment_payload_gets": self.max_segment_payload_gets,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecallBudget":
        raw = _exact(value, {
            "max_index_gets", "max_results", "max_segment_payload_bytes",
            "max_segment_payload_gets",
        }, where="RecallBudget")
        return cls(
            raw["max_index_gets"],
            raw["max_segment_payload_gets"],
            raw["max_segment_payload_bytes"],
            raw["max_results"],
        )


@dataclass(frozen=True)
class RecallObligation:
    """绑定 center、target、source/scope/ACL、预算和失败状态的召回请求。"""

    obligation_key: StableRecordKey
    center_key: StableRecordKey
    query_key: StableRecordKey
    target_kind: str
    target_key: StableRecordKey
    source_ref: SourceRef
    scope_key: StableRecordKey
    acl_key: StableRecordKey
    budget: RecallBudget
    failure_state: str

    def __post_init__(self) -> None:
        for name in ("obligation_key", "center_key", "query_key", "target_key"):
            _key(getattr(self, name), where=f"obligation {name}")
        _enum(self.target_kind, RECALL_TARGET_KINDS, where="obligation target kind")
        _source(self.source_ref, where="obligation source_ref")
        _key(self.scope_key, where="obligation scope key")
        _key(self.acl_key, where="obligation acl key")
        if not isinstance(self.budget, RecallBudget):
            raise FreeTextHierarchyRecallContractError("obligation budget 类型错误")
        _enum(self.failure_state, RECALL_FAILURE_STATES, where="failure state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "acl_key": self.acl_key.to_list(),
            "budget": self.budget.to_dict(),
            "center_key": self.center_key.to_list(),
            "failure_state": self.failure_state,
            "obligation_key": self.obligation_key.to_list(),
            "query_key": self.query_key.to_list(),
            "scope_key": self.scope_key.to_list(),
            "source_ref_key": _source_to_value(self.source_ref),
            "target_key": self.target_key.to_list(),
            "target_kind": self.target_kind,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecallObligation":
        raw = _exact(value, {
            "acl_key", "budget", "center_key", "failure_state",
            "obligation_key", "query_key", "scope_key", "source_ref_key",
            "target_key", "target_kind",
        }, where="RecallObligation")
        return cls(
            _key_from_value(raw["obligation_key"], where="obligation_key"),
            _key_from_value(raw["center_key"], where="center_key"),
            _key_from_value(raw["query_key"], where="query_key"),
            raw["target_kind"],
            _key_from_value(raw["target_key"], where="target_key"),
            _source_from_value(raw["source_ref_key"], where="source_ref_key"),
            _key_from_value(raw["scope_key"], where="scope_key"),
            _key_from_value(raw["acl_key"], where="acl_key"),
            RecallBudget.from_dict(raw["budget"]),
            raw["failure_state"],
        )


@dataclass(frozen=True, order=True)
class RecallCitation:
    """精确回读记录、SourceRef 与绝对 span 的引用。"""

    citation_key: StableRecordKey
    record_key: StableRecordKey
    source_ref: SourceRef
    span: AbsoluteSpan

    def __post_init__(self) -> None:
        _key(self.citation_key, where="citation key")
        _key(self.record_key, where="citation record key")
        _source(self.source_ref, where="citation source_ref")
        if not isinstance(self.span, AbsoluteSpan):
            raise FreeTextHierarchyRecallContractError("citation span 类型错误")
        if self.span.source_ref != self.source_ref:
            raise FreeTextHierarchyRecallContractError("citation span 跨 SourceRef")

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_key": self.citation_key.to_list(),
            "record_key": self.record_key.to_list(),
            "source_ref_key": _source_to_value(self.source_ref),
            "span": self.span.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecallCitation":
        raw = _exact(value, {
            "citation_key", "record_key", "source_ref_key", "span",
        }, where="RecallCitation")
        return cls(
            _key_from_value(raw["citation_key"], where="citation_key"),
            _key_from_value(raw["record_key"], where="record_key"),
            _source_from_value(raw["source_ref_key"], where="source_ref_key"),
            AbsoluteSpan.from_dict(raw["span"]),
        )


@dataclass(frozen=True)
class RecallReceipt:
    """index/segment、payload、citation、停止原因与越权读取收据。"""

    obligation_key: StableRecordKey
    index_keys: tuple[StableRecordKey, ...]
    segment_keys: tuple[StableRecordKey, ...]
    index_gets: int
    segment_payload_gets: int
    segment_payload_bytes: int
    result_keys: tuple[StableRecordKey, ...]
    citations: tuple[RecallCitation, ...]
    stop_reason: str
    unauthorized_payload_read_count: int

    def __post_init__(self) -> None:
        _key(self.obligation_key, where="receipt obligation key")
        _key_tuple(self.index_keys, where="receipt index keys", allow_empty=True)
        _key_tuple(self.segment_keys, where="receipt segment keys", allow_empty=True)
        _nonnegative(self.index_gets, where="receipt index gets")
        _nonnegative(self.segment_payload_gets, where="receipt payload gets")
        _nonnegative(self.segment_payload_bytes, where="receipt payload bytes")
        _key_tuple(self.result_keys, where="receipt result keys", allow_empty=True)
        if (not isinstance(self.citations, tuple)
                or any(not isinstance(item, RecallCitation)
                       for item in self.citations)):
            raise FreeTextHierarchyRecallContractError("receipt citations 类型错误")
        if self.citations != tuple(sorted(set(self.citations))):
            raise FreeTextHierarchyRecallContractError("receipt citations 必须排序去重")
        _enum(self.stop_reason, RECALL_STOP_REASONS, where="receipt stop reason")
        if _nonnegative(
                self.unauthorized_payload_read_count,
                where="unauthorized payload read count") != 0:
            raise FreeTextHierarchyRecallContractError("ACL 拒绝前后均不得越权读取 payload")

    def validate_obligation(self, obligation: RecallObligation) -> None:
        if self.obligation_key != obligation.obligation_key:
            raise FreeTextHierarchyRecallContractError("receipt 指向其他 obligation")
        budget = obligation.budget
        if (self.index_gets > budget.max_index_gets
                or self.segment_payload_gets > budget.max_segment_payload_gets
                or self.segment_payload_bytes > budget.max_segment_payload_bytes
                or len(self.result_keys) > budget.max_results):
            raise FreeTextHierarchyRecallContractError("receipt 超出 recall budget")
        if len(self.index_keys) > self.index_gets:
            raise FreeTextHierarchyRecallContractError("index identity 多于实际 get")
        if len(self.segment_keys) > self.segment_payload_gets:
            raise FreeTextHierarchyRecallContractError("segment identity 多于 payload get")
        if any(item.source_ref != obligation.source_ref for item in self.citations):
            raise FreeTextHierarchyRecallContractError("citation 跨 obligation SourceRef")
        if self.stop_reason == "UNAUTHORIZED":
            if any((
                    self.segment_payload_gets,
                    self.segment_payload_bytes,
                    len(self.segment_keys),
                    len(self.result_keys),
                    len(self.citations))):
                raise FreeTextHierarchyRecallContractError("ACL 必须在 payload 读取前拒绝")
        if self.stop_reason == "RESOLVED" and (
                not self.result_keys or not self.citations):
            raise FreeTextHierarchyRecallContractError("RESOLVED 缺 result/citation")
        if self.stop_reason == "NOT_FOUND" and (self.result_keys or self.citations):
            raise FreeTextHierarchyRecallContractError("NOT_FOUND 不得携带结果")
        expected_failure = (
            "NONE" if self.stop_reason == "RESOLVED" else self.stop_reason)
        if obligation.failure_state != expected_failure:
            raise FreeTextHierarchyRecallContractError("obligation failure 与 stop reason 不一致")

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations": [item.to_dict() for item in self.citations],
            "index_gets": self.index_gets,
            "index_keys": _keys_to_value(self.index_keys),
            "obligation_key": self.obligation_key.to_list(),
            "result_keys": _keys_to_value(self.result_keys),
            "segment_keys": _keys_to_value(self.segment_keys),
            "segment_payload_bytes": self.segment_payload_bytes,
            "segment_payload_gets": self.segment_payload_gets,
            "stop_reason": self.stop_reason,
            "unauthorized_payload_read_count": self.unauthorized_payload_read_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecallReceipt":
        raw = _exact(value, {
            "citations", "index_gets", "index_keys", "obligation_key",
            "result_keys", "segment_keys", "segment_payload_bytes",
            "segment_payload_gets", "stop_reason",
            "unauthorized_payload_read_count",
        }, where="RecallReceipt")
        if not isinstance(raw["citations"], list):
            raise FreeTextHierarchyRecallContractError("citations 必须是列表")
        return cls(
            _key_from_value(raw["obligation_key"], where="obligation_key"),
            _keys_from_value(raw["index_keys"], where="index_keys"),
            _keys_from_value(raw["segment_keys"], where="segment_keys"),
            raw["index_gets"],
            raw["segment_payload_gets"],
            raw["segment_payload_bytes"],
            _keys_from_value(raw["result_keys"], where="result_keys"),
            tuple(RecallCitation.from_dict(item) for item in raw["citations"]),
            raw["stop_reason"],
            raw["unauthorized_payload_read_count"],
        )


@dataclass(frozen=True)
class OwnerIsolation:
    """source/candidate/teacher/evaluator 的物理 owner 与可见面。"""

    source_owner_key: StableRecordKey
    candidate_owner_key: StableRecordKey
    teacher_owner_key: StableRecordKey
    evaluator_owner_key: StableRecordKey
    source_relative_path: str
    candidate_relative_path: str
    teacher_relative_path: str
    evaluator_relative_path: str
    candidate_visible_owner_kinds: tuple[str, ...]
    candidate_visible_evidence_keys: tuple[StableRecordKey, ...]
    candidate_evaluator_label_reads: int

    def __post_init__(self) -> None:
        owner_keys = tuple(getattr(self, name) for name in (
            "source_owner_key", "candidate_owner_key", "teacher_owner_key",
            "evaluator_owner_key"))
        for key in owner_keys:
            _key(key, where="owner isolation key")
        if len(set(owner_keys)) != 4:
            raise FreeTextHierarchyRecallContractError("四类 owner 必须物理隔离")
        paths = tuple(_relative_path(getattr(self, name), where=name) for name in (
            "source_relative_path", "candidate_relative_path",
            "teacher_relative_path", "evaluator_relative_path"))
        if len(set(paths)) != 4:
            raise FreeTextHierarchyRecallContractError("四类 owner path 必须物理隔离")
        if self.candidate_visible_owner_kinds != VISIBLE_OWNER_KINDS:
            raise FreeTextHierarchyRecallContractError("candidate 可见 owner 集合漂移")
        _key_tuple(
            self.candidate_visible_evidence_keys,
            where="candidate visible evidence keys",
            allow_empty=True,
        )
        if _flag(
                self.candidate_evaluator_label_reads,
                where="candidate evaluator reads") != 0:
            raise FreeTextHierarchyRecallContractError("candidate 偷看 evaluator 标签")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_evaluator_label_reads": self.candidate_evaluator_label_reads,
            "candidate_owner_key": self.candidate_owner_key.to_list(),
            "candidate_relative_path": self.candidate_relative_path,
            "candidate_visible_evidence_keys": _keys_to_value(
                self.candidate_visible_evidence_keys),
            "candidate_visible_owner_kinds": list(
                self.candidate_visible_owner_kinds),
            "evaluator_owner_key": self.evaluator_owner_key.to_list(),
            "evaluator_relative_path": self.evaluator_relative_path,
            "source_owner_key": self.source_owner_key.to_list(),
            "source_relative_path": self.source_relative_path,
            "teacher_owner_key": self.teacher_owner_key.to_list(),
            "teacher_relative_path": self.teacher_relative_path,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OwnerIsolation":
        raw = _exact(value, {
            "candidate_evaluator_label_reads", "candidate_owner_key",
            "candidate_relative_path", "candidate_visible_evidence_keys",
            "candidate_visible_owner_kinds", "evaluator_owner_key",
            "evaluator_relative_path", "source_owner_key", "source_relative_path",
            "teacher_owner_key", "teacher_relative_path",
        }, where="OwnerIsolation")
        if not isinstance(raw["candidate_visible_owner_kinds"], list):
            raise FreeTextHierarchyRecallContractError("visible owner kinds 必须是列表")
        return cls(
            _key_from_value(raw["source_owner_key"], where="source_owner_key"),
            _key_from_value(raw["candidate_owner_key"], where="candidate_owner_key"),
            _key_from_value(raw["teacher_owner_key"], where="teacher_owner_key"),
            _key_from_value(raw["evaluator_owner_key"], where="evaluator_owner_key"),
            raw["source_relative_path"],
            raw["candidate_relative_path"],
            raw["teacher_relative_path"],
            raw["evaluator_relative_path"],
            tuple(raw["candidate_visible_owner_kinds"]),
            _keys_from_value(
                raw["candidate_visible_evidence_keys"],
                where="candidate_visible_evidence_keys"),
            raw["candidate_evaluator_label_reads"],
        )


@dataclass(frozen=True)
class SplitIdentity:
    """按 source/document/template/structure/paraphrase cluster 分离样本。"""

    split: str
    source_cluster_key: StableRecordKey
    document_cluster_key: StableRecordKey
    template_cluster_key: StableRecordKey
    structure_cluster_key: StableRecordKey
    paraphrase_cluster_key: StableRecordKey

    def __post_init__(self) -> None:
        _enum(self.split, SPLITS, where="split")
        for name in (
                "source_cluster_key", "document_cluster_key",
                "template_cluster_key", "structure_cluster_key",
                "paraphrase_cluster_key"):
            _key(getattr(self, name), where=f"split {name}")

    def cluster_items(self) -> tuple[tuple[str, StableRecordKey], ...]:
        return tuple((name, getattr(self, name)) for name in (
            "source_cluster_key", "document_cluster_key", "template_cluster_key",
            "structure_cluster_key", "paraphrase_cluster_key"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_cluster_key": self.document_cluster_key.to_list(),
            "paraphrase_cluster_key": self.paraphrase_cluster_key.to_list(),
            "source_cluster_key": self.source_cluster_key.to_list(),
            "split": self.split,
            "structure_cluster_key": self.structure_cluster_key.to_list(),
            "template_cluster_key": self.template_cluster_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SplitIdentity":
        raw = _exact(value, {
            "document_cluster_key", "paraphrase_cluster_key",
            "source_cluster_key", "split", "structure_cluster_key",
            "template_cluster_key",
        }, where="SplitIdentity")
        return cls(
            raw["split"],
            _key_from_value(raw["source_cluster_key"], where="source_cluster_key"),
            _key_from_value(
                raw["document_cluster_key"], where="document_cluster_key"),
            _key_from_value(
                raw["template_cluster_key"], where="template_cluster_key"),
            _key_from_value(
                raw["structure_cluster_key"], where="structure_cluster_key"),
            _key_from_value(
                raw["paraphrase_cluster_key"], where="paraphrase_cluster_key"),
        )


@dataclass(frozen=True)
class EvaluatorLabel:
    """候选不可见的正确层级、中心、命题和 citation 标签。"""

    label_key: StableRecordKey
    evaluator_owner_key: StableRecordKey
    hierarchy_keys: tuple[StableRecordKey, ...]
    center_keys: tuple[StableRecordKey, ...]
    proposition_key: StableRecordKey
    citation_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        _key(self.label_key, where="label key")
        _key(self.evaluator_owner_key, where="label evaluator owner")
        _key_tuple(self.hierarchy_keys, where="label hierarchy keys")
        _key_tuple(self.center_keys, where="label center keys")
        _key(self.proposition_key, where="label proposition key")
        _key_tuple(
            self.citation_keys, where="label citation keys", allow_empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_keys": _keys_to_value(self.center_keys),
            "citation_keys": _keys_to_value(self.citation_keys),
            "evaluator_owner_key": self.evaluator_owner_key.to_list(),
            "hierarchy_keys": _keys_to_value(self.hierarchy_keys),
            "label_key": self.label_key.to_list(),
            "proposition_key": self.proposition_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluatorLabel":
        raw = _exact(value, {
            "center_keys", "citation_keys", "evaluator_owner_key",
            "hierarchy_keys", "label_key", "proposition_key",
        }, where="EvaluatorLabel")
        return cls(
            _key_from_value(raw["label_key"], where="label_key"),
            _key_from_value(raw["evaluator_owner_key"], where="evaluator_owner_key"),
            _keys_from_value(raw["hierarchy_keys"], where="hierarchy_keys"),
            _keys_from_value(raw["center_keys"], where="center_keys"),
            _key_from_value(raw["proposition_key"], where="proposition_key"),
            _keys_from_value(raw["citation_keys"], where="citation_keys"),
        )


@dataclass(frozen=True)
class FreeTextHierarchyRecallCase:
    """一个 authoring 侧完整 T0 case；runtime 必须按 owner 投影视图。"""

    format_version: int
    case_key: StableRecordKey
    document: SourceDocument
    query: FreeTextQuery
    evidence: tuple[CandidateEvidence, ...]
    hierarchy_candidates: tuple[HierarchyCandidate, ...]
    center_candidates: tuple[CenterCandidate, ...]
    obligation: RecallObligation
    receipt: RecallReceipt
    owner_isolation: OwnerIsolation
    split_identity: SplitIdentity
    evaluator_label: EvaluatorLabel

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != FORMAT_VERSION:
            raise FreeTextHierarchyRecallContractError("format_version 非法")
        _key(self.case_key, where="case key")
        if not isinstance(self.document, SourceDocument):
            raise FreeTextHierarchyRecallContractError("document 类型错误")
        if not isinstance(self.query, FreeTextQuery):
            raise FreeTextHierarchyRecallContractError("query 类型错误")
        self._validate_evidence()
        self._validate_hierarchy()
        self._validate_centers()
        if not isinstance(self.obligation, RecallObligation):
            raise FreeTextHierarchyRecallContractError("obligation 类型错误")
        if not isinstance(self.receipt, RecallReceipt):
            raise FreeTextHierarchyRecallContractError("receipt 类型错误")
        if not isinstance(self.owner_isolation, OwnerIsolation):
            raise FreeTextHierarchyRecallContractError("owner isolation 类型错误")
        if not isinstance(self.split_identity, SplitIdentity):
            raise FreeTextHierarchyRecallContractError("split identity 类型错误")
        if not isinstance(self.evaluator_label, EvaluatorLabel):
            raise FreeTextHierarchyRecallContractError("evaluator label 类型错误")
        self._validate_recall_and_owner()

    def _validate_evidence(self) -> None:
        if (not isinstance(self.evidence, tuple) or not self.evidence
                or any(not isinstance(item, CandidateEvidence)
                       for item in self.evidence)):
            raise FreeTextHierarchyRecallContractError("evidence 必须是非空 typed tuple")
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise FreeTextHierarchyRecallContractError("evidence identity 重复或未排序")
        for item in self.evidence:
            if item.source_ref != self.document.source_ref:
                raise FreeTextHierarchyRecallContractError("Evidence 跨 source/version")

    def _validate_hierarchy(self) -> None:
        if (not isinstance(self.hierarchy_candidates, tuple)
                or not self.hierarchy_candidates
                or any(not isinstance(item, HierarchyCandidate)
                       for item in self.hierarchy_candidates)):
            raise FreeTextHierarchyRecallContractError("hierarchy candidates 非法")
        if self.hierarchy_candidates != tuple(sorted(self.hierarchy_candidates)):
            raise FreeTextHierarchyRecallContractError("hierarchy candidates 未规范排序")
        keys = tuple(item.candidate_key for item in self.hierarchy_candidates)
        spans = tuple(item.span.span_key for item in self.hierarchy_candidates)
        if len(set(keys)) != len(keys) or len(set(spans)) != len(spans):
            raise FreeTextHierarchyRecallContractError("hierarchy candidate/span identity 重复")
        by_key = {item.candidate_key: item for item in self.hierarchy_candidates}
        evidence_keys = {item.evidence_key for item in self.evidence}
        span_keys = set(spans)
        if any(item.span_key is not None and item.span_key not in span_keys
               for item in self.evidence):
            raise FreeTextHierarchyRecallContractError("Evidence 指向未知 span")
        for item in self.hierarchy_candidates:
            item.span.validate_document(self.document)
            if not set(item.evidence_keys) <= evidence_keys:
                raise FreeTextHierarchyRecallContractError("candidate 指向未知 Evidence")
            if item.parent_key is not None and item.parent_key not in by_key:
                raise FreeTextHierarchyRecallContractError("candidate parent 越界")
        self._reject_parent_cycles(by_key)
        parent_kinds = {
            "SECTION": {None, "SECTION"},
            "PARAGRAPH": {"SECTION"},
            "PROPOSITION": {"PARAGRAPH", "PROPOSITION"},
        }
        for item in self.hierarchy_candidates:
            parent = None if item.parent_key is None else by_key[item.parent_key]
            parent_kind = None if parent is None else parent.candidate_kind
            if parent_kind not in parent_kinds[item.candidate_kind]:
                raise FreeTextHierarchyRecallContractError("candidate parent kind 非法")
            if parent is not None and not parent.span.contains(item.span):
                raise FreeTextHierarchyRecallContractError("parent span 未包含 child")
        groups: dict[tuple[StableRecordKey | None, str], list[int]] = {}
        for item in self.hierarchy_candidates:
            groups.setdefault(
                (item.parent_key, item.candidate_kind), []).append(item.ordinal)
        if any(sorted(values) != list(range(1, len(values) + 1))
               for values in groups.values()):
            raise FreeTextHierarchyRecallContractError("sibling ordinal 必须从 1 连续")

    @staticmethod
    def _reject_parent_cycles(
            by_key: dict[StableRecordKey, HierarchyCandidate]) -> None:
        for candidate in by_key.values():
            seen: set[StableRecordKey] = set()
            current = candidate
            while current.parent_key is not None:
                if current.parent_key in seen or current.parent_key == candidate.candidate_key:
                    raise FreeTextHierarchyRecallContractError("hierarchy parent 出现环")
                seen.add(current.parent_key)
                current = by_key[current.parent_key]

    def _validate_centers(self) -> None:
        if (not isinstance(self.center_candidates, tuple)
                or not self.center_candidates
                or any(not isinstance(item, CenterCandidate)
                       for item in self.center_candidates)):
            raise FreeTextHierarchyRecallContractError("center candidates 非法")
        if self.center_candidates != tuple(sorted(set(self.center_candidates))):
            raise FreeTextHierarchyRecallContractError("center identity 重复或未排序")
        evidence_keys = {item.evidence_key for item in self.evidence}
        hierarchy_keys = {item.candidate_key for item in self.hierarchy_candidates}
        for item in self.center_candidates:
            if item.query_key != self.query.query_key:
                raise FreeTextHierarchyRecallContractError("center 指向其他 query")
            if item.situation_key != self.query.situation_key:
                raise FreeTextHierarchyRecallContractError("center 指向其他 situation")
            if not set(item.evidence_keys) <= evidence_keys:
                raise FreeTextHierarchyRecallContractError("center 指向未知 Evidence")
            if (item.target_kind == "HIERARCHY_CANDIDATE"
                    and item.target_key not in hierarchy_keys):
                raise FreeTextHierarchyRecallContractError("center target 越界")

    def _validate_recall_and_owner(self) -> None:
        center_keys = {item.center_key for item in self.center_candidates}
        if self.obligation.center_key not in center_keys:
            raise FreeTextHierarchyRecallContractError("obligation center 越界")
        if self.obligation.query_key != self.query.query_key:
            raise FreeTextHierarchyRecallContractError("obligation query 不一致")
        if self.obligation.source_ref != self.document.source_ref:
            raise FreeTextHierarchyRecallContractError("obligation 跨 source/version")
        if self.obligation.scope_key != self.document.scope_key:
            raise FreeTextHierarchyRecallContractError("obligation scope 不一致")
        if self.obligation.acl_key != self.document.acl_key:
            raise FreeTextHierarchyRecallContractError("obligation ACL 不一致")
        self.receipt.validate_obligation(self.obligation)
        for citation in self.receipt.citations:
            citation.span.validate_document(self.document)
        evidence_keys = tuple(item.evidence_key for item in self.evidence)
        if self.owner_isolation.candidate_visible_evidence_keys != evidence_keys:
            raise FreeTextHierarchyRecallContractError("candidate 可见 Evidence 清单不闭合")
        label = self.evaluator_label
        if label.evaluator_owner_key != self.owner_isolation.evaluator_owner_key:
            raise FreeTextHierarchyRecallContractError("label owner 与隔离账不一致")
        if label.label_key in self.owner_isolation.candidate_visible_evidence_keys:
            raise FreeTextHierarchyRecallContractError("candidate 偷看 evaluator label")
        hierarchy_keys = {item.candidate_key for item in self.hierarchy_candidates}
        if not set(label.hierarchy_keys) <= hierarchy_keys:
            raise FreeTextHierarchyRecallContractError("label hierarchy key 越界")
        if not set(label.center_keys) <= center_keys:
            raise FreeTextHierarchyRecallContractError("label center key 越界")
        citation_keys = {item.citation_key for item in self.receipt.citations}
        if not set(label.citation_keys) <= citation_keys:
            raise FreeTextHierarchyRecallContractError("label citation key 越界")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "P3I_FREE_TEXT_HIERARCHY_RECALL_CASE",
            "case_key": self.case_key.to_list(),
            "center_candidates": [item.to_dict() for item in self.center_candidates],
            "document": self.document.to_dict(),
            "evaluator_label": self.evaluator_label.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "format_version": self.format_version,
            "hierarchy_candidates": [
                item.to_dict() for item in self.hierarchy_candidates],
            "obligation": self.obligation.to_dict(),
            "owner_isolation": self.owner_isolation.to_dict(),
            "query": self.query.to_dict(),
            "receipt": self.receipt.to_dict(),
            "split_identity": self.split_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FreeTextHierarchyRecallCase":
        raw = _exact(value, {
            "artifact_kind", "case_key", "center_candidates", "document",
            "evaluator_label", "evidence", "format_version",
            "hierarchy_candidates", "obligation", "owner_isolation", "query",
            "receipt", "split_identity",
        }, where="FreeTextHierarchyRecallCase")
        if raw["artifact_kind"] != "P3I_FREE_TEXT_HIERARCHY_RECALL_CASE":
            raise FreeTextHierarchyRecallContractError("artifact_kind 非法")
        for name in ("center_candidates", "evidence", "hierarchy_candidates"):
            if not isinstance(raw[name], list):
                raise FreeTextHierarchyRecallContractError(f"{name} 必须是列表")
        return cls(
            raw["format_version"],
            _key_from_value(raw["case_key"], where="case_key"),
            SourceDocument.from_dict(raw["document"]),
            FreeTextQuery.from_dict(raw["query"]),
            tuple(CandidateEvidence.from_dict(item) for item in raw["evidence"]),
            tuple(HierarchyCandidate.from_dict(item)
                  for item in raw["hierarchy_candidates"]),
            tuple(CenterCandidate.from_dict(item)
                  for item in raw["center_candidates"]),
            RecallObligation.from_dict(raw["obligation"]),
            RecallReceipt.from_dict(raw["receipt"]),
            OwnerIsolation.from_dict(raw["owner_isolation"]),
            SplitIdentity.from_dict(raw["split_identity"]),
            EvaluatorLabel.from_dict(raw["evaluator_label"]),
        )


__all__ = [
    "AbsoluteSpan",
    "CandidateEvidence",
    "CenterCandidate",
    "EvaluatorLabel",
    "FreeTextHierarchyRecallCase",
    "FreeTextHierarchyRecallContractError",
    "FreeTextQuery",
    "HierarchyCandidate",
    "OwnerIsolation",
    "RecallBudget",
    "RecallCitation",
    "RecallObligation",
    "RecallReceipt",
    "SourceDocument",
    "SplitIdentity",
]
