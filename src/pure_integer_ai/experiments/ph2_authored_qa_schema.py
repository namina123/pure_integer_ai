"""D-02E.2 typed 问答、候选采用与 reference winner 资料纯合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    ALLOWED_PERTURBATIONS,
    LICENSE_ID,
    REQUIRED_SAMPLE_ROLES,
    SOURCE_KEY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
)


QUESTION_KINDS = frozenset({
    "EXPLICIT_FACT",
    "EXACT_RELATION",
    "FINITE_LOGIC",
    "REFERENCE_SCOPE",
    "UNKNOWN",
    "AMBIGUOUS",
    "UNSUPPORTED",
    "CONFLICT",
})
ROUTE_STATUSES = frozenset({"REGISTERED", "UNSUPPORTED"})

_TARGET_FIELDS = frozenset({
    "end",
    "ordinal",
    "predicate_kind",
    "proposition_local_id",
    "start",
    "surface_fragment",
})
_CANDIDATE_FIELDS = frozenset({
    "candidate_id",
    "competition_id",
    "end",
    "evidence_ids",
    "evidence_refute",
    "evidence_support",
    "ordinal",
    "predicate_kind",
    "proposition_local_id",
    "start",
    "surface_fragment",
})
_OCCURRENCE_FIELDS = frozenset({
    "end",
    "occurrence_id",
    "ordinal",
    "semantic_kind",
    "start",
    "surface_fragment",
})
_REFERENCE_FIELDS = frozenset({
    "adopted_occurrence_ids",
    "candidate_occurrence_ids",
    "reference_occurrence_id",
    "winner_occurrence_id",
})
_REQUEST_FIELDS = frozenset({
    "max_candidates",
    "max_context_chars",
    "max_evidence_items",
    "max_occurrences",
})
_SEED_FIELDS = frozenset({
    "candidates",
    "consumer_request",
    "context_surface",
    "expected_payload",
    "expected_state",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "occurrences",
    "perturbation_kind",
    "query_scope_local_id",
    "question_kind",
    "question_surface",
    "reference_resolution",
    "required_refute",
    "required_support",
    "response_scope_local_id",
    "route_status",
    "sample_role",
    "seed_id",
    "source_key",
    "split",
    "supersedes_seed_id",
    "target",
    "template_family",
})


class AuthoredQACourseError(RuntimeError):
    """原创 QA seed 的 target、候选、reference、scope、owner 或预算非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本无首尾空白；按字段允许空字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredQACourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredQACourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求身份、scope 和预算为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredQACourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 span 和 ordinal 为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredQACourseError(f"{where} 必须是非负严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """要求 Evidence/required 位为严格整数 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredQACourseError(f"{where} 必须是严格整数 0/1")
    return value


def _text_tuple(value: Any, *, where: str, allow_empty: bool = False):
    """恢复无重复非空字符串列表。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise AuthoredQACourseError(f"{where} 非法或重复")
    return tuple(value)


def _int_tuple(value: Any, *, where: str, allow_empty: bool = False):
    """恢复无重复正严格整数列表。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(type(item) is not int or item <= 0 for item in value)
            or len(set(value)) != len(value)):
        raise AuthoredQACourseError(f"{where} 非法或重复")
    return tuple(value)


@dataclass(frozen=True)
class QATargetSeed:
    """已形成问题目标 Proposition 的来源 anchor。"""

    proposition_local_id: int
    predicate_kind: int
    surface_fragment: str
    start: int
    end: int
    ordinal: int

    def __post_init__(self) -> None:
        _positive_int(
            self.proposition_local_id,
            where="QATargetSeed.proposition_local_id",
        )
        _positive_int(self.predicate_kind, where="QATargetSeed.predicate_kind")
        _text(self.surface_fragment, where="QATargetSeed.surface_fragment")
        _nonnegative_int(self.start, where="QATargetSeed.start")
        _nonnegative_int(self.end, where="QATargetSeed.end")
        _nonnegative_int(self.ordinal, where="QATargetSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredQACourseError("QA target 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "QATargetSeed":
        """从严格字段集合恢复 target。"""
        if not isinstance(value, dict) or set(value) != _TARGET_FIELDS:
            raise AuthoredQACourseError("QA target 字段集合漂移")
        return cls(
            value["proposition_local_id"],
            value["predicate_kind"],
            _text(value["surface_fragment"], where="target.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
        )


@dataclass(frozen=True)
class QACandidateSeed:
    """一个可执行查询返回的候选 Proposition 及 Evidence 四态。"""

    candidate_id: str
    competition_id: int
    proposition_local_id: int
    predicate_kind: int
    surface_fragment: str
    start: int
    end: int
    ordinal: int
    evidence_support: int
    evidence_refute: int
    evidence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.candidate_id, where="QACandidateSeed.candidate_id")
        for name, value in (
                ("competition_id", self.competition_id),
                ("proposition_local_id", self.proposition_local_id),
                ("predicate_kind", self.predicate_kind)):
            _positive_int(value, where=f"QACandidateSeed.{name}")
        _text(self.surface_fragment, where="QACandidateSeed.surface_fragment")
        _nonnegative_int(self.start, where="QACandidateSeed.start")
        _nonnegative_int(self.end, where="QACandidateSeed.end")
        _nonnegative_int(self.ordinal, where="QACandidateSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredQACourseError("QA candidate 必须有正宽度")
        _bit(self.evidence_support, where="QACandidateSeed.evidence_support")
        _bit(self.evidence_refute, where="QACandidateSeed.evidence_refute")
        if (not isinstance(self.evidence_ids, tuple)
                or any(type(item) is not int or item <= 0
                       for item in self.evidence_ids)
                or len(set(self.evidence_ids)) != len(self.evidence_ids)):
            raise AuthoredQACourseError("QA candidate Evidence 非法或重复")
        if bool(self.evidence_ids) != bool(
                self.evidence_support or self.evidence_refute):
            raise AuthoredQACourseError(
                "QA candidate Evidence id 与四态可用性不一致")

    @classmethod
    def from_dict(cls, value: Any) -> "QACandidateSeed":
        """从严格字段集合恢复候选。"""
        if not isinstance(value, dict) or set(value) != _CANDIDATE_FIELDS:
            raise AuthoredQACourseError("QA candidate 字段集合漂移")
        return cls(
            _text(value["candidate_id"], where="candidate_id"),
            value["competition_id"],
            value["proposition_local_id"],
            value["predicate_kind"],
            _text(
                value["surface_fragment"],
                where="candidate.surface_fragment",
            ),
            value["start"],
            value["end"],
            value["ordinal"],
            value["evidence_support"],
            value["evidence_refute"],
            _int_tuple(
                value["evidence_ids"],
                where="candidate.evidence_ids",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True)
class QAOccurrenceSeed:
    """reference question 使用的来源化 Occurrence。"""

    occurrence_id: str
    surface_fragment: str
    start: int
    end: int
    ordinal: int
    semantic_kind: int

    def __post_init__(self) -> None:
        _text(self.occurrence_id, where="QAOccurrenceSeed.occurrence_id")
        _text(self.surface_fragment, where="QAOccurrenceSeed.surface_fragment")
        _nonnegative_int(self.start, where="QAOccurrenceSeed.start")
        _nonnegative_int(self.end, where="QAOccurrenceSeed.end")
        _nonnegative_int(self.ordinal, where="QAOccurrenceSeed.ordinal")
        _positive_int(self.semantic_kind, where="QAOccurrenceSeed.semantic_kind")
        if self.end <= self.start:
            raise AuthoredQACourseError("QA occurrence 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "QAOccurrenceSeed":
        """从严格字段集合恢复 occurrence。"""
        if not isinstance(value, dict) or set(value) != _OCCURRENCE_FIELDS:
            raise AuthoredQACourseError("QA occurrence 字段集合漂移")
        return cls(
            _text(value["occurrence_id"], where="occurrence_id"),
            _text(
                value["surface_fragment"],
                where="occurrence.surface_fragment",
            ),
            value["start"],
            value["end"],
            value["ordinal"],
            value["semantic_kind"],
        )


@dataclass(frozen=True)
class QAReferenceResolutionSeed:
    """A-01 adopted 集合及仅 singleton 时存在的 winner。"""

    reference_occurrence_id: str
    candidate_occurrence_ids: tuple[str, ...]
    adopted_occurrence_ids: tuple[str, ...]
    winner_occurrence_id: str

    def __post_init__(self) -> None:
        _text(
            self.reference_occurrence_id,
            where="QAReferenceResolutionSeed.reference_occurrence_id",
        )
        for name, value in (
                ("candidate", self.candidate_occurrence_ids),
                ("adopted", self.adopted_occurrence_ids)):
            if (not isinstance(value, tuple)
                    or (name == "candidate" and not value)
                    or len(set(value)) != len(value)
                    or any(not isinstance(item, str) or not item
                           for item in value)):
                raise AuthoredQACourseError(
                    f"QA reference {name} 非法或重复")
        if not set(self.adopted_occurrence_ids) <= set(
                self.candidate_occurrence_ids):
            raise AuthoredQACourseError(
                "QA adopted reference 必须来自候选集合")
        _text(
            self.winner_occurrence_id,
            where="QAReferenceResolutionSeed.winner_occurrence_id",
            allow_empty=True,
        )
        if len(self.adopted_occurrence_ids) == 1:
            if self.winner_occurrence_id != self.adopted_occurrence_ids[0]:
                raise AuthoredQACourseError(
                    "QA singleton adopted 必须声明同一 winner")
        elif self.winner_occurrence_id:
            raise AuthoredQACourseError(
                "QA 非 singleton adopted 不得伪造 winner")

    @classmethod
    def from_dict(cls, value: Any) -> "QAReferenceResolutionSeed":
        """从严格字段集合恢复 reference resolution。"""
        if not isinstance(value, dict) or set(value) != _REFERENCE_FIELDS:
            raise AuthoredQACourseError("QA reference 字段集合漂移")
        return cls(
            _text(
                value["reference_occurrence_id"],
                where="reference_occurrence_id",
            ),
            _text_tuple(
                value["candidate_occurrence_ids"],
                where="reference.candidates",
            ),
            _text_tuple(
                value["adopted_occurrence_ids"],
                where="reference.adopted",
                allow_empty=True,
            ),
            _text(
                value["winner_occurrence_id"],
                where="winner_occurrence_id",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True)
class QAConsumerRequestSeed:
    """QA context、candidate、occurrence 和 Evidence 预算。"""

    max_context_chars: int
    max_candidates: int
    max_occurrences: int
    max_evidence_items: int

    def __post_init__(self) -> None:
        for name, value in (
                ("max_context_chars", self.max_context_chars),
                ("max_candidates", self.max_candidates),
                ("max_occurrences", self.max_occurrences),
                ("max_evidence_items", self.max_evidence_items)):
            _positive_int(value, where=f"QAConsumerRequestSeed.{name}")

    @classmethod
    def from_dict(cls, value: Any) -> "QAConsumerRequestSeed":
        """从严格字段集合恢复 QA consumer。"""
        if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
            raise AuthoredQACourseError("QA consumer 字段集合漂移")
        return cls(
            value["max_context_chars"],
            value["max_candidates"],
            value["max_occurrences"],
            value["max_evidence_items"],
        )


@dataclass(frozen=True)
class AuthoredQASeed:
    """一条 F-00 typed 问答、Evidence 采用和 reference scope 课程记录。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    source_key: str
    context_surface: str
    question_surface: str
    question_kind: str
    route_status: str
    target: QATargetSeed
    required_support: int
    required_refute: int
    query_scope_local_id: int
    response_scope_local_id: int
    candidates: tuple[QACandidateSeed, ...]
    occurrences: tuple[QAOccurrenceSeed, ...]
    reference_resolution: QAReferenceResolutionSeed | None
    consumer_request: QAConsumerRequestSeed
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("context_surface", self.context_surface),
                ("question_surface", self.question_surface),
                ("question_kind", self.question_kind),
                ("route_status", self.route_status),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredQASeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredQASeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredQACourseError("label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredQACourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredQACourseError("sample_role 不属于 QA 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredQACourseError("QA supersede 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredQACourseError("非 supersede QA 不得声明替代目标")
        if self.source_key != SOURCE_KEY:
            raise AuthoredQACourseError("QA source key 漂移")
        if self.question_kind not in QUESTION_KINDS:
            raise AuthoredQACourseError("QA question kind 未注册")
        if self.route_status not in ROUTE_STATUSES:
            raise AuthoredQACourseError("QA route status 未注册")
        if not isinstance(self.target, QATargetSeed):
            raise AuthoredQACourseError("QA target 类型错误")
        _bit(self.required_support, where="AuthoredQASeed.required_support")
        _bit(self.required_refute, where="AuthoredQASeed.required_refute")
        if not self.required_support and not self.required_refute:
            raise AuthoredQACourseError("QA required 至少声明一个 Evidence 方向")
        _positive_int(
            self.query_scope_local_id,
            where="AuthoredQASeed.query_scope_local_id",
        )
        _positive_int(
            self.response_scope_local_id,
            where="AuthoredQASeed.response_scope_local_id",
        )
        if self.query_scope_local_id == self.response_scope_local_id:
            raise AuthoredQACourseError("QA evidence/response scope 不得混用")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, QACandidateSeed)
                       for item in self.candidates)):
            raise AuthoredQACourseError("QA candidates 类型错误")
        if (not isinstance(self.occurrences, tuple)
                or any(not isinstance(item, QAOccurrenceSeed)
                       for item in self.occurrences)):
            raise AuthoredQACourseError("QA occurrences 类型错误")
        candidate_ids = [item.candidate_id for item in self.candidates]
        occurrence_ids = [item.occurrence_id for item in self.occurrences]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise AuthoredQACourseError("QA candidate_id 重复")
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise AuthoredQACourseError("QA occurrence_id 重复")
        candidate_order = [item.ordinal for item in self.candidates]
        occurrence_order = [item.ordinal for item in self.occurrences]
        if (candidate_order != sorted(candidate_order)
                or len(set(candidate_order)) != len(candidate_order)):
            raise AuthoredQACourseError("QA candidate ordinal 必须严格递增")
        if (occurrence_order != sorted(occurrence_order)
                or len(set(occurrence_order)) != len(occurrence_order)):
            raise AuthoredQACourseError("QA occurrence ordinal 必须严格递增")
        for item in (self.target, *self.candidates, *self.occurrences):
            if item.end > len(self.context_surface) or self.context_surface[
                    item.start:item.end] != item.surface_fragment:
                raise AuthoredQACourseError(
                    "QA target/candidate/occurrence 与 context 不一致")
        if len(self.context_surface) > self.consumer_request.max_context_chars:
            raise AuthoredQACourseError("QA context 超过字符预算")
        if len(self.candidates) > self.consumer_request.max_candidates:
            raise AuthoredQACourseError("QA candidates 超过预算")
        if len(self.occurrences) > self.consumer_request.max_occurrences:
            raise AuthoredQACourseError("QA occurrences 超过预算")
        if sum(len(item.evidence_ids) for item in self.candidates) > (
                self.consumer_request.max_evidence_items):
            raise AuthoredQACourseError("QA Evidence 超过预算")
        if self.route_status == "UNSUPPORTED":
            if self.question_kind != "UNSUPPORTED" or self.candidates:
                raise AuthoredQACourseError(
                    "unsupported QA 不得伪造 route 或候选")
        elif self.question_kind == "UNSUPPORTED":
            raise AuthoredQACourseError(
                "UNSUPPORTED question 必须显式缺 route")
        if self.question_kind == "REFERENCE_SCOPE":
            if self.reference_resolution is None:
                raise AuthoredQACourseError(
                    "reference question 必须声明 A-01 resolution")
        elif self.reference_resolution is not None:
            raise AuthoredQACourseError(
                "非 reference question 不得伪造 A-01 resolution")
        if self.reference_resolution is not None:
            reference = self.reference_resolution
            known = set(occurrence_ids)
            ids = {
                reference.reference_occurrence_id,
                *reference.candidate_occurrence_ids,
                *reference.adopted_occurrence_ids,
            }
            if reference.winner_occurrence_id:
                ids.add(reference.winner_occurrence_id)
            if not ids <= known:
                raise AuthoredQACourseError(
                    "QA reference resolution 引用未知 occurrence")
            if reference.reference_occurrence_id in (
                    reference.candidate_occurrence_ids):
                raise AuthoredQACourseError(
                    "QA reference 自身不得成为 antecedent")
        if self.question_kind == "AMBIGUOUS":
            eligible = [
                item for item in self.candidates
                if ((not self.required_support or item.evidence_support)
                    and (not self.required_refute or item.evidence_refute)
                    and not (item.evidence_support and item.evidence_refute))
            ]
            groups = {}
            for item in eligible:
                groups.setdefault(item.competition_id, set()).add(
                    item.proposition_local_id)
            if not any(len(values) > 1 for values in groups.values()):
                raise AuthoredQACourseError(
                    "AMBIGUOUS QA 必须有同 competition 多命题")
        if self.question_kind == "CONFLICT" and not any(
                item.evidence_support and item.evidence_refute
                for item in self.candidates):
            raise AuthoredQACourseError(
                "CONFLICT QA 必须携带双向 Evidence")
        if self.question_kind == "UNKNOWN" and any(
                item.evidence_support or item.evidence_refute
                for item in self.candidates):
            raise AuthoredQACourseError(
                "UNKNOWN QA 不得携带可决 Evidence")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredQACourseError("QA expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredQACourseError("QA perturbation 未注册")
        _positive_int(self.logical_order, where="AuthoredQASeed.logical_order")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredQASeed":
        """从严格字段集合恢复 QA seed。"""
        if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
            raise AuthoredQACourseError("QA seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredQACourseError("QA seed 必须是 CC0-1.0")
        raw_candidates = value["candidates"]
        raw_occurrences = value["occurrences"]
        if not isinstance(raw_candidates, list) or not isinstance(
                raw_occurrences, list):
            raise AuthoredQACourseError(
                "QA candidates/occurrences 必须是列表")
        raw_reference = value["reference_resolution"]
        return cls(
            _text(value["seed_id"], where="seed_id"),
            _text(value["family"], where="family"),
            _text(value["template_family"], where="template_family"),
            _text(value["label_owner"], where="label_owner"),
            _text(value["split"], where="split"),
            _text(value["sample_role"], where="sample_role"),
            _text(value["source_key"], where="source_key"),
            _text(value["context_surface"], where="context_surface"),
            _text(value["question_surface"], where="question_surface"),
            _text(value["question_kind"], where="question_kind"),
            _text(value["route_status"], where="route_status"),
            QATargetSeed.from_dict(value["target"]),
            value["required_support"],
            value["required_refute"],
            value["query_scope_local_id"],
            value["response_scope_local_id"],
            tuple(QACandidateSeed.from_dict(item) for item in raw_candidates),
            tuple(QAOccurrenceSeed.from_dict(item) for item in raw_occurrences),
            (None if raw_reference is None
             else QAReferenceResolutionSeed.from_dict(raw_reference)),
            QAConsumerRequestSeed.from_dict(value["consumer_request"]),
            _text(value["expected_state"], where="expected_state"),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            _text(value["perturbation_kind"], where="perturbation_kind"),
            _text(
                value["supersedes_seed_id"],
                where="supersedes_seed_id",
                allow_empty=True,
            ),
            value["logical_order"],
        )


__all__ = [
    "AuthoredQACourseError",
    "AuthoredQASeed",
    "LICENSE_ID",
    "QACandidateSeed",
    "QAConsumerRequestSeed",
    "QAOccurrenceSeed",
    "QAReferenceResolutionSeed",
    "QATargetSeed",
    "QUESTION_KINDS",
    "REQUIRED_SAMPLE_ROLES",
    "ROUTE_STATUSES",
    "SOURCE_KEY",
]
