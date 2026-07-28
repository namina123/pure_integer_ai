"""D-02E 中文变体、篇章、reference 和 parser revision 资料纯合同。"""
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


VARIANT_KINDS = frozenset({
    "NEW_SURFACE",
    "NEW_LENGTH",
    "ELLIPSIS",
    "REFERENCE",
    "POLYSEMY",
    "PARAGRAPH",
    "SOURCE_CONFLICT",
    "PARSER_REVISION",
    "LATER_REINTERPRETATION",
})

_SPAN_FIELDS = frozenset({
    "end", "ordinal", "span_id", "span_kind", "start", "surface_fragment"})
_OCCURRENCE_FIELDS = frozenset({
    "end",
    "occurrence_id",
    "ordinal",
    "semantic_kind",
    "start",
    "surface_fragment",
})
_REFERENCE_FIELDS = frozenset({
    "candidate_occurrence_ids",
    "impacted_query_ids",
    "reference_occurrence_id",
    "rejected_occurrence_id",
    "replacement_occurrence_id",
    "window_occurrence_ids",
})
_MAPPING_FIELDS = frozenset({
    "old_occurrence_id", "replacement_occurrence_ids"})
_REVISION_FIELDS = frozenset({
    "affected_occurrence_ids",
    "mappings",
    "new_parser_version",
    "old_parser_version",
    "recompute_query_ids",
    "unaffected_occurrence_ids",
})
_REQUEST_FIELDS = frozenset({
    "max_candidates",
    "max_context_chars",
    "max_occurrences",
    "max_paragraphs",
    "max_recompute_queries",
})
_SEED_FIELDS = frozenset({
    "consumer_request",
    "evidence_refute",
    "evidence_support",
    "expected_payload",
    "expected_state",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "occurrences",
    "paragraphs",
    "parser_revision",
    "perturbation_kind",
    "reference_plan",
    "sample_role",
    "seed_id",
    "source_key",
    "split",
    "supersedes_seed_id",
    "surface",
    "template_family",
    "variant_kind",
})


class AuthoredDiscourseCourseError(RuntimeError):
    """原创 discourse seed 的 span、reference、revision、owner 或预算非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 discourse 文本为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredDiscourseCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredDiscourseCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求身份和预算为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredDiscourseCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 span 与 ordinal 为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredDiscourseCourseError(f"{where} 必须是非负严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """要求 Evidence 位只能是严格整数 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredDiscourseCourseError(f"{where} 必须是严格整数 0/1")
    return value


def _text_tuple(value: Any, *, where: str, allow_empty: bool = False):
    """恢复无重复字符串列表为 tuple。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise AuthoredDiscourseCourseError(f"{where} 非法或重复")
    return tuple(value)


def _int_tuple(value: Any, *, where: str, allow_empty: bool = False):
    """恢复无重复正严格整数列表为 tuple。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(type(item) is not int or item <= 0 for item in value)
            or len(set(value)) != len(value)):
        raise AuthoredDiscourseCourseError(f"{where} 非法或重复")
    return tuple(value)


@dataclass(frozen=True)
class DiscourseSpanSeed:
    """一个段落或句子 Span 的来源区间。"""

    span_id: str
    span_kind: str
    surface_fragment: str
    start: int
    end: int
    ordinal: int

    def __post_init__(self) -> None:
        _text(self.span_id, where="DiscourseSpanSeed.span_id")
        if self.span_kind not in {"PARAGRAPH", "SENTENCE"}:
            raise AuthoredDiscourseCourseError("discourse span kind 未注册")
        _text(
            self.surface_fragment,
            where="DiscourseSpanSeed.surface_fragment",
        )
        _nonnegative_int(self.start, where="DiscourseSpanSeed.start")
        _nonnegative_int(self.end, where="DiscourseSpanSeed.end")
        _nonnegative_int(self.ordinal, where="DiscourseSpanSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredDiscourseCourseError("discourse span 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "DiscourseSpanSeed":
        """从严格字段集合恢复 discourse span。"""
        if not isinstance(value, dict) or set(value) != _SPAN_FIELDS:
            raise AuthoredDiscourseCourseError("discourse span 字段集合漂移")
        return cls(
            _text(value["span_id"], where="span_id"),
            _text(value["span_kind"], where="span_kind"),
            _text(value["surface_fragment"], where="span.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
        )


@dataclass(frozen=True)
class DiscourseOccurrenceSeed:
    """一个可回源 Occurrence 及开放 semantic kind。"""

    occurrence_id: str
    surface_fragment: str
    start: int
    end: int
    ordinal: int
    semantic_kind: int

    def __post_init__(self) -> None:
        _text(
            self.occurrence_id,
            where="DiscourseOccurrenceSeed.occurrence_id",
        )
        _text(
            self.surface_fragment,
            where="DiscourseOccurrenceSeed.surface_fragment",
        )
        _nonnegative_int(self.start, where="DiscourseOccurrenceSeed.start")
        _nonnegative_int(self.end, where="DiscourseOccurrenceSeed.end")
        _nonnegative_int(self.ordinal, where="DiscourseOccurrenceSeed.ordinal")
        _positive_int(
            self.semantic_kind,
            where="DiscourseOccurrenceSeed.semantic_kind",
        )
        if self.end <= self.start:
            raise AuthoredDiscourseCourseError(
                "discourse occurrence 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "DiscourseOccurrenceSeed":
        """从严格字段集合恢复 discourse occurrence。"""
        if not isinstance(value, dict) or set(value) != _OCCURRENCE_FIELDS:
            raise AuthoredDiscourseCourseError(
                "discourse occurrence 字段集合漂移")
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
class DiscourseReferencePlanSeed:
    """有界 reference window、候选和可选后文替代。"""

    reference_occurrence_id: str
    window_occurrence_ids: tuple[str, ...]
    candidate_occurrence_ids: tuple[str, ...]
    rejected_occurrence_id: str
    replacement_occurrence_id: str
    impacted_query_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(
            self.reference_occurrence_id,
            where="reference.reference_occurrence_id",
        )
        if (not isinstance(self.window_occurrence_ids, tuple)
                or not self.window_occurrence_ids
                or len(set(self.window_occurrence_ids))
                != len(self.window_occurrence_ids)):
            raise AuthoredDiscourseCourseError(
                "reference window 非法或重复")
        if (not isinstance(self.candidate_occurrence_ids, tuple)
                or not self.candidate_occurrence_ids
                or len(set(self.candidate_occurrence_ids))
                != len(self.candidate_occurrence_ids)
                or not set(self.candidate_occurrence_ids) <= set(
                    self.window_occurrence_ids)):
            raise AuthoredDiscourseCourseError(
                "reference candidates 必须是 window 非空子集")
        _text(
            self.rejected_occurrence_id,
            where="reference.rejected_occurrence_id",
            allow_empty=True,
        )
        _text(
            self.replacement_occurrence_id,
            where="reference.replacement_occurrence_id",
            allow_empty=True,
        )
        if bool(self.rejected_occurrence_id) != bool(
                self.replacement_occurrence_id):
            raise AuthoredDiscourseCourseError(
                "reference revision 必须同时声明 rejected/replacement")
        if (self.rejected_occurrence_id
                and (self.rejected_occurrence_id
                     not in self.candidate_occurrence_ids
                     or self.replacement_occurrence_id
                     not in self.candidate_occurrence_ids
                     or self.rejected_occurrence_id
                     == self.replacement_occurrence_id)):
            raise AuthoredDiscourseCourseError(
                "reference revision 必须替换不同候选")
        if (not isinstance(self.impacted_query_ids, tuple)
                or any(type(item) is not int or item <= 0
                       for item in self.impacted_query_ids)
                or len(set(self.impacted_query_ids))
                != len(self.impacted_query_ids)):
            raise AuthoredDiscourseCourseError(
                "reference impacted query 非法或重复")

    @classmethod
    def from_dict(cls, value: Any) -> "DiscourseReferencePlanSeed":
        """从严格字段集合恢复 reference plan。"""
        if not isinstance(value, dict) or set(value) != _REFERENCE_FIELDS:
            raise AuthoredDiscourseCourseError(
                "reference plan 字段集合漂移")
        return cls(
            _text(
                value["reference_occurrence_id"],
                where="reference_occurrence_id",
            ),
            _text_tuple(
                value["window_occurrence_ids"], where="reference.window"),
            _text_tuple(
                value["candidate_occurrence_ids"],
                where="reference.candidates",
            ),
            _text(
                value["rejected_occurrence_id"],
                where="rejected_occurrence_id",
                allow_empty=True,
            ),
            _text(
                value["replacement_occurrence_id"],
                where="replacement_occurrence_id",
                allow_empty=True,
            ),
            _int_tuple(
                value["impacted_query_ids"],
                where="reference.impacted_query_ids",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True)
class DiscourseRevisionMappingSeed:
    """一个旧 occurrence 到零个或多个新 occurrence 的 mapping。"""

    old_occurrence_id: str
    replacement_occurrence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(
            self.old_occurrence_id,
            where="revision_mapping.old_occurrence_id",
        )
        if (not isinstance(self.replacement_occurrence_ids, tuple)
                or len(set(self.replacement_occurrence_ids))
                != len(self.replacement_occurrence_ids)):
            raise AuthoredDiscourseCourseError(
                "revision replacements 非法或重复")

    @classmethod
    def from_dict(cls, value: Any) -> "DiscourseRevisionMappingSeed":
        """从严格字段集合恢复 revision mapping。"""
        if not isinstance(value, dict) or set(value) != _MAPPING_FIELDS:
            raise AuthoredDiscourseCourseError(
                "revision mapping 字段集合漂移")
        return cls(
            _text(value["old_occurrence_id"], where="old_occurrence_id"),
            _text_tuple(
                value["replacement_occurrence_ids"],
                where="replacement_occurrence_ids",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True)
class DiscourseParserRevisionSeed:
    """跨 ParserVersion 的 anchor mapping 与局部 recompute 边界。"""

    old_parser_version: int
    new_parser_version: int
    mappings: tuple[DiscourseRevisionMappingSeed, ...]
    affected_occurrence_ids: tuple[str, ...]
    unaffected_occurrence_ids: tuple[str, ...]
    recompute_query_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _positive_int(
            self.old_parser_version,
            where="revision.old_parser_version",
        )
        _positive_int(
            self.new_parser_version,
            where="revision.new_parser_version",
        )
        if self.new_parser_version <= self.old_parser_version:
            raise AuthoredDiscourseCourseError(
                "new parser version 必须严格更高")
        if (not isinstance(self.mappings, tuple) or not self.mappings
                or any(not isinstance(item, DiscourseRevisionMappingSeed)
                       for item in self.mappings)):
            raise AuthoredDiscourseCourseError(
                "parser revision mappings 不能为空")
        if len({item.old_occurrence_id for item in self.mappings}) != len(
                self.mappings):
            raise AuthoredDiscourseCourseError(
                "parser revision old occurrence 重复")
        for name, value in (
                ("affected", self.affected_occurrence_ids),
                ("unaffected", self.unaffected_occurrence_ids)):
            if (not isinstance(value, tuple) or len(set(value)) != len(value)):
                raise AuthoredDiscourseCourseError(
                    f"parser revision {name} 非法或重复")
        if set(self.affected_occurrence_ids) & set(
                self.unaffected_occurrence_ids):
            raise AuthoredDiscourseCourseError(
                "affected/unaffected occurrence 不得重叠")
        if not set(item.old_occurrence_id for item in self.mappings) <= set(
                self.affected_occurrence_ids):
            raise AuthoredDiscourseCourseError(
                "anchor mapping old 必须属于 affected")
        if (not isinstance(self.recompute_query_ids, tuple)
                or not self.recompute_query_ids
                or any(type(item) is not int or item <= 0
                       for item in self.recompute_query_ids)
                or len(set(self.recompute_query_ids))
                != len(self.recompute_query_ids)):
            raise AuthoredDiscourseCourseError(
                "recompute query 非法或重复")

    @classmethod
    def from_dict(cls, value: Any) -> "DiscourseParserRevisionSeed":
        """从严格字段集合恢复 parser revision。"""
        if not isinstance(value, dict) or set(value) != _REVISION_FIELDS:
            raise AuthoredDiscourseCourseError(
                "parser revision 字段集合漂移")
        mappings = value["mappings"]
        if not isinstance(mappings, list):
            raise AuthoredDiscourseCourseError(
                "parser revision mappings 必须是列表")
        return cls(
            value["old_parser_version"],
            value["new_parser_version"],
            tuple(DiscourseRevisionMappingSeed.from_dict(item)
                  for item in mappings),
            _text_tuple(
                value["affected_occurrence_ids"],
                where="affected_occurrence_ids",
                allow_empty=True,
            ),
            _text_tuple(
                value["unaffected_occurrence_ids"],
                where="unaffected_occurrence_ids",
                allow_empty=True,
            ),
            _int_tuple(
                value["recompute_query_ids"],
                where="recompute_query_ids",
            ),
        )


@dataclass(frozen=True)
class DiscourseConsumerRequestSeed:
    """篇章字符、段落、occurrence、reference 和局部重算预算。"""

    max_context_chars: int
    max_paragraphs: int
    max_occurrences: int
    max_candidates: int
    max_recompute_queries: int

    def __post_init__(self) -> None:
        for name, value in (
                ("max_context_chars", self.max_context_chars),
                ("max_paragraphs", self.max_paragraphs),
                ("max_occurrences", self.max_occurrences),
                ("max_candidates", self.max_candidates),
                ("max_recompute_queries", self.max_recompute_queries)):
            _positive_int(value, where=f"DiscourseConsumerRequestSeed.{name}")
        if self.max_candidates > self.max_occurrences:
            raise AuthoredDiscourseCourseError(
                "reference candidate 预算不得超过 occurrence 预算")

    @classmethod
    def from_dict(cls, value: Any) -> "DiscourseConsumerRequestSeed":
        """从严格字段集合恢复 discourse consumer。"""
        if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
            raise AuthoredDiscourseCourseError(
                "discourse consumer 字段集合漂移")
        return cls(
            value["max_context_chars"],
            value["max_paragraphs"],
            value["max_occurrences"],
            value["max_candidates"],
            value["max_recompute_queries"],
        )


@dataclass(frozen=True)
class AuthoredDiscourseSeed:
    """一条 W-08/W-09 discourse、reference 和局部修正课程记录。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    source_key: str
    surface: str
    variant_kind: str
    paragraphs: tuple[DiscourseSpanSeed, ...]
    occurrences: tuple[DiscourseOccurrenceSeed, ...]
    reference_plan: DiscourseReferencePlanSeed | None
    parser_revision: DiscourseParserRevisionSeed | None
    consumer_request: DiscourseConsumerRequestSeed
    evidence_support: int
    evidence_refute: int
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
                ("surface", self.surface),
                ("variant_kind", self.variant_kind),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredDiscourseSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredDiscourseSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredDiscourseCourseError(
                "label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredDiscourseCourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredDiscourseCourseError(
                "sample_role 不属于 discourse 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredDiscourseCourseError(
                "supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredDiscourseCourseError(
                "非 supersede seed 不得声明替代目标")
        if self.source_key != SOURCE_KEY:
            raise AuthoredDiscourseCourseError("discourse source key 漂移")
        if self.variant_kind not in VARIANT_KINDS:
            raise AuthoredDiscourseCourseError("discourse variant 未注册")
        if (not isinstance(self.paragraphs, tuple) or not self.paragraphs
                or any(not isinstance(item, DiscourseSpanSeed)
                       for item in self.paragraphs)):
            raise AuthoredDiscourseCourseError("paragraphs 不能为空")
        if (not isinstance(self.occurrences, tuple) or not self.occurrences
                or any(not isinstance(item, DiscourseOccurrenceSeed)
                       for item in self.occurrences)):
            raise AuthoredDiscourseCourseError("occurrences 不能为空")
        paragraph_ids = [item.span_id for item in self.paragraphs]
        occurrence_ids = [item.occurrence_id for item in self.occurrences]
        if len(set(paragraph_ids)) != len(paragraph_ids):
            raise AuthoredDiscourseCourseError("paragraph span_id 重复")
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise AuthoredDiscourseCourseError("occurrence_id 重复")
        paragraph_order = [item.ordinal for item in self.paragraphs]
        occurrence_order = [item.ordinal for item in self.occurrences]
        if (paragraph_order != sorted(paragraph_order)
                or len(set(paragraph_order)) != len(paragraph_order)):
            raise AuthoredDiscourseCourseError(
                "paragraph ordinal 必须严格递增")
        if (occurrence_order != sorted(occurrence_order)
                or len(set(occurrence_order)) != len(occurrence_order)):
            raise AuthoredDiscourseCourseError(
                "occurrence ordinal 必须严格递增")
        if len({(item.start, item.end, item.ordinal)
                for item in self.paragraphs}) != len(self.paragraphs):
            raise AuthoredDiscourseCourseError("paragraph 身份坐标重复")
        if len({(item.start, item.end, item.ordinal)
                for item in self.occurrences}) != len(self.occurrences):
            raise AuthoredDiscourseCourseError("occurrence 身份坐标重复")
        for item in (*self.paragraphs, *self.occurrences):
            if item.end > len(self.surface) or self.surface[
                    item.start:item.end] != item.surface_fragment:
                raise AuthoredDiscourseCourseError(
                    "discourse span/occurrence 与 surface 不一致")
        if len(self.surface) > self.consumer_request.max_context_chars:
            raise AuthoredDiscourseCourseError(
                "surface 超过 context 字符预算")
        if len(self.paragraphs) > self.consumer_request.max_paragraphs:
            raise AuthoredDiscourseCourseError("paragraph 超过预算")
        if len(self.occurrences) > self.consumer_request.max_occurrences:
            raise AuthoredDiscourseCourseError("occurrence 超过预算")
        known = set(occurrence_ids)
        if self.reference_plan is not None:
            plan = self.reference_plan
            ids = {
                plan.reference_occurrence_id,
                *plan.window_occurrence_ids,
                *plan.candidate_occurrence_ids,
            }
            if not ids <= known:
                raise AuthoredDiscourseCourseError(
                    "reference plan 引用未知 occurrence")
            if plan.reference_occurrence_id in plan.window_occurrence_ids:
                raise AuthoredDiscourseCourseError(
                    "reference 自身不得进入 window")
            if len(plan.candidate_occurrence_ids) > (
                    self.consumer_request.max_candidates):
                raise AuthoredDiscourseCourseError(
                    "reference candidates 超过预算")
            if len(plan.impacted_query_ids) > (
                    self.consumer_request.max_recompute_queries):
                raise AuthoredDiscourseCourseError(
                    "reference impacted query 超过预算")
        if self.variant_kind in {"REFERENCE", "LATER_REINTERPRETATION"}:
            if self.reference_plan is None:
                raise AuthoredDiscourseCourseError(
                    "reference variant 必须声明有界 reference plan")
        if self.parser_revision is not None:
            revision = self.parser_revision
            revision_ids = {
                *revision.affected_occurrence_ids,
                *revision.unaffected_occurrence_ids,
                *(item.old_occurrence_id for item in revision.mappings),
                *(value for item in revision.mappings
                  for value in item.replacement_occurrence_ids),
            }
            if not revision_ids <= known:
                raise AuthoredDiscourseCourseError(
                    "parser revision 引用未知 occurrence")
            if len(revision.recompute_query_ids) > (
                    self.consumer_request.max_recompute_queries):
                raise AuthoredDiscourseCourseError(
                    "parser revision recompute 超过预算")
            if (not revision.affected_occurrence_ids
                    or not revision.unaffected_occurrence_ids):
                raise AuthoredDiscourseCourseError(
                    "parser revision 必须同时声明 affected/unaffected")
            mapped_old = {
                item.old_occurrence_id for item in revision.mappings}
            if mapped_old != set(revision.affected_occurrence_ids):
                raise AuthoredDiscourseCourseError(
                    "parser revision mapping 必须完整覆盖 affected")
            replacements = {
                value for item in revision.mappings
                for value in item.replacement_occurrence_ids}
            if replacements & set(revision.unaffected_occurrence_ids):
                raise AuthoredDiscourseCourseError(
                    "revision replacement 不得冒充 unaffected")
        if self.variant_kind in {"PARSER_REVISION", "LATER_REINTERPRETATION"}:
            if self.parser_revision is None:
                raise AuthoredDiscourseCourseError(
                    "revision variant 必须声明 parser revision")
        if self.variant_kind == "LATER_REINTERPRETATION":
            if (self.reference_plan is None
                    or not self.reference_plan.rejected_occurrence_id):
                raise AuthoredDiscourseCourseError(
                    "later reinterpretation 必须声明 reference replacement")
            if not self.reference_plan.impacted_query_ids:
                raise AuthoredDiscourseCourseError(
                    "later reinterpretation 必须声明局部受影响 query")
        _bit(
            self.evidence_support,
            where="AuthoredDiscourseSeed.evidence_support",
        )
        _bit(
            self.evidence_refute,
            where="AuthoredDiscourseSeed.evidence_refute",
        )
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredDiscourseCourseError("expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredDiscourseCourseError(
                "discourse perturbation 未注册")
        _positive_int(
            self.logical_order,
            where="AuthoredDiscourseSeed.logical_order",
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredDiscourseSeed":
        """从严格字段集合恢复 discourse seed。"""
        if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
            raise AuthoredDiscourseCourseError(
                "discourse seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredDiscourseCourseError(
                "discourse seed 必须是 CC0-1.0")
        paragraphs = value["paragraphs"]
        occurrences = value["occurrences"]
        if not isinstance(paragraphs, list) or not isinstance(occurrences, list):
            raise AuthoredDiscourseCourseError(
                "paragraphs/occurrences 必须是列表")
        raw_reference = value["reference_plan"]
        raw_revision = value["parser_revision"]
        return cls(
            _text(value["seed_id"], where="seed_id"),
            _text(value["family"], where="family"),
            _text(value["template_family"], where="template_family"),
            _text(value["label_owner"], where="label_owner"),
            _text(value["split"], where="split"),
            _text(value["sample_role"], where="sample_role"),
            _text(value["source_key"], where="source_key"),
            _text(value["surface"], where="surface"),
            _text(value["variant_kind"], where="variant_kind"),
            tuple(DiscourseSpanSeed.from_dict(item) for item in paragraphs),
            tuple(DiscourseOccurrenceSeed.from_dict(item)
                  for item in occurrences),
            (None if raw_reference is None
             else DiscourseReferencePlanSeed.from_dict(raw_reference)),
            (None if raw_revision is None
             else DiscourseParserRevisionSeed.from_dict(raw_revision)),
            DiscourseConsumerRequestSeed.from_dict(value["consumer_request"]),
            value["evidence_support"],
            value["evidence_refute"],
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
    "AuthoredDiscourseCourseError",
    "AuthoredDiscourseSeed",
    "DiscourseConsumerRequestSeed",
    "DiscourseOccurrenceSeed",
    "DiscourseParserRevisionSeed",
    "DiscourseReferencePlanSeed",
    "DiscourseRevisionMappingSeed",
    "DiscourseSpanSeed",
    "LICENSE_ID",
    "SOURCE_KEY",
    "VARIANT_KINDS",
]
