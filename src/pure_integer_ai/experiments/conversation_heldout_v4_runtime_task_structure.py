"""DLG-05 v4 无载荷运行时任务结构回放内核。

本模块只验证已构造的 QuestionRequest、Representation 与 Evidence plan 是否能机械地
闭合到一个明确 SourceRef 和 scalar 范围。它不读取原文、不持久化任务、不导入
runtime/capsule，也不从 hash 或固定样例猜测语义。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    OBJECT_SET_EXPR,
    OBJECT_SPAN,
    OBJECT_VARIABLE,
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4Representation,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import pack_key, strict_integer_tuple


_SOURCE_REF_KEY_SIZE = 11
_SOURCE_BEARING_KINDS = frozenset({
    OBJECT_BINDER,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    OBJECT_SET_EXPR,
    OBJECT_VARIABLE,
})
_EVIDENCE_STANCES = frozenset({
    EVIDENCE_SUPPORT,
    EVIDENCE_REFUTE,
    EVIDENCE_UNKNOWN,
})


class ConversationHeldOutV4RuntimeTaskStructureError(RuntimeError):
    """无载荷任务结构、来源锚点或范围无法闭合。"""


def _fail(message: str) -> None:
    """以统一、可由上层边界转换的错误拒绝结构漂移。"""
    raise ConversationHeldOutV4RuntimeTaskStructureError(message)


def _require_span(start: int, end: int, *, lower: int, upper: int, label: str) -> None:
    """验证一个非空 scalar 范围完全落在调用方给出的权威 witness 内。"""
    if (type(start) is not int or type(end) is not int
            or start < lower or start >= end or end > upper):
        _fail(f"{label} 越过 witness scalar 范围")


def parse_v4_source_anchor(
        anchor: ObjectIdentity, *, label: str,
        ) -> tuple[SourceRef, str, int, tuple[tuple[int, int], ...]]:
    """解析并重建 Occurrence/Span anchor，拒绝 opaque 或伪造 identity。"""
    if not isinstance(anchor, ObjectIdentity):
        _fail(f"{label} 必须是 ObjectIdentity")
    if anchor.object_kind not in {OBJECT_OCCURRENCE, OBJECT_SPAN}:
        _fail(f"{label} 必须是 Occurrence 或 Span")
    try:
        source = SourceRef.from_stable_key(tuple(anchor.components[:_SOURCE_REF_KEY_SIZE]))
    except (TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskStructureError(
            f"{label} 来源前缀非法") from exc
    try:
        if anchor.object_kind == OBJECT_OCCURRENCE:
            if len(anchor.components) != _SOURCE_REF_KEY_SIZE + 3:
                raise ValueError("Occurrence 长度非法")
            start, end, ordinal = anchor.components[_SOURCE_REF_KEY_SIZE:]
            rebuilt = occurrence_identity(source, start=start, end=end, ordinal=ordinal)
            members = ((start, end),)
            kind = "occurrence"
        else:
            if len(anchor.components) < _SOURCE_REF_KEY_SIZE + 4:
                raise ValueError("Span 长度非法")
            ordinal = anchor.components[_SOURCE_REF_KEY_SIZE]
            count = anchor.components[_SOURCE_REF_KEY_SIZE + 1]
            values = anchor.components[_SOURCE_REF_KEY_SIZE + 2:]
            if count <= 0 or len(values) != count * 2:
                raise ValueError("Span members 长度非法")
            members = tuple(
                (values[index], values[index + 1])
                for index in range(0, len(values), 2)
            )
            rebuilt = span_identity(source, members=members, ordinal=ordinal)
            kind = "span"
    except (TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskStructureError(
            f"{label} 结构非法") from exc
    if rebuilt != anchor:
        _fail(f"{label} identity 与结构不一致")
    return source, kind, ordinal, members


def validate_v4_bound_proposition_source(
        root: BoundProposition,
        *,
        source: SourceRef,
        scalar_start: int,
        scalar_end: int,
        label: str,
        ) -> None:
    """递归回放 BoundProposition 的全部来源化成员与 anchor scalar 范围。"""
    if not isinstance(root, BoundProposition):
        raise TypeError(f"{label} 必须是 BoundProposition")
    if not isinstance(source, SourceRef):
        raise TypeError(f"{label} source 类型错误")
    _require_span(scalar_start, scalar_end, lower=0, upper=scalar_end,
                  label=f"{label} witness")

    def require_source(identity: ObjectIdentity, *, where: str) -> None:
        try:
            actual = semantic_source(identity)
        except (TypeError, ValueError) as exc:
            raise ConversationHeldOutV4RuntimeTaskStructureError(
                f"{label} {where} 来源 identity 非法") from exc
        if actual != source:
            _fail(f"{label} {where} 跨 SourceRef")

    def visit(value: BoundProposition) -> None:
        require_source(value.template, where="template")
        require_source(value.context, where="context")
        anchor_source, _kind, _ordinal, members = parse_v4_source_anchor(
            value.source_anchor, label=f"{label} source anchor")
        if anchor_source != source:
            _fail(f"{label} source anchor 跨 SourceRef")
        for start, end in members:
            _require_span(start, end, lower=scalar_start, upper=scalar_end,
                          label=f"{label} source anchor")
        if len(set(value.introduced_binders)) != len(value.introduced_binders):
            _fail(f"{label} introduced Binder 不得重复")
        if value.introduced_binders != tuple(sorted(
                value.introduced_binders,
                key=lambda item: item.stable_key())):
            _fail(f"{label} introduced Binder 必须按 stable key 排序")
        for binder in value.introduced_binders:
            require_source(binder, where="Binder")
        for variable in value.applied_variables:
            require_source(variable, where="Variable")
        for binding in value.bindings:
            if isinstance(binding.filler, BoundProposition):
                visit(binding.filler)
            elif binding.filler.object_kind in {OBJECT_OCCURRENCE, OBJECT_SPAN}:
                anchor_source, _kind, _ordinal, members = parse_v4_source_anchor(
                    binding.filler, label=f"{label} role filler anchor")
                if anchor_source != source:
                    _fail(f"{label} role filler anchor 跨 SourceRef")
                for start, end in members:
                    _require_span(start, end, lower=scalar_start, upper=scalar_end,
                                  label=f"{label} role filler anchor")
            elif binding.filler.object_kind in _SOURCE_BEARING_KINDS:
                require_source(binding.filler, where="role filler")

    visit(root)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BlueprintRepresentation:
    """带显式 annotation/transform 来源的内存 Representation。"""

    representation: ConversationHeldOutV4Representation
    source_span_start: int
    source_span_end: int
    language_scope: ScopeIdentity
    annotation_source_identity: ProtocolKey
    annotation_revision_identity: ProtocolKey
    transform_code_identity: ProtocolKey

    def __post_init__(self) -> None:
        """限制 blueprint representation 只携带 typed 内容和完整构造身份。"""
        if not isinstance(self.representation, ConversationHeldOutV4Representation):
            raise TypeError("blueprint Representation 类型错误")
        if not isinstance(self.language_scope, ScopeIdentity):
            raise TypeError("blueprint Representation language scope 类型错误")
        for label, value in (
                ("annotation source", self.annotation_source_identity),
                ("annotation revision", self.annotation_revision_identity),
                ("transform code", self.transform_code_identity)):
            if not isinstance(value, ProtocolKey):
                raise TypeError(f"blueprint Representation {label} identity 类型错误")
        if (type(self.source_span_start) is not int
                or type(self.source_span_end) is not int):
            raise ValueError("blueprint Representation scalar span 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回仅供内存闭合和 receipt fingerprint 使用的完整 typed identity。"""
        result = []
        for value in (
                self.representation.stable_key(),
                (self.source_span_start, self.source_span_end),
                self.language_scope.stable_key(),
                self.annotation_source_identity.components,
                self.annotation_revision_identity.components,
                self.transform_code_identity.components):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4BlueprintEvidencePlan:
    """payload-free blueprint 阶段的 Evidence 目标、stance 与来源范围。"""

    target: BoundProposition
    competition_key: tuple[int, ...]
    stances: tuple[int, ...]
    source: SourceRef
    source_span_start: int
    source_span_end: int

    def __post_init__(self) -> None:
        """拒绝无 typed target、重复 stance 或无来源范围的计划。"""
        if not isinstance(self.target, BoundProposition):
            raise TypeError("blueprint Evidence target 类型错误")
        strict_integer_tuple(self.competition_key, label="blueprint Evidence competition key")
        if (not isinstance(self.stances, tuple) or not self.stances
                or any(item not in _EVIDENCE_STANCES for item in self.stances)
                or len(set(self.stances)) != len(self.stances)):
            raise ValueError("blueprint Evidence stances 非法")
        if not isinstance(self.source, SourceRef):
            raise TypeError("blueprint Evidence source 类型错误")
        if (type(self.source_span_start) is not int
                or type(self.source_span_end) is not int):
            raise ValueError("blueprint Evidence source span 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 Evidence identity；调用方不得以此替代 target 本体。"""
        result = []
        for value in (
                self.target.stable_key(), self.competition_key, self.stances,
                self.source.stable_key(),
                (self.source_span_start, self.source_span_end)):
            pack_key(result, value)
        return tuple(result)


def validate_v4_question_request_structure(
        request: QuestionRequest,
        *, source: SourceRef, scalar_start: int, scalar_end: int,
        label: str,
        ) -> None:
    """验证问题 target、scope 和全部授权 target 均闭合到同一 witness。"""
    if not isinstance(request, QuestionRequest):
        raise TypeError(f"{label} request 类型错误")
    if request.source != source:
        _fail(f"{label} request SourceRef 与 witness 不一致")
    if (request.evidence_scope.source != source
            or request.response_scope.source != source):
        _fail(f"{label} request scope 与 witness 不一致")
    validate_v4_bound_proposition_source(
        request.target, source=source, scalar_start=scalar_start,
        scalar_end=scalar_end, label=f"{label} target")
    for index, target in enumerate(request.authorized_candidate_targets):
        validate_v4_bound_proposition_source(
            target, source=source, scalar_start=scalar_start,
            scalar_end=scalar_end, label=f"{label} authorized target {index}")


def validate_v4_blueprint_representations(
        representations: tuple[ConversationHeldOutV4BlueprintRepresentation, ...],
        *, request: QuestionRequest, source: SourceRef,
        scalar_start: int, scalar_end: int, label: str,
        ) -> None:
    """验证 Representation 连续序、annotation 链及 language scope 的来源兼容性。"""
    if (not isinstance(representations, tuple) or not representations
            or any(not isinstance(item, ConversationHeldOutV4BlueprintRepresentation)
                   for item in representations)):
        raise TypeError(f"{label} Representations 类型错误")
    if tuple(item.representation.ordinal for item in representations) != tuple(
            range(len(representations))):
        _fail(f"{label} Representation ordinal 不连续")
    for item in representations:
        _require_span(item.source_span_start, item.source_span_end,
                      lower=scalar_start, upper=scalar_end,
                      label=f"{label} Representation")
        if item.language_scope.source != source:
            _fail(f"{label} Representation language scope 跨 SourceRef")
        if item.language_scope not in {
                request.evidence_scope, request.response_scope}:
            _fail(f"{label} Representation language scope 未显式绑定 request scope")


def validate_v4_blueprint_evidence_plans(
        plans: tuple[ConversationHeldOutV4BlueprintEvidencePlan, ...],
        *, request: QuestionRequest, source: SourceRef,
        scalar_start: int, scalar_end: int, label: str,
        ) -> None:
    """验证每个 Evidence plan 的来源范围，且 planned target 与 request 双向闭合。"""
    if (not isinstance(plans, tuple)
            or any(not isinstance(item, ConversationHeldOutV4BlueprintEvidencePlan)
                   for item in plans)):
        raise TypeError(f"{label} Evidence plans 类型错误")
    if plans != tuple(sorted(plans, key=lambda item: item.target.stable_key())):
        _fail(f"{label} Evidence plans 必须按 target 排序")
    planned = tuple(item.target for item in plans)
    if len(set(planned)) != len(planned):
        _fail(f"{label} Evidence plan target 不得重复")
    expected = request.authorized_candidate_targets or (request.target,)
    if set(planned) != set(expected):
        _fail(f"{label} Evidence plan target 与 request 未双向闭合")
    for item in plans:
        if item.source != source:
            _fail(f"{label} Evidence plan 跨 SourceRef")
        _require_span(item.source_span_start, item.source_span_end,
                      lower=scalar_start, upper=scalar_end,
                      label=f"{label} Evidence plan")
        validate_v4_bound_proposition_source(
            item.target, source=source, scalar_start=scalar_start,
            scalar_end=scalar_end, label=f"{label} Evidence target")


def validate_v4_runtime_task_structure(
        request: QuestionRequest,
        representations: tuple[ConversationHeldOutV4BlueprintRepresentation, ...],
        evidence_plans: tuple[ConversationHeldOutV4BlueprintEvidencePlan, ...],
        *, source: SourceRef, scalar_start: int, scalar_end: int,
        label: str,
        ) -> None:
    """一次性重放 blueprint turn 的问题、表示和 Evidence 结构，不读取 raw payload。"""
    validate_v4_question_request_structure(
        request, source=source, scalar_start=scalar_start, scalar_end=scalar_end,
        label=label)
    validate_v4_blueprint_representations(
        representations, request=request, source=source, scalar_start=scalar_start,
        scalar_end=scalar_end, label=label)
    validate_v4_blueprint_evidence_plans(
        evidence_plans, request=request, source=source, scalar_start=scalar_start,
        scalar_end=scalar_end, label=label)


__all__ = [
    "ConversationHeldOutV4BlueprintEvidencePlan",
    "ConversationHeldOutV4BlueprintRepresentation",
    "ConversationHeldOutV4RuntimeTaskStructureError",
    "parse_v4_source_anchor",
    "validate_v4_blueprint_evidence_plans",
    "validate_v4_blueprint_representations",
    "validate_v4_bound_proposition_source",
    "validate_v4_question_request_structure",
    "validate_v4_runtime_task_structure",
]
