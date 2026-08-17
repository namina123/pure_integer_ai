"""来源约束问答与多表面回答组织的 TRAIN-only 极小合同。

本模块只冻结资料形状和独立结构核验，不把问句直接映射为唯一答案字符串，也不
调用 teacher/LLM。实际 surface parser 后续可把输出恢复为 ``SurfaceRealization``，
再复用这里的同一验证维度接入既有 G-04。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_line,
    parse_canonical_json_bytes,
)


ARTIFACT_KIND = "PH2_GROUNDED_ANSWER_EPISODE_V1"
LICENSE_ID = "CC0-1.0"
RESPONSE_ACTS = frozenset({"ANSWER", "UNKNOWN", "CLARIFY", "CONFLICT"})
CARRIER_KINDS = frozenset({"PLAIN_TEXT", "MARKDOWN", "HTML", "CODE", "TABLE"})
SPLITS = frozenset({"train", "dev", "held_out"})
VERIFICATION_VIOLATIONS = (
    "RESPONSE_ACT_DRIFT",
    "SCOPE_DRIFT",
    "MISSING_REQUIRED_CLAIM",
    "FORBIDDEN_CLAIM",
    "UNSUPPORTED_CLAIM",
    "NONANSWER_CLAIM",
    "MISSING_CITATION",
    "FOREIGN_CITATION",
    "UNPLANNED_CITATION",
)
REFERENCE_GRANULARITY = "PROPOSITION_EVENT"
REFERENCE_STRATEGIES = (
    "ANTECEDENT_REFERENCE",
    "EXPLICIT_REPETITION",
)
REFERENCE_CHALLENGE_KINDS = (
    "AMBIGUOUS_ANTECEDENT",
    "FUTURE_ANTECEDENT",
    "MISSING_REFERENCE_RELATION",
    "SCOPE_OUTSIDE_ANTECEDENT",
    "SURFACE_EXTRA_CLAIM",
    "WRONG_ANTECEDENT",
)


# object-model: exception
class GroundedAnswerCourseError(ValueError):
    """grounded answer 资料、split 或 verification 不满足冻结合同。"""


def _exact(value: Any, keys: frozenset[str], *, where: str) -> dict[str, Any]:
    """核验 JSON object 使用精确字段集合。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise GroundedAnswerCourseError(f"{where} 字段集合漂移")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """核验无首尾空白文本。"""
    if not isinstance(value, str) or value.strip() != value:
        raise GroundedAnswerCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise GroundedAnswerCourseError(f"{where} 不能为空")
    return value


def _positive(value: Any, *, where: str) -> int:
    """核验正严格整数身份。"""
    if type(value) is not int or value <= 0:
        raise GroundedAnswerCourseError(f"{where} 必须是正严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """核验严格整数位。"""
    if type(value) is not int or value not in {0, 1}:
        raise GroundedAnswerCourseError(f"{where} 必须是严格整数 0/1")
    return value


def _strings(
        value: Any, *, where: str, allow_empty: bool = False,
        ) -> tuple[str, ...]:
    """从 JSON list 恢复无重复字符串 tuple。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise GroundedAnswerCourseError(f"{where} 非法或重复")
    return tuple(value)


def _positive_ints(
        value: Any, *, where: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """从 JSON list 恢复无重复正整数 tuple。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(type(item) is not int or item <= 0 for item in value)
            or len(set(value)) != len(value)):
        raise GroundedAnswerCourseError(f"{where} 非法或重复")
    return tuple(value)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerSplitClusters:
    """冻结 source、Proposition、问句构造和改写四条 split 轴。"""

    source: str
    proposition: str
    question_construction: str
    paraphrase: str

    def __post_init__(self) -> None:
        for name, value in (
                ("source", self.source),
                ("proposition", self.proposition),
                ("question_construction", self.question_construction),
                ("paraphrase", self.paraphrase)):
            _text(value, where=f"split_clusters.{name}")

    def to_dict(self) -> dict[str, str]:
        """导出规范 JSON 值。"""
        return {
            "paraphrase": self.paraphrase,
            "proposition": self.proposition,
            "question_construction": self.question_construction,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedAnswerSplitClusters":
        """从精确字段恢复四轴 split key。"""
        raw = _exact(value, frozenset({
            "source", "proposition", "question_construction", "paraphrase",
        }), where="split_clusters")
        return cls(raw["source"], raw["proposition"],
                   raw["question_construction"], raw["paraphrase"])


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedEvidence:
    """一条来源化 Evidence 对一个命题 claim 的支持或反驳。"""

    evidence_id: str
    proposition_id: str
    source_id: str
    scope_id: int
    claim_text: str
    evidence_text: str
    support: int
    refute: int

    def __post_init__(self) -> None:
        for name, value in (
                ("evidence_id", self.evidence_id),
                ("proposition_id", self.proposition_id),
                ("source_id", self.source_id),
                ("claim_text", self.claim_text),
                ("evidence_text", self.evidence_text)):
            _text(value, where=f"evidence.{name}")
        _positive(self.scope_id, where="evidence.scope_id")
        _bit(self.support, where="evidence.support")
        _bit(self.refute, where="evidence.refute")
        if not self.support and not self.refute:
            raise GroundedAnswerCourseError("Evidence 至少承担一个方向")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "claim_text": self.claim_text,
            "evidence_id": self.evidence_id,
            "evidence_text": self.evidence_text,
            "proposition_id": self.proposition_id,
            "refute": self.refute,
            "scope_id": self.scope_id,
            "source_id": self.source_id,
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedEvidence":
        """从严格 Evidence object 恢复记录。"""
        raw = _exact(value, frozenset({
            "claim_text", "evidence_id", "evidence_text", "proposition_id",
            "refute", "scope_id", "source_id", "support",
        }), where="evidence")
        return cls(
            raw["evidence_id"], raw["proposition_id"], raw["source_id"],
            raw["scope_id"], raw["claim_text"], raw["evidence_text"],
            raw["support"], raw["refute"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerPlan:
    """回答组织层消费的 response act、claim 顺序和来源要求。"""

    response_act: str
    ordered_claim_ids: tuple[str, ...]
    required_claim_ids: tuple[str, ...]
    forbidden_claim_ids: tuple[str, ...]
    citation_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.response_act not in RESPONSE_ACTS:
            raise GroundedAnswerCourseError("answer_plan response_act 未注册")
        for name, values in (
                ("ordered", self.ordered_claim_ids),
                ("required", self.required_claim_ids),
                ("forbidden", self.forbidden_claim_ids),
                ("citation", self.citation_source_ids)):
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, str) or not item
                           for item in values)
                    or len(set(values)) != len(values)):
                raise GroundedAnswerCourseError(
                    f"answer_plan {name} 非法或重复")
        if set(self.ordered_claim_ids) != set(self.required_claim_ids):
            raise GroundedAnswerCourseError(
                "answer_plan ordered 必须精确排列 required claim")
        if set(self.required_claim_ids) & set(self.forbidden_claim_ids):
            raise GroundedAnswerCourseError(
                "required/forbidden claim 不得重叠")
        if self.response_act == "ANSWER":
            if not self.required_claim_ids or not self.citation_source_ids:
                raise GroundedAnswerCourseError(
                    "ANSWER plan 必须有 claim 和 citation 来源")
        elif self.ordered_claim_ids or self.required_claim_ids:
            raise GroundedAnswerCourseError(
                "非 ANSWER plan 不得伪造肯定 claim")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "citation_source_ids": list(self.citation_source_ids),
            "forbidden_claim_ids": list(self.forbidden_claim_ids),
            "ordered_claim_ids": list(self.ordered_claim_ids),
            "required_claim_ids": list(self.required_claim_ids),
            "response_act": self.response_act,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedAnswerPlan":
        """从严格 answer plan object 恢复记录。"""
        raw = _exact(value, frozenset({
            "citation_source_ids", "forbidden_claim_ids",
            "ordered_claim_ids", "required_claim_ids", "response_act",
        }), where="answer_plan")
        return cls(
            _text(raw["response_act"], where="answer_plan.response_act"),
            _strings(raw["ordered_claim_ids"], where="ordered_claim_ids",
                     allow_empty=True),
            _strings(raw["required_claim_ids"], where="required_claim_ids",
                     allow_empty=True),
            _strings(raw["forbidden_claim_ids"], where="forbidden_claim_ids",
                     allow_empty=True),
            _strings(raw["citation_source_ids"], where="citation_source_ids",
                     allow_empty=True),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedQuestionEpisode:
    """分离原始问句、typed intent、Evidence 与 answer plan。"""

    typed_intent: str
    context_surface: str
    question_surface: str
    evidence_scope_id: int
    response_scope_id: int
    evidence: tuple[GroundedEvidence, ...]
    answer_plan: GroundedAnswerPlan

    def __post_init__(self) -> None:
        for name, value in (
                ("typed_intent", self.typed_intent),
                ("context_surface", self.context_surface),
                ("question_surface", self.question_surface)):
            _text(value, where=f"question.{name}")
        _positive(self.evidence_scope_id, where="question.evidence_scope_id")
        _positive(self.response_scope_id, where="question.response_scope_id")
        if self.evidence_scope_id == self.response_scope_id:
            raise GroundedAnswerCourseError(
                "Evidence scope 与 response scope 不得混用")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(item, GroundedEvidence)
                       for item in self.evidence)):
            raise GroundedAnswerCourseError("question evidence 类型错误")
        if not isinstance(self.answer_plan, GroundedAnswerPlan):
            raise GroundedAnswerCourseError("question answer_plan 类型错误")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise GroundedAnswerCourseError("question Evidence id 重复")
        if any(item.scope_id != self.evidence_scope_id for item in self.evidence):
            raise GroundedAnswerCourseError("Evidence scope 漂移")
        aggregate: dict[str, tuple[bool, bool]] = {}
        claim_texts: dict[str, str] = {}
        supporting_sources: dict[str, set[str]] = {}
        for item in self.evidence:
            prior_text = claim_texts.get(item.proposition_id)
            if prior_text is not None and prior_text != item.claim_text:
                raise GroundedAnswerCourseError(
                    "同一 Proposition 的 claim_text 不一致")
            claim_texts[item.proposition_id] = item.claim_text
            support, refute = aggregate.get(item.proposition_id, (False, False))
            aggregate[item.proposition_id] = (
                support or bool(item.support), refute or bool(item.refute))
            if item.support:
                supporting_sources.setdefault(
                    item.proposition_id, set()).add(item.source_id)
        answerable = {
            claim_id for claim_id, (support, refute) in aggregate.items()
            if support and not refute
        }
        if not set(self.answer_plan.required_claim_ids) <= answerable:
            raise GroundedAnswerCourseError(
                "answer plan required claim 没有无冲突 Evidence")
        sources = {item.source_id for item in self.evidence}
        if not set(self.answer_plan.citation_source_ids) <= sources:
            raise GroundedAnswerCourseError(
                "answer plan citation 不属于当前 Evidence")
        citations = set(self.answer_plan.citation_source_ids)
        if any(not citations & supporting_sources.get(claim_id, set())
               for claim_id in self.answer_plan.required_claim_ids):
            raise GroundedAnswerCourseError(
                "answer plan citation 未覆盖 required claim 支持来源")
        if (self.answer_plan.response_act == "CONFLICT"
                and not any(support and refute
                            for support, refute in aggregate.values())):
            raise GroundedAnswerCourseError(
                "CONFLICT plan 缺少双向 Evidence")
        if self.answer_plan.response_act == "CLARIFY":
            if any(support and refute
                   for support, refute in aggregate.values()):
                raise GroundedAnswerCourseError(
                    "CLARIFY plan 不得包含冲突 Evidence")
            if len(answerable) < 2:
                raise GroundedAnswerCourseError(
                    "CLARIFY plan 缺少多个可回答候选")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "answer_plan": self.answer_plan.to_dict(),
            "context_surface": self.context_surface,
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_scope_id": self.evidence_scope_id,
            "question_surface": self.question_surface,
            "response_scope_id": self.response_scope_id,
            "typed_intent": self.typed_intent,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedQuestionEpisode":
        """从严格 question object 恢复记录。"""
        raw = _exact(value, frozenset({
            "answer_plan", "context_surface", "evidence",
            "evidence_scope_id", "question_surface", "response_scope_id",
            "typed_intent",
        }), where="question")
        if not isinstance(raw["evidence"], list):
            raise GroundedAnswerCourseError("question evidence 必须是 list")
        return cls(
            _text(raw["typed_intent"], where="question.typed_intent"),
            _text(raw["context_surface"], where="question.context_surface"),
            _text(raw["question_surface"], where="question.question_surface"),
            raw["evidence_scope_id"], raw["response_scope_id"],
            tuple(GroundedEvidence.from_dict(item)
                  for item in raw["evidence"]),
            GroundedAnswerPlan.from_dict(raw["answer_plan"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """一个来源资料中的用户或助手会话 turn。"""

    turn_id: int
    speaker: str
    surface: str
    scope_ids: tuple[int, ...]
    reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _positive(self.turn_id, where="dialogue.turn_id")
        if self.speaker not in {"USER", "ASSISTANT"}:
            raise GroundedAnswerCourseError("dialogue speaker 未注册")
        _text(self.surface, where="dialogue.surface")
        if (not isinstance(self.scope_ids, tuple) or not self.scope_ids
                or any(type(item) is not int or item <= 0
                       for item in self.scope_ids)
                or len(set(self.scope_ids)) != len(self.scope_ids)):
            raise GroundedAnswerCourseError("dialogue scope_ids 非法或重复")
        if (not isinstance(self.reference_ids, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.reference_ids)
                or len(set(self.reference_ids)) != len(self.reference_ids)):
            raise GroundedAnswerCourseError(
                "dialogue reference_ids 非法或重复")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "reference_ids": list(self.reference_ids),
            "scope_ids": list(self.scope_ids),
            "speaker": self.speaker,
            "surface": self.surface,
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DialogueTurn":
        """从严格 turn object 恢复记录。"""
        raw = _exact(value, frozenset({
            "reference_ids", "scope_ids", "speaker", "surface", "turn_id",
        }), where="dialogue.turn")
        return cls(
            raw["turn_id"],
            _text(raw["speaker"], where="dialogue.speaker"),
            _text(raw["surface"], where="dialogue.surface"),
            _positive_ints(raw["scope_ids"], where="dialogue.scope_ids"),
            _strings(raw["reference_ids"], where="dialogue.reference_ids",
                     allow_empty=True),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class DialogueEpisode:
    """保留当前问句所在会话、活动 scope 与显式引用。"""

    turns: tuple[DialogueTurn, ...]
    active_scope_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.turns, tuple) or not self.turns
                or any(not isinstance(item, DialogueTurn)
                       for item in self.turns)):
            raise GroundedAnswerCourseError("dialogue turns 必须非空")
        ids = tuple(item.turn_id for item in self.turns)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedAnswerCourseError(
                "dialogue turn_id 必须严格递增")
        if self.turns[-1].speaker != "USER":
            raise GroundedAnswerCourseError("当前问句必须是最后一个 USER turn")
        if (not isinstance(self.active_scope_ids, tuple)
                or not self.active_scope_ids
                or any(type(item) is not int or item <= 0
                       for item in self.active_scope_ids)
                or len(set(self.active_scope_ids))
                != len(self.active_scope_ids)):
            raise GroundedAnswerCourseError(
                "dialogue active_scope_ids 非法或重复")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "active_scope_ids": list(self.active_scope_ids),
            "turns": [item.to_dict() for item in self.turns],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DialogueEpisode":
        """从严格 dialogue object 恢复记录。"""
        raw = _exact(value, frozenset({"active_scope_ids", "turns"}),
                     where="dialogue")
        if not isinstance(raw["turns"], list):
            raise GroundedAnswerCourseError("dialogue turns 必须是 list")
        return cls(
            tuple(DialogueTurn.from_dict(item) for item in raw["turns"]),
            _positive_ints(raw["active_scope_ids"],
                           where="dialogue.active_scope_ids"),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SurfaceRealization:
    """一个 surface parser 可恢复的回答表面及 typed 注释。"""

    realization_id: str
    surface: str
    carrier_kind: str
    response_act: str
    scope_id: int
    claim_ids: tuple[str, ...]
    cited_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.realization_id, where="realization.id")
        _text(self.surface, where="realization.surface")
        if self.carrier_kind not in CARRIER_KINDS:
            raise GroundedAnswerCourseError("realization carrier_kind 未注册")
        if self.response_act not in RESPONSE_ACTS:
            raise GroundedAnswerCourseError("realization response_act 未注册")
        _positive(self.scope_id, where="realization.scope_id")
        for name, values in (
                ("claim_ids", self.claim_ids),
                ("cited_source_ids", self.cited_source_ids)):
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, str) or not item
                           for item in values)
                    or len(set(values)) != len(values)):
                raise GroundedAnswerCourseError(
                    f"realization {name} 非法或重复")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "carrier_kind": self.carrier_kind,
            "cited_source_ids": list(self.cited_source_ids),
            "claim_ids": list(self.claim_ids),
            "realization_id": self.realization_id,
            "response_act": self.response_act,
            "scope_id": self.scope_id,
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SurfaceRealization":
        """从严格 surface object 恢复记录。"""
        raw = _exact(value, frozenset({
            "carrier_kind", "cited_source_ids", "claim_ids",
            "realization_id", "response_act", "scope_id", "surface",
        }), where="realization")
        return cls(
            _text(raw["realization_id"], where="realization.id"),
            _text(raw["surface"], where="realization.surface"),
            _text(raw["carrier_kind"], where="realization.carrier_kind"),
            _text(raw["response_act"], where="realization.response_act"),
            raw["scope_id"],
            _strings(raw["claim_ids"], where="realization.claim_ids",
                     allow_empty=True),
            _strings(raw["cited_source_ids"],
                     where="realization.cited_source_ids", allow_empty=True),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RejectedSurfaceRealization:
    """一个负例 surface 及其预注册失败维度。"""

    realization: SurfaceRealization
    expected_violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.realization, SurfaceRealization):
            raise GroundedAnswerCourseError("rejected realization 类型错误")
        if (not isinstance(self.expected_violations, tuple)
                or not self.expected_violations
                or any(item not in VERIFICATION_VIOLATIONS
                       for item in self.expected_violations)
                or self.expected_violations != tuple(sorted(
                    set(self.expected_violations),
                    key=VERIFICATION_VIOLATIONS.index))):
            raise GroundedAnswerCourseError(
                "rejected expected_violations 非规范")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "expected_violations": list(self.expected_violations),
            "realization": self.realization.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RejectedSurfaceRealization":
        """从严格 rejected object 恢复记录。"""
        raw = _exact(value, frozenset({
            "expected_violations", "realization",
        }), where="rejected_realization")
        return cls(
            SurfaceRealization.from_dict(raw["realization"]),
            _strings(raw["expected_violations"],
                     where="expected_violations"),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SurfaceRealizationSet:
    """同一 answer plan 的多个合法 surface 与分型负例。"""

    accepted: tuple[SurfaceRealization, ...]
    rejected: tuple[RejectedSurfaceRealization, ...]
    minimum_legal_surfaces: int

    def __post_init__(self) -> None:
        if (not isinstance(self.accepted, tuple)
                or any(not isinstance(item, SurfaceRealization)
                       for item in self.accepted)):
            raise GroundedAnswerCourseError("accepted surface 类型错误")
        if (not isinstance(self.rejected, tuple)
                or any(not isinstance(item, RejectedSurfaceRealization)
                       for item in self.rejected)):
            raise GroundedAnswerCourseError("rejected surface 类型错误")
        _positive(self.minimum_legal_surfaces,
                  where="minimum_legal_surfaces")
        if len(self.accepted) < self.minimum_legal_surfaces:
            raise GroundedAnswerCourseError("合法 surface 数低于冻结下限")
        all_items = (
            *self.accepted,
            *(item.realization for item in self.rejected),
        )
        ids = tuple(item.realization_id for item in all_items)
        surfaces = tuple(item.surface for item in all_items)
        if len(set(ids)) != len(ids) or len(set(surfaces)) != len(surfaces):
            raise GroundedAnswerCourseError(
                "realization id 或 surface 不得重复")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "accepted": [item.to_dict() for item in self.accepted],
            "minimum_legal_surfaces": self.minimum_legal_surfaces,
            "rejected": [item.to_dict() for item in self.rejected],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SurfaceRealizationSet":
        """从严格 surface set object 恢复记录。"""
        raw = _exact(value, frozenset({
            "accepted", "minimum_legal_surfaces", "rejected",
        }), where="surface_set")
        if not isinstance(raw["accepted"], list) or not isinstance(
                raw["rejected"], list):
            raise GroundedAnswerCourseError(
                "surface accepted/rejected 必须是 list")
        return cls(
            tuple(SurfaceRealization.from_dict(item)
                  for item in raw["accepted"]),
            tuple(RejectedSurfaceRealization.from_dict(item)
                  for item in raw["rejected"]),
            raw["minimum_legal_surfaces"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationVerification:
    """回答表面在 response act、scope、claim 与 citation 上的独立结果。"""

    verdict: str
    violations: tuple[str, ...]
    response_act_match: int
    scope_match: int
    required_claims_covered: int
    forbidden_claims_absent: int
    supported_claims_only: int
    citation_complete: int

    def __post_init__(self) -> None:
        if self.verdict not in {"PASS", "FAIL"}:
            raise GroundedAnswerCourseError("verification verdict 非法")
        if (not isinstance(self.violations, tuple)
                or any(item not in VERIFICATION_VIOLATIONS
                       for item in self.violations)
                or self.violations != tuple(sorted(
                    set(self.violations),
                    key=VERIFICATION_VIOLATIONS.index))):
            raise GroundedAnswerCourseError(
                "verification violations 非规范")
        for name, value in (
                ("response_act_match", self.response_act_match),
                ("scope_match", self.scope_match),
                ("required_claims_covered", self.required_claims_covered),
                ("forbidden_claims_absent", self.forbidden_claims_absent),
                ("supported_claims_only", self.supported_claims_only),
                ("citation_complete", self.citation_complete)):
            _bit(value, where=f"verification.{name}")
        if (self.verdict == "PASS") != (not self.violations):
            raise GroundedAnswerCourseError(
                "verification verdict 与 violations 不一致")

    @property
    def passed(self) -> bool:
        """仅在全部硬维度无失败时返回真。"""
        return self.verdict == "PASS"

    def to_dict(self) -> dict[str, object]:
        """导出可进入 teacher Evidence 的分维验证结果。"""
        return {
            "citation_complete": self.citation_complete,
            "forbidden_claims_absent": self.forbidden_claims_absent,
            "required_claims_covered": self.required_claims_covered,
            "response_act_match": self.response_act_match,
            "scope_match": self.scope_match,
            "supported_claims_only": self.supported_claims_only,
            "verdict": self.verdict,
            "violations": list(self.violations),
        }


def verify_surface_realization(
        question: GroundedQuestionEpisode,
        realization: SurfaceRealization,
        ) -> GenerationVerification:
    """只按 typed plan、Evidence、scope 和 parser 注释复核一个 surface。"""
    if not isinstance(question, GroundedQuestionEpisode):
        raise TypeError("verification question 类型错误")
    if not isinstance(realization, SurfaceRealization):
        raise TypeError("verification realization 类型错误")
    plan = question.answer_plan
    claim_ids = set(realization.claim_ids)
    required = set(plan.required_claim_ids)
    forbidden = set(plan.forbidden_claim_ids)
    aggregate: dict[str, tuple[bool, bool]] = {}
    for evidence in question.evidence:
        support, refute = aggregate.get(
            evidence.proposition_id, (False, False))
        aggregate[evidence.proposition_id] = (
            support or bool(evidence.support),
            refute or bool(evidence.refute),
        )
    supported = {
        claim_id for claim_id, (support, refute) in aggregate.items()
        if support and not refute
    }
    cited = set(realization.cited_source_ids)
    required_sources = set(plan.citation_source_ids)
    evidence_sources = {item.source_id for item in question.evidence}
    violations = []
    if realization.response_act != plan.response_act:
        violations.append("RESPONSE_ACT_DRIFT")
    if realization.scope_id != question.response_scope_id:
        violations.append("SCOPE_DRIFT")
    if not required <= claim_ids:
        violations.append("MISSING_REQUIRED_CLAIM")
    if claim_ids & forbidden:
        violations.append("FORBIDDEN_CLAIM")
    if not claim_ids <= supported:
        violations.append("UNSUPPORTED_CLAIM")
    if plan.response_act != "ANSWER" and claim_ids:
        violations.append("NONANSWER_CLAIM")
    if not required_sources <= cited:
        violations.append("MISSING_CITATION")
    if not cited <= evidence_sources:
        violations.append("FOREIGN_CITATION")
    if not required_sources and cited:
        violations.append("UNPLANNED_CITATION")
    ordered = tuple(
        item for item in VERIFICATION_VIOLATIONS if item in violations)
    return GenerationVerification(
        "PASS" if not ordered else "FAIL",
        ordered,
        int(realization.response_act == plan.response_act),
        int(realization.scope_id == question.response_scope_id),
        int(required <= claim_ids),
        int(not claim_ids & forbidden),
        int(claim_ids <= supported),
        int(required_sources <= cited and cited <= evidence_sources
            and (bool(required_sources) or not cited)),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedReferenceSurfaceLabel:
    """一个 accepted surface 上的 teacher-only reference 选择标签。"""

    realization_id: str
    strategy: str
    sentence_ordinal: int
    slot_id: str
    reference_surface: str
    span_start: int
    span_end: int

    def __post_init__(self) -> None:
        _text(self.realization_id, where="reference_label.realization_id")
        if self.strategy not in REFERENCE_STRATEGIES:
            raise GroundedAnswerCourseError(
                "reference label strategy 未注册")
        _positive(
            self.sentence_ordinal,
            where="reference_label.sentence_ordinal")
        _text(self.slot_id, where="reference_label.slot_id")
        _text(
            self.reference_surface,
            where="reference_label.reference_surface")
        if (type(self.span_start) is not int
                or type(self.span_end) is not int
                or self.span_start < 0
                or self.span_end <= self.span_start):
            raise GroundedAnswerCourseError("reference label span 非法")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "realization_id": self.realization_id,
            "reference_surface": self.reference_surface,
            "sentence_ordinal": self.sentence_ordinal,
            "slot_id": self.slot_id,
            "span_end": self.span_end,
            "span_start": self.span_start,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedReferenceSurfaceLabel":
        """从精确 teacher label object 恢复记录。"""
        raw = _exact(value, frozenset({
            "realization_id", "reference_surface", "sentence_ordinal",
            "slot_id", "span_end", "span_start", "strategy",
        }), where="reference_surface_label")
        return cls(
            _text(raw["realization_id"],
                  where="reference_label.realization_id"),
            _text(raw["strategy"], where="reference_label.strategy"),
            raw["sentence_ordinal"],
            _text(raw["slot_id"], where="reference_label.slot_id"),
            _text(raw["reference_surface"],
                  where="reference_label.reference_surface"),
            raw["span_start"],
            raw["span_end"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedReferenceChallenge:
    """一个必须 fail closed 的 reference 负挑战类别。"""

    challenge_id: str
    kind: str

    def __post_init__(self) -> None:
        _text(self.challenge_id, where="reference_challenge.challenge_id")
        if self.kind not in REFERENCE_CHALLENGE_KINDS:
            raise GroundedAnswerCourseError(
                "reference challenge kind 未注册")

    def to_dict(self) -> dict[str, str]:
        """导出规范 JSON 值。"""
        return {
            "challenge_id": self.challenge_id,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedReferenceChallenge":
        """从精确 challenge object 恢复记录。"""
        raw = _exact(value, frozenset({"challenge_id", "kind"}),
                     where="reference_challenge")
        return cls(
            _text(raw["challenge_id"],
                  where="reference_challenge.challenge_id"),
            _text(raw["kind"], where="reference_challenge.kind"),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedReferenceCourse:
    """冻结 Proposition/event 级 antecedent、referring origin 与标签。"""

    granularity: str
    antecedent_proposition_id: str
    referring_proposition_id: str
    antecedent_evidence_id: str
    referring_evidence_id: str
    source_id: str
    scope_id: int
    ordered_proposition_ids: tuple[str, ...]
    surface_labels: tuple[GroundedReferenceSurfaceLabel, ...]
    challenges: tuple[GroundedReferenceChallenge, ...]

    def __post_init__(self) -> None:
        if self.granularity != REFERENCE_GRANULARITY:
            raise GroundedAnswerCourseError(
                "reference granularity 只允许 Proposition/event")
        for name, value in (
                ("antecedent_proposition_id",
                 self.antecedent_proposition_id),
                ("referring_proposition_id", self.referring_proposition_id),
                ("antecedent_evidence_id", self.antecedent_evidence_id),
                ("referring_evidence_id", self.referring_evidence_id),
                ("source_id", self.source_id)):
            _text(value, where=f"reference_course.{name}")
        _positive(self.scope_id, where="reference_course.scope_id")
        if (not isinstance(self.ordered_proposition_ids, tuple)
                or len(self.ordered_proposition_ids) < 2
                or any(not isinstance(item, str) or not item
                       for item in self.ordered_proposition_ids)
                or len(set(self.ordered_proposition_ids))
                != len(self.ordered_proposition_ids)):
            raise GroundedAnswerCourseError(
                "reference ordered propositions 非法或不足两个")
        if (not isinstance(self.surface_labels, tuple)
                or any(not isinstance(item, GroundedReferenceSurfaceLabel)
                       for item in self.surface_labels)):
            raise GroundedAnswerCourseError("reference surface labels 类型错误")
        if (tuple(item.strategy for item in self.surface_labels)
                != REFERENCE_STRATEGIES):
            raise GroundedAnswerCourseError(
                "reference strategies 必须精确覆盖冻结顺序")
        if (not isinstance(self.challenges, tuple)
                or any(not isinstance(item, GroundedReferenceChallenge)
                       for item in self.challenges)
                or tuple(item.kind for item in self.challenges)
                != REFERENCE_CHALLENGE_KINDS):
            raise GroundedAnswerCourseError(
                "reference challenges 必须精确覆盖六类冻结顺序")
        challenge_ids = tuple(item.challenge_id for item in self.challenges)
        if len(set(challenge_ids)) != len(challenge_ids):
            raise GroundedAnswerCourseError("reference challenge id 重复")

    def validate_episode(self, episode: "GroundedAnswerEpisode") -> None:
        """把 teacher reference 标签绑定到同一 episode 的真实资料。"""
        plan = episode.question.answer_plan
        if plan.response_act != "ANSWER":
            raise GroundedAnswerCourseError(
                "reference course 只允许 ANSWER episode")
        if self.ordered_proposition_ids != plan.ordered_claim_ids:
            raise GroundedAnswerCourseError(
                "reference proposition 顺序与 answer plan 漂移")
        try:
            antecedent_ordinal = self.ordered_proposition_ids.index(
                self.antecedent_proposition_id)
            referring_ordinal = self.ordered_proposition_ids.index(
                self.referring_proposition_id)
        except ValueError as error:
            raise GroundedAnswerCourseError(
                "reference proposition 不属于 answer plan") from error
        if antecedent_ordinal >= referring_ordinal:
            raise GroundedAnswerCourseError(
                "reference antecedent 必须早于 referring proposition")
        evidence_by_id = {
            item.evidence_id: item for item in episode.question.evidence
        }
        antecedent = evidence_by_id.get(self.antecedent_evidence_id)
        referring = evidence_by_id.get(self.referring_evidence_id)
        if (antecedent is None or referring is None
                or antecedent.proposition_id
                != self.antecedent_proposition_id
                or referring.proposition_id
                != self.referring_proposition_id):
            raise GroundedAnswerCourseError(
                "reference forming Evidence 与 Proposition 不一致")
        if any(
                item.source_id != self.source_id
                or item.scope_id != self.scope_id
                or not item.support
                or item.refute
                for item in (antecedent, referring)):
            raise GroundedAnswerCourseError(
                "reference forming Evidence 的 source/scope/direction 漂移")
        if (self.scope_id != episode.question.evidence_scope_id
                or self.source_id not in plan.citation_source_ids):
            raise GroundedAnswerCourseError(
                "reference course source/scope 不属于当前问题")
        accepted = {
            item.realization_id: item for item in episode.surfaces.accepted
        }
        if set(accepted) != {
                item.realization_id for item in self.surface_labels}:
            raise GroundedAnswerCourseError(
                "reference labels 必须精确覆盖 accepted realization")
        expected_sentence = referring_ordinal + 1
        slot_ids = {item.slot_id for item in self.surface_labels}
        if len(slot_ids) != 1:
            raise GroundedAnswerCourseError(
                "reference strategies 必须竞争同一 sentence slot")
        for label in self.surface_labels:
            realization = accepted[label.realization_id]
            if (realization.claim_ids != self.ordered_proposition_ids
                    or realization.scope_id
                    != episode.question.response_scope_id
                    or self.source_id not in realization.cited_source_ids):
                raise GroundedAnswerCourseError(
                    "reference accepted realization 的 claim/source/scope 漂移")
            if label.sentence_ordinal != expected_sentence:
                raise GroundedAnswerCourseError(
                    "reference label sentence order 漂移")
            if (label.span_end > len(realization.surface)
                    or realization.surface[
                        label.span_start:label.span_end]
                    != label.reference_surface):
                raise GroundedAnswerCourseError(
                    "reference label span 与 accepted surface 不一致")
            first_sentence_end = realization.surface.find("。")
            if (first_sentence_end < 0
                    or label.span_start != first_sentence_end + 1):
                raise GroundedAnswerCourseError(
                    "reference label 必须从 referring sentence 起始")

    def to_dict(self) -> dict[str, object]:
        """导出 teacher-only reference 课程。"""
        return {
            "antecedent_evidence_id": self.antecedent_evidence_id,
            "antecedent_proposition_id": self.antecedent_proposition_id,
            "challenges": [item.to_dict() for item in self.challenges],
            "granularity": self.granularity,
            "ordered_proposition_ids": list(self.ordered_proposition_ids),
            "referring_evidence_id": self.referring_evidence_id,
            "referring_proposition_id": self.referring_proposition_id,
            "scope_id": self.scope_id,
            "source_id": self.source_id,
            "surface_labels": [
                item.to_dict() for item in self.surface_labels],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedReferenceCourse":
        """从精确 teacher reference object 恢复课程。"""
        raw = _exact(value, frozenset({
            "antecedent_evidence_id", "antecedent_proposition_id",
            "challenges", "granularity", "ordered_proposition_ids",
            "referring_evidence_id", "referring_proposition_id", "scope_id",
            "source_id", "surface_labels",
        }), where="reference_course")
        if (not isinstance(raw["surface_labels"], list)
                or not isinstance(raw["challenges"], list)):
            raise GroundedAnswerCourseError(
                "reference surface_labels/challenges 必须是 list")
        return cls(
            _text(raw["granularity"],
                  where="reference_course.granularity"),
            _text(raw["antecedent_proposition_id"],
                  where="reference_course.antecedent_proposition_id"),
            _text(raw["referring_proposition_id"],
                  where="reference_course.referring_proposition_id"),
            _text(raw["antecedent_evidence_id"],
                  where="reference_course.antecedent_evidence_id"),
            _text(raw["referring_evidence_id"],
                  where="reference_course.referring_evidence_id"),
            _text(raw["source_id"], where="reference_course.source_id"),
            raw["scope_id"],
            _strings(raw["ordered_proposition_ids"],
                     where="reference_course.ordered_proposition_ids"),
            tuple(GroundedReferenceSurfaceLabel.from_dict(item)
                  for item in raw["surface_labels"]),
            tuple(GroundedReferenceChallenge.from_dict(item)
                  for item in raw["challenges"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerEpisode:
    """把问题、会话、split 与多表面资料绑定成一条训练记录。"""

    episode_id: str
    split: str
    clusters: GroundedAnswerSplitClusters
    question: GroundedQuestionEpisode
    dialogue: DialogueEpisode
    surfaces: SurfaceRealizationSet
    reference_course: "GroundedReferenceCourse | None" = None

    def __post_init__(self) -> None:
        _text(self.episode_id, where="episode_id")
        if self.split not in SPLITS:
            raise GroundedAnswerCourseError("episode split 未注册")
        if not isinstance(self.clusters, GroundedAnswerSplitClusters):
            raise GroundedAnswerCourseError("episode clusters 类型错误")
        if not isinstance(self.question, GroundedQuestionEpisode):
            raise GroundedAnswerCourseError("episode question 类型错误")
        if not isinstance(self.dialogue, DialogueEpisode):
            raise GroundedAnswerCourseError("episode dialogue 类型错误")
        if not isinstance(self.surfaces, SurfaceRealizationSet):
            raise GroundedAnswerCourseError("episode surfaces 类型错误")
        if (self.reference_course is not None
                and not isinstance(
                    self.reference_course, GroundedReferenceCourse)):
            raise GroundedAnswerCourseError(
                "episode reference_course 类型错误")
        if self.dialogue.turns[-1].surface != self.question.question_surface:
            raise GroundedAnswerCourseError(
                "dialogue 当前 USER turn 必须等于 question surface")
        if not {
                self.question.evidence_scope_id,
                self.question.response_scope_id,
                } <= set(self.dialogue.active_scope_ids):
            raise GroundedAnswerCourseError(
                "Evidence/response scope 必须位于 dialogue active scopes")
        if self.reference_course is not None:
            self.reference_course.validate_episode(self)

    def to_dict(self) -> dict[str, object]:
        """导出规范、可公开的 episode JSON 值。"""
        result = {
            "artifact_kind": ARTIFACT_KIND,
            "clusters": self.clusters.to_dict(),
            "dialogue": self.dialogue.to_dict(),
            "episode_id": self.episode_id,
            "license_id": LICENSE_ID,
            "question": self.question.to_dict(),
            "schema_version": 2 if self.reference_course is not None else 1,
            "split": self.split,
            "surfaces": self.surfaces.to_dict(),
        }
        if self.reference_course is not None:
            result["reference_course"] = self.reference_course.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "GroundedAnswerEpisode":
        """从严格顶层 object 恢复 episode。"""
        if not isinstance(value, dict):
            raise GroundedAnswerCourseError(
                "grounded answer episode 字段集合漂移")
        schema = value.get("schema_version")
        keys = {
            "artifact_kind", "clusters", "dialogue", "episode_id",
            "license_id", "question", "schema_version", "split", "surfaces",
        }
        if schema == 2:
            keys.add("reference_course")
        raw = _exact(
            value, frozenset(keys), where="grounded answer episode")
        if (raw["artifact_kind"] != ARTIFACT_KIND
                or raw["license_id"] != LICENSE_ID
                or type(raw["schema_version"]) is not int
                or raw["schema_version"] not in {1, 2}):
            raise GroundedAnswerCourseError(
                "episode kind/license/schema 漂移")
        return cls(
            _text(raw["episode_id"], where="episode_id"),
            _text(raw["split"], where="split"),
            GroundedAnswerSplitClusters.from_dict(raw["clusters"]),
            GroundedQuestionEpisode.from_dict(raw["question"]),
            DialogueEpisode.from_dict(raw["dialogue"]),
            SurfaceRealizationSet.from_dict(raw["surfaces"]),
            (None if raw["schema_version"] == 1
             else GroundedReferenceCourse.from_dict(raw["reference_course"])),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerCourseAudit:
    """返回公开切片的记录、正负 surface 和 split cluster 计数。"""

    episode_count: int
    accepted_surface_count: int
    rejected_surface_count: int
    source_cluster_count: int
    proposition_cluster_count: int
    question_construction_cluster_count: int
    paraphrase_cluster_count: int


def audit_grounded_answer_course(
        episodes: tuple[GroundedAnswerEpisode, ...],
        *, train_only: bool = True,
        ) -> GroundedAnswerCourseAudit:
    """核对 split 防泄漏、多表面正例和全部预注册负例。"""
    if (not isinstance(episodes, tuple) or not episodes
            or any(not isinstance(item, GroundedAnswerEpisode)
                   for item in episodes)):
        raise GroundedAnswerCourseError("episodes 必须是非空 tuple")
    ids = tuple(item.episode_id for item in episodes)
    if len(set(ids)) != len(ids):
        raise GroundedAnswerCourseError("episode_id 重复")
    if train_only and any(item.split != "train" for item in episodes):
        raise GroundedAnswerCourseError("TRAIN-only 切片包含非 train record")
    axis_maps: dict[str, dict[str, str]] = {
        name: {} for name in (
            "source", "proposition", "question_construction", "paraphrase")
    }
    accepted_count = 0
    rejected_count = 0
    for episode in episodes:
        for name in axis_maps:
            key = getattr(episode.clusters, name)
            prior = axis_maps[name].get(key)
            if prior is not None and prior != episode.split:
                raise GroundedAnswerCourseError(
                    f"{name} cluster 跨 split 泄漏")
            axis_maps[name][key] = episode.split
        accepted_count += len(episode.surfaces.accepted)
        rejected_count += len(episode.surfaces.rejected)
        for realization in episode.surfaces.accepted:
            result = verify_surface_realization(episode.question, realization)
            if not result.passed:
                raise GroundedAnswerCourseError(
                    f"accepted surface {realization.realization_id} 未通过")
        for rejected in episode.surfaces.rejected:
            result = verify_surface_realization(
                episode.question, rejected.realization)
            if result.passed or result.violations != rejected.expected_violations:
                raise GroundedAnswerCourseError(
                    f"rejected surface {rejected.realization.realization_id} "
                    "与预注册失败维度不一致")
    return GroundedAnswerCourseAudit(
        len(episodes), accepted_count, rejected_count,
        len(axis_maps["source"]), len(axis_maps["proposition"]),
        len(axis_maps["question_construction"]),
        len(axis_maps["paraphrase"]),
    )


def read_grounded_answer_episodes(
        path: str | Path, *, train_only: bool = True,
        ) -> tuple[GroundedAnswerEpisode, ...]:
    """严格回读 canonical JSONL，并立即执行课程与 split 审计。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise GroundedAnswerCourseError("grounded answer sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise GroundedAnswerCourseError(
            "grounded answer sample 必须非空并以换行结束")
    episodes = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            raise GroundedAnswerCourseError(
                f"grounded answer 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise GroundedAnswerCourseError(
                f"grounded answer 第 {line_number} 行不是规范 JSON") from error
        if canonical_json_line(value) != line:
            raise GroundedAnswerCourseError(
                f"grounded answer 第 {line_number} 行不是规范字节")
        episodes.append(GroundedAnswerEpisode.from_dict(value))
    result = tuple(episodes)
    audit_grounded_answer_course(result, train_only=train_only)
    return result


__all__ = [
    "ARTIFACT_KIND",
    "CARRIER_KINDS",
    "DialogueEpisode",
    "DialogueTurn",
    "GenerationVerification",
    "GroundedAnswerCourseAudit",
    "GroundedAnswerCourseError",
    "GroundedAnswerEpisode",
    "GroundedAnswerPlan",
    "GroundedAnswerSplitClusters",
    "GroundedEvidence",
    "GroundedQuestionEpisode",
    "GroundedReferenceChallenge",
    "GroundedReferenceCourse",
    "GroundedReferenceSurfaceLabel",
    "LICENSE_ID",
    "REFERENCE_CHALLENGE_KINDS",
    "REFERENCE_GRANULARITY",
    "REFERENCE_STRATEGIES",
    "RESPONSE_ACTS",
    "RejectedSurfaceRealization",
    "SurfaceRealization",
    "SurfaceRealizationSet",
    "VERIFICATION_VIOLATIONS",
    "audit_grounded_answer_course",
    "read_grounded_answer_episodes",
    "verify_surface_realization",
]
