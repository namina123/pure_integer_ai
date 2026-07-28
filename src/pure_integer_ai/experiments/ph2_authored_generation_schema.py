"""D-02E.3 generation adoption、source requirement 和 postcheck 纯合同。"""
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


GENERATION_CASES = frozenset({
    "ADOPTION_ANSWER",
    "ADOPTION_UNKNOWN",
    "ADOPTION_AMBIGUOUS",
    "ADOPTION_REFUSE",
    "ADOPTION_CONFLICT",
    "POSTCHECK_PASS",
    "POSTCHECK_CITATION_FAIL",
    "POSTCHECK_TRUST_FAIL",
    "POSTCHECK_SOURCE_FAIL",
})
GENERATION_STANCES = frozenset({
    "ANSWER", "UNKNOWN", "CLARIFY", "REFUSE", "CONFLICT"})

_CANDIDATE_FIELDS = frozenset({
    "candidate_id",
    "end",
    "evidence_refute",
    "evidence_source_ids",
    "evidence_support",
    "ordinal",
    "predicate_kind",
    "proposition_local_id",
    "start",
    "surface_fragment",
})
_REQUIREMENT_FIELDS = frozenset({
    "candidate_id",
    "citation_required",
    "cited_source_ids",
    "refuted_source_ids",
    "source_match",
    "trust_required",
    "trusted_source_ids",
})
_REQUEST_FIELDS = frozenset({
    "max_candidates",
    "max_evidence_sources",
    "max_postcheck_checks",
    "max_surface_units",
})
_SEED_FIELDS = frozenset({
    "candidates",
    "consumer_request",
    "context_surface",
    "expected_payload",
    "expected_state",
    "family",
    "generation_case",
    "label_owner",
    "license_id",
    "logical_order",
    "perturbation_kind",
    "postcheck_enabled",
    "renderer_complete",
    "response_scope_local_id",
    "sample_role",
    "same_run_local_id",
    "seed_id",
    "selected_candidate_ids",
    "source_key",
    "source_requirements",
    "split",
    "stance",
    "supersedes_seed_id",
    "surface_units",
    "template_family",
})


class AuthoredGenerationCourseError(RuntimeError):
    """原创 generation seed 的采用、source/postcheck、owner 或预算非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本无首尾空白；按字段允许空字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredGenerationCourseError(
            f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredGenerationCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求身份、scope 和预算为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredGenerationCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 span、ordinal 和 surface unit 为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredGenerationCourseError(
            f"{where} 必须是非负严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """要求 postcheck 布尔位为严格整数 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredGenerationCourseError(
            f"{where} 必须是严格整数 0/1")
    return value


def _text_tuple(value: Any, *, where: str, allow_empty: bool = False):
    """恢复无重复非空字符串列表。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise AuthoredGenerationCourseError(f"{where} 非法或重复")
    return tuple(value)


def _int_tuple(value: Any, *, where: str, allow_empty: bool = False):
    """恢复无重复正严格整数列表。"""
    if (not isinstance(value, list)
            or (not allow_empty and not value)
            or any(type(item) is not int or item <= 0 for item in value)
            or len(set(value)) != len(value)):
        raise AuthoredGenerationCourseError(f"{where} 非法或重复")
    return tuple(value)


@dataclass(frozen=True)
class GenerationCandidateSeed:
    """一个 G-00 typed Proposition 候选及实际 Evidence 来源集合。"""

    candidate_id: str
    proposition_local_id: int
    predicate_kind: int
    surface_fragment: str
    start: int
    end: int
    ordinal: int
    evidence_support: int
    evidence_refute: int
    evidence_source_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.candidate_id, where="GenerationCandidateSeed.candidate_id")
        for name, value in (
                ("proposition_local_id", self.proposition_local_id),
                ("predicate_kind", self.predicate_kind)):
            _positive_int(value, where=f"GenerationCandidateSeed.{name}")
        _text(
            self.surface_fragment,
            where="GenerationCandidateSeed.surface_fragment",
        )
        _nonnegative_int(self.start, where="GenerationCandidateSeed.start")
        _nonnegative_int(self.end, where="GenerationCandidateSeed.end")
        _nonnegative_int(self.ordinal, where="GenerationCandidateSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredGenerationCourseError(
                "generation candidate 必须有正宽度")
        _bit(
            self.evidence_support,
            where="GenerationCandidateSeed.evidence_support",
        )
        _bit(
            self.evidence_refute,
            where="GenerationCandidateSeed.evidence_refute",
        )
        if (not isinstance(self.evidence_source_ids, tuple)
                or not self.evidence_source_ids
                or any(type(item) is not int or item <= 0
                       for item in self.evidence_source_ids)
                or len(set(self.evidence_source_ids))
                != len(self.evidence_source_ids)):
            raise AuthoredGenerationCourseError(
                "generation evidence source 非法或重复")

    @classmethod
    def from_dict(cls, value: Any) -> "GenerationCandidateSeed":
        """从严格字段集合恢复 generation candidate。"""
        if not isinstance(value, dict) or set(value) != _CANDIDATE_FIELDS:
            raise AuthoredGenerationCourseError(
                "generation candidate 字段集合漂移")
        return cls(
            _text(value["candidate_id"], where="candidate_id"),
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
                value["evidence_source_ids"],
                where="candidate.evidence_source_ids",
            ),
        )


@dataclass(frozen=True)
class GenerationSourceRequirementSeed:
    """G-04 对一个已采用候选的 citation/trust/source 观察。"""

    candidate_id: str
    citation_required: int
    trust_required: int
    source_match: int
    cited_source_ids: tuple[int, ...]
    trusted_source_ids: tuple[int, ...]
    refuted_source_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(
            self.candidate_id,
            where="GenerationSourceRequirementSeed.candidate_id",
        )
        for name, value in (
                ("citation_required", self.citation_required),
                ("trust_required", self.trust_required),
                ("source_match", self.source_match)):
            _bit(value, where=f"GenerationSourceRequirementSeed.{name}")
        if not self.citation_required and not self.trust_required:
            raise AuthoredGenerationCourseError(
                "source requirement 至少要求 citation 或 trust")
        for name, value in (
                ("cited", self.cited_source_ids),
                ("trusted", self.trusted_source_ids),
                ("refuted", self.refuted_source_ids)):
            if (not isinstance(value, tuple)
                    or any(type(item) is not int or item <= 0
                           for item in value)
                    or len(set(value)) != len(value)):
                raise AuthoredGenerationCourseError(
                    f"source requirement {name} 非法或重复")
        if set(self.trusted_source_ids) & set(self.refuted_source_ids):
            raise AuthoredGenerationCourseError(
                "同一 evidence source 不得同时 trusted/refuted")

    @classmethod
    def from_dict(cls, value: Any) -> "GenerationSourceRequirementSeed":
        """从严格字段集合恢复 source requirement。"""
        if not isinstance(value, dict) or set(value) != _REQUIREMENT_FIELDS:
            raise AuthoredGenerationCourseError(
                "source requirement 字段集合漂移")
        return cls(
            _text(value["candidate_id"], where="requirement.candidate_id"),
            value["citation_required"],
            value["trust_required"],
            value["source_match"],
            _int_tuple(
                value["cited_source_ids"],
                where="requirement.cited_source_ids",
                allow_empty=True,
            ),
            _int_tuple(
                value["trusted_source_ids"],
                where="requirement.trusted_source_ids",
                allow_empty=True,
            ),
            _int_tuple(
                value["refuted_source_ids"],
                where="requirement.refuted_source_ids",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True)
class GenerationConsumerRequestSeed:
    """generation candidate、Evidence source、surface 和 postcheck 预算。"""

    max_candidates: int
    max_evidence_sources: int
    max_surface_units: int
    max_postcheck_checks: int

    def __post_init__(self) -> None:
        for name, value in (
                ("max_candidates", self.max_candidates),
                ("max_evidence_sources", self.max_evidence_sources),
                ("max_surface_units", self.max_surface_units),
                ("max_postcheck_checks", self.max_postcheck_checks)):
            _positive_int(value, where=f"GenerationConsumerRequestSeed.{name}")

    @classmethod
    def from_dict(cls, value: Any) -> "GenerationConsumerRequestSeed":
        """从严格字段集合恢复 generation consumer。"""
        if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
            raise AuthoredGenerationCourseError(
                "generation consumer 字段集合漂移")
        return cls(
            value["max_candidates"],
            value["max_evidence_sources"],
            value["max_surface_units"],
            value["max_postcheck_checks"],
        )


@dataclass(frozen=True)
class AuthoredGenerationSeed:
    """一条 G-00 至 G-04 adoption/source/postcheck 课程记录。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    source_key: str
    context_surface: str
    generation_case: str
    stance: str
    candidates: tuple[GenerationCandidateSeed, ...]
    selected_candidate_ids: tuple[str, ...]
    response_scope_local_id: int
    same_run_local_id: int
    renderer_complete: int
    surface_units: int
    postcheck_enabled: int
    source_requirements: tuple[GenerationSourceRequirementSeed, ...]
    consumer_request: GenerationConsumerRequestSeed
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
                ("generation_case", self.generation_case),
                ("stance", self.stance),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredGenerationSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredGenerationSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredGenerationCourseError(
                "label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredGenerationCourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredGenerationCourseError(
                "sample_role 不属于 generation 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredGenerationCourseError(
                "generation supersede 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredGenerationCourseError(
                "非 supersede generation 不得声明替代目标")
        if self.source_key != SOURCE_KEY:
            raise AuthoredGenerationCourseError("generation source key 漂移")
        if self.generation_case not in GENERATION_CASES:
            raise AuthoredGenerationCourseError("generation case 未注册")
        if self.stance not in GENERATION_STANCES:
            raise AuthoredGenerationCourseError("generation stance 未注册")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, GenerationCandidateSeed)
                       for item in self.candidates)):
            raise AuthoredGenerationCourseError(
                "generation candidates 类型错误")
        if (not isinstance(self.selected_candidate_ids, tuple)
                or len(set(self.selected_candidate_ids))
                != len(self.selected_candidate_ids)):
            raise AuthoredGenerationCourseError(
                "selected candidate 非法或重复")
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise AuthoredGenerationCourseError("generation candidate_id 重复")
        if not set(self.selected_candidate_ids) <= set(candidate_ids):
            raise AuthoredGenerationCourseError(
                "selected candidate 不属于请求")
        ordinals = [item.ordinal for item in self.candidates]
        if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
            raise AuthoredGenerationCourseError(
                "generation candidate ordinal 必须严格递增")
        for item in self.candidates:
            if item.end > len(self.context_surface) or self.context_surface[
                    item.start:item.end] != item.surface_fragment:
                raise AuthoredGenerationCourseError(
                    "generation candidate 与 context 不一致")
        if self.stance == "ANSWER":
            if len(self.selected_candidate_ids) != 1:
                raise AuthoredGenerationCourseError(
                    "ANSWER 必须采用唯一候选")
        elif self.selected_candidate_ids:
            raise AuthoredGenerationCourseError(
                "非 ANSWER stance 不得采用内容候选")
        _positive_int(
            self.response_scope_local_id,
            where="AuthoredGenerationSeed.response_scope_local_id",
        )
        _positive_int(
            self.same_run_local_id,
            where="AuthoredGenerationSeed.same_run_local_id",
        )
        _bit(
            self.renderer_complete,
            where="AuthoredGenerationSeed.renderer_complete",
        )
        _nonnegative_int(
            self.surface_units,
            where="AuthoredGenerationSeed.surface_units",
        )
        if bool(self.surface_units) != bool(self.renderer_complete):
            raise AuthoredGenerationCourseError(
                "surface units 与 renderer complete 不一致")
        _bit(
            self.postcheck_enabled,
            where="AuthoredGenerationSeed.postcheck_enabled",
        )
        if (not isinstance(self.source_requirements, tuple)
                or any(not isinstance(item, GenerationSourceRequirementSeed)
                       for item in self.source_requirements)):
            raise AuthoredGenerationCourseError(
                "source requirements 类型错误")
        requirement_ids = [item.candidate_id for item in self.source_requirements]
        if len(set(requirement_ids)) != len(requirement_ids):
            raise AuthoredGenerationCourseError(
                "source requirement candidate 重复")
        if self.postcheck_enabled:
            if (not self.renderer_complete
                    or set(requirement_ids) != set(self.selected_candidate_ids)):
                raise AuthoredGenerationCourseError(
                    "postcheck 必须精确覆盖已采用且已渲染候选")
        elif self.source_requirements:
            raise AuthoredGenerationCourseError(
                "未启用 postcheck 不得伪造 requirement")
        candidate_by_id = {item.candidate_id: item for item in self.candidates}
        for item in self.source_requirements:
            candidate = candidate_by_id.get(item.candidate_id)
            if candidate is None:
                raise AuthoredGenerationCourseError(
                    "source requirement 引用未知 candidate")
            evidence = set(candidate.evidence_source_ids)
            observed = {
                *item.cited_source_ids,
                *item.trusted_source_ids,
                *item.refuted_source_ids,
            }
            if not observed <= evidence:
                raise AuthoredGenerationCourseError(
                    "source requirement 观察越过 Evidence 来源")
        if len(self.candidates) > self.consumer_request.max_candidates:
            raise AuthoredGenerationCourseError(
                "generation candidates 超过预算")
        evidence_count = sum(
            len(item.evidence_source_ids) for item in self.candidates)
        if evidence_count > self.consumer_request.max_evidence_sources:
            raise AuthoredGenerationCourseError(
                "generation Evidence sources 超过预算")
        if self.surface_units > self.consumer_request.max_surface_units:
            raise AuthoredGenerationCourseError("surface units 超过预算")
        if len(self.source_requirements) > (
                self.consumer_request.max_postcheck_checks):
            raise AuthoredGenerationCourseError("postcheck checks 超过预算")
        expected_stance = {
            "ADOPTION_ANSWER": "ANSWER",
            "ADOPTION_UNKNOWN": "UNKNOWN",
            "ADOPTION_AMBIGUOUS": "CLARIFY",
            "ADOPTION_REFUSE": "REFUSE",
            "ADOPTION_CONFLICT": "CONFLICT",
        }.get(self.generation_case)
        if expected_stance is not None and self.stance != expected_stance:
            raise AuthoredGenerationCourseError(
                "adoption case 与 stance 不一致")
        if self.generation_case == "ADOPTION_ANSWER":
            selected = candidate_by_id[self.selected_candidate_ids[0]]
            if (not (selected.evidence_support or selected.evidence_refute)
                    or (selected.evidence_support
                        and selected.evidence_refute)):
                raise AuthoredGenerationCourseError(
                    "ADOPTION_ANSWER 候选必须可决且无冲突")
        if self.generation_case == "ADOPTION_UNKNOWN" and any(
                item.evidence_support or item.evidence_refute
                for item in self.candidates):
            raise AuthoredGenerationCourseError(
                "ADOPTION_UNKNOWN 不得携带可决候选")
        if self.generation_case == "ADOPTION_AMBIGUOUS":
            eligible = [
                item for item in self.candidates
                if (item.evidence_support or item.evidence_refute)
                and not (item.evidence_support and item.evidence_refute)
            ]
            if len(eligible) < 2:
                raise AuthoredGenerationCourseError(
                    "ADOPTION_AMBIGUOUS 必须有多个可决候选")
        if self.generation_case == "ADOPTION_REFUSE" and self.candidates:
            raise AuthoredGenerationCourseError(
                "ADOPTION_REFUSE 不得伪造可用候选")
        if self.generation_case == "ADOPTION_CONFLICT" and not any(
                item.evidence_support and item.evidence_refute
                for item in self.candidates):
            raise AuthoredGenerationCourseError(
                "ADOPTION_CONFLICT 必须携带冲突候选")
        if self.generation_case.startswith("POSTCHECK_"):
            if self.stance != "ANSWER" or not self.postcheck_enabled:
                raise AuthoredGenerationCourseError(
                    "postcheck case 必须是已采用 ANSWER")
            checks = []
            for item in self.source_requirements:
                evidence = set(
                    candidate_by_id[item.candidate_id].evidence_source_ids)
                citation_ok = (
                    not item.citation_required
                    or evidence <= set(item.cited_source_ids))
                trust_ok = (
                    not item.trust_required
                    or (evidence <= set(item.trusted_source_ids)
                        and not evidence & set(item.refuted_source_ids)))
                checks.append((bool(item.source_match), citation_ok, trust_ok))
            if self.generation_case == "POSTCHECK_PASS" and not all(
                    all(values) for values in checks):
                raise AuthoredGenerationCourseError(
                    "POSTCHECK_PASS 必须满足全部 source/citation/trust")
            if (self.generation_case == "POSTCHECK_CITATION_FAIL"
                    and not any(not citation for _, citation, _ in checks)):
                raise AuthoredGenerationCourseError(
                    "citation fail case 必须缺实际 Evidence citation")
            if (self.generation_case == "POSTCHECK_TRUST_FAIL"
                    and not any(not trust for _, _, trust in checks)):
                raise AuthoredGenerationCourseError(
                    "trust fail case 必须有未通过 trust 的 Evidence source")
            if (self.generation_case == "POSTCHECK_SOURCE_FAIL"
                    and not any(not source_ok
                                for source_ok, _, _ in checks)):
                raise AuthoredGenerationCourseError(
                    "source fail case 必须有 source mismatch")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredGenerationCourseError(
                "generation expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredGenerationCourseError(
                "generation perturbation 未注册")
        _positive_int(
            self.logical_order,
            where="AuthoredGenerationSeed.logical_order",
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredGenerationSeed":
        """从严格字段集合恢复 generation seed。"""
        if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
            raise AuthoredGenerationCourseError(
                "generation seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredGenerationCourseError(
                "generation seed 必须是 CC0-1.0")
        raw_candidates = value["candidates"]
        raw_requirements = value["source_requirements"]
        if not isinstance(raw_candidates, list) or not isinstance(
                raw_requirements, list):
            raise AuthoredGenerationCourseError(
                "generation candidates/requirements 必须是列表")
        return cls(
            _text(value["seed_id"], where="seed_id"),
            _text(value["family"], where="family"),
            _text(value["template_family"], where="template_family"),
            _text(value["label_owner"], where="label_owner"),
            _text(value["split"], where="split"),
            _text(value["sample_role"], where="sample_role"),
            _text(value["source_key"], where="source_key"),
            _text(value["context_surface"], where="context_surface"),
            _text(value["generation_case"], where="generation_case"),
            _text(value["stance"], where="stance"),
            tuple(GenerationCandidateSeed.from_dict(item)
                  for item in raw_candidates),
            _text_tuple(
                value["selected_candidate_ids"],
                where="selected_candidate_ids",
                allow_empty=True,
            ),
            value["response_scope_local_id"],
            value["same_run_local_id"],
            value["renderer_complete"],
            value["surface_units"],
            value["postcheck_enabled"],
            tuple(GenerationSourceRequirementSeed.from_dict(item)
                  for item in raw_requirements),
            GenerationConsumerRequestSeed.from_dict(value["consumer_request"]),
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
    "AuthoredGenerationCourseError",
    "AuthoredGenerationSeed",
    "GENERATION_CASES",
    "GENERATION_STANCES",
    "GenerationCandidateSeed",
    "GenerationConsumerRequestSeed",
    "GenerationSourceRequirementSeed",
    "LICENSE_ID",
    "REQUIRED_SAMPLE_ROLES",
    "SOURCE_KEY",
]
