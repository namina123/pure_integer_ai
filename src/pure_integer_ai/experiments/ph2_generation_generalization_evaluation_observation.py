"""E-05 不携 accepted/rejected surface 的 executable evaluation Observation。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    LICENSE_ID,
    REFERENCE_STRATEGIES,
    DialogueEpisode,
    GroundedAnswerEpisode,
    GroundedAnswerSplitClusters,
    GroundedQuestionEpisode,
)


ARTIFACT_KIND = "PH2_GG_EXECUTABLE_EVALUATION_OBSERVATION_V1"
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset({
    "artifact_kind", "clusters", "dialogue", "episode_id",
    "label_payload_hidden", "license_id", "question", "reference_input",
    "resource_budget", "schema_version", "split",
})
_BUDGET_FIELDS = frozenset({
    "max_context_bytes", "max_evidence_records",
    "max_generation_candidates", "max_surface_units",
    "max_verifier_dimensions",
})
_REFERENCE_FIELDS = frozenset({
    "antecedent_evidence_id", "antecedent_proposition_id", "options",
    "ordered_proposition_ids", "referring_evidence_id",
    "referring_proposition_id", "scope_id", "source_id",
})
_REFERENCE_OPTION_FIELDS = frozenset({"reference_surface", "strategy"})


# object-model: exception
class GenerationGeneralizationEvaluationObservationError(ValueError):
    """evaluation Observation 含 surface label、字段漂移或资源越界。"""


def _exact(
        value: Any, fields: frozenset[str], *, where: str,
        ) -> dict[str, Any]:
    """核验 JSON object 使用精确字段集合。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise GenerationGeneralizationEvaluationObservationError(
            f"{where} 字段集合漂移")
    return value


def _text(value: Any, *, where: str) -> str:
    """核验无首尾空白的非空文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise GenerationGeneralizationEvaluationObservationError(
            f"{where} 文本非法")
    return value


def _positive(value: Any, *, where: str) -> int:
    """核验严格正整数。"""
    if type(value) is not int or value <= 0:
        raise GenerationGeneralizationEvaluationObservationError(
            f"{where} 必须为严格正整数")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationEvaluationBudget:
    """冻结单 Observation 的候选、surface、verifier 与输入资源上限。"""

    max_context_bytes: int
    max_evidence_records: int
    max_generation_candidates: int
    max_surface_units: int
    max_verifier_dimensions: int

    def __post_init__(self) -> None:
        for name in _BUDGET_FIELDS:
            _positive(getattr(self, name), where=f"evaluation budget.{name}")
        if self.max_verifier_dimensions < 6:
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation budget 必须容纳六路 verifier")

    def to_dict(self) -> dict[str, int]:
        """导出规范资源上限。"""
        return {name: getattr(self, name) for name in sorted(_BUDGET_FIELDS)}

    @classmethod
    def from_dict(
            cls, value: Any,
            ) -> "GenerationGeneralizationEvaluationBudget":
        """从精确 JSON object 恢复资源上限。"""
        raw = _exact(value, _BUDGET_FIELDS, where="evaluation budget")
        return cls(**{name: raw[name] for name in _BUDGET_FIELDS})


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationReferenceSurfaceOption:
    """只公开 reference lexical resource，不公开完整 accepted surface。"""

    strategy: str
    reference_surface: str

    def __post_init__(self) -> None:
        if self.strategy not in REFERENCE_STRATEGIES:
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation reference strategy 未注册")
        _text(self.reference_surface, where="evaluation reference surface")

    def to_dict(self) -> dict[str, str]:
        """导出 strategy 与局部 reference surface。"""
        return {
            "reference_surface": self.reference_surface,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(
            cls, value: Any,
            ) -> "GenerationGeneralizationReferenceSurfaceOption":
        """从精确 JSON object 恢复 reference option。"""
        raw = _exact(
            value, _REFERENCE_OPTION_FIELDS,
            where="evaluation reference option",
        )
        return cls(
            _text(raw["strategy"], where="reference strategy"),
            _text(raw["reference_surface"], where="reference surface"),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationEvaluationReferenceInput:
    """双命题 reference planning 所需的 label-free 局部资源。"""

    antecedent_proposition_id: str
    referring_proposition_id: str
    antecedent_evidence_id: str
    referring_evidence_id: str
    source_id: str
    scope_id: int
    ordered_proposition_ids: tuple[str, ...]
    options: tuple[GenerationGeneralizationReferenceSurfaceOption, ...]

    def __post_init__(self) -> None:
        for name in (
                "antecedent_proposition_id", "referring_proposition_id",
                "antecedent_evidence_id", "referring_evidence_id",
                "source_id"):
            _text(getattr(self, name), where=f"evaluation reference.{name}")
        _positive(self.scope_id, where="evaluation reference.scope_id")
        if (self.ordered_proposition_ids != (
                self.antecedent_proposition_id,
                self.referring_proposition_id)):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation reference 必须是 antecedent 在前的双命题")
        if (tuple(item.strategy for item in self.options)
                != REFERENCE_STRATEGIES):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation reference options 未覆盖冻结策略顺序")

    def validate_question(self, question: GroundedQuestionEpisode) -> None:
        """核对 reference 输入只引用当前问题的可见 Evidence 和 scope。"""
        if (question.answer_plan.response_act != "ANSWER"
                or question.answer_plan.ordered_claim_ids
                != self.ordered_proposition_ids
                or question.evidence_scope_id != self.scope_id
                or self.source_id not in question.answer_plan.citation_source_ids):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation reference 与 answer plan/source/scope 漂移")
        evidence = {item.evidence_id: item for item in question.evidence}
        antecedent = evidence.get(self.antecedent_evidence_id)
        referring = evidence.get(self.referring_evidence_id)
        if (antecedent is None or referring is None
                or antecedent.proposition_id
                != self.antecedent_proposition_id
                or referring.proposition_id != self.referring_proposition_id
                or any(item.source_id != self.source_id
                       or item.scope_id != self.scope_id
                       or not item.support or item.refute
                       for item in (antecedent, referring))):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation reference forming Evidence 漂移")

    def surface_for(self, strategy: str) -> str:
        """返回 selected strategy 的唯一局部 reference surface。"""
        if strategy not in REFERENCE_STRATEGIES:
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation reference strategy 未注册")
        return next(
            item.reference_surface for item in self.options
            if item.strategy == strategy)

    def to_dict(self) -> dict[str, object]:
        """导出不含完整 surface、accepted id 或失败标签的引用输入。"""
        return {
            "antecedent_evidence_id": self.antecedent_evidence_id,
            "antecedent_proposition_id": self.antecedent_proposition_id,
            "options": [item.to_dict() for item in self.options],
            "ordered_proposition_ids": list(self.ordered_proposition_ids),
            "referring_evidence_id": self.referring_evidence_id,
            "referring_proposition_id": self.referring_proposition_id,
            "scope_id": self.scope_id,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(
            cls, value: Any,
            ) -> "GenerationGeneralizationEvaluationReferenceInput":
        """从精确 JSON object 恢复 reference 输入。"""
        raw = _exact(value, _REFERENCE_FIELDS, where="evaluation reference")
        if (not isinstance(raw["ordered_proposition_ids"], list)
                or not isinstance(raw["options"], list)):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation reference list 字段非法")
        return cls(
            _text(raw["antecedent_proposition_id"],
                  where="antecedent proposition"),
            _text(raw["referring_proposition_id"],
                  where="referring proposition"),
            _text(raw["antecedent_evidence_id"],
                  where="antecedent evidence"),
            _text(raw["referring_evidence_id"],
                  where="referring evidence"),
            _text(raw["source_id"], where="reference source"),
            raw["scope_id"],
            tuple(_text(item, where="ordered proposition")
                  for item in raw["ordered_proposition_ids"]),
            tuple(GenerationGeneralizationReferenceSurfaceOption.from_dict(
                item) for item in raw["options"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationEvaluationObservation:
    """候选可见的 held-out grounded input，不含 evaluator surface 集。"""

    episode_id: str
    clusters: GroundedAnswerSplitClusters
    question: GroundedQuestionEpisode
    dialogue: DialogueEpisode
    resource_budget: GenerationGeneralizationEvaluationBudget
    reference_course: (
        GenerationGeneralizationEvaluationReferenceInput | None) = None
    split: str = "held_out"
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        _text(self.episode_id, where="evaluation observation episode_id")
        if self.split != "held_out":
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation Observation split 必须为 held_out")
        if not isinstance(self.clusters, GroundedAnswerSplitClusters):
            raise TypeError("evaluation observation clusters 类型错误")
        if not isinstance(self.question, GroundedQuestionEpisode):
            raise TypeError("evaluation observation question 类型错误")
        if not isinstance(self.dialogue, DialogueEpisode):
            raise TypeError("evaluation observation dialogue 类型错误")
        if not isinstance(
                self.resource_budget,
                GenerationGeneralizationEvaluationBudget):
            raise TypeError("evaluation observation budget 类型错误")
        if (self.reference_course is not None
                and not isinstance(
                    self.reference_course,
                    GenerationGeneralizationEvaluationReferenceInput)):
            raise TypeError("evaluation observation reference 类型错误")
        if self.dialogue.turns[-1].surface != self.question.question_surface:
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation dialogue 当前 USER turn 与 question 漂移")
        if not {
                self.question.evidence_scope_id,
                self.question.response_scope_id,
                } <= set(self.dialogue.active_scope_ids):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation Evidence/response scope 不在 active scopes")
        evidence_count = len(self.question.evidence)
        candidate_count = len({
            item.proposition_id for item in self.question.evidence})
        context_bytes = len(self.question.context_surface.encode("utf-8"))
        if (evidence_count > self.resource_budget.max_evidence_records
                or candidate_count
                > self.resource_budget.max_generation_candidates
                or context_bytes > self.resource_budget.max_context_bytes):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation Observation 超出冻结资源上限")
        if self.reference_course is None:
            if (self.question.answer_plan.response_act == "ANSWER"
                    and candidate_count != 1):
                raise GenerationGeneralizationEvaluationObservationError(
                    "evaluation 单命题 ANSWER candidate 数量非法")
        else:
            self.reference_course.validate_question(self.question)
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    def to_dict(self) -> dict[str, object]:
        """导出不含 `surfaces`、accepted/rejected 或 evaluator verdict 的值。"""
        return {
            "artifact_kind": ARTIFACT_KIND,
            "clusters": self.clusters.to_dict(),
            "dialogue": self.dialogue.to_dict(),
            "episode_id": self.episode_id,
            "label_payload_hidden": 1,
            "license_id": LICENSE_ID,
            "question": self.question.to_dict(),
            "reference_input": (
                None if self.reference_course is None
                else self.reference_course.to_dict()),
            "resource_budget": self.resource_budget.to_dict(),
            "schema_version": SCHEMA_VERSION,
            "split": self.split,
        }

    @classmethod
    def from_dict(
            cls, value: Any,
            ) -> "GenerationGeneralizationEvaluationObservation":
        """从精确 canonical JSON object 恢复 label-free Observation。"""
        raw = _exact(value, _TOP_LEVEL_FIELDS, where="evaluation observation")
        if (raw["artifact_kind"] != ARTIFACT_KIND
                or raw["schema_version"] != SCHEMA_VERSION
                or raw["license_id"] != LICENSE_ID
                or raw["label_payload_hidden"] != 1
                or raw["split"] != "held_out"):
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation Observation kind/version/license/split 漂移")
        reference = raw["reference_input"]
        return cls(
            _text(raw["episode_id"], where="evaluation episode_id"),
            GroundedAnswerSplitClusters.from_dict(raw["clusters"]),
            GroundedQuestionEpisode.from_dict(raw["question"]),
            DialogueEpisode.from_dict(raw["dialogue"]),
            GenerationGeneralizationEvaluationBudget.from_dict(
                raw["resource_budget"]),
            (None if reference is None
             else GenerationGeneralizationEvaluationReferenceInput.from_dict(
                 reference)),
            raw["split"],
        )

    @classmethod
    def from_held_out_episode(
            cls,
            episode: GroundedAnswerEpisode,
            resource_budget: GenerationGeneralizationEvaluationBudget,
            ) -> "GenerationGeneralizationEvaluationObservation":
        """在 evaluator 边界剥离完整 surface 集，只保留候选可见资料。"""
        if (not isinstance(episode, GroundedAnswerEpisode)
                or episode.split != "held_out"):
            raise GenerationGeneralizationEvaluationObservationError(
                "只允许从 held_out GroundedAnswerEpisode 剥离 Observation")
        reference = None
        course = episode.reference_course
        if course is not None:
            reference = GenerationGeneralizationEvaluationReferenceInput(
                course.antecedent_proposition_id,
                course.referring_proposition_id,
                course.antecedent_evidence_id,
                course.referring_evidence_id,
                course.source_id,
                course.scope_id,
                course.ordered_proposition_ids,
                tuple(GenerationGeneralizationReferenceSurfaceOption(
                    item.strategy, item.reference_surface)
                    for item in course.surface_labels),
            )
        return cls(
            episode.episode_id,
            episode.clusters,
            episode.question,
            episode.dialogue,
            resource_budget,
            reference,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 canonical label-free Observation 的有界身份。"""
        if not self._stable_key_cache:
            raise RuntimeError("evaluation observation stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """从规范公开字节形成确定性身份。"""
        return integer_tuple_fingerprint(
            tuple(canonical_json_line(self.to_dict())),
            domain="gg03.evaluation.observation.v1",
        )


def read_generation_generalization_evaluation_observations(
        path: str | Path,
        ) -> tuple[GenerationGeneralizationEvaluationObservation, ...]:
    """严格回读 canonical JSONL，并拒绝重复 Observation identity。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise GenerationGeneralizationEvaluationObservationError(
            "evaluation Observation inventory 无法读取") from error
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise GenerationGeneralizationEvaluationObservationError(
            "evaluation Observation inventory 换行非法")
    observations = []
    for line in payload.splitlines(keepends=True):
        try:
            value = parse_canonical_json_bytes(
                line[:-1], require_object=True)
        except DatasetContractError as error:
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation Observation inventory 非 canonical JSONL") from error
        if canonical_json_line(value) != line:
            raise GenerationGeneralizationEvaluationObservationError(
                "evaluation Observation inventory 字节非规范")
        observations.append(
            GenerationGeneralizationEvaluationObservation.from_dict(value))
    ids = tuple(item.episode_id for item in observations)
    keys = tuple(item.stable_key() for item in observations)
    if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
        raise GenerationGeneralizationEvaluationObservationError(
            "evaluation Observation identity 重复")
    return tuple(observations)


__all__ = [
    "ARTIFACT_KIND",
    "GenerationGeneralizationEvaluationBudget",
    "GenerationGeneralizationEvaluationObservation",
    "GenerationGeneralizationEvaluationObservationError",
    "GenerationGeneralizationEvaluationReferenceInput",
    "GenerationGeneralizationReferenceSurfaceOption",
    "read_generation_generalization_evaluation_observations",
]
