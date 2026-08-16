"""E-01 六路 executable TRAIN case 对既有 CC0 grounded 课程的引用合同。"""
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
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    ARTIFACT_KIND as GROUNDED_ARTIFACT_KIND,
    LICENSE_ID as GROUNDED_LICENSE_ID,
    REFERENCE_STRATEGIES,
    GroundedAnswerEpisode,
    read_grounded_answer_episodes,
)


ARTIFACT_KIND = "PH2_GG_EXECUTABLE_TRAIN_CASE_V1"
COURSE_VERSION = 1
RUNTIME_FAMILIES = (
    "REFERENCE_MULTI_PROPOSITION",
    "SINGLE_PROPOSITION",
)
REFERENCE_POLICIES = ("NONE", *REFERENCE_STRATEGIES)

_FIELDS = frozenset({
    "artifact_kind",
    "case_id",
    "course_version",
    "reference_strategy",
    "requirement",
    "response_act",
    "runtime_family",
    "source_artifact_kind",
    "source_episode_id",
    "source_license_id",
    "split",
})
_REQUIREMENT_SHAPES = {
    "ADDRESSEE_RECOVERABILITY": (
        "REFERENCE_MULTI_PROPOSITION", "ANTECEDENT_REFERENCE", "ANSWER"),
    "COMMUNICATIVE_TASK": (
        "SINGLE_PROPOSITION", "NONE", "CLARIFY"),
    "INDEPENDENT_UNDERSTANDING_READBACK": (
        "SINGLE_PROPOSITION", "NONE", "ANSWER"),
    "LEGAL_OBJECT_COMPOSITION": (
        "SINGLE_PROPOSITION", "NONE", "ANSWER"),
    "SOURCE_UNCERTAINTY_CITATION": (
        "SINGLE_PROPOSITION", "NONE", "CONFLICT"),
    "STRUCTURE_SLOT_ORDER": (
        "REFERENCE_MULTI_PROPOSITION", "EXPLICIT_REPETITION", "ANSWER"),
}


# object-model: exception
class GenerationGeneralizationExecutableTrainCourseError(ValueError):
    """E-01 case catalog 或其 grounded source 引用不满足冻结边界。"""


def _exact(value: Any, *, where: str) -> dict[str, Any]:
    """核验 case JSON object 使用精确字段集合。"""
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise GenerationGeneralizationExecutableTrainCourseError(
            f"{where} 字段集合漂移")
    return value


def _text(value: Any, *, where: str) -> str:
    """核验非空且无首尾空白的文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise GenerationGeneralizationExecutableTrainCourseError(
            f"{where} 必须是无首尾空白的非空文本")
    return value


def _text_values(*values: str) -> tuple[int, ...]:
    """把 catalog 文本编码成无歧义整数片段。"""
    result = []
    for value in values:
        raw = value.encode("utf-8")
        result.extend((len(raw), *raw))
    return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationExecutableTrainCase:
    """一项 requirement 对现有 TRAIN episode 和 runtime family 的引用。"""

    case_id: str
    requirement: str
    source_episode_id: str
    runtime_family: str
    reference_strategy: str
    response_act: str

    def __post_init__(self) -> None:
        for name in ("case_id", "source_episode_id", "response_act"):
            _text(getattr(self, name), where=f"E-01 case.{name}")
        if self.requirement not in INDEPENDENT_VERIFIER_REQUIREMENTS:
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 case requirement 未注册")
        if self.runtime_family not in RUNTIME_FAMILIES:
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 case runtime family 未注册")
        if self.reference_strategy not in REFERENCE_POLICIES:
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 case reference strategy 未注册")
        if _REQUIREMENT_SHAPES[self.requirement] != (
                self.runtime_family,
                self.reference_strategy,
                self.response_act):
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 case requirement/runtime/strategy/act 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出不含 surface、accepted set 或 expected answer 的规范值。"""
        return {
            "artifact_kind": ARTIFACT_KIND,
            "case_id": self.case_id,
            "course_version": COURSE_VERSION,
            "reference_strategy": self.reference_strategy,
            "requirement": self.requirement,
            "response_act": self.response_act,
            "runtime_family": self.runtime_family,
            "source_artifact_kind": GROUNDED_ARTIFACT_KIND,
            "source_episode_id": self.source_episode_id,
            "source_license_id": GROUNDED_LICENSE_ID,
            "split": "train",
        }

    @classmethod
    def from_dict(
            cls, value: Any,
            ) -> "GenerationGeneralizationExecutableTrainCase":
        """从严格 public catalog record 恢复一项 case。"""
        raw = _exact(value, where="E-01 executable train case")
        if (raw["artifact_kind"] != ARTIFACT_KIND
                or raw["course_version"] != COURSE_VERSION
                or raw["source_artifact_kind"] != GROUNDED_ARTIFACT_KIND
                or raw["source_license_id"] != GROUNDED_LICENSE_ID
                or raw["split"] != "train"):
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 case kind/version/source/license/split 漂移")
        return cls(
            _text(raw["case_id"], where="case_id"),
            _text(raw["requirement"], where="requirement"),
            _text(raw["source_episode_id"], where="source_episode_id"),
            _text(raw["runtime_family"], where="runtime_family"),
            _text(raw["reference_strategy"], where="reference_strategy"),
            _text(raw["response_act"], where="response_act"),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 case、requirement 和 source episode 的确定性身份。"""
        return integer_tuple_fingerprint(
            _text_values(
                self.case_id,
                self.requirement,
                self.source_episode_id,
                self.runtime_family,
                self.reference_strategy,
                self.response_act,
            ),
            domain="gg03.executable.train.case.v1",
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationExecutableTrainCourse:
    """六路 case 与现有 TRAIN episode 的完整公开引用课程。"""

    cases: tuple[GenerationGeneralizationExecutableTrainCase, ...]
    source_episodes: tuple[GroundedAnswerEpisode, ...]
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if (not isinstance(self.cases, tuple)
                or any(not isinstance(
                    item, GenerationGeneralizationExecutableTrainCase)
                    for item in self.cases)):
            raise TypeError("E-01 course cases 类型错误")
        if tuple(item.requirement for item in self.cases) != (
                INDEPENDENT_VERIFIER_REQUIREMENTS):
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 course 未按冻结顺序精确覆盖六路 requirement")
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 course case id 重复")
        if (not isinstance(self.source_episodes, tuple)
                or any(not isinstance(item, GroundedAnswerEpisode)
                       for item in self.source_episodes)):
            raise TypeError("E-01 course source episodes 类型错误")
        episodes = {item.episode_id: item for item in self.source_episodes}
        if len(episodes) != len(self.source_episodes):
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 course source episode 重复")
        if set(episodes) != {item.source_episode_id for item in self.cases}:
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 course source episode inventory 漂移")
        for case in self.cases:
            episode = episodes[case.source_episode_id]
            if (episode.split != "train"
                    or episode.question.answer_plan.response_act
                    != case.response_act):
                raise GenerationGeneralizationExecutableTrainCourseError(
                    "E-01 case source split/response act 漂移")
            reference_required = (
                case.runtime_family == "REFERENCE_MULTI_PROPOSITION")
            if reference_required != (episode.reference_course is not None):
                raise GenerationGeneralizationExecutableTrainCourseError(
                    "E-01 case runtime family 与 source reference 能力漂移")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    def episode_for(
            self, case: GenerationGeneralizationExecutableTrainCase,
            ) -> GroundedAnswerEpisode:
        """按已验证 source episode identity 返回 case 的训练材料。"""
        if case not in self.cases:
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 case 不属于 course")
        return next(
            item for item in self.source_episodes
            if item.episode_id == case.source_episode_id)

    def case_for_requirement(
            self, requirement: str,
            ) -> GenerationGeneralizationExecutableTrainCase:
        """按冻结 requirement 返回唯一 case。"""
        if requirement not in INDEPENDENT_VERIFIER_REQUIREMENTS:
            raise GenerationGeneralizationExecutableTrainCourseError(
                "E-01 requirement 未注册")
        return next(item for item in self.cases
                    if item.requirement == requirement)

    def stable_key(self) -> tuple[int, ...]:
        """返回六路 case/source inventory 的缓存身份。"""
        if not self._stable_key_cache:
            raise RuntimeError("E-01 course stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """从 case 和 source episode public value 形成有界内容身份。"""
        values = [len(self.cases)]
        for case in self.cases:
            values.extend(case.stable_key())
        values.append(len(self.source_episodes))
        for episode in self.source_episodes:
            payload = canonical_json_line(episode.to_dict())
            values.extend(integer_tuple_fingerprint(
                tuple(payload),
                domain="gg03.executable.train.source.episode.v1",
            ))
        return integer_tuple_fingerprint(
            tuple(values), domain="gg03.executable.train.course.v1")


def read_generation_generalization_executable_train_course(
        case_path: str | Path,
        grounded_path: str | Path,
        ) -> GenerationGeneralizationExecutableTrainCourse:
    """严格回读六路 case catalog，并核对其全部 grounded TRAIN source。"""
    try:
        payload = Path(case_path).read_bytes()
    except OSError as error:
        raise GenerationGeneralizationExecutableTrainCourseError(
            "E-01 case catalog 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise GenerationGeneralizationExecutableTrainCourseError(
            "E-01 case catalog 必须非空并以换行结束")
    cases = []
    for line_number, line in enumerate(
            payload.splitlines(keepends=True), start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            raise GenerationGeneralizationExecutableTrainCourseError(
                f"E-01 case catalog 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise GenerationGeneralizationExecutableTrainCourseError(
                f"E-01 case catalog 第 {line_number} 行不是规范 JSON") from error
        if canonical_json_line(value) != line:
            raise GenerationGeneralizationExecutableTrainCourseError(
                f"E-01 case catalog 第 {line_number} 行不是规范字节")
        cases.append(GenerationGeneralizationExecutableTrainCase.from_dict(
            value))
    referenced = {item.source_episode_id for item in cases}
    episodes = tuple(
        item for item in read_grounded_answer_episodes(grounded_path)
        if item.episode_id in referenced)
    return GenerationGeneralizationExecutableTrainCourse(
        tuple(cases), episodes)


__all__ = [
    "ARTIFACT_KIND",
    "COURSE_VERSION",
    "GenerationGeneralizationExecutableTrainCase",
    "GenerationGeneralizationExecutableTrainCourse",
    "GenerationGeneralizationExecutableTrainCourseError",
    "REFERENCE_POLICIES",
    "RUNTIME_FAMILIES",
    "read_generation_generalization_executable_train_course",
]
