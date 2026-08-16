"""E-02/E-03 六路 TRAIN readback 与 verifier 课程级收口合同。"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_rehearsal import (
    GenerationGeneralizationTrainRehearsal,
    GenerationGeneralizationTrainRehearsalItem,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)


TRAIN_CLOSURE_STATUSES = (
    "FAIL_TRAIN_VERIFIER_CONJUNCTION",
    "NE_TRAIN_REQUIREMENT_INPUT_MISSING",
    "NE_TRAIN_READBACK_INDETERMINATE",
    "NE_TRAIN_VERIFIER_INPUT_INDETERMINATE",
    "PASS_TRAIN_COURSE_CLOSURE",
)
READBACK_COVERAGE_KINDS = (
    "SURFACE_UNITS",
    "REPRESENTATIONS",
    "PROPOSITIONS",
    "STRUCTURE",
    "REFERENCE_RESOLUTION",
    "CITATION_SOURCES",
    "STANCE",
    "TASK_RESULT",
    "SOURCE_CONFLICT",
)
_REQUIRED_COVERAGE = {
    "ADDRESSEE_RECOVERABILITY": (
        "SURFACE_UNITS",
        "REPRESENTATIONS",
        "PROPOSITIONS",
        "STRUCTURE",
        "REFERENCE_RESOLUTION",
        "CITATION_SOURCES",
        "STANCE",
    ),
    "COMMUNICATIVE_TASK": (
        "SURFACE_UNITS",
        "REPRESENTATIONS",
        "STRUCTURE",
        "STANCE",
        "TASK_RESULT",
    ),
    "INDEPENDENT_UNDERSTANDING_READBACK": (
        "SURFACE_UNITS",
        "REPRESENTATIONS",
        "PROPOSITIONS",
        "STRUCTURE",
        "CITATION_SOURCES",
        "STANCE",
    ),
    "LEGAL_OBJECT_COMPOSITION": (
        "SURFACE_UNITS",
        "REPRESENTATIONS",
        "PROPOSITIONS",
        "STRUCTURE",
        "CITATION_SOURCES",
        "STANCE",
    ),
    "SOURCE_UNCERTAINTY_CITATION": (
        "SURFACE_UNITS",
        "REPRESENTATIONS",
        "STRUCTURE",
        "STANCE",
        "TASK_RESULT",
        "SOURCE_CONFLICT",
    ),
    "STRUCTURE_SLOT_ORDER": (
        "SURFACE_UNITS",
        "REPRESENTATIONS",
        "PROPOSITIONS",
        "STRUCTURE",
        "CITATION_SOURCES",
        "STANCE",
    ),
}


# object-model: exception
class GenerationGeneralizationTrainClosureError(ValueError):
    """TRAIN readback、exact evidence 或课程级聚合边界发生漂移。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _readback_coverage(
        item: GenerationGeneralizationTrainRehearsalItem,
        ) -> tuple[str, ...]:
    """只从 actual parse observation 和 requirement result 恢复覆盖类型。"""
    parsed = item.postcheck.parsed
    observation = parsed.observation
    if observation is None:
        return ()
    execution = item.execution
    structure = execution.surface.preview.request.structure
    planning = execution.plan.request
    emitted = {
        key for sentence in structure.syntax.sentences
        for key in sentence.proposition_keys
    }
    recovered = {value.candidate_key for value in observation.propositions}
    candidates = {
        candidate.stable_key(): candidate for candidate in planning.candidates}
    expected_citations = set()
    if emitted.issubset(candidates):
        expected_citations = {
            source
            for key in emitted
            for source in candidates[key].citation_sources
        }
    result = item.verification.result
    supported = (
        result.applicability == APPLICABILITY_APPLICABLE
        and result.verdict == VERDICT_SUPPORT
        and result.operational_failure is None
    )
    present = set()
    if (observation.parse_request_key == item.parse_request.stable_key()
            and item.parse_request.units == execution.rendered.units):
        present.add("SURFACE_UNITS")
    if observation.representations == execution.representations:
        present.add("REPRESENTATIONS")
    if emitted and recovered == emitted:
        present.add("PROPOSITIONS")
    if observation.structure_payload:
        present.add("STRUCTURE")
    if expected_citations and set(
            observation.cited_sources) == expected_citations:
        present.add("CITATION_SOURCES")
    if observation.stance == structure.selection.stance:
        present.add("STANCE")
    if observation.task_observations:
        present.add("TASK_RESULT")
    if (item.case.requirement == "ADDRESSEE_RECOVERABILITY"
            and item.choice.choice_kind == "DISCOURSE_REFERENCE_CHOICE"
            and supported):
        present.add("REFERENCE_RESOLUTION")
    if (item.case.requirement == "SOURCE_UNCERTAINTY_CITATION"
            and observation.propositions == ()
            and observation.cited_sources == ()
            and supported):
        present.add("SOURCE_CONFLICT")
    return tuple(
        kind for kind in READBACK_COVERAGE_KINDS if kind in present)


def _evidence_keys(
        item: GenerationGeneralizationTrainRehearsalItem,
        ) -> tuple[LosslessIntegerKey, ...]:
    """引用同次 choice/Use/output/readback/postcheck/verifier 的完整稳定键。"""
    keys = [
        item.choice.stable_key(),
        item.use.stable_key(),
        item.execution.stable_key(),
        item.parse_request.stable_key(),
        item.postcheck.stable_key(),
        item.verification.input_key.components,
        item.verification.stable_key(),
    ]
    observation = item.postcheck.parsed.observation
    if observation is not None:
        keys.append(observation.stable_key())
    return tuple(LosslessIntegerKey(key) for key in keys)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationTrainRequirementAudit:
    """一项 TRAIN requirement 的 actual readback、claim 与 evidence 引用。"""

    item: GenerationGeneralizationTrainRehearsalItem
    coverage: tuple[str, ...] = field(init=False)
    claim_keys: tuple[LosslessIntegerKey, ...] = field(init=False)
    evidence_keys: tuple[LosslessIntegerKey, ...] = field(init=False)
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(
                self.item, GenerationGeneralizationTrainRehearsalItem):
            raise TypeError("TRAIN requirement audit item 类型错误")
        result = self.item.verification.result
        claims = tuple(LosslessIntegerKey(key) for key in result.claim_keys)
        evidence = _evidence_keys(self.item)
        if (result.applicability == APPLICABILITY_APPLICABLE
                and self.item.verification.input_key not in claims):
            raise GenerationGeneralizationTrainClosureError(
                "TRAIN applicable requirement 缺 exact input claim")
        if not evidence or len(set(evidence)) != len(evidence):
            raise GenerationGeneralizationTrainClosureError(
                "TRAIN requirement evidence path 为空或重复")
        object.__setattr__(self, "coverage", _readback_coverage(self.item))
        object.__setattr__(self, "claim_keys", claims)
        object.__setattr__(self, "evidence_keys", evidence)
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    @property
    def readback_complete(self) -> int:
        """本 requirement 的 parser/route 覆盖全部出现时返回一。"""
        required = _REQUIRED_COVERAGE[self.item.case.requirement]
        return int(set(required).issubset(self.coverage))

    def stable_key(self) -> tuple[int, ...]:
        """返回 requirement、覆盖、claim 和 actual evidence path。"""
        if not self._stable_key_cache:
            raise RuntimeError("TRAIN requirement audit stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """形成不依赖对象地址的有界审计身份。"""
        values = [
            INDEPENDENT_VERIFIER_REQUIREMENTS.index(
                self.item.case.requirement) + 1,
            len(self.coverage),
        ]
        values.extend(
            READBACK_COVERAGE_KINDS.index(kind) + 1
            for kind in self.coverage)
        values.append(len(self.claim_keys))
        for key in self.claim_keys:
            values.extend(_packed(key.components))
        values.append(len(self.evidence_keys))
        for key in self.evidence_keys:
            values.extend(_packed(key.components))
        values.append(self.readback_complete)
        return integer_tuple_fingerprint(
            tuple(values), domain="gg03.train.requirement.audit.v1")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationTrainCourseClosure:
    """六个独立 TRAIN execution 的 E-02/E-03 课程级只读聚合。"""

    rehearsal: GenerationGeneralizationTrainRehearsal
    audits: tuple[GenerationGeneralizationTrainRequirementAudit, ...] = field(
        init=False)
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(
                self.rehearsal, GenerationGeneralizationTrainRehearsal):
            raise TypeError("TRAIN course closure rehearsal 类型错误")
        audits = tuple(
            GenerationGeneralizationTrainRequirementAudit(item)
            for item in self.rehearsal.items)
        requirements = tuple(
            audit.item.case.requirement for audit in audits)
        expected = tuple(
            item for item in INDEPENDENT_VERIFIER_REQUIREMENTS
            if item in requirements)
        if requirements != expected:
            raise GenerationGeneralizationTrainClosureError(
                "TRAIN course closure requirement 顺序或覆盖漂移")
        execution_keys = tuple(
            audit.item.execution.stable_key() for audit in audits)
        parse_keys = tuple(
            audit.item.parse_request.stable_key() for audit in audits)
        input_keys = tuple(
            audit.item.verification.input_key for audit in audits)
        routes = tuple(
            (audit.item.verification.result.dimension,
             audit.item.verification.result.verifier)
            for audit in audits)
        if (len(set(execution_keys)) != len(execution_keys)
                or len(set(parse_keys)) != len(parse_keys)
                or len(set(input_keys)) != len(input_keys)
                or len(set(routes)) != len(routes)):
            raise GenerationGeneralizationTrainClosureError(
                "TRAIN course closure 复用了 execution/parse/input/route")
        object.__setattr__(self, "audits", audits)
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    @property
    def readback_coverage(self) -> tuple[str, ...]:
        """返回六项 parser/route 联合覆盖的冻结类型集合。"""
        present = {
            kind for audit in self.audits for kind in audit.coverage}
        return tuple(
            kind for kind in READBACK_COVERAGE_KINDS if kind in present)

    @property
    def status(self) -> str:
        """按 refute、缺输入、readback、indeterminate、全 support 分型。"""
        results = tuple(
            result
            for audit in self.audits
            for report in audit.item.verification_reports
            for result in report.results)
        if any(
                result.applicability == APPLICABILITY_APPLICABLE
                and result.verdict == VERDICT_REFUTE
                for result in results):
            return "FAIL_TRAIN_VERIFIER_CONJUNCTION"
        if len(self.audits) != len(INDEPENDENT_VERIFIER_REQUIREMENTS):
            return "NE_TRAIN_REQUIREMENT_INPUT_MISSING"
        if (any(not audit.item.postcheck.parsed.succeeded
                or not audit.readback_complete for audit in self.audits)
                or self.readback_coverage != READBACK_COVERAGE_KINDS):
            return "NE_TRAIN_READBACK_INDETERMINATE"
        selected = tuple(
            audit.item.verification.result for audit in self.audits)
        if (any(result.applicability != APPLICABILITY_APPLICABLE
                or result.verdict != VERDICT_SUPPORT
                or result.operational_failure is not None
                for result in selected)
                or any(not audit.item.postcheck.complete
                       for audit in self.audits)
                or any(result.applicability == APPLICABILITY_UNKNOWN
                       or result.operational_failure is not None
                       for result in results)):
            return "NE_TRAIN_VERIFIER_INPUT_INDETERMINATE"
        return "PASS_TRAIN_COURSE_CLOSURE"

    @property
    def complete(self) -> int:
        """只在 E-02/E-03 TRAIN course closure 全部闭合时返回一。"""
        return int(self.status == "PASS_TRAIN_COURSE_CLOSURE")

    def stable_key(self) -> tuple[int, ...]:
        """返回 course、六项 audit、覆盖和状态的有界身份。"""
        if not self._stable_key_cache:
            raise RuntimeError("TRAIN course closure stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """从现有 actual objects 形成只读聚合身份，不构造 held-out evidence。"""
        values = [*_packed(self.rehearsal.course.stable_key()), len(self.audits)]
        for audit in self.audits:
            values.extend(_packed(audit.stable_key()))
        values.append(len(self.readback_coverage))
        values.extend(
            READBACK_COVERAGE_KINDS.index(kind) + 1
            for kind in self.readback_coverage)
        values.append(TRAIN_CLOSURE_STATUSES.index(self.status) + 1)
        return integer_tuple_fingerprint(
            tuple(values), domain="gg03.train.course.closure.v1")


def build_generation_generalization_train_course_closure(
        rehearsal: GenerationGeneralizationTrainRehearsal,
        ) -> GenerationGeneralizationTrainCourseClosure:
    """从 E-01 rehearsal 建立 E-02/E-03 只读课程级收口。"""
    return GenerationGeneralizationTrainCourseClosure(rehearsal)


__all__ = [
    "GenerationGeneralizationTrainClosureError",
    "GenerationGeneralizationTrainCourseClosure",
    "GenerationGeneralizationTrainRequirementAudit",
    "READBACK_COVERAGE_KINDS",
    "TRAIN_CLOSURE_STATUSES",
    "build_generation_generalization_train_course_closure",
]
