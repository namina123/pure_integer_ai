"""E-04 assessment-aware reference choice 的 actual execution 收口。"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.cognition.shared.evidence_candidate import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_generation_choice_assessment_consumer import (
    GenerationChoiceAssessmentConsumerReport,
)
from pure_integer_ai.experiments.ph2_generation_choice_assessment_selector import (
    ASSESSMENT_SELECTOR_REASONS,
    GenerationChoiceAssessmentSelection,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice import (
    GroundedAnswerReferenceSelection,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_verification import (
    GroundedAnswerReferenceGG02Run,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_SUPPORT,
)


ASSESSMENT_CHOICE_CLOSURE_STATUS = "PASS_ASSESSMENT_AWARE_ACTUAL_CHOICE"
_REFERENCE_KIND = "DISCOURSE_REFERENCE_CHOICE"
_STRUCTURE_KIND = "PROPOSITION_STRUCTURE_CHOICE"


# object-model: exception
class GenerationGeneralizationAssessmentChoiceClosureError(ValueError):
    """E-04 assessment、actual run 或关闭层边界发生漂移。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _selection(
        run: GroundedAnswerReferenceGG02Run,
        ) -> GroundedAnswerReferenceSelection:
    """从 GG-02 actual Use 回读先于 syntax 的 reference selection。"""
    return run.verification.uses.reference.installation.reference_selection


def _choice_attribution(run: GroundedAnswerReferenceGG02Run, kind: str):
    """返回一个 actual choice layer 的唯一 GG-02 attribution。"""
    matches = tuple(
        item for item in run.attribution.choices
        if item.choice.choice_kind == kind)
    if len(matches) != 1:
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 GG-02 attribution 未恢复唯一 choice layer")
    return matches[0]


def _validate_actual_run(run: GroundedAnswerReferenceGG02Run) -> None:
    """核验同次 compilation、surface、parse、Use 与 verifier claim。"""
    if not isinstance(run, GroundedAnswerReferenceGG02Run):
        raise TypeError("E-04 actual GG-02 run 类型错误")
    uses = run.verification.uses
    reference = uses.reference
    structure = uses.structure
    actual = reference.run
    installation = reference.installation
    selection = installation.reference_selection
    if (structure.run is not actual
            or structure.installation is not installation
            or installation.compilation != selection.compilation
            or reference.choice_before != selection.choice
            or reference.choice_after.exact_uses != (reference.use,)
            or structure.choice_after.exact_uses != (structure.use,)):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 compilation/reference/structure exact Use 未绑定同次 selection")
    execution = actual.generation
    postcheck = actual.postcheck
    if (execution is None or not execution.complete or postcheck is None
            or postcheck.request.execution != execution
            or not postcheck.parsed.succeeded
            or postcheck.parsed.observation is None):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 actual generation 或 parser readback 未闭合")
    parse_request = GenerationSurfaceParseRequest.from_execution(execution)
    observation = postcheck.parsed.observation
    if (observation.parse_request_key != parse_request.stable_key()
            or observation.representations != execution.representations
            or observation.source != selection.source
            or observation.scope != selection.scope):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 parser readback 未绑定 actual surface/source/scope")
    reference_attribution = _choice_attribution(run, _REFERENCE_KIND)
    structure_attribution = _choice_attribution(run, _STRUCTURE_KIND)
    if (reference_attribution.choice != reference.choice_after
            or reference_attribution.use != reference.use
            or structure_attribution.choice != structure.choice_after
            or structure_attribution.use != structure.use):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 GG-02 attribution 未绑定 reference/structure exact Use")
    reference_claims = run.verification.claims_for(_REFERENCE_KIND)
    structure_claims = run.verification.claims_for(_STRUCTURE_KIND)
    if (not reference_claims or not structure_claims
            or any(not item.evidence_keys
                   for item in (*reference_claims, *structure_claims))):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 reference/structure requirement evidence 缺失")
    if any(
            result.applicability == APPLICABILITY_APPLICABLE
            and (result.verdict != VERDICT_SUPPORT
                 or result.operational_failure is not None)
            for result in run.verification.report.results):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 actual applicable verifier 未全部 support")


def _validate_update(
        report: GenerationChoiceAssessmentConsumerReport,
        training: GroundedAnswerReferenceGG02Run,
        stance: int,
        ) -> None:
    """核验一个 assessment update 来自对应 TRAIN actual reference Use。"""
    if not isinstance(report, GenerationChoiceAssessmentConsumerReport):
        raise TypeError("E-04 assessment update report 类型错误")
    if (len(report.records) != 1 or report.teacher_call_count != 0
            or report.records[0].prepared.stance != stance):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 assessment update 数量、stance 或 teacher 边界漂移")
    expected = _choice_attribution(training, _REFERENCE_KIND)
    prepared = report.records[0].prepared
    if (prepared.attribution != expected
            or prepared.assessment.use != expected.use
            or prepared.assessment.choice_kind != _REFERENCE_KIND):
        raise GenerationGeneralizationAssessmentChoiceClosureError(
            "E-04 assessment update 未绑定 TRAIN actual choice/Use")


def _assessment_key(
        selection: GenerationChoiceAssessmentSelection,
        ) -> tuple[int, ...]:
    """无损编码 assessment 分母、投影、baseline、selected 与 trace。"""
    values = [len(selection.options)]
    for option in selection.options:
        values.extend(_packed(option.stable_key()))
    values.extend((
        *_packed(selection.baseline.stable_key()),
        *_packed(selection.selected.stable_key()),
        ASSESSMENT_SELECTOR_REASONS.index(selection.reason) + 1,
        len(selection.candidate_projections),
    ))
    for projection in selection.candidate_projections:
        values.extend(_packed(projection.trace_key()))
    values.extend(_packed(selection.trace))
    return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationAssessmentAwareTrainSlice:
    """TRAIN assessment 改变 reference choice 后的 E-04 单纵切证据。"""

    baseline_training: GroundedAnswerReferenceGG02Run
    alternative_training: GroundedAnswerReferenceGG02Run
    baseline_update: GenerationChoiceAssessmentConsumerReport
    alternative_update: GenerationChoiceAssessmentConsumerReport
    assessment: GenerationChoiceAssessmentSelection
    actual: GroundedAnswerReferenceGG02Run
    disabled: GenerationChoiceAssessmentSelection
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        for run in (
                self.baseline_training,
                self.alternative_training,
                self.actual):
            _validate_actual_run(run)
        _validate_update(
            self.baseline_update, self.baseline_training, EVIDENCE_REFUTE)
        _validate_update(
            self.alternative_update,
            self.alternative_training,
            EVIDENCE_SUPPORT,
        )
        if not isinstance(
                self.assessment, GenerationChoiceAssessmentSelection):
            raise TypeError("E-04 assessment selection 类型错误")
        if not isinstance(self.disabled, GenerationChoiceAssessmentSelection):
            raise TypeError("E-04 disabled selection 类型错误")

        baseline = _selection(self.baseline_training)
        alternative = _selection(self.alternative_training)
        actual = _selection(self.actual)
        if (baseline == alternative
                or baseline.options != alternative.options
                or baseline.context != alternative.context
                or baseline.structure_selection
                != alternative.structure_selection
                or baseline.lexical_selection != alternative.lexical_selection
                or baseline.choice.competition_key
                != alternative.choice.competition_key):
            raise GenerationGeneralizationAssessmentChoiceClosureError(
                "E-04 TRAIN strategies 未共享完整 competition denominator")
        options = (baseline.choice, alternative.choice)
        active = tuple(
            item.choice for item in self.assessment.candidate_projections
            if item.active)
        if (self.assessment.options != options
                or self.assessment.baseline != baseline.choice
                or self.assessment.selected != alternative.choice
                or self.assessment.reason != "UNIQUE_ACTIVE_ASSESSMENT"
                or active != (alternative.choice,)
                or actual != alternative
                or self.actual.verification.uses.reference.choice_before
                != self.assessment.selected):
            raise GenerationGeneralizationAssessmentChoiceClosureError(
                "E-04 assessment 未改变并绑定下一次 actual reference choice")

        baseline_uses = self.baseline_training.verification.uses
        actual_uses = self.actual.verification.uses
        baseline_run = baseline_uses.reference.run
        actual_run = actual_uses.reference.run
        if (actual_run is self.alternative_training.verification.uses.reference.run
                or baseline_run.generation is None
                or actual_run.generation is None
                or baseline_run.generation.rendered.units
                == actual_run.generation.rendered.units
                or baseline_uses.reference.use == actual_uses.reference.use):
            raise GenerationGeneralizationAssessmentChoiceClosureError(
                "E-04 assessment 后未形成独立且已改变的 actual surface/Use")
        for baseline_layer, actual_layer in (
                (baseline_uses.content, actual_uses.content),
                (baseline_uses.structure, actual_uses.structure),
                (baseline_uses.lexical, actual_uses.lexical),
                (baseline_uses.task, actual_uses.task)):
            if baseline_layer.choice_before != actual_layer.choice_before:
                raise GenerationGeneralizationAssessmentChoiceClosureError(
                    "E-04 reference assessment 非法改变了其他 choice layer")

        if (self.disabled.options != options
                or self.disabled.baseline != baseline.choice
                or self.disabled.selected != baseline.choice
                or self.disabled.reason != "DISABLED_BASELINE"
                or any(item.state != "NOT_READ_DISABLED"
                       for item in self.disabled.candidate_projections)):
            raise GenerationGeneralizationAssessmentChoiceClosureError(
                "E-04 disabled layer 未回退 baseline 或读取了 projection")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    @property
    def status(self) -> str:
        """返回 E-04 TRAIN 单纵切收口状态。"""
        return ASSESSMENT_CHOICE_CLOSURE_STATUS

    @property
    def complete(self) -> int:
        """合同构造成功即表示 E-04 单纵切闭合。"""
        return 1

    def stable_key(self) -> tuple[int, ...]:
        """返回训练 update、assessment、actual 与关闭层的有界身份。"""
        if not self._stable_key_cache:
            raise RuntimeError("E-04 assessment-aware slice stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """仅引用既有 actual objects，不构造 held-out evidence。"""
        values = []
        for run in (
                self.baseline_training,
                self.alternative_training,
                self.actual):
            uses = run.verification.uses
            actual = uses.reference.run
            assert actual.generation is not None and actual.postcheck is not None
            for key in (
                    _selection(run).stable_key(),
                    uses.reference.use.stable_key(),
                    uses.structure.use.stable_key(),
                    actual.generation.stable_key(),
                    actual.postcheck.stable_key()):
                values.extend(_packed(integer_tuple_fingerprint(
                    key, domain="gg03.e04.actual.object.v1")))
        for report in (self.baseline_update, self.alternative_update):
            values.extend(_packed(report.records[0].stable_key()))
        values.extend(_packed(_assessment_key(self.assessment)))
        values.extend(_packed(_assessment_key(self.disabled)))
        return integer_tuple_fingerprint(
            tuple(values), domain="gg03.e04.assessment.choice.closure.v1")


def build_generation_generalization_assessment_aware_train_slice(
        baseline_training: GroundedAnswerReferenceGG02Run,
        alternative_training: GroundedAnswerReferenceGG02Run,
        baseline_update: GenerationChoiceAssessmentConsumerReport,
        alternative_update: GenerationChoiceAssessmentConsumerReport,
        assessment: GenerationChoiceAssessmentSelection,
        actual: GroundedAnswerReferenceGG02Run,
        disabled: GenerationChoiceAssessmentSelection,
        ) -> GenerationGeneralizationAssessmentAwareTrainSlice:
    """从 TRAIN actual assessment 与下一次 run 建立 E-04 只读收口。"""
    return GenerationGeneralizationAssessmentAwareTrainSlice(
        baseline_training,
        alternative_training,
        baseline_update,
        alternative_update,
        assessment,
        actual,
        disabled,
    )


__all__ = [
    "ASSESSMENT_CHOICE_CLOSURE_STATUS",
    "GenerationGeneralizationAssessmentAwareTrainSlice",
    "GenerationGeneralizationAssessmentChoiceClosureError",
    "build_generation_generalization_assessment_aware_train_slice",
]
