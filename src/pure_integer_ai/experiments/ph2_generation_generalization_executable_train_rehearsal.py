"""E-01 六路 TRAIN catalog 的 actual rehearsal 组合与部分覆盖合同。"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    minimal_instruction_identity,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRun,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_contract import (
    GenerationGeneralizationIndependentVerification,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_course import (
    GenerationGeneralizationExecutableTrainCase,
    GenerationGeneralizationExecutableTrainCourse,
)
from pure_integer_ai.experiments.ph2_generation_generalization_source_conflict import (
    GenerationGeneralizationSourceConflictInput,
    GenerationGeneralizationSourceConflictProtocol,
    run_generation_generalization_source_conflict_verification,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedAnswerEpisode,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_choice_use import (
    GroundedResponseActLexicalAdoptionLedger,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_runtime_factory import (
    GroundedResponseActRunLocalInstallation,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningBuild,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRun,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_SUPPORT,
    VerificationReport,
    VerificationResult,
)


_NAMESPACE = 20966
_RESPONSE_ACT_REQUIREMENTS = (
    "COMMUNICATIVE_TASK",
    "SOURCE_UNCERTAINTY_CITATION",
)


# object-model: exception
class GenerationGeneralizationTrainRehearsalError(ValueError):
    """TRAIN case、actual run 或 requirement 归因发生漂移。"""


def default_source_conflict_protocol(
        ) -> GenerationGeneralizationSourceConflictProtocol:
    """返回与 G-04 routes 分离的 source-conflict 公开协议。"""
    return GenerationGeneralizationSourceConflictProtocol(
        ProtocolKey((_NAMESPACE, 1)),
        ProtocolKey((_NAMESPACE, 2)),
        minimal_instruction_identity((_NAMESPACE, 3, 1)),
        minimal_instruction_identity((_NAMESPACE, 3, 2)),
    )


def _report_result(
        report: VerificationReport,
        dimension: ProtocolKey,
        verifier: ProtocolKey,
        ) -> VerificationResult:
    """按独立 route 精确读取唯一 verifier result。"""
    matches = tuple(
        item for item in report.results
        if item.dimension == dimension and item.verifier == verifier)
    if len(matches) != 1:
        raise GenerationGeneralizationTrainRehearsalError(
            "TRAIN rehearsal verifier route 未唯一命中")
    return matches[0]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationTrainRehearsalItem:
    """一项 catalog case 的 actual execution、readback 与独立 requirement 结果。"""

    case: GenerationGeneralizationExecutableTrainCase
    source_episode: GroundedAnswerEpisode
    choice: GenerationChoiceHypothesis
    use: GenerationChoiceUseRef
    execution: TypedGenerationExecution
    parse_request: GenerationSurfaceParseRequest
    postcheck: GenerationPostcheckRun
    verification_reports: tuple[VerificationReport, ...]
    verification: GenerationGeneralizationIndependentVerification
    trace: tuple[int, ...]
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(
                self.case, GenerationGeneralizationExecutableTrainCase):
            raise TypeError("TRAIN rehearsal case 类型错误")
        if not isinstance(self.source_episode, GroundedAnswerEpisode):
            raise TypeError("TRAIN rehearsal episode 类型错误")
        if (self.case.source_episode_id != self.source_episode.episode_id
                or self.case.response_act
                != self.source_episode.question.answer_plan.response_act):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal case/source episode 漂移")
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("TRAIN rehearsal choice 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("TRAIN rehearsal use 类型错误")
        if self.use not in self.choice.exact_uses:
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal Use 未回填 exact choice")
        if not isinstance(self.execution, TypedGenerationExecution):
            raise TypeError("TRAIN rehearsal execution 类型错误")
        if not self.execution.complete:
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal execution 未完成")
        if self.parse_request != GenerationSurfaceParseRequest.from_execution(
                self.execution):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal parse request 未由 actual execution 派生")
        if (not isinstance(self.postcheck, GenerationPostcheckRun)
                or self.postcheck.request.execution != self.execution):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal postcheck 未绑定 actual execution")
        if (not isinstance(self.verification_reports, tuple)
                or not self.verification_reports
                or any(not isinstance(item, VerificationReport)
                       for item in self.verification_reports)
                or any(not item.read_only for item in self.verification_reports)
                or self.postcheck.report not in self.verification_reports):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal reports 必须只读并包含 actual G-04")
        if not isinstance(
                self.verification,
                GenerationGeneralizationIndependentVerification):
            raise TypeError("TRAIN rehearsal verification 类型错误")
        if self.verification.requirement != self.case.requirement:
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal requirement 漂移")
        if not any(
                self.verification.result in report.results
                for report in self.verification_reports):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal result 不属于只读 report")
        goal = self.execution.plan.request.goal
        result = self.verification.result
        if (result.source != goal.source or result.scope != goal.scope
                or self.use.scope != goal.scope
                or goal.source not in self.choice.forming_sources):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal choice/verifier source/scope 漂移")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(item) is not int for item in self.trace)):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal trace 必须是非空严格整数 tuple")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    @property
    def passed(self) -> int:
        """只在 actual parse 成功且本 requirement applicable/support 时为一。"""
        result = self.verification.result
        return int(
            self.postcheck.parsed.succeeded
            and result.applicability == APPLICABILITY_APPLICABLE
            and result.verdict == VERDICT_SUPPORT
            and result.operational_failure is None
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 case、choice/Use、actual output 与 requirement 结果引用。"""
        if not self._stable_key_cache:
            raise RuntimeError("TRAIN rehearsal item stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """在构造完成时计算一次有界内容键。"""
        values = []
        for key in (
                self.case.stable_key(),
                self.choice.stable_key(),
                self.use.stable_key(),
                self.execution.stable_key(),
                self.parse_request.stable_key(),
                self.postcheck.stable_key(),
                self.verification.stable_key()):
            values.extend((len(key), *key))
        values.extend((len(self.trace), *self.trace))
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationTrainRehearsal:
    """允许施工期部分覆盖、但禁止 route/input 广播的六项 rehearsal。"""

    course: GenerationGeneralizationExecutableTrainCourse
    items: tuple[GenerationGeneralizationTrainRehearsalItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(
                self.course, GenerationGeneralizationExecutableTrainCourse):
            raise TypeError("TRAIN rehearsal course 类型错误")
        if (not isinstance(self.items, tuple)
                or any(not isinstance(
                    item, GenerationGeneralizationTrainRehearsalItem)
                    for item in self.items)):
            raise TypeError("TRAIN rehearsal items 类型错误")
        case_order = {case.case_id: index
                      for index, case in enumerate(self.course.cases)}
        if any(item.case not in self.course.cases for item in self.items):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal item 不属于 course")
        if tuple(case_order[item.case.case_id] for item in self.items) != tuple(
                sorted(case_order[item.case.case_id] for item in self.items)):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal items 未按 catalog 顺序")
        case_ids = tuple(item.case.case_id for item in self.items)
        inputs = tuple(item.verification.input_key for item in self.items)
        routes = tuple(
            (item.verification.result.dimension,
             item.verification.result.verifier)
            for item in self.items)
        if (len(set(case_ids)) != len(case_ids)
                or len(set(inputs)) != len(inputs)
                or len(set(routes)) != len(routes)):
            raise GenerationGeneralizationTrainRehearsalError(
                "TRAIN rehearsal case/input/verifier route 重复")

    @property
    def complete(self) -> int:
        """六项均形成 actual support 前不得声明 E-01 complete。"""
        return int(
            len(self.items) == len(INDEPENDENT_VERIFIER_REQUIREMENTS)
            and all(item.passed for item in self.items)
        )


def rehearse_grounded_response_act_case(
        course: GenerationGeneralizationExecutableTrainCourse,
        requirement: str,
        planning: GroundedResponseActPlanningBuild,
        installation: GroundedResponseActRunLocalInstallation,
        request: QuestionRequest,
        *,
        source_conflict_protocol: (
            GenerationGeneralizationSourceConflictProtocol | None) = None,
        ) -> tuple[GenerationGeneralizationTrainRehearsalItem, QuestionAnswerRun]:
    """运行一项 CLARIFY/CONFLICT case，并绑定 actual choice/Use/verifier。"""
    if requirement not in _RESPONSE_ACT_REQUIREMENTS:
        raise GenerationGeneralizationTrainRehearsalError(
            "response-act rehearsal requirement 不受支持")
    if not isinstance(
            course, GenerationGeneralizationExecutableTrainCourse):
        raise TypeError("response-act rehearsal course 类型错误")
    case = course.case_for_requirement(requirement)
    episode = course.episode_for(case)
    if (not isinstance(planning, GroundedResponseActPlanningBuild)
            or planning.episode != episode
            or installation.planning != planning.planning
            or installation.variant.template.stance
            != installation.lexical_choice.target_obligation
            or case.response_act
            != installation.compilation.target.response_act):
        raise GenerationGeneralizationTrainRehearsalError(
            "response-act rehearsal course/planning/installation 漂移")
    if not isinstance(request, QuestionRequest):
        raise TypeError("response-act rehearsal request 类型错误")
    run = installation.runtime.run(request)
    lexical_use = GroundedResponseActLexicalAdoptionLedger(
        installation).adopt(run)
    if (run.generation is None or run.postcheck is None
            or not run.generation.complete):
        raise GenerationGeneralizationTrainRehearsalError(
            "response-act rehearsal 未形成完整 actual run")
    execution = run.generation
    postcheck = run.postcheck
    parse_request = GenerationSurfaceParseRequest.from_execution(execution)
    reports = [postcheck.report]
    if requirement == "COMMUNICATIVE_TASK":
        result = _report_result(
            postcheck.report,
            postcheck.protocol.task_dimension,
            postcheck.protocol.task_verifier,
        )
    else:
        protocol = source_conflict_protocol or default_source_conflict_protocol()
        conflict_input = GenerationGeneralizationSourceConflictInput(
            episode,
            planning,
            lexical_use.choice_after,
            lexical_use.use,
            execution,
            parse_request,
            postcheck,
        )
        conflict_report = (
            run_generation_generalization_source_conflict_verification(
                protocol, conflict_input))
        reports.append(conflict_report)
        result = _report_result(
            conflict_report, protocol.dimension, protocol.verifier)
    if not result.claim_keys:
        raise GenerationGeneralizationTrainRehearsalError(
            "response-act rehearsal verifier 缺 actual input claim")
    verification = GenerationGeneralizationIndependentVerification(
        requirement,
        LosslessIntegerKey(result.claim_keys[0]),
        result,
    )
    item = GenerationGeneralizationTrainRehearsalItem(
        case,
        episode,
        lexical_use.choice_after,
        lexical_use.use,
        execution,
        parse_request,
        postcheck,
        tuple(reports),
        verification,
        (_NAMESPACE, 10,
         INDEPENDENT_VERIFIER_REQUIREMENTS.index(requirement) + 1),
    )
    return item, run


__all__ = [
    "GenerationGeneralizationTrainRehearsal",
    "GenerationGeneralizationTrainRehearsalError",
    "GenerationGeneralizationTrainRehearsalItem",
    "default_source_conflict_protocol",
    "rehearse_grounded_response_act_case",
]
