"""E-05 label-free Observation 的 production generation actual runner。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanProtocol,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph,
    StructureOrderGraphPredicates,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
    StructureOrderLifecycleProtocol,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_evaluation,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
    GenerationPostcheckRun,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageGenerationConnector,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_runtime import (
    ProductionGenerationAliasRuntimeFactory,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    LoadedGenerationCandidatePack,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_generation_generalization_answer_verification import (
    GenerationGeneralizationAnswerVerificationInput,
    GenerationGeneralizationAnswerVerificationProtocol,
    run_generation_generalization_answer_verification,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_contract import (
    GenerationGeneralizationIndependentVerification,
)
from pure_integer_ai.experiments.ph2_generation_generalization_source_conflict import (
    GenerationGeneralizationSourceConflictInput,
    GenerationGeneralizationSourceConflictProtocol,
    run_generation_generalization_source_conflict_verification,
)
from pure_integer_ai.experiments.ph2_grounded_answer_choice_use import (
    GroundedAnswerLexicalAdoptionLedger,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorTarget,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice import (
    build_grounded_answer_reference_selection,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerClaimCandidateBinding,
    GroundedAnswerReferenceCompileRequest,
    compile_grounded_answer_reference_connector,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_episode_use import (
    GroundedAnswerReferenceEpisodeAdoptionLedger,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_postcheck import (
    GroundedAnswerReferenceEvidenceSourceVerifier,
    GroundedAnswerReferenceStructureVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_runtime_factory import (
    GroundedAnswerReferenceRunLocalBuild,
    GroundedAnswerReferenceRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_verification import (
    GroundedAnswerReferenceVerifierProtocol,
    build_grounded_answer_reference_verifier_protocol,
    verify_grounded_answer_reference_layers,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_choice_use import (
    GroundedResponseActLexicalAdoptionLedger,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileTarget,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_parser import (
    GroundedResponseActParserProtocol,
    GroundedResponseActStructureVerifier,
    GroundedResponseActTaskVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_runtime_factory import (
    GroundedResponseActRunLocalBuild,
    GroundedResponseActRunLocalComponents,
    GroundedResponseActRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalBuild,
    GroundedAnswerRunLocalComponents,
    GroundedAnswerRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_structure_choice_use import (
    GroundedAnswerStructureAdoptionLedger,
)
from pure_integer_ai.experiments.ph2_grounded_answer_verification import (
    GroundedAnswerEvidenceSourceVerifier,
    GroundedAnswerStructureVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningBuild,
    compile_grounded_answer_planning,
    compile_grounded_answer_reference_planning,
    compile_grounded_response_act_planning,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
    QuestionAnswerRun,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VerificationReport,
    VerificationResult,
)


_NAMESPACE = 22050
PATH_ANSWER = "ANSWER"
PATH_RESPONSE_ACT = "RESPONSE_ACT"
PATH_REFERENCE = "REFERENCE"
EVALUATION_PATHS = (PATH_ANSWER, PATH_RESPONSE_ACT, PATH_REFERENCE)
EVALUATION_ACTUAL_STATUSES = (
    "FAIL_EVALUATION_ACTUAL_CONJUNCTION",
    "NE_EVALUATION_ACTUAL_INPUT_INDETERMINATE",
    "NE_EVALUATION_ACTUAL_INPUT_MISSING",
    "PASS_EVALUATION_ACTUAL_CONJUNCTION",
)

_PATH_REQUIREMENTS = {
    (PATH_ANSWER, "ANSWER"): (
        "INDEPENDENT_UNDERSTANDING_READBACK",
        "LEGAL_OBJECT_COMPOSITION",
    ),
    (PATH_RESPONSE_ACT, "CLARIFY"): ("COMMUNICATIVE_TASK",),
    (PATH_RESPONSE_ACT, "CONFLICT"): (
        "SOURCE_UNCERTAINTY_CITATION",
    ),
    (PATH_REFERENCE, "ANSWER"): (
        "ADDRESSEE_RECOVERABILITY",
        "STRUCTURE_SLOT_ORDER",
    ),
}


# object-model: exception
class GenerationGeneralizationEvaluationRunnerError(RuntimeError):
    """candidate、Observation、actual path 或隔离审计发生漂移。"""


# object-model: exception
class GenerationGeneralizationEvaluationBatchRunError(
        GenerationGeneralizationEvaluationRunnerError):
    """携带安全 ordinal/path 的 batch 单条运行失败，不转发输入或异常消息。"""

    def __init__(self, observation_ordinal: int, evaluation_path: str) -> None:
        if type(observation_ordinal) is not int or observation_ordinal <= 0:
            raise ValueError("evaluation batch failure ordinal 非法")
        if evaluation_path not in EVALUATION_PATHS:
            raise ValueError("evaluation batch failure path 非法")
        self.observation_ordinal = observation_ordinal
        self.evaluation_path = evaluation_path
        super().__init__("evaluation batch item failed")


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放整数键增加长度边界。"""
    return len(value), *value


def _text_key(value: str) -> tuple[int, ...]:
    """把 actual surface 编码为确定性整数内容引用。"""
    return integer_tuple_fingerprint(
        tuple(value.encode("utf-8")),
        domain="gg03.evaluation.actual.surface.v1",
    )


def _positive_id(value: tuple[int, ...], *, domain: str) -> int:
    """把开放稳定键压为协议身份使用的正整数。"""
    fingerprint = integer_tuple_fingerprint(value, domain=domain)
    result = int.from_bytes(bytes(fingerprint[2:10]), "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _sha_key(value: str) -> tuple[int, ...]:
    """把已核验 SHA-256 转成严格整数 tuple。"""
    try:
        payload = bytes.fromhex(value)
    except ValueError as error:
        raise GenerationGeneralizationEvaluationRunnerError(
            "candidate pack SHA-256 非法") from error
    if len(payload) != 32:
        raise GenerationGeneralizationEvaluationRunnerError(
            "candidate pack SHA-256 长度非法")
    return tuple(payload)


def _instruction(run_id: int, family: int, ordinal: int) -> ObjectIdentity:
    """建立 run-local MinimalInstruction。"""
    return minimal_instruction_identity(
        (_NAMESPACE, run_id, family, ordinal))


def _protocol_key(run_id: int, family: int, ordinal: int) -> ProtocolKey:
    """建立 run-local verifier route。"""
    return ProtocolKey((_NAMESPACE, run_id, family, ordinal))


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
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation verifier route 未唯一命中")
    return matches[0]


def _visible_evidence_keys(
        planning: GroundedResponseActPlanningBuild,
        ) -> tuple[tuple[int, ...], ...]:
    """返回当前 Observation candidate 的真实 Evidence 内容引用。"""
    keys = {
        evidence.stable_key()
        for candidate in planning.planning.candidates
        for evidence in candidate.evidence
    }
    if not keys:
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation planning 缺 visible Evidence")
    return tuple(sorted(keys))


def _surface_attribution(
        theory: ObjectIdentity,
        candidate,
        purpose: ObjectIdentity,
        ) -> GenerationSurfaceAttribution:
    """把 exact connector 理论归属到当前 visible candidate Hypothesis。"""
    hypotheses = tuple(sorted(
        candidate.hypotheses, key=lambda item: item.stable_key()))
    if not hypotheses:
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation candidate 缺 surface attribution Hypothesis")
    return GenerationSurfaceAttribution(
        theory, hypotheses[0], purpose)


def _teacher_calls(teacher: object | None) -> int:
    """读取显式 teacher call counter；未知实现按零调用处理。"""
    value = 0 if teacher is None else getattr(teacher, "call_count", 0)
    if type(value) is not int or value < 0:
        raise GenerationGeneralizationEvaluationRunnerError(
            "teacher call counter 非法")
    return value


def _path_for(
        observation: GenerationGeneralizationEvaluationObservation,
        ) -> str:
    """只按 label-free typed input 选择现役 executable path。"""
    response_act = observation.question.answer_plan.response_act
    if response_act == "ANSWER":
        return PATH_REFERENCE if observation.reference_course else PATH_ANSWER
    if response_act in {"CLARIFY", "CONFLICT"}:
        if observation.reference_course is not None:
            raise GenerationGeneralizationEvaluationRunnerError(
                "non-answer Observation 不得携 reference input")
        return PATH_RESPONSE_ACT
    raise GenerationGeneralizationEvaluationRunnerError(
        "evaluation response act 未注册")


def _expected_requirements(
        path: str,
        observation: GenerationGeneralizationEvaluationObservation,
        ) -> tuple[str, ...]:
    """返回一条 actual path 必须形成的独立 requirement。"""
    key = (path, observation.question.answer_plan.response_act)
    try:
        return _PATH_REQUIREMENTS[key]
    except KeyError as error:
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation path/response act 组合未注册") from error


# object-model: protocol; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationEvaluationPolicy:
    """冻结 actual runner 的 pattern 与 reference 选择规则。"""

    pattern_selection: str = "LOWEST_PATTERN_ID"
    reference_strategy: str = "ANTECEDENT_REFERENCE"
    citation_required: bool = True
    trust_required: bool = True

    def __post_init__(self) -> None:
        if self.pattern_selection != "LOWEST_PATTERN_ID":
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation pattern policy 未注册")
        if self.reference_strategy not in {
                "ANTECEDENT_REFERENCE", "EXPLICIT_REPETITION"}:
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation reference policy 未注册")
        if (type(self.citation_required) is not bool
                or type(self.trust_required) is not bool
                or not (self.citation_required or self.trust_required)):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation source policy 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含路径和 Observation 的固定 policy identity。"""
        return (
            1,
            1 if self.reference_strategy == "ANTECEDENT_REFERENCE" else 2,
            int(self.citation_required),
            int(self.trust_required),
        )

    def to_dict(self) -> dict[str, object]:
        """导出 family freeze 可直接内容锁的完整显式 policy。"""
        return {
            "citation_required": int(self.citation_required),
            "pattern_selection": self.pattern_selection,
            "reference_strategy": self.reference_strategy,
            "trust_required": int(self.trust_required),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationEvaluationRequirementEvidence:
    """一项 held-out requirement 的 actual choice/Use 与只读结果。"""

    requirement: str
    observation_key: tuple[int, ...]
    choice: GenerationChoiceHypothesis
    use: GenerationChoiceUseRef
    verification_reports: tuple[VerificationReport, ...]
    verification: GenerationGeneralizationIndependentVerification

    def __post_init__(self) -> None:
        if self.requirement not in INDEPENDENT_VERIFIER_REQUIREMENTS:
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation requirement 未注册")
        if (not isinstance(self.observation_key, tuple)
                or not self.observation_key
                or any(type(item) is not int for item in self.observation_key)):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation Observation key 非法")
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("evaluation requirement choice 类型错误")
        if (not isinstance(self.use, GenerationChoiceUseRef)
                or self.use not in self.choice.exact_uses):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation requirement Use 未回填 exact choice")
        if (not isinstance(self.verification_reports, tuple)
                or not self.verification_reports
                or any(not isinstance(item, VerificationReport)
                       or not item.read_only
                       for item in self.verification_reports)):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation requirement reports 必须为非空只读 tuple")
        if (not isinstance(
                self.verification,
                GenerationGeneralizationIndependentVerification)
                or self.verification.requirement != self.requirement
                or not any(
                    self.verification.result in report.results
                    for report in self.verification_reports)):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation requirement verification/report 漂移")

    @property
    def status(self) -> str:
        """返回本 requirement 的 PASS/FAIL/NE。"""
        result = self.verification.result
        if (result.applicability == APPLICABILITY_APPLICABLE
                and result.verdict == VERDICT_REFUTE):
            return "FAIL"
        if (result.operational_failure is not None
                or result.applicability != APPLICABILITY_APPLICABLE
                or result.verdict != VERDICT_SUPPORT):
            return "NE"
        return "PASS"

    def stable_key(self) -> tuple[int, ...]:
        """返回 Observation、choice/Use 与 requirement result 引用。"""
        return (
            INDEPENDENT_VERIFIER_REQUIREMENTS.index(self.requirement) + 1,
            *_pack(self.observation_key),
            *_pack(self.choice.stable_key()),
            *_pack(self.use.stable_key()),
            *_pack(self.verification.stable_key()),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationEvaluationActualRun:
    """单条 label-free Observation 的 actual generation 与 requirement 集。"""

    observation: GenerationGeneralizationEvaluationObservation
    candidate_pack_sha256: str
    policy: GenerationGeneralizationEvaluationPolicy
    path: str
    evaluation_owner_key: tuple[int, ...]
    selection_key: tuple[int, ...]
    execution: TypedGenerationExecution
    parse_request: GenerationSurfaceParseRequest
    postcheck: GenerationPostcheckRun
    surface_text: str
    requirements: tuple[
        GenerationGeneralizationEvaluationRequirementEvidence, ...]
    verifier_dimension_count: int
    teacher_call_count: int = 0
    label_read_count: int = 0
    host_learning_write_count: int = 0
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(
                self.observation,
                GenerationGeneralizationEvaluationObservation):
            raise TypeError("evaluation actual Observation 类型错误")
        _sha_key(self.candidate_pack_sha256)
        if not isinstance(
                self.policy, GenerationGeneralizationEvaluationPolicy):
            raise TypeError("evaluation actual policy 类型错误")
        if self.path != _path_for(self.observation):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual path 与 Observation 漂移")
        for name, value in (
                ("evaluation owner", self.evaluation_owner_key),
                ("selection", self.selection_key)):
            if (not isinstance(value, tuple) or not value
                    or any(type(item) is not int for item in value)):
                raise GenerationGeneralizationEvaluationRunnerError(
                    f"{name} key 非法")
        if (not isinstance(self.execution, TypedGenerationExecution)
                or not self.execution.complete):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual execution 未完成")
        if self.parse_request != GenerationSurfaceParseRequest.from_execution(
                self.execution):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual parse request 漂移")
        if (not isinstance(self.postcheck, GenerationPostcheckRun)
                or self.postcheck.request.execution != self.execution
                or self.postcheck.report not in {
                    report
                    for evidence in self.requirements
                    for report in evidence.verification_reports
                }):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual G-04/report 漂移")
        if not isinstance(self.surface_text, str) or not self.surface_text:
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual surface 为空")
        expected = _expected_requirements(self.path, self.observation)
        if (not isinstance(self.requirements, tuple)
                or tuple(item.requirement for item in self.requirements)
                != tuple(
                    requirement for requirement
                    in INDEPENDENT_VERIFIER_REQUIREMENTS
                    if requirement in expected)
                or any(item.observation_key != self.observation.stable_key()
                       for item in self.requirements)):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual requirement 覆盖或顺序漂移")
        if len({item.verification.input_key
                for item in self.requirements}) != len(self.requirements):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual requirement input 发生广播")
        routes = tuple(
            (item.verification.result.dimension,
             item.verification.result.verifier)
            for item in self.requirements)
        if len(set(routes)) != len(routes):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual verifier route 发生广播")
        budget = self.observation.resource_budget
        if (len(self.execution.representations) > budget.max_surface_units
                or type(self.verifier_dimension_count) is not int
                or self.verifier_dimension_count <= 0
                or self.verifier_dimension_count
                > budget.max_verifier_dimensions):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual 超出 surface/verifier 资源上限")
        for name in (
                "teacher_call_count", "label_read_count",
                "host_learning_write_count"):
            if getattr(self, name) != 0:
                raise GenerationGeneralizationEvaluationRunnerError(
                    f"evaluation actual {name} 必须为零")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    @property
    def runtime_status(self) -> str:
        """按本 path 的 hard conjunction 返回 PASS/FAIL/NE。"""
        statuses = tuple(item.status for item in self.requirements)
        if "FAIL" in statuses:
            return "FAIL_EVALUATION_ACTUAL_CONJUNCTION"
        if len(self.requirements) != len(
                _expected_requirements(self.path, self.observation)):
            return "NE_EVALUATION_ACTUAL_INPUT_MISSING"
        if (not self.postcheck.complete
                or "NE" in statuses):
            return "NE_EVALUATION_ACTUAL_INPUT_INDETERMINATE"
        return "PASS_EVALUATION_ACTUAL_CONJUNCTION"

    def stable_key(self) -> tuple[int, ...]:
        """返回 pack/policy/Observation、actual output 与审计状态。"""
        if not self._stable_key_cache:
            raise RuntimeError("evaluation actual stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """在构造完成时保存路径无关、地址无关的内容引用。"""
        values = [
            EVALUATION_PATHS.index(self.path) + 1,
            *_pack(_sha_key(self.candidate_pack_sha256)),
            *_pack(self.policy.stable_key()),
            *_pack(self.observation.stable_key()),
            *_pack(self.evaluation_owner_key),
            *_pack(self.selection_key),
            *_pack(self.execution.stable_key()),
            *_pack(self.parse_request.stable_key()),
            *_pack(self.postcheck.stable_key()),
            *_pack(_text_key(self.surface_text)),
            self.verifier_dimension_count,
            len(self.requirements),
        ]
        for item in self.requirements:
            values.extend(_pack(item.stable_key()))
        values.extend((
            self.teacher_call_count,
            self.label_read_count,
            self.host_learning_write_count,
            EVALUATION_ACTUAL_STATUSES.index(self.runtime_status) + 1,
        ))
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationEvaluationBatch:
    """跨 Observation 汇总六路 actual evidence，保持 route/input 不广播。"""

    runs: tuple[GenerationGeneralizationEvaluationActualRun, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.runs, tuple)
                or any(not isinstance(
                    item, GenerationGeneralizationEvaluationActualRun)
                    for item in self.runs)):
            raise TypeError("evaluation batch runs 类型错误")
        observations = tuple(
            item.observation.stable_key() for item in self.runs)
        owners = tuple(item.evaluation_owner_key for item in self.runs)
        if (len(set(observations)) != len(observations)
                or len(set(owners)) != len(owners)):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation batch 复用 Observation 或 evaluation owner")
        evidence = self.evidence
        inputs = tuple(item.verification.input_key for item in evidence)
        routes = tuple(
            (item.verification.result.dimension,
             item.verification.result.verifier)
            for item in evidence)
        if (len(set(inputs)) != len(inputs)
                or len(set(routes)) != len(routes)):
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation batch requirement input/route 发生广播")

    @property
    def evidence(
            self,
            ) -> tuple[GenerationGeneralizationEvaluationRequirementEvidence, ...]:
        """按冻结 requirement 顺序汇总全部 actual evidence。"""
        return tuple(
            item
            for requirement in INDEPENDENT_VERIFIER_REQUIREMENTS
            for run in self.runs
            for item in run.requirements
            if item.requirement == requirement
        )

    def requirement_status(self, requirement: str) -> str:
        """按一项 requirement 的全部 Observation 返回 PASS/FAIL/NE。"""
        if requirement not in INDEPENDENT_VERIFIER_REQUIREMENTS:
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation batch requirement 未注册")
        items = tuple(
            item for item in self.evidence
            if item.requirement == requirement)
        if not items:
            return "NE"
        statuses = {item.status for item in items}
        if "FAIL" in statuses:
            return "FAIL"
        if statuses != {"PASS"}:
            return "NE"
        return "PASS"

    @property
    def status(self) -> str:
        """按六路 hard conjunction 返回 batch PASS/FAIL/NE。"""
        statuses = tuple(
            self.requirement_status(item)
            for item in INDEPENDENT_VERIFIER_REQUIREMENTS)
        if "FAIL" in statuses:
            return "FAIL"
        if "NE" in statuses:
            return "NE"
        return "PASS"

    @property
    def coverage(self) -> tuple[str, ...]:
        """返回至少形成一条 actual evidence 的 requirement。"""
        present = {item.requirement for item in self.evidence}
        return tuple(
            item for item in INDEPENDENT_VERIFIER_REQUIREMENTS
            if item in present)

    def stable_key(self) -> tuple[int, ...]:
        """返回 run 与六路状态的确定性批次键。"""
        values = [len(self.runs)]
        for run in self.runs:
            values.extend(_pack(run.stable_key()))
        values.append(("FAIL", "NE", "PASS").index(self.status) + 1)
        for requirement in INDEPENDENT_VERIFIER_REQUIREMENTS:
            values.append(("FAIL", "NE", "PASS").index(
                self.requirement_status(requirement)) + 1)
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _RuntimeProtocols:
    """一次 actual run 独占的全部开放协议身份。"""

    content: AnswerContentProtocol
    selector: AnswerContentSelector
    plan: GenerationPlanProtocol
    structure: GenerationStructureLayerProtocol
    surface: GenerationSurfaceProtocol
    postcheck: GenerationPostcheckProtocol
    question: QuestionAnswerProtocol
    renderer_identity: ObjectIdentity
    query_kind: ObjectIdentity
    route: ObjectIdentity
    execution_reason: ObjectIdentity


def _runtime_protocols(run_id: int) -> _RuntimeProtocols:
    """构造互异、确定且仅属于当前 Observation 的协议身份。"""
    content = AnswerContentProtocol(*tuple(
        _instruction(run_id, 1, index) for index in range(1, 6)))
    answer_policy = EvidenceAnswerPolicyProtocol(*tuple(
        _instruction(run_id, 2, index) for index in range(1, 5)))
    selector = AnswerContentSelector(
        content, EvidenceAnswerPolicy(content, answer_policy))
    plan = GenerationPlanProtocol(*tuple(
        _instruction(run_id, 3, index) for index in range(1, 11)))
    structure = GenerationStructureLayerProtocol(*tuple(
        _instruction(run_id, 4, index) for index in range(1, 4)))
    surface = GenerationSurfaceProtocol(*tuple(
        _instruction(run_id, 5, index) for index in range(1, 10)))
    postcheck = GenerationPostcheckProtocol(
        *tuple(_protocol_key(run_id, 6, index) for index in range(1, 13)),
        *tuple(_instruction(run_id, 7, index) for index in range(1, 16)),
    )
    question = QuestionAnswerProtocol(*tuple(
        _instruction(run_id, 8, index) for index in range(1, 4)))
    return _RuntimeProtocols(
        content,
        selector,
        plan,
        structure,
        surface,
        postcheck,
        question,
        _instruction(run_id, 9, 1),
        _instruction(run_id, 10, 1),
        _instruction(run_id, 10, 2),
        _instruction(run_id, 10, 3),
    )


def _lifecycle(ctx: TrainContext, run_id: int) -> StructureOrderLifecycleGraph:
    """在 evaluation clone 中建立一次 run 独占的 S-07 owner。"""
    ontology = ctx.graph_ontology
    identities = tuple(
        concept_identity((_NAMESPACE, run_id, 20, index))
        for index in range(1, 26))
    refs = tuple(ontology.materialize(item) for item in identities)
    graph = StructureOrderGraph(
        ontology, StructureOrderGraphPredicates(*refs[:19]))
    states_and_kinds = tuple(
        concept_identity((_NAMESPACE, run_id, 21, index))
        for index in range(1, 7))
    for identity in states_and_kinds:
        ontology.materialize(identity)
    protocol = StructureOrderLifecycleProtocol(
        *refs[19:],
        *states_and_kinds,
        (_NAMESPACE, run_id, 22, 1),
    )
    return StructureOrderLifecycleGraph(graph, protocol)


def _answer_protocol(
        run_id: int,
        ) -> GenerationGeneralizationAnswerVerificationProtocol:
    """构造两条 ANSWER requirement 的独立 route。"""
    return GenerationGeneralizationAnswerVerificationProtocol(
        *tuple(_protocol_key(run_id, 30, index) for index in range(1, 5)),
        *tuple(_instruction(run_id, 31, index) for index in range(1, 5)),
    )


def _source_conflict_protocol(
        run_id: int,
        ) -> GenerationGeneralizationSourceConflictProtocol:
    """构造 source-conflict 的独立 route。"""
    return GenerationGeneralizationSourceConflictProtocol(
        _protocol_key(run_id, 32, 1),
        _protocol_key(run_id, 32, 2),
        _instruction(run_id, 33, 1),
        _instruction(run_id, 33, 2),
    )


def _reference_protocol(
        run_id: int,
        ) -> GroundedAnswerReferenceVerifierProtocol:
    """构造十层 reference verifier 的独立公开协议。"""
    return build_grounded_answer_reference_verifier_protocol(
        (_NAMESPACE, run_id, 34))


def _question_request(
        protocols: _RuntimeProtocols,
        planning: GroundedResponseActPlanningBuild,
        branch: ObjectIdentity,
        run_id: int,
        ) -> QuestionRequest:
    """把同次 planning 转成只授权当前 candidates 的 typed QuestionRequest。"""
    request = planning.planning
    return QuestionRequest(
        protocols.query_kind,
        _instruction(run_id, 11, 1),
        request.goal.goal_kind,
        request.goal.proposition,
        request.goal.required,
        request.goal.scope,
        request.goal.scope,
        (_NAMESPACE, run_id, 11, 2),
        branch,
        tuple(item.proposition for item in request.candidates),
    )


def _run_id(
        loaded: LoadedGenerationCandidatePack,
        observation: GenerationGeneralizationEvaluationObservation,
        policy: GenerationGeneralizationEvaluationPolicy,
        ) -> int:
    """从 pack、policy 与完整 label-free Observation 建立 run identity。"""
    return _positive_id(
        (
            *_sha_key(loaded.pack.sha256()),
            *_pack(policy.stable_key()),
            *_pack(observation.stable_key()),
        ),
        domain="gg03.evaluation.actual.run.v1",
    )


def _require_complete_run(run: QuestionAnswerRun) -> None:
    """拒绝把 unsupported 或部分执行伪装为 actual evidence。"""
    if (not isinstance(run, QuestionAnswerRun)
            or run.generation is None
            or run.postcheck is None
            or not run.generation.complete):
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation path 未形成完整 G-00 至 G-04 run")


def _requirement_evidence(
        requirement: str,
        observation: GenerationGeneralizationEvaluationObservation,
        choice: GenerationChoiceHypothesis,
        use: GenerationChoiceUseRef,
        reports: tuple[VerificationReport, ...],
        result: VerificationResult,
        ) -> GenerationGeneralizationEvaluationRequirementEvidence:
    """把独立 result 绑定到 exact input claim、choice/Use 和 Observation。"""
    if not result.claim_keys:
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation verifier 缺 actual input claim")
    verification = GenerationGeneralizationIndependentVerification(
        requirement,
        LosslessIntegerKey(result.claim_keys[0]),
        result,
    )
    return GenerationGeneralizationEvaluationRequirementEvidence(
        requirement,
        observation.stable_key(),
        choice,
        use,
        reports,
        verification,
    )


def _verifier_dimension_count(
        evidence: tuple[
            GenerationGeneralizationEvaluationRequirementEvidence, ...],
        ) -> int:
    """统计 G-04 和 requirement reports 的实际互异 route 数。"""
    return len({
        (result.dimension, result.verifier)
        for item in evidence
        for report in item.verification_reports
        for result in report.results
    })


def _answer_actual(
        ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observation: GenerationGeneralizationEvaluationObservation,
        policy: GenerationGeneralizationEvaluationPolicy,
        run_id: int,
        protocols: _RuntimeProtocols,
        branch: ObjectIdentity,
        lifecycle: StructureOrderLifecycleGraph,
        renderer: UnicodeRepresentationRenderer,
        representation_family: tuple[int, ...],
        ) -> tuple[
            QuestionAnswerRun,
            tuple[GenerationGeneralizationEvaluationRequirementEvidence, ...],
            tuple[int, ...],
        ]:
    """运行单命题 ANSWER，并形成 readback/legal 两项 evidence。"""
    planning = compile_grounded_answer_planning(observation, branch)
    candidate = planning.planning.candidates[0]
    target = GroundedAnswerConnectorTarget(
        candidate.proposition,
        branch,
        representation_family,
    )
    compilation = compile_grounded_answer_connectors(
        loaded.pack.model, observation.question, target, protocols.surface)
    selected = min(
        compilation.variants, key=lambda item: item.option.pattern_id)
    production_purpose = _instruction(run_id, 40, 1)
    alias_factory = ProductionGenerationAliasRuntimeFactory(
        loaded.pack,
        ctx,
        visible_evidence_keys=_visible_evidence_keys(planning),
        disposable_evaluation=True,
    )
    components = GroundedAnswerRunLocalComponents(
        selector=protocols.selector,
        plan_protocol=protocols.plan,
        structure_protocol=protocols.structure,
        alias_factory=alias_factory,
        renderer=renderer,
        renderer_identity=protocols.renderer_identity,
        postcheck_protocol=protocols.postcheck,
        structure_verifier=GroundedAnswerStructureVerifier(
            _instruction(run_id, 41, 1),
            _instruction(run_id, 41, 2),
        ),
        source_verifier=GroundedAnswerEvidenceSourceVerifier(
            _instruction(run_id, 42, 1),
            _instruction(run_id, 42, 2),
        ),
        question_protocol=protocols.question,
        postcheck_mapper=EvidenceQuestionPostcheckMapper(
            (_NAMESPACE, run_id, 43),
            citation_required=policy.citation_required,
            trust_required=policy.trust_required,
        ),
        surface_attributions=(_surface_attribution(
            selected.template.connector,
            candidate,
            production_purpose,
        ),),
    )
    installation = GroundedAnswerRunLocalFactory(
        protocols.surface, lifecycle, components).build(
            GroundedAnswerRunLocalBuild(
                loaded.pack.model,
                observation.question,
                target,
                planning.planning,
                candidate,
                selected.option.structure_id,
                selected.option.pattern_id,
                GroundedAnswerParserProtocol(
                    *tuple(_instruction(run_id, 44, index)
                           for index in range(1, 6)),
                    protocols.content.answer,
                ),
                protocols.query_kind,
                protocols.route,
                protocols.execution_reason,
                (_NAMESPACE, run_id, 45),
            ))
    run = installation.runtime.run(
        _question_request(protocols, planning, branch, run_id))
    _require_complete_run(run)
    lexical = GroundedAnswerLexicalAdoptionLedger(installation).adopt(run)
    structure = GroundedAnswerStructureAdoptionLedger(installation).adopt(run)
    assert run.generation is not None and run.postcheck is not None
    parse_request = GenerationSurfaceParseRequest.from_execution(
        run.generation)
    verification_protocol = _answer_protocol(run_id)
    evidence = []
    records = {
        "INDEPENDENT_UNDERSTANDING_READBACK": lexical,
        "LEGAL_OBJECT_COMPOSITION": structure,
    }
    for requirement in _PATH_REQUIREMENTS[(PATH_ANSWER, "ANSWER")]:
        record = records[requirement]
        request = GenerationGeneralizationAnswerVerificationInput(
            requirement,
            observation,
            planning,
            record.choice_after,
            record.use,
            run.generation,
            parse_request,
            run.postcheck,
        )
        report = run_generation_generalization_answer_verification(
            verification_protocol, request)
        dimension, verifier = verification_protocol.route(requirement)
        result = _report_result(report, dimension, verifier)
        evidence.append(_requirement_evidence(
            requirement,
            observation,
            record.choice_after,
            record.use,
            (run.postcheck.report, report),
            result,
        ))
    return run, tuple(evidence), (
        *_pack(lexical.choice_after.stable_key()),
        *_pack(structure.choice_after.stable_key()),
    )


def _response_act_actual(
        ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observation: GenerationGeneralizationEvaluationObservation,
        policy: GenerationGeneralizationEvaluationPolicy,
        run_id: int,
        protocols: _RuntimeProtocols,
        branch: ObjectIdentity,
        lifecycle: StructureOrderLifecycleGraph,
        renderer: UnicodeRepresentationRenderer,
        representation_family: tuple[int, ...],
        ) -> tuple[
            QuestionAnswerRun,
            tuple[GenerationGeneralizationEvaluationRequirementEvidence, ...],
            tuple[int, ...],
        ]:
    """运行 CLARIFY/CONFLICT，并形成 task 或 source-conflict evidence。"""
    planning = compile_grounded_response_act_planning(observation, branch)
    response_act = observation.question.answer_plan.response_act
    stance = getattr(protocols.content, response_act.lower())
    target = GroundedResponseActCompileTarget(
        response_act,
        stance,
        branch,
        representation_family,
    )
    compilation = compile_grounded_response_act_patterns(
        loaded.pack.model, target)
    selected = min(compilation.variants, key=lambda item: item.pattern_id)
    production_purpose = _instruction(run_id, 50, 1)
    alias_factory = ProductionGenerationAliasRuntimeFactory(
        loaded.pack, ctx, disposable_evaluation=True)
    components = GroundedResponseActRunLocalComponents(
        selector=protocols.selector,
        plan_protocol=protocols.plan,
        structure_protocol=protocols.structure,
        surface_protocol=protocols.surface,
        alias_factory=alias_factory,
        renderer=renderer,
        renderer_identity=protocols.renderer_identity,
        postcheck_protocol=protocols.postcheck,
        structure_verifier=GroundedResponseActStructureVerifier(
            _instruction(run_id, 51, 1),
            _instruction(run_id, 51, 2),
        ),
        source_verifier=GroundedAnswerEvidenceSourceVerifier(
            _instruction(run_id, 52, 1),
            _instruction(run_id, 52, 2),
        ),
        task_verifier=GroundedResponseActTaskVerifier(
            _instruction(run_id, 53, 1),
            _instruction(run_id, 53, 2),
        ),
        question_protocol=protocols.question,
        surface_attribution=_surface_attribution(
            selected.template.sentence,
            planning.planning.candidates[0],
            production_purpose,
        ),
    )
    installation = GroundedResponseActRunLocalFactory(
        lifecycle, components).build(
            GroundedResponseActRunLocalBuild(
                loaded.pack.model,
                observation.question,
                target,
                planning.planning,
                selected.pattern_id,
                GroundedResponseActParserProtocol(*tuple(
                    _instruction(run_id, 54, index)
                    for index in range(1, 4))),
                protocols.query_kind,
                protocols.route,
                protocols.execution_reason,
                (_NAMESPACE, run_id, 55),
            ))
    run = installation.runtime.run(
        _question_request(protocols, planning, branch, run_id))
    _require_complete_run(run)
    lexical = GroundedResponseActLexicalAdoptionLedger(
        installation).adopt(run)
    assert run.generation is not None and run.postcheck is not None
    parse_request = GenerationSurfaceParseRequest.from_execution(
        run.generation)
    if response_act == "CLARIFY":
        result = _report_result(
            run.postcheck.report,
            run.postcheck.protocol.task_dimension,
            run.postcheck.protocol.task_verifier,
        )
        evidence = (_requirement_evidence(
            "COMMUNICATIVE_TASK",
            observation,
            lexical.choice_after,
            lexical.use,
            (run.postcheck.report,),
            result,
        ),)
    else:
        conflict_protocol = _source_conflict_protocol(run_id)
        conflict_input = GenerationGeneralizationSourceConflictInput(
            observation,
            planning,
            lexical.choice_after,
            lexical.use,
            run.generation,
            parse_request,
            run.postcheck,
        )
        report = run_generation_generalization_source_conflict_verification(
            conflict_protocol, conflict_input)
        result = _report_result(
            report, conflict_protocol.dimension, conflict_protocol.verifier)
        evidence = (_requirement_evidence(
            "SOURCE_UNCERTAINTY_CITATION",
            observation,
            lexical.choice_after,
            lexical.use,
            (run.postcheck.report, report),
            result,
        ),)
    return run, evidence, lexical.choice_after.stable_key()


def _reference_actual(
        ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observation: GenerationGeneralizationEvaluationObservation,
        policy: GenerationGeneralizationEvaluationPolicy,
        run_id: int,
        protocols: _RuntimeProtocols,
        branch: ObjectIdentity,
        lifecycle: StructureOrderLifecycleGraph,
        renderer: UnicodeRepresentationRenderer,
        representation_family: tuple[int, ...],
        ) -> tuple[
            QuestionAnswerRun,
            tuple[GenerationGeneralizationEvaluationRequirementEvidence, ...],
            tuple[int, ...],
        ]:
    """运行双命题 reference，并形成 recoverability/structure evidence。"""
    planning = compile_grounded_answer_reference_planning(
        observation, branch)
    reference = observation.reference_course
    if reference is None:
        raise GenerationGeneralizationEvaluationRunnerError(
            "reference path 缺 label-free reference input")
    claims = tuple(
        GroundedAnswerClaimCandidateBinding(
            proposition_id,
            planning.candidate_for(proposition_id),
        )
        for proposition_id in reference.ordered_proposition_ids
    )
    forming = _visible_evidence_keys(planning)
    production_purpose = _instruction(run_id, 60, 1)
    compiled = tuple(
        compile_grounded_answer_reference_connector(
            GroundedAnswerReferenceCompileRequest(
                observation,
                planning.planning,
                claims,
                branch,
                representation_family,
                strategy,
                forming,
            ),
            protocols.surface,
        )
        for strategy in loaded.pack.reference_strategies
    )
    compilations = tuple(
        replace(
            compilation,
            connector=LanguageGenerationConnector(
                compilation.connector.registry,
                compilation.connector.runtime_policy,
                compilation.connector.surface_protocol,
                tuple(
                    _surface_attribution(
                        sentence.template.connector,
                        sentence.candidate,
                        production_purpose,
                    )
                    for sentence in compilation.sentences
                ),
                compilation.connector.discourse_declarations,
                compilation.connector.anaphora_declarations,
            ),
        )
        for compilation in compiled
    )
    selection = build_grounded_answer_reference_selection(
        compilations,
        policy.reference_strategy,
        (_NAMESPACE, run_id, 61),
    )
    alias_factory = ProductionGenerationAliasRuntimeFactory(
        loaded.pack, ctx, disposable_evaluation=True)
    components = GroundedAnswerRunLocalComponents(
        selector=protocols.selector,
        plan_protocol=protocols.plan,
        structure_protocol=protocols.structure,
        alias_factory=alias_factory,
        renderer=renderer,
        renderer_identity=protocols.renderer_identity,
        postcheck_protocol=protocols.postcheck,
        structure_verifier=GroundedAnswerReferenceStructureVerifier(
            _instruction(run_id, 62, 1),
            _instruction(run_id, 62, 2),
        ),
        source_verifier=GroundedAnswerReferenceEvidenceSourceVerifier(
            _instruction(run_id, 63, 1),
            _instruction(run_id, 63, 2),
        ),
        question_protocol=protocols.question,
        postcheck_mapper=EvidenceQuestionPostcheckMapper(
            (_NAMESPACE, run_id, 64),
            citation_required=policy.citation_required,
            trust_required=policy.trust_required,
        ),
    )
    installation = GroundedAnswerReferenceRunLocalFactory(
        lifecycle, components).build(
            GroundedAnswerReferenceRunLocalBuild(
                selection.compilation,
                selection,
                GroundedAnswerParserProtocol(
                    *tuple(_instruction(run_id, 65, index)
                           for index in range(1, 6)),
                    protocols.content.answer,
                ),
                protocols.query_kind,
                protocols.route,
                protocols.execution_reason,
                (_NAMESPACE, run_id, 66),
            ))
    run = installation.runtime.run(
        _question_request(protocols, planning, branch, run_id))
    _require_complete_run(run)
    uses = GroundedAnswerReferenceEpisodeAdoptionLedger(
        installation).adopt(run)
    verification_protocol = _reference_protocol(run_id)
    layered = verify_grounded_answer_reference_layers(
        verification_protocol, installation, run, uses)
    route_by_requirement = {
        "ADDRESSEE_RECOVERABILITY": (
            "REFERENCE_UNIQUE_RESOLUTION", uses.reference),
        "STRUCTURE_SLOT_ORDER": (
            "STRUCTURE_EXECUTION", uses.structure),
    }
    evidence = []
    for requirement in _PATH_REQUIREMENTS[(PATH_REFERENCE, "ANSWER")]:
        route_name, record = route_by_requirement[requirement]
        route = verification_protocol.by_name()[route_name].route
        result = _report_result(
            layered.report, route.dimension, route.verifier)
        evidence.append(_requirement_evidence(
            requirement,
            observation,
            record.choice_after,
            record.use,
            (run.postcheck.report, layered.report),
            result,
        ))
    return run, tuple(evidence), selection.stable_key()


def run_generation_generalization_evaluation_actual(
        host_ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observation: GenerationGeneralizationEvaluationObservation,
        policy: GenerationGeneralizationEvaluationPolicy | None = None,
        ) -> GenerationGeneralizationEvaluationActualRun:
    """在独立 evaluation clone 中运行一条 label-free Observation。"""
    if not isinstance(host_ctx, TrainContext):
        raise TypeError("evaluation actual host context 类型错误")
    if not isinstance(loaded, LoadedGenerationCandidatePack):
        raise TypeError("evaluation actual loaded candidate pack 类型错误")
    if not isinstance(
            observation, GenerationGeneralizationEvaluationObservation):
        raise TypeError("evaluation actual Observation 类型错误")
    policy = policy or GenerationGeneralizationEvaluationPolicy()
    if not isinstance(policy, GenerationGeneralizationEvaluationPolicy):
        raise TypeError("evaluation actual policy 类型错误")
    if policy.reference_strategy not in loaded.pack.reference_strategies:
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation policy reference strategy 不属于 candidate pack")
    path = _path_for(observation)
    run_id = _run_id(loaded, observation, policy)
    label = f"gg03-e05-{run_id}"
    result = None
    with isolated_evaluation(host_ctx, label=label) as eval_ctx:
        before_teacher = _teacher_calls(eval_ctx.teacher)
        protocols = _runtime_protocols(run_id)
        branch = language_branch_identity((_NAMESPACE, run_id, 70))
        representation_family = (_NAMESPACE, run_id, 71)
        renderer = UnicodeRepresentationRenderer(
            representation_family, protocols.renderer_identity)
        lifecycle = _lifecycle(eval_ctx, run_id)
        if path == PATH_ANSWER:
            run, evidence, selection_key = _answer_actual(
                eval_ctx,
                loaded,
                observation,
                policy,
                run_id,
                protocols,
                branch,
                lifecycle,
                renderer,
                representation_family,
            )
        elif path == PATH_RESPONSE_ACT:
            run, evidence, selection_key = _response_act_actual(
                eval_ctx,
                loaded,
                observation,
                policy,
                run_id,
                protocols,
                branch,
                lifecycle,
                renderer,
                representation_family,
            )
        else:
            run, evidence, selection_key = _reference_actual(
                eval_ctx,
                loaded,
                observation,
                policy,
                run_id,
                protocols,
                branch,
                lifecycle,
                renderer,
                representation_family,
            )
        after_teacher = _teacher_calls(eval_ctx.teacher)
        if after_teacher != before_teacher:
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual 调用了 teacher/LLM")
        _require_complete_run(run)
        assert run.generation is not None and run.postcheck is not None
        owner = eval_ctx.scope_owner
        if owner is None:
            raise GenerationGeneralizationEvaluationRunnerError(
                "evaluation actual clone 缺独立 owner")
        result = GenerationGeneralizationEvaluationActualRun(
            observation,
            loaded.pack.sha256(),
            policy,
            path,
            owner.stable_key(),
            selection_key,
            run.generation,
            GenerationSurfaceParseRequest.from_execution(run.generation),
            run.postcheck,
            renderer.text(run.generation.rendered),
            evidence,
            _verifier_dimension_count(evidence),
        )
    if result is None:
        raise GenerationGeneralizationEvaluationRunnerError(
            "evaluation actual 未形成结果")
    return result


def run_generation_generalization_evaluation_batch(
        host_ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observations: tuple[
            GenerationGeneralizationEvaluationObservation, ...],
        policy: GenerationGeneralizationEvaluationPolicy | None = None,
        ) -> GenerationGeneralizationEvaluationBatch:
    """逐 Observation 建独立 clone，并汇总六路 actual PASS/FAIL/NE。"""
    if (not isinstance(observations, tuple)
            or any(not isinstance(
                item, GenerationGeneralizationEvaluationObservation)
                for item in observations)):
        raise TypeError("evaluation batch observations 类型错误")
    policy = policy or GenerationGeneralizationEvaluationPolicy()
    runs = []
    for ordinal, observation in enumerate(observations, start=1):
        path = _path_for(observation)
        try:
            runs.append(run_generation_generalization_evaluation_actual(
                host_ctx, loaded, observation, policy))
        except Exception as error:
            raise GenerationGeneralizationEvaluationBatchRunError(
                ordinal, path) from error
    return GenerationGeneralizationEvaluationBatch(tuple(runs))


def generation_generalization_evaluation_requirements(
        observation: GenerationGeneralizationEvaluationObservation,
        ) -> tuple[str, ...]:
    """返回单条 label-free Observation 将实际执行的冻结 requirement 集。"""
    if not isinstance(
            observation, GenerationGeneralizationEvaluationObservation):
        raise TypeError("evaluation requirement Observation 类型错误")
    path = _path_for(observation)
    expected = _expected_requirements(path, observation)
    return tuple(
        requirement for requirement in INDEPENDENT_VERIFIER_REQUIREMENTS
        if requirement in expected)


__all__ = [
    "EVALUATION_ACTUAL_STATUSES",
    "EVALUATION_PATHS",
    "GenerationGeneralizationEvaluationActualRun",
    "GenerationGeneralizationEvaluationBatch",
    "GenerationGeneralizationEvaluationBatchRunError",
    "GenerationGeneralizationEvaluationPolicy",
    "GenerationGeneralizationEvaluationRequirementEvidence",
    "GenerationGeneralizationEvaluationRunnerError",
    "PATH_ANSWER",
    "PATH_REFERENCE",
    "PATH_RESPONSE_ACT",
    "generation_generalization_evaluation_requirements",
    "run_generation_generalization_evaluation_actual",
    "run_generation_generalization_evaluation_batch",
]
