"""GG-03 H-05 assessment、held-out selection 与 revision 运行时。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateEvidenceRevisionOutcome,
    CandidateLearningOutcome,
    CandidateLearningRuntime,
    CandidateRecognitionRequest,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_authored_generation_generalization_course import (
    AuthoredGenerationGeneralizationSeed,
)
from pure_integer_ai.experiments.ph2_generation_choice_assessment_selector import (
    GenerationChoiceAssessmentSelection,
    GenerationChoiceAssessmentSelectorPolicy,
    select_generation_choice_by_assessment,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceCandidateMapper,
)
from pure_integer_ai.experiments.ph2_generation_generalization_choice_projection import (
    GG03_DIMENSION_CHOICE_KIND,
    GG03_EVALUATOR_OWNER,
    GG03_RUNTIME_VERSIONS,
    GG03_TRAINING_OWNER,
    GenerationGeneralizationAssessmentCase,
    GenerationGeneralizationAssessmentRuntimeError,
    GenerationGeneralizationSurfaceOption,
    project_generation_generalization_observation,
    project_generation_generalization_seed,
    strict_runtime_key,
)


GG03_HELD_OUT_RUNTIME_STATUSES = (
    "NE_INDEPENDENT_LAYER_INPUT_MISSING",
    "PASS",
)


def _stance(expected_state: str) -> int:
    """把冻结课程 verdict 映射为 H-05 三态 Evidence。"""
    if expected_state == "TRUE":
        return EVIDENCE_SUPPORT
    if expected_state == "FALSE":
        return EVIDENCE_REFUTE
    if expected_state in {"UNKNOWN", "CONFLICT"}:
        return EVIDENCE_UNKNOWN
    raise GenerationGeneralizationAssessmentRuntimeError(
        "GG-03 challenge verdict 未注册")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationAssessmentRuntimePolicy:
    """冻结 event/selector 域和小课程资源上限。"""

    event_namespace: tuple[int, ...]
    selector_policy: GenerationChoiceAssessmentSelectorPolicy
    max_training_cases: int = 14
    max_held_out_cases: int = 14
    max_options_per_case: int = 3

    def __post_init__(self) -> None:
        strict_runtime_key(
            self.event_namespace, where="GG-03 runtime event namespace")
        if not isinstance(
                self.selector_policy, GenerationChoiceAssessmentSelectorPolicy):
            raise TypeError("GG-03 runtime selector policy 类型错误")
        assert_int(
            self.max_training_cases,
            self.max_held_out_cases,
            self.max_options_per_case,
            _where="GG-03 runtime resource limits",
        )
        if (type(self.max_training_cases) is not int
                or type(self.max_held_out_cases) is not int
                or type(self.max_options_per_case) is not int
                or self.max_training_cases <= 0
                or self.max_held_out_cases <= 0
                or self.max_options_per_case != 3):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime resource limits 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationTrainingRecord:
    """一个 TRAIN challenge 的 H-05 update 与随后 selection。"""

    case: GenerationGeneralizationAssessmentCase
    stance: int
    learning: CandidateLearningOutcome
    selection: GenerationChoiceAssessmentSelection

    def __post_init__(self) -> None:
        if self.case.split != "train":
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 training record 混入 held-out")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 training stance 非法")
        if not isinstance(self.learning, CandidateLearningOutcome):
            raise TypeError("GG-03 training learning 类型错误")
        if not isinstance(self.selection, GenerationChoiceAssessmentSelection):
            raise TypeError("GG-03 training selection 类型错误")
        if (self.learning.prediction.predicted
                != self.case.challenge.choice.selected_object
                or self.learning.verification.stance != self.stance
                or self.selection.options != self.case.choices):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 training record 与 challenge/selection 漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationTrainingReport:
    """TRAIN-only candidate/update/selection 计数。"""

    records: tuple[GenerationGeneralizationTrainingRecord, ...]
    candidate_registrations: int
    assessment_updates: int
    teacher_call_count: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.records, tuple) or not self.records
                or any(not isinstance(item, GenerationGeneralizationTrainingRecord)
                       for item in self.records)):
            raise TypeError("GG-03 training report records 类型错误")
        for value, label in (
                (self.candidate_registrations, "registrations"),
                (self.assessment_updates, "updates"),
                (self.teacher_call_count, "teacher calls")):
            assert_int(value, _where=f"GG-03 training {label}")
            if type(value) is not int or value < 0:
                raise GenerationGeneralizationAssessmentRuntimeError(
                    f"GG-03 training {label} 非法")
        if (self.candidate_registrations != len(self.records)
                or self.assessment_updates != len(self.records)
                or self.teacher_call_count != 0):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 training 计数或 teacher 边界漂移")

    @property
    def changed_selections(self) -> int:
        """统计 assessment 后偏离显式 baseline 的记录数。"""
        return sum(
            item.selection.selected != item.case.baseline.choice
            for item in self.records)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationHeldOutRecord:
    """先 selection、后 label set comparison 的只读 held-out 记录。"""

    case: GenerationGeneralizationAssessmentCase
    selection: GenerationChoiceAssessmentSelection
    selected_surface_candidate_id: str
    accepted_surface_candidate_ids: tuple[str, ...]
    legal_selection: int
    runtime_status: str

    def __post_init__(self) -> None:
        if self.case.split != "held_out":
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out record 混入 TRAIN")
        if not isinstance(self.selection, GenerationChoiceAssessmentSelection):
            raise TypeError("GG-03 held-out selection 类型错误")
        expected_id = self.case.option_for_choice(
            self.selection.selected).surface_candidate_id
        if self.selected_surface_candidate_id != expected_id:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out selected surface 漂移")
        if (not isinstance(self.accepted_surface_candidate_ids, tuple)
                or len(self.accepted_surface_candidate_ids) < 2
                or len(set(self.accepted_surface_candidate_ids)) != len(
                    self.accepted_surface_candidate_ids)):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out accepted set 非法")
        if self.legal_selection not in (0, 1):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out legal flag 非法")
        if self.legal_selection != int(
                self.selected_surface_candidate_id
                in self.accepted_surface_candidate_ids):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out set comparison 漂移")
        if self.runtime_status not in GG03_HELD_OUT_RUNTIME_STATUSES:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out runtime status 未注册")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationHeldOutReport:
    """14 个未见组合的零写选择与诚实 runtime NE。"""

    records: tuple[GenerationGeneralizationHeldOutRecord, ...]
    host_write_count: int
    teacher_call_count: int
    runtime_status: str

    def __post_init__(self) -> None:
        if (not isinstance(self.records, tuple) or not self.records
                or any(not isinstance(item, GenerationGeneralizationHeldOutRecord)
                       for item in self.records)):
            raise TypeError("GG-03 held-out report records 类型错误")
        if self.host_write_count != 0 or self.teacher_call_count != 0:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out 必须 host/teacher 零写")
        if self.runtime_status != "NE_INDEPENDENT_LAYER_INPUT_MISSING":
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 held-out 当前不得冒充 runtime PASS")

    @property
    def legal_selection_count(self) -> int:
        """统计 selected surface id 落入 evaluator accepted set 的记录数。"""
        return sum(item.legal_selection for item in self.records)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationAssessmentRevision:
    """一次 append-only rollback/reverify 及其新 selection。"""

    case: GenerationGeneralizationAssessmentCase
    prior_evidence_id: int
    stance: int
    outcome: CandidateEvidenceRevisionOutcome
    selection: GenerationChoiceAssessmentSelection

    def __post_init__(self) -> None:
        if type(self.prior_evidence_id) is not int or self.prior_evidence_id <= 0:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 revision prior evidence id 非法")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 revision stance 非法")
        if not isinstance(self.outcome, CandidateEvidenceRevisionOutcome):
            raise TypeError("GG-03 revision outcome 类型错误")
        if (self.outcome.evidence.supersedes_evidence_id
                != self.prior_evidence_id
                or self.outcome.evidence.stance != self.stance
                or self.selection.options != self.case.choices):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 revision 与 prior/selection 漂移")


def _revealed(
        verifier_source: SourceRef,
        event_key: tuple[int, ...],
        predicted: ObjectIdentity,
        stance: int,
        trace: tuple[int, ...],
        ) -> RevealedObjectObservation:
    """按 assessment stance 构造独立 verifier reveal。"""
    supported = (predicted,) if stance == EVIDENCE_SUPPORT else ()
    refuted = (predicted,) if stance == EVIDENCE_REFUTE else ()
    return RevealedObjectObservation(
        verifier_source,
        document_scope(verifier_source),
        event_key,
        verifier_source,
        supported,
        refuted,
        trace,
    )


def apply_generation_generalization_training_assessments(
        seeds: tuple[AuthoredGenerationGeneralizationSeed, ...],
        mapper: GenerationChoiceCandidateMapper,
        learning: CandidateLearningRuntime,
        verifier_source: SourceRef,
        policy: GenerationGeneralizationAssessmentRuntimePolicy,
        ) -> GenerationGeneralizationTrainingReport:
    """只消费 TRAIN challenge label，批量写一个 exact candidate assessment。"""
    if (not isinstance(seeds, tuple) or not seeds
            or len(seeds) > policy.max_training_cases
            or any(not isinstance(item, AuthoredGenerationGeneralizationSeed)
                   or item.split != "train" or item.label_owner != "teacher"
                   for item in seeds)):
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 training seeds 非法或超预算")
    if not isinstance(mapper, GenerationChoiceCandidateMapper):
        raise TypeError("GG-03 training mapper 类型错误")
    if not isinstance(learning, CandidateLearningRuntime):
        raise TypeError("GG-03 training learning 类型错误")
    if not isinstance(verifier_source, SourceRef):
        raise TypeError("GG-03 training verifier source 类型错误")
    if not isinstance(policy, GenerationGeneralizationAssessmentRuntimePolicy):
        raise TypeError("GG-03 training policy 类型错误")
    cases = tuple(project_generation_generalization_seed(item) for item in seeds)
    if any(len(item.options) > policy.max_options_per_case for item in cases):
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 training option count 超预算")
    definitions = tuple(
        mapper.definition(item.challenge.choice) for item in cases)
    if len({item.candidate for item in definitions}) != len(definitions):
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 training challenge candidate 重复")
    for definition in definitions:
        try:
            learning.hypothesis_for_candidate(definition.candidate)
        except KeyError:
            continue
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 training assessment 不得重复消费")
    starts = learning.next_timestamps(len(definitions))
    hypotheses = learning.register_many(tuple(zip(
        definitions, starts, strict=True)))
    timestamp_values = learning.next_timestamps(3 * len(cases))
    requests = []
    stances = []
    for index, (seed, case, hypothesis) in enumerate(zip(
            seeds, cases, hypotheses, strict=True)):
        expected = seed.expected_payload.to_value()
        if expected["challenge_verdict"] != seed.expected_state:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 training challenge verdict 漂移")
        stance = _stance(seed.expected_state)
        stances.append(stance)
        event_ref = integer_tuple_fingerprint(
            _text_values(seed.seed_id, case.challenge.surface_candidate_id,
                         seed.expected_state),
            domain="gg03.runtime.assessment.event.v1",
        )
        event_key = (*policy.event_namespace, *event_ref)
        trace = integer_tuple_fingerprint(
            (*case.trace, stance),
            domain="gg03.runtime.assessment.trace.v1",
        )
        timestamp_seq, resolve_seq, projection_seq = timestamp_values[
            index * 3:index * 3 + 3]
        requests.append(CandidateRecognitionRequest(
            hypothesis,
            verifier_source,
            document_scope(verifier_source),
            event_key,
            case.visible_inputs,
            case.challenge.choice.selected_object,
            _revealed(
                verifier_source,
                event_key,
                case.challenge.choice.selected_object,
                stance,
                trace,
            ),
            timestamp_seq,
            resolve_seq,
            projection_seq,
        ))
    outcomes = learning.recognize_many(tuple(requests))
    records = []
    for case, stance, outcome in zip(
            cases, stances, outcomes, strict=True):
        selection = select_generation_choice_by_assessment(
            mapper,
            learning,
            policy.selector_policy,
            case.choices,
            case.baseline.choice,
        )
        records.append(GenerationGeneralizationTrainingRecord(
            case, stance, outcome, selection))
    return GenerationGeneralizationTrainingReport(
        tuple(records), len(definitions), len(outcomes))


def evaluate_generation_generalization_held_out(
        seeds: tuple[AuthoredGenerationGeneralizationSeed, ...],
        mapper: GenerationChoiceCandidateMapper,
        learning: CandidateLearningRuntime,
        policy: GenerationGeneralizationAssessmentRuntimePolicy,
        ) -> GenerationGeneralizationHeldOutReport:
    """先对未见 Observation 零写选择，再读取 evaluator accepted set。"""
    if (not isinstance(seeds, tuple) or not seeds
            or len(seeds) > policy.max_held_out_cases
            or any(not isinstance(item, AuthoredGenerationGeneralizationSeed)
                   or item.split != "held_out" or item.label_owner != "evaluator"
                   for item in seeds)):
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 held-out seeds 非法或超预算")
    cases = tuple(project_generation_generalization_seed(item) for item in seeds)
    before = learning.state_key()
    selections = tuple(
        select_generation_choice_by_assessment(
            mapper,
            learning,
            policy.selector_policy,
            case.choices,
            case.baseline.choice,
        )
        for case in cases
    )
    if learning.state_key() != before:
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 held-out selection 发生 host write")

    records = []
    # Label phase starts only after every held-out selection is frozen above.
    for seed, case, selection in zip(seeds, cases, selections, strict=True):
        accepted = tuple(
            seed.expected_payload.to_value()["accepted_surface_candidate_ids"])
        selected_id = case.option_for_choice(
            selection.selected).surface_candidate_id
        records.append(GenerationGeneralizationHeldOutRecord(
            case,
            selection,
            selected_id,
            accepted,
            int(selected_id in accepted),
            "NE_INDEPENDENT_LAYER_INPUT_MISSING",
        ))
    return GenerationGeneralizationHeldOutReport(
        tuple(records),
        0,
        0,
        "NE_INDEPENDENT_LAYER_INPUT_MISSING",
    )


def revise_generation_generalization_assessment(
        case: GenerationGeneralizationAssessmentCase,
        prior_evidence: EvidenceRecord,
        stance: int,
        mapper: GenerationChoiceCandidateMapper,
        learning: CandidateLearningRuntime,
        verifier_source: SourceRef,
        selector_policy: GenerationChoiceAssessmentSelectorPolicy,
        ) -> GenerationGeneralizationAssessmentRevision:
    """append-only 替代 challenge Evidence，并重算 selection。"""
    if not isinstance(case, GenerationGeneralizationAssessmentCase):
        raise TypeError("GG-03 revision case 类型错误")
    if not isinstance(prior_evidence, EvidenceRecord):
        raise TypeError("GG-03 revision prior evidence 类型错误")
    if stance not in {EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 revision stance 非法")
    candidate = mapper.candidate_identity(case.challenge.choice)
    hypothesis = learning.hypothesis_for_candidate(candidate)
    if (prior_evidence.hypothesis != hypothesis
            or prior_evidence not in learning.engine.ledger.evidence_history(
                hypothesis)):
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 revision prior evidence 不属于 challenge")
    evidence_ids = (
        item.evidence_id
        for definition in learning.engine.definitions()
        for item in learning.engine.ledger.evidence_history(
            definition.hypothesis(learning.engine.protocol))
    )
    evidence_id = max(evidence_ids, default=0) + 1
    timestamp_seq, resolve_seq, projection_seq = learning.next_timestamps(3)
    reason_key = integer_tuple_fingerprint(
        (*case.trace, prior_evidence.evidence_id, stance),
        domain="gg03.runtime.revision.reason.v1",
    )
    evidence = EvidenceRecord(
        evidence_id,
        hypothesis,
        stance,
        reason_key,
        verifier_source,
        timestamp_seq,
        prior_evidence.payload,
        prior_evidence.evidence_id,
    )
    outcome = learning.revise_evidence(
        evidence,
        resolve_timestamp_seq=resolve_seq,
        projection_timestamp_seq=projection_seq,
    )
    selection = select_generation_choice_by_assessment(
        mapper,
        learning,
        selector_policy,
        case.choices,
        case.baseline.choice,
    )
    return GenerationGeneralizationAssessmentRevision(
        case,
        prior_evidence.evidence_id,
        stance,
        outcome,
        selection,
    )


def _text_values(*values: str) -> tuple[int, ...]:
    """把 assessment event 文本编码成无歧义整数片段。"""
    encoded = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime text key 非法")
        raw = value.encode("utf-8")
        encoded.extend((len(raw), *raw))
    return tuple(encoded)


__all__ = [
    "GG03_DIMENSION_CHOICE_KIND",
    "GG03_EVALUATOR_OWNER",
    "GG03_HELD_OUT_RUNTIME_STATUSES",
    "GG03_RUNTIME_VERSIONS",
    "GG03_TRAINING_OWNER",
    "GenerationGeneralizationAssessmentCase",
    "GenerationGeneralizationAssessmentRevision",
    "GenerationGeneralizationAssessmentRuntimeError",
    "GenerationGeneralizationAssessmentRuntimePolicy",
    "GenerationGeneralizationHeldOutRecord",
    "GenerationGeneralizationHeldOutReport",
    "GenerationGeneralizationSurfaceOption",
    "GenerationGeneralizationTrainingRecord",
    "GenerationGeneralizationTrainingReport",
    "apply_generation_generalization_training_assessments",
    "evaluate_generation_generalization_held_out",
    "project_generation_generalization_observation",
    "project_generation_generalization_seed",
    "revise_generation_generalization_assessment",
]
