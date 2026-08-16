"""把 GG-02 exact layer outcome 消费为通用 H-05 choice assessment。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
    CandidateLearningRuntime,
    CandidateRecognitionRequest,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    CandidateVerification,
    CandidateRecognitionRecord,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    USE_KINDS,
    GenerationChoiceCandidateMapper,
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
)
from pure_integer_ai.experiments.ph2_generation_choice_outcome_bridge import (
    ASSESSMENT_STATES,
    GenerationChoiceAssessmentInput,
    GenerationChoiceEpisodeAttribution,
    GenerationChoiceLayerOutcome,
    GenerationChoiceUseAttribution,
    GenerationLayeredOutcomeReport,
    build_assessment_inputs,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)


# object-model: exception
class GenerationChoiceAssessmentConsumerError(ValueError):
    """assessment 输入、candidate、event 或 H-05 更新发生漂移。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise GenerationChoiceAssessmentConsumerError(
            f"{where} 必须是非空严格整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise GenerationChoiceAssessmentConsumerError(
            f"{where} 必须是非空严格整数 tuple")
    return value


def _choice_order(choice_kind: str) -> int:
    """返回冻结五层顺序，拒绝未知 choice kind。"""
    if choice_kind not in CHOICE_KINDS:
        raise GenerationChoiceAssessmentConsumerError(
            "assessment choice kind 未注册")
    return CHOICE_KINDS.index(choice_kind)


def _content_ref(values: tuple[int, ...], *, domain: str) -> tuple[int, ...]:
    """把已保存在 record 的开放整数内容压成固定长度路由引用。"""
    return integer_tuple_fingerprint(values, domain=domain)


def _use_ref(use: GenerationChoiceUseRef) -> tuple[int, ...]:
    """压缩 exact Use 的两个开放键，保留 kind 与授权 scope。"""
    if not isinstance(use, GenerationChoiceUseRef):
        raise TypeError("choice assessment use 类型错误")
    return (
        USE_KINDS.index(use.use_kind),
        *_packed(_content_ref(
            use.use_key.components,
            domain="generation.choice.assessment.use-key.v1")),
        *_packed(_content_ref(
            use.selection_key.components,
            domain="generation.choice.assessment.selection-key.v1")),
        *_packed(use.scope.stable_key()),
    )


def _choice_ref(
        choice: GenerationChoiceHypothesis,
        exact_use: GenerationChoiceUseRef,
        exact_use_ref: tuple[int, ...],
        ) -> tuple[int, ...]:
    """无重复展开地引用 GG-01 choice 的全部公开合同字段。"""
    values = [
        *_packed(choice.candidate.stable_key()),
        _choice_order(choice.choice_kind),
        *_packed(choice.target_obligation.stable_key()),
        *_packed(choice.condition.condition.stable_key()),
        *_packed(choice.condition.context.stable_key()),
        len(choice.condition.required_context_objects),
    ]
    for item in choice.condition.required_context_objects:
        values.extend(_packed(item.stable_key()))
    values.append(len(choice.condition.forbidden_context_objects))
    for item in choice.condition.forbidden_context_objects:
        values.extend(_packed(item.stable_key()))
    values.extend((
        *_packed(choice.condition.authorized_scope.stable_key()),
        *_packed(choice.selected_object.stable_key()),
        len(choice.forming_sources),
    ))
    for source in choice.forming_sources:
        values.extend(_packed(source.stable_key()))
    values.extend((
        *_packed(_content_ref(
            choice.competition_key,
            domain="generation.choice.assessment.competition.v1")),
        *_packed(choice.authorized_scope.stable_key()),
        len(choice.exact_uses),
    ))
    for item in choice.exact_uses:
        use_ref = exact_use_ref if item is exact_use else _use_ref(item)
        values.extend(_packed(use_ref))
    values.append(len(choice.typed_outcomes))
    for item in choice.typed_outcomes:
        values.extend((
            *_packed(_content_ref(
                item.outcome_key.components,
                domain="generation.choice.assessment.typed-outcome.v1")),
            *_packed(_content_ref(
                item.use_key.components,
                domain="generation.choice.assessment.typed-use.v1")),
            *_packed(_content_ref(
                item.dimension_key.components,
                domain="generation.choice.assessment.typed-dimension.v1")),
            *_packed(_content_ref(
                item.verifier_key.components,
                domain="generation.choice.assessment.typed-verifier.v1")),
            *_packed(_content_ref(
                item.result_key.components,
                domain="generation.choice.assessment.typed-result.v1")),
        ))
    return _content_ref(
        tuple(values), domain="generation.choice.assessment.choice.v1")


def _assessment_binding_key(
        policy: "GenerationChoiceAssessmentConsumerPolicy",
        episode: GenerationChoiceEpisodeAttribution,
        attribution: GenerationChoiceUseAttribution,
        assessment: GenerationChoiceAssessmentInput,
        ) -> tuple[int, ...]:
    """形成不重复内联 exact Use 的完整 assessment 内容引用输入。"""
    use_ref = _use_ref(attribution.use)
    choice_ref = _choice_ref(attribution.choice, attribution.use, use_ref)
    values = [
        *_packed(policy.stable_key()),
        *_packed(_content_ref(
            episode.context_key.components,
            domain="generation.choice.assessment.episode-context.v1")),
        *_packed(_content_ref(
            episode.query_key.components,
            domain="generation.choice.assessment.episode-query.v1")),
        *_packed(_content_ref(
            episode.generation_key.components,
            domain="generation.choice.assessment.episode-generation.v1")),
        *_packed(episode.source.stable_key()),
        *_packed(episode.scope.stable_key()),
        *_packed(choice_ref),
        *_packed(use_ref),
        *_packed(_content_ref(
            attribution.query_key.components,
            domain="generation.choice.assessment.attribution-query.v1")),
        *_packed(_content_ref(
            attribution.generation_key.components,
            domain="generation.choice.assessment.attribution-generation.v1")),
        len(attribution.verification_claim_keys),
    ]
    for item in attribution.verification_claim_keys:
        values.extend(_packed(_content_ref(
            item.components,
            domain="generation.choice.assessment.verifier-claim.v1")))
    values.extend((
        *_packed(attribution.source.stable_key()),
        *_packed(attribution.scope.stable_key()),
        *_packed(assessment.choice_candidate_key.components),
        _choice_order(assessment.choice_kind),
        ASSESSMENT_STATES.index(assessment.assessment_state),
        len(assessment.outcomes),
    ))
    for item in assessment.outcomes:
        values.extend((
            *_packed(item.dimension.stable_key()),
            *_packed(item.verifier.stable_key()),
            item.applicability,
            item.verdict,
            *_packed(_content_ref(
                item.detail.components,
                domain="generation.choice.assessment.outcome-detail.v1")),
            item.assessment_ready,
            *_packed(() if item.source is None else item.source.stable_key()),
            *_packed(() if item.scope is None else item.scope.stable_key()),
        ))
    return tuple(values)


def _projection_key(
        projection: CandidateGraphProjection | None,
        ) -> tuple[int, ...]:
    """展开 clone 可复现的候选 lifecycle，不编码存储本地引用。"""
    if projection is None:
        return ()
    values = [
        *_packed(projection.candidate.definition.stable_key()),
        *_packed(projection.candidate.hypothesis.stable_key()),
        *_packed(projection.state.stable_key()),
        len(projection.history),
    ]
    for item in projection.history:
        event = item.definition
        values.extend((
            *_packed(event.event.stable_key()),
            *_packed(event.definition.stable_key()),
            *_packed(event.event_kind.stable_key()),
            *_packed(event.from_state.stable_key()),
            *_packed(event.to_state.stable_key()),
            *_packed(event.hypothesis.stable_key()),
            len(event.evidence_keys),
        ))
        for evidence_key in event.evidence_keys:
            values.extend(_packed(evidence_key))
        values.extend((
            *_packed(event.decision_key),
            event.timestamp_seq,
            *_packed(() if event.replacement is None
                     else event.replacement.stable_key()),
        ))
    values.extend(_packed(
        () if projection.replacement is None
        else projection.replacement.stable_key()))
    return tuple(values)


def _verification_key(
        verification: CandidateVerification,
        ) -> tuple[int, ...]:
    """展开 CandidateVerification 的全部揭示字段。"""
    return (
        verification.stance,
        *_packed(verification.reason_key),
        *_packed(verification.source.stable_key()),
        *_packed(verification.authority.stable_key()),
        *_packed(verification.authority_version),
        *_packed(verification.trace),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationChoiceAssessmentConsumerPolicy:
    """注入 verifier source、event 域、关闭层和整数资源上限。"""

    verifier_source: SourceRef
    event_namespace: tuple[int, ...]
    disabled_choice_kinds: tuple[str, ...] = ()
    archive_refuted: bool = False
    max_updates_per_batch: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.verifier_source, SourceRef):
            raise TypeError("choice assessment verifier source 类型错误")
        _strict_key(
            self.event_namespace,
            where="choice assessment event namespace")
        if (not isinstance(self.disabled_choice_kinds, tuple)
                or any(item not in CHOICE_KINDS
                       for item in self.disabled_choice_kinds)
                or len(set(self.disabled_choice_kinds))
                != len(self.disabled_choice_kinds)):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment disabled layers 非法")
        object.__setattr__(self, "disabled_choice_kinds", tuple(sorted(
            self.disabled_choice_kinds, key=_choice_order)))
        if type(self.archive_refuted) is not bool:
            raise TypeError("choice assessment archive_refuted 类型错误")
        assert_int(
            self.max_updates_per_batch,
            _where="choice assessment max updates")
        if (type(self.max_updates_per_batch) is not int
                or self.max_updates_per_batch <= 0
                or self.max_updates_per_batch > len(CHOICE_KINDS)):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment max updates 超出五层边界")

    def stable_key(self) -> tuple[int, ...]:
        """返回 verifier、事件域、关闭层和资源边界。"""
        return (
            *_packed(self.verifier_source.stable_key()),
            *_packed(self.event_namespace),
            len(self.disabled_choice_kinds),
            *(_choice_order(item) for item in self.disabled_choice_kinds),
            int(self.archive_refuted),
            self.max_updates_per_batch,
        )


def _ready_outcomes(
        assessment: GenerationChoiceAssessmentInput,
        ) -> tuple[GenerationChoiceLayerOutcome, ...]:
    """只返回 GG-02 明确标记可供 assessment 的 applicable outcomes。"""
    return tuple(
        item for item in assessment.outcomes if item.assessment_ready)


def assessment_input_stance(
        assessment: GenerationChoiceAssessmentInput,
        ) -> int:
    """按 refute > unknown/conflicted > all-support 聚合为 H-05 三态。"""
    if not isinstance(assessment, GenerationChoiceAssessmentInput):
        raise TypeError("choice assessment input 类型错误")
    if assessment.assessment_state != "READY":
        raise GenerationChoiceAssessmentConsumerError(
            "非 READY assessment input 不得形成 Evidence stance")
    outcomes = _ready_outcomes(assessment)
    if not outcomes:
        raise GenerationChoiceAssessmentConsumerError(
            "READY assessment input 缺可消费 outcome")
    verdicts = tuple(item.verdict for item in outcomes)
    if VERDICT_REFUTE in verdicts:
        return EVIDENCE_REFUTE
    if all(item == VERDICT_SUPPORT for item in verdicts):
        return EVIDENCE_SUPPORT
    return EVIDENCE_UNKNOWN


def _event_key(
        policy: GenerationChoiceAssessmentConsumerPolicy,
        binding_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    """保留可路由 namespace，并追加固定长度 assessment 内容引用。"""
    return (
        *policy.event_namespace,
        *_content_ref(
            binding_key, domain="generation.choice.assessment.event.v1"),
    )


def _visible_inputs(
        attribution: GenerationChoiceUseAttribution,
        ) -> tuple[ObjectIdentity, ...]:
    """保存目标、context、condition 与 required objects，不内联答案。"""
    choice = attribution.choice
    values = (
        choice.target_obligation,
        choice.condition.context,
        choice.condition.condition,
        *choice.condition.required_context_objects,
    )
    return tuple(sorted(set(values), key=ObjectIdentity.stable_key))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PreparedGenerationChoiceAssessment:
    """宿主首写前冻结的 exact attribution、event、stance 与 trace。"""

    assessment: GenerationChoiceAssessmentInput
    attribution: GenerationChoiceUseAttribution
    event_key: tuple[int, ...]
    stance: int
    visible_inputs: tuple[ObjectIdentity, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assessment, GenerationChoiceAssessmentInput):
            raise TypeError("prepared choice assessment input 类型错误")
        if not isinstance(self.attribution, GenerationChoiceUseAttribution):
            raise TypeError("prepared choice attribution 类型错误")
        _strict_key(self.event_key, where="prepared choice event key")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise GenerationChoiceAssessmentConsumerError(
                "prepared choice stance 未注册")
        if (not isinstance(self.visible_inputs, tuple)
                or not self.visible_inputs
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.visible_inputs)):
            raise GenerationChoiceAssessmentConsumerError(
                "prepared choice visible inputs 非法")
        _strict_key(self.trace, where="prepared choice trace")
        choice = self.attribution.choice
        if (self.assessment.choice_candidate_key.components
                != choice.candidate.stable_key()
                or self.assessment.choice_kind != choice.choice_kind
                or self.assessment.use != self.attribution.use):
            raise GenerationChoiceAssessmentConsumerError(
                "prepared assessment 与 exact choice/use 漂移")

    def stable_key(self) -> tuple[int, ...]:
        """返回 assessment、attribution、event、stance 与可见输入。"""
        values = [
            *_packed(self.event_key),
            self.stance,
            len(self.visible_inputs),
        ]
        for item in self.visible_inputs:
            values.extend(_packed(item.stable_key()))
        values.extend(_packed(self.trace))
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationChoiceAssessmentUpdate:
    """一次 READY input 对 exact H-05 candidate 的真实学习结果。"""

    prepared: PreparedGenerationChoiceAssessment
    candidate_registered: int
    learning: CandidateLearningOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedGenerationChoiceAssessment):
            raise TypeError("choice assessment update prepared 类型错误")
        assert_int(
            self.candidate_registered,
            _where="choice assessment candidate registered")
        if self.candidate_registered not in (0, 1):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment candidate registered 必须为 0/1")
        if not isinstance(self.learning, CandidateLearningOutcome):
            raise TypeError("choice assessment learning outcome 类型错误")
        if (self.learning.prediction.event_key != self.prepared.event_key
                or self.learning.prediction.predicted
                != self.prepared.attribution.choice.selected_object
                or self.learning.verification.stance != self.prepared.stance):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment learning 未绑定 prepared event/stance")

    def stable_key(self) -> tuple[int, ...]:
        """返回 prepared、注册标志和 H-05 全链结果。"""
        projection_key = _projection_key(self.learning.projection)
        return (
            *_packed(self.prepared.stable_key()),
            self.candidate_registered,
            *_packed(self.learning.prediction.stable_key()),
            *_packed(_verification_key(self.learning.verification)),
            *_packed(self.learning.evidence.stable_key()),
            *_packed(self.learning.decision.stable_key()),
            *_packed(projection_key),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationChoiceAssessmentConsumerReport:
    """一次 apply 的 records、真实新增写入与重放计数。"""

    records: tuple[GenerationChoiceAssessmentUpdate, ...]
    candidate_registrations: int
    assessment_updates_executed: int
    replayed_updates: int
    teacher_call_count: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.records, tuple)
                or any(not isinstance(item, GenerationChoiceAssessmentUpdate)
                       for item in self.records)):
            raise TypeError("choice assessment report records 类型错误")
        ordered = tuple(sorted(
            self.records,
            key=lambda item: _choice_order(
                item.prepared.assessment.choice_kind),
        ))
        object.__setattr__(self, "records", ordered)
        if len({item.prepared.event_key for item in ordered}) != len(ordered):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment report event 重复")
        for value, label in (
                (self.candidate_registrations, "candidate registrations"),
                (self.assessment_updates_executed, "assessment updates"),
                (self.replayed_updates, "replayed updates"),
                (self.teacher_call_count, "teacher calls")):
            assert_int(value, _where=f"choice assessment {label}")
            if type(value) is not int or value < 0:
                raise GenerationChoiceAssessmentConsumerError(
                    f"choice assessment {label} 非法")
        if self.teacher_call_count != 0:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment consumer 不得调用 teacher")
        if (self.assessment_updates_executed + self.replayed_updates
                != len(self.records)):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment 新增/重放计数与 records 不一致")

    @property
    def support_records(self) -> int:
        """返回本报告中 support assessment record 数。"""
        return sum(
            item.prepared.stance == EVIDENCE_SUPPORT for item in self.records)

    @property
    def refute_records(self) -> int:
        """返回本报告中 refute assessment record 数。"""
        return sum(
            item.prepared.stance == EVIDENCE_REFUTE for item in self.records)

    @property
    def unknown_records(self) -> int:
        """返回本报告中 unknown assessment record 数。"""
        return sum(
            item.prepared.stance == EVIDENCE_UNKNOWN for item in self.records)


# object-model: runtime-consumer
class GenerationChoiceAssessmentConsumer:
    """消费 GG-02 assessment input，并更新通用 H-05 choice candidate。"""

    def __init__(
            self,
            mapper: GenerationChoiceCandidateMapper,
            learning: CandidateLearningRuntime,
            policy: GenerationChoiceAssessmentConsumerPolicy,
            ) -> None:
        if not isinstance(mapper, GenerationChoiceCandidateMapper):
            raise TypeError("choice assessment mapper 类型错误")
        if not isinstance(learning, CandidateLearningRuntime):
            raise TypeError("choice assessment learning 类型错误")
        if not isinstance(policy, GenerationChoiceAssessmentConsumerPolicy):
            raise TypeError("choice assessment policy 类型错误")
        self.mapper = mapper
        self.learning = learning
        self.policy = policy
        self._processed: dict[
            tuple[int, ...], GenerationChoiceAssessmentUpdate] = {}

    def _belongs_to_namespace(self, event_key: tuple[int, ...]) -> bool:
        """判断持久 event 是否属于当前显式 namespace。"""
        namespace = self.policy.event_namespace
        return (
            len(event_key) > len(namespace)
            and event_key[:len(namespace)] == namespace
        )

    def _history_by_event(
            self,
            ) -> dict[tuple[int, ...], CandidateRecognitionRecord]:
        """从 H-00 recognition history 恢复当前 namespace 的唯一事件索引。"""
        restored: dict[tuple[int, ...], CandidateRecognitionRecord] = {}
        protocol = self.learning.engine.protocol
        for definition in self.learning.engine.definitions():
            hypothesis = definition.hypothesis(protocol)
            for record in self.learning.engine.recognition_history(hypothesis):
                event_key = record.prediction.event_key
                if not self._belongs_to_namespace(event_key):
                    continue
                if record.prediction.observation != self.policy.verifier_source:
                    raise GenerationChoiceAssessmentConsumerError(
                        "choice assessment namespace 被其他 observation 占用")
                prior = restored.get(event_key)
                if prior is not None and prior != record:
                    raise GenerationChoiceAssessmentConsumerError(
                        "choice assessment 持久 event 重复绑定不同 recognition")
                restored[event_key] = record
        return restored

    def _revealed(
            self,
            item: PreparedGenerationChoiceAssessment,
            ) -> RevealedObjectObservation:
        """从 prepared stance 重建独立 verifier 的显式揭示输入。"""
        selected = item.attribution.choice.selected_object
        supported = (selected,) if item.stance == EVIDENCE_SUPPORT else ()
        refuted = (selected,) if item.stance == EVIDENCE_REFUTE else ()
        return RevealedObjectObservation(
            self.policy.verifier_source,
            document_scope(self.policy.verifier_source),
            item.event_key,
            self.policy.verifier_source,
            supported,
            refuted,
            item.trace,
        )

    def _projection_for_decision(
            self,
            hypothesis: HypothesisKey,
            decision,
            evidence_timestamp: int,
            ) -> CandidateGraphProjection | None:
        """从图 history 截取该 decision 当时返回的候选投影。"""
        definition = self.learning.engine.definition(hypothesis)
        candidate_ref = self.learning.graph.ontology.resolve(
            definition.candidate)
        if candidate_ref is None:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment 持久 candidate 图对象缺失")
        history = self.learning.graph.history(candidate_ref)
        decision_key = decision.stable_key()
        matches = tuple(
            index for index, event in enumerate(history)
            if event.definition.decision_key == decision_key)
        if not matches:
            return None
        if len(matches) != 1:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment decision 对应多个 candidate projection")
        index = matches[0]
        event = history[index].definition
        if event.timestamp_seq != evidence_timestamp + 2:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment projection 与 Evidence 逻辑序不相邻")
        materialized = self.learning.graph.read_definition(hypothesis)
        return CandidateGraphProjection(
            materialized,
            event.to_state,
            history[:index + 1],
            event.replacement,
        )

    def _recover_update(
            self,
            item: PreparedGenerationChoiceAssessment,
            record: CandidateRecognitionRecord,
            ) -> GenerationChoiceAssessmentUpdate:
        """把持久 recognition、相邻 decision 与图 event 恢复为学习结果。"""
        choice = item.attribution.choice
        candidate = self.mapper.candidate_identity(choice)
        try:
            hypothesis = self.learning.hypothesis_for_candidate(candidate)
        except KeyError as exc:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment event 存在但 candidate 未恢复") from exc
        prediction = record.prediction
        if (
                prediction.hypothesis != hypothesis
                or prediction.observation != self.policy.verifier_source
                or prediction.scope != document_scope(
                    self.policy.verifier_source)
                or prediction.event_key != item.event_key
                or prediction.visible_inputs != item.visible_inputs
                or prediction.predicted != choice.selected_object):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment 持久 prediction 与当前 prepared 漂移")
        expected_verification = self.learning.verifier.verify(
            prediction, self._revealed(item))
        if record.verification != expected_verification:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment 持久 verification 与当前 outcome 漂移")
        evidence = record.evidence
        if (
                evidence.hypothesis != hypothesis
                or evidence.stance != record.verification.stance
                or evidence.reason_key != record.verification.reason_key
                or evidence.source != record.verification.source
                or evidence.payload != record.verification.payload_for(
                    prediction)):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment 持久 Evidence 与 prediction/verification 漂移")
        decisions = tuple(
            decision
            for decision in self.learning.engine.resolver.decision_history(
                hypothesis)
            if decision.timestamp_seq == evidence.timestamp_seq + 1
        )
        if len(decisions) != 1:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment 缺唯一相邻 H-04 decision")
        decision = decisions[0]
        trace = decision.candidate(hypothesis)
        active_evidence_ids = frozenset((
            *trace.after.support_evidence_ids,
            *trace.after.refute_evidence_ids,
            *trace.after.unknown_evidence_ids,
        ))
        if evidence.evidence_id not in active_evidence_ids:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment 相邻 decision 未消费目标 Evidence")
        projection = self._projection_for_decision(
            hypothesis, decision, evidence.timestamp_seq)
        return GenerationChoiceAssessmentUpdate(
            item,
            0,
            CandidateLearningOutcome(
                prediction,
                record.verification,
                evidence,
                decision,
                projection,
            ),
        )

    def _preflight_projection_state(
            self,
            engine,
            anchor: HypothesisKey,
            projection_timestamp: int,
            ) -> None:
        """零写核验 probe H-04 结果可映射到当前候选图状态。"""
        graph = self.learning.graph
        for snapshot in engine.ledger.competition(anchor):
            hypothesis = snapshot.hypothesis
            definition = engine.definition(hypothesis)
            graph.preflight_definition(
                definition,
                hypothesis,
                **self.learning.metadata.kwargs(),
            )
            candidate_ref = graph.ontology.resolve(definition.candidate)
            history = () if candidate_ref is None else graph.history(candidate_ref)
            active = engine.active(hypothesis)
            if active is not None:
                active_ids = frozenset((
                    *active.snapshot.support_evidence_ids,
                    *active.snapshot.refute_evidence_ids,
                    *active.snapshot.unknown_evidence_ids,
                ))
                evidence = tuple(
                    item for item in engine.ledger.evidence_history(hypothesis)
                    if item.evidence_id in active_ids)
                if (
                        not evidence
                        or {item.evidence_id for item in evidence} != active_ids
                        or projection_timestamp < active.decision.timestamp_seq
                        or any(item.timestamp_seq > projection_timestamp
                               for item in evidence)):
                    raise GenerationChoiceAssessmentConsumerError(
                        "choice assessment active projection 预检失败")
                if history:
                    projection = graph.project(candidate_ref)
                    if projection.state == graph.protocol.superseded_state:
                        raise GenerationChoiceAssessmentConsumerError(
                            "choice assessment 不得重新激活 superseded candidate")
                    if projection_timestamp <= history[-1].definition.timestamp_seq:
                        raise GenerationChoiceAssessmentConsumerError(
                            "choice assessment projection 逻辑序未推进")
                continue
            if not history:
                continue
            projection = graph.project(candidate_ref)
            if projection.state != graph.protocol.active_state:
                continue
            decisions = engine.resolver.decision_history(hypothesis)
            if not decisions:
                raise GenerationChoiceAssessmentConsumerError(
                    "choice assessment demotion 缺 H-04 decision")
            decision = decisions[-1]
            current = engine.ledger.snapshot(hypothesis)
            active_ids = frozenset((
                *current.support_evidence_ids,
                *current.refute_evidence_ids,
                *current.unknown_evidence_ids,
            ))
            evidence = tuple(
                item for item in engine.ledger.evidence_history(hypothesis)
                if item.evidence_id in active_ids)
            if (
                    not evidence
                    or {item.evidence_id for item in evidence} != active_ids
                    or projection_timestamp < decision.timestamp_seq
                    or any(item.timestamp_seq > projection_timestamp
                           for item in evidence)
                    or projection_timestamp <= history[-1].definition.timestamp_seq):
                raise GenerationChoiceAssessmentConsumerError(
                    "choice assessment inactive projection 预检失败")

    def _preflight_pending(
            self,
            registrations: tuple[tuple[EvidenceCandidateDefinition, int], ...],
            requests: tuple[CandidateRecognitionRequest, ...],
            ) -> None:
        """在宿主首写前预演 forming、recognition、H-04 与图状态转换。"""
        if registrations:
            self.learning.preflight_register_many(registrations)
        probe = self.learning.engine.clone()
        if registrations:
            probe.register_many(registrations)
        for request in requests:
            prediction = probe.predict(
                request.hypothesis,
                observation=request.observation,
                scope=request.scope,
                event_key=request.event_key,
                visible_inputs=request.visible_inputs,
                predicted=request.predicted,
            )
            verification = self.learning.verifier.verify(
                prediction, request.revealed)
            probe.reveal(
                prediction,
                verification,
                timestamp_seq=request.timestamp_seq,
            )
            probe.resolve(
                request.hypothesis,
                timestamp_seq=request.resolve_timestamp_seq,
                scorers=request.scorers,
                archive_refuted=request.archive_refuted,
                replacement=request.replacement,
            )
            self._preflight_projection_state(
                probe,
                request.hypothesis,
                request.projection_timestamp_seq,
            )

    def prepare(
            self,
            report: GenerationLayeredOutcomeReport,
            ) -> tuple[PreparedGenerationChoiceAssessment, ...]:
        """零写形成 READY 层的 exact candidate event 和三态 stance。"""
        if not isinstance(report, GenerationLayeredOutcomeReport):
            raise TypeError("choice assessment layered report 类型错误")
        assessment = build_assessment_inputs(
            report,
            disabled_choice_kinds=self.policy.disabled_choice_kinds,
        )
        by_kind = {
            item.choice.choice_kind: item for item in report.episode.choices}
        prepared = []
        for item in assessment.inputs:
            if item.assessment_state != "READY":
                continue
            attribution = by_kind[item.choice_kind]
            binding_key = _assessment_binding_key(
                self.policy, report.episode, attribution, item)
            event_key = _event_key(self.policy, binding_key)
            trace = _content_ref(
                binding_key,
                domain="generation.choice.assessment.trace.v1")
            prepared.append(PreparedGenerationChoiceAssessment(
                item,
                attribution,
                event_key,
                assessment_input_stance(item),
                _visible_inputs(attribution),
                trace,
            ))
        if len(prepared) > self.policy.max_updates_per_batch:
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment READY 层超过批次上限")
        keys = tuple(item.event_key for item in prepared)
        if len(set(keys)) != len(keys):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment batch event 重复")
        return tuple(sorted(
            prepared,
            key=lambda item: _choice_order(item.assessment.choice_kind),
        ))

    def _definitions(
            self,
            prepared: tuple[PreparedGenerationChoiceAssessment, ...],
            ) -> tuple[
                dict[ObjectIdentity, EvidenceCandidateDefinition],
                tuple[tuple[EvidenceCandidateDefinition, int], ...],
                set[ObjectIdentity],
            ]:
        """回读已有 candidate 定义，并为缺失项分配无重叠 forming 时序。"""
        definitions = {
            self.mapper.candidate_identity(item.attribution.choice):
            self.mapper.definition(item.attribution.choice)
            for item in prepared
        }
        if len(definitions) != len(prepared):
            raise GenerationChoiceAssessmentConsumerError(
                "choice assessment batch candidate 重复")
        missing = []
        existing = set()
        for candidate, definition in definitions.items():
            try:
                hypothesis = self.learning.hypothesis_for_candidate(candidate)
            except KeyError:
                missing.append(definition)
                continue
            if self.learning.engine.definition(hypothesis) != definition:
                raise GenerationChoiceAssessmentConsumerError(
                    "choice assessment 已有 candidate 定义漂移")
            existing.add(candidate)
        total_forming = sum(len(item.forming_sources) for item in missing)
        starts = (
            () if not missing
            else self.learning.next_timestamps(total_forming))
        cursor = 0
        requests = []
        for definition in missing:
            requests.append((definition, starts[cursor]))
            cursor += len(definition.forming_sources)
        return definitions, tuple(requests), existing

    def apply(
            self,
            report: GenerationLayeredOutcomeReport,
            ) -> GenerationChoiceAssessmentConsumerReport:
        """批量登记缺失 choice，并提交 READY input 的 H-05 Evidence。"""
        prepared = self.prepare(report)
        if not prepared:
            return GenerationChoiceAssessmentConsumerReport((), 0, 0, 0)
        records = []
        pending = []
        history_by_event = self._history_by_event()
        for item in prepared:
            prior = self._processed.get(item.event_key)
            if prior is None:
                persisted = history_by_event.get(item.event_key)
                if persisted is None:
                    pending.append(item)
                    continue
                recovered = self._recover_update(item, persisted)
                self._processed[item.event_key] = recovered
                records.append(recovered)
                continue
            if prior.prepared != item:
                raise GenerationChoiceAssessmentConsumerError(
                    "choice assessment 重放 event 内容漂移")
            records.append(GenerationChoiceAssessmentUpdate(
                item, 0, prior.learning))
        if not pending:
            return GenerationChoiceAssessmentConsumerReport(
                tuple(records), 0, 0, len(records))

        definitions, registrations, existing = self._definitions(
            tuple(pending))
        if registrations:
            registration_end = max(
                timestamp_base + len(definition.forming_sources) - 1
                for definition, timestamp_base in registrations)
            timestamp_values = tuple(range(
                registration_end + 1,
                registration_end + 1 + 3 * len(pending),
            ))
        else:
            timestamp_values = self.learning.next_timestamps(
                3 * len(pending))
        recognition_requests = []
        for index, item in enumerate(pending):
            choice = item.attribution.choice
            candidate = self.mapper.candidate_identity(choice)
            definition = definitions[candidate]
            hypothesis = (
                self.learning.hypothesis_for_candidate(candidate)
                if candidate in existing
                else definition.hypothesis(self.learning.engine.protocol)
            )
            timestamp_seq, resolve_seq, projection_seq = (
                timestamp_values[index * 3:index * 3 + 3])
            recognition_requests.append(CandidateRecognitionRequest(
                hypothesis,
                self.policy.verifier_source,
                document_scope(self.policy.verifier_source),
                item.event_key,
                item.visible_inputs,
                choice.selected_object,
                self._revealed(item),
                timestamp_seq,
                resolve_seq,
                projection_seq,
                (),
                self.policy.archive_refuted,
                None,
            ))
        recognition_requests = tuple(recognition_requests)
        self._preflight_pending(registrations, recognition_requests)
        if registrations:
            self.learning.register_many(registrations)
            if self.learning.next_timestamps(len(timestamp_values)) != (
                    timestamp_values):
                raise GenerationChoiceAssessmentConsumerError(
                    "choice assessment forming 后逻辑时钟漂移")
        outcomes = self.learning.recognize_many(
            recognition_requests)
        new_records = []
        for item, outcome in zip(pending, outcomes, strict=True):
            update = GenerationChoiceAssessmentUpdate(
                item,
                int(self.mapper.candidate_identity(
                    item.attribution.choice) not in existing),
                outcome,
            )
            self._processed[item.event_key] = update
            new_records.append(update)
        records.extend(new_records)
        return GenerationChoiceAssessmentConsumerReport(
            tuple(records),
            len(registrations),
            len(new_records),
            len(records) - len(new_records),
        )


__all__ = [
    "GenerationChoiceAssessmentConsumer",
    "GenerationChoiceAssessmentConsumerError",
    "GenerationChoiceAssessmentConsumerPolicy",
    "GenerationChoiceAssessmentConsumerReport",
    "GenerationChoiceAssessmentUpdate",
    "PreparedGenerationChoiceAssessment",
    "assessment_input_stance",
]
