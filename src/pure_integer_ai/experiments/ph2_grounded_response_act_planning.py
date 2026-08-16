"""把 grounded episode 编译为保留来源归因的 typed planning。"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_LANGUAGE_BRANCH,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedAnswerEpisode,
)


_NAMESPACE = 20965


# object-model: exception
class GroundedResponseActPlanningError(ValueError):
    """grounded Evidence 不能无损形成 response-act planning。"""


def _text_values(*values: str) -> tuple[int, ...]:
    """把开放文本编码为带长度边界的整数序列。"""
    result = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise GroundedResponseActPlanningError("planning 文本身份不能为空")
        payload = value.encode("utf-8")
        result.extend((len(payload), *payload))
    return tuple(result)


def _fingerprint(*values: str, domain: str) -> tuple[int, ...]:
    """从课程开放标识形成确定性完整键。"""
    return integer_tuple_fingerprint(_text_values(*values), domain=domain)


def _positive(*values: str, domain: str) -> int:
    """把完整文本引用压成仅用于路由字段的稳定正整数。"""
    fingerprint = _fingerprint(*values, domain=domain)
    result = int.from_bytes(bytes(fingerprint[2:10]), "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedEvidenceSourceBinding:
    """把课程 source_id 显式绑定到运行期 SourceRef。"""

    source_id: str
    source: SourceRef

    def __post_init__(self) -> None:
        if (not isinstance(self.source_id, str) or not self.source_id
                or self.source_id.strip() != self.source_id):
            raise GroundedResponseActPlanningError("source_id 非法")
        if not isinstance(self.source, SourceRef):
            raise TypeError("grounded source binding 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回课程标识和完整 SourceRef 的无歧义键。"""
        text = self.source_id.encode("utf-8")
        source = self.source.stable_key()
        return len(text), *text, len(source), *source


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActCandidateBinding:
    """保存 proposition_id、aggregate 定义和实际 generation candidate。"""

    proposition_id: str
    definition: EvidenceCandidateDefinition
    candidate: GenerationCandidate

    def __post_init__(self) -> None:
        if (not isinstance(self.proposition_id, str) or not self.proposition_id
                or self.proposition_id.strip() != self.proposition_id):
            raise GroundedResponseActPlanningError("proposition_id 非法")
        if not isinstance(self.definition, EvidenceCandidateDefinition):
            raise TypeError("aggregate candidate definition 类型错误")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("grounded generation candidate 类型错误")
        if self.definition.candidate != self.candidate.proposition.template:
            raise GroundedResponseActPlanningError(
                "aggregate definition 未绑定 generation Proposition")

    def stable_key(self) -> tuple[int, ...]:
        """返回 proposition、aggregate 定义与 actual candidate 的完整键。"""
        proposition = self.proposition_id.encode("utf-8")
        definition = self.definition.stable_key()
        candidate = self.candidate.stable_key()
        return (
            len(proposition), *proposition,
            len(definition), *definition,
            len(candidate), *candidate,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActPlanningBuild:
    """一条 episode、来源映射和 typed planning 的不可变编译结果。"""

    episode: GroundedAnswerEpisode
    language_branch: ObjectIdentity
    aggregate_source: SourceRef
    response_scope: ScopeIdentity
    source_bindings: tuple[GroundedEvidenceSourceBinding, ...]
    candidate_bindings: tuple[GroundedResponseActCandidateBinding, ...]
    planning: GenerationPlanningRequest
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(self.episode, GroundedAnswerEpisode):
            raise TypeError("response-act planning episode 类型错误")
        if (not isinstance(self.language_branch, ObjectIdentity)
                or self.language_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise GroundedResponseActPlanningError(
                "response-act planning language branch 非法")
        if not isinstance(self.aggregate_source, SourceRef):
            raise TypeError("response-act aggregate source 类型错误")
        if not isinstance(self.response_scope, ScopeIdentity):
            raise TypeError("response-act response scope 类型错误")
        if (self.response_scope.source != self.aggregate_source
                or self.response_scope.local_id
                != self.episode.question.response_scope_id):
            raise GroundedResponseActPlanningError(
                "response-act scope 未绑定 aggregate source/课程 scope")
        if (not isinstance(self.source_bindings, tuple)
                or any(not isinstance(item, GroundedEvidenceSourceBinding)
                       for item in self.source_bindings)):
            raise TypeError("response-act source bindings 类型错误")
        source_ids = tuple(item.source_id for item in self.source_bindings)
        if source_ids != tuple(sorted(set(source_ids))):
            raise GroundedResponseActPlanningError(
                "response-act source bindings 必须唯一有序")
        expected_sources = {
            item.source_id for item in self.episode.question.evidence}
        if set(source_ids) != expected_sources:
            raise GroundedResponseActPlanningError(
                "response-act source bindings 未覆盖课程 Evidence")
        if (not isinstance(self.candidate_bindings, tuple)
                or any(not isinstance(
                    item, GroundedResponseActCandidateBinding)
                    for item in self.candidate_bindings)):
            raise TypeError("response-act candidate bindings 类型错误")
        proposition_ids = tuple(
            item.proposition_id for item in self.candidate_bindings)
        expected_propositions = tuple(sorted({
            item.proposition_id for item in self.episode.question.evidence}))
        if proposition_ids != expected_propositions:
            raise GroundedResponseActPlanningError(
                "response-act candidate bindings 未覆盖课程 Proposition")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("response-act planning 类型错误")
        if (self.planning.goal.source != self.aggregate_source
                or self.planning.goal.scope != self.response_scope
                or self.planning.goal.target_branch != self.language_branch
                or set(self.planning.candidates) != {
                    item.candidate for item in self.candidate_bindings}):
            raise GroundedResponseActPlanningError(
                "response-act planning 与编译归属漂移")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    def source_for(self, source_id: str) -> SourceRef:
        """返回一个已冻结课程来源的运行期身份。"""
        for binding in self.source_bindings:
            if binding.source_id == source_id:
                return binding.source
        raise GroundedResponseActPlanningError("课程 source_id 未绑定")

    def candidate_for(self, proposition_id: str) -> GenerationCandidate:
        """返回一个课程 Proposition 的 aggregate candidate。"""
        for binding in self.candidate_bindings:
            if binding.proposition_id == proposition_id:
                return binding.candidate
        raise GroundedResponseActPlanningError("课程 proposition_id 未绑定")

    def stable_key(self) -> tuple[int, ...]:
        """返回 episode、来源、candidate 与 planning 的完整内容引用。"""
        if not self._stable_key_cache:
            raise RuntimeError("response-act planning stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """在构造完成时形成一次有界确定性键。"""
        episode = self.episode.episode_id.encode("utf-8")
        values = [len(episode), *episode]
        for key in (
                self.language_branch.stable_key(),
                self.aggregate_source.stable_key(),
                self.response_scope.stable_key()):
            values.extend((len(key), *key))
        values.append(len(self.source_bindings))
        for binding in self.source_bindings:
            key = binding.stable_key()
            values.extend((len(key), *key))
        values.append(len(self.candidate_bindings))
        for binding in self.candidate_bindings:
            key = binding.stable_key()
            values.extend((len(key), *key))
        planning = self.planning.stable_key()
        values.extend((len(planning), *planning))
        return tuple(values)


def _source_ref(episode: GroundedAnswerEpisode, source_id: str) -> SourceRef:
    """按 episode/source_id 建立不依赖文本内容的来源身份。"""
    return SourceRef(
        _NAMESPACE,
        _positive(
            episode.episode_id, source_id,
            domain="grounded.response.act.evidence.source.v1"),
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _aggregate_source(episode: GroundedAnswerEpisode) -> SourceRef:
    """为一次问答建立显式 aggregate 运行来源。"""
    return SourceRef(
        _NAMESPACE,
        _positive(
            episode.episode_id,
            domain="grounded.response.act.aggregate.source.v1"),
        2,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _bound_propositions(
        episode: GroundedAnswerEpisode,
        aggregate: SourceRef,
        proposition_ids: tuple[str, ...],
        ) -> dict[str, object]:
    """把课程 Proposition id 编译为 aggregate-source opaque bound views。"""
    templates = []
    definitions = {}
    for ordinal, proposition_id in enumerate(proposition_ids, start=1):
        key = _fingerprint(
            episode.episode_id, proposition_id,
            domain="grounded.response.act.proposition.v1")
        definition = AtomicPropositionDefinition(
            proposition_identity(aggregate, (_NAMESPACE, 10, *key)),
            concept_identity((_NAMESPACE, 11, *key)),
            occurrence_identity(
                aggregate, start=ordinal, end=ordinal + 1, ordinal=0),
            context_scope_identity(aggregate, (_NAMESPACE, 12, *key)),
            (),
        )
        definitions[proposition_id] = definition
        templates.append(ScopedPropositionTemplate(
            definition,
            structure_concept_identity((_NAMESPACE, 13, *key)),
        ))
    graph = PropositionTemplateGraph(tuple(templates))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((_NAMESPACE, 14, index))
        for index in range(1, 10)
    ))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((_NAMESPACE, 15, 1)), failures))
    return {
        proposition_id: substituter.substitute(
            definitions[proposition_id].proposition,
            graph,
            BindingEnvironment(),
        )
        for proposition_id in proposition_ids
    }


def _evidence_records(
        episode: GroundedAnswerEpisode,
        proposition_id: str,
        definition: EvidenceCandidateDefinition,
        sources: dict[str, SourceRef],
        ) -> tuple[EvidenceRecord, ...]:
    """把逐来源 support/refute 原样映射为 aggregate candidate Evidence。"""
    records = []
    ordinal = 0
    for evidence in episode.question.evidence:
        if evidence.proposition_id != proposition_id:
            continue
        source = sources[evidence.source_id]
        directions = []
        if evidence.support:
            directions.append(EVIDENCE_SUPPORT)
        if evidence.refute:
            directions.append(EVIDENCE_REFUTE)
        if not directions:
            directions.append(EVIDENCE_UNKNOWN)
        hypothesis = HypothesisKey(
            (_NAMESPACE, 16, 1),
            definition.stable_key(),
            definition.competition_key,
            document_scope(source),
            source,
        )
        for direction in directions:
            ordinal += 1
            records.append(EvidenceRecord(
                _positive(
                    episode.episode_id,
                    evidence.evidence_id,
                    str(direction),
                    domain="grounded.response.act.evidence.event.v1"),
                hypothesis,
                direction,
                (_NAMESPACE, 17, direction),
                source,
                ordinal,
                _fingerprint(
                    evidence.evidence_id,
                    evidence.evidence_text,
                    str(evidence.scope_id),
                    domain="grounded.response.act.evidence.payload.v1"),
            ))
    return tuple(records)


def _compile_grounded_generation_planning(
        episode: GroundedAnswerEpisode,
        language_branch: ObjectIdentity,
        *,
        planning_order: tuple[str, ...] | None = None,
        independent_propositions: bool = False,
        ) -> GroundedResponseActPlanningBuild:
    """从 typed grounded Evidence 建立 ANSWER/CLARIFY/CONFLICT planning。"""
    if not isinstance(episode, GroundedAnswerEpisode):
        raise TypeError("response-act planning episode 类型错误")
    if episode.split != "train":
        raise GroundedResponseActPlanningError(
            "response-act planning 只接受 TRAIN episode")
    if episode.question.answer_plan.response_act not in {
            "ANSWER", "CLARIFY", "CONFLICT"}:
        raise GroundedResponseActPlanningError(
            "grounded planning response act 未注册")
    if (not isinstance(language_branch, ObjectIdentity)
            or language_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
        raise GroundedResponseActPlanningError("language branch 非法")

    aggregate = _aggregate_source(episode)
    response_scope = query_scope(
        episode.question.response_scope_id,
        parent=document_scope(aggregate),
    )
    source_ids = tuple(sorted({
        item.source_id for item in episode.question.evidence}))
    sources = {source_id: _source_ref(episode, source_id)
               for source_id in source_ids}
    source_bindings = tuple(
        GroundedEvidenceSourceBinding(source_id, sources[source_id])
        for source_id in source_ids
    )
    proposition_ids = tuple(sorted({
        item.proposition_id for item in episode.question.evidence}))
    bound = _bound_propositions(episode, aggregate, proposition_ids)
    shared_competition = _fingerprint(
        episode.episode_id,
        episode.question.typed_intent,
        domain="grounded.response.act.candidate.competition.v1",
    )
    candidate_bindings = []
    for proposition_id in proposition_ids:
        competition = (
            _fingerprint(
                episode.episode_id,
                episode.question.typed_intent,
                proposition_id,
                domain="grounded.answer.reference.candidate.competition.v1",
            )
            if independent_propositions else shared_competition
        )
        forming_sources = tuple(sorted({
            sources[item.source_id]
            for item in episode.question.evidence
            if item.proposition_id == proposition_id
        }, key=SourceRef.stable_key))
        proposition = bound[proposition_id]
        definition = EvidenceCandidateDefinition(
            proposition.template,
            competition,
            (CandidateBinding(
                concept_identity((_NAMESPACE, 18, 1)),
                concept_identity((_NAMESPACE, 18, *_fingerprint(
                    episode.episode_id,
                    proposition_id,
                    domain="grounded.response.act.candidate.binding.v1"))),
            ),),
            forming_sources,
        )
        records = _evidence_records(
            episode, proposition_id, definition, sources)
        state = LogicEvidenceState(
            any(item.stance == EVIDENCE_SUPPORT for item in records),
            any(item.stance == EVIDENCE_REFUTE for item in records),
        )
        candidate_bindings.append(GroundedResponseActCandidateBinding(
            proposition_id,
            definition,
            GenerationCandidate(
                proposition,
                state,
                aggregate,
                response_scope,
                records,
            ),
        ))
    by_proposition = {
        item.proposition_id: item.candidate for item in candidate_bindings}
    if planning_order is None:
        planning_order = proposition_ids
    if (not isinstance(planning_order, tuple)
            or planning_order != tuple(dict.fromkeys(planning_order))
            or set(planning_order) != set(proposition_ids)):
        raise GroundedResponseActPlanningError(
            "grounded planning order 必须精确覆盖全部 Proposition")
    candidates = tuple(by_proposition[item] for item in planning_order)
    goal = AnswerGenerationGoal(
        minimal_instruction_identity((_NAMESPACE, 19, *_fingerprint(
            episode.question.typed_intent,
            domain="grounded.response.act.goal.kind.v1"))),
        candidates[0].proposition,
        LogicEvidenceState(True, False),
        aggregate,
        response_scope,
        language_branch,
    )
    planning = GenerationPlanningRequest(goal, candidates)
    return GroundedResponseActPlanningBuild(
        episode,
        language_branch,
        aggregate,
        response_scope,
        source_bindings,
        tuple(candidate_bindings),
        planning,
    )


def compile_grounded_response_act_planning(
        episode: GroundedAnswerEpisode,
        language_branch: ObjectIdentity,
        ) -> GroundedResponseActPlanningBuild:
    """只接受 CLARIFY/CONFLICT，并建立真实 non-answer planning。"""
    if (not isinstance(episode, GroundedAnswerEpisode)
            or episode.question.answer_plan.response_act
            not in {"CLARIFY", "CONFLICT"}):
        raise GroundedResponseActPlanningError(
            "response-act planning 只接受 CLARIFY/CONFLICT")
    return _compile_grounded_generation_planning(episode, language_branch)


def compile_grounded_answer_planning(
        episode: GroundedAnswerEpisode,
        language_branch: ObjectIdentity,
        ) -> GroundedResponseActPlanningBuild:
    """只接受单命题 ANSWER episode，并建立 aggregate typed planning。"""
    if (not isinstance(episode, GroundedAnswerEpisode)
            or episode.question.answer_plan.response_act != "ANSWER"):
        raise GroundedResponseActPlanningError(
            "grounded answer planning 只接受 ANSWER")
    if len({item.proposition_id for item in episode.question.evidence}) != 1:
        raise GroundedResponseActPlanningError(
            "首轮 grounded answer planning 只接受单命题 episode")
    return _compile_grounded_generation_planning(episode, language_branch)


def compile_grounded_answer_reference_planning(
        episode: GroundedAnswerEpisode,
        language_branch: ObjectIdentity,
        ) -> GroundedResponseActPlanningBuild:
    """接受双命题 reference ANSWER，并保留课程声明的命题先后序。"""
    if (not isinstance(episode, GroundedAnswerEpisode)
            or episode.question.answer_plan.response_act != "ANSWER"
            or episode.reference_course is None):
        raise GroundedResponseActPlanningError(
            "grounded reference planning 只接受 reference ANSWER")
    course = episode.reference_course
    if (len(course.ordered_proposition_ids) != 2
            or course.ordered_proposition_ids
            != episode.question.answer_plan.ordered_claim_ids):
        raise GroundedResponseActPlanningError(
            "grounded reference planning 必须精确覆盖两个有序 claim")
    return _compile_grounded_generation_planning(
        episode,
        language_branch,
        planning_order=course.ordered_proposition_ids,
        independent_propositions=True,
    )


__all__ = [
    "GroundedEvidenceSourceBinding",
    "GroundedResponseActCandidateBinding",
    "GroundedResponseActPlanningBuild",
    "GroundedResponseActPlanningError",
    "compile_grounded_answer_planning",
    "compile_grounded_answer_reference_planning",
    "compile_grounded_response_act_planning",
]
