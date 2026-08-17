"""为 grounded reference 五层 actual choice 建立只读 claim 并接入 GG-02。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_generation_choice_outcome_bridge import (
    GenerationChoiceEpisodeAttribution,
    GenerationChoiceOutcomeBridge,
    GenerationChoiceUseAttribution,
    GenerationLayeredOutcomeReport,
    GenerationVerifierLayerRoute,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_episode_use import (
    GroundedAnswerReferenceFiveChoiceUses,
    GroundedAnswerReferenceLayerUse,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice_use import (
    GroundedAnswerReferenceChoiceUse,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_runtime_factory import (
    GroundedAnswerReferenceRunLocalInstallation,
)
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationReport,
    VerificationResult,
)


_NAMESPACE = 21030
REFERENCE_VERIFIER_NAMES = (
    "CONTENT_CANDIDATE_COVERAGE",
    "STRUCTURE_EXECUTION",
    "REFERENCE_CANDIDATE_SET_COMPLETENESS",
    "REFERENCE_ANTECEDENT_MEMBERSHIP",
    "REFERENCE_SCOPE_OWNER_VERSION",
    "REFERENCE_PAST_ONLY_ORDER",
    "REFERENCE_UNIQUE_RESOLUTION",
    "REFERENCE_SURFACE_AGREEMENT",
    "LEXICAL_DIRECT_EXECUTION",
    "TASK_STANCE_EXECUTION",
)
_EXPECTED_LAYER = {
    "CONTENT_CANDIDATE_COVERAGE": "CONTENT_CHOICE",
    "STRUCTURE_EXECUTION": "PROPOSITION_STRUCTURE_CHOICE",
    "REFERENCE_CANDIDATE_SET_COMPLETENESS": "DISCOURSE_REFERENCE_CHOICE",
    "REFERENCE_ANTECEDENT_MEMBERSHIP": "DISCOURSE_REFERENCE_CHOICE",
    "REFERENCE_SCOPE_OWNER_VERSION": "DISCOURSE_REFERENCE_CHOICE",
    "REFERENCE_PAST_ONLY_ORDER": "DISCOURSE_REFERENCE_CHOICE",
    "REFERENCE_UNIQUE_RESOLUTION": "DISCOURSE_REFERENCE_CHOICE",
    "REFERENCE_SURFACE_AGREEMENT": "DISCOURSE_REFERENCE_CHOICE",
    "LEXICAL_DIRECT_EXECUTION": "LEXICAL_REALIZATION_CHOICE",
    "TASK_STANCE_EXECUTION": "COMMUNICATIVE_TASK_CHOICE",
}


# object-model: exception
class GroundedAnswerReferenceVerificationError(ValueError):
    """R-04 verifier、claim、attribution 或 GG-02 连接不完整。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedAnswerReferenceVerificationError(
            f"{where} 必须是非空严格整数 tuple")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceVerifierRoute:
    """一个语义明确且只能归因到单层 choice 的 verifier route。"""

    name: str
    route: GenerationVerifierLayerRoute

    def __post_init__(self) -> None:
        if self.name not in REFERENCE_VERIFIER_NAMES:
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference verifier name 未注册")
        if not isinstance(self.route, GenerationVerifierLayerRoute):
            raise TypeError("grounded reference verifier route 类型错误")
        if self.route.choice_kinds != (_EXPECTED_LAYER[self.name],):
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference verifier 越过声明层")

    def stable_key(self) -> tuple[int, ...]:
        """返回语义 ordinal 与 GG-02 route 完整键。"""
        return (
            REFERENCE_VERIFIER_NAMES.index(self.name) + 1,
            *_packed(self.route.stable_key()),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceVerifierProtocol:
    """冻结十条互不跨层的 R-04 verifier 路由。"""

    routes: tuple[GroundedAnswerReferenceVerifierRoute, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.routes, tuple)
                or any(not isinstance(
                    item, GroundedAnswerReferenceVerifierRoute)
                    for item in self.routes)
                or tuple(item.name for item in self.routes)
                != REFERENCE_VERIFIER_NAMES):
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference verifier protocol 覆盖或顺序漂移")
        keys = tuple(
            (item.route.dimension, item.route.verifier)
            for item in self.routes)
        if len(set(keys)) != len(keys):
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference verifier dimension/verifier 重复")

    def by_name(self) -> dict[str, GroundedAnswerReferenceVerifierRoute]:
        """按冻结语义名返回 route 映射。"""
        return {item.name: item for item in self.routes}

    def gg02_routes(self) -> tuple[GenerationVerifierLayerRoute, ...]:
        """返回可直接注入既有 GG-02 bridge 的十条 route。"""
        return tuple(item.route for item in self.routes)


def build_grounded_answer_reference_verifier_protocol(
        key_prefix: tuple[int, ...],
        ) -> GroundedAnswerReferenceVerifierProtocol:
    """从调用方注入键建立确定性 R-04 protocol，不写全局注册表。"""
    _strict_key(key_prefix, where="reference verifier key prefix")
    routes = tuple(
        GroundedAnswerReferenceVerifierRoute(
            name,
            GenerationVerifierLayerRoute(
                ProtocolKey((*key_prefix, 1, ordinal)),
                ProtocolKey((*key_prefix, 2, ordinal)),
                (_EXPECTED_LAYER[name],),
            ),
        )
        for ordinal, name in enumerate(REFERENCE_VERIFIER_NAMES, start=1)
    )
    return GroundedAnswerReferenceVerifierProtocol(routes)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceVerifierClaim:
    """一个 verifier 对一个 exact layer Use 的无损只读证明路径。"""

    name: str
    choice_kind: str
    claim_key: LosslessIntegerKey
    evidence_keys: tuple[LosslessIntegerKey, ...]

    def __post_init__(self) -> None:
        if self.name not in REFERENCE_VERIFIER_NAMES:
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference claim name 未注册")
        if self.choice_kind != _EXPECTED_LAYER[self.name]:
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference claim 跨 choice layer")
        if not isinstance(self.claim_key, LosslessIntegerKey):
            raise TypeError("grounded reference claim key 类型错误")
        if (not isinstance(self.evidence_keys, tuple)
                or not self.evidence_keys
                or any(not isinstance(item, LosslessIntegerKey)
                       for item in self.evidence_keys)):
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference claim 缺 evidence path")
        if len(set(self.evidence_keys)) != len(self.evidence_keys):
            raise GroundedAnswerReferenceVerificationError(
                "grounded reference claim evidence path 重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回语义、层、claim 与全部实际 evidence 键。"""
        values = [
            REFERENCE_VERIFIER_NAMES.index(self.name) + 1,
            CHOICE_KINDS.index(self.choice_kind),
            *_packed(self.claim_key.components),
            len(self.evidence_keys),
        ]
        for evidence in self.evidence_keys:
            values.extend(_packed(evidence.components))
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceLayerVerification:
    """十条分层 result、适用 claim 与只读 report。"""

    protocol: GroundedAnswerReferenceVerifierProtocol
    uses: GroundedAnswerReferenceFiveChoiceUses
    claims: tuple[GroundedAnswerReferenceVerifierClaim, ...]
    report: VerificationReport

    def __post_init__(self) -> None:
        if not isinstance(
                self.protocol, GroundedAnswerReferenceVerifierProtocol):
            raise TypeError("reference layer verification protocol 类型错误")
        if not isinstance(self.uses, GroundedAnswerReferenceFiveChoiceUses):
            raise TypeError("reference layer verification uses 类型错误")
        if (not isinstance(self.claims, tuple)
                or any(not isinstance(
                    item, GroundedAnswerReferenceVerifierClaim)
                    for item in self.claims)):
            raise TypeError("reference layer verification claims 类型错误")
        if not isinstance(self.report, VerificationReport):
            raise TypeError("reference layer verification report 类型错误")
        if not self.report.read_only:
            raise GroundedAnswerReferenceVerificationError(
                "reference layer verification 必须只读")
        route_keys = {
            (item.route.dimension, item.route.verifier)
            for item in self.protocol.routes}
        result_keys = {
            (item.dimension, item.verifier) for item in self.report.results}
        if route_keys != result_keys:
            raise GroundedAnswerReferenceVerificationError(
                "reference layer verification 未逐 route 返回结果")
        claim_keys = tuple(item.claim_key for item in self.claims)
        if len(set(claim_keys)) != len(claim_keys):
            raise GroundedAnswerReferenceVerificationError(
                "reference verifier claim 跨维度复用")
        reported = {
            key
            for result in self.report.results
            for key in result.claim_keys}
        if reported != {item.components for item in claim_keys}:
            raise GroundedAnswerReferenceVerificationError(
                "reference verifier report 与 claim artifact 漂移")
        by_kind = {kind: 0 for kind in CHOICE_KINDS}
        for claim in self.claims:
            by_kind[claim.choice_kind] += 1
        if any(count == 0 for count in by_kind.values()):
            raise GroundedAnswerReferenceVerificationError(
                "reference verifier claims 未覆盖五层")

    def claims_for(
            self,
            choice_kind: str,
            ) -> tuple[GroundedAnswerReferenceVerifierClaim, ...]:
        """返回一个 choice layer 独占的有序 claim。"""
        if choice_kind not in CHOICE_KINDS:
            raise GroundedAnswerReferenceVerificationError(
                "reference verification choice kind 未注册")
        return tuple(
            item for item in self.claims
            if item.choice_kind == choice_kind)


def _record_by_kind(
        uses: GroundedAnswerReferenceFiveChoiceUses,
        ) -> dict[
            str,
            GroundedAnswerReferenceLayerUse | GroundedAnswerReferenceChoiceUse,
        ]:
    """按 GG-01 层名返回同次五个 actual Use 记录。"""
    records = (
        uses.content,
        uses.structure,
        uses.reference,
        uses.lexical,
        uses.task,
    )
    return {item.choice_before.choice_kind: item for item in records}


def _claim(
        name: str,
        record: GroundedAnswerReferenceLayerUse | GroundedAnswerReferenceChoiceUse,
        evidence: tuple[tuple[int, ...], ...],
        ) -> GroundedAnswerReferenceVerifierClaim:
    """把 actual choice/Use 与本维度 evidence 无损编入唯一 claim。"""
    if not evidence:
        raise GroundedAnswerReferenceVerificationError(
            "reference verifier applicable claim 缺 evidence")
    evidence_keys = tuple(LosslessIntegerKey(item) for item in evidence)
    values = [
        _NAMESPACE,
        10,
        REFERENCE_VERIFIER_NAMES.index(name) + 1,
        *_packed(record.choice_after.candidate.stable_key()),
        *_packed(record.use.stable_key()),
        len(evidence_keys),
    ]
    for key in evidence_keys:
        values.extend(_packed(key.components))
    return GroundedAnswerReferenceVerifierClaim(
        name,
        record.choice_before.choice_kind,
        LosslessIntegerKey(tuple(values)),
        evidence_keys,
    )


def _result(
        protocol: GroundedAnswerReferenceVerifierProtocol,
        name: str,
        record: GroundedAnswerReferenceLayerUse | GroundedAnswerReferenceChoiceUse,
        supported: bool | None,
        evidence: tuple[tuple[int, ...], ...],
        source,
        scope,
        ) -> tuple[VerificationResult, GroundedAnswerReferenceVerifierClaim | None]:
    """形成 applicable support/refute 或显式 N/A 的只读结果。"""
    route = protocol.by_name()[name].route
    ordinal = REFERENCE_VERIFIER_NAMES.index(name) + 1
    if supported is None:
        return VerificationResult(
            route.dimension,
            route.verifier,
            APPLICABILITY_NOT_APPLICABLE,
            VERDICT_UNKNOWN,
            detail=(_NAMESPACE, 30, ordinal, 0),
        ), None
    claim = _claim(name, record, evidence)
    result = VerificationResult(
        route.dimension,
        route.verifier,
        APPLICABILITY_APPLICABLE,
        VERDICT_SUPPORT if supported else VERDICT_REFUTE,
        (claim.claim_key.components,),
        detail=(_NAMESPACE, 30, ordinal, int(supported), len(evidence)),
        source=source,
        scope=scope,
    )
    return result, claim


def verify_grounded_answer_reference_layers(
        protocol: GroundedAnswerReferenceVerifierProtocol,
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        uses: GroundedAnswerReferenceFiveChoiceUses,
        ) -> GroundedAnswerReferenceLayerVerification:
    """只从 actual run/selection/parser/adoption 形成十条分层结论。"""
    if not isinstance(protocol, GroundedAnswerReferenceVerifierProtocol):
        raise TypeError("reference layer verifier protocol 类型错误")
    if not isinstance(
            installation, GroundedAnswerReferenceRunLocalInstallation):
        raise TypeError("reference layer verifier installation 类型错误")
    if not isinstance(run, QuestionAnswerRun):
        raise TypeError("reference layer verifier run 类型错误")
    if not isinstance(uses, GroundedAnswerReferenceFiveChoiceUses):
        raise TypeError("reference layer verifier uses 类型错误")
    if (uses.content.installation is not installation
            or uses.content.run is not run
            or run.generation is None
            or run.generation.surface is None
            or run.postcheck is None
            or run.postcheck.parsed.observation is None):
        raise GroundedAnswerReferenceVerificationError(
            "reference layer verifier 缺同次完整 generation/postcheck")
    source = installation.reference_selection.source
    scope = installation.reference_selection.scope
    records = _record_by_kind(uses)
    planning = installation.compilation.planning
    candidates = installation.compilation.ordered_candidates
    expected_candidate_keys = tuple(item.stable_key() for item in candidates)
    observation = run.postcheck.parsed.observation
    recovered_keys = tuple(
        item.candidate_key for item in observation.propositions)
    syntax = run.generation.surface.preview.request.structure.syntax
    execution = run.generation.surface.preview.request.execution
    recovery = uses.reference.recovery
    reference_selection = installation.reference_selection
    reference_sentence = tuple(
        item for item in syntax.sentences
        if item.address == recovery.sentence)
    if len(reference_sentence) != 1:
        raise GroundedAnswerReferenceVerificationError(
            "reference verifier 未恢复唯一 referring sentence")
    reference_ordinal = reference_sentence[0].ordinal
    prior_keys = {
        key
        for sentence in syntax.sentences
        if sentence.ordinal < reference_ordinal
        for key in sentence.proposition_keys
    }
    eligible = tuple(
        item for item in candidates if item.stable_key() in prior_keys)
    ordinal_by_candidate = {
        key: sentence.ordinal
        for sentence in syntax.sentences
        for key in sentence.proposition_keys
    }
    selected_antecedent = reference_selection.selected_antecedent
    reference_slot = tuple(
        item for item in run.generation.surface.preview.slots
        if (item.directive.sentence == recovery.sentence
            and item.value.slot == recovery.slot)
    )
    if len(reference_slot) != 1:
        raise GroundedAnswerReferenceVerificationError(
            "reference verifier 未恢复唯一 actual slot")
    slot = reference_slot[0]

    evaluations = []
    claims = []

    def add(
            name: str,
            supported: bool | None,
            evidence: tuple[tuple[int, ...], ...],
            ) -> None:
        """追加一个按协议分层的结果和可选 claim。"""
        result, claim = _result(
            protocol,
            name,
            records[_EXPECTED_LAYER[name]],
            supported,
            evidence,
            source,
            scope,
        )
        evaluations.append(result)
        if claim is not None:
            claims.append(claim)

    content_supported = (
        uses.content.layer.selected_candidate_keys == expected_candidate_keys
        and set(recovered_keys) == set(expected_candidate_keys)
        and observation.source == source
        and observation.scope == scope
    )
    add(
        "CONTENT_CANDIDATE_COVERAGE",
        content_supported,
        (
            uses.content.layer.stable_key(),
            observation.stable_key(),
        ),
    )

    sentence_candidate_keys = tuple(
        key for sentence in syntax.sentences for key in sentence.proposition_keys)
    structure_supported = (
        len(syntax.sentences) == 2
        and sentence_candidate_keys == expected_candidate_keys
        and execution.complete
        and all(sentence.source == source and sentence.scope == scope
                for sentence in syntax.sentences)
    )
    structure_evidence = tuple(
        item.stable_key() for item in syntax.sentences)
    add(
        "STRUCTURE_EXECUTION",
        structure_supported,
        (*structure_evidence, execution.stable_key()),
    )

    add(
        "REFERENCE_CANDIDATE_SET_COMPLETENESS",
        eligible == reference_selection.antecedent_candidates,
        (
            tuple(
                value
                for candidate in eligible
                for value in _packed(candidate.stable_key())),
            reference_selection.stable_key(),
        ),
    )
    add(
        "REFERENCE_ANTECEDENT_MEMBERSHIP",
        (
            selected_antecedent in eligible
            and reference_selection.selected_antecedent
            in reference_selection.antecedent_candidates
        ),
        (
            selected_antecedent.stable_key(),
            reference_selection.choice.stable_key(),
        ),
    )
    boundary_supported = (
        recovery.source == source
        and recovery.scope == scope
        and recovery.sentence.source == source
        and recovery.sentence.scope == scope
        and source.owner == scope.owner
        and source.versions == scope.versions
        and all(item.source == source and item.scope == scope
                for item in candidates)
    )
    add(
        "REFERENCE_SCOPE_OWNER_VERSION",
        boundary_supported,
        (
            recovery.stable_key(),
            source.stable_key(),
            scope.stable_key(),
        ),
    )
    past_supported = (
        selected_antecedent.stable_key() in ordinal_by_candidate
        and ordinal_by_candidate[selected_antecedent.stable_key()]
        < reference_ordinal
    )
    add(
        "REFERENCE_PAST_ONLY_ORDER",
        past_supported,
        (
            (_NAMESPACE, 40,
             ordinal_by_candidate.get(selected_antecedent.stable_key(), 0),
             reference_ordinal),
            recovery.sentence.stable_key(),
        ),
    )
    if reference_selection.selected.strategy == "ANTECEDENT_REFERENCE":
        proposal = slot.reference
        unique_supported = (
            proposal is not None
            and len(proposal.result.options) == 1
            and proposal.result.selected is not None
            and proposal.result.selected.value
            == selected_antecedent.proposition.template
            and uses.reference.reference_adoption is not None
            and uses.reference.reference_adoption.proposal == proposal
        )
        unique_evidence = (
            ((_NAMESPACE, 41, 1, 0) if proposal is None
             else proposal.stable_key()),
            ((_NAMESPACE, 41, 2, 0)
             if uses.reference.reference_adoption is None
             else uses.reference.reference_adoption.stable_key()),
        )
        add(
            "REFERENCE_UNIQUE_RESOLUTION",
            unique_supported,
            unique_evidence,
        )
    else:
        add("REFERENCE_UNIQUE_RESOLUTION", None, ())

    parse_request = GenerationSurfaceParseRequest.from_execution(
        run.generation)
    direct = uses.reference.direct_adoption
    surface_supported = (
        recovery.parse_request_key == parse_request.stable_key()
        and slot.surface is not None
        and slot.representation == recovery.representation
        and slot.surface.stable_key() == recovery.surface_proposal_key
        and direct.proposal == slot.surface
        and direct.use_key == slot.directive.surface_use_key
        and (
            (
                reference_selection.selected.strategy
                == "ANTECEDENT_REFERENCE"
                and slot.reference is not None
                and uses.reference.reference_adoption is not None
                and uses.reference.reference_adoption.proposal
                == slot.reference
                and recovery.reference_proposal_key
                == slot.reference.stable_key()
            )
            or (
                reference_selection.selected.strategy
                == "EXPLICIT_REPETITION"
                and slot.reference is None
                and uses.reference.reference_adoption is None
                and not recovery.reference_proposal_key
            )
        )
    )
    add(
        "REFERENCE_SURFACE_AGREEMENT",
        surface_supported,
        (
            recovery.stable_key(),
            direct.stable_key(),
            parse_request.stable_key(),
        ),
    )

    non_reference_slots = tuple(
        item for item in run.generation.surface.preview.slots
        if item.value.slot != installation.compilation.reference_slot)
    direct_adoption_keys = []
    lexical_supported = bool(non_reference_slots)
    for item in non_reference_slots:
        matches = tuple(
            adoption for adoption in run.generation.surface.adoptions
            if (adoption.sentence == item.directive.sentence
                and adoption.slot == item.value.slot
                and adoption.proposal == item.surface
                and adoption.use_key == item.directive.surface_use_key)
        )
        lexical_supported = lexical_supported and len(matches) == 1
        if matches:
            direct_adoption_keys.append(matches[0].stable_key())
    add(
        "LEXICAL_DIRECT_EXECUTION",
        lexical_supported,
        tuple(direct_adoption_keys) or ((_NAMESPACE, 42, 0),),
    )

    task_supported = (
        run.selection is not None
        and uses.task.layer.payload == run.selection.stable_key()
        and uses.task.choice_before.selected_object == run.selection.stance
        and observation.stance == run.selection.stance
        and run.status == run.selection.stance
    )
    add(
        "TASK_STANCE_EXECUTION",
        task_supported,
        (
            uses.task.layer.stable_key(),
            observation.stance.stable_key(),
        ),
    )

    ordered_results = tuple(sorted(
        evaluations,
        key=lambda item: (
            item.dimension.stable_key(), item.verifier.stable_key()),
    ))
    ordered_claims = tuple(sorted(
        claims,
        key=lambda item: item.claim_key.components,
    ))
    return GroundedAnswerReferenceLayerVerification(
        protocol,
        uses,
        ordered_claims,
        VerificationReport(True, ordered_results),
    )


def build_grounded_answer_reference_attribution(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        verification: GroundedAnswerReferenceLayerVerification,
        ) -> GenerationChoiceEpisodeAttribution:
    """把真实五层 choice/Use 与各自 claim 绑定到同一 query/generation。"""
    if not isinstance(
            installation, GroundedAnswerReferenceRunLocalInstallation):
        raise TypeError("reference attribution installation 类型错误")
    if not isinstance(run, QuestionAnswerRun):
        raise TypeError("reference attribution run 类型错误")
    if not isinstance(
            verification, GroundedAnswerReferenceLayerVerification):
        raise TypeError("reference attribution verification 类型错误")
    uses = verification.uses
    if (uses.content.installation is not installation
            or uses.content.run is not run
            or run.query is None
            or run.generation is None):
        raise GroundedAnswerReferenceVerificationError(
            "reference attribution 缺同次 query/generation")
    query_key = LosslessIntegerKey(run.query.stable_key())
    generation_key = LosslessIntegerKey(run.generation.stable_key())
    source = installation.reference_selection.source
    scope = installation.reference_selection.scope
    records = _record_by_kind(uses)
    attributions = []
    for choice_kind in CHOICE_KINDS:
        record = records[choice_kind]
        claims = tuple(sorted(
            (
                item.claim_key
                for item in verification.claims_for(choice_kind)
            ),
            key=lambda item: item.components,
        ))
        attributions.append(GenerationChoiceUseAttribution(
            record.choice_after,
            record.use,
            query_key,
            generation_key,
            claims,
            source,
            scope,
        ))
    return GenerationChoiceEpisodeAttribution(
        LosslessIntegerKey(
            installation.reference_selection.context.stable_key()),
        query_key,
        generation_key,
        source,
        scope,
        tuple(attributions),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceGG02Run:
    """同次 R-04 verification、attribution 与首次 GG-02 零写结果。"""

    verification: GroundedAnswerReferenceLayerVerification
    attribution: GenerationChoiceEpisodeAttribution
    outcome: GenerationLayeredOutcomeReport

    def __post_init__(self) -> None:
        if not isinstance(
                self.verification, GroundedAnswerReferenceLayerVerification):
            raise TypeError("reference GG-02 verification 类型错误")
        if not isinstance(
                self.attribution, GenerationChoiceEpisodeAttribution):
            raise TypeError("reference GG-02 attribution 类型错误")
        if not isinstance(self.outcome, GenerationLayeredOutcomeReport):
            raise TypeError("reference GG-02 outcome 类型错误")
        if self.outcome.episode != self.attribution:
            raise GroundedAnswerReferenceVerificationError(
                "reference GG-02 outcome 替换了 attribution")
        if (self.outcome.host_learning_write_count != 0
                or self.outcome.teacher_call_count != 0
                or self.outcome.assessment_consumer_status
                != "REQUIRED_NOT_CONNECTED"):
            raise GroundedAnswerReferenceVerificationError(
                "reference GG-02 越过零写/assessment 边界")


def run_grounded_answer_reference_gg02(
        protocol: GroundedAnswerReferenceVerifierProtocol,
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        uses: GroundedAnswerReferenceFiveChoiceUses,
        ) -> GroundedAnswerReferenceGG02Run:
    """先完成逐层 verifier/attribution，再首次调用既有 GG-02 bridge。"""
    verification = verify_grounded_answer_reference_layers(
        protocol, installation, run, uses)
    attribution = build_grounded_answer_reference_attribution(
        installation, run, verification)
    outcome = GenerationChoiceOutcomeBridge(
        protocol.gg02_routes()).compile(
            attribution, verification.report)
    return GroundedAnswerReferenceGG02Run(
        verification,
        attribution,
        outcome,
    )


__all__ = [
    "GroundedAnswerReferenceGG02Run",
    "GroundedAnswerReferenceLayerVerification",
    "GroundedAnswerReferenceVerificationError",
    "GroundedAnswerReferenceVerifierClaim",
    "GroundedAnswerReferenceVerifierProtocol",
    "GroundedAnswerReferenceVerifierRoute",
    "REFERENCE_VERIFIER_NAMES",
    "build_grounded_answer_reference_attribution",
    "build_grounded_answer_reference_verifier_protocol",
    "run_grounded_answer_reference_gg02",
    "verify_grounded_answer_reference_layers",
]
