"""公开 typed 对话课程的 production generation owner。

本模块把当前 S-02 semantic course 形成的 ``GenerationPlanningRequest`` 交给
已有 G-00..G-03 production 设施。表层 claim 只能来自当前 typed payload 的
显式候选字段；没有唯一候选或没有登记表层时 fail-closed，不回退旧 generate。
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
    GenerationSourceRequirement,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisLedger
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
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
from pure_integer_ai.cognition.shared.types import InputPayload, ObserveResult
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationInstallation,
    ProductionGenerationRequestDecision,
    ProductionGenerationRuntime,
)
from pure_integer_ai.experiments.language_semantic_runtime import (
    SemanticCourseGenerationRequestMapper,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_runtime import (
    ProductionGenerationAliasRuntimeFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records_from_payload,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerClaimInput,
    GroundedAnswerConnectorTarget,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    GroundedAnswerSurfaceModel,
    learn_grounded_answer_surface_model,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalBuild,
    GroundedAnswerRunLocalComponents,
    GroundedAnswerRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_verification import (
    GroundedAnswerEvidenceSourceVerifier,
    GroundedAnswerStructureVerifier,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRun,
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationPostcheckMapper,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.language_generation_connector_candidate import (
    LanguageConnectorCandidateProtocol,
    LanguageConnectorCandidateRuntime,
    CANDIDATE_PERSISTENCE_TRAINING,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageConnectorGraphPredicates,
    LanguageGenerationConnectorGraph,
)
from pure_integer_ai.experiments.language_generation_connector_stage4 import (
    LanguageConnectorSignalRoute,
    LanguageConnectorStage4Policy,
    LanguageConnectorStage4Report,
    LanguageConnectorStage4Runtime,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_SUPPORT,
    VERDICT_REFUTE,
)
from pure_integer_ai.experiments.train_context import TrainContext


_NAMESPACE = (21405, 1)
_REPRESENTATION_FAMILY = (21405, 2, 1)
_CLAIM_HASHER = Hasher("typed.dialogue.generation.claim.v1")
_FORMING_HASHER = Hasher("typed.dialogue.generation.stage4.forming.v1")
_STAGE4_ACTIVE_PURPOSE = minimal_instruction_identity((*_NAMESPACE, 19, 1))
_STAGE4_TRIAL_PURPOSE = minimal_instruction_identity((*_NAMESPACE, 19, 2))
_STAGE4_VERIFIER_SOURCE = SourceRef(
    21405, 60001, 0, GLOBAL_OWNER_SCOPE, VersionBundle())

_GENERATION_PROTOCOL_CACHE_LIMIT = 16
_GENERATION_PROTOCOL_CACHE: OrderedDict[
    tuple[int, ...], tuple[object, ...]] = OrderedDict()


def _instruction_series(group: int, count: int, branch):
    """在当前 LanguageBranch owner/version 中建立 run-local identities。"""
    return tuple(
        minimal_instruction_identity(
            (*_NAMESPACE, group, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, count + 1)
    )


def _generation_protocols(branch):
    """建立 grounded connector 所需的 G-00..G-04 协议。"""
    branch_key = branch.stable_key()
    cached = _GENERATION_PROTOCOL_CACHE.get(branch_key)
    if cached is not None:
        _GENERATION_PROTOCOL_CACHE.move_to_end(branch_key)
        return cached
    content = AnswerContentProtocol(*_instruction_series(10, 5, branch))
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(
            content,
            EvidenceAnswerPolicyProtocol(*_instruction_series(11, 4, branch)),
        ),
    )
    plan = __import__(
        "pure_integer_ai.cognition.shared.generation_plan",
        fromlist=["GenerationPlanProtocol"],
    ).GenerationPlanProtocol(*_instruction_series(12, 10, branch))
    structure = GenerationStructureLayerProtocol(
        *_instruction_series(13, 3, branch))
    surface = GenerationSurfaceProtocol(*_instruction_series(14, 9, branch))
    postcheck = GenerationPostcheckProtocol(
        *(ProtocolKey((*_NAMESPACE, 15, index)) for index in range(1, 13)),
        *_instruction_series(16, 15, branch),
    )
    question = QuestionAnswerProtocol(*_instruction_series(17, 3, branch))
    result = (content, selector, plan, structure, surface, postcheck, question)
    _GENERATION_PROTOCOL_CACHE[branch_key] = result
    _GENERATION_PROTOCOL_CACHE.move_to_end(branch_key)
    while len(_GENERATION_PROTOCOL_CACHE) > _GENERATION_PROTOCOL_CACHE_LIMIT:
        _GENERATION_PROTOCOL_CACHE.popitem(last=False)
    return result


def _claim_text(payload: Any) -> str | None:
    """从 typed payload 的登记候选字段恢复唯一当前 claim 表层。"""
    if not isinstance(payload, dict):
        return None
    propositions = payload.get("candidate_propositions")
    if isinstance(propositions, list) and len(propositions) == 1:
        # Adoption/postcheck records register their visible proposition as
        # context_surface rather than a generated surface fragment.  It is
        # still an explicit source field; no text is inferred from ids.
        value = payload.get("context_surface")
        if isinstance(value, str) and value.strip() == value and value:
            return value
    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) != 1:
        rows = payload.get("choice_candidates")
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        for key in ("surface_fragment", "claim_text", "fragment"):
            value = rows[0].get(key)
            if isinstance(value, str) and value.strip() == value and value:
                return value
    surfaces = payload.get("surface_candidates")
    if isinstance(surfaces, list) and len(surfaces) == 1:
        row = surfaces[0]
        if isinstance(row, dict):
            value = row.get("fragment")
            if isinstance(value, str) and value.strip() == value and value:
                return value
    return None


@dataclass(frozen=True, slots=True)
class _GenerationSpec:
    """一次请求对应的来源化 claim 和候选证据。"""

    claim: str
    candidate_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]


class _DynamicPostcheckMapper(ProductionGenerationPostcheckMapper):
    """把实际 typed execution 映射为同次来源要求，不读取标签或词面。"""

    def __init__(self) -> None:
        self._prefix = (*_NAMESPACE, 28, 1)

    def build(self, ctx, item, input_payload, observation,
              execution: TypedGenerationExecution) -> GenerationPostcheckRequest:
        if not isinstance(execution, TypedGenerationExecution):
            raise TypeError("typed postcheck mapper 需要实际 execution")
        structure = execution.surface.preview.request.structure
        candidates = {
            candidate.stable_key(): candidate
            for candidate in execution.plan.request.candidates
        }
        emitted_keys = {
            key
            for sentence in structure.syntax.sentences
            for key in sentence.proposition_keys
        }
        requirements = []
        for index, proposition in enumerate(
                (item for item in structure.propositions.propositions
                 if item.candidate_key in emitted_keys), start=1):
            candidate = candidates.get(proposition.candidate_key)
            if candidate is None:
                raise ValueError("typed postcheck 缺少同次 candidate")
            requirements.append(GenerationSourceRequirement(
                proposition.candidate_key,
                proposition.source,
                proposition.scope,
                True,
                True,
                (*self._prefix, index),
                candidate.citation_sources,
            ))
        return GenerationPostcheckRequest(execution, (), tuple(requirements))


class _DynamicPostcheckRuntime(GenerationPostcheckRuntime):
    """将每次动态安装的 parser/verifier 暂存到对应 execution。"""

    def __init__(self) -> None:
        self._runs: dict[tuple[int, ...], GenerationPostcheckRuntime] = {}

    def bind(self, execution: TypedGenerationExecution,
             runtime: GenerationPostcheckRuntime) -> None:
        if not isinstance(execution, TypedGenerationExecution):
            raise TypeError("typed postcheck bind execution 类型错误")
        if not isinstance(runtime, GenerationPostcheckRuntime):
            raise TypeError("typed postcheck bind runtime 类型错误")
        key = execution.stable_key()
        if key in self._runs:
            raise RuntimeError("同一 typed execution 重复绑定 G-04")
        self._runs[key] = runtime

    def run(self, request: GenerationPostcheckRequest) -> GenerationPostcheckRun:
        if not isinstance(request, GenerationPostcheckRequest):
            raise TypeError("typed postcheck runtime request 类型错误")
        key = request.execution.stable_key()
        runtime = self._runs.pop(key, None)
        if runtime is None:
            raise RuntimeError("typed postcheck 缺少同次动态 parser/verifier")
        return runtime.run(request)


class _RequestMapper:
    """把同次 semantic course request 交给动态 typed executor。"""

    def __init__(self) -> None:
        self.semantic = SemanticCourseGenerationRequestMapper()
        self.specs: dict[tuple[int, ...], _GenerationSpec] = {}

    def build(self, ctx: TrainContext, item: CollectedItem,
              input_payload: InputPayload,
              observation: ObserveResult) -> ProductionGenerationRequestDecision:
        decision = self.semantic.build(ctx, item, input_payload, observation)
        request = decision.request
        if request is None:
            return decision
        payload = item.typed_payload
        raw = None if payload is None else payload.to_value()
        claim = _claim_text(raw)
        if len(request.candidates) != 1 or claim is None:
            # 多候选及没有显式 claim 的课程保留 typed request 之外的无请求结论；
            # 这些样本等待多命题 surface owner，不会伪造单候选回答。
            return ProductionGenerationRequestDecision(
                decision.reason,
                (*decision.trace, 0, len(request.candidates)),
                None,
            )
        candidate = request.candidates[0]
        self.specs[request.stable_key()] = _GenerationSpec(
            claim,
            candidate.stable_key(),
            tuple(item.stable_key() for item in candidate.evidence),
        )
        return ProductionGenerationRequestDecision(
            decision.reason,
            (*decision.trace, 1, len(request.candidates), len(claim)),
            request,
        )


class _DynamicExecutor(TypedGenerationExecutor):
    """为每个真实 request 建立 run-local grounded connector executor。"""

    def __init__(self, ctx: TrainContext, model: GroundedAnswerSurfaceModel,
                 pack, mapper: _RequestMapper,
                 postcheck_runtime: _DynamicPostcheckRuntime,
                 stage4_owner: _TypedStage4Owner | None = None) -> None:
        # TypedGenerationExecutor 的运行时类型检查由 ProductionRuntime 完成；
        # 本 owner 的 planner/renderer 随当前 branch 和候选动态装配。
        self.ctx = ctx
        self.model = model
        self.pack = pack
        self.mapper = mapper
        self.postcheck_runtime = postcheck_runtime
        self.stage4_owner = stage4_owner
        # S-07 facade construction materializes the same branch protocol on
        # every request; retain only immutable branch-local facades here.
        self._lifecycle_cache: dict[
            tuple[int, ...], StructureOrderLifecycleGraph] = {}

    def execute(self, request: GenerationPlanningRequest) -> TypedGenerationExecution:
        spec = self.mapper.specs.get(request.stable_key())
        if spec is None:
            raise RuntimeError("typed generation request 缺少同次 claim spec")
        if len(request.candidates) != 1 or request.goal.target_branch is None:
            raise RuntimeError("typed generation owner 只接受唯一候选和目标语言分支")
        candidate = request.candidates[0]
        if candidate.stable_key() != spec.candidate_key:
            raise RuntimeError("typed generation candidate 与 mapper spec 漂移")
        branch = request.goal.target_branch
        claim = GroundedAnswerClaimInput(spec.claim)
        target = GroundedAnswerConnectorTarget(
            candidate.proposition, branch, _REPRESENTATION_FAMILY)
        content, selector, plan_protocol, structure_protocol, surface_protocol, postcheck_protocol, question_protocol = (
            _generation_protocols(branch))
        branch_key = branch.stable_key()
        lifecycle = self._lifecycle_cache.get(branch_key)
        if lifecycle is None:
            lifecycle = _build_lifecycle_for_branch(self.ctx, branch)
            self._lifecycle_cache[branch_key] = lifecycle
        compilation = compile_grounded_answer_connectors(
            self.model, claim, target, surface_protocol)
        # 选择模型中稳定排序的首个单 claim ANSWER pattern；其 literal/claim
        # 均由公开 TRAIN course 形成，当前 claim 仅作为 visible Evidence filler。
        variant = compilation.variants[0]
        connector_hypothesis = candidate.hypotheses[0]
        connector_purpose = _STAGE4_ACTIVE_PURPOSE
        if self.stage4_owner is not None:
            if not self.stage4_owner.enabled:
                self.stage4_owner = None
        if self.stage4_owner is not None:
            connector_hypothesis, connector_purpose = self.stage4_owner.purpose_for_template(
                variant.template, compilation.value_protocol)
        renderer_identity = _instruction_series(18, 1, branch)[0]
        attribution = GenerationSurfaceAttribution(
            variant.template.connector,
            connector_hypothesis,
            connector_purpose,
        )
        alias_factory = ProductionGenerationAliasRuntimeFactory(
            self.pack,
            self.ctx,
            visible_evidence_keys=spec.evidence_keys,
        )
        components = GroundedAnswerRunLocalComponents(
            selector,
            plan_protocol,
            structure_protocol,
            alias_factory,
            UnicodeRepresentationRenderer(
                _REPRESENTATION_FAMILY, renderer_identity),
            renderer_identity,
            postcheck_protocol,
            GroundedAnswerStructureVerifier(
                *_instruction_series(20, 2, branch)),
            GroundedAnswerEvidenceSourceVerifier(
                *_instruction_series(21, 2, branch)),
            question_protocol,
            EvidenceQuestionPostcheckMapper(
                (*_NAMESPACE, 22, 1),
                citation_required=True,
                trust_required=True,
            ),
            surface_attributions=(attribution,),
        )
        installation = GroundedAnswerRunLocalFactory(
            surface_protocol,
            lifecycle,
            components,
        ).build(GroundedAnswerRunLocalBuild(
            self.model,
            claim,
            target,
            request,
            candidate,
            variant.option.structure_id,
            variant.option.pattern_id,
            GroundedAnswerParserProtocol(
                *_instruction_series(23, 5, branch),
                content.answer,
            ),
            _instruction_series(24, 1, branch)[0],
            _instruction_series(25, 1, branch)[0],
            _instruction_series(26, 1, branch)[0],
            (*_NAMESPACE, 27, 1),
        ))
        if self.stage4_owner is not None:
            registered_hypothesis = self.stage4_owner.ensure_template(
                installation.variant.template, compilation.value_protocol)
            if registered_hypothesis != connector_hypothesis:
                raise RuntimeError("typed stage4 Hypothesis 预计算漂移")
        execution = installation.executor.execute(request)
        dynamic_postchecker = installation.runtime.postchecker
        if not isinstance(dynamic_postchecker, GenerationPostcheckRuntime):
            raise RuntimeError("grounded installation 缺少 G-04 postchecker")
        self.postcheck_runtime.bind(execution, dynamic_postchecker)
        return execution


def _build_lifecycle_for_branch(ctx: TrainContext, branch):
    """建立当前图归属的 S-07 lifecycle；不复用旧临时 context。"""
    ontology = ctx.graph_ontology
    predicates = tuple(
        concept_identity(
            (*_NAMESPACE, 30, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 26)
    )
    refs = tuple(ontology.materialize(item) for item in predicates)
    graph = StructureOrderGraph(
        ontology, StructureOrderGraphPredicates(*refs[:19]))
    states = tuple(
        concept_identity(
            (*_NAMESPACE, 31, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 7)
    )
    for identity in states:
        ontology.materialize(identity)
    return StructureOrderLifecycleGraph(
        graph,
        StructureOrderLifecycleProtocol(
            *refs[19:], *states, (*_NAMESPACE, 32, 1)))


def _stage4_policy(branch) -> LanguageConnectorStage4Policy:
    """把同一 G-04 proposition signal 接到 H-00/H-04 feedback。"""
    postcheck = _generation_protocols(branch)[5]
    return LanguageConnectorStage4Policy(
        (LanguageConnectorSignalRoute(
            postcheck.proposition_dimension,
            postcheck.proposition_verifier,
            ((APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),),
            ((APPLICABILITY_APPLICABLE, VERDICT_REFUTE),),
        ),),
        _STAGE4_VERIFIER_SOURCE,
        (*_NAMESPACE, 60, 1),
        _STAGE4_ACTIVE_PURPOSE,
        _STAGE4_TRIAL_PURPOSE,
    )


def _candidate_projection_protocol() -> CandidateProjectionProtocol:
    """建立 typed connector 专用的候选 lifecycle 整数协议。"""
    identities = tuple(concept_identity((*_NAMESPACE, 70, index))
                       for index in range(1, 14))
    return CandidateProjectionProtocol(
        *identities,
        (*_NAMESPACE, 71, 1),
    )


class _TypedStage4Owner:
    """为动态 grounded templates 提供真实 H-00/H-04 Stage 4 owner。"""

    def __init__(self, ctx: TrainContext) -> None:
        if not isinstance(ctx, TrainContext):
            raise TypeError("typed stage4 ctx 类型错误")
        if ctx.training_candidate_history is None:
            raise RuntimeError("typed stage4 缺少 Core candidate history")
        self.ctx = ctx
        # H2/floor clones deliberately omit Core H-00 history.  They are
        # read-only generation probes, so candidate lifecycle is host-only.
        self.enabled = not ctx.evaluation_strictly_isolated
        self.templates: dict[tuple[int, ...], LanguageGenerationConnectorTemplate] = {}
        self._runtime: LanguageConnectorStage4Runtime | None = None
        self._candidates: LanguageConnectorCandidateRuntime | None = None
        self._policy: LanguageConnectorStage4Policy | None = None
        self._branch = None
        self._value_protocol = None

    def _ensure_owner(
            self,
            template: LanguageGenerationConnectorTemplate,
            value_protocol=None,
            ) -> tuple[LanguageConnectorCandidateRuntime, LanguageConnectorStage4Policy]:
        """按首次真实模板建立候选图，随后所有模板共享该 run owner。"""
        branch = template.language_branch
        if self._candidates is not None:
            if branch != self._branch:
                raise RuntimeError(
                    "typed stage4 当前 run 出现多个 language branch，"
                    "无法共享同一 S-07 candidate graph")
            return self._candidates, self._policy
        if value_protocol is None:
            raise RuntimeError("typed stage4 首次安装缺少 grounded value protocol")
        lifecycle = _build_lifecycle_for_branch(self.ctx, branch)
        order_graph = lifecycle.order_graph
        ontology = self.ctx.graph_ontology
        connector_predicates = tuple(
            ontology.materialize(concept_identity(
                (*_NAMESPACE, 72, index),
                owner=branch.owner,
                versions=branch.versions,
            ))
            for index in range(1, 22)
        )
        definition_graph = LanguageGenerationConnectorGraph(
            ontology,
            order_graph,
            LanguageConnectorGraphPredicates(*connector_predicates),
            # All grounded variants for a branch use the same value protocol.
            value_protocol,
        )
        projection_protocol = _candidate_projection_protocol()
        candidate_graph = CandidateProjectionGraph(ontology, projection_protocol)
        aggregate_source = SourceRef(
            21405, 60000, 0, branch.owner, branch.versions)
        aggregate_protocol = EvidenceCandidateProtocol(
            (*_NAMESPACE, 73, 1),
            (*_NAMESPACE, 73, 2),
            aggregate_source,
            document_scope(aggregate_source),
            2,
        )
        candidate_protocol = LanguageConnectorCandidateProtocol(
            concept_identity((*_NAMESPACE, 76, 1)),
            concept_identity((*_NAMESPACE, 76, 2)),
            concept_identity((*_NAMESPACE, 76, 3)),
            (*_NAMESPACE, 76, 4),
        )
        history_protocol = TrainingHypothesisHistoryProtocol(
            candidate_protocol.stable_key(),
            aggregate_protocol.hypothesis_kind_key,
            aggregate_protocol.aggregate_source,
            aggregate_protocol.aggregate_scope,
        )
        sink = TrainingHypothesisEventSink(
            self.ctx.training_candidate_history, history_protocol)
        learning = CandidateLearningRuntime(
            EvidenceCandidateEngine(
                aggregate_protocol,
                ledger=HypothesisLedger(sink),
            ),
            candidate_graph,
            IndependentObjectVerifier(IndependentVerifierProtocol(
                concept_identity((*_NAMESPACE, 75, 1)),
                (*_NAMESPACE, 75, 2),
                (*_NAMESPACE, 75, 3),
                (*_NAMESPACE, 75, 4),
                (*_NAMESPACE, 75, 5),
            )),
            CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
        )
        candidates = LanguageConnectorCandidateRuntime(
            definition_graph,
            learning,
            candidate_protocol,
            persistence_kind=CANDIDATE_PERSISTENCE_TRAINING,
        )
        if self.ctx.training_candidate_history.entries(history_protocol):
            candidates = candidates.restore_for_training_graphs(
                definition_graph,
                candidate_graph,
                self.ctx.training_candidate_history,
            )
        self._branch = branch
        self._value_protocol = value_protocol
        self._candidates = candidates
        self._policy = _stage4_policy(branch)
        return candidates, self._policy

    def ensure_template(
            self,
            template: LanguageGenerationConnectorTemplate,
            value_protocol=None,
            ) -> object:
        """登记一个真实 grounded template，并返回其 H-00 hypothesis。"""
        if not isinstance(template, LanguageGenerationConnectorTemplate):
            raise TypeError("typed stage4 template 类型错误")
        candidates, _policy = self._ensure_owner(template, value_protocol)
        key = template.connector.stable_key()
        existing = self.templates.get(key)
        if existing is not None and existing != template:
            raise RuntimeError("typed stage4 connector template 身份漂移")
        if existing is None:
            first, second = self._forming_sources(template)
            candidates.register(
                template,
                (first, second),
                scope=document_scope(first),
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
                timestamp_base=0,
            )
            self.templates[key] = template
        return candidates.learning.hypothesis_for_candidate(template.connector)

    @staticmethod
    def _forming_sources(template):
        digest = _FORMING_HASHER.h63(template.connector.stable_key())
        first = SourceRef(21405, 100000 + (digest % 1000000000), 0,
                          GLOBAL_OWNER_SCOPE, VersionBundle())
        second = SourceRef(21405, 100000 + ((digest + 1) % 1000000000),
                           0, GLOBAL_OWNER_SCOPE, VersionBundle())
        return first, second

    def hypothesis_for_template(self, template, value_protocol=None):
        """在 graph materialize 前返回该模板将登记的 exact Hypothesis。"""
        if not isinstance(template, LanguageGenerationConnectorTemplate):
            raise TypeError("typed stage4 template 类型错误")
        candidates, _policy = self._ensure_owner(template, value_protocol)
        definition = candidates.mapper.definition(
            template, self._forming_sources(template))
        return definition.hypothesis(candidates.learning.engine.protocol)

    def purpose_for_template(self, template, value_protocol=None):
        """返回当前候选状态对应的 active/trial surface purpose。"""
        hypothesis = self.hypothesis_for_template(template, value_protocol)
        if (template.connector.stable_key() in self.templates
                and self._candidates.learning.engine.active(hypothesis) is not None):
            return hypothesis, _STAGE4_ACTIVE_PURPOSE
        return hypothesis, _STAGE4_TRIAL_PURPOSE

    def apply(self, episodes: tuple[TypedLanguageEpisode, ...]) -> LanguageConnectorStage4Report:
        """把完整 typed episodes 交给真实 Stage4 runtime。"""
        if self._candidates is None or self._policy is None:
            raise RuntimeError("typed stage4 在无真实 generation template 时被调用")
        eligible = tuple(
            episode for episode in episodes
            if episode.generation_complete
            and episode.production.execution is not None
            and episode.production.execution.surface is not None
            and episode.production.execution.surface.preview.request.sentence_attributions
        )
        if not eligible:
            raise RuntimeError("typed stage4 当前批次没有完整 surface generation episode")
        if self._runtime is None:
            self._runtime = LanguageConnectorStage4Runtime(
                self._candidates, self._policy)
        return self._runtime.apply(eligible)

    def state_key(self) -> tuple:
        """返回动态模板、候选和反馈策略的可比较状态。"""
        return (
            tuple(sorted(self.templates)),
            () if self._candidates is None else self._candidates.state_key(),
            () if self._policy is None else self._policy.stable_key(),
        )


class TypedDialogueGenerationRuntimeFactory:
    """从公开 grounded-answer TRAIN 课程建立 typed production owner。"""

    def __init__(self, model: GroundedAnswerSurfaceModel, pack) -> None:
        if not isinstance(model, GroundedAnswerSurfaceModel):
            raise TypeError("typed generation model 类型错误")
        self.model = model
        self.pack = pack

    @classmethod
    def from_project_root(cls, project_root: str | Path):
        root = Path(project_root).resolve()
        path = root / "data" / "ph2" / "grounded_answer_train_v1.jsonl.sample"
        payload = path.read_bytes()
        bundle = compile_grounded_answer_training_records_from_payload(
            payload, source_relative_path="data/ph2/grounded_answer_train_v1.jsonl.sample")
        model, _report = learn_grounded_answer_surface_model(bundle)
        from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
            build_generation_candidate_pack,
        )
        pack = build_generation_candidate_pack(
            model, hashlib.sha256(payload).hexdigest())
        return cls(model, pack)

    def build(self, ctx: TrainContext) -> ProductionGenerationRuntime:
        return self._build_runtime(ctx, None)

    def _build_runtime(
            self,
            ctx: TrainContext,
            stage4_owner: _TypedStage4Owner | None,
            ) -> ProductionGenerationRuntime:
        mapper = _RequestMapper()
        postcheck_runtime = _DynamicPostcheckRuntime()
        executor = _DynamicExecutor(
            ctx, self.model, self.pack, mapper, postcheck_runtime,
            stage4_owner)
        return ProductionGenerationRuntime(
            mapper,
            executor,
            postcheck_mapper=_DynamicPostcheckMapper(),
            postchecker=postcheck_runtime,
        )

    def build_installation(self, ctx: TrainContext) -> ProductionGenerationInstallation:
        """安装 G-00..G-04 与真实 connector Stage4 candidate owner。"""
        stage4_owner = _TypedStage4Owner(ctx)
        runtime = self._build_runtime(ctx, stage4_owner)
        return ProductionGenerationInstallation(runtime, stage4_owner)

    def clone_for_evaluation(self) -> "TypedDialogueGenerationRuntimeFactory":
        """复制公开课程模型和 pack 配置，不共享本次 run 的 mapper/runtime。"""
        return TypedDialogueGenerationRuntimeFactory(self.model, self.pack)

    def state_key(self) -> tuple:
        return self.pack.sha256(), self.pack.training_artifact_sha256


__all__ = ["TypedDialogueGenerationRuntimeFactory"]
