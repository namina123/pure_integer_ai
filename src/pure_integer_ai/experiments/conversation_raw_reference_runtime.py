"""DLG-RAW-05B：公开、无标签双 claim/reference 的实际问答运行时。

本模块只消费 V3 catalog 每轮 source-locked 的 Evidence planning、显式命题顺序与
双份词汇来源。它不会读取完整课程 parser、answer plan、surface label、训练 pack、
教师或外部模型；任一运行异常由 RAW-02 统一收敛为零输出拒绝。
"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
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
    OBJECT_LANGUAGE_BRANCH,
    ObjectIdentity,
    concept_identity,
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
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1,
    PublicFrame,
    PublicFrameReferenceRuntimeRecipe,
)
from pure_integer_ai.experiments.conversation_public_reference_catalog import (
    PublicReferenceFramePlanningBuild,
    materialize_public_reference_planning_from_closure,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    decode_utf8_v1,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ConversationRawLexicalIngressResult,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageGenerationConnector,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    REFERENCE_STRATEGIES,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice import (
    GroundedAnswerReferenceSelection,
    build_grounded_answer_reference_selection,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
    PublicGroundedAnswerReferenceClaimSurface,
    PublicGroundedAnswerReferenceCompileRequest,
    compile_public_grounded_answer_reference_connector,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_postcheck import (
    GroundedAnswerReferenceEvidenceSourceVerifier,
    GroundedAnswerReferenceStructureVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_runtime_factory import (
    GroundedAnswerReferenceRunLocalBuild,
    GroundedAnswerReferenceRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalComponents,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningEvidence,
)
from pure_integer_ai.experiments.public_grounded_answer_reference_alias_runtime import (
    PublicGroundedAnswerReferenceAliasRuntimeFactory,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
    QuestionAnswerRun,
)
from pure_integer_ai.experiments.train_context import TrainContext, make_train_context
from pure_integer_ai.storage.backend import DictBackend


_NAMESPACE = (65001, 54)
_REPRESENTATION_FAMILY = (65001, 53, 3)
_STRATEGY_COSTS = (
    ("ANTECEDENT_REFERENCE", 0),
    ("EXPLICIT_REPETITION", 1),
)


# object-model: exception; interop=DLG-RAW-05B
class ConversationRawReferenceRuntimeError(RuntimeError):
    """V3 public frame 不能完成 source-grounded 双句 reference 问答。"""


def _instruction_series(
        group: int,
        count: int,
        *,
        branch: ObjectIdentity,
        ) -> tuple[ObjectIdentity, ...]:
    """在当前 language branch owner/version 下建立互异运行指令身份。"""
    if (not isinstance(branch, ObjectIdentity)
            or branch.object_kind != OBJECT_LANGUAGE_BRANCH):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B planning 缺 language branch")
    if type(group) is not int or type(count) is not int or count <= 0:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B instruction series 参数非法")
    return tuple(
        minimal_instruction_identity(
            (*_NAMESPACE, group, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, count + 1)
    )


def _generation_protocols(
        branch: ObjectIdentity,
        ) -> tuple[
            AnswerContentProtocol,
            AnswerContentSelector,
            GenerationPlanProtocol,
            GenerationStructureLayerProtocol,
            GenerationSurfaceProtocol,
            GenerationPostcheckProtocol,
            QuestionAnswerProtocol,
        ]:
    """为一轮 V3 reference run 创建 G-01 至 G-04 的独占协议。"""
    content = AnswerContentProtocol(*_instruction_series(
        1, 5, branch=branch))
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(
            content,
            EvidenceAnswerPolicyProtocol(*_instruction_series(
                2, 4, branch=branch)),
        ),
    )
    plan = GenerationPlanProtocol(*_instruction_series(3, 10, branch=branch))
    structure = GenerationStructureLayerProtocol(*_instruction_series(
        4, 3, branch=branch))
    surface = GenerationSurfaceProtocol(*_instruction_series(
        5, 9, branch=branch))
    postcheck = GenerationPostcheckProtocol(
        *tuple(ProtocolKey((*_NAMESPACE, 6, index))
               for index in range(1, 13)),
        *_instruction_series(7, 15, branch=branch),
    )
    question = QuestionAnswerProtocol(*_instruction_series(8, 3, branch=branch))
    return content, selector, plan, structure, surface, postcheck, question


def _build_lifecycle(
        ctx: TrainContext,
        branch: ObjectIdentity,
        ) -> StructureOrderLifecycleGraph:
    """在本轮 fresh context 内建立 V3 仅运行期有效的 S-07 lifecycle。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("RAW-05B train context 类型错误")
    ontology = ctx.graph_ontology
    predicates = tuple(
        concept_identity(
            (*_NAMESPACE, 40, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 26)
    )
    predicate_refs = tuple(ontology.materialize(item) for item in predicates)
    graph = StructureOrderGraph(
        ontology,
        StructureOrderGraphPredicates(*predicate_refs[:19]),
    )
    states_and_kinds = tuple(
        concept_identity(
            (*_NAMESPACE, 41, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 7)
    )
    for identity in states_and_kinds:
        ontology.materialize(identity)
    return StructureOrderLifecycleGraph(
        graph,
        StructureOrderLifecycleProtocol(
            *predicate_refs[19:],
            *states_and_kinds,
            (*_NAMESPACE, 42, 1),
        ),
    )


def _claim_scalars(
        evidence: GroundedResponseActPlanningEvidence,
        ) -> tuple[int, ...]:
    """将已 source-locked Evidence claim 经 UTF-8 v1 显式往返为 scalar。"""
    if not isinstance(evidence, GroundedResponseActPlanningEvidence):
        raise TypeError("RAW-05B Evidence 类型错误")
    scalars = tuple(ord(character) for character in evidence.claim_text)
    try:
        encoded = encode_utf8_v1(scalars)
    except (TypeError, ValueError) as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B Evidence claim 不能编码 UTF-8 v1") from error
    restored = decode_utf8_v1(encoded)
    if restored is None or restored != scalars:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B Evidence claim UTF-8 v1 readback 漂移")
    return restored


def _course_payload(
        recipe: PublicFrameReferenceRuntimeRecipe,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> tuple[bytes, str]:
    """按 V3 recipe logical key 从 closure 取课程，并显式复核 SHA-256。"""
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("RAW-05B source payload closure 类型错误")
    relative_path = recipe.course_relative_path
    if (not isinstance(relative_path, str) or not relative_path
            or "\\" in relative_path):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B course logical key 非法")
    try:
        logical_key = relative_path.encode("ascii")
    except UnicodeEncodeError as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B course logical key 非 ASCII") from error
    parts = logical_key.split(b"/")
    if (len(parts) != 3 or tuple(parts[:2]) != (b"data", b"ph2")
            or any(part in (b"", b".", b"..") for part in parts)):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B course logical key 越界")
    try:
        record = source_payload_closure.record_for(logical_key)
        payload = source_payload_closure.payload_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B course 不在 public source payload closure") from error
    digest = public_source_payload_sha256_v1(payload)
    if (payload != record.raw_payload
            or len(payload) != record.payload_length
            or digest != record.raw_sha256):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B course closure payload record 漂移")
    if tuple(digest) != recipe.course_raw_sha256:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B course SHA-256 漂移")
    return payload, relative_path


def _ordered_claims_and_forming(
        build: PublicReferenceFramePlanningBuild,
        recipe: PublicFrameReferenceRuntimeRecipe,
        ) -> tuple[
            tuple[PublicGroundedAnswerReferenceClaimSurface, ...],
            tuple[tuple[int, ...], ...],
        ]:
    """按 recipe 命题序投影唯一 claim，并精确归集两个 candidate Evidence。"""
    if not isinstance(build, PublicReferenceFramePlanningBuild):
        raise TypeError("RAW-05B planning build 类型错误")
    if not isinstance(recipe, PublicFrameReferenceRuntimeRecipe):
        raise TypeError("RAW-05B reference recipe 类型错误")
    by_proposition: dict[str, tuple[int, ...]] = {}
    for evidence in build.planning_build.planning_input.evidence:
        scalars = _claim_scalars(evidence)
        prior = by_proposition.get(evidence.proposition_id)
        if prior is not None and prior != scalars:
            raise ConversationRawReferenceRuntimeError(
                "RAW-05B 同一 Proposition Evidence claim 不一致")
        by_proposition[evidence.proposition_id] = scalars
    ordered_ids = recipe.ordered_proposition_ids
    if (len(by_proposition) != 2
            or set(by_proposition) != set(ordered_ids)):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B Evidence 未精确覆盖 recipe 双命题")
    claims = tuple(PublicGroundedAnswerReferenceClaimSurface(
        proposition_id,
        by_proposition[proposition_id],
    ) for proposition_id in ordered_ids)
    candidates = tuple(
        build.planning_build.candidate_for(proposition_id)
        for proposition_id in ordered_ids)
    if (len(set(candidates)) != 2
            or set(candidates) != set(build.planning_build.planning.candidates)):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B recipe 命题顺序未精确覆盖 planning candidates")
    forming = tuple(sorted({
        evidence.stable_key()
        for candidate in candidates
        for evidence in candidate.evidence
    }))
    if not forming:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B candidates 缺 forming Evidence")
    return claims, forming


def _select_lowest_cost_strategy(
        compilations: tuple[GroundedAnswerReferenceCompilation, ...],
        recipe: PublicFrameReferenceRuntimeRecipe,
        ) -> GroundedAnswerReferenceSelection:
    """在两份成功编译之间只按冻结纯整数成本选择唯一最低策略。"""
    if (recipe.relation_kind_code != 1
            or recipe.g04_required != 1
            or recipe.strategy_selection_policy
            != PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B reference recipe 选择策略未注册")
    if tuple(strategy for strategy, _cost in _STRATEGY_COSTS) != REFERENCE_STRATEGIES:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B strategy enum/cost order 漂移")
    by_strategy = {item.strategy: item for item in compilations}
    if (len(compilations) != len(_STRATEGY_COSTS)
            or len(by_strategy) != len(_STRATEGY_COSTS)
            or set(by_strategy) != {item[0] for item in _STRATEGY_COSTS}):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B 未形成完整双策略竞争集")
    selected_strategy = ""
    selected_cost: int | None = None
    for strategy, cost in _STRATEGY_COSTS:
        if type(cost) is not int or cost < 0:
            raise ConversationRawReferenceRuntimeError(
                "RAW-05B strategy cost 非法")
        if selected_cost is None or cost < selected_cost:
            selected_strategy = strategy
            selected_cost = cost
        elif cost == selected_cost:
            raise ConversationRawReferenceRuntimeError(
                "RAW-05B strategy 最低成本不唯一")
    if not selected_strategy or selected_cost is None:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B strategy 无可选最低成本")
    try:
        return build_grounded_answer_reference_selection(
            compilations,
            selected_strategy,
            (*_NAMESPACE, 50, selected_cost + 1),
        )
    except (TypeError, ValueError) as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B strategy selection 无法闭合") from error


def _with_source_grounded_attributions(
        compilation: GroundedAnswerReferenceCompilation,
        purpose: ObjectIdentity,
        ) -> GroundedAnswerReferenceCompilation:
    """为每个已编译句绑定其同轮 candidate 的完整 Evidence Hypothesis。

    Connector 编译只形成不可变理论和 slot；Core Use 的 owner/context 则必须在
    本轮实际 run 前由句级 candidate 明确给出。这里不选择、合成或保存 Evidence，
    只把 ``GenerationCandidate.hypotheses`` 的规范首项接到对应 connector 理论。
    """
    if not isinstance(compilation, GroundedAnswerReferenceCompilation):
        raise TypeError("RAW-05B attribution compilation 类型错误")
    if not isinstance(purpose, ObjectIdentity):
        raise TypeError("RAW-05B attribution purpose 类型错误")
    attributions = []
    for sentence in compilation.sentences:
        hypotheses = sentence.candidate.hypotheses
        if not hypotheses:
            raise ConversationRawReferenceRuntimeError(
                "RAW-05B reference candidate 缺 source Hypothesis")
        attributions.append(GenerationSurfaceAttribution(
            sentence.template.connector,
            hypotheses[0],
            purpose,
        ))
    try:
        connector = LanguageGenerationConnector(
            compilation.connector.registry,
            compilation.connector.runtime_policy,
            compilation.connector.surface_protocol,
            tuple(attributions),
            compilation.connector.discourse_declarations,
            compilation.connector.anaphora_declarations,
        )
        return replace(compilation, connector=connector)
    except (TypeError, ValueError) as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B connector source attribution 无法闭合") from error


def _compile_selection(
        build: PublicReferenceFramePlanningBuild,
        recipe: PublicFrameReferenceRuntimeRecipe,
        surface_protocol: GenerationSurfaceProtocol,
        source_attribution_purpose: ObjectIdentity,
        ) -> GroundedAnswerReferenceSelection:
    """从同一无标签 planning 编译两个 strategy，再执行冻结整数成本选择。"""
    claims, forming = _ordered_claims_and_forming(build, recipe)
    compilations = []
    for strategy, _cost in _STRATEGY_COSTS:
        try:
            compilations.append(compile_public_grounded_answer_reference_connector(
                PublicGroundedAnswerReferenceCompileRequest(
                    build.planning_build,
                    claims,
                    _REPRESENTATION_FAMILY,
                    strategy,
                    build.antecedent_reference_scalars,
                    build.explicit_repetition_scalars,
                    forming,
                ),
                surface_protocol,
            ))
        except (TypeError, ValueError) as error:
            raise ConversationRawReferenceRuntimeError(
                "RAW-05B 无法编译完整 reference strategy 竞争集") from error
    attributed = tuple(_with_source_grounded_attributions(
        compilation,
        source_attribution_purpose,
    ) for compilation in compilations)
    return _select_lowest_cost_strategy(attributed, recipe)


def _require_complete_run(
        run: QuestionAnswerRun,
        *,
        selection: GroundedAnswerReferenceSelection,
        installation,
        answer_stance: ObjectIdentity,
        ) -> None:
    """验证 G-01/G-03/G-04、双句顺序、实际照应恢复与零长期提交。"""
    if (not isinstance(run, QuestionAnswerRun)
            or not run.complete
            or run.selection is None
            or run.status != answer_stance
            or run.selection.stance != answer_stance
            or run.planning_request != selection.compilation.planning
            or run.generation is None
            or not run.generation.complete
            or run.generation.rendered is None
            or run.generation.surface is None
            or run.postcheck is None
            or not run.postcheck.complete
            or run.selection_commit is not None
            or run.outcome_commit is not None):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B actual runtime 未完成 G-01/G-03/G-04 或发生提交")
    planned = selection.compilation.ordered_candidates
    if (set(run.selection.selected_candidate_keys)
            != {item.stable_key() for item in planned}):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B G-01 未精确选择两个 reference candidates")
    generation = run.generation
    preview = generation.surface.preview
    parse_request = GenerationSurfaceParseRequest.from_execution(generation)
    try:
        recovery = installation.parser.recover_reference(parse_request, preview)
    except (TypeError, ValueError) as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B actual reference 不能由 G-03/G-02 回读") from error
    if (recovery.strategy_object != selection.selected.declarative_object
            or recovery.source != selection.source
            or recovery.scope != selection.scope):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B recovered reference 与先行 selection 漂移")
    if selection.selected.strategy == "ANTECEDENT_REFERENCE":
        antecedent = selection.selected_antecedent
        if (recovery.antecedent_candidate_key != antecedent.stable_key()
                or recovery.antecedent
                != antecedent.proposition.template
                or not recovery.reference_proposal_key):
            raise ConversationRawReferenceRuntimeError(
                "RAW-05B antecedent strategy 未恢复唯一前序命题")
    elif (selection.selected.strategy != "EXPLICIT_REPETITION"
          or recovery.antecedent_candidate_key
          or recovery.antecedent is not None
          or recovery.reference_proposal_key):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B explicit strategy reference recovery 漂移")
    scalars = tuple(generation.rendered.units)
    output = encode_utf8_v1(scalars)
    readback = intake_raw_conversation_vector(output)
    if not readback.accepted or readback.unicode_scalars != scalars:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B G-03 UTF-8 output readback 漂移")


def _run_public_reference_frame(
        ingress: ConversationRawLexicalIngressResult,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> QuestionAnswerRun:
    """执行一轮 V3 source-grounded reference runtime，不组织 terminal 输出。"""
    if not isinstance(ingress, ConversationRawLexicalIngressResult):
        raise TypeError("RAW-05B ingress 类型错误")
    if not ingress.accepted or ingress.frame is None or ingress.request is None:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B 需要 accepted ingress/frame/request")
    frame = ingress.frame
    request = ingress.request
    if not isinstance(frame, PublicFrame):
        raise ConversationRawReferenceRuntimeError("RAW-05B frame 类型错误")
    if not isinstance(frame.recipe, PublicFrameReferenceRuntimeRecipe):
        raise ConversationRawReferenceRuntimeError("RAW-05B 只接受 V3 reference frame")
    if request.target_branch is None:
        raise ConversationRawReferenceRuntimeError("RAW-05B request 缺 target branch")
    _course_payload(frame.recipe, source_payload_closure)
    try:
        planning_build = materialize_public_reference_planning_from_closure(
            frame,
            request,
            source_payload_closure=source_payload_closure,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B 无法从当前公开 source 物化 planning") from error
    if not isinstance(planning_build, PublicReferenceFramePlanningBuild):
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B catalog materializer 返回类型错误")
    planning = planning_build.planning_build.planning
    branch = planning.goal.target_branch
    if branch is None or branch != request.target_branch:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B planning/request language branch 漂移")
    content, selector, plan_protocol, structure_protocol, surface_protocol, postcheck_protocol, question_protocol = (
        _generation_protocols(branch))
    selection = _compile_selection(
        planning_build,
        frame.recipe,
        surface_protocol,
        _instruction_series(16, 1, branch=branch)[0],
    )
    if selection.compilation.planning != planning:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B selected compilation 替换 public planning")
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        renderer_identity = _instruction_series(9, 1, branch=branch)[0]
        components = GroundedAnswerRunLocalComponents(
            selector=selector,
            plan_protocol=plan_protocol,
            structure_protocol=structure_protocol,
            alias_factory=PublicGroundedAnswerReferenceAliasRuntimeFactory(
                ctx,
                preflight_cache=preflight_cache,
            ),
            renderer=UnicodeRepresentationRenderer(
                _REPRESENTATION_FAMILY,
                renderer_identity,
            ),
            renderer_identity=renderer_identity,
            postcheck_protocol=postcheck_protocol,
            structure_verifier=GroundedAnswerReferenceStructureVerifier(
                *_instruction_series(10, 2, branch=branch),
                tuple(item.stable_key()
                      for item in selection.compilation.ordered_candidates),
            ),
            source_verifier=GroundedAnswerReferenceEvidenceSourceVerifier(
                *_instruction_series(11, 2, branch=branch),
            ),
            question_protocol=question_protocol,
            postcheck_mapper=EvidenceQuestionPostcheckMapper(
                (*_NAMESPACE, 60, 1),
                citation_required=True,
                trust_required=True,
            ),
        )
        installation = GroundedAnswerReferenceRunLocalFactory(
            _build_lifecycle(ctx, branch),
            components,
        ).build(GroundedAnswerReferenceRunLocalBuild(
            selection.compilation,
            selection,
            GroundedAnswerParserProtocol(
                *_instruction_series(12, 5, branch=branch),
                content.answer,
            ),
            request.query_kind,
            _instruction_series(13, 1, branch=branch)[0],
            _instruction_series(14, 1, branch=branch)[0],
            (*_NAMESPACE, 60, 2),
        ))
        run = installation.runtime.run(request)
        _require_complete_run(
            run,
            selection=selection,
            installation=installation,
            answer_stance=content.answer,
        )
        return run
    finally:
        backend.close()


def run_public_reference_frame(
        ingress: ConversationRawLexicalIngressResult,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> QuestionAnswerRun:
    """执行一轮 V3 运行时，并把所有内部失败映为统一公开异常。"""
    if not isinstance(ingress, ConversationRawLexicalIngressResult):
        raise TypeError("RAW-05B ingress 类型错误")
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("RAW-05B source payload closure 类型错误")
    try:
        return _run_public_reference_frame(
            ingress,
            source_payload_closure=source_payload_closure,
            preflight_cache=preflight_cache,
        )
    except ConversationRawReferenceRuntimeError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConversationRawReferenceRuntimeError(
            "RAW-05B runtime 执行失败") from error


__all__ = [
    "ConversationRawReferenceRuntimeError",
    "run_public_reference_frame",
]
