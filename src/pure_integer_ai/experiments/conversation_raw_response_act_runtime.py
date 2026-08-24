"""DLG-RAW-05A：公开课程派生的 non-answer response-act 实际运行时。

本模块不定义第二套 RAW terminal record。它只在 V2 frame 的公开课程、无标签
planning projection 与实际 G-01 一致时返回完整 ``QuestionAnswerRun``；调用方统一
从该 run 组织 UTF-8 输出及 DLG-RAW result code。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
    GenerationPlanProtocol,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
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
    PublicFrame,
    PublicFrameResponseActRuntimeRecipe,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
    prepare_public_course,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ConversationRawLexicalIngressResult,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_runtime import (
    ProductionGenerationAliasRuntimeFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileTarget,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_parser import (
    GroundedResponseActParserProtocol,
    GroundedResponseActSourceVerifier,
    GroundedResponseActStructureVerifier,
    GroundedResponseActTaskVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_runtime_factory import (
    GroundedResponseActQuestionInput,
    GroundedResponseActRunLocalBuild,
    GroundedResponseActRunLocalComponents,
    GroundedResponseActRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    PublicResponseActPlanningBuild,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerProtocol,
    QuestionAnswerRun,
)
from pure_integer_ai.experiments.train_context import TrainContext, make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)


_NAMESPACE = (65001, 52)
_REPRESENTATION_FAMILY = (65001, 53, 1)


# object-model: exception; interop=DLG-RAW-05A
class ConversationRawResponseActRuntimeError(RuntimeError):
    """V2 public frame 不能形成完整、无提交的 response-act runtime。"""


def _instruction_series(
        group: int,
        count: int,
        *,
        branch: ObjectIdentity,
        ) -> tuple[ObjectIdentity, ...]:
    """在当前 language branch owner/version 下建立互异运行指令身份。"""
    if (not isinstance(branch, ObjectIdentity)
            or branch.object_kind != OBJECT_LANGUAGE_BRANCH):
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A planning 缺 language branch")
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
    """为一轮 fresh non-answer runtime 创建 G-01 至 G-04 协议对象。"""
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
    """在本次 TrainContext 的 ontology 建立仅本轮有效的 S-07 lifecycle。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("RAW-05A train context 类型错误")
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
    protocol = StructureOrderLifecycleProtocol(
        *predicate_refs[19:],
        *states_and_kinds,
        (*_NAMESPACE, 42, 1),
    )
    return StructureOrderLifecycleGraph(graph, protocol)


def _course_payload(
        recipe: PublicFrameResponseActRuntimeRecipe,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> tuple[bytes, str]:
    """按 V2 recipe logical key 从 closure 取课程，并在 cache 前验证 raw bytes。"""
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("RAW-05A source payload closure 类型错误")
    relative_path = recipe.course_relative_path
    if (not isinstance(relative_path, str) or not relative_path
            or "\\" in relative_path):
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A course logical key 非法")
    try:
        logical_key = relative_path.encode("ascii")
    except UnicodeEncodeError as error:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A course logical key 非 ASCII") from error
    parts = logical_key.split(b"/")
    if (len(parts) != 3 or tuple(parts[:2]) != (b"data", b"ph2")
            or any(part in (b"", b".", b"..") for part in parts)):
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A course logical key 越界")
    try:
        record = source_payload_closure.record_for(logical_key)
        payload = source_payload_closure.payload_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A course 不在 public source payload closure") from error
    digest = public_source_payload_sha256_v1(payload)
    if (payload != record.raw_payload
            or len(payload) != record.payload_length
            or digest != record.raw_sha256):
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A course closure payload record 漂移")
    if tuple(digest) != recipe.course_raw_sha256:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A course SHA-256 漂移")
    return payload, relative_path


def _materialize_planning(
        frame: PublicFrame,
        request: QuestionRequest,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicResponseActPlanningBuild:
    """通过 V2 catalog 的受审计 materializer 重投影无标签 planning。"""
    # catalog projection 是 V2 专属模块；延迟导入使缺失时仍明确 fail closed。
    try:
        from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
            materialize_public_response_act_planning_from_closure,
        )
    except ImportError as error:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A catalog materializer 尚不可用") from error
    try:
        build = materialize_public_response_act_planning_from_closure(
            frame,
            request,
            source_payload_closure=source_payload_closure,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A 无法从当前公开课程物化 planning") from error
    if not isinstance(build, PublicResponseActPlanningBuild):
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A catalog materializer 返回类型错误")
    recipe = frame.recipe
    if not isinstance(recipe, PublicFrameResponseActRuntimeRecipe):
        raise ConversationRawResponseActRuntimeError("RAW-05A frame recipe 类型错误")
    if build.planning_input.canonical_record() != recipe.planning_input_record:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A public planning record 漂移")
    if build.planning.goal.target_branch != request.target_branch:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A planning/request branch 漂移")
    return build


def _response_act_for_selection(
        selector: AnswerContentSelector,
        planning: GenerationPlanningRequest,
        content: AnswerContentProtocol,
        ) -> tuple[str, AnswerContentSelection]:
    """只由同次实际 G-01 selection 映射允许的 non-answer response act。"""
    selection = selector.select(planning)
    mapping = (
        (content.unknown, "UNKNOWN"),
        (content.clarify, "CLARIFY"),
        (content.conflict, "CONFLICT"),
    )
    for stance, response_act in mapping:
        if selection.stance == stance:
            return response_act, selection
    raise ConversationRawResponseActRuntimeError(
        "RAW-05A G-01 selection 不属于 non-answer family")


def _surface_attribution(
        planning: GenerationPlanningRequest,
        *,
        theory: ObjectIdentity,
        branch: ObjectIdentity,
        ) -> GenerationSurfaceAttribution:
    """为零 claim non-answer surface 建立 aggregate-goal Core Use attribution。

    这里故意不从 ``planning.candidates`` 选择任一候选：UNKNOWN 没有候选，
    CLARIFY/CONFLICT 又必须保留全部候选。归属仅绑定同次 goal 的 source/scope，
    不把任一命题包装为输出事实。
    """
    if not isinstance(planning, GenerationPlanningRequest):
        raise TypeError("RAW-05A surface attribution planning 类型错误")
    goal = planning.goal
    candidate_key = goal.proposition.stable_key()
    hypothesis = HypothesisKey(
        (*_NAMESPACE, 50, 1),
        candidate_key,
        integer_tuple_fingerprint(
            candidate_key,
            domain="dlg.raw.response.act.surface.hypothesis.v1",
        ),
        goal.scope,
        goal.source,
    )
    return GenerationSurfaceAttribution(
        theory,
        hypothesis,
        _instruction_series(16, 1, branch=branch)[0],
    )


def _require_complete_run(
        run: QuestionAnswerRun,
        *,
        preselection: AnswerContentSelection,
        expected_stance: ObjectIdentity,
        ) -> None:
    """确认 factory 的重复 G-01、G-03、G-04 与零提交严格同次。"""
    if (not isinstance(run, QuestionAnswerRun)
            or not run.complete
            or run.selection is None
            or run.selection != preselection
            or run.status != expected_stance
            or run.selection.stance != expected_stance
            or run.generation is None
            or not run.generation.complete
            or run.generation.rendered is None
            or run.postcheck is None
            or not run.postcheck.complete
            or run.selection_commit is not None
            or run.outcome_commit is not None):
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A actual runtime 未完成或 G-01/G-04 不一致")
    scalars = tuple(run.generation.rendered.units)
    output = encode_utf8_v1(scalars)
    readback = intake_raw_conversation_vector(output)
    if not readback.accepted or readback.unicode_scalars != scalars:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A G-03 UTF-8 readback 漂移")


def _run_public_response_act_frame(
        ingress: ConversationRawLexicalIngressResult,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        preparation_cache: PublicCoursePreparationCache | None = None,
        ) -> QuestionAnswerRun:
    """执行已完成参数门的 V2 RAW-05A runtime，不组织 terminal 输出。"""
    if not isinstance(ingress, ConversationRawLexicalIngressResult):
        raise TypeError("RAW-05A ingress 类型错误")
    if not ingress.accepted or ingress.frame is None or ingress.request is None:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A 需要 accepted ingress/frame/request")
    frame = ingress.frame
    request = ingress.request
    if not isinstance(frame.recipe, PublicFrameResponseActRuntimeRecipe):
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A 只接受 V2 response-act frame")
    if request.target_branch is None:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A request 缺 target branch")
    planning_build = _materialize_planning(
        frame,
        request,
        source_payload_closure=source_payload_closure,
    )
    planning = planning_build.planning
    branch = planning.goal.target_branch
    if branch is None:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A planning 缺 target branch")
    content, selector, plan_protocol, structure_protocol, surface_protocol, postcheck_protocol, question_protocol = (
        _generation_protocols(branch))
    actual_act, preselection = _response_act_for_selection(
        selector,
        planning,
        content,
    )
    target = GroundedResponseActCompileTarget(
        actual_act,
        preselection.stance,
        branch,
        _REPRESENTATION_FAMILY,
    )
    payload, relative_path = _course_payload(
        frame.recipe,
        source_payload_closure,
    )
    prepared = prepare_public_course(
        payload,
        course_relative_path=relative_path,
        course_raw_sha256=frame.recipe.course_raw_sha256,
        cache=preparation_cache,
    )
    variants = compile_grounded_response_act_patterns(
        prepared.model,
        target,
    ).variants
    if not variants:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A G-02 response-act 没有 learned variant")
    selected_variant = variants[0]
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        alias_factory = ProductionGenerationAliasRuntimeFactory(prepared.pack, ctx)
        renderer_identity = _instruction_series(9, 1, branch=branch)[0]
        components = GroundedResponseActRunLocalComponents(
            selector,
            plan_protocol,
            structure_protocol,
            surface_protocol,
            alias_factory,
            UnicodeRepresentationRenderer(
                _REPRESENTATION_FAMILY,
                renderer_identity,
            ),
            renderer_identity,
            postcheck_protocol,
            GroundedResponseActStructureVerifier(*_instruction_series(
                10, 2, branch=branch)),
            GroundedResponseActSourceVerifier(*_instruction_series(
                11, 2, branch=branch)),
            GroundedResponseActTaskVerifier(*_instruction_series(
                12, 2, branch=branch)),
            question_protocol,
            _surface_attribution(
                planning,
                theory=selected_variant.template.sentence,
                branch=branch,
            ),
        )
        installation = GroundedResponseActRunLocalFactory(
            _build_lifecycle(ctx, branch),
            components,
        ).build(GroundedResponseActRunLocalBuild(
            prepared.model,
            GroundedResponseActQuestionInput(actual_act),
            target,
            planning,
            selected_variant.pattern_id,
            GroundedResponseActParserProtocol(*_instruction_series(
                13, 3, branch=branch)),
            request.query_kind,
            _instruction_series(14, 1, branch=branch)[0],
            _instruction_series(15, 1, branch=branch)[0],
            (*_NAMESPACE, 60, 1),
        ))
        if (installation.variant != selected_variant
                or installation.variant.template.stance != preselection.stance):
            raise ConversationRawResponseActRuntimeError(
                "RAW-05A lowest pattern 或 factory stance 漂移")
        run = installation.runtime.run(request)
        _require_complete_run(
            run,
            preselection=preselection,
            expected_stance=preselection.stance,
        )
        return run
    finally:
        backend.close()


def run_public_response_act_frame(
        ingress: ConversationRawLexicalIngressResult,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        preparation_cache: PublicCoursePreparationCache | None = None,
        ) -> QuestionAnswerRun:
    """执行一轮 V2 RAW-05A，并将所有运行期失败收敛为同一公开异常。"""
    if not isinstance(ingress, ConversationRawLexicalIngressResult):
        raise TypeError("RAW-05A ingress 类型错误")
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("RAW-05A source payload closure 类型错误")
    if (preparation_cache is not None
            and not isinstance(preparation_cache, PublicCoursePreparationCache)):
        raise TypeError("RAW-05A preparation cache 类型错误")
    try:
        return _run_public_response_act_frame(
            ingress,
            source_payload_closure=source_payload_closure,
            preparation_cache=preparation_cache,
        )
    except ConversationRawResponseActRuntimeError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConversationRawResponseActRuntimeError(
            "RAW-05A runtime 执行失败") from error


__all__ = [
    "ConversationRawResponseActRuntimeError",
    "run_public_response_act_frame",
]
