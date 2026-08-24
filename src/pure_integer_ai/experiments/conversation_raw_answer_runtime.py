"""DLG-RAW-02：公开 Frame 到真实 F-00/G-03/G-04 的一次性运行桥。

该模块只接受 RAW-01 已经形成的完整 ``QuestionRequest``。它从公开、内容锁
定的训练课程重新学习表层模型，以实际 ``QuestionAnswerRuntime`` 生成输出。任何
运行期写入只存在于本次 ``DictBackend``，函数返回前关闭；不会写会话 context、
SQLite、长期 Memory、selection commit 或 outcome commit。
"""
from __future__ import annotations

from dataclasses import dataclass

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
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
)
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
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrame,
    PublicFrameReferenceRuntimeRecipe,
    PublicFrameResponseActRuntimeRecipe,
    PublicFrameRuntimeRecipe,
    materialize_public_frame_candidate,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
    prepare_public_course,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    DLG_RAW_REJECT_RUNTIME,
    ConversationRawIntake,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ConversationRawLexicalIngressResult,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_generation_candidate_alias_runtime import (
    ProductionGenerationAliasRuntimeFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerClaimInput,
    GroundedAnswerConnectorTarget,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import surface_pattern_structure_id
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
    QuestionAnswerRun,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
)
from pure_integer_ai.experiments.train_context import TrainContext, make_train_context
from pure_integer_ai.storage.backend import DictBackend


RAW_ANSWER_RUNTIME_RECORD_V1 = 1

_NAMESPACE = (65001, 50)
_REPRESENTATION_FAMILY = (65001, 51, 1)


# object-model: exception; interop=DLG-RAW-02
class ConversationRawAnswerRuntimeError(RuntimeError):
    """公开课程、run-local runtime 或 byte readback 不能形成真实回答。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """将可变长度严格整数段写入规范 result record。"""
    result.extend((len(value), *value))


def _instruction_series(
        group: int,
        count: int,
        *,
        frame: PublicFrame,
        ) -> tuple:
    """在 target branch 的 owner/version 内建立互异的最小执行身份。"""
    branch = frame.question.target_branch
    if branch is None:
        raise ConversationRawAnswerRuntimeError("RAW-02 frame 缺 target branch")
    return tuple(
        minimal_instruction_identity(
            (*_NAMESPACE, group, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, count + 1)
    )


def _generation_protocols(frame: PublicFrame):
    """建立本 run 的纯整数 G-00/G-01/G-02/G-03/G-04 协议对象。"""
    content = AnswerContentProtocol(*_instruction_series(1, 5, frame=frame))
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(
            content,
            EvidenceAnswerPolicyProtocol(
                *_instruction_series(2, 4, frame=frame),
            ),
        ),
    )
    plan = GenerationPlanProtocol(*_instruction_series(3, 10, frame=frame))
    structure = GenerationStructureLayerProtocol(
        *_instruction_series(4, 3, frame=frame),
    )
    surface = GenerationSurfaceProtocol(
        *_instruction_series(5, 9, frame=frame),
    )
    keys = tuple(
        ProtocolKey((*_NAMESPACE, 6, index))
        for index in range(1, 13)
    )
    postcheck = GenerationPostcheckProtocol(
        *keys,
        *_instruction_series(7, 15, frame=frame),
    )
    question = QuestionAnswerProtocol(*_instruction_series(8, 3, frame=frame))
    return content, selector, plan, structure, surface, postcheck, question


def _build_lifecycle(
        ctx: TrainContext,
        frame: PublicFrame,
        ) -> StructureOrderLifecycleGraph:
    """在同一个临时 GraphOntology 建立 G-03 所需的 S-07 生命周期 owner。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("RAW-02 train context 类型错误")
    branch = frame.question.target_branch
    if branch is None:
        raise ConversationRawAnswerRuntimeError("RAW-02 frame 缺 target branch")
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


def _logical_payload_key(
        relative_path: str,
        *,
        label: str,
        ) -> tuple[str, bytes]:
    """把 catalog transport 名称降为冻结 ASCII logical key，不解释物理路径。"""
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ConversationRawAnswerRuntimeError(f"{label} 不是规范 logical key")
    try:
        logical_key = relative_path.encode("ascii")
    except UnicodeEncodeError as error:
        raise ConversationRawAnswerRuntimeError(
            f"{label} 不是 ASCII logical key") from error
    parts = logical_key.split(b"/")
    if (len(parts) != 3 or tuple(parts[:2]) != (b"data", b"ph2")
            or any(part in (b"", b".", b"..") for part in parts)):
        raise ConversationRawAnswerRuntimeError(
            f"{label} 不在冻结 data/ph2 logical namespace")
    return relative_path, logical_key


def _payload_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        relative_path: str,
        *,
        label: str,
        ) -> bytes:
    """按逻辑 key 取得 closure 内 raw bytes，并重验 record 自洽性。"""
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("RAW-02 source payload closure 类型错误")
    _relative_path, logical_key = _logical_payload_key(relative_path, label=label)
    try:
        record = source_payload_closure.record_for(logical_key)
        payload = source_payload_closure.payload_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise ConversationRawAnswerRuntimeError(
            f"{label} 未绑定到 public source payload closure") from error
    digest = public_source_payload_sha256_v1(payload)
    if (payload != record.raw_payload
            or len(payload) != record.payload_length
            or digest != record.raw_sha256):
        raise ConversationRawAnswerRuntimeError(
            f"{label} closure payload record 漂移")
    return payload


def _verify_frame_source_payloads(
        frame: PublicFrame,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """逐轮重验 frame 的全部 source SHA、byte span 与 UTF-8 scalar span。"""
    if not isinstance(frame, PublicFrame):
        raise TypeError("RAW-02 frame 类型错误")
    source_records = frame.source_records
    if (not isinstance(source_records, tuple) or not source_records
            or len({item.record_id for item in source_records})
            != len(source_records)):
        raise ConversationRawAnswerRuntimeError("RAW-02 frame source records 非法")
    for source_record in source_records:
        payload = _payload_from_closure(
            source_payload_closure,
            source_record.relative_path,
            label=f"RAW-02 source {source_record.record_id}",
        )
        if tuple(public_source_payload_sha256_v1(payload)) != source_record.raw_sha256:
            raise ConversationRawAnswerRuntimeError(
                "RAW-02 public source SHA-256 漂移")
        span = source_record.span
        if (not isinstance(span, tuple) or len(span) != 2
                or any(type(item) is not int for item in span)):
            raise ConversationRawAnswerRuntimeError("RAW-02 public source span 非法")
        start, end = span
        if start < 0 or end <= start or end > len(payload):
            raise ConversationRawAnswerRuntimeError("RAW-02 public source span 越界")
        if tuple(payload[start:end]) != source_record.span_bytes:
            raise ConversationRawAnswerRuntimeError(
                "RAW-02 public source byte span 漂移")
        readback = intake_raw_conversation_vector(source_record.span_bytes)
        if (not readback.accepted
                or readback.unicode_scalars != source_record.span_scalars):
            raise ConversationRawAnswerRuntimeError(
                "RAW-02 public source scalar span 漂移")


def _course_payload(
        frame: PublicFrame,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> tuple[bytes, str]:
    """从 closure 取得唯一公开课程，并在 course cache 前重验 recipe SHA。"""
    relative_path = frame.recipe.course_relative_path
    payload = _payload_from_closure(
        source_payload_closure,
        relative_path,
        label="RAW-02 course",
    )
    if tuple(public_source_payload_sha256_v1(payload)) != frame.recipe.course_raw_sha256:
        raise ConversationRawAnswerRuntimeError("RAW-02 course SHA-256 漂移")
    return payload, relative_path


def _claim_input(frame: PublicFrame) -> GroundedAnswerClaimInput:
    """把已验证 source scalar 限制在旧 connector 的临时 Python text 适配边界。"""
    scalars = frame.recipe.claim_scalars
    encoded = encode_utf8_v1(scalars)
    try:
        claim_text = bytes(encoded).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConversationRawAnswerRuntimeError("RAW-02 claim UTF-8 适配失败") from error
    if tuple(ord(character) for character in claim_text) != scalars:
        raise ConversationRawAnswerRuntimeError("RAW-02 claim scalar readback 漂移")
    return GroundedAnswerClaimInput(claim_text)


def _run_actual(
        ingress: ConversationRawLexicalIngressResult,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        preparation_cache: PublicCoursePreparationCache | None = None,
        ) -> QuestionAnswerRun:
    """从公开课程真实装配并调用一次 QuestionAnswerRuntime。"""
    frame = ingress.frame
    request = ingress.request
    if frame is None or request is None:
        raise ConversationRawAnswerRuntimeError("RAW-02 缺完整 frame/request")
    if request.target_branch is None:
        raise ConversationRawAnswerRuntimeError("RAW-02 request 缺 target branch")
    candidate, planning = materialize_public_frame_candidate(frame, request)
    hypotheses = candidate.hypotheses
    if len(hypotheses) != 1:
        raise ConversationRawAnswerRuntimeError("RAW-02 attribution 需要唯一 hypothesis")
    _verify_frame_source_payloads(frame, source_payload_closure)
    payload, relative_path = _course_payload(frame, source_payload_closure)
    prepared = prepare_public_course(
        payload,
        course_relative_path=relative_path,
        course_raw_sha256=frame.recipe.course_raw_sha256,
        cache=preparation_cache,
    )
    model = prepared.model
    pack = prepared.pack
    pattern = pack.pattern(frame.recipe.pattern_id)
    if surface_pattern_structure_id(pattern) != frame.recipe.structure_id:
        raise ConversationRawAnswerRuntimeError("RAW-02 recipe pattern/structure 漂移")
    content, selector, plan_protocol, structure_protocol, surface_protocol, postcheck_protocol, question_protocol = (
        _generation_protocols(frame))
    claim = _claim_input(frame)
    target = GroundedAnswerConnectorTarget(
        candidate.proposition,
        request.target_branch,
        _REPRESENTATION_FAMILY,
    )
    compilation = compile_grounded_answer_connectors(
        model, claim, target, surface_protocol)
    selected_variant = compilation.select_within_structure(
        frame.recipe.structure_id,
        frame.recipe.pattern_id,
    )
    renderer_identity = _instruction_series(9, 1, frame=frame)[0]
    attribution = GenerationSurfaceAttribution(
        selected_variant.template.connector,
        hypotheses[0],
        _instruction_series(10, 1, frame=frame)[0],
    )
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        alias_factory = ProductionGenerationAliasRuntimeFactory(
            pack,
            ctx,
            visible_evidence_keys=tuple(sorted(
                item.stable_key() for item in candidate.evidence)),
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
            GroundedAnswerStructureVerifier(*_instruction_series(11, 2, frame=frame)),
            GroundedAnswerEvidenceSourceVerifier(
                *_instruction_series(12, 2, frame=frame)),
            question_protocol,
            EvidenceQuestionPostcheckMapper(
                (*_NAMESPACE, 60, 1),
                citation_required=True,
                trust_required=True,
            ),
            surface_attributions=(attribution,),
        )
        installation = GroundedAnswerRunLocalFactory(
            surface_protocol,
            _build_lifecycle(ctx, frame),
            components,
        ).build(GroundedAnswerRunLocalBuild(
            model,
            claim,
            target,
            planning,
            candidate,
            frame.recipe.structure_id,
            frame.recipe.pattern_id,
            GroundedAnswerParserProtocol(
                *_instruction_series(13, 5, frame=frame),
                content.answer,
            ),
            request.query_kind,
            _instruction_series(14, 1, frame=frame)[0],
            _instruction_series(15, 1, frame=frame)[0],
            (*_NAMESPACE, 61, 1),
        ))
        run = installation.runtime.run(request)
        if (not run.complete or run.generation is None
                or run.generation.rendered is None
                or run.postcheck is None or not run.postcheck.complete
                or run.status != content.answer
                or run.selection_commit is not None or run.outcome_commit is not None):
            raise ConversationRawAnswerRuntimeError("RAW-02 actual runtime 未完成 G-03/G-04")
        return run
    finally:
        backend.close()


def _run_actual_frame(
        ingress: ConversationRawLexicalIngressResult,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> QuestionAnswerRun:
    """按已冻结 recipe family 分派真实 RAW-02 runtime，不引入表层 fallback。"""
    frame = ingress.frame
    if frame is None:
        raise ConversationRawAnswerRuntimeError("RAW-02 缺 public frame")
    if isinstance(frame.recipe, PublicFrameRuntimeRecipe):
        return _run_actual(
            ingress,
            source_payload_closure=source_payload_closure,
            preparation_cache=preparation_cache,
        )
    if isinstance(frame.recipe, PublicFrameResponseActRuntimeRecipe):
        # 延迟导入让 V1 保持独立装配，也避免 catalog/runtime 形成模块初始化环。
        from pure_integer_ai.experiments.conversation_raw_response_act_runtime import (
            run_public_response_act_frame,
        )

        return run_public_response_act_frame(
            ingress,
            source_payload_closure=source_payload_closure,
            preparation_cache=preparation_cache,
        )
    if isinstance(frame.recipe, PublicFrameReferenceRuntimeRecipe):
        # V3 从无标签公开投影直接重建，不能复用 V1/V2 的课程准备缓存。
        from pure_integer_ai.experiments.conversation_raw_reference_runtime import (
            run_public_reference_frame,
        )

        return run_public_reference_frame(
            ingress,
            source_payload_closure=source_payload_closure,
            preflight_cache=preflight_cache,
        )
    raise ConversationRawAnswerRuntimeError("RAW-02 recipe family 未注册")


# object-model: value; representation=struct; interop=DLG-RAW-02
@dataclass(frozen=True, slots=True)
class ConversationRawAnswerResult:
    """真实 RAW-02 结果；成功输出仅为 G-03 rendered scalar 的 UTF-8 bytes。"""

    result_code: int
    ingress: ConversationRawLexicalIngressResult
    run: QuestionAnswerRun | None = None
    output_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    output_readback: ConversationRawIntake | None = None
    persistent_state_delta: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """防止失败结果泄露半句输出，并冻结零长期状态副作用。"""
        if type(self.result_code) is not int:
            raise ConversationRawAnswerRuntimeError("RAW-02 result code 必须是严格整数")
        if not isinstance(self.ingress, ConversationRawLexicalIngressResult):
            raise TypeError("RAW-02 ingress 类型错误")
        if (not isinstance(self.output_scalars, tuple)
                or any(type(item) is not int for item in self.output_scalars)
                or not isinstance(self.output_bytes, tuple)
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.output_bytes)
                or not isinstance(self.persistent_state_delta, tuple)
                or any(type(item) is not int for item in self.persistent_state_delta)
                or self.persistent_state_delta):
            raise ConversationRawAnswerRuntimeError("RAW-02 输出或长期状态 record 非法")
        if not self.ingress.accepted:
            if (self.result_code != self.ingress.result_code or self.run is not None
                    or self.output_scalars or self.output_bytes
                    or self.output_readback is not None):
                raise ConversationRawAnswerRuntimeError("RAW-02 ingress 拒绝未被保留")
            return
        if self.result_code == DLG_RAW_ACCEPT:
            if (not isinstance(self.run, QuestionAnswerRun) or not self.run.complete
                    or self.run.generation is None
                    or self.run.generation.rendered is None
                    or self.run.postcheck is None or not self.run.postcheck.complete
                    or not self.output_scalars or not self.output_bytes
                    or not isinstance(self.output_readback, ConversationRawIntake)
                    or not self.output_readback.accepted
                    or self.output_readback.unicode_scalars != self.output_scalars
                    or encode_utf8_v1(self.output_scalars) != self.output_bytes):
                raise ConversationRawAnswerRuntimeError("RAW-02 answer record 不闭合")
            return
        if self.result_code not in {
                DLG_RAW_REJECT_RUNTIME,
                DLG_RAW_REJECT_OUTPUT_BUDGET}:
            raise ConversationRawAnswerRuntimeError("RAW-02 result code 未注册")
        if (self.run is not None or self.output_scalars or self.output_bytes
                or self.output_readback is not None):
            raise ConversationRawAnswerRuntimeError("RAW-02 拒绝不得携带输出")

    @property
    def accepted(self) -> bool:
        """仅真实 runtime、G-04 与 UTF-8 readback 全部完成时接受。"""
        return self.result_code == DLG_RAW_ACCEPT

    def canonical_record(self) -> tuple[int, ...]:
        """导出可由整数与有限 byte sequence 重放的 RAW-02 结果记录。"""
        result = [RAW_ANSWER_RUNTIME_RECORD_V1, self.result_code]
        for value in (
                self.ingress.canonical_record(),
                (() if self.run is None else self.run.stable_key()),
                self.output_scalars,
                self.output_bytes,
                (() if self.output_readback is None
                 else self.output_readback.canonical_record()),
                self.persistent_state_delta):
            _pack(result, value)
        return tuple(result)


def run_public_frame_answer(
        ingress: ConversationRawLexicalIngressResult,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> ConversationRawAnswerResult:
    """执行 RAW-02，输出只能来自同次实际 ``rendered.units``，不含语言 fallback。"""
    if not isinstance(ingress, ConversationRawLexicalIngressResult):
        raise TypeError("RAW-02 ingress 类型错误")
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("RAW-02 source payload closure 类型错误")
    if not ingress.accepted:
        return ConversationRawAnswerResult(ingress.result_code, ingress)
    try:
        run = _run_actual_frame(
            ingress,
            source_payload_closure=source_payload_closure,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
        rendered = run.generation.rendered
        if rendered is None:
            raise ConversationRawAnswerRuntimeError("RAW-02 缺 rendered units")
        scalars = tuple(rendered.units)
        output = encode_utf8_v1(scalars)
        frame = ingress.frame
        if frame is None:
            raise ConversationRawAnswerRuntimeError("RAW-02 缺 public frame")
        if len(output) > frame.recipe.output_max_bytes:
            return ConversationRawAnswerResult(
                DLG_RAW_REJECT_OUTPUT_BUDGET, ingress)
        readback = intake_raw_conversation_vector(output)
        if not readback.accepted or readback.unicode_scalars != scalars:
            raise ConversationRawAnswerRuntimeError("RAW-02 output byte readback 漂移")
        return ConversationRawAnswerResult(
            DLG_RAW_ACCEPT,
            ingress,
            run,
            scalars,
            output,
            readback,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return ConversationRawAnswerResult(DLG_RAW_REJECT_RUNTIME, ingress)


__all__ = [
    "RAW_ANSWER_RUNTIME_RECORD_V1",
    "ConversationRawAnswerResult",
    "ConversationRawAnswerRuntimeError",
    "run_public_frame_answer",
]
