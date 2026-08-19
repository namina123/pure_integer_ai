"""把 DLG-05 typed family 接到真实问答与 DLG-04 Memory consumer。

本模块只负责 selection-first 的运行编排。它不接收 evaluator label，也不
构造 expected answer；每个回合必须由真实 ``QuestionAnswerRuntime`` 完成
G-00 至 G-04，Memory ON 才允许调用 ``ConversationMemoryDemandConsumer``。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_EVENT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
    ConversationContextState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_CONFLICT,
    AXIS_EXPLICIT_REPEAT,
    AXIS_MEMORY_MISS,
    AXIS_OMISSION,
    AXIS_PROPOSITION_REFERENCE,
    AXIS_ROLLBACK,
    AXIS_SCOPE_DRIFT,
    AXIS_EVENT_REFERENCE,
    AXIS_UNSEEN_RELATION,
    AXIS_UNSEEN_SOURCE,
    AXIS_ORDER,
    AXIS_SYNONYM,
    ConversationHeldOutCase,
    ConversationHeldOutManifest,
    ConversationHeldOutObservation,
    ConversationHeldOutTurn,
    CONTEXT_CARRY,
    CONTEXT_EXPLICIT_REPEAT,
    CONTEXT_FRESH,
    CONTEXT_SCOPE_CHANGE,
    MEMORY_OFF,
    MEMORY_ON,
    REFERENCE_EVENT,
    REFERENCE_PROPOSITION,
    RESPONSE_CONFLICT,
    ROLLBACK_READ_ONLY,
    run_selection_first,
)
from pure_integer_ai.experiments.conversation_memory_demand_runtime import (
    ConversationMemoryDemandConsumer,
    MemoryDemandRead,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    MemoryExpansionProfile,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalMemoryCenter,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRun,
    QuestionAnswerRuntime,
)


_MEMORY_TRACE_DOMAIN = "conversation.heldout.memory.receipt.v1"
_CONTEXT_KEY_DOMAIN = "conversation.heldout.context.v1"
SELECTION_FIRST_LABEL_FREE_CONTRACT_KEY = (31005, 1300, 1)


class ConversationHeldOutRuntimeError(RuntimeError):
    """真实对话编排的 typed 合同未闭合。"""


@dataclass(frozen=True, slots=True)
class ConversationHeldOutSelectionReceipt:
    """selection-first 运行 receipt；其接口不接收 evaluator label。"""

    manifest_key: tuple[int, ...]
    observations: tuple[ConversationHeldOutObservation, ...]
    contract_key: tuple[int, ...] = SELECTION_FIRST_LABEL_FREE_CONTRACT_KEY

    def __post_init__(self) -> None:
        """核验运行 receipt 只包含无标签 observation 和固定协议身份。"""
        if (not isinstance(self.manifest_key, tuple)
                or not self.manifest_key
                or any(type(item) is not int or item < 0
                       for item in self.manifest_key)):
            raise ConversationHeldOutRuntimeError(
                "selection-first receipt manifest key 非法")
        if (not isinstance(self.observations, tuple)
                or not self.observations
                or any(not isinstance(item, ConversationHeldOutObservation)
                       for item in self.observations)):
            raise ConversationHeldOutRuntimeError(
                "selection-first receipt observations 非法")
        if (not isinstance(self.contract_key, tuple)
                or self.contract_key != SELECTION_FIRST_LABEL_FREE_CONTRACT_KEY):
            raise ConversationHeldOutRuntimeError(
                "selection-first receipt label-free contract 漂移")

    def stable_key(self) -> tuple[int, ...]:
        """返回无标签 selection-first receipt 的纯整数键。"""
        result = [1, len(self.manifest_key), *self.manifest_key]
        result.extend((len(self.contract_key), *self.contract_key,
                       len(self.observations)))
        for observation in self.observations:
            key = observation.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


class ConversationHeldOutResponseActResolver(Protocol):
    """从同次 QuestionAnswerRun 解析 response-act，不读取 label。"""

    def __call__(
            self,
            run: QuestionAnswerRun,
            ) -> tuple[ObjectIdentity, ProtocolKey]:
        """返回实际 stance identity 与公开 response-act protocol key。"""
        ...


@dataclass(frozen=True, slots=True)
class ConversationHeldOutMemoryPlan:
    """一个回合的真实 DLG-04 read 输入。"""

    consumer: ConversationMemoryDemandConsumer
    current: MemoryCurrentQuery
    center: DirectionalMemoryCenter
    profile: MemoryExpansionProfile
    access: MemoryAccessContext
    release: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.consumer, ConversationMemoryDemandConsumer):
            raise TypeError("held-out memory plan consumer 类型错误")
        if not isinstance(self.current, MemoryCurrentQuery):
            raise TypeError("held-out memory plan current 类型错误")
        if not isinstance(self.center, DirectionalMemoryCenter):
            raise TypeError("held-out memory plan center 类型错误")
        if not isinstance(self.profile, MemoryExpansionProfile):
            raise TypeError("held-out memory plan profile 类型错误")
        if not isinstance(self.access, MemoryAccessContext):
            raise TypeError("held-out memory plan access 类型错误")
        if self.release is not None and not callable(self.release):
            raise TypeError("held-out memory plan release 必须可调用")

    def release_resources(self) -> None:
        """释放本回合为 DLG-04 read 打开的 query 资源。

        release 属于 fixture/runtime owner，不是语义输入；调用方必须提供幂等
        callback。没有 callback 的既有静态 plan 保持原有生命周期行为。
        """
        if self.release is not None:
            self.release()


@dataclass(frozen=True, slots=True)
class ConversationHeldOutTurnPlan:
    """一个 typed turn 的真实 runtime、request 和可选 Memory demand。"""

    runtime: QuestionAnswerRuntime | None
    request: QuestionRequest
    response_act_identity: ObjectIdentity | None
    response_act_key: ProtocolKey | None
    memory: ConversationHeldOutMemoryPlan | None = None
    runtime_factory: "ConversationHeldOutRuntimeFactory | None" = None
    response_act_resolver: ConversationHeldOutResponseActResolver | None = None
    input_content: CanonicalIdentity | None = None
    proven_axis_keys: tuple[ProtocolKey, ...] = ()

    def __post_init__(self) -> None:
        if (self.runtime is None
                and self.runtime_factory is None):
            raise TypeError(
                "held-out turn 必须提供 runtime 或 runtime_factory")
        if (self.runtime is not None
                and not isinstance(self.runtime, QuestionAnswerRuntime)):
            raise TypeError("held-out turn runtime 类型错误")
        if not isinstance(self.request, QuestionRequest):
            raise TypeError("held-out turn request 类型错误")
        if self.response_act_identity is not None and not isinstance(
                self.response_act_identity, ObjectIdentity):
            raise TypeError("held-out response identity 类型错误")
        if self.response_act_key is not None and not isinstance(
                self.response_act_key, ProtocolKey):
            raise TypeError("held-out response key 类型错误")
        if self.memory is not None and not isinstance(
                self.memory, ConversationHeldOutMemoryPlan):
            raise TypeError("held-out memory plan 类型错误")
        if (self.runtime_factory is not None
                and not callable(self.runtime_factory)):
            raise TypeError("held-out runtime_factory 必须可调用")
        if (self.response_act_resolver is not None
                and not callable(self.response_act_resolver)):
            raise TypeError("held-out response_act_resolver 必须可调用")
        if ((self.response_act_identity is None)
                != (self.response_act_key is None)
                and self.response_act_resolver is None):
            raise TypeError(
                "显式 response-act identity/key 必须成对，或安装 resolver")
        if (self.response_act_resolver is not None
                and self.response_act_identity is not None
                and self.response_act_key is not None):
            raise TypeError(
                "response-act resolver 不得与显式 identity/key 混用")
        if self.input_content is not None and not isinstance(
                self.input_content, CanonicalIdentity):
            raise TypeError("held-out input content 类型错误")
        if (not isinstance(self.proven_axis_keys, tuple)
                or any(not isinstance(item, ProtocolKey)
                       for item in self.proven_axis_keys)
                or len(set(self.proven_axis_keys)) != len(self.proven_axis_keys)):
            raise TypeError("held-out proven axis keys 类型错误")


class ConversationHeldOutRuntimeFactory(Protocol):
    """在同次 Memory read 完成后装配真实 QuestionAnswerRuntime。"""

    def __call__(self, read: MemoryDemandRead) -> QuestionAnswerRuntime:
        """返回消费该 read 的生产问答 runtime。"""
        ...


class ConversationHeldOutTurnFactory(Protocol):
    """按无标签 case/turn/context read 提供真实执行计划。"""

    def __call__(
            self,
            case: ConversationHeldOutCase,
            turn: ConversationHeldOutTurn,
            context_read: ConversationContextRead,
            ) -> ConversationHeldOutTurnPlan:
        """返回当前回合实际使用的 QuestionAnswer/Memory 组件。"""
        ...


def conversation_turn_content_identity(
        request: QuestionRequest,
        *,
        language_input_key: tuple[int, ...] = (),
        ) -> CanonicalIdentity:
    """返回冻结 turn 的 QuestionRequest 与可选真实语言输入身份。"""
    if not isinstance(request, QuestionRequest):
        raise TypeError("conversation turn content request 类型错误")
    if (not isinstance(language_input_key, tuple)
            or any(type(item) is not int for item in language_input_key)):
        raise TypeError("conversation turn language input key 类型错误")
    return CanonicalIdentity.from_value(
        ("QUESTION_REQUEST_V2", request.stable_key(), language_input_key))


def conversation_turn_source_key(request: QuestionRequest) -> ProtocolKey:
    """返回冻结 turn 必须保存的目标知识来源完整键。"""
    if not isinstance(request, QuestionRequest):
        raise TypeError("conversation turn source request 类型错误")
    return ProtocolKey(request.source.stable_key())


def conversation_turn_scope_key(request: QuestionRequest) -> ProtocolKey:
    """返回冻结 turn 必须保存的实际回答 scope 完整键。"""
    if not isinstance(request, QuestionRequest):
        raise TypeError("conversation turn scope request 类型错误")
    return ProtocolKey(request.response_scope.stable_key())


def _validate_turn_request(
        turn: ConversationHeldOutTurn,
        request: QuestionRequest,
        input_content: CanonicalIdentity | None = None,
        ) -> None:
    """拒绝用其他内容、来源或 scope 的 request 替换冻结 turn。"""
    actual_content = (
        conversation_turn_content_identity(request)
        if input_content is None else input_content
    )
    if turn.content != actual_content:
        raise ConversationHeldOutRuntimeError(
            "held-out turn content 与实际 QuestionRequest 不一致")
    if turn.source_key != conversation_turn_source_key(request):
        raise ConversationHeldOutRuntimeError(
            "held-out turn source 与实际 QuestionRequest 不一致")
    if turn.scope_key != conversation_turn_scope_key(request):
        raise ConversationHeldOutRuntimeError(
            "held-out turn scope 与实际 QuestionRequest 不一致")


def _context_key(case: ConversationHeldOutCase) -> tuple[int, ...]:
    """为每个 held-out case 生成独立、可重放的纯整数会话键。"""
    return integer_tuple_fingerprint(
        (*case.family_key.components, 0, *case.case_key.components),
        domain=_CONTEXT_KEY_DOMAIN,
    )


def _bound_event_keys(
        proposition: BoundProposition,
        ) -> tuple[tuple[int, ...], ...]:
    """递归读取实际 BoundProposition 中的 Event filler。"""
    if not isinstance(proposition, BoundProposition):
        raise ConversationHeldOutRuntimeError(
            "event reference 目标必须是 BoundProposition")
    found: list[tuple[int, ...]] = []
    for binding in proposition.bindings:
        filler = binding.filler
        if isinstance(filler, BoundProposition):
            found.extend(_bound_event_keys(filler))
        elif filler.object_kind == OBJECT_EVENT:
            found.append(filler.stable_key())
    return tuple(sorted(set(found)))


def _bind_memory_receipt(
        request: QuestionRequest,
        read: MemoryDemandRead,
        ) -> QuestionRequest:
    """把真实 Memory receipt 的固定摘要绑定到同次 question trace。"""
    suffix = integer_tuple_fingerprint(
        read.receipt.stable_key(), domain=_MEMORY_TRACE_DOMAIN)
    if request.trace[-len(suffix):] == suffix:
        raise ConversationHeldOutRuntimeError(
            "同一 Memory receipt 不得重复绑定 request")
    return replace(request, trace=(*request.trace, *suffix))


def _run_case(
        case: ConversationHeldOutCase,
        prepare: ConversationHeldOutTurnFactory,
        *,
        visible_limit: int,
        ) -> ConversationHeldOutObservation:
    """执行一个 case 的全部实际回合并形成无标签 observation。"""
    if type(visible_limit) is not int or visible_limit < 0:
        raise ConversationHeldOutRuntimeError("visible_limit 非法")
    context: ConversationContextState = start_conversation_context(
        _context_key(case))
    final_response_key = None
    final_selected = ()
    final_cited = ()
    receipt_keys = []
    turn_response_keys = []
    proven_axis_keys: set[ProtocolKey] = set()
    memory_on_seen = False
    memory_causal = True
    previous_request = None
    active_memory = None
    rollback_backend = None
    rollback_snapshot = None
    try:
        for turn in case.turns:
            if turn.ordinal != context.revision + 1:
                raise ConversationHeldOutRuntimeError(
                    "held-out turn ordinal 与 context revision 不连续")
            context_read = context.read(visible_limit)
            plan = prepare(case, turn, context_read)
            if not isinstance(plan, ConversationHeldOutTurnPlan):
                raise TypeError("held-out turn factory 返回错误 plan")
            if any(axis not in case.axis_keys for axis in plan.proven_axis_keys):
                raise ConversationHeldOutRuntimeError(
                    "held-out turn proof axis 不属于当前 case")
            active_memory = plan.memory
            if turn.rollback_mode == ROLLBACK_READ_ONLY:
                if active_memory is None:
                    raise ConversationHeldOutRuntimeError(
                        "rollback read-only 回合缺少 Memory owner")
                rollback_backend = active_memory.consumer.ctx.backend
                rollback_snapshot = rollback_backend.snapshot()
            _validate_turn_request(turn, plan.request, plan.input_content)
            request = plan.request
            if turn.ordinal == 1:
                if turn.context_mode != CONTEXT_FRESH:
                    raise ConversationHeldOutRuntimeError(
                        "首回合 context 必须是 FRESH")
            else:
                if turn.context_mode == CONTEXT_FRESH:
                    raise ConversationHeldOutRuntimeError(
                        "后续回合不得伪装 FRESH context")
                if previous_request is None:
                    raise ConversationHeldOutRuntimeError(
                        "后续回合缺少 previous request")
                scope_changed = (
                    request.response_scope != previous_request.response_scope)
                if (turn.context_mode == CONTEXT_SCOPE_CHANGE) != scope_changed:
                    raise ConversationHeldOutRuntimeError(
                        "context scope-change mode 与实际 response scope 漂移")
                if turn.context_mode in {
                        CONTEXT_CARRY, CONTEXT_EXPLICIT_REPEAT} and scope_changed:
                    raise ConversationHeldOutRuntimeError(
                        "carry/repeat context 不得替换 response scope")
                if (turn.context_mode == CONTEXT_SCOPE_CHANGE
                        and request.response_scope != previous_request.response_scope):
                    proven_axis_keys.add(AXIS_SCOPE_DRIFT)
                if turn.reference_mode == REFERENCE_PROPOSITION \
                        and request.target != previous_request.target:
                    raise ConversationHeldOutRuntimeError(
                        "proposition reference 未复用前回合 typed target")
                if turn.reference_mode == REFERENCE_PROPOSITION:
                    proven_axis_keys.add(AXIS_PROPOSITION_REFERENCE)
                if turn.reference_mode == REFERENCE_EVENT:
                    current_events = _bound_event_keys(request.target)
                    previous_events = _bound_event_keys(previous_request.target)
                    if (not current_events or current_events != previous_events
                            or request.target == previous_request.target):
                        raise ConversationHeldOutRuntimeError(
                            "event reference 未由不同命题复用前回合真实 Event")
                    proven_axis_keys.add(AXIS_EVENT_REFERENCE)
            memory_read = None
            if turn.memory_mode == MEMORY_ON:
                memory_on_seen = True
                if plan.memory is None:
                    raise ConversationHeldOutRuntimeError(
                        "Memory ON 回合缺少真实 demand plan")
                memory_read = plan.memory.consumer.read(
                    plan.memory.current,
                    plan.memory.center,
                    plan.memory.profile,
                    access=plan.memory.access,
                    context_read=context_read,
                )
                if (plan.runtime_factory is not None
                        and (memory_read.receipt.source != request.source
                             or memory_read.receipt.scope
                             != request.response_scope)):
                    raise ConversationHeldOutRuntimeError(
                        "causal Memory runtime 的 source/scope 未绑定当前 request")
                receipt_keys.append(memory_read.receipt.stable_key())
                request = _bind_memory_receipt(request, memory_read)
                if turn.rollback_mode == ROLLBACK_READ_ONLY:
                    if (memory_read.receipt.rollback_before_key
                            != memory_read.receipt.rollback_after_key):
                        raise ConversationHeldOutRuntimeError(
                            "rollback read-only receipt 未闭合")
            elif turn.memory_mode == MEMORY_OFF:
                if plan.memory is not None:
                    raise ConversationHeldOutRuntimeError(
                        "Memory OFF 回合不得安装 demand plan")
            else:
                raise ConversationHeldOutRuntimeError("held-out memory mode 未注册")
            # Context suffix 必须最后写入，append_consumed 才能精确核验本次 read。
            request = context_read.bind_request(request)
            runtime = plan.runtime
            if plan.runtime_factory is not None:
                if memory_read is None:
                    raise ConversationHeldOutRuntimeError(
                        "runtime_factory 只能绑定 Memory ON read")
                runtime = plan.runtime_factory(memory_read)
                if not isinstance(runtime, QuestionAnswerRuntime):
                    raise TypeError("held-out runtime_factory 返回错误 runtime")
            if runtime is None:
                raise ConversationHeldOutRuntimeError(
                    "held-out turn 缺少可执行 QuestionAnswerRuntime")
            run = runtime.run(request)
            if not isinstance(run, QuestionAnswerRun) or not run.complete:
                raise ConversationHeldOutRuntimeError(
                    "held-out 回合必须完成真实 QuestionAnswerRun/G-04")
            response_identity = plan.response_act_identity
            response_key = plan.response_act_key
            if plan.response_act_resolver is not None:
                resolved = plan.response_act_resolver(run)
                if (not isinstance(resolved, tuple) or len(resolved) != 2
                        or not isinstance(resolved[0], ObjectIdentity)
                        or not isinstance(resolved[1], ProtocolKey)):
                    raise ConversationHeldOutRuntimeError(
                        "response-act resolver 返回错误身份")
                response_identity, response_key = resolved
            if response_identity is None or response_key is None:
                raise ConversationHeldOutRuntimeError(
                    "held-out response-act 未解析")
            if run.status != response_identity:
                raise ConversationHeldOutRuntimeError(
                    "held-out response-act 与真实 G-01 stance 漂移")
            if run.postcheck is None or run.postcheck.parsed.observation is None:
                raise ConversationHeldOutRuntimeError(
                    "held-out 回合缺少实际 parser/readback observation")
            observation = run.postcheck.parsed.observation
            proven_axis_keys.update(plan.proven_axis_keys)
            if response_key == RESPONSE_CONFLICT and AXIS_CONFLICT in case.axis_keys:
                proven_axis_keys.add(AXIS_CONFLICT)
            if (memory_read is not None
                    and memory_read.receipt.status == "UNKNOWN"
                    and AXIS_MEMORY_MISS in case.axis_keys):
                proven_axis_keys.add(AXIS_MEMORY_MISS)
            if memory_read is not None:
                memory_candidate_keys = {
                    evidence.candidate.stable_key()
                    for candidate in run.planning_request.candidates
                    for evidence in candidate.memory_evidence
                }
                selected_memory_keys = set(
                    memory_read.receipt.selected_candidate_keys)
                if memory_read.receipt.status == "UNKNOWN":
                    memory_causal &= not memory_candidate_keys
                elif memory_read.receipt.status == "HIT":
                    memory_causal &= bool(memory_candidate_keys)
                    memory_causal &= memory_candidate_keys.issubset(
                        selected_memory_keys)
                else:
                    memory_causal = False
                if plan.runtime_factory is not None and not memory_causal:
                    raise ConversationHeldOutRuntimeError(
                        "causal Memory runtime 未携带本次 read 的 generation evidence")
            final_response_key = response_key
            final_selected = run.selection.selected_candidate_keys
            final_cited = tuple(
                source.stable_key() for source in observation.cited_sources)
            turn_response_keys.append(response_key)
            context = context.append_consumed(run, context_read)
            previous_request = plan.request
            if active_memory is not None:
                active_memory.release_resources()
                active_memory = None
            if rollback_backend is not None:
                if rollback_backend.snapshot() != rollback_snapshot:
                    rollback_backend.load_snapshot(rollback_snapshot)
                    raise ConversationHeldOutRuntimeError(
                        "rollback read-only 回合改变了持久后端")
                rollback_backend = None
                rollback_snapshot = None
                proven_axis_keys.add(AXIS_ROLLBACK)
    except Exception as error:
        cleanup_error = None
        if active_memory is not None:
            try:
                active_memory.release_resources()
            except Exception as release_error:  # pragma: no cover - 双故障护栏
                cleanup_error = release_error
            active_memory = None
        if rollback_backend is not None and rollback_snapshot is not None:
            try:
                if rollback_backend.snapshot() != rollback_snapshot:
                    rollback_backend.load_snapshot(rollback_snapshot)
                if rollback_backend.snapshot() != rollback_snapshot:
                    raise ConversationHeldOutRuntimeError(
                        "rollback 故障恢复后持久后端仍漂移")
            except Exception as restore_error:  # pragma: no cover - 双故障护栏
                cleanup_error = restore_error
        if cleanup_error is not None:
            raise ConversationHeldOutRuntimeError(
                "held-out 故障后的资源或后端恢复失败") from cleanup_error
        raise error.with_traceback(error.__traceback__)
    if final_response_key is None:
        raise ConversationHeldOutRuntimeError("held-out case 不得为空")
    return ConversationHeldOutObservation(
        case.case_key,
        final_response_key,
        tuple(turn_response_keys),
        final_selected,
        final_cited,
        context.revision,
        tuple(receipt_keys),
        int(memory_on_seen and memory_causal),
        tuple(sorted(proven_axis_keys, key=lambda item: item.components)),
    )


def run_real_selection_first(
        manifest: ConversationHeldOutManifest,
        prepare: ConversationHeldOutTurnFactory,
        *,
        visible_limit: int = 8,
        ) -> tuple[ConversationHeldOutObservation, ...]:
    """在不接触 labels 的前提下运行真实 DLG-05 selection-first family。"""
    if not isinstance(manifest, ConversationHeldOutManifest):
        raise TypeError("held-out runtime manifest 类型错误")
    if not callable(prepare):
        raise TypeError("held-out runtime prepare 必须可调用")
    return run_selection_first(
        manifest,
        lambda case: _run_case(
            case, prepare, visible_limit=visible_limit),
    )


def run_real_selection_first_receipt(
        manifest: ConversationHeldOutManifest,
        prepare: ConversationHeldOutTurnFactory,
        *,
        visible_limit: int = 8,
        ) -> ConversationHeldOutSelectionReceipt:
    """执行 selection-first 并返回固定的无标签运行 receipt。"""
    if not isinstance(manifest, ConversationHeldOutManifest):
        raise TypeError("held-out runtime manifest 类型错误")
    if not callable(prepare):
        raise TypeError("held-out runtime prepare 必须可调用")
    return ConversationHeldOutSelectionReceipt(
        manifest.stable_key(),
        run_real_selection_first(
            manifest, prepare, visible_limit=visible_limit),
    )


def run_real_selection_first_case(
        case: ConversationHeldOutCase,
        prepare: ConversationHeldOutTurnFactory,
        *,
        visible_limit: int = 8,
        ) -> ConversationHeldOutObservation:
    """运行单一已冻结 case 的真实 selection-first 核心。

    该入口只用于公开施工 preflight；正式 family 仍必须使用完整 manifest，
    以保留 required axes、Memory mode 和 label-late 分母校验。
    """
    if not isinstance(case, ConversationHeldOutCase):
        raise TypeError("held-out runtime case 类型错误")
    if not callable(prepare):
        raise TypeError("held-out runtime prepare 必须可调用")
    return _run_case(case, prepare, visible_limit=visible_limit)


__all__ = [
    "ConversationHeldOutMemoryPlan",
    "ConversationHeldOutSelectionReceipt",
    "ConversationHeldOutRuntimeError",
    "ConversationHeldOutResponseActResolver",
    "ConversationHeldOutRuntimeFactory",
    "ConversationHeldOutTurnFactory",
    "ConversationHeldOutTurnPlan",
    "conversation_turn_content_identity",
    "conversation_turn_scope_key",
    "conversation_turn_source_key",
    "run_real_selection_first",
    "run_real_selection_first_receipt",
    "run_real_selection_first_case",
    "SELECTION_FIRST_LABEL_FREE_CONTRACT_KEY",
]
