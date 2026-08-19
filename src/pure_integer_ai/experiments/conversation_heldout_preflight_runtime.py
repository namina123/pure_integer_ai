"""DLG-05 catalog 到真实两回合 runner 的无标签 turn factory。

该模块只编排 typed catalog、context read、可选 Memory read 和调用方注入的
QuestionAnswerRuntime builder。builder 必须从实际 request/read 构造 runtime；
本模块不携带 expected answer、surface 或 evaluator label。Memory ON 使用
runner 的 ``runtime_factory``，因此可以在同次 demand read 完成后再决定实际
G-01 stance。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
)
from pure_integer_ai.experiments.conversation_heldout_language_input import (
    ConversationQuestionInputCompiler,
)
from pure_integer_ai.experiments.conversation_heldout_family import (
    ConversationHeldOutInputCatalog,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_EXPLICIT_REPEAT,
    AXIS_OMISSION,
    AXIS_ORDER,
    AXIS_SYNONYM,
    AXIS_UNSEEN_RELATION,
    AXIS_UNSEEN_SOURCE,
    ConversationHeldOutCase,
    ConversationHeldOutTurn,
    MEMORY_OFF,
    MEMORY_ON,
)
from pure_integer_ai.experiments.conversation_heldout_relation_input import (
    ConversationRelationInputError,
    ConversationUnseenRelationCompiler,
)
from pure_integer_ai.experiments.conversation_heldout_source_input import (
    ConversationSourceInputError,
    ConversationUnseenSourceCompiler,
)
from pure_integer_ai.experiments.conversation_heldout_runtime import (
    ConversationHeldOutMemoryPlan,
    ConversationHeldOutResponseActResolver,
    ConversationHeldOutRuntimeError,
    ConversationHeldOutTurnPlan,
)
from pure_integer_ai.experiments.conversation_memory_demand_runtime import (
    MemoryDemandRead,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRun,
    QuestionAnswerRuntime,
)


class ConversationHeldOutPreflightRuntimeBuilder(Protocol):
    """调用方注入的真实 QuestionAnswerRuntime builder。"""

    def build(
            self,
            request: QuestionRequest,
            context_read: ConversationContextRead,
            memory_read: MemoryDemandRead | None,
            ) -> QuestionAnswerRuntime:
        """从当前 typed request/read 构造实际 runtime，不读取 label。"""
        ...


@dataclass(frozen=True, slots=True)
class MappedConversationHeldOutResponseActResolver:
    """把真实 G-01 stance identity 映射为公开 response-act protocol key。"""

    mappings: tuple[tuple[ObjectIdentity, ProtocolKey], ...]

    def __post_init__(self) -> None:
        """核验 stance 映射唯一且没有空协议键。"""
        if (not isinstance(self.mappings, tuple)
                or not self.mappings
                or any(not isinstance(item, tuple) or len(item) != 2
                       for item in self.mappings)):
            raise TypeError("response-act mappings 类型错误")
        if any(not isinstance(identity, ObjectIdentity)
               or not isinstance(key, ProtocolKey)
               or not key.components
               for identity, key in self.mappings):
            raise TypeError("response-act mappings 含非法项")
        identities = tuple(identity for identity, _ in self.mappings)
        if len(set(identities)) != len(identities):
            raise ValueError("response-act stance identity 不得重复")

    def __call__(
            self,
            run: QuestionAnswerRun,
            ) -> tuple[ObjectIdentity, ProtocolKey]:
        """只按同次 run.status 查找，不读取任何外部标签。"""
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("response-act resolver run 类型错误")
        for identity, key in self.mappings:
            if identity == run.status:
                return identity, key
        raise ConversationHeldOutRuntimeError(
            "真实 G-01 stance 未注册 response-act protocol key")


class ConversationHeldOutCatalogTurnFactory:
    """从 typed catalog 为 selection-first runner 提供真实 turn plan。"""

    def __init__(
            self,
            catalog: ConversationHeldOutInputCatalog,
            builder: ConversationHeldOutPreflightRuntimeBuilder,
            *,
            memory_plan_factory=None,
            response_act_resolver: ConversationHeldOutResponseActResolver,
            question_input_compiler: ConversationQuestionInputCompiler | None = None,
            relation_input_compiler: ConversationUnseenRelationCompiler | None = None,
            source_input_compiler: ConversationUnseenSourceCompiler | None = None,
            ) -> None:
        """绑定 catalog、runtime builder 和可选 Memory demand plan 工厂。"""
        if not isinstance(catalog, ConversationHeldOutInputCatalog):
            raise TypeError("catalog turn factory catalog 类型错误")
        if not hasattr(builder, "build"):
            raise TypeError("catalog turn factory builder 缺少 build")
        if not callable(response_act_resolver):
            raise TypeError("catalog turn factory response resolver 不可调用")
        if memory_plan_factory is not None and not callable(memory_plan_factory):
            raise TypeError("catalog turn factory memory plan factory 不可调用")
        if (question_input_compiler is not None
                and not isinstance(
                    question_input_compiler, ConversationQuestionInputCompiler)):
            raise TypeError("catalog turn factory question input compiler 类型错误")
        if (relation_input_compiler is not None
                and not isinstance(
                    relation_input_compiler, ConversationUnseenRelationCompiler)):
            raise TypeError("catalog turn factory relation input compiler 类型错误")
        if (source_input_compiler is not None
                and not isinstance(
                    source_input_compiler, ConversationUnseenSourceCompiler)):
            raise TypeError("catalog turn factory source input compiler 类型错误")
        self.catalog = catalog
        self.builder = builder
        self.memory_plan_factory = memory_plan_factory
        self.response_act_resolver = response_act_resolver
        self.question_input_compiler = question_input_compiler
        self.relation_input_compiler = relation_input_compiler
        self.source_input_compiler = source_input_compiler

    def _catalog_turn(
            self,
            case: ConversationHeldOutCase,
            turn: ConversationHeldOutTurn,
            ):
        """按完整 case/turn 身份读取 typed catalog，并核验三类身份。"""
        try:
            item = self.catalog.turn_for(case.case_key, turn.turn_key)
        except (KeyError, RuntimeError, ValueError) as error:
            raise ConversationHeldOutRuntimeError(
                "catalog turn factory 找不到 typed turn") from error
        if (item.content != turn.content
                or item.source_key != turn.source_key
                or item.scope_key != turn.scope_key
                or item.memory_mode != turn.memory_mode
                or item.context_mode != turn.context_mode
                or item.reference_mode != turn.reference_mode
                or item.rollback_mode != turn.rollback_mode):
            raise ConversationHeldOutRuntimeError(
                "catalog typed turn 与 manifest turn 漂移")
        return item

    def __call__(
            self,
            case: ConversationHeldOutCase,
            turn: ConversationHeldOutTurn,
            context_read: ConversationContextRead,
            ) -> ConversationHeldOutTurnPlan:
        """返回当前回合 plan；Memory ON runtime 延迟到同次 read 后装配。"""
        if not isinstance(case, ConversationHeldOutCase):
            raise TypeError("catalog turn factory case 类型错误")
        if not isinstance(turn, ConversationHeldOutTurn):
            raise TypeError("catalog turn factory turn 类型错误")
        if not isinstance(context_read, ConversationContextRead):
            raise TypeError("catalog turn factory context read 类型错误")
        item = self._catalog_turn(case, turn)
        request = item.request
        proven_axes = []
        if (item.language_input is not None
                and turn.memory_mode == MEMORY_ON
                and self.memory_plan_factory is None):
            raise ConversationHeldOutRuntimeError(
                "Memory ON 缺少 typed memory plan")
        if item.language_input is not None:
            if self.question_input_compiler is None:
                raise ConversationHeldOutRuntimeError(
                    "typed language input 缺少 question compiler")
            request = self.question_input_compiler.compile(
                item.language_input, context_read=context_read)
            if request != item.request:
                raise ConversationHeldOutRuntimeError(
                    "typed language input 编译结果与 catalog request 漂移")
            if AXIS_SYNONYM in case.axis_keys:
                proven_axes.append(AXIS_SYNONYM)
            if AXIS_ORDER in case.axis_keys:
                proven_axes.append(AXIS_ORDER)
            if (AXIS_OMISSION in case.axis_keys
                    and item.language_input.provided_positions):
                proven_axes.append(AXIS_OMISSION)
        if (AXIS_EXPLICIT_REPEAT in case.axis_keys and turn.ordinal > 1
                and item.request == self.catalog.turn_for(
                    case.case_key, case.turns[0].turn_key).request):
            proven_axes.append(AXIS_EXPLICIT_REPEAT)
        if AXIS_UNSEEN_RELATION in case.axis_keys:
            if self.relation_input_compiler is None:
                raise ConversationHeldOutRuntimeError(
                    "unseen relation turn 缺少 TRAIN structure compiler")
            try:
                self.relation_input_compiler.compile(request)
            except ConversationRelationInputError as error:
                raise ConversationHeldOutRuntimeError(
                    "unseen relation typed structure 未通过 TRAIN 分母鉴权"
                ) from error
            proven_axes.append(AXIS_UNSEEN_RELATION)
        if AXIS_UNSEEN_SOURCE in case.axis_keys:
            if self.source_input_compiler is None:
                raise ConversationHeldOutRuntimeError(
                    "unseen source turn 缺少 TRAIN source compiler")
            try:
                self.source_input_compiler.compile(request)
            except ConversationSourceInputError as error:
                raise ConversationHeldOutRuntimeError(
                    "unseen source 未通过 TRAIN 分母鉴权") from error
            proven_axes.append(AXIS_UNSEEN_SOURCE)
        if turn.memory_mode == MEMORY_OFF:
            runtime = self.builder.build(request, context_read, None)
            if not isinstance(runtime, QuestionAnswerRuntime):
                raise TypeError("catalog turn factory builder 返回错误 runtime")
            return ConversationHeldOutTurnPlan(
                runtime,
                request,
                None,
                None,
                response_act_resolver=self.response_act_resolver,
                input_content=item.content,
                proven_axis_keys=tuple(proven_axes),
            )
        if turn.memory_mode != MEMORY_ON:
            raise ConversationHeldOutRuntimeError(
                "catalog turn factory memory mode 未注册")
        if self.memory_plan_factory is None:
            raise ConversationHeldOutRuntimeError(
                "Memory ON 缺少 typed memory plan factory")
        memory = self.memory_plan_factory(case, turn, context_read)
        if not isinstance(memory, ConversationHeldOutMemoryPlan):
            raise TypeError("catalog turn factory memory plan 类型错误")
        if (memory.current.source != request.source
                or memory.current.scope != request.response_scope):
            raise ConversationHeldOutRuntimeError(
                "catalog Memory plan 的 current source/scope 未绑定当前 request")

        def build_after_read(read: MemoryDemandRead) -> QuestionAnswerRuntime:
            """只消费 runner 刚完成的 Memory read，再构造真实 runtime。"""
            if not isinstance(read, MemoryDemandRead):
                raise TypeError("catalog turn factory memory read 类型错误")
            runtime = self.builder.build(request, context_read, read)
            if not isinstance(runtime, QuestionAnswerRuntime):
                raise TypeError("catalog turn factory builder 返回错误 runtime")
            return runtime

        return ConversationHeldOutTurnPlan(
            None,
            request,
            None,
            None,
            memory=memory,
            runtime_factory=build_after_read,
            response_act_resolver=self.response_act_resolver,
            input_content=item.content,
            proven_axis_keys=tuple(proven_axes),
        )


__all__ = [
    "ConversationHeldOutCatalogTurnFactory",
    "ConversationHeldOutPreflightRuntimeBuilder",
    "MappedConversationHeldOutResponseActResolver",
]
