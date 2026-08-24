"""DLG-RAW-13：将 DLG-RAW-12 carrier 显式投影为统一终端 response record。

这是一层独立的 outer state，不改写 V4 mixed state 或 V3 focus ledger。它只把已经
完成的 carrier 转为终端可复制的 raw-u8 response；只有 raw lexical miss/ambiguity
可由公开课程转换为 coverage/route clarification act。Python 类型检查仅在旧 carrier
adapter 边界消除既有 union，之后的 response/state/turn 均由闭合整数 tag 定义。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    ProviderOriginFollowupResultV1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_public_terminal_dialogue_act_catalog import (
    PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1,
    PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1,
    PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1,
    PublicTerminalDialogueActCatalogError,
    TerminalDialogueActCatalogV1,
    TerminalDialogueActFormV1,
    load_public_terminal_dialogue_act_catalog_from_closure,
    validate_public_terminal_dialogue_act_catalog_v1,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    ConversationRawAnswerResult,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    ConversationRawIntake,
    ConversationRawIntakeError,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_session import (
    ConversationRawMixedFocusDialogueStateV1,
    ConversationRawMixedFocusDialogueTurnV1,
    run_public_mixed_focus_dialogue_turn_v1,
    start_public_mixed_focus_dialogue,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PublicProofSentenceProviderResultV1,
)


TERMINAL_DIALOGUE_ACT_RUNTIME_PROTOCOL_V1 = 1
TERMINAL_DIALOGUE_ACT_RUNTIME_BINDING_RECORD_V1 = 1
TERMINAL_DIALOGUE_ACT_RUNTIME_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-13/TERMINAL-DIALOGUE-ACT-RUNTIME/V1")
TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1 = 1
TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1 = 1
TERMINAL_DIALOGUE_RESPONSE_RECORD_V1 = 1
TERMINAL_DIALOGUE_RESPONSE_SCHEMA_RECORD_V1 = 1
TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1 = 1
TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1 = 2
TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1 = 1
TERMINAL_DIALOGUE_BASE_CARRIER_PROVIDER_V1 = 2
TERMINAL_DIALOGUE_BASE_CARRIER_FOLLOWUP_V1 = 3
TERMINAL_DIALOGUE_ACT_NONE_V1 = 0

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-13
class ConversationRawTerminalDialogueActError(ValueError):
    """DLG-RAW-13 outer state、response carrier 或公开 catalog 不闭合。"""


def _u64(value: int, *, label: str, minimum: int = 0) -> int:
    """验证所有 protocol scalar 的显式 unsigned 64-bit 边界。"""
    if (type(value) is not int or value < minimum
            or value >= _U64_EXCLUSIVE):
        raise ConversationRawTerminalDialogueActError(
            f"{label} 必须是范围内的严格 u64")
    return value


def _record(value: tuple[int, ...], *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """验证有限非负整数 record，拒绝 Python list/bool/negative 值。"""
    if (type(value) is not tuple or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawTerminalDialogueActError(
            f"{label} 不是{'可空' if allow_empty else '非空'}非负整数 record")
    return value


def _u8(value: tuple[int, ...], *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """验证 immutable raw-u8 vector，禁止宿主 str/bytearray 进入逻辑 record。"""
    if (type(value) is not tuple or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawTerminalDialogueActError(
            f"{label} 不是{'可空' if allow_empty else '非空'} u8 vector")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    """以明确 count framing 写入子 record，所有元素都必须是非负整数。"""
    values = _record(value, label=label, allow_empty=True)
    _u64(len(values), label=f"{label} count")
    result.extend((len(values), *values))


def _identity(record: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """从版本化 runtime binding 形成 portable SHA-256 raw-u8 identity。"""
    try:
        value = tuple(portable_sha256_v1(
            TERMINAL_DIALOGUE_ACT_RUNTIME_IDENTITY_DOMAIN_V1,
            (record,),
        ))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ConversationRawTerminalDialogueActError(
            f"{label} identity 无法形成") from error
    return _u8(value, label=f"{label} identity", allow_empty=False)


def _ascii_nonnegative_integer(value: int) -> tuple[int, ...]:
    """以显式十进制整数协议形成原 reject 的固定 ASCII surface。"""
    _u64(value, label="terminal base result code")
    if value == 0:
        return (0x30,)
    digits: list[int] = []
    current = value
    while current:
        digits.append(0x30 + current % 10)
        current //= 10
    return tuple(reversed(digits))


def _reject_surface(result_code: int) -> tuple[int, ...]:
    """构造既有 `[REJECT:n]` protocol surface；不是自然语言 fallback。"""
    return (
        0x5B, 0x52, 0x45, 0x4A, 0x45, 0x43, 0x54, 0x3A,
        *_ascii_nonnegative_integer(result_code), 0x5D,
    )


def terminal_dialogue_response_schema_record_v1() -> tuple[int, ...]:
    """冻结 response/carrier/act 的闭合整数 tag，供 snapshot 与移植实现共同绑定。"""
    return (
        TERMINAL_DIALOGUE_RESPONSE_SCHEMA_RECORD_V1,
        TERMINAL_DIALOGUE_RESPONSE_RECORD_V1,
        TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1,
        TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1,
        TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1,
        TERMINAL_DIALOGUE_BASE_CARRIER_PROVIDER_V1,
        TERMINAL_DIALOGUE_BASE_CARRIER_FOLLOWUP_V1,
        TERMINAL_DIALOGUE_ACT_NONE_V1,
        PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1,
        PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1,
        PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1,
    )


def _output_readback(output_u8: tuple[int, ...]) -> ConversationRawIntake:
    """将所有 terminal 输出按同一 raw ingress 读回，固定 UTF-8/预算边界。"""
    try:
        readback = intake_raw_conversation_vector(output_u8)
    except (ConversationRawIntakeError, TypeError, ValueError) as error:
        raise ConversationRawTerminalDialogueActError(
            "terminal response output readback 无法形成") from error
    if (not readback.accepted or readback.raw_input_bytes != output_u8
            or readback.canonical_record()
            != intake_raw_conversation_vector(output_u8).canonical_record()):
        raise ConversationRawTerminalDialogueActError(
            "terminal response output readback 漂移")
    return readback


def _base_carrier_from_inner_turn(
        turn: ConversationRawMixedFocusDialogueTurnV1,
        ) -> tuple[int, int, tuple[int, ...], tuple[int, ...], ConversationRawIntake]:
    """在旧 carrier adapter 边界消除 Python union，导出统一 tag/record/output。"""
    if type(turn) is not ConversationRawMixedFocusDialogueTurnV1:
        raise TypeError("terminal dialogue inner turn 类型错误")
    carriers = (
        (TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1, turn.answer),
        (TERMINAL_DIALOGUE_BASE_CARRIER_PROVIDER_V1, turn.provider_answer),
        (TERMINAL_DIALOGUE_BASE_CARRIER_FOLLOWUP_V1,
         turn.provider_followup_answer),
    )
    selected_kind = 0
    selected = None
    for kind, carrier in carriers:
        if carrier is not None:
            if selected is not None:
                raise ConversationRawTerminalDialogueActError(
                    "terminal dialogue inner turn carrier 不唯一")
            selected_kind = kind
            selected = carrier
    if selected is None:
        raise ConversationRawTerminalDialogueActError(
            "terminal dialogue inner turn 缺 carrier")
    if type(selected) is ConversationRawAnswerResult:
        result_code = selected.result_code
        intake = selected.ingress.intake
        output = selected.output_bytes if selected.accepted else _reject_surface(result_code)
    elif type(selected) is PublicProofSentenceProviderResultV1:
        result_code = selected.mapped_dlg_result_code
        intake = selected.intake
        output = selected.output_bytes if selected.accepted else _reject_surface(result_code)
    elif type(selected) is ProviderOriginFollowupResultV1:
        result_code = selected.mapped_dlg_result_code
        intake = selected.intake
        output = selected.output_u8 if selected.accepted else _reject_surface(result_code)
    else:
        raise ConversationRawTerminalDialogueActError(
            "terminal dialogue carrier 类型未注册")
    _u64(result_code, label="terminal dialogue base result code")
    if type(intake) is not ConversationRawIntake:
        raise ConversationRawTerminalDialogueActError(
            "terminal dialogue carrier intake 类型错误")
    if intake.canonical_record() != turn.intake.canonical_record():
        raise ConversationRawTerminalDialogueActError(
            "terminal dialogue carrier/turn intake 漂移")
    record = _record(selected.canonical_record(),
                     label="terminal dialogue base carrier", allow_empty=False)
    return selected_kind, result_code, record, _u8(
        output,
        label="terminal dialogue base output",
        allow_empty=False,
    ), intake


# object-model: value; representation=struct; interop=DLG-RAW-13
@dataclass(frozen=True, slots=True)
class TerminalDialogueResponseV1:
    """终端唯一消费的 closed-tag response，不再让 renderer 推断 carrier 类。"""

    response_kind: int
    base_carrier_kind: int
    base_result_code: int
    base_carrier_record: tuple[int, ...]
    input_intake: ConversationRawIntake
    dialogue_state_effect: int
    act_code: int
    catalog_record: tuple[int, ...]
    catalog_identity_u8: tuple[int, ...]
    form_record: tuple[int, ...]
    form_identity_u8: tuple[int, ...]
    output_u8: tuple[int, ...]
    output_readback: ConversationRawIntake

    def __post_init__(self) -> None:
        """冻结 response union、base carrier、act record 与 output readback。"""
        if self.response_kind not in (
                TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1,
                TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1):
            raise ConversationRawTerminalDialogueActError(
                "terminal response kind 未注册")
        if self.base_carrier_kind not in (
                TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1,
                TERMINAL_DIALOGUE_BASE_CARRIER_PROVIDER_V1,
                TERMINAL_DIALOGUE_BASE_CARRIER_FOLLOWUP_V1):
            raise ConversationRawTerminalDialogueActError(
                "terminal response base carrier kind 未注册")
        _u64(self.base_result_code, label="terminal response base result code")
        _record(self.base_carrier_record,
                label="terminal response base carrier", allow_empty=False)
        if type(self.input_intake) is not ConversationRawIntake:
            raise TypeError("terminal response input intake 类型错误")
        if self.dialogue_state_effect != PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1:
            raise ConversationRawTerminalDialogueActError(
                "terminal response 不得改变 dialogue state")
        output = _u8(self.output_u8, label="terminal response output",
                     allow_empty=False)
        if type(self.output_readback) is not ConversationRawIntake:
            raise TypeError("terminal response output readback 类型错误")
        if (not self.output_readback.accepted
                or self.output_readback.raw_input_bytes != output):
            raise ConversationRawTerminalDialogueActError(
                "terminal response output/readback 漂移")
        act_fields = (
            self.catalog_record,
            self.catalog_identity_u8,
            self.form_record,
            self.form_identity_u8,
        )
        if self.response_kind == TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1:
            if (self.act_code != TERMINAL_DIALOGUE_ACT_NONE_V1
                    or any(item for item in act_fields)):
                raise ConversationRawTerminalDialogueActError(
                    "terminal passthrough 不得携带 act binding")
            return
        expected_mapping = (
            (DLG_RAW_REJECT_LEXICAL_MISS,
             PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1),
            (DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
             PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1),
        )
        matches = tuple(item for item in expected_mapping
                        if item[0] == self.base_result_code)
        if (self.base_carrier_kind != TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1
                or len(matches) != 1 or matches[0][1] != self.act_code
                or any(not item for item in act_fields)):
            raise ConversationRawTerminalDialogueActError(
                "terminal meta-act mapping 或 binding 漂移")
        _record(self.catalog_record, label="terminal response catalog",
                allow_empty=False)
        _u8(self.catalog_identity_u8,
            label="terminal response catalog identity", allow_empty=False)
        _record(self.form_record, label="terminal response form",
                allow_empty=False)
        _u8(self.form_identity_u8,
            label="terminal response form identity", allow_empty=False)

    def canonical_record(self) -> tuple[int, ...]:
        """导出端到端可迁移的 terminal response integer record。"""
        result = [
            TERMINAL_DIALOGUE_RESPONSE_RECORD_V1,
            self.response_kind,
            self.base_carrier_kind,
            self.base_result_code,
            self.dialogue_state_effect,
            self.act_code,
        ]
        for label, value in (
                ("base carrier", self.base_carrier_record),
                ("input intake", self.input_intake.canonical_record()),
                ("catalog", self.catalog_record),
                ("catalog identity", self.catalog_identity_u8),
                ("form", self.form_record),
                ("form identity", self.form_identity_u8),
                ("output", self.output_u8),
                ("output readback", self.output_readback.canonical_record())):
            _pack(result, value, label=f"terminal response {label}")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-13
@dataclass(frozen=True, slots=True)
class TerminalDialogueActRuntimeV1:
    """把冻结 DLG-RAW-12 runtime 与独立 terminal-act course 绑定为配置 struct。"""

    inner_runtime: PublicDialogueRuntimeV1
    catalog: TerminalDialogueActCatalogV1

    def __post_init__(self) -> None:
        """验证 catalog 完全属于 inner runtime 的同一 public payload closure。"""
        if type(self.inner_runtime) is not PublicDialogueRuntimeV1:
            raise TypeError("terminal dialogue act inner runtime 类型错误")
        if type(self.catalog) is not TerminalDialogueActCatalogV1:
            raise TypeError("terminal dialogue act catalog 类型错误")
        if (self.catalog.source_payload_closure_identity_u8
                != tuple(self.inner_runtime.source_payload_closure.closure_identity)):
            raise ConversationRawTerminalDialogueActError(
                "terminal dialogue act catalog closure identity 漂移")
        try:
            validate_public_terminal_dialogue_act_catalog_v1(
                self.catalog,
                self.inner_runtime.source_payload_closure,
            )
        except (PublicTerminalDialogueActCatalogError, TypeError, ValueError) as error:
            raise ConversationRawTerminalDialogueActError(
                "terminal dialogue act runtime catalog 无法重建") from error
        self.binding_record()

    def binding_record(self) -> tuple[int, ...]:
        """冻结 inner runtime、course catalog、state/turn/response schema。"""
        result = [
            TERMINAL_DIALOGUE_ACT_RUNTIME_BINDING_RECORD_V1,
            TERMINAL_DIALOGUE_ACT_RUNTIME_PROTOCOL_V1,
            TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1,
            TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1,
            PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1,
        ]
        for label, value in (
                ("inner dialogue runtime binding", self.inner_runtime.binding_record()),
                ("inner dialogue runtime identity", self.inner_runtime.runtime_identity()),
                ("terminal act catalog", self.catalog.canonical_record()),
                ("terminal act catalog identity", self.catalog.catalog_identity_u8),
                ("terminal dialogue response schema",
                 terminal_dialogue_response_schema_record_v1())):
            _pack(result, value, label=label)
        return tuple(result)

    def runtime_identity_u8(self) -> tuple[int, ...]:
        """返回完整 outer runtime binding 的 portable SHA-256 identity。"""
        return _identity(self.binding_record(), label="terminal dialogue act runtime")


def build_terminal_dialogue_act_runtime_v1(
        inner_runtime: PublicDialogueRuntimeV1,
        ) -> TerminalDialogueActRuntimeV1:
    """从已有 public dialogue runtime 的同一 closure 构建 DLG-RAW-13 runtime。"""
    if type(inner_runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("terminal dialogue act build inner runtime 类型错误")
    catalog = load_public_terminal_dialogue_act_catalog_from_closure(
        inner_runtime.source_payload_closure)
    return TerminalDialogueActRuntimeV1(inner_runtime, catalog)


def _passthrough_response(
        carrier_kind: int,
        base_result_code: int,
        base_record: tuple[int, ...],
        intake: ConversationRawIntake,
        output_u8: tuple[int, ...],
        ) -> TerminalDialogueResponseV1:
    """构造不改变任何既有 carrier 可见字节的统一 terminal response。"""
    output = _u8(output_u8, label="terminal passthrough output", allow_empty=False)
    return TerminalDialogueResponseV1(
        TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1,
        carrier_kind,
        base_result_code,
        base_record,
        intake,
        PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1,
        TERMINAL_DIALOGUE_ACT_NONE_V1,
        (),
        (),
        (),
        (),
        output,
        _output_readback(output),
    )


def _meta_act_response(
        carrier_kind: int,
        base_result_code: int,
        base_record: tuple[int, ...],
        intake: ConversationRawIntake,
        catalog: TerminalDialogueActCatalogV1,
        form: TerminalDialogueActFormV1,
        ) -> TerminalDialogueResponseV1:
    """从已验证课程 form 复制 output，不组织自然语言或读取用户文本。"""
    return TerminalDialogueResponseV1(
        TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1,
        carrier_kind,
        base_result_code,
        base_record,
        intake,
        PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1,
        form.act_code,
        catalog.canonical_record(),
        catalog.catalog_identity_u8,
        form.canonical_record(),
        form.form_identity_u8,
        form.output_u8,
        _output_readback(form.output_u8),
    )


def terminal_dialogue_response_from_inner_turn_v1(
        inner_turn: ConversationRawMixedFocusDialogueTurnV1,
        runtime: TerminalDialogueActRuntimeV1,
        ) -> TerminalDialogueResponseV1:
    """将一次已完成 inner turn 投影为可渲染 response；act 失败保持原 reject。"""
    if type(runtime) is not TerminalDialogueActRuntimeV1:
        raise TypeError("terminal dialogue response runtime 类型错误")
    carrier_kind, code, base_record, output, intake = _base_carrier_from_inner_turn(
        inner_turn)
    if (carrier_kind != TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1
            or code not in (
                DLG_RAW_REJECT_LEXICAL_MISS,
                DLG_RAW_REJECT_LEXICAL_AMBIGUOUS)):
        return _passthrough_response(
            carrier_kind, code, base_record, intake, output)
    try:
        catalog = validate_public_terminal_dialogue_act_catalog_v1(
            runtime.catalog,
            runtime.inner_runtime.source_payload_closure,
        )
        form = catalog.form_for_base_result_code(code)
        if form is None:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act catalog 缺对应 base result code")
        return _meta_act_response(
            carrier_kind,
            code,
            base_record,
            intake,
            catalog,
            form,
        )
    except (ConversationRawTerminalDialogueActError,
            PublicTerminalDialogueActCatalogError, TypeError, ValueError):
        # 课程/来源任何漂移均只回退已完成的 protocol reject，绝不组织未验证文本。
        return _passthrough_response(
            carrier_kind, code, base_record, intake, output)


# object-model: value; representation=struct; interop=DLG-RAW-13
@dataclass(frozen=True, slots=True)
class TerminalDialogueActStateV1:
    """独立 outer state；它不向冻结的 V4/V3 state 注入 terminal act tag。"""

    inner_state: ConversationRawMixedFocusDialogueStateV1

    def __post_init__(self) -> None:
        """限制 state 只封装一个已验证 inner state，无 shadow cursor 或 object identity。"""
        if type(self.inner_state) is not ConversationRawMixedFocusDialogueStateV1:
            raise TypeError("terminal dialogue act inner state 类型错误")

    @property
    def conversation_key(self) -> tuple[int, ...]:
        """暴露 inner 的唯一 conversation owner，不复制可漂移的 key。"""
        return self.inner_state.conversation_key

    @property
    def next_operation_ordinal(self) -> int:
        """外层 ordinal 与 inner 严格相同，reject/control 不得形成第二个时钟。"""
        return self.inner_state.next_operation_ordinal

    def canonical_record(self) -> tuple[int, ...]:
        """导出仅含 inner state 的新版本 outer state record。"""
        result = [TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1]
        _pack(result, self.inner_state.canonical_record(),
              label="terminal dialogue act inner state")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-13
@dataclass(frozen=True, slots=True)
class TerminalDialogueActTurnV1:
    """一次 outer transition：保留完整 inner turn 与统一 terminal response。"""

    before: TerminalDialogueActStateV1
    inner_turn: ConversationRawMixedFocusDialogueTurnV1
    response: TerminalDialogueResponseV1
    after: TerminalDialogueActStateV1

    def __post_init__(self) -> None:
        """核验 outer/inner state 连续、唯一 base carrier 与 response 输入绑定。"""
        if type(self.before) is not TerminalDialogueActStateV1:
            raise TypeError("terminal dialogue act turn before 类型错误")
        if type(self.inner_turn) is not ConversationRawMixedFocusDialogueTurnV1:
            raise TypeError("terminal dialogue act turn inner turn 类型错误")
        if type(self.response) is not TerminalDialogueResponseV1:
            raise TypeError("terminal dialogue act turn response 类型错误")
        if type(self.after) is not TerminalDialogueActStateV1:
            raise TypeError("terminal dialogue act turn after 类型错误")
        if (self.inner_turn.before != self.before.inner_state
                or self.inner_turn.after != self.after.inner_state
                or self.before.conversation_key != self.after.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1):
            raise ConversationRawTerminalDialogueActError(
                "terminal dialogue act outer/inner state transition 漂移")
        kind, code, carrier_record, _output, intake = _base_carrier_from_inner_turn(
            self.inner_turn)
        if (self.response.base_carrier_kind != kind
                or self.response.base_result_code != code
                or self.response.base_carrier_record != carrier_record
                or self.response.input_intake.canonical_record()
                != intake.canonical_record()):
            raise ConversationRawTerminalDialogueActError(
                "terminal dialogue act response 未绑定 inner carrier")

    def canonical_record(self) -> tuple[int, ...]:
        """导出 input、inner transition、response 和前后 state 的完整 trace record。"""
        result = [TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1]
        for label, value in (
                ("before", self.before.canonical_record()),
                ("inner turn", self.inner_turn.canonical_record()),
                ("response", self.response.canonical_record()),
                ("after", self.after.canonical_record())):
            _pack(result, value, label=f"terminal dialogue act turn {label}")
        return tuple(result)


def start_public_terminal_dialogue_act(
        conversation_key: tuple[int, ...],
        ) -> TerminalDialogueActStateV1:
    """建立 DLG-RAW-13 空 outer state；没有 terminal act cursor 或隐式历史。"""
    return TerminalDialogueActStateV1(
        start_public_mixed_focus_dialogue(conversation_key))


def run_public_terminal_dialogue_act_turn_v1(
        state: TerminalDialogueActStateV1,
        raw_input_bytes: tuple[int, ...],
        runtime: TerminalDialogueActRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> TerminalDialogueActTurnV1:
    """执行一次 inner dialogue 后形成唯一 response；meta-act 不重跑 inner runtime。"""
    if type(state) is not TerminalDialogueActStateV1:
        raise TypeError("terminal dialogue act state 类型错误")
    if type(runtime) is not TerminalDialogueActRuntimeV1:
        raise TypeError("terminal dialogue act runtime 类型错误")
    inner_turn = run_public_mixed_focus_dialogue_turn_v1(
        state.inner_state,
        raw_input_bytes,
        runtime.inner_runtime,
        preparation_cache=preparation_cache,
        preflight_cache=preflight_cache,
    )
    response = terminal_dialogue_response_from_inner_turn_v1(inner_turn, runtime)
    after = TerminalDialogueActStateV1(inner_turn.after)
    return TerminalDialogueActTurnV1(state, inner_turn, response, after)


__all__ = [
    "TERMINAL_DIALOGUE_ACT_NONE_V1",
    "TERMINAL_DIALOGUE_ACT_RUNTIME_BINDING_RECORD_V1",
    "TERMINAL_DIALOGUE_ACT_RUNTIME_IDENTITY_DOMAIN_V1",
    "TERMINAL_DIALOGUE_ACT_RUNTIME_PROTOCOL_V1",
    "TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1",
    "TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1",
    "TERMINAL_DIALOGUE_BASE_CARRIER_FOLLOWUP_V1",
    "TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1",
    "TERMINAL_DIALOGUE_BASE_CARRIER_PROVIDER_V1",
    "TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1",
    "TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1",
    "TERMINAL_DIALOGUE_RESPONSE_RECORD_V1",
    "TERMINAL_DIALOGUE_RESPONSE_SCHEMA_RECORD_V1",
    "ConversationRawTerminalDialogueActError",
    "TerminalDialogueActRuntimeV1",
    "TerminalDialogueActStateV1",
    "TerminalDialogueActTurnV1",
    "TerminalDialogueResponseV1",
    "build_terminal_dialogue_act_runtime_v1",
    "run_public_terminal_dialogue_act_turn_v1",
    "start_public_terminal_dialogue_act",
    "terminal_dialogue_response_schema_record_v1",
    "terminal_dialogue_response_from_inner_turn_v1",
]
