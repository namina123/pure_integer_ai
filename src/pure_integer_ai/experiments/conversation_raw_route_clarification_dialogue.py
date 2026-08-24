"""DLG-RAW-14：来源绑定 route ambiguity 的一轮显式重输 outer dialogue。

本层只包裹 DLG-RAW-13。它不会修改 V3/V4、terminal-act response 或回答 runtime：
首轮仅重演已经发生的 V3 ambiguity 以形成来源绑定候选提示；下一轮仍完整进入
DLG-RAW-13，且只有实际 Frame answer 与 pending candidate 严格一致时才追加选择
event。所有持久数据均可降解为有序整数 record 与 raw-u8 vector。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrame,
)
from pure_integer_ai.experiments.conversation_public_route_clarification_catalog import (
    PUBLIC_ROUTE_CLARIFICATION_CATALOG_RECORD_V1,
    PublicRouteClarificationCatalogError,
    PublicRouteClarificationCatalogValidationCacheV1,
    PublicRouteClarificationCatalogV1,
    RouteClarificationFormV1,
    RouteClarificationOptionV1,
    RouteClarificationOutputReadbackV1,
    candidate_identity_v1,
    load_public_route_clarification_catalog_from_closure,
    route_clarification_output_readback_v1,
    route_identity_from_source_bound_resolution_v1,
    validate_public_route_clarification_catalog_cached_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
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
    ConversationRawIntake,
    ConversationRawIntakeError,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act import (
    TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1,
    TerminalDialogueActRuntimeV1,
    TerminalDialogueActStateV1,
    TerminalDialogueActTurnV1,
    TerminalDialogueResponseV1,
    build_terminal_dialogue_act_runtime_v1,
    run_public_terminal_dialogue_act_turn_v1,
    start_public_terminal_dialogue_act,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3,
    SourceBoundSlotCompositionError,
    SourceBoundSlotCompositionResolution,
    SourceBoundSlotTargetCandidateV3,
    resolve_source_bound_slot_composition,
)


ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_PROTOCOL_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_BINDING_RECORD_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_STATE_RECORD_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_TURN_RECORD_V1 = 1
ROUTE_CLARIFICATION_RESPONSE_RECORD_V1 = 1
ROUTE_CLARIFICATION_PENDING_RECORD_V1 = 1
ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_RECORD_V1 = 1
ROUTE_CLARIFICATION_SELECTION_EVENT_RECORD_V1 = 1
ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1 = 1
ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1 = 2
ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1 = 3
ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1 = 0
ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1 = 1
ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1 = 2
ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1 = 3
ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1 = 64

ROUTE_CLARIFICATION_RUNTIME_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-CLARIFICATION-RUNTIME/V1")
ROUTE_CLARIFICATION_STATE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/TERMINAL-ACT-STATE/V1")
ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/CANDIDATE-PROJECTION/V1")
ROUTE_CLARIFICATION_PENDING_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/PENDING/V1")
ROUTE_CLARIFICATION_SELECTION_EVENT_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/SELECTION-EVENT/V1")

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-14
class ConversationRawRouteClarificationDialogueError(ValueError):
    """DLG-RAW-14 outer state、candidate projection 或 runtime binding 不闭合。"""


def _u64(value: int, *, label: str, minimum: int = 0) -> int:
    """验证协议标量使用明确、可移植的无符号 64-bit 范围。"""
    if (type(value) is not int or value < minimum
            or value >= _U64_EXCLUSIVE):
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} 必须是范围内的严格 u64")
    return value


def _record(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证有限非负整数 record，拒绝 Python list/bool/负值进入状态。"""
    if (type(value) is not tuple or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} 不是{'可空' if allow_empty else '非空'}非负整数 record")
    return value


def _u8(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证 immutable raw-u8 vector，不让宿主 bytes/str 定义核心状态。"""
    if (type(value) is not tuple or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} 不是{'可空' if allow_empty else '非空'} raw-u8 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    """以明确 count framing 写入子 record，禁止容器长度成为隐式 wire。"""
    record = _record(value, label=label, allow_empty=True)
    _u64(len(record), label=f"{label} count")
    result.extend((len(record), *record))


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """使用现有 portable SHA framing 形成 raw-u8[32] identity。"""
    try:
        identity = tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} identity 无法形成") from error
    identity = _u8(identity, label=f"{label} identity", allow_empty=False)
    if len(identity) != 32:
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} identity 长度漂移")
    return identity


def _ascii_u8(value: str, *, label: str) -> tuple[int, ...]:
    """把已有 ASCII frame key 显式降为 u8，不让 locale/默认编码参与。"""
    if type(value) is not str or not value:
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} 必须是非空 ASCII str")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} 不是 ASCII") from error
    if any(item < 0x21 or item > 0x7E for item in encoded):
        raise ConversationRawRouteClarificationDialogueError(
            f"{label} 含非规范 ASCII byte")
    return tuple(encoded)


def _state_identity(state: TerminalDialogueActStateV1) -> tuple[int, ...]:
    """为内层 terminal-act state 建立可重算的跨层 linkage identity。"""
    if type(state) is not TerminalDialogueActStateV1:
        raise TypeError("route clarification inner state 类型错误")
    return _identity(
        ROUTE_CLARIFICATION_STATE_IDENTITY_DOMAIN_V1,
        state.canonical_record(),
        label="route clarification inner state",
    )


def _output_readback(
        output_u8: tuple[int, ...],
        ) -> RouteClarificationOutputReadbackV1:
    """以独立 UTF-8 output record 回读可见字节，不将多行输出当作用户输入。"""
    output = _u8(output_u8, label="route clarification output", allow_empty=False)
    try:
        readback = route_clarification_output_readback_v1(output)
    except (PublicRouteClarificationCatalogError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueError(
            "route clarification output readback 无法形成") from error
    if (readback.output_u8 != output
            or readback.canonical_record()
            != route_clarification_output_readback_v1(
                output).canonical_record()):
        raise ConversationRawRouteClarificationDialogueError(
            "route clarification output/readback 漂移")
    return readback


def _terminal_turn_intake(
        inner_turn: TerminalDialogueActTurnV1,
        ) -> ConversationRawIntake:
    """读取 DLG-RAW-13 已绑定 intake，并核验 response 与其 inner turn 未漂移。"""
    if type(inner_turn) is not TerminalDialogueActTurnV1:
        raise TypeError("route clarification inner turn 类型错误")
    intake = inner_turn.response.input_intake
    if (type(intake) is not ConversationRawIntake
            or intake.canonical_record()
            != inner_turn.inner_turn.intake.canonical_record()):
        raise ConversationRawRouteClarificationDialogueError(
            "route clarification terminal turn intake binding 漂移")
    return intake


def _intake_matches_surface(
        intake: ConversationRawIntake,
        surface_u8: tuple[int, ...],
        ) -> bool:
    """只按 RAW-00 canonical scalar 比较完整重输问句，允许 body/LF/CRLF。"""
    if type(intake) is not ConversationRawIntake:
        raise TypeError("route clarification selection intake 类型错误")
    surface = _u8(surface_u8, label="route option surface", allow_empty=False)
    try:
        expected = intake_raw_conversation_vector(surface)
    except (ConversationRawIntakeError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueError(
            "route option surface readback 无法形成") from error
    return (intake.accepted and expected.accepted
            and intake.unicode_scalars == expected.unicode_scalars)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteAmbiguityCandidateProjectionV1:
    """把一个真实 V3 candidate 与一条公开重输 option 显式绑定。"""

    candidate_record: tuple[int, ...]
    candidate_identity_u8: tuple[int, ...]
    base_frame_key_u8: tuple[int, ...]
    base_frame_raw_sha256_u8: tuple[int, ...]
    target_key: tuple[int, ...]
    recipe_record: tuple[int, ...]
    option_record: tuple[int, ...]
    option_identity_u8: tuple[int, ...]
    option_surface_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        """冻结 candidate/frame/recipe 与 course option 的逐字段关系。"""
        _record(self.candidate_record, label="route candidate record",
                allow_empty=False)
        candidate_identity = _u8(
            self.candidate_identity_u8,
            label="route candidate identity",
            allow_empty=False,
        )
        if len(candidate_identity) != 32:
            raise ConversationRawRouteClarificationDialogueError(
                "route candidate identity 必须是 raw-u8[32]")
        frame_key = _u8(self.base_frame_key_u8,
                        label="route candidate base frame key",
                        allow_empty=False)
        if any(item < 0x21 or item > 0x7E for item in frame_key):
            raise ConversationRawRouteClarificationDialogueError(
                "route candidate base frame key 不是规范 ASCII")
        digest = _u8(self.base_frame_raw_sha256_u8,
                     label="route candidate base frame SHA-256",
                     allow_empty=False)
        if len(digest) != 32:
            raise ConversationRawRouteClarificationDialogueError(
                "route candidate base frame SHA-256 长度漂移")
        _record(self.target_key, label="route candidate target key",
                allow_empty=False)
        _record(self.recipe_record, label="route candidate recipe",
                allow_empty=False)
        _record(self.option_record, label="route candidate option record",
                allow_empty=False)
        option_identity = _u8(self.option_identity_u8,
                              label="route candidate option identity",
                              allow_empty=False)
        if len(option_identity) != 32:
            raise ConversationRawRouteClarificationDialogueError(
                "route candidate option identity 必须是 raw-u8[32]")
        surface = _u8(self.option_surface_u8,
                      label="route candidate option surface",
                      allow_empty=False)
        if any(item in (0x0A, 0x0D) for item in surface):
            raise ConversationRawRouteClarificationDialogueError(
                "route candidate option surface 不得含行 framing")
        if not _intake_matches_surface(
                intake_raw_conversation_vector(surface), surface):
            raise ConversationRawRouteClarificationDialogueError(
                "route candidate option surface 无法按 RAW-00 读回")

    def base_record(self) -> tuple[int, ...]:
        """导出不含 projection identity 的完整 candidate/option binding。"""
        result = [ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_RECORD_V1]
        for label, value in (
                ("candidate", self.candidate_record),
                ("candidate identity", self.candidate_identity_u8),
                ("base frame key", self.base_frame_key_u8),
                ("base frame SHA-256", self.base_frame_raw_sha256_u8),
                ("target key", self.target_key),
                ("recipe", self.recipe_record),
                ("option", self.option_record),
                ("option identity", self.option_identity_u8),
                ("option surface", self.option_surface_u8)):
            _pack(result, value, label=f"route candidate projection {label}")
        return tuple(result)

    @property
    def projection_identity_u8(self) -> tuple[int, ...]:
        """返回可重算 candidate projection identity，供 pending/event linkage 使用。"""
        return _identity(
            ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="route candidate projection",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整、无 Python object identity 的 candidate projection record。"""
        result = list(self.base_record())
        _pack(result, self.projection_identity_u8,
              label="route candidate projection identity")
        return tuple(result)


def _candidate_projection(
        candidate: SourceBoundSlotTargetCandidateV3,
        option: RouteClarificationOptionV1,
        ) -> RouteAmbiguityCandidateProjectionV1:
    """从同次 V3 resolver candidate 与已验证 course option 建立 projection。"""
    if type(candidate) is not SourceBoundSlotTargetCandidateV3:
        raise TypeError("route clarification candidate 类型错误")
    if type(option) is not RouteClarificationOptionV1:
        raise TypeError("route clarification option 类型错误")
    candidate_identity = candidate_identity_v1(candidate)
    if option.candidate_identity_u8 != candidate_identity:
        raise ConversationRawRouteClarificationDialogueError(
            "route clarification option/candidate identity 漂移")
    return RouteAmbiguityCandidateProjectionV1(
        candidate.canonical_record(),
        candidate_identity,
        _ascii_u8(candidate.base_frame_key,
                  label="route candidate base frame key"),
        candidate.base_frame_raw_sha256,
        candidate.target_key,
        candidate.recipe_record,
        option.canonical_record(),
        option.option_identity_u8,
        option.option_surface_u8,
    )


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteAmbiguityPendingV1:
    """一条仅对下一 operation 有效的来源绑定 route clarification 控制状态。"""

    created_operation_ordinal: int
    selection_operation_ordinal: int
    input_intake: ConversationRawIntake
    inner_turn_record: tuple[int, ...]
    inner_response_record: tuple[int, ...]
    inner_before_state_identity_u8: tuple[int, ...]
    inner_after_state_identity_u8: tuple[int, ...]
    resolution_record: tuple[int, ...]
    route_identity_u8: tuple[int, ...]
    selector_catalog_record: tuple[int, ...]
    selector_catalog_identity_u8: tuple[int, ...]
    selector_form_record: tuple[int, ...]
    selector_form_identity_u8: tuple[int, ...]
    candidates: tuple[RouteAmbiguityCandidateProjectionV1, ...]
    output_u8: tuple[int, ...]
    output_readback: RouteClarificationOutputReadbackV1

    def __post_init__(self) -> None:
        """使 pending 只在一轮内生效，且完整绑定 inner/resolver/course/output。"""
        created = _u64(self.created_operation_ordinal,
                       label="route pending created operation")
        selection = _u64(self.selection_operation_ordinal,
                         label="route pending selection operation")
        if selection != created + 1:
            raise ConversationRawRouteClarificationDialogueError(
                "route pending 必须只对下一 operation 有效")
        if type(self.input_intake) is not ConversationRawIntake:
            raise TypeError("route pending input intake 类型错误")
        if not self.input_intake.accepted:
            raise ConversationRawRouteClarificationDialogueError(
                "route pending input 必须已通过 RAW-00")
        for label, value in (
                ("inner turn", self.inner_turn_record),
                ("inner response", self.inner_response_record),
                ("resolution", self.resolution_record),
                ("selector catalog", self.selector_catalog_record),
                ("selector form", self.selector_form_record)):
            _record(value, label=f"route pending {label}", allow_empty=False)
        for label, value in (
                ("inner before state", self.inner_before_state_identity_u8),
                ("inner after state", self.inner_after_state_identity_u8),
                ("route", self.route_identity_u8),
                ("selector catalog", self.selector_catalog_identity_u8),
                ("selector form", self.selector_form_identity_u8)):
            identity = _u8(value, label=f"route pending {label} identity",
                           allow_empty=False)
            if len(identity) != 32:
                raise ConversationRawRouteClarificationDialogueError(
                    f"route pending {label} identity 必须是 raw-u8[32]")
        if (type(self.candidates) is not tuple or len(self.candidates) < 2
                or any(type(item) is not RouteAmbiguityCandidateProjectionV1
                       for item in self.candidates)):
            raise ConversationRawRouteClarificationDialogueError(
                "route pending candidate projection 数量或类型漂移")
        candidate_identities: list[tuple[int, ...]] = []
        for candidate in self.candidates:
            identity = candidate.candidate_identity_u8
            if any(identity == prior for prior in candidate_identities):
                raise ConversationRawRouteClarificationDialogueError(
                    "route pending candidate identity 重复")
            candidate_identities.append(identity)
        output = _u8(self.output_u8, label="route pending output",
                     allow_empty=False)
        if type(self.output_readback) is not RouteClarificationOutputReadbackV1:
            raise TypeError("route pending output readback 类型错误")
        if self.output_readback.output_u8 != output:
            raise ConversationRawRouteClarificationDialogueError(
                "route pending output/readback 漂移")

    def base_record(self) -> tuple[int, ...]:
        """导出不含 pending identity 的完整一轮 candidate offer 控制状态。"""
        result = [
            ROUTE_CLARIFICATION_PENDING_RECORD_V1,
            self.created_operation_ordinal,
            self.selection_operation_ordinal,
        ]
        for label, value in (
                ("input intake", self.input_intake.canonical_record()),
                ("inner turn", self.inner_turn_record),
                ("inner response", self.inner_response_record),
                ("inner before state identity",
                 self.inner_before_state_identity_u8),
                ("inner after state identity",
                 self.inner_after_state_identity_u8),
                ("resolution", self.resolution_record),
                ("route identity", self.route_identity_u8),
                ("selector catalog", self.selector_catalog_record),
                ("selector catalog identity",
                 self.selector_catalog_identity_u8),
                ("selector form", self.selector_form_record),
                ("selector form identity", self.selector_form_identity_u8)):
            _pack(result, value, label=f"route pending {label}")
        result.append(len(self.candidates))
        for ordinal, candidate in enumerate(self.candidates, start=1):
            _pack(result, candidate.canonical_record(),
                  label=f"route pending candidate {ordinal}")
        for label, value in (
                ("output", self.output_u8),
                ("output readback", self.output_readback.canonical_record())):
            _pack(result, value, label=f"route pending {label}")
        return tuple(result)

    @property
    def pending_identity_u8(self) -> tuple[int, ...]:
        """返回完整 pending record 的 portable identity。"""
        return _identity(
            ROUTE_CLARIFICATION_PENDING_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="route pending",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 pending record，供 snapshot 和 selection event 严格绑定。"""
        result = list(self.base_record())
        _pack(result, self.pending_identity_u8, label="route pending identity")
        return tuple(result)

    def candidate_for_intake(
            self,
            intake: ConversationRawIntake,
            ) -> RouteAmbiguityCandidateProjectionV1 | None:
        """仅按完整 option 的 RAW-00 canonical scalar 选择一个候选，不解析裸名词。"""
        if type(intake) is not ConversationRawIntake:
            raise TypeError("route pending selection intake 类型错误")
        selected = None
        for candidate in self.candidates:
            if _intake_matches_surface(intake, candidate.option_surface_u8):
                if selected is not None:
                    raise ConversationRawRouteClarificationDialogueError(
                        "route pending option surface 不唯一")
                selected = candidate
        return selected

    def candidate_for_identity(
            self,
            candidate_identity_u8: tuple[int, ...],
            ) -> RouteAmbiguityCandidateProjectionV1 | None:
        """按冻结 raw-u8 identity 定位候选，不使用 display text 或 host map。"""
        identity = _u8(candidate_identity_u8,
                       label="route pending candidate lookup identity",
                       allow_empty=False)
        if len(identity) != 32:
            raise ConversationRawRouteClarificationDialogueError(
                "route pending candidate lookup identity 长度漂移")
        selected = None
        for candidate in self.candidates:
            if candidate.candidate_identity_u8 == identity:
                if selected is not None:
                    raise ConversationRawRouteClarificationDialogueError(
                        "route pending candidate identity 不唯一")
                selected = candidate
        return selected


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationSelectionEventV1:
    """一条已被真实 inner Frame answer 证实的 append-only explicit selection event。"""

    event_ordinal: int
    predecessor_identity_u8: tuple[int, ...]
    pending_record: tuple[int, ...]
    pending_identity_u8: tuple[int, ...]
    selected_candidate_record: tuple[int, ...]
    selected_candidate_identity_u8: tuple[int, ...]
    selection_intake: ConversationRawIntake
    inner_turn_record: tuple[int, ...]
    inner_response_record: tuple[int, ...]
    inner_before_state_identity_u8: tuple[int, ...]
    inner_after_state_identity_u8: tuple[int, ...]
    output_u8: tuple[int, ...]
    output_readback: RouteClarificationOutputReadbackV1

    def __post_init__(self) -> None:
        """冻结 event 序、pending/candidate link、actual answer trace 与 byte readback。"""
        _u64(self.event_ordinal, label="route selection event ordinal", minimum=1)
        predecessor = _u8(self.predecessor_identity_u8,
                          label="route selection predecessor identity",
                          allow_empty=True)
        if predecessor and len(predecessor) != 32:
            raise ConversationRawRouteClarificationDialogueError(
                "route selection predecessor identity 长度漂移")
        for label, value in (
                ("pending", self.pending_record),
                ("selected candidate", self.selected_candidate_record),
                ("inner turn", self.inner_turn_record),
                ("inner response", self.inner_response_record)):
            _record(value, label=f"route selection {label}", allow_empty=False)
        for label, value in (
                ("pending", self.pending_identity_u8),
                ("selected candidate", self.selected_candidate_identity_u8),
                ("inner before state", self.inner_before_state_identity_u8),
                ("inner after state", self.inner_after_state_identity_u8)):
            identity = _u8(value, label=f"route selection {label} identity",
                           allow_empty=False)
            if len(identity) != 32:
                raise ConversationRawRouteClarificationDialogueError(
                    f"route selection {label} identity 长度漂移")
        if type(self.selection_intake) is not ConversationRawIntake:
            raise TypeError("route selection intake 类型错误")
        if not self.selection_intake.accepted:
            raise ConversationRawRouteClarificationDialogueError(
                "route selection intake 必须通过 RAW-00")
        output = _u8(self.output_u8, label="route selection output",
                     allow_empty=False)
        if type(self.output_readback) is not RouteClarificationOutputReadbackV1:
            raise TypeError("route selection output readback 类型错误")
        if self.output_readback.output_u8 != output:
            raise ConversationRawRouteClarificationDialogueError(
                "route selection output/readback 漂移")

    def base_record(self) -> tuple[int, ...]:
        """导出不含 self identity 的 append-only selection evidence。"""
        result = [ROUTE_CLARIFICATION_SELECTION_EVENT_RECORD_V1,
                  self.event_ordinal]
        for label, value in (
                ("predecessor identity", self.predecessor_identity_u8),
                ("pending", self.pending_record),
                ("pending identity", self.pending_identity_u8),
                ("selected candidate", self.selected_candidate_record),
                ("selected candidate identity",
                 self.selected_candidate_identity_u8),
                ("selection intake", self.selection_intake.canonical_record()),
                ("inner turn", self.inner_turn_record),
                ("inner response", self.inner_response_record),
                ("inner before state identity",
                 self.inner_before_state_identity_u8),
                ("inner after state identity",
                 self.inner_after_state_identity_u8),
                ("output", self.output_u8),
                ("output readback", self.output_readback.canonical_record())):
            _pack(result, value, label=f"route selection {label}")
        return tuple(result)

    @property
    def event_identity_u8(self) -> tuple[int, ...]:
        """返回本 event 的 portable identity，形成独立 append-only predecessor chain。"""
        return _identity(
            ROUTE_CLARIFICATION_SELECTION_EVENT_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="route selection event",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出 event 本体及可重算 identity。"""
        result = list(self.base_record())
        _pack(result, self.event_identity_u8,
              label="route selection event identity")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationDialogueStateV1:
    """DLG-RAW-14 state：内层 terminal-act、至多一条 pending 与选择 event 账本。"""

    inner_state: TerminalDialogueActStateV1
    pending: RouteAmbiguityPendingV1 | None = None
    selection_events: tuple[RouteClarificationSelectionEventV1, ...] = ()

    def __post_init__(self) -> None:
        """验证 inner 时钟、pending 生命周期和 selection ledger predecessor chain。"""
        if type(self.inner_state) is not TerminalDialogueActStateV1:
            raise TypeError("route clarification inner state 类型错误")
        if self.pending is not None and type(self.pending) is not RouteAmbiguityPendingV1:
            raise TypeError("route clarification pending 类型错误")
        if (type(self.selection_events) is not tuple
                or len(self.selection_events) > ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1
                or any(type(item) is not RouteClarificationSelectionEventV1
                       for item in self.selection_events)):
            raise ConversationRawRouteClarificationDialogueError(
                "route clarification selection event ledger 类型或预算漂移")
        predecessor: tuple[int, ...] = ()
        known_pending: list[tuple[int, ...]] = []
        for ordinal, event in enumerate(self.selection_events, start=1):
            if (event.event_ordinal != ordinal
                    or event.predecessor_identity_u8 != predecessor
                    or event.pending_identity_u8 in known_pending
                    or event.inner_after_state_identity_u8 == ()):
                raise ConversationRawRouteClarificationDialogueError(
                    "route clarification selection ledger 序或 predecessor 漂移")
            known_pending.append(event.pending_identity_u8)
            predecessor = event.event_identity_u8
        if self.pending is not None:
            if (self.pending.selection_operation_ordinal
                    != self.inner_state.next_operation_ordinal
                    or self.pending.pending_identity_u8 in known_pending
                    or self.pending.inner_after_state_identity_u8
                    != _state_identity(self.inner_state)):
                raise ConversationRawRouteClarificationDialogueError(
                    "route clarification active pending 生命周期或 inner state 漂移")

    @property
    def conversation_key(self) -> tuple[int, ...]:
        """暴露唯一内层 conversation key，不创建第二个 session owner。"""
        return self.inner_state.conversation_key

    @property
    def next_operation_ordinal(self) -> int:
        """外层 operation ordinal 永远直接来自 DLG-RAW-13 inner state。"""
        return self.inner_state.next_operation_ordinal

    def canonical_record(self) -> tuple[int, ...]:
        """导出可移植 outer state，不将 temporary pending 误称为长期记忆。"""
        result = [ROUTE_CLARIFICATION_DIALOGUE_STATE_RECORD_V1]
        _pack(result, self.inner_state.canonical_record(),
              label="route clarification inner state")
        if self.pending is None:
            result.append(0)
        else:
            result.append(1)
            _pack(result, self.pending.canonical_record(),
                  label="route clarification pending")
        result.append(len(self.selection_events))
        for ordinal, event in enumerate(self.selection_events, start=1):
            _pack(result, event.canonical_record(),
                  label=f"route clarification selection event {ordinal}")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationDialogueResponseV1:
    """默认 terminal 消费的 closed-tag DLG-RAW-14 outer response。"""

    response_kind: int
    state_effect: int
    base_result_code: int
    inner_response_record: tuple[int, ...]
    input_intake: ConversationRawIntake
    pending_identity_u8: tuple[int, ...]
    route_identity_u8: tuple[int, ...]
    selected_candidate_identity_u8: tuple[int, ...]
    selector_catalog_record: tuple[int, ...]
    selector_catalog_identity_u8: tuple[int, ...]
    selector_form_record: tuple[int, ...]
    selector_form_identity_u8: tuple[int, ...]
    output_u8: tuple[int, ...]
    output_readback: RouteClarificationOutputReadbackV1

    def __post_init__(self) -> None:
        """关闭 response/effect union，禁止 renderer 从 Python carrier 类型推断语义。"""
        if self.response_kind not in (
                ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1,
                ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1,
                ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1):
            raise ConversationRawRouteClarificationDialogueError(
                "route clarification response kind 未注册")
        if self.state_effect not in (
                ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1,
                ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1,
                ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1,
                ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1):
            raise ConversationRawRouteClarificationDialogueError(
                "route clarification response state effect 未注册")
        _u64(self.base_result_code, label="route response base result code")
        _record(self.inner_response_record,
                label="route response inner response", allow_empty=False)
        if type(self.input_intake) is not ConversationRawIntake:
            raise TypeError("route response input intake 类型错误")
        bindings = (
            self.pending_identity_u8,
            self.route_identity_u8,
            self.selected_candidate_identity_u8,
            self.selector_catalog_record,
            self.selector_catalog_identity_u8,
            self.selector_form_record,
            self.selector_form_identity_u8,
        )
        output = _u8(self.output_u8, label="route response output",
                     allow_empty=False)
        if type(self.output_readback) is not RouteClarificationOutputReadbackV1:
            raise TypeError("route response output readback 类型错误")
        if self.output_readback.output_u8 != output:
            raise ConversationRawRouteClarificationDialogueError(
                "route response output/readback 漂移")
        if self.response_kind == ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1:
            if (self.state_effect not in (
                    ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1,
                    ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1)
                    or any(bindings)):
                raise ConversationRawRouteClarificationDialogueError(
                    "route passthrough 不得携带 candidate/course binding")
            return
        for label, value in (
                ("pending", self.pending_identity_u8),
                ("route", self.route_identity_u8),
                ("selector catalog", self.selector_catalog_identity_u8),
                ("selector form", self.selector_form_identity_u8)):
            identity = _u8(value, label=f"route response {label} identity",
                           allow_empty=False)
            if len(identity) != 32:
                raise ConversationRawRouteClarificationDialogueError(
                    f"route response {label} identity 长度漂移")
        for label, value in (
                ("selector catalog", self.selector_catalog_record),
                ("selector form", self.selector_form_record)):
            _record(value, label=f"route response {label}", allow_empty=False)
        selected = _u8(self.selected_candidate_identity_u8,
                       label="route response selected candidate identity",
                       allow_empty=True)
        if selected and len(selected) != 32:
            raise ConversationRawRouteClarificationDialogueError(
                "route response selected candidate identity 长度漂移")
        if self.response_kind == ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1:
            if (self.state_effect != ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1
                    or self.base_result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
                    or selected):
                raise ConversationRawRouteClarificationDialogueError(
                    "route options response mapping 漂移")
            return
        if (self.state_effect != ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1
                or self.base_result_code != DLG_RAW_ACCEPT
                or len(selected) != 32):
            raise ConversationRawRouteClarificationDialogueError(
                "route selection response mapping 漂移")

    def canonical_record(self) -> tuple[int, ...]:
        """导出统一 outer terminal response 的完整整数/bytes record。"""
        result = [
            ROUTE_CLARIFICATION_RESPONSE_RECORD_V1,
            self.response_kind,
            self.state_effect,
            self.base_result_code,
        ]
        for label, value in (
                ("inner response", self.inner_response_record),
                ("input intake", self.input_intake.canonical_record()),
                ("pending identity", self.pending_identity_u8),
                ("route identity", self.route_identity_u8),
                ("selected candidate identity",
                 self.selected_candidate_identity_u8),
                ("selector catalog", self.selector_catalog_record),
                ("selector catalog identity",
                 self.selector_catalog_identity_u8),
                ("selector form", self.selector_form_record),
                ("selector form identity", self.selector_form_identity_u8),
                ("output", self.output_u8),
                ("output readback", self.output_readback.canonical_record())):
            _pack(result, value, label=f"route response {label}")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationDialogueRuntimeV1:
    """把 DLG-RAW-13 runtime 与独立 selector course 绑定为无路径配置 struct。"""

    terminal_runtime: TerminalDialogueActRuntimeV1
    selector_catalog: PublicRouteClarificationCatalogV1
    # 派生 selector 编译缓存；compare/canonical/binding 均明确排除。
    validation_cache: PublicRouteClarificationCatalogValidationCacheV1 = field(
        default_factory=PublicRouteClarificationCatalogValidationCacheV1,
        compare=False,
        repr=False,
        init=False,
    )

    def __post_init__(self) -> None:
        """确保 course、terminal act 与 inner dialogue 使用同一 public closure。"""
        if type(self.terminal_runtime) is not TerminalDialogueActRuntimeV1:
            raise TypeError("route clarification terminal runtime 类型错误")
        if type(self.selector_catalog) is not PublicRouteClarificationCatalogV1:
            raise TypeError("route clarification selector catalog 类型错误")
        closure = self.terminal_runtime.inner_runtime.source_payload_closure
        if (self.selector_catalog.source_payload_closure_identity_u8
                != tuple(closure.closure_identity)):
            raise ConversationRawRouteClarificationDialogueError(
                "route clarification selector catalog closure identity 漂移")
        try:
            validate_public_route_clarification_catalog_cached_v1(
                self.selector_catalog,
                closure,
                self.validation_cache,
            )
        except (PublicRouteClarificationCatalogError, TypeError, ValueError) as error:
            raise ConversationRawRouteClarificationDialogueError(
                "route clarification selector catalog 无法重建") from error
        self.binding_record()

    def binding_record(self) -> tuple[int, ...]:
        """冻结内层 runtime、selector catalog、outer state/response/event schema 与预算。"""
        result = [
            ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_BINDING_RECORD_V1,
            ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_PROTOCOL_V1,
            ROUTE_CLARIFICATION_DIALOGUE_STATE_RECORD_V1,
            ROUTE_CLARIFICATION_DIALOGUE_TURN_RECORD_V1,
            ROUTE_CLARIFICATION_RESPONSE_RECORD_V1,
            ROUTE_CLARIFICATION_PENDING_RECORD_V1,
            ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_RECORD_V1,
            ROUTE_CLARIFICATION_SELECTION_EVENT_RECORD_V1,
            ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1,
            ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1,
            ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1,
            ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1,
            ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1,
            ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1,
            ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1,
            ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1,
        ]
        for label, value in (
                ("terminal dialogue act runtime",
                 self.terminal_runtime.binding_record()),
                ("terminal dialogue act runtime identity",
                 self.terminal_runtime.runtime_identity_u8()),
                ("selector catalog", self.selector_catalog.canonical_record()),
                ("selector catalog identity",
                 self.selector_catalog.catalog_identity_u8),
                ("selector catalog record tag",
                 (PUBLIC_ROUTE_CLARIFICATION_CATALOG_RECORD_V1,))):
            _pack(result, value, label=f"route clarification {label}")
        return tuple(result)

    def runtime_identity_u8(self) -> tuple[int, ...]:
        """返回完整 runtime binding 的 portable identity。"""
        return _identity(
            ROUTE_CLARIFICATION_RUNTIME_IDENTITY_DOMAIN_V1,
            self.binding_record(),
            label="route clarification runtime",
        )


def build_route_clarification_dialogue_runtime_v1(
        terminal_runtime: TerminalDialogueActRuntimeV1,
        ) -> RouteClarificationDialogueRuntimeV1:
    """从已验证 DLG-RAW-13 runtime 和同一 closure 建立 DLG-RAW-14 runtime。"""
    if type(terminal_runtime) is not TerminalDialogueActRuntimeV1:
        raise TypeError("route clarification build terminal runtime 类型错误")
    catalog = load_public_route_clarification_catalog_from_closure(
        terminal_runtime.inner_runtime.source_payload_closure)
    return RouteClarificationDialogueRuntimeV1(terminal_runtime, catalog)


def build_public_route_clarification_dialogue_runtime_v1(
        inner_runtime: PublicDialogueRuntimeV1,
        ) -> RouteClarificationDialogueRuntimeV1:
    """便捷地从 public dialogue runtime 建立完整 DLG-RAW-13/14 outer runtime。"""
    if type(inner_runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("route clarification build inner runtime 类型错误")
    return build_route_clarification_dialogue_runtime_v1(
        build_terminal_dialogue_act_runtime_v1(inner_runtime))


def _passthrough_response(
        inner_turn: TerminalDialogueActTurnV1,
        *,
        state_effect: int,
        ) -> RouteClarificationDialogueResponseV1:
    """复制 DLG-RAW-13 已验证 output；不组织未绑定的替代语言。"""
    if type(inner_turn) is not TerminalDialogueActTurnV1:
        raise TypeError("route clarification inner turn 类型错误")
    if state_effect not in (
            ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1,
            ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1):
        raise ConversationRawRouteClarificationDialogueError(
            "route passthrough state effect 未注册")
    base = inner_turn.response
    return RouteClarificationDialogueResponseV1(
        ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1,
        state_effect,
        base.base_result_code,
        base.canonical_record(),
        _terminal_turn_intake(inner_turn),
        (), (), (), (), (), (), (),
        base.output_u8,
        _output_readback(base.output_u8),
    )


def _options_response(
        inner_turn: TerminalDialogueActTurnV1,
        pending: RouteAmbiguityPendingV1,
        ) -> RouteClarificationDialogueResponseV1:
    """从来源绑定 pending/form 复制 route options output，绝不拼接自由文本。"""
    if type(inner_turn) is not TerminalDialogueActTurnV1:
        raise TypeError("route clarification inner turn 类型错误")
    if type(pending) is not RouteAmbiguityPendingV1:
        raise TypeError("route clarification pending 类型错误")
    return RouteClarificationDialogueResponseV1(
        ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1,
        ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1,
        inner_turn.response.base_result_code,
        inner_turn.response.canonical_record(),
        _terminal_turn_intake(inner_turn),
        pending.pending_identity_u8,
        pending.route_identity_u8,
        (),
        pending.selector_catalog_record,
        pending.selector_catalog_identity_u8,
        pending.selector_form_record,
        pending.selector_form_identity_u8,
        pending.output_u8,
        pending.output_readback,
    )


def _selection_response(
        inner_turn: TerminalDialogueActTurnV1,
        pending: RouteAmbiguityPendingV1,
        candidate: RouteAmbiguityCandidateProjectionV1,
        ) -> RouteClarificationDialogueResponseV1:
    """复制已实际运行的 DLG-RAW-13 answer，并标出已验证的 explicit selection。"""
    if type(inner_turn) is not TerminalDialogueActTurnV1:
        raise TypeError("route clarification inner turn 类型错误")
    if type(pending) is not RouteAmbiguityPendingV1:
        raise TypeError("route clarification pending 类型错误")
    if type(candidate) is not RouteAmbiguityCandidateProjectionV1:
        raise TypeError("route clarification selected candidate 类型错误")
    base = inner_turn.response
    return RouteClarificationDialogueResponseV1(
        ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1,
        ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1,
        base.base_result_code,
        base.canonical_record(),
        _terminal_turn_intake(inner_turn),
        pending.pending_identity_u8,
        pending.route_identity_u8,
        candidate.candidate_identity_u8,
        pending.selector_catalog_record,
        pending.selector_catalog_identity_u8,
        pending.selector_form_record,
        pending.selector_form_identity_u8,
        base.output_u8,
        _output_readback(base.output_u8),
    )


def _current_selector_catalog(
        runtime: RouteClarificationDialogueRuntimeV1,
        ) -> PublicRouteClarificationCatalogV1:
    """每次 outer dispatch 重建 selector course binding，内存漂移只会关闭新行为。"""
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification runtime 类型错误")
    try:
        return validate_public_route_clarification_catalog_cached_v1(
            runtime.selector_catalog,
            runtime.terminal_runtime.inner_runtime.source_payload_closure,
            runtime.validation_cache,
        )
    except (PublicRouteClarificationCatalogError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueError(
            "route clarification selector catalog 当前验证失败") from error


def _pending_from_ambiguity(
        before: RouteClarificationDialogueStateV1,
        inner_turn: TerminalDialogueActTurnV1,
        runtime: RouteClarificationDialogueRuntimeV1,
        selector_catalog: PublicRouteClarificationCatalogV1,
        ) -> RouteAmbiguityPendingV1 | None:
    """仅为真实 Frame code-8 重演 V3 resolver 并创建一轮来源绑定 pending。"""
    if type(before) is not RouteClarificationDialogueStateV1:
        raise TypeError("route clarification before state 类型错误")
    if type(inner_turn) is not TerminalDialogueActTurnV1:
        raise TypeError("route clarification inner turn 类型错误")
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification runtime 类型错误")
    if type(selector_catalog) is not PublicRouteClarificationCatalogV1:
        raise TypeError("route clarification selector catalog 类型错误")
    response = inner_turn.response
    if (response.base_carrier_kind != TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1
            or response.base_result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS):
        return None
    intake = _terminal_turn_intake(inner_turn)
    if not intake.accepted:
        return None
    public_runtime = runtime.terminal_runtime.inner_runtime
    try:
        resolution = resolve_source_bound_slot_composition(
            public_runtime.source_bound_slot_catalog,
            public_runtime.base_catalog,
            public_runtime.active_catalog,
            intake.unicode_scalars,
            public_runtime.source_payload_closure,
        )
        if (type(resolution) is not SourceBoundSlotCompositionResolution
                or resolution.catalog.catalog_schema
                != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3
                or resolution.result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
                or resolution.input_scalars != intake.unicode_scalars
                or resolution.matched_frame_count < 2
                or len(resolution.target_candidates)
                != resolution.matched_frame_count
                or any(candidate.verdict
                       != SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3
                       for candidate in resolution.target_candidates)):
            return None
        form = selector_catalog.form_for_source_bound_resolution(resolution)
        if form is None:
            return None
        route_identity = route_identity_from_source_bound_resolution_v1(resolution)
        if (form.route_identity_u8 != route_identity
                or form.result_code != response.base_result_code
                or form.matched_frame_count != resolution.matched_frame_count
                or form.input_scalars != intake.unicode_scalars
                or len(form.options) != len(resolution.target_candidates)):
            return None
        projections = tuple(
            _candidate_projection(candidate, form.options[ordinal])
            for ordinal, candidate in enumerate(resolution.target_candidates)
        )
    except (ConversationRawRouteClarificationDialogueError,
            PublicRouteClarificationCatalogError, SourceBoundSlotCompositionError,
            TypeError, ValueError):
        return None
    if (inner_turn.before != before.inner_state
            or inner_turn.after.next_operation_ordinal
            != before.next_operation_ordinal + 1):
        raise ConversationRawRouteClarificationDialogueError(
            "route clarification inner ambiguity transition 不连续")
    return RouteAmbiguityPendingV1(
        before.next_operation_ordinal,
        inner_turn.after.next_operation_ordinal,
        intake,
        inner_turn.canonical_record(),
        response.canonical_record(),
        _state_identity(before.inner_state),
        _state_identity(inner_turn.after),
        resolution.canonical_record(),
        route_identity,
        selector_catalog.canonical_record(),
        selector_catalog.catalog_identity_u8,
        form.canonical_record(),
        form.form_identity_u8,
        projections,
        form.output_u8,
        _output_readback(form.output_u8),
    )


def _pending_still_binds_runtime(
        pending: RouteAmbiguityPendingV1,
        runtime: RouteClarificationDialogueRuntimeV1,
        selector_catalog: PublicRouteClarificationCatalogV1,
        ) -> bool:
    """在选择前重演 pending 的 route/course binding，漂移时不给 selection event。"""
    if type(pending) is not RouteAmbiguityPendingV1:
        raise TypeError("route clarification pending 类型错误")
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification runtime 类型错误")
    if type(selector_catalog) is not PublicRouteClarificationCatalogV1:
        raise TypeError("route clarification selector catalog 类型错误")
    if (pending.selector_catalog_record != selector_catalog.canonical_record()
            or pending.selector_catalog_identity_u8
            != selector_catalog.catalog_identity_u8):
        return False
    public_runtime = runtime.terminal_runtime.inner_runtime
    try:
        resolution = resolve_source_bound_slot_composition(
            public_runtime.source_bound_slot_catalog,
            public_runtime.base_catalog,
            public_runtime.active_catalog,
            pending.input_intake.unicode_scalars,
            public_runtime.source_payload_closure,
        )
        form = selector_catalog.form_for_source_bound_resolution(resolution)
        if form is None:
            return False
        if (resolution.canonical_record() != pending.resolution_record
                or route_identity_from_source_bound_resolution_v1(resolution)
                != pending.route_identity_u8
                or form.canonical_record() != pending.selector_form_record
                or form.form_identity_u8 != pending.selector_form_identity_u8
                or len(form.options) != len(pending.candidates)
                or len(resolution.target_candidates) != len(pending.candidates)):
            return False
        rebuilt = tuple(
            _candidate_projection(candidate, form.options[ordinal])
            for ordinal, candidate in enumerate(resolution.target_candidates)
        )
        return tuple(item.canonical_record() for item in rebuilt) == tuple(
            item.canonical_record() for item in pending.candidates)
    except (ConversationRawRouteClarificationDialogueError,
            PublicRouteClarificationCatalogError, SourceBoundSlotCompositionError,
            TypeError, ValueError):
        return False


def _frame_matches_candidate(
        answer: ConversationRawAnswerResult,
        candidate: RouteAmbiguityCandidateProjectionV1,
        intake: ConversationRawIntake,
        ) -> bool:
    """检查实际 RAW-02 answer 的 Frame/target/recipe 与 pending candidate 完全一致。"""
    if type(answer) is not ConversationRawAnswerResult:
        raise TypeError("route clarification answer 类型错误")
    if type(candidate) is not RouteAmbiguityCandidateProjectionV1:
        raise TypeError("route clarification candidate 类型错误")
    if type(intake) is not ConversationRawIntake:
        raise TypeError("route clarification intake 类型错误")
    if (not answer.accepted or answer.result_code != DLG_RAW_ACCEPT
            or answer.ingress.intake.canonical_record() != intake.canonical_record()
            or type(answer.ingress.frame) is not PublicFrame):
        return False
    frame = answer.ingress.frame
    try:
        frame_key = _ascii_u8(frame.frame_key, label="route answer frame key")
        target_key = frame.question.target.stable_key()
        recipe = frame.recipe.canonical_record()
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        frame_key == candidate.base_frame_key_u8
        and tuple(frame.raw_line_sha256) == candidate.base_frame_raw_sha256_u8
        and target_key == candidate.target_key
        and recipe == candidate.recipe_record
    )


def _selection_event_from_inner_turn(
        state: RouteClarificationDialogueStateV1,
        pending: RouteAmbiguityPendingV1,
        candidate: RouteAmbiguityCandidateProjectionV1,
        inner_turn: TerminalDialogueActTurnV1,
        ) -> RouteClarificationSelectionEventV1 | None:
    """只有完整重输和实际 accepted Frame answer 都匹配时才追加 selection event。"""
    if type(state) is not RouteClarificationDialogueStateV1:
        raise TypeError("route clarification selection state 类型错误")
    if type(pending) is not RouteAmbiguityPendingV1:
        raise TypeError("route clarification pending 类型错误")
    if type(candidate) is not RouteAmbiguityCandidateProjectionV1:
        raise TypeError("route clarification selected candidate 类型错误")
    if type(inner_turn) is not TerminalDialogueActTurnV1:
        raise TypeError("route clarification inner turn 类型错误")
    if (len(state.selection_events) >= ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1
            or state.inner_state != inner_turn.before
            or state.next_operation_ordinal
            != pending.selection_operation_ordinal
            or pending.inner_after_state_identity_u8
            != _state_identity(state.inner_state)
            or not _intake_matches_surface(
                _terminal_turn_intake(inner_turn), candidate.option_surface_u8)):
        return None
    response = inner_turn.response
    answer = inner_turn.inner_turn.answer
    if (response.base_carrier_kind != TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1
            or response.base_result_code != DLG_RAW_ACCEPT
            or type(answer) is not ConversationRawAnswerResult
            or response.output_u8 != answer.output_bytes
            or not _frame_matches_candidate(
                answer, candidate, _terminal_turn_intake(inner_turn))):
        return None
    predecessor = (() if not state.selection_events
                   else state.selection_events[-1].event_identity_u8)
    return RouteClarificationSelectionEventV1(
        len(state.selection_events) + 1,
        predecessor,
        pending.canonical_record(),
        pending.pending_identity_u8,
        candidate.canonical_record(),
        candidate.candidate_identity_u8,
        _terminal_turn_intake(inner_turn),
        inner_turn.canonical_record(),
        response.canonical_record(),
        _state_identity(inner_turn.before),
        _state_identity(inner_turn.after),
        response.output_u8,
        _output_readback(response.output_u8),
    )


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationDialogueTurnV1:
    """一次 DLG-RAW-14 transition：保留完整 DLG-RAW-13 turn 与可选 selection event。"""

    before: RouteClarificationDialogueStateV1
    inner_turn: TerminalDialogueActTurnV1
    response: RouteClarificationDialogueResponseV1
    selection_event: RouteClarificationSelectionEventV1 | None
    after: RouteClarificationDialogueStateV1

    def __post_init__(self) -> None:
        """验证 outer/inner 连续性、response provenance 和 pending/event 转换边界。"""
        if type(self.before) is not RouteClarificationDialogueStateV1:
            raise TypeError("route clarification turn before 类型错误")
        if type(self.inner_turn) is not TerminalDialogueActTurnV1:
            raise TypeError("route clarification inner turn 类型错误")
        if type(self.response) is not RouteClarificationDialogueResponseV1:
            raise TypeError("route clarification response 类型错误")
        if (self.selection_event is not None
                and type(self.selection_event)
                is not RouteClarificationSelectionEventV1):
            raise TypeError("route clarification selection event 类型错误")
        if type(self.after) is not RouteClarificationDialogueStateV1:
            raise TypeError("route clarification turn after 类型错误")
        if (self.inner_turn.before != self.before.inner_state
                or self.inner_turn.after != self.after.inner_state
                or self.before.conversation_key != self.after.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1
                or self.response.inner_response_record
                != self.inner_turn.response.canonical_record()
                or self.response.base_result_code
                != self.inner_turn.response.base_result_code
                or self.response.input_intake.canonical_record()
                != _terminal_turn_intake(self.inner_turn).canonical_record()):
            raise ConversationRawRouteClarificationDialogueError(
                "route clarification outer/inner transition 或 response binding 漂移")
        before_pending = self.before.pending
        after_pending = self.after.pending
        if self.response.response_kind == (
                ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1):
            if (self.selection_event is not None or after_pending is None
                    or self.response.pending_identity_u8
                    != after_pending.pending_identity_u8
                    or (before_pending is not None
                        and after_pending.pending_identity_u8
                        == before_pending.pending_identity_u8)):
                raise ConversationRawRouteClarificationDialogueError(
                    "route options transition 必须打开新的 pending 且不得追加 selection")
            return
        if self.response.response_kind == (
                ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1):
            event = self.selection_event
            if (event is None or after_pending is not None
                    or self.response.pending_identity_u8 != event.pending_identity_u8
                    or self.response.selected_candidate_identity_u8
                    != event.selected_candidate_identity_u8
                    or self.after.selection_events != (
                        *self.before.selection_events, event)):
                raise ConversationRawRouteClarificationDialogueError(
                    "route selection transition 未严格追加 event 或清除 pending")
            return
        if self.selection_event is not None:
            raise ConversationRawRouteClarificationDialogueError(
                "route passthrough 不得追加 selection event")
        if self.response.state_effect == ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1:
            if (before_pending is not None or after_pending is not None
                    or self.after.selection_events != self.before.selection_events):
                raise ConversationRawRouteClarificationDialogueError(
                    "route state-none passthrough 不得改变 outer state")
            return
        if self.response.state_effect == ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1:
            if (before_pending is None or after_pending is not None
                    or self.after.selection_events != self.before.selection_events):
                raise ConversationRawRouteClarificationDialogueError(
                    "route pending expire passthrough state 漂移")
            return
        raise ConversationRawRouteClarificationDialogueError(
            "route passthrough state effect 未注册")

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 outer transition，以 immutable records 取代隐式 terminal history。"""
        result = [ROUTE_CLARIFICATION_DIALOGUE_TURN_RECORD_V1]
        for label, value in (
                ("before", self.before.canonical_record()),
                ("inner turn", self.inner_turn.canonical_record()),
                ("response", self.response.canonical_record()),
                ("selection event", (() if self.selection_event is None
                                     else self.selection_event.canonical_record())),
                ("after", self.after.canonical_record())):
            _pack(result, value, label=f"route clarification turn {label}")
        return tuple(result)


def start_public_route_clarification_dialogue(
        conversation_key: tuple[int, ...],
        ) -> RouteClarificationDialogueStateV1:
    """建立空 DLG-RAW-14 state；没有候选 cache、shadow cursor 或长期记忆。"""
    return RouteClarificationDialogueStateV1(
        start_public_terminal_dialogue_act(conversation_key))


def run_public_route_clarification_dialogue_turn_v1(
        state: RouteClarificationDialogueStateV1,
        raw_input_bytes: tuple[int, ...],
        runtime: RouteClarificationDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> RouteClarificationDialogueTurnV1:
    """执行一轮：先且只先运行 DLG-RAW-13，再投影候选或验证显式重输选择。"""
    if type(state) is not RouteClarificationDialogueStateV1:
        raise TypeError("route clarification state 类型错误")
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification runtime 类型错误")
    try:
        selector_catalog = _current_selector_catalog(runtime)
    except ConversationRawRouteClarificationDialogueError:
        # 内层仍应可运行；course 漂移关闭新增 capability 而不改变既有 DLG-RAW-13。
        selector_catalog = None
    inner_turn = run_public_terminal_dialogue_act_turn_v1(
        state.inner_state,
        raw_input_bytes,
        runtime.terminal_runtime,
        preparation_cache=preparation_cache,
        preflight_cache=preflight_cache,
    )
    prior_pending = state.pending
    selected_candidate = (None if prior_pending is None
                          else prior_pending.candidate_for_intake(
                              _terminal_turn_intake(inner_turn)))
    selection = None
    if (prior_pending is not None and selected_candidate is not None
            and selector_catalog is not None
            and _pending_still_binds_runtime(
                prior_pending, runtime, selector_catalog)):
        selection = _selection_event_from_inner_turn(
            state,
            prior_pending,
            selected_candidate,
            inner_turn,
        )
    if selection is not None:
        after = RouteClarificationDialogueStateV1(
            inner_turn.after,
            None,
            (*state.selection_events, selection),
        )
        response = _selection_response(
            inner_turn,
            prior_pending,
            selected_candidate,
        )
        return RouteClarificationDialogueTurnV1(
            state, inner_turn, response, selection, after)

    pending = (None if selector_catalog is None else _pending_from_ambiguity(
        state,
        inner_turn,
        runtime,
        selector_catalog,
    ))
    if pending is not None:
        after = RouteClarificationDialogueStateV1(
            inner_turn.after,
            pending,
            state.selection_events,
        )
        return RouteClarificationDialogueTurnV1(
            state,
            inner_turn,
            _options_response(inner_turn, pending),
            None,
            after,
        )

    after = RouteClarificationDialogueStateV1(
        inner_turn.after,
        None,
        state.selection_events,
    )
    effect = (ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1
              if prior_pending is None
              else ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1)
    return RouteClarificationDialogueTurnV1(
        state,
        inner_turn,
        _passthrough_response(inner_turn, state_effect=effect),
        None,
        after,
    )


__all__ = [
    "ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_RECORD_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_BINDING_RECORD_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_PROTOCOL_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_STATE_RECORD_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_TURN_RECORD_V1",
    "ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1",
    "ROUTE_CLARIFICATION_PENDING_RECORD_V1",
    "ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1",
    "ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1",
    "ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1",
    "ROUTE_CLARIFICATION_RESPONSE_RECORD_V1",
    "ROUTE_CLARIFICATION_SELECTION_EVENT_RECORD_V1",
    "ROUTE_CLARIFICATION_STATE_EFFECT_NONE_V1",
    "ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1",
    "ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1",
    "ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1",
    "ConversationRawRouteClarificationDialogueError",
    "RouteAmbiguityCandidateProjectionV1",
    "RouteAmbiguityPendingV1",
    "RouteClarificationDialogueResponseV1",
    "RouteClarificationDialogueRuntimeV1",
    "RouteClarificationDialogueStateV1",
    "RouteClarificationDialogueTurnV1",
    "RouteClarificationSelectionEventV1",
    "build_public_route_clarification_dialogue_runtime_v1",
    "build_route_clarification_dialogue_runtime_v1",
    "run_public_route_clarification_dialogue_turn_v1",
    "start_public_route_clarification_dialogue",
]
