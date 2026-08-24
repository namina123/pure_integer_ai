"""DLG-RAW-12：连续来源焦点的独立 V3 append-only 会话纯状态。

V2 mixed context 及其 V4 session 均已冻结。本模块不扩写它们，而是用独立的
三分支 tagged union 保存 ``Frame``、provider anchor 和 follow-up focus event。
每个可观察状态均可写成有限非负整数 record；Python dataclass 只是一层当前宿主
的结构体便利，不能参与状态机、身份或恢复语义。

这里不读取 terminal、路径、SQLite、课程、runtime、网络或模型。follow-up 的语言
归约由后续 caller 完成；本模块只接纳已经逐项证明的 ``ProviderOriginFocusAdmissionV1``。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationTurnState,
)
from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    ProviderOriginAnchorProjectionV1,
    ProviderOriginOccurrenceV1,
    ProviderOriginRoleBindingV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    ConversationRawIntake,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)


MIXED_FOCUS_CONTEXT_SCHEMA_V3 = 3
MIXED_FOCUS_CONTEXT_STATE_RECORD_V3 = 3
MIXED_FOCUS_CONTEXT_READ_WITNESS_RECORD_V3 = 3
MIXED_FOCUS_CONTEXT_READ_RECORD_V3 = 3
MIXED_FOCUS_CONTEXT_FRAME_TURN_RECORD_V3 = 3
MIXED_FOCUS_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V3 = 3
MIXED_FOCUS_CONTEXT_FOLLOWUP_FOCUS_TURN_RECORD_V1 = 1
MIXED_FOCUS_CONTEXT_FOCUS_ADMISSION_RECORD_V1 = 1
MIXED_FOCUS_CONTEXT_APPEND_RESULT_RECORD_V1 = 1

MIXED_FOCUS_CONTEXT_TURN_KIND_FRAME_QA_RUN = 1
MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION = 2
MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_FOLLOWUP_FOCUS = 3

MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE = 0
MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN = 1
MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION = 2
MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_ORIGIN_FOLLOWUP_FOCUS = 3

MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED = 0
MIXED_FOCUS_CONTEXT_APPEND_REJECT_ANCHOR_NONE = 1
MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS = 2
MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL = 3
MIXED_FOCUS_CONTEXT_APPEND_REJECT_ADMISSION = 4

MIXED_FOCUS_CONTEXT_SNAPSHOT_IDENTITY_DOMAIN_V3 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/MIXED-FOCUS-CONTEXT-SNAPSHOT/V3")
MIXED_FOCUS_CONTEXT_READ_WITNESS_IDENTITY_DOMAIN_V3 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/MIXED-FOCUS-CONTEXT-READ-WITNESS/V3")
MIXED_FOCUS_CONTEXT_TURN_IDENTITY_DOMAIN_V3 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/MIXED-FOCUS-CONTEXT-TURN/V3")
MIXED_FOCUS_CONTEXT_ADMISSION_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/FOCUS-ADMISSION/V1")
MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/FOCUS-INPUT-INTAKE/V1")
MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/FOCUS-OUTPUT-READBACK/V1")

_DIGEST_SIZE = 32


# object-model: exception; interop=DLG-RAW-12
class ProviderOriginFocusContextError(ValueError):
    """V3 focus context 的整数 record、来源锚点或前驱链不闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """以显式 count framing 写入一段有限非负整数。"""
    record = _vector(value, label="focus context canonical segment", allow_empty=True)
    result.extend((len(record), *record))


def _vector(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """核验不依赖宿主 collection 语义的有序非负整数 vector。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ProviderOriginFocusContextError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验不可为空的稳定整数 key。"""
    return _vector(value, label=label, allow_empty=False)


def _digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 raw u8[32] identity，拒绝 host hex 与对象身份。"""
    result = _vector(value, label=label, allow_empty=False)
    if len(result) != _DIGEST_SIZE or any(item > 255 for item in result):
        raise ProviderOriginFocusContextError(f"{label} 必须是 raw u8[32]")
    return result


def _u8(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """核验显式 ``0..255`` byte vector。"""
    result = _vector(value, label=label, allow_empty=allow_empty)
    if any(item > 255 for item in result):
        raise ProviderOriginFocusContextError(f"{label} 必须只含 u8")
    return result


def _unicode_scalars(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """核验 UTF-8 编码前的 Unicode scalar 序列。"""
    result = _vector(value, label=label, allow_empty=allow_empty)
    if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in result):
        raise ProviderOriginFocusContextError(f"{label} 含非法 Unicode scalar")
    return result


def _nonnegative(value: int, *, label: str) -> int:
    """核验协议标量，拒绝 bool、子类和负数。"""
    if type(value) is not int or value < 0:
        raise ProviderOriginFocusContextError(f"{label} 必须是非负严格整数")
    return value


def _positive(value: int, *, label: str) -> int:
    """核验必须大于零的协议标量。"""
    result = _nonnegative(value, label=label)
    if result == 0:
        raise ProviderOriginFocusContextError(f"{label} 必须大于零")
    return result


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """以 frozen portable SHA framing 形成一个 raw u8 identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ProviderOriginFocusContextError(f"{label} 无法形成") from error


def _read_scalar(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """从已验证 record 读取一个标量，拒绝截断。"""
    if cursor >= len(record):
        raise ProviderOriginFocusContextError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = False,
        ) -> tuple[tuple[int, ...], int]:
    """读取 count-framed 整数段并拒绝非规范范围。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    _nonnegative(count, label=f"{label} count")
    if count > len(record) - cursor:
        raise ProviderOriginFocusContextError(f"{label} 长度越界")
    value = _vector(
        record[cursor:cursor + count], label=label, allow_empty=allow_empty)
    return value, cursor + count


def _trailing_identity_matches(
        record: tuple[int, ...],
        identity: tuple[int, ...],
        *,
        label: str,
        ) -> bool:
    """核验外来 canonical record 的末尾 identity 段未被替换。"""
    return (
        len(record) >= _DIGEST_SIZE + 1
        and record[-(_DIGEST_SIZE + 1)] == _DIGEST_SIZE
        and record[-_DIGEST_SIZE:] == identity
    )


def _read_witness_body(value: "FocusContextReadWitnessV3") -> tuple[int, ...]:
    """写出不含 self identity 的 V3 read witness record。"""
    result = [MIXED_FOCUS_CONTEXT_READ_WITNESS_RECORD_V3]
    _pack(result, value.conversation_key)
    result.append(value.revision)
    _pack(result, value.snapshot_digest_u8)
    result.extend((value.requested_limit, value.visible_start_ordinal))
    result.append(len(value.visible_turn_identities_u8))
    for identity in value.visible_turn_identities_u8:
        _pack(result, identity)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class FocusContextReadWitnessV3:
    """一次明确尾部读取的可持久化 V3 整数见证。"""

    conversation_key: tuple[int, ...]
    revision: int
    snapshot_digest_u8: tuple[int, ...]
    requested_limit: int
    visible_start_ordinal: int
    visible_turn_identities_u8: tuple[tuple[int, ...], ...]
    witness_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 read scope、尾部范围与 raw witness identity。"""
        key = _key(self.conversation_key, label="focus context read conversation key")
        revision = _nonnegative(self.revision, label="focus context read revision")
        digest = _digest(
            self.snapshot_digest_u8,
            label="focus context read snapshot digest",
        )
        limit = _nonnegative(
            self.requested_limit,
            label="focus context read requested limit",
        )
        start = _nonnegative(
            self.visible_start_ordinal,
            label="focus context read visible start ordinal",
        )
        identities = self.visible_turn_identities_u8
        if (type(identities) is not tuple
                or any(type(item) is not tuple for item in identities)):
            raise ProviderOriginFocusContextError(
                "focus context read visible identities 必须是 tuple")
        identities = tuple(_digest(
            item,
            label=f"focus context read visible identity[{ordinal}]",
        ) for ordinal, item in enumerate(identities))
        if (start > revision
                or len(identities) != revision - start
                or len(identities) > limit):
            raise ProviderOriginFocusContextError(
                "focus context read visible range 与 requested limit 不一致")
        object.__setattr__(self, "conversation_key", key)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "snapshot_digest_u8", digest)
        object.__setattr__(self, "requested_limit", limit)
        object.__setattr__(self, "visible_start_ordinal", start)
        object.__setattr__(self, "visible_turn_identities_u8", identities)
        expected = _identity(
            MIXED_FOCUS_CONTEXT_READ_WITNESS_IDENTITY_DOMAIN_V3,
            _read_witness_body(self),
            label="focus context read witness identity",
        )
        supplied = self.witness_identity_u8
        if supplied and _digest(
                supplied,
                label="focus context read witness identity") != expected:
            raise ProviderOriginFocusContextError(
                "focus context read witness identity 漂移")
        object.__setattr__(self, "witness_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 future V3 codec 可直接恢复的完整 witness record。"""
        result = list(_read_witness_body(self))
        _pack(result, self.witness_identity_u8)
        return tuple(result)


def _frame_turn_body(value: "FrameQuestionAnswerTurnV3") -> tuple[int, ...]:
    """写出 Frame tagged turn 的不含 self identity envelope。"""
    result = [
        MIXED_FOCUS_CONTEXT_FRAME_TURN_RECORD_V3,
        MIXED_FOCUS_CONTEXT_TURN_KIND_FRAME_QA_RUN,
        value.append_ordinal,
    ]
    for segment in (
            value.previous_snapshot_digest_u8,
            value.prior_read_witness.canonical_record(),
            value.frame_turn.stable_key()):
        _pack(result, segment)
    result.append(value.context_write_origin)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class FrameQuestionAnswerTurnV3:
    """V3 中包装完整 legacy Frame typed record 的独立 tagged turn。"""

    append_ordinal: int
    previous_snapshot_digest_u8: tuple[int, ...]
    prior_read_witness: FocusContextReadWitnessV3
    frame_turn: ConversationTurnState
    context_write_origin: int = MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN
    turn_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 Frame payload、V3 predecessor 与 explicit prior read。"""
        ordinal = _nonnegative(
            self.append_ordinal,
            label="focus context Frame append ordinal",
        )
        previous = _digest(
            self.previous_snapshot_digest_u8,
            label="focus context Frame previous snapshot digest",
        )
        if type(self.prior_read_witness) is not FocusContextReadWitnessV3:
            raise TypeError("focus context Frame prior read witness 类型错误")
        if type(self.frame_turn) is not ConversationTurnState:
            raise TypeError("focus context Frame turn 必须包装 ConversationTurnState")
        if (type(self.context_write_origin) is not int
                or self.context_write_origin
                != MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN):
            raise ProviderOriginFocusContextError("focus context Frame write origin 漂移")
        if (self.prior_read_witness.revision != ordinal
                or self.prior_read_witness.snapshot_digest_u8 != previous):
            raise ProviderOriginFocusContextError(
                "focus context Frame predecessor 或 read witness 漂移")
        _vector(
            self.frame_turn.stable_key(),
            label="focus context Frame legacy typed record",
            allow_empty=False,
        )
        object.__setattr__(self, "append_ordinal", ordinal)
        object.__setattr__(self, "previous_snapshot_digest_u8", previous)
        expected = _identity(
            MIXED_FOCUS_CONTEXT_TURN_IDENTITY_DOMAIN_V3,
            _frame_turn_body(self),
            label="focus context Frame turn identity",
        )
        supplied = self.turn_identity_u8
        if supplied and _digest(
                supplied,
                label="focus context Frame turn identity") != expected:
            raise ProviderOriginFocusContextError(
                "focus context Frame turn identity 漂移")
        object.__setattr__(self, "turn_identity_u8", expected)

    @property
    def turn_kind(self) -> int:
        """返回 V3 tagged union 的冻结 Frame kind。"""
        return MIXED_FOCUS_CONTEXT_TURN_KIND_FRAME_QA_RUN

    @property
    def target_key(self) -> tuple[int, ...]:
        """暴露 legacy Frame 已有 target key，不从表面文本推断。"""
        return self.frame_turn.target_key

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 V3 Frame envelope 与 identity。"""
        result = list(_frame_turn_body(self))
        _pack(result, self.turn_identity_u8)
        return tuple(result)


def _provider_turn_body(value: "ProviderOriginContextTurnV3") -> tuple[int, ...]:
    """写出 V3 provider-origin turn 的不含 self identity envelope。"""
    result = [
        MIXED_FOCUS_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V3,
        MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
        value.append_ordinal,
    ]
    for segment in (
            value.previous_snapshot_digest_u8,
            value.prior_read_witness.canonical_record(),
            value.anchor_projection.canonical_record(),
            value.provider_result_identity_u8):
        _pack(result, segment)
    result.append(value.context_write_origin)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ProviderOriginContextTurnV3:
    """V3 中完整保存一个可消费 provider source anchor 的 tagged turn。"""

    append_ordinal: int
    previous_snapshot_digest_u8: tuple[int, ...]
    prior_read_witness: FocusContextReadWitnessV3
    anchor_projection: ProviderOriginAnchorProjectionV1
    provider_result_identity_u8: tuple[int, ...]
    context_write_origin: int = (
        MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION)
    turn_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """只接纳完整 ANCHOR_ANSWER，禁止以短文本替代来源锚点。"""
        ordinal = _nonnegative(
            self.append_ordinal,
            label="focus context provider append ordinal",
        )
        previous = _digest(
            self.previous_snapshot_digest_u8,
            label="focus context provider previous snapshot digest",
        )
        if type(self.prior_read_witness) is not FocusContextReadWitnessV3:
            raise TypeError("focus context provider prior read witness 类型错误")
        if type(self.anchor_projection) is not ProviderOriginAnchorProjectionV1:
            raise TypeError("focus context provider anchor projection 类型错误")
        if not self.anchor_projection.accepted:
            raise ProviderOriginFocusContextError(
                "focus context provider turn 只能消费 ANCHOR_ANSWER")
        provider_result = _digest(
            self.provider_result_identity_u8,
            label="focus context provider result identity",
        )
        if provider_result != self.anchor_projection.provider_result_identity_u8:
            raise ProviderOriginFocusContextError(
                "focus context provider result identity 与 anchor 漂移")
        if (type(self.context_write_origin) is not int
                or self.context_write_origin
                != MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION):
            raise ProviderOriginFocusContextError(
                "focus context provider write origin 漂移")
        if (self.prior_read_witness.revision != ordinal
                or self.prior_read_witness.snapshot_digest_u8 != previous):
            raise ProviderOriginFocusContextError(
                "focus context provider predecessor 或 read witness 漂移")
        object.__setattr__(self, "append_ordinal", ordinal)
        object.__setattr__(self, "previous_snapshot_digest_u8", previous)
        object.__setattr__(self, "provider_result_identity_u8", provider_result)
        expected = _identity(
            MIXED_FOCUS_CONTEXT_TURN_IDENTITY_DOMAIN_V3,
            _provider_turn_body(self),
            label="focus context provider turn identity",
        )
        supplied = self.turn_identity_u8
        if supplied and _digest(
                supplied,
                label="focus context provider turn identity") != expected:
            raise ProviderOriginFocusContextError(
                "focus context provider turn identity 漂移")
        object.__setattr__(self, "turn_identity_u8", expected)

    @property
    def turn_kind(self) -> int:
        """返回 V3 tagged union 的 provider anchor kind。"""
        return MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 V3 provider turn record。"""
        result = list(_provider_turn_body(self))
        _pack(result, self.turn_identity_u8)
        return tuple(result)


def _admission_body(value: "ProviderOriginFocusAdmissionV1") -> tuple[int, ...]:
    """写出不含 self identity 的已验证 focus admission record。"""
    result = [
        MIXED_FOCUS_CONTEXT_FOCUS_ADMISSION_RECORD_V1,
        value.provider_kind,
        value.relation_kind_code,
        value.reference_start,
        value.reference_end,
        value.target_start,
        value.target_end,
    ]
    for segment in (
            value.catalog_record,
            value.catalog_identity_u8,
            value.form_record,
            value.form_identity_u8,
            value.candidate_record,
            value.candidate_identity_u8,
            value.input_intake_record,
            value.input_intake_identity_u8,
            value.output_readback_record,
            value.output_readback_identity_u8,
            value.provider_identity_u8,
            value.runtime_identity_u8,
            value.provider_catalog_identity_u8,
            value.provider_result_identity_u8,
            value.anchor_identity_u8,
            value.source_record_key,
            value.source_ref_stable_key,
            value.source_commitment_u8,
            value.w03_observation_key,
            value.w04_observation_key,
            value.w05_observation_key,
            value.generation_construction_key,
            value.proposition_key,
            value.predicate_key,
            value.reference_role_binding_key,
            value.reference_role_key,
            value.reference_filler_key,
            value.reference_occurrence_key,
            value.target_role_binding_key,
            value.target_role_key,
            value.target_filler_key,
            value.target_occurrence_key,
            value.target_output_scalars,
            value.target_output_u8):
        _pack(result, segment)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ProviderOriginFocusAdmissionV1:
    """一条已验证 follow-up 的完整、可迁移的 focus append 输入。

    这是 reducer 与 V3 state 的明确边界。它保留 form/catalog/candidate/input/
    output-readback 的完整 canonical records 和 raw identities，并显式带出 candidate
    的 reference 与 target occurrence；不把 V2 ``context_read`` 偷渡为 V3 语义。
    """

    catalog_record: tuple[int, ...]
    catalog_identity_u8: tuple[int, ...]
    form_record: tuple[int, ...]
    form_identity_u8: tuple[int, ...]
    candidate_record: tuple[int, ...]
    candidate_identity_u8: tuple[int, ...]
    input_intake_record: tuple[int, ...]
    input_intake_identity_u8: tuple[int, ...]
    output_readback_record: tuple[int, ...]
    output_readback_identity_u8: tuple[int, ...]
    provider_kind: int
    provider_identity_u8: tuple[int, ...]
    runtime_identity_u8: tuple[int, ...]
    provider_catalog_identity_u8: tuple[int, ...]
    provider_result_identity_u8: tuple[int, ...]
    anchor_identity_u8: tuple[int, ...]
    source_record_key: tuple[int, ...]
    source_ref_stable_key: tuple[int, ...]
    source_commitment_u8: tuple[int, ...]
    w03_observation_key: tuple[int, ...]
    w04_observation_key: tuple[int, ...]
    w05_observation_key: tuple[int, ...]
    generation_construction_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    relation_kind_code: int
    reference_role_binding_key: tuple[int, ...]
    reference_role_key: tuple[int, ...]
    reference_filler_key: tuple[int, ...]
    reference_occurrence_key: tuple[int, ...]
    reference_start: int
    reference_end: int
    target_role_binding_key: tuple[int, ...]
    target_role_key: tuple[int, ...]
    target_filler_key: tuple[int, ...]
    target_occurrence_key: tuple[int, ...]
    target_start: int
    target_end: int
    target_output_scalars: tuple[int, ...]
    target_output_u8: tuple[int, ...]
    admission_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 carrier，拒绝任何未回读、未绑定或不可重放的字段。"""
        for name in (
                "catalog_record", "form_record", "candidate_record",
                "input_intake_record", "output_readback_record"):
            object.__setattr__(self, name, _vector(
                getattr(self, name),
                label=f"focus admission {name}",
                allow_empty=False,
            ))
        for name in (
                "catalog_identity_u8", "form_identity_u8",
                "candidate_identity_u8", "input_intake_identity_u8",
                "output_readback_identity_u8", "provider_identity_u8",
                "runtime_identity_u8", "provider_catalog_identity_u8",
                "provider_result_identity_u8", "anchor_identity_u8",
                "source_commitment_u8"):
            object.__setattr__(self, name, _digest(
                getattr(self, name), label=f"focus admission {name}"))
        for name in (
                "source_record_key", "source_ref_stable_key",
                "w03_observation_key", "w04_observation_key",
                "w05_observation_key", "generation_construction_key",
                "proposition_key", "predicate_key",
                "reference_role_binding_key", "reference_role_key",
                "reference_filler_key", "reference_occurrence_key",
                "target_role_binding_key", "target_role_key",
                "target_filler_key", "target_occurrence_key"):
            object.__setattr__(self, name, _key(
                getattr(self, name), label=f"focus admission {name}"))
        if type(self.provider_kind) is not int or self.provider_kind <= 0:
            raise ProviderOriginFocusContextError(
                "focus admission provider kind 未注册")
        relation = _positive(
            self.relation_kind_code,
            label="focus admission relation kind",
        )
        reference_start = _nonnegative(
            self.reference_start,
            label="focus admission reference start",
        )
        reference_end = _positive(
            self.reference_end,
            label="focus admission reference end",
        )
        target_start = _nonnegative(
            self.target_start,
            label="focus admission target start",
        )
        target_end = _positive(
            self.target_end,
            label="focus admission target end",
        )
        if reference_end <= reference_start or target_end <= target_start:
            raise ProviderOriginFocusContextError(
                "focus admission occurrence span 非法")
        if (self.reference_role_binding_key == self.target_role_binding_key
                or self.reference_filler_key == self.target_filler_key
                or self.reference_occurrence_key == self.target_occurrence_key):
            raise ProviderOriginFocusContextError(
                "focus admission reference 与 target 不得是同一 occurrence")
        scalars = _unicode_scalars(
            self.target_output_scalars,
            label="focus admission target output scalars",
            allow_empty=False,
        )
        output = _u8(
            self.target_output_u8,
            label="focus admission target output u8",
            allow_empty=False,
        )
        if encode_utf8_v1(scalars) != output:
            raise ProviderOriginFocusContextError(
                "focus admission target UTF-8 output 漂移")
        if not _trailing_identity_matches(
                self.catalog_record, self.catalog_identity_u8,
                label="focus admission catalog"):
            raise ProviderOriginFocusContextError(
                "focus admission catalog record identity 漂移")
        if not _trailing_identity_matches(
                self.form_record, self.form_identity_u8,
                label="focus admission form"):
            raise ProviderOriginFocusContextError(
                "focus admission form record identity 漂移")
        if not _trailing_identity_matches(
                self.candidate_record, self.candidate_identity_u8,
                label="focus admission candidate"):
            raise ProviderOriginFocusContextError(
                "focus admission candidate record identity 漂移")
        expected_input = _identity(
            MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1,
            self.input_intake_record,
            label="focus admission input intake identity",
        )
        if self.input_intake_identity_u8 != expected_input:
            raise ProviderOriginFocusContextError(
                "focus admission input intake identity 漂移")
        try:
            readback = intake_raw_conversation_vector(output)
        except (TypeError, ValueError) as error:
            raise ProviderOriginFocusContextError(
                "focus admission output readback 无法形成") from error
        if (not readback.accepted
                or readback.unicode_scalars != scalars
                or readback.canonical_record() != self.output_readback_record):
            raise ProviderOriginFocusContextError(
                "focus admission output readback record 漂移")
        expected_readback = _identity(
            MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1,
            self.output_readback_record,
            label="focus admission output readback identity",
        )
        if self.output_readback_identity_u8 != expected_readback:
            raise ProviderOriginFocusContextError(
                "focus admission output readback identity 漂移")
        object.__setattr__(self, "relation_kind_code", relation)
        object.__setattr__(self, "reference_start", reference_start)
        object.__setattr__(self, "reference_end", reference_end)
        object.__setattr__(self, "target_start", target_start)
        object.__setattr__(self, "target_end", target_end)
        object.__setattr__(self, "target_output_scalars", scalars)
        object.__setattr__(self, "target_output_u8", output)
        expected = _identity(
            MIXED_FOCUS_CONTEXT_ADMISSION_IDENTITY_DOMAIN_V1,
            _admission_body(self),
            label="focus admission identity",
        )
        supplied = self.admission_identity_u8
        if supplied and _digest(
                supplied, label="focus admission identity") != expected:
            raise ProviderOriginFocusContextError(
                "focus admission identity 漂移")
        object.__setattr__(self, "admission_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 follow-up admission，不保留 Python object 引用。"""
        result = list(_admission_body(self))
        _pack(result, self.admission_identity_u8)
        return tuple(result)


def _single_binding(
        anchor: ProviderOriginAnchorProjectionV1,
        *,
        binding_key: tuple[int, ...],
        role_key: tuple[int, ...],
        filler_key: tuple[int, ...],
        ) -> ProviderOriginRoleBindingV1 | None:
    """从同一 anchor 的显式 bindings 中查找唯一 structural target。"""
    found = tuple(
        item for item in anchor.ordered_role_bindings
        if (item.binding_key == binding_key
            and item.role_key == role_key
            and item.filler_key == filler_key))
    return found[0] if len(found) == 1 else None


def _single_occurrence(
        anchor: ProviderOriginAnchorProjectionV1,
        *,
        occurrence_key: tuple[int, ...],
        filler_key: tuple[int, ...],
        start: int,
        end: int,
        ) -> ProviderOriginOccurrenceV1 | None:
    """从同一 anchor 的显式 occurrences 中查找唯一 span。"""
    found = tuple(
        item for item in anchor.ordered_occurrences
        if (item.occurrence_key == occurrence_key
            and item.semantic_object_key == filler_key
            and item.start == start
            and item.end == end))
    return found[0] if len(found) == 1 else None


def _anchor_slice(
        anchor: ProviderOriginAnchorProjectionV1,
        *,
        start: int,
        end: int,
        ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """从 anchor 原始 scalar span 构造并逐 byte 回读输出。"""
    if start < 0 or end <= start or end > len(anchor.output_scalars):
        return None
    scalars = anchor.output_scalars[start:end]
    try:
        output = encode_utf8_v1(scalars)
        byte_start = len(encode_utf8_v1(anchor.output_scalars[:start]))
    except (TypeError, ValueError):
        return None
    byte_end = byte_start + len(output)
    if (byte_end > len(anchor.output_u8)
            or anchor.output_u8[byte_start:byte_end] != output):
        return None
    return scalars, output


def _admission_matches_anchor(
        admission: ProviderOriginFocusAdmissionV1,
        anchor: ProviderOriginAnchorProjectionV1,
        ) -> bool:
    """逐字段验证 admission 仍绑定同一完整 source anchor 与 target slice。"""
    if not anchor.accepted:
        return False
    try:
        if (admission.provider_kind != anchor.provider_kind
                or admission.provider_identity_u8 != anchor.provider_identity_u8
                or admission.runtime_identity_u8 != anchor.runtime_identity_u8
                or admission.provider_catalog_identity_u8
                != anchor.catalog_record_identity_u8
                or admission.provider_result_identity_u8
                != anchor.provider_result_identity_u8
                or admission.anchor_identity_u8 != anchor.anchor_identity_u8
                or admission.source_record_key != anchor.source_record_key
                or admission.source_ref_stable_key != anchor.source_ref_stable_key
                or admission.source_commitment_u8 != anchor.source_commitment_u8
                or admission.w03_observation_key != anchor.w03_observation_key
                or admission.w04_observation_key != anchor.w04_observation_key
                or admission.w05_observation_key != anchor.w05_observation_key
                or admission.generation_construction_key
                != anchor.generation_construction_key
                or admission.proposition_key != anchor.proposition_key
                or admission.predicate_key != anchor.predicate_key
                or admission.relation_kind_code != anchor.relation_kind_code
                or _single_binding(
                    anchor,
                    binding_key=admission.target_role_binding_key,
                    role_key=admission.target_role_key,
                    filler_key=admission.target_filler_key,
                ) is None
                or _single_occurrence(
                    anchor,
                    occurrence_key=admission.target_occurrence_key,
                    filler_key=admission.target_filler_key,
                    start=admission.target_start,
                    end=admission.target_end,
                ) is None):
            return False
        sliced = _anchor_slice(
            anchor,
            start=admission.target_start,
            end=admission.target_end,
        )
        return sliced == (
            admission.target_output_scalars,
            admission.target_output_u8,
        )
    except (ProviderOriginFocusContextError, TypeError, ValueError):
        return False


def _admission_reference_matches_provider(
        admission: ProviderOriginFocusAdmissionV1,
        anchor: ProviderOriginAnchorProjectionV1,
        ) -> bool:
    """验证 provider->focus 的 reference 正是 provider 已声明的 question focus。"""
    return (
        admission.reference_role_binding_key == anchor.focus_role_binding_key
        and admission.reference_role_key == anchor.focus_role_key
        and admission.reference_filler_key == anchor.focus_filler_key
        and admission.reference_occurrence_key == anchor.focus_occurrence_key
        and admission.reference_start == anchor.focus_answer_start
        and admission.reference_end == anchor.focus_answer_end
        and _single_binding(
            anchor,
            binding_key=admission.reference_role_binding_key,
            role_key=admission.reference_role_key,
            filler_key=admission.reference_filler_key,
        ) is not None
        and _single_occurrence(
            anchor,
            occurrence_key=admission.reference_occurrence_key,
            filler_key=admission.reference_filler_key,
            start=admission.reference_start,
            end=admission.reference_end,
        ) is not None
    )


def _focus_turn_body(
        value: "ProviderOriginFollowupFocusTurnV1",
        ) -> tuple[int, ...]:
    """写出不含 self identity 的 V3 follow-up-focus event record。"""
    result = [
        MIXED_FOCUS_CONTEXT_FOLLOWUP_FOCUS_TURN_RECORD_V1,
        MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_FOLLOWUP_FOCUS,
        value.append_ordinal,
    ]
    for segment in (
            value.previous_snapshot_digest_u8,
            value.prior_read_witness.canonical_record(),
            value.parent_anchor_projection.canonical_record(),
            value.parent_turn_identity_u8,
            value.admission.canonical_record()):
        _pack(result, segment)
    result.append(value.context_write_origin)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ProviderOriginFollowupFocusTurnV1:
    """一次可恢复的 source-bound focus append event。

    该 event 自身就是当前 focus，不依赖可变游标或外层 Python state。其完整 parent
    anchor、立即父轮 identity 和 admission 都被写入 canonical record，因此 restore
    可以逐 event 重放而不扫描较早历史。
    """

    append_ordinal: int
    previous_snapshot_digest_u8: tuple[int, ...]
    prior_read_witness: FocusContextReadWitnessV3
    parent_anchor_projection: ProviderOriginAnchorProjectionV1
    parent_turn_identity_u8: tuple[int, ...]
    admission: ProviderOriginFocusAdmissionV1
    context_write_origin: int = (
        MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_ORIGIN_FOLLOWUP_FOCUS)
    turn_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 event 本体；parent-tail relation 在 state replay 中逐轮验证。"""
        ordinal = _nonnegative(
            self.append_ordinal,
            label="focus context focus append ordinal",
        )
        previous = _digest(
            self.previous_snapshot_digest_u8,
            label="focus context focus previous snapshot digest",
        )
        if type(self.prior_read_witness) is not FocusContextReadWitnessV3:
            raise TypeError("focus context focus prior read witness 类型错误")
        if type(self.parent_anchor_projection) is not ProviderOriginAnchorProjectionV1:
            raise TypeError("focus context focus parent anchor 类型错误")
        if not self.parent_anchor_projection.accepted:
            raise ProviderOriginFocusContextError(
                "focus context focus parent anchor 必须为 ANSWER")
        parent_identity = _digest(
            self.parent_turn_identity_u8,
            label="focus context focus parent turn identity",
        )
        if type(self.admission) is not ProviderOriginFocusAdmissionV1:
            raise TypeError("focus context focus admission 类型错误")
        if (type(self.context_write_origin) is not int
                or self.context_write_origin
                != MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_ORIGIN_FOLLOWUP_FOCUS):
            raise ProviderOriginFocusContextError(
                "focus context focus write origin 漂移")
        if (self.prior_read_witness.revision != ordinal
                or self.prior_read_witness.snapshot_digest_u8 != previous):
            raise ProviderOriginFocusContextError(
                "focus context focus predecessor 或 read witness 漂移")
        if not _admission_matches_anchor(self.admission, self.parent_anchor_projection):
            raise ProviderOriginFocusContextError(
                "focus context focus admission 与 parent anchor 漂移")
        object.__setattr__(self, "append_ordinal", ordinal)
        object.__setattr__(self, "previous_snapshot_digest_u8", previous)
        object.__setattr__(self, "parent_turn_identity_u8", parent_identity)
        expected = _identity(
            MIXED_FOCUS_CONTEXT_TURN_IDENTITY_DOMAIN_V3,
            _focus_turn_body(self),
            label="focus context focus turn identity",
        )
        supplied = self.turn_identity_u8
        if supplied and _digest(
                supplied,
                label="focus context focus turn identity") != expected:
            raise ProviderOriginFocusContextError(
                "focus context focus turn identity 漂移")
        object.__setattr__(self, "turn_identity_u8", expected)

    @property
    def turn_kind(self) -> int:
        """返回 V3 tagged union 的 append-only focus kind。"""
        return MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_FOLLOWUP_FOCUS

    @property
    def current_role_binding_key(self) -> tuple[int, ...]:
        """返回本 event 成为下一轮唯一 visible focus 的 binding。"""
        return self.admission.target_role_binding_key

    @property
    def current_role_key(self) -> tuple[int, ...]:
        """返回本 event 的 current focus role。"""
        return self.admission.target_role_key

    @property
    def current_filler_key(self) -> tuple[int, ...]:
        """返回本 event 的 current focus filler。"""
        return self.admission.target_filler_key

    @property
    def current_occurrence_key(self) -> tuple[int, ...]:
        """返回本 event 的 current focus occurrence。"""
        return self.admission.target_occurrence_key

    @property
    def current_start(self) -> int:
        """返回本 event target occurrence 的 scalar start。"""
        return self.admission.target_start

    @property
    def current_end(self) -> int:
        """返回本 event target occurrence 的 scalar end。"""
        return self.admission.target_end

    @property
    def current_output_scalars(self) -> tuple[int, ...]:
        """返回仅由 anchor occurrence slice 派生的 scalar 输出。"""
        return self.admission.target_output_scalars

    @property
    def current_output_u8(self) -> tuple[int, ...]:
        """返回仅由 anchor occurrence slice 派生的 UTF-8 输出。"""
        return self.admission.target_output_u8

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 follow-up focus event record。"""
        result = list(_focus_turn_body(self))
        _pack(result, self.turn_identity_u8)
        return tuple(result)


MixedFocusContextTurnV3 = (
    FrameQuestionAnswerTurnV3
    | ProviderOriginContextTurnV3
    | ProviderOriginFollowupFocusTurnV1
)


def _turn(
        value: MixedFocusContextTurnV3,
        *,
        label: str,
        ) -> MixedFocusContextTurnV3:
    """验证 V3 tagged union 只含三种冻结 turn 分支。"""
    if (type(value) is not FrameQuestionAnswerTurnV3
            and type(value) is not ProviderOriginContextTurnV3
            and type(value) is not ProviderOriginFollowupFocusTurnV1):
        raise TypeError(f"{label} 必须是已登记 focus context turn")
    return value


def _state_record(
        conversation_key: tuple[int, ...],
        revision: int,
        previous_snapshot_digest_u8: tuple[int, ...],
        turns: tuple[MixedFocusContextTurnV3, ...],
        ) -> tuple[int, ...]:
    """写出完整 V3 append-only snapshot record。"""
    result = [
        MIXED_FOCUS_CONTEXT_STATE_RECORD_V3,
        MIXED_FOCUS_CONTEXT_SCHEMA_V3,
    ]
    _pack(result, conversation_key)
    result.append(revision)
    _pack(result, previous_snapshot_digest_u8)
    result.append(len(turns))
    for turn in turns:
        _pack(result, turn.canonical_record())
    return tuple(result)


def _snapshot_digest(
        conversation_key: tuple[int, ...],
        revision: int,
        previous_snapshot_digest_u8: tuple[int, ...],
        turns: tuple[MixedFocusContextTurnV3, ...],
        ) -> tuple[int, ...]:
    """由完整 V3 state record 形成唯一 raw snapshot identity。"""
    return _identity(
        MIXED_FOCUS_CONTEXT_SNAPSHOT_IDENTITY_DOMAIN_V3,
        _state_record(
            conversation_key,
            revision,
            previous_snapshot_digest_u8,
            turns,
        ),
        label="focus context snapshot identity",
    )


def _read_matches_prefix(
        read: "FocusContextReadV3",
        conversation_key: tuple[int, ...],
        revision: int,
        snapshot_digest_u8: tuple[int, ...],
        prefix: tuple[MixedFocusContextTurnV3, ...],
        ) -> bool:
    """验证 read witness 精确描述该 prefix 请求的可见尾部。"""
    witness = read.witness
    if (witness.conversation_key != conversation_key
            or witness.revision != revision
            or witness.snapshot_digest_u8 != snapshot_digest_u8):
        return False
    visible = prefix[-witness.requested_limit:] if witness.requested_limit else ()
    if tuple(turn.canonical_record() for turn in read.turns) != tuple(
            turn.canonical_record() for turn in visible):
        return False
    return witness.visible_turn_identities_u8 == tuple(
        turn.turn_identity_u8 for turn in visible)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class FocusContextReadV3:
    """由 V3 snapshot 产生的 explicit read，含见证及可消费 typed tail。"""

    witness: FocusContextReadWitnessV3
    turns: tuple[MixedFocusContextTurnV3, ...]

    def __post_init__(self) -> None:
        """核验 visible ordinal、tagged turn identity 与 witness 一致。"""
        if type(self.witness) is not FocusContextReadWitnessV3:
            raise TypeError("focus context read witness 类型错误")
        if type(self.turns) is not tuple:
            raise ProviderOriginFocusContextError(
                "focus context read turns 必须是 tuple")
        turns = tuple(_turn(item, label=f"focus context read turn[{ordinal}]")
                      for ordinal, item in enumerate(self.turns))
        expected_ordinals = tuple(range(
            self.witness.visible_start_ordinal,
            self.witness.revision,
        ))
        if tuple(item.append_ordinal for item in turns) != expected_ordinals:
            raise ProviderOriginFocusContextError(
                "focus context read turns 不是 witness 所示连续尾部")
        identities = tuple(item.turn_identity_u8 for item in turns)
        if identities != self.witness.visible_turn_identities_u8:
            raise ProviderOriginFocusContextError(
                "focus context read turn identity 与 witness 漂移")
        object.__setattr__(self, "turns", turns)

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 V3 read record，供 append 与 snapshot 审计。"""
        result = [MIXED_FOCUS_CONTEXT_READ_RECORD_V3]
        _pack(result, self.witness.canonical_record())
        result.append(len(self.turns))
        for turn in self.turns:
            _pack(result, turn.canonical_record())
        return tuple(result)

    def latest_frame_target_turn(
            self,
            target_key: tuple[int, ...],
            ) -> FrameQuestionAnswerTurnV3 | None:
        """只在可见尾轮是同 target Frame 时返回，不跨 provider/focus 回溯。"""
        target = _key(target_key, label="focus context target key")
        if not self.turns:
            return None
        last = self.turns[-1]
        if type(last) is not FrameQuestionAnswerTurnV3:
            return None
        return last if last.target_key == target else None


def _focus_parent_matches(
        turn: ProviderOriginFollowupFocusTurnV1,
        prefix: tuple[MixedFocusContextTurnV3, ...],
        conversation_key: tuple[int, ...],
        revision: int,
        snapshot_digest_u8: tuple[int, ...],
        ) -> bool:
    """逐 event 重放 parent tail、anchor 与 reference-to-focus 关系。"""
    try:
        witness = turn.prior_read_witness
        if (witness.requested_limit != 1
                or witness.conversation_key != conversation_key
                or witness.revision != revision
                or witness.snapshot_digest_u8 != snapshot_digest_u8
                or len(prefix) == 0
                or len(witness.visible_turn_identities_u8) != 1):
            return False
        read = FocusContextReadV3(witness, prefix[-1:])
        if not _read_matches_prefix(
                read,
                conversation_key,
                revision,
                snapshot_digest_u8,
                prefix,
        ):
            return False
        parent = read.turns[0]
        if (type(parent) is not ProviderOriginContextTurnV3
                and type(parent) is not ProviderOriginFollowupFocusTurnV1):
            return False
        if parent.turn_identity_u8 != turn.parent_turn_identity_u8:
            return False
        parent_anchor = (
            parent.anchor_projection if type(parent) is ProviderOriginContextTurnV3
            else parent.parent_anchor_projection)
        if (parent_anchor.canonical_record()
                != turn.parent_anchor_projection.canonical_record()):
            return False
        if not _admission_matches_anchor(turn.admission, parent_anchor):
            return False
        if type(parent) is ProviderOriginContextTurnV3:
            return _admission_reference_matches_provider(
                turn.admission,
                parent_anchor,
            )
        return (
            turn.admission.reference_role_binding_key
            == parent.current_role_binding_key
            and turn.admission.reference_role_key == parent.current_role_key
            and turn.admission.reference_filler_key == parent.current_filler_key
            and turn.admission.reference_occurrence_key
            == parent.current_occurrence_key
            and turn.admission.reference_start == parent.current_start
            and turn.admission.reference_end == parent.current_end
            and _single_binding(
                parent_anchor,
                binding_key=turn.admission.reference_role_binding_key,
                role_key=turn.admission.reference_role_key,
                filler_key=turn.admission.reference_filler_key,
            ) is not None
            and _single_occurrence(
                parent_anchor,
                occurrence_key=turn.admission.reference_occurrence_key,
                filler_key=turn.admission.reference_filler_key,
                start=turn.admission.reference_start,
                end=turn.admission.reference_end,
            ) is not None
        )
    except (ProviderOriginFocusContextError, TypeError, ValueError):
        return False


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class MixedConversationFocusContextStateV3:
    """V3 append-only context：Frame、provider anchor 与 focus event 都可恢复。"""

    conversation_key: tuple[int, ...]
    revision: int
    previous_snapshot_digest_u8: tuple[int, ...]
    turns: tuple[MixedFocusContextTurnV3, ...] = ()

    def __post_init__(self) -> None:
        """逐 turn 重放 predecessor/read/focus-parent 链，拒绝游标式伪状态。"""
        key = _key(self.conversation_key, label="focus context conversation key")
        revision = _nonnegative(self.revision, label="focus context revision")
        if type(self.turns) is not tuple:
            raise ProviderOriginFocusContextError("focus context turns 必须是 tuple")
        turns = tuple(_turn(item, label=f"focus context turn[{ordinal}]")
                      for ordinal, item in enumerate(self.turns))
        if revision != len(turns):
            raise ProviderOriginFocusContextError(
                "focus context revision 必须等于 turn 数")
        if revision == 0:
            if self.previous_snapshot_digest_u8 != ():
                raise ProviderOriginFocusContextError(
                    "focus context 初始 snapshot 不得携带 previous digest")
            previous: tuple[int, ...] = ()
        else:
            previous = _digest(
                self.previous_snapshot_digest_u8,
                label="focus context previous snapshot digest",
            )
        object.__setattr__(self, "conversation_key", key)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "previous_snapshot_digest_u8", previous)
        object.__setattr__(self, "turns", turns)

        expected_prior = _snapshot_digest(key, 0, (), ())
        prefix: tuple[MixedFocusContextTurnV3, ...] = ()
        for ordinal, turn in enumerate(turns):
            if (turn.append_ordinal != ordinal
                    or turn.previous_snapshot_digest_u8 != expected_prior):
                raise ProviderOriginFocusContextError(
                    "focus context turn ordinal 或 predecessor digest 漂移")
            witness = turn.prior_read_witness
            read = FocusContextReadV3(
                witness,
                prefix[-witness.requested_limit:]
                if witness.requested_limit else (),
            )
            if not _read_matches_prefix(
                    read,
                    key,
                    ordinal,
                    expected_prior,
                    prefix,
            ):
                raise ProviderOriginFocusContextError(
                    "focus context turn prior read witness 漂移")
            if (type(turn) is ProviderOriginFollowupFocusTurnV1
                    and not _focus_parent_matches(
                        turn,
                        prefix,
                        key,
                        ordinal,
                        expected_prior,
                    )):
                raise ProviderOriginFocusContextError(
                    "focus context follow-up parent event 漂移")
            prefix = (*prefix, turn)
            expected_prior = _snapshot_digest(
                key,
                ordinal + 1,
                turn.previous_snapshot_digest_u8,
                prefix,
            )
        if revision and previous != turns[-1].previous_snapshot_digest_u8:
            raise ProviderOriginFocusContextError(
                "focus context state previous digest 与末轮不一致")

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 V3 state 的唯一 canonical integer record。"""
        return _state_record(
            self.conversation_key,
            self.revision,
            self.previous_snapshot_digest_u8,
            self.turns,
        )

    def digest(self) -> tuple[int, ...]:
        """返回当前 V3 snapshot 的 raw u8[32] identity。"""
        return _snapshot_digest(
            self.conversation_key,
            self.revision,
            self.previous_snapshot_digest_u8,
            self.turns,
        )

    def turn_records(self) -> tuple[tuple[int, ...], ...]:
        """返回按 append ordinal 排列的所有 tagged turn record。"""
        return tuple(turn.canonical_record() for turn in self.turns)

    def visible_turns(self, limit: int) -> tuple[MixedFocusContextTurnV3, ...]:
        """只返回显式预算内的尾轮；零上限不读任何 turn。"""
        bounded = _nonnegative(limit, label="focus context visible limit")
        return self.turns[-bounded:] if bounded else ()

    def read(self, limit: int) -> FocusContextReadV3:
        """形成绑定当前 digest 的 explicit tail read witness。"""
        bounded = _nonnegative(limit, label="focus context read limit")
        visible = self.visible_turns(bounded)
        witness = FocusContextReadWitnessV3(
            self.conversation_key,
            self.revision,
            self.digest(),
            bounded,
            self.revision - len(visible),
            tuple(turn.turn_identity_u8 for turn in visible),
        )
        return FocusContextReadV3(witness, visible)

    def latest_frame_target_turn(
            self,
            target_key: tuple[int, ...],
            ) -> FrameQuestionAnswerTurnV3 | None:
        """按一轮可见预算查询 Frame target，绝不穿过 provider/focus。"""
        return self.read(1).latest_frame_target_turn(target_key)

    def _read_is_current(self, read: FocusContextReadV3 | None) -> bool:
        """比较完整 canonical read，拒绝其他 revision、scope 或物理 replay。"""
        if type(read) is not FocusContextReadV3:
            return False
        expected = self.read(read.witness.requested_limit)
        return read.canonical_record() == expected.canonical_record()

    def admit_frame_qa_run(
            self,
            frame_turn: ConversationTurnState,
            prior_read: FocusContextReadV3 | None,
            ) -> "FocusContextAppendResultV1":
        """以当前 V3 witness append 一条 Frame turn，失败保持 snapshot 不变。"""
        if type(frame_turn) is not ConversationTurnState:
            raise TypeError("focus context Frame admission 需要 ConversationTurnState")
        if not self._read_is_current(prior_read):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS,
            )
        if prior_read is None:
            raise AssertionError("focus context current read 不得为空")
        if self.revision == 0 and prior_read.witness.requested_limit != 0:
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS,
            )
        turn = FrameQuestionAnswerTurnV3(
            self.revision,
            self.digest(),
            prior_read.witness,
            frame_turn,
        )
        after = MixedConversationFocusContextStateV3(
            self.conversation_key,
            self.revision + 1,
            self.digest(),
            (*self.turns, turn),
        )
        return FocusContextAppendResultV1(
            self,
            prior_read,
            MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
            MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED,
            turn,
            after,
        )

    def admit_provider_origin_projection(
            self,
            anchor_projection: ProviderOriginAnchorProjectionV1,
            prior_read: FocusContextReadV3 | None = None,
            ) -> "FocusContextAppendResultV1":
        """append 一个完整 provider anchor；ANCHOR_NONE 永远不读、不写。"""
        if type(anchor_projection) is not ProviderOriginAnchorProjectionV1:
            raise TypeError(
                "focus context provider admission 需要 ProviderOriginAnchorProjectionV1")
        if not anchor_projection.accepted:
            return _rejected_append_result(
                self,
                None,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
            )
        if not self._read_is_current(prior_read):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS,
            )
        if prior_read is None:
            raise AssertionError("focus context current read 不得为空")
        if self.revision == 0 and prior_read.witness.requested_limit != 0:
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS,
            )
        turn = ProviderOriginContextTurnV3(
            self.revision,
            self.digest(),
            prior_read.witness,
            anchor_projection,
            anchor_projection.provider_result_identity_u8,
        )
        after = MixedConversationFocusContextStateV3(
            self.conversation_key,
            self.revision + 1,
            self.digest(),
            (*self.turns, turn),
        )
        return FocusContextAppendResultV1(
            self,
            prior_read,
            MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
            MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED,
            turn,
            after,
        )

    def admit_provider_origin_followup_focus(
            self,
            admission: ProviderOriginFocusAdmissionV1,
            prior_read: FocusContextReadV3 | None,
            ) -> "FocusContextAppendResultV1":
        """只消费 ``read(1)`` 可见尾轮，append 下一条 source-bound focus event。"""
        if type(admission) is not ProviderOriginFocusAdmissionV1:
            raise TypeError(
                "focus context follow-up admission 需要 ProviderOriginFocusAdmissionV1")
        if not self._read_is_current(prior_read):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS,
            )
        if (prior_read is None
                or prior_read.witness.requested_limit != 1
                or len(prior_read.turns) != 1):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL,
            )
        parent = prior_read.turns[0]
        if type(parent) is ProviderOriginContextTurnV3:
            anchor = parent.anchor_projection
        elif type(parent) is ProviderOriginFollowupFocusTurnV1:
            anchor = parent.parent_anchor_projection
        else:
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL,
            )
        try:
            turn = ProviderOriginFollowupFocusTurnV1(
                self.revision,
                self.digest(),
                prior_read.witness,
                anchor,
                parent.turn_identity_u8,
                admission,
            )
        except (ProviderOriginFocusContextError, TypeError, ValueError):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_ADMISSION,
            )
        try:
            after = MixedConversationFocusContextStateV3(
                self.conversation_key,
                self.revision + 1,
                self.digest(),
                (*self.turns, turn),
            )
        except (ProviderOriginFocusContextError, TypeError, ValueError):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_FOCUS_CONTEXT_APPEND_REJECT_ADMISSION,
            )
        return FocusContextAppendResultV1(
            self,
            prior_read,
            MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_ORIGIN_FOLLOWUP_FOCUS,
            MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED,
            turn,
            after,
        )


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class FocusContextAppendResultV1:
    """一次 V3 admission 的显式 append/no-op transition 证据。"""

    before: MixedConversationFocusContextStateV3
    prior_read: FocusContextReadV3 | None
    context_write_origin: int
    result_code: int
    appended_turn: MixedFocusContextTurnV3 | None
    after: MixedConversationFocusContextStateV3

    def __post_init__(self) -> None:
        """冻结 write origin、result code、前后状态与 append/no-op 对应关系。"""
        if type(self.before) is not MixedConversationFocusContextStateV3:
            raise TypeError("focus context append before 类型错误")
        if (self.prior_read is not None
                and type(self.prior_read) is not FocusContextReadV3):
            raise TypeError("focus context append prior read 类型错误")
        if type(self.after) is not MixedConversationFocusContextStateV3:
            raise TypeError("focus context append after 类型错误")
        if (type(self.result_code) is not int
                or self.result_code not in {
                    MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED,
                    MIXED_FOCUS_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
                    MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS,
                    MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL,
                    MIXED_FOCUS_CONTEXT_APPEND_REJECT_ADMISSION,
                }):
            raise ProviderOriginFocusContextError(
                "focus context append result code 未注册")
        if (type(self.context_write_origin) is not int
                or self.context_write_origin not in {
                    MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE,
                    MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
                    MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
                    MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_ORIGIN_FOLLOWUP_FOCUS,
                }):
            raise ProviderOriginFocusContextError(
                "focus context append write origin 未注册")
        if self.result_code == MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED:
            if self.prior_read is None or self.appended_turn is None:
                raise ProviderOriginFocusContextError(
                    "focus context accepted append 缺 prior read 或 turn")
            appended = _turn(
                self.appended_turn,
                label="focus context appended turn",
            )
            if (self.context_write_origin == MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE
                    or self.after.conversation_key != self.before.conversation_key
                    or self.after.revision != self.before.revision + 1
                    or self.after.turn_records() != (*self.before.turn_records(),
                                                     appended.canonical_record())
                    or appended.previous_snapshot_digest_u8
                    != self.before.digest()
                    or not self.before._read_is_current(self.prior_read)
                    or (appended.prior_read_witness.canonical_record()
                        != self.prior_read.witness.canonical_record())):
                raise ProviderOriginFocusContextError(
                    "focus context accepted append 前后状态或 witness 漂移")
            expected_origin = (
                MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN
                if type(appended) is FrameQuestionAnswerTurnV3 else
                MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION
                if type(appended) is ProviderOriginContextTurnV3 else
                MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_ORIGIN_FOLLOWUP_FOCUS
            )
            if self.context_write_origin != expected_origin:
                raise ProviderOriginFocusContextError(
                    "focus context append write origin 与 tagged turn 不一致")
            return
        if (self.context_write_origin != MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE
                or self.appended_turn is not None
                or self.after.canonical_record() != self.before.canonical_record()):
            raise ProviderOriginFocusContextError(
                "focus context rejected append 必须为 NONE/no-op")
        if (self.result_code == MIXED_FOCUS_CONTEXT_APPEND_REJECT_ANCHOR_NONE
                and self.prior_read is not None):
            raise ProviderOriginFocusContextError(
                "focus context anchor-none 拒绝不得消费 prior read")

    @property
    def accepted(self) -> bool:
        """返回这次 admission 是否真正追加一条 V3 turn。"""
        return self.result_code == MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED

    def canonical_record(self) -> tuple[int, ...]:
        """导出 transition 的完整整数证据，供 runtime/codec 审计。"""
        result = [
            MIXED_FOCUS_CONTEXT_APPEND_RESULT_RECORD_V1,
            self.result_code,
            self.context_write_origin,
        ]
        for segment in (
                self.before.canonical_record(),
                (() if self.prior_read is None
                 else self.prior_read.canonical_record()),
                (() if self.appended_turn is None
                 else self.appended_turn.canonical_record()),
                self.after.canonical_record()):
            _pack(result, segment)
        return tuple(result)


def _rejected_append_result(
        state: MixedConversationFocusContextStateV3,
        prior_read: FocusContextReadV3 | None,
        result_code: int,
        ) -> FocusContextAppendResultV1:
    """统一构造写入来源为 NONE 的 immutable admission rejection。"""
    return FocusContextAppendResultV1(
        state,
        prior_read,
        MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE,
        result_code,
        None,
        state,
    )


def provider_origin_focus_admission_from_followup_result_v1(
        result: object,
        catalog: object,
        anchor: ProviderOriginAnchorProjectionV1,
        ) -> ProviderOriginFocusAdmissionV1:
    """从已有 follow-up result 显式复制 admission；不复用其 V2 read witness。

    该便利函数仅处于 reducer 与 V3 core 的边界。它读取完整 form/catalog/candidate/
    intake records，重新形成本模块定义的 input/output identities；V3 append 仍须由
    当前 ``read(1)`` 另行验证。
    """
    from pure_integer_ai.experiments.conversation_provider_origin_followup import (
        ProviderOriginFollowupCatalogV1,
        ProviderOriginFollowupResultV1,
    )

    if type(result) is not ProviderOriginFollowupResultV1:
        raise TypeError("focus admission result 类型错误")
    if type(catalog) is not ProviderOriginFollowupCatalogV1:
        raise TypeError("focus admission catalog 类型错误")
    if type(anchor) is not ProviderOriginAnchorProjectionV1:
        raise TypeError("focus admission anchor 类型错误")
    if (not result.accepted or result.form is None or result.candidate is None
            or not anchor.accepted
            or result.catalog_identity_u8 != catalog.catalog_identity_u8):
        raise ProviderOriginFocusContextError(
            "focus admission 只能从同一 catalog 的 accepted follow-up 形成")
    candidate = result.candidate
    form = result.form
    if (candidate.catalog_identity_u8 != catalog.catalog_identity_u8
            or candidate.form_identity_u8 != form.form_identity_u8):
        raise ProviderOriginFocusContextError(
            "focus admission candidate 未绑定当前 catalog/form")
    profiles = tuple(
        profile for profile in catalog.profiles_for_form(form)
        if (profile.profile_identity_u8 == candidate.profile_identity_u8
            and profile.profile_revision == candidate.profile_revision
            and profile.form_identity_u8 == candidate.form_identity_u8)
    )
    if len(profiles) != 1:
        raise ProviderOriginFocusContextError(
            "focus admission candidate 未唯一绑定 profile revision")
    profile = profiles[0]
    if (profile.provider_kind != candidate.provider_kind
            or profile.provider_identity_u8 != candidate.provider_identity_u8
            or profile.runtime_identity_u8 != candidate.runtime_identity_u8
            or (profile.provider_catalog_identity_u8
                != candidate.provider_catalog_identity_u8)
            or profile.source_record_key != candidate.source_record_key
            or profile.source_ref_stable_key != candidate.source_ref_stable_key
            or profile.source_commitment_u8 != candidate.source_commitment_u8
            or profile.w03_observation_key != candidate.w03_observation_key
            or profile.w04_observation_key != candidate.w04_observation_key
            or profile.w05_observation_key != candidate.w05_observation_key
            or (profile.generation_construction_key
                != candidate.generation_construction_key)
            or profile.proposition_key != anchor.proposition_key
            or profile.predicate_key != anchor.predicate_key
            or profile.relation_kind_code != candidate.relation_kind_code
            or (profile.origin_focus_role_binding_key
                != candidate.reference_role_binding_key)
            or profile.origin_focus_role_key != candidate.reference_role_key
            or (profile.origin_focus_filler_key
                != candidate.reference_filler_key)
            or (profile.origin_focus_occurrence_key
                != candidate.reference_occurrence_key)
            or (profile.target_role_binding_key
                != candidate.answer_role_binding_key)
            or profile.target_role_key != candidate.answer_role_key
            or profile.target_filler_key != candidate.answer_filler_key
            or (profile.target_occurrence_key
                != candidate.answer_occurrence_key)
            or profile.target_start != candidate.answer_start
            or profile.target_end != candidate.answer_end):
        raise ProviderOriginFocusContextError(
            "focus admission candidate 与完整 profile 字段漂移")
    if (candidate.provider_kind != anchor.provider_kind
            or candidate.provider_identity_u8 != anchor.provider_identity_u8
            or candidate.runtime_identity_u8 != anchor.runtime_identity_u8
            or (candidate.provider_catalog_identity_u8
                != anchor.catalog_record_identity_u8)
            or (candidate.provider_result_identity_u8
                != anchor.provider_result_identity_u8)
            or candidate.anchor_identity_u8 != anchor.anchor_identity_u8
            or candidate.source_record_key != anchor.source_record_key
            or candidate.source_ref_stable_key != anchor.source_ref_stable_key
            or candidate.source_commitment_u8 != anchor.source_commitment_u8
            or candidate.w03_observation_key != anchor.w03_observation_key
            or candidate.w04_observation_key != anchor.w04_observation_key
            or candidate.w05_observation_key != anchor.w05_observation_key
            or (candidate.generation_construction_key
                != anchor.generation_construction_key)
            or candidate.relation_kind_code != anchor.relation_kind_code
            or _single_binding(
                anchor,
                binding_key=candidate.answer_role_binding_key,
                role_key=candidate.answer_role_key,
                filler_key=candidate.answer_filler_key,
            ) is None
            or _single_occurrence(
                anchor,
                occurrence_key=candidate.answer_occurrence_key,
                filler_key=candidate.answer_filler_key,
                start=candidate.answer_start,
                end=candidate.answer_end,
            ) is None
            or _anchor_slice(
                anchor,
                start=candidate.answer_start,
                end=candidate.answer_end,
            ) != (candidate.output_scalars, candidate.output_u8)):
        raise ProviderOriginFocusContextError(
            "focus admission candidate 与当前 anchor 字段漂移")
    output_readback = intake_raw_conversation_vector(candidate.output_u8)
    if (not output_readback.accepted
            or output_readback.unicode_scalars != candidate.output_scalars):
        raise ProviderOriginFocusContextError(
            "focus admission output readback 不可接受")
    input_record = result.intake.canonical_record()
    readback_record = output_readback.canonical_record()
    return ProviderOriginFocusAdmissionV1(
        catalog.canonical_record(),
        catalog.catalog_identity_u8,
        form.canonical_record(),
        form.form_identity_u8,
        candidate.canonical_record(),
        candidate.candidate_identity_u8,
        input_record,
        _identity(
            MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1,
            input_record,
            label="focus admission input identity",
        ),
        readback_record,
        _identity(
            MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1,
            readback_record,
            label="focus admission output readback identity",
        ),
        candidate.provider_kind,
        candidate.provider_identity_u8,
        candidate.runtime_identity_u8,
        candidate.provider_catalog_identity_u8,
        candidate.provider_result_identity_u8,
        candidate.anchor_identity_u8,
        candidate.source_record_key,
        candidate.source_ref_stable_key,
        candidate.source_commitment_u8,
        candidate.w03_observation_key,
        candidate.w04_observation_key,
        candidate.w05_observation_key,
        candidate.generation_construction_key,
        anchor.proposition_key,
        anchor.predicate_key,
        candidate.relation_kind_code,
        candidate.reference_role_binding_key,
        candidate.reference_role_key,
        candidate.reference_filler_key,
        candidate.reference_occurrence_key,
        profile.origin_focus_start,
        profile.origin_focus_end,
        candidate.answer_role_binding_key,
        candidate.answer_role_key,
        candidate.answer_filler_key,
        candidate.answer_occurrence_key,
        candidate.answer_start,
        candidate.answer_end,
        candidate.output_scalars,
        candidate.output_u8,
    )


def start_mixed_conversation_focus_context_v3(
        conversation_key: tuple[int, ...],
        ) -> MixedConversationFocusContextStateV3:
    """创建 revision 0 V3 context；首次 append 也必须持有 ``read(0)``。"""
    return MixedConversationFocusContextStateV3(conversation_key, 0, ())


__all__ = [
    "MIXED_FOCUS_CONTEXT_ADMISSION_IDENTITY_DOMAIN_V1",
    "MIXED_FOCUS_CONTEXT_APPEND_ACCEPTED",
    "MIXED_FOCUS_CONTEXT_APPEND_REJECT_ADMISSION",
    "MIXED_FOCUS_CONTEXT_APPEND_REJECT_ANCHOR_NONE",
    "MIXED_FOCUS_CONTEXT_APPEND_REJECT_PARENT_TAIL",
    "MIXED_FOCUS_CONTEXT_APPEND_REJECT_READ_WITNESS",
    "MIXED_FOCUS_CONTEXT_APPEND_RESULT_RECORD_V1",
    "MIXED_FOCUS_CONTEXT_FOCUS_ADMISSION_RECORD_V1",
    "MIXED_FOCUS_CONTEXT_FOLLOWUP_FOCUS_TURN_RECORD_V1",
    "MIXED_FOCUS_CONTEXT_FRAME_TURN_RECORD_V3",
    "MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1",
    "MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1",
    "MIXED_FOCUS_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V3",
    "MIXED_FOCUS_CONTEXT_READ_RECORD_V3",
    "MIXED_FOCUS_CONTEXT_READ_WITNESS_IDENTITY_DOMAIN_V3",
    "MIXED_FOCUS_CONTEXT_READ_WITNESS_RECORD_V3",
    "MIXED_FOCUS_CONTEXT_SCHEMA_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_IDENTITY_DOMAIN_V3",
    "MIXED_FOCUS_CONTEXT_STATE_RECORD_V3",
    "MIXED_FOCUS_CONTEXT_TURN_IDENTITY_DOMAIN_V3",
    "MIXED_FOCUS_CONTEXT_TURN_KIND_FRAME_QA_RUN",
    "MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_FOLLOWUP_FOCUS",
    "MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION",
    "MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN",
    "MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_NONE",
    "MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_ORIGIN_FOLLOWUP_FOCUS",
    "MIXED_FOCUS_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION",
    "FocusContextAppendResultV1",
    "FocusContextReadV3",
    "FocusContextReadWitnessV3",
    "FrameQuestionAnswerTurnV3",
    "MixedConversationFocusContextStateV3",
    "MixedFocusContextTurnV3",
    "ProviderOriginContextTurnV3",
    "ProviderOriginFollowupFocusTurnV1",
    "ProviderOriginFocusAdmissionV1",
    "ProviderOriginFocusContextError",
    "provider_origin_focus_admission_from_followup_result_v1",
    "start_mixed_conversation_focus_context_v3",
]
