"""DLG-RAW-12：同一 provider 锚点内的短暂话语焦点链。

本模块不读取路径、不调用 provider/Frame、不写入 V2 context，也不搜索文本。
它只在一个已经验证的 provider-origin tail 上，以公开课程 profile 的整数结构
把当前焦点从一个 occurrence 移到同一 anchor 内的另一个 occurrence。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
    ProviderOriginAnchorProjectionV1,
    ProviderOriginOccurrenceV1,
    ProviderOriginRoleBindingV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    MixedContextReadV2,
    MixedConversationContextStateV2,
    ProviderOriginContextTurnV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER,
    PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED,
    ProviderOriginFollowupCatalogV1,
    ProviderOriginFollowupFormV1,
    ProviderOriginFollowupProfileV1,
    ProviderOriginFollowupResultV1,
    ProviderOriginReferenceCandidateV1,
    compare_nonnegative_integer_records_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
    DLG_RAW_REJECT_RUNTIME,
    ConversationRawIntake,
    encode_utf8_v1,
)


PROVIDER_ORIGIN_DISCOURSE_FOCUS_RECORD_V1 = 1
PROVIDER_ORIGIN_DISCOURSE_FOCUS_SCHEMA_V1 = 1
PROVIDER_ORIGIN_FOCUS_CHAIN_SELECTOR_V1 = 1
PROVIDER_ORIGIN_FOCUS_CHAIN_SAME_FORM_REJECT_V1 = 1

PROVIDER_ORIGIN_DISCOURSE_FOCUS_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/DISCOURSE-FOCUS/V1")
PROVIDER_ORIGIN_DISCOURSE_FOCUS_SCHEMA_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/DISCOURSE-FOCUS-SCHEMA/V1")

_DIGEST_SIZE = 32


# object-model: exception; interop=DLG-RAW-12
class ProviderOriginFocusChainError(ValueError):
    """焦点记录、同锚点读取或结构缩减不满足冻结合同。"""


def _record(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """核验跨语言 record 只含有限非负严格整数。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ProviderOriginFocusChainError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验焦点中不可为空的稳定整数 key。"""
    return _record(value, label=label, allow_empty=False)


def _digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 raw u8[32] identity，避免 host hex 或对象 identity 参与语义。"""
    result = _record(value, label=label, allow_empty=False)
    if len(result) != _DIGEST_SIZE or any(item > 255 for item in result):
        raise ProviderOriginFocusChainError(f"{label} 必须是 raw u8[32]")
    return result


def _scalar(value: int, *, label: str, minimum: int = 0) -> int:
    """核验冻结协议标量，拒绝 bool 与整数子类。"""
    if type(value) is not int or value < minimum:
        raise ProviderOriginFocusChainError(f"{label} 必须是大于等于 {minimum} 的严格整数")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """用显式 count framing 写入一个可变长度整数段。"""
    record = _record(value, label="focus canonical segment", allow_empty=True)
    result.extend((len(record), *record))


def _read_scalar(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """从已验证 record 读取一个标量，不允许截断。"""
    if cursor >= len(record):
        raise ProviderOriginFocusChainError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = False,
        ) -> tuple[tuple[int, ...], int]:
    """读取 count-framed segment，并拒绝超界或非规范空段。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    _scalar(count, label=f"{label} count")
    if count > len(record) - cursor:
        raise ProviderOriginFocusChainError(f"{label} 长度越界")
    value = _record(
        record[cursor:cursor + count], label=label, allow_empty=allow_empty)
    return value, cursor + count


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """以既有 portable SHA framing 计算一个 raw u8 identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ProviderOriginFocusChainError(f"{label} 无法形成") from error


def _focus_body(value: "ProviderOriginDiscourseFocusV1") -> tuple[int, ...]:
    """写出不含 self identity 的焦点本体 record。"""
    result = [
        PROVIDER_ORIGIN_DISCOURSE_FOCUS_RECORD_V1,
        value.context_revision,
        value.provider_kind,
        value.relation_kind_code,
        value.current_start,
        value.current_end,
    ]
    for item in (
            value.context_digest_u8,
            value.provider_turn_identity_u8,
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
            value.current_role_binding_key,
            value.current_role_key,
            value.current_filler_key,
            value.current_occurrence_key,
            value.source_candidate_identity_u8,
            value.previous_form_identity_u8):
        _pack(result, item)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ProviderOriginDiscourseFocusV1:
    """一个锚定 V2 tail 的瞬态话语焦点，不是新的来源事实。"""

    context_revision: int
    context_digest_u8: tuple[int, ...]
    provider_turn_identity_u8: tuple[int, ...]
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
    current_role_binding_key: tuple[int, ...]
    current_role_key: tuple[int, ...]
    current_filler_key: tuple[int, ...]
    current_occurrence_key: tuple[int, ...]
    current_start: int
    current_end: int
    source_candidate_identity_u8: tuple[int, ...]
    previous_form_identity_u8: tuple[int, ...]
    focus_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 tail、来源链与当前 occurrence，不接受弱化的焦点记录。"""
        revision = _scalar(
            self.context_revision,
            label="discourse focus context revision",
            minimum=1,
        )
        if (type(self.provider_kind) is not int
                or self.provider_kind != PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05):
            raise ProviderOriginFocusChainError(
                "discourse focus provider kind 未注册")
        relation = _scalar(
            self.relation_kind_code,
            label="discourse focus relation kind",
            minimum=1,
        )
        for name in (
                "context_digest_u8", "provider_turn_identity_u8",
                "provider_identity_u8", "runtime_identity_u8",
                "provider_catalog_identity_u8", "provider_result_identity_u8",
                "anchor_identity_u8", "source_commitment_u8",
                "source_candidate_identity_u8", "previous_form_identity_u8"):
            object.__setattr__(self, name, _digest(
                getattr(self, name), label=f"discourse focus {name}"))
        for name in (
                "source_record_key", "source_ref_stable_key",
                "w03_observation_key", "w04_observation_key",
                "w05_observation_key", "generation_construction_key",
                "proposition_key", "predicate_key", "current_role_binding_key",
                "current_role_key", "current_filler_key", "current_occurrence_key"):
            object.__setattr__(self, name, _key(
                getattr(self, name), label=f"discourse focus {name}"))
        start = _scalar(
            self.current_start,
            label="discourse focus current start",
        )
        end = _scalar(
            self.current_end,
            label="discourse focus current end",
            minimum=1,
        )
        if end <= start:
            raise ProviderOriginFocusChainError(
                "discourse focus current occurrence span 非法")
        object.__setattr__(self, "context_revision", revision)
        object.__setattr__(self, "relation_kind_code", relation)
        object.__setattr__(self, "current_start", start)
        object.__setattr__(self, "current_end", end)
        expected = _identity(
            PROVIDER_ORIGIN_DISCOURSE_FOCUS_IDENTITY_DOMAIN_V1,
            _focus_body(self),
            label="discourse focus identity",
        )
        supplied = self.focus_identity_u8
        if supplied and _digest(
                supplied, label="discourse focus identity") != expected:
            raise ProviderOriginFocusChainError("discourse focus identity 漂移")
        object.__setattr__(self, "focus_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出可由整数实现重建的完整焦点记录。"""
        result = list(_focus_body(self))
        _pack(result, self.focus_identity_u8)
        return tuple(result)


def restore_provider_origin_discourse_focus_v1(
        record: tuple[int, ...],
        ) -> ProviderOriginDiscourseFocusV1:
    """严格恢复 V1 focus record，不猜测未来字段或旧版本。"""
    record = _record(record, label="discourse focus record", allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="discourse focus record version")
    if version != PROVIDER_ORIGIN_DISCOURSE_FOCUS_RECORD_V1:
        raise ProviderOriginFocusChainError("discourse focus record version 未注册")
    revision, cursor = _read_scalar(
        record, cursor, label="discourse focus context revision")
    provider_kind, cursor = _read_scalar(
        record, cursor, label="discourse focus provider kind")
    relation, cursor = _read_scalar(
        record, cursor, label="discourse focus relation")
    start, cursor = _read_scalar(
        record, cursor, label="discourse focus current start")
    end, cursor = _read_scalar(
        record, cursor, label="discourse focus current end")
    values: list[tuple[int, ...]] = []
    labels = (
        "context digest", "provider turn identity", "provider identity",
        "runtime identity", "provider catalog identity", "provider result identity",
        "anchor identity", "source record key", "source ref stable key",
        "source commitment", "W03 observation key", "W04 observation key",
        "W05 observation key", "generation construction key", "proposition key",
        "predicate key", "current role binding key", "current role key",
        "current filler key", "current occurrence key", "source candidate identity",
        "previous form identity", "focus identity")
    for label in labels:
        value, cursor = _read_segment(
            record, cursor, label=f"discourse focus {label}")
        values.append(value)
    if cursor != len(record):
        raise ProviderOriginFocusChainError("discourse focus record 含尾随整数")
    return ProviderOriginDiscourseFocusV1(
        revision,
        values[0], values[1], provider_kind, values[2], values[3], values[4],
        values[5], values[6], values[7], values[8], values[9], values[10],
        values[11], values[12], values[13], values[14], values[15], relation,
        values[16], values[17], values[18], values[19], start, end, values[20],
        values[21], values[22],
    )


def _single_binding(
        anchor: ProviderOriginAnchorProjectionV1,
        *,
        binding_key: tuple[int, ...],
        role_key: tuple[int, ...],
        filler_key: tuple[int, ...],
        ) -> ProviderOriginRoleBindingV1 | None:
    """在同一 anchor 的固定 binding 序列中读取一个唯一候选。"""
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
    """在同一 anchor 的固定 occurrence 序列中读取一个唯一候选。"""
    found = tuple(
        item for item in anchor.ordered_occurrences
        if (item.occurrence_key == occurrence_key
            and item.semantic_object_key == filler_key
            and item.start == start
            and item.end == end))
    return found[0] if len(found) == 1 else None


def _slice_anchor_occurrence(
        anchor: ProviderOriginAnchorProjectionV1,
        *,
        start: int,
        end: int,
        ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """只从已验证 anchor 的 scalar span 构造并回读输出 bytes。"""
    if start < 0 or end <= start or end > len(anchor.output_scalars):
        return None
    scalars = anchor.output_scalars[start:end]
    try:
        output = encode_utf8_v1(scalars)
    except (TypeError, ValueError):
        return None
    byte_start = len(encode_utf8_v1(anchor.output_scalars[:start]))
    byte_end = byte_start + len(output)
    if (byte_end > len(anchor.output_u8)
            or anchor.output_u8[byte_start:byte_end] != output):
        return None
    return scalars, output


def _focus_matches_anchor(
        focus: ProviderOriginDiscourseFocusV1,
        context: MixedConversationContextStateV2,
        read: MixedContextReadV2,
        ) -> ProviderOriginAnchorProjectionV1 | None:
    """验证 focus 仍精确绑定当前 context 的唯一 provider tail。"""
    try:
        witness = read.witness
        if (witness.requested_limit != 1
                or witness.conversation_key != context.conversation_key
                or witness.revision != focus.context_revision
                or witness.snapshot_digest_u8 != focus.context_digest_u8
                or witness.revision != context.revision
                or witness.snapshot_digest_u8 != context.digest()
                or len(read.turns) != 1
                or not context.turns
                or type(read.turns[0]) is not ProviderOriginContextTurnV1
                or read.turns[0].turn_kind
                != MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION
                or (read.turns[0].canonical_record()
                    != context.turns[-1].canonical_record())):
            return None
        tail = read.turns[0]
        anchor = tail.anchor_projection
        if (not anchor.accepted
                or tail.turn_identity_u8 != focus.provider_turn_identity_u8
                or anchor.provider_kind != focus.provider_kind
                or anchor.provider_identity_u8 != focus.provider_identity_u8
                or anchor.runtime_identity_u8 != focus.runtime_identity_u8
                or anchor.catalog_record_identity_u8
                != focus.provider_catalog_identity_u8
                or anchor.provider_result_identity_u8
                != focus.provider_result_identity_u8
                or anchor.anchor_identity_u8 != focus.anchor_identity_u8
                or anchor.source_record_key != focus.source_record_key
                or anchor.source_ref_stable_key != focus.source_ref_stable_key
                or anchor.source_commitment_u8 != focus.source_commitment_u8
                or anchor.w03_observation_key != focus.w03_observation_key
                or anchor.w04_observation_key != focus.w04_observation_key
                or anchor.w05_observation_key != focus.w05_observation_key
                or anchor.generation_construction_key
                != focus.generation_construction_key
                or anchor.proposition_key != focus.proposition_key
                or anchor.predicate_key != focus.predicate_key
                or anchor.relation_kind_code != focus.relation_kind_code
                or _single_binding(
                    anchor,
                    binding_key=focus.current_role_binding_key,
                    role_key=focus.current_role_key,
                    filler_key=focus.current_filler_key,
                ) is None
                or _single_occurrence(
                    anchor,
                    occurrence_key=focus.current_occurrence_key,
                    filler_key=focus.current_filler_key,
                    start=focus.current_start,
                    end=focus.current_end,
                ) is None):
            return None
        return anchor
    except (ProviderOriginFocusChainError, TypeError, ValueError):
        return None


def focus_from_provider_origin_followup_v1(
        context: MixedConversationContextStateV2,
        result: ProviderOriginFollowupResultV1,
        ) -> ProviderOriginDiscourseFocusV1:
    """把成功的 11C/11D 或焦点链结果投影为下一轮可消费的焦点。"""
    if type(context) is not MixedConversationContextStateV2:
        raise TypeError("discourse focus context 类型错误")
    if type(result) is not ProviderOriginFollowupResultV1:
        raise TypeError("discourse focus follow-up result 类型错误")
    if (not result.accepted or result.candidate is None or result.form is None
            or result.context_read is None):
        raise ProviderOriginFocusChainError(
            "discourse focus 只能从成功 follow-up 投影")
    read = result.context_read
    tail = _focus_matches_result_anchor(context, result, read)
    if tail is None:
        raise ProviderOriginFocusChainError(
            "discourse focus result/context/anchor 闭环漂移")
    candidate = result.candidate
    anchor = tail.anchor_projection
    return ProviderOriginDiscourseFocusV1(
        read.witness.revision,
        read.witness.snapshot_digest_u8,
        tail.turn_identity_u8,
        anchor.provider_kind,
        anchor.provider_identity_u8,
        anchor.runtime_identity_u8,
        anchor.catalog_record_identity_u8,
        anchor.provider_result_identity_u8,
        anchor.anchor_identity_u8,
        anchor.source_record_key,
        anchor.source_ref_stable_key,
        anchor.source_commitment_u8,
        anchor.w03_observation_key,
        anchor.w04_observation_key,
        anchor.w05_observation_key,
        anchor.generation_construction_key,
        anchor.proposition_key,
        anchor.predicate_key,
        anchor.relation_kind_code,
        candidate.answer_role_binding_key,
        candidate.answer_role_key,
        candidate.answer_filler_key,
        candidate.answer_occurrence_key,
        candidate.answer_start,
        candidate.answer_end,
        candidate.candidate_identity_u8,
        result.form.form_identity_u8,
    )


def _focus_matches_result_anchor(
        context: MixedConversationContextStateV2,
        result: ProviderOriginFollowupResultV1,
        read: MixedContextReadV2,
        ) -> ProviderOriginContextTurnV1 | None:
    """验证 result candidate 的 answer 确为当前 provider tail 中的 occurrence。"""
    candidate = result.candidate
    if candidate is None:
        return None
    try:
        witness = read.witness
        if (witness.requested_limit != 1
                or witness.conversation_key != context.conversation_key
                or witness.revision != context.revision
                or witness.snapshot_digest_u8 != context.digest()
                or len(read.turns) != 1
                or not context.turns
                or type(read.turns[0]) is not ProviderOriginContextTurnV1):
            return None
        tail = read.turns[0]
        anchor = tail.anchor_projection
        if ((tail.canonical_record() != context.turns[-1].canonical_record())
                or not anchor.accepted
                or candidate.catalog_identity_u8 != result.catalog_identity_u8
                or candidate.provider_kind != anchor.provider_kind
                or candidate.provider_identity_u8 != anchor.provider_identity_u8
                or candidate.runtime_identity_u8 != anchor.runtime_identity_u8
                or candidate.provider_catalog_identity_u8
                != anchor.catalog_record_identity_u8
                or candidate.provider_result_identity_u8
                != anchor.provider_result_identity_u8
                or candidate.anchor_identity_u8 != anchor.anchor_identity_u8
                or candidate.source_record_key != anchor.source_record_key
                or candidate.source_ref_stable_key != anchor.source_ref_stable_key
                or candidate.source_commitment_u8 != anchor.source_commitment_u8
                or candidate.w03_observation_key != anchor.w03_observation_key
                or candidate.w04_observation_key != anchor.w04_observation_key
                or candidate.w05_observation_key != anchor.w05_observation_key
                or candidate.generation_construction_key
                != anchor.generation_construction_key
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
                ) is None):
            return None
        sliced = _slice_anchor_occurrence(
            anchor, start=candidate.answer_start, end=candidate.answer_end)
        if sliced is None or sliced != (candidate.output_scalars, candidate.output_u8):
            return None
        return tail
    except (ProviderOriginFocusChainError, TypeError, ValueError):
        return None


def _profile_matches_focus(
        profile: ProviderOriginFollowupProfileV1,
        form: ProviderOriginFollowupFormV1,
        focus: ProviderOriginDiscourseFocusV1,
        ) -> bool:
    """只比较课程明示的同锚点链字段，不以表层词或 role ordinal 推导。"""
    return (
        profile.form_identity_u8 == form.form_identity_u8
        and profile.form_identity_u8 != focus.previous_form_identity_u8
        and profile.provider_kind == focus.provider_kind
        and profile.provider_identity_u8 == focus.provider_identity_u8
        and profile.runtime_identity_u8 == focus.runtime_identity_u8
        and profile.provider_catalog_identity_u8 == focus.provider_catalog_identity_u8
        and profile.source_record_key == focus.source_record_key
        and profile.source_ref_stable_key == focus.source_ref_stable_key
        and profile.source_commitment_u8 == focus.source_commitment_u8
        and profile.w03_observation_key == focus.w03_observation_key
        and profile.w04_observation_key == focus.w04_observation_key
        and profile.w05_observation_key == focus.w05_observation_key
        and profile.generation_construction_key == focus.generation_construction_key
        and profile.proposition_key == focus.proposition_key
        and profile.predicate_key == focus.predicate_key
        and profile.relation_kind_code == focus.relation_kind_code
        and profile.origin_focus_role_binding_key == focus.current_role_binding_key
        and profile.origin_focus_role_key == focus.current_role_key
        and profile.origin_focus_filler_key == focus.current_filler_key
        and profile.origin_focus_occurrence_key == focus.current_occurrence_key
        and profile.origin_focus_start == focus.current_start
        and profile.origin_focus_end == focus.current_end
    )


def _candidate_from_focus_profile(
        catalog: ProviderOriginFollowupCatalogV1,
        form: ProviderOriginFollowupFormV1,
        profile: ProviderOriginFollowupProfileV1,
        focus: ProviderOriginDiscourseFocusV1,
        anchor: ProviderOriginAnchorProjectionV1,
        ) -> ProviderOriginReferenceCandidateV1 | None:
    """以 active focus 为 reference，构造同 anchor target occurrence candidate。"""
    if not _profile_matches_focus(profile, form, focus):
        return None
    if (_single_binding(
            anchor,
            binding_key=profile.origin_focus_role_binding_key,
            role_key=profile.origin_focus_role_key,
            filler_key=profile.origin_focus_filler_key,
    ) is None or _single_occurrence(
            anchor,
            occurrence_key=profile.origin_focus_occurrence_key,
            filler_key=profile.origin_focus_filler_key,
            start=profile.origin_focus_start,
            end=profile.origin_focus_end,
    ) is None or _single_binding(
            anchor,
            binding_key=profile.target_role_binding_key,
            role_key=profile.target_role_key,
            filler_key=profile.target_filler_key,
    ) is None or _single_occurrence(
            anchor,
            occurrence_key=profile.target_occurrence_key,
            filler_key=profile.target_filler_key,
            start=profile.target_start,
            end=profile.target_end,
    ) is None):
        return None
    sliced = _slice_anchor_occurrence(
        anchor, start=profile.target_start, end=profile.target_end)
    if sliced is None:
        raise ProviderOriginFocusChainError(
            "focus chain target 不是 anchor 的精确 occurrence slice")
    scalars, output = sliced
    if len(output) > form.output_max_bytes:
        raise ProviderOriginFocusChainError("focus chain target 超出 form output 预算")
    return ProviderOriginReferenceCandidateV1(
        catalog.catalog_identity_u8,
        form.form_identity_u8,
        profile.profile_identity_u8,
        profile.profile_revision,
        anchor.provider_kind,
        anchor.provider_identity_u8,
        anchor.runtime_identity_u8,
        anchor.catalog_record_identity_u8,
        anchor.provider_result_identity_u8,
        anchor.source_record_key,
        anchor.source_ref_stable_key,
        anchor.source_commitment_u8,
        anchor.w03_observation_key,
        anchor.w04_observation_key,
        anchor.w05_observation_key,
        anchor.generation_construction_key,
        anchor.relation_kind_code,
        anchor.anchor_identity_u8,
        profile.origin_focus_role_binding_key,
        profile.origin_focus_role_key,
        profile.origin_focus_filler_key,
        profile.origin_focus_occurrence_key,
        profile.target_role_binding_key,
        profile.target_role_key,
        profile.target_filler_key,
        profile.target_occurrence_key,
        profile.target_start,
        profile.target_end,
        scalars,
        output,
    )


def _candidate_outcome_record(
        value: ProviderOriginReferenceCandidateV1,
        ) -> tuple[int, ...]:
    """写出忽略课程证据副本后的同锚点结构结果，用于 0/1/many 归约。"""
    result = [
        value.provider_kind,
        value.relation_kind_code,
        value.answer_start,
        value.answer_end,
    ]
    for item in (
            value.catalog_identity_u8,
            value.form_identity_u8,
            value.provider_identity_u8,
            value.runtime_identity_u8,
            value.provider_catalog_identity_u8,
            value.provider_result_identity_u8,
            value.source_record_key,
            value.source_ref_stable_key,
            value.source_commitment_u8,
            value.w03_observation_key,
            value.w04_observation_key,
            value.w05_observation_key,
            value.generation_construction_key,
            value.anchor_identity_u8,
            value.reference_role_binding_key,
            value.reference_role_key,
            value.reference_filler_key,
            value.reference_occurrence_key,
            value.answer_role_binding_key,
            value.answer_role_key,
            value.answer_filler_key,
            value.answer_occurrence_key,
            value.output_scalars,
            value.output_u8):
        _pack(result, item)
    return tuple(result)


def _reduce_same_outcome_candidates(
        candidates: tuple[ProviderOriginReferenceCandidateV1, ...],
        ) -> tuple[ProviderOriginReferenceCandidateV1, ...]:
    """将等价课程证据归为一个结构 outcome，保留整数序最小的可审计 carrier。"""
    reduced: list[ProviderOriginReferenceCandidateV1] = []
    for candidate in candidates:
        outcome = _candidate_outcome_record(candidate)
        matching_index: int | None = None
        for index, existing in enumerate(reduced):
            if compare_nonnegative_integer_records_v1(
                    outcome,
                    _candidate_outcome_record(existing),
                    label="focus chain structural outcome equality") == 0:
                matching_index = index
                break
        if matching_index is None:
            reduced.append(candidate)
        elif compare_nonnegative_integer_records_v1(
                candidate.canonical_record(),
                reduced[matching_index].canonical_record(),
                label="focus chain equivalent evidence order") < 0:
            reduced[matching_index] = candidate
    return tuple(reduced)


def _reject(
        code: int,
        intake: ConversationRawIntake,
        catalog: ProviderOriginFollowupCatalogV1,
        form: ProviderOriginFollowupFormV1,
        *,
        forms: int,
        read: MixedContextReadV2 | None,
        candidates: int,
        ) -> ProviderOriginFollowupResultV1:
    """构造 active-focus form 已处理后的零输出、零写入拒绝 carrier。"""
    return ProviderOriginFollowupResultV1(
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED,
        code,
        intake,
        catalog.catalog_identity_u8,
        forms,
        candidates,
        form,
        read,
    )


def run_provider_origin_focus_followup_v1(
        intake: ConversationRawIntake,
        context: MixedConversationContextStateV2,
        catalog: ProviderOriginFollowupCatalogV1,
        focus: ProviderOriginDiscourseFocusV1,
        ) -> ProviderOriginFollowupResultV1 | None:
    """处理 active focus 的已学习 form；无 form 时返回 ``None`` 允许 V4 fallback。

    已命中 form 后，这个函数必定返回 handled result，绝不把无候选情况交回
    11C/11D 的 base-anchor reducer。
    """
    if type(intake) is not ConversationRawIntake:
        raise TypeError("focus chain intake 类型错误")
    if type(context) is not MixedConversationContextStateV2:
        raise TypeError("focus chain context 类型错误")
    if type(catalog) is not ProviderOriginFollowupCatalogV1:
        raise TypeError("focus chain catalog 类型错误")
    if type(focus) is not ProviderOriginDiscourseFocusV1:
        raise TypeError("focus chain focus 类型错误")
    if not intake.accepted:
        return None
    forms = catalog.matching_forms(intake.unicode_scalars)
    if not forms:
        return None
    form = forms[0]
    if len(forms) != 1:
        return _reject(
            DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
            intake,
            catalog,
            form,
            forms=len(forms),
            read=None,
            candidates=0,
        )
    try:
        read = context.read(1)
    except (ProviderOriginFocusChainError, TypeError, ValueError):
        return _reject(
            DLG_RAW_REJECT_RUNTIME,
            intake,
            catalog,
            form,
            forms=1,
            read=None,
            candidates=0,
        )
    anchor = _focus_matches_anchor(focus, context, read)
    if anchor is None:
        return _reject(
            DLG_RAW_REJECT_RUNTIME,
            intake,
            catalog,
            form,
            forms=1,
            read=read,
            candidates=0,
        )
    if form.form_identity_u8 == focus.previous_form_identity_u8:
        return _reject(
            DLG_RAW_REJECT_CONTEXT,
            intake,
            catalog,
            form,
            forms=1,
            read=read,
            candidates=0,
        )
    candidates: list[ProviderOriginReferenceCandidateV1] = []
    try:
        for profile in catalog.profiles_for_form(form):
            candidate = _candidate_from_focus_profile(
                catalog, form, profile, focus, anchor)
            if candidate is not None:
                candidates.append(candidate)
    except (ProviderOriginFocusChainError, TypeError, ValueError):
        return _reject(
            DLG_RAW_REJECT_RUNTIME,
            intake,
            catalog,
            form,
            forms=1,
            read=read,
            candidates=0,
        )
    reduced = _reduce_same_outcome_candidates(tuple(candidates))
    if not reduced:
        return _reject(
            DLG_RAW_REJECT_CONTEXT,
            intake,
            catalog,
            form,
            forms=1,
            read=read,
            candidates=0,
        )
    if len(reduced) != 1:
        return _reject(
            DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
            intake,
            catalog,
            form,
            forms=1,
            read=read,
            candidates=len(reduced),
        )
    selected = reduced[0]
    return ProviderOriginFollowupResultV1(
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER,
        DLG_RAW_ACCEPT,
        intake,
        catalog.catalog_identity_u8,
        1,
        1,
        form,
        read,
        selected,
        selected.output_scalars,
        selected.output_u8,
    )


def provider_origin_discourse_focus_schema_record_v1() -> tuple[int, ...]:
    """导出 V5 runtime binding 所需的 focus record、selector 和拒绝语义。"""
    return (
        PROVIDER_ORIGIN_DISCOURSE_FOCUS_SCHEMA_V1,
        PROVIDER_ORIGIN_DISCOURSE_FOCUS_RECORD_V1,
        PROVIDER_ORIGIN_FOCUS_CHAIN_SELECTOR_V1,
        PROVIDER_ORIGIN_FOCUS_CHAIN_SAME_FORM_REJECT_V1,
        DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
        DLG_RAW_REJECT_CONTEXT,
        DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
        DLG_RAW_REJECT_RUNTIME,
        _DIGEST_SIZE,
    )


def provider_origin_discourse_focus_schema_identity_v1() -> tuple[int, ...]:
    """返回 V5 binding 可锁定的 raw focus schema identity。"""
    return _identity(
        PROVIDER_ORIGIN_DISCOURSE_FOCUS_SCHEMA_IDENTITY_DOMAIN_V1,
        provider_origin_discourse_focus_schema_record_v1(),
        label="discourse focus schema identity",
    )


__all__ = [
    "PROVIDER_ORIGIN_DISCOURSE_FOCUS_RECORD_V1",
    "PROVIDER_ORIGIN_DISCOURSE_FOCUS_SCHEMA_V1",
    "PROVIDER_ORIGIN_FOCUS_CHAIN_SAME_FORM_REJECT_V1",
    "PROVIDER_ORIGIN_FOCUS_CHAIN_SELECTOR_V1",
    "ProviderOriginDiscourseFocusV1",
    "ProviderOriginFocusChainError",
    "focus_from_provider_origin_followup_v1",
    "provider_origin_discourse_focus_schema_identity_v1",
    "provider_origin_discourse_focus_schema_record_v1",
    "restore_provider_origin_discourse_focus_v1",
    "run_provider_origin_focus_followup_v1",
]
