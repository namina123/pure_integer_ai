"""DLG-RAW-11C：来源内 provider 追问的纯整数候选缩减。

本模块不读取路径、不调用 provider、不运行 Frame runtime，也不保存会话。它只
消费已经写入 V2 mixed context 的相邻 provider-origin anchor，以及由公开课程
适配器构造的有限整数 catalog。Python dataclass 只是当前实现的结构体便利；每个
可观察值都能降解为有序整数 record 与 raw u8 identity。
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
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    MixedContextReadV2,
    ProviderOriginContextTurnV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
    DLG_RAW_REJECT_RUNTIME,
    ConversationRawIntake,
    encode_utf8_v1,
)


PROVIDER_ORIGIN_FOLLOWUP_CATALOG_SCHEMA_V1 = 1
PROVIDER_ORIGIN_FOLLOWUP_LEXICAL_EVIDENCE_RECORD_V1 = 1
PROVIDER_ORIGIN_FOLLOWUP_FORM_RECORD_V1 = 1
PROVIDER_ORIGIN_FOLLOWUP_PROFILE_RECORD_V1 = 1
PROVIDER_ORIGIN_FOLLOWUP_CATALOG_RECORD_V1 = 1
PROVIDER_ORIGIN_REFERENCE_CANDIDATE_RECORD_V1 = 1
PROVIDER_ORIGIN_FOLLOWUP_RESULT_RECORD_V1 = 1
PROVIDER_ORIGIN_FOLLOWUP_RECORD_ORDER_LEXICOGRAPHIC_V1 = 1

PROVIDER_ORIGIN_FOLLOWUP_SELECTOR_ORIGIN_FOCUS_TO_CONTRAST_FOCUS_V1 = 1
PROVIDER_ORIGIN_FOLLOWUP_OUTPUT_ANCHOR_OCCURRENCE_SLICE_V1 = 1

PROVIDER_ORIGIN_FOLLOWUP_STATUS_NOT_APPLICABLE = 0
PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER = 1
PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED = 2

_CANDIDATE_BUILD_OUTCOME_NONE = 0
_CANDIDATE_BUILD_OUTCOME_CANDIDATE = 1
_CANDIDATE_BUILD_OUTCOME_RUNTIME = 2

PROVIDER_ORIGIN_FOLLOWUP_LEXICAL_EVIDENCE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11C/FOLLOWUP-LEXICAL-EVIDENCE/V1")
PROVIDER_ORIGIN_FOLLOWUP_FORM_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11C/FOLLOWUP-FORM/V1")
PROVIDER_ORIGIN_FOLLOWUP_PROFILE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11C/FOLLOWUP-PROFILE/V1")
PROVIDER_ORIGIN_FOLLOWUP_CATALOG_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11C/FOLLOWUP-CATALOG/V1")
PROVIDER_ORIGIN_REFERENCE_CANDIDATE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11C/REFERENCE-CANDIDATE/V1")

_DIGEST_SIZE = 32
_MAX_OUTPUT_BYTES = 4096
_FOLLOWUP_STATUSES = frozenset({
    PROVIDER_ORIGIN_FOLLOWUP_STATUS_NOT_APPLICABLE,
    PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER,
    PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED,
})
_FOLLOWUP_RESULT_CODES = frozenset({
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_RUNTIME,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
})


# object-model: exception; interop=DLG-RAW-11C
class ProviderOriginFollowupError(ValueError):
    """来源内追问的 catalog、candidate 或 result record 未闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长整数段编码为显式 count 与原始有序内容。"""
    result.extend((len(value), *value))


def _vector(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证不依赖宿主 collection 语义的严格非负整数 vector。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ProviderOriginFollowupError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证会话、来源、结构或 profile 的非空稳定整数 key。"""
    return _vector(value, label=label, allow_empty=False)


def _u8(value: tuple[int, ...], *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """验证 raw u8 vector，不让 Python bytes 成为核心语义。"""
    result = _vector(value, label=label, allow_empty=allow_empty)
    if any(item > 255 for item in result):
        raise ProviderOriginFollowupError(f"{label} 含非 u8 整数")
    return result


def _digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证 protocol identity 固定为 raw u8[32]。"""
    result = _u8(value, label=label, allow_empty=False)
    if len(result) != _DIGEST_SIZE:
        raise ProviderOriginFollowupError(f"{label} 必须是 raw u8[32]")
    return result


def _scalar(value: int, *, label: str, minimum: int = 0) -> int:
    """验证 strict nonnegative scalar，拒绝 bool 与整数子类。"""
    if type(value) is not int or value < minimum:
        raise ProviderOriginFollowupError(f"{label} 必须是不小于 {minimum} 的严格整数")
    return value


def _unicode_scalars(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证可由 UTF-8 v1 编码的有限 Unicode scalar 序。"""
    result = _vector(value, label=label, allow_empty=allow_empty)
    if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in result):
        raise ProviderOriginFollowupError(f"{label} 含非法 Unicode scalar")
    return result


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """按冻结 portable SHA framing 形成 raw u8[32] identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ProviderOriginFollowupError(f"{label} 无法形成") from error


def compare_nonnegative_integer_records_v1(
        left: tuple[int, ...],
        right: tuple[int, ...],
        *,
        label: str,
        ) -> int:
    """按 versioned lexicographic rule 比较两个有限整数 record。

    逐项比较第一个不同整数；若公共前缀完全相同，短记录排在前。调用方不得使用
    Python tuple 的默认排序、hash 或对象比较来决定 catalog 的可观察顺序。
    """
    first = _vector(left, label=f"{label} left", allow_empty=True)
    second = _vector(right, label=f"{label} right", allow_empty=True)
    limit = len(first) if len(first) < len(second) else len(second)
    for index in range(limit):
        if first[index] < second[index]:
            return -1
        if first[index] > second[index]:
            return 1
    if len(first) < len(second):
        return -1
    if len(first) > len(second):
        return 1
    return 0


def order_items_by_nonnegative_integer_record_v1(
        values: tuple[object, ...],
        *,
        key,
        label: str,
        ) -> tuple[object, ...]:
    """用显式 insertion rule 排序有限项目，并拒绝重复 record。

    该函数只在 catalog 构造期使用；排序规则本身是协议的一部分，故不能依赖宿主
    collection 的默认 order。相等 record 在插入时立即拒绝，避免 hash/set 参与。
    """
    if type(values) is not tuple:
        raise ProviderOriginFollowupError(f"{label} 必须是 tuple")
    ordered_items: list[object] = []
    ordered_records: list[tuple[int, ...]] = []
    for index, item in enumerate(values):
        record = _vector(
            key(item),
            label=f"{label}[{index}] canonical record",
            allow_empty=False,
        )
        insertion = 0
        while insertion < len(ordered_records):
            comparison = compare_nonnegative_integer_records_v1(
                record,
                ordered_records[insertion],
                label=label,
            )
            if comparison == 0:
                raise ProviderOriginFollowupError(
                    f"{label} 含重复 canonical record")
            if comparison < 0:
                break
            insertion += 1
        ordered_items.insert(insertion, item)
        ordered_records.insert(insertion, record)
    return tuple(ordered_items)


def _sorted_unique(
        values: tuple[object, ...],
        *,
        key,
        label: str,
        ) -> tuple[object, ...]:
    """验证 caller 已按显式 canonical integer rule 排序且无重复。"""
    if type(values) is not tuple:
        raise ProviderOriginFollowupError(f"{label} 必须是 tuple")
    previous: tuple[int, ...] | None = None
    for index, item in enumerate(values):
        record = _vector(
            key(item),
            label=f"{label}[{index}] canonical record",
            allow_empty=False,
        )
        if previous is not None:
            comparison = compare_nonnegative_integer_records_v1(
                previous,
                record,
                label=label,
            )
            if comparison == 0:
                raise ProviderOriginFollowupError(
                    f"{label} 含重复 canonical record")
            if comparison > 0:
                raise ProviderOriginFollowupError(
                    f"{label} 未按 canonical record 排序")
        previous = record
    return values


def _lexical_evidence_body(
        value: "ProviderOriginFollowupLexicalEvidenceV1",
        ) -> tuple[int, ...]:
    """写出不含 self identity 的 lexical evidence record。"""
    result = [PROVIDER_ORIGIN_FOLLOWUP_LEXICAL_EVIDENCE_RECORD_V1]
    for item in (
            value.logical_key_u8,
            value.raw_sha256_u8,
            value.license_id_u8,
            value.attribution_u8,
            value.span_u8,
            value.span_scalars):
        _pack(result, item)
    result.extend((value.span_start, value.span_end))
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11C
@dataclass(frozen=True, slots=True)
class ProviderOriginFollowupLexicalEvidenceV1:
    """一份来源化 follow-up 表层证据，只携带 raw span 与 hash。"""

    logical_key_u8: tuple[int, ...]
    raw_sha256_u8: tuple[int, ...]
    license_id_u8: tuple[int, ...]
    attribution_u8: tuple[int, ...]
    span_start: int
    span_end: int
    span_u8: tuple[int, ...]
    span_scalars: tuple[int, ...]
    evidence_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """固定证据 source、span、UTF-8 readback 与 canonical identity。"""
        logical = _key(self.logical_key_u8, label="follow-up lexical logical key")
        if any(item < 0x21 or item > 0x7E for item in logical):
            raise ProviderOriginFollowupError("follow-up lexical logical key 非 ASCII")
        digest = _digest(self.raw_sha256_u8, label="follow-up lexical raw SHA-256")
        license_id = _u8(
            self.license_id_u8,
            label="follow-up lexical license id",
            allow_empty=False,
        )
        if license_id != tuple(b"CC0-1.0"):
            raise ProviderOriginFollowupError(
                "follow-up lexical license 必须是 CC0-1.0")
        attribution = _u8(
            self.attribution_u8,
            label="follow-up lexical attribution",
            allow_empty=False,
        )
        start = _scalar(self.span_start, label="follow-up lexical span start")
        end = _scalar(self.span_end, label="follow-up lexical span end", minimum=1)
        if end <= start:
            raise ProviderOriginFollowupError("follow-up lexical span 边界非法")
        span_u8 = _u8(
            self.span_u8,
            label="follow-up lexical span u8",
            allow_empty=False,
        )
        scalars = _unicode_scalars(
            self.span_scalars,
            label="follow-up lexical span scalars",
            allow_empty=False,
        )
        if encode_utf8_v1(scalars) != span_u8:
            raise ProviderOriginFollowupError("follow-up lexical UTF-8 span 漂移")
        object.__setattr__(self, "logical_key_u8", logical)
        object.__setattr__(self, "raw_sha256_u8", digest)
        object.__setattr__(self, "license_id_u8", license_id)
        object.__setattr__(self, "attribution_u8", attribution)
        object.__setattr__(self, "span_start", start)
        object.__setattr__(self, "span_end", end)
        object.__setattr__(self, "span_u8", span_u8)
        object.__setattr__(self, "span_scalars", scalars)
        expected = _identity(
            PROVIDER_ORIGIN_FOLLOWUP_LEXICAL_EVIDENCE_IDENTITY_DOMAIN_V1,
            _lexical_evidence_body(self),
            label="follow-up lexical evidence identity",
        )
        supplied = self.evidence_identity_u8
        if supplied and _digest(
                supplied, label="follow-up lexical evidence identity") != expected:
            raise ProviderOriginFollowupError(
                "follow-up lexical evidence identity 漂移")
        object.__setattr__(self, "evidence_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出不依赖路径对象或文本对象的完整 evidence record。"""
        result = list(_lexical_evidence_body(self))
        _pack(result, self.evidence_identity_u8)
        return tuple(result)


def _form_body(value: "ProviderOriginFollowupFormV1") -> tuple[int, ...]:
    """写出不含 self identity 的 follow-up form record。"""
    result = [
        PROVIDER_ORIGIN_FOLLOWUP_FORM_RECORD_V1,
        value.selector_code,
        value.output_mode,
        value.output_max_bytes,
    ]
    _pack(result, value.input_scalars)
    result.append(len(value.lexical_evidence))
    for evidence in value.lexical_evidence:
        _pack(result, evidence.canonical_record())
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11C
@dataclass(frozen=True, slots=True)
class ProviderOriginFollowupFormV1:
    """已学习 follow-up 表层构式，与具体来源 profile 分离。"""

    input_scalars: tuple[int, ...]
    lexical_evidence: tuple[ProviderOriginFollowupLexicalEvidenceV1, ...]
    selector_code: int = (
        PROVIDER_ORIGIN_FOLLOWUP_SELECTOR_ORIGIN_FOCUS_TO_CONTRAST_FOCUS_V1)
    output_mode: int = PROVIDER_ORIGIN_FOLLOWUP_OUTPUT_ANCHOR_OCCURRENCE_SLICE_V1
    output_max_bytes: int = _MAX_OUTPUT_BYTES
    form_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """要求所有来源 span 回读同一构式，且不靠 Python 去重或排序。"""
        scalars = _unicode_scalars(
            self.input_scalars,
            label="follow-up form input scalars",
            allow_empty=False,
        )
        evidence = self.lexical_evidence
        if (type(evidence) is not tuple or len(evidence) < 2
                or any(type(item) is not ProviderOriginFollowupLexicalEvidenceV1
                       for item in evidence)):
            raise ProviderOriginFollowupError(
                "follow-up form 至少需要两份 lexical evidence")
        _sorted_unique(
            evidence,
            key=ProviderOriginFollowupLexicalEvidenceV1.canonical_record,
            label="follow-up form lexical evidence",
        )
        if any(item.span_scalars != scalars for item in evidence):
            raise ProviderOriginFollowupError(
                "follow-up lexical evidence 未回读同一构式")
        if (type(self.selector_code) is not int
                or self.selector_code
                != PROVIDER_ORIGIN_FOLLOWUP_SELECTOR_ORIGIN_FOCUS_TO_CONTRAST_FOCUS_V1):
            raise ProviderOriginFollowupError("follow-up selector 未注册")
        if (type(self.output_mode) is not int
                or self.output_mode
                != PROVIDER_ORIGIN_FOLLOWUP_OUTPUT_ANCHOR_OCCURRENCE_SLICE_V1):
            raise ProviderOriginFollowupError("follow-up output mode 未注册")
        maximum = _scalar(
            self.output_max_bytes,
            label="follow-up output max bytes",
            minimum=1,
        )
        if maximum > _MAX_OUTPUT_BYTES:
            raise ProviderOriginFollowupError("follow-up output max bytes 超预算")
        object.__setattr__(self, "input_scalars", scalars)
        object.__setattr__(self, "lexical_evidence", evidence)
        object.__setattr__(self, "output_max_bytes", maximum)
        expected = _identity(
            PROVIDER_ORIGIN_FOLLOWUP_FORM_IDENTITY_DOMAIN_V1,
            _form_body(self),
            label="follow-up form identity",
        )
        supplied = self.form_identity_u8
        if supplied and _digest(
                supplied, label="follow-up form identity") != expected:
            raise ProviderOriginFollowupError("follow-up form identity 漂移")
        object.__setattr__(self, "form_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出构式、来源证据、selector 和输出预算的完整 record。"""
        result = list(_form_body(self))
        _pack(result, self.form_identity_u8)
        return tuple(result)


def _profile_body(value: "ProviderOriginFollowupProfileV1") -> tuple[int, ...]:
    """写出不含 self identity 的 source-bound origin/contrast profile。"""
    result = [
        PROVIDER_ORIGIN_FOLLOWUP_PROFILE_RECORD_V1,
        value.profile_revision,
        value.provider_kind,
        value.relation_kind_code,
        value.origin_focus_start,
        value.origin_focus_end,
        value.target_start,
        value.target_end,
    ]
    for item in (
            value.profile_key_u8,
            value.form_identity_u8,
            value.provider_identity_u8,
            value.runtime_identity_u8,
            value.provider_catalog_identity_u8,
            value.origin_provider_result_identity_u8,
            value.contrast_provider_result_identity_u8,
            value.source_record_key,
            value.source_ref_stable_key,
            value.source_commitment_u8,
            value.w03_observation_key,
            value.w04_observation_key,
            value.w05_observation_key,
            value.generation_construction_key,
            value.proposition_key,
            value.predicate_key,
            value.origin_anchor_identity_u8,
            value.contrast_anchor_identity_u8,
            value.origin_focus_role_binding_key,
            value.origin_focus_role_key,
            value.origin_focus_filler_key,
            value.origin_focus_occurrence_key,
            value.target_role_binding_key,
            value.target_role_key,
            value.target_filler_key,
            value.target_occurrence_key):
        _pack(result, item)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11C
@dataclass(frozen=True, slots=True)
class ProviderOriginFollowupProfileV1:
    """一条课程证明过的 origin-focus 到 contrast-focus 结构 profile。"""

    profile_key_u8: tuple[int, ...]
    profile_revision: int
    form_identity_u8: tuple[int, ...]
    provider_kind: int
    provider_identity_u8: tuple[int, ...]
    runtime_identity_u8: tuple[int, ...]
    provider_catalog_identity_u8: tuple[int, ...]
    origin_provider_result_identity_u8: tuple[int, ...]
    contrast_provider_result_identity_u8: tuple[int, ...]
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
    origin_anchor_identity_u8: tuple[int, ...]
    contrast_anchor_identity_u8: tuple[int, ...]
    origin_focus_role_binding_key: tuple[int, ...]
    origin_focus_role_key: tuple[int, ...]
    origin_focus_filler_key: tuple[int, ...]
    origin_focus_occurrence_key: tuple[int, ...]
    origin_focus_start: int
    origin_focus_end: int
    target_role_binding_key: tuple[int, ...]
    target_role_key: tuple[int, ...]
    target_filler_key: tuple[int, ...]
    target_occurrence_key: tuple[int, ...]
    target_start: int
    target_end: int
    profile_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 provider/source/anchor/role/occurrence 的全量候选约束。"""
        key = _key(self.profile_key_u8, label="follow-up profile key")
        if any(item < 0x21 or item > 0x7E for item in key):
            raise ProviderOriginFollowupError("follow-up profile key 非 ASCII")
        revision = _scalar(
            self.profile_revision,
            label="follow-up profile revision",
            minimum=1,
        )
        if (type(self.provider_kind) is not int
                or self.provider_kind != PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05):
            raise ProviderOriginFollowupError("follow-up profile provider kind 未注册")
        relation = _scalar(
            self.relation_kind_code,
            label="follow-up profile relation kind",
            minimum=1,
        )
        for name in (
                "form_identity_u8", "provider_identity_u8", "runtime_identity_u8",
                "provider_catalog_identity_u8", "origin_provider_result_identity_u8",
                "contrast_provider_result_identity_u8", "source_commitment_u8",
                "origin_anchor_identity_u8", "contrast_anchor_identity_u8"):
            object.__setattr__(self, name, _digest(
                getattr(self, name), label=f"follow-up profile {name}"))
        for name in (
                "source_record_key", "source_ref_stable_key", "w03_observation_key",
                "w04_observation_key", "w05_observation_key",
                "generation_construction_key", "proposition_key", "predicate_key",
                "origin_focus_role_binding_key", "origin_focus_role_key",
                "origin_focus_filler_key", "origin_focus_occurrence_key",
                "target_role_binding_key", "target_role_key", "target_filler_key",
                "target_occurrence_key"):
            object.__setattr__(self, name, _key(
                getattr(self, name), label=f"follow-up profile {name}"))
        origin_start = _scalar(
            self.origin_focus_start,
            label="follow-up profile origin focus start",
        )
        origin_end = _scalar(
            self.origin_focus_end,
            label="follow-up profile origin focus end",
            minimum=1,
        )
        target_start = _scalar(
            self.target_start,
            label="follow-up profile target start",
        )
        target_end = _scalar(
            self.target_end,
            label="follow-up profile target end",
            minimum=1,
        )
        if origin_end <= origin_start or target_end <= target_start:
            raise ProviderOriginFollowupError("follow-up profile occurrence span 非法")
        if (self.origin_focus_role_binding_key == self.target_role_binding_key
                or self.origin_focus_filler_key == self.target_filler_key
                or self.origin_focus_occurrence_key == self.target_occurrence_key):
            raise ProviderOriginFollowupError(
                "follow-up profile origin 与 target 不得是同一结构候选")
        object.__setattr__(self, "profile_key_u8", key)
        object.__setattr__(self, "profile_revision", revision)
        object.__setattr__(self, "relation_kind_code", relation)
        object.__setattr__(self, "origin_focus_start", origin_start)
        object.__setattr__(self, "origin_focus_end", origin_end)
        object.__setattr__(self, "target_start", target_start)
        object.__setattr__(self, "target_end", target_end)
        expected = _identity(
            PROVIDER_ORIGIN_FOLLOWUP_PROFILE_IDENTITY_DOMAIN_V1,
            _profile_body(self),
            label="follow-up profile identity",
        )
        supplied = self.profile_identity_u8
        if supplied and _digest(
                supplied, label="follow-up profile identity") != expected:
            raise ProviderOriginFollowupError("follow-up profile identity 漂移")
        object.__setattr__(self, "profile_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 route revision 与 origin/contrast structural binding。"""
        result = list(_profile_body(self))
        _pack(result, self.profile_identity_u8)
        return tuple(result)


def _catalog_body(value: "ProviderOriginFollowupCatalogV1") -> tuple[int, ...]:
    """写出不含 self identity 的 catalog record。"""
    result = [
        PROVIDER_ORIGIN_FOLLOWUP_CATALOG_RECORD_V1,
        value.catalog_schema,
    ]
    _pack(result, value.source_payload_closure_identity_u8)
    result.append(len(value.forms))
    for form in value.forms:
        _pack(result, form.canonical_record())
    result.append(len(value.profiles))
    for profile in value.profiles:
        _pack(result, profile.canonical_record())
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11C
@dataclass(frozen=True, slots=True)
class ProviderOriginFollowupCatalogV1:
    """公开课程编译出的纯整数 form/profile catalog。"""

    source_payload_closure_identity_u8: tuple[int, ...]
    forms: tuple[ProviderOriginFollowupFormV1, ...]
    profiles: tuple[ProviderOriginFollowupProfileV1, ...]
    catalog_schema: int = PROVIDER_ORIGIN_FOLLOWUP_CATALOG_SCHEMA_V1
    catalog_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 closure、form/profile 排序和 form-to-profile 全覆盖关系。"""
        closure = _digest(
            self.source_payload_closure_identity_u8,
            label="follow-up catalog closure identity",
        )
        if (type(self.catalog_schema) is not int
                or self.catalog_schema != PROVIDER_ORIGIN_FOLLOWUP_CATALOG_SCHEMA_V1):
            raise ProviderOriginFollowupError("follow-up catalog schema 未注册")
        forms = self.forms
        profiles = self.profiles
        if (type(forms) is not tuple or not forms
                or any(type(item) is not ProviderOriginFollowupFormV1
                       for item in forms)):
            raise ProviderOriginFollowupError("follow-up catalog forms 非法")
        if (type(profiles) is not tuple or not profiles
                or any(type(item) is not ProviderOriginFollowupProfileV1
                       for item in profiles)):
            raise ProviderOriginFollowupError("follow-up catalog profiles 非法")
        _sorted_unique(
            forms,
            key=ProviderOriginFollowupFormV1.canonical_record,
            label="follow-up catalog forms",
        )
        _sorted_unique(
            profiles,
            key=ProviderOriginFollowupProfileV1.canonical_record,
            label="follow-up catalog profiles",
        )
        form_identities = tuple(item.form_identity_u8 for item in forms)
        for left_index, left in enumerate(form_identities):
            for right in form_identities[left_index + 1:]:
                if compare_nonnegative_integer_records_v1(
                        left,
                        right,
                        label="follow-up catalog form identity") == 0:
                    raise ProviderOriginFollowupError(
                        "follow-up catalog form identity 重复")
        for profile in profiles:
            if not any(
                    compare_nonnegative_integer_records_v1(
                        profile.form_identity_u8,
                        form_identity,
                        label="follow-up profile form identity") == 0
                    for form_identity in form_identities):
                raise ProviderOriginFollowupError(
                    "follow-up profile 未绑定 catalog form")
        object.__setattr__(self, "source_payload_closure_identity_u8", closure)
        object.__setattr__(self, "forms", forms)
        object.__setattr__(self, "profiles", profiles)
        expected = _identity(
            PROVIDER_ORIGIN_FOLLOWUP_CATALOG_IDENTITY_DOMAIN_V1,
            _catalog_body(self),
            label="follow-up catalog identity",
        )
        supplied = self.catalog_identity_u8
        if supplied and _digest(
                supplied, label="follow-up catalog identity") != expected:
            raise ProviderOriginFollowupError("follow-up catalog identity 漂移")
        object.__setattr__(self, "catalog_identity_u8", expected)

    def matching_forms(
            self,
            input_scalars: tuple[int, ...],
            ) -> tuple[ProviderOriginFollowupFormV1, ...]:
        """仅按已学习的 exact scalar 构式匹配，不对表层做猜测或归一化。"""
        scalars = _unicode_scalars(
            input_scalars,
            label="follow-up lookup scalars",
            allow_empty=False,
        )
        return tuple(item for item in self.forms if item.input_scalars == scalars)

    def profiles_for_form(
            self,
            form: ProviderOriginFollowupFormV1,
            ) -> tuple[ProviderOriginFollowupProfileV1, ...]:
        """返回绑定给唯一 canonical form 的所有来源 profile。"""
        if type(form) is not ProviderOriginFollowupFormV1:
            raise TypeError("follow-up catalog form 类型错误")
        if form.form_identity_u8 not in tuple(
                item.form_identity_u8 for item in self.forms):
            raise ProviderOriginFollowupError("follow-up form 不属于 catalog")
        return tuple(
            item for item in self.profiles
            if item.form_identity_u8 == form.form_identity_u8)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 source closure 与课程本体完整绑定的 record。"""
        result = list(_catalog_body(self))
        _pack(result, self.catalog_identity_u8)
        return tuple(result)


def _profile_matches_anchor(
        profile: ProviderOriginFollowupProfileV1,
        anchor: ProviderOriginAnchorProjectionV1,
        ) -> bool:
    """只比较 profile 明示承诺的字段，不从 output 或 SourceRef 反向检索。"""
    return (
        anchor.accepted
        and anchor.provider_kind == profile.provider_kind
        and anchor.provider_identity_u8 == profile.provider_identity_u8
        and anchor.runtime_identity_u8 == profile.runtime_identity_u8
        and anchor.catalog_record_identity_u8 == profile.provider_catalog_identity_u8
        and anchor.provider_result_identity_u8
        == profile.origin_provider_result_identity_u8
        and anchor.source_record_key == profile.source_record_key
        and anchor.source_ref_stable_key == profile.source_ref_stable_key
        and anchor.source_commitment_u8 == profile.source_commitment_u8
        and anchor.w03_observation_key == profile.w03_observation_key
        and anchor.w04_observation_key == profile.w04_observation_key
        and anchor.w05_observation_key == profile.w05_observation_key
        and anchor.generation_construction_key == profile.generation_construction_key
        and anchor.proposition_key == profile.proposition_key
        and anchor.predicate_key == profile.predicate_key
        and anchor.relation_kind_code == profile.relation_kind_code
        and anchor.anchor_identity_u8 == profile.origin_anchor_identity_u8
        and anchor.focus_role_binding_key == profile.origin_focus_role_binding_key
        and anchor.focus_role_key == profile.origin_focus_role_key
        and anchor.focus_filler_key == profile.origin_focus_filler_key
        and anchor.focus_occurrence_key == profile.origin_focus_occurrence_key
        and anchor.focus_answer_start == profile.origin_focus_start
        and anchor.focus_answer_end == profile.origin_focus_end
    )


def _single_binding(
        bindings: tuple[ProviderOriginRoleBindingV1, ...],
        *,
        binding_key: tuple[int, ...],
        role_key: tuple[int, ...],
        filler_key: tuple[int, ...],
        label: str,
        ) -> ProviderOriginRoleBindingV1:
    """从已经固定顺序的 role bindings 读取一个明确结构候选。"""
    candidates = tuple(
        item for item in bindings
        if (item.binding_key == binding_key
            and item.role_key == role_key
            and item.filler_key == filler_key))
    if len(candidates) != 1:
        raise ProviderOriginFollowupError(f"{label} 不是唯一 role binding")
    return candidates[0]


def _single_occurrence(
        occurrences: tuple[ProviderOriginOccurrenceV1, ...],
        *,
        occurrence_key: tuple[int, ...],
        semantic_object_key: tuple[int, ...],
        start: int,
        end: int,
        label: str,
        ) -> ProviderOriginOccurrenceV1:
    """按 occurrence identity、semantic object 和冻结 span 选唯一事实。"""
    candidates = tuple(
        item for item in occurrences
        if (item.occurrence_key == occurrence_key
            and item.semantic_object_key == semantic_object_key
            and item.start == start
            and item.end == end))
    if len(candidates) != 1:
        raise ProviderOriginFollowupError(f"{label} 不是唯一 occurrence")
    return candidates[0]


def _candidate_body(value: "ProviderOriginReferenceCandidateV1") -> tuple[int, ...]:
    """写出不含 self identity 的已缩减 reference candidate record。"""
    result = [
        PROVIDER_ORIGIN_REFERENCE_CANDIDATE_RECORD_V1,
        value.profile_revision,
        value.provider_kind,
        value.relation_kind_code,
        value.answer_start,
        value.answer_end,
    ]
    for item in (
            value.catalog_identity_u8,
            value.form_identity_u8,
            value.profile_identity_u8,
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


# object-model: value; representation=struct; interop=DLG-RAW-11C
@dataclass(frozen=True, slots=True)
class ProviderOriginReferenceCandidateV1:
    """一个同时闭合 form、provider、source、anchor 和 role occurrence 的候选。"""

    catalog_identity_u8: tuple[int, ...]
    form_identity_u8: tuple[int, ...]
    profile_identity_u8: tuple[int, ...]
    profile_revision: int
    provider_kind: int
    provider_identity_u8: tuple[int, ...]
    runtime_identity_u8: tuple[int, ...]
    provider_catalog_identity_u8: tuple[int, ...]
    provider_result_identity_u8: tuple[int, ...]
    source_record_key: tuple[int, ...]
    source_ref_stable_key: tuple[int, ...]
    source_commitment_u8: tuple[int, ...]
    w03_observation_key: tuple[int, ...]
    w04_observation_key: tuple[int, ...]
    w05_observation_key: tuple[int, ...]
    generation_construction_key: tuple[int, ...]
    relation_kind_code: int
    anchor_identity_u8: tuple[int, ...]
    reference_role_binding_key: tuple[int, ...]
    reference_role_key: tuple[int, ...]
    reference_filler_key: tuple[int, ...]
    reference_occurrence_key: tuple[int, ...]
    answer_role_binding_key: tuple[int, ...]
    answer_role_key: tuple[int, ...]
    answer_filler_key: tuple[int, ...]
    answer_occurrence_key: tuple[int, ...]
    answer_start: int
    answer_end: int
    output_scalars: tuple[int, ...]
    output_u8: tuple[int, ...]
    candidate_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 candidate 载荷，确保输出只是已验证 anchor occurrence 的 byte slice。"""
        revision = _scalar(
            self.profile_revision,
            label="reference candidate profile revision",
            minimum=1,
        )
        if (type(self.provider_kind) is not int
                or self.provider_kind != PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05):
            raise ProviderOriginFollowupError("reference candidate provider kind 未注册")
        relation = _scalar(
            self.relation_kind_code,
            label="reference candidate relation kind",
            minimum=1,
        )
        for name in (
                "catalog_identity_u8", "form_identity_u8", "profile_identity_u8",
                "provider_identity_u8", "runtime_identity_u8",
                "provider_catalog_identity_u8", "provider_result_identity_u8",
                "source_commitment_u8", "anchor_identity_u8"):
            object.__setattr__(self, name, _digest(
                getattr(self, name), label=f"reference candidate {name}"))
        for name in (
                "source_record_key", "source_ref_stable_key", "w03_observation_key",
                "w04_observation_key", "w05_observation_key",
                "generation_construction_key", "reference_role_binding_key",
                "reference_role_key", "reference_filler_key",
                "reference_occurrence_key", "answer_role_binding_key",
                "answer_role_key", "answer_filler_key", "answer_occurrence_key"):
            object.__setattr__(self, name, _key(
                getattr(self, name), label=f"reference candidate {name}"))
        start = _scalar(
            self.answer_start,
            label="reference candidate answer start",
        )
        end = _scalar(
            self.answer_end,
            label="reference candidate answer end",
            minimum=1,
        )
        if end <= start:
            raise ProviderOriginFollowupError("reference candidate answer span 非法")
        scalars = _unicode_scalars(
            self.output_scalars,
            label="reference candidate output scalars",
            allow_empty=False,
        )
        output = _u8(
            self.output_u8,
            label="reference candidate output u8",
            allow_empty=False,
        )
        if encode_utf8_v1(scalars) != output:
            raise ProviderOriginFollowupError("reference candidate output UTF-8 漂移")
        if len(output) > _MAX_OUTPUT_BYTES:
            raise ProviderOriginFollowupError("reference candidate output 超预算")
        object.__setattr__(self, "profile_revision", revision)
        object.__setattr__(self, "relation_kind_code", relation)
        object.__setattr__(self, "answer_start", start)
        object.__setattr__(self, "answer_end", end)
        object.__setattr__(self, "output_scalars", scalars)
        object.__setattr__(self, "output_u8", output)
        expected = _identity(
            PROVIDER_ORIGIN_REFERENCE_CANDIDATE_IDENTITY_DOMAIN_V1,
            _candidate_body(self),
            label="reference candidate identity",
        )
        supplied = self.candidate_identity_u8
        if supplied and _digest(
                supplied, label="reference candidate identity") != expected:
            raise ProviderOriginFollowupError("reference candidate identity 漂移")
        object.__setattr__(self, "candidate_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出可由任意整数实现重放的完整 candidate record。"""
        result = list(_candidate_body(self))
        _pack(result, self.candidate_identity_u8)
        return tuple(result)


def candidate_from_provider_origin_profile_v1(
        catalog: ProviderOriginFollowupCatalogV1,
        form: ProviderOriginFollowupFormV1,
        profile: ProviderOriginFollowupProfileV1,
        anchor: ProviderOriginAnchorProjectionV1,
        ) -> ProviderOriginReferenceCandidateV1 | None:
    """从一个可见 anchor 形成 candidate；不匹配 profile 时返回 ``None``。

    该函数不搜索输出文本、不重新运行 provider。origin 和 target 两端的 role /
    occurrence 全部来自课程同次 proof 已冻结的 profile，并在当前 anchor 中逐项回读。
    """
    if (type(catalog) is not ProviderOriginFollowupCatalogV1
            or type(form) is not ProviderOriginFollowupFormV1
            or type(profile) is not ProviderOriginFollowupProfileV1
            or type(anchor) is not ProviderOriginAnchorProjectionV1):
        raise TypeError("reference candidate 输入类型错误")
    if profile.form_identity_u8 != form.form_identity_u8:
        raise ProviderOriginFollowupError("reference profile 与 form 漂移")
    if (not any(
            compare_nonnegative_integer_records_v1(
                profile.canonical_record(),
                item.canonical_record(),
                label="reference candidate profile membership") == 0
            for item in catalog.profiles)
            or not any(
                compare_nonnegative_integer_records_v1(
                    form.canonical_record(),
                    item.canonical_record(),
                    label="reference candidate form membership") == 0
                for item in catalog.forms)):
        raise ProviderOriginFollowupError("reference candidate 输入不属于 catalog")
    if not _profile_matches_anchor(profile, anchor):
        return None
    _single_binding(
        anchor.ordered_role_bindings,
        binding_key=profile.origin_focus_role_binding_key,
        role_key=profile.origin_focus_role_key,
        filler_key=profile.origin_focus_filler_key,
        label="reference candidate origin focus",
    )
    _single_occurrence(
        anchor.ordered_occurrences,
        occurrence_key=profile.origin_focus_occurrence_key,
        semantic_object_key=profile.origin_focus_filler_key,
        start=profile.origin_focus_start,
        end=profile.origin_focus_end,
        label="reference candidate origin focus",
    )
    _single_binding(
        anchor.ordered_role_bindings,
        binding_key=profile.target_role_binding_key,
        role_key=profile.target_role_key,
        filler_key=profile.target_filler_key,
        label="reference candidate target",
    )
    _single_occurrence(
        anchor.ordered_occurrences,
        occurrence_key=profile.target_occurrence_key,
        semantic_object_key=profile.target_filler_key,
        start=profile.target_start,
        end=profile.target_end,
        label="reference candidate target",
    )
    if profile.target_end > len(anchor.output_scalars):
        raise ProviderOriginFollowupError("reference candidate target span 越出 anchor output")
    output_scalars = anchor.output_scalars[profile.target_start:profile.target_end]
    output_u8 = encode_utf8_v1(output_scalars)
    byte_start = len(encode_utf8_v1(anchor.output_scalars[:profile.target_start]))
    byte_end = byte_start + len(output_u8)
    if (anchor.output_u8[byte_start:byte_end] != output_u8
            or byte_end > len(anchor.output_u8)):
        raise ProviderOriginFollowupError(
            "reference candidate output 不是 anchor 的精确 byte slice")
    if len(output_u8) > form.output_max_bytes:
        raise ProviderOriginFollowupError("reference candidate output 超出 form 预算")
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
        output_scalars,
        output_u8,
    )


def _result_body(value: "ProviderOriginFollowupResultV1") -> tuple[int, ...]:
    """写出不含 Python object identity 的 follow-up result body。"""
    result = [
        PROVIDER_ORIGIN_FOLLOWUP_RESULT_RECORD_V1,
        value.status,
        value.mapped_dlg_result_code,
        value.matched_form_count,
        value.candidate_count,
        value.context_write_origin,
    ]
    for item in (
            value.intake.canonical_record(),
            value.catalog_identity_u8,
            (() if value.form is None else value.form.canonical_record()),
            (() if value.context_read is None
             else value.context_read.canonical_record()),
            (() if value.candidate is None
             else value.candidate.canonical_record()),
            value.output_scalars,
            value.output_u8,
            value.persistent_state_delta):
        _pack(result, item)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11C
@dataclass(frozen=True, slots=True)
class ProviderOriginFollowupResultV1:
    """一次来源内追问的无写入结果 carrier。"""

    status: int
    mapped_dlg_result_code: int
    intake: ConversationRawIntake
    catalog_identity_u8: tuple[int, ...]
    matched_form_count: int
    candidate_count: int
    form: ProviderOriginFollowupFormV1 | None = None
    context_read: MixedContextReadV2 | None = None
    candidate: ProviderOriginReferenceCandidateV1 | None = None
    output_scalars: tuple[int, ...] = ()
    output_u8: tuple[int, ...] = ()
    context_write_origin: int = MIXED_CONTEXT_WRITE_ORIGIN_NONE
    persistent_state_delta: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """分账 not-applicable、answer 与零输出 reject，不允许隐式 context write。"""
        if type(self.status) is not int or self.status not in _FOLLOWUP_STATUSES:
            raise ProviderOriginFollowupError("follow-up result status 未注册")
        if (type(self.mapped_dlg_result_code) is not int
                or self.mapped_dlg_result_code not in _FOLLOWUP_RESULT_CODES):
            raise ProviderOriginFollowupError("follow-up result code 未注册")
        if type(self.intake) is not ConversationRawIntake:
            raise TypeError("follow-up result intake 类型错误")
        catalog = _digest(
            self.catalog_identity_u8,
            label="follow-up result catalog identity",
        )
        forms = _scalar(
            self.matched_form_count,
            label="follow-up result matched form count",
        )
        candidates = _scalar(
            self.candidate_count,
            label="follow-up result candidate count",
        )
        if self.form is not None and type(self.form) is not ProviderOriginFollowupFormV1:
            raise TypeError("follow-up result form 类型错误")
        if self.context_read is not None and type(self.context_read) is not MixedContextReadV2:
            raise TypeError("follow-up result context read 类型错误")
        if (self.candidate is not None
                and type(self.candidate) is not ProviderOriginReferenceCandidateV1):
            raise TypeError("follow-up result candidate 类型错误")
        scalars = _unicode_scalars(
            self.output_scalars,
            label="follow-up result output scalars",
            allow_empty=True,
        )
        output = _u8(
            self.output_u8,
            label="follow-up result output u8",
            allow_empty=True,
        )
        delta = _vector(
            self.persistent_state_delta,
            label="follow-up result persistent state delta",
            allow_empty=True,
        )
        if (type(self.context_write_origin) is not int
                or self.context_write_origin != MIXED_CONTEXT_WRITE_ORIGIN_NONE
                or delta != ()):
            raise ProviderOriginFollowupError(
                "follow-up result 不得写入 context 或 persistent state")
        if self.status == PROVIDER_ORIGIN_FOLLOWUP_STATUS_NOT_APPLICABLE:
            if (self.form is not None or self.context_read is not None
                    or self.candidate is not None or forms != 0
                    or candidates != 0 or scalars or output):
                raise ProviderOriginFollowupError(
                    "follow-up not-applicable 不得携带消费 payload")
        elif self.status == PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER:
            if (not self.intake.accepted
                    or self.mapped_dlg_result_code != DLG_RAW_ACCEPT
                    or self.form is None or self.context_read is None
                    or self.candidate is None or forms != 1 or candidates != 1
                    or not scalars or not output
                    or scalars != self.candidate.output_scalars
                    or output != self.candidate.output_u8
                    or len(output) > self.form.output_max_bytes):
                raise ProviderOriginFollowupError("follow-up answer result 非法")
        else:
            if (not self.intake.accepted or self.form is None
                    or self.candidate is not None or scalars or output):
                raise ProviderOriginFollowupError("follow-up reject result 非法")
            if self.mapped_dlg_result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
                if self.context_read is not None or forms < 2 or candidates != 0:
                    raise ProviderOriginFollowupError(
                        "follow-up lexical ambiguity result 非法")
            elif self.mapped_dlg_result_code == DLG_RAW_REJECT_CONTEXT:
                if forms != 1:
                    raise ProviderOriginFollowupError("follow-up context result form 漂移")
            elif self.mapped_dlg_result_code == DLG_RAW_REJECT_REFERENCE_AMBIGUOUS:
                if self.context_read is None or forms != 1 or candidates < 2:
                    raise ProviderOriginFollowupError(
                        "follow-up reference ambiguity result 非法")
            elif self.mapped_dlg_result_code in {
                    DLG_RAW_REJECT_RUNTIME, DLG_RAW_REJECT_OUTPUT_BUDGET}:
                if forms != 1:
                    raise ProviderOriginFollowupError("follow-up runtime result form 漂移")
            else:
                raise ProviderOriginFollowupError("follow-up reject result code 未注册")
        object.__setattr__(self, "catalog_identity_u8", catalog)
        object.__setattr__(self, "matched_form_count", forms)
        object.__setattr__(self, "candidate_count", candidates)
        object.__setattr__(self, "output_scalars", scalars)
        object.__setattr__(self, "output_u8", output)
        object.__setattr__(self, "persistent_state_delta", delta)

    @property
    def accepted(self) -> bool:
        """只有唯一 candidate 的 occurrence slice 被复用时返回真。"""
        return self.status == PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER

    @property
    def handled(self) -> bool:
        """表明已匹配 follow-up form，调用方不得再回退到 provider。"""
        return self.status != PROVIDER_ORIGIN_FOLLOWUP_STATUS_NOT_APPLICABLE

    def canonical_record(self) -> tuple[int, ...]:
        """导出 intake/read/candidate/output/no-write 的完整结果记录。"""
        return _result_body(self)


def _not_applicable(
        intake: ConversationRawIntake,
        catalog: ProviderOriginFollowupCatalogV1,
        ) -> ProviderOriginFollowupResultV1:
    """构造允许调用方继续现有 Frame/provider fallback 的零 payload carrier。"""
    mapped = intake.result_code if not intake.accepted else DLG_RAW_REJECT_LEXICAL_MISS
    if mapped not in _FOLLOWUP_RESULT_CODES:
        # RAW-00 代码并非 follow-up 语义输出，调用方只检查 handled。
        mapped = DLG_RAW_REJECT_LEXICAL_MISS
    return ProviderOriginFollowupResultV1(
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_NOT_APPLICABLE,
        mapped,
        intake,
        catalog.catalog_identity_u8,
        0,
        0,
    )


def _reject(
        code: int,
        intake: ConversationRawIntake,
        catalog: ProviderOriginFollowupCatalogV1,
        form: ProviderOriginFollowupFormV1,
        *,
        context_read: MixedContextReadV2 | None,
        candidate_count: int,
    ) -> ProviderOriginFollowupResultV1:
    """构造已匹配 form 后的零输出、零写入 fail-closed result。"""
    return ProviderOriginFollowupResultV1(
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED,
        code,
        intake,
        catalog.catalog_identity_u8,
        1,
        candidate_count,
        form,
        context_read,
    )


def _candidate_build_outcome(
        catalog: ProviderOriginFollowupCatalogV1,
        form: ProviderOriginFollowupFormV1,
        profile: ProviderOriginFollowupProfileV1,
        anchor: ProviderOriginAnchorProjectionV1,
        ) -> tuple[int, ProviderOriginReferenceCandidateV1 | None]:
    """把 candidate 构造归约为显式 tagged outcome。

    正常 reducer 不从异常类别推导语言结果。已冻结 catalog 或 context 的宿主实现
    故障统一落到预登记的 ``RUNTIME`` outcome，随后由调用方形成零输出拒绝。
    """
    try:
        candidate = candidate_from_provider_origin_profile_v1(
            catalog,
            form,
            profile,
            anchor,
        )
    except Exception:
        return _CANDIDATE_BUILD_OUTCOME_RUNTIME, None
    if candidate is None:
        return _CANDIDATE_BUILD_OUTCOME_NONE, None
    return _CANDIDATE_BUILD_OUTCOME_CANDIDATE, candidate


def run_provider_origin_followup_v1(
        intake: ConversationRawIntake,
        context_read: MixedContextReadV2 | None,
        catalog: ProviderOriginFollowupCatalogV1,
        ) -> ProviderOriginFollowupResultV1:
    """运行一次严格相邻 provider-origin follow-up candidate reduction。

    无 form match 时返回 ``not_applicable``，以保留已有公开 Frame/provider
    fallback。匹配 form 后任何 context 缺口都终止于本模块，绝不跨域扫描或
    回退到新的 provider dispatch。
    """
    if type(intake) is not ConversationRawIntake:
        raise TypeError("follow-up intake 类型错误")
    if context_read is not None and type(context_read) is not MixedContextReadV2:
        raise TypeError("follow-up context read 类型错误")
    if type(catalog) is not ProviderOriginFollowupCatalogV1:
        raise TypeError("follow-up catalog 类型错误")
    if not intake.accepted:
        return _not_applicable(intake, catalog)
    forms = catalog.matching_forms(intake.unicode_scalars)
    if not forms:
        return _not_applicable(intake, catalog)
    if len(forms) != 1:
        return _reject(
            DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
            intake,
            catalog,
            forms[0],
            context_read=None,
            candidate_count=0,
        )
    form = forms[0]
    if context_read is None:
        return _reject(
            DLG_RAW_REJECT_CONTEXT,
            intake,
            catalog,
            form,
            context_read=None,
            candidate_count=0,
        )
    if (context_read.witness.requested_limit != 1
            or len(context_read.turns) != 1
            or type(context_read.turns[0]) is not ProviderOriginContextTurnV1
            or context_read.turns[0].turn_kind
            != MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION):
        return _reject(
            DLG_RAW_REJECT_CONTEXT,
            intake,
            catalog,
            form,
            context_read=context_read,
            candidate_count=0,
    )
    anchor = context_read.turns[0].anchor_projection
    candidates: list[ProviderOriginReferenceCandidateV1] = []
    for profile in catalog.profiles_for_form(form):
        outcome, candidate = _candidate_build_outcome(
            catalog,
            form,
            profile,
            anchor,
        )
        if outcome == _CANDIDATE_BUILD_OUTCOME_RUNTIME:
            return _reject(
                DLG_RAW_REJECT_RUNTIME,
                intake,
                catalog,
                form,
                context_read=context_read,
                candidate_count=0,
            )
        if outcome == _CANDIDATE_BUILD_OUTCOME_CANDIDATE:
            if candidate is None:
                return _reject(
                    DLG_RAW_REJECT_RUNTIME,
                    intake,
                    catalog,
                    form,
                    context_read=context_read,
                    candidate_count=0,
                )
            candidates.append(candidate)
    if not candidates:
        return _reject(
            DLG_RAW_REJECT_CONTEXT,
            intake,
            catalog,
            form,
            context_read=context_read,
            candidate_count=0,
        )
    for index, candidate in enumerate(candidates):
        current_record = candidate.canonical_record()
        for prior in candidates[:index]:
            if compare_nonnegative_integer_records_v1(
                    current_record,
                    prior.canonical_record(),
                    label="follow-up candidate duplicate") == 0:
                return _reject(
                    DLG_RAW_REJECT_RUNTIME,
                    intake,
                    catalog,
                    form,
                    context_read=context_read,
                    candidate_count=len(candidates),
                )
    if len(candidates) > 1:
        return _reject(
            DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
            intake,
            catalog,
            form,
            context_read=context_read,
            candidate_count=len(candidates),
        )
    selected = candidates[0]
    return ProviderOriginFollowupResultV1(
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER,
        DLG_RAW_ACCEPT,
        intake,
        catalog.catalog_identity_u8,
        1,
        1,
        form,
        context_read,
        selected,
        selected.output_scalars,
        selected.output_u8,
    )


def provider_origin_followup_schema_record_v1() -> tuple[int, ...]:
    """导出 V4 runtime binding 可锁定的 reducer schema 与结果码。"""
    return (
        PROVIDER_ORIGIN_FOLLOWUP_CATALOG_SCHEMA_V1,
        PROVIDER_ORIGIN_FOLLOWUP_LEXICAL_EVIDENCE_RECORD_V1,
        PROVIDER_ORIGIN_FOLLOWUP_FORM_RECORD_V1,
        PROVIDER_ORIGIN_FOLLOWUP_PROFILE_RECORD_V1,
        PROVIDER_ORIGIN_FOLLOWUP_CATALOG_RECORD_V1,
        PROVIDER_ORIGIN_REFERENCE_CANDIDATE_RECORD_V1,
        PROVIDER_ORIGIN_FOLLOWUP_RESULT_RECORD_V1,
        PROVIDER_ORIGIN_FOLLOWUP_RECORD_ORDER_LEXICOGRAPHIC_V1,
        PROVIDER_ORIGIN_FOLLOWUP_SELECTOR_ORIGIN_FOCUS_TO_CONTRAST_FOCUS_V1,
        PROVIDER_ORIGIN_FOLLOWUP_OUTPUT_ANCHOR_OCCURRENCE_SLICE_V1,
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_NOT_APPLICABLE,
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER,
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED,
        DLG_RAW_ACCEPT,
        DLG_RAW_REJECT_LEXICAL_MISS,
        DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
        DLG_RAW_REJECT_CONTEXT,
        DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
        DLG_RAW_REJECT_RUNTIME,
        DLG_RAW_REJECT_OUTPUT_BUDGET,
        _CANDIDATE_BUILD_OUTCOME_NONE,
        _CANDIDATE_BUILD_OUTCOME_CANDIDATE,
        _CANDIDATE_BUILD_OUTCOME_RUNTIME,
        _DIGEST_SIZE,
        _MAX_OUTPUT_BYTES,
    )


def provider_origin_followup_schema_identity_v1() -> tuple[int, ...]:
    """返回 schema record 的 raw identity，避免 V4 只绑定偶然样本。"""
    return _identity(
        b"PURE-INTEGER-AI/DLG-RAW-11C/FOLLOWUP-SCHEMA/V1",
        provider_origin_followup_schema_record_v1(),
        label="follow-up schema identity",
    )


__all__ = [
    "PROVIDER_ORIGIN_FOLLOWUP_CATALOG_RECORD_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_CATALOG_SCHEMA_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_FORM_RECORD_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_LEXICAL_EVIDENCE_RECORD_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_OUTPUT_ANCHOR_OCCURRENCE_SLICE_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_PROFILE_RECORD_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_RECORD_ORDER_LEXICOGRAPHIC_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_RESULT_RECORD_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_SELECTOR_ORIGIN_FOCUS_TO_CONTRAST_FOCUS_V1",
    "PROVIDER_ORIGIN_FOLLOWUP_STATUS_ANSWER",
    "PROVIDER_ORIGIN_FOLLOWUP_STATUS_NOT_APPLICABLE",
    "PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED",
    "PROVIDER_ORIGIN_REFERENCE_CANDIDATE_RECORD_V1",
    "ProviderOriginFollowupCatalogV1",
    "ProviderOriginFollowupError",
    "ProviderOriginFollowupFormV1",
    "ProviderOriginFollowupLexicalEvidenceV1",
    "ProviderOriginFollowupProfileV1",
    "ProviderOriginFollowupResultV1",
    "ProviderOriginReferenceCandidateV1",
    "candidate_from_provider_origin_profile_v1",
    "compare_nonnegative_integer_records_v1",
    "order_items_by_nonnegative_integer_record_v1",
    "provider_origin_followup_schema_identity_v1",
    "provider_origin_followup_schema_record_v1",
    "run_provider_origin_followup_v1",
]
