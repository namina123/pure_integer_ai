"""DLG-RAW-11C：公开 provider-origin follow-up 课程的 host 适配器。

这个模块读取已经冻结的 public payload closure，并通过课程明确列出的两次实际
provider dispatch 建立方向 profile。路径、JSON、文本对象与 legacy typed proof
只停留在此适配器；输出给 reducer 的是完整有序整数 catalog。它不写 context、
SQLite、长期记忆或训练数据。
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    ProviderOriginAnchorProjectionV1,
    project_provider_origin_anchor_v1,
    provider_origin_legacy_proof_from_same_dispatch_v1,
    provider_origin_provider_binding_from_public_provider_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    PROVIDER_ORIGIN_FOLLOWUP_CATALOG_SCHEMA_V1,
    ProviderOriginFollowupCatalogV1,
    ProviderOriginFollowupError,
    ProviderOriginFollowupFormV1,
    ProviderOriginFollowupLexicalEvidenceV1,
    ProviderOriginFollowupProfileV1,
    compare_nonnegative_integer_records_v1,
    order_items_by_nonnegative_integer_record_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PublicProofSentenceProviderV1,
    run_public_proof_sentence_provider_vector_with_typed_proof,
    verify_public_proof_sentence_provider_result,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    decode_utf8_v1,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_provider_followup_course_v1.jsonl.sample")
PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_LOGICAL_KEYS_V1 = (
    PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_LOGICAL_KEY_V1,
    b"data/ph2/dlg_raw_public_provider_result_followup_course_v1.jsonl.sample",
)
PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_SCHEMA_V1 = 1

_COURSE_FIELDS = frozenset({
    "catalog_schema",
    "contrast_question_surface",
    "followup_lexical_sources",
    "origin_question_surface",
    "route_id",
    "route_revision",
})
_LEXICAL_SOURCE_FIELDS = frozenset({
    "attribution",
    "license_id",
    "raw_sha256",
    "relative_path",
    "span_utf8_hex",
})
_HEX = frozenset("0123456789abcdef")
_DATA_PH2_PREFIX_U8 = tuple(b"data/ph2/")
_DATA_COMPONENT_U8 = tuple(b"data")
_PH2_COMPONENT_U8 = tuple(b"ph2")
_DOT_COMPONENT_U8 = (0x2E,)
_DOT_DOT_COMPONENT_U8 = (0x2E, 0x2E)

# DLG-RAW-00 accepts exactly these physical terminal endings.  The provider
# anchor deliberately commits the complete raw intake record, so each allowed
# transport spelling needs its own proof-backed profile.  This keeps a later
# follow-up from treating a raw carrier identity as if it were line-normalized.
_ORIGIN_INPUT_LINE_SUFFIXES_V1 = (
    (),
    (0x0A,),
    (0x0D, 0x0A),
)


# object-model: exception; interop=DLG-RAW-11C
class PublicProviderOriginFollowupCatalogError(ValueError):
    """公开 follow-up 课程、source witness 或 provider profile 不能闭合。"""


def _exact(value: object, fields: frozenset[str], *, label: str) -> dict:
    """拒绝缺失/尾随 JSON fields，避免 parser 默认值进入协议。"""
    if (type(value) is not dict or len(value) != len(fields)
            or any(field not in value for field in fields)
            or any(field not in fields for field in value)):
        raise PublicProviderOriginFollowupCatalogError(f"{label} 字段集合漂移")
    return value


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    """只接受 strict integer，拒绝 bool、float 或隐式数值转换。"""
    if type(value) is not int or value < minimum:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 必须是不小于 {minimum} 的严格整数")
    return value


def _unicode_scalars(value: object, *, label: str) -> tuple[int, ...]:
    """把 adapter text 显式降为 Unicode scalar sequence，不交给默认编码。"""
    if (type(value) is not str or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 必须是无首尾空白的非空文本")
    result = tuple(ord(item) for item in value)
    try:
        encode_utf8_v1(result)
    except (TypeError, ValueError) as error:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 含非法 Unicode scalar") from error
    return result


def _ascii_u8(value: object, *, label: str) -> tuple[int, ...]:
    """冻结 logical key/route id 等稳定 ASCII 字段。"""
    if (type(value) is not str or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"
            or any(ord(item) < 0x21 or ord(item) > 0x7E for item in value)):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 必须是无空白稳定 ASCII")
    return tuple(ord(item) for item in value)


def _logical_key(value: object, *, label: str) -> tuple[int, ...]:
    """验证 source path 是受 closure 注册保护的 data/ph2 logical key。"""
    key = _ascii_u8(value, label=label)
    if key[:len(_DATA_PH2_PREFIX_U8)] != _DATA_PH2_PREFIX_U8:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 越出 data/ph2 logical key")
    components: list[tuple[int, ...]] = []
    current: list[int] = []
    for scalar in key:
        if scalar == 0x2F:
            if not current:
                raise PublicProviderOriginFollowupCatalogError(
                    f"{label} 越出 data/ph2 logical key")
            components.append(tuple(current))
            current = []
        else:
            current.append(scalar)
    if not current:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 越出 data/ph2 logical key")
    components.append(tuple(current))
    if (len(components) != 3
            or components[0] != _DATA_COMPONENT_U8
            or components[1] != _PH2_COMPONENT_U8
            or any(component == _DOT_COMPONENT_U8
                   or component == _DOT_DOT_COMPONENT_U8
                   for component in components)):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 越出 data/ph2 logical key")
    return key


def _hex_bytes(value: object, *, label: str) -> tuple[int, ...]:
    """以固定 nibble 规则把 canonical lowercase hex 还原为 raw u8。"""
    if (type(value) is not str or not value or len(value) % 2
            or any(item not in _HEX for item in value)):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 不是 canonical lowercase hex")
    result: list[int] = []
    for offset in range(0, len(value), 2):
        high = ord(value[offset])
        low = ord(value[offset + 1])
        high_value = high - 0x30 if high <= 0x39 else high - 0x61 + 10
        low_value = low - 0x30 if low <= 0x39 else low - 0x61 + 10
        result.append((high_value << 4) | low_value)
    return tuple(result)


def _source_payload(
        closure: PublicSourcePayloadClosureV1,
        logical_key_u8: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[bytes, tuple[int, ...]]:
    """从 closure 读取一份已登记 payload，并复核 record 长度/hash。"""
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("follow-up catalog closure 类型错误")
    key = bytes(logical_key_u8)
    try:
        record = closure.record_for(key)
        payload = closure.payload_for(key)
    except PublicSourcePayloadProviderError as error:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 不在 public payload closure 内") from error
    digest = tuple(public_source_payload_sha256_v1(payload))
    if (record.logical_key != key or record.raw_payload != payload
            or record.payload_length != len(payload)
            or tuple(record.raw_sha256) != digest):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} public payload record 漂移")
    return payload, digest


def _unique_span(
        payload: bytes,
        needle: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, int]:
    """只接受 public source 中的唯一 raw byte span，不按最近位置任选。"""
    if not needle:
        raise PublicProviderOriginFollowupCatalogError(f"{label} span 不得为空")
    haystack = tuple(payload)
    if len(needle) > len(haystack):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} span 缺失或不唯一")
    found_start = -1
    limit = len(haystack) - len(needle)
    for start in range(limit + 1):
        equal = True
        for offset, scalar in enumerate(needle):
            if haystack[start + offset] != scalar:
                equal = False
                break
        if equal:
            if found_start >= 0:
                raise PublicProviderOriginFollowupCatalogError(
                    f"{label} span 缺失或不唯一")
            found_start = start
    if found_start < 0:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} span 缺失或不唯一")
    return found_start, found_start + len(needle)


def _single_lf_jsonl_lines(payload: bytes, *, label: str) -> tuple[bytes, ...]:
    """以显式 u8 scan 读取单 LF 结尾的非空 JSONL records。"""
    if (not payload or payload[-1] != 0x0A
            or (len(payload) >= 2 and payload[-2] == 0x0A)):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 必须是单 LF 结尾的 JSONL")
    records: list[bytes] = []
    start = 0
    for index, scalar in enumerate(payload):
        if scalar != 0x0A:
            continue
        if index == start:
            raise PublicProviderOriginFollowupCatalogError(
                f"{label} 不得有空 JSONL 行")
        records.append(payload[start:index])
        start = index + 1
    if start != len(payload):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} JSONL 末尾 framing 漂移")
    return tuple(records)


def _lexical_evidence(
        raw: object,
        closure: PublicSourcePayloadClosureV1,
        *,
        label: str,
        ) -> ProviderOriginFollowupLexicalEvidenceV1:
    """将一份带许可的 lexical source 声明回读为 canonical evidence record。"""
    value = _exact(raw, _LEXICAL_SOURCE_FIELDS, label=label)
    logical_key = _logical_key(value["relative_path"], label=f"{label}.relative_path")
    expected_digest = _hex_bytes(value["raw_sha256"], label=f"{label}.raw_sha256")
    if len(expected_digest) != 32:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label}.raw_sha256 长度漂移")
    license_id = _ascii_u8(value["license_id"], label=f"{label}.license_id")
    if license_id != tuple(b"CC0-1.0"):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label}.license_id 必须是 CC0-1.0")
    attribution_scalars = _unicode_scalars(
        value["attribution"], label=f"{label}.attribution")
    attribution_u8 = encode_utf8_v1(attribution_scalars)
    span = _hex_bytes(value["span_utf8_hex"], label=f"{label}.span_utf8_hex")
    scalars = decode_utf8_v1(span)
    if scalars is None or encode_utf8_v1(scalars) != span:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label}.span_utf8_hex 不是 canonical UTF-8")
    payload, actual_digest = _source_payload(
        closure, logical_key, label=label)
    if actual_digest != expected_digest:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} raw SHA-256 漂移")
    start, end = _unique_span(payload, span, label=label)
    return ProviderOriginFollowupLexicalEvidenceV1(
        logical_key,
        actual_digest,
        license_id,
        attribution_u8,
        start,
        end,
        span,
        scalars,
    )


def _anchor_for_course_surface(
        provider: PublicProofSentenceProviderV1,
        scalars: tuple[int, ...],
        *,
        label: str,
        line_suffix: tuple[int, ...] = (),
        ) -> ProviderOriginAnchorProjectionV1:
    """只通过课程明确 surface 的同次 typed proof 形成可审计 anchor。"""
    if line_suffix not in _ORIGIN_INPUT_LINE_SUFFIXES_V1:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 采用未注册 physical line suffix")
    raw = (*encode_utf8_v1(scalars), *line_suffix)
    try:
        same_dispatch = run_public_proof_sentence_provider_vector_with_typed_proof(
            provider,
            raw,
        )
        result = same_dispatch.provider_result
        projection = same_dispatch.demo_proof_projection
        carrier = None
        if projection is not None and projection.sparse_proof_projection is not None:
            carrier = provider_origin_legacy_proof_from_same_dispatch_v1(
                projection.sparse_proof_projection)
        anchor = project_provider_origin_anchor_v1(
            provider_origin_provider_binding_from_public_provider_v1(provider),
            result,
            carrier,
        )
    except (ProviderOriginFollowupError, TypeError, ValueError) as error:
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} provider same-dispatch 无法形成") from error
    if (not verify_public_proof_sentence_provider_result(provider, raw, result)
            or not anchor.accepted):
        raise PublicProviderOriginFollowupCatalogError(
            f"{label} 不是可验证 provider ANSWER anchor")
    return anchor


def _contains_target_structure(
        origin: ProviderOriginAnchorProjectionV1,
        contrast: ProviderOriginAnchorProjectionV1,
        ) -> None:
    """确保 contrast focus 是 origin 已有的显式 binding/occurrence，不反查文本。"""
    bindings = tuple(
        item for item in origin.ordered_role_bindings
        if (item.binding_key == contrast.focus_role_binding_key
            and item.role_key == contrast.focus_role_key
            and item.filler_key == contrast.focus_filler_key))
    occurrences = tuple(
        item for item in origin.ordered_occurrences
        if (item.occurrence_key == contrast.focus_occurrence_key
            and item.semantic_object_key == contrast.focus_filler_key
            and item.start == contrast.focus_answer_start
            and item.end == contrast.focus_answer_end))
    if len(bindings) != 1 or len(occurrences) != 1:
        raise PublicProviderOriginFollowupCatalogError(
            "follow-up contrast focus 未在 origin anchor 中以同一结构出现")


def _profile_from_course_row(
        value: dict,
        form: ProviderOriginFollowupFormV1,
        provider: PublicProofSentenceProviderV1,
        *,
        row_ordinal: int,
        line_suffix: tuple[int, ...],
        ) -> ProviderOriginFollowupProfileV1:
    """以一对课程 source-bound provider routes 形成方向 profile。"""
    route_key = _ascii_u8(
        value["route_id"],
        label=f"follow-up course row[{row_ordinal}].route_id",
    )
    if line_suffix == (0x0A,):
        route_key = (*route_key, *b"-lf")
    elif line_suffix == (0x0D, 0x0A):
        route_key = (*route_key, *b"-crlf")
    elif line_suffix:
        raise PublicProviderOriginFollowupCatalogError(
            f"follow-up course row[{row_ordinal}] 采用未注册 physical line suffix")
    origin = _anchor_for_course_surface(
        provider,
        _unicode_scalars(
            value["origin_question_surface"],
            label=f"follow-up course row[{row_ordinal}].origin_question_surface",
        ),
        label=f"follow-up course row[{row_ordinal}].origin",
        line_suffix=line_suffix,
    )
    contrast = _anchor_for_course_surface(
        provider,
        _unicode_scalars(
            value["contrast_question_surface"],
            label=f"follow-up course row[{row_ordinal}].contrast_question_surface",
        ),
        label=f"follow-up course row[{row_ordinal}].contrast",
        line_suffix=line_suffix,
    )
    for name in (
            "provider_kind", "provider_identity_u8", "runtime_identity_u8",
            "catalog_record_identity_u8", "source_record_key",
            "source_ref_stable_key", "source_commitment_u8", "w03_observation_key",
            "w04_observation_key", "w05_observation_key",
            "generation_construction_key", "proposition_key", "predicate_key",
            "relation_kind_code"):
        if getattr(origin, name) != getattr(contrast, name):
            raise PublicProviderOriginFollowupCatalogError(
                f"follow-up course row[{row_ordinal}] origin/contrast {name} 漂移")
    if (origin.focus_role_binding_key == contrast.focus_role_binding_key
            or origin.focus_filler_key == contrast.focus_filler_key
            or origin.focus_occurrence_key == contrast.focus_occurrence_key):
        raise PublicProviderOriginFollowupCatalogError(
            f"follow-up course row[{row_ordinal}] 没有方向性结构差异")
    _contains_target_structure(origin, contrast)
    return ProviderOriginFollowupProfileV1(
        route_key,
        _strict_int(
            value["route_revision"],
            label=f"follow-up course row[{row_ordinal}].route_revision",
            minimum=1,
        ),
        form.form_identity_u8,
        origin.provider_kind,
        origin.provider_identity_u8,
        origin.runtime_identity_u8,
        origin.catalog_record_identity_u8,
        origin.provider_result_identity_u8,
        contrast.provider_result_identity_u8,
        origin.source_record_key,
        origin.source_ref_stable_key,
        origin.source_commitment_u8,
        origin.w03_observation_key,
        origin.w04_observation_key,
        origin.w05_observation_key,
        origin.generation_construction_key,
        origin.proposition_key,
        origin.predicate_key,
        origin.relation_kind_code,
        origin.anchor_identity_u8,
        contrast.anchor_identity_u8,
        origin.focus_role_binding_key,
        origin.focus_role_key,
        origin.focus_filler_key,
        origin.focus_occurrence_key,
        origin.focus_answer_start,
        origin.focus_answer_end,
        contrast.focus_role_binding_key,
        contrast.focus_role_key,
        contrast.focus_filler_key,
        contrast.focus_occurrence_key,
        contrast.focus_answer_start,
        contrast.focus_answer_end,
    )


def load_public_provider_origin_followup_catalog_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        provider: PublicProofSentenceProviderV1,
        ) -> ProviderOriginFollowupCatalogV1:
    """从完整公开 closure 和冻结 provider 建立 DLG-RAW-11C catalog。

    课程 row 只声明两条原始 provider question 及来源化 follow-up construct；
    答案、output bytes、role key 和 anchor identity 全部从同次 proof 自动导出，
    因而不能通过课程文本直接塞入答案表。
    """
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("follow-up catalog 需要 PublicSourcePayloadClosureV1")
    if type(provider) is not PublicProofSentenceProviderV1:
        raise TypeError("follow-up catalog 需要 PublicProofSentenceProviderV1")
    rows: list[dict] = []
    for course_ordinal, course_key in enumerate(
            PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_LOGICAL_KEYS_V1,
            start=1):
        course_payload, _course_digest = _source_payload(
            source_payload_closure,
            tuple(course_key),
            label=f"follow-up course[{course_ordinal}]",
        )
        for row_ordinal, line in enumerate(_single_lf_jsonl_lines(
                course_payload,
                label=f"follow-up course[{course_ordinal}]"), start=1):
            label = f"follow-up course[{course_ordinal}] row[{row_ordinal}]"
            try:
                parsed = parse_canonical_json_bytes(line, require_object=True)
            except (DatasetContractError, TypeError, ValueError) as error:
                raise PublicProviderOriginFollowupCatalogError(
                    f"{label} 不是 canonical JSON") from error
            row = _exact(parsed, _COURSE_FIELDS, label=label)
            if _strict_int(
                    row["catalog_schema"],
                    label=f"{label}.catalog_schema",
                ) != PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_SCHEMA_V1:
                raise PublicProviderOriginFollowupCatalogError(
                    "follow-up course schema 未注册")
            sources = row["followup_lexical_sources"]
            if type(sources) is not list or len(sources) < 2:
                raise PublicProviderOriginFollowupCatalogError(
                    f"{label} 缺少双 lexical source")
            rows.append(row)
    if not rows:
        raise PublicProviderOriginFollowupCatalogError("follow-up course 不得为空")

    forms: list[ProviderOriginFollowupFormV1] = []
    profiles: list[ProviderOriginFollowupProfileV1] = []
    for ordinal, row in enumerate(rows, start=1):
        evidence = order_items_by_nonnegative_integer_record_v1(
            tuple(_lexical_evidence(
                item,
                source_payload_closure,
                label=f"follow-up course row[{ordinal}].lexical[{index}]",
            ) for index, item in enumerate(row["followup_lexical_sources"])),
            key=ProviderOriginFollowupLexicalEvidenceV1.canonical_record,
            label=f"follow-up course row[{ordinal}] lexical evidence",
        )
        source_scalars = evidence[0].span_scalars
        form = ProviderOriginFollowupFormV1(source_scalars, evidence)
        existing = tuple(
            item for item in forms
            if compare_nonnegative_integer_records_v1(
                item.canonical_record(),
                form.canonical_record(),
                label=f"follow-up course row[{ordinal}] form equality") == 0)
        if existing:
            form = existing[0]
        else:
            forms.append(form)
        for line_suffix in _ORIGIN_INPUT_LINE_SUFFIXES_V1:
            profiles.append(_profile_from_course_row(
                row,
                form,
                provider,
                row_ordinal=ordinal,
                line_suffix=line_suffix,
            ))
    try:
        return ProviderOriginFollowupCatalogV1(
            tuple(source_payload_closure.closure_identity),
            order_items_by_nonnegative_integer_record_v1(
                tuple(forms),
                key=ProviderOriginFollowupFormV1.canonical_record,
                label="follow-up course forms",
            ),
            order_items_by_nonnegative_integer_record_v1(
                tuple(profiles),
                key=ProviderOriginFollowupProfileV1.canonical_record,
                label="follow-up course profiles",
            ),
            PROVIDER_ORIGIN_FOLLOWUP_CATALOG_SCHEMA_V1,
        )
    except ProviderOriginFollowupError as error:
        raise PublicProviderOriginFollowupCatalogError(
            "follow-up catalog 无法闭合") from error


__all__ = [
    "PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_LOGICAL_KEY_V1",
    "PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_LOGICAL_KEYS_V1",
    "PUBLIC_PROVIDER_ORIGIN_FOLLOWUP_COURSE_SCHEMA_V1",
    "PublicProviderOriginFollowupCatalogError",
    "load_public_provider_origin_followup_catalog_from_closure",
]
