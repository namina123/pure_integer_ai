"""DLG-RAW-05A 的公开课程派生多回答行为 Frame catalog。

本模块只把内容锁定的公开课程投影为既有 ``PublicFrame``。完整 JSONL 可以为了
schema/source 校验而被解析，但其中的 ``answer_plan``、已接受/拒绝表层和
response-act 标签不进入目录或 planning record，也不进入 selection/output 决策；
运行期必须仅凭 ``PublicFrameResponseActRuntimeRecipe`` 回读课程并重新建立同一
无标签 planning input。V1 catalog、RAW-01 ingress 和会话状态均不在这里修改。
"""
from __future__ import annotations

import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    language_atom_identity,
    language_branch_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
    PUBLIC_FRAME_PATTERN_SELECTION_LOWEST_VALID_V1,
    PublicFrame,
    PublicFrameCatalog,
    PublicFrameCatalogError,
    PublicFrameLexicalRoute,
    PublicFrameQuestionTemplate,
    PublicFrameResponseActRuntimeRecipe,
    PublicFrameSourceRecord,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    decode_utf8_v1,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedAnswerCourseError,
    GroundedAnswerEpisode,
    read_grounded_answer_episodes_from_payload,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningInput,
    PublicResponseActPlanningBuild,
    compile_public_response_act_planning,
    public_response_act_planning_input_from_episode,
)


PUBLIC_RESPONSE_ACT_FRAME_CATALOG_SCHEMA_V2 = 2
PUBLIC_RESPONSE_ACT_DERIVED_FRAME_CATALOG_SCHEMA_V3 = 3
PUBLIC_RESPONSE_ACT_CONTEXTUAL_FRAME_CATALOG_SCHEMA_V4 = 4
PUBLIC_RESPONSE_ACT_CATALOG_MERGE_RECORD_V1 = 1
PUBLIC_RESPONSE_ACT_PATTERN_SELECTION_LOWEST_VALID_V1 = (
    PUBLIC_FRAME_PATTERN_SELECTION_LOWEST_VALID_V1)

_NAMESPACE = 65001
_LEXICAL_SOURCE_KIND = 65052
_LANGUAGE_BRANCH = language_branch_identity((_NAMESPACE, 52, 2))
_REPRESENTATION_FAMILY = (_NAMESPACE, 52, 3)
_COURSE_ATTRIBUTION = "Pure Integer AI authored public grounded answer course"
_MERGE_DOMAIN = "dlg.raw.public.response.act.catalog.merge.v1"

_MANIFEST_FIELDS = frozenset({
    "catalog_schema",
    "course_raw_sha256",
    "course_relative_path",
    "episode_id",
    "frame_key",
    "lexical_source_a",
    "lexical_source_b",
    "output_max_bytes",
})
_DERIVED_MANIFEST_FIELDS = frozenset({
    *_MANIFEST_FIELDS,
    "question_derivation",
})
_CONTEXTUAL_DERIVED_MANIFEST_FIELDS = frozenset({
    *_DERIVED_MANIFEST_FIELDS,
    "context_derivation",
})
_LEXICAL_SOURCE_FIELDS = frozenset({
    "attribution",
    "license_id",
    "raw_sha256",
    "relative_path",
})
_QUESTION_DERIVATION_EXACT_FIELDS = frozenset({"kind"})
_QUESTION_DERIVATION_OMIT_FIELDS = frozenset({"kind", "omitted_utf8_hex"})
_QUESTION_DERIVATION_EXACT_V1 = "COURSE_QUESTION_EXACT_V1"
_QUESTION_DERIVATION_OMIT_ONCE_V1 = "OMIT_UTF8_SUBSTRING_ONCE_V1"
_CONTEXT_DERIVATION_SELF_QUESTION_TARGET_FIELDS = frozenset({"kind"})
_CONTEXT_DERIVATION_SELF_QUESTION_TARGET_V1 = "SELF_QUESTION_TARGET_V1"

# DLG-RAW-07 只允许从 payload closure 读取这三份派生 catalog manifest。
# 名称是协议 logical key，不是宿主文件系统路径。
PUBLIC_RESPONSE_ACT_CATALOG_LOGICAL_KEYS_V1 = (
    b"data/ph2/dlg_raw_public_contextual_ellipsis_frame_v4.jsonl.sample",
    b"data/ph2/dlg_raw_public_derived_frame_v3.jsonl.sample",
    b"data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample",
)


# object-model: exception; interop=DLG-RAW-05A
class PublicResponseActCatalogError(ValueError):
    """公开课程、词汇来源或无标签派生 record 无法闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把变长整数向量追加为显式长度定界的可迁移 record 段。"""
    result.extend((len(value), *value))


def _u8_record(value: str, *, label: str) -> tuple[int, ...]:
    """把文本冻结为 UTF-8 byte vector，拒绝空值和隐式编码。"""
    if (not isinstance(value, str) or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"):
        raise PublicResponseActCatalogError(f"{label} 必须是无首尾空白文本")
    scalars = tuple(ord(character) for character in value)
    try:
        return encode_utf8_v1(scalars)
    except (TypeError, ValueError) as error:
        raise PublicResponseActCatalogError(f"{label} 含非法 Unicode scalar") from error


def _fingerprint_text(*values: str, domain: str) -> tuple[int, ...]:
    """使用显式 UTF-8 length-prefix 形成跨语言确定性 identity key。"""
    encoded: list[int] = []
    for value in values:
        payload = _u8_record(value, label="identity text")
        _pack(encoded, payload)
    return integer_tuple_fingerprint(tuple(encoded), domain=domain)


def _positive_identifier(values: tuple[int, ...], *, domain: str) -> int:
    """从规范指纹前八个 digest byte 以明确大端序生成正整数。"""
    fingerprint = integer_tuple_fingerprint(values, domain=domain)
    if len(fingerprint) != 34 or any(
            type(item) is not int or item < 0 or item > 255
            for item in fingerprint[2:]):
        raise PublicResponseActCatalogError("整数指纹实现未返回固定 SHA-256 record")
    result = 0
    for value in fingerprint[2:10]:
        result = (result << 8) | value
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    """拒绝缺失、未知或由宿主 dict 默认值补出的 manifest 字段。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicResponseActCatalogError(f"{label} 字段集合漂移")
    return value


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    """读取严格整数，不让 bool 或浮点通过公开协议边界。"""
    if type(value) is not int or value < minimum:
        raise PublicResponseActCatalogError(f"{label} 必须是不小于 {minimum} 的严格整数")
    return value


def _ascii_id(value: Any, *, label: str) -> str:
    """校验 transport 内的稳定 ASCII id，不接受路径或空白歧义。"""
    if (not isinstance(value, str) or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"
            or any(ord(character) > 0x7F for character in value)
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "abcdefghijklmnopqrstuvwxyz0123456789-_.:"
                   for character in value)):
        raise PublicResponseActCatalogError(f"{label} 不是稳定 ASCII id")
    return value


def _hex_bytes(value: Any, *, label: str, size: int) -> tuple[int, ...]:
    """恢复严格小写十六进制 SHA-256，不容许大小写或长度变体。"""
    if (not isinstance(value, str) or len(value) != size * 2
            or any(character not in "0123456789abcdef" for character in value)):
        raise PublicResponseActCatalogError(f"{label} 不是固定长度小写十六进制")
    try:
        result = tuple(bytes.fromhex(value))
    except ValueError as error:
        raise PublicResponseActCatalogError(f"{label} 十六进制损坏") from error
    if len(result) != size:
        raise PublicResponseActCatalogError(f"{label} 长度漂移")
    return result


def _logical_payload_key(
        value: Any,
        *,
        label: str,
        ) -> tuple[str, bytes]:
    """把 manifest 的 ASCII logical key 规范为 closure 查询键。

    这里仅校验逻辑资源名的有限 transport 语法；实际资源是否存在、是否完整以及
    raw bytes 身份均由已经构造的 payload closure 决定。
    """
    if (not isinstance(value, str) or not value
            or any(ord(character) < 0x21 or ord(character) > 0x7E
                   for character in value)):
        raise PublicResponseActCatalogError(f"{label} 不是规范 ASCII logical key")
    parts = tuple(value.split("/"))
    if (len(parts) != 3 or parts[:2] != ("data", "ph2")
            or any(not part or part in {".", ".."} for part in parts)
            or "\\" in value):
        raise PublicResponseActCatalogError(f"{label} 越出 data/ph2 logical key")
    return value, bytes(ord(character) for character in value)


def _payload_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        logical_key: bytes,
        expected_sha256: tuple[int, ...] | None,
        *,
        label: str,
        ) -> bytes:
    """由闭包取得一份 raw payload，并逐项重核 key、长度和 SHA-256。

    closure 自身已经完成 27 项 registry 的整体核验；此处仍逐调用复核 record，
    使 manifest 声明和 frame source record 无法借缓存或同名条目绕过内容锁。
    """
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise PublicResponseActCatalogError("public source payload closure 类型错误")
    if (type(logical_key) is not bytes or not logical_key
            or any(value < 0x21 or value > 0x7E for value in logical_key)):
        raise PublicResponseActCatalogError(f"{label} logical key 非法")
    try:
        record = source_payload_closure.record_for(logical_key)
        payload = source_payload_closure.payload_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise PublicResponseActCatalogError(
            f"{label} 不在 public source payload closure 内") from error
    digest = tuple(public_source_payload_sha256_v1(payload))
    if (record.logical_key != logical_key
            or record.raw_payload != payload
            or record.payload_length != len(payload)
            or tuple(record.raw_sha256) != digest):
        raise PublicResponseActCatalogError(f"{label} payload record 漂移")
    if expected_sha256 is not None and digest != expected_sha256:
        raise PublicResponseActCatalogError(f"{label} raw SHA-256 漂移")
    return payload


def _decode_scalars(payload: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证 source span 是可无损回写的 UTF-8 scalar 序。"""
    scalars = decode_utf8_v1(payload)
    if scalars is None:
        raise PublicResponseActCatalogError(f"{label} 不是有效 UTF-8")
    if encode_utf8_v1(scalars) != payload:
        raise PublicResponseActCatalogError(f"{label} UTF-8 scalar readback 漂移")
    return scalars


def _unique_span(
        payload: bytes,
        needle: bytes,
        *,
        label: str,
        ) -> tuple[int, int]:
    """定位唯一 raw span，避免以首次匹配静默吞掉词汇歧义。"""
    if not needle:
        raise PublicResponseActCatalogError(f"{label} 不得为空")
    start = payload.find(needle)
    if start < 0 or payload.find(needle, start + 1) >= 0:
        raise PublicResponseActCatalogError(f"{label} 在公开 source 中缺失或不唯一")
    return start, start + len(needle)


def _lexical_source_ref(
        relative_path: str,
        raw_sha256: tuple[int, ...],
        ) -> SourceRef:
    """从公开物理词汇 source identity 建立稳定 SourceRef。"""
    path = _u8_record(relative_path, label="lexical relative path")
    source_id = _positive_identifier(
        (len(path), *path, len(raw_sha256), *raw_sha256),
        domain="dlg.raw.public.response.act.lexical.source.v1",
    )
    return SourceRef(
        _LEXICAL_SOURCE_KIND,
        source_id,
        1,
        GLOBAL_OWNER_SCOPE,
        # SourceRef 的版本是协议状态的一部分；这里显式使用全零冻结版本，
        # 不从宿主环境读取版本或时间。
        VersionBundle(),
    )


def _lexical_observation(
        raw: Any,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        question_bytes: tuple[int, ...],
        record_id: str,
        ) -> PublicFrameSourceRecord:
    """回读独立词汇观察，并将整句的唯一 raw span 绑定为来源证据。"""
    value = _exact(raw, _LEXICAL_SOURCE_FIELDS, label=record_id)
    relative_path, logical_key = _logical_payload_key(
        value["relative_path"], label=f"{record_id}.relative_path")
    digest = _hex_bytes(value["raw_sha256"], label=f"{record_id}.raw_sha256",
                        size=32)
    license_id = value["license_id"]
    attribution = value["attribution"]
    if license_id != "CC0-1.0":
        raise PublicResponseActCatalogError(f"{record_id}.license 必须是 CC0-1.0")
    if (not isinstance(attribution, str) or not attribution
            or attribution[0] in " \t\r\n" or attribution[-1] in " \t\r\n"):
        raise PublicResponseActCatalogError(f"{record_id}.attribution 非法")
    payload = _payload_from_closure(
        source_payload_closure,
        logical_key,
        digest,
        label=record_id,
    )
    question = bytes(question_bytes)
    span = _unique_span(payload, question, label=f"{record_id}.surface")
    scalars = _decode_scalars(question_bytes, label=f"{record_id}.surface")
    return PublicFrameSourceRecord(
        record_id,
        _lexical_source_ref(relative_path, digest),
        relative_path,
        digest,
        license_id,
        attribution,
        span,
        question_bytes,
        scalars,
    )


def _course_source_record(
        record_id: str,
        *,
        source: SourceRef,
        relative_path: str,
        raw_sha256: tuple[int, ...],
        course_payload: bytes,
        observation_bytes: tuple[int, ...],
        ) -> PublicFrameSourceRecord:
    """绑定课程内唯一、非标签的原始 observation span，不复制整份课程。"""
    span = _unique_span(
        course_payload,
        bytes(observation_bytes),
        label=f"{record_id}.course observation",
    )
    return PublicFrameSourceRecord(
        record_id,
        source,
        relative_path,
        raw_sha256,
        "CC0-1.0",
        _COURSE_ATTRIBUTION,
        span,
        observation_bytes,
        _decode_scalars(observation_bytes, label=f"{record_id}.course"),
    )


def _course_ascii_field_observation(
        field: str,
        value: str,
        *,
        label: str,
        ) -> tuple[int, ...]:
    """形成 canonical JSON 内唯一的非标签 ASCII 字段观察字节。"""
    identifier = _ascii_id(value, label=label)
    text = f'"{field}":"{identifier}"'
    result = tuple(ord(character) for character in text)
    if any(value > 0x7F for value in result):
        raise PublicResponseActCatalogError(f"{label} 不是 ASCII 字段观察")
    return result


def _episode_for_id(
        episodes: tuple[GroundedAnswerEpisode, ...],
        episode_id: str,
        ) -> GroundedAnswerEpisode:
    """从已经完整解析的 TRAIN 课程按唯一公开 episode identity 选择。"""
    matches = tuple(item for item in episodes if item.episode_id == episode_id)
    if len(matches) != 1:
        raise PublicResponseActCatalogError("课程中找不到唯一 episode_id")
    return matches[0]


def _question_bytes_from_manifest(
        episode: GroundedAnswerEpisode,
        manifest: dict[str, Any],
        *,
        catalog_schema: int,
        ) -> tuple[int, ...]:
    """从公开 question surface 按冻结规则派生本 frame 的 raw input。

    V3/V4 只支持一次精确 omission，避免把 Python 字符串改写器变成未审计的
    问句生成器。完整课程可为 schema/source 校验被解析；本派生规则和后续
    planning/selection/output 决策均不消费 answer plan 或 accepted/rejected surface。
    """
    source = _u8_record(
        episode.question.question_surface,
        label="课程 question surface",
    )
    if catalog_schema == PUBLIC_RESPONSE_ACT_FRAME_CATALOG_SCHEMA_V2:
        return source
    derivation = manifest["question_derivation"]
    if not isinstance(derivation, dict):
        raise PublicResponseActCatalogError("question_derivation 必须是 object")
    kind = derivation.get("kind")
    if kind == _QUESTION_DERIVATION_EXACT_V1:
        _exact(derivation, _QUESTION_DERIVATION_EXACT_FIELDS,
               label="question_derivation")
        return source
    if kind != _QUESTION_DERIVATION_OMIT_ONCE_V1:
        raise PublicResponseActCatalogError("question_derivation kind 未注册")
    _exact(derivation, _QUESTION_DERIVATION_OMIT_FIELDS,
           label="question_derivation")
    omitted_hex = derivation["omitted_utf8_hex"]
    if (not isinstance(omitted_hex, str) or not omitted_hex
            or len(omitted_hex) % 2
            or any(character not in "0123456789abcdef"
                   for character in omitted_hex)):
        raise PublicResponseActCatalogError(
            "question_derivation.omitted_utf8_hex 非法")
    omitted = tuple(
        int(omitted_hex[index:index + 2], 16)
        for index in range(0, len(omitted_hex), 2))
    if not omitted:
        raise PublicResponseActCatalogError("question_derivation omission 不得为空")
    start = bytes(source).find(bytes(omitted))
    if start < 0 or bytes(source).find(bytes(omitted), start + 1) >= 0:
        raise PublicResponseActCatalogError(
            "question_derivation omission 在课程 question 中缺失或不唯一")
    derived = source[:start] + source[start + len(omitted):]
    if not derived or _decode_scalars(derived, label="派生 question surface") is None:
        raise PublicResponseActCatalogError("question_derivation 产生非法 question")
    return derived


def _verify_frame_source_records(
        frame: PublicFrame,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> tuple[tuple[str, bytes], ...]:
    """逐轮从 closure 回读 frame source，并复核 SHA、span 与 UTF-8 scalar 映射。

    返回值只是本调用内的 logical bytes cache，不属于 catalog、session 或持久状态。
    线性查找保持 source 数量有界时的显式、可重放读取顺序。
    """
    payloads: list[tuple[str, tuple[int, ...], bytes]] = []
    for source in frame.source_records:
        relative_path, logical_key = _logical_payload_key(
            source.relative_path,
            label=f"runtime source {source.record_id}.relative_path",
        )
        payload = None
        for known_path, known_sha256, known_payload in payloads:
            if known_path == relative_path and known_sha256 == source.raw_sha256:
                payload = known_payload
                break
        if payload is None:
            payload = _payload_from_closure(
                source_payload_closure,
                logical_key,
                source.raw_sha256,
                label=f"runtime source {source.record_id}",
            )
            payloads.append((relative_path, source.raw_sha256, payload))
        if (len(source.span) != 2 or source.span[0] < 0
                or source.span[0] >= source.span[1]
                or source.span[1] > len(payload)):
            raise PublicResponseActCatalogError("runtime source span 越界")
        if tuple(payload[source.span[0]:source.span[1]]) != source.span_bytes:
            raise PublicResponseActCatalogError("runtime source span bytes 漂移")
        if (_decode_scalars(source.span_bytes,
                            label=f"runtime source {source.record_id}.span")
                != source.span_scalars):
            raise PublicResponseActCatalogError("runtime source span scalar 漂移")
    return tuple((path, payload) for path, _sha256, payload in payloads)


def _payload_for_relative_path(
        payloads: tuple[tuple[str, bytes], ...],
        relative_path: str,
        ) -> bytes:
    """从本轮已验证 source bytes 取回课程，不允许路径 fallback。"""
    matches = tuple(payload for path, payload in payloads if path == relative_path)
    if len(matches) != 1:
        raise PublicResponseActCatalogError("runtime course 不在 frame source records 内")
    return matches[0]


def _question_template(
        planning_input: GroundedResponseActPlanningInput,
        build: PublicResponseActPlanningBuild,
        *,
        frame_key: str,
        ) -> PublicFrameQuestionTemplate:
    """从无标签 planning build 机械形成完整十字段 QuestionRequest 模板。"""
    input_key = planning_input.canonical_record()
    intent_key = _fingerprint_text(
        planning_input.typed_intent,
        domain="dlg.raw.public.response.act.intent.v1",
    )
    trace_key = integer_tuple_fingerprint(
        input_key,
        domain="dlg.raw.public.response.act.question.trace.v1",
    )
    authorized = tuple(sorted({
        item.candidate.proposition for item in build.candidate_bindings
    }, key=lambda item: item.stable_key()))
    if not authorized:
        authorized = (build.planning.goal.proposition,)
    if build.planning.goal.proposition not in authorized:
        raise PublicResponseActCatalogError("label-free planning 未授权其自身 goal target")
    return PublicFrameQuestionTemplate(
        minimal_instruction_identity((_NAMESPACE, 52, 10)),
        minimal_instruction_identity((_NAMESPACE, 52, 11, *intent_key)),
        build.planning.goal.goal_kind,
        build.planning.goal.proposition,
        build.planning.goal.required,
        query_scope(
            planning_input.evidence_scope_id,
            parent=document_scope(build.aggregate_source),
        ),
        build.response_scope,
        (_NAMESPACE, 52, 12, *trace_key),
        _LANGUAGE_BRANCH,
        authorized,
    )


def _context_binding_from_manifest(
        manifest: dict[str, Any],
        question: PublicFrameQuestionTemplate,
        *,
        catalog_schema: int,
        ) -> tuple[int, tuple[int, ...]]:
    """从冻结的上下文派生规则形成 frame gate，不接受预写 target key。"""
    if catalog_schema in {
            PUBLIC_RESPONSE_ACT_FRAME_CATALOG_SCHEMA_V2,
            PUBLIC_RESPONSE_ACT_DERIVED_FRAME_CATALOG_SCHEMA_V3}:
        return PUBLIC_FRAME_CONTEXT_NONE, ()
    if catalog_schema != PUBLIC_RESPONSE_ACT_CONTEXTUAL_FRAME_CATALOG_SCHEMA_V4:
        raise PublicResponseActCatalogError("context derivation catalog schema 未注册")
    derivation = manifest["context_derivation"]
    if not isinstance(derivation, dict):
        raise PublicResponseActCatalogError("context_derivation 必须是 object")
    _exact(
        derivation,
        _CONTEXT_DERIVATION_SELF_QUESTION_TARGET_FIELDS,
        label="context_derivation",
    )
    if derivation["kind"] != _CONTEXT_DERIVATION_SELF_QUESTION_TARGET_V1:
        raise PublicResponseActCatalogError("context_derivation kind 未注册")
    target_key = question.target.stable_key()
    if not target_key:
        raise PublicResponseActCatalogError(
            "context_derivation 无法形成 self question target")
    return PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR, target_key


def _frame_from_manifest(
        raw: Any,
        *,
        raw_line_sha256: tuple[int, ...],
        source_payload_closure: PublicSourcePayloadClosureV1,
        ordinal: int,
        ) -> PublicFrame:
    """从一条 V2 manifest 记录和当前 raw 课程重派生完整 public frame。"""
    if not isinstance(raw, dict):
        raise PublicResponseActCatalogError("response-act frame 字段集合漂移")
    catalog_schema = _strict_int(
        raw.get("catalog_schema"), label="catalog_schema", minimum=1)
    if catalog_schema == PUBLIC_RESPONSE_ACT_FRAME_CATALOG_SCHEMA_V2:
        manifest = _exact(raw, _MANIFEST_FIELDS, label="response-act frame")
    elif catalog_schema == PUBLIC_RESPONSE_ACT_DERIVED_FRAME_CATALOG_SCHEMA_V3:
        manifest = _exact(raw, _DERIVED_MANIFEST_FIELDS,
                          label="response-act frame")
    elif catalog_schema == PUBLIC_RESPONSE_ACT_CONTEXTUAL_FRAME_CATALOG_SCHEMA_V4:
        manifest = _exact(raw, _CONTEXTUAL_DERIVED_MANIFEST_FIELDS,
                          label="response-act frame")
    else:
        raise PublicResponseActCatalogError("response-act catalog schema 未注册")
    frame_key = _ascii_id(manifest["frame_key"], label="frame_key")
    episode_id = _ascii_id(manifest["episode_id"], label="episode_id")
    output_max_bytes = _strict_int(
        manifest["output_max_bytes"], label="output_max_bytes", minimum=1)
    course_relative_path, course_logical_key = _logical_payload_key(
        manifest["course_relative_path"],
        label="course_relative_path",
    )
    course_sha256 = _hex_bytes(
        manifest["course_raw_sha256"], label="course_raw_sha256", size=32)
    course_payload = _payload_from_closure(
        source_payload_closure,
        course_logical_key,
        course_sha256,
        label="公开课程",
    )
    try:
        episodes = read_grounded_answer_episodes_from_payload(
            course_payload, train_only=True)
    except GroundedAnswerCourseError as error:
        raise PublicResponseActCatalogError("公开课程结构或 train split 漂移") from error
    episode = _episode_for_id(episodes, episode_id)

    # 完整课程对象可能含标签字段以供 schema/source 校验；这个 label-free
    # projection 不把它们带入 planning record、selection 或 output 决策。
    planning_input = public_response_act_planning_input_from_episode(episode)
    try:
        build = compile_public_response_act_planning(
            planning_input, _LANGUAGE_BRANCH)
    except (TypeError, ValueError) as error:
        raise PublicResponseActCatalogError("公开课程无法形成无标签 planning") from error
    question_bytes = _question_bytes_from_manifest(
        episode,
        manifest,
        catalog_schema=catalog_schema,
    )
    surface_scalars = _decode_scalars(question_bytes, label="课程 question surface")
    if not question_bytes:
        raise PublicResponseActCatalogError("课程 question surface 为空")

    course_record_id = f"course-{ordinal}"
    source_records: list[PublicFrameSourceRecord] = [
        _course_source_record(
            course_record_id,
            source=build.aggregate_source,
            relative_path=course_relative_path,
            raw_sha256=course_sha256,
            course_payload=course_payload,
            observation_bytes=(
                _course_ascii_field_observation(
                    "episode_id", episode_id, label="course episode_id")
                if catalog_schema == PUBLIC_RESPONSE_ACT_FRAME_CATALOG_SCHEMA_V2 else
                _u8_record(
                    '"question_surface":"'
                    + episode.question.question_surface + '"',
                    label="课程 source question surface")
            ),
        ),
    ]
    evidence_record_ids = []
    for source_ordinal, binding in enumerate(build.source_bindings, start=1):
        record_id = f"evidence-{ordinal}-{source_ordinal}"
        evidence_ids = tuple(sorted(
            item.evidence_id for item in planning_input.evidence
            if item.source_id == binding.source_id))
        if not evidence_ids:
            raise PublicResponseActCatalogError(
                "public planning source 缺少可观察 Evidence")
        source_records.append(_course_source_record(
            record_id,
            source=binding.source,
            relative_path=course_relative_path,
            raw_sha256=course_sha256,
            course_payload=course_payload,
            observation_bytes=_course_ascii_field_observation(
                "evidence_id", evidence_ids[0], label="course evidence_id"),
        ))
        evidence_record_ids.append(record_id)
    lexical_a = _lexical_observation(
        manifest["lexical_source_a"],
        source_payload_closure=source_payload_closure,
        question_bytes=question_bytes,
        record_id=f"lexical-a-{ordinal}",
    )
    lexical_b = _lexical_observation(
        manifest["lexical_source_b"],
        source_payload_closure=source_payload_closure,
        question_bytes=question_bytes,
        record_id=f"lexical-b-{ordinal}",
    )
    if lexical_a.source == lexical_b.source:
        raise PublicResponseActCatalogError("V2 lexical source 必须是两个不同 SourceRef")
    source_records.extend((lexical_a, lexical_b))
    source_records_tuple = tuple(sorted(
        source_records, key=lambda item: item.source.stable_key()))

    atom_key = _fingerprint_text(
        frame_key,
        "".join(chr(value) for value in surface_scalars),
        domain="dlg.raw.public.response.act.language.atom.v1",
    )
    atom = language_atom_identity(
        _LANGUAGE_BRANCH, (_NAMESPACE, 52, 20, *atom_key))
    route = PublicFrameLexicalRoute(
        0,
        (0, len(surface_scalars)),
        _LANGUAGE_BRANCH,
        representation_identity(
            _REPRESENTATION_FAMILY,
            (0, len(surface_scalars), *surface_scalars),
        ),
        atom,
        tuple(sorted((lexical_a, lexical_b),
                     key=lambda item: item.source.stable_key())),
        surface_scalars,
    )
    construction_key = _fingerprint_text(
        frame_key,
        domain="dlg.raw.public.response.act.construction.v1",
    )
    recipe_key = _fingerprint_text(
        frame_key,
        domain="dlg.raw.public.response.act.recipe.v1",
    )
    recipe = PublicFrameResponseActRuntimeRecipe(
        minimal_instruction_identity((_NAMESPACE, 52, 30, *recipe_key)),
        course_relative_path,
        course_sha256,
        episode_id,
        planning_input.canonical_record(),
        course_record_id,
        tuple(sorted(evidence_record_ids)),
        output_max_bytes,
        1,
        PUBLIC_RESPONSE_ACT_PATTERN_SELECTION_LOWEST_VALID_V1,
    )
    question = _question_template(planning_input, build, frame_key=frame_key)
    context_requirement, context_target_key = _context_binding_from_manifest(
        manifest,
        question,
        catalog_schema=catalog_schema,
    )
    return PublicFrame(
        frame_key,
        raw_line_sha256,
        question_bytes,
        surface_scalars,
        source_records_tuple,
        (route,),
        structure_concept_identity((_NAMESPACE, 52, 31, *construction_key)),
        (atom,),
        question,
        recipe,
        context_requirement,
        context_target_key,
    )


def derive_public_response_act_frame_from_manifest_record(
        raw: Any,
        *,
        raw_line_sha256: tuple[int, ...],
        source_payload_closure: PublicSourcePayloadClosureV1,
        ordinal: int,
        ) -> PublicFrame:
    """由已解析 V2 transport record 重建一条可审计的派生 frame。

    这是给同一公开课程派生器复用的纯 closure helper；调用方仍须自行冻结
    manifest logical key、raw line SHA 与后续 recipe family，不能把它当成
    任意物理资源入口。
    """
    if (type(raw_line_sha256) is not tuple or len(raw_line_sha256) != 32
            or any(type(item) is not int or item < 0 or item > 255
                   for item in raw_line_sha256)
            or type(ordinal) is not int or ordinal < 1):
        raise PublicResponseActCatalogError("派生 frame transport identity 非法")
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("派生 frame 需要完整 public payload closure")
    return _frame_from_manifest(
        raw,
        raw_line_sha256=raw_line_sha256,
        source_payload_closure=source_payload_closure,
        ordinal=ordinal,
    )


def load_public_response_act_frame_catalog_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        catalog_logical_key: bytes,
        ) -> PublicFrameCatalog:
    """从 logical payload closure 加载 V2/V3/V4 public Frame catalog。

    ``catalog_logical_key`` 必须是本协议登记的三份派生 manifest 之一。它和所有
    manifest 声明 source 都通过相同 closure 查询，因此 catalog identity 不依赖
    安装位置、读取顺序或任何宿主文件系统细节。
    """
    if catalog_logical_key not in PUBLIC_RESPONSE_ACT_CATALOG_LOGICAL_KEYS_V1:
        raise PublicResponseActCatalogError("response-act catalog logical key 未注册")
    payload = _payload_from_closure(
        source_payload_closure,
        catalog_logical_key,
        None,
        label="response-act catalog manifest",
    )
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise PublicResponseActCatalogError("V2 catalog JSONL 换行非法")
    lines = payload[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise PublicResponseActCatalogError("V2 catalog 含空记录")
    frames = []
    for ordinal, line in enumerate(lines, start=1):
        try:
            manifest = parse_canonical_json_bytes(line, require_object=True)
        except DatasetContractError as error:
            raise PublicResponseActCatalogError("V2 manifest 不是 canonical JSON") from error
        frames.append(_frame_from_manifest(
            manifest,
            raw_line_sha256=tuple(hashlib.sha256(line).digest()),
            source_payload_closure=source_payload_closure,
            ordinal=ordinal,
        ))
    frame_keys = tuple(item.frame_key for item in frames)
    episode_ids = tuple(item.recipe.episode_id for item in frames)
    surfaces = tuple(item.surface_scalars for item in frames)
    if len(set(frame_keys)) != len(frame_keys):
        raise PublicResponseActCatalogError("V2 manifest frame_key 重复")
    if len(set(episode_ids)) != len(episode_ids):
        raise PublicResponseActCatalogError("V2 manifest episode_id 重复")
    if len(set(surfaces)) != len(surfaces):
        raise PublicResponseActCatalogError("V2 manifest exact scalar surface 重复")
    return PublicFrameCatalog(
        tuple(hashlib.sha256(payload).digest()),
        tuple(sorted(frames, key=PublicFrame.canonical_record)),
    )


def materialize_public_response_act_planning_from_closure(
        frame: PublicFrame,
        request: QuestionRequest,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicResponseActPlanningBuild:
    """按 response-act frame 回读公开课程并重建已锁定的无标签 planning。

    本函数不从 ``planning_input_record`` 反解对象，也不返回课程 episode，因此调用方
    无法用返回值消费 ``answer_plan`` 或任何训练答案表层。完整 JSONL 可在下游为
    schema/source 校验被解析；冻结 record、重派生 input、selection 和 output 决策
    仍不消费这些标签。冻结 record 只承担本轮重派生输入的相等性门；实际
    ``GenerationPlanningRequest`` 一律来自当前受 SHA 锁定的 public course projection。
    """
    if not isinstance(frame, PublicFrame) or not isinstance(request, QuestionRequest):
        raise TypeError("response-act materialization 需要 PublicFrame 和 QuestionRequest")
    if not isinstance(frame.recipe, PublicFrameResponseActRuntimeRecipe):
        raise PublicResponseActCatalogError("frame 不是 V2 response-act recipe")
    prefix = frame.question.trace_prefix
    if (request.query_kind != frame.question.query_kind
            or request.intent != frame.question.intent
            or request.goal_kind != frame.question.goal_kind
            or request.target != frame.question.target
            or request.required != frame.question.required
            or request.evidence_scope != frame.question.evidence_scope
            or request.response_scope != frame.question.response_scope
            or request.target_branch != frame.question.target_branch
            or request.authorized_candidate_targets
            != frame.question.authorized_candidate_targets
            or request.trace[:len(prefix)] != prefix
            or len(request.trace) == len(prefix)):
        raise PublicResponseActCatalogError(
            "materialization 收到漂移完整 QuestionRequest")
    recipe = frame.recipe
    course_relative_path, _course_logical_key = _logical_payload_key(
        recipe.course_relative_path,
        label="runtime course_relative_path",
    )
    if course_relative_path != recipe.course_relative_path:
        raise PublicResponseActCatalogError("runtime course logical key 规范化漂移")
    source_payloads = _verify_frame_source_records(
        frame,
        source_payload_closure=source_payload_closure,
    )
    payload = _payload_for_relative_path(source_payloads, course_relative_path)
    if tuple(hashlib.sha256(payload).digest()) != recipe.course_raw_sha256:
        raise PublicResponseActCatalogError("runtime 公开课程 raw SHA-256 漂移")
    try:
        episodes = read_grounded_answer_episodes_from_payload(
            payload, train_only=True)
    except GroundedAnswerCourseError as error:
        raise PublicResponseActCatalogError("runtime 公开课程结构或 train split 漂移") from error
    episode = _episode_for_id(episodes, recipe.episode_id)
    planning_input = public_response_act_planning_input_from_episode(episode)
    if planning_input.canonical_record() != recipe.planning_input_record:
        raise PublicResponseActCatalogError(
            "runtime 课程重派生 planning input 与 V2 recipe 漂移")
    try:
        build = compile_public_response_act_planning(
            planning_input, frame.question.target_branch)
    except (TypeError, ValueError) as error:
        raise PublicResponseActCatalogError("runtime 无法形成无标签 planning") from error
    if (build.planning.goal.goal_kind != request.goal_kind
            or build.planning.goal.proposition != request.target
            or build.planning.goal.required != request.required
            or build.planning.goal.source != request.source
            or build.response_scope != request.response_scope
            or build.planning.goal.target_branch != request.target_branch):
        raise PublicResponseActCatalogError("runtime 无标签 planning 与 QuestionRequest 漂移")
    return build


def _merged_catalog_sha256(catalogs: tuple[PublicFrameCatalog, ...]) -> tuple[int, ...]:
    """把已验证子 catalog 的完整 canonical record 以固定 byte framing 汇成来源身份。"""
    digest = hashlib.sha256()
    domain = _u8_record(_MERGE_DOMAIN, label="merge domain")
    digest.update(len(domain).to_bytes(8, "big"))
    digest.update(bytes(domain))
    digest.update(PUBLIC_RESPONSE_ACT_CATALOG_MERGE_RECORD_V1.to_bytes(8, "big"))
    digest.update(len(catalogs).to_bytes(8, "big"))
    for catalog in catalogs:
        fingerprint = integer_tuple_fingerprint(
            catalog.canonical_record(), domain=_MERGE_DOMAIN)
        digest.update(bytes(catalog.source_sha256))
        digest.update(bytes(fingerprint[2:]))
    return tuple(digest.digest())


def merge_public_frame_catalogs(*catalogs: PublicFrameCatalog) -> PublicFrameCatalog:
    """合并公开组件并保证每个 context frame 有可达的 NONE target anchor。"""
    if not catalogs or any(not isinstance(item, PublicFrameCatalog)
                           for item in catalogs):
        raise TypeError("catalog merge 需要至少一个 PublicFrameCatalog")
    ordered = tuple(sorted(catalogs, key=PublicFrameCatalog.canonical_record))
    frames = tuple(frame for catalog in ordered for frame in catalog.frames)
    frame_keys = tuple(frame.frame_key for frame in frames)
    if len(set(frame_keys)) != len(frame_keys):
        raise PublicResponseActCatalogError("合并 catalog 的 frame_key 重复")
    surfaces = tuple(frame.surface_scalars for frame in frames)
    if len(set(surfaces)) != len(surfaces):
        raise PublicResponseActCatalogError("合并 catalog 产生 exact scalar ambiguity")
    cold_start_targets = {
        frame.question.target.stable_key()
        for frame in frames
        if frame.context_requirement == PUBLIC_FRAME_CONTEXT_NONE
    }
    if any(
            frame.context_requirement == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR
            and frame.context_target_key not in cold_start_targets
            for frame in frames):
        raise PublicResponseActCatalogError(
            "合并 catalog 的 TARGET_ANCHOR 没有可达 NONE target")
    source_identity: dict[tuple[int, ...], tuple[str, tuple[int, ...], str, str]] = {}
    for frame in frames:
        for source in frame.source_records:
            metadata = (
                source.relative_path,
                source.raw_sha256,
                source.license_id,
                source.attribution,
            )
            previous = source_identity.get(source.source.stable_key())
            if previous is not None and previous != metadata:
                raise PublicResponseActCatalogError(
                    "合并 catalog 的同一 SourceRef 元数据冲突")
            source_identity[source.source.stable_key()] = metadata
    return PublicFrameCatalog(
        _merged_catalog_sha256(ordered),
        tuple(sorted(frames, key=PublicFrame.canonical_record)),
    )


__all__ = [
    "PUBLIC_RESPONSE_ACT_CATALOG_MERGE_RECORD_V1",
    "PUBLIC_RESPONSE_ACT_CATALOG_LOGICAL_KEYS_V1",
    "PUBLIC_RESPONSE_ACT_FRAME_CATALOG_SCHEMA_V2",
    "PUBLIC_RESPONSE_ACT_PATTERN_SELECTION_LOWEST_VALID_V1",
    "PUBLIC_RESPONSE_ACT_DERIVED_FRAME_CATALOG_SCHEMA_V3",
    "PUBLIC_RESPONSE_ACT_CONTEXTUAL_FRAME_CATALOG_SCHEMA_V4",
    "PublicResponseActCatalogError",
    "derive_public_response_act_frame_from_manifest_record",
    "load_public_response_act_frame_catalog_from_closure",
    "materialize_public_response_act_planning_from_closure",
    "merge_public_frame_catalogs",
]
