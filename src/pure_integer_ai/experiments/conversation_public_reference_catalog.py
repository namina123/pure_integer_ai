"""DLG-RAW-05B 的公开双 claim/reference Frame catalog。

本模块把内容锁定的公开课程投影为 V3 ``PublicFrame``。生产路径只重派生
Evidence、显式 claim order 和来源化 reference lexeme；课程 answer plan、已接受/拒绝
surface、reference teacher course 与策略标签均不进入 catalog record 或 materializer 返回值。
"""
from __future__ import annotations

from dataclasses import dataclass
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
    PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1,
    PublicFrame,
    PublicFrameCatalog,
    PublicFrameQuestionTemplate,
    PublicFrameReferenceRuntimeRecipe,
    PublicFrameSourceRecord,
    PublicFrameLexicalRoute,
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
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningInput,
    PublicResponseActPlanningBuild,
    compile_public_reference_planning,
)
from pure_integer_ai.experiments.public_grounded_answer_course_projection import (
    PublicGroundedAnswerCourseProjectionError,
    project_public_grounded_answer_from_payload,
)


PUBLIC_REFERENCE_FRAME_CATALOG_SCHEMA_V3 = 3
PUBLIC_REFERENCE_RELATION_PROPOSITION_EVENT_V1 = 1
PUBLIC_REFERENCE_STRATEGY_LOWEST_COST_V1 = (
    PUBLIC_FRAME_REFERENCE_SELECTION_LOWEST_COST_V1)

_NAMESPACE = 65001
_LEXICAL_SOURCE_KIND = 65053
_LANGUAGE_BRANCH = language_branch_identity((_NAMESPACE, 53, 2))
_REPRESENTATION_FAMILY = (_NAMESPACE, 53, 3)
_COURSE_ATTRIBUTION = "Pure Integer AI authored public grounded answer course"
_HEX = frozenset("0123456789abcdef")

_MANIFEST_FIELDS = frozenset({
    "antecedent_lexical_sources",
    "catalog_schema",
    "course_raw_sha256",
    "course_relative_path",
    "episode_id",
    "explicit_lexical_sources",
    "frame_key",
    "input_lexical_sources",
    "ordered_proposition_ids",
    "output_max_bytes",
    "relation_kind_code",
})
_INPUT_LEXICAL_SOURCE_FIELDS = frozenset({
    "attribution",
    "license_id",
    "raw_sha256",
    "relative_path",
})
_REFERENCE_LEXICAL_SOURCE_FIELDS = frozenset({
    "attribution",
    "license_id",
    "raw_sha256",
    "relative_path",
    "span_utf8_hex",
})

PUBLIC_REFERENCE_CATALOG_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_reference_frame_v3.jsonl.sample")


# object-model: exception; interop=DLG-RAW-05B
class PublicReferenceCatalogError(ValueError):
    """公开双 claim 课程、词汇来源或无标签 V3 record 无法闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """以长度前缀把有限整数段加入 canonical record。"""
    result.extend((len(value), *value))


def _u8_record(value: str, *, label: str) -> tuple[int, ...]:
    """将 transport text 显式转换为 UTF-8 byte record，不把宿主编码当语义。"""
    if (not isinstance(value, str) or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"):
        raise PublicReferenceCatalogError(f"{label} 必须是无首尾空白文本")
    scalars = tuple(ord(character) for character in value)
    try:
        return encode_utf8_v1(scalars)
    except (TypeError, ValueError) as error:
        raise PublicReferenceCatalogError(f"{label} 含非法 Unicode scalar") from error


def _fingerprint_text(*values: str, domain: str) -> tuple[int, ...]:
    """以显式 UTF-8 长度前缀构成版本化 deterministic identity 输入。"""
    encoded: list[int] = []
    for value in values:
        _pack(encoded, _u8_record(value, label="identity text"))
    return integer_tuple_fingerprint(tuple(encoded), domain=domain)


def _positive_identifier(values: tuple[int, ...], *, domain: str) -> int:
    """按冻结大端规则把整数指纹前八个 digest byte 映射为正身份号。"""
    fingerprint = integer_tuple_fingerprint(values, domain=domain)
    if (len(fingerprint) != 34
            or any(type(item) is not int or item < 0 or item > 255
                   for item in fingerprint[2:])):
        raise PublicReferenceCatalogError("整数指纹实现未返回固定 SHA-256 record")
    result = 0
    for value in fingerprint[2:10]:
        result = (result << 8) | value
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    """拒绝缺字段、尾随字段和宿主默认值进入 manifest 边界。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicReferenceCatalogError(f"{label} 字段集合漂移")
    return value


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    """只接受严格整数，拒绝 bool、float 与隐式数值转换。"""
    if type(value) is not int or value < minimum:
        raise PublicReferenceCatalogError(f"{label} 必须是不小于 {minimum} 的严格整数")
    return value


def _ascii_id(value: Any, *, label: str) -> str:
    """校验稳定 ASCII id，避免 path、空白和 Unicode normalize 参与身份。"""
    if (not isinstance(value, str) or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"
            or any(ord(character) > 0x7F for character in value)
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "abcdefghijklmnopqrstuvwxyz0123456789-_.:"
                   for character in value)):
        raise PublicReferenceCatalogError(f"{label} 不是稳定 ASCII id")
    return value


def _hex_nibble(value: str) -> int:
    """以固定 ASCII 规则将一位小写 hex 转为整数。"""
    code = ord(value)
    return code - 0x30 if code <= 0x39 else code - 0x61 + 10


def _hex_bytes(value: Any, *, label: str, size: int | None = None) -> tuple[int, ...]:
    """手工还原小写 hex 到 u8 vector，不以宿主 codec 定义 wire。"""
    if (not isinstance(value, str) or not value or len(value) % 2
            or any(character not in _HEX for character in value)):
        raise PublicReferenceCatalogError(f"{label} 不是小写 hex")
    result = tuple(
        (_hex_nibble(value[index]) << 4) | _hex_nibble(value[index + 1])
        for index in range(0, len(value), 2))
    if size is not None and len(result) != size:
        raise PublicReferenceCatalogError(f"{label} 长度漂移")
    return result


def _logical_payload_key(
        value: Any,
        *,
        label: str,
        ) -> tuple[str, bytes]:
    """将 manifest 的 ASCII logical key 规范成 closure 查询键。

    不读取、解析或保留宿主位置；资源的存在性、完整性和读取副作用已在 payload
    closure 的 host 边界冻结，core 只消费其有限 bytes/integer record。
    """
    if (not isinstance(value, str) or not value
            or any(ord(character) < 0x21 or ord(character) > 0x7E
                   for character in value)):
        raise PublicReferenceCatalogError(f"{label} 不是规范 ASCII logical key")
    parts = tuple(value.split("/"))
    if (len(parts) != 3 or parts[:2] != ("data", "ph2")
            or any(not part or part in {".", ".."} for part in parts)
            or "\\" in value):
        raise PublicReferenceCatalogError(f"{label} 越出 data/ph2 logical key")
    return value, bytes(ord(character) for character in value)


def _payload_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        logical_key: bytes,
        expected_sha256: tuple[int, ...] | None,
        *,
        label: str,
        ) -> bytes:
    """从 closure 读取一项 raw payload，并显式核验 key、长度和 SHA-256。"""
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise PublicReferenceCatalogError("public source payload closure 类型错误")
    if (type(logical_key) is not bytes or not logical_key
            or any(value < 0x21 or value > 0x7E for value in logical_key)):
        raise PublicReferenceCatalogError(f"{label} logical key 非法")
    try:
        record = source_payload_closure.record_for(logical_key)
        payload = source_payload_closure.payload_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise PublicReferenceCatalogError(
            f"{label} 不在 public source payload closure 内") from error
    digest = tuple(public_source_payload_sha256_v1(payload))
    if (record.logical_key != logical_key
            or record.raw_payload != payload
            or record.payload_length != len(payload)
            or tuple(record.raw_sha256) != digest):
        raise PublicReferenceCatalogError(f"{label} payload record 漂移")
    if expected_sha256 is not None and digest != expected_sha256:
        raise PublicReferenceCatalogError(f"{label} raw SHA-256 漂移")
    return payload


def _decode_scalars(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """按 DLG-RAW-00 严格 UTF-8 状态机回读一个 source span。"""
    scalars = decode_utf8_v1(value)
    if scalars is None or encode_utf8_v1(scalars) != value:
        raise PublicReferenceCatalogError(f"{label} 不是可回读 UTF-8")
    return scalars


def _unique_span(
        payload: bytes,
        needle: bytes,
        *,
        label: str,
        ) -> tuple[int, int]:
    """定位唯一 raw byte span；多个命中必须由来源重新拆分而非任取其一。"""
    if not needle:
        raise PublicReferenceCatalogError(f"{label} 不得为空")
    start = payload.find(needle)
    if start < 0 or payload.find(needle, start + 1) >= 0:
        raise PublicReferenceCatalogError(f"{label} 在公开 source 中缺失或不唯一")
    return start, start + len(needle)


def _course_line_bounds(
        payload: bytes,
        episode_id: str,
        ) -> tuple[int, int]:
    """从唯一非标签 episode-id span 确定其 JSONL raw line 的绝对边界。"""
    marker = bytes(_course_ascii_field_observation(
        "episode_id", episode_id, label="course episode id"))
    start, end = _unique_span(payload, marker, label="course episode id")
    line_start = payload.rfind(b"\n", 0, start) + 1
    line_end = payload.find(b"\n", end)
    if line_end < 0 or line_start >= line_end:
        raise PublicReferenceCatalogError("course episode JSONL line 边界非法")
    return line_start, line_end


def _json_array_span(
        payload: bytes,
        *,
        line_start: int,
        line_end: int,
        field: str,
        ) -> tuple[int, int]:
    """用显式 byte/string 状态机定位一行 canonical JSON 内唯一 array 字段。"""
    marker = bytes(_course_ascii_field_prefix(field))
    start = payload.find(marker, line_start, line_end)
    if start < 0 or payload.find(marker, start + 1, line_end) >= 0:
        raise PublicReferenceCatalogError(f"course {field} array 缺失或不唯一")
    cursor = start + len(marker)
    if cursor >= line_end or payload[cursor] != 0x5B:
        raise PublicReferenceCatalogError(f"course {field} array 起始非法")
    depth = 0
    in_string = False
    escaped = False
    for index in range(cursor, line_end):
        value = payload[index]
        if in_string:
            if escaped:
                escaped = False
            elif value == 0x5C:
                escaped = True
            elif value == 0x22:
                in_string = False
            continue
        if value == 0x22:
            in_string = True
        elif value == 0x5B:
            depth += 1
        elif value == 0x5D:
            depth -= 1
            if depth == 0:
                return start, index + 1
            if depth < 0:
                break
    raise PublicReferenceCatalogError(f"course {field} array 未闭合")


def _lexical_source_ref(
        relative_path: str,
        raw_sha256: tuple[int, ...],
        ) -> SourceRef:
    """由物理 source path/hash 建立可审计、稳定的独立 lexical SourceRef。"""
    path = _u8_record(relative_path, label="lexical relative path")
    source_id = _positive_identifier(
        (len(path), *path, len(raw_sha256), *raw_sha256),
        domain="dlg.raw.public.reference.lexical.source.v1",
    )
    return SourceRef(
        _LEXICAL_SOURCE_KIND,
        source_id,
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _lexical_observation(
        raw: Any,
        *,
        fields: frozenset[str],
        expected_bytes: tuple[int, ...],
        source_payload_closure: PublicSourcePayloadClosureV1,
        record_id: str,
        ) -> PublicFrameSourceRecord:
    """回读一份独立 lexical source，并把声明的唯一 span 绑定为来源 record。"""
    value = _exact(raw, fields, label=record_id)
    relative_path, logical_key = _logical_payload_key(
        value["relative_path"], label=f"{record_id}.relative_path")
    digest = _hex_bytes(value["raw_sha256"], label=f"{record_id}.raw_sha256",
                        size=32)
    license_id = value["license_id"]
    attribution = value["attribution"]
    if license_id != "CC0-1.0":
        raise PublicReferenceCatalogError(f"{record_id}.license 必须是 CC0-1.0")
    if (not isinstance(attribution, str) or not attribution
            or attribution[0] in " \t\r\n" or attribution[-1] in " \t\r\n"):
        raise PublicReferenceCatalogError(f"{record_id}.attribution 非法")
    payload = _payload_from_closure(
        source_payload_closure,
        logical_key,
        digest,
        label=record_id,
    )
    span = _unique_span(payload, bytes(expected_bytes), label=f"{record_id}.span")
    return PublicFrameSourceRecord(
        record_id,
        _lexical_source_ref(relative_path, digest),
        relative_path,
        digest,
        license_id,
        attribution,
        span,
        expected_bytes,
        _decode_scalars(expected_bytes, label=f"{record_id}.span"),
    )


def _course_source_record(
        record_id: str,
        *,
        source: SourceRef,
        relative_path: str,
        raw_sha256: tuple[int, ...],
        course_payload: bytes,
        span: tuple[int, int],
        ) -> PublicFrameSourceRecord:
    """从已验证 course bytes 形成单一非标签 observation source record。"""
    if (len(span) != 2 or span[0] < 0 or span[0] >= span[1]
            or span[1] > len(course_payload)):
        raise PublicReferenceCatalogError("course source span 越界")
    observation = tuple(course_payload[span[0]:span[1]])
    return PublicFrameSourceRecord(
        record_id,
        source,
        relative_path,
        raw_sha256,
        "CC0-1.0",
        _COURSE_ATTRIBUTION,
        span,
        observation,
        _decode_scalars(observation, label=f"{record_id}.course"),
    )


def _course_ascii_field_observation(
        field: str,
        value: str,
        *,
        label: str,
        ) -> tuple[int, ...]:
    """形成 canonical JSON 中唯一、非标签 ASCII field span。"""
    identifier = _ascii_id(value, label=label)
    result = tuple(ord(character) for character in f'"{field}":"{identifier}"')
    if any(item > 0x7F for item in result):
        raise PublicReferenceCatalogError(f"{label} 不是 ASCII field observation")
    return result


def _course_ascii_field_prefix(field: str) -> tuple[int, ...]:
    """形成 JSON array field 的固定 ASCII prefix，不解析答案或标签字段。"""
    if (not isinstance(field, str) or not field
            or any(character not in "abcdefghijklmnopqrstuvwxyz_"
                   for character in field)):
        raise PublicReferenceCatalogError("course array field 非法")
    return tuple(ord(character) for character in f'"{field}":')


def _question_template(
        planning_input: GroundedResponseActPlanningInput,
        build: PublicResponseActPlanningBuild,
        *,
        frame_key: str,
        ) -> PublicFrameQuestionTemplate:
    """从 label-free planning 机械形成完整十字段 QuestionRequest 模板。"""
    intent_key = _fingerprint_text(
        planning_input.typed_intent,
        domain="dlg.raw.public.reference.intent.v1",
    )
    trace_key = integer_tuple_fingerprint(
        planning_input.canonical_record(),
        domain="dlg.raw.public.reference.question.trace.v1",
    )
    authorized = tuple(sorted({
        item.candidate.proposition for item in build.candidate_bindings
    }, key=lambda item: item.stable_key()))
    if (len(authorized) != 2
            or build.planning.goal.proposition not in authorized):
        raise PublicReferenceCatalogError(
            "reference planning 未授权两个完整 candidate target")
    return PublicFrameQuestionTemplate(
        minimal_instruction_identity((_NAMESPACE, 53, 10)),
        minimal_instruction_identity((_NAMESPACE, 53, 11, *intent_key)),
        build.planning.goal.goal_kind,
        build.planning.goal.proposition,
        build.planning.goal.required,
        query_scope(
            planning_input.evidence_scope_id,
            parent=document_scope(build.aggregate_source),
        ),
        build.response_scope,
        (_NAMESPACE, 53, 12, *trace_key),
        _LANGUAGE_BRANCH,
        authorized,
    )


def _verify_frame_source_records(
        frame: PublicFrame,
        *,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> tuple[tuple[str, bytes], ...]:
    """逐轮从 closure 读取所有 frame source，并验证 hash、span 与 UTF-8 scalar。"""
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
            raise PublicReferenceCatalogError("runtime source span 越界")
        if tuple(payload[source.span[0]:source.span[1]]) != source.span_bytes:
            raise PublicReferenceCatalogError("runtime source span bytes 漂移")
        if (_decode_scalars(source.span_bytes,
                            label=f"runtime source {source.record_id}.span")
                != source.span_scalars):
            raise PublicReferenceCatalogError("runtime source span scalar 漂移")
    return tuple((path, payload) for path, _sha256, payload in payloads)


def _payload_for_relative_path(
        payloads: tuple[tuple[str, bytes], ...],
        relative_path: str,
        ) -> bytes:
    """只从本轮已验证 source payload 取课程，不允许路径或缓存 fallback。"""
    matches = tuple(payload for path, payload in payloads if path == relative_path)
    if len(matches) != 1:
        raise PublicReferenceCatalogError("runtime course 不在 frame source records 内")
    return matches[0]


def _source_scalars(
        frame: PublicFrame,
        record_ids: tuple[str, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """从两个独立 source record 读取同一 reference lexeme scalar。"""
    records = tuple(
        source for source in frame.source_records
        if source.record_id in record_ids)
    if len(records) != len(record_ids):
        raise PublicReferenceCatalogError(f"{label} source record 缺失")
    scalars = tuple(source.span_scalars for source in records)
    if len(set(scalars)) != 1:
        raise PublicReferenceCatalogError(f"{label} lexical source scalar 不一致")
    return scalars[0]


# object-model: value; representation=struct; interop=DLG-RAW-05B
@dataclass(frozen=True, slots=True)
class PublicReferenceFramePlanningBuild:
    """V3 runtime 的无标签 planning 与两种来源化 reference lexeme。"""

    planning_build: PublicResponseActPlanningBuild
    antecedent_reference_scalars: tuple[int, ...]
    explicit_repetition_scalars: tuple[int, ...]

    def __post_init__(self) -> None:
        """确保两个 candidate 与两个可回读 scalar lexeme 全部独立闭合。"""
        if not isinstance(self.planning_build, PublicResponseActPlanningBuild):
            raise TypeError("public reference planning build 类型错误")
        if len(self.planning_build.planning.candidates) != 2:
            raise PublicReferenceCatalogError("public reference planning 缺两个 candidate")
        for label, scalars in (
                ("antecedent", self.antecedent_reference_scalars),
                ("explicit", self.explicit_repetition_scalars)):
            if (not isinstance(scalars, tuple) or not scalars
                    or any(type(item) is not int or item < 0
                           or item > 0x10FFFF
                           or 0xD800 <= item <= 0xDFFF
                           for item in scalars)):
                raise PublicReferenceCatalogError(
                    f"public reference {label} scalar 非法")

    def canonical_record(self) -> tuple[int, ...]:
        """导出 runtime consumer 可独立重放的完整整数 planning 投影。"""
        result = [1]
        for value in (
                self.planning_build.stable_key(),
                self.antecedent_reference_scalars,
                self.explicit_repetition_scalars):
            _pack(result, value)
        return tuple(result)


def _source_list(
        raw: Any,
        *,
        label: str,
        ) -> list[Any]:
    """验证每类 lexical 观察精确由两个独立 source 声明组成。"""
    if not isinstance(raw, list) or len(raw) != 2:
        raise PublicReferenceCatalogError(f"{label} 必须精确含两个 lexical source")
    return raw


def _frame_from_manifest(
        raw: Any,
        *,
        raw_line_sha256: tuple[int, ...],
        source_payload_closure: PublicSourcePayloadClosureV1,
        ordinal: int,
        ) -> PublicFrame:
    """由一条 V3 manifest 与受 SHA 锁定课程重派生完整 public frame。"""
    manifest = _exact(raw, _MANIFEST_FIELDS, label="reference frame")
    if _strict_int(manifest["catalog_schema"], label="catalog_schema", minimum=1) != (
            PUBLIC_REFERENCE_FRAME_CATALOG_SCHEMA_V3):
        raise PublicReferenceCatalogError("reference catalog schema 未注册")
    frame_key = _ascii_id(manifest["frame_key"], label="frame_key")
    episode_id = _ascii_id(manifest["episode_id"], label="episode_id")
    output_max_bytes = _strict_int(
        manifest["output_max_bytes"], label="output_max_bytes", minimum=1)
    relation_kind = _strict_int(
        manifest["relation_kind_code"], label="relation_kind_code", minimum=1)
    if relation_kind != PUBLIC_REFERENCE_RELATION_PROPOSITION_EVENT_V1:
        raise PublicReferenceCatalogError("reference relation kind 未注册")
    order_raw = manifest["ordered_proposition_ids"]
    if not isinstance(order_raw, list) or len(order_raw) != 2:
        raise PublicReferenceCatalogError("ordered_proposition_ids 必须精确含两个 id")
    ordered_proposition_ids = tuple(
        _ascii_id(value, label="ordered proposition id") for value in order_raw)
    if len(set(ordered_proposition_ids)) != 2:
        raise PublicReferenceCatalogError("ordered_proposition_ids 重复")
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
        projection = project_public_grounded_answer_from_payload(
            course_payload, episode_id, train_only=True)
    except PublicGroundedAnswerCourseProjectionError as error:
        raise PublicReferenceCatalogError("公开课程结构或 train split 漂移") from error
    planning_input = projection.planning_input
    try:
        build = compile_public_reference_planning(
            planning_input,
            _LANGUAGE_BRANCH,
            ordered_proposition_ids,
        )
    except (TypeError, ValueError) as error:
        raise PublicReferenceCatalogError(
            "公开课程无法形成无标签 reference planning") from error
    question_bytes = _u8_record(
        projection.question_surface,
        label="课程 question surface",
    )
    surface_scalars = _decode_scalars(
        question_bytes, label="课程 question surface")

    input_records = tuple(_lexical_observation(
        value,
        fields=_INPUT_LEXICAL_SOURCE_FIELDS,
        expected_bytes=question_bytes,
        source_payload_closure=source_payload_closure,
        record_id=f"input-{ordinal}-{index}",
    ) for index, value in enumerate(_source_list(
        manifest["input_lexical_sources"], label="input lexical sources"), start=1))
    antecedent_records = tuple(_lexical_observation(
        value,
        fields=_REFERENCE_LEXICAL_SOURCE_FIELDS,
        expected_bytes=_hex_bytes(
            _exact(value, _REFERENCE_LEXICAL_SOURCE_FIELDS,
                   label=f"antecedent-{ordinal}-{index}")["span_utf8_hex"],
            label=f"antecedent-{ordinal}-{index}.span_utf8_hex"),
        source_payload_closure=source_payload_closure,
        record_id=f"antecedent-{ordinal}-{index}",
    ) for index, value in enumerate(_source_list(
        manifest["antecedent_lexical_sources"],
        label="antecedent lexical sources"), start=1))
    explicit_records = tuple(_lexical_observation(
        value,
        fields=_REFERENCE_LEXICAL_SOURCE_FIELDS,
        expected_bytes=_hex_bytes(
            _exact(value, _REFERENCE_LEXICAL_SOURCE_FIELDS,
                   label=f"explicit-{ordinal}-{index}")["span_utf8_hex"],
            label=f"explicit-{ordinal}-{index}.span_utf8_hex"),
        source_payload_closure=source_payload_closure,
        record_id=f"explicit-{ordinal}-{index}",
    ) for index, value in enumerate(_source_list(
        manifest["explicit_lexical_sources"], label="explicit lexical sources"), start=1))
    if (len({item.source.stable_key() for item in input_records}) != 2
            or len({item.source.stable_key() for item in antecedent_records}) != 2
            or len({item.source.stable_key() for item in explicit_records}) != 2):
        raise PublicReferenceCatalogError(
            "V3 每类 lexical source 必须来自两个不同 SourceRef")
    if len({item.span_scalars for item in antecedent_records}) != 1:
        raise PublicReferenceCatalogError("antecedent lexical scalar 不一致")
    if len({item.span_scalars for item in explicit_records}) != 1:
        raise PublicReferenceCatalogError("explicit lexical scalar 不一致")

    line_start, line_end = _course_line_bounds(course_payload, episode_id)
    episode_span = _unique_span(
        course_payload,
        bytes(_course_ascii_field_observation(
            "episode_id", episode_id, label="course episode id")),
        label="course episode id",
    )
    source_ids = tuple(sorted({
        item.source_id for item in planning_input.evidence}))
    if len(source_ids) != 1:
        raise PublicReferenceCatalogError(
            "V3 窄切片要求两个 claim 同一 course source")
    evidence_span = _json_array_span(
        course_payload,
        line_start=line_start,
        line_end=line_end,
        field="evidence",
    )
    course_record = _course_source_record(
        f"course-{ordinal}",
        source=build.aggregate_source,
        relative_path=course_relative_path,
        raw_sha256=course_sha256,
        course_payload=course_payload,
        span=episode_span,
    )
    evidence_record = _course_source_record(
        f"evidence-{ordinal}",
        source=build.source_for(source_ids[0]),
        relative_path=course_relative_path,
        raw_sha256=course_sha256,
        course_payload=course_payload,
        span=evidence_span,
    )
    source_records = tuple(sorted(
        (course_record, evidence_record, *input_records,
         *antecedent_records, *explicit_records),
        key=lambda item: item.source.stable_key(),
    ))

    atom_key = _fingerprint_text(
        frame_key,
        projection.question_surface,
        domain="dlg.raw.public.reference.language.atom.v1",
    )
    atom = language_atom_identity(
        _LANGUAGE_BRANCH, (_NAMESPACE, 53, 20, *atom_key))
    route = PublicFrameLexicalRoute(
        0,
        (0, len(surface_scalars)),
        _LANGUAGE_BRANCH,
        representation_identity(
            _REPRESENTATION_FAMILY,
            (0, len(surface_scalars), *surface_scalars),
        ),
        atom,
        tuple(sorted(input_records, key=lambda item: item.source.stable_key())),
        surface_scalars,
    )
    construction_key = _fingerprint_text(
        frame_key,
        domain="dlg.raw.public.reference.construction.v1",
    )
    recipe_key = _fingerprint_text(
        frame_key,
        domain="dlg.raw.public.reference.recipe.v1",
    )
    recipe = PublicFrameReferenceRuntimeRecipe(
        minimal_instruction_identity((_NAMESPACE, 53, 30, *recipe_key)),
        course_relative_path,
        course_sha256,
        episode_id,
        planning_input.canonical_record(),
        ordered_proposition_ids,
        ordered_proposition_ids[0],
        ordered_proposition_ids[1],
        relation_kind,
        course_record.record_id,
        (evidence_record.record_id,),
        tuple(sorted(item.record_id for item in antecedent_records)),
        tuple(sorted(item.record_id for item in explicit_records)),
        output_max_bytes,
        1,
        PUBLIC_REFERENCE_STRATEGY_LOWEST_COST_V1,
    )
    return PublicFrame(
        frame_key,
        raw_line_sha256,
        question_bytes,
        surface_scalars,
        source_records,
        (route,),
        structure_concept_identity((_NAMESPACE, 53, 31, *construction_key)),
        (atom,),
        _question_template(planning_input, build, frame_key=frame_key),
        recipe,
        PUBLIC_FRAME_CONTEXT_NONE,
        (),
    )


def load_public_reference_frame_catalog_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrameCatalog:
    """从已闭合的 logical payload 资源构造 V3 reference Frame catalog。"""
    payload = _payload_from_closure(
        source_payload_closure,
        PUBLIC_REFERENCE_CATALOG_LOGICAL_KEY_V1,
        None,
        label="reference catalog manifest",
    )
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise PublicReferenceCatalogError("V3 catalog JSONL 换行非法")
    lines = payload[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise PublicReferenceCatalogError("V3 catalog 含空记录")
    frames = []
    for ordinal, line in enumerate(lines, start=1):
        try:
            manifest = parse_canonical_json_bytes(line, require_object=True)
        except DatasetContractError as error:
            raise PublicReferenceCatalogError("V3 manifest 不是 canonical JSON") from error
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
        raise PublicReferenceCatalogError("V3 manifest frame_key 重复")
    if len(set(episode_ids)) != len(episode_ids):
        raise PublicReferenceCatalogError("V3 manifest episode_id 重复")
    if len(set(surfaces)) != len(surfaces):
        raise PublicReferenceCatalogError("V3 manifest exact scalar surface 重复")
    return PublicFrameCatalog(
        tuple(hashlib.sha256(payload).digest()),
        tuple(sorted(frames, key=PublicFrame.canonical_record)),
    )


def materialize_public_reference_planning_from_closure(
        frame: PublicFrame,
        request: QuestionRequest,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicReferenceFramePlanningBuild:
    """每轮回读 V3 source，重建无标签双 claim planning 和 reference scalar。"""
    if not isinstance(frame, PublicFrame) or not isinstance(request, QuestionRequest):
        raise TypeError("reference materialization 需要 PublicFrame 和 QuestionRequest")
    if not isinstance(frame.recipe, PublicFrameReferenceRuntimeRecipe):
        raise PublicReferenceCatalogError("frame 不是 V3 reference recipe")
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
        raise PublicReferenceCatalogError(
            "reference materialization 收到漂移完整 QuestionRequest")
    recipe = frame.recipe
    course_relative_path, _course_logical_key = _logical_payload_key(
        recipe.course_relative_path,
        label="runtime course_relative_path",
    )
    if course_relative_path != recipe.course_relative_path:
        raise PublicReferenceCatalogError("runtime course logical key 规范化漂移")
    payloads = _verify_frame_source_records(
        frame,
        source_payload_closure=source_payload_closure,
    )
    payload = _payload_for_relative_path(payloads, course_relative_path)
    if tuple(hashlib.sha256(payload).digest()) != recipe.course_raw_sha256:
        raise PublicReferenceCatalogError("runtime 公开课程 raw SHA-256 漂移")
    try:
        projection = project_public_grounded_answer_from_payload(
            payload, recipe.episode_id, train_only=True)
    except PublicGroundedAnswerCourseProjectionError as error:
        raise PublicReferenceCatalogError(
            "runtime 公开课程结构或 train split 漂移") from error
    planning_input = projection.planning_input
    if planning_input.canonical_record() != recipe.planning_input_record:
        raise PublicReferenceCatalogError(
            "runtime 课程重派生 planning input 与 V3 recipe 漂移")
    try:
        build = compile_public_reference_planning(
            planning_input,
            frame.question.target_branch,
            recipe.ordered_proposition_ids,
        )
    except (TypeError, ValueError) as error:
        raise PublicReferenceCatalogError(
            "runtime 无法形成无标签 reference planning") from error
    if (build.planning.goal.goal_kind != request.goal_kind
            or build.planning.goal.proposition != request.target
            or build.planning.goal.required != request.required
            or build.planning.goal.source != request.source
            or build.response_scope != request.response_scope
            or build.planning.goal.target_branch != request.target_branch):
        raise PublicReferenceCatalogError(
            "runtime reference planning 与 QuestionRequest 漂移")
    bindings = tuple(
        item.proposition_id for item in build.candidate_bindings)
    if (recipe.antecedent_proposition_id != recipe.ordered_proposition_ids[0]
            or recipe.referring_proposition_id
            != recipe.ordered_proposition_ids[1]
            or not set(recipe.ordered_proposition_ids).issubset(set(bindings))):
        raise PublicReferenceCatalogError("runtime reference relation 漂移")
    return PublicReferenceFramePlanningBuild(
        build,
        _source_scalars(
            frame,
            recipe.antecedent_reference_source_record_ids,
            label="antecedent reference",
        ),
        _source_scalars(
            frame,
            recipe.explicit_repetition_source_record_ids,
            label="explicit repetition",
        ),
    )


__all__ = [
    "PUBLIC_REFERENCE_FRAME_CATALOG_SCHEMA_V3",
    "PUBLIC_REFERENCE_CATALOG_LOGICAL_KEY_V1",
    "PUBLIC_REFERENCE_RELATION_PROPOSITION_EVENT_V1",
    "PUBLIC_REFERENCE_STRATEGY_LOWEST_COST_V1",
    "PublicReferenceCatalogError",
    "PublicReferenceFramePlanningBuild",
    "load_public_reference_frame_catalog_from_closure",
    "materialize_public_reference_planning_from_closure",
]
