"""DLG-RAW-15 G1 route-form 证据层（只生成/核验，不接生产 runtime）。

本模块解决一个很窄、但必须先闭合的问题：给定一个真正独立的 V3
``SourceBoundSlotCompositionResolution``，如何形成可迁移的 route identity、两
个 candidate identity、canonical route course，以及两份独立的 raw surface
witness。它不修改 DLG-RAW-14 production catalog，也不把本模块的可行性结果
当作 G1 PASS。

所有语义字段在 record 中均降解为整数或 raw-u8；``str``/``dict`` 只存在于
canonical JSONL 的边界。dataclass 仅是不可变 struct carrier，不使用对象身份、
宿主 hash 顺序或物理路径定义语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.conversation_public_route_clarification_catalog import (
    PUBLIC_ROUTE_CLARIFICATION_CANDIDATE_IDENTITY_DOMAIN_V1,
    PUBLIC_ROUTE_CLARIFICATION_ROUTE_IDENTITY_DOMAIN_V1,
    route_identity_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    portable_sha256_v1,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    decode_utf8_v1,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3,
    SourceBoundSlotCompositionResolution,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)


DLG_RAW15_G1_ROUTE_FORM_EVIDENCE_SCHEMA_V1 = 1
DLG_RAW15_G1_ROUTE_SOURCE_RECORD_V1 = 1
DLG_RAW15_G1_ROUTE_FORM_RECORD_V1 = 1
DLG_RAW15_G1_ROUTE_FORM_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-15/G1-ROUTE-FORM/V1")
DLG_RAW15_G1_ROUTE_SOURCE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-15/G1-ROUTE-SOURCE/V1")
DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1 = 1
DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1 = 2
DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_B_V1 = 3

_LICENSE = tuple(b"CC0-1.0")
_U64_EXCLUSIVE = 1 << 64
_FORM_FIELDS = (
    "candidate_identities",
    "course_attribution",
    "form_id",
    "input_surface",
    "license_id",
    "matched_frame_count",
    "options",
    "output_max_bytes",
    "output_surface",
    "result_code",
    "route_identity",
    "schema",
    "surface_a",
    "surface_b",
)
_OPTION_FIELDS = ("candidate_identity", "option_surface")
_SURFACE_FIELDS = ("attribution", "raw_sha256", "relative_path")


# object-model: exception; interop=DLG-RAW-15
class DlgRaw15RouteFormEvidenceError(ValueError):
    """独立 route-form 课程、来源 witness 或 identity 未能闭合。"""


def _u8(value: Any, *, label: str, allow_empty: bool = True) -> tuple[int, ...]:
    """验证显式 0..255 tuple，拒绝 bool、bytes 子类和隐式转换。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise DlgRaw15RouteFormEvidenceError(
            f"{label} 必须是{'非空' if not allow_empty else ''} raw-u8 tuple")
    return value


def _record(value: Any, *, label: str, allow_empty: bool = True) -> tuple[int, ...]:
    """验证可迁移的非负整数 record。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise DlgRaw15RouteFormEvidenceError(
            f"{label} 必须是{'非空' if not allow_empty else ''}整数 record")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    """以显式 u64 长度前缀追加子 record。"""
    checked = _record(value, label=label)
    if len(checked) >= _U64_EXCLUSIVE:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 超出 u64 长度")
    result.extend((len(checked), *checked))


def _sha_payload(payload: bytes, *, label: str) -> tuple[int, ...]:
    """形成标准 raw payload SHA-256 byte vector。"""
    if type(payload) is not bytes:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 必须是 bytes")
    return _u8(tuple(public_source_payload_sha256_v1(payload)), label=f"{label} SHA")


def _utf8(value: Any, *, label: str, allow_lf: bool, allow_empty: bool = False) -> tuple[int, ...]:
    """用显式 UTF-8 编码文本，不做 Unicode 归一化。"""
    if type(value) is not str:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 必须是 str")
    if (not allow_empty and not value) or (not allow_empty and
            (value[0] in " \t\r\n" or value[-1] in " \t\r\n")):
        raise DlgRaw15RouteFormEvidenceError(f"{label} 不能为空或含首尾空白")
    if not allow_lf and ("\r" in value or "\n" in value):
        raise DlgRaw15RouteFormEvidenceError(f"{label} 不得含换行")
    scalars = tuple(ord(item) for item in value)
    if any((item < 0x20 and (not allow_lf or item != 0x0A))
           or item == 0x7F for item in scalars):
        raise DlgRaw15RouteFormEvidenceError(f"{label} 含未注册控制字符")
    try:
        encoded = encode_utf8_v1(scalars)
    except (TypeError, ValueError) as error:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 含非法 Unicode scalar") from error
    return _u8(tuple(encoded), label=label, allow_empty=allow_empty)


def _ascii_key(value: Any, *, label: str) -> tuple[int, ...]:
    """验证独立 source pack 的 ASCII logical key。"""
    if type(value) is not bytes or not value:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 必须是非空 bytes")
    if (any(item < 0x21 or item > 0x7E for item in value)
            or b"\\" in value or b"//" in value
            or b"/../" in value or value.startswith(b"../")
            or value.endswith(b"/..")):
        raise DlgRaw15RouteFormEvidenceError(f"{label} 不是规范 ASCII logical key")
    parts = value.split(b"/")
    if len(parts) < 3 or parts[0:2] != [b"data", b"ph2"] or any(
            not part or part in (b".", b"..") for part in parts):
        raise DlgRaw15RouteFormEvidenceError(
            f"{label} 必须位于 data/ph2 logical namespace")
    return tuple(value)


def _ascii_id(value: Any, *, label: str) -> tuple[int, ...]:
    """验证不带路径语义的稳定 ASCII form id。"""
    if type(value) is not bytes or not value:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 必须是非空 bytes")
    if any(item < 0x21 or item > 0x7E for item in value):
        raise DlgRaw15RouteFormEvidenceError(f"{label} 含不可见 ASCII")
    return tuple(value)


def _ascii_text(value: tuple[int, ...], *, label: str) -> str:
    """把已验证 ASCII transport 仅用于 canonical JSON 边界。"""
    checked = _u8(value, label=label, allow_empty=False)
    if any(item < 0x21 or item > 0x7E for item in checked):
        raise DlgRaw15RouteFormEvidenceError(f"{label} 不是 printable ASCII")
    return bytes(checked).decode("ascii")


def _text_from_u8(value: tuple[int, ...], *, label: str, allow_lf: bool = True) -> str:
    """将 UTF-8 raw-u8 机械回读为 JSON 字符串，不进行归一化。"""
    raw = bytes(_u8(value, label=label, allow_empty=False))
    scalars = decode_utf8_v1(tuple(raw))
    if scalars is None or tuple(encode_utf8_v1(scalars)) != tuple(raw):
        raise DlgRaw15RouteFormEvidenceError(f"{label} UTF-8 readback 失败")
    if (not allow_lf and any(item in (0x0A, 0x0D) for item in scalars)
            or any((item < 0x20 and (not allow_lf or item != 0x0A))
                   or item == 0x7F for item in scalars)):
        raise DlgRaw15RouteFormEvidenceError(f"{label} 含未注册控制字符")
    return "".join(chr(item) for item in scalars)


def _hex(value: tuple[int, ...], *, label: str) -> str:
    """将 raw-u8[32] 固定编码为小写 hex。"""
    checked = _u8(value, label=label, allow_empty=False)
    if len(checked) != 32:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 必须是 raw-u8[32]")
    return bytes(checked).hex()


def _unique_span(payload: bytes, needle: bytes, *, label: str) -> tuple[int, int]:
    """逐 byte 查找唯一 span，拒绝缺失与重复。"""
    if type(payload) is not bytes or not needle:
        raise DlgRaw15RouteFormEvidenceError(f"{label} source/needle 为空")
    first = payload.find(needle)
    if first < 0 or payload.find(needle, first + 1) >= 0:
        raise DlgRaw15RouteFormEvidenceError(f"{label} 在 source 中缺失或不唯一")
    return first, first + len(needle)


def _json_member(field: str, value: str, *, label: str) -> bytes:
    """构造 canonical JSON object 中一个 field 的精确 raw member fragment。"""
    if type(field) is not str or field not in _FORM_FIELDS:
        raise DlgRaw15RouteFormEvidenceError(f"{label} field 未注册")
    fragment = canonical_json_bytes({field: value})
    prefix = (b"{" + b'"' + field.encode("ascii") + b'":')
    suffix = b"}"
    if not fragment.startswith(prefix) or not fragment.endswith(suffix):
        raise DlgRaw15RouteFormEvidenceError(f"{label} member framing 漂移")
    return fragment[len(b"{"): -len(suffix)]


def _option_fragment(candidate_hex: str, option_surface: str) -> bytes:
    """构造 canonical options[] 中一个完整 object 的 raw bytes。"""
    if not candidate_hex or not option_surface:
        raise DlgRaw15RouteFormEvidenceError("option fragment 输入为空")
    return canonical_json_bytes({
        "candidate_identity": candidate_hex,
        "option_surface": option_surface,
    })


def _candidate_identity_from_record(record: tuple[int, ...]) -> tuple[int, ...]:
    """用既有 DLG-RAW-14 candidate identity domain 重算无循环 identity。"""
    return _u8(tuple(portable_sha256_v1(
        PUBLIC_ROUTE_CLARIFICATION_CANDIDATE_IDENTITY_DOMAIN_V1,
        (record,),
    )), label="candidate identity", allow_empty=False)


def _source_base_record(
        source_kind: int,
        logical_key_u8: tuple[int, ...],
        raw_sha256_u8: tuple[int, ...],
        span_start: int,
        span_end: int,
        span_u8: tuple[int, ...],
        license_id_u8: tuple[int, ...],
        attribution_u8: tuple[int, ...],
        ) -> tuple[int, ...]:
    """形成 source identity 的无循环本体 record。"""
    result = [DLG_RAW15_G1_ROUTE_SOURCE_RECORD_V1, source_kind]
    for label, value in (
            ("source logical key", logical_key_u8),
            ("source SHA", raw_sha256_u8),
            ("source span", span_u8),
            ("source license", license_id_u8),
            ("source attribution", attribution_u8)):
        _pack(result, value, label=label)
    result.extend((span_start, span_end))
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-15
@dataclass(frozen=True, slots=True)
class G1RouteSourceWitnessV1:
    """一份独立课程或 A/B raw source 的精确 span witness。"""

    source_kind: int
    logical_key_u8: tuple[int, ...]
    raw_sha256_u8: tuple[int, ...]
    span_start: int
    span_end: int
    span_u8: tuple[int, ...]
    license_id_u8: tuple[int, ...]
    attribution_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        """验证 source key、raw digest、许可和 span 长度闭合。"""
        if self.source_kind not in (
                DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1,
                DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1,
                DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_B_V1):
            raise DlgRaw15RouteFormEvidenceError("source kind 未注册")
        _ascii_key(bytes(_u8(self.logical_key_u8, label="source key", allow_empty=False)),
                   label="source key")
        digest = _u8(self.raw_sha256_u8, label="source SHA", allow_empty=False)
        if len(digest) != 32:
            raise DlgRaw15RouteFormEvidenceError("source SHA 长度漂移")
        if (type(self.span_start) is not int or self.span_start < 0
                or type(self.span_end) is not int or self.span_end <= self.span_start):
            raise DlgRaw15RouteFormEvidenceError("source span 边界非法")
        span = _u8(self.span_u8, label="source span", allow_empty=False)
        if self.span_end - self.span_start != len(span):
            raise DlgRaw15RouteFormEvidenceError("source span 长度漂移")
        if tuple(self.license_id_u8) != _LICENSE:
            raise DlgRaw15RouteFormEvidenceError("source license 必须是 CC0-1.0")
        _u8(self.attribution_u8, label="source attribution", allow_empty=False)

    def base_record(self) -> tuple[int, ...]:
        """返回可独立重算 source identity 的本体。"""
        return _source_base_record(
            self.source_kind,
            self.logical_key_u8,
            self.raw_sha256_u8,
            self.span_start,
            self.span_end,
            self.span_u8,
            self.license_id_u8,
            self.attribution_u8,
        )

    @property
    def source_identity_u8(self) -> tuple[int, ...]:
        """形成独立 G1 source identity。"""
        return _u8(tuple(portable_sha256_v1(
            DLG_RAW15_G1_ROUTE_SOURCE_IDENTITY_DOMAIN_V1,
            (self.base_record(),),
        )), label="source identity", allow_empty=False)

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 source witness integer record。"""
        result = list(self.base_record())
        _pack(result, self.source_identity_u8, label="source identity")
        return tuple(result)


def _source_witness(
        *,
        source_kind: int,
        logical_key_u8: tuple[int, ...],
        payload: bytes,
        span_u8: bytes,
        attribution_u8: tuple[int, ...],
        label: str,
        ) -> G1RouteSourceWitnessV1:
    """由 payload 中唯一 raw span 构造 source witness。"""
    start, end = _unique_span(payload, span_u8, label=label)
    return G1RouteSourceWitnessV1(
        source_kind,
        logical_key_u8,
        _sha_payload(payload, label=label),
        start,
        end,
        tuple(span_u8),
        _LICENSE,
        attribution_u8,
    )


# object-model: value; representation=struct; interop=DLG-RAW-15
@dataclass(frozen=True, slots=True)
class G1RouteFormEvidenceV1:
    """独立 route/form 的课程、候选与双 surface witness 闭包。"""

    form_id_u8: tuple[int, ...]
    result_code: int
    matched_frame_count: int
    input_scalars: tuple[int, ...]
    route_identity_u8: tuple[int, ...]
    candidate_identities_u8: tuple[tuple[int, ...], ...]
    candidate_records: tuple[tuple[int, ...], ...]
    course_payload: bytes
    course_source: G1RouteSourceWitnessV1
    output_u8: tuple[int, ...]
    output_course_source: G1RouteSourceWitnessV1
    output_surface_a: G1RouteSourceWitnessV1
    output_surface_b: G1RouteSourceWitnessV1
    option_surfaces_u8: tuple[tuple[int, ...], ...]
    option_course_sources: tuple[G1RouteSourceWitnessV1, ...]
    option_surface_a_sources: tuple[G1RouteSourceWitnessV1, ...]
    option_surface_b_sources: tuple[G1RouteSourceWitnessV1, ...]
    surface_a_payload: bytes
    surface_b_payload: bytes

    def __post_init__(self) -> None:
        """重算 route/candidate identity，并逐 byte 回读 course/A/B witness。"""
        _ascii_id(bytes(_u8(self.form_id_u8, label="form id", allow_empty=False)),
                  label="form id")
        if self.result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
            raise DlgRaw15RouteFormEvidenceError("form 只能描述 lexical ambiguity")
        if type(self.matched_frame_count) is not int or self.matched_frame_count != 2:
            raise DlgRaw15RouteFormEvidenceError("G1 form 必须恰有两个候选")
        if type(self.input_scalars) is not tuple or not self.input_scalars:
            raise DlgRaw15RouteFormEvidenceError("form input 不得为空")
        if (type(self.candidate_records) is not tuple
                or len(self.candidate_records) != 2
                or any(not _record(item, label="candidate record", allow_empty=False)
                       for item in self.candidate_records)):
            raise DlgRaw15RouteFormEvidenceError("candidate records 数量/类型漂移")
        expected_candidates = tuple(
            _candidate_identity_from_record(item)
            for item in self.candidate_records)
        identities = tuple(
            _u8(item, label="candidate identity", allow_empty=False)
            for item in self.candidate_identities_u8)
        if (len(identities) != 2 or any(len(item) != 32 for item in identities)
                or identities != expected_candidates
                or identities[0] == identities[1]):
            raise DlgRaw15RouteFormEvidenceError("candidate identity 与 record 漂移")
        expected_route = route_identity_v1(
            self.result_code,
            self.matched_frame_count,
            self.input_scalars,
            identities,
        )
        if tuple(self.route_identity_u8) != expected_route:
            raise DlgRaw15RouteFormEvidenceError("route identity 漂移")
        for payload, label in (
                (self.course_payload, "course payload"),
                (self.surface_a_payload, "surface A payload"),
                (self.surface_b_payload, "surface B payload")):
            if type(payload) is not bytes or not payload:
                raise DlgRaw15RouteFormEvidenceError(f"{label} 必须是非空 bytes")
        if self.surface_a_payload == self.surface_b_payload:
            raise DlgRaw15RouteFormEvidenceError("A/B source payload 不得相同")
        if (len(self.option_surfaces_u8) != 2
                or len(self.option_course_sources) != 2
                or len(self.option_surface_a_sources) != 2
                or len(self.option_surface_b_sources) != 2):
            raise DlgRaw15RouteFormEvidenceError("option witness 数量漂移")
        if (len(set(self.option_surfaces_u8)) != 2
                or any(not _u8(item, label="option surface", allow_empty=False)
                       for item in self.option_surfaces_u8)):
            raise DlgRaw15RouteFormEvidenceError("option surface 不唯一或为空")
        if (any(type(item) is not G1RouteSourceWitnessV1 for item in (
                self.course_source, self.output_course_source,
                self.output_surface_a, self.output_surface_b,
                *self.option_course_sources,
                *self.option_surface_a_sources,
                *self.option_surface_b_sources))):
            raise DlgRaw15RouteFormEvidenceError("source witness 类型漂移")
        _verify_course_and_witnesses(self)

    @property
    def course_payload_sha256_u8(self) -> tuple[int, ...]:
        """返回 canonical course raw payload SHA。"""
        return _sha_payload(self.course_payload, label="course payload")

    @property
    def surface_a_payload_sha256_u8(self) -> tuple[int, ...]:
        """返回 surface A raw payload SHA。"""
        return _sha_payload(self.surface_a_payload, label="surface A payload")

    @property
    def surface_b_payload_sha256_u8(self) -> tuple[int, ...]:
        """返回 surface B raw payload SHA。"""
        return _sha_payload(self.surface_b_payload, label="surface B payload")

    @property
    def form_identity_u8(self) -> tuple[int, ...]:
        """从完整独立证据 record 导出 form identity。"""
        return _u8(tuple(portable_sha256_v1(
            DLG_RAW15_G1_ROUTE_FORM_IDENTITY_DOMAIN_V1,
            (self._base_record(),),
        )), label="form identity", allow_empty=False)

    def _base_record(self) -> tuple[int, ...]:
        """形成不含 form identity 的完整证据 record。"""
        result = [
            DLG_RAW15_G1_ROUTE_FORM_RECORD_V1,
            self.result_code,
            self.matched_frame_count,
        ]
        for label, value in (
                ("form id", self.form_id_u8),
                ("input", self.input_scalars),
                ("route identity", self.route_identity_u8),
                ("candidate identities", tuple(
                    item for identity in self.candidate_identities_u8
                    for item in (len(identity), *identity))),
                ("candidate records", tuple(
                    item for record in self.candidate_records
                    for item in (len(record), *record))),
                ("course payload SHA", self.course_payload_sha256_u8),
                ("course source", self.course_source.canonical_record()),
                ("output", self.output_u8),
                ("output course source", self.output_course_source.canonical_record()),
                ("output A source", self.output_surface_a.canonical_record()),
                ("output B source", self.output_surface_b.canonical_record()),
                ("option surfaces", tuple(
                    item for surface in self.option_surfaces_u8
                    for item in (len(surface), *surface))),
                ("option course sources", tuple(
                    item for source in self.option_course_sources
                    for item in (len(source.canonical_record()),
                                 *source.canonical_record()))),
                ("option A sources", tuple(
                    item for source in self.option_surface_a_sources
                    for item in (len(source.canonical_record()),
                                 *source.canonical_record()))),
                ("option B sources", tuple(
                    item for source in self.option_surface_b_sources
                    for item in (len(source.canonical_record()),
                                 *source.canonical_record()))),
                ("surface A payload SHA", self.surface_a_payload_sha256_u8),
                ("surface B payload SHA", self.surface_b_payload_sha256_u8),
        ):
            _pack(result, value, label=label)
        return tuple(result)

    def canonical_record(self) -> tuple[int, ...]:
        """导出含 form identity 的完整可迁移证据 record。"""
        result = list(self._base_record())
        _pack(result, self.form_identity_u8, label="form identity")
        return tuple(result)


def _verify_course_and_witnesses(evidence: G1RouteFormEvidenceV1) -> None:
    """验证 canonical route row、course span 和两份 A/B raw witness。"""
    course = evidence.course_payload
    if not course.endswith(b"\n") or b"\r" in course:
        raise DlgRaw15RouteFormEvidenceError("course 必须是单行 LF JSONL")
    try:
        row = parse_canonical_json_bytes(course[:-1], require_object=True)
    except Exception as error:  # parser contract maps all malformed JSON to one error
        raise DlgRaw15RouteFormEvidenceError("course JSON canonical readback 失败") from error
    if type(row) is not dict:  # 防止把非 dict 的宿主值带入后续判断
        raise DlgRaw15RouteFormEvidenceError("course row 类型错误")
    if set(row) != set(_FORM_FIELDS):
        raise DlgRaw15RouteFormEvidenceError("course 字段集合漂移")
    expected_ids = tuple(_hex(item, label="course candidate identity")
                         for item in evidence.candidate_identities_u8)
    expected_options = tuple({
        "candidate_identity": expected_ids[index],
        "option_surface": _text_from_u8(surface, label="course option surface", allow_lf=False),
    } for index, surface in enumerate(evidence.option_surfaces_u8))
    expected_output = _text_from_u8(evidence.output_u8, label="course output")
    if (row["schema"] != DLG_RAW15_G1_ROUTE_FORM_EVIDENCE_SCHEMA_V1
            or row["result_code"] != evidence.result_code
            or row["matched_frame_count"] != evidence.matched_frame_count
            or row["form_id"] != _ascii_text(evidence.form_id_u8, label="form id")
            or row["input_surface"]
            != _text_from_u8(tuple(encode_utf8_v1(evidence.input_scalars)),
                             label="course input", allow_lf=False)
            or row["route_identity"] != _hex(evidence.route_identity_u8,
                                              label="course route identity")
            or row["candidate_identities"] != list(expected_ids)
            or row["options"] != list(expected_options)
            or row["output_surface"] != expected_output
            or row["output_max_bytes"] != len(evidence.output_u8)
            or row["license_id"] != "CC0-1.0"):
        raise DlgRaw15RouteFormEvidenceError("course row 与证据 record 漂移")
    line = course[:-1]
    course_digest = _sha_payload(course, label="course payload")
    course_key = tuple(evidence.course_source.logical_key_u8)
    if (evidence.course_source.source_kind
            != DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1
            or bytes(evidence.course_source.raw_sha256_u8) != bytes(course_digest)
            or evidence.course_source.span_end > len(line)
            or row["course_attribution"]
            != _text_from_u8(evidence.course_source.attribution_u8,
                             label="course attribution", allow_lf=False)):
        raise DlgRaw15RouteFormEvidenceError("course source descriptor 漂移")
    if (evidence.course_source.span_start != 0
            or evidence.course_source.span_end != len(line)
            or bytes(evidence.course_source.span_u8) != line
            or course_key != tuple(evidence.output_course_source.logical_key_u8)):
        raise DlgRaw15RouteFormEvidenceError("course source span/SHA 漂移")
    if (evidence.output_course_source.source_kind
            != DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1
            or tuple(evidence.output_course_source.logical_key_u8) != course_key
            or bytes(evidence.output_course_source.raw_sha256_u8)
            != bytes(course_digest)
            or evidence.output_course_source.attribution_u8
            != evidence.course_source.attribution_u8):
        raise DlgRaw15RouteFormEvidenceError("course output source descriptor 漂移")
    output_member = _json_member("output_surface", expected_output, label="output")
    output_start, output_end = _unique_span(line, output_member, label="course output member")
    if (evidence.output_course_source.span_start != output_start
            or evidence.output_course_source.span_end != output_end
            or bytes(evidence.output_course_source.span_u8)
            != line[output_start:output_end]):
        raise DlgRaw15RouteFormEvidenceError("course output member witness 漂移")
    for index, (source, surface) in enumerate(zip(
            evidence.option_course_sources, evidence.option_surfaces_u8), 1):
        if (source.source_kind != DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1
                or tuple(source.logical_key_u8) != course_key
                or bytes(source.raw_sha256_u8) != bytes(course_digest)
                or source.attribution_u8 != evidence.course_source.attribution_u8):
            raise DlgRaw15RouteFormEvidenceError(
                f"course option {index} source descriptor 漂移")
        fragment = _option_fragment(
            expected_ids[index - 1],
            _text_from_u8(surface, label=f"option {index}", allow_lf=False),
        )
        start, end = _unique_span(line, fragment, label=f"course option {index}")
        if bytes(source.span_u8) != line[start:end]:
            raise DlgRaw15RouteFormEvidenceError(
                f"course option {index} witness 漂移")
        if source.span_start != start or source.span_end != end:
            raise DlgRaw15RouteFormEvidenceError(
                f"course option {index} span boundary 漂移")
    for label, source, kind, key, digest, attribution in (
            ("output A", evidence.output_surface_a,
             DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1,
             tuple(evidence.output_surface_a.logical_key_u8),
             _sha_payload(evidence.surface_a_payload, label="surface A payload"),
             evidence.output_surface_a.attribution_u8),
            ("output B", evidence.output_surface_b,
             DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_B_V1,
             tuple(evidence.output_surface_b.logical_key_u8),
             _sha_payload(evidence.surface_b_payload, label="surface B payload"),
             evidence.output_surface_b.attribution_u8)):
        if (source.source_kind != kind
                or bytes(source.raw_sha256_u8) != bytes(digest)
                or not key
                or not attribution):
            raise DlgRaw15RouteFormEvidenceError(f"{label} source descriptor 漂移")
    if (tuple(evidence.output_surface_a.logical_key_u8)
            == tuple(evidence.output_surface_b.logical_key_u8)):
        raise DlgRaw15RouteFormEvidenceError("A/B source logical key 不独立")
    for descriptor, source, payload, label in (
            (row["surface_a"], evidence.output_surface_a,
             evidence.surface_a_payload, "surface A"),
            (row["surface_b"], evidence.output_surface_b,
             evidence.surface_b_payload, "surface B")):
        if (descriptor["relative_path"]
                != bytes(source.logical_key_u8).decode("ascii")
                or descriptor["raw_sha256"]
                != bytes(_sha_payload(payload, label=f"{label} payload")).hex()
                or descriptor["attribution"]
                != _text_from_u8(source.attribution_u8,
                                 label=f"{label} attribution", allow_lf=False)):
            raise DlgRaw15RouteFormEvidenceError(
                f"{label} course descriptor 漂移")
    output = bytes(evidence.output_u8)
    for label, source, payload in (
            ("output A", evidence.output_surface_a, evidence.surface_a_payload),
            ("output B", evidence.output_surface_b, evidence.surface_b_payload)):
        start, end = _unique_span(payload, output, label=label)
        if (bytes(source.span_u8) != output or source.span_start != start
                or source.span_end != end
                or bytes(source.raw_sha256_u8) != bytes(_sha_payload(payload, label=label))):
            raise DlgRaw15RouteFormEvidenceError(f"{label} witness 漂移")
    for index, surface in enumerate(evidence.option_surfaces_u8, 1):
        needle = bytes(surface)
        for label, source, payload in (
                (f"option {index} A", evidence.option_surface_a_sources[index - 1],
                 evidence.surface_a_payload),
                (f"option {index} B", evidence.option_surface_b_sources[index - 1],
                 evidence.surface_b_payload)):
            start, end = _unique_span(payload, needle, label=label)
            if (bytes(source.span_u8) != needle or source.span_start != start
                    or source.span_end != end
                    or bytes(source.raw_sha256_u8) != bytes(_sha_payload(payload, label=label))):
                raise DlgRaw15RouteFormEvidenceError(f"{label} witness 漂移")
            expected_kind = (
                DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1
                if label.endswith(" A") else
                DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_B_V1)
            if source.source_kind != expected_kind:
                raise DlgRaw15RouteFormEvidenceError(f"{label} source kind 漂移")
            expected_key = (tuple(evidence.output_surface_a.logical_key_u8)
                            if expected_kind
                            == DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1
                            else tuple(evidence.output_surface_b.logical_key_u8))
            if tuple(source.logical_key_u8) != expected_key:
                raise DlgRaw15RouteFormEvidenceError(f"{label} source key 漂移")


def build_g1_route_form_evidence_v1(
        resolution: SourceBoundSlotCompositionResolution,
        *,
        form_id: str,
        prompt_surface: str,
        option_surfaces: tuple[str, ...],
        course_logical_key: bytes,
        surface_a_logical_key: bytes,
        surface_b_logical_key: bytes,
        surface_a_payload: bytes,
        surface_b_payload: bytes,
        course_attribution: str,
        surface_a_attribution: str,
        surface_b_attribution: str,
        ) -> G1RouteFormEvidenceV1:
    """从独立 V3 ambiguity 生成 route identity/course/A-B evidence。

    该函数只编译证据，不读取 host、训练、评测或 production catalog。两个
    ``option_surfaces`` 必须由独立 authored source pack 提供；函数不会从实体名、
    子串或 candidate ordinal 猜造问题。
    """
    if type(resolution) is not SourceBoundSlotCompositionResolution:
        raise TypeError("resolution 必须是 V3 composition resolution struct")
    if (resolution.catalog.catalog_schema != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3
            or resolution.result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
            or resolution.matched_frame_count != 2
            or len(resolution.target_candidates) != 2
            or any(item.verdict != SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3
                   for item in resolution.target_candidates)):
        raise DlgRaw15RouteFormEvidenceError("resolution 不是可 offer 的独立 V3 ambiguity")
    try:
        form_id_u8 = _ascii_id(form_id.encode("ascii"), label="form id")
    except UnicodeEncodeError as error:
        raise DlgRaw15RouteFormEvidenceError("form id 必须是 ASCII") from error
    prompt_u8 = _utf8(prompt_surface, label="route prompt", allow_lf=False,
                       allow_empty=False)
    if (type(option_surfaces) is not tuple or len(option_surfaces) != 2
            or len(set(option_surfaces)) != 2):
        raise DlgRaw15RouteFormEvidenceError("必须提供两个不同完整 option surface")
    options_u8 = tuple(
        _utf8(item, label=f"option {index}", allow_lf=False, allow_empty=False)
        for index, item in enumerate(option_surfaces, 1))
    input_scalars = tuple(resolution.input_scalars)
    candidate_records = tuple(item.canonical_record()
                              for item in resolution.target_candidates)
    candidate_ids = tuple(_candidate_identity_from_record(item)
                          for item in candidate_records)
    route_identity = route_identity_v1(
        resolution.result_code,
        resolution.matched_frame_count,
        input_scalars,
        candidate_ids,
    )
    output_u8 = _utf8(
        prompt_surface + "\n" + "\n".join(option_surfaces),
        label="route output",
        allow_lf=True,
        allow_empty=False,
    )
    course_key = _ascii_key(course_logical_key, label="course logical key")
    surface_a_key = _ascii_key(surface_a_logical_key, label="surface A logical key")
    surface_b_key = _ascii_key(surface_b_logical_key, label="surface B logical key")
    if len({course_key, surface_a_key, surface_b_key}) != 3:
        raise DlgRaw15RouteFormEvidenceError("course/A/B logical key 必须互异")
    course_attr_u8 = _utf8(course_attribution, label="course attribution",
                           allow_lf=False, allow_empty=False)
    surface_a_attr_u8 = _utf8(surface_a_attribution, label="surface A attribution",
                              allow_lf=False, allow_empty=False)
    surface_b_attr_u8 = _utf8(surface_b_attribution, label="surface B attribution",
                              allow_lf=False, allow_empty=False)
    digest_a = _sha_payload(surface_a_payload, label="surface A payload")
    digest_b = _sha_payload(surface_b_payload, label="surface B payload")
    candidate_hexes = tuple(_hex(item, label="candidate identity") for item in candidate_ids)
    output_text = _text_from_u8(output_u8, label="output")
    input_text = _text_from_u8(tuple(encode_utf8_v1(input_scalars)),
                               label="input", allow_lf=False)
    course_row = {
        "candidate_identities": list(candidate_hexes),
        "course_attribution": _text_from_u8(course_attr_u8, label="course attribution",
                                             allow_lf=False),
        "form_id": _ascii_text(form_id_u8, label="form id"),
        "input_surface": input_text,
        "license_id": "CC0-1.0",
        "matched_frame_count": 2,
        "options": [{
            "candidate_identity": candidate_hexes[index],
            "option_surface": option_surfaces[index],
        } for index in range(2)],
        "output_max_bytes": len(output_u8),
        "output_surface": output_text,
        "result_code": resolution.result_code,
        "route_identity": _hex(route_identity, label="route identity"),
        "schema": DLG_RAW15_G1_ROUTE_FORM_EVIDENCE_SCHEMA_V1,
        "surface_a": {
            "attribution": _text_from_u8(surface_a_attr_u8, label="surface A attribution",
                                          allow_lf=False),
            "raw_sha256": bytes(digest_a).hex(),
            "relative_path": bytes(surface_a_key).decode("ascii"),
        },
        "surface_b": {
            "attribution": _text_from_u8(surface_b_attr_u8, label="surface B attribution",
                                          allow_lf=False),
            "raw_sha256": bytes(digest_b).hex(),
            "relative_path": bytes(surface_b_key).decode("ascii"),
        },
    }
    course_payload = canonical_json_line(course_row)
    line = course_payload[:-1]
    course_source = _source_witness(
        source_kind=DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1,
        logical_key_u8=course_key,
        payload=course_payload,
        span_u8=line,
        attribution_u8=course_attr_u8,
        label="course line",
    )
    output_member = _json_member("output_surface", output_text, label="course output")
    output_start, output_end = _unique_span(line, output_member, label="course output member")
    output_course_source = G1RouteSourceWitnessV1(
        DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1,
        course_key,
        _sha_payload(course_payload, label="course payload"),
        output_start,
        output_end,
        tuple(line[output_start:output_end]),
        _LICENSE,
        course_attr_u8,
    )
    option_course_sources = []
    for index, (candidate_hex, option) in enumerate(zip(candidate_hexes, option_surfaces), 1):
        fragment = _option_fragment(candidate_hex, option)
        start, end = _unique_span(line, fragment, label=f"course option {index}")
        option_course_sources.append(G1RouteSourceWitnessV1(
            DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1,
            course_key,
            _sha_payload(course_payload, label="course payload"),
            start,
            end,
            tuple(line[start:end]),
            _LICENSE,
            course_attr_u8,
        ))
    output_a = _source_witness(
        source_kind=DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1,
        logical_key_u8=surface_a_key,
        payload=surface_a_payload,
        span_u8=bytes(output_u8),
        attribution_u8=surface_a_attr_u8,
        label="surface A output",
    )
    output_b = _source_witness(
        source_kind=DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_B_V1,
        logical_key_u8=surface_b_key,
        payload=surface_b_payload,
        span_u8=bytes(output_u8),
        attribution_u8=surface_b_attr_u8,
        label="surface B output",
    )
    option_a = tuple(_source_witness(
        source_kind=DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1,
        logical_key_u8=surface_a_key,
        payload=surface_a_payload,
        span_u8=bytes(item),
        attribution_u8=surface_a_attr_u8,
        label=f"surface A option {index}",
    ) for index, item in enumerate(options_u8, 1))
    option_b = tuple(_source_witness(
        source_kind=DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_B_V1,
        logical_key_u8=surface_b_key,
        payload=surface_b_payload,
        span_u8=bytes(item),
        attribution_u8=surface_b_attr_u8,
        label=f"surface B option {index}",
    ) for index, item in enumerate(options_u8, 1))
    return G1RouteFormEvidenceV1(
        form_id_u8,
        resolution.result_code,
        resolution.matched_frame_count,
        input_scalars,
        tuple(route_identity),
        candidate_ids,
        candidate_records,
        course_payload,
        course_source,
        output_u8,
        output_course_source,
        output_a,
        output_b,
        options_u8,
        tuple(option_course_sources),
        option_a,
        option_b,
        surface_a_payload,
        surface_b_payload,
    )


def verify_g1_route_form_evidence_v1(
        evidence: G1RouteFormEvidenceV1,
        ) -> G1RouteFormEvidenceV1:
    """独立重算一份 evidence；返回原 struct，失败即拒绝。"""
    if type(evidence) is not G1RouteFormEvidenceV1:
        raise TypeError("evidence 必须是 G1RouteFormEvidenceV1 struct")
    _verify_course_and_witnesses(evidence)
    return evidence


__all__ = [
    "DLG_RAW15_G1_ROUTE_FORM_EVIDENCE_SCHEMA_V1",
    "DLG_RAW15_G1_ROUTE_FORM_IDENTITY_DOMAIN_V1",
    "DLG_RAW15_G1_ROUTE_SOURCE_KIND_COURSE_V1",
    "DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_A_V1",
    "DLG_RAW15_G1_ROUTE_SOURCE_KIND_SURFACE_B_V1",
    "DlgRaw15RouteFormEvidenceError",
    "G1RouteFormEvidenceV1",
    "G1RouteSourceWitnessV1",
    "build_g1_route_form_evidence_v1",
    "verify_g1_route_form_evidence_v1",
]
