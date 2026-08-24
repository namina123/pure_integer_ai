"""DLG-RAW-09 独立公开 ANSWER frame 的内容锁定目录。

本模块读取一个小型、固定字段的公开 manifest，并借助同一 closure 的课程、词汇
观察和无标签 Evidence planning 重建 ``PublicFrameRuntimeRecipe``。它不把答案表层、
response-act 标签或宿主路径带入运行期 recipe；实际输出仍由 RAW-02 的 G-01/G-03/G-04
生成。基础 V1 manifest 因而保持冻结原字节不变。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrame,
    PublicFrameCatalog,
    PublicFrameRuntimeRecipe,
)
from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    PublicFrameResponseActRuntimeRecipe,
    PublicResponseActCatalogError,
    derive_public_response_act_frame_from_manifest_record,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    ConversationRawCoursePreparationError,
    prepare_public_course,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    GenerationCandidatePackError,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedAnswerCourseError,
    GroundedAnswerEpisode,
    read_grounded_answer_episodes_from_payload,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    surface_pattern_structure_id,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_public_response_act_planning,
    public_response_act_planning_input_from_episode,
)


PUBLIC_ANSWER_FRAME_CATALOG_SCHEMA_V1 = 1
PUBLIC_ANSWER_FRAME_CATALOG_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_answer_frame_v1.jsonl.sample")

_HEX = frozenset("0123456789abcdef")

_MANIFEST_FIELDS = frozenset({
    "catalog_schema",
    "course_raw_sha256",
    "course_relative_path",
    "episode_id",
    "frame_key",
    "lexical_source_a",
    "lexical_source_b",
    "output_max_bytes",
    "pattern_id",
    "recipe_identity_key",
    "structure_id",
})
_RESPONSE_ACT_FIELDS = frozenset({
    "catalog_schema",
    "course_raw_sha256",
    "course_relative_path",
    "episode_id",
    "frame_key",
    "lexical_source_a",
    "lexical_source_b",
    "output_max_bytes",
})


# object-model: exception; interop=DLG-RAW-09
class PublicAnswerFrameCatalogError(ValueError):
    """独立 ANSWER manifest、课程或无标签 Evidence 不能闭合。"""


def _exact(value: Any, *, label: str) -> dict[str, Any]:
    """拒绝由宿主默认值、未知字段或缺字段改变公开目录语义。"""
    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise PublicAnswerFrameCatalogError(f"{label} 字段集合漂移")
    return value


def _strict_positive(value: Any, *, label: str) -> int:
    """只接受协议里可移植的严格正整数，不让 bool/float 混入。"""
    if type(value) is not int or value <= 0:
        raise PublicAnswerFrameCatalogError(f"{label} 必须是严格正整数")
    return value


def _ascii_id(value: Any, *, label: str) -> str:
    """校验 manifest 内有限 ASCII id，不将路径或 locale 写入身份。"""
    if (not isinstance(value, str) or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"
            or any(ord(item) > 0x7F for item in value)
            or any(item not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.:"
                   for item in value)):
        raise PublicAnswerFrameCatalogError(f"{label} 不是稳定 ASCII id")
    return value


def _u8_key(value: Any, *, label: str) -> tuple[int, ...]:
    """恢复完整对象 stable key，拒绝动态对象或负整数 transport。"""
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise PublicAnswerFrameCatalogError(f"{label} 必须是非空严格整数数组")
    return tuple(value)


def _sha256(value: Any, *, label: str) -> tuple[int, ...]:
    """手工把固定小写 SHA-256 hex 映射为明确 raw u8 tuple。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in _HEX for item in value)):
        raise PublicAnswerFrameCatalogError(f"{label} 不是小写 SHA-256")
    return tuple(
        (int(value[cursor], 16) << 4) | int(value[cursor + 1], 16)
        for cursor in range(0, len(value), 2)
    )


def _utf8_text_record(
        value: Any,
        *,
        label: str,
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """把课程 parser 给出的文本显式回读为 scalar 与 UTF-8 整数记录。"""
    if not isinstance(value, str) or not value:
        raise PublicAnswerFrameCatalogError(f"{label} 必须是非空文本")
    scalars = tuple(ord(item) for item in value)
    try:
        payload = encode_utf8_v1(scalars)
    except (TypeError, ValueError) as error:
        raise PublicAnswerFrameCatalogError(f"{label} 无法编码为 UTF-8") from error
    readback = intake_raw_conversation_vector(payload)
    if (not readback.accepted or readback.unicode_scalars != scalars
            or encode_utf8_v1(readback.unicode_scalars) != payload):
        raise PublicAnswerFrameCatalogError(f"{label} UTF-8 readback 漂移")
    return scalars, payload


def _find_all_u8_subsequence_v1(
        payload: bytes,
        needle: tuple[int, ...],
        ) -> tuple[int, ...]:
    """按逐位置、允许重叠的冻结 u8 规则扫描 raw course bytes。"""
    if type(payload) is not bytes:
        raise TypeError("answer catalog course payload 必须是 raw bytes")
    if (not isinstance(needle, tuple) or not needle
            or any(type(item) is not int or item < 0 or item > 255
                   for item in needle)):
        raise PublicAnswerFrameCatalogError("answer catalog scan needle 非法")
    if len(needle) > len(payload):
        return ()
    starts = []
    for start in range(len(payload) - len(needle) + 1):
        if all(payload[start + offset] == expected
               for offset, expected in enumerate(needle)):
            starts.append(start)
    return tuple(starts)


def _unique_claim_text_span(
        course_payload: bytes,
        claim_bytes: tuple[int, ...],
        ) -> tuple[int, int]:
    """定位唯一 canonical ``claim_text`` field，并返回其内容而非 JSON 标记 span。"""
    prefix = tuple(b'"claim_text":"')
    marker = (*prefix, *claim_bytes, 0x22)
    starts = _find_all_u8_subsequence_v1(course_payload, marker)
    if len(starts) != 1:
        raise PublicAnswerFrameCatalogError("ANSWER claim source span 缺失或不唯一")
    start = starts[0] + len(prefix)
    end = start + len(claim_bytes)
    if tuple(course_payload[start:end]) != claim_bytes:
        raise PublicAnswerFrameCatalogError("ANSWER claim source span bytes 漂移")
    return start, end


def _validate_answer_pattern(
        *,
        course_payload: bytes,
        course_relative_path: str,
        course_sha256: tuple[int, ...],
        episode: GroundedAnswerEpisode,
        pattern_id: int,
        structure_id: int,
        ) -> None:
    """在目录装载时确认 recipe 只能指向当前课程实际学得的 ANSWER pattern。"""
    if episode.question.answer_plan.response_act != "ANSWER":
        raise PublicAnswerFrameCatalogError("ANSWER frame 引用了非 ANSWER episode")
    if len(episode.question.answer_plan.ordered_claim_ids) != 1:
        raise PublicAnswerFrameCatalogError("ANSWER frame 只支持单个有序 claim")
    try:
        prepared = prepare_public_course(
            course_payload,
            course_relative_path=course_relative_path,
            course_raw_sha256=course_sha256,
        )
        pattern = prepared.pack.pattern(pattern_id)
    except (ConversationRawCoursePreparationError, GenerationCandidatePackError,
            TypeError, ValueError, RuntimeError) as error:
        raise PublicAnswerFrameCatalogError(
            "ANSWER recipe pattern 不属于当前内容锁课程") from error
    if pattern.response_act != "ANSWER":
        raise PublicAnswerFrameCatalogError("ANSWER recipe pattern response act 漂移")
    if pattern.claim_count != len(episode.question.answer_plan.ordered_claim_ids):
        raise PublicAnswerFrameCatalogError("ANSWER recipe pattern claim 数漂移")
    if surface_pattern_structure_id(pattern) != structure_id:
        raise PublicAnswerFrameCatalogError("ANSWER recipe pattern/structure 漂移")


def _course_payload(
        closure: PublicSourcePayloadClosureV1,
        relative_path: str,
        expected_sha256: tuple[int, ...],
        ) -> bytes:
    """从固定 logical key 回读课程 bytes 并逐字节核验其内容身份。"""
    if (not isinstance(relative_path, str) or "\\" in relative_path
            or not relative_path):
        raise PublicAnswerFrameCatalogError("course_relative_path 非法")
    try:
        logical_key = relative_path.encode("ascii")
    except UnicodeEncodeError as error:
        raise PublicAnswerFrameCatalogError("course_relative_path 必须是 ASCII") from error
    parts = logical_key.split(b"/")
    if (len(parts) != 3 or parts[:2] != [b"data", b"ph2"]
            or any(item in {b"", b".", b".."} for item in parts)):
        raise PublicAnswerFrameCatalogError("course_relative_path 越出 logical namespace")
    try:
        record = closure.record_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise PublicAnswerFrameCatalogError("课程不在 public payload closure") from error
    if tuple(record.raw_sha256) != expected_sha256:
        raise PublicAnswerFrameCatalogError("课程 raw SHA-256 漂移")
    return record.raw_payload


def _episode_for_id(
        episodes: tuple[GroundedAnswerEpisode, ...],
        episode_id: str,
        ) -> GroundedAnswerEpisode:
    """选择唯一训练 episode，避免按列表序或可答性回退。"""
    matches = tuple(item for item in episodes if item.episode_id == episode_id)
    if len(matches) != 1:
        raise PublicAnswerFrameCatalogError("课程中找不到唯一 ANSWER episode")
    return matches[0]


def _manifest_identity(value: Any) -> ObjectIdentity:
    """从完整 record 恢复 recipe identity，确保它是最小指令对象。"""
    try:
        identity = ObjectIdentity.from_stable_key(
            _u8_key(value, label="recipe_identity_key"))
    except (TypeError, ValueError) as error:
        raise PublicAnswerFrameCatalogError("recipe identity 损坏") from error
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise PublicAnswerFrameCatalogError("recipe identity 不是 MinimalInstruction")
    return identity


def _answer_frame_from_manifest(
        raw: Any,
        *,
        raw_line_sha256: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrame:
    """把紧凑公开 manifest 规约为 RAW-02 所需的完整 ANSWER frame。"""
    manifest = _exact(raw, label="public answer frame")
    if manifest["catalog_schema"] != PUBLIC_ANSWER_FRAME_CATALOG_SCHEMA_V1:
        raise PublicAnswerFrameCatalogError("answer catalog schema 未注册")
    course_relative_path = manifest["course_relative_path"]
    episode_id = _ascii_id(manifest["episode_id"], label="episode_id")
    _ascii_id(manifest["frame_key"], label="frame_key")
    course_sha256 = _sha256(manifest["course_raw_sha256"], label="course_raw_sha256")
    output_max_bytes = _strict_positive(
        manifest["output_max_bytes"], label="output_max_bytes")
    pattern_id = _strict_positive(manifest["pattern_id"], label="pattern_id")
    structure_id = _strict_positive(manifest["structure_id"], label="structure_id")
    recipe_identity = _manifest_identity(manifest["recipe_identity_key"])
    course_payload = _course_payload(closure, course_relative_path, course_sha256)
    try:
        episodes = read_grounded_answer_episodes_from_payload(
            course_payload, train_only=True)
    except GroundedAnswerCourseError as error:
        raise PublicAnswerFrameCatalogError("公开课程结构或 train split 漂移") from error
    episode = _episode_for_id(episodes, episode_id)
    response_manifest = {
        key: manifest[key]
        for key in _RESPONSE_ACT_FIELDS
        if key != "catalog_schema"
    }
    response_manifest["catalog_schema"] = 2
    try:
        frame = derive_public_response_act_frame_from_manifest_record(
            response_manifest,
            raw_line_sha256=raw_line_sha256,
            source_payload_closure=closure,
            ordinal=1,
        )
    except (PublicResponseActCatalogError, TypeError, ValueError) as error:
        raise PublicAnswerFrameCatalogError("ANSWER frame 基础来源投影失败") from error
    if not isinstance(frame.recipe, PublicFrameResponseActRuntimeRecipe):
        raise PublicAnswerFrameCatalogError("ANSWER frame 未形成受限课程投影")
    try:
        planning_input = public_response_act_planning_input_from_episode(episode)
        build = compile_public_response_act_planning(
            planning_input, frame.question.target_branch)
    except (TypeError, ValueError) as error:
        raise PublicAnswerFrameCatalogError("ANSWER 无法形成无标签 Evidence planning") from error
    if (len(build.candidate_bindings) != 1
            or len(build.candidate_bindings[0].candidate.evidence) != 1):
        raise PublicAnswerFrameCatalogError("ANSWER 必须有唯一 candidate/evidence")
    candidate = build.candidate_bindings[0].candidate
    if candidate.state.stable_key() != (1, 0):
        raise PublicAnswerFrameCatalogError("ANSWER candidate 必须是支持且无反证")
    source_evidence = tuple(
        item for item in episode.question.evidence
        if item.support == 1 and item.refute == 0)
    if len(source_evidence) != 1:
        raise PublicAnswerFrameCatalogError("ANSWER 课程缺唯一支持 evidence")
    claim_scalars, claim_bytes = _utf8_text_record(
        source_evidence[0].claim_text,
        label="ANSWER claim text",
    )
    claim_start, claim_end = _unique_claim_text_span(
        course_payload,
        claim_bytes,
    )
    _validate_answer_pattern(
        course_payload=course_payload,
        course_relative_path=course_relative_path,
        course_sha256=course_sha256,
        episode=episode,
        pattern_id=pattern_id,
        structure_id=structure_id,
    )
    course_record = next(
        (item for item in frame.source_records
         if item.record_id == frame.recipe.course_source_record_id),
        None,
    )
    if course_record is None:
        raise PublicAnswerFrameCatalogError("ANSWER course source record 缺失")
    evidence = candidate.evidence[0]
    evidence_record = next(
        (item for item in frame.source_records
         if item.source == evidence.hypothesis.observation),
        None,
    )
    if evidence_record is None:
        raise PublicAnswerFrameCatalogError("ANSWER Evidence source record 缺失")
    source_records = tuple(sorted(
        (replace(
            course_record,
            span=(claim_start, claim_end),
            span_bytes=tuple(claim_bytes),
            span_scalars=claim_scalars,
        ),
         *(item for item in frame.source_records if item != course_record)),
        key=lambda item: item.source.stable_key(),
    ))
    recipe = PublicFrameRuntimeRecipe(
        recipe_identity,
        course_relative_path,
        course_sha256,
        episode_id,
        candidate.state,
        candidate.evidence,
        (evidence_record.record_id,),
        course_record.record_id,
        claim_scalars,
        pattern_id,
        structure_id,
        output_max_bytes,
        1,
    )
    try:
        return replace(frame, source_records=source_records, recipe=recipe)
    except (TypeError, ValueError) as error:
        raise PublicAnswerFrameCatalogError("ANSWER frame/recipe 不闭合") from error


def load_public_answer_frame_catalog_from_closure(
        closure: PublicSourcePayloadClosureV1,
        ) -> PublicFrameCatalog:
    """从完整 closure 重建独立 ANSWER catalog，绝不读取物理目录。"""
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise PublicAnswerFrameCatalogError("answer catalog 需要完整 payload closure")
    try:
        payload = closure.payload_for(PUBLIC_ANSWER_FRAME_CATALOG_LOGICAL_KEY_V1)
    except PublicSourcePayloadProviderError as error:
        raise PublicAnswerFrameCatalogError("answer manifest 不在 payload closure") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise PublicAnswerFrameCatalogError("answer catalog JSONL 换行非法")
    lines = payload[:-1].split(b"\n")
    if len(lines) != 1 or not lines[0]:
        raise PublicAnswerFrameCatalogError("answer catalog 必须恰有一条 record")
    try:
        raw = parse_canonical_json_bytes(lines[0], require_object=True)
    except DatasetContractError as error:
        raise PublicAnswerFrameCatalogError("answer manifest 不是 canonical JSON") from error
    frame = _answer_frame_from_manifest(
        raw,
        raw_line_sha256=tuple(public_source_payload_sha256_v1(lines[0])),
        closure=closure,
    )
    return PublicFrameCatalog(
        tuple(public_source_payload_sha256_v1(payload)),
        (frame,),
    )


__all__ = [
    "PUBLIC_ANSWER_FRAME_CATALOG_LOGICAL_KEY_V1",
    "PUBLIC_ANSWER_FRAME_CATALOG_SCHEMA_V1",
    "PublicAnswerFrameCatalogError",
    "load_public_answer_frame_catalog_from_closure",
]
