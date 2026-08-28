"""公开对话课程到真实训练输入的最小装配层。

JSONL 只作为交换载体；本模块保留原始字段的来源摘要，并把每条公开课程
投影为可跨语言复现的整数记录和 ``CollectedItem``。它不生成标注，也不把
模板回放伪装成理解结果。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    LANG_ZH,
    MODALITY_LANGUAGE,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
    DialogueContentSpan,
    SpeakerSpan,
    SOURCE_BARE_TEXT,
)
from pure_integer_ai.experiments.integer_token_index import (
    IntegerAggregateIndex,
    IntegerTokenIndex,
    load_integer_aggregate_index,
    load_integer_token_index,
)


CONVERSATION_TRAINING_PACK_PROTOCOL_V1 = 2
OASST1_DIALOGUE_COURSE_FORMAT_V2 = "PURE_INTEGER_AI_OASST1_DIALOGUE_COURSE_V2"
OPENASSISTANT_DIALOGUE_COURSE_FORMAT_V2 = (
    "PURE_INTEGER_AI_OPENASSISTANT_DIALOGUE_COURSE_V2")
KDCONV_DIALOGUE_COURSE_FORMAT_V1 = (
    "PURE_INTEGER_AI_KDCONV_DIALOGUE_COURSE_V1")
LLM_ASSISTED_DIALOGUE_COURSE_FORMAT_V1 = (
    "PURE_INTEGER_AI_LLM_ASSISTED_DIALOGUE_COURSE_V1")
_DIALOGUE_COURSE_FORMATS = frozenset({
    OASST1_DIALOGUE_COURSE_FORMAT_V2,
    OPENASSISTANT_DIALOGUE_COURSE_FORMAT_V2,
    KDCONV_DIALOGUE_COURSE_FORMAT_V1,
    LLM_ASSISTED_DIALOGUE_COURSE_FORMAT_V1,
})
DIALOGUE_SPEAKER_USER = 1
DIALOGUE_SPEAKER_ASSISTANT = 2
_DIALOGUE_SPEAKERS = frozenset({
    DIALOGUE_SPEAKER_USER, DIALOGUE_SPEAKER_ASSISTANT,
})
_GENERALIZATION_SOURCE_HASHER = Hasher(
    "conversation.typed.generalization.source.v1")
_SPLITS = frozenset({"train", "heldout", "negative"})
_SKIP_KEYS = frozenset({
    "license_id", "raw_sha256", "sha256", "source_key", "source_namespace",
    "source_cluster", "upstream_url", "relative_path", "attribution",
    "snapshot_id", "source_revision", "item_id", "seed_id", "episode_id",
    "frame_key", "candidate_id", "candidate_ids", "source_ids", "use_ref_id",
})
# 这些键可以出现在 authored/course record 的任意嵌套层。只提取语言表层和
# 结构载体原文；不能把 evaluation dimension、整数 id、license 或候选状态
# 重新当作语言送进 formal_train。
_SURFACE_KEYS = (
    "question_surface", "contrast_question_surface", "origin_question_surface",
    "input_surface", "question", "context_surface", "context", "observed_text",
    "raw_text", "answer_surface", "expected_answer", "response_surface",
    "output_surface", "generated_proposition_surface", "accepted_surface_variants",
    "accepted_surfaces", "surface", "surface_form", "surface_fragment",
    "derived_text", "text",
)
_SURFACE_KEY_SET = frozenset(_SURFACE_KEYS)


class ConversationTrainingPackError(ValueError):
    """公开对话课程无法形成确定、隔离的训练 pack。"""


# object-model: value; representation=struct; interop=stable-integer-key
@dataclass(frozen=True, slots=True)
class DialogueTurnRecord:
    """公开对话路径中的一个显式 turn。"""

    turn_ordinal: int
    speaker_role: int
    message_id: str
    surface: str

    def __post_init__(self) -> None:
        if (type(self.turn_ordinal) is not int or self.turn_ordinal <= 0
                or type(self.speaker_role) is not int
                or self.speaker_role not in _DIALOGUE_SPEAKERS
                or not isinstance(self.message_id, str) or not self.message_id
                or not isinstance(self.surface, str)
                or len(self.surface.strip()) < 2
                or self.surface != self.surface.strip()):
            raise ConversationTrainingPackError("dialogue turn 字段非法")

    @property
    def rendered_surface(self) -> str:
        """Return content only; role is carried by the integer speaker span."""
        return self.surface

    def canonical_record(self) -> tuple[int, ...]:
        message = tuple(map(ord, self.message_id))
        surface = tuple(map(ord, self.surface))
        return (
            self.turn_ordinal, self.speaker_role,
            len(message), *message, len(surface), *surface,
        )


def _speaker_identity(role: int) -> ObjectIdentity:
    """把公开 speaker role 映射为稳定纯整数图身份。"""
    if role not in _DIALOGUE_SPEAKERS:
        raise ConversationTrainingPackError("speaker role 未注册")
    return role_identity((21402, 60, role))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encode_integer_record(values: tuple[int, ...]) -> bytes:
    """跨语言编码非负整数记录，避免把稳定键截断为固定 32 位。"""
    out = bytearray()
    for value in values:
        if type(value) is not int or value < 0:
            raise ConversationTrainingPackError(
                "integer record 只能包含非负严格整数")
        raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8),
                             "big", signed=False)
        out.extend(len(raw).to_bytes(4, "big", signed=False))
        out.extend(raw)
    return bytes(out)


def _scalar_text(value: Any) -> tuple[str, ...]:
    """递归提取可作为自然语言表层的字符串，跳过身份和许可证字段。"""
    if isinstance(value, str):
        value = value.strip()
        return (value,) if value else ()
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_scalar_text(item))
        return tuple(result)
    if isinstance(value, dict):
        result: list[str] = []
        for key in sorted(value):
            if key in _SKIP_KEYS:
                continue
            result.extend(_scalar_text(value[key]))
        return tuple(result)
    return ()


def _surface_for_record(record: dict[str, Any]) -> tuple[str, ...]:
    """从嵌套公开记录提取可读表层，拒绝把 metadata 送进语言通道。

    authored 课程常把 ``accepted_surface_variants`` 放在
    ``expected_payload`` 内，把场景文本放在 ``observation_payload`` 内；
    DLG/LC 课程又把原文放在 ``question``、``accepted`` 或 ``raw_text`` 内。
    这里按固定键序递归提取，并对同一记录去重。没有任何登记表层时直接
    拒绝该记录，而不是退回“递归所有字符串”的不安全兜底。
    """
    values: list[str] = []

    def take(key: str, value: Any) -> None:
        """按字段形状提取表层，避免 accepted object 的 id 泄漏。"""
        if key == "question" and isinstance(value, (dict, list)):
            # ``question`` 可能是带 question_surface/context_surface 的对象；
            # 继续走白名单，避免把 question object 的 id/slot 元数据送入语言。
            visit(value)
            return
        if key in {"accepted_surface_variants", "accepted_surfaces"} \
                and isinstance(value, list):
            values.extend(item.strip() for item in value
                          if isinstance(item, str) and item.strip())
            for item in value:
                if isinstance(item, (dict, list)):
                    visit(item)
            return
        values.extend(_scalar_text(value))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            # 固定字段序优先于宿主 dict 顺序；同一字段在不同嵌套层按出现
            # 的结构顺序继续递归，保证 canonical record 可跨语言重放。
            for key in _SURFACE_KEYS:
                if key in value:
                    take(key, value[key])
            for key in sorted(value):
                if key not in _SURFACE_KEY_SET:
                    child = value[key]
                    if isinstance(child, (dict, list)):
                        visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    visit(child)

    visit(record)
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen and len(value) >= 2:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _record_id(record: dict[str, Any], path: Path, line_number: int,
               source_identity: str | None = None) -> str:
    label = path.as_posix() if source_identity is None else source_identity
    for key in ("seed_id", "episode_id", "item_id", "frame_key"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return f"{label}::{value}"
    return f"{label}::line-{line_number}"


@dataclass(frozen=True, slots=True)
class DialogueTrainingCase:
    """一条公开课程记录的不可变训练投影。"""

    case_id: str
    split: str
    family: str
    source_path: str
    source_line: int
    surfaces: tuple[str, ...]
    causal_pairs: tuple[tuple[int, int], ...]
    integer_record: tuple[int, ...]
    # Optional typed-course attachments are appended after the legacy fields so
    # older positional constructors remain valid.  They are never inferred
    # from raw surface text.
    typed_payload: CanonicalJsonObject | None = None
    payload_kind: str | None = None
    source_ref: SourceRef | None = None
    expected_state: str | None = None
    expected_payload: CanonicalJsonObject | None = None
    indexed_surface: bool = False
    token_index: IntegerTokenIndex | None = None
    token_index_ordinal: int | None = None
    aggregate_index: IntegerAggregateIndex | None = None
    aggregate_index_ordinal: int | None = None
    dialogue_turns: tuple[DialogueTurnRecord, ...] = ()

    def __post_init__(self) -> None:
        if (not self.case_id or self.split not in _SPLITS
                or (not self.surfaces and self.aggregate_index is None)):
            raise ConversationTrainingPackError("dialogue case 字段非法")
        if any(type(item) is not int or item < 0 for item in self.integer_record):
            raise ConversationTrainingPackError("dialogue case integer record 非法")
        if (self.typed_payload is None) != (self.payload_kind is None):
            raise ConversationTrainingPackError(
                "typed payload 与 payload_kind 必须成对存在")
        if self.typed_payload is not None and not isinstance(
                self.typed_payload, CanonicalJsonObject):
            raise ConversationTrainingPackError("typed payload 类型非法")
        if self.payload_kind is not None and (
                not self.payload_kind or self.payload_kind.strip() != self.payload_kind):
            raise ConversationTrainingPackError("payload_kind 非法")
        if self.source_ref is not None and not isinstance(self.source_ref, SourceRef):
            raise ConversationTrainingPackError("typed source_ref 类型非法")
        if self.expected_state is not None and (
                not isinstance(self.expected_state, str)
                or not self.expected_state
                or self.expected_state.strip() != self.expected_state):
            raise ConversationTrainingPackError("expected_state 非法")
        if self.expected_payload is not None and not isinstance(
                self.expected_payload, CanonicalJsonObject):
            raise ConversationTrainingPackError("expected_payload 类型非法")
        if type(self.indexed_surface) is not bool:
            raise ConversationTrainingPackError("indexed_surface 类型非法")
        if self.indexed_surface != (self.token_index is not None):
            raise ConversationTrainingPackError(
                "indexed_surface 与 token_index 必须一致")
        if (self.token_index is not None and self.aggregate_index is None and (
                type(self.token_index_ordinal) is not int
                or self.token_index_ordinal < 0)):
            raise ConversationTrainingPackError("token_index ordinal 非法")
        if self.aggregate_index is None and self.aggregate_index_ordinal is not None:
            raise ConversationTrainingPackError(
                "aggregate_index_ordinal 必须配套 aggregate_index")
        if self.aggregate_index is not None and (
                self.token_index is None
                or type(self.aggregate_index_ordinal) is not int
                or self.aggregate_index_ordinal < 0):
            raise ConversationTrainingPackError(
                "aggregate_index 必须配套 token_index 与 ordinal")
        if (not isinstance(self.dialogue_turns, tuple)
                or any(not isinstance(turn, DialogueTurnRecord)
                       for turn in self.dialogue_turns)):
            raise ConversationTrainingPackError(
                "dialogue_turns 必须是 DialogueTurnRecord tuple")
        if self.dialogue_turns:
            ordinals = tuple(turn.turn_ordinal for turn in self.dialogue_turns)
            roles = tuple(turn.speaker_role for turn in self.dialogue_turns)
            if (ordinals != tuple(range(1, len(ordinals) + 1))
                    or len(self.dialogue_turns) < 2
                    or roles[-1] != DIALOGUE_SPEAKER_ASSISTANT
                    or any(role != (DIALOGUE_SPEAKER_USER
                                     if ordinal % 2 else
                                     DIALOGUE_SPEAKER_ASSISTANT)
                           for ordinal, role in zip(ordinals, roles))
                    or self.surfaces != tuple(
                        turn.rendered_surface for turn in self.dialogue_turns)):
                raise ConversationTrainingPackError(
                    "dialogue turns 顺序、角色或表层绑定非法")

    @property
    def raw_text(self) -> str:
        """返回供 observe 消费的完整表层，保留问句、上下文和回答顺序。"""
        if not self.surfaces and self.aggregate_index is not None:
            return self.aggregate_index.render(
                self.token_index, self.aggregate_index_ordinal)
        return "\n".join(self.surfaces)

    def canonical_record(self) -> tuple[int, ...]:
        return self.integer_record


@dataclass(frozen=True, slots=True)
class DialogueTrainingPack:
    """公开课程的 train/heldout/negative 分账。"""

    cases: tuple[DialogueTrainingCase, ...]
    source_files: tuple[tuple[str, str, int], ...]
    pack_sha256: str

    def __post_init__(self) -> None:
        if not self.cases or not self.source_files:
            raise ConversationTrainingPackError("dialogue pack 不能为空")
        ids = tuple(item.case_id for item in self.cases)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ConversationTrainingPackError("dialogue case identity 不稳定")

    @property
    def split_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple((split, sum(item.split == split for item in self.cases))
                     for split in ("train", "heldout", "negative"))

    @property
    def dialogue_structure_counts(self) -> tuple[tuple[str, int], ...]:
        structured = tuple(case for case in self.cases if case.dialogue_turns)
        return (
            ("structured_cases", len(structured)),
            ("turns", sum(len(case.dialogue_turns) for case in structured)),
            ("prompt_response_pairs", len(structured)),
            ("multiturn_cases", sum(
                len(case.dialogue_turns) > 2 for case in structured)),
        )

    def training_items(self, *, split: str = "train",
                       causal_only: bool = False,
                       defer_indexed_surface: bool = False) -> list[CollectedItem]:
        """把指定 train/heldout split 投影为 formal_train 可直接消费的语言项。"""
        if split not in {"train", "heldout"}:
            raise ConversationTrainingPackError("训练投影只允许 train/heldout")
        return self.items_for_split(
            split=split, causal_only=causal_only,
            defer_indexed_surface=defer_indexed_surface)

    def evaluation_items(self, *, split: str | None = None,
                         causal_only: bool = False) -> list[CollectedItem]:
        """投影公开评测项，允许显式 negative/ambiguous 进入只读 V-00。"""
        if split is not None and split not in _SPLITS:
            raise ConversationTrainingPackError("评测投影 split 非法")
        return self.items_for_split(split=split, causal_only=causal_only)

    def items_for_split(self, *, split: str | None,
                        causal_only: bool = False,
                        defer_indexed_surface: bool = False) -> list[CollectedItem]:
        """按公开 split 建立统一 CollectedItem；不改变课程标签或来源。"""
        result = []
        for case in self.cases:
            if split is not None and case.split != split:
                continue
            if causal_only and not case.causal_pairs:
                continue
            text = case.raw_text
            # Indexed long-form courses keep the reconstructed raw surface and
            # let the language provider create the final token sequence once.
            # Legacy courses retain the historical eager character projection.
            tokens = ([]) if (case.indexed_surface and defer_indexed_surface) \
                else list(text)
            source_ref = case.source_ref
            source = SOURCE_BARE_TEXT
            if source_ref is not None:
                # Authored typed records carry authoritative provenance.  The
                # round runtime switches WorkMemory sessions at version
                # boundaries, so this source must remain intact.
                source = source_ref.source_kind
            speaker_spans: tuple[SpeakerSpan, ...] = ()
            content_spans: tuple[DialogueContentSpan, ...] = ()
            if case.dialogue_turns:
                spans = []
                contents = []
                cursor = 0
                for ordinal, turn in enumerate(case.dialogue_turns):
                    rendered = turn.rendered_surface
                    content_start = cursor + len(rendered) - len(turn.surface)
                    content_end = cursor + len(rendered)
                    end = content_end
                    if ordinal + 1 < len(case.dialogue_turns):
                        end += 1  # raw_text 中相邻 turn 的单个换行归前一 turn。
                    spans.append(SpeakerSpan(
                        cursor,
                        end,
                        turn.turn_ordinal,
                        _speaker_identity(turn.speaker_role),
                    ))
                    contents.append(DialogueContentSpan(
                        content_start, content_end, turn.turn_ordinal))
                    cursor = end
                if cursor != len(text):
                    raise ConversationTrainingPackError(
                        "dialogue speaker spans 与 raw_text 不一致")
                speaker_spans = tuple(spans)
                content_spans = tuple(contents)
            result.append(CollectedItem(
                tokens=tokens,
                raw_text=text,
                role_seq=[1] * len(tokens),
                causal_pairs=list(case.causal_pairs),
                collect_type=COLLECT_PRECEDES,
                source=source,
                strength=1,
                domain=DOMAIN_TEXT,
                lang=LANG_ZH,
                modality=MODALITY_LANGUAGE,
                source_ref=source_ref,
                typed_payload=case.typed_payload,
                payload_kind=case.payload_kind,
                token_index=case.token_index,
                token_index_ordinal=case.token_index_ordinal,
                aggregate_index=case.aggregate_index,
                aggregate_index_ordinal=case.aggregate_index_ordinal,
                speaker_spans=speaker_spans,
                dialogue_content_spans=content_spans,
            ))
        return result


def _canonical_case_record(case_id: str, split: str, family: str,
                           source_path: str, line: int,
                           surfaces: tuple[str, ...],
                           causal_pairs: tuple[tuple[int, int], ...],
                           typed_payload: CanonicalJsonObject | None,
                           payload_kind: str | None,
                           source_ref: SourceRef | None,
                           expected_state: str | None,
                           expected_payload: CanonicalJsonObject | None,
                           indexed_surface: bool,
                           token_index: IntegerTokenIndex | None,
                           token_index_ordinal: int | None,
                           aggregate_index: IntegerAggregateIndex | None,
                           aggregate_index_ordinal: int | None,
                           dialogue_turns: tuple[DialogueTurnRecord, ...],
                           ) -> tuple[int, ...]:
    values: list[int] = [CONVERSATION_TRAINING_PACK_PROTOCOL_V1, len(case_id)]
    values.extend(ord(item) for item in case_id)
    values.extend((len(split), *map(ord, split), len(family), *map(ord, family)))
    values.extend((len(source_path), *map(ord, source_path), line, len(surfaces)))
    for surface in surfaces:
        values.extend((len(surface), *map(ord, surface)))
    values.append(len(causal_pairs))
    for left, right in causal_pairs:
        values.extend((left, right))
    values.append(0 if typed_payload is None else 1)
    if typed_payload is not None:
        encoded = canonical_json_bytes(typed_payload.to_value())
        values.extend((len(encoded), *encoded))
        assert payload_kind is not None
        values.extend((len(payload_kind), *payload_kind.encode("utf-8")))
    source_key = () if source_ref is None else source_ref.stable_key()
    values.extend((len(source_key), *source_key))
    if expected_state is None:
        values.append(0)
    else:
        values.extend((1, len(expected_state), *map(ord, expected_state)))
    if expected_payload is None:
        values.append(0)
    else:
        encoded = canonical_json_bytes(expected_payload.to_value())
        values.extend((1, len(encoded), *encoded))
    values.append(1 if indexed_surface else 0)
    if token_index is None:
        values.append(0)
    else:
        encoded = token_index.sha256.encode("ascii")
        values.extend((1, len(encoded), *encoded,
                       token_index_ordinal if token_index_ordinal is not None else 0))
    # Keep legacy records byte-identical; aggregate-bearing records carry a
    # trailing versioned reference so existing pack identities remain stable.
    if aggregate_index is not None:
        encoded = aggregate_index.sha256.encode("ascii")
        values.extend((1, len(encoded), *encoded,
                       aggregate_index_ordinal
                       if aggregate_index_ordinal is not None else 0))
    if dialogue_turns:
        values.extend((2, len(dialogue_turns)))
        for turn in dialogue_turns:
            record = turn.canonical_record()
            values.extend((len(record), *record))
    return tuple(values)


def _public_source_ref(record: dict[str, Any], path: Path,
                       line_number: int) -> SourceRef | None:
    """接纳普通公开课程的完整 SourceRef，非法声明必须 fail closed。"""
    raw = record.get("source_ref_key")
    if raw is None:
        return None
    if (not isinstance(raw, list) or len(raw) != 11
            or any(type(item) is not int for item in raw)):
        raise ConversationTrainingPackError(
            f"source_ref_key 非法: {path.name}:{line_number}")
    try:
        source = SourceRef.from_stable_key(tuple(raw))
    except (TypeError, ValueError) as error:
        raise ConversationTrainingPackError(
            f"source_ref_key 不可恢复: {path.name}:{line_number}") from error
    declared = (
        record.get("source_kind"), record.get("source_id"),
        record.get("source_record_id"),
    )
    expected = (source.source_kind, source.source_id, source.document_id)
    for value, target in zip(declared, expected):
        if value is not None and value != target:
            raise ConversationTrainingPackError(
                f"source_ref_key 与来源字段冲突: {path.name}:{line_number}")
    return source


def _dialogue_projection(record: dict[str, Any], path: Path,
                         line_number: int) -> tuple[DialogueTurnRecord, ...]:
    """解析正式公开 turn schema；非该格式保持普通表层课程行为。"""
    if record.get("format") not in _DIALOGUE_COURSE_FORMATS:
        return ()
    raw_turns = record.get("dialogue_turns")
    if not isinstance(raw_turns, list) or len(raw_turns) < 2:
        raise ConversationTrainingPackError(
            f"dialogue_turns 非法: {path.name}:{line_number}")
    turns = []
    for raw in raw_turns:
        if not isinstance(raw, dict) or set(raw) != {
                "message_id", "speaker_role", "surface", "turn_ordinal"}:
            raise ConversationTrainingPackError(
                f"dialogue turn schema 非法: {path.name}:{line_number}")
        turns.append(DialogueTurnRecord(
            raw["turn_ordinal"], raw["speaker_role"],
            raw["message_id"], raw["surface"],
        ))
    result = tuple(turns)
    path_ids = record.get("path_message_ids")
    if path_ids != [turn.message_id for turn in result]:
        raise ConversationTrainingPackError(
            f"dialogue path identity 漂移: {path.name}:{line_number}")
    if (record.get("path_turn_count") != len(result)
            or record.get("context_turn_count") != len(result) - 1
            or record.get("prompt_turn_ordinal") != len(result) - 1
            or record.get("response_turn_ordinal") != len(result)
            or not _input_surface_binds_turns(
                record.get("input_surface"), result[:-1])
            or record.get("response_surface") != result[-1].surface):
        raise ConversationTrainingPackError(
            f"dialogue prompt/response 绑定漂移: {path.name}:{line_number}")
    return result


def _input_surface_binds_turns(
        value: Any, turns: tuple[DialogueTurnRecord, ...]) -> bool:
    """Validate prompt content without assigning a human-language role label.

    Older exchange files may contain a localized role prefix.  It is treated
    as opaque transport text and never copied into the training projection;
    new courses use the canonical newline-joined content form.
    """
    if not isinstance(value, str) or not value.strip() or not turns:
        return False
    cursor = 0
    for ordinal, turn in enumerate(turns):
        start = value.find(turn.surface, cursor)
        if start < cursor:
            return False
        cursor = start + len(turn.surface)
        if ordinal + 1 < len(turns):
            next_start = value.find(turns[ordinal + 1].surface, cursor)
            if next_start < cursor:
                return False
            cursor = next_start
    return True


def _typed_projection(
        record: dict[str, Any], path: Path, line_number: int,
        source_identity: str | None = None,
        ) -> tuple[CanonicalJsonObject | None, str | None, SourceRef | None]:
    """保留已登记 authored observation；普通对话记录明确返回空。"""
    name = path.name
    if name in {
            "authored_generation_postcheck_seed_v1.jsonl.sample",
            "dialogue_postcheck_bridge_train_v1.course.jsonl.sample",
    }:
        from pure_integer_ai.experiments.ph2_authored_generation_compile import (
            compile_generation_seed,
        )
        from pure_integer_ai.experiments.ph2_authored_generation_schema import (
            AuthoredGenerationSeed,
        )
        try:
            seed = AuthoredGenerationSeed.from_dict(record)
            compiled = compile_generation_seed(seed)
            payload = compiled.observation_payload
            value = payload.to_value()
            raw_source = value.get("source_ref_key")
            source = (SourceRef.from_stable_key(tuple(raw_source))
                      if isinstance(raw_source, list) else None)
            return payload, compiled.payload_kind, source
        except (TypeError, ValueError, KeyError, RuntimeError):
            return None, None, None
    if name == "authored_generation_generalization_seed_v1.jsonl.sample":
        raw = record.get("observation_payload")
        if isinstance(raw, dict):
            source_id = _GENERALIZATION_SOURCE_HASHER.h63(
                (path.as_posix() if source_identity is None else source_identity,
                 line_number,
                 canonical_json_bytes(raw))) or 1
            source = SourceRef(
                214,
                source_id,
                line_number,
                GLOBAL_OWNER_SCOPE,
                VersionBundle(),
            )
            return (CanonicalJsonObject.from_value(raw),
                    "GenerationGeneralizationCandidateV1", source)
    return None, None, None


def _compact_surface_for_record(
        record: dict[str, Any], path: Path,
        token_sidecars: dict[Path, IntegerTokenIndex],
        aggregate_sidecars: dict[Path, IntegerAggregateIndex],
        ) -> tuple[tuple[str, ...], IntegerTokenIndex, int,
                   IntegerAggregateIndex | None, int | None] | None:
    """Resolve an indexed course record without copying its surface text.

    The sidecar path is deliberately relative to the course file.  This keeps
    the exchange format portable and prevents a course record from reaching
    outside its source directory.  A declared sidecar is authoritative: any
    missing, malformed, out-of-range, or hash-drifted reference fails closed.
    """
    file_value = record.get("token_index_file")
    ordinal = record.get("token_index_ordinal")
    aggregate_file_value = record.get("aggregate_index_file")
    aggregate_ordinal = record.get("aggregate_index_ordinal")
    if (file_value is None and ordinal is None
            and aggregate_file_value is None and aggregate_ordinal is None):
        return None
    has_aggregate = (aggregate_file_value is not None
                     or aggregate_ordinal is not None)
    if (not isinstance(file_value, str) or not file_value.strip()
            or (not has_aggregate
                and (type(ordinal) is not int or ordinal < 0))):
        raise ConversationTrainingPackError("token index reference 非法")
    relative = Path(file_value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ConversationTrainingPackError("token index path 越界")
    sidecar = (path.parent / relative).resolve()
    try:
        sidecar.relative_to(path.parent.resolve())
    except ValueError as error:
        raise ConversationTrainingPackError("token index path 越界") from error
    try:
        index = token_sidecars.get(sidecar)
        if index is None:
            index = load_integer_token_index(sidecar)
            token_sidecars[sidecar] = index
    except (OSError, ValueError, TypeError) as error:
        raise ConversationTrainingPackError(
            f"token index sidecar 不可回读: {sidecar}") from error
    declared_hash = record.get("token_index_sha256")
    if declared_hash is not None and (
            not isinstance(declared_hash, str) or declared_hash != index.sha256):
        raise ConversationTrainingPackError("token index hash 漂移")
    aggregate = None
    if aggregate_file_value is not None or aggregate_ordinal is not None:
        if (not isinstance(aggregate_file_value, str)
                or not aggregate_file_value.strip()
                or type(aggregate_ordinal) is not int
                or aggregate_ordinal < 0):
            raise ConversationTrainingPackError("aggregate index reference 非法")
        aggregate_relative = Path(aggregate_file_value)
        if (aggregate_relative.is_absolute() or aggregate_relative.drive
                or ".." in aggregate_relative.parts):
            raise ConversationTrainingPackError("aggregate index path 越界")
        aggregate_path = (path.parent / aggregate_relative).resolve()
        try:
            aggregate_path.relative_to(path.parent.resolve())
        except ValueError as error:
            raise ConversationTrainingPackError("aggregate index path 越界") from error
        try:
            aggregate = aggregate_sidecars.get(aggregate_path)
            if aggregate is None:
                aggregate = load_integer_aggregate_index(aggregate_path)
                aggregate_sidecars[aggregate_path] = aggregate
        except (OSError, ValueError, TypeError) as error:
            raise ConversationTrainingPackError(
                f"aggregate index sidecar 不可回读: {aggregate_path}") from error
        declared_aggregate_hash = record.get("aggregate_index_sha256")
        if (declared_aggregate_hash is not None
                and (not isinstance(declared_aggregate_hash, str)
                     or declared_aggregate_hash != aggregate.sha256)):
            raise ConversationTrainingPackError("aggregate index hash 漂移")
        try:
            surface = aggregate.render(index, aggregate_ordinal)
        except (IndexError, TypeError, ValueError) as error:
            raise ConversationTrainingPackError(
                "aggregate index ordinal 或绑定越界") from error
    else:
        try:
            surface = index.render(ordinal)
        except (IndexError, TypeError) as error:
            raise ConversationTrainingPackError("token index ordinal 越界") from error
    if len(surface) < 2:
        raise ConversationTrainingPackError("token index surface 为空或过短")
    # Aggregate-only records intentionally keep no duplicated surface tuple in
    # the course object; raw_text is reconstructed from the referenced index.
    return (() if has_aggregate else (surface,)), index, (
        None if has_aggregate else ordinal), aggregate, aggregate_ordinal


def load_dialogue_training_pack(paths: Iterable[str | Path], *,
                                max_cases: int | None = None,
                                source_path_identities: Mapping[str | Path, str]
                                | None = None) -> DialogueTrainingPack:
    """读取公开 JSONL 课程并形成确定性 pack；重复 identity 直接失败。"""
    files = tuple(sorted(Path(path).resolve() for path in paths))
    if not files:
        raise ConversationTrainingPackError("未提供公开课程文件")
    cases: list[DialogueTrainingCase] = []
    source_files: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    token_sidecars: dict[Path, IntegerTokenIndex] = {}
    aggregate_sidecars: dict[Path, IntegerAggregateIndex] = {}
    identities = {
        Path(key).resolve(): value
        for key, value in (source_path_identities or {}).items()
    }
    if any(not isinstance(value, str) or not value.strip()
           for value in identities.values()):
        raise ConversationTrainingPackError("source path identity 非法")
    for path in files:
        if not path.is_file():
            raise ConversationTrainingPackError(f"课程文件不存在: {path}")
        payload = path.read_bytes()
        count = 0
        for line_number, raw in enumerate(payload.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ConversationTrainingPackError(
                    f"课程 JSONL 非法: {path.name}:{line_number}") from error
            if not isinstance(record, dict):
                continue
            split = record.get("split", "train")
            if split == "held_out":
                split = "heldout"
            elif split == "course":
                # DLG-RAW-16 的公开 course 是可训练的正向表层课程；它没有
                # evaluator split，因此进入 train，但仍保留原 source/family。
                split = "train"
            if record.get("sample_role") in {"refute", "negative"}:
                split = "negative"
            if record.get("sample_kind") == "NEGATIVE":
                split = "negative"
            elif record.get("sample_kind") == "AMBIGUOUS" and split == "train":
                # 结构歧义样本只作为未见输入观察，不作为正向训练事实。
                split = "heldout"
            if split not in _SPLITS:
                continue
            compact_reference = _compact_surface_for_record(
                record, path, token_sidecars, aggregate_sidecars)
            dialogue_turns = _dialogue_projection(record, path, line_number)
            if compact_reference is None:
                surfaces = (
                    tuple(turn.rendered_surface for turn in dialogue_turns)
                    if dialogue_turns else _surface_for_record(record))
                token_index = None
                token_index_ordinal = None
                aggregate_index = None
                aggregate_index_ordinal = None
            else:
                (surfaces, token_index, token_index_ordinal,
                 aggregate_index, aggregate_index_ordinal) = compact_reference
            indexed_surface = token_index is not None
            if not surfaces and aggregate_index is None:
                continue
            source_identity = identities.get(path)
            case_id = _record_id(record, path, line_number, source_identity)
            if case_id in seen:
                raise ConversationTrainingPackError(f"重复 case identity: {case_id}")
            seen.add(case_id)
            family = record.get("family") or record.get("template_family") or path.stem
            if not isinstance(family, str) or not family:
                family = path.stem
            causal_pairs: tuple[tuple[int, int], ...] = ()
            if record.get("relation_family") == "CAUSES":
                endpoints = record.get("endpoints")
                if isinstance(endpoints, list) and len(endpoints) >= 2:
                    spans = []
                    for endpoint in endpoints[:2]:
                        if isinstance(endpoint, dict):
                            start = endpoint.get("start")
                            if type(start) is int and start >= 0:
                                spans.append(start)
                    if len(spans) == 2 and spans[0] != spans[1]:
                        causal_pairs = ((spans[0], spans[1]),)
            typed_payload, payload_kind, typed_source_ref = _typed_projection(
                record, path, line_number, source_identity)
            public_source_ref = _public_source_ref(record, path, line_number)
            if (typed_source_ref is not None and public_source_ref is not None
                    and typed_source_ref != public_source_ref):
                raise ConversationTrainingPackError(
                    f"typed/public SourceRef 冲突: {path.name}:{line_number}")
            source_ref = typed_source_ref or public_source_ref
            expected_state = record.get("expected_state")
            if expected_state is not None and not isinstance(expected_state, str):
                raise ConversationTrainingPackError(
                    f"expected_state 非法: {path.name}:{line_number}")
            expected_payload_value = record.get("expected_payload")
            if (expected_payload_value is not None
                    and not isinstance(expected_payload_value, dict)):
                raise ConversationTrainingPackError(
                    f"expected_payload 非法: {path.name}:{line_number}")
            expected_payload = (
                None if expected_payload_value is None
                else CanonicalJsonObject.from_value(expected_payload_value))
            cases.append(DialogueTrainingCase(
                case_id, split, family, path.as_posix(), line_number, surfaces,
                causal_pairs,
                _canonical_case_record(
                    case_id, split, family,
                    path.as_posix() if source_identity is None else source_identity,
                    line_number,
                    surfaces, causal_pairs, typed_payload, payload_kind,
                    source_ref, expected_state, expected_payload,
                    indexed_surface, token_index, token_index_ordinal,
                    aggregate_index, aggregate_index_ordinal,
                    dialogue_turns),
                typed_payload=typed_payload,
                payload_kind=payload_kind,
                source_ref=source_ref,
                expected_state=expected_state,
                expected_payload=expected_payload,
                indexed_surface=indexed_surface,
                token_index=token_index,
                token_index_ordinal=token_index_ordinal,
                aggregate_index=aggregate_index,
                aggregate_index_ordinal=aggregate_index_ordinal,
                dialogue_turns=dialogue_turns,
            ))
            count += 1
            if max_cases is not None and len(cases) >= max_cases:
                break
        source_files.append((path.as_posix(), _sha256(payload), count))
        if max_cases is not None and len(cases) >= max_cases:
            break
    ordered = tuple(sorted(cases, key=lambda item: item.case_id))
    if not ordered:
        raise ConversationTrainingPackError("公开课程没有可消费记录")
    digest_payload = b"".join(
        _encode_integer_record(item.canonical_record())
        for item in ordered
    )
    return DialogueTrainingPack(
        ordered, tuple(source_files), _sha256(digest_payload))


__all__ = [
    "CONVERSATION_TRAINING_PACK_PROTOCOL_V1",
    "ConversationTrainingPackError",
    "DialogueTrainingCase",
    "DialogueTrainingPack",
    "load_dialogue_training_pack",
]
