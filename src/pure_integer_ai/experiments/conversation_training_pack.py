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
from typing import Any, Iterable

from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    LANG_ZH,
    MODALITY_LANGUAGE,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
    SOURCE_BARE_TEXT,
)


CONVERSATION_TRAINING_PACK_PROTOCOL_V1 = 2
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


def _record_id(record: dict[str, Any], path: Path, line_number: int) -> str:
    for key in ("seed_id", "episode_id", "item_id", "frame_key"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return f"{path.as_posix()}::{value}"
    return f"{path.as_posix()}::line-{line_number}"


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

    def __post_init__(self) -> None:
        if not self.case_id or self.split not in _SPLITS or not self.surfaces:
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

    @property
    def raw_text(self) -> str:
        """返回供 observe 消费的完整表层，保留问句、上下文和回答顺序。"""
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

    def training_items(self, *, split: str = "train",
                       causal_only: bool = False) -> list[CollectedItem]:
        """把指定 train/heldout split 投影为 formal_train 可直接消费的语言项。"""
        if split not in {"train", "heldout"}:
            raise ConversationTrainingPackError("训练投影只允许 train/heldout")
        return self.items_for_split(split=split, causal_only=causal_only)

    def evaluation_items(self, *, split: str | None = None,
                         causal_only: bool = False) -> list[CollectedItem]:
        """投影公开评测项，允许显式 negative/ambiguous 进入只读 V-00。"""
        if split is not None and split not in _SPLITS:
            raise ConversationTrainingPackError("评测投影 split 非法")
        return self.items_for_split(split=split, causal_only=causal_only)

    def items_for_split(self, *, split: str | None,
                        causal_only: bool = False) -> list[CollectedItem]:
        """按公开 split 建立统一 CollectedItem；不改变课程标签或来源。"""
        result = []
        for case in self.cases:
            if split is not None and case.split != split:
                continue
            if causal_only and not case.causal_pairs:
                continue
            text = case.raw_text
            tokens = list(text)
            source_ref = case.source_ref
            source = SOURCE_BARE_TEXT
            if source_ref is not None:
                # Authored typed records carry authoritative provenance.  The
                # round runtime switches WorkMemory sessions at version
                # boundaries, so this source must remain intact.
                source = source_ref.source_kind
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
    return tuple(values)


def _typed_projection(
        record: dict[str, Any], path: Path, line_number: int,
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
                (path.as_posix(), line_number,
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


def load_dialogue_training_pack(paths: Iterable[str | Path], *,
                                max_cases: int | None = None) -> DialogueTrainingPack:
    """读取公开 JSONL 课程并形成确定性 pack；重复 identity 直接失败。"""
    files = tuple(sorted(Path(path).resolve() for path in paths))
    if not files:
        raise ConversationTrainingPackError("未提供公开课程文件")
    cases: list[DialogueTrainingCase] = []
    source_files: list[tuple[str, str, int]] = []
    seen: set[str] = set()
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
            surfaces = _surface_for_record(record)
            if not surfaces:
                continue
            case_id = _record_id(record, path, line_number)
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
            typed_payload, payload_kind, source_ref = _typed_projection(
                record, path, line_number)
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
                    case_id, split, family, path.as_posix(), line_number,
                    surfaces, causal_pairs, typed_payload, payload_kind,
                    source_ref, expected_state, expected_payload),
                typed_payload=typed_payload,
                payload_kind=payload_kind,
                source_ref=source_ref,
                expected_state=expected_state,
                expected_payload=expected_payload,
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
