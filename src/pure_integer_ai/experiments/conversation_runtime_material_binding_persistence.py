"""Runtime response binding 的纯整数封存与严格回读。

该模块只封存显式问题路由和 qualification，不封存自然语言推断结果。回读时必须
由调用方提供同一批真实 ``RuntimeMaterialLanguageObservation``，再验证 observation、
Runtime item、relation candidate 和 qualification identity；任何缺失或漂移均失败关闭。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RAW_PROPOSITION_CONSUMER_PROTOCOL_V1,
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_runtime_material_response import (
    RuntimeMaterialResponseError,
    RuntimeMaterialResponseProvider,
    RuntimeMaterialResponseSpec,
    build_runtime_material_response_provider,
)
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunRoot,
    ensure_normal_relative_directory,
    open_existing_run_root,
    open_plain_binary,
    write_exclusive_bytes,
)
from pure_integer_ai.storage.source_record import SourceRecordRepository


RUNTIME_MATERIAL_BINDING_LEDGER_VERSION = 1
RUNTIME_MATERIAL_BINDING_LEDGER_RELATIVE = (
    "runtime_material_response/bindings.int")


class RuntimeMaterialBindingPersistenceError(RuntimeMaterialResponseError):
    """Runtime response binding ledger 缺失、损坏或 identity 漂移。"""


def _strict_key(value: tuple[int, ...], *, label: str,
                empty: bool = False) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or (not empty and not value)
            or any(type(item) is not int for item in value)):
        raise RuntimeMaterialBindingPersistenceError(
            f"{label} 必须是整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str,
          empty: bool = True) -> None:
    checked = _strict_key(value, label=label, empty=empty)
    result.extend((len(checked), *checked))


def _text(value: str, *, label: str) -> tuple[int, ...]:
    if type(value) is not str or not value.strip():
        raise RuntimeMaterialBindingPersistenceError(
            f"{label} 必须是非空文本")
    return tuple(ord(item) for item in value)


def _optional_text(value: str | None, *, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    return _text(value, label=label)


def _decode_key(record: tuple[int, ...], cursor: int, *, label: str,
                empty: bool = True) -> tuple[tuple[int, ...], int]:
    if cursor >= len(record):
        raise RuntimeMaterialBindingPersistenceError(
            f"{label} 缺少长度")
    length = record[cursor]
    cursor += 1
    if type(length) is not int or length < 0:
        raise RuntimeMaterialBindingPersistenceError(f"{label} 长度非法")
    end = cursor + length
    if end > len(record):
        raise RuntimeMaterialBindingPersistenceError(f"{label} 被截断")
    value = tuple(record[cursor:end])
    if not empty and not value:
        raise RuntimeMaterialBindingPersistenceError(f"{label} 不得为空")
    if any(type(item) is not int for item in value):
        raise RuntimeMaterialBindingPersistenceError(f"{label} 不是整数")
    return value, end


def _decode_text(value: tuple[int, ...], *, label: str,
                 optional: bool = False) -> str | None:
    if optional and not value:
        return None
    if not value or any(item < 0 or item > 0x10FFFF
                        or 0xD800 <= item <= 0xDFFF for item in value):
        raise RuntimeMaterialBindingPersistenceError(f"{label} scalar 非法")
    try:
        result = "".join(chr(item) for item in value)
    except (TypeError, ValueError) as error:
        raise RuntimeMaterialBindingPersistenceError(
            f"{label} 无法恢复") from error
    if not result.strip():
        raise RuntimeMaterialBindingPersistenceError(f"{label} 为空")
    return result


def _qualification_record(qualification: RawPropositionQualification) -> tuple[int, ...]:
    return qualification.canonical_record()


def _decode_qualification(record: tuple[int, ...]) -> RawPropositionQualification:
    cursor = 0
    if not record or record[cursor] != RAW_PROPOSITION_CONSUMER_PROTOCOL_V1:
        raise RuntimeMaterialBindingPersistenceError(
            "qualification protocol 不匹配")
    cursor += 1
    fields: list[str] = []
    for index in range(11):
        value, cursor = _decode_key(
            record, cursor, label=f"qualification.field[{index}]", empty=False)
        text = _decode_text(value, label=f"qualification.field[{index}]")
        if text is None:
            raise AssertionError("non-optional qualification field decoded None")
        fields.append(text)
    if cursor >= len(record):
        raise RuntimeMaterialBindingPersistenceError(
            "qualification evidence count 缺失")
    count = record[cursor]
    cursor += 1
    if type(count) is not int or count <= 0:
        raise RuntimeMaterialBindingPersistenceError(
            "qualification evidence count 非法")
    evidence: list[str] = []
    for index in range(count):
        value, cursor = _decode_key(
            record, cursor, label=f"qualification.evidence[{index}]", empty=False)
        text = _decode_text(value, label=f"qualification.evidence[{index}]")
        if text is None:
            raise AssertionError("non-optional evidence decoded None")
        evidence.append(text)
    if cursor != len(record):
        raise RuntimeMaterialBindingPersistenceError(
            "qualification record 含尾随整数")
    qualification = RawPropositionQualification(
        fields[0], fields[1], fields[2], fields[3], fields[4], fields[5],
        fields[6], fields[7], fields[8], fields[9], tuple(evidence), fields[10],
    )
    if qualification.canonical_record() != record:
        raise RuntimeMaterialBindingPersistenceError(
            "qualification canonical record 漂移")
    return qualification


@dataclass(frozen=True, slots=True)
class RuntimeMaterialBindingRecord:
    """一个可跨进程回放的 provider binding 描述。"""

    question: str
    memory_item_key: tuple[int, ...]
    relation_index: int
    qualification: RawPropositionQualification
    source_title: str | None
    source_url: str | None


def _record_sort_key(item: RuntimeMaterialBindingRecord) -> tuple[object, ...]:
    return (
        tuple(ord(value) for value in item.question),
        item.memory_item_key,
        item.relation_index,
        item.qualification.canonical_record(),
        tuple(ord(value) for value in (item.source_title or "")),
        tuple(ord(value) for value in (item.source_url or "")),
    )


def _binding_record(candidate) -> RuntimeMaterialBindingRecord:
    binding = candidate.binding
    gate = binding.qualification_gate
    return RuntimeMaterialBindingRecord(
        candidate.question,
        binding.memory_item_key,
        candidate.relation_index,
        gate.qualification,
        binding.source_title,
        binding.source_url,
    )


def _encode_record(item: RuntimeMaterialBindingRecord) -> tuple[int, ...]:
    if not isinstance(item, RuntimeMaterialBindingRecord):
        raise TypeError("binding record 类型错误")
    if type(item.relation_index) is not int or item.relation_index < 0:
        raise RuntimeMaterialBindingPersistenceError("relation_index 非法")
    result: list[int] = []
    _pack(result, _text(item.question, label="question"), label="question",
          empty=False)
    _pack(result, _strict_key(item.memory_item_key, label="memory item",
                              empty=False), label="memory item", empty=False)
    result.append(item.relation_index)
    _pack(result, _optional_text(item.source_title, label="source title"),
          label="source title")
    _pack(result, _optional_text(item.source_url, label="source url"),
          label="source url")
    _pack(result, _qualification_record(item.qualification),
          label="qualification", empty=False)
    return tuple(result)


def _decode_record(record: tuple[int, ...]) -> RuntimeMaterialBindingRecord:
    cursor = 0
    question_key, cursor = _decode_key(record, cursor, label="question", empty=False)
    memory_key, cursor = _decode_key(record, cursor, label="memory item", empty=False)
    if cursor >= len(record):
        raise RuntimeMaterialBindingPersistenceError("relation_index 缺失")
    relation_index = record[cursor]
    cursor += 1
    if type(relation_index) is not int or relation_index < 0:
        raise RuntimeMaterialBindingPersistenceError("relation_index 非法")
    title_key, cursor = _decode_key(record, cursor, label="source title")
    url_key, cursor = _decode_key(record, cursor, label="source url")
    qualification_key, cursor = _decode_key(
        record, cursor, label="qualification", empty=False)
    if cursor != len(record):
        raise RuntimeMaterialBindingPersistenceError("binding record 含尾随整数")
    question = _decode_text(question_key, label="question")
    title = _decode_text(title_key, label="source title", optional=True)
    url = _decode_text(url_key, label="source url", optional=True)
    if question is None:
        raise AssertionError("question decoded None")
    return RuntimeMaterialBindingRecord(
        question, memory_key, relation_index,
        _decode_qualification(qualification_key), title, url,
    )


def encode_runtime_material_response_bindings(
        provider: RuntimeMaterialResponseProvider,
        ) -> bytes:
    """编码 provider binding ledger 为规范整数 bytes。"""
    if not isinstance(provider, RuntimeMaterialResponseProvider):
        raise TypeError("provider 类型错误")
    records = tuple(_binding_record(item) for item in provider.candidates)
    result: list[int] = [RUNTIME_MATERIAL_BINDING_LEDGER_VERSION, len(records)]
    for item in records:
        _pack(result, _encode_record(item), label="binding record", empty=False)
    return encode_integer_tuple(tuple(result))


def decode_runtime_material_response_bindings(
        payload: bytes,
        ) -> tuple[RuntimeMaterialBindingRecord, ...]:
    """严格回读规范整数 binding ledger。"""
    try:
        record = decode_integer_tuple(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeMaterialBindingPersistenceError(
            "binding ledger integer bytes 损坏") from error
    if len(record) < 2 or record[0] != RUNTIME_MATERIAL_BINDING_LEDGER_VERSION:
        raise RuntimeMaterialBindingPersistenceError("binding ledger version 不匹配")
    count = record[1]
    if type(count) is not int or count <= 0:
        raise RuntimeMaterialBindingPersistenceError("binding ledger count 非法")
    cursor = 2
    result: list[RuntimeMaterialBindingRecord] = []
    for index in range(count):
        item_record, cursor = _decode_key(
            record, cursor, label=f"binding[{index}]", empty=False)
        result.append(_decode_record(item_record))
    if cursor != len(record):
        raise RuntimeMaterialBindingPersistenceError(
            "binding ledger 含尾随整数")
    identities = tuple((item.question, item.memory_item_key,
                        item.qualification.qualification_id)
                       for item in result)
    if len(set(identities)) != len(identities):
        raise RuntimeMaterialBindingPersistenceError("binding ledger 含重复 binding")
    if tuple(result) != tuple(sorted(result, key=_record_sort_key)):
        raise RuntimeMaterialBindingPersistenceError(
            "binding ledger 未按规范顺序排列")
    return tuple(result)


def persist_runtime_material_response_bindings(
        root: KRunRoot,
        provider: RuntimeMaterialResponseProvider,
        *,
        relative_path: str | Path = RUNTIME_MATERIAL_BINDING_LEDGER_RELATIVE,
        ) -> Path:
    """在 K run 内排他发布 provider binding ledger。"""
    if not isinstance(root, KRunRoot):
        raise TypeError("root 必须是 KRunRoot")
    relative = Path(relative_path)
    if not relative.parts:
        raise RuntimeMaterialBindingPersistenceError("relative_path 不能为空")
    if relative.parent.parts:
        ensure_normal_relative_directory(
            root, relative.parent, label="runtime response binding directory")
    return write_exclusive_bytes(
        root, relative, encode_runtime_material_response_bindings(provider),
        label="runtime response binding ledger")


def load_runtime_material_response_provider(
        root_path: str | Path,
        *,
        source_records: SourceRecordRepository,
        observations: tuple[object, ...],
        relative_path: str | Path = RUNTIME_MATERIAL_BINDING_LEDGER_RELATIVE,
        require_k_drive: bool = True,
        ) -> RuntimeMaterialResponseProvider:
    """从整数 ledger 和真实 observations 严格重建 provider。"""
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    if not isinstance(observations, tuple) or not observations:
        raise RuntimeMaterialBindingPersistenceError(
            "observations 必须是非空 tuple")
    root = open_existing_run_root(
        root_path, require_k_drive=require_k_drive,
        label="runtime response binding root")
    relative = Path(relative_path)
    try:
        with open_plain_binary(root, relative,
                               label="runtime response binding ledger") as stream:
            payload = stream.read()
    except OSError as error:
        raise RuntimeMaterialBindingPersistenceError(
            "binding ledger 读取失败") from error
    records = decode_runtime_material_response_bindings(payload)
    by_observation = {}
    for observation in observations:
        observation_id = getattr(
            getattr(observation, "raw_observation", None), "observation_id", None)
        if not isinstance(observation_id, str) or not observation_id:
            raise RuntimeMaterialBindingPersistenceError(
                "observation 缺少稳定 observation_id")
        if observation_id in by_observation:
            raise RuntimeMaterialBindingPersistenceError(
                "observations 含重复 observation_id")
        by_observation[observation_id] = observation
    specs: list[RuntimeMaterialResponseSpec] = []
    for item in records:
        observation = by_observation.get(item.qualification.observation_id)
        if observation is None:
            raise RuntimeMaterialBindingPersistenceError(
                "binding 缺少对应真实 observation")
        memory_item_key = observation.ingest.event.memory_item_key
        if memory_item_key != item.memory_item_key:
            raise RuntimeMaterialBindingPersistenceError(
                "binding Runtime item identity 漂移")
        specs.append(RuntimeMaterialResponseSpec(
            observation, item.qualification, item.question,
            item.relation_index, item.source_title, item.source_url,
        ))
    try:
        return build_runtime_material_response_provider(
            tuple(specs), source_records=source_records)
    except (RuntimeMaterialResponseError, TypeError, ValueError) as error:
        raise RuntimeMaterialBindingPersistenceError(
            "binding ledger 与真实 observation 无法闭合") from error


__all__ = [
    "RUNTIME_MATERIAL_BINDING_LEDGER_RELATIVE",
    "RUNTIME_MATERIAL_BINDING_LEDGER_VERSION",
    "RuntimeMaterialBindingPersistenceError",
    "RuntimeMaterialBindingRecord",
    "decode_runtime_material_response_bindings",
    "encode_runtime_material_response_bindings",
    "load_runtime_material_response_provider",
    "persist_runtime_material_response_bindings",
]
