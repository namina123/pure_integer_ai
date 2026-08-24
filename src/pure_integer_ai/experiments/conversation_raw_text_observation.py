"""T1-G0：原始文本 observation 到显式 scalar/span units 的可迁移编译。

本切片只建立物理观察与显式结构标注之间的边界：它保存 raw UTF-8 bytes、Unicode
scalar、字节/标量 span 和上游给出的 unit role，但不猜词义、命题、关系或来源真值。
所有核心值都可投影为有限整数 record；JSONL 仅是公开课程的边界载体。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.experiments.conversation_raw_intake import (
    decode_utf8_v1,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


RAW_TEXT_OBSERVATION_PROTOCOL_V1 = 1
RAW_TEXT_OBSERVATION_RECORD_KIND = "DLG_RAW_TEXT_OBSERVATION_V1"
RAW_TEXT_OBSERVATION_LICENSE = "CC0-1.0"
RAW_TEXT_OBSERVATION_SPLITS = frozenset({"train", "heldout", "negative"})
_OBSERVATION_FIELDS = frozenset({
    "context_id", "family_id", "license_id", "observation_id",
    "protocol_version", "raw_u8", "record_kind", "scalars", "source_id",
    "source_namespace", "split", "units",
})
_UNIT_FIELDS = frozenset({
    "end_byte", "end_scalar", "role", "start_byte", "start_scalar", "unit_id",
})


class RawTextObservationError(ValueError):
    """raw observation 或显式 span unit 越过可迁移边界。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawTextObservationError(f"{where} 必须是无首尾空白的非空字符串")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise RawTextObservationError(f"{where} 含非 Unicode scalar")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise RawTextObservationError(f"{where} 必须是非负严格整数")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise RawTextObservationError(f"{where} 必须是正严格整数")
    return value


def _u8(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise RawTextObservationError(f"{where} 必须是 byte 序列")
    result = tuple(value)
    if any(type(item) is not int or item < 0 or item > 255 for item in result):
        raise RawTextObservationError(f"{where} 必须只含 0..255 严格整数")
    return result


def _scalars(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise RawTextObservationError(f"{where} 必须是 scalar 序列")
    result = tuple(value)
    if any(type(item) is not int or item < 0 or item > 0x10FFFF
           or 0xD800 <= item <= 0xDFFF for item in result):
        raise RawTextObservationError(f"{where} 含非法 Unicode scalar")
    return result


def _pack(result: list[int], values: tuple[int, ...]) -> None:
    result.extend((len(values), *values))


def _pack_text(result: list[int], value: str, where: str) -> None:
    scalars = tuple(ord(item) for item in _text(value, where))
    _pack(result, scalars)


@dataclass(frozen=True, slots=True)
class RawTextSpanUnit:
    """上游显式提供的结构槽；role 不由本模块从文本猜测。"""

    unit_id: str
    role: str
    start_scalar: int
    end_scalar: int
    start_byte: int
    end_byte: int

    def __post_init__(self) -> None:
        _text(self.unit_id, "unit.unit_id")
        _text(self.role, "unit.role")
        _nonnegative(self.start_scalar, "unit.start_scalar")
        _positive(self.end_scalar, "unit.end_scalar")
        _nonnegative(self.start_byte, "unit.start_byte")
        _positive(self.end_byte, "unit.end_byte")
        if self.end_scalar <= self.start_scalar:
            raise RawTextObservationError("unit scalar span 必须非空")
        if self.end_byte <= self.start_byte:
            raise RawTextObservationError("unit byte span 必须非空")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_TEXT_OBSERVATION_PROTOCOL_V1]
        _pack_text(result, self.unit_id, "unit.unit_id")
        _pack_text(result, self.role, "unit.role")
        result.extend((self.start_scalar, self.end_scalar,
                       self.start_byte, self.end_byte))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawTextObservation:
    """一条 raw observation 与显式结构 units 的不可变值记录。"""

    observation_id: str
    source_id: str
    context_id: str
    family_id: str
    source_namespace: str
    split: str
    raw_bytes: tuple[int, ...]
    scalars: tuple[int, ...]
    units: tuple[RawTextSpanUnit, ...]

    def __post_init__(self) -> None:
        for name in ("observation_id", "source_id", "context_id", "family_id",
                     "source_namespace"):
            _text(getattr(self, name), f"observation.{name}")
        if self.split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawTextObservationError("observation.split 未注册")
        raw = _u8(self.raw_bytes, "observation.raw_bytes")
        scalars = _scalars(self.scalars, "observation.scalars")
        if not scalars:
            raise RawTextObservationError("observation.scalars 不能为空")
        if tuple(decode_utf8_v1(raw) or ()) != scalars:
            raise RawTextObservationError("raw bytes/scalars UTF-8 回读漂移")
        if tuple(encode_utf8_v1(scalars)) != raw:
            raise RawTextObservationError("scalars/raw bytes 编码漂移")
        if not isinstance(self.units, tuple) or not self.units:
            raise RawTextObservationError("observation.units 不能为空")
        if any(type(item) is not RawTextSpanUnit for item in self.units):
            raise TypeError("observation.units 类型错误")
        if len({item.unit_id for item in self.units}) != len(self.units):
            raise RawTextObservationError("unit_id 不得重复")
        previous_scalar_end = 0
        previous_byte_end = 0
        byte_offsets = [0]
        for scalar in scalars:
            byte_offsets.append(byte_offsets[-1] + len(encode_utf8_v1((scalar,))))
        for unit in self.units:
            if unit.end_scalar > len(scalars):
                raise RawTextObservationError("unit scalar span 越界")
            if (unit.start_scalar < previous_scalar_end
                    or unit.start_byte < previous_byte_end):
                raise RawTextObservationError("units 必须按非重叠物理顺序排列")
            if (unit.start_byte, unit.end_byte) != (
                    byte_offsets[unit.start_scalar], byte_offsets[unit.end_scalar]):
                raise RawTextObservationError("unit scalar/byte span 不一致")
            previous_scalar_end = unit.end_scalar
            previous_byte_end = unit.end_byte
        object.__setattr__(self, "raw_bytes", raw)
        object.__setattr__(self, "scalars", scalars)

    def unit_scalars(self, unit: RawTextSpanUnit) -> tuple[int, ...]:
        if unit not in self.units:
            raise RawTextObservationError("unit 不属于 observation")
        return self.scalars[unit.start_scalar:unit.end_scalar]

    def unit_bytes(self, unit: RawTextSpanUnit) -> tuple[int, ...]:
        if unit not in self.units:
            raise RawTextObservationError("unit 不属于 observation")
        return self.raw_bytes[unit.start_byte:unit.end_byte]

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_TEXT_OBSERVATION_PROTOCOL_V1]
        for value in (self.observation_id, self.source_id, self.context_id,
                      self.family_id, self.source_namespace, self.split):
            _pack_text(result, value, "observation.identity")
        _pack(result, self.raw_bytes)
        _pack(result, self.scalars)
        for unit in self.units:
            record = unit.canonical_record()
            _pack(result, record)
        return tuple(result)


def compile_raw_text_observation(
        raw_bytes: tuple[int, ...] | list[int],
        *,
        observation_id: str,
        source_id: str,
        context_id: str,
        family_id: str,
        source_namespace: str,
        split: str,
        units: Iterable[RawTextSpanUnit],
        ) -> RawTextObservation:
    """只做 UTF-8 物理编译与显式 span 验证，不推断语言结构。"""
    raw = _u8(raw_bytes, "raw_bytes")
    scalars = decode_utf8_v1(raw)
    if scalars is None or not scalars:
        raise RawTextObservationError("raw_bytes 不是非空严格 UTF-8")
    return RawTextObservation(
        observation_id, source_id, context_id, family_id, source_namespace,
        split, raw, scalars, tuple(units),
    )


def raw_text_observation_to_json_object(
        observation: RawTextObservation,
        ) -> dict[str, Any]:
    """把 observation 投影为无浮点、可跨语言回读的公开 JSON object。"""
    if not isinstance(observation, RawTextObservation):
        raise TypeError("需要 RawTextObservation")
    return {
        "context_id": observation.context_id,
        "family_id": observation.family_id,
        "license_id": RAW_TEXT_OBSERVATION_LICENSE,
        "observation_id": observation.observation_id,
        "protocol_version": RAW_TEXT_OBSERVATION_PROTOCOL_V1,
        "raw_u8": list(observation.raw_bytes),
        "record_kind": RAW_TEXT_OBSERVATION_RECORD_KIND,
        "scalars": list(observation.scalars),
        "source_id": observation.source_id,
        "source_namespace": observation.source_namespace,
        "split": observation.split,
        "units": [
            {
                "end_byte": unit.end_byte,
                "end_scalar": unit.end_scalar,
                "role": unit.role,
                "start_byte": unit.start_byte,
                "start_scalar": unit.start_scalar,
                "unit_id": unit.unit_id,
            }
            for unit in observation.units
        ],
    }


def compile_raw_text_observation_json(
        observation: RawTextObservation,
        ) -> bytes:
    """返回单条以换行结束的规范公开 JSONL observation。"""
    return canonical_json_line(raw_text_observation_to_json_object(observation))


def parse_raw_text_observation_record(value: Any) -> RawTextObservation:
    """严格从公开 object 恢复 observation，不从 raw 表面猜测 unit role。"""
    if not isinstance(value, dict) or set(value) != _OBSERVATION_FIELDS:
        raise RawTextObservationError("raw observation 字段集合漂移")
    if value["record_kind"] != RAW_TEXT_OBSERVATION_RECORD_KIND:
        raise RawTextObservationError("raw observation record kind 不匹配")
    if value["protocol_version"] != RAW_TEXT_OBSERVATION_PROTOCOL_V1:
        raise RawTextObservationError("raw observation protocol version 不匹配")
    if value["license_id"] != RAW_TEXT_OBSERVATION_LICENSE:
        raise RawTextObservationError("raw observation 许可不匹配")
    raw = _u8(value["raw_u8"], "record.raw_u8")
    scalars = _scalars(value["scalars"], "record.scalars")
    units_raw = value["units"]
    if not isinstance(units_raw, list) or not units_raw:
        raise RawTextObservationError("record.units 必须是非空列表")
    units = []
    for index, item in enumerate(units_raw):
        if not isinstance(item, dict) or set(item) != _UNIT_FIELDS:
            raise RawTextObservationError(f"record.units[{index}] 字段集合漂移")
        units.append(RawTextSpanUnit(
            _text(item["unit_id"], f"record.units[{index}].unit_id"),
            _text(item["role"], f"record.units[{index}].role"),
            _nonnegative(item["start_scalar"], f"record.units[{index}].start_scalar"),
            _positive(item["end_scalar"], f"record.units[{index}].end_scalar"),
            _nonnegative(item["start_byte"], f"record.units[{index}].start_byte"),
            _positive(item["end_byte"], f"record.units[{index}].end_byte"),
        ))
    observation = RawTextObservation(
        _text(value["observation_id"], "record.observation_id"),
        _text(value["source_id"], "record.source_id"),
        _text(value["context_id"], "record.context_id"),
        _text(value["family_id"], "record.family_id"),
        _text(value["source_namespace"], "record.source_namespace"),
        _text(value["split"], "record.split"), raw, scalars, tuple(units),
    )
    if raw_text_observation_to_json_object(observation) != value:
        raise RawTextObservationError("raw observation 规范回读漂移")
    return observation


def load_raw_text_observation_jsonl(
        payload: bytes,
        *,
        expected_split: str | None = None,
        ) -> tuple[RawTextObservation, ...]:
    """严格回读公开 JSONL，并可要求整个 pack 只属于一个 split。"""
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise RawTextObservationError("raw observation JSONL 必须以换行结束")
    rows = payload.splitlines(keepends=True)
    if not rows:
        raise RawTextObservationError("raw observation JSONL 不能为空")
    observations = []
    for index, line in enumerate(rows):
        if line == b"\n" or not line.endswith(b"\n"):
            raise RawTextObservationError(f"raw observation JSONL 第 {index} 行无效")
        value = parse_canonical_json_bytes(line[:-1], require_object=True)
        if canonical_json_line(value) != line:
            raise RawTextObservationError(f"raw observation JSONL 第 {index} 行非规范")
        observations.append(parse_raw_text_observation_record(value))
    if len({item.observation_id for item in observations}) != len(observations):
        raise RawTextObservationError("observation_id 不得重复")
    if expected_split is not None:
        if expected_split not in RAW_TEXT_OBSERVATION_SPLITS:
            raise RawTextObservationError("expected_split 未注册")
        if any(item.split != expected_split for item in observations):
            raise RawTextObservationError("JSONL 混入了不允许的 split")
    return tuple(observations)


__all__ = [
    "RAW_TEXT_OBSERVATION_LICENSE", "RAW_TEXT_OBSERVATION_PROTOCOL_V1",
    "RAW_TEXT_OBSERVATION_RECORD_KIND", "RawTextObservation",
    "RawTextObservationError", "RawTextSpanUnit", "compile_raw_text_observation",
    "compile_raw_text_observation_json", "load_raw_text_observation_jsonl",
    "parse_raw_text_observation_record", "raw_text_observation_to_json_object",
]
