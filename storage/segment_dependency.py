"""run、location 和后续增量 segment 共用的完整依赖身份。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 segment 依赖的非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """将可变长整数键按长度分帧写入稳定键。"""
    result.extend((len(value), *value))


@dataclass(frozen=True, order=True)
class SegmentDependency:
    """segment 所需逻辑对象的完整版本和内容校验身份。"""

    descriptor_key: tuple[int, ...]
    version_key: tuple[int, ...]
    checksum_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝空键、非整数键和只保存裸 hash 的依赖。"""
        _key(self.descriptor_key, label="segment dependency descriptor_key")
        _key(self.version_key, label="segment dependency version_key")
        _key(self.checksum_key, label="segment dependency checksum_key")

    @property
    def dependency_key(self) -> tuple[int, ...]:
        """返回 run manifest 使用的通用依赖键别名。"""
        return self.descriptor_key

    def stable_key(self) -> tuple[int, ...]:
        """返回描述、版本和校验的完整稳定键。"""
        result: list[int] = []
        _pack(result, self.descriptor_key)
        _pack(result, self.version_key)
        _pack(result, self.checksum_key)
        return tuple(result)

    def to_payload(self) -> dict[str, list[int]]:
        """转为 run manifest 可确定序列化的依赖字段。"""
        return {
            "dependency_key": list(self.descriptor_key),
            "version_key": list(self.version_key),
            "checksum_key": list(self.checksum_key),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "SegmentDependency":
        """从 run manifest 字段严格恢复依赖。"""
        if not isinstance(payload, dict):
            raise ValueError("segment dependency 必须是 object")
        try:
            return cls(
                tuple(payload["dependency_key"]),
                tuple(payload["version_key"]),
                tuple(payload["checksum_key"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("segment dependency 字段非法") from exc


def canonical_dependencies(
        dependencies: tuple[SegmentDependency, ...],
        ) -> tuple[SegmentDependency, ...]:
    """规范化依赖顺序并拒绝同描述身份重复。"""
    if (not isinstance(dependencies, tuple)
            or any(not isinstance(item, SegmentDependency)
                   for item in dependencies)):
        raise TypeError("segment dependencies 必须是 SegmentDependency tuple")
    normalized = tuple(sorted(dependencies))
    if len({item.descriptor_key for item in normalized}) != len(normalized):
        raise ValueError("segment dependencies 不得重复 descriptor")
    return normalized


__all__ = ["SegmentDependency", "canonical_dependencies"]
