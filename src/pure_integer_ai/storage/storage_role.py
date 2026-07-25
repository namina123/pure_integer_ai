"""存储对象的角色、访问方式和可重建依赖描述。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


STORAGE_ROLE_AUTHORITATIVE = 1
STORAGE_ROLE_REBUILDABLE = 2
STORAGE_ROLE_EPHEMERAL = 3

STORAGE_ACCESS_APPEND_ONLY = 1
STORAGE_ACCESS_MUTABLE = 2
STORAGE_ACCESS_INDEXED_READ = 3
STORAGE_ACCESS_REBUILD = 4


def _key(value: tuple[int, ...], *, label: str, empty: bool = False) -> tuple[int, ...]:
    """核验开放存储身份键为严格整数 tuple。"""
    if not isinstance(value, tuple) or (not empty and not value):
        raise ValueError(f"{label} 必须是非空整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise ValueError(f"{label} 必须使用严格整数")
    return value


@dataclass(frozen=True, order=True)
class StorageRoleDescriptor:
    """一个逻辑存储对象的角色、访问协议和重建依赖。"""

    descriptor_key: tuple[int, ...]
    role: int
    access_modes: tuple[int, ...]
    dependency_keys: tuple[tuple[int, ...], ...] = ()
    rebuild_protocol_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验角色描述、访问方式唯一性和可重建依赖闭合条件。"""
        _key(self.descriptor_key, label="storage descriptor_key")
        assert_int(self.role, _where="storage role")
        if type(self.role) is not int or self.role <= 0:
            raise ValueError("storage role 必须是正严格整数")
        _key(self.access_modes, label="storage access_modes")
        if len(set(self.access_modes)) != len(self.access_modes):
            raise ValueError("storage access_modes 不得重复")
        for mode in self.access_modes:
            if mode <= 0:
                raise ValueError("storage access mode 必须是正整数")
        object.__setattr__(self, "access_modes", tuple(sorted(self.access_modes)))
        if not isinstance(self.dependency_keys, tuple):
            raise TypeError("dependency_keys 必须是 tuple")
        dependencies = tuple(
            _key(item, label="storage dependency_key")
            for item in self.dependency_keys
        )
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("storage dependency_keys 不得重复")
        object.__setattr__(self, "dependency_keys", tuple(sorted(dependencies)))
        _key(self.rebuild_protocol_key, label="rebuild_protocol_key", empty=True)
        if self.role == STORAGE_ROLE_REBUILDABLE:
            if not self.dependency_keys or not self.rebuild_protocol_key:
                raise ValueError("可重建存储必须声明依赖和 rebuild protocol")

    def stable_key(self) -> tuple[int, ...]:
        """返回角色描述的完整稳定整数键。"""
        result = [len(self.descriptor_key), *self.descriptor_key, self.role]
        result.append(len(self.access_modes))
        result.extend(self.access_modes)
        result.append(len(self.dependency_keys))
        for dependency in self.dependency_keys:
            result.extend((len(dependency), *dependency))
        result.extend((len(self.rebuild_protocol_key), *self.rebuild_protocol_key))
        return tuple(result)

    def can_discard(self) -> bool:
        """判断物理释放是否有明确的语义安全依据。"""
        return self.role in {
            STORAGE_ROLE_REBUILDABLE,
            STORAGE_ROLE_EPHEMERAL,
        }


class StorageRoleRegistry:
    """按完整 descriptor key 管理上下文内存储角色，拒绝静默覆盖。"""

    def __init__(self) -> None:
        """创建不带全局状态的角色注册表。"""
        self._descriptors: dict[tuple[int, ...], StorageRoleDescriptor] = {}

    def register(self, descriptor: StorageRoleDescriptor) -> None:
        """注册角色描述；相同完整描述幂等，不同描述 fail closed。"""
        if not isinstance(descriptor, StorageRoleDescriptor):
            raise TypeError("storage descriptor 类型错误")
        previous = self._descriptors.get(descriptor.descriptor_key)
        if previous is not None and previous != descriptor:
            raise ValueError("同一 storage descriptor_key 发生定义漂移")
        self._descriptors[descriptor.descriptor_key] = descriptor

    def get(self, descriptor_key: tuple[int, ...]) -> StorageRoleDescriptor:
        """按完整 descriptor key 读取角色描述，未知项拒绝继续。"""
        key = _key(descriptor_key, label="storage registry lookup")
        try:
            return self._descriptors[key]
        except KeyError as exc:
            raise KeyError(f"未注册 storage descriptor: {key}") from exc

    def descriptors(self) -> tuple[StorageRoleDescriptor, ...]:
        """按完整 descriptor key 返回确定性注册快照。"""
        return tuple(self._descriptors[key]
                     for key in sorted(self._descriptors))

    def stable_key(self) -> tuple[int, ...]:
        """返回注册表中全部角色描述的稳定键。"""
        result: list[int] = [len(self._descriptors)]
        for descriptor in self.descriptors():
            key = descriptor.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


__all__ = [
    "STORAGE_ACCESS_APPEND_ONLY",
    "STORAGE_ACCESS_INDEXED_READ",
    "STORAGE_ACCESS_MUTABLE",
    "STORAGE_ACCESS_REBUILD",
    "STORAGE_ROLE_AUTHORITATIVE",
    "STORAGE_ROLE_EPHEMERAL",
    "STORAGE_ROLE_REBUILDABLE",
    "StorageRoleDescriptor",
    "StorageRoleRegistry",
]
