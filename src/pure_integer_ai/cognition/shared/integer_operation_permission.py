"""训练来源对既有整数程序的显式许可增量；不提供事实、词表或对象常量。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.relation_composition import schema_from_key
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


SOURCE_PERMISSION = 92290
PERMISSION_SOURCE = 92291
_PROGRAM_WITNESS = 92206
_SEQUENCE = 92202


def _key(value):
    """许可字段只能使用非空规范非负整数序列。"""
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 or item >= 1 << 63 for item in value)):
        raise ValueError('operation permission requires a canonical integer key')
    return value


@dataclass(frozen=True, slots=True)
class OperationSourcePermission:
    """完整原始程序、教师指令、课程身份、schema及forming来源的一等记录。"""

    base_witness_key: tuple[int, ...]
    source: SourceRef
    source_sha256: tuple[int, ...]
    course_source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    schema_keys: tuple[tuple[int, ...], ...]
    forming_sources: tuple[SourceRef, ...]

    def __post_init__(self):
        """只接受非递归旧witness、完整SHA与规范来源/schema；不凭id授予权限。"""
        if len(unpack_record(_key(self.base_witness_key), _PROGRAM_WITNESS)) != 6:
            raise ValueError('operation permission must name an unextended base witness')
        digest = _key(self.source_sha256)
        if (not isinstance(self.source, SourceRef) or self.source.source_kind != PERMISSION_SOURCE
                or len(digest) != 32 or any(value > 255 for value in digest)
                or self.source.source_id != ((int.from_bytes(bytes(digest[:8]), 'big') & ((1 << 63) - 1)) or 1)):
            raise ValueError('operation permission lacks its physical teacher source digest')
        _key(self.course_source_identity)
        _key(self.license_identity)
        if (type(self.schema_keys) is not tuple or not self.schema_keys
                or self.schema_keys != tuple(sorted(set(self.schema_keys)))):
            raise ValueError('operation permission requires complete canonical schemas')
        for key in self.schema_keys:
            schema_from_key(_key(key))
        if (type(self.forming_sources) is not tuple or not self.forming_sources
                or any(not isinstance(source, SourceRef) for source in self.forming_sources)
                or self.forming_sources != tuple(sorted(set(self.forming_sources), key=SourceRef.stable_key))):
            raise ValueError('operation permission requires complete canonical forming sources')

    def stable_key(self):
        """完整成员有序编码；摘要和枚举不能替代许可本体。"""
        return pack_record(SOURCE_PERMISSION, self.base_witness_key, self.source.stable_key(),
            self.source_sha256, self.course_source_identity, self.license_identity,
            pack_record(_SEQUENCE, *self.schema_keys),
            pack_record(_SEQUENCE, *(source.stable_key() for source in self.forming_sources)))

    @classmethod
    def from_stable_key(cls, key):
        """严格恢复全部七字段，拒绝截断、非规范成员和递归根。"""
        fields = unpack_record(key, SOURCE_PERMISSION)
        if len(fields) != 7:
            raise ValueError('operation source permission is incomplete')
        result = cls(fields[0], SourceRef.from_stable_key(fields[1]), fields[2], fields[3], fields[4],
            unpack_record(fields[5], _SEQUENCE),
            tuple(SourceRef.from_stable_key(row) for row in unpack_record(fields[6], _SEQUENCE)))
        if result.stable_key() != key:
            raise ValueError('operation source permission is not canonical')
        return result
