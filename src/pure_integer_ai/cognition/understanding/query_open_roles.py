"""训练图可变角色的整数匹配；解析假设与 Core 事实身份严格分离。"""
from __future__ import annotations

from dataclasses import dataclass


OPEN_FRAME_VERSION = 91512
OPEN_ROLE_VERSION = 91513
OPEN_RELATION_VERSION = 91514
OPEN_SCHEMA_VERSION = 91515


def pack_record(tag: int, *parts: tuple[int, ...]) -> tuple[int, ...]:
    """编码完整整数记录；空序列只作为显式字段边界保存。"""
    if type(tag) is not int or tag <= 0:
        raise ValueError("开放角色记录版本必须为正整数")
    values = [tag]
    for part in parts:
        if type(part) is not tuple or any(type(v) is not int or v < 0 for v in part):
            raise ValueError("开放角色记录只接受非负严格整数")
        values.extend((len(part), *part))
    return tuple(values)


def unpack_record(key: tuple[int, ...], tag: int) -> tuple[tuple[int, ...], ...]:
    """无损解码规范记录，拒绝截断和非法整数。"""
    pack_record(tag, key)
    if not key or key[0] != tag:
        raise ValueError("开放角色记录版本不符")
    cursor = 1
    result = []
    while cursor < len(key):
        end = cursor + 1 + key[cursor]
        if end > len(key):
            raise ValueError("开放角色记录被截断")
        result.append(key[cursor + 1:end])
        cursor = end
    return tuple(result)


@dataclass(frozen=True, slots=True)
class OpenRoleFrame:
    """训练后框架本体投影；每个间隔和角色均保留原图身份及来源。"""

    proposition: tuple[int, ...]
    predicate: tuple[int, ...]
    source_ref: tuple[int, ...]
    roles: tuple[tuple[int, ...], ...]
    gaps: tuple[tuple[int, ...], ...]
    envelope: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.proposition or not self.predicate or not self.source_ref:
            raise ValueError("开放框架缺少训练图引用")
        if (type(self.roles) is not tuple or not self.roles
                or any(not role for role in self.roles)
                or type(self.gaps) is not tuple or len(self.gaps) != len(self.roles) + 1):
            raise ValueError("开放框架角色和间隔不守恒")
        # An all-empty gap frame is an explicitly generic binary parse
        # envelope.  It carries only trained proposition/role identities; the
        # input tokens remain dynamic OpenRoleSpan values and therefore cannot
        # become a Core fact without independent graph evidence.
        if len(self.envelope) != 2 or self.envelope[1] <= self.envelope[0]:
            raise ValueError("开放框架训练包络非法")
        self.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        """角色、顺序、间隔和来源全部进入可逆身份，不只保存摘要。"""
        return pack_record(OPEN_FRAME_VERSION, self.proposition, self.predicate,
                           self.source_ref, self.envelope,
                           pack_record(1, *self.roles), pack_record(2, *self.gaps))

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> OpenRoleFrame:
        """从整数记录恢复完整训练框架。"""
        fields = unpack_record(key, OPEN_FRAME_VERSION)
        if len(fields) != 6:
            raise ValueError("开放框架字段数量不符")
        return cls(*fields[:3], unpack_record(fields[4], 1),
                   unpack_record(fields[5], 2), fields[3])

    def schema_key(self) -> tuple[int, ...]:
        """框架资格证据使用独立 namespace，不冒充其示例命题的事实证据。"""
        return pack_record(OPEN_SCHEMA_VERSION, self.stable_key())


@dataclass(frozen=True, slots=True)
class OpenRoleSpan:
    """输入中尚未解析为已有实体的角色区间，不创建或合并概念节点。"""

    role: tuple[int, ...]
    source_ref: tuple[int, ...]
    ordinal: int
    start: int
    end: int
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.role or not self.source_ref or not self.values:
            raise ValueError("开放角色缺少身份、来源或内容")
        if any(type(v) is not int or v < 0 for v in (self.ordinal, self.start, self.end)):
            raise ValueError("开放角色序和区间必须为非负整数")
        if self.end - self.start != len(self.values):
            raise ValueError("开放角色区间与内容长度不符")
        self.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        """保留来源、区间和全部 token；同字面不同来源不合并。"""
        return pack_record(OPEN_ROLE_VERSION, self.role, self.source_ref,
                           (self.ordinal, self.start, self.end), self.values)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> OpenRoleSpan:
        """无损恢复开放角色。"""
        fields = unpack_record(key, OPEN_ROLE_VERSION)
        if len(fields) != 4 or len(fields[2]) != 3:
            raise ValueError("开放角色字段数量不符")
        return cls(fields[0], fields[1], *fields[2], fields[3])


@dataclass(frozen=True, slots=True)
class OpenRelationCandidate:
    """完整输入上的一条框架解析假设；无真值、无 Core 命题资格。"""

    frame: OpenRoleFrame
    source_ref: tuple[int, ...]
    values: tuple[int, ...]
    bindings: tuple[OpenRoleSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, OpenRoleFrame) or not self.values:
            raise ValueError("开放关系缺少框架或输入")
        if (type(self.bindings) is not tuple or len(self.bindings) != len(self.frame.roles)
                or any(not isinstance(item, OpenRoleSpan) for item in self.bindings)):
            raise ValueError("开放关系角色未完整覆盖")
        cursor = len(self.frame.gaps[0])
        if self.values[:cursor] != self.frame.gaps[0]:
            raise ValueError("开放关系前导结构不符")
        for ordinal, binding in enumerate(self.bindings):
            if (binding.ordinal != ordinal or binding.role != self.frame.roles[ordinal]
                    or binding.source_ref != self.source_ref or binding.start != cursor
                    or self.values[binding.start:binding.end] != binding.values):
                raise ValueError("开放关系角色顺序、来源或区间不闭合")
            gap = self.frame.gaps[ordinal + 1]
            cursor = binding.end + len(gap)
            if self.values[binding.end:cursor] != gap:
                raise ValueError("开放关系连接结构不符")
        if cursor != len(self.values):
            raise ValueError("开放关系不得丢弃未解析输入")
        self.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        """保留完整框架、输入与全部动态角色，作为 Memory 开放候选本体。"""
        return pack_record(OPEN_RELATION_VERSION, self.frame.stable_key(), self.source_ref,
                           self.values, *(item.stable_key() for item in self.bindings))

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> OpenRelationCandidate:
        """无需来源正文或索引即可重建动态角色假设。"""
        fields = unpack_record(key, OPEN_RELATION_VERSION)
        if len(fields) < 4:
            raise ValueError("开放关系记录被截断")
        return cls(OpenRoleFrame.from_stable_key(fields[0]), fields[1], fields[2],
                   tuple(OpenRoleSpan.from_stable_key(item) for item in fields[3:]))


class OpenRoleBudgetExceeded(ValueError):
    """候选枚举超出显式预算；不得截断成一个貌似无歧义的解析。"""


def match_open_roles(frames: tuple[OpenRoleFrame, ...], values: tuple[int, ...],
                     source_ref: tuple[int, ...], *, max_steps: int = 65536,
                     ) -> tuple[OpenRelationCandidate, ...]:
    """按图中间隔约束枚举全输入解析；无字符近邻、正则词表或固定递归深度。

    未知区间只取得角色，不取得示例 filler 身份。连续空间隔的全部非空切分均保留；
    预算耗尽显式失败，不返回局部候选，也不把不完整解析当成没有歧义。
    """
    pack_record(1, values, source_ref)
    if not values or not source_ref or type(max_steps) is not int or max_steps <= 0:
        raise ValueError("开放角色输入和预算非法")
    result = set()
    steps = 0
    for frame in sorted(set(frames), key=lambda item: item.stable_key()):
        prefix = frame.gaps[0]
        if values[:len(prefix)] != prefix:
            continue
        stack = [(0, len(prefix), ())]
        while stack:
            ordinal, start, bindings = stack.pop()
            steps += 1
            if steps > max_steps:
                raise OpenRoleBudgetExceeded("开放角色枚举超出请求预算")
            if ordinal == len(frame.roles):
                if start == len(values):
                    result.add(OpenRelationCandidate(frame, source_ref, values, bindings))
                continue
            gap = frame.gaps[ordinal + 1]
            # 为后续每个角色预留至少一个 token，并保留全部剩余间隔。
            suffix_size = (len(frame.roles) - ordinal - 1
                           + sum(len(item) for item in frame.gaps[ordinal + 1:]))
            last_end = len(values) - suffix_size
            ends = (last_end,) if ordinal == len(frame.roles) - 1 else range(start + 1, last_end + 1)
            for end in ends:
                steps += 1
                if steps > max_steps:
                    raise OpenRoleBudgetExceeded("开放角色枚举超出请求预算")
                if end <= start or values[end:end + len(gap)] != gap:
                    continue
                binding = OpenRoleSpan(frame.roles[ordinal], source_ref, ordinal,
                                       start, end, values[start:end])
                stack.append((ordinal + 1, end + len(gap), (*bindings, binding)))
    return tuple(sorted(result, key=lambda item: item.stable_key()))
