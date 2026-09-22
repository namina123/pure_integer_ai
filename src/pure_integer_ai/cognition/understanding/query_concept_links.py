"""训练图的显式概念关联及输入候选路径；闭包不合并任何对象身份。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def _key(value: tuple[int, ...]) -> tuple[int, ...]:
    """核验完整非负整数引用，不接受隐式布尔或浮点。"""
    if type(value) is not tuple or not value or any(
            type(item) is not int or item < 0 for item in value):
        raise ValueError("概念关联引用必须是非空非负整数 tuple")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """以长度边界保留完整图键。"""
    return len(value), *value


def _parts(value: tuple[int, ...], count: int) -> tuple[tuple[int, ...], ...]:
    """恢复版本化长度前缀记录，拒绝截断、尾随和未知版本。"""
    _key(value)
    if value[0] != 1:
        raise ValueError("概念关联记录版本未注册")
    cursor = 1
    result = []
    for _ in range(count):
        if cursor >= len(value) or value[cursor] <= 0:
            raise ValueError("概念关联记录缺少字段")
        end = cursor + 1 + value[cursor]
        if end > len(value):
            raise ValueError("概念关联记录字段被截断")
        result.append(value[cursor + 1:end])
        cursor = end
    if cursor != len(value):
        raise ValueError("概念关联记录有尾随字段")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TrainedConceptLink:
    """一个带方向的图内概念关系；语义资格由训练图适配器核验。"""

    origin_key: tuple[int, ...]
    target_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """保留两个独立节点、完整命题、来源和范围。"""
        for name in self.__dataclass_fields__:
            _key(getattr(self, name))

    def stable_key(self) -> tuple[int, ...]:
        """返回可以恢复节点和证据关系的整数记录，而非闭包签名。"""
        return (1, *_pack(self.origin_key), *_pack(self.target_key),
                *_pack(self.proposition_key), *_pack(self.predicate_key),
                *_pack(self.source_ref), *_pack(self.scope_key))

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> TrainedConceptLink:
        """从持久证据/trace 恢复完整图关系，不依赖邻接缓存。"""
        return cls(*_parts(key, 6))


@dataclass(frozen=True, slots=True)
class InputConceptRoute:
    """输入 span 经显式概念关系投影到另一训练角色的开放候选。"""

    span_ref: tuple[int, ...]
    origin_key: tuple[int, ...]
    target_key: tuple[int, ...]
    origin_proposition_key: tuple[int, ...]
    target_proposition_key: tuple[int, ...]
    origin_member_key: tuple[int, ...]
    target_member_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """完整保留两端训练成员，不能以 target 表层替换输入 span。"""
        for name in self.__dataclass_fields__:
            _key(getattr(self, name))

    def stable_key(self) -> tuple[int, ...]:
        """返回输入位置、节点及两端角色/来源的完整引用。"""
        return (1, *_pack(self.span_ref), *_pack(self.origin_key),
                *_pack(self.target_key), *_pack(self.origin_proposition_key),
                *_pack(self.target_proposition_key),
                *_pack(self.origin_member_key),
                *_pack(self.target_member_key))

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> InputConceptRoute:
        """恢复两端训练成员和原始输入位置，不重新分析来源正文。"""
        return cls(*_parts(key, 7))


@dataclass(frozen=True, slots=True)
class TrainedConceptLinks:
    """只持有显式关系的可重建邻接索引；不把索引当作语义本体。"""

    links: tuple[TrainedConceptLink, ...]
    adjacency: dict[tuple[int, ...], tuple[TrainedConceptLink, ...]] = field(
        init=False, repr=False, compare=False)
    expression_origins: frozenset[tuple[tuple[int, ...], tuple[int, ...]]] = field(
        init=False, compare=False)

    def __post_init__(self) -> None:
        """按图中原始方向索引全部关系，包括并列证据和环。"""
        links = self.links
        if type(links) is not tuple or any(not isinstance(link, TrainedConceptLink)
                                           for link in links):
            raise TypeError("concept links 必须是 TrainedConceptLink tuple")
        object.__setattr__(self, "links", tuple(sorted(
            set(links), key=lambda item: item.stable_key())))
        adjacency: dict[tuple[int, ...], list[TrainedConceptLink]] = {}
        for link in self.links:
            adjacency.setdefault(link.origin_key, []).append(link)
        object.__setattr__(self, "adjacency", {
            key: tuple(value) for key, value in adjacency.items()})
        object.__setattr__(self, "expression_origins", frozenset(
            (link.proposition_key, link.origin_key) for link in self.links))

    def reachable(
            self, origin: tuple[int, ...],
            *, evidence_claims: frozenset[tuple[int, ...]] | None = None,
            ) -> tuple[frozenset[tuple[int, ...]], tuple[TrainedConceptLink, ...]]:
        """有限图 visited 到稳定点；保留全部遇到的边，不截断深度或任取路径。

        无 evidence_claims 时只形成开放候选；有该参数时只能沿共同 QueryState
        中已绑定的命题证据行进，未展开的边不能用于回答闭合。
        """
        _key(origin)
        reached = {origin}
        pending = deque((origin,))
        traversed: set[TrainedConceptLink] = set()
        while pending:
            current = pending.popleft()
            for link in self.adjacency.get(current, ()):
                if evidence_claims is not None and link.proposition_key not in evidence_claims:
                    continue
                traversed.add(link)
                if link.target_key not in reached:
                    reached.add(link.target_key)
                    pending.append(link.target_key)
        return frozenset(reached), tuple(sorted(traversed, key=lambda item: item.stable_key()))


__all__ = ["InputConceptRoute", "TrainedConceptLink", "TrainedConceptLinks"]
