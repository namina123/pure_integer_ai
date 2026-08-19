"""DLG-05 未见来源的 TRAIN 分母与 typed 鉴权边界。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest


# object-model: exception
class ConversationSourceInputError(RuntimeError):
    """held-out 来源重放 TRAIN 或脱离已知来源域。"""


def _source_domain(source: SourceRef) -> tuple[int, ...]:
    """返回来源 kind、owner 与版本域，不包含具体 source/document id。"""
    if not isinstance(source, SourceRef):
        raise TypeError("source domain 需要 SourceRef")
    return (
        source.source_kind,
        *source.owner.stable_key(),
        *source.versions.stable_key(),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationSourceTrainingInventory:
    """正式运行前冻结的 TRAIN SourceRef 分母。"""

    sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        """核验 TRAIN 来源非空、唯一且规范排序。"""
        if (not isinstance(self.sources, tuple) or not self.sources
                or any(not isinstance(item, SourceRef)
                       for item in self.sources)):
            raise ConversationSourceInputError(
                "source TRAIN inventory 必须包含 SourceRef")
        ordered = tuple(sorted(self.sources, key=SourceRef.stable_key))
        if ordered != self.sources:
            raise ConversationSourceInputError(
                "source TRAIN inventory 必须规范排序")
        if len(set(self.sources)) != len(self.sources):
            raise ConversationSourceInputError(
                "source TRAIN inventory 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回全部 TRAIN SourceRef 的完整整数键。"""
        result = [1, len(self.sources)]
        for source in self.sources:
            key = source.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


# object-model: runtime-owner; owns=immutable-train-source-index
class ConversationUnseenSourceCompiler:
    """鉴权精确新 SourceRef，但拒绝完全未见来源域。"""

    def __init__(self, inventory: ConversationSourceTrainingInventory) -> None:
        """冻结 TRAIN exact source 与来源域索引。"""
        if not isinstance(inventory, ConversationSourceTrainingInventory):
            raise TypeError("unseen source compiler inventory 类型错误")
        self.inventory = inventory
        self._sources = set(inventory.sources)
        self._domains = {_source_domain(item) for item in inventory.sources}

    def compile_source(self, source: SourceRef) -> SourceRef:
        """鉴权一个 SourceRef；供 request 边界和反向专项共用。"""
        if not isinstance(source, SourceRef):
            raise TypeError("unseen source compiler source 类型错误")
        if source in self._sources:
            raise ConversationSourceInputError(
                "held-out SourceRef 已出现在 TRAIN")
        if _source_domain(source) not in self._domains:
            raise ConversationSourceInputError(
                "held-out SourceRef 来源域未在 TRAIN 出现")
        return source

    def compile(self, request: QuestionRequest) -> SourceRef:
        """返回已鉴权的新来源；不改变或创建任何来源事实。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("unseen source compiler request 类型错误")
        return self.compile_source(request.source)


__all__ = [
    "ConversationSourceInputError",
    "ConversationSourceTrainingInventory",
    "ConversationUnseenSourceCompiler",
]
