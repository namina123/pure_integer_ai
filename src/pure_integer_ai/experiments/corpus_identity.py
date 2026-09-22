"""受控语料项的显式来源、文档 scope 和稳定整数桥键。

生产 collection source 应尽量直接提供 ``SourceRef``。手工 fixture 或旧 loader 缺来源时，
本模块使用输入内容生成匿名 source id，并用同内容 occurrence ordinal 作 document id；这
比对象地址稳定，但不冒充真实出版物或外部数据集 provenance。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.crosscut.determinism.hasher import Hasher

if TYPE_CHECKING:
    from pure_integer_ai.experiments.collection import CollectedItem


_ANONYMOUS_SOURCE_HASHER = Hasher("collected_item.anonymous_source.v1")
_CONTENT_HASHER = Hasher("collected_item.content_integer.v1")


class CorpusContentUniquenessError(RuntimeError):
    """训练 corpus 中同一内容以多个记录出现。"""


def _scalar_sequence(value: str) -> tuple[int, ...]:
    """把表层直接编码为 Unicode 码点整数，不做规范化或词表映射。"""
    if not isinstance(value, str):
        raise TypeError("内容必须是字符串")
    return tuple(ord(char) for char in value)


def integer_content_key(item: "CollectedItem") -> tuple[int, ...]:
    """返回跨宿主可重建的纯整数内容身份。"""
    if not hasattr(item, "modality"):
        raise TypeError("内容身份需要 CollectedItem")
    dimensions = (item.modality, item.lang, item.domain)
    if any(type(value) is not int or value < 0 for value in dimensions):
        raise ValueError("内容身份维度必须是非负严格整数")
    prefix = list(dimensions)
    if item.arith_source is not None:
        values = _scalar_sequence(item.arith_source)
        return tuple((*prefix, 1, len(values), *values))
    if item.code_source is not None:
        values = _scalar_sequence(item.code_source)
        return tuple((*prefix, 2, len(values), *values))
    if item.raw_text is not None:
        values = _scalar_sequence(item.raw_text)
        return tuple((*prefix, 3, len(values), *values))
    reader = getattr(item, "token_values", None)
    tokens = tuple(reader()) if callable(reader) else tuple(item.tokens)
    result = [*prefix, 4, len(tokens)]
    for token in tokens:
        values = _scalar_sequence(token)
        result.extend((len(values), *values))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CorpusContentUniquenessReport:
    """训练输入内容精确唯一性审计结果。"""

    item_count: int
    unique_content_count: int
    duplicate_group_count: int
    duplicate_excess: int
    groups: tuple[tuple[int, int, tuple[tuple[int, ...], ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "identity": "integer_unicode_sequence_exact",
            "item_count": self.item_count,
            "unique_content_count": self.unique_content_count,
            "duplicate_group_count": self.duplicate_group_count,
            "duplicate_excess": self.duplicate_excess,
            "groups": [
                {"content_hash": content_hash, "count": count,
                 "source_keys": [list(key) for key in source_keys]}
                for content_hash, count, source_keys in self.groups
            ],
        }


@dataclass(frozen=True, slots=True)
class CorpusContentAggregationReport:
    """内容聚合结果；被合并来源仍以整数键保留在审计中。"""

    input_count: int
    output_count: int
    aggregated_group_count: int
    aggregated_excess: int
    groups: tuple[tuple[int, int, tuple[tuple[int, ...], ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "identity": "integer_unicode_sequence_exact",
            "input_count": self.input_count,
            "output_count": self.output_count,
            "aggregated_group_count": self.aggregated_group_count,
            "aggregated_excess": self.aggregated_excess,
            "groups": [
                {"content_hash": content_hash, "count": count,
                 "source_keys": [list(key) for key in source_keys]}
                for content_hash, count, source_keys in self.groups
            ],
        }


def aggregate_corpus_content(
        items: Iterable["CollectedItem"], *,
        protected: Iterable["CollectedItem"] = (),
        ) -> tuple[tuple["CollectedItem", ...], CorpusContentAggregationReport]:
    """按完整整数内容聚合 corpus，保留 protected 项（如 held-out）。

    不删除来源证据：每个聚合组的所有 SourceRef 整数键写入报告；只将完全相同
    的训练输入减少为一个图消费项。protected 项发生冲突时替换未保护项，保证
    评测输入仍可回读。
    """
    materialized = tuple(items)
    protected_ids = {id(item) for item in protected}
    kept: list[CollectedItem] = []
    index_by_key: dict[tuple[int, ...], int] = {}
    source_groups: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for item in materialized:
        key = integer_content_key(item)
        source = item.source_ref
        source_groups.setdefault(key, []).append(
            () if source is None else source.stable_key())
        index = index_by_key.get(key)
        if index is None:
            index_by_key[key] = len(kept)
            kept.append(item)
            continue
        prior = kept[index]
        if id(item) in protected_ids and id(prior) not in protected_ids:
            kept[index] = item
    groups = []
    excess = 0
    for key, sources in sorted(source_groups.items()):
        if len(sources) <= 1:
            continue
        excess += len(sources) - 1
        groups.append((_CONTENT_HASHER.h63(key), len(sources),
                       tuple(sorted(sources))))
    return tuple(kept), CorpusContentAggregationReport(
        len(materialized), len(kept), len(groups), excess, tuple(groups))


def audit_corpus_content_uniqueness(
        items: Iterable["CollectedItem"], *, fail_closed: bool = True,
        ) -> CorpusContentUniquenessReport:
    """按完整整数内容键审计 corpus，重复时默认阻断训练。"""
    grouped: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    materialized = tuple(items)
    for item in materialized:
        key = integer_content_key(item)
        source = item.source_ref
        source_key = () if source is None else source.stable_key()
        grouped.setdefault(key, []).append(source_key)
    duplicate_groups = []
    duplicate_excess = 0
    for key, source_keys in sorted(grouped.items()):
        if len(source_keys) <= 1:
            continue
        duplicate_excess += len(source_keys) - 1
        duplicate_groups.append((_CONTENT_HASHER.h63(key), len(source_keys),
                                 tuple(sorted(source_keys))))
    report = CorpusContentUniquenessReport(
        len(materialized), len(grouped), len(duplicate_groups),
        duplicate_excess, tuple(duplicate_groups))
    if fail_closed and report.duplicate_group_count:
        raise CorpusContentUniquenessError(
            "corpus content duplicate: " + str(report.to_dict()))
    return report


def _item_content_key(item: "CollectedItem") -> tuple:
    """提取文档内容身份所需的稳定输入，不混入运行期缓存字段。"""
    primary_content = (
        item.arith_source
        if item.arith_source is not None
        else item.code_source
        if item.code_source is not None
        else item.raw_text
        if item.raw_text is not None
        else tuple(item.tokens)
    )
    return (
        item.source,
        item.collect_type,
        item.modality,
        item.lang,
        item.domain,
        primary_content,
    )


def _positive_hash(value: tuple) -> int:
    """生成可作 SourceRef.source_id 的非零稳定整数。"""
    source_id = _ANONYMOUS_SOURCE_HASHER.h63(value)
    return source_id if source_id > 0 else 1


def assign_corpus_source_refs(
        items: Iterable["CollectedItem"], *,
        source_namespace: str | int | None = None) -> None:
    """给缺来源的语料项批量补匿名 SourceRef。

    该函数只分配可回溯的来源键；训练入口随后必须通过
    :func:`audit_corpus_content_uniqueness`，不会借 document ordinal 掩盖重复内容。
    """
    occurrence_by_content: dict[tuple, int] = {}
    for item in items:
        if item.source_ref is not None:
            if item.source_ref.source_kind != item.source:
                raise ValueError("CollectedItem.source 与 SourceRef.source_kind 不一致")
            continue
        content_key = _item_content_key(item)
        ordinal = occurrence_by_content.get(content_key, 0)
        occurrence_by_content[content_key] = ordinal + 1
        item.source_ref = SourceRef(
            item.source,
            _positive_hash((source_namespace, content_key)),
            ordinal,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(),
        )


def ensure_item_scope(item: "CollectedItem",
                      store: ScopedIdentityStore) -> tuple[ScopeIdentity, int]:
    """确保语料项具有 document scope，并返回经全键核验的 registry 索引。"""
    if item.source_ref is None:
        assign_corpus_source_refs((item,))
    if item.source_ref is None:
        raise RuntimeError("CollectedItem SourceRef 补全失败")
    scope = document_scope(item.source_ref)
    scope_hash = store.register_scope(scope)
    if item.document_scope_hash not in (0, scope_hash):
        raise ValueError("CollectedItem 缓存的 document scope hash 与完整身份不一致")
    item.document_scope_hash = scope_hash
    return scope, scope_hash


__all__ = [
    "CorpusContentUniquenessError", "CorpusContentUniquenessReport",
    "CorpusContentAggregationReport",
    "aggregate_corpus_content",
    "assign_corpus_source_refs", "audit_corpus_content_uniqueness",
    "ensure_item_scope", "integer_content_key",
]
