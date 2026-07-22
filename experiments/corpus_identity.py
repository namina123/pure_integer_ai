"""受控语料项的显式来源、文档 scope 和稳定整数桥键。

生产 collection source 应尽量直接提供 ``SourceRef``。手工 fixture 或旧 loader 缺来源时，
本模块使用输入内容生成匿名 source id，并用同内容 occurrence ordinal 作 document id；这
比对象地址稳定，但不冒充真实出版物或外部数据集 provenance。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

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
    """给缺来源的语料项批量补匿名 SourceRef，并保留同内容重复 occurrence。"""
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


__all__ = ["assign_corpus_source_refs", "ensure_item_scope"]
