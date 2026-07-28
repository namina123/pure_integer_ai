"""长输入分块重组、绝对 Span 层级和 prefix/content digest 合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_PROPOSITION,
    OBJECT_SPAN,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    TypedRef,
    normalize_span_members,
    span_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.understanding.span_index import SpanIndex
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


HIERARCHY_LEVEL_DOCUMENT = 1
HIERARCHY_LEVEL_SECTION = 2
HIERARCHY_LEVEL_PARAGRAPH = 3
HIERARCHY_LEVEL_PROPOSITION = 4
_DIGEST_SIZE = 32


class LongInputHierarchyError(RuntimeError):
    """分块、来源、绝对 Span、层级或 digest 不闭合。"""


def _digest(values: tuple[int, ...]) -> tuple[int, ...]:
    """对规范整数流计算 SHA-256，并以无文本整数元组返回。"""
    return tuple(hashlib.sha256(encode_integer_tuple(values)).digest())


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """为可变长稳定键增加长度边界。"""
    result.extend((len(value), *value))


def _key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """要求调用方键非空且只含严格整数。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise LongInputHierarchyError(f"{where} 必须是非空严格整数 tuple")
    return value


def _digest_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """要求 digest 是完整 SHA-256 字节整数流。"""
    if (not isinstance(value, tuple) or len(value) != _DIGEST_SIZE
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise LongInputHierarchyError(f"{where} 必须是 SHA-256 字节 tuple")
    return value


def _identity(
        value: ObjectIdentity, kind: int, *, where: str,
        ) -> ObjectIdentity:
    """核验一等身份类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if value.object_kind != kind:
        raise LongInputHierarchyError(f"{where} 对象类型不匹配")
    return value


def _members_key(members: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """把规范绝对区间编码为稳定整数键。"""
    normalized = normalize_span_members(members)
    return (
        len(normalized),
        *(value for member in normalized for value in member),
    )


def _contains(
        outer: tuple[tuple[int, int], ...],
        inner: tuple[tuple[int, int], ...],
        ) -> bool:
    """判断 inner 的每个绝对成员是否完整落在某个 outer 成员内。"""
    return all(
        any(outer_start <= start and end <= outer_end
            for outer_start, outer_end in outer)
        for start, end in inner
    )


def _surface_units(
        text: str, members: tuple[tuple[int, int], ...],
        ) -> tuple[int, ...]:
    """按规范成员顺序提取实际原文码点，不拼接宿主分隔符。"""
    values: list[int] = []
    for start, end in members:
        values.extend(ord(char) for char in text[start:end])
    return tuple(values)


@dataclass(frozen=True, order=True)
class LongInputChunk:
    """一个 SourceRef 内带绝对起点和自身内容摘要的输入块。"""

    source: SourceRef
    absolute_start: int
    text: str
    content_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 source、绝对起点、非空文本和块摘要。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("long input chunk source 类型错误")
        if type(self.absolute_start) is not int or self.absolute_start < 0:
            raise LongInputHierarchyError("chunk absolute_start 非法")
        if not isinstance(self.text, str) or not self.text:
            raise LongInputHierarchyError("chunk text 必须非空")
        _digest_key(self.content_digest, where="chunk content_digest")
        expected = _digest(tuple(ord(char) for char in self.text))
        if self.content_digest != expected:
            raise LongInputHierarchyError("chunk 内容与 digest 不一致")

    @classmethod
    def from_text(
            cls, source: SourceRef, absolute_start: int, text: str,
            ) -> "LongInputChunk":
        """从原始文本块形成自校验输入。"""
        if not isinstance(text, str) or not text:
            raise LongInputHierarchyError("chunk text 必须非空")
        return cls(
            source,
            absolute_start,
            text,
            _digest(tuple(ord(char) for char in text)),
        )

    @property
    def absolute_end(self) -> int:
        """返回块在完整来源中的绝对尾位置。"""
        return self.absolute_start + len(self.text)

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、绝对区间和内容摘要的稳定键。"""
        return (
            *self.source.stable_key(),
            self.absolute_start,
            self.absolute_end,
            *self.content_digest,
        )


@dataclass(frozen=True, order=True)
class LongInputHierarchySeed:
    """调用方已经形成的层级边界、命题和 episode 绑定。"""

    proposition: ObjectIdentity
    episode: ObjectIdentity
    proposition_ordinal: int
    document_members: tuple[tuple[int, int], ...]
    section_members: tuple[tuple[int, int], ...]
    paragraph_members: tuple[tuple[int, int], ...]
    proposition_members: tuple[tuple[int, int], ...]
    document_ordinal: int
    section_ordinal: int
    paragraph_ordinal: int
    span_ordinal: int

    def __post_init__(self) -> None:
        """核验 typed 命题/episode、全局 ordinal 和嵌套绝对边界。"""
        _identity(self.proposition, OBJECT_PROPOSITION, where="proposition")
        _identity(self.episode, OBJECT_CONTEXT_SCOPE, where="episode")
        for name in (
                "proposition_ordinal", "document_ordinal", "section_ordinal",
                "paragraph_ordinal", "span_ordinal"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise LongInputHierarchyError(f"{name} 必须为非负严格整数")
        document = normalize_span_members(self.document_members)
        section = normalize_span_members(self.section_members)
        paragraph = normalize_span_members(self.paragraph_members)
        proposition = normalize_span_members(self.proposition_members)
        object.__setattr__(self, "document_members", document)
        object.__setattr__(self, "section_members", section)
        object.__setattr__(self, "paragraph_members", paragraph)
        object.__setattr__(self, "proposition_members", proposition)
        if (not _contains(document, section)
                or not _contains(section, paragraph)
                or not _contains(paragraph, proposition)):
            raise LongInputHierarchyError("document/section/paragraph/proposition 未嵌套")


@dataclass(frozen=True, order=True)
class LongInputHierarchyRecord:
    """一条命题的来源化四层 Span、episode、绝对成员和摘要。"""

    source: SourceRef
    scope: ScopeIdentity
    proposition: ObjectIdentity
    episode: ObjectIdentity
    proposition_ordinal: int
    document_span: ObjectIdentity
    section_span: ObjectIdentity
    paragraph_span: ObjectIdentity
    proposition_span: ObjectIdentity
    document_members: tuple[tuple[int, int], ...]
    section_members: tuple[tuple[int, int], ...]
    paragraph_members: tuple[tuple[int, int], ...]
    proposition_members: tuple[tuple[int, int], ...]
    prefix_digest: tuple[int, ...]
    content_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        """重验来源、scope、Span identity、绝对层级和两个摘要形状。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("hierarchy record source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("hierarchy record scope 类型错误")
        if self.scope.source != self.source:
            raise LongInputHierarchyError("hierarchy scope 未绑定 source")
        if (semantic_source(self.proposition) != self.source
                or semantic_source(self.episode) != self.source):
            raise LongInputHierarchyError("proposition/episode 来源漂移")
        span_bindings = (
            (self.document_span, self.document_members, "document span"),
            (self.section_span, self.section_members, "section span"),
            (self.paragraph_span, self.paragraph_members, "paragraph span"),
            (self.proposition_span, self.proposition_members,
             "proposition span"),
        )
        for value, _, where in span_bindings:
            _identity(value, OBJECT_SPAN, where=where)
            if value.owner != self.source.owner or value.versions != self.source.versions:
                raise LongInputHierarchyError(f"{where} owner/version 漂移")
        if type(self.proposition_ordinal) is not int or self.proposition_ordinal < 0:
            raise LongInputHierarchyError("proposition ordinal 非法")
        for name in (
                "document_members", "section_members", "paragraph_members",
                "proposition_members"):
            object.__setattr__(
                self, name, normalize_span_members(getattr(self, name)))
        if (not _contains(self.document_members, self.section_members)
                or not _contains(self.section_members, self.paragraph_members)
                or not _contains(
                    self.paragraph_members, self.proposition_members)):
            raise LongInputHierarchyError("hierarchy record 绝对层级断裂")
        for value, members, where in span_bindings:
            if len(value.components) < 12:
                raise LongInputHierarchyError(f"{where} identity 被截断")
            expected = span_identity(
                self.source,
                members=members,
                ordinal=value.components[11],
            )
            if value != expected:
                raise LongInputHierarchyError(f"{where} 未绑定完整 SourceRef")
        _digest_key(self.prefix_digest, where="prefix_digest")
        _digest_key(self.content_digest, where="content_digest")

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、层级 identity、absolute span 和摘要的完整键。"""
        result: list[int] = []
        for value in (
                self.source.stable_key(),
                self.scope.stable_key(),
                self.proposition.stable_key(),
                self.episode.stable_key()):
            _pack(result, value)
        result.append(self.proposition_ordinal)
        for value in (
                self.document_span.stable_key(),
                self.section_span.stable_key(),
                self.paragraph_span.stable_key(),
                self.proposition_span.stable_key(),
                _members_key(self.document_members),
                _members_key(self.section_members),
                _members_key(self.paragraph_members),
                _members_key(self.proposition_members),
                self.prefix_digest,
                self.content_digest):
            _pack(result, value)
        return tuple(result)


@dataclass(frozen=True)
class LongInputHierarchy:
    """一个完整 SourceRef 文档的规范摘要和全部命题层级记录。"""

    source: SourceRef
    scope: ScopeIdentity
    document_digest: tuple[int, ...]
    text_length: int
    records: tuple[LongInputHierarchyRecord, ...]

    def __post_init__(self) -> None:
        """核验记录来源一致、ordinal 唯一且规范排序。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("long input hierarchy source 类型错误")
        if not isinstance(self.scope, ScopeIdentity) or self.scope.source != self.source:
            raise LongInputHierarchyError("long input hierarchy scope 非法")
        _digest_key(self.document_digest, where="document_digest")
        if type(self.text_length) is not int or self.text_length <= 0:
            raise LongInputHierarchyError("long input text_length 非法")
        if (not isinstance(self.records, tuple) or not self.records
                or any(not isinstance(item, LongInputHierarchyRecord)
                       for item in self.records)):
            raise LongInputHierarchyError("long input records 非法")
        ordered = tuple(sorted(
            self.records,
            key=lambda item: (
                item.proposition_ordinal,
                item.proposition.stable_key(),
            ),
        ))
        if ordered != self.records:
            raise LongInputHierarchyError("long input records 未规范排序")
        if len({item.proposition_ordinal for item in ordered}) != len(ordered):
            raise LongInputHierarchyError("proposition ordinal 重复")
        if any(item.source != self.source or item.scope != self.scope
               for item in ordered):
            raise LongInputHierarchyError("hierarchy record 跨 source/scope")

    def stable_key(self) -> tuple[int, ...]:
        """返回与 chunk 宽度和调用顺序无关的规范 hierarchy key。"""
        result = [
            *self.source.stable_key(),
            len(self.scope.stable_key()),
            *self.scope.stable_key(),
            *self.document_digest,
            self.text_length,
            len(self.records),
        ]
        for item in self.records:
            _pack(result, item.stable_key())
        return tuple(result)


@dataclass(frozen=True)
class LongInputHierarchyProtocol:
    """注入 document/section/paragraph/proposition 四个 StructureConcept。"""

    document_structure: ObjectIdentity
    section_structure: ObjectIdentity
    paragraph_structure: ObjectIdentity
    proposition_structure: ObjectIdentity

    def __post_init__(self) -> None:
        """要求四种结构一等、互异且不从 surface 名称推断。"""
        values = (
            self.document_structure,
            self.section_structure,
            self.paragraph_structure,
            self.proposition_structure,
        )
        for index, value in enumerate(values):
            _identity(
                value,
                OBJECT_STRUCTURE_CONCEPT,
                where=f"hierarchy structure[{index}]",
            )
        if len(set(values)) != len(values):
            raise LongInputHierarchyError("hierarchy structure 必须互异")


class LongInputHierarchyBuilder:
    """按绝对 offset 重组任意 chunk，并形成来源化层级和摘要。"""

    @staticmethod
    def assemble(
            chunks: tuple[LongInputChunk, ...],
            ) -> tuple[SourceRef, str, tuple[int, ...]]:
        """按绝对起点重组完整原文，拒绝空洞、重叠和 source 混合。"""
        if (not isinstance(chunks, tuple) or not chunks
                or any(not isinstance(item, LongInputChunk) for item in chunks)):
            raise LongInputHierarchyError("chunks 必须是非空 LongInputChunk tuple")
        source = chunks[0].source
        if any(item.source != source for item in chunks):
            raise LongInputHierarchyError("long input chunks 跨 SourceRef")
        ordered = tuple(sorted(
            chunks, key=lambda item: (item.absolute_start, item.absolute_end)))
        cursor = 0
        text_parts = []
        for item in ordered:
            if item.absolute_start != cursor:
                raise LongInputHierarchyError("chunk absolute offset 存在空洞或重叠")
            text_parts.append(item.text)
            cursor = item.absolute_end
        text = "".join(text_parts)
        digest = _digest(tuple(ord(char) for char in text))
        return source, text, digest

    def build(
            self,
            chunks: tuple[LongInputChunk, ...],
            scope: ScopeIdentity,
            seeds: tuple[LongInputHierarchySeed, ...],
            *, expected_document_digest: tuple[int, ...] | None = None,
            ) -> LongInputHierarchy:
        """形成与 chunk 切法/输入顺序无关的规范 hierarchy。"""
        source, text, document_digest = self.assemble(chunks)
        if not isinstance(scope, ScopeIdentity) or scope.source != source:
            raise LongInputHierarchyError("builder scope 未绑定 chunk source")
        if expected_document_digest is not None:
            _digest_key(
                expected_document_digest,
                where="expected document digest",
            )
            if document_digest != expected_document_digest:
                raise LongInputHierarchyError("document content/prefix digest 漂移")
        if (not isinstance(seeds, tuple) or not seeds
                or any(not isinstance(item, LongInputHierarchySeed)
                       for item in seeds)):
            raise LongInputHierarchyError("hierarchy seeds 非法")
        records = []
        for seed in seeds:
            if (semantic_source(seed.proposition) != source
                    or semantic_source(seed.episode) != source):
                raise LongInputHierarchyError("seed proposition/episode 跨 source")
            if any(end > len(text) for _, end in seed.document_members):
                raise LongInputHierarchyError("seed absolute span 超出完整原文")
            proposition_start = seed.proposition_members[0][0]
            prefix = _digest(tuple(ord(char) for char in text[:proposition_start]))
            content = _digest(_surface_units(text, seed.proposition_members))
            records.append(LongInputHierarchyRecord(
                source,
                scope,
                seed.proposition,
                seed.episode,
                seed.proposition_ordinal,
                span_identity(
                    source,
                    members=seed.document_members,
                    ordinal=seed.document_ordinal,
                ),
                span_identity(
                    source,
                    members=seed.section_members,
                    ordinal=seed.section_ordinal,
                ),
                span_identity(
                    source,
                    members=seed.paragraph_members,
                    ordinal=seed.paragraph_ordinal,
                ),
                span_identity(
                    source,
                    members=seed.proposition_members,
                    ordinal=seed.span_ordinal,
                ),
                seed.document_members,
                seed.section_members,
                seed.paragraph_members,
                seed.proposition_members,
                prefix,
                content,
            ))
        return LongInputHierarchy(
            source,
            scope,
            document_digest,
            len(text),
            tuple(sorted(
                records,
                key=lambda item: (
                    item.proposition_ordinal,
                    item.proposition.stable_key(),
                ),
            )),
        )

    def materialize(
            self,
            hierarchy: LongInputHierarchy,
            chunks: tuple[LongInputChunk, ...],
            spans: SpanIndex,
            protocol: LongInputHierarchyProtocol,
            ) -> tuple[tuple[ObjectIdentity, TypedRef], ...]:
        """把规范四层 Span 写入现役 SpanIndex，并建立无环 constituent 链。"""
        if not isinstance(hierarchy, LongInputHierarchy):
            raise TypeError("hierarchy 类型错误")
        source, text, digest = self.assemble(chunks)
        if source != hierarchy.source or digest != hierarchy.document_digest:
            raise LongInputHierarchyError("materialize 原文 identity 漂移")
        if not isinstance(spans, SpanIndex):
            raise TypeError("spans 必须是 SpanIndex")
        if not isinstance(protocol, LongInputHierarchyProtocol):
            raise TypeError("hierarchy protocol 类型错误")
        refs: dict[ObjectIdentity, object] = {}

        def ensure(
                identity: ObjectIdentity,
                members: tuple[tuple[int, int], ...],
                ordinal: int,
                structure: ObjectIdentity,
                ):
            ref = refs.get(identity)
            if ref is None:
                ref = spans.ensure_ref(
                    source=source,
                    raw_text=text,
                    scope=hierarchy.scope,
                    members=members,
                    ordinal=ordinal,
                    structures=(structure,),
                )
                if spans.ontology.identity_of(ref) != identity:
                    raise LongInputHierarchyError("SpanIndex identity 漂移")
                refs[identity] = ref
            return ref

        links = set()
        proposition_refs = []
        for record in hierarchy.records:
            document_ref = ensure(
                record.document_span,
                record.document_members,
                record.document_span.components[11],
                protocol.document_structure,
            )
            section_ref = ensure(
                record.section_span,
                record.section_members,
                record.section_span.components[11],
                protocol.section_structure,
            )
            paragraph_ref = ensure(
                record.paragraph_span,
                record.paragraph_members,
                record.paragraph_span.components[11],
                protocol.paragraph_structure,
            )
            proposition_ref = ensure(
                record.proposition_span,
                record.proposition_members,
                record.proposition_span.components[11],
                protocol.proposition_structure,
            )
            for parent, child, ordinal in (
                    (document_ref, section_ref,
                     record.section_span.components[11]),
                    (section_ref, paragraph_ref,
                     record.paragraph_span.components[11]),
                    (paragraph_ref, proposition_ref,
                     record.proposition_ordinal)):
                key = (parent, child)
                if key not in links:
                    spans.add_constituent(
                        parent, child, member_ordinal=ordinal)
                    links.add(key)
            proposition_refs.append((
                record.proposition,
                proposition_ref,
            ))
        return tuple(proposition_refs)


__all__ = [
    "HIERARCHY_LEVEL_DOCUMENT",
    "HIERARCHY_LEVEL_PARAGRAPH",
    "HIERARCHY_LEVEL_PROPOSITION",
    "HIERARCHY_LEVEL_SECTION",
    "LongInputChunk",
    "LongInputHierarchy",
    "LongInputHierarchyBuilder",
    "LongInputHierarchyError",
    "LongInputHierarchyProtocol",
    "LongInputHierarchyRecord",
    "LongInputHierarchySeed",
]
