"""LC-16 MARKDOWN payload 的确定性 Markdown tree adapter。

该模块只负责把冻结的 :class:`MarkdownCarrierRecord` 物化为共享
``ArtifactEnvelope``/``ArtifactAnchor``/``ArtifactStructureNode`` 对象。
它不作语义预选、不训练，也不把 Markdown 的 render 结果当作原文；token
的完整 canonical receipt 保存在 structure node 的整数 ``qualifiers`` 中。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import markdown_it
    from markdown_it import MarkdownIt
except ImportError as error:  # pragma: no cover - dependency contract
    raise RuntimeError("MARKDOWN adapter 需要 markdown-it-py") from error

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TREE_PATH,
    RAW_UNIT_UNICODE_SCALAR,
    REVISION_MAP_ANCHOR,
    REVISION_MAP_STRUCTURE_NODE,
    ArtifactAnchor,
    ArtifactCarrierRevision,
    ArtifactEnvelope,
    ArtifactRevisionMapping,
    ArtifactStructureNode,
    make_artifact_anchor,
    make_artifact_carrier_revision,
    make_artifact_envelope,
    make_artifact_structure_node,
)
from pure_integer_ai.cognition.shared.formal_artifact import ArtifactAuthority
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
    concept_identity,
    language_branch_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_contract import (
    MarkdownCarrierRecord,
    PARSER_VERSION,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_MARKDOWN_MATERIALIZATION"
MARKDOWN_SOURCE_KIND = 16616502
_IDENTITY_BASE = 16616520


class MarkdownCarrierAdapterError(RuntimeError):
    """MARKDOWN adapter 输入、树对象或 canonical 表示不闭合。"""


def _key(record: MarkdownCarrierRecord, domain: int, *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: MarkdownCarrierRecord, parser_version: int) -> SourceRef:
    values = record.case_key.stable_key()
    case_index = values[-1]
    owner = OwnerScope(values[0], values[-2], case_index, VISIBILITY_SESSION)
    versions = VersionBundle(
        CorpusVersion(1), ParserVersion(parser_version), PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        MARKDOWN_SOURCE_KIND,
        values[0] + values[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: MarkdownCarrierRecord, domain: int,
             *tail: int):
    return concept_identity(
        _key(record, domain, *tail), owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, record: MarkdownCarrierRecord, domain: int,
               *tail: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain, *tail),
        _concept(source, record, domain + 1, *tail),
    )


def _parser(source: SourceRef, record: MarkdownCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 20)


def _renderer(source: SourceRef, record: MarkdownCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 22)


def _json_safe(value: Any, *, where: str = "receipt") -> Any:
    """把 markdown-it 的 metadata 转成无浮点 canonical JSON 值。"""
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, where=f"{where}[]") for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise MarkdownCarrierAdapterError(f"{where} metadata key 非法")
            result[key] = _json_safe(item, where=f"{where}.{key}")
        return result
    raise MarkdownCarrierAdapterError(
        f"{where} metadata 含不支持类型 {type(value).__name__}")


def _line_starts(text: str) -> tuple[int, ...]:
    starts = [0]
    for index, character in enumerate(text):
        if character == "\n":
            starts.append(index + 1)
    return tuple(starts)


def _token_range(token: Any, text: str, starts: tuple[int, ...],
                 fallback: tuple[int, int]) -> tuple[int, int]:
    token_map = getattr(token, "map", None)
    if (isinstance(token_map, (list, tuple)) and len(token_map) == 2
            and all(type(value) is int for value in token_map)
            and 0 <= token_map[0] <= token_map[1]):
        first, last = token_map
        if first < len(starts):
            start = starts[first]
            end = starts[last] if last < len(starts) else len(text)
            return min(start, len(text)), min(max(start, end), len(text))
    return fallback


def _token_receipt(token: Any, path: tuple[int, ...], ordinal: int,
                   span: tuple[int, int]) -> bytes:
    attrs = getattr(token, "attrs", None)
    if attrs is None:
        attrs_value: Any = []
    else:
        attrs_value = _json_safe(attrs, where="token.attrs")
    token_map = getattr(token, "map", None)
    children = []
    for child in (getattr(token, "children", None) or ()):
        child_attrs = getattr(child, "attrs", None)
        children.append({
            "attrs": ([] if child_attrs is None else _json_safe(
                child_attrs, where="token.children.attrs")),
            "block": bool(getattr(child, "block", False)),
            "children": [],
            "content": str(getattr(child, "content", "")),
            "hidden": bool(getattr(child, "hidden", False)),
            "info": str(getattr(child, "info", "")),
            "level": int(getattr(child, "level", 0)),
            "map": ([] if getattr(child, "map", None) is None else [
                int(item) for item in child.map]),
            "markup": str(getattr(child, "markup", "")),
            "meta": _json_safe(getattr(child, "meta", {}) or {},
                                where="token.children.meta"),
            "nesting": int(getattr(child, "nesting", 0)),
            "tag": str(getattr(child, "tag", "")),
            "type": str(getattr(child, "type", "")),
        })
    receipt = {
        "attrs": attrs_value,
        "block": bool(getattr(token, "block", False)),
        "children": children,
        "content": str(getattr(token, "content", "")),
        "hidden": bool(getattr(token, "hidden", False)),
        "info": str(getattr(token, "info", "")),
        "level": int(getattr(token, "level", 0)),
        "map": ([] if token_map is None else [int(item) for item in token_map]),
        "markup": str(getattr(token, "markup", "")),
        "meta": _json_safe(getattr(token, "meta", {}) or {}, where="token.meta"),
        "nesting": int(getattr(token, "nesting", 0)),
        "ordinal": ordinal,
        "path": list(path),
        "range": [span[0], span[1]],
        "tag": str(getattr(token, "tag", "")),
        "type": str(getattr(token, "type", "")),
    }
    return canonical_json_bytes(receipt)


def _receipt_with_ordinal(payload: bytes, ordinal: int) -> bytes:
    value = parse_canonical_json_bytes(payload, require_object=True)
    value["ordinal"] = ordinal
    return canonical_json_bytes(value)


@dataclass(frozen=True)
class _TokenMeta:
    ordinal: int
    path: tuple[int, ...]
    parent_ordinal: int | None
    span: tuple[int, int]
    receipt: bytes
    token_type: str
    nesting: int


def _token_metas(text: str) -> tuple[_TokenMeta, ...]:
    if getattr(markdown_it, "__version__", None) != PARSER_VERSION:
        raise MarkdownCarrierAdapterError(
            f"markdown-it-py 版本必须固定为 {PARSER_VERSION}")
    try:
        tokens = MarkdownIt("commonmark", {"html": True}).parse(text)
    except Exception as error:
        raise MarkdownCarrierAdapterError("Markdown parse 失败") from error

    starts = _line_starts(text)
    # stack entries are (open ordinal, path, parent ordinal, span, next child).
    stack: list[list[Any]] = []
    root_next = 0
    metas: list[_TokenMeta] = []
    top_level_tokens: list[Any] = []
    for ordinal, token in enumerate(tokens):
        nesting = int(getattr(token, "nesting", 0))
        if nesting < 0 and stack:
            opened = stack[-1]
            ancestors = stack[:-1]
            parent_ordinal = opened[2]
            if ancestors:
                parent_path = ancestors[-1][1]
                child = ancestors[-1][4]
                ancestors[-1][4] += 1
            else:
                parent_path = ()
                child = root_next
                root_next += 1
            path = (*parent_path, child)
            span = tuple(opened[3])
            stack.pop()
        else:
            if stack:
                parent_ordinal = stack[-1][0]
                parent_path = stack[-1][1]
                child = stack[-1][4]
                stack[-1][4] += 1
            else:
                parent_ordinal = None
                parent_path = ()
                child = root_next
                root_next += 1
            path = (*parent_path, child)
            fallback = tuple(stack[-1][3]) if stack else (0, len(text))
            span = _token_range(token, text, starts, fallback)
        receipt = _token_receipt(token, path, ordinal, span)
        metas.append(_TokenMeta(
            ordinal, path, parent_ordinal, span, receipt,
            str(getattr(token, "type", "")), nesting))
        top_level_tokens.append(token)
        if nesting > 0:
            stack.append([ordinal, path, parent_ordinal, span, 0])
    if stack:
        raise MarkdownCarrierAdapterError("Markdown token tree 未闭合")

    # markdown-it 将 emphasis/link 等 inline token 存在 ``inline.children``，
    # 而不是顶层解析列表；这里把 children 也物化为结构节点，避免
    # attrs/title 和顺序在父 token 的 render content 中丢失。
    def subtree_size(token: Any) -> int:
        children = getattr(token, "children", None) or ()
        return 1 + sum(subtree_size(child) for child in children)

    if not any(getattr(token, "children", None) for token in top_level_tokens):
        return tuple(metas)
    old_to_new: dict[int, int] = {}
    cursor = 0
    for meta, token in zip(metas, top_level_tokens):
        old_to_new[meta.ordinal] = cursor
        cursor += subtree_size(token)

    expanded: list[_TokenMeta] = []

    def emit(token: Any, path: tuple[int, ...], parent: int | None,
             span: tuple[int, int], old_meta: _TokenMeta | None = None) -> None:
        ordinal = len(expanded)
        actual_span = _token_range(token, text, starts, span)
        if old_meta is None:
            receipt = _token_receipt(token, path, ordinal, actual_span)
        else:
            receipt = _receipt_with_ordinal(old_meta.receipt, ordinal)
        current = _TokenMeta(
            ordinal, path, parent, actual_span, receipt,
            str(getattr(token, "type", "")),
            int(getattr(token, "nesting", 0)),
        )
        expanded.append(current)
        for child_index, child in enumerate(getattr(token, "children", None) or ()):
            emit(child, (*path, child_index), ordinal, actual_span)

    for meta, token in zip(metas, top_level_tokens):
        parent = (None if meta.parent_ordinal is None
                  else old_to_new[meta.parent_ordinal])
        emit(token, meta.path, parent, meta.span, meta)
    return tuple(expanded)


@dataclass(frozen=True)
class MarkdownCarrierMaterialization:
    """一个 MARKDOWN case 的完整 envelope、anchor、tree node 与 revision。"""

    record: MarkdownCarrierRecord
    sources: tuple[SourceRef, ...]
    scopes: tuple[ScopeIdentity, ...]
    envelopes: tuple[ArtifactEnvelope, ...]
    anchors: tuple[ArtifactAnchor, ...]
    structure_nodes: tuple[ArtifactStructureNode, ...]
    revisions: tuple[ArtifactCarrierRevision, ...]

    @property
    def text_anchors(self) -> tuple[ArtifactAnchor, ...]:
        """按 envelope 顺序返回完整原文范围锚。"""
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_TEXT_RANGE)

    @property
    def document_anchors(self) -> tuple[ArtifactAnchor, ...]:
        """按 envelope 顺序返回 document-region 锚。"""
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_DOCUMENT_REGION)

    @property
    def tree_anchors(self) -> tuple[ArtifactAnchor, ...]:
        """按 parser token 顺序返回 tree-path 锚。"""
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_TREE_PATH)

    def __post_init__(self) -> None:
        if not isinstance(self.record, MarkdownCarrierRecord):
            raise MarkdownCarrierAdapterError("materialization record 类型非法")
        expected_count = 2 if self.record.sample_kind == "REVISION" else 1
        for name, cls in (
                ("sources", SourceRef), ("scopes", ScopeIdentity),
                ("envelopes", ArtifactEnvelope), ("anchors", ArtifactAnchor),
                ("structure_nodes", ArtifactStructureNode)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or len(values) == 0
                    or any(not isinstance(item, cls) for item in values)):
                raise MarkdownCarrierAdapterError(f"materialization {name} 类型非法")
        if any(len(group) == 0 for group in (self.sources, self.scopes, self.envelopes)):
            raise MarkdownCarrierAdapterError("materialization 顶层对象不能为空")
        if len(self.sources) != expected_count or len(self.scopes) != expected_count:
            raise MarkdownCarrierAdapterError("materialization source/scope 数量漂移")
        if len(self.envelopes) != expected_count:
            raise MarkdownCarrierAdapterError("materialization envelope 数量漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise MarkdownCarrierAdapterError("materialization revisions 数量漂移")

        expected_texts = ((self.record.previous_text, self.record.raw_text)
                          if expected_count == 2 else (self.record.raw_text,))
        for index, (source, scope, envelope, text) in enumerate(zip(
                self.sources, self.scopes, self.envelopes, expected_texts)):
            parser = _parser(source, self.record)
            if scope != document_scope(source):
                raise MarkdownCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise MarkdownCarrierAdapterError("envelope raw/source 漂移")
            group = self._anchors_for_envelope(envelope.identity)
            if len(group) < 3:
                raise MarkdownCarrierAdapterError("MARKDOWN 缺少 text/document/tree anchors")
            if not any(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group):
                raise MarkdownCarrierAdapterError("MARKDOWN 缺少 full text anchor")
            if not any(item.anchor_kind == ANCHOR_DOCUMENT_REGION for item in group):
                raise MarkdownCarrierAdapterError("MARKDOWN 缺少 document region anchor")
            if any(item.source != source or item.scope != scope
                   or item.envelope_identity != envelope.identity
                   or item.parser != parser for item in group):
                raise MarkdownCarrierAdapterError("anchor context 漂移")
            nodes = tuple(item for item in self.structure_nodes
                          if item.envelope_identity == envelope.identity)
            if not nodes:
                raise MarkdownCarrierAdapterError("MARKDOWN 缺少 structure nodes")
            node_ids = {item.identity for item in nodes}
            anchor_ids = {item.identity for item in group
                          if item.anchor_kind == ANCHOR_TREE_PATH}
            if any(item.anchor_identity not in anchor_ids for item in nodes):
                raise MarkdownCarrierAdapterError("structure node 未绑定 tree anchor")
            if any(item.parent_identity is not None
                   and item.parent_identity not in node_ids for item in nodes):
                raise MarkdownCarrierAdapterError("structure node parent 逃逸")
            if tuple(item.ordinal for item in nodes) != tuple(
                    sorted(item.ordinal for item in nodes)):
                raise MarkdownCarrierAdapterError("structure node ordinal 漂移")
            for artifact in (envelope, *group, *nodes):
                try:
                    if type(artifact).from_stable_key(artifact.stable_key()) != artifact:
                        raise MarkdownCarrierAdapterError("对象无法稳定回读")
                except MarkdownCarrierAdapterError:
                    raise
                except Exception as error:
                    raise MarkdownCarrierAdapterError("对象无法稳定回读") from error

        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise MarkdownCarrierAdapterError("revision parser lineage 漂移")
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source):
                raise MarkdownCarrierAdapterError("revision envelope/hypothesis 漂移")

    def _anchors_for_envelope(self, identity: Any) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.envelope_identity == identity)


def _build_envelope(record: MarkdownCarrierRecord, source: SourceRef,
                    text: str, variant: int, *, ordinal_offset: int = 0) -> tuple[ArtifactEnvelope, tuple[ArtifactAnchor, ...], tuple[ArtifactStructureNode, ...]]:
    scope = document_scope(source)
    parser = _parser(source, record)
    envelope = make_artifact_envelope(
        source=source,
        scope=scope,
        carrier_family=structure_concept_identity(
            _key(record, 10), owner=source.owner, versions=source.versions),
        raw_unit_kind=RAW_UNIT_UNICODE_SCALAR,
        raw_units=tuple(ord(item) for item in text),
        media_profile=_concept(source, record, 11),
        language_branch=language_branch_identity(
            _key(record, 12), owner=source.owner, versions=source.versions),
        parser=parser,
        renderer=_renderer(source, record),
        envelope_key=_key(record, 30, variant),
    )
    metas = _token_metas(text)
    text_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TEXT_RANGE, coordinates=(0, len(text)),
        parser=parser, linked_text_anchor=None,
        anchor_key=_key(record, 31, variant, 1),
    )
    document_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_DOCUMENT_REGION, coordinates=(0, len(text)),
        parser=parser, linked_text_anchor=None,
        anchor_key=_key(record, 31, variant, 2),
    )
    anchors = [text_anchor, document_anchor]
    tree_anchors: dict[int, ArtifactAnchor] = {}
    for meta in metas:
        anchor = make_artifact_anchor(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_kind=ANCHOR_TREE_PATH, coordinates=meta.path,
            parser=parser, linked_text_anchor=None,
            anchor_key=_key(record, 31, variant, 3, meta.ordinal, *meta.path),
        )
        anchors.append(anchor)
        tree_anchors[meta.ordinal] = anchor

    family = structure_concept_identity(
        _key(record, 60), owner=source.owner, versions=source.versions)
    nodes: list[ArtifactStructureNode] = []
    for meta in metas:
        token_type_key = tuple(meta.token_type.encode("utf-8")) or (0,)
        node_kind = structure_concept_identity(
            _key(record, 61, meta.nesting + 2, *token_type_key),
            owner=source.owner, versions=source.versions)
        parent = (None if meta.parent_ordinal is None else
                  nodes[meta.parent_ordinal].identity)
        nodes.append(make_artifact_structure_node(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_identity=tree_anchors[meta.ordinal].identity,
            structure_family=family, node_kind=node_kind, role=None,
            parent_identity=parent, ordinal=ordinal_offset + meta.ordinal,
            qualifiers=tuple(meta.receipt),
            node_key=_key(record, 62, variant, meta.ordinal, *meta.path),
        ))
    return envelope, tuple(anchors), tuple(nodes)


def _revision(record: MarkdownCarrierRecord, old: tuple[ArtifactAnchor, ...],
              new: tuple[ArtifactAnchor, ...], old_nodes: tuple[ArtifactStructureNode, ...],
              new_nodes: tuple[ArtifactStructureNode, ...], old_envelope: ArtifactEnvelope,
              new_envelope: ArtifactEnvelope, new_source: SourceRef,
              new_scope: ScopeIdentity) -> ArtifactCarrierRevision:
    old_tree = {item.coordinates: item for item in old if item.anchor_kind == ANCHOR_TREE_PATH}
    new_tree = {item.coordinates: item for item in new if item.anchor_kind == ANCHOR_TREE_PATH}
    mappings: list[ArtifactRevisionMapping] = []
    for item in old:
        if item.anchor_kind == ANCHOR_TEXT_RANGE:
            targets = tuple(x.identity for x in new if x.anchor_kind == ANCHOR_TEXT_RANGE)
        elif item.anchor_kind == ANCHOR_DOCUMENT_REGION:
            targets = tuple(x.identity for x in new if x.anchor_kind == ANCHOR_DOCUMENT_REGION)
        else:
            target = new_tree.get(item.coordinates)
            targets = () if target is None else (target.identity,)
        mappings.append(ArtifactRevisionMapping(REVISION_MAP_ANCHOR, item.identity, targets))
    def node_signature(node: ArtifactStructureNode) -> tuple[Any, ...]:
        try:
            receipt = parse_canonical_json_bytes(
                bytes(node.qualifiers), require_object=True)
            return (
                tuple(receipt["path"]), str(receipt["type"]),
                int(receipt["nesting"]),
            )
        except Exception as error:
            raise MarkdownCarrierAdapterError(
                "structure node token receipt 损坏") from error

    new_by_signature = {node_signature(node): node for node in new_nodes}
    for node in old_nodes:
        # path 加 token kind 构成局部结构身份；内容和来源范围可以变化，
        # 找不到签名时显式记录为删除映射。
        target = new_by_signature.get(node_signature(node))
        mappings.append(ArtifactRevisionMapping(
            REVISION_MAP_STRUCTURE_NODE, node.identity,
            () if target is None else (target.identity,),
        ))
    hypothesis = HypothesisKey(
        _key(record, 70), _key(record, 71), _key(record, 72), new_scope, new_source)
    evidence = EvidenceRecord(
        record.case_key.stable_key()[-1], hypothesis, EVIDENCE_SUPPORT,
        _key(record, 73), new_source, 1, _key(record, 74),
    )
    return make_artifact_carrier_revision(
        old_envelope_identity=old_envelope.identity,
        new_envelope_identity=new_envelope.identity,
        reason=_concept(new_source, record, 75), hypothesis=hypothesis,
        mappings=tuple(sorted(mappings, key=ArtifactRevisionMapping.stable_key)),
        evidence=(evidence,), revision_key=_key(record, 76),
    )


def adapt_markdown_carrier_record(record: MarkdownCarrierRecord) -> MarkdownCarrierMaterialization:
    """不训练、不选义地把一个冻结 Markdown payload 物化为 carrier 对象。"""
    if not isinstance(record, MarkdownCarrierRecord):
        raise MarkdownCarrierAdapterError("adapter 只接受 MarkdownCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope = document_scope(source)
        envelope, anchors, nodes = _build_envelope(record, source, record.raw_text, 1)
        return MarkdownCarrierMaterialization(
            record, (source,), (scope,), (envelope,), anchors, nodes, ())
    old_source = _source(record, 1)
    new_source = _source(record, 2)
    old_scope = document_scope(old_source)
    new_scope = document_scope(new_source)
    old_envelope, old_anchors, old_nodes = _build_envelope(
        record, old_source, record.previous_text, 1)
    new_envelope, new_anchors, new_nodes = _build_envelope(
        record, new_source, record.raw_text, 2,
        ordinal_offset=len(old_nodes))
    revision = _revision(
        record, old_anchors, new_anchors, old_nodes, new_nodes,
        old_envelope, new_envelope, new_source, new_scope)
    return MarkdownCarrierMaterialization(
        record, (old_source, new_source), (old_scope, new_scope),
        (old_envelope, new_envelope), (*old_anchors, *new_anchors),
        (*old_nodes, *new_nodes), (revision,),
    )


def _stable_lists(values: tuple[Any, ...]) -> list[list[int]]:
    return [list(item.stable_key()) for item in values]


def serialize_markdown_carrier_materialization(
        materialization: MarkdownCarrierMaterialization) -> bytes:
    """以 canonical JSON 保存全部共享对象的完整 stable keys。"""
    if not isinstance(materialization, MarkdownCarrierMaterialization):
        raise MarkdownCarrierAdapterError("serializer 输入类型非法")
    value = {
        "anchors": _stable_lists(materialization.anchors),
        "artifact_kind": MATERIALIZATION_KIND,
        "case_key": materialization.record.case_key.to_list(),
        "envelopes": _stable_lists(materialization.envelopes),
        "format_version": MATERIALIZATION_FORMAT_VERSION,
        "revisions": _stable_lists(materialization.revisions),
        "sample_kind": materialization.record.sample_kind,
        "scopes": _stable_lists(materialization.scopes),
        "sources": _stable_lists(materialization.sources),
        "structure_nodes": _stable_lists(materialization.structure_nodes),
    }
    return canonical_json_bytes(value) + b"\n"


def _strict_stable_keys(value: Any, *, where: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise MarkdownCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result: list[tuple[int, ...]] = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise MarkdownCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_markdown_carrier_materialization(
        payload: bytes, record: MarkdownCarrierRecord) -> MarkdownCarrierMaterialization:
    """严格回读 canonical bytes，并对照冻结 payload 重验对象图。"""
    if not isinstance(record, MarkdownCarrierRecord):
        raise MarkdownCarrierAdapterError("deserializer record 类型非法")
    try:
        if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
                or payload.endswith(b"\n\n")):
            raise MarkdownCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        expected = {
            "anchors", "artifact_kind", "case_key", "envelopes",
            "format_version", "revisions", "sample_kind", "scopes",
            "sources", "structure_nodes",
        }
        if set(value) != expected:
            raise MarkdownCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise MarkdownCarrierAdapterError("materialization record 身份漂移")
        sources = tuple(SourceRef.from_stable_key(item) for item in
                        _strict_stable_keys(value["sources"], where="sources"))
        scopes = tuple(ScopeIdentity.from_stable_key(item) for item in
                       _strict_stable_keys(value["scopes"], where="scopes"))
        envelopes = tuple(ArtifactEnvelope.from_stable_key(item) for item in
                          _strict_stable_keys(value["envelopes"], where="envelopes"))
        anchors = tuple(ArtifactAnchor.from_stable_key(item) for item in
                        _strict_stable_keys(value["anchors"], where="anchors"))
        nodes = tuple(ArtifactStructureNode.from_stable_key(item) for item in
                      _strict_stable_keys(value["structure_nodes"], where="structure_nodes"))
        revisions = tuple(ArtifactCarrierRevision.from_stable_key(item) for item in
                          _strict_stable_keys(value["revisions"], where="revisions"))
        result = MarkdownCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, nodes, revisions)
    except MarkdownCarrierAdapterError:
        raise
    except Exception as error:
        raise MarkdownCarrierAdapterError("materialization 损坏") from error
    if serialize_markdown_carrier_materialization(result) != payload:
        raise MarkdownCarrierAdapterError("materialization 不是 canonical 表示")
    return result


# 为兼容 LC-16 catalog/test 接口保留短名称。
serialize_markdown_materialization = serialize_markdown_carrier_materialization
deserialize_markdown_materialization = deserialize_markdown_carrier_materialization


__all__ = [
    "MARKDOWN_SOURCE_KIND",
    "MATERIALIZATION_FORMAT_VERSION",
    "MATERIALIZATION_KIND",
    "MarkdownCarrierAdapterError",
    "MarkdownCarrierMaterialization",
    "adapt_markdown_carrier_record",
    "deserialize_markdown_materialization",
    "deserialize_markdown_carrier_materialization",
    "serialize_markdown_materialization",
    "serialize_markdown_carrier_materialization",
]
