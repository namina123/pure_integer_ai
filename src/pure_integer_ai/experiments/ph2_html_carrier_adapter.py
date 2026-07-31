"""LC-16 HTML payload 的确定性 raw + DOM adapter。

raw HTML 始终以 Unicode scalar 原样保存在 ``ArtifactEnvelope`` 中；DOM
只是同一观测的结构视图，不替代、规范化或重写原文。解析器不访问网络，
引用只物化为显式 resolved/unresolved/access-blocked 状态。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

try:
    import lxml
    from lxml import etree
except ImportError as error:  # pragma: no cover - dependency contract
    raise RuntimeError("HTML adapter 需要 lxml") from error

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TREE_PATH,
    RAW_UNIT_UNICODE_SCALAR,
    REFERENCE_ACCESS_BLOCKED,
    REFERENCE_RESOLVED,
    REFERENCE_UNRESOLVED,
    REVISION_MAP_ANCHOR,
    REVISION_MAP_REFERENCE,
    REVISION_MAP_STRUCTURE_NODE,
    ArtifactAnchor,
    ArtifactCarrierRevision,
    ArtifactEnvelope,
    ArtifactReferenceBinding,
    ArtifactRevisionMapping,
    ArtifactStructureNode,
    make_artifact_anchor,
    make_artifact_carrier_revision,
    make_artifact_envelope,
    make_artifact_reference_binding,
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
from pure_integer_ai.experiments.ph2_html_carrier_contract import (
    HtmlCarrierRecord,
    PARSER_VERSION,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_HTML_MATERIALIZATION"
HTML_SOURCE_KIND = 16616503
_IDENTITY_BASE = 16616620
_REFERENCE_ATTRIBUTES = frozenset({
    "action", "cite", "formaction", "href", "longdesc", "manifest",
    "poster", "src", "usemap",
})


class HtmlCarrierAdapterError(RuntimeError):
    """HTML adapter 输入、DOM 对象或 canonical 表示不闭合。"""


def _key(record: HtmlCarrierRecord, domain: int, *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: HtmlCarrierRecord, parser_version: int) -> SourceRef:
    values = record.case_key.stable_key()
    case_index = values[-1]
    owner = OwnerScope(values[0], values[-2], case_index, VISIBILITY_SESSION)
    versions = VersionBundle(
        CorpusVersion(1), ParserVersion(parser_version), PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        HTML_SOURCE_KIND,
        values[0] + values[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: HtmlCarrierRecord, domain: int,
             *tail: int):
    return concept_identity(
        _key(record, domain, *tail), owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, record: HtmlCarrierRecord, domain: int,
               *tail: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain, *tail),
        _concept(source, record, domain + 1, *tail),
    )


def _parser(source: SourceRef, record: HtmlCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 20)


def _renderer(source: SourceRef, record: HtmlCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 22)


def _utf8_key(value: str) -> tuple[int, ...]:
    return tuple(value.encode("utf-8")) or (0,)


def _local_name(value: str) -> str:
    if value.startswith("{") and "}" in value:
        return value.split("}", 1)[1].lower()
    return value.lower()


@dataclass(frozen=True)
class _DomMeta:
    ordinal: int
    path: tuple[int, ...]
    parent_ordinal: int | None
    node_type: str
    name: str
    text_role: str
    receipt: bytes


@dataclass(frozen=True)
class _ReferenceMeta:
    owner_ordinal: int
    attribute_ordinal: int
    attribute_name: str
    target_text: str


def _dom_receipt(
        *,
        node_type: str,
        name: str,
        text_role: str,
        content: str,
        attributes: list[dict[str, str]],
        path: tuple[int, ...],
        parent_path: tuple[int, ...] | None,
        source_line: int,
        source_order: int,
        render_order: int,
        ) -> bytes:
    return canonical_json_bytes({
        "attributes": attributes,
        "content": content,
        "name": name,
        "node_type": node_type,
        "parent_path": [] if parent_path is None else list(parent_path),
        "path": list(path),
        "render_order": render_order,
        "source_line": source_line,
        "source_order": source_order,
        "text_role": text_role,
    })


def _parse_dom(text: str) -> tuple[tuple[_DomMeta, ...], tuple[_ReferenceMeta, ...]]:
    if getattr(lxml, "__version__", None) != PARSER_VERSION:
        raise HtmlCarrierAdapterError(f"lxml 版本必须固定为 {PARSER_VERSION}")
    try:
        parser = etree.HTMLParser(
            encoding="utf-8",
            recover=True,
            remove_comments=False,
            no_network=True,
        )
        root = etree.fromstring(text, parser=parser)
    except Exception as error:
        raise HtmlCarrierAdapterError("HTML parse 失败") from error
    if root is None:
        raise HtmlCarrierAdapterError("HTML parse 未产生 DOM root")

    metas: list[_DomMeta] = []
    references: list[_ReferenceMeta] = []

    def emit_text(
            content: str,
            path: tuple[int, ...],
            parent_ordinal: int,
            role: str,
            source_line: int,
            ) -> None:
        ordinal = len(metas)
        parent_path = metas[parent_ordinal].path
        metas.append(_DomMeta(
            ordinal,
            path,
            parent_ordinal,
            "text",
            "#text",
            role,
            _dom_receipt(
                node_type="text",
                name="#text",
                text_role=role,
                content=content,
                attributes=[],
                path=path,
                parent_path=parent_path,
                source_line=max(0, source_line),
                source_order=ordinal,
                render_order=ordinal,
            ),
        ))

    def emit_node(
            node: Any,
            path: tuple[int, ...],
            parent_ordinal: int | None,
            ) -> None:
        ordinal = len(metas)
        parent_path = None if parent_ordinal is None else metas[parent_ordinal].path
        source_line = int(getattr(node, "sourceline", 0) or 0)
        if isinstance(node, etree._Comment):
            metas.append(_DomMeta(
                ordinal,
                path,
                parent_ordinal,
                "comment",
                "#comment",
                "",
                _dom_receipt(
                    node_type="comment",
                    name="#comment",
                    text_role="",
                    content=str(node.text or ""),
                    attributes=[],
                    path=path,
                    parent_path=parent_path,
                    source_line=max(0, source_line),
                    source_order=ordinal,
                    render_order=ordinal,
                ),
            ))
            return
        if not isinstance(getattr(node, "tag", None), str):
            raise HtmlCarrierAdapterError("DOM 含未登记节点类型")
        tag = str(node.tag)
        attributes = [
            {"name": str(name), "value": str(value)}
            for name, value in node.attrib.items()
        ]
        metas.append(_DomMeta(
            ordinal,
            path,
            parent_ordinal,
            "element",
            tag,
            "",
            _dom_receipt(
                node_type="element",
                name=tag,
                text_role="",
                content="",
                attributes=attributes,
                path=path,
                parent_path=parent_path,
                source_line=max(0, source_line),
                source_order=ordinal,
                render_order=ordinal,
            ),
        ))
        for attribute_ordinal, (name, value) in enumerate(node.attrib.items()):
            attribute_name = str(name)
            if _local_name(attribute_name) in _REFERENCE_ATTRIBUTES:
                references.append(_ReferenceMeta(
                    ordinal,
                    attribute_ordinal,
                    attribute_name,
                    str(value),
                ))

        child_slot = 0
        if node.text is not None:
            emit_text(str(node.text), (*path, child_slot), ordinal, "text",
                      source_line)
            child_slot += 1
        for child in node:
            emit_node(child, (*path, child_slot), ordinal)
            child_slot += 1
            if child.tail is not None:
                emit_text(
                    str(child.tail),
                    (*path, child_slot),
                    ordinal,
                    "tail",
                    int(getattr(child, "sourceline", source_line) or 0),
                )
                child_slot += 1

    emit_node(root, (0,), None)
    if not metas:
        raise HtmlCarrierAdapterError("DOM 结构节点为空")
    return tuple(metas), tuple(references)


@dataclass(frozen=True)
class HtmlCarrierMaterialization:
    """一个 HTML case 的 raw、DOM、引用与可选 revision 完整物化。"""

    record: HtmlCarrierRecord
    sources: tuple[SourceRef, ...]
    scopes: tuple[ScopeIdentity, ...]
    envelopes: tuple[ArtifactEnvelope, ...]
    anchors: tuple[ArtifactAnchor, ...]
    structure_nodes: tuple[ArtifactStructureNode, ...]
    references: tuple[ArtifactReferenceBinding, ...]
    revisions: tuple[ArtifactCarrierRevision, ...]

    @property
    def text_anchors(self) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_TEXT_RANGE)

    @property
    def document_anchors(self) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_DOCUMENT_REGION)

    @property
    def tree_anchors(self) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_TREE_PATH)

    @property
    def reference_anchors(self) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_REFERENCE_SLOT)

    @property
    def reference_bindings(self) -> tuple[ArtifactReferenceBinding, ...]:
        """共享合同名称下的显式引用绑定视图。"""
        return self.references

    def __post_init__(self) -> None:
        if not isinstance(self.record, HtmlCarrierRecord):
            raise HtmlCarrierAdapterError("materialization record 类型非法")
        expected_count = 2 if self.record.sample_kind == "REVISION" else 1
        for name, cls, allow_empty in (
                ("sources", SourceRef, False),
                ("scopes", ScopeIdentity, False),
                ("envelopes", ArtifactEnvelope, False),
                ("anchors", ArtifactAnchor, False),
                ("structure_nodes", ArtifactStructureNode, False),
                ("references", ArtifactReferenceBinding, True)):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or (not allow_empty and not values)
                    or any(not isinstance(item, cls) for item in values)):
                raise HtmlCarrierAdapterError(f"materialization {name} 类型非法")
        if (len(self.sources) != expected_count
                or len(self.scopes) != expected_count
                or len(self.envelopes) != expected_count):
            raise HtmlCarrierAdapterError("materialization 顶层对象数量漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise HtmlCarrierAdapterError("materialization revisions 数量漂移")

        expected_texts = ((self.record.previous_text, self.record.raw_text)
                          if expected_count == 2 else (self.record.raw_text,))
        for source, scope, envelope, text in zip(
                self.sources, self.scopes, self.envelopes, expected_texts):
            parser = _parser(source, self.record)
            if scope != document_scope(source):
                raise HtmlCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise HtmlCarrierAdapterError("envelope raw/source 漂移")
            group = self._anchors_for_envelope(envelope.identity)
            if (not any(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group)
                    or not any(item.anchor_kind == ANCHOR_DOCUMENT_REGION
                               for item in group)
                    or not any(item.anchor_kind == ANCHOR_TREE_PATH
                               for item in group)):
                raise HtmlCarrierAdapterError("HTML 缺少 text/document/tree anchor")
            if any(item.source != source or item.scope != scope
                   or item.envelope_identity != envelope.identity
                   or item.parser != parser for item in group):
                raise HtmlCarrierAdapterError("anchor context 漂移")
            nodes = tuple(item for item in self.structure_nodes
                          if item.envelope_identity == envelope.identity)
            if not nodes:
                raise HtmlCarrierAdapterError("HTML 缺少 DOM structure nodes")
            node_ids = {item.identity for item in nodes}
            tree_anchor_ids = {item.identity for item in group
                               if item.anchor_kind == ANCHOR_TREE_PATH}
            if any(item.anchor_identity not in tree_anchor_ids for item in nodes):
                raise HtmlCarrierAdapterError("DOM node 未绑定 tree anchor")
            if any(item.parent_identity is not None
                   and item.parent_identity not in node_ids for item in nodes):
                raise HtmlCarrierAdapterError("DOM node parent 逃逸")
            if tuple(item.ordinal for item in nodes) != tuple(
                    sorted(item.ordinal for item in nodes)):
                raise HtmlCarrierAdapterError("DOM node ordinal 漂移")
            refs = tuple(item for item in self.references
                         if item.envelope_identity == envelope.identity)
            reference_anchor_ids = {item.identity for item in group
                                    if item.anchor_kind == ANCHOR_REFERENCE_SLOT}
            if any(item.source != source or item.scope != scope
                   or item.anchor_identity not in reference_anchor_ids
                   for item in refs):
                raise HtmlCarrierAdapterError("reference context 漂移")
            local_tree_ids = tree_anchor_ids
            if any(item.target_state == REFERENCE_RESOLVED
                   and (item.target_source != source
                        or item.target_anchor not in local_tree_ids)
                   for item in refs):
                raise HtmlCarrierAdapterError("resolved fragment 未绑定片内 DOM")
            for artifact in (envelope, *group, *nodes, *refs):
                try:
                    if type(artifact).from_stable_key(artifact.stable_key()) != artifact:
                        raise HtmlCarrierAdapterError("对象无法稳定回读")
                except HtmlCarrierAdapterError:
                    raise
                except Exception as error:
                    raise HtmlCarrierAdapterError("对象无法稳定回读") from error

        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise HtmlCarrierAdapterError("revision parser lineage 漂移")
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source):
                raise HtmlCarrierAdapterError("revision identity 漂移")

    def _anchors_for_envelope(self, identity: Any) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.envelope_identity == identity)


def _reference_state(
        target: str,
        source: SourceRef,
        id_targets: dict[str, tuple[ArtifactAnchor, ...]],
        ) -> tuple[int, SourceRef | None, Any | None, tuple[int, ...]]:
    if not target:
        return REFERENCE_UNRESOLVED, None, None, ()
    try:
        split = urlsplit(target)
    except ValueError:
        split = None
    if (split is not None and target.startswith("#") and split.fragment
            and not split.scheme and not split.netloc
            and not split.path and not split.query):
        matches = id_targets.get(split.fragment, ())
        if len(matches) == 1:
            return (
                REFERENCE_RESOLVED,
                source,
                matches[0].identity,
                tuple(target.encode("utf-8")),
            )
        return REFERENCE_UNRESOLVED, None, None, ()
    return (
        REFERENCE_ACCESS_BLOCKED,
        None,
        None,
        tuple(target.encode("utf-8")),
    )


def _build_envelope(
        record: HtmlCarrierRecord,
        source: SourceRef,
        text: str,
        variant: int,
        *,
        ordinal_offset: int = 0,
        ) -> tuple[
            ArtifactEnvelope,
            tuple[ArtifactAnchor, ...],
            tuple[ArtifactStructureNode, ...],
            tuple[ArtifactReferenceBinding, ...],
            ]:
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
    metas, reference_metas = _parse_dom(text)
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
        kind_key = _utf8_key(f"{meta.node_type}:{meta.name}:{meta.text_role}")
        node_kind = structure_concept_identity(
            _key(record, 61, *kind_key),
            owner=source.owner,
            versions=source.versions,
        )
        parent = (None if meta.parent_ordinal is None
                  else nodes[meta.parent_ordinal].identity)
        nodes.append(make_artifact_structure_node(
            envelope_identity=envelope.identity,
            source=source,
            scope=scope,
            anchor_identity=tree_anchors[meta.ordinal].identity,
            structure_family=family,
            node_kind=node_kind,
            role=None,
            parent_identity=parent,
            ordinal=ordinal_offset + meta.ordinal,
            qualifiers=tuple(meta.receipt),
            node_key=_key(record, 62, variant, meta.ordinal, *meta.path),
        ))

    id_targets_lists: dict[str, list[ArtifactAnchor]] = {}
    for meta in metas:
        if meta.node_type != "element":
            continue
        receipt = parse_canonical_json_bytes(meta.receipt, require_object=True)
        for attribute in receipt["attributes"]:
            if _local_name(str(attribute["name"])) == "id":
                id_targets_lists.setdefault(
                    str(attribute["value"]), []).append(tree_anchors[meta.ordinal])
    id_targets = {key: tuple(value) for key, value in id_targets_lists.items()}

    references: list[ArtifactReferenceBinding] = []
    for reference_ordinal, meta in enumerate(reference_metas):
        owner = metas[meta.owner_ordinal]
        reference_anchor = make_artifact_anchor(
            envelope_identity=envelope.identity,
            source=source,
            scope=scope,
            anchor_kind=ANCHOR_REFERENCE_SLOT,
            coordinates=(*owner.path, meta.attribute_ordinal),
            parser=parser,
            linked_text_anchor=None,
            anchor_key=_key(
                record, 31, variant, 4, reference_ordinal,
                *owner.path, meta.attribute_ordinal),
        )
        anchors.append(reference_anchor)
        state, target_source, target_anchor, fingerprint = _reference_state(
            meta.target_text, source, id_targets)
        references.append(make_artifact_reference_binding(
            envelope_identity=envelope.identity,
            source=source,
            scope=scope,
            anchor_identity=reference_anchor.identity,
            relation=_concept(
                source, record, 80, *_utf8_key(_local_name(meta.attribute_name))),
            target_state=state,
            target_source=target_source,
            target_anchor=target_anchor,
            target_fingerprint=fingerprint,
            reference_key=_key(
                record, 81, variant, reference_ordinal,
                *owner.path, meta.attribute_ordinal),
        ))
    return envelope, tuple(anchors), tuple(nodes), tuple(references)


def _node_signature(node: ArtifactStructureNode) -> tuple[Any, ...]:
    try:
        receipt = parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        return (
            tuple(receipt["path"]),
            str(receipt["node_type"]),
            str(receipt["name"]),
            str(receipt["text_role"]),
        )
    except Exception as error:
        raise HtmlCarrierAdapterError("DOM node receipt 损坏") from error


def _reference_signatures(
        anchors: tuple[ArtifactAnchor, ...],
        nodes: tuple[ArtifactStructureNode, ...],
        references: tuple[ArtifactReferenceBinding, ...],
        ) -> dict[Any, tuple[tuple[int, ...], str]]:
    """返回不受属性重排影响的引用修订签名。"""
    anchors_by_identity = {item.identity: item for item in anchors}
    receipts_by_path: dict[tuple[int, ...], dict[str, Any]] = {}
    try:
        for node in nodes:
            receipt = parse_canonical_json_bytes(
                bytes(node.qualifiers), require_object=True)
            receipts_by_path[tuple(receipt["path"])] = receipt
        result = {}
        for reference in references:
            anchor = anchors_by_identity[reference.anchor_identity]
            if (anchor.anchor_kind != ANCHOR_REFERENCE_SLOT
                    or len(anchor.coordinates) < 2):
                raise HtmlCarrierAdapterError(
                    "reference anchor 坐标非法")
            owner_path = tuple(anchor.coordinates[:-1])
            attribute_ordinal = anchor.coordinates[-1]
            attributes = receipts_by_path[owner_path]["attributes"]
            if (type(attribute_ordinal) is not int
                    or not 0 <= attribute_ordinal < len(attributes)):
                raise HtmlCarrierAdapterError(
                    "reference attribute ordinal 非法")
            attribute_name = _local_name(
                str(attributes[attribute_ordinal]["name"]))
            result[reference.identity] = (owner_path, attribute_name)
        return result
    except HtmlCarrierAdapterError:
        raise
    except Exception as error:
        raise HtmlCarrierAdapterError(
            "reference revision signature 损坏") from error


def _revision(
        record: HtmlCarrierRecord,
        old_anchors: tuple[ArtifactAnchor, ...],
        new_anchors: tuple[ArtifactAnchor, ...],
        old_nodes: tuple[ArtifactStructureNode, ...],
        new_nodes: tuple[ArtifactStructureNode, ...],
        old_references: tuple[ArtifactReferenceBinding, ...],
        new_references: tuple[ArtifactReferenceBinding, ...],
        old_envelope: ArtifactEnvelope,
        new_envelope: ArtifactEnvelope,
        new_source: SourceRef,
        new_scope: ScopeIdentity,
        ) -> ArtifactCarrierRevision:
    new_nodes_by_signature = {_node_signature(item): item for item in new_nodes}
    old_node_by_anchor = {item.anchor_identity: item for item in old_nodes}
    new_anchor_by_identity = {item.identity: item for item in new_anchors}
    old_reference_signatures = _reference_signatures(
        old_anchors, old_nodes, old_references)
    new_reference_signatures = _reference_signatures(
        new_anchors, new_nodes, new_references)
    new_reference_by_signature = {
        new_reference_signatures[item.identity]: item
        for item in new_references
    }
    old_reference_by_anchor = {
        item.anchor_identity: item for item in old_references
    }
    mappings: list[ArtifactRevisionMapping] = []
    for anchor in old_anchors:
        targets: tuple[Any, ...]
        if anchor.anchor_kind == ANCHOR_TEXT_RANGE:
            targets = tuple(item.identity for item in new_anchors
                            if item.anchor_kind == ANCHOR_TEXT_RANGE)
        elif anchor.anchor_kind == ANCHOR_DOCUMENT_REGION:
            targets = tuple(item.identity for item in new_anchors
                            if item.anchor_kind == ANCHOR_DOCUMENT_REGION)
        elif anchor.anchor_kind == ANCHOR_REFERENCE_SLOT:
            old_reference = old_reference_by_anchor.get(anchor.identity)
            signature = (None if old_reference is None else
                         old_reference_signatures[old_reference.identity])
            target = (None if signature is None else
                      new_reference_by_signature.get(signature))
            targets = () if target is None else (target.anchor_identity,)
        else:
            old_node = old_node_by_anchor.get(anchor.identity)
            target_node = (None if old_node is None else
                           new_nodes_by_signature.get(_node_signature(old_node)))
            target_anchor = (None if target_node is None else
                             new_anchor_by_identity.get(target_node.anchor_identity))
            targets = () if target_anchor is None else (target_anchor.identity,)
        mappings.append(ArtifactRevisionMapping(
            REVISION_MAP_ANCHOR, anchor.identity, targets))
    for node in old_nodes:
        target = new_nodes_by_signature.get(_node_signature(node))
        mappings.append(ArtifactRevisionMapping(
            REVISION_MAP_STRUCTURE_NODE,
            node.identity,
            () if target is None else (target.identity,),
        ))
    for reference in old_references:
        target = new_reference_by_signature.get(
            old_reference_signatures[reference.identity])
        mappings.append(ArtifactRevisionMapping(
            REVISION_MAP_REFERENCE,
            reference.identity,
            () if target is None else (target.identity,),
        ))
    hypothesis = HypothesisKey(
        _key(record, 70), _key(record, 71), _key(record, 72),
        new_scope, new_source)
    evidence = EvidenceRecord(
        record.case_key.stable_key()[-1], hypothesis, EVIDENCE_SUPPORT,
        _key(record, 73), new_source, 1, _key(record, 74),
    )
    return make_artifact_carrier_revision(
        old_envelope_identity=old_envelope.identity,
        new_envelope_identity=new_envelope.identity,
        reason=_concept(new_source, record, 75),
        hypothesis=hypothesis,
        mappings=tuple(sorted(mappings, key=ArtifactRevisionMapping.stable_key)),
        evidence=(evidence,),
        revision_key=_key(record, 76),
    )


def adapt_html_carrier_record(record: HtmlCarrierRecord) -> HtmlCarrierMaterialization:
    """不训练、不选义地把一个冻结 HTML payload 物化为共享 carrier 对象。"""
    if not isinstance(record, HtmlCarrierRecord):
        raise HtmlCarrierAdapterError("adapter 只接受 HtmlCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope = document_scope(source)
        envelope, anchors, nodes, references = _build_envelope(
            record, source, record.raw_text, 1)
        return HtmlCarrierMaterialization(
            record, (source,), (scope,), (envelope,), anchors, nodes,
            references, ())
    old_source = _source(record, 1)
    new_source = _source(record, 2)
    old_scope = document_scope(old_source)
    new_scope = document_scope(new_source)
    old_envelope, old_anchors, old_nodes, old_references = _build_envelope(
        record, old_source, record.previous_text, 1)
    new_envelope, new_anchors, new_nodes, new_references = _build_envelope(
        record, new_source, record.raw_text, 2,
        ordinal_offset=len(old_nodes))
    revision = _revision(
        record, old_anchors, new_anchors, old_nodes, new_nodes,
        old_references, new_references, old_envelope, new_envelope,
        new_source, new_scope)
    return HtmlCarrierMaterialization(
        record,
        (old_source, new_source),
        (old_scope, new_scope),
        (old_envelope, new_envelope),
        (*old_anchors, *new_anchors),
        (*old_nodes, *new_nodes),
        (*old_references, *new_references),
        (revision,),
    )


def _stable_lists(values: tuple[Any, ...]) -> list[list[int]]:
    return [list(item.stable_key()) for item in values]


def serialize_html_materialization(
        materialization: HtmlCarrierMaterialization,
        ) -> bytes:
    """以 canonical JSON 保存全部共享对象的完整 stable keys。"""
    if not isinstance(materialization, HtmlCarrierMaterialization):
        raise HtmlCarrierAdapterError("serializer 输入类型非法")
    value = {
        "anchors": _stable_lists(materialization.anchors),
        "artifact_kind": MATERIALIZATION_KIND,
        "case_key": materialization.record.case_key.to_list(),
        "envelopes": _stable_lists(materialization.envelopes),
        "format_version": MATERIALIZATION_FORMAT_VERSION,
        "references": _stable_lists(materialization.references),
        "revisions": _stable_lists(materialization.revisions),
        "sample_kind": materialization.record.sample_kind,
        "scopes": _stable_lists(materialization.scopes),
        "sources": _stable_lists(materialization.sources),
        "structure_nodes": _stable_lists(materialization.structure_nodes),
    }
    return canonical_json_bytes(value) + b"\n"


def _strict_stable_keys(value: Any, *, where: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise HtmlCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result: list[tuple[int, ...]] = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise HtmlCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_html_materialization(
        payload: bytes,
        record: HtmlCarrierRecord,
        ) -> HtmlCarrierMaterialization:
    """严格回读 canonical bytes，并对照冻结 payload 重验对象图。"""
    if not isinstance(record, HtmlCarrierRecord):
        raise HtmlCarrierAdapterError("deserializer record 类型非法")
    try:
        if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
                or payload.endswith(b"\n\n")):
            raise HtmlCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        expected = {
            "anchors", "artifact_kind", "case_key", "envelopes",
            "format_version", "references", "revisions", "sample_kind",
            "scopes", "sources", "structure_nodes",
        }
        if set(value) != expected:
            raise HtmlCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise HtmlCarrierAdapterError("materialization record 身份漂移")
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
        references = tuple(ArtifactReferenceBinding.from_stable_key(item) for item in
                           _strict_stable_keys(value["references"], where="references"))
        revisions = tuple(ArtifactCarrierRevision.from_stable_key(item) for item in
                          _strict_stable_keys(value["revisions"], where="revisions"))
        result = HtmlCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, nodes, references,
            revisions)
    except HtmlCarrierAdapterError:
        raise
    except Exception as error:
        raise HtmlCarrierAdapterError("materialization 损坏") from error
    if serialize_html_materialization(result) != payload:
        raise HtmlCarrierAdapterError("materialization 不是 canonical 表示")
    return result


# 全称别名供 catalog 与审计调用者使用。
serialize_html_carrier_materialization = serialize_html_materialization
deserialize_html_carrier_materialization = deserialize_html_materialization


__all__ = [
    "HTML_SOURCE_KIND",
    "MATERIALIZATION_FORMAT_VERSION",
    "MATERIALIZATION_KIND",
    "HtmlCarrierAdapterError",
    "HtmlCarrierMaterialization",
    "adapt_html_carrier_record",
    "deserialize_html_carrier_materialization",
    "deserialize_html_materialization",
    "serialize_html_carrier_materialization",
    "serialize_html_materialization",
]
