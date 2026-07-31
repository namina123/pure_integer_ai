"""LC-16 REFERENCE_LINK_EMBED payload 的确定性 raw + reference adapter。

raw reference payload 始终以 Unicode scalar 原样保存在 ``ArtifactEnvelope``
中；slot 只是同一观测的结构视图，不替代、规范化或重写原文。解析器
不访问网络，引用只物化为显式 resolved/unresolved/access-blocked 状态。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TEXT_RANGE,
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
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_contract import (
    ReferenceLinkEmbedCarrierRecord,
    PARSER_VERSION,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_REFERENCE_LINK_EMBED_MATERIALIZATION"
REFERENCE_LINK_EMBED_SOURCE_KIND = 16616507
_IDENTITY_BASE = 16616580


class ReferenceLinkEmbedCarrierAdapterError(RuntimeError):
    """REFERENCE_LINK_EMBED adapter 输入、引用对象或 canonical 表示不闭合。"""


def _key(record: ReferenceLinkEmbedCarrierRecord, domain: int, *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: ReferenceLinkEmbedCarrierRecord, parser_version: int) -> SourceRef:
    values = record.case_key.stable_key()
    case_index = values[-1]
    owner = OwnerScope(values[0], values[-2], case_index, VISIBILITY_SESSION)
    versions = VersionBundle(
        CorpusVersion(1), ParserVersion(parser_version), PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        REFERENCE_LINK_EMBED_SOURCE_KIND,
        values[0] + values[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: ReferenceLinkEmbedCarrierRecord, domain: int,
             *tail: int):
    return concept_identity(
        _key(record, domain, *tail), owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, record: ReferenceLinkEmbedCarrierRecord, domain: int,
               *tail: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain, *tail),
        _concept(source, record, domain + 1, *tail),
    )


def _parser(source: SourceRef, record: ReferenceLinkEmbedCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 20)


def _renderer(source: SourceRef, record: ReferenceLinkEmbedCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 22)


def _utf8_key(value: str) -> tuple[int, ...]:
    return tuple(value.encode("utf-8")) or (0,)


@dataclass(frozen=True)
class _LocalTargetMeta:
    identifier: str
    start: int
    end: int


@dataclass(frozen=True)
class _ReferenceMeta:
    identifier: str
    reference_kind: str
    start: int
    end: int
    surface: str
    target: str


@dataclass(frozen=True)
class _ParsedReferencePayload:
    content: str
    local_targets: tuple[_LocalTargetMeta, ...]
    references: tuple[_ReferenceMeta, ...]
    parser_state: str
    parser_details: dict[str, Any]


def _reference_receipt(
        *, reference: _ReferenceMeta, target_state: int) -> bytes:
    return canonical_json_bytes({
        "family": "REFERENCE",
        "kind": reference.reference_kind,
        "parser": PARSER_VERSION,
        "slot_id": reference.identifier,
        "span": [reference.start, reference.end],
        "surface": reference.surface,
        "target": reference.target,
        "target_state": target_state,
    })


def _parser_receipt(state: str, details: dict[str, Any]) -> bytes:
    return canonical_json_bytes({
        "details": details,
        "family": "PARSER_STATE",
        "parser": PARSER_VERSION,
        "type": state,
    })


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_integer_number(value: str) -> Any:
    raise ValueError(f"non-integer JSON number: {value}")


def _parse_reference_payload(text: str) -> _ParsedReferencePayload:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_non_integer_number,
            parse_float=_reject_non_integer_number,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        return _ParsedReferencePayload(
            "", (), (), "JSON_ERROR",
            {"column": getattr(error, "colno", 0),
             "line": getattr(error, "lineno", 0),
             "message": str(error),
             "offset": getattr(error, "pos", 0)},
        )
    if (not isinstance(value, dict)
            or set(value) != {"content", "local_targets", "references"}
            or not isinstance(value["content"], str)
            or not isinstance(value["local_targets"], list)
            or not isinstance(value["references"], list)):
        return _ParsedReferencePayload(
            "", (), (), "REFERENCE_SCHEMA_ERROR",
            {"reason": "TOP_LEVEL_FIELDS"},
        )
    content = value["content"]
    local_targets: list[_LocalTargetMeta] = []
    references: list[_ReferenceMeta] = []
    errors: list[str] = []
    for index, item in enumerate(value["local_targets"]):
        if (not isinstance(item, dict) or set(item) != {"id", "span"}
                or not isinstance(item["id"], str) or not item["id"]
                or not isinstance(item["span"], list)
                or len(item["span"]) != 2
                or any(type(number) is not int or number < 0
                       for number in item["span"])
                or item["span"][0] > item["span"][1]
                or item["span"][1] > len(content)):
            errors.append(f"LOCAL_TARGET:{index}")
            continue
        local_targets.append(_LocalTargetMeta(
            item["id"], item["span"][0], item["span"][1]))
    for index, item in enumerate(value["references"]):
        if (not isinstance(item, dict)
                or set(item) != {"id", "kind", "span", "surface", "target"}
                or not isinstance(item["id"], str) or not item["id"]
                or not isinstance(item["kind"], str) or not item["kind"]
                or not isinstance(item["span"], list)
                or len(item["span"]) != 2
                or any(type(number) is not int or number < 0
                       for number in item["span"])
                or item["span"][0] > item["span"][1]
                or item["span"][1] > len(content)
                or not isinstance(item["surface"], str)
                or not isinstance(item["target"], str)):
            errors.append(f"REFERENCE:{index}")
            continue
        start, end = item["span"]
        if content[start:end] != item["surface"]:
            errors.append(f"REFERENCE_SURFACE:{index}")
            continue
        references.append(_ReferenceMeta(
            item["id"], item["kind"], start, end,
            item["surface"], item["target"],
        ))
    local_ids = [item.identifier for item in local_targets]
    reference_ids = [item.identifier for item in references]
    if len(local_ids) != len(set(local_ids)):
        errors.append("DUPLICATE_LOCAL_TARGET")
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("DUPLICATE_REFERENCE")
    if errors:
        return _ParsedReferencePayload(
            content, (), (), "REFERENCE_SCHEMA_ERROR",
            {"errors": sorted(set(errors))},
        )
    return _ParsedReferencePayload(
        content, tuple(local_targets), tuple(references),
        "REFERENCE_SET_OK",
        {"local_target_count": len(local_targets),
         "reference_count": len(references)},
    )


@dataclass(frozen=True)
class ReferenceLinkEmbedCarrierMaterialization:
    """一个 reference case 的 raw、slot、binding 与可选 revision 完整物化。"""

    record: ReferenceLinkEmbedCarrierRecord
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
    def reference_anchors(self) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_REFERENCE_SLOT)

    @property
    def reference_bindings(self) -> tuple[ArtifactReferenceBinding, ...]:
        """共享合同名称下的显式引用绑定视图。"""
        return self.references

    def __post_init__(self) -> None:
        if not isinstance(self.record, ReferenceLinkEmbedCarrierRecord):
            raise ReferenceLinkEmbedCarrierAdapterError("materialization record 类型非法")
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
                raise ReferenceLinkEmbedCarrierAdapterError(f"materialization {name} 类型非法")
        if (len(self.sources) != expected_count
                or len(self.scopes) != expected_count
                or len(self.envelopes) != expected_count):
            raise ReferenceLinkEmbedCarrierAdapterError("materialization 顶层对象数量漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise ReferenceLinkEmbedCarrierAdapterError("materialization revisions 数量漂移")

        expected_texts = ((self.record.previous_text, self.record.raw_text)
                          if expected_count == 2 else (self.record.raw_text,))
        for source, scope, envelope, text in zip(
                self.sources, self.scopes, self.envelopes, expected_texts):
            parser = _parser(source, self.record)
            if scope != document_scope(source):
                raise ReferenceLinkEmbedCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise ReferenceLinkEmbedCarrierAdapterError("envelope raw/source 漂移")
            group = self._anchors_for_envelope(envelope.identity)
            if (not any(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group)
                    or not any(item.anchor_kind == ANCHOR_DOCUMENT_REGION
                               for item in group)):
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "REFERENCE_LINK_EMBED 缺少 text/document anchor")
            if any(item.anchor_kind not in {
                    ANCHOR_TEXT_RANGE, ANCHOR_DOCUMENT_REGION,
                    ANCHOR_REFERENCE_SLOT} for item in group):
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "REFERENCE_LINK_EMBED anchor kind 越界")
            if any(item.source != source or item.scope != scope
                   or item.envelope_identity != envelope.identity
                   or item.parser != parser for item in group):
                raise ReferenceLinkEmbedCarrierAdapterError("anchor context 漂移")
            nodes = tuple(item for item in self.structure_nodes
                          if item.envelope_identity == envelope.identity)
            if not nodes:
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "REFERENCE_LINK_EMBED 缺少 structure nodes")
            node_ids = {item.identity for item in nodes}
            structure_anchor_ids = {item.identity for item in group
                                    if item.anchor_kind in {
                                        ANCHOR_DOCUMENT_REGION,
                                        ANCHOR_REFERENCE_SLOT}}
            if any(item.anchor_identity not in structure_anchor_ids
                   for item in nodes):
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "reference node 未绑定 document/reference anchor")
            if any(item.parent_identity is not None
                   and item.parent_identity not in node_ids for item in nodes):
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "reference node parent 逃逸")
            if tuple(item.ordinal for item in nodes) != tuple(
                    sorted(item.ordinal for item in nodes)):
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "reference node ordinal 漂移")
            refs = tuple(item for item in self.references
                         if item.envelope_identity == envelope.identity)
            reference_anchor_ids = {item.identity for item in group
                                    if item.anchor_kind == ANCHOR_REFERENCE_SLOT}
            if any(item.source != source or item.scope != scope
                   or item.anchor_identity not in reference_anchor_ids
                   for item in refs):
                raise ReferenceLinkEmbedCarrierAdapterError("reference context 漂移")
            local_document_ids = {item.identity for item in group
                                  if item.anchor_kind == ANCHOR_DOCUMENT_REGION}
            if any(item.target_state == REFERENCE_RESOLVED
                   and (item.target_source != source
                        or item.target_anchor not in local_document_ids)
                   for item in refs):
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "resolved target 未绑定片内 document region")
            for artifact in (envelope, *group, *nodes, *refs):
                try:
                    if type(artifact).from_stable_key(artifact.stable_key()) != artifact:
                        raise ReferenceLinkEmbedCarrierAdapterError("对象无法稳定回读")
                except ReferenceLinkEmbedCarrierAdapterError:
                    raise
                except Exception as error:
                    raise ReferenceLinkEmbedCarrierAdapterError("对象无法稳定回读") from error

        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise ReferenceLinkEmbedCarrierAdapterError("revision parser lineage 漂移")
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source):
                raise ReferenceLinkEmbedCarrierAdapterError("revision identity 漂移")

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
        record: ReferenceLinkEmbedCarrierRecord,
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
    parsed = _parse_reference_payload(text)
    text_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TEXT_RANGE, coordinates=(0, len(text)),
        parser=parser, linked_text_anchor=None,
        anchor_key=_key(record, 31, variant, 1),
    )
    document_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_DOCUMENT_REGION,
        coordinates=(0, len(parsed.content)),
        parser=parser, linked_text_anchor=None,
        anchor_key=_key(record, 31, variant, 2),
    )
    anchors = [text_anchor, document_anchor]
    local_targets: dict[str, tuple[ArtifactAnchor, ...]] = {}
    for target_ordinal, target in enumerate(parsed.local_targets):
        anchor = make_artifact_anchor(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_kind=ANCHOR_DOCUMENT_REGION,
            coordinates=(target.start, target.end, *_utf8_key(target.identifier)),
            parser=parser, linked_text_anchor=None,
            anchor_key=_key(
                record, 31, variant, 3, target_ordinal,
                *_utf8_key(target.identifier)),
        )
        anchors.append(anchor)
        local_targets.setdefault(target.identifier, tuple())
        local_targets[target.identifier] = (
            *local_targets[target.identifier], anchor)

    references: list[ArtifactReferenceBinding] = []
    reference_anchors: list[ArtifactAnchor] = []
    reference_receipts: list[bytes] = []
    for reference_ordinal, meta in enumerate(parsed.references):
        reference_anchor = make_artifact_anchor(
            envelope_identity=envelope.identity,
            source=source,
            scope=scope,
            anchor_kind=ANCHOR_REFERENCE_SLOT,
            coordinates=(meta.start, meta.end, *_utf8_key(meta.identifier)),
            parser=parser,
            linked_text_anchor=None,
            anchor_key=_key(
                record, 31, variant, 4, reference_ordinal,
                *_utf8_key(meta.identifier)),
        )
        anchors.append(reference_anchor)
        reference_anchors.append(reference_anchor)
        state, target_source, target_anchor, fingerprint = _reference_state(
            meta.target, source, local_targets)
        reference_receipts.append(_reference_receipt(
            reference=meta, target_state=state))
        references.append(make_artifact_reference_binding(
            envelope_identity=envelope.identity,
            source=source,
            scope=scope,
            anchor_identity=reference_anchor.identity,
            relation=_concept(
                source, record, 80, *_utf8_key(meta.reference_kind)),
            target_state=state,
            target_source=target_source,
            target_anchor=target_anchor,
            target_fingerprint=fingerprint,
            reference_key=_key(
                record, 81, variant, reference_ordinal,
                *_utf8_key(meta.identifier)),
        ))

    family = structure_concept_identity(
        _key(record, 60), owner=source.owner, versions=source.versions)
    nodes: list[ArtifactStructureNode] = []
    for ordinal, (meta, anchor, receipt) in enumerate(zip(
            parsed.references, reference_anchors, reference_receipts)):
        node_kind = structure_concept_identity(
            _key(record, 61, *_utf8_key(f"REFERENCE:{meta.reference_kind}")),
            owner=source.owner, versions=source.versions)
        nodes.append(make_artifact_structure_node(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_identity=anchor.identity, structure_family=family,
            node_kind=node_kind, role=None, parent_identity=None,
            ordinal=ordinal_offset + ordinal, qualifiers=tuple(receipt),
            node_key=_key(
                record, 62, variant, ordinal, *_utf8_key(meta.identifier)),
        ))
    parser_receipt = _parser_receipt(
        parsed.parser_state, parsed.parser_details)
    parser_ordinal = len(nodes)
    parser_node_kind = structure_concept_identity(
        _key(record, 61, *_utf8_key(f"PARSER_STATE:{parsed.parser_state}")),
        owner=source.owner, versions=source.versions)
    nodes.append(make_artifact_structure_node(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_identity=document_anchor.identity, structure_family=family,
        node_kind=parser_node_kind, role=None, parent_identity=None,
        ordinal=ordinal_offset + parser_ordinal,
        qualifiers=tuple(parser_receipt),
        node_key=_key(
            record, 62, variant, parser_ordinal,
            *_utf8_key(parsed.parser_state)),
    ))
    return envelope, tuple(anchors), tuple(nodes), tuple(references)


def _node_signature(node: ArtifactStructureNode) -> tuple[Any, ...]:
    try:
        receipt = parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        family = str(receipt["family"])
        if family == "REFERENCE":
            return family, str(receipt["slot_id"]), str(receipt["kind"])
        return family, str(receipt["type"])
    except Exception as error:
        raise ReferenceLinkEmbedCarrierAdapterError(
            "reference node receipt 损坏") from error


def _anchor_identifier(anchor: ArtifactAnchor, *, where: str) -> str:
    if len(anchor.coordinates) < 3:
        raise ReferenceLinkEmbedCarrierAdapterError(f"{where} 坐标缺少 identity")
    try:
        return bytes(anchor.coordinates[2:]).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise ReferenceLinkEmbedCarrierAdapterError(
            f"{where} identity 损坏") from error


def _reference_signatures(
        anchors: tuple[ArtifactAnchor, ...],
        references: tuple[ArtifactReferenceBinding, ...],
        ) -> dict[Any, str]:
    """返回不受 span 移动影响的 slot identity 修订签名。"""
    anchors_by_identity = {item.identity: item for item in anchors}
    try:
        result = {}
        for reference in references:
            anchor = anchors_by_identity[reference.anchor_identity]
            if anchor.anchor_kind != ANCHOR_REFERENCE_SLOT:
                raise ReferenceLinkEmbedCarrierAdapterError(
                    "reference binding 未绑定 reference anchor")
            result[reference.identity] = _anchor_identifier(
                anchor, where="reference anchor")
        return result
    except ReferenceLinkEmbedCarrierAdapterError:
        raise
    except Exception as error:
        raise ReferenceLinkEmbedCarrierAdapterError(
            "reference revision signature 损坏") from error


def _revision(
        record: ReferenceLinkEmbedCarrierRecord,
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
    old_reference_signatures = _reference_signatures(
        old_anchors, old_references)
    new_reference_signatures = _reference_signatures(
        new_anchors, new_references)
    new_reference_by_signature = {
        new_reference_signatures[item.identity]: item
        for item in new_references
    }
    old_reference_by_anchor = {
        item.anchor_identity: item for item in old_references
    }
    new_full_document = tuple(
        item for item in new_anchors
        if item.anchor_kind == ANCHOR_DOCUMENT_REGION
        and len(item.coordinates) == 2)
    new_document_by_identifier = {
        _anchor_identifier(item, where="document anchor"): item
        for item in new_anchors
        if item.anchor_kind == ANCHOR_DOCUMENT_REGION
        and len(item.coordinates) >= 3
    }
    mappings: list[ArtifactRevisionMapping] = []
    for anchor in old_anchors:
        targets: tuple[Any, ...]
        if anchor.anchor_kind == ANCHOR_TEXT_RANGE:
            targets = tuple(item.identity for item in new_anchors
                            if item.anchor_kind == ANCHOR_TEXT_RANGE)
        elif anchor.anchor_kind == ANCHOR_DOCUMENT_REGION:
            if len(anchor.coordinates) == 2:
                targets = tuple(item.identity for item in new_full_document)
            else:
                target = new_document_by_identifier.get(
                    _anchor_identifier(anchor, where="document anchor"))
                targets = () if target is None else (target.identity,)
        elif anchor.anchor_kind == ANCHOR_REFERENCE_SLOT:
            old_reference = old_reference_by_anchor.get(anchor.identity)
            signature = (None if old_reference is None else
                         old_reference_signatures[old_reference.identity])
            target = (None if signature is None else
                      new_reference_by_signature.get(signature))
            targets = () if target is None else (target.anchor_identity,)
        else:
            raise ReferenceLinkEmbedCarrierAdapterError(
                "revision anchor kind 越界")
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


def adapt_reference_link_embed_carrier_record(record: ReferenceLinkEmbedCarrierRecord) -> ReferenceLinkEmbedCarrierMaterialization:
    """不训练、不选义地把一个冻结 REFERENCE_LINK_EMBED payload 物化为共享 carrier 对象。"""
    if not isinstance(record, ReferenceLinkEmbedCarrierRecord):
        raise ReferenceLinkEmbedCarrierAdapterError("adapter 只接受 ReferenceLinkEmbedCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope = document_scope(source)
        envelope, anchors, nodes, references = _build_envelope(
            record, source, record.raw_text, 1)
        return ReferenceLinkEmbedCarrierMaterialization(
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
    return ReferenceLinkEmbedCarrierMaterialization(
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


def serialize_reference_link_embed_materialization(
        materialization: ReferenceLinkEmbedCarrierMaterialization,
        ) -> bytes:
    """以 canonical JSON 保存全部共享对象的完整 stable keys。"""
    if not isinstance(materialization, ReferenceLinkEmbedCarrierMaterialization):
        raise ReferenceLinkEmbedCarrierAdapterError("serializer 输入类型非法")
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
        raise ReferenceLinkEmbedCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result: list[tuple[int, ...]] = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise ReferenceLinkEmbedCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_reference_link_embed_materialization(
        payload: bytes,
        record: ReferenceLinkEmbedCarrierRecord,
        ) -> ReferenceLinkEmbedCarrierMaterialization:
    """严格回读 canonical bytes，并对照冻结 payload 重验对象图。"""
    if not isinstance(record, ReferenceLinkEmbedCarrierRecord):
        raise ReferenceLinkEmbedCarrierAdapterError("deserializer record 类型非法")
    try:
        if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
                or payload.endswith(b"\n\n")):
            raise ReferenceLinkEmbedCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        expected = {
            "anchors", "artifact_kind", "case_key", "envelopes",
            "format_version", "references", "revisions", "sample_kind",
            "scopes", "sources", "structure_nodes",
        }
        if set(value) != expected:
            raise ReferenceLinkEmbedCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise ReferenceLinkEmbedCarrierAdapterError("materialization record 身份漂移")
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
        result = ReferenceLinkEmbedCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, nodes, references,
            revisions)
    except ReferenceLinkEmbedCarrierAdapterError:
        raise
    except Exception as error:
        raise ReferenceLinkEmbedCarrierAdapterError("materialization 损坏") from error
    if serialize_reference_link_embed_materialization(result) != payload:
        raise ReferenceLinkEmbedCarrierAdapterError("materialization 不是 canonical 表示")
    return result


# 全称别名供 catalog 与审计调用者使用。
serialize_reference_link_embed_carrier_materialization = serialize_reference_link_embed_materialization
deserialize_reference_link_embed_carrier_materialization = deserialize_reference_link_embed_materialization


__all__ = [
    "REFERENCE_LINK_EMBED_SOURCE_KIND",
    "MATERIALIZATION_FORMAT_VERSION",
    "MATERIALIZATION_KIND",
    "ReferenceLinkEmbedCarrierAdapterError",
    "ReferenceLinkEmbedCarrierMaterialization",
    "adapt_reference_link_embed_carrier_record",
    "deserialize_reference_link_embed_carrier_materialization",
    "deserialize_reference_link_embed_materialization",
    "serialize_reference_link_embed_carrier_materialization",
    "serialize_reference_link_embed_materialization",
]
