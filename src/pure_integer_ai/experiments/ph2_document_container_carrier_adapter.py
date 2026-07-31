"""LC-16 DOCUMENT_CONTAINER payload 的确定性 container structure adapter。

该模块只负责把冻结的 :class:`DocumentContainerCarrierRecord` 物化为共享
``ArtifactEnvelope``/``ArtifactAnchor``/``ArtifactStructureNode`` 对象。
它不作语义预选、不训练，也不把容器解析结果当作原文；结构节点的完整
canonical receipt 保存在 structure node 的整数 ``qualifiers`` 中。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

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
from pure_integer_ai.experiments.ph2_document_container_carrier_contract import (
    DocumentContainerCarrierRecord,
    PARSER_VERSION,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_DOCUMENT_CONTAINER_MATERIALIZATION"
DOCUMENT_CONTAINER_SOURCE_KIND = 16616500
_IDENTITY_BASE = 16616480


class DocumentContainerCarrierAdapterError(RuntimeError):
    """DOCUMENT_CONTAINER adapter 输入、结构对象或 canonical 表示不闭合。"""


def _key(record: DocumentContainerCarrierRecord, domain: int, *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: DocumentContainerCarrierRecord, parser_version: int) -> SourceRef:
    values = record.case_key.stable_key()
    case_index = values[-1]
    owner = OwnerScope(values[0], values[-2], case_index, VISIBILITY_SESSION)
    versions = VersionBundle(
        CorpusVersion(1), ParserVersion(parser_version), PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        DOCUMENT_CONTAINER_SOURCE_KIND,
        values[0] + values[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: DocumentContainerCarrierRecord, domain: int,
             *tail: int):
    return concept_identity(
        _key(record, domain, *tail), owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, record: DocumentContainerCarrierRecord, domain: int,
               *tail: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain, *tail),
        _concept(source, record, domain + 1, *tail),
    )


def _parser(source: SourceRef, record: DocumentContainerCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 20)


def _renderer(source: SourceRef, record: DocumentContainerCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 22)


@dataclass(frozen=True)
class _NodeMeta:
    ordinal: int
    path: tuple[int, ...]
    parent_ordinal: int | None
    region: tuple[int, int]
    receipt: bytes
    node_type: str
    nesting: int


def _receipt(
        *, family: str, kind: str, path: tuple[int, ...], ordinal: int,
        region: tuple[int, int], nesting: int,
        details: dict[str, Any]) -> bytes:
    return canonical_json_bytes({
        "details": details,
        "family": family,
        "language": "document-container",
        "nesting": nesting,
        "ordinal": ordinal,
        "parser": PARSER_VERSION,
        "path": list(path),
        "region": [region[0], region[1]],
        "type": kind,
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


def _container_metas(text: str) -> tuple[_NodeMeta, ...]:
    metas: list[_NodeMeta] = []

    def emit(
            family: str, kind: str, path: tuple[int, ...],
            parent_ordinal: int | None, region: tuple[int, int], nesting: int,
            details: dict[str, Any]) -> int:
        ordinal = len(metas)
        metas.append(_NodeMeta(
            ordinal, path, parent_ordinal, region,
            _receipt(
                family=family, kind=kind, path=path, ordinal=ordinal,
                region=region, nesting=nesting, details=details),
            f"{family}:{kind}", nesting,
        ))
        return ordinal

    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_non_integer_number,
            parse_float=_reject_non_integer_number,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        emit(
            "PARSER_STATE", "JSON_ERROR", (3, 0), None, (0, 1), 0,
            {"column": getattr(error, "colno", 0),
             "line": getattr(error, "lineno", 0),
             "message": str(error),
             "offset": getattr(error, "pos", 0)},
        )
        return tuple(metas)

    schema_errors: list[str] = []
    expected_top = {"blocks", "document_id", "reading_order", "title"}
    if not isinstance(value, dict) or set(value) != expected_top:
        schema_errors.append("TOP_LEVEL_FIELDS")
        document_id = ""
        title = ""
        raw_blocks: list[Any] = []
        raw_reading_order: list[Any] = []
    else:
        document_id = value["document_id"]
        title = value["title"]
        raw_blocks = value["blocks"]
        raw_reading_order = value["reading_order"]
        if not isinstance(document_id, str) or not document_id:
            schema_errors.append("DOCUMENT_ID")
            document_id = ""
        if not isinstance(title, str):
            schema_errors.append("TITLE")
            title = ""
        if not isinstance(raw_blocks, list):
            schema_errors.append("BLOCKS")
            raw_blocks = []
        if (not isinstance(raw_reading_order, list)
                or any(not isinstance(item, str)
                       for item in raw_reading_order)):
            schema_errors.append("READING_ORDER")
            raw_reading_order = []

    expected_block = {
        "attributes", "id", "kind", "order", "parent", "target", "text",
    }
    blocks: list[dict[str, Any]] = []
    for index, item in enumerate(raw_blocks):
        if not isinstance(item, dict) or set(item) != expected_block:
            schema_errors.append(f"BLOCK_FIELDS:{index}")
            continue
        if (not isinstance(item["id"], str) or not item["id"]
                or not isinstance(item["kind"], str) or not item["kind"]
                or type(item["order"]) is not int or item["order"] < 0
                or not isinstance(item["parent"], str)
                or not isinstance(item["target"], str)
                or not isinstance(item["text"], str)
                or not isinstance(item["attributes"], dict)):
            schema_errors.append(f"BLOCK_TYPES:{index}")
            continue
        blocks.append(item)

    ids = [item["id"] for item in blocks]
    if len(ids) != len(set(ids)):
        schema_errors.append("DUPLICATE_BLOCK_ID")
    if len({item["order"] for item in blocks}) != len(blocks):
        schema_errors.append("DUPLICATE_BLOCK_ORDER")
    if (len(raw_reading_order) != len(set(raw_reading_order))
            or set(raw_reading_order) != set(ids)):
        schema_errors.append("READING_ORDER_COVERAGE")

    by_id = {item["id"]: item for item in blocks}
    depth_cache: dict[str, int] = {}

    def depth(block_id: str, trail: tuple[str, ...] = ()) -> int:
        if block_id in depth_cache:
            return depth_cache[block_id]
        if block_id in trail:
            schema_errors.append(f"PARENT_CYCLE:{block_id}")
            depth_cache[block_id] = 1
            return 1
        parent_id = by_id[block_id]["parent"]
        if not parent_id:
            result = 1
        elif parent_id not in by_id:
            schema_errors.append(f"DANGLING_PARENT:{block_id}")
            result = 1
        else:
            result = depth(parent_id, (*trail, block_id)) + 1
        depth_cache[block_id] = result
        return result

    full_region = (0, max(1, len(blocks)))
    root_ordinal = emit(
        "CONTAINER", "DOCUMENT", (0,), None, full_region, 0,
        {"block_count": len(blocks), "document_id": document_id,
         "title": title},
    )
    emitted: dict[str, int] = {}
    for block in sorted(
            blocks, key=lambda item: (depth(item["id"]), item["order"],
                                      item["id"])):
        parent_id = block["parent"]
        parent_ordinal = emitted.get(parent_id, root_ordinal)
        target = block["target"]
        target_candidates = [target] if target and target in by_id else []
        region = (block["order"], block["order"] + 1)
        emitted[block["id"]] = emit(
            "CONTAINER", block["kind"],
            (1, *block["id"].encode("utf-8")), parent_ordinal,
            region, depth(block["id"]),
            {"attributes": block["attributes"], "id": block["id"],
             "order": block["order"], "parent": parent_id,
             "target": target, "target_candidates": target_candidates,
             "text": block["text"]},
        )
    emit(
        "CONTAINER", "READ_ORDER", (2,), root_ordinal, full_region, 1,
        {"block_ids": raw_reading_order},
    )
    target_issues = sorted(
        item["id"] for item in blocks
        if item["target"] and item["target"] not in by_id)
    if target_issues:
        emit(
            "PARSER_STATE", "UNRESOLVED_TARGETS_PRESENT", (3, 0),
            root_ordinal, full_region, 1, {"block_ids": target_issues},
        )
    if schema_errors:
        emit(
            "PARSER_STATE", "CONTAINER_SCHEMA_ERROR", (3, 1),
            root_ordinal, full_region, 1,
            {"errors": sorted(set(schema_errors))},
        )
    else:
        emit(
            "PARSER_STATE", "CONTAINER_OK", (3, 1), root_ordinal,
            full_region, 1, {"block_count": len(blocks)},
        )
    return tuple(metas)


@dataclass(frozen=True)
class DocumentContainerCarrierMaterialization:
    """一个 DOCUMENT_CONTAINER case 的完整 envelope、anchor、structure node 与 revision。"""

    record: DocumentContainerCarrierRecord
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
    def tree_anchors(self) -> tuple[ArtifactAnchor, ...]:
        """按 container node 顺序返回 tree-path 锚。"""
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_TREE_PATH)

    @property
    def document_anchors(self) -> tuple[ArtifactAnchor, ...]:
        """按 container node 顺序返回 document-region 锚。"""
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_DOCUMENT_REGION)

    def __post_init__(self) -> None:
        if not isinstance(self.record, DocumentContainerCarrierRecord):
            raise DocumentContainerCarrierAdapterError("materialization record 类型非法")
        expected_count = 2 if self.record.sample_kind == "REVISION" else 1
        for name, cls in (
                ("sources", SourceRef), ("scopes", ScopeIdentity),
                ("envelopes", ArtifactEnvelope), ("anchors", ArtifactAnchor),
                ("structure_nodes", ArtifactStructureNode)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or len(values) == 0
                    or any(not isinstance(item, cls) for item in values)):
                raise DocumentContainerCarrierAdapterError(f"materialization {name} 类型非法")
        if any(len(group) == 0 for group in (self.sources, self.scopes, self.envelopes)):
            raise DocumentContainerCarrierAdapterError("materialization 顶层对象不能为空")
        if len(self.sources) != expected_count or len(self.scopes) != expected_count:
            raise DocumentContainerCarrierAdapterError("materialization source/scope 数量漂移")
        if len(self.envelopes) != expected_count:
            raise DocumentContainerCarrierAdapterError("materialization envelope 数量漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise DocumentContainerCarrierAdapterError("materialization revisions 数量漂移")

        expected_texts = ((self.record.previous_text, self.record.raw_text)
                          if expected_count == 2 else (self.record.raw_text,))
        for index, (source, scope, envelope, text) in enumerate(zip(
                self.sources, self.scopes, self.envelopes, expected_texts)):
            parser = _parser(source, self.record)
            if scope != document_scope(source):
                raise DocumentContainerCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise DocumentContainerCarrierAdapterError("envelope raw/source 漂移")
            group = self._anchors_for_envelope(envelope.identity)
            if len(group) < 3:
                raise DocumentContainerCarrierAdapterError(
                    "DOCUMENT_CONTAINER 缺少 text/tree/document anchors")
            if not any(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group):
                raise DocumentContainerCarrierAdapterError("DOCUMENT_CONTAINER 缺少 full text anchor")
            if not any(item.anchor_kind == ANCHOR_TREE_PATH for item in group):
                raise DocumentContainerCarrierAdapterError("DOCUMENT_CONTAINER 缺少 tree anchor")
            if not any(item.anchor_kind == ANCHOR_DOCUMENT_REGION for item in group):
                raise DocumentContainerCarrierAdapterError(
                    "DOCUMENT_CONTAINER 缺少 document region anchor")
            if any(item.source != source or item.scope != scope
                   or item.envelope_identity != envelope.identity
                   or item.parser != parser for item in group):
                raise DocumentContainerCarrierAdapterError("anchor context 漂移")
            nodes = tuple(item for item in self.structure_nodes
                          if item.envelope_identity == envelope.identity)
            if not nodes:
                raise DocumentContainerCarrierAdapterError("DOCUMENT_CONTAINER 缺少 structure nodes")
            node_ids = {item.identity for item in nodes}
            anchor_ids = {item.identity for item in group
                          if item.anchor_kind == ANCHOR_TREE_PATH}
            if any(item.anchor_identity not in anchor_ids for item in nodes):
                raise DocumentContainerCarrierAdapterError("structure node 未绑定 tree anchor")
            if any(item.parent_identity is not None
                   and item.parent_identity not in node_ids for item in nodes):
                raise DocumentContainerCarrierAdapterError("structure node parent 逃逸")
            if tuple(item.ordinal for item in nodes) != tuple(
                    sorted(item.ordinal for item in nodes)):
                raise DocumentContainerCarrierAdapterError("structure node ordinal 漂移")
            for artifact in (envelope, *group, *nodes):
                try:
                    if type(artifact).from_stable_key(artifact.stable_key()) != artifact:
                        raise DocumentContainerCarrierAdapterError("对象无法稳定回读")
                except DocumentContainerCarrierAdapterError:
                    raise
                except Exception as error:
                    raise DocumentContainerCarrierAdapterError("对象无法稳定回读") from error

        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise DocumentContainerCarrierAdapterError("revision parser lineage 漂移")
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source):
                raise DocumentContainerCarrierAdapterError("revision envelope/hypothesis 漂移")

    def _anchors_for_envelope(self, identity: Any) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.envelope_identity == identity)


def _build_envelope(record: DocumentContainerCarrierRecord, source: SourceRef,
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
    metas = _container_metas(text)
    text_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TEXT_RANGE, coordinates=(0, len(text)),
        parser=parser, linked_text_anchor=None,
        anchor_key=_key(record, 31, variant, 1),
    )
    anchors = [text_anchor]
    tree_anchors: dict[int, ArtifactAnchor] = {}
    for meta in metas:
        tree_anchor = make_artifact_anchor(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_kind=ANCHOR_TREE_PATH, coordinates=meta.path,
            parser=parser, linked_text_anchor=None,
            anchor_key=_key(record, 31, variant, 3, meta.ordinal, *meta.path),
        )
        region_anchor = make_artifact_anchor(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_kind=ANCHOR_DOCUMENT_REGION, coordinates=meta.region,
            parser=parser, linked_text_anchor=None,
            anchor_key=_key(
                record, 31, variant, 4, meta.ordinal, *meta.region),
        )
        anchors.extend((tree_anchor, region_anchor))
        tree_anchors[meta.ordinal] = tree_anchor

    family = structure_concept_identity(
        _key(record, 60), owner=source.owner, versions=source.versions)
    nodes: list[ArtifactStructureNode] = []
    for meta in metas:
        node_type_key = tuple(meta.node_type.encode("utf-8")) or (0,)
        node_kind = structure_concept_identity(
            _key(record, 61, meta.nesting + 2, *node_type_key),
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


def _revision(record: DocumentContainerCarrierRecord, old: tuple[ArtifactAnchor, ...],
              new: tuple[ArtifactAnchor, ...], old_nodes: tuple[ArtifactStructureNode, ...],
              new_nodes: tuple[ArtifactStructureNode, ...], old_envelope: ArtifactEnvelope,
              new_envelope: ArtifactEnvelope, new_source: SourceRef,
              new_scope: ScopeIdentity) -> ArtifactCarrierRevision:
    new_groups: dict[tuple[int, tuple[int, ...]], list[ArtifactAnchor]] = {}
    for item in new:
        if item.anchor_kind != ANCHOR_TEXT_RANGE:
            new_groups.setdefault(
                (item.anchor_kind, item.coordinates), []).append(item)
    offsets: dict[tuple[int, tuple[int, ...]], int] = {}
    mappings: list[ArtifactRevisionMapping] = []
    for item in old:
        if item.anchor_kind == ANCHOR_TEXT_RANGE:
            targets = tuple(x.identity for x in new if x.anchor_kind == ANCHOR_TEXT_RANGE)
        else:
            key = (item.anchor_kind, item.coordinates)
            offset = offsets.get(key, 0)
            candidates = new_groups.get(key, [])
            target = candidates[offset] if offset < len(candidates) else None
            offsets[key] = offset + 1
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
            raise DocumentContainerCarrierAdapterError(
                "structure node container receipt 损坏") from error

    new_by_signature = {node_signature(node): node for node in new_nodes}
    for node in old_nodes:
        # path 加 node kind 构成局部结构身份；内容和来源范围可以变化，
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


def adapt_document_container_carrier_record(record: DocumentContainerCarrierRecord) -> DocumentContainerCarrierMaterialization:
    """不训练、不选义地把一个冻结 DocumentContainer payload 物化为 carrier 对象。"""
    if not isinstance(record, DocumentContainerCarrierRecord):
        raise DocumentContainerCarrierAdapterError("adapter 只接受 DocumentContainerCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope = document_scope(source)
        envelope, anchors, nodes = _build_envelope(record, source, record.raw_text, 1)
        return DocumentContainerCarrierMaterialization(
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
    return DocumentContainerCarrierMaterialization(
        record, (old_source, new_source), (old_scope, new_scope),
        (old_envelope, new_envelope), (*old_anchors, *new_anchors),
        (*old_nodes, *new_nodes), (revision,),
    )


def _stable_lists(values: tuple[Any, ...]) -> list[list[int]]:
    return [list(item.stable_key()) for item in values]


def serialize_document_container_carrier_materialization(
        materialization: DocumentContainerCarrierMaterialization) -> bytes:
    """以 canonical JSON 保存全部共享对象的完整 stable keys。"""
    if not isinstance(materialization, DocumentContainerCarrierMaterialization):
        raise DocumentContainerCarrierAdapterError("serializer 输入类型非法")
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
        raise DocumentContainerCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result: list[tuple[int, ...]] = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise DocumentContainerCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_document_container_carrier_materialization(
        payload: bytes, record: DocumentContainerCarrierRecord) -> DocumentContainerCarrierMaterialization:
    """严格回读 canonical bytes，并对照冻结 payload 重验对象图。"""
    if not isinstance(record, DocumentContainerCarrierRecord):
        raise DocumentContainerCarrierAdapterError("deserializer record 类型非法")
    try:
        if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
                or payload.endswith(b"\n\n")):
            raise DocumentContainerCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        expected = {
            "anchors", "artifact_kind", "case_key", "envelopes",
            "format_version", "revisions", "sample_kind", "scopes",
            "sources", "structure_nodes",
        }
        if set(value) != expected:
            raise DocumentContainerCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise DocumentContainerCarrierAdapterError("materialization record 身份漂移")
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
        result = DocumentContainerCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, nodes, revisions)
    except DocumentContainerCarrierAdapterError:
        raise
    except Exception as error:
        raise DocumentContainerCarrierAdapterError("materialization 损坏") from error
    if serialize_document_container_carrier_materialization(result) != payload:
        raise DocumentContainerCarrierAdapterError("materialization 不是 canonical 表示")
    return result


# 为兼容 LC-16 catalog/test 接口保留短名称。
serialize_document_container_materialization = serialize_document_container_carrier_materialization
deserialize_document_container_materialization = deserialize_document_container_carrier_materialization


__all__ = [
    "DOCUMENT_CONTAINER_SOURCE_KIND",
    "MATERIALIZATION_FORMAT_VERSION",
    "MATERIALIZATION_KIND",
    "DocumentContainerCarrierAdapterError",
    "DocumentContainerCarrierMaterialization",
    "adapt_document_container_carrier_record",
    "deserialize_document_container_materialization",
    "deserialize_document_container_carrier_materialization",
    "serialize_document_container_materialization",
    "serialize_document_container_carrier_materialization",
]
