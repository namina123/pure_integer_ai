"""LC-16 TABLE_GRID payload 的确定性 grid structure adapter。

该模块只负责把冻结的 :class:`TableGridCarrierRecord` 物化为共享
``ArtifactEnvelope``/``ArtifactAnchor``/``ArtifactStructureNode`` 对象。
它不作语义预选、不训练，也不把 grid 解析结果当作原文；结构节点的完整
canonical receipt 保存在 structure node 的整数 ``qualifiers`` 中。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_GRID_RECT,
    ANCHOR_TEXT_RANGE,
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
from pure_integer_ai.experiments.ph2_table_grid_carrier_contract import (
    TableGridCarrierRecord,
    PARSER_VERSION,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_TABLE_GRID_MATERIALIZATION"
TABLE_GRID_SOURCE_KIND = 16616506
_IDENTITY_BASE = 16616560


class TableGridCarrierAdapterError(RuntimeError):
    """TABLE_GRID adapter 输入、树对象或 canonical 表示不闭合。"""


def _key(record: TableGridCarrierRecord, domain: int, *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: TableGridCarrierRecord, parser_version: int) -> SourceRef:
    values = record.case_key.stable_key()
    case_index = values[-1]
    owner = OwnerScope(values[0], values[-2], case_index, VISIBILITY_SESSION)
    versions = VersionBundle(
        CorpusVersion(1), ParserVersion(parser_version), PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        TABLE_GRID_SOURCE_KIND,
        values[0] + values[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: TableGridCarrierRecord, domain: int,
             *tail: int):
    return concept_identity(
        _key(record, domain, *tail), owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, record: TableGridCarrierRecord, domain: int,
               *tail: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain, *tail),
        _concept(source, record, domain + 1, *tail),
    )


def _parser(source: SourceRef, record: TableGridCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 20)


def _renderer(source: SourceRef, record: TableGridCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 22)


@dataclass(frozen=True)
class _GridMeta:
    ordinal: int
    path: tuple[int, ...]
    parent_ordinal: int | None
    span: tuple[int, int]
    receipt: bytes
    node_type: str
    nesting: int


def _receipt(
        *, family: str, kind: str, path: tuple[int, ...], ordinal: int,
        span: tuple[int, int], nesting: int, details: dict[str, Any]) -> bytes:
    return canonical_json_bytes({
        "details": details,
        "family": family,
        "language": "table-grid",
        "nesting": nesting,
        "ordinal": ordinal,
        "parser": PARSER_VERSION,
        "path": list(path),
        "range": [span[0], span[1]],
        "type": kind,
    })


def _grid_metas(
        record: TableGridCarrierRecord, text: str) -> tuple[_GridMeta, ...]:
    metas: list[_GridMeta] = []
    rows: list[list[str]] = []
    parser_error = ""
    try:
        rows = [list(row) for row in csv.reader(
            io.StringIO(text, newline=""), delimiter=record.delimiter,
            strict=True)]
    except csv.Error as error:
        parser_error = str(error)
    width = max((len(row) for row in rows), default=1)
    height = max(len(rows), 1)
    full_rect = (0, height - 1, 0, width - 1)

    def emit(family: str, kind: str, rect: tuple[int, int, int, int],
             details: dict[str, Any]) -> None:
        ordinal = len(metas)
        path = rect
        span = (0, len(text))
        metas.append(_GridMeta(
            ordinal, path, None, span,
            _receipt(
                family=family, kind=kind, path=path, ordinal=ordinal,
                span=span, nesting=0, details=details),
            f"{family}:{kind}", 0,
        ))

    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            emit("GRID", "CELL", (
                row_index, row_index, column_index, column_index), {
                    "header_column_candidate": column_index in record.header_columns,
                    "header_row_candidate": row_index in record.header_rows,
                    "value": value,
                })
    if rows:
        for row_index in range(len(rows)):
            emit("GRID", "ROW", (row_index, row_index, 0, width - 1), {
                "header_candidate": row_index in record.header_rows,
                "observed_cell_count": len(rows[row_index]),
            })
        for column_index in range(width):
            emit("GRID", "COLUMN", (0, len(rows) - 1,
                                      column_index, column_index), {
                "header_candidate": column_index in record.header_columns,
            })

    occupied: set[tuple[int, int]] = set()
    for rectangle in record.merged_rectangles:
        r0, r1, c0, c1 = rectangle
        if r1 >= height or c1 >= width:
            raise TableGridCarrierAdapterError("merged rectangle 超出 grid")
        coordinates = tuple(
            (row, column) for row in range(r0, r1 + 1)
            for column in range(c0, c1 + 1))
        if any(item in occupied for item in coordinates):
            raise TableGridCarrierAdapterError("merged rectangle 重叠")
        occupied.update(coordinates)
        emit("GRID", "MERGED_REGION", rectangle, {
            "cell_count": len(coordinates),
        })

    if any(index >= height for index in record.header_rows):
        raise TableGridCarrierAdapterError("header row 超出 grid")
    if any(index >= width for index in record.header_columns):
        raise TableGridCarrierAdapterError("header column 超出 grid")
    if record.read_order == "ROW_MAJOR":
        read_cells = tuple(
            (row, column) for row in range(len(rows))
            for column in range(len(rows[row])))
    else:
        read_cells = tuple(
            (row, column) for column in range(width)
            for row in range(len(rows)) if column < len(rows[row]))
    emit("GRID", "READ_ORDER", full_rect, {
        "candidate": record.read_order,
        "cells": [list(item) for item in read_cells],
    })
    widths = [len(row) for row in rows]
    if parser_error:
        emit("PARSER_STATE", "CSV_ERROR", full_rect, {
            "message": parser_error,
        })
    elif len(set(widths)) > 1:
        emit("PARSER_STATE", "RAGGED_GRID", full_rect, {
            "row_widths": widths,
        })
    else:
        emit("PARSER_STATE", "GRID_OK", full_rect, {
            "column_count": width,
            "row_count": len(rows),
        })
    return tuple(metas)


@dataclass(frozen=True)
class TableGridCarrierMaterialization:
    """一个 TABLE_GRID case 的完整 envelope、anchor、structure node 与 revision。"""

    record: TableGridCarrierRecord
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
    def grid_anchors(self) -> tuple[ArtifactAnchor, ...]:
        """按 parser 顺序返回 grid rectangle 锚。"""
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_GRID_RECT)

    def __post_init__(self) -> None:
        if not isinstance(self.record, TableGridCarrierRecord):
            raise TableGridCarrierAdapterError("materialization record 类型非法")
        expected_count = 2 if self.record.sample_kind == "REVISION" else 1
        for name, cls in (
                ("sources", SourceRef), ("scopes", ScopeIdentity),
                ("envelopes", ArtifactEnvelope), ("anchors", ArtifactAnchor),
                ("structure_nodes", ArtifactStructureNode)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or len(values) == 0
                    or any(not isinstance(item, cls) for item in values)):
                raise TableGridCarrierAdapterError(f"materialization {name} 类型非法")
        if any(len(group) == 0 for group in (self.sources, self.scopes, self.envelopes)):
            raise TableGridCarrierAdapterError("materialization 顶层对象不能为空")
        if len(self.sources) != expected_count or len(self.scopes) != expected_count:
            raise TableGridCarrierAdapterError("materialization source/scope 数量漂移")
        if len(self.envelopes) != expected_count:
            raise TableGridCarrierAdapterError("materialization envelope 数量漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise TableGridCarrierAdapterError("materialization revisions 数量漂移")

        expected_texts = ((self.record.previous_text, self.record.raw_text)
                          if expected_count == 2 else (self.record.raw_text,))
        for index, (source, scope, envelope, text) in enumerate(zip(
                self.sources, self.scopes, self.envelopes, expected_texts)):
            parser = _parser(source, self.record)
            if scope != document_scope(source):
                raise TableGridCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise TableGridCarrierAdapterError("envelope raw/source 漂移")
            group = self._anchors_for_envelope(envelope.identity)
            if len(group) < 3:
                raise TableGridCarrierAdapterError("TABLE_GRID 缺少 text/document/grid anchors")
            if not any(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group):
                raise TableGridCarrierAdapterError("TABLE_GRID 缺少 full text anchor")
            if not any(item.anchor_kind == ANCHOR_DOCUMENT_REGION for item in group):
                raise TableGridCarrierAdapterError("TABLE_GRID 缺少 document region anchor")
            if not any(item.anchor_kind == ANCHOR_GRID_RECT for item in group):
                raise TableGridCarrierAdapterError("TABLE_GRID 缺少 grid rectangle anchor")
            if any(item.source != source or item.scope != scope
                   or item.envelope_identity != envelope.identity
                   or item.parser != parser for item in group):
                raise TableGridCarrierAdapterError("anchor context 漂移")
            nodes = tuple(item for item in self.structure_nodes
                          if item.envelope_identity == envelope.identity)
            if not nodes:
                raise TableGridCarrierAdapterError("TABLE_GRID 缺少 structure nodes")
            node_ids = {item.identity for item in nodes}
            anchor_ids = {item.identity for item in group
                          if item.anchor_kind == ANCHOR_GRID_RECT}
            if any(item.anchor_identity not in anchor_ids for item in nodes):
                raise TableGridCarrierAdapterError("structure node 未绑定 grid anchor")
            if any(item.parent_identity is not None
                   and item.parent_identity not in node_ids for item in nodes):
                raise TableGridCarrierAdapterError("structure node parent 逃逸")
            if tuple(item.ordinal for item in nodes) != tuple(
                    sorted(item.ordinal for item in nodes)):
                raise TableGridCarrierAdapterError("structure node ordinal 漂移")
            for artifact in (envelope, *group, *nodes):
                try:
                    if type(artifact).from_stable_key(artifact.stable_key()) != artifact:
                        raise TableGridCarrierAdapterError("对象无法稳定回读")
                except TableGridCarrierAdapterError:
                    raise
                except Exception as error:
                    raise TableGridCarrierAdapterError("对象无法稳定回读") from error

        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise TableGridCarrierAdapterError("revision parser lineage 漂移")
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source):
                raise TableGridCarrierAdapterError("revision envelope/hypothesis 漂移")

    def _anchors_for_envelope(self, identity: Any) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.envelope_identity == identity)


def _build_envelope(record: TableGridCarrierRecord, source: SourceRef,
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
    metas = _grid_metas(record, text)
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
    grid_anchors: dict[int, ArtifactAnchor] = {}
    for meta in metas:
        anchor = make_artifact_anchor(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_kind=ANCHOR_GRID_RECT, coordinates=meta.path,
            parser=parser, linked_text_anchor=None,
            anchor_key=_key(record, 31, variant, 3, meta.ordinal, *meta.path),
        )
        anchors.append(anchor)
        grid_anchors[meta.ordinal] = anchor

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
            anchor_identity=grid_anchors[meta.ordinal].identity,
            structure_family=family, node_kind=node_kind, role=None,
            parent_identity=parent, ordinal=ordinal_offset + meta.ordinal,
            qualifiers=tuple(meta.receipt),
            node_key=_key(record, 62, variant, meta.ordinal, *meta.path),
        ))
    return envelope, tuple(anchors), tuple(nodes)


def _revision(record: TableGridCarrierRecord, old: tuple[ArtifactAnchor, ...],
              new: tuple[ArtifactAnchor, ...], old_nodes: tuple[ArtifactStructureNode, ...],
              new_nodes: tuple[ArtifactStructureNode, ...], old_envelope: ArtifactEnvelope,
              new_envelope: ArtifactEnvelope, new_source: SourceRef,
              new_scope: ScopeIdentity) -> ArtifactCarrierRevision:
    new_grid: dict[tuple[int, ...], list[ArtifactAnchor]] = {}
    for item in new:
        if item.anchor_kind == ANCHOR_GRID_RECT:
            new_grid.setdefault(item.coordinates, []).append(item)
    grid_offsets: dict[tuple[int, ...], int] = {}
    mappings: list[ArtifactRevisionMapping] = []
    for item in old:
        if item.anchor_kind == ANCHOR_TEXT_RANGE:
            targets = tuple(x.identity for x in new if x.anchor_kind == ANCHOR_TEXT_RANGE)
        elif item.anchor_kind == ANCHOR_DOCUMENT_REGION:
            targets = tuple(x.identity for x in new if x.anchor_kind == ANCHOR_DOCUMENT_REGION)
        else:
            offset = grid_offsets.get(item.coordinates, 0)
            candidates = new_grid.get(item.coordinates, [])
            target = candidates[offset] if offset < len(candidates) else None
            grid_offsets[item.coordinates] = offset + 1
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
            raise TableGridCarrierAdapterError(
                "structure node grid receipt 损坏") from error

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


def adapt_table_grid_carrier_record(record: TableGridCarrierRecord) -> TableGridCarrierMaterialization:
    """不训练、不选义地把一个冻结 TableGrid payload 物化为 carrier 对象。"""
    if not isinstance(record, TableGridCarrierRecord):
        raise TableGridCarrierAdapterError("adapter 只接受 TableGridCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope = document_scope(source)
        envelope, anchors, nodes = _build_envelope(record, source, record.raw_text, 1)
        return TableGridCarrierMaterialization(
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
    return TableGridCarrierMaterialization(
        record, (old_source, new_source), (old_scope, new_scope),
        (old_envelope, new_envelope), (*old_anchors, *new_anchors),
        (*old_nodes, *new_nodes), (revision,),
    )


def _stable_lists(values: tuple[Any, ...]) -> list[list[int]]:
    return [list(item.stable_key()) for item in values]


def serialize_table_grid_carrier_materialization(
        materialization: TableGridCarrierMaterialization) -> bytes:
    """以 canonical JSON 保存全部共享对象的完整 stable keys。"""
    if not isinstance(materialization, TableGridCarrierMaterialization):
        raise TableGridCarrierAdapterError("serializer 输入类型非法")
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
        raise TableGridCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result: list[tuple[int, ...]] = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise TableGridCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_table_grid_carrier_materialization(
        payload: bytes, record: TableGridCarrierRecord) -> TableGridCarrierMaterialization:
    """严格回读 canonical bytes，并对照冻结 payload 重验对象图。"""
    if not isinstance(record, TableGridCarrierRecord):
        raise TableGridCarrierAdapterError("deserializer record 类型非法")
    try:
        if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
                or payload.endswith(b"\n\n")):
            raise TableGridCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        expected = {
            "anchors", "artifact_kind", "case_key", "envelopes",
            "format_version", "revisions", "sample_kind", "scopes",
            "sources", "structure_nodes",
        }
        if set(value) != expected:
            raise TableGridCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise TableGridCarrierAdapterError("materialization record 身份漂移")
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
        result = TableGridCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, nodes, revisions)
    except TableGridCarrierAdapterError:
        raise
    except Exception as error:
        raise TableGridCarrierAdapterError("materialization 损坏") from error
    if serialize_table_grid_carrier_materialization(result) != payload:
        raise TableGridCarrierAdapterError("materialization 不是 canonical 表示")
    return result


# 为兼容 LC-16 catalog/test 接口保留短名称。
serialize_table_grid_materialization = serialize_table_grid_carrier_materialization
deserialize_table_grid_materialization = deserialize_table_grid_carrier_materialization


__all__ = [
    "TABLE_GRID_SOURCE_KIND",
    "MATERIALIZATION_FORMAT_VERSION",
    "MATERIALIZATION_KIND",
    "TableGridCarrierAdapterError",
    "TableGridCarrierMaterialization",
    "adapt_table_grid_carrier_record",
    "deserialize_table_grid_materialization",
    "deserialize_table_grid_carrier_materialization",
    "serialize_table_grid_materialization",
    "serialize_table_grid_carrier_materialization",
]
