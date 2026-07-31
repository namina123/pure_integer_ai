"""LC-16 SOURCE_CODE payload 的确定性 SourceCode tree adapter。

该模块只负责把冻结的 :class:`SourceCodeCarrierRecord` 物化为共享
``ArtifactEnvelope``/``ArtifactAnchor``/``ArtifactStructureNode`` 对象。
它不作语义预选、不训练，也不把 SourceCode 的 render 结果当作原文；token
的完整 canonical receipt 保存在 structure node 的整数 ``qualifiers`` 中。
"""
from __future__ import annotations

import ast
import io
import tokenize
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
from pure_integer_ai.experiments.ph2_source_code_carrier_contract import (
    SourceCodeCarrierRecord,
    PARSER_VERSION,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_SOURCE_CODE_MATERIALIZATION"
SOURCE_CODE_SOURCE_KIND = 16616504
_IDENTITY_BASE = 16616520


class SourceCodeCarrierAdapterError(RuntimeError):
    """SOURCE_CODE adapter 输入、树对象或 canonical 表示不闭合。"""


def _key(record: SourceCodeCarrierRecord, domain: int, *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: SourceCodeCarrierRecord, parser_version: int) -> SourceRef:
    values = record.case_key.stable_key()
    case_index = values[-1]
    owner = OwnerScope(values[0], values[-2], case_index, VISIBILITY_SESSION)
    versions = VersionBundle(
        CorpusVersion(1), ParserVersion(parser_version), PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        SOURCE_CODE_SOURCE_KIND,
        values[0] + values[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: SourceCodeCarrierRecord, domain: int,
             *tail: int):
    return concept_identity(
        _key(record, domain, *tail), owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, record: SourceCodeCarrierRecord, domain: int,
               *tail: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain, *tail),
        _concept(source, record, domain + 1, *tail),
    )


def _parser(source: SourceRef, record: SourceCodeCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 20)


def _renderer(source: SourceRef, record: SourceCodeCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 22)


def _line_starts(text: str) -> tuple[int, ...]:
    starts = [0]
    for index, character in enumerate(text):
        if character == "\n":
            starts.append(index + 1)
    return tuple(starts)


@dataclass(frozen=True)
class _TokenMeta:
    ordinal: int
    path: tuple[int, ...]
    parent_ordinal: int | None
    span: tuple[int, int]
    receipt: bytes
    token_type: str
    nesting: int


def _position_offset(
        starts: tuple[int, ...], text: str, position: tuple[int, int]) -> int:
    line, column = position
    if line <= 0 or line > len(starts):
        return len(text)
    return min(len(text), starts[line - 1] + max(0, column))


def _receipt(
        *, family: str, kind: str, path: tuple[int, ...], ordinal: int,
        span: tuple[int, int], nesting: int, details: dict[str, Any]) -> bytes:
    return canonical_json_bytes({
        "details": details,
        "family": family,
        "language": "python",
        "nesting": nesting,
        "ordinal": ordinal,
        "parser": PARSER_VERSION,
        "path": list(path),
        "range": [span[0], span[1]],
        "type": kind,
    })


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return value
    if isinstance(value, bytes):
        return {"bytes": list(value)}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    # 浮点、复数和未来新增的 AST 标量字段只保存结构字符串；权威词形仍由
    # raw envelope 与 token 完整保留。
    return {"repr": repr(value), "value_type": type(value).__name__}


def _ast_scalar_fields(node: ast.AST) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            continue
        if isinstance(value, list) and any(isinstance(item, ast.AST)
                                           for item in value):
            continue
        result[name] = _json_scalar(value)
    return result


def _token_metas(text: str) -> tuple[_TokenMeta, ...]:
    starts = _line_starts(text)
    metas: list[_TokenMeta] = []
    token_failure: BaseException | None = None
    try:
        for token_index, token in enumerate(tokenize.generate_tokens(
                io.StringIO(text).readline)):
            ordinal = len(metas)
            path = (0, token_index)
            span = (
                _position_offset(starts, text, token.start),
                _position_offset(starts, text, token.end),
            )
            kind = tokenize.tok_name.get(token.type, str(token.type))
            details = {
                "end": [token.end[0], token.end[1]],
                "exact": token.string,
                "line": token.line,
                "start": [token.start[0], token.start[1]],
                "token_id": token.type,
            }
            metas.append(_TokenMeta(
                ordinal, path, None, span,
                _receipt(
                    family="TOKEN", kind=kind, path=path, ordinal=ordinal,
                    span=span, nesting=0, details=details),
                f"TOKEN:{kind}", 0,
            ))
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        token_failure = error

    tree: ast.AST | None = None
    ast_failure: SyntaxError | None = None
    try:
        tree = ast.parse(text, filename="<lc16-source-code>", type_comments=True)
    except SyntaxError as error:
        ast_failure = error

    def emit_ast(node: ast.AST, path: tuple[int, ...],
                 parent_ordinal: int | None, depth: int) -> None:
        ordinal = len(metas)
        start = _position_offset(starts, text, (
            int(getattr(node, "lineno", 1)),
            int(getattr(node, "col_offset", 0)),
        ))
        end = _position_offset(starts, text, (
            int(getattr(node, "end_lineno", len(starts))),
            int(getattr(node, "end_col_offset", 0)),
        ))
        if not hasattr(node, "lineno"):
            start, end = 0, len(text)
        span = (min(start, end), max(start, end))
        kind = type(node).__name__
        details = {
            "field_names": list(getattr(node, "_fields", ())),
            "scalar_fields": _ast_scalar_fields(node),
        }
        metas.append(_TokenMeta(
            ordinal, path, parent_ordinal, span,
            _receipt(
                family="AST", kind=kind, path=path, ordinal=ordinal,
                span=span, nesting=depth, details=details),
            f"AST:{kind}", depth,
        ))
        for child_index, child in enumerate(ast.iter_child_nodes(node)):
            emit_ast(child, (*path, child_index), ordinal, depth + 1)

    if tree is not None:
        emit_ast(tree, (1,), None, 0)

    def emit_state(index: int, kind: str, details: dict[str, Any]) -> None:
        ordinal = len(metas)
        path = (2, index)
        span = (0, len(text))
        metas.append(_TokenMeta(
            ordinal, path, None, span,
            _receipt(
                family="PARSER_STATE", kind=kind, path=path,
                ordinal=ordinal, span=span, nesting=0, details=details),
            f"PARSER_STATE:{kind}", 0,
        ))

    if token_failure is None:
        emit_state(0, "TOKENIZE_OK", {})
    else:
        emit_state(0, "TOKENIZE_ERROR", {
            "error_type": type(token_failure).__name__,
            "message": str(token_failure),
        })
    if ast_failure is None:
        emit_state(1, "AST_OK", {})
    else:
        emit_state(1, "AST_SYNTAX_ERROR", {
            "column": int(ast_failure.offset or 0),
            "line": int(ast_failure.lineno or 0),
            "message": str(ast_failure.msg),
        })
    return tuple(metas)


@dataclass(frozen=True)
class SourceCodeCarrierMaterialization:
    """一个 SOURCE_CODE case 的完整 envelope、anchor、tree node 与 revision。"""

    record: SourceCodeCarrierRecord
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
        if not isinstance(self.record, SourceCodeCarrierRecord):
            raise SourceCodeCarrierAdapterError("materialization record 类型非法")
        expected_count = 2 if self.record.sample_kind == "REVISION" else 1
        for name, cls in (
                ("sources", SourceRef), ("scopes", ScopeIdentity),
                ("envelopes", ArtifactEnvelope), ("anchors", ArtifactAnchor),
                ("structure_nodes", ArtifactStructureNode)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or len(values) == 0
                    or any(not isinstance(item, cls) for item in values)):
                raise SourceCodeCarrierAdapterError(f"materialization {name} 类型非法")
        if any(len(group) == 0 for group in (self.sources, self.scopes, self.envelopes)):
            raise SourceCodeCarrierAdapterError("materialization 顶层对象不能为空")
        if len(self.sources) != expected_count or len(self.scopes) != expected_count:
            raise SourceCodeCarrierAdapterError("materialization source/scope 数量漂移")
        if len(self.envelopes) != expected_count:
            raise SourceCodeCarrierAdapterError("materialization envelope 数量漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise SourceCodeCarrierAdapterError("materialization revisions 数量漂移")

        expected_texts = ((self.record.previous_text, self.record.raw_text)
                          if expected_count == 2 else (self.record.raw_text,))
        for index, (source, scope, envelope, text) in enumerate(zip(
                self.sources, self.scopes, self.envelopes, expected_texts)):
            parser = _parser(source, self.record)
            if scope != document_scope(source):
                raise SourceCodeCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise SourceCodeCarrierAdapterError("envelope raw/source 漂移")
            group = self._anchors_for_envelope(envelope.identity)
            if len(group) < 3:
                raise SourceCodeCarrierAdapterError("SOURCE_CODE 缺少 text/document/tree anchors")
            if not any(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group):
                raise SourceCodeCarrierAdapterError("SOURCE_CODE 缺少 full text anchor")
            if not any(item.anchor_kind == ANCHOR_DOCUMENT_REGION for item in group):
                raise SourceCodeCarrierAdapterError("SOURCE_CODE 缺少 document region anchor")
            if any(item.source != source or item.scope != scope
                   or item.envelope_identity != envelope.identity
                   or item.parser != parser for item in group):
                raise SourceCodeCarrierAdapterError("anchor context 漂移")
            nodes = tuple(item for item in self.structure_nodes
                          if item.envelope_identity == envelope.identity)
            if not nodes:
                raise SourceCodeCarrierAdapterError("SOURCE_CODE 缺少 structure nodes")
            node_ids = {item.identity for item in nodes}
            anchor_ids = {item.identity for item in group
                          if item.anchor_kind == ANCHOR_TREE_PATH}
            if any(item.anchor_identity not in anchor_ids for item in nodes):
                raise SourceCodeCarrierAdapterError("structure node 未绑定 tree anchor")
            if any(item.parent_identity is not None
                   and item.parent_identity not in node_ids for item in nodes):
                raise SourceCodeCarrierAdapterError("structure node parent 逃逸")
            if tuple(item.ordinal for item in nodes) != tuple(
                    sorted(item.ordinal for item in nodes)):
                raise SourceCodeCarrierAdapterError("structure node ordinal 漂移")
            for artifact in (envelope, *group, *nodes):
                try:
                    if type(artifact).from_stable_key(artifact.stable_key()) != artifact:
                        raise SourceCodeCarrierAdapterError("对象无法稳定回读")
                except SourceCodeCarrierAdapterError:
                    raise
                except Exception as error:
                    raise SourceCodeCarrierAdapterError("对象无法稳定回读") from error

        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise SourceCodeCarrierAdapterError("revision parser lineage 漂移")
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source):
                raise SourceCodeCarrierAdapterError("revision envelope/hypothesis 漂移")

    def _anchors_for_envelope(self, identity: Any) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.envelope_identity == identity)


def _build_envelope(record: SourceCodeCarrierRecord, source: SourceRef,
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


def _revision(record: SourceCodeCarrierRecord, old: tuple[ArtifactAnchor, ...],
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
            raise SourceCodeCarrierAdapterError(
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


def adapt_source_code_carrier_record(record: SourceCodeCarrierRecord) -> SourceCodeCarrierMaterialization:
    """不训练、不选义地把一个冻结 SourceCode payload 物化为 carrier 对象。"""
    if not isinstance(record, SourceCodeCarrierRecord):
        raise SourceCodeCarrierAdapterError("adapter 只接受 SourceCodeCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope = document_scope(source)
        envelope, anchors, nodes = _build_envelope(record, source, record.raw_text, 1)
        return SourceCodeCarrierMaterialization(
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
    return SourceCodeCarrierMaterialization(
        record, (old_source, new_source), (old_scope, new_scope),
        (old_envelope, new_envelope), (*old_anchors, *new_anchors),
        (*old_nodes, *new_nodes), (revision,),
    )


def _stable_lists(values: tuple[Any, ...]) -> list[list[int]]:
    return [list(item.stable_key()) for item in values]


def serialize_source_code_carrier_materialization(
        materialization: SourceCodeCarrierMaterialization) -> bytes:
    """以 canonical JSON 保存全部共享对象的完整 stable keys。"""
    if not isinstance(materialization, SourceCodeCarrierMaterialization):
        raise SourceCodeCarrierAdapterError("serializer 输入类型非法")
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
        raise SourceCodeCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result: list[tuple[int, ...]] = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise SourceCodeCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_source_code_carrier_materialization(
        payload: bytes, record: SourceCodeCarrierRecord) -> SourceCodeCarrierMaterialization:
    """严格回读 canonical bytes，并对照冻结 payload 重验对象图。"""
    if not isinstance(record, SourceCodeCarrierRecord):
        raise SourceCodeCarrierAdapterError("deserializer record 类型非法")
    try:
        if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
                or payload.endswith(b"\n\n")):
            raise SourceCodeCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        expected = {
            "anchors", "artifact_kind", "case_key", "envelopes",
            "format_version", "revisions", "sample_kind", "scopes",
            "sources", "structure_nodes",
        }
        if set(value) != expected:
            raise SourceCodeCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise SourceCodeCarrierAdapterError("materialization record 身份漂移")
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
        result = SourceCodeCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, nodes, revisions)
    except SourceCodeCarrierAdapterError:
        raise
    except Exception as error:
        raise SourceCodeCarrierAdapterError("materialization 损坏") from error
    if serialize_source_code_carrier_materialization(result) != payload:
        raise SourceCodeCarrierAdapterError("materialization 不是 canonical 表示")
    return result


# 为兼容 LC-16 catalog/test 接口保留短名称。
serialize_source_code_materialization = serialize_source_code_carrier_materialization
deserialize_source_code_materialization = deserialize_source_code_carrier_materialization


__all__ = [
    "SOURCE_CODE_SOURCE_KIND",
    "MATERIALIZATION_FORMAT_VERSION",
    "MATERIALIZATION_KIND",
    "SourceCodeCarrierAdapterError",
    "SourceCodeCarrierMaterialization",
    "adapt_source_code_carrier_record",
    "deserialize_source_code_materialization",
    "deserialize_source_code_carrier_materialization",
    "serialize_source_code_materialization",
    "serialize_source_code_carrier_materialization",
]
