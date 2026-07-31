"""LC-16 OCR/ASR 转写 payload 到共享 carrier 对象的确定性 adapter。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TRANSCRIPT_ALIGNMENT,
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
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity, document_scope
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_contract import (
    PARSER_VERSION,
    TranscriptSegment,
    TranscribedOcrAsrCarrierRecord,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_TRANSCRIBED_OCR_ASR_MATERIALIZATION"
TRANSCRIBED_OCR_ASR_SOURCE_KIND = 16616508
_IDENTITY_BASE = 16616580


class TranscribedOcrAsrCarrierAdapterError(RuntimeError):
    """转写 adapter 输入、对象图或 canonical 表示不闭合。"""


def _key(record: TranscribedOcrAsrCarrierRecord, domain: int,
         *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: TranscribedOcrAsrCarrierRecord,
            parser_version: int) -> SourceRef:
    values = record.case_key.stable_key()
    case_index = values[-1]
    owner = OwnerScope(values[0], values[-2], case_index, VISIBILITY_SESSION)
    versions = VersionBundle(
        CorpusVersion(1), ParserVersion(parser_version), PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        TRANSCRIBED_OCR_ASR_SOURCE_KIND,
        values[0] + values[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: TranscribedOcrAsrCarrierRecord,
             domain: int, *tail: int):
    return concept_identity(
        _key(record, domain, *tail), owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, record: TranscribedOcrAsrCarrierRecord,
               domain: int, *tail: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain, *tail),
        _concept(source, record, domain + 1, *tail),
    )


def _parser(source: SourceRef, record: TranscribedOcrAsrCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 20)


def _renderer(source: SourceRef, record: TranscribedOcrAsrCarrierRecord) -> ArtifactAuthority:
    return _authority(source, record, 22)


@dataclass(frozen=True)
class _SegmentMeta:
    ordinal: int
    segment_id: int | None
    text_span: tuple[int, int]
    time_span: tuple[int, int]
    receipt: bytes
    node_type: str


def _receipt(*, family: str, kind: str, ordinal: int,
             details: dict[str, Any]) -> bytes:
    return canonical_json_bytes({
        "details": details,
        "family": family,
        "language": "transcribed-ocr-asr",
        "ordinal": ordinal,
        "parser": PARSER_VERSION,
        "type": kind,
    })


def _segment_metas(record: TranscribedOcrAsrCarrierRecord,
                   text: str,
                   segments: tuple[TranscriptSegment, ...]) -> tuple[_SegmentMeta, ...]:
    metas: list[_SegmentMeta] = []
    for segment in segments:
        metas.append(_SegmentMeta(
            segment.ordinal,
            segment.segment_id,
            (segment.text_start, segment.text_end),
            (segment.time_start_ms, segment.time_end_ms),
            _receipt(
                family="TRANSCRIPT",
                kind="SEGMENT",
                ordinal=segment.ordinal,
                details={
                    "confidence_candidates": list(segment.confidence_candidates),
                    "segment_id": segment.segment_id,
                    "source_mode": segment.source_mode,
                    "speaker_candidates": list(segment.speaker_candidates),
                    "temporal_state": segment.temporal_state,
                    "text": text[segment.text_start:segment.text_end],
                    "text_range": [segment.text_start, segment.text_end],
                    "time_ms": [segment.time_start_ms, segment.time_end_ms],
                },
            ),
            "TRANSCRIPT:SEGMENT",
        ))
    modes = sorted({item.source_mode for item in segments})
    state = "UNKNOWN_SOURCE_MODE" if "UNKNOWN" in modes else "ALIGNMENT_OK"
    metas.append(_SegmentMeta(
        len(metas), None, (0, len(text)), (0, 0),
        _receipt(
            family="PARSER_STATE",
            kind=state,
            ordinal=len(metas),
            details={"segment_count": len(segments), "source_modes": modes},
        ),
        f"PARSER_STATE:{state}",
    ))
    return tuple(metas)


@dataclass(frozen=True)
class TranscribedOcrAsrCarrierMaterialization:
    record: TranscribedOcrAsrCarrierRecord
    sources: tuple[SourceRef, ...]
    scopes: tuple[ScopeIdentity, ...]
    envelopes: tuple[ArtifactEnvelope, ...]
    anchors: tuple[ArtifactAnchor, ...]
    structure_nodes: tuple[ArtifactStructureNode, ...]
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
    def transcript_alignment_anchors(self) -> tuple[ArtifactAnchor, ...]:
        return tuple(item for item in self.anchors
                     if item.anchor_kind == ANCHOR_TRANSCRIPT_ALIGNMENT)

    def __post_init__(self) -> None:
        if not isinstance(self.record, TranscribedOcrAsrCarrierRecord):
            raise TranscribedOcrAsrCarrierAdapterError("materialization record 类型非法")
        expected_count = 2 if self.record.sample_kind == "REVISION" else 1
        for name, cls in (
                ("sources", SourceRef), ("scopes", ScopeIdentity),
                ("envelopes", ArtifactEnvelope), ("anchors", ArtifactAnchor),
                ("structure_nodes", ArtifactStructureNode)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(item, cls) for item in values)):
                raise TranscribedOcrAsrCarrierAdapterError(
                    f"materialization {name} 类型非法")
        if (len(self.sources) != expected_count
                or len(self.scopes) != expected_count
                or len(self.envelopes) != expected_count):
            raise TranscribedOcrAsrCarrierAdapterError("materialization 顶层数量漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise TranscribedOcrAsrCarrierAdapterError("materialization revisions 数量漂移")
        expected_texts = ((self.record.previous_text, self.record.raw_text)
                          if expected_count == 2 else (self.record.raw_text,))
        expected_segments = ((self.record.previous_segments, self.record.segments)
                             if expected_count == 2 else (self.record.segments,))
        for source, scope, envelope, text, segments in zip(
                self.sources, self.scopes, self.envelopes, expected_texts,
                expected_segments):
            parser = _parser(source, self.record)
            if scope != document_scope(source):
                raise TranscribedOcrAsrCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise TranscribedOcrAsrCarrierAdapterError("envelope raw/source 漂移")
            group = tuple(item for item in self.anchors
                          if item.envelope_identity == envelope.identity)
            if (sum(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group) != 1
                    or sum(item.anchor_kind == ANCHOR_DOCUMENT_REGION
                           for item in group) != 1
                    or not any(item.anchor_kind == ANCHOR_TRANSCRIPT_ALIGNMENT
                               for item in group)):
                raise TranscribedOcrAsrCarrierAdapterError(
                    "TRANSCRIBED_OCR_ASR anchors 不完整")
            if any(item.source != source or item.scope != scope
                   or item.envelope_identity != envelope.identity
                   or item.parser != parser for item in group):
                raise TranscribedOcrAsrCarrierAdapterError("anchor context 漂移")
            nodes = tuple(item for item in self.structure_nodes
                          if item.envelope_identity == envelope.identity)
            if not nodes:
                raise TranscribedOcrAsrCarrierAdapterError("缺少 structure nodes")
            anchor_ids = {item.identity for item in group}
            if any(item.anchor_identity not in anchor_ids for item in nodes):
                raise TranscribedOcrAsrCarrierAdapterError("structure node 未绑定 local anchor")
            if tuple(item.ordinal for item in nodes) != tuple(
                    sorted(item.ordinal for item in nodes)):
                raise TranscribedOcrAsrCarrierAdapterError("structure node ordinal 漂移")
            if sum(item.anchor_kind == ANCHOR_TRANSCRIPT_ALIGNMENT for item in group) != len(segments):
                raise TranscribedOcrAsrCarrierAdapterError("alignment anchor 数量漂移")
            for artifact in (envelope, *group, *nodes):
                try:
                    if type(artifact).from_stable_key(artifact.stable_key()) != artifact:
                        raise TranscribedOcrAsrCarrierAdapterError("对象无法稳定回读")
                except TranscribedOcrAsrCarrierAdapterError:
                    raise
                except Exception as error:
                    raise TranscribedOcrAsrCarrierAdapterError("对象无法稳定回读") from error
        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise TranscribedOcrAsrCarrierAdapterError("revision parser lineage 漂移")
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source):
                raise TranscribedOcrAsrCarrierAdapterError("revision envelope/hypothesis 漂移")


def _build_envelope(record: TranscribedOcrAsrCarrierRecord, source: SourceRef,
                    text: str, segments: tuple[TranscriptSegment, ...],
                    variant: int, *, ordinal_offset: int = 0) -> tuple[ArtifactEnvelope, tuple[ArtifactAnchor, ...],
                                             tuple[ArtifactStructureNode, ...]]:
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
    text_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_TEXT_RANGE, coordinates=(0, len(text)), parser=parser,
        linked_text_anchor=None, anchor_key=_key(record, 31, variant, 1),
    )
    document_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity, source=source, scope=scope,
        anchor_kind=ANCHOR_DOCUMENT_REGION, coordinates=(0, len(text)), parser=parser,
        linked_text_anchor=None, anchor_key=_key(record, 31, variant, 2),
    )
    anchors = [text_anchor, document_anchor]
    alignment_anchors: dict[int, ArtifactAnchor] = {}
    for segment in segments:
        anchor = make_artifact_anchor(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_kind=ANCHOR_TRANSCRIPT_ALIGNMENT,
            coordinates=(segment.time_start_ms, segment.time_end_ms),
            parser=parser, linked_text_anchor=None,
            anchor_key=_key(record, 31, variant, 3, segment.segment_id),
        )
        anchors.append(anchor)
        alignment_anchors[segment.segment_id] = anchor
    state_anchor = document_anchor
    family = structure_concept_identity(
        _key(record, 60), owner=source.owner, versions=source.versions)
    nodes: list[ArtifactStructureNode] = []
    for meta in _segment_metas(record, text, segments):
        if meta.segment_id is None:
            anchor = state_anchor
        else:
            anchor = alignment_anchors[meta.segment_id]
        node_kind = structure_concept_identity(
            _key(record, 61, *tuple(meta.node_type.encode("utf-8"))),
            owner=source.owner, versions=source.versions)
        nodes.append(make_artifact_structure_node(
            envelope_identity=envelope.identity, source=source, scope=scope,
            anchor_identity=anchor.identity, structure_family=family,
            node_kind=node_kind, role=None, parent_identity=None,
            ordinal=ordinal_offset + meta.ordinal, qualifiers=tuple(meta.receipt),
            node_key=_key(record, 62, variant, meta.ordinal,
                         0 if meta.segment_id is None else meta.segment_id),
        ))
    return envelope, tuple(anchors), tuple(nodes)


def _revision(record: TranscribedOcrAsrCarrierRecord,
              old_anchors: tuple[ArtifactAnchor, ...],
              new_anchors: tuple[ArtifactAnchor, ...],
              old_nodes: tuple[ArtifactStructureNode, ...],
              new_nodes: tuple[ArtifactStructureNode, ...],
              old_envelope: ArtifactEnvelope, new_envelope: ArtifactEnvelope,
              new_source: SourceRef, new_scope: ScopeIdentity) -> ArtifactCarrierRevision:
    new_alignment = {}
    for item in new_anchors:
        if item.anchor_kind == ANCHOR_TRANSCRIPT_ALIGNMENT:
            new_alignment.setdefault(item.anchor_key[-1], []).append(item)
    mappings: list[ArtifactRevisionMapping] = []
    for item in old_anchors:
        if item.anchor_kind in {ANCHOR_TEXT_RANGE, ANCHOR_DOCUMENT_REGION}:
            targets = tuple(x.identity for x in new_anchors
                            if x.anchor_kind == item.anchor_kind)
        else:
            key = item.anchor_key[-1]
            targets = tuple(x.identity for x in new_alignment.get(key, ()))
        mappings.append(ArtifactRevisionMapping(REVISION_MAP_ANCHOR,
                                                item.identity, targets))

    def signature(node: ArtifactStructureNode) -> tuple[Any, ...]:
        receipt = parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        details = receipt["details"]
        return (receipt["type"], details.get("segment_id"))

    new_by_signature = {signature(node): node for node in new_nodes}
    for node in old_nodes:
        target = new_by_signature.get(signature(node))
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


def adapt_transcribed_ocr_asr_carrier_record(
        record: TranscribedOcrAsrCarrierRecord,
        ) -> TranscribedOcrAsrCarrierMaterialization:
    """不训练、不推断，把已产生的 OCR/ASR 转写物化为来源化结构。"""
    if not isinstance(record, TranscribedOcrAsrCarrierRecord):
        raise TranscribedOcrAsrCarrierAdapterError(
            "adapter 只接受 TranscribedOcrAsrCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope = document_scope(source)
        envelope, anchors, nodes = _build_envelope(
            record, source, record.raw_text, record.segments, 1)
        return TranscribedOcrAsrCarrierMaterialization(
            record, (source,), (scope,), (envelope,), anchors, nodes, ())
    old_source = _source(record, 1)
    new_source = _source(record, 2)
    old_scope = document_scope(old_source)
    new_scope = document_scope(new_source)
    old_envelope, old_anchors, old_nodes = _build_envelope(
        record, old_source, record.previous_text, record.previous_segments, 1)
    new_envelope, new_anchors, new_nodes = _build_envelope(
        record, new_source, record.raw_text, record.segments, 2,
        ordinal_offset=len(old_nodes))
    revision = _revision(
        record, old_anchors, new_anchors, old_nodes, new_nodes,
        old_envelope, new_envelope, new_source, new_scope)
    return TranscribedOcrAsrCarrierMaterialization(
        record, (old_source, new_source), (old_scope, new_scope),
        (old_envelope, new_envelope), (*old_anchors, *new_anchors),
        (*old_nodes, *new_nodes), (revision,))


def _stable_lists(values: tuple[Any, ...]) -> list[list[int]]:
    return [list(item.stable_key()) for item in values]


def serialize_transcribed_ocr_asr_carrier_materialization(
        materialization: TranscribedOcrAsrCarrierMaterialization) -> bytes:
    if not isinstance(materialization, TranscribedOcrAsrCarrierMaterialization):
        raise TranscribedOcrAsrCarrierAdapterError("serializer 输入类型非法")
    return canonical_json_bytes({
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
    }) + b"\n"


def _strict_stable_keys(value: Any, *, where: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise TranscribedOcrAsrCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise TranscribedOcrAsrCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_transcribed_ocr_asr_carrier_materialization(
        payload: bytes, record: TranscribedOcrAsrCarrierRecord,
        ) -> TranscribedOcrAsrCarrierMaterialization:
    if not isinstance(record, TranscribedOcrAsrCarrierRecord):
        raise TranscribedOcrAsrCarrierAdapterError("deserializer record 类型非法")
    try:
        if not isinstance(payload, bytes) or not payload.endswith(b"\n") \
                or payload.endswith(b"\n\n"):
            raise TranscribedOcrAsrCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        expected = {
            "anchors", "artifact_kind", "case_key", "envelopes", "format_version",
            "revisions", "sample_kind", "scopes", "sources", "structure_nodes",
        }
        if set(value) != expected:
            raise TranscribedOcrAsrCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise TranscribedOcrAsrCarrierAdapterError("materialization record 身份漂移")
        sources = tuple(SourceRef.from_stable_key(item)
                        for item in _strict_stable_keys(value["sources"], where="sources"))
        scopes = tuple(ScopeIdentity.from_stable_key(item)
                       for item in _strict_stable_keys(value["scopes"], where="scopes"))
        envelopes = tuple(ArtifactEnvelope.from_stable_key(item)
                          for item in _strict_stable_keys(value["envelopes"], where="envelopes"))
        anchors = tuple(ArtifactAnchor.from_stable_key(item)
                        for item in _strict_stable_keys(value["anchors"], where="anchors"))
        nodes = tuple(ArtifactStructureNode.from_stable_key(item)
                      for item in _strict_stable_keys(value["structure_nodes"], where="structure_nodes"))
        revisions = tuple(ArtifactCarrierRevision.from_stable_key(item)
                          for item in _strict_stable_keys(value["revisions"], where="revisions"))
        result = TranscribedOcrAsrCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, nodes, revisions)
    except TranscribedOcrAsrCarrierAdapterError:
        raise
    except Exception as error:
        raise TranscribedOcrAsrCarrierAdapterError("materialization 损坏") from error
    if serialize_transcribed_ocr_asr_carrier_materialization(result) != payload:
        raise TranscribedOcrAsrCarrierAdapterError("materialization 不是 canonical 表示")
    return result


serialize_transcribed_ocr_asr_carrier = serialize_transcribed_ocr_asr_carrier_materialization
deserialize_transcribed_ocr_asr_carrier = deserialize_transcribed_ocr_asr_carrier_materialization


__all__ = [
    "MATERIALIZATION_FORMAT_VERSION", "MATERIALIZATION_KIND",
    "TRANSCRIBED_OCR_ASR_SOURCE_KIND",
    "TranscribedOcrAsrCarrierAdapterError",
    "TranscribedOcrAsrCarrierMaterialization",
    "adapt_transcribed_ocr_asr_carrier_record",
    "deserialize_transcribed_ocr_asr_carrier",
    "deserialize_transcribed_ocr_asr_carrier_materialization",
    "serialize_transcribed_ocr_asr_carrier",
    "serialize_transcribed_ocr_asr_carrier_materialization",
]
