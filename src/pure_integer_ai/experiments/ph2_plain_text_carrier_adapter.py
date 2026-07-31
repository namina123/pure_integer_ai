"""LC-16 PLAIN_TEXT payload 到 ArtifactEnvelope 的薄确定性 adapter。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_TEXT_RANGE,
    RAW_UNIT_UNICODE_SCALAR,
    REVISION_MAP_ANCHOR,
    ArtifactAnchor,
    ArtifactCarrierRevision,
    ArtifactEnvelope,
    ArtifactRevisionMapping,
    make_artifact_anchor,
    make_artifact_carrier_revision,
    make_artifact_envelope,
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
from pure_integer_ai.experiments.ph2_plain_text_carrier_contract import (
    PlainTextCarrierRecord,
)


MATERIALIZATION_FORMAT_VERSION = 1
MATERIALIZATION_KIND = "PH2_LC16_PLAIN_TEXT_MATERIALIZATION"
PLAIN_TEXT_SOURCE_KIND = 16616501
_IDENTITY_BASE = 16616510


class PlainTextCarrierAdapterError(RuntimeError):
    """PLAIN_TEXT adapter 输入、对象图或 canonical 表示不闭合。"""


def _key(record: PlainTextCarrierRecord, domain: int, *tail: int) -> tuple[int, ...]:
    return (*record.case_key.stable_key(), _IDENTITY_BASE, domain, *tail)


def _source(record: PlainTextCarrierRecord, parser_version: int) -> SourceRef:
    case_index = record.case_key.stable_key()[-1]
    owner = OwnerScope(
        record.case_key.stable_key()[0],
        record.case_key.stable_key()[-2],
        case_index,
        VISIBILITY_SESSION,
    )
    versions = VersionBundle(
        CorpusVersion(1),
        ParserVersion(parser_version),
        PrimitiveVersion(1),
        CurriculumVersion(1),
    )
    return SourceRef(
        PLAIN_TEXT_SOURCE_KIND,
        record.case_key.stable_key()[0] + record.case_key.stable_key()[-2],
        case_index,
        owner,
        versions,
    )


def _concept(source: SourceRef, record: PlainTextCarrierRecord, domain: int):
    return concept_identity(
        _key(record, domain), owner=source.owner, versions=source.versions)


def _authority(
        source: SourceRef,
        record: PlainTextCarrierRecord,
        domain: int,
        ) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, record, domain),
        _concept(source, record, domain + 1),
    )


def _envelope_and_anchor(
        record: PlainTextCarrierRecord,
        *,
        source: SourceRef,
        text: str,
        key_variant: int,
        ) -> tuple[ScopeIdentity, ArtifactEnvelope, ArtifactAnchor]:
    scope = document_scope(source)
    parser = _authority(source, record, 20)
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
        renderer=_authority(source, record, 22),
        envelope_key=_key(record, 30, key_variant),
    )
    anchor = make_artifact_anchor(
        envelope_identity=envelope.identity,
        source=source,
        scope=scope,
        anchor_kind=ANCHOR_TEXT_RANGE,
        coordinates=(0, len(text)),
        parser=parser,
        linked_text_anchor=None,
        anchor_key=_key(record, 31, key_variant),
    )
    return scope, envelope, anchor


@dataclass(frozen=True)
class PlainTextCarrierMaterialization:
    """一个 payload 的直接 Artifact carrier 对象集合。"""

    record: PlainTextCarrierRecord
    sources: tuple[SourceRef, ...]
    scopes: tuple[ScopeIdentity, ...]
    envelopes: tuple[ArtifactEnvelope, ...]
    anchors: tuple[ArtifactAnchor, ...]
    revisions: tuple[ArtifactCarrierRevision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.record, PlainTextCarrierRecord):
            raise PlainTextCarrierAdapterError("materialization record 类型非法")
        expected_count = 2 if self.record.sample_kind == "REVISION" else 1
        for name, cls in (
                ("sources", SourceRef),
                ("scopes", ScopeIdentity),
                ("envelopes", ArtifactEnvelope),
                ("anchors", ArtifactAnchor)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or len(values) != expected_count
                    or any(not isinstance(item, cls) for item in values)):
                raise PlainTextCarrierAdapterError(
                    f"materialization {name} 数量或类型漂移")
        expected_revision_count = 1 if expected_count == 2 else 0
        if (not isinstance(self.revisions, tuple)
                or len(self.revisions) != expected_revision_count
                or any(not isinstance(item, ArtifactCarrierRevision)
                       for item in self.revisions)):
            raise PlainTextCarrierAdapterError(
                "materialization revisions 数量或类型漂移")

        texts = ((self.record.previous_text, self.record.raw_text)
                 if expected_count == 2 else (self.record.raw_text,))
        for index, (source, scope, envelope, anchor, text) in enumerate(zip(
                self.sources, self.scopes, self.envelopes, self.anchors, texts)):
            if scope != document_scope(source):
                raise PlainTextCarrierAdapterError("document_scope 漂移")
            if (envelope.source != source or envelope.scope != scope
                    or envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR
                    or envelope.raw_units != tuple(ord(item) for item in text)):
                raise PlainTextCarrierAdapterError("envelope raw/source 漂移")
            if (anchor.source != source or anchor.scope != scope
                    or anchor.envelope_identity != envelope.identity
                    or anchor.anchor_kind != ANCHOR_TEXT_RANGE
                    or anchor.coordinates != (0, len(text))):
                raise PlainTextCarrierAdapterError("full-range anchor 漂移")
            if (ArtifactEnvelope.from_stable_key(envelope.stable_key())
                    != envelope
                    or ArtifactAnchor.from_stable_key(anchor.stable_key())
                    != anchor):
                raise PlainTextCarrierAdapterError(
                    f"materialization 对象 {index} 无法稳定回读")

        if expected_count == 2:
            old_source, new_source = self.sources
            revision = self.revisions[0]
            if (parser_lineage_key(old_source) != parser_lineage_key(new_source)
                    or old_source.versions.parser == new_source.versions.parser):
                raise PlainTextCarrierAdapterError("revision parser lineage 漂移")
            mapping = revision.mappings
            if (revision.old_envelope_identity != self.envelopes[0].identity
                    or revision.new_envelope_identity != self.envelopes[1].identity
                    or revision.hypothesis.observation != new_source
                    or len(mapping) != 1
                    or mapping[0].mapping_kind != REVISION_MAP_ANCHOR
                    or mapping[0].old_identity != self.anchors[0].identity
                    or mapping[0].new_identities != (self.anchors[1].identity,)):
                raise PlainTextCarrierAdapterError("revision mapping 漂移")
            if (ArtifactCarrierRevision.from_stable_key(revision.stable_key())
                    != revision):
                raise PlainTextCarrierAdapterError("revision 无法稳定回读")


def adapt_plain_text_carrier_record(
        record: PlainTextCarrierRecord,
        ) -> PlainTextCarrierMaterialization:
    """不训练、不选义地把一个冻结 raw payload 物化为 carrier 对象。"""
    if not isinstance(record, PlainTextCarrierRecord):
        raise PlainTextCarrierAdapterError("adapter 只接受 PlainTextCarrierRecord")
    if record.sample_kind != "REVISION":
        source = _source(record, 1)
        scope, envelope, anchor = _envelope_and_anchor(
            record, source=source, text=record.raw_text, key_variant=1)
        return PlainTextCarrierMaterialization(
            record, (source,), (scope,), (envelope,), (anchor,), ())

    old_source = _source(record, 1)
    new_source = _source(record, 2)
    old_scope, old_envelope, old_anchor = _envelope_and_anchor(
        record, source=old_source, text=record.previous_text, key_variant=1)
    new_scope, new_envelope, new_anchor = _envelope_and_anchor(
        record, source=new_source, text=record.raw_text, key_variant=2)
    hypothesis = HypothesisKey(
        _key(record, 40),
        _key(record, 41),
        _key(record, 42),
        new_scope,
        new_source,
    )
    evidence = EvidenceRecord(
        record.case_key.stable_key()[-1],
        hypothesis,
        EVIDENCE_SUPPORT,
        _key(record, 43),
        new_source,
        1,
        _key(record, 44),
    )
    mapping = ArtifactRevisionMapping(
        REVISION_MAP_ANCHOR,
        old_anchor.identity,
        (new_anchor.identity,),
    )
    revision = make_artifact_carrier_revision(
        old_envelope_identity=old_envelope.identity,
        new_envelope_identity=new_envelope.identity,
        reason=_concept(new_source, record, 45),
        hypothesis=hypothesis,
        mappings=(mapping,),
        evidence=(evidence,),
        revision_key=_key(record, 46),
    )
    return PlainTextCarrierMaterialization(
        record,
        (old_source, new_source),
        (old_scope, new_scope),
        (old_envelope, new_envelope),
        (old_anchor, new_anchor),
        (revision,),
    )


def _stable_lists(values: tuple[Any, ...]) -> list[list[int]]:
    return [list(item.stable_key()) for item in values]


def serialize_plain_text_materialization(
        materialization: PlainTextCarrierMaterialization,
        ) -> bytes:
    """以 canonical JSON 完整保存 Source/scope/carrier stable keys。"""
    if not isinstance(materialization, PlainTextCarrierMaterialization):
        raise PlainTextCarrierAdapterError("serializer 输入类型非法")
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
    }
    return canonical_json_bytes(value) + b"\n"


def _strict_stable_keys(
        value: Any,
        *,
        where: str,
        ) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise PlainTextCarrierAdapterError(f"{where} 必须是 stable key 列表")
    result: list[tuple[int, ...]] = []
    for item in value:
        if (not isinstance(item, list) or not item
                or any(type(number) is not int for number in item)):
            raise PlainTextCarrierAdapterError(f"{where} stable key 非法")
        result.append(tuple(item))
    return tuple(result)


def deserialize_plain_text_materialization(
        payload: bytes,
        record: PlainTextCarrierRecord,
        ) -> PlainTextCarrierMaterialization:
    """严格回读 canonical bytes，并对照冻结 payload 重验全部对象。"""
    if not isinstance(record, PlainTextCarrierRecord):
        raise PlainTextCarrierAdapterError("deserializer record 类型非法")
    try:
        if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
                or payload.endswith(b"\n\n")):
            raise PlainTextCarrierAdapterError("materialization newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        if set(value) != {
                "anchors", "artifact_kind", "case_key", "envelopes",
                "format_version", "revisions", "sample_kind", "scopes",
                "sources"}:
            raise PlainTextCarrierAdapterError("materialization 字段不精确")
        if (value["artifact_kind"] != MATERIALIZATION_KIND
                or value["format_version"] != MATERIALIZATION_FORMAT_VERSION
                or value["case_key"] != record.case_key.to_list()
                or value["sample_kind"] != record.sample_kind):
            raise PlainTextCarrierAdapterError("materialization record 身份漂移")
        sources = tuple(SourceRef.from_stable_key(item) for item in
                        _strict_stable_keys(value["sources"], where="sources"))
        scopes = tuple(ScopeIdentity.from_stable_key(item) for item in
                       _strict_stable_keys(value["scopes"], where="scopes"))
        envelopes = tuple(ArtifactEnvelope.from_stable_key(item) for item in
                          _strict_stable_keys(
                              value["envelopes"], where="envelopes"))
        anchors = tuple(ArtifactAnchor.from_stable_key(item) for item in
                        _strict_stable_keys(value["anchors"], where="anchors"))
        revisions = tuple(ArtifactCarrierRevision.from_stable_key(item)
                          for item in _strict_stable_keys(
                              value["revisions"], where="revisions"))
        result = PlainTextCarrierMaterialization(
            record, sources, scopes, envelopes, anchors, revisions)
    except PlainTextCarrierAdapterError:
        raise
    except Exception as error:
        raise PlainTextCarrierAdapterError("materialization 损坏") from error
    if serialize_plain_text_materialization(result) != payload:
        raise PlainTextCarrierAdapterError("materialization 不是 canonical 表示")
    return result


__all__ = [
    "MATERIALIZATION_FORMAT_VERSION",
    "MATERIALIZATION_KIND",
    "PLAIN_TEXT_SOURCE_KIND",
    "PlainTextCarrierAdapterError",
    "PlainTextCarrierMaterialization",
    "adapt_plain_text_carrier_record",
    "deserialize_plain_text_materialization",
    "serialize_plain_text_materialization",
]
