"""LC-16 carrier revision 与 split/merge 局部映射合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.artifact_carrier_core import (
    ARTIFACT_CARRIER_IDENTITY_ENVELOPE,
    ARTIFACT_CARRIER_IDENTITY_REVISION,
    REVISION_MAP_ANCHOR,
    REVISION_MAP_PROJECTION,
    REVISION_MAP_REFERENCE,
    REVISION_MAP_STRUCTURE_NODE,
    _KeyReader,
    _REVISION_MAPPING_LOCAL_KINDS,
    _carrier_identity,
    _classifier,
    _fingerprint,
    _ints,
    _packed,
    _require_local_envelope,
    artifact_carrier_local_kind,
    artifact_carrier_source,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key


@dataclass(frozen=True)
class ArtifactRevisionMapping:
    """一次 carrier revision 中可表达 delete/split/merge 的局部对象映射。"""

    mapping_kind: int
    old_identity: ObjectIdentity
    new_identities: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        expected_kind = _REVISION_MAPPING_LOCAL_KINDS.get(self.mapping_kind)
        if expected_kind is None:
            raise ValueError("revision mapping_kind 未登记")
        if artifact_carrier_local_kind(self.old_identity) != expected_kind:
            raise ValueError("revision old_identity 类型与 mapping_kind 不匹配")
        if not isinstance(self.new_identities, tuple):
            raise ValueError("revision new_identities 必须是 tuple")
        if any(artifact_carrier_local_kind(item) != expected_kind
               for item in self.new_identities):
            raise ValueError("revision new identity 类型不匹配")
        if self.new_identities != tuple(sorted(
                set(self.new_identities), key=ObjectIdentity.stable_key)):
            raise ValueError("revision new_identities 必须排序去重")

    def stable_key(self) -> tuple[int, ...]:
        result = [
            1,
            self.mapping_kind,
            *_packed(self.old_identity.stable_key()),
            len(self.new_identities),
        ]
        for item in self.new_identities:
            result.extend(_packed(item.stable_key()))
        return tuple(result)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ArtifactRevisionMapping":
        reader = _KeyReader(key, where="ArtifactRevisionMapping.stable_key")
        if reader.integer("version") != 1:
            raise ValueError("ArtifactRevisionMapping stable key version 非法")
        mapping_kind = reader.integer("mapping_kind")
        old = ObjectIdentity.from_stable_key(reader.part("old_identity"))
        new_count = reader.integer("new_count")
        if new_count < 0:
            raise ValueError("revision new_count 非法")
        new = tuple(
            ObjectIdentity.from_stable_key(reader.part("new_identity"))
            for _ in range(new_count))
        reader.finish()
        return cls(mapping_kind, old, new)


def _validated_revision_mappings(
        mappings: tuple[ArtifactRevisionMapping, ...],
        ) -> tuple[ArtifactRevisionMapping, ...]:
    if (not isinstance(mappings, tuple) or not mappings
            or not all(isinstance(item, ArtifactRevisionMapping)
                       for item in mappings)):
        raise ValueError("carrier revision mappings 不得为空")
    ordered = tuple(sorted(mappings, key=ArtifactRevisionMapping.stable_key))
    if mappings != ordered:
        raise ValueError("carrier revision mappings 必须按稳定键排序")
    old_ids = tuple(item.old_identity for item in mappings)
    if len(old_ids) != len(set(old_ids)):
        raise ValueError("carrier revision old identity 不得重复映射")
    return mappings


def artifact_carrier_revision_identity(
        *,
        old_envelope_identity: ObjectIdentity,
        new_envelope_identity: ObjectIdentity,
        reason: ObjectIdentity,
        hypothesis: HypothesisKey,
        mappings: tuple[ArtifactRevisionMapping, ...],
        revision_key: tuple[int, ...],
        ) -> ObjectIdentity:
    if artifact_carrier_local_kind(
            old_envelope_identity) != ARTIFACT_CARRIER_IDENTITY_ENVELOPE:
        raise ValueError("old_envelope_identity 类型非法")
    if artifact_carrier_local_kind(
            new_envelope_identity) != ARTIFACT_CARRIER_IDENTITY_ENVELOPE:
        raise ValueError("new_envelope_identity 类型非法")
    old_source = artifact_carrier_source(old_envelope_identity)
    new_source = artifact_carrier_source(new_envelope_identity)
    if parser_lineage_key(old_source) != parser_lineage_key(new_source):
        raise ValueError("carrier revision 必须属于同一 parser lineage")
    if old_source.versions.parser == new_source.versions.parser:
        raise ValueError("carrier revision old/new parser version 必须不同")
    _classifier(reason, where="ArtifactCarrierRevision.reason")
    if not isinstance(hypothesis, HypothesisKey):
        raise TypeError("carrier revision hypothesis 必须是 HypothesisKey")
    if hypothesis.observation != new_source:
        raise ValueError("carrier revision hypothesis 必须绑定新 SourceRef")
    mappings = _validated_revision_mappings(mappings)
    mapping_payload = tuple(
        value
        for mapping in mappings
        for value in _packed(mapping.stable_key())
    )
    revision_key = _ints(
        revision_key, where="ArtifactCarrierRevision.revision_key")
    components = (
        *_fingerprint(
            old_envelope_identity.stable_key(),
            domain="lc16.artifact-revision.old-envelope.v1"),
        *_fingerprint(
            new_envelope_identity.stable_key(),
            domain="lc16.artifact-revision.new-envelope.v1"),
        *_fingerprint(
            reason.stable_key(), domain="lc16.artifact-revision.reason.v1"),
        *_fingerprint(
            hypothesis.stable_key(),
            domain="lc16.artifact-revision.hypothesis.v1"),
        *_fingerprint(
            (len(mappings), *mapping_payload),
            domain="lc16.artifact-revision.mappings.v1"),
        *_fingerprint(
            revision_key, domain="lc16.artifact-revision.key.v1"),
    )
    return _carrier_identity(
        ARTIFACT_CARRIER_IDENTITY_REVISION, new_source, components)


@dataclass(frozen=True)
class ArtifactCarrierRevision:
    """不改变历史 stable key 的 envelope/anchor/node/reference/projection 修订。"""

    identity: ObjectIdentity
    old_envelope_identity: ObjectIdentity
    new_envelope_identity: ObjectIdentity
    reason: ObjectIdentity
    hypothesis: HypothesisKey
    mappings: tuple[ArtifactRevisionMapping, ...]
    evidence: tuple[EvidenceRecord, ...]
    revision_key: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = artifact_carrier_revision_identity(
            old_envelope_identity=self.old_envelope_identity,
            new_envelope_identity=self.new_envelope_identity,
            reason=self.reason,
            hypothesis=self.hypothesis,
            mappings=self.mappings,
            revision_key=self.revision_key,
        )
        if self.identity != expected:
            raise ValueError("ArtifactCarrierRevision identity 与字段不一致")
        for mapping in self.mappings:
            expected_kind = _REVISION_MAPPING_LOCAL_KINDS[
                mapping.mapping_kind]
            _require_local_envelope(
                mapping.old_identity,
                expected_kind=expected_kind,
                envelope_identity=self.old_envelope_identity,
                where="revision mapping old identity",
            )
            for item in mapping.new_identities:
                _require_local_envelope(
                    item,
                    expected_kind=expected_kind,
                    envelope_identity=self.new_envelope_identity,
                    where="revision mapping new identity",
                )
        if (not isinstance(self.evidence, tuple) or not self.evidence
                or not all(isinstance(item, EvidenceRecord)
                           for item in self.evidence)):
            raise ValueError("carrier revision evidence 不得为空")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("carrier revision evidence 必须归同一 hypothesis")
        if self.evidence != tuple(sorted(
                self.evidence, key=EvidenceRecord.stable_key)):
            raise ValueError("carrier revision evidence 必须按稳定键排序")
        ids = tuple(item.evidence_id for item in self.evidence)
        if len(ids) != len(set(ids)):
            raise ValueError("carrier revision evidence_id 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        result = [
            1,
            *_packed(self.identity.stable_key()),
            *_packed(self.old_envelope_identity.stable_key()),
            *_packed(self.new_envelope_identity.stable_key()),
            *_packed(self.reason.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            len(self.mappings),
        ]
        for mapping in self.mappings:
            result.extend(_packed(mapping.stable_key()))
        result.append(len(self.evidence))
        for item in self.evidence:
            result.extend(_packed(item.stable_key()))
        result.extend(_packed(self.revision_key))
        return tuple(result)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ArtifactCarrierRevision":
        reader = _KeyReader(key, where="ArtifactCarrierRevision.stable_key")
        if reader.integer("version") != 1:
            raise ValueError("ArtifactCarrierRevision stable key version 非法")
        identity = ObjectIdentity.from_stable_key(reader.part("identity"))
        old_envelope = ObjectIdentity.from_stable_key(
            reader.part("old_envelope"))
        new_envelope = ObjectIdentity.from_stable_key(
            reader.part("new_envelope"))
        reason = ObjectIdentity.from_stable_key(reader.part("reason"))
        hypothesis = HypothesisKey.from_stable_key(reader.part("hypothesis"))
        mapping_count = reader.integer("mapping_count")
        if mapping_count <= 0:
            raise ValueError("carrier revision mapping_count 非法")
        mappings = tuple(
            ArtifactRevisionMapping.from_stable_key(reader.part("mapping"))
            for _ in range(mapping_count))
        evidence_count = reader.integer("evidence_count")
        if evidence_count <= 0:
            raise ValueError("carrier revision evidence_count 非法")
        evidence = tuple(
            EvidenceRecord.from_stable_key(reader.part("evidence"))
            for _ in range(evidence_count))
        revision_key = reader.part("revision_key")
        reader.finish()
        return cls(
            identity, old_envelope, new_envelope, reason, hypothesis,
            mappings, evidence, revision_key)


def make_artifact_carrier_revision(**kwargs) -> ArtifactCarrierRevision:
    identity_kwargs = dict(kwargs)
    identity_kwargs.pop("evidence")
    identity = artifact_carrier_revision_identity(**identity_kwargs)
    return ArtifactCarrierRevision(identity=identity, **kwargs)

__all__ = [
    "ArtifactCarrierRevision",
    "ArtifactRevisionMapping",
    "artifact_carrier_revision_identity",
    "make_artifact_carrier_revision",
]
