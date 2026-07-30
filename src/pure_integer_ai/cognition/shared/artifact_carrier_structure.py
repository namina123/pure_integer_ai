"""LC-16 carrier structure node、引用和共享语义投影合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.artifact_carrier_core import (
    ARTIFACT_CARRIER_IDENTITY_ANCHOR,
    ARTIFACT_CARRIER_IDENTITY_PROJECTION,
    ARTIFACT_CARRIER_IDENTITY_REFERENCE,
    ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE,
    REFERENCE_UNRESOLVED,
    _KeyReader,
    _PROJECTION_DIRECTIONS,
    _REFERENCE_STATES,
    _carrier_identity,
    _classifier,
    _fingerprint,
    _ints,
    _nonnegative,
    _object,
    _optional_fingerprint,
    _optional_key,
    _packed,
    _require_envelope_context,
    _require_local_context,
    artifact_carrier_local_kind,
    artifact_carrier_source,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    describe_artifact_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ARTIFACT,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity


def artifact_structure_node_identity(
        *,
        envelope_identity: ObjectIdentity,
        source: SourceRef,
        scope: ScopeIdentity,
        anchor_identity: ObjectIdentity,
        structure_family: ObjectIdentity,
        node_kind: ObjectIdentity,
        role: ObjectIdentity | None,
        parent_identity: ObjectIdentity | None,
        ordinal: int,
        qualifiers: tuple[int, ...],
        node_key: tuple[int, ...],
        ) -> ObjectIdentity:
    _require_envelope_context(envelope_identity, source, scope)
    _require_local_context(
        anchor_identity,
        expected_kind=ARTIFACT_CARRIER_IDENTITY_ANCHOR,
        envelope_identity=envelope_identity,
        source=source,
        scope=scope,
        where="structure node anchor",
    )
    _classifier(
        structure_family, where="ArtifactStructureNode.structure_family",
        object_kind=OBJECT_STRUCTURE_CONCEPT)
    _classifier(
        node_kind, where="ArtifactStructureNode.node_kind",
        object_kind=OBJECT_STRUCTURE_CONCEPT)
    if role is not None:
        _classifier(
            role, where="ArtifactStructureNode.role", object_kind=OBJECT_ROLE)
    if parent_identity is not None:
        _require_local_context(
            parent_identity,
            expected_kind=ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE,
            envelope_identity=envelope_identity,
            source=source,
            scope=scope,
            where="parent structure node",
        )
    _nonnegative(ordinal, where="ArtifactStructureNode.ordinal")
    qualifiers = _ints(
        qualifiers, where="ArtifactStructureNode.qualifiers",
        allow_empty=True)
    node_key = _ints(node_key, where="ArtifactStructureNode.node_key")
    components = (
        *_fingerprint(
            envelope_identity.stable_key(),
            domain="lc16.artifact-node.envelope.v1"),
        *_fingerprint(scope.stable_key(), domain="lc16.artifact-node.scope.v1"),
        *_fingerprint(
            anchor_identity.stable_key(),
            domain="lc16.artifact-node.anchor.v1"),
        *_fingerprint(
            structure_family.stable_key(),
            domain="lc16.artifact-node.family.v1"),
        *_fingerprint(
            node_kind.stable_key(), domain="lc16.artifact-node.kind.v1"),
        *_optional_fingerprint(
            None if role is None else role.stable_key(),
            domain="lc16.artifact-node.role.v1"),
        *_optional_fingerprint(
            None if parent_identity is None else parent_identity.stable_key(),
            domain="lc16.artifact-node.parent.v1"),
        ordinal,
        *_fingerprint(qualifiers, domain="lc16.artifact-node.qualifiers.v1"),
        *_fingerprint(node_key, domain="lc16.artifact-node.key.v1"),
    )
    return _carrier_identity(
        ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE, source, components)


@dataclass(frozen=True)
class ArtifactStructureNode:
    """固定到 envelope/anchor 的开放 carrier structure node。"""

    identity: ObjectIdentity
    envelope_identity: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    anchor_identity: ObjectIdentity
    structure_family: ObjectIdentity
    node_kind: ObjectIdentity
    role: ObjectIdentity | None
    parent_identity: ObjectIdentity | None
    ordinal: int
    qualifiers: tuple[int, ...]
    node_key: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = artifact_structure_node_identity(
            envelope_identity=self.envelope_identity,
            source=self.source,
            scope=self.scope,
            anchor_identity=self.anchor_identity,
            structure_family=self.structure_family,
            node_kind=self.node_kind,
            role=self.role,
            parent_identity=self.parent_identity,
            ordinal=self.ordinal,
            qualifiers=self.qualifiers,
            node_key=self.node_key,
        )
        if self.identity != expected:
            raise ValueError("ArtifactStructureNode identity 与字段不一致")

    def stable_key(self) -> tuple[int, ...]:
        values = (
            self.identity,
            self.envelope_identity,
            self.anchor_identity,
            self.structure_family,
            self.node_kind,
        )
        result = [1]
        result.extend(_packed(self.source.stable_key()))
        result.extend(_packed(self.scope.stable_key()))
        for value in values:
            result.extend(_packed(value.stable_key()))
        result.extend(_packed(_optional_key(
            None if self.role is None else self.role.stable_key())))
        result.extend(_packed(_optional_key(
            None if self.parent_identity is None
            else self.parent_identity.stable_key())))
        result.append(self.ordinal)
        result.extend(_packed(self.qualifiers))
        result.extend(_packed(self.node_key))
        return tuple(result)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ArtifactStructureNode":
        reader = _KeyReader(key, where="ArtifactStructureNode.stable_key")
        if reader.integer("version") != 1:
            raise ValueError("ArtifactStructureNode stable key version 非法")
        source = SourceRef.from_stable_key(reader.part("source"))
        scope = ScopeIdentity.from_stable_key(reader.part("scope"))
        identities = tuple(
            ObjectIdentity.from_stable_key(reader.part(label))
            for label in (
                "identity", "envelope", "anchor", "structure_family",
                "node_kind")
        )
        role_key = reader.part("role", allow_empty=True)
        parent_key = reader.part("parent", allow_empty=True)
        ordinal = reader.integer("ordinal")
        qualifiers = reader.part("qualifiers", allow_empty=True)
        node_key = reader.part("node_key")
        reader.finish()
        return cls(
            identities[0], identities[1], source, scope, identities[2],
            identities[3], identities[4],
            None if not role_key else ObjectIdentity.from_stable_key(role_key),
            None if not parent_key else ObjectIdentity.from_stable_key(
                parent_key),
            ordinal, qualifiers, node_key,
        )


def make_artifact_structure_node(**kwargs) -> ArtifactStructureNode:
    identity = artifact_structure_node_identity(**kwargs)
    return ArtifactStructureNode(identity=identity, **kwargs)


def artifact_reference_binding_identity(
        *,
        envelope_identity: ObjectIdentity,
        source: SourceRef,
        scope: ScopeIdentity,
        anchor_identity: ObjectIdentity,
        relation: ObjectIdentity,
        target_state: int,
        target_source: SourceRef | None,
        target_anchor: ObjectIdentity | None,
        target_fingerprint: tuple[int, ...],
        reference_key: tuple[int, ...],
        ) -> ObjectIdentity:
    _require_envelope_context(envelope_identity, source, scope)
    _require_local_context(
        anchor_identity,
        expected_kind=ARTIFACT_CARRIER_IDENTITY_ANCHOR,
        envelope_identity=envelope_identity,
        source=source,
        scope=scope,
        where="reference anchor",
    )
    _classifier(relation, where="ArtifactReferenceBinding.relation")
    if target_state not in _REFERENCE_STATES:
        raise ValueError("reference target_state 未登记")
    if target_source is not None and not isinstance(target_source, SourceRef):
        raise TypeError("reference target_source 必须是 SourceRef")
    if target_anchor is not None:
        if artifact_carrier_local_kind(
                target_anchor) != ARTIFACT_CARRIER_IDENTITY_ANCHOR:
            raise ValueError("reference target_anchor 类型非法")
        target_anchor_source = artifact_carrier_source(target_anchor)
        if target_source is not None and target_anchor_source != target_source:
            raise ValueError("target_anchor 与 target_source 不一致")
    target_fingerprint = _ints(
        target_fingerprint,
        where="ArtifactReferenceBinding.target_fingerprint",
        allow_empty=True,
    )
    has_target = bool(target_source or target_anchor or target_fingerprint)
    if target_state == REFERENCE_UNRESOLVED and has_target:
        raise ValueError("UNRESOLVED reference 不得携带已解析目标")
    if target_state != REFERENCE_UNRESOLVED and not has_target:
        raise ValueError("非 UNRESOLVED reference 必须保留目标身份依据")
    reference_key = _ints(
        reference_key, where="ArtifactReferenceBinding.reference_key")
    components = (
        *_fingerprint(
            envelope_identity.stable_key(),
            domain="lc16.artifact-reference.envelope.v1"),
        *_fingerprint(
            scope.stable_key(), domain="lc16.artifact-reference.scope.v1"),
        *_fingerprint(
            anchor_identity.stable_key(),
            domain="lc16.artifact-reference.anchor.v1"),
        *_fingerprint(
            relation.stable_key(),
            domain="lc16.artifact-reference.relation.v1"),
        target_state,
        *_optional_fingerprint(
            None if target_source is None else target_source.stable_key(),
            domain="lc16.artifact-reference.target-source.v1"),
        *_optional_fingerprint(
            None if target_anchor is None else target_anchor.stable_key(),
            domain="lc16.artifact-reference.target-anchor.v1"),
        *_optional_fingerprint(
            None if not target_fingerprint else target_fingerprint,
            domain="lc16.artifact-reference.target-fingerprint.v1"),
        *_fingerprint(
            reference_key, domain="lc16.artifact-reference.key.v1"),
    )
    return _carrier_identity(
        ARTIFACT_CARRIER_IDENTITY_REFERENCE, source, components)


@dataclass(frozen=True)
class ArtifactReferenceBinding:
    """显式区分已解析、未解析、撤回和访问受阻的载体引用。"""

    identity: ObjectIdentity
    envelope_identity: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    anchor_identity: ObjectIdentity
    relation: ObjectIdentity
    target_state: int
    target_source: SourceRef | None
    target_anchor: ObjectIdentity | None
    target_fingerprint: tuple[int, ...]
    reference_key: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = artifact_reference_binding_identity(
            envelope_identity=self.envelope_identity,
            source=self.source,
            scope=self.scope,
            anchor_identity=self.anchor_identity,
            relation=self.relation,
            target_state=self.target_state,
            target_source=self.target_source,
            target_anchor=self.target_anchor,
            target_fingerprint=self.target_fingerprint,
            reference_key=self.reference_key,
        )
        if self.identity != expected:
            raise ValueError("ArtifactReferenceBinding identity 与字段不一致")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_packed(self.identity.stable_key()),
            *_packed(self.envelope_identity.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.anchor_identity.stable_key()),
            *_packed(self.relation.stable_key()),
            self.target_state,
            *_packed(_optional_key(
                None if self.target_source is None
                else self.target_source.stable_key())),
            *_packed(_optional_key(
                None if self.target_anchor is None
                else self.target_anchor.stable_key())),
            *_packed(self.target_fingerprint),
            *_packed(self.reference_key),
        )

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...],
            ) -> "ArtifactReferenceBinding":
        reader = _KeyReader(key, where="ArtifactReferenceBinding.stable_key")
        if reader.integer("version") != 1:
            raise ValueError("ArtifactReferenceBinding stable key version 非法")
        identity = ObjectIdentity.from_stable_key(reader.part("identity"))
        envelope = ObjectIdentity.from_stable_key(reader.part("envelope"))
        source = SourceRef.from_stable_key(reader.part("source"))
        scope = ScopeIdentity.from_stable_key(reader.part("scope"))
        anchor = ObjectIdentity.from_stable_key(reader.part("anchor"))
        relation = ObjectIdentity.from_stable_key(reader.part("relation"))
        target_state = reader.integer("target_state")
        target_source_key = reader.part("target_source", allow_empty=True)
        target_anchor_key = reader.part("target_anchor", allow_empty=True)
        target_fingerprint = reader.part(
            "target_fingerprint", allow_empty=True)
        reference_key = reader.part("reference_key")
        reader.finish()
        return cls(
            identity,
            envelope,
            source,
            scope,
            anchor,
            relation,
            target_state,
            None if not target_source_key else SourceRef.from_stable_key(
                target_source_key),
            None if not target_anchor_key else ObjectIdentity.from_stable_key(
                target_anchor_key),
            target_fingerprint,
            reference_key,
        )


def make_artifact_reference_binding(**kwargs) -> ArtifactReferenceBinding:
    identity = artifact_reference_binding_identity(**kwargs)
    return ArtifactReferenceBinding(identity=identity, **kwargs)


def artifact_semantic_projection_identity(
        *,
        envelope_identity: ObjectIdentity,
        source: SourceRef,
        scope: ScopeIdentity,
        anchor_identities: tuple[ObjectIdentity, ...],
        structure_node_identities: tuple[ObjectIdentity, ...],
        projection_kind: ObjectIdentity,
        semantic_object: ObjectIdentity,
        lifecycle_state: ObjectIdentity,
        hypothesis: HypothesisKey,
        directions: tuple[int, ...],
        projection_key: tuple[int, ...],
        ) -> ObjectIdentity:
    _require_envelope_context(envelope_identity, source, scope)
    if not isinstance(anchor_identities, tuple):
        raise ValueError("projection anchors 必须是 tuple")
    for anchor in anchor_identities:
        _require_local_context(
            anchor,
            expected_kind=ARTIFACT_CARRIER_IDENTITY_ANCHOR,
            envelope_identity=envelope_identity,
            source=source,
            scope=scope,
            where="projection anchor",
        )
    if anchor_identities != tuple(sorted(
            set(anchor_identities), key=ObjectIdentity.stable_key)):
        raise ValueError("projection anchors 必须排序去重")
    if not isinstance(structure_node_identities, tuple):
        raise ValueError("projection structure nodes 必须是 tuple")
    for node in structure_node_identities:
        _require_local_context(
            node,
            expected_kind=ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE,
            envelope_identity=envelope_identity,
            source=source,
            scope=scope,
            where="projection structure node",
        )
    if structure_node_identities != tuple(sorted(
            set(structure_node_identities), key=ObjectIdentity.stable_key)):
        raise ValueError("projection structure nodes 必须排序去重")
    if not anchor_identities and not structure_node_identities:
        raise ValueError("projection anchors/nodes 不得同时为空")
    _classifier(projection_kind, where="ArtifactSemanticProjection.kind")
    _object(semantic_object, where="ArtifactSemanticProjection.semantic_object")
    if semantic_object.object_kind == OBJECT_ARTIFACT:
        try:
            describe_artifact_identity(semantic_object)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "projection Artifact target 必须是 FormalArtifact 身份"
            ) from exc
    _classifier(
        lifecycle_state, where="ArtifactSemanticProjection.lifecycle_state")
    if not isinstance(hypothesis, HypothesisKey):
        raise TypeError("projection hypothesis 必须是 HypothesisKey")
    if hypothesis.observation != source or hypothesis.scope != scope:
        raise ValueError("projection hypothesis 与 source/scope 不一致")
    directions = _ints(
        directions, where="ArtifactSemanticProjection.directions")
    if (set(directions) - _PROJECTION_DIRECTIONS
            or tuple(sorted(set(directions))) != directions):
        raise ValueError("projection directions 必须排序去重且已登记")
    projection_key = _ints(
        projection_key, where="ArtifactSemanticProjection.projection_key")
    anchor_fingerprints = tuple(
        value
        for anchor in anchor_identities
        for value in _fingerprint(
            anchor.stable_key(), domain="lc16.artifact-projection.anchor.v1")
    )
    node_fingerprints = tuple(
        value
        for node in structure_node_identities
        for value in _fingerprint(
            node.stable_key(), domain="lc16.artifact-projection.node.v1")
    )
    components = (
        *_fingerprint(
            envelope_identity.stable_key(),
            domain="lc16.artifact-projection.envelope.v1"),
        *_fingerprint(
            scope.stable_key(), domain="lc16.artifact-projection.scope.v1"),
        len(anchor_identities),
        *anchor_fingerprints,
        len(structure_node_identities),
        *node_fingerprints,
        *_fingerprint(
            projection_kind.stable_key(),
            domain="lc16.artifact-projection.kind.v1"),
        *_fingerprint(
            semantic_object.stable_key(),
            domain="lc16.artifact-projection.semantic-object.v1"),
        *_fingerprint(
            lifecycle_state.stable_key(),
            domain="lc16.artifact-projection.lifecycle.v1"),
        *_fingerprint(
            hypothesis.stable_key(),
            domain="lc16.artifact-projection.hypothesis.v1"),
        *_fingerprint(
            directions, domain="lc16.artifact-projection.directions.v1"),
        *_fingerprint(
            projection_key, domain="lc16.artifact-projection.key.v1"),
    )
    return _carrier_identity(
        ARTIFACT_CARRIER_IDENTITY_PROJECTION, source, components)


@dataclass(frozen=True)
class ArtifactSemanticProjection:
    """把 carrier anchors/nodes 以候选/Evidence 投影到共享语言对象。"""

    identity: ObjectIdentity
    envelope_identity: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    anchor_identities: tuple[ObjectIdentity, ...]
    structure_node_identities: tuple[ObjectIdentity, ...]
    projection_kind: ObjectIdentity
    semantic_object: ObjectIdentity
    lifecycle_state: ObjectIdentity
    hypothesis: HypothesisKey
    evidence: tuple[EvidenceRecord, ...]
    directions: tuple[int, ...]
    projection_key: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = artifact_semantic_projection_identity(
            envelope_identity=self.envelope_identity,
            source=self.source,
            scope=self.scope,
            anchor_identities=self.anchor_identities,
            structure_node_identities=self.structure_node_identities,
            projection_kind=self.projection_kind,
            semantic_object=self.semantic_object,
            lifecycle_state=self.lifecycle_state,
            hypothesis=self.hypothesis,
            directions=self.directions,
            projection_key=self.projection_key,
        )
        if self.identity != expected:
            raise ValueError("ArtifactSemanticProjection identity 与字段不一致")
        if (not isinstance(self.evidence, tuple) or not self.evidence
                or not all(isinstance(item, EvidenceRecord)
                           for item in self.evidence)):
            raise ValueError("projection evidence 必须是非空 EvidenceRecord tuple")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("projection evidence 必须归同一 hypothesis")
        ordered = tuple(sorted(self.evidence, key=EvidenceRecord.stable_key))
        if self.evidence != ordered:
            raise ValueError("projection evidence 必须按稳定键排序")
        ids = tuple(item.evidence_id for item in self.evidence)
        if len(ids) != len(set(ids)):
            raise ValueError("projection evidence_id 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        result = [
            1,
            *_packed(self.identity.stable_key()),
            *_packed(self.envelope_identity.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            len(self.anchor_identities),
        ]
        for anchor in self.anchor_identities:
            result.extend(_packed(anchor.stable_key()))
        result.append(len(self.structure_node_identities))
        for node in self.structure_node_identities:
            result.extend(_packed(node.stable_key()))
        for value in (
                self.projection_kind, self.semantic_object,
                self.lifecycle_state):
            result.extend(_packed(value.stable_key()))
        result.extend(_packed(self.hypothesis.stable_key()))
        result.append(len(self.evidence))
        for item in self.evidence:
            result.extend(_packed(item.stable_key()))
        result.extend(_packed(self.directions))
        result.extend(_packed(self.projection_key))
        return tuple(result)

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...],
            ) -> "ArtifactSemanticProjection":
        reader = _KeyReader(key, where="ArtifactSemanticProjection.stable_key")
        if reader.integer("version") != 1:
            raise ValueError("ArtifactSemanticProjection stable key version 非法")
        identity = ObjectIdentity.from_stable_key(reader.part("identity"))
        envelope = ObjectIdentity.from_stable_key(reader.part("envelope"))
        source = SourceRef.from_stable_key(reader.part("source"))
        scope = ScopeIdentity.from_stable_key(reader.part("scope"))
        anchor_count = reader.integer("anchor_count")
        if anchor_count < 0:
            raise ValueError("projection anchor_count 非法")
        anchors = tuple(
            ObjectIdentity.from_stable_key(reader.part("anchor"))
            for _ in range(anchor_count))
        node_count = reader.integer("structure_node_count")
        if node_count < 0 or anchor_count + node_count <= 0:
            raise ValueError("projection structure_node_count 非法")
        structure_nodes = tuple(
            ObjectIdentity.from_stable_key(reader.part("structure_node"))
            for _ in range(node_count))
        projection_kind = ObjectIdentity.from_stable_key(
            reader.part("projection_kind"))
        semantic_object = ObjectIdentity.from_stable_key(
            reader.part("semantic_object"))
        lifecycle_state = ObjectIdentity.from_stable_key(
            reader.part("lifecycle_state"))
        hypothesis = HypothesisKey.from_stable_key(reader.part("hypothesis"))
        evidence_count = reader.integer("evidence_count")
        if evidence_count <= 0:
            raise ValueError("projection evidence_count 非法")
        evidence = tuple(
            EvidenceRecord.from_stable_key(reader.part("evidence"))
            for _ in range(evidence_count))
        directions = reader.part("directions")
        projection_key = reader.part("projection_key")
        reader.finish()
        return cls(
            identity, envelope, source, scope, anchors, structure_nodes,
            projection_kind, semantic_object, lifecycle_state, hypothesis,
            evidence, directions, projection_key,
        )


def make_artifact_semantic_projection(**kwargs) -> ArtifactSemanticProjection:
    identity_kwargs = dict(kwargs)
    identity_kwargs.pop("evidence")
    identity = artifact_semantic_projection_identity(**identity_kwargs)
    return ArtifactSemanticProjection(identity=identity, **kwargs)

__all__ = [
    "ArtifactReferenceBinding",
    "ArtifactSemanticProjection",
    "ArtifactStructureNode",
    "artifact_reference_binding_identity",
    "artifact_semantic_projection_identity",
    "artifact_structure_node_identity",
    "make_artifact_reference_binding",
    "make_artifact_semantic_projection",
    "make_artifact_structure_node",
]
