"""LC-16 ArtifactEnvelope 与 carrier-local 对象纯合同测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_GRID_RECT,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TRANSCRIPT_ALIGNMENT,
    ANCHOR_TREE_PATH,
    ARTIFACT_CARRIER_IDENTITY_ANCHOR,
    ARTIFACT_CARRIER_IDENTITY_ENVELOPE,
    ARTIFACT_CARRIER_IDENTITY_PROJECTION,
    ARTIFACT_CARRIER_IDENTITY_REFERENCE,
    ARTIFACT_CARRIER_IDENTITY_REVISION,
    ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE,
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
    RAW_UNIT_OCTET,
    RAW_UNIT_UNICODE_SCALAR,
    REFERENCE_RESOLVED,
    REFERENCE_UNRESOLVED,
    REVISION_MAP_ANCHOR,
    ArtifactAnchor,
    ArtifactCarrierRevision,
    ArtifactEnvelope,
    ArtifactReferenceBinding,
    ArtifactRevisionMapping,
    ArtifactSemanticProjection,
    ArtifactStructureNode,
    artifact_carrier_local_kind,
    artifact_carrier_source,
    make_artifact_anchor,
    make_artifact_carrier_revision,
    make_artifact_envelope,
    make_artifact_reference_binding,
    make_artifact_semantic_projection,
    make_artifact_structure_node,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactAuthority,
    describe_artifact_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    proposition_identity,
    role_identity,
)


def _source(parser_version: int, *, source_id: int = 16001) -> SourceRef:
    owner = OwnerScope()
    versions = VersionBundle(
        CorpusVersion(3),
        ParserVersion(parser_version),
        PrimitiveVersion(5),
        CurriculumVersion(7),
    )
    return SourceRef(9, source_id, 11, owner, versions)


def _concept(source: SourceRef, *key: int):
    return concept_identity(key, owner=source.owner, versions=source.versions)


def _structure(source: SourceRef, *key: int):
    return structure_concept_identity(
        key, owner=source.owner, versions=source.versions)


def _authority(source: SourceRef, seed: int) -> ArtifactAuthority:
    return ArtifactAuthority(
        _concept(source, 16010, seed, 1),
        _concept(source, 16010, seed, 2),
    )


def _envelope(
        source: SourceRef,
        *,
        text: str = "甲乙引用",
        raw_unit_kind: int = RAW_UNIT_UNICODE_SCALAR,
        raw_units: tuple[int, ...] | None = None,
        ) -> ArtifactEnvelope:
    scope = document_scope(source)
    units = tuple(ord(item) for item in text) if raw_units is None else raw_units
    return make_artifact_envelope(
        source=source,
        scope=scope,
        carrier_family=_structure(source, 16020, 1),
        raw_unit_kind=raw_unit_kind,
        raw_units=units,
        media_profile=_concept(source, 16020, 2),
        language_branch=language_branch_identity(
            (16020, 3), owner=source.owner, versions=source.versions),
        parser=_authority(source, 1),
        renderer=_authority(source, 2),
        envelope_key=(16020, 4),
    )


def _anchor(
        envelope: ArtifactEnvelope,
        *,
        anchor_kind: int = ANCHOR_TEXT_RANGE,
        coordinates: tuple[int, ...] = (0, 2),
        key: tuple[int, ...] = (16030, 1),
        ):
    return make_artifact_anchor(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_kind=anchor_kind,
        coordinates=coordinates,
        parser=envelope.parser,
        linked_text_anchor=None,
        anchor_key=key,
    )


def _hypothesis(source: SourceRef, seed: int = 1) -> HypothesisKey:
    return HypothesisKey(
        (16040, seed, 1),
        (16040, seed, 2),
        (16040, seed, 3),
        document_scope(source),
        source,
    )


def _evidence(hypothesis: HypothesisKey, evidence_id: int = 1) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id,
        hypothesis,
        EVIDENCE_SUPPORT,
        (16041, evidence_id),
        hypothesis.observation,
        evidence_id,
        (16042, evidence_id),
    )


def test_envelope_round_trips_unicode_and_octets_without_formal_artifact_alias():
    source = _source(1)
    envelope = _envelope(source)
    assert ArtifactEnvelope.from_stable_key(envelope.stable_key()) == envelope
    assert artifact_carrier_local_kind(envelope.identity) == (
        ARTIFACT_CARRIER_IDENTITY_ENVELOPE)
    assert artifact_carrier_source(envelope.identity) == source
    with pytest.raises(ValueError, match="版本或长度"):
        describe_artifact_identity(envelope.identity)

    octets = _envelope(
        source,
        raw_unit_kind=RAW_UNIT_OCTET,
        raw_units=(0, 1, 127, 255),
    )
    assert ArtifactEnvelope.from_stable_key(octets.stable_key()) == octets
    assert octets.identity != envelope.identity
    with pytest.raises(ValueError, match="0..255"):
        _envelope(source, raw_unit_kind=RAW_UNIT_OCTET, raw_units=(256,))


def test_envelope_identity_uses_compact_raw_fingerprint_but_stable_key_keeps_raw():
    source = _source(1)
    raw = tuple(index % 251 for index in range(4096))
    envelope = _envelope(
        source, raw_unit_kind=RAW_UNIT_OCTET, raw_units=raw)
    assert len(envelope.identity.stable_key()) < 400
    assert len(envelope.stable_key()) > len(raw)
    assert ArtifactEnvelope.from_stable_key(envelope.stable_key()).raw_units == raw
    changed = _envelope(
        source, raw_unit_kind=RAW_UNIT_OCTET,
        raw_units=(*raw[:-1], raw[-1] + 1))
    assert changed.identity != envelope.identity


def test_anchor_preserves_all_coordinate_families_and_optional_span_link():
    envelope = _envelope(_source(1))
    text = _anchor(envelope)
    tree = _anchor(
        envelope, anchor_kind=ANCHOR_TREE_PATH,
        coordinates=(2, 5, 1), key=(16030, 2))
    grid = _anchor(
        envelope, anchor_kind=ANCHOR_GRID_RECT,
        coordinates=(1, 3, 2, 4), key=(16030, 3))
    document = _anchor(
        envelope, anchor_kind=ANCHOR_DOCUMENT_REGION,
        coordinates=(2, 10, 20, 30), key=(16030, 5))
    reference = _anchor(
        envelope, anchor_kind=ANCHOR_REFERENCE_SLOT,
        coordinates=(4,), key=(16030, 6))
    transcript = _anchor(
        envelope, anchor_kind=ANCHOR_TRANSCRIPT_ALIGNMENT,
        coordinates=(10, 20), key=(16030, 7))
    for anchor in (text, tree, grid, document, reference, transcript):
        assert ArtifactAnchor.from_stable_key(anchor.stable_key()) == anchor
        assert artifact_carrier_local_kind(anchor.identity) == (
            ARTIFACT_CARRIER_IDENTITY_ANCHOR)

    linked = span_identity(envelope.source, members=((0, 2),), ordinal=0)
    linked_anchor = make_artifact_anchor(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_kind=ANCHOR_TEXT_RANGE,
        coordinates=(0, 2),
        parser=envelope.parser,
        linked_text_anchor=linked,
        anchor_key=(16030, 4),
    )
    assert linked_anchor.linked_text_anchor == linked
    with pytest.raises(ValueError, match="有序四元矩形"):
        _anchor(envelope, anchor_kind=ANCHOR_GRID_RECT, coordinates=(3, 1, 0, 2))


def test_anchor_rejects_scope_or_source_drift_from_envelope():
    envelope = _envelope(_source(1))
    other = _source(1, source_id=16002)
    with pytest.raises(ValueError, match="SourceRef 不一致"):
        make_artifact_anchor(
            envelope_identity=envelope.identity,
            source=other,
            scope=document_scope(other),
            anchor_kind=ANCHOR_TEXT_RANGE,
            coordinates=(0, 1),
            parser=_authority(other, 1),
            linked_text_anchor=None,
            anchor_key=(1,),
        )


def test_local_consumers_reject_anchor_from_other_envelope_with_same_source():
    source = _source(1)
    envelope = _envelope(source, text="甲乙")
    other_envelope = _envelope(source, text="丙丁")
    local_anchor = _anchor(envelope)
    foreign_anchor = _anchor(other_envelope)
    family = _structure(source, 16045, 1)
    with pytest.raises(ValueError, match="不属于指定 envelope"):
        make_artifact_structure_node(
            envelope_identity=envelope.identity,
            source=source,
            scope=envelope.scope,
            anchor_identity=foreign_anchor.identity,
            structure_family=family,
            node_kind=_structure(source, 16045, 2),
            role=None,
            parent_identity=None,
            ordinal=0,
            qualifiers=(),
            node_key=(16045, 3),
        )
    foreign_parent = make_artifact_structure_node(
        envelope_identity=other_envelope.identity,
        source=source,
        scope=other_envelope.scope,
        anchor_identity=foreign_anchor.identity,
        structure_family=family,
        node_kind=_structure(source, 16045, 10),
        role=None,
        parent_identity=None,
        ordinal=0,
        qualifiers=(),
        node_key=(16045, 11),
    )
    with pytest.raises(ValueError, match="不属于指定 envelope"):
        make_artifact_structure_node(
            envelope_identity=envelope.identity,
            source=source,
            scope=envelope.scope,
            anchor_identity=local_anchor.identity,
            structure_family=family,
            node_kind=_structure(source, 16045, 12),
            role=None,
            parent_identity=foreign_parent.identity,
            ordinal=1,
            qualifiers=(),
            node_key=(16045, 13),
        )
    with pytest.raises(ValueError, match="不属于指定 envelope"):
        make_artifact_reference_binding(
            envelope_identity=envelope.identity,
            source=source,
            scope=envelope.scope,
            anchor_identity=foreign_anchor.identity,
            relation=_concept(source, 16045, 4),
            target_state=REFERENCE_UNRESOLVED,
            target_source=None,
            target_anchor=None,
            target_fingerprint=(),
            reference_key=(16045, 5),
        )
    hypothesis = _hypothesis(source, seed=45)
    with pytest.raises(ValueError, match="不属于指定 envelope"):
        make_artifact_semantic_projection(
            envelope_identity=envelope.identity,
            source=source,
            scope=envelope.scope,
            anchor_identities=(foreign_anchor.identity,),
            structure_node_identities=(),
            projection_kind=_concept(source, 16045, 6),
            semantic_object=proposition_identity(source, (16045, 7)),
            lifecycle_state=_concept(source, 16045, 8),
            hypothesis=hypothesis,
            evidence=(_evidence(hypothesis, 45),),
            directions=(PROJECTION_UNDERSTANDING,),
            projection_key=(16045, 9),
        )


def test_structure_node_keeps_anchor_role_parent_order_and_qualifiers():
    envelope = _envelope(_source(1))
    anchor = _anchor(envelope)
    family = _structure(envelope.source, 16050, 1)
    root = make_artifact_structure_node(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_identity=anchor.identity,
        structure_family=family,
        node_kind=_structure(envelope.source, 16050, 2),
        role=None,
        parent_identity=None,
        ordinal=0,
        qualifiers=(1, 2),
        node_key=(16050, 3),
    )
    child = make_artifact_structure_node(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_identity=anchor.identity,
        structure_family=family,
        node_kind=_structure(envelope.source, 16050, 4),
        role=role_identity(
            (16050, 5), owner=envelope.source.owner,
            versions=envelope.source.versions),
        parent_identity=root.identity,
        ordinal=1,
        qualifiers=(3, 4),
        node_key=(16050, 6),
    )
    assert ArtifactStructureNode.from_stable_key(root.stable_key()) == root
    assert ArtifactStructureNode.from_stable_key(child.stable_key()) == child
    assert artifact_carrier_local_kind(child.identity) == (
        ARTIFACT_CARRIER_IDENTITY_STRUCTURE_NODE)
    assert len(child.identity.stable_key()) < 500


def test_reference_binding_distinguishes_unresolved_resolved_and_invalid_states():
    envelope = _envelope(_source(1))
    anchor = _anchor(envelope)
    relation = _concept(envelope.source, 16060, 1)
    unresolved = make_artifact_reference_binding(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_identity=anchor.identity,
        relation=relation,
        target_state=REFERENCE_UNRESOLVED,
        target_source=None,
        target_anchor=None,
        target_fingerprint=(),
        reference_key=(16060, 2),
    )
    resolved = make_artifact_reference_binding(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_identity=anchor.identity,
        relation=relation,
        target_state=REFERENCE_RESOLVED,
        target_source=envelope.source,
        target_anchor=anchor.identity,
        target_fingerprint=(16060, 3),
        reference_key=(16060, 4),
    )
    for reference in (unresolved, resolved):
        assert ArtifactReferenceBinding.from_stable_key(
            reference.stable_key()) == reference
        assert artifact_carrier_local_kind(reference.identity) == (
            ARTIFACT_CARRIER_IDENTITY_REFERENCE)
    with pytest.raises(ValueError, match="不得携带"):
        replace(unresolved, target_fingerprint=(1,))
    with pytest.raises(ValueError, match="必须保留目标"):
        replace(resolved, target_source=None, target_anchor=None,
                target_fingerprint=())


def test_semantic_projection_requires_same_hypothesis_evidence_and_sorted_directions():
    envelope = _envelope(_source(1))
    anchor = _anchor(envelope)
    node = make_artifact_structure_node(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_identity=anchor.identity,
        structure_family=_structure(envelope.source, 16070, 10),
        node_kind=_structure(envelope.source, 16070, 11),
        role=None,
        parent_identity=None,
        ordinal=0,
        qualifiers=(),
        node_key=(16070, 12),
    )
    hypothesis = _hypothesis(envelope.source)
    evidence = (_evidence(hypothesis),)
    projection = make_artifact_semantic_projection(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_identities=(anchor.identity,),
        structure_node_identities=(node.identity,),
        projection_kind=_concept(envelope.source, 16070, 1),
        semantic_object=proposition_identity(envelope.source, (16070, 2)),
        lifecycle_state=_concept(envelope.source, 16070, 3),
        hypothesis=hypothesis,
        evidence=evidence,
        directions=(
            PROJECTION_UNDERSTANDING,
            PROJECTION_REASONING,
            PROJECTION_GENERATION,
        ),
        projection_key=(16070, 4),
    )
    assert ArtifactSemanticProjection.from_stable_key(
        projection.stable_key()) == projection
    assert artifact_carrier_local_kind(projection.identity) == (
        ARTIFACT_CARRIER_IDENTITY_PROJECTION)
    node_only = make_artifact_semantic_projection(
        envelope_identity=envelope.identity,
        source=envelope.source,
        scope=envelope.scope,
        anchor_identities=(),
        structure_node_identities=(node.identity,),
        projection_kind=_concept(envelope.source, 16070, 20),
        semantic_object=proposition_identity(envelope.source, (16070, 21)),
        lifecycle_state=_concept(envelope.source, 16070, 22),
        hypothesis=hypothesis,
        evidence=evidence,
        directions=(PROJECTION_UNDERSTANDING,),
        projection_key=(16070, 23),
    )
    assert ArtifactSemanticProjection.from_stable_key(
        node_only.stable_key()) == node_only
    with pytest.raises(ValueError, match="排序去重"):
        replace(projection, directions=(PROJECTION_REASONING,
                                        PROJECTION_UNDERSTANDING))
    foreign = _hypothesis(_source(1, source_id=16003), seed=2)
    with pytest.raises(ValueError, match="归同一 hypothesis"):
        replace(projection, evidence=(_evidence(foreign),))
    with pytest.raises(ValueError, match="FormalArtifact"):
        replace(projection, semantic_object=anchor.identity)


def test_carrier_revision_round_trips_split_mapping_across_parser_versions():
    old_envelope = _envelope(_source(1), text="甲乙")
    new_envelope = _envelope(_source(2), text="甲 乙")
    old_anchor = _anchor(old_envelope, coordinates=(0, 2), key=(16080, 1))
    new_left = _anchor(new_envelope, coordinates=(0, 1), key=(16080, 2))
    new_right = _anchor(new_envelope, coordinates=(2, 3), key=(16080, 3))
    mapping = ArtifactRevisionMapping(
        REVISION_MAP_ANCHOR,
        old_anchor.identity,
        tuple(sorted(
            (new_left.identity, new_right.identity),
            key=ObjectIdentity.stable_key)),
    )
    hypothesis = _hypothesis(new_envelope.source, seed=8)
    revision = make_artifact_carrier_revision(
        old_envelope_identity=old_envelope.identity,
        new_envelope_identity=new_envelope.identity,
        reason=_concept(new_envelope.source, 16080, 4),
        hypothesis=hypothesis,
        mappings=(mapping,),
        evidence=(_evidence(hypothesis, 8),),
        revision_key=(16080, 5),
    )
    assert ArtifactRevisionMapping.from_stable_key(
        mapping.stable_key()) == mapping
    assert ArtifactCarrierRevision.from_stable_key(
        revision.stable_key()) == revision
    assert artifact_carrier_local_kind(revision.identity) == (
        ARTIFACT_CARRIER_IDENTITY_REVISION)

    deletion = ArtifactRevisionMapping(
        REVISION_MAP_ANCHOR,
        old_anchor.identity,
        (),
    )
    assert ArtifactRevisionMapping.from_stable_key(
        deletion.stable_key()) == deletion
    deleted = make_artifact_carrier_revision(
        old_envelope_identity=old_envelope.identity,
        new_envelope_identity=new_envelope.identity,
        reason=_concept(new_envelope.source, 16080, 6),
        hypothesis=hypothesis,
        mappings=(deletion,),
        evidence=(_evidence(hypothesis, 8),),
        revision_key=(16080, 7),
    )
    assert ArtifactCarrierRevision.from_stable_key(
        deleted.stable_key()) == deleted

    old_right = _anchor(
        old_envelope, coordinates=(1, 2), key=(16080, 8))
    merge_mappings = tuple(sorted((
        ArtifactRevisionMapping(
            REVISION_MAP_ANCHOR, old_anchor.identity, (new_left.identity,)),
        ArtifactRevisionMapping(
            REVISION_MAP_ANCHOR, old_right.identity, (new_left.identity,)),
    ), key=ArtifactRevisionMapping.stable_key))
    merged = make_artifact_carrier_revision(
        old_envelope_identity=old_envelope.identity,
        new_envelope_identity=new_envelope.identity,
        reason=_concept(new_envelope.source, 16080, 9),
        hypothesis=hypothesis,
        mappings=merge_mappings,
        evidence=(_evidence(hypothesis, 8),),
        revision_key=(16080, 10),
    )
    assert ArtifactCarrierRevision.from_stable_key(
        merged.stable_key()) == merged


def test_carrier_revision_identity_binds_mapping_topology():
    old_envelope = _envelope(_source(1), text="甲乙")
    new_envelope = _envelope(_source(2), text="甲 乙")
    old_anchor = _anchor(old_envelope, coordinates=(0, 2), key=(16085, 1))
    new_left = _anchor(new_envelope, coordinates=(0, 1), key=(16085, 2))
    new_right = _anchor(new_envelope, coordinates=(2, 3), key=(16085, 3))
    hypothesis = _hypothesis(new_envelope.source, seed=85)
    common = dict(
        old_envelope_identity=old_envelope.identity,
        new_envelope_identity=new_envelope.identity,
        reason=_concept(new_envelope.source, 16085, 4),
        hypothesis=hypothesis,
        evidence=(_evidence(hypothesis, 85),),
        revision_key=(16085, 5),
    )
    left = make_artifact_carrier_revision(
        mappings=(ArtifactRevisionMapping(
            REVISION_MAP_ANCHOR, old_anchor.identity, (new_left.identity,)),),
        **common,
    )
    right = make_artifact_carrier_revision(
        mappings=(ArtifactRevisionMapping(
            REVISION_MAP_ANCHOR, old_anchor.identity, (new_right.identity,)),),
        **common,
    )
    assert left.identity != right.identity


def test_carrier_revision_rejects_cross_lineage_and_same_parser_version():
    old_envelope = _envelope(_source(1), text="甲乙")
    same_parser = _envelope(_source(1), text="甲 乙")
    hypothesis = _hypothesis(same_parser.source, seed=9)
    with pytest.raises(ValueError, match="parser version 必须不同"):
        make_artifact_carrier_revision(
            old_envelope_identity=old_envelope.identity,
            new_envelope_identity=same_parser.identity,
            reason=_concept(same_parser.source, 16090, 1),
            hypothesis=hypothesis,
            mappings=(ArtifactRevisionMapping(
                REVISION_MAP_ANCHOR,
                _anchor(old_envelope).identity,
                (_anchor(same_parser).identity,),
            ),),
            evidence=(_evidence(hypothesis, 9),),
            revision_key=(16090, 2),
        )

    other_lineage = _envelope(_source(2, source_id=16099), text="甲 乙")
    with pytest.raises(ValueError, match="parser lineage"):
        make_artifact_carrier_revision(
            old_envelope_identity=old_envelope.identity,
            new_envelope_identity=other_lineage.identity,
            reason=_concept(other_lineage.source, 16090, 3),
            hypothesis=_hypothesis(other_lineage.source, seed=10),
            mappings=(ArtifactRevisionMapping(
                REVISION_MAP_ANCHOR,
                _anchor(old_envelope).identity,
                (_anchor(other_lineage).identity,),
            ),),
            evidence=(_evidence(_hypothesis(other_lineage.source, seed=10), 10),),
            revision_key=(16090, 4),
        )


def test_carrier_revision_rejects_mapping_from_other_same_source_envelope():
    old_source = _source(1)
    old_envelope = _envelope(old_source, text="甲乙")
    unrelated_old = _envelope(old_source, text="丙丁")
    new_envelope = _envelope(_source(2), text="甲 乙")
    hypothesis = _hypothesis(new_envelope.source, seed=11)
    mapping = ArtifactRevisionMapping(
        REVISION_MAP_ANCHOR,
        _anchor(unrelated_old).identity,
        (_anchor(new_envelope).identity,),
    )
    with pytest.raises(ValueError, match="不属于指定 envelope"):
        make_artifact_carrier_revision(
            old_envelope_identity=old_envelope.identity,
            new_envelope_identity=new_envelope.identity,
            reason=_concept(new_envelope.source, 16095, 1),
            hypothesis=hypothesis,
            mappings=(mapping,),
            evidence=(_evidence(hypothesis, 11),),
            revision_key=(16095, 2),
        )


def test_local_kind_rejects_plain_artifact_object_and_truncated_stable_keys():
    source = _source(1)
    plain = ObjectIdentity(
        9, (1, *source.stable_key(), 1), source.owner, source.versions)
    with pytest.raises(ValueError, match="magic/kind"):
        artifact_carrier_local_kind(plain)
    forged = ObjectIdentity(
        9,
        (16001, ARTIFACT_CARRIER_IDENTITY_ANCHOR, *source.stable_key()),
        source.owner,
        source.versions,
    )
    with pytest.raises(ValueError, match="被截断|缺少"):
        artifact_carrier_local_kind(forged)
    envelope = _envelope(source)
    with pytest.raises(ValueError):
        ArtifactEnvelope.from_stable_key(envelope.stable_key()[:-1])
