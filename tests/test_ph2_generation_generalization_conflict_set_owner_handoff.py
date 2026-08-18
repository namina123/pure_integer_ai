from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_handoff import (
    CAPABILITY_KEY,
    CODE_IDENTITY,
    EVALUATOR_OWNER,
    FAMILY_NAMESPACE,
    GENERATION_CONTRACT,
    HANDOFF_STATUS,
    NEGATIVE_MATRIX_CASE_COUNT,
    PROJECTION_VERSION,
    RESPONSE_ACT,
    SOURCE_OWNER,
    ConflictSetOwnerHandoff,
    ConflictSetOwnerHandoffError,
    ConflictSetOwnerObservation,
    ConflictSetObservationClaim,
    ConflictSetObservationSourceBinding,
    ConflictSetPublicPreflight,
    ConflictSetResourceBudget,
    ConflictSetSourceManifestEntry,
    ConflictSetSplitAxes,
    expected_conflict_set_artifact_roles,
    parse_conflict_set_owner_handoff_bytes,
    read_conflict_set_owner_handoff,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_public_preflight import (
    ARTIFACT_KIND as PREFLIGHT_ARTIFACT_KIND,
    PUBLIC_PREFLIGHT_STATUS,
    build_conflict_set_code_identity,
    build_conflict_set_public_preflight_freeze,
    parse_conflict_set_public_preflight_bytes,
    read_conflict_set_public_preflight,
    ConflictSetPublicPreflightError,
)


def _handoff() -> ConflictSetOwnerHandoff:
    manifests = []
    observations = []
    source_number = 0
    for split_index, split in enumerate(("train", "dev", "held_out"), start=1):
        observation_id = f"{split}-01"
        cluster = f"cluster-{split}-01"
        bindings = []
        claims = []
        for claim_index, claim_id in enumerate(("claim-a", "claim-b"), start=1):
            source_ids = []
            for stance_index in (1, 2):
                source_number += 1
                source_id = f"src-{split}-{claim_index}-{stance_index}"
                source_ids.append(source_id)
                source = SourceRef(
                    94001,
                    source_number,
                    split_index,
                    GLOBAL_OWNER_SCOPE,
                    VersionBundle(),
                )
                manifests.append(ConflictSetSourceManifestEntry(
                    source_id,
                    source,
                    cluster,
                    (split_index,),
                    f"{source_number:064x}",
                    "AUTHORED_PUBLIC_DOCUMENT",
                    "CC0-1.0",
                    "PUBLIC",
                    split,
                ))
                bindings.append(ConflictSetObservationSourceBinding(
                    source_id, source))
            claims.append(ConflictSetObservationClaim(
                claim_id,
                (split_index, claim_index),
                tuple(sorted(source_ids)),
                f"surface-{claim_index}",
            ))
        bindings = tuple(sorted(bindings, key=lambda item: item.source_id))
        observations.append(ConflictSetOwnerObservation(
            observation_id,
            split_index,
            RESPONSE_ACT,
            split,
            ("claim-a", "claim-b"),
            tuple(claims),
            bindings,
            GENERATION_CONTRACT,
            ConflictSetSplitAxes(
                2,
                4,
                "surface-family-a",
                "order-family-a",
                (cluster,),
                "lexical-structure-a",
            ),
            ConflictSetResourceBudget(2, 4, 2, 128, 8),
        ))

    return ConflictSetOwnerHandoff(
        CAPABILITY_KEY,
        CODE_IDENTITY,
        EVALUATOR_OWNER,
        SOURCE_OWNER,
        FAMILY_NAMESPACE,
        RESPONSE_ACT,
        PROJECTION_VERSION,
        ConflictSetPublicPreflight(
            "a" * 40,
            "PASS",
            1,
            2,
            NEGATIVE_MATRIX_CASE_COUNT,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
        ),
        tuple(sorted(manifests, key=lambda item: item.source_id)),
        tuple(observations),
        expected_conflict_set_artifact_roles(),
        1,
        HANDOFF_STATUS,
    )


def test_conflict_set_owner_handoff_round_trips_canonical_bytes():
    handoff = _handoff()
    parsed = parse_conflict_set_owner_handoff_bytes(handoff.canonical_bytes())
    assert parsed == handoff
    assert len(handoff.artifact_roles) == 10
    assert handoff.sha256() == parsed.sha256()


def test_public_sample_is_canonical_and_source_clusters_are_split_isolated():
    sample = (
        Path(__file__).parents[1]
        / "data/ph2/conflict_set_owner_handoff_v1.jsonl.sample"
    )
    handoff = read_conflict_set_owner_handoff(sample)
    assert handoff.canonical_bytes() == sample.read_bytes()
    assert len(handoff.source_manifest) == 12
    assert tuple(item.split for item in handoff.observations) == (
        "train", "dev", "held_out")
    clusters = {
        split: {
            item.source_cluster_id
            for item in handoff.source_manifest
            if item.split == split
        }
        for split in ("train", "dev", "held_out")
    }
    assert clusters["train"].isdisjoint(clusters["dev"])
    assert clusters["train"].isdisjoint(clusters["held_out"])
    assert clusters["dev"].isdisjoint(clusters["held_out"])


def test_public_preflight_freeze_binds_sample_and_code_closure():
    repository = Path(__file__).parents[1]
    sample = repository / "data/ph2/conflict_set_owner_handoff_v1.jsonl.sample"
    identity = build_conflict_set_code_identity(repository)
    assert any(
        item.relative_path.endswith(
            "ph2_generation_generalization_conflict_set_owner_handoff.py")
        for item in identity.files
    )
    freeze = build_conflict_set_public_preflight_freeze(
        repository, public_head_sha1="b" * 40)
    assert freeze.code_identity == identity
    assert freeze.sample_size_bytes == sample.stat().st_size
    assert freeze.status == PUBLIC_PREFLIGHT_STATUS
    assert freeze.teacher_api_llm_call_count == 0
    assert freeze.private_label_read_count == 0
    assert freeze.formal_run_count == 0
    assert parse_conflict_set_public_preflight_bytes(
        freeze.canonical_bytes()) == freeze


def test_public_preflight_freeze_rejects_identity_drift():
    repository = Path(__file__).parents[1]
    freeze = build_conflict_set_public_preflight_freeze(
        repository, public_head_sha1="c" * 40)
    value = freeze.to_dict()
    value["artifact_kind"] = PREFLIGHT_ARTIFACT_KIND + "_OLD"
    with pytest.raises(ConflictSetPublicPreflightError):
        type(freeze).from_dict(value)


def test_public_preflight_manifest_is_canonical_and_anchored():
    manifest = (
        Path(__file__).parents[1]
        / "data/ph2/manifests/gg03_conflict_set_public_preflight_v2.json"
    )
    freeze = read_conflict_set_public_preflight(manifest)
    assert freeze.canonical_bytes() == manifest.read_bytes()
    assert freeze.public_head_sha1 == (
        "b057fc098667f438b690481653c1c501a8ba7cd2")
    assert freeze.sample_sha256 == (
        "d6250c956b20a30912bf335b4a88904e2ad80445091f4bdc97f518bfec1c2082")
    assert freeze.formal_run_count == 0
    live = build_conflict_set_public_preflight_freeze(
        Path(__file__).parents[1], public_head_sha1=freeze.public_head_sha1)
    assert live == freeze
    legacy = read_conflict_set_public_preflight(
        Path(__file__).parents[1]
        / "data/ph2/manifests/gg03_conflict_set_public_preflight_v1.json")
    assert legacy.sha256() == (
        "5e1ba013d2889108169678370319cc43e7f492dd2b2f1a53d88a678767afa7f4")
    assert legacy.code_identity.aggregate_sha256 == (
        "954443b882014c784ebb277295fd1a322e522ab02e65d38c3b93c896cf8d7ff6")


def test_conflict_set_owner_handoff_is_label_free_and_rejects_unknown_fields():
    value = _handoff().to_dict()
    value["observations"][0]["labels"] = []
    with pytest.raises(ConflictSetOwnerHandoffError):
        ConflictSetOwnerHandoff.from_dict(value)


@pytest.mark.parametrize("field", ("code_identity", "family_namespace"))
def test_conflict_set_owner_handoff_rejects_legacy_identity(field):
    value = _handoff().to_dict()
    value[field] = "PH2_GG03_EXECUTABLE_SEMANTIC_EVALUATION_FAMILY_FREEZE_V2"
    with pytest.raises(ConflictSetOwnerHandoffError):
        ConflictSetOwnerHandoff.from_dict(value)


def test_conflict_set_owner_handoff_requires_held_out_and_complete_artifacts():
    value = _handoff().to_dict()
    value["observations"] = value["observations"][:2]
    with pytest.raises(ConflictSetOwnerHandoffError):
        ConflictSetOwnerHandoff.from_dict(value)

    value = _handoff().to_dict()
    value["artifact_roles"] = value["artifact_roles"][:-1]
    with pytest.raises(ConflictSetOwnerHandoffError):
        ConflictSetOwnerHandoff.from_dict(value)


def test_conflict_set_owner_handoff_rejects_source_ref_drift_and_duplicate_mapping():
    value = _handoff().to_dict()
    value["observations"][0]["source_bindings"][0]["source_ref"] = [
        94001, 99999, 1, 0, 0, 0, 1, 0, 0, 0, 0,
    ]
    with pytest.raises(ConflictSetOwnerHandoffError):
        ConflictSetOwnerHandoff.from_dict(value)

    handoff = _handoff()
    duplicate = replace(
        handoff.source_manifest[1], source=handoff.source_manifest[0].source)
    with pytest.raises(ConflictSetOwnerHandoffError):
        replace(handoff, source_manifest=(
            handoff.source_manifest[0], duplicate, *handoff.source_manifest[2:]))


def test_conflict_set_owner_handoff_rejects_cross_split_source_cluster_leak():
    handoff = _handoff()
    held_out = handoff.observations[2]
    train_cluster = handoff.observations[0].split_axes.source_cluster_ids[0]
    changed_axes = replace(
        held_out.split_axes, source_cluster_ids=(train_cluster,))
    with pytest.raises(ConflictSetOwnerHandoffError):
        replace(handoff, observations=(
            handoff.observations[0], handoff.observations[1],
            replace(held_out, split_axes=changed_axes),
        ))


def test_conflict_set_owner_handoff_rejects_noncanonical_json_bytes():
    payload = _handoff().canonical_bytes()
    with pytest.raises(ConflictSetOwnerHandoffError):
        parse_conflict_set_owner_handoff_bytes(payload + b"\n")
