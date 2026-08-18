from dataclasses import replace
import hashlib

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_handoff import (
    ARTIFACT_ROLES,
    CODE_IDENTITY,
    EVALUATOR_OWNER,
    SOURCE_OWNER,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_metadata import (
    OWNER_METADATA_ARTIFACT_KIND,
    OWNER_METADATA_FORMAT_VERSION,
    OWNER_METADATA_STATUS,
    ConflictSetOwnerMetadataError,
    build_conflict_set_private_transport_from_owner_metadata,
    build_conflict_set_run_guard_from_owner_metadata,
    parse_conflict_set_owner_metadata_bytes,
    read_conflict_set_owner_metadata,
    validate_conflict_set_owner_metadata,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    TRANSPORT_ROOT_NAMESPACE,
    ConflictSetPrivateArtifact,
    build_conflict_set_private_transport,
)


_SPECS = {
    "code_freeze": (CODE_IDENTITY, "PUBLIC", "CODE_FREEZE"),
    "observation_pack": (SOURCE_OWNER, "PUBLIC", "OBSERVATION"),
    "source_manifest": (SOURCE_OWNER, "PUBLIC", "SOURCE"),
    "candidate_manifest": (CODE_IDENTITY, "PUBLIC", "CANDIDATE"),
    "public_preflight": (CODE_IDENTITY, "PUBLIC", "PUBLIC_PREFLIGHT"),
    "private_labels": (EVALUATOR_OWNER, "PRIVATE", "PRIVATE_LABEL"),
    "prediction_seal": (EVALUATOR_OWNER, "PRIVATE", "PREDICTION"),
    "aggregate_report": (EVALUATOR_OWNER, "PUBLIC", "PUBLICATION"),
    "runtime_receipt": (EVALUATOR_OWNER, "PUBLIC", "PUBLICATION"),
    "formal_failure_report": (EVALUATOR_OWNER, "PUBLIC", "PUBLICATION"),
}
_RESERVED_ROLES = frozenset({
    "prediction_seal", "aggregate_report", "runtime_receipt",
    "formal_failure_report",
})


def _transport():
    content_sha = {
        "observation_pack": "2" * 64,
        "source_manifest": "3" * 64,
        "candidate_manifest": "4" * 64,
        "public_preflight": "1" * 64,
    }
    artifacts = tuple(
        ConflictSetPrivateArtifact(
            role,
            *_SPECS[role],
            f"{TRANSPORT_ROOT_NAMESPACE}/{role}.json",
            *(
                (None, 0, None, 0, 0, "RESERVED")
                if role in _RESERVED_ROLES else
                (f"{index:064x}", index,
                 content_sha.get(role, f"{index + 20:064x}"),
                 index + 20,
                 2 if role in {"observation_pack", "private_labels"} else 1,
                 "MATERIALIZED")
            ),
        )
        for index, role in enumerate(ARTIFACT_ROLES, start=1)
    )
    return build_conflict_set_private_transport(
        public_preflight_manifest_sha256="1" * 64,
        observation_pack_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        candidate_manifest_sha256="4" * 64,
        artifacts=artifacts,
    )


def _metadata(transport):
    value = {
        "artifact_kind": OWNER_METADATA_ARTIFACT_KIND,
        "artifacts": [item.to_dict() for item in transport.artifacts],
        "capability_key": "GG03_CONFLICT_SET_RUNTIME",
        "clone_evaluation_writes_before": 0,
        "code_identity": "GG03_CONFLICT_SET_PUBLIC_V1",
        "family_namespace": "GG03_CONFLICT_SET_FORMAL_V1",
        "formal_run_count_before": 0,
        "format_version": OWNER_METADATA_FORMAT_VERSION,
        "host_learning_writes_before": 0,
        "label_writes_before": 0,
        "private_payload_reads_before": 0,
        "source_owner": SOURCE_OWNER,
        "status": OWNER_METADATA_STATUS,
        "teacher_api_llm_calls_before": 0,
        "transport_commitment_sha256": transport.commitment_sha256(),
        "owner_id": EVALUATOR_OWNER,
    }
    value["owner_receipt_sha256"] = hashlib.sha256(
        canonical_json_line(value).rstrip(b"\n")).hexdigest()
    return parse_conflict_set_owner_metadata_bytes(canonical_json_line(value))


def test_owner_metadata_round_trip_closes_transport_and_guard(tmp_path):
    transport = _transport()
    metadata = _metadata(transport)
    assert metadata.status == OWNER_METADATA_STATUS
    assert parse_conflict_set_owner_metadata_bytes(
        metadata.canonical_bytes()) == metadata
    assert metadata.metadata_sha256() == hashlib.sha256(
        metadata.canonical_bytes()).hexdigest()
    validate_conflict_set_owner_metadata(transport, metadata)
    guard = build_conflict_set_run_guard_from_owner_metadata(
        transport, metadata)
    assert guard.owner_receipt_sha256 == metadata.owner_receipt_sha256
    target = tmp_path / "owner-receipt.jsonl"
    target.write_bytes(metadata.canonical_bytes())
    assert read_conflict_set_owner_metadata(target) == metadata


def test_candidate_transport_assembly_closes_owner_commitment():
    original = _transport()
    metadata = _metadata(original)
    assembled = build_conflict_set_private_transport_from_owner_metadata(
        metadata,
        public_preflight_manifest_sha256="1" * 64,
        observation_pack_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        candidate_manifest_sha256="4" * 64,
    )
    assert assembled == original
    with pytest.raises(ConflictSetOwnerMetadataError):
        build_conflict_set_private_transport_from_owner_metadata(
            metadata,
            public_preflight_manifest_sha256="9" * 64,
            observation_pack_sha256="2" * 64,
            source_manifest_sha256="3" * 64,
            candidate_manifest_sha256="4" * 64,
        )


def test_owner_metadata_receipt_and_transport_commitments_are_self_consistent():
    transport = _transport()
    metadata = _metadata(transport)
    with pytest.raises(ConflictSetOwnerMetadataError):
        replace(metadata, owner_receipt_sha256="a" * 64)
    with pytest.raises(ConflictSetOwnerMetadataError):
        replace(metadata, transport_commitment_sha256="b" * 64)
    with pytest.raises(ConflictSetOwnerMetadataError):
        validate_conflict_set_owner_metadata(
            transport, replace(
                metadata,
                transport_commitment_sha256="c" * 64,
                owner_receipt_sha256=metadata.owner_receipt_sha256,
            ))


def test_owner_metadata_rejects_identity_status_and_unknown_fields():
    metadata = _metadata(_transport())
    value = metadata.to_dict()
    value["labels"] = []
    with pytest.raises(ConflictSetOwnerMetadataError):
        type(metadata).from_dict(value)
    with pytest.raises(ConflictSetOwnerMetadataError):
        replace(metadata, status="FORMAL_READY")
    with pytest.raises(ConflictSetOwnerMetadataError):
        replace(metadata, owner_id="OLD_OWNER")


def test_owner_metadata_rejects_artifact_drift_and_nonzero_audits():
    metadata = _metadata(_transport())
    broken = replace(metadata.artifacts[0], relative_path="gg03-conflict-set-v1/other.json")
    with pytest.raises(ConflictSetOwnerMetadataError):
        replace(metadata, artifacts=(broken,) + metadata.artifacts[1:])
    broken_label = replace(metadata.artifacts[5], record_count=3)
    with pytest.raises(ConflictSetOwnerMetadataError):
        replace(metadata, artifacts=metadata.artifacts[:5] + (broken_label,)
                + metadata.artifacts[6:])
    for field in (
            "formal_run_count_before", "private_payload_reads_before",
            "host_learning_writes_before", "label_writes_before",
            "clone_evaluation_writes_before",
            "teacher_api_llm_calls_before"):
        with pytest.raises(ConflictSetOwnerMetadataError):
            replace(metadata, **{field: 1})


def test_owner_metadata_parser_rejects_noncanonical_bytes():
    metadata = _metadata(_transport())
    value = metadata.to_dict()
    payload = canonical_json_line(value) + b"\n"
    with pytest.raises(ConflictSetOwnerMetadataError):
        parse_conflict_set_owner_metadata_bytes(payload)
    with pytest.raises(ConflictSetOwnerMetadataError):
        parse_conflict_set_owner_metadata_bytes(b"{}")


def test_owner_metadata_reader_rejects_non_owner_filename(tmp_path):
    metadata = _metadata(_transport())
    wrong = tmp_path / "labels.jsonl"
    wrong.write_bytes(metadata.canonical_bytes())
    with pytest.raises(ConflictSetOwnerMetadataError):
        read_conflict_set_owner_metadata(wrong)
