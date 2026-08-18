from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_family import (
    FAMILY_CODE_ROOT_MODULES,
    FAMILY_FREEZE_MANIFEST_NAME,
    FAMILY_FREEZE_STATUS,
    PUBLIC_PREFLIGHT_MANIFEST_SHA256,
    ConflictSetFamilyFreeze,
    ConflictSetFamilyFreezeError,
    assert_conflict_set_family_freeze_matches_live_public_code,
    build_conflict_set_family_code_identity,
    build_conflict_set_family_freeze,
    parse_conflict_set_family_freeze_bytes,
    read_conflict_set_family_freeze,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_handoff import (
    ARTIFACT_ROLES,
    CAPABILITY_KEY,
    CODE_IDENTITY,
    EVALUATOR_OWNER,
    FAMILY_NAMESPACE,
    SOURCE_OWNER,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_metadata import (
    OWNER_METADATA_ARTIFACT_KIND,
    OWNER_METADATA_FORMAT_VERSION,
    OWNER_METADATA_STATUS,
    parse_conflict_set_owner_metadata_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    TRANSPORT_ROOT_NAMESPACE,
    ConflictSetPrivateArtifact,
    build_conflict_set_private_transport,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_public_preflight import (
    read_conflict_set_public_preflight,
)


_REPOSITORY = Path(__file__).parents[1]
_PREFLIGHT_PATH = (
    _REPOSITORY
    / "data/ph2/manifests/gg03_conflict_set_public_preflight_v2.json"
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


def _transport(code_identity, public_preflight):
    code_payload = code_identity.canonical_bytes()
    code_sha = hashlib.sha256(code_payload).hexdigest()
    public_payload = public_preflight.canonical_bytes()
    content_sha = {
        "code_freeze": code_sha,
        "observation_pack": "2" * 64,
        "source_manifest": "3" * 64,
        "candidate_manifest": "4" * 64,
        "public_preflight": PUBLIC_PREFLIGHT_MANIFEST_SHA256,
        "private_labels": "6" * 64,
    }
    content_size = {
        "code_freeze": len(code_payload),
        "public_preflight": len(public_payload),
    }
    artifacts = tuple(
        ConflictSetPrivateArtifact(
            role,
            *_SPECS[role],
            f"{TRANSPORT_ROOT_NAMESPACE}/{role}.json",
            *(
                (None, 0, None, 0, 0, "RESERVED")
                if role in _RESERVED_ROLES else
                (
                    content_sha[role]
                    if role in {"code_freeze", "public_preflight"}
                    else f"{index:064x}",
                    content_size.get(role, index + 10),
                    content_sha[role],
                    content_size.get(role, index + 20),
                    2 if role in {"observation_pack", "private_labels"}
                    else 1,
                    "MATERIALIZED",
                )
            ),
        )
        for index, role in enumerate(ARTIFACT_ROLES, start=1)
    )
    return build_conflict_set_private_transport(
        public_preflight_manifest_sha256=(
            PUBLIC_PREFLIGHT_MANIFEST_SHA256),
        observation_pack_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        candidate_manifest_sha256="4" * 64,
        artifacts=artifacts,
    )


def _metadata(transport):
    value = {
        "artifact_kind": OWNER_METADATA_ARTIFACT_KIND,
        "artifacts": [item.to_dict() for item in transport.artifacts],
        "capability_key": CAPABILITY_KEY,
        "clone_evaluation_writes_before": 0,
        "code_identity": CODE_IDENTITY,
        "family_namespace": FAMILY_NAMESPACE,
        "formal_run_count_before": 0,
        "format_version": OWNER_METADATA_FORMAT_VERSION,
        "host_learning_writes_before": 0,
        "label_writes_before": 0,
        "owner_id": EVALUATOR_OWNER,
        "private_payload_reads_before": 0,
        "source_owner": SOURCE_OWNER,
        "status": OWNER_METADATA_STATUS,
        "teacher_api_llm_calls_before": 0,
        "transport_commitment_sha256": transport.commitment_sha256(),
    }
    value["owner_receipt_sha256"] = hashlib.sha256(
        canonical_json_line(value).rstrip(b"\n")).hexdigest()
    return parse_conflict_set_owner_metadata_bytes(canonical_json_line(value))


def _freeze():
    public = read_conflict_set_public_preflight(_PREFLIGHT_PATH)
    code = build_conflict_set_family_code_identity(_REPOSITORY)
    transport = _transport(code, public)
    metadata = _metadata(transport)
    return build_conflict_set_family_freeze(
        public_preflight=public,
        family_code_identity=code,
        transport=transport,
        owner_metadata=metadata,
    )


def test_family_code_identity_is_deterministic_and_covers_all_roots():
    first = build_conflict_set_family_code_identity(_REPOSITORY)
    second = build_conflict_set_family_code_identity(_REPOSITORY)
    assert first == second
    paths = {item.relative_path for item in first.files}
    for module in FAMILY_CODE_ROOT_MODULES:
        path = f"src/{Path(*module.split('.')).with_suffix('.py').as_posix()}"
        assert path in paths


def test_family_freeze_round_trip_closes_all_pre_run_commitments():
    freeze = _freeze()
    parsed = parse_conflict_set_family_freeze_bytes(freeze.canonical_bytes())
    assert parsed == freeze
    assert freeze.status == FAMILY_FREEZE_STATUS
    assert freeze.unique_formal_run_limit == 1
    assert freeze.available_guard.state == "AVAILABLE"
    assert freeze.transport.artifacts == freeze.owner_metadata.artifacts
    assert all(getattr(freeze, field) == 0 for field in (
        "formal_run_count_before", "private_payload_reads_before",
        "host_learning_writes_before", "label_writes_before",
        "clone_evaluation_writes_before", "teacher_api_llm_calls_before",
    ))


def test_family_freeze_live_public_code_recheck_is_read_only():
    freeze = _freeze()
    assert_conflict_set_family_freeze_matches_live_public_code(
        freeze, _REPOSITORY)


def test_family_freeze_rejects_code_freeze_artifact_drift():
    public = read_conflict_set_public_preflight(_PREFLIGHT_PATH)
    code = build_conflict_set_family_code_identity(_REPOSITORY)
    transport = _transport(code, public)
    broken = replace(transport.artifacts[0], content_sha256="9" * 64)
    transport = replace(
        transport, artifacts=(broken,) + transport.artifacts[1:])
    metadata = _metadata(transport)
    with pytest.raises(ConflictSetFamilyFreezeError):
        build_conflict_set_family_freeze(
            public_preflight=public,
            family_code_identity=code,
            transport=transport,
            owner_metadata=metadata,
        )


def test_family_freeze_rejects_public_preflight_identity_drift():
    freeze = _freeze()
    changed = replace(freeze.public_preflight, public_head_sha1="f" * 40)
    with pytest.raises(ConflictSetFamilyFreezeError):
        replace(freeze, public_preflight=changed)


def test_family_freeze_rejects_guard_inventory_and_counter_drift():
    freeze = _freeze()
    with pytest.raises(ConflictSetFamilyFreezeError):
        replace(
            freeze,
            available_guard=replace(freeze.available_guard, state="CONSUMED"),
        )
    with pytest.raises(ConflictSetFamilyFreezeError):
        replace(freeze, artifact_inventory_sha256="8" * 64)
    with pytest.raises(ConflictSetFamilyFreezeError):
        replace(freeze, formal_run_count_before=1)
    with pytest.raises(ConflictSetFamilyFreezeError):
        replace(freeze, unique_formal_run_limit=2)


def test_family_freeze_rejects_self_commitment_drift():
    freeze = _freeze()
    with pytest.raises(ConflictSetFamilyFreezeError):
        replace(freeze, family_commitment_sha256="a" * 64)


def test_family_freeze_parser_rejects_unknown_and_noncanonical_fields():
    freeze = _freeze()
    value = freeze.to_dict()
    value["labels"] = []
    with pytest.raises(ConflictSetFamilyFreezeError):
        ConflictSetFamilyFreeze.from_dict(value)
    with pytest.raises(ConflictSetFamilyFreezeError):
        parse_conflict_set_family_freeze_bytes(
            freeze.canonical_bytes() + b"\n")


def test_family_freeze_file_reader_is_name_bound_and_canonical(tmp_path):
    freeze = _freeze()
    target = tmp_path / FAMILY_FREEZE_MANIFEST_NAME
    target.write_bytes(freeze.canonical_bytes())
    assert read_conflict_set_family_freeze(target) == freeze
    wrong = tmp_path / "other.json"
    wrong.write_bytes(freeze.canonical_bytes())
    with pytest.raises(ConflictSetFamilyFreezeError):
        read_conflict_set_family_freeze(wrong)
