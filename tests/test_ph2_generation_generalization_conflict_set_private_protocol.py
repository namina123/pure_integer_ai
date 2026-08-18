from dataclasses import replace

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_owner_handoff import (
    ARTIFACT_ROLES,
    CODE_IDENTITY,
    EVALUATOR_OWNER,
    SOURCE_OWNER,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    TRANSPORT_ROOT_NAMESPACE,
    ConflictSetPrivateArtifact,
    ConflictSetPrivateProtocolError,
    assert_conflict_set_transport_matches_public_freeze,
    build_conflict_set_private_transport,
    build_conflict_set_run_guard,
    consume_conflict_set_run_guard,
    parse_conflict_set_private_transport_bytes,
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
                 index + 20, 1, "MATERIALIZED")
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


def test_private_transport_round_trip_is_metadata_only():
    transport = _transport()
    assert parse_conflict_set_private_transport_bytes(
        canonical_json_line(transport.to_dict())) == transport
    assert transport.commitment_sha256() != ""
    assert_conflict_set_transport_matches_public_freeze(
        transport,
        public_preflight_manifest_sha256="1" * 64,
        observation_pack_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        candidate_manifest_sha256="4" * 64,
    )


def test_private_transport_rejects_incomplete_role_inventory():
    transport = _transport()
    with pytest.raises(ConflictSetPrivateProtocolError):
        replace(transport, artifacts=transport.artifacts[:-1])


def test_private_transport_rejects_role_visibility_or_owner_drift():
    transport = _transport()
    with pytest.raises(ConflictSetPrivateProtocolError):
        broken = replace(transport.artifacts[5], visibility="PUBLIC")
        replace(transport, artifacts=transport.artifacts[:5] + (broken,)
                + transport.artifacts[6:])
    for field in ("owner", "phase"):
        with pytest.raises(ConflictSetPrivateProtocolError):
            broken = replace(transport.artifacts[5], **{field: "DRIFT"})
            replace(transport, artifacts=transport.artifacts[:5] + (broken,)
                    + transport.artifacts[6:])


def test_private_transport_keeps_future_outputs_reserved_until_formal_run():
    transport = _transport()
    for role in _RESERVED_ROLES:
        item = next(row for row in transport.artifacts if row.role == role)
        assert item.materialization == "RESERVED"
        assert item.content_sha256 is None
        assert item.record_count == 0
    with pytest.raises(ConflictSetPrivateProtocolError):
        ConflictSetPrivateArtifact(
            "runtime_receipt", *_SPECS["runtime_receipt"],
            f"{TRANSPORT_ROOT_NAMESPACE}/runtime_receipt.json",
            "a" * 64, 1, "b" * 64, 1, 1, "RESERVED")


def test_private_transport_rejects_legacy_path_and_commitment_drift():
    transport = _transport()
    with pytest.raises(ConflictSetPrivateProtocolError):
        broken = replace(
            transport.artifacts[0],
            relative_path=(
                f"{TRANSPORT_ROOT_NAMESPACE}/formal-family-20260818-a/code.json"),
        )
        replace(transport, artifacts=(broken,) + transport.artifacts[1:])
    with pytest.raises(ConflictSetPrivateProtocolError):
        assert_conflict_set_transport_matches_public_freeze(
            transport,
            public_preflight_manifest_sha256="9" * 64,
            observation_pack_sha256="2" * 64,
            source_manifest_sha256="3" * 64,
            candidate_manifest_sha256="4" * 64,
        )
    with pytest.raises(ConflictSetPrivateProtocolError):
        assert_conflict_set_transport_matches_public_freeze(
            transport,
            public_preflight_manifest_sha256="1" * 64,
            observation_pack_sha256="2" * 64,
            source_manifest_sha256="3" * 64,
            candidate_manifest_sha256="9" * 64,
        )
    with pytest.raises(ConflictSetPrivateProtocolError):
        replace(transport, family_namespace="PH2_GG03_EXECUTABLE_EVALUATION_FAMILY_FREEZE_V1")
    with pytest.raises(ConflictSetPrivateProtocolError):
        broken = replace(transport.artifacts[1], relative_path=transport.artifacts[0].relative_path)
        replace(transport, artifacts=(transport.artifacts[0], broken) + transport.artifacts[2:])
    with pytest.raises(ConflictSetPrivateProtocolError):
        broken = replace(transport.artifacts[3], content_sha256="f" * 64)
        replace(transport, artifacts=transport.artifacts[:3] + (broken,) + transport.artifacts[4:])


def test_private_transport_rejects_any_prior_reads_or_writes():
    transport = _transport()
    for field in (
            "formal_run_count_before", "private_payload_reads_before",
            "host_learning_writes_before", "label_writes_before",
            "clone_evaluation_writes_before", "teacher_api_llm_calls_before"):
        with pytest.raises(ConflictSetPrivateProtocolError):
            replace(transport, **{field: 1})


def test_private_transport_rejects_non_artifact_inventory_and_uppercase_sha():
    transport = _transport()
    with pytest.raises(ConflictSetPrivateProtocolError):
        replace(transport, artifacts=(object(),) + transport.artifacts[1:])
    with pytest.raises(ConflictSetPrivateProtocolError):
        replace(transport, candidate_manifest_sha256="A" * 64)


def test_private_run_guard_is_one_shot():
    transport = _transport()
    guard = build_conflict_set_run_guard(
        transport, owner_receipt_sha256="5" * 64)
    consumed, intent = consume_conflict_set_run_guard(guard)
    assert consumed.state == "CONSUMED"
    assert intent.consumed_guard_sha256 == consumed.sha256()
    with pytest.raises(ConflictSetPrivateProtocolError):
        consume_conflict_set_run_guard(consumed)


def test_private_transport_parser_rejects_unknown_fields_and_noncanonical_bytes():
    transport = _transport()
    value = transport.to_dict()
    value["labels"] = []
    with pytest.raises(ConflictSetPrivateProtocolError):
        type(transport).from_dict(value)
    payload = b"{}\n\n"
    with pytest.raises(ConflictSetPrivateProtocolError):
        parse_conflict_set_private_transport_bytes(payload)
