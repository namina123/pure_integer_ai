from dataclasses import replace

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetEvidence,
    build_conflict_set_plan,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_protocol import (
    CONFLICT_SET_FORMAL_REQUIREMENTS,
    ConflictSetFormalProtocolError,
    build_conflict_set_semantic_formal_aggregate,
    build_conflict_set_semantic_label_record,
    build_conflict_set_semantic_prediction_record,
    build_conflict_set_semantic_prediction_seal,
    conflict_set_projection_dimension_sha256,
    conflict_set_semantic_verdict_contract_sha256,
)


def _plan():
    return build_conflict_set_plan(
        scope_id=901,
        claim_ids=("claim-b", "claim-a"),
        evidence=(
            ConflictSetEvidence("e1", "claim-b", "source-c", 901, 1, 0),
            ConflictSetEvidence("e2", "claim-b", "source-d", 901, 0, 1),
            ConflictSetEvidence("e3", "claim-a", "source-a", 901, 1, 0),
            ConflictSetEvidence("e4", "claim-a", "source-b", 901, 0, 1),
        ),
    )


def _prediction(projection):
    return build_conflict_set_semantic_prediction_seal(
        (
            build_conflict_set_semantic_prediction_record(
                "a" * 64, "c" * 64,
                "NE" if projection is None else "PASS", projection),
            build_conflict_set_semantic_prediction_record(
                "b" * 64, "d" * 64,
                "NE" if projection is None else "PASS", projection),
        ),
        family_manifest_sha256="1" * 64,
        family_commitment_sha256="2" * 64,
        candidate_manifest_sha256="3" * 64,
    )


def _labels(projection):
    return tuple(
        build_conflict_set_semantic_label_record(key, projection)
        for key in ("a" * 64, "b" * 64)
    )


def test_projection_commits_source_attribution_per_claim():
    plan = _plan()
    assert plan.projection.claim_source_ids == (
        ("claim-b", ("source-c", "source-d")),
        ("claim-a", ("source-a", "source-b")),
    )
    label = build_conflict_set_semantic_label_record("a" * 64, plan.projection)
    assert label.to_dict()["expected_dimensions"]
    assert conflict_set_semantic_verdict_contract_sha256() == (
        conflict_set_semantic_verdict_contract_sha256())


def test_four_dimensions_all_pass_and_round_trip_is_canonical():
    plan = _plan()
    prediction = _prediction(plan.projection)
    aggregate = build_conflict_set_semantic_formal_aggregate(
        prediction,
        _labels(plan.projection),
        label_commitment_sha256="4" * 64,
        label_transport_bytes=128,
    )
    assert aggregate.status == "PASS"
    assert tuple(item.status for item in aggregate.dimensions) == ("PASS",) * 4
    assert canonical_json_line(aggregate.to_dict()).endswith(b"\n")


@pytest.mark.parametrize("field", (
    "claim_ids", "claim_source_ids", "claim_states", "scope_id"))
def test_each_dimension_detects_a_typed_projection_change(field):
    plan = _plan()
    projection = plan.projection
    if field == "claim_ids":
        changed = replace(projection, claim_ids=("claim-a", "claim-b"),
                          claim_states=(
                              ("claim-a", 1, 1), ("claim-b", 1, 1)),
                          claim_source_ids=(
                              ("claim-a", ("source-a", "source-b")),
                              ("claim-b", ("source-c", "source-d")),))
    elif field == "claim_source_ids":
        changed = replace(projection, claim_source_ids=(
            ("claim-b", ("source-a", "source-d")),
            ("claim-a", ("source-b", "source-c")),
        ))
    elif field == "claim_states":
        changed = replace(projection, claim_states=(
            ("claim-b", 1, 0), ("claim-a", 1, 1)))
    else:
        changed = replace(projection, scope_id=902)
    aggregate = build_conflict_set_semantic_formal_aggregate(
        _prediction(changed),
        _labels(projection),
        label_commitment_sha256="4" * 64,
        label_transport_bytes=128,
    )
    statuses = dict((item.requirement, item.status)
                    for item in aggregate.dimensions)
    expected = {
        "claim_ids": "CLAIM_ORDER",
        "claim_source_ids": "CLAIM_SOURCE_CLOSURE",
        "claim_states": "CLAIM_STANCE_CLOSURE",
        "scope_id": "SCOPE_CLOSURE",
    }[field]
    assert statuses[expected] == "FAIL"
    assert sum(status == "FAIL" for status in statuses.values()) >= 1


def test_unavailable_projection_is_ne_for_all_dimensions():
    prediction = _prediction(None)
    plan = _plan()
    aggregate = build_conflict_set_semantic_formal_aggregate(
        prediction,
        _labels(plan.projection),
        label_commitment_sha256="4" * 64,
        label_transport_bytes=128,
    )
    assert aggregate.status == "NE"
    assert tuple(item.status for item in aggregate.dimensions) == ("NE",) * 4


def test_complete_projection_commitment_mismatch_cannot_hide_behind_dimensions():
    plan = _plan()
    record = build_conflict_set_semantic_prediction_record(
        "a" * 64, "c" * 64, "PASS", plan.projection)
    tampered = replace(record, projection_sha256="f" * 64)
    prediction = build_conflict_set_semantic_prediction_seal(
        (tampered,
         build_conflict_set_semantic_prediction_record(
             "b" * 64, "d" * 64, "PASS", plan.projection)),
        family_manifest_sha256="1" * 64,
        family_commitment_sha256="2" * 64,
        candidate_manifest_sha256="3" * 64,
    )
    aggregate = build_conflict_set_semantic_formal_aggregate(
        prediction,
        _labels(plan.projection),
        label_commitment_sha256="4" * 64,
        label_transport_bytes=128,
    )
    assert aggregate.status == "FAIL"
    assert tuple(item.status for item in aggregate.dimensions) == ("FAIL",) * 4


def test_prediction_seal_rejects_unsorted_or_nonzero_prelabel_counters():
    plan = _plan()
    first = build_conflict_set_semantic_prediction_record(
        "a" * 64, "c" * 64, "PASS", plan.projection)
    second = build_conflict_set_semantic_prediction_record(
        "b" * 64, "d" * 64, "PASS", plan.projection)
    with pytest.raises(ConflictSetFormalProtocolError):
        build_conflict_set_semantic_prediction_seal(
            (second, first), family_manifest_sha256="1" * 64,
            family_commitment_sha256="2" * 64,
            candidate_manifest_sha256="3" * 64)
    with pytest.raises(ConflictSetFormalProtocolError):
        type(_prediction(plan.projection))(
            "1" * 64, "2" * 64, "3" * 64, (first, second),
            teacher_call_count=1)


def test_label_round_trip_rejects_unknown_field_and_identity_mismatch():
    plan = _plan()
    label = build_conflict_set_semantic_label_record("a" * 64, plan.projection)
    value = label.to_dict()
    value["unexpected"] = 1
    with pytest.raises(ConflictSetFormalProtocolError):
        type(label).from_dict(value)
    with pytest.raises(ConflictSetFormalProtocolError):
        build_conflict_set_semantic_formal_aggregate(
            _prediction(plan.projection),
            (label, build_conflict_set_semantic_label_record(
                "c" * 64, plan.projection)),
            label_commitment_sha256="4" * 64,
            label_transport_bytes=128,
        )
