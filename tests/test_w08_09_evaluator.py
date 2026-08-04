"""W08-09 private family、五维消融与 fault-NE 协议专项。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_evaluator import (
    W08EvaluatorSnapshot,
    W08PrivateCaseOutcome,
    W08PrivateEvaluationPair,
    assess_w08_orthogonal_ablations,
    assess_w08_private_lc16,
    assess_w08_private_open_generation,
    evaluate_w08_private_pairs,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    evidence_commitment,
    public_safe_w08_aggregate,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_family import (
    build_w08_private_family_documents,
    consume_w08_private_first_run_guard,
    publish_w08_private_family,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_runtime import (
    W08PrivateEvaluatorRuntimeConfig,
    _candidate_inference_available,
    run_w08_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import (
    W08_RUNTIME_HARD_CONJUNCT_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]


def _key(*values: int) -> StableRecordKey:
    return StableRecordKey(values)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def external_tmp_path():
    with tempfile.TemporaryDirectory(
        prefix="w08-private-test-", dir=ROOT.parent
    ) as value:
        yield Path(value)


def _documents():
    return build_w08_private_family_documents(
        ROOT,
        candidate_contract_sha256="1" * 64,
        candidate_guard_sha256="2" * 64,
        candidate_host_sha256="3" * 64,
        candidate_seal_sha256="4" * 64,
        evaluator_public_head_commit_sha1=_head(),
        nonce=(8, 29, 47),
    )


def _pair(index: int) -> W08PrivateEvaluationPair:
    observation = ObservationRecord(
        1,
        1,
        1,
        _key(1),
        _key(2),
        _key(800, index),
        "W-08",
        "TEST",
        "held_out",
        "zh",
        "typed-test",
        _key(3, index),
        "CC0-1.0",
        _key(4, index),
        _key(5, index),
        _key(6, index),
        _key(7, index),
        "evaluator",
        "read_only_probe",
        "TypedPrivateProbeV1",
        CanonicalJsonObject.from_value({"probe_key": [index]}),
        "HELD_OUT_COMBINATION",
        None,
        (),
        index,
    )
    label = EvaluatorLabelRecord(
        1,
        1,
        1,
        _key(1),
        _key(2),
        _key(900, index),
        observation.stable_key,
        _key(10, index),
        "TRUE",
        CanonicalJsonObject.from_value({"accepted": 1}),
        100,
        1,
        "W-08",
        _key(11),
    )
    return W08PrivateEvaluationPair(f"PACK-{index}", observation, label)


def _snapshot() -> W08EvaluatorSnapshot:
    return W08EvaluatorSnapshot(
        tuple((8, index) for index in range(1, 6)),
        tuple(
            (dimension, consumer, "RESOLVED")
            for dimension in W08_DIMENSION_KEYS
            for consumer in W08_CONSUMER_KEYS
        ),
        tuple(
            (key, "PUBLIC_BOUNDED_PASS")
            for key in W08_RUNTIME_HARD_CONJUNCT_KEYS
        ),
        (8, 9, 10),
        "a" * 64,
        0,
        0,
        0,
        0,
    )


def _case_outcomes(
    pairs: tuple[W08PrivateEvaluationPair, ...],
) -> tuple[W08PrivateCaseOutcome, ...]:
    shortcuts = (
        ("exact_surface_reads", 0),
        ("fifo_or_recency_choices", 0),
        ("full_recompute_runs", 0),
        ("preloaded_hot_records", 0),
        ("w09_future_reads", 0),
    )
    return tuple(
        W08PrivateCaseOutcome(
            dimension,
            pair.observation.stable_key.components,
            pair.label.expected_state,
            evidence_commitment(pair.label.expected_payload.to_value()),
            tuple((consumer, "RESOLVED") for consumer in W08_CONSUMER_KEYS),
            shortcuts,
            (80809, dimension_ordinal, pair_ordinal),
        )
        for dimension_ordinal, dimension in enumerate(W08_DIMENSION_KEYS, start=1)
        for pair_ordinal, pair in enumerate(pairs, start=1)
    )


def test_private_family_freezes_metadata_before_any_payload_read(external_tmp_path):
    documents = _documents()
    cases = json.loads(documents.case_bytes)
    labels = json.loads(documents.label_bytes)
    assert cases["formal_run_count"] == labels["formal_run_count"] == 0
    assert cases["bindings"] and labels["bindings"]
    root = external_tmp_path / "family"
    freeze, freeze_sha = publish_w08_private_family(
        root,
        documents,
        forbidden_roots=(ROOT, external_tmp_path / "candidate"),
    )
    freeze_value = json.loads(freeze.read_bytes())
    assert freeze_value["private_payload_reads"] == 0
    assert tuple(freeze_value["dimension_order"]) == W08_DIMENSION_KEYS
    assert tuple(freeze_value["ablation_order"]) == W08_ABLATION_KEYS
    with pytest.raises(RuntimeError, match="root 必须全新"):
        publish_w08_private_family(
            root,
            documents,
            forbidden_roots=(ROOT, external_tmp_path / "candidate"),
        )
    guard, _ = consume_w08_private_first_run_guard(
        root,
        family_freeze_sha256=freeze_sha,
    )
    assert guard.is_file()
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w08_private_first_run_guard(
            root,
            family_freeze_sha256=freeze_sha,
        )


def test_five_bearings_and_ablations_are_orthogonal():
    pairs = tuple(_pair(index) for index in range(1, 4))
    snapshot = _snapshot()
    outcomes = _case_outcomes(pairs)
    baseline = evaluate_w08_private_pairs(
        snapshot,
        pairs,
        case_outcomes=outcomes,
    )
    assert tuple(item.status for item in baseline) == ("PASS",) * 5
    ablations = assess_w08_orthogonal_ablations(
        snapshot,
        pairs,
        case_outcomes=outcomes,
    )
    assert len(ablations) == 5
    for ordinal, item in enumerate(ablations):
        assert item["status"] == "PASS"
        assert tuple(item["dimension_statuses"]) == tuple(
            "FAIL" if index == ordinal else "PASS" for index in range(5)
        )
    open_generation = assess_w08_private_open_generation(
        snapshot,
        pairs,
        layer_states=tuple(
            (key, "PASS")
            for key in (
                "CONTENT",
                "STRUCTURE_LOGIC",
                "DISCOURSE_REFERENCE",
                "SURFACE_MORPHOLOGY",
                "TASK_COMMUNICATIVE",
            )
        ),
    )
    lc16 = assess_w08_private_lc16(
        snapshot,
        bearing_cell_states=("PASS",) * 27,
    )
    assert open_generation["status"] == "PASS"
    assert open_generation["exact_surface_read_count"] == 0
    assert open_generation["complete_template_replay_count"] == 0
    assert lc16["status"] == "PASS"
    assert lc16["bearing_cell_count"] == 27
    assert lc16["wall_dimension_state"] == "NE"


def test_candidate_summary_without_private_case_execution_is_ne():
    pairs = tuple(_pair(index) for index in range(1, 4))
    snapshot = _snapshot()
    baseline = evaluate_w08_private_pairs(snapshot, pairs)
    assert tuple(item.status for item in baseline) == ("NE",) * 5
    assert all(item.ne_count == len(pairs) for item in baseline)
    assert assess_w08_private_open_generation(snapshot, pairs)["status"] == "NE"
    assert assess_w08_private_lc16(snapshot)["status"] == "NE"
    aggregate = public_safe_w08_aggregate(
        baseline,
        family_commitment="1" * 64,
        payload_commitment="2" * 64,
        case_commitment="3" * 64,
        label_commitment="4" * 64,
        cluster_commitment="5" * 64,
        failure_phase="NONE",
        formal_run_count=1,
        write_counts={
            "candidate_writes": 0,
            "label_writes": 0,
            "public_writes": 0,
        },
    )
    assert aggregate["status"] == "NE"
    assert aggregate["fail_count"] == 0
    assert not _candidate_inference_available({})
    assert not _candidate_inference_available({
        "private_inference_interface": {
            "state_commitment": "not-a-sha",
            "version": "PH2-W08-PRIVATE-INFERENCE-V1",
        }
    })
    assert _candidate_inference_available({
        "private_inference_interface": {
            "state_commitment": "a" * 64,
            "version": "PH2-W08-PRIVATE-INFERENCE-V1",
        }
    })

    failed_layers = tuple(
        (key, "FAIL" if index == 0 else "PASS")
        for index, key in enumerate((
            "CONTENT",
            "STRUCTURE_LOGIC",
            "DISCOURSE_REFERENCE",
            "SURFACE_MORPHOLOGY",
            "TASK_COMMUNICATIVE",
        ))
    )
    assert assess_w08_private_open_generation(
        snapshot,
        pairs,
        layer_states=failed_layers,
    )["status"] == "FAIL"
    assert assess_w08_private_lc16(
        snapshot,
        bearing_cell_states=("FAIL",) + ("PASS",) * 26,
    )["status"] == "FAIL"


def test_fault_after_guard_seals_safe_ne_without_reading_private_payload(
    external_tmp_path,
):
    documents = _documents()
    family = external_tmp_path / "family"
    candidate = external_tmp_path / "candidate"
    candidate.mkdir()
    _, freeze_sha = publish_w08_private_family(
        family,
        documents,
        forbidden_roots=(ROOT, candidate),
    )
    result = run_w08_private_evaluation_once(
        W08PrivateEvaluatorRuntimeConfig(
            ROOT,
            candidate,
            family,
            family / "execution",
            fault_phase="CANDIDATE_VERIFY",
        ),
        family_freeze_sha256=freeze_sha,
    )
    aggregate = json.loads(result.aggregate_path.read_bytes())
    assert result.status == "NE"
    assert aggregate["failure_phase"] == "CANDIDATE_VERIFY"
    assert aggregate["dimension_results"] == []
    assert result.recommendation_path is None
    with pytest.raises(RuntimeError, match="不可重跑"):
        run_w08_private_evaluation_once(
            W08PrivateEvaluatorRuntimeConfig(
                ROOT,
                candidate,
                family,
                family / "execution",
            ),
            family_freeze_sha256=freeze_sha,
        )
