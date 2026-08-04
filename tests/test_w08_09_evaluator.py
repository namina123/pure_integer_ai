"""W08-09 private evaluator 的真实逐 case inference 与 fail-closed 协议。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    StableRecordKey,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_evaluator import (
    W08EvaluatorSnapshot,
    W08PrivateEvaluationPair,
    assess_w08_orthogonal_ablations,
    assess_w08_private_lc16,
    assess_w08_private_open_generation,
    evaluate_w08_private_pairs,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    public_safe_w08_aggregate,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_family import (
    build_w08_private_family_documents,
    consume_w08_private_first_run_guard,
    publish_w08_private_family,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_runtime import (
    W08PrivateEvaluatorRuntimeConfig,
    _ReadAudit,
    _candidate_inference_available,
    _validate_public_safe_aggregate,
    run_w08_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_inference import (
    W08CandidateInferenceAdapter,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
)
from pure_integer_ai.experiments.ph2_w08_inference_training import (
    compile_w08_candidate_inference_state,
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


@pytest.fixture(scope="module")
def evaluator_fixture():
    contract = open_w08_frozen_contract(ROOT)
    payload = W08PayloadFirewall.open(
        ROOT, contract, make_w08_request(contract)
    ).read_training_payload()
    state = compile_w08_candidate_inference_state(payload)
    evidence_by_observation = {
        item.observation_key: item for item in payload.teacher_evidence
    }
    pairs = []
    for ordinal, original in enumerate(payload.observations, start=1):
        observation = replace(original, split="held_out")
        evidence = evidence_by_observation[original.stable_key]
        typed = evidence.typed_evidence.to_value()
        if observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1":
            expected_state = "TRUE"
            expected_payload = {
                "definitive_truth_authoritative": typed[
                    "definitive_truth_authoritative"
                ],
                "raw_observation_sha256": typed["raw_observation_sha256"],
                "source_binding_required": 1,
            }
        else:
            expected_state = typed.get("expected_state", "TRUE")
            expected_payload = typed.get("expected_payload", typed)
        label = EvaluatorLabelRecord(
            1,
            1,
            1,
            observation.dataset_key,
            _key(809, 1, ordinal),
            _key(809, 2, ordinal),
            observation.stable_key,
            _key(809, 3, ordinal),
            expected_state,
            CanonicalJsonObject.from_value(expected_payload),
            160,
            1,
            "W-08",
            _key(809, 4),
        )
        pairs.append(W08PrivateEvaluationPair(
            f"PUBLIC-SYNTHETIC-{ordinal:03d}", observation, label
        ))
    snapshot = W08EvaluatorSnapshot(
        tuple((8, ordinal) for ordinal in range(1, 6)),
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
        state.state_key,
        state.sha256(),
        state.interface_version,
        len(state.rules),
        0,
        0,
        0,
        0,
    )
    adapter = W08CandidateInferenceAdapter(state)
    pair_tuple = tuple(pairs)
    baseline = tuple(
        adapter.infer(pair.observation, dimension_key=dimension)
        for pair in pair_tuple
        for dimension in W08_DIMENSION_KEYS
    )
    ablation_families = tuple(
        (
            ablation,
            tuple(
                adapter.infer(
                    pair.observation,
                    dimension_key=dimension,
                    disabled_components=(W08_DIMENSION_KEYS[ordinal],),
                )
                for pair in pair_tuple
                for dimension in W08_DIMENSION_KEYS
            ),
        )
        for ordinal, ablation in enumerate(W08_ABLATION_KEYS)
    )
    return state, snapshot, pair_tuple, baseline, ablation_families


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


def test_private_read_audit_keeps_observation_and_label_accounts_separate():
    audit = _ReadAudit({})
    audit.record("observation-pack", 17, 2, "observation")
    audit.record("label-pack", 19, 2, "evaluator")
    assert audit.reads_by_path == {
        "label-pack": (1, 19),
        "observation-pack": (1, 17),
    }
    assert audit.payload_gets == 2
    assert audit.payload_bytes == 36
    assert audit.observation_records == audit.label_records == 2


def test_five_bearings_pass_only_actual_adapter_outcomes(evaluator_fixture):
    _, snapshot, pairs, outcomes, _ = evaluator_fixture
    results = evaluate_w08_private_pairs(
        snapshot, pairs, case_outcomes=outcomes
    )
    assert tuple(item.status for item in results) == ("PASS",) * 5
    assert all(item.passed_count == len(pairs) for item in results)
    assert len(outcomes) == len(pairs) * len(W08_DIMENSION_KEYS)
    assert len({item.invocation_key for item in outcomes}) == len(outcomes)


def test_non_course_matching_uses_typed_semantics_not_label_wording(
    evaluator_fixture,
):
    _, snapshot, pairs, outcomes, _ = evaluator_fixture
    revision_index = next(
        index
        for index, pair in enumerate(pairs)
        if pair.observation.payload_kind == "DiscourseRevisionQuery"
    )
    revision = pairs[revision_index]
    expected = revision.label.expected_payload.to_value()
    expected["decision"] = "independent-evaluator-wording"
    relabeled = replace(
        revision,
        label=replace(
            revision.label,
            expected_payload=CanonicalJsonObject.from_value(expected),
        ),
    )
    revised_pairs = (
        *pairs[:revision_index],
        relabeled,
        *pairs[revision_index + 1:],
    )
    assert all(
        item.status == "PASS"
        for item in evaluate_w08_private_pairs(
            snapshot,
            revised_pairs,
            case_outcomes=outcomes,
        )
    )

    expected["result_bits"] = [
        1 - expected["result_bits"][0],
        expected["result_bits"][1],
    ]
    wrong = replace(
        relabeled,
        label=replace(
            relabeled.label,
            expected_payload=CanonicalJsonObject.from_value(expected),
        ),
    )
    wrong_pairs = (
        *pairs[:revision_index],
        wrong,
        *pairs[revision_index + 1:],
    )
    assert all(
        item.status == "FAIL"
        for item in evaluate_w08_private_pairs(
            snapshot,
            wrong_pairs,
            case_outcomes=outcomes,
        )
    )

    source = next(
        pair
        for pair in pairs
        if pair.observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1"
    )
    source_outcome = next(
        item
        for item in outcomes
        if item.observation_key == source.observation.stable_key.components
    )
    assert source_outcome.actual_payload.to_value()["result"] == (
        source.label.expected_payload.to_value()
    )


def test_five_real_ablations_disable_only_the_target(evaluator_fixture):
    _, snapshot, pairs, baseline, outcome_families = evaluator_fixture
    results = assess_w08_orthogonal_ablations(
        snapshot,
        pairs,
        outcome_families=outcome_families,
    )
    assert len(results) == len(W08_DIMENSION_KEYS)
    baseline_keys = {item.invocation_key for item in baseline}
    for ordinal, (result, (_, outcomes)) in enumerate(zip(results, outcome_families)):
        target = W08_DIMENSION_KEYS[ordinal]
        assert result["status"] == "PASS"
        assert result["real_component_disabled"] == 1
        assert tuple(result["dimension_statuses"]) == tuple(
            "FAIL" if dimension == target else "PASS"
            for dimension in W08_DIMENSION_KEYS
        )
        assert all(
            item.component_state
            == ("DISABLED" if item.dimension_key == target else "ACTIVE")
            for item in outcomes
        )
        assert baseline_keys.isdisjoint(item.invocation_key for item in outcomes)


def test_open_generation_and_lc16_consume_actual_outputs(evaluator_fixture):
    _, snapshot, pairs, outcomes, _ = evaluator_fixture
    open_generation = assess_w08_private_open_generation(
        snapshot, pairs, case_outcomes=outcomes
    )
    lc16 = assess_w08_private_lc16(
        snapshot, pairs, case_outcomes=outcomes
    )
    assert open_generation["status"] == "PASS"
    assert open_generation["output_invocation_count"] == len(outcomes)
    assert open_generation["exact_surface_read_count"] == 0
    assert open_generation["complete_template_replay_count"] == 0
    assert lc16["status"] == "PASS"
    assert lc16["bearing_cell_count"] == 27
    assert lc16["output_invocation_count"] == len(outcomes)

    wrong_source = replace(outcomes[0], source_key=(809, 99))
    wrong_outputs = (wrong_source, *outcomes[1:])
    assert evaluate_w08_private_pairs(
        snapshot, pairs, case_outcomes=wrong_outputs
    )[0].status == "FAIL"
    assert assess_w08_private_open_generation(
        snapshot, pairs, case_outcomes=wrong_outputs
    )["status"] == "FAIL"

    wrong_consumers = replace(
        outcomes[0],
        consumer_states=tuple(
            (consumer, "FAIL_CLOSED" if consumer == "REASONING" else state)
            for consumer, state in outcomes[0].consumer_states
        ),
    )
    assert assess_w08_private_lc16(
        snapshot,
        pairs,
        case_outcomes=(wrong_consumers, *outcomes[1:]),
    )["status"] == "FAIL"


def test_missing_outcomes_are_ne_and_safe_aggregate_contains_no_labels(
    evaluator_fixture,
):
    _, snapshot, pairs, outcomes, outcome_families = evaluator_fixture
    results = evaluate_w08_private_pairs(snapshot, pairs)
    assert tuple(item.status for item in results) == ("NE",) * 5
    assert all(item.ne_count == len(pairs) for item in results)
    assert assess_w08_private_open_generation(snapshot, pairs)["status"] == "NE"
    assert assess_w08_private_lc16(snapshot, pairs)["status"] == "NE"
    partial = outcomes[:-1]
    assert assess_w08_private_open_generation(
        snapshot, pairs, case_outcomes=partial
    )["status"] == "NE"
    assert assess_w08_private_lc16(
        snapshot, pairs, case_outcomes=partial
    )["status"] == "NE"
    partial_ablation = tuple(
        (key, values[:-1]) for key, values in outcome_families
    )
    assert all(
        item["status"] == "NE"
        for item in assess_w08_orthogonal_ablations(
            snapshot, pairs, outcome_families=partial_ablation
        )
    )
    aggregate = public_safe_w08_aggregate(
        results,
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
    encoded = canonical_json_bytes(aggregate)
    assert aggregate["status"] == "NE"
    assert aggregate["fail_count"] == 0
    assert all(token not in encoded for token in (
        b"expected_payload", b"expected_state", b"typed_payload", b"surface"
    ))


def test_report_safety_allows_surface_counters_but_rejects_private_fields(
    evaluator_fixture,
):
    _, snapshot, pairs, outcomes, outcome_families = evaluator_fixture
    aggregate = public_safe_w08_aggregate(
        evaluate_w08_private_pairs(snapshot, pairs, case_outcomes=outcomes),
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
        ablation_results=assess_w08_orthogonal_ablations(
            snapshot,
            pairs,
            outcome_families=outcome_families,
        ),
        open_generation=assess_w08_private_open_generation(
            snapshot,
            pairs,
            case_outcomes=outcomes,
        ),
        lc16=assess_w08_private_lc16(
            snapshot,
            pairs,
            case_outcomes=outcomes,
        ),
    )
    assert aggregate["open_generation"]["exact_surface_read_count"] == 0
    _validate_public_safe_aggregate(aggregate)

    aggregate["open_generation"]["surface"] = "forbidden"
    with pytest.raises(RuntimeError, match="泄漏 private 字段"):
        _validate_public_safe_aggregate(aggregate)


def test_candidate_inference_preflight_requires_complete_v2_interface(
    evaluator_fixture,
):
    state, _, _, _, _ = evaluator_fixture
    interface = {
        "component_keys": list(W08_DIMENSION_KEYS),
        "evaluator_label_inputs": 0,
        "executable": 1,
        "per_case_invocation_required": 1,
        "rule_count": len(state.rules),
        "state_commitment": state.sha256(),
        "state_key": list(state.state_key),
        "version": W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
    }
    assert _candidate_inference_available({"private_inference_interface": interface})
    for key, bad_value in (
        ("version", "PH2-W08-PRIVATE-INFERENCE-V1"),
        ("executable", 0),
        ("evaluator_label_inputs", 1),
        ("per_case_invocation_required", 0),
        ("state_commitment", "not-a-sha"),
    ):
        corrupted = {**interface, key: bad_value}
        assert not _candidate_inference_available({
            "private_inference_interface": corrupted
        })


def test_fault_before_candidate_verification_seals_ne_without_private_reads(
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
    assert aggregate["infrastructure"]["fault_ne_protocol"] == 1
    assert result.recommendation_path is None
    with pytest.raises(RuntimeError, match="不可重跑|aggregate 已存在"):
        run_w08_private_evaluation_once(
            W08PrivateEvaluatorRuntimeConfig(
                ROOT,
                candidate,
                family,
                family / "execution",
            ),
            family_freeze_sha256=freeze_sha,
        )
