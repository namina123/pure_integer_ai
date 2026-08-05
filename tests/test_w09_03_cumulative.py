"""W09-03 cumulative runtime 的父身份、train-only 和 consumer 隔离专项。"""
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_cumulative import (
    W09CumulativeError,
    W09CumulativeRuntime,
    open_w09_cumulative_runtime,
    read_w09_public_parent_receipts,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_types import (
    W09ConsumerChoice,
    W09ConsumerRequest,
    W09DirectionalResult,
    W09ResultState,
    W09UseOutcome,
    W09VerifierResult,
)
from pure_integer_ai.experiments.ph2_w09_weaning import w09_commitment


ROOT = Path(__file__).parents[1]


def _context_and_payload():
    context = open_w09_frozen_contract(ROOT)
    firewall = W09PayloadFirewall.open(ROOT, context, make_w09_request(context))
    return context, firewall.read_training_payload()


def _directional(consumer_key: str) -> W09DirectionalResult:
    def key(label: str) -> tuple[int, ...]:
        return tuple(bytes.fromhex(w09_commitment((label, consumer_key))))

    request_key = key("request")
    choice_key = key("choice")
    candidate_key = key("candidate")
    use_key = key("use")
    outcome_key = key("outcome")
    return W09DirectionalResult(
        W09ConsumerRequest(consumer_key, request_key, w09_commitment(("input", consumer_key))),
        W09ConsumerChoice(consumer_key, request_key, choice_key, candidate_key),
        W09UseOutcome(
            consumer_key,
            request_key,
            choice_key,
            candidate_key,
            use_key,
            outcome_key,
            "RESOLVED",
        ),
        W09VerifierResult(
            consumer_key,
            request_key,
            use_key,
            outcome_key,
            key("verifier"),
            W09ResultState.PASS,
            "NONE",
        ),
    )


def test_cumulative_runtime_absorbs_all_train_and_keeps_formal_state_zero():
    context, payload = _context_and_payload()
    runtime = open_w09_cumulative_runtime(ROOT, context)
    delta = runtime.ingest_training_payload(payload)
    report = runtime.report()
    assert (delta.source_ref_count, delta.observation_count, delta.evidence_count) == (535, 309, 309)
    assert delta.new_pack_count == 34
    assert report.formal_evidenced == 0
    assert report.language_capability_mastered == 0
    assert report.language_readiness == 0
    assert report.complete is False


def test_missing_parent_and_incomplete_payload_fail_closed():
    context, payload = _context_and_payload()
    parents = read_w09_public_parent_receipts(ROOT)
    with pytest.raises(W09CumulativeError):
        W09CumulativeRuntime(ROOT, context, parent_receipts=parents[:-1])

    runtime = open_w09_cumulative_runtime(ROOT, context)
    incomplete = replace(payload, training_evidence=payload.training_evidence[:-1])
    with pytest.raises(W09CumulativeError):
        runtime.ingest_training_payload(incomplete)


def test_training_payload_replay_and_consumer_misdirection_fail_closed():
    context, payload = _context_and_payload()
    runtime = open_w09_cumulative_runtime(ROOT, context)
    runtime.ingest_training_payload(payload)
    with pytest.raises(W09CumulativeError):
        runtime.ingest_training_payload(payload)

    with pytest.raises(W09CumulativeError):
        runtime.consume_directional("HTML", "UNDERSTANDING", _directional("REASONING"))
    runtime.consume_directional("HTML", "UNDERSTANDING", _directional("UNDERSTANDING"))
    with pytest.raises(W09CumulativeError):
        runtime.consume_directional("HTML", "UNDERSTANDING", _directional("UNDERSTANDING"))


def test_all_carriers_share_one_engine_and_keep_a_canonical_state_key():
    context, payload = _context_and_payload()
    left = open_w09_cumulative_runtime(ROOT, context)
    right = open_w09_cumulative_runtime(ROOT, context)
    left.ingest_training_payload(payload)
    right.ingest_training_payload(payload)
    for carrier_key in W09_CARRIER_KEYS:
        for consumer_key in W09_CONSUMER_KEYS:
            left.consume_directional(carrier_key, consumer_key, _directional(consumer_key))
            right.consume_directional(carrier_key, consumer_key, _directional(consumer_key))
    report = left.report()
    assert report.complete is True
    assert report.shared_engine_count == 1
    assert len(report.consumer_cells) == 27
    assert left.state_key() == right.state_key()
