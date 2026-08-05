"""W09-04 216-cell dimensional/J-LC public bounded 专项。"""
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import SCOPE_KEYS
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_cumulative import (
    open_w09_cumulative_runtime,
)
from pure_integer_ai.experiments.ph2_w09_dimensional import (
    W09ContinualLearningEvidence,
    W09DimensionalError,
    open_w09_dimensional_runtime,
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


def _key(label: object) -> tuple[int, ...]:
    return tuple(bytes.fromhex(w09_commitment(label)))


def _directional(
        label: str,
        consumer_key: str,
        status: W09ResultState = W09ResultState.PASS,
        ) -> W09DirectionalResult:
    request_key = _key((label, "request"))
    choice_key = _key((label, "choice"))
    candidate_key = _key((label, "candidate"))
    use_key = _key((label, "use"))
    outcome_key = _key((label, "outcome"))
    failure_kind = "NONE" if status is W09ResultState.PASS else "BOUNDED_CELL_FAILURE"
    return W09DirectionalResult(
        W09ConsumerRequest(
            consumer_key,
            request_key,
            w09_commitment((label, "input")),
        ),
        W09ConsumerChoice(
            consumer_key,
            request_key,
            choice_key,
            candidate_key,
        ),
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
            _key((label, "verifier")),
            status,
            failure_kind,
        ),
    )


@pytest.fixture(scope="module")
def context_payload():
    context = open_w09_frozen_contract(ROOT)
    payload = W09PayloadFirewall.open(
        ROOT,
        context,
        make_w09_request(context),
    ).read_training_payload()
    return context, payload


def _cumulative(context_payload, *, complete: bool = True):
    context, payload = context_payload
    cumulative = open_w09_cumulative_runtime(ROOT, context)
    cumulative.ingest_training_payload(payload)
    for carrier_key in W09_CARRIER_KEYS:
        for consumer_key in W09_CONSUMER_KEYS:
            if (
                not complete
                and carrier_key == W09_CARRIER_KEYS[-1]
                and consumer_key == W09_CONSUMER_KEYS[-1]
            ):
                continue
            label = f"cumulative:{carrier_key}:{consumer_key}"
            cumulative.consume_directional(
                carrier_key,
                consumer_key,
                _directional(label, consumer_key),
            )
    return cumulative


def _runtime(context_payload):
    cumulative = _cumulative(context_payload)
    return open_w09_dimensional_runtime(ROOT, cumulative)


def _continual(label: str, result: W09DirectionalResult):
    return W09ContinualLearningEvidence(
        _key((label, "before")),
        _key((label, "after")),
        _key((label, "source-evidence")),
        result.use_outcome.use_key,
        result.use_outcome.outcome_key,
        0,
        0,
    )


def _fill(
        runtime,
        *,
        skip: tuple[str, str, str] | None = None,
        override: tuple[tuple[str, str, str], W09ResultState] | None = None,
        reverse: bool = False,
        ):
    keys = [
        (scope_key, carrier_key, consumer_key)
        for scope_key in SCOPE_KEYS
        for carrier_key in W09_CARRIER_KEYS
        for consumer_key in W09_CONSUMER_KEYS
    ]
    if reverse:
        keys.reverse()
    for key in keys:
        if key == skip:
            continue
        scope_key, carrier_key, consumer_key = key
        label = ":".join(key)
        status = override[1] if override is not None and key == override[0] else W09ResultState.PASS
        result = _directional(label, consumer_key, status)
        runtime.record_cell(
            scope_key,
            carrier_key,
            consumer_key,
            result,
            continual_learning=(
                _continual(label, result)
                if scope_key == "RETENTION_CONTINUAL_LEARNING"
                else None
            ),
        )
    return runtime


def test_all_216_cells_are_current_bounded_and_not_formal(context_payload):
    report = _fill(_runtime(context_payload), reverse=True).report()
    assert len(report.task_audits) == 16
    assert report.task_audits[0].prior_state == "HISTORICAL_SCOPE_ONLY"
    assert report.task_audits[-1].prior_state == "AUDITED_ABSENT"
    assert len(report.parent_receipts) == 7
    assert len(report.cells) == 216
    assert report.retention_cell_count == 189
    assert report.continual_learning_cell_count == 27
    assert report.dimensional_status == "PUBLIC_BOUNDED_PASS"
    assert report.j_lc_w09_state == "PUBLIC_BOUNDED_NOT_FORMAL"
    assert report.formal_evidenced == 0
    assert report.language_capability_mastered == 0
    assert report.language_readiness == 0
    assert report.wall_states == (
        ("W-09-W1_PHYSICAL_GROUNDING", "NE_WALL"),
        ("W-09-W2_DEFINITIVE_TRUTH", "NE_WALL"),
    )
    assert report.stable_key()


def test_missing_cell_misdirection_and_shared_identity_fail_closed(context_payload):
    missing = (SCOPE_KEYS[-1], W09_CARRIER_KEYS[-1], W09_CONSUMER_KEYS[-1])
    with pytest.raises(W09DimensionalError):
        _fill(_runtime(context_payload), skip=missing).report()

    runtime = _runtime(context_payload)
    wrong = _directional("wrong-direction", "REASONING")
    with pytest.raises(W09DimensionalError):
        runtime.record_cell(
            "BOUNDARY_OOV",
            "HTML",
            "UNDERSTANDING",
            wrong,
        )

    runtime = _runtime(context_payload)
    shared = _directional("shared-result", "UNDERSTANDING")
    runtime.record_cell(
        "BOUNDARY_OOV",
        "HTML",
        "UNDERSTANDING",
        shared,
    )
    with pytest.raises(W09DimensionalError):
        runtime.record_cell(
            "BOUNDARY_OOV",
            "MARKDOWN",
            "UNDERSTANDING",
            shared,
        )

    runtime = _runtime(context_payload)
    crossed = _directional("crossed-component", "UNDERSTANDING")
    crossed = replace(
        crossed,
        choice=replace(
            crossed.choice,
            choice_key=crossed.request.request_key,
        ),
        use_outcome=replace(
            crossed.use_outcome,
            choice_key=crossed.request.request_key,
        ),
    )
    with pytest.raises(W09DimensionalError):
        runtime.record_cell(
            "BOUNDARY_OOV",
            "HTML",
            "UNDERSTANDING",
            crossed,
        )

    with pytest.raises(W09DimensionalError):
        open_w09_dimensional_runtime(
            ROOT,
            _cumulative(context_payload, complete=False),
        )
    broken_registry = _cumulative(context_payload)
    broken_registry.registry = object()
    with pytest.raises(W09DimensionalError):
        open_w09_dimensional_runtime(ROOT, broken_registry)


def test_fail_ne_and_ablations_cannot_be_hidden_by_an_average(context_payload):
    target = ("ROLE_PROPOSITION_SCOPE", "SOURCE_CODE", "REASONING")
    failed = _fill(
        _runtime(context_payload),
        override=(target, W09ResultState.FAIL),
    ).report()
    assert failed.dimensional_status == "PUBLIC_BOUNDED_FAIL"
    assert failed.j_lc_w09_state == "PUBLIC_BOUNDED_FAIL"

    ne_report = _fill(
        _runtime(context_payload),
        override=(target, W09ResultState.NE),
    ).report()
    assert ne_report.dimensional_status == "PUBLIC_BOUNDED_NE"
    assert ne_report.j_lc_w09_state == "PUBLIC_BOUNDED_NE"

    passed = _fill(_runtime(context_payload)).report()
    aggregator = _runtime(context_payload).ablate_aggregator(passed)
    cell = _runtime(context_payload).ablate_cell(passed, *target)
    assert (
        aggregator.affected_cell_count,
        aggregator.preserved_cell_count,
        aggregator.unrelated_failure_count,
    ) == (0, 216, 0)
    assert (
        cell.affected_cell_count,
        cell.preserved_cell_count,
        cell.unrelated_failure_count,
    ) == (1, 215, 0)
