"""W09-06 append-only rollback、局部失效与恢复等价专项。"""
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_rollback import (
    W09RollbackError,
    W09RollbackLedger,
)


ROOT = Path(__file__).parents[1]


def _key(label: str) -> tuple[int, ...]:
    return digest_value(("w09-rollback", label))


def _baseline() -> tuple[W09RollbackLedger, dict[str, tuple[int, ...]]]:
    ledger = W09RollbackLedger(_key("core"))
    keys = {
        label: _key(label)
        for label in (
            "source",
            "observation",
            "observation-unaffected",
            "evidence",
            "use",
        )
    }
    ledger.append("OBSERVATION", keys["source"], "LANGUAGE")
    ledger.append(
        "OBSERVATION",
        keys["observation"],
        "LANGUAGE",
        depends_on=(keys["source"],),
    )
    ledger.append(
        "OBSERVATION",
        keys["observation-unaffected"],
        "LANGUAGE",
    )
    ledger.append(
        "EVIDENCE",
        keys["evidence"],
        "LANGUAGE",
        depends_on=(keys["observation"],),
    )
    ledger.append(
        "USE_OUTCOME",
        keys["use"],
        "LANGUAGE",
        depends_on=(keys["evidence"],),
    )
    return ledger, keys


def test_source_withdrawal_invalidates_only_dependency_closure():
    ledger, keys = _baseline()
    before_core = ledger.core_identity
    ledger.retract_source(keys["source"], _key("source-retract"), "LANGUAGE")
    evaluation = ledger.evaluate()
    assert set(evaluation.invalidated_keys) == {
        keys["source"],
        keys["observation"],
        keys["evidence"],
        keys["use"],
    }
    assert set(evaluation.preserved_keys) == {
        keys["observation-unaffected"],
        _key("source-retract"),
    }
    assert evaluation.core_identity == before_core
    assert evaluation.host_write_count == 0
    assert len(ledger.events) == 6


def test_parser_revision_scope_contraction_and_use_retract_are_append_only():
    ledger, keys = _baseline()
    ledger.revise_parser(
        keys["source"],
        _key("source-revision"),
        (keys["observation"],),
        "LANGUAGE",
    )
    ledger.retract_use(keys["use"], _key("use-retract"), "LANGUAGE")
    ledger.contract_scope(
        "LANGUAGE_NARROWED",
        _key("scope-contraction"),
        (keys["observation-unaffected"],),
    )
    invalidated = set(ledger.evaluate().invalidated_keys)
    assert keys["source"] in invalidated
    assert keys["observation"] in invalidated
    assert keys["evidence"] in invalidated
    assert keys["use"] in invalidated
    assert keys["observation-unaffected"] in invalidated
    assert len(ledger.events) == 8
    assert tuple(item.ordinal for item in ledger.events) == tuple(range(1, 9))


def test_transaction_rollback_and_fresh_resume_clone_are_equivalent():
    ledger, keys = _baseline()
    before = ledger.state_key()
    transaction = ledger.begin()
    pending_key = _key("pending")
    transaction.append(
        ledger.events[-1].__class__(
            len(ledger.events) + 1,
            "USE_OUTCOME",
            pending_key,
            "LANGUAGE",
            (keys["use"],),
            (),
        )
    )
    preview = transaction.preview_state_key()
    receipt = transaction.rollback()
    assert preview != before
    assert ledger.state_key() == before
    assert receipt.leaked_write_count == 0

    report = ledger.report(receipt)
    assert report.status == "PUBLIC_BOUNDED_PASS"
    assert report.fresh_state_key == report.resume_state_key == report.clone_state_key
    clone = ledger.clone_for_evaluation()
    clone.append("OBSERVATION", _key("clone-only"), "CLONE")
    assert clone.state_key() != ledger.state_key()
    assert report.evaluation.core_identity == ledger.core_identity


def test_unknown_dependency_event_replay_and_ablation_fail_closed():
    ledger, keys = _baseline()
    with pytest.raises(W09RollbackError):
        ledger.append(
            "EVIDENCE",
            _key("orphan"),
            "LANGUAGE",
            depends_on=(_key("missing"),),
        )
    with pytest.raises(W09RollbackError):
        ledger.append("OBSERVATION", keys["source"], "LANGUAGE")
    with pytest.raises(W09RollbackError):
        ledger.retract_source(_key("missing"), _key("bad-retract"), "LANGUAGE")
    ablation = ledger.ablate_dependency_invalidation()
    assert ablation.target_dimension_key == "W-09-ROLLBACK"
    assert ablation.target_status == "FAIL"
    assert ablation.unrelated_dimension_failure_count == 0
