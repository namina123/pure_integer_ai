"""Public tests for the R4 V5-first private evaluator revision."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
    build_v2_private_family_registration,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r4 import (
    W02_MORPH_V3_PRIVATE_R4_EVALUATOR_VERSION,
    run_w02_morphology_successor_v3_private_r4_evaluation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4 import (
    read_w02_morphology_successor_v3_private_owner_r4_receipt,
)


def _registration():
    return build_v2_private_family_registration(
        "W-02",
        payload_commitment="1" * 64,
        case_commitment="2" * 64,
        label_commitment="3" * 64,
        cluster_commitment="4" * 64,
        candidate_freeze_sha256="5" * 64,
        code_freeze_sha256="6" * 64,
        resource_budget=V2EvaluatorResourceBudget(
            512, 9_000_000, 536_870_912, 300_000, 100_000, 4),
    )


def test_r4_evaluator_version_is_append_only() -> None:
    assert W02_MORPH_V3_PRIVATE_R4_EVALUATOR_VERSION.endswith("R4-V1")


def test_r4_formal_run_closes_all_source_refs_before_pair_evaluation(
        monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[1]
    _, files = read_w02_morphology_successor_v3_private_owner_r4_receipt(
        repository)
    events: list[str] = []

    def authorize(*args, **kwargs):
        return {}

    def close_sources(*args, **kwargs):
        events.append("source_refs_closed")
        return tuple(object() for _ in range(500))

    def evaluate(*args, **kwargs):
        assert events == ["source_refs_closed"]
        events.append("pair_evaluation_entered")
        return {"input_pair_count": 500, "source_count": 500, "status": "PASS"}

    module = (
        "pure_integer_ai.experiments."
        "ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r4")
    monkeypatch.setattr(
        module + ".authorize_w02_morphology_successor_v3_private_r4_files",
        authorize)
    monkeypatch.setattr(
        module + ".read_and_close_w02_morphology_successor_v3_private_r4_sources",
        close_sources)
    monkeypatch.setattr(
        module + ".evaluate_w02_morphology_successor_v3_private_r4_pair_stream",
        evaluate)
    monkeypatch.setattr(module + ".validate_v2_safe_report", lambda value: None)

    report = run_w02_morphology_successor_v3_private_r4_evaluation(
        object(), object(), _registration(), files,
        "candidate", "v1", "v2", family_freeze_sha256="7" * 64)

    assert events == ["source_refs_closed", "pair_evaluation_entered"]
    assert report["source_ref_closure_before_pair_stream"] == 1
    assert report["source_ref_records_closed"] == 500
    assert report["observation_reads"] == 500
    assert report["label_record_reads"] == 500


def test_r4_source_closure_is_frozen_before_pair_generator_creation() -> None:
    import inspect
    from pure_integer_ai.experiments import (
        ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r4 as module,
    )

    source = inspect.getsource(
        module.run_w02_morphology_successor_v3_private_r4_evaluation)
    assert source.index("read_and_close_w02") < source.index("pairs = (")
    assert "blind_private_source_specs_v5" in inspect.getsource(module)
