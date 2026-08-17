"""GG-03 V10 public expansion probes remain label-free and deterministic."""
from pathlib import Path

from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family import (
    double_scan_generation_generalization_observation_inventory,
)
from pure_integer_ai.experiments.ph2_generation_generalization_public_stress_v10 import (
    PUBLIC_V10_STRESS_BUDGET,
    PUBLIC_V10_STRESS_CASE_IDS,
    build_generation_generalization_public_v10_stress_observations,
    generation_generalization_public_v10_stress_bytes,
)


_SOURCE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def test_v10_public_expansion_is_balanced_label_free_and_reproducible(
        tmp_path: Path) -> None:
    """V10 只公开 typed Observation，并覆盖新的组合边界。"""
    observations = (
        build_generation_generalization_public_v10_stress_observations(_SOURCE))
    payload = generation_generalization_public_v10_stress_bytes(observations)
    assert len(observations) == 12
    assert {item.episode_id for item in observations} == set(
        PUBLIC_V10_STRESS_CASE_IDS)
    assert all(item.resource_budget == PUBLIC_V10_STRESS_BUDGET
               for item in observations)
    assert b'"surfaces"' not in payload
    assert b'"accepted"' not in payload
    assert b'"rejected"' not in payload
    assert b'"labels"' not in payload

    target = tmp_path / "public-v10-stress.jsonl"
    target.write_bytes(payload)
    inventory = double_scan_generation_generalization_observation_inventory(
        target, resource_ceiling=PUBLIC_V10_STRESS_BUDGET)
    assert inventory.record_count == 12
    assert len({item.stable_key_sha256 for item in inventory.records}) == 12
    assert len({item.content_sha256 for item in inventory.records}) == 12
    assert {count for _requirement, count in inventory.requirement_counts} == {3}

    by_id = {item.episode_id: item for item in observations}
    assert len(by_id[PUBLIC_V10_STRESS_CASE_IDS[0]].question.evidence) == 2
    assert len(by_id[PUBLIC_V10_STRESS_CASE_IDS[1]].question.evidence) == 3
    assert len(by_id[PUBLIC_V10_STRESS_CASE_IDS[3]].question.evidence) == 3
    assert len(by_id[PUBLIC_V10_STRESS_CASE_IDS[6]].question.evidence) == 3
    assert len(by_id[PUBLIC_V10_STRESS_CASE_IDS[7]].question.evidence) == 3
    assert all(
        len({item.source_id for item in by_id[case].question.evidence}) == 3
        for case in PUBLIC_V10_STRESS_CASE_IDS[6:9]
    )
    assert any(len(item.dialogue.turns) == 3 for item in observations)
    assert any(len(item.question.context_surface.encode("utf-8")) >= 256
               for item in observations)
