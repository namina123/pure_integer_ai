"""GG-03 owner-independent public stress inventory 快速合同测试。"""
from pathlib import Path

from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family import (
    double_scan_generation_generalization_observation_inventory,
)
from pure_integer_ai.experiments.ph2_generation_generalization_public_stress import (
    PUBLIC_STRESS_BUDGET,
    PUBLIC_STRESS_CASE_IDS,
    build_generation_generalization_public_stress_observations,
    generation_generalization_public_stress_bytes,
)


_SOURCE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_INVENTORY = Path(
    "data/ph2/generation_generalization_public_stress_v1.jsonl")


def test_public_stress_inventory_is_reproducible_balanced_and_label_free(
        tmp_path: Path) -> None:
    """公开 builder、committed transport、六路分母和压力特征必须全等。"""
    observations = (
        build_generation_generalization_public_stress_observations(_SOURCE))
    payload = generation_generalization_public_stress_bytes(observations)
    assert payload == _INVENTORY.read_bytes()
    assert {item.episode_id for item in observations} == set(
        PUBLIC_STRESS_CASE_IDS)
    assert len(observations) == 12
    assert all(item.resource_budget == PUBLIC_STRESS_BUDGET
               for item in observations)
    assert b'"surfaces"' not in payload
    assert b'"accepted"' not in payload
    assert b'"rejected"' not in payload
    assert b'"expected_violations"' not in payload

    target = tmp_path / "public-stress.jsonl"
    target.write_bytes(payload)
    inventory = double_scan_generation_generalization_observation_inventory(
        target, resource_ceiling=PUBLIC_STRESS_BUDGET)
    assert inventory.record_count == 12
    assert {count for _requirement, count in inventory.requirement_counts} == {3}
    assert len({item.stable_key_sha256 for item in inventory.records}) == 12
    assert len({item.content_sha256 for item in inventory.records}) == 12

    by_id = {item.episode_id: item for item in observations}
    near_context = by_id["gg03-public-stress-clarify-near-context-v1"]
    assert len(near_context.question.context_surface.encode("utf-8")) >= 480
    assert any(len(item.dialogue.turns) == 3 for item in observations)
    assert by_id[
        "gg03-public-stress-answer-no-forbidden-v1"
    ].question.answer_plan.forbidden_claim_ids == ()
    assert any(
        "%" in evidence.claim_text
        for item in observations
        for evidence in item.question.evidence)
    long_reference = by_id["gg03-public-stress-reference-long-v1"]
    assert sum(len(item.claim_text)
               for item in long_reference.question.evidence) >= 140
