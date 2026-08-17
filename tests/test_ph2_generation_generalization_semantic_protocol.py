"""GG-03 V2 语义 prediction、aggregate 与 publication 聚焦测试。"""
from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    generation_generalization_sha256_bytes,
    scan_generation_generalization_observation_inventory,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    GenerationGeneralizationSemanticLabelRecord,
    generation_generalization_semantic_verdict_contract_sha256,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_protocol import (
    GenerationGeneralizationSemanticPredictionRecord,
    GenerationGeneralizationSemanticPredictionSeal,
    build_generation_generalization_semantic_formal_aggregate,
    build_generation_generalization_semantic_publication,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def _sha(value: str) -> str:
    """返回测试用稳定 SHA-256。"""
    return generation_generalization_sha256_bytes(value.encode("utf-8"))


def _inventory(tmp_path: Path):
    """构造覆盖六项 requirement 的四路径公开 inventory。"""
    budget = GenerationGeneralizationEvaluationBudget(512, 4, 4, 96, 16)
    episodes = tuple(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.question.answer_plan.response_act in {
            "ANSWER", "CLARIFY", "CONFLICT"}
    )
    observations = tuple(sorted((
        GenerationGeneralizationEvaluationObservation.from_held_out_episode(
            replace(
                item,
                episode_id=f"semantic-protocol-{index}-{item.episode_id}",
                split="held_out",
            ),
            budget,
        )
        for index, item in enumerate(episodes, start=1)
    ), key=lambda item: item.stable_key()))
    path = tmp_path / "formal-observations.jsonl"
    path.write_bytes(b"".join(
        canonical_json_line(item.to_dict()) for item in observations))
    return scan_generation_generalization_observation_inventory(
        path, resource_ceiling=budget)


def _prediction_and_labels(tmp_path: Path):
    """构造全部内部与语义 verdict 均 PASS 的 V2 输入。"""
    inventory = _inventory(tmp_path)
    records = []
    labels = []
    for ordinal, identity in enumerate(inventory.records, start=1):
        semantic_sha = _sha(f"semantic-{ordinal}")
        records.append(GenerationGeneralizationSemanticPredictionRecord(
            identity.stable_key_sha256,
            semantic_sha,
            _sha(f"run-{ordinal}"),
            tuple((requirement, "PASS")
                  for requirement in identity.requirements),
        ))
        labels.append(GenerationGeneralizationSemanticLabelRecord(
            identity.stable_key_sha256,
            semantic_sha,
            identity.requirements,
        ))
    prediction = GenerationGeneralizationSemanticPredictionSeal(
        _sha("manifest"),
        _sha("family"),
        _sha("candidate"),
        _sha("policy"),
        _sha("batch"),
        generation_generalization_semantic_verdict_contract_sha256(),
        tuple(records),
    )
    return inventory, prediction, tuple(labels)


def test_semantic_aggregate_distinguishes_pass_fail_and_ne(
        tmp_path: Path) -> None:
    """同义 identity PASS、不同 identity FAIL、缺 projection 为 NE。"""
    _inventory_value, prediction, labels = _prediction_and_labels(tmp_path)
    passed = build_generation_generalization_semantic_formal_aggregate(
        prediction,
        labels,
        label_commitment_sha256=_sha("labels"),
        label_transport_bytes=123,
    )
    assert passed.status == "PASS"
    assert all(item.status == "PASS" for item in passed.dimensions)
    decision, receipt, failure = (
        build_generation_generalization_semantic_publication(passed))
    assert decision["status"] == "PASS"
    assert receipt is not None
    assert failure is None

    failed_records = (
        replace(
            prediction.records[0],
            semantic_projection_sha256=_sha("different-semantic"),
        ),
        *prediction.records[1:],
    )
    failed_prediction = replace(prediction, records=failed_records)
    failed = build_generation_generalization_semantic_formal_aggregate(
        failed_prediction,
        labels,
        label_commitment_sha256=_sha("labels"),
        label_transport_bytes=123,
    )
    assert failed.status == "FAIL"
    assert any(item.failed_count > 0 for item in failed.dimensions)
    _decision, receipt, failure = (
        build_generation_generalization_semantic_publication(failed))
    assert receipt is None
    assert failure is not None and failure["status"] == "FAIL"

    ne_records = (
        replace(prediction.records[0], semantic_projection_sha256=None),
        *prediction.records[1:],
    )
    ne_prediction = replace(prediction, records=ne_records)
    unavailable = build_generation_generalization_semantic_formal_aggregate(
        ne_prediction,
        labels,
        label_commitment_sha256=_sha("labels"),
        label_transport_bytes=123,
    )
    assert unavailable.status == "NE"
    assert any(item.ne_count > 0 for item in unavailable.dimensions)
    assert unavailable.to_dict()["verdict_contract_sha256"] == (
        generation_generalization_semantic_verdict_contract_sha256())
    assert generation_generalization_sha256_bytes(canonical_json_bytes(
        unavailable.to_dict())) == unavailable.sha256()
