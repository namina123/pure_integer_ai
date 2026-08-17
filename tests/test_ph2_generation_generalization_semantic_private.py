"""V2 私有语义传输及 guard 绑定读取测试。"""
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    build_available_guard_for_identity,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.runtime import (
    consume_evaluation_guard_once,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationEvaluationFamilyError,
    generation_generalization_sha256_bytes,
    read_generation_generalization_private_label_owner_receipt,
    scan_generation_generalization_observation_inventory,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    build_generation_generalization_semantic_label_record,
    generation_generalization_semantic_verdict_contract_sha256,
)
from pure_integer_ai.experiments import (
    ph2_generation_generalization_semantic_private as semantic_private,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def _inventory(tmp_path: Path):
    """构造覆盖六项 requirement 的最小四路径公开 inventory。"""
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
                episode_id=f"semantic-private-{index}-{item.episode_id}",
                split="held_out",
            ),
            budget,
        )
        for index, item in enumerate(episodes, start=1)
    ), key=lambda item: item.stable_key()))
    path = tmp_path / "formal-observations.jsonl"
    path.write_bytes(b"".join(
        canonical_json_line(item.to_dict()) for item in observations))
    return (
        observations,
        scan_generation_generalization_observation_inventory(
            path, resource_ceiling=budget),
    )


def test_semantic_private_transport_is_immutable_and_guard_bound(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observations, inventory = _inventory(tmp_path)
    records = tuple(
        build_generation_generalization_semantic_label_record(item)
        for item in observations)
    private_root = tmp_path / "private-label-owner"
    private_root.mkdir()
    monkeypatch.setattr(
        semantic_private, "_private_root",
        lambda value: Path(value).resolve(),
    )
    contract = generation_generalization_semantic_verdict_contract_sha256()
    publication = semantic_private.publish_generation_generalization_semantic_labels(
        records,
        private_label_root=private_root,
        label_relative_path="labels/formal-labels.jsonl.gz",
        observation_inventory=inventory,
        verdict_contract_sha256=contract,
    )
    owner, owner_sha = read_generation_generalization_private_label_owner_receipt(
        private_root / "owner-receipt.json")
    assert owner_sha == publication["owner_receipt_sha256"]
    assert owner.verdict_contract_sha256 == contract
    with pytest.raises(
            GenerationGeneralizationEvaluationFamilyError,
            match="已存在"):
        semantic_private.publish_generation_generalization_semantic_labels(
            records,
            private_label_root=private_root,
            label_relative_path="labels/formal-labels.jsonl.gz",
            observation_inventory=inventory,
            verdict_contract_sha256=contract,
        )

    family = tmp_path / "family"
    family.mkdir()
    guard = build_available_guard_for_identity(
        "1" * 64, "2" * 64, owner_sha, "3" * 64, "4" * 64)
    write_immutable_json(guard.to_dict(), family / "guard.available.json")
    prediction = {"artifact_kind": "PUBLIC_TEST_SEMANTIC_PREDICTION", "version": 1}
    prediction_path = family / "predictions.seal.json"
    write_immutable_json(prediction, prediction_path)
    prediction_sha = generation_generalization_sha256_bytes(
        canonical_json_bytes(prediction))
    with pytest.raises(
            EvaluationKernelContractError,
            match="available guard still exists"):
        semantic_private.read_generation_generalization_semantic_labels_after_guard(
            family_root=family,
            expected_guard=guard,
            prediction_seal_path=prediction_path,
            prediction_seal_sha256=prediction_sha,
            private_label_root=private_root,
            owner_receipt=owner,
            observation_inventory=inventory,
        )
    consume_evaluation_guard_once(family, guard)
    assert semantic_private.read_generation_generalization_semantic_labels_after_guard(
        family_root=family,
        expected_guard=guard,
        prediction_seal_path=prediction_path,
        prediction_seal_sha256=prediction_sha,
        private_label_root=private_root,
        owner_receipt=owner,
        observation_inventory=inventory,
    ) == records
