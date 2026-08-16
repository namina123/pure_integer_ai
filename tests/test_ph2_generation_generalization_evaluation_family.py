"""GG-03 E-05E family freeze、双遍 inventory 与共享 guard 聚焦专项。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    build_available_guard_for_identity,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family import (
    PRIVATE_OWNER_ARTIFACT_KIND,
    GenerationGeneralizationCodeFileIdentity,
    GenerationGeneralizationCodeIdentity,
    GenerationGeneralizationPublicDryRunReceipt,
    build_generation_generalization_code_identity,
    build_generation_generalization_evaluation_family_freeze,
    double_scan_generation_generalization_observation_inventory,
    generation_generalization_verdict_contract_sha256,
    read_generation_generalization_private_label_owner_receipt,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationPolicy,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_SAMPLE = _REPOSITORY / "data/ph2/grounded_answer_train_v1.jsonl.sample"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation_inventory(path: Path, budget) -> None:
    """从公开课程剥离四类输入并按 stable key 冻结规范顺序。"""
    episodes = tuple(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.question.answer_plan.response_act in {
            "ANSWER", "CLARIFY", "CONFLICT"}
    )
    observations = tuple(sorted((
        GenerationGeneralizationEvaluationObservation.from_held_out_episode(
            replace(
                item,
                episode_id=f"family-e05-{index}-{item.episode_id}",
                split="held_out",
            ),
            budget,
        )
        for index, item in enumerate(episodes, start=1)
    ), key=lambda item: item.stable_key()))
    path.write_bytes(b"".join(
        canonical_json_line(item.to_dict()) for item in observations))


def test_family_freeze_binds_double_pass_owner_metadata_and_shared_guard(
        tmp_path: Path) -> None:
    """freeze 不读 label，且锁定六路、代码、policy、阈值与唯一 guard。"""
    budget = GenerationGeneralizationEvaluationBudget(512, 4, 4, 96, 16)
    observation_path = tmp_path / "held-out.observations.jsonl"
    _observation_inventory(observation_path, budget)
    inventory = double_scan_generation_generalization_observation_inventory(
        observation_path, resource_ceiling=budget)
    assert inventory.record_count == 4
    assert all(count > 0 for _requirement, count in inventory.requirement_counts)

    label_relative = "labels/held-out.labels.jsonl"
    assert not (tmp_path / label_relative).exists()
    owner_value = {
        "artifact_kind": PRIVATE_OWNER_ARTIFACT_KIND,
        "format_version": 1,
        "label_commitment_sha256": _sha("labels"),
        "label_file": {
            "content_sha256": _sha("label-content"),
            "content_size_bytes": 101,
            "record_count": inventory.record_count,
            "relative_path": label_relative,
            "transport_sha256": _sha("label-transport"),
            "transport_size_bytes": 79,
        },
        "observation_inventory_sha256": inventory.transport_sha256,
        "status": "SEALED_UNREAD",
        "verdict_contract_sha256": (
            generation_generalization_verdict_contract_sha256()),
    }
    owner_path = tmp_path / "owner-receipt.json"
    owner_path.write_bytes(canonical_json_line(owner_value))
    owner, owner_receipt_sha = (
        read_generation_generalization_private_label_owner_receipt(owner_path))
    assert not (tmp_path / label_relative).exists()

    actual_code = build_generation_generalization_code_identity(_REPOSITORY)
    code_paths = {item.relative_path for item in actual_code.files}
    assert any(path.endswith(
        "ph2_generation_generalization_evaluation_runner.py")
        for path in code_paths)
    assert any(path.endswith(
        "ph2_generation_generalization_evaluation_family.py")
        for path in code_paths)

    policy = GenerationGeneralizationEvaluationPolicy()
    candidate_sha = _sha("candidate")
    dry_run = GenerationGeneralizationPublicDryRunReceipt(
        candidate_sha,
        actual_code.aggregate_sha256,
        hashlib.sha256(canonical_json_bytes(policy.to_dict())).hexdigest(),
        _sha("public-batch"),
        _sha("public-observations"),
        inventory.record_count,
        "PASS",
    )
    freeze = build_generation_generalization_evaluation_family_freeze(
        public_head_sha1="1" * 40,
        candidate_manifest_relative_path="candidate/manifest.json",
        candidate_manifest_sha256=_sha("candidate-manifest"),
        candidate_manifest_size_bytes=123,
        candidate_payload_sha256=candidate_sha,
        candidate_training_artifact_sha256=_sha("train"),
        code_identity=actual_code,
        policy=policy,
        observation_inventory_relative_path="observations/held-out.jsonl",
        observation_inventory=inventory,
        private_owner_receipt_relative_path="owner-receipt.json",
        private_owner_receipt_sha256=owner_receipt_sha,
        private_owner=owner,
        public_dry_run=dry_run,
    )
    assert freeze["observation_inventory"]["double_pass_equal"] == 1
    assert freeze["threshold_contract"][
        "hidden_or_numeric_threshold_count"] == 0
    assert freeze["label_read_count_before_prediction_seal"] == 0
    assert freeze["runner_contract"][
        "parallel_private_generation_logic_allowed"] == 0

    manifest_sha = hashlib.sha256(canonical_json_line(freeze)).hexdigest()
    guard = build_available_guard_for_identity(
        manifest_sha,
        freeze["family_commitment_sha256"],
        owner_receipt_sha,
        candidate_sha,
        actual_code.aggregate_sha256,
    )
    assert guard.state == "AVAILABLE"
    assert guard.formal_run_count_before == 0
    assert guard.private_payload_reads_before == 0
