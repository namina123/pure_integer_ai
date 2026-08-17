"""GG-03 formal prediction-first publication 的聚焦协议测试。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    build_available_guard_for_identity,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.runtime import (
    consume_evaluation_guard_once,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationEvaluationFamilyError,
    GenerationGeneralizationPrivateLabelOwnerReceipt,
    scan_generation_generalization_observation_inventory,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationBatchRunError,
    GenerationGeneralizationEvaluationPolicy,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_labels import (
    GenerationGeneralizationFormalLabelRecord,
    generation_generalization_surface_sha256,
    read_generation_generalization_private_labels_after_guard,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_protocol import (
    GenerationGeneralizationPredictionRecord,
    GenerationGeneralizationPredictionSeal,
)
from pure_integer_ai.experiments import (
    ph2_generation_generalization_formal_runner as formal_runner,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.storage.backend import DictBackend


_REPOSITORY = Path(__file__).resolve().parents[1]
_SAMPLE = _REPOSITORY / "data/ph2/grounded_answer_train_v1.jsonl.sample"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _inventory(tmp_path: Path):
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
                episode_id=f"formal-e05-{index}-{item.episode_id}",
                split="held_out",
            ),
            budget,
        )
        for index, item in enumerate(episodes, start=1)
    ), key=lambda item: item.stable_key()))
    path = tmp_path / "held-out.observations.jsonl"
    path.write_bytes(b"".join(
        canonical_json_line(item.to_dict()) for item in observations))
    return (
        budget,
        observations,
        path,
        scan_generation_generalization_observation_inventory(
            path, resource_ceiling=budget),
    )


def test_formal_runner_publishes_pass_seals_ne_and_rejects_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """标签只在 guard+prediction 后可读；PASS/NE 均不可覆盖且不可重跑。"""
    budget, observations, observation_path, inventory = _inventory(tmp_path)
    candidate_sha = _sha("candidate")
    owner_sha = _sha("owner")
    code_sha = _sha("code")
    manifest_sha = _sha("manifest")
    commitment_sha = _sha("family")
    policy = GenerationGeneralizationEvaluationPolicy()
    policy_sha = hashlib.sha256(
        canonical_json_bytes(policy.to_dict())).hexdigest()
    guard = build_available_guard_for_identity(
        manifest_sha, commitment_sha, owner_sha, candidate_sha, code_sha)

    surfaces = tuple(f"formal surface {index}"
                     for index in range(1, len(observations) + 1))
    prediction_records = tuple(
        GenerationGeneralizationPredictionRecord(
            identity.stable_key_sha256,
            generation_generalization_surface_sha256(surface),
            _sha(f"run-{index}"),
            tuple((requirement, "PASS")
                  for requirement in identity.requirements),
        )
        for index, (identity, surface) in enumerate(
            zip(inventory.records, surfaces, strict=True), start=1)
    )
    prediction = GenerationGeneralizationPredictionSeal(
        manifest_sha,
        commitment_sha,
        candidate_sha,
        policy_sha,
        _sha("batch"),
        prediction_records,
    )
    labels = tuple(
        GenerationGeneralizationFormalLabelRecord(
            predicted.observation_stable_key_sha256,
            tuple(sorted((
                predicted.surface_sha256,
                generation_generalization_surface_sha256(
                    f"accepted alternative {index}"),
            ))),
            (generation_generalization_surface_sha256(
                f"rejected surface {index}"),),
            tuple(requirement for requirement, _status
                  in predicted.requirement_statuses),
        )
        for index, predicted in enumerate(prediction.records, start=1)
    )
    owner = GenerationGeneralizationPrivateLabelOwnerReceipt(
        inventory.transport_sha256,
        "labels/held-out.labels.jsonl.gz",
        100,
        _sha("label transport"),
        200,
        _sha("label content"),
        inventory.record_count,
        _sha("label commitment"),
        _sha("verdict contract"),
    )
    freeze = {
        "code_identity": {"aggregate_sha256": code_sha},
        "family_commitment_sha256": commitment_sha,
        "manifest_sha256": manifest_sha,
        "observation_inventory": {
            **inventory.to_dict(),
            "double_pass_equal": 1,
            "relative_path": "held-out.observations.jsonl",
        },
        "policy_sha256": policy_sha,
    }
    loaded = SimpleNamespace(pack=SimpleNamespace(sha256=lambda: candidate_sha))
    private_root = tmp_path / "private-label-owner"
    private_root.mkdir()
    owner_path = private_root / "owner-receipt.json"

    monkeypatch.setattr(
        formal_runner,
        "read_generation_generalization_evaluation_family_freeze",
        lambda *_args, **_kwargs: freeze,
    )
    monkeypatch.setattr(
        formal_runner,
        "read_generation_generalization_private_label_owner_receipt",
        lambda _path: (owner, owner_sha),
    )
    monkeypatch.setattr(
        formal_runner,
        "scan_generation_generalization_observation_inventory",
        lambda *_args, **_kwargs: inventory,
    )
    monkeypatch.setattr(
        formal_runner,
        "read_generation_generalization_evaluation_observations",
        lambda _path: observations,
    )

    def prepare_family(name: str) -> Path:
        family = tmp_path / name
        family.mkdir()
        write_immutable_json(guard.to_dict(), family / "guard.available.json")
        return family

    precondition_family = prepare_family("precondition-family")
    with pytest.raises(
            EvaluationKernelContractError,
            match="available guard still exists"):
        read_generation_generalization_private_labels_after_guard(
            family_root=precondition_family,
            expected_guard=guard,
            prediction_seal_path=precondition_family / "predictions.seal.json",
            prediction_seal_sha256=prediction.sha256(),
            private_label_root=private_root,
            owner_receipt=owner,
            observation_inventory=inventory,
        )
    consumed_family = prepare_family("consumed-family")
    consume_evaluation_guard_once(consumed_family, guard)
    with pytest.raises(
            GenerationGeneralizationEvaluationFamilyError,
            match="prediction seal"):
        read_generation_generalization_private_labels_after_guard(
            family_root=consumed_family,
            expected_guard=guard,
            prediction_seal_path=consumed_family / "predictions.seal.json",
            prediction_seal_sha256=prediction.sha256(),
            private_label_root=private_root,
            owner_receipt=owner,
            observation_inventory=inventory,
        )

    current_family = {"path": prepare_family("pass-family")}

    def fake_batch(*_args, **_kwargs):
        assert (current_family["path"] / "guard.consumed.json").is_file()
        return object()

    def fake_prediction(_batch, **_kwargs):
        assert not (current_family["path"] / "publication").exists()
        return prediction

    def pass_labels(**kwargs):
        assert kwargs["family_root"] == current_family["path"]
        assert kwargs["prediction_seal_sha256"] == prediction.sha256()
        assert Path(kwargs["prediction_seal_path"]).is_file()
        return labels

    monkeypatch.setattr(
        formal_runner, "run_generation_generalization_evaluation_batch",
        fake_batch)
    monkeypatch.setattr(
        formal_runner, "build_generation_generalization_prediction_seal",
        fake_prediction)
    monkeypatch.setattr(
        formal_runner,
        "read_generation_generalization_private_labels_after_guard",
        pass_labels,
    )

    backend = DictBackend()
    try:
        host = make_train_context(backend)
        arguments = {
            "repository_root": _REPOSITORY,
            "run_root": tmp_path,
            "candidate_visible_root": tmp_path,
            "private_label_root": private_root,
            "loaded_candidate": loaded,
            "observation_inventory_path": observation_path,
            "private_owner_receipt_path": owner_path,
            "public_dry_run_receipt_path": (
                tmp_path / "public-preflight"
                / "public-dry-run.receipt.json"),
            "policy": policy,
            "resource_ceiling": budget,
        }
        passed = formal_runner.run_generation_generalization_formal_evaluation_once(
            host, family_dir=current_family["path"], **arguments)
        assert passed.aggregate.status == "PASS"
        assert passed.runtime_receipt is not None
        assert passed.failure_seal is None
        assert passed.failure_diagnostic is None
        assert (current_family["path"] / "publication"
                / "runtime_receipt.json").is_file()
        with pytest.raises(
                GenerationGeneralizationEvaluationFamilyError,
                match="formal outputs"):
            formal_runner.run_generation_generalization_formal_evaluation_once(
                host, family_dir=current_family["path"], **arguments)

        current_family["path"] = prepare_family("ne-family")

        def unavailable_labels(**kwargs):
            assert Path(kwargs["prediction_seal_path"]).is_file()
            raise GenerationGeneralizationEvaluationFamilyError(
                "synthetic private label unavailable")

        monkeypatch.setattr(
            formal_runner,
            "read_generation_generalization_private_labels_after_guard",
            unavailable_labels,
        )
        unavailable = (
            formal_runner.run_generation_generalization_formal_evaluation_once(
                host, family_dir=current_family["path"], **arguments))
        assert unavailable.aggregate.status == "NE"
        assert unavailable.aggregate.failure_phase == "PRIVATE_LABEL_READ"
        assert unavailable.runtime_receipt is None
        assert unavailable.failure_seal is not None
        assert unavailable.failure_diagnostic is not None
        assert (current_family["path"] / "publication"
                / "failure_seal.json").is_file()
        diagnostic_path = (
            current_family["path"] / "publication"
            / "failure_diagnostic.json")
        assert diagnostic_path.is_file()
        diagnostic = read_canonical_object(diagnostic_path)
        assert diagnostic["failure_phase"] == "PRIVATE_LABEL_READ"
        assert diagnostic["observation_ordinal"] == 0
        assert diagnostic["evaluation_path"] == "UNAVAILABLE"
        assert diagnostic["message_or_input_field_count"] == 0
        assert "synthetic private label unavailable" not in str(diagnostic)
        assert unavailable.failure_seal[
            "failure_diagnostic_sha256"] == hashlib.sha256(
                diagnostic_path.read_bytes()).hexdigest()
        with pytest.raises(
                GenerationGeneralizationEvaluationFamilyError,
                match="formal outputs"):
            formal_runner.run_generation_generalization_formal_evaluation_once(
                host, family_dir=current_family["path"], **arguments)

        current_family["path"] = prepare_family("candidate-ne-family")

        def unavailable_candidate(*_args, **_kwargs):
            try:
                raise RuntimeError("private observation text must not escape")
            except RuntimeError as cause:
                raise GenerationGeneralizationEvaluationBatchRunError(
                    2, "RESPONSE_ACT") from cause

        monkeypatch.setattr(
            formal_runner,
            "run_generation_generalization_evaluation_batch",
            unavailable_candidate,
        )
        candidate_unavailable = (
            formal_runner.run_generation_generalization_formal_evaluation_once(
                host, family_dir=current_family["path"], **arguments))
        assert candidate_unavailable.aggregate.status == "NE"
        assert candidate_unavailable.aggregate.failure_phase == "CANDIDATE_RUN"
        assert candidate_unavailable.aggregate.label_read_count == 0
        candidate_diagnostic = candidate_unavailable.failure_diagnostic
        assert candidate_diagnostic is not None
        assert candidate_diagnostic["observation_ordinal"] == 2
        assert candidate_diagnostic["evaluation_path"] == "RESPONSE_ACT"
        assert candidate_diagnostic["exception_type"] == "builtins.RuntimeError"
        assert "private observation text" not in str(candidate_diagnostic)
        assert not (current_family["path"] / "predictions.seal.json").exists()
    finally:
        backend.close()
