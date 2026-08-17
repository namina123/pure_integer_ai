"""GG-03 V2 semantic family 与 one-shot runner 聚焦协议测试。"""
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments import (
    ph2_generation_generalization_semantic_family as semantic_family,
    ph2_generation_generalization_semantic_formal_runner as semantic_runner,
    ph2_generation_generalization_semantic_preflight as semantic_preflight,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    build_available_guard_for_identity,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationCodeFileIdentity,
    GenerationGeneralizationCodeIdentity,
    GenerationGeneralizationEvaluationFamilyError,
    GenerationGeneralizationPrivateLabelOwnerReceipt,
    generation_generalization_sha256_bytes,
    scan_generation_generalization_observation_inventory,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationPolicy,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_family import (
    SEMANTIC_FAMILY_ARTIFACT_KIND,
    build_generation_generalization_semantic_evaluation_family_freeze,
    publish_generation_generalization_semantic_evaluation_family_freeze,
    read_generation_generalization_semantic_evaluation_family_freeze,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    GenerationGeneralizationSemanticLabelRecord,
    generation_generalization_semantic_verdict_contract_sha256,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_preflight import (
    GenerationGeneralizationSemanticPublicDryRunReceipt,
    publish_generation_generalization_semantic_public_dry_run_receipt,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_protocol import (
    GenerationGeneralizationSemanticPredictionRecord,
    GenerationGeneralizationSemanticPredictionSeal,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend


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
                episode_id=f"semantic-formal-{index}-{item.episode_id}",
                split="held_out",
            ),
            budget,
        )
        for index, item in enumerate(episodes, start=1)
    ), key=lambda item: item.stable_key()))
    path = tmp_path / "formal-observations.jsonl"
    path.write_bytes(b"".join(
        canonical_json_line(item.to_dict()) for item in observations))
    inventory = scan_generation_generalization_observation_inventory(
        path, resource_ceiling=budget)
    return budget, observations, path, inventory


def _code_identity() -> GenerationGeneralizationCodeIdentity:
    """构造不依赖工作树状态的最小公开 code identity。"""
    item = GenerationGeneralizationCodeFileIdentity(
        "src/pure_integer_ai/semantic_protocol.py", 1, _sha("code"))
    return GenerationGeneralizationCodeIdentity(
        (item,),
        generation_generalization_sha256_bytes(canonical_json_bytes(
            [item.to_dict()])),
    )


def test_semantic_family_freeze_binds_v2_contract_and_preflight(
        tmp_path: Path) -> None:
    """V2 freeze 绑定 semantic owner、preflight、code 与零污染边界。"""
    budget, _observations, _path, inventory = _inventory(tmp_path)
    assert budget.max_surface_units > 0
    code = _code_identity()
    policy = GenerationGeneralizationEvaluationPolicy()
    policy_sha = generation_generalization_sha256_bytes(
        canonical_json_bytes(policy.to_dict()))
    candidate_sha = _sha("candidate")
    contract = generation_generalization_semantic_verdict_contract_sha256()
    owner = GenerationGeneralizationPrivateLabelOwnerReceipt(
        inventory.transport_sha256,
        "labels/formal-semantic-labels.jsonl.gz",
        101,
        _sha("label-transport"),
        202,
        _sha("label-content"),
        inventory.record_count,
        _sha("label-commitment"),
        contract,
    )
    public = GenerationGeneralizationSemanticPublicDryRunReceipt(
        candidate_sha,
        code.aggregate_sha256,
        policy_sha,
        _sha("public-batch"),
        _sha("public-observations"),
        tuple(sorted(_sha(f"public-stable-{index}")
                     for index in range(inventory.record_count))),
        tuple(sorted(_sha(f"public-content-{index}")
                     for index in range(inventory.record_count))),
        _sha("public-semantic-projections"),
        contract,
        inventory.record_count,
        "PASS",
    )
    freeze = build_generation_generalization_semantic_evaluation_family_freeze(
        public_head_sha1="1" * 40,
        candidate_manifest_relative_path="candidate/manifest.json",
        candidate_manifest_sha256=_sha("manifest"),
        candidate_manifest_size_bytes=123,
        candidate_payload_sha256=candidate_sha,
        candidate_training_artifact_sha256=_sha("training"),
        code_identity=code,
        policy=policy,
        observation_inventory_relative_path="observations/formal.jsonl",
        observation_inventory=inventory,
        private_owner_receipt_relative_path="owner-receipt.json",
        private_owner_receipt_sha256=_sha("owner-receipt"),
        private_owner=owner,
        public_dry_run_receipt_relative_path=(
            "public-preflight/public-semantic-dry-run.receipt.json"),
        public_dry_run_receipt_sha256=_sha("public-receipt"),
        public_dry_run=public,
    )
    assert freeze["artifact_kind"] == SEMANTIC_FAMILY_ARTIFACT_KIND
    assert freeze["format_version"] == 2
    assert freeze["aggregate_contract"][
        "projection_verdict_contract_sha256"] == contract
    assert freeze["private_owner"]["verdict_contract_sha256"] == contract
    assert freeze["public_semantic_dry_run"]["status"] == "PASS"
    assert freeze["label_read_count_before_prediction_seal"] == 0
    assert freeze["threshold_contract"][
        "hidden_or_numeric_threshold_count"] == 0


def test_semantic_family_publication_is_immutable_and_strictly_readable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """freeze/guard 事务发布可严格重算回读，重复 target fail closed。"""
    root = tmp_path / "semantic-run"
    candidate_root = root / "candidate-visible"
    private_root = root / "private-label-owner"
    preflight_root = root / "public-preflight"
    candidate_root.mkdir(parents=True)
    private_root.mkdir()
    preflight_root.mkdir()
    budget, _observations, observation_path, inventory = _inventory(
        candidate_root)
    manifest_path = candidate_root / "candidate-manifest.json"
    write_immutable_json({"candidate": "semantic-test"}, manifest_path)
    code = _code_identity()
    candidate_sha = _sha("candidate")
    training_sha = _sha("training")
    policy = GenerationGeneralizationEvaluationPolicy()
    policy_sha = generation_generalization_sha256_bytes(
        canonical_json_bytes(policy.to_dict()))
    contract = generation_generalization_semantic_verdict_contract_sha256()
    owner = GenerationGeneralizationPrivateLabelOwnerReceipt(
        inventory.transport_sha256,
        "labels/formal-semantic-labels.jsonl.gz",
        101,
        _sha("label-transport"),
        202,
        _sha("label-content"),
        inventory.record_count,
        _sha("label-commitment"),
        contract,
    )
    owner_path = private_root / "owner-receipt.json"
    write_immutable_json(owner.to_dict(), owner_path)
    public = GenerationGeneralizationSemanticPublicDryRunReceipt(
        candidate_sha,
        code.aggregate_sha256,
        policy_sha,
        _sha("public-batch"),
        _sha("public-observations"),
        tuple(sorted(_sha(f"public-stable-{index}")
                     for index in range(inventory.record_count))),
        tuple(sorted(_sha(f"public-content-{index}")
                     for index in range(inventory.record_count))),
        _sha("public-semantic-projections"),
        contract,
        inventory.record_count,
        "PASS",
    )
    public_path = (
        preflight_root / "public-semantic-dry-run.receipt.json")
    monkeypatch.setattr(
        semantic_preflight,
        "require_generation_generalization_k_run_root",
        lambda value: Path(value).resolve(),
    )
    publish_generation_generalization_semantic_public_dry_run_receipt(
        public, run_root=root, target_path=public_path)
    with pytest.raises(
            GenerationGeneralizationEvaluationFamilyError,
            match="receipt 已存在"):
        publish_generation_generalization_semantic_public_dry_run_receipt(
            public, run_root=root, target_path=public_path)
    loaded = SimpleNamespace(
        manifest_path=manifest_path,
        pack=SimpleNamespace(
            sha256=lambda: candidate_sha,
            training_artifact_sha256=training_sha,
        ),
    )
    monkeypatch.setattr(
        semantic_family, "LoadedGenerationCandidatePack", SimpleNamespace)
    monkeypatch.setattr(
        semantic_family,
        "require_generation_generalization_k_run_root",
        lambda value: Path(value).resolve(),
    )
    monkeypatch.setattr(
        semantic_family,
        "_published_git_head",
        lambda _repository: "1" * 40,
    )
    monkeypatch.setattr(
        semantic_family,
        "build_generation_generalization_semantic_code_identity",
        lambda _repository: code,
    )
    arguments = {
        "repository_root": tmp_path,
        "run_root": root,
        "candidate_visible_root": candidate_root,
        "private_label_root": private_root,
        "loaded_candidate": loaded,
        "observation_inventory_path": observation_path,
        "private_owner_receipt_path": owner_path,
        "public_dry_run_receipt_path": public_path,
        "policy": policy,
        "resource_ceiling": budget,
    }
    target = root / "formal-family"
    published = (
        publish_generation_generalization_semantic_evaluation_family_freeze(
            target_dir=target, **arguments))
    assert {item.name for item in target.iterdir()} == {
        "family-freeze.json", "guard.available.json"}
    readback = (
        read_generation_generalization_semantic_evaluation_family_freeze(
            target, **arguments))
    assert readback == published
    with pytest.raises(
            GenerationGeneralizationEvaluationFamilyError,
            match="target 已存在"):
        publish_generation_generalization_semantic_evaluation_family_freeze(
            target_dir=target, **arguments)


def test_semantic_runner_publishes_pass_fail_ne_and_rejects_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fresh guard 分别发布 PASS/FAIL/NE，且 capability 态不带 diagnostic。"""
    budget, observations, observation_path, inventory = _inventory(tmp_path)
    manifest_sha = _sha("manifest")
    commitment_sha = _sha("family")
    candidate_sha = _sha("candidate")
    code_sha = _sha("code")
    owner_sha = _sha("owner")
    policy = GenerationGeneralizationEvaluationPolicy()
    policy_sha = hashlib.sha256(
        canonical_json_bytes(policy.to_dict())).hexdigest()
    contract = generation_generalization_semantic_verdict_contract_sha256()
    records = tuple(
        GenerationGeneralizationSemanticPredictionRecord(
            identity.stable_key_sha256,
            _sha(f"semantic-{ordinal}"),
            _sha(f"run-{ordinal}"),
            tuple((requirement, "PASS")
                  for requirement in identity.requirements),
        )
        for ordinal, identity in enumerate(inventory.records, start=1)
    )
    baseline_prediction = GenerationGeneralizationSemanticPredictionSeal(
        manifest_sha,
        commitment_sha,
        candidate_sha,
        policy_sha,
        _sha("batch"),
        contract,
        records,
    )
    baseline_labels = tuple(
        GenerationGeneralizationSemanticLabelRecord(
            record.observation_stable_key_sha256,
            record.semantic_projection_sha256,
            tuple(requirement for requirement, _status
                  in record.requirement_statuses),
        )
        for record in baseline_prediction.records
    )
    owner = GenerationGeneralizationPrivateLabelOwnerReceipt(
        inventory.transport_sha256,
        "labels/formal-semantic-labels.jsonl.gz",
        111,
        _sha("label-transport"),
        222,
        _sha("label-content"),
        inventory.record_count,
        _sha("label-commitment"),
        contract,
    )
    freeze = {
        "code_identity": {"aggregate_sha256": code_sha},
        "family_commitment_sha256": commitment_sha,
        "manifest_sha256": manifest_sha,
        "observation_inventory": {
            **inventory.to_dict(),
            "double_pass_equal": 1,
            "relative_path": "observations/formal.jsonl",
        },
        "policy_sha256": policy_sha,
    }
    guard = build_available_guard_for_identity(
        manifest_sha, commitment_sha, owner_sha, candidate_sha, code_sha)
    loaded = SimpleNamespace(pack=SimpleNamespace(
        sha256=lambda: candidate_sha))
    private_root = tmp_path / "private-label-owner"
    private_root.mkdir()
    owner_path = private_root / "owner-receipt.json"

    monkeypatch.setattr(
        semantic_runner,
        "read_generation_generalization_semantic_evaluation_family_freeze",
        lambda *_args, **_kwargs: freeze,
    )
    monkeypatch.setattr(
        semantic_runner,
        "read_generation_generalization_private_label_owner_receipt",
        lambda _path: (owner, owner_sha),
    )
    monkeypatch.setattr(
        semantic_runner,
        "scan_generation_generalization_observation_inventory",
        lambda *_args, **_kwargs: inventory,
    )
    monkeypatch.setattr(
        semantic_runner,
        "read_generation_generalization_evaluation_observations",
        lambda _path: observations,
    )
    monkeypatch.setattr(
        semantic_runner,
        "run_generation_generalization_evaluation_batch",
        lambda *_args, **_kwargs: object(),
    )
    current = {
        "prediction": baseline_prediction,
        "labels": baseline_labels,
    }

    def fake_prediction(_batch, **_kwargs):
        return current["prediction"]

    def fake_labels(**kwargs):
        assert Path(kwargs["prediction_seal_path"]).is_file()
        assert kwargs["prediction_seal_sha256"] == (
            current["prediction"].sha256())
        return current["labels"]

    monkeypatch.setattr(
        semantic_runner,
        "build_generation_generalization_semantic_prediction_seal",
        fake_prediction,
    )
    monkeypatch.setattr(
        semantic_runner,
        "read_generation_generalization_semantic_labels_after_guard",
        fake_labels,
    )

    def prepare_family(name: str) -> Path:
        family = tmp_path / name
        family.mkdir()
        write_immutable_json(
            guard.to_dict(), family / "guard.available.json")
        return family

    arguments = {
        "repository_root": tmp_path,
        "run_root": tmp_path,
        "candidate_visible_root": tmp_path,
        "private_label_root": private_root,
        "loaded_candidate": loaded,
        "observation_inventory_path": observation_path,
        "private_owner_receipt_path": owner_path,
        "public_dry_run_receipt_path": (
            tmp_path / "public-preflight"
            / "public-semantic-dry-run.receipt.json"),
        "policy": policy,
        "resource_ceiling": budget,
    }
    backend = DictBackend()
    try:
        host = make_train_context(backend)
        pass_family = prepare_family("semantic-pass-family")
        passed = (
            semantic_runner.run_generation_generalization_semantic_formal_evaluation_once(
                host, family_dir=pass_family, **arguments))
        assert passed.aggregate.status == "PASS"
        assert passed.runtime_receipt is not None
        assert passed.failure_seal is None
        assert passed.failure_diagnostic is None
        assert (pass_family / "publication" / "runtime_receipt.json").is_file()
        with pytest.raises(
                GenerationGeneralizationEvaluationFamilyError,
                match="formal outputs"):
            semantic_runner.run_generation_generalization_semantic_formal_evaluation_once(
                host, family_dir=pass_family, **arguments)

        fail_family = prepare_family("semantic-fail-family")
        current["labels"] = (
            replace(
                baseline_labels[0],
                expected_semantic_sha256=_sha("different-semantic"),
            ),
            *baseline_labels[1:],
        )
        failed = (
            semantic_runner.run_generation_generalization_semantic_formal_evaluation_once(
                host, family_dir=fail_family, **arguments))
        assert failed.aggregate.status == "FAIL"
        assert failed.aggregate.failure_phase == "NONE"
        assert failed.runtime_receipt is None
        assert failed.failure_seal is not None
        assert failed.failure_diagnostic is None
        assert not (fail_family / "publication"
                    / "failure_diagnostic.json").exists()

        ne_family = prepare_family("semantic-ne-family")
        current["labels"] = baseline_labels
        current["prediction"] = replace(
            baseline_prediction,
            records=(
                replace(
                    baseline_prediction.records[0],
                    semantic_projection_sha256=None,
                ),
                *baseline_prediction.records[1:],
            ),
        )
        unavailable = (
            semantic_runner.run_generation_generalization_semantic_formal_evaluation_once(
                host, family_dir=ne_family, **arguments))
        assert unavailable.aggregate.status == "NE"
        assert unavailable.aggregate.failure_phase == "NONE"
        assert unavailable.runtime_receipt is None
        assert unavailable.failure_seal is not None
        assert unavailable.failure_diagnostic is None
        assert not (ne_family / "publication"
                    / "failure_diagnostic.json").exists()
    finally:
        backend.close()
