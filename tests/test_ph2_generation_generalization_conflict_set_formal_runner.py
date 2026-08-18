from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetEvidence,
    build_conflict_set_plan,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_family import (
    build_conflict_set_family_code_identity,
    build_conflict_set_family_freeze,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_private import (
    encode_conflict_set_semantic_labels,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_protocol import (
    build_conflict_set_semantic_label_record,
)
from pure_integer_ai.experiments import (
    ph2_generation_generalization_conflict_set_formal_private as formal_private,
    ph2_generation_generalization_conflict_set_formal_runner as formal_runner,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    TRANSPORT_ROOT_NAMESPACE,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_public_preflight import (
    read_conflict_set_public_preflight,
)
from tests.test_ph2_generation_generalization_conflict_set_family import (
    _PREFLIGHT_PATH,
    _REPOSITORY,
    _metadata,
    _transport,
)


def _plan():
    return build_conflict_set_plan(
        scope_id=901,
        claim_ids=("claim-a", "claim-b"),
        evidence=(
            ConflictSetEvidence("e1", "claim-a", "source-a", 901, 1, 0),
            ConflictSetEvidence("e2", "claim-a", "source-b", 901, 0, 1),
            ConflictSetEvidence("e3", "claim-b", "source-c", 901, 1, 0),
            ConflictSetEvidence("e4", "claim-b", "source-d", 901, 0, 1),
        ),
    )


def _setup(tmp_path: Path):
    plan = _plan()
    keys = ("a" * 64, "b" * 64)
    labels = tuple(
        build_conflict_set_semantic_label_record(key, plan.projection)
        for key in keys)
    transport_bytes, content = encode_conflict_set_semantic_labels(
        labels, compressed=True)
    public = read_conflict_set_public_preflight(_PREFLIGHT_PATH)
    code = build_conflict_set_family_code_identity(_REPOSITORY)
    transport = _transport(code, public)
    artifacts = list(transport.artifacts)
    label_index = next(i for i, item in enumerate(artifacts)
                       if item.role == "private_labels")
    artifacts[label_index] = replace(
        artifacts[label_index],
        relative_path=f"{TRANSPORT_ROOT_NAMESPACE}/private-labels.jsonl.gz",
        transport_sha256=hashlib.sha256(transport_bytes).hexdigest(),
        transport_size_bytes=len(transport_bytes),
        content_sha256=hashlib.sha256(content).hexdigest(),
        content_size_bytes=len(content),
        record_count=len(labels),
    )
    prediction_index = next(i for i, item in enumerate(artifacts)
                            if item.role == "prediction_seal")
    artifacts[prediction_index] = replace(
        artifacts[prediction_index],
        relative_path=f"{TRANSPORT_ROOT_NAMESPACE}/predictions.seal.json",
    )
    transport = replace(transport, artifacts=tuple(artifacts))
    metadata = _metadata(transport)
    freeze = build_conflict_set_family_freeze(
        public_preflight=public,
        family_code_identity=code,
        transport=transport,
        owner_metadata=metadata,
    )
    transport_root = tmp_path / TRANSPORT_ROOT_NAMESPACE
    transport_root.mkdir()
    (transport_root / "private-labels.jsonl.gz").write_bytes(transport_bytes)
    family = tmp_path / "formal-family-v1"
    return plan, keys, freeze, family, transport_root


def _patch_roots(monkeypatch):
    monkeypatch.setattr(formal_runner, "_require_k_run_root",
                        lambda value: Path(value).resolve())
    monkeypatch.setattr(formal_private, "_run_root",
                        lambda value: Path(value).resolve())


def test_runner_passes_seals_before_label_and_rejects_replay(tmp_path, monkeypatch):
    plan, keys, freeze, family, _transport_root = _setup(tmp_path)
    _patch_roots(monkeypatch)
    formal_runner.publish_conflict_set_family_freeze(
        run_root=tmp_path, family_root=family, freeze=freeze)

    def execute(key):
        return formal_runner.ConflictSetActualRun(
            key, "c" * 64, "PASS", plan.projection)

    publication = formal_runner.run_conflict_set_formal_evaluation_once(
        repository_root=_REPOSITORY,
        run_root=tmp_path,
        family_root=family,
        freeze=freeze,
        observation_keys=keys,
        actual_executor=execute,
    )
    assert publication.aggregate.status == "PASS"
    assert publication.runtime_receipt is not None
    assert publication.failure_seal is None
    assert (family / "publication" / "runtime_receipt.json").is_file()
    with pytest.raises(formal_runner.ConflictSetFormalRunnerError):
        formal_runner.run_conflict_set_formal_evaluation_once(
            repository_root=_REPOSITORY,
            run_root=tmp_path,
            family_root=family,
            freeze=freeze,
            observation_keys=keys,
            actual_executor=execute,
        )


@pytest.mark.parametrize("status", ("FAIL", "NE"))
def test_runner_publishes_nonpass_failure_seal_without_receipt(
        tmp_path, monkeypatch, status):
    plan, keys, freeze, family, _transport_root = _setup(tmp_path)
    _patch_roots(monkeypatch)
    formal_runner.publish_conflict_set_family_freeze(
        run_root=tmp_path, family_root=family, freeze=freeze)

    def execute(key):
        return formal_runner.ConflictSetActualRun(
            key, "c" * 64, status,
            None if status == "NE" else plan.projection)

    publication = formal_runner.run_conflict_set_formal_evaluation_once(
        repository_root=_REPOSITORY,
        run_root=tmp_path,
        family_root=family,
        freeze=freeze,
        observation_keys=keys,
        actual_executor=execute,
    )
    assert publication.aggregate.status == status
    assert publication.runtime_receipt is None
    assert publication.failure_seal is not None
    assert not (family / "publication" / "runtime_receipt.json").exists()
    assert (family / "publication" / "failure_seal.json").is_file()
