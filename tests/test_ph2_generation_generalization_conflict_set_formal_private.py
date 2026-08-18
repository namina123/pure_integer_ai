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
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_protocol import (
    build_conflict_set_semantic_label_record,
    build_conflict_set_semantic_prediction_record,
    build_conflict_set_semantic_prediction_seal,
)
from pure_integer_ai.experiments import (
    ph2_generation_generalization_conflict_set_formal_private as formal_private,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    ConflictSetPrivateArtifact,
    EVALUATOR_OWNER,
    TRANSPORT_ROOT_NAMESPACE,
    consume_conflict_set_run_guard,
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


def _fixture(tmp_path: Path):
    plan = _plan()
    keys = ("a" * 64, "b" * 64)
    labels = tuple(
        build_conflict_set_semantic_label_record(key, plan.projection)
        for key in keys)
    transport_bytes, content = formal_private.encode_conflict_set_semantic_labels(
        labels, compressed=True)

    public = read_conflict_set_public_preflight(_PREFLIGHT_PATH)
    code = build_conflict_set_family_code_identity(_REPOSITORY)
    transport = _transport(code, public)
    artifacts = list(transport.artifacts)
    private_index = next(
        index for index, item in enumerate(artifacts)
        if item.role == "private_labels")
    artifacts[private_index] = replace(
        artifacts[private_index],
        relative_path=(
            f"{TRANSPORT_ROOT_NAMESPACE}/private-labels.jsonl.gz"),
        transport_sha256=hashlib.sha256(transport_bytes).hexdigest(),
        transport_size_bytes=len(transport_bytes),
        content_sha256=hashlib.sha256(content).hexdigest(),
        content_size_bytes=len(content),
        record_count=len(labels),
    )
    prediction_index = next(
        index for index, item in enumerate(artifacts)
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
    predictions = tuple(
        build_conflict_set_semantic_prediction_record(
            key, candidate_key, "PASS", plan.projection)
        for key, candidate_key in zip(
            keys, ("c" * 64, "d" * 64), strict=True))
    prediction = build_conflict_set_semantic_prediction_seal(
        predictions,
        family_manifest_sha256=freeze.sha256(),
        family_commitment_sha256=freeze.family_commitment_sha256,
        candidate_manifest_sha256=transport.candidate_manifest_sha256,
    )
    transport_root = tmp_path / TRANSPORT_ROOT_NAMESPACE
    transport_root.mkdir()
    family = tmp_path / "formal-family-v1"
    family.mkdir()
    (family / "family-freeze.json").write_bytes(freeze.canonical_bytes())
    (transport_root / "private-labels.jsonl.gz").write_bytes(transport_bytes)
    write_immutable_json(
        prediction.to_dict(), transport_root / "predictions.seal.json")
    write_immutable_json(
        freeze.available_guard.to_dict(), family / "guard.available.json")
    return family, freeze, prediction, labels


def _consume(family: Path, freeze):
    consumed, intent = consume_conflict_set_run_guard(
        freeze.available_guard)
    (family / "guard.available.json").unlink()
    write_immutable_json(consumed.to_dict(), family / "guard.consumed.json")
    write_immutable_json(intent.to_dict(), family / "run.intent.json")


def test_label_codec_is_deterministic_and_canonical():
    plan = _plan()
    records = tuple(
        build_conflict_set_semantic_label_record(key, plan.projection)
        for key in ("a" * 64, "b" * 64))
    first, content = formal_private.encode_conflict_set_semantic_labels(
        records, compressed=True)
    second, repeated = formal_private.encode_conflict_set_semantic_labels(
        records, compressed=True)
    assert first == second
    assert content == repeated
    assert formal_private.parse_conflict_set_semantic_label_content(
        content) == records


def test_label_publisher_matches_artifact_and_rejects_replay(tmp_path, monkeypatch):
    plan = _plan()
    records = tuple(
        build_conflict_set_semantic_label_record(key, plan.projection)
        for key in ("a" * 64, "b" * 64))
    transport, content = formal_private.encode_conflict_set_semantic_labels(
        records, compressed=True)
    artifact = ConflictSetPrivateArtifact(
        "private_labels", EVALUATOR_OWNER, "PRIVATE", "PRIVATE_LABEL",
        f"{TRANSPORT_ROOT_NAMESPACE}/private-labels.jsonl.gz",
        hashlib.sha256(transport).hexdigest(), len(transport),
        hashlib.sha256(content).hexdigest(), len(content), len(records),
    )
    monkeypatch.setattr(formal_private, "_run_root",
                        lambda value: Path(value).resolve())
    publication = formal_private.publish_conflict_set_semantic_labels(
        records, run_root=tmp_path, artifact=artifact)
    assert publication["status"] == "PUBLISHED"
    assert publication["record_count"] == len(records)
    with pytest.raises(formal_private.ConflictSetFormalPrivateError,
                       match="不可覆盖"):
        formal_private.publish_conflict_set_semantic_labels(
            records, run_root=tmp_path, artifact=artifact)


def test_private_read_requires_consumed_guard_and_prediction_seal(
        tmp_path, monkeypatch):
    family, freeze, prediction, labels = _fixture(tmp_path)
    monkeypatch.setattr(formal_private, "_run_root",
                        lambda value: Path(value).resolve())
    with pytest.raises(
            formal_private.ConflictSetFormalPrivateError,
            match="guard"):
        formal_private.read_conflict_set_semantic_labels_after_prediction_seal(
            run_root=tmp_path, family_root=family,
            freeze=freeze, prediction=prediction)
    _consume(family, freeze)
    result = formal_private.read_conflict_set_semantic_labels_after_prediction_seal(
        run_root=tmp_path, family_root=family,
        freeze=freeze, prediction=prediction)
    assert result.records == labels
    assert result.read_count == 1


def test_private_read_requires_persisted_run_intent(tmp_path, monkeypatch):
    family, freeze, prediction, _labels = _fixture(tmp_path)
    monkeypatch.setattr(formal_private, "_run_root",
                        lambda value: Path(value).resolve())
    _consume(family, freeze)
    (family / "run.intent.json").unlink()
    with pytest.raises(
            formal_private.ConflictSetFormalPrivateError,
            match="guard/intent"):
        formal_private.read_conflict_set_semantic_labels_after_prediction_seal(
            run_root=tmp_path, family_root=family,
            freeze=freeze, prediction=prediction)


def test_private_read_rejects_missing_prediction_and_label_tamper(
        tmp_path, monkeypatch):
    family, freeze, prediction, _labels = _fixture(tmp_path)
    monkeypatch.setattr(formal_private, "_run_root",
                        lambda value: Path(value).resolve())
    _consume(family, freeze)
    prediction_path = tmp_path / TRANSPORT_ROOT_NAMESPACE / "predictions.seal.json"
    prediction_bytes = prediction_path.read_bytes()
    prediction_path.unlink()
    with pytest.raises(
            formal_private.ConflictSetFormalPrivateError,
            match="prediction seal"):
        formal_private.read_conflict_set_semantic_labels_after_prediction_seal(
            run_root=tmp_path, family_root=family,
            freeze=freeze, prediction=prediction)
    prediction_path.write_bytes(prediction_bytes)
    label_path = tmp_path / TRANSPORT_ROOT_NAMESPACE / "private-labels.jsonl.gz"
    payload = bytearray(label_path.read_bytes())
    payload[-1] ^= 1
    label_path.write_bytes(bytes(payload))
    with pytest.raises(
            formal_private.ConflictSetFormalPrivateError,
            match="transport identity"):
        formal_private.read_conflict_set_semantic_labels_after_prediction_seal(
            run_root=tmp_path, family_root=family,
            freeze=freeze, prediction=prediction)
