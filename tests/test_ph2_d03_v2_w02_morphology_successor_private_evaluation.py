"""W-02 morphology successor blind private evaluator 的公开合成专项。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2AccessPermit,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    run_w02_candidate_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _owner_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import W02FileFreeze
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    run_w02_morphology_overlay_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_private_evaluator import (
    W02MorphologySuccessorPrivateEvaluationError,
    evaluate_w02_private_pair_stream,
    iter_w02_private_records,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_private_family import (
    W02_PRIVATE_FAMILY_CODE_PATHS,
    W02_PRIVATE_GUARD_AVAILABLE,
    W02MorphologySuccessorPrivateFamilyError,
    build_w02_morphology_successor_private_family_freeze,
    consume_w02_morphology_successor_private_guard,
    verify_w02_morphology_successor_private_consumed_guard,
    w02_private_guard_value,
    w02_private_registration_from_freeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    TeacherEvidenceRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(split: str, ordinal: int, surface: str):
    return _source_record(
        "UD_ZH_GSDSIMP_R2_18", split, ordinal,
        snapshot_id=f"public-synthetic-{split}", revision_id="1",
        official_url="https://example.invalid/public-synthetic",
        source_identity=f"public-synthetic:{split}:{ordinal}",
        upstream_checksum="sha256:" + _sha(f"upstream:{split}:{ordinal}"),
        local_sha256=_sha(f"local:{split}:{ordinal}"),
        license_id="CC-BY-SA-4.0", attribution="public synthetic fixture",
        locator_kind="record", locator_value=str(ordinal),
        span_end=len(surface))


def _expected(form: str, upos: str = "VERB") -> dict[str, object]:
    return {
        "boundary_spans": [{"end": len(form), "form": form, "start": 0}],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [{
            "feats": [], "form": form, "lemma": form,
            "node_id": [1], "upos": upos,
        }],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }


def _training_pairs():
    rows = ("猫化", "纸化", "木化", "石化", "新词")
    pairs = []
    for ordinal, form in enumerate(rows, start=1):
        source = _source("train", ordinal, form)
        observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source,
            carrier_kind="plain_text", surface=form, family_ordinal=ordinal,
            sample_role="support", perturbation_kind="NONE")
        evidence = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source, observation,
            _expected(form, "VERB" if form.endswith("化") else "NOUN"),
            dimension_name=W02_DEV_DIMENSIONS[2])
        assert isinstance(evidence, TeacherEvidenceRecord)
        pairs.append((observation, evidence))
    return tuple(pairs)


def _private_pairs(*, morphology_upos: str = "VERB"):
    pairs = []
    for ordinal, dimension in enumerate(W02_DEV_DIMENSIONS, start=1):
        form = "新化"
        source = _source("held_out", ordinal, form)
        observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "held_out", ordinal, source,
            carrier_kind="plain_text", surface=form, family_ordinal=ordinal,
            sample_role="read_only_probe",
            perturbation_kind="HELD_OUT_DOCUMENT")
        evaluation = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "held_out", ordinal, source, observation,
            _expected(form, morphology_upos if dimension == W02_DEV_DIMENSIONS[2]
                      else "VERB"),
            dimension_name=dimension)
        assert isinstance(evaluation, EvaluatorLabelRecord)
        pairs.append((observation, evaluation))
    return tuple(pairs)


def _artifacts(tmp_path: Path):
    candidate = run_w02_candidate_fixture(
        fixture_root=tmp_path / "candidate", pairs=_training_pairs(),
        run_id=1, requested_workers=2, mode="fresh")
    overlay = run_w02_morphology_overlay_fixture(
        fixture_root=tmp_path / "overlay",
        candidate_artifact_root=candidate.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    return candidate.artifact_path, overlay.artifact_path


def _budget(max_logic_operations: int = 9_000_000) -> V2EvaluatorResourceBudget:
    return V2EvaluatorResourceBudget(
        512, max_logic_operations, 536_870_912, 300_000, 100_000, 4)


def test_public_synthetic_private_core_closes_all_nine_hard_gates(
        tmp_path: Path) -> None:
    candidate, overlay = _artifacts(tmp_path)
    result = evaluate_w02_private_pair_stream(
        candidate, overlay, _private_pairs(), _budget())
    assert result["status"] == "PASS"
    assert len(result["hard_conjunct_results"]) == 9
    assert all(row["status"] == "PASS"
               for row in result["hard_conjunct_results"])
    assert result["family_counts"]["UD_ANNOTATION"] == 5
    assert result["generalized_candidate_count"] > 0


def test_private_core_preserves_real_fail_and_resource_ne(tmp_path: Path) -> None:
    candidate, overlay = _artifacts(tmp_path)
    failed = evaluate_w02_private_pair_stream(
        candidate, overlay, _private_pairs(morphology_upos="AUX"), _budget())
    morphology = failed["dimension_results"][2]
    assert failed["status"] == "FAIL"
    assert morphology["status"] == "FAIL" and morphology["failed"] == 1

    stopped = evaluate_w02_private_pair_stream(
        candidate, overlay, _private_pairs(), _budget(1))
    assert stopped["status"] == "NE"
    assert stopped["support_results"][0]["status"] == "NE"
    assert stopped["evaluation_count"] == 0
    assert stopped["input_pair_count"] == 5


def test_family_freeze_is_metadata_only_and_binds_shadow_pass() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = build_w02_morphology_successor_private_family_freeze(repository)
    assert value["status"] == "W02_BLIND_PRIVATE_FAMILY_REGISTERED_AND_FROZEN"
    assert value["formal_private_evaluation_runs"] == 0
    assert value["private_payload_reads"] == 0
    assert value["private_family_registered"] == 1
    assert len(value["private_input_files"]) == 7
    assert value["private_file_counts"]["PRIVATE_SOURCE"] == 12_012
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_PRIVATE_FAMILY_CODE_PATHS)
    assert value["registration"]["policy"]["ne_policy"] == "BLOCK"
    assert w02_private_registration_from_freeze(value).to_dict() == (
        value["registration"])


def test_private_reader_closes_content_and_transport_identity(tmp_path: Path) -> None:
    root = tmp_path / "private-evaluator"
    target = root / "observations" / "held_out.jsonl.gz"
    target.parent.mkdir(parents=True)
    observation, _ = _private_pairs()[0]
    line = canonical_json_bytes(observation.to_dict()) + b"\n"
    with target.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(line)
    transport = target.read_bytes()
    identity = W02FileFreeze(
        "PRIVATE_HELD_OUT_OBSERVATION", "PRIVATE_EVALUATOR_ROOT",
        "observation", "held_out", 1, len(line),
        hashlib.sha256(line).hexdigest(), len(transport),
        hashlib.sha256(transport).hexdigest(),
        observation.stable_key.components, observation.stable_key.components,
        ("CC-BY-SA-4.0",))
    permit = V2AccessPermit(
        "PH2_V2_PRIVATE_EVALUATOR", "PRIVATE_EVALUATOR_ROOT", "W-02",
        "held_out", "observation", target, identity.transport_sha256,
        identity.transport_size_bytes, "a" * 64)
    records = tuple(iter_w02_private_records(identity, permit))
    assert len(records) == 1 and isinstance(records[0], ObservationRecord)

    mutated = bytearray(transport)
    mutated[4] ^= 1
    target.write_bytes(bytes(mutated))
    with pytest.raises(
            W02MorphologySuccessorPrivateEvaluationError,
            match="transport.*漂移"):
        tuple(iter_w02_private_records(identity, permit))


def test_private_guard_is_single_use_without_payload_read(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    freeze = build_w02_morphology_successor_private_family_freeze(repository)
    registration = freeze["registration"]
    assert isinstance(registration, dict)
    root = tmp_path / "private-family"
    target = root / Path(*W02_PRIVATE_GUARD_AVAILABLE.split("/"))
    guard = w02_private_guard_value(
        family_commitment=str(registration["family_commitment"]),
        candidate_binding_sha256=str(freeze["candidate_binding_sha256"]),
        code_freeze_sha256=str(freeze["code_freeze_sha256"]))
    write_immutable_json(guard, target)
    run_identity = "a" * 64
    consume_w02_morphology_successor_private_guard(
        root, expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
        run_identity_sha256=run_identity)
    verify_w02_morphology_successor_private_consumed_guard(
        root, expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
        run_identity_sha256=run_identity)
    with pytest.raises(W02MorphologySuccessorPrivateFamilyError, match="已消费"):
        consume_w02_morphology_successor_private_guard(
            root, expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
            run_identity_sha256=run_identity)
