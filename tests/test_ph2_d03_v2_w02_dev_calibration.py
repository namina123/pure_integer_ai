"""PH2-D03-V2 W-02 dev calibration 的公开合成专项。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    run_w02_candidate_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    W02_CARRIER_KINDS,
    _authored_expected,
    _carrier_serialization,
    _dimension_key,
    _observation_record,
    _owner_record,
    _source_record,
    _unicode_expected,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import W02FileFreeze
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_CODE_PATHS,
    W02_DEV_DIMENSIONS,
    W02DevCalibrationError,
    W02DevInputRoot,
    _evaluate_pair,
    build_w02_dev_calibration_freeze,
    iter_w02_dev_records,
    load_w02_dev_candidate_index,
    predict_w02_dev_observation,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    TeacherEvidenceRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(source_key: str, split: str, ordinal: int, surface: str):
    license_id = "CC0-1.0" if source_key == "AUTHORED_CC0" else "CC-BY-SA-4.0"
    return _source_record(
        source_key, split, ordinal,
        snapshot_id=f"fixture-{source_key.casefold()}-{split}",
        revision_id="1",
        official_url="https://example.invalid/public-fixture",
        source_identity=f"fixture:{source_key}:{split}:{ordinal}",
        upstream_checksum="sha256:" + _sha(f"upstream:{source_key}:{split}:{ordinal}"),
        local_sha256=_sha(f"local:{source_key}:{split}:{ordinal}"),
        license_id=license_id,
        attribution="public synthetic fixture",
        locator_kind="record",
        locator_value=str(ordinal),
        span_end=len(surface),
    )


def _training_pairs():
    pairs = []
    units = ("qz", "灥", "z1")
    surface = "".join(units)
    source = _source("AUTHORED_CC0", "train", 1, surface)
    for ordinal, carrier_kind in enumerate(W02_CARRIER_KINDS, start=1):
        observation = _observation_record(
            "AUTHORED_CC0", "train", ordinal, source,
            carrier_kind=carrier_kind,
            surface=surface,
            family_ordinal=1,
            sample_role="support",
            perturbation_kind="NONE",
        )
        raw, start, end, _, _ = _carrier_serialization(carrier_kind, surface)
        evidence = _owner_record(
            "AUTHORED_CC0", "train", ordinal, source, observation,
            _authored_expected(units, carrier_kind, raw, start, end),
            dimension_name=W02_DEV_DIMENSIONS[3],
        )
        assert isinstance(evidence, TeacherEvidenceRecord)
        pairs.append((observation, evidence))
    ud_surface = "我 爱 AI"
    ud_source = _source("UD_ZH_GSDSIMP_R2_18", "train", 1, ud_surface)
    ud_observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "train", 1, ud_source,
        carrier_kind="plain_text",
        surface=ud_surface,
        family_ordinal=1,
        sample_role="support",
        perturbation_kind="NONE",
    )
    ud_expected = {
        "boundary_spans": [
            {"end": 1, "form": "我", "start": 0},
            {"end": 3, "form": "爱", "start": 2},
            {"end": 6, "form": "AI", "start": 4},
        ],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [
            {"feats": [], "form": "我", "lemma": "我", "node_id": [1],
             "upos": "PRON"},
            {"feats": [], "form": "爱", "lemma": "爱", "node_id": [2],
             "upos": "VERB"},
            {"feats": [], "form": "AI", "lemma": "AI", "node_id": [3],
             "upos": "PROPN"},
        ],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }
    ud_evidence = _owner_record(
        "UD_ZH_GSDSIMP_R2_18", "train", 1, ud_source, ud_observation,
        ud_expected, dimension_name=W02_DEV_DIMENSIONS[2])
    assert isinstance(ud_evidence, TeacherEvidenceRecord)
    pairs.append((ud_observation, ud_evidence))
    unicode_surface = "甲A\u0301"
    unicode_source = _source(
        "ZHWIKTIONARY_20260701", "train", 1, unicode_surface)
    unicode_observation = _observation_record(
        "ZHWIKTIONARY_20260701", "train", 1, unicode_source,
        carrier_kind="plain_text",
        surface=unicode_surface,
        family_ordinal=1,
        sample_role="support",
        perturbation_kind="NONE",
    )
    unicode_expected = _unicode_expected(unicode_surface, "plain_text")
    unicode_expected["page_title_sha256"] = _sha(unicode_surface)
    unicode_evidence = _owner_record(
        "ZHWIKTIONARY_20260701", "train", 1, unicode_source,
        unicode_observation, unicode_expected,
        dimension_name=W02_DEV_DIMENSIONS[0])
    assert isinstance(unicode_evidence, TeacherEvidenceRecord)
    pairs.append((unicode_observation, unicode_evidence))
    return tuple(pairs)


def _dev_authored(dimension: str):
    units = ("rx", "龘", "z2")
    surface = "".join(units)
    source = _source("AUTHORED_CC0", "dev", 1, surface)
    observation = _observation_record(
        "AUTHORED_CC0", "dev", 1, source,
        carrier_kind="markdown",
        surface=surface,
        family_ordinal=1,
        sample_role="read_only_probe",
        perturbation_kind="OOV_UNSEEN_FAMILY",
    )
    raw, start, end, _, _ = _carrier_serialization("markdown", surface)
    evaluation = _owner_record(
        "AUTHORED_CC0", "dev", 1, source, observation,
        _authored_expected(units, "markdown", raw, start, end),
        dimension_name=dimension,
    )
    assert isinstance(evaluation, EvaluatorLabelRecord)
    return observation, evaluation


def test_batch_predictor_equals_frozen_predictor(tmp_path: Path) -> None:
    result = run_w02_candidate_fixture(
        fixture_root=tmp_path / "candidate",
        pairs=_training_pairs(),
        run_id=1,
        requested_workers=2,
        mode="fresh",
    )
    observation, _ = _dev_authored(W02_DEV_DIMENSIONS[4])
    before = hashlib.sha256(
        (result.artifact_path / "candidate.sqlite3").read_bytes()).hexdigest()
    with open_w02_candidate_predictor(result.artifact_path) as predictor:
        direct = predictor.predict(observation)
        index = load_w02_dev_candidate_index(predictor)
        batch, operations = predict_w02_dev_observation(index, observation)
    after = hashlib.sha256(
        (result.artifact_path / "candidate.sqlite3").read_bytes()).hexdigest()
    assert batch.to_dict() == direct.to_dict()
    assert operations > 0 and index.row_count > 0
    assert before == after


def test_dev_dimension_reports_real_pass_and_failure(tmp_path: Path) -> None:
    result = run_w02_candidate_fixture(
        fixture_root=tmp_path / "candidate",
        pairs=_training_pairs(),
        run_id=1,
        requested_workers=1,
        mode="fresh",
    )
    dimension_by_key = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    with open_w02_candidate_predictor(result.artifact_path) as predictor:
        index = load_w02_dev_candidate_index(predictor)
        authored, authored_evaluation = _dev_authored(W02_DEV_DIMENSIONS[4])
        authored_prediction, _ = predict_w02_dev_observation(index, authored)
        _, family, passed, _ = _evaluate_pair(
            authored, authored_evaluation, authored_prediction, dimension_by_key)
        assert family == "AUTHORED_OOV" and passed is True

        surface = "未登录词"
        source = _source("UD_ZH_GSDSIMP_R2_18", "dev", 1, surface)
        observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "dev", 1, source,
            carrier_kind="plain_text", surface=surface, family_ordinal=1,
            sample_role="read_only_probe", perturbation_kind="HELD_OUT_DOCUMENT")
        expected = {
            "boundary_spans": [{"end": len(surface), "form": surface, "start": 0}],
            "carrier_kind": "plain_text",
            "definitive_truth_authoritative": 0,
            "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
            "morphology": [{"feats": [], "form": surface, "lemma": surface,
                            "node_id": [1], "upos": "NOUN"}],
            "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
        }
        evaluation = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "dev", 1, source, observation, expected,
            dimension_name=W02_DEV_DIMENSIONS[2])
        assert isinstance(evaluation, EvaluatorLabelRecord)
        prediction, _ = predict_w02_dev_observation(index, observation)
        _, family, passed, _ = _evaluate_pair(
            observation, evaluation, prediction, dimension_by_key)
        assert family == "UD_ANNOTATION" and passed is False


def test_dev_transport_reader_rejects_drift(tmp_path: Path) -> None:
    dev_root = tmp_path / "dev-calibration"
    target = dev_root / "observations" / "dev.jsonl.gz"
    target.parent.mkdir(parents=True)
    observation, _ = _dev_authored(W02_DEV_DIMENSIONS[0])
    line = canonical_json_bytes(observation.to_dict()) + b"\n"
    with target.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(line)
    transport = target.read_bytes()
    identity = W02FileFreeze(
        "DEV_OBSERVATION", "DEV_CALIBRATION_ROOT", "observation", "dev", 1,
        len(line), hashlib.sha256(line).hexdigest(), len(transport),
        hashlib.sha256(transport).hexdigest(), observation.stable_key.components,
        observation.stable_key.components, ("CC0-1.0",),
    )
    freeze = SimpleNamespace(files=(identity,))
    records = tuple(iter_w02_dev_records(
        freeze, W02DevInputRoot(dev_root), "DEV_OBSERVATION"))
    assert len(records) == 1 and isinstance(records[0], ObservationRecord)
    target.write_bytes(transport + b"x")
    with pytest.raises(W02DevCalibrationError, match="transport identity"):
        tuple(iter_w02_dev_records(
            freeze, W02DevInputRoot(dev_root), "DEV_OBSERVATION"))


def test_dev_freeze_binds_current_public_candidate() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    value = build_w02_dev_calibration_freeze(repository_root)
    assert value["status"] == "W02_DEV_CALIBRATION_FREEZE_COMPLETE"
    assert value["formal_training_runs"] == 1
    assert value["formal_private_evaluation_runs"] == 0
    assert value["private_payload_reads"] == 0
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_DEV_CODE_PATHS)
