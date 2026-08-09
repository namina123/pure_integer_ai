"""Public fail-closed tests for the R3 private evaluator revision."""
from __future__ import annotations

from dataclasses import replace
import hashlib

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _owner_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
    _hash_value,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r3 import (
    _apply_label_fail_closed,
    _prepare_label_stream,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair() -> tuple[object, EvaluatorLabelRecord]:
    surface = "新化"
    source = _source_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1,
        snapshot_id="r3-public-fixture",
        revision_id="r3-public-revision",
        official_url=(
            "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp"),
        source_identity="r3-public-fixture:1",
        upstream_checksum="sha256:" + _sha("upstream"),
        local_sha256=_sha("local"),
        license_id="CC-BY-SA-4.0",
        attribution="public synthetic fixture",
        locator_kind="record",
        locator_value="1",
        span_end=len(surface),
    )
    observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, source,
        carrier_kind="plain_text", surface=surface, family_ordinal=1,
        sample_role="read_only_probe",
        perturbation_kind="HELD_OUT_DOCUMENT",
    )
    expected = {
        "boundary_spans": [{"end": len(surface), "form": surface, "start": 0}],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [{
            "feats": [], "form": surface, "lemma": "新",
            "node_id": [1], "upos": "VERB",
        }],
        "source_annotation": "PUBLIC_SYNTHETIC_FIXTURE",
    }
    evaluation = _owner_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, source, observation,
        expected, dimension_name=W02_DEV_DIMENSIONS[0])
    assert isinstance(evaluation, EvaluatorLabelRecord)
    return observation, evaluation


def _pass_row(name: str, denominator: int = 1) -> dict[str, object]:
    return {
        "denominator": denominator,
        "dimension_key": name,
        "evidence_sha256": _hash_value([name]),
        "failed": 0,
        "ne": 0,
        "numerator": denominator,
        "status": "PASS",
    }


def _core() -> dict[str, object]:
    dimensions = [_pass_row(name) for name in W02_DEV_DIMENSIONS]
    support = [_pass_row(name) for name in (
        "W-02-V2-RESOURCE",
        "W-02-V2-ROLLBACK",
        "W-02-V2-ZERO-CALL-WINDOWS",
        "W-02-V2-V06-CLONE",
    )]
    return {
        "dimension_results": dimensions,
        "family_counts": {
            "AUTHORED_OOV": 0,
            "UD_ANNOTATION": 1,
            "UNICODE_ANNOTATION": 0,
        },
        "hard_conjunct_results": [*dimensions, *support],
        "status": "PASS",
        "support_results": support,
    }


def test_r3_valid_label_remains_unsanitized() -> None:
    observation, evaluation = _pair()
    prepared, audit = _prepare_label_stream(((observation, evaluation),))
    result = _apply_label_fail_closed(_core(), audit)

    assert prepared == ((observation, evaluation),)
    assert audit["sanitization_count"] == 0
    assert result["status"] == "PASS"
    assert result["unregistered_label_result"]["status"] == "PASS"
    assert result["label_sanitization_count"] == 0


def test_r3_unknown_dimension_is_fully_accounted_as_ne() -> None:
    observation, evaluation = _pair()
    unknown = replace(
        evaluation,
        dimension_key=StableRecordKey((99, 99, 99)),
    )
    prepared, audit = _prepare_label_stream(((observation, unknown),))
    result = _apply_label_fail_closed(_core(), audit)

    assert len(prepared) == 1
    assert prepared[0][1].dimension_key != unknown.dimension_key
    assert audit["unknown_dimension_key_count"] == 1
    assert result["status"] == "NE"
    assert result["unregistered_label_result"]["denominator"] == 1
    assert result["unregistered_label_result"]["ne"] == 1
    assert len(result["hard_conjunct_results"]) == 9


def test_r3_unknown_state_marks_registered_dimension_ne_without_abort() -> None:
    observation, evaluation = _pair()
    unknown = replace(evaluation, expected_state="FALSE")
    prepared, audit = _prepare_label_stream(((observation, unknown),))
    result = _apply_label_fail_closed(_core(), audit)

    assert len(prepared) == 1
    assert prepared[0][1].expected_state == "TRUE"
    assert audit["unknown_expected_state_count"] == 1
    assert result["status"] == "NE"
    assert result["dimension_results"][0]["denominator"] == 1
    assert result["dimension_results"][0]["ne"] == 1
    assert result["dimension_results"][0]["status"] == "NE"
