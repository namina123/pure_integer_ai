"""The public observation path compiles to label-free evaluation graphs."""
from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.integer_graph_novelty import (
    integer_graph_topology,
)
from pure_integer_ai.experiments.integer_relation_evaluation_graph import (
    EVALUATION_SPLITS,
    EvaluationGraphAssignment,
    IntegerRelationEvaluationGraphError,
    compile_w06_evaluation_graph_splits,
    read_integer_relation_evaluation_graph_course,
    write_integer_relation_evaluation_graph_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06ParsedRelationGraph,
    W06TypedAdapterError,
    parse_w06_relation_graph_structure,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "ph2_w06_dataset_artifacts" / "source_semantic_overlay_v1" / "packs"


def _public_causes_held_out():
    manifest_path = next(PACK_ROOT.glob(
        "AUTHORED_CC0_V1--CC0-1.0--causes-v1/manifest.json"))
    manifest = read_artifact_manifest(manifest_path)
    sources = []
    observations = []
    for identity in manifest.files:
        records = read_record_artifact(manifest_path.parent, identity)
        if identity.record_kind == RECORD_SOURCE_REF:
            sources.extend(records)
        elif identity.record_kind == RECORD_OBSERVATION:
            observations.extend(
                item for item in records if item.split == "held_out")
    return tuple(sources), tuple(sorted(
        observations, key=lambda item: item.stable_key))


def _public_type_mismatch_observation():
    sources = []
    mismatches = []
    for manifest_path in sorted(PACK_ROOT.glob(
            "AUTHORED_CC0_V1--CC0-1.0--*/manifest.json")):
        manifest = read_artifact_manifest(manifest_path)
        for identity in manifest.files:
            records = read_record_artifact(manifest_path.parent, identity)
            if identity.record_kind == RECORD_SOURCE_REF:
                sources.extend(records)
            elif identity.record_kind == RECORD_OBSERVATION:
                mismatches.extend(
                    item for item in records
                    if item.split == "train"
                    and item.perturbation_kind == "TYPE_MISMATCH")
    if not mismatches:
        raise AssertionError("public authored pack has no TYPE_MISMATCH observation")
    return tuple(sources), mismatches[0]


def _assignments(observations):
    result = {split: [] for split in EVALUATION_SPLITS}
    inputs = {
        "held_out": "这两个事件有什么因果关系？",
        "negative": "现有证据能否支持这个因果关系？",
        "conflict": "这些因果证据是否互相冲突？",
    }
    for ordinal, (split, observation) in enumerate(
            zip(EVALUATION_SPLITS, observations, strict=True), start=1):
        result[split].append(EvaluationGraphAssignment(
            split,
            (99100, ordinal),
            tuple(map(ord, inputs[split])),
            ((99101, ordinal),),
            (tuple(observation.stable_key.components),),
        ))
    return {name: tuple(items) for name, items in result.items()}


def test_public_observation_compiles_and_round_trips_without_labels(tmp_path):
    sources, observations = _public_causes_held_out()
    assert len(observations) >= len(EVALUATION_SPLITS)
    selected = observations[:len(EVALUATION_SPLITS)]
    assignments = _assignments(selected)
    courses = compile_w06_evaluation_graph_splits(
        selected,
        sources,
        assignments,
        owner_identity=(99110, 1),
        source_identity=(99111, 1),
        license_identity=(99112, 1),
    )

    assert tuple(courses) == EVALUATION_SPLITS
    for split in EVALUATION_SPLITS:
        course = courses[split]
        assert len(course.cases) == 1
        assert len(course.records) == 1
        assert not course.rejections
        record = course.records[0]
        serialized = record.to_dict()
        assert not {
            "sample_role", "perturbation_kind", "teacher_expected_state_code",
            "teacher_decision_values", "teacher_expected_payload",
            "evaluator_label", "expected_state", "expected_payload",
        }.intersection(serialized)
        topology = integer_graph_topology((record,))
        assert topology.relation_count == 1

        path = tmp_path / f"{split}.int.json"
        digest = write_integer_relation_evaluation_graph_course(path, course)
        restored = read_integer_relation_evaluation_graph_course(path)
        assert restored == course
        assert len(digest) == 64
        raw = json.loads(path.read_text(encoding="ascii"))
        assert raw["free_dialogue_complete"] == 0
        assert raw["records"][0] == serialized
        assert "teacher_expected_state_code" not in path.read_text(encoding="ascii")


def test_evaluation_graph_cold_reader_rejects_label_injection(tmp_path):
    sources, observations = _public_causes_held_out()
    selected = observations[:len(EVALUATION_SPLITS)]
    courses = compile_w06_evaluation_graph_splits(
        selected,
        sources,
        _assignments(selected),
        owner_identity=(99120, 1),
        source_identity=(99121, 1),
        license_identity=(99122, 1),
    )
    path = tmp_path / "held_out.int.json"
    write_integer_relation_evaluation_graph_course(path, courses["held_out"])
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["records"][0]["teacher_expected_state_code"] = 1
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(IntegerRelationEvaluationGraphError, match="labels"):
        read_integer_relation_evaluation_graph_course(path)


def test_type_mismatch_becomes_label_free_rejection_not_graph_record():
    sources, mismatch = _public_type_mismatch_observation()
    mismatch = replace(mismatch, split="held_out")
    _valid_sources, valid_observations = _public_causes_held_out()
    observations = (mismatch, *valid_observations[:2])
    assignments = {
        "held_out": (EvaluationGraphAssignment(
            "held_out", (99300, 1), (30001,), ((99301, 1),),
            (tuple(valid_observations[0].stable_key.components),)),),
        "negative": (EvaluationGraphAssignment(
            "negative", (99300, 2), (30002,), ((99301, 2),),
            (tuple(valid_observations[1].stable_key.components),)),),
        "conflict": (EvaluationGraphAssignment(
            "conflict", (99300, 3), (30003,), ((99301, 3),),
            (tuple(mismatch.stable_key.components),)),),
    }
    courses = compile_w06_evaluation_graph_splits(
        observations,
        sources,
        assignments,
        owner_identity=(99310, 1),
        source_identity=(99311, 1),
        license_identity=(99312, 1),
    )
    assert len(courses["conflict"].records) == 0
    assert len(courses["conflict"].rejections) == 1
    assert courses["conflict"].rejections[0].observation_key == tuple(
        mismatch.stable_key.components)


def test_evaluation_compiler_does_not_access_observation_labels(monkeypatch):
    sources, observations = _public_causes_held_out()
    selected = observations[:len(EVALUATION_SPLITS)]
    assignments = _assignments(selected)
    get_attribute = ObservationRecord.__getattribute__

    def without_labels(self, name):
        if name in {"sample_role", "perturbation_kind"}:
            raise AssertionError(f"evaluation compiler accessed {name}")
        return get_attribute(self, name)

    monkeypatch.setattr(ObservationRecord, "__getattribute__", without_labels)
    courses = compile_w06_evaluation_graph_splits(
        selected, sources, assignments,
        owner_identity=(99410, 1), source_identity=(99411, 1),
        license_identity=(99412, 1))
    assert sum(len(course.records) for course in courses.values()) == 3


def test_schema_rejection_is_structural_without_type_mismatch_label(monkeypatch):
    sources, mismatch = _public_type_mismatch_observation()
    mismatch = replace(mismatch, sample_role="support", perturbation_kind="NONE")
    source = next(item for item in sources if item.stable_key == mismatch.source_ref_key)
    get_attribute = ObservationRecord.__getattribute__

    def without_labels(self, name):
        if name in {"sample_role", "perturbation_kind"}:
            raise AssertionError(f"structural parser accessed {name}")
        return get_attribute(self, name)

    monkeypatch.setattr(ObservationRecord, "__getattribute__", without_labels)
    parsed = parse_w06_relation_graph_structure(mismatch, source)
    assert isinstance(parsed, W06ParsedRelationGraph)
    assert len(parsed.violations) == 1
    assert not {
        "observation", "source_record", "sample_role", "perturbation_kind",
        "consumer_request", "proposal_payload", "teacher_record",
    }.intersection(field.name for field in fields(parsed))


def test_structural_parser_output_is_independent_of_sample_labels():
    sources, observations = _public_causes_held_out()
    observation = observations[0]
    source = next(item for item in sources if item.stable_key == observation.source_ref_key)
    parsed = parse_w06_relation_graph_structure(observation, source)
    for role, perturbation in (("support", "NONE"), ("refute", "TYPE_MISMATCH"),
                               ("conflict", "NONE")):
        changed = replace(observation, sample_role=role, perturbation_kind=perturbation)
        assert parse_w06_relation_graph_structure(changed, source) == parsed


def test_structural_parser_keeps_source_license_boundary():
    sources, observations = _public_causes_held_out()
    observation = observations[0]
    wrong_source = next(item for item in sources if item.stable_key != observation.source_ref_key)
    with pytest.raises(W06TypedAdapterError, match="source, stage, or license"):
        parse_w06_relation_graph_structure(observation, wrong_source)
