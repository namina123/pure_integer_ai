"""CC0 mixed-stage bridge pack and public consumer integration tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_contract import (
    BRIDGE_LICENSE_ID,
    BRIDGE_PACK_NAME,
    BRIDGE_SOURCE_KEY,
    BRIDGE_STAGES,
    read_authored_semantic_primitive_bridge_seeds,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    compile_authored_semantic_primitive_bridge_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_w04_public_bridge import (
    run_w03_w04_public_bridge_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_public_bridge_contract import (
    W03W04PublicBridgeQuery,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_query import (
    W03V2PublicQuery,
    run_w03_v2_public_query,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    build_w03_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_query import (
    run_w04_v2_public_query,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicQuery,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    build_w04_v2_public_evaluation_batch,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_semantic_primitive_bridge_seed_v1.jsonl.sample")


def _records(build, kind: str) -> tuple[object, ...]:
    values = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            values.extend(read_record_artifact(build.pack_root, identity))
    return tuple(values)


def _public_batches(build):
    sources = _records(build, RECORD_SOURCE_REF)
    observations = _records(build, RECORD_OBSERVATION)
    teachers = _records(build, RECORD_TEACHER_EVIDENCE)
    w03_observations = tuple(
        item for item in observations
        if item.w_stage == "W-03" and item.split == "train")
    w04_observations = tuple(
        item for item in observations
        if item.w_stage == "W-04" and item.split == "train")
    w03_teachers = tuple(
        item for item in teachers if item.visible_from_stage == "W-03")
    w04_teachers = tuple(
        item for item in teachers if item.visible_from_stage == "W-04")
    return (
        build_w03_v2_public_evaluation_batch(W03TrainingPayload(
            sources, w03_observations, w03_teachers)),
        build_w04_v2_public_evaluation_batch(W04TrainingPayload(
            sources, w04_observations, w04_teachers)),
    )


def test_bridge_seed_is_cc0_owner_isolated_and_explicitly_cross_stage() -> None:
    seeds = read_authored_semantic_primitive_bridge_seeds(SAMPLE)

    assert len(seeds) == 7
    assert BRIDGE_LICENSE_ID == "CC0-1.0"
    assert BRIDGE_STAGES == ("W-03", "W-04")
    assert {item.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert {item.family for item in seeds if item.label_owner == "teacher"}.isdisjoint(
        {item.family for item in seeds if item.label_owner == "evaluator"})
    assert {
        item.template_family for item in seeds if item.label_owner == "teacher"
    }.isdisjoint({
        item.template_family for item in seeds if item.label_owner == "evaluator"
    })
    superseder = next(item for item in seeds if item.sample_role == "supersede")
    assert superseder.supersedes_bridge_id == "teacher-causes-v1"
    assert (superseder.candidate_sense, superseder.primitive_registry,
            superseder.primitive_kind) == (
                "causal_relation_cue", "relation", 4)


def test_bridge_compiler_is_bit_identical_and_keeps_labels_separate(
        tmp_path) -> None:
    first = compile_authored_semantic_primitive_bridge_course(
        SAMPLE, tmp_path / "first")
    second = compile_authored_semantic_primitive_bridge_course(
        SAMPLE, tmp_path / "second")

    assert first.pack_root.name == BRIDGE_PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == BRIDGE_SOURCE_KEY
    assert first.manifest.w_stages == BRIDGE_STAGES
    assert first.validation.source_ref_count == 7
    assert first.validation.observation_count == 14
    assert first.validation.teacher_evidence_count == 8
    assert first.validation.evaluator_label_count == 6
    assert first.validation.source_cluster_count == 2
    for left, right in zip(
            first.manifest.files, second.manifest.files, strict=True):
        assert left == right
        assert (first.pack_root / left.relative_path).read_bytes() == (
            second.pack_root / right.relative_path).read_bytes()

    assert all(isinstance(item, SourceRefRecord)
               for item in _records(first, RECORD_SOURCE_REF))
    assert all(isinstance(item, ObservationRecord)
               for item in _records(first, RECORD_OBSERVATION))
    assert all(isinstance(item, TeacherEvidenceRecord)
               for item in _records(first, RECORD_TEACHER_EVIDENCE))
    assert all(isinstance(item, EvaluatorLabelRecord)
               for item in _records(first, RECORD_EVALUATOR_LABEL))


def test_every_w04_observation_explicitly_requires_same_source_w03_record(
        tmp_path) -> None:
    build = compile_authored_semantic_primitive_bridge_course(SAMPLE, tmp_path)
    observations = _records(build, RECORD_OBSERVATION)
    by_source_stage = {
        (item.source_ref_key, item.w_stage): item for item in observations}

    assert len(by_source_stage) == 14
    for source_key in {item.source_ref_key for item in observations}:
        sense = by_source_stage[(source_key, "W-03")]
        primitive = by_source_stage[(source_key, "W-04")]
        assert primitive.prerequisite_keys == (sense.stable_key,)
        assert sense.prerequisite_keys == ()
        assert sense.logical_order < primitive.logical_order
        assert "expected_state" not in sense.typed_payload.to_value()
        assert "expected_payload" not in sense.typed_payload.to_value()
        assert "expected_state" not in primitive.typed_payload.to_value()
        assert "expected_payload" not in primitive.typed_payload.to_value()

    superseding = tuple(
        item for item in observations if item.supersedes_key is not None)
    by_key = {item.stable_key: item for item in observations}
    assert len(superseding) == 2
    assert {item.w_stage for item in superseding} == {"W-03", "W-04"}
    assert all(by_key[item.supersedes_key].w_stage == item.w_stage
               for item in superseding)
    assert all(by_key[item.supersedes_key].logical_order < item.logical_order
               for item in superseding)


def test_public_w03_and_w04_consumers_share_linked_source_without_labels(
        tmp_path) -> None:
    build = compile_authored_semantic_primitive_bridge_course(SAMPLE, tmp_path)
    w03_batch, w04_batch = _public_batches(build)

    assert len(w03_batch.source_records) == len(w03_batch.pairs) == 4
    assert len(w04_batch.source_records) == len(w04_batch.pairs) == 4
    assert "EvaluatorLabelRecord" not in {
        type(item).__name__ for item in (*w03_batch.records, *w04_batch.records)}
    w03 = run_w03_v2_public_query(
        w03_batch, W03V2PublicQuery("使得", "暴雨使得河水上涨。"))
    w04 = run_w04_v2_public_query(
        w04_batch, W04V2PublicQuery("使得", "暴雨使得河水上涨。"))

    assert w03.status == w04.status == "UNIQUE"
    assert (w04.selected_primitive_registry,
            w04.selected_primitive_kind) == ("relation", 4)
    assert w03.candidates[0].source_ref_key == w04.candidates[0].source_ref_key
    assert w03.candidates[0].source_commitment == (
        w04.candidates[0].source_commitment)


def test_explicit_prerequisite_authorizes_typed_bridge_and_is_deterministic(
        tmp_path) -> None:
    build = compile_authored_semantic_primitive_bridge_course(SAMPLE, tmp_path)
    w03_batch, w04_batch = _public_batches(build)
    query = W03W04PublicBridgeQuery("使得", "暴雨使得河水上涨。")

    first = run_w03_w04_public_bridge_query(w03_batch, w04_batch, query)
    second = run_w03_w04_public_bridge_query(w03_batch, w04_batch, query)

    assert first.status == "BRIDGED"
    assert first.link is not None
    assert first.w03_result.status == first.w04_result.status == "UNIQUE"
    assert (first.link.primitive_registry,
            first.link.primitive_kind) == ("relation", 4)
    assert first.link.source_ref_key == first.w03_result.candidates[0].source_ref_key
    assert first.link.source_ref_key == first.w04_result.candidates[0].source_ref_key
    assert {item.surface for item in first.w04_result.generation_options} == {
        "使得"}
    assert first.sha256() == second.sha256()


def test_missing_prerequisite_fails_closed_as_unknown(tmp_path) -> None:
    build = compile_authored_semantic_primitive_bridge_course(SAMPLE, tmp_path)
    w03_batch, w04_batch = _public_batches(build)
    target = next(
        item.observation for item in w04_batch.pairs
        if item.observation.typed_payload.to_value().get("surface_form") == "使得")
    broken_observations = tuple(
        replace(item.observation, prerequisite_keys=())
        if item.observation.stable_key == target.stable_key
        else item.observation
        for item in w04_batch.pairs
    )
    broken = build_w04_v2_public_evaluation_batch(W04TrainingPayload(
        tuple(item.record for item in w04_batch.source_records),
        broken_observations,
        tuple(item.evidence for item in w04_batch.pairs),
    ))

    result = run_w03_w04_public_bridge_query(
        w03_batch,
        broken,
        W03W04PublicBridgeQuery("使得", "暴雨使得河水上涨。"),
    )

    assert result.status == "UNKNOWN"
    assert result.link is None
    assert result.w03_result.status == result.w04_result.status == "UNIQUE"
