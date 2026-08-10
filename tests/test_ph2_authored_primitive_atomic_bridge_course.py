"""W-04 到 W-05 CC0 mixed-stage bridge pack 与公开 consumer 集成测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_contract import (
    PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID,
    PRIMITIVE_ATOMIC_BRIDGE_PACK_NAME,
    PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
    PRIMITIVE_ATOMIC_BRIDGE_STAGES,
    read_authored_primitive_atomic_bridge_seeds,
)
from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    compile_authored_primitive_atomic_bridge_course,
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
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_plugin import (
    build_w04_v2_public_capability_plugin,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_preflight import (
    build_w04_v2_public_preflight,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query import (
    run_w04_v2_public_query,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicQuery,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    build_w04_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w04_w05_public_bridge import (
    run_w04_w05_public_bridge_query,
)
from pure_integer_ai.experiments.ph2_w04_w05_public_bridge_contract import (
    W04W05PublicBridgeQuery,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_plugin import (
    build_w05_v2_public_capability_plugin,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_preflight import (
    build_w05_v2_public_preflight,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query import (
    run_w05_v2_public_query,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05V2PublicQuery,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    build_w05_v2_public_evaluation_batch,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MAP_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_primitive_atomic_bridge_map_v1.jsonl.sample")
ATOMIC_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_primitive_atomic_bridge_seed_v1.jsonl.sample")


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
    w04_observations = tuple(
        item for item in observations
        if item.w_stage == "W-04" and item.split == "train")
    w05_observations = tuple(
        item for item in observations
        if item.w_stage == "W-05" and item.split == "train")
    w04_teachers = tuple(
        item for item in teachers if item.visible_from_stage == "W-04")
    w05_teachers = tuple(
        item for item in teachers if item.visible_from_stage == "W-05")
    return (
        build_w04_v2_public_evaluation_batch(W04TrainingPayload(
            sources, w04_observations, w04_teachers)),
        build_w05_v2_public_evaluation_batch(W05TrainingPayload(
            sources, w05_observations, w05_teachers)),
    )


def _bridge_query() -> W04W05PublicBridgeQuery:
    return W04W05PublicBridgeQuery(
        "使得", "暴雨使得河水上涨。", "暴雨使得河水上涨。")


def test_bridge_seed_is_cc0_split_closed_and_predicate_linked() -> None:
    seeds = read_authored_primitive_atomic_bridge_seeds(
        MAP_SAMPLE, ATOMIC_SAMPLE)

    assert len(seeds) == 10
    assert PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID == "CC0-1.0"
    assert PRIMITIVE_ATOMIC_BRIDGE_STAGES == ("W-04", "W-05")
    assert sum(item.primitive.split == "train" for item in seeds) == 6
    assert sum(item.atomic.split == "train" for item in seeds) == 6
    assert all(item.primitive.split == item.atomic.split for item in seeds)
    assert all(
        item.primitive.surface_form == next(
            occurrence.surface_fragment for occurrence in item.atomic.occurrences
            if occurrence.occurrence_id == item.atomic.predicate_occurrence_id)
        for item in seeds
    )
    assert {item.primitive.sample_role for item in seeds} == {
        "support", "refute", "conflict", "supersede"}
    assert {item.primitive.family for item in seeds}.isdisjoint(
        {item.atomic.family for item in seeds})
    restored = next(
        item for item in seeds if item.atomic.sample_role == "supersede")
    assert restored.atomic.supersedes_seed_id == "bridge-causes-omission-v1"
    assert restored.primitive.supersedes_seed_id == "bridge-causes-omission-v1"


def test_bridge_compiler_is_bit_identical_and_keeps_owners_separate(
        tmp_path) -> None:
    first = compile_authored_primitive_atomic_bridge_course(
        MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path / "first")
    second = compile_authored_primitive_atomic_bridge_course(
        MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path / "second")

    assert first.pack_root.name == PRIMITIVE_ATOMIC_BRIDGE_PACK_NAME
    assert first.manifest == second.manifest
    assert first.manifest.source_key == PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY
    assert first.manifest.w_stages == PRIMITIVE_ATOMIC_BRIDGE_STAGES
    assert (first.validation.source_ref_count,
            first.validation.observation_count,
            first.validation.teacher_evidence_count,
            first.validation.evaluator_label_count,
            first.validation.source_cluster_count) == (10, 20, 12, 8, 2)
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


def test_every_w05_observation_requires_same_source_w04_predicate(
        tmp_path) -> None:
    build = compile_authored_primitive_atomic_bridge_course(
        MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path)
    observations = _records(build, RECORD_OBSERVATION)
    by_source_stage = {
        (item.source_ref_key, item.w_stage): item for item in observations}

    assert len(by_source_stage) == 20
    for source_key in {item.source_ref_key for item in observations}:
        primitive = by_source_stage[(source_key, "W-04")]
        atomic = by_source_stage[(source_key, "W-05")]
        primitive_payload = primitive.typed_payload.to_value()
        atomic_payload = atomic.typed_payload.to_value()
        anchor = atomic_payload["candidate_definition"]["source_anchor_key"]
        predicate_occurrence = next(
            item for item in atomic_payload["occurrences"]
            if item["identity_key"] == anchor)

        assert primitive.prerequisite_keys == ()
        assert atomic.prerequisite_keys == (primitive.stable_key,)
        assert primitive.split == atomic.split
        assert primitive.logical_order < atomic.logical_order
        assert primitive_payload["surface_form"] == (
            predicate_occurrence["surface_fragment"])
        assert "expected_state" not in primitive_payload
        assert "expected_payload" not in primitive_payload
        assert "expected_state" not in atomic_payload
        assert "expected_payload" not in atomic_payload

    superseding = tuple(
        item for item in observations if item.supersedes_key is not None)
    by_key = {item.stable_key: item for item in observations}
    assert len(superseding) == 2
    assert {item.w_stage for item in superseding} == {"W-04", "W-05"}
    assert all(by_key[item.supersedes_key].w_stage == item.w_stage
               for item in superseding)


def test_public_stage_queries_share_exact_source_without_labels(
        tmp_path) -> None:
    build = compile_authored_primitive_atomic_bridge_course(
        MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path)
    w04_batch, w05_batch = _public_batches(build)

    assert len(w04_batch.source_records) == len(w04_batch.pairs) == 6
    assert len(w05_batch.source_records) == len(w05_batch.pairs) == 6
    assert "EvaluatorLabelRecord" not in {
        type(item).__name__
        for item in (*w04_batch.records, *w05_batch.records)}
    w04 = run_w04_v2_public_query(
        w04_batch, W04V2PublicQuery("使得", "暴雨使得河水上涨。"))
    w05 = run_w05_v2_public_query(
        w05_batch, W05V2PublicQuery("暴雨使得河水上涨。"))

    assert w04.status == w05.status == "UNIQUE"
    assert (w04.selected_primitive_registry,
            w04.selected_primitive_kind) == ("relation", 4)
    active = next(
        item for item in w05.candidates if item.lifecycle_status == "ACTIVE")
    assert w04.candidates[0].source_ref_key == active.source_record_key
    assert w04.candidates[0].source_commitment == active.source_commitment
    assert w04.reasoning_status == w05.selected_reasoning_status == "AUTHORIZED"
    assert w04.generation_status == w05.generation_status == "READY"


def test_bridge_pack_preserves_both_public_p0_p2_capabilities(
        tmp_path) -> None:
    build = compile_authored_primitive_atomic_bridge_course(
        MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path)
    w04_batch, w05_batch = _public_batches(build)
    w04_plugin = build_w04_v2_public_capability_plugin(REPOSITORY)
    w05_plugin = build_w05_v2_public_capability_plugin(REPOSITORY)
    w04 = build_w04_v2_public_preflight(
        REPOSITORY, w04_batch, w04_plugin)
    w05 = build_w05_v2_public_preflight(
        REPOSITORY, w05_batch, w05_plugin)

    assert (w04.p0.status, w04.p1.status,
            w04.p2.status) == ("PASS", "PASS", "PASS")
    assert (w05.p0.status, w05.p1.status,
            w05.p2.status) == ("PASS", "PASS", "PASS")
    assert w04.outcome.result_set.status == "PASS"
    assert w05.outcome.result_set.status == "PASS"
    assert {item.status for item in w04.outcome.result_set.results} == {"PASS"}
    assert {item.status for item in w05.outcome.result_set.results} == {"PASS"}


def test_explicit_prerequisite_authorizes_bridge_and_is_deterministic(
        tmp_path) -> None:
    build = compile_authored_primitive_atomic_bridge_course(
        MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path)
    w04_batch, w05_batch = _public_batches(build)
    query = _bridge_query()

    first = run_w04_w05_public_bridge_query(w04_batch, w05_batch, query)
    second = run_w04_w05_public_bridge_query(w04_batch, w05_batch, query)

    assert first.status == "BRIDGED"
    assert first.link is not None
    assert first.w04_result.status == first.w05_result.status == "UNIQUE"
    assert (first.link.primitive_registry,
            first.link.primitive_kind) == ("relation", 4)
    assert first.link.predicate_occurrence_key in first.link.occurrence_order
    assert len(first.link.role_binding_keys) == 2
    assert {item.surface for item in first.w04_result.generation_options} == {
        "使得"}
    assert {item.surface for item in first.w05_result.generation_options} == {
        "暴雨使得河水上涨。"}
    assert first.sha256() == second.sha256()


def test_missing_prerequisite_keeps_both_stages_unique_but_bridge_unknown(
        tmp_path) -> None:
    build = compile_authored_primitive_atomic_bridge_course(
        MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path)
    w04_batch, w05_batch = _public_batches(build)
    target = next(
        item.observation for item in w05_batch.pairs
        if item.observation.typed_payload.to_value().get("surface")
        == "暴雨使得河水上涨。"
        and item.observation.perturbation_kind == "OCCURRENCE_RESTORE")
    observations = tuple(
        replace(item.observation, prerequisite_keys=())
        if item.observation.stable_key == target.stable_key
        else item.observation
        for item in w05_batch.pairs
    )
    broken = build_w05_v2_public_evaluation_batch(W05TrainingPayload(
        tuple(item.record for item in w05_batch.source_records),
        observations,
        tuple(item.evidence for item in w05_batch.pairs),
    ))

    result = run_w04_w05_public_bridge_query(
        w04_batch, broken, _bridge_query())

    assert result.status == "UNKNOWN"
    assert result.link is None
    assert result.w04_result.status == result.w05_result.status == "UNIQUE"
