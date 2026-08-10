"""Experimental public W-05 source, plugin and P0-P2 capability tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_atomic_course import (
    compile_authored_atomic_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_plugin import (
    W05_V2_PUBLIC_EXPERIMENTAL,
    W05_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
    W05_V2_PUBLIC_W05_STARTED,
    build_w05_v2_public_capability_plugin,
    w05_v2_public_plugin_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_preflight import (
    build_w05_v2_public_preflight,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query import (
    run_w05_v2_public_queries,
    run_w05_v2_public_query,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN,
    W05V2PublicQuery,
    W05V2PublicQueryError,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicSourceError,
    build_w05_v2_public_evaluation_batch,
    build_w05_v2_public_run_context,
    w05_v2_public_training_payload,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _records(build, kind: str) -> tuple[object, ...]:
    values = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            values.extend(read_record_artifact(build.pack_root, identity))
    return tuple(values)


@pytest.fixture(scope="module")
def public_payload(tmp_path_factory) -> W05TrainingPayload:
    root = tmp_path_factory.mktemp("w05-v2-public")
    build = compile_authored_atomic_course(
        REPOSITORY / "data/ph2/authored_atomic_seed_v1.jsonl.sample",
        root,
    )
    observations = tuple(
        item for item in _records(build, RECORD_OBSERVATION)
        if item.split == "train"
    )
    observation_keys = {item.stable_key for item in observations}
    evidence = tuple(
        item for item in _records(build, RECORD_TEACHER_EVIDENCE)
        if item.observation_key in observation_keys
    )
    return W05TrainingPayload(
        _records(build, RECORD_SOURCE_REF), observations, evidence)


@pytest.fixture(scope="module")
def public_preflight(public_payload):
    batch = build_w05_v2_public_evaluation_batch(public_payload)
    plugin = build_w05_v2_public_capability_plugin(REPOSITORY)
    return batch, plugin, build_w05_v2_public_preflight(
        REPOSITORY, batch, plugin)


@pytest.fixture(scope="module")
def public_query_results(public_payload):
    batch = build_w05_v2_public_evaluation_batch(public_payload)
    queries = {
        "support": W05V2PublicQuery("小猫追逐小鸟。"),
        "scope": W05V2PublicQuery("老师说学生完成作业。"),
        "restore": W05V2PublicQuery("小鸟停在树枝上。"),
        "unknown": W05V2PublicQuery("未学习的来源化命题。"),
    }
    order = ("support", "scope", "restore", "unknown", "support")
    projected = run_w05_v2_public_queries(
        batch, tuple(queries[name] for name in order))
    results = dict(zip(order[:4], projected[:4], strict=True))
    results["support_repeat"] = projected[4]
    refuted = next(
        item for item in results["support"].candidates
        if item.lifecycle_status == "REFUTED")
    results["source_filtered_refuted"] = run_w05_v2_public_query(
        batch,
        W05V2PublicQuery(
            "小猫追逐小鸟。", source_ref_key=refuted.source_ref_key),
    )
    results["generation_disabled"] = run_w05_v2_public_query(
        batch,
        W05V2PublicQuery("小猫追逐小鸟。", allow_generation=0),
    )
    return batch, results


def test_public_source_is_train_only_source_first_and_label_free(
        public_payload) -> None:
    """Only six referenced CC0 train SourceRefs and teacher pairs enter."""
    batch = build_w05_v2_public_evaluation_batch(public_payload)

    assert len(public_payload.source_refs) == 10
    assert len(batch.source_records) == len(batch.pairs) == 6
    assert batch.source_binding.record_count == 6
    assert batch.records == (*batch.source_records, *batch.pairs)
    assert all(item.record.redistribution_policy == "PUBLIC"
               for item in batch.source_records)
    assert all(item.observation.split == "train" for item in batch.pairs)
    assert all(item.evidence.observation_key == item.observation.stable_key
               for item in batch.pairs)
    assert "EvaluatorLabelRecord" not in {
        type(item).__name__ for item in batch.records}
    assert w05_v2_public_training_payload(batch) == W05TrainingPayload(
        tuple(item.record for item in batch.source_records),
        tuple(item.observation for item in batch.pairs),
        tuple(item.evidence for item in batch.pairs),
    )


def test_public_plugin_closes_all_nine_p0_p2_conjuncts(
        public_preflight) -> None:
    """Current W-05 runtime learns and consumes exact public structures."""
    batch, plugin, preflight = public_preflight
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-05")

    assert (preflight.p0.status, preflight.p1.status,
            preflight.p2.status) == ("PASS", "PASS", "PASS")
    assert tuple(
        item.result_key for item in preflight.outcome.result_set.results
    ) == policy.hard_conjunct_keys
    assert {item.status for item in preflight.outcome.result_set.results} == {
        "PASS"}
    assert tuple(
        item.role for item in preflight.outcome.result_set.results
    ) == (*("BEARING" for _ in range(4)), "GENERATION",
          *("SUPPORT" for _ in range(4)))
    assert preflight.outcome.result_set.status == "PASS"

    audit = preflight.outcome.run_audit
    assert (audit.source_ref_count, audit.pair_count,
            audit.private_record_reads, audit.private_payload_gets) == (
                6, 6, 18, 18)
    assert audit.transport_bytes_read == batch.transport_bytes
    assert 0 < audit.logic_operations <= 100_000
    assert audit.zero_call_window_count == 3
    assert audit.write_account is not None and audit.write_account.is_zero
    assert (preflight.experimental, preflight.formal_mastery_claim,
            preflight.w05_started) == (1, 0, 0)
    assert (W05_V2_PUBLIC_EXPERIMENTAL,
            W05_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
            W05_V2_PUBLIC_W05_STARTED) == (1, 0, 0)
    assert plugin.declaration.semantic_sha256 == (
        w05_v2_public_plugin_semantic_sha256(REPOSITORY))


def test_public_plugin_reports_ne_when_required_structures_are_absent(
        public_payload) -> None:
    """One supported proposition cannot stand in for role/scope/lifecycle."""
    observation = next(
        item for item in public_payload.observations
        if item.perturbation_kind == "NONE")
    evidence = next(
        item for item in public_payload.teacher_evidence
        if item.observation_key == observation.stable_key)
    payload = W05TrainingPayload(
        public_payload.source_refs,
        (observation,),
        (evidence,),
    )
    batch = build_w05_v2_public_evaluation_batch(payload)
    plugin = build_w05_v2_public_capability_plugin(REPOSITORY)
    outcome = plugin.evaluate(
        build_w05_v2_public_run_context(batch, plugin.declaration),
        batch.records,
    )
    by_key = {
        item.result_key: item for item in outcome.result_set.results}

    assert by_key["W-05-V2-OCCURRENCE-IDENTITY"].status == "NE"
    assert by_key["W-05-V2-PROPOSITION-CONSUMER"].status == "PASS"
    assert by_key["W-05-V2-ROLE-SWAP"].status == "NE"
    assert by_key["W-05-V2-SCOPE"].status == "NE"
    assert by_key["W-05-V2-GENERATION-HARD-CONJUNCT"].status == "PASS"
    assert by_key["W-05-V2-ROLLBACK"].status == "PASS"
    assert by_key["W-05-V2-V06-CLONE"].status == "PASS"
    assert outcome.result_set.status == "NE"


def test_public_source_rejects_local_only_and_non_train_records(
        public_payload) -> None:
    """P0 fails closed before W-05 capability code sees local/held-out data."""
    referenced = public_payload.observations[0].source_ref_key
    sources = tuple(
        replace(item, redistribution_policy="LOCAL_ONLY")
        if item.stable_key == referenced else item
        for item in public_payload.source_refs
    )
    with pytest.raises(W05V2PublicSourceError, match="nonpublic"):
        build_w05_v2_public_evaluation_batch(replace(
            public_payload, source_refs=sources))

    held_out = replace(public_payload.observations[0], split="held_out")
    with pytest.raises(W05V2PublicSourceError, match="train-only"):
        build_w05_v2_public_evaluation_batch(replace(
            public_payload,
            observations=(held_out, *public_payload.observations[1:])))


def test_public_plugin_has_no_private_family_or_formal_guard_route() -> None:
    """The W-05 consumer cannot import private evaluator/family machinery."""
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/ph2_w05_v2_public_plugin.py"
    )).read_text(encoding="utf-8")

    assert "ph2_w05_evaluator" not in source
    assert "ph2_w05_evaluator_family" not in source
    assert "ph2_w05_evaluator_runtime" not in source
    assert "ph2_w05_firewall" not in source
    assert "w02_artifacts" not in source
    assert "run_evaluation_family_once" not in source
    assert "consume_guard" not in source

    query_source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/ph2_w05_v2_public_query.py"
    )).read_text(encoding="utf-8")
    assert "ph2_w05_evaluator" not in query_source
    assert "ph2_w05_evaluator_family" not in query_source
    assert "ph2_w05_evaluator_runtime" not in query_source
    assert "ph2_w05_firewall" not in query_source
    assert "run_evaluation_family_once" not in query_source
    assert "consume_guard" not in query_source


def test_public_query_projects_exact_source_bound_proposition_structure(
        public_query_results) -> None:
    """Active support is unique while same-surface ROLE_SWAP stays refuted."""
    batch, results = public_query_results
    result = results["support"]

    assert result.status == "UNIQUE"
    assert result.selected_reasoning_status == "AUTHORIZED"
    assert result.generation_status == "READY"
    assert len(result.candidates) == 2
    active = next(
        item for item in result.candidates if item.lifecycle_status == "ACTIVE")
    refuted = next(
        item for item in result.candidates if item.lifecycle_status == "REFUTED")
    assert active.proposition_key == result.selected_proposition_key
    assert (active.active, active.superseded) == (1, 0)
    assert (active.understanding_status,
            active.reasoning_status) == ("UNIQUE", "AUTHORIZED")
    assert len(active.occurrences) == len(active.occurrence_order) == 3
    assert len(active.role_bindings) == 2
    assert active.source_anchor_key in active.occurrence_order
    assert all(item.surface_fragment for item in active.occurrences)
    assert all(item.identity_key for item in active.role_bindings)
    assert (refuted.active, refuted.superseded) == (0, 0)
    assert (refuted.understanding_status,
            refuted.reasoning_status) == ("UNKNOWN", "REJECTED")
    assert active.source_ref_key != refuted.source_ref_key
    assert active.proposition_key != refuted.proposition_key
    assert result.source_binding_sha256 == batch.source_binding.sha256()
    assert result.record_commitment == batch.record_commitment
    assert (result.experimental, result.formal_mastery_claim,
            result.w05_started) == (1, 0, 0)

    assert result.generation_options
    assert {item.surface for item in result.generation_options} == {
        "小猫追逐小鸟。"}
    assert all(
        item.target_proposition_key == active.proposition_key
        and item.target_source_ref_key == active.source_ref_key
        and item.target_source_commitment == active.source_commitment
        for item in result.generation_options
    )


def test_public_query_keeps_conflict_superseded_and_unknown_non_authorized(
        public_query_results) -> None:
    """Scope conflict, omission replacement and absent data never reactivate."""
    _, results = public_query_results
    scope = results["scope"]
    restore = results["restore"]
    unknown = results["unknown"]
    source_filtered = results["source_filtered_refuted"]

    assert scope.status == "CONFLICT"
    assert len(scope.candidates) == 1
    assert scope.candidates[0].lifecycle_status == "CONFLICT"
    assert (scope.candidates[0].understanding_status,
            scope.candidates[0].reasoning_status) == ("CONFLICT", "CONFLICT")

    assert restore.status == "UNIQUE"
    assert {item.lifecycle_status for item in restore.candidates} == {
        "ACTIVE", "SUPERSEDED"}
    superseded = next(
        item for item in restore.candidates
        if item.lifecycle_status == "SUPERSEDED")
    assert (superseded.active, superseded.superseded) == (0, 1)
    assert superseded.reasoning_status == "SUPERSEDED"

    assert unknown.status == "UNKNOWN"
    assert unknown.candidates == ()
    assert source_filtered.status == "UNKNOWN"
    assert len(source_filtered.candidates) == 1
    assert source_filtered.candidates[0].lifecycle_status == "REFUTED"

    for result in (scope, unknown, source_filtered):
        assert result.selected_proposition_key is None
        assert result.selected_reasoning_status == "NOT_RUN"
        assert (result.generation_status
                == W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN)
        assert result.generation_options == ()


def test_public_query_is_label_free_repeatable_and_generation_gated(
        public_query_results) -> None:
    """Repeated projections are identical and no expected answer enters input."""
    _, results = public_query_results
    disabled = results["generation_disabled"]

    assert results["support"].sha256() == results["support_repeat"].sha256()
    assert disabled.status == "UNIQUE"
    assert disabled.selected_reasoning_status == "AUTHORIZED"
    assert (disabled.generation_status
            == W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN)
    assert disabled.generation_options == ()
    assert "expected" not in str(results["support"].to_dict()).lower()
    assert set(results["support"].query.to_dict()) == {
        "allow_generation", "source_ref_key", "surface"}

    with pytest.raises(W05V2PublicQueryError, match="SourceRef"):
        W05V2PublicQuery("小猫追逐小鸟。", source_ref_key=(1, 2, 3))
    with pytest.raises(W05V2PublicQueryError, match="SourceRef"):
        W05V2PublicQuery(
            "小猫追逐小鸟。", source_ref_key=(-1, *range(1, 11)))
