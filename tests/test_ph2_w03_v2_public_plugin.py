"""Real public-only W-03 V2 consumer tests for P0/P1/P2."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_construction_course import (
    compile_authored_construction_course,
)
from pure_integer_ai.experiments.ph2_authored_sense_course import (
    compile_authored_sense_course,
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
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_v2_public_plugin import (
    W03_V2_PUBLIC_EXPERIMENTAL,
    W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
    W03_V2_PUBLIC_W03_STARTED,
    build_w03_v2_public_capability_plugin,
    w03_v2_public_plugin_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_preflight import (
    build_w03_v2_public_preflight,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_query import (
    W03_V2_PUBLIC_QUERY_GENERATION_NOT_RUN,
    W03V2PublicQuery,
    run_w03_v2_public_query,
    run_w03_v2_public_queries,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicSourceError,
    build_w03_v2_public_evaluation_batch,
    build_w03_v2_public_run_context,
    w03_v2_public_training_payload,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _records(build, kind: str) -> tuple[object, ...]:
    values = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            values.extend(read_record_artifact(build.pack_root, identity))
    return tuple(values)


@pytest.fixture(scope="module")
def public_payloads(tmp_path_factory):
    root = tmp_path_factory.mktemp("w03-v2-public")
    sense = compile_authored_sense_course(
        REPOSITORY / "data/ph2/authored_sense_seed_v1.jsonl.sample",
        root / "sense",
    )
    construction = compile_authored_construction_course(
        REPOSITORY / "data/ph2/authored_construction_seed_v1.jsonl.sample",
        root / "construction",
    )

    def payload(builds) -> W03TrainingPayload:
        sources = tuple(
            item for build in builds
            for item in _records(build, RECORD_SOURCE_REF))
        observations = tuple(
            item for build in builds
            for item in _records(build, RECORD_OBSERVATION)
            if item.split == "train")
        evidence = tuple(
            item for build in builds
            for item in _records(build, RECORD_TEACHER_EVIDENCE))
        return W03TrainingPayload(sources, observations, evidence)

    return {
        "combined": payload((sense, construction)),
        "sense": payload((sense,)),
        "root": root,
    }


@pytest.fixture(scope="module")
def public_preflight(public_payloads):
    batch = build_w03_v2_public_evaluation_batch(
        public_payloads["combined"])
    plugin = build_w03_v2_public_capability_plugin(REPOSITORY)
    return batch, plugin, build_w03_v2_public_preflight(
        REPOSITORY, batch, plugin)


@pytest.fixture(scope="module")
def public_query_results(public_payloads):
    batch = build_w03_v2_public_evaluation_batch(
        public_payloads["combined"])
    queries = {
        "unique": W03V2PublicQuery(
            "银行", "他去银行办理存款，并向柜员出示证件。"),
        "ambiguous": W03V2PublicQuery(
            "我也想起来了", "我也想起来了"),
        "unknown": W03V2PublicQuery("不存在的公开词项"),
        "clarify": W03V2PublicQuery("银行", "尚未学习的新银行语境。"),
        "multi_surface": W03V2PublicQuery("把门打开", "把门打开"),
    }
    projected = run_w03_v2_public_queries(
        batch, tuple(queries.values()))
    results = dict(zip(queries, projected, strict=True))
    results["unique_repeat"] = run_w03_v2_public_query(
        batch, queries["unique"])
    return batch, results


def test_public_source_adapter_is_exact_source_first_and_label_free(
        public_payloads) -> None:
    """Only the 16 referenced public train sources and 16 teacher pairs enter."""
    payload = public_payloads["combined"]
    batch = build_w03_v2_public_evaluation_batch(payload)

    assert len(payload.source_refs) == 30
    assert len(batch.source_records) == len(batch.pairs) == 16
    assert batch.source_binding.record_count == 16
    assert batch.records == (*batch.source_records, *batch.pairs)
    assert all(item.record.redistribution_policy == "PUBLIC"
               for item in batch.source_records)
    assert all(item.observation.split == "train" for item in batch.pairs)
    assert all(item.evidence.observation_key == item.observation.stable_key
               for item in batch.pairs)
    assert "EvaluatorLabelRecord" not in {
        type(item).__name__ for item in batch.records}
    assert w03_v2_public_training_payload(batch) == W03TrainingPayload(
        tuple(item.record for item in batch.source_records),
        tuple(item.observation for item in batch.pairs),
        tuple(item.evidence for item in batch.pairs),
    )


def test_public_plugin_produces_real_pass_p0_p2_without_formal_claim(
        public_preflight) -> None:
    """The active W-03 runtime closes all nine public experimental conjuncts."""
    batch, plugin, preflight = public_preflight
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-03")

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
                16, 16, 48, 48)
    assert audit.transport_bytes_read == batch.transport_bytes
    assert audit.logic_operations > 0
    assert audit.zero_call_window_count == 3
    assert audit.write_account is not None and audit.write_account.is_zero
    assert (preflight.experimental, preflight.formal_mastery_claim,
            preflight.w03_started) == (1, 0, 0)
    assert (W03_V2_PUBLIC_EXPERIMENTAL,
            W03_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
            W03_V2_PUBLIC_W03_STARTED) == (1, 0, 0)
    assert plugin.declaration.semantic_sha256 == (
        w03_v2_public_plugin_semantic_sha256(REPOSITORY))


def test_public_plugin_reports_ne_when_generation_coverage_is_missing(
        public_payloads) -> None:
    """Sense-only public data cannot paper over the multi-surface hard case."""
    batch = build_w03_v2_public_evaluation_batch(public_payloads["sense"])
    plugin = build_w03_v2_public_capability_plugin(REPOSITORY)
    outcome = plugin.evaluate(
        build_w03_v2_public_run_context(batch, plugin.declaration),
        batch.records,
    )
    by_key = {
        item.result_key: item for item in outcome.result_set.results}

    assert by_key["W-03-V2-GENERATION-HARD-CONJUNCT"].status == "NE"
    assert outcome.result_set.status == "NE"
    assert by_key["W-03-V2-V06-CLONE"].status == "PASS"


def test_public_source_rejects_local_only_and_non_train_records(
        public_payloads) -> None:
    """Public P0 fails closed before capability code sees a private/local row."""
    payload = public_payloads["combined"]
    referenced = payload.observations[0].source_ref_key
    sources = tuple(
        replace(item, redistribution_policy="LOCAL_ONLY")
        if item.stable_key == referenced else item
        for item in payload.source_refs
    )
    with pytest.raises(W03V2PublicSourceError, match="nonpublic"):
        build_w03_v2_public_evaluation_batch(replace(
            payload, source_refs=sources))

    held_out = replace(payload.observations[0], split="held_out")
    with pytest.raises(W03V2PublicSourceError, match="train-only"):
        build_w03_v2_public_evaluation_batch(replace(
            payload, observations=(held_out, *payload.observations[1:])))


def test_public_plugin_has_no_private_family_or_formal_guard_route() -> None:
    """The first consumer cannot import old private evaluator/family machinery."""
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/ph2_w03_v2_public_plugin.py"
    )).read_text(encoding="utf-8")

    assert "ph2_w03_evaluator" not in source
    assert "ph2_w03_evaluator_family" not in source
    assert "ph2_w03_evaluator_runtime" not in source
    assert "ph2_w03_firewall" not in source
    assert "w02_artifacts" not in source
    assert "run_evaluation_family_once" not in source
    assert "consume_guard" not in source


def test_public_query_projects_unique_ambiguous_unknown_and_clarify(
        public_query_results) -> None:
    """External surface/context queries expose W-03 states without prose rules."""
    batch, results = public_query_results
    unique = results["unique"]
    ambiguous = results["ambiguous"]
    unknown = results["unknown"]
    clarify = results["clarify"]

    assert unique.status == "UNIQUE"
    assert unique.selected_sense_key is not None
    assert unique.generation_status == "READY"
    assert {item.surface for item in unique.generation_options} == {"银行"}
    assert len(unique.candidates) == 1
    assert unique.candidates[0].active == 1

    assert ambiguous.status == "AMBIGUOUS"
    assert ambiguous.clarify_required == 1
    assert ambiguous.selected_sense_key is None
    assert len(ambiguous.candidates) == 2
    assert ambiguous.generation_status == (
        W03_V2_PUBLIC_QUERY_GENERATION_NOT_RUN)

    assert unknown.status == "UNKNOWN"
    assert unknown.candidates == unknown.generation_options == ()
    assert unknown.clarify_required == 0

    assert clarify.status == "CLARIFY"
    assert clarify.clarify_required == 1
    assert len({item.concept_key for item in clarify.candidates}) > 1
    assert clarify.generation_options == ()
    assert all(item.source_commitment for item in clarify.candidates)
    assert all(result.source_binding_sha256 == batch.source_binding.sha256()
               for result in results.values())


def test_public_query_returns_all_authorized_surfaces_and_is_deterministic(
        public_query_results) -> None:
    """Generation reads active projections and repeated query output is identical."""
    _, results = public_query_results
    multi = results["multi_surface"]

    assert multi.status == "UNIQUE"
    assert multi.generation_status == "READY"
    assert {item.surface for item in multi.generation_options} == {
        "把门打开", "把门再次打开",
    }
    assert len({item.sense_key for item in multi.generation_options}) == 2
    assert all(item.source_ref_key for item in multi.generation_options)
    assert multi.experimental == 1
    assert multi.formal_mastery_claim == multi.w03_started == 0
    assert results["unique"].sha256() == results["unique_repeat"].sha256()
