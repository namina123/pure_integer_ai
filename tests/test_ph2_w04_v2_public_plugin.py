"""Real public-only W-04 V2 consumer tests for P0/P1/P2."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_primitive_course import (
    compile_authored_primitive_course,
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
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_plugin import (
    W04_V2_PUBLIC_EXPERIMENTAL,
    W04_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
    W04_V2_PUBLIC_W04_STARTED,
    build_w04_v2_public_capability_plugin,
    w04_v2_public_plugin_semantic_sha256,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_preflight import (
    build_w04_v2_public_preflight,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicSourceError,
    build_w04_v2_public_evaluation_batch,
    build_w04_v2_public_run_context,
    w04_v2_public_training_payload,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _records(build, kind: str) -> tuple[object, ...]:
    values = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            values.extend(read_record_artifact(build.pack_root, identity))
    return tuple(values)


@pytest.fixture(scope="module")
def public_payload(tmp_path_factory) -> W04TrainingPayload:
    root = tmp_path_factory.mktemp("w04-v2-public")
    build = compile_authored_primitive_course(
        REPOSITORY / "data/ph2/authored_primitive_seed_v1.jsonl.sample",
        root,
    )
    return W04TrainingPayload(
        _records(build, RECORD_SOURCE_REF),
        tuple(item for item in _records(build, RECORD_OBSERVATION)
              if item.split == "train"),
        _records(build, RECORD_TEACHER_EVIDENCE),
    )


@pytest.fixture(scope="module")
def public_preflight(public_payload):
    batch = build_w04_v2_public_evaluation_batch(public_payload)
    plugin = build_w04_v2_public_capability_plugin(REPOSITORY)
    return batch, plugin, build_w04_v2_public_preflight(
        REPOSITORY, batch, plugin)


def test_public_source_is_train_only_source_first_and_label_free(
        public_payload) -> None:
    """Only four referenced CC0 train SourceRefs and teacher pairs enter."""
    batch = build_w04_v2_public_evaluation_batch(public_payload)

    assert len(public_payload.source_refs) == 7
    assert len(batch.source_records) == len(batch.pairs) == 4
    assert batch.source_binding.record_count == 4
    assert batch.records == (*batch.source_records, *batch.pairs)
    assert all(item.record.redistribution_policy == "PUBLIC"
               for item in batch.source_records)
    assert all(item.observation.split == "train" for item in batch.pairs)
    assert all(item.evidence.observation_key == item.observation.stable_key
               for item in batch.pairs)
    assert "EvaluatorLabelRecord" not in {
        type(item).__name__ for item in batch.records}
    assert w04_v2_public_training_payload(batch) == W04TrainingPayload(
        tuple(item.record for item in batch.source_records),
        tuple(item.observation for item in batch.pairs),
        tuple(item.evidence for item in batch.pairs),
    )


def test_public_plugin_closes_all_nine_p0_p2_conjuncts(
        public_preflight) -> None:
    """Current W-04 runtime learns replacement and generates from public data."""
    batch, plugin, preflight = public_preflight
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-04")

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
                4, 4, 12, 12)
    assert audit.transport_bytes_read == batch.transport_bytes
    assert 0 < audit.logic_operations <= 100_000
    assert audit.zero_call_window_count == 3
    assert audit.write_account is not None and audit.write_account.is_zero
    assert (preflight.experimental, preflight.formal_mastery_claim,
            preflight.w04_started) == (1, 0, 0)
    assert (W04_V2_PUBLIC_EXPERIMENTAL,
            W04_V2_PUBLIC_FORMAL_MASTERY_CLAIM,
            W04_V2_PUBLIC_W04_STARTED) == (1, 0, 0)
    assert plugin.declaration.semantic_sha256 == (
        w04_v2_public_plugin_semantic_sha256(REPOSITORY))


def test_public_plugin_reports_ne_for_single_seed_theater(
        public_payload) -> None:
    """One support seed cannot masquerade as replacement or seed ablation."""
    observation = next(
        item for item in public_payload.observations
        if item.sample_role == "support")
    evidence = next(
        item for item in public_payload.teacher_evidence
        if item.observation_key == observation.stable_key)
    payload = W04TrainingPayload(
        public_payload.source_refs,
        (observation,),
        (evidence,),
    )
    batch = build_w04_v2_public_evaluation_batch(payload)
    plugin = build_w04_v2_public_capability_plugin(REPOSITORY)
    outcome = plugin.evaluate(
        build_w04_v2_public_run_context(batch, plugin.declaration),
        batch.records,
    )
    by_key = {
        item.result_key: item for item in outcome.result_set.results}

    assert by_key["W-04-V2-CONTENT-REPLACEMENT"].status == "NE"
    assert by_key["W-04-V2-CUE-REPLACEMENT"].status == "NE"
    assert by_key["W-04-V2-SEED-ABLATION"].status == "NE"
    assert by_key["W-04-V2-GENERATION-HARD-CONJUNCT"].status == "PASS"
    assert by_key["W-04-V2-V06-CLONE"].status == "PASS"
    assert outcome.result_set.status == "NE"


def test_public_source_rejects_local_only_and_non_train_records(
        public_payload) -> None:
    """P0 fails closed before W-04 capability code sees local/held-out data."""
    referenced = public_payload.observations[0].source_ref_key
    sources = tuple(
        replace(item, redistribution_policy="LOCAL_ONLY")
        if item.stable_key == referenced else item
        for item in public_payload.source_refs
    )
    with pytest.raises(W04V2PublicSourceError, match="nonpublic"):
        build_w04_v2_public_evaluation_batch(replace(
            public_payload, source_refs=sources))

    held_out = replace(public_payload.observations[0], split="held_out")
    with pytest.raises(W04V2PublicSourceError, match="train-only"):
        build_w04_v2_public_evaluation_batch(replace(
            public_payload,
            observations=(held_out, *public_payload.observations[1:])))


def test_public_plugin_has_no_private_family_or_formal_guard_route() -> None:
    """The W-04 consumer cannot import private evaluator/family machinery."""
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/ph2_w04_v2_public_plugin.py"
    )).read_text(encoding="utf-8")

    assert "ph2_w04_evaluator" not in source
    assert "ph2_w04_evaluator_family" not in source
    assert "ph2_w04_evaluator_runtime" not in source
    assert "ph2_w04_firewall" not in source
    assert "w02_artifacts" not in source
    assert "run_evaluation_family_once" not in source
    assert "consume_guard" not in source
