"""W-02 LC-01/02 到真实 Candidate/Evidence、消费者和持久回读。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from pure_integer_ai.experiments.ph2_w02_contract import (
    D03_GLOBAL_MANIFEST_PATH,
    W02_OWNER_KEY,
    W02_RUNNER_KEY,
    W02PayloadFirewall,
    W02RunRequest,
    W02TrainingPayload,
    open_w02_frozen_context,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_w02_learning import (
    GENERATION_GENERATED,
    GENERATION_UNKNOWN,
    OUTCOME_SUCCESS,
    SELECTION_ADOPTED,
    SELECTION_CONFLICT,
    W02LearningError,
    W02MorphologyTarget,
    open_w02_learning_runtime,
)
from pure_integer_ai.storage.backend import SQLiteBackend


_REPOSITORY = Path(__file__).resolve().parents[1]
_BASE_REMOTE_COMMIT = "6322ed3d6aedf1a0fceeaffd1990ed5c9015e3f8"


def _training_payload():
    context = open_w02_frozen_context(
        _REPOSITORY,
        D03_GLOBAL_MANIFEST_PATH,
        current_remote_commit_sha1=_BASE_REMOTE_COMMIT,
    )
    request = W02RunRequest(
        2, 1, 1, context.stage_key, W02_OWNER_KEY, W02_RUNNER_KEY,
        context.current_remote_commit_sha1, context.stable_key(),
        context.w01_receipt_sha256, (1, 20260729), (1, 1, 20260729),
        1, "fresh",
        tuple(item.relative_path
              for item in context.candidate_payload_bindings),
        tuple(item.relative_path
              for item in context.teacher_evidence_bindings),
    )
    return W02PayloadFirewall.open(
        _REPOSITORY, context, request).read_training_payload()


def test_lc01_raw_receipt_candidate_and_supersede_persist_to_real_history(
        tmp_path):
    """raw 永不被派生文本覆盖，错误候选退出且旧 Observation/Evidence 仍可回读。"""
    backend = SQLiteBackend(str(tmp_path / "w02.sqlite3"))
    try:
        runtime = open_w02_learning_runtime(backend, mode="fresh")
        report = runtime.consume(_training_payload())

        width = runtime.candidate("teacher-width-candidate-v1")
        old = runtime.candidate("teacher-whitespace-collapse-v1")
        revision = runtime.candidate("teacher-whitespace-preserve-v2")
        assert width.raw_text == "ＡＢ，测试。"
        assert width.derived_text == "AB，测试。"
        assert width.raw_sha256 != width.derived_sha256
        assert width.normalization_operations == ("FULLWIDTH_TO_HALFWIDTH",)
        assert old.raw_text == revision.raw_text == "甲  乙"
        assert old.active is False
        assert old.lifecycle == "SUPERSEDED"
        assert revision.active is True
        assert report.raw_observation_count == 8
        assert report.teacher_evidence_count == 19
        assert report.candidate_count == 19
        assert report.active_candidate_count == 14
        assert report.active_lifecycle_count == 16
        assert report.core_learning_writes > 0
        assert report.memory_learning_writes == 0
    finally:
        backend.close()


def test_understanding_keeps_multi_candidate_and_contiguous_oov(tmp_path):
    """歧义边界并存；未知连续串不能被单字符回退伪装成已知词。"""
    backend = SQLiteBackend(str(tmp_path / "w02.sqlite3"))
    try:
        runtime = open_w02_learning_runtime(backend, mode="fresh")
        runtime.consume(_training_payload())

        ambiguous = runtime.understand("研究生命起源")
        assert {item.tokens for item in ambiguous.candidates} >= {
            ("研究生", "命", "起源"),
            ("研究", "生命", "起源"),
        }
        assert len(ambiguous.active_boundary_candidates) == 2
        unknown = runtime.understand("陌生连续串")
        assert any(
            len(item.parts) == 1
            and item.parts[0].surface == "陌生连续串"
            and item.parts[0].known_word_form is False
            for item in unknown.candidates
        )
        assert unknown.known_word_form_count == 0
    finally:
        backend.close()


def test_lc02_relations_and_typed_generation_are_productive_not_replay(
        tmp_path):
    """六类关系进入候选；生成从关系组合，整词回放和 OOV 均不能通过。"""
    backend = SQLiteBackend(str(tmp_path / "w02.sqlite3"))
    try:
        runtime = open_w02_learning_runtime(backend, mode="fresh")
        runtime.consume(_training_payload())

        assert set(runtime.active_morphology_relation_kinds()) == {
            "HAS_STEM", "FILLS_SLOT", "ATTACHES_AFFIX",
            "COMPOUND_COMPONENT", "REDUPLICATES", "EXCEPTION_TO",
        }
        redup = runtime.generate(W02MorphologyTarget(
            construction_key="redup-aa-construction-v1",
            stem_surface="看",
        ))
        novel_combo = runtime.generate(W02MorphologyTarget(
            construction_key="suffix-hua-construction-v1",
            stem_surface="纸",
        ))
        replay = runtime.generate(W02MorphologyTarget(
            construction_key="dictionary-replay-construction-v1",
            stem_surface="雨伞",
        ))
        oov = runtime.generate(W02MorphologyTarget(
            construction_key="suffix-hua-construction-v1",
            stem_surface="从未见词干",
        ))

        assert redup.status == GENERATION_GENERATED
        assert redup.surfaces == ("看看",)
        assert novel_combo.status == GENERATION_GENERATED
        assert novel_combo.surfaces == ("纸化",)
        assert replay.status == GENERATION_UNKNOWN and replay.surfaces == ()
        assert oov.status == GENERATION_UNKNOWN and oov.surfaces == ()
        assert all("expected" not in field
                   for field in W02MorphologyTarget.__dataclass_fields__)
    finally:
        backend.close()


def test_sqlite_fresh_connection_restores_candidate_evidence_and_consumers(
        tmp_path):
    """新连接只从 Core 图和 H-00/H-04 历史恢复相同候选、撤回和生成规则。"""
    path = tmp_path / "w02.sqlite3"
    first_backend = SQLiteBackend(str(path))
    try:
        first = open_w02_learning_runtime(first_backend, mode="fresh")
        first.consume(_training_payload())
        state = first.state_key()
        understanding = first.understand("研究生命起源").stable_key()
        generation = first.generate(W02MorphologyTarget(
            construction_key="redup-aa-construction-v1",
            stem_surface="看",
        )).stable_key()
    finally:
        first_backend.close()

    second_backend = SQLiteBackend(str(path))
    try:
        restored = open_w02_learning_runtime(second_backend, mode="resume")
        assert restored.state_key() == state
        assert restored.understand("研究生命起源").stable_key() == understanding
        assert restored.generate(W02MorphologyTarget(
            construction_key="redup-aa-construction-v1",
            stem_surface="看",
        )).stable_key() == generation
        assert restored.candidate("teacher-whitespace-collapse-v1").lifecycle == (
            "SUPERSEDED")
    finally:
        second_backend.close()


def test_malformed_lc_payload_and_duplicate_teacher_fail_before_learning_writes(
        tmp_path):
    """typed payload 或 teacher 唯一 owner 损坏时，Candidate/Core 必须保持零写。"""
    payload = _training_payload()
    malformed = replace(
        payload.observations[0],
        typed_payload=CanonicalJsonObject.from_value({"invalid": 1}),
    )
    cases = (
        W02TrainingPayload(
            payload.source_refs,
            (malformed, *payload.observations[1:]),
            payload.teacher_evidence,
        ),
        W02TrainingPayload(
            payload.source_refs,
            payload.observations,
            (*payload.teacher_evidence, payload.teacher_evidence[0]),
        ),
    )
    for index, bad_payload in enumerate(cases):
        backend = SQLiteBackend(str(tmp_path / f"bad-{index}.sqlite3"))
        try:
            runtime = open_w02_learning_runtime(backend, mode="fresh")
            before = backend.recovery_state_snapshot()
            with pytest.raises(W02LearningError):
                runtime.consume(bad_payload)
            assert backend.recovery_state_snapshot() == before
            assert runtime.candidate_runtime.report().candidate_count == 0
        finally:
            backend.close()


def test_same_payload_replay_is_idempotent_and_reports_zero_new_writes(tmp_path):
    """同字节 train payload 可安全重放，但不得重复 Candidate/Evidence/词形写。"""
    path = tmp_path / "w02.sqlite3"
    payload = _training_payload()
    backend = SQLiteBackend(str(path))
    try:
        runtime = open_w02_learning_runtime(backend, mode="fresh")
        first = runtime.consume(payload)
        state = runtime.state_key()
        replay = runtime.consume(payload)
        assert replay.replayed is True
        assert replay.candidate_count == first.candidate_count
        assert replay.evidence_count == first.evidence_count
        assert replay.core_learning_writes == 0
        assert replay.memory_learning_writes == 0
        assert replay.word_form_writes == 0
        assert runtime.state_key() == state
    finally:
        backend.close()

    backend = SQLiteBackend(str(path))
    try:
        restored = open_w02_learning_runtime(backend, mode="resume")
        before = restored.state_key()
        replay = restored.consume(payload)
        assert replay.replayed is True
        assert replay.core_learning_writes == 0
        assert restored.state_key() == before
    finally:
        backend.close()


def test_envelope_tamper_fails_closed_on_fresh_connection_readback(tmp_path):
    """持久 envelope 的 JSON、size 或 SHA 任一漂移都不能被 Candidate 恢复。"""
    path = tmp_path / "w02.sqlite3"
    backend = SQLiteBackend(str(path))
    try:
        runtime = open_w02_learning_runtime(backend, mode="fresh")
        runtime.consume(_training_payload())
    finally:
        backend.close()


def test_directional_use_outcome_is_exact_persistent_and_ablatable(tmp_path):
    """理解/生成分账到实际 Candidate；关闭 assessment 或形态 consumer 必须退化。"""
    path = tmp_path / "w02.sqlite3"
    backend = SQLiteBackend(str(path))
    try:
        runtime = open_w02_learning_runtime(backend, mode="fresh")
        runtime.consume(_training_payload())
        raw_text = "研究生命起源"
        initial = runtime.select_understanding(raw_text)
        assert initial.status == SELECTION_CONFLICT
        chosen = initial.understanding.active_boundary_candidates[0]
        understanding_use = runtime.record_understanding_outcome(
            raw_text,
            chosen,
            outcome_kind=OUTCOME_SUCCESS,
        )
        selected = runtime.select_understanding(raw_text)
        assert selected.status == SELECTION_ADOPTED
        assert selected.candidate_id == chosen
        assert runtime.select_understanding(
            raw_text, outcome_assessment_enabled=False
        ).status == SELECTION_CONFLICT

        target = W02MorphologyTarget(
            construction_key="suffix-hua-construction-v1",
            stem_surface="纸",
        )
        generated = runtime.generate(target)
        generation_use = runtime.record_generation_outcome(
            target,
            generated.surfaces[0],
            outcome_kind=OUTCOME_SUCCESS,
        )
        assert understanding_use.direction == "UNDERSTANDING"
        assert generation_use.direction == "GENERATION"
        assert understanding_use.consumer_key != generation_use.consumer_key
        assert understanding_use.candidate_key != generation_use.candidate_key
        assert runtime.attribution_report().use_count_by_direction == (
            ("GENERATION", 1), ("UNDERSTANDING", 1))
        assert runtime.generate(
            target, morphology_consumer_enabled=False
        ).status == GENERATION_UNKNOWN
        state = runtime.state_key()
    finally:
        backend.close()

    backend = SQLiteBackend(str(path))
    try:
        restored = open_w02_learning_runtime(backend, mode="resume")
        assert restored.state_key() == state
        assert restored.select_understanding(raw_text).candidate_id == chosen
        assert restored.attribution_report().outcome_count == 2
        assert restored.attribution_report().assessment_count == 2
    finally:
        backend.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE ph2_w02_envelope SET payload_json = payload_json || ' ' "
            "WHERE envelope_id = ("
            "SELECT MIN(envelope_id) FROM ph2_w02_envelope)"
        )
        connection.commit()
    finally:
        connection.close()

    backend = SQLiteBackend(str(path))
    try:
        restored = open_w02_learning_runtime(backend, mode="resume")
        with pytest.raises(W02LearningError, match="envelope"):
            restored.state_key()
    finally:
        backend.close()
