"""W04-02 typed adapter、H-05 lifecycle 与 U/R/G consumer 专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w04_adapter import (
    adapt_w04_training_payload,
)
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_FORMAL_RUN_ID,
    W04_RESOURCE_BUDGET,
    W04_RUNNER_KEY,
    W04_STAGE_KEY,
    W04_W03_BASE_RUN_ID,
    W04RunRequest,
)
from pure_integer_ai.experiments.ph2_w04_firewall import W04PayloadFirewall
from pure_integer_ai.experiments.ph2_w04_generation import (
    build_w04_generation_runtime,
)
from pure_integer_ai.experiments.ph2_w04_generation_contract import (
    W04_GENERATION_READY,
    W04GenerationRequest,
)
from pure_integer_ai.experiments.ph2_w04_learning import (
    build_w04_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_reasoning import (
    W04_REASONING_AUTHORIZED,
    build_w04_reasoning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_understanding import (
    W04_UNDERSTANDING_UNIQUE,
    build_w04_understanding_runtime,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from tests.w04_historical_context import open_historical_w04_context


ROOT = Path(__file__).resolve().parents[1]
HEAD = "da69958c1f149a2f264053f7b7407a53f575cd93"


def _context_and_payload(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "context.sqlite"))
    try:
        context = open_historical_w04_context(
            ROOT,
            current_remote_commit_sha1=HEAD,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
    finally:
        backend.close()
    request = W04RunRequest(
        run_id=W04_FORMAL_RUN_ID,
        parent_run_id=W04_W03_BASE_RUN_ID,
        base_run_id=W04_W03_BASE_RUN_ID,
        stage_key=W04_STAGE_KEY,
        owner_key=context.owner_key,
        runner_key=W04_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=1,
        mode="fresh",
        resource_budget=tuple(sorted(W04_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    payload = W04PayloadFirewall.open(ROOT, context, request).read_training_payload()
    return context, payload


def test_w04_adapter_consumes_primitive_typed_payload_without_cue_table(tmp_path):
    """adapter 直接读 typed payload，保留同 surface 多 primitive 与同 primitive 多 surface。"""
    _context, payload = _context_and_payload(tmp_path)
    adapted = adapt_w04_training_payload(payload)
    assert len(adapted.candidates) == 4
    assert {item.surface_form for item in adapted.candidates} >= {"导致", "使得", "是"}
    assert len(adapted.candidates_for_surface("导致")) == 2
    assert len(adapted.candidates_for_primitive("relation", 4)) == 2
    assert all(item.selection_state == "UNSELECTED" for item in adapted.candidates)
    assert all(item.observation.split == "train" for item in adapted.candidates)
    assert {item.decision for item in adapted.evidence} == {
        "bind", "reject_mismatch", "keep_ambiguous"}


def test_w04_learning_and_consumers_use_active_evidence(tmp_path):
    """learning 真调用 H-05/H-04，三向 consumer 随 active Evidence 工作。"""
    _context, payload = _context_and_payload(tmp_path)
    adapted = adapt_w04_training_payload(payload)
    backend = SQLiteBackend(str(tmp_path / "w04_learning.sqlite"))
    try:
        learning = build_w04_learning_runtime(backend, adapted)
        report = learning.report()
        assert report.candidate_count == 4
        assert report.evidence_application_count == 4
        assert report.active_candidate_count == 1
        assert report.superseded_candidate_count == 1

        understanding = build_w04_understanding_runtime(learning)
        resolution = understanding.resolve("使得", "暴雨使得河水上涨。")
        assert resolution.status == W04_UNDERSTANDING_UNIQUE
        assert resolution.selected is not None
        assert resolution.selected.coordinate_key() == ("relation", 4)

        reasoning = build_w04_reasoning_runtime(learning)
        use = reasoning.authorize("relation", 4)
        assert use.status == W04_REASONING_AUTHORIZED
        assert use.evidence_count >= 1

        generation = build_w04_generation_runtime(learning)
        choice = generation.choose(W04GenerationRequest(
            "relation", 4, "暴雨使得河水上涨。", True))
        assert choice.status == W04_GENERATION_READY
        assert {item.surface_form for item in choice.options} == {"使得"}
        generated_use = generation.adopt(choice, choice.options)
        outcome = generation.verify_use(generated_use)
        assert outcome.verdict == "SUPPORT"
    finally:
        backend.close()
