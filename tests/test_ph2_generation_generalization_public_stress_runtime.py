"""GG-03 public stress 性能 receipt 的快速、无重型生成合同测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    LoadedGenerationCandidatePack,
    build_generation_candidate_pack,
)
from pure_integer_ai.experiments import (
    ph2_generation_generalization_public_stress_runtime as runtime_module,
)
from pure_integer_ai.experiments.ph2_generation_generalization_public_stress import (
    PUBLIC_STRESS_CASE_IDS,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend

from tests.test_ph2_grounded_answer_course import (
    _connector_question_and_candidate,
)


_OBSERVATIONS = Path(
    "data/ph2/generation_generalization_public_stress_v1.jsonl")
_SOURCE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def test_public_stress_runtime_publishes_complete_path_free_receipt(
        tmp_path, monkeypatch):
    """假 actual 证明 12 条计时、内容锁、宿主零写和不可覆盖发布。"""
    model, _question, _planning, _candidate, _branch = (
        _connector_question_and_candidate())
    pack = build_generation_candidate_pack(
        model, hashlib.sha256(_SOURCE.read_bytes()).hexdigest())
    loaded = LoadedGenerationCandidatePack(
        tmp_path / "pack", tmp_path / "pack" / "manifest.json", pack)
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        monkeypatch.setattr(
            runtime_module, "_require_k_run_root",
            lambda _path: tmp_path.resolve())
        monkeypatch.setattr(
            runtime_module, "build_generation_generalization_code_identity",
            lambda _path: SimpleNamespace(aggregate_sha256="1" * 64))

        def fake_actual(_ctx, _loaded, observation, _policy):
            """返回不写 host、覆盖三路径的最小 actual 结构。"""
            path = (
                "REFERENCE" if observation.reference_course is not None
                else "ANSWER" if (
                    observation.question.answer_plan.response_act == "ANSWER")
                else "RESPONSE_ACT")
            return SimpleNamespace(
                runtime_status="PASS_EVALUATION_ACTUAL_CONJUNCTION",
                path=path,
                surface_text="公开输出",
                execution=SimpleNamespace(representations=((1,), (2,))),
                verifier_dimension_count=3,
                stable_key=lambda: (1, len(observation.episode_id)),
            )

        monkeypatch.setattr(
            runtime_module,
            "run_generation_generalization_evaluation_actual",
            fake_actual,
        )
        values = iter(range(0, 10_000, 100))
        target, receipt, receipt_sha = (
            runtime_module.
            run_and_publish_generation_generalization_public_stress(
                ctx,
                loaded,
                _OBSERVATIONS,
                Path.cwd(),
                tmp_path,
                clock_ns=lambda: next(values),
            ))
        assert target.read_bytes().endswith(b"\n")
        assert len(receipt_sha) == 64
        assert receipt["status"] == "PASS"
        assert receipt["host_snapshot_equal"] == 1
        assert receipt["record_count"] == 12
        assert receipt["total_elapsed_ns"] == 1200
        assert {item["episode_id"] for item in receipt["cases"]} == set(
            PUBLIC_STRESS_CASE_IDS)
        assert {item["path"] for item in receipt["cases"]} == {
            "ANSWER", "REFERENCE", "RESPONSE_ACT"}
        payload = target.read_text(encoding="utf-8")
        assert str(tmp_path) not in payload
        assert "surface_text" not in payload
    finally:
        backend.close()
