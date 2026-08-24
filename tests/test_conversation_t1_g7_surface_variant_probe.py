"""T1-G7 opt-in runtime probe；只验证真实结构学习，不验证事实答案。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_t1_g7_surface_variant_probe import (
    G7_PROBE_OBSERVED,
    build_g7_surface_variant_probe,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    load_trained_surface_runtime,
)


_ROOT = Path(__file__).resolve().parents[1]
_V6_RUN = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v6-clean-surface")
_V7_RUN = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v7-g7-variants")
_V6_SHA = "1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d"
_V7_SHA = "d56850d9f80fc087612e486e2cff38534358917ad7a69e4bcad13422a8f7eaaf"
_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"
_INPUT = "设备C的状态是待机。"


def test_default_v6_does_not_load_g7_variant_model() -> None:
    if not _V6_RUN.is_dir():
        pytest.skip("K: v6 dialogue training run is not present")
    runtime = load_trained_surface_runtime(
        project_root=_ROOT, training_run_root=_V6_RUN,
        expected_pack_sha256=_V6_SHA,
    )
    assert runtime.variant_models == {}
    result = runtime.render(_INPUT, response_act="ANSWER")
    assert result.used is False
    assert result.surface == _INPUT


def test_opt_in_g7_recombines_unseen_surface_identity_and_guards_non_answer() -> None:
    if not _V7_RUN.is_dir():
        pytest.skip("K: v7 dialogue training run is not present")
    report = build_g7_surface_variant_probe(
        project_root=_ROOT,
        training_run_root=_V7_RUN,
        expected_pack_sha256=_V7_SHA,
        variant_course_path=_COURSE,
        variant_evidence_path=_EVIDENCE,
        input_surface=_INPUT,
    )
    assert report["probe_status"] == G7_PROBE_OBSERVED
    assert report["capability_scope"] == "surface_structure_variant_only"
    assert report["not_claimed"] == [
        "broad_qa", "fact_generation", "weaning_readiness"]
    assert report["replay_bit_identical"] is True
    assert report["replay_record_sha256"] == report["replay_record_sha256_replay"]
    assert report["observation"]["output_surface"] == "设备C状态为待机。"
    assert report["observation"]["reason"] == "variant_selected"
    assert report["observation"]["pattern_id"] > 0
    assert all(item["used"] is False for item in report["non_answer_guards"])
