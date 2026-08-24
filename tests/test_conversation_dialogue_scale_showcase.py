"""公开对话规模展示的最小结构回归。"""

from pure_integer_ai.experiments.conversation_broad_qa_runtime import DialogueTurn
from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    DialogueScaleShowcaseTurn,
    build_dialogue_scale_showcase,
    showcase_questions,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    load_trained_surface_runtime,
)

from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_V6_RUN = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v6-clean-surface")
_V7_RUN = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v7-g7-variants")
_DATABASE = Path(
    "K:/pure_integer_ai_work/broad_qa_week_v1/evaluation/"
    "interactive-development-pack-v1/joint-index.sqlite3")
_G7_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_G7_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"
_G9_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_course_v1.jsonl.sample"
_G9_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_evidence_v1.jsonl.sample"
_V9_RUN = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v9-g7-qualifier-boundaries")
_G7_V3_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v3_qualifier.jsonl.sample"
_G7_V3_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v3_qualifier.jsonl.sample"


def test_showcase_has_complete_and_long_question_buckets() -> None:
    questions = showcase_questions()
    assert len(questions) == 14
    assert sum(len(value.encode("utf-8")) >= 48 for value in questions) == 5
    assert questions[7] == "它分布在哪些地区？"


def test_showcase_preserves_full_evidence_and_display_projection() -> None:
    turn = DialogueTurn(
        0, "完整问题？", "完整证据句。\n相邻证据句。", "完整证据句。",
        "ANSWER", "公开来源", "https://example.invalid/source", (1, 2, 3),
    )
    projected = DialogueScaleShowcaseTurn.from_turn(turn)
    assert projected.answer == "完整证据句。"
    assert projected.evidence_answer == "完整证据句。\n相邻证据句。"
    assert projected.answer_utf8_bytes == len("完整证据句。".encode("utf-8"))


def test_g7_is_opt_in_for_real_showcase() -> None:
    if not (_V7_RUN.is_dir() and _DATABASE.is_file()):
        pytest.skip("K: v7 training run or showcase database is not present")
    value = build_dialogue_scale_showcase(
        project_root=_ROOT,
        database_path=_DATABASE,
        training_run_root=_V7_RUN,
        extra_training_course_paths=(_G7_COURSE,),
        extra_variant_course_paths=(_G7_COURSE,),
        extra_variant_evidence_paths=(_G7_EVIDENCE,),
        variant_probe_inputs=("设备C的状态是待机。", "设备C状态为待机。"),
    )
    trained = value["training_observation"]
    assert trained["run_id"] == "dialogue-pack-v7-g7-variants"
    assert value["trained_surface_consumer"]["bound"] is True
    assert value["trained_surface_consumer"]["typed_probe_used_count"] == 4
    assert value["trained_surface_consumer"]["variant_probe_used_count"] == 1
    probes = value["trained_surface_consumer"]["variant_probe_results"]
    assert probes[0]["output_surface"] == "设备C状态为待机。"
    assert probes[0]["replay_bit_identical"] is True
    assert probes[1]["output_surface"] == "设备C状态为待机。"
    assert probes[1]["used"] is False
    assert probes[1]["reason"] == "no_learned_surface_shape"
    assert value["replay_bit_identical"] is True


def test_g9_order_is_opt_in_for_real_showcase() -> None:
    if not (_V9_RUN.is_dir() and _DATABASE.is_file()):
        pytest.skip("K: v9 training run or showcase database is not present")
    value = build_dialogue_scale_showcase(
        project_root=_ROOT,
        database_path=_DATABASE,
        training_run_root=_V9_RUN,
        extra_training_course_paths=(_G7_V3_COURSE,),
        extra_variant_course_paths=(_G7_V3_COURSE,),
        extra_variant_evidence_paths=(_G7_V3_EVIDENCE,),
        extra_order_course_paths=(_G9_COURSE,),
        extra_order_evidence_paths=(_G9_EVIDENCE,),
        variant_probe_inputs=("设备C的状态是待机。",),
        order_probe_values=("待机", "状态", "装置C"),
        order_probe_roles=("object", "predicate", "subject"),
    )
    runtime_result = value["trained_surface_consumer"]
    assert runtime_result["bound"] is True
    assert value["replay_bit_identical"] is True
    assert runtime_result["order_probe_used_count"] == 1
    assert runtime_result["order_probe_results"][0]["output_surface"] == "待机状态属于装置C。"
    assert runtime_result["order_probe_results"][0]["replay_bit_identical"] is True
    runtime = load_trained_surface_runtime(
        project_root=_ROOT,
        training_run_root=_V9_RUN,
        expected_pack_sha256="86b9b4e0839f8992bcd0738414d77310346c975b251639e51c711f54b523440d",
        extra_order_course_paths=(_G9_COURSE,),
        extra_order_evidence_paths=(_G9_EVIDENCE,),
    )
    rejected = runtime.render_order_typed(
        SurfaceSemantic("g9-unknown", "unknown", "装置C", "状态", "待机"),
        response_act="UNKNOWN", register="neutral",
        ordered_roles=("object", "predicate", "subject"),
        slot_values=("待机", "状态", "装置C"),
    )
    assert rejected.used is False
    assert rejected.surface == ""
    rendered = runtime.render_order_typed(
        SurfaceSemantic("g9-showcase", "state", "装置C", "状态", "待机"),
        response_act="ANSWER", register="neutral",
        ordered_roles=("object", "predicate", "subject"),
        source_id="g9-showcase-source", context_id="g9-showcase-context",
        family_id="g9-showcase-family",
    )
    assert rendered.used is True
    assert rendered.surface == "待机状态属于装置C。"
