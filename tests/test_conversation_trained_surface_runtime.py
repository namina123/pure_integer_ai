from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    load_trained_surface_runtime,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceSemantic,
)


_ROOT = Path(__file__).resolve().parents[1]
_PACK_SHA = "1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d"
_RUN_ROOT = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v6-clean-surface")


def _runtime_or_skip():
    if not _RUN_ROOT.is_dir():
        pytest.skip("K: dialogue training run is not present in this environment")
    return load_trained_surface_runtime(
        project_root=_ROOT,
        training_run_root=_RUN_ROOT,
        expected_pack_sha256=_PACK_SHA,
    )


def test_public_training_state_drives_typed_causal_surface() -> None:
    runtime = _runtime_or_skip()
    result = runtime.render("暴雨使得河水上涨。")
    assert result.used is True
    assert result.surface == "暴雨使得河水上涨。"
    assert result.pattern_id > 0
    assert result.graph_size == 1154


def test_untyped_surface_fails_closed_without_changing_answer() -> None:
    runtime = _runtime_or_skip()
    result = runtime.render("矮寨大桥于2012年建成通车。")
    assert result.used is False
    assert result.surface == "矮寨大桥于2012年建成通车。"
    assert result.reason == "no_learned_surface_shape"


def test_typed_non_causal_and_non_answer_surfaces_are_rebuilt_from_new_values() -> None:
    runtime = _runtime_or_skip()
    cases = (
        (
            "ANSWER", "polite", ("subject", "predicate", "qualifier", "object"),
            ("新入口", "启用时间", "审计记录", "2030年1月"),
            "新入口的启用时间（审计记录）为2030年1月。",
        ),
        (
            "UNKNOWN", "neutral", ("source", "scope"),
            ("当前", "青石台的运行预算"),
            "当前资料没有提供青石台的运行预算。",
        ),
        (
            "CLARIFY", "polite", ("choice", "target"),
            ("甲区还是乙区", "数量"),
            "请先选择甲区还是乙区，再说明要查询的数量。",
        ),
        (
            "REPAIR", "polite", ("acknowledge", "request"),
            ("前面的条件不够明确", "具体时间"),
            "前面的条件不够明确，请说明具体时间。",
        ),
    )
    for act, register, roles, values, expected in cases:
        result = runtime.render_typed(
            SurfaceSemantic(
                f"new-{act.lower()}", act.lower(),
                ("新入口" if act == "ANSWER" else "新对象"),
                ("启用时间" if act == "ANSWER" else "新属性"),
                ("2030年1月" if act == "ANSWER" else "新值"),
            ),
            response_act=act,
            register=register,
            ordered_roles=roles,
            slot_values=values,
            source_id="new-source",
            context_id="new-context",
            family_id="new-family",
        )
        assert result.used is True
        assert result.surface == expected
        assert result.pattern_id > 0
    assert "东岸入口" not in cases[0][4]
    assert "玄衡台" not in cases[1][4]
    assert "北区" not in cases[2][4]
