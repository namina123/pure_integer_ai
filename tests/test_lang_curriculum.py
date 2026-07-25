"""Language curriculum planning regressions."""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.lang_curriculum import (
    LANGUAGE_RELATION_CURRICULUM,
    LanguageCurriculumStage,
    build_language_curriculum_plan,
)
from pure_integer_ai.training.stages import (
    STAGE1_SKELETON, STAGE2_CAUSES_ABS, STAGE3_REWARD,
    STAGE4_PROMOTE_WEAN,
)


def test_full_stage_explicitly_switches_to_all_relations():
    plan = build_language_curriculum_plan()

    assert len(plan) == len(LANGUAGE_RELATION_CURRICULUM)
    assert plan[-2].active_relations is not None
    assert "sense" not in plan[-2].active_relations
    assert "number" not in plan[-2].active_relations
    assert plan[-1].stage.load_all is True
    assert plan[-1].active_relations is None
    assert plan[-1].boot_relations == frozenset({"sense", "number"})
    assert [state.stage.training_stage for state in plan] == [
        STAGE1_SKELETON,
        STAGE2_CAUSES_ABS,
        STAGE2_CAUSES_ABS,
        STAGE2_CAUSES_ABS,
        STAGE2_CAUSES_ABS,
        STAGE3_REWARD,
        STAGE3_REWARD,
        STAGE4_PROMOTE_WEAN,
    ]
    assert plan[5].stage.name.startswith("T-L2b")
    assert plan[5].stage.capability_kind == "correspondence"


def test_empty_increment_keeps_cumulative_scope_without_meaning_all():
    plan = build_language_curriculum_plan(7)

    assert plan[-1].stage.name.startswith("T-L6d")
    assert plan[-1].active_relations == plan[-2].active_relations
    assert plan[-1].active_relations is not None
    assert plan[-1].boot_relations == frozenset()


def test_load_all_and_finite_increment_are_mutually_exclusive():
    with pytest.raises(ValueError, match="load_all"):
        LanguageCurriculumStage(
            "invalid", add_relations=frozenset({"alias"}), load_all=True)
