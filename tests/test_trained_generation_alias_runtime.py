"""Regression tests for multiple frozen R-01 profiles on one branch."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import AliasRouteSearchBudget
from pure_integer_ai.experiments.trained_generation_alias_runtime import (
    ProfiledAliasRelationRuntime,
)
from tests.test_r00_relation_closure import _r01_fixture


def _option(value: tuple[int, ...]):
    return SimpleNamespace(value=SimpleNamespace(stable_key=lambda: value))


def _proposal(values: tuple[tuple[int, ...], ...]):
    return SimpleNamespace(
        result=SimpleNamespace(options=tuple(_option(value) for value in values)))


def test_identical_profiles_are_deduplicated_and_surface_routes_remain_unique():
    """A repeated protocol record must not create a false ambiguity."""
    fixture = _r01_fixture()
    try:
        runtime = ProfiledAliasRelationRuntime(
            (fixture.alias_runtime, fixture.alias_runtime),
            conflict_error=ValueError,
        )
        assert len(runtime._profiled_runtimes) == 1
        proposal = runtime.preview_surface(
            fixture.objects["d"], fixture.objects["branch"],
            budget=AliasRouteSearchBudget(20, 20, 20),
        )
        assert proposal.result.selected is not None
        assert proposal.result.selected.value == fixture.objects["rep1"]
    finally:
        fixture.close()


def test_profiled_runtime_rejects_conflicting_representation_options():
    """Different profile outputs fail closed instead of choosing arbitrarily."""
    runtime = object.__new__(ProfiledAliasRelationRuntime)
    runtime._conflict_error = ValueError
    with pytest.raises(ValueError, match="冲突 Representation"):
        runtime._choose_preview((_proposal(((1,),)), _proposal(((2,),))))
