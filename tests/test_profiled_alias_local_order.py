"""多协议规范序只由身份决定，实际词形仍走完整局部关系闭环。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionSelector,
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.identity import (
    minimal_instruction_identity,
)
from pure_integer_ai.experiments.alias_relation_runtime import AliasRelationRuntime
from pure_integer_ai.experiments.trained_generation_alias_runtime import (
    ProfiledAliasRelationRuntime,
)
from tests.test_r00_relation_closure import _r01_fixture


@pytest.mark.parametrize("order", ((0, 1), (1, 0), (1, 0, 1)))
def test_profile_order_is_local_deterministic_and_preserves_current_routes(monkeypatch, order):
    """不同调用序和重复不触发整关系读取，当前词形及采用协议唯一。"""
    fixture = _r01_fixture()
    try:
        alternate = replace(
            fixture.protocol,
            missing_outcome=minimal_instruction_identity((9370, 7)),
        )
        profiles = (
            fixture.alias_runtime,
            AliasRelationRuntime(fixture.closure, AliasResolutionSelector(alternate)),
        )

        def forbidden(*args, **kwargs):
            """局部查询和协议排序均不得扫描整类关系。"""
            raise AssertionError("禁止整类关系读取")

        monkeypatch.setattr(fixture.closure.consumer, "lookup_relation", forbidden)
        runtime = ProfiledAliasRelationRuntime(tuple(profiles[index] for index in order))
        expected = min((fixture.protocol, alternate), key=lambda item: item.stable_key())
        assert tuple(item.selector.protocol.stable_key() for item in runtime._profiled_runtimes) == tuple(
            sorted((fixture.protocol.stable_key(), alternate.stable_key())))
        assert runtime.selector.protocol.stable_key() == expected.stable_key()
        proposal = runtime.preview_surface(
            fixture.objects["d"], fixture.objects["branch"],
            budget=AliasRouteSearchBudget(20, 20, 20),
        )
        assert proposal.result.protocol == expected
        assert proposal.result.selected.value == fixture.objects["rep1"]
        assert proposal.result.selected.routes
    finally:
        fixture.close()


def test_local_profile_still_propagates_current_relation_validation_failure(monkeypatch):
    """排序不读无关事实，不得因此忽略本次真实路由的校验异常。"""
    fixture = _r01_fixture()
    try:
        runtime = ProfiledAliasRelationRuntime((fixture.alias_runtime,))

        def damaged(*args, **kwargs):
            """模拟当前局部 role/filler 的证据损坏。"""
            raise ValueError("当前局部关系损坏")

        monkeypatch.setattr(fixture.closure.consumer, "lookup_role_filler", damaged)
        with pytest.raises(ValueError, match="当前局部关系损坏"):
            runtime.preview_surface(
                fixture.objects["d"], fixture.objects["branch"],
                budget=AliasRouteSearchBudget(20, 20, 20),
            )
    finally:
        fixture.close()
