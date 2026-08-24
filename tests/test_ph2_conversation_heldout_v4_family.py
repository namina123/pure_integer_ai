"""DLG-05 v4 新 family 构造点专项。"""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.conversation_heldout_v4_family import (
    build_v4_family,
    write_v4_family_artifacts,
)


def test_v4_family_builds_six_cases_from_same_execution_candidates():
    """六 case/十二 turn 必须保留请求、完整候选和互异 SourceRef fixture。"""
    family = build_v4_family()
    assert len(family.executions) == 12
    assert len(family.bundle.turns) == 12
    assert len(family.bundle.sources) == 6
    assert sum(len(item.candidates) for item in family.bundle.turns) == 12
    assert sum(
        not item.candidates for item in family.bundle.turns
        if item.case_key.components[-1] == 5
    ) == 2
    assert all(
        execution.query.request == turn.request
        for execution, turn in zip(family.executions, family.bundle.turns)
    )
    conflict = tuple(
        item for item in family.bundle.turns
        if item.case_key.components[-1] == 4
    )
    assert conflict and all(
        candidate.candidate.state.support and candidate.candidate.state.refute
        for turn in conflict for candidate in turn.candidates
    )


def test_v4_family_is_deterministic_and_keeps_input_output_boundaries():
    """重复构造逐整数一致，输入 Representation 不被答案 surface 替代。"""
    first = build_v4_family()
    second = build_v4_family()
    assert first.bundle.canonical_payload == second.bundle.canonical_payload
    assert first.freeze == second.freeze
    first_turn = first.bundle.turns[0]
    assert first_turn.representations
    assert first_turn.surface_representations
    assert first_turn.representations != first_turn.surface_representations


def test_v4_family_artifact_writer_is_non_overwriting(tmp_path):
    """fixture writer 只能幂等复写测试目录，且不得写 K 盘 source 根。"""
    family = build_v4_family()
    paths = write_v4_family_artifacts(family, tmp_path)
    assert all(path.exists() for path in paths.values())
    assert write_v4_family_artifacts(family, tmp_path) == paths
    with pytest.raises(ValueError, match="synthetic family"):
        write_v4_family_artifacts(family, "K:\\dlg05_v4_fixture_forbidden")
