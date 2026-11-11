"""Focused contracts for integer discourse topic derivation."""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    TrainedInputStructureProjector,
)
from pure_integer_ai.experiments.trained_discourse_topic_runtime import (
    derive_discourse_topic_candidates,
)

from test_query_input_structure import _structure


def test_closed_relation_without_filler_members_is_not_a_topic():
    """Scoped composition may leave a closed syntax row without content."""
    projected = TrainedInputStructureProjector((_structure(),)).project(
        (33, 11, 22))
    assert projected.relation_candidates
    assert any(item.state == 2 for item in projected.relation_candidates)
    syntax_only = replace(
        projected,
        semantic_candidates=tuple(
            item for item in projected.semantic_candidates
            if item.projection_kind != PROJECTION_FILLER),
    )
    assert derive_discourse_topic_candidates(syntax_only) == ()
