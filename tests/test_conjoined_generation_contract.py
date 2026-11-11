"""Predicate guards keep compound generation from relabeling reordered facts."""
from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.conjoined_relation_generation_course import (
    compile_train_conjoined_generation,
)
from pure_integer_ai.experiments.integer_relation_connected_course import _seed
from pure_integer_ai.experiments.ph2_authored_relation_compile import authored_relation_identity
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    PART_CLAIM_PREDICATE, PART_CLAIM_ROLE, PART_LITERAL, MultiClaimCompositionVariant,
    TrainedMultiClaimComposer, materialize_multi_claim_composition_graph,
    read_multi_claim_composition_graph,
)


def _seeds():
    return tuple(_seed(row, index) for index, row in enumerate((
        {"anchor": "导致", "case_id": "a", "relation": "CAUSES",
         "sample_role": "support", "seed_id": "a", "split": "train",
         "surface": "青阀导致赤灯。", "terms": [["青阀", "event", 1], ["赤灯", "event", 2]]},
        {"anchor": "之前", "case_id": "b", "relation": "EVENT_BEFORE",
         "sample_role": "support", "seed_id": "b", "split": "train",
         "surface": "赤灯发生在蓝铃之前。", "terms": [["赤灯", "event", 2], ["蓝铃", "event", 3]]},
    ), 1))


def _variant(seeds=None, *, variant_id=1):
    seeds = seeds or _seeds()
    return compile_train_conjoined_generation(
        seeds, "；".join(seed.surface for seed in seeds), separator="；",
        variant_id=variant_id, source_identity=(92034, 100),
        license_identity=tuple(map(ord, "CC0-1.0")))


def _components():
    return tuple((None, SimpleNamespace(
        predicate=authored_relation_identity(seed.relation_kind),
        bindings=tuple(SimpleNamespace(surface=value) for value in values)), None)
        for seed, values in zip(_seeds(), (("新阀", "新灯"), ("新灯", "新铃"))))


def test_guarded_source_frame_rebinds_entities_and_rejects_clause_reversal(tmp_path):
    graph = tmp_path / "guarded-composer.sqlite3"
    variant = _variant()
    materialize_multi_claim_composition_graph((variant,), graph)
    restored, metadata = read_multi_claim_composition_graph(graph)
    assert restored == (variant,) and metadata["format_version"] == 2
    composer = TrainedMultiClaimComposer(graph, SimpleNamespace())
    components = _components()
    predicates = tuple(item[1].predicate.stable_key() for item in components)
    assert composer.supports_predicates(predicates)
    assert not composer.supports_predicates(tuple(reversed(predicates)))
    state = SimpleNamespace(evidence_closed=1, conflict_open=0, required_slots_open=0)
    assert composer.generate_composition(components, state).surface == "新阀导致新灯。；新灯发生在新铃之前。"
    with pytest.raises(ValueError, match="not unique"):
        composer.generate_composition(tuple(reversed(components)), state)
    assert not {ord(value) for value in "青阀赤灯蓝铃"} & {
        value for part in variant.parts if part[0] == PART_LITERAL for value in part[1]}


def test_trained_reversed_variant_is_selected_by_its_current_predicates(tmp_path):
    graph = tmp_path / "two-orders.sqlite3"
    materialize_multi_claim_composition_graph(
        (_variant(), _variant(tuple(reversed(_seeds())), variant_id=2)), graph)
    composer = TrainedMultiClaimComposer(graph, SimpleNamespace())
    state = SimpleNamespace(evidence_closed=1, conflict_open=0, required_slots_open=0)
    assert composer.generate_composition(tuple(reversed(_components())), state).surface == (
        "新灯发生在新铃之前。；新阀导致新灯。")


def test_missing_or_corrupt_predicate_part_fails_cold_restore(tmp_path):
    graph = tmp_path / "incomplete.sqlite3"
    materialize_multi_claim_composition_graph((_variant(),), graph)
    with sqlite3.connect(graph) as connection:
        connection.execute("DELETE FROM multi_claim_composition_part WHERE part_kind=3 AND value_ordinal=1")
    with pytest.raises(ValueError, match="incomplete|hash"):
        read_multi_claim_composition_graph(graph)


def test_partial_or_duplicate_guard_is_not_a_valid_variant():
    variant = _variant()
    with pytest.raises(ValueError, match="every claim exactly once"):
        replace(variant, parts=tuple(part for part in variant.parts
                                     if part[0] != PART_CLAIM_PREDICATE or part[1] == 1))
    with pytest.raises(ValueError, match="every claim exactly once"):
        replace(variant, parts=(*variant.parts, variant.parts[0]))


def test_legacy_count_only_frame_cannot_authorize_scoped_language(tmp_path):
    graph = tmp_path / "legacy.sqlite3"
    variant = replace(_variant(), parts=tuple(part for part in _variant().parts
                                              if part[0] != PART_CLAIM_PREDICATE))
    materialize_multi_claim_composition_graph((variant,), graph)
    composer = TrainedMultiClaimComposer(graph, SimpleNamespace())
    assert not composer.supports_predicates(tuple(
        item[1].predicate.stable_key() for item in _components()))


def test_generator_compiler_rejects_body_or_non_train_source():
    seeds = _seeds()
    with pytest.raises(ValueError, match="differs"):
        compile_train_conjoined_generation(
            seeds, "外来答案", separator="；", variant_id=1,
            source_identity=(92034, 100), license_identity=tuple(map(ord, "CC0-1.0")))
    with pytest.raises(ValueError, match="licensed train teachers"):
        _variant((replace(seeds[0], split="held_out", label_owner="evaluator"), seeds[1]))
