"""FT18 construction-candidate dispatch over the frozen FT17 index."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    compile_authored_primitive_atomic_bridge_course,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    compile_authored_semantic_primitive_bridge_course,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY,
    QUESTION_CONSTRUCTION_INDEX_SHA256,
    build_raw_question_construction_index,
    indexed_question_feature_candidate_counts,
    lookup_indexed_alias_normalization_constructions,
    lookup_indexed_question_feature_constructions,
    reuse_indexed_alias_normalization_constructions,
    run_indexed_question_construction_registry_answer,
    run_indexed_question_construction_registry_answer_with_lookup,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    raw_question_feature_catalog,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_composition import (
    build_three_role_question_feature_composition,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    build_raw_question_feature_index,
    lookup_indexed_question_feature_entries,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    RawQuestionFeatureRegistryEntry,
    build_raw_question_feature_registry,
    run_raw_question_feature_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    build_learned_predicate_alias_bridge,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_generalization import (
    build_raw_question_generalization,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    build_implicit_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_three_role import (
    build_three_role_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_generalization import (
    build_w03_w04_w05_vertical_generalization_overlay,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_three_role import (
    build_w03_w04_w05_three_role_vertical_overlay,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "data/ph2"


def _compile_overlay(root, semantic, primitive_map, atomic, builder):
    base = compile_authored_semantic_primitive_bridge_course(
        DATA / semantic,
        root / "base",
    )
    donor = compile_authored_primitive_atomic_bridge_course(
        DATA / primitive_map,
        DATA / atomic,
        root / "donor",
    )
    return builder(base, donor)


def build_construction_index_fixture(tmp_path_factory, stage="ft18"):
    """Build the shared two-entry public registry for indexed tests."""
    two_root = tmp_path_factory.mktemp(f"{stage}_two_role")
    two_overlay = _compile_overlay(
        two_root,
        "authored_semantic_primitive_bridge_generalization_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_map_generalization_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_seed_generalization_v1.jsonl.sample",
        build_w03_w04_w05_vertical_generalization_overlay,
    )
    two_explicit = build_raw_question_generalization(
        two_overlay,
        DATA / "authored_vertical_question_cause_generalization_v1.jsonl.sample",
        DATA / "authored_vertical_question_effect_generalization_v1.jsonl.sample",
    )
    two_catalog = raw_question_feature_catalog(two_explicit)
    two_entry = RawQuestionFeatureRegistryEntry(
        two_catalog,
        build_learned_predicate_alias_bridge(two_catalog),
        build_implicit_question_bundle(
            two_catalog,
            DATA / "authored_vertical_question_implicit_reason_v1.jsonl.sample",
            DATA / "authored_vertical_question_implicit_result_v1.jsonl.sample",
        ),
    )

    three_root = tmp_path_factory.mktemp(f"{stage}_three_role")
    three_overlay = _compile_overlay(
        three_root,
        "authored_semantic_primitive_bridge_three_role_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_map_three_role_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_seed_three_role_v1.jsonl.sample",
        build_w03_w04_w05_three_role_vertical_overlay,
    )
    three_explicit = build_three_role_question_bundle(
        three_overlay,
        DATA / "authored_vertical_question_three_role_actor_v1.jsonl.sample",
        DATA / "authored_vertical_question_three_role_location_v1.jsonl.sample",
    )
    three_composition = build_three_role_question_feature_composition(
        three_explicit,
        DATA / (
            "authored_vertical_question_three_role_implicit_actor_v1.jsonl.sample"),
        DATA / (
            "authored_vertical_question_three_role_implicit_location_v1.jsonl.sample"),
    )
    three_entry = RawQuestionFeatureRegistryEntry(
        three_composition.feature_catalog,
        three_composition.alias_bridge,
        three_composition.implicit_bundle,
    )
    registry = build_raw_question_feature_registry((two_entry, three_entry))
    feature_index = build_raw_question_feature_index(registry)
    return (
        build_raw_question_construction_index(feature_index),
        two_entry,
        three_entry,
    )


@pytest.fixture(scope="module")
def construction_index(tmp_path_factory):
    return build_construction_index_fixture(tmp_path_factory)


def _learned_alias_surface(entry, construction):
    predicate_ordinal = next(
        ordinal for ordinal, segment in enumerate(construction.segments)
        if segment.kind == "PREDICATE"
    )
    proposition_key = construction.vertical_result.link.proposition_key
    alias = next(
        route.alias_surface for route in entry.alias_bridge.routes
        if route.proposition_key == proposition_key
    )
    return "".join(
        alias if ordinal == predicate_ordinal else segment.surface
        for ordinal, segment in enumerate(construction.segments)
    )


def _unlearned_alias_surface(construction):
    predicate_ordinal = next(
        ordinal for ordinal, segment in enumerate(construction.segments)
        if segment.kind == "PREDICATE"
    )
    return "".join(
        "未学习替代" if ordinal == predicate_ordinal else segment.surface
        for ordinal, segment in enumerate(construction.segments)
    )


def _assert_scan_equal(index, request):
    scan = run_raw_question_feature_registry_answer(index.registry, request)
    indexed = run_indexed_question_construction_registry_answer(index, request)
    assert indexed.to_dict() == scan.to_dict()
    assert indexed.sha256() == scan.sha256()
    return indexed


def _candidate(index, phase, request, entry_sha):
    return next(
        item for item in lookup_indexed_question_feature_constructions(
            index, phase, request)
        if item.entry_sha256 == entry_sha
    )


def _project_rows(rows):
    return tuple(
        (
            row.question_surface,
            tuple(
                (
                    posting.source_record_key,
                    tuple(
                        item.entry_sha256 for item in posting.candidates),
                )
                for posting in row.postings
            ),
        )
        for row in rows
    )


def _feature_rows(rows):
    return tuple(
        (
            row.question_surface,
            tuple(
                (posting.source_record_key, posting.entry_sha256s)
                for posting in row.postings
            ),
        )
        for row in rows
    )


def test_construction_index_is_derived_and_frozen(construction_index) -> None:
    index, _, _ = construction_index
    assert QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY == (
        ("index_source", "FT17_INDEX_AND_LEARNED_CONSTRUCTIONS"),
        ("posting_identity", "REGISTRY_ENTRY_AND_CONSTRUCTION_SHA256"),
        ("candidate_effect", "NARROW_ENTRY_AND_INTERNAL_CONSTRUCTION"),
        ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
        ("result_projection", "BYTE_IDENTICAL_TO_FT16_SCAN_RUNTIME"),
        ("normalization", "ALIAS_REUSES_INDEXED_EXACT_CONSTRUCTIONS"),
        ("handwritten_dispatch", "FORBIDDEN"),
    )
    assert index.identity_sha256 == QUESTION_CONSTRUCTION_INDEX_SHA256
    assert index.sha256() == QUESTION_CONSTRUCTION_INDEX_SHA256
    assert index.feature_index.identity_sha256 == (
        "908786f4da77414a5c7728d01f8c2c1528c23f4787a1efcc068b166f70be095b")
    for construction_rows, feature_rows in zip(
            (index.exact_rows, index.alias_rows, index.implicit_rows),
            (
                index.feature_index.exact_rows,
                index.feature_index.alias_rows,
                index.feature_index.implicit_rows,
            )):
        assert _project_rows(construction_rows) == _feature_rows(feature_rows)
    assert tuple(
        (
            row.sort_key(),
            tuple(
                (
                    posting.source_record_key,
                    tuple(
                        item.entry_sha256 for item in posting.candidates),
                )
                for posting in row.postings
            ),
        )
        for row in index.alias_frame_rows
    ) == tuple(
        (
            row.sort_key(),
            tuple(
                (posting.source_record_key, posting.entry_sha256s)
                for posting in row.postings
            ),
        )
        for row in index.feature_index.alias_frame_rows
    )


def test_every_learned_construction_has_an_exact_alias_or_implicit_posting(
        construction_index) -> None:
    index, two_entry, three_entry = construction_index
    for entry in (two_entry, three_entry):
        entry_sha = entry.sha256()
        for construction in entry.feature_catalog.catalog:
            exact_request = RawQuestionRequest(
                construction.question_surface,
                construction.source_record_key,
            )
            alias_request = RawQuestionRequest(
                _learned_alias_surface(entry, construction),
                construction.source_record_key,
            )
            assert construction.sha256() in _candidate(
                index, "EXACT", exact_request, entry_sha).construction_sha256s
            assert construction.sha256() in _candidate(
                index, "ALIAS", alias_request, entry_sha).construction_sha256s
        for construction in entry.implicit_bundle.catalog:
            request = RawQuestionRequest(
                construction.question_surface,
                construction.source_record_key,
            )
            assert construction.sha256() in _candidate(
                index, "IMPLICIT", request, entry_sha).construction_sha256s


def test_all_learned_routes_match_ft16_byte_for_byte(
        construction_index) -> None:
    index, two_entry, three_entry = construction_index
    for entry in (two_entry, three_entry):
        for construction in entry.feature_catalog.catalog:
            for surface in (
                    construction.question_surface,
                    _learned_alias_surface(entry, construction)):
                result = _assert_scan_equal(index, RawQuestionRequest(
                    surface,
                    construction.source_record_key,
                ))
                assert result.status == "ANSWER"
        for construction in entry.implicit_bundle.catalog:
            result = _assert_scan_equal(index, RawQuestionRequest(
                construction.question_surface,
                construction.source_record_key,
            ))
            assert result.status == "ANSWER"


def test_unlearned_alias_wrong_source_missing_structure_and_repetition(
        construction_index) -> None:
    index, two_entry, three_entry = construction_index
    construction = two_entry.feature_catalog.catalog[0]
    unlearned = RawQuestionRequest(
        _unlearned_alias_surface(construction),
        construction.source_record_key,
    )
    result = _assert_scan_equal(index, unlearned)
    assert result.status == "UNKNOWN"
    assert any(
        trace.alias_result is not None and trace.alias_result.matches
        for trace in result.traces
    )
    wrong_source = three_entry.feature_catalog.catalog[0].source_record_key
    wrong = RawQuestionRequest(construction.question_surface, wrong_source)
    missing = RawQuestionRequest("未学习的构式索引问题？")
    for request in (wrong, missing):
        assert _assert_scan_equal(index, request).status == "UNKNOWN"
    assert (
        run_indexed_question_construction_registry_answer(
            index, missing).sha256()
        == run_indexed_question_construction_registry_answer(
            index, missing).sha256()
    )


@pytest.mark.parametrize("phase", ("EXACT", "ALIAS", "IMPLICIT"))
def test_construction_entries_equal_ft17_candidates_and_retain_source_scope(
        construction_index, phase) -> None:
    index, two_entry, _ = construction_index
    if phase == "EXACT":
        construction = two_entry.feature_catalog.catalog[0]
        surface = construction.question_surface
    elif phase == "ALIAS":
        construction = two_entry.feature_catalog.catalog[0]
        surface = _learned_alias_surface(two_entry, construction)
    else:
        construction = two_entry.implicit_bundle.catalog[0]
        surface = construction.question_surface
    unscoped_request = RawQuestionRequest(surface)
    scoped_request = RawQuestionRequest(surface, construction.source_record_key)
    unscoped = lookup_indexed_question_feature_constructions(
        index, phase, unscoped_request)
    scoped = lookup_indexed_question_feature_constructions(
        index, phase, scoped_request)
    assert tuple(item.entry_sha256 for item in unscoped) == (
        lookup_indexed_question_feature_entries(
            index.feature_index, phase, unscoped_request))
    assert tuple(item.entry_sha256 for item in scoped) == (
        lookup_indexed_question_feature_entries(
            index.feature_index, phase, scoped_request))
    assert sum(len(item.constructions) for item in scoped) <= sum(
        len(item.constructions) for item in unscoped)


def test_actual_execution_narrows_entries_and_internal_constructions(
        construction_index, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_question_construction_index as runtime,
    )

    index, two_entry, _ = construction_index
    explicit = two_entry.feature_catalog.catalog[0]
    implicit = two_entry.implicit_bundle.catalog[0]
    cases = (
        (
            "EXACT",
            RawQuestionRequest(
                explicit.question_surface, explicit.source_record_key),
            "run_raw_question_feature_candidate_answer",
            2,
        ),
        (
            "ALIAS",
            RawQuestionRequest(
                _learned_alias_surface(two_entry, explicit),
                explicit.source_record_key,
            ),
            "continue_question_feature_predicate_alias_candidate_answer",
            4,
        ),
        (
            "IMPLICIT",
            RawQuestionRequest(
                implicit.question_surface, implicit.source_record_key),
            "continue_implicit_predicate_question_candidate_answer",
            4,
        ),
    )
    total_entries = len(index.registry.entries)
    total_constructions = sum(
        len(entry.feature_catalog.catalog) + len(entry.implicit_bundle.catalog)
        for entry in index.registry.entries
    )
    for phase, request, attribute, constructions_ordinal in cases:
        expected_counts = indexed_question_feature_candidate_counts(
            index, phase, request)
        calls = []
        original = getattr(runtime, attribute)

        def probe(*args, _original=original, **kwargs):
            calls.append(len(args[constructions_ordinal]))
            return _original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, attribute, probe)
            result = runtime.run_indexed_question_construction_registry_answer(
                index, request)
        assert result.status == "ANSWER"
        assert len(calls) == expected_counts[0] < total_entries
        assert sum(calls) == expected_counts[1] < total_constructions
        assert all(value < len(two_entry.feature_catalog.catalog) for value in calls)


def test_alias_normalization_reuses_candidates_without_exact_relookup(
        construction_index) -> None:
    index, two_entry, _ = construction_index
    construction = two_entry.feature_catalog.catalog[0]
    cases = (
        (
            RawQuestionRequest(
                _learned_alias_surface(two_entry, construction),
                construction.source_record_key,
            ),
            "ANSWER",
            ("EXACT", "ALIAS"),
        ),
        (
            RawQuestionRequest(
                _unlearned_alias_surface(construction),
                construction.source_record_key,
            ),
            "UNKNOWN",
            ("EXACT", "ALIAS", "IMPLICIT"),
        ),
    )
    for request, status, expected_phases in cases:
        phases = []

        def lookup(source, phase, current_request):
            phases.append(phase)
            return lookup_indexed_question_feature_constructions(
                source, phase, current_request)

        result = run_indexed_question_construction_registry_answer_with_lookup(
            index,
            request,
            index,
            lookup,
        )
        assert result.status == status
        assert tuple(phases) == expected_phases
        assert phases.count("EXACT") == 1

    alias_request = cases[0][0]
    alias_candidate = _candidate(
        index,
        "ALIAS",
        alias_request,
        two_entry.sha256(),
    )

    def forbidden_lookup(*args, **kwargs):
        raise AssertionError("alias normalization performed an exact relookup")

    assert lookup_indexed_alias_normalization_constructions(
        index,
        two_entry.sha256(),
        alias_request,
        alias_candidate.constructions,
        index,
        forbidden_lookup,
    ) == alias_candidate.normalization_constructions


def test_alias_candidates_are_exact_catalog_owned_and_source_compatible(
        construction_index) -> None:
    index, _, _ = construction_index
    exact_by_entry = {
        entry.sha256(): {
            item.sha256(): item for item in entry.feature_catalog.catalog
        }
        for entry in index.registry.entries
    }
    for rows in (index.alias_rows, index.alias_frame_rows):
        for row in rows:
            for posting in row.postings:
                for candidate in posting.candidates:
                    assert all(
                        exact_by_entry[candidate.entry_sha256][item.sha256()]
                        == item
                        for item in candidate.constructions
                    )
                    assert all(
                        item.source_record_key == posting.source_record_key
                        for item in candidate.constructions
                    )
                    request = RawQuestionRequest(
                        row.question_surface
                        if hasattr(row, "question_surface")
                        else f"{row.prefix_surface}替代{row.suffix_surface}",
                        posting.source_record_key,
                    )
                    target_surfaces = {
                        item.question_surface
                        for item in candidate.constructions
                    }
                    expected_normalizations = tuple(sorted(
                        (
                            item
                            for item in exact_by_entry[
                                candidate.entry_sha256].values()
                            if item.source_record_key
                            == posting.source_record_key
                            and item.question_surface in target_surfaces
                        ),
                        key=lambda item: item.sha256(),
                    ))
                    assert candidate.normalization_constructions == (
                        expected_normalizations)
                    assert reuse_indexed_alias_normalization_constructions(
                        request, candidate) == expected_normalizations

    row = index.alias_rows[0]
    posting = row.postings[0]
    bad_posting = replace(
        posting,
        source_record_key=(
            *posting.source_record_key[:-1],
            posting.source_record_key[-1] + 1,
        ),
    )
    bad_row = replace(
        row,
        postings=tuple(sorted(
            (bad_posting, *row.postings[1:]),
            key=lambda item: item.source_record_key,
        )),
    )
    with pytest.raises(ValueError, match="escaped its SourceRef"):
        replace(index, alias_rows=(bad_row, *index.alias_rows[1:]))


def test_same_surface_ambiguity_and_overlapping_entries_match_ft16(
        construction_index) -> None:
    index, two_entry, three_entry = construction_index
    shared = two_entry.feature_catalog.catalog[0]
    collision_pattern = replace(
        shared.pattern,
        construction_id=f"{shared.pattern.construction_id}_COLLISION",
    )
    collision = replace(shared, pattern=collision_pattern)
    collision_catalog = replace(
        two_entry.feature_catalog,
        patterns=tuple(sorted(
            (*two_entry.feature_catalog.patterns, collision_pattern),
            key=lambda item: item.sha256(),
        )),
        catalog=tuple(sorted(
            (shared, collision), key=lambda item: item.sha256())),
    )
    collision_entry = RawQuestionFeatureRegistryEntry(
        collision_catalog,
        two_entry.alias_bridge,
        replace(
            two_entry.implicit_bundle,
            explicit_catalog=collision_catalog,
        ),
    )
    collision_registry = build_raw_question_feature_registry(
        (collision_entry, three_entry))
    collision_index = build_raw_question_construction_index(
        build_raw_question_feature_index(collision_registry))
    collision_result = _assert_scan_equal(
        collision_index,
        RawQuestionRequest(shared.question_surface, shared.source_record_key),
    )
    assert collision_result.status == "CLARIFY"
    assert len(next(
        trace.exact_result.matched_construction_sha256s
        for trace in collision_result.traces
        if trace.entry_sha256 == collision_entry.sha256()
    )) == 2
    collision_alias = _assert_scan_equal(
        collision_index,
        RawQuestionRequest(
            _learned_alias_surface(two_entry, shared),
            shared.source_record_key,
        ),
    )
    assert collision_alias.status == "CLARIFY"
    assert collision_alias.decisive_phase == "ALIAS"

    projected_catalog = replace(two_entry.feature_catalog, catalog=(shared,))
    projected_entry = RawQuestionFeatureRegistryEntry(
        projected_catalog,
        two_entry.alias_bridge,
        replace(
            two_entry.implicit_bundle,
            explicit_catalog=projected_catalog,
        ),
    )
    overlap_registry = build_raw_question_feature_registry(
        (two_entry, projected_entry))
    overlap_index = build_raw_question_construction_index(
        build_raw_question_feature_index(overlap_registry))
    overlap = _assert_scan_equal(
        overlap_index,
        RawQuestionRequest(shared.question_surface, shared.source_record_key),
    )
    assert overlap.status == "ANSWER"
    assert len(overlap.interpretations) == 1
    assert sum(
        trace.exact_result.status == "ANSWER" for trace in overlap.traces
    ) == 2


def test_runtime_contains_no_question_or_bundle_dispatch_table() -> None:
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_question_construction_index.py"
    )).read_text(encoding="utf-8")
    for value in ("暴雨", "寒潮", "山区", "桥面", "什么", "哪里"):
        assert value not in source
    assert "bundle_name" not in source
