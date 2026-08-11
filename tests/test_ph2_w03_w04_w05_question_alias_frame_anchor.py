"""FT19 prefix/suffix anchor dispatch over every FT18 alias frame."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor import (
    QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY,
    QUESTION_ALIAS_FRAME_ANCHOR_SHA256,
    build_raw_question_alias_frame_anchor_index,
    indexed_alias_frame_anchor_candidate_counts,
    lookup_indexed_alias_frame_anchor_constructions,
    lookup_indexed_alias_frame_ordinals,
    run_indexed_alias_frame_anchor_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    build_raw_question_construction_index,
    lookup_indexed_question_feature_constructions,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    build_raw_question_feature_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    RawQuestionFeatureRegistryEntry,
    build_raw_question_feature_registry,
    run_raw_question_feature_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from tests.test_ph2_w03_w04_w05_question_construction_index import (
    _learned_alias_surface,
    _unlearned_alias_surface,
    build_construction_index_fixture,
)


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def anchor_index(tmp_path_factory):
    construction_index, two_entry, three_entry = (
        build_construction_index_fixture(tmp_path_factory, "ft19"))
    return (
        build_raw_question_alias_frame_anchor_index(construction_index),
        two_entry,
        three_entry,
    )


def _assert_scan_equal(index, request):
    scan = run_raw_question_feature_registry_answer(
        index.construction_index.registry,
        request,
    )
    anchored = run_indexed_alias_frame_anchor_registry_answer(index, request)
    assert anchored.to_dict() == scan.to_dict()
    assert anchored.sha256() == scan.sha256()
    return anchored


def _published_frames(nodes):
    return tuple(sorted(
        terminal.frame_ordinal
        for node in nodes
        for terminal in node.terminals
    ))


def test_anchor_index_is_derived_complete_and_frozen(anchor_index) -> None:
    index, _, _ = anchor_index
    assert QUESTION_ALIAS_FRAME_ANCHOR_EXPRESSION_BOUNDARY == (
        ("index_source", "FT18_ALIAS_FRAMES_ONLY"),
        ("anchor_structure", "PREFIX_TRIE_AND_REVERSED_SUFFIX_TRIE"),
        ("candidate_operation", "PREFIX_SUFFIX_FRAME_INTERSECTION"),
        ("source_binding", "FILTERED_AT_TERMINAL_POSTING"),
        ("structural_unknown", "PRESERVED_FOR_UNLEARNED_ALIAS_SURFACES"),
        ("result_projection", "BYTE_IDENTICAL_TO_FT16_SCAN_RUNTIME"),
        ("handwritten_dispatch", "FORBIDDEN"),
    )
    assert index.identity_sha256 == QUESTION_ALIAS_FRAME_ANCHOR_SHA256
    assert index.sha256() == QUESTION_ALIAS_FRAME_ANCHOR_SHA256
    assert index.construction_index.identity_sha256 == (
        "0d4495b612e5d6e54fa47f1066ad48543fd5bbb00ffc08d04a67de2fc2067ab8")
    expected = tuple(range(len(index.construction_index.alias_frame_rows)))
    assert _published_frames(index.prefix_nodes) == expected
    assert _published_frames(index.suffix_nodes) == expected


def test_anchor_terminal_path_validation_is_fail_closed(anchor_index) -> None:
    index, _, _ = anchor_index
    field, nodes = next(
        (field, getattr(index, field))
        for field in ("prefix_nodes", "suffix_nodes")
        if any(node.terminals for node in getattr(index, field)[1:])
    )
    source_ordinal = next(
        ordinal for ordinal, node in enumerate(nodes[1:], start=1)
        if node.terminals
    )
    terminal = nodes[source_ordinal].terminals[0]
    changed = list(nodes)
    changed[0] = replace(
        changed[0],
        terminals=tuple(sorted(
            (*changed[0].terminals, terminal),
            key=lambda item: item.frame_ordinal,
        )),
    )
    changed[source_ordinal] = replace(
        changed[source_ordinal],
        terminals=tuple(
            item for item in changed[source_ordinal].terminals
            if item != terminal
        ),
    )
    with pytest.raises(ValueError, match="terminal path drifted"):
        replace(index, **{field: tuple(changed)})


def test_anchor_candidates_equal_ft18_for_every_learned_and_unlearned_alias(
        anchor_index) -> None:
    index, two_entry, three_entry = anchor_index
    for entry in (two_entry, three_entry):
        for construction in entry.feature_catalog.catalog:
            for surface in (
                    _learned_alias_surface(entry, construction),
                    _unlearned_alias_surface(construction)):
                for source in (None, construction.source_record_key):
                    request = RawQuestionRequest(surface, source)
                    anchored = lookup_indexed_alias_frame_anchor_constructions(
                        index, "ALIAS", request)
                    scanned = lookup_indexed_question_feature_constructions(
                        index.construction_index, "ALIAS", request)
                    assert anchored == scanned
                    assert lookup_indexed_alias_frame_ordinals(index, request)


def test_exact_implicit_wrong_source_and_missing_delegate_without_drift(
        anchor_index) -> None:
    index, two_entry, three_entry = anchor_index
    exact = two_entry.feature_catalog.catalog[0]
    implicit = three_entry.implicit_bundle.catalog[0]
    requests = (
        ("EXACT", RawQuestionRequest(
            exact.question_surface, exact.source_record_key)),
        ("IMPLICIT", RawQuestionRequest(
            implicit.question_surface, implicit.source_record_key)),
        ("ALIAS", RawQuestionRequest(
            _learned_alias_surface(two_entry, exact),
            implicit.source_record_key,
        )),
        ("ALIAS", RawQuestionRequest("未学习的 anchor 结构？")),
    )
    for phase, request in requests:
        anchored = lookup_indexed_alias_frame_anchor_constructions(
            index, phase, request)
        scanned = lookup_indexed_question_feature_constructions(
            index.construction_index, phase, request)
        assert anchored == scanned


def test_alias_runtime_matches_ft16_complete_traces(anchor_index) -> None:
    index, two_entry, three_entry = anchor_index
    for entry in (two_entry, three_entry):
        for construction in entry.feature_catalog.catalog:
            learned = _assert_scan_equal(index, RawQuestionRequest(
                _learned_alias_surface(entry, construction),
                construction.source_record_key,
            ))
            assert learned.status == "ANSWER"
        construction = entry.feature_catalog.catalog[0]
        unlearned = _assert_scan_equal(index, RawQuestionRequest(
            _unlearned_alias_surface(construction),
            construction.source_record_key,
        ))
        assert unlearned.status == "UNKNOWN"
        assert any(
            trace.alias_result is not None and trace.alias_result.matches
            for trace in unlearned.traces
        )
    missing = RawQuestionRequest("未学习的 anchor 回答结构？")
    assert _assert_scan_equal(index, missing).status == "UNKNOWN"
    assert (
        run_indexed_alias_frame_anchor_registry_answer(index, missing).sha256()
        == run_indexed_alias_frame_anchor_registry_answer(index, missing).sha256()
    )


def test_actual_structural_match_visits_only_anchor_intersection(
        anchor_index, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_question_construction_index as ft18_runtime,
    )

    index, two_entry, _ = anchor_index
    construction = two_entry.feature_catalog.catalog[0]
    request = RawQuestionRequest(
        _unlearned_alias_surface(construction),
        construction.source_record_key,
    )
    expected_ordinals = lookup_indexed_alias_frame_ordinals(index, request)
    expected_counts = indexed_alias_frame_anchor_candidate_counts(
        index, request)
    calls = []
    original = ft18_runtime._matches_alias_frame

    def probe(row, surface):
        calls.append(row.sort_key())
        return original(row, surface)

    monkeypatch.setattr(ft18_runtime, "_matches_alias_frame", probe)
    candidates = lookup_indexed_alias_frame_anchor_constructions(
        index, "ALIAS", request)
    assert len(calls) == len(expected_ordinals) == expected_counts[0]
    assert len(calls) < len(index.construction_index.alias_frame_rows)
    assert len(candidates) == expected_counts[1]
    assert sum(len(item.constructions) for item in candidates) == (
        expected_counts[2])


def test_source_scope_can_only_reduce_visited_frames(anchor_index) -> None:
    index, two_entry, _ = anchor_index
    construction = two_entry.feature_catalog.catalog[0]
    surface = _learned_alias_surface(two_entry, construction)
    unscoped = lookup_indexed_alias_frame_ordinals(
        index,
        RawQuestionRequest(surface),
    )
    scoped = lookup_indexed_alias_frame_ordinals(
        index,
        RawQuestionRequest(surface, construction.source_record_key),
    )
    assert scoped
    assert set(scoped).issubset(unscoped)


def test_real_overlapping_alias_entries_still_converge(anchor_index) -> None:
    _, two_entry, _ = anchor_index
    shared = two_entry.feature_catalog.catalog[0]
    projected_catalog = replace(two_entry.feature_catalog, catalog=(shared,))
    projected_entry = RawQuestionFeatureRegistryEntry(
        projected_catalog,
        two_entry.alias_bridge,
        replace(
            two_entry.implicit_bundle,
            explicit_catalog=projected_catalog,
        ),
    )
    registry = build_raw_question_feature_registry(
        (two_entry, projected_entry))
    index = build_raw_question_alias_frame_anchor_index(
        build_raw_question_construction_index(
            build_raw_question_feature_index(registry)))
    result = _assert_scan_equal(index, RawQuestionRequest(
        _learned_alias_surface(two_entry, shared),
        shared.source_record_key,
    ))
    assert result.status == "ANSWER"
    assert len(result.interpretations) == 1
    assert sum(
        trace.alias_result is not None
        and trace.alias_result.status == "ANSWER"
        for trace in result.traces
    ) == 2


def test_anchor_runtime_contains_no_question_alias_or_bundle_table() -> None:
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_question_alias_frame_anchor.py"
    )).read_text(encoding="utf-8")
    for value in ("暴雨", "寒潮", "山区", "桥面", "什么", "哪里"):
        assert value not in source
    assert "bundle_name" not in source
