"""FT20 candidate-only dispatch and on-demand complete audit projection."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor import (
    build_raw_question_alias_frame_anchor_index,
    run_indexed_alias_frame_anchor_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    build_raw_question_construction_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    build_raw_question_feature_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    RawQuestionFeatureRegistryEntry,
    build_raw_question_feature_registry,
    run_raw_question_feature_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch import (
    QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY,
    QUESTION_SPARSE_DISPATCH_SHA256,
    RawQuestionSparseDecision,
    RawQuestionSparsePhaseVisit,
    build_raw_question_sparse_dispatch_index,
    project_sparse_question_dispatch_audit,
    run_sparse_question_dispatch,
    run_sparse_question_dispatch_registry_answer,
    sparse_question_dispatch_probe,
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
def sparse_index(tmp_path_factory):
    construction_index, two_entry, three_entry = (
        build_construction_index_fixture(tmp_path_factory, "ft20"))
    anchor_index = build_raw_question_alias_frame_anchor_index(
        construction_index)
    return (
        build_raw_question_sparse_dispatch_index(anchor_index),
        two_entry,
        three_entry,
    )


def _assert_complete_equal(index, request):
    registry = index.anchor_index.construction_index.registry
    scan = run_raw_question_feature_registry_answer(registry, request)
    anchored = run_indexed_alias_frame_anchor_registry_answer(
        index.anchor_index, request)
    record = run_sparse_question_dispatch(index, request)
    projected = project_sparse_question_dispatch_audit(index, record)
    facade = run_sparse_question_dispatch_registry_answer(index, request)
    assert projected.to_dict() == scan.to_dict() == anchored.to_dict()
    assert projected.sha256() == scan.sha256() == anchored.sha256()
    assert facade.to_dict() == projected.to_dict()
    assert facade.sha256() == projected.sha256()
    return record, projected


def _build_sparse_index(registry):
    return build_raw_question_sparse_dispatch_index(
        build_raw_question_alias_frame_anchor_index(
            build_raw_question_construction_index(
                build_raw_question_feature_index(registry))))


def test_sparse_dispatch_index_is_derived_complete_and_frozen(
        sparse_index) -> None:
    index, _, _ = sparse_index
    assert QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY == (
        ("index_source", "FT19_ANCHOR_AND_FT16_REGISTRY"),
        ("hot_execution", "CANDIDATE_ENTRIES_ONLY"),
        ("missing_unknowns", "OMITTED_FROM_HOT_RECORD"),
        ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
        ("audit_projection", "ON_DEMAND_BYTE_IDENTICAL_TO_FT16"),
        ("source_binding", "PRESERVED"),
        ("wall_clock_gate", "FORBIDDEN"),
        ("handwritten_dispatch", "FORBIDDEN"),
    )
    assert index.identity_sha256 == QUESTION_SPARSE_DISPATCH_SHA256
    assert index.sha256() == QUESTION_SPARSE_DISPATCH_SHA256
    assert index.anchor_index.identity_sha256 == (
        "af4dcb9afb8ee0a96ff5883e057afed9527fd118e21bd5f93dea8cae9583dad2")
    assert tuple(item.entry_sha256 for item in index.entries) == tuple(
        entry.sha256()
        for entry in index.anchor_index.construction_index.registry.entries
    )


def test_exact_alias_and_implicit_projection_remain_ft16_byte_identical(
        sparse_index) -> None:
    index, two_entry, three_entry = sparse_index
    for entry in (two_entry, three_entry):
        explicit = entry.feature_catalog.catalog[0]
        implicit = entry.implicit_bundle.catalog[0]
        cases = (
            ("EXACT", RawQuestionRequest(
                explicit.question_surface, explicit.source_record_key)),
            ("ALIAS", RawQuestionRequest(
                _learned_alias_surface(entry, explicit),
                explicit.source_record_key,
            )),
            ("IMPLICIT", RawQuestionRequest(
                implicit.question_surface, implicit.source_record_key)),
        )
        for phase, request in cases:
            record, projected = _assert_complete_equal(index, request)
            assert record.decision.status == "ANSWER"
            assert record.decision.decisive_phase == phase
            assert projected.status == "ANSWER"
            assert tuple(item.phase for item in record.phase_visits) == (
                ("EXACT", "ALIAS", "IMPLICIT")
                [:len(record.phase_visits)])


def test_hot_path_omits_noncandidate_unknowns_and_probe_is_deterministic(
        sparse_index, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_question_sparse_dispatch as runtime,
    )

    index, two_entry, _ = sparse_index
    registry_count = len(index.entries)
    exact = two_entry.feature_catalog.catalog[0]
    implicit = two_entry.implicit_bundle.catalog[0]
    requests = (
        RawQuestionRequest(exact.question_surface, exact.source_record_key),
        RawQuestionRequest(
            _learned_alias_surface(two_entry, exact),
            exact.source_record_key,
        ),
        RawQuestionRequest(
            implicit.question_surface, implicit.source_record_key),
    )
    for request in requests:
        record = run_sparse_question_dispatch(index, request)
        probe = sparse_question_dispatch_probe(index, record)
        assert probe.sparse_trace_count == len(record.traces)
        assert probe.projected_trace_count == registry_count
        assert probe.sparse_trace_count < probe.projected_trace_count
        assert probe.phase_candidate_entry_counts == tuple(
            len(item.entry_sha256s) for item in record.phase_visits)
        assert probe.phase_candidate_construction_counts == tuple(
            item.construction_count for item in record.phase_visits)
        assert probe.created_exact_result_count == len(record.traces)
        assert probe.created_alias_result_count == sum(
            item.alias_result is not None for item in record.traces)
        assert probe.created_implicit_result_count == sum(
            item.implicit_result is not None for item in record.traces)
        assert probe.created_dispatch_trace_count == sum(
            probe.phase_candidate_entry_counts)

    missing = RawQuestionRequest("未学习的稀疏问题结构？")
    calls = {"exact": 0, "alias": 0, "implicit": 0}
    originals = {
        "exact": runtime._unknown_exact,
        "alias": runtime._unknown_alias,
        "implicit": runtime._unknown_implicit,
    }

    def exact_probe(*args, **kwargs):
        calls["exact"] += 1
        return originals["exact"](*args, **kwargs)

    def alias_probe(*args, **kwargs):
        calls["alias"] += 1
        return originals["alias"](*args, **kwargs)

    def implicit_probe(*args, **kwargs):
        calls["implicit"] += 1
        return originals["implicit"](*args, **kwargs)

    monkeypatch.setattr(runtime, "_unknown_exact", exact_probe)
    monkeypatch.setattr(runtime, "_unknown_alias", alias_probe)
    monkeypatch.setattr(runtime, "_unknown_implicit", implicit_probe)
    record = runtime.run_sparse_question_dispatch(index, missing)
    assert calls == {"exact": 0, "alias": 0, "implicit": 0}
    assert record.decision.status == "UNKNOWN"
    assert record.traces == ()
    assert all(not item.entry_sha256s for item in record.phase_visits)
    probe = runtime.sparse_question_dispatch_probe(index, record)
    assert probe.phase_candidate_entry_counts == (0, 0, 0)
    assert probe.phase_candidate_construction_counts == (0, 0, 0)
    assert probe.created_exact_result_count == 0
    assert probe.created_alias_result_count == 0
    assert probe.created_implicit_result_count == 0
    assert probe.created_dispatch_trace_count == 0
    projected = runtime.project_sparse_question_dispatch_audit(index, record)
    assert calls == {
        "exact": registry_count,
        "alias": registry_count,
        "implicit": registry_count,
    }
    assert len(projected.traces) == probe.projected_trace_count
    assert projected.status == "UNKNOWN"


def test_unlearned_alias_source_scope_and_repetition_preserve_audit(
        sparse_index) -> None:
    index, two_entry, three_entry = sparse_index
    construction = two_entry.feature_catalog.catalog[0]
    unlearned = RawQuestionRequest(
        _unlearned_alias_surface(construction),
        construction.source_record_key,
    )
    record, projected = _assert_complete_equal(index, unlearned)
    assert record.decision.status == "UNKNOWN"
    assert record.traces
    assert any(
        item.alias_result is not None and item.alias_result.matches
        for item in record.traces
    )
    assert any(
        item.alias_result is not None and item.alias_result.matches
        for item in projected.traces
    )
    wrong_source = three_entry.feature_catalog.catalog[0].source_record_key
    wrong = RawQuestionRequest(construction.question_surface, wrong_source)
    wrong_record, wrong_projected = _assert_complete_equal(index, wrong)
    assert wrong_record.decision.status == wrong_projected.status == "UNKNOWN"
    assert (
        run_sparse_question_dispatch(index, unlearned).sha256()
        == run_sparse_question_dispatch(index, unlearned).sha256()
    )


def test_sparse_alias_normalization_performs_no_exact_relookup(
        sparse_index, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_question_sparse_dispatch as runtime,
    )

    index, two_entry, _ = sparse_index
    construction = two_entry.feature_catalog.catalog[0]
    request = RawQuestionRequest(
        _learned_alias_surface(two_entry, construction),
        construction.source_record_key,
    )
    phases = []
    original = runtime.lookup_indexed_alias_frame_anchor_constructions

    def probe(candidate_index, phase, current_request):
        phases.append(phase)
        return original(candidate_index, phase, current_request)

    monkeypatch.setattr(
        runtime,
        "lookup_indexed_alias_frame_anchor_constructions",
        probe,
    )
    record = runtime.run_sparse_question_dispatch(index, request)
    assert record.decision.status == "ANSWER"
    assert tuple(phases) == ("EXACT", "ALIAS")


def test_same_surface_ambiguity_and_overlapping_entries_stay_global(
        sparse_index) -> None:
    _, two_entry, three_entry = sparse_index
    shared = two_entry.feature_catalog.catalog[0]
    collision_pattern = replace(
        shared.pattern,
        construction_id=f"{shared.pattern.construction_id}_FT20_COLLISION",
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
    collision_index = _build_sparse_index(build_raw_question_feature_registry(
        (collision_entry, three_entry)))
    collision_record, collision_result = _assert_complete_equal(
        collision_index,
        RawQuestionRequest(
            shared.question_surface, shared.source_record_key),
    )
    assert collision_record.decision.status == "CLARIFY"
    assert collision_record.decision.decisive_phase == "EXACT"
    assert collision_result.status == "CLARIFY"

    projected_catalog = replace(two_entry.feature_catalog, catalog=(shared,))
    projected_entry = RawQuestionFeatureRegistryEntry(
        projected_catalog,
        two_entry.alias_bridge,
        replace(
            two_entry.implicit_bundle,
            explicit_catalog=projected_catalog,
        ),
    )
    overlap_index = _build_sparse_index(build_raw_question_feature_registry(
        (two_entry, projected_entry)))
    overlap_record, overlap_result = _assert_complete_equal(
        overlap_index,
        RawQuestionRequest(
            _learned_alias_surface(two_entry, shared),
            shared.source_record_key,
        ),
    )
    assert overlap_record.decision.status == "ANSWER"
    assert overlap_record.decision.decisive_phase == "ALIAS"
    assert len(overlap_record.decision.interpretations) == 1
    assert len(overlap_record.traces) == 2
    assert len(overlap_result.traces) == 2


def test_sparse_contract_rejects_candidate_or_index_drift(sparse_index) -> None:
    index, two_entry, _ = sparse_index
    with pytest.raises(ValueError, match="entries are not canonical"):
        replace(index, entries=index.entries[::-1])
    construction = two_entry.feature_catalog.catalog[0]
    record = run_sparse_question_dispatch(index, RawQuestionRequest(
        construction.question_surface,
        construction.source_record_key,
    ))
    with pytest.raises(ValueError, match="traces escaped candidate entries"):
        replace(
            record,
            phase_visits=(RawQuestionSparsePhaseVisit("EXACT", (), 0),),
        )
    with pytest.raises(ValueError, match="decision escaped"):
        replace(
            record,
            decision=RawQuestionSparseDecision(
                "CLARIFY", None, "EXACT", (), None),
        )


def test_sparse_runtime_contains_no_registry_scan_or_dispatch_table() -> None:
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_question_sparse_dispatch.py"
    )).read_text(encoding="utf-8")
    assert "registry.entries" not in source
    for value in ("暴雨", "寒潮", "山区", "桥面", "什么", "哪里"):
        assert value not in source
    assert "bundle_name" not in source
