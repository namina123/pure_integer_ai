"""Train-only natural revision cue graph and Memory binding contracts."""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    memory_input_candidate_keys,
)
from pure_integer_ai.experiments.natural_revision_runtime import (
    NaturalRevisionBindingError,
    NaturalRevisionRuntime,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_revision_cue_graph import (
    RevisionCueGraphRecord,
    TrainedRevisionCue,
    materialize_revision_cue_graph,
    read_revision_cue_graph,
)

from test_query_input_structure import _structure


SOURCE = (91660, 25, 20261025, 1)
LICENSE = tuple(map(ord, "CC0-1.0"))
CUE = TrainedRevisionCue(
    (91690, 1), tuple(map(ord, "fix:")), (91690, 10))
RECORD = RevisionCueGraphRecord(CUE, SOURCE, LICENSE)


def _projector():
    structure = _structure(
        predicate_values=(ord("="),),
        filler_values=(ord("a"), ord("b")),
        filler_candidates=((4, 101), (4, 102)),
    )
    return TrainedInputStructureProjector((structure,))


def _runtime(records=(RECORD,)):
    return NaturalRevisionRuntime(tuple(sorted(
        records, key=lambda item: item.stable_key())))


def test_revision_cue_graph_cold_restores_integer_provenance(tmp_path):
    database = tmp_path / "revision-cues.sqlite3"
    receipt = materialize_revision_cue_graph(
        (CUE,), database,
        source_identity=SOURCE,
        license_identity=LICENSE,
    )

    records, metadata = read_revision_cue_graph(database)

    assert receipt["cue_count"] == 1
    assert records == (RECORD,)
    assert metadata["source_identity"] == SOURCE
    assert metadata["license_identity"] == LICENSE
    with pytest.raises(FileExistsError):
        materialize_revision_cue_graph(
            (CUE,), database,
            source_identity=SOURCE,
            license_identity=LICENSE,
        )


def test_revision_binding_requires_one_cue_one_proposition_and_prefix_order():
    projector = _projector()
    runtime = _runtime()
    ordinary = projector.project(tuple(map(ord, "=ab")))
    corrected = projector.project(tuple(map(ord, "fix:=ab")))

    assert runtime.bind_input(ordinary) is None
    binding = runtime.bind_input(
        corrected, memory_input_structure=corrected)
    assert binding is not None
    assert binding.proposition_key == corrected.relation_candidates[0].proposition_key
    assert binding.cue_identity == CUE.cue_identity
    assert binding.source_identity == SOURCE
    assert binding.license_identity == LICENSE

    trailing_cue = projector.project(tuple(map(ord, "=abfix:")))
    with pytest.raises(NaturalRevisionBindingError, match="before"):
        runtime.bind_input(trailing_cue)

    ambiguous_runtime = _runtime((
        RECORD,
        RevisionCueGraphRecord(
            TrainedRevisionCue((91690, 2), CUE.surface_values, (91690, 10)),
            SOURCE,
            LICENSE,
        ),
    ))
    with pytest.raises(NaturalRevisionBindingError, match="exactly one"):
        ambiguous_runtime.bind_input(corrected)


def test_revision_rejects_missing_target_and_competing_refutation_before_write(
        tmp_path):
    projector = _projector()
    runtime = _runtime()
    corrected = projector.project(tuple(map(ord, "fix:=ab")))
    binding = runtime.bind_input(corrected)
    assert binding is not None

    empty_path = tmp_path / "empty-memory.sqlite3"
    with TrainedDialogueMemoryGraph(
            empty_path, session_id=11, input_projector=projector) as empty:
        before = empty.backend.count("memory_event")
        with pytest.raises(NaturalRevisionBindingError, match="no preceding"):
            runtime.resolve_target(empty, corrected, binding)
        assert empty.backend.count("memory_event") == before

    conflict_path = tmp_path / "conflict-memory.sqlite3"
    with TrainedDialogueMemoryGraph(
            conflict_path, session_id=12, input_projector=projector) as memory:
        memory.append("=ab", speaker_kind=1,
                      input_structure=projector.project(tuple(map(ord, "=ab"))))
        memory.retract("=ab", speaker_kind=1)
        before_count = memory.backend.count("memory_event")
        before_turn = memory._next_turn_seq
        with pytest.raises(NaturalRevisionBindingError, match="competing"):
            runtime.resolve_target(memory, corrected, binding)
        assert memory.backend.count("memory_event") == before_count
        assert memory._next_turn_seq == before_turn


def test_revision_supersedes_target_from_correction_source_and_restores_cold(
        tmp_path):
    projector = _projector()
    runtime = _runtime()
    original = projector.project(tuple(map(ord, "=ab")))
    corrected = projector.project(tuple(map(ord, "fix:=ab")))
    binding = runtime.bind_input(corrected)
    assert binding is not None
    database = tmp_path / "revision-memory.sqlite3"

    with TrainedDialogueMemoryGraph(
            database, session_id=13, input_projector=projector) as memory:
        first = memory.append("=ab", speaker_kind=1,
                              input_structure=original)
        first_observation_ref = memory.intake.result_for_source(
            first.source).observation_ref
        assert first_observation_ref is not None
        target = runtime.resolve_target(memory, corrected, binding)
        assert target.prior_observation_key == first_observation_ref.stable_key()
        corrected_append = memory.append(
            "fix:=ab", speaker_kind=1, input_structure=corrected)
        commit = runtime.commit(memory, target, corrected_append.source)
        assert commit.target_binding == target
        assert commit.correction_source_ref == corrected_append.source.stable_key()
        assert commit.replacement_evidence_key

    with TrainedDialogueMemoryGraph(
            database, session_id=13, input_projector=projector) as restored:
        snapshot = restored.active_structures(candidate_keys=memory_input_candidate_keys(
            corrected, session_id=13, speaker_kind=1))
        replacement = next(
            item for item in snapshot.evidence
            if item.evidence_key == commit.replacement_evidence_key)
        prior = next(
            item for item in snapshot.evidence
            if item.evidence_key == target.prior_evidence_key)
        assert replacement.stance == EVIDENCE_SUPPORT
        assert replacement.source_ref[1:] == corrected_append.source.stable_key()
        assert replacement.supersedes_key == prior.evidence_key
        assert prior.stance == EVIDENCE_SUPPORT
