"""Independent new-fact truth, parse ambiguity and real Core/cold-Evidence gates."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef, VersionBundle
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, OpenRoleFrame,
)
from pure_integer_ai.experiments.independent_relation_fact_admission import (
    ADMISSION_READY, admit_independent_relation_fact, grant_independent_fact_inputs,
    learn_independent_relation_fact_admission,
)
from pure_integer_ai.experiments.independent_relation_fact_source import (
    compile_independent_relation_fact_source,
)
from pure_integer_ai.experiments.independent_fact_preflight_core_audit import verify_cold_admission_history
from pure_integer_ai.experiments.integer_relation_graph_course import (
    integer_relation_graph_course_to_w06_adapter,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_learning import W06RelationLearningRuntime, build_w06_learning_runtime
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.conversation_typed_relation_bridge import _materialize_relation_spans
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.run_conversation_training import dialogue_semantic_protocols
from pure_integer_ai.storage.backend import SQLiteBackend


INPUT_SOURCE = SourceRef(92040, 1, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
SURFACE = "青苔闸导致萤砂灯。"


def _source(tmp_path, name, *, split="train", role="support", terms=None, additional_rows=()):
    directory = tmp_path / name / "data" / "ph2"
    directory.mkdir(parents=True)
    terms = terms or [["青苔闸", "event", 202610130081], ["萤砂灯", "event", 202610130082]]
    row = {"anchor": "导致", "case_id": name, "relation": "CAUSES",
           "sample_role": role, "seed_id": name, "split": split,
           "surface": terms[0][0] + "导致" + terms[1][0] + "。", "terms": terms}
    path = directory / "facts.jsonl"
    path.write_bytes(b"".join(canonical_json_bytes(item) + b"\n"
                              for item in (row, *additional_rows)))
    return compile_independent_relation_fact_source(path, tmp_path / name / "pack")


def _context(backend):
    context = make_train_context(backend)
    _semantic, occurrence, span = dialogue_semantic_protocols()
    install_language_graph_protocols(context, occurrence_protocol=occurrence, span_protocol=span)
    return context


@pytest.fixture
def facts(tmp_path):
    """Independent new entities are absent from the old graph-trained syntax."""
    base = _source(tmp_path, "base", terms=[
        ["晨阀启动", "event", 202610130071], ["暮笛鸣响", "event", 202610130072]])
    backend = SQLiteBackend(str(tmp_path / "base.sqlite3"))
    try:
        context = _context(backend)
        adapter = integer_relation_graph_course_to_w06_adapter(base.course, require_full_curriculum=False)
        runtime = build_w06_learning_runtime(backend, adapter, context=context)
        _materialize_relation_spans(context, runtime, adapter)
        backend.commit()
    finally:
        backend.close()
    with TrainedRelationGraphRuntime(tmp_path / "base.sqlite3") as owner:
        frames = restore_open_role_frames(owner.active_surface_frames(), owner.active_surface_facts())
        assert frames
        assert all(tuple(map(ord, "青苔闸")) not in frame.gaps for frame in frames)
    # No source/teacher record from the new entities was used to construct frames.
    evidence = _source(tmp_path, "independent")
    grant = grant_independent_fact_inputs(evidence, (INPUT_SOURCE,))
    return evidence, grant, frames


def test_new_entities_require_separate_source_and_real_core_learning(tmp_path, facts):
    evidence, grant, frames = facts
    admitted = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, frames, grant=grant)
    assert admitted.status == ADMISSION_READY
    assert admitted.course == evidence.course
    assert all(parse.frame.proposition != admitted.course.records[0].proposition_key
               for parse in admitted.parses)
    assert tuple(span.values for span in admitted.parses[0].bindings) == (
        tuple(map(ord, "青苔闸")), tuple(map(ord, "萤砂灯")))
    assert OpenRelationCandidate.from_stable_key(admitted.parses[0].stable_key()) == admitted.parses[0]
    database = tmp_path / "new_core.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = _context(backend)
        before = W06RelationLearningRuntime(backend, context=context)
        assert not before.active_candidates()
        runtime = learn_independent_relation_fact_admission(backend, context, admitted)
        assert runtime.report().active_candidate_count == 1
        histories = tuple(runtime.learning.engine.ledger.evidence_history(
            runtime.learning.hypothesis_for_candidate(item.proposition.proposition))
            for item in runtime.active_candidates())
        witnesses = tuple(record.verification.trace
                          for candidate in runtime.active_candidates()
                          for record in runtime.learning.engine.recognition_history(
                              runtime.learning.hypothesis_for_candidate(candidate.proposition.proposition)))
        assert witnesses and all(_witness(trace) == admitted.stable_key() for trace in witnesses)
        backend.commit()
    finally:
        backend.close()
    with TrainedRelationGraphRuntime(database) as restored:
        assert len(restored.active_propositions()) == 1
        assert tuple(restored.owner.learning.engine.ledger.evidence_history(
            restored.owner.learning.hypothesis_for_candidate(item.definition.proposition))
            for item in restored.active_propositions()) == histories
        assert tuple(record.verification.trace
                     for item in restored.active_propositions()
                     for record in restored.owner.learning.engine.recognition_history(
                         restored.owner.learning.hypothesis_for_candidate(item.definition.proposition))) == witnesses
    cold = verify_cold_admission_history(database, [{
        "seed_id": "independent", "status": admitted.status,
        "witness": list(admitted.stable_key()),
        "proposition_keys": [list(row.proposition_key) for row in admitted.course.records],
    }])
    assert cold["status"] == "PASS_COLD_ADMISSION_CORE_HISTORY"
    assert len(cold["claims"]) == 1 and cold["claims"][0]["full_witness_matches"] == 1


def _witness(key):
    return key[2:2 + key[1]]


@pytest.mark.parametrize("surface,source,has_grant,status,reason", [
    (SURFACE, INPUT_SOURCE, False, "UNKNOWN", "NO_INDEPENDENT_TEACHER_EVIDENCE"),
    (SURFACE, replace(INPUT_SOURCE, source_id=2), True, "REFUSE", "INPUT_SOURCE_NOT_AUTHORIZED"),
    ("青苔闸导致未许可灯。", INPUT_SOURCE, True, "UNKNOWN", "NO_INDEPENDENT_FACT_FOR_CURRENT_ROLES"),
    ("萤砂灯导致青苔闸。", INPUT_SOURCE, True, "UNKNOWN", "NO_INDEPENDENT_FACT_FOR_CURRENT_ROLES"),
    (SURFACE + "多余尾段", INPUT_SOURCE, True, "UNKNOWN", "NO_COMPLETE_TRAINED_ROLE_PARSE"),
])
def test_unsupported_or_unauthorized_input_has_no_core_course(facts, surface, source, has_grant, status, reason):
    _evidence, grant, frames = facts
    result = admit_independent_relation_fact(surface, source, frames, grant=grant if has_grant else None)
    assert (result.status, result.reason, result.course) == (status, reason, None)
    with pytest.raises(ValueError, match="independently admitted"):
        learn_independent_relation_fact_admission(None, None, result)


def test_teacher_truth_cannot_resolve_ambiguous_syntax(facts):
    _evidence, grant, frames = facts
    alternate = replace(frames[0], roles=tuple(reversed(frames[0].roles)))
    result = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, (*frames, alternate), grant=grant)
    assert (result.status, result.reason) == ("CLARIFY", "MULTIPLE_COMPLETE_ROLE_INTERPRETATIONS")
    assert result.course is None


@pytest.mark.parametrize("role", ["conflict", "refute"])
def test_non_support_teacher_never_becomes_active_truth(tmp_path, facts, role):
    _evidence, _grant, frames = facts
    evidence = _source(tmp_path, "non_support", role=role)
    grant = grant_independent_fact_inputs(evidence, (INPUT_SOURCE,))
    result = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, frames, grant=grant)
    assert (result.status, result.reason, result.course) == (
        "UNKNOWN", "INDEPENDENT_EVIDENCE_NOT_UNCONTESTED_SUPPORT", None)


@pytest.mark.parametrize("budget", ["parse", "witness"])
def test_complete_parse_and_full_proof_budgets_fail_closed(facts, budget):
    _evidence, grant, frames = facts
    result = admit_independent_relation_fact(
        SURFACE, INPUT_SOURCE, frames, grant=grant,
        **({"max_parse_steps": 1} if budget == "parse" else {"max_witness_integers": 1}))
    assert result.status == "UNKNOWN" and result.course is None
    assert result.reason.endswith("BUDGET_EXCEEDED")


def test_evaluator_source_and_course_drift_are_rejected(tmp_path, facts):
    evidence, _grant, _frames = facts
    with pytest.raises(ValueError, match="train teacher"):
        _source(tmp_path, "evaluation", split="held_out")
    changed = replace(evidence, course=replace(evidence.course, source_identity=(92034, 999)))
    with pytest.raises(ValueError, match="permission"):
        grant_independent_fact_inputs(changed, (INPUT_SOURCE,))
    altered_record = replace(evidence.course.records[0], logical_order=999)
    changed = replace(evidence, course=replace(evidence.course, records=(altered_record,)))
    with pytest.raises(ValueError, match="teacher course"):
        grant_independent_fact_inputs(changed, (INPUT_SOURCE,))


def test_all_teacher_observations_are_checked_before_support(tmp_path, facts):
    _evidence, _grant, frames = facts
    extra = {"anchor": "导致", "case_id": "mixed", "relation": "CAUSES",
             "sample_role": "refute", "seed_id": "mixed-refute", "split": "train",
             "surface": SURFACE,
             "terms": [["青苔闸", "event", 202610130081], ["萤砂灯", "event", 202610130082]]}
    evidence = _source(tmp_path, "mixed", additional_rows=(extra,))
    assert len(evidence.course.records) == 2
    grant = grant_independent_fact_inputs(evidence, (INPUT_SOURCE,))
    result = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, frames, grant=grant)
    assert (result.status, result.reason, result.course) == (
        "UNKNOWN", "INDEPENDENT_EVIDENCE_NOT_UNCONTESTED_SUPPORT", None)


def test_same_surface_does_not_merge_independent_entity_identities(tmp_path, facts):
    _evidence, _grant, frames = facts
    extra = {"anchor": "导致", "case_id": "ambiguous", "relation": "CAUSES",
             "sample_role": "support", "seed_id": "ambiguous-second", "split": "train",
             "surface": SURFACE,
             "terms": [["青苔闸", "event", 202610130091], ["萤砂灯", "event", 202610130092]]}
    evidence = _source(tmp_path, "ambiguous", additional_rows=(extra,))
    grant = grant_independent_fact_inputs(evidence, (INPUT_SOURCE,))
    result = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, frames, grant=grant)
    assert (result.status, result.reason, result.course) == (
        "CLARIFY", "INDEPENDENT_ENTITY_IDENTITY_AMBIGUOUS", None)


def test_constructed_grant_cannot_replace_physical_teacher(facts):
    evidence, grant, frames = facts
    forged = replace(grant, course=replace(
        evidence.course, records=(replace(evidence.course.records[0], logical_order=999),)))
    result = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, frames, grant=forged)
    assert (result.status, result.reason, result.course) == ("UNKNOWN", "INDEPENDENT_GRANT_DRIFT", None)


def test_source_drift_is_rechecked_before_core_writes(facts):
    evidence, grant, frames = facts
    admitted = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, frames, grant=grant)
    assert admitted.status == ADMISSION_READY
    evidence.source_path.write_bytes(evidence.source_path.read_bytes() + b"\n")
    result = admit_independent_relation_fact(SURFACE, INPUT_SOURCE, frames, grant=grant)
    assert result.status == "UNKNOWN" and result.course is None
    assert result.reason == "INDEPENDENT_SOURCE_REVALIDATION_FAILED"
    with pytest.raises(ValueError, match="drifted before Core"):
        learn_independent_relation_fact_admission(None, None, admitted)
