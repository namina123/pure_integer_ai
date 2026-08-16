"""A-04 GG-03 assessment、held-out、rollback 与 restore 聚焦纵切。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisLedger
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.experiments.evaluation_isolation import clone_backend
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_authored_generation_generalization_course import (
    read_authored_generation_generalization_seeds,
)
from pure_integer_ai.experiments.ph2_generation_choice_assessment_selector import (
    GenerationChoiceAssessmentSelectorPolicy,
    select_generation_choice_by_assessment,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    GenerationChoiceCandidateMapper,
)
from pure_integer_ai.experiments.ph2_generation_generalization_assessment_runtime import (
    GG03_RUNTIME_VERSIONS,
    GG03_TRAINING_OWNER,
    GenerationGeneralizationAssessmentRuntimePolicy,
    apply_generation_generalization_training_assessments,
    evaluate_generation_generalization_held_out,
    project_generation_generalization_seed,
    revise_generation_generalization_assessment,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT

from tests.test_d02_gg01_generation_choice_contract import _candidate_protocol
from tests.test_r00_relation_closure import _projection_protocol, _verifier


SAMPLE_PATH = Path(
    "data/ph2/authored_generation_generalization_seed_v1.jsonl.sample")
_BASE = 21160


def _assessment_owner(backend):
    context = make_train_context(backend)
    aggregate = SourceRef(
        _BASE, 1, 0, GG03_TRAINING_OWNER, GG03_RUNTIME_VERSIONS)
    evidence_protocol = EvidenceCandidateProtocol(
        (_BASE, 2),
        (_BASE, 3),
        aggregate,
        document_scope(aggregate),
        1,
    )
    history_protocol = TrainingHypothesisHistoryProtocol(
        (_BASE, 7),
        evidence_protocol.hypothesis_kind_key,
        evidence_protocol.aggregate_source,
        evidence_protocol.aggregate_scope,
    )
    assert context.training_candidate_history is not None
    learning = CandidateLearningRuntime(
        EvidenceCandidateEngine(
            evidence_protocol,
            ledger=HypothesisLedger(TrainingHypothesisEventSink(
                context.training_candidate_history, history_protocol)),
        ),
        CandidateProjectionGraph(
            context.graph_ontology, _projection_protocol()),
        _verifier(),
        CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
    )
    mapper = GenerationChoiceCandidateMapper(_candidate_protocol(_BASE))
    verifier_source = SourceRef(
        _BASE, 4, 0, GG03_TRAINING_OWNER, GG03_RUNTIME_VERSIONS)
    policy = GenerationGeneralizationAssessmentRuntimePolicy(
        (_BASE, 5),
        GenerationChoiceAssessmentSelectorPolicy((_BASE, 6)),
    )
    return (
        context,
        evidence_protocol,
        history_protocol,
        learning,
        mapper,
        verifier_source,
        policy,
    )


def test_gg03_assessment_held_out_rollback_and_restore_are_bounded():
    """TRAIN 可改选；held-out 零写；revision/restore 保持 exact scope。"""
    seeds = read_authored_generation_generalization_seeds(SAMPLE_PATH)
    teacher = tuple(item for item in seeds if item.split == "train")
    evaluator = tuple(item for item in seeds if item.split == "held_out")
    backend = DictBackend()
    cloned_backend = None
    try:
        (context, evidence_protocol, history_protocol, learning, mapper,
         verifier_source, policy) = _assessment_owner(backend)

        training = apply_generation_generalization_training_assessments(
            teacher, mapper, learning, verifier_source, policy)

        assert len(training.records) == 14
        assert training.candidate_registrations == 14
        assert training.assessment_updates == 14
        assert training.teacher_call_count == 0
        assert training.changed_selections == 7
        assert {item.case.choice_kind for item in training.records} == set(
            CHOICE_KINDS)
        teacher_by_id = {item.seed_id: item for item in teacher}
        for record in training.records:
            seed = teacher_by_id[record.case.seed_id]
            selected_id = record.case.option_for_choice(
                record.selection.selected).surface_candidate_id
            assert selected_id in seed.expected_payload.to_value()[
                "accepted_surface_candidate_ids"]
            if record.stance == EVIDENCE_SUPPORT:
                assert record.selection.reason == "UNIQUE_ACTIVE_ASSESSMENT"
                assert record.selection.selected == record.case.challenge.choice
            else:
                assert record.selection.reason == "NO_ACTIVE_BASELINE_NE"
                assert record.selection.selected == record.case.baseline.choice

        before_held_out_backend = backend.snapshot()
        before_held_out_state = learning.state_key()
        held_out = evaluate_generation_generalization_held_out(
            evaluator, mapper, learning, policy)

        assert len(held_out.records) == 14
        assert held_out.legal_selection_count == 14
        assert held_out.host_write_count == 0
        assert held_out.teacher_call_count == 0
        assert held_out.runtime_status == (
            "NE_INDEPENDENT_LAYER_INPUT_MISSING")
        assert all(
            item.selection.reason == "NO_ACTIVE_BASELINE_NE"
            and item.selection.selected == item.case.baseline.choice
            and item.runtime_status == "NE_INDEPENDENT_LAYER_INPUT_MISSING"
            for item in held_out.records)
        assert backend.snapshot() == before_held_out_backend
        assert learning.state_key() == before_held_out_state

        revision_record = next(
            item for item in training.records
            if item.case.seed_id == "GG03_T_S10")
        anchor_record = next(
            item for item in training.records
            if item.case.seed_id == "GG03_T_S08")
        assert revision_record.selection.selected == (
            revision_record.case.challenge.choice)
        anchor_before = select_generation_choice_by_assessment(
            mapper,
            learning,
            policy.selector_policy,
            anchor_record.case.choices,
            anchor_record.case.baseline.choice,
        )
        disabled_before = learning.state_key()
        disabled = select_generation_choice_by_assessment(
            mapper,
            learning,
            GenerationChoiceAssessmentSelectorPolicy(
                (_BASE, 8), (revision_record.case.choice_kind,)),
            revision_record.case.choices,
            revision_record.case.baseline.choice,
        )
        assert disabled.reason == "DISABLED_BASELINE"
        assert disabled.selected == revision_record.case.baseline.choice
        assert learning.state_key() == disabled_before

        rollback = revise_generation_generalization_assessment(
            revision_record.case,
            revision_record.learning.evidence,
            EVIDENCE_UNKNOWN,
            mapper,
            learning,
            verifier_source,
            policy.selector_policy,
        )
        assert rollback.selection.reason == "NO_ACTIVE_BASELINE_NE"
        assert rollback.selection.selected == revision_record.case.baseline.choice
        anchor_after_rollback = select_generation_choice_by_assessment(
            mapper,
            learning,
            policy.selector_policy,
            anchor_record.case.choices,
            anchor_record.case.baseline.choice,
        )
        assert anchor_after_rollback == anchor_before

        reverified = revise_generation_generalization_assessment(
            revision_record.case,
            rollback.outcome.evidence,
            EVIDENCE_SUPPORT,
            mapper,
            learning,
            verifier_source,
            policy.selector_policy,
        )
        assert reverified.selection.reason == "UNIQUE_ACTIVE_ASSESSMENT"
        assert reverified.selection.selected == (
            revision_record.case.challenge.choice)

        cloned_backend = clone_backend(backend)
        cloned_context = make_train_context(cloned_backend)
        assert cloned_context.training_candidate_history is not None
        restored = CandidateLearningRuntime.restore_for_training_graph(
            evidence_protocol,
            CandidateProjectionGraph(
                cloned_context.graph_ontology, _projection_protocol()),
            _verifier(),
            CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
            cloned_context.training_candidate_history,
            history_protocol,
        )
        restored_state = restored.state_key()
        for seed_id in ("GG03_T_S10", "GG03_T_S13", "GG03_T_S14"):
            seed = next(item for item in teacher if item.seed_id == seed_id)
            case = project_generation_generalization_seed(seed)
            selection = select_generation_choice_by_assessment(
                mapper,
                restored,
                policy.selector_policy,
                case.choices,
                case.baseline.choice,
            )
            assert selection.reason == "UNIQUE_ACTIVE_ASSESSMENT"
            assert selection.selected == case.challenge.choice
        assert restored.state_key() == restored_state
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()
