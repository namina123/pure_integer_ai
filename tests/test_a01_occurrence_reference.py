"""A-01 occurrence 级多候选、上下文 Evidence 和后文回溯对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.occurrence_reference import (
    OccurrenceReferenceEvidence,
    OccurrenceReferenceRequest,
    OccurrenceReferenceRevision,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from tests.test_f00_reference_question_runtime import _reference_fixture


_BASE = 19700


def _claim(
        world,
        antecedent,
        *,
        dimension: int,
        stance: int,
        timestamp_seq: int,
        reason: int,
        visible_occurrences=(),
        supersedes_evidence_id: int = 0,
        ) -> OccurrenceReferenceEvidence:
    """构造不写死维度语义的来源化 reference Evidence。"""
    return OccurrenceReferenceEvidence(
        antecedent,
        concept_identity((_BASE + 1, dimension)),
        stance,
        minimal_instruction_identity((_BASE + 2, reason)),
        world.current.source,
        timestamp_seq,
        tuple(visible_occurrences),
        (concept_identity((_BASE + 3, dimension, stance)),),
        (_BASE + 4, dimension, stance, timestamp_seq),
        supersedes_evidence_id,
    )


def _request(
        world,
        *,
        timestamp_seq: int,
        window=None,
        evidence=(),
        revisions=(),
        ) -> OccurrenceReferenceRequest:
    """为同一 reference 和完整候选集建立一次有界 A-01 请求。"""
    reference = world.current.occurrences[0]
    return OccurrenceReferenceRequest(
        reference,
        (world.candidate_occurrences if window is None else tuple(window)),
        world.candidate_occurrences,
        world.current.runtime_scope,
        timestamp_seq,
        tuple(evidence),
        tuple(revisions),
    )


def test_a01_ambiguous_candidates_remain_adopted_without_stable_tiebreak():
    """无定向 Evidence 的两个先行 occurrence 必须都 adopted，不能按位置私选。"""
    world = _reference_fixture(ambiguous=True)
    try:
        first = world.reference_runtime.resolve(
            _request(world, timestamp_seq=10))
        state = world.reference_runtime.state_key()
        replay = world.reference_runtime.resolve(
            _request(world, timestamp_seq=10))

        assert len(first.candidates) == 2
        assert len(first.adopted_candidates) == 2
        assert first.winner is None
        assert all(
            item.stance == EVIDENCE_UNKNOWN for item in first.evidence)
        assert replay == first
        assert world.reference_runtime.state_key() == state
    finally:
        world.close()


def test_a01_injected_dimensions_and_later_context_replace_old_winner():
    """speaker/time/context 三维 Evidence 可先形成 winner，再由后文定向替代。"""
    world = _reference_fixture(ambiguous=True)
    try:
        first_antecedent, second_antecedent = world.candidate_occurrences
        reference = world.current.occurrences[0]
        first_context = _claim(
            world,
            first_antecedent,
            dimension=3,
            stance=EVIDENCE_UNKNOWN,
            timestamp_seq=10,
            reason=3,
            visible_occurrences=(first_antecedent, reference),
        )
        first = world.reference_runtime.resolve(_request(
            world,
            timestamp_seq=10,
            evidence=(
                _claim(
                    world,
                    first_antecedent,
                    dimension=1,
                    stance=EVIDENCE_SUPPORT,
                    timestamp_seq=10,
                    reason=1,
                    visible_occurrences=(first_antecedent, reference),
                ),
                _claim(
                    world,
                    first_antecedent,
                    dimension=2,
                    stance=EVIDENCE_SUPPORT,
                    timestamp_seq=10,
                    reason=2,
                    visible_occurrences=(first_antecedent, reference),
                ),
                first_context,
                _claim(
                    world,
                    second_antecedent,
                    dimension=1,
                    stance=EVIDENCE_UNKNOWN,
                    timestamp_seq=10,
                    reason=4,
                    visible_occurrences=(second_antecedent, reference),
                ),
                _claim(
                    world,
                    second_antecedent,
                    dimension=2,
                    stance=EVIDENCE_UNKNOWN,
                    timestamp_seq=10,
                    reason=5,
                    visible_occurrences=(second_antecedent, reference),
                ),
                _claim(
                    world,
                    second_antecedent,
                    dimension=3,
                    stance=EVIDENCE_UNKNOWN,
                    timestamp_seq=10,
                    reason=6,
                    visible_occurrences=(second_antecedent, reference),
                ),
            ),
        ))
        assert first.winner is not None
        assert first.winner.antecedent == first_antecedent
        first_context_record = next(
            item for item in first.evidence
            if (item.hypothesis == first.winner.hypothesis
                and item.reason_key == first_context.reason.stable_key())
        )

        reference_record = world.ctx.occurrence_index.read(reference)
        later = world.ctx.occurrence_index.record(
            source=reference_record.source,
            raw_text=reference_record.raw_text,
            scope=reference_record.scope,
            start=len(reference_record.raw_text),
            end=len(reference_record.raw_text),
            ordinal=1,
            segment_index=3,
            local_index=0,
            document_index=3,
        ).occurrence
        refute_old = _claim(
            world,
            first_antecedent,
            dimension=3,
            stance=EVIDENCE_REFUTE,
            timestamp_seq=11,
            reason=7,
            visible_occurrences=(reference, later),
            supersedes_evidence_id=first_context_record.evidence_id,
        )
        support_new = _claim(
            world,
            second_antecedent,
            dimension=3,
            stance=EVIDENCE_SUPPORT,
            timestamp_seq=11,
            reason=8,
            visible_occurrences=(reference, later),
        )
        revision = OccurrenceReferenceRevision(
            first_antecedent,
            second_antecedent,
            refute_old,
        )
        second_request = _request(
            world,
            timestamp_seq=11,
            window=(*world.candidate_occurrences, later),
            evidence=(support_new,),
            revisions=(revision,),
        )
        second = world.reference_runtime.resolve(second_request)
        state = world.reference_runtime.state_key()
        replay = world.reference_runtime.resolve(second_request)

        assert second.winner is not None
        assert second.winner.antecedent == second_antecedent
        old_candidate = next(
            item for item in second.candidates
            if item.antecedent == first_antecedent)
        assert second.decision.candidate(
            old_candidate.hypothesis).after.lifecycle == LIFECYCLE_SUPERSEDED
        assert len(world.reference_runtime.resolver.decision_history(
            old_candidate.hypothesis)) == 2
        assert any(
            item.evidence_id == first_context_record.evidence_id
            for item in second.evidence)
        assert replay == second
        assert world.reference_runtime.state_key() == state
    finally:
        world.close()


def test_a01_scope_window_and_source_fail_before_state_write():
    """生命周期关闭、窗口越界和跨来源 occurrence 都必须在首个状态写前失败。"""
    world = _reference_fixture(ambiguous=True)
    try:
        baseline = world.reference_runtime.state_key()
        reference = world.current.occurrences[0]
        record = world.ctx.occurrence_index.read(reference)
        too_far = world.ctx.occurrence_index.record(
            source=record.source,
            raw_text=record.raw_text,
            scope=record.scope,
            start=len(record.raw_text),
            end=len(record.raw_text),
            ordinal=2,
            segment_index=99,
            local_index=0,
            document_index=99,
        ).occurrence
        with pytest.raises(ValueError, match="后文预算"):
            world.reference_runtime.resolve(_request(
                world,
                timestamp_seq=10,
                window=(*world.candidate_occurrences, too_far),
            ))
        assert world.reference_runtime.state_key() == baseline

        other_source = SourceRef(
            SOURCE_BARE_TEXT,
            _BASE + 10,
            2,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(),
        )
        other_scope = replace(
            record.scope,
            local_id=2,
            source=other_source,
        )
        foreign = world.ctx.occurrence_index.record(
            source=other_source,
            raw_text="外",
            scope=other_scope,
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
        ).occurrence
        with pytest.raises(ValueError, match="不得跨 SourceRef"):
            world.reference_runtime.resolve(_request(
                world,
                timestamp_seq=10,
                window=(*world.candidate_occurrences, foreign),
            ))
        assert world.reference_runtime.state_key() == baseline

        world.ctx.work_memory.end_episode()
        with pytest.raises(ValueError, match="活动 episode"):
            world.reference_runtime.resolve(
                _request(world, timestamp_seq=10))
        assert world.reference_runtime.state_key() == baseline
        world.ctx.work_memory.begin_episode(world.current.runtime_scope)
    finally:
        world.close()


def test_a01_clone_keeps_host_candidate_and_decision_state_unchanged():
    """评测 clone 可真实形成候选决策，但宿主 A-01 状态保持位级不变。"""
    world = _reference_fixture(ambiguous=True)
    clone_work_memory = WorkMemory()
    try:
        source = world.current.source
        clone_work_memory.begin_session(session_scope(
            _BASE + 20,
            owner=source.owner,
            versions=source.versions,
            source=source,
        ))
        clone_work_memory.begin_document(world.current.occurrence_scope)
        clone_work_memory.begin_episode(world.current.runtime_scope)
        cloned = world.reference_runtime.clone_for_context(
            world.ctx.occurrence_index,
            clone_work_memory,
        )
        host_state = world.reference_runtime.state_key()

        result = cloned.resolve(_request(world, timestamp_seq=10))

        assert len(result.adopted_candidates) == 2
        assert cloned.state_key() != host_state
        assert world.reference_runtime.state_key() == host_state
    finally:
        clone_work_memory.end_episode()
        clone_work_memory.end_document()
        clone_work_memory.end_session()
        world.close()
