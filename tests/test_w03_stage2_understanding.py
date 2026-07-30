"""PH2 W-03 typed Sense 候选的理解闭环与 active readback。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_w03_understanding import (
    W03_UNDERSTANDING_AMBIGUOUS,
    W03_UNDERSTANDING_CLARIFY,
    W03_UNDERSTANDING_UNIQUE,
    W03_UNDERSTANDING_UNKNOWN,
    W03UnderstandingError,
    build_w03_understanding_runtime,
)
from pure_integer_ai.storage.backend import DictBackend
from tests.test_w03_stage2_adapter import _formal_output


def _authored_evidence(logical_order: int):
    """按正式 authored logical order 返回唯一 Evidence binding。"""
    matches = tuple(
        item for item in _formal_output().evidence
        if (item.observation.payload_kind == "SenseBoundaryQuery"
            and item.logical_order == logical_order)
    )
    if len(matches) != 1:
        raise AssertionError("测试资料缺唯一 authored Evidence")
    return matches[0]


def _runtime(output=None):
    """在测试内存图上形成 W03-03 runtime，不进入正式 run 4。"""
    backend = DictBackend()
    context = make_train_context(backend)
    runtime = build_w03_understanding_runtime(
        _formal_output() if output is None else output,
        context.graph_ontology,
    )
    return backend, runtime


def test_w03_understanding_forms_all_candidates_without_preactivation() -> None:
    """59 个 typed 定义只形成 unknown，active consumer 不得旁路 lifecycle。"""
    backend, runtime = _runtime()
    try:
        report = runtime.report()
        assert report.candidate_count == 59
        assert report.applied_observation_evidence_count == 0
        assert report.active_sense_count == 0
        assert sum(
            item.report().prediction_count
            for item in runtime.candidate_runtimes) == 0
        assert all(
            runtime.consumer.lookup(item.anchor.atom, context=item.context) == ()
            for item in runtime.output.candidates
        )
        assert dict(report.execution_state) == {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W03_STARTED": 0,
            "W04_STARTED": 0,
            "formal_w03_training_runs": 0,
            "learning_writes": 0,
            "teacher_calls": 0,
        }
    finally:
        backend.close()


def test_concept_split_revision_supersedes_without_identity_rewrite() -> None:
    """新 context 先形成可澄清分支，再令旧投影退出且不改旧身份。"""
    backend, runtime = _runtime()
    try:
        old_binding = _authored_evidence(1)
        revision_binding = _authored_evidence(4)
        old = runtime.candidate_for_observation(
            old_binding.observation.stable_key)[0]
        revision = runtime.candidate_for_observation(
            revision_binding.observation.stable_key)[0]
        old_observation = old.observation
        old_definition_key = old.definition.stable_key()

        runtime.apply_evidence(old_binding)
        assert runtime.resolve(
            old.anchor.atom, context=old.context).status == (
                W03_UNDERSTANDING_UNIQUE)
        application = runtime.apply_evidence(revision_binding)

        assert application.before_supersede is not None
        assert application.before_supersede.status == W03_UNDERSTANDING_CLARIFY
        assert {item.sense for item in application.before_supersede.active} == {
            old.sense,
            revision.sense,
        }
        assert runtime.resolve(old.anchor.atom).status == W03_UNDERSTANDING_UNIQUE
        assert runtime.resolve(old.anchor.atom).selected.sense == revision.sense
        assert runtime.consumer.lookup(
            old.anchor.atom, context=old.context) == ()
        assert runtime.consumer.require_unique(
            revision.anchor.atom, context=revision.context).sense == revision.sense
        assert old.observation is old_observation
        assert old.definition.stable_key() == old_definition_key
        assert old.sense != revision.sense
        assert old.context != revision.context
        assert application.superseded_candidates == (old.sense,)
        assert runtime.supersedes_observation(revision.sense) == (
            old.observation.stable_key)
    finally:
        backend.close()


def test_polysemy_competition_and_conflict_are_not_sort_first() -> None:
    """同 atom 多义完整保留，冲突分账且不能按稳定序私选一项。"""
    backend, runtime = _runtime()
    try:
        finance_binding = _authored_evidence(1)
        river_binding = _authored_evidence(2)
        conflict_binding = _authored_evidence(3)
        finance = runtime.candidate_for_observation(
            finance_binding.observation.stable_key)[0]
        river = runtime.candidate_for_observation(
            river_binding.observation.stable_key)[0]
        conflict = runtime.candidate_for_observation(
            conflict_binding.observation.stable_key)[0]

        assert finance.anchor.atom == river.anchor.atom
        assert finance.competition_key == river.competition_key
        assert finance.sense != river.sense
        assert finance.concept != river.concept
        runtime.apply_evidence(finance_binding)
        runtime.apply_evidence(river_binding)
        assert runtime.resolve(
            finance.anchor.atom, context=finance.context).selected.sense == (
                finance.sense)

        runtime.apply_evidence(conflict_binding)
        resolution = runtime.resolve(
            conflict.anchor.atom, context=conflict.context)
        assert resolution.status == W03_UNDERSTANDING_UNKNOWN
        assert resolution.selected is None
        accounts = runtime.evidence_accounts(conflict.sense)
        assert tuple(item.stance for item in accounts) == (
            EVIDENCE_SUPPORT, EVIDENCE_REFUTE)
        assert len({item.teacher_record for item in accounts}) == 1
        assert len({item.observation_source for item in accounts}) == 1
        assert len({item.scope for item in accounts}) == 1
        owner = runtime.candidate_runtime_for(conflict.sense)
        hypothesis = owner.hypothesis_for_candidate(conflict.sense)
        assert owner.engine.ledger.snapshot(
            hypothesis).epistemic_status == EPISTEMIC_CONFLICTED
        with pytest.raises(LookupError, match="唯一"):
            runtime.consumer.require_unique(
                conflict.anchor.atom, context=conflict.context)
    finally:
        backend.close()


def test_formal_evidence_produces_ambiguous_unknown_clarify_and_active_readback() -> None:
    """正式 train Evidence 同时覆盖四种查询结果与 Evidence-only 记录。"""
    backend, runtime = _runtime()
    try:
        applications = runtime.apply_all_evidence()
        report = runtime.report()
        entry = next(
            item for item in runtime.output.observations
            if (item.parser_provenance.to_value().get("extractor")
                == "WIKTIONARY_WIKITEXT_SENSE_V1"
                and len(item.candidates) == 3)
        )
        singleton = next(
            item for item in runtime.output.candidates
            if item.external_nondefinitive
            and len(tuple(
                value for value in runtime.output.candidates
                if (value.anchor.atom == item.anchor.atom
                    and value.context == item.context)
            )) == 1
        )
        revision = runtime.candidate_for_observation(
            _authored_evidence(4).observation.stable_key)[0]

        ambiguous = runtime.resolve(
            entry.candidates[0].anchor.atom,
            context=entry.candidates[0].context,
        )
        unknown = runtime.resolve(
            singleton.anchor.atom, context=singleton.context)
        unique = runtime.resolve(
            revision.anchor.atom, context=revision.context)
        clarify = next(
            item.before_supersede for item in applications
            if item.before_supersede is not None)

        assert ambiguous.status == W03_UNDERSTANDING_AMBIGUOUS
        assert len(ambiguous.candidates) == 3
        assert ambiguous.active == ()
        assert ambiguous.clarify_required
        assert unknown.status == W03_UNDERSTANDING_UNKNOWN
        assert len(unknown.candidates) == 1
        assert not unknown.clarify_required
        assert clarify.status == W03_UNDERSTANDING_CLARIFY
        assert clarify.clarify_required
        assert unique.status == W03_UNDERSTANDING_UNIQUE
        assert unique.selected == runtime.consumer.require_unique(
            revision.anchor.atom, context=revision.context)
        assert report.applied_observation_evidence_count == 21
        assert report.unbound_evidence_count == 2
        assert len(runtime.unbound_evidence) == 2
        assert report.source_conflict_candidate_count == 5
        assert set(dict(report.execution_state).values()) == {0}
    finally:
        backend.close()


def test_order_reversal_preserves_outcomes_and_graph_never_merges_nodes() -> None:
    """输入顺序不充当 selector，同 surface 的 Sense/Concept 图节点保持独立。"""
    output = _formal_output()
    reversed_output = replace(
        output,
        observations=tuple(
            replace(item, candidates=tuple(reversed(item.candidates)))
            for item in reversed(output.observations)
        ),
        candidates=tuple(reversed(output.candidates)),
        evidence=tuple(reversed(output.evidence)),
    )
    backend_a, runtime_a = _runtime(output)
    backend_b, runtime_b = _runtime(reversed_output)
    try:
        runtime_a.apply_all_evidence()
        runtime_b.apply_all_evidence()
        authored = tuple(
            item for item in output.candidates
            if item.observation.payload_kind == "SenseBoundaryQuery"
            and item.anchor.extracted.surface == "银行"
        )
        atom = authored[0].anchor.atom
        assert runtime_a.resolve(atom).stable_key() == (
            runtime_b.resolve(atom).stable_key())
        assert runtime_a.candidate_runtime_state_key() == (
            runtime_b.candidate_runtime_state_key())
        assert len({item.sense for item in authored}) == 3
        assert len({
            runtime_a.graph.ontology.resolve(item.sense) for item in authored
        }) == 3
        split_concepts = {
            item.concept for item in authored
            if item.context == authored[0].context
        }
        assert len(split_concepts) == 2
        assert len({
            runtime_a.graph.ontology.resolve(item) for item in split_concepts
        }) == 2
    finally:
        backend_a.close()
        backend_b.close()


def test_duplicate_evidence_application_fails_before_state_change() -> None:
    """同一 teacher binding 不得重复累计或改写既有 Evidence。"""
    backend, runtime = _runtime()
    try:
        binding = _authored_evidence(1)
        runtime.apply_evidence(binding)
        before = runtime.candidate_runtime_state_key()
        with pytest.raises(W03UnderstandingError, match="重复"):
            runtime.apply_evidence(binding)
        assert runtime.candidate_runtime_state_key() == before
    finally:
        backend.close()


def test_bad_supersede_identity_fails_before_candidate_graph_write() -> None:
    """不存在的旧 Observation 必须在形成任何候选图定义前被拒绝。"""
    output = _formal_output()
    target = _authored_evidence(4)
    bad_binding = replace(
        target,
        supersedes_observation_key=StableRecordKey((999999,)),
    )
    bad_output = replace(
        output,
        evidence=tuple(
            bad_binding if item is target else item
            for item in output.evidence
        ),
    )
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        with pytest.raises(W03UnderstandingError, match="supersede"):
            build_w03_understanding_runtime(
                bad_output, context.graph_ontology)
        assert context.graph_ontology.resolve(output.candidates[0].sense) is None
    finally:
        backend.close()


def test_understanding_source_has_no_legacy_evaluator_or_host_route() -> None:
    """理解闭环只依赖 typed adapter/candidate 设施，不接旧表或 host 边。"""
    repository = Path(__file__).resolve().parents[1]
    source = "".join(
        (repository / "src/pure_integer_ai/experiments" / name).read_text(
            encoding="utf-8")
        for name in (
            "ph2_w03_understanding.py",
            "ph2_w03_understanding_contract.py",
        )
    )

    assert "language_sense_candidate_runtime" not in source
    assert "storage.sense_candidates" not in source
    assert "EvaluatorLabelRecord" not in source
    assert "W03PayloadFirewall" not in source
    assert "pathlib" not in source
    assert "transaction" not in source.lower()
