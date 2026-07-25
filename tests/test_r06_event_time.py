"""R-06 Event/Proposition scoped 时间事实和独立 verifier 对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_BEFORE,
    EVENT_TIME_CONFLICTED,
    EVENT_TIME_CONSISTENT,
    EVENT_TIME_DIRECTION_UNKNOWN,
    EVENT_TIME_SAME,
    EVENT_TIME_UNKNOWN,
    EventTimeFactIndex,
    EventTimeVerifier,
    ResolvedEventTimeRelation,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.order_facts import OrderFactIndex
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_OBSERVATION,
    LogicalClock,
    LogicalClockIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    event_identity,
    proposition_identity,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationAdapter,
    EventTimeVerificationProtocol,
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_NOT_APPLICABLE,
    MultiVerifierOrchestrator,
    VERDICT_CONFLICTED,
    VERDICT_SUPPORT,
    VerificationEvaluation,
    VerifierRegistration,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.training.cursor import dump_run, load_run


_BASE = 31900


def _source(document_id: int, *, parser: int = 1) -> SourceRef:
    """构造同一数据源中的版本化语义文档。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        _BASE + 1,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


class _Resolver:
    """按显式 relation 注册表提供最小方向执行语义。"""

    def __init__(self, directions):
        self.directions = dict(directions)

    def resolve(self, relation):
        """返回 relation 自身、注入方向和独立审计键。"""
        return ResolvedEventTimeRelation(
            relation,
            self.directions[relation],
            (_BASE + 2, self.directions[relation]),
        )


def _runtime(backend, directions):
    """在当前权威图上装配 event-time typed facade 和 verifier。"""
    ctx = make_train_context(backend)
    facts = EventTimeFactIndex(OrderFactIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
    ))
    return ctx, facts, EventTimeVerifier(facts, _Resolver(directions))


def _record(facts, relation, first, second, scope):
    """以统一 provenance 写入一条测试时间事实。"""
    return facts.record(
        relation,
        first,
        second,
        scope=scope,
        provenance_kind=_BASE + 3,
        content_version=scope.source.versions.parser.value,
    )


def test_event_time_same_and_before_are_normalized_without_occurrence_edges():
    """same 等价类与方向边可组合，Occurrence 端点被 typed facade 拒绝。"""
    backend = DictBackend()
    try:
        before = concept_identity((_BASE + 10, 1))
        same = concept_identity((_BASE + 10, 2))
        ctx, facts, verifier = _runtime(
            backend,
            {before: EVENT_TIME_BEFORE, same: EVENT_TIME_SAME},
        )
        source = _source(1)
        scope = document_scope(source)
        first = event_identity(source, (_BASE + 11, 1))
        peer = proposition_identity(source, (_BASE + 11, 2))
        last = event_identity(source, (_BASE + 11, 3))
        _record(facts, same, first, peer, scope)
        _record(facts, before, peer, last, scope)

        result = verifier.verify((same, before), scope=scope)

        assert result.status == EVENT_TIME_CONSISTENT
        assert set(result.same_groups[0]) == {first, peer}
        assert result.before_edges == ((min(first, peer), last),)
        with pytest.raises(ValueError, match="Event 或 Proposition"):
            _record(
                facts,
                before,
                occurrence_identity(source, start=0, end=1, ordinal=0),
                last,
                scope,
            )
    finally:
        backend.close()


def test_event_time_conflict_and_unknown_are_preserved_independently():
    """同序内方向、自反压缩和未知 relation 均保留原事实而不选唯一序。"""
    backend = DictBackend()
    try:
        before = concept_identity((_BASE + 20, 1))
        same = concept_identity((_BASE + 20, 2))
        unknown = concept_identity((_BASE + 20, 3))
        _, facts, verifier = _runtime(
            backend,
            {
                before: EVENT_TIME_BEFORE,
                same: EVENT_TIME_SAME,
                unknown: EVENT_TIME_DIRECTION_UNKNOWN,
            },
        )
        source = _source(2)
        scope = document_scope(source)
        first = event_identity(source, (_BASE + 21, 1))
        second = event_identity(source, (_BASE + 21, 2))
        same_fact = _record(facts, same, first, second, scope)
        conflict = _record(facts, before, first, second, scope)

        conflicted = verifier.verify((same, before), scope=scope)

        assert conflicted.status == EVENT_TIME_CONFLICTED
        assert conflict.assertion_hash in conflicted.conflict_assertion_hashes
        assert same_fact in conflicted.fact_set.facts

        third = proposition_identity(source, (_BASE + 21, 3))
        _record(facts, unknown, second, third, scope)
        unknown_result = verifier.verify((unknown,), scope=scope)
        assert unknown_result.status == EVENT_TIME_UNKNOWN
        assert unknown_result.unknown_relations == (unknown,)
        assert unknown_result.before_edges == ()
    finally:
        backend.close()


def test_event_time_cycle_attribution_excludes_downstream_edges():
    """方向环只归因真正位于环上的 assertion，不污染单向下游事实。"""
    backend = DictBackend()
    try:
        before = concept_identity((_BASE + 25, 1))
        _, facts, verifier = _runtime(backend, {before: EVENT_TIME_BEFORE})
        source = _source(25)
        scope = document_scope(source)
        first = event_identity(source, (_BASE + 26, 1))
        second = event_identity(source, (_BASE + 26, 2))
        third = event_identity(source, (_BASE + 26, 3))
        fourth = proposition_identity(source, (_BASE + 26, 4))
        cycle_forward = _record(facts, before, first, second, scope)
        cycle_reverse = _record(facts, before, second, first, scope)
        downstream_first = _record(facts, before, second, third, scope)
        downstream_second = _record(facts, before, third, fourth, scope)

        result = verifier.verify((before,), scope=scope)

        assert result.status == EVENT_TIME_CONFLICTED
        assert result.conflict_assertion_hashes == tuple(sorted((
            cycle_forward.assertion_hash,
            cycle_reverse.assertion_hash,
        )))
        assert downstream_first.assertion_hash not in (
            result.conflict_assertion_hashes)
        assert downstream_second.assertion_hash not in (
            result.conflict_assertion_hashes)
    finally:
        backend.close()


def test_event_time_supersede_is_append_only_and_scope_exact():
    """parser 修正显式 supersede 旧方向，旧 statement 保留但 active 查询隔离。"""
    backend = DictBackend()
    try:
        before = concept_identity((_BASE + 30, 1))
        _, facts, _ = _runtime(backend, {before: EVENT_TIME_BEFORE})
        old_source = _source(3, parser=1)
        new_source = _source(3, parser=2)
        old_scope = document_scope(old_source)
        new_scope = document_scope(new_source)
        old = _record(
            facts,
            before,
            event_identity(old_source, (_BASE + 31, 1)),
            event_identity(old_source, (_BASE + 31, 2)),
            old_scope,
        )
        new = _record(
            facts,
            before,
            event_identity(new_source, (_BASE + 31, 2)),
            event_identity(new_source, (_BASE + 31, 1)),
            new_scope,
        )
        clock = LogicalClock(LogicalClockIdentity(
            new_scope,
            CLOCK_OBSERVATION,
        ))

        facts.supersede(old, new, clock.advance())

        assert facts.read((before,), scope=old_scope).facts == ()
        assert facts.read((before,), scope=old_scope, active_only=False).facts == (
            old,)
        assert facts.read((before,), scope=new_scope).facts == (new,)
    finally:
        backend.close()


def test_event_time_r09_registration_is_independent_and_order_invariant():
    """event-time conflicted 不覆盖其他维 support，调换注册顺序结果不变。"""
    backend = DictBackend()
    try:
        before = concept_identity((_BASE + 40, 1))
        _, facts, verifier = _runtime(backend, {before: EVENT_TIME_BEFORE})
        source = _source(4)
        scope = document_scope(source)
        first = event_identity(source, (_BASE + 41, 1))
        second = event_identity(source, (_BASE + 41, 2))
        _record(facts, before, first, second, scope)
        _record(facts, before, second, first, scope)
        event_protocol = EventTimeVerificationProtocol(
            ProtocolKey((_BASE + 42, 1)),
            ProtocolKey((_BASE + 42, 2)),
        )
        event_registration = EventTimeVerificationAdapter(
            verifier, event_protocol).registration()
        other_registration = VerifierRegistration(
            ProtocolKey((_BASE + 43, 1)),
            ProtocolKey((_BASE + 43, 2)),
            applies=lambda _request: True,
            evaluate=lambda _request: VerificationEvaluation(
                VERDICT_SUPPORT,
                claim_keys=((_BASE + 43, 3),),
                source=source,
                scope=scope,
            ),
        )
        request = EventTimeVerificationRequest(scope, (before,))
        orchestrator = MultiVerifierOrchestrator()

        first_report = orchestrator.run(
            request,
            (event_registration, other_registration),
            read_only=True,
        )
        second_report = orchestrator.run(
            request,
            (other_registration, event_registration),
            read_only=True,
        )

        assert first_report.verdict_key() == second_report.verdict_key()
        assert first_report.dimension_results(
            event_protocol.dimension)[0].verdict == VERDICT_CONFLICTED
        assert first_report.dimension_results(
            other_registration.dimension)[0].verdict == VERDICT_SUPPORT

        empty_source = _source(5)
        empty_report = orchestrator.run(
            EventTimeVerificationRequest(
                document_scope(empty_source),
                (before,),
            ),
            (event_registration,),
            read_only=True,
        )
        assert empty_report.results[0].applicability == (
            APPLICABILITY_NOT_APPLICABLE)
    finally:
        backend.close()


def test_event_time_conflict_survives_graph_dump_load(tmp_path):
    """图恢复后 typed reader 仍得到相同事实、方向和冲突来源。"""
    backend = DictBackend()
    restored_backend = DictBackend()
    try:
        before = concept_identity((_BASE + 50, 1))
        ctx, facts, verifier = _runtime(
            backend, {before: EVENT_TIME_BEFORE})
        source = _source(6)
        scope = document_scope(source)
        first = event_identity(source, (_BASE + 51, 1))
        second = proposition_identity(source, (_BASE + 51, 2))
        _record(facts, before, first, second, scope)
        _record(facts, before, second, first, scope)
        expected = verifier.verify((before,), scope=scope)
        dump_run(
            backend,
            str(tmp_path),
            "r06-event-time",
            spaces=[ctx.space_id],
        )

        restored_ctx = make_train_context(restored_backend)
        load_run(restored_backend, str(tmp_path), "r06-event-time")
        restored_facts = EventTimeFactIndex(OrderFactIndex(
            restored_ctx.graph_ontology,
            restored_ctx.scoped_identity_store,
        ))
        restored = EventTimeVerifier(
            restored_facts,
            _Resolver({before: EVENT_TIME_BEFORE}),
        ).verify((before,), scope=scope)

        assert restored.status == expected.status == EVENT_TIME_CONFLICTED
        assert restored.before_edges == expected.before_edges
        assert restored.conflict_assertion_hashes == (
            expected.conflict_assertion_hashes)
        assert tuple(
            fact.statement.assertion.stable_key()
            for fact in restored.fact_set.facts
        ) == tuple(
            fact.statement.assertion.stable_key()
            for fact in expected.fact_set.facts
        )
    finally:
        backend.close()
        restored_backend.close()
