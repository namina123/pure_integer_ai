"""S-02 span/occurrence 到 typed 语义候选、ledger 和图物化测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisTransition,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    OBJECT_SPAN,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    entity_identity,
    role_identity,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    LocalSemanticRef,
    SemanticBindingSpec,
    SemanticBuildError,
    SemanticBuildPlan,
    SemanticBuilderProtocol,
    SemanticCandidateBuilder,
    SemanticCandidateLedgerAdapter,
    SemanticFillerSpec,
    SemanticGuidanceEvidence,
    SemanticObjectSpec,
    SemanticPropositionSpec,
)
from pure_integer_ai.cognition.understanding.semantic_builder_graph import (
    SemanticBuilderGraphError,
    SemanticBuilderTracePredicates,
    SemanticCandidateGraphAdapter,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _backend(kind: str):
    """为 S-02 持久化契约建立独立后端。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend(":memory:")
    raise ValueError(kind)


def _source(document_id: int = 1, *, parser: int = 3) -> SourceRef:
    """构造显式绑定 parser version 的来源身份。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        21801,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _indexes(backend, *, document_id: int = 1):
    """装配同一 ontology 上的 L-03/L-04 真源并建立两个 anchor。"""
    context = make_train_context(backend)
    occurrences = OccurrenceIndex(
        context.graph_ontology,
        context.scoped_identity_store,
        OccurrenceProtocol((21810, 1), (21810, 2)),
    )
    spans = SpanIndex(
        context.graph_ontology,
        context.scoped_identity_store,
        SpanProtocol(
            (21811, 1),
            (21811, 2),
            (21811, 3),
            (21811, 4),
        ),
        occurrences,
    )
    source = _source(document_id)
    scope = document_scope(source)
    raw_text = "甲乙丙"
    occurrence = occurrences.record(
        source=source,
        raw_text=raw_text,
        scope=scope,
        start=0,
        end=1,
        ordinal=0,
        segment_index=0,
        local_index=0,
        document_index=0,
    ).occurrence
    span = spans.ensure_ref(
        source=source,
        raw_text=raw_text,
        scope=scope,
        members=((0, 3),),
    )
    return context, occurrences, spans, source, scope, occurrence, span


def _upstream(source: SourceRef, variant: int = 1) -> HypothesisKey:
    """构造同一 parse 竞争组中的一个来源化上游候选。"""
    return HypothesisKey(
        (21820, 1),
        (21820, variant),
        (21820, 9),
        document_scope(source),
        source,
    )


def _protocol(*, builder_key: int = 1, primitive_version: int = 0):
    """注入 builder MinimalInstruction 和 S-02 候选 kind。"""
    versions = VersionBundle(
        primitive=PrimitiveVersion(primitive_version))
    return SemanticBuilderProtocol(
        minimal_instruction_identity(
            (21821, builder_key), versions=versions),
        (21821, 99),
    )


def _plan(
        upstream: HypothesisKey, occurrence=None, *, reverse: bool = False,
        predicate_shift: int = 0, entity_key: int = 1,
        ) -> SemanticBuildPlan:
    """构造含三类局部对象和 Proposition 前向嵌套的开放计划。"""
    objects = (
        SemanticObjectSpec(OBJECT_ENTITY, (entity_key,)),
        SemanticObjectSpec(OBJECT_EVENT, (2,)),
        SemanticObjectSpec(OBJECT_SET_EXPR, (3,)),
    )
    outer = SemanticPropositionSpec(
        local_key=(10,),
        competition_key=(21830, 1),
        predicate=concept_identity((21831, 1 + predicate_shift)),
        structure=structure_concept_identity((21832, 1)),
        bindings=(SemanticBindingSpec(
            role_identity((21833, 1)),
            SemanticFillerSpec(
                local_ref=LocalSemanticRef(OBJECT_PROPOSITION, (20,))),
        ),),
    )
    inner = SemanticPropositionSpec(
        local_key=(20,),
        competition_key=(21830, 1),
        predicate=concept_identity((21831, 2 + predicate_shift)),
        structure=structure_concept_identity((21832, 2)),
        bindings=(SemanticBindingSpec(
            role_identity((21833, 2)),
            SemanticFillerSpec(
                local_ref=LocalSemanticRef(OBJECT_ENTITY, (entity_key,))),
        ),),
        source_anchor=occurrence,
    )
    if reverse:
        objects = tuple(reversed(objects))
        propositions = (inner, outer)
    else:
        propositions = (outer, inner)
    return SemanticBuildPlan(
        upstream,
        (21834, 1),
        objects,
        propositions,
    )


def _single_plan(
        upstream: HypothesisKey, *, local_key: int,
        filler_key: int = 1,
        ) -> SemanticBuildPlan:
    """构造一个可与其他 parse 共享竞争组的单命题计划。"""
    return SemanticBuildPlan(
        upstream,
        (21840, 1),
        (),
        (SemanticPropositionSpec(
            local_key=(local_key,),
            competition_key=(21840, 9),
            predicate=concept_identity((21841, 1)),
            structure=structure_concept_identity((21842, 1)),
            bindings=(SemanticBindingSpec(
                role_identity((21843, 1)),
                SemanticFillerSpec(external=entity_identity(
                    upstream.observation, (filler_key,))),
            ),),
        ),),
    )


def _graph_adapter(context, *, family: int = 21850):
    """在同一 ontology 中注入互异的 S-00 和 S-02 图 predicate。"""
    refs = tuple(
        context.graph_ontology.materialize(
            relation_concept_identity((family, ordinal)))
        for ordinal in range(1, 10)
    )
    graph = SemanticGraph(
        context.graph_ontology,
        AtomicPropositionPredicates(*refs[:6]),
    )
    adapter = SemanticCandidateGraphAdapter(
        graph,
        SemanticBuilderTracePredicates(*refs[6:]),
    )
    return graph, adapter, refs


def test_span_and_occurrence_anchors_build_typed_objects_and_nested_proposition():
    """Span/Occurrence 均可作根，三类对象和 Proposition filler 保持一等身份。"""
    backend = DictBackend()
    try:
        _, occurrences, spans, source, scope, occurrence, span = _indexes(backend)
        builder = SemanticCandidateBuilder(spans, _protocol(), occurrences)
        plan = _plan(_upstream(source), occurrence)

        from_span = builder.compile(span, plan)
        from_occurrence = builder.compile(occurrence, plan)

        assert from_span.scope == scope
        assert from_span.root_anchor.object_kind == OBJECT_SPAN
        assert from_occurrence.root_anchor.object_kind == OBJECT_OCCURRENCE
        assert {item.identity.object_kind for item in from_span.objects} == {
            OBJECT_ENTITY,
            OBJECT_EVENT,
            OBJECT_SET_EXPR,
        }
        outer, inner = from_span.propositions
        assert outer.definition.bindings[0].filler == inner.definition.proposition
        assert inner.definition.source_anchor.object_kind == OBJECT_OCCURRENCE
    finally:
        backend.close()


def test_source_scope_and_anchor_type_mismatches_fail_closed():
    """上游来源、局部 anchor scope 和非 Span/Occurrence 根均不得隐式改写。"""
    backend = DictBackend()
    try:
        context, occurrences, spans, source, _, occurrence, span = _indexes(backend)
        builder = SemanticCandidateBuilder(spans, _protocol(), occurrences)
        wrong_upstream = _upstream(_source(2))
        with pytest.raises(SemanticBuildError, match="来源或 scope"):
            builder.compile(span, _plan(wrong_upstream, occurrence))

        wrong_occurrence = occurrences.record(
            source=_source(3),
            raw_text="丁",
            scope=document_scope(_source(3)),
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
        ).occurrence
        with pytest.raises(SemanticBuildError, match="Proposition anchor"):
            builder.compile(span, _plan(_upstream(source), wrong_occurrence))

        bad_anchor = context.graph_ontology.materialize(
            concept_identity((21860, 1)))
        with pytest.raises(SemanticBuildError, match="Span 或 Occurrence"):
            builder.compile(bad_anchor, _plan(_upstream(source), occurrence))
    finally:
        backend.close()


def test_compiler_is_deterministic_and_shape_excludes_content_identity():
    """计划容器顺序不影响结果；替换 predicate/filler 内容不改变逻辑形状。"""
    backend = DictBackend()
    try:
        _, occurrences, spans, source, _, occurrence, span = _indexes(backend)
        builder = SemanticCandidateBuilder(spans, _protocol(), occurrences)
        forward = builder.compile(span, _plan(_upstream(source), occurrence))
        reversed_result = builder.compile(
            span, _plan(_upstream(source), occurrence, reverse=True))
        assert forward.objects == reversed_result.objects
        assert forward.propositions == reversed_result.propositions

        first = builder.compile(
            span, _single_plan(_upstream(source), local_key=70, filler_key=1))
        second = builder.compile(
            span, _single_plan(_upstream(source), local_key=71, filler_key=2))
        shifted = replace(
            second.propositions[0].spec,
            predicate=concept_identity((21841, 99)),
        )
        shifted_plan = replace(
            _single_plan(_upstream(source), local_key=71, filler_key=2),
            propositions=(shifted,),
        )
        shifted_result = builder.compile(span, shifted_plan)
        assert first.propositions[0].shape_key() == (
            shifted_result.propositions[0].shape_key())
        assert first.propositions[0].definition != (
            shifted_result.propositions[0].definition)
    finally:
        backend.close()


def test_builder_identity_and_version_are_part_of_every_semantic_identity():
    """builder key 或 primitive version 改变时，局部对象、context 和命题均不合并。"""
    backend = DictBackend()
    try:
        _, occurrences, spans, source, _, occurrence, span = _indexes(backend)
        plan = _plan(_upstream(source), occurrence)
        first = SemanticCandidateBuilder(
            spans, _protocol(), occurrences).compile(span, plan)
        changed_key = SemanticCandidateBuilder(
            spans, _protocol(builder_key=2), occurrences).compile(span, plan)
        changed_version = SemanticCandidateBuilder(
            spans,
            _protocol(primitive_version=1),
            occurrences,
        ).compile(span, plan)

        assert first.context != changed_key.context != changed_version.context
        assert tuple(item.identity for item in first.objects) != tuple(
            item.identity for item in changed_key.objects)
        assert tuple(item.definition.proposition for item in first.propositions) != tuple(
            item.definition.proposition for item in changed_version.propositions)
    finally:
        backend.close()


def test_undeclared_local_ref_duplicate_role_slot_and_bad_payload_are_rejected():
    """未声明引用、同 Role+ordinal 竞争 filler 和坏 guidance payload 必须明确失败。"""
    backend = DictBackend()
    try:
        _, occurrences, spans, source, _, occurrence, span = _indexes(backend)
        builder = SemanticCandidateBuilder(spans, _protocol(), occurrences)
        plan = _single_plan(_upstream(source), local_key=1)
        missing = replace(
            plan.propositions[0],
            bindings=(SemanticBindingSpec(
                role_identity((21870, 1)),
                SemanticFillerSpec(
                    local_ref=LocalSemanticRef(OBJECT_ENTITY, (404,))),
            ),),
        )
        with pytest.raises(SemanticBuildError, match="未声明"):
            builder.compile(span, replace(plan, propositions=(missing,)))

        role = role_identity((21870, 2))
        duplicate = replace(
            plan.propositions[0],
            bindings=(
                SemanticBindingSpec(
                    role,
                    SemanticFillerSpec(external=entity_identity(source, (1,))),
                ),
                SemanticBindingSpec(
                    role,
                    SemanticFillerSpec(external=entity_identity(source, (2,))),
                ),
            ),
        )
        with pytest.raises(ValueError, match="Role 和 ordinal"):
            builder.compile(span, replace(plan, propositions=(duplicate,)))

        hypothesis = builder.compile(span, plan).propositions[0].hypothesis
        with pytest.raises(SemanticBuildError, match="payload"):
            SemanticGuidanceEvidence(
                hypothesis,
                1,
                (21870, 3),
                source,
                1,
                payload=[1],
            )
    finally:
        backend.close()


def test_multiple_upstream_parses_coexist_and_later_evidence_can_supersede():
    """同竞争组解析并存；后续 refute 和 replacement 才能显式退出旧候选。"""
    backend = DictBackend()
    try:
        _, occurrences, spans, source, _, _, span = _indexes(backend)
        builder = SemanticCandidateBuilder(spans, _protocol(), occurrences)
        first = builder.compile(
            span, _single_plan(_upstream(source, 1), local_key=1))
        second = builder.compile(
            span, _single_plan(_upstream(source, 2), local_key=2))
        ledger = HypothesisLedger()
        adapter = SemanticCandidateLedgerAdapter(ledger)
        adapter.register_unknown(first)
        adapter.register_unknown(second)
        old = first.propositions[0].hypothesis
        replacement = second.propositions[0].hypothesis

        assert tuple(
            item.hypothesis for item in ledger.competition(old)
        ) == tuple(sorted((old, replacement)))
        ledger.append_evidence(EvidenceRecord(
            901,
            old,
            EVIDENCE_REFUTE,
            (21880, 1),
            source,
            1,
        ))
        ledger.append_transition(HypothesisTransition(
            902,
            old,
            LIFECYCLE_ACTIVE,
            LIFECYCLE_SUPERSEDED,
            901,
            (21880, 2),
            2,
            replacement,
        ))
        assert ledger.snapshot(old).lifecycle == LIFECYCLE_SUPERSEDED
        assert ledger.snapshot(replacement).lifecycle == LIFECYCLE_ACTIVE
    finally:
        backend.close()


def test_guidance_stays_unknown_and_batch_conflict_preflight_writes_nothing():
    """cue guidance 只能形成 unknown；后项 Evidence 冲突不得留下候选或前项事件。"""
    backend = DictBackend()
    try:
        _, occurrences, spans, source, _, _, span = _indexes(backend)
        builder = SemanticCandidateBuilder(spans, _protocol(), occurrences)
        result = builder.compile(span, _plan(_upstream(source)))
        ledger = HypothesisLedger()
        adapter = SemanticCandidateLedgerAdapter(ledger)
        first, second = result.propositions
        conflict = (
            SemanticGuidanceEvidence(
                first.hypothesis, 911, (21881, 1), source, 1),
            SemanticGuidanceEvidence(
                second.hypothesis, 911, (21881, 2), source, 2),
        )
        baseline = ledger.state_key()
        with pytest.raises(ValueError, match="evidence_id"):
            adapter.register_unknown(result, conflict)
        assert ledger.state_key() == baseline

        knowledge = adapter.register_unknown(result, (
            SemanticGuidanceEvidence(
                first.hypothesis, 912, (21881, 3), source, 3),
        ))
        assert knowledge[0].snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        assert knowledge[0].snapshot.support_evidence_ids == ()
        assert knowledge[0].snapshot.refute_evidence_ids == ()
        assert knowledge[0].snapshot.unknown_evidence_ids == (912,)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_graph_materializes_full_trace_unbound_objects_and_no_evidence(kind: str):
    """两类后端完整保存结构/upstream/builder，未绑定对象也物化且不造 Evidence。"""
    backend = _backend(kind)
    try:
        context, occurrences, spans, source, _, occurrence, span = _indexes(backend)
        result = SemanticCandidateBuilder(
            spans, _protocol(), occurrences).compile(
                span, _plan(_upstream(source), occurrence))
        graph, adapter, _ = _graph_adapter(context)
        ledger = HypothesisLedger()
        SemanticCandidateLedgerAdapter(ledger).register_unknown(result)
        ledger_before = ledger.state_key()

        written = adapter.materialize(
            result,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            content_version=4,
            qualifiers=(17, 23),
        )
        snapshot = backend.snapshot()
        replayed = adapter.materialize(
            result,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            content_version=4,
            qualifiers=(17, 23),
        )

        assert {ref.object_kind for ref in written.objects} == {
            OBJECT_ENTITY,
            OBJECT_EVENT,
            OBJECT_SET_EXPR,
        }
        assert written == replayed
        assert backend.snapshot() == snapshot
        assert ledger.state_key() == ledger_before
        restored_by_proposition = {
            item.atomic.definition.proposition: item
            for item in written.candidates
        }
        for candidate in result.propositions:
            restored = restored_by_proposition[
                candidate.definition.proposition]
            assert restored.atomic.definition == candidate.definition
            assert restored.structure == candidate.spec.structure
            assert restored.upstream_hypothesis == (
                candidate.upstream_hypothesis.object_identity())
            assert restored.builder == candidate.builder
            assert len(restored.trace_assertion_hashes) == 3

        context.graph_ontology.clear_runtime_caches()
        proposition = context.graph_ontology.resolve(
            result.propositions[0].definition.proposition)
        assert proposition is not None
        assert adapter.read(proposition).atomic == graph.read_atomic(proposition)
    finally:
        backend.close()


def test_graph_lookup_by_concept_filler_recovers_all_sources_without_selection():
    """同一 Concept filler 可跨来源反查全部命题，adapter 只去重不私选。"""
    backend = DictBackend()
    try:
        context, occurrences, spans, source, scope, _, span = _indexes(backend)
        other_source = _source(2)
        other_scope = document_scope(other_source)
        other_occurrence = occurrences.record(
            source=other_source,
            raw_text="乙丙",
            scope=other_scope,
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
        ).occurrence
        other_span = spans.ensure_ref(
            source=other_source,
            raw_text="乙丙",
            scope=other_scope,
            members=((0, 2),),
        )
        filler = concept_identity((21892, 1))

        def build_plan(observation, local_key, source_anchor=None):
            """把同一开放 Concept 注入不同来源命题的 Role filler。"""
            plan = _single_plan(
                _upstream(observation), local_key=local_key)
            proposition = replace(
                plan.propositions[0],
                bindings=(SemanticBindingSpec(
                    plan.propositions[0].bindings[0].role,
                    SemanticFillerSpec(external=filler),
                ),),
                source_anchor=source_anchor,
            )
            return replace(plan, propositions=(proposition,))

        builder = SemanticCandidateBuilder(spans, _protocol(), occurrences)
        first = builder.compile(span, build_plan(source, 1))
        second = builder.compile(
            other_span, build_plan(other_source, 2, other_occurrence))
        _, adapter, _ = _graph_adapter(context)
        written = tuple(
            adapter.materialize(
                result,
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            ).candidates[0]
            for result in (first, second)
        )

        assert adapter.lookup(fillers=(filler,)) == tuple(sorted(
            written,
            key=lambda item: item.atomic.definition.proposition.stable_key(),
        ))
        assert adapter.lookup(
            anchors=(context.graph_ontology.identity_of(span),),
            fillers=(filler,),
        ) == tuple(sorted(
                written,
                key=lambda item: item.atomic.definition.proposition.stable_key(),
            ))
    finally:
        backend.close()


def test_graph_partial_or_competing_trace_fails_before_batch_write():
    """已有单条竞争 trace 时，整批命题均不得先写或自动补齐。"""
    backend = DictBackend()
    try:
        context, occurrences, spans, source, scope, _, span = _indexes(backend)
        result = SemanticCandidateBuilder(
            spans, _protocol(), occurrences).compile(
                span, _plan(_upstream(source)))
        graph, adapter, refs = _graph_adapter(context)
        candidate = result.propositions[1]
        proposition = context.graph_ontology.materialize(
            candidate.definition.proposition)
        wrong = context.graph_ontology.materialize(
            structure_concept_identity((21890, 99)))
        context.graph_ontology.relate(
            refs[6],
            proposition,
            wrong,
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        baseline = backend.snapshot()

        with pytest.raises(SemanticBuilderGraphError, match="部分拓扑"):
            adapter.materialize(
                result,
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            )
        assert backend.snapshot() == baseline
        for item in result.propositions:
            ref = context.graph_ontology.resolve(item.definition.proposition)
            if ref is not None:
                assert context.graph_ontology.statements(
                    predicate=graph.predicates.proposition_predicate,
                    subject=ref,
                ) == ()
    finally:
        backend.close()


def test_existing_atomic_without_builder_trace_is_not_silently_repaired():
    """独立 S-00 定义缺 S-02 provenance 时必须报部分定义，不能追加三边伪装完整。"""
    backend = DictBackend()
    try:
        context, occurrences, spans, source, scope, _, span = _indexes(backend)
        result = SemanticCandidateBuilder(
            spans, _protocol(), occurrences).compile(
                span, _single_plan(_upstream(source), local_key=1))
        graph, adapter, _ = _graph_adapter(context)
        candidate = result.propositions[0]
        graph.define_atomic(
            candidate.definition,
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        baseline = backend.snapshot()

        with pytest.raises(SemanticBuilderGraphError, match="拒绝部分修补"):
            adapter.materialize(
                result,
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            )
        assert backend.snapshot() == baseline
    finally:
        backend.close()


def test_competing_complete_trace_slot_is_never_resolved_by_first_row():
    """完整 provenance 中再出现第二个 structure 端点时，reader 和重放均须拒绝。"""
    backend = DictBackend()
    try:
        context, occurrences, spans, source, scope, _, span = _indexes(backend)
        result = SemanticCandidateBuilder(
            spans, _protocol(), occurrences).compile(
                span, _single_plan(_upstream(source), local_key=1))
        _, adapter, refs = _graph_adapter(context)
        written = adapter.materialize(
            result,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        proposition = written.candidates[0].atomic.proposition
        competing = context.graph_ontology.materialize(
            structure_concept_identity((21891, 1)))
        context.graph_ontology.relate(
            refs[6],
            proposition,
            competing,
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        baseline = backend.snapshot()

        with pytest.raises(SemanticBuilderGraphError, match="实际 2"):
            adapter.read(proposition)
        with pytest.raises(SemanticBuilderGraphError, match="竞争端点"):
            adapter.materialize(
                result,
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            )
        assert backend.snapshot() == baseline
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_semantic_builder_graph_roundtrips_through_authoritative_dump(
        kind: str, tmp_path):
    """S-02 只依赖通用图表，Dict/SQLite dump/load 后可恢复完整 provenance。"""
    first_backend = _backend(kind)
    try:
        first, occurrences, spans, source, _, occurrence, span = _indexes(
            first_backend)
        result = SemanticCandidateBuilder(
            spans, _protocol(), occurrences).compile(
                span, _plan(_upstream(source), occurrence))
        _, adapter, predicate_refs = _graph_adapter(first)
        written = adapter.materialize(
            result,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            content_version=5,
        )
        predicate_identities = tuple(
            first.graph_ontology.identity_of(ref) for ref in predicate_refs)
        proposition_identities = tuple(
            item.definition.proposition for item in result.propositions)
        dump_run(
            first_backend,
            str(tmp_path),
            "run_s02",
            spaces=[first.space_id],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    second_backend = _backend(kind)
    try:
        second = make_train_context(second_backend)
        assert load_run(second_backend, str(tmp_path), "run_s02") == [1]
        restored_refs = tuple(
            second.graph_ontology.resolve(identity)
            for identity in predicate_identities
        )
        graph = SemanticGraph(
            second.graph_ontology,
            AtomicPropositionPredicates(*restored_refs[:6]),
        )
        adapter = SemanticCandidateGraphAdapter(
            graph,
            SemanticBuilderTracePredicates(*restored_refs[6:]),
        )
        restored = {
            identity: adapter.read(second.graph_ontology.resolve(identity))
            for identity in proposition_identities
        }
        written_by_proposition = {
            item.atomic.definition.proposition: item
            for item in written.candidates
        }
        assert set(restored) == set(written_by_proposition)
        for identity, item in restored.items():
            expected = written_by_proposition[identity]
            assert item.atomic.definition == expected.atomic.definition
            assert (
                item.structure,
                item.upstream_hypothesis,
                item.builder,
            ) == (
                expected.structure,
                expected.upstream_hypothesis,
                expected.builder,
            )
    finally:
        second_backend.close()
