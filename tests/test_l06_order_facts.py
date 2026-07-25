"""L-06 分型、来源化顺序事实测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.order_facts import OrderFact, OrderFactIndex
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_OBSERVATION,
    LogicalClock,
    LogicalClockIdentity,
    document_scope,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderIntegrityError,
    OccurrenceOrderProtocol,
    OccurrenceOrderReader,
    OccurrenceOrderWriter,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES
from pure_integer_ai.training.cursor import dump_run, load_run
from pure_integer_ai.training.stages import STAGE1_SKELETON, STAGE3_REWARD


def _backend(kind: str):
    """为两类后端建立独立测试存储。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend(":memory:")
    raise ValueError(kind)


def _source(parser: int) -> SourceRef:
    """构造同一文档在不同 parser version 下的来源身份。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        101,
        7,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _occurrences(ctx, source: SourceRef, raw_text: str):
    """为顺序事实测试物化两个来源 occurrence。"""
    index = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        OccurrenceProtocol((98001,)),
    )
    scope = document_scope(source)
    first = index.record(
        source=source,
        raw_text=raw_text,
        scope=scope,
        start=0,
        end=1,
        ordinal=0,
        segment_index=0,
        local_index=0,
        document_index=0,
    )
    second = index.record(
        source=source,
        raw_text=raw_text,
        scope=scope,
        start=1,
        end=2,
        ordinal=0,
        segment_index=0,
        local_index=1,
        document_index=1,
    )
    return scope, first.occurrence, second.occurrence


def _install_occurrence_order(ctx):
    """给训练上下文装配 L-03 occurrence 与 L-06 来源位置序读写器。"""
    ctx.occurrence_index = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        OccurrenceProtocol((98300, 1)),
    )
    facts = OrderFactIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
    )
    protocol = OccurrenceOrderProtocol((98300, 2))
    ctx.occurrence_order_reader = OccurrenceOrderReader(
        facts,
        ctx.occurrence_index,
        protocol,
    )
    ctx.occurrence_order_writer = OccurrenceOrderWriter(
        facts,
        protocol,
    )
    return ctx.occurrence_order_writer


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_order_predicates_are_separate_scoped_facts_without_strength(kind: str):
    """多种序使用不同一等 predicate，不能靠宽枚举、strength 或全局 order_index 混型。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        source = _source(1)
        scope, first, second = _occurrences(ctx, source, "甲乙")
        facts = OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        )
        relations = tuple(
            relation_concept_identity((98100, index))
            for index in range(1, 6)
        )
        edge_rows_before = backend.select("edge", where=None)

        recorded = tuple(facts.record(
            relation,
            first,
            second,
            scope=scope,
            provenance_kind=source.source_kind,
            content_version=source.versions.parser.value,
            qualifiers=(0, 1),
        ) for relation in relations)

        assert len({fact.assertion_hash for fact in recorded}) == 5
        for relation, fact in zip(relations, recorded):
            assert facts.facts(relation, scope=scope) == (fact,)
            assert fact.statement.assertion.content_version == 1
            assert fact.statement.assertion.qualifiers == (0, 1)
        assert backend.select("edge", where=None) == edge_rows_before
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_occurrence_order_reparse_supersedes_without_deleting_history(kind: str):
    """重新解析可替代旧位置序，旧 statement 和旧 scope 仍可完整回读。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        facts = OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        )
        writer = OccurrenceOrderWriter(
            facts,
            OccurrenceOrderProtocol((98200, 1)),
        )
        old_source = _source(1)
        new_source = _source(2)
        old_scope, old_first, old_second = _occurrences(
            ctx, old_source, "甲乙")
        new_scope, new_first, new_second = _occurrences(
            ctx, new_source, "乙甲")
        old = writer.record_adjacent(
            old_first,
            old_second,
            source=old_source,
            scope=old_scope,
            previous_position=0,
            current_position=1,
        )
        assert writer.record_adjacent(
            old_first,
            old_second,
            source=old_source,
            scope=old_scope,
            previous_position=0,
            current_position=1,
        ) == old
        new = writer.record_adjacent(
            new_first,
            new_second,
            source=new_source,
            scope=new_scope,
            previous_position=0,
            current_position=1,
        )
        clock = LogicalClock(LogicalClockIdentity(
            new_scope,
            CLOCK_OBSERVATION,
        ))

        event_hash = facts.supersede(old, new, clock.advance())

        assert event_hash > 0
        assert writer.facts_in_scope(old_scope) == ()
        assert writer.facts_in_scope(
            old_scope, active_only=False) == (old,)
        assert writer.facts_in_scope(new_scope) == (new,)
        assert writer.count(active_only=False) == 2
        assert writer.count(active_only=True) == 1
    finally:
        backend.close()


def test_supersede_rejects_registered_assertion_without_graph_statement():
    """仅登记 assertion 不能伪造可替代事实，事实必须真实存在于当前图。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        facts = OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        )
        source = _source(1)
        scope, first, second = _occurrences(ctx, source, "甲乙")
        relation = relation_concept_identity((98250, 1))
        real = facts.record(
            relation,
            first,
            second,
            scope=scope,
            provenance_kind=source.source_kind,
            content_version=source.versions.parser.value,
            qualifiers=(0, 1),
        )
        forged_assertion = replace(
            real.statement.assertion,
            qualifiers=(0, 2),
        )
        forged_hash = ctx.scoped_identity_store.register_assertion(
            forged_assertion)
        forged = OrderFact(replace(
            real.statement,
            assertion_hash=forged_hash,
            assertion=forged_assertion,
        ))
        clock = LogicalClock(LogicalClockIdentity(
            scope,
            CLOCK_OBSERVATION,
        ))

        with pytest.raises(ValueError, match="当前图中唯一存在"):
            facts.supersede(forged, real, clock.advance())
    finally:
        backend.close()


def test_supersede_rejects_cross_predicate_replacement():
    """不同顺序类型的生命周期必须分离，不能用替代事件重新混型。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        facts = OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        )
        source = _source(1)
        scope, first, second = _occurrences(ctx, source, "甲乙")
        old = facts.record(
            relation_concept_identity((98260, 1)),
            first,
            second,
            scope=scope,
            provenance_kind=source.source_kind,
        )
        other_kind = facts.record(
            relation_concept_identity((98260, 2)),
            first,
            second,
            scope=scope,
            provenance_kind=source.source_kind,
        )
        clock = LogicalClock(LogicalClockIdentity(
            scope,
            CLOCK_OBSERVATION,
        ))

        with pytest.raises(ValueError, match="不同顺序 predicate"):
            facts.supersede(old, other_kind, clock.advance())
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_observe_writes_idempotent_order_without_cross_document_folding(kind: str):
    """同来源重放幂等，跨文档 A-B/B-A 使用不同 occurrence 端点和 scope。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        writer = _install_occurrence_order(ctx)
        source_ab = SourceRef(
            SOURCE_BARE_TEXT, 201, 1,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(parser=ParserVersion(3)),
        )
        source_ba = SourceRef(
            SOURCE_BARE_TEXT, 201, 2,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(parser=ParserVersion(3)),
        )
        item_ab = CollectedItem(
            tokens=["甲", "乙"],
            raw_text="甲乙",
            role_seq=[1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
            source_ref=source_ab,
        )
        item_ba = CollectedItem(
            tokens=["乙", "甲"],
            raw_text="乙甲",
            role_seq=[1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
            source_ref=source_ba,
        )
        runner = DefaultRoundRunner()

        runner.run_round(ctx, item_ab, STAGE1_SKELETON, 0)
        runner.run_round(ctx, item_ab, STAGE1_SKELETON, 1)
        runner.run_round(ctx, item_ba, STAGE1_SKELETON, 0)

        scope_ab = document_scope(source_ab)
        scope_ba = document_scope(source_ba)
        facts_ab = writer.facts_in_scope(scope_ab)
        facts_ba = writer.facts_in_scope(scope_ba)
        assert len(facts_ab) == len(facts_ba) == 1
        assert facts_ab[0].statement.subject != facts_ba[0].statement.object
        assert facts_ab[0].statement.object != facts_ba[0].statement.subject
        assert facts_ab[0].statement.assertion.qualifiers == (0, 1)
        assert facts_ba[0].statement.assertion.qualifiers == (0, 1)
        assert facts_ab[0].statement.assertion.content_version == 3
        assert facts_ba[0].statement.assertion.content_version == 3
        assert writer.count() == 2

        expected_ab = (
            ctx.graph_ontology.resolve(occurrence_identity(
                source_ab, start=0, end=1, ordinal=0)),
            ctx.graph_ontology.resolve(occurrence_identity(
                source_ab, start=1, end=2, ordinal=0)),
        )
        assert (
            facts_ab[0].statement.subject,
            facts_ab[0].statement.object,
        ) == expected_ab
        chain_ab = ctx.occurrence_order_reader.read_chain(scope_ab)
        assert chain_ab.occurrences == expected_ab
        assert tuple(
            record.document_index for record in chain_ab.records) == (0, 1)
        assert tuple(
            fact.assertion_hash for fact in chain_ab.facts
        ) == (facts_ab[0].assertion_hash,)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_occurrence_order_reader_rejects_gap_branch_and_cross_source(kind: str):
    """reader 遇位置 gap、分支或跨来源端点时必须拒绝返回局部链。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        index = OccurrenceIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            OccurrenceProtocol((98350, 1)),
        )
        facts = OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        )
        protocol = OccurrenceOrderProtocol((98350, 2))
        writer = OccurrenceOrderWriter(facts, protocol)
        reader = OccurrenceOrderReader(facts, index, protocol)

        gap_source = SourceRef(
            SOURCE_BARE_TEXT, 401, 1,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(parser=ParserVersion(5)),
        )
        gap_scope = document_scope(gap_source)
        gap_first = index.record(
            source=gap_source, raw_text="甲乙丙", scope=gap_scope,
            start=0, end=1, ordinal=0,
            segment_index=0, local_index=0, document_index=0,
        )
        gap_last = index.record(
            source=gap_source, raw_text="甲乙丙", scope=gap_scope,
            start=2, end=3, ordinal=0,
            segment_index=0, local_index=2, document_index=2,
        )
        writer.record_adjacent(
            gap_first.occurrence,
            gap_last.occurrence,
            source=gap_source,
            scope=gap_scope,
            previous_position=0,
            current_position=2,
        )
        with pytest.raises(OccurrenceOrderIntegrityError, match="gap"):
            reader.read_chain(gap_scope)

        branch_source = SourceRef(
            SOURCE_BARE_TEXT, 401, 2,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(parser=ParserVersion(5)),
        )
        branch_scope = document_scope(branch_source)
        branch_first = index.record(
            source=branch_source, raw_text="甲乙丙", scope=branch_scope,
            start=0, end=1, ordinal=0,
            segment_index=0, local_index=0, document_index=0,
        )
        branch_second = index.record(
            source=branch_source, raw_text="甲乙丙", scope=branch_scope,
            start=1, end=2, ordinal=0,
            segment_index=0, local_index=1, document_index=1,
        )
        branch_alternate = index.record(
            source=branch_source, raw_text="甲乙丙", scope=branch_scope,
            start=2, end=3, ordinal=0,
            segment_index=1, local_index=0, document_index=1,
        )
        for current in (branch_second, branch_alternate):
            writer.record_adjacent(
                branch_first.occurrence,
                current.occurrence,
                source=branch_source,
                scope=branch_scope,
                previous_position=0,
                current_position=1,
            )
        with pytest.raises(OccurrenceOrderIntegrityError, match="分支"):
            reader.read_chain(branch_scope)

        cross_source = SourceRef(
            SOURCE_BARE_TEXT, 401, 3,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(parser=ParserVersion(5)),
        )
        cross_scope = document_scope(cross_source)
        cross_first = index.record(
            source=cross_source, raw_text="甲", scope=cross_scope,
            start=0, end=1, ordinal=0,
            segment_index=0, local_index=0, document_index=0,
        )
        other_source = SourceRef(
            SOURCE_BARE_TEXT, 401, 4,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(parser=ParserVersion(5)),
        )
        other_scope = document_scope(other_source)
        other = index.record(
            source=other_source, raw_text="乙", scope=other_scope,
            start=0, end=1, ordinal=0,
            segment_index=0, local_index=0, document_index=0,
        )
        facts.record(
            writer.relation,
            cross_first.occurrence,
            other.occurrence,
            scope=cross_scope,
            provenance_kind=cross_source.source_kind,
            content_version=cross_source.versions.parser.value,
            qualifiers=(0, 1),
        )
        with pytest.raises(OccurrenceOrderIntegrityError, match="跨越"):
            reader.read_chain(cross_scope)
    finally:
        backend.close()


def test_order_facts_round_trip_and_evaluation_clone_are_isolated(tmp_path):
    """顺序 statement 可恢复，评测 clone 的新增事实不回写宿主。"""
    backend = DictBackend()
    restored_backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        writer = _install_occurrence_order(ctx)
        source = _source(4)
        item = CollectedItem(
            tokens=["甲", "乙"],
            raw_text="甲乙",
            role_seq=[1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
            source_ref=source,
        )
        DefaultRoundRunner().run_round(
            ctx, item, STAGE1_SKELETON, 0)
        assert writer.count() == 1
        dump_run(
            backend,
            str(tmp_path),
            "l06",
            spaces=[ctx.space_id],
        )

        restored_ctx = make_train_context(restored_backend)
        load_run(restored_backend, str(tmp_path), "l06")
        restored_writer = _install_occurrence_order(restored_ctx)
        assert len(restored_writer.facts_in_scope(
            document_scope(source))) == 1
        assert len(restored_ctx.occurrence_order_reader.read_chain(
            document_scope(source)).records) == 2

        host_count = writer.count()
        with isolated_evaluation(ctx, label="l06") as eval_ctx:
            eval_source = SourceRef(
                SOURCE_BARE_TEXT, 301, 9,
                GLOBAL_OWNER_SCOPE,
                VersionBundle(parser=ParserVersion(4)),
            )
            eval_item = CollectedItem(
                tokens=["乙", "甲"],
                raw_text="乙甲",
                role_seq=[1, 1],
                collect_type=COLLECT_PRECEDES,
                source=SOURCE_BARE_TEXT,
                source_ref=eval_source,
            )
            DefaultRoundRunner().run_round(
                eval_ctx, eval_item, STAGE1_SKELETON, 0)
            assert eval_ctx.occurrence_order_writer.count() == 2
            assert len(eval_ctx.occurrence_order_reader.read_chain(
                document_scope(eval_source)).records) == 2

        assert writer.count() == host_count
    finally:
        backend.close()
        restored_backend.close()


def test_formal_train_requires_l03_and_reports_order_facts(tmp_path, monkeypatch):
    """顶层入口只有同时配置 L-03/L-06 才写来源位置序并报告计数。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    item = CollectedItem(
        tokens=["甲", "乙", "丙"],
        raw_text="甲乙丙",
        role_seq=[1, 1, 1],
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
    )
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="l06_formal",
        rounds_per_stage=2,
        active_training_stages=(STAGE1_SKELETON,),
        language_occurrence_protocol=OccurrenceProtocol((98400, 1)),
        language_occurrence_order_protocol=OccurrenceOrderProtocol((98400, 2)),
    )

    result = formal_train(
        config,
        [item],
        backend=DictBackend(),
        runner=DefaultRoundRunner(),
    )

    assert result.occurrence_count == 3
    assert result.occurrence_order_fact_count == 2

    invalid = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="l06_invalid",
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        language_occurrence_order_protocol=OccurrenceOrderProtocol((98400, 3)),
    )
    with pytest.raises(ValueError, match="L-03 occurrence"):
        formal_train(
            invalid,
            [CollectedItem(
                tokens=["甲", "乙"],
                raw_text="甲乙",
                role_seq=[1, 1],
                collect_type=COLLECT_PRECEDES,
                source=SOURCE_BARE_TEXT,
            )],
            backend=DictBackend(),
            runner=DefaultRoundRunner(),
        )


def test_occurrence_order_verifier_ignores_legacy_token_cycle():
    """occurrence cue 验序不得读取旧 token 宽边，即使宽边形成反向环。"""
    from pure_integer_ai.config import gates

    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_time = gates.TIME_SEQ_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.TIME_SEQ_PROOF_MODE = True
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _install_occurrence_order(ctx)
        item = CollectedItem(
            tokens=["a", "然后", "b"],
            raw_text="a然后b",
            role_seq=[1, 1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
        )
        runner = DefaultRoundRunner()
        runner.run_round(ctx, item, STAGE1_SKELETON, 0)
        a = ctx.concept_index.lookup("a", ctx.space_id)
        b = ctx.concept_index.lookup("b", ctx.space_id)
        assert a is not None and b is not None
        ctx.edge_store.add(
            space_id_from=b[0],
            local_id_from=b[1],
            space_id_to=a[0],
            local_id_to=a[1],
            edge_type=EDGE_PRECEDES,
            source=SOURCE_BARE_TEXT,
            order_index=99,
        )

        result = runner.run_round_full(ctx, item, STAGE3_REWARD, 1)

        assert result.episode is not None
        assert result.episode.reward == 1
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.TIME_SEQ_PROOF_MODE = saved_time
        backend.close()


def test_time_verifier_requires_typed_occurrence_order_reader():
    """关闭 typed reader 后事件时序验证必须退化，不能回读旧 PRECEDES。"""
    from pure_integer_ai.config import gates

    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_time = gates.TIME_SEQ_PROOF_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.TIME_SEQ_PROOF_MODE = True
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _install_occurrence_order(ctx)
        item = CollectedItem(
            tokens=["a", "然后", "b"],
            raw_text="a然后b",
            role_seq=[1, 1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
        )
        runner = DefaultRoundRunner()
        runner.run_round(ctx, item, STAGE1_SKELETON, 0)
        ctx.occurrence_order_reader = None

        result = runner.run_round_full(ctx, item, STAGE3_REWARD, 1)

        assert result.episode is None
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.TIME_SEQ_PROOF_MODE = saved_time
        backend.close()
