"""R-04 部分整体关系族、显式闭包、课程接线和隔离测试。"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_ENTITY,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.mereology_relation import (
    MereologyBudget,
    MereologyBudgetExceeded,
    MereologyPattern,
    MereologyProtocol,
    MereologyRelationProtocol,
    MereologyStatement,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    CompositionRule,
    InverseRule,
    IrreflexiveRule,
    RelationSchema,
    RelationSlotSchema,
    TransitiveRule,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.mechanism_inventory import (
    STATUS_OPT_IN,
    inventory_by_id,
    readiness_candidates,
    validate_inventory,
)
from pure_integer_ai.experiments.mereology_relation_runtime import (
    LegacyMereologyRecord,
    MappedLegacyMereology,
    MereologyFormationRequest,
    MereologyQuery,
    MereologyRelationRuntime,
    MereologyRoundRequest,
    MereologyRuntimeError,
    install_mereology_relation_runtime,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureIncompleteError,
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.training.stages import STAGE1_SKELETON
from tests.test_r00_relation_closure import (
    _candidate_runtime,
    _projection_protocol,
    _relation_protocol,
    _semantic_graph,
)


def _source(source_id: int) -> SourceRef:
    """构造共享 owner/version 且来源号互异的 R-04 测试来源。"""
    return SourceRef(
        204,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _variant(seed: int) -> MereologyRelationProtocol:
    """构造端点 Role 与 schema slot 顺序相反的二元 relation variant。"""
    relation = relation_concept_identity((10500, seed))
    part_role = role_identity((10510, seed, 1))
    whole_role = role_identity((10510, seed, 2))
    schema = RelationSchema(
        structure_concept_identity((10520, seed)),
        relation,
        (
            RelationSlotSchema(
                whole_role, frozenset({OBJECT_ENTITY}), 1, 1),
            RelationSlotSchema(
                part_role, frozenset({OBJECT_ENTITY}), 1, 1),
        ),
    )
    return MereologyRelationProtocol(schema, part_role, whole_role)


def _protocol() -> tuple[MereologyProtocol, tuple[MereologyRelationProtocol, ...]]:
    """构造四个关系 variant 和全部显式允许规则。"""
    part_of = _variant(1)
    material = _variant(2)
    has_part = _variant(3)
    no_rule = _variant(4)
    protocol = MereologyProtocol(
        (part_of, material, has_part, no_rule),
        transitive_rules=(TransitiveRule(
            minimal_instruction_identity((10530, 1)),
            part_of.relation,
            part_of.part_role,
            part_of.whole_role,
        ),),
        composition_rules=(CompositionRule(
            minimal_instruction_identity((10530, 2)),
            part_of.relation,
            part_of.part_role,
            part_of.whole_role,
            material.relation,
            material.part_role,
            material.whole_role,
            material.relation,
            material.part_role,
            material.whole_role,
        ),),
        inverse_rules=(InverseRule(
            minimal_instruction_identity((10530, 3)),
            part_of.relation,
            part_of.part_role,
            part_of.whole_role,
            has_part.relation,
            has_part.whole_role,
            has_part.part_role,
        ),),
        irreflexive_rules=(IrreflexiveRule(
            minimal_instruction_identity((10530, 4)),
            part_of.relation,
            part_of.part_role,
            part_of.whole_role,
        ),),
    )
    return protocol, (part_of, material, has_part, no_rule)


@dataclass
class _Fixture:
    """保存可追加多关系事实的完整 R-00/R-04 测试设施。"""

    backend: DictBackend
    ctx: object
    protocol: MereologyProtocol
    variants: tuple[MereologyRelationProtocol, ...]
    extra_schema: RelationSchema
    closure: RelationClosureRuntime
    runtime: MereologyRelationRuntime
    objects: tuple[ObjectIdentity, ...]
    next_fact_id: int = 1

    def close(self) -> None:
        """关闭测试后端。"""
        self.backend.close()

    def statement(
            self, relation_index: int, part: int, whole: int,
            ) -> MereologyStatement:
        """按测试域下标构造一个 canonical relation+part+whole statement。"""
        return MereologyStatement(
            self.variants[relation_index].relation,
            self.objects[part],
            self.objects[whole],
        )

    def definition(
            self,
            statement: MereologyStatement,
            source: SourceRef,
            fact_id: int,
            ) -> AtomicPropositionDefinition:
        """把 canonical statement 编成故意逆序的 S-00 RoleBinding。"""
        relation = self.protocol.require_relation(statement.relation)
        return AtomicPropositionDefinition(
            proposition_identity(source, (10540, fact_id)),
            statement.relation,
            occurrence_identity(source, start=0, end=1, ordinal=0),
            context_scope_identity(source, (10541, fact_id)),
            (
                AtomicRoleBinding(relation.whole_role, statement.whole),
                AtomicRoleBinding(relation.part_role, statement.part),
            ),
        )

    def add(
            self,
            statement: MereologyStatement,
            *,
            stance: str,
            archive_refuted: bool = False,
            replacement: ObjectIdentity | None = None,
            competition_key: tuple[int, ...] | None = None,
            ) -> ObjectIdentity:
        """写入 typed 部分整体事实，并可显式归档或替代当前候选。"""
        fact_id = self.next_fact_id
        self.next_fact_id += 1
        source = _source(1000 + fact_id)
        definition = self.definition(statement, source, fact_id)
        relation = self.protocol.require_relation(statement.relation)
        spec = RelationClosureCandidateSpec(
            definition,
            relation.schema,
            competition_key or (10542, fact_id),
            (_source(2000 + fact_id * 2), _source(2001 + fact_id * 2)),
        )
        self.closure.semantic_graph.define_atomic(
            definition,
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        self.closure.form(spec, timestamp_base=fact_id * 10)
        recognition = _recognition_for(
            definition,
            _source(3000 + fact_id),
            fact_id,
            stance=stance,
            archive_refuted=archive_refuted,
            replacement=replacement,
        )
        self.closure.recognize(recognition)
        return definition.proposition

    def add_extra_relation(self, part: int, whole: int) -> ObjectIdentity:
        """写入同为二元 shape 但不属于 MEREOLOGY 协议的关系事实。"""
        fact_id = self.next_fact_id
        self.next_fact_id += 1
        source = _source(5000 + fact_id)
        roles = tuple(slot.role for slot in self.extra_schema.slots)
        definition = AtomicPropositionDefinition(
            proposition_identity(source, (10543, fact_id)),
            self.extra_schema.relation,
            occurrence_identity(source, start=0, end=1, ordinal=0),
            context_scope_identity(source, (10544, fact_id)),
            (
                AtomicRoleBinding(roles[0], self.objects[part]),
                AtomicRoleBinding(roles[1], self.objects[whole]),
            ),
        )
        spec = RelationClosureCandidateSpec(
            definition,
            self.extra_schema,
            (10545, fact_id),
            (_source(6000 + fact_id * 2), _source(6001 + fact_id * 2)),
        )
        self.closure.semantic_graph.define_atomic(
            definition,
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        self.closure.form(spec, timestamp_base=fact_id * 10)
        self.closure.recognize(_recognition_for(
            definition,
            _source(7000 + fact_id),
            fact_id,
            stance="support",
        ))
        return definition.proposition


def _extra_schema() -> RelationSchema:
    """构造 shape 相同但语义不属于部分整体关系族的额外 schema。"""
    roles = (
        role_identity((10550, 1)),
        role_identity((10550, 2)),
    )
    return RelationSchema(
        structure_concept_identity((10551, 1)),
        relation_concept_identity((10552, 1)),
        tuple(
            RelationSlotSchema(role, frozenset({OBJECT_ENTITY}), 1, 1)
            for role in roles
        ),
    )


def _fixture(*, budget: MereologyBudget | None = None) -> _Fixture:
    """建立共享 SemanticGraph、H-05 owner、关系族、规则和测试端点。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        _projection_protocol(),
    )
    candidate_runtime = _candidate_runtime(candidate_graph)
    protocol, variants = _protocol()
    extra = _extra_schema()
    schemas = tuple(item.schema for item in variants) + (extra,)
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        _relation_protocol(),
        schemas,
        engine=candidate_runtime.engine,
    )
    closure = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        _relation_protocol(),
    )
    objects = tuple(
        entity_identity(_source(100 + index), (10560, index))
        for index in range(8)
    )
    runtime = MereologyRelationRuntime(
        closure,
        protocol,
        budget or MereologyBudget(100, 100, 500, 100),
    )
    return _Fixture(
        backend,
        ctx,
        protocol,
        variants,
        extra,
        closure,
        runtime,
        objects,
    )


def _recognition_for(
        definition: AtomicPropositionDefinition,
        observation: SourceRef,
        seed: int,
        *,
        stance: str = "support",
        archive_refuted: bool = False,
        replacement: ObjectIdentity | None = None,
        ) -> RelationClosureRecognitionInput:
    """构造与 forming 分源的显式 support/refute recognition。"""
    anchor = occurrence_identity(
        observation,
        start=0,
        end=1,
        ordinal=0,
    )
    supported = (definition.proposition,) if stance == "support" else ()
    refuted = (definition.proposition,) if stance == "refute" else ()
    return RelationClosureRecognitionInput(
        definition.proposition,
        observation,
        document_scope(observation),
        ProtocolKey((10570, seed)),
        (10571, seed),
        anchor,
        (anchor,),
        RevealedObjectObservation(
            observation,
            document_scope(observation),
            (10571, seed),
            _source(8000 + seed),
            supported_targets=supported,
            refuted_targets=refuted,
            trace=(10572, seed),
        ),
        archive_refuted=archive_refuted,
        replacement=replacement,
    )


def test_canonical_roles_explicit_transitivity_and_no_default_transitivity():
    """方向只看 Role；显式传递成立，未声明同型链保持 unknown。"""
    fixture = _fixture()
    try:
        first = fixture.add(fixture.statement(0, 0, 1), stance="support")
        second = fixture.add(fixture.statement(0, 1, 2), stance="support")
        fixture.add(fixture.statement(3, 0, 1), stance="support")
        fixture.add(fixture.statement(3, 1, 2), stance="support")

        derived = fixture.runtime.query(MereologyQuery(
            MereologyPattern(
                fixture.variants[0].relation,
                fixture.objects[0],
                fixture.objects[2],
            ),
            use_key=(10580, 1),
        ))
        evaluation = derived.selection.evaluations[0]
        assert evaluation.state == LogicEvidenceState(True, False)
        assert evaluation.support_proof is not None
        assert len(evaluation.support_proof.applications) == 1
        assert {item.proposition for item in derived.uses} == {first, second}

        unknown = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[3].relation,
            fixture.objects[0],
            fixture.objects[2],
        )))
        assert unknown.selection.evaluations[0].state == LogicEvidenceState(
            False, False)
    finally:
        fixture.close()


def test_explicit_composition_inverse_and_non_mereology_shape_stay_separate():
    """跨关系复合与逆向只按规则执行，其他同型关系不混入闭包。"""
    fixture = _fixture()
    try:
        direct = fixture.add(fixture.statement(0, 0, 1), stance="support")
        material = fixture.add(fixture.statement(1, 1, 2), stance="support")
        extra = fixture.add_extra_relation(1, 2)

        composed = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[1].relation,
            fixture.objects[0],
            fixture.objects[2],
        )))
        assert composed.selection.evaluations[0].state.support is True
        assert {
            item.proposition
            for item in composed.selection.evaluations[0].active_premises()
        } == {direct, material}

        inverse = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[2].relation,
            fixture.objects[0],
            fixture.objects[1],
        )))
        assert inverse.selection.evaluations[0].state == LogicEvidenceState(
            True, False)
        assert tuple(
            trace.spec.proposition.predicate
            for trace in fixture.closure.formation_traces()
        ).count(fixture.variants[2].relation) == 0
        assert extra not in {
            item.proposition for item in fixture.runtime.knowledge().evidence}

        unlisted = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[3].relation,
            fixture.objects[0],
            fixture.objects[2],
        )))
        assert unlisted.selection.evaluations[0].state == LogicEvidenceState(
            False, False)
    finally:
        fixture.close()


def test_multi_whole_conflict_archived_refute_and_refute_do_not_propagate():
    """多 whole 合法；冲突保留；反证不沿链传播且只纯支持写 Use。"""
    fixture = _fixture()
    try:
        first = fixture.add(fixture.statement(3, 0, 1), stance="support")
        second = fixture.add(fixture.statement(3, 0, 2), stance="support")
        fixture.add(
            fixture.statement(3, 0, 3),
            stance="refute",
            archive_refuted=True,
        )
        fixture.add(fixture.statement(3, 0, 4), stance="support")
        fixture.add(
            fixture.statement(3, 0, 4),
            stance="refute",
            archive_refuted=True,
        )
        result = fixture.runtime.query(MereologyQuery(
            MereologyPattern(
                relation=fixture.variants[3].relation,
                part=fixture.objects[0],
            ),
            use_key=(10581, 1),
        ))
        states = {
            item.statement.whole: item.state
            for item in result.selection.evaluations
        }
        assert states[fixture.objects[1]] == LogicEvidenceState(True, False)
        assert states[fixture.objects[2]] == LogicEvidenceState(True, False)
        assert states[fixture.objects[3]] == LogicEvidenceState(False, True)
        assert states[fixture.objects[4]] == LogicEvidenceState(True, True)
        assert len(result.selection.pure_supported()) == 2
        assert {item.proposition for item in result.uses} == {first, second}

        fixture.add(fixture.statement(0, 5, 6), stance="support")
        fixture.add(
            fixture.statement(0, 6, 7),
            stance="refute",
            archive_refuted=True,
        )
        no_refute_closure = fixture.runtime.query(MereologyQuery(
            MereologyPattern(
                fixture.variants[0].relation,
                fixture.objects[5],
                fixture.objects[7],
            )
        ))
        assert no_refute_closure.selection.evaluations[0].state == (
            LogicEvidenceState(False, False))
    finally:
        fixture.close()


def test_irreflexive_rule_handles_direct_and_derived_cycles_only_when_declared():
    """显式反自反使直接/闭包自环冲突，未声明 relation 的自环仍可纯支持。"""
    fixture = _fixture()
    try:
        fixture.add(fixture.statement(0, 0, 0), stance="support")
        direct_loop = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[0].relation,
            fixture.objects[0],
            fixture.objects[0],
        )))
        assert direct_loop.selection.evaluations[0].state == LogicEvidenceState(
            True, True)

        fixture.add(fixture.statement(0, 1, 2), stance="support")
        fixture.add(fixture.statement(0, 2, 1), stance="support")
        derived_loop = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[0].relation,
            fixture.objects[1],
            fixture.objects[1],
        )))
        assert derived_loop.selection.evaluations[0].state == LogicEvidenceState(
            True, True)

        fixture.add(fixture.statement(3, 3, 3), stance="support")
        allowed_loop = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[3].relation,
            fixture.objects[3],
            fixture.objects[3],
        )))
        assert allowed_loop.selection.evaluations[0].state == LogicEvidenceState(
            True, False)
    finally:
        fixture.close()


def test_derived_support_direct_refute_conflicts_and_writes_no_use():
    """派生支持遇同 statement 直接反证必须保留冲突，不能消费任一前提。"""
    fixture = _fixture()
    try:
        fixture.add(fixture.statement(0, 0, 1), stance="support")
        fixture.add(fixture.statement(0, 1, 2), stance="support")
        fixture.add(
            fixture.statement(0, 0, 2),
            stance="refute",
            archive_refuted=True,
        )
        result = fixture.runtime.query(MereologyQuery(
            MereologyPattern(
                fixture.variants[0].relation,
                fixture.objects[0],
                fixture.objects[2],
            ),
            use_key=(10582, 1),
        ))
        evaluation = result.selection.evaluations[0]
        assert evaluation.state == LogicEvidenceState(True, True)
        assert evaluation.support_proof is not None
        assert evaluation.rule_refutes == ()
        assert result.uses == ()
    finally:
        fixture.close()


def test_all_budget_dimensions_fail_closed():
    """直接事实、闭包、规则应用和返回选项任一耗尽都拒绝部分结果。"""
    direct = _fixture(budget=MereologyBudget(1, 100, 500, 100))
    try:
        direct.add(direct.statement(3, 0, 1), stance="support")
        direct.add(direct.statement(3, 0, 2), stance="support")
        with pytest.raises(MereologyBudgetExceeded, match="直接事实"):
            direct.runtime.query(MereologyQuery(MereologyPattern()))
    finally:
        direct.close()

    closure = _fixture(budget=MereologyBudget(10, 2, 500, 100))
    try:
        closure.add(closure.statement(0, 0, 1), stance="support")
        closure.add(closure.statement(0, 1, 2), stance="support")
        with pytest.raises(MereologyBudgetExceeded, match="闭包 statement"):
            closure.runtime.query(MereologyQuery(MereologyPattern()))
    finally:
        closure.close()

    applications = _fixture(budget=MereologyBudget(10, 100, 1, 100))
    try:
        applications.add(
            applications.statement(0, 0, 1), stance="support")
        applications.add(
            applications.statement(0, 1, 2), stance="support")
        with pytest.raises(MereologyBudgetExceeded, match="规则应用"):
            applications.runtime.query(MereologyQuery(MereologyPattern()))
    finally:
        applications.close()

    options = _fixture(budget=MereologyBudget(10, 100, 500, 1))
    try:
        options.add(options.statement(3, 0, 1), stance="support")
        options.add(options.statement(3, 0, 2), stance="support")
        with pytest.raises(MereologyBudgetExceeded, match="option"):
            options.runtime.query(MereologyQuery(MereologyPattern(
                relation=options.variants[3].relation,
                part=options.objects[0],
            )))
    finally:
        options.close()


class _MappedLegacy:
    """返回固定映射结果的 legacy mapper 测试替身。"""

    def __init__(self, mapped):
        self.mapped = mapped

    def map(self, record):
        """返回预设映射，不自行修正错误。"""
        return self.mapped

    def clone_for_evaluation(self):
        """复制固定映射替身。"""
        return _MappedLegacy(self.mapped)

    def state_key(self):
        """返回固定 mapper 版本键。"""
        return 10590, 1


def test_superseded_excluded_and_legacy_mapper_preserves_schema_source_scope():
    """superseded 不进入查询；legacy 必须补全当前 schema 且保留来源 scope。"""
    fixture = _fixture()
    try:
        replacement = fixture.add(
            fixture.statement(3, 0, 2),
            stance="unknown",
            competition_key=(10591, 1),
        )
        fixture.add(
            fixture.statement(3, 0, 1),
            stance="refute",
            replacement=replacement,
            competition_key=(10591, 1),
        )
        old_result = fixture.runtime.query(MereologyQuery(MereologyPattern(
            fixture.variants[3].relation,
            fixture.objects[0],
            fixture.objects[1],
        )))
        assert old_result.selection.evaluations[0].state == LogicEvidenceState(
            False, False)

        source = _source(9000)
        statement = fixture.statement(3, 3, 4)
        definition = fixture.definition(statement, source, 500)
        formation = MereologyFormationRequest(
            RelationClosureCandidateSpec(
                definition,
                fixture.variants[3].schema,
                (10592, 1),
                (_source(9001), _source(9002)),
            ),
            document_scope(source),
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
        )
        mapped = MappedLegacyMereology(
            formation,
            _recognition_for(definition, _source(9003), 501),
        )
        record = LegacyMereologyRecord(
            (1,), (2,), (3,), source, document_scope(source), (4,),
        )
        assert fixture.runtime.map_legacy(record, _MappedLegacy(mapped)) == mapped

        wrong_source = _source(9010)
        wrong_definition = fixture.definition(statement, wrong_source, 501)
        wrong = MappedLegacyMereology(
            replace(
                formation,
                spec=replace(
                    formation.spec,
                    proposition=wrong_definition,
                ),
                scope=document_scope(wrong_source),
            ),
            _recognition_for(wrong_definition, _source(9011), 502),
        )
        with pytest.raises(MereologyRuntimeError, match="不得替换"):
            fixture.runtime.map_legacy(record, _MappedLegacy(wrong))

        wrong_relation = fixture.add_extra_relation(3, 4)
        extra_formation = next(
            trace for trace in fixture.closure.formation_traces()
            if trace.spec.proposition.proposition == wrong_relation
        )
        invalid = MappedLegacyMereology(
            MereologyFormationRequest(
                extra_formation.spec,
                document_scope(extra_formation.spec.proposition.source),
                SOURCE_BARE_TEXT,
                EPI_STRUCTURED,
            ),
            _recognition_for(
                extra_formation.spec.proposition,
                _source(9012),
                503,
            ),
        )
        with pytest.raises(MereologyRuntimeError, match="typed mereology"):
            fixture.runtime.map_legacy(record, _MappedLegacy(invalid))

        stale_schema = RelationSchema(
            structure_concept_identity((10593, 1)),
            fixture.variants[3].relation,
            fixture.variants[3].schema.slots,
        )
        stale = MappedLegacyMereology(
            replace(
                formation,
                spec=replace(formation.spec, schema=stale_schema),
            ),
            mapped.recognition,
        )
        with pytest.raises(MereologyRuntimeError, match="typed mereology"):
            fixture.runtime.map_legacy(record, _MappedLegacy(stale))
    finally:
        fixture.close()


@dataclass(frozen=True)
class _RuntimeBuilder:
    """在任意 TrainContext 图上重建同一 R-04/R-00 协议。"""

    protocol: MereologyProtocol
    budget: MereologyBudget
    bound: MereologyRelationRuntime | None = None

    def build(self, ctx) -> MereologyRelationRuntime:
        """复用同图 bound owner，否则建立新的 SemanticGraph 和 H-05 owner。"""
        if (self.bound is not None
                and self.bound.semantic_graph.ontology is ctx.graph_ontology):
            return self.bound
        semantic_graph = _semantic_graph(ctx.graph_ontology)
        candidate_graph = CandidateProjectionGraph(
            ctx.graph_ontology,
            _projection_protocol(),
        )
        candidate_runtime = _candidate_runtime(candidate_graph)
        consumer = ActiveRelationClosureConsumer(
            semantic_graph,
            candidate_graph,
            _relation_protocol(),
            tuple(item.schema for item in self.protocol.relations),
            engine=candidate_runtime.engine,
        )
        closure = RelationClosureRuntime(
            candidate_runtime,
            semantic_graph,
            consumer,
            _relation_protocol(),
        )
        return MereologyRelationRuntime(
            closure,
            self.protocol,
            self.budget,
        )

    def clone_for_evaluation(self):
        """复制不可变协议并清除宿主 owner 引用。"""
        return _RuntimeBuilder(self.protocol, self.budget)

    def state_key(self):
        """返回协议、规则和预算的完整纯整数键。"""
        return (
            10600,
            *self.protocol.stable_key(),
            *self.budget.stable_key(),
        )


@dataclass(frozen=True)
class _EmptyCourse:
    """任何 scope 都返回空 typed 请求的安装和隔离测试课程。"""

    version: int

    def request(self, scope, *, read_only):
        """返回不学习、不查询的空 R-04 round。"""
        return MereologyRoundRequest(scope)

    def legacy_mapper(self):
        """声明当前课程不迁移旧边。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变空课程。"""
        return _EmptyCourse(self.version)

    def state_key(self):
        """返回课程版本键。"""
        return 10601, self.version


@dataclass(frozen=True)
class _DriftingCourse(_EmptyCourse):
    """故意在评测 clone 中改变版本的隔离负例课程。"""

    def clone_for_evaluation(self):
        """故意改变版本，验证 runtime 拒绝评测状态漂移。"""
        return _DriftingCourse(self.version + 1)


@dataclass(frozen=True)
class _StaticCourse:
    """返回测试注入请求的不可变 R-04 课程。"""

    round_request: MereologyRoundRequest
    version: int

    def request(self, scope, *, read_only):
        """仅在目标 scope 和写入轮返回预设请求。"""
        if scope != self.round_request.scope or read_only:
            return MereologyRoundRequest(scope)
        return self.round_request

    def legacy_mapper(self):
        """声明测试课程不迁移旧边。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变请求和版本。"""
        return _StaticCourse(self.round_request, self.version)

    def state_key(self):
        """返回测试课程的固定版本键。"""
        return 10602, self.version


def _formal_request(
        protocol: MereologyProtocol,
        relation: MereologyRelationProtocol,
        part: ObjectIdentity,
        whole: ObjectIdentity,
        source: SourceRef,
        ) -> MereologyRoundRequest:
    """构造与一个 formal 语言 item 来源绑定的 typed R-04 写入请求。"""
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (10610, 1)),
        relation.relation,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (10610, 2)),
        (
            AtomicRoleBinding(relation.whole_role, whole),
            AtomicRoleBinding(relation.part_role, part),
        ),
    )
    formation = MereologyFormationRequest(
        RelationClosureCandidateSpec(
            definition,
            relation.schema,
            (10610, 3),
            (_source(9201), _source(9202)),
        ),
        document_scope(source),
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
    )
    recognition = _recognition_for(definition, source, 601)
    return MereologyRoundRequest(
        document_scope(source),
        formations=(formation,),
        recognitions=(recognition,),
    )


def test_high_timestamp_preflight_and_v06_clone_boundaries():
    """逻辑序后移、失败预演零写、clone 可读且宿主状态守恒。"""
    fixture = _fixture()
    try:
        source = _source(9300)
        request = _formal_request(
            fixture.protocol,
            fixture.variants[3],
            fixture.objects[0],
            fixture.objects[1],
            source,
        )
        formation = replace(request.formations[0], timestamp_base=100_000)
        request = replace(request, formations=(formation,))
        installed = install_mereology_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime,
            ),
            _StaticCourse(request, 1),
        )
        report = installed.process(request.scope, read_only=False)
        forming_end = (
            formation.timestamp_base + len(formation.spec.forming_sources) - 1)
        assert report.recognitions[0].outcome.evidence.timestamp_seq > forming_end

        baseline = installed.state_key()
        report_count = len(fixture.ctx.mereology_relation_reports)
        with isolated_evaluation(fixture.ctx, label="r04-held-out") as eval_ctx:
            result = eval_ctx.mereology_relation_runtime.owner.query(
                MereologyQuery(
                    MereologyPattern(
                        fixture.variants[3].relation,
                        fixture.objects[0],
                        fixture.objects[1],
                    ),
                    use_key=(10620, 1),
                )
            )
            assert result.selection.pure_supported()
            assert len(result.uses) == 1
        assert installed.state_key() == baseline
        assert len(fixture.ctx.mereology_relation_reports) == report_count

        with pytest.raises(ValueError, match="必须分开"):
            MereologyRoundRequest(
                request.scope,
                formations=(formation,),
                queries=(MereologyQuery(MereologyPattern()),),
            )
    finally:
        fixture.close()

    failed = _fixture()
    try:
        source = _source(9310)
        request = _formal_request(
            failed.protocol,
            failed.variants[3],
            failed.objects[0],
            failed.objects[1],
            source,
        )
        missing = proposition_identity(source, (10621, 1))
        request = replace(
            request,
            recognitions=(replace(
                request.recognitions[0], proposition=missing),),
        )
        installed = install_mereology_relation_runtime(
            failed.ctx,
            _RuntimeBuilder(
                failed.protocol,
                failed.runtime.budget,
                failed.runtime,
            ),
            _StaticCourse(request, 1),
        )
        baseline_backend = failed.backend.snapshot()
        baseline_owner = failed.closure.state_key()
        with pytest.raises(RelationClosureIncompleteError, match="缺少 forming"):
            installed.process(request.scope, read_only=False)
        assert failed.backend.snapshot() == baseline_backend
        assert failed.closure.state_key() == baseline_owner
        assert failed.closure.semantic_graph.ontology.resolve(
            request.formations[0].spec.proposition.proposition
        ) is None
    finally:
        failed.close()


def test_v06_rejects_course_drift_and_formal_train_reports_inventory(
        tmp_path, monkeypatch):
    """V-06 拒绝课程漂移，formal 成对安装并保持台账 opt-in。"""
    fixture = _fixture()
    try:
        install_mereology_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime,
            ),
            _DriftingCourse(1),
        )
        with pytest.raises(ValueError, match="改变课程状态"):
            with isolated_evaluation(fixture.ctx, label="r04-drift"):
                pass
    finally:
        fixture.close()

    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="r04-partial",
        active_training_stages=(),
        persist_graph_dump=False,
        language_mereology_relation_builder=object(),
    )
    with pytest.raises(ValueError, match="必须成对配置"):
        formal_train(config, [], backend=DictBackend())

    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    protocol, variants = _protocol()
    source = _source(9400)
    part = entity_identity(_source(9401), (10630, 1))
    whole = entity_identity(_source(9402), (10630, 2))
    request = _formal_request(
        protocol, variants[3], part, whole, source)
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r04-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            persist_graph_dump=False,
            language_occurrence_protocol=OccurrenceProtocol((10631, 1)),
            language_mereology_relation_builder=_RuntimeBuilder(
                protocol,
                MereologyBudget(20, 50, 100, 20),
            ),
            language_mereology_relation_course=_StaticCourse(request, 2),
        ),
        [CollectedItem(
            tokens=["部件", "整体"],
            raw_text="部件整体",
            role_seq=[1, 1],
            source=source.source_kind,
            source_ref=source,
        )],
        backend=DictBackend(),
    )
    assert result.mereology_relation_reports
    assert result.mereology_relation_reports[-1].recognitions[0].active_fact is not None

    record = inventory_by_id()["relation.mereology_typed_closure"]
    candidates = {item.mechanism_id for item in readiness_candidates()}
    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert "FormalTrainConfig.language_mereology_relation_builder" in record.gates
    assert "V06" in record.recovery[-1]
    assert "K 盘" in record.limitation
    assert record.mechanism_id not in candidates
    assert validate_inventory() == ()
