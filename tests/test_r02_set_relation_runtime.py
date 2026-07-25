"""R-02 集合关系四态闭包、有限域、Use 和旧边迁移测试。"""
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
    OBJECT_SET_EXPR,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import SemanticTopologyError
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
    set_expr_identity,
)
from pure_integer_ai.cognition.shared.set_relation import (
    MappingMemberTypeResolver,
    SetBinaryRelationProtocol,
    SetQuantifierBranch,
    SetRelationBudget,
    SetRelationBudgetExceeded,
    SetRelationProtocol,
    SetRelationRules,
    SetRelationStatement,
    SetUnaryRelationProtocol,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureIncompleteError,
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.set_relation_runtime import (
    LegacySetRelationRecord,
    MappedLegacySetRelation,
    SetRelationDomainQuery,
    SetRelationFormationRequest,
    SetRelationQuery,
    SetRelationRuntime,
    SetRelationRoundRequest,
    SetRelationRuntimeError,
    install_set_relation_runtime,
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
    """构造共享 owner/version 且来源号互异的 R-02 测试来源。"""
    return SourceRef(
        202,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _single_schema(
        family: int,
        relation: ObjectIdentity,
        role: ObjectIdentity,
        allowed: frozenset[int],
        ) -> RelationSchema:
    """构造一个单值 Role 的 typed relation schema。"""
    return RelationSchema(
        structure_concept_identity((family, 1)),
        relation,
        (RelationSlotSchema(role, allowed, 1, 1),),
    )


def _binary_schema(
        family: int,
        relation: ObjectIdentity,
        left_role: ObjectIdentity,
        right_role: ObjectIdentity,
        left_allowed: frozenset[int],
        right_allowed: frozenset[int],
        ) -> RelationSchema:
    """构造两个方向 Role 都恰出现一次的 typed relation schema。"""
    return RelationSchema(
        structure_concept_identity((family, 1)),
        relation,
        (
            RelationSlotSchema(left_role, left_allowed, 1, 1),
            RelationSlotSchema(right_role, right_allowed, 1, 1),
        ),
    )


def _set_protocol() -> SetRelationProtocol:
    """构造互异 relation、schema、Role 和 MinimalInstruction 的 R-02 协议。"""
    relations = tuple(
        relation_concept_identity((9100, ordinal))
        for ordinal in range(1, 6)
    )
    roles = tuple(role_identity((9110, ordinal)) for ordinal in range(1, 10))
    subset = SetBinaryRelationProtocol(
        _binary_schema(
            9120,
            relations[0],
            roles[0],
            roles[1],
            frozenset({OBJECT_SET_EXPR}),
            frozenset({OBJECT_SET_EXPR}),
        ),
        roles[0],
        roles[1],
    )
    proper = SetBinaryRelationProtocol(
        _binary_schema(
            9130,
            relations[1],
            roles[2],
            roles[3],
            frozenset({OBJECT_SET_EXPR}),
            frozenset({OBJECT_SET_EXPR}),
        ),
        roles[2],
        roles[3],
    )
    member = SetBinaryRelationProtocol(
        _binary_schema(
            9140,
            relations[2],
            roles[4],
            roles[5],
            frozenset({OBJECT_ENTITY, OBJECT_SET_EXPR}),
            frozenset({OBJECT_SET_EXPR}),
        ),
        roles[4],
        roles[5],
    )
    equal = SetBinaryRelationProtocol(
        _binary_schema(
            9150,
            relations[3],
            roles[6],
            roles[7],
            frozenset({OBJECT_SET_EXPR}),
            frozenset({OBJECT_SET_EXPR}),
        ),
        roles[6],
        roles[7],
    )
    closed = SetUnaryRelationProtocol(
        _single_schema(
            9160,
            relations[4],
            roles[8],
            frozenset({OBJECT_SET_EXPR}),
        ),
        roles[8],
    )
    instructions = tuple(
        minimal_instruction_identity((9170, ordinal))
        for ordinal in range(1, 15)
    )
    return SetRelationProtocol(
        subset,
        proper,
        member,
        equal,
        closed,
        SetRelationRules(*instructions),
    )


@dataclass
class _Fixture:
    """保存一套可追加多来源集合事实的完整 R-00/R-02 测试设施。"""

    backend: DictBackend
    ctx: object
    protocol: SetRelationProtocol
    closure: RelationClosureRuntime
    runtime: SetRelationRuntime
    sets: tuple[ObjectIdentity, ...]
    members: tuple[ObjectIdentity, ...]
    next_fact_id: int = 1

    def close(self) -> None:
        """关闭测试后端。"""
        self.backend.close()

    def add(
            self,
            statement: SetRelationStatement,
            *,
            stance: str,
            archive_refuted: bool = False,
            replacement: ObjectIdentity | None = None,
            competition_key: tuple[int, ...] | None = None,
            ) -> ObjectIdentity:
        """写入一个 typed 关系事实，并可显式归档或替代当前候选。"""
        fact_id = self.next_fact_id
        self.next_fact_id += 1
        source = _source(1000 + fact_id)
        relation_protocol = self.protocol.relation_protocol(statement.relation)
        if isinstance(relation_protocol, SetUnaryRelationProtocol):
            bindings = (
                AtomicRoleBinding(
                    relation_protocol.value_role,
                    statement.left,
                ),
            )
        elif isinstance(relation_protocol, SetBinaryRelationProtocol):
            bindings = (
                AtomicRoleBinding(
                    relation_protocol.left_role,
                    statement.left,
                ),
                AtomicRoleBinding(
                    relation_protocol.right_role,
                    statement.right,
                ),
            )
        else:
            raise AssertionError("测试 statement relation 未注册")
        definition = AtomicPropositionDefinition(
            proposition_identity(source, (9200, fact_id)),
            statement.relation,
            occurrence_identity(source, start=0, end=1, ordinal=0),
            context_scope_identity(source, (9201, fact_id)),
            bindings,
        )
        spec = RelationClosureCandidateSpec(
            definition,
            relation_protocol.schema,
            competition_key or (9202, fact_id),
            (_source(2000 + fact_id * 2), _source(2001 + fact_id * 2)),
        )
        self.closure.semantic_graph.define_atomic(
            definition,
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        self.closure.form(spec, timestamp_base=fact_id * 10)
        observation = _source(3000 + fact_id)
        anchor = occurrence_identity(
            observation,
            start=0,
            end=1,
            ordinal=0,
        )
        supported = (definition.proposition,) if stance == "support" else ()
        refuted = (definition.proposition,) if stance == "refute" else ()
        recognition = RelationClosureRecognitionInput(
            definition.proposition,
            observation,
            document_scope(observation),
            ProtocolKey((9203, fact_id)),
            (9204, fact_id),
            anchor,
            (anchor,),
            RevealedObjectObservation(
                observation,
                document_scope(observation),
                (9204, fact_id),
                _source(4000 + fact_id),
                supported_targets=supported,
                refuted_targets=refuted,
                trace=(9205, fact_id),
            ),
            archive_refuted=archive_refuted,
            replacement=replacement,
        )
        self.closure.recognize(recognition)
        return definition.proposition


def _fixture(*, budget: SetRelationBudget | None = None) -> _Fixture:
    """建立共享 SemanticGraph、H-05 owner、五类 schema 和成员类型 resolver。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        _projection_protocol(),
    )
    candidate_runtime = _candidate_runtime(candidate_graph)
    protocol = _set_protocol()
    schemas = (
        protocol.subset_eq.schema,
        protocol.proper_subset.schema,
        protocol.member.schema,
        protocol.equal.schema,
        protocol.closed_domain.schema,
    )
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
    set_source = _source(10)
    sets = tuple(
        set_expr_identity(set_source, (9300, ordinal))
        for ordinal in range(1, 8)
    )
    member_source = _source(11)
    members = tuple(
        entity_identity(member_source, (9310, ordinal))
        for ordinal in range(1, 4)
    )
    member_type = concept_identity((9320, 1))
    resolver = MappingMemberTypeResolver(tuple(
        (member, member_type) for member in members
    ))
    runtime = SetRelationRuntime(
        closure,
        protocol,
        budget or SetRelationBudget(100, 200, 2000, 20, 20),
        resolver,
    )
    return _Fixture(backend, ctx, protocol, closure, runtime, sets, members)


def test_subset_proper_and_member_closures_preserve_direction_and_strictness():
    """非严格、严格和 MEMBER 闭包必须完整，但 MEMBER 不得反生 SUBSET。"""
    fixture = _fixture()
    try:
        a, b, c, d, e, *_rest = fixture.sets
        x = fixture.members[0]
        subset = fixture.protocol.subset_eq.relation
        proper = fixture.protocol.proper_subset.relation
        member = fixture.protocol.member.relation
        first = fixture.add(SetRelationStatement(subset, a, b), stance="support")
        second = fixture.add(SetRelationStatement(proper, b, c), stance="support")
        third = fixture.add(SetRelationStatement(member, x, a), stance="support")
        fixture.add(SetRelationStatement(member, d, e), stance="support")

        subset_result = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(subset, a, c),
            use_key=(9400, 1),
        ))
        proper_result = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(proper, a, c),
            use_key=(9400, 2),
        ))
        member_result = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(member, x, c),
            use_key=(9400, 3),
        ))
        reverse_result = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(subset, d, e),
        ))

        assert subset_result.evaluation.state == LogicEvidenceState(True, False)
        assert proper_result.evaluation.state == LogicEvidenceState(True, False)
        assert member_result.evaluation.state == LogicEvidenceState(True, False)
        assert reverse_result.evaluation.state == LogicEvidenceState(False, False)
        assert {item.proposition for item in subset_result.uses} == {first, second}
        assert {item.proposition for item in proper_result.uses} == {first, second}
        assert {item.proposition for item in member_result.uses} == {
            first,
            second,
            third,
        }
    finally:
        fixture.close()


def test_proper_subset_uses_explicit_inequality_and_keeps_conflict():
    """SUBSET 加 EQUAL refute 可支持严格子集，EQUAL support 同时形成反驳。"""
    fixture = _fixture()
    try:
        a, b, *_rest = fixture.sets
        subset = fixture.protocol.subset_eq.relation
        proper = fixture.protocol.proper_subset.relation
        equal = fixture.protocol.equal.relation
        fixture.add(SetRelationStatement(subset, a, b), stance="support")
        fixture.add(SetRelationStatement(equal, a, b), stance="refute")
        supported = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(proper, a, b),
        ))
        assert supported.evaluation.state == LogicEvidenceState(True, False)

        equal_support = fixture.add(
            SetRelationStatement(equal, a, b),
            stance="support",
        )
        conflicted = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(proper, a, b),
            use_key=(9410, 1),
        ))
        assert conflicted.evaluation.state == LogicEvidenceState(True, True)
        assert equal_support in {item.proposition for item in conflicted.uses}
    finally:
        fixture.close()


def test_proper_subset_self_fact_is_conflicted_by_irreflexive_axiom():
    """直接支持 A 严格包含自身也不能覆盖反自反形式反驳。"""
    fixture = _fixture()
    try:
        a = fixture.sets[0]
        proper = fixture.protocol.proper_subset.relation
        fixture.add(SetRelationStatement(proper, a, a), stance="support")
        result = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(proper, a, a),
        ))
        assert result.evaluation.state == LogicEvidenceState(True, True)
        assert fixture.protocol.rules.proper_irreflexive in {
            proof.rule for proof in result.evaluation.proofs
        }
    finally:
        fixture.close()


def test_missing_edges_and_negative_member_do_not_propagate():
    """普通缺边保持 unknown，not MEMBER 不沿子集方向传播。"""
    fixture = _fixture()
    try:
        a, b, c, *_rest = fixture.sets
        x = fixture.members[0]
        subset = fixture.protocol.subset_eq.relation
        member = fixture.protocol.member.relation
        fixture.add(SetRelationStatement(subset, a, b), stance="support")
        fixture.add(SetRelationStatement(member, x, a), stance="refute")
        missing = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(subset, b, c),
        ))
        lifted_negative = fixture.runtime.evaluate(SetRelationQuery(
            SetRelationStatement(member, x, b),
        ))
        assert missing.evaluation.state == LogicEvidenceState(False, False)
        assert lifted_negative.evaluation.state == LogicEvidenceState(False, False)
    finally:
        fixture.close()


def test_archived_refute_remains_explicit_but_superseded_fact_is_excluded():
    """归档反证仍可反驳查询，被替代候选则不得进入当前知识。"""
    fixture = _fixture()
    try:
        a, b, c, *_rest = fixture.sets
        subset = fixture.protocol.subset_eq.relation
        archived = SetRelationStatement(subset, a, b)
        fixture.add(
            archived,
            stance="refute",
            archive_refuted=True,
        )
        archived_result = fixture.runtime.evaluate(SetRelationQuery(archived))
        assert archived_result.evaluation.state == LogicEvidenceState(False, True)
        assert archived_result.uses == ()

        competition = (9430, 1)
        replacement = fixture.add(
            SetRelationStatement(subset, b, c),
            stance="unknown",
            competition_key=competition,
        )
        superseded = SetRelationStatement(subset, a, c)
        fixture.add(
            superseded,
            stance="refute",
            replacement=replacement,
            competition_key=competition,
        )
        superseded_result = fixture.runtime.evaluate(
            SetRelationQuery(superseded)
        )
        assert superseded_result.evaluation.state == LogicEvidenceState(
            False,
            False,
        )
    finally:
        fixture.close()


def test_closed_open_and_empty_domains_follow_four_state_quantifier_rules():
    """闭域、开放域和空闭域的 EXISTS/FORALL 必须遵守有限域四态语义。"""
    fixture = _fixture()
    try:
        closed_set, open_set, empty_set, *_rest = fixture.sets
        x, y, _z = fixture.members
        member = fixture.protocol.member.relation
        closed = fixture.protocol.closed_domain.relation
        fixture.add(SetRelationStatement(closed, closed_set), stance="support")
        fixture.add(SetRelationStatement(member, x, closed_set), stance="support")
        fixture.add(SetRelationStatement(member, y, closed_set), stance="support")
        fixture.add(SetRelationStatement(member, x, open_set), stance="support")
        fixture.add(SetRelationStatement(closed, empty_set), stance="support")

        closed_domain = fixture.runtime.finite_domain(
            SetRelationDomainQuery(closed_set),
        ).domain
        open_domain = fixture.runtime.finite_domain(
            SetRelationDomainQuery(open_set),
        ).domain
        empty_domain = fixture.runtime.finite_domain(
            SetRelationDomainQuery(empty_set),
        ).domain
        assert closed_domain.domain.closed is True
        assert open_domain.domain.closed is False
        assert empty_domain.domain.closed is True
        assert empty_domain.domain.values == ()

        closed_branches = tuple(
            SetQuantifierBranch(
                value,
                LogicEvidenceState(index == 0, index == 1),
            )
            for index, value in enumerate(closed_domain.domain.values)
        )
        exists = fixture.runtime.quantify_exists(closed_domain, closed_branches)
        forall = fixture.runtime.quantify_forall(closed_domain, closed_branches)
        assert exists.state == LogicEvidenceState(True, False)
        assert forall.state == LogicEvidenceState(False, True)

        open_branches = tuple(
            SetQuantifierBranch(value, LogicEvidenceState(True, False))
            for value in open_domain.domain.values
        )
        assert fixture.runtime.quantify_exists(
            open_domain,
            open_branches,
        ).state == LogicEvidenceState(True, False)
        assert fixture.runtime.quantify_forall(
            open_domain,
            open_branches,
        ).state == LogicEvidenceState(False, False)
        assert fixture.runtime.quantify_exists(
            empty_domain,
            (),
        ).state == LogicEvidenceState(False, True)
        assert fixture.runtime.quantify_forall(
            empty_domain,
            (),
        ).state == LogicEvidenceState(True, False)
    finally:
        fixture.close()


def test_conflicted_member_keeps_values_but_prevents_closed_domain_claim():
    """MEMBER 同时支持和反驳时可保留候选值，但有限域不得宣称闭合。"""
    fixture = _fixture()
    try:
        domain = fixture.sets[0]
        member_value = fixture.members[0]
        member = fixture.protocol.member.relation
        closed = fixture.protocol.closed_domain.relation
        statement = SetRelationStatement(member, member_value, domain)
        fixture.add(SetRelationStatement(closed, domain), stance="support")
        fixture.add(statement, stance="support")
        fixture.add(statement, stance="refute")

        result = fixture.runtime.finite_domain(SetRelationDomainQuery(domain))
        assert result.domain.domain.closed is False
        assert tuple(
            item.value for item in result.domain.domain.values
        ) == (member_value,)
        assert result.domain.members[0][1].state == LogicEvidenceState(True, True)
    finally:
        fixture.close()


def test_budget_exhaustion_fails_closed_before_returning_partial_query():
    """直接事实超过预算时必须整体失败，不能返回已扫描前缀的闭包。"""
    fixture = _fixture(budget=SetRelationBudget(1, 20, 100, 10, 10))
    try:
        a, b, c, *_rest = fixture.sets
        subset = fixture.protocol.subset_eq.relation
        fixture.add(SetRelationStatement(subset, a, b), stance="support")
        fixture.add(SetRelationStatement(subset, b, c), stance="support")
        with pytest.raises(SetRelationBudgetExceeded, match="直接事实"):
            fixture.runtime.evaluate(SetRelationQuery(
                SetRelationStatement(subset, a, c),
            ))
    finally:
        fixture.close()


class _ImproperLegacyMapper:
    """测试用错误 mapper：故意把旧记录升级为 PROPER_SUBSET。"""

    def __init__(self, mapped: MappedLegacySetRelation) -> None:
        """保存调用方构造的错误映射。"""
        self.mapped = mapped

    def map(self, record):
        """忽略旧键并返回错误的严格子集映射。"""
        return self.mapped

    def clone_for_evaluation(self):
        """返回同一不可变错误 mapper。"""
        return _ImproperLegacyMapper(self.mapped)

    def state_key(self):
        """返回测试 mapper 的固定协议键。"""
        return (9500, 1)


def test_legacy_mapper_cannot_upgrade_old_record_to_proper_subset():
    """旧关系记录即使经 mapper，也只能进入 typed SUBSET_EQ。"""
    fixture = _fixture()
    try:
        a, b, *_rest = fixture.sets
        proper = fixture.protocol.proper_subset
        source = _source(5000)
        definition = AtomicPropositionDefinition(
            proposition_identity(source, (9510, 1)),
            proper.relation,
            occurrence_identity(source, start=0, end=1, ordinal=0),
            context_scope_identity(source, (9510, 2)),
            (
                AtomicRoleBinding(proper.left_role, a),
                AtomicRoleBinding(proper.right_role, b),
            ),
        )
        spec = RelationClosureCandidateSpec(
            definition,
            proper.schema,
            (9510, 3),
            (_source(5001), _source(5002)),
        )
        formation = SetRelationFormationRequest(
            spec,
            document_scope(source),
            SOURCE_BARE_TEXT,
        )
        observation = _source(5003)
        anchor = occurrence_identity(
            observation,
            start=0,
            end=1,
            ordinal=0,
        )
        recognition = RelationClosureRecognitionInput(
            definition.proposition,
            observation,
            document_scope(observation),
            ProtocolKey((9510, 4)),
            (9510, 5),
            anchor,
            (anchor,),
            RevealedObjectObservation(
                observation,
                document_scope(observation),
                (9510, 5),
                _source(5004),
                supported_targets=(definition.proposition,),
                trace=(9510, 6),
            ),
        )
        mapped = MappedLegacySetRelation(formation, recognition)
        record = LegacySetRelationRecord(
            (10,),
            (1, 2),
            (3, 4),
            source,
            document_scope(source),
            (9510, 7),
        )
        with pytest.raises(SetRelationRuntimeError, match="SUBSET_EQ"):
            fixture.runtime.map_legacy(
                record,
                _ImproperLegacyMapper(mapped),
            )
    finally:
        fixture.close()


@dataclass(frozen=True)
class _RuntimeBuilder:
    """在任意 TrainContext 图上重建同一 R-02/R-00 协议。"""

    protocol: SetRelationProtocol
    budget: SetRelationBudget
    resolver: MappingMemberTypeResolver
    bound: SetRelationRuntime | None = None

    def build(self, ctx) -> SetRelationRuntime:
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
        schemas = (
            self.protocol.subset_eq.schema,
            self.protocol.proper_subset.schema,
            self.protocol.member.schema,
            self.protocol.equal.schema,
            self.protocol.closed_domain.schema,
        )
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
        return SetRelationRuntime(
            closure,
            self.protocol,
            self.budget,
            self.resolver.clone_for_evaluation(),
        )

    def clone_for_evaluation(self):
        """复制不可变协议并清除宿主 owner 引用。"""
        return _RuntimeBuilder(
            self.protocol,
            self.budget,
            self.resolver.clone_for_evaluation(),
        )

    def state_key(self):
        """返回协议、预算和类型 resolver 的完整纯整数键。"""
        return (
            9600,
            *self.protocol.stable_key(),
            *self.budget.stable_key(),
            *self.resolver.state_key(),
        )


@dataclass(frozen=True)
class _EmptyCourse:
    """任何 scope 都返回空 typed 请求的安装和隔离测试课程。"""

    version: int

    def request(self, scope, *, read_only):
        """返回不学习、不查询的空 R-02 round。"""
        return SetRelationRoundRequest(scope)

    def legacy_mapper(self):
        """声明当前课程不迁移旧边。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变空课程。"""
        return _EmptyCourse(self.version)

    def state_key(self):
        """返回课程版本键。"""
        return 9601, self.version


@dataclass(frozen=True)
class _StaticCourse:
    """返回测试注入请求的不可变 R-02 课程。"""

    round_request: SetRelationRoundRequest
    version: int

    def request(self, scope, *, read_only):
        """仅在目标 scope 和写入轮返回预设请求。"""
        if scope != self.round_request.scope or read_only:
            return SetRelationRoundRequest(scope)
        return self.round_request

    def legacy_mapper(self):
        """声明测试课程不迁移旧边。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变请求和版本。"""
        return _StaticCourse(self.round_request, self.version)

    def state_key(self):
        """返回测试课程的固定版本键。"""
        return 9603, self.version


@dataclass(frozen=True)
class _TrainingCourse:
    """在指定来源 scope 提交一个 SUBSET_EQ formation 和 recognition。"""

    formation: SetRelationFormationRequest
    recognition: RelationClosureRecognitionInput

    def request(self, scope, *, read_only):
        """训练轮提交 typed 写入，held-out 轮保持只读空请求。"""
        if scope != self.formation.scope or read_only:
            return SetRelationRoundRequest(scope)
        return SetRelationRoundRequest(
            scope,
            formations=(self.formation,),
            recognitions=(self.recognition,),
        )

    def legacy_mapper(self):
        """声明正式 fixture 不依赖旧边迁移。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变 typed 课程请求。"""
        return _TrainingCourse(self.formation, self.recognition)

    def state_key(self):
        """返回课程版本和 Proposition 完整身份键。"""
        proposition = self.formation.spec.proposition.proposition.stable_key()
        return 9602, len(proposition), *proposition


def _formal_course(
        protocol: SetRelationProtocol,
        source: SourceRef,
        ) -> _TrainingCourse:
    """构造与一个 formal 语言 item 来源绑定的 typed SUBSET_EQ 课程。"""
    left = set_expr_identity(_source(6100), (9610, 1))
    right = set_expr_identity(_source(6100), (9610, 2))
    relation = protocol.subset_eq
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (9610, 3)),
        relation.relation,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (9610, 4)),
        (
            AtomicRoleBinding(relation.left_role, left),
            AtomicRoleBinding(relation.right_role, right),
        ),
    )
    spec = RelationClosureCandidateSpec(
        definition,
        relation.schema,
        (9610, 5),
        (_source(6101), _source(6102)),
    )
    formation = SetRelationFormationRequest(
        spec,
        document_scope(source),
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
    )
    anchor = occurrence_identity(source, start=0, end=1, ordinal=0)
    recognition = RelationClosureRecognitionInput(
        definition.proposition,
        source,
        document_scope(source),
        ProtocolKey((9610, 6)),
        (9610, 7),
        anchor,
        (anchor,),
        RevealedObjectObservation(
            source,
            document_scope(source),
            (9610, 7),
            _source(6103),
            supported_targets=(definition.proposition,),
            trace=(9610, 8),
        ),
    )
    return _TrainingCourse(formation, recognition)


def test_v06_clone_keeps_set_relation_owner_and_host_use_state_isolated():
    """评测 clone 可读取已学闭包并写本地 Use，宿主 owner 和报告保持不变。"""
    fixture = _fixture()
    try:
        a, b, *_rest = fixture.sets
        subset = fixture.protocol.subset_eq.relation
        fixture.add(SetRelationStatement(subset, a, b), stance="support")
        builder = _RuntimeBuilder(
            fixture.protocol,
            fixture.runtime.budget,
            fixture.runtime.member_type_resolver,
            fixture.runtime,
        )
        installed = install_set_relation_runtime(
            fixture.ctx,
            builder,
            _EmptyCourse(1),
        )
        baseline = installed.state_key()
        report_count = len(fixture.ctx.set_relation_reports)

        with isolated_evaluation(fixture.ctx, label="r02-held-out") as eval_ctx:
            result = eval_ctx.set_relation_runtime.owner.evaluate(
                SetRelationQuery(
                    SetRelationStatement(subset, a, b),
                    use_key=(9620, 1),
                )
            )
            assert result.evaluation.state == LogicEvidenceState(True, False)
            assert len(result.uses) == 1

        assert installed.state_key() == baseline
        assert len(fixture.ctx.set_relation_reports) == report_count
    finally:
        fixture.close()


def test_high_formation_timestamp_schedules_recognition_after_forming_end():
    """课程显式高 forming 序时，recognition 三段序必须整体后移。"""
    fixture = _fixture()
    try:
        source = _source(6150)
        course = _formal_course(fixture.protocol, source)
        formation = replace(course.formation, timestamp_base=100_000)
        installed = install_set_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime.member_type_resolver,
                fixture.runtime,
            ),
            _TrainingCourse(formation, course.recognition),
        )

        report = installed.process(formation.scope, read_only=False)
        forming_end = (
            formation.timestamp_base
            + len(formation.spec.forming_sources)
            - 1
        )
        outcome = report.recognitions[0].outcome
        assert outcome.evidence.timestamp_seq > forming_end
        assert outcome.decision.timestamp_seq > outcome.evidence.timestamp_seq
    finally:
        fixture.close()


def test_learning_and_query_mix_is_rejected_before_any_write():
    """同轮学习和查询在请求构造期失败，不得留下语义或候选写入。"""
    fixture = _fixture()
    try:
        course = _formal_course(fixture.protocol, _source(6160))
        baseline_backend = fixture.backend.snapshot()
        baseline_owner = fixture.closure.state_key()
        with pytest.raises(ValueError, match="必须分开"):
            SetRelationRoundRequest(
                course.formation.scope,
                formations=(course.formation,),
                recognitions=(course.recognition,),
                queries=(SetRelationQuery(SetRelationStatement(
                    fixture.protocol.subset_eq.relation,
                    fixture.sets[0],
                    fixture.sets[1],
                )),),
            )
        assert fixture.backend.snapshot() == baseline_backend
        assert fixture.closure.state_key() == baseline_owner
    finally:
        fixture.close()


def test_semantic_preflight_failure_leaves_no_relation_candidate_write():
    """已有竞争 Proposition 拒绝课程定义时，R-00 forming 必须保持零写。"""
    fixture = _fixture()
    try:
        course = _formal_course(fixture.protocol, _source(6170))
        definition = course.formation.spec.proposition
        first, second = definition.bindings
        conflicting = AtomicPropositionDefinition(
            definition.proposition,
            definition.predicate,
            definition.source_anchor,
            definition.context,
            (
                AtomicRoleBinding(first.role, second.filler, first.ordinal),
                AtomicRoleBinding(second.role, first.filler, second.ordinal),
            ),
        )
        fixture.closure.semantic_graph.define_atomic(
            conflicting,
            scope=course.formation.scope,
            **course.formation.metadata(),
        )
        installed = install_set_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime.member_type_resolver,
                fixture.runtime,
            ),
            course,
        )
        baseline_backend = fixture.backend.snapshot()
        baseline_owner = fixture.closure.state_key()

        with pytest.raises(SemanticTopologyError, match="不同原子命题"):
            installed.process(course.formation.scope, read_only=False)
        assert fixture.backend.snapshot() == baseline_backend
        assert fixture.closure.state_key() == baseline_owner
    finally:
        fixture.close()


def test_recognition_preflight_failure_leaves_semantic_and_forming_zero_write():
    """recognition 缺 forming 时，课程不得先提交已通过检查的语义定义。"""
    fixture = _fixture()
    try:
        course = _formal_course(fixture.protocol, _source(6180))
        missing = proposition_identity(_source(6180), (9640, 1))
        request = SetRelationRoundRequest(
            course.formation.scope,
            formations=(course.formation,),
            recognitions=(replace(course.recognition, proposition=missing),),
        )
        installed = install_set_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime.member_type_resolver,
                fixture.runtime,
            ),
            _StaticCourse(request, 1),
        )
        baseline_backend = fixture.backend.snapshot()
        baseline_owner = fixture.closure.state_key()

        with pytest.raises(RelationClosureIncompleteError, match="缺少 forming"):
            installed.process(request.scope, read_only=False)
        assert fixture.backend.snapshot() == baseline_backend
        assert fixture.closure.state_key() == baseline_owner
        assert fixture.closure.semantic_graph.ontology.resolve(
            course.formation.spec.proposition.proposition
        ) is None
    finally:
        fixture.close()


def test_formal_train_installs_set_relation_course_and_returns_reports(
        tmp_path, monkeypatch):
    """顶层 formal_train 成对安装 R-02 builder/course 并返回真实写入报告。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    protocol = _set_protocol()
    source = _source(6200)
    resolver = MappingMemberTypeResolver(())
    builder = _RuntimeBuilder(
        protocol,
        SetRelationBudget(20, 40, 400, 10, 10),
        resolver,
    )
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r02-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            persist_graph_dump=False,
            language_occurrence_protocol=OccurrenceProtocol((9630, 1)),
            language_set_relation_builder=builder,
            language_set_relation_course=_formal_course(protocol, source),
        ),
        [CollectedItem(
            tokens=["甲", "乙"],
            raw_text="甲乙",
            role_seq=[1, 1],
            source=source.source_kind,
            source_ref=source,
        )],
        backend=DictBackend(),
    )

    assert result.set_relation_reports
    report = result.set_relation_reports[-1]
    assert report.formations
    assert report.recognitions
    assert report.recognitions[0].active_fact is not None


def test_formal_train_rejects_partial_set_relation_configuration(tmp_path):
    """R-02 builder/course 缺任一项时必须在安装前失败。"""
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="r02-partial",
        active_training_stages=(),
        persist_graph_dump=False,
        language_set_relation_builder=object(),
    )
    with pytest.raises(ValueError, match="必须成对配置"):
        formal_train(config, [], backend=DictBackend())
