"""R-03 六维 PROPERTY、四态聚合、课程接线和隔离测试。"""
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
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.property_relation import (
    MappingPropertyIntensityResolver,
    PropertyClaim,
    PropertyPattern,
    PropertyQueryBudget,
    PropertyRelationBudgetExceeded,
    PropertyRelationError,
    PropertyRelationProtocol,
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
    semantic_source,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.crosscut.integer.valtypes import Rational
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
)
from pure_integer_ai.experiments.property_relation_runtime import (
    LegacyPropertyRecord,
    MappedLegacyProperty,
    PropertyFormationRequest,
    PropertyQuery,
    PropertyRelationRuntime,
    PropertyRelationRuntimeError,
    PropertyRoundRequest,
    install_property_relation_runtime,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureIncompleteError,
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
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
    """构造共享 owner/version 且来源号互异的 R-03 测试来源。"""
    return SourceRef(
        203,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _property_protocol() -> PropertyRelationProtocol:
    """构造 relation、六个 Role 和允许 filler 类型均显式注入的协议。"""
    relation = relation_concept_identity((10100, 1))
    roles = tuple(role_identity((10110, ordinal)) for ordinal in range(1, 7))
    allowed = (
        frozenset({OBJECT_ENTITY}),
        frozenset({OBJECT_CONCEPT}),
        frozenset({OBJECT_CONCEPT, OBJECT_ENTITY}),
        frozenset({OBJECT_CONCEPT}),
        frozenset({OBJECT_CONCEPT}),
        frozenset({OBJECT_CONCEPT}),
    )
    schema = RelationSchema(
        structure_concept_identity((10120, 1)),
        relation,
        tuple(
            RelationSlotSchema(role, kinds, 1, 1)
            for role, kinds in zip(roles, allowed, strict=True)
        ),
    )
    return PropertyRelationProtocol(schema, *roles)


def _domain_objects() -> tuple[
        tuple[ObjectIdentity, ...],
        tuple[ObjectIdentity, ...],
        tuple[ObjectIdentity, ...],
        tuple[ObjectIdentity, ...],
        tuple[ObjectIdentity, ...],
        tuple[ObjectIdentity, ...],
        MappingPropertyIntensityResolver,
        ]:
    """构造六维 filler，并只用显式映射赋予两个强度 Rational 解释。"""
    subjects = tuple(
        entity_identity(_source(10), (10130, ordinal))
        for ordinal in range(1, 3)
    )
    attributes = tuple(concept_identity((10140, ordinal)) for ordinal in range(1, 3))
    values = tuple(concept_identity((10150, ordinal)) for ordinal in range(1, 4))
    polarities = tuple(concept_identity((10160, ordinal)) for ordinal in range(1, 3))
    modalities = tuple(concept_identity((10170, ordinal)) for ordinal in range(1, 3))
    intensities = tuple(concept_identity((10180, ordinal)) for ordinal in range(1, 3))
    resolver = MappingPropertyIntensityResolver((
        (intensities[0], Rational(1, 1)),
        (intensities[1], Rational(3, 2)),
    ))
    return (
        subjects,
        attributes,
        values,
        polarities,
        modalities,
        intensities,
        resolver,
    )


@dataclass
class _Fixture:
    """保存可追加多来源 PROPERTY 的完整 R-00/R-03 测试设施。"""

    backend: DictBackend
    ctx: object
    protocol: PropertyRelationProtocol
    closure: RelationClosureRuntime
    runtime: PropertyRelationRuntime
    subjects: tuple[ObjectIdentity, ...]
    attributes: tuple[ObjectIdentity, ...]
    values: tuple[ObjectIdentity, ...]
    polarities: tuple[ObjectIdentity, ...]
    modalities: tuple[ObjectIdentity, ...]
    intensities: tuple[ObjectIdentity, ...]
    next_fact_id: int = 1

    def close(self) -> None:
        """关闭测试后端。"""
        self.backend.close()

    def claim(
            self,
            *,
            subject: int = 0,
            attribute: int = 0,
            value: int = 0,
            polarity: int = 0,
            modality: int = 0,
            intensity: int = 0,
            ) -> PropertyClaim:
        """按测试域下标构造一个完整六维 PROPERTY claim。"""
        return PropertyClaim(
            self.subjects[subject],
            self.attributes[attribute],
            self.values[value],
            self.polarities[polarity],
            self.modalities[modality],
            self.intensities[intensity],
        )

    def definition(
            self,
            claim: PropertyClaim,
            source: SourceRef,
            fact_id: int,
            ) -> AtomicPropositionDefinition:
        """把六维 claim 编成不依赖 binding 顺序的 S-00 原子命题。"""
        bindings = tuple(
            AtomicRoleBinding(role, filler)
            for role, filler in zip(
                self.protocol.roles(),
                claim.values(),
                strict=True,
            )
        )
        return AtomicPropositionDefinition(
            proposition_identity(source, (10200, fact_id)),
            self.protocol.relation,
            occurrence_identity(source, start=0, end=1, ordinal=0),
            context_scope_identity(source, (10201, fact_id)),
            bindings,
        )

    def add(
            self,
            claim: PropertyClaim,
            *,
            stance: str,
            archive_refuted: bool = False,
            replacement: ObjectIdentity | None = None,
            competition_key: tuple[int, ...] | None = None,
            ) -> ObjectIdentity:
        """写入 typed PROPERTY，并可显式归档或替代当前候选。"""
        fact_id = self.next_fact_id
        self.next_fact_id += 1
        source = _source(1000 + fact_id)
        definition = self.definition(claim, source, fact_id)
        spec = RelationClosureCandidateSpec(
            definition,
            self.protocol.schema,
            competition_key or (10202, fact_id),
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
            ProtocolKey((10203, fact_id)),
            (10204, fact_id),
            anchor,
            (anchor,),
            RevealedObjectObservation(
                observation,
                document_scope(observation),
                (10204, fact_id),
                _source(4000 + fact_id),
                supported_targets=supported,
                refuted_targets=refuted,
                trace=(10205, fact_id),
            ),
            archive_refuted=archive_refuted,
            replacement=replacement,
        )
        self.closure.recognize(recognition)
        return definition.proposition


def _fixture(
        *,
        budget: PropertyQueryBudget | None = None,
        resolver: MappingPropertyIntensityResolver | None = None,
        ) -> _Fixture:
    """建立共享 SemanticGraph、H-05 owner、PROPERTY schema 和强度 resolver。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        _projection_protocol(),
    )
    candidate_runtime = _candidate_runtime(candidate_graph)
    protocol = _property_protocol()
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        _relation_protocol(),
        (protocol.schema,),
        engine=candidate_runtime.engine,
    )
    closure = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        _relation_protocol(),
    )
    domain = _domain_objects()
    runtime = PropertyRelationRuntime(
        closure,
        protocol,
        budget or PropertyQueryBudget(100, 20),
        resolver or domain[-1],
    )
    return _Fixture(
        backend,
        ctx,
        protocol,
        closure,
        runtime,
        *domain[:-1],
    )


def test_multi_source_same_claim_aggregates_and_consumes_all_active_premises():
    """同一六维 claim 的多来源支持应聚合为一个选项并全部记 Use。"""
    fixture = _fixture()
    try:
        claim = fixture.claim()
        first = fixture.add(claim, stance="support")
        second = fixture.add(claim, stance="support")

        result = fixture.runtime.select(PropertyQuery(
            PropertyPattern(claim.subject, claim.attribute),
            use_key=(10300, 1),
        ))

        selected = result.selection.selected()
        assert selected is not None
        assert selected.claim == claim
        assert selected.intensity_value == Rational(1, 1)
        assert selected.evaluation.state == LogicEvidenceState(True, False)
        assert len(selected.evaluation.evidence) == 2
        assert {item.proposition for item in result.uses} == {first, second}
        assert all(
            item.scope.source == semantic_source(item.proposition)
            for item in selected.evaluation.evidence
        )
    finally:
        fixture.close()


def test_value_polarity_modality_and_intensity_never_merge():
    """四个可变维度必须形成独立 claim，宽查询保持歧义而精确查询可采用。"""
    fixture = _fixture()
    try:
        claims = (
            fixture.claim(),
            fixture.claim(value=1),
            fixture.claim(polarity=1),
            fixture.claim(modality=1),
            fixture.claim(intensity=1),
        )
        for claim in claims:
            fixture.add(claim, stance="support")

        broad = fixture.runtime.select(PropertyQuery(
            PropertyPattern(claims[0].subject, claims[0].attribute),
            use_key=(10310, 1),
        ))
        assert broad.selection.ambiguous() is True
        assert broad.selection.selected() is None
        assert broad.uses == ()
        assert len(broad.selection.options) == len(claims)

        intense = claims[-1]
        exact = fixture.runtime.select(PropertyQuery(
            PropertyPattern(
                intense.subject,
                intense.attribute,
                intense.value,
                intense.polarity,
                intense.modality,
                intense.intensity,
            ),
            use_key=(10310, 2),
        ))
        assert exact.selection.selected() is not None
        assert exact.selection.selected().intensity_value == Rational(3, 2)
        assert intense.subject == claims[0].subject
        assert intense.attribute == claims[0].attribute
        assert intense.value == claims[0].value
    finally:
        fixture.close()


def test_conflicted_unknown_and_refuted_claims_never_write_use():
    """冲突、未知和纯反驳 PROPERTY 均保留审计四态但不得正式采用。"""
    fixture = _fixture()
    try:
        conflicted = fixture.claim()
        unknown = fixture.claim(value=1)
        refuted = fixture.claim(value=2)
        fixture.add(conflicted, stance="support")
        fixture.add(conflicted, stance="refute")
        fixture.add(unknown, stance="unknown")
        fixture.add(refuted, stance="refute")

        expectations = (
            (conflicted, LogicEvidenceState(True, True)),
            (unknown, LogicEvidenceState(False, False)),
            (refuted, LogicEvidenceState(False, True)),
        )
        for ordinal, (claim, state) in enumerate(expectations, start=1):
            result = fixture.runtime.select(PropertyQuery(
                PropertyPattern(
                    claim.subject,
                    claim.attribute,
                    claim.value,
                    claim.polarity,
                    claim.modality,
                    claim.intensity,
                ),
                use_key=(10320, ordinal),
            ))
            assert result.selection.options[0].evaluation.state == state
            assert result.selection.selected() is None
            assert result.uses == ()
    finally:
        fixture.close()


def test_archived_refute_remains_but_superseded_property_is_excluded():
    """显式归档反证仍参与四态，被 replacement 替代的候选不进入当前知识。"""
    fixture = _fixture()
    try:
        archived = fixture.claim()
        fixture.add(archived, stance="refute", archive_refuted=True)
        archived_result = fixture.runtime.select(PropertyQuery(
            PropertyPattern(
                archived.subject,
                archived.attribute,
                archived.value,
            ),
        ))
        assert archived_result.selection.options[0].evaluation.state == (
            LogicEvidenceState(False, True))

        competition = (10330, 1)
        replacement = fixture.add(
            fixture.claim(value=1),
            stance="unknown",
            competition_key=competition,
        )
        superseded = fixture.claim(value=2)
        fixture.add(
            superseded,
            stance="refute",
            replacement=replacement,
            competition_key=competition,
        )
        result = fixture.runtime.select(PropertyQuery(PropertyPattern(
            superseded.subject,
            superseded.attribute,
            superseded.value,
            superseded.polarity,
            superseded.modality,
            superseded.intensity,
        )))
        assert result.selection.options == ()
    finally:
        fixture.close()


def test_bare_s00_unknown_and_legacy_or_cooccurrence_inputs_cannot_become_fact():
    """裸命题、unknown recognition 和未映射旧记录均不能绕过 R-00 升为事实。"""
    fixture = _fixture()
    try:
        bare = fixture.claim()
        bare_source = _source(5000)
        definition = fixture.definition(bare, bare_source, 5000)
        fixture.closure.semantic_graph.define_atomic(
            definition,
            scope=document_scope(bare_source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        ontology = fixture.closure.semantic_graph.ontology
        ontology.relate(
            ontology.materialize(concept_identity((10325, 1))),
            ontology.materialize(bare.subject),
            ontology.materialize(bare.value),
            scope=document_scope(bare_source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        unknown = fixture.claim(value=1)
        fixture.add(unknown, stance="unknown")
        LegacyPropertyRecord(
            (1,), (2,), (3,), (4,), (5,), (6,), (7,),
            bare_source,
            document_scope(bare_source),
            (8,),
        )

        bare_result = fixture.runtime.select(PropertyQuery(PropertyPattern(
            bare.subject,
            bare.attribute,
            bare.value,
            bare.polarity,
            bare.modality,
            bare.intensity,
        )))
        unknown_result = fixture.runtime.select(PropertyQuery(PropertyPattern(
            unknown.subject,
            unknown.attribute,
            unknown.value,
            unknown.polarity,
            unknown.modality,
            unknown.intensity,
        )))
        assert bare_result.selection.options == ()
        assert unknown_result.selection.selected() is None
        assert unknown_result.uses == ()
    finally:
        fixture.close()


def test_missing_intensity_mapping_and_query_budget_fail_closed():
    """未知强度和超预算快照必须整体失败，不能返回部分 PROPERTY 选择。"""
    domain = _domain_objects()
    resolver = MappingPropertyIntensityResolver((
        (domain[5][0], Rational(1, 1)),
    ))
    fixture = _fixture(resolver=resolver)
    try:
        claim = fixture.claim(intensity=1)
        fixture.add(claim, stance="support")
        with pytest.raises(PropertyRelationError, match="缺少 Rational"):
            fixture.runtime.select(PropertyQuery(PropertyPattern(
                claim.subject,
                claim.attribute,
            )))
    finally:
        fixture.close()

    budget_fixture = _fixture(budget=PropertyQueryBudget(1, 20))
    try:
        budget_fixture.add(budget_fixture.claim(), stance="support")
        budget_fixture.add(
            budget_fixture.claim(value=1),
            stance="support",
        )
        with pytest.raises(PropertyRelationBudgetExceeded, match="直接事实"):
            budget_fixture.runtime.select(PropertyQuery(PropertyPattern(
                budget_fixture.subjects[0],
                budget_fixture.attributes[0],
            )))
    finally:
        budget_fixture.close()


class _BadLegacyMapper:
    """测试用错误 mapper，故意返回另一个 relation/schema。"""

    def __init__(self, mapped: MappedLegacyProperty) -> None:
        """保存调用方构造的错误映射。"""
        self.mapped = mapped

    def map(self, record):
        """忽略旧键并返回错误 schema 的映射。"""
        return self.mapped

    def clone_for_evaluation(self):
        """返回同一不可变错误 mapper。"""
        return _BadLegacyMapper(self.mapped)

    def state_key(self):
        """返回测试 mapper 的固定协议键。"""
        return 10340, 1


def _recognition_for(
        definition: AtomicPropositionDefinition,
        source: SourceRef,
        key: int,
        ) -> RelationClosureRecognitionInput:
    """为指定 Proposition 构造一个来源化 support recognition。"""
    anchor = occurrence_identity(source, start=0, end=1, ordinal=0)
    return RelationClosureRecognitionInput(
        definition.proposition,
        source,
        document_scope(source),
        ProtocolKey((10350, key)),
        (10351, key),
        anchor,
        (anchor,),
        RevealedObjectObservation(
            source,
            document_scope(source),
            (10351, key),
            _source(6000 + key),
            supported_targets=(definition.proposition,),
            trace=(10352, key),
        ),
    )


def test_legacy_mapper_must_return_current_complete_property_schema():
    """旧 PROPERTY/ATTR 记录不能被 mapper 接入错误 relation 或陈旧 schema。"""
    fixture = _fixture()
    try:
        source = _source(6100)
        claim = fixture.claim()
        wrong_relation = relation_concept_identity((10360, 1))
        wrong_schema = RelationSchema(
            structure_concept_identity((10360, 2)),
            wrong_relation,
            fixture.protocol.schema.slots,
        )
        definition = AtomicPropositionDefinition(
            proposition_identity(source, (10360, 3)),
            wrong_relation,
            occurrence_identity(source, start=0, end=1, ordinal=0),
            context_scope_identity(source, (10360, 4)),
            tuple(
                AtomicRoleBinding(role, filler)
                for role, filler in zip(
                    fixture.protocol.roles(),
                    claim.values(),
                    strict=True,
                )
            ),
        )
        formation = PropertyFormationRequest(
            RelationClosureCandidateSpec(
                definition,
                wrong_schema,
                (10360, 5),
                (_source(6101),),
            ),
            document_scope(source),
            SOURCE_BARE_TEXT,
        )
        mapped = MappedLegacyProperty(
            formation,
            _recognition_for(definition, source, 1),
        )
        record = LegacyPropertyRecord(
            (1,), (2,), (3,), (4,), (5,), (6,), (7,),
            source,
            document_scope(source),
            (8,),
        )
        with pytest.raises(PropertyRelationRuntimeError, match="typed schema"):
            fixture.runtime.map_legacy(record, _BadLegacyMapper(mapped))

        replacement_source = _source(6110)
        replacement_definition = fixture.definition(
            claim,
            replacement_source,
            2,
        )
        replacement_formation = PropertyFormationRequest(
            RelationClosureCandidateSpec(
                replacement_definition,
                fixture.protocol.schema,
                (10360, 6),
                (_source(6111),),
            ),
            document_scope(replacement_source),
            SOURCE_BARE_TEXT,
        )
        replacement_mapped = MappedLegacyProperty(
            replacement_formation,
            _recognition_for(replacement_definition, replacement_source, 3),
        )
        with pytest.raises(PropertyRelationRuntimeError, match="不得替换"):
            fixture.runtime.map_legacy(
                record,
                _BadLegacyMapper(replacement_mapped),
            )
    finally:
        fixture.close()


@dataclass(frozen=True)
class _RuntimeBuilder:
    """在任意 TrainContext 图上重建同一 R-03/R-00 协议。"""

    protocol: PropertyRelationProtocol
    budget: PropertyQueryBudget
    resolver: MappingPropertyIntensityResolver
    bound: PropertyRelationRuntime | None = None

    def build(self, ctx) -> PropertyRelationRuntime:
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
            (self.protocol.schema,),
            engine=candidate_runtime.engine,
        )
        closure = RelationClosureRuntime(
            candidate_runtime,
            semantic_graph,
            consumer,
            _relation_protocol(),
        )
        return PropertyRelationRuntime(
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
        """返回协议、预算和强度 resolver 的完整纯整数键。"""
        return (
            10400,
            *self.protocol.stable_key(),
            *self.budget.stable_key(),
            *self.resolver.state_key(),
        )


@dataclass(frozen=True)
class _EmptyCourse:
    """任何 scope 都返回空 typed 请求的安装和隔离测试课程。"""

    version: int

    def request(self, scope, *, read_only):
        """返回不学习、不查询的空 R-03 round。"""
        return PropertyRoundRequest(scope)

    def legacy_mapper(self):
        """声明当前课程不迁移旧边。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变空课程。"""
        return _EmptyCourse(self.version)

    def state_key(self):
        """返回课程版本键。"""
        return 10401, self.version


@dataclass(frozen=True)
class _DriftingCourse:
    """故意在评测 clone 中改变版本的隔离负例课程。"""

    version: int

    def request(self, scope, *, read_only):
        """返回空请求，使测试只聚焦 clone 状态守恒。"""
        return PropertyRoundRequest(scope)

    def legacy_mapper(self):
        """声明负例课程不迁移旧边。"""
        return None

    def clone_for_evaluation(self):
        """故意改变版本，验证 runtime 会拒绝漂移。"""
        return _DriftingCourse(self.version + 1)

    def state_key(self):
        """返回当前负例课程版本。"""
        return 10404, self.version


@dataclass(frozen=True)
class _StaticCourse:
    """返回测试注入请求的不可变 R-03 课程。"""

    round_request: PropertyRoundRequest
    version: int

    def request(self, scope, *, read_only):
        """仅在目标 scope 和写入轮返回预设请求。"""
        if scope != self.round_request.scope or read_only:
            return PropertyRoundRequest(scope)
        return self.round_request

    def legacy_mapper(self):
        """声明测试课程不迁移旧边。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变请求和版本。"""
        return _StaticCourse(self.round_request, self.version)

    def state_key(self):
        """返回测试课程的固定版本键。"""
        return 10402, self.version


@dataclass(frozen=True)
class _TrainingCourse:
    """在指定来源 scope 提交一个 PROPERTY formation 和 recognition。"""

    formation: PropertyFormationRequest
    recognition: RelationClosureRecognitionInput

    def request(self, scope, *, read_only):
        """训练轮提交 typed 写入，held-out 轮保持只读空请求。"""
        if scope != self.formation.scope or read_only:
            return PropertyRoundRequest(scope)
        return PropertyRoundRequest(
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
        return 10403, len(proposition), *proposition


def _formal_course(
        protocol: PropertyRelationProtocol,
        claim: PropertyClaim,
        source: SourceRef,
        ) -> _TrainingCourse:
    """构造与一个 formal 语言 item 来源绑定的 typed PROPERTY 课程。"""
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (10410, 1)),
        protocol.relation,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (10410, 2)),
        tuple(
            AtomicRoleBinding(role, filler)
            for role, filler in zip(
                protocol.roles(),
                claim.values(),
                strict=True,
            )
        ),
    )
    formation = PropertyFormationRequest(
        RelationClosureCandidateSpec(
            definition,
            protocol.schema,
            (10410, 3),
            (_source(6201), _source(6202)),
        ),
        document_scope(source),
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
    )
    return _TrainingCourse(
        formation,
        _recognition_for(definition, source, 2),
    )


def test_high_timestamp_mixed_round_and_failed_preflight_preserve_boundaries():
    """高序 recognition 后移，混轮与缺 forming 预演失败均不得留下部分写入。"""
    fixture = _fixture()
    try:
        source = _source(6300)
        course = _formal_course(fixture.protocol, fixture.claim(), source)
        formation = replace(course.formation, timestamp_base=100_000)
        installed = install_property_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime.intensity_resolver,
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
        assert report.recognitions[0].outcome.evidence.timestamp_seq > forming_end

        with pytest.raises(ValueError, match="必须分开"):
            PropertyRoundRequest(
                formation.scope,
                formations=(formation,),
                queries=(PropertyQuery(PropertyPattern(
                    fixture.subjects[0],
                    fixture.attributes[0],
                )),),
            )
    finally:
        fixture.close()

    failed = _fixture()
    try:
        source = _source(6310)
        course = _formal_course(failed.protocol, failed.claim(), source)
        missing = proposition_identity(source, (10420, 1))
        request = PropertyRoundRequest(
            course.formation.scope,
            formations=(course.formation,),
            recognitions=(replace(course.recognition, proposition=missing),),
        )
        installed = install_property_relation_runtime(
            failed.ctx,
            _RuntimeBuilder(
                failed.protocol,
                failed.runtime.budget,
                failed.runtime.intensity_resolver,
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
            course.formation.spec.proposition.proposition
        ) is None
    finally:
        failed.close()


def test_v06_clone_reads_property_but_keeps_host_use_and_reports_isolated():
    """评测 clone 可采用已学 PROPERTY，宿主 owner、Use 和报告保持不变。"""
    fixture = _fixture()
    try:
        claim = fixture.claim()
        fixture.add(claim, stance="support")
        installed = install_property_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime.intensity_resolver,
                fixture.runtime,
            ),
            _EmptyCourse(1),
        )
        baseline = installed.state_key()
        report_count = len(fixture.ctx.property_relation_reports)

        with isolated_evaluation(fixture.ctx, label="r03-held-out") as eval_ctx:
            result = eval_ctx.property_relation_runtime.owner.select(
                PropertyQuery(
                    PropertyPattern(claim.subject, claim.attribute),
                    use_key=(10430, 1),
                )
            )
            assert result.selection.selected() is not None
            assert len(result.uses) == 1

        assert installed.state_key() == baseline
        assert len(fixture.ctx.property_relation_reports) == report_count
    finally:
        fixture.close()


def test_v06_rejects_property_course_clone_state_drift():
    """held-out clone 不得借 course 副本改变协议或课程配置。"""
    fixture = _fixture()
    try:
        install_property_relation_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.protocol,
                fixture.runtime.budget,
                fixture.runtime.intensity_resolver,
                fixture.runtime,
            ),
            _DriftingCourse(1),
        )
        with pytest.raises(ValueError, match="改变课程状态"):
            with isolated_evaluation(fixture.ctx, label="r03-drift"):
                pass
    finally:
        fixture.close()


def test_formal_train_installs_property_course_and_returns_reports(
        tmp_path, monkeypatch):
    """顶层 formal_train 成对安装 R-03 builder/course 并返回真实写入报告。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    protocol = _property_protocol()
    domain = _domain_objects()
    claim = PropertyClaim(
        domain[0][0],
        domain[1][0],
        domain[2][0],
        domain[3][0],
        domain[4][0],
        domain[5][0],
    )
    source = _source(6400)
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r03-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            persist_graph_dump=False,
            language_occurrence_protocol=OccurrenceProtocol((10440, 1)),
            language_property_relation_builder=_RuntimeBuilder(
                protocol,
                PropertyQueryBudget(20, 10),
                domain[-1],
            ),
            language_property_relation_course=_formal_course(
                protocol,
                claim,
                source,
            ),
        ),
        [CollectedItem(
            tokens=["甲", "属性"],
            raw_text="甲属性",
            role_seq=[1, 1],
            source=source.source_kind,
            source_ref=source,
        )],
        backend=DictBackend(),
    )

    assert result.property_relation_reports
    report = result.property_relation_reports[-1]
    assert report.formations
    assert report.recognitions
    assert report.recognitions[0].active_fact is not None


def test_partial_property_configuration_and_inventory_are_honest(tmp_path):
    """R-03 配置必须成对，机制台账保持 opt-in 且不得进入 readiness。"""
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="r03-partial",
        active_training_stages=(),
        persist_graph_dump=False,
        language_property_relation_builder=object(),
    )
    with pytest.raises(ValueError, match="必须成对配置"):
        formal_train(config, [], backend=DictBackend())

    record = inventory_by_id()["relation.property_typed"]
    candidates = {item.mechanism_id for item in readiness_candidates()}
    assert record.status == STATUS_OPT_IN
    assert record.readiness_eligible is False
    assert "FormalTrainConfig.language_property_relation_builder" in record.gates
    assert "V06" in record.recovery[-1]
    assert "K 盘" in record.limitation
    assert record.mechanism_id not in candidates
