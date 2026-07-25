"""H-05 cue 结构、typed Sense、独立 verifier 和领域消费者专项。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateHistoryUnavailableError,
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CANDIDATE_AS_OBJECT,
    CANDIDATE_AS_SUBJECT,
    CandidateBinding,
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_SENSE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_atom_identity,
    language_branch_identity,
    sense_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    role_identity,
)
from pure_integer_ai.cognition.understanding.language_candidate import (
    ActiveCueStructureConsumer,
    ActiveSenseConsumer,
    CandidateFieldProtocol,
    CueStructureCandidateProtocol,
    CueStructureCandidateSpec,
    SenseCandidateProtocol,
    SenseCandidateSpec,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.language_candidate_runtime import (
    CandidateDomainRuntimeProtocol,
    LanguageCandidateRuntimeProtocol,
    install_language_candidate_runtime,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.language_structure_runtime import (
    _discover_and_recognize_lang_structures,
)
from pure_integer_ai.experiments.language_structure_candidate_runtime import (
    MappedStructureRecognition,
)
from pure_integer_ai.experiments.language_structure_boundary_runtime import (
    StructureBoundaryEvidenceProposal,
    apply_structure_boundary_evidence,
)
from pure_integer_ai.experiments.language_sense_candidate_runtime import (
    MappedSenseFormation,
    MappedSenseRecognition,
    observe_sense_lookup,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.cognition.process.structure_discover import (
    load_discovered_operators,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.boundary_hypothesis import (
    BoundaryCandidate,
    BoundaryHypothesisProtocol,
)
from pure_integer_ai.cognition.understanding.boundary_span import (
    BoundarySpanProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanProtocol,
)
from pure_integer_ai.cognition.understanding.span_index import SpanProtocol
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_REALIZES
from pure_integer_ai.storage.sense_candidates import (
    SENSE_CANDIDATES_TABLE,
    read_sense_candidates,
    sense_surface_hash,
)
from pure_integer_ai.training.cursor import dump_run, load_run
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    LANG_ZH,
    MODALITY_LANGUAGE,
)
from pure_integer_ai.training.stages import STAGE1_SKELETON


def _source(source_id: int) -> SourceRef:
    """构造共享 owner/version 但来源身份独立的测试 SourceRef。"""
    return SourceRef(
        27,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _projection_protocol() -> CandidateProjectionProtocol:
    """构造互不冲突的候选 lifecycle 图协议。"""
    values = tuple(concept_identity((2101, item)) for item in range(13))
    return CandidateProjectionProtocol(
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        (2201, 1),
    )


def _verifier() -> IndependentObjectVerifier:
    """构造只接收显式 reveal、没有图引用的独立 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((2301, 1)),
        (2301, 2),
        (2301, 3),
        (2301, 4),
        (2301, 5),
    ))


def _engine(kind_key: int) -> EvidenceCandidateEngine:
    """为一个候选领域构造独立 aggregate owner。"""
    aggregate = _source(900 + kind_key)
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        (2401, kind_key),
        (2402, kind_key),
        aggregate,
        document_scope(aggregate),
        2,
    ))


def _runtime(
        graph: CandidateProjectionGraph, *, kind_key: int,
        ) -> CandidateLearningRuntime:
    """把候选 owner、独立 verifier 和正式图投影装成完整运行时。"""
    return CandidateLearningRuntime(
        _engine(kind_key),
        graph,
        _verifier(),
        CandidateProjectionMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
        ),
    )


def _domain_protocol(kind_key: int) -> CandidateDomainRuntimeProtocol:
    """从测试 owner 和 verifier 提取正式上下文所需领域协议。"""
    return CandidateDomainRuntimeProtocol(
        _engine(kind_key).protocol,
        _verifier().protocol,
    )


class _NoopStructureMapper:
    """为上下文装配测试提供无候选、无可变状态的课程 mapper。"""

    def form(self, input_value):
        """装配测试不生成 structure 候选。"""
        return ()

    def recognize(self, input_value):
        """装配测试不生成 structure recognition。"""
        return ()

    def clone_for_evaluation(self):
        """返回独立的无状态 mapper 实例。"""
        return _NoopStructureMapper()

    def state_key(self) -> tuple:
        """无状态 mapper 使用空稳定键。"""
        return ()


class _NoopSenseMapper:
    """为不涉及 Sense 生产的测试提供显式空课程边界。"""

    def form(self, input_value):
        """空课程不接收旧 sense 候选。"""
        return ()

    def recognize(self, input_value):
        """空课程不产生 occurrence prediction。"""
        return ()

    def clone_for_evaluation(self):
        """返回独立的无状态 mapper。"""
        return _NoopSenseMapper()

    def state_key(self) -> tuple:
        """无状态 mapper 使用空稳定键。"""
        return ()


class _MappedSenseMapper:
    """用显式课程把旧候选全量映射，并只支持指定的一个 context 词义。"""

    def __init__(self) -> None:
        self.branch = language_branch_identity((3350, 1))
        self.atom = language_atom_identity(self.branch, (3350, 2))
        self.context = context_scope_identity(_source(907), (3350, 3))
        self._candidate_concepts = {}
        self._primary = None

    def form(self, input_value):
        """为每个旧 ref 给出独立 Sense/Concept，并保留同一竞争边界。"""
        values = []
        for index, legacy in enumerate(
                input_value.legacy_candidates, start=1):
            sense = sense_identity(
                _source(800 + index), sense_key=(3350, 10 + index))
            concept = concept_identity((3350, 20 + index))
            spec = SenseCandidateSpec(
                sense,
                (3350, 30),
                self.atom,
                concept,
                self.context,
                (_source(11), _source(12)),
            )
            values.append(MappedSenseFormation(
                legacy.legacy_ref,
                self.branch,
                spec,
            ))
            self._candidate_concepts[sense] = concept
            if self._primary is None:
                self._primary = sense
        return tuple(values)

    def recognize(self, input_value):
        """用独立课程 authority 支持 primary，其他候选继续保持 forming unknown。"""
        if self._primary not in input_value.available_candidates:
            return ()
        concept = self._candidate_concepts[self._primary]
        event_key = (
            3351,
            *input_value.occurrence.components[-3:],
        )
        visible = (input_value.occurrence,)
        if input_value.representation is not None:
            visible = (*visible, input_value.representation)
        return (MappedSenseRecognition(
            self._primary,
            visible,
            concept,
            RevealedObjectObservation(
                input_value.observation,
                input_value.scope,
                event_key,
                _source(1950 + input_value.observation.source_id),
                (concept,),
                (),
                (3352, input_value.observation.source_id),
            ),
        ),)

    def clone_for_evaluation(self):
        """复制课程映射，不共享候选字典。"""
        cloned = _MappedSenseMapper()
        cloned._candidate_concepts = dict(self._candidate_concepts)
        cloned._primary = self._primary
        return cloned

    def state_key(self) -> tuple:
        """返回 primary 与全部 Sense->Concept 映射。"""
        return (
            () if self._primary is None else self._primary.stable_key(),
            tuple(sorted(
                (sense.stable_key(), concept.stable_key())
                for sense, concept in self._candidate_concepts.items()
            )),
        )


class _MappedStructureMapper:
    """用测试课程显式提供结构、cue、关系、context 和独立 reveal。"""

    def __init__(self, *, require_formation: bool = True) -> None:
        self.candidate = structure_concept_identity((3301, 1))
        self.cue = language_atom_identity(
            language_branch_identity((3301, 2)), (3301, 3))
        self.relation = concept_identity((3301, 4))
        self.family = structure_concept_identity((3301, 5))
        self.context = context_scope_identity(_source(905), (3301, 6))
        self._operators: set[str] = set()
        self._require_formation = require_formation

    def form(self, input_value):
        """完整保留 forming 来源并形成一个 typed cue 结构候选。"""
        self._operators.add(input_value.operator_name)
        return (CueStructureCandidateSpec(
            self.candidate,
            (3301, 7),
            self.cue,
            self.relation,
            self.family,
            self.context,
            (),
            input_value.forming_sources,
        ),)

    def recognize(self, input_value):
        """只按来源输入揭示课程支持，不读取 graph、REALIZES 或 tally。"""
        if (self._require_formation
                and input_value.operator_name not in self._operators):
            return ()
        event_key = (
            3302,
            input_value.input_root[0],
            input_value.input_root[1],
        )
        return (MappedStructureRecognition(
            self.candidate,
            (self.cue, self.context, *input_value.occurrences),
            self.relation,
            RevealedObjectObservation(
                input_value.observation,
                input_value.scope,
                event_key,
                _source(1900 + input_value.observation.source_id),
                (self.relation,),
                (),
                (3303, input_value.observation.source_id),
            ),
        ),)

    def clone_for_evaluation(self):
        """复制课程映射状态但不共享可变 operator 集。"""
        cloned = _MappedStructureMapper(
            require_formation=self._require_formation)
        cloned._operators = set(self._operators)
        return cloned

    def state_key(self) -> tuple:
        """返回当前已形成 operator 名集合供隔离核验。"""
        return self._require_formation, tuple(sorted(self._operators))


class _MappedStructureBoundaryMapper:
    """把已核验完整文档结构映射为注入式边界候选，不读取旧边界决定。"""

    def propose(self, input_value):
        """用课程指定的倒数第二个 token 右界提出支持 Evidence。"""
        if len(input_value.token_spans) < 2:
            return ()
        anchor = input_value.token_spans[-2][1]
        return (StructureBoundaryEvidenceProposal(
            BoundaryCandidate((anchor,)),
            EVIDENCE_SUPPORT,
            (3370, 1),
            (3370, 2),
        ),)

    def clone_for_evaluation(self):
        """该测试 mapper 无可变状态，返回同型新实例。"""
        return _MappedStructureBoundaryMapper()

    def state_key(self) -> tuple:
        """无状态 mapper 使用空稳定键。"""
        return ()


class _EmptyStructureBoundaryMapper:
    """记录 eligible 调用但不提出 Evidence，用于核验零 proposal 零状态变化。"""

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, input_value):
        """接受完整输入但显式返回空 proposal。"""
        self.calls += 1
        return ()

    def clone_for_evaluation(self):
        """复制调用计数，避免评测共享可变状态。"""
        cloned = _EmptyStructureBoundaryMapper()
        cloned.calls = self.calls
        return cloned

    def state_key(self) -> tuple:
        """返回调用次数供隔离核验。"""
        return (self.calls,)


class _RefutingStructureMapper(_MappedStructureMapper):
    """产生显式 refute 的结构课程 mapper，核验边界 adapter 不消费负结果。"""

    def recognize(self, input_value):
        """用独立 verifier 定向反驳已形成结构，并请求 H-04 归档。"""
        if (self._require_formation
                and input_value.operator_name not in self._operators):
            return ()
        event_key = (
            3380,
            input_value.input_root[0],
            input_value.input_root[1],
        )
        return (MappedStructureRecognition(
            self.candidate,
            (self.cue, self.context, *input_value.occurrences),
            self.relation,
            RevealedObjectObservation(
                input_value.observation,
                input_value.scope,
                event_key,
                _source(1950 + input_value.observation.source_id),
                (),
                (self.relation,),
                (3381, input_value.observation.source_id),
            ),
            archive_refuted=True,
        ),)


def _language_runtime_protocol(
        structure_mapper=None,
        sense_mapper=None,
        structure_boundary_mapper=None) -> LanguageCandidateRuntimeProtocol:
    """构造 structure/Sense owner、字段和共享图的完整安装协议。"""
    mapper = (
        _NoopStructureMapper()
        if structure_mapper is None else structure_mapper)
    mapped_sense = (
        _NoopSenseMapper() if sense_mapper is None else sense_mapper)
    return LanguageCandidateRuntimeProtocol(
        _projection_protocol(),
        CandidateProjectionMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
        ),
        _domain_protocol(5),
        _structure_protocol(),
        mapper,
        _domain_protocol(6),
        _sense_protocol(),
        mapped_sense,
        structure_boundary_mapper,
    )


def _span_protocol() -> SegmentationSpanProtocol:
    """为 formal Sense caller 注入 occurrence/span 图协议。"""
    return SegmentationSpanProtocol(
        SpanProtocol(
            (3360, 1),
            (3360, 2),
            (3360, 3),
            (3360, 4),
        ),
        (3360, 5),
        (3360, 6),
        (3360, 7),
        (3360, 8),
    )


def _boundary_protocol() -> BoundarySpanProtocol:
    """为 H-05 structure-boundary 正式 caller 注入完整句界协议。"""
    return BoundarySpanProtocol(
        BoundaryHypothesisProtocol((3365, 1)),
        _span_protocol().span_protocol,
        (3365, 2),
        (3365, 3),
        (3365, 4),
        (3365, 5),
        (3365, 6),
        (3365, 7),
        336508,
    )


def _seed_legacy_sense_rows(backend, surface: str) -> None:
    """写入两个旧候选，专门核验 typed 迁移、兼容旁路和恢复边界。"""
    surface_hash = sense_surface_hash(surface)
    for local_id in (501, 502):
        backend.insert(SENSE_CANDIDATES_TABLE, {
            "space_id": 1,
            "surface_hash": surface_hash,
            "sense_sid": 1,
            "sense_lid": local_id,
            "base_count": 1,
            "sc_sn": 0,
            "sc_tn": 0,
        })


def _activate_mapped_sense(ctx, formation, *, source_id: int) -> None:
    """用独立来源支持并采用一个已形成的 typed Sense。"""
    hypothesis = ctx.sense_candidate_runtime.hypothesis_for_candidate(
        formation.spec.sense)
    observation = _source(source_id)
    event_key = (3362, source_id)
    evidence_seq, decision_seq, projection_seq = (
        ctx.sense_candidate_runtime.next_timestamps(3))
    ctx.sense_candidate_runtime.recognize(
        hypothesis,
        observation=observation,
        scope=document_scope(observation),
        event_key=event_key,
        visible_inputs=(formation.spec.atom, formation.spec.context),
        predicted=formation.spec.concept,
        revealed=_reveal(
            observation,
            event_key=event_key,
            supported=(formation.spec.concept,),
        ),
        timestamp_seq=evidence_seq,
        resolve_timestamp_seq=decision_seq,
        projection_timestamp_seq=projection_seq,
    )


def _reveal(
        observation: SourceRef, *, event_key: tuple[int, ...],
        supported: tuple[ObjectIdentity, ...] = (),
        refuted: tuple[ObjectIdentity, ...] = (),
        ) -> RevealedObjectObservation:
    """构造与 prediction 来源和事件严格对齐的独立揭示。"""
    return RevealedObjectObservation(
        observation,
        document_scope(observation),
        event_key,
        _source(1800 + observation.source_id),
        supported,
        refuted,
        (2501, observation.source_id),
    )


def _sense_protocol() -> SenseCandidateProtocol:
    """注入 atom->Sense、Sense->Concept 和 context 三个 typed 字段。"""
    return SenseCandidateProtocol(
        (OBJECT_SENSE,),
        CandidateFieldProtocol(
            concept_identity((2601, 1)),
            CANDIDATE_AS_OBJECT,
            0,
            (OBJECT_LANGUAGE_ATOM,),
        ),
        CandidateFieldProtocol(
            concept_identity((2601, 2)),
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_CONCEPT,),
        ),
        CandidateFieldProtocol(
            concept_identity((2601, 3)),
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_CONTEXT_SCOPE,),
        ),
    )


def _structure_protocol() -> CueStructureCandidateProtocol:
    """注入结构候选四个必需字段及其开放 object kind 契约。"""
    return CueStructureCandidateProtocol(
        (OBJECT_STRUCTURE_CONCEPT,),
        CandidateFieldProtocol(
            concept_identity((2701, 1)),
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_LANGUAGE_ATOM,),
        ),
        CandidateFieldProtocol(
            concept_identity((2701, 2)),
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_CONCEPT,),
        ),
        CandidateFieldProtocol(
            concept_identity((2701, 3)),
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_STRUCTURE_CONCEPT,),
        ),
        CandidateFieldProtocol(
            concept_identity((2701, 4)),
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_CONTEXT_SCOPE,),
        ),
    )


def test_independent_verifier_requires_explicit_refute_and_keeps_gap_unknown():
    """另一已知目标和未覆盖目标都不是 false，只有显式反驳集合产生 refute。"""
    engine = _engine(1)
    target = concept_identity((2801, 1))
    other = concept_identity((2801, 2))
    definition = CueStructureCandidateSpec(
        structure_concept_identity((2802, 1)),
        (2802, 2),
        language_atom_identity(language_branch_identity((2802, 3)), (1,)),
        target,
        structure_concept_identity((2802, 4)),
        context_scope_identity(_source(901), (2802, 5)),
        (),
        (_source(1), _source(2)),
    ).definition(_structure_protocol())
    hypothesis = engine.register(definition)
    observation = _source(3)
    prediction = engine.predict(
        hypothesis,
        observation=observation,
        scope=document_scope(observation),
        event_key=(2803, 1),
        visible_inputs=(definition.candidate,),
        predicted=target,
    )
    verifier = _verifier()

    unknown = verifier.verify(
        prediction,
        _reveal(
            observation,
            event_key=(2803, 1),
            supported=(other,),
        ),
    )
    refuted = verifier.verify(
        prediction,
        _reveal(
            observation,
            event_key=(2803, 1),
            refuted=(target,),
        ),
    )
    assert unknown.stance == EVIDENCE_UNKNOWN
    assert refuted.stance == EVIDENCE_REFUTE


def test_sense_graph_is_atom_to_sense_to_concept_and_consumer_needs_active():
    """Sense 拓扑方向可核验，forming 图存在也不能绕过 active lifecycle。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        graph = CandidateProjectionGraph(
            context.graph_ontology, _projection_protocol())
        runtime = _runtime(graph, kind_key=2)
        protocol = _sense_protocol()
        branch = language_branch_identity((2901, 1))
        atom = language_atom_identity(branch, (2901, 2))
        concept = concept_identity((2901, 3))
        sense = sense_identity(_source(701), sense_key=(2901, 4))
        semantic_context = context_scope_identity(_source(902), (2901, 5))
        definition = SenseCandidateSpec(
            sense,
            (2901, 6),
            atom,
            concept,
            semantic_context,
            (_source(1), _source(2)),
        ).definition(protocol)
        hypothesis = runtime.register(definition, timestamp_base=1)
        consumer = ActiveSenseConsumer(graph, protocol)

        atom_ref = context.graph_ontology.resolve(atom)
        sense_ref = context.graph_ontology.resolve(sense)
        concept_ref = context.graph_ontology.resolve(concept)
        atom_predicate = context.graph_ontology.resolve(
            protocol.atom.predicate)
        concept_predicate = context.graph_ontology.resolve(
            protocol.concept.predicate)
        assert len(context.graph_ontology.statements(
            predicate=atom_predicate,
            subject=atom_ref,
            object_ref=sense_ref,
        )) == 1
        assert len(context.graph_ontology.statements(
            predicate=concept_predicate,
            subject=sense_ref,
            object_ref=concept_ref,
        )) == 1
        assert consumer.lookup(atom, context=semantic_context) == ()

        observation = _source(3)
        outcome = runtime.recognize(
            hypothesis,
            observation=observation,
            scope=document_scope(observation),
            event_key=(2902, 1),
            visible_inputs=(atom, semantic_context),
            predicted=concept,
            revealed=_reveal(
                observation,
                event_key=(2902, 1),
                supported=(concept,),
            ),
            timestamp_seq=20,
            resolve_timestamp_seq=21,
            projection_timestamp_seq=30,
        )
        adopted = consumer.require_unique(atom, context=semantic_context)
        assert outcome.verification.stance == EVIDENCE_SUPPORT
        assert adopted.sense == sense
        assert adopted.concept == concept
        assert adopted.projection == outcome.projection
    finally:
        backend.close()


def test_multiple_active_senses_remain_visible_until_targeted_archive():
    """同 atom/context 的多 Sense 保持并存，定向反例归档后才退出消费者。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        projection_protocol = _projection_protocol()
        graph = CandidateProjectionGraph(
            context.graph_ontology, projection_protocol)
        runtime = _runtime(graph, kind_key=3)
        protocol = _sense_protocol()
        atom = language_atom_identity(
            language_branch_identity((3001, 1)), (3001, 2))
        semantic_context = context_scope_identity(_source(903), (3001, 3))
        concepts = (
            concept_identity((3001, 4)),
            concept_identity((3001, 5)),
        )
        hypotheses = []
        for index, concept in enumerate(concepts, start=1):
            definition = SenseCandidateSpec(
                sense_identity(
                    _source(710 + index), sense_key=(3001, 10 + index)),
                (3001, 20),
                atom,
                concept,
                semantic_context,
                (_source(1), _source(2)),
            ).definition(protocol)
            hypotheses.append(runtime.register(
                definition, timestamp_base=index))

        for index, (hypothesis, concept) in enumerate(
                zip(hypotheses, concepts), start=3):
            observation = _source(index)
            runtime.recognize(
                hypothesis,
                observation=observation,
                scope=document_scope(observation),
                event_key=(3002, index),
                visible_inputs=(atom, semantic_context),
                predicted=concept,
                revealed=_reveal(
                    observation,
                    event_key=(3002, index),
                    supported=(concept,),
                ),
                timestamp_seq=index * 10,
                resolve_timestamp_seq=index * 10 + 1,
                projection_timestamp_seq=index * 10 + 2,
            )

        consumer = ActiveSenseConsumer(graph, protocol)
        active = consumer.lookup(atom, context=semantic_context)
        assert len(active) == 2
        with pytest.raises(LookupError, match="唯一"):
            consumer.require_unique(atom, context=semantic_context)
        assert all(
            item.projection.history[-1].definition.decision_key
            == runtime.engine.resolver.decision_history(
                item.projection.candidate.hypothesis)[-1].stable_key()
            for item in active
        )

        rejected = hypotheses[0]
        rejected_concept = concepts[0]
        observation = _source(5)
        outcome = runtime.recognize(
            rejected,
            observation=observation,
            scope=document_scope(observation),
            event_key=(3003, 1),
            visible_inputs=(atom, semantic_context),
            predicted=rejected_concept,
            revealed=_reveal(
                observation,
                event_key=(3003, 1),
                refuted=(rejected_concept,),
            ),
            timestamp_seq=60,
            resolve_timestamp_seq=61,
            projection_timestamp_seq=70,
            archive_refuted=True,
        )
        remaining = consumer.require_unique(atom, context=semantic_context)
        assert outcome.verification.stance == EVIDENCE_REFUTE
        assert outcome.projection.state == projection_protocol.inactive_state
        assert remaining.concept == concepts[1]
        assert runtime.report().active_projection_count == 1
    finally:
        backend.close()


def test_cue_structure_consumer_changes_only_after_typed_promotion_and_demotion():
    """cue 结构 consumer 的 ON/OFF 由 typed lifecycle 决定，旧形状边不能旁路。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        projection_protocol = _projection_protocol()
        graph = CandidateProjectionGraph(
            context.graph_ontology, projection_protocol)
        runtime = _runtime(graph, kind_key=4)
        protocol = _structure_protocol()
        cue = language_atom_identity(
            language_branch_identity((3101, 1)), (3101, 2))
        relation = concept_identity((3101, 3))
        family = structure_concept_identity((3101, 4))
        semantic_context = context_scope_identity(_source(904), (3101, 5))
        role = role_identity((3101, 6))
        definition = CueStructureCandidateSpec(
            structure_concept_identity((3101, 7)),
            (3101, 8),
            cue,
            relation,
            family,
            semantic_context,
            (CandidateBinding(
                concept_identity((3101, 9)),
                role,
                0,
                CANDIDATE_AS_SUBJECT,
            ),),
            (_source(1), _source(2)),
        ).definition(protocol)
        hypothesis = runtime.register(definition)
        consumer = ActiveCueStructureConsumer(graph, protocol)
        assert consumer.lookup(
            cue, target_relation=relation, context=semantic_context) == ()

        observation = _source(3)
        runtime.recognize(
            hypothesis,
            observation=observation,
            scope=document_scope(observation),
            event_key=(3102, 1),
            visible_inputs=(cue, semantic_context),
            predicted=relation,
            revealed=_reveal(
                observation,
                event_key=(3102, 1),
                supported=(relation,),
            ),
            timestamp_seq=20,
            resolve_timestamp_seq=21,
            projection_timestamp_seq=30,
        )
        active = consumer.require_unique(
            cue, target_relation=relation, context=semantic_context)
        assert active.structure == definition.candidate
        assert active.projection.state == projection_protocol.active_state

        negative_observation = _source(4)
        outcome = runtime.recognize(
            hypothesis,
            observation=negative_observation,
            scope=document_scope(negative_observation),
            event_key=(3102, 2),
            visible_inputs=(cue, semantic_context),
            predicted=relation,
            revealed=_reveal(
                negative_observation,
                event_key=(3102, 2),
                refuted=(relation,),
            ),
            timestamp_seq=40,
            resolve_timestamp_seq=41,
            projection_timestamp_seq=50,
            archive_refuted=True,
        )
        assert outcome.projection.state == projection_protocol.inactive_state
        assert consumer.lookup(
            cue, target_relation=relation, context=semantic_context) == ()
    finally:
        backend.close()


def test_context_install_requires_foundation_and_eval_clone_isolates_candidate_owner():
    """正式安装拒绝缺 occurrence/span，评测 clone 的 Evidence 和图写不回宿主。"""
    backend = DictBackend()
    eval_backend = None
    try:
        context = make_train_context(backend)
        protocol = _language_runtime_protocol()
        with pytest.raises(ValueError, match="occurrence.*Span"):
            install_language_candidate_runtime(context, protocol)
        context.occurrence_index = object()
        context.span_index = object()
        install_language_candidate_runtime(context, protocol)

        sense_protocol = protocol.sense_fields
        atom = language_atom_identity(
            language_branch_identity((3201, 1)), (3201, 2))
        concept = concept_identity((3201, 3))
        semantic_context = context_scope_identity(_source(906), (3201, 4))
        definition = SenseCandidateSpec(
            sense_identity(_source(720), sense_key=(3201, 5)),
            (3201, 6),
            atom,
            concept,
            semantic_context,
            (_source(1), _source(2)),
        ).definition(sense_protocol)
        hypothesis = context.sense_candidate_runtime.register(definition)
        host_state = context.sense_candidate_runtime.state_key()
        host_backend_state = backend.snapshot()

        context.occurrence_index = None
        context.span_index = None
        eval_backend = clone_backend(backend)
        cloned = clone_train_context(
            context,
            eval_backend,
            label="h05-candidate-isolation",
        )
        observation = _source(3)
        cloned.sense_candidate_runtime.recognize(
            hypothesis,
            observation=observation,
            scope=document_scope(observation),
            event_key=(3202, 1),
            visible_inputs=(atom, semantic_context),
            predicted=concept,
            revealed=_reveal(
                observation,
                event_key=(3202, 1),
                supported=(concept,),
            ),
            timestamp_seq=20,
            resolve_timestamp_seq=21,
            projection_timestamp_seq=30,
        )

        assert cloned.sense_candidate_consumer.require_unique(
            atom, context=semantic_context).concept == concept
        assert context.sense_candidate_consumer.lookup(
            atom, context=semantic_context) == ()
        assert context.sense_candidate_runtime.state_key() == host_state
        assert backend.snapshot() == host_backend_state
    finally:
        if eval_backend is not None:
            eval_backend.close()
        backend.close()


def test_production_structure_hook_uses_typed_chain_and_disables_legacy_self_proof(
        monkeypatch):
    """正式 discovery 形成 unknown、held-out support 后晋升，并且不调用 REALIZES/tally。"""
    from pure_integer_ai.cognition.process import structure_discover
    from pure_integer_ai.config import gates

    def unexpected_legacy_call(*args, **kwargs):
        """旧自证 writer 被调用时立即使测试失败。"""
        raise AssertionError("H-05 启用后不得调用 legacy REALIZES/tally")

    monkeypatch.setattr(gates, "REALIZES_MODE", True)
    monkeypatch.setattr(gates, "ORACLE_PROMOTE_MODE", True)
    monkeypatch.setattr(
        structure_discover,
        "label_realizes_is_a",
        unexpected_legacy_call,
    )
    monkeypatch.setattr(
        structure_discover,
        "label_realizes_causes",
        unexpected_legacy_call,
    )
    monkeypatch.setattr(
        structure_discover,
        "tally_cue_slot_matches",
        unexpected_legacy_call,
    )

    backend = DictBackend()
    try:
        context = make_train_context(backend)
        install_language_graph_protocols(
            context,
            occurrence_protocol=OccurrenceProtocol((3366, 1)),
            span_protocol=_span_protocol(),
            boundary_protocol=_boundary_protocol(),
        )
        mapper = _MappedStructureMapper()
        boundary_mapper = _MappedStructureBoundaryMapper()
        install_language_candidate_runtime(
            context,
            _language_runtime_protocol(
                mapper,
                structure_boundary_mapper=boundary_mapper,
            ),
        )
        corpus = [
            CollectedItem(
                tokens=list(tokens),
                raw_text="".join(tokens),
                source=SOURCE_BARE_TEXT,
                modality=MODALITY_LANGUAGE,
                lang=LANG_ZH,
                domain=DOMAIN_TEXT,
            )
            for tokens in (
                ("甲", "记号", "乙"),
                ("丙", "记号", "丁"),
                ("戊", "记号", "己"),
            )
        ]

        discovered, recognitions, summary = (
            _discover_and_recognize_lang_structures(context, corpus))
        active = context.structure_candidate_consumer.require_unique(
            mapper.cue,
            target_relation=mapper.relation,
            context=mapper.context,
        )
        report = context.structure_candidate_reports[-1]

        assert len(discovered) == 1
        assert len(recognitions) == 1
        assert summary.tally_stats.calls == 0
        assert report.candidates_registered == 1
        assert report.prediction_count == 1
        assert report.support_count == 1
        assert active.structure == mapper.candidate
        assert context.structure_candidate_runtime.report().active_projection_count == 1
        assert context.structure_boundary_report.eligible_traces == 1
        assert context.structure_boundary_report.evidence_added == 1
        assert context.structure_boundary_report.items_reparsed == 1
        assert corpus[0].boundary_profile is None
        assert corpus[1].boundary_profile is None
        assert corpus[2].boundary_decision.anchors == (3,)
        assert context.edge_store.query_from(
            discovered[0].skeleton_ref[0],
            discovered[0].skeleton_ref[1],
            edge_type=EDGE_REALIZES,
        ) == []
    finally:
        backend.close()


def test_structure_boundary_empty_proposal_keeps_absent_profile():
    """eligible structure 没有边界 proposal 时不得制造空 profile 或触发重解析。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        install_language_graph_protocols(
            context,
            occurrence_protocol=OccurrenceProtocol((3366, 1)),
            span_protocol=_span_protocol(),
            boundary_protocol=_boundary_protocol(),
        )
        mapper = _MappedStructureMapper()
        boundary_mapper = _EmptyStructureBoundaryMapper()
        install_language_candidate_runtime(
            context,
            _language_runtime_protocol(
                mapper,
                structure_boundary_mapper=boundary_mapper,
            ),
        )
        corpus = [
            CollectedItem(
                tokens=list(tokens),
                raw_text="".join(tokens),
                source=SOURCE_BARE_TEXT,
                modality=MODALITY_LANGUAGE,
                lang=LANG_ZH,
                domain=DOMAIN_TEXT,
            )
            for tokens in (
                ("甲", "记号", "乙"),
                ("丙", "记号", "丁"),
                ("戊", "记号", "己"),
            )
        ]

        _discover_and_recognize_lang_structures(context, corpus)

        assert boundary_mapper.calls == 1
        assert context.structure_boundary_report.eligible_traces == 1
        assert context.structure_boundary_report.proposals == 0
        assert context.structure_boundary_report.evidence_added == 0
        assert context.structure_boundary_report.items_reparsed == 0
        assert all(item.boundary_profile is None for item in corpus)
    finally:
        backend.close()


def test_refuted_structure_cannot_feed_boundary_evidence():
    """结构 prediction 被独立反驳并归档后不得调用边界 mapper。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        install_language_graph_protocols(
            context,
            occurrence_protocol=OccurrenceProtocol((3366, 1)),
            span_protocol=_span_protocol(),
            boundary_protocol=_boundary_protocol(),
        )
        mapper = _RefutingStructureMapper()
        boundary_mapper = _EmptyStructureBoundaryMapper()
        install_language_candidate_runtime(
            context,
            _language_runtime_protocol(
                mapper,
                structure_boundary_mapper=boundary_mapper,
            ),
        )
        corpus = [
            CollectedItem(
                tokens=list(tokens),
                raw_text="".join(tokens),
                source=SOURCE_BARE_TEXT,
                modality=MODALITY_LANGUAGE,
                lang=LANG_ZH,
                domain=DOMAIN_TEXT,
            )
            for tokens in (
                ("甲", "记号", "乙"),
                ("丙", "记号", "丁"),
                ("戊", "记号", "己"),
            )
        ]

        _discover_and_recognize_lang_structures(context, corpus)

        assert boundary_mapper.calls == 0
        assert context.structure_boundary_report.non_support_blocked == 1
        assert context.structure_boundary_report.eligible_traces == 0
        assert context.structure_boundary_report.evidence_added == 0
        assert all(item.boundary_profile is None for item in corpus)
    finally:
        backend.close()


def test_structure_boundary_blocks_inactive_and_partial_document_traces(
        monkeypatch):
    """support trace 未 adopted 或未覆盖完整文档时都不得进入边界 mapper。"""
    from pure_integer_ai.experiments import (
        language_structure_candidate_runtime as candidate_runtime,
    )

    captured = []
    real_recognize = candidate_runtime.recognize_structure_candidates

    def capture_traces(*args, **kwargs):
        """保留正式 adapter 产生的 trace，供边界资格分支专项核验。"""
        traces = real_recognize(*args, **kwargs)
        captured.extend(traces)
        return traces

    monkeypatch.setattr(
        candidate_runtime,
        "recognize_structure_candidates",
        capture_traces,
    )
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        install_language_graph_protocols(
            context,
            occurrence_protocol=OccurrenceProtocol((3366, 1)),
            span_protocol=_span_protocol(),
            boundary_protocol=_boundary_protocol(),
        )
        mapper = _MappedStructureMapper()
        install_language_candidate_runtime(
            context,
            _language_runtime_protocol(mapper),
        )
        corpus = [
            CollectedItem(
                tokens=list(tokens),
                raw_text="".join(tokens),
                source=SOURCE_BARE_TEXT,
                modality=MODALITY_LANGUAGE,
                lang=LANG_ZH,
                domain=DOMAIN_TEXT,
            )
            for tokens in (
                ("甲", "记号", "乙"),
                ("丙", "记号", "丁"),
                ("戊", "记号", "己"),
            )
        ]
        _discover_and_recognize_lang_structures(context, corpus)
        assert len(captured) == 1
        trace = captured[0]
        assert trace.adopted is True

        inactive_mapper = _EmptyStructureBoundaryMapper()
        inactive_report = apply_structure_boundary_evidence(
            context,
            corpus,
            (replace(trace, adopted=False),),
            inactive_mapper,
        )
        assert inactive_report.inactive_blocked == 1
        assert inactive_report.eligible_traces == 0
        assert inactive_mapper.calls == 0

        partial_input = replace(
            trace.input_value,
            occurrences=trace.input_value.occurrences[:-1],
            token_spans=trace.input_value.token_spans[:-1],
        )
        partial_mapper = _EmptyStructureBoundaryMapper()
        partial_report = apply_structure_boundary_evidence(
            context,
            corpus,
            (replace(trace, input_value=partial_input),),
            partial_mapper,
        )
        assert partial_report.partial_document_blocked == 1
        assert partial_report.eligible_traces == 0
        assert partial_mapper.calls == 0
        assert all(item.boundary_profile is None for item in corpus)
    finally:
        backend.close()


def test_restored_structure_projection_is_read_only_and_cannot_feed_boundary(
        tmp_path):
    """载入 operator 无 forming roots 时仍可只读识别，但不得续写候选或边界 Evidence。"""
    first_backend = DictBackend()
    try:
        first = make_train_context(first_backend)
        install_language_graph_protocols(
            first,
            occurrence_protocol=OccurrenceProtocol((3366, 1)),
            span_protocol=_span_protocol(),
            boundary_protocol=_boundary_protocol(),
        )
        first_mapper = _MappedStructureMapper()
        install_language_candidate_runtime(
            first,
            _language_runtime_protocol(first_mapper),
        )
        training = [
            CollectedItem(
                tokens=list(tokens),
                raw_text="".join(tokens),
                source=SOURCE_BARE_TEXT,
                modality=MODALITY_LANGUAGE,
                lang=LANG_ZH,
                domain=DOMAIN_TEXT,
            )
            for tokens in (
                ("甲", "记号", "乙"),
                ("丙", "记号", "丁"),
                ("戊", "记号", "己"),
            )
        ]
        discovered, _, _ = _discover_and_recognize_lang_structures(
            first, training)
        assert len(discovered) == 1
        dump_run(
            first_backend,
            str(tmp_path),
            "h05_structure_restore",
            spaces=[first.space_id],
            tables=FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id="h05_structure_restore",
            ).dump_tables,
        )
    finally:
        first_backend.close()

    restored_backend = DictBackend()
    try:
        restored = make_train_context(restored_backend)
        assert load_run(
            restored_backend,
            str(tmp_path),
            "h05_structure_restore",
        ) == [1]
        install_language_graph_protocols(
            restored,
            occurrence_protocol=OccurrenceProtocol((3366, 1)),
            span_protocol=_span_protocol(),
            boundary_protocol=_boundary_protocol(),
        )
        restored_mapper = _MappedStructureMapper(require_formation=False)
        install_language_candidate_runtime(
            restored,
            _language_runtime_protocol(
                restored_mapper,
                structure_boundary_mapper=_MappedStructureBoundaryMapper(),
            ),
        )
        loaded = load_discovered_operators(restored_backend, space_id=1)
        assert loaded and all(not item.forming_roots for item in loaded)
        held_out = CollectedItem(
            tokens=["庚", "记号", "辛"],
            raw_text="庚记号辛",
            source=SOURCE_BARE_TEXT,
            modality=MODALITY_LANGUAGE,
            lang=LANG_ZH,
            domain=DOMAIN_TEXT,
        )

        _, recognitions, _ = _discover_and_recognize_lang_structures(
            restored,
            [held_out],
            existing_operators=loaded,
        )

        report = restored.structure_candidate_reports[-1]
        assert len(recognitions) == 1
        assert report.prediction_count == 0
        assert report.recovered_read_count == 1
        assert restored.structure_boundary_report.read_only_blocked == 1
        assert restored.structure_boundary_report.evidence_added == 0
        assert held_out.boundary_profile is None
        with pytest.raises(
                CandidateHistoryUnavailableError,
                match="持久历史"):
            restored.structure_candidate_runtime.hypothesis_for_candidate(
                restored_mapper.candidate)
    finally:
        restored_backend.close()


def test_formal_sense_hook_uses_occurrence_typed_chain_and_not_legacy_first(
        tmp_path, monkeypatch):
    """正式 boot 全量迁移旧目录，round 只由 occurrence 独立揭示晋升 typed Sense。"""
    import importlib
    from pure_integer_ai.training import stages as training_stages

    formal_module = importlib.import_module(
        "pure_integer_ai.experiments.formal_train")
    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    monkeypatch.setattr(
        formal_module,
        "resolve_sense_facts",
        lambda language: [("甲", ["义一", "义二"])],
    )
    mapper = _MappedSenseMapper()
    backend = DictBackend()
    try:
        item = CollectedItem(
            tokens=["甲"],
            raw_text="甲",
            source=SOURCE_BARE_TEXT,
            modality=MODALITY_LANGUAGE,
            lang=LANG_ZH,
            domain=DOMAIN_TEXT,
        )
        result = formal_train(
            FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id="h05-sense-formal",
                rounds_per_stage=1,
                active_training_stages=(STAGE1_SKELETON,),
                persist_graph_dump=False,
                language_occurrence_protocol=OccurrenceProtocol((3361, 1)),
                language_span_protocol=_span_protocol(),
                language_candidate_protocol=_language_runtime_protocol(
                    sense_mapper=mapper),
            ),
            [item],
            backend=backend,
            runner=DefaultRoundRunner(),
        )

        report = result.sense_candidate_report
        rows = read_sense_candidates(
            backend, 1, sense_surface_hash("甲"))
        assert report.formation_inputs == 1
        assert report.candidates_registered == 2
        assert report.prediction_count == 1
        assert report.support_count == 1
        assert report.adopted_count == 1
        assert all(observed == 0 for _ref, _base, _success, observed in rows)
        assert mapper._primary is not None
        assert result.structure_candidate_reports[-1].prediction_count == 0
    finally:
        backend.close()


def test_production_sense_lookup_does_not_choose_between_active_senses():
    """正式单值 caller 面对两个 active typed Sense 时必须保持无选择。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        context.occurrence_index = object()
        context.span_index = object()
        mapper = _MappedSenseMapper()
        install_language_candidate_runtime(
            context,
            _language_runtime_protocol(sense_mapper=mapper),
        )
        _seed_legacy_sense_rows(backend, "甲")
        formations = context.sense_candidate_course_runtime.form_legacy(
            context,
            runtime_language=LANG_ZH,
            surface="甲",
        )
        for source_id, formation in enumerate(formations, start=31):
            _activate_mapped_sense(
                context, formation, source_id=source_id)

        lookup, used_legacy = observe_sense_lookup(
            context, runtime_language=LANG_ZH)
        assert used_legacy is False
        assert lookup("甲") == []
        assert len(context.sense_candidate_consumer.lookup(mapper.atom)) == 2
    finally:
        backend.close()


def test_archived_typed_sense_cannot_fall_back_to_legacy_directory():
    """typed Sense 退出后旧目录仍有候选也不能重新成为正式识别结果。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        context.occurrence_index = object()
        context.span_index = object()
        mapper = _MappedSenseMapper()
        install_language_candidate_runtime(
            context,
            _language_runtime_protocol(sense_mapper=mapper),
        )
        _seed_legacy_sense_rows(backend, "甲")
        formations = context.sense_candidate_course_runtime.form_legacy(
            context,
            runtime_language=LANG_ZH,
            surface="甲",
        )
        primary = next(
            item for item in formations if item.spec.sense == mapper._primary)
        _activate_mapped_sense(context, primary, source_id=41)

        hypothesis = context.sense_candidate_runtime.hypothesis_for_candidate(
            primary.spec.sense)
        observation = _source(42)
        event_key = (3363, 42)
        evidence_seq, decision_seq, projection_seq = (
            context.sense_candidate_runtime.next_timestamps(3))
        context.sense_candidate_runtime.recognize(
            hypothesis,
            observation=observation,
            scope=document_scope(observation),
            event_key=event_key,
            visible_inputs=(primary.spec.atom, primary.spec.context),
            predicted=primary.spec.concept,
            revealed=_reveal(
                observation,
                event_key=event_key,
                refuted=(primary.spec.concept,),
            ),
            timestamp_seq=evidence_seq,
            resolve_timestamp_seq=decision_seq,
            projection_timestamp_seq=projection_seq,
            archive_refuted=True,
        )

        lookup, used_legacy = observe_sense_lookup(
            context, runtime_language=LANG_ZH)
        assert used_legacy is False
        assert lookup("甲") == []
        assert len(read_sense_candidates(
            backend, 1, sense_surface_hash("甲"))) == 2
    finally:
        backend.close()


def test_dump_load_restores_active_sense_read_only_and_blocks_new_evidence(
        tmp_path):
    """恢复图保留 active Sense 行为，但缺持久历史时不能伪造 H-00 续写。"""
    backend = DictBackend()
    mapper = _MappedSenseMapper()
    try:
        context = make_train_context(backend)
        context.occurrence_index = object()
        context.span_index = object()
        install_language_candidate_runtime(
            context,
            _language_runtime_protocol(sense_mapper=mapper),
        )
        _seed_legacy_sense_rows(backend, "甲")
        formations = context.sense_candidate_course_runtime.form_legacy(
            context,
            runtime_language=LANG_ZH,
            surface="甲",
        )
        primary = next(
            item for item in formations if item.spec.sense == mapper._primary)
        _activate_mapped_sense(context, primary, source_id=51)
        lookup, _ = observe_sense_lookup(context, runtime_language=LANG_ZH)
        expected = lookup("甲")
        assert len(expected) == 1
        dump_run(
            backend,
            str(tmp_path),
            "h05_sense_restore",
            spaces=[context.space_id],
            tables=FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id="h05_sense_restore",
            ).dump_tables,
        )
    finally:
        backend.close()

    restored_backend = DictBackend()
    try:
        restored_context = make_train_context(restored_backend)
        assert load_run(
            restored_backend,
            str(tmp_path),
            "h05_sense_restore",
        ) == [1]
        restored_context.occurrence_index = object()
        restored_context.span_index = object()
        restored_mapper = _MappedSenseMapper()
        install_language_candidate_runtime(
            restored_context,
            _language_runtime_protocol(sense_mapper=restored_mapper),
        )
        restored_formations = (
            restored_context.sense_candidate_course_runtime.form_legacy(
                restored_context,
                runtime_language=LANG_ZH,
                surface="甲",
            ))
        restored_lookup, used_legacy = observe_sense_lookup(
            restored_context, runtime_language=LANG_ZH)

        assert used_legacy is False
        assert restored_lookup("甲") == expected
        assert (
            restored_context.sense_candidate_course_runtime.report()
            .candidates_recovered == 2)
        with pytest.raises(
                CandidateHistoryUnavailableError,
                match="持久历史"):
            restored_context.sense_candidate_runtime.hypothesis_for_candidate(
                restored_formations[0].spec.sense)
    finally:
        restored_backend.close()
