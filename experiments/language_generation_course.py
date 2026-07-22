"""版本化加载 connector 理论课程并建立 PH2 Core 候选 owner。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisLedger
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorValueProtocol,
    LanguageGenerationConnector,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorRuntimePolicy,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.language_generation_connector_candidate import (
    LanguageConnectorCandidateMapper,
    LanguageConnectorCandidateProtocol,
    LanguageConnectorCandidateRuntime,
)
from pure_integer_ai.experiments.language_generation_connector_factory import (
    ActiveLanguageConnectorFactory,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageConnectorGraphPredicates,
    LanguageGenerationConnectorGraph,
)
from pure_integer_ai.experiments.language_generation_connector_stage4 import (
    LanguageConnectorStage4Policy,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend


_COURSE_SCHEMA_VERSION = 1


class LanguageGenerationCourseError(RuntimeError):
    """课程版本、内容、图状态或引导 Evidence 不一致。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验版本和事件命名使用的非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _identity_key(value: ObjectIdentity) -> tuple[int, ...]:
    """返回一等对象的完整稳定键。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError("课程身份必须是 ObjectIdentity")
    return value.stable_key()


@dataclass(frozen=True)
class LanguageConnectorCourseRecognition:
    """声明一次独立 reveal 及其 H-00/H-04/图投影逻辑序。"""

    connector: ObjectIdentity
    visible_inputs: tuple[ObjectIdentity, ...]
    revealed: RevealedObjectObservation
    timestamp_seq: int
    resolve_timestamp_seq: int
    projection_timestamp_seq: int

    def __post_init__(self) -> None:
        """核验一次引导识别的输入、揭示和三段逻辑序完整一致。"""
        _identity_key(self.connector)
        if (not isinstance(self.visible_inputs, tuple)
                or not self.visible_inputs
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.visible_inputs)):
            raise TypeError("connector 课程 visible_inputs 必须是非空对象 tuple")
        if len(set(self.visible_inputs)) != len(self.visible_inputs):
            raise ValueError("connector 课程 visible_inputs 不得重复")
        if self.connector not in self.visible_inputs:
            raise ValueError("connector 课程 visible_inputs 必须包含预测理论")
        if not isinstance(self.revealed, RevealedObjectObservation):
            raise TypeError("connector 课程 revealed 类型错误")
        assert_int(
            self.timestamp_seq,
            self.resolve_timestamp_seq,
            self.projection_timestamp_seq,
            _where="LanguageConnectorCourseRecognition",
        )
        if (type(self.timestamp_seq) is not int
                or type(self.resolve_timestamp_seq) is not int
                or type(self.projection_timestamp_seq) is not int
                or self.timestamp_seq < 0
                or self.resolve_timestamp_seq <= self.timestamp_seq
                or self.projection_timestamp_seq
                <= self.resolve_timestamp_seq):
            raise ValueError("connector 课程三段逻辑序必须严格递增且非负")

    def stable_key(self) -> tuple[int, ...]:
        """返回预测输入、独立揭示和三段逻辑序的完整键。"""
        revealed = self.revealed
        result = [
            *_packed(self.connector.stable_key()),
            len(self.visible_inputs),
        ]
        for item in self.visible_inputs:
            result.extend(_packed(item.stable_key()))
        result.extend((
            *_packed(revealed.observation.stable_key()),
            *_packed(revealed.scope.stable_key()),
            *_packed(revealed.event_key),
            *_packed(revealed.verifier_source.stable_key()),
            len(revealed.supported_targets),
        ))
        for item in revealed.supported_targets:
            result.extend(_packed(item.stable_key()))
        result.append(len(revealed.refuted_targets))
        for item in revealed.refuted_targets:
            result.extend(_packed(item.stable_key()))
        result.extend((
            *_packed(revealed.trace),
            self.timestamp_seq,
            self.resolve_timestamp_seq,
            self.projection_timestamp_seq,
        ))
        return tuple(result)


@dataclass(frozen=True)
class LanguageConnectorCourseTemplate:
    """把一个 connector 理论绑定到形成来源、图 scope 和引导揭示。"""

    template: LanguageGenerationConnectorTemplate
    forming_sources: tuple[SourceRef, ...]
    scope: ScopeIdentity
    timestamp_base: int
    recognitions: tuple[LanguageConnectorCourseRecognition, ...]

    def __post_init__(self) -> None:
        """核验单个理论课程的形成来源、作用域和独立揭示边界。"""
        if not isinstance(self.template, LanguageGenerationConnectorTemplate):
            raise TypeError("connector 课程 template 类型错误")
        if (not isinstance(self.forming_sources, tuple)
                or not self.forming_sources
                or any(not isinstance(item, SourceRef)
                       for item in self.forming_sources)):
            raise TypeError("connector 课程 forming_sources 必须是非空来源 tuple")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("connector 课程 forming_sources 不得重复")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("connector 课程 scope 类型错误")
        assert_int(self.timestamp_base, _where="connector timestamp_base")
        if type(self.timestamp_base) is not int or self.timestamp_base < 0:
            raise ValueError("connector timestamp_base 必须为非负严格整数")
        if (not isinstance(self.recognitions, tuple)
                or not self.recognitions
                or any(not isinstance(
                    item, LanguageConnectorCourseRecognition)
                       for item in self.recognitions)):
            raise TypeError("connector 课程 recognitions 必须是非空声明 tuple")
        if any(item.connector != self.template.connector
               for item in self.recognitions):
            raise ValueError("connector 课程 recognition 指向其他理论")
        event_keys = tuple(
            item.revealed.event_key for item in self.recognitions)
        if len(set(event_keys)) != len(event_keys):
            raise ValueError("同一 connector 课程不得重复 reveal event_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回理论、形成来源、scope 和全部 reveal 的规范键。"""
        result = [
            *_packed(self.template.stable_key()),
            len(self.forming_sources),
        ]
        for source in self.forming_sources:
            result.extend(_packed(source.stable_key()))
        result.extend((
            *_packed(self.scope.stable_key()),
            self.timestamp_base,
            len(self.recognitions),
        ))
        for recognition in self.recognitions:
            result.extend(_packed(recognition.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class LanguageGenerationCourseManifest:
    """保存默认 connector 课程的版本、开放协议、理论和运行策略。"""

    schema_version: int
    course_version: tuple[int, ...]
    connector_predicates: tuple[ObjectIdentity, ...]
    candidate_projection: CandidateProjectionProtocol
    candidate_protocol: LanguageConnectorCandidateProtocol
    learning_protocol: EvidenceCandidateProtocol
    verifier_protocol: IndependentVerifierProtocol
    projection_metadata: CandidateProjectionMetadata
    value_protocol: LanguageConnectorValueProtocol
    surface_protocol: object
    runtime_policy: LanguageGenerationConnectorRuntimePolicy
    stage4_policy: LanguageConnectorStage4Policy
    templates: tuple[LanguageConnectorCourseTemplate, ...]

    def __post_init__(self) -> None:
        """核验 manifest 协议、理论身份、运行策略和课程覆盖关系。"""
        from pure_integer_ai.cognition.shared.generation_surface import (
            GenerationSurfaceProtocol,
        )

        assert_int(self.schema_version, _where="connector course schema")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("connector course schema_version 必须为严格正整数")
        _strict_key(self.course_version, label="connector course version")
        if (not isinstance(self.connector_predicates, tuple)
                or len(self.connector_predicates) != 21
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_CONCEPT
                       for item in self.connector_predicates)):
            raise ValueError("connector course 必须声明 21 个一等 Concept predicate")
        if len(set(self.connector_predicates)) != len(
                self.connector_predicates):
            raise ValueError("connector course predicate 不得重复")
        for label, value, expected in (
                ("candidate projection", self.candidate_projection,
                 CandidateProjectionProtocol),
                ("candidate protocol", self.candidate_protocol,
                 LanguageConnectorCandidateProtocol),
                ("learning protocol", self.learning_protocol,
                 EvidenceCandidateProtocol),
                ("verifier protocol", self.verifier_protocol,
                 IndependentVerifierProtocol),
                ("projection metadata", self.projection_metadata,
                 CandidateProjectionMetadata),
                ("value protocol", self.value_protocol,
                 LanguageConnectorValueProtocol),
                ("surface protocol", self.surface_protocol,
                 GenerationSurfaceProtocol),
                ("runtime policy", self.runtime_policy,
                 LanguageGenerationConnectorRuntimePolicy),
                ("stage4 policy", self.stage4_policy,
                 LanguageConnectorStage4Policy)):
            if not isinstance(value, expected):
                raise TypeError(f"connector course {label} 类型错误")
        if (not isinstance(self.templates, tuple) or not self.templates
                or any(not isinstance(item, LanguageConnectorCourseTemplate)
                       for item in self.templates)):
            raise TypeError("connector course templates 必须是非空课程 tuple")
        definitions = tuple(item.template for item in self.templates)
        if len({item.connector for item in definitions}) != len(definitions):
            raise ValueError("connector course 不得重复理论身份")
        if len({item.match_key() for item in definitions}) != len(definitions):
            raise ValueError("默认 connector course 不得声明歧义匹配模板")
        owned = tuple(
            identity
            for definition in definitions
            for identity in (
                definition.connector,
                definition.constraint_set,
                definition.context_set,
                *(item.binding for item in definition.bindings),
                *(item.directive for item in definition.surface),
                *(item.prefix_route for item in definition.surface),
            )
        )
        if len(set(owned)) != len(owned):
            raise ValueError("connector course 不得跨模板复用内部结构身份")
        if any(len(item.forming_sources)
               < self.learning_protocol.minimum_forming_sources
               for item in self.templates):
            raise ValueError("connector course forming 来源未达到学习协议下限")
        LanguageGenerationConnector(
            LanguageGenerationConnectorRegistry(
                self.value_protocol,
                definitions,
            ),
            self.runtime_policy,
            self.surface_protocol,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 manifest 全字段规范整数键，供内容锁和漂移核验。"""
        projection = self.candidate_projection
        verifier = self.verifier_protocol
        learning = self.learning_protocol
        metadata = self.projection_metadata
        result = [
            self.schema_version,
            *_packed(self.course_version),
            len(self.connector_predicates),
        ]
        for identity in self.connector_predicates:
            result.extend(_packed(identity.stable_key()))
        for identity in (
                *projection.predicate_identities(),
                *projection.state_identities(),
                *projection.kind_identities()):
            result.extend(_packed(identity.stable_key()))
        result.extend((
            *_packed(projection.event_namespace_key),
            *_packed(self.candidate_protocol.stable_key()),
            *_packed(learning.hypothesis_kind_key),
            *_packed(learning.formation_reason_key),
            *_packed(learning.aggregate_source.stable_key()),
            *_packed(learning.aggregate_scope.stable_key()),
            learning.minimum_forming_sources,
            *_packed(verifier.authority.stable_key()),
            *_packed(verifier.authority_version),
            *_packed(verifier.support_reason_key),
            *_packed(verifier.refute_reason_key),
            *_packed(verifier.unknown_reason_key),
            metadata.provenance_kind,
            metadata.epistemic_origin,
            metadata.content_version,
            *_packed(metadata.qualifiers),
            *_packed(self.value_protocol.stable_key()),
            *_packed(self.surface_protocol.stable_key()),
            *_packed(self.runtime_policy.stable_key()),
            *_packed(self.stage4_policy.stable_key()),
            len(self.templates),
        ))
        for template in self.templates:
            result.extend(_packed(template.stable_key()))
        return tuple(result)

    def sha256(self) -> str:
        """对完整规范整数键计算不依赖路径和对象地址的 SHA-256。"""
        payload = json.dumps(
            self.stable_key(),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LanguageGenerationCourseReport:
    """记录一次课程装配的内容锁和候选全链计数。"""

    manifest_sha256: str
    schema_version: int
    course_version: tuple[int, ...]
    template_count: int
    evidence_count: int
    decision_count: int
    projection_event_count: int
    active_count: int


@dataclass(frozen=True)
class LoadedLanguageGenerationCourse:
    """返回可交给 production 装配的 active factory、stage4 策略和报告。"""

    connector_factory: ActiveLanguageConnectorFactory
    stage4_policy: LanguageConnectorStage4Policy
    candidates: LanguageConnectorCandidateRuntime
    report: LanguageGenerationCourseReport


@dataclass(frozen=True)
class _PreflightCourseState:
    """保存只读预检已恢复的 facade 和候选 owner，避免正式阶段重复解码。"""

    definition_graph: LanguageGenerationConnectorGraph | None
    candidate_graph: CandidateProjectionGraph | None
    candidates: LanguageConnectorCandidateRuntime | None


class LanguageGenerationCourseLoader:
    """在正式后端首写前用隔离副本验证并加载一个内容锁定课程。"""

    def __init__(
            self,
            manifest: LanguageGenerationCourseManifest,
            expected_sha256: str,
            ) -> None:
        """绑定不可变 manifest，并规范化调用方声明的预期内容锁。"""
        if not isinstance(manifest, LanguageGenerationCourseManifest):
            raise TypeError("connector course manifest 类型错误")
        digest = expected_sha256.lower()
        if (len(digest) != 64
                or any(item not in "0123456789abcdef" for item in digest)):
            raise ValueError("connector course expected_sha256 格式错误")
        self.manifest = manifest
        self.expected_sha256 = digest

    def load(self, ctx: TrainContext) -> LoadedLanguageGenerationCourse:
        """核验版本和内容锁，经克隆全链预检后幂等写入正式 Core。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("connector course ctx 类型错误")
        digest = self.manifest.sha256()
        if digest != self.expected_sha256:
            raise LanguageGenerationCourseError("connector course 内容哈希漂移")
        if self.manifest.schema_version != _COURSE_SCHEMA_VERSION:
            raise LanguageGenerationCourseError(
                "connector course schema 版本不受支持")
        precedence = ctx.precedence_relation_runtime
        if precedence is None:
            raise LanguageGenerationCourseError(
                "connector course 需要先安装 R-06/S-07 runtime")
        order_graph = getattr(precedence, "order_graph", None)
        if not isinstance(order_graph, StructureOrderGraph):
            raise LanguageGenerationCourseError(
                "connector course 无法取得当前 S-07 order graph")
        if order_graph.ontology is not ctx.graph_ontology:
            raise LanguageGenerationCourseError(
                "connector course S-07 未绑定当前 TrainContext")
        prepared = self._preflight(ctx, order_graph)
        return self._apply(ctx, order_graph, prepared)

    def state_key(self) -> tuple:
        """返回预期内容锁和 manifest 完整键，供配置隔离核验。"""
        return self.expected_sha256, self.manifest.stable_key()

    def _preflight(
            self,
            ctx: TrainContext,
            order_graph: StructureOrderGraph,
            ) -> _PreflightCourseState:
        """只读核验既有图，并在内存 engine 中预演全部课程状态转换。"""
        manifest = self.manifest
        ontology = ctx.graph_ontology
        metadata = manifest.projection_metadata
        for item in manifest.templates:
            LanguageGenerationConnectorGraph.validate_against_order_graph(
                order_graph,
                manifest.value_protocol,
                item.template,
            )

        connector_refs = tuple(
            ontology.resolve(item) for item in manifest.connector_predicates)
        connector_exists = tuple(item is not None for item in connector_refs)
        definition_graph = None
        if any(connector_exists) and not all(connector_exists):
            raise LanguageGenerationCourseError(
                "connector course predicate 只恢复了部分协议")
        if all(connector_exists):
            definition_graph = LanguageGenerationConnectorGraph(
                ontology,
                order_graph,
                LanguageConnectorGraphPredicates(*connector_refs),
                manifest.value_protocol,
            )
            for item in manifest.templates:
                definition_graph.preflight(
                    item.template,
                    scope=item.scope,
                    provenance_kind=metadata.provenance_kind,
                    epistemic_origin=metadata.epistemic_origin,
                    content_version=metadata.content_version,
                    qualifiers=metadata.qualifiers,
                )

        projection_identities = (
            *manifest.candidate_projection.predicate_identities(),
            *manifest.candidate_projection.state_identities(),
            *manifest.candidate_projection.kind_identities(),
        )
        projection_exists = tuple(
            ontology.resolve(item) is not None
            for item in projection_identities
        )
        if any(projection_exists) and not all(projection_exists):
            raise LanguageGenerationCourseError(
                "connector candidate projection 只恢复了部分协议")
        history = ctx.training_candidate_history
        if history is None:
            raise LanguageGenerationCourseError(
                "connector course 缺少 PH2 Core 训练历史")
        history_protocol = TrainingHypothesisHistoryProtocol(
            manifest.candidate_protocol.stable_key(),
            manifest.learning_protocol.hypothesis_kind_key,
            manifest.learning_protocol.aggregate_source,
            manifest.learning_protocol.aggregate_scope,
        )
        sink = TrainingHypothesisEventSink(history, history_protocol)
        historical_hypotheses = sink.hypotheses()
        if historical_hypotheses and not all(projection_exists):
            raise LanguageGenerationCourseError(
                "connector Core 历史存在但候选图协议缺失")

        candidate_graph = None
        base_runtime = None
        if all(projection_exists):
            candidate_graph = CandidateProjectionGraph(
                ontology,
                manifest.candidate_projection,
            )
            if definition_graph is None:
                raise LanguageGenerationCourseError(
                    "connector 候选图存在但理论图协议缺失")
            base_runtime = self._restored_runtime(
                ctx,
                definition_graph,
                candidate_graph,
            )
            engine = base_runtime.learning.engine.clone()
        else:
            engine = EvidenceCandidateEngine(manifest.learning_protocol)
        mapper = LanguageConnectorCandidateMapper(
            manifest.candidate_protocol)
        verifier = IndependentObjectVerifier(manifest.verifier_protocol)
        definitions = tuple(
            mapper.definition(item.template, item.forming_sources)
            for item in manifest.templates
        )
        for item, definition in zip(manifest.templates, definitions):
            if base_runtime is not None:
                base_runtime.learning.preflight_register(
                    definition,
                    timestamp_base=item.timestamp_base,
                )
        self._preflight_lifecycle(definitions, engine, verifier)
        return _PreflightCourseState(
            definition_graph,
            candidate_graph,
            base_runtime,
        )

    def _preflight_lifecycle(
            self,
            definitions,
            engine: EvidenceCandidateEngine,
            verifier: IndependentObjectVerifier,
            ) -> None:
        """在小型临时图真实执行候选定义和 lifecycle，验证全部 projection 原子前置。"""
        backend = DictBackend()
        try:
            ctx = make_train_context(backend)
            graph = CandidateProjectionGraph(
                ctx.graph_ontology,
                self.manifest.candidate_projection,
            )
            runtime = CandidateLearningRuntime(
                engine,
                graph,
                verifier,
                self.manifest.projection_metadata,
            )
            for item, definition in zip(
                    self.manifest.templates, definitions):
                hypothesis = runtime.register(
                    definition,
                    timestamp_base=item.timestamp_base,
                )
                for recognition in item.recognitions:
                    runtime.recognize(
                        hypothesis,
                        observation=recognition.revealed.observation,
                        scope=recognition.revealed.scope,
                        event_key=recognition.revealed.event_key,
                        visible_inputs=recognition.visible_inputs,
                        predicted=item.template.connector,
                        revealed=recognition.revealed,
                        timestamp_seq=recognition.timestamp_seq,
                        resolve_timestamp_seq=recognition.resolve_timestamp_seq,
                        projection_timestamp_seq=(
                            recognition.projection_timestamp_seq),
                    )
                if runtime.engine.active(hypothesis) is None:
                    raise LanguageGenerationCourseError(
                        "connector course 未使全部默认理论进入唯一 active 投影")
        finally:
            backend.close()

    def _restored_runtime(
            self,
            ctx: TrainContext,
            definition_graph: LanguageGenerationConnectorGraph,
            candidate_graph: CandidateProjectionGraph,
            ) -> LanguageConnectorCandidateRuntime:
        """从当前 Core 历史只读恢复可供预检或正式续写的候选 owner。"""
        manifest = self.manifest
        history = ctx.training_candidate_history
        if history is None:
            raise LanguageGenerationCourseError(
                "connector course 缺少 PH2 Core 训练历史")
        history_protocol = TrainingHypothesisHistoryProtocol(
            manifest.candidate_protocol.stable_key(),
            manifest.learning_protocol.hypothesis_kind_key,
            manifest.learning_protocol.aggregate_source,
            manifest.learning_protocol.aggregate_scope,
        )
        sink = TrainingHypothesisEventSink(history, history_protocol)
        learning = CandidateLearningRuntime(
            EvidenceCandidateEngine(
                manifest.learning_protocol,
                ledger=HypothesisLedger(sink),
            ),
            candidate_graph,
            IndependentObjectVerifier(manifest.verifier_protocol),
            manifest.projection_metadata,
        )
        return LanguageConnectorCandidateRuntime(
            definition_graph,
            learning,
            manifest.candidate_protocol,
        ).restore_for_training_graphs(
            definition_graph,
            candidate_graph,
            history,
        )

    def _apply(
            self,
            ctx: TrainContext,
            order_graph: StructureOrderGraph,
            prepared: _PreflightCourseState,
            ) -> LoadedLanguageGenerationCourse:
        """在一个已选后端恢复候选历史、物化理论并提交声明的 reveal。"""
        manifest = self.manifest
        ontology = ctx.graph_ontology
        if prepared.candidates is None:
            predicate_refs = tuple(
                ontology.materialize(item)
                for item in manifest.connector_predicates
            )
            definition_graph = LanguageGenerationConnectorGraph(
                ontology,
                order_graph,
                LanguageConnectorGraphPredicates(*predicate_refs),
                manifest.value_protocol,
            )
            candidate_graph = CandidateProjectionGraph(
                ontology,
                manifest.candidate_projection,
            )
            candidates = self._restored_runtime(
                ctx,
                definition_graph,
                candidate_graph,
            )
        else:
            definition_graph = prepared.definition_graph
            candidate_graph = prepared.candidate_graph
            candidates = prepared.candidates
            if (definition_graph is None or candidate_graph is None
                    or definition_graph.ontology is not ontology
                    or candidate_graph.ontology is not ontology):
                raise RuntimeError("connector course 预检 facade 归属发生漂移")
        metadata = manifest.projection_metadata
        for item in manifest.templates:
            hypothesis = candidates.register(
                item.template,
                item.forming_sources,
                scope=item.scope,
                provenance_kind=metadata.provenance_kind,
                epistemic_origin=metadata.epistemic_origin,
                content_version=metadata.content_version,
                qualifiers=metadata.qualifiers,
                timestamp_base=item.timestamp_base,
            )
            for recognition in item.recognitions:
                candidates.recognize(
                    hypothesis,
                    observation=recognition.revealed.observation,
                    scope=recognition.revealed.scope,
                    event_key=recognition.revealed.event_key,
                    visible_inputs=recognition.visible_inputs,
                    predicted=item.template.connector,
                    revealed=recognition.revealed,
                    timestamp_seq=recognition.timestamp_seq,
                    resolve_timestamp_seq=recognition.resolve_timestamp_seq,
                    projection_timestamp_seq=(
                        recognition.projection_timestamp_seq),
                )
        active = candidates.active_templates()
        expected = tuple(sorted(
            (item.template for item in manifest.templates),
            key=lambda item: item.connector.stable_key(),
        ))
        if active != expected:
            raise LanguageGenerationCourseError(
                "connector course 未使全部默认理论进入唯一 active 投影")
        report = candidates.learning.report()
        factory = ActiveLanguageConnectorFactory(
            candidates,
            manifest.runtime_policy,
            manifest.surface_protocol,
            manifest.stage4_policy.active_purpose,
        )
        return LoadedLanguageGenerationCourse(
            factory,
            manifest.stage4_policy,
            candidates,
            LanguageGenerationCourseReport(
                manifest.sha256(),
                manifest.schema_version,
                manifest.course_version,
                len(manifest.templates),
                report.evidence_count,
                report.decision_count,
                report.projection_event_count,
                report.active_projection_count,
            ),
        )


__all__ = [
    "LanguageConnectorCourseRecognition",
    "LanguageConnectorCourseTemplate",
    "LanguageGenerationCourseError",
    "LanguageGenerationCourseLoader",
    "LanguageGenerationCourseManifest",
    "LanguageGenerationCourseReport",
    "LoadedLanguageGenerationCourse",
]
