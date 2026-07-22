"""版本化加载 R-01 alias/refers/realizes 课程并恢复 PH2 Core owner。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
    AliasResolutionSelector,
)
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
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureCandidateSpec,
    RelationClosureProtocol,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseGraph,
    RelationUseGraphProtocol,
    RelationUseOwner,
    RelationUseWriteMetadata,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.typed_relation import RelationSchema
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)


_COURSE_SCHEMA_VERSION = 1


class AliasRelationCourseError(RuntimeError):
    """R-01 课程版本、内容锁、图状态或训练历史不一致。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验课程版本、命名空间和协议键使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长完整键增加长度边界。"""
    return len(value), *value


def _identity_key(value: ObjectIdentity) -> tuple[int, ...]:
    """返回一等对象的完整稳定键。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError("课程身份必须是 ObjectIdentity")
    return value.stable_key()


def _relation_field_key(field) -> tuple[int, ...]:
    """编码 R-00 开放字段的 predicate、方向和 ordinal。"""
    return (
        *_packed(field.predicate.stable_key()),
        field.ordinal,
        field.candidate_endpoint,
    )


def _relation_protocol_key(
        protocol: RelationClosureProtocol) -> tuple[int, ...]:
    """编码 relation/schema 两个动态候选字段。"""
    return (
        *_packed(_relation_field_key(protocol.relation)),
        *_packed(_relation_field_key(protocol.schema)),
    )


def _schema_key(schema: RelationSchema) -> tuple[int, ...]:
    """编码 schema、relation、全部 Role slot 和同型约束。"""
    result = [
        *_packed(schema.schema.stable_key()),
        *_packed(schema.relation.stable_key()),
        len(schema.slots),
    ]
    for slot in schema.slots:
        result.extend((
            *_packed(slot.role.stable_key()),
            len(slot.allowed_object_kinds),
            *sorted(slot.allowed_object_kinds),
            slot.min_count,
            -1 if slot.max_count is None else slot.max_count,
        ))
    result.append(len(schema.same_kind_constraints))
    for constraint in schema.same_kind_constraints:
        result.extend((
            *_packed(constraint.constraint.stable_key()),
            len(constraint.roles),
        ))
        for role in constraint.roles:
            result.extend(_packed(role.stable_key()))
    return tuple(result)


def _proposition_key(spec: RelationClosureCandidateSpec) -> tuple[int, ...]:
    """编码 S-00 命题拓扑和 R-00 候选定义，避免以对象地址参与内容锁。"""
    definition = spec.proposition
    result = [
        *_packed(definition.proposition.stable_key()),
        *_packed(definition.predicate.stable_key()),
        *_packed(definition.source_anchor.stable_key()),
        *_packed(definition.context.stable_key()),
        len(definition.bindings),
    ]
    for binding in definition.canonical_bindings():
        result.extend((
            *_packed(binding.role.stable_key()),
            *_packed(binding.filler.stable_key()),
            binding.ordinal,
        ))
    result.extend((
        *_packed(_schema_key(spec.schema)),
        *_packed(spec.competition_key),
        len(spec.forming_sources),
    ))
    for source in spec.forming_sources:
        result.extend(_packed(source.stable_key()))
    result.append(len(spec.domain_bindings))
    for binding in spec.domain_bindings:
        result.extend(_packed(binding.stable_key()))
    return tuple(result)


@dataclass(frozen=True)
class AliasRelationStatementMetadata:
    """S-00 relation Proposition statement 使用的注入式来源元数据。"""

    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验 S-00 来源元数据全部为严格整数且 provenance 为正。"""
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("relation statement qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="AliasRelationStatementMetadata",
        )
        if (type(self.provenance_kind) is not int
                or self.provenance_kind <= 0
                or type(self.epistemic_origin) is not int
                or self.epistemic_origin < 0
                or type(self.content_version) is not int
                or self.content_version < 0
                or any(type(item) is not int for item in self.qualifiers)):
            raise ValueError("relation statement 来源元数据非法")

    def kwargs(self) -> dict:
        """返回 SemanticGraph 写入接口接受的统一关键字参数。"""
        return {
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifiers": self.qualifiers,
        }

    def stable_key(self) -> tuple[int, ...]:
        """返回来源类型、认识论来源、内容版本和限定键。"""
        return (
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *_packed(self.qualifiers),
        )


@dataclass(frozen=True)
class AliasRelationCourseRecognition:
    """声明一次独立 relation reveal 及其 H-00/H-04/投影逻辑序。"""

    input: RelationClosureRecognitionInput
    timestamp_seq: int
    resolve_timestamp_seq: int
    projection_timestamp_seq: int

    def __post_init__(self) -> None:
        """核验 recognition 输入和三段可重放逻辑序。"""
        if not isinstance(self.input, RelationClosureRecognitionInput):
            raise TypeError("relation course recognition input 类型错误")
        assert_int(
            self.timestamp_seq,
            self.resolve_timestamp_seq,
            self.projection_timestamp_seq,
            _where="AliasRelationCourseRecognition",
        )
        if (type(self.timestamp_seq) is not int
                or type(self.resolve_timestamp_seq) is not int
                or type(self.projection_timestamp_seq) is not int
                or self.timestamp_seq < 0
                or self.resolve_timestamp_seq <= self.timestamp_seq
                or self.projection_timestamp_seq
                <= self.resolve_timestamp_seq):
            raise ValueError("relation course 三段逻辑序必须严格递增且非负")

    def stable_key(self) -> tuple[int, ...]:
        """返回来源化 recognition、reveal 和三段逻辑序的完整键。"""
        return (
            *_packed(self.input.stable_key()),
            self.timestamp_seq,
            self.resolve_timestamp_seq,
            self.projection_timestamp_seq,
        )


@dataclass(frozen=True)
class AliasRelationCourseEntry:
    """把一个 typed relation 候选绑定到 S-00 scope、forming 和独立 reveal。"""

    spec: RelationClosureCandidateSpec
    statement_scope: ScopeIdentity
    timestamp_base: int
    recognitions: tuple[AliasRelationCourseRecognition, ...]

    def __post_init__(self) -> None:
        """核验单候选的命题来源、形成来源和 recognition 路由。"""
        if not isinstance(self.spec, RelationClosureCandidateSpec):
            raise TypeError("relation course spec 类型错误")
        if not isinstance(self.statement_scope, ScopeIdentity):
            raise TypeError("relation course statement_scope 类型错误")
        if self.statement_scope.source != self.spec.proposition.source:
            raise ValueError("relation course S-00 scope 必须绑定 Proposition 来源")
        assert_int(self.timestamp_base, _where="relation course timestamp_base")
        if type(self.timestamp_base) is not int or self.timestamp_base < 0:
            raise ValueError("relation course timestamp_base 必须为非负严格整数")
        if (not isinstance(self.recognitions, tuple)
                or not self.recognitions
                or any(not isinstance(item, AliasRelationCourseRecognition)
                       for item in self.recognitions)):
            raise TypeError("relation course recognitions 必须是非空声明 tuple")
        proposition = self.spec.proposition.proposition
        if any(item.input.proposition != proposition
               for item in self.recognitions):
            raise ValueError("relation course recognition 指向其他 Proposition")
        routes = tuple(item.input.route_key() for item in self.recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("同一 relation course 不得重复 recognition 路由")
        forming = frozenset(self.spec.forming_sources)
        if any(
                item.input.observation in forming
                or item.input.revealed.verifier_source in forming
                for item in self.recognitions):
            raise ValueError("relation forming 与 recognition/reveal 来源必须分离")

    def stable_key(self) -> tuple[int, ...]:
        """返回 S-00/R-00 定义、scope、forming 序和全部 reveal 的规范键。"""
        result = [
            *_packed(_proposition_key(self.spec)),
            *_packed(self.statement_scope.stable_key()),
            self.timestamp_base,
            len(self.recognitions),
        ]
        for recognition in self.recognitions:
            result.extend(_packed(recognition.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AliasRelationCourseManifest:
    """保存 R-01 课程的版本、全部开放协议、schema、事实和 Use owner。"""

    schema_version: int
    course_version: tuple[int, ...]
    semantic_predicates: tuple[ObjectIdentity, ...]
    candidate_projection: CandidateProjectionProtocol
    training_history_namespace: tuple[int, ...]
    learning_protocol: EvidenceCandidateProtocol
    verifier_protocol: IndependentVerifierProtocol
    projection_metadata: CandidateProjectionMetadata
    statement_metadata: AliasRelationStatementMetadata
    relation_protocol: RelationClosureProtocol
    schemas: tuple[RelationSchema, ...]
    alias_protocol: AliasResolutionProtocol
    use_protocol: RelationUseGraphProtocol
    use_metadata: RelationUseWriteMetadata
    entries: tuple[AliasRelationCourseEntry, ...]

    def __post_init__(self) -> None:
        """核验协议、schema 覆盖、课程候选和来源独立性。"""
        assert_int(self.schema_version, _where="relation course schema")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("relation course schema_version 必须为严格正整数")
        _strict_key(self.course_version, label="relation course version")
        _strict_key(
            self.training_history_namespace,
            label="relation training history namespace",
        )
        if (not isinstance(self.semantic_predicates, tuple)
                or len(self.semantic_predicates) != 6
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_CONCEPT
                       for item in self.semantic_predicates)):
            raise ValueError("relation course 必须声明六个 S-00 Concept predicate")
        if len(set(self.semantic_predicates)) != 6:
            raise ValueError("relation course S-00 predicate 不得重复")
        for label, value, expected in (
                ("candidate projection", self.candidate_projection,
                 CandidateProjectionProtocol),
                ("learning protocol", self.learning_protocol,
                 EvidenceCandidateProtocol),
                ("verifier protocol", self.verifier_protocol,
                 IndependentVerifierProtocol),
                ("projection metadata", self.projection_metadata,
                 CandidateProjectionMetadata),
                ("statement metadata", self.statement_metadata,
                 AliasRelationStatementMetadata),
                ("relation protocol", self.relation_protocol,
                 RelationClosureProtocol),
                ("alias protocol", self.alias_protocol,
                 AliasResolutionProtocol),
                ("use protocol", self.use_protocol,
                 RelationUseGraphProtocol),
                ("use metadata", self.use_metadata,
                 RelationUseWriteMetadata)):
            if not isinstance(value, expected):
                raise TypeError(f"relation course {label} 类型错误")
        if (not isinstance(self.schemas, tuple) or not self.schemas
                or any(not isinstance(item, RelationSchema)
                       for item in self.schemas)):
            raise TypeError("relation course schemas 必须是非空 RelationSchema tuple")
        by_identity = {item.schema: item for item in self.schemas}
        if len(by_identity) != len(self.schemas):
            raise ValueError("relation course schema 身份不得重复")
        declared_groups = (
            (self.alias_protocol.alias_relation,
             self.alias_protocol.alias_schemas),
            (self.alias_protocol.refers_relation,
             self.alias_protocol.refers_schemas),
            (self.alias_protocol.realizes_relation,
             self.alias_protocol.realizes_schemas),
        )
        schema_identities = tuple(
            identity for _relation, identities in declared_groups
            for identity in identities)
        if (len(set(schema_identities)) != len(schema_identities)
                or set(schema_identities) != set(by_identity)):
            raise ValueError("relation course schemas 必须精确覆盖 R-01 三类 schema")
        for relation, identities in declared_groups:
            if any(by_identity[identity].relation != relation
                   for identity in identities):
                raise ValueError("R-01 schema 与所属 relation 身份不一致")
        if (not isinstance(self.entries, tuple) or not self.entries
                or any(not isinstance(item, AliasRelationCourseEntry)
                       for item in self.entries)):
            raise TypeError("relation course entries 必须是非空课程 tuple")
        propositions = tuple(
            item.spec.proposition.proposition for item in self.entries)
        if len(set(propositions)) != len(propositions):
            raise ValueError("relation course 不得重复 Proposition 身份")
        if any(item.spec.schema.schema not in by_identity
               or by_identity[item.spec.schema.schema] != item.spec.schema
               for item in self.entries):
            raise ValueError("relation course entry 使用了未声明或漂移的 schema")
        if any(len(item.spec.forming_sources)
               < self.learning_protocol.minimum_forming_sources
               for item in self.entries):
            raise ValueError("relation course forming 来源未达到学习协议下限")
        aggregate = self.learning_protocol.aggregate_source
        if any(
                item.spec.proposition.source.owner != aggregate.owner
                or item.spec.proposition.source.versions != aggregate.versions
                for item in self.entries):
            raise ValueError("relation Proposition 必须归属当前 aggregate owner/version")

    def training_history_protocol(self) -> TrainingHypothesisHistoryProtocol:
        """返回 R-01 候选专用、与物理 backend 无关的 Core 历史协议。"""
        return TrainingHypothesisHistoryProtocol(
            self.training_history_namespace,
            self.learning_protocol.hypothesis_kind_key,
            self.learning_protocol.aggregate_source,
            self.learning_protocol.aggregate_scope,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 manifest 全字段规范整数键，供内容锁和漂移核验。"""
        projection = self.candidate_projection
        learning = self.learning_protocol
        verifier = self.verifier_protocol
        metadata = self.projection_metadata
        result = [
            self.schema_version,
            *_packed(self.course_version),
            len(self.semantic_predicates),
        ]
        for identity in self.semantic_predicates:
            result.extend(_packed(identity.stable_key()))
        for identity in (
                *projection.predicate_identities(),
                *projection.state_identities(),
                *projection.kind_identities()):
            result.extend(_packed(identity.stable_key()))
        result.extend((
            *_packed(projection.event_namespace_key),
            *_packed(self.training_history_namespace),
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
            *_packed(self.statement_metadata.stable_key()),
            *_packed(_relation_protocol_key(self.relation_protocol)),
            len(self.schemas),
        ))
        for schema in self.schemas:
            result.extend(_packed(_schema_key(schema)))
        result.extend((
            *_packed(self.alias_protocol.stable_key()),
            *_packed(self.use_protocol.stable_key()),
            *_packed(self.use_metadata.stable_key()),
            len(self.entries),
        ))
        for entry in self.entries:
            result.extend(_packed(entry.stable_key()))
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
class AliasRelationCourseReport:
    """记录一次 R-01 课程装配的内容锁和候选全链计数。"""

    manifest_sha256: str
    schema_version: int
    course_version: tuple[int, ...]
    entry_count: int
    recognition_count: int
    evidence_count: int
    decision_count: int
    projection_event_count: int
    active_count: int
    relation_use_count: int


@dataclass(frozen=True)
class LoadedAliasRelationCourse:
    """返回当前 R-01 runtime、可跨 context 重建的 factory 和装配报告。"""

    alias: AliasRelationRuntime
    factory: "AliasRelationRuntimeFactory"
    report: AliasRelationCourseReport


@dataclass(frozen=True)
class _PreflightCourseState:
    """保存宿主只读预检已恢复的图 facade 和候选 owner。"""

    semantic_graph: SemanticGraph | None
    candidate_graph: CandidateProjectionGraph | None
    candidate_runtime: CandidateLearningRuntime | None
    use_owner: RelationUseOwner | None


class AliasRelationRuntimeFactory:
    """按同一内容锁为宿主或 V-06 context 重建独立 R-01 owner。"""

    def __init__(
            self,
            manifest: AliasRelationCourseManifest,
            expected_sha256: str,
            *,
            bound_alias: AliasRelationRuntime | None = None,
            ) -> None:
        """绑定不可变课程配置，并可复用同一宿主图上已加载的 runtime。"""
        if not isinstance(manifest, AliasRelationCourseManifest):
            raise TypeError("relation runtime factory manifest 类型错误")
        if bound_alias is not None and not isinstance(
                bound_alias, AliasRelationRuntime):
            raise TypeError("relation runtime factory bound_alias 类型错误")
        self.manifest = manifest
        self.expected_sha256 = expected_sha256
        self._bound_alias = bound_alias

    def build(self, ctx: TrainContext) -> AliasRelationRuntime:
        """返回绑定当前 context 图的 R-01 owner，其他 context 从克隆图恢复。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("relation runtime factory ctx 类型错误")
        if (self._bound_alias is not None
                and self._bound_alias.closure.semantic_graph.ontology
                is ctx.graph_ontology):
            return self._bound_alias
        return AliasRelationCourseLoader(
            self.manifest,
            self.expected_sha256,
        ).load(ctx).alias

    def clone_for_evaluation(self) -> "AliasRelationRuntimeFactory":
        """复制不可变课程配置并清空宿主 runtime 引用。"""
        return AliasRelationRuntimeFactory(
            self.manifest,
            self.expected_sha256,
        )

    def state_key(self) -> tuple:
        """返回预期内容锁和 manifest 完整键，不包含宿主可变状态。"""
        return self.expected_sha256, self.manifest.stable_key()


class AliasRelationCourseLoader:
    """在正式后端首写前用隔离副本验证并加载一个内容锁定 R-01 课程。"""

    def __init__(
            self,
            manifest: AliasRelationCourseManifest,
            expected_sha256: str,
            ) -> None:
        """绑定不可变 manifest，并规范化调用方声明的预期内容锁。"""
        if not isinstance(manifest, AliasRelationCourseManifest):
            raise TypeError("relation course manifest 类型错误")
        digest = expected_sha256.lower()
        if (len(digest) != 64
                or any(item not in "0123456789abcdef" for item in digest)):
            raise ValueError("relation course expected_sha256 格式错误")
        self.manifest = manifest
        self.expected_sha256 = digest

    def load(self, ctx: TrainContext) -> LoadedAliasRelationCourse:
        """核验版本和内容锁，经隔离全链预演后幂等写入正式 Core。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("relation course ctx 类型错误")
        digest = self.manifest.sha256()
        if digest != self.expected_sha256:
            raise AliasRelationCourseError("relation course 内容哈希漂移")
        if self.manifest.schema_version != _COURSE_SCHEMA_VERSION:
            raise AliasRelationCourseError(
                "relation course schema 版本不受支持")
        prepared = self._preflight_host(ctx)
        self._preflight_isolated()
        alias, report = self._apply(ctx, prepared)
        return LoadedAliasRelationCourse(
            alias,
            AliasRelationRuntimeFactory(
                self.manifest,
                self.expected_sha256,
                bound_alias=alias,
            ),
            report,
        )

    def state_key(self) -> tuple:
        """返回预期内容锁和 manifest 完整键，供配置隔离核验。"""
        return self.expected_sha256, self.manifest.stable_key()

    def _preflight_host(self, ctx: TrainContext) -> _PreflightCourseState:
        """只读核验宿主现有 S-00、候选历史、投影图和 Use 图边界。"""
        manifest = self.manifest
        ontology = ctx.graph_ontology
        semantic_refs = self._resolved_group(
            ontology,
            manifest.semantic_predicates,
            label="relation S-00 predicate",
        )
        projection_identities = (
            *manifest.candidate_projection.predicate_identities(),
            *manifest.candidate_projection.state_identities(),
            *manifest.candidate_projection.kind_identities(),
        )
        projection_refs = self._resolved_group(
            ontology,
            projection_identities,
            label="relation candidate projection",
        )
        use_identities = (
            *manifest.use_protocol.predicate_identities(),
            *manifest.use_protocol.state_identities(),
        )
        use_refs = self._resolved_group(
            ontology,
            use_identities,
            label="relation Core Use protocol",
        )
        history = ctx.training_candidate_history
        if history is None:
            raise AliasRelationCourseError(
                "relation course 缺少 PH2 Core 训练历史")
        sink = TrainingHypothesisEventSink(
            history,
            manifest.training_history_protocol(),
        )
        historical_hypotheses = sink.hypotheses()
        if historical_hypotheses and projection_refs is None:
            raise AliasRelationCourseError(
                "relation Core 历史存在但候选图协议缺失")
        if historical_hypotheses and semantic_refs is None:
            raise AliasRelationCourseError(
                "relation Core 历史存在但 S-00 图协议缺失")

        semantic_graph = None
        if semantic_refs is not None:
            semantic_graph = SemanticGraph(
                ontology,
                AtomicPropositionPredicates(*semantic_refs),
            )
            for entry in manifest.entries:
                semantic_graph.preflight_atomic(
                    entry.spec.proposition,
                    scope=entry.statement_scope,
                    **manifest.statement_metadata.kwargs(),
                )
        else:
            self._reject_orphan_semantic_roots(ontology)

        candidate_graph = None
        candidate_runtime = None
        if projection_refs is not None:
            candidate_graph = CandidateProjectionGraph(
                ontology,
                manifest.candidate_projection,
            )
            candidate_runtime = CandidateLearningRuntime.restore_for_training_graph(
                manifest.learning_protocol,
                candidate_graph,
                IndependentObjectVerifier(manifest.verifier_protocol),
                manifest.projection_metadata,
                history,
                manifest.training_history_protocol(),
            )
            expected_definitions = {
                entry.spec.candidate_definition(manifest.relation_protocol)
                for entry in manifest.entries
            }
            restored_definitions = set(
                candidate_runtime.engine.definitions())
            if not restored_definitions.issubset(expected_definitions):
                raise AliasRelationCourseError(
                    "relation Core 历史包含当前 manifest 未声明的候选")
            candidate_runtime.preflight_register_many(tuple(
                (
                    entry.spec.candidate_definition(manifest.relation_protocol),
                    entry.timestamp_base,
                )
                for entry in manifest.entries
            ))
        else:
            self._reject_orphan_hypotheses(ontology)

        use_owner = None
        if use_refs is not None:
            use_owner = RelationUseOwner(
                RelationUseGraph(ontology, manifest.use_protocol),
                manifest.use_metadata,
            )
        return _PreflightCourseState(
            semantic_graph,
            candidate_graph,
            candidate_runtime,
            use_owner,
        )

    def _preflight_isolated(self) -> None:
        """在独立 DictBackend 真实预演 S-00、forming、reveal、投影和 owner 恢复。"""
        backend = DictBackend()
        try:
            ctx = make_train_context(backend)
            prepared = _PreflightCourseState(None, None, None, None)
            alias, report = self._apply(ctx, prepared)
            if report.active_count != len(self.manifest.entries):
                raise AliasRelationCourseError(
                    "relation course 未使全部声明候选进入 active 投影")
            if alias.closure.use_owner is None:
                raise AliasRelationCourseError(
                    "relation course 预演未建立 Core Use owner")
        finally:
            backend.close()

    def _apply(
            self,
            ctx: TrainContext,
            prepared: _PreflightCourseState,
            ) -> tuple[AliasRelationRuntime, AliasRelationCourseReport]:
        """在当前后端恢复 owner、物化 S-00 并重放显式逻辑序课程。"""
        manifest = self.manifest
        ontology = ctx.graph_ontology
        semantic_graph = prepared.semantic_graph
        if semantic_graph is None:
            semantic_graph = SemanticGraph(
                ontology,
                AtomicPropositionPredicates(*tuple(
                    ontology.materialize(item)
                    for item in manifest.semantic_predicates
                )),
            )
        candidate_graph = prepared.candidate_graph
        candidate_runtime = prepared.candidate_runtime
        if candidate_graph is None:
            candidate_graph = CandidateProjectionGraph(
                ontology,
                manifest.candidate_projection,
            )
        if candidate_runtime is None:
            history = ctx.training_candidate_history
            if history is None:
                raise AliasRelationCourseError(
                    "relation course 缺少 PH2 Core 训练历史")
            candidate_runtime = CandidateLearningRuntime.restore_for_training_graph(
                manifest.learning_protocol,
                candidate_graph,
                IndependentObjectVerifier(manifest.verifier_protocol),
                manifest.projection_metadata,
                history,
                manifest.training_history_protocol(),
            )
        use_owner = prepared.use_owner
        if use_owner is None:
            use_owner = RelationUseOwner(
                RelationUseGraph(ontology, manifest.use_protocol),
                manifest.use_metadata,
            )
        consumer = ActiveRelationClosureConsumer(
            semantic_graph,
            candidate_graph,
            manifest.relation_protocol,
            manifest.schemas,
            engine=candidate_runtime.engine,
        )
        closure = RelationClosureRuntime(
            candidate_runtime,
            semantic_graph,
            consumer,
            manifest.relation_protocol,
            use_owner,
        )
        for entry in manifest.entries:
            semantic_graph.define_atomic(
                entry.spec.proposition,
                scope=entry.statement_scope,
                **manifest.statement_metadata.kwargs(),
            )
        closure.form_many(tuple(
            (entry.spec, entry.timestamp_base)
            for entry in manifest.entries
        ))
        closure.recognize_many_at(tuple(
            (
                recognition.input,
                recognition.timestamp_seq,
                recognition.resolve_timestamp_seq,
                recognition.projection_timestamp_seq,
            )
            for entry in manifest.entries
            for recognition in entry.recognitions
        ))
        active = tuple(
            entry for entry in manifest.entries
            if consumer.lookup_proposition(
                entry.spec.proposition.proposition)
        )
        if len(active) != len(manifest.entries):
            raise AliasRelationCourseError(
                "relation course 未使全部声明候选进入 active 投影")
        alias = AliasRelationRuntime(
            closure,
            AliasResolutionSelector(manifest.alias_protocol),
        )
        candidate_report = candidate_runtime.report()
        report = AliasRelationCourseReport(
            manifest.sha256(),
            manifest.schema_version,
            manifest.course_version,
            len(manifest.entries),
            sum(len(item.recognitions) for item in manifest.entries),
            candidate_report.evidence_count,
            candidate_report.decision_count,
            candidate_report.projection_event_count,
            candidate_report.active_projection_count,
            len(use_owner.history()),
        )
        return alias, report

    def _reject_orphan_semantic_roots(self, ontology) -> None:
        """S-00 协议尚不存在时拒绝已物化的课程 Proposition/RoleBinding 半拓扑。"""
        for entry in self.manifest.entries:
            identities = (
                entry.spec.proposition.proposition,
                *(binding.identity_for(entry.spec.proposition.proposition)
                  for binding in entry.spec.proposition.bindings),
            )
            if any(ontology.resolve(item) is not None for item in identities):
                raise AliasRelationCourseError(
                    "relation S-00 协议缺失但课程命题已有部分拓扑")

    def _reject_orphan_hypotheses(self, ontology) -> None:
        """候选协议尚不存在时拒绝已有 Hypothesis 图对象形成伪恢复。"""
        manifest = self.manifest
        for entry in manifest.entries:
            hypothesis = entry.spec.candidate_definition(
                manifest.relation_protocol).hypothesis(
                    manifest.learning_protocol)
            if ontology.resolve(hypothesis.object_identity()) is not None:
                raise AliasRelationCourseError(
                    "relation candidate 协议缺失但 Hypothesis 已物化")

    @staticmethod
    def _resolved_group(ontology, identities, *, label: str):
        """只读恢复一组协议身份；全无返回 None，部分存在立即失败。"""
        refs = tuple(ontology.resolve(item) for item in identities)
        present = tuple(item is not None for item in refs)
        if any(present) and not all(present):
            raise AliasRelationCourseError(f"{label} 只恢复了部分协议")
        return refs if all(present) else None


__all__ = [
    "AliasRelationCourseEntry",
    "AliasRelationCourseError",
    "AliasRelationCourseLoader",
    "AliasRelationCourseManifest",
    "AliasRelationCourseRecognition",
    "AliasRelationCourseReport",
    "AliasRelationRuntimeFactory",
    "AliasRelationStatementMetadata",
    "LoadedAliasRelationCourse",
]
