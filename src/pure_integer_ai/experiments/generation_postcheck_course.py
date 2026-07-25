"""版本化加载 G-04 parser/verifier 课程并重建独立 runtime owner。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Protocol

from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    FormalArtifactVerifier,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParser,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
)
from pure_integer_ai.cognition.shared.typed_relation import RelationSchema
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationPostcheckMapper,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
    GenerationPostcheckRuntime,
    GenerationSourceVerifier,
    GenerationStructureVerifier,
    GenerationTaskVerifier,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend


_COURSE_SCHEMA_VERSION = 1


class GenerationPostcheckCourseError(RuntimeError):
    """G-04 课程版本、内容锁、图声明或组件工厂不一致。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验版本、工厂和限定键使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _require_kind(
        value: ObjectIdentity,
        kind: int,
        *,
        label: str,
        ) -> ObjectIdentity:
    """核验课程图端点使用调用方声明的一等对象种类。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != kind:
        raise ValueError(f"{label} 对象种类错误")
    return value


def _schema_key(schema: RelationSchema) -> tuple[int, ...]:
    """编码 G-04 课程 relation schema 的全部开放 Role 和基数。"""
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


def _proposition_key(
        definition: AtomicPropositionDefinition,
        ) -> tuple[int, ...]:
    """编码 G-04 课程 Proposition、anchor、context 和全部 RoleBinding。"""
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
    return tuple(result)


@dataclass(frozen=True)
class GenerationPostcheckCourseProtocol:
    """声明 G-04 课程 relation、schema 及 schema/branch/组件 Role。"""

    relation: ObjectIdentity
    schema: ObjectIdentity
    schema_role: ObjectIdentity
    branch_role: ObjectIdentity
    mapper_role: ObjectIdentity
    parser_role: ObjectIdentity
    structure_verifier_role: ObjectIdentity
    source_verifier_role: ObjectIdentity
    artifact_verifier_role: ObjectIdentity
    task_verifier_role: ObjectIdentity

    def __post_init__(self) -> None:
        """核验 relation/schema/Role 分型且全部身份互不混用。"""
        _require_kind(self.relation, OBJECT_CONCEPT, label="G-04 course relation")
        _require_kind(
            self.schema, OBJECT_STRUCTURE_CONCEPT, label="G-04 course schema")
        roles = self.roles()
        if len(set(roles)) != len(roles):
            raise ValueError("G-04 course Role 必须互不相同")
        for role in roles:
            _require_kind(role, OBJECT_ROLE, label="G-04 course Role")

    def roles(self) -> tuple[ObjectIdentity, ...]:
        """返回 schema、branch、mapper、parser 和四类 verifier Role。"""
        return (
            self.schema_role,
            self.branch_role,
            self.mapper_role,
            self.parser_role,
            self.structure_verifier_role,
            self.source_verifier_role,
            self.artifact_verifier_role,
            self.task_verifier_role,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、schema 和全部 Role 的完整身份键。"""
        return tuple(
            value
            for identity in (self.relation, self.schema, *self.roles())
            for value in _packed(identity.stable_key())
        )


@dataclass(frozen=True)
class GenerationPostcheckCourseRoute:
    """把一个 LanguageBranch 绑定到独立 mapper/parser/verifier 标识。"""

    branch: ObjectIdentity
    mapper: ObjectIdentity
    parser: ObjectIdentity
    structure_verifier: ObjectIdentity
    source_verifier: ObjectIdentity
    artifact_verifier: ObjectIdentity | None = None
    task_verifier: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验分支和最小指令标识，禁止同一路由组件身份自证复用。"""
        _require_kind(
            self.branch, OBJECT_LANGUAGE_BRANCH, label="G-04 course branch")
        instructions = self.instructions()
        if len(set(instructions)) != len(instructions):
            raise ValueError("G-04 mapper/parser/verifier 必须使用不同最小指令")
        for instruction in instructions:
            _require_kind(
                instruction,
                OBJECT_MINIMAL_INSTRUCTION,
                label="G-04 course component",
            )

    def instructions(self) -> tuple[ObjectIdentity, ...]:
        """返回当前路由实际声明的全部最小执行指令。"""
        return tuple(item for item in (
            self.mapper,
            self.parser,
            self.structure_verifier,
            self.source_verifier,
            self.artifact_verifier,
            self.task_verifier,
        ) if item is not None)

    def stable_key(self) -> tuple[int, ...]:
        """返回分支和必选/可选组件身份的完整键。"""
        result = [*_packed(self.branch.stable_key())]
        for identity in (
                self.mapper,
                self.parser,
                self.structure_verifier,
                self.source_verifier,
                self.artifact_verifier,
                self.task_verifier):
            result.append(0 if identity is None else 1)
            if identity is not None:
                result.extend(_packed(identity.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class GenerationPostcheckCourseEntry:
    """把一条来源化 S-00 课程命题绑定到 G-04 分支组件路由。"""

    proposition: AtomicPropositionDefinition
    statement_scope: ScopeIdentity
    route: GenerationPostcheckCourseRoute

    def __post_init__(self) -> None:
        """核验命题来源 scope 和显式 route 类型。"""
        if not isinstance(self.proposition, AtomicPropositionDefinition):
            raise TypeError("G-04 course proposition 类型错误")
        if not isinstance(self.statement_scope, ScopeIdentity):
            raise TypeError("G-04 course statement_scope 类型错误")
        if self.statement_scope.source != self.proposition.source:
            raise ValueError("G-04 course scope 必须绑定 Proposition 来源")
        if not isinstance(self.route, GenerationPostcheckCourseRoute):
            raise TypeError("G-04 course route 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回命题、写入 scope 和组件路由完整键。"""
        return (
            *_packed(_proposition_key(self.proposition)),
            *_packed(self.statement_scope.stable_key()),
            *_packed(self.route.stable_key()),
        )


@dataclass(frozen=True)
class GenerationPostcheckCourseMetadata:
    """保存 G-04 课程图 statement 的来源元数据。"""

    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验来源元数据全部为严格整数且 provenance 为正。"""
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("G-04 course qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="GenerationPostcheckCourseMetadata",
        )
        if (type(self.provenance_kind) is not int
                or self.provenance_kind <= 0
                or type(self.epistemic_origin) is not int
                or self.epistemic_origin < 0
                or type(self.content_version) is not int
                or self.content_version < 0
                or any(type(item) is not int for item in self.qualifiers)):
            raise ValueError("G-04 course 来源元数据非法")

    def kwargs(self) -> dict:
        """返回 SemanticGraph 写入接口接受的统一关键字参数。"""
        return {
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifiers": self.qualifiers,
        }

    def stable_key(self) -> tuple[int, ...]:
        """返回来源类型、认识论来源、版本和限定键。"""
        return (
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *_packed(self.qualifiers),
        )


@dataclass(frozen=True)
class GenerationPostcheckCourseManifest:
    """保存版本、S-00 协议、G-04 路由理论和组件工厂配置键。"""

    schema_version: int
    course_version: tuple[int, ...]
    semantic_predicates: tuple[ObjectIdentity, ...]
    course_protocol: GenerationPostcheckCourseProtocol
    relation_schema: RelationSchema
    postcheck_protocol: GenerationPostcheckProtocol
    component_factory_key: tuple[int, ...]
    statement_metadata: GenerationPostcheckCourseMetadata
    entries: tuple[GenerationPostcheckCourseEntry, ...]

    def __post_init__(self) -> None:
        """核验 schema、课程路由、图命题和组件工厂键完整一致。"""
        assert_int(self.schema_version, _where="G-04 course schema")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("G-04 course schema_version 必须为严格正整数")
        _strict_key(self.course_version, label="G-04 course version")
        _strict_key(
            self.component_factory_key, label="G-04 component factory key")
        if (not isinstance(self.semantic_predicates, tuple)
                or len(self.semantic_predicates) != 6
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_CONCEPT
                       for item in self.semantic_predicates)):
            raise ValueError("G-04 course 必须声明六个 S-00 Concept predicate")
        if len(set(self.semantic_predicates)) != 6:
            raise ValueError("G-04 course S-00 predicate 不得重复")
        for label, value, expected in (
                ("course protocol", self.course_protocol,
                 GenerationPostcheckCourseProtocol),
                ("relation schema", self.relation_schema, RelationSchema),
                ("postcheck protocol", self.postcheck_protocol,
                 GenerationPostcheckProtocol),
                ("statement metadata", self.statement_metadata,
                 GenerationPostcheckCourseMetadata)):
            if not isinstance(value, expected):
                raise TypeError(f"G-04 course {label} 类型错误")
        if (self.relation_schema.schema != self.course_protocol.schema
                or self.relation_schema.relation
                != self.course_protocol.relation):
            raise ValueError("G-04 course schema 与课程协议不一致")
        if (not isinstance(self.entries, tuple) or not self.entries
                or any(not isinstance(item, GenerationPostcheckCourseEntry)
                       for item in self.entries)):
            raise TypeError("G-04 course entries 必须是非空课程 tuple")
        propositions = tuple(item.proposition.proposition for item in self.entries)
        branches = tuple(item.route.branch for item in self.entries)
        if len(set(propositions)) != len(propositions):
            raise ValueError("G-04 course Proposition 不得重复")
        if len(set(branches)) != len(branches):
            raise ValueError("G-04 course 同一 LanguageBranch 不得重复路由")
        for entry in self.entries:
            self._validate_entry(entry)
        for label, values in (
                ("artifact", tuple(
                    item.route.artifact_verifier is not None
                    for item in self.entries)),
                ("task", tuple(
                    item.route.task_verifier is not None
                    for item in self.entries))):
            if len(set(values)) != 1:
                raise ValueError(
                    f"G-04 course 多分支 {label} verifier 声明必须一致")

    def _validate_entry(self, entry: GenerationPostcheckCourseEntry) -> None:
        """核验一条课程命题的 RoleBinding 与显式组件路由逐点相同。"""
        definition = self.relation_schema.validate_definition(entry.proposition)
        if definition.predicate != self.course_protocol.relation:
            raise ValueError("G-04 course Proposition 使用了其他 relation")
        by_role = {binding.role: binding.filler for binding in definition.bindings}
        route = entry.route
        expected = {
            self.course_protocol.schema_role: self.course_protocol.schema,
            self.course_protocol.branch_role: route.branch,
            self.course_protocol.mapper_role: route.mapper,
            self.course_protocol.parser_role: route.parser,
            self.course_protocol.structure_verifier_role: route.structure_verifier,
            self.course_protocol.source_verifier_role: route.source_verifier,
        }
        if route.artifact_verifier is not None:
            expected[self.course_protocol.artifact_verifier_role] = (
                route.artifact_verifier)
        if route.task_verifier is not None:
            expected[self.course_protocol.task_verifier_role] = route.task_verifier
        if by_role != expected:
            raise ValueError("G-04 course Proposition RoleBinding 与 route 不一致")

    def routes(self) -> tuple[GenerationPostcheckCourseRoute, ...]:
        """按 LanguageBranch 完整键返回确定序组件路由。"""
        return tuple(sorted(
            (item.route for item in self.entries),
            key=lambda item: item.branch.stable_key(),
        ))

    def stable_key(self) -> tuple[int, ...]:
        """返回 manifest 全字段规范整数键，供内容锁和漂移核验。"""
        result = [
            self.schema_version,
            *_packed(self.course_version),
            len(self.semantic_predicates),
        ]
        for predicate in self.semantic_predicates:
            result.extend(_packed(predicate.stable_key()))
        result.extend((
            *_packed(self.course_protocol.stable_key()),
            *_packed(_schema_key(self.relation_schema)),
            *_packed(self.postcheck_protocol.stable_key()),
            *_packed(self.component_factory_key),
            *_packed(self.statement_metadata.stable_key()),
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
class GenerationPostcheckCourseComponents:
    """保存与 manifest 路由对齐的 mapper、parser 和独立 verifier。"""

    routes: tuple[GenerationPostcheckCourseRoute, ...]
    mapper: ProductionGenerationPostcheckMapper
    parser: GenerationSurfaceParser
    structure_verifier: GenerationStructureVerifier
    source_verifier: GenerationSourceVerifier
    artifact_verifier: FormalArtifactVerifier | None = None
    task_verifier: GenerationTaskVerifier | None = None

    def __post_init__(self) -> None:
        """核验组件协议完整，并要求可选 verifier 与路由声明一致。"""
        if (not isinstance(self.routes, tuple) or not self.routes
                or any(not isinstance(item, GenerationPostcheckCourseRoute)
                       for item in self.routes)):
            raise TypeError("G-04 course components routes 类型错误")
        if not hasattr(self.mapper, "build"):
            raise TypeError("G-04 course mapper 必须实现 build")
        if not hasattr(self.parser, "parse"):
            raise TypeError("G-04 course parser 必须实现 parse")
        if not hasattr(self.structure_verifier, "verify"):
            raise TypeError("G-04 course structure verifier 必须实现 verify")
        if not hasattr(self.source_verifier, "verify"):
            raise TypeError("G-04 course source verifier 必须实现 verify")
        if self.artifact_verifier is not None and not hasattr(
                self.artifact_verifier, "verify"):
            raise TypeError("G-04 course artifact verifier 必须实现 verify")
        if self.task_verifier is not None and not hasattr(
                self.task_verifier, "verify"):
            raise TypeError("G-04 course task verifier 必须实现 verify")
        artifact_flags = {
            item.artifact_verifier is not None for item in self.routes}
        task_flags = {item.task_verifier is not None for item in self.routes}
        if len(artifact_flags) != 1 or len(task_flags) != 1:
            raise ValueError("G-04 components 多分支可选 verifier 声明必须一致")
        requires_artifact = True in artifact_flags
        requires_task = True in task_flags
        if requires_artifact != (self.artifact_verifier is not None):
            raise ValueError("G-04 artifact verifier 实例与课程路由不一致")
        if requires_task != (self.task_verifier is not None):
            raise ValueError("G-04 task verifier 实例与课程路由不一致")


class GenerationPostcheckCourseComponentFactory(Protocol):
    """为宿主或 V-06 context 重建不共享状态的 G-04 组件。"""

    def build(
            self,
            ctx: TrainContext,
            ) -> GenerationPostcheckCourseComponents:
        """返回绑定当前上下文的 mapper、受限 parser 和独立 verifier。"""
        ...

    def clone_for_evaluation(
            self,
            ) -> "GenerationPostcheckCourseComponentFactory":
        """复制不可变配置并清空 parser/verifier 调用状态。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回被 manifest 内容锁覆盖的纯整数组件配置键。"""
        ...


@dataclass(frozen=True)
class GenerationPostcheckRuntimeBinding:
    """返回 production postcheck mapper 与同一课程 runtime。"""

    mapper: ProductionGenerationPostcheckMapper
    runtime: GenerationPostcheckRuntime

    def __post_init__(self) -> None:
        """核验 production 绑定同时具备 mapper 和 G-04 runtime。"""
        if not hasattr(self.mapper, "build"):
            raise TypeError("G-04 runtime binding mapper 协议不完整")
        if not isinstance(self.runtime, GenerationPostcheckRuntime):
            raise TypeError("G-04 runtime binding runtime 类型错误")


@dataclass(frozen=True)
class GenerationPostcheckCourseReport:
    """记录一次 G-04 课程装配的内容锁、版本和分支数。"""

    manifest_sha256: str
    schema_version: int
    course_version: tuple[int, ...]
    route_count: int


@dataclass(frozen=True)
class LoadedGenerationPostcheckCourse:
    """返回可供默认 builder 和 V-06 使用的 runtime factory 与报告。"""

    factory: "GenerationPostcheckRuntimeFactory"
    report: GenerationPostcheckCourseReport


class GenerationPostcheckRuntimeFactory:
    """从已物化课程图为当前 context 重建独立 G-04 owner。"""

    def __init__(
            self,
            manifest: GenerationPostcheckCourseManifest,
            expected_sha256: str,
            component_factory: GenerationPostcheckCourseComponentFactory,
            ) -> None:
        """绑定不可变课程和可克隆组件工厂，不保存运行期组件。"""
        self.manifest = manifest
        self.expected_sha256 = expected_sha256
        self._component_factory = component_factory

    def build(self, ctx: TrainContext) -> GenerationPostcheckRuntimeBinding:
        """只读恢复当前图课程声明并建立 context 独占 parser/verifier。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("G-04 runtime factory ctx 类型错误")
        self._restore_graph(ctx)
        components = self._component_factory.build(ctx)
        self._validate_components(components)
        return GenerationPostcheckRuntimeBinding(
            components.mapper,
            GenerationPostcheckRuntime(
                self.manifest.postcheck_protocol,
                components.parser,
                components.structure_verifier,
                components.source_verifier,
                artifact_verifier=components.artifact_verifier,
                task_verifier=components.task_verifier,
            ),
        )

    def clone_for_evaluation(self) -> "GenerationPostcheckRuntimeFactory":
        """复制课程配置并克隆组件工厂，禁止共享 parser/verifier 状态。"""
        return GenerationPostcheckRuntimeFactory(
            self.manifest,
            self.expected_sha256,
            self._component_factory.clone_for_evaluation(),
        )

    def branches(self) -> tuple[ObjectIdentity, ...]:
        """返回课程显式覆盖的全部 LanguageBranch，供跨课程装配预检。"""
        return tuple(route.branch for route in self.manifest.routes())

    def state_key(self) -> tuple:
        """返回内容锁、manifest 和组件工厂配置，不含可变调用状态。"""
        return (
            self.expected_sha256,
            self.manifest.stable_key(),
            self._component_factory.state_key(),
        )

    def _restore_graph(self, ctx: TrainContext) -> None:
        """从当前 Core 图严格恢复全部课程 Proposition，缺失或漂移即失败。"""
        refs = tuple(
            ctx.graph_ontology.resolve(item)
            for item in self.manifest.semantic_predicates)
        if not all(item is not None for item in refs):
            raise GenerationPostcheckCourseError("G-04 course S-00 协议未完整恢复")
        graph = SemanticGraph(
            ctx.graph_ontology,
            AtomicPropositionPredicates(*refs),
        )
        for entry in self.manifest.entries:
            restored = graph.preflight_atomic(
                entry.proposition,
                scope=entry.statement_scope,
                **self.manifest.statement_metadata.kwargs(),
            )
            if restored is None:
                raise GenerationPostcheckCourseError("G-04 course Proposition 未恢复")

    def _validate_components(
            self,
            components: GenerationPostcheckCourseComponents,
            ) -> None:
        """核验组件路由和 factory state 未偏离 manifest 内容锁。"""
        if not isinstance(components, GenerationPostcheckCourseComponents):
            raise TypeError("G-04 component factory 返回类型错误")
        if tuple(sorted(
                components.routes,
                key=lambda item: item.branch.stable_key(),
                )) != self.manifest.routes():
            raise GenerationPostcheckCourseError("G-04 组件路由与课程 manifest 不一致")
        if self._component_factory.state_key() != (
                self.manifest.component_factory_key):
            raise GenerationPostcheckCourseError("G-04 组件 factory 配置发生漂移")


class GenerationPostcheckCourseLoader:
    """在正式 Core 首写前预演并加载内容锁定的 G-04 课程。"""

    def __init__(
            self,
            manifest: GenerationPostcheckCourseManifest,
            expected_sha256: str,
            component_factory: GenerationPostcheckCourseComponentFactory,
            ) -> None:
        """绑定 manifest、预期哈希和可克隆组件工厂。"""
        if not isinstance(manifest, GenerationPostcheckCourseManifest):
            raise TypeError("G-04 course manifest 类型错误")
        digest = expected_sha256.lower()
        if (len(digest) != 64
                or any(item not in "0123456789abcdef" for item in digest)):
            raise ValueError("G-04 course expected_sha256 格式错误")
        if any(not hasattr(component_factory, method) for method in (
                "build", "clone_for_evaluation", "state_key")):
            raise TypeError("G-04 component factory 协议不完整")
        self.manifest = manifest
        self.expected_sha256 = digest
        self.component_factory = component_factory

    def load(self, ctx: TrainContext) -> LoadedGenerationPostcheckCourse:
        """核验内容锁，经隔离组件/图预演后幂等物化课程 Proposition。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("G-04 course ctx 类型错误")
        if self.manifest.sha256() != self.expected_sha256:
            raise GenerationPostcheckCourseError("G-04 course 内容哈希漂移")
        if self.manifest.schema_version != _COURSE_SCHEMA_VERSION:
            raise GenerationPostcheckCourseError("G-04 course schema 版本不受支持")
        if self.component_factory.state_key() != (
                self.manifest.component_factory_key):
            raise GenerationPostcheckCourseError("G-04 component factory key 漂移")
        graph = self._preflight_host(ctx)
        self._preflight_isolated()
        self._apply(ctx, graph)
        return LoadedGenerationPostcheckCourse(
            GenerationPostcheckRuntimeFactory(
                self.manifest,
                self.expected_sha256,
                self.component_factory,
            ),
            GenerationPostcheckCourseReport(
                self.manifest.sha256(),
                self.manifest.schema_version,
                self.manifest.course_version,
                len(self.manifest.entries),
            ),
        )

    def state_key(self) -> tuple:
        """返回内容锁、manifest 和组件工厂配置键。"""
        return (
            self.expected_sha256,
            self.manifest.stable_key(),
            self.component_factory.state_key(),
        )

    def _preflight_host(self, ctx: TrainContext) -> SemanticGraph | None:
        """只读核验宿主 S-00 协议和既有课程命题是否完整一致。"""
        ontology = ctx.graph_ontology
        refs = tuple(
            ontology.resolve(item) for item in self.manifest.semantic_predicates)
        present = tuple(item is not None for item in refs)
        if any(present) and not all(present):
            raise GenerationPostcheckCourseError("G-04 course 只恢复了部分 S-00 协议")
        if not any(present):
            self._reject_orphan_entries(ontology)
            return None
        graph = SemanticGraph(
            ontology,
            AtomicPropositionPredicates(*refs),
        )
        for entry in self.manifest.entries:
            graph.preflight_atomic(
                entry.proposition,
                scope=entry.statement_scope,
                **self.manifest.statement_metadata.kwargs(),
            )
        return graph

    def _preflight_isolated(self) -> None:
        """在独立 DictBackend 预演课程图和克隆组件工厂，不污染宿主。"""
        backend = DictBackend()
        try:
            ctx = make_train_context(backend)
            graph = SemanticGraph(
                ctx.graph_ontology,
                AtomicPropositionPredicates(*tuple(
                    ctx.graph_ontology.materialize(item)
                    for item in self.manifest.semantic_predicates
                )),
            )
            self._apply(ctx, graph)
            factory = GenerationPostcheckRuntimeFactory(
                self.manifest,
                self.expected_sha256,
                self.component_factory.clone_for_evaluation(),
            )
            factory.build(ctx)
        finally:
            backend.close()

    def _apply(
            self,
            ctx: TrainContext,
            graph: SemanticGraph | None,
            ) -> None:
        """在当前 Core 图幂等写入全部来源化 G-04 课程 Proposition。"""
        if graph is None:
            graph = SemanticGraph(
                ctx.graph_ontology,
                AtomicPropositionPredicates(*tuple(
                    ctx.graph_ontology.materialize(item)
                    for item in self.manifest.semantic_predicates
                )),
            )
        for entry in self.manifest.entries:
            graph.define_atomic(
                entry.proposition,
                scope=entry.statement_scope,
                **self.manifest.statement_metadata.kwargs(),
            )

    def _reject_orphan_entries(self, ontology) -> None:
        """S-00 协议不存在时拒绝已物化的课程 Proposition/RoleBinding 半拓扑。"""
        for entry in self.manifest.entries:
            identities = (
                entry.proposition.proposition,
                *(binding.identity_for(entry.proposition.proposition)
                  for binding in entry.proposition.bindings),
            )
            if any(ontology.resolve(item) is not None for item in identities):
                raise GenerationPostcheckCourseError(
                    "G-04 course S-00 协议缺失但课程已有部分拓扑")


__all__ = [
    "GenerationPostcheckCourseComponentFactory",
    "GenerationPostcheckCourseComponents",
    "GenerationPostcheckCourseEntry",
    "GenerationPostcheckCourseError",
    "GenerationPostcheckCourseLoader",
    "GenerationPostcheckCourseManifest",
    "GenerationPostcheckCourseMetadata",
    "GenerationPostcheckCourseProtocol",
    "GenerationPostcheckCourseReport",
    "GenerationPostcheckCourseRoute",
    "GenerationPostcheckRuntimeBinding",
    "GenerationPostcheckRuntimeFactory",
    "LoadedGenerationPostcheckCourse",
]
