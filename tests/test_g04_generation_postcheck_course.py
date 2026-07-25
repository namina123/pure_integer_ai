"""G-04 版本化课程、受限 parser 和独立 runtime owner 对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import SemanticTopologyError
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.experiments.evaluation_isolation import clone_backend
from pure_integer_ai.experiments.generation_postcheck_course import (
    GenerationPostcheckCourseComponents,
    GenerationPostcheckCourseEntry,
    GenerationPostcheckCourseError,
    GenerationPostcheckCourseLoader,
    GenerationPostcheckCourseManifest,
    GenerationPostcheckCourseMetadata,
    GenerationPostcheckCourseProtocol,
    GenerationPostcheckCourseRoute,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE

from tests.test_g04_generation_postcheck import (
    _ExecutionParser,
    _ProductionPostcheckMapper,
    _StaticVerifier,
    _observation,
    _protocol as _postcheck_protocol,
    _source_requirements,
)
from tests.test_l05b2_typed_production_generation import _production_fixture


_BASE = 18100


def _source(variant: int) -> SourceRef:
    """构造来源化课程命题使用的稳定测试来源。"""
    return SourceRef(
        _BASE + 1,
        _BASE + 2,
        variant,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _course_protocol(variant: int) -> GenerationPostcheckCourseProtocol:
    """构造 relation、schema 和七个开放 Role 的课程协议。"""
    roles = tuple(
        role_identity((_BASE + variant, ordinal))
        for ordinal in range(1, 9)
    )
    return GenerationPostcheckCourseProtocol(
        concept_identity((_BASE + 100, variant)),
        structure_concept_identity((_BASE + 101, variant)),
        *roles,
    )


def _route(variant: int) -> GenerationPostcheckCourseRoute:
    """构造一个 LanguageBranch 及互不复用的四类最小执行组件。"""
    return GenerationPostcheckCourseRoute(
        language_branch_identity((_BASE + 200, variant)),
        minimal_instruction_identity((_BASE + 201, variant)),
        minimal_instruction_identity((_BASE + 202, variant)),
        minimal_instruction_identity((_BASE + 203, variant)),
        minimal_instruction_identity((_BASE + 204, variant)),
    )


def _schema(
        protocol: GenerationPostcheckCourseProtocol,
        ) -> RelationSchema:
    """把 branch 和必选/可选组件 Role 编成开放 n 元关系 schema。"""
    slots = (
        RelationSlotSchema(
            protocol.schema_role,
            frozenset({protocol.schema.object_kind}),
            1,
            1,
        ),
        RelationSlotSchema(
            protocol.branch_role,
            frozenset({OBJECT_LANGUAGE_BRANCH}),
            1,
            1,
        ),
        *(RelationSlotSchema(
            role,
            frozenset({OBJECT_MINIMAL_INSTRUCTION}),
            1,
            1,
        ) for role in (
            protocol.mapper_role,
            protocol.parser_role,
            protocol.structure_verifier_role,
            protocol.source_verifier_role,
        )),
        RelationSlotSchema(
            protocol.artifact_verifier_role,
            frozenset({OBJECT_MINIMAL_INSTRUCTION}),
            0,
            1,
        ),
        RelationSlotSchema(
            protocol.task_verifier_role,
            frozenset({OBJECT_MINIMAL_INSTRUCTION}),
            0,
            1,
        ),
    )
    return RelationSchema(protocol.schema, protocol.relation, slots)


def _entry(
        protocol: GenerationPostcheckCourseProtocol,
        route: GenerationPostcheckCourseRoute,
        *,
        variant: int,
        ) -> GenerationPostcheckCourseEntry:
    """把一条来源化 Proposition 的 RoleBinding 与组件路由逐点对齐。"""
    source = _source(variant)
    bindings = (
        AtomicRoleBinding(protocol.schema_role, protocol.schema, 0),
        AtomicRoleBinding(protocol.branch_role, route.branch, 0),
        AtomicRoleBinding(protocol.mapper_role, route.mapper, 0),
        AtomicRoleBinding(protocol.parser_role, route.parser, 0),
        AtomicRoleBinding(
            protocol.structure_verifier_role,
            route.structure_verifier,
            0,
        ),
        AtomicRoleBinding(
            protocol.source_verifier_role,
            route.source_verifier,
            0,
        ),
        *(()
          if route.artifact_verifier is None
          else (AtomicRoleBinding(
              protocol.artifact_verifier_role,
              route.artifact_verifier,
              0,
          ),)),
        *(()
          if route.task_verifier is None
          else (AtomicRoleBinding(
              protocol.task_verifier_role,
              route.task_verifier,
              0,
          ),)),
    )
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (_BASE + 300, variant)),
        protocol.relation,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (_BASE + 301, variant)),
        bindings,
    )
    return GenerationPostcheckCourseEntry(
        definition,
        document_scope(source),
        route,
    )


class _ComponentFactory:
    """按 manifest 路由为每个 context 新建测试 mapper/parser/verifier。"""

    def __init__(
            self,
            routes: tuple[GenerationPostcheckCourseRoute, ...],
            variant: int,
            ) -> None:
        self.routes = routes
        self.variant = variant
        self.drift = False
        self.components = []

    def build(self, _ctx):
        """建立共享一次 fixture 登记、但不跨 context 共享状态的 G-04 组件。"""
        parser = _ExecutionParser()
        components = GenerationPostcheckCourseComponents(
            self.routes,
            _ProductionPostcheckMapper(parser=parser),
            parser,
            _StaticVerifier(VERDICT_SUPPORT, 1),
            _StaticVerifier(VERDICT_SUPPORT, 2),
        )
        self.components.append(components)
        return components

    def clone_for_evaluation(self):
        """复制不可变路由和配置，清空已建立的运行组件。"""
        return _ComponentFactory(self.routes, self.variant)

    def state_key(self):
        """返回被 manifest 锁定的组件配置键，并允许测试注入漂移。"""
        return (
            _BASE + 400,
            self.variant,
            1 if self.drift else 0,
        )


def _manifest_and_factory(
        variant: int = 1,
        *,
        factory_routes: tuple[GenerationPostcheckCourseRoute, ...] | None = None,
        ) -> tuple[GenerationPostcheckCourseManifest, _ComponentFactory]:
    """构造内容锁覆盖完整协议、schema、路由和 factory 配置的课程。"""
    protocol = _course_protocol(variant)
    route = _route(variant)
    routes = (route,) if factory_routes is None else factory_routes
    factory = _ComponentFactory(routes, variant)
    manifest = GenerationPostcheckCourseManifest(
        1,
        (_BASE + 500, variant),
        tuple(
            concept_identity((_BASE + 600 + variant, ordinal))
            for ordinal in range(1, 7)
        ),
        protocol,
        _schema(protocol),
        _postcheck_protocol(),
        factory.state_key(),
        GenerationPostcheckCourseMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
            content_version=variant,
        ),
        (_entry(protocol, route, variant=variant),),
    )
    return manifest, factory


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_postcheck_course_is_idempotent_and_memory_free(backend_type):
    """同一课程在 Dict/SQLite 上零增长恢复，且只写 Core 图不写 Memory。"""
    backend = backend_type()
    try:
        manifest, factory = _manifest_and_factory(1)
        loader = GenerationPostcheckCourseLoader(
            manifest, manifest.sha256(), factory)
        ctx = make_train_context(backend)

        first = loader.load(ctx)
        snapshot = backend.snapshot()
        second = loader.load(make_train_context(backend))
        binding = second.factory.build(ctx)

        assert backend.snapshot() == snapshot
        assert first.report == second.report
        assert first.report.manifest_sha256 == manifest.sha256()
        assert first.report.route_count == 1
        assert first.factory.branches() == (manifest.entries[0].route.branch,)
        assert ctx.graph_ontology.resolve(
            manifest.course_protocol.schema) is not None
        assert isinstance(binding.runtime, GenerationPostcheckRuntime)
        assert backend.count(MEMORY_EVENT_TABLE) == 0
    finally:
        backend.close()


def test_postcheck_course_rejects_hash_and_factory_drift_before_host_write():
    """内容锁或组件配置漂移必须在宿主课程首写前失败。"""
    backend = DictBackend()
    try:
        manifest, factory = _manifest_and_factory(2)
        ctx = make_train_context(backend)
        baseline = backend.snapshot()
        changed = replace(
            manifest,
            course_version=(_BASE + 500, 2, 1),
        )
        with pytest.raises(GenerationPostcheckCourseError, match="内容哈希漂移"):
            GenerationPostcheckCourseLoader(
                changed, manifest.sha256(), factory).load(ctx)
        assert backend.snapshot() == baseline

        factory.drift = True
        with pytest.raises(GenerationPostcheckCourseError, match="factory key 漂移"):
            GenerationPostcheckCourseLoader(
                manifest, manifest.sha256(), factory).load(ctx)
        assert backend.snapshot() == baseline
    finally:
        backend.close()


def test_postcheck_course_rejects_partial_s00_and_partial_proposition():
    """部分协议或部分命题拓扑不得被 loader 静默补成看似完整课程。"""
    partial_s00 = DictBackend()
    try:
        manifest, factory = _manifest_and_factory(3)
        ctx = make_train_context(partial_s00)
        ctx.graph_ontology.materialize(manifest.semantic_predicates[0])
        baseline = partial_s00.snapshot()
        with pytest.raises(GenerationPostcheckCourseError, match="部分 S-00"):
            GenerationPostcheckCourseLoader(
                manifest, manifest.sha256(), factory).load(ctx)
        assert partial_s00.snapshot() == baseline
    finally:
        partial_s00.close()

    partial_entry = DictBackend()
    try:
        manifest, factory = _manifest_and_factory(4)
        ctx = make_train_context(partial_entry)
        refs = tuple(
            ctx.graph_ontology.materialize(item)
            for item in manifest.semantic_predicates
        )
        entry = manifest.entries[0]
        proposition_ref = ctx.graph_ontology.materialize(
            entry.proposition.proposition)
        relation_ref = ctx.graph_ontology.materialize(
            entry.proposition.predicate)
        ctx.graph_ontology.relate(
            refs[0],
            proposition_ref,
            relation_ref,
            scope=entry.statement_scope,
            **manifest.statement_metadata.kwargs(),
        )
        baseline = partial_entry.snapshot()
        with pytest.raises(SemanticTopologyError):
            GenerationPostcheckCourseLoader(
                manifest, manifest.sha256(), factory).load(ctx)
        assert partial_entry.snapshot() == baseline
    finally:
        partial_entry.close()


def test_postcheck_course_rejects_component_routes_not_declared_by_graph():
    """组件 factory 不得以相同配置键替换 manifest 声明的 LanguageBranch 路由。"""
    manifest, _ = _manifest_and_factory(5)
    wrong_factory = _ComponentFactory((_route(6),), 5)
    changed = replace(manifest, component_factory_key=wrong_factory.state_key())
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        baseline = backend.snapshot()
        with pytest.raises(GenerationPostcheckCourseError, match="组件路由"):
            GenerationPostcheckCourseLoader(
                changed, changed.sha256(), wrong_factory).load(ctx)
        assert backend.snapshot() == baseline
    finally:
        backend.close()


def test_postcheck_course_rejects_mixed_optional_verifier_routes():
    """单 runtime 不能把 artifact/task verifier 伪装成只适用于部分分支。"""
    protocol = _course_protocol(6)
    first = _route(6)
    second = replace(
        _route(60),
        artifact_verifier=minimal_instruction_identity((_BASE + 205, 60)),
    )
    entries = (
        _entry(protocol, first, variant=61),
        _entry(protocol, second, variant=62),
    )
    with pytest.raises(ValueError, match="artifact verifier 声明必须一致"):
        GenerationPostcheckCourseManifest(
            1,
            (_BASE + 500, 6),
            tuple(
                concept_identity((_BASE + 606, ordinal))
                for ordinal in range(1, 7)
            ),
            protocol,
            _schema(protocol),
            _postcheck_protocol(),
            (_BASE + 400, 6, 0),
            GenerationPostcheckCourseMetadata(
                SOURCE_BARE_TEXT,
                EPI_STRUCTURED,
                content_version=6,
            ),
            entries,
        )


def test_postcheck_runtime_factory_restores_graph_and_clones_independent_owners():
    """runtime 只能从完整课程图恢复，V-06 clone 不共享 parser/verifier 状态。"""
    backend = DictBackend()
    try:
        manifest, factory = _manifest_and_factory(7)
        ctx = make_train_context(backend)
        loaded = GenerationPostcheckCourseLoader(
            manifest, manifest.sha256(), factory).load(ctx)
        host = loaded.factory.build(ctx)

        cloned_backend = clone_backend(backend)
        try:
            cloned_ctx = make_train_context(cloned_backend)
            cloned_factory = loaded.factory.clone_for_evaluation()
            cloned = cloned_factory.build(cloned_ctx)
            assert cloned_factory.state_key() == loaded.factory.state_key()
            assert cloned.mapper is not host.mapper
            assert cloned.runtime is not host.runtime
            assert cloned.runtime.parser is not host.runtime.parser
            assert cloned.runtime.structure_verifier is not (
                host.runtime.structure_verifier)
            assert cloned.runtime.source_verifier is not (
                host.runtime.source_verifier)
        finally:
            cloned_backend.close()

        empty_backend = DictBackend()
        try:
            with pytest.raises(GenerationPostcheckCourseError, match="S-00"):
                loaded.factory.build(make_train_context(empty_backend))
        finally:
            empty_backend.close()
    finally:
        backend.close()


def test_surface_parser_request_exposes_no_plan_or_representation_answer():
    """独立 parser 只能读取实际 units 和运行上下文，不能读取计划稳定键答案。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)
        request = GenerationSurfaceParseRequest.from_execution(execution)
        assert set(request.__dataclass_fields__) == {
            "renderer", "units", "branch", "source", "scope",
        }
        for forbidden in (
                "execution", "execution_key", "rendered_key", "plan",
                "representations", "propositions", "structure_payload"):
            assert not hasattr(request, forbidden)

        parser = _ExecutionParser()
        parser.record(execution)
        runtime = GenerationPostcheckRuntime(
            _postcheck_protocol(),
            parser,
            _StaticVerifier(VERDICT_SUPPORT, 1),
            _StaticVerifier(VERDICT_SUPPORT, 2),
        )
        run = runtime.run(GenerationPostcheckRequest(
            execution,
            (),
            _source_requirements(execution),
        ))
        assert run.parsed.observation == _observation(execution)
    finally:
        fixture.close()
