"""L-05B2A formal typed generation 接线和逐调用禁旧链对抗测试。"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentSelector,
    ContentArtifactAttachment,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
    GenerationLayerRegistration,
    GenerationPlanner,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructurePlan,
    GenerationStructurePlanner,
)
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    language_branch_identity,
    minimal_instruction_identity,
    representation_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.cognition.shared.types import MODALITY_LANGUAGE
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
)
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationRequestDecision,
    ProductionGenerationRuntime,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES
from pure_integer_ai.training.stages import STAGE3_REWARD
from tests.boundary_fixtures import attach_boundary_fixture
from tests.test_g01_generation_content import _artifact
from tests.test_g02_generation_structure_plan import (
    _Policy,
    _plan_protocol,
)
from tests.test_g03_generation_surface import (
    _FixedLayerResolver,
    _StaticMapper,
    _StaticSurfaceBuilder,
    _alias_fixture,
    _structure_plan,
    _surface_protocol,
    _templates,
)


_BASE = 13400
_EPISODE_HASHER = Hasher("formal_train.episode_scope.v1")


@dataclass
class _RuntimeFixture:
    """保存 production runtime、预期请求、Unicode renderer 和 R-01 测试 owner。"""

    runtime: ProductionGenerationRuntime
    request: object
    renderer: UnicodeRepresentationRenderer
    alias: object
    artifacts: tuple[ContentArtifactAttachment, ...]

    def close(self) -> None:
        """关闭 R-01 测试后端。"""
        self.alias.close()


class _RequestMapper:
    """为 formal caller 注入已由测试构造的完整 GenerationPlanningRequest。"""

    def __init__(self, request, *, available: bool = True) -> None:
        self.request = request
        self.available = available
        self.calls = 0

    def build(self, ctx, item, input_payload, observation):
        """核验正式 generation scope 后返回请求或显式无请求决策。"""
        self.calls += 1
        assert ctx.work_memory.active_generation_scope is not None
        assert item.modality == MODALITY_LANGUAGE
        assert input_payload.scope_identity == ctx.work_memory.active_episode_scope
        assert observation.struct_refs
        return ProductionGenerationRequestDecision(
            minimal_instruction_identity((_BASE + 1, 1 if self.available else 2)),
            (_BASE + 2, self.calls),
            self.request if self.available else None,
        )


def _production_fixture(
        *,
        available: bool = True,
        with_artifact: bool = False,
        ) -> _RuntimeFixture:
    """建立完整 production runtime，并可选择真实采用一个 S-06 attachment。"""
    branch = language_branch_identity((_BASE + 10, 1))
    base_structure = _structure_plan(branch)
    base_request = base_structure.selection.request
    episode_id = _EPISODE_HASHER.h63((STAGE3_REWARD, 1)) or 1
    runtime_scope = query_scope(1, parent=episode_scope(
        episode_id,
        parent=document_scope(base_request.goal.source),
    ))
    request = GenerationPlanningRequest(
        replace(base_request.goal, scope=runtime_scope),
        tuple(replace(candidate, scope=runtime_scope)
              for candidate in base_request.candidates),
    )
    structure = _structure_plan(branch, request=request)
    artifacts: tuple[ContentArtifactAttachment, ...] = ()
    content_policy = _Policy(structure.selection.protocol)
    if with_artifact:
        request = structure.selection.request
        attachment = ContentArtifactAttachment(
            request.candidate_keys()[0],
            _artifact(request.candidates[0]),
        )
        artifacts = (attachment,)
        content_policy = _Policy(
            structure.selection.protocol,
            AnswerContentDecision(
                structure.selection.protocol.answer,
                minimal_instruction_identity((_BASE + 9, 1)),
                request.candidate_keys(),
                (attachment.stable_key(),),
                (_BASE + 9, 2),
            ),
        )
        selection = AnswerContentSelector(
            structure.selection.protocol,
            content_policy,
        ).select(request, artifacts)
        selection_key = selection.stable_key()
        structure = GenerationStructurePlan(
            selection,
            replace(structure.discourse, selection_key=selection_key),
            replace(structure.propositions, selection_key=selection_key),
            replace(structure.syntax, selection_key=selection_key),
        )
    first, second = _templates(structure)
    family = (_BASE + 11, 1)
    rep_first = representation_identity(family, (0x7532,))
    rep_second = representation_identity(family, (0x5E8F,))
    alias = _alias_fixture(
        branch,
        ((first, rep_first), (second, rep_second)),
    )
    planner_protocol = _plan_protocol(_BASE + 12)
    surface_protocol = _surface_protocol(_BASE + 13)
    structure_planner = GenerationStructurePlanner(
        _StaticMapper(structure.discourse),
        _StaticMapper(structure.propositions),
        _StaticMapper(structure.syntax),
    )
    surface_runtime = GenerationSurfaceRuntime(alias.runtime)
    surface_resolver = GenerationSurfaceLayerResolver(
        planner_protocol,
        AnswerContentSelector(
            structure.selection.protocol,
            content_policy,
        ),
        structure_planner,
        _StaticSurfaceBuilder(surface_protocol),
        surface_runtime,
        artifacts,
        commit=False,
    )
    payloads = (
        structure.selection.stable_key(),
        structure.selection.stable_key(),
        structure.discourse.stable_key(),
        structure.propositions.stable_key(),
        structure.syntax.stable_key(),
    )
    registrations = tuple(
        GenerationLayerRegistration(
            layer,
            _FixedLayerResolver(
                layer,
                planner_protocol,
                structure.selection.selected_candidate_keys,
                payload,
                ordinal,
            ),
        )
        for ordinal, (layer, payload) in enumerate(zip(
            planner_protocol.layers()[:5], payloads), start=1)
    ) + (GenerationLayerRegistration(
        planner_protocol.surface_layer,
        surface_resolver,
    ),)
    renderer = UnicodeRepresentationRenderer(
        family,
        minimal_instruction_identity((_BASE + 14, 1)),
    )
    executor = TypedGenerationExecutor(
        GenerationPlanner(planner_protocol, registrations),
        renderer,
        surface_runtime,
    )
    mapper = _RequestMapper(structure.selection.request, available=available)
    return _RuntimeFixture(
        ProductionGenerationRuntime(mapper, executor),
        structure.selection.request,
        renderer,
        alias,
        artifacts,
    )


def _language_item(source_ref: SourceRef | None = None) -> CollectedItem:
    """构造会触发旧多句 path 的语言项，供新分支证明其未被调用。"""
    source_kind = (
        SOURCE_BARE_TEXT if source_ref is None else source_ref.source_kind)
    return attach_boundary_fixture(CollectedItem(
        tokens=["a", "b。", "c", "d。"],
        role_seq=[1, 1, 1, 1],
        collect_type=COLLECT_PRECEDES,
        source=source_kind,
        source_ref=source_ref,
        modality=MODALITY_LANGUAGE,
    ), cut_after=(2,), source_ref=source_ref)


def _forbid_legacy(monkeypatch) -> None:
    """把所有 L-05B2A typed 分支禁止触达的 legacy runtime 设为硬失败。"""
    import pure_integer_ai.experiments.round_runtime as runtime_module

    def forbidden(*args, **kwargs):
        """任何调用都表示 typed formal round 回退了旧链。"""
        del args, kwargs
        raise AssertionError("typed production round 不得调用 legacy generation path")

    monkeypatch.setattr(runtime_module, "episode_loop", forbidden)
    monkeypatch.setattr(runtime_module, "generate_output", forbidden)
    monkeypatch.setattr(runtime_module, "_rebuild_path", forbidden)
    monkeypatch.setattr(runtime_module, "_run_emergence_hook", forbidden)


def test_typed_executor_retains_exact_surface_artifact_and_renderer_output():
    """G-00 最终层保留同次 preview，executor 渲染后提交且不得重建 surface。"""
    fixture = _production_fixture()
    try:
        execution = fixture.runtime._executor.execute(fixture.request)

        assert execution.complete
        assert execution.plan.layers[-1].artifact == execution.preview
        assert execution.plan.layers[-1].payload == execution.preview.stable_key()
        assert execution.surface.preview == execution.preview
        assert execution.representations == execution.surface.representations
        assert fixture.renderer.text(execution.rendered) == "甲序"
    finally:
        fixture.close()


def test_formal_round_renderer_failure_is_atomic_and_never_falls_back(
        monkeypatch):
    """正式 round 的 renderer 失败不得提交采用、回退旧链或泄漏运行 scope。"""
    fixture = _production_fixture()
    backend = DictBackend()

    class _FailingRenderer:
        """故意拒绝完整 Representation 序列以验证 production 原子性。"""

        def render(self, representations):
            """确认收到完整序列后抛错，不返回任何宿主输出。"""
            assert representations
            raise RuntimeError("renderer failed")

    fixture.runtime._executor._renderer = _FailingRenderer()
    try:
        ctx = make_train_context(backend)
        ctx.language_generation_runtime = fixture.runtime
        _forbid_legacy(monkeypatch)
        before_alias = fixture.alias.runtime.state_key()
        before_closure = fixture.alias.closure.state_key()

        with pytest.raises(RuntimeError, match="renderer failed"):
            DefaultRoundRunner().run_round_full(
                ctx,
                _language_item(fixture.request.goal.source),
                STAGE3_REWARD,
                1,
            )

        assert fixture.alias.runtime.state_key() == before_alias
        assert fixture.alias.closure.state_key() == before_closure
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.active_generation_scope is None
        assert ctx.work_memory.active_episode_scope is None
        assert ctx.work_memory.active_document_scope is None
    finally:
        backend.close()
        fixture.close()


@pytest.mark.parametrize("available", [True, False])
def test_formal_round_typed_branch_never_reads_or_writes_legacy_sequence(
        monkeypatch, available):
    """有请求和无请求都不得 fallback，且本调用不新增 PRECEDES 或 def_array。"""
    fixture = _production_fixture(available=available)
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ctx.language_generation_runtime = fixture.runtime
        _forbid_legacy(monkeypatch)
        before_precedes = len(backend.select(
            "edge", where={"edge_type": EDGE_PRECEDES}))
        before_def_array = tuple(backend.select("def_array", where=None))

        result = DefaultRoundRunner().run_round_full(
            ctx,
            _language_item(fixture.request.goal.source),
            STAGE3_REWARD,
            1,
        )

        assert result.episode is None
        assert result.dag_path is None
        assert result.output is not None
        assert result.output.complete is available
        assert (result.output.execution is not None) is available
        assert len(backend.select(
            "edge", where={"edge_type": EDGE_PRECEDES})) == before_precedes
        assert tuple(backend.select("def_array", where=None)) == before_def_array
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.active_generation_scope is None
    finally:
        backend.close()
        fixture.close()
