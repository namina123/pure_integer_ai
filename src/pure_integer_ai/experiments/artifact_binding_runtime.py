"""A-06 从真实 STRUCT_BIND 候选执行 typed Artifact 调用的运行入口。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.artifact_binding import (
    ArtifactBindingReadTrace,
    ArtifactBindingRequest,
    ArtifactBindingRun,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactArgument,
    ArtifactInvocation,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    FormalArtifactBridge,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.struct_bind_typed_adapter import (
    StructBindTypedAdapter,
    TypedStructBindCorrespondence,
    TypedStructBindEndpoint,
)
from pure_integer_ai.storage.edge_store import EdgeStore


class ArtifactBindingRuntimeError(RuntimeError):
    """A-06 reader、显式选择或 WorkMemory 提交不满足契约。"""


class ArtifactBindingRuntime:
    """重读 STRUCT_BIND、核验显式选择并把形式调用记录到当前 query。"""

    def __init__(
            self, edge_store: EdgeStore, adapter: StructBindTypedAdapter,
            bridge: FormalArtifactBridge, work_memory: WorkMemory,
            ) -> None:
        """绑定同一调用方提供的 edge reader、S-06 bridge 和 WorkMemory。"""
        if not isinstance(edge_store, EdgeStore):
            raise TypeError("edge_store 必须是 EdgeStore")
        if not isinstance(adapter, StructBindTypedAdapter):
            raise TypeError("adapter 必须是 StructBindTypedAdapter")
        if not isinstance(bridge, FormalArtifactBridge):
            raise TypeError("bridge 必须是 FormalArtifactBridge")
        if not isinstance(work_memory, WorkMemory):
            raise TypeError("work_memory 必须是 WorkMemory")
        self.edge_store = edge_store
        self.adapter = adapter
        self.bridge = bridge
        self.work_memory = work_memory

    def run(self, request: ArtifactBindingRequest) -> ArtifactBindingRun:
        """执行一次显式绑定；失败选择零调用，记录异常恢复 WorkMemory 原状态。"""
        if not isinstance(request, ArtifactBindingRequest):
            raise TypeError("request 必须是 ArtifactBindingRequest")
        if self.work_memory.active_query_scope != request.scope:
            raise ArtifactBindingRuntimeError("A-06 请求不属于当前活动 query")

        typed_endpoints = tuple(
            TypedStructBindEndpoint(item.slot_ref, item.variable)
            for item in request.endpoints)
        selected, read_traces = self._read_selected(request, typed_endpoints)
        value_by_variable = {
            item.endpoint.variable: item.artifact for item in request.values}
        choice_by_target = {
            item.target.variable: item for item in request.choices}
        selected_by_target = {
            item.target.variable: item for item in selected}
        arguments = []
        for parameter in request.definition.parameters:
            choice = choice_by_target[parameter.variable]
            correspondence = selected_by_target[parameter.variable]
            if (correspondence.source.variable != choice.source.variable
                    or correspondence.target.variable
                    != choice.target.variable):
                raise ArtifactBindingRuntimeError("A-06 候选端点与显式选择漂移")
            arguments.append(ArtifactArgument(
                parameter.variable,
                value_by_variable[choice.source.variable],
            ))
        invocation = ArtifactInvocation(
            request.proposition,
            request.definition,
            tuple(arguments),
            request.source,
            request.scope,
            request.invocation_key,
            request.expected,
        )
        result = self.bridge.invoke(invocation)
        run = ArtifactBindingRun(
            request,
            read_traces,
            invocation,
            result,
        )
        episode_before = dict(self.work_memory.episode_artifacts)
        results_before = list(self.work_memory.query_artifact_results)
        try:
            self.work_memory.record_artifact_result(result)
        except Exception:
            self.work_memory.episode_artifacts.clear()
            self.work_memory.episode_artifacts.update(episode_before)
            self.work_memory.query_artifact_results[:] = results_before
            raise
        return run

    def _read_selected(
            self, request: ArtifactBindingRequest,
            endpoints: tuple[TypedStructBindEndpoint, ...],
            ) -> tuple[
                tuple[TypedStructBindCorrespondence, ...],
                tuple[ArtifactBindingReadTrace, ...],
            ]:
        """按 source slot 各读一次并要求每个显式候选键唯一命中真实边。"""
        by_slot = {}
        traces = []
        source_slots = tuple(sorted({
            choice.source.slot_ref for choice in request.choices}))
        for source_slot in source_slots:
            read = self.adapter.read_from(
                self.edge_store, source_slot, endpoints)
            traces.append(ArtifactBindingReadTrace(
                source_slot,
                tuple(item.stable_key() for item in read.correspondences),
                tuple(item.stable_key() for item in read.failures),
            ))
            for item in read.correspondences:
                by_slot.setdefault(
                    (source_slot, item.stable_key()), []).append(item)

        selected = []
        for choice in request.choices:
            matches = by_slot.get(
                (choice.source.slot_ref, choice.correspondence_key), ())
            if len(matches) != 1:
                raise ArtifactBindingRuntimeError(
                    "A-06 显式候选必须唯一命中当前真实 STRUCT_BIND 边")
            candidate = matches[0]
            if (candidate.source.slot_ref != choice.source.slot_ref
                    or candidate.source.variable != choice.source.variable
                    or candidate.target.slot_ref != choice.target.slot_ref
                    or candidate.target.variable != choice.target.variable):
                raise ArtifactBindingRuntimeError("A-06 显式候选端点不一致")
            selected.append(candidate)
        return tuple(selected), tuple(traces)


__all__ = [
    "ArtifactBindingRuntime",
    "ArtifactBindingRuntimeError",
]
