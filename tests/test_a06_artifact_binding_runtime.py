"""A-06 STRUCT_BIND 到 typed Artifact 的真实消费与失败边界测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.artifact_binding import (
    ArtifactBindingChoice,
    ArtifactBindingEndpoint,
    ArtifactBindingRequest,
    ArtifactBindingValue,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    FormalArtifactBridge,
)
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    binder_identity,
    proposition_identity,
    variable_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.shared.typed_binding import (
    ExactTypeCompatibilityResolver,
)
from pure_integer_ai.cognition.understanding.struct_bind_typed_adapter import (
    StructBindTypedAdapter,
    TypedStructBindEndpoint,
)
from pure_integer_ai.experiments.artifact_binding_runtime import (
    ArtifactBindingRuntime,
    ArtifactBindingRuntimeError,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.storage.edge_types import EDGE_STRUCT_BIND
from pure_integer_ai.storage.node_store import TIER_PRIMARY

from test_s03_typed_binding import _failures as _binding_failures
from test_s06_formal_artifact import _artifact, _bridge, _case, _scopes


def _add_edge(context, source_slot, target_slot, *, order_index: int) -> None:
    """写入一条来源化 legacy STRUCT_BIND 边，供 A-06 运行期重读。"""
    context.edge_store.add(
        space_id_from=source_slot[0],
        local_id_from=source_slot[1],
        space_id_to=target_slot[0],
        local_id_to=target_slot[1],
        edge_type=EDGE_STRUCT_BIND,
        strength=1,
        source=20601,
        tier=TIER_PRIMARY,
        epistemic_origin=EPI_STRUCTURED,
        order_index=order_index,
    )


def _start_work_memory(context, source):
    """为测试请求开启完整 session/document/episode/query 生命周期。"""
    _, document, episode, query = _scopes(source)
    session = session_scope(
        1, owner=source.owner, versions=source.versions)
    context.work_memory.begin_session(session)
    context.work_memory.begin_document(document)
    context.work_memory.begin_episode(episode)
    context.work_memory.begin_query(query)
    return session, document, episode, query


def _stop_work_memory(context) -> None:
    """按逆序关闭测试生命周期，确保 query Artifact 不泄漏。"""
    context.work_memory.end_query()
    context.work_memory.end_episode()
    context.work_memory.end_document()
    context.work_memory.end_session()


def _runtime_case(
        *, left_payload=(2, 1), right_payload=(3, 1),
        expected_payload=(5, 1),
        ):
    """组装双参数真实边、显式候选选择和 S-06 独立执行链。"""
    backend = DictBackend()
    context = make_train_context(backend)
    formal = _case(
        left_payload=left_payload,
        right_payload=right_payload,
        expected_payload=expected_payload,
    )
    _, _, _, query = _start_work_memory(context, formal["source"])
    definition = formal["definition"]
    number_type = formal["number_type"]
    source_binder = binder_identity(formal["source"], (20610, 1))
    source_variables = (
        variable_identity(source_binder, (20610, 2), number_type),
        variable_identity(source_binder, (20610, 3), number_type),
    )
    source_slots = (
        (context.space_id, 20611),
        (context.space_id, 20612),
    )
    target_slots = (
        (context.space_id, 20621),
        (context.space_id, 20622),
    )
    for ordinal, (source_slot, target_slot) in enumerate(
            zip(source_slots, target_slots), start=1):
        _add_edge(
            context, source_slot, target_slot,
            order_index=10 - ordinal,
        )

    endpoints = tuple(
        item
        for pair in zip(
            (
                ArtifactBindingEndpoint(source_slots[0], source_variables[0]),
                ArtifactBindingEndpoint(source_slots[1], source_variables[1]),
            ),
            (
                ArtifactBindingEndpoint(
                    target_slots[0], definition.parameters[0].variable),
                ArtifactBindingEndpoint(
                    target_slots[1], definition.parameters[1].variable),
            ),
        )
        for item in pair
    )
    adapter = StructBindTypedAdapter(
        ExactTypeCompatibilityResolver(), _binding_failures(20630))
    typed_endpoints = tuple(
        TypedStructBindEndpoint(item.slot_ref, item.variable)
        for item in endpoints)
    correspondences = []
    for source_slot in source_slots:
        read = adapter.read_from(
            context.edge_store, source_slot, typed_endpoints)
        assert len(read.correspondences) == 1
        correspondences.append(read.correspondences[0])
    reason = minimal_instruction_identity((20640, 1))
    choices = tuple(
        ArtifactBindingChoice(
            endpoints[index * 2],
            endpoints[index * 2 + 1],
            correspondences[index].stable_key(),
            reason,
            (20641, index + 1),
        )
        for index in range(2)
    )
    values = tuple(
        ArtifactBindingValue(
            endpoints[index * 2],
            formal["invocation"].arguments[index].value,
        )
        for index in range(2)
    )
    request = ArtifactBindingRequest(
        formal["invocation"].proposition,
        definition,
        formal["source"],
        query,
        (20642, 1),
        endpoints,
        values,
        choices,
        formal["invocation"].expected,
    )
    runtime = ArtifactBindingRuntime(
        context.edge_store,
        adapter,
        _bridge(formal),
        context.work_memory,
    )
    return backend, context, formal, runtime, request


def test_a06_consumes_real_bindings_and_records_query_artifact_use():
    """两个实际 STRUCT_BIND 候选须覆盖参数并进入 S-06 与 WorkMemory trace。"""
    backend, context, _, runtime, request = _runtime_case()
    try:
        before = backend.snapshot()
        run = runtime.run(request)

        assert run.succeeded is True
        assert run.result.value is not None
        assert run.result.value.payload == (5, 1)
        assert len(run.reads) == 2
        assert all(len(item.correspondence_keys) == 1 for item in run.reads)
        assert tuple(
            item.parameter.variable for item in run.result.bound_arguments
        ) == tuple(
            item.variable for item in request.definition.parameters)
        assert context.work_memory.query_artifact_results == [run.result]
        assert context.memory_use_runtime is None
        assert run.stable_key()
        assert backend.snapshot() == before
    finally:
        _stop_work_memory(context)
        backend.close()


def test_a06_forged_candidate_stops_before_bridge_and_work_memory(monkeypatch):
    """候选完整键不命中当前真实边时不得调用 executor 或写 WorkMemory。"""
    backend, context, _, runtime, request = _runtime_case()
    try:
        forged_choice = replace(
            request.choices[0], correspondence_key=(20650, 999))
        forged = replace(
            request, choices=(forged_choice, request.choices[1]))
        calls = []

        def forbidden_invoke(invocation):
            """记录非法调用并立即失败，证明绑定核验位于 S-06 之前。"""
            calls.append(invocation)
            raise AssertionError("伪造候选不应调用 FormalArtifactBridge")

        monkeypatch.setattr(runtime.bridge, "invoke", forbidden_invoke)
        artifacts_before = dict(context.work_memory.episode_artifacts)
        with pytest.raises(ArtifactBindingRuntimeError, match="唯一命中"):
            runtime.run(forged)
        assert calls == []
        assert context.work_memory.episode_artifacts == artifacts_before
        assert context.work_memory.query_artifact_results == []
    finally:
        _stop_work_memory(context)
        backend.close()


def test_a06_request_rejects_missing_parameter_choice():
    """显式选择未覆盖全部 definition 参数时在任何 runtime 调用前拒绝。"""
    backend, context, _, _, request = _runtime_case()
    try:
        with pytest.raises(ValueError, match="一一覆盖"):
            replace(request, choices=request.choices[:1])
        assert context.work_memory.query_artifact_results == []
    finally:
        _stop_work_memory(context)
        backend.close()


def test_a06_verifier_rejection_is_query_trace_not_memory_use():
    """独立 verifier 拒绝仍保留形式调用 trace，但不得伪造长期 Memory Use。"""
    backend, context, _, runtime, request = _runtime_case(
        expected_payload=(6, 1))
    try:
        run = runtime.run(request)
        assert run.succeeded is False
        assert run.result.verification is not None
        assert run.result.verification.accepted is False
        assert context.work_memory.query_artifact_results == [run.result]
        assert context.memory_use_runtime is None
    finally:
        _stop_work_memory(context)
        backend.close()


def test_a06_work_memory_failure_restores_partial_query_state(monkeypatch):
    """WorkMemory 记录中途异常必须恢复 episode Artifact 与 query trace。"""
    backend, context, _, runtime, request = _runtime_case()
    try:
        sentinel = concept_identity((20660, 1))
        before_artifacts = dict(context.work_memory.episode_artifacts)
        before_results = list(context.work_memory.query_artifact_results)

        def partial_failure(result):
            """模拟先写 query trace 后失败的非原子下游实现。"""
            context.work_memory.query_artifact_results.append(result)
            context.work_memory.episode_artifacts[sentinel] = (
                request.values[0].artifact)
            raise RuntimeError("work memory commit failed")

        monkeypatch.setattr(
            context.work_memory, "record_artifact_result", partial_failure)
        with pytest.raises(RuntimeError, match="commit failed"):
            runtime.run(request)
        assert context.work_memory.episode_artifacts == before_artifacts
        assert context.work_memory.query_artifact_results == before_results
    finally:
        _stop_work_memory(context)
        backend.close()


def test_a06_v06_rebuilds_query_values_without_host_write():
    """V-06 clone 保留 program 来源，但以独立 owner 重建 query 值和消费 trace。"""
    backend, context, formal, _, _ = _runtime_case()
    try:
        host_backend = backend.snapshot()
        host_artifacts = dict(context.work_memory.episode_artifacts)
        host_results = list(context.work_memory.query_artifact_results)
        definition = formal["definition"]

        with isolated_evaluation(context, label="a06-artifact-binding") as clone:
            clone.work_memory.end_session()
            eval_source = SourceRef(
                formal["source"].source_kind,
                formal["source"].source_id,
                formal["source"].document_id,
                clone.scope_owner,
                formal["source"].versions,
            )
            _, _, _, query = _start_work_memory(clone, eval_source)
            source_binder = binder_identity(eval_source, (20670, 1))
            source_variables = (
                variable_identity(
                    source_binder, (20670, 2), formal["number_type"]),
                variable_identity(
                    source_binder, (20670, 3), formal["number_type"]),
            )
            source_slots = (
                (clone.space_id, 20611),
                (clone.space_id, 20612),
            )
            target_slots = (
                (clone.space_id, 20621),
                (clone.space_id, 20622),
            )
            endpoints = (
                ArtifactBindingEndpoint(source_slots[0], source_variables[0]),
                ArtifactBindingEndpoint(
                    target_slots[0], definition.parameters[0].variable),
                ArtifactBindingEndpoint(source_slots[1], source_variables[1]),
                ArtifactBindingEndpoint(
                    target_slots[1], definition.parameters[1].variable),
            )
            values = tuple(
                ArtifactBindingValue(
                    endpoints[index * 2],
                    _artifact(
                        eval_source,
                        query,
                        argument.value.artifact_kind,
                        argument.value.schema,
                        20 + index,
                        argument.value.payload,
                    ),
                )
                for index, argument in enumerate(
                    formal["invocation"].arguments)
            )
            expected_source = formal["invocation"].expected
            assert expected_source is not None
            expected = _artifact(
                eval_source,
                query,
                expected_source.artifact_kind,
                expected_source.schema,
                29,
                expected_source.payload,
            )
            adapter = StructBindTypedAdapter(
                ExactTypeCompatibilityResolver(),
                _binding_failures(20671),
            )
            typed_endpoints = tuple(
                TypedStructBindEndpoint(item.slot_ref, item.variable)
                for item in endpoints)
            correspondences = tuple(
                adapter.read_from(
                    clone.edge_store, source_slot, typed_endpoints,
                ).correspondences[0]
                for source_slot in source_slots
            )
            reason = minimal_instruction_identity(
                (20672, 1), owner=clone.scope_owner,
                versions=eval_source.versions)
            choices = tuple(
                ArtifactBindingChoice(
                    endpoints[index * 2],
                    endpoints[index * 2 + 1],
                    correspondences[index].stable_key(),
                    reason,
                    (20673, index + 1),
                )
                for index in range(2)
            )
            request = ArtifactBindingRequest(
                proposition_identity(eval_source, (20674, 1)),
                definition,
                eval_source,
                query,
                (20675, 1),
                endpoints,
                values,
                choices,
                expected,
            )
            runtime = ArtifactBindingRuntime(
                clone.edge_store,
                adapter,
                _bridge(formal),
                clone.work_memory,
            )

            run = runtime.run(request)

            assert run.succeeded is True
            assert run.invocation.definition.program.source == formal["source"]
            assert run.invocation.source == eval_source
            assert clone.work_memory.query_artifact_results == [run.result]
            _stop_work_memory(clone)

        assert backend.snapshot() == host_backend
        assert context.work_memory.episode_artifacts == host_artifacts
        assert context.work_memory.query_artifact_results == host_results
    finally:
        _stop_work_memory(context)
        backend.close()
