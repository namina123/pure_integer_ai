"""脱离课程和 QA SQLite 运行训练后 typed relation 图终端。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import json
import os
from pathlib import Path
import sys
import time
from typing import BinaryIO

from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    GRAPH_RELATION_CONFLICT,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)


def _locate_release_root(training_database: Path) -> Path:
    """从训练图 SQLite 向上恢复发布根（含 trained_graph_release.json）。"""
    candidate = training_database.parent
    while True:
        if (candidate / "trained_graph_release.json").is_file():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            raise ValueError("训练图 SQLite 不在发布根内")
        candidate = parent


def _require_session_outside_release(session: str | Path,
                                     release_root: Path) -> None:
    """拒绝把会话库解析到 release root 内；包外是唯一允许的会话位置。"""
    resolved = Path(session).expanduser().resolve()
    release = release_root.resolve()
    try:
        resolved.relative_to(release)
    except ValueError:
        return
    raise ValueError("runtime memory 会话库不得位于 release root 内")


def _graph_object_keys(value: object) -> tuple[tuple[int, ...], ...]:
    """规范化 JSONL 显式图输入；语义载荷只能是非负严格整数。"""
    if value is None:
        return ()
    if (not isinstance(value, list)
            or any(not isinstance(key, list) or not key
                   or any(type(item) is not int or item < 0 for item in key)
                   for key in value)):
        raise ValueError("graph_object_keys 必须是非空非负整数数组的数组")
    keys = tuple(tuple(key) for key in value)
    if len(set(keys)) != len(keys):
        raise ValueError("graph_object_keys 不得重复")
    return tuple(sorted(keys))


def _trace_graph_key_set(value: object, *, label: str) -> set[tuple[int, ...]]:
    """Recover hashable integer graph identities from a query trace field."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} 必须是整数键序列")
    result = set()
    for key in value:
        if (not isinstance(key, (list, tuple)) or not key
                or any(type(item) is not int or item < 0 for item in key)):
            raise ValueError(f"{label} 含无效整数键")
        result.add(tuple(key))
    return result


def _response_plan_graph_keys(trace: dict[str, object]) -> set[tuple[int, ...]]:
    """从完整或 compact trace 的最终 ResponsePlan 恢复对象键。"""
    plan = trace.get("response_plan")
    if not isinstance(plan, dict):
        return set()
    table_value = trace.get("integer_key_table", ())
    table = tuple(
        tuple(key) for key in table_value
        if (isinstance(key, list)
            and all(type(item) is int for item in key))
    ) if isinstance(table_value, list) else ()

    def decode(value: object) -> tuple[int, ...] | None:
        if (isinstance(value, list) and value
                and all(type(item) is int for item in value)):
            return tuple(value)
        if isinstance(value, dict):
            ref = value.get("integer_key_ref")
            if type(ref) is int and 0 < ref <= len(table):
                return table[ref - 1]
        return None

    result = set()
    for field in ("claim_refs", "event_refs", "discourse_links"):
        values = plan.get(field, ())
        if not isinstance(values, list):
            continue
        for value in values:
            key = decode(value)
            if key is not None:
                result.add(key)
    slots = plan.get("slot_sequence", ())
    if isinstance(slots, list):
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            key = decode(slot.get("filler"))
            if key is not None:
                result.add(key)
    return result


def _nearest_rank(values: list[int], percentile: int) -> int:
    """以整数 nearest-rank 返回非空延迟序列的分位数。"""
    if not values or any(type(item) is not int or item < 0 for item in values):
        raise ValueError("latency values 必须是非空非负整数 list")
    if type(percentile) is not int or not 1 <= percentile <= 100:
        raise ValueError("percentile 必须是 1..100 整数")
    ordered = sorted(values)
    rank = (len(ordered) * percentile + 99) // 100
    return ordered[rank - 1]


def _peak_working_set_bytes() -> int:
    """读取宿主进程峰值工作集；不可用时返回零而不改变模型语义。"""
    if sys.platform == "win32":
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong)
        get_info.restype = ctypes.c_int
        if get_info(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.peak_working_set_size)
        return 0
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError):
        return 0
    return value * (1024 if sys.platform != "darwin" else 1)


def run_trained_relation_graph_terminal(
        *,
        training_database: str | Path,
        fallback_surfaces: tuple[str, ...] | None,
        memory_database: str | Path | None = None,
        memory_tenant_id: int = 1,
        memory_user_id: int = 1,
        memory_session_id: int = 1,
        input_stream: BinaryIO | None = None,
        output_stream: BinaryIO | None = None,
        protocol_stream: bool = False,
        metrics_output: str | Path | None = None,
        strict_graph: bool = False,
        ) -> int:
    """运行图优先交互；strict_graph 发布模式只接受三类图路由。

    三类图全部无组合结果时不伪造表层：JSONL 协议输出 type=no_answer
    （text 为空 + source.kind=uncovered），terminal 交互输出空行，进程保持
    运行以继续后续请求（阶段 D/E 前该语义缺口保留为内部 no_answer）。
    """
    if strict_graph:
        if fallback_surfaces not in (None, ()):
            raise ValueError("strict graph 不接受 fallback_surfaces")
    else:
        # 旧入口会加载 successor occurrence、Memory raw recall 或固定
        # fallback surface；这些路径不属于三图 QueryState，也不允许继续
        # 作为任何生产/兼容回答来源。保留函数名只为让旧调用显式失败，
        # 不再让调用方误以为它是可发布的诊断后备链。
        raise ValueError(
            "仅允许 strict graph 三图发布入口；旧 surface/successor/fallback "
            "路径已隔离")
    metrics_path = (
        None if metrics_output is None else Path(metrics_output).resolve())
    if metrics_path is not None and metrics_path.exists():
        raise ValueError("metrics_output 已存在，拒绝覆盖")
    stream_in = sys.stdin.buffer if input_stream is None else input_stream
    stream_out = sys.stdout.buffer if output_stream is None else output_stream
    if strict_graph and memory_database is None:
        # 发布 runtime 的每一轮交互都必须进入 Interaction Memory 图；默认把
        # 会话库放在模型旁的运行时文件，不污染只读训练 SQLite 或发布清单。
        training_path = Path(training_database).resolve()
        # 阶段 A 边界修正：release root 必须保持只读闭合。默认会话库不再
        # 推导到 release root 任何祖先/同级目录，而是使用用户级包外会话
        # 目录（可用 PURE_INTEGER_AI_SESSION_DIR 覆盖），并拒绝落在
        # release root 内；模型迁移时会话库按独立会话携带。
        release_root = _locate_release_root(training_path)
        session_dir = Path(
            os.environ.get("PURE_INTEGER_AI_SESSION_DIR")
            or (Path.home() / ".pure_integer_ai_sessions")
        ).expanduser().resolve()
        memory_database = session_dir / (
            release_root.name + "_runtime_memory.sqlite3")
        _require_session_outside_release(memory_database, release_root)
    startup_started = time.perf_counter_ns()
    memory = (
        None if memory_database is None or strict_graph
        else TrainedDialogueMemoryGraph(
            memory_database,
            tenant_id=memory_tenant_id,
            user_id=memory_user_id,
            session_id=memory_session_id,
        ))
    if strict_graph:
        # successor occurrence 只保留给非 strict 兼容/诊断入口。strict 的
        # Dialogue owner 由并行桥中的 ResponseAct/realization frame 承重。
        dialogue = None
    else:
        from pure_integer_ai.experiments.dialogue_successor_graph import (
            SqliteDialogueSuccessorRuntime,
        )
        try:
            dialogue = SqliteDialogueSuccessorRuntime(
                training_database, graph_dialogue=False)
        except ValueError:
            dialogue = None
        else:
            if dialogue.count() <= 0:
                dialogue.close()
                dialogue = None
    history = (
        [] if memory is None or strict_graph
        else [(item.speaker_kind, item.surface)
              for item in memory.recent_turns(limit=6)]
    )
    latencies_us: list[int] = []
    core_fact_reads = 0
    memory_posting_reads = 0
    dialogue_posting_reads = 0
    graph_input_request_count = 0
    graph_input_consumed_count = 0
    graph_input_memory_consumed_count = 0
    graph_input_generation_consumed_count = 0
    route_counts = {
        "core_graph": 0,
        "memory_graph": 0,
        "dialogue_graph": 0,
        "boundary": 0,
    }
    generation_runtime = None
    parallel_bridge = None
    try:
        if strict_graph:
            parallel_bridge = TrainedGraphQueryBridge(
                training_database,
                memory_database=memory_database,
                tenant_id=memory_tenant_id,
                user_id=memory_user_id,
                session_id=memory_session_id,
                successor_evidence=False,
            )
            from pure_integer_ai.cognition.understanding.query_structure_adapter import (
                RelationSurfaceStructureAdapter,
            )
            relation_pairs = tuple(sorted({
                (RelationSurfaceStructureAdapter.from_facts((fact,))[0]
                 .relation_structure_key(), fact.predicate.stable_key())
                for fact in parallel_bridge.facts
            }))
            generation_runtime = TrainedGenerationConnectorRuntime(
                training_database,
                shared_relation_runtime=parallel_bridge.core_runtime,
                relation_pairs=relation_pairs,
            )
            parallel_bridge.surface_generator = generation_runtime
            # strict 每轮写入和查询复用桥拥有的唯一 Memory owner，避免同一
            # session SQLite 出现双连接写锁或两个不一致的 Observation 视图。
            memory = parallel_bridge.memory
        query_local_id = 0
        runtime_owner = (
            nullcontext(None)
            if strict_graph
            else TrainedRelationGraphRuntime(training_database)
        )
        with runtime_owner as runtime:
            startup_us = max(
                0, (time.perf_counter_ns() - startup_started) // 1000)
            while True:
                if not protocol_stream:
                    stream_out.write(b"> ")
                    stream_out.flush()
                raw = stream_in.readline()
                if raw == b"":
                    break
                request_id = None
                request_graph_keys: tuple[tuple[int, ...], ...] = ()
                generated_graph_keys: set[tuple[int, ...]] = set()
                if protocol_stream:
                    # Windows 管道或编辑器可能在首行带 UTF-8 BOM；它不应
                    # 改变 JSONL 协议语义，因此仅在输入边界容忍一次 BOM。
                    request = json.loads(raw.decode("utf-8-sig"))
                    if not isinstance(request, dict):
                        raise ValueError("JSONL 请求必须是对象")
                    request_id = request.get("id")
                    operation = request.get("op", "turn")
                    if operation in {"quit", "exit"}:
                        break
                    if operation != "turn":
                        raise ValueError("JSONL op 未注册")
                    request_graph_keys = _graph_object_keys(
                        request.get("graph_object_keys"))
                    text = request.get("text")
                    if text is None:
                        if not request_graph_keys:
                            raise ValueError(
                                "JSONL turn 必须包含非空 text 或 graph_object_keys")
                    elif type(text) is not str or not text.strip():
                        raise ValueError("JSONL turn.text 必须是非空文本")
                    if request_graph_keys and not strict_graph:
                        raise ValueError(
                            "graph_object_keys 只允许 strict graph 发布入口")
                else:
                    payload = raw.rstrip(b"\r\n")
                    if payload in {b":quit", b":exit"}:
                        break
                    text = payload.decode("utf-8")
                if text is not None and not text.strip():
                    continue
                query_local_id += 1
                # 记忆是正式运行时的承重路径：先记录用户输入，再执行三图
                # 查询。即使后续图核验失败，该次交互也不会静默丢失。
                input_observation_ref = None
                prepared_input_structure = None
                if strict_graph and parallel_bridge is not None:
                    prepared_input_structure = (
                        parallel_bridge.prepare_input_structure(
                            text, graph_object_keys=request_graph_keys))
                if memory is not None:
                    input_append = (
                        memory.append_graph_input(
                            request_graph_keys, speaker_kind=1)
                        if text is None else
                        memory.append(
                            text,
                            speaker_kind=1,
                            input_structure=prepared_input_structure,
                        )
                    )
                    input_observation_ref = memory.intake.result_for_source(input_append.source).observation_ref
                started = time.perf_counter_ns()
                parallel_trace = (
                    parallel_bridge.query(
                        text,
                        graph_object_keys=request_graph_keys,
                        input_observation_ref=input_observation_ref,
                        compact_trace=True,
                        input_structure=prepared_input_structure,
                    )
                    if strict_graph and parallel_bridge is not None
                    else None
                )
                work_memory_projection = None
                if strict_graph and parallel_bridge is not None and parallel_trace is not None:
                    from pure_integer_ai.experiments.work_memory_generation_adapter import (
                        query_work_memory_projection,
                    )
                    work_memory_projection = query_work_memory_projection(
                        parallel_bridge,
                        parallel_trace,
                        query_local_id=query_local_id,
                    )
                    consumed_graph_keys = _trace_graph_key_set(
                        parallel_trace.get(
                            "discourse_graph_input_keys", ()),
                        label="Dialogue graph inputs")
                    consumed_graph_keys.update(_trace_graph_key_set(
                        parallel_trace.get(
                            "event_time_graph_input_keys", ()),
                        label="Event/Time graph inputs"))
                    consumed_graph_keys.update(_trace_graph_key_set(
                        parallel_trace.get("core_graph_input_keys", ()),
                        label="Core graph inputs"))
                    consumed_graph_keys.update(_trace_graph_key_set(
                        parallel_trace.get("generic_graph_input_keys", ()),
                        label="Generic Core graph inputs"))
                    memory_graph_keys = _trace_graph_key_set(
                        parallel_trace.get("memory_graph_input_keys", ()),
                        label="Memory graph inputs")
                    if (request_graph_keys
                            and not set(request_graph_keys) <= memory_graph_keys):
                        raise ValueError(
                            "显式图输入未进入当前 Memory Observation/Hypothesis/Evidence")
                    if (request_graph_keys
                            and not set(request_graph_keys) <= consumed_graph_keys):
                        raise ValueError(
                            "显式图输入未被当前三图 QueryState 完整消费")
                    graph_input_request_count += len(request_graph_keys)
                    graph_input_consumed_count += len(
                        set(request_graph_keys) & consumed_graph_keys)
                    graph_input_memory_consumed_count += len(
                        set(request_graph_keys) & memory_graph_keys)
                    generated_graph_keys = _response_plan_graph_keys(
                        parallel_trace)
                    if (request_graph_keys
                            and not set(request_graph_keys) <= generated_graph_keys):
                        raise ValueError(
                            "显式图输入未进入当前最终 ResponsePlan")
                    graph_input_generation_consumed_count += len(
                        set(request_graph_keys) & generated_graph_keys)
                # 非 strict 入口仅作旧兼容/诊断；它也必须经过显式生成器，
                # 因此没有 connector 时不会回到 SourceRecord 表层。
                decision = (
                    None if strict_graph
                    else runtime.query(text, surface_generator=None)
                )
                result = None if decision is None else decision.answer
                # 阶段 A 边界：strict 发布路由不再把表层相似召回当作回答来源。
                # TrainedDialogueMemoryGraph.recall 目前仍只按 1--3 宽度码点
                # 片段重叠返回 raw_text，不能形成实体/事件/属性/时间/指代
                # 结构候选；在阶段 D 改造成结构候选接口前，发布路由断开该
                # 路径并记录阻塞。Memory 仍保留 append 观察写入与 recent_turns
                # 热区（供 Dialogue 承接），但不再直接输出旧表层。
                recalled = (
                    None if strict_graph or result is not None
                    or decision.result_code == GRAPH_RELATION_CONFLICT
                    or memory is None
                    or strict_graph
                    else memory.recall(
                        text,
                        minimum_similarity_permille=500))
                dialogue_answer = (
                    None if strict_graph or result is not None or recalled is not None
                    or dialogue is None
                    else dialogue.respond(
                        text, history=tuple(history[-6:])))
                surface = (
                    parallel_trace["response_surface"]
                    if parallel_trace is not None
                    else result.surface if result is not None
                    else recalled.surface if recalled is not None
                    else dialogue_answer.surface
                    if dialogue_answer is not None
                    else ""
                )
                if strict_graph and not surface.strip():
                    # A release must never expose a boundary route.  If all
                    # three trained graph owners fail to produce a path, the
                    # protocol fails closed after the input has been recorded.
                    # 阶段 D/E 前该语义缺口保留为可验证的内部状态；对外仍以
                    # 协议级 stop（不是伪造表层）结束本轮，进程可继续服务
                    # 后续请求，不会因一句未覆盖闲聊崩溃。
                    surface = ""
                if parallel_trace is not None and surface.strip():
                    evidence_spaces = {
                        item["space"] for item in parallel_trace["evidence"]}
                    if 1 in evidence_spaces:
                        route_counts["core_graph"] += 1
                        core_fact_reads += sum(
                            1 for item in parallel_trace["hops"]
                            if item["space"] == 1)
                    if 2 in evidence_spaces:
                        route_counts["memory_graph"] += 1
                        memory_posting_reads += sum(
                            1 for item in parallel_trace["hops"]
                            if item["space"] == 2)
                    if 3 in evidence_spaces:
                        route_counts["dialogue_graph"] += 1
                        dialogue_posting_reads += sum(
                            1 for item in parallel_trace["hops"]
                            if item["space"] == 3)
                elif result is not None:
                    route_counts["core_graph"] += 1
                    core_fact_reads += result.fact_reads
                elif recalled is not None:
                    route_counts["memory_graph"] += 1
                    memory_posting_reads += recalled.posting_reads
                elif dialogue_answer is not None:
                    route_counts["dialogue_graph"] += 1
                    dialogue_posting_reads += dialogue_answer.posting_rows_read
                elif strict_graph:
                    # 三图均无可组合结果：不伪造表层，不计入任何 route。
                    pass
                else:
                    route_counts["boundary"] += 1
                if protocol_stream:
                    response = {
                        "id": request_id,
                        "text": surface,
                        "type": "turn",
                    }
                    if parallel_trace is not None and surface.strip():
                        response["source"] = {
                            "kind": "parallel_three_graph",
                            "termination": parallel_trace["termination"],
                            "roots": parallel_trace["roots"],
                            "evidence": parallel_trace["evidence"],
                            # compact V4 traces carry the same integer edge
                            # references under ``expanded_edge_refs``; expose
                            # whichever canonical field this query produced.
                            "expanded_edges": parallel_trace.get(
                                "expanded_edges",
                                parallel_trace.get("expanded_edge_refs", [])),
                            "state_key": parallel_trace["state_key"],
                            "response_plan": parallel_trace["response_plan"],
                            "graph_object_keys": [
                                list(key) for key in request_graph_keys],
                            "consumed_graph_object_keys": [
                                list(key) for key in sorted(consumed_graph_keys)
                                if key in set(request_graph_keys)],
                            "memory_graph_object_keys": [
                                list(key) for key in sorted(memory_graph_keys)
                                if key in set(request_graph_keys)],
                            "generation_consumed_graph_object_keys": [
                                list(key) for key in sorted(generated_graph_keys)
                                if key in set(request_graph_keys)],
                        }
                        if work_memory_projection is not None:
                            response["source"]["work_memory_projection"] = (
                                work_memory_projection)
                    elif result is not None:
                        response["source"] = {
                            "kind": ("core_graph" if strict_graph
                                     else "core_relation_graph"),
                            "source_hash": result.source_hash,
                            "proposition": list(
                                result.proposition.stable_key()),
                        }
                        if strict_graph:
                            response["source"]["generation"] = {
                                "kind": (
                                    "typed_connector_graph"
                                    if result.generation.representations
                                    else "structural_connector_graph"),
                                "connector": list(
                                    result.generation.connector.stable_key()),
                                "representation_count": len(
                                    result.generation.representations),
                                "trace": list(result.generation.trace),
                            }
                        else:
                            response["source"]["frame_source_hash"] = (
                                result.generation.frame_source_hash)
                    elif recalled is not None:
                        response["source"] = {
                            "kind": ("memory_graph" if strict_graph
                                     else "interaction_memory_graph"),
                            "source_hash": recalled.source_hash,
                            "source_ref": list(
                                recalled.source.stable_key()),
                        }
                    elif dialogue_answer is not None:
                        response["source"] = {
                            "kind": "dialogue_graph",
                            "source_hash": dialogue_answer.source_hash,
                            "proposition": list(dialogue_answer.proposition_ref),
                            "confidence_permille": dialogue_answer.confidence_permille,
                            "trace": {
                                "schema": "dialogue_pipeline_v1",
                                "understanding_tokens": [
                                    list(token) for token in
                                    dialogue_answer.trace.understanding_tokens],
                                "understanding_token_count": len(
                                    dialogue_answer.trace.understanding_tokens),
                                "process_candidate_count": dialogue_answer.trace.process_candidate_count,
                                "process_selected_key": list(
                                    dialogue_answer.trace.process_selected_key),
                                "input_exact": (
                                    dialogue_answer.trace.input_exact),
                                "result_mode": (
                                    dialogue_answer.trace.result_mode),
                                "transformation_count": (
                                    dialogue_answer.trace.transformation_count),
                                "support_count": (
                                    dialogue_answer.trace.support_count),
                                "confidence_permille": (
                                    dialogue_answer.trace.confidence_permille),
                                "result_tokens": [
                                    list(token) for token in
                                    dialogue_answer.trace.result_tokens],
                                "result_token_count": len(dialogue_answer.trace.result_tokens),
                                "stable_key": list(
                                    dialogue_answer.trace.stable_key()),
                            },
                        }
                    # The strict JSONL boundary is a user-facing surface.  A
                    # failed graph composition remains available in the
                    # internal session trace/metrics, but protocol responses
                    # must not leak termination codes, QueryState internals,
                    # or an ``uncovered`` pseudo-answer.  Keep the turn
                    # shape stable and fail closed with an empty text value.
                    payload = json.dumps(response,
                        ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode("utf-8") + b"\n"
                elif strict_graph and not surface.strip():
                    payload = b"\n"
                else:
                    payload = surface.encode("utf-8") + b"\n"
                if stream_out.write(payload) != len(payload):
                    raise OSError("输出流未完整接收本次回应，不能登记生成采用")
                stream_out.flush()
                if memory is not None and surface.strip():
                    adoption = (parallel_bridge.acknowledge_response_delivery(surface)
                                if parallel_trace is not None else None)
                    if adoption is None:
                        memory.append(surface, speaker_kind=2)
                if text is not None:
                    history.extend(((1, text), (2, surface)))
                    if len(history) > 6:
                        del history[:-6]
                latencies_us.append(max(
                    0, (time.perf_counter_ns() - started) // 1000))
    finally:
        if parallel_bridge is not None:
            parallel_bridge.close()
        if generation_runtime is not None:
            generation_runtime.close()
        if memory is not None:
            memory.close()
        if dialogue is not None:
            dialogue.close()
    if metrics_path is not None:
        if not latencies_us:
            raise ValueError("metrics_output 没有可记录的会话轮次")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({
            "format": "TRAINED_RELATION_GRAPH_DIALOGUE_METRICS_V1",
            "schema_version": 1,
            "turn_count": len(latencies_us),
            "startup_us": startup_us,
            "latency_p50_us": _nearest_rank(latencies_us, 50),
            "latency_p95_us": _nearest_rank(latencies_us, 95),
            "latency_max_us": max(latencies_us),
            "core_fact_reads": core_fact_reads,
            "memory_posting_reads": memory_posting_reads,
            "dialogue_posting_reads": dialogue_posting_reads,
            "graph_input_request_count": graph_input_request_count,
            "graph_input_consumed_count": graph_input_consumed_count,
            "graph_input_memory_consumed_count": (
                graph_input_memory_consumed_count),
            "graph_input_generation_consumed_count": (
                graph_input_generation_consumed_count),
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "route_counts": route_counts,
        }, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析独立训练图终端参数。"""
    parser = argparse.ArgumentParser(
        description="run trained integer relation graph terminal")
    parser.add_argument("--release-root", default=None)
    parser.add_argument("--training-database", default=None)
    parser.add_argument(
        "--memory-database", default=None,
        help="可选 interaction Memory SQLite；会话输入将跨进程持久化")
    parser.add_argument("--memory-tenant-id", type=int, default=1)
    parser.add_argument("--memory-user-id", type=int, default=1)
    parser.add_argument("--memory-session-id", type=int, default=1)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument(
        "--protocol", choices=("terminal", "jsonl"), default="terminal")
    args = parser.parse_args(argv)
    if args.release_root is not None:
        if args.training_database is not None:
            parser.error("--release-root 不得与训练数据库/表层文件同时指定")
        from pure_integer_ai.experiments.trained_graph_release import (
            load_trained_graph_release,
        )
        release = load_trained_graph_release(args.release_root)
        training_database = release.training_database
        # 发布 strict graph 不读取边界表层文件；所有输出只能来自三类图。
        fallback_surfaces = ()
    else:
        parser.error("必须指定 --release-root；旧训练数据库/表层兼容入口已隔离")
    return run_trained_relation_graph_terminal(
        training_database=training_database,
        fallback_surfaces=fallback_surfaces,
        memory_database=args.memory_database,
        memory_tenant_id=args.memory_tenant_id,
        memory_user_id=args.memory_user_id,
        memory_session_id=args.memory_session_id,
        metrics_output=args.metrics_output,
        protocol_stream=args.protocol == "jsonl",
        strict_graph=args.release_root is not None,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_graph_object_keys",
    "_response_plan_graph_keys",
    "_trace_graph_key_set",
    "main",
    "run_trained_relation_graph_terminal",
]
