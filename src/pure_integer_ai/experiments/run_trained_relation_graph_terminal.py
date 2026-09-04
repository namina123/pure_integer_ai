"""脱离课程和 QA SQLite 运行训练后 typed relation 图终端。"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
from typing import BinaryIO

from pure_integer_ai.crosscut.determinism.hasher import Hasher
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
from pure_integer_ai.experiments.dialogue_successor_graph import (
    SqliteDialogueSuccessorRuntime,
)
from pure_integer_ai.cognition.shared.dialogue_pipeline import (
    DIALOGUE_RESULT_CLARIFICATION,
)


_FALLBACK_HASHER = Hasher("trained_relation_graph.fallback.v1")


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


def load_fallback_surfaces(path: str | Path) -> tuple[str, ...]:
    """读取发布模型内的自然语言未命中表层；空行不进入可选集合。"""
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError("自然语言未命中表层文件不存在")
    values = tuple(
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not values:
        raise ValueError("自然语言未命中表层文件为空")
    return values


def _fallback(text: str, surfaces: tuple[str, ...]) -> str:
    """按输入码点确定性选择一个已训练表层，不解释具体语言。"""
    key = tuple(ord(character) for character in text)
    index = (_FALLBACK_HASHER.h63(key) or 1) % len(surfaces)
    return surfaces[index]


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
    elif (not isinstance(fallback_surfaces, tuple)
          or not fallback_surfaces
          or any(type(item) is not str or not item.strip()
                 for item in fallback_surfaces)):
        raise ValueError("fallback_surfaces 必须是非空文本 tuple")
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
        None if memory_database is None
        else TrainedDialogueMemoryGraph(
            memory_database,
            tenant_id=memory_tenant_id,
            user_id=memory_user_id,
            session_id=memory_session_id,
        ))
    try:
        dialogue = SqliteDialogueSuccessorRuntime(
            training_database, graph_dialogue=strict_graph)
    except ValueError:
        if strict_graph:
            raise
        dialogue = None
    else:
        if dialogue.count() <= 0:
            dialogue.close()
            if strict_graph:
                raise RuntimeError("strict graph 缺少 Dialogue 图路径")
            dialogue = None
    history = (
        [] if memory is None
        else [(item.speaker_kind, item.surface)
              for item in memory.recent_turns(limit=6)]
    )
    latencies_us: list[int] = []
    core_fact_reads = 0
    memory_posting_reads = 0
    dialogue_posting_reads = 0
    route_counts = {
        "core_graph": 0,
        "memory_graph": 0,
        "dialogue_graph": 0,
        "boundary": 0,
    }
    generation_runtime = None
    try:
        if strict_graph:
            generation_runtime = TrainedGenerationConnectorRuntime(
                training_database)
        with TrainedRelationGraphRuntime(training_database) as runtime:
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
                    text = request.get("text")
                    if type(text) is not str or not text.strip():
                        raise ValueError("JSONL turn.text 必须是非空文本")
                else:
                    payload = raw.rstrip(b"\r\n")
                    if payload in {b":quit", b":exit"}:
                        break
                    text = payload.decode("utf-8")
                    if not text.strip():
                        continue
                # 记忆是正式运行时的承重路径：先记录用户输入，再执行三图
                # 查询。即使后续图核验失败，该次交互也不会静默丢失。
                if memory is not None:
                    memory.append(text, speaker_kind=1)
                started = time.perf_counter_ns()
                decision = runtime.query(
                    text,
                    surface_generator=generation_runtime,
                )
                result = decision.answer
                if (strict_graph and result is not None
                        and result.generation.connector is None):
                    raise RuntimeError(
                        "strict graph Core 回答绕过 typed connector")
                # 阶段 A 边界：strict 发布路由不再把表层相似召回当作回答来源。
                # TrainedDialogueMemoryGraph.recall 目前仍只按 1--3 宽度码点
                # 片段重叠返回 raw_text，不能形成实体/事件/属性/时间/指代
                # 结构候选；在阶段 D 改造成结构候选接口前，发布路由断开该
                # 路径并记录阻塞。Memory 仍保留 append 观察写入与 recent_turns
                # 热区（供 Dialogue 承接），但不再直接输出旧表层。
                recalled = (
                    None if result is not None
                    or decision.result_code == GRAPH_RELATION_CONFLICT
                    or memory is None
                    or strict_graph
                    else memory.recall(
                        text,
                        minimum_similarity_permille=500))
                dialogue_answer = (
                    None if result is not None or recalled is not None
                    or dialogue is None
                    else (dialogue.respond_graph(
                        text, history=tuple(history[-6:])) if strict_graph
                          else dialogue.respond(
                              text, history=tuple(history[-6:]))))
                surface = (
                    result.surface if result is not None
                    else recalled.surface if recalled is not None
                    else dialogue_answer.surface
                    if dialogue_answer is not None
                    else ("" if strict_graph else _fallback(text, fallback_surfaces))
                )
                if strict_graph and not surface.strip():
                    # A release must never expose a boundary route.  If all
                    # three trained graph owners fail to produce a path, the
                    # protocol fails closed after the input has been recorded.
                    # 阶段 D/E 前该语义缺口保留为可验证的内部状态；对外仍以
                    # 协议级 stop（不是伪造表层）结束本轮，进程可继续服务
                    # 后续请求，不会因一句未覆盖闲聊崩溃。
                    surface = ""
                if result is not None:
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
                    if result is not None:
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
                        if (strict_graph
                                and dialogue_answer.trace.result_mode
                                == DIALOGUE_RESULT_CLARIFICATION):
                            response["type"] = "clarify"
                            response["source"]["status"] = "insufficient_evidence"
                    if strict_graph and not surface.strip():
                        response["type"] = "no_answer"
                        response["source"] = {
                            "kind": "uncovered",
                            "status": "three_graph_no_combination",
                        }
                    payload = json.dumps(response,
                        ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode("utf-8") + b"\n"
                    stream_out.write(payload)
                elif strict_graph and not surface.strip():
                    stream_out.write(b"\n")
                else:
                    stream_out.write(surface.encode("utf-8") + b"\n")
                stream_out.flush()
                if memory is not None and surface.strip():
                    memory.append(surface, speaker_kind=2)
                history.extend(((1, text), (2, surface)))
                if len(history) > 6:
                    del history[:-6]
                latencies_us.append(max(
                    0, (time.perf_counter_ns() - started) // 1000))
    finally:
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
    parser.add_argument("--fallback-surfaces", default=None)
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
        if args.training_database is not None or args.fallback_surfaces is not None:
            parser.error("--release-root 不得与训练数据库/表层文件同时指定")
        from pure_integer_ai.experiments.trained_graph_release import (
            load_trained_graph_release,
        )
        release = load_trained_graph_release(args.release_root)
        training_database = release.training_database
        # 发布 strict graph 不读取边界表层文件；所有输出只能来自三类图。
        fallback_surfaces = ()
    else:
        if args.training_database is None or args.fallback_surfaces is None:
            parser.error("必须指定 --release-root 或训练数据库与表层文件")
        training_database = args.training_database
        fallback_surfaces = load_fallback_surfaces(args.fallback_surfaces)
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
    "load_fallback_surfaces",
    "main",
    "run_trained_relation_graph_terminal",
]
