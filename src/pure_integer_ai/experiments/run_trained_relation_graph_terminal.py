"""脱离课程和 QA SQLite 运行训练后 typed relation 图终端。"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
from typing import BinaryIO

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    GRAPH_RELATION_CONFLICT,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.dialogue_successor_graph import (
    SqliteDialogueSuccessorRuntime,
)


_FALLBACK_HASHER = Hasher("trained_relation_graph.fallback.v1")


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
        fallback_surfaces: tuple[str, ...],
        memory_database: str | Path | None = None,
        memory_tenant_id: int = 1,
        memory_user_id: int = 1,
        memory_session_id: int = 1,
        input_stream: BinaryIO | None = None,
        output_stream: BinaryIO | None = None,
        protocol_stream: bool = False,
        metrics_output: str | Path | None = None,
        ) -> int:
    """运行图优先交互；未唯一命中时输出模型数据中的自然语言表层。"""
    if (not isinstance(fallback_surfaces, tuple)
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
        dialogue = SqliteDialogueSuccessorRuntime(training_database)
    except ValueError:
        dialogue = None
    else:
        if dialogue.count() <= 0:
            dialogue.close()
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
    try:
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
                started = time.perf_counter_ns()
                decision = runtime.query(text)
                result = decision.answer
                recalled = (
                    None if result is not None
                    or decision.result_code == GRAPH_RELATION_CONFLICT
                    or memory is None
                    else memory.recall(text))
                dialogue_answer = (
                    None if result is not None or recalled is not None
                    or decision.result_code == GRAPH_RELATION_CONFLICT
                    or dialogue is None
                    else dialogue.respond(text, history=tuple(history[-6:])))
                surface = (
                    result.surface if result is not None
                    else recalled.surface if recalled is not None
                    else dialogue_answer.surface
                    if dialogue_answer is not None
                    else _fallback(text, fallback_surfaces)
                )
                if result is not None:
                    route_counts["core_graph"] += 1
                    core_fact_reads += result.fact_reads
                elif recalled is not None:
                    route_counts["memory_graph"] += 1
                    memory_posting_reads += recalled.posting_reads
                elif dialogue_answer is not None:
                    route_counts["dialogue_graph"] += 1
                    dialogue_posting_reads += dialogue_answer.posting_rows_read
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
                            "kind": "core_relation_graph",
                            "source_hash": result.source_hash,
                            "proposition": list(
                                result.proposition.stable_key()),
                            "frame_source_hash": (
                                result.generation.frame_source_hash),
                        }
                    elif recalled is not None:
                        response["source"] = {
                            "kind": "interaction_memory_graph",
                            "source_hash": recalled.source_hash,
                            "source_ref": list(
                                recalled.source.stable_key()),
                        }
                    elif dialogue_answer is not None:
                        response["source"] = {
                            "kind": "core_dialogue_successor_graph",
                            "source_hash": dialogue_answer.source_hash,
                            "proposition": list(
                                dialogue_answer.proposition_ref),
                        }
                    payload = json.dumps(response,
                        ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode("utf-8") + b"\n"
                    stream_out.write(payload)
                else:
                    stream_out.write(surface.encode("utf-8") + b"\n")
                stream_out.flush()
                if memory is not None:
                    memory.append(text, speaker_kind=1)
                    memory.append(surface, speaker_kind=2)
                history.extend(((1, text), (2, surface)))
                if len(history) > 6:
                    del history[:-6]
                latencies_us.append(max(
                    0, (time.perf_counter_ns() - started) // 1000))
    finally:
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
        fallback_surfaces = load_fallback_surfaces(release.fallback_surfaces)
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
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_fallback_surfaces",
    "main",
    "run_trained_relation_graph_terminal",
]
