"""在只读训练图上运行一次闭合 QueryState 与 G-02 WorkMemory 投影。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.experiments.work_memory_generation_adapter import (
    integer_trace_projection as _integer_projection,
    memory_structure_snapshot as _memory_snapshot,
    query_work_memory_projection as _query_work_memory_projection,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)


PROTOCOL_VERSION = 1


def _query_collection(query: dict[str, object], name: str) -> tuple[object, ...]:
    """Read compact or legacy QueryState collection without miscounting refs."""
    for key in (f"{name}_refs", f"{name}s", f"{name}_keys", name):
        value = query.get(key)
        if value is not None:
            if not isinstance(value, (tuple, list)):
                raise TypeError(f"QueryState {key} must be a collection")
            return tuple(value)
    return ()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


def _write_trace(value: object, target: Path) -> str:
    encoder = json.JSONEncoder(
        ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    with target.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for chunk in encoder.iterencode(value):
                stream.write(chunk.encode("ascii"))
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_slice(
        database: str | Path,
        run_root: str | Path,
        input_surface: str,
        *,
        tenant_id: int = 1,
        user_id: int = 1,
        session_id: int = 1,
        max_depth: int = 64,
        ) -> dict[str, object]:
    """运行真实输入；模型只读，session 与 receipt 写入新的 K 盘目录。"""
    model = Path(database).resolve(strict=True)
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or root == Path(root.anchor):
        raise ValueError("run_root 必须是 K 盘非根目录")
    if root.exists():
        raise ValueError("run_root 已存在，禁止覆盖")
    if type(input_surface) is not str or not input_surface:
        raise ValueError("input_surface 必须是非空文本")
    if type(max_depth) is not int or max_depth <= 0:
        raise ValueError("max_depth 必须是正整数")
    for label, value in (
            ("tenant_id", tenant_id), ("user_id", user_id),
            ("session_id", session_id)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} 必须是正整数")
    root.mkdir(parents=True)
    session = root / "session.sqlite3"
    surface_values = tuple(map(ord, input_surface))
    with TrainedGenerationConnectorRuntime(model) as generator, \
            TrainedGraphQueryBridge(
                model,
                memory_database=session,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                surface_generator=generator,
            ) as bridge:
        before = _memory_snapshot(bridge.memory)
        # Memory intake and QueryState consume the same graph-augmented
        # integer structure; otherwise KdConv closure creates divergent O/H/E
        # candidate identities before query() starts.
        input_structure = bridge.input_projector.project(tuple(
            map(ord, input_surface)))
        input_structure = bridge._augment_kdconv_input(input_structure)
        # Prepare the same graph-canonical factual topic before Memory O/H/E
        # intake; query() repeats the operation after the new Observation ref
        # exists, but must receive this identical integer structure.
        input_structure = bridge._restore_factual_topic_input(input_structure)
        appended = bridge.memory.append(
            input_surface, speaker_kind=1, input_structure=input_structure)
        observation_ref = bridge.memory.intake.result_for_source(
            appended.source).observation_ref
        if not isinstance(observation_ref, MemoryObjectRef):
            raise RuntimeError("输入 append 没有返回 Observation 引用")
        query = bridge.query(
            input_surface,
            max_depth=max_depth,
            input_observation_ref=observation_ref,
            input_structure=input_structure,
            # Frontier is persisted as an integer key table plus references.
            # Keeping this contract here prevents a long query from expanding
            # the same integer identities once per frontier step.
            compact_trace=True,
        )
        projection = _query_work_memory_projection(
            bridge, query, query_local_id=1)
        after = _memory_snapshot(bridge.memory)
        trace = {
            "protocol": PROTOCOL_VERSION,
            "input_values": list(surface_values),
            "observation_ref": list(observation_ref.stable_key()),
            "query": _integer_projection({
                "termination": query.get("termination"),
                "depth": query.get("depth"),
                "roots": query.get("roots"),
                "evidence": query.get("evidence"),
                "bindings": query.get("bindings"),
                "visited": query.get("visited"),
                "expanded_edges": query.get("expanded_edges"),
                "generic_response_feature_mask": query.get(
                    "generic_response_feature_mask"),
                "discourse_relation_feature_state": query.get(
                    "discourse_relation_feature_state"),
                "discourse_topic_query_projection": query.get(
                    "discourse_topic_query_projection"),
                "response_plan": query.get("response_plan"),
                "generation": query.get("generation"),
                "discourse_topic_candidates": _query_collection(
                    query, "discourse_topic_candidate"),
                "discourse_topic_hypothesis_keys": _query_collection(
                    query, "discourse_topic_hypothesis"),
                "work_memory_projection": projection,
            }),
            "response_surface_values": list(
                map(ord, query.get("response_surface", ""))),
        }
        trace_sha256 = _write_trace(trace, root / "query-001.json.gz")
    with TrainedGraphQueryBridge(
            model,
            memory_database=session,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
    ) as restored:
        restored_snapshot = _memory_snapshot(restored.memory)
    if restored_snapshot != after:
        raise AssertionError("session O/H/E 冷恢复不等")
    receipt = {
        "protocol": PROTOCOL_VERSION,
        "input_values": list(surface_values),
        "model_read_only": 1,
        "core_training_performed": 0,
        "before_ohe": [len(item) for item in before],
        "after_ohe": [len(item) for item in after],
        "cold_restore_equal": 1,
        "query_trace": "query-001.json.gz",
        "query_trace_sha256_values": list(bytes.fromhex(trace_sha256)),
        "termination_code": query.get("termination"),
        "root_count": len(query.get("roots", ())),
        "evidence_count": len(query.get("evidence", ())),
        "response_surface_values": list(map(ord, query.get("response_surface", ""))),
        "work_memory_projection_status": projection.get("status", 0),
        "work_memory_projection_item_count": projection.get("item_count", 0),
        "work_memory_situation_entry_count": projection.get("entry_count", 0),
        "discourse_topic_candidate_count": len(_query_collection(
            query, "discourse_topic_candidate")),
        "discourse_topic_hypothesis_count": len(_query_collection(
            query, "discourse_topic_hypothesis")),
        "free_dialogue_complete": 0,
    }
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def run_session(
        database: str | Path,
        run_root: str | Path,
        inputs: tuple[str, ...],
        *,
        graph_object_keys: tuple[tuple[int, ...], ...] = (),
        tenant_id: int = 1,
        user_id: int = 1,
        session_id: int = 1,
        max_depth: int = 64,
        ) -> dict[str, object]:
    """Run several real turns through one compact integer O/H/E session.

    Each turn appends its own Observation and feeds the resulting reference
    into the same three-space QueryState.  No response text is used as a
    lookup key; only the trained graph and the session's integer closure are
    consumed by the bridge.
    """
    model = Path(database).resolve(strict=True)
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or root == Path(root.anchor):
        raise ValueError("run_root 必须是 K 盘非根目录")
    if root.exists() or not inputs:
        raise ValueError("run_root 必须不存在且 inputs 非空")
    if any(type(item) is not str or not item for item in inputs):
        raise ValueError("inputs 必须是非空文本 tuple")
    if (type(graph_object_keys) is not tuple
            or any(type(key) is not tuple or not key
                   or any(type(value) is not int or value < 0 for value in key)
                   for key in graph_object_keys)):
        raise ValueError("graph_object_keys 必须是整数 tuple 集")
    root.mkdir(parents=True)
    session = root / "session.sqlite3"
    rows = []
    detailed = []
    with TrainedGenerationConnectorRuntime(model) as generator, \
            TrainedGraphQueryBridge(
                model, memory_database=session, tenant_id=tenant_id,
                user_id=user_id, session_id=session_id,
                surface_generator=generator) as bridge:
        before = _memory_snapshot(bridge.memory)
        for ordinal, input_surface in enumerate(inputs, 1):
            input_structure = bridge.input_projector.project(tuple(
                map(ord, input_surface)))
            input_structure = bridge._augment_kdconv_input(input_structure)
            input_structure = bridge._restore_factual_topic_input(input_structure)
            appended = bridge.memory.append(
                input_surface, speaker_kind=1, input_structure=input_structure)
            observation_ref = bridge.memory.intake.result_for_source(
                appended.source).observation_ref
            query = bridge.query(
                input_surface, max_depth=max_depth,
                input_observation_ref=observation_ref,
                input_structure=input_structure,
                graph_object_keys=graph_object_keys,
                compact_trace=True)
            projection = _query_work_memory_projection(
                bridge, query, query_local_id=ordinal)
            rows.append({
                "ordinal": ordinal,
                "input_values": list(map(ord, input_surface)),
                "observation_ref": list(observation_ref.stable_key()),
                "termination": query.get("termination"),
                "active_spaces": query.get("active_spaces"),
                "root_count": len(query.get("roots", ())),
                "evidence_count": len(query.get("evidence", ())),
                "generic_response_feature_mask": query.get(
                    "generic_response_feature_mask", 0),
                "discourse_relation_feature_state": list(query.get(
                    "discourse_relation_feature_state", ())),
                "discourse_graph_input_keys": [list(item) for item in query.get(
                    "discourse_graph_input_keys", ())],
                "discourse_graph_input_requested_keys": [
                    list(item) for item in query.get(
                        "discourse_graph_input_requested_keys", ())],
                "topic_candidate_count": len(_query_collection(
                    query, "discourse_topic_candidate")),
                "topic_hypothesis_count": len(_query_collection(
                    query, "discourse_topic_hypothesis")),
                "topic_graph_identity_count": len(_query_collection(
                    query, "discourse_topic_graph_identity")),
                "response_surface_values": list(map(ord, query.get(
                    "response_surface", ""))),
            })
            # The full QueryState is written once to the compressed session
            # trace below.  Receipt rows intentionally retain only counters
            # and output code points; duplicating frontier/binding/evidence
            # trees here was the source of avoidable artifact growth.
            rows[-1]["trace_offset"] = ordinal - 1
            detailed.append({
                "ordinal": ordinal,
                "query": _integer_projection({
                    "query_key": query.get("query_key"),
                    "termination": query.get("termination"),
                    "depth": query.get("depth"),
                    "roots": query.get("roots"),
                    "frontier_trace": query.get("frontier_trace"),
                    "bindings": query.get("bindings"),
                    "evidence": query.get("evidence"),
                    "visited": query.get("visited"),
                    "expanded_edges": query.get("expanded_edges"),
                    "generic_response_feature_mask": query.get(
                        "generic_response_feature_mask"),
                    "discourse_relation_feature_state": query.get(
                        "discourse_relation_feature_state"),
                    "discourse_graph_input_keys": query.get(
                        "discourse_graph_input_keys"),
                    "discourse_graph_input_requested_keys": query.get(
                        "discourse_graph_input_requested_keys"),
                    "response_plan": query.get("response_plan"),
                    "generation": query.get("generation"),
                    "discourse_topic_candidates": _query_collection(
                        query, "discourse_topic_candidate"),
                    "discourse_topic_hypothesis_keys": _query_collection(
                        query, "discourse_topic_hypothesis"),
                    "discourse_topic_graph_identity_keys": _query_collection(
                        query, "discourse_topic_graph_identity"),
                    "work_memory_projection": projection,
                }),
            })
            if query.get("response_surface"):
                bridge.acknowledge_response_delivery(
                    query["response_surface"])
        after = _memory_snapshot(bridge.memory)
    with TrainedGraphQueryBridge(
            model, memory_database=session, tenant_id=tenant_id,
            user_id=user_id, session_id=session_id) as restored:
        restored_snapshot = _memory_snapshot(restored.memory)
    if restored_snapshot != after:
        raise AssertionError("multi-turn session O/H/E cold restore differs")
    receipt = {
        "protocol": PROTOCOL_VERSION,
        "model_read_only": 1,
        "core_training_performed": 0,
        "turn_count": len(rows),
        "before_ohe": [len(item) for item in before],
        "after_ohe": [len(item) for item in after],
        "cold_restore_equal": 1,
        "active_spaces_contract": [1, 2, 3],
        "topic_candidate_count": sum(item["topic_candidate_count"] for item in rows),
        "topic_hypothesis_count": sum(item["topic_hypothesis_count"] for item in rows),
        "topic_graph_identity_count": sum(item["topic_graph_identity_count"] for item in rows),
        "discourse_graph_input_count": len(graph_object_keys),
        "free_dialogue_complete": 0,
        "turns": rows,
    }
    trace_payload = {"protocol": PROTOCOL_VERSION, "turns": detailed}
    trace_sha256 = _write_trace(trace_payload, root / "session-trace.json.gz")
    receipt["trace_sha256_values"] = list(bytes.fromhex(trace_sha256))
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input")
    parser.add_argument("--turn-input", action="append", default=[])
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-id", type=int, default=1)
    parser.add_argument("--graph-object-key", action="append", default=[])
    parser.add_argument("--max-depth", type=int, default=64)
    args = parser.parse_args(argv)
    if args.turn_input:
        graph_object_keys = tuple(
            tuple(int(item, 10) for item in value.split(","))
            for value in args.graph_object_key)
        receipt = run_session(
            args.database, args.run_root, tuple(args.turn_input),
            graph_object_keys=graph_object_keys,
            tenant_id=args.tenant_id, user_id=args.user_id,
            session_id=args.session_id, max_depth=args.max_depth)
    elif args.input:
        receipt = run_slice(
            args.database, args.run_root, args.input,
            tenant_id=args.tenant_id, user_id=args.user_id,
            session_id=args.session_id, max_depth=args.max_depth)
    else:
        parser.error("必须提供 --input 或至少一个 --turn-input")
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL_VERSION", "run_slice", "run_session", "main"]
