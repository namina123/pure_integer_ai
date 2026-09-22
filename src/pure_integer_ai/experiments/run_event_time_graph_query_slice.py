"""以显式 Event/Time 图对象运行同一次三图库 QueryState 切片。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import ROOT_EXPANDED
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)


_SLICE_FORMAT = "PURE_INTEGER_EVENT_TIME_GRAPH_QUERY_SLICE_V1"


def _json_bytes(value: object) -> bytes:
    """以冻结 ASCII JSON 编码宿主 receipt。"""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")


def _sha256(path: Path) -> str:
    """流式计算紧凑 trace 身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_trace(trace: dict[str, object], target: Path) -> str:
    """流式写入确定性 gzip trace，不复制训练模型。"""
    encoder = json.JSONEncoder(
        ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    with target.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for piece in encoder.iterencode(trace):
                stream.write(piece.encode("ascii"))
    return _sha256(target)


def _new_k_root(path: Path) -> Path:
    """只允许在此前不存在的 K 盘非根目录创建会话产物。"""
    root = path.resolve()
    if root.drive.upper() != "K:" or root == Path(root.anchor):
        raise ValueError("event-time query run root 必须是 K 盘非根目录")
    if root.exists():
        raise ValueError("event-time query run root 已存在，禁止覆盖")
    root.mkdir(parents=True)
    return root


def _surface(values: tuple[int, ...]) -> str:
    """从调用方给出的 Unicode scalar 整数恢复输入边界文本。"""
    if (type(values) is not tuple or not values
            or any(type(value) is not int or value < 0 or value > 0x10FFFF
                   or 0xD800 <= value <= 0xDFFF for value in values)):
        raise ValueError("surface_values 必须是非空合法 Unicode scalar tuple")
    return "".join(chr(value) for value in values)


def _object_keys(
        keys: tuple[tuple[int, ...], ...],
        ) -> tuple[tuple[int, ...], ...]:
    """拒绝重复或非整数图对象键，不在 runner 内按序号猜对象。"""
    if (type(keys) is not tuple or not keys
            or any(type(key) is not tuple or not key for key in keys)
            or any(type(value) is not int or value < 0
                   for key in keys for value in key)):
        raise ValueError("graph_object_keys 必须是非空非负整数 tuple 集")
    if len(set(keys)) != len(keys):
        raise ValueError("graph_object_keys 不得重复")
    return tuple(sorted(keys))


def run_slice(
        database: Path,
        run_root: Path,
        *,
        surface_values: tuple[int, ...],
        graph_object_keys: tuple[tuple[int, ...], ...],
        tenant_id: int = 1,
        user_id: int = 1,
        session_id: int = 1,
        max_depth: int = 256,
        ) -> dict[str, object]:
    """运行真实图输入、O/H/E 和三图 frontier，并保存完整可重放 trace。"""
    model = database.resolve(strict=True)
    if model.drive.upper() != "K:" or not model.is_file():
        raise ValueError("只读训练模型必须是 K 盘现有文件")
    root = _new_k_root(run_root)
    session = root / "session.sqlite3"
    values = tuple(surface_values)
    surface = _surface(values)
    input_keys = _object_keys(tuple(graph_object_keys))
    before = model.stat()
    with TrainedGraphQueryBridge(
            model,
            memory_database=session,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            ) as bridge:
        appended = bridge.memory.append(surface, speaker_kind=1)
        observation_ref = bridge.memory.intake.result_for_source(
            appended.source).observation_ref
        if not isinstance(observation_ref, MemoryObjectRef):
            raise RuntimeError("图输入没有建立包外 Memory Observation")
        trace = bridge.query(
            surface,
            graph_object_keys=input_keys,
            input_observation_ref=observation_ref,
            max_depth=max_depth,
        )

        root_spaces = {item["space"] for item in trace["roots"]}
        expanded_spaces = {
            item["space"] for item in trace["roots"]
            if item["status"] == ROOT_EXPANDED}
        evidence_spaces = {item["space"] for item in trace["evidence"]}
        if root_spaces != {1, 2, 3} or expanded_spaces != {1, 2, 3}:
            raise AssertionError("Event/Time query 没有在同一次 QueryState 展开三图 roots")
        if evidence_spaces != {1, 2, 3}:
            raise AssertionError("Event/Time query 没有同轮三图 Evidence")
        event_trace = trace["event_time_graph"]
        if tuple(tuple(item) for item in event_trace["input_keys"]) != input_keys:
            raise AssertionError("Event/Time trace 丢失显式图输入身份")
        expanded = {tuple(item) for item in event_trace["expanded_edges"]}
        if not expanded:
            raise AssertionError("Event/Time graph root 没有展开 topology adjacency")
        structural_evidence = tuple(
            item for item in trace["evidence"]
            if item["space"] == 1 and tuple(item["payload_key"]) in expanded)
        if (not structural_evidence
                or any(item["polarity"] != 3 for item in structural_evidence)):
            raise AssertionError("Event/Time topology 没有保持 UNKNOWN 结构证据")
        if trace["termination_reason"] == "BUDGET_EXHAUSTED":
            raise AssertionError("Event/Time topology 未在给定深度内自然终止")
        trace_sha256 = _write_trace(trace, root / "query-001.json.gz")
        topology_counts = {
            name: len(event_trace["topology"][name])
            for name in (
                "frame_keys", "proposition_keys", "event_state_keys",
                "anchor_keys", "interval_keys", "aspect_keys",
                "revision_keys", "retention_keys", "edges")
        }
    after = model.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AssertionError("只读 Event/Time query 改写了训练模型")

    receipt = {
        "format": _SLICE_FORMAT,
        "schema_version": 1,
        "model_read_only": 1,
        "core_training_performed": 0,
        "free_dialogue_complete": 0,
        "model_path": str(model),
        "model_size": before.st_size,
        "session_path": str(session.resolve()),
        "surface_values": list(values),
        "graph_object_keys": [list(item) for item in input_keys],
        "topology_counts": topology_counts,
        "expanded_topology_edge_count": len(expanded),
        "structural_unknown_evidence_count": len(structural_evidence),
        "root_count": len(trace["roots"]),
        "roots_by_space": [
            sum(item["space"] == space for item in trace["roots"])
            for space in (1, 2, 3)],
        "evidence_by_space": [
            sum(item["space"] == space for item in trace["evidence"])
            for space in (1, 2, 3)],
        "frontier_step_count": len(trace["frontier_trace"]),
        "visited_count": len(trace["visited"]),
        "hop_count": len(trace["hops"]),
        "depth": trace["depth"],
        "termination": trace["termination_reason"],
        "event_time_answer_available": 0,
        "response_surface_values": list(map(ord, trace["response_surface"])),
        "trace": "query-001.json.gz",
        "trace_sha256": trace_sha256,
    }
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def _csv_ints(value: str) -> tuple[int, ...]:
    """解析 CLI 的逗号分隔整数，不赋予任何语言语义。"""
    try:
        result = tuple(int(item, 10) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是逗号分隔整数") from error
    if not result:
        raise argparse.ArgumentTypeError("整数序列不能为空")
    return result


def main() -> int:
    """显式接收只读模型、整数输入、完整图对象键和新的 K 盘根。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--surface-values", type=_csv_ints, required=True)
    parser.add_argument(
        "--graph-object-key", type=_csv_ints, action="append", required=True)
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=256)
    args = parser.parse_args()
    receipt = run_slice(
        args.database,
        args.run_root,
        surface_values=args.surface_values,
        graph_object_keys=tuple(args.graph_object_key),
        tenant_id=args.tenant_id,
        user_id=args.user_id,
        session_id=args.session_id,
        max_depth=args.max_depth,
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
