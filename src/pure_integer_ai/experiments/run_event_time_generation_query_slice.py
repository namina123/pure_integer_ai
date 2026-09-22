"""用普通整数文本执行 Event/Time 绑定、三图查询和既有 connector 生成。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED,
    TERMINATION_ANSWER_CLOSED,
)
from pure_integer_ai.experiments.trained_event_time_generation_binding import (
    GENERATION_BINDING_NAMESPACE,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)


_FORMAT = "PURE_INTEGER_EVENT_TIME_GENERATION_QUERY_V1"


def _json_bytes(value: object) -> bytes:
    """返回规范 ASCII JSON。"""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _surface(values: tuple[int, ...]) -> str:
    """只在输入边界把调用方整数恢复为 Unicode scalar。"""
    if (type(values) is not tuple or not values
            or any(type(value) is not int or value < 0 or value > 0x10FFFF
                   or 0xD800 <= value <= 0xDFFF for value in values)):
        raise ValueError("surface_values 必须是合法非空 Unicode scalar tuple")
    return "".join(chr(value) for value in values)


def _new_root(path: Path) -> Path:
    """只创建此前不存在的 K 盘运行目录。"""
    root = path.resolve()
    if root.drive.upper() != "K:" or root == Path(root.anchor) or root.exists():
        raise ValueError("Event/Time query root 必须是新的 K 盘非根目录")
    root.mkdir(parents=True)
    return root


def _write_trace(path: Path, trace: dict[str, object]) -> str:
    """流式写入确定 gzip trace，避免复制模型或来源正文。"""
    encoder = json.JSONEncoder(
        ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for piece in encoder.iterencode(trace):
                stream.write(piece.encode("ascii"))
    return _sha256(path)


def run_slice(
        database: Path,
        run_root: Path,
        *,
        surface_values: tuple[int, ...],
        tenant_id: int = 1,
        user_id: int = 1,
        session_id: int = 1,
        max_depth: int = 256,
        ) -> dict[str, object]:
    """运行普通输入到 Event/Time frame 和 ResponsePlan 的完整只读链。"""
    model = database.resolve(strict=True)
    if model.drive.upper() != "K:" or not model.is_file():
        raise ValueError("Event/Time query model 必须是现有 K 盘 SQLite")
    root = _new_root(run_root)
    session = root / "session.sqlite3"
    values = tuple(surface_values)
    surface = _surface(values)
    before = (model.stat().st_size, model.stat().st_mtime_ns, _sha256(model))
    with TrainedGenerationConnectorRuntime(model) as connector, \
            TrainedGraphQueryBridge(
                model,
                memory_database=session,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                surface_generator=connector,
            ) as bridge:
        appended = bridge.memory.append(surface, speaker_kind=1)
        observation_ref = bridge.memory.intake.result_for_source(
            appended.source).observation_ref
        if not isinstance(observation_ref, MemoryObjectRef):
            raise RuntimeError("Event/Time 普通输入没有建立包外 Observation")
        trace = bridge.query(
            surface,
            input_observation_ref=observation_ref,
            max_depth=max_depth,
        )
    after = (model.stat().st_size, model.stat().st_mtime_ns, _sha256(model))
    if before != after:
        raise AssertionError("Event/Time generation query 改写了训练模型")
    if trace["termination"] != TERMINATION_ANSWER_CLOSED:
        raise AssertionError(
            f"Event/Time generation query 未闭合: {trace['termination_reason']}")
    root_spaces = {item["space"] for item in trace["roots"]}
    expanded_spaces = {item["space"] for item in trace["roots"]
                       if item["status"] == ROOT_EXPANDED}
    evidence_spaces = {item["space"] for item in trace["evidence"]}
    if (root_spaces != {1, 2, 3} or expanded_spaces != {1, 2, 3}
            or evidence_spaces != {1, 2, 3}):
        raise AssertionError("Event/Time generation query 未同次展开三图库")
    event_trace = trace["event_time_graph"]
    query_bindings = event_trace["query_generation_bindings"]
    if not query_bindings or event_trace["input_keys"]:
        raise AssertionError("普通输入没有唯一走训练后 Event/Time binding")
    response_plan = trace["response_plan"]
    generation = trace["generation"]
    if (not response_plan or not trace["response_surface"]
            or not generation or not generation.get("structure_plan")):
        raise AssertionError("Event/Time 回答未消费既有结构生成计划")
    expected_events = {
        tuple(key) for binding in query_bindings for key in binding["events"]}
    actual_events = {tuple(key) for key in response_plan["event_refs"]}
    if not expected_events <= actual_events:
        raise AssertionError("ResponsePlan 丢失 Event refs")
    if GENERATION_BINDING_NAMESPACE not in response_plan["scope_and_time"]:
        raise AssertionError("ResponsePlan 丢失 Event/Time scope_and_time")
    binding_edges = {
        tuple(key) for binding in query_bindings
        for key in binding["topology_edges"]}
    expanded_edges = {tuple(key) for key in event_trace["expanded_edges"]}
    if not binding_edges <= expanded_edges:
        raise AssertionError("Event/Time generation frame 未在共同 frontier 完整展开")
    trace_path = root / "query-001.json.gz"
    trace_sha = _write_trace(trace_path, trace)
    receipt = {
        "format": _FORMAT,
        "schema_version": 1,
        "model_read_only": 1,
        "core_training_performed": 0,
        "successor_answer_route": 0,
        "source_body_answer_route": 0,
        "free_dialogue_complete": 0,
        "model_path": str(model),
        "model_size": before[0],
        "model_sha256": before[2],
        "session_path": str(session.resolve()),
        "surface_values": list(values),
        "termination": trace["termination_reason"],
        "depth": trace["depth"],
        "root_count": len(trace["roots"]),
        "roots_by_space": [
            sum(item["space"] == space for item in trace["roots"])
            for space in (1, 2, 3)],
        "evidence_by_space": [
            sum(item["space"] == space for item in trace["evidence"])
            for space in (1, 2, 3)],
        "query_generation_binding_count": len(query_bindings),
        "expanded_event_time_edge_count": len(expanded_edges),
        "event_ref_count": len(actual_events),
        "anchor_ref_count": sum(len(item["anchors"]) for item in query_bindings),
        "interval_ref_count": sum(len(item["intervals"]) for item in query_bindings),
        "aspect_ref_count": sum(len(item["aspects"]) for item in query_bindings),
        "revision_ref_count": sum(len(item["revisions"]) for item in query_bindings),
        "response_surface_values": list(map(ord, trace["response_surface"])),
        "generation_structure_plan_present": 1,
        "trace": trace_path.name,
        "trace_sha256": trace_sha,
    }
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def _csv_ints(value: str) -> tuple[int, ...]:
    """解析逗号分隔整数，不赋予语言含义。"""
    try:
        result = tuple(int(item, 10) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是逗号分隔整数") from error
    if not result:
        raise argparse.ArgumentTypeError("整数序列不能为空")
    return result


def main(argv: list[str] | None = None) -> int:
    """执行一个新 K 盘只读模型查询切片。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--surface-values", required=True, type=_csv_ints)
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=256)
    args = parser.parse_args(argv)
    result = run_slice(**vars(args))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_slice"]
