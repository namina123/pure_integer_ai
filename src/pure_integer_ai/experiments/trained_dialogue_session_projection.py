"""多轮 DialogueTrainingCase 到统一三图库 QueryState 的整数化运行切片。

课程中的 turn 只作为已许可的交互观察顺序进入包外 Interaction Memory；
用户 turn 才触发一次 ``TrainedGraphQueryBridge.query``。助手期望 turn
不会被当作答案返回，也不会绕过当前 QueryState 的 ResponsePlan。持久化
trace 只保存整数值；V3 课程也只保存整数，运行边界才把一个 turn 临时交给
既有 Memory/connector 字符串 API。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path

from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    session_scope,
)
from pure_integer_ai.experiments.work_memory_generation_adapter import (
    integer_trace_projection as _integer_projection,
    memory_structure_snapshot as _memory_snapshot,
    query_work_memory_projection as _query_work_memory_projection,
)
from pure_integer_ai.experiments.structural_dialogue_course import (
    DIALOGUE_SPEAKER_ASSISTANT,
    DIALOGUE_SPEAKER_USER,
    StructuralDialogueCase,
    load_structural_dialogue_course,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.heldout_response_runtime import (
    HeldoutResponseRuntime,
)


SESSION_PROJECTION_PROTOCOL_V1 = 1


def _json_bytes(value: object) -> bytes:
    """将整数 schema 以稳定 ASCII JSON 写出。"""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    """Hash a model/input file in bounded chunks for the receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_gzip(value: object, target: Path) -> str:
    """流式写出整数 trace，避免构造第二份大字符串。"""
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
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _turn_record(case_index: int, case: StructuralDialogueCase, ordinal: int,
                 role: int, surface_values: tuple[int, ...],
                 observation_ref: MemoryObjectRef | None = None) -> dict[str, object]:
    """构造不含文本的 turn 身份记录。"""
    result: dict[str, object] = {
        "protocol": SESSION_PROJECTION_PROTOCOL_V1,
        "case_index": case_index,
        "case_id_values": list(case.case_id_values),
        "turn_ordinal": ordinal,
        "speaker_role": role,
        "surface_values": list(surface_values),
        "graph_object_keys": [list(key) for key in case.dialogue_turns[ordinal - 1].graph_object_keys],
    }
    if observation_ref is not None:
        result["observation_ref"] = list(observation_ref.stable_key())
    return result


def _structured_cases(pack, max_cases: int | None) -> tuple[StructuralDialogueCase, ...]:
    cases = tuple(
        case for case in pack.cases
        if case.split == "train" and case.dialogue_turns
    )
    if max_cases is not None:
        if type(max_cases) is not int or max_cases <= 0:
            raise ValueError("max_cases 必须是正整数")
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("课程没有可运行的 train dialogue_turns")
    return cases


def run_session_projection(
        database: str | Path,
        run_root: str | Path,
        course_path: str | Path,
        *,
        max_cases: int | None = None,
        tenant_id: int = 1,
        user_id: int = 1,
        session_id: int = 1,
        max_depth: int = 64,
        heldout_course_path: str | Path | None = None,
        heldout_manifest_path: str | Path | None = None,
        ) -> dict[str, object]:
    """在只读 Core 上运行结构化多轮会话并发布整数 trace/receipt。"""
    database = Path(database).resolve(strict=True)
    course_path = Path(course_path).resolve(strict=True)
    if (heldout_course_path is None) != (heldout_manifest_path is None):
        raise ValueError("heldout course and manifest must be supplied together")
    heldout_course = (None if heldout_course_path is None
                      else Path(heldout_course_path).resolve(strict=True))
    heldout_manifest = (None if heldout_manifest_path is None
                        else Path(heldout_manifest_path).resolve(strict=True))
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or root == Path(root.anchor):
        raise ValueError("run_root 必须是 K 盘显式子目录")
    if root.exists():
        raise ValueError("run_root 已存在，禁止覆盖")
    if type(max_depth) is not int or max_depth <= 0:
        raise ValueError("max_depth 必须是正整数")
    for label, value in (("tenant_id", tenant_id), ("user_id", user_id),
                         ("session_id", session_id)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} 必须是正整数")
    root.mkdir(parents=True)
    pack = load_structural_dialogue_course((course_path,))
    cases = _structured_cases(pack, max_cases)
    session = root / "session.sqlite3"
    traces: list[dict[str, object]] = []
    before: tuple[tuple[tuple[int, ...], ...], ...]
    after: tuple[tuple[tuple[int, ...], ...], ...]
    delivery_count = 0
    observed_assistant_count = 0
    queried_user_count = 0
    work_memory_projection_count = 0
    work_memory_projection_item_count = 0
    work_memory_situation_entry_count = 0
    heldout_match_count = 0
    active_spaces_seen: set[int] = set()
    three_graph_query_count = 0
    canonical_model_sha256_before = _sha256_file(database)
    with ExitStack() as stack:
        canonical = stack.enter_context(TrainedGenerationConnectorRuntime(database))
        generator_context = (HeldoutResponseRuntime(
            canonical,
            course_path=heldout_course,
            manifest_path=heldout_manifest,
        ) if heldout_course is not None else nullcontext(canonical))
        generator = stack.enter_context(generator_context)
        bridge = stack.enter_context(TrainedGraphQueryBridge(
            database,
            memory_database=session,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            surface_generator=generator,
        ))
        before = _memory_snapshot(bridge.memory)
        for case_index, case in enumerate(cases, 1):
            for turn in case.dialogue_turns:
                # The model/course state remains integer-only.  This is the
                # single compatibility boundary for the established bridge.
                surface = "".join(map(chr, turn.surface_values))
                if turn.speaker_role == DIALOGUE_SPEAKER_ASSISTANT:
                    # 这是课程提供的外部观察，只用于让下一轮拥有真实的
                    # 交互上下文；绝不把它接到当前回答出口。
                    bridge.memory.append(
                        surface, speaker_kind=DIALOGUE_SPEAKER_ASSISTANT)
                    observed_assistant_count += 1
                    traces.append({
                        "turn": _turn_record(
                            case_index, case, turn.turn_ordinal,
                            turn.speaker_role, turn.surface_values),
                        "queried": 0,
                        "external_observation": 1,
                    })
                    continue
                if turn.speaker_role != DIALOGUE_SPEAKER_USER:
                    raise ValueError("dialogue speaker role 未注册")
                appended = bridge.memory.append(
                    surface, speaker_kind=DIALOGUE_SPEAKER_USER)
                input_ref = bridge.memory.intake.result_for_source(
                    appended.source).observation_ref
                query = bridge.query(
                    surface,
                    max_depth=max_depth,
                    graph_object_keys=turn.graph_object_keys,
                    input_observation_ref=input_ref,
                )
                active_spaces = tuple(query.get("active_spaces", ()))
                if any(type(item) is not int for item in active_spaces):
                    raise RuntimeError("QueryState active_spaces 必须是整数")
                active_spaces_seen.update(active_spaces)
                if set(active_spaces) == {1, 2, 3}:
                    three_graph_query_count += 1
                work_memory_projection = _query_work_memory_projection(
                    bridge, query, query_local_id=queried_user_count + 1)
                work_memory_projection_count += int(
                    work_memory_projection.get("status", 0) == 1)
                work_memory_projection_item_count += int(
                    work_memory_projection.get("item_count", 0))
                work_memory_situation_entry_count += int(
                    work_memory_projection.get("entry_count", 0))
                response_surface = query.get("response_surface", "")
                generation = query.get("generation")
                heldout_match = int(
                    heldout_course is not None
                    and isinstance(generation, dict)
                    and hasattr(generator, "is_heldout_connector")
                    and generator.is_heldout_connector(
                        tuple(generation.get("connector", ()))))
                heldout_match_count += heldout_match
                delivery = None
                if isinstance(response_surface, str) and response_surface:
                    delivery = bridge.acknowledge_response_delivery(response_surface)
                    if delivery is None:
                        raise RuntimeError("ResponsePlan 有表层但没有可采用 delivery")
                    delivery_count += 1
                trace = {
                    "protocol": SESSION_PROJECTION_PROTOCOL_V1,
                    "turn": _turn_record(
                        case_index, case, turn.turn_ordinal,
                        turn.speaker_role, turn.surface_values, input_ref),
                    "queried": 1,
                    "query": _integer_projection({
                        "termination": query.get("termination"),
                        "depth": query.get("depth"),
                        "evidence_closed": query.get("evidence_closed"),
                        "conflict_open": query.get("conflict_open"),
                        "generation_ready": query.get("generation_ready"),
                        "active_spaces": active_spaces,
                        "roots": query.get("roots"),
                        "anchors": query.get("anchors"),
                        "frontier_trace": query.get("frontier_trace"),
                        "bindings": query.get("bindings"),
                        "evidence": query.get("evidence"),
                        "visited": query.get("visited"),
                        "expanded_edges": query.get("expanded_edges"),
                        "response_plan": query.get("response_plan"),
                        "generation": query.get("generation"),
                        "memory_routes": query.get("memory_routes"),
                        "discourse_topic_candidates": query.get(
                            "discourse_topic_candidates"),
                        "discourse_topic_hypothesis_keys": query.get(
                            "discourse_topic_hypothesis_keys"),
                        "discourse_topic_query_projection": query.get(
                            "discourse_topic_query_projection"),
                        # Keep the graph-derived generic selection inputs in
                        # the session trace.  Without these integer fields a
                        # cold recovery cannot distinguish an unmatched
                        # heldout condition from a connector-generation error.
                        "generic_response_feature_mask": query.get(
                            "generic_response_feature_mask", 0),
                        "discourse_relation_feature_state": query.get(
                            "discourse_relation_feature_state", ()),
                        "discourse_topic_graph_identity_keys": query.get(
                            "discourse_topic_graph_identity_keys", ()),
                        "discourse_topic_graph_topology": query.get(
                            "discourse_topic_graph_topology", ()),
                        "event_time_graph_input_keys": query.get(
                            "event_time_graph_input_keys", ()),
                        "discourse_graph_input_requested_keys": query.get(
                            "discourse_graph_input_requested_keys", ()),
                        "discourse_graph_input_keys": query.get(
                            "discourse_graph_input_keys", ()),
                        "event_time_graph_topology": query.get(
                            "event_time_graph_topology", ()),
                        "event_time_graph_expanded_edges": query.get(
                            "event_time_graph_expanded_edges", ()),
                        "event_time_generation_bindings": query.get(
                            "event_time_generation_bindings", ()),
                        "event_time_query_bindings": query.get(
                            "event_time_query_bindings", ()),
                        "response_contexts": query.get(
                            "response_contexts", ()),
                        "delivered_graph_slot_projections": query.get(
                            "delivered_graph_slot_projections", ()),
                        "open_generation_contexts": query.get("open_generation_contexts"),
                        "input_role_occurrences": query.get("input_role_occurrences"),
                        "reference_candidate_sets": query.get("reference_candidate_sets"),
                        "reference_resolutions": query.get("reference_resolutions"),
                        "open_response_generations": query.get("open_response_generations"),
                        "work_memory_projection": work_memory_projection,
                        "heldout_match": heldout_match,
                    }),
                    "delivery": None if delivery is None else _integer_projection(
                        delivery.summary()),
                }
                trace_path = root / f"query-{len(traces) + 1:04d}.json.gz"
                trace_sha256 = _write_json_gzip(trace, trace_path)
                # Hash is receipt metadata, not part of the integer trace.
                trace["trace_sha256_values"] = list(
                    bytes.fromhex(trace_sha256))
                traces.append(trace)
                queried_user_count += 1
        after = _memory_snapshot(bridge.memory)
    canonical_model_sha256_after = _sha256_file(database)
    if not all(set(old) <= set(new) for old, new in zip(before, after)):
        raise AssertionError("多轮 session 丢失已有 Memory O/H/E")
    with TrainedGraphQueryBridge(
            database,
            memory_database=session,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
    ) as restored_bridge:
        restored = _memory_snapshot(restored_bridge.memory)
    if restored != after:
        raise AssertionError("多轮 session SQLite 冷恢复 O/H/E 不等")
    receipt = {
        "protocol": SESSION_PROJECTION_PROTOCOL_V1,
        "course_sha256_values": list(bytes.fromhex(pack.pack_sha256)),
        "case_count": len(cases),
        "turn_count": sum(len(case.dialogue_turns) for case in cases),
        "queried_user_count": queried_user_count,
        "observed_assistant_count": observed_assistant_count,
        "delivery_count": delivery_count,
        "work_memory_projection_count": work_memory_projection_count,
        "work_memory_projection_item_count": work_memory_projection_item_count,
        "work_memory_situation_entry_count": work_memory_situation_entry_count,
        "before_ohe": [len(item) for item in before],
        "after_ohe": [len(item) for item in after],
        "trace_count": len(traces),
        "integer_trace": 1,
        "cold_restore_equal": 1,
        "model_read_only": 1,
        "canonical_model_sha256_before": canonical_model_sha256_before,
        "canonical_model_sha256_after": canonical_model_sha256_after,
        "database_copy_count": 0,
        "source_text_rows_added": 0,
        "source_body_answer_route": 0,
        "successor_answer_route": 0,
        "character_nearest_route": 0,
        "opencc_route": 0,
        "language_vocabulary_route": 0,
        "active_spaces": sorted(active_spaces_seen),
        "three_graph_query_count": three_graph_query_count,
        "ohe_observation_count": len(after[0]),
        "ohe_hypothesis_count": len(after[1]),
        "ohe_evidence_count": len(after[2]),
        "free_dialogue_complete": 0,
        "heldout_consumption": int(heldout_course is not None),
        "heldout_course_sha256": (None if heldout_course is None else
                                   hashlib.sha256(heldout_course.read_bytes()).hexdigest()),
        "heldout_manifest_sha256": (None if heldout_manifest is None else
                                   hashlib.sha256(heldout_manifest.read_bytes()).hexdigest()),
        "heldout_graph_role_categories": (
            [] if heldout_course is None else
            list(generator.manifest.get("graph_role_categories", ()))
        ),
        "heldout_whole_sentence_representation_count": (
            0 if heldout_course is None else
            int(generator.manifest.get(
                "whole_sentence_representation_count", 0))
        ),
        "heldout_minimum_graph_slots_per_variant": (
            0 if heldout_course is None else
            int(generator.manifest.get(
                "minimum_graph_slots_per_variant", 1))
        ),
        "heldout_relation_qualified_graph_slots": (
            0 if heldout_course is None else
            int(generator.manifest.get(
                "relation_qualified_graph_slots", 0))
        ),
        "semantic_variant_registry_normalized": int(
            heldout_course is not None
            and generator.manifest.get("format")
            == "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2"
        ),
        "heldout_match_count": heldout_match_count,
        "heldout_unmatched_count": queried_user_count - heldout_match_count,
    }
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-id", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--heldout-course", type=Path)
    parser.add_argument("--heldout-manifest", type=Path)
    args = parser.parse_args(argv)
    receipt = run_session_projection(
        args.database, args.run_root, args.course,
        max_cases=args.max_cases,
        tenant_id=args.tenant_id,
        user_id=args.user_id,
        session_id=args.session_id,
        max_depth=args.max_depth,
        heldout_course_path=args.heldout_course,
        heldout_manifest_path=args.heldout_manifest,
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SESSION_PROJECTION_PROTOCOL_V1", "run_session_projection", "main"]
