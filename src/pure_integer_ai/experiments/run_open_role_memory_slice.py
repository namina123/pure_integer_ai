"""只读现役模型，产生开放角色三图会话、完整 trace 与冷恢复 receipt。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.identity import OBJECT_EVENT, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPEN_RELATION, frame_key,
    memory_candidate_category, memory_input_candidate_keys,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRelationCandidate
from pure_integer_ai.cognition.understanding.query_open_role_generation import OpenRoleGenerationContext
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.work_memory_generation_adapter import (
    query_work_memory_projection,
)
from pure_integer_ai.experiments.trained_response_delivery import read_delivery_proof
from pure_integer_ai.experiments.trained_response_context import restore_response_contexts
from pure_integer_ai.experiments.trained_reference_resolution import (
    REFERENCE_CONFLICT,
    REFERENCE_DIMENSION_ENTITY,
    REFERENCE_DIMENSION_EVENT,
    REFERENCE_DIMENSION_PROPERTY,
    REFERENCE_DIMENSION_REFERS,
    REFERENCE_DIMENSION_TIME,
    REFERENCE_NOT_APPLICABLE,
    REFERENCE_OPEN,
    REFERENCE_RESOLVED,
)
from pure_integer_ai.experiments.trained_response_occurrence import (
    RESPONSE_OCCURRENCE_VERSION, materialize_response_roles, read_input_roles,
    read_response_roles,
)


def _json_bytes(value: object) -> bytes:
    """冻结宿主 artifact 的 UTF-8 JSON 编码；语义键仍为完整整数记录。"""
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def _write_json_gzip(value: object, target: Path) -> str:
    """以标准库流式编码完整 trace，避免为巨型 frontier 复制整段 JSON。"""
    encoder = json.JSONEncoder(
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def _snapshot(memory) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """提取全部权威对象引用，确认原始历史没有被覆盖或截断。"""
    state = memory.active_structures()
    return (tuple(sorted(item.observation_key for item in state.observations)),
            tuple(sorted(item.hypothesis_key for item in state.hypotheses)),
            tuple(sorted(item.evidence_key for item in state.evidence)))


def run_slice(database: Path, run_root: Path, inputs: tuple[str, ...],
              source_session: Path | None = None, *, tenant_id: int = 1,
              user_id: int = 1, session_id: int = 1,
              materialize_prior_roles: bool = False,
              reference_mode: bool = False,
              max_nodes: int = 4096, max_edges: int = 8192,
              max_reads: int = 16384, max_depth: int = 64) -> dict[str, object]:
    """复用训练图，在唯一 K 盘目录运行实际 O/H/E 与共同 frontier。

    输入是开发运行观察，不读取或修改评测标签、不训练 Core。只复制指定会话，
    不修改既有模型或原始会话；失败的部分产物保留供定位，禁止覆盖重跑。
    """
    database = database.resolve(strict=True)
    root = run_root.resolve()
    if root.drive.upper() != "K:" or root == Path(root.anchor):
        raise ValueError("运行产物必须显式位于 K 盘子目录")
    if root.exists():
        raise ValueError("运行目录已存在，禁止覆盖")
    if not inputs:
        raise ValueError("必须提供真实输入")
    root.mkdir(parents=True)
    session = root / "session.sqlite3"
    if source_session is not None:
        original = sqlite3.connect(source_session.resolve(strict=True).as_uri() + "?mode=ro", uri=True)
        try:
            copied = sqlite3.connect(str(session))
            try:
                original.backup(copied)
            finally:
                copied.close()
        finally:
            original.close()
    rows = []
    delivery_proofs = []
    delivery_events = []
    occurrence_snapshots = []
    input_occurrence_snapshots = []
    last_snapshot = None
    scope = dict(tenant_id=tenant_id, user_id=user_id, session_id=session_id)
    with TrainedGenerationConnectorRuntime(database) as generator, TrainedGraphQueryBridge(
            database, memory_database=session, surface_generator=generator, **scope) as bridge:
        before = _snapshot(bridge.memory)
        if source_session is not None and not before[0]:
            raise ValueError("指定来源会话在当前 owner/session 中为空，请核实作用域")
        for ordinal, surface in enumerate(inputs, 1):
            projected = bridge.input_projector.project(tuple(map(ord, surface)))
            if not projected.open_relations:
                raise ValueError("输入没有完整图框架解析；不得标记为开放角色通过")
            for candidate in projected.open_relations:
                if OpenRelationCandidate.from_stable_key(candidate.stable_key()) != candidate:
                    raise AssertionError("开放角色整数往返不等")
            input_append = bridge.memory.append(surface, speaker_kind=1)
            input_observation_ref = bridge.memory.intake.result_for_source(input_append.source).observation_ref
            selected = bridge.memory.active_structures(
                candidate_keys=memory_input_candidate_keys(projected), observation_refs=(input_observation_ref,))
            open_hypotheses = tuple(item for item in selected.hypotheses
                                    if memory_candidate_category(item.candidate_key)
                                    == MEMORY_CATEGORY_OPEN_RELATION
                                    and item.observation_key == input_observation_ref.stable_key())
            keys = {item.hypothesis_key for item in open_hypotheses}
            open_evidence = tuple(item for item in selected.evidence if item.hypothesis_key in keys)
            if not open_hypotheses or not open_evidence or any(item.stance != 3 for item in open_evidence):
                raise AssertionError("开放角色没有以 UNKNOWN 保存 O/H/E")
            migrated_roles = 0
            if materialize_prior_roles:
                for prior in restore_response_contexts(bridge.memory, selected):
                    output_source = SourceRef.from_stable_key(prior.source_ref[1:])
                    roles = materialize_response_roles(bridge.memory, prior.context, prior.proof_ref, output_source)
                    occurrence_snapshots.append((prior.context, prior.proof_ref, output_source, roles))
                    migrated_roles += len(roles)
                bridge.memory.backend.commit()
            trace = bridge.query(
                surface, max_depth=max_depth, max_nodes=max_nodes,
                max_edges=max_edges, max_reads=max_reads,
                input_observation_ref=input_observation_ref)
            work_memory_projection = query_work_memory_projection(
                bridge, trace, query_local_id=ordinal)
            trace["work_memory_projection"] = work_memory_projection
            by_candidate = {
                frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,),
                          candidate.stable_key()): candidate
                for candidate in projected.open_relations}
            current_input_roles = []
            for hypothesis in open_hypotheses:
                candidate = by_candidate.get(hypothesis.candidate_key)
                if candidate is None:
                    raise AssertionError("当前开放 Hypothesis 缺少原输入候选")
                hypothesis_ref = MemoryObjectRef.from_stable_key(hypothesis.hypothesis_key)
                roles = read_input_roles(bridge.memory, candidate, input_observation_ref,
                                         hypothesis_ref, required=True)
                if not roles or any(item.query_evidence().polarity != 3 for item in roles):
                    raise AssertionError("输入角色 Occurrence 没有保持 UNKNOWN")
                current_input_roles.extend(roles)
                input_occurrence_snapshots.append(
                    (candidate, input_observation_ref, hypothesis_ref, roles))
            traced_input_roles = {tuple(item) for item in trace["input_role_occurrences"]}
            expected_input_roles = {item.stable_key() for item in current_input_roles}
            if traced_input_roles != expected_input_roles:
                raise AssertionError("Query trace 没有完整记录当前输入角色 Occurrence")
            rooted_input_roles = {
                tuple(root["root"]) for root in trace["roots"]
                if root["space"] == 2 and tuple(root["root"]) in expected_input_roles}
            if rooted_input_roles != expected_input_roles:
                raise AssertionError("当前输入角色没有全部作为同次 Memory roots 展开")
            traced_reference_sets = {tuple(item) for item in trace["reference_candidate_sets"]}
            rooted_reference_sets = {
                tuple(root["root"]) for root in trace["roots"]
                if root["space"] == 2 and tuple(root["root"]) in traced_reference_sets}
            if rooted_reference_sets != traced_reference_sets:
                raise AssertionError("跨来源 reference 候选集合没有全部进入同次 Memory roots")
            resolutions = bridge._reference_resolutions
            if {tuple(item) for item in trace["reference_resolutions"]} != {
                    item.stable_key() for item in resolutions}:
                raise AssertionError("Query trace 没有完整记录跨轮指代裁决")
            status_counts = {
                status: sum(item.status == status for item in resolutions)
                for status in (REFERENCE_NOT_APPLICABLE, REFERENCE_OPEN,
                               REFERENCE_RESOLVED, REFERENCE_CONFLICT)
            }
            dimension_counts = {
                dimension: sum(evidence.dimension == dimension
                               for item in resolutions for evidence in item.evidence)
                for dimension in (REFERENCE_DIMENSION_ENTITY, REFERENCE_DIMENSION_EVENT,
                                  REFERENCE_DIMENSION_PROPERTY, REFERENCE_DIMENSION_TIME,
                                  REFERENCE_DIMENSION_REFERS)
            }
            contexts = tuple(OpenRoleGenerationContext.from_stable_key(tuple(key))
                             for key in trace["open_generation_contexts"])
            context_keys = {item.hypothesis.stable_key() for item in contexts}
            if context_keys != keys and not (
                    reference_mode and context_keys <= keys
                    and all(item.occurrence.stable_key() in expected_input_roles
                            for item in current_input_roles)
                    and resolutions):
                raise AssertionError("开放角色生成上下文没有完整覆盖本次 Memory 假设")
            if trace["response_surface"] and (not trace["generation"]
                    or not trace["generation"].get("representations")):
                raise AssertionError("实际回答没有消费 connector 的图内表示")
            carried_winners = 0
            resolved = tuple(item for item in resolutions
                             if item.status == REFERENCE_RESOLVED)
            if resolved and trace["open_response_generations"]:
                generation_context = {
                    tuple(item) for item in trace["generation"]["discourse_context"]}
                plan_links = {
                    tuple(item) for item in trace["response_plan"]["discourse_links"]}
                plan_events = {
                    tuple(item) for item in trace["response_plan"]["event_refs"]}
                plan_memory = {
                    tuple(item) for item in trace["response_plan"]["memory_refs"]}
                for resolution in resolved:
                    winner = resolution.winner
                    target = resolution.winner_target
                    objects = {item.stable_key()
                               for item in resolution.discourse_objects()}
                    if (winner is None or target is None
                            or not objects <= generation_context
                            or not objects <= plan_links
                            or winner.occurrence.observation.stable_key() not in plan_memory
                            or winner.occurrence.hypothesis.stable_key() not in plan_memory
                            or (target.object_kind == OBJECT_EVENT
                                and target.stable_key() not in plan_events)):
                        raise AssertionError("跨轮指代 winner 未进入 G-02 与 ResponsePlan")
                    carried_winners += 1
            spaces = {item["space"] for item in trace["evidence"]}
            if spaces != {1, 2, 3} or any(item["status"] == 1 for item in trace["roots"]):
                raise AssertionError("开放角色没有完成同次三图根和证据参与")
            delivery = None
            if trace["response_surface"]:
                output_bytes = (trace["response_surface"] + "\n").encode("utf-8")
                with (root / f"response-{ordinal:03d}.txt").open("xb") as stream:
                    if stream.write(output_bytes) != len(output_bytes):
                        raise OSError("实际回应文件未完整写出")
                    stream.flush()
                delivery = bridge.acknowledge_response_delivery(trace["response_surface"])
                if delivery is None:
                    raise AssertionError("实际开放回应未消费当前生成采用")
                trace["delivery"] = delivery.summary()
                delivery_proofs.append((delivery.proof_ref, read_delivery_proof(bridge.memory, delivery.proof_ref)))
                delivery_events.extend((delivery.episode, *delivery.uses, *delivery.outcomes))
                # Generic ResponsePlan delivery intentionally has no
                # source-role context.  Only the open-role receipt exposes
                # the context/occurrence contract needed for later
                # reference candidates; factual and generic receipts are
                # restored through their own delivery proof only.
                if hasattr(delivery, "input_context"):
                    occurrence_snapshots.append((delivery.input_context, delivery.proof_ref,
                                                 delivery.output.source, delivery.role_occurrences))
            trace_name = f"query-{ordinal:03d}.json.gz"
            trace_path = root / trace_name
            trace_sha256 = _write_json_gzip(trace, trace_path)
            rows.append({
                "input": surface,
                "open_candidates": len(projected.open_relations),
                "roles": [[list(binding.values) for binding in candidate.bindings]
                          for candidate in projected.open_relations],
                "open_hypotheses": len(open_hypotheses), "open_evidence": len(open_evidence),
                "open_generation_contexts": len(contexts),
                "generation_role_spans": sum(len(item.candidate.bindings) for item in contexts),
                "observed_surface_proposals": len(trace["observed_surface_proposals"]),
                "open_response_generations": len(trace["open_response_generations"]),
                "response_contexts": len(trace["response_contexts"]),
                # Delivered contexts are polymorphic: open-role contexts
                # retain their generation candidate, while factual and
                # generic contexts retain only their bounded plan proof.
                # This receipt field is diagnostic only; do not assume every
                # restored response context carries an open-role candidate.
                "prior_role_candidates": sum(
                    len(item.context.candidate.bindings)
                    if hasattr(item, "context") else 0
                    for item in bridge._response_contexts),
                "materialized_prior_roles": migrated_roles,
                "prior_role_occurrences": sum(len(item.role_occurrences) for item in bridge._response_contexts),
                "input_role_occurrences": len(current_input_roles),
                "input_occurrence_memory_roots": len(rooted_input_roles),
                "reference_candidate_sets": len(traced_reference_sets),
                "reference_candidate_memory_roots": len(rooted_reference_sets),
                "reference_hot_candidates": sum(item.hot_count for item in bridge._reference_candidate_sets),
                "reference_cold_candidates": sum(item.cold_count for item in bridge._reference_candidate_sets),
                "reference_applicable": sum(item.requires_resolution for item in resolutions),
                "reference_not_applicable": status_counts[REFERENCE_NOT_APPLICABLE],
                "reference_open": status_counts[REFERENCE_OPEN],
                "reference_resolved": status_counts[REFERENCE_RESOLVED],
                "reference_conflict": status_counts[REFERENCE_CONFLICT],
                "reference_evidence_by_dimension": [dimension_counts[item] for item in (
                    REFERENCE_DIMENSION_ENTITY, REFERENCE_DIMENSION_EVENT,
                    REFERENCE_DIMENSION_PROPERTY, REFERENCE_DIMENSION_TIME,
                    REFERENCE_DIMENSION_REFERS)],
                "reference_hot_winners": sum(item.status == REFERENCE_RESOLVED
                                             and item.hot_winner for item in resolutions),
                "reference_cold_winners": sum(item.status == REFERENCE_RESOLVED
                                              and not item.hot_winner for item in resolutions),
                "reference_response_plan_winners": carried_winners,
                "occurrence_memory_roots": sum(root["space"] == 2 and root["root"][0] == RESPONSE_OCCURRENCE_VERSION
                                                 for root in trace["roots"]),
                "generation_discourse_objects": len((trace["generation"] or {}).get("discourse_context", ())),
                "trace": trace_name, "trace_sha256": trace_sha256,
                "evidence_by_space": [sum(item["space"] == space for item in trace["evidence"])
                                      for space in (1, 2, 3)],
                "reason": trace["termination_reason"],
                "response_surface": trace["response_surface"],
                "representation_count": len((trace["generation"] or {}).get("representations", ())),
                "delivery": None if delivery is None else delivery.summary(),
                "work_memory_projection": work_memory_projection,
            })
        last_snapshot = _snapshot(bridge.memory)
        if any(not set(old) <= set(new) for old, new in zip(before, last_snapshot)):
            raise AssertionError("原始 Memory 对象丢失")
        last_selected = selected
    with TrainedGraphQueryBridge(database, memory_database=session, **scope) as bridge:
        if _snapshot(bridge.memory) != last_snapshot:
            raise AssertionError("Memory 全部对象冷恢复不等")
        restored = bridge.memory.active_structures(candidate_keys=memory_input_candidate_keys(
            bridge.input_projector.project(tuple(map(ord, inputs[-1])))),
            observation_refs=(input_observation_ref,))
        if restored != last_selected:
            raise AssertionError("Memory 相关完整内容冷恢复不等")
        for ref, proof in delivery_proofs:
            if read_delivery_proof(bridge.memory, ref) != proof:
                raise AssertionError("完整输出采用证明冷恢复不等")
        for original in delivery_events:
            restored_event = bridge.memory.intake.event_log.read(
                original.event_hash, access=MemoryAccessContext(tenant_id, user_id, session_id))
            if restored_event != original:
                raise AssertionError("输出 Episode/Use/Outcome 冷恢复不等")
        for context, proof_ref, output_source, roles in occurrence_snapshots:
            if read_response_roles(bridge.memory, context, proof_ref, output_source, required=True) != roles:
                raise AssertionError("实际角色 Occurrence 的完整图关系冷恢复不等")
        for candidate, observation, hypothesis, roles in input_occurrence_snapshots:
            if read_input_roles(bridge.memory, candidate, observation, hypothesis,
                                required=True) != roles:
                raise AssertionError("输入角色 Occurrence 的完整图关系冷恢复不等")
    receipt = {
        "schema_version": 1, "model_read_only": 1, "core_training_performed": 0,
        "strict_connector": 1,
        "owner_scope": [tenant_id, user_id, session_id],
        "before_ohe": list(map(len, before)), "after_ohe": list(map(len, last_snapshot)),
        "history_retained": 1, "cold_restore_equal": 1, "queries": rows,
        "delivery_count": len(delivery_proofs), "delivery_event_count": len(delivery_events),
        "role_occurrence_count": sum(len(items) for _, _, _, items in occurrence_snapshots),
        "role_occurrence_cold_restore_equal": 1,
        "input_role_occurrence_count": sum(len(items) for _, _, _, items in input_occurrence_snapshots),
        "input_role_occurrence_cold_restore_equal": 1,
        "reference_candidate_set_count": sum(item["reference_candidate_sets"] for item in rows),
        "reference_candidate_set_cold_restore_equal": 1,
        "reference_resolution_count": sum(
            item["reference_not_applicable"] + item["reference_open"]
            + item["reference_resolved"] + item["reference_conflict"]
            for item in rows),
        "reference_resolved_count": sum(item["reference_resolved"] for item in rows),
        "reference_response_plan_winner_count": sum(
            item["reference_response_plan_winners"] for item in rows),
        "work_memory_projection_count": sum(
            item["work_memory_projection"].get("status", 0) for item in rows),
        "work_memory_projection_item_count": sum(
            item["work_memory_projection"].get("item_count", 0) for item in rows),
        "work_memory_situation_entry_count": sum(
            item["work_memory_projection"].get("entry_count", 0) for item in rows),
        "free_dialogue_complete": 0,
    }
    (root / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main() -> int:
    """显式接收 K 盘运行根和开发输入，不内置问题、答案或语言词表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-session", type=Path)
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--materialize-prior-roles", action="store_true")
    parser.add_argument(
        "--reference-mode", action="store_true",
        help="允许闭合 Core 回答同时携带动态 reference 角色；仍要求完整三图 resolution")
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--max-edges", type=int, default=8192)
    parser.add_argument("--max-reads", type=int, default=16384)
    parser.add_argument("--max-depth", type=int, default=64)
    args = parser.parse_args()
    receipt = run_slice(args.database, args.run_root, tuple(args.input), args.source_session,
                        tenant_id=args.tenant_id, user_id=args.user_id, session_id=args.session_id,
                        materialize_prior_roles=args.materialize_prior_roles,
                        reference_mode=args.reference_mode,
                        max_nodes=args.max_nodes, max_edges=args.max_edges,
                        max_reads=args.max_reads, max_depth=args.max_depth)
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
