"""普通自然输入的有界真实查询、当前 O/H/E 和同次交付捕获。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.query_state import TERMINATION_ANSWER_CLOSED
from pure_integer_ai.experiments.batch_case_provenance import capture_factual_case_provenance
from pure_integer_ai.experiments.batch_case_storage import build_compact_case_provenance
from pure_integer_ai.experiments.batch_phase_host import current_process_resource_endpoint
from pure_integer_ai.experiments.batch_trace_storage import save_case_trace_snapshot
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.relation_resource_budget import (
    RelationResourceLimits, before_resource_write, bounded_relation_resources, resource_checkpoint,
)


def file_sha256(path: Path) -> str:
    """流式读取 artifact 身份，不将正文送入查询器。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_preflight_json(root: Path, relative_path: str, value: object) -> None:
    """仅向已解析的显式 K root 写规范 receipt/trace，拒绝路径逃逸。"""
    root = root.resolve(strict=True)
    target = (root / relative_path).resolve()
    if root.drive.upper() != "K:" or target == root or not target.is_relative_to(root):
        raise ValueError("preflight outputs require the declared K run root")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    before_resource_write(target, len(raw))
    target.write_bytes(raw)


def projection_summary(structure) -> dict[str, object]:
    """记录当前问式/角色资格；投影不等于 QueryState 行为通过。"""
    return {
        "operation_keys": [list(key) for key in structure.operation_keys],
        "question_bindings": [list(item.stable_key())
                              for item in structure.relation_question_bindings],
        "question_frame_versions": [item.frame_key[0]
                                    for item in structure.relation_question_bindings],
        "operation_request_bindings": [list(item.stable_key())
                                       for item in structure.operation_request_bindings],
        "closed_relation_count": sum(item.state == 2 for item in structure.relation_candidates),
        "open_relation_count": len(structure.open_relations),
    }


def _delivery_counts(delivery) -> tuple[int, int]:
    """按三种实际收据恢复计数；非事实不能借缺省属性获得事实资格。"""
    from pure_integer_ai.experiments.trained_response_delivery import (
        FactualResponseDeliveryReceipt, GenericResponseDeliveryReceipt, ResponseDeliveryReceipt,
    )
    if delivery is None:
        return 0, 0
    if isinstance(delivery, FactualResponseDeliveryReceipt):
        return delivery.claim_count, len(delivery.role_occurrences)
    if isinstance(delivery, GenericResponseDeliveryReceipt):
        return 0, 0
    if isinstance(delivery, ResponseDeliveryReceipt):
        return 0, len(delivery.role_occurrences)
    raise TypeError("自然调用收到未注册的交付收据类型")


def _run_natural_relation_case(
        bridge, surface: str, root: Path, relative_root: str, *,
        max_depth: int, max_nodes: int, max_edges: int, max_reads: int,
        compress_trace: bool = False,
        ) -> dict[str, object]:
    """以普通文本进入同次 O/H/E→三图→Plan→生成→delivery；不接收目标或标签。"""
    started = current_process_resource_endpoint()
    structure = bridge.prepare_input_structure(surface)
    memory_structure = bridge.active_memory_input_structure(structure)
    revision_binding = bridge.prepare_natural_revision_binding(
        structure, memory_input_structure=memory_structure)
    revision_target = None
    if revision_binding is not None:
        # Resolve before the first write.  Ambiguous, missing, or competing
        # targets therefore leave the current correction completely absent
        # from Interaction Memory.
        revision_target = bridge.resolve_natural_revision_target(
            structure, revision_binding)
    appended = bridge.memory.append(
        surface, speaker_kind=1,
        input_structure=memory_structure)
    revision_commit = None
    if revision_target is not None:
        revision_commit = bridge.commit_natural_revision(
            revision_target, appended.source)
    intake = bridge.memory.intake.result_for_source(appended.source)
    result = bridge.query(
        surface, input_structure=structure, input_observation_ref=intake.observation_ref,
        compact_trace=True, max_depth=max_depth, max_nodes=max_nodes,
        max_edges=max_edges, max_reads=max_reads)
    trace_ref, trace_snapshot = save_case_trace_snapshot(root / relative_root, result, compressed=compress_trace,
        checkpoint=lambda: resource_checkpoint("trace_encode_block", retain=False),
        before_write=before_resource_write)
    trace_bytes = trace_snapshot.raw
    trace = GraphQueryTraceEvidence(result)
    counts = trace.counts()
    output = result.get("response_surface")
    deliverable = (bool(result.get("response_plan")) and bool(result.get("generation"))
                   and isinstance(output, str) and bool(output))
    # 非事实也必须提交同次生成的实际输出；桥接继续核验完整pending和交付门。
    # 事实capture仍只属于ANSWER_CLOSED，不给零claim回应附加事实资格。
    delivery = bridge.acknowledge_response_delivery(output) if deliverable else None
    capture = None
    closed = (result.get("termination") == TERMINATION_ANSWER_CLOSED
              and deliverable)
    if closed:
        if delivery is not None and result["generation"].get("kind") == 6:
            from pure_integer_ai.cognition.shared.integer_operation_execution import OperationExecution
            from pure_integer_ai.experiments.operation_program_query import operation_plan_trace_matches
            encoded = result["generation"].get("operation_execution")
            if not isinstance(encoded, dict):
                raise ValueError("operation delivery lacks the original full program proof")
            execution = OperationExecution.from_stable_key(trace.resolve_ref(encoded["integer_key_ref"]))
            primary = trace.resolve_ref(result["state_key"]["scalars"][11], "operation selected claim")
            if (execution.query_key != trace.resolve_ref(result["query_ref"])
                    or structure.operation_keys != (execution.witness.program.operation_key,)
                    or not execution.outputs or primary != execution.outputs[0][0]
                    or not operation_plan_trace_matches(execution, result["response_plan"], trace=trace)):
                raise ValueError("operation delivery differs from the actual input/selected claims")
            capture = capture_factual_case_provenance(
                query_result=result, query_values=tuple(map(ord, surface)), expected_claim=primary,
                expected_operation=execution.witness.program.operation_key,
                input_observation=intake.observation_ref.stable_key(), input_source=appended.source.stable_key(),
                delivery_summary=delivery.summary())
        elif delivery is not None and structure.relation_question_bindings:
            from pure_integer_ai.experiments.natural_question_capture import delivered_question_binding
            binding = delivered_question_binding(structure, result)
            relation = next(item for item in structure.relation_candidates
                            if item.stable_key() == binding.relation_key)
            capture = capture_factual_case_provenance(
                query_result=result, query_values=tuple(map(ord, surface)),
                expected_claim=relation.proposition_key,
                expected_operation=binding.operation_key,
                input_observation=intake.observation_ref.stable_key(),
                input_source=appended.source.stable_key(),
                delivery_summary=delivery.summary())
        elif delivery is not None and structure.pending_relation_composition_bindings:
            from pure_integer_ai.cognition.shared.relation_composition_verifier import VerifiedRelationComposition
            bindings = structure.pending_relation_composition_bindings
            encoded = result["generation"].get("verified_composition")
            if len(bindings) != 1 or not isinstance(encoded, dict):
                raise ValueError("derived delivery lacks one original question/proof binding")
            verified = VerifiedRelationComposition.from_stable_key(
                trace.resolve_ref(encoded["integer_key_ref"]))
            if verified.pending.stable_key() != bindings[0].pending_key:
                raise ValueError("derived delivery proof differs from the original input binding")
            capture = capture_factual_case_provenance(
                query_result=result, query_values=tuple(map(ord, surface)),
                expected_claim=verified.claim.stable_key(), expected_operation=bindings[0].operation_key,
                input_observation=intake.observation_ref.stable_key(),
                input_source=appended.source.stable_key(), delivery_summary=delivery.summary())
    if capture is not None:
        compact = build_compact_case_provenance(capture, trace_ref=trace_ref,
            trace_root=root / relative_root, trace_bytes=trace_bytes,
            validated_trace=trace_snapshot)
        write_preflight_json(root, relative_root + "/case_provenance.json", compact)
    ended = current_process_resource_endpoint()
    from pure_integer_ai.experiments.trained_reference_resolution import (
        REFERENCE_CONFLICT, REFERENCE_OPEN, REFERENCE_RESOLVED,
    )
    response_contexts = tuple(getattr(bridge, "_response_contexts", ()))
    reference_candidate_sets = tuple(getattr(bridge, "_reference_candidate_sets", ()))
    reference_resolutions = tuple(getattr(bridge, "_reference_resolutions", ()))
    delivery_claim_count, delivery_role_count = _delivery_counts(delivery)
    report = {
        "projection": projection_summary(structure),
        "input_values": list(map(ord, surface)),
        "observation_ref": list(intake.observation_ref.stable_key()),
        "input_source": list(appended.source.stable_key()),
        "revision_binding": (
            None if revision_binding is None
            else list(revision_binding.stable_key())),
        "revision_target": (
            None if revision_target is None
            else list(revision_target.stable_key())),
        "revision_commit": (
            None if revision_commit is None
            else list(revision_commit.stable_key())),
        "revision_binding_detail": (
            None if revision_binding is None else {
                "cue_identity": list(revision_binding.cue_identity),
                "action_key": list(revision_binding.action_key),
                "proposition_key": list(revision_binding.proposition_key),
                "relation_key": list(revision_binding.relation_key),
                "input_source_ref": list(revision_binding.input_source_ref),
                "cue_interval": [revision_binding.cue_start,
                                 revision_binding.cue_end],
                "source_identity": list(revision_binding.source_identity),
                "license_identity": list(revision_binding.license_identity),
            }),
        "revision_target_detail": (
            None if revision_target is None else {
                "prior_observation_key": list(
                    revision_target.prior_observation_key),
                "prior_hypothesis_key": list(
                    revision_target.prior_hypothesis_key),
                "prior_evidence_key": list(
                    revision_target.prior_evidence_key),
                "prior_source_ref": list(revision_target.prior_source_ref),
            }),
        "revision_commit_detail": (
            None if revision_commit is None else {
                "correction_source_ref": list(
                    revision_commit.correction_source_ref),
                "replacement_evidence_key": list(
                    revision_commit.replacement_evidence_key),
            }),
        "hypothesis_count": len(intake.hypothesis_refs),
        "evidence_count": len(intake.evidence_refs),
        "hypothesis_refs": [list(ref.stable_key()) for ref in intake.hypothesis_refs],
        "evidence_refs": [list(ref.stable_key()) for ref in intake.evidence_refs],
        "consumed_operation_keys": result.get("operation_keys", []),
        "termination": result.get("termination"),
        "trace_counts": counts,
        "response_plan_present": int(bool(result.get("response_plan"))),
        "generation_present": int(bool(result.get("generation"))),
        "response_surface": output,
        "delivery_claim_count": delivery_claim_count,
        "delivery_role_occurrence_count": delivery_role_count,
        "delivery_present": int(delivery is not None),
        "delivery_proof_ref": (
            None if delivery is None else list(delivery.proof_ref.stable_key())),
        "delivery_summary": None if delivery is None else delivery.summary(),
        "response_context_count": len(response_contexts),
        "response_role_occurrence_count": sum(
            len(item.role_occurrences) for item in response_contexts),
        "input_role_occurrence_count": len(getattr(
            bridge, "_input_role_occurrences", ())),
        "reference_candidate_set_count": len(reference_candidate_sets),
        "reference_resolution_count": len(reference_resolutions),
        "reference_required_count": sum(
            item.requires_resolution for item in reference_resolutions),
        "reference_open_count": sum(
            item.status == REFERENCE_OPEN for item in reference_resolutions),
        "reference_resolved_count": sum(
            item.status == REFERENCE_RESOLVED for item in reference_resolutions),
        "reference_conflict_count": sum(
            item.status == REFERENCE_CONFLICT for item in reference_resolutions),
        "capture_present": int(capture is not None),
        "trace_sha256": trace_ref["sha256"],
        "trace_byte_count": trace_ref["byte_count"],
        "trace_stored_sha256": trace_ref.get("stored_sha256", trace_ref["sha256"]),
        "trace_stored_byte_count": trace_ref.get("stored_byte_count", trace_ref["byte_count"]),
        "trace_storage_encoding": trace_ref.get("encoding", "CANONICAL_JSON_LF"),
        "trace_descriptor_sha256": file_sha256(root / relative_root / "trace.int.json"),
        "resource_started": started.to_value(), "resource_ended": ended.to_value(),
        "wall_ns": ended.sampled_ended_ns - started.sampled_started_ns,
        "cpu_ns": ended.cpu_ns - started.cpu_ns,
        "peak_scope": "PROCESS_CUMULATIVE_NOT_CASE_PEAK",
        "held_out_count": 0,
        "formal_wave2_denominator_count": 0,
        "production_authorized": 0,
        "free_dialogue_complete": 0,
        "evaluator_read_count": 0,
    }
    write_preflight_json(root, relative_root + "/call_receipt.json", report)
    return report


def run_natural_relation_case(
        bridge, surface: str, root: Path, relative_root: str, *,
        max_depth: int, max_nodes: int, max_edges: int, max_reads: int,
        max_case_wall_seconds: int = 600,
        case_ordinal: int = 0,
        compress_trace: bool = False,
        resource_limits: RelationResourceLimits | None = None,
        ) -> dict[str, object]:
    """同一次自然输入接实际 phase caller 与被动硬上限，缺调用点不补造覆盖。"""
    from pure_integer_ai.experiments.relation_case_budget import passive_relation_case_budget
    from pure_integer_ai.experiments.relation_runtime_phase import capture_relation_phases

    root = root.resolve(strict=True)
    case = (root / relative_root).resolve()
    if root.drive.upper() != "K:" or case == root or not case.is_relative_to(root):
        raise ValueError("case outputs must remain within the declared K root")
    case.mkdir(parents=True, exist_ok=False)
    with passive_relation_case_budget(case / "wall_budget", max_wall_seconds=max_case_wall_seconds):
        with bounded_relation_resources(root, case / "resource_budget", bridge.memory, resource_limits) as resources:
            with capture_relation_phases(
                    case / "phase_attempt", run_identity=root.name,
                    attempt_key=relative_root, case_ordinal=case_ordinal,
                    required_phases=("materialization", "memory_append", "memory_recover", "query")) as phases:
                report = _run_natural_relation_case(
                    bridge, surface, root, relative_root,
                    max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges, max_reads=max_reads,
                    compress_trace=compress_trace)
                if resources is not None:
                    resources.commit()
            report["phase_receipt_sha256"] = file_sha256(case / "phase_attempt/receipt.json")
            report["phase_boundary_status"] = phases.report["status"]
            report["positive_chain_phase_coverage"] = phases.report["positive_chain_phase_coverage"]
            report["max_case_wall_seconds"] = max_case_wall_seconds
            write_preflight_json(root, relative_root + "/call_receipt.json", report)
        if resources is not None:
            report["resource_budget_sha256"] = file_sha256(case / "resource_budget/receipt.json")
            write_preflight_json(root, relative_root + "/call_receipt.json", report)
    return report


__all__ = ["file_sha256", "projection_summary", "run_natural_relation_case", "write_preflight_json"]
