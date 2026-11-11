"""冷核对原生多 claim 普通调用；不执行 QueryBridge、G planner 或任何写入。"""
from __future__ import annotations

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

from dataclasses import replace
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.integer_operation_execution import OperationExecution
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.typed_binding import BoundRoleBinding
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.cognition.shared.generation_observed_surface import delivered_graph_surface_role
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.experiments.connected_relation_question_course import compile_train_relation_question
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_connected_course import _FACT_FIELDS, _read_lines, _seed
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256
from pure_integer_ai.experiments.operation_composition_course import compile_train_operation_composition
from pure_integer_ai.experiments.operation_composition_materialization import read_native_operation_composition_graph
from pure_integer_ai.experiments.operation_realization_contract import require_native_operation_role_order
from pure_integer_ai.experiments.operation_program_call_audit import restore_saved_operation_question, verify_removed_operation_trace
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_program_query import (
    OPERATION_EXECUTION_ROLE, _request_variables, operation_evidence_matches,
    operation_plan_trace_matches, read_operation_core_context,
)
from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.saved_relation_memory_audit import audit_saved_relation_memory
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime, _bound_relation_proposition, _generation_protocols, _slot_ordinal,
)
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_multi_claim_composer import GRAPH_KIND, PART_CLAIM_PREDICATE, PART_CLAIM_ROLE
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_operation_program_graph import read_operation_program_graph
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_question_graph import read_relation_question_graph


def _json(path):
    """只读取已保存宿主 receipt 或训练交换记录。"""
    return json.loads(path.read_bytes())


def _hashes(root):
    """冻结小型原始调用目录的完整物理身份。"""
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _read_sequence(generator, bound, template):
    """从冷 S-07/R-01 读取每个原生槽的唯一 Representation，不重跑生成。"""
    recovered = generator.connector_for(bound, branch=template.language_branch,
                                        connector_identity=template.connector)
    if any(binding.source == recovered.connector.registry.value_protocol.proposition_source
           for binding in template.bindings):
        raise ValueError("native grammar contains a whole-proposition replay source")
    selected = {}
    for slot in recovered.connector.registry.values(template, bound):
        route = recovered.surface_for(slot.filler)
        if route.result.selected is None or len(route.result.options) != 1:
            raise ValueError("native grammar lacks a unique cold slot realization")
        selected[slot.slot] = route.result.selected.value
    return tuple(selected[slot.slot] for slot in sorted(template.slots,
                                                       key=lambda item: _slot_ordinal(item.slot)))


def verify_prior_delivery_slot_projections(value, prior):
    """将历史 UNKNOWN 槽逐项连到已独立核验的 prior proof/Plan 及同次输出 O/H/E。"""
    trace = GraphQueryTraceEvidence(value)
    state = value["state_key"]
    entries = []
    for ref in state["evidence"]:
        key = trace.resolve_ref(ref)
        fields = _unpack(key, 3, 6)
        entries.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
    prior_trace = GraphQueryTraceEvidence.from_integer_artifact(prior["trace"])
    plan = _decode_input(prior["trace"]["response_plan"], prior_trace)
    targets = {tuple(slot["filler"]) for slot in plan["slot_sequence"]}
    proof = tuple(prior["delivery_proof_ref"])
    output_source = SourceRef.from_stable_key(tuple(prior["delivery_output_source"]))
    scope = document_scope(output_source).stable_key()
    bindings = {trace.resolve_ref(ref) for ref in state["bindings"]}
    count = 0
    for entry in entries:
        if entry.space != 2 or entry.evidence_key[:2] != (91572, 1):
            continue
        fields = unpack_record(entry.evidence_key, 91572)
        if len(fields) != 6 or len(fields[0]) != 1 or len(fields[5]) != 1:
            raise ValueError("prior delivered slot lacks its complete structural graph key")
        category, target, selected_proof, source, selected_scope, _depth = fields
        expected_category = {16: 1, 17: 2, 7: 4, 4: 5}.get(target[0]) if target else None
        if (target not in targets or category != (expected_category,) or selected_proof != proof
                or source != entry.source_ref or source[1:] != output_source.stable_key()
                or selected_scope != scope or entry.scope_key != scope
                or entry.polarity != 3 or entry.trust != 1 or entry.supersedes_key
                or entry.payload_key != pack_record(91572, proof, target)
                or not any(item.space == 2 and item.evidence_key[:1] == (1,)
                    and item.hypothesis_key == entry.hypothesis_key and item.source_ref == source
                    for item in entries)
                or BindingEntry(delivered_graph_surface_role(category[0], ObjectIdentity.from_stable_key(proof)),
                    target, space=2, scope_key=scope).stable_key() not in bindings):
            raise ValueError("prior delivered slot is not the audited proof target/current output OHE")
        count += 1
    return {"prior_delivered_graph_slot_count": count,
            "prior_delivery_slot_join": "MATCHED_PRIOR_AUDITED_PROOF_PLAN_AND_CURRENT_OUTPUT_OHE",
            "prior_context_depth_qualification": "UNKNOWN_NOT_DISTANCE_AUDITED",
            "prior_delivery_is_new_claim_support": 0}


def verify_native_plan_components(core, generator, execution, value, trace, encoded):
    """冷核当前 Plan 与每个原子生成的角色、来源和表示；不执行生成规划器。"""
    if encoded["kind"] != 6 or encoded["operation_truth_added"] != 0:
        raise ValueError("original native generation changed the operation truth contract")
    surfaces = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
    chosen = tuple(surfaces[key] for key, _roles in execution.outputs)
    if len(encoded["claim_generations"]) != len(chosen):
        raise ValueError("native generation lost an actual claim or claim-local role")
    plan = _decode_input(value["response_plan"], trace)
    if not operation_plan_trace_matches(execution, plan):
        raise ValueError("original native Plan scope or output order drifted")
    ranges = {tuple(row["claim"]): (row["start"], row["end"]) for row in plan["claim_slot_ranges"]}
    sources = []
    for (key, roles), fact, actual in zip(execution.outputs, chosen, encoded["claim_generations"]):
        by_role = {binding.role.stable_key(): binding for binding in fact.bindings}
        start, end = ranges[key]
        if tuple((tuple(slot["role"]), tuple(slot["filler"]), slot["source_hash"])
                 for slot in plan["slot_sequence"][start:end]) != tuple(
                     (role, by_role[role].filler.stable_key(), fact.source_hash) for role in roles):
            raise ValueError("original native Plan role/filler/source differs from the Core claim")
        source = core.generation_input(fact.proposition)
        sources.append(source)
        template = generator.generation_template_for(fact)
        protocols = _generation_protocols(generator.context, template.language_branch)
        representations = _read_sequence(generator,
            _bound_relation_proposition(source, template, protocols.content.answer), template)
        if (tuple(actual["claim"]) != key or tuple(actual["connector"]) != template.connector.stable_key()
                or tuple(map(tuple, actual["representations"])) != tuple(item.stable_key() for item in representations)):
            raise ValueError("original component generation differs from cold native role atoms")
    return chosen, tuple(sources), plan


def read_native_member_representations(generator, outputs, chosen, sources, variant, witness):
    """按完整物理 witness 冷读一个原生语法，核对有序 predicate 与全部 scoped roles。"""
    guards = tuple(part[2] for part in sorted((part for part in variant.parts
        if part[0] == PART_CLAIM_PREDICATE), key=lambda part: part[1]))
    coordinates = tuple((part[1], part[2]) for part in variant.parts if part[0] == PART_CLAIM_ROLE)
    if (variant.variant_id != witness.variant_id or variant.claim_count != len(chosen)
            or guards != tuple(fact.predicate.stable_key() for fact in chosen)
            or coordinates != tuple((claim, role) for claim, fact in enumerate(chosen, 1)
                                    for role in range(1, len(fact.bindings) + 1))):
        raise ValueError("native generation lost an actual claim or claim-local role")
    templates = tuple(template for template in generator.templates() if template.stable_key() == witness.template_key)
    if len(templates) != 1 or len(witness.readers) != sum(len(fact.bindings) for fact in chosen):
        raise ValueError("native witness does not recover one complete cold template")
    template = templates[0]
    expected = {(claim, role) for claim, (_key, roles) in enumerate(outputs, 1) for role in roles}
    if {(claim, role.stable_key()) for claim, role, _scoped, _ordinal in witness.readers} != expected:
        raise ValueError("native scoped readers do not cover every selected role")
    require_native_operation_role_order(outputs, template, witness)
    bindings = []
    for claim, role, scoped, ordinal in witness.readers:
        fillers = tuple(binding for binding in sources[claim - 1].proposition.definition.bindings
                        if binding.role == role and binding.ordinal == ordinal)
        if len(fillers) != 1:
            raise ValueError("native reader does not select one current typed Core filler")
        bindings.append(BoundRoleBinding(scoped, fillers[0].filler, 0))
    protocols = _generation_protocols(generator.context, template.language_branch)
    bound = replace(_bound_relation_proposition(sources[0], template, protocols.content.answer),
                    bindings=tuple(bindings))
    representations = _read_sequence(generator, bound, template)
    return template, bound, representations


def verify_native_operation_generation(core, generator, execution, value, trace, variant, witness):
    """核对实际角色填充者、完整 native witness 和原调用 Representation 序列。"""
    encoded = _decode_input(value["generation"], trace)
    program = execution.witness.program
    realization_key = program.realization_key
    permitted_variant = (realization_key == (GRAPH_KIND, witness.variant_id)
        if program.path_query is None else
        len(realization_key) >= 2 and realization_key[0] == GRAPH_KIND
        and witness.variant_id in realization_key[1:])
    if (tuple(encoded["native_composition_witness"]) != witness.stable_key()
            or not permitted_variant
            or variant.source_identity != execution.witness.source_identity
            or variant.license_identity != execution.witness.license_identity):
        raise ValueError("original native generation witness or permission drifted")
    chosen, sources, plan = verify_native_plan_components(core, generator, execution, value, trace, encoded)
    template, _bound, representations = read_native_member_representations(
        generator, execution.outputs, chosen, sources, variant, witness)
    units = tuple(unit for item in representations for unit in representation_parts(item)[1])
    if (tuple(encoded["connector"]) != template.connector.stable_key()
            or tuple(map(tuple, encoded["representations"])) != tuple(item.stable_key() for item in representations)
            or units != tuple(encoded["surface_values"]) or units != tuple(map(ord, value["response_surface"]))):
        raise ValueError("original composed output differs from cold native Representation sequence")
    return {"native_representation_count": len(representations), "scoped_reader_count": len(witness.readers),
            "claim_count": len(chosen), "semantic_slot_count": len(plan["slot_sequence"])}


def audit_saved_native_operation_calls(source, core_source, *, require_operation_request_memory=False):
    """冷读取原调用的完整来源、图生成、O/H/E、proof、Episode 和 Use。"""
    source, core_source = Path(source).resolve(strict=True), Path(core_source).resolve(strict=True)
    before, original = _hashes(source), _hashes(core_source)
    identity, checkpoint = _json(source / "identity.json"), _json(source / "checkpoint.json")
    is_request = "readonly_request_source" in identity
    call_count = identity["query_count_planned"]
    if (identity["source_files"] != original or type(call_count) is not int or not 1 <= call_count <= 2
            or len(identity["planned_calls"]) != call_count
            or checkpoint["stage"] != ("ORIGINAL_REQUEST_CALLS_ENDED" if is_request else "ORIGINAL_NATIVE_CALLS_ENDED")
            or checkpoint["completed_calls"] != call_count):
        raise ValueError("native original source identity or completed checkpoint drifted")
    program_source = Path(identity["readonly_request_source"]).resolve(strict=True) if is_request else core_source
    program_original = _hashes(program_source)
    if is_request:
        if identity["request_source_files"] != program_original:
            raise ValueError("request original program/frame physical identity drifted")
    elif (identity["question_sha256"] != file_sha256(source / "question.train.json")
            or identity["question_graph_sha256"] != file_sha256(source / "questions.sqlite3")):
        raise ValueError("native question physical source identity drifted")
    manifests = tuple((core_source / "teachers").rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError("native Core lacks one physical licensed teacher pack")
    teachers = read_independent_relation_fact_source(core_source / "data/ph2/facts.jsonl", manifests[0].parent)
    program = compile_train_operation_program(program_source / "program.train.json", teachers)
    if read_operation_program_graph(program_source / "program.sqlite3") != (program,):
        raise ValueError("native program differs from the physical licensed train instruction")
    if is_request:
        from pure_integer_ai.experiments.operation_request_course import compile_train_operation_request, read_operation_request_graph
        from pure_integer_ai.experiments.operation_request_consumer import validate_operation_request_frames
        frames = read_operation_request_graph(program_source / "requests.sqlite3")
        if frames != (compile_train_operation_request(program_source / "request.train.json", teachers, program),):
            raise ValueError("request frame differs from its physical licensed train grammar")
        validate_operation_request_frames(frames, (program,))
    else:
        frames, metadata = read_relation_question_graph(source / "questions.sqlite3")
        syntax = _json(source / "question.train.json")
        seeds = tuple(_seed(row, number) for number, row in enumerate(
            _read_lines(core_source / "data/ph2/facts.jsonl", _FACT_FIELDS), 1) if row["seed_id"] == syntax["seed_id"])
        if ((syntax["owner"], syntax["split"], syntax["license_id"]) != ("teacher", "train", "CC0-1.0")
                or len(seeds) != 1 or frames != (compile_train_relation_question(seeds[0], syntax["question"],
                    source_case_key=tuple(syntax["source_case_key"]), operation_key=tuple(syntax["operation_key"])),)
                or metadata["source_identity"] != program.source_identity or metadata["license_identity"] != program.license_identity):
            raise ValueError("native question differs from its complete physical train syntax/permission")
    cases, prior = [], None
    with TrainedRelationGraphRuntime(core_source / "core.sqlite3") as core, TrainedGenerationConnectorRuntime(
            core_source / "core.sqlite3") as generator:
        course = compile_train_operation_composition(core_source / "composition.train.json", teachers, core, generator)
        variants, witnesses, _metadata = read_native_operation_composition_graph(core_source / "composition.sqlite3")
        if variants != (course.variant,) or witnesses != (course.witness,):
            raise ValueError("native grammar graph differs from its physical train course")
        facts, surfaces = read_operation_core_context(core), core.active_surface_facts()
        projector = TrainedInputStructureProjector(RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(core), open_role_frames=restore_open_role_frames(core.active_surface_frames(), surfaces))
        for number, planned in enumerate(identity["planned_calls"], 1):
            root = source / f"cases/{number:03d}"
            value, report = load_case_query_trace(root / "trace.int.json"), _json(root / "call_receipt.json")
            trace = GraphQueryTraceEvidence(value)
            if (planned["ordinal"] != number or report["input_values"] != planned["input_values"]
                    or report.get("trace_descriptor_sha256", report["trace_sha256"]) != file_sha256(root / "trace.int.json")
                    or report["trace_counts"] != trace.counts() or report["phase_boundary_status"] != "PASS"
                    or report["phase_receipt_sha256"] != file_sha256(root / "phase_attempt/receipt.json")):
                raise ValueError("native original call/phase receipt differs from the actual trace")
            captured, generation = None, {}
            if is_request:
                from pure_integer_ai.experiments.operation_request_call_audit import restore_saved_operation_request
                prepared = restore_saved_operation_request(value, report, projector, frames)
            if planned["split"] != "negative":
                if not is_request:
                    prepared = restore_saved_operation_question(value, report, projector, frames)
                execution = OperationExecution.from_stable_key(trace.resolve_ref(value["generation"]["operation_execution"]["integer_key_ref"]))
                verify_integer_operation(execution, facts, (program,))
                state = value["state_key"]
                proof = BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), space=1, scope_key=execution.query_key)
                evidence = []
                for ref in state["evidence"]:
                    key = trace.resolve_ref(ref)
                    fields = _unpack(key, 3, 6)
                    evidence.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
                if (execution.query_key != trace.resolve_ref(value["query_ref"])
                        or execution.variables != _request_variables(program, prepared, facts, core=core)
                        or proof.stable_key() not in tuple(trace.resolve_ref(ref) for ref in state["bindings"])
                        or not operation_evidence_matches(execution, facts, tuple(evidence))
                        or state["scalars"][6:9] != [0, 1, 0] or state["scalars"][15:] != [1, 1]
                        or trace.resolve_ref(state["scalars"][11]) != execution.outputs[0][0]
                        or report["termination"] != 1 or report["delivery_claim_count"] != 2):
                    raise ValueError("native current QueryState lacks the exact input/program/premise closure")
                generation = verify_native_operation_generation(core, generator, execution, value, trace, variants[0], witnesses[0])
                captured = load_case_provenance(root / "case_provenance.json")
            else:
                if not is_request:
                    prepared = projector.project(tuple(report["input_values"]))
                if _decode_input(value["input_structure"], trace) != prepared.integer_trace():
                    raise ValueError("unknown native input differs from its complete cold projection")
                verify_removed_operation_trace(value, report)
                if prior is None:
                    raise ValueError("unknown native call lacks the separately audited prior delivery")
                generation = verify_prior_delivery_slot_projections(value, prior)
            memory = audit_saved_relation_memory(source / "memory.sqlite3", value, report, prepared, captured=captured,
                require_operation_request_memory=require_operation_request_memory)
            if captured is not None:
                prior = captured
            cases.append({"split": planned["split"], "termination": report["termination"],
                "response_surface": report["response_surface"], "trace_counts": trace.counts(),
                "wall_ns": report["wall_ns"], "cpu_ns": report["cpu_ns"], "resource_ended": report["resource_ended"],
                "trace_sha256": report["trace_sha256"],
                "positive_chain_phase_coverage": report["positive_chain_phase_coverage"],
                "memory_request_binding_qualification": ("UNKNOWN_NOT_MATERIALIZED_IN_LEGACY_CALL"
                    if is_request and not require_operation_request_memory else "VERIFIED" if is_request else "NOT_APPLICABLE"),
                **generation, **memory})
    if before != _hashes(source) or original != _hashes(core_source) or program_original != _hashes(program_source):
        raise ValueError("native cold audit changed an immutable original source")
    return {"status": ("PASS_DEVELOPMENT_INDEPENDENT_REQUEST_CORE_OHE_PLAN_DELIVERY"
                       if is_request and require_operation_request_memory
                       else "PASS_DEVELOPMENT_REQUEST_JOIN_LEGACY_MEMORY_BOUNDARY"
                       if is_request else "PASS_DEVELOPMENT_NATIVE_MULTI_CLAIM_CORE_OHE_PLAN_DELIVERY"),
        "cases": cases, "source_sha256": before, "core_source_sha256": original,
        "program_source_sha256": program_original,
        "source_write_count": 0, "query_reexecution_count": 0, "generation_reexecution_count": 0,
        "held_out_novelty": identity["held_out_novelty"], "qualified_semantic_operation_families": 0,
        "session_tail_budget": "UNKNOWN_DEVELOPMENT_CALLS_ONLY", "free_dialogue_complete": 0}
