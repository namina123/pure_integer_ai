"""Join original query evidence to immutable intake O/H/E declarations.

This bounded audit supports initial intake and independently restored response
roles. It does not rerun a query, resolve surfaces, or write Memory.
Current manifests qualify the audit snapshot, not lifecycle at query time.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_EVIDENCE, INTAKE_DERIVED_HYPOTHESIS,
    INTAKE_DERIVED_OBSERVATION, INTAKE_OUTCOME_SUCCESS,
    MEMORY_EVENT_EVIDENCE, MEMORY_EVENT_HYPOTHESIS, MEMORY_EVENT_OBSERVATION,
    MEMORY_OBJECT_EVIDENCE, MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_OBSERVATION,
    EvidencePayload, HypothesisPayload, IntakeManifestPayload,
    MemoryObjectRef, ObservationPayload,
)
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry, EvidenceEntry, QueryRoot, ROOT_EXPANDED,
)
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.query_plan_memory_references import (
    allowed_plan_memory_references, read_plan_memory_references,
)
from pure_integer_ai.experiments.trained_response_occurrence import InputRoleOccurrence


class QueryMemoryOheEvidenceError(ValueError):
    """A durable O/H/E member differs from the original query or plan."""


DeclarationReader = Callable[[MemoryObjectRef], Sequence[MaterializedMemoryEvent]]


def _key(value: object, label: str) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value or len(value) > 1_000_000
            or any(type(item) is not int or item < 0 for item in value)):
        raise QueryMemoryOheEvidenceError(f"{label} requires a bounded integer key")
    return tuple(value)


def _rows(value: object, label: str, maximum: int = 256) -> list | tuple:
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise QueryMemoryOheEvidenceError(f"{label} is missing or exceeds audit budget")
    return value


def _unpack(key: tuple[int, ...], prefix: int, count: int,
            tail: int = 0) -> tuple[tuple[int, ...], ...]:
    cursor, parts = prefix, []
    if len(key) < prefix + count + tail:
        raise QueryMemoryOheEvidenceError("truncated QueryState member")
    for _ in range(count):
        if cursor >= len(key):
            raise QueryMemoryOheEvidenceError("truncated QueryState member")
        size = key[cursor]
        cursor += 1
        if size > len(key) - cursor:
            raise QueryMemoryOheEvidenceError("invalid QueryState member length")
        parts.append(key[cursor:cursor + size])
        cursor += size
    if cursor + tail != len(key):
        raise QueryMemoryOheEvidenceError("trailing QueryState member data")
    return tuple(parts)


def verify_query_memory_ohe(
        provenance: Mapping[str, Any], *, declaration_reader: DeclarationReader,
        current_manifest_reader: Callable[[SourceRef], IntakeManifestPayload],
        source_hash_reader: Callable[[SourceRef], int | None],
        max_state_members: int = 256,
        max_ohe_declarations: int = 128,
        input_role_occurrences: tuple[InputRoleOccurrence, ...] = (),
        response_contexts: tuple = (),
        reference_contexts: tuple | None = None,
        reference_candidate_sets: tuple = (), reference_resolutions: tuple = (),
        revision_cue_records: tuple = (), core_runtime=None, reference_transition=None,
        ) -> dict[str, Any]:
    """Match intake declarations, consumed hops, state members and plan refs.

Readers must be authorized and bound to one completed, immutable Memory.
The caller separately qualifies the delivery proof, split and source license.
"""
    if type(max_state_members) is not int or not 0 < max_state_members <= 4096:
        raise QueryMemoryOheEvidenceError("state member audit budget must be a strict integer from 1 to 4096")
    if (not isinstance(provenance, Mapping)
            or provenance.get("status") != "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS"):
        raise QueryMemoryOheEvidenceError("original call provenance is missing")
    trace = GraphQueryTraceEvidence.from_integer_artifact(provenance.get("trace"))
    state, plan = trace.trace["state_key"], trace.trace.get("response_plan")
    if (state.get("active_spaces") != [1, 2, 3]
            or trace.trace.get("active_spaces") != [1, 2, 3]
            or not isinstance(plan, Mapping)):
        raise QueryMemoryOheEvidenceError("shared QueryState or ResponsePlan is missing")
    input_source = SourceRef.from_stable_key(_key(provenance.get("input_source"), "input source"))
    input_ref = MemoryObjectRef.from_stable_key(_key(
        provenance.get("input_observation"), "input Observation"))
    return _verify_query_memory_state(
        trace, input_source, input_ref, plan=plan,
        declaration_reader=declaration_reader,
        current_manifest_reader=current_manifest_reader,
        source_hash_reader=source_hash_reader, max_state_members=max_state_members,
        max_ohe_declarations=max_ohe_declarations,
        input_role_occurrences=input_role_occurrences, response_contexts=response_contexts,
        reference_contexts=reference_contexts,
        reference_candidate_sets=reference_candidate_sets, reference_resolutions=reference_resolutions,
        revision_cue_records=revision_cue_records, core_runtime=core_runtime,
        reference_transition=reference_transition)


def verify_query_memory_ohe_without_plan(
        trace_value: Mapping[str, Any], call_receipt: Mapping[str, Any], *,
        declaration_reader: DeclarationReader,
        current_manifest_reader: Callable[[SourceRef], IntakeManifestPayload],
        source_hash_reader: Callable[[SourceRef], int | None],
        max_state_members: int = 256,
        max_ohe_declarations: int = 128,
        input_role_occurrences: tuple[InputRoleOccurrence, ...] = (),
        response_contexts: tuple = (),
        reference_contexts: tuple | None = None,
        reference_candidate_sets: tuple = (), reference_resolutions: tuple = (),
        revision_cue_records: tuple = (), reference_transition=None,
        ) -> dict[str, Any]:
    """Cold-join an UNKNOWN call to durable intake without inventing a Plan."""
    trace = GraphQueryTraceEvidence(trace_value)
    if (trace.trace.get("termination") not in {2, 3, 4, 5}
            or call_receipt.get("termination") != trace.trace.get("termination")
            or trace.trace.get("response_plan") or trace.trace.get("generation")
            or trace.trace.get("response_surface")
            or call_receipt.get("delivery_claim_count") != 0
            or call_receipt.get("response_plan_present") != 0
            or call_receipt.get("generation_present") != 0):
        raise QueryMemoryOheEvidenceError("UNKNOWN audit requires no Plan/generation/delivery")
    source = SourceRef.from_stable_key(_key(call_receipt.get("input_source"), "input source"))
    observation = MemoryObjectRef.from_stable_key(_key(
        call_receipt.get("observation_ref"), "input Observation"))
    return _verify_query_memory_state(
        trace, source, observation, plan=None,
        declaration_reader=declaration_reader,
        current_manifest_reader=current_manifest_reader,
        source_hash_reader=source_hash_reader, max_state_members=max_state_members,
        max_ohe_declarations=max_ohe_declarations,
        input_role_occurrences=input_role_occurrences, response_contexts=response_contexts,
        reference_contexts=reference_contexts,
        reference_candidate_sets=reference_candidate_sets, reference_resolutions=reference_resolutions,
        revision_cue_records=revision_cue_records, reference_transition=reference_transition)


def verify_query_memory_ohe_nonfactual(trace_value, call_receipt, *, max_state_members=256,
                                      input_structure=None, core_fact_index=None,
                                      relation_question_frames=(), assertion_scope_frames=(),
                                      core_runtime=None, authorized_input=None, **readers):
    """只核对原调用已经保存的非事实 Plan，不从历史或摘要补造新 Plan。"""
    trace = GraphQueryTraceEvidence(trace_value)
    plan = trace.trace.get('response_plan')
    generation = trace.trace.get('generation')
    state = trace.trace['state_key']['scalars']
    if (not isinstance(plan, Mapping) or plan.get('claim_refs') or plan.get('claim_evidence_refs')
            or trace.trace.get('termination') not in (2, 3, 4, 8)
            or call_receipt.get('termination') != trace.trace.get('termination')
            or call_receipt.get('delivery_claim_count') != 0 or call_receipt.get('response_plan_present') != 1
            or call_receipt.get('generation_present') != 1 or not isinstance(generation, Mapping)
            or generation.get('kind') != 3 or state[7] != 0 or state[15] != 1):
        raise QueryMemoryOheEvidenceError('nonfactual O/H/E requires the saved generated non-claim Plan')
    source = SourceRef.from_stable_key(_key(call_receipt.get('input_source'), 'input source'))
    observation = MemoryObjectRef.from_stable_key(_key(call_receipt.get('observation_ref'), 'input Observation'))
    return _verify_query_memory_state(trace, source, observation, plan=plan,
        max_state_members=max_state_members, nonfactual_plan=True,
        input_conflict_context=(input_structure, core_fact_index, relation_question_frames),
        assertion_scope_frames=assertion_scope_frames, core_runtime=core_runtime,
        authorized_input=authorized_input, **readers)


def _verify_query_memory_state(
        trace: GraphQueryTraceEvidence, input_source: SourceRef, input_ref: MemoryObjectRef,
        *, plan: Mapping[str, Any] | None, declaration_reader: DeclarationReader,
        current_manifest_reader: Callable[[SourceRef], IntakeManifestPayload],
        source_hash_reader: Callable[[SourceRef], int | None], max_state_members: int,
        max_ohe_declarations: int = 128,
        input_role_occurrences: tuple[InputRoleOccurrence, ...] = (),
        response_contexts: tuple = (),
        reference_contexts: tuple | None = None,
        reference_candidate_sets: tuple = (), reference_resolutions: tuple = (),
        nonfactual_plan: bool = False,
        input_conflict_context=None,
        assertion_scope_frames=(), core_runtime=None, authorized_input=None,
        revision_cue_records: tuple = (), reference_transition=None,
        ) -> dict[str, Any]:
    if type(max_state_members) is not int or not 0 < max_state_members <= 4096:
        raise QueryMemoryOheEvidenceError("state member audit budget must be a strict integer from 1 to 4096")
    if type(max_ohe_declarations) is not int or not 0 < max_ohe_declarations <= 4096:
        raise QueryMemoryOheEvidenceError("O/H/E declaration audit budget must be a strict integer from 1 to 4096")
    state = trace.trace["state_key"]
    if (state.get("active_spaces") != [1, 2, 3]
            or trace.trace.get("active_spaces") != [1, 2, 3]):
        raise QueryMemoryOheEvidenceError("shared QueryState is missing")
    declarations: dict[MemoryObjectRef, MaterializedMemoryEvent] = {}
    manifests: dict[SourceRef, IntakeManifestPayload] = {}
    source_hashes: dict[SourceRef, int] = {}
    expected_kinds = {
        MEMORY_OBJECT_OBSERVATION: (MEMORY_EVENT_OBSERVATION, ObservationPayload),
        MEMORY_OBJECT_HYPOTHESIS: (MEMORY_EVENT_HYPOTHESIS, HypothesisPayload),
        MEMORY_OBJECT_EVIDENCE: (MEMORY_EVENT_EVIDENCE, EvidencePayload),
    }

    def declaration(ref: MemoryObjectRef) -> MaterializedMemoryEvent:
        if ref not in declarations:
            if len(declarations) >= max_ohe_declarations:
                raise QueryMemoryOheEvidenceError("O/H/E exceeds explicit declaration audit budget")
            if (ref.object_kind not in expected_kinds
                    or ref.owner != input_source.owner or ref.versions != input_source.versions
                    or ref.memory_space != input_ref.memory_space):
                raise QueryMemoryOheEvidenceError("O/H/E crosses owner/version/space or budget")
            rows = declaration_reader(ref)
            if not isinstance(rows, (list, tuple)) or len(rows) != 1:
                raise QueryMemoryOheEvidenceError("O/H/E lacks a unique declaration")
            row = rows[0]
            kind, payload_type = expected_kinds[ref.object_kind]
            if (not isinstance(row, MaterializedMemoryEvent)
                    or type(row.event_hash) is not int or row.event_hash <= 0
                    or not row.event.is_declaration or row.event.object_ref != ref
                    or row.event.event_kind != kind or not isinstance(row.event.payload, payload_type)):
                raise QueryMemoryOheEvidenceError("O/H/E declaration identity differs")
            declarations[ref] = row
        return declarations[ref]

    def manifest(source: SourceRef) -> IntakeManifestPayload:
        if source not in manifests:
            item = current_manifest_reader(source)
            if (not isinstance(item, IntakeManifestPayload) or item.source != source
                    or item.outcome_kind != INTAKE_OUTCOME_SUCCESS):
                raise QueryMemoryOheEvidenceError("current intake manifest differs or is unsuccessful")
            if len(item.bindings) > max_ohe_declarations:
                raise QueryMemoryOheEvidenceError("intake manifest exceeds explicit declaration audit budget")
            manifests[source] = item
        return manifests[source]

    def source_hash(source: SourceRef) -> int:
        if source not in source_hashes:
            value = source_hash_reader(source)
            if type(value) is not int or value <= 0:
                raise QueryMemoryOheEvidenceError("durable source hash is absent")
            source_hashes[source] = value
        return source_hashes[source]

    def observation(source: SourceRef) -> MemoryObjectRef:
        refs = tuple(binding.object_ref for binding in manifest(source).bindings
                     if binding.binding_kind == INTAKE_DERIVED_OBSERVATION)
        if len(refs) != 1 or declaration(refs[0]).event.payload.source != source:
            raise QueryMemoryOheEvidenceError("manifest Observation differs from source")
        return refs[0]

    if input_ref.object_kind != MEMORY_OBJECT_OBSERVATION or observation(input_source) != input_ref:
        raise QueryMemoryOheEvidenceError("input Observation differs from manifest")
    compound_evidence = {}

    def evidence_key(raw, label):
        if not isinstance(raw, (list, tuple)) or len(raw) <= 1_000_000:
            return _key(raw, label)
        raw = tuple(raw)
        if raw not in compound_evidence:
            readers = dict(core_runtime=core_runtime, key_reader=_key, rows_reader=_rows,
                max_state_members=max_state_members, error=QueryMemoryOheEvidenceError)
            if raw[:3] == (3, 1, 1):
                from pure_integer_ai.experiments.compound_dialogue_evidence_audit import verify_compound_dialogue_evidence
                restored = verify_compound_dialogue_evidence(raw, trace, **readers)
            elif raw[:3] == (1, 3, 1):
                from pure_integer_ai.experiments.compound_reference_adoption_evidence_audit import verify_compound_reference_adoption_evidence
                restored = verify_compound_reference_adoption_evidence(raw, trace,
                    transition=reference_transition, resolutions=reference_resolutions, **readers)
            else:
                raise QueryMemoryOheEvidenceError('unregistered compound Evidence type')
            compound_evidence[raw] = restored
        return compound_evidence[raw]

    evidence = []
    for ref in _rows(state.get("evidence"), "state evidence", max_state_members):
        key = evidence_key(trace.resolve_ref(ref), "state evidence")
        source, hypothesis, member, scope, payload, supersedes = _unpack(key, 3, 6)
        evidence.append(EvidenceEntry(source, hypothesis, key[0], key[1], key[2],
                                      member, scope, payload, supersedes))
    if len({entry.stable_key() for entry in evidence}) != len(evidence):
        raise QueryMemoryOheEvidenceError("QueryState evidence is duplicated")
    memory_evidence = tuple(entry for entry in evidence
                            if entry.space == 2 and entry.evidence_key[:1] == (1,))
    delivered_slot_evidence = tuple(entry for entry in evidence
                                    if entry.space == 2 and entry.evidence_key[:2] == (91572, 1))
    if (type(input_role_occurrences) is not tuple or len(input_role_occurrences) > 64
            or any(not isinstance(item, InputRoleOccurrence) for item in input_role_occurrences)
            or len({item.stable_key() for item in input_role_occurrences}) != len(input_role_occurrences)):
        raise QueryMemoryOheEvidenceError("input role audit requires bounded cold graph records")
    role_keys = {item.stable_key() for item in input_role_occurrences}
    role_evidence = tuple(entry for entry in evidence
                          if entry.space == 2 and entry.evidence_key in role_keys)
    if (len(role_evidence) != len(input_role_occurrences)
            or set(role_evidence) != {item.query_evidence() for item in input_role_occurrences}
            or any(item.observation != input_ref or item.source_ref != (
                source_hash(input_source), *input_source.stable_key())
                for item in input_role_occurrences)):
        raise QueryMemoryOheEvidenceError("input role Evidence differs from cold source occurrences")
    bindings = set()
    for ref in _rows(state.get("bindings"), "state bindings", max_state_members):
        raw = trace.resolve_ref(ref)
        if len(raw) > 1_000_000:
            from pure_integer_ai.experiments.compound_generation_binding_audit import verify_compound_generation_binding
            key = verify_compound_generation_binding(raw, trace, evidence, key_reader=_key,
                rows_reader=_rows, max_state_members=max_state_members,
                max_ohe_declarations=max_ohe_declarations, error=QueryMemoryOheEvidenceError)
        else:
            key = _key(raw, "state binding")
        role, filler, scope = _unpack(key, 1, 3, 1)
        bindings.add(BindingEntry(role, filler, key[0], scope, key[-1]).stable_key())
    if any(item.query_binding().stable_key() not in bindings for item in input_role_occurrences):
        raise QueryMemoryOheEvidenceError("QueryState omits input role occurrence binding")
    roots = []
    for ref in _rows(state.get("roots"), "state roots", max_state_members):
        key = _key(trace.resolve_ref(ref), "state root")
        root, source, scope = _unpack(key, 3, 3)
        roots.append(QueryRoot(key[0], root, key[1], key[2], source, scope))
    if not any(root.owner_space == 2 and root.status == ROOT_EXPANDED for root in roots):
        raise QueryMemoryOheEvidenceError("Memory root was not expanded")
    plan_keys = (set() if plan is None else {evidence_key(key, "plan evidence")
                 for key in _rows(plan.get("evidence_refs"), "plan evidence", max_state_members)})
    if any(entry.polarity != 3 or entry.trust != 1 or entry.supersedes_key
           or (plan is not None and entry.stable_key() not in plan_keys)
           for entry in delivered_slot_evidence):
        raise QueryMemoryOheEvidenceError("delivered slot Evidence differs from ResponsePlan")
    allowed_plan_refs = allowed_plan_memory_references(
        input_ref, evidence, roots, input_role_occurrences, response_contexts)
    plan_refs = (set() if plan is None else read_plan_memory_references(
        plan.get("memory_refs"), allowed=allowed_plan_refs, input_ref=input_ref,
        maximum=max_ohe_declarations, key_reader=_key, error=QueryMemoryOheEvidenceError))
    adopted_hypothesis_keys = None
    generic_adoption = {}
    conflict_clarification = {}
    if nonfactual_plan:
        if trace.trace.get('termination') == 2:
            from pure_integer_ai.experiments.input_conflict_clarification_audit import verify_input_conflict_clarification
            if input_conflict_context is None:
                raise QueryMemoryOheEvidenceError('conflict clarification lacks its cold trained input context')
            conflict_clarification = verify_input_conflict_clarification(
                trace, *input_conflict_context, roots=roots, evidence=evidence, bindings=bindings,
                error=QueryMemoryOheEvidenceError, assertion_scope_frames=assertion_scope_frames,
                core_runtime=core_runtime, authorized_input=authorized_input)
        from pure_integer_ai.experiments.generic_plan_memory_audit import verify_generic_plan_memory
        adopted_hypothesis_keys, generic_adoption = verify_generic_plan_memory(
            trace, input_source, input_ref, evidence, roots, declaration=declaration,
            manifest=manifest, error=QueryMemoryOheEvidenceError)
    from pure_integer_ai.experiments.response_role_memory_audit import verify_response_role_memory
    response_role_evidence = verify_response_role_memory(response_contexts, evidence=evidence,
        bindings=bindings, roots=roots, plan_keys=plan_keys, plan_refs=plan_refs,
        has_plan=plan is not None, error=QueryMemoryOheEvidenceError,
        adopted_hypothesis_keys=adopted_hypothesis_keys)
    from pure_integer_ai.experiments.query_reference_memory_audit import verify_reference_memory
    from pure_integer_ai.experiments.reference_history_qualification import qualify_reference_history
    reference_history = qualify_reference_history(response_contexts, reference_contexts,
        trace=trace, evidence=evidence, bindings=bindings,
        candidate_sets=reference_candidate_sets, resolutions=reference_resolutions,
        nonfactual_plan=nonfactual_plan, adopted_hypothesis_keys=adopted_hypothesis_keys,
        generic_adoption=generic_adoption, error=QueryMemoryOheEvidenceError)
    reference_evidence = verify_reference_memory(reference_candidate_sets, reference_resolutions,
        input_roles=input_role_occurrences, contexts=reference_history, evidence=evidence,
        bindings=bindings, roots=roots, plan_keys=plan_keys, plan_refs=plan_refs,
        has_plan=plan is not None, error=QueryMemoryOheEvidenceError,
        adopted_hypothesis_keys=adopted_hypothesis_keys)
    if sum(entry.space == 2 for entry in evidence) != (
            len(memory_evidence) + len(delivered_slot_evidence) + len(role_evidence)
            + len(response_role_evidence) + len(reference_evidence)):
        raise QueryMemoryOheEvidenceError("unregistered Memory Evidence identity")
    if not memory_evidence:
        raise QueryMemoryOheEvidenceError("QueryState contains no consumed Memory Evidence")
    hops = trace.hops()
    ohe_hops = {(hop.proposition_key, hop.predicate_key, hop.source_hash)
                for hop in hops if hop.space == 2 and hop.predicate_key in {
                    (MEMORY_EVENT_OBSERVATION,), (MEMORY_EVENT_HYPOTHESIS,), (MEMORY_EVENT_EVIDENCE,)}}
    from pure_integer_ai.experiments.memory_provenance_inputs import verify_context_input_observations
    expected_hops = verify_context_input_observations(response_contexts, roots=roots, bindings=bindings,
        declaration=declaration, observation=observation, source_hash=source_hash,
        error=QueryMemoryOheEvidenceError)
    chains = []
    appended_chains = []
    for entry in memory_evidence:
        eref = MemoryObjectRef.from_stable_key(entry.evidence_key)
        erow = declaration(eref)
        epayload = erow.event.payload
        if not isinstance(epayload, EvidencePayload) or epayload.source is None:
            raise QueryMemoryOheEvidenceError("Evidence requires its own SourceRef")
        href = epayload.hypothesis_ref
        hrow = declaration(href)
        if not isinstance(hrow.event.payload, HypothesisPayload):
            raise QueryMemoryOheEvidenceError("Evidence target is not a declared Hypothesis")
        hypothesis = hrow.event.payload.hypothesis
        hsource = hypothesis.observation
        oref = observation(hsource)
        orow = declaration(oref)
        hbindings = tuple(binding for binding in manifest(hsource).bindings
                          if binding.binding_kind == INTAKE_DERIVED_HYPOTHESIS and binding.object_ref == href)
        ebindings = tuple(binding for binding in manifest(epayload.source).bindings
                          if binding.binding_kind == INTAKE_DERIVED_EVIDENCE and binding.object_ref == eref)
        if (len(hbindings) != 1 or len(ebindings) != 1 or hsource != epayload.source
                or hbindings[0].lineage_key != ebindings[0].lineage_key):
            from pure_integer_ai.experiments.revision_memory_chain_audit import verify_revision_memory_chain
            length = verify_revision_memory_chain(erow, hrow, declaration=declaration,
                manifest=manifest, observation=observation, cue_records=revision_cue_records,
                max_steps=max_ohe_declarations, error=QueryMemoryOheEvidenceError)
            if not length:
                raise QueryMemoryOheEvidenceError('initial intake O/H/E bindings differ; appended Evidence is unqualified')
            appended_chains.append(dict(evidence_event_hash=erow.event_hash, append_count=length))
        hscope = hypothesis.scope.stable_key()
        if hrow.event.scope.stable_key() != hscope or orow.event.scope.stable_key() != hscope:
            raise QueryMemoryOheEvidenceError("Observation/Hypothesis scope differs")
        expected = EvidenceEntry(
            (source_hash(epayload.source), *epayload.source.stable_key()), href.stable_key(),
            2, epayload.stance, 1, eref.stable_key(), erow.event.scope.stable_key(),
            epayload.stable_key(), () if epayload.supersedes_ref is None else epayload.supersedes_ref.stable_key())
        mirrored = EvidenceEntry(expected.source_ref, expected.hypothesis_key, 3,
                                 expected.polarity, expected.trust, expected.evidence_key,
                                 expected.scope_key, expected.payload_key, expected.supersedes_key)
        if entry != expected or mirrored not in evidence:
            raise QueryMemoryOheEvidenceError("Memory/Companion Evidence differs from durable payload")
        if plan is not None and (entry.stable_key() not in plan_keys
                                 or mirrored.stable_key() not in plan_keys):
            raise QueryMemoryOheEvidenceError("ResponsePlan omits consumed O/H/E Evidence")
        adopted_refs = ({oref, href} if adopted_hypothesis_keys is None else
                        ({href, eref} if href.stable_key() in adopted_hypothesis_keys else set()))
        if plan is not None and not adopted_refs.issubset(plan_refs):
            raise QueryMemoryOheEvidenceError("ResponsePlan omits consumed Observation/Hypothesis")
        for role, filler, scope in (
                ((MEMORY_OBJECT_OBSERVATION,), oref.stable_key(), hscope),
                (hypothesis.hypothesis_kind, hypothesis.candidate_key, hscope),
                ((MEMORY_OBJECT_EVIDENCE,), eref.stable_key(), erow.event.scope.stable_key())):
            if BindingEntry(role, filler, 2, scope).stable_key() not in bindings:
                raise QueryMemoryOheEvidenceError("QueryState omits consumed O/H/E binding")
        expected_hops.update((ref.stable_key(), (kind,), source_hash(source)) for ref, kind, source in (
            (oref, MEMORY_EVENT_OBSERVATION, hsource), (href, MEMORY_EVENT_HYPOTHESIS, hsource),
            (eref, MEMORY_EVENT_EVIDENCE, epayload.source)))
        chains.append({"observation_event_hash": orow.event_hash, "hypothesis_event_hash": hrow.event_hash,
                       "evidence_event_hash": erow.event_hash, "stance": epayload.stance})
    if ohe_hops != expected_hops:
        raise QueryMemoryOheEvidenceError("consumed O/H/E hops differ from durable declarations")
    return {
        "format": "PURE_INTEGER_QUERY_MEMORY_OHE_AUDIT_V1",
        "status": ("MATCHED_DURABLE_INTAKE_OHE_QUERYSTATE_WITHOUT_PLAN" if plan is None
                   else "MATCHED_DURABLE_INTAKE_OHE_QUERYSTATE_AND_PLAN_REFERENCES"),
        "max_state_members": max_state_members,
        "conflict_clarification": conflict_clarification,
        "max_ohe_declarations": max_ohe_declarations,
        "inspected_ohe_declaration_count": len(declarations),
        "inspected_state_member_counts": {name: len(state[name]) for name in ("bindings", "evidence", "roots")},
        "chains": chains, "independent_evidence_count": len(memory_evidence),
        "physical_revision_append_chains": appended_chains,
        "delivered_slot_evidence_count": len(delivered_slot_evidence),
        "unknown_input_role_occurrence_count": len(input_role_occurrences),
        "unknown_response_role_occurrence_count": len(response_role_evidence),
        "reference_candidate_set_count":len(reference_candidate_sets),
        "reference_resolution_count":len(reference_resolutions),
        "reference_evidence_count":len(reference_evidence),
        "reference_history_context_count":len(reference_history),
        "actually_consumed_context_count":len(response_contexts),
        'generic_plan_adoption':generic_adoption,
        "response_role_durable_qualification": ("MATCHED_COLD_PROOF_EPISODE_OCCURRENCE_GRAPH_STRUCTURAL_ONLY"
                                               if response_role_evidence else "NOT_CONSUMED"),
        "delivered_slot_durable_qualification": "UNKNOWN_outside_initial_intake_OHE_audit",
        "companion_projection_is_independent_evidence": 0,
        "manifest_snapshot": "MATCHED_CURRENT_INITIAL_INTAKE_BINDINGS",
        "query_time_lifecycle": "UNKNOWN_requires_original_lifecycle_capture",
        "claim_inference_attribution": "UNKNOWN_reference_membership_is_not_causal_proof",
        "source_license": "UNKNOWN_requires_frozen_source_manifest_audit",
        "delivery_qualification": "UNKNOWN_requires_independent_delivery_audit",
        "novelty_qualification": "UNKNOWN_requires_split_audit", "free_dialogue_complete": 0,
    }
