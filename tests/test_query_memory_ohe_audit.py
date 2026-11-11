"""Missing, cross-source and semantically drifting O/H/E must fail closed."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey, LIFECYCLE_ACTIVE
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion, CurriculumVersion, OBJECT_CONTEXT_SCOPE, ObjectIdentity,
    OwnerScope, ParserVersion, PrimitiveVersion, SourceRef, VersionBundle, VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_EVIDENCE, INTAKE_DERIVED_HYPOTHESIS, INTAKE_DERIVED_OBSERVATION,
    INTAKE_OUTCOME_SUCCESS, MEMORY_EVENT_EVIDENCE, MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_EVIDENCE, MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_OBSERVATION, RETENTION_EPISODIC,
    EvidencePayload, HypothesisPayload, IntakeDerivedBinding,
    IntakeManifestPayload, MemoryEvent, MemoryLinkedRef, ObservationPayload, memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry, QueryRoot, ROOT_EXPANDED
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED, CLOCK_MEMORY_OBSERVED, CLOCK_MEMORY_TIMELINE,
    LogicalClockIdentity, LogicalTimestamp, document_scope, session_scope,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import (
    QueryMemoryOheEvidenceError, verify_query_memory_ohe,
    verify_query_memory_ohe_without_plan,
)
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceIdentity


def _fixture(stance: int = 1, *, candidate_key=(88,), document_id=1):
    owner = OwnerScope(1, 2, 3, VISIBILITY_SESSION)
    versions = VersionBundle(CorpusVersion(1), ParserVersion(2), PrimitiveVersion(3), CurriculumVersion(4))
    source = SourceRef(91501, 10, document_id, owner, versions)
    source_hash = 19 + (document_id - 1) * 1000
    scope = document_scope(source)
    session = session_scope(3, owner=owner, versions=versions)
    space = SpaceIdentity(SPACE_TYPE_MEMORY, 31, 32)

    def timestamp(kind: int, seq: int):
        return LogicalTimestamp(LogicalClockIdentity(scope, kind), seq)

    def row(event_hash: int, kind: int, object_kind: int, payload, key=None):
        event_hash += (document_id - 1) * 1000
        ref = memory_object_ref(space, object_kind, payload.stable_key() if key is None else key,
                                owner=owner, versions=versions)
        return MaterializedMemoryEvent(
            event_hash, event_hash + 100, MemoryEvent(kind, ref, scope, payload),
            LogicalTimestamp(LogicalClockIdentity(session, CLOCK_MEMORY_TIMELINE), event_hash))

    act = ObjectIdentity(OBJECT_CONTEXT_SCOPE, (77,), owner, versions)
    orow = row(101, MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_OBSERVATION, ObservationPayload(
        source, MemoryLinkedRef.object(act), (), (), None, (), (), timestamp(CLOCK_MEMORY_OBSERVED, 1)))
    hypothesis = HypothesisKey((1,), candidate_key, (99,), scope, source)
    hrow = row(102, MEMORY_EVENT_HYPOTHESIS, MEMORY_OBJECT_HYPOTHESIS, HypothesisPayload(
        hypothesis, RETENTION_EPISODIC, LIFECYCLE_ACTIVE, timestamp(CLOCK_MEMORY_CREATED, 1)),
        hypothesis.stable_key())
    epayload = EvidencePayload(hrow.event.object_ref, stance, None, (7,), source, None, (8,), None,
                               timestamp(CLOCK_MEMORY_OBSERVED, 2))
    erow = row(103, MEMORY_EVENT_EVIDENCE, MEMORY_OBJECT_EVIDENCE, epayload)
    oref, href, eref = (item.event.object_ref for item in (orow, hrow, erow))
    manifest = IntakeManifestPayload(source, 1, 2, INTAKE_OUTCOME_SUCCESS, (
        IntakeDerivedBinding(INTAKE_DERIVED_OBSERVATION, (1,), oref),
        IntakeDerivedBinding(INTAKE_DERIVED_HYPOTHESIS, (2,), href),
        IntakeDerivedBinding(INTAKE_DERIVED_EVIDENCE, (2,), eref),
    ), None, timestamp(CLOCK_MEMORY_OBSERVED, 3))
    evidence = EvidenceEntry((source_hash, *source.stable_key()), href.stable_key(), 2, stance, 1,
                             eref.stable_key(), scope.stable_key(), epayload.stable_key())
    mirror = replace(evidence, space=3)
    table = []

    def intern(key):
        value = list(key)
        if value not in table:
            table.append(value)
        return table.index(value) + 1

    query = intern((65,))
    captured = {
        "status": "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS",
        "input_source": list(source.stable_key()), "input_observation": list(oref.stable_key()),
        "trace": {
            "schema_version": 4, "integer_key_table": table, "query_ref": query,
            "input_structure": {}, "frontier_trace": {"steps": []}, "active_spaces": [1, 2, 3],
            "state_key": {
                "query_ref": query, "active_spaces": [1, 2, 3], "scalars": [0] * 17,
                "evidence": [intern(evidence.stable_key()), intern(mirror.stable_key())],
                "bindings": [intern(binding.stable_key()) for binding in (
                    BindingEntry((MEMORY_OBJECT_OBSERVATION,), oref.stable_key(), 2, scope.stable_key()),
                    BindingEntry(hypothesis.hypothesis_kind, hypothesis.candidate_key, 2, scope.stable_key()),
                    BindingEntry((MEMORY_OBJECT_EVIDENCE,), eref.stable_key(), 2, scope.stable_key()))],
                "roots": [intern(QueryRoot(2, oref.stable_key(), ROOT_EXPANDED, 1).stable_key())],
            },
            "hops": [{"space": 2, "proposition_ref": intern(ref.stable_key()),
                      "predicate_ref": intern((kind,)), "source_hash": source_hash}
                     for ref, kind in ((oref, MEMORY_EVENT_OBSERVATION), (href, MEMORY_EVENT_HYPOTHESIS),
                                       (eref, MEMORY_EVENT_EVIDENCE))],
            "response_plan": {"evidence_refs": [list(evidence.stable_key()), list(mirror.stable_key())],
                              "memory_refs": [list(oref.stable_key()), list(href.stable_key())]},
        },
    }
    return captured, {item.event.object_ref: item for item in (orow, hrow, erow)}, manifest, evidence


def _verify(captured, declarations, manifest, source_hash=19, **budgets):
    return verify_query_memory_ohe(
        captured, declaration_reader=lambda ref: (declarations[ref],) if ref in declarations else (),
        current_manifest_reader=lambda source: manifest,
        source_hash_reader=lambda source: source_hash, **budgets)


@pytest.mark.parametrize("stance", [1, 2, 3])
def test_exact_intake_chain_preserves_stance_without_claiming_causality(stance):
    captured, declarations, manifest, _ = _fixture(stance)
    original = deepcopy(captured)
    audited = _verify(captured, declarations, manifest)
    assert audited["status"] == "MATCHED_DURABLE_INTAKE_OHE_QUERYSTATE_AND_PLAN_REFERENCES"
    assert audited["chains"] == [{"observation_event_hash": 101, "hypothesis_event_hash": 102,
                                  "evidence_event_hash": 103, "stance": stance}]
    assert audited["independent_evidence_count"] == 1
    assert audited["companion_projection_is_independent_evidence"] == 0
    assert audited["claim_inference_attribution"].startswith("UNKNOWN")
    assert audited["query_time_lifecycle"].startswith("UNKNOWN")
    assert audited["source_license"].startswith("UNKNOWN")
    assert captured == original


@pytest.mark.parametrize("missing_hash", [101, 102, 103])
def test_missing_o_h_or_e_declaration_fails(missing_hash):
    captured, declarations, manifest, _ = _fixture()
    declarations = {ref: row for ref, row in declarations.items() if row.event_hash != missing_hash}
    with pytest.raises(QueryMemoryOheEvidenceError, match="unique declaration"):
        _verify(captured, declarations, manifest)


@pytest.mark.parametrize("field,value", [
    ("source_ref", (20,)), ("polarity", 2), ("trust", 2),
    ("payload_key", (9,)), ("supersedes_key", (1,)), ("scope_key", (2,)),
])
def test_valid_query_member_with_durable_payload_drift_is_rejected(field, value):
    captured, declarations, manifest, evidence = _fixture()
    ref = captured["trace"]["state_key"]["evidence"][0]
    captured["trace"]["integer_key_table"][ref - 1] = list(replace(evidence, **{field: value}).stable_key())
    with pytest.raises(QueryMemoryOheEvidenceError, match="durable payload"):
        _verify(captured, declarations, manifest)


@pytest.mark.parametrize("mutation,match", [
    (lambda t: t["state_key"]["evidence"].pop(), "durable payload"),
    (lambda t: t["state_key"]["bindings"].pop(), "binding"),
    (lambda t: t["state_key"]["roots"].clear(), "root"),
    (lambda t: t["hops"].pop(1), "hops"),
    (lambda t: t["hops"][0].__setitem__("source_hash", 20), "hops"),
    (lambda t: t["response_plan"]["memory_refs"].pop(), "Hypothesis"),
    (lambda t: t["response_plan"]["evidence_refs"].pop(), "ResponsePlan"),
])
def test_missing_state_hop_or_plan_members_cannot_qualify(mutation, match):
    captured, declarations, manifest, _ = _fixture()
    mutation(captured["trace"])
    with pytest.raises(QueryMemoryOheEvidenceError, match=match):
        _verify(captured, declarations, manifest)


def test_appended_or_unbound_evidence_cannot_adopt_initial_manifest():
    captured, declarations, manifest, _ = _fixture()
    observation_only = replace(manifest, bindings=manifest.bindings[:1])
    with pytest.raises(QueryMemoryOheEvidenceError, match="initial intake"):
        _verify(captured, declarations, observation_only)


@pytest.mark.parametrize("source_hash", [None, True, 0, 20])
def test_source_hash_must_be_the_independent_durable_identity(source_hash):
    captured, declarations, manifest, _ = _fixture()
    with pytest.raises(QueryMemoryOheEvidenceError):
        _verify(captured, declarations, manifest, source_hash)


def test_duplicate_declaration_is_not_unique_evidence():
    captured, declarations, manifest, _ = _fixture()
    with pytest.raises(QueryMemoryOheEvidenceError, match="unique declaration"):
        verify_query_memory_ohe(captured, declaration_reader=lambda ref: (declarations[ref],) * 2,
                                current_manifest_reader=lambda source: manifest,
                                source_hash_reader=lambda source: 19)


def test_truncated_stable_key_is_rejected():
    captured, declarations, manifest, _ = _fixture()
    ref = captured["trace"]["state_key"]["evidence"][0]
    captured["trace"]["integer_key_table"][ref - 1].pop()
    with pytest.raises(QueryMemoryOheEvidenceError, match="QueryState member"):
        _verify(captured, declarations, manifest)


def test_delivered_slot_evidence_is_counted_without_becoming_intake_ohe():
    captured, declarations, manifest, evidence = _fixture()
    structural = EvidenceEntry(evidence.source_ref, (91532, 1), 2, 3, 1,
                               (91572, 1, 5), evidence.scope_key, (91572, 2))
    trace = captured["trace"]
    trace["integer_key_table"].append(list(structural.stable_key()))
    trace["state_key"]["evidence"].append(len(trace["integer_key_table"]))
    trace["response_plan"]["evidence_refs"].append(list(structural.stable_key()))
    result = _verify(captured, declarations, manifest)
    assert result["independent_evidence_count"] == 1
    assert result["delivered_slot_evidence_count"] == 1
    assert result["delivered_slot_durable_qualification"].startswith("UNKNOWN")
    trace["response_plan"]["evidence_refs"].pop()
    with pytest.raises(QueryMemoryOheEvidenceError, match="delivered slot Evidence"):
        _verify(captured, declarations, manifest)


def test_unregistered_memory_evidence_is_not_silently_excluded():
    captured, declarations, manifest, evidence = _fixture()
    ref = captured["trace"]["state_key"]["evidence"][0]
    unknown = replace(evidence, evidence_key=(91573, 1)).stable_key()
    captured["trace"]["integer_key_table"][ref - 1] = list(unknown)
    with pytest.raises(QueryMemoryOheEvidenceError, match="unregistered Memory Evidence"):
        _verify(captured, declarations, manifest)


def test_malformed_intake_memory_ref_still_fails():
    captured, declarations, manifest, evidence = _fixture()
    ref = captured["trace"]["state_key"]["evidence"][0]
    captured["trace"]["integer_key_table"][ref - 1] = list(
        replace(evidence, evidence_key=(1,)).stable_key())
    with pytest.raises(ValueError, match="MemoryObjectRef"):
        _verify(captured, declarations, manifest)


def _with_large_bindings(captured):
    trace = captured["trace"]
    for number in range(552):
        key = list(BindingEntry((91599,), (number + 1,), 1).stable_key())
        trace["integer_key_table"].append(key)
        trace["state_key"]["bindings"].append(len(trace["integer_key_table"]))


def test_explicit_capacity_checks_every_binding_and_preserves_default_limit():
    captured, declarations, manifest, _ = _fixture()
    _with_large_bindings(captured)
    with pytest.raises(QueryMemoryOheEvidenceError, match="state bindings.*budget"):
        _verify(captured, declarations, manifest)
    with pytest.raises(QueryMemoryOheEvidenceError, match="state bindings.*budget"):
        _verify(captured, declarations, manifest, max_state_members=554)
    result = _verify(captured, declarations, manifest, max_state_members=1024)
    assert result["inspected_state_member_counts"]["bindings"] == 555
    assert result["max_state_members"] == 1024
    final_ref = captured["trace"]["state_key"]["bindings"][-1]
    captured["trace"]["integer_key_table"][final_ref - 1].pop()
    with pytest.raises(QueryMemoryOheEvidenceError, match="QueryState member"):
        _verify(captured, declarations, manifest, max_state_members=1024)


@pytest.mark.parametrize("value", [True, 0, -1, 4097, None, "1024"])
def test_state_member_budget_rejects_non_integer_or_unbounded_capacity(value):
    captured, declarations, manifest, _ = _fixture()
    with pytest.raises(QueryMemoryOheEvidenceError, match="state member audit budget"):
        _verify(captured, declarations, manifest, max_state_members=value)


@pytest.mark.parametrize("value", [True, 0, -1, 4097, None, "256"])
def test_ohe_declaration_capacity_remains_bounded_and_explicit(value):
    captured, declarations, manifest, _ = _fixture()
    with pytest.raises(QueryMemoryOheEvidenceError, match="declaration audit budget"):
        _verify(captured, declarations, manifest, max_ohe_declarations=value)


def _manifest_with_many_consumed_chains(count):
    """构造同一来源的完整O/H/E链，尾页成员仍实际进入原QueryState与Plan。"""
    captured, declarations, manifest, first = _fixture()
    trace = captured['trace']
    template_hypothesis = next(row for ref, row in declarations.items()
                               if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS)
    template_evidence = next(row for ref, row in declarations.items()
                             if ref.object_kind == MEMORY_OBJECT_EVIDENCE)
    bindings = list(manifest.bindings)

    def intern(key):
        """完整整数记录逐项放入trace，不用hash替代对象本体。"""
        value = list(key)
        if value not in trace['integer_key_table']:
            trace['integer_key_table'].append(value)
        return trace['integer_key_table'].index(value) + 1

    for ordinal in range(1, count):
        hypothesis = replace(template_hypothesis.event.payload.hypothesis, candidate_key=(88, ordinal))
        hpayload = replace(template_hypothesis.event.payload, hypothesis=hypothesis)
        href = replace(template_hypothesis.event.object_ref, object_key=hypothesis.stable_key())
        hrow = replace(template_hypothesis, event_hash=201 + ordinal * 2,
            event=replace(template_hypothesis.event, object_ref=href, payload=hpayload))
        epayload = replace(template_evidence.event.payload, hypothesis_ref=href)
        eref = replace(template_evidence.event.object_ref, object_key=epayload.stable_key())
        erow = replace(template_evidence, event_hash=202 + ordinal * 2,
            event=replace(template_evidence.event, object_ref=eref, payload=epayload))
        declarations.update(((href, hrow), (eref, erow)))
        bindings.extend((IntakeDerivedBinding(INTAKE_DERIVED_HYPOTHESIS, (2, ordinal), href),
                         IntakeDerivedBinding(INTAKE_DERIVED_EVIDENCE, (2, ordinal), eref)))
        evidence = replace(first, hypothesis_key=href.stable_key(), evidence_key=eref.stable_key(),
                           payload_key=epayload.stable_key())
        for member in (evidence, replace(evidence, space=3)):
            trace['state_key']['evidence'].append(intern(member.stable_key()))
            trace['response_plan']['evidence_refs'].append(list(member.stable_key()))
        for binding in (BindingEntry(hypothesis.hypothesis_kind, hypothesis.candidate_key, 2, first.scope_key),
                        BindingEntry((MEMORY_OBJECT_EVIDENCE,), eref.stable_key(), 2, first.scope_key)):
            trace['state_key']['bindings'].append(intern(binding.stable_key()))
        trace['response_plan']['memory_refs'].append(list(href.stable_key()))
        trace['hops'].extend(dict(space=2, proposition_ref=intern(ref.stable_key()),
            predicate_ref=intern((kind,)), source_hash=19) for ref, kind in (
                (href, MEMORY_EVENT_HYPOTHESIS), (eref, MEMORY_EVENT_EVIDENCE)))
    return captured, declarations, replace(manifest, bindings=tuple(bindings))


def test_manifest_uses_original_explicit_budget_and_checks_every_tail_declaration():
    """显式131预算处理完整131成员；默认128和尾页缺声明仍严格拒绝。"""
    captured, declarations, manifest = _manifest_with_many_consumed_chains(65)
    assert len(manifest.bindings) == len(declarations) == 131
    with pytest.raises(QueryMemoryOheEvidenceError, match='intake manifest.*budget'):
        _verify(captured, declarations, manifest)
    result = _verify(captured, declarations, manifest, max_ohe_declarations=131)
    assert result['inspected_ohe_declaration_count'] == 131
    assert result['independent_evidence_count'] == 65
    del declarations[manifest.bindings[-1].object_ref]
    with pytest.raises(QueryMemoryOheEvidenceError, match='unique declaration'):
        _verify(captured, declarations, manifest, max_ohe_declarations=131)


def _unknown_fixture():
    captured, declarations, manifest, _ = _fixture(3)
    trace = captured["trace"]
    trace["integer_key_encoding"] = "INTEGER_KEY_TABLE_V1"
    trace.update(termination=3, response_plan=None, generation=None, response_surface=None)
    receipt = {"termination": 3, "delivery_claim_count": 0, "response_plan_present": 0,
               "generation_present": 0, "input_source": captured["input_source"],
               "observation_ref": captured["input_observation"]}
    return trace, receipt, declarations, manifest


def _verify_unknown(trace, receipt, declarations, manifest):
    return verify_query_memory_ohe_without_plan(
        trace, receipt,
        declaration_reader=lambda ref: (declarations[ref],) if ref in declarations else (),
        current_manifest_reader=lambda source: manifest, source_hash_reader=lambda source: 19)


def test_unknown_call_audits_durable_ohe_without_a_fabricated_plan():
    trace, receipt, declarations, manifest = _unknown_fixture()
    frozen = deepcopy(trace)
    result = _verify_unknown(trace, receipt, declarations, manifest)
    assert result["status"] == "MATCHED_DURABLE_INTAKE_OHE_QUERYSTATE_WITHOUT_PLAN"
    assert result["independent_evidence_count"] == 1 and trace == frozen


@pytest.mark.parametrize("polarity", [3, 1])
def test_prior_delivery_in_unknown_query_never_requires_or_creates_a_current_plan(polarity):
    """旧交付仅作 UNKNOWN 结构证据；缺当前 Plan 不删除或提升它的资格。"""
    trace, receipt, declarations, manifest = _unknown_fixture()
    structural = EvidenceEntry((19, *receipt["input_source"]), (91532, 1), 2, polarity, 1,
                               (91572, 1, 5), (1,), (91572, 2))
    trace["integer_key_table"].append(list(structural.stable_key()))
    trace["state_key"]["evidence"].append(len(trace["integer_key_table"]))
    if polarity != 3:
        with pytest.raises(QueryMemoryOheEvidenceError, match="delivered slot Evidence"):
            _verify_unknown(trace, receipt, declarations, manifest)
        return
    result = _verify_unknown(trace, receipt, declarations, manifest)
    assert result["delivered_slot_evidence_count"] == 1
    assert result["delivered_slot_durable_qualification"].startswith("UNKNOWN")
    assert trace["response_plan"] is None


@pytest.mark.parametrize("mutation,match", [
    (lambda t: t["state_key"]["evidence"].pop(), "durable payload"),
    (lambda t: t["state_key"]["bindings"].pop(), "binding"),
    (lambda t: t["hops"].pop(), "hops"),
    (lambda t: t.__setitem__("response_plan", {"claim_refs": []}), "no Plan"),
])
def test_unknown_audit_rejects_missing_members_and_added_plan(mutation, match):
    trace, receipt, declarations, manifest = _unknown_fixture()
    mutation(trace)
    with pytest.raises(QueryMemoryOheEvidenceError, match=match):
        _verify_unknown(trace, receipt, declarations, manifest)
