"""完整语义编辑、同次消费 proof 和无效接纳的有界专项。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.integer_operation_behavior import (
    EDIT_GUARDS, OperationSemanticEditReceipt, semantic_selection_obligation,
)
from pure_integer_ai.cognition.shared.integer_operation_execution import execute_integer_operation, operation_context_identity
from pure_integer_ai.cognition.shared.integer_operation_program import EQUAL, UNEQUAL, OperationProgramBudget, OperationProjection
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.experiments.operation_program_query import OPERATION_EXECUTION_ROLE
from pure_integer_ai.experiments.operation_semantic_trace_audit import audit_consumed_semantic_pair

# 只复用既有独立 H-05/Evidence 测试构造，不读取生产语料或评测标签。
from test_integer_operation_program import _atom, _fact, _schema, _source, _witness, LEFT, RIGHT
from pure_integer_ai.cognition.shared.semantic_object import entity_identity


@pytest.fixture(scope="module")
def pair():
    """生成真实等值 guard 差分；测试结构只给当前闭合角色绑定。"""
    a, b = (entity_identity(_source(200), (number,)) for number in (1, 2))
    schema = _schema(1)
    facts = (_fact(1, schema.relation, a, b),)
    from pure_integer_ai.cognition.shared.integer_operation_program import IntegerOperationProgram
    program = IntegerOperationProgram((92910, 1), (1, 2), (_atom(schema, (1, 2)),),
        ((UNEQUAL, 1, 2),), (OperationProjection(1, (LEFT, RIGHT)),))
    witnesses = (_witness(program, facts), _witness(replace(program, guards=((EQUAL, 1, 2),)), facts))
    # 私有 prepared 对象只用于测试契约；实际生产审计重新冷投影完整语言图。
    structure = {"source_ref": [92911, 1], "token_values": [88]}
    relation_key = (92912, 1)
    from pure_integer_ai.cognition.understanding.query_input_structure import PROJECTION_FILLER
    candidates = tuple(SimpleNamespace(candidate_key=filler.stable_key(), projection_kind=PROJECTION_FILLER,
        proposition_key=facts[0].proposition_key, role_key=role.stable_key(),
        predicate_key=schema.relation.stable_key(), source_ref=(5,), span_ref=(6,))
        for role, filler in ((LEFT, a), (RIGHT, b)))
    prepared = SimpleNamespace(source_ref=(92911, 1), operation_keys=(program.operation_key,),
        operation_request_bindings=(), relation_question_bindings=(SimpleNamespace(
            operation_key=program.operation_key, relation_key=relation_key, slot_span_refs=((6,),)),),
        relation_candidates=(SimpleNamespace(stable_key=lambda: relation_key,
            predicate_key=schema.relation.stable_key(), proposition_key=facts[0].proposition_key, source_ref=(5,)),),
        semantic_candidates=candidates, integer_trace=lambda: deepcopy(structure), stable_key=lambda: (92913, 1))
    executions = tuple(execute_integer_operation(witness, prepared.source_ref, ((1, a), (2, b)), facts)
                       for witness in witnesses)
    edit = OperationSemanticEditReceipt(*witnesses, operation_context_identity(facts, OperationProgramBudget()),
        prepared.stable_key(), (1,) * 32, (2,) * 32, (EDIT_GUARDS,))
    return facts, witnesses, prepared, executions, edit


def _trace(execution, prepared, facts):
    """构造包含完整 consumed proof/permission/Evidence 的有界 trace，不代替生产。"""
    keys = []
    def ref(key):
        if not key:
            return 0
        if key not in keys:
            keys.append(key)
        return keys.index(key) + 1
    binding = BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), scope_key=execution.query_key)
    permission = EvidenceEntry(execution.witness.source.stable_key(), execution.witness.program.operation_key,
        polarity=3, evidence_key=(92230, *execution.query_key), scope_key=execution.query_key,
        payload_key=execution.stable_key())
    ready = execution.status == 1
    plan = ({"claim_refs": [list(facts[0].proposition_key)], "claim_slot_ranges": [],
        "slot_sequence": [{"role": list(row.role.stable_key()), "filler": list(row.filler.stable_key())}
                          for row in facts[0].definition.bindings]} if ready else None)
    generation = ({"operation_execution": {"integer_key_ref": ref(execution.stable_key())},
        "operation_truth_added": 0, "surface_values": [88]} if ready else None)
    scalars = [0] * 17
    scalars[7], scalars[15] = int(ready), int(ready)
    value = {"schema_version": 4, "integer_key_encoding": "INTEGER_KEY_TABLE_V1",
        "integer_key_table": keys, "query_ref": ref(execution.query_key),
        "operation_keys": [list(execution.witness.program.operation_key)],
        "operation_contracts": [[list(execution.witness.program.operation_key), [91672, 4]]],
        "input_structure": prepared.integer_trace(), "termination": 1 if ready else 3,
        "response_plan": plan, "generation": generation, "response_surface": "X" if ready else "",
        "state_key": {"query_ref": ref(execution.query_key), "scalars": scalars,
            "bindings": [ref(binding.stable_key())],
            "evidence": [ref(permission.stable_key()), *(ref(key) for fact in facts for key in fact.evidence_keys)]}}
    value["integer_key_table"] = [list(key) for key in keys]
    return value


def test_actual_guard_edit_consumes_different_matches_and_fails_closed(pair):
    facts, _witnesses, prepared, executions, edit = pair
    result = audit_consumed_semantic_pair(edit, *(_trace(e, prepared, facts) for e in executions), facts, prepared)
    assert result["verified_semantic_pair_count"] == result["counterfactual_fail_closed"] == 1
    assert result["qualified_semantic_operation_families"] == result["free_dialogue_complete"] == 0
    assert "complete_matches" in result["changed_fields"]
    assert OperationSemanticEditReceipt.from_stable_key(edit.stable_key()) == edit


@pytest.mark.parametrize("change", ["key", "projection", "realization", "none", "permission", "fields", "digest"])
def test_labels_output_projection_or_unfrozen_changes_do_not_establish_semantic_edit(pair, change):
    _facts, witnesses, _prepared, _executions, edit = pair
    before = witnesses[0]
    changed = {"key": replace(before, program=replace(before.program, operation_key=(92910, 2))),
        "projection": replace(before, program=replace(before.program,
            projections=(OperationProjection(1, (RIGHT, LEFT)),))),
        "realization": replace(before, program=replace(before.program, realization_key=(1,))),
        "none": before, "permission": replace(edit.after, source_identity=(99,)),
        "fields": edit.after, "digest": edit.after}[change]
    with pytest.raises(ValueError):
        replace(edit, after=changed, changed_fields=() if change == "fields" else edit.changed_fields,
                after_graph_sha256=edit.before_graph_sha256 if change == "digest" else edit.after_graph_sha256)
    if change in {"key", "projection", "realization", "none"}:
        assert semantic_selection_obligation(before.program) == semantic_selection_obligation(changed.program)


@pytest.mark.parametrize("change", ["binding", "permission", "variables", "premise", "scope", "output", "plan", "context"])
def test_missing_or_transplanted_evidence_never_becomes_semantic_admission(pair, change):
    facts, witnesses, prepared, executions, edit = pair
    baseline, counter = (_trace(e, prepared, facts) for e in executions)
    if change == "binding":
        baseline["state_key"]["bindings"] = []
    elif change == "permission":
        baseline["state_key"]["evidence"] = baseline["state_key"]["evidence"][1:]
    elif change == "premise":
        baseline["state_key"]["evidence"] = baseline["state_key"]["evidence"][:1]
    elif change == "scope":
        foreign = BindingEntry(OPERATION_EXECUTION_ROLE, executions[0].stable_key(), scope_key=(88,))
        baseline["integer_key_table"].append(list(foreign.stable_key()))
        baseline["state_key"]["bindings"] = [len(baseline["integer_key_table"])]
    elif change == "variables":
        first, second = executions[0].variables
        forged = replace(executions[0], variables=((first[0], second[1]), (second[0], first[1])))
        binding = BindingEntry(OPERATION_EXECUTION_ROLE, forged.stable_key(), scope_key=forged.query_key)
        baseline["integer_key_table"].append(list(binding.stable_key()))
        baseline["state_key"]["bindings"] = [len(baseline["integer_key_table"])]
    elif change == "output":
        counter["response_surface"] = "X"
    elif change == "plan":
        baseline["response_plan"]["slot_sequence"][0]["filler"] = [88]
    else:
        edit = replace(edit, context_identity=(3,) * 32)
    with pytest.raises(ValueError):
        audit_consumed_semantic_pair(edit, baseline, counter, facts, prepared)


def test_unused_guard_edit_does_not_count_as_observed_behavior(pair):
    facts, witnesses, prepared, executions, edit = pair
    unchanged = _witness(replace(witnesses[0].program, guards=()), facts)
    unguarded = execute_integer_operation(unchanged, prepared.source_ref, executions[0].variables, facts)
    with pytest.raises(ValueError, match="did not change actual"):
        audit_consumed_semantic_pair(replace(edit, after=unchanged), _trace(executions[0], prepared, facts),
            _trace(unguarded, prepared, facts), facts, prepared)
