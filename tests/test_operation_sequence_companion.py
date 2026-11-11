"""真实原生单 claim 语法重复组合，以及来源缺失和原预算关闭部分回答。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from test_native_operation_composition import _actual_components, prepared  # noqa: F401
from pure_integer_ai.cognition.shared.query_state import (
    TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CLARIFY_MISSING_BINDING, QueryBudget,
)
from pure_integer_ai.experiments.operation_companion_query import (
    consume_operation_companion, operation_companion_consumed,
)
from pure_integer_ai.experiments.operation_composition_materialization import materialize_native_operation_compositions
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_program_generation import generate_operation_response
from pure_integer_ai.experiments.operation_sequence_course import compile_train_claim_sequence
from pure_integer_ai.experiments.operation_sequence_graph import SEQUENCE_KIND, materialize_claim_sequence_rules
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.trained_operation_composition import TrainedOperationCompositionRuntime
from pure_integer_ai.experiments.trained_operation_program_graph import materialize_operation_program_graph
from pure_integer_ai.experiments.trained_operation_sequence import TrainedOperationSequenceRuntime


def _indexes(core):
    """测试只读取训练 Core 的真实来源和完整角色，不构造影子证据。"""
    return ({frame.proposition.stable_key(): frame for frame in core.active_surface_frames()},
            {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()})


@pytest.mark.parametrize("failure", ["missing_frame", "changed_frame", "missing_evidence", "budget_after_one"])
def test_companion_rejects_incomplete_sources_and_budget_without_partial_answer(prepared, failure):
    root, _, _, witness = prepared
    core, generator, execution, _, state = _actual_components(root, witness)
    try:
        frames, facts = _indexes(core)
        premises = sorted({key for match in execution.matches for key in match})
        if failure == "missing_frame":
            del frames[premises[-1]]
        elif failure == "changed_frame":
            key = premises[-1]
            frames[key] = replace(frames[key], source_hash=frames[key].source_hash + 1)
        elif failure == "missing_evidence":
            state = state.with_(evidence=tuple(item for item in state.evidence
                if item.hypothesis_key != premises[-1]))
        else:
            # 首条前提有两条 H-00 记录时成本高于预检下界，必须在原预算内止损。
            total = sum(2 + len(core.evidence_history(facts[key].proposition)) for key in premises)
            state = state.with_(budget=QueryBudget(max_reads=max(1, total - 1)))
        changed, _reads = consume_operation_companion(core, frames, facts, execution, state)
        expected = TERMINATION_BUDGET_EXHAUSTED if failure == "budget_after_one" else TERMINATION_CLARIFY_MISSING_BINDING
        assert changed.termination == expected
        assert changed.generation_ready == 0 and changed.evidence_closed == 0
        assert changed.read_count <= changed.budget.max_reads
        assert not operation_companion_consumed(execution, changed)
        _, plan, generated = generate_operation_response(core, generator, execution, changed)
        assert plan is None and generated is None
    finally:
        generator.close()
        core.close()


def test_learned_native_sequence_reuses_one_predicate_grammar_and_requires_actual_companion(prepared):
    root, facts, _, old_witness = prepared
    core, generator, _, components, _ = _actual_components(root, old_witness)
    try:
        claim = components[0][1]
        instruction = root / "sequence.train.json"
        instruction.write_bytes(canonical_json_bytes({
            "rule_id": 1, "minimum_claims": 1, "maximum_claims": 6,
            "members": [{"claim_key": list(claim.proposition.stable_key()),
                "output_gaps": [[], list(map(ord, "归于")), []],
                "body_variant_id": 2, "terminal_variant_id": 3}],
            "separator_values": [65307], "owner": "teacher", "split": "train", "license_id": "CC0-1.0",
        }) + b"\n")
        rule, courses = compile_train_claim_sequence(instruction, root / "sequence-courses", facts, core, generator)
    finally:
        generator.close()
        core.close()
    materialize_native_operation_compositions(courses, root / "core.sqlite3", root / "sequence-native.sqlite3")
    materialize_claim_sequence_rules((rule,), root / "sequence.sqlite3")
    source = root / "sequence-program.train.json"
    source.write_bytes(canonical_json_bytes({
        "program_key": list(replace(old_witness.program, realization_key=(SEQUENCE_KIND, 1)).stable_key()),
        "owner": "teacher", "split": "train", "license_id": "CC0-1.0",
    }) + b"\n")
    witness = compile_train_operation_program(source, facts)
    materialize_operation_program_graph((witness,), root / "sequence-program.sqlite3")
    core, generator, execution, components, state = _actual_components(root, witness)
    try:
        native = TrainedOperationCompositionRuntime(root / "sequence-native.sqlite3", generator, core,
                                                   root / "sequence-program.sqlite3")
        runtime = TrainedOperationSequenceRuntime(root / "sequence.sqlite3", native)
        assert generate_operation_response(core, runtime, execution, state)[1:] == (None, None)
        state, reads = consume_operation_companion(core, *_indexes(core), execution, state)
        assert len(reads) == 2 and operation_companion_consumed(execution, state)
        _ready, plan, generated = generate_operation_response(core, runtime, execution, state)
        assert plan is not None and len(plan.claim_refs) == 2
        assert plan.surface() == "；".join(f"{fact.bindings[0].surface}归于{fact.bindings[1].surface}"
                                          for _, fact, _ in components)
        assert generated["native_sequence_witness"] == list(rule.stable_key())
        assert len(runtime.variants) == 2 and all(variant.claim_count == 1 for variant in runtime.variants)
        with pytest.raises(ValueError, match="outside"):
            rule.selected_members((claim.predicate.stable_key(),) * 7)
        with pytest.raises(ValueError, match="outside"):
            rule.selected_members(((999999,),))
    finally:
        generator.close()
        core.close()
