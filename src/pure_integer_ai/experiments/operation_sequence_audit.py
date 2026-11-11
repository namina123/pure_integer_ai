"""独立冷核实际序列生成；只读当前图和保存的表示，不执行 G planner。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer, representation_parts,
)
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.native_operation_call_audit import (
    read_native_member_representations, verify_native_plan_components,
)
from pure_integer_ai.experiments.operation_sequence_graph import (
    SEQUENCE_KIND, SEQUENCE_STEP, SEQUENCE_TRACE,
)
from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.trained_generation_connector_runtime import _generation_protocols


def _verify_native_step_trace(generator, template, bound, representations, key):
    """逐槽核对保存的 R-01 路由与 renderer；G 层内容引用只保留为原始证据。"""
    if (len(key) < 2 or key[0] != 2 or key[1] != len(representations) + 2
            or any(type(value) is not int or value < 0 for value in key)):
        raise ValueError("native sequence step lacks the original complete native trace")
    fields = _unpack(key, 2, key[1])
    recovered = generator.connector_for(bound, branch=template.language_branch,
                                         connector_identity=template.connector)
    routes = tuple(recovered.surface_for(slot.filler).stable_key()
                   for slot in recovered.connector.registry.values(template, bound))
    protocols = _generation_protocols(generator.context, template.language_branch)
    family = representation_parts(representations[0])[0]
    rendered = UnicodeRepresentationRenderer(family, protocols.renderer).render(representations)
    if not fields[0] or fields[1] != rendered.stable_key() or fields[2:] != routes:
        raise ValueError("native sequence step trace differs from cold native slot routes or rendering")


def verify_native_operation_sequence(core, generator, execution, value, trace, rules, variants, witnesses):
    """从物理规则独立选择 body* terminal，核对完整 Plan、成员、角色、表示和原调用 trace。"""
    encoded = _decode_input(value["generation"], trace)
    selected = tuple(rule for rule in rules
                     if rule.stable_key() == tuple(encoded.get("native_sequence_witness", ())))
    key = execution.witness.program.realization_key
    if len(selected) != 1 or len(key) < 2 or key[0] != SEQUENCE_KIND:
        raise ValueError("native sequence generation lacks one exact physical rule")
    rule = selected[0]
    if (rule.rule_id not in key[1:] or rule.source_identity != execution.witness.source_identity
            or rule.license_identity != execution.witness.license_identity
            or tuple(encoded["connector"]) != rule.connector.stable_key()
            or "native_composition_witness" in encoded):
        raise ValueError("native sequence generation rule or source permission drifted")
    chosen, sources, plan = verify_native_plan_components(core, generator, execution, value, trace, encoded)
    predicates = tuple(fact.predicate.stable_key() for fact in chosen)
    authorized = tuple(item for item in rules if item.rule_id in key[1:]
        and item.source_identity == execution.witness.source_identity
        and item.license_identity == execution.witness.license_identity
        and item.minimum_claims <= len(chosen) <= item.maximum_claims
        and all(predicate in {row[0] for row in item.members} for predicate in predicates))
    if authorized != (rule,):
        raise ValueError("native sequence generation permission is missing or ambiguous")
    if (not rule.minimum_claims <= len(chosen) <= rule.maximum_claims
            or len(chosen) == 0):
        raise ValueError("native sequence length is outside the physical grammar")
    # 独立构造成员序列，不调用 runtime 的 selected_members 或生成 API。
    by_predicate = {predicate: (body, terminal) for predicate, body, terminal in rule.members}
    if any(predicate not in by_predicate for predicate in predicates):
        raise ValueError("native sequence contains an untrained predicate")
    member_keys = tuple(by_predicate[predicate][1 if index + 1 == len(chosen) else 0]
                        for index, predicate in enumerate(predicates))
    fields = unpack_record(tuple(encoded["trace"]), SEQUENCE_TRACE)
    if len(fields) != len(chosen) + 1 or fields[0] != rule.stable_key():
        raise ValueError("native sequence trace lost the complete rule or a current claim")
    representations, reader_count = [], 0
    for index, (fact, source, member_key, saved) in enumerate(
            zip(chosen, sources, member_keys, fields[1:], strict=True)):
        native_witnesses = tuple(item for item in witnesses if item.stable_key() == member_key)
        if len(native_witnesses) != 1:
            raise ValueError("native sequence member lacks one complete physical witness")
        witness = native_witnesses[0]
        native_variants = tuple(item for item in variants if item.variant_id == witness.variant_id)
        if (len(native_variants) != 1 or native_variants[0].source_identity != rule.source_identity
                or native_variants[0].license_identity != rule.license_identity):
            raise ValueError("native sequence member permission differs from the learned rule")
        template, bound, current = read_native_member_representations(generator,
            (execution.outputs[index],), (fact,), (source,), native_variants[0], witness)
        parts = unpack_record(saved, SEQUENCE_STEP)
        if (len(parts) != 5 or parts[0] != fact.proposition.stable_key()
                or parts[1] != member_key or parts[2] != template.connector.stable_key()
                or unpack_record(parts[3], SEQUENCE_STEP)
                    != tuple(item.stable_key() for item in current)):
            raise ValueError("native sequence step differs from its current claim or cold native members")
        _verify_native_step_trace(generator, template, bound, current, parts[4])
        representations.extend(current)
        reader_count += len(witness.readers)
    units = tuple(unit for item in representations for unit in representation_parts(item)[1])
    if (len({representation_parts(item)[0] for item in representations}) != 1
            or tuple(map(tuple, encoded["representations"])) != tuple(item.stable_key() for item in representations)
            or units != tuple(encoded["surface_values"])
            or units != tuple(map(ord, value["response_surface"]))):
        raise ValueError("native sequence delivered surface differs from the exact cold native representations")
    return {"native_representation_count": len(representations), "scoped_reader_count": reader_count,
            "claim_count": len(chosen), "semantic_slot_count": len(plan["slot_sequence"]),
            "native_sequence_rule_id": rule.rule_id, "native_single_claim_step_count": len(chosen),
            "g_planner_reexecution_count": 0, "native_layer_content_hashes_independently_reconstructed": 0}
