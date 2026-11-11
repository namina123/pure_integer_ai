"""按当前执行证明复用单 claim 原生语法；不存储完整谓词序列或回答。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.experiments.operation_composition_course import OperationCompositionWitness
from pure_integer_ai.experiments.operation_sequence_graph import (
    SEQUENCE_KIND, SEQUENCE_STEP, SEQUENCE_TRACE, read_claim_sequence_rules,
)
from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase
from pure_integer_ai.experiments.trained_multi_claim_composer import PART_CLAIM_PREDICATE


# object-model: resource_owner; representation=runtime; interop=native-claim-sequence-v1
class TrainedOperationSequenceRuntime:
    """只读生成 owner：持有物理规则及原生组件 owner，统一校验后有界执行。"""

    def __init__(self, sequence_database, native_runtime):
        """冷恢复规则并核对所有完整原生成员，不使用课程正文。"""
        self.rules = read_claim_sequence_rules(sequence_database)
        self._native = native_runtime
        variants = {variant.variant_id: variant for variant in native_runtime.variants}
        for rule in self.rules:
            for predicate, body, terminal in rule.members:
                for key in (body, terminal):
                    witness = OperationCompositionWitness.from_stable_key(key)
                    current = native_runtime.operation_composition_witness_for(witness.variant_id)
                    variant = variants.get(witness.variant_id)
                    if (current != witness or variant is None or variant.claim_count != 1
                            or variant.source_identity != rule.source_identity
                            or variant.license_identity != rule.license_identity
                            or tuple(part[2] for part in variant.parts if part[0] == PART_CLAIM_PREDICATE)
                            != (predicate,)):
                        raise ValueError("native sequence member differs from its physical native graph")

    def __getattr__(self, name):
        """保留原生关系生成与固定组合接口，由共同 production caller 消费。"""
        return getattr(self._native, name)

    def select_operation_sequence(self, execution, predicates):
        """当前 program、谓词、长度、来源和许可必须唯一许可物理规则。"""
        key = execution.witness.program.realization_key
        matches = []
        if len(key) < 2 or key[0] != SEQUENCE_KIND:
            raise ValueError("operation has no learned native sequence permission")
        for rule in self.rules:
            if (rule.rule_id not in key[1:]
                    or rule.source_identity != execution.witness.source_identity
                    or rule.license_identity != execution.witness.license_identity):
                continue
            try:
                rule.selected_members(predicates)
            except ValueError:
                continue
            matches.append(rule)
        if len(matches) != 1:
            raise ValueError("operation native sequence permission is missing or ambiguous")
        return matches[0]

    @observed_relation_phase("generation")
    def generate_operation_sequence(self, components, state, *, execution, rule_id):
        """按 body* terminal 执行实际单 claim 图，逐步 trace 保留全部角色来源。"""
        self._native.validate_current_operation(components, state, execution)
        predicates = tuple(component[1].predicate.stable_key() for component in components)
        rule = self.select_operation_sequence(execution, predicates)
        if rule.rule_id != rule_id:
            raise ValueError("native sequence selection drifted")
        generated, steps = [], []
        for component, key in zip(components, rule.selected_members(predicates), strict=True):
            witness = OperationCompositionWitness.from_stable_key(key)
            current = self._native._generate_components((component,), witness.variant_id,
                                                        rule.source_identity, rule.license_identity)
            generated.append(current)
            steps.append(pack_record(SEQUENCE_STEP, component[1].proposition.stable_key(), key,
                current.connector.stable_key(),
                pack_record(SEQUENCE_STEP, *(item.stable_key() for item in current.representations)), current.trace))
        representations = tuple(item for current in generated for item in current.representations)
        if len({representation_parts(item)[0] for item in representations}) != 1:
            raise ValueError("native sequence representation families differ")
        units = tuple(unit for item in representations for unit in representation_parts(item)[1])
        return replace(generated[0], surface="".join(map(chr, units)), connector=rule.connector,
            frame_source_hash=int.from_bytes(bytes(rule.source_digest[:8]), "big") & ((1 << 63) - 1) or 1,
            representations=representations, trace=pack_record(SEQUENCE_TRACE, rule.stable_key(), *steps),
            slot_count=sum(len(component[1].bindings) for component in components), structure_plan=None)
