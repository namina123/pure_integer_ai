"""通过真实 G-00--G-03/S-07/R-01 owner 只读生成多 claim。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.shared.integer_operation_execution import READY
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import BindingEntry
from pure_integer_ai.cognition.shared.typed_binding import BoundRoleBinding
from pure_integer_ai.experiments.operation_composition_materialization import read_native_operation_composition_graph
from pure_integer_ai.experiments.operation_program_query import (
    OPERATION_EXECUTION_ROLE, operation_evidence_matches, read_operation_core_context,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import _bound_relation_proposition, _generation_protocols
from pure_integer_ai.experiments.trained_multi_claim_composer import GRAPH_KIND, PART_CLAIM_PREDICATE, PART_CLAIM_ROLE
from pure_integer_ai.experiments.trained_operation_program_graph import read_operation_program_graph
from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase
from pure_integer_ai.experiments.operation_realization_contract import require_native_operation_role_order


class TrainedOperationCompositionRuntime:
    """只读资源 owner，持有冷恢复的原生语法和当前 Core。"""

    def __init__(self, graph_database, relation_generator, core, operation_program_database):
        """所有模板必须对应同一 Core 中唯一的完整原生 connector。"""
        if relation_generator.path != core.path:
            raise ValueError("native composition must share the exact readonly Core source")
        self.variants, self.witnesses, self.metadata = read_native_operation_composition_graph(graph_database)
        self.operation_witnesses = read_operation_program_graph(operation_program_database)
        self._generator, self._core = relation_generator, core
        templates = relation_generator.templates()
        self._templates = {}
        for witness in self.witnesses:
            matches = tuple(template for template in templates if template.stable_key() == witness.template_key)
            if len(matches) != 1:
                raise ValueError("native composition graph has no exact cold G/S-07 connector")
            self._templates[witness.variant_id] = matches[0]

    def __getattr__(self, name):
        """共同 caller 的原子生成接口由同一底层 owner 承担。"""
        return getattr(self._generator, name)

    def operation_composition_witness_for(self, variant_id):
        """按课程授权 ID 返回唯一完整 witness，缺失或重复均拒绝。"""
        matches = tuple(witness for witness in self.witnesses if witness.variant_id == variant_id)
        if len(matches) != 1:
            raise ValueError("native composition witness is not unique")
        return matches[0]

    @observed_relation_phase("generation")
    def generate_operation_composition(self, components, state, *, execution, variant_id):
        """当前完整执行授权固定语法；动态角色和文字均走原生生成。"""
        self.validate_current_operation(components, state, execution)
        program = execution.witness.program
        realization_key = program.realization_key
        permitted_variant = (realization_key == (GRAPH_KIND, variant_id)
            if program.path_query is None else
            len(realization_key) >= 2 and realization_key[0] == GRAPH_KIND
            and variant_id in realization_key[1:])
        if not permitted_variant:
            raise ValueError("native composition lacks the exact graph selection")
        return self._generate_components(components, variant_id,
            execution.witness.source_identity, execution.witness.license_identity)

    def validate_current_operation(self, components, state, execution):
        """核对完整当前执行和逐 claim 角色，供固定语法与序列语法共用。"""
        if (execution.status != READY or not state.evidence_closed or state.conflict_open
                or state.required_slots_open or state.query_key != execution.query_key
                or BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), space=1,
                    scope_key=state.query_key) not in state.bindings):
            raise ValueError("native composition lacks the actual closed QueryState operation proof")
        facts = read_operation_core_context(self._core,
            witness=execution.witness, variables=execution.variables)
        verify_integer_operation(execution, facts, self.operation_witnesses)
        if (not operation_evidence_matches(execution, facts, state.evidence)
                or tuple((item[1].proposition.stable_key(),
                    tuple(binding.role.stable_key() for binding in item[1].bindings)) for item in components)
                != execution.outputs):
            raise ValueError("native composition lacks the exact current claims, roles and premise Evidence")

    def _generate_components(self, components, variant_id, source_identity, license_identity):
        """在调用方已核验完整执行后，用原生图执行一个确切的语法成员。"""
        variants = tuple(variant for variant in self.variants if variant.variant_id == variant_id)
        witnesses = tuple(witness for witness in self.witnesses if witness.variant_id == variant_id)
        if len(variants) != 1 or len(witnesses) != 1:
            raise ValueError("native composition graph selection is not unique")
        variant, witness, template = variants[0], witnesses[0], self._templates[variant_id]
        if (variant.claim_count != len(components)
                or variant.source_identity != source_identity
                or variant.license_identity != license_identity):
            raise ValueError("native composition claims or physical source permission drifted")
        predicates = tuple(component[1].predicate.stable_key() for component in components)
        guards = tuple(part[2] for part in sorted((part for part in variant.parts
            if part[0] == PART_CLAIM_PREDICATE), key=lambda part: part[1]))
        if guards != predicates:
            raise ValueError("native composition has no exact ordered predicate guards")
        coordinates = tuple((part[1], part[2]) for part in variant.parts if part[0] == PART_CLAIM_ROLE)
        expected = tuple((claim, index) for claim, component in enumerate(components, 1)
                         for index in range(1, len(component[1].bindings) + 1))
        if coordinates != expected:
            raise ValueError("native composition must mention each claim-local role exactly once in order")
        reader_keys = {(claim, role.stable_key()) for claim, role, _scoped, _ordinal in witness.readers}
        if reader_keys != {(claim, binding.role.stable_key()) for claim, component in enumerate(components, 1)
                           for binding in component[1].bindings}:
            raise ValueError("native composition scoped role readers do not cover the actual claims")
        require_native_operation_role_order(tuple((fact.proposition.stable_key(),
            tuple(binding.role.stable_key() for binding in fact.bindings))
            for _source, fact, _generated in components), template, witness)
        bindings = []
        for claim, role, scoped, ordinal in witness.readers:
            source, fact, _generated = components[claim - 1]
            current = self._core.generation_input(fact.proposition)
            if source != current:
                raise ValueError("native composition component is not the exact current Core generation input")
            matches = tuple(binding for binding in source.proposition.definition.bindings
                            if binding.role == role and binding.ordinal == ordinal)
            if len(matches) != 1:
                raise ValueError("native composition role/ordinal does not select one current Core filler")
            bindings.append(BoundRoleBinding(scoped, matches[0].filler, 0))
        source = components[0][0]
        protocols = _generation_protocols(self._generator.context, template.language_branch)
        bound = replace(_bound_relation_proposition(source, template, protocols.content.answer),
                        bindings=tuple(bindings))
        # 这里只组织已核验的 claim；原子事实及其 Evidence 继续由 Plan 承载。
        generated = self._generator._generate_bound_relation(bound, template, protocols,
            source.snapshot, source.evidence, source.proposition.scope, witness.source.source_id)
        return replace(generated, slot_count=sum(len(component[1].bindings) for component in components))
