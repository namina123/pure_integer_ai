"""按请求稀疏恢复当前 Core 完整证据；全量模式仅供独立对照。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.integer_operation_execution import (
    OperationGraphFact, request_environment,
)
from pure_integer_ai.cognition.shared.integer_operation_context_join import join_operation_context
from pure_integer_ai.cognition.shared.integer_operation_program import OperationProgramBudget
from pure_integer_ai.experiments.trained_query_evidence import (
    core_query_evidence, current_query_evidence,
)
from pure_integer_ai.storage.telemetry import record_candidate_count, record_diagnostic_event


def request_context_proposition_keys(witness, structure):
    """仅定位当前输入 reader 引用的命题；变量和许可仍由原消费门完整核验。"""
    if structure.operation_request_bindings:
        referenced = {key for binding in structure.operation_request_bindings
                      for _variable, _target, _span, keys in binding.readings for key in keys}
        return tuple(sorted({member.proposition_key for member in structure.semantic_candidates
                             if member.stable_key() in referenced}))
    relations = {row.stable_key(): row for row in structure.relation_candidates}
    return tuple(sorted({relations[binding.relation_key].proposition_key
        for binding in structure.relation_question_bindings
        if binding.operation_key == witness.program.operation_key
        and binding.relation_key in relations}))


def _current_fact(core, definition, candidate):
    """从同一 H-05 声明恢复当前快照及全部未替代 Evidence，不按 active 过滤。"""
    engine = core.owner.learning.engine
    hypothesis = candidate.hypothesis(engine.protocol)
    registry = core.context.scoped_identity_store.registry
    history = engine.ledger.evidence_history(hypothesis)
    evidence = current_query_evidence(core_query_evidence(definition.proposition, history, registry))
    return OperationGraphFact(definition, engine.ledger.snapshot(hypothesis), candidate,
                              tuple(sorted(entry.stable_key() for entry in evidence)))


def _sparse_definitions(core, witness, variables, budget):
    """使用权威反向索引枚举完整请求候选；读取和路径展开均有显式整数预算。"""
    program, graph = witness.program, core.owner.semantic_graph
    external = request_environment(program, variables)
    selected, cached = {}, {}
    attempts = 0

    def lookup(predicate, role=None, filler=None):
        """同次读取复用完整索引结果，任何预算失败都不交付部分上下文。"""
        nonlocal attempts
        key = predicate, role, filler
        if key not in cached:
            attempts += 1
            if attempts > budget.max_match_attempts:
                raise ValueError("operation Core lookup budget exceeded; no partial result")
            record_diagnostic_event("operation_core_index_lookups", 1)
            rows = (graph.lookup_atomic_by_predicate(predicate) if role is None else
                    graph.lookup_atomic_by_binding(predicate, role, filler))
            if len(rows) > budget.max_facts:
                raise ValueError("operation Core candidate budget exceeded; no partial result")
            cached[key] = rows
        return cached[key]

    def include(definition):
        """按完整命题身份去重并在物化 ledger 之前执行候选上限。"""
        key = definition.proposition.stable_key()
        if key in selected and selected[key] != definition:
            raise ValueError("operation Core proposition identity differs")
        selected[key] = definition
        if len(selected) > budget.max_facts:
            raise ValueError("operation Core candidate budget exceeded; no partial result")

    if program.path_query is None:
        for row in join_operation_context(program, external, lookup, budget):
            include(row.definition)
    else:
        path = program.path_query
        frontier = {external[path.start_variable]}
        target = external[path.end_variable]
        expansions = 0
        for _depth in range(program.maximum_matches):
            following = set()
            for current in sorted(frontier, key=ObjectIdentity.stable_key):
                for index, left, right in path.edge_templates:
                    schema = program.atoms[index - 1].schema
                    for row in lookup(schema.relation, left, current):
                        expansions += 1
                        if expansions > budget.max_match_attempts:
                            raise ValueError("operation Core path budget exceeded; no partial result")
                        schema.validate_definition(row.definition)
                        include(row.definition)
                        fillers = {item.role: item.filler for item in row.definition.bindings}
                        if fillers[right] != target:
                            following.add(fillers[right])
            frontier = following
            if not frontier:
                break
    return tuple(selected[key] for key in sorted(selected))


def read_operation_core_context(core, budget=OperationProgramBudget(), *,
                                proposition_keys=None, witness=None, variables=None):
    """恢复全量对照、明确输入命题或当前程序上下文，三种模式互斥且保持原证据合同。"""
    if (witness is None) != (variables is None) or (
            proposition_keys is not None and witness is not None):
        raise ValueError("operation Core context selectors must be exclusive and complete")
    engine = core.owner.learning.engine
    if proposition_keys is None and witness is None:
        candidates = engine.definitions()
        if len(candidates) > budget.max_facts:
            raise ValueError("operation Core fact budget exceeded")
        definitions = []
        for candidate in candidates:
            ref = core.context.graph_ontology.resolve(candidate.candidate)
            if ref is None:
                raise ValueError("operation Core declaration is not materialized")
            definitions.append((core.owner.semantic_graph.read_atomic(ref).definition, candidate))
    else:
        if witness is not None:
            selected = _sparse_definitions(core, witness, variables, budget)
        else:
            keys = tuple(sorted(set(proposition_keys)))
            if len(keys) > budget.max_facts:
                raise ValueError("operation Core fact budget exceeded")
            selected = []
            for key in keys:
                identity = ObjectIdentity.from_stable_key(key)
                ref = core.context.graph_ontology.resolve(identity)
                if ref is None:
                    raise ValueError("operation Core input proposition is not materialized")
                selected.append(core.owner.semantic_graph.read_atomic(ref).definition)
        definitions = []
        for definition in selected:
            hypothesis = core.owner.learning.hypothesis_for_candidate(definition.proposition)
            definitions.append((definition, engine.definition(hypothesis)))
    record_candidate_count("operation_core_context", len(definitions))
    result = tuple(sorted((_current_fact(core, definition, candidate)
        for definition, candidate in definitions), key=lambda fact: fact.proposition_key))
    return result


__all__ = ["read_operation_core_context", "request_context_proposition_keys"]
