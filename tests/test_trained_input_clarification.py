"""未绑定输入只可消费当前 UNKNOWN 观察，不能借事实根或历史生成作答。"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.query_driver import ExpansionOutcome
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY, ROOT_EXPANDED, ROOT_PENDING, SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY,
    TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CLARIFY_CONFLICT,
    BindingEntry, EvidenceEntry, FrontierEntry,
    QueryAnchor, QueryBudget, QueryRoot, seed_query,
)
from pure_integer_ai.experiments.trained_input_clarification import (
    INPUT_CLARIFICATION_VERSION, clarification_query_state, run_input_clarification,
    unresolved_role_context_allowed,
)


def _query():
    """构造同一查询的事实、当前观察、旧观察和两个生成根。"""
    roots = tuple(QueryRoot(space, key, ROOT_PENDING, 1) for space, key in (
        (SPACE_CORE, (10,)), (SPACE_MEMORY, (20,)), (SPACE_MEMORY, (21,)),
        (SPACE_DIALOGUE, (30,)), (SPACE_DIALOGUE, (31,))))
    frontier = tuple(FrontierEntry((root.owner_space, *root.root_key), root.owner_space,
                                   target_key=root.root_key, root_key=root.root_key)
                     for root in roots)
    state = seed_query(
        (99,), roots=roots,
        anchors=tuple(QueryAnchor(root.owner_space, root.root_key, kind=1)
                      for root in roots),
        budget=QueryBudget(max_nodes=17, max_edges=19, max_reads=23, max_depth=7),
    ).with_(evidence=(EvidenceEntry((40,), (10,), space=SPACE_CORE, polarity=3),))
    hypotheses = (SimpleNamespace(observation_key=(20,)),)
    nodes = (SimpleNamespace(key=(30,)),)
    return state, frontier, hypotheses, nodes


def test_clarification_preserves_budget_and_denied_roots_without_expanding_them():
    """历史及事实根不能参与，完整拒绝记录和输入投影仍在原查询身份中。"""
    state, frontier, hypotheses, nodes = _query()
    result = clarification_query_state(state, frontier, hypotheses, nodes)
    assert result.query_key == state.query_key and result.budget == state.budget
    assert result.active_spaces == state.active_spaces and result.evidence == state.evidence
    assert {(root.owner_space, root.root_key) for root in result.roots
            if root.status == ROOT_PENDING} == {
                (SPACE_MEMORY, (20,)), (SPACE_DIALOGUE, (30,))}
    assert all(root.status in (ROOT_PENDING, ROOT_EMPTY) for root in result.roots)
    assert {(edge.owner_space, edge.root_key) for edge in result.frontier} == {
        (SPACE_MEMORY, (20,)), (SPACE_DIALOGUE, (30,))}
    denied = {item.filler_key for item in result.bindings
              if item.role_key == (INPUT_CLARIFICATION_VERSION, 1)}
    assert denied == {root.stable_key() for root in state.roots
                      if root.root_key in {(10,), (21,), (31,)}}
    assert all(root.status == ROOT_PENDING for root in state.roots)


def test_clarification_rejects_conflict_truth_evidence_and_missing_current_roots():
    """不能删除竞争或真值证据来获得非事实输出，也不能借别轮观察。"""
    state, frontier, hypotheses, nodes = _query()
    assert clarification_query_state(state.with_(conflict_open=1), frontier, hypotheses, nodes) is None
    positive = replace(state.evidence[0], polarity=1)
    assert clarification_query_state(state.with_(evidence=(positive,)), frontier, hypotheses, nodes) is None
    conflict = BindingEntry((50,), (51,), space=SPACE_CORE, conflict_kept=1)
    assert clarification_query_state(state.with_(bindings=(conflict,)), frontier, hypotheses, nodes) is None
    assert clarification_query_state(state, frontier, (*hypotheses, SimpleNamespace(
        observation_key=(21,))), nodes) is None
    assert clarification_query_state(state, frontier, (SimpleNamespace(
        observation_key=(22,)),), nodes) is None


def test_clarification_cannot_reopen_or_increase_an_exhausted_query_budget():
    """预算已耗尽时不调用扩展器，不用额外查询生成兜底文本。"""
    state, frontier, hypotheses, nodes = _query()
    state = state.with_(node_count=state.budget.max_nodes + 1)

    def forbidden_expand(*_args):
        raise AssertionError("耗尽预算不得展开")

    expander = SimpleNamespace(expand=forbidden_expand)
    tracer = SimpleNamespace(record=lambda *_args: None)
    result, generic = run_input_clarification(
        state, frontier, hypotheses, nodes, expander=expander, tracer=tracer)
    assert result.termination == TERMINATION_BUDGET_EXHAUSTED
    assert result.budget == state.budget and generic is None


@pytest.mark.parametrize("generated", (False, True))
@pytest.mark.parametrize("with_input_role", (False, True))
def test_endpoint_conflict_uses_only_current_frontier_and_retains_conflict(generated, with_input_role):
    """端点竞争可交付零事实澄清，但不能访问事实/历史根或清除最终歧义。"""
    state, frontier, hypotheses, nodes = _query()
    calls, generations, hops, records = [], [], [], []
    role_key = (91534, 7, 44)
    roles = ()
    if with_input_role:
        role = SimpleNamespace(observation=SimpleNamespace(stable_key=lambda: (20,)),
                               stable_key=lambda: role_key)
        roots = (*state.roots, QueryRoot(SPACE_MEMORY, role_key, ROOT_PENDING, 1))
        state = state.with_(roots=roots)
        frontier = (*frontier, FrontierEntry((SPACE_MEMORY, *role_key), SPACE_MEMORY,
            target_key=role_key, root_key=role_key))
        roles = (role,)
    generic = SimpleNamespace(
        response_plan=SimpleNamespace(claim_refs=(), claim_evidence_refs=()),
        graph_surfaces=())

    def expand(current, edge):
        """按协议实际推进当前两个准入根，记录触达的space和完整root。"""
        calls.append((edge.owner_space, edge.root_key))
        hops.append(SimpleNamespace(space=edge.owner_space))
        if edge.owner_space == SPACE_DIALOGUE and generated:
            generations.append(generic)
        result = current.with_(
            roots=tuple(replace(root, status=ROOT_EXPANDED)
                        if (root.owner_space, root.root_key) == calls[-1] else root
                        for root in current.roots),
            frontier=tuple(item for item in current.frontier if item != edge),
            read_count=current.read_count + 1, node_count=current.node_count + 1)
        if edge.root_key == role_key:
            result = result.with_(bindings=(*result.bindings,
                BindingEntry((44,), role_key, space=SPACE_MEMORY)),
                evidence=(*result.evidence,
                EvidenceEntry((40,), (20,), space=SPACE_MEMORY, polarity=3, evidence_key=role_key)))
        return ExpansionOutcome(result, edge, 1)

    expander = SimpleNamespace(expand=expand, hops=hops,
                              generic_response_generations=generations)
    tracer = SimpleNamespace(record=lambda current, reason: records.append((current, reason)))
    result, response = run_input_clarification(
        state, frontier, hypotheses, nodes, expander=expander, tracer=tracer,
        input_role_occurrences=roles,
        input_relation_conflict=True)
    expected = {(SPACE_MEMORY, (20,)), (SPACE_DIALOGUE, (30,))}
    if with_input_role:
        expected.add((SPACE_MEMORY, role_key))
        assert any(item.evidence_key == role_key and item.polarity == 3 for item in result.evidence)
        assert any(item.filler_key == role_key for item in result.bindings)
    else:
        assert result.evidence == state.evidence
    assert len(calls) == len(expected) and set(calls) == expected
    assert result.query_key == state.query_key and result.budget == state.budget
    assert result.read_count == len(expected)
    assert result.termination == TERMINATION_CLARIFY_CONFLICT
    assert result.conflict_open == 1 and result.evidence_closed == 0
    assert result.generation_ready == int(generated)
    assert response is (generic if generated else None)
    assert records[-1] == (result, TERMINATION_CLARIFY_CONFLICT)


@pytest.mark.parametrize("failure", ("conflict", "truth", "budget"))
def test_endpoint_conflict_cannot_reopen_existing_failures(failure):
    """已有冲突、真值或耗尽预算保持拒绝，新的歧义输出路径不能抹除它们。"""
    state, frontier, hypotheses, nodes = _query()
    if failure == "conflict":
        state = state.with_(conflict_open=1)
    elif failure == "truth":
        state = state.with_(evidence=(replace(state.evidence[0], polarity=1),))
    else:
        state = state.with_(node_count=state.budget.max_nodes + 1)

    def forbidden(*_args):
        raise AssertionError("已有失败不得展开非事实frontier")

    result, generic = run_input_clarification(
        state, frontier, hypotheses, nodes,
        expander=SimpleNamespace(expand=forbidden),
        tracer=SimpleNamespace(record=lambda *_args: None), input_relation_conflict=True)
    assert generic is None and result.budget == state.budget
    assert result.evidence == state.evidence
    if failure == "budget":
        assert result.termination == TERMINATION_BUDGET_EXHAUSTED
    else:
        assert result == state


def test_partial_input_retains_legal_memory_stances_and_unknown_core_projection():
    """缺槽并不否定已观察词项，非事实生成资格不能改写来源化合法立场。"""
    state, frontier, hypotheses, nodes = _query()
    support = EvidenceEntry((60,), (61,), space=SPACE_MEMORY, polarity=1,
                            evidence_key=(62,), scope_key=(63,))
    refute = EvidenceEntry((64,), (65,), space=SPACE_MEMORY, polarity=2,
                           evidence_key=(66,), scope_key=(67,))
    state = state.with_(evidence=(*state.evidence, support, refute), required_slots_open=2)
    result = clarification_query_state(state, frontier, hypotheses, nodes)
    assert result is not None
    assert {row.stable_key() for row in result.evidence} == {
        row.stable_key() for row in state.evidence}
    assert result.required_slots_open == 2 and result.evidence_closed == 0
    assert state.read_count == result.read_count == 0


def test_unresolved_role_expression_retains_truth_conflict_and_frontier_gates():
    """动态缺槽可表达 UNKNOWN，但未完成、真值、冲突及超预算均不能获此资格。"""
    state, _frontier, _hypotheses, _nodes = _query()
    assert not unresolved_role_context_allowed(state)
    completed = state.with_(roots=tuple(replace(root, status=ROOT_EXPANDED)
                                        for root in state.roots), frontier=())
    assert unresolved_role_context_allowed(completed)
    for stance in (1, 2):
        assert not unresolved_role_context_allowed(completed.with_(
            evidence=(replace(completed.evidence[0], polarity=stance),)))
    assert not unresolved_role_context_allowed(completed.with_(conflict_open=1))
    assert not unresolved_role_context_allowed(completed.with_(
        bindings=(BindingEntry((50,), (51,), space=SPACE_CORE, conflict_kept=1),)))
    assert not unresolved_role_context_allowed(completed.with_(
        node_count=completed.budget.max_nodes + 1))
    assert not unresolved_role_context_allowed(completed.with_(
        read_count=completed.budget.max_reads + 1))
    assert not unresolved_role_context_allowed(completed.with_(
        termination=TERMINATION_BUDGET_EXHAUSTED))


def test_current_open_role_syntax_is_not_content_truth_and_must_match_completely():
    """真实开放投影的语法存在证明可保留；来源、payload 或内容真值不能混入。"""
    from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
    from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
    from pure_integer_ai.cognition.understanding.query_open_role_state import open_role_query_nodes

    frame = OpenRoleFrame((1, 10), (5, 11), (9, 12), ((8, 1),),
                          ((44,), (88,)), (0, 3))
    structure = TrainedInputStructureProjector((), open_role_frames=(frame,)).project((44, 55, 88))
    node, = (item for item in open_role_query_nodes(structure.open_relations)
              if item.space == SPACE_CORE)
    root, anchor, edge = node.seed(3)
    state = seed_query((99,), roots=(root,), anchors=(anchor,)).with_(frontier=(edge,))
    expanded = node.expand(state, edge).with_(roots=(replace(root, status=ROOT_EXPANDED),))
    assert unresolved_role_context_allowed(expanded, structure)
    assert not unresolved_role_context_allowed(expanded)
    syntax, = expanded.evidence
    for changed in (replace(syntax, source_ref=(9, 99)),
                    replace(syntax, payload_key=(77,)), replace(syntax, polarity=2),
                    replace(syntax, hypothesis_key=(1, 42))):
        assert not unresolved_role_context_allowed(expanded.with_(evidence=(changed,)), structure)
    content = EvidenceEntry((9,), (1, 42), space=SPACE_CORE, polarity=1)
    assert not unresolved_role_context_allowed(expanded.with_(evidence=(syntax, content)), structure)
