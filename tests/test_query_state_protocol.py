"""阶段 B 查询内核协议专项：QueryState/frontier/termination/驱动。"""
from pure_integer_ai.cognition.shared.query_driver import (
    ExpansionOutcome,
    QueryExpander,
    run_query,
)
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED,
    ROOT_PENDING,
    SPACE_CORE,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_NO_FRONTIER,
    TERMINATION_OPEN,
    BindingEntry,
    FrontierEntry,
    QueryAnchor,
    QueryBudget,
    QueryRoot,
    QueryState,
    VisitedKey,
    seed_query,
    sort_frontier,
)
from pure_integer_ai.cognition.shared.query_termination import (
    TerminationPredicateInput,
    evaluate_termination,
)


def test_frontier_priority_orders_slot_gain_descending():
    shallow = FrontierEntry((2,), SPACE_CORE, depth=0, required_slot_gain=1)
    deep = FrontierEntry((1,), SPACE_CORE, depth=3, required_slot_gain=9)
    ordered = sort_frontier((shallow, deep))
    assert ordered[0].edge_key == (1,)
    # 相同 slot gain 时浅层优先（-depth 编码），edge_key 决胜。
    a = FrontierEntry((5,), SPACE_CORE, depth=2, required_slot_gain=7)
    b = FrontierEntry((4,), SPACE_CORE, depth=1, required_slot_gain=7)
    assert sort_frontier((a, b))[0].edge_key == (4,)
    # 语义 gain 相同时，owner 定点权重只改变共同 frontier 次序。
    core = FrontierEntry(
        (1,), SPACE_CORE, owner_weight=300, required_slot_gain=7)
    memory = FrontierEntry(
        (2,), SPACE_MEMORY, owner_weight=200, required_slot_gain=7)
    dialogue = FrontierEntry(
        (3,), SPACE_DIALOGUE, owner_weight=100, required_slot_gain=7)
    assert tuple(item.owner_space for item in sort_frontier(
        (dialogue, memory, core))) == (
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE)


def test_termination_reasons_are_protocol_ints():
    assert evaluate_termination(
        TerminationPredicateInput(
            depth=0, minimum_depth=1, node_count=0, edge_count=0,
            read_count=0, frontier_count=0),
        budgets=(10, 10, 10, 3)) == TERMINATION_NO_FRONTIER
    # 缺必要绑定：协议要求自然语言澄清，而不是假装回答或继续空转。
    assert evaluate_termination(
        TerminationPredicateInput(
            depth=1, minimum_depth=1, node_count=1, edge_count=0,
            read_count=1, frontier_count=0, required_slots_open=2,
            evidence_closed=0),
        budgets=(10, 10, 10, 3)) == TERMINATION_CLARIFY_MISSING_BINDING
    # 结构/证据/生成均闭合时，任何未扩展图库根仍阻止提前回答。
    assert evaluate_termination(
        TerminationPredicateInput(
            depth=1, minimum_depth=1, node_count=3, edge_count=3,
            read_count=3, frontier_count=2, roots_pending=1,
            required_slots_open=0, evidence_closed=1,
            generation_ready=1, best_score=3),
        budgets=(10, 10, 10, 3)) == TERMINATION_OPEN
    assert evaluate_termination(
        TerminationPredicateInput(
            depth=1, minimum_depth=1, node_count=3, edge_count=3,
            read_count=3, frontier_count=0, roots_pending=0,
            required_slots_open=0, evidence_closed=1,
            generation_ready=1, best_score=3),
        budgets=(10, 10, 10, 3)) == TERMINATION_ANSWER_CLOSED


class _LinearExpander(QueryExpander):
    """把一条边扩展为下一层节点：按 edge_key 前进步数。"""

    def __init__(self, steps: int, minimum_depth: int = 2):
        self.steps = steps
        self.minimum_depth = minimum_depth

    def expand(self, state: QueryState,
               edge: FrontierEntry) -> ExpansionOutcome:
        depth = state.depth + 1
        visited = tuple(sorted(
            state.visited
            + (VisitedKey(edge.owner_space, edge.edge_key, direction=1),),
            key=lambda item: item.stable_key()))
        if depth < self.steps:
            next_edge = FrontierEntry(
                (edge.edge_key[0] + 1,), edge.owner_space, depth=depth,
                required_slot_gain=10)
            frontier = (next_edge,)
            reason_open = 0
        else:
            frontier = ()
            reason_open = 0
        bindings = (
            BindingEntry((10,), edge.edge_key, space=edge.owner_space),)
        return ExpansionOutcome(
            state.with_(
                depth=depth,
                frontier=frontier,
                visited=visited,
                bindings=bindings,
                node_count=state.node_count + 1,
                edge_count=state.edge_count + 1,
                read_count=state.read_count + 1,
            ),
            expanded_edge=edge,
            reads=1,
        )


def test_driver_walks_to_target_depth_with_budget():
    seed = seed_query(
        (1,),
        anchors=(QueryAnchor(SPACE_CORE, (1,)),),
        minimum_depth=2,
        budget=QueryBudget(max_depth=5),
    )
    frontier = (FrontierEntry((1,), SPACE_CORE, depth=0, required_slot_gain=10),)
    result = run_query(
        seed.with_(frontier=frontier),
        expander=_LinearExpander(steps=3, minimum_depth=2),
    )
    # depth 达到 3（数据决定的深度），frontier 已空但尚未满足回答条件；
    # 按协议返回 NO_FRONTIER（诚实无结果），不猜测。
    assert result.depth == 3
    assert result.termination == TERMINATION_NO_FRONTIER
    assert len(result.visited) == 3
    assert result.read_count == 3
    # stable_key 可重建且稳定。
    assert result.stable_key() == result.canonical().stable_key()


def test_driver_never_loops_on_duplicate_state():
    class _Repeater(QueryExpander):
        def expand(self, state: QueryState,
                   edge: FrontierEntry) -> ExpansionOutcome:
            # 返回与当前相同的关键状态但加了新边，驱动必须丢弃等价推进。
            return ExpansionOutcome(
                state.with_(frontier=state.frontier, read_count=state.read_count),
                expanded_edge=edge,
                reads=0,
            )

    seed = seed_query((1,), minimum_depth=1)
    result = run_query(
        seed.with_(frontier=(FrontierEntry((1,), SPACE_CORE, depth=0),)),
        expander=_Repeater(),
    )
    assert result.termination == 6
    assert result.cycle_hit == 1
    assert result.read_count == 0


def test_visited_tail_preserves_conflict_and_missing_binding_reasons():
    """已访问尾边不是计算失败，冲突/缺槽仍按原证据状态闭合，trace 不丢边。"""
    class _NoExpansion(QueryExpander):
        def expand(self, state, edge):
            raise AssertionError("已访问尾边不能重复读取")

    tail = FrontierEntry((1,), SPACE_CORE, target_key=(2,), direction=1)
    state = QueryState(
        (29,), roots=(QueryRoot(SPACE_CORE, (2,), ROOT_EXPANDED, 300),),
        frontier=(tail,), visited=(VisitedKey(SPACE_CORE, (2,), direction=1),),
        depth=3, evidence_closed=1, conflict_open=1, read_count=29)
    result = run_query(state, expander=_NoExpansion())
    assert result.termination == TERMINATION_CLARIFY_CONFLICT
    assert result.conflict_open == 1 and result.cycle_hit == 0
    assert result.frontier == state.frontier and result.read_count == state.read_count
    missing = run_query(state.with_(required_slots_open=1), expander=_NoExpansion())
    assert missing.termination == TERMINATION_CLARIFY_MISSING_BINDING


def test_query_state_serialization_is_order_stable():
    a = QueryState(
        query_key=(9,),
        roots=(QueryRoot(SPACE_CORE, (3,), ROOT_PENDING, 300),
               QueryRoot(SPACE_MEMORY, (2,), ROOT_PENDING, 200)),
        anchors=(QueryAnchor(SPACE_CORE, (3,)), QueryAnchor(SPACE_CORE, (1,))),
        visited=(VisitedKey(SPACE_MEMORY, (2,)), VisitedKey(SPACE_CORE, (1,))),
    )
    b = QueryState(
        query_key=(9,),
        roots=(QueryRoot(SPACE_MEMORY, (2,), ROOT_PENDING, 200),
               QueryRoot(SPACE_CORE, (3,), ROOT_PENDING, 300)),
        anchors=(QueryAnchor(SPACE_CORE, (1,)), QueryAnchor(SPACE_CORE, (3,))),
        visited=(VisitedKey(SPACE_CORE, (1,)), VisitedKey(SPACE_MEMORY, (2,))),
    )
    assert a.stable_key() == b.stable_key()
