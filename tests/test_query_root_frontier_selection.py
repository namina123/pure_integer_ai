"""共同前沿根参与调度的饥饿、来源保留和真实驱动边界。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.query_driver import ExpansionOutcome, run_query
from pure_integer_ai.cognition.shared.query_frontier_selection import select_frontier_entry
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY, ROOT_EXPANDED, ROOT_PENDING,
    SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY,
    TERMINATION_ANSWER_CLOSED, TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_CLARIFY_CONFLICT, TERMINATION_CLARIFY_MISSING_BINDING,
    FrontierEntry, QueryBudget, QueryRoot, QueryState, VisitedKey, sort_frontier,
)


def _root(space=SPACE_MEMORY, key=(10,), status=ROOT_PENDING):
    """构造具有完整来源与scope的测试根。"""
    return QueryRoot(space,key,status,200,source_ref=(71,),scope_key=(81,))


def _entry(target, *, root=(10,), space=SPACE_MEMORY, gain=1, source=(71,), direction=0):
    """原边的全部身份、来源、方向和scope进入调度视图。"""
    return FrontierEntry((space,*target),space,target_key=target,root_key=root,
        source_ref=source,scope_key=(81,),direction=direction,required_slot_gain=gain)


def test_pending_root_seed_precedes_higher_priority_descendants_without_mutation():
    """保留所有高优先级后代，但先选择实际指向根的原种子。"""
    child,seed = _entry((20,),gain=9),_entry((10,))
    frontier = sort_frontier((seed,child))
    roots = (_root(),)
    assert frontier == (child,seed)
    selected = select_frontier_entry(frontier,roots)
    assert selected == seed
    assert selected.stable_key() == seed.stable_key()
    assert frontier == (child,seed) and roots == (_root(),)


def test_root_seed_order_uses_existing_full_priority_and_stable_ties():
    """待参与根之间仍消费原排序，不以space名称或根枚举顺序改次序。"""
    roots = (_root(key=(10,)),_root(SPACE_DIALOGUE,(30,)))
    a,b = _entry((10,)),_entry((30,),root=(30,),space=SPACE_DIALOGUE,gain=2)
    frontier = sort_frontier((a,b))
    assert select_frontier_entry(frontier,roots) == b
    assert select_frontier_entry(frontier,tuple(reversed(roots))) == b


@pytest.mark.parametrize('status',(ROOT_EXPANDED,ROOT_EMPTY))
def test_participated_or_empty_roots_restore_original_priority(status):
    """完成或空根不再享有根参与优先级。"""
    child,seed = _entry((20,),gain=9),_entry((10,))
    assert select_frontier_entry((child,seed),(_root(status=status),)) == child


def test_foreign_space_descendant_or_wrong_root_cannot_impersonate_seed():
    """目标、所属space和声明root必须同时匹配；缺种子仍走原首边。"""
    child = _entry((20,),gain=9)
    foreign = _entry((10,),space=SPACE_CORE)
    wrong = _entry((10,),root=(99,))
    frontier = sort_frontier((wrong,foreign,child))
    assert select_frontier_entry(frontier,(_root(),)) == child


def test_seed_without_explicit_target_uses_original_edge_key():
    """沿用driver既有目标缺省合同，不合成新边。"""
    child = _entry((20,),gain=9)
    seed = FrontierEntry((10,),SPACE_MEMORY,root_key=(10,))
    assert select_frontier_entry((child,seed),(_root(),)) == seed
    with pytest.raises(ValueError):
        select_frontier_entry((),(_root(),))


def test_driver_consumes_every_root_within_original_budget_and_keeps_children():
    """高gain链不能饿死低gain根；实际读根后才允许原终止器回答。"""
    roots = (_root(SPACE_CORE,(1,)),_root(SPACE_MEMORY,(2,)),_root(SPACE_DIALOGUE,(3,)))
    seeds = tuple(_entry(r.root_key,root=r.root_key,space=r.owner_space) for r in roots)
    child = _entry((20,),root=(1,),space=SPACE_CORE,gain=9)
    seen = []
    class Expander:
        """只模拟有限图的真实读取及根状态，不修改终止门或预算。"""
        def expand(self,state,edge):
            """逐条追加访问证据，未消费后代始终留在同一frontier。"""
            seen.append((edge.owner_space,edge.target_key))
            updated = tuple(replace(root,status=ROOT_EXPANDED)
                if (root.owner_space,root.root_key) == (edge.owner_space,edge.target_key)
                else root for root in state.roots)
            return ExpansionOutcome(state.with_(
                roots=updated,frontier=tuple(item for item in state.frontier if item != edge),
                visited=(*state.visited,VisitedKey(edge.owner_space,edge.target_key)),
                node_count=state.node_count+1,edge_count=state.edge_count+1,
                read_count=state.read_count+1,depth=1,evidence_closed=1,
                generation_ready=1,best_score=100),edge,1)
    state = QueryState((91,),roots=roots,frontier=(*seeds,child),
        budget=QueryBudget(max_nodes=3,max_edges=3,max_reads=3,max_depth=2))
    result = run_query(state,expander=Expander())
    assert result.termination == TERMINATION_ANSWER_CLOSED
    assert set(seen) == {(r.owner_space,r.root_key) for r in roots}
    assert result.node_count == result.edge_count == result.read_count == 3
    assert result.frontier == (child,)
    assert result.budget == state.budget


@pytest.mark.parametrize(
    'changes,expected',
    (({'conflict_open':1},TERMINATION_CLARIFY_CONFLICT),
     ({'required_slots_open':1},TERMINATION_CLARIFY_MISSING_BINDING),
     ({'node_count':2},TERMINATION_BUDGET_EXHAUSTED)),
)
def test_root_selection_does_not_override_conflict_missing_or_budget(changes,expected):
    """调度不能授权失败状态；原综合终止器继续先于读取执行。"""
    class NoRead:
        """终止后不应发生任何扩展。"""
        def expand(self,state,edge):
            """捕获终止后的非法读取。"""
            raise AssertionError('unexpected graph read')
    state = QueryState((92,),roots=(_root(status=ROOT_EXPANDED),),
        depth=1,evidence_closed=1,generation_ready=1,best_score=100,
        budget=QueryBudget(max_nodes=1,max_edges=1,max_reads=1,max_depth=2))
    assert run_query(state.with_(**changes),expander=NoRead()).termination == expected
