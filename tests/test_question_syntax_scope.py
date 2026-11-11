"""问式作用域与独立整数filler投影对齐，保持未闭合关系的回退保护。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    TrainedInputMember,
    TrainedInputStructure,
    TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
    relation_question_syntax_predicates,
)


def _project(values, *, left=(11,), right=(22,)):
    """构造带完整独立端点的整数关系，不嵌入自然语言词形或答案。"""
    proposition, predicate, source = (1, 50), (5, 60), (9, 70)
    left_end = len(left)
    members = (
        TrainedInputMember(left, (4, 101), 4, PROJECTION_FILLER,
                           proposition, predicate, source, 0, left_end, 0, (8, 1)),
        TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
                           proposition, predicate, source, left_end, left_end + 1, 1),
        TrainedInputMember(right, (4, 102), 4, PROJECTION_FILLER,
                           proposition, predicate, source, left_end + 1,
                           left_end + 1 + len(right), 2, (8, 2)),
    )
    fact = TrainedInputStructure(proposition, predicate, source, (7, 1), members)
    return TrainedInputStructureProjector((fact,)).project(values)


def _frame(*, predicate=(5, 61), gaps=((), (44,), (55,))):
    """请求另一谓词，确保作用域校验不偷借Core关系的真值。"""
    return TrainedRelationQuestionFrame(
        (92002, 1), (92004, 1), predicate, ((8, 1), (8, 2)), gaps)


@pytest.mark.parametrize("values,left,right,gaps", (
    ((11, 66, 44, 22, 55), (11,), (22,), ((), (44,), (55,))),
    ((11, 66, 44, 22, 55), (11, 66), (22,), ((), (66, 44), (55,))),
    ((11, 44, 66, 22, 55), (11,), (22,), ((), (44,), (55,))),
    ((11, 44, 22, 66, 55), (11,), (22, 66), ((), (44,), (66, 55))),
))
def test_syntax_cannot_swallow_gaps_or_split_independent_fillers(values, left, right, gaps):
    """前后角色吞入间隔或截断已有filler，均不得产生Core谓词过滤。"""
    structure = _project(values, left=left, right=right)
    assert structure.semantic_candidates
    assert relation_question_syntax_predicates((_frame(gaps=gaps),), structure) == ()


def test_aligned_syntax_still_limits_unclosed_requested_predicate():
    """请求谓词没有闭合Core命题时仍保留精确问式作用域。"""
    structure = _project((11, 66, 44, 22, 55))
    frame = _frame(gaps=((), (66, 44), (55,)))
    assert all(item.predicate_key != frame.predicate_key
               for item in structure.relation_candidates)
    assert relation_question_syntax_predicates((frame,), structure) == ((5, 61),)


@pytest.mark.parametrize("change", ("no_projection", "other_source", "routed_only"))
def test_span_or_derived_route_alone_is_not_independent_filler_evidence(change):
    """保留相同字面但移除独立投影、来源或只留路由，不能冒充作用域证据。"""
    structure = _project((11, 44, 22, 55))
    if change == "no_projection":
        structure = replace(structure, semantic_candidates=())
    elif change == "other_source":
        from pure_integer_ai.cognition.understanding.query_open_roles import match_open_roles
        from pure_integer_ai.cognition.understanding.query_question_syntax_scope import (
            exact_question_filler_intervals,
            question_parse_aligns_with_fillers,
        )
        parsed, = match_open_roles((_frame().open_frame(),),
                                   structure.token_values, (9, 999))
        assert not question_parse_aligns_with_fillers(
            parsed, exact_question_filler_intervals(structure))
        return
    else:
        from pure_integer_ai.cognition.understanding.query_concept_links import InputConceptRoute
        routes = tuple(InputConceptRoute(
            item.span_ref, (4, 999), item.candidate_key,
            (1, 999), item.proposition_key, (7, 999), item.stable_key())
            for item in structure.semantic_candidates)
        structure = replace(
            structure,
            semantic_candidates=tuple(sorted((
                replace(item, concept_route_key=route.stable_key())
                for item, route in zip(structure.semantic_candidates, routes, strict=True)),
                key=lambda item: item.stable_key())),
            concept_routes=tuple(sorted(routes, key=lambda item: item.stable_key())))
    assert relation_question_syntax_predicates((_frame(),), structure) == ()


@pytest.mark.parametrize("mode", ("slot", "composed"))
def test_slot_and_composed_question_scopes_share_exact_filler_rule(mode):
    """WH及来源组合问式都经过同一作用域门，不退回纯间隔匹配。"""
    if mode == "slot":
        structure = _project((11, 44, 88))
        frame = TrainedRelationQuestionFrame(
            (92002, 3), (92004, 1), (5, 61), ((8, 1),),
            ((), (44, 88)), (8, 2), (88,))
        assert relation_question_syntax_predicates((frame,), structure) == ((5, 61),)
        wrong = replace(frame, gaps=((), (88,)))
        assert relation_question_syntax_predicates((wrong,), structure) == ()
    else:
        structure = _project((66, 11, 44, 22, 55))
        first = _frame(gaps=((66,), (44,), (77,)))
        second = replace(_frame(), source_case_key=(92002, 2))
        assert relation_question_syntax_predicates((first, second), structure) == ((5, 61),)


def test_aligned_and_misaligned_predicates_do_not_create_false_competition():
    """同一输入只保留精确角色问式，错误间隔解析不构成多谓词deny-all。"""
    structure = _project((11, 66, 44, 22, 55))
    good = _frame(gaps=((), (66, 44), (55,)))
    bad = replace(_frame(), source_case_key=(92002, 2), predicate_key=(5, 62))
    assert relation_question_syntax_predicates((good, bad), structure) == ((5, 61),)


def test_aligned_distinct_predicates_remain_ambiguous():
    """作用域对齐并不替消费者挑选谓词，真实多谓词竞争完整保留。"""
    structure = _project((11, 44, 22, 55))
    first = _frame()
    second = replace(first, source_case_key=(92002, 2), predicate_key=(5, 62))
    assert relation_question_syntax_predicates((first, second), structure) == ((5, 61), (5, 62))
