"""未知请求变量不能让已闭合的子关系越过完整输入合同。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.identity import OwnerScope, SourceRef, VersionBundle
from pure_integer_ai.cognition.shared.query_state import ROOT_EMPTY, EvidenceEntry
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleBudgetExceeded
from pure_integer_ai.cognition.understanding.query_operation_request_scope import (
    PENDING_REQUEST_SCOPE, PendingOperationRequestScope, match_pending_operation_request_scopes,
    pending_operation_request_input_view,
)
from pure_integer_ai.cognition.understanding.query_operation_requests import (
    FRAME, OperationRequestFrame, attach_operation_request, match_operation_request_frames,
)
from pure_integer_ai.experiments.operation_request_scope_query import (
    pending_request_context_allowed, pending_request_query_records,
)
from test_query_input_provenance import PREDICATE, ROLES, _project, _structure
from test_trained_input_clarification import _query


def _fixture(last=99):
    """用实际整数结构投影产生闭合子关系，第三个 reader 有完整语法但可能未知。"""
    projected = _project(_structure(101), values=(9, 41, 43, 42, 7, last, 8))
    kinds = tuple(next(item.object_kind for item in projected.semantic_candidates
                       if item.role_key == role.stable_key()) for role in ROLES)
    digest = tuple(range(32))
    frame = OperationRequestFrame((902, 1), SourceRef(FRAME, int.from_bytes(bytes(digest[:8]), 'big'), 0,
                                                    OwnerScope(), VersionBundle()),
        digest, (903, 1), (904, 1),
        ((1, PREDICATE.stable_key(), ROLES[0].stable_key(), (kinds[0],)),
         (2, PREDICATE.stable_key(), ROLES[1].stable_key(), (kinds[1],)),
         (3, PREDICATE.stable_key(), ROLES[1].stable_key(), (kinds[1],))),
        ((9,), (43,), (7,), (8,)))
    return frame, projected


def _closed_state(frame, projected):
    """只完成原 QueryState 的 frontier，保留缺绑定和所有 UNKNOWN 原记录。"""
    state, _frontier, _hypotheses, _nodes = _query()
    records = pending_request_query_records((frame,), projected)
    return state.with_(query_key=projected.source_ref,
        roots=tuple(replace(root, status=ROOT_EMPTY) for root in state.roots), frontier=(),
        bindings=tuple(binding for binding, _ in records),
        evidence=(*state.evidence, *(evidence for _, evidence in records))), records


def test_missing_variable_retains_full_scope_despite_a_closed_known_subrelation():
    frame, projected = _fixture()
    before = projected.stable_key()
    assert any(item.required_count == item.support_count for item in projected.relation_candidates)
    assert match_operation_request_frames((frame,), projected) == ()
    scope, = match_pending_operation_request_scopes((frame,), projected)
    assert tuple(row[:3] for row in scope.readings) == ((1, 1, 2), (2, 3, 4), (3, 5, 6))
    assert scope.readings[0][3] and scope.readings[1][3] and not scope.readings[2][3]
    assert PendingOperationRequestScope.from_stable_key(scope.stable_key()) == scope
    assert scope.input_values == projected.token_values and scope.source_ref == projected.source_ref
    assert projected.stable_key() == before
    records = pending_request_query_records((frame,), projected)
    assert len(records) == 1 and records[0][1].polarity == 3
    assert records[0][1].payload_key == scope.stable_key()
    assert records[0][1].source_ref == frame.source.stable_key()


def test_unique_complete_request_keeps_original_binding_and_has_no_pending_scope():
    frame, projected = _fixture(42)
    assert len(match_operation_request_frames((frame,), projected)) == 1
    assert match_pending_operation_request_scopes((frame,), projected) == ()
    bound = attach_operation_request((frame,), projected)
    assert match_pending_operation_request_scopes((frame,), bound) == ()


def test_same_surface_ambiguity_retains_every_physical_projection():
    frame, projected = _fixture(42)
    member = next(item for item in projected.semantic_candidates
                  if projected.spans and item.role_key == ROLES[1].stable_key())
    changed = replace(member, candidate_key=(*member.candidate_key[:-1], member.candidate_key[-1] + 1))
    projected = replace(projected, semantic_candidates=tuple(sorted(
        (*projected.semantic_candidates, changed), key=lambda item: item.stable_key())))
    assert match_operation_request_frames((frame,), projected) == ()
    scope, = match_pending_operation_request_scopes((frame,), projected)
    assert member.stable_key() in {key for row in scope.readings for key in row[3]}
    assert changed.stable_key() in {key for row in scope.readings for key in row[3]}


@pytest.mark.parametrize('values', [(6, 9, 41, 43, 42, 7, 99, 8), (9, 41, 43, 42, 7, 99, 8, 6),
                                   (9, 41, 43, 42, 7, 8)])
def test_partial_input_and_unlicensed_outer_gaps_do_not_acquire_request_scope(values):
    frame, _ = _fixture()
    assert match_pending_operation_request_scopes((frame,), _project(_structure(101), values=values)) == ()


def test_pending_parse_respects_original_explicit_enumeration_budget():
    frame, projected = _fixture()
    with pytest.raises(OpenRoleBudgetExceeded):
        match_pending_operation_request_scopes((frame,), projected, max_steps=1)


def test_pending_context_requires_all_original_current_records_without_truth_or_conflict():
    frame, projected = _fixture()
    state, records = _closed_state(frame, projected)
    assert pending_request_context_allowed(state, projected, (frame,))
    assert not pending_request_context_allowed(state.with_(evidence=state.evidence[:-1]), projected, (frame,))
    assert not pending_request_context_allowed(state.with_(query_key=(999,)), projected, (frame,))
    assert not pending_request_context_allowed(state.with_(conflict_open=1), projected, (frame,))
    truth = EvidenceEntry((60,), (61,), space=1, polarity=1)
    assert not pending_request_context_allowed(state.with_(evidence=(*state.evidence, truth)), projected, (frame,))
    assert not pending_request_context_allowed(state.with_(read_count=state.budget.max_reads + 1), projected, (frame,))
    altered = replace(records[0][1], source_ref=(123,))
    assert not pending_request_context_allowed(state.with_(evidence=(*state.evidence[:-1], altered)), projected, (frame,))


def test_scope_decoder_rejects_incomplete_and_noncanonical_records():
    frame, projected = _fixture()
    scope, = match_pending_operation_request_scopes((frame,), projected)
    with pytest.raises(ValueError):
        PendingOperationRequestScope.from_stable_key(scope.stable_key()[:-1])
    with pytest.raises(ValueError):
        replace(scope, readings=(*scope.readings[:-1], (3, 4, 6, ())))
    assert scope.stable_key()[0] == PENDING_REQUEST_SCOPE


def test_production_pending_view_preserves_complete_unknown_reader_proof_and_is_idempotent():
    """真实bridge入口先区分请求gap与事实，原待定reader来源和输入逐项不变。"""
    from types import SimpleNamespace
    from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge

    frame, projected = _fixture()
    scopes = match_pending_operation_request_scopes((frame,), projected)
    assert projected.relation_candidates
    owner = SimpleNamespace(operation_request_frames=(frame,))
    view = TrainedGraphQueryBridge._attach_operation_request_binding(owner, projected)
    assert view.tokens == projected.tokens and view.spans == projected.spans
    assert view.carrier_candidates == projected.carrier_candidates
    assert not view.operation_request_bindings and not view.operation_keys
    assert not view.relation_candidates and not view.open_relations
    assert match_pending_operation_request_scopes((frame,), view) == scopes
    expected = {key for scope in scopes for row in scope.readings for key in row[3]}
    assert {item.stable_key() for item in view.semantic_candidates} == expected
    assert TrainedGraphQueryBridge._attach_operation_request_binding(owner, view) == view


def test_pending_reader_view_keeps_all_competing_targets_without_selecting_one():
    """同一reader跨度的两个完整目标都保留，缺唯一绑定不被伪造成已知对象。"""
    frame, projected = _fixture(42)
    member = next(item for item in projected.semantic_candidates if item.role_key == ROLES[1].stable_key())
    competitor = replace(member, candidate_key=(*member.candidate_key[:-1], member.candidate_key[-1] + 1))
    projected = replace(projected, semantic_candidates=tuple(sorted(
        (*projected.semantic_candidates, competitor), key=lambda item: item.stable_key())))
    scopes = match_pending_operation_request_scopes((frame,), projected)
    view = pending_operation_request_input_view((frame,), projected)
    assert match_operation_request_frames((frame,), view) == ()
    assert match_pending_operation_request_scopes((frame,), view) == scopes
    assert {member.stable_key(), competitor.stable_key()} <= {item.stable_key() for item in view.semantic_candidates}


def test_all_unknown_readers_retain_the_full_request_syntax_without_any_factual_projection():
    """所有reader均未知也保留完整输入范围，不猜对象、不依赖已知子关系。"""
    frame, _ = _fixture()
    projected = _project(_structure(101), values=(9, 98, 43, 99, 7, 100, 8))
    view = pending_operation_request_input_view((frame,), projected)
    scope, = match_pending_operation_request_scopes((frame,), view)
    assert all(not row[3] for row in scope.readings)
    assert view.token_values == projected.token_values and not view.semantic_candidates
    assert not view.relation_candidates and not view.open_relations


@pytest.mark.parametrize('values', [(6, 9, 41, 43, 42, 7, 99, 8), (9, 41, 43, 42, 7, 99, 8, 6)])
def test_pending_reader_view_does_not_remove_unlicensed_outer_input(values):
    """完整语法之外的输入仍无请求资格，原关系和投影不被裁去。"""
    frame, _ = _fixture()
    projected = _project(_structure(101), values=values)
    assert pending_operation_request_input_view((frame,), projected) is projected
