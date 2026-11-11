"""新开发图上的稀疏读取、完整证据等价和确定性资源上限。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateVerification, EvidenceCandidateEngine, EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.integer_operation_execution import (
    CLARIFY, READY, UNKNOWN, execute_integer_operation, requested_operation_context,
)
from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationPathQuery, OperationProgramBudget,
    OperationProjection, ORDER_DIRECTED_CHAIN,
)
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import (
    TERMINATION_CLARIFY_CONFLICT, QueryState,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import SemanticGraph
from pure_integer_ai.cognition.shared.semantic_object import entity_identity
from pure_integer_ai.experiments.batch_phase_host import current_process_resource_endpoint
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.operation_core_context import (
    read_operation_core_context, request_context_proposition_keys,
)
from pure_integer_ai.experiments.operation_program_query import consume_operation_program
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence
from pure_integer_ai.storage.assertion_identity import IntegerIdentityRegistry
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.telemetry import collect_backend_telemetry

from test_integer_operation_program import _atom, _fact, _schema, _source, _witness, LEFT, RIGHT
from test_s00_semantic_objects import _predicates


@contextmanager
def _core(facts, backend):
    """物化真实 S-00 图及共同 H-05 ledger；轻量 owner 只负责公开索引适配。"""
    context = make_train_context(backend)
    graph = SemanticGraph(context.graph_ontology, _predicates(context.graph_ontology))
    aggregate = _source(9900)
    engine = EvidenceCandidateEngine(EvidenceCandidateProtocol((92806, 1), (92806, 2),
        aggregate, document_scope(aggregate), 1))
    hypotheses = {}
    for ordinal, fact in enumerate(facts, 1):
        graph.define_atomic(fact.definition, scope=document_scope(fact.definition.source),
            provenance_kind=SOURCE_BARE_TEXT, epistemic_origin=EPI_STRUCTURED,
            content_version=1, qualifiers=())
        hypothesis = engine.register(fact.candidate)
        hypotheses[fact.definition.proposition] = hypothesis
        observation = _source(20000 + ordinal)
        prediction = engine.predict(hypothesis, observation=observation,
            scope=document_scope(observation), event_key=(92807, ordinal),
            visible_inputs=tuple(item.filler for item in fact.definition.bindings),
            predicted=fact.definition.proposition)
        stance = (2 if fact.snapshot.refute_evidence_ids else
                  1 if fact.snapshot.support_evidence_ids else 3)
        engine.reveal(prediction, CandidateVerification(stance, (92808, 1), observation,
            fact.definition.predicate, (1,), (ordinal,)), timestamp_seq=2)
        engine.resolve(hypothesis, timestamp_seq=3)
    core = SimpleNamespace(owner=SimpleNamespace(semantic_graph=graph,
        learning=SimpleNamespace(engine=engine, hypothesis_for_candidate=hypotheses.__getitem__)),
        context=SimpleNamespace(graph_ontology=context.graph_ontology,
            scoped_identity_store=SimpleNamespace(registry=IntegerIdentityRegistry(backend))))
    try:
        yield core
    finally:
        backend.close()


def _proof_pair(core, witness, variables, *, budget=OperationProgramBudget()):
    """全量对照和稀疏执行必须产生相同完整 proof，独立 verifier 消费两份上下文。"""
    full = read_operation_core_context(core, budget)
    sparse = read_operation_core_context(core, budget, witness=witness, variables=variables)
    assert requested_operation_context(witness, variables, full, budget) == requested_operation_context(
        witness, variables, sparse, budget)
    before = execute_integer_operation(witness, (92812, 99), variables, full, budget=budget)
    after = execute_integer_operation(witness, (92812, 99), variables, sparse, budget=budget)
    assert before.stable_key() == after.stable_key()
    assert verify_integer_operation(after, full, (witness,), budget=budget)
    assert verify_integer_operation(before, sparse, (witness,), budget=budget)
    return sparse, after


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_real_predicate_index_preserves_all_sources_and_current_evidence(kind, tmp_path, monkeypatch):
    first, other = _schema(1), _schema(2)
    a, b = (entity_identity(_source(100), (index,)) for index in (1, 2))
    facts = tuple(_fact(index, first.relation, a, b, stance=index) for index in (1, 2, 3))
    noise = _fact(4, other.relation, a, b)
    backend = DictBackend() if kind == "dict" else SQLiteBackend(str(tmp_path / "core.sqlite3"))
    with _core((*facts, noise), backend) as core:
        graph = core.owner.semantic_graph
        assert tuple(row.definition for row in graph.lookup_atomic_by_predicate(first.relation)) == tuple(
            fact.definition for fact in facts)
        assert graph.lookup_atomic_by_predicate(_schema(999).relation) == ()
        with pytest.raises(ValueError):
            graph.lookup_atomic_by_predicate(LEFT)
        program = IntegerOperationProgram((92811, 98), (1,),
            (_atom(other, (1, 3)), _atom(first, (1, 4))), (),
            (OperationProjection(1, (LEFT, RIGHT)),))
        witness = _witness(program, (*facts, noise))
        sparse, result = _proof_pair(core, witness, ((1, a),))
        assert len(sparse) == 4 and result.status == CLARIFY
        assert {len(fact.snapshot.refute_evidence_ids) for fact in sparse} == {0, 1}
        monkeypatch.setattr(core.owner.learning.engine, "definitions",
                            lambda: pytest.fail("sparse reader scanned all definitions"))
        assert read_operation_core_context(core, witness=witness, variables=((1, a),)) == sparse
        exact = read_operation_core_context(core, proposition_keys=(noise.proposition_key,))
        assert tuple(row.proposition_key for row in exact) == (noise.proposition_key,)


@pytest.mark.parametrize('refuted', (False, True))
def test_single_anchor_sparse_join_matches_full_evidence_and_avoids_unbound_predicate_scan(refuted, monkeypatch):
    """真实Core索引传播隐含变量，既不读取无关反证，也不裁掉相关反证。"""
    first, second, third = (_schema(index) for index in (1, 2, 3))
    a, b, c, d, x, y = (entity_identity(_source(33000), (index,)) for index in range(1, 7))
    facts = (_fact(33001, first.relation, a, b), _fact(33002, second.relation, a, c),
             _fact(33003, third.relation, a, d), _fact(33004, second.relation, x, y),
             _fact(33005, second.relation, x, y, stance=2))
    if refuted:
        facts = (*facts, _fact(33006, second.relation, a, c, stance=2))
    program = IntegerOperationProgram((92811, 33000), (2,),
        (_atom(first, (1, 2)), _atom(second, (1, 3)), _atom(third, (1, 4))), (),
        tuple(OperationProjection(index, (LEFT, RIGHT)) for index in (1, 2, 3)))
    witness = _witness(program, facts)
    with _core(facts, DictBackend()) as core:
        monkeypatch.setattr(core.owner.semantic_graph, 'lookup_atomic_by_predicate',
                            lambda *args: pytest.fail('single-anchor context read an unbound predicate'))
        selected, execution = _proof_pair(core, witness, ((2, b),))
        assert len(selected) == (4 if refuted else 3)
        assert execution.status == (CLARIFY if refuted else READY)


@pytest.mark.parametrize("stance,expected", [(1, READY), (2, UNKNOWN), (3, UNKNOWN)])
def test_bound_roles_cross_source_refute_unknown_and_unauthorized_are_preserved(stance, expected):
    schema = _schema(1)
    a, b, c = (entity_identity(_source(100), (index,)) for index in (1, 2, 3))
    facts = (_fact(1, schema.relation, a, b), _fact(2, schema.relation, b, c, stance=stance),
             _fact(3, schema.relation, c, a))
    program = IntegerOperationProgram((92811, 99), (1, 2),
        (_atom(schema, (1, 2)), _atom(schema, (2, 3))), (),
        (OperationProjection(1, (LEFT, RIGHT)),))
    witness = _witness(program, facts)
    with _core(facts, DictBackend()) as core:
        sparse, execution = _proof_pair(core, witness, ((1, a), (2, b)))
        assert len(sparse) == 2 and execution.status == expected
        unlicensed = replace(witness, authorized_forming_sources=facts[0].candidate.forming_sources)
        assert _proof_pair(core, unlicensed, ((1, a), (2, b)))[1].status == UNKNOWN
    contradicted = (*facts, _fact(4, schema.relation, a, b, stance=2))
    with _core(contradicted, DictBackend()) as core:
        sparse, execution = _proof_pair(core, witness, ((1, a), (2, b)))
        assert len(sparse) == 3 and execution.status == CLARIFY


@pytest.mark.parametrize("change", ["unique", "ambiguous", "refute", "missing", "cycle"])
def test_path_keeps_full_current_proof_and_fails_without_partial_results(change):
    # 参数仅选择图变化；不向 reader 提供答案或输出 proof。
    first, second = _schema(1), _schema(2)
    a, b, c, d, dead = (entity_identity(_source(100), (index,)) for index in range(1, 6))
    facts = (_fact(1, first.relation, a, b), _fact(2, second.relation, b, c),
             _fact(3, first.relation, a, dead), _fact(4, second.relation, dead, a))
    outcomes = {"unique": READY, "ambiguous": CLARIFY, "refute": CLARIFY,
                "missing": UNKNOWN, "cycle": READY}
    if change == "ambiguous":
        facts += (_fact(5, first.relation, a, d), _fact(6, second.relation, d, c))
    elif change == "refute":
        facts += (_fact(5, second.relation, b, c, stance=2),)
    elif change == "missing":
        facts = facts[1:]
    elif change == "cycle":
        facts += (_fact(7, second.relation, b, a),)
    program = IntegerOperationProgram((92811, 100), (1, 2),
        (_atom(first, (1, 2)), _atom(second, (1, 2))), (),
        (OperationProjection(1, (LEFT, RIGHT)), OperationProjection(2, (LEFT, RIGHT))),
        2, 4, ORDER_DIRECTED_CHAIN, (),
        path_query=OperationPathQuery(1, 2, ((1, LEFT, RIGHT), (2, LEFT, RIGHT))))
    witness = _witness(program, facts)
    with _core(facts, DictBackend()) as core:
        sparse, execution = _proof_pair(core, witness, ((1, a), (2, c)))
        assert execution.status == outcomes[change]
        for budget in (OperationProgramBudget(max_facts=1),
                       OperationProgramBudget(max_match_attempts=1)):
            with pytest.raises(ValueError, match="budget"):
                read_operation_core_context(core, budget, witness=witness, variables=((1, a), (2, c)))


def test_input_locator_and_consumer_use_sparse_current_evidence(monkeypatch):
    schema = _schema(1)
    a, b = (entity_identity(_source(100), (index,)) for index in (1, 2))
    fact = _fact(1, schema.relation, a, b)
    program = IntegerOperationProgram((92811, 101), (1, 2), (_atom(schema, (1, 2)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),))
    witness = _witness(program, (fact,))
    members = tuple(SimpleNamespace(stable_key=lambda key=(900, index): key,
        proposition_key=fact.proposition_key, projection_kind=1, role_key=role.stable_key(),
        candidate_key=filler.stable_key(), predicate_key=schema.relation.stable_key(),
        source_ref=(700,), span_ref=(800, index))
        for index, (role, filler) in enumerate(((LEFT, a), (RIGHT, b)), 1))
    relation = SimpleNamespace(stable_key=lambda: (600,), predicate_key=schema.relation.stable_key(),
        proposition_key=fact.proposition_key, source_ref=(700,))
    question = SimpleNamespace(operation_key=program.operation_key,
        relation_key=(600,), slot_span_refs=((800, 1), (800, 2)))
    structure = SimpleNamespace(operation_keys=(program.operation_key,), operation_request_bindings=(),
        relation_question_bindings=(question,), relation_candidates=(relation,), semantic_candidates=members)
    with _core((fact,), DictBackend()) as core:
        whole = read_operation_core_context(core)
        entries = tuple(entry for row in whole for entry in core_query_evidence(row.definition.proposition,
            core.owner.learning.engine.ledger.evidence_history(row.snapshot.hypothesis),
            core.context.scoped_identity_store.registry))
        monkeypatch.setattr(core.owner.learning.engine, "definitions",
                            lambda: pytest.fail("production operation scanned all definitions"))
        state = QueryState((92812, 100)).with_(evidence=entries)
        updated, execution = consume_operation_program(core, (witness,), structure, state)
        assert execution.status == READY and updated.read_count == state.read_count + 2
        assert request_context_proposition_keys(witness, structure) == (fact.proposition_key,)
        reading = SimpleNamespace(readings=((1, a.stable_key(), (800, 1), ((900, 1),)),))
        structure.operation_request_bindings = (reading,)
        assert request_context_proposition_keys(witness, structure) == (fact.proposition_key,)


def test_current_operation_conflict_sets_shared_query_state_conflict():
    schema = _schema(1)
    a, b = (entity_identity(_source(100), (index,)) for index in (1, 2))
    facts = (_fact(31, schema.relation, a, b),
             _fact(32, schema.relation, a, b, stance=2))
    program = IntegerOperationProgram((92811, 103), (1, 2), (_atom(schema, (1, 2)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),))
    witness = _witness(program, facts)
    support = facts[0]
    members = tuple(SimpleNamespace(stable_key=lambda key=(901, index): key,
        proposition_key=support.proposition_key, projection_kind=1, role_key=role.stable_key(),
        candidate_key=filler.stable_key(), predicate_key=schema.relation.stable_key(),
        source_ref=(701,), span_ref=(801, index))
        for index, (role, filler) in enumerate(((LEFT, a), (RIGHT, b)), 1))
    relation = SimpleNamespace(stable_key=lambda: (601,), predicate_key=schema.relation.stable_key(),
        proposition_key=support.proposition_key, source_ref=(701,))
    question = SimpleNamespace(operation_key=program.operation_key,
        relation_key=(601,), slot_span_refs=((801, 1), (801, 2)))
    structure = SimpleNamespace(operation_keys=(program.operation_key,), operation_request_bindings=(),
        relation_question_bindings=(question,), relation_candidates=(relation,), semantic_candidates=members)
    with _core(facts, DictBackend()) as core:
        whole = read_operation_core_context(core)
        entries = tuple(entry for row in whole for entry in core_query_evidence(row.definition.proposition,
            core.owner.learning.engine.ledger.evidence_history(row.snapshot.hypothesis),
            core.context.scoped_identity_store.registry))
        state = QueryState((92812, 103)).with_(evidence=entries)
        updated, execution = consume_operation_program(core, (witness,), structure, state)
        assert execution.status == CLARIFY and execution.outputs == ()
        assert updated.termination == TERMINATION_CLARIFY_CONFLICT
        assert updated.conflict_open == 1
        assert updated.evidence_closed == updated.generation_ready == 0


@pytest.mark.parametrize("noise_count", [0, 32, 128])
def test_sqlite_candidate_growth_profile_is_fixture_only(noise_count, tmp_path, monkeypatch):
    schema = _schema(1)
    a, b = (entity_identity(_source(100), (index,)) for index in (1, 2))
    facts = (_fact(1, schema.relation, a, b), *(_fact(100 + index, schema.relation,
        entity_identity(_source(1000 + index), (1,)), entity_identity(_source(1000 + index), (2,)))
        for index in range(noise_count)))
    program = IntegerOperationProgram((92811, 102), (1, 2), (_atom(schema, (1, 2)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),))
    witness = _witness(program, facts[:1])
    variables, measurements, proofs = ((1, a), (2, b)), {}, []
    with _core(facts, SQLiteBackend(str(tmp_path / "core.sqlite3"))) as core:
        for mode in ("full", "sparse"):
            started = current_process_resource_endpoint()
            with collect_backend_telemetry() as telemetry:
                rows = (read_operation_core_context(core) if mode == "full" else
                    read_operation_core_context(core, witness=witness, variables=variables))
            ended = current_process_resource_endpoint()
            proof = execute_integer_operation(witness, (92812, 101), variables, rows)
            assert proof.status == READY
            proofs.append(proof.stable_key())
            measurements[mode] = {"candidate_count": len(rows), "telemetry": telemetry.to_json(),
                "wall_ns": ended.sampled_ended_ns - started.sampled_started_ns,
                "cpu_ns": ended.cpu_ns - started.cpu_ns,
                "resource_started": started.to_value(), "resource_ended": ended.to_value(),
                "proof_sha256": hashlib.sha256(canonical_json_bytes(list(proof.stable_key()))).hexdigest()}
        assert proofs[0] == proofs[1]
        assert measurements["full"]["candidate_count"] == noise_count + 1
        assert measurements["sparse"]["candidate_count"] == 1
        monkeypatch.setattr(core.owner.learning.engine, "definitions",
                            lambda: pytest.fail("sparse reader scanned unrelated candidates"))
        read_operation_core_context(core, OperationProgramBudget(max_facts=1),
                                    witness=witness, variables=variables)
    report_root = os.environ.get("PIA_SPARSE_CONTEXT_REPORT_ROOT")
    if report_root:
        root = Path(report_root).resolve(strict=True)
        if root.drive.upper() != "K:" or root.parent == Path("K:/"):
            raise ValueError("profile requires an explicit named K root")
        report = {"status": "PASS_SQLITE_SPARSE_DEVELOPMENT_FIXTURE_ONLY",
            "noise_count": noise_count, "measurements": measurements,
            "cache_scope": "SAME_OWNER_AFTER_FIXTURE_BUILD_FULL_THEN_SPARSE",
            "sqlite_vfs_physical_reads": "UNKNOWN", "os_hard_page_in": "UNKNOWN",
            "end_to_end_batch_fit": "UNKNOWN", "free_dialogue_complete": 0}
        (root / f"sparse-context-noise-{noise_count}.json").write_bytes(canonical_json_bytes(report) + b"\n")
