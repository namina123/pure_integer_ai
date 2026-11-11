from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import os

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_PROPOSITION,
    ObjectIdentity,
    concept_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED,
    SPACE_CORE,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    TERMINATION_ANSWER_CLOSED,
    BindingEntry,
    QueryRoot,
    QueryState,
)
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    MultiClaimCompositionVariant,
    PART_CLAIM_ROLE,
    PART_LITERAL,
    TrainedMultiClaimComposer,
    materialize_multi_claim_composition_graph,
    read_multi_claim_composition_graph,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    GraphRelationGeneration,
    RelationSurfaceBinding,
)
from pure_integer_ai.cognition.understanding.query_structure_adapter import (
    RelationSurfaceStructureAdapter,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    QueryHop,
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence
from pure_integer_ai.experiments.generation_organization import (
    ClosedGenerationEvidence,
    bind_query_evidence,
    build_multi_response_plan,
)
from pure_integer_ai.experiments.trained_response_delivery import (
    FactualResponseGeneration,
    read_delivery_proof,
    record_factual_response_delivery,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import _parts
from pure_integer_ai.experiments.trained_response_delivery import DELIVERY_PROOF_IDENTITY


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RELEASE = Path(
    os.environ.get("PIA_TEST_CANONICAL_MODEL", "__optional_canonical_model_unset__"))
CANONICAL_COMPOSITION_GRAPH = Path(
    os.environ.get("PIA_TEST_COMPOSITION_GRAPH", "__optional_composition_graph_unset__"))
SURFACE_VARIANT_GRAPH = Path(
    os.environ.get("PIA_TEST_SURFACE_VARIANT_GRAPH", "__optional_surface_variant_graph_unset__"))


def _relation_courses() -> tuple[Path, ...]:
    """返回本专项可复现的公开 authored relation 训练输入。"""
    data_root = ROOT / "data" / "ph2"
    return tuple(sorted(data_root.glob("authored_relation_*_w06_seed_v2.jsonl.sample"))) + tuple(
        sorted(
            path for path in data_root.glob("authored_relation_*_seed_v1.jsonl.sample")
            if path.name != "authored_relation_alias_refers_seed_v1.jsonl.sample"
        )
    )


def _trained_database(tmp_path: Path) -> Path:
    """建立只供测试使用的 active Core 图，不读取 K 盘 canonical。"""
    from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
        build_authored_w06_learning_runtime,
    )
    from pure_integer_ai.experiments.language_protocol_runtime import (
        install_language_graph_protocols,
    )
    from pure_integer_ai.experiments.run_conversation_training import (
        dialogue_semantic_protocols,
    )
    from pure_integer_ai.experiments.train_context import make_train_context
    from pure_integer_ai.storage.backend import SQLiteBackend

    database = tmp_path / "core.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(
            context, occurrence_protocol=occurrence, span_protocol=span)
        build_authored_w06_learning_runtime(
            backend, context, _relation_courses(), tmp_path / "typed-pack")
        backend.commit()
    finally:
        backend.close()
    return database


def _variant():
    return MultiClaimCompositionVariant(
        1,
        2,
        (
            (PART_CLAIM_ROLE, 1, 1),
            (PART_LITERAL, (65292,)),
            (PART_CLAIM_ROLE, 1, 2),
            (PART_LITERAL, (65292,)),
            (PART_CLAIM_ROLE, 2, 1),
            (PART_LITERAL, (12290,)),
            (PART_CLAIM_ROLE, 2, 2),
        ),
        (91641, 1),
        (91642, 1),
    )


def test_composition_graph_cold_restore_and_bind(tmp_path):
    graph = tmp_path / "composition.sqlite3"
    receipt = materialize_multi_claim_composition_graph((_variant(),), graph)
    restored, metadata = read_multi_claim_composition_graph(graph)
    assert receipt["format"] == metadata["format"]
    assert restored[0].stable_key() == _variant().stable_key()
    assert metadata["source_identity"] == (91641, 1)
    assert metadata["license_identity"] == (91642, 1)

    class Delegate:
        def generate_relation(self, source, fact):
            raise AssertionError("composition test must not use relation fallback")

    composer = TrainedMultiClaimComposer(graph, Delegate())
    role_a = structure_concept_identity((91643, 1))
    role_b = structure_concept_identity((91643, 2))
    facts = []
    for ordinal, surfaces in enumerate((("甲", "一"), ("乙", "二")), 1):
        facts.append(SimpleNamespace(bindings=(
            RelationSurfaceBinding(role_a, concept_identity((91644, ordinal, 1)),
                                   surfaces[0], 100 + ordinal, 0, 1),
            RelationSurfaceBinding(role_b, concept_identity((91644, ordinal, 2)),
                                   surfaces[1], 100 + ordinal, 1, 2),
        )))
    result = composer.generate_composition(
        tuple((None, fact, None) for fact in facts),
        SimpleNamespace(evidence_closed=1, conflict_open=0, required_slots_open=0),
    )
    assert tuple(map(ord, result.surface)) == (
        ord("甲"), 65292, ord("一"), 65292,
        ord("乙"), 12290, ord("二"),
    )
    assert result.connector is not None


def test_claim_slot_contract_requires_one_complete_coordinate_map(tmp_path):
    graph = tmp_path / "composition-contract.sqlite3"
    materialize_multi_claim_composition_graph((_variant(),), graph)
    composer = TrainedMultiClaimComposer(graph, object())
    roles = (structure_concept_identity((91643, 1)),
             structure_concept_identity((91643, 2)))
    facts = tuple(
        SimpleNamespace(
            proposition=ObjectIdentity(OBJECT_PROPOSITION, (91644, ordinal)),
            bindings=(
                RelationSurfaceBinding(
                    roles[0], concept_identity((91645, 1)),
                    left, 100 + ordinal, 0, 1),
                RelationSurfaceBinding(
                    roles[1], concept_identity((91645, ordinal, 2)),
                    right, 100 + ordinal, 1, 2),
            ),
        )
        for ordinal, (left, right) in enumerate((("甲", "一"), ("甲", "二")), 1)
    )
    components = tuple((None, fact, None) for fact in facts)
    contract = composer.claim_slot_contract(components)
    assert contract == tuple(
        (fact.proposition.stable_key(),
         tuple(binding.role.stable_key() for binding in fact.bindings))
        for fact in facts)
    evidences = tuple(
        ClosedGenerationEvidence(
            fact.bindings,
            GraphRelationGeneration(
                "甲，一，甲。一", structure_concept_identity((91646, ordinal)),
                700 + ordinal, 4),
            fact.proposition.stable_key(), (91647, 1), 800 + ordinal)
        for ordinal, fact in enumerate(facts, 1)
    )
    composed = composer.generate_composition(
        components,
        SimpleNamespace(evidence_closed=1, conflict_open=0,
                        required_slots_open=0),
    )
    assert build_multi_response_plan(evidences, composed) is None
    assert build_multi_response_plan(
        evidences, composed, claim_slot_contract=contract) is not None

    incomplete = MultiClaimCompositionVariant(
        1,
        2,
        (
            (PART_CLAIM_ROLE, 1, 1),
            (PART_LITERAL, (65292,)),
            (PART_CLAIM_ROLE, 2, 1),
        ),
        (91641, 1),
        (91642, 1),
    )
    incomplete_graph = tmp_path / "composition-contract-incomplete.sqlite3"
    materialize_multi_claim_composition_graph((incomplete,), incomplete_graph)
    incomplete_composer = TrainedMultiClaimComposer(incomplete_graph, object())
    assert incomplete_composer.claim_slot_contract(components) is None

    duplicate = MultiClaimCompositionVariant(
        1,
        2,
        (
            (PART_CLAIM_ROLE, 1, 1),
            (PART_CLAIM_ROLE, 1, 1),
            (PART_CLAIM_ROLE, 2, 1),
            (PART_CLAIM_ROLE, 2, 2),
        ),
        (91641, 1),
        (91642, 1),
    )
    duplicate_graph = tmp_path / "composition-contract-duplicate.sqlite3"
    materialize_multi_claim_composition_graph((duplicate,), duplicate_graph)
    assert TrainedMultiClaimComposer(
        duplicate_graph, object()).claim_slot_contract(components) is None

    ambiguous_parts = list(_variant().parts)
    ambiguous_parts[1] = (PART_LITERAL, (124,))
    ambiguous = MultiClaimCompositionVariant(
        2,
        2,
        tuple(ambiguous_parts),
        (91641, 1),
        (91642, 1),
    )
    ambiguous_graph = tmp_path / "composition-contract-ambiguous.sqlite3"
    materialize_multi_claim_composition_graph((_variant(), ambiguous), ambiguous_graph)
    assert TrainedMultiClaimComposer(
        ambiguous_graph, object()).claim_slot_contract(components) is None


def test_composition_graph_rejects_open_state(tmp_path):
    graph = tmp_path / "composition.sqlite3"
    materialize_multi_claim_composition_graph((_variant(),), graph)
    composer = TrainedMultiClaimComposer(graph, object())
    try:
        composer.generate_composition(
            ((None, SimpleNamespace(bindings=()), None),) * 2,
            SimpleNamespace(evidence_closed=0, conflict_open=0,
                            required_slots_open=1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("open QueryState must remain fail-closed")


def test_composition_graph_requires_explicit_variant_when_claim_count_collides(
        tmp_path):
    graph = tmp_path / "composition-variants.sqlite3"
    first = _variant()
    second = MultiClaimCompositionVariant(
        2,
        2,
        (
            (PART_CLAIM_ROLE, 1, 2),
            (PART_LITERAL, (124,)),
            (PART_CLAIM_ROLE, 1, 1),
            (PART_LITERAL, (124,)),
            (PART_CLAIM_ROLE, 2, 2),
            (PART_LITERAL, (33,)),
            (PART_CLAIM_ROLE, 2, 1),
        ),
        (91641, 1),
        (91642, 1),
    )
    materialize_multi_claim_composition_graph((first, second), graph)
    composer = TrainedMultiClaimComposer(graph, object())
    role_a = structure_concept_identity((91643, 1))
    role_b = structure_concept_identity((91643, 2))
    facts = tuple(
        SimpleNamespace(bindings=(
            RelationSurfaceBinding(role_a, concept_identity((91644, ordinal, 1)),
                                   left, 100 + ordinal, 0, 1),
            RelationSurfaceBinding(role_b, concept_identity((91644, ordinal, 2)),
                                   right, 100 + ordinal, 1, 2),
        ))
        for ordinal, (left, right) in enumerate((("甲", "一"), ("乙", "二")), 1)
    )
    components = tuple((None, fact, None) for fact in facts)
    state = SimpleNamespace(evidence_closed=1, conflict_open=0,
                            required_slots_open=0)
    with pytest.raises(ValueError, match="not unique"):
        composer.generate_composition(components, state)
    selected = composer.generate_composition(components, state, variant_id=2)
    assert selected.surface == "一|甲|二!乙"


def test_multi_claim_delivery_keeps_one_query_state_and_memory_proof(tmp_path):
    """多 claim 生成、O/H/E 与 delivery 必须共享同一闭合 QueryState。"""
    if not CANONICAL_RELEASE.is_file() or not CANONICAL_COMPOSITION_GRAPH.is_file():
        pytest.skip("真实发布图或多 claim sidecar 不可用")
    database = CANONICAL_RELEASE
    graph = CANONICAL_COMPOSITION_GRAPH
    memory_database = tmp_path / "interaction.sqlite3"
    with TrainedGraphQueryBridge(
            database, memory_database=memory_database, session_id=99101) as bridge:
        relation_pairs = tuple(sorted({
            (RelationSurfaceStructureAdapter.from_facts((fact,))[0]
             .relation_structure_key(), fact.predicate.stable_key())
            for fact in bridge.facts
        }))
        with TrainedGenerationConnectorRuntime(
                database, shared_relation_runtime=bridge.core_runtime,
                relation_pairs=relation_pairs) as relation_generator:
            composer = TrainedMultiClaimComposer(graph, relation_generator)
            bridge.surface_generator = composer

            # This is a real Memory O/H/E observation.  It participates in
            # the same frontier but carries no factual claim for the query.
            context_append = bridge.memory.append(
                "用户在讨论汽车相关话题。", speaker_kind=1)
            input_observation = bridge.memory.intake.result_for_source(
                context_append.source).observation_ref
            assert input_observation is not None

            # Use the production bridge to obtain a closed QueryState and its
            # factual pending generation.  The composer remains installed but
            # a single-claim input still takes the ordinary factual route.
            result = bridge.query(
                "按开关导致灯亮",
                minimum_depth=1,
                max_depth=3,
                input_observation_ref=input_observation,
            )
            pending = bridge._pending_response_delivery
            assert result["termination"] == TERMINATION_ANSWER_CLOSED
            assert pending is not None
            state, single_generation = pending
            assert isinstance(single_generation, FactualResponseGeneration)
            assert state.evidence_closed == 1
            assert state.conflict_open == 0
            assert state.required_slots_open == 0
            assert {root.owner_space for root in state.roots} == {
                SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
            assert all(root.status == ROOT_EXPANDED for root in state.roots)

            first = bridge.core_fact_index[state.best_candidate_key]
            second = next(
                fact for fact in bridge.facts
                if fact.proposition.stable_key() != state.best_candidate_key
                and tuple(binding.role.stable_key() for binding in fact.bindings)
                == tuple(binding.role.stable_key() for binding in first.bindings))
            facts = (first, second)
            generations = {
                fact.proposition.stable_key(): relation_generator.generate_relation(
                    bridge.core_runtime.generation_input(fact.proposition), fact)
                for fact in facts
            }
            hops = tuple(
                QueryHop(
                    SPACE_CORE,
                    fact.proposition.stable_key(),
                    fact.predicate.stable_key(),
                    fact.bindings[0].filler.stable_key(),
                    fact.bindings[1].filler.stable_key(),
                    tuple(map(ord, fact.bindings[0].surface)),
                    1,
                    fact.source_hash,
                    tuple(binding.filler.stable_key()
                          for binding in fact.bindings),
                )
                for fact in facts
            )
            bindings = list(state.bindings)
            evidence = list(state.evidence)
            for fact in facts:
                source = bridge.core_runtime.generation_input(fact.proposition)
                bindings.extend(
                    BindingEntry(
                        binding.role.stable_key(),
                        binding.filler.stable_key(),
                        space=SPACE_CORE,
                        scope_key=source.hypothesis.scope.stable_key(),
                    )
                    for binding in fact.bindings)
                evidence.extend(core_query_evidence(
                    fact.proposition,
                    bridge.core_runtime.evidence_history(fact.proposition),
                    bridge.core_runtime.context.scoped_identity_store.registry,
                ))
            claims = tuple(fact.proposition.stable_key() for fact in facts)
            state = state.with_(
                bindings=tuple(sorted(set(bindings), key=lambda item: item.stable_key())),
                evidence=tuple(sorted(set(evidence), key=lambda item: item.stable_key())),
                best_candidate_key=claims[0],
                second_candidate_key=claims[1],
                best_score=200,
                second_score=100,
                termination=TERMINATION_ANSWER_CLOSED,
                evidence_closed=1,
                conflict_open=0,
                required_slots_open=0,
                generation_ready=1,
            )
            evidences = tuple(
                ClosedGenerationEvidence(
                    fact.bindings,
                    generations[fact.proposition.stable_key()],
                    fact.proposition.stable_key(),
                    fact.predicate.stable_key(),
                    fact.source_hash,
                    anchor_surface=fact.evidence_surface,
                    connector=generations[fact.proposition.stable_key()].connector,
                )
                for fact in facts
            )
            plan = build_multi_response_plan(
                evidences,
                composer.generate_composition(
                    tuple(
                        (
                            bridge.core_runtime.generation_input(fact.proposition),
                            fact,
                            generations[fact.proposition.stable_key()],
                        )
                        for fact in facts
                    ),
                    state,
                ),
            )
            assert plan is not None
            assert tuple(item.stable_key() for item in plan.claim_refs) == tuple(sorted(claims))
            # bind_query_evidence is called by the bridge route; call it here
            # explicitly as the delivery contract under test.
            plan = bind_query_evidence(plan, state)
            assert {
                claim for claim, _refs in plan.claim_evidence_refs
            } == set(claims)
            for claim, refs in plan.claim_evidence_refs:
                assert refs
                assert all(
                    entry.space == SPACE_CORE
                    and entry.polarity == 1
                    and entry.hypothesis_key == claim
                    and entry.stable_key() in refs
                    for entry in state.evidence
                    if entry.space == SPACE_CORE
                    and entry.polarity == 1
                    and entry.hypothesis_key == claim
                )
            generation = FactualResponseGeneration(
                plan,
                input_observation,
                claims[0],
                (91630, state.depth, state.node_count,
                 state.edge_count, state.read_count),
                tuple(map(ord, plan.surface())),
                tuple(item.stable_key() for item in plan.claim_refs),
            )
            assert generation.stable_key()
            assert generation.claim_keys == tuple(
                item.stable_key() for item in plan.claim_refs)
            receipt = record_factual_response_delivery(
                bridge.memory, state, generation)
            summary = receipt.summary()
            assert summary["claim_count"] == 2
            assert summary["evidence_ref_count"] >= 2
            assert len(summary["role_occurrences"]) == len(plan.slot_sequence)
            assert summary["delivered"] == 1
            proof_key = read_delivery_proof(bridge.memory, receipt.proof_ref)
            outer = _parts(proof_key, DELIVERY_PROOF_IDENTITY)
            assert outer[0] == (4,)
            delivery_parts = _parts(outer[4], 4)
            encoded_claims = _parts(delivery_parts[1], 5)
            decoded_claims = tuple(
                _parts(item, ordinal + 1)[0]
                for ordinal, item in enumerate(encoded_claims)
            )
            assert set(decoded_claims) == set(claims)
            structures = bridge.memory.active_structures(
                observation_refs=(input_observation,), context_depth_limit=2)
            assert structures.observations
            assert structures.hypotheses
            assert structures.evidence


def test_three_claim_surface_variant_delivery_keeps_one_query_state_and_proof(
        tmp_path):
    """A trained three-claim surface variant must close and deliver atomically."""
    if not CANONICAL_RELEASE.is_file() or not SURFACE_VARIANT_GRAPH.is_file():
        pytest.skip("真实发布图或三 claim sidecar 不可用")
    with TrainedGraphQueryBridge(
            CANONICAL_RELEASE,
            memory_database=tmp_path / "interaction-three.sqlite3",
            session_id=99102) as bridge:
        relation_pairs = tuple(sorted({
            (RelationSurfaceStructureAdapter.from_facts((fact,))[0]
             .relation_structure_key(), fact.predicate.stable_key())
            for fact in bridge.facts
        }))
        with TrainedGenerationConnectorRuntime(
                CANONICAL_RELEASE, shared_relation_runtime=bridge.core_runtime,
                relation_pairs=relation_pairs) as relation_generator:
            composer = TrainedMultiClaimComposer(
                SURFACE_VARIANT_GRAPH, relation_generator)
            bridge.surface_generator = composer
            context_append = bridge.memory.append(
                "用户在讨论汽车相关话题。", speaker_kind=1)
            input_observation = bridge.memory.intake.result_for_source(
                context_append.source).observation_ref
            assert input_observation is not None
            result = bridge.query(
                "按开关导致灯亮",
                minimum_depth=1,
                max_depth=3,
                input_observation_ref=input_observation,
            )
            pending = bridge._pending_response_delivery
            assert result["termination"] == TERMINATION_ANSWER_CLOSED
            assert pending is not None
            state, _single_generation = pending
            # The release graph has only two facts for the query's exact role
            # schema.  A multi-claim composition may combine distinct role
            # schemas, so choose three two-slot propositions with disjoint
            # role/filler pairs instead of assuming schema repetition.
            selected = []
            occupied = set()
            for candidate in bridge.facts:
                if len(candidate.bindings) != 2:
                    continue
                pairs = {(binding.role.stable_key(), binding.filler.stable_key())
                         for binding in candidate.bindings}
                if pairs & occupied:
                    continue
                selected.append(candidate)
                occupied.update(pairs)
                if len(selected) == 3:
                    break
            assert len(selected) == 3
            facts = tuple(selected)
            generations = {
                fact.proposition.stable_key(): relation_generator.generate_relation(
                    bridge.core_runtime.generation_input(fact.proposition), fact)
                for fact in facts
            }
            bindings = list(state.bindings)
            evidence = list(state.evidence)
            for fact in facts:
                source = bridge.core_runtime.generation_input(fact.proposition)
                bindings.extend(
                    BindingEntry(
                        binding.role.stable_key(), binding.filler.stable_key(),
                        space=SPACE_CORE,
                        scope_key=source.hypothesis.scope.stable_key(),
                    ) for binding in fact.bindings)
                evidence.extend(core_query_evidence(
                    fact.proposition,
                    bridge.core_runtime.evidence_history(fact.proposition),
                    bridge.core_runtime.context.scoped_identity_store.registry,
                ))
            claims = tuple(fact.proposition.stable_key() for fact in facts)
            state = state.with_(
                bindings=tuple(sorted(set(bindings), key=lambda item: item.stable_key())),
                evidence=tuple(sorted(set(evidence), key=lambda item: item.stable_key())),
                best_candidate_key=claims[0],
                second_candidate_key=claims[1],
                best_score=300,
                second_score=200,
                termination=TERMINATION_ANSWER_CLOSED,
                evidence_closed=1,
                conflict_open=0,
                required_slots_open=0,
                generation_ready=1,
            )
            evidences = tuple(
                ClosedGenerationEvidence(
                    fact.bindings,
                    generations[fact.proposition.stable_key()],
                    fact.proposition.stable_key(),
                    fact.predicate.stable_key(),
                    fact.source_hash,
                    anchor_surface=fact.evidence_surface,
                    connector=generations[fact.proposition.stable_key()].connector,
                ) for fact in facts)
            components = tuple(
                (bridge.core_runtime.generation_input(fact.proposition),
                 fact, generations[fact.proposition.stable_key()])
                for fact in facts)
            composed = composer.generate_composition(
                components, state, variant_id=2)
            plan = build_multi_response_plan(evidences, composed)
            assert plan is not None
            assert len(plan.claim_refs) == 3
            plan = bind_query_evidence(plan, state)
            assert {claim for claim, _refs in plan.claim_evidence_refs} == set(claims)
            assert all(refs for _claim, refs in plan.claim_evidence_refs)
            generation = FactualResponseGeneration(
                plan,
                input_observation,
                claims[0],
                (91630, state.depth, state.node_count,
                 state.edge_count, state.read_count),
                tuple(map(ord, plan.surface())),
                tuple(item.stable_key() for item in plan.claim_refs),
            )
            receipt = record_factual_response_delivery(
                bridge.memory, state, generation)
            summary = receipt.summary()
            assert summary["claim_count"] == 3
            assert summary["evidence_ref_count"] >= 3
            assert summary["delivered"] == 1
            proof_key = read_delivery_proof(bridge.memory, receipt.proof_ref)
            outer = _parts(proof_key, DELIVERY_PROOF_IDENTITY)
            delivery_parts = _parts(outer[4], 4)
            encoded_claims = _parts(delivery_parts[1], 5)
            decoded_claims = tuple(
                _parts(item, ordinal + 1)[0]
                for ordinal, item in enumerate(encoded_claims))
            assert set(decoded_claims) == set(claims)
            structures = bridge.memory.active_structures(
                observation_refs=(input_observation,), context_depth_limit=2)
            assert structures.observations
            assert structures.hypotheses
            assert structures.evidence
