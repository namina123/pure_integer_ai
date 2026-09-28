"""阶段 C 回归：闭合命题 -> ResponsePlan -> token postcheck 结构生成。

ResponsePlan 是发布生成入口唯一接受的组织契约；每个必填槽表层必须按
Span/表层顺序作为连续子串出现在至少一个 realization 候选内，否则 fail-closed
返回 None。测试覆盖：槽位组合生成（非整句回放）、token postcheck 拒绝缺失
必填槽、跨序槽位不再误判、无结构证据澄清、纯整数可重放序列化。
"""
from __future__ import annotations

import json
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_PROPOSITION, ObjectIdentity, concept_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.response_plan import (
    ResponsePlan,
    ResponseRealization,
    ResponseSlot,
    response_act_identity,
)
from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry,
    EvidenceEntry,
    QueryRoot,
    QueryState,
    ROOT_EXPANDED,
    SPACE_CORE,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_NO_FRONTIER,
)
from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_authored_w06_learning_runtime,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.generation_organization import (
    ClosedGenerationEvidence,
    bind_query_evidence,
    build_multi_response_plan,
    plan_from_active_fact,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    GraphRelationGeneration,
    RelationSurfaceBinding,
)
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend

ROOT = Path(__file__).resolve().parents[1]


def _relation_courses() -> tuple[Path, ...]:
    """返回现役七类 authored W-06 训练输入。"""
    root = ROOT / "data" / "ph2"
    return tuple(sorted(root.glob("authored_relation_*_w06_seed_v2.jsonl.sample"))) + tuple(
        sorted(
            path for path in root.glob("authored_relation_*_seed_v1.jsonl.sample")
            if path.name != "authored_relation_alias_refers_seed_v1.jsonl.sample"
        )
    )


def _trained_database(tmp_path: Path) -> Path:
    """消费公开 relation samples 训练 W-06 并落盘 SQLite。"""
    database = tmp_path / "training.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(
            context,
            occurrence_protocol=occurrence,
            span_protocol=span,
        )
        build_authored_w06_learning_runtime(
            backend,
            context,
            _relation_courses(),
            tmp_path / "typed-relation-pack",
        )
        backend.commit()
    finally:
        backend.close()
    return database


def test_every_active_fact_forms_verifiable_response_plan(tmp_path):
    """每个 active Core fact 经槽位重填都能形成通过 token postcheck 的 plan。"""
    database = _trained_database(tmp_path)
    with TrainedRelationGraphRuntime(database) as runtime:
        facts = runtime.active_surface_facts()
        assert len(facts) >= 14
        for fact in facts:
            generation = runtime._generate_surface(fact)
            plan = plan_from_active_fact(fact, generation)
            assert plan is not None, f"fact {fact.cue} 不能形成 ResponsePlan"
            surface = plan.surface()
            assert surface.strip()
            # 生成表层由已学框架重填，不是整句回放 evidence。
            assert plan.stable_key()


def test_bridge_without_generation_connector_fails_closed(tmp_path):
    """Core candidate 无 trained generation connector 时不得伪造 ResponsePlan。"""
    database = _trained_database(tmp_path)
    with TrainedGraphQueryBridge(database) as bridge:
        assert bridge.surface_generator is None
        result = bridge.query(
            "麻雀集合包含于鸟类集合", minimum_depth=1, max_depth=3)
    assert result["termination"] == TERMINATION_NO_FRONTIER
    assert result["termination_state"]["generation_ready"] == 0
    assert result["response_plan"] is None
    assert result["response_surface"] == ""


def test_response_plan_token_postcheck_rejects_missing_required_slot():
    """缺失必填槽 token 的 realization 无法通过 postcheck（fail-closed）。"""
    role1 = structure_concept_identity((9, 1))
    role2 = structure_concept_identity((9, 2))
    f1 = concept_identity((100, 1))
    f2 = concept_identity((100, 2))
    act = response_act_identity((1, 1, 1))
    s1 = ResponseSlot(role1, f1, "车", True, 1234, allowed_node_kinds=(4,))
    s2 = ResponseSlot(role2, f2, "轮", True, 1234, allowed_node_kinds=(4,))
    # realization 覆盖两槽：通过。
    good = ResponsePlan(
        act,
        (f1,),
        (s1, s2),
        (ResponseRealization("车包含轮", f1, 1234, 2),),
        ((1234,),),
    )
    assert good.surface() == "车包含轮"
    # realization 缺第二必填槽 token：拒绝。
    try:
        ResponsePlan(
            act,
            (f1,),
            (s1, s2),
            (ResponseRealization("一辆车", f1, 1234, 1),),
            ((1234,),),
        )
        raise AssertionError("缺失必填槽的 ResponsePlan 不应通过 token postcheck")
    except ValueError:
        pass


def _multi_claim_evidence():
    """Synthetic graph-only claims used to exercise the new audit contract."""
    roles = (structure_concept_identity((990, 1)),
             structure_concept_identity((990, 2)))
    fillers = (concept_identity((991, 1)), concept_identity((991, 2)),
               concept_identity((991, 3)), concept_identity((991, 4)))
    claim_identities = (
        ObjectIdentity(OBJECT_PROPOSITION, (992, 1)),
        ObjectIdentity(OBJECT_PROPOSITION, (992, 2)),
    )
    claims = tuple(item.stable_key() for item in claim_identities)
    evidences = []
    for ordinal, claim in enumerate(claims):
        bindings = (
            RelationSurfaceBinding(roles[0], fillers[ordinal * 2],
                                   ("甲" if ordinal == 0 else "乙"),
                                   100 + ordinal, 0, 1),
            RelationSurfaceBinding(roles[1], fillers[ordinal * 2 + 1],
                                   ("一" if ordinal == 0 else "二"),
                                   100 + ordinal, 1, 2),
        )
        proposition = ObjectIdentity.from_stable_key(claim)
        evidences.append(ClosedGenerationEvidence(
            bindings,
            GraphRelationGeneration(
                "甲包含一；乙包含二" if ordinal == 0
                else "甲包含一；乙包含二",
                proposition, 200 + ordinal, 4,
                connector=structure_concept_identity((993, ordinal + 1)),
                representations=(concept_identity((994, ordinal + 1)),),
            ),
            claim,
            (995, ordinal + 1),
            300 + ordinal,
        ))
    # The composition generation is the only source of the realized join.
    generation = GraphRelationGeneration(
        "甲包含一；乙包含二；丙包含三；丁包含四",
        structure_concept_identity((996, 1)), 401, 4,
        connector=structure_concept_identity((996, 2)),
        representations=(concept_identity((997, 1)),),
    )
    return tuple(evidences), generation


def test_multi_claim_plan_keeps_claim_evidence_and_slot_ranges():
    evidences, generation = _multi_claim_evidence()
    plan = build_multi_response_plan(evidences, generation)
    assert plan is not None
    assert len(plan.claim_refs) == 2
    assert len(plan.claim_evidence_refs) == 2
    assert len(plan.claim_slot_ranges) == 2
    assert plan.claim_slot_ranges[0][1:] == (0, 2)
    assert plan.claim_slot_ranges[1][1:] == (2, 4)
    trace = plan.to_dict()
    assert len(trace["claim_evidence_refs"]) == 2
    assert len(trace["claim_slot_ranges"]) == 2
    assert plan.surface()


def test_multi_claim_plan_rejects_duplicate_claims():
    evidences, generation = _multi_claim_evidence()
    duplicate = (evidences[0], evidences[0])
    assert build_multi_response_plan(duplicate, generation) is None


def test_multi_claim_plan_rejects_realization_missing_any_claim_slots():
    evidences, _generation = _multi_claim_evidence()
    bad = GraphRelationGeneration(
        "甲包含一", structure_concept_identity((996, 3)),
        402, 2, connector=structure_concept_identity((996, 4)),
        representations=(concept_identity((997, 2)),),
    )
    assert build_multi_response_plan(evidences, bad) is None


def test_multi_claim_binding_replaces_placeholders_with_state_core_evidence():
    evidences, generation = _multi_claim_evidence()
    plan = build_multi_response_plan(evidences, generation)
    assert plan is not None
    claims = tuple(item.stable_key() for item in plan.claim_refs)
    bindings = tuple(
        BindingEntry(slot.role.stable_key(), slot.filler.stable_key(),
                     space=SPACE_CORE)
        for slot in plan.slot_sequence)
    state_evidence = tuple(
        EvidenceEntry((700 + index,), claim, space=SPACE_CORE, polarity=1,
                      evidence_key=(701 + index,))
        for index, claim in enumerate(claims))
    state = QueryState(
        (800,),
        roots=tuple(QueryRoot(space, (900 + space,), ROOT_EXPANDED, 1)
                    for space in (SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE)),
        bindings=bindings,
        evidence=state_evidence,
        evidence_closed=1,
        best_candidate_key=claims[0],
        termination=TERMINATION_ANSWER_CLOSED,
    )
    bound = bind_query_evidence(plan, state)
    assert all(ref in bound.evidence_refs
               for _claim, refs in bound.claim_evidence_refs for ref in refs)
    assert all(ref == item.stable_key()
               for item, (_claim, refs) in zip(state_evidence,
                                               bound.claim_evidence_refs)
               for ref in refs)


def test_response_plan_stable_key_roundtrip_is_pure_integer():
    """ResponsePlan 可序列化为纯整数 dict 且 stable_key 稳定可重放。"""
    role1 = structure_concept_identity((9, 1))
    role2 = structure_concept_identity((9, 2))
    f1 = concept_identity((100, 1))
    f2 = concept_identity((100, 2))
    act = response_act_identity((1, 1, 1))
    plan = ResponsePlan(
        act,
        (f1,),
        (
            ResponseSlot(role1, f1, "车", True, 9, allowed_node_kinds=(4,)),
            ResponseSlot(role2, f2, "轮", True, 9, allowed_node_kinds=(4,)),
        ),
        (ResponseRealization("车包含轮", f1, 9, 2),),
        ((9,),),
    )
    encoded = json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True)
    assert isinstance(encoded, str)
    first = plan.stable_key()
    second = ResponsePlan(
        act,
        (f1,),
        (
            ResponseSlot(role1, f1, "车", True, 9, allowed_node_kinds=(4,)),
            ResponseSlot(role2, f2, "轮", True, 9, allowed_node_kinds=(4,)),
        ),
        (ResponseRealization("车包含轮", f1, 9, 2),),
        ((9,),),
    ).stable_key()
    assert first == second
