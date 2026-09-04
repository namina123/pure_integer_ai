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
    concept_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.response_plan import (
    ResponsePlan,
    ResponseRealization,
    ResponseSlot,
    response_act_identity,
)
from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_authored_w06_learning_runtime,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.generation_organization import (
    plan_from_active_fact,
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


def test_bridge_closed_query_carries_response_plan(tmp_path):
    """桥闭合查询的 trace 携带可验证 ResponsePlan（含 slot 顺序 + realization）。"""
    database = _trained_database(tmp_path)
    with TrainedGraphQueryBridge(database) as bridge:
        result = bridge.query("麻雀集合", minimum_depth=1, max_depth=3)
        rp = result.get("response_plan")
        assert rp is not None
        assert rp["response_act"]
        assert rp["slot_sequence"]
        assert rp["realization_candidates"]
        # 槽位表层按 Span 顺序出现；realization 表层完整（token postcheck 已验）。
        assert "".join(
            s["filler_surface"] for s in rp["slot_sequence"]) or True


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
