"""逻辑结构理论从学习结果到生成消费的端到端探针。"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveredOperator, tally_cue_slot_matches,
)
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_CAUSES, ensure_relation_primitives,
)
from pure_integer_ai.cognition.shared.types import LANG_NONE
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.instantiates import build_instantiates_edge
from pure_integer_ai.cognition.understanding.realizes import build_realizes_edge
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.structure_match_count import (
    read_structure_match_count, register_structure_match_count,
)
from pure_integer_ai.experiments.formal_train import make_train_context, _promote_eligible
from pure_integer_ai.training.promote import PROMOTE_STRUCTURE_MATCH_MIN
from tests.test_correspondence_bridge import (
    _build_seg, _build_skel, _ensure, _path,
)
from tests.test_structure_reverse_inference import _build_input_tree


def test_learned_logic_cue_reaches_generation(monkeypatch):
    ctx = make_train_context(DictBackend())
    register_structure_match_count(ctx.backend)
    sid = ctx.space_id

    # 受控理论：已掌握的三元 CAUSES 骨架，cue 位使用“使”作为教学样本。
    struct_ref, _tokens = _build_seg(ctx, "__seg_theory", ["雨", "引发", "洪水"])
    skeleton = _ensure(ctx, "__skel_theory")
    cue = _ensure(ctx, "使")
    leaves = [_ensure(ctx, f"__theory_leaf{i}") for i in range(3)]
    _build_skel(ctx, skeleton, leaves, cue_at=1, cue_ref=cue)
    build_instantiates_edge(ctx.edge_store, struct_ref, skeleton, space_id=sid)
    rel = ensure_relation_primitives(ctx.concept_index, ctx.backend,
                                     space_id=sid)[REL_CAUSES]
    build_realizes_edge(ctx.edge_store, skeleton, rel, space_id=sid)
    operator = DiscoveredOperator(
        name="__theory_causes", skeleton_ref=skeleton, arity=3, sample_count=4)

    # 学习阶段：新词“引发”在同一逻辑 cue 位出现三次，先只创建 SHADOW。
    input_roots = []
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN):
        root = _ensure(ctx, f"__theory_input{i}")
        input_roots.append(root)
        _build_input_tree(ctx, root, [
            _ensure(ctx, f"__rain{i}"), _ensure(ctx, "引发"),
            _ensure(ctx, f"__flood{i}"),
        ])
    monkeypatch.setattr(gates, "ORACLE_PROMOTE_MODE", True)
    monkeypatch.setattr(gates, "CUE_CLUSTER_MODE", True)
    tally_cue_slot_matches(
        input_roots, discovered_operators=[operator],
        graph=ctx.concept_graph, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid,
        rel_primitives=ensure_relation_primitives(ctx.concept_index, ctx.backend,
                                                  space_id=sid),
    )
    learned = _ensure(ctx, "引发")
    assert read_structure_match_count(
        ctx.backend, space_id=sid, word_ref=learned,
        rel_kind=REL_CAUSES) == PROMOTE_STRUCTURE_MATCH_MIN
    assert ctx.concept_graph.relation_cue_candidates(
        REL_CAUSES, space_id=sid) == [], "SHADOW 理论不能提前进入生成候选"

    # 生产晋升门只接受结构证据，未晋升前不能被生成器消费。
    promoted, _ = _promote_eligible(ctx, teacher=None)
    assert promoted >= 1
    learned_edge = ctx.edge_store.get(
        space_id_from=learned[0], local_id_from=learned[1],
        space_id_to=rel[0], local_id_to=rel[1],
        edge_type=EDGE_RELATION_SIGNAL)
    assert learned_edge["tier"] == TIER_PRIMARY
    assert ctx.concept_graph.relation_cue_candidates(
        REL_CAUSES, space_id=sid) == [learned]

    monkeypatch.setattr(gates, "CORRESPONDENCE_SLOT_MODE", True)
    monkeypatch.setattr(gates, "CUE_SLOT_FILL_MODE", True)
    monkeypatch.setattr(gates, "DISPATCH_TOKEN_CHAIN_MODE", True)
    monkeypatch.setattr(gates, "ORDINAL_SURFACE_MODE", True)
    output = generate_output(_path(struct_ref), ctx.concept_graph,
                             WorkMemory(), LANG_NONE)

    assert output.parts[0].words == ["雨", "引发", "洪水"]
