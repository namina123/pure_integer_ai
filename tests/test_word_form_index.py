"""语言地基词形目录与正向最大匹配测试。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_CAUSES, ensure_relation_primitives,
)
from pure_integer_ai.cognition.understanding.emergent_relation_signal import (
    record_emergent_relation_signal_shadow,
)
from pure_integer_ai.cognition.understanding.word_form_index import WordFormIndex
from pure_integer_ai.cognition.understanding.cue_words import (
    CAUSES_CUE_FORWARD, cue_type_of,
)
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.node_store import NODE_CONCEPT, NODE_WORD, TIER_PRIMARY
from pure_integer_ai.storage.word_form_index import WORD_FORM_INDEX_TABLE
from pure_integer_ai.training.cursor import dump_run, load_run


def test_word_form_registration_is_semantic_free_and_typed():
    backend = DictBackend()
    ctx = make_train_context(backend)
    concept = ctx.concept_index.ensure(
        "南京市", space_id=ctx.space_id, node_type=NODE_CONCEPT)
    index = WordFormIndex(backend, ctx.concept_index)

    word = index.register("南京市", language=LANG_ZH, space_id=ctx.space_id)

    assert word != concept
    assert ctx.node_store.get(*concept)["type"] == NODE_CONCEPT
    assert ctx.node_store.get(*word)["type"] == NODE_WORD
    assert backend.count("edge") == 0, "词形学习不得注入语义边"
    assert index.register("南京市", language=LANG_ZH,
                          space_id=ctx.space_id) == word
    assert len(backend.select(WORD_FORM_INDEX_TABLE)) == len("南京市")


def test_forward_maximum_match_uses_persistent_inventory():
    backend = DictBackend()
    ctx = make_train_context(backend)
    index = WordFormIndex(backend, ctx.concept_index)
    for surface in ("南京", "南京市", "市长", "长江", "大桥"):
        index.register(surface, language=LANG_ZH, space_id=ctx.space_id)

    assert index.segment("南京市长江大桥X", language=LANG_ZH,
                         space_id=ctx.space_id) == ["南京市", "长江", "大桥", "X"]


def test_word_form_inventory_survives_dump_and_resume(tmp_path):
    first = DictBackend()
    ctx1 = make_train_context(first)
    index1 = WordFormIndex(first, ctx1.concept_index)
    expected = index1.register("逻辑结构", language=LANG_ZH,
                               space_id=ctx1.space_id)
    config = FormalTrainConfig(run_dir=str(tmp_path), run_id="word_forms")
    dump_run(first, config.run_dir, config.run_id, spaces=[ctx1.space_id],
             tables=config.dump_tables)

    second = DictBackend()
    ctx2 = make_train_context(second)
    load_run(second, config.run_dir, config.run_id)
    index2 = WordFormIndex(second, ctx2.concept_index)

    assert index2.forms(language=LANG_ZH,
                        space_id=ctx2.space_id)[tuple(map(ord, "逻辑结构"))] == expected
    assert index2.segment("逻辑结构", language=LANG_ZH,
                          space_id=ctx2.space_id) == ["逻辑结构"]


def test_learned_relation_can_read_back_through_typed_word_form(monkeypatch):
    backend = DictBackend()
    ctx = make_train_context(backend)
    index = WordFormIndex(backend, ctx.concept_index)
    word = index.register("引发", language=LANG_ZH, space_id=ctx.space_id)
    rel = ensure_relation_primitives(ctx.concept_index, backend,
                                     space_id=ctx.space_id)[REL_CAUSES]
    record_emergent_relation_signal_shadow(ctx.edge_store, word, rel,
                                           space_id=ctx.space_id)
    ctx.edge_store.set_tier(
        space_id_from=word[0], local_id_from=word[1],
        space_id_to=rel[0], local_id_to=rel[1],
        edge_type=EDGE_RELATION_SIGNAL, new_tier=TIER_PRIMARY)
    monkeypatch.setattr(gates, "EMERGENT_RELATION_CUE_READBACK_MODE", True)

    assert ctx.concept_index.lookup("引发", ctx.space_id) is None
    assert cue_type_of(
        "引发", LANG_ZH, backend=backend, edge_store=ctx.edge_store,
        space_id=ctx.space_id, concept_index=ctx.concept_index,
    ) == CAUSES_CUE_FORWARD
