"""语料相关 KB 过滤（perf fix）·两个 helper 的直接单测（对抗审 #2 H3 收口）。

承接 doc/重来_语料相关KB过滤_2026-07-16.md。formalize 进 boot code（collection.py 两个 helper：
corpus_relevant_vocab + filter_pairs_to_vocab）后·两函数此前无直接单测（仅经 CI 空路径 transitively 覆盖）。
本测直接覆盖两函数的全部逻辑分支（正例 / 排除 / 空短路 / alias 四元 idx / 序保）。

机制：
  CR1 corpus_relevant_vocab：language 模态 item tokens 并集 → frozenset[str]。
  CR2 corpus_relevant_vocab：非 language（arith/code）item tokens 不入 vocab（KB 按 language 语料过滤）。
  CR3 corpus_relevant_vocab：空语料 → 空 frozenset（filter_pairs_to_vocab 空 vocab 短路返原 pairs）。
  CR4 corpus_relevant_vocab：language item 无 tokens（空 list）不崩（frozenset 略过）。
  FP1 filter_pairs_to_vocab：≥1 surface 在 vocab → 留（idx_a 或 idx_b 命中均留）。
  FP2 filter_pairs_to_vocab：两 surface 均不在 vocab → 去（out-of-corpus inert pair）。
  FP3 filter_pairs_to_vocab：空 vocab → 返原 pairs 逐字（bit-identical 守·短路·无过滤副作用）。
  FP4 filter_pairs_to_vocab：空 pairs → 返 []（bit-identical 守·CI resolve [] 短路）。
  FP5 filter_pairs_to_vocab：alias 四元 (surf_a,lang_a,surf_b,lang_b) idx=0,2 → lang int 位（1/3）永不参与
      membership（str in frozenset[str]·lang int 不会被误判·守 alias 跨语言桥）。
  FP6 filter_pairs_to_vocab：保输入序（order-preserving comprehension·resolve 行序确定）。

铁律：确定性（frozenset membership / 输入序确定）/ 不写死（vocab 来自语料外部数据·过滤是 scoping 非语义）/
  bit-identical（空 vocab/空 pairs 短路·CI resolve [] → 返 [] ·零过滤副作用）。
"""
from __future__ import annotations

from pure_integer_ai.experiments.collection import (
    corpus_relevant_vocab, filter_pairs_to_vocab,
    CollectedItem, COLLECT_PRECEDES, SOURCE_BARE_TEXT,
    MODALITY_LANGUAGE, MODALITY_ARITH,
)


def _lang_item(tokens: list[str]) -> CollectedItem:
    """language 模态 item（modality 默认 MODALITY_LANGUAGE·tokens=给值）。"""
    return CollectedItem(tokens=tokens, collect_type=COLLECT_PRECEDES, source=SOURCE_BARE_TEXT)


def _arith_item(tokens: list[str]) -> CollectedItem:
    """arith 模态 item（modality=MODALITY_ARITH·KB 语料相关过滤应排除其 tokens）。"""
    return CollectedItem(tokens=tokens, modality=MODALITY_ARITH)


# ---- CR1 corpus_relevant_vocab：language tokens 并集 ----

def test_cr1_corpus_relevant_vocab_unions_language_tokens():
    """CR1：language 模态 item 的 tokens 并集 → frozenset[str]。

    两 language item（tokens {苹果,是,水果} + {香蕉,水果}）→ 并集 {苹果,是,水果,香蕉}。
    返 frozenset 类型（确定性·order-independent）。
    """
    corpus = [_lang_item(["苹果", "是", "水果"]), _lang_item(["香蕉", "水果"])]
    v = corpus_relevant_vocab(corpus)
    assert isinstance(v, frozenset), "返 frozenset（确定性·membership 用）"
    assert v == frozenset({"苹果", "是", "水果", "香蕉"}), "language tokens 并集"


# ---- CR2 非 language（arith/code）tokens 不入 vocab ----

def test_cr2_corpus_relevant_vocab_excludes_non_language_modalities():
    """CR2：arith/code 模态 item 的 tokens 不入 vocab（KB facts 是语言域 alias/similar/...·按 language 语料过滤）。

    language item {苹果} + arith item {算术记号} → vocab 只 {苹果}（算术记号 不入）。
    """
    corpus = [_lang_item(["苹果"]), _arith_item(["算术记号"])]
    v = corpus_relevant_vocab(corpus)
    assert v == frozenset({"苹果"}), "arith tokens 不入 vocab"
    assert "算术记号" not in v, "非 language 模态 token 严格排除"


# ---- CR3 空语料 → 空 frozenset ----

def test_cr3_corpus_relevant_vocab_empty_corpus_returns_empty():
    """CR3：空语料 → 空 frozenset（filter_pairs_to_vocab 空 vocab 短路返原 pairs·守 bit-identical）。"""
    assert corpus_relevant_vocab([]) == frozenset(), "空语料 → 空 vocab"


# ---- CR4 language item 无 tokens 不崩 ----

def test_cr4_corpus_relevant_vocab_item_without_tokens_no_crash():
    """CR4：language item tokens=空 list 不崩（frozenset 略过空序列·getattr tokens truthy 守）。

    language item（空 tokens）+ language item（有 tokens）→ vocab 只后者 tokens（空不贡献）。
    """
    corpus = [_lang_item([]), _lang_item(["苹果"])]
    assert corpus_relevant_vocab(corpus) == frozenset({"苹果"}), "空 tokens item 不崩不贡献"


# ---- FP1 ≥1 surface 在 vocab → 留 ----

def test_fp1_filter_keeps_pair_with_at_least_one_surface_in_vocab():
    """FP1：pair 的 idx_a 或 idx_b 任一在 vocab → 留（corpus word 的关系全保留）。"""
    vocab = frozenset({"苹果"})
    pairs = [("苹果", "梨"), ("梨", "苹果"), ("梨", "香蕉")]   # 前两留（≥1 命中）·末去
    out = filter_pairs_to_vocab(pairs, vocab)
    assert out == [("苹果", "梨"), ("梨", "苹果")], "≥1 surface 在 vocab 留·双不命中去"


# ---- FP2 两 surface 均不在 vocab → 去 ----

def test_fp2_filter_drops_pair_with_no_surface_in_vocab():
    """FP2：两 surface 均不在 vocab → 去（out-of-corpus inert pair·语料永不提及→零学习信号）。"""
    vocab = frozenset({"苹果"})
    pairs = [("梨", "桃"), ("李子", "杏")]
    assert filter_pairs_to_vocab(pairs, vocab) == [], "out-of-corpus pair 全去"


# ---- FP3 空 vocab → 返原 pairs 逐字（bit-identical 守）----

def test_fp3_filter_empty_vocab_returns_pairs_unchanged():
    """FP3：空 vocab（空语料）→ 返原 pairs 逐字（短路·守 bit-identical·无过滤副作用）。

    bit-identical crux：空 vocab 不过滤（返原 pairs）·非返 []。这是守 CI 空路径之外的「空语料」语义——
    无 language 语料时不过滤 KB（PURE_ALIAS 不入 reward/拓扑·inert·文档已知边界）。
    """
    pairs = [("苹果", "梨"), ("梨", "桃")]
    out = filter_pairs_to_vocab(pairs, frozenset())
    assert out == pairs, "空 vocab → 返原 pairs 逐字（bit-identical 短路）"
    assert out is not pairs or out == pairs   # 逐字一致（值等·不要求同对象）


# ---- FP4 空 pairs → 返 []（bit-identical 守·CI resolve [] 短路）----

def test_fp4_filter_empty_pairs_returns_empty():
    """FP4：空 pairs（CI resolve [] ·无 PURE_INTEGER_AI_LOCAL_DIR）→ 返 []（短路·零 bootstrap 调用）。

    bit-identical crux：CI 无文件→resolve_*_facts 返 []→filter `not pairs` 短路返 []→bootstrap 空 pairs
    首行短路 return 0·零节点/边/MARK_LANG 变化·逐字现状。
    """
    assert filter_pairs_to_vocab([], frozenset({"苹果"})) == [], "空 pairs → []（CI bit-identical 短路）"


# ---- FP5 alias 四元 idx=0,2（lang int 位 1/3 永不参与 membership）----

def test_fp5_filter_alias_four_tuple_idx_0_2_lang_ints_never_matched():
    """FP5：alias 四元 (surf_a,lang_a,surf_b,lang_b) idx=0,2 → lang int 位（1/3）永不参与 membership。

    formal_train boot：filter_pairs_to_vocab(resolve_alias_facts(), _kb_vocab, 0, 2)。lang int（LANG_ZH=1 /
    LANG_EN=2）在 idx 1/3·membership 是 `str in frozenset[str]`·lang int 不会被误判为命中（int≠str surface）·
    亦不会被 vocab（str 集）包含 → 只 surface 位（0/2）判定。守 alias 跨语言桥（apple↔苹果）。
    """
    # alias_facts 四元：apple en 苹果 zh / dog en 狗 zh（lang_a=LANG_EN=2·lang_b=LANG_ZH=1）
    from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
    pairs = [
        ("apple", LANG_EN, "苹果", LANG_ZH),    # 苹果 在 vocab → 留
        ("dog", LANG_EN, "狗", LANG_ZH),         # 两 surface 均不在 vocab → 去
        ("banana", LANG_EN, "香蕉", LANG_ZH),    # 香蕉 在 vocab → 留
    ]
    vocab = frozenset({"苹果", "香蕉"})
    out = filter_pairs_to_vocab(pairs, vocab, 0, 2)
    assert out == [
        ("apple", LANG_EN, "苹果", LANG_ZH),
        ("banana", LANG_EN, "香蕉", LANG_ZH),
    ], "alias idx=0,2：surface 位判定·lang int 位 1/3 永不参与 membership"
    # 显证：lang int（1/2）不在 vocab（str 集）→ 纵 lang 位巧合亦不命中（type 不同）
    assert LANG_ZH not in vocab and LANG_EN not in vocab, "vocab 是 str 集·lang int 不在其中"


# ---- FP6 保输入序（order-preserving）----

def test_fp6_filter_preserves_input_order():
    """FP6：过滤保 resolve 行序（list comprehension 保序·确定性·PYTHONHASHSEED=0·frozenset membership 不乱序）。

    frozenset 是 membership 数据结构（过滤判定用）·不决定输出序·输出序由 pairs 输入序（resolve 行序）决定。
    """
    vocab = frozenset({"b", "d"})
    pairs = [("d", "x"), ("a", "b"), ("z", "y"), ("b", "c")]   # 留 [0,1,3]·去 [2]
    out = filter_pairs_to_vocab(pairs, vocab)
    assert out == [("d", "x"), ("a", "b"), ("b", "c")], "保输入序·frozenset membership 不乱输出序"
