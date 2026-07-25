"""cognition.understanding.word_concept_signal — D:11 EDGE_RELATION_SIGNAL 词→关系概念边（L4·刀3 件1）。

激活 D:11 EDGE_RELATION_SIGNAL（storage/edge_types.py:74 已注册零产消者）·词→REL_* 关系概念 NODE_CONCEPT
first-class 节点的 typed signal 边。record_word_concept 建边·lookup_word_concept 读边。

**D:11 不接 reward**（effective_weight.py:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·D:11 不内·
若误传入 loud assert fail 非偷注入）·不进 PR/closure（PR 邻接只 {PRECEDES,CAUSES,REFERS_TO}·D:11 不内）·
§8.1c-bis 合规：D:11 边端点 exempt closure type filter + effective_weight assertion。

**迁移兼容来源**：_REL_LEXICAL_CUE 只为旧 D:11 boot 和兼容回归保留，不是元定义
或 readiness 真源。正式 reader 读取来源化语言信号图；formal_train 仅在显式开启
``language_signal_compatibility_enabled`` 时调用本模块的五类静态 boot。

**consumers**（刀3 无生产 caller·基建）：刀4 涌现学习（emergent 假设新词→D:11 SHADOW→验证→晋升）/
刀5 件8 词→概念（cue_extractor 从硬编码 frozenset→种子+涌现）。lookup_word_concept 提供 read API·
刀3 测 round-trip 证活。

铁律：纯整数 / 确定性（query_from 幂等 + stable surface hash）/ 不写死（自然语言表仅 compatibility）/
  外部只启发（D:11 是 signal 边非语义内容边·REL_* 是 meta定义概念非事实断言）/ IS_A 永不写死
  （D:11≠IS_A·REL_* 新建 NODE_CONCEPT 非合并）/ §8.1c（D:11 不接 reward）/ epistemic 闭合（镜像 is_a.py:54 assert）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.edge_store import (
    EdgeStore, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM, SOURCE_TEACHER,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.composes_attr import read_composes_attrs, ATTR_RELATION_PRIMITIVE
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_SUBSET, REL_MEMBER, REL_CAUSES, REL_PRECEDES, REL_MEREOLOGY, REL_EQUAL, REL_PROPERTY, REL_SIMILAR,
    ensure_relation_primitives,
)

# D:11 tentative strength（不接 reward·非学习对象初值·刀4 验证下晋升才接 reward）
RELATION_SIGNAL_STRENGTH = 1

# ---- 元定义层种子词（lang → {word: rel_kind}·镜像 cue_words.py:29 _CUE_WORDS 范式） ----
# 关系标记词（closed-class 系词/连词为主·mereology 借 content-word 关系名词"部分"/"part"作标记·
# doc §5 刀3 line181 接受 mereology 无 closed-class 连词借名词）→ REL_*。极小种子（doc §5 刀3"教师极小种子"）。
# 对抗审计纠偏（审2 P1）：标注勿统称"closed-class 系词/连词"——"部分"/"part" 是 open-class 名词·
# 仅 mereology 借用（设计接受边界·非所有 REL_* 都有 closed-class 锚）。
# 刀4 涌现学习超越此种子（emergent 假设新词）。
_REL_LEXICAL_CUE: dict[int, dict[str, int]] = {
    LANG_ZH: {
        "是": REL_SUBSET,        # 系词（copula）→ ⊂
        "属于": REL_MEMBER,      # 成员 → ∈
        "导致": REL_CAUSES,      # 因果连词 → 因果
        "引起": REL_CAUSES,
        "先于": REL_PRECEDES,    # 时序 → 前驱
        "部分": REL_MEREOLOGY,   # part-of → mereology
        "等于": REL_EQUAL,       # 相等系词 → =（STEP5·closed-class 核心·开放变体 等同于 走 D:11 教师晋升非硬编码）
        "具有": REL_PROPERTY,    # 领属 → has-property（STEP5 PR3·closed-class 核心·开放变体 拥有 走 D:11 教师晋升·possess un-defer）
        "像": REL_SIMILAR,       # 相似 → ~（STEP5 PR4·closed-class 核心·开放变体 相似于 走 D:11 教师晋升·EDGE_SIMILAR slot-filler）
    },
    LANG_EN: {
        "is": REL_SUBSET,
        "belongs": REL_MEMBER,
        "causes": REL_CAUSES,
        "before": REL_PRECEDES,
        "part": REL_MEREOLOGY,
        "equals": REL_EQUAL,     # 相等系词 → =（STEP5·closed-class 核心）
        "has": REL_PROPERTY,     # 领属 → has-property（STEP5 PR3·closed-class 核心·possess un-defer）
        "resembles": REL_SIMILAR,  # 相似 → ~（STEP5 PR4·closed-class 核心·EDGE_SIMILAR slot-filler）
    },
}


def record_word_concept(concept_index, edge_store: EdgeStore,
                        word_surface: str, rel_ref: tuple[int, int], *,
                        space_id: int, source: int = SOURCE_TEACHER,
                        epistemic: int = EPI_STRUCTURED) -> int:
    """建一条 D:11 EDGE_RELATION_SIGNAL 边（word concept → REL_* concept）。

    **Plan agent 修点2/3/4**：
    - epistemic 闭合 assert（镜像 is_a.py:54）：禁裸共现·须 EPI_STRUCTURED/EPI_CUE/EPI_LLM_CONFIRM。
    - 默认 source=SOURCE_TEACHER（教师元定义·非 ConceptNet·与 causes.py 同范式）。caller（未来
      ConceptNet pluggable loader）可覆写 SOURCE_CONCEPTNET。
    - 入口防御短路（镜像 bootstrap_is_a_edges:119）：word_surface 空/rel_ref None → return 0。

    flow：ensure word concept（NODE_CONCEPT）→ query_from 幂等 skip（同 (word,rel_ref,D:11,source)
    已建则 skip·镜像 is_a.py:129-135）→ edge_store.add(edge_type=EDGE_RELATION_SIGNAL, from=word,
    to=rel_ref, strength=RELATION_SIGNAL_STRENGTH, source, epistemic_origin=epistemic, tier=PRIMARY)。
    返建边数。
    """
    # 修点4 防御短路（镜像 bootstrap_is_a_edges:119）
    if not word_surface or rel_ref is None or rel_ref == (0, 0):
        return 0
    # 修点2 epistemic 闭合（镜像 is_a.py:54）
    assert epistemic in (EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM), \
        "RELATION_SIGNAL 必须有认识论来源·禁裸共现"
    assert_int(space_id, source, _where="record_word_concept.args")
    word_ref = concept_index.ensure(word_surface, space_id=space_id,
                                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    # query_from 幂等 skip（按源细化·镜像 is_a.py:129-135）
    existing = edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    already = any(
        row.get("space_id_to") == rel_ref[0]
        and row.get("local_id_to") == rel_ref[1]
        and row.get("source") == source
        for row in existing
    )
    if already:
        return 0
    edge_store.add(
        space_id_from=word_ref[0], local_id_from=word_ref[1],
        space_id_to=rel_ref[0], local_id_to=rel_ref[1],
        edge_type=EDGE_RELATION_SIGNAL, strength=RELATION_SIGNAL_STRENGTH,
        source=source, epistemic_origin=epistemic, tier=TIER_PRIMARY,
    )
    # 清 D:11 lookup cache（建边改 D:11·cache 失效·防御·boot observe 前无 cache·刀4 gate ON 晋升清）
    _cache = getattr(edge_store, "_d11_lookup_cache", None)
    if _cache is not None:
        _cache.clear()
    return 1


def lookup_word_concept(backend: StorageBackend, edge_store: EdgeStore,
                        word_ref: tuple[int, int], *, space_id: int,
                        tier_filter: int | None = None,
                        ) -> list[tuple[tuple[int, int], int]]:
    """读 word_ref 的 D:11 边 → [(rel_ref, rel_kind), ...]（round-trip read API·刀3 无生产 caller·刀4/5 消费）。

    query_from(word_ref, D:11) → 每 target 读 read_composes_attrs 得 ATTR_RELATION_PRIMITIVE int_a=rel_kind。

    **tier_filter**（刀4 决断5 加·反 theater）：传 TIER_PRIMARY 只返 PRIMARY 边（已验证晋升）·
    None（默认）返全 tier（含 SHADOW·刀3 round-trip 测用·bit-identical）。反 theater 判据：
    未验证 SHADOW 不注入 cue_type_of readback·caller（_cue_type_from_d11_primary）传 TIER_PRIMARY。
    """
    if word_ref is None:
        return []
    # run-scoped cache（D:11 PRIMARY 边 + ATTR_RELATION_PRIMITIVE run 内不变·bit-identical·省 query_from +
    # read_composes_attrs 热路径·cProfile n=30 lookup 46k calls 占 _do_select ~43%）。cache 挂 edge_store 实例
    # （run-scoped 随 edge_store 生灭·record_word_concept 建边后清防御 D:11 变化）。tier_filter 入 key。
    # D:11 只 boot 种（record_word_concept·observe 不建·刀4 emergent gate OFF）-> run 内 PRIMARY 不变 -> cache safe。
    cache = getattr(edge_store, "_d11_lookup_cache", None)
    if cache is None:
        cache = {}
        try:
            edge_store._d11_lookup_cache = cache
        except (AttributeError, TypeError):
            cache = None  # edge_store 不可设属性（代理/__slots__）-> 不 cache（bit-identical·退化重算）
    key = (word_ref, tier_filter)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return list(cached)  # copy（元素 (rel_ref,kind) 不可变·list copy 防 caller mutate 污染 cache）
    rows = edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    out: list[tuple[tuple[int, int], int]] = []
    for r in rows:
        if tier_filter is not None and r.get("tier") != tier_filter:
            continue   # tier 过滤（刀4 决断5·反 theater：未验证 SHADOW 不注入 readback）
        rel_ref = (r["space_id_to"], r["local_id_to"])
        attrs = read_composes_attrs(backend, rel_ref)
        kind = attrs.get(ATTR_RELATION_PRIMITIVE, (0, 0))[0]
        if kind == 0:
            # 防御（对抗审1 P2-1·提前到刀3）：D:11 target 无 ATTR_RELATION_PRIMITIVE 标记（脏边/手动建）
            # → kind=0 非合法 REL_*（enum 1-8）·skip 不返。生产 D:11 唯一产消者 ensure_relation_primitives
            # 全标·此分支仅防御未来脏边·刀4 caller 免判 kind!=0。
            continue
        out.append((rel_ref, kind))
    if cache is not None:
        cache[key] = tuple(out)  # 不可变 tuple 存（防返值 mutate 污染 cache）
    return out


def bootstrap_word_concept_signals(concept_index, edge_store: EdgeStore,
                                   backend: StorageBackend, *,
                                   space_id: int, langs: set[int]) -> int:
    """按旧关系字面表写入迁移期 D:11 兼容种子。

    ① ensure_relation_primitives(concept_index, backend, space_id) 建 REL_* NODE_CONCEPT + 标记 → rel_refs map
    ② 按 langs 集（corpus 唯一 lang）循环 _REL_LEXICAL_CUE → record_word_concept 种 D:11 边。
    返建边总数。

    正式训练仅在 compatibility 开启时调用；本入口不能计 U-04 readiness。
    backend 显式传（ensure_relation_primitives 的 record_composes_attr 需 backend）。
    """
    assert_int(space_id, _where="bootstrap_word_concept_signals.space_id")
    rel_refs = ensure_relation_primitives(concept_index, backend, space_id=space_id)
    n = 0
    for lang in langs:
        cues = _REL_LEXICAL_CUE.get(lang, {})
        for word, rel_kind in cues.items():
            rel_ref = rel_refs.get(rel_kind)
            if rel_ref is None:
                continue
            n += record_word_concept(concept_index, edge_store, word, rel_ref,
                                     space_id=space_id)
    return n


def bootstrap_operator_signals(concept_index, edge_store: EdgeStore,
                               backend: StorageBackend, *,
                               space_id: int, langs: set[int]) -> int:
    """formal_train boot 种子入口（STEP5 PR2·镜像 bootstrap_word_concept_signals·operator D:11 种子）。

    ① ensure_operator_primitives(concept_index, backend, space_id) 建 OP_* NODE_CONCEPT + ATTR_OPERATOR_PRIMITIVE
       标记 → op_refs map
    ② 按 langs 集（corpus 唯一 lang）循环 _OP_LEXICAL_CUE → record_word_concept 种 D:11 边（word→OP_* ref）。
    返建边总数。

    **复用 record_word_concept**（D:11 边 word→any ConceptRef·source=SOURCE_TEACHER·epistemic=EPI_STRUCTURED·
    tier=PRIMARY·幂等）。OP_* target 挂 ATTR_OPERATOR_PRIMITIVE=18·与 REL_* target（ATTR_RELATION_PRIMITIVE=10）
    隔离（lookup_word_concept 过滤 REL_PRIMITIVE·lookup_word_operator 过滤 OP_PRIMITIVE·kind==0 skip·无交叉污染）。

    U-04 后本入口只保留迁移兼容数据。正式算术和比较 reader 优先读取来源化语言
    信号图，并由调用方注入指令到 opcode 的绑定；兼容关闭后不读取这些 D:11 关联。

    位于 word_concept_signal（L4）·import operator_primitives（L0）向下合规（L0 不能 import L4 record_word_concept）。
    """
    assert_int(space_id, _where="bootstrap_operator_signals.space_id")
    from pure_integer_ai.cognition.shared.operator_primitives import (
        ensure_operator_primitives, _OP_LEXICAL_CUE,
    )
    op_refs = ensure_operator_primitives(concept_index, backend, space_id=space_id)
    n = 0
    for lang in langs:
        cues = _OP_LEXICAL_CUE.get(lang, {})
        for word, op_kind in cues.items():
            op_ref = op_refs.get(op_kind)
            if op_ref is None:
                continue
            n += record_word_concept(concept_index, edge_store, word, op_ref,
                                     space_id=space_id)
    return n


def bootstrap_modal_signals(concept_index, edge_store: EdgeStore,
                            backend: StorageBackend, *,
                            space_id: int, langs: set[int]) -> int:
    """formal_train boot 种子入口（审计根治 [严重-1]·镜像 bootstrap_operator_signals·modal D:11 种子）。

    ① ensure_modal_primitives(concept_index, backend, space_id) 建 MODAL_KIND_* NODE_CONCEPT + ATTR_MODAL_KIND=22
       readback 标记 + abstract_mark MARK_MODAL_KIND=5 D6 归属 → modal_refs map
    ② 按 langs 集（corpus 唯一 lang）循环 _MODAL_LEXICAL_CUE → record_word_concept 种 D:11 边（word→MODAL_KIND ref）。
    返建边总数。

    **复用 record_word_concept**（D:11 边 word→any ConceptRef·source=SOURCE_TEACHER·epistemic=EPI_STRUCTURED·
    tier=PRIMARY·幂等）。MODAL_KIND target 挂 ATTR_MODAL_KIND=22·与 REL_* target（ATTR_RELATION_PRIMITIVE=10）/
    OP_* target（ATTR_OPERATOR_PRIMITIVE=18）隔离（lookup_word_concept/operator/modality 各过滤·kind==0 skip·
    无交叉污染）。

    U-04 后本入口只保留迁移兼容数据：_MODAL_LEXICAL_CUE 用于旧 D:11 boot，
    不再是 readiness 真源。正式情态 reader 优先读取来源化语言信号图，并由调用方
    注入指令到 modality 值的绑定；兼容关闭后不读取本入口产生的 D:11 关联。

    位于 word_concept_signal（L4）·import modal_primitives（L0）向下合规（L0 不能 import L4 record_word_concept）。
    """
    assert_int(space_id, _where="bootstrap_modal_signals.space_id")
    from pure_integer_ai.cognition.shared.modal_primitives import (
        ensure_modal_primitives, _MODAL_LEXICAL_CUE,
    )
    modal_refs = ensure_modal_primitives(concept_index, backend, space_id=space_id)
    n = 0
    for lang in langs:
        cues = _MODAL_LEXICAL_CUE.get(lang, {})
        for word, modal_kind in cues.items():
            modal_ref = modal_refs.get(modal_kind)
            if modal_ref is None:
                continue
            n += record_word_concept(concept_index, edge_store, word, modal_ref,
                                     space_id=space_id)
    return n


def bootstrap_negation_signals(concept_index, edge_store: EdgeStore,
                               backend: StorageBackend, *,
                               space_id: int, langs: set[int]) -> int:
    """formal_train boot 种子入口（#940 否定词 D:11 readback·镜像 bootstrap_modal_signals·negation D:11 种子）。

    ① ensure_symbol_types(concept_index, backend, space_id) 建 TYPE_NEGATION NODE_CONCEPT + ATTR_SYMBOL_TYPE=17
       标记 → neg_ref（{TYPE_NEGATION: ref}·单一·否定无种类）
    ② 按 langs 集（corpus 唯一 lang）循环 _NEGATION_LEXICAL_CUE → record_word_concept 种 D:11 边（word→__TYPE_NEGATION__ ref）。
    返建边总数。

    **复用 record_word_concept**（D:11 边 word→any ConceptRef·source=SOURCE_TEACHER·epistemic=EPI_STRUCTURED·
    tier=PRIMARY·幂等）。TYPE_NEGATION target 挂 ATTR_SYMBOL_TYPE=17·与 REL_*（ATTR_RELATION_PRIMITIVE=10）/
    OP_*（ATTR_OPERATOR_PRIMITIVE=18）/MODAL_KIND（ATTR_MODAL_KIND=22）隔离（各 lookup 过滤·无交叉污染）。

    U-04 后本入口只保留迁移兼容数据：_NEGATION_LEXICAL_CUE 用于旧 D:11 boot，
    不再是 readiness 真源。正式否定 reader 优先读取来源化语言信号图；兼容关闭后
    不读取本入口产生的 D:11 关联。

    **否定=符号域先天**（同 operator·异 modal）：¬ 概念先天冻结（TYPE_NEGATION）·D:11 readback 意义=否定词
    文字 alias 可学习·非概念可学。只挂 ATTR_SYMBOL_TYPE·不挂 abstract_mark（operator 范式·非 modal 双挂）。

    位于 word_concept_signal（L4）·import symbol_types（L0）向下合规。
    """
    assert_int(space_id, _where="bootstrap_negation_signals.space_id")
    from pure_integer_ai.cognition.shared.symbol_types import (
        ensure_symbol_types, _NEGATION_LEXICAL_CUE, TYPE_NEGATION,
    )
    type_refs = ensure_symbol_types(concept_index, backend, space_id=space_id)
    neg_ref = type_refs.get(TYPE_NEGATION)
    if neg_ref is None:
        return 0
    n = 0
    for lang in langs:
        cues = _NEGATION_LEXICAL_CUE.get(lang, frozenset())
        for word in cues:
            n += record_word_concept(concept_index, edge_store, word, neg_ref,
                                     space_id=space_id)
    return n


def bootstrap_action_signals(concept_index, edge_store: EdgeStore,
                             backend: StorageBackend, *,
                             space_id: int, langs: set[int]) -> int:
    """按旧动作字面表写入迁移期 D:11 兼容种子。

    ① ensure_action_primitives(concept_index, backend, space_id) 建 5 ACTION_INTENT_* NODE_CONCEPT +
       ATTR_OPERATION_INTENT=23 旗标 → action_refs map
    ② 按 langs 集（corpus 唯一 lang）循环 _ACTION_LEXICAL_CUE → record_word_concept 种 D:11 边（word→ACTION_INTENT_* ref）。
    返建边总数。

    **复用 record_word_concept**（D:11 边 word→any ConceptRef·source=SOURCE_TEACHER·epistemic=EPI_STRUCTURED·
    tier=PRIMARY·幂等）。动作意图 target 挂 ATTR_OPERATION_INTENT=23·与 REL_*（ATTR_RELATION_PRIMITIVE）/
    OP_*（ATTR_OPERATOR_PRIMITIVE）/MODAL_KIND（ATTR_MODAL_KIND）/TYPE_NEGATION（ATTR_SYMBOL_TYPE）隔离
    （各 lookup 过滤·无交叉污染）。

    U-04 正式 reader 以来源化图为主；Python 检测表和本入口产生的 D:11 都只在
    compatibility 开启时回退，图冲突或未绑定不得被旧来源覆盖。

    **命令词 + 动作词同基建**（doc §16.4）：_ACTION_LEXICAL_CUE 含两类词——命令 mood 词（→INTENT_COMMAND_MOOD·
    帮我/请·W7 命令判定）+ 动作词（→ACTION_* 类别·B-PR1）。W7 命令判定 = 命令词 OR 动作词命中任一。

    位于 word_concept_signal（L4）·import action_primitives（L0）向下合规（L0 不能 import L4 record_word_concept）。
    """
    assert_int(space_id, _where="bootstrap_action_signals.space_id")
    from pure_integer_ai.cognition.shared.action_primitives import (
        ensure_action_primitives, _ACTION_LEXICAL_CUE,
    )
    action_refs = ensure_action_primitives(concept_index, backend, space_id=space_id)
    n = 0
    for lang in langs:
        cues = _ACTION_LEXICAL_CUE.get(lang, {})
        for word, action_kind in cues.items():
            action_ref = action_refs.get(action_kind)
            if action_ref is None:
                continue
            n += record_word_concept(concept_index, edge_store, word, action_ref,
                                     space_id=space_id)
    return n
