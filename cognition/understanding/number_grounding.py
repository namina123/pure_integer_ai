"""cognition.understanding.number_grounding — 数字词接地 boot 种（language-grounding piece 1·语言→算数 桥）。

bootstrap_number_grounding(concept_index, edge_store, backend, facts, *, space_id) -> int
  boot 时批量种数字词↔整数概念 PURE_ALIAS 图边 + 整数概念 CORR_NUMERIC 值（来源：number_facts.txt 外部数据·EPI_STRUCTURED）。
  每 (surface, lang, value)：ensure 整数概念 __int_{value}（NODE_CONCEPT·跨词/lang dedup）+ record_numeric(value)
  + ensure 词 NODE_WORD + set_mark MARK_LANG + 双向 PURE_ALIAS（词↔整数概念·**关联在图中**）。

**用户铁律（2026-07-16）**：① 不能写死（数据来自 number_facts.txt 外部文件·非代码字典）② 关联在图中
（词↔整数 是 PURE_ALIAS 图边·可遍历可见·非旁侧表）③ 例外测试种子（number_facts 是冷启动种子数据）。
直击"语言嵌入算数"（doc/重来_语言通用接地 §七）·语言域主攻首刀。

**关联在图中（命门）**：词"三" —PURE_ALIAS 边→ 整数概念 `__int_3`（图边 = 关联·镜像 apple↔苹果）。
整数概念挂 CORR_NUMERIC=3（值·属性·同码点 correspondence 范式·非关联）。两分离：关联=图边·值=概念属性。
下游 language→arith：词 token → PURE_ALIAS → 整数概念 → CORR_NUMERIC → arith IMM operand（vm_proof 验）。
**复用既有解决（closure-falsified）**：从语料学机制（cue_extractor + 控制环）已 solved·本模块只补数字接地这一片。

**为何整数是概念节点（非词）**：整数是抽象数（非某语言词形）·三/three/3 都是它的 surface。`__int_{value}`
NODE_CONCEPT 跨词/lang dedup（三和 three 共指 __int_3）·PURE_ALIAS 把词形挂到抽象整数（同 apple/苹果 共指
等价类·但此处显式 integer concept 节点携值·因整数需携 numeric value 供 arith 消费）。

**bit-identical（gate OFF·CI）**：无 number_facts.txt（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve_number_facts
返 [] → bootstrap 空 facts 首行短路 return 0·**绝不调 ensure/record_numeric/set_mark/build_refers_stable_edge**
（镜像 bootstrap_alias_edges:94-95）。零新节点/边/MARK_LANG/CORR_NUMERIC·核心空间零整数概念 → bit-identical。

铁律：纯整数（local_id/lang/value/mark_kind 全 int·assert_int 守）/ bit-identical（空 facts 零副作用 +
query_from 幂等防 resume 重复边）/ 不写死（surface/lang/value 来自外部文件·本模块只机制非语义）/ 单向依赖
（L4 understanding·import storage L0 + cognition L4·不环）/ 关联在图中（PURE_ALIAS 图边非旁侧表）。
诚实边界：数字词真伪（三↔3 对错）= 外部数据责任（接地墙·#479 教师定义权）·系统不判·只落图边 + 值·
piece 1=数据接地（table 种子）·可学接地（从例证学 三↔3）= piece 3 defer（#479 墙内统计）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore, EPI_STRUCTURED, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT, NODE_WORD
from pure_integer_ai.storage.abstract_mark import set_mark, MARK_LANG
from pure_integer_ai.storage.concept_correspondence import record_numeric, load_numeric
from pure_integer_ai.cognition.shared.edge_types import EDGE_REFERS_TO
from pure_integer_ai.cognition.understanding.refers_stable import build_refers_stable_edge


def _pure_alias_exists(edge_store: EdgeStore,
                       a: tuple[int, int], b: tuple[int, int]) -> bool:
    """查 a→b PURE_ALIAS 边是否已建（幂等 skip·防 resume 重复边·镜像 alias_bridge._pure_alias_exists）。"""
    existing = edge_store.query_from(a[0], a[1], edge_type=EDGE_REFERS_TO)
    return any(
        row.get("space_id_to") == b[0]
        and row.get("local_id_to") == b[1]
        and row.get("subtype") == SUBTYPE_PURE_ALIAS
        for row in existing
    )


def bootstrap_number_grounding(concept_index, edge_store: EdgeStore, backend,
                               facts: list[tuple[str, int, int]],
                               *, space_id: int) -> int:
    """数字词接地批量 boot 种图边 + 值（piece 1·facts=(surface,lang,value) → 整数概念+值+词↔整数 PURE_ALIAS 图边）。

    入参 facts：(surface, lang, value) 列表·caller 不依赖语料 token 切片（boot 时种·早于 observe）·
      来自 number_facts 文件 loader（来源① EPI_STRUCTURED 合规·类比 alias_facts）。

    **无文件零副作用硬守（bit-identical·P0）**：facts 空 → 立即 return 0·**绝不调 concept_index.ensure /
      record_numeric / set_mark / build_refers_stable_edge / query_from**（无 ZERO_AI_LOCAL_DIR → resolve 返 []
      → 图与不接 number_grounding bit-identical）。

    每 (surf, lang, value)：
      - ensure 整数概念 `__int_{value}`（TIER_PRIMARY·NODE_CONCEPT·跨词/lang dedup·content_hash(surface) 幂等）。
      - record_numeric(整数概念, value)（CORR_NUMERIC·整数概念的值·APPEND_ONLY 幂等·bare fixture 表未注册 skip）。
      - ensure 词 `surf`（TIER_PRIMARY·NODE_WORD·性质A 稳定定义同指·同 alias 词形）。
      - set_mark MARK_LANG 词 lang（幂等·dispatch_slot target_lang 偏好读）。
      - build_refers_stable_edge 双向（词→整数概念 + 整数概念→词·PURE_ALIAS·对称 activate_candidates·
        **关联在图中**·每方向 query_from 幂等 skip 防 resume 重复边）。

    返建边数（每 fact 最多 2·词==整数概念 ref 时 build_refers_stable_edge:59 自环 guard 返 0·理论不触发）。

    铁律：纯整数（ConceptRef + lang + value 全整·零浮点）/ 不写死（surface/lang/value 来自外部文件·本函数只机制非语义）/
      §8.1c（来源① EPI_STRUCTURED 合规）/ bit-identical（空 facts 零副作用 + query_from 幂等）/ 关联在图中（PURE_ALIAS 图边）。
    诚实边界：surface↔value 真伪 = 外部数据责任（接地墙·#479）·PURE_ALIAS 不接 reward（effective_weight:82 assert
      只认 PRECEDES/CAUSES/REFERS_TO·PURE_ALIAS return 0·boot 边无 reward 误判风险）·piece 1=数据接地非可学（defer #479）。
    """
    if not facts:
        return 0   # P0·无文件零副作用硬守（不调 ensure/record_numeric/set_mark/build/query·CI/生产 default bit-identical）
    assert_int(space_id, _where="bootstrap_number_grounding.args")
    n = 0
    for surf, lang, value in facts:
        assert_int(lang, value, _where="bootstrap_number_grounding.fact")
        # 整数概念（抽象数·跨词/lang dedup·__int_{value}·携 CORR_NUMERIC 值）
        int_ref = concept_index.ensure(
            f"__int_{value}", space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        record_numeric(backend, space_id=int_ref[0], local_id=int_ref[1], value=value)
        # 词形（surface·NODE_WORD·挂 MARK_LANG）
        word_ref = concept_index.ensure(
            surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_WORD)
        set_mark(backend, ref=word_ref, mark_kind=MARK_LANG, mark_value=lang)
        # 双向 PURE_ALIAS（词↔整数概念·关联在图中·对称 activate_candidates·每方向 query_from 幂等 skip）
        if not _pure_alias_exists(edge_store, word_ref, int_ref):
            n += build_refers_stable_edge(edge_store, concept_index, word_ref, int_ref,
                                          epistemic=EPI_STRUCTURED, space_id=space_id)
        if not _pure_alias_exists(edge_store, int_ref, word_ref):
            n += build_refers_stable_edge(edge_store, concept_index, int_ref, word_ref,
                                          epistemic=EPI_STRUCTURED, space_id=space_id)
    return n


def resolve_number_word(concept_index, edge_store: EdgeStore, backend,
                        token: str, *, space_id: int) -> int | None:
    """数字词 → 接地整数（读侧·word-problem 解析用·**走图**：token→PURE_ALIAS→整数概念→CORR_NUMERIC）。

    flow：concept_index.lookup(token) → word_ref | None（词未概念化→None·冷启动）
      → edge_store.query_from(word_ref, EDGE_REFERS_TO) → PURE_ALIAS 目标 → load_numeric(目标) → 首非 None int。
    无 PURE_ALIAS 目标 / 目标无 CORR_NUMERIC → None（该词非数字词·word_problem 据 None 跳·守反统计契约）。

    **关联在图中（命门·读侧）**：数字值经 **图遍历**（PURE_ALIAS 边）取得·非旁侧查询表 lookup·
    与 bootstrap 写侧对称（写建图边·读走图边）。冷启动（未 bootstrap·无图边）→ None。
    纯读（concept_index.lookup + edge_store.query_from + load_numeric 均 read·无写）。
    """
    word_ref = concept_index.lookup(token, space_id)
    if word_ref is None:
        return None   # 词未概念化（冷启动·未 observe/boot）·退化
    for row in edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_REFERS_TO):
        if row.get("subtype") != SUBTYPE_PURE_ALIAS:
            continue   # 非 PURE_ALIAS 边（代词 OCCURRENCE 等）·skip
        target = (row.get("space_id_to"), row.get("local_id_to"))
        value = load_numeric(backend, space_id=target[0], local_id=target[1])
        if value is not None:
            return value   # PURE_ALIAS 目标携 CORR_NUMERIC → 该词接地此整数
    return None   # 无数字接地（非数字词）
