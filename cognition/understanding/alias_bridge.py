"""cognition.understanding.alias_bridge — 模块4 性质A 跨语言 PURE_ALIAS 桥 boot 种（P0b·C 身份跨语言§3.1/§7.4）。

bootstrap_alias_edges(concept_index, edge_store, backend, pairs, *, space_id) -> int
  boot 时批量种跨语言/同义 PURE_ALIAS 边（来源① 结构化外部词典·Wikidata QID 翻译等价·EPI_STRUCTURED）。
  每 (surface_a, lang_a, surface_b, lang_b)：ensure 两 NODE_WORD + 各自 set_mark MARK_LANG + 双向
  build_refers_stable_edge PURE_ALIAS。

**身份模型 = Model A（词形↔词形双向 PURE_ALIAS·铁律合规）**：
  apple/苹果 各自是 NODE_WORD（各携自己码点 correspondence（P0a hook）+ MARK_LANG）·两者间双向 PURE_ALIAS 边
  → REFERS_TO 等价类合一（"同节点"=语义等价类非存储同 local_id·§7.4 L212·铁律"永不合并节点(同指靠 REFERS_TO 边)"
  + struct_bind.py:16 + #2-bis L699/L725 apple↔苹果 列 REFERS_TO 性质A）。**非 Model C**（同 local_id·违铁律·不取）。

**为何双向**：activate_candidates（graph_view.py:170·REFERS_TO 反向 query_to 取 from 端）只返指向 ref 的边之 from。
  单向 a→b 时 activate_candidates(b) 得 {a}（b 自身 excluded·PURE_ALIAS-gated 自包含 fix 补回）·activate_candidates(a)
  得 {a}（无 a→?  反向命中）→ a 看不到 b。双向（a→b + b→a）→ 两端 activate_candidates 都得 {self(via fix), 对方}·
  对称·dispatch_slot 按 target_lang + MARK_LANG 选对词形。

**PURE_ALIAS-gated 自包含 fix（graph_view.activate_candidates·P0b-4）**：有候选时 activate_candidates 默认不含 self。
  一旦 apple 有 PURE_ALIAS 边·activate_candidates(apple) 只返 {苹果}→dispatch_slot 永选不出 apple（EN 目标）。
  fix：rows 含 PURE_ALIAS 边时 cands.add(ref)（self 回补）。**bit-identical**：CI 零 PURE_ALIAS 边→any(...) 恒 False
  →退化 `if not cands` 现状（OCCURRENCE 代词边 subtype≠PURE_ALIAS 不触发）。

**生成侧全已就位**（P0b 只补桥·零改生成）：activate_candidates（concept→候选词集）+ dispatch_slot target_lang
  + lang_of MARK_LANG（按语言偏好）+ P0a correspondence（词形携码点）+ surface_of（词形→真字）。

**bit-identical（gate OFF·CI）**：无 alias_facts.txt（CI/生产 default 无 ZERO_AI_LOCAL_DIR）→ resolve_alias_facts 返 []
  → bootstrap 空 pairs 首行短路 return 0·**绝不调 ensure/set_mark/build_refers_stable_edge**（镜像 bootstrap_is_a_edges:119-120）。
  零新节点/边/MARK_LANG·核心空间零 PURE_ALIAS 边·activate_candidates fix 退化现状 → 生成/judge/reward 逐字不变。

**PURE_ALIAS 不动拓扑**（bit-identical crux）：effective_weight.py:73-85 REFERS_TO PURE_ALIAS return 0（不进 PR）·
  a3_pr_wrapper.py:168-172 PURE_ALIAS 不进 PR 邻接·dag_path 不遍历 REFERS_TO·observe/reward/judge/COOCCURS 不读
  PURE_ALIAS 边·transitive_closure 仅 EDGE_IS_A（REFERS_TO 闭包不活算）。故 PURE_ALIAS 边只影响 activate_candidates
  （生成侧候选）·gate ON 时只有 generate 输出按 lang 变（预期非 bit-identical·审计 L278"仅桥活时触发"）。

铁律：纯整数（local_id/lang/mark_kind 全 int·assert_int 守）/ bit-identical（空 pairs 零副作用 + query_from 幂等
  防 resume 重复边·EdgeStore.add append-only 不去重）/ 不写死（alias surface/lang 来自外部文件·本模块只桥非语义）/
  单向依赖（L4 understanding·import storage L0 + cognition L4·不环）/ 永不合并节点（Model A 边连接非同 local_id）。
诚实边界：alias 真伪（翻译对错）= 外部数据责任（接地墙·#479 教师定义权）·系统不判·只落 PURE_ALIAS 边·
  sense×alias 交叉 defer（sense 概念 ref 与词形 NODE_WORD ref 分离·pre-existing 特性非 P0b 回归）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore, EPI_STRUCTURED, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_WORD
from pure_integer_ai.storage.abstract_mark import set_mark, MARK_LANG
from pure_integer_ai.cognition.shared.edge_types import EDGE_REFERS_TO
from pure_integer_ai.cognition.understanding.refers_stable import build_refers_stable_edge


def _pure_alias_exists(edge_store: EdgeStore,
                       a: tuple[int, int], b: tuple[int, int]) -> bool:
    """查 a→b PURE_ALIAS 边是否已建（幂等 skip 用·防 resume 重复边）。

    EdgeStore.add append-only 不去重（edge_store.py:166）·boot 在 resume 路径重跑（load_run 已还原边）·
    无此查则重复 insert PURE_ALIAS 边（activate_candidates 用 set 去重不致错·但 dump 边冗余 + 非幂等）。
    镜像 bootstrap_is_a_edges:128-137 query_from 幂等范式（按 subtype 精细化·不挡 observe OCCURRENCE 路径）。
    """
    existing = edge_store.query_from(a[0], a[1], edge_type=EDGE_REFERS_TO)
    return any(
        row.get("space_id_to") == b[0]
        and row.get("local_id_to") == b[1]
        and row.get("subtype") == SUBTYPE_PURE_ALIAS
        for row in existing
    )


def bootstrap_alias_edges(concept_index, edge_store: EdgeStore, backend,
                          pairs: list[tuple[str, int, str, int]],
                          *, space_id: int) -> int:
    """跨语言/同义 PURE_ALIAS 批量 boot 种边（P0b·surface+lang 对 → ensure 词形 + MARK_LANG + 双向 PURE_ALIAS）。

    入参 pairs：(surface_a, lang_a, surface_b, lang_b) 列表·caller 不依赖语料 token 切片（boot 时种·早于 observe）·
      来自 alias_facts 文件 loader（来源① EPI_STRUCTURED 合规·类比刀0 IS_A loader）。

    **无文件零副作用硬守（bit-identical·P0）**：pairs 空 → 立即 return 0·**绝不调 concept_index.ensure / set_mark /
      build_refers_stable_edge / query_from**（无 ZERO_AI_LOCAL_DIR → resolve 返 [] → 图与不接 P0b bit-identical）。

    每 (surf_a, lang_a, surf_b, lang_b)：
      - concept_index.ensure 两 ref（TIER_PRIMARY·NODE_WORD·性质A 稳定定义同指·同 lemmatizer lemma 路径 refers_to.py:101·
        ensure dedup by content_hash 干净·dedup 命中不重写 correspondence·P0a hook 仅新建时写码点）。
      - set_mark MARK_LANG 各自 lang（幂等 query-then-upsert·同 status skip·abstract_mark.set_mark）。
      - build_refers_stable_edge 双向（a→b + b→a·PURE_ALIAS·epistemic=EPI_STRUCTURED·对称 activate_candidates）·
        每方向 query_from 幂等 skip（防 resume 重复边·_pure_alias_exists）。

    返建边数（每对最多 2·a==b ref 时 build_refers_stable_edge:59 自环 guard 返 0）。

    铁律：纯整数（ConceptRef + lang + mark_kind 全整·零浮点）/ 不写死（surface/lang 来自外部文件·本函数只机制非语义）/
      §8.1c（来源① EPI_STRUCTURED 合规·build_refers_stable_edge:57 assert 允许）/ bit-identical（空 pairs 零副作用 +
      query_from 幂等）/ 永不合并节点（Model A 边连接·ensure 两不同 ref 非同 local_id）。
    诚实边界：surface 真伪 = 外部数据责任（接地墙·#479）·PURE_ALIAS 不接 reward（effective_weight:82 assert
      只认 PRECEDES/CAUSES/REFERS_TO·PURE_ALIAS return 0·boot 边无 reward 误判风险）。
    """
    if not pairs:
        return 0   # P0·无文件零副作用硬守（不调 ensure/set_mark/build/query·CI/生产 default bit-identical）
    assert_int(space_id, _where="bootstrap_alias_edges.args")
    n = 0
    for surf_a, lang_a, surf_b, lang_b in pairs:
        ref_a = concept_index.ensure(
            surf_a, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_WORD)
        ref_b = concept_index.ensure(
            surf_b, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_WORD)
        # MARK_LANG 各自 lang（幂等·dispatch_slot target_lang 偏好读 lang_of → get_mark MARK_LANG）
        set_mark(backend, ref=ref_a, mark_kind=MARK_LANG, mark_value=lang_a)
        set_mark(backend, ref=ref_b, mark_kind=MARK_LANG, mark_value=lang_b)
        # 双向 PURE_ALIAS（a→b + b→a·对称 activate_candidates·每方向 query_from 幂等 skip 防 resume 重复边）
        if not _pure_alias_exists(edge_store, ref_a, ref_b):
            n += build_refers_stable_edge(edge_store, concept_index, ref_a, ref_b,
                                          epistemic=EPI_STRUCTURED, space_id=space_id)
        if not _pure_alias_exists(edge_store, ref_b, ref_a):
            n += build_refers_stable_edge(edge_store, concept_index, ref_b, ref_a,
                                          epistemic=EPI_STRUCTURED, space_id=space_id)
    return n
