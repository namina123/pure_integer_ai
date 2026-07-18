"""cognition.understanding.emergent_relation_signal — 刀4 件2 涌现关系信号（子环1+2·初心核心）。

子环2（SHADOW 落边）：record_emergent_relation_signal_shadow — 涌现假设词→REL_* 落 D:11 SHADOW
  信号边（参照 COOCCURS 范本·epistemic_origin=None + tier=SHADOW + source=SOURCE_BARE_TEXT·
  绕 record_word_concept 的 epistemic 闭合 assert :89-90·三重隔离不进 A1/PR/reward）。

子环1（涌现假设生成器）：generate_emergent_hypotheses — PRECEDES 链 connector 定位（用户 2026-07-05 选）。
  对每非-cue 词 w（有 PRECEDES 前驱 a 与后继 b·COOCCURS(a,b)≥MIN ∧ 无 CAUSES(a,b)）→ 假设 w→REL_CAUSES。
  签名→REL_CAUSES 映射 = 元定义层立法（共现+前驱是因果指纹·§8.1c-bis §6 同源立法·非词义写死·
  §九铁律承认 enum 例外）。首版只涌 REL_CAUSES（SUBSET/MEREOLOGY 须行为后果信号 defer）。

**struct_ref 自然过滤**：struct_ref 的 PRECEDES 邻居跨段（build_inter_segment_precedes 末token→下段
struct_ref + build_struct_anchor struct_ref→首token）·COOCCURS 是段内（cooccurs.py 段内 i<j）·
故 struct_ref 的 (前驱, 后继) 对 COOCCURS=0 < MIN·不匹配签名·天然排除·无需显式过滤。

**§8.1c 合规**：涌现假设**不经共现直落关系边**（三死刑）·经 staging SHADOW→experience 验证→promote
合法通路（§8.1c-bis 第6节）。本模块只产 SHADOW 候选边（sign=0·不进 A1/PR/reward）·晋升由 promote 决（子环4）。

铁律：纯整数（ConceptRef/REL_*/threshold 全 int·assert_int 守）/ 确定性（sorted 迭代 bit-identical）/
  不写死（观察签名非词义规则·签名→REL_CAUSES 元定义 enum 例外·同 cue_words/OPCODE_*）/ §8.1c（staging
  非直落）/ 单向依赖（L4→L0 storage + L0 cognition/shared 全向下）。
诚实边界：构造性（reward 验证非真独立源=#479 defer）/ 首版只 REL_CAUSES（SUBSET/MEREOLOGY defer）/
  PRECEDES 链运行时重建 O(段长)·远期可建段 token 序留档表优化（C.1 defer）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.cognition.shared.edge_types import (
    EDGE_RELATION_SIGNAL, EDGE_PRECEDES, EDGE_COOCCURS, EDGE_CAUSES,
)
from pure_integer_ai.cognition.shared.hub_detect import compute_hub_set
from pure_integer_ai.cognition.shared.relation_primitives import REL_CAUSES, ensure_relation_primitives

# ---- 涌现假设签名阈值（oracle 标占位·§十一#4 能力达标∧依赖度低二维调） ----
EMERGENT_COOCCURS_MIN = 3      # (a,b) 段内共现频次 ≥ 3（防单次噪声·镜像 PROMOTE_FREQ_MIN 范式）
EMERGENT_CONNECTOR_MIN = 1     # w 的 connector 命中数 ≥ 1（至少一对 (a,b) 满足签名）

# D:11 SHADOW 信号边 strength（不接 reward·非学习对象初值·刀4 验证下晋升 PRIMARY 才接 reward）
EMERGENT_SIGNAL_STRENGTH = 1


def record_emergent_relation_signal_shadow(edge_store: EdgeStore,
                                           word_ref: tuple[int, int],
                                           rel_ref: tuple[int, int], *,
                                           space_id: int) -> int:
    """子环2：涌现假设→D:11 SHADOW 信号边（参照 COOCCURS 范本·绕 epistemic assert）。

    落边字段组合（三重隔离·镜像 cooccurs.py:54-60）：
      edge_type        = EDGE_RELATION_SIGNAL (D:11)
      strength         = EMERGENT_SIGNAL_STRENGTH (=1·不接 reward·非学习对象初值)
      source           = SOURCE_BARE_TEXT (涌现假设来自裸文本共现+前驱·非教师元定义)
      epistemic_origin = None（伴随检疫·无认识论来源·禁 reward·绕 record_word_concept:89-90 assert）
      tier             = TIER_SHADOW（结构边非 PR 邻接头 / 不进 PR
                                     effective_weight:82 assert / _reward_ok 读 sn/tn=0/0→False）

    word_ref 取 ConceptRef（word 概念已 observe 存在·generator 从 PRECEDES 边得之·非 surface·
      免 redundant ensure）。rel_ref = REL_* 框架节点 ConceptRef（刀3 ensure_relation_primitives 已建）。

    返建边数（0=skip·1=建）。**幂等**：query_from 按 (word, rel_ref, D:11, source=BARE_TEXT) 去重
    （镜像 record_word_concept:94-103 范式·同 source 才 skip·不挡 SOURCE_TEACHER 教师种子异源并存）。
    """
    # 防御短路（镜像 record_word_concept:86）
    if word_ref is None or word_ref == (0, 0) or rel_ref is None or rel_ref == (0, 0):
        return 0
    assert_int(space_id, word_ref[0], word_ref[1], rel_ref[0], rel_ref[1],
               _where="record_emergent_relation_signal_shadow.args")
    # query_from 幂等 skip（按 source 细化·镜像 is_a.py:129-135 / record_word_concept:94-103）
    existing = edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    already = any(
        row.get("space_id_to") == rel_ref[0]
        and row.get("local_id_to") == rel_ref[1]
        and row.get("source") == SOURCE_BARE_TEXT
        for row in existing
    )
    if already:
        return 0
    edge_store.add(
        space_id_from=word_ref[0], local_id_from=word_ref[1],
        space_id_to=rel_ref[0], local_id_to=rel_ref[1],
        edge_type=EDGE_RELATION_SIGNAL, strength=EMERGENT_SIGNAL_STRENGTH,
        source=SOURCE_BARE_TEXT, epistemic_origin=None, tier=TIER_SHADOW,
    )
    return 1


def generate_emergent_hypotheses(backend: StorageBackend, edge_store: EdgeStore,
                                 concept_index, *, space_id: int,
                                 excluded_word_refs: set[tuple[int, int]],
                                 rel_kind: int = REL_CAUSES,
                                 ) -> list[tuple[tuple[int, int], int, tuple[int, int]]]:
    """子环1：涌现假设生成器（PRECEDES 链 connector 定位·用户 2026-07-05 选·运行时重建·无新表）。

    返 [(word_ref, rel_kind, rel_ref), ...]·caller 调 record_emergent_relation_signal_shadow 落 SHADOW 边。

    签名（每候选词 w·纯整可观察·全既有表可读）：
      w 有 PRECEDES 前驱集 A（query_to(w, PRECEDES)）∧ PRECEDES 后继集 B（query_from(w, PRECEDES)）
      对每 (a,b) ∈ A×B：COOCCURS(a,b) count ≥ EMERGENT_COOCCURS_MIN ∧ 无 CAUSES(a,b) → w 是 connector·hits++
      w 的 hits ≥ EMERGENT_CONNECTOR_MIN → 假设 w→rel_kind（首版 REL_CAUSES）

    **struct_ref 自然过滤**：struct_ref 的 PRECEDES 邻居跨段·COOCCURS(a,b)=0 < MIN·不匹配·天然排除。
    **excluded_word_refs** = `_CUE_WORDS ∪ _REL_LEXICAL_CUE` 已种词 ConceptRefs（caller resolve surfaces→refs·
      C9-bis §D 候选池排除清单·防 reward 调固化件）。REL_* 框架节点本身也排除（不应作 connector·防自指）。
      已有 D:11 边的词排除（已种/已涌现·不重复假设）。

    **确定性**：candidates sorted by ConceptRef·(a,b) sorted·bit-identical（同输入同输出·CI===生产）。
    **签名→REL_CAUSES 映射** = 元定义层立法（共现+前驱是因果指纹·§8.1c-bis §6·非词义写死）。
    """
    assert_int(space_id, _where="generate_emergent_hypotheses.space_id")
    # ensure REL_* primitives·取 rel_ref（idempotent·boot 已建·防御）
    rel_refs = ensure_relation_primitives(concept_index, backend, space_id=space_id)
    rel_ref = rel_refs.get(rel_kind)
    if rel_ref is None:
        return []   # 该 REL_* 无框架节点（不应发生·ensure 建 8 个）·防御

    # 1. 扫 PRECEDES 边建 pred/succ maps（本 space·intra-space 过滤·防跨 space struct_ref 干扰）
    preds: dict[tuple[int, int], set[tuple[int, int]]] = {}
    succs: dict[tuple[int, int], set[tuple[int, int]]] = {}
    try:
        rows = backend.select("edge", where={"edge_type": EDGE_PRECEDES})
    except KeyError:
        rows = []   # 表未注册（bare fixture）·向后兼容
    for r in rows:
        if r["space_id_from"] != space_id or r["space_id_to"] != space_id:
            continue   # 仅本 space
        a = (r["space_id_from"], r["local_id_from"])
        b = (r["space_id_to"], r["local_id_to"])
        succs.setdefault(a, set()).add(b)
        preds.setdefault(b, set()).add(a)

    # 2. 候选 connector：有前驱 ∧ 有后继·排除 cue 词 + REL_* 框架节点
    excluded = set(excluded_word_refs) | set(rel_refs.values())
    candidates = (set(preds.keys()) & set(succs.keys())) - excluded

    # 归一化半 A：hub 预计算（caller 侧·gate 守·决断 A3 落点精化）。
    # **perf**（2026-07-08 训练测试实测·两轮修复）：(1) _cooccurs_count 内 per-pair is_hub → caller 预计算；
    # (2) per-ref is_hub 每 ref 2 全表扫 → compute_hub_set **单遍** O(#COOCCURS) 建 degree map。
    # n=10 从 385s（per-ref）根治。循环内 O(1) 查表跳过 hub 对·_cooccurs_count 复原纯计数。
    # **语义同**（hub 对不计 hits 不达 EMERGENT_COOCCURS_MIN）。
    exclude_func = getattr(gates, "EXCLUDE_FUNCTION_MODE", False)
    hub_set = compute_hub_set(edge_store) if exclude_func else set()

    # 3. 对每候选 w 查 connector 签名（确定性 sorted 迭代）
    out: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for w in sorted(candidates):
        # 排除已有 D:11 边的词（已种 PRIMARY 教师种子 / 已涌现 SHADOW·不重复假设）
        if edge_store.query_from(w[0], w[1], edge_type=EDGE_RELATION_SIGNAL):
            continue
        a_set = preds[w]
        b_set = succs[w]
        hits = 0
        for a in sorted(a_set):
            if a in hub_set:
                continue   # hub a 不计 hits（caller 侧排除·gate OFF→hub_set 空→不 skip bit-identical）
            for b in sorted(b_set):
                if a == b or a == w or b == w:
                    continue
                if b in hub_set:
                    continue   # hub b 不计 hits
                if _has_causes(edge_store, a, b):
                    continue   # 已确认 CAUSES(a,b)·非涌现候选
                if _cooccurs_count(edge_store, a, b) >= EMERGENT_COOCCURS_MIN:
                    hits += 1
        if hits >= EMERGENT_CONNECTOR_MIN:
            out.append((w, rel_kind, rel_ref))
    return out


def _cooccurs_count(edge_store: EdgeStore,
                    a: tuple[int, int], b: tuple[int, int]) -> int:
    """a-b 共现频次（双向·读 strength 累加·总收口 0.1 COOCCURS A'·纯整数）。

    COOCCURS i<j 跨段两方向都可能（a→b / b→a）·双向 query_from 累加 strength。读 strength 非 row count：
    gate OFF（旧 add strength 恒 1）累加=数行=旧语义 bit-identical；gate ON（add_cooccurs_dedup·strength=频次）
    累加=真共现频次。与 hub_degree/compute_hub_set 协同（同读 strength）。

    hub 过滤由 caller generate_emergent_hypotheses 预计算 hub_set 跳过（perf·O(unique refs) 一次性
    vs 每对 fresh 4 query·2026-07-08 训练测试实测 3-7× 灾难修复·归一化半 A 决断 A3 落点精化）·
    语义同：hub 对不计 hits·不达 EMERGENT_COOCCURS_MIN·不伪产 REL_CAUSES 喂 reward。
    """
    n = 0
    for r in edge_store.query_from(a[0], a[1], edge_type=EDGE_COOCCURS):
        if (r["space_id_to"], r["local_id_to"]) == b:
            n += r["strength"]   # 总收口 0.1：读 strength（gate OFF 恒 1·等价数行；gate ON 频次）
    for r in edge_store.query_from(b[0], b[1], edge_type=EDGE_COOCCURS):
        if (r["space_id_to"], r["local_id_to"]) == a:
            n += r["strength"]
    assert_int(n, _where="_cooccurs_count.strength_sum")   # 总收口 0.1：读 strength 累加·守纯整数
    return n


def _has_causes(edge_store: EdgeStore,
                a: tuple[int, int], b: tuple[int, int]) -> bool:
    """已确认 CAUSES(a,b)（方向性 a→b·query_from(a, CAUSES) 命中 b）。"""
    for r in edge_store.query_from(a[0], a[1], edge_type=EDGE_CAUSES):
        if (r["space_id_to"], r["local_id_to"]) == b:
            return True
    return False
