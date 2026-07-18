"""cognition.shared.hub_detect — 功能词/hub 词 read-time 统计判据（归一化半 A·2026-07-08）。

hub_degree(word_ref) = word 的 COOCCURS 关联边总数（from + to 双向·read-time 数行·绝对计数）。
hub_degree ≥ THETA_HUB_DEGREE → is_hub True → 3 个 live 消费/污染点 read-time 排除（caller gate 守）：
  collide_score（slot_dispatch caller·卷三消歧主轴·解"分子是曾经"排序污染）
  _cooccurs_count（emergent_relation_signal·喂 REL_CAUSES→reward·解伪产因果污染）
  refers_occurrence（代词候选池·解"他"→"曾经"语义层污染）

**层=READ-time**（非 write-time·决断 A1）：保 COOCCURS 数据（hub_degree 判据本身读这些边·write-time
排除=自举死锁）+ 不碰 build_cooccurs/build_precedes write 路径（无 PRECEDES 序对齐风险）+ 与半 B 同层统一。

**判据措辞（诚实·审2 铁律 7·决断 A2）**：hub_degree 是统计 hub 度判据·语料共现结构涌现（corpus 自定·
非硬编码词表·守 §十五 C5 字面）·但**非 FUNCTION_CLASS(12) 学习型信号边+reward 真·涌现形式**·THETA oracle
写死（软阈值）·**过渡统计判据**·FUNCTION_CLASS 真·涌现 defer（决断 A5）。

**为什么 hub_degree 而非 effective_freq/permille（审1 Q2/Q4/Q7 一刀解）**：
  Q2 e_tn 时漂：effective_freq=base+e_tn 随 reward 漂→中途跨 θ 惩罚好词·hub_degree 读 COOCCURS 与 reward 无关。
  Q4 高频实词 hub 洞：freq/permille 只挡 closed-class（曾经/的）挡不住高频实词 hub（说/做）·hub_degree 直测 hub-ness 两者都挡。
  Q7 corpus_total crash：permille×1000//corpus_total 除零 crash·hub_degree 绝对计数无除法无 crash。

铁律：纯整数（计数全整）/ 确定性（query 确定性）/ 位一致（gate OFF 守 CI===生产 bit-identical）/
单向依赖（cognition/shared→storage 全向下·守 result→understanding finer：graph_view(result)+
emergent_relation_signal(understanding)+refers_occurrence(understanding) 三调 shared 公共向下层）/
不纸面闭合（生产 try/finally 翻 ON）/ §8.5 不碰 schema（read-time 计算永不持久化）。
诚实边界：stable≠correct（hub 作 collide ctx 是非判别性噪声·移除增判别力·但小语料 θ 可能误排主题中心
实词）·过渡统计判据·不产生新语义能力（#479 墙）·共现排序清洗非语义消歧突破。
"""
from __future__ import annotations

import weakref

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.cognition.shared.edge_types import EDGE_COOCCURS

# θ oracle 标定起点（§十五 B 组初值·hub_degree = COOCCURS 关联边总数·≥ θ → hub）。
# 软阈值·**过渡统计判据**·非 reward 学得（决断 A2/A5·FUNCTION_CLASS 真·涌现 defer）。
# 小语料 hub_degree 间隔小可能误排主题中心实词 → stable≠correct·gate OFF 守 CI 回归零行为变。
# 标定逻辑：windowed COOCCURS（K=2）下单 occurrence 产 ≤2 边·hub 跨多段累行（曾经/的 类 closed-class +
# 说/做 类高频实词 hub 跨 N 段 → ≥θ）·content word 单段 ~2-4 边 < θ 不挡。oracle 验后调。
THETA_HUB_DEGREE = 8

# perf round3 dirty-flag 缓存（审2 建议 WeakKeyDictionary·2026-07-13）：compute_hub_set 结果集缓存留
# cognition 层·不污染 EdgeStore 实例（解跨层状态泄漏 + 未来 __slots__ 静默 AttributeError 脆性）·
# WeakKeyDictionary 与实例属性同等 GC 安全（EdgeStore GC -> 条目自动清·无 id 重用串值）。
# key=edge_store · value=(cooccurs_version, theta, hub_set)·失效即 fresh 重算·bit-identical。
_HUB_SET_CACHE = weakref.WeakKeyDictionary()


def hub_degree(word_ref: tuple[int, int], edge_store: EdgeStore) -> int:
    """word 的 COOCCURS 关联强度总和（from + to 双向·read-time 读 strength 累加·纯整数）。

    = Σ strength（word 作 from 的 COOCCURS 边）+ Σ strength（word 作 to 的 COOCCURS 边）。
    **读 strength 累加（总收口 0.1·COOCCURS A'）**：gate OFF（旧 add strength 恒 1）累加=数行=旧语义
    bit-identical；gate ON（add_cooccurs_dedup·strength=频次）累加=真共现频次。与 _cooccurs_count
    （双向累加）协同；collide_score 是集合语义不计重复（独立）。COOCCURS 无向·from/to 双向计无重复
    （from-rows 与 to-rows 是 word 作源/靶的不相交边集·无 double-count）。

    表未注册（bare fixture）→ KeyError 容错返 0（冷启动退化·bit-identical OFF·无 crash·审1 Q7）。
    """
    sid, lid = word_ref
    assert_int(sid, lid, _where="hub_degree.word_ref")
    try:
        n_from = sum(r["strength"] for r in edge_store.query_from(sid, lid, edge_type=EDGE_COOCCURS))
        n_to = sum(r["strength"] for r in edge_store.query_to(sid, lid, edge_type=EDGE_COOCCURS))
    except KeyError:
        return 0   # edge 表未注册（bare fixture）·向后兼容·冷启动退化·无 crash
    assert_int(n_from, n_to, _where="hub_degree.strength_sum")   # strength TYPE_INT·sum int·守纯整数
    return n_from + n_to


def is_hub(word_ref: tuple[int, int], edge_store: EdgeStore) -> bool:
    """word 是否 hub（hub_degree ≥ THETA_HUB_DEGREE·read-time 判据·gate 守由 caller）。

    冷启动（无 COOCCURS 边）→ hub_degree=0 < θ → False（不排除·bit-identical OFF 退化）。
    **fresh compute 每调·零 module-level side effects**（审1 Q8 P2·无 import-time storage read·
    无全局 mutation·无 module cache·避 stale-cache：COOCCURS append-only 单调增·fresh 读当前态永远正确）。

    **小 N 场景用此**（collide_score candidates≤少 / ctx_refs≤64 / refers_occurrence 候选少）。
    大批量场景（emergence preds/succs 数百 ref）用 compute_hub_set（单遍 O(#COOCCURS)）。
    """
    return hub_degree(word_ref, edge_store) >= THETA_HUB_DEGREE


def compute_hub_set(edge_store: EdgeStore, *, theta: int = THETA_HUB_DEGREE) -> set[tuple[int, int]]:
    """单遍扫 COOCCURS 边建 degree map → 返 hub_degree ≥ θ 的 ref 集。

    **perf**（2026-07-08 训练测试 cProfile 实测）：per-ref is_hub 每 ref 2 全表扫·refers_occurrence 候选过滤
    7194 调 × 2 = 276M 行扫描 = 218s（n=5 占 65%）。改单遍 edge_store.query_type 建 degree map。**复杂度注**
    （审 P2-2/7）：query_type 调 backend.select("edge", where={"edge_type": COOCCURS})·DictBackend **无索引**
    （_do_ensure_index no-op）→ 实际 O(#all_edges) 全表扫返 #COOCCURS 行·SQLite 用 ix_edge_type 近 O(#COOCCURS)。
    win 不在"O(#COOCCURS)"而在"**1 遍** vs per-ref N×2 遍"（n=5: 1 vs 14388 遍）。**大批量场景必需**
    （emergence preds/succs + refers_occurrence 候选池）·小 N 场景（collide_score candidates≤少·ctx_refs≤64）仍用 is_hub。

    **perf round3 dirty-flag 缓存**（2026-07-13·解 O(n²) 残留）：单遍改后仍每代词 occurrence（refers_occurrence:85）/
    每 emergence 信号（emergent_relation_signal:144）调 1 次 = O(#calls × #COOCCURS) = O(n²)。加 dirty-flag 缓存：
    COOCCURS 未变（edge_store.cooccurs_version 不变）→ 返缓存集 O(1)；失效（COOCCURS 插入/strength 更新 bump）→
    fresh 重算。observe token loop（代词解析）在 build_cooccurs 之前（observe.py:115 < :265）·段内多代词共享版本→缓存命中。
    **bit-identical**：缓存仅加速·失效即 fresh 重算·永不 stale（尊重原 fresh-compute 设计 :68-69·COOCCURS append-only 单调增·
    version 精确追踪）。**只 COOCCURS bump**（代词解析自己 add REFERS_TO/PROPERTY·通用 version 误失效零收益）。
    缓存挂 cognition 层模块级 WeakKeyDictionary（审2 建议·不污染 EdgeStore 实例·GC 安全·无 id 重用风险）。key=(version, theta)。

    degree[from] += strength ∧ degree[to] += strength（COOCCURS 无向·与 hub_degree 读 strength 同语义·总收口 0.1）。
    表未注册 → KeyError 容错返空集（冷启动退化·bit-identical OFF·无 crash）。
    """
    # dirty-flag 缓存：COOCCURS 未变 → O(1) 返缓存；失效 → fresh 重算（bit-identical）。
    cache = _HUB_SET_CACHE.get(edge_store)
    if cache is not None and cache[0] == edge_store.cooccurs_version and cache[1] == theta:
        return cache[2]
    try:
        rows = edge_store.query_type(EDGE_COOCCURS)
    except KeyError:
        result: set[tuple[int, int]] = set()   # edge 表未注册（bare fixture）·向后兼容
        _HUB_SET_CACHE[edge_store] = (edge_store.cooccurs_version, theta, result)
        return result
    deg: dict[tuple[int, int], int] = {}
    for r in rows:
        s = r["strength"]   # 总收口 0.1：读 strength（gate OFF 恒 1·等价数行；gate ON 频次）
        assert_int(s, _where="compute_hub_set.strength")   # 守纯整数（与 hub_degree 一致·对抗审 P2-1）
        a = (r["space_id_from"], r["local_id_from"])
        b = (r["space_id_to"], r["local_id_to"])
        deg[a] = deg.get(a, 0) + s
        deg[b] = deg.get(b, 0) + s
    result = {ref for ref, d in deg.items() if d >= theta}
    _HUB_SET_CACHE[edge_store] = (edge_store.cooccurs_version, theta, result)
    return result
