"""cognition.understanding.cooccurs — 模块6 共现 COOCCURS 分桶（§7.4 C1 防跨语言污染）。

  - lang/域分桶配对（C1）：仅桶内配对·不跨桶（中文-中文 / 代码-代码·不中英配对）
  - SHADOW tier：不进默认 A1 头聚合 / 不进 PR 传播（隔离·持有不删·防塌柱①保护）
  - strength = 频次计数测度（纯整数·不接 reward·防塌柱①·credit_sink 弃用 COOCCURS 见 reward_propagate 落点③）
  - 段内节流（COOCCURS O(n²) 风险·cap·§训练性能）

共现 ≠ 关系（裸共现给不了同指/因果·§十一#2-bis/§8.1c-bis·共现只作 staging 候选生成器）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.cognition.shared.edge_types import EDGE_COOCCURS
from pure_integer_ai.cognition.shared.hub_detect import HubDegreeState
from pure_integer_ai.config import gates

DEFAULT_COOCCURS_PAIR_CAP = 256   # 段内配对上限（O(n²) 节流·§训练性能）
COOCCURS_WINDOW_K = 2             # 窗口化邻接配对步长（gate ON 时 i 仅配 j∈[i+1,i+K]·K=2=邻接+1-skip·O(L·K)）


def make_bucket(lang: int, domain: int) -> int:
    """分桶 id = (lang, domain)（C1·仅桶内配对·防跨语言污染）。"""
    assert_int(lang, domain, _where="make_bucket")
    return (lang << 16) | domain


def segment_cooccurrence_pairs(n: int, *, cap: int = DEFAULT_COOCCURS_PAIR_CAP) -> list[tuple[int, int]]:
    """段内配对（节流·确定性·i<j·按 (i,j) 升序）。

    gate COOCCURS_WINDOW_MODE（性能修复 2026-07-08）：
      OFF → i<j **全配对** O(L²)（bit-identical 现状·段越长越爆：53-token 段 C(53,2)=1378 对）。
      ON  → **窗口化** i 仅配 j∈[i+1, i+K]（K=COOCCURS_WINDOW_K=2·邻接+1-skip·O(L·K)）。
    生产 formal_train 入口翻 ON（解训练 scaling 爆炸·镜像 CUE_EXTRACTOR_MODE 范式）·单测 OFF 守回归。
    诚实边界：治段内 O(L²) 主项·留跨段 append-only 重复（edge_store.add 不去重·独立 track）·不破 reader 语义。
    """
    pairs: list[tuple[int, int]] = []
    windowed = getattr(gates, "COOCCURS_WINDOW_MODE", False)
    for i in range(n):
        j_hi = min(i + 1 + COOCCURS_WINDOW_K, n) if windowed else n   # ON: j∈[i+1,i+K] / OFF: j∈[i+1,n)
        for j in range(i + 1, j_hi):
            pairs.append((i, j))
            if len(pairs) >= cap:
                return pairs
    return pairs


def build_cooccurs(edge_store: EdgeStore, refs: list[tuple[int, int]],
                   *, lang: int, domain: int, source: int, space_id: int,
                   cap: int = DEFAULT_COOCCURS_PAIR_CAP,
                   hub_degree_state: HubDegreeState | None = None) -> int:
    """共现 COOCCURS 建边（同桶内配对·SHADOW·频次计数）。

    C1 同桶内配对·不跨桶：refs 由 caller（observe）按单 lang/domain 段传入·同段同桶·
    段内 i<j 配对天然同桶（防中文答 apple 泄数学）·SHADOW 隔离不进默认 A1/PR。
    stub #4 修：旧版算 bucket_id 丢弃（死代码）·C1 强制点在 caller 传单语言段·make_bucket 留 utility。
    lang/domain 标识段所属桶（caller 契约·段内天然同桶故体内不重算）。
    返回建边数。
    """
    pairs = segment_cooccurrence_pairs(len(refs), cap=cap)
    dedup = getattr(gates, "COOCCURS_DEDUP_MODE", False)
    n = 0
    for i, j in pairs:
        a, b = refs[i], refs[j]
        if a == b:
            continue
        if dedup:
            # 总收口 0.1：跨段去重（同 pair 合并 strength=频次·解 append-only 堆叠·LIVE 病灶①）。
            # 仅新建边计 n（返 True）·UPDATE 不增边·built_edges=真实边数（dedup 后大降=解阻塞可观测）。
            created = edge_store.add_cooccurs_dedup(
                space_id_from=a[0], local_id_from=a[1],
                space_id_to=b[0], local_id_to=b[1],
                edge_type=EDGE_COOCCURS, source=source, tier=TIER_SHADOW,
            )
            if hub_degree_state is not None:
                hub_degree_state.observe_cooccurs(a, b, 1)
            if created:
                n += 1
        else:
            edge_store.add(
                space_id_from=a[0], local_id_from=a[1],
                space_id_to=b[0], local_id_to=b[1],
                edge_type=EDGE_COOCCURS, strength=1,   # 频次计数测度·纯整数
                source=source, epistemic_origin=None,
                tier=TIER_SHADOW,   # §十一缺口#4: 裸文本共现 SHADOW 脏持有
            )
            if hub_degree_state is not None:
                hub_degree_state.observe_cooccurs(a, b, 1)
            n += 1
    return n
