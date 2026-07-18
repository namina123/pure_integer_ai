"""cognition.understanding.similar — STEP5 PR4 EDGE_SIMILAR 相似关系建边。

"X 像 Y" → EDGE_SIMILAR(X→Y) 边（离散符号关系·D2 合规非向量·slot-filler 候选扩展）。
镜像 build_is_a_edges 范式（resolve refs + query_from 幂等 skip + 自环守 + edge_store.add）。

**D2 合规**（AGENT.md:18-36 三维度·三满才违禁）：EDGE_SIMILAR 二元离散边（非实数向量·不连续）+
确定性文本提取（非训练·不学习型）+ 结构关系边 slot-filler 扩展（非相似度 SCORE·非语义承载）→
三维度全不满→非向量→D2 合法（同 EDGE_IS_A/EDGE_CAUSES 范式）。

**EDGE_SIMILAR 不接 reward**：effective_weight.py:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·
EDGE_SIMILAR=24 不内→不进 PR 传播·不接 sn/tn·loud assert fail 防偷注入（同 IS_A/COOCCURS/PROPERTY）。
strength 恒=1（结构真值·非学习对象·同 PRECEDES/IS_A）。

**消费者**：dispatch_slot（slot_dispatch.py·gate SIMILAR_SLOT_MODE·读 EDGE_SIMILAR 双向扩展 slot 候选）。

铁律：纯整数（ConceptRef + EDGE_SIMILAR 整边·零浮点·assert_int 守）/ 确定性（query_from 幂等 + stable
  surface）/ 不写死（cue 词表元定义·builder 只机制）/ D:11 不接 reward / D2 合规（离散符号关系非向量）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import (
    EdgeStore, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM, SOURCE_CONCEPTNET, DEFAULT_STRENGTH,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT, TIER_SHADOW
from pure_integer_ai.storage.edge_types import EDGE_SIMILAR


def build_similar_edges(edge_store: EdgeStore, refs: list[tuple[int, int]],
                        *, similar_claims: list[tuple[int, int]],
                        source: int, space_id: int) -> int:
    """STEP5 PR4：similar_claims 2-tuple → EDGE_SIMILAR 边（X→Y·TIER_SHADOW·strength=1）。

    similar_claims：``(left_idx, right_idx)`` 2-int tuple·经 refs 解析为概念 ref。
    "X 像 Y" → EDGE_SIMILAR(X→Y)。返建边数。

    每条 claim：
      ① 越界守（idx<0 / idx>=len(refs) → skip·fail-soft·同 build_property_edges 范式）。
      ② 自环守（X==Y → skip·防御·同 build_causes a==b 守）。
      ③ query_from 幂等 skip（同 (X,Y,EDGE_SIMILAR,source) 已建则 skip·EdgeStore.add append-only 不去重）。
      ④ edge_store.add(EDGE_SIMILAR, X→Y, strength=1, source, epistemic_origin=None, tier=TIER_SHADOW)。
    """
    assert_int(source, space_id, _where="build_similar_edges.args")
    if not similar_claims:
        return 0
    n = 0
    for left_idx, right_idx in similar_claims:
        if (left_idx < 0 or right_idx < 0
                or left_idx >= len(refs) or right_idx >= len(refs)):
            continue   # 越界守（refs 切片映射错位 fail-soft·不崩）
        x_ref = refs[left_idx]
        y_ref = refs[right_idx]
        if x_ref == y_ref:
            continue   # 自环不建（X→X·防御·同 build_causes _insert a==b 守）
        # query_from 幂等 skip（同 (X,Y,EDGE_SIMILAR,source) 已建则 skip）
        existing = edge_store.query_from(x_ref[0], x_ref[1], edge_type=EDGE_SIMILAR)
        already = any(
            row.get("space_id_to") == y_ref[0]
            and row.get("local_id_to") == y_ref[1]
            and row.get("source") == source
            for row in existing
        )
        if already:
            continue
        edge_store.add(
            space_id_from=x_ref[0], local_id_from=x_ref[1],
            space_id_to=y_ref[0], local_id_to=y_ref[1],
            edge_type=EDGE_SIMILAR, strength=DEFAULT_STRENGTH,   # 恒=1（结构真值·非学习对象）
            source=source, epistemic_origin=None,   # bare text 断言·非 structured/cue/llm 认识论源
            tier=TIER_SHADOW,
        )
        n += 1
    return n


def build_similar_edge(edge_store: EdgeStore,
                       a: tuple[int, int], b: tuple[int, int],
                       *, source: int, epistemic: int, space_id: int,
                       strength: int = DEFAULT_STRENGTH) -> int:
    """SIMILAR 近义建边（a~b 对称·单边 a→b 存储·reader 双向查·镜像 build_antonym_edge）。

    epistemic ∈ {EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM}（认识论来源·禁裸共现·镜像 build_antonym_edge）。
    strength：静态结构真值（M9·默认 DEFAULT_STRENGTH=1·恒=1·非学习对象）。自环不建（a≠b·同 build_antonym_edge:59）。
    **对称约定**：建单边 a→b（文件序）·SIMILAR 无向对称·reader（dispatch_slot SIMILAR_SLOT_MODE out_edges+in_edges
    双向查·#898）得 a~b 闭包。boot 路径 tier=TIER_PRIMARY（外部词典权威源·异 observe 路径 build_similar_edges TIER_SHADOW bare-text）。
    """
    assert epistemic in (EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM), \
        "SIMILAR 必须有认识论来源·禁裸共现"
    assert_int(strength, _where="build_similar_edge.strength")
    if a == b:
        return 0
    edge_store.add(
        space_id_from=a[0], local_id_from=a[1],
        space_id_to=b[0], local_id_to=b[1],
        edge_type=EDGE_SIMILAR, strength=strength,
        source=source, epistemic_origin=epistemic,
        order_index=None, role=None,   # SIMILAR 无 order_index 时序语义（同 ANTONYM/MEREOLOGY）
        tier=TIER_PRIMARY,
    )
    return 1


def bootstrap_similar_edges(concept_index, edge_store: EdgeStore,
                            surface_pairs: list[tuple[str, str]],
                            *, space_id: int,
                            source: int = SOURCE_CONCEPTNET,
                            epistemic: int = EPI_STRUCTURED,
                            strength: int = DEFAULT_STRENGTH) -> int:
    """SIMILAR 批量 boot 种边（T-L1c 近义对称·surface pairs → ensure → build·镜像 bootstrap_antonym_edges 范式）。

    入参是 surface 文本对（a_surface, b_surface·无序对称·文件每对一次）·caller 不依赖语料 token 切片（boot 时种边·早于 observe）。
    默认来源① ConceptNet /r/Synonym（source=SOURCE_CONCEPTNET·epistemic=EPI_STRUCTURED）·zh 生产源 ChineseSemanticKB 同义关系库（caller 传 SOURCE_CHINESE_KB）。
    **幂等 skip 按源细化**（镜像 bootstrap_antonym_edges:108-115）：query_from 查同 (a,b,EDGE_SIMILAR,source) 已建则 skip。

    **无文件零副作用硬守（bit-identical·P0·镜像 bootstrap_antonym_edges:99-100）**：surface_pairs 空 → 立即 return 0·
    **绝不调 concept_index.ensure / query_from / build**（无 ZERO_AI_LOCAL_DIR → resolve 返 [] → 图与不接 T-L1c bit-identical）。

    每 (a_surface, b_surface)：concept_index.ensure 两 ref（TIER_PRIMARY·NODE_CONCEPT）→
    query_from 幂等 skip → build_similar_edge（a==b 跳·单边 a→b）。返建边数。

    **对称存储**：单边 a→b（文件序）·reader 双向查（dispatch_slot SIMILAR_SLOT_MODE·#898）。文件应每对一次（canonical 序）避冗余。

    铁律：纯整数（ConceptRef + EDGE_SIMILAR 整边·零浮点）/ 不写死（surface 来自外部文件·本函数只机制非语义）/
      §8.1c（来源① EPI_STRUCTURED 合规）/ bit-identical（空 pairs 零副作用 + query_from 幂等）。
    诚实边界：surface 真伪 = 外部数据责任（接地墙·#479）·SIMILAR 不接 reward（effective_weight:82 assert 只认
      PRECEDES/CAUSES/REFERS_TO·SIMILAR 不内·同 ANTONYM/IS_A/MEREOLOGY·boot 边无 reward 误判风险）。
    """
    if not surface_pairs:
        return 0   # P0·无文件零副作用硬守（不调 ensure/query_from/build·CI/生产 default bit-identical）
    assert_int(space_id, source, strength, _where="bootstrap_similar_edges.args")
    n = 0
    for a_surf, b_surf in surface_pairs:
        a_ref = concept_index.ensure(
            a_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        b_ref = concept_index.ensure(
            b_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 幂等 skip（按源细化·镜像 bootstrap_antonym_edges:108-115）：query_from 查 a 已有同 (b,source) SIMILAR 边。
        existing = edge_store.query_from(a_ref[0], a_ref[1], edge_type=EDGE_SIMILAR)
        already = any(
            row.get("space_id_to") == b_ref[0]
            and row.get("local_id_to") == b_ref[1]
            and row.get("source") == source
            for row in existing
        )
        if already:
            continue   # 同源同三元组已建→skip（幂等·resume 跨 run / 重复 boot 不 corrupt·EdgeStore.add 不去重）
        n += build_similar_edge(edge_store, a_ref, b_ref,
                                source=source, epistemic=epistemic, space_id=space_id,
                                strength=strength)
    return n
