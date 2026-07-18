"""cognition.understanding.antonym — ANTONYM 反义对称建边（T-L1e·客观序 gap 补·EDGE_ANTONYM=26）。

ANTONYM = 反义对称 typed 边（大↔小·冷↔热·concept↔concept 1 阶）。**近 EDGE_SIMILAR 对称形·异 SIMILAR 语义**：
反义=语义对立（大 vs 小）·非相似（大 ~ 高）。**非 verify_inverse**——代数逆 = transform↔transform（d/dx↔∫·+↔−·T-L4
·verify_inverse 真验 B∘A=identity·只验数学 transform 不验语言词对）·**语言反义 = concept↔concept 1 阶**（近 SIMILAR·
verify_inverse 对词对返 None can't-verify·#479 外部 seed 非 verify·doc/重来_语言域断奶客观序_2026-07-15 §三 T-L1e·§四.3）。

构造器 = 把合法来源落成 EDGE_ANTONYM 边（建边接线·非"算关系"·镜像 similar.py / mereology.py 范式）。

**对称模型 = 单边 a→b 存储 + reader 双向查**（镜像 EDGE_SIMILAR build_similar_edges X→Y 单边·similar_candidates
out_edges+in_edges 双向查·异 alias_bridge 双边 a→b+b→a 显式存储）：antonym_facts 文件每对一次（a b·无序对称）·
本构造器建单边 a→b·reader（待接线·镜像 similar_candidates）双向查得对称闭包。idempotent skip 防 resume 重边。

**来源分层（§8.1c-bis 同构·epistemic_origin 守认识论来源·禁裸共现）**：
  ① 结构化源（ConceptNet /r/Antonym / WordNet antonym 对·照搬·对称对无方向）→ EPI_STRUCTURED
  ② 反义 cue + 句法位置提取（defer·"X 与 Y 相反/对立"·observe-time·boot loader 是首版主路径·同 mereology/is_a/causes）
  ③ 断奶前 LLM 教师确认（teacher·歧义·defer）→ EPI_LLM_CONFIRM
  ④ 反义无传递闭包（大↔小 ∧ 小↔中 ⊬ 大↔中·反义不传递·无闭包）

**测度编码（M9·镜像 EDGE_SIMILAR）**：ANTONYM **不接 reward 反传**（effective_weight:82 assert 只认
  {PRECEDES,CAUSES,REFERS_TO}·ANTONYM 不在内·同 SIMILAR/IS_A/MEREOLOGY）。测度走**strength 恒=1 静态结构真值**
  （同 EDGE_SIMILAR/PRECEDES·非学习对象·sn/tn 建 0/0 不动）。

**方向（对称无向·单边存储约定）**：from=a to=b（文件序）·reader 双向查得 a↔b。自环不建（a≠b·词非自身反义）。

**PR 邻接安全（镜像 EDGE_SIMILAR）**：ANTONYM 是结构边（非 PR 邻接头·用静态 strength·不调 effective_weight）·
  dag_path head_types={PRECEDES,CAUSES} + a3_pr_wrapper 双过滤挡 ANTONYM 入 PR·effective_weight:82 assert
  拒 ANTONYM（同 SIMILAR=24/IS_A=10 不在 {PRECEDES,CAUSES,REFERS_TO}）·无须改 effective_weight。

**#479 truth 墙（守不破·反 theater）**：antonym 对来自外部 ConceptNet/WordNet（EPI_STRUCTURED·来源①）·系统只落边
  不验真（不跑 vm_proof/cross-verify 验词对反义真值）·契合=语义层非验真偷渡（代数 verify_inverse 只验 transform 不验语言）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import (
    EdgeStore, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM, SOURCE_CONCEPTNET, DEFAULT_STRENGTH,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_ANTONYM


def build_antonym_edge(edge_store: EdgeStore,
                       a: tuple[int, int], b: tuple[int, int],
                       *, source: int, epistemic: int, space_id: int,
                       strength: int = DEFAULT_STRENGTH) -> int:
    """ANTONYM 反义建边（a↔b 对称·单边 a→b 存储·reader 双向查）。

    epistemic ∈ {EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM}（认识论来源·禁裸共现·镜像 build_mereology_edge）。
    strength：静态结构真值（M9·默认 DEFAULT_STRENGTH=1·恒=1·非学习对象·镜像 EDGE_SIMILAR）。
    自环不建（a≠b·词非自身反义·同 build_mereology_edge:58）。返建边数。

    **对称约定**：建单边 a→b（文件序）·ANTONYM 是无向对称关系·reader（待接线·镜像 similar_candidates out_edges+in_edges
    双向查）得 a↔b 闭包。异 alias_bridge 显式双边存储（antonym 单边够·reader 双向查）。
    """
    assert epistemic in (EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM), \
        "ANTONYM 必须有认识论来源·禁裸共现"
    assert_int(strength, _where="build_antonym_edge.strength")
    if a == b:
        return 0
    edge_store.add(
        space_id_from=a[0], local_id_from=a[1],
        space_id_to=b[0], local_id_to=b[1],
        edge_type=EDGE_ANTONYM, strength=strength,
        source=source, epistemic_origin=epistemic,
        order_index=None, role=None,   # ANTONYM 无 order_index 时序语义（同 SIMILAR/MEREOLOGY）
        tier=TIER_PRIMARY,
    )
    return 1


def bootstrap_antonym_edges(concept_index, edge_store: EdgeStore,
                            surface_pairs: list[tuple[str, str]],
                            *, space_id: int,
                            source: int = SOURCE_CONCEPTNET,
                            epistemic: int = EPI_STRUCTURED,
                            strength: int = DEFAULT_STRENGTH) -> int:
    """ANTONYM 批量 boot 种边（T-L1e·surface pairs → ensure → build·镜像 bootstrap_mereology_edges 范式）。

    入参是 surface 文本对（a_surface, b_surface·无序对称·文件每对一次）·caller 不依赖语料 token 切片（boot 时种边·早于 observe）。
    默认来源① ConceptNet /r/Antonym / WordNet antonym（source=SOURCE_CONCEPTNET·epistemic=EPI_STRUCTURED）。
    **幂等 skip 按源细化**（镜像 bootstrap_mereology_edges:107-115）：query_from 查同 (a,b,EDGE_ANTONYM,source) 已建则 skip。

    **无文件零副作用硬守（bit-identical·P0·镜像 bootstrap_mereology_edges:97-98）**：surface_pairs 空 → 立即 return 0·
    **绝不调 concept_index.ensure / query_from / build**（无 ZERO_AI_LOCAL_DIR → resolve 返 [] → 图与不接 T-L1e bit-identical）。

    每 (a_surface, b_surface)：concept_index.ensure 两 ref（TIER_PRIMARY·NODE_CONCEPT）→
    query_from 幂等 skip → build_antonym_edge（a==b 跳·:58 已守·单边 a→b）。返建边数。

    **对称存储**：单边 a→b（文件序）·reader 双向查（镜像 similar_candidates）。若文件含逆序对（b a）·query_from(b) 查 to==a
    不命中（a→b 是 from=a）→ 建第二边 b→a（对称关系双边无害·同 logical pair）。文件应每对一次（canonical 序）避冗余。

    铁律：纯整数（ConceptRef + EDGE_ANTONYM 整边·零浮点）/ 不写死（surface 来自外部文件·本函数只机制非语义）/
      §8.1c（来源① EPI_STRUCTURED 合规）/ bit-identical（空 pairs 零副作用 + query_from 幂等）。
    诚实边界：surface 真伪 = 外部数据责任（接地墙·#479）·ANTONYM 不接 reward（effective_weight:82 assert 只认
      PRECEDES/CAUSES/REFERS_TO·ANTONYM 不内·同 SIMILAR/IS_A/MEREOLOGY·boot 边无 reward 误判风险）·非 verify_inverse（语言反义
      concept↔concept·代数 verify_inverse 只验 transform·#479）。
    """
    if not surface_pairs:
        return 0   # P0·无文件零副作用硬守（不调 ensure/query_from/build·CI/生产 default bit-identical）
    assert_int(space_id, source, strength, _where="bootstrap_antonym_edges.args")
    n = 0
    for a_surf, b_surf in surface_pairs:
        a_ref = concept_index.ensure(
            a_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        b_ref = concept_index.ensure(
            b_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 幂等 skip（按源细化·镜像 bootstrap_mereology_edges:107-115）：query_from 查 a 已有同 (b,source) ANTONYM 边。
        existing = edge_store.query_from(a_ref[0], a_ref[1], edge_type=EDGE_ANTONYM)
        already = any(
            row.get("space_id_to") == b_ref[0]
            and row.get("local_id_to") == b_ref[1]
            and row.get("source") == source
            for row in existing
        )
        if already:
            continue   # 同源同三元组已建→skip（幂等·resume 跨 run / 重复 boot 不 corrupt·EdgeStore.add 不去重）
        n += build_antonym_edge(edge_store, a_ref, b_ref,
                                source=source, epistemic=epistemic, space_id=space_id,
                                strength=strength)
    return n
