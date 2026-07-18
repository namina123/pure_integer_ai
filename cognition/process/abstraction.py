"""cognition.process.abstraction — IS_A 抽象层 LCA 查询基建（S3 D3·钥匙①发现线·第二刀前置）。

激活 `algorithm.closure.transitive_closure` 的首个 caller（全 repo 零 caller 的泛型闭包引擎·
反"零 caller theater"）+ 提供 `common_is_a_ancestor` LCA（最近共同祖先）查询。

**用途（钥匙①第二刀抽象对撞·§四抽象层）**：语言结构发现的 shape_signature 抽象级——token 沿 IS_A
链上卷到共同抽象层（"追"/"咬"在"动词"抽象层同形）。上卷深度 = LCA（最近共同祖先·最具体共同抽象·
**非硬编码词性**·守"不写死"）。§8.1c 合规：沿合规 IS_A 边做查询（LCA climb）非边建造·不撞死刑
（§四已判：遍历合规边不重新触发建造禁令）。IS_A 边本身只从结构化源/系词/LLM 教师建（is_a.py:54
assert epistemic_origin·禁裸共现）。

**消费者**（build_isa_ancestor_map 的 4 caller·反 theater 真 live·非零 caller theater）：
  - ``structure_discover._cluster_by_lca``（钥匙①第二刀抽象级 shape_signature·**ungated**·set_lca 沿
    IS_A 链上卷聚类·ancestor_map 在 structure_discover:1068/1173/1354 建）——D3"基建先行"原旨已兑现。
  - ``selection_pref.build_selection_pref_count``（gate ``SELECTION_PREF_MODE`` 默认 OFF·写侧 class_of）。
  - ``graph_view`` selection_pref_score read-side（gate ``GENERATE_SELECTION_PREF_MODE`` 默认 OFF·读侧 class_of）。
  - ``reward_propagate`` 落点⑥ sp_sn feed（gate ``PROCESS_REWARD_PROP_MODE`` 默认 OFF·class_of 配对）。
  3 gate-OFF 默认守回归（生产 ON 时活）·structure_discover ungated 即第二刀 live。
  **诚实边界（#1133）**：abstract 重进 closure 后 4 consumer 皆可达·但 n=20 默认 gate 下仅 ungated 的
  structure_discover 实际驱动·reward delta 不变（15144·**wired-not-load-bearing at 语料当前规模**·benefit defer 训练轴）。

**致命陷阱路排除（§四·设计强调）**：抽象层**绝不用 role_seq/位置桶**（emergent_role 冷启动全
SUBJECT·位置桶是硬编码退化·违"不写死"）。必须走 IS_A LCA 上卷（学出/教师/结构源边·沿链 climb
是查询非写死·合规）。

接口：
  build_isa_ancestor_map(backend, *, space_id) -> dict[ConceptRef, set[ConceptRef]]
      读全 space IS_A 边 → transitive_closure(types={EDGE_IS_A}) → 返 {ref: 祖先集}（run-scoped cache·
      避免每对查询重算闭包·closure"CLOSURE 派生不存储"故 caller 缓存）。
  common_is_a_ancestor(ref_a, ref_b, ancestor_map) -> ConceptRef | None
      两 ref 的最近共同祖先（LCA·最具体共同抽象·偏序无须 chain_len）。
  set_lca(tokens, ancestor_map) -> ConceptRef | None
      多 token 集合 LCA（**含自身 closure·集交集**·S3 第二刀 Interp2 抽象聚类用·解 pairwise-reduce drift）。

铁律：纯整数（ConceptRef=int 二元组·无浮点）/ 确定性（NodeRef 自然序 tiebreak·bit-identical）/
单向依赖（process L5→algorithm closure L2 向下·L5→storage L0 读边）/ 不写死（上卷深度=LCA 结构涌现
非词性预设·无规则硬编码）/ 幂等（纯读·重复调同果）。
诚实边界：LCA 是结构查询非语义理解（沿 IS_A 边 climb·边本身须合规来源·§8.1c）/ IS_A DAG 无环
（proper subset 传递·is_a.py:57 自环不建）/ diamond 多 LCA 候选 NodeRef 升序 tiebreak（确定性·
非"唯一正确"）/ 无共同祖先→None（两 ref 无 IS_A 抽象交集·上卷到顶）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.algorithm.graph_algebra import isa_ancestor_map
from pure_integer_ai.algorithm.closure import transitive_closure, closure_pure_refers_to
from pure_integer_ai.storage.edge_types import EDGE_IS_A, EDGE_MEREOLOGY, EDGE_REFERS_TO
from pure_integer_ai.storage.edge_store import SOURCE_CONCEPTNET, EPI_STRUCTURED, SUBTYPE_PURE_ALIAS
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.config import gates


def build_isa_ancestor_map(backend, *, space_id: int) -> dict[ConceptRef, set[ConceptRef]]:
    """IS_A 祖先图 cache（run-scoped·S3 D3·激活 transitive_closure caller）。

    从 backend 读全 space IS_A 边（child→parent·is_a.py from=child to=parent）→ 调
    `algorithm.transitive_closure(types={EDGE_IS_A})` 算派生闭包（激活全 repo 零 caller 的泛型闭包引擎·
    反"零 caller theater"·§十五决策9 CLOSURE 派生不存储）→ 返 {ref: 祖先集}（每 ref 的全部 IS_A 祖先）。

    返 {ref: set[祖先 ConceptRef]}（run-scoped cache·common_is_a_ancestor 多次查用·避免每对重算闭包）。
    空 dict = 该 space 无 IS_A 边（冷启动 / 无结构化源 / 无系词提取）。

    铁律：纯整数 / 确定性（transitive_closure BFS 节点自然序·bit-identical）/ 单向依赖（L5→algorithm L2
    向下 + L5→storage L0 读边）/ 幂等（纯读·重复同果）。
    """
    assert_int(space_id, _where="build_isa_ancestor_map.space_id")
    # #1115 perf 迭代2：observe hoist 增量优先（所有 caller 统一增量 map·免 gen-cache 频繁失效重建）。
    # cProfile n=50 证：structure_discover/graph_view/reward_propagate 走 gen-cache·observe 建 IS_A → gen bump →
    # cache miss → 全图 Tarjan 重建（188×0.4s=75s tottime 第一）。hoist 优先 → 这些 caller 返增量 map·免重建。
    # gate OFF（CI）/ 环禁增量（hoist[space]=None）/ 非 observe 上下文（hoist 未建）→ 走既有 gen-cache（bit-identical）。
    # hoist amap 由 observe 增量 apply 维护·消费者纯读返引用安全（单线程顺序·镜像 gen-cache cached[1] 引用范式）。
    _hoist = getattr(backend, "_isa_ancestor_hoist", None)
    if _hoist is not None:
        _entry = _hoist.get(space_id)
        if _entry is not None:   # (amap, didx) observe 增量 map（None=环禁增量→退化 gen-cache）
            return _entry[0]
    # 读全 space IS_A 边（L5 读路径·edge 表·is_a.py from=child to=parent）。
    # **#1133 DONE**：abstract SOURCE_CHINESE_KB **重进 closure**（cycle-cleaned 成 DAG·enrich 抽象层 LCA）。
    # 原	stopgap（排除 SOURCE_CHINESE_KB）已撤——cycle-cleaning（scratch/clean_abstract_cycles.py·DFS back-edge removal·
    # 306998 raw → 300233 DAG·0 环·giant SCC=3923 + 65 小环 4115 节点破净）+ #1142 语料相关过滤（boot 时 abstract 只留
    # corpus-relevant）→ abstract 子集是干净 DAG（子集 of DAG is DAG）。graph_algebra.isa_ancestor_map（SCC 凝聚 O(V+E)·#1136）
    # 处理（纵 residual 环亦不崩·凝聚·bit-identical）。**bit-identical 守**：CI 无 abstract_facts → resolve [] → 无 CHINESE_KB 边
    # → edges 同旧（零 abstract）→ 逐字旧输出。生产有 abstract → ancestor_map 加 abstract 祖先（预期 enrichment·非偷渡）。
    # ★ perf #1144：O(1) gen-hit 短路。IS_A 拓扑（child→parent 边集）**仅经 insert 变**（update 只动
    # tier/strength/sn/tn 非拓扑·edge_store:320/361/373/379/419 where 含 edge_type 全验·IS_A reward-inert
    # effective_weight:82 故无 reward update·全仓零 delete("edge")）→ backend.isa_edge_generation（insert bump）
    # gen-match ⟺ 拓扑不变 ⟺ ancestor_map 不变（bit-identical）→ 跳 select(全 IS_A 行)+frozenset。
    # **根因**：build_selection_pref_count 每 generate 调一次（n=20 is_a stage 440 次·每次旧路径 select 53k 行 +
    # frozenset 53k 仅查 cache 有效 = O(E)/call·cProfile 占 cumtime 30s/90s = 33%）→ gen 稳定（generate 期无 IS_A
    # insert）则全 O(1) 命中。gen 变/首建/backend 无 isa_edge_generation → 走权威重建（frozenset 退化·bit-identical）。
    _gen_fn = getattr(backend, "isa_edge_generation", None)
    gen = _gen_fn(space_id) if _gen_fn is not None else None
    cache = getattr(backend, "_isa_ancestor_cache", None)
    if cache is None:
        cache = {}
        try:
            backend._isa_ancestor_cache = cache
        except (AttributeError, TypeError):
            cache = None
    if cache is not None and gen is not None:
        cached = cache.get(space_id)
        if cached is not None and cached[0] == gen:
            return cached[1]   # O(1) 命中（IS_A 拓扑未变·跳 select+frozenset·消费者纯读返引用安全）
    # gen 变 / 首建 / backend 无 gen signal → 全 select + 权威闭包重建
    rows = backend.select("edge", where={
        "space_id_from": space_id, "edge_type": EDGE_IS_A,
    })
    edges = [((r["space_id_from"], r["local_id_from"]),
              (r["space_id_to"], r["local_id_to"]),
              EDGE_IS_A, None)
             for r in rows]   # 全 IS_A 边（abstract cycle-cleaned·#1133 DONE·不再排除 SOURCE_CHINESE_KB）
    # 图代数祖先闭包（#1136·algorithm/graph_algebra.isa_ancestor_map·IS_A DAG 拓扑序单遍传播 O(V+E)·
    # 远快于 closure.transitive_closure BFS-per-source O(V·E)·解 307k IS_A perf 阻断·「图代数派生」§九.3 落地）。
    # 反"零 caller theater"：graph_algebra 环 fallback 仍调 transitive_closure（caller 保留）。bit-identical（DAG 闭包唯一·
    # 含直接父 include_direct 等价·环噪声 fallback closure 守正确）。撤 #1133 stopgap（abstract 重进 LCA·enrich 抽象层）。
    ancestor_map, _fell_back = isa_ancestor_map(edges)
    if cache is not None:
        if gen is not None:
            cache[space_id] = (gen, ancestor_map)   # gen-keyed（跳 frozenset·gen 权威信号）
        else:
            # 无 gen signal（backend 不支持 isa_edge_generation）→ 退化 edge_key frozenset 自失效（既有 bit-identical）。
            edge_key = frozenset((e[0], e[1]) for e in edges)
            cache[space_id] = (edge_key, ancestor_map)
    return ancestor_map


def build_isa_ancestor_map_with_index(backend, *, space_id: int
                                      ) -> tuple[dict[ConceptRef, set[ConceptRef]],
                                                 dict[ConceptRef, set[ConceptRef]]]:
    """IS_A 祖先图 + **后代索引 desc_index**（observe hoist 增量维护用·#1115 perf）。

    返 (ancestor_map_fresh, desc_index)：
      - ancestor_map_fresh = build_isa_ancestor_map 结果的**深拷贝**（fresh·observe 可 mutate·不污染
        backend._isa_ancestor_cache 槽·审1 MED-3：cache 槽返引用·若 observe 持引用 + apply mutate →
        cache 槽 (gen_old, mutated_map_with_future_state) gen_old 与内容不一致·未来改 gen-bump 时序即破）。
      - desc_index[a] = {d : a ∈ ancestor_map[d]}（a 的后代集·逆向索引·apply_isa_edge_to_map O(1) 查后代用）。

    从 ancestor_map 一次性反推 desc_index：for d, ancs: for a in ancs: desc_index[a].add(d)。
    O(total ancestor pairs)·14745 边深 closure 几十万 pairs·per-run 首建一次可接受（cProfile 主热点是
    每段重建·首建摊薄到 per-run）。

    **生命周期**：caller（observe hoist）per-run 建**一次**（非 per-observe-call·否则 90×0.3s=27s 更差）·
    observe 期间 build_is_a_edges 建新 IS_A 时调 apply_isa_edge_to_map 增量维护（环检测 fall back）。
    详见 doc/重来_observe性能_#1115_修法设计_2026-07-18.md §7/§13。

    铁律：纯整数 / 确定性（ancestor_map 顺序无关·desc_index 反推确定）/ 单向依赖（L5→storage L0 读 + L5→algorithm L2 闭包·
    同 build_isa_ancestor_map）/ 幂等（纯读 backend·fresh copy 每次新建但内容同）。
    """
    assert_int(space_id, _where="build_isa_ancestor_map_with_index.space_id")
    base = build_isa_ancestor_map(backend, space_id=space_id)
    # fresh copy（深拷贝·observe mutate 不污染 backend gen-cache 槽·审1 MED-3）
    fresh: dict[ConceptRef, set[ConceptRef]] = {k: set(v) for k, v in base.items()}
    desc_index: dict[ConceptRef, set[ConceptRef]] = {}
    for d, ancs in fresh.items():
        for a in ancs:
            desc_index.setdefault(a, set()).add(d)
    return fresh, desc_index


def apply_isa_edge_to_map(ancestor_map: dict[ConceptRef, set[ConceptRef]],
                          desc_index: dict[ConceptRef, set[ConceptRef]],
                          child: ConceptRef, parent: ConceptRef) -> bool:
    """增量加 IS_A 边 (child→parent) 到 ancestor_map + desc_index（#1115 perf·非全图重建）。

    返 bool：True=增量成功（DAG 边）/ False=闭环（caller 须 fall back 全量重建）。

    **★ HIGH-1 环检测（审1 BLOCKER·守既有 SCC 环契约）**：
    加 (child→parent) 闭环 ⟺ parent 是 child 的后代（child 是 parent 祖先）⟺ parent ∈ desc_index[child]。
    命中 → 返 False（不改 map）→ caller invalidate hoist map → 后续走 backend gen-cache 全量重建
    （gen 已 bump·自然 miss·跑 graph_algebra.isa_ancestor_map 的 SCC 凝聚·正确处理环）。
    既有系统显式支持环（graph_algebra SCC + test_graph_algebra:59-73 锁契约·"raw 数据噪声成环"）·
    朴素增量丢此安全网——2-cycle (a→b)+(b→a) 逐边 apply 会使 anc[a] 含自身（违反无自环不变量·
    nearest/common/set_lca 依赖·abstraction.py:188/226/280）→ class_of/sp_tn/reward 错。
    DAG 边增量（多数）/ 环边 fall back 全量（少数）→ 两路合守 bit-identical。

    算法（DAG 域·闭包单调 + 并集交换·增量 == 全量子集应用）：
      new_anc = ancestor_map[parent] ∪ {parent}
      affected = {child} ∪ desc_index[child]（child 及其所有后代）
      for d in affected: ancestor_map[d] |= new_anc; desc_index[new_anc 中每个] |= affected

    O(|affected| × |new_anc|) per edge。observe 期间每段建几个 IS_A → 增量几次（局部·非全图 0.3s）。

    守卫：
      - child == parent → 返 True（自环 no-op·镜像 isa_ancestor_map:149·LOW-1）
      - parent 已是 child 祖先（重复边）→ 返 True（幂等 no-op）

    铁律：纯整数 / 确定性（集合并集交换·多边入序/排序皆 work·LOW-3 DAG 上序无关）/ bit-identical
    （DAG 增量==全量数学证·环 fall back 全量 SCC·doc §7）/ 单向依赖（L5 纯算·入参 map mutate·无 I/O）。
    """
    assert_int(child[0], child[1], parent[0], parent[1], _where="apply_isa_edge_to_map.refs")
    if child == parent:
        return True   # 自环 no-op（LOW-1 守卫·镜像 isa_ancestor_map:149 child==parent continue）
    # ★ HIGH-1 环检测：parent 是 child 的后代（child ∈ ancestor_map[parent]）⟺ parent ∈ desc_index[child]
    # → 加 (child→parent) 闭环 → fall back 全量（caller 责任）
    if parent in desc_index.get(child, ()):
        return False
    # parent 已是 child 祖先（重复边）→ 幂等 no-op
    if parent in ancestor_map.get(child, ()):
        return True
    new_anc = ancestor_map.get(parent, set()) | {parent}
    if not new_anc:
        return True   # parent 无祖先且自身已查（理论不可达·防御）
    affected = {child} | desc_index.get(child, set())
    for d in affected:
        cur = ancestor_map.get(d)
        if cur is None:
            cur = set()
            ancestor_map[d] = cur
        cur |= new_anc
        for a in new_anc:
            desc_index.setdefault(a, set()).add(d)
    return True


def build_isa_ancestor_map_external(backend, *, space_id: int
                                    ) -> dict[ConceptRef, set[ConceptRef]]:
    """IS_A 祖先图（**仅外部源① ConceptNet**·刀 C 量化验证用·反 single-source theater）。

    与 `build_isa_ancestor_map`（混三来源·结构发现/聚类/selection_pref/nearest_is_a_ancestor 用·统计侧要全
    IS_A 信号·不论来源）的差异：select 加 ``source=SOURCE_CONCEPTNET + epistemic_origin=EPI_STRUCTURED``
    双 filter → 仅 ConceptNet 外部断言进闭包·cue 自产 IS_A（EPI_CUE·"X 是 Y" 系词提取）/ LLM（EPI_LLM_CONFIRM）
    不混入。刀 C 量化验证查此图 → 数据来自外部 R6 独立源 → 构造性**验证**（verify_source=EXTERNAL·非 SELF_PRODUCED
    检查）。**反 theater 核心**：若刀 C 用混图·"所有 X 都是 Y" 的系词还自产 IS_A 对 (X,Y) → 验证平凡通过 = 自证闭环。

    **为何 source + epistemic 双 filter**：ConceptNet loader（is_a.py:93-94 bootstrap_is_a_edges default）
    同时设 source=SOURCE_CONCEPTNET + epistemic=EPI_STRUCTURED·双 filter 守"同源同认识论层"·防未来某来源误标
    source=CONCEPTNET 但 epistemic=CUE 混入（防御性·既有 loader 两字段必同设·双 filter 无副作用）。

    **bit-identical 守卫**：CI/生产 default 无 ZERO_AI_LOCAL_DIR → resolve_is_a_facts 返 [] → bootstrap 零副作用
    → 无 ConceptNet 边 → 本函数返空 dict → 刀 C 所有 claim can't-verify（None·诚实降级·非 theater·非平凡通过）。
    UNIVERSAL_PROOF_MODE default OFF → 本函数不被调（路由守）。

    返 {ref: 祖先集}（仅外部源·空 dict = 该 space 无 ConceptNet 边）。纯读 filter·不改边·幂等·镜像
    build_isa_ancestor_map 结构（filter 后走同 transitive_closure include_direct=True）。

    铁律：纯整数 / 确定性 / 单向依赖（L5→algorithm L2 + L5→storage L0 读边）/ 幂等（纯读·重复同果）/
    bit-identical（filter 纯加 where 键·既有 build_isa_ancestor_map 及三消费者零影响·独立函数隔离）。
    诚实边界：ConceptNet 覆盖=外部数据责任（接地墙·判别靠外部源结构涌现非系统写死）/ 闭包不计 cue 自产边。
    """
    assert_int(space_id, _where="build_isa_ancestor_map_external.space_id")
    # 读仅外部源① ConceptNet IS_A 边（双 filter·source + epistemic_origin·镜像 build_isa_ancestor_map 加 where 键）
    rows = backend.select("edge", where={
        "space_id_from": space_id, "edge_type": EDGE_IS_A,
        "source": SOURCE_CONCEPTNET, "epistemic_origin": EPI_STRUCTURED,
    })
    edges = [((r["space_id_from"], r["local_id_from"]),
              (r["space_id_to"], r["local_id_to"]),
              EDGE_IS_A, None) for r in rows]
    # 图代数祖先闭包（#1136·isa_ancestor_map·同 build_isa_ancestor_map 派生层·仅 filter 异·bit-identical）。
    ancestor_map, _fell_back = isa_ancestor_map(edges)
    return ancestor_map


def nearest_isa_ancestor(ancestor_map: dict[ConceptRef, set[ConceptRef]],
                         ref: ConceptRef) -> ConceptRef:
    """ref 的 IS_A 最近祖先（最具体·最深·非 NodeRef 升序首·S4 后续加固·项2）。

    单 ref 的最近祖先 = ancestors 中最深的（无其他 ancestor 是其后代=更深更具体）。
    x 是最近祖先 ⟺ ancestors 中无 y≠x 使 x ∈ ancestor_map[y]（x 是 y 祖先·y 比 x 深→x 非最近）。
    diamond 多候选（不可比祖先）→ NodeRef 升序 tiebreak（bit-identical·同 common_is_a_ancestor:111 范式）。

    替代三处 min(ancestors)（graph_view:200 + selection_pref:59 + reward_propagate:71）·min 取 NodeRef
    升序首可能非最深（多层 IS_A 狐狸→动物→生物·生物 local_id 小则 min=生物·非最近动物·泛化过粗）。

    与 common_is_a_ancestor 区别：common 是两 ref 共同祖先中最深·本函数是单 ref 所有祖先中最深。
    无祖先（ref 不在 ancestor_map / 空集）→ 返 ref 自身（冷启动退化·token 级无上卷）。

    铁律：纯整数 / 确定性（NodeRef 升序 tiebreak·bit-identical）/ 幂等（纯读 ancestor_map）/ O(|ancestors|·|anc|)
    （MED-2 perf #1147·原 O(|ancestors|²)·T-L1a 30k IS_A 深 closure → ancestors 大非链短·155k calls 176s self）。
    写读一致铁律：三处调用方须同 release 同改（写时 class_of(b)=X·读时 class_of(c)=X·row 命中）·否则 pair-rate 失配死信。
    """
    ancestors = ancestor_map.get(ref)
    if not ancestors:
        return ref   # 冷启动退化·无 IS_A 祖先→class = token 自身
    # x 是最近祖先 ⟺ ancestors 中无 y≠x 使 x ∈ ancestor_map[y]（y 比 x 深→x 非最近）
    # bit-identical 优化（MED-2 perf #1147·镜像 set_lca）：原逐 x 扫全 ancestors `for y: y!=x and x in anc[y]`
    # = O(|ancestors|²)·T-L1a 30k IS_A 深 closure → ancestors 大非链短 → 155k calls 176s self。改 anc 并集：
    # ancestor_union = ∪_{y∈ancestors} anc[y]·x∉union ⟺ 无 y∈ancestors 使 x∈anc[y]（ancestor_map 无自环·
    # isa_ancestor_map child==parent continue·故 y==x 时 x∉anc[x]→union 不含 x 自身·与原 `y!=x and ...` 逐字等价）。
    # O(|ancestors|²) → O(|ancestors|·|anc|)。
    ancestor_union: set[ConceptRef] = set()
    for y in ancestors:
        ancestor_union |= ancestor_map.get(y, set())
    candidates = [x for x in ancestors if x not in ancestor_union]
    if not candidates:
        return min(ancestors)   # DAG 上不触发（ancestors 非空必有最具体）·纯互祖先 SCC 可达·退 min 守 bit-identical（#1147 review 证 OLD 同达）
    return min(candidates)   # NodeRef 升序 tiebreak（diamond 多候选·bit-identical）


def common_is_a_ancestor(ref_a: ConceptRef, ref_b: ConceptRef,
                         ancestor_map: dict[ConceptRef, set[ConceptRef]]) -> ConceptRef | None:
    """两 ref 的最近共同祖先（LCA·最具体共同抽象·S3 D3·钥匙①第二刀抽象上卷用）。

    LCA = ref_a 与 ref_b 的共同 IS_A 祖先中**最具体**的（离 ref 最近·标准 LCA·非最抽象）。
    上卷深度=LCA（结构涌现·非硬编码词性·§8.1c 合规）。

    **偏序实现（无须 chain_len）**：x ∈ 共同祖先集 common 是 LCA ⟺ common 中无其他节点 y 使 x 是 y 的
    祖先（即无 y 比 x 更具体更深·x 在 common 中最深最具体）。x 是 y 祖先 ⟺ x ∈ ancestor_map[y]
    （y 的祖先集含 x·y 沿 IS_A 可达 x·y 比 x 更具体）。

    返 LCA ConceptRef | None（无共同祖先→None·两 ref 无 IS_A 抽象交集）。diamond 多 LCA 候选（不可比
    共同祖先）→ NodeRef 自然序升序取首（bit-identical tiebreak·同 closure.py:90 sorted 范式·非"唯一正确"）。

    铁律：纯整数 / 确定性（NodeRef 升序 tiebreak·bit-identical）/ 幂等（纯读 ancestor_map·重复同果）。
    诚实边界：O(|common|·|anc|)（MED-2 perf #1147·原 O(|common|²)·镜像 set_lca/nearest_isa_ancestor）/ LCA 是结构查询非语义理解 / 无共同祖先→None。
    """
    assert_int(ref_a[0], ref_a[1], ref_b[0], ref_b[1],
               _where="common_is_a_ancestor.refs")
    anc_a = ancestor_map.get(ref_a, set())
    anc_b = ancestor_map.get(ref_b, set())
    common = anc_a & anc_b
    if not common:
        return None   # 无共同 IS_A 祖先（两 ref 无抽象交集·上卷到顶无果）
    # LCA = common 中最具体的（无 common 其他节点是其后代=更深更具体）
    # x 是 LCA ⟺ ¬∃ y∈common, y≠x, x ∈ ancestor_map[y]（x 是 y 祖先·y 比 x 深→x 非 LCA）
    # bit-identical 优化（MED-2 perf #1147·镜像 set_lca/nearest_isa_ancestor）：原逐 x 扫全 common = O(|common|²)·
    # 改 anc 并集：ancestor_union = ∪_{y∈common} anc[y]·x∉union ⟺ 无 y∈common 使 x∈anc[y]（ancestor_map 无自环·
    # 故 y==x 时 x∉anc[x]→union 不含 x 自身·与原 `y!=x and ...` 逐字等价）。O(|common|²) → O(|common|·|anc|)。
    ancestor_union: set[ConceptRef] = set()
    for y in common:
        ancestor_union |= ancestor_map.get(y, set())
    candidates = [x for x in common if x not in ancestor_union]
    if not candidates:
        return None   # DAG 上不触发（common 非空必有最具体节点）·纯互祖先 SCC（a↔b 互祖先·candidates 空）可达此 fallback·返 None 守 bit-identical（#1147 review 证 OLD 同达·cycle-cleaned DAG #1133 实际不可达）
    return min(candidates)   # NodeRef 升序 tiebreak（diamond 多候选·bit-identical）


def set_lca(tokens: list[ConceptRef],
            ancestor_map: dict[ConceptRef, set[ConceptRef]]) -> ConceptRef | None:
    """多 token 集合的最近共同抽象祖先（集 LCA·**含自身 closure·集交集**·S3 第二刀 Interp2 抽象聚类用）。

    **与 common_is_a_ancestor 的关键差异（解 pairwise-reduce drift bug）**：
    - common_is_a_ancestor(ref_a, ref_b) 用 anc[a] ∩ anc[b]·**不含 a/b 自身**（传递闭包不含自环）。
    - 若用 pairwise reduce 算多 token LCA：LCA(t1,t2,t3) = LCA(LCA(t1,t2), t3)·中间 LCA(LCA(t1,t2)=r12, t3)
      把抽象节点 r12 当后代查·anc[r12] 不含 r12 → 当 r12 本是 t3 的 LCA 时**误卷到 r12 的更上祖先**。
      例：猫/狗 IS_A 动物·狐狸 IS_A 动物·动物 IS_A 生物。pairwise: LCA(猫,狗)=动物·
      LCA(动物,狐狸)=anc[动物]∩anc[狐狸]={生物}∩{动物,生物}={生物}→生物（**错·应动物**）。
    - **set_lca 用 abstract_closure(t) = {t} ∪ anc[t]**（含自身）·common = 全 closure 交集·
      LCA = common 中最深·上述例 closure(猫)∩closure(狗)∩closure(狐狸) = {动物,生物}∩... = {动物,生物}·
      最深=动物（生物是动物祖先·排除）→ **动物（正确）**。

    用途：S3 第二刀 Interp2 增量 LCA 聚类·簇内 slot p 已聚 token 集 + 候选 sample token·求集合 LCA·
    非 None → joinable。can_join 不维护"当前 LCA"作后代查（避 r12 当后代 bug）·维护 token 集合·
    query-time 调本 helper。

    返 LCA ConceptRef | None（tokens 空 / 无共同 closure → None）。diamond 多 LCA 候选 NodeRef 升序 tiebreak。

    铁律：纯整数（closure/交集/最深判定全 ConceptRef 整数二元组·无浮点）/ 确定性（NodeRef 升序 tiebreak·
    bit-identical）/ 幂等（纯读 ancestor_map·重复同果）。
    诚实边界：O(|tokens|·|common|·|anc|)（MED-2 perf #1133/#1136·原 O(|tokens|·|common|²)·|common| 随 ancestor_map
    变大膨胀 → T-L1a 1464s·改 anc 并集最深判定·见下）/ 单 token LCA = 自身
    （closure 含自身·最深=自身）/ 无共同祖先→None。
    """
    if not tokens:
        return None
    # abstract_closure(t) = {t} ∪ anc[t]·含自身（解 pairwise-drift）
    common: set[ConceptRef] | None = None
    for t in tokens:
        closure = {t} | ancestor_map.get(t, set())
        if common is None:
            common = set(closure)
        else:
            common &= closure
        if not common:
            return None   # 无共同 closure（含自身）→ 无共同抽象交集
    assert common is not None   # tokens 非空 → common 至少经历过一次赋值
    # common 中最深（最具体）：x 是 LCA ⟺ x 不是 common 中任何 y 的祖先（y 比 x 深·同 common_is_a_ancestor 偏序）
    # bit-identical 优化（MED-2 perf #1133/#1136）：原逐 x 扫全 common `any(y!=x and x in anc[y] for y in common)`
    # = O(|common|²·|anc|)·|common| 随 ancestor_map 增大膨胀（307k IS_A → 深 closure）→ T-L1a 1464s。
    # 改 anc 并集：ancestor_union = ∪_{y∈common} anc[y]·x∉union ⟺ 无 y∈common 使 x∈anc[y]（ancestor_map
    # 无自环·isa_ancestor_map child==parent continue·故 y==x 时 x∉anc[x]→union 不含 x 自身·与原 `y!=x and ...`
    # 逐字等价·候选集与 min(tiebreak) 全一致）。O(|common|²·|anc|) → O(|common|·|anc|)。
    ancestor_union: set[ConceptRef] = set()
    for y in common:
        ancestor_union |= ancestor_map.get(y, set())
    candidates = [x for x in common if x not in ancestor_union]
    if not candidates:
        return None   # DAG 上不触发（common 非空必有最具体节点）·纯互祖先 SCC（a↔b 互祖先·candidates 空）可达此 fallback·返 None 守 bit-identical（#1147 review 证 OLD 同达·cycle-cleaned DAG #1133 实际不可达）
    return min(candidates)   # NodeRef 升序 tiebreak（diamond 多候选·bit-identical）


# ===== Phase B 种子床扩（§十四-bis·mereology part-of 预序闭包 + PURE_ALIAS 等价闭包） =====
# 两 builder 皆 self-gate（入口 `if not gates.<MODE>: return {}`·default OFF·bit-identical）·无生产 caller
# （基建·consumer 落 Phase D/E/F）·诚实 seed-bed algebra（确定性派生·零学习内容·学习验 floor Phase F）。


def build_mereology_ancestor_map_external(backend, *, space_id: int
                                          ) -> dict[ConceptRef, set[ConceptRef]]:
    """MEREOLOGY part-of 祖先图（**仅外部源① ConceptNet**·Phase B §十四-bis B.1·镜像 build_isa_ancestor_map_external）。

    与 build_isa_ancestor_map_external 的差异：``edge_type=EDGE_MEREOLOGY``（part→whole·非 IS_A child→parent）。
    engine = isa_ancestor_map（graph_algebra·edge-type-agnostic·`_et` 忽略·SCC 凝聚内联处理 part-of 环·
    无 break_back_edges·strategy 2 低摩擦）。双滤 source=SOURCE_CONCEPTNET+epistemic=EPI_STRUCTURED（同 IS_A·
    anti-self-proving·现 mereology cue 路径 deferred 故 vacuous-now + forward-sound）。

    **bit-identical**：MEREOLOGY_CLOSURE_MODE default OFF→self-gate 返 {}→逐字现状。无 ConceptNet mereology 边
    （CI 无文件）→ 返空 dict（诚实降级）。consumer（whole_of）落 Phase D/E。

    返 {part: whole 集}（仅外部源·空 dict = 该 space 无外部 mereology 边）。纯读 filter·不改边·幂等。
    """
    assert_int(space_id, _where="build_mereology_ancestor_map_external.space_id")
    if not getattr(gates, "MEREOLOGY_CLOSURE_MODE", False):
        return {}   # self-gate·default OFF·bit-identical（基建·无 consumer）
    rows = backend.select("edge", where={
        "space_id_from": space_id, "edge_type": EDGE_MEREOLOGY,
        "source": SOURCE_CONCEPTNET, "epistemic_origin": EPI_STRUCTURED,
    })
    edges = [((r["space_id_from"], r["local_id_from"]),
              (r["space_id_to"], r["local_id_to"]),
              EDGE_MEREOLOGY, None) for r in rows]
    ancestor_map, _fell_back = isa_ancestor_map(edges)
    return ancestor_map


def whole_of(ancestor_map: dict[ConceptRef, set[ConceptRef]],
             ref: ConceptRef) -> ConceptRef | None:
    """ref 的 MEREOLOGY 最近 whole（最具体·最深·镜像 nearest_isa_ancestor·**冷启动返 None 非 ref**·Phase B §十四-bis B.1）。

    **mereology-specific 冷启动**：无 whole → 返 **None**（part 非自身 whole·异 nearest_isa_ancestor 的 return-ref·
    审1 MED-6）。consumer 须 handle None（whole_of 零 caller·Phase B 首个 reader·无现状 break）。

    算法镜像 nearest_isa_ancestor（最深 whole·ancestor_union O(|anc|·|anc|) perf #1147）·diamond 多候选 NodeRef
    升序 tiebreak（bit-identical）。
    """
    ancestors = ancestor_map.get(ref)
    if not ancestors:
        return None   # mereology 冷启动：无 whole→None（part 非自身 whole·异 nearest_isa_ancestor return ref）
    ancestor_union: set[ConceptRef] = set()
    for y in ancestors:
        ancestor_union |= ancestor_map.get(y, set())
    candidates = [x for x in ancestors if x not in ancestor_union]
    if not candidates:
        return min(ancestors)   # SCC 退化·min tiebreak bit-identical（mirror nearest_isa_ancestor）
    return min(candidates)


def build_pure_alias_closure_external(backend, *, space_id: int
                                      ) -> dict[ConceptRef, set[ConceptRef]]:
    """PURE_ALIAS 等价类（**外源-only EPI_STRUCTURED**·Phase B §十四-bis B.2·transitive_closure 首个 live caller）。

    select REFERS_TO subtype=PURE_ALIAS + epistemic==EPI_STRUCTURED（**positive 外源滤·非 source-agnostic 否定**·
    mirror IS_A positive 双滤精神·审1/审2 承重修）：boot alias_bridge/number_grounding(EPI_STRUCTURED) +
    lemmatizer(refers_to EPI_STRUCTURED·来源① legit external) 入·cue alias_cue_pairs(EPI_CUE·Phase B 纠
    observe.py:331 误标后) 出 → **anti-self-proving**（cue 自证 alias 不混入等价类·防 floor 自证闭环）。

    **等价 vs 可达性命门修**（审2 #4）：PURE_ALIAS 边方向混合（boot 双向·observe/lemmatizer 单向）·
    transitive_closure 有向可达非对称 → **symmetrize 输入**（每 (a,b) 加 (b,a)）→ 双向闭包 = 真**等价类**
    （mutual reachability·PURE_ALIAS 语义本对称）。非改 write 侧（守 bit-identical）。

    **transitive_closure meta 命门**（审1 #3）：须建 meta={"subtype":PURE_ALIAS} 边（closure.py:59 purity_filter
    仅 meta 非 None 时跑·naive 复制 IS_A meta=None 会漏 METAPHOR）。select 预滤 subtype=PURE_ALIAS 是真守卫·
    purity_filter=closure_pure_refers_to 冗余 defense-in-depth。

    **bit-identical**：PURE_ALIAS_CLOSURE_MODE default OFF→self-gate 返 {}→逐字现状。consumer（activate_candidates
    读等价类替直边）落 Phase D/E（behavior change·gate ON·非本 Phase）。

    返 {ref: 等价类（含自身）}（仅外部源 EPI_STRUCTURED·空 dict = 无外部 PURE_ALIAS 边）。consumer：class = equiv.get(ref, {ref})。
    """
    assert_int(space_id, _where="build_pure_alias_closure_external.space_id")
    if not getattr(gates, "PURE_ALIAS_CLOSURE_MODE", False):
        return {}   # self-gate·default OFF·bit-identical（基建·无 consumer）
    rows = backend.select("edge", where={
        "space_id_from": space_id, "edge_type": EDGE_REFERS_TO,
        "subtype": SUBTYPE_PURE_ALIAS, "epistemic_origin": EPI_STRUCTURED,
    })
    # 建 meta-carrying 边（purity_filter 须 meta 非 None）+ symmetrize（PURE_ALIAS 对称·解方向混合→等价类）。
    edges: list = []
    for r in rows:
        a = (r["space_id_from"], r["local_id_from"])
        b = (r["space_id_to"], r["local_id_to"])
        meta = {"subtype": SUBTYPE_PURE_ALIAS}
        edges.append((a, b, EDGE_REFERS_TO, meta))
        edges.append((b, a, EDGE_REFERS_TO, meta))   # symmetrize·双向→等价类
    closure = transitive_closure(edges, types={EDGE_REFERS_TO},
                                 purity_filter=closure_pure_refers_to, include_direct=True)
    # 等价类 dict[ref, set[ref 含自身]]（mirror ancestor_map shape·consumer class=equiv.get(ref,{ref})）。
    equiv: dict[ConceptRef, set[ConceptRef]] = {}
    for a, b, _et in closure:
        equiv.setdefault(a, set()).add(b)
        equiv.setdefault(b, set()).add(a)
    for a in list(equiv):
        equiv[a].add(a)   # 自身入类（ref 有别名时）
    return equiv
