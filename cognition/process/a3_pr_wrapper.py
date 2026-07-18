"""cognition.process.a3_pr_wrapper — 模块2 A3 PR 多种子 wrapper（决策9自建·§十三B）。

PersonalRank 线性系统 (I−αA)x=(1−α)e·A 行归一化邻接（热区会话内固定）。
  - B1 主精确：有理高斯·线性性 e=Σe_s ⟹ x=Σx_s（A 固定零精度损失·§十三B 核心洞察）
  - B2 兜底：定点迭代 + 残差 ‖r‖∞（大 n 时·首版热区小用 B1·residual=None for B1）
  - B3 LU defer

wrapper 持 PRMatrix（A 固定）+ pr_cache（seed→x_s 单种子向量）+ 累积 x（Σx_s）。
  solve(seeds)    B1 多种子叠加（personal_rank·B1 奇异自动回退 B2·D1）
  add_seed(c)     扩张：x_s = cache or solve_single(c)；x += x_s（B4 逐个叠加零损失）
  remove_seed(c)  逐出：x −= cache[c]（精确 O(n)·不删缓存可能再加回）
  seed_rank(c)    当前累积 x 在 c 处的值（attractor 相干判据 x_c≥θ_coh 用·§十三A）

衔接条件③：PR 邻接装配读 effective_weight=strength×rate（模块7·H4）非裸 strength。
纯整数（Rational 闭运算）/ 确定性（节点自然序·bit-identical）。
诚实边界：PR 是结构传播非语义理解（x 高分=结构连通高非"语义相关"·stable≠correct）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.valtypes import Rational
from pure_integer_ai.crosscut.integer.rational import ZERO, ONE, make, add, sub, mul
from pure_integer_ai.algorithm.a3_personal_rank import (
    PRMatrix, build_matrix, solve_exact, PRSingular,
    DEFAULT_ALPHA_NUM, DEFAULT_ALPHA_DEN,
)
# solve_exact_multi 是 algorithm 层 live API（a3_personal_rank.py:177 定义 + :311 personal_rank 内调
# + test_stage2.py:524/567 真消费）·wrapper 内零调用（wrapper 自循环 solve_exact 填 per-seed cache·
# 非用 solve_exact_multi 合并返）·A3 设计 session 2026-07-07 删 wrapper-内死 import·algorithm 层不动。
from pure_integer_ai.config import gates
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO
from pure_integer_ai.storage.edge_store import SUBTYPE_OCCURRENCE
from pure_integer_ai.storage.experience_count import (
    read_effective_freq, DEFAULT_CTX_CODE, DEFAULT_SPEAKER_CODE,
)
from pure_integer_ai.storage.selection_pref_count import read_selection_pref_agg
from pure_integer_ai.cognition.process.effective_weight import effective_weight
from pure_integer_ai.cognition.shared.types import ConceptRef

NodeRef = ConceptRef

# PR seed 向量 freq 归一化分母（阶段8 落点 A·oracle 标定起点·独立于 THETA_FREQ·真训练 run 前校准）。
# _seed_weight：eff_freq=0 / backend=None → ONE（bit-identical）·eff_freq>0 → make(SCALE+eff_freq,SCALE)。
FREQ_SEED_SCALE = 1000

# PR seed 向量 selection_pref 归一化分母（S4 片2·selection_pref 维 dock·乘积 w_freq·w_sp 的 w_sp 分母）。
# _seed_weight：sp_agg=0 / gate OFF → w_sp ONE（bit-identical）·sp_agg>0 → make(SCALE+sp_agg,SCALE)。
# sp_agg = sum_base + sum_sp_tn（concept_a 总搭配次数·镜像 effective_freq 两源范式）·oracle 起点同 FREQ_SEED_SCALE。
SP_SEED_SCALE = 1000

# PR B2 大 n 切换阈值（matrix.n 超此 → B2 定点迭代替 B1·gate PR_B2_LARGE_N_MODE 守）。
# B1 O(n³)+Rational 膨胀在 n~500+ 变慢（实测 n=1000=30s·n^2.8）·阈值 512 守小 n 走 B1 精确（≤~4s）·
# 大 n 走 B2 稀疏 O(E×iters)。gate default OFF → 不读此阈值·CI 全 B1 bit-identical。
PR_B2_LARGE_N_THRESHOLD = 512

# PR B2 大 n 路径定点精度 k（迭代收敛阈值 θ = 1/B^k·B=2^30）。
# solve_iterative 默认 k=DEFAULT_K=8（θ=2^-240）·收敛需 ~1024 迭代过慢。PR 消费者只做**相对比较**
# （anti-collapse 方差 / 路径选择 / selection_pref）·值差远大于 2^-60·k=2（θ=2^-60≈1e-18·~256 迭代）
# 保 rank 序完整（A/B 实测 n=1000 top-20 Jaccard=1.000·vs B1 精确）·12x 快于 B1。
# k=1（2^-30≈1e-9·~128 迭代·22x）亦保序但 k=2 更鲁棒（防极近 rank tie）·取 2 平衡。
PR_B2_LARGE_N_K = 2

# PR 热区过滤（gate HOTZONE_MODE·perf·解全图 8677²·设计本意卷二:110 hotzone_adj radius）：
# A3PRWrapper.build 内 BFS k-hop 从 local_seeds 出发·matrix 缩到 k-hop 子图。k=2 依据 α²=0.72>>θ_coh=1/1000·
# 2-hop PR 可微·PRECEDES 线性 fanout 1-2 不爆·k=3 α³=0.61 边际小而 BFS 跳增数千。PR_MAX_NODES 硬上界防 BFS 爆炸。
# 超界按 BFS 近度（distance 升序）+ ConceptRef 自然序 tiebreak 截断（确定 bit-identical）。
HOTZONE_K = 2
PR_MAX_NODES = 2048


def _bfs_hotzone(seeds: list[NodeRef],
                 adj: dict[NodeRef, list[NodeRef]],
                 *, k: int, max_nodes: int) -> set[NodeRef]:
    """BFS k-hop from seeds（出边 only·PR 沿出边传播）·返 hot 节点集。

    seeds distance 0（必含·即使无出边）·distance 1=出邻居·distance 2=出邻居的出邻居。
    超 max_nodes 按 BFS 近度（distance 升序）+ ConceptRef 自然序 tiebreak 截断（确定 bit-identical·同 distance 取 ConceptRef 小者）。
    同层节点集去重（COOCCURS 无向但 PR 邻接 PRECEDES/CAUSES 有向·同 (u,v) 重复不扩 BFS·set 去重）。
    """
    if not seeds:
        return set()
    visited: dict[NodeRef, int] = {}
    frontier = sorted(set(seeds))   # distance 0·ConceptRef 序确定
    for n in frontier:
        visited[n] = 0
    for dist in range(1, k + 1):
        nxt: set[NodeRef] = set()
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in visited:
                    nxt.add(v)
        if not nxt:
            break
        for v in nxt:
            visited[v] = dist
        frontier = sorted(nxt)   # 下一层 ConceptRef 序
    if len(visited) > max_nodes:
        kept = sorted(visited.keys(), key=lambda n: (visited[n], n))[:max_nodes]
        return set(kept)
    return set(visited.keys())


def _fq_to_rational(fq: Any) -> Rational:
    """FixedQuotient → 精确 Rational（值 = M/B^k + r/(b·B^k) = (M·b+r)/(b·B^k)）。

    B2 迭代返回 FixedQuotient（定点近似）·转 Rational 供 PR 累积（线性叠加零损失·stub #6 修：
    旧版 make(int(vals.M),1) 丢 r/b/k/B 全部标度→数值全错·且两路不一致·生产永不触发但须诚实）。
    """
    from pure_integer_ai.crosscut.integer.constants import BASE
    bk = BASE ** fq.k
    return make(fq.M * fq.b + fq.r, fq.b * bk)


class A3PRWrapper:
    """A3 PR 多种子 wrapper（A 固定·cache·线性性零损失）。

    持 PRMatrix（热区会话内固定）+ pr_cache + 累积 x。memory_active=False 时 PR 邻接
    只读 {PRECEDES, CAUSES}（衔接条件②·记忆主导场景 OCCURRENCE 进邻接 defer Stage 5）。
    """

    def __init__(self, matrix: PRMatrix, *, alpha_num: int = DEFAULT_ALPHA_NUM,
                 alpha_den: int = DEFAULT_ALPHA_DEN,
                 backend: Any = None,
                 ctx_code: int = DEFAULT_CTX_CODE,
                 speaker_code: int = DEFAULT_SPEAKER_CODE) -> None:
        self.matrix = matrix
        self._cache: dict[NodeRef, dict[NodeRef, Rational]] = {}
        self._x: dict[NodeRef, Rational] = {n: ZERO for n in matrix.nodes}
        self.mode = "B1"
        self.exact = True
        self._alpha_num = alpha_num
        self._alpha_den = alpha_den
        # 阶段8 落点 A：freq 进 PR seed 向量数据源（read_effective_freq）+ 复合 key 桶（阶段6）。
        # backend=None → _seed_weight 退 ONE（bit-identical·无 backend / 冷启动退化）。
        self._backend = backend
        self._ctx_code = ctx_code
        self._speaker_code = speaker_code

    # ---- 构建 ----

    @classmethod
    def build(cls, subgraph_edges: list[dict[str, Any]], *,
              current_seq: int = 0,
              memory_active: bool = False,
              alpha_num: int = DEFAULT_ALPHA_NUM,
              alpha_den: int = DEFAULT_ALPHA_DEN,
              backend: Any = None,
              ctx_code: int = DEFAULT_CTX_CODE,
              speaker_code: int = DEFAULT_SPEAKER_CODE,
              seeds: list[NodeRef] | None = None) -> "A3PRWrapper":
        """从热区边集构建 wrapper（A 固定·邻接权重 = effective_weight·H4）。

        subgraph_edges：热区相关边（dict 行·含 PRECEDES/CAUSES/[OCCURRENCE REFERS_TO]）。
        memory_active=True 时 OCCURRENCE REFERS_TO 性质B 边进 PR 邻接（记忆主导场景·M8/F3）。
        seeds：gate HOTZONE_MODE ON 时 BFS k-hop 起点（local_seeds·出边 only·matrix 缩到 k-hop 子图）。
            None/空/OFF -> 全图 PR（现状 bit-identical）。reward 影响零（resolver 证 PR 不回流 path）。
        """
        assert_no_float(current_seq, alpha_num, alpha_den, _where="A3PRWrapper.build")
        alpha = make(alpha_num, alpha_den)
        nodes: set[NodeRef] = set()
        weighted: list[tuple[NodeRef, NodeRef, int]] = []
        adj: dict[NodeRef, list[NodeRef]] = {}   # 热区 BFS 出边邻接（PR-edge only·w>0）
        for e in subgraph_edges:
            et = e["edge_type"]
            in_pr = et in (EDGE_PRECEDES, EDGE_CAUSES)
            if et == EDGE_REFERS_TO and e.get("subtype") == SUBTYPE_OCCURRENCE:
                in_pr = memory_active   # 记忆主导场景才进邻接（defer Stage 5 默认关）
            if not in_pr:
                continue
            w = effective_weight(e, current_seq=current_seq)
            if w <= 0:
                continue   # 零权不进邻接（新 CAUSES 边 sn=tn=0·PRECEDES strength=1 w=1）
            u = (e["space_id_from"], e["local_id_from"])
            v = (e["space_id_to"], e["local_id_to"])
            nodes.add(u)
            nodes.add(v)
            weighted.append((u, v, w))
            adj.setdefault(u, []).append(v)
        # gate HOTZONE_MODE（perf·设计本意卷二:110 hotzone_adj radius·全图是 defer 意外态）：BFS k-hop 从 seeds
        # 出发缩 matrix·reward 影响零（resolver 证 PR 不回流 path·stepper.advance 不读 PR）。OFF/无 seeds -> 全图 bit-identical。
        if getattr(gates, "HOTZONE_MODE", False) and seeds:
            hot = _bfs_hotzone(seeds, adj, k=HOTZONE_K, max_nodes=PR_MAX_NODES)
            nodes = {n for n in nodes if n in hot}
            weighted = [(u, v, w) for u, v, w in weighted if u in hot and v in hot]
        matrix = build_matrix(sorted(nodes), weighted, alpha)
        return cls(matrix, alpha_num=alpha_num, alpha_den=alpha_den,
                   backend=backend, ctx_code=ctx_code, speaker_code=speaker_code)

    # ---- 求解 ----

    def solve(self, seeds: list[NodeRef]) -> dict[NodeRef, Rational]:
        """B1 多种子叠加：x = Σ x_s（personal_rank·线性性零损失·D1 B1→B2 回退）。

        重置累积 x 为 seeds 的叠加（热区刷新重置 e=e₀·无跨遍历累积·§十三A）。
        返回累积 x（{node: Rational}）。B1 奇异 → 回退 B2（mode/exact 标记）。

        诚实边界：B2 路径 residual property 仍恒 None（残差 ‖r‖∞ 安全网 defer·见 :247 property
        + §十三B L1166 修订注）。B2 触发两路：① gate PR_B2_LARGE_N_MODE ON + n>阈值（生产大 n 真路径·
        解 O(V³) 炸弹·非 theater）② B1 奇异回退（monkeypatch-only·(I−αA) M-matrix 严格对角占优数学上恒可逆·
        test_stage2.py:560 自承无法构造自然奇异）。两路同走 _solve_b2。
        B2 路径 _fq_to_rational 转 Rational 后 error_bound 丢失（self._x 内 B1/B2 值混血无 per-node 标记·
        生产大 n（gate ON）B2 主导·真混血 caller 须自查 self.mode/self.exact·A/B 实测 rank 序与 B1 一致）。
        gate default OFF：CI 全 B1 精确 bit-identical；生产大 n 翻 ON 走 B2 近似（用户"统计上做到即可"框架）。
        """
        for s in seeds:
            assert_int(s[0], s[1], _where="A3PRWrapper.solve.seed")
        self._x = {n: ZERO for n in self.matrix.nodes}
        self._cache.clear()
        if not seeds:
            self.mode, self.exact = "B1", True
            return self._x
        # 大 n 切 B2（perf·gate PR_B2_LARGE_N_MODE 守·CI default OFF 走 B1 bit-identical·生产大 n 翻 ON
        # → B2 稀疏定点迭代替 B1 O(n³)·解 O(V³) 密集高斯炸弹·设计 docstring 本意"大 n 兜底"接线）。
        # A/B 实测 B2 与 B1 rank 序一致（top-20 Jaccard=1.0·消费者相对比较→语义等价·值差~1/B^k）。
        if getattr(gates, "PR_B2_LARGE_N_MODE", False) and self.matrix.n > PR_B2_LARGE_N_THRESHOLD:
            self._x = self._solve_b2(seeds, k=PR_B2_LARGE_N_K)
            self.mode, self.exact = "B2", False
            return self._x
        try:
            # B1 线性性：per-seed 解（可缓存·A 固定杠杆·seed 权重 _seed_weight 落点 A）
            for s in seeds:
                xs = solve_exact(self.matrix, {s: self._seed_weight(s)})
                self._cache[s] = xs
                for n, v in xs.items():
                    self._x[n] = add(self._x[n], v)
            self.mode, self.exact = "B1", True
        except PRSingular:
            # D1 落盘：B1 奇异（零权退化）→ 回退 B2 定点迭代（_solve_b2·与 large-n 路径同函数）
            self._x = self._solve_b2(seeds)
            self.mode, self.exact = "B2", False
        return self._x

    def _solve_b2(self, seeds: list[NodeRef], *, k: int | None = None) -> dict[NodeRef, Rational]:
        """B2 定点迭代多种子（合并 e=Σ e_s 一次解·近似·_fq_to_rational 转 Rational）。

        k：定点精度（θ=1/B^k）。**None = solve_iterative 默认 k=DEFAULT_K=8**（PRSingular 回退守原行为
        bit-identical·不改 k）；**显式传 k=PR_B2_LARGE_N_K=2**（large-n 路径·PR 相对比较够·~256 迭代 vs
        默认 ~1024·12x 快·A/B 实测 rank 序与 B1 一致 Jaccard=1.0）。两路同函数·k 区分精度/速度权衡。
        返 dict[NodeRef, Rational]（FixedQuotient 经 _fq_to_rational 精确转换·误差<1/B^k）。
        """
        from pure_integer_ai.algorithm.a3_personal_rank import solve_iterative
        e = {s: self._seed_weight(s) for s in seeds}
        if k is not None:
            vals = solve_iterative(self.matrix, e, k=k, theta_den_exp=k)
        else:
            vals = solve_iterative(self.matrix, e)   # 默认 k=8（PRSingular 原行为·守 bit-identical）
        return {n: _fq_to_rational(vals[n]) for n in self.matrix.nodes}

    def _solve_single(self, c: NodeRef) -> dict[NodeRef, Rational]:
        """单种子解（cache miss 时算·A 固定）。大 n → B2 / 小 n → B1（奇异回退 B2）。"""
        if c in self._cache:
            return self._cache[c]
        # 大 n 切 B2（perf·gate 守·与 solve 同范式·防 add_seed cache miss 走 B1 solve_exact 炸）。
        if getattr(gates, "PR_B2_LARGE_N_MODE", False) and self.matrix.n > PR_B2_LARGE_N_THRESHOLD:
            xs = self._solve_b2([c], k=PR_B2_LARGE_N_K)
        else:
            try:
                xs = solve_exact(self.matrix, {c: self._seed_weight(c)})
            except PRSingular:
                from pure_integer_ai.algorithm.a3_personal_rank import solve_iterative
                xs = solve_iterative(self.matrix, {c: self._seed_weight(c)})
                # B2 返回 FixedQuotient·转精确 Rational（_fq_to_rational·stub #6 修·cache 标记近似）
                xs = {n: _fq_to_rational(fq) for n, fq in xs.items()}
        self._cache[c] = xs
        return xs

    def _seed_weight(self, c: NodeRef) -> Rational:
        """seed 权重 = freq 维 × selection_pref 维 乘积 dock（落点 A·S4 片2 selection_pref 维扩）。

        PR 公式 (I−αA)x=(1−α)e·e 是 seed 向量。乘子 dock 走 **权重缩放**（w_s 线性缩放 x_s·solve_exact
        e→x 线性·a3_personal_rank.py:130-134）·**不动 seed 节点集**·§8.4 乘子吸收进 PR 不单做·合规。

        **乘积 dock（S4 片2·决断 2 选项 A）**：w = mul(w_freq, w_sp)（rational.mul 闭运算·线性性保持）。
            w_freq = ONE if eff_freq<=0 else make(FREQ_SEED_SCALE + eff_freq, FREQ_SEED_SCALE)   # 落点 A·freq 维
            w_sp   = ONE if sp_agg<=0  else make(SP_SEED_SCALE + sp_agg, SP_SEED_SCALE)         # S4 片2·selection_pref 维
            sp_agg = sum_base + sum_sp_tn（concept_a 总搭配次数·镜像 effective_freq 两源范式·不含 sp_sn 单独·
                     sp_sn 是 sp_tn 成功子集·镜像 eff_freq 不含 e_sn 单独）
        任一维为 ONE 时乘积不变（gate OFF / sp_agg=0 / eff_freq=0 → 该维 ONE → bit-identical 退化）。

        **seed=struct_ref 数据真空（S4 Plan agent 纠偏·doc L264）**：struct_ref 是段标签非 token·
            _inject_base_freq 不注（formal_train.py:536）→ eff_freq=0 → w_freq ONE；selection_pref_count
            无 struct_ref 行（struct_ref 非动词 token·无搭配）→ sp_agg=0 → w_sp ONE。故生产 seed w=ONE·bit-identical。
            **真生效路径 = attractor 扩张**（ATTRACTOR_MODE ON reward 阶段·dag_path.py:228-233 把 path 内 token
            concept 加进 e_set·add_seed → _solve_single → _seed_weight(token_concept)·token 有 eff_freq + sp_agg）。
            freq 维与 selection_pref 维同走 attractor 扩张路径生效·与 seed=struct_ref 无关·**seed=struct_ref
            不阻塞乘子 dock**（doc L264 "seed=struct_ref 真阻塞 / 强 dock 破 episode 遍历" 是误判·S4 纠偏）。

        **两层正交（决断 2·设计故意）**：PR 侧 w_sp 聚合丢 argument_class 区分（粗筛·"高搭配动词传播力强"）·
            生成侧 read_selection_pref_count pair rate 精查（精查·defer 独立线）。A 丢区分不是缺陷——B（pair-key
            当 seed）撞 PR matrix 单 NodeRef 假设·pair-key seed 的 x_s 仍落 concept_a 节点·argument_class 无独立表达。

        §8.4（乘子吸收进 PR 不单做）：freq + selection_pref 进 seed 向量 e（PR 计算输入）非 attractor 字段·合规。
        §8.5（边 schema 不预留乘子字段）：不动 edge schema / effective_weight 签名·乘子在 _seed_weight 计算层·合规。
        reward CAUSES-only：selection_pref_count 是独立统计表·不进 effective_weight:82 assert 闭集·合规。
        bit-identical：gate SELECTION_PREF_DOCK_MODE 默认 OFF → w_sp 恒 ONE → w = w_freq（落点 A 不变·CI===生产）。
        """
        if self._backend is None:
            return ONE
        # —— freq 维（落点 A·既有） ——
        eff_freq = read_effective_freq(self._backend, c,
                                       ctx_code=self._ctx_code,
                                       speaker_code=self._speaker_code,
                                       observe_mode=bool(getattr(gates, "FREQ_OBSERVE_MODE", False)))
        w = ONE if eff_freq <= 0 else make(FREQ_SEED_SCALE + eff_freq, FREQ_SEED_SCALE)
        # —— selection_pref 维（S4 片2·乘积 dock·gate SELECTION_PREF_DOCK_MODE 守） ——
        # gate OFF → w_sp 恒 ONE → w 不变（bit-identical·落点 A 退化·CI===生产）。
        # sp_agg = sum_base + sum_tn（concept_a 总搭配次数·不含 sp_sn 单独·镜像 eff_freq 不含 e_sn 单独）。
        # observe_mode（方案3 tn路 B5 β_arith 修法·gate SP_OBSERVE_MODE 守）：ON→sum_tn=sum_sp_observe_tn（避 reward 染色）/
        # OFF→sum_tn=sum_sp_tn（既有 bit-identical·β_arith 染色）。对偶 B4 w_freq（read_effective_freq observe_mode）。
        if getattr(gates, "SELECTION_PREF_DOCK_MODE", False):
            _sum_base, _sum_sn, sum_tn = read_selection_pref_agg(
                self._backend, c, observe_mode=bool(getattr(gates, "SP_OBSERVE_MODE", False)))
            sp_agg = _sum_base + sum_tn
            if sp_agg > 0:
                w = mul(w, make(SP_SEED_SCALE + sp_agg, SP_SEED_SCALE))
        return w

    def add_seed(self, c: NodeRef) -> None:
        """扩张：x += x_c（B4 逐个叠加·线性性零损失·§十三B·热区小重算廉价）。"""
        assert_int(c[0], c[1], _where="A3PRWrapper.add_seed")
        if c not in self.matrix.index:
            return   # 热区外 c 的 x_c 无定义→e 限热区（近视·§十三A 自然发散上界）
        xs = self._solve_single(c)
        for n, v in xs.items():
            self._x[n] = add(self._x[n], v)

    def remove_seed(self, c: NodeRef) -> None:
        """逐出：x −= cache[c]（精确 O(n)·不删缓存可能再加回）。"""
        assert_int(c[0], c[1], _where="A3PRWrapper.remove_seed")
        xs = self._cache.get(c)
        if xs is None:
            return
        for n, v in xs.items():
            self._x[n] = sub(self._x[n], v)

    def seed_rank(self, c: NodeRef) -> Rational:
        """当前累积 x 在 c 处的值（attractor 相干判据 x_c≥θ_coh·§十三A）。"""
        return self._x.get(c, ZERO)

    def snapshot(self) -> dict[NodeRef, Rational]:
        """当前累积 x 快照（item3 缺漏4·attractor 扩张后 dag_path 重设 pr_vector 用·F5 聚合读扩张后 x）。"""
        return dict(self._x)

    @property
    def residual(self) -> Rational | None:
        """残差 ‖r‖∞（仅 B2 路径非 None·B1 精确路径 None·D1 落盘）。

        首版 B1 精确无残差（热区小·B1 够用）·B2 残差安全网 defer 大 n 场景。
        """
        return None   # B1 精确路径·诚实 defer B2 残差
