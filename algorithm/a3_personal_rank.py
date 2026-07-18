"""algorithm.a3_personal_rank — A3 PR 求解（决策9自建·依赖 storage+crosscut+numeric）。

§十五决策9 A3 系统内自建无一外包。PersonalRank 线性系统：

    (I − αA) x = (1−α) e

  A —— 行归一化邻接（A_ij = w_ij / Σ_k w_ik·dangling 行空）
  α —— 阻尼（teleport·有理·0<α<1·(I−αA) 严格对角占优 M-matrix·可逆良态）
  e —— 个性化/种子向量（语境偏置·多种子 e=Σ e_s）
  x —— rank 结果

**B1 主精确**（线性性零损失）：e=Σ e_s ⟹ x=Σ x_s（A 固定·线性叠加）。
  有理高斯消元（Rational 闭运算·纯整数·零浮点）。小子图用（HotZone 子图·n 小）。
  线性性杠杆：A 固定时可预算每种子 x_s·多次查询组合复用（缓存·Stage 4 接线）。
**B2 迭代兜底**（残差）：大 n 时 B1 有理系数膨胀 → 定点 FixedQuotient 迭代
  x_{t+1}=αA x_t+(1−α)e·残差 |x_t−x_{t+1}|<θ 止（cross_compare/compare_fq 零误差判停）。
**B3 LU cache**（perf round3·2026-07-13 落地·§十五决策9 B3）：重复解的 LU 分解优化。
  A 固定 -> M=(I−αA^T) forward 消元一次缓存（_lu_decompose·记录 L/U/P）·同 matrix 多 seed 复用
  （_lu_solve L 前代 + U 回代·O(n²)/seed）。exact Rational·L/U/P 与 _gauss_solve 同 forward 数学 -> bit-identical。
  接线在 solve_exact（首 solve 建 matrix._lu_cache·后续命中）。solve_lu 占位保留向后兼容（见该函数）。

residual NULL / B2 路径赋值（D1 落盘）：B1 奇异（零权图退化）→ 自动回退 B2。
纯整数·不依赖 numpy/scipy（§五库依赖）。只参考原理非搬代码。

NodeRef = tuple[int, int] = (space_id, local_id)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.constants import DEFAULT_K
from pure_integer_ai.crosscut.integer.valtypes import Rational, FixedQuotient
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.crosscut.integer.rational import ZERO, ONE, make, add, sub, mul, is_zero, sign
from pure_integer_ai.crosscut.integer import fixed_point
from pure_integer_ai.crosscut.integer.fixed_point import rational_div
from pure_integer_ai.crosscut.integer import compare as cmp

NodeRef = tuple[int, int]

# 默认 α（有理·§八 PR 邻接·α 自动满足良态·旧资产 ZERO_AI_PR_ALPHA）
DEFAULT_ALPHA_NUM = 85
DEFAULT_ALPHA_DEN = 100

# B2 迭代默认
DEFAULT_ITER_K = DEFAULT_K          # 定点精度位
DEFAULT_THETA_NUM = 1               # 残差阈值 θ = 1/B^k（与精度同阶）
DEFAULT_THETA_DEN_EXP = DEFAULT_K   # θ = 1/B^DEFAULT_THETA_DEN_EXP
DEFAULT_MAX_ITER = 1 << 12          # 4096 上限


class PRSingular(RuntimeError):
    """(I−αA) 奇异（零权图退化·dangling 全图无正权）·B1 失败·回退 B2。"""


@dataclass
class PRMatrix:
    """(I − αA) 稀疏有理矩阵（A 行归一化·α 有理·A 固定供多种子复用）。

    nodes      : 节点列表（index↔node·确定性排序）。
    index      : node → index。
    alpha      : Rational（0<α<1）。
    rows       : {i: [(j, A_ij: Rational), ...]}·行归一化（Σ A_ij=1·dangling 行空）。
    """

    nodes: list[NodeRef]
    index: dict[NodeRef, int]
    alpha: Rational
    rows: dict[int, list[tuple[int, Rational]]]
    # B3 LU cache（perf round3·2026-07-13·§十五决策9 B3 落地）：A 固定->M=(I−αA^T) forward 消元一次
    # 缓存 (L, U, P)·同 matrix 多 seed solve 复用（decompose O(n³) 一次·每 seed O(n²) 前代/回代）。
    # bit-identical：L/U/P 与 _gauss_solve 同 forward 数学（同主元选+同消元·exact Rational·解唯一->路径无关）。
    # lazy（首 solve_exact 建）·matrix 生命周期内 A 不变->无需失效。None=未建·(L,U,P)=已建·singular 不缓存。
    _lu_cache: Any = field(default=None, repr=False, compare=False)

    @property
    def n(self) -> int:
        return len(self.nodes)


def build_matrix(
    nodes: list[NodeRef],
    edges: list[tuple[NodeRef, NodeRef, int]],
    alpha: Rational,
) -> PRMatrix:
    """从边构建 PRMatrix（行归一化 A·alpha 有理）。

    edges : [(from, to, weight:int), ...]·weight = strength（纯整数·≥0）。
    alpha : Rational（0<α<1·良态）。dangling 行（无正权出边）→ 空行（A 行=0）。
    """
    assert_no_float(alpha.num, alpha.den, _where="build_matrix.alpha")
    if not (0 < sign(alpha) and sign(sub(alpha, ONE)) < 0):
        raise ValueError(f"alpha 须 ∈ (0,1): {alpha}")
    sorted_nodes = sorted(set(nodes))
    index = {n: i for i, n in enumerate(sorted_nodes)}
    # 累加权重
    wsum: dict[int, int] = {i: 0 for i in range(len(sorted_nodes))}
    acc: dict[int, dict[int, int]] = {i: {} for i in range(len(sorted_nodes))}
    for u, v, w in edges:
        if w <= 0:
            continue
        iu = index[u]
        iv = index[v]
        acc[iu][iv] = acc[iu].get(iv, 0) + w
        wsum[iu] += w
    rows: dict[int, list[tuple[int, Rational]]] = {}
    for i in range(len(sorted_nodes)):
        lst: list[tuple[int, Rational]] = []
        if wsum[i] > 0:
            for j in sorted(acc[i]):
                lst.append((j, make(acc[i][j], wsum[i])))
        rows[i] = lst  # dangling → 空列表
    return PRMatrix(sorted_nodes, index, alpha, rows)


# ---- B1 主精确：有理高斯消元 ----

def solve_exact(matrix: PRMatrix, e: dict[NodeRef, Rational]) -> dict[NodeRef, Rational]:
    """B1：有理高斯消元解 (I−αA)x=(1−α)e·返回 {node: Rational}（精确·零损失）。

    e : node → Rational（种子/语境向量·未列节点视为 0）。
    奇异（零权退化）→ PRSingular（调用方回退 B2）。
    """
    n = matrix.n
    if n == 0:
        return {}
    # M 构建 + forward 消元移至 _lu_decompose（B3 cache·同 matrix 多 seed 复用·见 _lu_decompose）。
    one_minus_alpha = sub(ONE, matrix.alpha)
    rhs: list[Rational] = [ZERO] * n
    for node, val in e.items():
        if node in matrix.index:
            rhs[matrix.index[node]] = mul(one_minus_alpha, val)
    lu = matrix._lu_cache
    if lu is None:
        lu = _lu_decompose(matrix)   # 建 M + forward 消元·cache (L, U, P)·PRSingular 不缓存
        matrix._lu_cache = lu
    L, U, P = lu
    _lu_solve(L, U, P, rhs, n)        # L 前代 + U 回代（紧凑循环·镜像 _gauss_solve rhs 处理）
    return {matrix.nodes[i]: rhs[i] for i in range(n)}


def _build_M(matrix: PRMatrix) -> list[list[Rational]]:
    """构建 dense (I − αA^T)（Rational）。

    前向 PersonalRank：rank 沿出边传播（seed 的出邻居得 rank）。
    PRMatrix.rows[j] = 出边 j->i（A_ji·行随机）·故 M[i][j] = δ_ij − α·A_ji（转置）。
    """
    n = matrix.n
    M: list[list[Rational]] = [[ZERO] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = ONE
    for j in range(n):
        for i, aji in matrix.rows.get(j, []):  # 出边 j->i·A_ji
            M[i][j] = sub(M[i][j], mul(matrix.alpha, aji))
    return M


def _lu_decompose(matrix: PRMatrix) -> tuple[list[list[Rational]], list[list[Rational]], list[int]]:
    """B3 LU：建 M=(I−αA^T)·forward 消元（首非零主元·无 rhs）·返 (L, U, P) 缓存。

    L = 单位下三角（multipliers）·U = 上三角（消元后 M·含行交换）·P = 行置换（P[i]=现位 i 的原行号）。
    M[P[i]] = (L@U)[i]（PA=LU）。_lu_solve 用紧凑前代/回代循环（非 ops-list replay·去 tuple 拆箱开销）。
    bit-identical：与 _gauss_solve 同 forward 数学（同主元选 + 同消元·Rational 结合律·解唯一序无关）·
    L/P 记录与 _gauss_solve 同 swap+elim（只组织成数组非 ops tuple）。奇异 -> PRSingular（不缓存·caller 回退 B2）。
    """
    n = matrix.n
    M = _build_M(matrix)
    P = list(range(n))
    L = [[ZERO] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = ONE
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if not is_zero(M[r][col]):
                piv = r
                break
        if piv == -1:
            raise PRSingular(f"(I−αA) 奇异 at col={col}（零权退化）")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            P[col], P[piv] = P[piv], P[col]
            # 部分 L 行交换：仅已算乘子（cols<col）随行换·守 L 对角 ONE 与上三角零结构不动
            # （全行换破对角·审1 round3 catch：一般矩阵 PA≠LU·但 PR M-matrix 永不触发此分支）
            for k in range(col):
                L[col][k], L[piv][k] = L[piv][k], L[col][k]
        pivval = M[col][col]
        for r in range(col + 1, n):
            if not is_zero(M[r][col]):
                factor = _rdiv(M[r][col], pivval)
                L[r][col] = factor
                for c in range(col, n):
                    M[r][c] = sub(M[r][c], mul(factor, M[col][c]))
    return (L, M, P)   # M = U


def _lu_solve(L: list[list[Rational]], U: list[list[Rational]], P: list[int],
              rhs: list[Rational], n: int) -> None:
    """B3 LU solve：P 置换 rhs + L 前代 + U 回代·结果写入 rhs。

    紧凑嵌套循环（非 ops-list replay·去 tuple 拆箱开销·profile 实测 sub 从 40s->~1-2s）。
    bit-identical：与 _gauss_solve 同数学（PA=LU·前代 y[i]=b[i]-Σ_{j<i}L[i][j]y[j]·
    回代 x[i]=(y[i]-Σ_{j>i}U[i][j]x[j])/U[i][i]·Rational 结合律序无关）。
    b[i]=rhs[P[i]]（置换）·前代/回代后 b[i]=x[i]（变量 i·行置换不改变量序）·写回 rhs[i]=b[i]。
    """
    b = [rhs[P[i]] for i in range(n)]          # 置换：b[i] = rhs[P[i]]
    for i in range(n):                          # 前代 L y = b（L 单位下三角）
        bi = b[i]
        Li = L[i]
        for j in range(i):
            lval = Li[j]
            if not is_zero(lval):
                bi = sub(bi, mul(lval, b[j]))
        b[i] = bi
    for i in range(n - 1, -1, -1):              # 回代 U x = y
        bi = b[i]
        Ui = U[i]
        for j in range(i + 1, n):
            uval = Ui[j]
            if not is_zero(uval):
                bi = sub(bi, mul(uval, b[j]))
        if is_zero(Ui[i]):
            raise PRSingular(f"(I−αA) 奇异 at back-sub row={i}")
        b[i] = _rdiv(bi, Ui[i])
    for i in range(n):                          # 写回：rhs[i] = x[i] = b[i]
        rhs[i] = b[i]



def _gauss_solve(M: list[list[Rational]], rhs: list[Rational], n: int) -> None:
    """原地有理高斯消元（部分主元·首非零元·Rational 精确）。结果写入 rhs。

    **参考实现（reference·B3 LU 落地后 solve_exact 不再调此·改走 _lu_decompose+_lu_solve cache 路径）**。
    保留供审：_lu_decompose forward 镜像本函数 forward（同主元选+同消元·存 L/U/P 不存 ops·不触 rhs）·
    _lu_solve 镜像本函数 rhs 处理（forward replay + back-sub）·parity 测试坐实两路同 exact 解。
    """
    for col in range(n):
        # 选主元：首非零行（Rational 精确·任意非零主元可·首非零简化）
        piv = -1
        for r in range(col, n):
            if not is_zero(M[r][col]):
                piv = r
                break
        if piv == -1:
            raise PRSingular(f"(I−αA) 奇异 at col={col}（零权退化）")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            rhs[col], rhs[piv] = rhs[piv], rhs[col]
        pivval = M[col][col]
        for r in range(col + 1, n):
            if not is_zero(M[r][col]):
                factor = _rdiv(M[r][col], pivval)
                for c in range(col, n):
                    M[r][c] = sub(M[r][c], mul(factor, M[col][c]))
                rhs[r] = sub(rhs[r], mul(factor, rhs[col]))
    # 回代
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for c in range(i + 1, n):
            s = sub(s, mul(M[i][c], rhs[c]))
        if is_zero(M[i][i]):
            raise PRSingular(f"(I−αA) 奇异 at back-sub row={i}")
        rhs[i] = _rdiv(s, M[i][i])


def _rdiv(a: Rational, b: Rational) -> Rational:
    """有理除 a/b = a·(1/b)（精确·Rational 闭运算）。b≠0。"""
    if is_zero(b):
        raise ZeroDivisionError(f"_rdiv: b=0")
    return make(a.num * b.den, a.den * b.num)


def solve_exact_multi(matrix: PRMatrix, seeds: list[NodeRef]) -> dict[NodeRef, Rational]:
    """B1 线性性：x = Σ x_s（每种子 indicator e_s·A 固定·线性叠加零损失）。

    等价于 solve_exact(combined e=Σ e_s)·但暴露 per-seed 解供缓存复用（A 固定杠杆）。
    """
    acc: dict[NodeRef, Rational] = {n: ZERO for n in matrix.nodes}
    for s in seeds:
        e_s = {s: ONE}
        xs = solve_exact(matrix, e_s)
        for node, val in xs.items():
            acc[node] = add(acc[node], val)
    return acc


# ---- B2 迭代兜底：定点 FixedQuotient ----

def solve_iterative(
    matrix: PRMatrix,
    e: dict[NodeRef, Rational],
    *,
    k: int = DEFAULT_ITER_K,
    theta_num: int = DEFAULT_THETA_NUM,
    theta_den_exp: int = DEFAULT_THETA_DEN_EXP,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[NodeRef, FixedQuotient]:
    """B2：定点迭代 x_{t+1}=αA x_t+(1−α)e·残差 |x_t−x_{t+1}|<θ 止。

    返回 {node: FixedQuotient}（近似·error_bound < 1/B^k）。
    θ = theta_num / B^theta_den_exp（默认 1/B^k·与精度同阶）。
    收敛保证：α<1 + A 行随机 → 压缩映射·必收敛。
    """
    assert_int(k, theta_num, theta_den_exp, max_iter, _where="solve_iterative")
    n = matrix.n
    if n == 0:
        return {}
    base = _base(k)
    theta = make(theta_num, base ** theta_den_exp)
    theta_fp = _to_fp(theta, k)  # 定点化阈值（_fq_lt 须 FixedQuotient）
    # 定点化 α / (1−α) / e·前向 PersonalRank（rank 沿出边传播）
    alpha_fp = _to_fp(matrix.alpha, k)
    one_minus_alpha_fp = _to_fp(sub(ONE, matrix.alpha), k)
    # 入边视图：in_of[i] = [(j, A_ji)] 出边 j→i·供 x_new[i] += α·A_ji·x[j]
    in_of: dict[int, list[tuple[int, FixedQuotient]]] = {i: [] for i in range(n)}
    for j in range(n):
        for i, aji in matrix.rows.get(j, []):
            in_of[i].append((j, _to_fp(aji, k)))
    e_fp: list[FixedQuotient] = [_to_fp(e.get(matrix.nodes[i], ZERO), k) for i in range(n)]
    # x_0 = (1−α)e（teleport 初值）
    x: list[FixedQuotient] = [fixed_point.mul(one_minus_alpha_fp, e_fp[i]) for i in range(n)]
    for _ in range(max_iter):
        x_new: list[FixedQuotient] = [None] * n  # type: ignore[list-item]
        for i in range(n):
            # x_new[i] = (1−α)·e[i] + α·Σ_{j→i} A_ji·x[j]（前向·转置）
            s = fixed_point.mul(one_minus_alpha_fp, e_fp[i])
            for j, aji_fp in in_of[i]:
                term = fixed_point.mul(fixed_point.mul(alpha_fp, aji_fp), x[j])
                s = fixed_point.add(s, term)
            x_new[i] = s
        # 残差 max_i |x_new[i] − x[i]|（compare_fq 零误差判停）
        below = True
        for i in range(n):
            d = fixed_point.sub(x_new[i], x[i])
            # |d| < θ ⟺ −θ < d < θ
            if not (_fq_lt(d, theta_fp) and _fq_lt(_fq_neg(d), theta_fp)):
                below = False
                break
        x = x_new
        if below:
            break
    return {matrix.nodes[i]: x[i] for i in range(n)}


def _base(k: int) -> int:
    from pure_integer_ai.crosscut.integer.constants import BASE
    return BASE


def _to_fp(r: Rational, k: int) -> FixedQuotient:
    """Rational → FixedQuotient（定点 longdiv·精度 k）。"""
    return rational_div(r, ONE, k)


def _fq_neg(a: FixedQuotient) -> FixedQuotient:
    """定点负（守 0≤r<b 不变量：r>0 时 M'=−M−1, r'=b−r）。"""
    if a.r == 0:
        return FixedQuotient(-a.M, 0, a.k, a.b)
    return FixedQuotient(-a.M - 1, a.b - a.r, a.k, a.b)


def _fq_lt(a: FixedQuotient, b: FixedQuotient) -> bool:
    """a < b（定点比序·同 k 同 b 比 M·否则回退交叉积·零误差）。"""
    return fixed_point.compare_fq(a, b) < 0


# ---- B3 LU defer ----

def solve_lu(matrix: PRMatrix, e: dict[NodeRef, Rational]) -> dict[NodeRef, Rational]:
    """B3 LU 占位（向后兼容·§十五决策9 B3）。

    **B3 已落地**（perf round3·2026-07-13）：真实现走 solve_exact -> _lu_decompose（cache (L,U,P) on
    matrix._lu_cache）+ _lu_solve（L 前代 + U 回代）·非此函数。此 solve_lu 保留为 NotImplementedError
    占位供 test_pr_b3_lu_defer 向后兼容（测占位契约非真 B3）。调用 B3 用 solve_exact。
    """
    raise NotImplementedError(
        "B3 真实现走 solve_exact->_lu_decompose+_lu_solve（cache on matrix._lu_cache）·"
        "此占位保留供 test_pr_b3_lu_defer 向后兼容"
    )


# ---- 多种子 wrapper（A 固定·线性性·D1 落盘 B1→B2 回退） ----

@dataclass
class PRResult:
    """PR 结果（values + mode + exact 标志·诚实标注近似/精确）。"""

    values: dict  # node → Rational（B1）或 FixedQuotient（B2）
    mode: str     # "B1" / "B2"
    exact: bool   # True=B1 有理精确 / False=B2 定点近似


def personal_rank(
    matrix: PRMatrix,
    seeds: list[NodeRef],
    *,
    mode: str = "B1",
    **kwargs,
) -> PRResult:
    """多种子 wrapper：e = Σ e_s（每种子 indicator）·按 mode 求解。

    mode="B1"：solve_exact（线性性 x=Σx_s·零损失）。奇异 → 自动回退 B2（D1 落盘）。
    mode="B2"：solve_iterative（定点兜底·kwargs 透传 k/theta/max_iter）。
    线性性：B1 用 solve_exact_multi（per-seed 解可缓存·A 固定杠杆）。
    """
    if not seeds:
        return PRResult({n: ZERO for n in matrix.nodes}, mode, exact=(mode == "B1"))
    if mode == "B1":
        try:
            vals = solve_exact_multi(matrix, seeds)
            return PRResult(vals, "B1", exact=True)
        except PRSingular:
            # D1 落盘：B1 奇异（零权退化）→ 回退 B2
            e = {s: ONE for s in seeds}
            vals = solve_iterative(matrix, e, **kwargs)
            return PRResult(vals, "B2", exact=False)
    if mode == "B2":
        e = {s: ONE for s in seeds}
        vals = solve_iterative(matrix, e, **kwargs)
        return PRResult(vals, "B2", exact=False)
    raise ValueError(f"personal_rank: 未知 mode {mode!r}（B1/B2）")
