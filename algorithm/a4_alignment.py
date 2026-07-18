"""algorithm.a4_alignment — A4 结构对齐（LCS·pairwise 折叠·决策9自建）。

§十五决策9 A4 系统内自建。结构对齐 = 在纯整数符号序列上求最长公共子序列（LCS），
pairwise 折叠（progressive alignment）做多序列对齐。

  lcs(a, b)           —— 两序列 LCS（DP O(|a|·|b|)·确定性 tiebreak：a 的较早位置优先）
  lcs_score(a, b)     —— LCS 长度（对齐得分·纯整数）
  pairwise_fold(seqs) —— 多序列对齐：以最长 seq 为种子·依次 LCS 折叠成 consensus

A4 用途（§十三子点C / 卷二模块3）：结构映射对齐——如 role_seq 序列对齐（结构合仅 AND
用 _lcs_alignment·汇聚按头唯一不退化）。序列元素是 symbol_id（纯整数·节点 ref / role 标签）。

确定性：DP tiebreak 固定（a 较早位置 + b 较早位置）·bit-identical。
纯整数·不依赖外部库。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def lcs_score(a: list[int], b: list[int]) -> int:
    """LCS 长度（对齐得分·纯整数 DP）。"""
    for x in a:
        assert_int(x, _where="lcs_score.a")
    for x in b:
        assert_int(x, _where="lcs_score.b")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # 滚动数组 DP
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev, cur = cur, prev
    return prev[m]


def lcs(a: list[int], b: list[int]) -> list[int]:
    """两序列的一个 LCS（确定性·tiebreak：a 较早位置 + b 较早位置优先）。

    回溯固定方向：优先往上（a 较早）·再往左（b 较早）·匹配时取。
    """
    for x in a:
        assert_int(x, _where="lcs.a")
    for x in b:
        assert_int(x, _where="lcs.b")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    # DP 表（n+1)×(m+1)
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = dp[i]
        prow = dp[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                row[j] = prow[j - 1] + 1
            else:
                row[j] = prow[j] if prow[j] >= row[j - 1] else row[j - 1]
    # 回溯（确定性 tiebreak）
    out: list[int] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1  # 优先 a 较早位置
        else:
            j -= 1
    out.reverse()
    return out


def pairwise_fold(
    seqs: list[list[int]],
    *,
    order: list[int] | None = None,
) -> tuple[list[int], int]:
    """多序列对齐：pairwise 折叠（progressive alignment）。

    以 order 指定折叠序（默认：按 (len 降序, 序列自身) 排序·确定性）。
    consensus = seqs[order[0]]；依次 consensus = lcs(consensus, seqs[order[k]])。
    返回 (consensus, score)·score = Σ pairwise LCS 长度（折叠累计·对齐质量度量）。

    pairwise 折叠是 A4 自建实现（决策9·不外包 multiple-sequence alignment 库）。
    """
    if not seqs:
        return [], 0
    for s in seqs:
        for x in s:
            assert_int(x, _where="pairwise_fold.seq")
    if order is None:
        # 按 (len 降序, 序列 tuple) 排序·确定性（同长按内容字典序）
        order = sorted(range(len(seqs)),
                       key=lambda k: (-len(seqs[k]), tuple(seqs[k])))
    if len(order) != len(seqs):
        raise ValueError(f"order 长度须等于 seqs 数: {len(order)} vs {len(seqs)}")
    consensus = list(seqs[order[0]])
    score = 0
    for k in order[1:]:
        s = seqs[k]
        consensus = lcs(consensus, s)
        score += len(consensus)
    return consensus, score


def alignment_matches(a: list[int], b: list[int]) -> list[tuple[int, int]]:
    """两序列 LCS 的对齐位置对（i, j）·确定性 tiebreak（同 lcs 回溯）。

    供结构映射对齐消费（卷二模块3·A4）：返回匹配的 (a 位置, b 位置) 对。
    """
    for x in a:
        assert_int(x, _where="alignment_matches.a")
    for x in b:
        assert_int(x, _where="alignment_matches.b")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs
