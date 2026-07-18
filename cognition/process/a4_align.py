"""cognition.process.a4_align — 模块3 A4 结构映射对齐（coverage_overlap 合质量）。

A4 范畴（§十五决策9 + §十三子点 C 结构合）= pairwise LCS 折叠 + coverage_overlap 合质量 + AND 合语义。
范畴满足方式（A4 设计 session 2026-07-07 核证·2 对抗审 CONFIRMED）：
  - LCS 折叠原语 = algorithm.a4_alignment（pairwise_fold/lcs/lcs_score·件4 lang_structure_align
    discover-time 真 live·formal_train caller·跨样本建骨架）
  - coverage_overlap 合质量 = 本模块（judge J1 + dag_path goal + structure_discover 形状识别·3 真 caller）
  - AND 合语义 = HeadStepper（a2_stepper·PRECEDES AND 全前驱 order_index 到齐·dag_path live）

**a4_align 函数（process-time 汇聚点 fold）已删 2026-07-07**：生产零 caller（theater·dag_path:225
汇聚点裸 append ref 不调 fold）+ 4 条潜在消费路径全 DEAD（judge/generate/discover/reward 都不消费
fold 结果·generate 逐 unit 读 graph.read_role_seq 绕 fold）+ process-time 汇聚点 consensus 折叠首版
无消费场景。范畴满足不依赖 a4_align 函数（由 coverage_overlap + 件4 + HeadStepper 三处真活代码 cover）。
未来若需 process-time 汇聚点 consensus（如 §8.8 L537 PR 共属抽象落地后）·可直接调
algorithm.a4_alignment pairwise_fold + 本模块 coverage_overlap 重建（底层原语都在·重建廉价）·非恢复旧 a4_align。

文件名 a4_align 保留作 A4 范畴模块名（coverage_overlap 是 A4 范畴的活组件·非文件名误导·
cognition/process/__init__.py 模块清单同此语义）。

复用 algorithm.a4_alignment（lcs_score·决策9 自建·LCS DP 纯整标准算法）。
诚实边界：结构对齐不判语义等价（LCS 是结构同构非"语义等价"·stable≠correct·接地墙 #479/钥匙②/钥匙③相2）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.algorithm.a4_alignment import lcs_score

# coverage_overlap ×1000 缩放（ordered 质量分 0..1000）
COVERAGE_SCALE = 1000
MAX_QUALITY = COVERAGE_SCALE   # 单前驱直通·满质量


def coverage_overlap(consensus: list[int], pred_seqs: list[list[int]],
                     *, ordered: bool = True) -> int:
    """合质量分（纯整 ×1000·ordered=True 用 LCS 序保真）。

    = 1000 × (Σ_s lcs_score(consensus, s)) / (len(pred_seqs) × max(1,|consensus|))
    满分 1000 = 每个 pred_seq 都按序含 consensus 全体。空 consensus → 0。
    """
    if not consensus or not pred_seqs:
        return 0
    for s in pred_seqs:
        for x in s:
            assert_int(x, _where="coverage_overlap.seq")
    total_lcs = 0
    for s in pred_seqs:
        total_lcs += lcs_score(consensus, s) if ordered else len(set(consensus) & set(s))
    denom = len(pred_seqs) * max(1, len(consensus))
    return (COVERAGE_SCALE * total_lcs) // denom
