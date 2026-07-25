"""cognition.result.output_measure — 统计层产出度量 reader（#1041 构造②·判据②③信号质量）。

产出真词度量：读 OutputPart.token_refs（#1040 携段 token concept 序）判产出**真 token** 非 truthiness
（非空字符串）。解 review-2 钉死：judge slot_fill_rate `if w:` 只判非空字符串 → 真词/`__seg_*` label
同分 → reward 信号假（判据②③）。

  output_word_ratio(output) -> int   产出真 token 覆盖率（×1000·Σ token_refs / Σ words）→ judge J4word 项

**bit-identical（gate OFF·CI）**：DISPATCH_TOKEN_CHAIN_MODE OFF → token_refs 全空（generate 不填）→
  ratio=0（次守·主守在 judge 调用方 gate OUTPUT_WORD_REWARD_MODE）。
**token_refs ≤ words**（LOW-2 守·generate emitted_tokens 只收 slot_idx<len(token_seq) 的真 token·
  extra slot 退 unit 不入 token_refs）→ real≤total → ratio∈[0,1000]。
**modality-safe**：code/arith output 无 token_refs（语言 dispatch 才填）→ ratio=0 → judge J4word=0
  不扰 code/arith reward（与 vm_proof reward 正交）。

铁律：纯整数（ratio ×1000 / assert_no_float）/ 确定性（token_refs+words 内容定·无墙钟）/
  单向依赖（cognition 层·judge downward import）/ 反 theater（读 token_refs 真信号·非 truthiness）。
诚实边界：token_refs 覆盖率=产出真 token 比例（结构度量·stable≠correct·非语义正确判据）·
  "来自语料词表"更强对齐 defer（首版 token_refs 覆盖即真 token·#1040 dispatched concept）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.cognition.shared.types import OutputResult

_SCALE = 1000


def output_word_ratio(output: OutputResult) -> int:
    """产出真 token 覆盖率（纯整 0..1000）= Σ token_refs / Σ words × 1000。

    消费方：judge J4word 项（#1041 构造② reward truthiness 校准·gate OUTPUT_WORD_REWARD_MODE 门控）。
    total=0（空 output）→ 0（防除零·冷启动不报假信号）。
    modality-safe：code/arith 无 token_refs → 0（不扰非语言 reward）。
    """
    total = 0
    real = 0
    for part in output.parts:
        total += len(part.words)
        real += len(part.token_refs)
    if total <= 0:
        return 0
    # 防御性 clamp（LOW-1·纵深防御·匹配 judge.path_strength_weighted min(J_SCALE, total) 范式）：
    # 不变量 token_refs≤words（generate emitted_tokens 只收 slot_idx<len(token_seq) 真 token）→ real≤total
    # → ratio∈[0,1000]·clamp 守直接构造 OutputPart 破不变量的边界（防 J4word>1000 过度主导 reward）。
    ratio = min(_SCALE, (real * _SCALE) // total)
    assert_no_float(ratio, _where="output_word_ratio")
    return ratio
