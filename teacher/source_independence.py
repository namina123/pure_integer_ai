"""teacher.source_independence — D3 裁判≠训练教师守第二层塌缩（§十一 #4-bis line710·#358 完整实现）。

断奶评估**裁判实例≠训练教师语义来源**——旧代码 identity 比对是简化·重来须**语义源比对(source_id)**：
裁判所用"标准"来源须与训练教师不同源（非同一录放层 episode/非同一教师 source_id·纯整 source_id 比对
集合不相交）。第二层塌缩风险=判断力=模仿度（裁判与教师同源则断奶后教师退场判断力断崖·同 weaning
永正另一面）。not independent→can_wean=False（硬条件）。

**诚实标注（当前违 D3 待分离）**：当前 build_judge_fn 裁判=训练教师本尊（绑 teacher.judge_ground_truth）
·source_id 同源→sources_disjoint 返 False→D3 硬前置挡→can_wean=False。这是诚实的（当前确未断奶）。
独立裁判实例分离（独立录放层 source / 断奶后自锚 J1-J4）是工程量·标"待分离"·不伪造通过。
D3 闸门**先就位挡假断奶**·独立裁判落地后自然通过。

铁律：纯整数（source_id 整数·集合运算）/ 不写死（source_id 由构造传入非硬编码）/ 依赖单向向下。
诚实边界：source 不相交是结构独立性非语义独立性（同源必不独立·不同源未必真独立·D 墙）/ stable≠correct。
"""
from __future__ import annotations

from typing import Iterable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def sources_disjoint(judge_source_ids: Iterable[int],
                     teacher_source_ids: Iterable[int]) -> bool:
    """D3·裁判源与训练教师源集合不相交（纯整集合运算）。

    not disjoint→can_wean=False（硬条件·防第二层塌缩：判断力=模仿度）。
    judge_source_ids   裁判所用"标准"来源 source_id 集
    teacher_source_ids 训练教师 source_id 集
    """
    return set(judge_source_ids).isdisjoint(set(teacher_source_ids))


def judge_source_independent_arith(*, verify_uses_vm_proof: bool,
                                   teacher_not_judge: bool) -> bool:
    """D3·算术域裁判源独立判定（W3·架构保证·非 sources_disjoint）。

    判据（全 True 才算术域 D3 就位）：
      1. verify_uses_vm_proof  算术域 verify 走 vm_proof_fn（VM 执行值自锚·非教师 GT·
                               _run_verify_round:570-576 PRE / :597-620 POST cross-verify）
      2. teacher_not_judge     算术域 verify 绕 judge（_is_verify_modality:374 早返·
                               teacher=None·judge_fn 不构建）

    算术域裁判源 = VM 执行值（R6 外部锚·非教师 source_id）→ 天然独立（架构保证·非集合运算）。
    算术域 self_proof_fn=None（teacher=None·stages.py:156 条件不满足）——算术域裁判源=vm_proof_fn·非 self_proof_fn 路径。

    **W3 scope**：只建判定 + 算术域能 True·**不调** weaning_check（通用路径 teacher=None 同源→D3 仍 False）·
    W7 才接全（路径 B 读算术域判定结果）。

    诚实边界：架构保证非语义独立（D 墙·stable≠correct）·算术域 vm_proof 是执行值接地（非 #479 truth）·
    独立裁判 source_id 是结构独立非语义独立（同源必不独立·不同源未必真独立）。
    """
    assert_int(int(verify_uses_vm_proof), int(teacher_not_judge),
               _where="judge_source_independent_arith")
    return verify_uses_vm_proof and teacher_not_judge
