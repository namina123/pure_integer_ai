"""teacher.weaning_e2 — E2 教师下线独立产出硬验骨架（§十一 #4-bis line708·#358 完整实现）。

E2 是断奶**最硬闸门**：教师下线后整数图须对"教师从未评过"的输入**独立产出**（非回放教师锚）·
布尔终止无法伪造此硬验。D1 防了"布尔阈值伪造"假断奶·E2 防"回放伪装独立"假断奶——系统可 weaning
趋势满足但产出实为回放教师锚→断奶后教师退场即断崖。E2 验"产出真自锚非回放"。

**唯一真 defer（带确切理由）**：E2 执行依赖完整训练管线（教师真退场 MODE_OFF·非 MODE_REPLAY +
独立探针输入 D4 + 产出非回放教师锚自锚 J1-J4 + C6 Mode B）·当前无真训练 run·gate 全 OFF·E2 无执行
条件。建**骨架**（接口 + 验证逻辑）+ 诚实标注执行条件未就位·不 defer 接口。

**诚实声明**：当前断奶 can_wean(语言域)永False 是决策层 truth（E2 未就位·三路堵·非'统计层不能学'）·统计层持续学习就绪判据另建（连 A/C·未建 defer）·不可信真断奶。

铁律：纯整数（输入/判定 0/1）/ 不走外挂 LLM（断奶后 LLM 退场）/ 不写死（判定条件元定义非语义规则）。
诚实边界：E2 骨架就位·执行条件依赖真训练 run·当前 can_wean 永 False（诚实·不伪造真断奶）。

**W2 算术域第三条件就位**（2026-07-11·produced_without_teacher_anchor_arith）：算术域 POST cross-verify
（cross_verify_pair·两路独立 execute_composes_value + rational.eq·all_agree）→ VM 执行值自锚（非教师锚·
非录放层命中）→ 第三条件算术域判定接口就位。e2_independent_production 仍 False（teacher_offline defer W6 /
probe_input_novel defer W4）·W7 才接全 E2。e2_execution_ready() 仍返 False（诚实·算术域第三条件就位非 E2 过）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def e2_independent_production(*, teacher_offline: bool,
                              probe_input_novel: bool,
                              produced_without_teacher_anchor: bool) -> bool:
    """E2 骨架·教师下线独立产出判定（断奶最硬闸门）。

    执行条件（全部就位才返 True·缺一即 False·诚实）：
      1. teacher_offline          教师真退场（MODE_OFF·非 MODE_REPLAY 回放）
      2. probe_input_novel        输入是教师从未评过的探针（D4 探针集·非训练集泄漏）
      3. produced_without_teacher_anchor  产出非回放教师锚（自锚 J1-J4 + C6 Mode B·非录放层命中）
    当前执行条件未就位（无真训练 run·gate 全 OFF·E2 无执行条件）→永 False。
    建骨架非 defer 接口·真训练 run 落地后自然能判。
    """
    assert_int(int(teacher_offline), int(probe_input_novel),
               int(produced_without_teacher_anchor), _where="e2_independent_production")
    return teacher_offline and probe_input_novel and produced_without_teacher_anchor


def e2_execution_ready() -> bool:
    """E2 执行条件就绪判定（诚实·当前永 False）。

    E2 执行依赖真训练 run（教师真退场 + 独立探针 + 自锚产出全就位）·当前 gate 全 OFF·无真训练 run·
    执行条件未就位。返 False → can_wean 永 False（诚实声明当前断奶 theatrical·E2 未就位前不可信真断奶）。
    真训练 run 落地后此函数接真执行条件判定·非伪造通过。
    """
    return False


def produced_without_teacher_anchor_arith(*, cross_verify_ran: bool,
                                          cv_all_agree: bool) -> bool:
    """算术域 produced_without_teacher_anchor 判定（W2·E2 第三条件算术域就位）。

    判据（全 True 才算术域第三条件就位）：
      1. cross_verify_ran   POST cross-verify 真跑（gates.MODE_B_CROSS_VERIFY_MODE=True AND arith_source_b
                            非 None·formal_train.py:597 双条件·非短路 reward=0 占位）。
      2. cv_all_agree       两路独立 execute_composes_value 值一致（cross_verify_pair all_agree·
                            mode_b_cross_verify.py:60·VM 执行值自锚·两路独立编译路径 R6 真守）。

    **非录放层命中**（架构保证·非本函数判）：POST 路径 probes 丢 spec.expected（formal_train.py:617 只取
    input_args）·绕 judge（:375 早返·不绑 teacher GT）·verify_source=VERIFY_SOURCE_EXTERNAL（:638）·
    无 teacher recording 路径。本函数依赖此架构保证·不内部查 recording hit（defer·若未来 POST 接
    teacher replay 须加检查）。

    **W2 scope**：只建算术域第三条件判定 + 算术域能 True·**不调** e2_independent_production（teacher_offline
    defer W6 / probe_input_novel defer W4）·e2_execution_ready() 仍返 False·can_wean 永 False。
    W7 才接 e2_independent_production 第三入参用此函数（算术域三条件全就位）。

    诚实边界：cross_verify 验统计学内一致（agreement 非 identity·Rice 有限基底）·非验语义正确
    （stable≠correct·grounding 墙外）·single-source 脆弱（两 DSL 同出 corpus·系统性中毒 agree wrong
    无法检出·mode_b_cross_verify.py:22-24）。
    """
    assert_int(int(cross_verify_ran), int(cv_all_agree),
               _where="produced_without_teacher_anchor_arith")
    return cross_verify_ran and cv_all_agree


def e2_execution_ready_arith(*, teacher_offline: bool,
                             probe_input_novel: bool,
                             produced_without_teacher_anchor: bool) -> bool:
    """E2 算术域执行条件就绪判定（W6·模拟退场预验·解 teacher_offline 循环依赖）。

    镜像 W3 judge_source_independent_arith 范式（域特化判定接口·通用 e2_execution_ready() 仍 False defer W8）。
    三条件由 _run_simulated_offline_eval 采（formal_train.py stage4 末·weaning_check 之前·预验非后验·解循环依赖）：
      1. teacher_offline                  算术域 teacher=None→真退场（架构事实·无 recording/replay/GT·反 theater①：
                                         eval guard 守 ctx.teacher is None + cross_verify_pair 零教师 import 机制保证）
      2. probe_input_novel                探针集隔离（W4 ctx.probe_set_disjoint·held-out 不相交训练集·caller 传 probe_holdout>0）
      3. produced_without_teacher_anchor  探针产出非教师锚（produced_without_teacher_anchor_arith·
                                         cross_verify_ran + cv_all_agree·VM 执行值自锚·W2 已建复用）

    算术域 teacher=None 天然退场（无 teacher 实例可翻 mode·无 recording 可回放·无 GT 可注入）·语言域真翻
    MODE_OFF + 恢复 MODE_REPLAY defer W8。解循环依赖：eval 在 weaning_ready 判定之前（预验·stage4 末
    weaning_check 之前）·非之后（真退场）·ready 读 eval 结果（ctx.e2_eval_passed）非驱动 eval。

    **W6 scope**：只建算术域判定接口 + 算术域 eval 子阶段·通用 e2_execution_ready() 仍 False（语言域 defer W8）。
    weaning_ready 仍 False（D1-D5 defer·只 E2 单闸门算术域可过·诚实非真断奶）·W7 才接全六闸门达 can_ween=True。

    诚实边界：算术域 teacher=None 是架构事实非"模拟退场"（语言域真翻 MODE_OFF defer W8）·算术域 fixture 同源
    trivial（probe/training 都 square n²→cross_verify 恒 agree→holdout_retention 恒 1000·真泛化保持 defer W8）·
    eval observe 探针建学树是 fresh-compile（非 recognize_operators 回忆）→ 验"自锚产出非教师锚"（E2 核心）非保持率泛化。
    """
    assert_int(int(teacher_offline), int(probe_input_novel),
               int(produced_without_teacher_anchor), _where="e2_execution_ready_arith")
    return e2_independent_production(
        teacher_offline=teacher_offline,
        probe_input_novel=probe_input_novel,
        produced_without_teacher_anchor=produced_without_teacher_anchor)
