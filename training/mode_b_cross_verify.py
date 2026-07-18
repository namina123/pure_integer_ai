"""training.mode_b_cross_verify — Mode B POST-weaning 异算法统计一致性 cross-verify（#479 加强腿）。

POST-weaning 教师退场 → 无 expected 独立源（vm_proof.py docstring 真缺口·formal_train.py:446 POST
路径 reward=0 防 vacuous theater）。本模块给 POST-weaning CODE/ARITH 域一条**统计学一致**加强腿：
两 DSL 表达同函数（异 shape·迭代 vs 闭式）→ 各自独立编译（异 builder 代码路径·CTRL_WHILE 迭代 vs
直线 BinOp）→ N 探针交叉 execute_composes_value → rational.eq agreement。

**哲学定位（用户决断 2026-07-06·统计学重定向）**：不追求 correctness 真墙（Rice·须墙外 #478/#493）·
**只求统计学内一致**。Rice 墙对统计一致目标不构成障碍——目标是"K=2 独立编译路径 × N 探针一致事件"·
非 identity·非 truth。异 shape = 异 builder 代码路径 = R6 真守（非同源编译·非单源 self-check theater）。
这把 cross-verify 从"够不到 correctness 的妥协"翻成 on-target 的统计一致性机制（目标本身）。

**Mechanism Y（非 test_stage9 Mechanism X）**：test_stage9_arith_observe.py:531-569 用 vm_proof_fn
双路 vs 手编 oracle（Mechanism X·**需 expected → POST 不可用**）。本模块用 execute_composes_value
双路取值 + rational.eq（Mechanism Y·**无 oracle·"另一棵树"即独立源·POST 可用**）。

**caller（formal_train.py _run_verify_round POST 分支）**：arith_source（学树·observe 建）+
arith_source_b（参树·build_composes_from_arith 二次独立建·异 shape）→ cross_verify_pair →
reward = 1 iff all_agree。probes = spec.input_args（复用既有测试输入·**丢 expected**·避教师 oracle·
守 #479 不破）。reward 进 Episode metrics（verify propagate 永久 no-op·不落边 strength·不写 op_confidence）。

**诚实边界**：
  - agreement 非 identity（Rice 有限基底）——统计一致的有限覆盖·统计学 framing 下是目标本身非缺陷。
  - single-source 脆弱——两 DSL 同出 corpus·系统性中毒（两表达同方式错）→ agree wrong → 漏。仅
    corpus 内部冗余（两表达异方式同函数）才抓获。声称守"agreement 非 truth"。
  - LANGUAGE 域墙（#479 cleavage）本模块不涉。cross-verify pair 机制模态无关（execute_composes_value + rational.eq
    ·两域都返 Rational）·caller（formal_train _run_verify_round POST 分支）按模态选 source_b/builder：
    ARITH 用 arith_source_b / build_composes_from_arith · CODE 用 code_source_b / build_composes_from_source（§施工序 1.2）·
    LANGUAGE 非-VM 统计链另线 defer。
  - stable≠correct（接地墙外）——agreement 不保证语义正确。

铁律：纯整数（probe int→(int,1) Rational·eq 交叉积整运算·reward int 0/1）/ 确定性（两路独立
build+execute·无随机源·probe tuple 序确定）/ fail-loud（任一 None→vacate 计数·n_valid=0→reward=0
诚实 no-op 非 theater）/ 单向依赖（training L7→vm_proof L7 + crosscut.integer L1 向下·零 cognition
import·graph 是 ConceptGraph 入参不 import）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.training.vm_proof import execute_composes_value


@dataclass(frozen=True)
class CrossVerifyResult:
    """cross_verify_pair 结果（纯统计台账·零 IO）。

    n_probes : 探针总数（= len(probes)）。
    n_valid  : 两路都非 None 的探针数（任一 None→该探针 vacate 不计 valid 也不计 agree）。
    n_agree  : n_valid 中 rational.eq 相等的探针数。
    all_agree: n_valid > 0 且 n_agree == n_valid（全一致·caller reward=1 判据·n_valid=0 退化 False）。
    """
    n_probes: int
    n_valid: int
    n_agree: int

    @property
    def all_agree(self) -> bool:
        return self.n_valid > 0 and self.n_agree == self.n_valid


def cross_verify_pair(graph: Any, root_a: Any, root_b: Any,
                      probes: tuple[tuple[int, ...], ...]) -> CrossVerifyResult:
    """两路独立 execute_composes_value + rational.eq agreement（纯函数·零 IO·零 backend 写）。

    graph   : ConceptGraph（read_composes_tree 读两路 COMPOSES 子树·compile_graph 编译·execute 执行）。
    root_a  : 学树 COMPOSES 根 ConceptRef（observe 建·formal_train POST 路径 struct_refs[0]）。
    root_b  : 参树 COMPOSES 根 ConceptRef（build_composes_from_arith 二次独立建·异 shape·R6 真守）。
    probes  : 函数输入探针序·每元一组 int（DFS 阅读序对齐 make_variable index·= spec.input_args 序）。

    每探针 int→(int,1) Rational 预载 PARAM env·两路各 execute_composes_value 取值·rational.eq。
    任一路 None（root 非 COMPOSES 根 / StepLimit / KeyError=arity 不匹配 unbound LOAD）→ 该探针
    vacate（n_valid 不增·n_agree 不增·诚实非 theater·同 vm_proof.py StepLimit→None 语义）。

    返 CrossVerifyResult。all_agree = (n_valid > 0 and n_agree == n_valid)。
    空探针 / 全 vacate → n_valid=0 → all_agree=False → caller reward=0（诚实 no-op·非 vacuous agree）。
    """
    n_valid = 0
    n_agree = 0
    for probe in probes:
        probe_rat = tuple((n, 1) for n in probe)
        try:
            va = execute_composes_value(graph, root_a, probe_rat)
        except KeyError:
            va = None   # arity 不匹配 unbound LOAD → vacate（同 StepLimit→None 语义·漏洞 2 修）
        try:
            vb = execute_composes_value(graph, root_b, probe_rat)
        except KeyError:
            vb = None
        if va is None or vb is None:
            continue   # vacate（不计 valid 也不计 agree·诚实）
        n_valid += 1
        if rational.eq(va, vb):
            n_agree += 1
    return CrossVerifyResult(n_probes=len(probes), n_valid=n_valid, n_agree=n_agree)
