"""training.value_synthesize — 相1 算术归纳合成（骨架引导 PARAM 绑定行为匹配搜索·doc §三/§二十·G-PR1）。

**性质**：归纳程序合成（programming by examples）——给定 I/O spec_pairs·搜骨架池找行为匹配骨架。
真搜索（枚举骨架池 + PARAM 绑定方案·每方案 execute_composes_value）+ 真验证（Rational 精确执行比对）。
新结构限于骨架池已有结构（参数化实例化·非结构创新·诚实边界·§相1.5）。

**关键洞察**（§相1.1·B agent 读码）：execute_composes_value 的 env 预载机制（vm_proof.py:82）让 PARAM 值
搜索**不需树构造**——骨架即树，PARAM 值通过 env 传。搜索 = 枚举骨架池 + 绑定方案，每方案一次 execute。

**两级搜索**（§相1.1）：
  1. 直接试执行（skeleton.arity == n_args）：identity 绑定·param_values = input_args 序·每 spec 一次 execute。
  2. PARAM 绑定搜索（skeleton.arity != n_args）：_enumerate_bindings(arity, n_args)=itertools.product
     输入位置^槽（n_args^arity 方案·小 arity 可控）·每绑定方案 execute 全 specs。
每候选 execute 全 spec_pairs·全 rational.eq(expected) → 命中（首匹配绑定 per 骨架·确定性）。

返 list[(skeleton_ref, binding)]（命中·空=无匹配诚实·pool 已按 skeleton_ref 升序→结果同序 bit-identical）。
binding = tuple[int, ...]（PARAM slot i → input_args[binding[i]]·identity=()range(n_args)·空 tuple=nullary）。

**复用（零改）**：
- vm_proof.py:45 execute_composes_value — 验证（PARAM env 预载·compile_graph→execute→Rational|None）
- structure_discover.py:1003 load_discovered_operators — 骨架池（**caller 传 pool·本模块不 load**·守单向 + 可单测）

铁律：纯整数（input_args/expected/Rational 全 int·assert_int 守）/ 确定性（stable sort + itertools.product
  确定性序·bit-identical）/ 单向依赖（training L7·import vm_proof L7 + crosscut 向下·**不 import cognition**
  [pool caller 传]·execute_composes_value 接受 graph Any 传入非向上）/ 不写死（行为匹配+结构 arity/binding·
  **非硬编码** skeleton→spec/action→spec mapping）/ 反 theater（DISAGREE 牙：无匹配返空非伪造·§20.4 测2）。
诚实边界：stable≠correct（行为匹配=执行一致非意图正确·#479 墙）·自骨架可能（pool 或含自观察 DISCOVERED 骨架
  →弱信号构造性·排除自 defer refinement）·production 全 specs 搜索（held-out 测试级·§20.1 决断4）·
  arity==n_args 仅 identity 绑定（排列绑定 defer·§相1.1 level1 scope）。
"""
from __future__ import annotations

from itertools import product
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.training.vm_proof import execute_composes_value


def _enumerate_bindings(arity: int, n_args: int) -> list[tuple[int, ...]]:
    """PARAM 槽 → 输入位置绑定枚举（arity 槽·每槽选 n_args 个输入位置之一·n_args^arity 方案）。

    binding[i] = 输入位置 index（PARAM slot i 绑 input_args[binding[i]]·DFS 阅读序=make_variable index 序）。
    arity=0（nullary）→ [()]（一空绑定·直接执行无 PARAM·input_args 不进 env）。
    n_args=0 且 arity>0 → []（无输入可绑·不可达·caller 守 specs 非空 + input_args 非空）。

    itertools.product(range(n_args), repeat=arity) 确定性序（lexicographic·bit-identical）。
    小 arity 可控（n_args^arity·同 §相2 深度闸范式·组合爆炸守 caller 侧 arity 小·§相1.5）。
    """
    assert_int(arity, n_args, _where="_enumerate_bindings.arity_n_args")
    if arity == 0:
        return [()]
    if n_args == 0:
        return []   # arity>0 但无输入→不可绑（防御·caller 传非空 input_args）
    return list(product(range(n_args), repeat=arity))


def _binding_param_values(input_args: tuple[int, ...],
                          binding: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """绑定方案 → PARAM 值序（execute_composes_value param_values·DFS 阅读序）。

    param_values[i] = (input_args[binding[i]], 1)（PARAM slot i ← 输入位置 binding[i] 的值·Rational int/1）。
    空 binding（nullary·arity=0）→ ()（无 PARAM·execute 纯立即数）。
    binding 长度须 == arity（caller 保证：identity=tuple(range(n_args))[arity==n_args] / product[arity]）。
    """
    return tuple((input_args[binding[i]], 1) for i in range(len(binding)))


def _skeleton_matches(graph: Any, skeleton_ref: tuple[int, int],
                      specs: tuple[Any, ...], binding: tuple[int, ...]) -> bool:
    """单骨架+绑定方案：execute 全 specs·全 rational.eq(expected)→True（首失败短路·确定性）。

    每 spec 用**自身 input_args** + 同一 binding 构 param_values（specs 须同 n_args·函数输入 arity 一致）。
    execute_composes_value 返 None（非 COMPOSES 根/StepLimit）→ False（诚实·非 theater）。
    rational.eq 外比（expected=教师 Mode A 独立源·Rational 精确）。
    """
    for spec in specs:
        param_values = _binding_param_values(spec.input_args, binding)
        v = execute_composes_value(graph, skeleton_ref, param_values)
        if v is None:
            return False   # 非 COMPOSES 根 / StepLimit → 不匹配（R1 诚实·非 vacate）
        if not rational.eq(v, rational.make(spec.expected[0], spec.expected[1])):
            return False   # 行为不匹配 → 短路（确定性·后续 spec 不跑）
    return True


def synthesize_value(graph: Any, skeleton_pool: list[Any],
                     spec_pairs: tuple[Any, ...]
                     ) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    """相1 算术归纳合成：spec_pairs → 搜骨架池找行为匹配骨架（doc §三/§二十·G-PR1）。

    graph : ConceptGraph（execute_composes_value 用·read_composes_tree+compile_graph+execute）。
    skeleton_pool : list[DiscoveredOperator]（caller load_discovered_operators 产·按 skeleton_ref 升序）。
    spec_pairs : tuple[CodeSpec, ...]（I/O 对·input_args+expected·教师 Mode A 独立源）。

    返 list[(skeleton_ref, binding)]（命中·**pool 升序遍历→结果同序 bit-identical**·空=无匹配诚实）。
    每 skeleton 至多一条（首匹配 binding·itertools.product 确定性序内首个全 spec 匹配）。

    **两级搜索**（§相1.1）：
      arity == n_args → identity 绑定 [tuple(range(n_args))]（level1·直接·每 spec 一次 execute）
      arity != n_args → _enumerate_bindings(arity, n_args)（level2·n_args^arity 方案）
    n_args = len(spec_pairs[0].input_args)（specs 须同 n_args·函数输入 arity 一致·不另守·教师数据保证）。

    **反 theater DISAGREE 牙**（§20.4 测2）：pool 无行为匹配骨架→返空（非伪造 reward·caller 诚实 reward=0）。
    **不写死**：selection by 行为匹配（execute+compare）+ 结构（arity/binding）·非硬编码 mapping。
    """
    if not spec_pairs or not skeleton_pool:
        return []   # 无 spec / 空池 → 无匹配诚实（caller 守·防御）
    n_args = len(spec_pairs[0].input_args)
    # 防御：specs 须同 n_args（函数输入 arity 一致·教师数据保证）·不一致→无匹配诚实（避 opaque IndexError·
    # _binding_param_values 用 specs[0] n_args 建绑定·异长 spec 索引越界·防御返空非崩溃）。
    if any(len(s.input_args) != n_args for s in spec_pairs):
        return []
    matches: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    for op in skeleton_pool:
        arity = op.arity
        # 两级绑定集（§相1.1·level1 identity / level2 enumerate）
        if arity == n_args:
            bindings: list[tuple[int, ...]] = [tuple(range(n_args))]   # identity（直接·level1）
        else:
            bindings = _enumerate_bindings(arity, n_args)   # 全枚举（level2·arity != n_args）
        if not bindings:
            continue   # arity>0 + n_args==0 不可绑（防御）→ 跳过此骨架
        for binding in bindings:
            if _skeleton_matches(graph, op.skeleton_ref, spec_pairs, binding):
                matches.append((op.skeleton_ref, binding))
                break   # 首匹配 binding per 骨架（确定性·itertools.product 序内首个·不收集多绑定）
    # pool 已按 skeleton_ref 升序（load_discovered_operators 保证）→ matches 同序（bit-identical·不再 sort）
    return matches
