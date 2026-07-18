"""vm — 图即程序虚拟机层（依赖 storage + crosscut·代码域 COMPOSES·首版语言接口留空）。

§九 2a 图即程序：符号标定算子（能指·symbol_domain 纯整数 opcode）+ 外部可换实现（所指·VM 墙内）。
COMPOSES 边（父→子）定义组合：父=算子节点·子=操作数。graph_compile 沿 COMPOSES 后序 emit
指令序列·dispatch 把 opcode 映射到墙内纯整数实现·vm_core 栈机执行（step_limit 禁无限步）。

首版语言接口留空：只编译既有 COMPOSES 子图·不从高级语言生成图（代码域 COMPOSES 随代码域
阶段激活·§7.4）。VM dispatch training 层真实活（execute_composes_value·formal_train verify/task-driven 轮）·
cognition 不调单向依赖守·gate DISPATCH_MODE 装饰位零读取（机制不读 gate·OFF/ON 等价 bit-identical·见 gates.py 装饰位范式）。
"""
from __future__ import annotations
