"""numeric.symbol_domain — 符号域轴框架 + opcode↔symbol 桥（依赖 crosscut）。

§九 2a 图即程序范式：符号标定算子（能指·核心整数）+ 外部可换实现（所指·VM 墙内/外部墙外）。
symbol_domain 是"能指"层——把 opcode / 变量 / 参数 标定为纯整数 symbol_id，VM 沿此 dispatch。

  TYPE_AXIS       数值关联轴枚举（概念在哪个轴上有定点值）
  OPCODE_*        算子 opcode（图即程序 dispatch 用·Stage 2 VM 填实完整集）
  VARIABLE / PARAM 标记常量变量 / 参数符号
  register_opcode / opcode_to_symbol / symbol_to_opcode  opcode↔symbol 桥（L1 硬编码对齐 encoding）

【诚实标注】Stage 0 落轴框架 + 桥机制 + 最小 opcode 占位（NOP/ADD/SUB/MUL/DIV/EQ/LT/GT/LOAD/STORE）。
A1（2026-07-02·doc/重来_VM图灵完备与C6设计补充.md）加 4 控制流 opcode（PUSH_IMM/JZ/JMP/HALT）→ VM 直线求值器升图灵完备。
完整 opcode 集随 VM 阶段填实——只追加不重排（L1 硬编码对齐 encoding·确定性）。
桥是 L1 硬编码对齐 encoding（opcode↔symbol_id 双射·确定性·跨宿主一致）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

# ---- 数值关联轴 ----
# 概念可在多个轴上有定点值（一概念多抽象身份·§八限制空间）。轴是纯整数枚举。
TYPE_AXIS_NONE = 0      # 无数值轴（纯结构概念）
TYPE_AXIS_INT = 1       # 整数轴（concept_numeric M/B^k = 整数值）
TYPE_AXIS_RATIONAL = 2  # 有理轴（M/B^k 定点有理数）
TYPE_AXIS_ORDINAL = 3   # 序轴（ranking·cross_compare 比序）

# ---- 符号种类（axis_symbol_id 高位区分图即程序 vs 数值关联） ----
# axis_symbol_id = (kind << KIND_SHIFT) | index：kind 区分 opcode/变量/参数/数值关联。
KIND_SHIFT = 60  # 高 4 bit 种类·低 60 bit index（index < 2^60 充裕）
KIND_NUMERIC = 0       # 数值关联标记（概念在某轴的定点值）
KIND_OPCODE = 1        # 图即程序 opcode
KIND_VARIABLE = 2      # 变量
KIND_PARAM = 3         # 参数

# ---- 最小 opcode 占位集（Stage 2 VM 填实完整 20 opcode） ----
# 编码 = OPCODE_BASE + i（L1 硬编码对齐 encoding·确定性）。Stage 2 扩展时只追加不重排。
OPCODE_BASE = 1 << KIND_SHIFT  # opcode 的 axis_symbol_id 基址
OPCODE_NOP = OPCODE_BASE | 0
OPCODE_ADD = OPCODE_BASE | 1
OPCODE_SUB = OPCODE_BASE | 2
OPCODE_MUL = OPCODE_BASE | 3
OPCODE_DIV = OPCODE_BASE | 4   # 有理除·走 longdiv 定点
OPCODE_EQ = OPCODE_BASE | 5
OPCODE_LT = OPCODE_BASE | 6
OPCODE_GT = OPCODE_BASE | 7
OPCODE_LOAD = OPCODE_BASE | 8
OPCODE_STORE = OPCODE_BASE | 9
# A1 控制流（2026-07-02·图灵完备·见 doc/重来_VM图灵完备与C6设计补充.md §二）
OPCODE_PUSH_IMM = OPCODE_BASE | 10   # 有理立即数入栈 (num,den)→make(num,den)
OPCODE_JZ = OPCODE_BASE | 11         # 跳零：pop；零则 pc=target 否则 pc+1
OPCODE_JMP = OPCODE_BASE | 12        # 无条件跳：pc=target
OPCODE_HALT = OPCODE_BASE | 13       # 显式终止·返栈顶

# 符号数学扩展 Phase 2b（doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis.4）：
# OPCODE_POW_PATTERN = pattern-level Pow 算子（变量指数·d/dx LHS/RHS 模板用）·**非 VM opcode·不进 _OPCODE_TABLE**。
# 作 ATTR_OPERATOR int_a（_align_walk:1307 ATTR_OPERATOR 匹配命中）·VM 不执行（compile_graph/execute 未定义→fail-loud
# 若达执行·Pow 须先 lower MUL·见 symbolic_transform._lower_pow）。用户 DSL `a**k` 仍 _build_pow 展开 MUL（不变）。
OPCODE_POW_PATTERN = OPCODE_BASE | 14

# opcode 名 → axis_symbol_id（L1 硬编码对齐 encoding·桥的双射源）
_OPCODE_TABLE: dict[str, int] = {
    "NOP": OPCODE_NOP, "ADD": OPCODE_ADD, "SUB": OPCODE_SUB, "MUL": OPCODE_MUL,
    "DIV": OPCODE_DIV, "EQ": OPCODE_EQ, "LT": OPCODE_LT, "GT": OPCODE_GT,
    "LOAD": OPCODE_LOAD, "STORE": OPCODE_STORE,
    "PUSH_IMM": OPCODE_PUSH_IMM, "JZ": OPCODE_JZ, "JMP": OPCODE_JMP, "HALT": OPCODE_HALT,
}
# 反向双射
_OPCODE_REVERSE: dict[int, str] = {v: k for k, v in _OPCODE_TABLE.items()}


def register_opcode(name: str, symbol_id: int) -> None:
    """Stage 2 VM 扩展 opcode 用：追加 name↔symbol_id（不许覆盖既有·守双射）。"""
    assert_no_float(symbol_id, _where="register_opcode")
    if name in _OPCODE_TABLE or symbol_id in _OPCODE_REVERSE:
        raise ValueError(f"register_opcode: name/symbol 已存在（守双射）: {name}/{symbol_id}")
    _OPCODE_TABLE[name] = symbol_id
    _OPCODE_REVERSE[symbol_id] = name


def opcode_to_symbol(name: str) -> int:
    """opcode 名 → axis_symbol_id（桥·L1 硬编码对齐）。"""
    if name not in _OPCODE_TABLE:
        raise KeyError(f"opcode_to_symbol: 未知 opcode {name!r}")
    return _OPCODE_TABLE[name]


def symbol_to_opcode(symbol_id: int) -> str:
    """axis_symbol_id → opcode 名（桥反向）。"""
    if symbol_id not in _OPCODE_REVERSE:
        raise KeyError(f"symbol_to_opcode: 非 opcode symbol {symbol_id!r}")
    return _OPCODE_REVERSE[symbol_id]


def make_variable(index: int) -> int:
    """变量 symbol_id（kind=VARIABLE）。index 须 ≥ 0。"""
    assert_int(index, _where="make_variable.index")
    if index < 0 or index >= (1 << KIND_SHIFT):
        raise ValueError(f"make_variable: index 越界 [0, 2^60): {index}")
    return (KIND_VARIABLE << KIND_SHIFT) | index


def make_param(index: int) -> int:
    """参数 symbol_id（kind=PARAM）。index 须 ≥ 0。"""
    assert_int(index, _where="make_param.index")
    if index < 0 or index >= (1 << KIND_SHIFT):
        raise ValueError(f"make_param: index 越界 [0, 2^60): {index}")
    return (KIND_PARAM << KIND_SHIFT) | index


def kind_of(axis_symbol_id: int) -> int:
    """返回 symbol 的种类（KIND_*）。"""
    assert_int(axis_symbol_id, _where="kind_of")
    return axis_symbol_id >> KIND_SHIFT


def index_of(axis_symbol_id: int) -> int:
    """返回 symbol 的低 60 bit index。"""
    assert_int(axis_symbol_id, _where="index_of")
    return axis_symbol_id & ((1 << KIND_SHIFT) - 1)
