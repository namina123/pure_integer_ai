"""teacher.recursive_define — G8 教师提问循环恢复（RECORD-time 递归 define·断奶前 bootstrap）。

define_recursive(root_ref, teacher, provider, *, max_depth, budget, atom_fn) -> int
  从 root_ref 递归定义：provider(ref) -> (text, [prereq_refs])。每节点经 teacher.define 落录放层
  （hash-recorded·MODE_REPLAY hash 查表 → bit-identical）·递归 prereqs 到原子或闭环。

**G8 = 正资产 restoration（定位重锚决断_2026-07-08 §纠偏 2）**：旧架构（mock_v1/legacy_v1 learner.py）
  有教师提问循环（递归 define + 自指检测 + 必闭环）·新架构被砍——理由=bit-identical E4 禁运行时 LLM 的
  **误读**（E4 只禁运行时 LLM·不禁离线预录）。纠正：递归挪到 MODE_RECORD（离线录整条定义链）·MODE_REPLAY
  时每节点 hash 查表 → bit-identical 保住 + 断奶前 bootstrap 回来。录放层本为此设计（旧只录单层·G8 录递归链）。

**为何 bit-identical（核心论证）**：driver 是纯确定性编排——provider 同 ref 必返同 (text, prereqs)·三守
  （depth/budget/visited）纯整数确定性·每节点经 teacher.define（_call: hash=Hasher.h63((kind,args))·
  MODE_RECORD 录 / MODE_REPLAY 同 hash 查表）。故 MODE_RECORD 录的递归链与 MODE_REPLAY 走的查表链**同形**
  → 跨进程跨 run bit-identical。driver 自身不改录放 schema / 不改 _call / 不引运行时 LLM。

**三守（防爆 + 反自指 + 原子停·镜像 legacy learner.py:109-191）**：
  - max_depth=5：递归深度（路径长）上限（legacy MAX_DEFINE_DEPTH=5·防无限下挖）。
  - budget=12：总 define 查询数上限（legacy max_defines_per_learn=12·防 3^depth≈243 爆炸）。
  - visited set：本次调用**全局共享**已定义概念集·重复到访=自指/环→终止该分支（legacy self-ref detection
    守值镜像）·**memo 机制异 legacy**：legacy 用 path-based `visited | {ref}`（兄弟分支不共享·菱形重定义）·
    本模块用全局 visited（菱形依赖 D 经 B 已定义→经 C 不重定义·每概念每调用至多一次定义·更严）。
  - atom_fn(ref)->bool：caller 判"已已知/稳定概念"→不定义不递归（原子·闭环节点·如稳定 attractor basin）。

**provider 契约**：caller-supplied deterministic（同 ref → 同 (text, prereq_refs)）。encapsulates "定义文本 +
  其前置依赖"——新架构无 legacy 拆词教师·故前置依赖由 caller（bootstrap/corpus 预处理）显式提供·非运行时
  解析定义文本（那须 tokenize+lookup·schema 改动·Phase 2 defer）。未知 ref → provider 抛 KeyError →
  driver 当原子停（闭环节点·不强行定义）。

铁律：纯整数（ConceptRef/depth/budget 全整）/ bit-identical（driver 确定性 + teacher.define hash 录放）/
  不写死（provider/atom_fn caller 注入·本模块只机制）/ 单向依赖（L7 teacher·只 import cognition.shared.types）/
  反 theater（真递归链 + 三守 + cross-verify RECORD↔REPLAY bit-identical·非空转）/ 闭项守（provider 未提供
  → KeyError 当原子停·不抛崩训练）。
诚实边界：本模块是**机制恢复 + 合成测验证**·formal_train 生产接线（bootstrap caller + 真 corpus provider）
  defer 到 #731 真语料驱动时（同符号数学 Phase 3 范式·mechanism ready pending real consumer）。语义正确性是
  #479 W2 truth 墙（教师定义真伪·系统不判·只录链）。auto-extract prereqs from text（legacy 拆词）Phase 2 defer。
"""
from __future__ import annotations

from typing import Callable

from pure_integer_ai.cognition.shared.types import ConceptRef

# 三守默认（镜像 legacy learner.py:109 max_depth=5 / __init__ max_defines_per_learn=12）
DEFAULT_MAX_DEPTH = 5
DEFAULT_BUDGET = 12

# provider 契约类型：(ref) -> (text, prereq_refs)。deterministic·同 ref 同返。
Provider = Callable[[ConceptRef], "tuple[str, list[ConceptRef]]"]
# atom 谓词类型：(ref) -> 已已知/稳定（不递归）。
AtomPredicate = Callable[[ConceptRef], bool]


def define_recursive(root_ref: ConceptRef, teacher, provider: Provider, *,
                     max_depth: int = DEFAULT_MAX_DEPTH,
                     budget: int = DEFAULT_BUDGET,
                     atom_fn: AtomPredicate | None = None) -> int:
    """RECORD-time 递归 define（G8·断奶前 bootstrap·bit-identical）。

    从 root_ref DFS 递归：provider(ref) 取 (text, prereq_refs) → teacher.define(ref, text) 落录放层
    → 递归每个 prereq（depth+1）。三守（max_depth / budget / visited）+ atom_fn 闭包。

    **bit-identical**：driver 纯确定性编排·每 define 经 teacher.define hash 录/查 → MODE_RECORD 录的链
    与 MODE_REPLAY 查的链同形 → 跨 run bit-identical。teacher.define miss→None（E4）·driver 不 fallback。

    返本次 define 尝试数（teacher.define 调用次数·不含 atom/visited/depth/budget 跳过的）。
    """
    visited: set[ConceptRef] = set()
    budget_left = [int(budget)]   # mutable box（嵌套闭包共享递减）
    count = 0

    def _go(ref, depth: int) -> None:
        nonlocal count
        if depth >= max_depth:
            return   # 深度上限（legacy MAX_DEFINE_DEPTH=5）
        if budget_left[0] <= 0:
            return   # 预算耗尽（legacy max_defines_per_learn=12·防爆）
        if not _valid_concept_ref(ref):
            return   # malformed ref（caller 违约·如 3-tuple/str）→ skip 不抛崩（闭项守·审 F2）
        if ref in visited:
            return   # 自指/环 或 已 memo（菱形依赖不重定义）→ 终止该分支
        if atom_fn is not None and atom_fn(ref):
            return   # 原子（已已知/稳定 attractor）→ 闭包不递归
        try:
            text, prereqs = provider(ref)
        except (KeyError, TypeError, ValueError):
            return   # provider 未提供（未知 ref）或返非 (text, prereqs) 二元 → 当原子停·不抛崩（闭项守）
        if not isinstance(prereqs, (list, tuple)):
            prereqs = []   # malformed prereqs（None 等·caller 违约）→ 当叶子·不抛崩（闭项守·审 F3）
        visited.add(ref)
        budget_left[0] -= 1
        count += 1
        teacher.define(ref, text)   # hash-recorded（MODE_RECORD）/ hash-lookup（MODE_REPLAY）/ None（OFF）
        for pre in prereqs:
            _go(pre, depth + 1)

    _go(root_ref, 0)
    return count


def _valid_concept_ref(ref) -> bool:
    """ConceptRef 形状校验（tuple[int, int]）·malformed ref（caller 违约）→ False → skip 不抛崩。

    teacher.define 内 `sid, lid = ref` 解包在 assert_int 前·3-tuple 会 ValueError 崩·故入口校验守
    闭项（审 F2：provider 返 malformed prereq 不致整递归 abort）。
    """
    return (isinstance(ref, tuple) and len(ref) == 2
            and isinstance(ref[0], int) and isinstance(ref[1], int)
            and not isinstance(ref[0], bool) and not isinstance(ref[1], bool))
