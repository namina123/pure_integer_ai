"""teacher.teacher_boundary — 白黑词汇表机械核查（§9 A2 执行点·外部只启发绝不注入边语义）。

verify_teacher_boundary(response) -> list[Violation]
  教师响应须带 kind 字段（内容类型）·机械核查：
    白词汇表（允许注入的内容类型）= {NAME, DEFINE, REWARD, ERROR_LABEL}
      —— 教师只给"事实判断/定义/奖励标签/错误标签"·非推理规则。
    黑词汇表（禁止注入·系统自有结构判断）= {PLACEMENT, ORDER, PUNCTUATION, COOCCURS}
      —— 词位/语序/标点/共现是结构层系统自有·教师注入=注入边语义违铁律。
  核查规则（机械·非语义）：
    ① kind ∈ 黑表 → Violation(BLACKLISTED) 拒写
    ② kind ∉ 白表 → Violation(NOT_WHITELISTED) 拒写
    ③ response text 含黑表关键词（机械子串·非语义判）→ Violation(STRUCT_DIRECTIVE) 拒写
  返 [] = 通过可写 / 非空 = 拒写（caller 丢弃·不进核心·防教师越界注入边语义）。

**外部只启发绝不注入边语义**的执行点（§二 line56 / §9 A2）：教师给的是"事实判断"
（A 是否导致 B / A 是否同指 B / 答案是否正确）非"推理规则/边关系真伪定义"。白黑词汇表
是机械闸非语义判（kind 字段 + 关键词子串·不判内容真伪·真伪走检疫闸/promote/reward）。

铁律：不写死（白黑表是元定义层枚举非语义规则·§九元定义层例外）/ 纯整数（kind 整数枚举）/
  外部只启发（机械核查拒越界·教师不定义边关系真伪）。
诚实边界：核查是机械的（kind+子串）·不判教师内容真伪（真伪走检疫闸 sign=0 / reward / promote）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---- 内容类型 kind 枚举（元定义层·白黑词汇表·§9 A2） ----

# 白词汇表（允许注入·教师给事实判断非推理规则）
KIND_NAME = 1          # 命名（概念词形/同指别称·性质A 来源②③）
KIND_DEFINE = 2        # 定义（元定义/知识 define·递归展开§十一防塌C8）
KIND_REWARD = 3        # 奖励标签（G5/C6 Mode A 教师 ground-truth·断奶前）
KIND_ERROR_LABEL = 4   # 错误标签（教师标错误类型·训练信号）

# 黑词汇表（禁止注入·系统自有结构判断·注入=注入边语义违铁律）
KIND_PLACEMENT = 101    # 词位（结构层系统自有·role_seq 涌现非教师给）
KIND_ORDER = 102        # 语序（PRECEDES 结构真值·observe 建·教师不给）
KIND_PUNCTUATION = 103  # 标点（observe parse_segment·教师不给）
KIND_COOCCURS = 104     # 共现（COOCCURS observe 段内分桶·教师不给）

WHITELIST: frozenset[int] = frozenset({KIND_NAME, KIND_DEFINE, KIND_REWARD, KIND_ERROR_LABEL})
BLACKLIST: frozenset[int] = frozenset({KIND_PLACEMENT, KIND_ORDER, KIND_PUNCTUATION, KIND_COOCCURS})

# 黑表关键词（机械子串扫描·防教师 text 里塞结构指令·非语义判）
# 用稳定英文 token + 中文关键词·机械匹配·不涉词义规则。
_BLACK_KEYWORDS: tuple[str, ...] = (
    "placement", "order:", "punctuation", "cooccurs",
    "词位", "语序", "标点", "共现",
)

# 违例类型
V_BLACKLISTED = "blacklisted"             # kind 在黑表
V_NOT_WHITELISTED = "not_whitelisted"    # kind 不在白表
V_STRUCT_DIRECTIVE = "struct_directive"  # text 含黑表关键词（结构指令越界）
V_NO_KIND = "no_kind"                    # 响应缺 kind 字段


@dataclass(frozen=True)
class Violation:
    """教师越界违例（机械核查产出·caller 拒写）。"""
    code: str
    kind: int | None
    detail: str


def verify_teacher_boundary(response: dict[str, Any]) -> list[Violation]:
    """白黑词汇表机械核查（§9 A2 执行点）。

    response 须含 kind（int·内容类型）·可选 text（str·机械子串扫描）。
    返 [] = 通过可写 / 非空 = 拒写（不进核心·防教师越界注入边语义）。
    """
    out: list[Violation] = []
    kind = response.get("kind")
    if kind is None:
        out.append(Violation(V_NO_KIND, None, "教师响应缺 kind 字段·无法机械核查"))
        return out
    # ① 黑表 → 拒
    if kind in BLACKLIST:
        out.append(Violation(V_BLACKLISTED, kind, f"kind={kind} 在黑词汇表·系统自有结构判断禁止注入"))
        return out   # 黑表最严重·直接返
    # ② 不在白表 → 拒
    if kind not in WHITELIST:
        out.append(Violation(V_NOT_WHITELISTED, kind, f"kind={kind} 不在白词汇表·教师只给{name_whitelist()}"))
        return out
    # ③ text 机械子串扫描（黑关键词·防结构指令越界·非语义判）
    text = response.get("text")
    if isinstance(text, str):
        low = text.lower()
        for kw in _BLACK_KEYWORDS:
            if kw in low or kw in text:
                out.append(Violation(V_STRUCT_DIRECTIVE, kind,
                                     f"text 含黑关键词 '{kw}'·结构指令越界拒写"))
                break
    return out


def is_acceptable(response: dict[str, Any]) -> bool:
    """便捷：教师响应是否通过机械核查（可写）。"""
    return not verify_teacher_boundary(response)


def name_whitelist() -> str:
    """白词汇表人类可读名（错误信息用·非语义规则）。"""
    return "{name/define/reward/error_label}"
