"""cognition.understanding.latex_intake — LaTeX 数学记号子集 → arith DSL（换皮·复用 _ArithBuilder）。

打通算术域 LaTeX intake：把 LaTeX 数学记号子集 → Python lambda DSL 字符串 → 喂
build_composes_from_arith（_ArithBuilder 全复用·零新 builder/边/opcode·换皮=好典型）。
权威设计 = doc/重来_符号算子与一种最根本表达设计补充.md §五（intake 分流：LaTeX=字面文法承载→
墙内 parser 合法·裸文本/字符=D 墙走 observe·永不进 parser）。2 对抗智能体验证落地（见 doc §五②）。

子集（字面文法承载 → 墙内 parser 合法）：
  \\sum_{X=LO}^{HI} BODY    → Sigma(LO, HI, BODY)    （X 局部重命名→DSL 魔法名 i）
  \\prod_{X=LO}^{HI} BODY   → Prod(LO, HI, BODY)
  \\frac{A}{B}              → (A)/(B)                （/ 走 OPCODE_DIV 精确有理·doc §六）
  A^{K}（K 字面非负 int）/ A^K 单字符  → A**K         （负/变量指数 fail-loud）
  \\cdot  \\times           → *
  \\left(  \\right)         → ( )
  + - * / 裸变量 数字       → 直通

拒清单（fail-loud UnsupportedConstruct·非静默）：
  - 超越函数值/超越数（C7 超越值墙·查表=硬编码数学词典双违·违"外部只启发绝不注入"+C7）：
    \\sin \\cos \\tan \\cot \\sec \\csc \\log \\ln \\exp \\sqrt
  - 极限/积分值计算（C7·建树=空壳 theater·vm_proof 永远 None）：\\lim \\int \\oint
  - 语法不支持：负/变量指数 · 连续上标 · 隐式乘法 n(n+1) · 复合 body 须 {..}/(..) 包裹 ·
    自定义宏 \\newcommand · 复杂环境 \\begin · DSL 无比较 \\leq/\\geq/\\neq
  - DSL 单魔法名限制：内层 body 引用外层 index（DSL i 仅达本层）· 自由变量名撞魔法名 i/a

i/a 魔法名（DSL hardwired·arith_observe._IDX_NAME=i/_ACC_NAME=a）：LaTeX 任意 index 名 X 一律
  局部重命名→i（DSL 只认 i·非写死语义·对齐 DSL 关键字同 def/while）。lo/hi 可引用外层 index
  （DSL scope 栈解析为外层 i·建在外层 scope）·body 仅达本层 i（引用外层 index→fail-loud·不可表达）。
  嵌套同名 index 合法（DSL scope 栈 shadow·arith_observe._scope reversed 查找保证内层优先）。
  Recur 无 LaTeX 标准记号 → LaTeX intake 不映射 Recur（DSL 仍支持·用户直写 DSL）。

铁律：纯整数（无浮点·遇小数点 fail-loud）/ 确定性 bit-identical（单遍前序 tokenize·list 序·禁 set 收集·
  AST tuple·自由变量 sorted 锁序）/ 只标准库（无 LaTeX 第三方库·自建子集 parser）/ fail-loud 超出子集 /
  复用 _ArithBuilder（零新 builder/边/opcode）/ 不写死（命令白名单是元定义 enum·非语义规则）/
  核心无墙钟（全词法/语法确定·无 time/random）。
诚实边界：输出 = Python lambda DSL **字符串**（非 AST·避耦合 _ArithBuilder 内部状态破坏 bit-identical
  计数器契约）+ param_order（lambda 参数 sorted 字典序·vm_proof_fn input_args 须按此序传·语义契约）。
"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.arith_observe import (
    UnsupportedConstruct, build_composes_from_arith,
)
from pure_integer_ai.cognition.shared.types import ConceptRef

# ---- token kind（单遍前序·list 序·禁 set 收集） ----
_T_CMD = "CMD"
_T_LBRACE = "LBRACE"
_T_RBRACE = "RBRACE"
_T_CARET = "CARET"
_T_UNDER = "UNDER"
_T_LPAREN = "LPAREN"
_T_RPAREN = "RPAREN"
_T_PLUS = "PLUS"
_T_MINUS = "MINUS"
_T_STAR = "STAR"
_T_SLASH = "SLASH"
_T_NUM = "NUM"
_T_NAME = "NAME"
_T_EQ = "EQ"
_T_EOF = "EOF"

# 命令白名单（元定义 enum·frozenset 查询无序依赖·非语义规则）
_SUPPORTED_CMDS = frozenset({
    "\\sum", "\\prod", "\\frac", "\\cdot", "\\times", "\\left", "\\right",
})
# 显式拒命令（C7 墙外 / 查表双违·非静默·带理由消息）
_REJECT_CMDS: dict[str, str] = {
    "\\sqrt": "\\sqrt 不支持（无 OPCODE_SQRT·C7 超越值墙·纯整数 VM 须用 Recur 逼近或引用已学结论）",
    "\\sin": "\\sin 不支持（超越函数值·C7 超越值墙·查表=硬编码数学词典双违）",
    "\\cos": "\\cos 不支持（超越函数值·C7 超越值墙）",
    "\\tan": "\\tan 不支持（超越函数值·C7）",
    "\\cot": "\\cot 不支持（超越函数值·C7）",
    "\\sec": "\\sec 不支持（超越函数值·C7）",
    "\\csc": "\\csc 不支持（超越函数值·C7）",
    "\\log": "\\log 不支持（超越函数值·C7）",
    "\\ln": "\\ln 不支持（超越函数值·C7）",
    "\\exp": "\\exp 不支持（超越函数值·C7）",
    "\\lim": "\\lim 不支持（极限值计算 defer·CAS·C7·引用已学 lim/求导关系结论走算子 inline L1/L1.5）",
    "\\int": "\\int 不支持（定积分 defer·CAS·建树空壳 theater·vm_proof 永远 None）",
    "\\oint": "\\oint 不支持（环路积分 defer·CAS）",
    "\\binom": "\\binom 不支持（组合数须 Recur·非字面文法承载）",
    "\\leq": "\\leq 不支持（DSL 无比较·cond 由 builder 内部生成为 LT）",
    "\\geq": "\\geq 不支持（DSL 无比较）",
    "\\neq": "\\neq 不支持（DSL 无比较）",
}
_MAGIC = frozenset({"i", "a"})   # DSL 魔法名（index/累加器·lambda args 禁用）


def _tokenize(s: str) -> list[tuple[str, str]]:
    """LaTeX 子集 → 有序 token list（单遍前序·禁 set·遇小数点/未知字符 fail-loud）。"""
    out: list[tuple[str, str]] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\n\r":
            i += 1
            continue
        if c == "\\":
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            if j == i + 1:
                raise UnsupportedConstruct(f"LaTeX 命令语法错误（反斜杠后须字母）: {s[i:i + 4]!r}")
            out.append((_T_CMD, s[i:j]))
            i = j
            continue
        if c.isalpha():
            j = i
            while j < n and (s[j].isalpha() or s[j].isdigit()):
                j += 1
            out.append((_T_NAME, s[i:j]))
            i = j
            continue
        if c.isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            if j < n and s[j] == ".":
                raise UnsupportedConstruct("小数点不支持（纯整数铁律·须用 \\frac{p}{q} 表有理）")
            out.append((_T_NUM, s[i:j]))
            i = j
            continue
        simple = {"{": _T_LBRACE, "}": _T_RBRACE, "^": _T_CARET, "_": _T_UNDER,
                  "(": _T_LPAREN, ")": _T_RPAREN, "+": _T_PLUS, "-": _T_MINUS,
                  "*": _T_STAR, "/": _T_SLASH, "=": _T_EQ}
        if c in simple:
            out.append((simple[c], c))
            i += 1
            continue
        raise UnsupportedConstruct(f"词法不支持的字符: {c!r}（ord={ord(c)}）")
    out.append((_T_EOF, ""))
    return out


# ---- AST（tuple·确定性·禁 dict） ----
# ('num', int) / ('var', str) / ('binop', op, left, right) / ('pow', base, exp_int)
# ('sigma', lo, hi, idx_name, body) / ('prod', lo, hi, idx_name, body)


class _Parser:
    """递归下降 parser（expr → term → power → atom·运算符优先级 +- < */<^ < 原子）。

    持 token list + pos。\\sum/\\prod body：花括号 {expr}（完整 expr）或单 power（atom+可选^）·
    复合 body 须显式括号包裹（裸 \\sum_{..}^{..} a \\cdot b fail-loud·body=a 后 \\cdot b 残留）。
    """

    def __init__(self, toks: list[tuple[str, str]]) -> None:
        self._t = toks
        self._p = 0

    def _peek(self) -> tuple[str, str]:
        return self._t[self._p]

    def _next(self) -> tuple[str, str]:
        tok = self._t[self._p]
        self._p += 1
        return tok

    def _expect(self, kind: str) -> tuple[str, str]:
        tok = self._next()
        if tok[0] != kind:
            raise UnsupportedConstruct(f"期望 {kind}·得 {tok}")
        return tok

    def parse_top(self) -> tuple:
        node = self._parse_expr()
        if self._peek()[0] != _T_EOF:
            raise UnsupportedConstruct(f"顶层表达式后有多余 token: {self._peek()}")
        return node

    def _parse_expr(self) -> tuple:
        node = self._parse_term()
        while self._peek()[0] in (_T_PLUS, _T_MINUS):
            op = self._next()[1]
            node = ("binop", op, node, self._parse_term())
        return node

    def _parse_term(self) -> tuple:
        node = self._parse_power()
        while True:
            k = self._peek()[0]
            if k == _T_STAR:
                self._next()
                node = ("binop", "*", node, self._parse_power())
            elif k == _T_SLASH:
                self._next()
                node = ("binop", "/", node, self._parse_power())
            elif k == _T_CMD and self._peek()[1] in ("\\cdot", "\\times"):
                self._next()
                node = ("binop", "*", node, self._parse_power())
            else:
                break
        return node

    def _parse_power(self) -> tuple:
        base = self._parse_atom()
        if self._peek()[0] == _T_CARET:
            self._next()
            exp = self._parse_exponent()
            if exp[0] != "num" or exp[1] < 0:
                raise UnsupportedConstruct(
                    f"指数须字面非负整数·得 {exp}（负指数须用 \\frac·变量指数须用 Recur）")
            if self._peek()[0] == _T_CARET:
                raise UnsupportedConstruct("连续上标不支持（嵌套幂须字面·写 A^{k}）")
            return ("pow", base, exp[1])
        return base

    def _parse_exponent(self) -> tuple:
        if self._peek()[0] == _T_LBRACE:
            self._next()
            neg = False
            if self._peek()[0] == _T_MINUS:   # ^{-...} 前导负号（负指数·交 _parse_power 校验拒）
                self._next()
                neg = True
            e = self._parse_expr()
            self._expect(_T_RBRACE)
            if neg:
                if e[0] == "num":
                    return ("num", -e[1])     # 负字面指数·交 exp<0 校验拒
                raise UnsupportedConstruct("指数须字面非负整数（负变量/复合指数不支持·须用 \\frac）")
            return e
        tok = self._next()
        if tok[0] == _T_NUM:
            return ("num", int(tok[1]))
        if tok[0] == _T_NAME:
            return ("var", tok[1])
        raise UnsupportedConstruct(f"指数 token 不支持: {tok}")

    def _parse_atom(self) -> tuple:
        k, txt = self._peek()
        if k == _T_NUM:
            self._next()
            return ("num", int(txt))
        if k == _T_NAME:
            self._next()
            return ("var", txt)
        if k == _T_LPAREN:
            self._next()
            e = self._parse_expr()
            self._expect(_T_RPAREN)
            return e
        if k == _T_CMD:
            if txt == "\\left":
                self._next()
                self._expect(_T_LPAREN)
                e = self._parse_expr()
                rt = self._next()
                if not (rt[0] == _T_CMD and rt[1] == "\\right"):
                    raise UnsupportedConstruct("\\left( 须配 \\right)")
                self._expect(_T_RPAREN)
                return e
            if txt == "\\frac":
                self._next()
                self._expect(_T_LBRACE)
                num = self._parse_expr()
                self._expect(_T_RBRACE)
                self._expect(_T_LBRACE)
                den = self._parse_expr()
                self._expect(_T_RBRACE)
                return ("binop", "/", num, den)
            if txt in ("\\sum", "\\prod"):
                return self._parse_sum_prod(txt)
            if txt in _REJECT_CMDS:
                raise UnsupportedConstruct(_REJECT_CMDS[txt])
            if txt in _SUPPORTED_CMDS:
                raise UnsupportedConstruct(f"命令 {txt} 出现在非预期位置（仅 \\cdot/\\times 在项内·\\right 配 \\left）")
            raise UnsupportedConstruct(f"未知/不支持命令: {txt}")
        raise UnsupportedConstruct(f"非预期 token: {(k, txt)}")

    def _parse_sum_prod(self, cmd: str) -> tuple:
        self._next()   # consume \sum/\prod
        # 严格顺序 _{X=LO}^{HI}（_ 在前 ^ 在后·反序 fail-loud）
        self._expect(_T_UNDER)
        self._expect(_T_LBRACE)
        idx_name = self._expect(_T_NAME)[1]
        self._expect(_T_EQ)
        lo = self._parse_expr()
        self._expect(_T_RBRACE)
        self._expect(_T_CARET)
        self._expect(_T_LBRACE)
        hi = self._parse_expr()
        self._expect(_T_RBRACE)
        # body：花括号 {expr}（完整 expr）或单 power（atom + 可选 ^·复合须括号）
        if self._peek()[0] == _T_LBRACE:
            self._next()
            body = self._parse_expr()
            self._expect(_T_RBRACE)
        else:
            body = self._parse_power()
        kind = "sigma" if cmd == "\\sum" else "prod"
        return (kind, lo, hi, idx_name, body)


def _transform(node: tuple, scope: list[str], allow_outer: bool,
               free_acc: set[str]) -> tuple:
    """重命名 index 名→i（DSL hardwired）+ 收集自由变量（带 scope·单遍）。

    scope：外→内 index 名 list（innermost = scope[-1]）。allow_outer：lo/hi 上下文（外层 index
    引用 OK→i·DSL scope 栈解析为外层）vs body 上下文（外层 index 引用→fail-loud·DSL i 仅达本层）。
    自由变量（非 index·含名为 i/a 的裸引用）记入 free_acc（带 scope 判定·避免把 index 误当自由变量）·
    后由 latex_to_arith_dsl 查 _MAGIC 撞名 fail-loud + sorted 锁 param_order 确定性。
    """
    tag = node[0]
    if tag == "num":
        return node
    if tag == "var":
        name = node[1]
        if scope and name == scope[-1]:
            return ("var", "i")                       # 本层（innermost）index → i
        if name in scope[:-1]:
            if allow_outer:
                return ("var", "i")                   # lo/hi 引用外层 index → i（建在外层 scope）
            raise UnsupportedConstruct(
                f"内层 body 引用外层 index {name!r}（DSL 内层 i 仅达本层·不可表达·须展平或重构）")
        free_acc.add(name)                            # 自由变量（非 index·含裸 i/a）·后查撞魔法名
        return node                                   # 保留原名（DSL lambda 参数）
    if tag == "binop":
        return ("binop", node[1],
                _transform(node[2], scope, allow_outer, free_acc),
                _transform(node[3], scope, allow_outer, free_acc))
    if tag == "pow":
        return ("pow", _transform(node[1], scope, allow_outer, free_acc), node[2])
    if tag in ("sigma", "prod"):
        _, lo, hi, idx, body = node
        # lo/hi：idx 未入 scope·外层 index 引用 OK（allow_outer=True）
        lo_t = _transform(lo, scope, True, free_acc)
        hi_t = _transform(hi, scope, True, free_acc)
        # body：idx 入 scope 为 innermost·外层 index 引用 fail-loud（allow_outer=False）
        body_t = _transform(body, scope + [idx], False, free_acc)
        return (tag, lo_t, hi_t, idx, body_t)
    raise AssertionError(f"unreachable node: {node!r}")


def _codegen(node: tuple) -> str:
    """AST → DSL 字符串（binop/pow 强制括号·避 1/2/3 左结合歧义·保守确定性）。"""
    tag = node[0]
    if tag == "num":
        return str(node[1])
    if tag == "var":
        return node[1]                                 # 'i'（index）或自由变量名
    if tag == "binop":
        return f"({_codegen(node[2])}{node[1]}{_codegen(node[3])})"
    if tag == "pow":
        return f"({_codegen(node[1])}**{node[2]})"
    if tag == "sigma":
        _, lo, hi, _idx, body = node
        return f"Sigma({_codegen(lo)}, {_codegen(hi)}, {_codegen(body)})"
    if tag == "prod":
        _, lo, hi, _idx, body = node
        return f"Prod({_codegen(lo)}, {_codegen(hi)}, {_codegen(body)})"
    raise AssertionError(f"unreachable: {node!r}")


def latex_to_arith_dsl(latex: str) -> tuple[str, tuple[str, ...]]:
    """LaTeX 子集 → (dsl_lambda_str, param_order)。

    dsl_lambda_str：喂 build_composes_from_arith 的 Python lambda 字符串（魔法名 i·如
      "lambda n: Sigma(1, n, i)"）。
    param_order：lambda 参数 sorted 字典序（vm_proof_fn input_args 须按此序传·语义契约·
      空 = nullary "lambda: ..."）。

    超出子集 fail-loud UnsupportedConstruct（复用 arith_observe.UnsupportedConstruct·同 code_observe 子集范式）。
    """
    if not latex or not latex.strip():
        raise UnsupportedConstruct("LaTeX 输入为空")
    toks = _tokenize(latex)
    ast = _Parser(toks).parse_top()
    free_acc: set[str] = set()
    ast = _transform(ast, [], True, free_acc)          # 顶层 scope 空·自由变量收集进 free_acc
    free = sorted(free_acc)                            # ★ 确定性：sorted 锁 param_order 序
    bad = _MAGIC & set(free)
    if bad:
        raise UnsupportedConstruct(
            f"自由变量名 {sorted(bad)} 撞 DSL 魔法名 i/a（i/a 是 index/累加器魔法名·请重命名 LaTeX 变量）")
    body_str = _codegen(ast)
    if free:
        dsl = f"lambda {', '.join(free)}: {body_str}"
    else:
        dsl = f"lambda: {body_str}"
    return dsl, tuple(free)


def build_composes_from_latex(latex_source: str, *, concept_index, edge_store,
                              backend, space_id: int, source: int,
                              root_ref: ConceptRef) -> ConceptRef:
    """LaTeX 数学记号 → arith DSL → COMPOSES 树（复用 build_composes_from_arith·换皮）。

    返 root=struct_ref。超出子集 fail-loud UnsupportedConstruct。
    param_order（vm_proof_fn input_args 序）须经 latex_to_arith_dsl 另取（本函数只建树）。
    """
    dsl, _param_order = latex_to_arith_dsl(latex_source)
    return build_composes_from_arith(dsl, concept_index=concept_index, edge_store=edge_store,
                                     backend=backend, space_id=space_id, source=source,
                                     root_ref=root_ref)
