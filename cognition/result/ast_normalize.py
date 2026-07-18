"""cognition.result.ast_normalize — AST 结构规范化比（#730 路径 W·Mode A 构造性验证）。

unparse_composes 产 body 源码串 vs code_source（Python 全 def）→ AST 规范化（变量名统一 _v{k} + 剥签名）
→ ast.dump 比·结构同即 verified（skeleton 派生自 code_source·构造性必然·同 arith skeleton(args)==expected）。

**为何 normalize 比**（决断 3）：code_observe 丢变量名（保 make_variable(index)·code_observe.py:74 _var_index）
+ 丢函数名 + arg 名·序化器出 var{index} 人造名 ≠ code_source 原始名 → 字面不等·须结构级比。
normalize 按位置统一变量名（两树结构同→位置序对齐→_v{k} 对齐→ast.dump 等）。

**剥 FunctionDef 签名**（body 层比·code_observe 保 body 结构非签名）：code_source 取 funcs[0].body·
unparse 串取 Module.body（body-only·无 FunctionDef 包装）·两路归一到 body stmt list 再 normalize。
签名（函数名 + arg 名 + arg 列表）排除比（code_observe 部分丢·保 body 计算结构·非验签名）。
**arg 变量对齐**：code_observe arg 先分配 var index（build:83-84·0..n_args-1=params）·body Name 经 AST walk
遇见序统一 _v{k}·params 与 locals 按位置自动对齐（code_source body 的 a/n 与 unparse 的 var0/var1 同结构位置）。

**AST 标准库**（Python ast·无外部依赖·守"只标准库"铁律）。NodeTransformer 统一 Name.id 按 ast.walk 遇见序。

铁律：纯读（零 storage·零写）/ 确定性（ast.walk BFS 确定序 + 名映射确定·bit-identical）/ 单向依赖（cognition=result
  L5·零 pure_integer_ai 内部 import·只 stdlib ast·无向下破层）/ 不写死（通用 AST 结构比·非特定程序）。
诚实边界：stable≠correct（结构对称 COMPOSES→源码 AST 形状·非字面还原·变量名 var{index} 人造·原始名 defer）/
  Mode A 构造性（unparse(__prog_*)==code_source 构造性必然·非真生成）/ **ast.unparse 跨 Python 版本不保证
  bit-identical**（本模块用 ast.dump 比·dump 跨 patch 版本稳定·但 CI/生产须锁同一 Python 版本·同版本同输入
  确定输出·决断 3 风险 3·CPython ast 实现确定）/ AugAssign 歧义（统一 Assign 形式比·见 composes_unparse docstring）。
"""
from __future__ import annotations

import ast


def normalize_code_body(src: str) -> str:
    """源码串 → 规范化 AST dump（变量名统一 _v{k}·AugAssign→Assign 归一·剥 FunctionDef 签名·body 层）。

    src : Python 源码串（code_source 全 def / unparse_composes body 串）。

    规则：
      1. ast.parse（语法错误 → raise·fail-loud·不伪造 verified）。
      2. 若 Module.body 含 FunctionDef → 取**首个** FunctionDef.body（剥签名·code_observe 首版一段一函数）。
         否则 → Module.body（body-only unparse 串·无 def 包装）。
      3. **AugAssign→Assign 归一**（审2 P1-1 修）：code_observe 对 `n+=1` 与 `n=n+1` 产**同构 COMPOSES 树**·
         序化器统一出 Assign 形式 → 须把 code_source 的 AugAssign 也归一为 Assign·否则同构树假阴性 not verified。
         n op= v → n = n op v（Name target·code_observe 仅支持 Name·非 Name 透传不转）。
      4. NodeTransformer 统一 Name.id → _v{k}（ast.walk 遇见序·两树结构同则位置对齐）。
      5. ast.dump（annotate_fields=False·indent 无·单行结构串·确定性）。

    返规范化 dump 串（两 src 结构同 → 同串·bodies_match 判等）。
    """
    tree = ast.parse(src)   # SyntaxError 上抛（fail-loud·Mode A 诚实判 not-verified）
    body = _extract_body(tree)
    # AugAssign→Assign 归一（code_observe 同构树·序化器 Assign 形式·须双向归一·审2 P1-1）
    _flatten_augassign(body)
    # 统一变量名（in-place·Name.id 按遇见序 _v{k}）
    _rename_names(body)
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.dump(mod, annotate_fields=False)


def bodies_match(unparsed: str, code_source: str) -> bool:
    """unparse_composes body 串 vs code_source·AST 规范化后结构等价判（#730 Mode A 验证）。

    两路 normalize_code_body → dump 比·等即 body 计算结构同构（verified）。
    语法错误 / 结构异 → False（诚实判 not-verified·不伪造）。
    """
    try:
        return normalize_code_body(unparsed) == normalize_code_body(code_source)
    except SyntaxError:
        return False   # unparse 串或 code_source 语法错 → 结构异 → not-verified（诚实）


def _extract_body(tree: ast.Module) -> list[ast.stmt]:
    """取 body stmt list：含 FunctionDef 取首函数 body·否则取 Module.body。"""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return list(node.body)
    return list(tree.body)


def _flatten_augassign(stmts: list[ast.stmt]) -> None:
    """AugAssign → Assign 归一（in-place·审2 P1-1 修）。

    code_observe 对 `n op= v` 与 `n = n op v` 产**同构 COMPOSES 树**（STORE+BINOP·code_observe.py:142-154）·
    序化器统一出 Assign 形式 `var{t} = var{t} op v`·故 code_source 的 AugAssign 须归一为 Assign 才能对齐
    （否则同构树假阴性 not verified）。n op= v → n = n op v（Name target·code_observe 仅支持 Name·
    非 Name target 透传不转·白名单内无）。

    NodeTransformer generic_visit 先递归子（value 子树内可能嵌套·虽 code_observe 白名单浅·防御）。
    """
    mod = ast.Module(body=stmts, type_ignores=[])
    _AugAssignToAssign().visit(mod)


class _AugAssignToAssign(ast.NodeTransformer):
    """AugAssign(Name target, op, value) → Assign([Name(t,Store)], BinOp(Name(t,Load), op, value))。"""

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)   # 先递归 value 子树
        if not isinstance(node.target, ast.Name):
            return node   # code_observe 仅支持 Name target·非 Name 透传（白名单内无·防御）
        tid = node.target.id
        return ast.Assign(
            targets=[ast.Name(id=tid, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id=tid, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            ),
        )


def _rename_names(stmts: list[ast.stmt]) -> None:
    """stmt list 上 Name.id 按 ast.walk 遇见序统一 _v{k}（in-place·确定性）。

    ast.walk BFS 确定序·两结构同树 Name 遇见序同 → _v{k} 分配对齐。
    仅改 Name.id（arg 节点在签名·已剥·body 无 arg 节点）。
    """
    mod = ast.Module(body=stmts, type_ignores=[])
    renames: dict[str, str] = {}
    for node in ast.walk(mod):
        if isinstance(node, ast.Name):
            if node.id not in renames:
                renames[node.id] = f"_v{len(renames)}"
            node.id = renames[node.id]
