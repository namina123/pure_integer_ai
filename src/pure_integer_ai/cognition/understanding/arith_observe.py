"""cognition.understanding.arith_observe — 算术域 数学记号 DSL → COMPOSES 程序建造者（A3 兄弟件）。

打通算术域上游：把数学记号（闭式/Σ/Π/recurrence）→ COMPOSES 树 + operator_of/operand_of/
immediate_of/store_target_of·落 EDGE_COMPOSES 边 + composes_attr 属性·喂 compile_graph
（经 ConceptGraph.read_composes_tree 读回·vm_proof_fn 执行验对错）。

权威设计 = doc/重来_算术域observe设计补充.md。code 跟它。镜像 code_observe 的 _ComposesBuilder
模式·但算术域有 4 处领域关键差异（对抗审查拍出·doc §三 5 必改）：
  1. **入口接 lambda**（非 FunctionDef）：ast.parse → Expr(Lambda) → args→make_variable + body。
  2. **`/`→OPCODE_DIV 精确有理除**（非拒·doc §六）：算术 DSL `/` 是符号记号→VM 精确除·不经 Python `/`。
  3. **词法作用域栈**（非单层 _var_sid·doc §五）：Sigma/Prod body 内 `i`=本层索引·Recur body 内
     `a`=累加器/`i`=索引·经 scope 栈解析（避单和 KeyError/嵌套和静默错值）。
  4. **Sigma/Prod/Recur→CTRL_WHILE+累加器迭代块**（doc §五·复用 A2 CTRL_WHILE lower）。

DSL（lambda 表达式·body 用 Sigma/Prod/Recur/Lim 调用 + 算术）：
  lambda n: Sigma(1, n, i)        # Σ_{i=1}^{n} i  （i 隐式索引魔法名）
  lambda n: Prod(1, n, i)         # n!
  lambda n: Recur(1, n, a*i)      # 阶乘（a=累加器·i=索引·单状态递推）
  lambda n: n*(n+1)/2             # 闭式（/ 精确有理）
  lambda n: Lim(...)              # fail-loud·0/0 族极限值计算 defer（CAS·C7 超越值墙）·表达层平凡（引用已学 lim 结论=算子 inline·L1/L1.5）
  lambda: S100 + 1                # 引用已学 nullary 结论 S100（Name 路径→建造期 inline 嫁接·L1）
  lambda n: square(n)             # 引用已学参数化算子 square（Call 路径→inline + β-归约·L1.5·
                                 # doc/重来_符号算子与一种最根本表达设计补充.md）

魔法名（DSL 关键字·非写死语义·同 def/while）：Sigma/Prod body `i`=本层循环索引·Recur body
  `a`=累加器 `i`=索引。Lambda args 禁名 i/a（fail-loud 避撞）。

铁律：纯整数（属性全 int·`/`走 OPCODE_DIV 精确除零浮点·immediate den=1·Rational 经 make 运行时产生非
  build 期求值）/ 确定性（单一递归前序遍历·禁 ast.walk·surface 序号+var index 共用计数器·scope 栈
  插入序 dict·order_index 显式·禁 set→list）/ 核心无墙钟 / 不写死（AST 节点类型→opcode 通用映射）。
诚实边界：支持子集=闭式算术(+-*/·`**`字面非负指数)+Sigma/Prod/Recur(单状态)+引用已学算子
  （nullary Name 路径 inline·L1 / 参数化 Call 路径 inline+β-归约·L1.5）·超出 fail-loud
  UnsupportedConstruct·lim【值】计算 defer（0/0 族·求导+极限原语库=符号 CAS·C7 超越值墙·纯整数 VM 不可达无穷迭代）/
  表达层平凡（引用已学 lim/求导【关系】结论=算子 inline·L1/L1.5·无特设 lim 机制·C7 只拦超越值不拦变换关系）/
  多状态 recurrence（Fibonacci）defer / 跨段 inline defer /
  observe 自动命名算子 defer / stable≠correct。
"""
from __future__ import annotations

import ast
import sys

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV, OPCODE_LT, OPCODE_NOP,
    OPCODE_POW_PATTERN,
    make_variable,
)
from pure_integer_ai.vm.graph_compile import CTRL_WHILE
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_OPERATOR, NODE_CONCEPT
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, read_composes_attrs,
    ATTR_OPERATOR, ATTR_CTRL_TAG, ATTR_OPERAND,
    ATTR_IMMEDIATE, ATTR_STORE_TARGET, ATTR_OPERATOR_DEF, ATTR_ARITY,
)
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.result.graph_view import ConceptGraph


class UnsupportedConstruct(ValueError):
    """算术记号超出支持子集（fail-loud·非静默跳过·非兜底）。

    算术域首版支持子集 = 闭式算术(+-*/·`/`精确有理·`**`字面非负指数)+Sigma/Prod/Recur(单状态)
    （doc §四）。超出子集 fail-loud 拒绝·诚实 scope。lim 全 defer（C7）/ 多状态 recurrence defer。
    """


# ---- 魔法名（DSL 关键字·Sigma/Prod body i=索引·Recur body a=累加器 i=索引） ----
_IDX_NAME = "i"   # 循环索引魔法名（Sigma/Prod/Recur body 内）
_ACC_NAME = "a"   # 累加器魔法名（Recur body 内）
_RESERVED = frozenset({_IDX_NAME, _ACC_NAME})   # lambda args 禁用（fail-loud 避撞）

# ---- DSL 迭代构造关键字（Call func name） ----
_SIGMA = "Sigma"   # Σ 求和：(lo, hi, body)·acc=0·acc+=body
_PROD = "Prod"     # Π 求积：(lo, hi, body)·acc=1·acc*=body
_RECUR = "Recur"   # 单状态递推：(init, count, body)·acc=init·acc=body(a,i)
_LIM = "Lim"       # 极限值计算 → fail-loud defer（0/0 族·CAS·C7 超越值墙·表达层平凡：引用已学关系结论=算子 inline）
_POW = "Pow"       # 符号 Pow（变量指数·Phase 2b）→ Pow 节点（OPCODE_POW_PATTERN·pattern-level 不展开 MUL·符号变换 LHS/RHS 模板用·d/dx Pow(base,n)）

# COMPOSES 子树节点的结构属性 kind（deep-copy 仅复制这些·防御性显式允许表·
# ATTR_OPERATOR_DEF/ATTR_ARITY 挂 name 节点不在子树内·复制循环遇不到）。
_STRUCTURAL_KINDS = frozenset({
    ATTR_OPERATOR, ATTR_CTRL_TAG, ATTR_OPERAND, ATTR_IMMEDIATE, ATTR_STORE_TARGET,
})


# ---- 算子名→struct_ref 注册（L1 inline-on-reference 查表·doc/重来_符号算子与一种最根本表达设计补充.md §三3） ----

def register_arith_operator(backend, concept_index, name: str,
                            struct_ref: ConceptRef, *, arity: int = 0) -> ConceptRef:
    """注册算子名→struct_ref + arity（L1.5 inline 查表·非语义注入·同 Sigma/i 魔法名范式）。

    在 struct_ref 同 space 建 name 概念点·挂 ATTR_OPERATOR_DEF（int_a=struct_ref space_id·
    int_b=local_id）+ ATTR_ARITY（int_a=arity·L1.5 β-归约 param i↔make_variable(i) 分类）。
    name 节点永不在 COMPOSES 子树内 → 复制循环遇不到 ATTR_ARITY/ATTR_OPERATOR_DEF（无 spurious 复制）。
    inline 时 _ArithBuilder._try_inline_learned 经 concept_index.lookup(name)
    →read_composes_attrs→ATTR_OPERATOR_DEF→struct_ref + ATTR_ARITY→arity→deep-copy + β-归约。

    返 name 节点 ConceptRef（§8.7-洗·op_confidence 台账键·auto_discover_operators 捕获填
    DiscoveredOperator.name_ref·recognize 择优读 + _verify_generalization 写置信度用它）。

    arity=0（默认）= nullary 算子（L1·Name 路径引用）·arity>0 = 参数化算子（L1.5·Call 路径 + β-归约）。
    误把参数化算子注册成 arity=0 → inline 时 fail-loud（Name 撞自由变量 / Call 撞 arity 不匹配）·安全网。

    fail-loud 拒重名冲突：同 name 已映射到不同 (struct_ref, arity) → UnsupportedConstruct。
    同 (name, struct_ref, arity) 重注册幂等（不变）。
    跨 space inline = defer（name 概念点与 struct_ref 同 space）。
    """
    sid, lid = struct_ref
    assert_int(sid, lid, arity, _where="register_arith_operator")
    name_ref = concept_index.ensure(name, space_id=sid,
                                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    existing = read_composes_attrs(backend, name_ref)
    if ATTR_OPERATOR_DEF in existing:
        prev_def = existing[ATTR_OPERATOR_DEF]
        prev_arity = existing.get(ATTR_ARITY, (0, 0))[0]
        if prev_def != (sid, lid) or prev_arity != arity:
            raise UnsupportedConstruct(
                f"算子名重名冲突: {name!r} 已映射→{prev_def}(arity={prev_arity})"
                f"·试图重映射→{(sid, lid)}(arity={arity})"
                f"（fail-loud 拒歧义·同义重注册须同 struct_ref+arity）")
        return name_ref   # 幂等：同 (name, struct_ref, arity) 重注册不变·仍返 name_ref
    record_composes_attr(backend, ref=name_ref, kind=ATTR_OPERATOR_DEF,
                         int_a=sid, int_b=lid)
    record_composes_attr(backend, ref=name_ref, kind=ATTR_ARITY, int_a=arity)
    return name_ref


class _ArithBuilder:
    """算术 COMPOSES 树建造者（持 concept_index/edge_store/backend + 遍历状态 + 词法 scope 栈）。

    单一递归前序遍历（_build_expr/_build_* 递归·先建当前节点再递归子）·surface 序号 + var index
    共用此前序遍历计数器（确定性 bit-identical）。词法 scope 栈解析魔法名 i/a（doc §三必改#1）。
    """

    def __init__(self, *, concept_index, edge_store, backend,
                 space_id: int, source: int, root_ref: ConceptRef) -> None:
        self._ci = concept_index
        self._es = edge_store
        self._b = backend
        self._space_id = space_id
        self._source = source
        self._root = root_ref
        self._seq = 0                          # AST 节点序号（前序遍历计数器）
        self._next_var = 0                     # 变量 index 计数器（lambda args 先 + internal 后）
        self._py_major = sys.version_info.major   # 钉死 Python 版本位（防跨版本 AST 字段序变）
        self._scope: list[dict[str, int]] = []    # 词法作用域栈（list[dict[name,sid]]·bottom=lambda args）
        self._args_sig = ""                    # lambda 参数签名（surface 隔离不同段）

    # ---- build 入口（Lambda·非 FunctionDef·doc §三必改#2） ----

    def build(self, lambdanode: ast.Lambda) -> ConceptRef:
        """建 lambda body COMPOSES 树·root=struct_ref（SEQ NOP·单子=body 树·结果留栈顶）。"""
        arg_names = [a.arg for a in lambdanode.args.args]
        base: dict[str, int] = {}
        for nm in arg_names:
            if nm in _RESERVED:
                raise UnsupportedConstruct(
                    f"lambda 参数禁用保留名（魔法名 {sorted(_RESERVED)}）: {nm}")
            sid = make_variable(self._next_var)
            self._next_var += 1
            base[nm] = sid
        self._scope.append(base)               # base scope = lambda args（input_args 位置映射）
        self._args_sig = "_".join(arg_names) if arg_names else "_"
        # root = struct_ref = SEQ NOP（函数根·单子=body 树·doc §三必改#2）
        record_composes_attr(self._b, ref=self._root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
        body_tree = self._build_expr(lambdanode.body)
        self._edge(self._root, body_tree, 0)
        self._scope.pop()
        return self._root

    # ---- 词法 scope 解析（doc §三必改#1·禁单层 _var_sid） ----

    def _resolve_name_or_inline(self, name: str) -> ConceptRef:
        """name → var_leaf(bound sid) | inline 子树根（已学 nullary 算子）| fail-loud。

        先查 scope 栈（bound var→var_leaf）·落空再查算子注册表（inline-on-reference·Name 路径=
        nullary only·call_args=None·参数化算子须走 _build_call 的 Call 路径·doc §三4）·都落空 fail-loud。
        """
        for frame in reversed(self._scope):
            if name in frame:
                return self._var_leaf(frame[name])
        inlined = self._try_inline_learned(name, call_args=None)
        if inlined is not None:
            return inlined
        raise UnsupportedConstruct(f"未绑定变量（且非已注册算子）: {name}")

    def _try_inline_learned(self, name: str,
                            call_args: list[ast.expr] | None = None) -> ConceptRef | None:
        """查算子名注册表·命中则 deep-copy + β-归约 struct_ref COMPOSES 子树到当前建造点·返嫁接子树根。

        L1 nullary（Name 路径·call_args=None·arity=0）+ L1.5 参数化（Call 路径·call_args=node.args·β-归约）
        统一于此（doc/重来_符号算子与一种最根本表达设计补充.md §三4/§三5）：
          name → concept_index.lookup → name 节点 attrs → ATTR_OPERATOR_DEF（struct_ref）+ ATTR_ARITY（arity）。
          · Name 路径（call_args=None·_resolve_name_or_inline 调）：arity 须 0·arg_subst={}（nullary inline·L1）。
          · Call 路径（call_args=node.args·_build_call 调）：len(call_args)==arity·在 caller scope 建实参子树·
            arg_subst={make_variable(i): arg_trees[i]}（param i↔make_variable(i) 不变量·build 保证 args 先分配从 index 0）。
          空壳 struct_ref（无 COMPOSES 子树）→ fail-loud。
          deep-copy + β-归约 委托 _deep_copy_subtree（param_subst=arg_subst·fail_on_external=True·自由变量 fail-loud）。

        arg_subst 按 key 查非迭代产出（确定性来自 copy_order BFS + sorted internal_sids·非 arg_subst 序）。
        嵌套 inline（A 体引 B）：B 在 A 建造期已 inline 进 A 树·inline A 复制预展开树·caller inline 期不递归
          解析算子（注册序或 fail-loud保证·对抗 Finding3）。
        循环引用：建造期解析已注册算子（struct_ref 已建·鸡生蛋不可自指）+ 结构 deep-copy（非再解析·不递归
          inline）→ 建造期天然无环（不需运行时环检测）。
        未命中注册表 → None（回退 fail-loud）。
        """
        name_ref = self._ci.lookup(name, self._space_id)
        if name_ref is None:
            return None
        attrs = read_composes_attrs(self._b, name_ref)
        op_def = attrs.get(ATTR_OPERATOR_DEF)
        if op_def is None:
            return None
        arity = attrs.get(ATTR_ARITY, (0, 0))[0]
        struct_ref: ConceptRef = (op_def[0], op_def[1])
        # ---- arity / 路径校验 + arg_subst 构建 ----
        if call_args is None:
            # Name 路径：nullary only（参数化算子无函数值·VM 无 CALL）
            if arity != 0:
                raise UnsupportedConstruct(
                    f"参数化算子 {name!r}(arity={arity}) 须 Call 引用·不可裸名 Name")
            arg_subst: dict[int, ConceptRef] = {}
        else:
            # Call 路径：arity 须匹配（fail-loud 拒静默丢参/越界）
            if len(call_args) != arity:
                raise UnsupportedConstruct(
                    f"算子 {name!r} arity={arity}·Call 给 {len(call_args)} 参（arity 不匹配）")
            arg_trees = [self._build_expr(a) for a in call_args]
            arg_subst = {make_variable(i): arg_trees[i] for i in range(arity)}
        # ---- 空壳校验（struct_ref 无 COMPOSES 子树→fail-loud·同 L1）----
        graph = ConceptGraph(self._b)
        children_of, operator_of, _operand_of, _immediate_of, _store_target_of = \
            graph.read_composes_tree(struct_ref)
        if not children_of and not operator_of:
            raise UnsupportedConstruct(
                f"算子 {name!r} 的 struct_ref 无 COMPOSES 子树（空壳不可 inline）")
        # ---- deep-copy + β-归约（param operand 叶→fresh 实参拷贝·internal alpha·自由变量 fail-loud）----
        return self._deep_copy_subtree(struct_ref,
                                       param_subst=arg_subst or None,
                                       fail_on_external=True)

    def _deep_copy_subtree(self, root: ConceptRef, *,
                           param_subst: dict[int, ConceptRef] | None = None,
                           fail_on_external: bool = False) -> ConceptRef:
        """deep-copy COMPOSES 子树·alpha-rename internal store_target sid·β-替换 param operand 叶。

        统一帮手·服务两面（L1.5·doc §三5）：
          · 算子 inline：param_subst={param_sid: arg_subtree}·fail_on_external=True
            （operand sid 非 internal 非 param = 自由变量→fail-loud·守闭项）。
          · 实参子树拷贝：param_subst=None·fail_on_external=False
            （operand sid 非 internal = external caller sid·保留·arg 是 caller 表达式引用 caller scope）。

        alpha-renaming：internal store_target sid（sorted 锁确定性·set 迭代序不保证跨运行一致）一律
          _alloc_internal() fresh 重分配·建 sid_remap 一致替换（防引用段内 capture·同 L1）。
        β-替换：operand 叶 sid ∈ param_subst → 嫁接 **fresh** _deep_copy_subtree(arg·无 param·external 保留)
          拷贝（每用 fresh·arg 拷贝 alpha 它自己的 internal STORE sid→无别名·对抗 Finding4 关键：caller 把
          含 STORE 的 Sigma 当实参传给多用算子·共享则 STORE 跨用别名→静默错值）。
        复制：只挑 _STRUCTURAL_KINDS（防御显式允许表·name 节点的 ATTR_OPERATOR_DEF/ATTR_ARITY 不在子树内）。
        边：children_of 已按 (order_index, NodeRef) 排·enumerate 复现槽位序（确定性 bit-identical）。
        """
        graph = ConceptGraph(self._b)
        children_of, _operator_of, operand_of, _immediate_of, store_target_of = \
            graph.read_composes_tree(root)
        # ---- alpha-renaming：internal store_target sid（sorted 锁确定性）----
        internal_sids = sorted(set(store_target_of.values()))
        sid_remap: dict[int, int] = {orig: self._alloc_internal() for orig in internal_sids}
        param_sids = set(param_subst or {})
        # ---- 自由变量校验（仅算子 inline·fail_on_external·实参拷贝 external 保留）----
        if fail_on_external:
            for orig_sid in operand_of.values():
                if orig_sid not in internal_sids and orig_sid not in param_sids:
                    raise UnsupportedConstruct(
                        f"inline 算子含自由变量（sid={orig_sid}）·须为 internal store_target 或 param")
        # ---- BFS copy_order（确定性访问序·visited 防重复）----
        ref_remap: dict[ConceptRef, ConceptRef] = {}
        copy_order: list[ConceptRef] = []
        pending: list[ConceptRef] = [root]
        seen: set[ConceptRef] = set()
        while pending:
            orig = pending.pop(0)
            if orig in seen:
                continue
            seen.add(orig)
            copy_order.append(orig)
            for child in children_of.get(orig, []):
                if child not in seen:
                    pending.append(child)
        # ---- 逐节点复制（param operand 叶→β-替换嫁接 fresh 实参拷贝·否则 fresh 节点记结构属性）----
        for orig in copy_order:
            attrs = read_composes_attrs(self._b, orig)
            operand_sid = attrs.get(ATTR_OPERAND, (None,))[0]
            if param_subst and operand_sid in param_sids:
                # β-替换：param operand 叶 → fresh 实参子树拷贝（每用 fresh·无 param·external 保留）
                ref_remap[orig] = self._deep_copy_subtree(
                    param_subst[operand_sid], param_subst=None, fail_on_external=False)
                continue   # 嫁接的实参子树根·不记本叶属性
            fresh = self._new_node("INL")
            ref_remap[orig] = fresh
            for kind, (ia, ib) in attrs.items():
                if kind not in _STRUCTURAL_KINDS:
                    continue   # 防御：只复制结构 kind（name 节点 ATTR_* 不在子树内）
                if kind == ATTR_OPERAND:
                    ia = sid_remap.get(ia, ia)   # internal→fresh·external（fail_on_external=False）→保留
                elif kind == ATTR_STORE_TARGET:
                    ia = sid_remap[ia]
                record_composes_attr(self._b, ref=fresh, kind=kind, int_a=ia, int_b=ib)
        # ---- 重挂边（children_of 已按 order_index 排·enumerate 复现槽位序·确定性 bit-identical）----
        for orig in copy_order:
            fresh = ref_remap[orig]
            for slot, orig_child in enumerate(children_of.get(orig, [])):
                self._edge(fresh, ref_remap[orig_child], slot)
        return ref_remap[root]

    def _alloc_internal(self) -> int:
        """分配 internal var sid（__acc/__idx·per-construct·确定性 counter·禁 set→list）。"""
        sid = make_variable(self._next_var)
        self._next_var += 1
        return sid

    # ---- 节点/边原语 ----

    def _new_node(self, node_type: str) -> ConceptRef:
        """建 AST 节点 ConceptRef（surface 含 args_sig+root_lid+前序序号+类型+py_major·per-space dedup）。

        root_lid（struct_ref local_id）隔离不同段的 AST 节点（同 code_observe·doc §三必改#2）。
        """
        self._seq += 1
        surface = f"__arith_{self._args_sig}_{self._root[1]}_{self._seq}_{node_type}_{self._py_major}"
        return self._ci.ensure(surface, space_id=self._space_id,
                               tier=TIER_PRIMARY, node_type=NODE_OPERATOR)

    def _edge(self, parent: ConceptRef, child: ConceptRef, order_index: int) -> None:
        """落 EDGE_COMPOSES 边（父→子·**order_index 强制显式**·read_composes_tree 按 (order_index,NodeRef) 排）。

        doc §三必改#3：漏传 order_index→CTRL_WHILE cond/body 槽位错乱→编译出错乱字节码。
        """
        assert_int(order_index, _where="_ArithBuilder._edge.order_index")
        self._es.add(space_id_from=parent[0], local_id_from=parent[1],
                     space_id_to=child[0], local_id_to=child[1],
                     edge_type=EDGE_COMPOSES, strength=1, source=self._source,
                     epistemic_origin=EPI_STRUCTURED, order_index=order_index)

    def _imm_leaf(self, num: int, den: int = 1) -> ConceptRef:
        """常量叶（immediate(num,den)·den>0 fail-loud·算术域字面量 den=1·非 1 den 由 OPCODE_DIV 运行时产生）。"""
        if den == 0:
            raise UnsupportedConstruct(f"immediate den=0（纯整数铁律）: {num}/{den}")
        leaf = self._new_node("IMM")
        record_composes_attr(self._b, ref=leaf, kind=ATTR_IMMEDIATE, int_a=num, int_b=den)
        return leaf

    def _var_leaf(self, sid: int) -> ConceptRef:
        """变量叶（operand·LOAD sid·经 scope 栈解析的 sid）。"""
        leaf = self._new_node("VAR")
        record_composes_attr(self._b, ref=leaf, kind=ATTR_OPERAND, int_a=sid)
        return leaf

    def _new_store(self, target_sid: int, value_child: ConceptRef) -> ConceptRef:
        """STORE 节点（emit 值源子 + STORE 目标变量·控制流体回写 env·doc §五）。"""
        store = self._new_node("STORE")
        record_composes_attr(self._b, ref=store, kind=ATTR_STORE_TARGET, int_a=target_sid)
        self._edge(store, value_child, 0)
        return store

    def _binop_node(self, opcode: int, left: ast.expr, right: ast.expr) -> ConceptRef:
        """二元算子节点（后序 emit 两子 + opcode·栈机消费）。"""
        bnode = self._new_node("BINOP")
        record_composes_attr(self._b, ref=bnode, kind=ATTR_OPERATOR, int_a=opcode)
        self._edge(bnode, self._build_expr(left), 0)
        self._edge(bnode, self._build_expr(right), 1)
        return bnode

    # ---- 表达式 ----

    def _build_expr(self, node: ast.expr) -> ConceptRef:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):           # bool 先于 int（bool 是 int 子类·显式规范化）
                return self._imm_leaf(1 if v else 0)
            if isinstance(v, int):
                return self._imm_leaf(v)
            raise UnsupportedConstruct(
                f"Constant 类型不支持（纯整数铁律·float/str/None 拒绝）: {type(v).__name__}")
        if isinstance(node, ast.Name):
            return self._resolve_name_or_inline(node.id)
        if isinstance(node, ast.UnaryOp):
            # 仅 USub/UAdd on int 常量 → immediate(±num,1) 特例（负数字面量是 UnaryOp 非 Constant）
            if (isinstance(node.op, (ast.USub, ast.UAdd))
                    and isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, int)
                    and not isinstance(node.operand.value, bool)):
                num = node.operand.value
                if isinstance(node.op, ast.USub):
                    num = -num
                return self._imm_leaf(num)
            raise UnsupportedConstruct("UnaryOp 仅支持 USub/UAdd on int 常量（Not 等不支持）")
        if isinstance(node, ast.BinOp):
            return self._build_binop(node)
        if isinstance(node, ast.Compare):
            raise UnsupportedConstruct("Compare 不支持（DSL 用户表达式无比较·cond 由 builder 内部生成为 LT）")
        if isinstance(node, ast.Call):
            return self._build_call(node)
        raise UnsupportedConstruct(f"表达式不支持: {type(node).__name__}")

    def _build_binop(self, node: ast.BinOp) -> ConceptRef:
        op = node.op
        if isinstance(op, ast.Add):
            return self._binop_node(OPCODE_ADD, node.left, node.right)
        if isinstance(op, ast.Sub):
            return self._binop_node(OPCODE_SUB, node.left, node.right)
        if isinstance(op, ast.Mult):
            return self._binop_node(OPCODE_MUL, node.left, node.right)
        if isinstance(op, ast.Div):
            # 精确有理除（doc §六·与 code_observe 拒 `/` 不矛盾·DSL 符号记号→VM OPCODE_DIV 非求值）
            return self._binop_node(OPCODE_DIV, node.left, node.right)
        if isinstance(op, ast.Pow):
            return self._build_pow(node.left, node.right)
        raise UnsupportedConstruct(
            f"BinOp 运算符不支持（// % 无 opcode·Pow 负/变量指数 reject）: {type(op).__name__}")

    def _build_pow(self, base_node: ast.expr, exp_node: ast.expr) -> ConceptRef:
        """a**k·仅字面非负 int 指数 → 重复 MUL（左结合）·k=0→IMM(1)（VM 约定 a^0=1 含 0^0=1·doc §七）。"""
        if not (isinstance(exp_node, ast.Constant)
                and isinstance(exp_node.value, int)
                and not isinstance(exp_node.value, bool)
                and exp_node.value >= 0):
            raise UnsupportedConstruct(
                "Pow 指数须字面非负 int（负指数=UnaryOp/变量指数/Pow(b,a) 全 reject·须用 Recur 显式表达）")
        k = exp_node.value
        if k == 0:
            return self._imm_leaf(1)   # a^0 = 1（VM 约定·含 0^0=1）
        # a**k = ((a*a)*...*a)·左结合·每次 fresh base 子树（避共享子边 dedup·doc §三必改）
        result = self._build_expr(base_node)
        for _ in range(k - 1):
            mul_node = self._new_node("BINOP")
            record_composes_attr(self._b, ref=mul_node, kind=ATTR_OPERATOR, int_a=OPCODE_MUL)
            self._edge(mul_node, result, 0)
            self._edge(mul_node, self._build_expr(base_node), 1)   # fresh base（同 sid 不同 ConceptRef）
            result = mul_node
        return result

    # ---- 迭代构造（Sigma/Prod/Recur → CTRL_WHILE + 累加器·doc §五） ----

    def _build_call(self, node: ast.Call) -> ConceptRef:
        if not isinstance(node.func, ast.Name):
            raise UnsupportedConstruct("Call func 须 Name（Sigma/Prod/Recur/Lim/已注册算子）")
        name = node.func.id
        if name == _LIM:
            # L4 诚实定位：Lim DSL = 0/0 族极限【值】计算 defer（求导+极限原语库=符号 CAS·C7 超越值墙）。
            # 表达层平凡：引用已学 lim/求导【关系】结论走算子 inline（L1/L1.5·无特设 lim 机制·C7 拦值不拦关系）。
            # to_rational_interval 区间套仅支撑代数极限包夹（√2 逼近）·非 0/0 洛必达判定→0/0 族仍真缺口/defer。
            raise UnsupportedConstruct(
                "Lim: 0/0 族极限【值】计算 defer（收敛值需求导+极限原语库=符号 CAS·C7 超越值墙·"
                "纯整数 VM 不可达无穷迭代）·引用已学 lim/求导【关系】结论走算子 inline（L1/L1.5）"
                "·非此 DSL Lim 值计算")
        if name in (_SIGMA, _PROD):
            return self._build_sigma_prod(name, node.args)
        if name == _RECUR:
            return self._build_recur(node.args)
        if name == _POW:
            return self._build_pow_call(node.args)
        # L1.5：已注册算子 → inline + β-归约（param i↔make_variable(i)·实参子树每用 fresh 嫁接）
        inlined = self._try_inline_learned(name, call_args=node.args)
        if inlined is not None:
            return inlined
        raise UnsupportedConstruct(f"Call 不支持（仅 Sigma/Prod/Recur/Lim/已注册算子）: {name}")

    def _build_sigma_prod(self, name: str, args: list[ast.expr]) -> ConceptRef:
        """Sigma(lo,hi,body)/Prod(lo,hi,body) → 迭代块（acc=0/1·acc+=/*body·i=lo..hi·doc §五）。"""
        if len(args) != 3:
            raise UnsupportedConstruct(f"{name} 须 3 参 (lo, hi, body)·得 {len(args)}")
        lo_node, hi_node, body_node = args
        # lo/hi 在 push scope 前建（外层解析·i 若出现指外层索引·doc §五 scope 时序）
        lo_tree = self._build_expr(lo_node)
        hi_tree = self._build_expr(hi_node)
        acc_sid = self._alloc_internal()
        idx_sid = self._alloc_internal()
        init_num = 0 if name == _SIGMA else 1
        # body 在 push {i:idx} 后建（内层解析·i=本层索引）
        self._scope.append({_IDX_NAME: idx_sid})
        try:
            body_tree = self._build_expr(body_node)
        finally:
            self._scope.pop()
        update_opcode = OPCODE_ADD if name == _SIGMA else OPCODE_MUL
        return self._build_iterative_block(
            acc_sid, idx_sid,
            init_tree=self._imm_leaf(init_num), lo_tree=lo_tree, hi_tree=hi_tree,
            body_tree=body_tree, update_opcode=update_opcode)

    def _build_recur(self, args: list[ast.expr]) -> ConceptRef:
        """Recur(init,count,body) → 单状态递推块（acc=init·repeat count: acc=body(a,i)·i=1..count·doc §五）。"""
        if len(args) != 3:
            raise UnsupportedConstruct(f"Recur 须 3 参 (init, count, body)·得 {len(args)}")
        init_node, count_node, body_node = args
        # init/count 在 push scope 前建（外层解析）
        init_tree = self._build_expr(init_node)
        count_tree = self._build_expr(count_node)
        acc_sid = self._alloc_internal()
        idx_sid = self._alloc_internal()
        # body 在 push {a:acc, i:idx} 后建（内层解析·a=累加器 i=索引）
        self._scope.append({_ACC_NAME: acc_sid, _IDX_NAME: idx_sid})
        try:
            body_tree = self._build_expr(body_node)
        finally:
            self._scope.pop()
        # Recur acc_update = body 直接（body 已含 a·非 ADD/MUL 包裹）·update_opcode=None 标记
        return self._build_iterative_block(
            acc_sid, idx_sid,
            init_tree=init_tree, lo_tree=self._imm_leaf(1), hi_tree=count_tree,
            body_tree=body_tree, update_opcode=None)

    def _build_pow_call(self, args: list[ast.expr]) -> ConceptRef:
        """Pow(base, exp) → Pow 节点（OPCODE_POW_PATTERN·pattern-level 不展开 MUL·允许变量指数）。

        符号变换规则 LHS/RHS 模板用（d/dx Pow(base,n)·变量指数 n）·非 DSL 用户表达式
        （用户 a**k 仍走 _build_binop→_build_pow 字面非负指数展开 MUL·不变）。
        2 参：base 子树 + exp 子树（exp 可变量/字面/表达式·symbolic_transform._eval_rhs/_lower_pow 求值）。
        VM 不执行 Pow（OPCODE_POW_PATTERN 非 _OPCODE_TABLE·达 execute fail-loud）·apply 后 _lower_pow concrete 指数→MUL。
        """
        if len(args) != 2:
            raise UnsupportedConstruct(f"Pow 须 2 参 (base, exp)·得 {len(args)}")
        base = self._build_expr(args[0])
        exp = self._build_expr(args[1])
        pow_node = self._new_node("POW")
        record_composes_attr(self._b, ref=pow_node, kind=ATTR_OPERATOR, int_a=OPCODE_POW_PATTERN)
        self._edge(pow_node, base, 0)   # order_index 0 = base 槽位
        self._edge(pow_node, exp, 1)    # order_index 1 = exp 槽位
        return pow_node

    def _build_iterative_block(
        self, acc_sid: int, idx_sid: int, *,
        init_tree: ConceptRef, lo_tree: ConceptRef, hi_tree: ConceptRef,
        body_tree: ConceptRef, update_opcode: int | None,
    ) -> ConceptRef:
        """迭代块 SEQ NOP：[init_acc, init_idx, CTRL_WHILE(cond, body_block), load_acc] → acc 留栈顶。

        cond = CMP(LT, idx, hi+1)（idx<hi+1⟺idx<=hi·整数/有理均成立·避无 LE opcode·doc §五）。
        body_block = SEQ NOP [STORE(acc)←acc_update, STORE(idx)←idx+1]。
        update_opcode=None（Recur）→ acc=body 直接·否则 Sigma=ADD(acc,body)/Prod=MUL(acc,body)。
        栈效应（对抗审查 trace 验证）：c0/c1 STORE 净栈·cond 被 JZ pop·body_block 两 STORE 净栈·c3 LOAD acc 留栈顶。
        """
        block = self._new_node("ITER")        # SEQ NOP·root 单子
        record_composes_attr(self._b, ref=block, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
        # c0: STORE(__acc) ← init
        self._edge(block, self._new_store(acc_sid, init_tree), 0)
        # c1: STORE(__idx) ← lo
        self._edge(block, self._new_store(idx_sid, lo_tree), 1)
        # c2: CTRL_WHILE [cond, body_block]
        cond = self._build_cond_lt_idx_hi_plus1(idx_sid, hi_tree)
        body_block = self._build_loop_body(acc_sid, idx_sid, body_tree, update_opcode)
        wnode = self._new_node("WHILE")
        record_composes_attr(self._b, ref=wnode, kind=ATTR_CTRL_TAG, int_a=CTRL_WHILE)
        self._edge(wnode, cond, 0)            # order_index=0 = COND 槽位
        self._edge(wnode, body_block, 1)      # order_index=1 = BODY 槽位
        self._edge(block, wnode, 2)
        # c3: LOAD __acc（operand 叶·结果留栈顶）
        self._edge(block, self._var_leaf(acc_sid), 3)
        return block

    def _build_cond_lt_idx_hi_plus1(self, idx_sid: int, hi_tree: ConceptRef) -> ConceptRef:
        """cond = CMP(LT, LOAD idx, ADD(hi, IMM(1,1)))（idx < hi+1 ⟺ idx <= hi·doc §五）。"""
        idx_leaf = self._var_leaf(idx_sid)
        plus1 = self._new_node("BINOP")
        record_composes_attr(self._b, ref=plus1, kind=ATTR_OPERATOR, int_a=OPCODE_ADD)
        self._edge(plus1, hi_tree, 0)
        self._edge(plus1, self._imm_leaf(1), 1)
        cmp_node = self._new_node("CMP")
        record_composes_attr(self._b, ref=cmp_node, kind=ATTR_OPERATOR, int_a=OPCODE_LT)
        self._edge(cmp_node, idx_leaf, 0)
        self._edge(cmp_node, plus1, 1)
        return cmp_node

    def _build_loop_body(self, acc_sid: int, idx_sid: int, body_tree: ConceptRef,
                         update_opcode: int | None) -> ConceptRef:
        """body_block = SEQ NOP [STORE(acc)←acc_update, STORE(idx)←idx+1]（每迭代净栈·doc §五）。"""
        block = self._new_node("BODY")
        record_composes_attr(self._b, ref=block, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
        # acc_update：Recur(body 直接) / Sigma(ADD(acc,body)) / Prod(MUL(acc,body))
        if update_opcode is None:
            update_value = body_tree
        else:
            acc_leaf = self._var_leaf(acc_sid)
            bnode = self._new_node("BINOP")
            record_composes_attr(self._b, ref=bnode, kind=ATTR_OPERATOR, int_a=update_opcode)
            self._edge(bnode, acc_leaf, 0)
            self._edge(bnode, body_tree, 1)
            update_value = bnode
        self._edge(block, self._new_store(acc_sid, update_value), 0)
        # idx += 1: STORE(idx) ← ADD(idx, IMM1)
        idx_leaf = self._var_leaf(idx_sid)
        inc = self._new_node("BINOP")
        record_composes_attr(self._b, ref=inc, kind=ATTR_OPERATOR, int_a=OPCODE_ADD)
        self._edge(inc, idx_leaf, 0)
        self._edge(inc, self._imm_leaf(1), 1)
        self._edge(block, self._new_store(idx_sid, inc), 1)
        return block


def build_composes_from_arith(arith_source: str, *, concept_index, edge_store,
                              backend, space_id: int, source: int,
                              root_ref: ConceptRef) -> ConceptRef:
    """算术记号 DSL（lambda 表达式）→ COMPOSES 树（落 EDGE_COMPOSES 边 + composes_attr 属性）·返 root=struct_ref。

    arith_source : lambda DSL 字符串（Segment.arith_source·MODALITY_ARITH）·如 "lambda n: Sigma(1,n,i)"。
    source       : SOURCE_* 枚举（edge_store·算术域 SOURCE_MATH）。
    root_ref     : 该段 struct_ref（= lambda 根 = COMPOSES 根·解致命#3·dag_path.sink=struct_ref=root）。

    入口接 ast.Expr(Lambda)（非 FunctionDef·doc §三必改#2）。超出支持子集 fail-loud UnsupportedConstruct。
    """
    assert_int(space_id, source, _where="build_composes_from_arith")
    assert_no_float(space_id, source, _where="build_composes_from_arith")
    try:
        tree = ast.parse(arith_source)
    except SyntaxError as e:
        raise UnsupportedConstruct(f"算术记号语法错误: {e}") from e
    if (len(tree.body) != 1
            or not isinstance(tree.body[0], ast.Expr)
            or not isinstance(tree.body[0].value, ast.Lambda)):
        raise UnsupportedConstruct(
            "算术记号须是单个 lambda 表达式（如 lambda n: Sigma(1, n, i)）")
    lambdanode = tree.body[0].value
    builder = _ArithBuilder(concept_index=concept_index, edge_store=edge_store,
                            backend=backend, space_id=space_id,
                            source=source, root_ref=root_ref)
    return builder.build(lambdanode)
