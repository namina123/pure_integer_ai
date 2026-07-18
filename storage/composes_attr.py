"""storage.composes_attr — COMPOSES 程序属性持久化（A3·致命#1·依赖 crosscut+storage）。

compile_graph 需 5 dict（children_of/operator_of/operand_of/immediate_of/store_target_of）。
children_of 由 EDGE_COMPOSES 边重建（graph_view.out_edges）·其余 4 dict 无既有持久化路径
（concept_node 无 opcode/operand 列·def_array 单行装不下 immediate (num,den) 二元组）。

本表补此缺口：每个 COMPOSES 节点的算子/操作数/立即数/STORE 目标/控制流标签属性落盘·
vm_proof_fn 在 judge 时（另一 episode）经 ConceptGraph.read_composes_tree 读回重建 5 dict。

表 composes_attr（扩展表·core=False·DISC_APPEND_ONLY）：
  (space_id, local_id, kind, int_a, int_b)
  - kind ∈ ATTR_*（节点属性种类）
  - int_a/int_b 按 kind 释义（见下）

kind 释义：
  ATTR_OPERATOR     算子节点 opcode      int_a=OPCODE_* symbol_id        int_b=0
  ATTR_CTRL_TAG     控制流编译指令标签    int_a=CTRL_IF/IFELSE/WHILE       int_b=0
  ATTR_OPERAND      变量叶 operand        int_a=make_variable(index) sid   int_b=0
  ATTR_IMMEDIATE    常量叶立即数          int_a=num                        int_b=den(>0)
  ATTR_STORE_TARGET STORE 节点目标变量    int_a=make_variable(index) sid   int_b=0
  ATTR_OPERATOR_DEF 算子名→struct_ref注册 int_a=struct_ref space_id        int_b=struct_ref local_id
  ATTR_ARITY        算子 arity（参数数）   int_a=arity                       int_b=0
  ATTR_ORIGIN       生成方式来源          int_a=ORIGIN_*                   int_b=0
  ATTR_SLOT_ROLE    抽象级 slot LCA 类    int_a=LCA space_id               int_b=LCA local_id（S3 第二刀 Interp2）
  ATTR_RELATION_PRIMITIVE 关系原语节点标记 int_a=REL_*                     int_b=0（刀3 件1 种概念·挂 REL_* NODE_CONCEPT）
  ATTR_PROPOSITION     命题节点标记          int_a=0                        int_b=0（G1 reification·挂 __prop_* NODE_CONCEPT·#774 命题承载 subject/attr_type/value 三元·G3b 读判矛盾）
  ATTR_SYMBOL_TYPE     通用符号类型标记      int_a=TYPE_*                   int_b=0（STEP3 命名层·挂 TYPE_* first-class NODE_CONCEPT·D6 符号空间 type_ref 先天分类·shadow 空挂载 defer）
  ATTR_OPERATOR_PRIMITIVE 算子/比较原语节点标记 int_a=OP_*                int_b=0（STEP5 PR2·挂 OP_* first-class NODE_CONCEPT·D6 算子原语先天·D:11 target·勿复用 ATTR_OPERATOR=1 结构 kind）

ATTR_ORIGIN int_a 值域（生成方式·doc/重来_结构发现设计补充.md §4.4 选B·零 core 迁移）：
  ORIGIN_BUILT       observer 硬编码建造（4 固定 builder：language/code/arith/latex）
  ORIGIN_DISCOVERED  结构发现抽骨架（discover_skeleton·§八序列1·本 kind 的首 caller）
  ORIGIN_INJECTED    教师注入（录放层·断奶前·结论即程序）

铁律：纯整数（kind/int_a/int_b 全 int·assert_int 守）/ 确定性（(ref,kind) 唯一·读回按 kind 重建）/
  append-only / 核心无墙钟 / 不写死（kind 元定义枚举·值由建造者按 AST 映射填非硬编码）。
诚实边界：本表持久化结构属性非语义（COMPOSES 节点 opcode/operand/immediate/store_target/ctrl_tag
  + ATTR_OPERATOR_DEF 算子名→struct_ref 注册 + ATTR_ARITY 算子 arity + ATTR_ORIGIN 生成方式标记
  + ATTR_SLOT_ROLE 抽象级 PARAM slot 的 IS_A LCA 类·均在 name/struct concept 节点上·name 节点永不在
  COMPOSES 子树内（复制循环遇不到 ATTR_ARITY/ATTR_OPERATOR_DEF/ATTR_ORIGIN）·接 reward=否·sn/tn 在 edge 表 inert）。
  ATTR_ORIGIN 是 struct_ref root 的标记属性·非结构 kind（read_composes_tree 只读 5 已知结构 kind 忽略它·
  _deep_copy_subtree 的 _STRUCTURAL_KINDS 不含它→inline 嫁接不传播 discovered 标记·消费非重生·§4.4/§八.7 反 theater）。
  ATTR_SLOT_ROLE 同 ATTR_ORIGIN 范式（非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播·
  S3 第二刀 Interp2·挂 skeleton CONCEPT slot fresh 节点·消费在 recognize _align_walk 抽象匹配·非 inline）。
  ATTR_RELATION_PRIMITIVE 同 ATTR_ORIGIN 范式（非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·
  inline 不传播·刀3 件1·挂 REL_* first-class NODE_CONCEPT 节点·消费在 lookup_word_concept 读 ATTR 标记·非 inline·
  §8.8 关系概念=被 typed edge D:11 引用的 first-class 节点·非层次链复活·非 META_* 复活）。
ATTR_PROPOSITION 同 ATTR_ORIGIN/ATTR_RELATION_PRIMITIVE 范式（非结构 kind·read_composes_tree 忽略·
  _STRUCTURAL_KINDS 不含·inline 不传播·G1 reification+#774·挂 __prop_* 命题 NODE_CONCEPT 节点·消费在
  graph_view.iter_proposition_nodes 读 ATTR 标记 + G3b counterfactual_value_check 扫命题节点 PROPERTY 出边·
  非层次链复活·非 META_* 复活·reification 给表达力非验证力·命题 truth=#479 墙·G3b 只判结构矛盾非语义对立）。
ATTR_SYMBOL_TYPE 同 ATTR_ORIGIN/ATTR_RELATION_PRIMITIVE/ATTR_PROPOSITION 范式（非结构 kind·read_composes_tree 忽略·
  _STRUCTURAL_KINDS 不含·inline 不传播·STEP3 命名层·挂 TYPE_* first-class NODE_CONCEPT 节点·D6 符号空间 type_ref
  先天分类挂载点·非 ATTR_NEGATION 专用·doc:193 ¬ 走命题 surface polarity 不建 ATTR_NEGATION·消费者 WHERE
  kind=ATTR_SYMBOL_TYPE + int_a=TYPE_* 查·功能等价 iter_proposition_nodes 范式·shadow 空挂载 defer·bit-identical）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

# ---- kind 枚举（COMPOSES 节点属性种类·元定义枚举） ----
ATTR_OPERATOR = 1      # 算子节点 opcode（int_a=OPCODE_* symbol_id）
ATTR_CTRL_TAG = 2      # 控制流编译指令标签（int_a=CTRL_IF/IFELSE/WHILE）
ATTR_OPERAND = 3       # 变量叶 operand（int_a=make_variable(index) symbol_id）
ATTR_IMMEDIATE = 4     # 常量叶立即数（int_a=num, int_b=den>0）
ATTR_STORE_TARGET = 5  # STORE 节点目标变量（int_a=make_variable(index) symbol_id）
ATTR_OPERATOR_DEF = 6  # 算子名→struct_ref 注册（int_a=struct_ref space_id·int_b=struct_ref local_id·L1 inline 查表）
ATTR_ARITY = 7         # 算子 arity（int_a=参数数·L1.5 β-归约 param i↔make_variable(i) 分类·挂 name 节点非 root）
ATTR_ORIGIN = 8        # 生成方式来源（int_a=ORIGIN_*·挂 struct_ref root·§4.4 选B 零 core 迁移·结构发现首 caller）
ATTR_SLOT_ROLE = 9     # 抽象级 PARAM slot 的 IS_A LCA 类（int_a=LCA space_id·int_b=LCA local_id·S3 第二刀 Interp2
                       # 抽象对撞·挂 skeleton CONCEPT slot fresh 节点·与 ATTR_OPERAND 同节点第二 attr·absence=无类约束）
ATTR_RELATION_PRIMITIVE = 10  # 关系原语节点标记（int_a=REL_*·挂 REL_* first-class NODE_CONCEPT 节点·刀3 件1 种概念）
                              # 同 ATTR_ORIGIN/ATTR_SLOT_ROLE 范式（非结构 kind·read_composes_tree 忽略·
                              # _STRUCTURAL_KINDS 不含·inline 不传播·§8.8 关系概念=被 typed edge D:11 引用的
                              # first-class 节点·非层次链复活·非 META_* 复活）
ATTR_PROPOSITION = 11         # 命题节点标记（int_a=int_b=0·仅标记·G1 reification·挂 __prop_* 命题 NODE_CONCEPT 节点·
                              # 同 ATTR_ORIGIN/ATTR_RELATION_PRIMITIVE 范式·非结构 kind·read_composes_tree 忽略·
                              # _STRUCTURAL_KINDS 不含·inline 不传播·#774 在命题节点建 PROPERTY 出边（value）·
                              # G3b 读命题节点 PROPERTY 出边判同(subject,attr_type)多值结构矛盾·命题节点身份=
                              # (subject,attr_type)·确定性 surface __prop_{subj}_{attr}·concept_index.ensure 去重·
                              # 不进 dag_path/structure_units·判断层载体非路径层载体·解 G3b 读 struct_ref theater）
ATTR_SYMBOL_TYPE = 17         # 通用符号类型标记（int_a=TYPE_*·int_b=0·STEP3 命名层·挂 TYPE_* first-class NODE_CONCEPT 节点·
                              # 同 ATTR_ORIGIN/ATTR_RELATION_PRIMITIVE/ATTR_PROPOSITION 范式·非结构 kind·
                              # read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播·D6 符号空间 type_ref
                              # 先天分类挂载点·非 ATTR_NEGATION 专用·doc:193 ¬ 走命题 surface polarity 不建 ATTR_NEGATION·
                              # 消费者 WHERE kind=ATTR_SYMBOL_TYPE + int_a=TYPE_* 查·功能等价 iter_proposition_nodes 范式·
                              # STEP3 shadow 空挂载（ensure_symbol_types ship 不调用·defer·bit-identical AST 级零变））
ATTR_OPERATOR_PRIMITIVE = 18  # 算子/比较原语节点标记（int_a=OP_*·int_b=0·STEP5 PR2·挂 OP_* first-class NODE_CONCEPT 节点·
                              # 同 ATTR_RELATION_PRIMITIVE=10 范式·非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS
                              # 不含·inline 不传播·D6 符号空间算子原语先天分类·**勿复用 ATTR_OPERATOR=1**（结构 kind·
                              # VM COMPOSES 树专用·复用污染 5-dict 重建）·消费者 lookup_word_operator WHERE kind=ATTR_OPERATOR_PRIMITIVE
                              # + int_a=OP_* 查·D:11 共享边类型与 REL_* 隔离（kind==0 skip·无交叉污染）·boot 种 D:11 边前先建 target）
ATTR_PROP_SUBJ = 19           # 命题节点 subject ref 结构存（int_a=subj_sid·int_b=subj_lid·STEP6 PR3·命题 identity 可读）·
                              # 同 ATTR_PROPOSITION=11 范式·非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·
                              # 解 ref→surface defer（node_store 无 surface 列·gates.py:444 标注）·G3b 模态对当跨节点按
                              # (subj,attr) 分组读此 + ATTR_PROP_ATTR·build_property_edges 建 ATTR_PROPOSITION 时同 record·幂等
ATTR_PROP_ATTR = 20           # 命题节点 attr_type ref 结构存（int_a=attr_sid·int_b=attr_lid·STEP6 PR3·同 ATTR_PROP_SUBJ 范式）·
                              # G3b 模态对当分组用
ATTR_PROP_POLMOD = 21         # 命题节点 polarity+modality 结构存（int_a=pol·int_b=mod·STEP6 PR3·P0.3 pol/mod 进 surface 亦结构存）·
                              # G3b 模态对当读 (pol,mod) 判模态方阵·非结构 kind·同 ATTR_PROP_SUBJ 范式
ATTR_MODAL_KIND = 22          # 模态种类原语节点标记（int_a=MODAL_KIND_*·int_b=0·审计根治·挂 MODAL_KIND_* first-class NODE_CONCEPT 节点·
                              # 同 ATTR_OPERATOR_PRIMITIVE=18 / ATTR_RELATION_PRIMITIVE=10 范式·非结构 kind·read_composes_tree 忽略·
                              # _STRUCTURAL_KINDS 不含·inline 不传播·D:11 共享边类型与 REL_*/OP_* 隔离（kind==0 skip·无交叉污染）·
                              # **D6 职责分离**：composes_attr ATTR_MODAL_KIND=22 是存储 readback 标记（lookup_word_modality 读·镜像
                              # lookup_word_operator）·abstract_mark MARK_MODAL_KIND=5 是 D6 语义归属声明（模态种类归抽象空间·set_mark
                              # 在 ensure 时调）·两者职责不同非重复·readback 走 composes_attr 镜像既有 4 个 D:11 范式工程一致·
                              # abstract_mark 守 D6 模态种类归抽象空间·非 TYPE_* 不违 STOP·boot 种 D:11 边前先建 target）
ATTR_OPERATION_INTENT = 23     # 动作意图原语节点标记（int_a=ACTION_INTENT_*·int_b=0·B-PR1·挂 INTENT_COMMAND_MOOD + ACTION_GENERATE/COMPUTE/ANALYZE/SOLVE
                              # first-class NODE_CONCEPT 节点·**镜像 ATTR_OPERATOR_PRIMITIVE=18 范式**（符号空间先天 closed-class·单挂·**不挂 abstract_mark**·doc §16.3）·
                              # 非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播·D:11 共享边与 REL_*/OP_*/MODAL_KIND 隔离（kind==0 skip）·
                              # **boot concept 旗标**（概念身份·ensure 时挂·非学习）·**非 B-PR2 经验回写**（回写走 experience_count 避幂等 noop·doc §16.2）·
                              # lookup_word_action 读 D:11 边 target ATTR·W7 命令判定（命中任一→type=COMMAND）+ B-PR1 类别判定（int_a=ACTION_* 1-4）·doc §16）
# kind=24 freed（ATTR_SKELETON_BINDING 已删·Phase A §十三-bis A.1·结构一等化·维度桥绑定迁 EDGE_INSTANTIATES 真边·
#   关联在图中·observe.build_instantiates_edge + graph_view.read_instantiates·替 effect-dormant 注解·enum 值不重号留空）。
ATTR_TRANSFORM_LHS = 25       # 符号变换规则 LHS 模式 struct_ref（int_a=lhs space_id·int_b=lhs local_id·符号数学扩展 Phase 2·doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis）·
                              # 镜像 ATTR_OPERATOR_DEF=6 "存 struct_ref 作 (int_a=sid,int_b=lid)" 范式·非结构 kind·
                              # read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播·挂规则名 concept 节点（register_transform_rule·同 register_arith_operator 范式）·
                              # LHS 模式 = COMPOSES 树·PARAM 槽（ATTR_OPERAND make_variable(slot)）= 通配符（_align_walk subtree_binding 绑子树 + value_binding 绑值）·
                              # apply_transform 读 ATTR_TRANSFORM_LHS+RHS→_align_walk 匹配 LHS vs input→_deep_copy_subtree β-替换 RHS→输出表达式
ATTR_TRANSFORM_RHS = 26       # 符号变换规则 RHS 模板 struct_ref（int_a=rhs space_id·int_b=rhs local_id·同 ATTR_TRANSFORM_LHS 范式·符号数学扩展 Phase 2）·
                              # RHS 模板 = COMPOSES 树·PARAM 槽与 LHS 同 sid 对齐（make_variable(slot)·教师陈述模板 lambda 同 arg 序保证对齐）·
                              # apply_transform 部分求值：subtree_binding→子树 β-替换（_deep_copy_subtree）+ value_binding→IMM 叶·（Phase 2b 待加 Pow lower + 值算术 VM 求值·d/dx n-1）
ATTR_RELATION_KIND = 27       # 运算间关联节点标记（int_a=RELATION_KIND_*·int_b=0·S8 符号间运算关联·doc/重来_S8符号间关联机制设计_2026-07-15）·
                              # 镜像 ATTR_PROPOSITION=11 范式（非结构 kind·marker·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播）·
                              # 挂关系名 concept 节点（__rel_inv_{ruleA}_{ruleB}·concept_index.ensure 去重·同命题节点范式）·
                              # RELATION_KIND_INVERSE=1（Phase 1 唯一·两规则互逆）·COMPOSITION（链式法则）defer Phase 2。
                              # 关系身份=(kind, ruleA, ruleB) 三元·挂在关系名 concept·不污染 edge 表语义（edge 概念间关系·这是规则间元关系·层次不同）。
                              # 消费者 symbolic_relation.verify_inverse_relation 读 KIND+RULE_A+RULE_B→构造验证 B∘A=identity @ 采样（反 theater·非教师声称）·
                              # 不进 dag_path/structure_units（同命题节点范式·判断层载体非路径层载体）·bit-identical（gate OFF 零注册）
ATTR_RELATION_RULE_A = 28     # 关联规则 A name-ref（int_a=ruleA name space_id·int_b=ruleA name local_id·镜像 ATTR_TRANSFORM_LHS=25 "存 name-ref 作 (int_a=sid,int_b=lid)" 范式·
                              # 非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播·S8 符号间运算关联 Phase 1）·
                              # ruleA/ruleB 是既有变换规则 name-ref（register_transform_rule 产物·ATTR_TRANSFORM_LHS/RHS）·关系名 concept 是新节点。
                              # verify_inverse_relation 读 RULE_A/B→load_transform_rule 得 (lhs,rhs)→apply_transform 串联验证
ATTR_RELATION_RULE_B = 29     # 关联规则 B name-ref（int_a=ruleB name space_id·int_b=ruleB name local_id·同 ATTR_RELATION_RULE_A 范式·S8 Phase 1）·
                              # A 与 B 互逆：B∘A=identity（apply A→apply B→还原原值 @ 采样）·非结构 kind·镜像 ATTR_TRANSFORM_RHS=26 范式
ATTR_PROP_INTENSITY = 30      # 命题节点 intensity（值强度·int_a=num·int_b=den>0·#1134 程度 augment·平行 ATTR_PROP_POLMOD=21 范式）·
                              # 程度副词（很/非常=2/1·较=3/2·稍=2/5·Rational·非 float）缩放命题值强度·来自外部 degree_cues_zh.txt（file-driven·非 §九 frozenset）·
                              # 非结构 kind（read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播·同 ATTR_PROP_POLMOD）·build_property_edges 建命题时 record·幂等。
ATTR_CUE_SIG = 31              # 闭类 cue 固定位标记（int_a=cue token space_id·int_b=cue token local_id·§十八 condition 6a cue-保留结构类·
                              # **镜像 ATTR_SLOT_ROLE=9 范式**（非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播）·
                              # 挂 skeleton CONCEPT slot fresh 节点（cue 拆簇的拆位·与 ATTR_OPERAND/ATTR_SLOT_ROLE 同节点第二 attr·absence=非 cue 位）·
                              # build CONCEPT_LEAF 分支仅 cue_sig 非 None 时写（gate CUE_CLUSTER_MODE + sustainable-split 拆簇）·幂等·
                              # 消费 _collect_cue_sig 读（load_discovered 重派生 _shape_name cue_sig 第四参·修 cue_sig 版 B6 Bug 1）·
                              # 反 theater：cue 身份**仅区分骨架发现期分桶**（route+auto_discover 名键）·关系 label 走外源 oracle（label_realizes·非 cue 路由·§十八 condition 6 复合键非单 primitive 单射）·
                              # 诚实 scope：6a cue-in-名+routing（§十八 partial）·非 cue 作结构节点 model-reversal（6b defer））。

# ATTR_RELATION_KIND int_a 值域（关系种类·Phase 1 唯一 INVERSE·COMPOSITION defer Phase 2）
RELATION_KIND_INVERSE = 1   # 逆关系（两变换规则互逆·d/dx↔∫·+/−·×/÷·B∘A=identity @ 采样可构造验证）

# ATTR_ORIGIN int_a 值域（生成方式·非数据源类型·与 edge.source 的 SOURCE_* 正交）
ORIGIN_BUILT = 1       # observer 硬编码建造（language/code/arith/latex 4 固定 builder）
ORIGIN_DISCOVERED = 2  # 结构发现抽骨架（discover_skeleton·§八序列1）
ORIGIN_INJECTED = 3    # 教师注入（录放层·断奶前·结论即程序）

_COMPOSES_ATTR_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("kind", TYPE_INT),
    ("int_a", TYPE_INT),
    ("int_b", TYPE_INT),
]
_COMPOSES_ATTR_INDEXES = [
    ("space_id", "local_id"),   # 节点端点主键
    ("space_id", "local_id", "kind"),   # (节点,kind) 唯一查
]
COMPOSES_ATTR_TABLE = "composes_attr"


def register_composes_attr(backend: StorageBackend) -> None:
    """注册 composes_attr 扩展表（core=False·启动/用前调·幂等）。"""
    register_extension_table(backend, COMPOSES_ATTR_TABLE,
                             _COMPOSES_ATTR_COLUMNS,
                             disc.DISC_APPEND_ONLY, _COMPOSES_ATTR_INDEXES)


def record_composes_attr(backend: StorageBackend, *, ref: tuple[int, int],
                         kind: int, int_a: int, int_b: int = 0) -> None:
    """落一个 COMPOSES 节点属性行（append-only）。

    ref=(space_id, local_id)·kind ∈ ATTR_*·int_a/int_b 按 kind 释义。
    幂等：同 (ref,kind) 已存在则跳过（建造者建树时同节点同属性不重复写）。
    """
    sid, lid = ref
    assert_int(sid, lid, kind, int_a, int_b, _where="record_composes_attr")
    # 幂等：同 (ref,kind) 已存在跳过（append-only 防重复行）
    existing = backend.select(COMPOSES_ATTR_TABLE, where={
        "space_id": sid, "local_id": lid, "kind": kind,
    }, limit=1)
    if existing:
        return
    backend.insert(COMPOSES_ATTR_TABLE, {
        "space_id": sid, "local_id": lid, "kind": kind,
        "int_a": int_a, "int_b": int_b,
    })


def read_composes_attrs(backend: StorageBackend,
                        ref: tuple[int, int]) -> dict[int, tuple[int, int]]:
    """读一个 COMPOSES 节点的全部属性 → {kind: (int_a, int_b)}。

    同节点多属性（如 STORE 节点既有 ATTR_STORE_TARGET 又有 ATTR_OPERATOR）·按 kind 聚合。
    无属性 → {}（caller 判）。
    """
    sid, lid = ref
    rows = backend.select(COMPOSES_ATTR_TABLE, where={
        "space_id": sid, "local_id": lid,
    })
    out: dict[int, tuple[int, int]] = {}
    for r in rows:
        out[r["kind"]] = (r["int_a"], r["int_b"])
    return out
