"""storage.edge_types — 边类型枚举（纯整数·L1 硬编码对齐 encoding）。

边类型是 domain 常量·storage 拥有（与 edge_store 的 source/epistemic_origin/subtype 枚举同层）。
位于 storage 层（低于 vm/algorithm/cognition）·各上层均可引用·守单向依赖
（algorithm/cognition 不反向定义·cognition/shared/edge_types.py Stage 3 再导出本源）。

编码确定性：整数枚举·值由设计 C9-bis 权威边类型枚举表固定（doc §十五 line1417·完备性单一事实源·
2026-07-01 落盘后取代 Stage1 旧顺序赋值·正式实现期对齐·无持久化数据迁移负担）。
register_edge_type 守完备性检查 #1（边类型须在 C9-bis 表 A-E 登记·否则拒）。

EDGE_CLOSURE = 127 是**派生类型**（closure 派生不存储·§十五 line159）·
不进 edge 宽表 source 列·仅 algorithm/closure 返回值用·设计表外 sentinel（不与设计表 1-64 ID 冲突）。
"""
from __future__ import annotations

# ---- 边类型枚举（§7.4 + §十一 + §十五 + 三卷伪代码） ----
# 整数值 = 设计 C9-bis 权威边类型枚举表（doc/重来·主线重审与重画.md §十五 line1417·完备性单一事实源）。
# Stage1 旧顺序赋值（1..12）已纠正对齐 C9-bis·守"code follows design"+ 单一事实源。
# 留空值（2/4/5/8/11/12/13/15-19）= 设计表 E 砍 / D 降字段 / C defer 的类型·register_edge_type 登记但不激活。

# A. 聚合排序头（强度边·参与 A1 多头 typed 聚合·§8.6/§7.2）
EDGE_COMPOSES = 1          # 组合（C 域专用·vm/graph_compile 沿此 emit·§十四C6）—— C9-bis A/C:1
EDGE_IS_A = 10             # 抽象子集（proper subset·typed 边·支撑头聚合+tier·M9 不接反传）—— C9-bis A:10
EDGE_PROPERTY = 9          # 属性（出边·output 填槽·META_MAP·M9 不接反传）—— C9-bis A:9
EDGE_CAUSES = 14           # 因果（独立 typed 边·§8.1c 硬边界·reward 反传唯一头）—— C9-bis A:14
EDGE_COOCCURS = 6          # 共现（SHADOW·lang/域分桶·仅桶内·C1 防跨语言污染）—— C9-bis A:6

# B. 结构边（各有专门消费·不进 A1 强度归一·§8.6 A12）
EDGE_REFERS_TO = 3         # 同指（性质A稳定/性质B occurrence·subtype 分流·闭包纯净性）—— C9-bis B:3
EDGE_CONDITION = 7         # 条件（if P→Q·§8.1c 条件包含≠因果 硬边界·C9-bis B:7 登记但不激活·
                            #   写侧 build_condition_edge 2026-07-09 删 YAGNI·总收口 §五1.2·
                            #   无 parser 设 has_condition+零读侧消费者·补活须独立设计 session）
EDGE_T_STEP = 20           # 时序步进（A 类型·order_index 隐含步序·不建时钟节点）—— C9-bis B:20
EDGE_PRECEDES = 21         # 句间/步骤序（A 类型·DAG·Kahn 分层载体·strength 恒=1）—— C9-bis B:21（NEW）
EDGE_SIMILAR = 24          # 相似关系 X~Y（"X 像 Y"·离散符号关系边·slot-filler 候选扩展·D2 合规非向量·
                            #   strength 恒=1·不接 reward·STEP5 PR4·dispatch_slot 读扩展 slot 候选）—— C9-bis B:24（NEW）
EDGE_MEREOLOGY = 25        # 部分-整体 X 是 Y 的一部分（part→whole 有向 typed 边·mereology·异 IS_A child→parent 子集·
                            #   解 cue_words REL_MEREOLOGY 误路由入 IS_A_CUE·独立 edge 守语义正交·strength 静态 base·
                            #   不接 reward 反传 / 不进 PR 邻接 / dag_path 不遍历 / closure part-of 预序闭包（Phase B §十四-bis·政策反转·build_mereology_ancestor_map_external）·
                            #   boot loader mereology_facts_{lang}.txt → bootstrap_mereology_edges 种边·客观序 T-L1d）—— C9-bis B:25（NEW·设计 doc §十五 单一事实源登记）
EDGE_ANTONYM = 26          # 反义 X↔Y 对称（大↔小·冷↔热·concept↔concept 1 阶·近 EDGE_SIMILAR 对称形·**异 SIMILAR 语义**：
                            #   反义=对立非相似·**非 verify_inverse**（代数逆=transform↔transform T-L4·语言反义 concept↔concept·#479 外部 seed 非 verify）·
                            #   对称 by-read（单边 a→b 存储·reader 双向查镜像 similar_candidates·异 alias 双边存储）·strength 恒=1 结构真值·
                            #   不接 reward / 不进 PR / dag_path 不遍历 / closure 不闭包·镜像 EDGE_SIMILAR 结构边纪律·
                            #   boot loader antonym_facts_{lang}.txt → bootstrap_antonym_edges 种边·客观序 T-L1e）—— C9-bis B:26（NEW·设计 doc §十五 单一事实源登记）
EDGE_REALIZES = 27         # 对应图对象 skeleton → relation-type 节点（"此结构实现此逻辑关系"·layer3 对应·Phase D §十六-bis D.1）。
                            #   target = reified __REL_*__ NODE_CONCEPT（REL_SUBSET=1 for IS_A·relation_primitives.ensure_relation_primitives）。
                            #   **来源 sound**：option (b) skeleton 通用文本独立发现 + oracle-pair-match 标（forming-sample token-pair 命中外源
                            #   EDGE_IS_A·EPI_STRUCTURED·**oracle 定 IS_A 非读 Cue**·禁 pairs→文本渲染 Cue 泄漏·Phase A审2 REJECT 命门解）。
                            #   labeled bed（同 boot IS_A）·学习 claim 严禁前置·验 floor Phase F·consumer Phase E。
                            #   strength 恒=1·不接 reward / 不进 PR / dag_path 不遍历（结构对应边非强度头）·镜像 EDGE_INSTANTIATES 纪律。
                            #   gate REALIZES_MODE default OFF·self-gate at writer + 幂等（mirror build_instantiates_edge）·bit-identical —— C9-bis B:27（NEW·§十六-bis）

# C. 域专用结构边（按域激活·首版语言模态不激活）
EDGE_SPATIAL_ADJ = 23      # 空间邻接（无向图拓扑·M1·strength 恒=1·非学习对象）—— C9-bis C:23（NEW）
EDGE_QUARANTINE_LINK = 22  # 检疫链接（伴随 sign=0 隔离·跨 space 留档·C9-ter）—— C9-bis C:22（NEW）

# 派生类型（不存储·algorithm/closure 返回值 tag·设计表外 sentinel·不与设计表 1-64 ID 冲突）
EDGE_CLOSURE = 127

# 反向名（调试/审计用·确定性）
EDGE_TYPE_NAME: dict[int, str] = {
    EDGE_PRECEDES: "PRECEDES",
    EDGE_SIMILAR: "SIMILAR",
    EDGE_MEREOLOGY: "MEREOLOGY",
    EDGE_ANTONYM: "ANTONYM",
    EDGE_REALIZES: "REALIZES",
    EDGE_CAUSES: "CAUSES",
    EDGE_IS_A: "IS_A",
    EDGE_PROPERTY: "PROPERTY",
    EDGE_CONDITION: "CONDITION",
    EDGE_REFERS_TO: "REFERS_TO",
    EDGE_COOCCURS: "COOCCURS",
    EDGE_COMPOSES: "COMPOSES",
    EDGE_T_STEP: "T_STEP",
    EDGE_SPATIAL_ADJ: "SPATIAL_ADJ",
    EDGE_QUARANTINE_LINK: "QUARANTINE_LINK",
    EDGE_CLOSURE: "CLOSURE",
}


# ---- 头聚合分组（§7.2 多头 typed 聚合·按头分发用） ----
# HEAD_*_LIKE frozenset 已删（2026-07-07·A2 设计 session 判死刑·零真消费 caller·Phase 0 选项 a）。
# 活路径按头分发用字面量（dag_path:179 {PRECEDES,CAUSES} / a2_stepper:129/135 if-branch /
#   attractor:50 (T_STEP,PRECEDES) / effective_weight:82 / a3_pr_wrapper:110）·不引 frozenset。
# drift 收口（dag_path/attractor 字面值统一）仍 defer M1（task #697·随 T_STEP 闭包归属一并决断）。


# ---- C9-bis 登记但 defer 的边类型（首版语言模态不激活·登记入完备性单一事实源） ----
# 这些类型在设计 C9-bis 表登记为合法 edge_type·但首版不激活（域/阶段门控）·
# register_edge_type 接纳（在 REGISTERED_EDGE_TYPES 内）·live 代码不产不消费。
EDGE_CALLS = 17            # 代码域函数间调用 —— C9-bis C:17（defer·代码域）
EDGE_INSTANTIATES = 15     # VF 结构抽象 —— C9-bis C:15（defer·VF/多模态阶段）
EDGE_TOPO_GENERALIZES = 16  # 拓扑层级 —— C9-bis C:16（defer·VF 阶段）
EDGE_STRUCT_BIND = 18      # 跨结构绑定 —— C9-bis C:18（defer·VF 阶段）
EDGE_IMPLEMENTS_BY = 64    # VM 跨空间实现映射 —— C9-bis C:64（defer·算术域随 C6）
EDGE_RELATION_SIGNAL = 11  # 词→关系边 symbol —— C9-bis D:11（defer·断奶后 reward 涌现）
EDGE_FUNCTION_CLASS = 12   # 词→功能词/内容词 symbol —— C9-bis D:12（defer）
EDGE_ROLE_STAT = 13        # 概念→主/宾槽偏好 symbol —— C9-bis D:13（defer）

# ---- C9-bis 完备性单一事实源：合法 edge_type 集合（A-D + QUARANTINE_LINK） ----
# 设计 E 行（CONTAINS=2/ROLE=4/ORDER_INDEX=5/REALIZES=8/COLLOCATION=19）砍/降字段·
# 非 edge_type·不在集合内→register_edge_type 拒（防误用废类型建边）。
# EDGE_CLOSURE=127 派生-only·不进此集合（从不经 EdgeStore.add 落表）。
REGISTERED_EDGE_TYPES = frozenset({
    # A 聚合排序头
    EDGE_COMPOSES, EDGE_IS_A, EDGE_PROPERTY, EDGE_CAUSES, EDGE_COOCCURS,
    # B 结构边
    EDGE_REFERS_TO, EDGE_CONDITION, EDGE_T_STEP, EDGE_PRECEDES, EDGE_SIMILAR, EDGE_MEREOLOGY, EDGE_ANTONYM, EDGE_REALIZES,
    # C 域专用（live + deferred 登记）
    EDGE_SPATIAL_ADJ, EDGE_QUARANTINE_LINK,
    EDGE_CALLS, EDGE_INSTANTIATES, EDGE_TOPO_GENERALIZES, EDGE_STRUCT_BIND, EDGE_IMPLEMENTS_BY,
    # D 学习型信号边（deferred 登记）
    EDGE_RELATION_SIGNAL, EDGE_FUNCTION_CLASS, EDGE_ROLE_STAT,
})


def is_registered_edge_type(edge_type: int) -> bool:
    """C9-bis 完备性检查 #1：edge_type 是否在权威表 A-D 登记（合法 edge_type）。"""
    return edge_type in REGISTERED_EDGE_TYPES
