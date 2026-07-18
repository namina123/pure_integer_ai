"""cognition.process.struct_bind — STRUCT_BIND 跨模态槽位级绑定建边（#478·§8.7-P2）。

EDGE_STRUCT_BIND=18（C9-bis C:18）= **通用跨模态槽位级绑定边**（决断 1·修订原"算术↔语言专用"窄措辞）。
  from = 模态A骨架 PARAM 槽 ref（ConceptRef·COMPOSES 叶·build:310/328/376 写 ATTR_OPERAND）
  to   = 模态B骨架 PARAM 槽 ref
  order_index = 槽序（slot_map 内序·C4 句间序同域 OFFSET 范式）
  strength = STRUCT_BIND_STRENGTH（=1·结构边·静态·不接 reward·同 SPATIAL_ADJ/T_STEP·
             effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·STRUCT_BIND 不内）

绑定来源（决断 2）：(a) 教师标注先行（本模块建边·SOURCE_TEACHER/EPI_STRUCTURED·minimal viable·墙内可做）·
  (b) 跨模态结构对齐 defer（撞指称锚墙·#479 子问题·两模态 shape 共用 OPCODE_* 空间 coverage_overlap 数值
  可算·但**对齐位无指称锚**=语义绑定墙·非"shape 符号表不共享"·决断 2 审 1 P0-2 纠偏）。

模态无关（决断 1·§8.5）：边 schema 19 列无模态列·模态组合由 caller 标注（首版 caller 侧·决断 8·
  M1 后读节点 abstract_mark MARK_MODALITY·MARK_MODALITY vestigial 当前生产 caller 未接线）。STRUCT_BIND
  是边非节点合并·同指身份靠 REFERS_TO 性质A（PURE_ALIAS·edge_store.py:53）·STRUCT_BIND 只表槽位结构绑定（铁律 6·审 2 P2-5）。

单向依赖（决断 3·§8.7-洗-证伪 :437,590）：本模块位于 cognition/process=5·向 storage=2 向下 OK·
  建边纯写 edge 表·**不调 execute_composes_value**（L7 training·cognition=5→training=7 向上·lint.py:151-176
  硬拦判死）。消费侧（generate 读 STRUCT_BIND 拿绑定槽位 ref）落 #730·"算术值消费"经 L8 episode task-driven
  预置 → L0 memory_space → generate 中转（决断 3 候选②·generate 不直读 L8 episode·破单向依赖）。

反 theater（决断 7）：本模块=真建者（来源 a loader）·真消费 reader 落 #730（generate 读 STRUCT_BIND 填语言槽）。
  **#478 落地瞬态（#730 未跟进）= 边 theater 形态 2**（建边 caller live 但零消费·task doc:80）·诚实标。
  对照 SPATIAL_ADJ I1（build_spatial_adj live + observe.py:244 接线 live·合法接口预留范式）·等价性须
  #478 实施 1（本模块）+ 实施 3（boot 接线）+ #730 reader 跟进后成立（决断 7 审 1 P1）。

诚实边界：stable≠correct（结构对齐非语义绑定·跨模态指称锚=接地墙 #479）/ 来源 b 跨模态结构对齐 defer /
  其他模态（非语言）消费者 defer M1 / #730 reader 真前置含 L8→L0 read path 数据通路设计（非纯读边）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.edge_types import EDGE_STRUCT_BIND
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.composes_attr import read_composes_attrs, ATTR_OPERAND, ATTR_OPERATOR
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.result.graph_view import ConceptGraph

# STRUCT_BIND 边 strength（结构边·静态·不接 reward·同 SPATIAL_ADJ_STRENGTH=1 / T_STEP·§7.4.2 结构真值）。
# → effective_weight.py:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·STRUCT_BIND 不内（铁律 7 守门·reward 永不调）。
STRUCT_BIND_STRENGTH = 1


def collect_skeleton_slot_refs(backend, graph: ConceptGraph,
                               skeleton_ref: ConceptRef) -> list[ConceptRef]:
    """骨架 COMPOSES 子树 PARAM 槽 ref 序（DFS 前序·纯读·镜像 _collect_slot_lcas:662 范式·#478）。

    PARAM 槽谓词 ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs（build:310/328/376 写
    ATTR_OPERAND=make_variable(slot)·镜像 test_stage12 _read_slot_roles / structure_discover._collect_slot_lcas）。
    children_of 已按 (order_index, NodeRef) 排（read_composes_tree:241）→ DFS 前序 == build
    _concept_slot_idx 首遇序（纯语言两序一致·:379）。返 PARAM 叶 ref 序（caller 按 slot_map 配对建边）。

    返 list[ConceptRef]（每 PARAM 槽一个 ref·空骨架/无 PARAM → 空表）。纯读零写·确定
    （DFS 前序 + visited 防重复入队·bit-identical）。#730 reader 复用此函数拿绑定槽位 ref。

    诚实边界（混合 skeleton·caller 须守单模态）：
      - 谓词 ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs 只匹配 **PARAM operand 叶**·
        immediate 叶（ATTR_IMMEDIATE·无 ATTR_OPERAND/ATTR_OPERATOR）**静默不入 refs**·operator 节点亦不入。
      - **故 refs 序号 ≠ skeleton 全槽序号**——slot_map 的 a_idx/b_idx 须按 **PARAM-only 序** 标（非 build 全槽序）。
      - 混合 skeleton（CONCEPT+operand/immediate 同 sample 树）caller 须守单模态（formal_train _run_arith /
        _run_lang 路由已守·structure_discover.py:681-685 _collect_slot_lcas 同边界）·否则 slot_map idx 错位绑定
        而不报错（E5 graceful 的暗面）·当前测试只用纯 PARAM 叶骨架（caller 守单模态）。
    """
    children_of = graph.read_composes_tree(skeleton_ref)[0]
    refs: list[ConceptRef] = []
    visited: set[ConceptRef] = set()

    def _dfs(node: ConceptRef) -> None:
        if node in visited:
            return
        visited.add(node)
        attrs = read_composes_attrs(backend, node)
        if ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs:
            # PARAM 槽叶（CONCEPT 或 operand·无子·下方 for-loop 自然不递归·镜像 _collect_slot_lcas 不 early-return）
            refs.append(node)
        for child in children_of.get(node, []):
            _dfs(child)

    _dfs(skeleton_ref)
    return refs


def build_struct_bind_edge(edge_store: EdgeStore,
                           slot_ref_a: ConceptRef, slot_ref_b: ConceptRef,
                           *, source: int, space_id: int,
                           order_index: int) -> int:
    """建单条 STRUCT_BIND 槽位绑定边（slot_a → slot_b·order_index=槽序）。

    自环跳（slot_a==slot_b·同 ref 无义·反同模态自绑）。返建边数（0/1）。
    strength=STRUCT_BIND_STRENGTH（结构边·静态·不接 reward）·epistemic_origin=EPI_STRUCTURED
    （教师标注=结构化源①·来源 a·§8.1c-bis 合规）·tier=TIER_PRIMARY。

    **space_id 入参守对称签名（镜像 build_is_a_edge）·实际 edge 端点 space 从 slot_ref 解包**——
    STRUCT_BIND 绑已有 PARAM 槽 ref（discover 产·同 ctx.space_id）·非 ensure 新概念点（与 is_a 异）。
    守 `slot_ref[0] == space_id` 防 slot ref 跨 space 错配（caller formal_train 传 ctx.space_id·discover 产物同空间）。
    """
    if slot_ref_a == slot_ref_b:
        return 0
    assert slot_ref_a[0] == space_id and slot_ref_b[0] == space_id, \
        f"STRUCT_BIND slot_ref 须同 space_id={space_id}·" \
        f"实得 slot_a space={slot_ref_a[0]} / slot_b space={slot_ref_b[0]}（跨空间错配防御）"
    edge_store.add(
        space_id_from=slot_ref_a[0], local_id_from=slot_ref_a[1],
        space_id_to=slot_ref_b[0], local_id_to=slot_ref_b[1],
        edge_type=EDGE_STRUCT_BIND, strength=STRUCT_BIND_STRENGTH,
        source=source, epistemic_origin=EPI_STRUCTURED,
        order_index=order_index, role=None,
        tier=TIER_PRIMARY,
    )
    return 1


def bootstrap_struct_bind_edges(edge_store: EdgeStore,
                                bind_pairs: list[tuple[ConceptRef, ConceptRef]],
                                *, space_id: int,
                                source: int = SOURCE_TEACHER) -> int:
    """STRUCT_BIND 批量 boot 建边（镜像 is_a.bootstrap_is_a_edges:90 范式·#478·§8.7-P2 决断 4）。

    bind_pairs : [(slot_ref_a, slot_ref_b), ...] **已解析**的槽位 ref 对（每对一条边·
      order_index = 表内序）。caller 经 collect_skeleton_slot_refs(graph, skel_a) + slot_map 解析得到
      （决断 2·来源 a 教师标注 slot_map 按位序·文本可解析·非按 role 名增耦合）。
      **slot ref 已存在**（discover 产）·本函数纯写边·**不调 concept_index.ensure**（与 is_a bootstrap 异——
      is_a 建 surface 概念点 ensure·STRUCT_BIND 绑已有骨架 PARAM 槽 ref·name→skeleton_ref 解析在 caller
      form_train boot 经 discovered_operators 索引·决断 4"name 映射机制待实施期 loader 设计定"落 caller）。

    **无文件零副作用硬守（bit-identical·P0·镜像 bootstrap_is_a_edges:119）**：bind_pairs 空 → 立即
      return 0·**绝不调 query_from / build**（无 ZERO_AI_LOCAL_DIR → loader resolve 返 [] → 空 pairs →
      退化 bit-identical·CI===生产 default 守）。

    **幂等 skip 按源细化**（镜像 bootstrap_is_a_edges:128-137 + 决断 4 审 2 P2-1）：query_from 查 slot_a
      已有同 (slot_b, EDGE_STRUCT_BIND, **source**) 边 → skip（不挡异源·同源同三元组才 skip·防同三元组异源
      错 merge + resume 跨 run / 重复 boot corrupt·EdgeStore.add 不去重）。

    **source 取值契约**（审 2 P1-1）：default SOURCE_TEACHER（来源 a 教师标注）·异源（如 SOURCE_CONCEPTNET
      外部本体）须 caller 显式传不同 source 常量·**禁止复用同 source 值表异语义**（否则异源被当同源误 skip·
      静默 corrupt·违幂等按源细化意图）。source ∈ {SOURCE_TEACHER, SOURCE_CONCEPTNET, ...}（edge_store.py:36-44）。

    返建边数。
    铁律：纯整数（ConceptRef + EDGE_STRUCT_BIND 整边·零浮点）/ 确定性（bind_pairs 序 + query_from 序确定·
      bit-identical）/ 单向依赖（cognition/process→storage 向下·不调 L7 execute）/ §8.5（边 schema 不碰·
      19 列零改）/ 不写死（slot refs 来自外部 discover 产物·本函数只机制非语义）。
    诚实边界：STRUCT_BIND 不接 reward（effective_weight:82 assert 只认 PRECEDES/CAUSES/REFERS_TO·
      STRUCT_BIND 不内·base_strength reward-immutable）/ stable≠correct（结构对齐非语义绑定·#479 墙）/
      边 theater 形态 2（#730 reader 未跟进·本模块建边 live 但零消费·反 theater 锚守 #730 必跟进）。
    """
    if not bind_pairs:
        return 0   # P0·无文件零副作用硬守（不调 query_from/build·CI/生产 default bit-identical）
    assert_int(space_id, source, _where="bootstrap_struct_bind_edges.args")
    n = 0
    for ord_idx, (slot_a, slot_b) in enumerate(bind_pairs):
        # 幂等 skip（按源细化·镜像 bootstrap_is_a_edges:128-137）：query_from 查 slot_a 已有同 (slot_b,source) STRUCT_BIND 边。
        existing = edge_store.query_from(slot_a[0], slot_a[1], edge_type=EDGE_STRUCT_BIND)
        already = any(
            row.get("space_id_to") == slot_b[0]
            and row.get("local_id_to") == slot_b[1]
            and row.get("source") == source
            for row in existing
        )
        if already:
            continue   # 同源同三元组已建→skip（幂等·resume 跨 run / 重复 boot 不 corrupt）
        n += build_struct_bind_edge(edge_store, slot_a, slot_b,
                                    source=source, space_id=space_id,
                                    order_index=ord_idx)
    return n
