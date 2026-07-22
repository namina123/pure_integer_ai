"""cognition.process.structure_discover — 结构发现最小闭环（§八序列1·多视角三要素的拓扑/结构切片）。

对齐多样本 COMPOSES 程序 → 抽共性骨架（固定位 + PARAM 槽）→ 落 struct_ref + COMPOSES
（ATTR_ORIGIN=discovered·§4.4 选B）→ 复用既有 register_arith_operator + inline + β-归约 + vm_proof 消费。

权威设计 = doc/重来_结构发现设计补充.md §四/§八。code 跟它。结构发现 = 系统生成核心 + 多模态根基
（§〇/§五）·本模块是其**最小闭环**（序列1）：两样本 → 标准结构在主线跑通（复刻 legacy extract_pattern
先例·诚实 Rice 边界）。

**机制**（等长结构对齐·多样本并行 DFS 前序）：
  样本 = COMPOSES 程序树根（observer 已建·如 arith lambda 闭式）。并行 DFS 前序各样本同位节点：
    · 算子节点（ATTR_OPERATOR）：opcode 须全样一致 → 固定位（fresh 节点记同 opcode）。
    · 立即数叶（ATTR_IMMEDIATE）：全样同值 → 固定位（fresh IMM）；有异 → **PARAM 槽**（fresh operand
      叶·sid=make_variable(0..arity-1)·按 **DFS 前序左→右叶阅读序**·与 inline Call 路径 arg_subst
      按 AST 位置序一致——caller f(a,b,c) 的 a/b/c 填源码阅读序叶槽·契约对齐）。
  落 skeleton = NOP-root struct_ref + COMPOSES（镜像样本0 形状·相异位参数化）·root 标 ATTR_ORIGIN=discovered。
  返 SkeletonResult(skeleton_ref, arity) | None（无共性骨架 → None·caller 判）。

  **DFS 前序非 BFS**（对抗正确性审计纠错）：PARAM sid 须按 AST 阅读序（左→右叶）分配·与 inline
  arg_subst={make_variable(i):arg_i} 的 AST 位置序契约对齐。BFS 层序在非交换算子(SUB/DIV)或非均匀
  深度树发散（如 a-(b-c) 的 BFS 叶序 ≠ 阅读序）→ β-归约后实参进错槽→静默错值。DFS 前序天然阅读序。

**下游消费**（§8.7 反 theater·落 struct_ref 是写入·须被读消费非摆设）：
  · inline 消费：register_arith_operator(name, skeleton_ref, arity) → Call 引用 → β-归约 → vm_proof
    （发现的骨架被当算子复用进新程序·复现样本 + 泛化新值）。
  · coverage_overlap 识别消费：shape_signature 提取算子形状序列·新样本同形 → coverage=1000（认出结构）。
  · vm_proof 自验：骨架 PARAM 槽 = make_variable(0..arity-1)·input_args 绑参 → 直接可执行验对错。

**最小闭环范围（诚实）**：样本须为**算子 + 立即数/operand 叶树**（NOP root + BINOP + IMM/OPERAND 叶）·结构同构。
  **operand 叶（lambda 参数）= 序列2 已支持**（跨样本 sid 对应 + within-sample 变量同一性 + cross-sample 一致性门·
  见 `_SkeletonBuilder.build` OPERAND 分支）。**ctrl（CTRL_WHILE）/ store（STORE）= S3 已支持**（迭代骨架·doc/重来_S3S4迭代机制设计
  §三-bis·internal sid alpha-重命名到 make_variable(arity+k) 避 PARAM 区·镜像 _deep_copy_subtree·跨样本 sorted 锁对应）。异构结构
  （opcode/形状不同）→ None（变长锚对齐 = 序列2 另半 defer）。非立即数固定位（概念 ref·非程序）→ defer（序列3 coverage 识别）。

**集成诚实边界（§8.7）**：序列1 机制 done + 测试可达。**序列6-min（2026-07-03）部分 de-theater**——
  `auto_discover_operators`（本模块·见末段）是 discover_skeleton 的**真生产 caller**（formal_train 触发·
  de-theater 序列1"零 caller"：discover_skeleton 在生产期真跑·从真语料抽真骨架 + register）。注册名**可消费**
  （_try_inline_learned inline+β·测试证明机制活）。**但生产训练 loop 当前不引用 `__op_disc_*` 名**（对抗审计
  grep 核证零下游 caller）——故 doc §8.7 line306 "存进去没人读=theater" 的**生产期 READ 消费**尚未接通（须
  序列3 coverage_overlap 识别新输入 / 生成侧引用）。诚实定位：序列6-min = WRITE 进生产（discover+register
  真跑）+ READ 消费机制活（测试证）·**非"生产期已读消费"全闭环**（序列3 / 洗净循环 / observe 多程序去重 = 后续）。
  **序列3-min（2026-07-03）补生产期 READ 消费**——`recognize_operators`（本模块·见末段）让发现的 struct_ref+
  COMPOSES 骨架在生产期被**真读**（read_composes_tree + DFS 前序对齐 + PARAM 抽值 + 固定位值等）·formal_train
  per-shape 留 **held-out** 输入（同形≥3：发现首 2·识别余）→ 识别 held-out 新实例 = **真泛化非循环**（骨架从 {5,6}
  学·识别 {7,8} 新输入）。识别的 PARAM 绑定可被 vm_proof 验（骨架绑参执行 == 新输入执行·caller/test 验·本模块
  L5 不调 L7 vm_proof 守单向依赖）。**诚实边界**：序列3-min = 单 run 内 per-shape held-out（跨 run 识别新语料
  = follow-up·须 dump/load 发现算子）/ loop1 scope（operand/ctrl/store 输入→不识别·序列2+ defer）。**识别解了
  §8.7 对骨架的 theater**（骨架现被 recognize_operators 真读·vm_proof 验绑定复现 held-out 新输入值）·**但识别产物
  recognitions 当前为 terminal 可观测**（formal_train 写 result.recognitions 字段·生产 loop 不读·仅测试/反 theater
  锚点）——"识别驱动生成改进"须洗净循环回写生成/reward（§8.7 洗净循环消费闭环·follow-up·非本步）。
  **序列2（2026-07-03·operand 对应 + 变量同一性）**：loop1 只发现退化 mul（`5*5`/`6*6`→相异槽 arity2·认 `7*8`=mul(7,8)）·
  序列2 补 OPERAND 叶发现（`lambda x:x*x`→square arity1·两 OPERAND 同 sid→同槽复用=变量同一性）+ cross-sample 一致性门
  （sample0 槽细分于 sample_i·坍缩允许/拆分拒）+ recognize slot 感知绑定（同槽值须等·**`7*8` 拒**·变量同一性牙真）。
  让发现/识别产**有意义算子**（square 拒 `7*8`·mul 不拒·差异实证）。诚实边界：operand-input 识别 defer / 混类 shape 组
  发现脆弱（须同类 operand 样本在前）/ 变长锚对齐（异构）= 序列2 另半 defer。
  **operand-input 识别（2026-07-03·探针值执行比对·补序列2 operand READ 闭环）**：序列2 让发现产 operand 算子（square）·序列3-min
  让识别 immediate 实例（7*7）·但 operand 输入（held-out `λz:z*z`）识别缺（`_align_walk` 骨架 PARAM 遇 input OPERAND 叶→False）→ operand
  语料 held-out 识别率恒 0。本步补：`_align_walk` 加 operand 分支（slot→input operand slot·同 skeleton slot 须对齐同 input operand=变量同一性
  牙·**`λa,b:a*b` 拒**为 square·坍缩允 mul 识 square 为 mul(z,z)）·binding 拆 value/operand 两 dict（混合 input `λz:z+3` 兼容）·探针值
  执行验证（distinct 素数·复用 execute_composes_value·input 探针纯从 Recognition 字段反演无须 import 常量）。Recognition 加
  is_operand_input/operand_binding 字段（默认·向后兼容）。within-run operand 闭环收口（operand 语料泛化率从 0 升有意义）。
  诚实边界：探针 arity 上限 / 探针比对构造性（同 immediate 诚实定位·真牙在变量同一性判定）/ 变长锚对齐仍 defer。

复用（不新造·§4.5）：ConceptGraph.read_composes_tree / record_composes_attr,read_composes_attrs /
  make_variable,index_of（symbol_domain·sid↔slot 可逆）/ concept_index.ensure / edge_store.add /
  register_arith_operator（arith_observe）/ vm_proof_fn（training）/ coverage_overlap（process.a4_align·形状识别消费）。

铁律：纯整数（opcode/sid/immediate 全 int·assert_int 守）/ 确定性（DFS 前序固定 + children 按
  (order_index,NodeRef) 排 + PARAM sid 按 DFS 阅读序·bit-identical·禁 set 迭代序）/ 核心无墙钟 /
  不写死（无对齐规则硬编码·同构 = 结构等价比对·PARAM = 自由度涌现非预设语法·MIN_DISCOVER_SAMPLES
  进元定义常量同 MIN_DOMINANCE）/ 单向依赖（process→result.graph_view/storage/numeric 向下·同层
  process.a4_align OK）/ 不写死（K=2 = MIN_DISCOVER_SAMPLES 模级常量非魔数）。
诚实边界：等长同构 only / Rice（universal_guarantee=0·只证过有限基底）/ stable≠correct（频次门控本模块
  无·caller 选样）/ 小样本过早抽象（K=2 可抽象噪音·须 caller 质量闸·§二.3）/ struct_bind 不判匹配（序列4）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.numeric.symbol_domain import make_variable, index_of
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES, EDGE_IS_A, EDGE_CAUSES, EDGE_REALIZES
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_CONCEPTNET, SOURCE_CHINESE_KB
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_OPERATOR
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, read_composes_attrs,
    ATTR_OPERATOR, ATTR_IMMEDIATE, ATTR_OPERAND,
    ATTR_CTRL_TAG, ATTR_STORE_TARGET,
    ATTR_ORIGIN, ORIGIN_DISCOVERED, ATTR_OPERATOR_DEF, ATTR_ARITY,
    ATTR_SLOT_ROLE, ATTR_CUE_SIG,
    COMPOSES_ATTR_TABLE,
)
from pure_integer_ai.storage.op_confidence import read_op_confidence
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.shared.relation_primitives import REL_SUBSET, REL_CAUSES
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.arith_observe import register_arith_operator
from pure_integer_ai.cognition.understanding.realizes import build_realizes_edge
from pure_integer_ai.cognition.understanding.emergent_relation_signal import record_emergent_relation_signal_shadow
from pure_integer_ai.storage.structure_match_count import record_structure_match
from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map, set_lca
from pure_integer_ai.config import gates

# ---- 元定义常量（同 MIN_DOMINANCE 范式·非魔数·doc §4.6） ----
MIN_DISCOVER_SAMPLES = 2    # 触发结构发现的最小样本数（K=2·§二 legacy extract_pattern 先例）
_MAX_DISCOVER_DEPTH = 64    # DFS 深度闸（防病态深·同 read_composes_tree _COMPOSES_MAX_DEPTH）

# 形状签名叶哨兵（coverage_overlap 识别消费用·算子形状序列里所有叶统一标此·PARAM/IMM 不区分）
_LEAF_SIG = -1

# cue_sig 在 _shape_name payload 中的分隔哨兵（§十八 condition 6a）：distinct _LEAF_SIG=-1 与 ≥0 ConceptRef int·
# 仅 cue_sig 非空时插 abstract_sig 与 cue_sig 间·避两者拼接碰撞（abs=(X,) cue=() vs abs=() cue=(X,)·后者裸 NL+cue 拆
# 是真实场景·sentinel 区分）。cue_sig=() 不加 payload → 名同今 bit-identical。
_CUE_SIG_SEP = -2

# operand-input 识别探针值（per input logical variable·distinct 素数·纯整数确定·避 0/1 退化）
# 用于"探针值执行比对"——operand 输入（λz:z*z）无具体值·选探针喂骨架+input·比执行值（复用 execute_composes_value）。
# 避 0（mul 退化）+ 避 1（mul 加法恒等）+ distinct（抓获变量同一性错配）→ 小素数。arity 超 _MAX_PROBE_ARITY → 不识别 defer。
_PROBE_VALUES: tuple[tuple[int, int], ...] = (
    (2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1),
)
_MAX_PROBE_ARITY = len(_PROBE_VALUES)   # 探针覆盖最大 arity（超→operand-input 不识别 defer·现实 lambda arity≤6）

# §8.7-洗 洗净循环反馈半闭环：算子置信率 ×1000 缩放（sn/tn→rate·recognize 择优排序用·同 COVERAGE_SCALE 既有约定）
_OP_CONF_RATE_SCALE = 1000


@dataclass
class DiscoveryRouteStats:
    """结构发现路由的生产诊断计数。"""

    calls: int = 0
    input_roots: int = 0
    shape_groups: int = 0
    cue_clusters: int = 0
    lca_clusters: int = 0
    existing_key_clusters: int = 0
    new_discovery_clusters: int = 0
    fallback_clusters: int = 0
    discover_samples: int = 0
    recognize_samples: int = 0
    dropped_samples: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_roots": self.input_roots,
            "shape_groups": self.shape_groups,
            "cue_clusters": self.cue_clusters,
            "lca_clusters": self.lca_clusters,
            "existing_key_clusters": self.existing_key_clusters,
            "new_discovery_clusters": self.new_discovery_clusters,
            "fallback_clusters": self.fallback_clusters,
            "discover_samples": self.discover_samples,
            "recognize_samples": self.recognize_samples,
            "dropped_samples": self.dropped_samples,
        }


@dataclass
class StructureTallyStats:
    """结构到关系 tally 的生产诊断计数。"""

    calls: int = 0
    input_roots: int = 0
    realizes_skeletons: int = 0
    shape_matched_roots: int = 0
    aligned_roots: int = 0
    candidate_alignments: int = 0
    distinct_matches_added: int = 0
    shadow_edges_added: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_roots": self.input_roots,
            "realizes_skeletons": self.realizes_skeletons,
            "shape_matched_roots": self.shape_matched_roots,
            "aligned_roots": self.aligned_roots,
            "candidate_alignments": self.candidate_alignments,
            "distinct_matches_added": self.distinct_matches_added,
            "shadow_edges_added": self.shadow_edges_added,
        }


class SkeletonResult(NamedTuple):
    """结构发现产物（最小闭环·§八序列1）。"""
    skeleton_ref: ConceptRef   # 发现的骨架 struct_ref（NOP-root + COMPOSES·ATTR_ORIGIN=discovered）
    arity: int                 # PARAM 槽数（相异立即数位数·= make_variable(0..arity-1)·DFS 阅读序）


class _NoSkeleton(Exception):
    """内部控制流：无共性骨架（异构/越界 kind/深度超界）·discover 顶层 catch → 返 None。

    非外部 fail-loud（arith UnsupportedConstruct 是 caller 输入错须 raise；本模块"对不齐"是正常信号·
    caller 判 None 决定下一步·序列2 变长对齐）。仅内部用·不外泄。
    """


def _is_immediate_leaf(attrs: dict[int, tuple[int, int]]) -> bool:
    """立即数叶判定（ATTR_IMMEDIATE 在·且无算子/控制/STORE/OPERAND·loop1 scope）。"""
    return (ATTR_IMMEDIATE in attrs
            and ATTR_OPERATOR not in attrs
            and ATTR_CTRL_TAG not in attrs
            and ATTR_STORE_TARGET not in attrs
            and ATTR_OPERAND not in attrs)


def _is_operator_node(attrs: dict[int, tuple[int, int]]) -> bool:
    """算子节点判定（ATTR_OPERATOR 在·且无立即数/控制/STORE/OPERAND·loop1 scope）。"""
    return (ATTR_OPERATOR in attrs
            and ATTR_IMMEDIATE not in attrs
            and ATTR_CTRL_TAG not in attrs
            and ATTR_STORE_TARGET not in attrs
            and ATTR_OPERAND not in attrs)


def _is_operand_leaf(attrs: dict[int, tuple[int, int]]) -> bool:
    """operand 叶判定（ATTR_OPERAND 在·且无算子/立即数/控制/STORE·序列2 scope）。

    operand 叶 = lambda 参数叶（变量）·int_a=sid（make_variable(i)·arith builder 按 lambda 形参序分配）。
    与立即数叶互斥（ATTR_OPERAND 在↔ATTR_IMMEDIATE 不在）·故 build 里两分支独立。
    """
    return (ATTR_OPERAND in attrs
            and ATTR_OPERATOR not in attrs
            and ATTR_IMMEDIATE not in attrs
            and ATTR_CTRL_TAG not in attrs
            and ATTR_STORE_TARGET not in attrs)


def _is_concept_leaf(attrs: dict[int, tuple[int, int]]) -> bool:
    """语言 token 概念叶判定（S3 件2·钥匙①发现线·**无任何结构属性**）。

    语言 token 叶 = observe 件1 建的 COMPOSES 边 to 端 concept_ref·**不挂 composes_attr**（最小新机制·
    省 ATTR_CONCEPT_LEAF kind·决断§8.8）。判据 = attrs 为空 dict（无 OPERATOR/CTRL/OPERAND/IMMEDIATE/
    STORE_TARGET）。arith/code 叶全挂 IMM/OPERAND·operator 节点挂 OPERATOR·**无属性叶唯一来自语言 token**·
    判据充分互斥（与 _is_immediate_leaf/_is_operand_leaf/_is_operator_node 互斥：那些要求某 kind 在·本判定全无）。
    """
    return not attrs


def _is_store_node(attrs: dict[int, tuple[int, int]]) -> bool:
    """STORE 节点判定（ATTR_STORE_TARGET 在·且有子·无算子/立即数/控制/OPERAND·S3 迭代骨架支持）。

    STORE 节点 = 迭代累加器/索引回写（arith_observe._new_store·doc §五）。int_a=目标 sid（acc/idx·internal）·
    1 子（值源·slot 0）。与算子节点（ATTR_OPERATOR）/立即数叶/operand 叶/CTRL 节点互斥。
    """
    return (ATTR_STORE_TARGET in attrs
            and ATTR_OPERATOR not in attrs
            and ATTR_IMMEDIATE not in attrs
            and ATTR_CTRL_TAG not in attrs
            and ATTR_OPERAND not in attrs)


def _is_ctrl_node(attrs: dict[int, tuple[int, int]]) -> bool:
    """CTRL 节点判定（ATTR_CTRL_TAG 在·无算子/立即数/STORE/OPERAND·S3 迭代骨架支持）。

    CTRL 节点 = 控制流根（CTRL_WHILE·arith_observe._build_iterative_block c2·doc §五）。int_a=CTRL_* tag·
    2 子 [cond(slot0), body(slot1)]。COMPOSES 图是严格 DAG（回边在字节码·graph_compile:136-138/242 钉死）·
    CTRL_WHILE 是普通 DAG 节点可对齐（doc/重来_S3S4迭代机制设计 §三-bis）。
    """
    return (ATTR_CTRL_TAG in attrs
            and ATTR_OPERATOR not in attrs
            and ATTR_IMMEDIATE not in attrs
            and ATTR_STORE_TARGET not in attrs
            and ATTR_OPERAND not in attrs)


def _collect_internal_store_sids(backend, graph: ConceptGraph,
                                 root: ConceptRef) -> list[int]:
    """收集 root 子树全部 ATTR_STORE_TARGET sid（S3 internal sid 预扫·纯读 L5）。

    返 sorted unique list（确定性·sorted 锁·镜像 arith_observe._deep_copy_subtree:269 `sorted(set(...))` 范式）。
    internal sid = acc/idx 等迭代内部变量（DSL STORE 目标永是 internal·lambda 参只读·二者互斥→判据干净）。
    空 list = 无 STORE（直线算子树·非迭代·S3 加性零行为变）。
    """
    store_target_of = graph.read_composes_tree(root)[4]
    return sorted(set(store_target_of.values()))


def _build_cross_internal_sid(backend, graph: ConceptGraph, samples: list[ConceptRef],
                              internal_sids: list[int]) -> dict[int, dict[int, int]] | None:
    """跨样本 internal sid 对应表（sorted 锁·sample0 sorted[k] ↔ sample_i sorted[k]）。

    S3 ctrl/store-迭代骨架支持的命门：不同样本可能分配异 internal sid（如 `lambda a,b: Sigma(1,a,i)`
    的 acc=mv2 vs `lambda n: Sigma(1,n,i)` 的 acc=mv1·异 lambda arity 致 internal sid 偏移）·
    sorted 序 = 分配序（acc 永远先于 idx 分配·make_variable 单调→sorted 保分配序）·故 sample0 sorted[k]
    与 sample_i sorted[k] 是同一逻辑变量（acc/idx/...）·位置对应。镜像 _deep_copy_subtree sid_remap 确定性。

    返 dict[int, dict[int,int]]（sample_i(i>0) → {sample0 internal sid → sample_i internal sid}）。
    None = 某 sample_i 的 internal sid 数 ≠ sample0 → 不同 ctrl/store 结构 → 无共性骨架（caller 返 None）。
    无 internal sid（直线算子）→ {} （所有 sample_i 循环不进·空字典·零行为变）。
    """
    cross: dict[int, dict[int, int]] = {}
    for i in range(1, len(samples)):
        si_sorted = _collect_internal_store_sids(backend, graph, samples[i])
        if len(si_sorted) != len(internal_sids):
            return None   # internal sid 数异 → 不同 ctrl/store 结构
        cross[i] = dict(zip(internal_sids, si_sorted))
    return cross


def shape_signature(graph: ConceptGraph, root: ConceptRef) -> list[int]:
    """root COMPOSES 子树的**算子形状签名**（BFS 序·算子→opcode·叶→_LEAF_SIG 哨兵）。

    供 coverage_overlap 识别消费（§8.7）：两程序算子形状同构 → 同签名 → coverage=1000 = 认出。
    叶统一标 _LEAF_SIG（PARAM 槽与 IMM 固定位形状不区分·纯结构形状·值/自由度不进签名）。
    确定性：BFS + children 按 (order_index,NodeRef) 排（read_composes_tree 已排）。形状指纹用 BFS
    （与 build 的 DFS 区分·无妨：两同形树任一确定遍历产同签名·识别只比同形）。
    """
    from collections import deque
    children_of, operator_of, _operand_of, _immediate_of, _store_target_of = \
        graph.read_composes_tree(root)
    sig: list[int] = []
    visited: set[ConceptRef] = set()
    queue: deque[ConceptRef] = deque([root])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node in operator_of:
            sig.append(operator_of[node])
        else:
            sig.append(_LEAF_SIG)   # 叶（IMM/OPERAND/STORE 等非算子节点·形状统一标叶）
        for child in children_of.get(node, []):
            if child not in visited:
                queue.append(child)
    return sig


def discover_skeleton(samples: list[ConceptRef], *, concept_index, edge_store,
                      backend, space_id: int, source: int,
                      skeleton_label: str,
                      slot_lcas: list[ConceptRef | None] | None = None,
                      cue_sig: tuple[ConceptRef | None, ...] | None = None) -> SkeletonResult | None:
    """多样本 COMPOSES 程序 → 等长结构对齐 → 抽共性骨架 → 落 struct_ref+COMPOSES（ATTR_ORIGIN=discovered）。

    samples        : ≥ MIN_DISCOVER_SAMPLES 个 COMPOSES 程序树根（observer 已建·同 space）。须为
                     算子 + 立即数/operand 叶树（operand=序列2 已支持·变量同一性）·结构同构。
    skeleton_label : 骨架 struct_ref 的 surface 前缀（per-space dedup·隔离不同发现）。
    slot_lcas      : S3 第二刀 Interp2 抽象对撞·每 PARAM CONCEPT slot 的 IS_A LCA 类（ConceptRef|None·
                     DFS CONCEPT_LEAF 阅读序·与 build CONCEPT_LEAF 分支 _concept_slot_idx 对齐）。None/缺省 =
                     不挂 ATTR_SLOT_ROLE（arith/裸 NL·bit-identical 向后兼容）。slot 序 = build CONCEPT_LEAF
                     首遇序（_concept_slot_idx·非 _param_idx·后者含 immediate/operand PARAM 槽）。
    cue_sig         : §十八 condition 6a cue 拆簇·每 PARAM CONCEPT slot 的闭类 cue token ConceptRef（ConceptRef|None·
                     DFS CONCEPT_LEAF 阅读序·与 _concept_slot_idx 对齐·镜像 slot_lcas）。None/缺省 = 不挂 ATTR_CUE_SIG
                     （非 cue 拆 / arith / 裸 NL·bit-identical 向后兼容）。仅 caller auto_discover _cluster_by_cue 透传。
    返 SkeletonResult(skeleton_ref, arity) | None（<K 样本 / 结构异构 / 深度超界
    / operand cross-sample 变量同一性不一致 / internal sid 数异 → None）。
    ctrl（CTRL_WHILE）/ store（STORE）迭代骨架 = S3 已支持（§三-bis·Sigma/Prod/Recur 可发现）。

    机制：并行 DFS 前序各样本同位节点（样本0 作模板）·算子位 opcode 须一致·立即数位同值→固定位/异值
    →PARAM 槽（DFS 阅读序赋 sid·与 inline arg_subst AST 位置序契约对齐）。落 skeleton 镜像样本0 形状
    （NOP-root struct_ref + COMPOSES）·PARAM 槽 sid=make_variable(0..arity-1)。
    反 theater：骨架经 register_arith_operator 注册后可被 inline 消费（Call+β）/ coverage_overlap 识别。
    """
    if len(samples) < MIN_DISCOVER_SAMPLES:
        return None   # 须 ≥K 样本对齐（K=MIN_DISCOVER_SAMPLES·元定义常量）
    for s in samples:
        assert_int(s[0], s[1], _where="discover_skeleton.sample")
        # 样本必须是算子 root（OPCODE_NOP/OPCODE_*·COMPOSES 程序根·design§二"root 必须算子节点"·
        # 件2 CONCEPT_LEAF 无属性叶判定须防 root 误判：bare 裸概念点无 ATTR_OPERATOR→非程序根→None）。
        if ATTR_OPERATOR not in read_composes_attrs(backend, s):
            return None
    assert_int(space_id, source, _where="discover_skeleton")
    graph = ConceptGraph(backend)
    k = len(samples)
    children_of_all = [graph.read_composes_tree(s)[0] for s in samples]   # 各样本 children_of（对齐地基）

    # S3 ctrl/store-迭代骨架支持（doc/重来_S3S4迭代机制设计 §三-bis·加性·直线算子树零行为变）：
    # 预扫 sample0 internal STORE 目标 sid + 跨样本对应 + arity 预计（两遍：先 probe 计 arity·再 build）。
    # internal sid alpha-重命名到 make_variable(arity+k)·避 PARAM 区 mv0..mv(arity-1)·镜像 _deep_copy_subtree:269。
    internal_sids = _collect_internal_store_sids(backend, graph, samples[0])
    cross_internal_sid: dict[int, dict[int, int]] = {}
    if internal_sids:
        cross_internal_sid = _build_cross_internal_sid(backend, graph, samples, internal_sids)  # type: ignore[assignment]
        if cross_internal_sid is None:
            return None   # 某 sample_i internal sid 数异 → 不同 ctrl/store 结构 → 无共性骨架
        # arity 预计（_probe_walk 已扩 ctrl/store·drift twin·须==build _param_idx·test_probe_arity_matches_discover 守）
        state = _ProbeState()
        try:
            _probe_walk(backend, tuple(samples), children_of_all, k, state, depth=0,
                        internal_sids=set(internal_sids),
                        cross_internal_sid=cross_internal_sid)
        except _NoSkeleton:
            return None
        arity_probe = state.param_idx
        internal_sid_remap = {orig: make_variable(arity_probe + kk)
                              for kk, orig in enumerate(internal_sids)}
    else:
        internal_sid_remap = {}   # 无 STORE（直线算子）→ 无 internal 重映射（零行为变）

    builder = _SkeletonBuilder(
        concept_index=concept_index, edge_store=edge_store, backend=backend,
        space_id=space_id, source=source, skeleton_label=skeleton_label,
        slot_lcas=slot_lcas,
        cue_sig=cue_sig,
        internal_sid_remap=internal_sid_remap,
        cross_internal_sid=cross_internal_sid,
        internal_sids=set(internal_sids))
    try:
        skeleton_root = builder.build(tuple(samples), children_of_all, k, depth=0)
    except _NoSkeleton:
        return None
    # root 标 ATTR_ORIGIN=discovered（§4.4 选B·零 core 迁移）
    record_composes_attr(backend, ref=skeleton_root, kind=ATTR_ORIGIN, int_a=ORIGIN_DISCOVERED)
    return SkeletonResult(skeleton_ref=skeleton_root, arity=builder.param_count)


class _SkeletonBuilder:
    """骨架建造者（并行 DFS 前序对齐 + fresh 节点/边落盘·持 concept_index/edge_store/backend + 计数器）。

    DFS 前序（非 BFS）：PARAM sid 按左→右叶阅读序分配（与 inline arg_subst AST 位置序契约对齐·对抗
    正确性审计纠错）。fresh 节点按 DFS 前序建。DAG 共享：seen 跳过第二次·复用 fresh_map[t0]（loop1
    严格树不触发·arith builder surface 唯一无共享）。
    """

    def __init__(self, *, concept_index, edge_store, backend,
                 space_id: int, source: int, skeleton_label: str,
                 slot_lcas: list[ConceptRef | None] | None = None,
                 cue_sig: tuple[ConceptRef | None, ...] | None = None,
                 internal_sid_remap: dict[int, int] | None = None,
                 cross_internal_sid: dict[int, dict[int, int]] | None = None,
                 internal_sids: set[int] | None = None) -> None:
        self._ci = concept_index
        self._es = edge_store
        self._b = backend
        self._space_id = space_id
        self._source = source
        self._label = skeleton_label
        self._seq = 0                # fresh 节点序号（DFS 前序·surface dedup）
        self._param_idx = 0          # PARAM 槽计数器（make_variable index·DFS 阅读序）
        self._fresh_map: dict[ConceptRef, ConceptRef] = {}   # 样本0 节点 → fresh 骨架节点
        self._seen: set[ConceptRef] = set()
        # 序列2 operand 对应 + 变量同一性状态（discover OPERAND 叶用）
        self._sample0_sid_to_slot: dict[int, int] = {}            # sample0 operand sid → slot（within-sample 同一性·同 sid 复用同槽）
        self._cross_slot_to_sid: dict[int, dict[int, int]] = {}  # sample_i(i>0) → {slot → sample_i sid}（cross-sample 一致性门）
        # S3 第二刀 Interp2 抽象对撞：CONCEPT slot LCA 类（ATTR_SLOT_ROLE 写盘·DFS CONCEPT_LEAF 阅读序）
        self._slot_lcas = slot_lcas
        # §十八 condition 6a：CONCEPT slot 闭类 cue token（ATTR_CUE_SIG 写盘·DFS CONCEPT_LEAF 阅读序 _concept_slot_idx·镜像 _slot_lcas）
        self._cue_sig = cue_sig
        self._concept_slot_idx = 0   # CONCEPT_LEAF DFS 首遇序（≠ _param_idx·后者含 immediate/operand PARAM·纯 CONCEPT 计数）
        # S3 ctrl/store-迭代骨架支持（§三-bis）：internal STORE 目标 sid alpha-重命名（避 PARAM 区·镜像 _deep_copy_subtree）
        self._internal_sid_remap: dict[int, int] = internal_sid_remap or {}
        self._cross_internal_sid: dict[int, dict[int, int]] = cross_internal_sid or {}
        self._internal_sids: set[int] = internal_sids or set()

    @property
    def param_count(self) -> int:
        return self._param_idx

    def build(self, nodes: tuple[ConceptRef, ...],
              children_of_all: list[dict[ConceptRef, list[ConceptRef]]],
              k: int, depth: int) -> ConceptRef:
        """并行 DFS 前序：同构门控 + fresh 节点 + 边落盘 → 返 fresh 骨架节点。raise _NoSkeleton = 无共性骨架。"""
        if depth > _MAX_DISCOVER_DEPTH:
            raise _NoSkeleton()      # 病态深（防栈溢出·同 read_composes_tree max_depth）
        t0 = nodes[0]                # 样本0 模板节点
        if t0 in self._seen:
            return self._fresh_map[t0]   # DAG 共享：复用已建 fresh（loop1 严格树不触发）
        self._seen.add(t0)
        all_attrs = [read_composes_attrs(self._b, n) for n in nodes]
        a0 = all_attrs[0]

        if _is_immediate_leaf(a0):
            # ---- 立即数叶：全样须同类·同值→固定位 / 异值→PARAM 槽（DFS 阅读序赋 sid） ----
            for a in all_attrs[1:]:
                if not _is_immediate_leaf(a):
                    raise _NoSkeleton()   # 同位叶类型不一致 → 无共性骨架
            vals = [a[ATTR_IMMEDIATE] for a in all_attrs]
            fresh = self._new_node("IMM" if len(set(vals)) == 1 else "PARAM")
            if len(set(vals)) == 1:
                num, den = vals[0]
                record_composes_attr(self._b, ref=fresh, kind=ATTR_IMMEDIATE,
                                     int_a=num, int_b=den)
            else:
                sid = make_variable(self._param_idx)   # DFS 前序阅读序（左→右叶·与 inline arg_subst 对齐）
                self._param_idx += 1
                record_composes_attr(self._b, ref=fresh, kind=ATTR_OPERAND, int_a=sid)
            self._fresh_map[t0] = fresh
            return fresh

        if _is_operand_leaf(a0):
            # ---- 序列2 operand 叶：within-sample 变量同一性 + cross-sample 一致性 ----
            # sample0 operand 叶（lambda 参数·sid=make_variable(i)）·与立即数叶互斥。
            for a in all_attrs[1:]:
                if not _is_operand_leaf(a):
                    raise _NoSkeleton()   # 同位 operand/立即数 类型不一致 → 不同算子类 → 无共性骨架
            s0 = a0[ATTR_OPERAND][0]      # sample0 此位的变量 sid
            if s0 in self._internal_sids:
                # S3 ctrl/store-迭代骨架（§三-bis）：internal LOAD（acc/idx 引用）·alpha-rename 非 PARAM 槽。
                # DSL STORE 目标永是 internal（acc/idx）·lambda 参只读·二者互斥→判据干净。不增 _param_idx
                # （internal 非参数自由度·vm_proof 绑 input_args 到 PARAM sid·STORE/LOAD 管 internal sid·互不踩）。
                alpha = self._internal_sid_remap.get(s0)
                if alpha is None:
                    raise _NoSkeleton()   # 防御（预扫应已收集此 internal sid）
                fresh = self._new_node("IPARAM")   # internal operand 叶（LOAD acc/idx·alpha sid）
                record_composes_attr(self._b, ref=fresh, kind=ATTR_OPERAND, int_a=alpha)
                # cross-sample 一致性：sample_i 此位须引用对应 internal sid（cross_internal_sid 位置对应表）
                for i in range(1, k):
                    si = all_attrs[i][ATTR_OPERAND][0]
                    expected = self._cross_internal_sid.get(i, {}).get(s0)
                    if expected is None or si != expected:
                        raise _NoSkeleton()   # sample_i 引用非对应 internal sid → 结构错位
                self._fresh_map[t0] = fresh
                return fresh
            # PARAM 路径（lambda 参数·既有·bit-identical）
            slot = self._sample0_sid_to_slot.get(s0)
            if slot is None:
                slot = self._param_idx        # DFS 首遇序（与立即数 PARAM 共计数器·阅读序·与 inline arg_subst 对齐）
                self._param_idx += 1
                self._sample0_sid_to_slot[s0] = slot   # 同 s0 复用同 slot = within-sample 变量同一性（x 两次→同槽）
            # 镜像 sample0 形状：每 operand 出现位 = 独立 fresh 叶（同 sid=make_variable(slot) 编同一性·非 DAG 共享）
            fresh = self._new_node("PARAM")
            record_composes_attr(self._b, ref=fresh, kind=ATTR_OPERAND,
                                 int_a=make_variable(slot))
            # cross-sample 一致性门（sample0 为模板·同 loop1 范式）：sample0 同槽位的诸位置→sample_i 须同 sid。
            # 允许坍缩（sample0 多槽→sample_i 同 sid·square 是 mul 实例）·禁止拆分（square 样本集混 mul→拒）。
            for i in range(1, k):
                si = all_attrs[i][ATTR_OPERAND][0]
                slot_map = self._cross_slot_to_sid.setdefault(i, {})
                prev = slot_map.get(slot)
                if prev is not None and prev != si:
                    raise _NoSkeleton()   # sample_i 在 sample0 同槽位出现两不同 sid → 拆 sample0 变量 → 非实例
                slot_map[slot] = si
            self._fresh_map[t0] = fresh
            return fresh

        if _is_operator_node(a0):
            # ---- 算子节点：全样须同 opcode·无越界 kind·同子数 → 递归子（DFS 前序）----
            opcode = a0[ATTR_OPERATOR][0]
            for a in all_attrs[1:]:
                if not _is_operator_node(a) or a[ATTR_OPERATOR][0] != opcode:
                    raise _NoSkeleton()   # opcode 不一致 → 无共性骨架
            child_lists = [children_of_all[i].get(nodes[i], []) for i in range(k)]
            cc0 = len(child_lists[0])
            for cl in child_lists[1:]:
                if len(cl) != cc0:
                    raise _NoSkeleton()   # 子数不一致 → 结构异构
            fresh = self._new_node("OP")
            record_composes_attr(self._b, ref=fresh, kind=ATTR_OPERATOR, int_a=opcode)
            self._fresh_map[t0] = fresh
            for slot in range(cc0):
                child_nodes = tuple(cl[slot] for cl in child_lists)
                child_fresh = self.build(child_nodes, children_of_all, k, depth + 1)
                self._edge(fresh, child_fresh, slot)   # 父已知 fresh·子递归返 fresh·直接落边
            return fresh

        if _is_concept_leaf(a0):
            # 件2·语言 token 概念叶（S3 钥匙①·**全参化无固定位**·语言 token 永远是参数·Plan agent 修正：镜像
            # OPERAND 非 IMMEDIATE·无"固定词"概念）。全样须同位为概念叶（无属性叶·否则异质→_NoSkeleton）。
            # sample0 此位 token ref = t0。永远 PARAM 槽（无固定位）。within-sample 同一性靠 DAG 共享（:263
            # 同 token concept_ref→同 t0→复用同 fresh 同 slot·"猫追猫"两猫同 ref→DAG 共享同槽·**与 operand 用
            # _sample0_sid_to_slot 不同**：语言 token 同 ref concept_index 幂等·arith operand 两 x 是两 AST 节点须
            # sid 映射守同槽）·故只 _param_idx 首次序·无须 ref→slot 映射。cross-sample 全放开（D2 弱化门极致·语言
            # PARAM=词槽非形参实例·开放词表跨样本异 ref 同槽允许=语言泛化牙·"猫追狗"/"狗追猫"同结构异词·无门控无须状态）。
            for a in all_attrs[1:]:
                if not _is_concept_leaf(a):
                    raise _NoSkeleton()
            slot = self._param_idx
            self._param_idx += 1
            fresh = self._new_node("CONCEPT")
            record_composes_attr(self._b, ref=fresh, kind=ATTR_OPERAND,
                                 int_a=make_variable(slot))
            # S3 第二刀 Interp2：抽象级 PARAM slot 的 IS_A LCA 类挂 fresh 节点（ATTR_SLOT_ROLE·与 ATTR_OPERAND
            # 同节点第二 attr·caller auto_discover LCA 聚类透传）。slot_lcas 按 DFS CONCEPT_LEAF 首遇序（_concept_slot_idx）·
            # 非 _param_idx（后者含 immediate/operand PARAM 槽·纯语言场景两序一致·混合场景分离守语义）。None/越界→不写（absence=无类约束）。
            if self._slot_lcas is not None and self._concept_slot_idx < len(self._slot_lcas):
                lca_ref = self._slot_lcas[self._concept_slot_idx]
                if lca_ref is not None:
                    record_composes_attr(self._b, ref=fresh, kind=ATTR_SLOT_ROLE,
                                         int_a=lca_ref[0], int_b=lca_ref[1])
            # §十八 condition 6a：cue 拆簇拆位的闭类 cue token 挂 fresh 节点（ATTR_CUE_SIG·镜像 ATTR_SLOT_ROLE·
            # 同节点第三 attr·caller auto_discover _cluster_by_cue 透传）。按 _concept_slot_idx·仅拆位非 None 写
            # （gate CUE_CLUSTER_MODE + sustainable-split）·absence=非 cue 位（load _collect_cue_sig 读 None→名同今 bit-identical）。
            # **写侧匹配读侧**（6a-1 review 关注点）：_collect_cue_sig 读 attrs.get(ATTR_CUE_SIG)→None if absent·此处仅 cue_ref 非 None 时写。
            if self._cue_sig is not None and self._concept_slot_idx < len(self._cue_sig):
                cue_ref = self._cue_sig[self._concept_slot_idx]
                if cue_ref is not None:
                    record_composes_attr(self._b, ref=fresh, kind=ATTR_CUE_SIG,
                                         int_a=cue_ref[0], int_b=cue_ref[1])
            self._concept_slot_idx += 1
            self._fresh_map[t0] = fresh
            return fresh

        if _is_store_node(a0):
            # S3 ctrl/store-迭代骨架（§三-bis）：STORE 节点（ATTR_STORE_TARGET·累加器/索引回写）。
            # 全样须同位 STORE·store_target sid 经 internal alpha-重映射写 fresh 节点·递归唯一 value 子（slot 0）。
            for a in all_attrs[1:]:
                if not _is_store_node(a):
                    raise _NoSkeleton()   # 同位非 STORE → 结构异构
            s0 = a0[ATTR_STORE_TARGET][0]
            if s0 not in self._internal_sids:
                raise _NoSkeleton()   # STORE 目标非预扫 internal（防御·DSL STORE 目标永 internal）
            alpha = self._internal_sid_remap.get(s0)
            if alpha is None:
                raise _NoSkeleton()   # 防御
            fresh = self._new_node("STORE")
            record_composes_attr(self._b, ref=fresh, kind=ATTR_STORE_TARGET, int_a=alpha)
            # cross-sample 一致性：sample_i 此位 STORE 目标须为对应 internal sid
            for i in range(1, k):
                si = all_attrs[i][ATTR_STORE_TARGET][0]
                expected = self._cross_internal_sid.get(i, {}).get(s0)
                if expected is None or si != expected:
                    raise _NoSkeleton()   # sample_i STORE 目标非对应 internal sid → 结构错位
            self._fresh_map[t0] = fresh
            child_lists = [children_of_all[i].get(nodes[i], []) for i in range(k)]
            cc0 = len(child_lists[0])
            for cl in child_lists[1:]:
                if len(cl) != cc0:
                    raise _NoSkeleton()   # 子数不一致 → 结构异构
            for slot in range(cc0):
                child_nodes = tuple(cl[slot] for cl in child_lists)
                child_fresh = self.build(child_nodes, children_of_all, k, depth + 1)
                self._edge(fresh, child_fresh, slot)
            return fresh

        if _is_ctrl_node(a0):
            # S3 ctrl/store-迭代骨架（§三-bis）：CTRL 节点（ATTR_CTRL_TAG·CTRL_WHILE 迭代）。
            # 全样须同 ATTR_CTRL_TAG（int_a 一致·CTRL_WHILE）·fresh 记 ATTR_CTRL_TAG·递归 [cond(slot0), body(slot1)]。
            # COMPOSES 图严格 DAG（graph_compile 钉死）·CTRL_WHILE 是普通 DAG 节点可对齐（无图回边·回边在字节码）。
            ctrl_tag = a0[ATTR_CTRL_TAG][0]
            for a in all_attrs[1:]:
                if not _is_ctrl_node(a) or a[ATTR_CTRL_TAG][0] != ctrl_tag:
                    raise _NoSkeleton()   # 同位非 CTRL 或 tag 异 → 结构异构
            fresh = self._new_node("CTRL")
            record_composes_attr(self._b, ref=fresh, kind=ATTR_CTRL_TAG, int_a=ctrl_tag)
            self._fresh_map[t0] = fresh
            child_lists = [children_of_all[i].get(nodes[i], []) for i in range(k)]
            cc0 = len(child_lists[0])
            for cl in child_lists[1:]:
                if len(cl) != cc0:
                    raise _NoSkeleton()   # 子数不一致 → 结构异构（CTRL_WHILE 须 2 子 [cond,body]）
            for slot in range(cc0):
                child_nodes = tuple(cl[slot] for cl in child_lists)
                child_fresh = self.build(child_nodes, children_of_all, k, depth + 1)
                self._edge(fresh, child_fresh, slot)
            return fresh

        # 越界 kind（混合属性/未知）→ 无共性骨架（ctrl/store 已支持·此处仅病态混合触达）
        raise _NoSkeleton()

    # ---- 节点/边原语（镜像 _ArithBuilder·per-space dedup）----

    def _new_node(self, role: str) -> ConceptRef:
        """建 fresh 骨架节点 ConceptRef（surface 含 label+前序序号+role·per-space dedup）。"""
        self._seq += 1
        surface = f"__skel_{self._label}_{self._seq}_{role}"
        return self._ci.ensure(surface, space_id=self._space_id,
                               tier=TIER_PRIMARY, node_type=NODE_OPERATOR)

    def _edge(self, parent: ConceptRef, child: ConceptRef, order_index: int) -> None:
        """落 EDGE_COMPOSES 边（父→子·order_index 显式·read_composes_tree 按 (order_index,NodeRef) 排）。"""
        assert_int(order_index, _where="_SkeletonBuilder._edge.order_index")
        self._es.add(space_id_from=parent[0], local_id_from=parent[1],
                     space_id_to=child[0], local_id_to=child[1],
                     edge_type=EDGE_COMPOSES, strength=1, source=self._source,
                     epistemic_origin=EPI_STRUCTURED, order_index=order_index)


# ---- 序列6-min：算子自动发现生产触发（de-theater 序列1·§八序列6·2026-07-03）----

# 形状→确定名 的 Hasher 种子（固定字面·跨 run bit-identical·§八.6 幂等门 + inline 查表用）
_SHAPE_NAME_SEED = "structure_discover.shape_name"


class DiscoveredOperator(NamedTuple):
    """auto_discover_operators 产物（序列6-min·§八序列6）。

    name_ref（§8.7-洗·2026-07-03）：算子 name 节点 ConceptRef（ATTR_OPERATOR_DEF/ARITY 挂载点·不在
    COMPOSES 子树内）·op_confidence 台账键·recognize 择优读 + _verify_generalization 写置信度用它。
    默认 (0,0) 向后兼容（无置信度场景不读）。
    """
    name: str                 # 注册名 __op_disc_{h63}（caller / inline 查表用）
    skeleton_ref: ConceptRef  # 发现的骨架 struct_ref（ATTR_ORIGIN=discovered·discover_skeleton 产）
    arity: int                # PARAM 槽数（= make_variable(0..arity-1)）
    sample_count: int         # 该形状累积的相异程序根数（≥ MIN_DISCOVER_SAMPLES）
    name_ref: ConceptRef = (0, 0)   # 算子 name 节点（op_confidence 键·§8.7-洗·默认向后兼容）
    forming_roots: tuple[ConceptRef, ...] = ()   # Phase D §十六-bis D.1：本骨架的 forming-sample roots（option-b oracle-pair-match REALIZES 标用·默认空向后兼容·load 重建不知→()）


def _normalize_abstract_sig(abstract_sig: tuple) -> tuple:
    """abstract_sig 全 None（无类约束）归一为 ()·守 bit-identical（arith/裸 NL 同名）·S3 第二刀 Interp2。

    abstract_sig = tuple[ConceptRef | None]（每 PARAM slot 的 IS_A LCA ref·DFS 阅读序·None=无 LCA/通配）。
    全 None（arith 骨架无 CONCEPT_LEAF / 裸 NL 语言骨架 ancestor_map 空无 LCA）→ 归一 ()·与历史 Interp1 名一致·
    守跨 run resume（load_discovered_operators 重派生 name 命中不 orphan）+ 幂等门（auto_discover lookup 命中）。
    任一 slot 有真 LCA ref → 原样返（异名·真行为变·正确）。
    """
    if not abstract_sig:
        return ()
    if all(item is None for item in abstract_sig):
        return ()
    return abstract_sig


def _flatten_abstract_sig(abstract_sig: tuple) -> tuple[int, ...]:
    """归一后的 abstract_sig → hash 输入扁平 int 序列（每 slot: ConceptRef→(sid,lid)·None 占位不进此函数·归一后无 None）。

    归一后 abstract_sig 至少含一个真 LCA ref（None 已 _normalize_abstract_sig 滤除全 None 情形）·
    但 mixed（部分 slot None + 部分 slot ref）仍可能·None slot 扁平为 (0,0) 占位保 slot 位置序。
    """
    flat: list[int] = []
    for item in abstract_sig:
        if item is None:
            flat.extend((0, 0))
        else:
            flat.extend((item[0], item[1]))
    return tuple(flat)


def _shape_name(sig: tuple[int, ...], arity: int, abstract_sig: tuple = (),
                cue_sig: tuple = ()) -> str:
    """形状签名 + arity + abstract_sig → 确定性注册名（Hasher 63-bit·固定种子·跨 run bit-identical）。

    **Half B（§八.7②·arity 进名·Finding1 真修）**：arity 进 hash 输入 → square(sig,arity1) 与 mul(sig,arity2)
    虽同 shape_signature（叶统一 _LEAF_SIG·跨类识别用·**不改**——operand 骨架须匹配 immediate 输入做识别·改签名破
    跨类识别）但 arity 异 → **异名** → 跨 run 载 square 后 mul 样本（同形异 arity）仍可独立发现（非 sig-only 路由吞）。

    **Interp2 抽象对撞（S3 第二刀·2026-07-05）**：abstract_sig 进 hash → 同 (sig,arity) 异抽象类（动物 vs 非生物）
    → **异名** → 同 shape 同 arity 语言骨架按 IS_A LCA 类分桶独立发现（破 D2 弱化门全 PARAM collapse）。
    abstract_sig 经 _normalize_abstract_sig 归一：全 None（无 LCA·arith/裸 NL）→ () → **名同今**（bit-identical 守跨 run
    resume + 幂等门）·仅真有 LCA ref 时异名。区分靠 abstract_sig 进名非改 shape_signature（shape 仍纯拓扑·跨类识别用）。

    **condition 6a cue_sig（§十八·2026-07-17）**：cue_sig 进 hash → 同 (sig,arity,abstract_sig) 异闭类 cue（是 vs 使·
    same-shape-same-LCA·唯 cue 区分）→ **异名** → 各产独立骨架（破 cue 坍缩·REALIZES 可异标 IS_A/CAUSES）。cue_sig
    经 _normalize_abstract_sig 归一（复用·同 tuple[ConceptRef|None] 形·全 None→()）+ _flatten_abstract_sig 扁平。**sentinel
    隔离**（_CUE_SIG_SEP=-2·仅 cue_sig 非空时插 abstract_sig 与 cue_sig 间）：避两者拼接碰撞（abs=(X,) cue=() vs
    abs=() cue=(X,)·后者裸 NL+cue 拆是真实场景·sentinel 区分）。cue_sig=() → 不加 payload → **名同今**（bit-identical）。
    **关系 label 走外源 oracle 非读 cue**（label_realizes·§十八 condition 6 复合键·禁单 primitive 单射 relation·anti-theater）。

    名 = `__op_disc_{h63((*sig, arity, *flatten(normalize(abstract_sig))[, _CUE_SIG_SEP, *flatten(normalize(cue_sig))]))}`
    （arity+abstract_sig 进 hash·cue_sig 非空时 sentinel+cue_sig 进 hash·
    名仍单 opacity token·debug 由 DiscoveredOperator.arity/ATTR_SLOT_ROLE 字段）。
    名前缀 __op_disc_ 隔离（避免撞用户/observe 概念）。
    """
    norm_abs = _normalize_abstract_sig(abstract_sig)
    payload: tuple = (*sig, arity, *_flatten_abstract_sig(norm_abs))
    norm_cue = _normalize_abstract_sig(cue_sig)   # 复用（同 tuple[ConceptRef|None] 形·全 None→()）
    if norm_cue:   # cue_sig 非空 → sentinel + 扁平（空时不加·bit-identical·守名同今）
        payload = (*payload, _CUE_SIG_SEP, *_flatten_abstract_sig(norm_cue))
    return f"__op_disc_{Hasher(_SHAPE_NAME_SEED).h63(payload)}"


# ---- Half B：只读 arity 探针（arity-in-name 幂等/路由 pre-check·避免重 build orphan·§八.7②） ----


class _ProbeState:
    """probe_arity 的可变对齐状态（镜像 _SkeletonBuilder 计数器·纯读不写盘）。

    param_idx            PARAM 槽计数器（= discover_skeleton arity·DFS 阅读序）
    sample0_sid_to_slot  sample0 operand sid → slot（within-sample 变量同一性·同 sid 复用同槽）
    cross_slot_to_sid    sample_i(i>0) → {slot → sample_i sid}（cross-sample 一致性门·坍缩允/拆分拒）
    seen                 已访问 sample0 节点集（DAG 共享跳过·镜像 _SkeletonBuilder._seen·loop1 严格树不触发）
    """

    __slots__ = ("param_idx", "sample0_sid_to_slot", "cross_slot_to_sid", "seen")

    def __init__(self) -> None:
        self.param_idx = 0
        self.sample0_sid_to_slot: dict[int, int] = {}
        self.cross_slot_to_sid: dict[int, dict[int, int]] = {}
        self.seen: set[ConceptRef] = set()


def _probe_walk(backend, nodes: tuple[ConceptRef, ...],
                children_of_all: list[dict[ConceptRef, list[ConceptRef]]],
                k: int, state: _ProbeState, depth: int, *,
                internal_sids: set[int] | None = None,
                cross_internal_sid: dict[int, dict[int, int]] | None = None) -> None:
    """镜像 `_SkeletonBuilder.build` 的并行 DFS 前序对齐 walk·只更 state 计数器**不建盘**。raise _NoSkeleton=异构。

    与 build 同分支序 + 同槽规则 + 同一致性门（drift 防线·probe_arity 须==discover_skeleton.arity·test 守）：
      immediate 叶：全样同类·同值→固定位（不增 arity）/ 异值→PARAM 槽（param_idx+1·DFS 阅读序）。
      operand 叶：internal sid（acc/idx·∈ internal_sids）→ 非 PARAM 不增 arity + cross-sample 一致性；
                  否则 within-sample 同 sid 复用同 slot（变量同一性）+ cross-sample 一致性门（坍缩允/拆分拒）。
      算子节点：全样同 opcode·同子数·递归子（DFS 前序）。
      STORE 节点：全样同位 STORE·cross-sample 一致性·递归 value 子（S3 ctrl/store-迭代骨架）。
      CTRL 节点：全样同 ATTR_CTRL_TAG·同子数·递归 [cond, body]（S3）。
      越界 kind（混合属性）→ _NoSkeleton。
    纯读：read_composes_attrs 只读·无 _new_node/_edge/record_composes_attr（守幂等不重 build）。
    internal_sids/cross_internal_sid 默认 None/空 = 直线算子树（无 ctrl/store·零行为变·bit-identical）。
    """
    if depth > _MAX_DISCOVER_DEPTH:
        raise _NoSkeleton()
    t0 = nodes[0]
    if t0 in state.seen:
        return   # DAG 共享：已计此子树 PARAM（镜像 build _seen·loop1 严格树不触发）
    state.seen.add(t0)
    all_attrs = [read_composes_attrs(backend, n) for n in nodes]
    a0 = all_attrs[0]
    _internal = internal_sids if internal_sids is not None else set()
    _cross_internal = cross_internal_sid or {}

    if _is_immediate_leaf(a0):
        # 立即数叶：全样须同类·同值→固定位 / 异值→PARAM 槽（DFS 阅读序）
        for a in all_attrs[1:]:
            if not _is_immediate_leaf(a):
                raise _NoSkeleton()
        vals = [a[ATTR_IMMEDIATE] for a in all_attrs]
        if len(set(vals)) != 1:
            state.param_idx += 1   # PARAM 槽（镜像 build·DFS 阅读序）
        return

    if _is_operand_leaf(a0):
        # operand 叶：internal sid（acc/idx）→ 非 PARAM 不增 arity / 否则 PARAM（变量同一性·镜像 build OPERAND 分支）
        for a in all_attrs[1:]:
            if not _is_operand_leaf(a):
                raise _NoSkeleton()
        s0 = a0[ATTR_OPERAND][0]
        if s0 in _internal:
            # S3: internal LOAD（acc/idx 引用）·非 PARAM·cross-sample 一致性校验（镜像 build OPERAND-internal 分支）
            for i in range(1, k):
                si = all_attrs[i][ATTR_OPERAND][0]
                expected = _cross_internal.get(i, {}).get(s0)
                if expected is None or si != expected:
                    raise _NoSkeleton()   # sample_i 引用非对应 internal sid → 结构错位
            return
        slot = state.sample0_sid_to_slot.get(s0)
        if slot is None:
            slot = state.param_idx
            state.param_idx += 1
            state.sample0_sid_to_slot[s0] = slot
        for i in range(1, k):
            si = all_attrs[i][ATTR_OPERAND][0]
            slot_map = state.cross_slot_to_sid.setdefault(i, {})
            prev = slot_map.get(slot)
            if prev is not None and prev != si:
                raise _NoSkeleton()   # sample_i 拆 sample0 变量 → 非实例（镜像 build 一致性门）
            slot_map[slot] = si
        return

    if _is_operator_node(a0):
        # 算子节点：全样同 opcode·同子数·递归子（DFS 前序）
        opcode = a0[ATTR_OPERATOR][0]
        for a in all_attrs[1:]:
            if not _is_operator_node(a) or a[ATTR_OPERATOR][0] != opcode:
                raise _NoSkeleton()
        child_lists = [children_of_all[i].get(nodes[i], []) for i in range(k)]
        cc0 = len(child_lists[0])
        for cl in child_lists[1:]:
            if len(cl) != cc0:
                raise _NoSkeleton()
        for slot in range(cc0):
            child_nodes = tuple(cl[slot] for cl in child_lists)
            _probe_walk(backend, child_nodes, children_of_all, k, state, depth + 1,
                        internal_sids=internal_sids, cross_internal_sid=cross_internal_sid)
        return

    if _is_concept_leaf(a0):
        # 件2·语言 token 概念叶 drift 镜像（S3 钥匙①·全参化无固定位·只更 state.param_idx 不写盘）。
        # drift 防线：probe_arity==discover_skeleton.arity（全参化 arity=distinct sample0 token ref 数·DAG seen :425
        # 共享同槽·同 build）。cross-sample 全放开（无门控·语言泛化牙·镜像 build）。testprobe_arity_matches_discover_skeleton 守。
        for a in all_attrs[1:]:
            if not _is_concept_leaf(a):
                raise _NoSkeleton()
        state.param_idx += 1
        return

    if _is_store_node(a0):
        # S3 ctrl/store-迭代骨架 drift 镜像：STORE 节点·cross-sample 一致性·递归 value 子（不增 arity）
        for a in all_attrs[1:]:
            if not _is_store_node(a):
                raise _NoSkeleton()
        s0 = a0[ATTR_STORE_TARGET][0]
        if s0 not in _internal:
            raise _NoSkeleton()   # 镜像 build（STORE 目标须 internal）
        for i in range(1, k):
            si = all_attrs[i][ATTR_STORE_TARGET][0]
            expected = _cross_internal.get(i, {}).get(s0)
            if expected is None or si != expected:
                raise _NoSkeleton()
        child_lists = [children_of_all[i].get(nodes[i], []) for i in range(k)]
        cc0 = len(child_lists[0])
        for cl in child_lists[1:]:
            if len(cl) != cc0:
                raise _NoSkeleton()
        for slot in range(cc0):
            child_nodes = tuple(cl[slot] for cl in child_lists)
            _probe_walk(backend, child_nodes, children_of_all, k, state, depth + 1,
                        internal_sids=internal_sids, cross_internal_sid=cross_internal_sid)
        return

    if _is_ctrl_node(a0):
        # S3 ctrl/store-迭代骨架 drift 镜像：CTRL 节点·全样同 tag·递归 [cond, body]（不增 arity）
        ctrl_tag = a0[ATTR_CTRL_TAG][0]
        for a in all_attrs[1:]:
            if not _is_ctrl_node(a) or a[ATTR_CTRL_TAG][0] != ctrl_tag:
                raise _NoSkeleton()
        child_lists = [children_of_all[i].get(nodes[i], []) for i in range(k)]
        cc0 = len(child_lists[0])
        for cl in child_lists[1:]:
            if len(cl) != cc0:
                raise _NoSkeleton()
        for slot in range(cc0):
            child_nodes = tuple(cl[slot] for cl in child_lists)
            _probe_walk(backend, child_nodes, children_of_all, k, state, depth + 1,
                        internal_sids=internal_sids, cross_internal_sid=cross_internal_sid)
        return

    raise _NoSkeleton()   # 越界 kind（混合属性/未知）→ 无共性骨架


def probe_arity(backend, samples: list[ConceptRef]) -> "int | None":
    """只读结构对齐探针（Half B·§八.7②）：返多样本共性骨架的 arity | None（异构/越界/混合/<K）。

    供 arity-in-name 幂等/路由 pre-check（auto_discover / formal_train）——须先知 arity 才能定 _shape_name(sig,arity)
    做幂等/路由判定·避免"先 build 再查=重 build orphan"（守幂等不重 build）。**纯读零写盘**（read_composes_attrs +
    read_composes_tree·无 _new_node/_edge/record_composes_attr）。

    **不变量（drift 防线）**：`probe_arity(samples) == discover_skeleton(samples,...).arity`（两函数同 DFS + 同槽规则
    + 同一致性门·faithful twin）·testprobe_arity_matches_discover 全语料族断言（drift detector·防 twin 漂移）。

    **S3 ctrl/store-迭代骨架支持（§三-bis）**：预扫 sample0 internal STORE 目标 sid + 跨样本对应 → _probe_walk
    传 internal_sids 区分 internal LOAD（acc/idx·非 PARAM）与 PARAM（lambda 参数）。直线算子树（无 STORE）→
    internal_sids 空 → _probe_walk internal 分支不触发 → bit-identical 零行为变。

    返 int（PARAM 槽数）| None（<MIN_DISCOVER_SAMPLES / 异构 opcode·子数 / 病态混合 / operand cross-sample
    拆分冲突 / internal sid 数异 / 深度超界）。
    """
    if len(samples) < MIN_DISCOVER_SAMPLES:
        return None
    for s in samples:
        assert_int(s[0], s[1], _where="probe_arity.sample")
        # drift 镜像 discover_skeleton：样本必须是算子 root（OPCODE_NOP/OPCODE_*·design§二·件2 防 root 误判）。
        if ATTR_OPERATOR not in read_composes_attrs(backend, s):
            return None
    graph = ConceptGraph(backend)
    k = len(samples)
    children_of_all = [graph.read_composes_tree(s)[0] for s in samples]
    # S3: 预扫 internal sids + 跨样本对应（ctrl/store-迭代骨架·直线算子树→空集零行为变）
    internal_sids = _collect_internal_store_sids(backend, graph, samples[0])
    cross_internal_sid: dict[int, dict[int, int]] = {}
    if internal_sids:
        cross_internal_sid = _build_cross_internal_sid(backend, graph, samples, internal_sids)  # type: ignore[assignment]
        if cross_internal_sid is None:
            return None   # internal sid 数异 → 不同 ctrl/store 结构
    state = _ProbeState()
    try:
        _probe_walk(backend, tuple(samples), children_of_all, k, state, depth=0,
                    internal_sids=set(internal_sids),
                    cross_internal_sid=cross_internal_sid)
    except _NoSkeleton:
        return None
    return state.param_idx


def _operand_arity_hint(graph: ConceptGraph, root: ConceptRef) -> int:
    """per-sample operand-arity 提示 = distinct operand sid 数（**grouping 用**·非最终 arity·Task #476）。

    立即数样本（无 OPERAND 叶）→ 0（arity 由 cross-sample 立即数值定）/ square(`x*x`·两 OPERAND 同 sid) → 1
    / mul(`a*b`·两 OPERAND 异 sid) → 2。**用途**：(shape_signature, hint) 分组·分离同形异 operand 结构
    （square vs mul 同 shape_signature 但 hint 异）·解 within-run 混合组 probe_arity cross-sample 门 None。

    纯读 L5（read_composes_tree·零写）·确定（distinct sid 集合基数·序无关 bit-identical）。
    **非最终 arity**：立即数样本 hint=0 但 cross-sample arity 可 0..N（`5*5`/`6*6`→2·`5*5`/`5*5`→0）·
    operand 样本 hint==arity（operand 必 PARAM·跨样本异变量故必参化）。hint 仅分组键·最终 arity 仍 probe_arity 定。
    """
    _children, _op, operand_of, _imm, _st = graph.read_composes_tree(root)
    return len(set(operand_of.values()))


# ---- S3 第二刀 Interp2：抽象级 LCA 聚类（同 shape_sig 组内按 IS_A LCA 类分桶·§四/§八 line 175） ----


def _collect_concept_leaf_tokens(backend, graph: ConceptGraph,
                                 root: ConceptRef) -> list[ConceptRef]:
    """DFS 前序收集 root 子树的 CONCEPT_LEAF token（slot 序·= build CONCEPT_LEAF 首遇序·纯读 L5）。

    S3 第二刀 Interp2 抽象聚类用：每 sample 的 CONCEPT_LEAF token 按 DFS 阅读序收集·与 _SkeletonBuilder.build
    CONCEPT_LEAF 分支的 _concept_slot_idx 同序（slot p 对齐）。空树/无 CONCEPT_LEAF（arith 立即数/operand 树）→ []。

    纯语言场景（NOP SEQ + token 叶·扁平）：DFS = children 序·token 叶 = direct children。
    纯读（read_composes_attrs + read_composes_tree·零写）·确定（DFS 前序 + children 按 (order_index,NodeRef) 排）。
    """
    children_of = graph.read_composes_tree(root)[0]
    tokens: list[ConceptRef] = []
    visited: set[ConceptRef] = set()

    def _dfs(node: ConceptRef) -> None:
        if node in visited:
            return
        visited.add(node)
        attrs = read_composes_attrs(backend, node)
        if _is_concept_leaf(attrs):
            tokens.append(node)
            return   # 叶·无子
        for child in children_of.get(node, []):
            _dfs(child)

    _dfs(root)
    return tokens


def _collect_slot_lcas(backend, graph: ConceptGraph,
                       skeleton_ref: ConceptRef) -> tuple[ConceptRef | None, ...]:
    """skeleton 子树 DFS 前序 → PARAM slot 序的 ATTR_SLOT_ROLE（abstract_sig 重建·纯读 L5）。

    load_discovered_operators 用：从已载 skeleton 重建 abstract_sig（BUILD 端 _shape_name(sig,arity,abstract_sig)
    的第三参）·修 B6 Bug 1（LOAD 名缺 abstract_sig 致同 (sig,arity) 异 abstract_sig 双算子撞同 name →
    op_by_name 字面覆盖 → 验证用错 skeleton）。

    children_of 已按 (order_index,NodeRef) 排（read_composes_tree:241）→ DFS 前序 == build _concept_slot_idx
    首遇序（纯语言两序一致·:379）。PARAM slot 谓词 ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs
    （镜像 test_stage12:607 test helper·防御未来 NODE_CONCEPT 算子节点误进）·逐 slot 读 ATTR_SLOT_ROLE
    （None 若无·build:381-385 仅非 None LCA 写）。

    纯算术 skeleton：PARAM slots 全 operand·全无 ATTR_SLOT_ROLE → 全 None → _normalize_abstract_sig → ()
    （名同今·bit-identical）。纯语言类级 skeleton：PARAM slots 全 CONCEPT·DFS 序 = _concept_slot_idx
    → parts == BUILD abstract_sig → 名 == BUILD 名。纯语言 None-LCA slot：parts[p]=None 与 BUILD 一致。

    诚实边界：混合 skeleton（CONCEPT+operand/immediate 同 sample 树）两序分离 → parts 长度 ≠ BUILD
    abstract_sig 长度 → 名对齐失效（orphan 重 build·非 corrupt·op_by_name 查不到→verify 防御跳过）。
    LOAD 侧不可修：CONCEPT slot 与 operand/immediate-PARAM slot 在 ATTR_SLOT_ROLE 缺失时同构（都仅
    ATTR_OPERAND·build:376/310/328）·谓词无法区分。**caller 须守单样本单模态**——当前 formal_train 的
    _run_arith / _run_lang 分流 + shape_signature（语言 NOP+无属性叶 vs 算术 NOP+OPCODE+IMM/OPERAND 叶）
    上游隔离·生产路径不产混合 skeleton（非 (sig,hint) 路由本身守：hint=distinct operand sid·纯语言与
    CONCEPT+immediate 混合 hint 都=0·不能单靠 hint 分离）。直调 auto_discover_operators 的 caller 须自守
    （:766-769 已标）。纯读零写·确定（DFS 前序 + visited 防重复入队·bit-identical）。
    """
    children_of = graph.read_composes_tree(skeleton_ref)[0]
    parts: list[ConceptRef | None] = []
    visited: set[ConceptRef] = set()

    def _dfs(node: ConceptRef) -> None:
        if node in visited:
            return
        visited.add(node)
        attrs = read_composes_attrs(backend, node)
        if ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs:
            # PARAM slot leaf（CONCEPT 或 operand·build:310/328/376 写 ATTR_OPERAND=make_variable(slot)·
            # 无子·下方 for-loop 自然不递归·镜像 test_stage12 _read_slot_roles 不 early-return）。
            slot_role = attrs.get(ATTR_SLOT_ROLE)
            parts.append((slot_role[0], slot_role[1]) if slot_role is not None else None)
        for child in children_of.get(node, []):
            _dfs(child)

    _dfs(skeleton_ref)
    return tuple(parts)


def _collect_cue_sig(backend, graph: ConceptGraph,
                     skeleton_ref: ConceptRef) -> tuple[ConceptRef | None, ...]:
    """skeleton 子树 DFS 前序 → PARAM slot 序的 ATTR_CUE_SIG（cue_sig 重建·纯读 L5·镜像 _collect_slot_lcas）。

    load_discovered_operators 用：从已载 skeleton 重建 cue_sig（BUILD 端 _shape_name(sig,arity,abstract_sig,cue_sig)
    第四参）·修 cue_sig 版 B6 Bug 1（LOAD 名缺 cue_sig 致同 (sig,arity,abstract_sig) 异 cue 双算子撞同 name →
    op_by_name 字面覆盖 → 验证用错 skeleton）。

    **镜像 _collect_slot_lcas（逐字范式）**：children_of DFS 前序·PARAM slot 叶（ATTR_OPERAND in attrs and
    ATTR_OPERATOR not in attrs）逐 slot 读 ATTR_CUE_SIG（None 若无·build CONCEPT_LEAF 分支仅 cue 拆簇的拆位非 None
    写）。全 None（无 cue 拆 / arith / 裸 NL）→ _normalize_abstract_sig 归一 () → 名同今（bit-identical）。

    诚实边界（同 _collect_slot_lcas）：混合 skeleton（CONCEPT+operand/immediate）两序分离 → parts 长度 ≠ BUILD
    cue_sig 长度 → 名对齐失效（orphan 重 build·非 corrupt·caller 须守单样本单模态·formal_train _run_arith/_run_lang
    分流 + shape_signature 上游隔离已守）。纯读零写·确定（DFS 前序 + visited 防重复入队·bit-identical）。
    """
    children_of = graph.read_composes_tree(skeleton_ref)[0]
    parts: list[ConceptRef | None] = []
    visited: set[ConceptRef] = set()

    def _dfs(node: ConceptRef) -> None:
        if node in visited:
            return
        visited.add(node)
        attrs = read_composes_attrs(backend, node)
        if ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs:
            # PARAM slot leaf（CONCEPT 或 operand·build CONCEPT_LEAF 写 ATTR_OPERAND=make_variable(slot)）·读 ATTR_CUE_SIG
            # （cue 拆簇拆位·build 仅 cue_sig 非 None 写·absence=非 cue 位 None·镜像 _collect_slot_lcas 读 ATTR_SLOT_ROLE）。
            cue = attrs.get(ATTR_CUE_SIG)
            parts.append((cue[0], cue[1]) if cue is not None else None)
        for child in children_of.get(node, []):
            _dfs(child)

    _dfs(skeleton_ref)
    return tuple(parts)


def _aligns_to_skeleton(backend, graph: ConceptGraph,
                        root: ConceptRef, skeleton_ref: ConceptRef) -> bool:
    """纯读：root 的 CONCEPT_LEAF token 序是否对齐 skeleton 的 ATTR_SLOT_ROLE + ATTR_CUE_SIG。

    Bug C2 修法（_skel_by_sig 键扩展）：single-dim shape_signature 键对语言坍缩为长度
    （实测 held-out sig 全 _LEAF_SIG=-1 哨兵·formal_train.py:1781 单值 first-wins 误绑）·
    同 shape 异 cue/异 LCA 的 skeleton 误绑·INSTANTIATES 不 fire 或 fire 到错 skeleton·
    floor measured 永假。本 helper 补 cue+LCA 维 per-root 对齐检查。

    · C4 长度 pre-check：root_tokens / slot_lcas / cue_sig 长度须一致·否则 False（silent veto·
      混合 skeleton 防御·caller 须守单样本单模态·镜像 _collect_slot_lcas:956-963 诚实边界）。
    · C2 逐 slot 独立（非 all-or-nothing）：
        - cue_sig[slot] 非 None → root_tokens[slot] 须 == cue_sig[slot]（cue token 精确匹配·
          破 是/使 误绑·镜像 _align_walk:1980-1983）。
        - slot_lcas[slot] 非 None → graph.is_a_descendant_of(root_tokens[slot], slot_lcas[slot])
          须 True（reflexive-transitive·c==slot_lca or slot_lca in ancestors(c)·破同类异 LCA 误绑·
          镜像 _align_walk:1965-1970）。
    · 全 None（arith/裸 NL/CUE_CLUSTER_MODE OFF 无 cue 写/_cluster_by_lca 未触发无 LCA 写）
      → 恒 True → 退化 first-match（与现状 shape-only setdefault first-wins 等价·bit-identical）。

    C1 ancestor_map 走 graph.is_a_descendant_of 的 _ancestor_map_cache（per-space lazy build·
    invalidate_ancestor_map per-item post-observe pre-generate 清·floor/维度桥 cache 稳定）·
    **非 per-call build_isa_ancestor_map**（n=300 floor 869s 基线上 per-call O(V+E) Tarjan 不可接受）。

    反 theater（O2）：纯读零写·cue token 来源是 held-out observed input（observe.py:235）·
    D:11 由训练侧 tally→promote 建（floor_measure.py 不变量锁）·helper 不自证。
    镜像 recognize_operators._align_walk concept 分支但简化（无 param binding/op_confidence·
    _skel_by_item 仅须 first-match）。确定（DFS 前序+visited·bit-identical）。
    """
    # root tokens = direct COMPOSES children（按 order_index·扁平语言 = token 叶序）。
    # ★ 不用 _collect_concept_leaf_tokens：它要求叶 ATTR_OPERAND（skeleton builder 标记）·
    # 但 floor orchestrator held-out root __disc_lang_* pre-build 只 ensure + COMPOSES edge
    # 无 ATTR_OPERAND（formal_train.py:1710-1717）→ _is_concept_leaf 漏收 → root_tokens=[] →
    # 长度错位 C4 silent veto → FC9 回归。direct children 对 skeleton 与 held-out root 都 work
    # （都建 COMPOSES edge·扁平语言 DFS = direct children）。
    _children_of = graph.read_composes_tree(root)[0]
    root_tokens = list(_children_of.get(root, []))
    slot_lcas = _collect_slot_lcas(backend, graph, skeleton_ref)
    cue_sig = _collect_cue_sig(backend, graph, skeleton_ref)
    # C4：长度 pre-check（混合 skeleton 防御·silent veto）。
    if len(root_tokens) != len(slot_lcas) or len(root_tokens) != len(cue_sig):
        return False
    # C2：逐 slot 独立（cue 只在有 ATTR_CUE_SIG 的 slot·LCA 只在有 ATTR_SLOT_ROLE 的 slot）。
    for tok, lca, cue in zip(root_tokens, slot_lcas, cue_sig):
        if cue is not None and tok != cue:
            return False   # cue slot token 不匹配（是 vs 使 区分·核心修）
        if lca is not None and not graph.is_a_descendant_of(tok, lca):
            return False   # 非 IS_A descendant（类级抽象约束·破同类异 LCA 误绑）
    return True


def _cluster_by_lca(backend, graph: ConceptGraph, roots: list[ConceptRef],
                    ancestor_map: dict[ConceptRef, set[ConceptRef]]
                    ) -> list[tuple[list[ConceptRef], list[ConceptRef | None]]]:
    """同 shape_sig 组内样本 → 按 IS_A LCA 类增量聚类 → list[(cluster_roots, slot_lcas)]。

    S3 第二刀 Interp2 真抽象对撞核心：同形（shape_signature 同）语言样本按 PARAM slot 的 IS_A LCA 上卷类分桶。
    "猫追老鼠/狗追兔子"（slot LCA 动物）≠"石头砸墙/砖砸地"（slot LCA 非生物）→ 产异簇→异骨架（破 D2 弱化门
    全 PARAM collapse）。slot_lcas[p] = 簇内 slot p 全 token 的 set_lca（ConceptRef|None·None=无共同祖先但同 token）。

    **算法（增量·确定·bit-identical）**：
      1. 收集每 sample 的 CONCEPT_LEAF token 序（DFS·_collect_concept_leaf_tokens）。首样本空（arith）→ 单簇
         slot_lcas=None（caller 走当前路径·bit-identical·LCA 聚类仅语言生效）。
      2. sort roots by NodeRef（聚类序确定·bit-identical·同 closure.py:90 / load_discovered:708 范式）。
      3. 增量聚类：每 sample 首入可 join 簇（can_join：每 slot tentative token 集 set_lca 非 None）·否则起新簇。
      4. 每簇 slot_lcas[p] = set_lca(簇内 slot p 全 token·ancestor_map)。

    **can_join（解 pairwise-drift·修改点 A）**：维护 cluster.tokens[p] 集合（非 slot_lca 作后代查）·
      tentative = cluster.tokens[p] + [sample.toks[p]]·set_lca(tentative) 非 None → joinable。set_lca 用
      abstract_closure(t)={t}∪anc[t]（含自身）·解 pairwise reduce LCA(LCA(t1,t2)=r12,t3) 把 r12 当后代的 drift bug。

    无 CONCEPT_LEAF 样本（arith）→ 单簇 slot_lcas=None（caller 走当前路径）·守 bit-identical。
    返 list[(cluster_roots, slot_lcas)]（slot_lcas=None 仅 arith 首样本空场景·否则 list[ConceptRef|None]）。
    """
    sample_tokens = [_collect_concept_leaf_tokens(backend, graph, r) for r in roots]
    # 首 sample 无 CONCEPT_LEAF（arith）→ LCA 聚类不适用 → 单簇 slot_lcas=None（caller 走当前路径·bit-identical）
    if not sample_tokens or not sample_tokens[0]:
        return [(list(roots), None)]
    arity_concept = len(sample_tokens[0])   # CONCEPT_LEAF 槽数（DFS 阅读序）
    # sort roots by NodeRef（聚类序确定·bit-identical·同 sample_tokens 索引对齐）
    order = sorted(range(len(roots)), key=lambda i: roots[i])
    # 增量聚类：cluster = {roots, tokens_per_slot（list[list[ConceptRef]]·每 slot 已聚 token 集）}
    clusters: list[dict] = []
    for i in order:
        toks = sample_tokens[i]
        # 防御：同 shape_sig 组内 CONCEPT_LEAF 槽数应一致（结构同构）·不一致跳过（理论不触发·shape_sig 已守同形）
        if len(toks) != arity_concept:
            continue
        placed = False
        for cluster in clusters:
            joinable = True
            for p in range(arity_concept):
                tentative = cluster["tokens_per_slot"][p] + [toks[p]]
                if set_lca(tentative, ancestor_map) is None:
                    joinable = False   # slot p 无共同抽象类 → 不 joinable
                    break
            if joinable:
                cluster["roots"].append(roots[i])
                for p in range(arity_concept):
                    cluster["tokens_per_slot"][p].append(toks[p])
                placed = True
                break
        if not placed:
            clusters.append({
                "roots": [roots[i]],
                "tokens_per_slot": [[toks[p]] for p in range(arity_concept)],
            })
    # 每簇 slot_lcas[p] = set_lca(簇内 slot p 全 token·ancestor_map)
    result: list[tuple[list[ConceptRef], list[ConceptRef | None]]] = []
    for cluster in clusters:
        slot_lcas: list[ConceptRef | None] = []
        for p in range(arity_concept):
            slot_lcas.append(set_lca(cluster["tokens_per_slot"][p], ancestor_map))
        result.append((cluster["roots"], slot_lcas))
    return result


def _cluster_by_cue(backend, graph: ConceptGraph, c_roots: list[ConceptRef]
                    ) -> list[tuple[list[ConceptRef], tuple[ConceptRef | None, ...]]]:
    """同 (sig,hint,LCA) 簇内样本 → 按 sustainable-split cue 位子聚类 → list[(sub_roots, cue_sig)]。

    §十八 condition 6a cue 子聚类（route_samples + auto_discover 双层·镜像 _cluster_by_lca 在两层）：
    同形同 LCA 但异闭类 cue（[苹果,是,甜] vs [糖,使,甜]·slot LCA 同·唯 cue 区分）的样本·按 cue 位 token 拆簇
    → 各产独立骨架（破 cue 坍缩·REALIZES 可异标 IS_A/CAUSES·§十八 condition 6 复合键）。

    **sustainable split**（exposure-driven·无 frozenset·反 theater 命门·D6 #2）：某位按 token 划分 c_roots 得 ≥2 组
    **每组各 ≥ MIN_DISCOVER_SAMPLES** → 该位是 cue 位（闭类 cue 重复→可持续拆·如 是×4/使×4 各≥K）·
    内容词（苹果/猫 各 1 次→每组 1<K）不可持续拆→不拆（**天然区分闭类 vs 开类·无须词表**）。首个 sustainable-split
    位（最低 slot·确定）拆·单拆（多 cue 位 defer·incremental）。

    **cue_sig**（进 _shape_name 名·sentinel 隔 abstract_sig 避碰撞）：per-position tuple[ConceptRef|None]·
    仅拆位 = cue token ConceptRef·非拆位 None（→ flatten (0,0) 占位·跨子簇一致·名差异仅来自拆位 cue token）。
    无 sustainable 拆 → 返 [(c_roots, ())]（cue_sig=() → _shape_name 不加 payload → **名同今 bit-identical**）。

    **gate OFF** → 返 [(c_roots, ())]（不拆·bit-identical·镜像 condition 3 dormant 范式）。无 CONCEPT_LEAF（arith）/
    spine 异长 / 总样本 < 2K（拆须每子簇≥K）→ 不拆。
    确定性：sort roots by NodeRef + 子簇按 cue token ConceptRef 排序（镜像 _cluster_by_lca·PYTHONHASHSEED=0·bit-identical）。

    **反 theater**：cue 身份仅区分骨架发现期分桶（route+auto_discover 名键）·关系 label 走外源 oracle（label_realizes·
    非读 cue·非 cue 路由 relation）。cue token 来自 concept_index.ensure（按 surface 幂等去重→同 token 同 ref·
    exposure-driven·非 frozenset）。
    """
    if not getattr(gates, "CUE_CLUSTER_MODE", False):
        return [(list(c_roots), ())]   # gate OFF → 不拆·cue_sig=()·bit-identical
    import os as _os, sys as _sys
    _dbg = _os.environ.get("ZERO_AI_CUE_DEBUG")
    if _dbg:
        print(f"[CUE_DEBUG] entry c_roots={len(c_roots)} (2K={2*MIN_DISCOVER_SAMPLES})", file=_sys.stderr)
    if len(c_roots) < 2 * MIN_DISCOVER_SAMPLES:
        return [(list(c_roots), ())]   # 拆须每子簇 ≥K → 总 ≥2K·不足不拆（诚实·cue 分离需 ≥K 同 cue）
    seqs = [_collect_concept_leaf_tokens(backend, graph, r) for r in c_roots]
    if not seqs or not seqs[0]:
        return [(list(c_roots), ())]   # 无 CONCEPT_LEAF（arith 立即数/operand 树）→ 不拆
    length = len(seqs[0])
    if any(len(s) != length for s in seqs):
        return [(list(c_roots), ())]   # spine 异长（discover_skeleton 同构门已守·防御 malformed）→ 不拆
    pairs = sorted(zip(c_roots, seqs), key=lambda p: p[0])   # sort by root（确定·bit-identical·镜像 _cluster_by_lca）
    for slot in range(length):
        part: dict[ConceptRef, list[ConceptRef]] = {}
        for root, seq in pairs:
            part.setdefault(seq[slot], []).append(root)
        groups = sorted(part.items(), key=lambda kv: kv[0])   # 子簇按 cue token ConceptRef 排序（确定）
        # 修法 A（doc/重来_语料聚簇规模 §15·2 审 APPROVE）：判据从 all(groups≥K)（含单例·实文内容词单例多→永不触发）
        # 改 big=[≥K 组]·≥2 组各≥K 拆。docstring 形式判据"≥2 组各≥K"与语义例子（是×4/使×4 + 苹果/猫 各 1 不阻断）
        # 自相矛盾·code 取字面严读·本修法对齐语义意图（big-groups）·是 docstring+code 对齐非单方面修 bug（审2 点1）。
        # 单例（内容词各 1 次<K·非 cue）合并 cue_sig=() 余簇：余簇走既有 uncued discover 路径（同修法前 fallback·
        # 非新发现）·floor_measure:65-67 skip 余簇 cue_sig=() → 对 floor 指标零贡献（结构性反 theater·审2 点3）。
        big = [(tok, rl) for tok, rl in groups if len(rl) >= MIN_DISCOVER_SAMPLES]
        if len(big) >= 2:
            if _dbg:
                print(f"[CUE_DEBUG] ★ SPLIT found at slot={slot} c_roots={len(c_roots)} "
                      f"big={[len(rl) for _t, rl in big]} singletons={len(groups) - len(big)}", file=_sys.stderr)
            remainder: list[ConceptRef] = []
            for _tok, rl in groups:
                if len(rl) < MIN_DISCOVER_SAMPLES:
                    remainder.extend(rl)
            result: list[tuple[list[ConceptRef], tuple[ConceptRef | None, ...]]] = []
            for token, root_list in big:
                cue_sig: list[ConceptRef | None] = [None] * length
                cue_sig[slot] = token   # 仅拆位 = cue token·非拆位 None（flatten (0,0) 占位·跨子簇一致）
                result.append((root_list, tuple(cue_sig)))
            if remainder:
                # 单例余簇 cue_sig=()（全 None·无 cue 位）·与不拆退化解共享 _shape_name(sig,arity,asig,())·幂等门去重
                result.append((remainder, tuple([None] * length)))
            return result
    return [(list(c_roots), ())]   # 无 sustainable-split 位（全内容词 / 全同 cue）→ 不拆


def route_samples_for_discovery(
        backend, graph: ConceptGraph, roots: list[ConceptRef], *,
        existing_keys: set[tuple[tuple[int, ...], int, tuple, tuple]],
        existing_sigs: set[tuple[int, ...]],
        space_id: int,
        stats: DiscoveryRouteStats | None = None,
        ) -> tuple[list[ConceptRef], list[ConceptRef]]:
    """同 shape 程序根 → 按 (sig,hint) 分组 → 每组 LCA 聚类 → 按簇 abstract_sig 路由 discover/recognize。

    B6 Bug 2+3 修（聚类前置·doc/重来_待办与defer总清单_2026-07-06.md §B6）：
      · **Bug 2（existing_keys 缺 abstract_sig）**：路由键 (sig,arity) → (sig,arity,abstract_sig)。
        resume 时载入动物类骨架后·同 (sig,arity) 异 abstract_sig 新样本（非生物类）不再误判"已载"全送
        recognize→不命中静默丢·改为按簇 abstract_sig 独立路由→新抽象类本轮发现（解跨 run 覆盖渐失）。
      · **Bug 3（per-cluster held-out）**：held-out 切分从整组 grp[:K]/grp[K:] 改为**按簇** c[:K]/c[K:]。
        混合簇组（动物+非生物同 shape）不再因前 K 横跨簇致每簇 <K→auto_discover 不发现（cluster-blind 病根）。

    聚类与 auto_discover_operators 内部聚类**同逻辑**（_cluster_by_lca·build_isa_ancestor_map·
    has_isa=False/arith 无 CONCEPT_LEAF → 单簇 None·:733-734）·路由级聚类决定 discover/recognize 归属·
    auto_discover 级聚类（在 discover_roots 子集上）决定实际产骨架·两层聚类幂等一致（同 ancestor_map+同算法）。

    **路由规则（每簇 c_roots·slot_lcas·算 abstract_sig=normalize(tuple(slot_lcas) 或 ())·再 cue 子聚类）**：
      · cue 子聚类（§十八 condition 6a-3·gate-gated·镜像 auto_discover:1346）：每 LCA 簇再 _cluster_by_cue →
        list[(sub_roots, cue_sig)]·cue_key=normalize(cue_sig)（gate OFF→()·gate ON sustainable-split→拆位 cue token）。
      · (sig,arity,abstract_sig,cue_key) ∈ existing_keys → 子簇全 held-out 识别（载入算子可识别新输入=跨 run 泛化）。
      · 否则 arity 非 None 且子簇 ≥K → 子簇首 K 发现·余识别（held-out·§八.3·真泛化非循环 theater）。
      · 否则（arity None / 子簇 <K 未载）→ sig ∈ existing_sigs 则识别候选·否则弃（<K 无载入·诚实不伪造）。

    返 (discover_roots, recognize_roots)·discover_roots 喂 auto_discover_operators（WRITE 注册）·
    recognize_roots 喂 recognize_operators（held-out READ 消费·§八.3）。

    **bit-identical 守卫**（核心）：arith / 裸 NL 首样本无 CONCEPT_LEAF → _cluster_by_lca 返单簇 slot_lcas=None
    → abstract_sig=() → 路由键 (sig,arity,()) 与现状 (sig,arity) 等价（arith/裸 NL abstract_sig 恒 ()·
    existing_keys 载入算子 abstract_sig 亦 ()·match·行为不变）。仅 has_isa 且多 LCA 类组（动物/非生物同形）
    行为变=bug 修·既有 single-cluster 测零翻。**condition 6a-3 cue 维**：cue 子聚类 gate OFF（default）→
    _cluster_by_cue 恒返 [(c_roots,())]→cue_key=()→路由键 (sig,arity,abstract_sig,()) 第4维恒 ()·与 3-tuple 路由
    行为逐字等价（existing_keys 载入算子 cue_sig 亦 ()·match·bit-identical）。gate ON → cue 拆→异键→预期非
    bit-identical（是/使 独立发现·condition 6 真行为变·floor P5 held-out 区分需要）。

    铁律：纯整数（sig/arity/abstract_sig 全 int·assert_int 守）/ 确定性（聚类 sort by NodeRef + tuple 分组·
    bit-identical）/ 幂等（纯读聚类·零写盘·两次调同果）/ 单向依赖（formal_train → process.structure_discover
    同层 L5·无环）。**诚实边界（两层聚类语义·对抗审 2026-07-06）**：路由级聚类决定 discover/recognize 归属·
    auto_discover 级聚类（在 discover_roots 上）决定实际产骨架——两层幂等一致（同 ancestor_map + 同 _cluster_by_lca·
    sort by NodeRef·sample0 主导确定）。**词例级 fallback 簇**（auto_discover has_class_cluster 时追加 (all_roots,None)·
    abstract_sig=()）**非路由级独立簇**——其样本已归入各 class 簇路由·词例级是 discover 侧产物兜底骨架·路由不重复计。
    故路由判定（"新簇→discover 首 K"）≠ auto_discover 实际产出（discover_roots 上可能额外产词例级 fallback）·词例级
    经 auto_discover lookup 幂等门（concept_index.lookup·_index 重建守）skip 不 orphan——此幂等门依赖 ConceptIndex._index
    跨 run 重建（task #475·formal_train 生产路径已守·直调 caller 须自保 _index 已填·auto_discover docstring :900-903
    自承 _index 假阴性场景）。若 (sig,arity,()) ∈ existing_keys（词例级已载）则落单/未成簇样本经 abstract_sig=()
    分支识别匹配。stable≠correct（结构身份路由·非语义·#479 墙）。
    """
    assert_int(space_id, _where="route_samples_for_discovery.space_id")
    for r in roots:
        assert_int(r[0], r[1], _where="route_samples_for_discovery.root")
    if stats is not None:
        stats.calls += 1
        stats.input_roots += len(roots)
    # 1. 按 (shape_signature, operand_arity_hint) 分组（同 auto_discover:829·序列2 另半·同类样本分组）。
    groups: dict[tuple[tuple[int, ...], int], list[ConceptRef]] = {}
    for root in roots:
        sig = tuple(shape_signature(graph, root))
        if not sig:
            continue   # 无 COMPOSES 树（非程序根·如语言 struct_ref）→ 不路由
        hint = _operand_arity_hint(graph, root)
        groups.setdefault((sig, hint), []).append(root)
    if stats is not None:
        stats.shape_groups += len(groups)
    # 2. 建 ancestor_map 一次（同 auto_discover:843·run-scoped·空则 has_isa=False 跳过聚类守 bit-identical）。
    ancestor_map = build_isa_ancestor_map(backend, space_id=space_id)
    has_isa = bool(ancestor_map)
    discover_roots: list[ConceptRef] = []
    recognize_roots: list[ConceptRef] = []
    for (sig, _hint), grp in groups.items():
        grp_sorted = sorted(grp)   # NodeRef tuple 确定序（= 创建序 = 语料序·bit-identical）
        # Half B：probe arity（纯读·≥K 才探·sig 组级探一次）→ 路由须 arity（同 _run_arith/_run_lang 原路径；
        # arity 是 shape 级·cue 拆不影响·审1 F4 sound）
        arity = (probe_arity(backend, grp_sorted)
                 if len(grp_sorted) >= MIN_DISCOVER_SAMPLES else None)
        # 修法 B（doc/重来_语料聚簇规模 §15·2 审 APPROVE）：cue-first——外层 _cluster_by_cue 直接拆 sig 组·内层再 LCA。
        # 先结构（功能词脚手架分桶）后语义（内容词 LCA 参数化）·解 LCA-then-cue 让内容词 LCA 打散句法脚手架之病。
        # gate OFF → _cluster_by_cue 返 [(grp_sorted,())]·内层 LCA 照原跑 → 退化逐字 bit-identical（审1 Q4）。
        cue_groups = _cluster_by_cue(backend, graph, grp_sorted)
        if stats is not None:
            stats.cue_clusters += len(cue_groups)
        for cue_roots, cue_sig in cue_groups:
            cue_sorted = sorted(cue_roots)
            cue_key = _normalize_abstract_sig(cue_sig)
            # 内层 LCA 聚类（在 cue 子簇内·同 auto_discover 两层镜像·幂等一致）·has_isa 且 ≥K 才聚类·否则单簇 None
            if has_isa and len(cue_sorted) >= MIN_DISCOVER_SAMPLES:
                clusters = _cluster_by_lca(backend, graph, cue_sorted, ancestor_map)
            else:
                clusters = [(list(cue_sorted), None)]
            # F3（审1）：兜底门变量重绑至 cue_sorted（原绑 grp_sorted·sig 组级外层）——否则 cue 子簇塌缩样本
            # 跨 cue 子簇混回 sig 组单簇·直接销毁 cue 拆分（silent bug·不报错）。
            if not any(len(c_roots) >= MIN_DISCOVER_SAMPLES for c_roots, _c_sig in clusters):
                clusters = [(list(cue_sorted), None)]
            if stats is not None:
                stats.lca_clusters += len(clusters)
            for c_roots, slot_lcas in clusters:
                abstract_sig = _normalize_abstract_sig(
                    tuple(slot_lcas) if slot_lcas is not None else ())
                sub_sorted = sorted(c_roots)
                # §十八 condition 6a-3：cue_sig 进路由键（闭命门1·resume 同 (sig,arity,asig) 异 cue 不共享键吞掉）。
                if arity is not None and (sig, arity, abstract_sig, cue_key) in existing_keys:
                    # (sig,arity,abstract_sig,cue_sig) 已载 → 全 held-out 识别（跨 run 泛化·守幂等不 re-discover）
                    recognize_roots.extend(sub_sorted)
                    if stats is not None:
                        stats.existing_key_clusters += 1
                elif arity is not None and len(sub_sorted) >= MIN_DISCOVER_SAMPLES:
                    # 新 cue 子簇 → 发现首 K·识别余（held-out·§八.3·识别须新输入非发现集→真泛化）
                    discover_roots.extend(sub_sorted[:MIN_DISCOVER_SAMPLES])
                    recognize_roots.extend(sub_sorted[MIN_DISCOVER_SAMPLES:])
                    if stats is not None:
                        stats.new_discovery_clusters += 1
                else:
                    # arity None（probe 不符/<K）或 子簇 <K 未载 → sig fallback（同原 _run_arith/_run_lang 路径）
                    if sig in existing_sigs:
                        recognize_roots.extend(sub_sorted)
                    if stats is not None:
                        stats.fallback_clusters += 1
                    # else 弃（<K 无载入·不发现不识别·诚实）
    if stats is not None:
        stats.discover_samples += len(discover_roots)
        stats.recognize_samples += len(recognize_roots)
        stats.dropped_samples += max(len(roots) - len(discover_roots) - len(recognize_roots), 0)
    return discover_roots, recognize_roots


def auto_discover_operators(program_roots: list[ConceptRef], *,
                            concept_index, edge_store, backend,
                            space_id: int, source: int) -> list[DiscoveredOperator]:
    """多样本 COMPOSES 程序根 → 按算子形状分组 → 同形≥K 抽骨架 + 注册为可复用算子（序列6-min 生产机制）。

    de-theater 序列1：discover_skeleton 的**生产 caller**（非仅 tests·§八.6）。模态无关——按
    shape_signature 分组·loop1 scope 自然过滤（operand/ctrl/store 树 / 非程序根 / 异构 → 不发现）。

    program_roots : COMPOSES 程序树根 ConceptRef 列表（caller 保独立根·如 formal_train 按 arith_source
                    内容哈希建·绕 observe 多程序撞 struct_ref 限制）。
    返 list[DiscoveredOperator]（**新发现**的·已注册过的同形 skip 不重列）。

    机制（§八.6）：
      1. 去重 roots（同程序只算一份·caller 内容哈希根已天然去重·此处防御·保序确定）。
      2. 按 shape_signature tuple 分组（同形程序聚一组·空签名=非程序根跳过）。
      3. 每组 ≥ MIN_DISCOVER_SAMPLES（Half B·§八.7②）：
         · probe arity（probe_arity·纯读不建盘）→ None（loop1 scope 不符/异构/operand 拆分）→ skip（诚实不发现）。
         · name = _shape_name(sig, arity)（**arity 进名**·解同形异 arity 碰撞·square(1)≠mul(2)）。
         · 幂等门：concept_index.lookup(name) 已有 ATTR_OPERATOR_DEF → skip **不 build**（同 (sig,arity) 已落·守幂等不重 build）。
         · discover_skeleton(组内 roots) → register_arith_operator(name, skeleton_ref, arity) → 注册为可 inline 复用算子。

    反 theater（§8.7 序列6="被复用注册"·诚实分层）：本函数让 discover_skeleton 有**真生产 caller**
      （formal_train 触发·de-theater 序列1"零 caller"）+ 注册名**可消费**（_try_inline_learned Call 路径
      inline+β-归约·L1.5·测试证明机制活：后续 `lambda: <name>(args)` 嫁接骨架 + vm_proof 复现）。
      **诚实边界**：生产训练 loop 当前**不引用** `__op_disc_*` 名（grep 零下游 caller·对抗审计核证）——故
      生产期"真读消费"须序列3（coverage_overlap 识别新输入命中已学骨架）/ 生成侧引用·非本步。即序列6-min
      = 序列1 零 caller 的 de-theater + 消费机制活（测试证）·**非"生产期已消费"全闭环**（那是序列3）。

    铁律：纯整数（shape/arity/name hash 全 int·assert_int 守）/ 确定性（Hasher 固定种子 + tuple 分组
      + lookup 幂等·bit-identical）/ 幂等（lookup 门 + register 幂等·跨 run 同语料同发现）/ 单向依赖
      （process→understanding.arith_observe 同层 L5·arith_observe 不 import process·无环）。
    诚实边界：operand 叶已支持（序列2·square=λx.x*x 可发现·变量同一性）/ ctrl/store 树→None（defer）/ Rice
      有限基底（继承 discover_skeleton）/ **同名碰撞已修（Half B·§八.7②·2026-07-03）**：原 shape_signature 叶统一
      _LEAF_SIG → square(arity1) 与 mul(arity2) 同 shape 同名 → 跨 run 碰撞静默 skip。**现 _shape_name(sig,arity)
      arity 进名** → 异名 + probe_arity 幂等 pre-check（守不重 build·避免 orphan）→ 跨 run 载 square 后 mul 仍可独立
      发现（见 formal_train route_samples_for_discovery 路由·键 (sig,arity,abstract_sig)·B6 Bug 2+3）/ 单 run 每 (sig,hint) 组只产一算子（首 K 样本模式·sample0 主导·同 Rice 小样本）/
      不判语义命名（__op_disc_{h63(sig,arity)} 确定性名·非"square"人名）/ **混类 shape 组已解（Task #476·序列2 另半）**：
      (shape_sig, operand_arity_hint) 分组·square(hint=1)/mul-operand(hint=2)/立即数(hint=0) 分离→各同质组独立发现
      （原 sig-only 合并异构组→probe None·within-run 混合 defer·**已修**）/ stable≠correct /
      **幂等门靠 `concept_index.lookup`（_index·run-scoped 内存）**：跨 run `load_run` 后 _index 不重建→**直调本函数**
      （绕 formal_train `existing_operators` 预过滤）于已载算子同形 roots 会 orphan 重 build（lookup 假阴性→进 discover）。
      生产路径 formal_train 经 `existing_operators` 参 + route_samples_for_discovery（键 (sig,arity,abstract_sig)·聚类前置）预过滤·discover_roots 不含载入簇→**不触发**。
      直调 caller（非 formal_train）须自行 `load_discovered_operators` 预过滤或保证 _index 已填（对抗审计 MED·已标注）。
    """
    assert_int(space_id, source, _where="auto_discover_operators")
    for r in program_roots:
        assert_int(r[0], r[1], _where="auto_discover_operators.root")
    graph = ConceptGraph(backend)
    # 1. 去重 roots（保序·确定性·同程序只算一份样本）
    seen_roots: set[ConceptRef] = set()
    unique_roots: list[ConceptRef] = []
    for r in program_roots:
        if r not in seen_roots:
            seen_roots.add(r)
            unique_roots.append(r)
    # 2. 按 (shape_signature, operand_arity_hint) 分组（Task #476·序列2 另半·同类样本分组）。
    #    同 shape 但异 operand 结构（square hint=1 vs mul hint=2）须分离·否则混合组 probe_arity
    #    cross-sample 门 None→不发现。hint= 立即数样本（0）聚一起（arity cross-sample 定）。
    groups: dict[tuple[tuple[int, ...], int], list[ConceptRef]] = {}
    for root in unique_roots:
        sig = tuple(shape_signature(graph, root))
        if not sig:
            continue   # 无 COMPOSES 树（非程序根·如语言 struct_ref）→ 不进发现
        hint = _operand_arity_hint(graph, root)
        groups.setdefault((sig, hint), []).append(root)
    # 3. 每组 ≥K → probe arity → Interp2 LCA 聚类 → arity+abstract_sig-in-name 幂等门 → discover_skeleton → register
    discovered: list[DiscoveredOperator] = []
    # S3 第二刀 Interp2：建 ancestor_map 一次（run-scoped cache·全组复用·空则跳过聚类守 bit-identical）。
    # bare NL（无 IS_A 边）→ ancestor_map 全空 → has_isa=False → 走当前路径（单组一骨架·bit-identical 零行为变）。
    ancestor_map = build_isa_ancestor_map(backend, space_id=space_id)
    has_isa = bool(ancestor_map)
    for (sig, _hint), roots in groups.items():
        if len(roots) < MIN_DISCOVER_SAMPLES:
            continue   # 不足 K 样本 → 不发现（K=MIN_DISCOVER_SAMPLES·元定义常量）
        # Half B：probe arity 先（纯读·不建盘·sig 组级探一次）→ 定 arity-in-name + 幂等 pre-check（守幂等不重 build·避免 orphan）。
        # arity 是 shape 级·cue 拆不影响（审1 F4 sound）。
        arity = probe_arity(backend, roots)
        if arity is None:
            continue   # loop1 scope 不符（异构/越界/operand 拆分冲突）→ 诚实不发现（且减 discover mid-build orphan）
        # 修法 B（doc/重来_语料聚簇规模 §15·2 审 APPROVE）：cue-first——外层 _cluster_by_cue 直接拆 sig 组·内层再 LCA。
        # 先结构（功能词脚手架分桶）后语义（内容词 LCA 参数化）·解 LCA-then-cue 让内容词 LCA 打散句法脚手架之病。
        # gate OFF → _cluster_by_cue 返 [(roots,())]·内层 LCA 照原跑 → 退化逐字 bit-identical（审1 Q4）。
        for cue_roots, cue_sig in _cluster_by_cue(backend, graph, roots):
            cue_sorted = sorted(cue_roots)
            if len(cue_sorted) < MIN_DISCOVER_SAMPLES:
                continue   # cue 拆后子簇 <K（防御·单例已并入余簇·理论不触发）
            # 内层 Interp2 抽象聚类：has_isa 时在 cue 子簇内按 IS_A LCA 类分桶（仅语言 CONCEPT_LEAF 生效·arith 首
            # sample 无 CONCEPT_LEAF → _cluster_by_lca 返单簇 None·走当前路径 bit-identical）。has_isa=False → 单簇 None。
            if has_isa:
                clusters = _cluster_by_lca(backend, graph, cue_sorted, ancestor_map)
                # 刀1 件3 多级共存：词例级骨架（abstract_sig=()·全 cue 子簇 roots 无 LCA 约束·PARAM 接受任何 token）
                # 与类级骨架（slot_lcas 非 None·IS_A LCA 上卷）共存·都存都有效名不撞。**存在有效类级簇（len≥K 且
                # slot_lcas 非 None）时**·追加词例级簇。
                # F2（审1）：词例级 fallback 从 sig 组级 (roots,None) 重定位至 cue 子簇级 (cue_sorted,None)——
                # 否则 sig 组级 fallback 被 _cluster_by_cue 再拆·命名碰撞/重复·与 route 层不可对齐。
                # gate OFF 时 cue_sorted=roots 全量·行为同原（bit-identical·审1 Q4）。
                has_class_cluster = any(
                    len(c_roots) >= MIN_DISCOVER_SAMPLES and c_sig is not None
                    for c_roots, c_sig in clusters)
                if has_class_cluster:
                    clusters.append((list(cue_sorted), None))
            else:
                clusters = [(list(cue_sorted), None)]
            # F3（审1）：兜底门变量重绑至 cue_sorted（原绑 roots·sig 组级外层）——否则 cue 子簇塌缩样本跨 cue 子簇混回
            # sig 组单簇·直接销毁 cue 拆分（silent bug·不报错）。
            if not any(len(c_roots) >= MIN_DISCOVER_SAMPLES for c_roots, _c_sig in clusters):
                clusters = [(list(cue_sorted), None)]
            for c_roots, slot_lcas in clusters:
                if len(c_roots) < MIN_DISCOVER_SAMPLES:
                    continue
                # abstract_sig = 簇 slot LCA 序（ConceptRef|None·DFS 阅读序）·_shape_name 经 _normalize_abstract_sig
                # 全 None 归一 () → 名同今（bit-identical）·仅真有 LCA ref 时异名（Interp2 真行为变）。cue-first 后
                # cue slot 全同 → slot_lcas[cue_slot]=cue token 本身进 abstract_sig（与 cue_sig 冗余·sentinel 隔不撞名·审1 F4）。
                abstract_sig = tuple(slot_lcas) if slot_lcas is not None else ()
                # cue_sig 已在外层 _cluster_by_cue 定（不再内层 cue 循环）·喂 _shape_name（名键）+ discover_skeleton（写 ATTR_CUE_SIG）。
                name = _shape_name(sig, arity, abstract_sig, cue_sig)
                # 幂等门：同 (sig,arity,abstract_sig,cue_sig) 已发现注册过 → skip **不 build**（跨 run/续训·守幂等）
                existing = concept_index.lookup(name, space_id)
                if existing is not None and ATTR_OPERATOR_DEF in read_composes_attrs(backend, existing):
                    continue   # 已注册（同 (sig,arity,abstract_sig,cue_sig) 骨架已落）·不重抽不重 build
                result = discover_skeleton(c_roots, concept_index=concept_index,
                                           edge_store=edge_store, backend=backend,
                                           space_id=space_id, source=source,
                                           skeleton_label=name,
                                           slot_lcas=slot_lcas, cue_sig=cue_sig)
                if result is None:
                    continue   # drift 防御（probe!=discover 理论不触发·drift detector 测守）→ 诚实不发现
                # arity=probe 之值（与 result.arity 等·invariant）·名/register/字段皆用同一 arity → 名↔ATTR_ARITY↔字段一致
                name_ref = register_arith_operator(backend, concept_index, name,
                                                   result.skeleton_ref, arity=arity)
                discovered.append(DiscoveredOperator(
                    name=name, skeleton_ref=result.skeleton_ref,
                    arity=arity, sample_count=len(c_roots), name_ref=name_ref,
                    forming_roots=tuple(c_roots)))
    return discovered


# ---- Phase D §十六-bis D.1：option-b oracle-pair-match REALIZES labeled bed ----


def _has_external_isa(edge_store, child: ConceptRef, parent: ConceptRef) -> bool:
    """外源 EDGE_IS_A(child→parent) 存在？（SOURCE_CONCEPTNET/SOURCE_CHINESE_KB·anti-self-proving·排 cue EPI_CUE）。

    Phase D §十六-bis D.1 oracle = boot 已种的外源 IS_A 边（ConceptNet/ChineseSemanticKB·独立 `_CUE_WORDS`）。
    **oracle 定 IS_A 方向（child→parent）·非读 Cue**·禁 pairs→文本渲染（Cue 泄漏陷阱）。
    """
    rows = edge_store.query_from(child[0], child[1], edge_type=EDGE_IS_A)
    for r in rows:
        if ((r.get("space_id_to"), r.get("local_id_to")) == parent
                and r.get("source") in (SOURCE_CONCEPTNET, SOURCE_CHINESE_KB)):
            return True
    return False


def _has_external_causes(edge_store, cause: ConceptRef, effect: ConceptRef) -> bool:
    """外源 EDGE_CAUSES(cause→effect) 存在？（SOURCE_CONCEPTNET·anti-self-proving·排 observe/cue 派生）。

    Phase D §十六-bis D.1 CAUSES oracle = boot 已种的外源 ConceptNet CAUSES 边（来源① 有向三元组
    cause Causes effect 照搬不反转·独立 `_CUE_WORDS`/observe）。**oracle 定 CAUSES 方向（cause→effect）·非读 使 cue**·
    守 condition-6 命门（使 cue→observe CAUSES→labeler 循环 = self-proving theater·option (b) oracle 直查避开）。

    **source == SOURCE_CONCEPTNET 是 CAUSES 唯一外源结构化 KB**（observe 走 raw.source≠CONCEPTNET·
    cue_pairs EPI_CUE·teacher source=5 → 全排除）。镜像 _has_external_isa source 过滤；IS_A 含 SOURCE_CHINESE_KB
    因中文 KB 亦供 IS_A·CAUSES 仅 ConceptNet 故单源（2026-07-15 corpus build：causes 14k←ConceptNet·无中文 KB CAUSES）。
    """
    rows = edge_store.query_from(cause[0], cause[1], edge_type=EDGE_CAUSES)
    for r in rows:
        if ((r.get("space_id_to"), r.get("local_id_to")) == effect
                and r.get("source") == SOURCE_CONCEPTNET):
            return True
    return False


def label_realizes_is_a(discovered: list[DiscoveredOperator], *, graph: ConceptGraph,
                        edge_store, rel_primitives: dict, space_id: int) -> int:
    """option-b oracle-pair-match 标 REALIZES→__REL_SUBSET__（labeled bed·Phase D §十六-bis D.1）。

    skeleton **通用文本独立发现**（auto_discover_operators·不 partition 发现输入）+ iff skeleton 的 forming-sample
    token-pair（ordered·child 在 parent 前）**命中外源 EDGE_IS_A** → 写 REALIZES skeleton→`__REL_SUBSET__`。
    **oracle-pair-match 定 IS_A·非读 `_CUE_WORDS`**·**禁 pairs→文本渲染**（Cue 经渲染文本泄漏 = Phase A审2
    REJECT 命门隐蔽复现·本设计走 option (b) oracle 直查避开）。

    **sound labeled bed**（同 boot IS_A）·学习 claim 严禁前置·验 floor Phase F·consumer Phase E。
    **self-gate**（REALIZES_MODE default OFF·bit-identical）·**幂等**（build_realizes_edge query_from skip）。
    load 重建（forming_roots=()）→ skip（REALIZES 边已在图·resume 不重标）。

    返建边数（0=gate OFF / 无 forming_roots / 无 oracle 命中）。
    """
    if not getattr(gates, "REALIZES_MODE", False):
        return 0   # self-gate·default OFF·bit-identical（基建·无 consumer 时 dormant）
    rel_subset_ref = rel_primitives.get(REL_SUBSET)
    if rel_subset_ref is None:
        return 0   # relation primitives 未 ensure（防御·observe 已 ensure 故正常不触）
    backend = graph._b   # ConceptGraph.backend（_collect_concept_leaf_tokens 读 composes_attr 用）
    n = 0
    for op in discovered:
        if not op.forming_roots:
            continue   # load 重建（forming_roots=()）→ skip（REALIZES 已在图）
        realize = False
        for root in op.forming_roots:
            tokens = _collect_concept_leaf_tokens(backend, graph, root)
            # ordered pairs：child a 在 parent b 前（IS_A child→parent 方向·oracle 定方向非 Cue·避无序对误匹配）
            for i, a in enumerate(tokens):
                if realize:
                    break
                for b in tokens[i + 1:]:
                    if _has_external_isa(edge_store, a, b):
                        realize = True
                        break
            if realize:
                break
        if realize:
            n += build_realizes_edge(edge_store, op.skeleton_ref, rel_subset_ref,
                                     space_id=space_id)
    return n


def label_realizes_causes(discovered: list[DiscoveredOperator], *, graph: ConceptGraph,
                          edge_store, rel_primitives: dict, space_id: int) -> int:
    """option-b oracle-pair-match 标 REALIZES→__REL_CAUSES__（labeled bed·Phase D §十六-bis D.1·镜像 label_realizes_is_a）。

    skeleton **通用文本独立发现**（auto_discover_operators·不 partition 发现输入）+ iff skeleton 的 forming-sample
    token-pair（ordered·cause 在 effect 前）**命中外源 EDGE_CAUSES** → 写 REALIZES skeleton→`__REL_CAUSES__`。
    **oracle-pair-match 定 CAUSES·非读 使 cue**（condition-6 命门：使 cue→observe CAUSES→labeler = self-proving
    theater·本设计走 option (b) oracle 直查 SOURCE_CONCEPTNET 避开·同 label_realizes_is_a 排 cue 的 sound 路径）。

    与 label_realizes_is_a 的差异**仅**：oracle 边类型（EDGE_CAUSES vs EDGE_IS_A）+ rel 目标（__REL_CAUSES__
    vs __REL_SUBSET__）+ ordered 方向语义（cause→effect vs child→parent）。余同（self-gate / forming_roots
    迭代 / ordered-pair / build_realizes_edge 幂等 / load 重建 forming_roots=() skip）。

    **sound labeled bed**（同 boot CAUSES）·学习 claim 严禁前置·验 floor Phase F·consumer Phase E。
    **self-gate**（REALIZES_MODE default OFF·与 IS_A 同 REALIZES labeled-bed pass·bit-identical）·**幂等**
    （build_realizes_edge query_from skip）。load 重建（forming_roots=()）→ skip（REALIZES 已在图·resume 不重标）。

    返建边数（0=gate OFF / 无 forming_roots / 无 oracle 命中）。
    """
    if not getattr(gates, "REALIZES_MODE", False):
        return 0   # self-gate（同 label_realizes_is_a·REALIZES labeled-bed pass default OFF·bit-identical）
    rel_causes_ref = rel_primitives.get(REL_CAUSES)
    if rel_causes_ref is None:
        return 0   # relation primitives 未 ensure（防御·observe 已 ensure 故正常不触）
    backend = graph._b   # ConceptGraph.backend（_collect_concept_leaf_tokens 读 composes_attr 用）
    n = 0
    for op in discovered:
        if not op.forming_roots:
            continue   # load 重建（forming_roots=()）→ skip（REALIZES 已在图）
        realize = False
        for root in op.forming_roots:
            tokens = _collect_concept_leaf_tokens(backend, graph, root)
            # ordered pairs：cause a 在 effect b 前（CAUSES cause→effect 方向·oracle 定方向非 使 cue·避循环）
            for i, a in enumerate(tokens):
                if realize:
                    break
                for b in tokens[i + 1:]:
                    if _has_external_causes(edge_store, a, b):
                        realize = True
                        break
            if realize:
                break
        if realize:
            n += build_realizes_edge(edge_store, op.skeleton_ref, rel_causes_ref,
                                     space_id=space_id)
    return n


# ---- 对应泛化 v2：结构反推 tally（cue-blind 对齐 REALIZES-R-skeleton cue slot → tally W·审1C3/审2条件1 三路分离）----


def _collect_cue_slot_candidates(skeleton_ref: ConceptRef, input_ref: ConceptRef,
                                 backend, arity: int,
                                 ancestor_map: dict[ConceptRef, set[ConceptRef]] | None
                                 ) -> list[ConceptRef]:
    """cue-blind 对齐 input→skeleton·捕 cue slot 落位词 W（结构反推证据·审2 条件1 三路分离·独立于 recognize）。

    复用 _align_extract + cue_capture 参（cue_capture 非 None → _align_walk cue 分支捕 inp 不拒·非精确匹配轨）。
    返 W ConceptRef list（cue slot 落位词·**任意词非闭类**·新词引发 落 使-skeleton cue slot 亦捕·W 可不在 oracle）。
    空 = 不对齐（shape/opcode/abstract_sig 异）或 skeleton 无 cue slot（ATTR_CUE_SIG·cue_sig=() 不写→不捕）。

    **三路分离**（审1C3/审2条件1）：本函数 tally 轨（cue_capture）≠ recognize 精确匹配轨（cue_capture=None·6a-3 闭命门2
    不动·held-out verify sound）≠ cue_type_of readback 轨（promote 果）。tally 不产 Recognition·不污染 recognize 下游。
    **纯读对齐**（无 side-effect·record_structure_match 在 caller tally_cue_slot_matches·非此处）。
    """
    graph = ConceptGraph(backend)
    cue_capture: list[ConceptRef] = []
    aligned = _align_extract(graph, skeleton_ref, input_ref, backend, arity,
                             ancestor_map, cue_capture=cue_capture)
    if aligned is None:
        return []
    return cue_capture


def tally_cue_slot_matches(input_roots: list[ConceptRef], *,
                           discovered_operators: list[DiscoveredOperator],
                           graph: ConceptGraph, edge_store, backend, space_id: int,
                           rel_primitives: dict[int, ConceptRef],
                           stats: StructureTallyStats | None = None) -> int:
    """结构反推 tally 编排（对应泛化 v2·审1C3/审2条件1+2·三路分离 + SHADOW 创建）。

    对每个 **REALIZES-R-skeleton**（EDGE_REALIZES skeleton→rel_ref 存在）+ cue-blind 对齐 input → 捕 cue slot W →
    record_structure_match(W,R,input_root)（distinct tally sample；生产为 recognition-routed input_root·append-only 幂等去重）→ 首次 new（该 input_root
    for (W,R)）→ record_emergent_relation_signal_shadow 建 D:11 SHADOW 行（generator 关后唯一创建者·审2 条件2·
    record_emergent_relation_signal_shadow 自身 query_from 幂等防同 (W,R) 多 sample 重复建边）。

    **反推锚=结构**（REALIZES-R-skeleton·oracle 确认·内容对命中 ConceptNet·非词在 oracle）：W 可新词（不在 oracle/frozenset）
    → 真泛化·syntactic bootstrapping。**非自证**：R 来自 REALIZES oracle（source==CONCEPTNET 滤·非 cue）·W 是观察·
    提升反馈在 REALIZES source filter 断（详见 doc/重来_对应泛化_结构反推_学全 §四）。

    **gate**：caller formal_train 守 ORACLE_PROMOTE_MODE（OFF 不调 → 零 tally → bit-identical）。本函数自身不读 gate
    （caller 守·防双 gate 误配）。
    **性能**（审2 条件5）：input_roots 去重 + shape 过滤候选（仅 REALIZES-skeleton·子集）+ seen_inputs 每 input 一次。
    n=20 可控·大 n 可 preload（defer）。
    返新建 SHADOW 数（0=gate OFF caller 不调 / 无 REALIZES-skeleton / 无 cue slot 命中）。
    """
    assert_int(space_id, _where="tally_cue_slot_matches.space_id")
    if stats is not None:
        stats.calls += 1
    # rel_ref → rel_kind 反查（promote _structure_match_ok 用 rel_kind·REALIZES 边存 rel_ref）
    rel_ref_to_kind: dict[ConceptRef, int] = {ref: kind for kind, ref in rel_primitives.items()}
    ancestor_map = build_isa_ancestor_map(backend, space_id=space_id)
    # REALIZES-skeleton 索引（by shape_signature）+ 其 REALIZES 的 (rel_kind, rel_ref) 列表
    by_shape: dict[tuple[int, ...], list[tuple[ConceptRef, int, list[tuple[int, ConceptRef]]]]] = {}
    for op in discovered_operators:
        rows = edge_store.query_from(op.skeleton_ref[0], op.skeleton_ref[1], edge_type=EDGE_REALIZES)
        rels: list[tuple[int, ConceptRef]] = []
        for r in rows:
            rel_ref = (r.get("space_id_to"), r.get("local_id_to"))
            rel_kind = rel_ref_to_kind.get(rel_ref)
            if rel_kind is not None:
                rels.append((rel_kind, rel_ref))
        if not rels:
            continue   # skeleton 不 REALIZES 任何 R → 不参与 tally（非"已学结构"basis）
        sig = tuple(shape_signature(graph, op.skeleton_ref))
        if not sig:
            continue   # 空签名（非程序骨架）→ 跳过
        by_shape.setdefault(sig, []).append((op.skeleton_ref, op.arity, rels))
    if stats is not None:
        stats.realizes_skeletons += sum(len(v) for v in by_shape.values())
    if not by_shape:
        return 0   # 无 REALIZES-skeleton → 无 tally（REALIZES_MODE OFF 或无 oracle 命中）
    n_new_shadow = 0
    seen: set[ConceptRef] = set()
    for root in input_roots:
        if root in seen:
            continue   # 去重（同 input_root 一次·保序确定·镜像 recognize_operators seen_inputs）
        seen.add(root)
        if stats is not None:
            stats.input_roots += 1
        sig = tuple(shape_signature(graph, root))
        candidates = by_shape.get(sig)
        if not candidates:
            continue   # 无同形 REALIZES-skeleton → 跳过
        if stats is not None:
            stats.shape_matched_roots += 1
        root_aligned = False
        for skeleton_ref, arity, rels in candidates:
            ws = _collect_cue_slot_candidates(skeleton_ref, root, backend, arity, ancestor_map)
            if not ws:
                continue   # 不对齐 / 无 cue slot → 不 tally
            if stats is not None:
                stats.candidate_alignments += 1
                if not root_aligned:
                    stats.aligned_roots += 1
                    root_aligned = True
            for w in ws:
                for rel_kind, rel_ref in rels:
                    new = record_structure_match(backend, space_id=space_id,
                                                 word_ref=w, rel_kind=rel_kind, sample_root=root)
                    if new:
                        if stats is not None:
                            stats.distinct_matches_added += 1
                        # 首次该 (W,R,input_root) → 建 D:11 SHADOW 行（generator 关后唯一创建者·
                        # record_emergent_relation_signal_shadow query_from 幂等·同 (W,R) 多 sample 不重复建边）
                        added = record_emergent_relation_signal_shadow(
                            edge_store, w, rel_ref, space_id=space_id)
                        n_new_shadow += added
                        if stats is not None:
                            stats.shadow_edges_added += added
    return n_new_shadow


# ---- 序列7：跨 run READ 重建（load 已 dump 的发现算子·§八序列7·2026-07-03）----


def load_discovered_operators(backend, *, space_id: int) -> list[DiscoveredOperator]:
    """从已载图重建发现算子列表（序列7 跨 run READ·纯读 L5·§八序列7）。

    扫 composes_attr ATTR_OPERATOR_DEF 行（= name 节点·int_a/int_b = struct_ref）→ 读 name 节点
    ATTR_ARITY 得 arity → 读 struct_ref ATTR_ORIGIN 过滤 ==ORIGIN_DISCOVERED（排除 observer BUILT /
    教师 INJECTED 手注册·observer 每 run 重建不持久化识别）→ 重派生 name=_shape_name(sig, arity,
    abstract_sig)（Half B·arity 进名·S3 第二刀 abstract_sig 进名·经 _collect_slot_lcas 从 skeleton
    ATTR_SLOT_ROLE 重建·修 B6 Bug 1）→ DiscoveredOperator(sample_count=0·载后未知·诊断字段识别不用)。

    跨 run 闭环（run N dump → run N+1 load_run → recognize 全新 held-out 命中载入算子·§8.7）。与
    auto_discover_operators（本 run WRITE 产新算子）互补：formal_train resume load 后调它取已载算子·
    合并本 run 新发现 → 识别 held-out。纯读零写·幂等（重复调同果）。

    返 list[DiscoveredOperator]（按 skeleton_ref NodeRef 升序·bit-identical·backend.select 序不保证）。
    空 = 无已载发现算子（fresh run / 无算术语料 / 未 dump composes_attr）。

    铁律：纯整数（shape/arity/hash 全 int·assert_int 守）/ 确定性（NodeRef sort + Hasher 固定种子名·
      bit-identical）/ 单向依赖（process L5 纯读·不调 vm_proof L7·不 import training）。
    诚实边界：ATTR_ARITY 缺失→arity=0（向后兼容防御）/ ATTR_ORIGIN 非 DISCOVERED→跳过（不载入 observer
      手注册）/ shape_signature 空（struct_ref 非程序根）→跳过 / ConceptIndex._index 不重建（载入算子
      可 recognize/verify 不可 inline·_try_inline_learned 的 concept_index.lookup 返 None·Half B defer）/
      abstract_sig 经 _collect_slot_lcas 重建须 caller 守单样本单模态（formal_train _run_arith/_run_lang
      分流 + shape_signature 上游隔离已守·混合 skeleton LOAD 侧不可修·见 _collect_slot_lcas docstring）。
    """
    assert_int(space_id, _where="load_discovered_operators.space_id")
    graph = ConceptGraph(backend)
    rows = backend.select(COMPOSES_ATTR_TABLE, where={
        "space_id": space_id, "kind": ATTR_OPERATOR_DEF})
    ops: list[DiscoveredOperator] = []
    for r in rows:
        name_ref = (r["space_id"], r["local_id"])
        struct_ref = (r["int_a"], r["int_b"])
        # 仅发现算子（struct_ref root ATTR_ORIGIN==DISCOVERED·排除 observer BUILT / 教师 INJECTED）
        if read_composes_attrs(backend, struct_ref).get(
                ATTR_ORIGIN, (0, 0))[0] != ORIGIN_DISCOVERED:
            continue
        arity = read_composes_attrs(backend, name_ref).get(ATTR_ARITY, (0, 0))[0]
        sig = tuple(shape_signature(graph, struct_ref))
        if not sig:
            continue   # struct_ref 非程序根（无 COMPOSES 树）→ 跳过
        abstract_sig = _collect_slot_lcas(backend, graph, struct_ref)
        cue_sig = _collect_cue_sig(backend, graph, struct_ref)   # §十八 condition 6a-2：cue_sig 重建（镜像 abstract_sig·修 cue_sig 版 B6 Bug 1·全 None→()→名同今 bit-identical）
        ops.append(DiscoveredOperator(
            name=_shape_name(sig, arity, abstract_sig, cue_sig), skeleton_ref=struct_ref,
            arity=arity, sample_count=0, name_ref=name_ref))
    ops.sort(key=lambda op: op.skeleton_ref)   # NodeRef 升序（bit-identical·backend.select 序不保证）
    return ops


# ---- 序列3-min：识别消费（新输入命中已学骨架·READ 消费·§八序列3·2026-07-03）----


class Recognition(NamedTuple):
    """recognize_operators 产物（序列3-min·§八序列3）。

    生产期 READ 消费的证据：新输入（held-out·非发现样本集）命中已学骨架 → 抽 PARAM 绑定。
    """
    input_root: ConceptRef                       # 被识别的新输入程序根（held-out·非发现样本）
    operator_name: str                           # 命中的已注册算子名 __op_disc_{h63}
    param_values: tuple[tuple[int, int], ...]    # 骨架 PARAM 槽值（slot 序·= make_variable index 序·Rational (num,den)）
    arity: int                                   # 命中算子的 arity（= len(param_values)）
    # operand-input 识别（探针值执行比对·补序列2 operand READ 闭环）：
    is_operand_input: bool = False               # True = input 含 OPERAND 叶（参数化如 λz:z*z）·param_values=派生探针值
    operand_binding: tuple[int, ...] = ()        # skeleton slot → input operand slot（诊断·展示结构映射·位置 j=-1=非 operand 位）
    input_probe_values: tuple[tuple[int, int], ...] = ()  # input operand slot-序探针值（连续 0..input_arity-1·含未用 slot·_verify 直接执行 input·消除反演洞·对抗审计 F1）
    # 件5 语言识别（concept_binding·S3 第二片·钥匙①发现线 READ 消费）：
    is_concept_input: bool = False              # True = input 含无属性叶（语言 token·_is_concept_leaf）·param_values 占位 (0,0) 非值
    concept_binding: tuple = ()                  # skeleton slot → input token ConceptRef（语言识别产物·slot 序·变量同一性：同槽须同 ref）


def recognize_operators(input_roots: list[ConceptRef], *,
                        discovered_operators: list[DiscoveredOperator],
                        backend, space_id: int) -> list[Recognition]:
    """新输入 COMPOSES 程序 → 命中已学骨架 → 抽 PARAM 绑定（序列3 READ 消费·§八序列3·"新样本命中已学骨架"）。

    生产期 READ（§8.7 让结构"被读"）：发现的骨架被**真读**消费——read_composes_tree(skeleton) 读骨架结构
    （哪些位 PARAM·哪些位 fixed）→ shape_signature 同形过滤 → _align_walk 并行 DFS 前序对齐 →
    PARAM 位抽新输入立即数值 → 绑定。模态无关（按 shape + 算子/立即数结构·loop1 scope）。

    input_roots         : 被识别的新输入程序根（**须 held-out**·非发现样本集→识别新输入=真泛化·非循环 theater）。
    discovered_operators: 已注册的发现算子（auto_discover_operators 产·含 skeleton_ref/arity）。
    返 list[Recognition]（命中的·未命中/无候选→不列·保 input_roots 首见序·**刀2 件6 多解析**：同 input_root
    可返 ≥2 Recognition·组内按 rate 降序·不同抽象级/不同算子不同置信）。

    机制：
      1. 按 shape_signature 索引已发现算子（同形候选·BFS 序快速过滤）。
      2. 每个新输入（去重保序）：shape_signature → 候选算子 → _align_extract 精确对齐（PARAM 抽值·fixed 值须等·
         opcode/子数须符·loop1 越界 kind→不识别）→ **§8.7-洗 洗净**：同形候选收集全部对齐（非首匹配）·
         读 op_confidence → 滤 tested-never-verified（sn==0 验过皆败=非泛化）→ **刀2 返全列**按 rate 降序（稳定·
         同率保 BFS 序·bit-identical）·cold-start(None)→rate 0 给机会但末位。**多解析**：同形异构骨架（PARAM/IMM
         形状不分·square/mul 同 shape·类级/词例级双骨架）都 align 时全返·非 aligning[0] 单选。

    反 theater（§8.7 序列3="让结构被读"·诚实）：本函数让发现的 struct_ref+COMPOSES 骨架在生产期被**真读**
      （read_composes_tree + 结构对齐 + 值抽取·非仅写入死货）+ 识别**新输入**（held-out·非发现集→真泛化·非循环）。
      识别的 PARAM 绑定可被 vm_proof 验（骨架绑参执行 == 新输入执行·caller/test 验·本模块 L5 不调 L7 vm_proof
      守单向依赖）。比 a4_align.coverage_overlap（纯 shape 同形→coverage=1000）更精：固定位值须等（shape 同但
      固定立即数异→不识别·如骨架 ADD[PARAM,IMM3] 不认 ADD[IMM7,IMM4]）。

    铁律：纯整数（shape/opcode/param 全 int·assert_int 守）/ 确定性（shape_signature BFS 确定 + DFS 前序对齐·
      bit-identical）/ 单向依赖（process L5·不调 vm_proof L7·验证留 caller·不写边无 source 需求）/ 幂等（纯读·重复
      调同果）/ 不写死（对齐 = 结构等价比对·无规则硬编码）。
    诚实边界：ctrl/store-bearing 输入→不识别（loop1 defer）/ **operand 输入（λz:z*z）已支持**（operand-input 识别·
      探针值执行比对·见 _align_walk operand 分支 + _align_extract 探针派生）/ fixed 位值须等（非纯形状指纹）/ 探针 arity
      上限 _MAX_PROBE_ARITY（超→不识别 defer）/ **§8.7-洗 洗净**：置信度滤只在坏算子（PARAM 序错/编译发散/shape
      异配）触发·正确算子 held-out vm_proof 必过（构造性必然）·mul/square 不可区分（置信度正交于变量同一性判别器）/
      Rice 有限基底（继承 discover_skeleton）/ stable≠correct / 跨 run 识别（dump/load）= follow-up。
    """
    assert_int(space_id, _where="recognize_operators")
    for r in input_roots:
        assert_int(r[0], r[1], _where="recognize_operators.input_root")
    graph = ConceptGraph(backend)
    # S3 第二刀 Interp2：建 ancestor_map 一次（run-scoped·同 auto_discover·_align_walk 抽象匹配用）。
    # bare NL（无 IS_A）→ ancestor_map 空 → ATTR_SLOT_ROLE 骨架无 abstract 命中（守诚实·非 theater）·
    # 无 ATTR_SLOT_ROLE 骨架走原 ref 等价路径（bit-identical）。
    ancestor_map = build_isa_ancestor_map(backend, space_id=space_id)
    # 1. 按 shape_signature 索引已发现算子（同形候选·空签名=非程序骨架跳过）
    by_shape: dict[tuple[int, ...], list[DiscoveredOperator]] = {}
    for op in discovered_operators:
        sig = tuple(shape_signature(graph, op.skeleton_ref))
        if sig:
            by_shape.setdefault(sig, []).append(op)
    if not by_shape:
        return []   # 无已学骨架→无可识别（caller 须先 auto_discover）
    # 2. 每个新输入：去重保序 → shape 过滤候选 → 精确对齐 → 抽 PARAM 绑定
    recognitions: list[Recognition] = []
    seen_inputs: set[ConceptRef] = set()
    for root in input_roots:
        if root in seen_inputs:
            continue   # 去重（同根只识一次·保序确定）
        seen_inputs.add(root)
        sig = tuple(shape_signature(graph, root))
        candidates = by_shape.get(sig)
        if not candidates:
            continue   # 无同形已学骨架→不识别
        # §8.7-洗 洗净循环反馈半闭环：同形候选收集全部对齐（非首匹配）→ 读 op_confidence →
        # 滤 tested-never-verified（sn==0·验过皆败=非泛化）→ 按 rate 降序择优（verified 优先·cold-start 末位给机会）。
        aligning: list[tuple[int, DiscoveredOperator, list, bool, tuple, tuple, bool, tuple]] = []
        for op in candidates:
            aligned = _align_extract(graph, op.skeleton_ref, root, backend, op.arity, ancestor_map)
            if aligned is None:
                continue
            params, is_operand, op_binding, input_pv, is_concept, concept_bind = aligned
            conf = read_op_confidence(backend, op.name_ref)   # (sn,tn,strength)|None·纯读 L5→L0
            if conf is not None and conf[0] == 0:
                continue   # 洗净：tested-never-verified（有行 sn==0=验过皆败）滤除·cold-start(None)给机会不滤
            rate = ((conf[0] * _OP_CONF_RATE_SCALE // max(conf[1], 1))
                    if conf is not None else 0)   # cold-start→0（给机会但排序末位·verified 率高优先）
            aligning.append((rate, op, params, is_operand, op_binding, input_pv,
                             is_concept, concept_bind))
        if not aligning:
            continue   # 全滤/无对齐→不识别（诚实·非首匹配兜底）
        # 稳定排序·同率保候选 BFS 序（=发现序·bit-identical）·reverse=True 不破稳定性（equal 保输入序）
        aligning.sort(key=lambda x: x[0], reverse=True)
        # 刀2 件6 多解析（doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀2）：返**全列**按 rate 降序（非 aligning[0]
        # 单选）·一句多解（不同抽象级/不同算子·不同置信）。shape_signature 把 PARAM 与 IMM 都标 _LEAF_SIG
        # （:179）→ 同形异构骨架共享候选池（square λx.x*x 与 mul λa.λb.a*b 同 shape·两都 align 7*7；狐狸追鸡
        # 类级 ATTR_SLOT_ROLE 抽象命中 + 词例级 loose 兜底·两都 align）。aligning 已 stable sort·**循环 append 全部**
        # → 每 Recognition 同 input_root 异 op/params。返列按 input_roots 首 见序·组内 rate 降序。
        # **洗净滤 sn==0 仍在**（:1001-1002 循环内·aligning 收集前）→ 返多路不含 tested-never-verified 坏算子。
        # **反 theater**：同 input_root 返 ≥2 Recognition = 真两级描述（类级抽象 + 词例级 loose / square + mul）
        # ·非伪造。**bit-identical**：单候选场景 aligning 长度 1 → 循环 append 1 个 == aligning[0] 同果。
        # **防双计=caller 责任**（formal_train _verify_generalization/_discover_and_recognize_lang_structures·
        # summary recognized/verified 计 distinct input_root·op_confidence 半环 per-op 不双计）。
        for _rate, op, params, is_operand, op_binding, input_pv, is_concept, concept_bind in aligning:
            recognitions.append(Recognition(
                input_root=root, operator_name=op.name,
                param_values=tuple(params), arity=op.arity,
                is_operand_input=is_operand, operand_binding=op_binding,
                input_probe_values=input_pv,
                is_concept_input=is_concept, concept_binding=concept_bind))
    return recognitions


def _align_extract(graph: ConceptGraph, skeleton_ref: ConceptRef,
                   input_ref: ConceptRef, backend, arity: int,
                   ancestor_map: dict[ConceptRef, set[ConceptRef]] | None = None,
                   cue_capture: list[ConceptRef] | None = None):
    """并行 DFS 前序对齐 skeleton/input·PARAM 位抽 input 立即数值（value_binding）或 operand slot（operand_binding）。

    读骨架结构（PARAM/fixed/算子位）应用到 input 抽绑定 = READ 消费。返 `(params, is_operand_input, operand_binding, input_probe_values) | None`：
      - params : 骨架 PARAM 槽值（slot 序 0..arity-1·= make_variable index 序·与 inline arg_subst 契约一致）。
        immediate 输入→抽具体立即数值；operand 输入→**派生探针值**（slot j ← 探针[input_slot]·经 operand_binding）。
      - is_operand_input : input 含 OPERAND 叶（参数化·如 λz:z*z）→ True（探针验证路径）·否则 False（既有 immediate 路径）。
      - operand_binding : tuple·位置 j = skeleton slot j → input operand slot（operand 位）/ -1（immediate/fixed 位·诊断用）。
      - input_probe_values : input operand slot-序探针值元组（连续 0..input_arity-1·含未用 slot·_verify 直接执行 input·消除反演洞）。
        **关键**（对抗审计 F1）：连续含未用 slot 探针——input λp,q:q*q（p 未用·q=slot1）operand_binding={0:1,1:1}·input_arity=2·
        input_probe_values=((探针0),(探针1)) 连续·_verify 执行 input 绑 make_variable(0..1) 全到位无 KeyError（未用 slot never LOADed·值 harmless）。
    None = 结构/opcode/fixed 值/同槽值不等/operand 同槽 input slot 异（变量同一性）/探针 arity 超界 → 不识别。

    **slot 感知**（序列2）：skeleton 同 sid 槽位（变量同一性·如 square 两叶同槽）→ input 须同值（immediate·7*8 拒）/
    同 input operand slot（operand·a*b 拒·两异 operand 对齐同 skeleton slot）。binding 按 slot 索引非 DFS 追加。

    **S3 第二刀 Interp2 抽象匹配**（ancestor_map 透传 _align_walk concept 分支）：skeleton CONCEPT slot 有 ATTR_SLOT_ROLE
    （LCA ref r）→ input token 沿 IS_A 可达 r（r ∈ anc[input]∪{input}）→ 抽象命中（"狐狸"命中"动物"槽·词级零交集）。
    ancestor_map=None/空 → ATTR_SLOT_ROLE 骨架 abstract 不命中（守诚实）·无 ATTR_SLOT_ROLE 骨架走原 ref 等价（bit-identical）。
    """
    sk_children, _sk_op, _sk_operand, _sk_imm, sk_store_target_of = graph.read_composes_tree(skeleton_ref)
    in_children, _in_op, _in_operand, _in_imm, in_store_target_of = graph.read_composes_tree(input_ref)
    value_binding: dict[int, tuple[int, int]] = {}    # slot → (num,den)·immediate 输入位（既有路径）
    operand_binding: dict[int, int] = {}              # slot → input operand slot·operand 输入位（operand-input 识别）
    concept_binding: dict[int, ConceptRef] = {}       # slot → input token ConceptRef·语言 token 输入位（件5 concept-input 识别）
    # S3 ctrl/store-迭代骨架：skeleton + input internal STORE 目标 sid 集（区分 internal LOAD 与 PARAM·_align_walk 用）。
    # input internal sorted（变量同一性牙·审2 MED-2）：skeleton internal LOAD 须对齐 input 对应 internal sid（acc/idx·sorted-position
    # 对应）·非 PARAM（拒 Sigma(1,z,z) body=z=PARAM 误识为 Sigma=三角数——两路独立 vm_proof 兜底外·recognize 单函数亦拒）。
    sk_internal_sids = set(sk_store_target_of.values())
    in_internal_sids = tuple(sorted(set(in_store_target_of.values())))
    if len(in_internal_sids) != len(sk_internal_sids):
        return None   # 迭代 internal 变量数异 → 不同迭代结构（shape/STORE/CTRL 同形守已拒·此为越界防御）
    if not _align_walk(backend, sk_children, in_children, skeleton_ref, input_ref,
                       value_binding, operand_binding, concept_binding, depth=0,
                       ancestor_map=ancestor_map,
                       sk_internal_sids=sk_internal_sids,
                       in_internal_sids=in_internal_sids, sk_arity=arity,
                       cue_capture=cue_capture):
        return None
    if concept_binding:
        # 件5 concept-input 识别（语言 token·钥匙① READ 消费）：param_values 占位 (0,0)（token 非值·concept_binding 存真 ref）
        # 同槽变量同一性（"猫追猫"两猫同槽同 ref）_align_walk 已守。语言骨架全 PARAM（件2 全参化）·concept_binding 全槽。
        params = [value_binding.get(j, (0, 0)) for j in range(arity)]
        return (params, False, (), (), True,
                tuple(concept_binding.get(j, (0, 0)) for j in range(arity)))
    if operand_binding:
        # operand-input 识别：派生骨架探针 param_values（slot j ← 探针[input operand slot]·per logical variable）
        input_arity = max(operand_binding.values()) + 1
        if input_arity > _MAX_PROBE_ARITY:
            return None   # 探针覆盖上限→不识别 defer（现实 lambda arity≤6·元定义常量界）
        input_probe = {k: _PROBE_VALUES[k] for k in range(input_arity)}
        # input 探针连续元组（0..input_arity-1·含未用 slot 的探针值——未用 slot never LOADed 故值 harmless·
        # 消除 _verify 反演洞：input λp,q:q*q（p 未用·仅 q=slot1 出现）→ operand_binding={0:1,1:1}→input_arity=2
        # → input_probe_values=((探针0),(探针1)) 连续·execute 绑 make_variable(0..1) 全到位无 KeyError（对抗审计 F1））。
        input_probe_values = tuple(input_probe[k] for k in range(input_arity))
        params: list[tuple[int, int]] = []
        for j in range(arity):
            if j in operand_binding:
                params.append(input_probe[operand_binding[j]])   # operand 位：派生探针值（slot→input slot→探针）
            else:
                params.append(value_binding[j])                  # 混合 input：该位 immediate（具体值·fixed/抽值）
        return (params, True,
                tuple(operand_binding.get(j, -1) for j in range(arity)),
                input_probe_values, False, ())
    # immediate-input 识别（既有·bit-identical）：param_values = value_binding slot 序
    return [value_binding[i] for i in range(arity)], False, (), (), False, ()


def _align_walk(backend, sk_children: dict, in_children: dict,
                sk: ConceptRef, inp: ConceptRef,
                value_binding: dict[int, tuple[int, int]],
                operand_binding: dict[int, int],
                concept_binding: dict[int, ConceptRef], depth: int,
                ancestor_map: dict[ConceptRef, set[ConceptRef]] | None = None,
                subtree_binding: dict[int, ConceptRef] | None = None,
                sk_internal_sids: set[int] | None = None,
                in_internal_sids: tuple[int, ...] | None = None,
                sk_arity: int | None = None,
                cue_capture: list[ConceptRef] | None = None) -> bool:
    """递归 DFS 前序对齐单节点对·slot 感知 PARAM 绑定（值/operand 两路）/ fixed 值等 / 算子 opcode 等+同子数递归。返成功否。

    children_of 已按 (order_index,NodeRef) 排（read_composes_tree）→ enumerate/zip 复现阅读序（确定性）。
    深度闸 _MAX_DISCOVER_DEPTH 防病态深（同 discover_skeleton / read_composes_tree）。

    **slot 感知·两路绑定**（序列2 + operand-input 识别）：
      - skeleton PARAM 槽（ATTR_OPERAND·sid=make_variable(slot)）遇 input 立即数叶 → value_binding[slot]=值·同槽值须等
        （变量同一性·square 两叶同槽·immediate 7*8 第二叶 8≠首叶 7→拒）。
      - skeleton PARAM 槽遇 input **OPERAND 叶** → operand_binding[slot]=input operand slot·同 skeleton slot 须对齐同 input
        operand slot（变量同一性牙·square 两叶同槽·operand `a*b` 两异 operand 对齐同 slot→拒·mul 不拒坍缩允两 slot→同 input slot）。
      - 一位一绑定（slot 不可同时 value/operand-bound·混合 input 各位独立）。

    **S3 ctrl/store-迭代骨架（§三-bis）**：sk_internal_sids = skeleton STORE 目标 sid 集（_align_extract 派生）。
      - skeleton OPERAND 叶 sid ∈ sk_internal_sids → internal LOAD（acc/idx）·input 须 OPERAND 叶**且 sid == input 对应
        internal sid**（sorted-position 对应 skeleton mv(arity+k)↔input sorted[k]·变量同一性牙·镜像 PARAM 路径 :1517-1519·
        拒 Sigma(1,z,z) body=z=PARAM 误识为 Sigma·审2 MED-2）。in_internal_sids/sk_arity 默认 None（symbolic_transform caller·
        sk_internal_sids=None 恒不入此分支）→ 退化宽松判定·bit-identical。
      - skeleton STORE 节点 → input 须 STORE·递归 value 子。
      - skeleton CTRL 节点 → input 须同 ATTR_CTRL_TAG·递归 [cond, body]。
      sk_internal_sids=None/空（默认）→ 直线算子骨架·internal/STORE/CTRL 分支不触发·bit-identical（symbolic_transform caller 守）。
    """
    if depth > _MAX_DISCOVER_DEPTH:
        return False   # 病态深→不识别（防栈溢出·loop1 严格树不触发）
    sk_attrs = read_composes_attrs(backend, sk)
    in_attrs = read_composes_attrs(backend, inp)
    _sk_internal = sk_internal_sids if sk_internal_sids is not None else set()
    if ATTR_OPERAND in sk_attrs:
        # S3: skeleton internal LOAD（acc/idx·sid ∈ sk_internal_sids）→ input 须 OPERAND 叶 + 对应 input internal sid。
        # 变量同一性牙（镜像 PARAM 路径 同槽值须等）：sorted-position 对应 skeleton mv(arity+k)↔input sorted[k]
        # ·拒 Sigma(1,z,z)（body=z=PARAM 非 internal）误识为 Sigma（=三角数）·审2 MED-2 修。
        if sk_attrs[ATTR_OPERAND][0] in _sk_internal:
            if ATTR_OPERAND not in in_attrs:
                return False
            if in_internal_sids is None or sk_arity is None:
                return True   # 默认（symbolic_transform caller·sk_internal_sids=None 恒不入此分支·防御）
            in_sid = in_attrs[ATTR_OPERAND][0]
            sk_k = index_of(sk_attrs[ATTR_OPERAND][0]) - sk_arity
            if not (0 <= sk_k < len(in_internal_sids)):
                return False   # skeleton internal sid 越界 input internal 集 → 结构异（防御）
            return in_sid == in_internal_sids[sk_k]   # sorted-position 对应 + 变量同一性（acc↔acc·idx↔idx）
        # skeleton PARAM 槽（sid=make_variable(slot)）→ input 须立即数叶（值绑定）或 operand 叶（operand 绑定）
        slot = index_of(sk_attrs[ATTR_OPERAND][0])   # sid → slot（make_variable 可逆·symbol_domain）
        if ATTR_IMMEDIATE in in_attrs:
            # immediate 输入位（既有路径·bit-identical）：slot 感知值绑定·同槽值须等（变量同一性）
            imm = in_attrs[ATTR_IMMEDIATE]
            val = (imm[0], imm[1])
            prev = value_binding.get(slot)
            if prev is not None and prev != val:
                return False   # 同槽值异（square 两叶·7*8 第二叶 8≠已绑 7）→ 拒（变量同一性）
            if slot in operand_binding:
                return False   # slot 已 operand-bound（一位一绑定·混合冲突）
            value_binding[slot] = val
            return True
        if ATTR_OPERAND in in_attrs:
            if subtree_binding is not None:
                # 符号模式（Phase 2b·doc §八-bis）：operand 叶 → subtree_binding（绑 slot→inp 叶 ConceptRef·
                # _eval_rhs 子树替换·d/dx base=x 是 operand 叶）。None 守既有 caller 零变（bit-identical）。
                prev = subtree_binding.get(slot)
                if prev is not None and prev != inp:
                    return False   # 同槽变量同一性
                if slot in value_binding or slot in operand_binding or slot in concept_binding:
                    return False   # 一位一绑定（混合冲突）
                subtree_binding[slot] = inp
                return True
            # operand 输入位（operand-input 识别·新）：slot→input operand slot·同 skeleton slot 须对齐同 input operand
            in_slot = index_of(in_attrs[ATTR_OPERAND][0])
            prev = operand_binding.get(slot)
            if prev is not None and prev != in_slot:
                return False   # 同 skeleton slot 对齐两不同 input operand（square 两叶·a*b 两异 operand）→ 拒（变量同一性牙）
            if slot in value_binding:
                return False   # slot 已 value-bound（一位一绑定·混合冲突）
            operand_binding[slot] = in_slot
            return True
        if not in_attrs:
            # 件5：input 无属性叶=语言 token（_is_concept_leaf 判定）→ concept_binding[slot]=input token ConceptRef。
            # S3 第二刀 Interp2 抽象匹配：skeleton slot 有 ATTR_SLOT_ROLE（LCA ref r）→ input token 沿 IS_A 可达 r
            # （含 inp==r）→ 抽象命中（"狐狸"命中"动物"槽·词级零交集·反 theater 牙）。无 ATTR_SLOT_ROLE → 原 ref 等价路径。
            # 两路都守同槽变量同一性（"猫追猫"两猫同槽须同 ref·对称防御·对抗审 L1 采纳）。
            prev = concept_binding.get(slot)
            if prev is not None and prev != inp:
                return False   # 同槽须同 token concept_ref（变量同一性·两路对称）
            if ATTR_SLOT_ROLE in sk_attrs:
                lca_sid, lca_lid = sk_attrs[ATTR_SLOT_ROLE]
                lca_ref: ConceptRef = (lca_sid, lca_lid)
                anc_inp = (ancestor_map.get(inp, set()) if ancestor_map else set())
                if lca_ref != inp and lca_ref not in anc_inp:
                    return False   # input token 沿 IS_A 不可达 skeleton slot LCA → 抽象不命中
            if ATTR_CUE_SIG in sk_attrs:
                # §十八 condition 6a-3 cue slot + 对应泛化 v2 三路分离（审1C3/审2条件1·阻塞·核心）：
                #  - cue_capture is None（recognize 精确匹配轨·闭命门2·**永不动**）：inp 须 == 闭类 cue token
                #    （是-skeleton 只认 是 input·非 使·held-out 按 cue 区分 是/使 skeleton·破 cue-blind 双计）。
                #    gate OFF 无 ATTR_CUE_SIG 写 → 不触·bit-identical。同槽变量同一性由上方 concept_binding 守。
                #  - cue_capture is not None（结构反推 tally 轨·独立于 recognize·不产 Recognition 不污染下游）：
                #    捕 inp 为 cue slot 落位词 W（**任意词非闭类**·突破精确匹配·新词引发 落 使-skeleton cue slot 亦计·
                #    W 可不在 oracle/frozenset → 真泛化·syntactic bootstrapping）·不拒（concept_binding[slot]=inp 续）。
                #    反推锚=结构（REALIZES-R-skeleton·caller tally_cue_slot_matches 守）非词在 oracle·非自证。
                if cue_capture is None:
                    cue_sid, cue_lid = sk_attrs[ATTR_CUE_SIG]
                    if inp != (cue_sid, cue_lid):
                        return False   # recognize 精确匹配：input 非 skeleton cue token → 不命中
                else:
                    cue_capture.append(inp)   # tally：捕 cue slot W（任意词·非闭类）·不拒·concept_binding 续
            if slot in value_binding or slot in operand_binding:
                return False   # 一位一绑定（混合冲突）
            concept_binding[slot] = inp
            return True
        if subtree_binding is not None and ATTR_OPERATOR in in_attrs:
            # 符号变换子树绑定（SYMBOLIC_TRANSFORM·doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis.5）：
            # PARAM 槽遇 input 复合子树（ATTR_OPERATOR 节点）→绑 slot→inp 子树根·解符号变换命门
            # （d/dx 的 VAR 须绑 x+1 子树·原此位 return False 拒）。subtree_binding=None（默认）→
            # 既有 recognize_operators/_align_extract caller 零行为变（bit-identical·None 守）。
            # 同槽变量同一性（同 slot 须绑同子树·对称既有 value/operand/concept 三 binding 防御）+
            # 一位一绑定（混合冲突拒）。
            prev = subtree_binding.get(slot)
            if prev is not None and prev != inp:
                return False   # 同槽变量同一性（Pow(VAR,VAR) 两 VAR 同槽须同子树）
            if slot in value_binding or slot in operand_binding or slot in concept_binding:
                return False   # 一位一绑定（混合冲突）
            subtree_binding[slot] = inp
            return True
        return False   # input 既非 IMM/OPERAND/无属性/子树绑定未启（ctrl/store 越界）→ 不识别
    if ATTR_IMMEDIATE in sk_attrs:
        # skeleton 固定位 → input 须立即数叶且值等（fixed 位值须等·非纯形状指纹·operand 输入的 fixed 位亦适用·如 z+3 须 IMM3 等）
        if ATTR_IMMEDIATE not in in_attrs:
            return False
        if sk_attrs[ATTR_IMMEDIATE] != in_attrs[ATTR_IMMEDIATE]:
            return False
        return True
    if ATTR_OPERATOR in sk_attrs:
        # skeleton 算子位 → input 须同 opcode 算子·同子数·递归子（DFS 前序·zip 复现阅读序）
        if ATTR_OPERATOR not in in_attrs:
            return False
        if sk_attrs[ATTR_OPERATOR][0] != in_attrs[ATTR_OPERATOR][0]:
            return False   # opcode 异（如 MUL vs ADD）→ 不识别
        sk_kids = sk_children.get(sk, [])
        in_kids = in_children.get(inp, [])
        if len(sk_kids) != len(in_kids):
            return False   # 子数异→结构异构
        for skc, inc in zip(sk_kids, in_kids):
            if not _align_walk(backend, sk_children, in_children, skc, inc,
                               value_binding, operand_binding, concept_binding, depth + 1,
                               ancestor_map=ancestor_map,
                               subtree_binding=subtree_binding,
                               sk_internal_sids=sk_internal_sids,
                               in_internal_sids=in_internal_sids, sk_arity=sk_arity,
                               cue_capture=cue_capture):
                return False
        return True
    if ATTR_STORE_TARGET in sk_attrs:
        # S3 ctrl/store-迭代骨架：skeleton STORE 节点 → input 须 STORE·递归 value 子（sid 结构对应·非值绑定·internal 非 PARAM）
        if ATTR_STORE_TARGET not in in_attrs:
            return False
        sk_kids = sk_children.get(sk, [])
        in_kids = in_children.get(inp, [])
        if len(sk_kids) != len(in_kids):
            return False   # 子数异→结构异构（STORE 须 1 值子）
        for skc, inc in zip(sk_kids, in_kids):
            if not _align_walk(backend, sk_children, in_children, skc, inc,
                               value_binding, operand_binding, concept_binding, depth + 1,
                               ancestor_map=ancestor_map,
                               subtree_binding=subtree_binding,
                               sk_internal_sids=sk_internal_sids,
                               in_internal_sids=in_internal_sids, sk_arity=sk_arity,
                               cue_capture=cue_capture):
                return False
        return True
    if ATTR_CTRL_TAG in sk_attrs:
        # S3 ctrl/store-迭代骨架：skeleton CTRL 节点 → input 须同 ATTR_CTRL_TAG·递归 [cond, body]
        if ATTR_CTRL_TAG not in in_attrs:
            return False
        if sk_attrs[ATTR_CTRL_TAG][0] != in_attrs[ATTR_CTRL_TAG][0]:
            return False   # CTRL tag 异 → 不识别
        sk_kids = sk_children.get(sk, [])
        in_kids = in_children.get(inp, [])
        if len(sk_kids) != len(in_kids):
            return False   # 子数异→结构异构（CTRL_WHILE 须 2 子 [cond,body]）
        for skc, inc in zip(sk_kids, in_kids):
            if not _align_walk(backend, sk_children, in_children, skc, inc,
                               value_binding, operand_binding, concept_binding, depth + 1,
                               ancestor_map=ancestor_map,
                               subtree_binding=subtree_binding,
                               sk_internal_sids=sk_internal_sids,
                               in_internal_sids=in_internal_sids, sk_arity=sk_arity,
                               cue_capture=cue_capture):
                return False
        return True
    # skeleton 越界 kind（混合属性/未知）→ 不识别（ctrl/store 已支持·此处仅病态混合触达）
    return False
