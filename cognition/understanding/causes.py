"""cognition.understanding.causes — 模块2-bis CAUSES 因果建边（§8.1c 硬边界·D1 落盘）。

CAUSES 是 reward 反传唯一落点（§十五C9-bis）、J3 G3a 因果锚依据（卷三）。
§8.1c-bis 来源分层：
  ① 结构化源（代码 AST def→use / ConceptNet Causes 有向三元组 / 证明步骤）——照搬不反转方向（M1）
  ② 指向词 + 句法位置提取（导致/所以/使得/引发·§8.1c-bis"结构推断"=指向词锚定）
  ③ 断奶前 LLM 教师确认（走录放层·断奶后退场）
  ④ 传递闭包涌现（algorithm/closure·按需派生·此处只存原始边）

**build_causes ≠ build_condition**（§8.1c 硬边界·条件包含 ≠ 因果·辛普森/反向/confounding）。
CAUSES 只从结构化源/指向词/断奶前 LLM·不从统计涌现（裸共现给不了因果·§8.1c-bis）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.config import gates
from pure_integer_ai.storage.edge_store import (
    EdgeStore, DEFAULT_STRENGTH, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM,
    SOURCE_CONCEPTNET,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.cognition.shared.edge_types import EDGE_CAUSES
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    concept_assertion,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore


def build_causes_edges(edge_store: EdgeStore,
                       refs: list[tuple[int, int]],
                       *,
                       structured_pairs: list[tuple[int, int]],
                       cue_pairs: list[tuple[int, int]],
                       source: int, space_id: int,
                       teacher_confirmed: list[tuple[int, int]] | None = None,
                       weaning_phase: int = 0,
                       assertion_scope: ScopeIdentity | None = None,
                       assertion_store: ScopedIdentityStore | None = None,
                       qualifier_prefix: tuple[int, ...] = ()) -> int:
    """CAUSES 因果建边（来源①②③·④闭包按需派生非此）。

    structured_pairs / cue_pairs / teacher_confirmed：token index 对·经 refs 解析为概念 ref。
    teacher_confirmed 仅断奶前合法（来源③·weaning_phase==PRE）·断奶后忽略。
    返回建边数。
    """
    n = 0
    if (assertion_scope is None) != (assertion_store is None):
        raise ValueError("CAUSES assertion_scope 与 assertion_store 必须同时提供")
    # ① 结构化源（有向照搬不反转·M1·epistemic_origin=STRUCTURED）
    for ai, bi in structured_pairs:
        n += _insert_causes(edge_store, refs[ai], refs[bi],
                            source=source, epistemic=EPI_STRUCTURED, space_id=space_id,
                            assertion_scope=assertion_scope,
                            assertion_store=assertion_store,
                            qualifiers=(*qualifier_prefix, ai, bi))
    # ② 指向词 + 句法（epistemic_origin=CUE）
    for ai, bi in cue_pairs:
        n += _insert_causes(edge_store, refs[ai], refs[bi],
                            source=source, epistemic=EPI_CUE, space_id=space_id,
                            assertion_scope=assertion_scope,
                            assertion_store=assertion_store,
                            qualifiers=(*qualifier_prefix, ai, bi))
    # ③ 断奶前 LLM 教师确认（epistemic_origin=LLM_CONFIRM·断奶后退场）
    if teacher_confirmed and weaning_phase == 0:  # WEANING_PRE=0
        for ai, bi in teacher_confirmed:
            n += _insert_causes(edge_store, refs[ai], refs[bi],
                                source=5,  # SOURCE_TEACHER
                                epistemic=EPI_LLM_CONFIRM, space_id=space_id,
                                assertion_scope=assertion_scope,
                                assertion_store=assertion_store,
                                qualifiers=(*qualifier_prefix, ai, bi))
    return n


def _insert_causes(edge_store: EdgeStore, a: tuple[int, int], b: tuple[int, int],
                   *, source: int, epistemic: int, space_id: int,
                   assertion_scope: ScopeIdentity | None = None,
                   assertion_store: ScopedIdentityStore | None = None,
                   qualifiers: tuple[int, ...] = ()) -> int:
    if a == b:
        return 0
    # space_id 来自参数（M4·按 stage）·端点 ref 已在目标 space 建
    # gate CAUSES_DEDUP_MODE（perf round5·mirror PRECEDES_DEDUP_MODE）：ON 走 add_causes_dedup 跨 round
    # 去重（解 observe 16× 重复边膨胀·silent skip·**reward 影响零**·终审 resolver 3 路径全证伪·纯 perf + 数据卫生）。
    # OFF 走旧 add（16×·CI bit-identical）·详见 add_causes_dedup docstring。
    # 有 scoped assertion 时，旧 edge 只能保留关系聚合的一行；每次 episode 的
    # 独立事实已经进入 assertion registry，不能再让兼容行重复膨胀并失去 EdgeRef 唯一性。
    if assertion_scope is not None or getattr(gates, "CAUSES_DEDUP_MODE", False):
        inserted = edge_store.add_causes_dedup(
            space_id_from=a[0], local_id_from=a[1],
            space_id_to=b[0], local_id_to=b[1],
            edge_type=EDGE_CAUSES, source=source,
            epistemic_origin=epistemic, tier=TIER_PRIMARY,
        )
    else:
        edge_store.add(
            space_id_from=a[0], local_id_from=a[1],
            space_id_to=b[0], local_id_to=b[1],
            edge_type=EDGE_CAUSES, strength=DEFAULT_STRENGTH,
            source=source, epistemic_origin=epistemic,
            order_index=None, role=None,   # CAUSES 无 order_index 时序语义（§十三C·OR 语义）
            tier=TIER_PRIMARY,
        )
        inserted = True
    if assertion_scope is not None and assertion_store is not None:
        assertion_store.register_assertion(concept_assertion(
            EDGE_CAUSES,
            a,
            b,
            scope=assertion_scope,
            provenance_kind=source,
            epistemic_origin=epistemic,
            qualifiers=qualifiers,
        ))
    return 1 if inserted else 0


def bootstrap_causes_edges(concept_index, edge_store: EdgeStore,
                           surface_pairs: list[tuple[str, str]],
                           *, space_id: int,
                           source: int = SOURCE_CONCEPTNET,
                           epistemic: int = EPI_STRUCTURED) -> int:
    """CAUSES 批量 boot 种边（入手④·surface pairs → ensure → build·给 CAUSES 外部 R6 独立源）。

    与 `build_causes_edges`（observe caller·token index 对·来源② 指向词 EPI_CUE / ③ LLM）的差异：
      - 入参是 surface 文本对（非 token index）·caller 不依赖语料 token 切片（boot 时种边·早于 observe）。
      - 默认来源① ConceptNet（source=SOURCE_CONCEPTNET·epistemic=EPI_STRUCTURED·ConceptNet Causes 有向三元组
        cause Causes effect 照搬不反转·M1·§8.1c-bis 来源①）。
      - **幂等 skip 按源细化**：query_from 查同 (cause,effect,EDGE_CAUSES,source) 已建则 skip（不挡 observe
        EPI_CUE 路径·同源同三元组才 skip·镜像 bootstrap_is_a_edges:128-137 范式）。

    **无文件零副作用硬守（bit-identical·P0）**：surface_pairs 空 → 立即 return 0·**绝不调
      concept_index.ensure / query_from / _insert_causes**（无 ZERO_AI_LOCAL_DIR → resolve 返 [] →
      图与不接入手④ bit-identical·镜像 bootstrap_is_a_edges:119-120）。

    每 (cause_surface, effect_surface)：concept_index.ensure 两 ref（TIER_PRIMARY·NODE_CONCEPT）→
    query_from 幂等 skip → _insert_causes（a==b 跳·:57 已守）。返建边数。

    **用途**（入手④·总收口 §三簇1④）：formal_train boot 段（make_train_context 后·observe 前）调·让
    ConceptNet 外部 CAUSES 边早于 observe 种入 → CAUSES 有外部 R6 独立源（非仅 cue 自产/LLM）→ 因果 reward
    反传有外部锚·刀G R6 合规路径。**CAUSES 接 reward 反传**（effective_weight:82 assert 认 CAUSES·异 IS_A
    不接 reward）·故 active causes_facts 文件会改变训练 reward（**预期·R6 外部因果信号**）·无文件则零副作用。

    铁律：纯整数（ConceptRef + EDGE_CAUSES 整边·零浮点）/ 不写死（surface 来自外部文件·本函数只机制非语义）/
      §8.1c-bis（来源① EPI_STRUCTURED 合规）/ bit-identical（空 pairs 零副作用 + query_from 幂等 + 文件存在性是输入）。
    诚实边界：因果真伪/方向 = 外部数据责任（ConceptNet 可错·stable≠correct·#479 墙·照搬不反转不校验）·系统不判·只落边。
    """
    if not surface_pairs:
        return 0   # P0·无文件零副作用硬守（不调 ensure/query_from/build·CI/生产 default bit-identical）
    assert_int(space_id, source, _where="bootstrap_causes_edges.args")
    n = 0
    for cause_surf, effect_surf in surface_pairs:
        cause_ref = concept_index.ensure(
            cause_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        effect_ref = concept_index.ensure(
            effect_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 幂等 skip（按源细化·不挡 observe EPI_CUE 路径·镜像 bootstrap_is_a_edges:128-137）：
        # query_from 查 cause 已有同 (effect,source) CAUSES 边。
        existing = edge_store.query_from(cause_ref[0], cause_ref[1], edge_type=EDGE_CAUSES)
        already = any(
            row.get("space_id_to") == effect_ref[0]
            and row.get("local_id_to") == effect_ref[1]
            and row.get("source") == source
            for row in existing
        )
        if already:
            continue   # 同源同三元组已建→skip（幂等·resume 跨 run / 重复 boot 不 corrupt·EdgeStore.add 不去重）
        n += _insert_causes(edge_store, cause_ref, effect_ref,
                            source=source, epistemic=epistemic, space_id=space_id)
    return n


# CONDITION（EDGE_CONDITION=7·§8.1c 条件包含≠因果）写侧 2026-07-09 删（YAGNI·总收口 §五1.2）：
# 死写侧 8 年——无 parser 设 has_condition=True（生产永不 fire）+零读侧消费者（CONDITION 不进 reward
# 反传·不进 judge/闭包/dag_path 任何推理）。形式检查走 self_proof_fn（Layer0·done）不需 CONDITION 边。
# EDGE_CONDITION=7 保留注册（C9-bis B:7 登记但不激活·同 EDGE_CALLS/EDGE_INSTANTIATES 模式·设计合规）。
# 补活（condition_evaluate/resolve_by_role §7）须独立设计 session·非 YAGNI 清理范围。
