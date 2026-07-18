"""cognition.shared.modal_primitives — 模态种类原语 first-class NODE_CONCEPT（L0 元定义层·审计根治·D6 对齐）。

模态种类（□必然/◇可能/道义必然/道义可能）作 first-class NODE_CONCEPT 节点·D:11 EDGE_RELATION_SIGNAL
词→模态种类概念边的 typed target。镜像 operator_primitives.py 范式（OP_* 算子/比较原语）·MODAL_KIND_*
是模态种类的元定义命名空间（**D6 抽象空间内容·非符号空间 type_ref 先天原语**·§九铁律承认 enum 例外·
同 REL_* / OP_* / OPCODE_* / ORIGIN_*·reward 不调·断奶前后不变）。

**符号空间 vs 抽象空间**（D6·AGENT.md:54-77·勿混）：模态种类（alethic/deontic）= 抽象空间**后天·可学习·归纳**
（D6:60 明确·"节点**属于**哪个抽象"）。文字 alias（必然/可能/必须/应该/可以）是开放类 surface·
走 D:11 learnable 二源（frozenset 冷启动种子 + D:11 readback 教师晋升）·非硬编码穷举。

**审计根治 [严重-1]**（解 _MODAL_CUES 换名字写死）：STEP6 B2 情态 _MODAL_CUES dict 把模态种类从词硬编码到
int·modal_op_of 无 D:11 readback 4 参（异 arith_op_of/comparison_op_of/is_property_possess_cue 全有二源）·
"无 REL_MODALITY 故无 D:11"循环论证偷渡（REL_MODALITY 不存在是实施方自选·用其缺失证硬编码合理）。
本模块根治：建 MODAL_KIND_* concept + D:11 边 + composes_attr readback（镜像 OP_*/REL_*）+ abstract_mark
D6 归属·开放变体（想必/势必/说不定）走 D:11 教师晋升有路径。

**D6 职责分离**（双挂·非重复）：
  - composes_attr ATTR_MODAL_KIND=22：存储 readback 标记（int_a=MODAL_KIND_*·lookup_word_modality 读·
    镜像 lookup_word_operator 读 ATTR_OPERATOR_PRIMITIVE=18·readback 走 composes_attr 同既有 4 个 D:11 范式工程一致）
  - abstract_mark MARK_MODAL_KIND=5：D6 语义归属声明（mark_value=MODAL_KIND_*·模态种类归抽象空间 D6:60·
    set_mark 在 ensure 时调·D6 合规）
  两者职责不同：composes_attr 是存储 readback·abstract_mark 是 D6 归属·非重复。

**为何不建 REL_MODALITY/OP_MODAL**（STOP + D6 双违排除）：
  - REL_* 是符号空间关系原语（D6:59·先天冻结·STOP 后不扩）·建 REL_MODALITY 违 STOP + 违 D6（模态种类归抽象空间非关系原语）
  - OP_* 是符号空间算子原语（D6:59·先天冻结·STOP 后不扩）·建 OP_MODAL 违 STOP + 违 D6（□/◇ 后天归纳非算子原语）
  - abstract_mark mark_kind 是抽象空间（D6:60·非 STOP 管辖 TYPE_* 符号域）·用 MARK_MODAL_KIND=5 不违 STOP 不违 D6

**ATTR_MODAL_KIND=22**（composes_attr·非结构 kind·_STRUCTURAL_KINDS 不含·read_composes_tree 忽略·
inline 不传播·镜像 ATTR_OPERATOR_PRIMITIVE=18 / ATTR_RELATION_PRIMITIVE=10）。

**D:11 共享边类型隔离**：EDGE_RELATION_SIGNAL 边 word→any concept。REL_* target 挂 ATTR_RELATION_PRIMITIVE·
OP_* target 挂 ATTR_OPERATOR_PRIMITIVE·MODAL_KIND target 挂 ATTR_MODAL_KIND。lookup_word_concept 过滤
ATTR_RELATION_PRIMITIVE·lookup_word_operator 过滤 ATTR_OPERATOR_PRIMITIVE·lookup_word_modality 过滤
ATTR_MODAL_KIND（kind==0 skip·无交叉污染）。

**MODAL_KIND_* = modality 编码**（1-4·与 P0.3 surface modality int 一致·modal_op_of readback 返 modal_kind
即 modality 值·不需 OP_*→opcode 映射·比 operator 简单）：
  MODAL_KIND_BOX_NECESSITY=1     □ 必然（认识·epistemic necessity）
  MODAL_KIND_BOX_POSSIBILITY=2   ◇ 可能（认识·epistemic possibility）
  MODAL_KIND_DEONTIC_NECESSITY=3 道义必然（deontic necessity·must）
  MODAL_KIND_DEONTIC_POSSIBILITY=4 道义可能（deontic possibility·can）

位于 cognition/shared（L0）·import storage 向下合规·非 re-export（同 operator_primitives.py 范式）。

铁律：纯整数（MODAL_KIND_*/ConceptRef 全 int·assert_int 守）/ 确定性（stable surface hash bit-identical）/
  不写死（MODAL_KIND_* enum=meta定义例外·非语义规则·closed-class 种子·开放变体走 D:11 教师晋升）/
  单向依赖（L0 依赖 storage 向下）/ D:11 不接 reward（effective_weight.py:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_MODAL_KIND, read_composes_attrs
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.abstract_mark import set_mark, MARK_MODAL_KIND
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN

# ---- MODAL_KIND_* 枚举（模态种类·meta定义·审计根治·D6 抽象空间·非符号空间 type_ref） ----
# 元定义层固化·非语义规则（同 REL_* / OP_* / OPCODE_*·reward 不调·断奶前后不变）
# = modality 编码（P0.3 surface modality int 一致·modal_op_of readback 返此即 modality 值）
MODAL_KIND_BOX_NECESSITY = 1        # □ 必然（认识·epistemic necessity）
MODAL_KIND_BOX_POSSIBILITY = 2      # ◇ 可能（认识·epistemic possibility）
MODAL_KIND_DEONTIC_NECESSITY = 3    # 道义必然（deontic necessity·must）
MODAL_KIND_DEONTIC_POSSIBILITY = 4  # 道义可能（deontic possibility·can）

# 稳定 surface（content_hash dedup·跨 run identity·bit-identical·镜像 _OP_SURFACE 范式）
_MODAL_SURFACE: dict[int, str] = {
    MODAL_KIND_BOX_NECESSITY: "__MODAL_BOX_NECESSITY__",
    MODAL_KIND_BOX_POSSIBILITY: "__MODAL_BOX_POSSIBILITY__",
    MODAL_KIND_DEONTIC_NECESSITY: "__MODAL_DEONTIC_NECESSITY__",
    MODAL_KIND_DEONTIC_POSSIBILITY: "__MODAL_DEONTIC_POSSIBILITY__",
}

# ---- 元定义层种子词（lang → {word: modal_kind}·镜像 cue_words._MODAL_CUES + operator_primitives._OP_LEXICAL_CUE
# closed-class 情态副词核心·D6·开放变体（想必/势必/说不定 等穷举不尽）走 D:11 教师晋升非硬编码） ----
# 与 cue_words._MODAL_CUES 重叠是自然——两者都识别情态词·_MODAL_CUES=cue 检测 dict（第一源·返 modality int）·
# _MODAL_LEXICAL_CUE=D:11 boot 种子（第二源·建 word→MODAL_KIND concept D:11 边·readback 返 modal_kind=modality）·
# 加二源非替换（D6 关键区分4）·互补非冲突。
_MODAL_LEXICAL_CUE: dict[int, dict[str, int]] = {
    LANG_ZH: {
        "必然": MODAL_KIND_BOX_NECESSITY,
        "可能": MODAL_KIND_BOX_POSSIBILITY,
        "也许": MODAL_KIND_BOX_POSSIBILITY,
        "必须": MODAL_KIND_DEONTIC_NECESSITY,
        "应该": MODAL_KIND_DEONTIC_NECESSITY,
        "可以": MODAL_KIND_DEONTIC_POSSIBILITY,
    },
    # EN defer（modal 词 must/can/may/should/might·同 property cue ZH-first·EN 情态窗口 defer·须 tokenization）
}


def ensure_modal_primitives(concept_index, backend: StorageBackend, *,
                            space_id: int) -> dict[int, tuple[int, int]]:
    """ensure 全部 MODAL_KIND_* first-class NODE_CONCEPT 节点 + ATTR_MODAL_KIND=22 readback 标记
    + abstract_mark MARK_MODAL_KIND=5 D6 归属声明。

    镜像 ensure_operator_primitives·每 MODAL_KIND_*：
      concept_index.ensure(_MODAL_SURFACE[kind], NODE_CONCEPT, TIER_PRIMARY) → ref
      + record_composes_attr(backend, ref, kind=ATTR_MODAL_KIND, int_a=kind)  # 存储 readback 标记
      + set_mark(backend, ref=ref, mark_kind=MARK_MODAL_KIND, mark_value=kind)  # D6 归属声明
    返 {modal_kind: ConceptRef}（caller bootstrap_modal_signals 用·D:11 target 解析）。

    **D6 职责分离双挂**：composes_attr ATTR_MODAL_KIND=22 是存储 readback 标记（lookup_word_modality 读）·
    abstract_mark MARK_MODAL_KIND=5 是 D6 语义归属（模态种类归抽象空间）·两者职责不同非重复。

    **幂等**（ConceptIndex.ensure 同 hash 返既有 tier 单调升 + record_composes_attr 同 (ref,kind) skip +
    set_mark 同 status 幂等 skip）→ 每 boot 调安全（resume 跨 run / 重复 boot 不 corrupt）。

    无条件 ensure 全部 MODAL_KIND_*（元定义层常驻·类 REL_*/OP_*·boot 种 D:11 边前先建 target）。

    backend 显式传（镜像 ensure_operator_primitives·record_composes_attr/set_mark 需 backend·不触 ConceptIndex 私有 _b）。
    """
    assert_int(space_id, _where="ensure_modal_primitives.space_id")
    out: dict[int, tuple[int, int]] = {}
    for kind, surface in _MODAL_SURFACE.items():
        ref = concept_index.ensure(surface, space_id=space_id,
                                   tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        record_composes_attr(backend, ref=ref,
                             kind=ATTR_MODAL_KIND, int_a=kind)
        set_mark(backend, ref=ref, mark_kind=MARK_MODAL_KIND, mark_value=kind)
        out[kind] = ref
    return out


def lookup_word_modality(backend: StorageBackend, edge_store: EdgeStore,
                         word_ref: tuple[int, int], *, space_id: int,
                         tier_filter: int | None = None,
                         ) -> list[tuple[tuple[int, int], int]]:
    """读 word_ref 的 D:11 边 → [(modal_ref, modal_kind), ...]（modal D:11 readback API·镜像 lookup_word_operator）。

    query_from(word_ref, D:11) → 每 target 读 read_composes_attrs 得 ATTR_MODAL_KIND int_a=modal_kind。
    kind==0 skip（非 MODAL_KIND 节点·如 REL_* target 挂 ATTR_RELATION_PRIMITIVE·OP_* target 挂
    ATTR_OPERATOR_PRIMITIVE·过滤隔离无交叉污染）。

    **modal_kind = modality 编码**（1-4·与 P0.3 surface modality int 一致·caller modal_op_of 直接返此作 modality 值·
    不需 OP_*→opcode 映射·比 operator 简单）。

    **tier_filter**（反 theater）：传 TIER_PRIMARY 只返 PRIMARY 边（已验证晋升/教师种子）·
    None（默认）返全 tier（含 SHADOW·bit-identical）。caller（_modal_from_d11_primary）
    传 TIER_PRIMARY（未验证 SHADOW 不注入 readback·反 theater）。
    """
    if word_ref is None:
        return []
    rows = edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    out: list[tuple[tuple[int, int], int]] = []
    for r in rows:
        if tier_filter is not None and r.get("tier") != tier_filter:
            continue   # tier 过滤（反 theater：未验证 SHADOW 不注入 readback）
        modal_ref = (r["space_id_to"], r["local_id_to"])
        attrs = read_composes_attrs(backend, modal_ref)
        kind = attrs.get(ATTR_MODAL_KIND, (0, 0))[0]
        if kind == 0:
            # 防御：D:11 target 无 ATTR_MODAL_KIND（如 REL_*/OP_* target·挂 ATTR_RELATION_PRIMITIVE/ATTR_OPERATOR_PRIMITIVE）
            # → kind=0 非合法 MODAL_KIND（enum 1-4）·skip 不返（无交叉污染·lookup_word_concept/operator 同范式过滤）
            continue
        out.append((modal_ref, kind))
    return out
