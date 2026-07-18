"""cognition.shared.symbol_types - 符号空间 type_ref 先天分类命名层（L0 元定义层·STEP3·D6 对齐）。

符号空间 type_ref 先天分类的 canonical 命名层。TYPE_* 是符号原语的元定义命名空间（逻辑/算子/关系/系词原语）·
非语义规则（§九铁律承认 enum 例外·同 REL_* / OPCODE_* / ORIGIN_*·reward 不调·断奶前后不变）。

**符号空间 vs 抽象空间**（D6·AGENT.md:54-77·勿混）：
  - 符号空间（type_ref·**先天·冻结·元定义·不学**）：逻辑/算子/关系/系词原语（¬/CAUSES/COPULA/ATTR_MARKER）。
    canonical 真值源·"节点**是**什么"。参照 UD closed-class + 形式逻辑原语 + Spelke 核心知识。
  - 抽象空间（abstract_mark·**后天·可学习·归纳**）：元算子拓扑（META_AGG/META_MAP）/ 模态种类 / 场合性。
    "节点**属于**哪个抽象"。3 态 PENDING/PROMOTED/ARCHIVED。
  - 否定/算子/关系 = 符号域先天·绝不进抽象空间。

**TYPE_PROPOSITION 别名 ATTR_PROPOSITION=11**（既有·命题节点 observe build_property_edges 建 __prop_*·挂
ATTR_PROPOSITION=11·G1 reification·G3b 消费·非本模块 ensure 建 boot 种 __TYPE_PROPOSITION__）。

**¬ 走命题 surface polarity 不建 ATTR_NEGATION**（doc:193·¬ 是 type_ref 先天分类非二元关系·B1 ¬ 由命题
surface polarity 承载 P0.3 done `_1_0` 后缀·不需要 ATTR_NEGATION marker·仅未来消费者查"所有否定节点"才 ensure）。

**abstract_mark 迁移目标登记**（TYPE_ATTR_MARKER canonical·D6 §一 abstract_mark MARK_MODALITY/LANG/DOMAIN
全是先天分类该挂符号域却挂抽象空间·迁移目标登记于此·迁移本身 defer·disrupt 抽象空间消费者非 gating）。

**激活**（2026-07-11 #940 否定词 D:11 readback）：ensure_symbol_types 有消费者 bootstrap_negation_signals
（formal_train boot 调·镜像 ensure_operator_primitives/ensure_modal_primitives）·建 __TYPE_NEGATION__ concept
+ ATTR_SYMBOL_TYPE=17 作否定词 D:11 readback target。readback 由 gate NEGATION_D11_READBACK_MODE 守
（OFF 退化 frozenset is_negation_cue 第一源·bit-identical·D:11 边种但 readback 不读·同 operator/modal 范式）。
**否定=符号域先天**（D6·¬ = TYPE_NEGATION 先天·非抽象空间）·故 ensure 只挂 ATTR_SYMBOL_TYPE·不挂 abstract_mark
（同 operator_primitives·异 modal_primitives 双挂 ATTR+MARK）。

位于 cognition/shared（L0）·import storage.composes_attr/storage.node_store 跨层向下合规·非 re-export
（同 relation_primitives.py 范式）。

铁律：纯整数（TYPE_* int/ConceptRef 整/ATTR_SYMBOL_TYPE 整·零浮点·assert_int 守）/ 确定性（稳定 surface
  hash bit-identical）/ 单向依赖（L0 依赖 storage 向下）/ bit-identical（enum + 函数 ship 不调用 = AST 级零变）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.composes_attr import record_composes_attr, read_composes_attrs, ATTR_SYMBOL_TYPE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN

# ---- TYPE_* 枚举（符号空间 type_ref 先天分类·meta 定义·§九 enum 例外·D6 对齐·非抽象空间 abstract_mark） ----
# 元定义层固化·非语义规则（同 REL_* / OPCODE_* / ORIGIN_*·reward 不调·断奶前后不变）
TYPE_PROPOSITION = 11   # 别名 ATTR_PROPOSITION=11（既有·命题节点挂此·G1 reification·G3b 消费）
TYPE_NEGATION = 12      # ¬ 否定（B1·先天分类·¬ 走命题 surface polarity _1_0·不挂 ATTR_NEGATION·doc:193）
# 余登记不激活（defer 范式·同 EDGE_CALLS·消费者出现才 ensure）：
TYPE_COPULA = 13        # 系词 是（值系词·属性命题窗口·登记不激活）
TYPE_CMP = 14           # 比较算子 >/</>=/<=（刀D·登记不激活）
TYPE_CAUSES = 15        # 因果 CAUSES（登记不激活）
TYPE_ATTR_MARKER = 16   # 属性标记 的（abstract_mark MARK_MODALITY/LANG/DOMAIN 迁移目标·迁移 defer·doc:179）

# ---- ★STEP6 STOP 符号域扩张点（不上完备 TYPE_* 枚举·doc/重来_符号域修正分析_2026-07-10.md §五 line 175/210） ----
# STEP6（A1∃ + B2 情态 + STOP）后停止符号域 type_ref 扩张·转断奶（E3/E1·独立 session）。
# B2 情态走命题 surface modality int(0-4) + ATTR_PROP_POLMOD=21 结构存（P0.3+PR3）·**非 TYPE_MODALITY**。
# A1∃ 走 EXISTENTIAL_CUE=7 + existential_proof_fn（PR1）·**非 TYPE_EXISTENTIAL**。
# 完备 TYPE_* 枚举（¬/∀/∃/□/◇/∧/∨/→ 全活）defer 断奶后演化闸 propose/validate/freeze。
# abstract_mark MARK_MODALITY/LANG/DOMAIN 迁符号域 defer（disrupt 抽象空间消费者·cleanup defer）。
# ensure_symbol_types 已激活（#940 否定词 D:11 readback·bootstrap_negation_signals 消费·formal_train boot 调·readback gate NEGATION_D11_READBACK_MODE 守）。

# 稳定 surface（content_hash dedup·跨 run identity·bit-identical）
# 只建消费者需要的 surface（TYPE_NEGATION：¬ first-class NODE_CONCEPT·E3 defeater 未来查询目标）。
# TYPE_PROPOSITION 不建 surface（命题节点 observe 建 __prop_*·挂 ATTR_PROPOSITION=11·非 boot 种 __TYPE_PROPOSITION__）。
# 登记不激活的 TYPE_* 不建 surface（defer·消费者出现才 ensure）。
_TYPE_SURFACE: dict[int, str] = {
    TYPE_NEGATION: "__TYPE_NEGATION__",
}


def ensure_symbol_types(concept_index, backend: StorageBackend, *,
                        space_id: int) -> dict[int, tuple[int, int]]:
    """ensure 消费者需要的 TYPE_* first-class NODE_CONCEPT 节点 + ATTR_SYMBOL_TYPE=17 标记。

    镜像 ensure_relation_primitives。当前只建 TYPE_NEGATION（¬ first-class NODE_CONCEPT·E3 defeater 未来查询目标）。
    TYPE_PROPOSITION 不建（命题节点 observe build_property_edges 建 __prop_*·挂 ATTR_PROPOSITION=11）。
    登记不激活的 TYPE_* 不建（defer·消费者出现才 ensure）。

    每 TYPE_*：concept_index.ensure(_TYPE_SURFACE[kind], NODE_CONCEPT, TIER_PRIMARY) -> ref
    + record_composes_attr(backend, ref, kind=ATTR_SYMBOL_TYPE, int_a=kind)。
    返 {type_kind: ConceptRef}。

    **幂等**（ConceptIndex.ensure 同 hash 返既有 tier 单调升 + record_composes_attr 同 (ref,kind) skip）->
    每 boot 调安全（resume 跨 run / 重复 boot 不 corrupt）。

    无条件 ensure 消费者需要的（元定义层常驻·类 REL_*）。

    **已激活**（#940 否定词 D:11 readback）：本函数被 bootstrap_negation_signals 消费（formal_train boot
    调·镜像 ensure_operator_primitives/ensure_modal_primitives）。readback 由 gate NEGATION_D11_READBACK_MODE
    守（OFF 退化 frozenset is_negation_cue 第一源·bit-identical·D:11 边种但 readback 不读·同 operator/modal 范式）。

    backend 显式传（镜像 ensure_relation_primitives 接 concept_index+backend 范式·record_composes_attr
    需 backend·不触 ConceptIndex 私有 _b）。
    """
    assert_int(space_id, _where="ensure_symbol_types.space_id")
    out: dict[int, tuple[int, int]] = {}
    for kind, surface in _TYPE_SURFACE.items():
        ref = concept_index.ensure(surface, space_id=space_id,
                                   tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        record_composes_attr(backend, ref=ref,
                             kind=ATTR_SYMBOL_TYPE, int_a=kind)
        out[kind] = ref
    return out


# ---- 元定义层种子词（lang → frozenset·镜像 cue_words._NEGATION_CUES·D:11 boot 种子·#940） ----
# closed-class 否定词核心（不/没/非/无 + not/no/never）·D6·开放变体（未必/绝非/谈不上/休想 穷举不尽）走 D:11 教师晋升。
# 与 cue_words._NEGATION_CUES 重叠是自然——_NEGATION_CUES=cue 检测第一源（frozenset·返 bool）·
# _NEGATION_LEXICAL_CUE=D:11 boot 种子（第二源·建 word→__TYPE_NEGATION__ D:11 边·readback 返 bool）·加二源非替换。
_NEGATION_LEXICAL_CUE: dict[int, frozenset[str]] = {
    LANG_ZH: frozenset({"不", "没", "非", "无"}),
    LANG_EN: frozenset({"not", "no", "never"}),
}


def lookup_word_negation(backend: StorageBackend, edge_store: EdgeStore,
                         word_ref: tuple[int, int], *, space_id: int,
                         tier_filter: int | None = None) -> bool:
    """读 word_ref 的 D:11 边 → 是否指向 TYPE_NEGATION concept（否定 D:11 readback API·#940·镜像 lookup_word_operator）。

    query_from(word_ref, D:11) → 每 target 读 read_composes_attrs 得 ATTR_SYMBOL_TYPE int_a·
    int_a==TYPE_NEGATION(12) → True（此词是否定 cue·D:11 教师晋升/种子）。全不命中→False。

    与 lookup_word_concept/operator/modality 隔离：否定 target 挂 ATTR_SYMBOL_TYPE（int_a=12）·
    REL_*/OP_*/MODAL_KIND target 挂各自 ATTR（无 ATTR_SYMBOL_TYPE·或 int_a≠12）→ 过滤无交叉污染。

    **tier_filter**（反 theater）：传 TIER_PRIMARY 只认 PRIMARY 边（已验证晋升/教师种子）·
    None（默认）认全 tier。caller（is_negation_cue 第二源）传 TIER_PRIMARY（未验证 SHADOW 不注入·反 theater）。

    **否定=符号域先天**（D6·¬ = TYPE_NEGATION 先天非抽象空间）·故只挂 ATTR_SYMBOL_TYPE 不挂 abstract_mark
    （同 operator·异 modal 双挂）。D:11 readback 意义=否定词文字 alias 可学习（教师晋升新否定词如未必/绝非）·
    非 ¬ 概念本身可学（¬ 先天冻结·同 OP_*）。
    """
    if word_ref is None:
        return False
    rows = edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    for r in rows:
        if tier_filter is not None and r.get("tier") != tier_filter:
            continue   # tier 过滤（反 theater：未验证 SHADOW 不注入 readback）
        neg_ref = (r["space_id_to"], r["local_id_to"])
        attrs = read_composes_attrs(backend, neg_ref)
        kind = attrs.get(ATTR_SYMBOL_TYPE, (0, 0))[0]
        if kind == TYPE_NEGATION:
            return True   # 命中否定 D:11 边（word→__TYPE_NEGATION__）
    return False
