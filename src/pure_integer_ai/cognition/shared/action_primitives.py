"""cognition.shared.action_primitives — 动作意图原语 first-class NODE_CONCEPT（L0 元定义层·B-PR1·doc §16·D6 对齐）。

动作意图原语（INTENT_COMMAND_MOOD 命令 mood + ACTION_GENERATE/COMPUTE/ANALYZE/SOLVE 4 类）作 first-class
NODE_CONCEPT 节点·D:11 EDGE_RELATION_SIGNAL 词→动作意图概念边的 typed target。**镜像 operator_primitives.py
范式**（单挂 ATTR·符号空间先天 closed-class 种子·D:11 二源·**非 modal 双挂**·doc §16.3）。

**符号空间 vs 抽象空间**（D6·AGENT.md:54-77·勿混）：动作意图原语 = 符号域**先天·冻结·元定义·closed-class 种子**
（首版 5 类·新类别走演化闸 defer·§14.2 E3·类 OP_* +/×）。文字 alias（帮我/请/生成/计算/劳驾/编写）是开放类
surface·走 D:11 learnable 二源（frozenset 冷启动种子 + D:11 readback 教师晋升）·非硬编码穷举。

**镜像 operator 非 modal**（doc §16.3 决断·B-PR1 设计智能体核证）：只挂 composes_attr ATTR_OPERATION_INTENT=23
readback 标记·**不挂 abstract_mark**（动作意图归符号空间先天·非抽象空间后天·异 modal_primitives 双挂 ATTR+MARK）。
证据：工程清单 §B1.2"镜像 operator_primitives.py" + §14.4"镜像 #940" + §14.2 E3"镜像 ATTR_OPERATOR_PRIMITIVE=18"
+ §14.6"走 composes_attr 非 type_ref"（只提 composes_attr 不提 abstract_mark）。

**ATTR_OPERATION_INTENT=23 boot concept 旗标**（概念身份·ensure 时挂·非学习·镜像 ATTR_OPERATOR_PRIMITIVE=18）·
**非 B-PR2 经验回写**（doc §16.2：record_composes_attr 同 (ref,kind) 幂等 skip → B-PR2 回写会 noop·故经验回写走
experience_count 对偶 op_confidence·§14.2 路 B·三层职责分离不混）。

**W7 命令判定 + B-PR1 类别判定共用此基建**（doc §16.4）：
  - W7（intent_classify._has_action_intent）：lookup_word_action D:11 命中任一 ATTR_OPERATION_INTENT concept
    （命令词 OR 动作词·int_a 0-4）→ type=INTENT_COMMAND
  - B-PR1（类别）：具体 int_a（1-4 ACTION_*·B-PR3 gate③ 按类分流 COMPUTE→VM/GENERATE→序化器）
  - 命令 mood（int_a=0）无动作类别·纯命令判定

**doc §15.1 纠正③ 推翻**（doc §16.1）：命令词（帮我/请）**走 D:11**（非"不走 D:11"）——命令 mood 概念先天·命令词
alias 开放·D:11 学（同 operator +概念先天/相加 alias D:11）。ATTR 回写标记 sink·不学新命令词。

**D:11 共享边类型隔离**：EDGE_RELATION_SIGNAL 边 word→any concept。REL_* target 挂 ATTR_RELATION_PRIMITIVE·
OP_* target 挂 ATTR_OPERATOR_PRIMITIVE·MODAL_KIND target 挂 ATTR_MODAL_KIND·动作意图 target 挂
ATTR_OPERATION_INTENT。lookup_word_concept/operator/modality/action 各过滤（ATTR 存在判据·见 lookup_word_action）·无交叉污染。

位于 cognition/shared（L0）·import storage 向下合规·非 re-export（同 operator_primitives.py 范式）。

铁律：纯整数（ACTION_INTENT_*/ConceptRef 全 int·assert_int 守）/ 确定性（stable surface hash bit-identical）/
  不写死（ACTION_INTENT_* enum=meta定义 closed-class 种子·开放 alias 走 D:11 教师晋升）/ 单向依赖（L0 依赖 storage 向下）/
  D:11 不接 reward（effective_weight.py assert 只认 {PRECEDES,CAUSES,REFERS_TO}）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, ATTR_OPERATION_INTENT, read_composes_attrs)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN

# ---- ACTION_INTENT_* 枚举（动作意图原语·meta定义·B-PR1·D6 符号空间先天·非抽象空间 abstract_mark） ----
# 元定义层固化·非语义规则（同 REL_* / OP_* / MODAL_KIND_*·reward 不调·断奶前后不变）
# int_a 值域（ATTR_OPERATION_INTENT）：0=命令 mood / 1-4=动作类别
INTENT_COMMAND_MOOD = 0   # 命令 mood（帮我/请/劳驾 D:11 target·W7 命令判定·无动作类别）
ACTION_GENERATE = 1       # 生成动作（生成/编写/创建）
ACTION_COMPUTE = 2        # 计算动作（计算/求）
ACTION_ANALYZE = 3        # 分析动作（分析/解析）
ACTION_SOLVE = 4          # 解决动作（解决/求解）

# 动作类别集合（B-PR1 类别判定用·不含 COMMAND_MOOD=0·COMMAND_MOOD 无类别）
_ACTION_CLASSES = frozenset({ACTION_GENERATE, ACTION_COMPUTE, ACTION_ANALYZE, ACTION_SOLVE})

# 稳定 surface（content_hash dedup·跨 run identity·bit-identical·镜像 _OP_SURFACE 范式）
_ACTION_SURFACE: dict[int, str] = {
    INTENT_COMMAND_MOOD: "__INTENT_COMMAND_MOOD__",
    ACTION_GENERATE: "__ACTION_GENERATE__",
    ACTION_COMPUTE: "__ACTION_COMPUTE__",
    ACTION_ANALYZE: "__ACTION_ANALYZE__",
    ACTION_SOLVE: "__ACTION_SOLVE__",
}

# ---- 元定义层种子词（lang → {word: action_intent_kind}·镜像 operator_primitives._OP_LEXICAL_CUE
# closed-class 核心·D6·开放变体（劳驾/烦请/编写/运算 等穷举不尽）走 D:11 教师晋升非硬编码） ----
# 命令 mood 词（→INTENT_COMMAND_MOOD·祈使引导词·帮我/请·**非动作动词**·职责正交）+ 动作词（→ACTION_* 类别·B-PR1）。
# W7 命令判定 = 命令词 OR 动作词命中任一（doc §16.4）。
# 动作词去歧义大的（"写"写信/"算"算命 不入种子·开放变体走 D:11 教师晋升 + D3 三重防巧合缓解误判）。
_ACTION_LEXICAL_CUE: dict[int, dict[str, int]] = {
    LANG_ZH: {
        # 命令 mood（祈使引导词·帮我/请·非动作动词·W7 doc §15/§16）
        "帮我": INTENT_COMMAND_MOOD, "请": INTENT_COMMAND_MOOD,
        "给我": INTENT_COMMAND_MOOD, "能不能": INTENT_COMMAND_MOOD,
        "可不可以": INTENT_COMMAND_MOOD, "麻烦": INTENT_COMMAND_MOOD,
        # 动作词（→ACTION_* 类别·每类 closed-class 核心 1 词·开放 alias 编写/创建/运算/解析/求解 走 D:11 学）
        "生成": ACTION_GENERATE,
        "计算": ACTION_COMPUTE,
        "分析": ACTION_ANALYZE,
        "解决": ACTION_SOLVE,
    },
    LANG_EN: {
        # EN 单 token（多 token "can you" 等死条目·tokenizer 空白分词·doc §15 审1 B1·EN 扩展 defer）
        "please": INTENT_COMMAND_MOOD,
        "generate": ACTION_GENERATE,
        "compute": ACTION_COMPUTE, "calculate": ACTION_COMPUTE,
        "analyze": ACTION_ANALYZE,
        "solve": ACTION_SOLVE,
    },
}


def ensure_action_primitives(concept_index, backend: StorageBackend, *,
                             space_id: int) -> dict[int, tuple[int, int]]:
    """ensure 全部 ACTION_INTENT_* first-class NODE_CONCEPT 节点 + ATTR_OPERATION_INTENT=23 标记。

    镜像 ensure_operator_primitives。每 ACTION_INTENT_*：
      concept_index.ensure(_ACTION_SURFACE[kind], NODE_CONCEPT, TIER_PRIMARY) → ref
      + record_composes_attr(backend, ref, kind=ATTR_OPERATION_INTENT, int_a=kind)
    返 {action_intent_kind: ConceptRef}（caller bootstrap_action_signals 用·D:11 target 解析）。

    **ATTR_OPERATION_INTENT=23 boot concept 旗标**（概念身份·非 B-PR2 经验回写·doc §16.2）。
    **不挂 abstract_mark**（镜像 operator·符号空间先天·非 modal 双挂·doc §16.3）。

    **幂等**（ConceptIndex.ensure 同 hash 返既有 tier 单调升 + record_composes_attr 同 (ref,kind) skip）→
    每 boot 调安全（resume 跨 run / 重复 boot 不 corrupt）。

    无条件 ensure 全部 ACTION_INTENT_*（元定义层常驻·类 REL_*/OP_*/MODAL_KIND_*·boot 种 D:11 边前先建 target）。

    backend 显式传（镜像 ensure_operator_primitives·record_composes_attr 需 backend·不触 ConceptIndex 私有 _b）。
    """
    assert_int(space_id, _where="ensure_action_primitives.space_id")
    out: dict[int, tuple[int, int]] = {}
    for kind, surface in _ACTION_SURFACE.items():
        ref = concept_index.ensure(surface, space_id=space_id,
                                   tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        record_composes_attr(backend, ref=ref,
                             kind=ATTR_OPERATION_INTENT, int_a=kind)
        out[kind] = ref
    return out


def lookup_word_action(backend: StorageBackend, edge_store: EdgeStore,
                       word_ref: tuple[int, int], *, space_id: int,
                       tier_filter: int | None = None,
                       ) -> list[tuple[tuple[int, int], int]]:
    """读 word_ref 的 D:11 边 → [(action_ref, action_intent_kind), ...]（action D:11 readback API·镜像 lookup_word_operator）。

    query_from(word_ref, D:11) → 每 target 读 read_composes_attrs 得 ATTR_OPERATION_INTENT int_a=action_intent_kind。

    **None 判据**（非 kind==0·因 INTENT_COMMAND_MOOD=0 是合法值·异 lookup_word_operator 的 OP_* 1-7 哨兵）：
    ATTR_OPERATION_INTENT 不存在 → None → skip（非动作意图 target·如 REL_*/OP_*/MODAL_KIND target·
    过滤隔离无交叉污染）。存在则返（含 int_a=0 COMMAND_MOOD）。

    **action_intent_kind 值域**：0=INTENT_COMMAND_MOOD（命令 mood） / 1-4=ACTION_*（动作类别）。
    W7 命令判定用"命中任一"（len>0）·B-PR1 类别判定用具体 int_a（is_action_class_kind）。

    **tier_filter**（反 theater）：传 TIER_PRIMARY 只返 PRIMARY 边（已验证晋升/教师种子）·
    None（默认）返全 tier（含 SHADOW·bit-identical）。caller 传 TIER_PRIMARY（未验证 SHADOW 不注入 readback）。
    """
    if word_ref is None:
        return []
    rows = edge_store.query_from(word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    out: list[tuple[tuple[int, int], int]] = []
    for r in rows:
        if tier_filter is not None and r.get("tier") != tier_filter:
            continue   # tier 过滤（反 theater：未验证 SHADOW 不注入 readback）
        action_ref = (r["space_id_to"], r["local_id_to"])
        attrs = read_composes_attrs(backend, action_ref)
        attr = attrs.get(ATTR_OPERATION_INTENT)
        if attr is None:
            # 防御：D:11 target 无 ATTR_OPERATION_INTENT（如 REL_*/OP_*/MODAL_KIND target·挂别的 ATTR）→ skip 不返（无交叉污染）
            continue
        out.append((action_ref, attr[0]))
    return out


def is_command_mood_kind(kind: int) -> bool:
    """kind 是否命令 mood（INTENT_COMMAND_MOOD=0·W7 命令判定·无动作类别）。"""
    return kind == INTENT_COMMAND_MOOD


def is_action_class_kind(kind: int) -> bool:
    """kind 是否动作类别（ACTION_GENERATE/COMPUTE/ANALYZE/SOLVE·B-PR1 类别判定·不含 COMMAND_MOOD=0）。"""
    return kind in _ACTION_CLASSES


# ---- 断桥粗粒度 meta（B-PR3·doc §18·CHANNEL_* enum + 类别→通道映射·inert·Phase2 dispatch 接入） ----
# meta 定义 closed-class 种子（同 OP_* / _ACTION_SURFACE 范式·非语义规则·不写死合规）。
# **inert**：无运行时消费者（无生产代码路径读 _ACTION_CHANNEL_MAP·action_channel 纯读函数仅单测用）。
# 实际 dispatch（COMPUTE→VM 执行 / GENERATE→序化器 composes_unparse / ANALYZE→judge ΠG）= 断桥细粒度 Phase2
# （doc §14.4 #7·generate dispatch_slot 是未来消费者）。B-PR3 只交付 meta 定义·声称"通道 dispatch 已接通"= theater。
# ACTION_SOLVE→CHANNEL_VM 暂定（doc §14.3 E2 只提 3 通道·"求解"≈compute·Phase2 修订可能改）。
CHANNEL_NONE = 0          # 无通道（COMMAND_MOOD 命令词·无动作执行）
CHANNEL_VM = 1            # ACTION_COMPUTE/SOLVE → VM 执行（算术/代码 vm_proof）
CHANNEL_SERIALIZER = 2    # ACTION_GENERATE → 序化器（composes_unparse·路径 W #730）
CHANNEL_JUDGE = 3         # ACTION_ANALYZE → judge 分析（卷三 ΠG 判定）

# 类别→通道映射（int_a action_kind → CHANNEL_*·断桥粗粒度·meta·inert）
_ACTION_CHANNEL_MAP: dict[int, int] = {
    INTENT_COMMAND_MOOD: CHANNEL_NONE,
    ACTION_GENERATE: CHANNEL_SERIALIZER,
    ACTION_COMPUTE: CHANNEL_VM,
    ACTION_ANALYZE: CHANNEL_JUDGE,
    ACTION_SOLVE: CHANNEL_VM,   # 暂定·Phase2 修订（doc §14.3 E2 未定 SOLVE 通道）
}


def action_channel(action_kind: int) -> int:
    """动作类别→通道（断桥粗粒度 meta 查询·B-PR3·doc §18·纯读·inert）。

    返 CHANNEL_* enum。未知 kind（非 0-4）→ CHANNEL_NONE（安全默认）。
    **inert**：当前无生产消费者（Phase2 dispatch 接入·generate dispatch_slot 未来调）。
    单测断言映射正确性用（_ACTION_CHANNEL_MAP[ACTION_COMPUTE]==CHANNEL_VM）。
    """
    assert_int(action_kind, _where="action_channel.action_kind")
    return _ACTION_CHANNEL_MAP.get(action_kind, CHANNEL_NONE)
