"""storage.op_confidence — 发现算子置信度台账（洗净循环反馈半闭环·§8.7-洗·2026-07-03）。

发现算子（mul/square·结构发现学到）的 vm_proof 验证置信度（sn/tn/strength）·跨 episode 累积·
recognize_operators 读之择优（"洗净"=滤非泛化算子·§8.7-洗）。镜像 edge_store.record_episode_result
的 sn/tn/strength 单调更新范式（已批准·非发明新机制）+ composes_attr register 范式。

**为何独立表非 composes_attr**（3 对抗智能体收敛·§8.7-洗）：composes_attr 是 DISC_APPEND_ONLY
（composes_attr.py:80）·record_composes_attr 同 (ref,kind) 幂等 skip（:92-97）·read_composes_attrs
按 kind dict 覆写（:116-117）→ 物理上无法跨 episode 累积 sn/tn（首写即锁）·改纪律会放行全部 7 结构
kind 的 update → 腐化 COMPOSES 子树。op_confidence 独立 MUTABLE_MONOTONE 表不碰 composes_attr 不变量·
亦不碰 §4.5 COMPOSES-inert 真值（算子置信度挂 name 节点·非 CAUSES 边·算子是结构工具非因果 agent·
§六"CAUSES strength"仅语言域 H4 闭环·算子域走本表）。

**键 = 算子 name 节点**（`__op_disc_{tag}`·ATTR_OPERATOR_DEF 挂载点·arith_observe.py:109·**不在
COMPOSES 子树内**故不污染结构子树）。name 节点 = concept_node（DUMP_TABLES 核心表·持久）·op_confidence
加进 formal_train.dump_tables（同序列7 composes_attr 模式）·load_run table-agnostic 还原 → 跨 run 置信度
累积（"学习累积"环）。

R1 episode 符号契约（同 reward_propagate.py:125-146）：verified→sn++&tn++&strength+=Δ / fail→tn++ only
（sn 单调不降·率自然降·非 sn--）。

铁律：纯整数（sn/tn/strength 全 int·assert_int 守）/ MUTABLE_MONOTONE（表纪律·delta 固定 +1 无负·
表纪律双保险）/ append-only 行级（insert 一次 + update 列·同 edge 表范式·不动 composes_attr 不变量）/
确定性（bit-identical）/ 单向依赖（L0 storage·L8 formal_train 写·L5 recognize 读·皆向下）/ 不写死
（schema 元定义列·计数器非语义规则）。
诚实边界：算子置信度是 vm_proof 验证计数非语义正确性（stable≠correct）/ 生产算术域正确算子 held-out
必过（构造性必然）·置信度滤只在坏算子（PARAM 序错/编译发散/shape 异配）触发 / mul/square 不可区分
（置信度正交于变量同一性判别器·§8.7-洗 诚实边界①）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

OP_CONFIDENCE_TABLE = "operator_confidence"

# 初始 strength（同 edge_store.DEFAULT_STRENGTH·先验·verified 增·reward 不调 base 此处无 base/strength 分立）
DEFAULT_OP_STRENGTH = 1

_OP_CONFIDENCE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("sn", TYPE_INT),
    ("tn", TYPE_INT),
    ("strength", TYPE_INT),
]
_OP_CONFIDENCE_INDEXES = [
    ("space_id", "local_id"),   # name 节点 ref 主键查
]


def register_op_confidence(backend: StorageBackend) -> None:
    """注册 operator_confidence 扩展表（core=False·MUTABLE_MONOTONE·启动/用前调·幂等）。"""
    register_extension_table(backend, OP_CONFIDENCE_TABLE,
                             _OP_CONFIDENCE_COLUMNS,
                             disc.DISC_MUTABLE_MONOTONE, _OP_CONFIDENCE_INDEXES)


def read_op_confidence(backend: StorageBackend,
                       ref: tuple[int, int]) -> tuple[int, int, int] | None:
    """读算子置信度 → (sn, tn, strength) | None（无行=未验证冷启动·recognize 给机会非滤）。

    表未注册→None（环境未启 op_confidence 台账·冷启动·向后兼容 recognize_operators 在 bare 测试
    fixture 上 bit-identical 首匹配·非静默降级：无台账=无置信数据=给机会·production formal_train
    make_train_context 注册本表后读真置信度）。
    """
    sid, lid = ref
    assert_int(sid, lid, _where="read_op_confidence.ref")
    try:
        rows = backend.select(OP_CONFIDENCE_TABLE, where={
            "space_id": sid, "local_id": lid,
        }, limit=1)
    except KeyError:
        return None   # 表未注册（caller 未 register_op_confidence）·冷启动·向后兼容
    if not rows:
        return None   # 冷启动（未验过）·caller 判 None→rate=0 不滤
    r = rows[0]
    return (r["sn"], r["tn"], r["strength"])


def record_op_outcome(backend: StorageBackend, *, ref: tuple[int, int],
                      verified: bool) -> None:
    """记一次 vm_proof 验证结果（R1 episode 符号·镜像 edge_store.record_episode_result）。

    verified=True  → sn+=1, tn+=1, strength+=1（参与即成功·episode 级·strength 单调）
    verified=False → tn+=1 only（失败计数·sn 不降·率自然降·非 sn--）
    首次：insert with outcome applied（sn=1 if verified else 0·tn=1·strength=DEFAULT+[verified]）·
      同 update 路径从 (0,0,DEFAULT) 起的终态（无 (0,0,DEFAULT) 中间态·op_confidence 无独立 add 步）。
    守 MUTABLE_MONOTONE：delta 固定 +1（无负·MonotoneViolation 不可达·表纪律双保险）。
    """
    sid, lid = ref
    assert_int(sid, lid, _where="record_op_outcome.ref")
    existing = backend.select(OP_CONFIDENCE_TABLE, where={
        "space_id": sid, "local_id": lid,
    }, limit=1)
    if not existing:
        sn0 = 1 if verified else 0
        st0 = DEFAULT_OP_STRENGTH + (1 if verified else 0)
        backend.insert(OP_CONFIDENCE_TABLE, {
            "space_id": sid, "local_id": lid,
            "sn": sn0, "tn": 1, "strength": st0,
        })
        return
    set_: dict[str, tuple[str, int]] = {"tn": ("+=", 1)}
    if verified:
        set_["sn"] = ("+=", 1)
        set_["strength"] = ("+=", 1)
    backend.update(OP_CONFIDENCE_TABLE, where={
        "space_id": sid, "local_id": lid,
    }, set_=set_)
