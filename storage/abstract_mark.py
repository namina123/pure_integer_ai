"""storage.abstract_mark — 节点抽象归属标记扩展表（§7.4 L212 + §7.7.1 路径 B·modality_subspace）。

引用全局节点身份 (space_id, local_id)·多维标记（modality/lang/domain/topo）·
相交查询 = 同节点多标记维度都有归属（set 交集·非 Venn·参照 legacy abstract_space.py:175 原理）。

**为何独立扩展表非 concept_node 加列**（路径 B 决断·§7.7.1·守铁律 6 不污染节点列）：
  concept_node 是核心 MUTABLE_MONOTONE 表（discipline.py:54 CORE_TABLES）·
  §7.4 L212 用户定调"复用 abstract_mark...不污染节点列"——节点列冻结·模态/语言/域等抽象归属
  走 core=False 扩展表（同 op_confidence/concept_identity/composes_attr/experience_count 先例·
  不碰 concept_node 不变量）。modality_marker 节点列（§十五决策3 L1357 临时扩列）迁此表
  MARK_MODALITY·节点列回归 6 列（路径 B·守 L212 单一事实源）。

**为何 DISC_NONE 非 APPEND_ONLY**（§7.7.1 决断 2·解 status flip 矛盾）：
  APPEND_ONLY 拒 update（discipline.check_write :93）·无法 status flip（PENDING/PROMOTED/ARCHIVED
  三态非单调·参照 legacy abstract_space.py:42-46）。status 非单调不适 MUTABLE_MONOTONE。故 DISC_NONE
  （自由 insert/update/delete·core=False 扩展表合法·discipline.py:18）+ set_mark query-then-upsert
  覆盖语义（参照 legacy abstract_space.py:122 原理·主线 backend 无 insert_or_replace 故 query-then-update-or-insert）。

**set_mark query-then-upsert**（query PK 四列·存在→update status·否则 insert）：
  幂等（同 status 不写·不同 status flip）·无 race（单线程确定执行）·PK 唯一性应用层守
  （同 experience_count record_base_freq first-write-wins 范式·DB 层无 UNIQUE·set_mark 协议守）。

**modality 迁移等价**（§7.7.1 决断 1·B6 P0）：
  亲核 MODALITY_LANGUAGE=1 非 0（types.py:26）·modality_marker 节点列当前恒=0（observe/ensure
  从未传非 0 给概念点 ensure·cognition/ 零真消费读）= vestigial 死列。迁移语义：概念点 modality_marker=0
  （未设）→ 不挂 abstract_mark → get_mark(MARK_MODALITY) 返 None ≡ 现状 modality_marker=0（未标）。
  完全等价（0 本就未设·非 LANGUAGE=1）。LANGUAGE 默认不挂 mark（查无 mark = 未标 modality·对称默认语义）。

mark_kind 枚举（mark_value 含义由 kind 定·表 dimension-agnostic）：
  MARK_MODALITY=1  一级模态（LANGUAGE/AUDIO/2D/3D/ANIMATION/CODE/ARITH·迁自节点列 modality_marker·mark_value=MODALITY_*）
  MARK_LANG=2      lang（ZH/EN·挂词形 NODE_WORD·§7.4 L213·mark_value=LANG_*）
  MARK_DOMAIN=3    域（TEXT/CODE/MATH/BARE·挂词形·mark_value=DOMAIN_*）
  MARK_TOPO=4      拓扑描述符（defer·非语言模态触发）

status 三态（参照 legacy abstract_space.py:42-46）：
  MARK_PENDING=1   待定（caller 标注弱信号）
  MARK_PROMOTED=2  提升（默认·强信号·set_mark 默认 status）
  MARK_ARCHIVED=3  归档（脱离活跃集·不删·append 原则）

铁律：纯整数（space_id/local_id/mark_kind/mark_value/status 全 int·assert_int 守）/ DISC_NONE
（status flip 非单调·core=False 合法）/ 确定性（set_mark query-then-upsert 幂等 + query_intersection
sorted 输出·bit-identical）/ 单向依赖（L0 storage 模块·L8 formal_train 写·L4 ConceptGraph 读·皆向下）/
不污染节点列（core=False 扩展表·守铁律 6）/ 不写死（schema 元定义列·mark_kind 枚举元定义例外·
mark_value 语义由 caller kind 定·表 agnostic）。
诚实边界：MARK_TOPO defer（schema 预留 mark_kind=4·实施待非语言模态触发）/ 拓扑描述符维度设计待
A2 拓扑分层（Phase 2）/ modality_subspace 是抽象归属标记系统非函数模块（§7.7.1 决断 5）。
参照 _archive/legacy_v1/pure_integer_ai/storage/abstract_space.py:71-192 原理·主线 API 重写非搬代码（铁律 9）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

ABSTRACT_MARK_TABLE = "abstract_mark"

# ---- mark_kind 枚举（mark_value 含义由 kind 定·表 dimension-agnostic） ----
MARK_MODALITY = 1   # 一级模态（迁自节点列 modality_marker·mark_value=MODALITY_*）
MARK_LANG     = 2   # lang（挂词形 NODE_WORD·mark_value=LANG_*）
MARK_DOMAIN   = 3   # 域（挂词形·mark_value=DOMAIN_*·**生产 caller 未接线·domain 维 defer**：
                     # lang 维 MARK_LANG 全链活（refers_to.py:116 OOV 分支写 + graph_view lang_of 读 + slot_dispatch 消费）；
                     # domain 配套设计 §7.4（TEXT/CODE/MATH）待接（须 Segment.domain 透传 + refers_to set_mark caller）。
                     # 非死码：mark_kind 枚举值·test_abstract_mark 多 kind infra 独立性测试用·保留。）
MARK_TOPO     = 4   # 拓扑描述符（defer·非语言模态触发）
MARK_MODAL_KIND = 5  # B2 模态种类（alethic/deontic·审计根治·mark_value=MODAL_KIND_*·D6:60 模态种类归抽象空间后天可学习）·
                     # **异 MARK_MODALITY=1**（后者=一级感官模态 LANGUAGE/AUDIO·先天分类·D6:75 标注装错待迁符号域·
                     # mark_value 值域碰撞 MODALITY_LANGUAGE=1 vs MODAL_KIND_BOX_NECESSITY=1·故新 mark_kind 避歧义）·
                     # D6 职责分离：abstract_mark MARK_MODAL_KIND=5 是 D6 语义归属声明（模态种类归抽象空间）·
                     # composes_attr ATTR_MODAL_KIND=22 是存储 readback 标记（lookup_word_modality 读·镜像 OP_*/REL_* 范式）·
                     # ensure_modal_primitives 双挂（set_mark D6 归属 + record_composes_attr readback 标记）·职责分离非重复·
                     # 生产消费者 defer（D6 归属元数据声明·非 readback·readback 走 ATTR_MODAL_KIND=22·lookup_word_modality）

# ---- status 三态（参照 legacy abstract_space.py:42-46） ----
MARK_PENDING  = 1
MARK_PROMOTED = 2   # 默认（强信号·set_mark 默认 status）
MARK_ARCHIVED = 3

_ABSTRACT_MARK_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("mark_kind", TYPE_INT),    # 维度（MARK_MODALITY/LANG/DOMAIN/TOPO）
    ("mark_value", TYPE_INT),   # 值（含义由 kind 定·MODALITY_*/LANG_*/DOMAIN_*/topo 描述符）
    ("status", TYPE_INT),       # 三态（PENDING/PROMOTED/ARCHIVED·DISC_NONE 可 flip）
]
_ABSTRACT_MARK_INDEXES = [
    ("space_id", "local_id"),   # 节点反向查（get_marks·set_mark PK 前缀覆盖）
    ("mark_kind", "mark_value"),  # 维度正向查（query_nodes_by_mark / query_intersection）
]


def register_abstract_mark(backend: StorageBackend) -> None:
    """注册 abstract_mark 扩展表（core=False·DISC_NONE·启动/用前调·幂等）。

    DISC_NONE（非 APPEND_ONLY）：status flip（PENDING/PROMOTED/ARCHIVED）非单调·APPEND_ONLY 拒 update
    无法 flip（§7.7.1 决断 2）。core=False 扩展表合法（discipline.py:18 DISC_NONE）。
    """
    register_extension_table(backend, ABSTRACT_MARK_TABLE,
                             _ABSTRACT_MARK_COLUMNS,
                             disc.DISC_NONE, _ABSTRACT_MARK_INDEXES)


def set_mark(backend: StorageBackend, *, ref: tuple[int, int],
             mark_kind: int, mark_value: int,
             status: int = MARK_PROMOTED) -> None:
    """挂标记（幂等 upsert·query-then-update-or-insert·参照 legacy abstract_space.py:122 原理）。

    ref=(space_id, local_id)·(mark_kind, mark_value) 是 PK 一半（节点+维度+值唯一）。
    - 行不存在 → insert（status 落）。
    - 行存在且 status 相同 → 幂等 skip（零写·bit-identical）。
    - 行存在且 status 不同 → update status（flip·DISC_NONE 允许）。
    表未注册（bare fixture / 未 register_abstract_mark）→ KeyError 静默 skip（向后兼容·
    同 record_base_freq / attach_chapter_seq 范式·observe 在 bare fixture 跑不崩）。
    """
    sid, lid = ref
    assert_int(sid, lid, mark_kind, mark_value, status, _where="set_mark.args")
    try:
        existing = backend.select(ABSTRACT_MARK_TABLE, where={
            "space_id": sid, "local_id": lid,
            "mark_kind": mark_kind, "mark_value": mark_value,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if existing:
        if existing[0]["status"] == status:
            return   # 幂等：同 status 不写
        backend.update(ABSTRACT_MARK_TABLE, where={
            "space_id": sid, "local_id": lid,
            "mark_kind": mark_kind, "mark_value": mark_value,
        }, set_={"status": status})
        return
    backend.insert(ABSTRACT_MARK_TABLE, {
        "space_id": sid, "local_id": lid,
        "mark_kind": mark_kind, "mark_value": mark_value, "status": status,
    })


def get_marks(backend: StorageBackend, *, ref: tuple[int, int],
              mark_kind: int | None = None,
              status: int | None = None) -> list[tuple[int, int, int]]:
    """读节点的标记列表 → [(mark_kind, mark_value, status), ...]。

    mark_kind=None → 全 kind；status=None → 全 status（含 PENDING/ARCHIVED）。
    输出按 (mark_kind, mark_value) 确定序（bit-identical·同插入序+值排序）。
    表未注册→[]（向后兼容·同 read_op_confidence 范式）。
    """
    sid, lid = ref
    assert_int(sid, lid, _where="get_marks.ref")
    where: dict[str, int] = {"space_id": sid, "local_id": lid}
    if mark_kind is not None:
        assert_int(mark_kind, _where="get_marks.mark_kind")
        where["mark_kind"] = mark_kind
    try:
        rows = backend.select(ABSTRACT_MARK_TABLE, where=where)
    except KeyError:
        return []
    out = [(r["mark_kind"], r["mark_value"], r["status"]) for r in rows]
    if status is not None:
        assert_int(status, _where="get_marks.status")
        out = [t for t in out if t[2] == status]
    # Python 层 sort (mark_kind, mark_value) 守字典序确定（backend order_by 单列·多值同 kind 时不稳）。
    return sorted(out)


def get_mark(backend: StorageBackend, *, ref: tuple[int, int],
             mark_kind: int, status: int = MARK_PROMOTED) -> int | None:
    """读节点单一 kind 的首 mark_value → int | None（单值便捷读·lang_of/modality_of 用）。

    status 默认 PROMOTED（强信号）·无行/表未注册→None。
    多值（同 kind 多 mark_value·如多 lang 词形）返首个（order_by mark_kind,mark_value 确定·
    caller 需全集用 get_marks）。
    """
    sid, lid = ref
    assert_int(sid, lid, mark_kind, status, _where="get_mark.args")
    marks = get_marks(backend, ref=ref, mark_kind=mark_kind, status=status)
    return marks[0][1] if marks else None


def query_nodes_by_mark(backend: StorageBackend, *, mark_kind: int,
                        mark_value: int,
                        status: int = MARK_PROMOTED) -> list[tuple[int, int]]:
    """按维度正向查节点 → [(space_id, local_id), ...]（带该 mark 且 status 匹配）。

    输出按 (space_id, local_id) 确定序（bit-identical）。表未注册→[]。
    """
    assert_int(mark_kind, mark_value, status, _where="query_nodes_by_mark.args")
    try:
        rows = backend.select(ABSTRACT_MARK_TABLE, where={
            "mark_kind": mark_kind, "mark_value": mark_value, "status": status,
        })
    except KeyError:
        return []
    # Python 层 sort (space_id, local_id) 守字典序确定（backend order_by 单列·同 space 多节点时不稳·
    # 与 query_intersection sorted 同契约·独立于插入序·bit-identical）。
    return sorted((r["space_id"], r["local_id"]) for r in rows)


def query_intersection(backend: StorageBackend,
                       *, marks: list[tuple[int, int]],
                       status: int = MARK_PROMOTED) -> list[tuple[int, int]]:
    """多维相交查询 → [(space_id, local_id), ...]（所有 mark 都命中的节点·set 交集·非 Venn）。

    marks = [(mark_kind, mark_value), ...]·返每个 mark 维度都有归属（且 status 匹配）的节点。
    单 mark = query_nodes_by_mark 退化。空 marks → []。
    参照 legacy abstract_space.py:175 set 交集原理（多维都有归属 = 同节点多 mark AND）。
    输出 sorted (space_id, local_id) 确定序（set.intersection 无序·sorted 守 bit-identical）。
    表未注册→[]。
    """
    if not marks:
        return []
    for mk, mv in marks:
        assert_int(mk, mv, _where="query_intersection.mark")
    assert_int(status, _where="query_intersection.status")
    per_dim: list[set[tuple[int, int]]] = []
    for mk, mv in marks:
        try:
            rows = backend.select(ABSTRACT_MARK_TABLE, where={
                "mark_kind": mk, "mark_value": mv, "status": status,
            })
        except KeyError:
            return []
        per_dim.append({(r["space_id"], r["local_id"]) for r in rows})
    common = set.intersection(*per_dim) if per_dim else set()
    return sorted(common)
