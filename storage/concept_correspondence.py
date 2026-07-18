"""storage.concept_correspondence - 概念↔对应（码点等）持久化（P0a·让系统 SPEAK + dump 留文本）。

**问题（A 偏离）**：`concept_index.ensure()`（concept_index.py:85）dedup 用 content_hash(surface) 后**丢弃
surface** -> 码点（unicode ord 整数·本可入核心·用户从始至终不变量）从未存储 -> surface_of 无源 ->
generate 吐 `#1:42` 占位 + dump 零文字 + load 回无法产字。

**修（P0a·纯 additive·bit-identical）**：observe 建 concept 时同步写码点到本表 -> surface_of resolver
读码点 -> chr -> 文本。一个抽象概念（local_id·单调计数器·概念身份）↔ 多条对应行（corr_kind 区分模态
表示·镜像 NODE_CONCEPT 抽象 / NODE_WORD 词形 / REFERS_TO 链接的现有架构·用户的"概念↔对应"基调）。

**为何独立表非 concept_node 加列**（决策3）：concept_node 冻结 6 列纯整数（无 text/hash/码点列）。
码点是有序变长数据 -> 走决策4"序列符号"范式（独立整数表 + order_index 编序·同 def_array/chapter_seq）·
**不触决策3**·concept_node 保持纯整数。core=False 扩展表（同 concept_identity/composes_attr 范式）。

**corr_kind（多模态对应扩展位·首版只写 ordinal）**：
  0 = ordinal（语言模态·码点有序数组·本表首版落点）
  1 = topology（2D/3D 拓扑 fingerprint·defer 到非语言模态触发）
  2 = voiceprint（声模态·defer）
schema 留 corr_kind 列（廉价 1 int）·首版只写 kind=0·未来加 kind 行免 migration（§3.5"新模态按此接入
不动主图"）。corr_kind 是 meta-definition enum（同 NODE_* type enum·AGENT.md"不写死"例外）。

**一行一码点 + order_index**（def_array 有序范式·node_store.py:64-74）：无 TEXT 列（守纯整数 + 决策3
精神·core=False 表虽不抛 CoreStrViolation 但破坏概念纯洁性）。record 幂等（同 concept+kind 已写 -> skip）。

铁律：纯整数（码点 ord int + local_id/corr_kind/order_index 全 int）/ 确定性（有序编解码·bit-identical）/
单向依赖（L0 storage·cognition L4 ConceptIndex 写 / graph_view 读·皆向下）/ APPEND_ONLY（写一次·不
update/delete）/ 不写死（schema 元定义列 + corr_kind meta enum 例外）/ 决策3 守（concept_node 不动）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

CONCEPT_CORRESPONDENCE_TABLE = "concept_correspondence"

# corr_kind 常量（meta-definition enum·首版只写 CORR_ORDINAL·TOPO/VOICEPRINT/NUMERIC 预留扩展位）
CORR_ORDINAL = 0       # 语言模态：码点有序数组（ord·P0a 落点）
CORR_TOPO = 1          # 2D/3D 拓扑 fingerprint（defer·非语言模态触发）
CORR_VOICEPRINT = 2    # 声模态（defer）
CORR_NUMERIC = 3       # 数字词接地（语言→形式域·三/three→3·language-grounding piece 1·单 int 值 order_index=0）

_CONCEPT_CORRESPONDENCE_COLUMNS = [
    ("space_id",    TYPE_INT),   # ref 端点（concept_node PK）
    ("local_id",    TYPE_INT),   # ref 端点（concept_node PK·抽象概念身份）
    ("corr_kind",   TYPE_INT),   # 0 ordinal / 1 topo / 2 voiceprint（多模态对应扩展位）
    ("order_index", TYPE_INT),   # 码点序位（0,1,2,...·def_array 有序范式·解码按此排序）
    ("value",       TYPE_INT),   # 一个码点一行（ord(ch)·纯整数）
]
_CONCEPT_CORRESPONDENCE_INDEXES = [
    ("space_id", "local_id"),                        # 概念维查（该 concept 全对应）
    ("space_id", "local_id", "corr_kind"),           # resolver 按 kind 过滤 + record 幂等查
    ("space_id", "local_id", "corr_kind", "order_index"),  # resolver ORDER BY order_index 覆盖
]


def register_concept_correspondence(backend: StorageBackend) -> None:
    """注册 concept_correspondence 扩展表（core=False·DISC_APPEND_ONLY·启动/用前调·幂等）。"""
    register_extension_table(backend, CONCEPT_CORRESPONDENCE_TABLE,
                             _CONCEPT_CORRESPONDENCE_COLUMNS,
                             disc.DISC_APPEND_ONLY, _CONCEPT_CORRESPONDENCE_INDEXES)


def load_correspondence(backend: StorageBackend, *,
                        space_id: int, local_id: int,
                        corr_kind: int) -> tuple[int, ...]:
    """读某 (concept, kind) 的有序码点数组（surface_of resolver 用）。

    返空 tuple = 表未注册（bare fixture·向后兼容）或该 (concept, kind) 无行。
    **确定性**：同 (concept, kind) 内 order_index 唯一连续（record 写一次·ensure 幂等）·按 order_index
    排序 -> bit-identical（select 序无关）。
    """
    assert_int(space_id, local_id, corr_kind, _where="load_correspondence")
    try:
        rows = backend.select(CONCEPT_CORRESPONDENCE_TABLE, where={
            "space_id": space_id, "local_id": local_id, "corr_kind": corr_kind,
        })
    except KeyError:
        return ()   # 表未注册（bare fixture 未 register）·向后兼容
    ordered = sorted(rows, key=lambda r: r["order_index"])
    return tuple(r["value"] for r in ordered)


def record_correspondence(backend: StorageBackend, *,
                          space_id: int, local_id: int, corr_kind: int,
                          codepoints) -> None:
    """持久化 (concept, kind) 的有序码点数组（ensure 新建概念点后调·best-effort）。

    幂等：同 (concept, kind) 已有行 -> skip（APPEND_ONLY·对应写一次·身份稳定）。表未注册（bare fixture）
    -> KeyError 静默 skip（向后兼容·镜像 record_concept_identity 范式：select-gate 守 insert·未注册早退）。
    一行一码点（order_index 编序·def_array 范式）。
    """
    assert_int(space_id, local_id, corr_kind, _where="record_correspondence")
    # codepoints 元素逐个 assert_int（防 float/str 混入·守纯整数）
    cp_list = list(codepoints)
    for i, cp in enumerate(cp_list):
        assert_int(cp, _where=f"record_correspondence.codepoint[{i}]")
    try:
        existing = backend.select(CONCEPT_CORRESPONDENCE_TABLE, where={
            "space_id": space_id, "local_id": local_id, "corr_kind": corr_kind,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture 未 register）-> 向后兼容 skip（insert 前早退·守 best-effort）
    if existing:
        return   # 幂等：该 (concept, kind) 对应已写（APPEND_ONLY·不重写）
    for order_index, cp in enumerate(cp_list):
        backend.insert(CONCEPT_CORRESPONDENCE_TABLE, {
            "space_id": space_id, "local_id": local_id,
            "corr_kind": corr_kind, "order_index": order_index, "value": cp,
        })


def load_numeric(backend: StorageBackend, *, space_id: int, local_id: int) -> int | None:
    """读数字词概念的接地整数值（language-grounding piece 1·CORR_NUMERIC）。

    数字词概念（三/three/十二）挂单 int 值（order_index=0）·返该 int | None（无接地行/表未注册）。
    供 language→arith word-problem：数字词 token → 整数 → arith IMM operand（形式域 vm_proof 验）。
    镜像 load_correspondence（kind=CORR_NUMERIC·取 order_index=0 行 value）·确定性 bit-identical。
    """
    cps = load_correspondence(backend, space_id=space_id, local_id=local_id,
                              corr_kind=CORR_NUMERIC)
    return cps[0] if cps else None


def record_numeric(backend: StorageBackend, *, space_id: int, local_id: int,
                   value: int) -> None:
    """持久化数字词概念的接地整数值（language-grounding piece 1·CORR_NUMERIC·ensure 后调·best-effort）。

    单 int 值（order_index=0）·幂等（同 concept 已写 skip·APPEND_ONLY）·表未注册 KeyError 静默 skip（向后兼容）。
    镜像 record_correspondence（kind=CORR_NUMERIC·codepoints=[value]）。
    """
    assert_int(value, _where="record_numeric.value")
    record_correspondence(backend, space_id=space_id, local_id=local_id,
                          corr_kind=CORR_NUMERIC, codepoints=[value])
