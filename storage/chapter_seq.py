"""storage.chapter_seq — 篇章结构层级承载（缺口①·修正分析九v2·阶段1c·独立扩展表）。

带章节标记的结构化文档（HTML/Markdown/LaTeX/代码/论文）的篇章结构序承载。inter-seg 现状仅扁平
PRECEDES（role_precedes.py:99 order_index 一维线性·章末→下章首与段末→下段首同形不可辨）→ 10 章
HTML 与 10 段流水账在图上同构·章节结构序丢失。本表补此缺口：每段 struct_ref 的 (chapter_seq,
section_seq, doc_seq) 标记·生成/检索可分章。

**为何独立扩展表（v2·镜像 op_confidence 范式·4 先例）**：
  - 不撞 def_array 二元区分（ref_space_id==0/!=0 撑 role/记忆序列两用途·第三用同 sentinel=0 会被
    read_role_seq 当伪 role 读出污染 role_seq·graph_view.py:55-62）。def_array schema 无 kind 列
    （node_store.py:60-64）·消费方无法区分 role 行 vs 章节序行。
  - 不复活 CONTAINS（最少边·C9-bis 砍成立·PROPERTY 覆盖概念 meronomy 非文档结构包含）。
  - 不建章节概念点（章节是结构序标记非概念·污染概念纯净化）。
  - 不污染 order_index（仍是 PRECEDES 步序字段·L1507·图算法层零读 oi 数值·a2_stepper 精确相等分组）。
  - 复用 op_confidence/composes_attr/concept_identity 先例（core=False 扩展表·register_extension_table）。

**输入层来源**（§十一 Q1·§三输入层契约）：Segment.chapter_seq/section_seq 由机器可读结构源 parse 填
（HTML 的 h1-h6 / Markdown 的 #/## / LaTeX 的 section/subsection 命令 / 代码 AST·**文学卷章回标题
正则 defer 偷渡语义**）·observe 读 segment.chapter_seq 调 attach_chapter_seq 落表（同 role_seq 从
输入 parse role 标记）。**章节序是输入结构真值（同 PRECEDES/order_index/role_seq 范式）·纯结构 parse
不涉语义·守 §8.1c 三死刑**。

disc=DISC_APPEND_ONLY（章节标记写一次 per struct_ref·(space_id,local_id) 唯一·结构标记稳定不重写）。
作用域仅带标记结构化文档（无标记主流文本章节推断=钥匙①范畴 defer·退化 chapter_seq=0 同流水账）。

铁律：纯整数（chapter_seq/section_seq/doc_seq 全 int·assert_int 守）/ 确定性（(space_id,local_id)
唯一·读回单行·bit-identical）/ APPEND_ONLY（写一次·不 update/delete·幂等 skip）/ 单向依赖（L0 storage·
L4 observe 写·L4 ConceptGraph 读·皆向下）/ 最少边（不复活 CONTAINS）/ 不污染 def_array 二元区分 /
不写死（章节序从输入机器可读源 parse·非硬编码）。
诚实边界：承载价值集中在带标记结构化文档（observe 真输入含此类）·无标记主流章节推断 defer 钥匙①
（**常态 defer 非边角**）·篇章语义理解=D 墙（L1243 跨章节语义回指）·M5/M6 真分页 defer（Stage 6）·
文学卷章回标题识别 defer（偷渡语义·首版只机器可读源）。详见 doc/重来_篇章结构层级设计_缺口①补.md。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

CHAPTER_SEQ_TABLE = "chapter_seq_table"

# doc_seq 首版单文档=0（多文档远期）
DEFAULT_DOC_SEQ = 0

_CHAPTER_SEQ_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("chapter_seq", TYPE_INT),    # 章序（机器可读结构源 parse·h1/# /section 等）
    ("section_seq", TYPE_INT),    # 节序（h2/## /\subsection 等）
    ("doc_seq", TYPE_INT),        # 文档序（首版单文档=0·多文档远期）
]
_CHAPTER_SEQ_INDEXES = [
    ("space_id", "local_id"),   # struct_ref 主键查（章节标记 per struct_ref 唯一）
]


def register_chapter_seq(backend: StorageBackend) -> None:
    """注册 chapter_seq_table 扩展表（core=False·APPEND_ONLY·启动/用前调·幂等）。"""
    register_extension_table(backend, CHAPTER_SEQ_TABLE,
                             _CHAPTER_SEQ_COLUMNS,
                             disc.DISC_APPEND_ONLY, _CHAPTER_SEQ_INDEXES)


def attach_chapter_seq(backend: StorageBackend, *, ref: tuple[int, int],
                       chapter_seq: int, section_seq: int,
                       doc_seq: int = DEFAULT_DOC_SEQ) -> None:
    """落段 struct_ref 的篇章结构序标记（observe 创建 struct_ref 后调·镜像 attach_role_seq 语义但落独立表）。

    ref=(space_id, local_id)·章节序从 segment.chapter_seq/section_seq（机器可读结构源 parse 填）。
    幂等：(space_id, local_id) 已有 → skip（APPEND_ONLY·章节标记稳定写一次·结构标记不重写）。
    表未注册（bare fixture / observe 热路径未 register_chapter_seq）→ KeyError 静默 skip
    （向后兼容·同 record_concept_identity 范式·observe 在 bare fixture 跑不崩）。
    """
    sid, lid = ref
    assert_int(sid, lid, chapter_seq, section_seq, doc_seq,
               _where="attach_chapter_seq.args")
    try:
        existing = backend.select(CHAPTER_SEQ_TABLE, where={
            "space_id": sid, "local_id": lid,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if existing:
        return   # 幂等：该 struct_ref 章节标记已落（APPEND_ONLY·不重写）
    backend.insert(CHAPTER_SEQ_TABLE, {
        "space_id": sid, "local_id": lid,
        "chapter_seq": chapter_seq, "section_seq": section_seq,
        "doc_seq": doc_seq,
    })
