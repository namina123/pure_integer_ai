"""storage.structure_match_count — 结构反推 tally 台账（对应泛化 v2·cue↔rel 学全）。

结构反推（用户定义 2026-07-17）：新词 W 落在 oracle-确认结构（REALIZES-R-skeleton）的 cue 槽位 →
推 W↔R。tally (W,R) distinct input_root ≥K（生产为 recognition-routed sample）→ promote W→R D:11 PRIMARY（D:11 删 ∨·只认结构匹配轨·
修审2 BLOCKER 1）。本表 = tally 计数落点（relation-specific + structure-grounded·修审2 BLOCKER 2）。

**一行 per (W,R,input_root) append-only**（非 experience_count 的 update 计数器范式）：
  - 去重键 = (space_id, word_sid, word_lid, rel_kind, sample_sid, sample_lid)（审2 条件4b·distinct
    tally sample 去重·sample_root_ref 维）。
  - distinct tally sample count for (W,R) = 该 (W,R) 的行数（每行=1 distinct input_root）。
  - append-only（DISC_APPEND_ONLY·insert only·重放同语料→同行集→同 count·天然幂等抗刷数·防 theater）。
  - 无计数器列：distinct 语义=行数·非次数（同 input_root 重见=skip·不计权重·distinct tally sample）。

**为何独立表非 experience_count 加列**（同 experience_count 范式）：experience_count 是 word 级
**不区分关系**（= BLOCKER 1 病源·reward-feed 染·跨域污染）；本表 **relation-specific**（(W,R) 键）+
**structure-grounded**（只计 REALIZES-R-skeleton cue slot 落位·非任意 reward feed·tally hook gated
ORACLE_PROMOTE_MODE）+ **落盘**（跨 run 可恢复·修 BLOCKER 2·REALIZES 边 + ATTR_CUE_SIG 已落盘·resume
recognize 仍见 REALIZES+ATTR_CUE_SIG → 可续 tally）。core=False 扩展表（同 op_confidence/experience_count
范式·不碰 concept_node 不变量）。

**SHADOW 边创建（审2 条件2·promote 前提·不在本表）**：tally hook 调 record_structure_match 返 new=True
（首次该 input_root for (W,R)）→ caller 调 record_emergent_relation_signal_shadow 建 D:11 SHADOW 行
（generator 关后的唯一创建者·record_emergent_relation_signal_shadow 自身 query_from 幂等 skip·
防同 (W,R) 多 input_root 重复建边）。本表只计 distinct sample·不建边（建边在 emergent_relation_signal）。

铁律：纯整数（ConceptRef/rel_kind 全 int·assert_int 守）/ APPEND_ONLY（表纪律·insert only·重放幂等）/
  确定性（bit-identical·sorted 无关·check-then-insert 单线程无双 insert）/ 不写死（schema 元定义列·
  计数非语义规则·rel_kind enum 例外同 REL_*）/ 单向依赖（L0 storage·tally hook L8 formal_train 写·
  promote L7 读·皆向下）/ 反 theater（distinct sample 去重防刷数·structure-grounded 非 reward feed）。
诚实边界：本表是 tally 地基（计数 + 落盘）·非泛化判定（promote 闸在 _structure_match_ok·specificity +
false-positive 实测在 held-out·stable≠correct·详见 doc/重来_对应泛化_结构反推_学全_2026-07-17.md）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

STRUCTURE_MATCH_COUNT_TABLE = "structure_match_count"


_STRUCTURE_MATCH_COUNT_COLUMNS = [
    ("space_id", TYPE_INT),       # 语言 space（W 与 sample_root 同 space·存一份主 space）
    ("word_sid", TYPE_INT),       # W ConceptRef space（cue slot 落位词）
    ("word_lid", TYPE_INT),       # W ConceptRef local
    ("rel_kind", TYPE_INT),       # REL_CAUSES / REL_SUBSET 等（int enum·relation-specific 键）
    ("sample_sid", TYPE_INT),     # input_root ConceptRef space（distinct tally sample 去重维）
    ("sample_lid", TYPE_INT),     # input_root ConceptRef local
]
_STRUCTURE_MATCH_COUNT_INDEXES = [
    # (W, R) 主查询（promote _structure_match_ok 读 distinct count）
    ("space_id", "word_sid", "word_lid", "rel_kind"),
    # (W, R, input_root) 去重唯一键（append-only 幂等·防同 sample 重计）
    ("space_id", "word_sid", "word_lid", "rel_kind", "sample_sid", "sample_lid"),
]


def register_structure_match_count(backend: StorageBackend) -> None:
    """注册 structure_match_count 扩展表（core=False·APPEND_ONLY·启动/用前调·幂等）。

    APPEND_ONLY（非 experience_count 的 MUTABLE_MONOTONE）：本表 insert only·无计数器列·distinct
    count=行数·重放幂等（同 input_root 重见=skip·非 +=1）·天然抗刷数。
    """
    register_extension_table(backend, STRUCTURE_MATCH_COUNT_TABLE,
                             _STRUCTURE_MATCH_COUNT_COLUMNS,
                             disc.DISC_APPEND_ONLY, _STRUCTURE_MATCH_COUNT_INDEXES)


def record_structure_match(backend: StorageBackend, *, space_id: int,
                           word_ref: tuple[int, int], rel_kind: int,
                           sample_root: tuple[int, int]) -> bool:
    """记一次 W 落 REALIZES-R-skeleton cue slot（distinct tally sample++）。

    append-only check-then-insert：行 (W,R,input_root) 不存在→insert·返 True（new·首次该 sample for (W,R)）；
    已存在→skip·返 False（幂等·重放同语料/同 input_root 重见不重计）。
    **caller 见 True → 调 record_emergent_relation_signal_shadow 建 D:11 SHADOW 行**（审2 条件2·
    generator 关后唯一创建者·record_emergent_relation_signal_shadow 自身幂等防同 (W,R) 多 sample 重复建边）。

    返 new=True 不代表"首次该 (W,R)"·代表"首次该 (W,R,input_root)"——多个 distinct input_root 各返 True·
    但 SHADOW 创建由 record_emergent_relation_signal_shadow 的 query_from 幂等兜底（同 (W,R) 只一 SHADOW 边）。
    表未注册（bare fixture）→ KeyError 静默 skip·返 False（向后兼容·同 record_base_freq 范式）。
    """
    assert_int(space_id, word_ref[0], word_ref[1], rel_kind,
               sample_root[0], sample_root[1], _where="record_structure_match.args")
    # 审1 L1：space_id 列须 == W 的 ConceptRef space（word_sid）。read 路径用 space_id=word_sid 查·
    # record 存 space_id=caller 传值——两者须恒等（W 在 caller 的语言 space 建·w[0]==ctx.space_id）。
    # 若未来 W 跨 space 建（w[0]≠ctx.space_id）→ read 读不到此 record → _structure_match_ok 恒 False →
    # 静默失效（无报错）。此 assert fail-loud 守不变量（当前生产 tally + 测恒满足）。
    assert space_id == word_ref[0], (
        f"structure_match_count invariant (审1 L1): space_id ({space_id}) must equal word_ref space "
        f"({word_ref[0]}) — W must live in the tally's language space, else reads (keyed on word_sid) "
        f"silently miss and promote gate vacuously fails. doc/重来_对应泛化_结构反推 §3.3.")
    try:
        existing = backend.select(STRUCTURE_MATCH_COUNT_TABLE, where={
            "space_id": space_id,
            "word_sid": word_ref[0], "word_lid": word_ref[1],
            "rel_kind": rel_kind,
            "sample_sid": sample_root[0], "sample_lid": sample_root[1],
        }, limit=1)
    except KeyError:
        return False   # 表未注册（bare fixture/未注册场景）·向后兼容 skip
    if existing:
        return False   # 幂等：该 (W,R,input_root) 已计·distinct sample 不重计（append-only 抗刷数）
    backend.insert(STRUCTURE_MATCH_COUNT_TABLE, {
        "space_id": space_id,
        "word_sid": word_ref[0], "word_lid": word_ref[1],
        "rel_kind": rel_kind,
        "sample_sid": sample_root[0], "sample_lid": sample_root[1],
    })
    return True


def read_structure_match_count(backend: StorageBackend, *, space_id: int,
                               word_ref: tuple[int, int], rel_kind: int) -> int:
    """读 (W,R) distinct tally sample count（= 该 (W,R) 行数·每行=1 distinct input_root）。

    无行/表未注册→0（冷启动·promote 闸 _structure_match_ok 判 <K→False）。promote 用此判 ≥K。
    """
    assert_int(space_id, word_ref[0], word_ref[1], rel_kind,
               _where="read_structure_match_count.args")
    try:
        rows = backend.select(STRUCTURE_MATCH_COUNT_TABLE, where={
            "space_id": space_id,
            "word_sid": word_ref[0], "word_lid": word_ref[1],
            "rel_kind": rel_kind,
        })
    except KeyError:
        return 0   # 表未注册→0（向后兼容）
    return len(rows)


def read_structure_match_per_rel(backend: StorageBackend, *, space_id: int,
                                 word_ref: tuple[int, int]) -> dict[int, int]:
    """读 W 各 rel_kind 的 distinct tally sample count（specificity gate 用·审1 CONDITION 1）。

    返 {rel_kind: count}（W 落各 REALIZES-R-skeleton cue slot 的 distinct sample 分布）。
    specificity：W' promote R 须 count(W,R) 显著 > Σ count(W,other R)（W' 特异 R·非通用连接词·
    过滤"和"误晋：和 落多 R skeleton 各计→不特异→不晋；引发 落主 R skeleton→特异→晋）。
    无行/表未注册→{}（caller 判 specificity vacuously pass 或 fail·由 _structure_match_ok 定）。
    """
    assert_int(space_id, word_ref[0], word_ref[1],
               _where="read_structure_match_per_rel.args")
    try:
        rows = backend.select(STRUCTURE_MATCH_COUNT_TABLE, where={
            "space_id": space_id,
            "word_sid": word_ref[0], "word_lid": word_ref[1],
        })
    except KeyError:
        return {}   # 表未注册→{}（向后兼容）
    per_rel: dict[int, int] = {}
    for r in rows:
        rk = r["rel_kind"]
        per_rel[rk] = per_rel.get(rk, 0) + 1
    return per_rel
