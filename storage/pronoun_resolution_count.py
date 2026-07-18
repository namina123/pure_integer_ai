"""storage.pronoun_resolution_count — 指代消解统计台账（B6·方案3 tn+fn 路·§十二.2 三路写）。

pronoun_resolution_count = (pronoun, antecedent) pair-key 指代消解决策计数·指代维学习地基。
  镜像 selection_pref_count / experience_count 范式（MUTABLE_MONOTONE 扩展表·core=False·独立表非边）。

**方案3 三路写**（§十二.2·避 β_arith·判据各走各的不吃 episode 符号）：
  pr_tn   observe 路·决策时（resolve 选 best antecedent·sign-agnostic·不计对错·只记"此决策点出现过"）
           → record_pronoun_resolution_decision 写·resolve_pronoun_occurrence 内 per-occurrence
  pr_fn   失败侧归因路·悬空时（无候选 / θ 不达·J4 dangling·零教师·§九.2 病灶"attribute 给谁"=per-occurrence 落 pronoun）
           → record_pronoun_resolution_dangling 写·self-loop key (pronoun, pronoun)·from==to 表悬空标记
  pr_sn   成功侧教师路·教师采样时（§十一.4 代表性采样·**P2 教师段 defer·建列 default 0·P1 零教师段不写**）

**key 设计**（§九.4 D5·指代二元 (pronoun,antecedent)·不退化单 key）：
  key = (pronoun ConceptRef, antecedent ConceptRef)。
  悬空无 antecedent → self-loop (pronoun, pronoun)·from==to=pronoun·pair 结构不退化（D5 说不退化=不降单 key·
  自环仍是 4 列 pair）·consumer 读时 from==to 即知悬空标记·不计入 antecedent 选择率。
  成功决策 → (pronoun, best_antecedent)·from=pronoun·to=best_antecedent。

**β_arith 规避**（§十二.1 病根=判据来自 episode 末标量·非 key 粒度）：
  pr_tn/pr_fn 都 per-occurrence 决策时写（resolve 内立刻写·不等 episode 末 reward·sign-agnostic·
  独立 episode reward 符号）·rate=pr_fn/pr_tn 不塞 episode 符号 → 避 β_arith（§九.2 B2 数学抵消点）。
  对比 B4/B5 observe_tn/sp_observe_tn 是因既有 e_tn/sp_tn 染 reward 需分离·指代维新表 pr_tn 本身
  observe 路（reward 路是 pr_sn=P2 defer·不染 pr_tn）·不需 observe_mode 切换。

**reward CAUSES-only 防塌柱①**（铁律）：pronoun_resolution_count 是统计台账非 edge reward·
  pr_sn 教师 feed P2 defer（未接 reward_propagate·独立表不进 edge reward 多头）·不进 causes_edges/
  distributed/record_episode_result·reward_propagate.py assert 不动·effective_weight assert 不内。

**consumer**（B0 dim·consumer=dispatch_slot/refers_occurrence·§十二.5 主面件 C slot 级）：
  B6 P1 零教师段生产 consumer = resolve_pronoun_occurrence 自消费（observe 侧·读历史 pr_tn 加候选分·
  gate PRONOUN_RESOLVE_COUNT_MODE 守·reward>0 鲁棒·J4 bool veto 只查 dangling 不查 antecedent 质量）。
  件 C dispatch_slot pronoun scoring defer STEP6。指代维 reward=J4 bool veto（非 graded·与 B4/B5 count
  进 _seed_weight reward 加权不同）·count 不进 reward 公式·consumer 在 observe 侧自消费 + 件 C 生成侧。

铁律：纯整数（pr_tn/pr_fn/pr_sn 全 TYPE_INT·assert_int 守）/ MUTABLE_MONOTONE（表纪律·delta +1 无负）/
  append-only 行级（insert 一次 + update·同 edge 表范式·不动 concept_node 不变量）/ 确定性（bit-identical）/
  单向依赖（L0 storage·L4 refers_occurrence 写·L8 formal_train register·皆向下）/ 不写死（schema 元定义列）。
诚实边界：本表是地基非楼（解指代消解 stats substrate·失败侧归因 per-occurrence 落 pronoun·consumer 自消费已落·
  件 C dispatch_slot defer·pr_sn 教师 P2 defer·stable≠correct "它们→最近 token 可能功能词"接地墙外·
  代词消解结构非墙 vs sense 消歧 #479 真墙·§九.7.6 W2 拆分）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

PRONOUN_RESOLUTION_COUNT_TABLE = "pronoun_resolution_count"


_PRONOUN_RESOLUTION_COUNT_COLUMNS = [
    ("space_id_from", TYPE_INT),
    ("local_id_from", TYPE_INT),
    ("space_id_to", TYPE_INT),     # antecedent ConceptRef space（悬空自环=pronoun space·from==to 表悬空标记）
    ("local_id_to", TYPE_INT),     # antecedent ConceptRef local（悬空自环=pronoun local）
    ("pr_tn", TYPE_INT),           # observe 路·决策时（resolve 选 best antecedent·sign-agnostic·方案3 tn路·per-occurrence·不染 reward）
    ("pr_fn", TYPE_INT),           # 失败侧归因路·悬空时（无候选/θ不达·J4 dangling·零教师·self-loop key·per-occurrence·§九.2 病灶"attribute 给谁"=per-occurrence 落 pronoun）
    ("pr_sn", TYPE_INT),           # 成功侧教师路（§十一.4 代表性采样·P2 教师段 defer·建列 default 0·P1 零教师段不写）
]
_PRONOUN_RESOLUTION_COUNT_INDEXES = [
    ("space_id_from", "local_id_from", "space_id_to", "local_id_to"),   # 主查询 key
    # perf round3（2026-07-13）：read_pronoun_resolution_agg 查 {space_id_from, local_id_from}（2 键·
    # 悬空聚合路径）·同 selection_pref_count 同型全表扫（profile n=4 冷路径未触发但结构在·n=656 放大风险）。
    # 加 2 列前缀索引覆盖 agg 读·bit-identical 同 selection_pref_count（agg sum 聚合 + 重排序无关）。
    ("space_id_from", "local_id_from"),
]


def register_pronoun_resolution_count(backend: StorageBackend) -> None:
    """注册 pronoun_resolution_count 扩展表（core=False·MUTABLE_MONOTONE·启动/用前调·幂等）。"""
    register_extension_table(backend, PRONOUN_RESOLUTION_COUNT_TABLE,
                             _PRONOUN_RESOLUTION_COUNT_COLUMNS,
                             disc.DISC_MUTABLE_MONOTONE, _PRONOUN_RESOLUTION_COUNT_INDEXES)


def read_pronoun_resolution_count(backend: StorageBackend,
                                  pronoun_ref: tuple[int, int],
                                  antecedent_ref: tuple[int, int]
                                  ) -> tuple[int, int, int] | None:
    """读 (pronoun, antecedent) 指代消解 → (pr_sn, pr_tn, pr_fn) | None。

    返 3-tuple（pr_sn, pr_tn, pr_fn）·caller 算 rate=pr_fn/pr_tn 或 pr_sn/pr_tn（不塞 episode 符号·避 β_arith）。
    悬空 self-loop 行（from==to=pronoun）pr_fn>0·pr_tn=0（悬空无决策）·consumer 读时 from==to 即知悬空标记。
    无行=冷启动（未 observe 此 pair·caller 判 None→count 0）。表未注册→None（向后兼容·同 read_selection_pref_count 范式）。
    """
    sid_p, lid_p = pronoun_ref
    sid_a, lid_a = antecedent_ref
    assert_int(sid_p, lid_p, sid_a, lid_a, _where="read_pronoun_resolution_count.args")
    try:
        rows = backend.select(PRONOUN_RESOLUTION_COUNT_TABLE, where={
            "space_id_from": sid_p, "local_id_from": lid_p,
            "space_id_to": sid_a, "local_id_to": lid_a,
        }, limit=1)
    except KeyError:
        return None   # 表未注册（caller 未 register_pronoun_resolution_count）·向后兼容
    if not rows:
        return None   # 冷启动（该 (pronoun, antecedent) pair 未落过决策）·caller 判 None
    r = rows[0]
    return (r["pr_sn"], r["pr_tn"], r["pr_fn"])


def read_pronoun_resolution_agg(backend: StorageBackend,
                                pronoun_ref: tuple[int, int]) -> tuple[int, int, int]:
    """读 pronoun 的所有 antecedent 行聚合 → (sum_pr_sn, sum_pr_tn, sum_pr_fn)。

    聚合 pronoun 的所有 (pronoun, *) 行（不同 antecedent + 悬空 self-loop 行）·sum 各列。
    consumer（resolve 自消费）读 sum_pr_tn 判 pronoun 总决策频次·反测 harness 读判 dim-key 分化。
    冷启动（无行/表未注册）→ (0, 0, 0)（caller 判 pr_agg=0 → 无历史 → 不影响候选排序 bit-identical）。
    确定性：按 (space_id_to, local_id_to) 升序遍历聚合（NodeRef 升序 tiebreak·bit-identical·同 read_selection_pref_agg 范式）。
    """
    sid_p, lid_p = pronoun_ref
    assert_int(sid_p, lid_p, _where="read_pronoun_resolution_agg.args")
    try:
        rows = backend.select(PRONOUN_RESOLUTION_COUNT_TABLE, where={
            "space_id_from": sid_p, "local_id_from": lid_p,
        })
    except KeyError:
        return (0, 0, 0)   # 表未注册·向后兼容
    if not rows:
        return (0, 0, 0)   # 冷启动（该 pronoun 无任何决策行）→ pr_agg=0 → 无历史不影响候选
    rows_sorted = sorted(rows, key=lambda r: (r["space_id_to"], r["local_id_to"]))
    sum_sn = sum(r["pr_sn"] for r in rows_sorted)
    sum_tn = sum(r["pr_tn"] for r in rows_sorted)
    sum_fn = sum(r["pr_fn"] for r in rows_sorted)
    return (sum_sn, sum_tn, sum_fn)


def record_pronoun_resolution_decision(backend: StorageBackend, *,
                                       pronoun_ref: tuple[int, int],
                                       antecedent_ref: tuple[int, int]) -> None:
    """记一次指代消解决策（observe 调·pr_tn++·方案3 tn路·sign-agnostic·per-occurrence·reward 不调此函数）。

    首次：insert(pr_tn=1, pr_fn=0, pr_sn=0)。
    已存在：pr_tn += 1（MUTABLE_MONOTONE·delta 固定 +1·无负·表纪律双保险）。
    pr_tn 是 observe 路纯计数（sign-agnostic·独立 episode reward 符号·per-occurrence 决策时写·
      避 β_arith rate 塌缩·consumer PRONOUN_RESOLVE_COUNT_MODE 守读加候选分）。
    pr_fn/pr_sn observe 路不碰（pr_fn 由 record_pronoun_resolution_dangling 写·pr_sn P2 教师段 defer）。
    表未注册（bare fixture/未注册场景）→ KeyError 静默 skip（向后兼容·镜像 record_selection_pref_cooccur 范式）。
    """
    sid_p, lid_p = pronoun_ref
    sid_a, lid_a = antecedent_ref
    assert_int(sid_p, lid_p, sid_a, lid_a, _where="record_pronoun_resolution_decision.args")
    try:
        existing = backend.select(PRONOUN_RESOLUTION_COUNT_TABLE, where={
            "space_id_from": sid_p, "local_id_from": lid_p,
            "space_id_to": sid_a, "local_id_to": lid_a,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if not existing:
        backend.insert(PRONOUN_RESOLUTION_COUNT_TABLE, {
            "space_id_from": sid_p, "local_id_from": lid_p,
            "space_id_to": sid_a, "local_id_to": lid_a,
            "pr_tn": 1, "pr_fn": 0, "pr_sn": 0,
        })
        return
    backend.update(PRONOUN_RESOLUTION_COUNT_TABLE, where={
        "space_id_from": sid_p, "local_id_from": lid_p,
        "space_id_to": sid_a, "local_id_to": lid_a,
    }, set_={"pr_tn": ("+=", 1)})


def record_pronoun_resolution_dangling(backend: StorageBackend, *,
                                       pronoun_ref: tuple[int, int]) -> None:
    """记一次指代悬空失败归因（observe 调·pr_fn++·self-loop key (pronoun, pronoun)·零教师·per-occurrence·§九.2 病灶"attribute 给谁"=per-occurrence 落 pronoun）。

    首次：insert(pr_tn=0, pr_fn=1, pr_sn=0)（self-loop from==to=pronoun·悬空标记）。
    已存在：pr_fn += 1（MUTABLE_MONOTONE·delta 固定 +1·无负）。
    pr_fn 是失败侧归因（J4 dangling·零教师·per-occurrence 决策时写·独立 episode reward 符号·避 β_arith）。
    self-loop key (pronoun, pronoun)·from==to 表悬空标记·consumer 读时 from==to 即知悬空·不计入 antecedent 选择率。
    pr_tn/pr_sn 失败侧不碰（pr_tn 由 record_pronoun_resolution_decision 写·pr_sn P2 教师段 defer）。
    表未注册（bare fixture/未注册场景）→ KeyError 静默 skip（向后兼容）。

    铁律：reward CAUSES-only 防塌柱①（本函数是独立表写·不进 causes_edges/distributed/record_episode_result）。
    诚实边界：pr_fn 是悬空频次非语义失败归因（"该 pronoun 悬空次数"·stable≠correct·#479 墙·
      代词消解结构非墙 vs sense 消歧 #479 真墙）。
    """
    sid_p, lid_p = pronoun_ref
    assert_int(sid_p, lid_p, _where="record_pronoun_resolution_dangling.args")
    try:
        existing = backend.select(PRONOUN_RESOLUTION_COUNT_TABLE, where={
            "space_id_from": sid_p, "local_id_from": lid_p,
            "space_id_to": sid_p, "local_id_to": lid_p,   # self-loop·悬空标记 from==to=pronoun
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if not existing:
        backend.insert(PRONOUN_RESOLUTION_COUNT_TABLE, {
            "space_id_from": sid_p, "local_id_from": lid_p,
            "space_id_to": sid_p, "local_id_to": lid_p,   # self-loop·悬空标记
            "pr_tn": 0, "pr_fn": 1, "pr_sn": 0,
        })
        return
    backend.update(PRONOUN_RESOLUTION_COUNT_TABLE, where={
        "space_id_from": sid_p, "local_id_from": lid_p,
        "space_id_to": sid_p, "local_id_to": lid_p,
    }, set_={"pr_fn": ("+=", 1)})
