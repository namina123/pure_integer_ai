"""storage.sense_candidates — 多义 sense 候选台账（刀6 件7·sense 多义管线修通·摄入侧地基）。

sense_candidates = (token surface_hash, sense ConceptRef) 1:N 多义候选台账·sense 消歧（polysemy
  disambiguation）的摄入侧地基。镜像 selection_pref_count / experience_count 范式（MUTABLE_MONOTONE
  扩展表·core=False·独立表非 concept_node 加列）。

**设计源头**（doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀6 + 侦察核证）：刀0-5 学习放开 6 刀·刀6 件7
  sense 消歧原设计为 #479 墙（教师定义权·真消歧断奶后内生判据缺）。侦察（Explore agent·证据链完整）揭示
  MultiRef 管线实现上完全断裂（纸面闭合）——observe:111 塌缩 refs[0]·multi_refs 死列表·生成侧 activate_candidates
  是 concept→词形非选 sense·recognize grep sense 零命中。**用户拍板 Option B**（修通 MultiRef 域内管线）。
  本表 = 摄入侧写真（observe 持久化多 sense·非死列表）+ 理解侧 recognize clone 选 sense 的候选源。

**key 设计**（决断 1·解 N10）：key = surface_hash（Hasher.h63 of token surface·非 ConceptRef）。
  concept_index.ensure 同 surface 返同 ref（content_hash dedup）·两 sense 不能用同 token ref（"老鼠"两 ensure
  撞同 ref）·用 surface_hash 解耦 token 标识与 sense ref。sense ref 用不同 surface ensure（"动物老鼠"/"鼠标"）
  得不同 ref。一 token 多 sense → 多行（同 surface_hash·不同 sense ref）。

**消歧插入点**（决断 2·核证）：理解侧 `_discover_and_recognize_lang_structures` recognize_roots（formal_train）
  非 observe 非 dispatch。observe MultiRef 塌缩路径保留（PRECEDES 结构序·bit-identical）——observe MultiRef
  是 theater 死码·**本表才是真摄入侧**。dispatch 是 concept→词形选词形非选 sense（refers_to.py:5 docstring
  "消歧在生成侧" 错·刀6 修正为"消歧在理解侧 recognize"）。

**两源同表**（镜像 experience_count / selection_pref_count）：
  base_count  通识先验（append-only·boot 种·录放层教师/sense_facts 文件注入·reward 不调·首版=1）
  sc_sn       经验成功数（MUTABLE_MONOTONE·reward>0 feed·**刀6 首版 defer feed·列预留=0**）
  sc_tn       token 出现总数（observe 写·段内该 token 出现 +1·不论选哪 sense·刀6 首版唯一 observe 写路径）

**reward CAUSES-only 防塌柱①**（铁律·刀6 首版守）：sense_candidates 是统计台账非 edge reward·sc_sn reward feed
  defer S4（随 PR docking 落）·不进 causes_edges/distributed/record_episode_result·reward_propagate.py:131
  assert 不动·effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}·sense_candidates 不内。

**§8.5 边 schema 不预留乘子字段**（铁律）：sense_candidates 是独立表非边·不挂乘子·PR dock defer S4。

**#479 墙**（诚实边界·doc §5 刀6 + §6）：reward 只标记多义不定义·"包含"消歧到 ⊂ vs mereology vs 属性 = 语义判据·
  系统无闭式判定·真墙（同钥匙③相2 含义命中墙）。本表给候选 + 结构选优（IS_A 共祖 / collide_score 共现）·
  非语义消歧·共现也无法区分时撞墙·code 标注 reward_propagate.py:34 + promote.py:18-19 不动。

铁律：纯整数（surface_hash/sense_ref/sc_*全 TYPE_INT·assert_int 守）/ MUTABLE_MONOTONE（表纪律·delta +1 无负）/
  append-only 行级（insert 一次 + sc_tn update·同 edge 表范式）/ 确定性（bit-identical·Hasher 固定种子·sorted
  NodeRef 升序）/ 单向依赖（L0 storage·crosscut hasher/int_blocker·L4 observe/sense_lookup_hook 写·L8
  formal_train register/boot·皆向下·不 import cognition）/ 不写死（schema 元定义列·计数器非语义规则）。
诚实边界：本表是地基非楼（解 sense 多义摄入侧写真 + 理解侧 clone 候选源·PR dock/sc_sn feed/多 token 笛卡尔积
  全 defer·反 theater 用 IS_A 共祖 + collide_score 结构选优非语义消歧·stable≠correct "老鼠"数据见过就高
  count 接地墙外·#479 墙不破）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

SENSE_CANDIDATES_TABLE = "sense_candidates"

# surface_hash 固定种子（跨 run 确定·bit-identical·caller 经 sense_surface_hash 算·不私自复算）
_SENSE_SURFACE_HASHER = Hasher("sense_candidates.surface")


def sense_surface_hash(surface: str) -> int:
    """token surface → 整数 hash（固定种子·跨 run 确定·bit-identical）。

    sense_candidates 表 key 用此 hash（非 ConceptRef·解 ensure 幂等撞 ref·决断 1）。
    caller（observe record / boot bootstrap / recognize clone 反查）统一调此函数·hash 算法只一处。
    """
    return _SENSE_SURFACE_HASHER.h63(surface)


_SENSE_CANDIDATES_COLUMNS = [
    ("space_id", TYPE_INT),
    ("surface_hash", TYPE_INT),    # token surface 的 hash（sense_surface_hash·非 ConceptRef·解 ensure 幂等）
    ("sense_sid", TYPE_INT),       # sense ConceptRef space（boot ensure 的 sense surface·不同 sense 不同 ref）
    ("sense_lid", TYPE_INT),       # sense ConceptRef local
    ("base_count", TYPE_INT),      # 通识先验（append-only·boot 种·reward 不调·首版=1）
    ("sc_sn", TYPE_INT),           # 经验成功数（reward>0 feed·刀6 首版 defer·列预留=0）
    ("sc_tn", TYPE_INT),           # token 出现总数（observe 写·段内该 token 出现 +1·刀6 首版唯一 observe 写路径）
]
_SENSE_CANDIDATES_INDEXES = [
    ("space_id", "surface_hash"),   # 主查询 key（一 token 多 sense·各 sense 一行）
]


def register_sense_candidates(backend: StorageBackend) -> None:
    """注册 sense_candidates 扩展表（core=False·MUTABLE_MONOTONE·启动/用前调·幂等）。"""
    register_extension_table(backend, SENSE_CANDIDATES_TABLE,
                             _SENSE_CANDIDATES_COLUMNS,
                             disc.DISC_MUTABLE_MONOTONE, _SENSE_CANDIDATES_INDEXES)


def read_sense_candidates(backend: StorageBackend, space_id: int,
                          surface_hash: int) -> list[tuple[tuple[int, int], int, int, int]]:
    """读 token surface 的 sense 候选 → [(sense_ref, base_count, sc_sn, sc_tn), ...]（按 sense_ref NodeRef 升序）。

    空列表=冷启动（该 token 无 sense 候选·caller 走单 ensure(tok) 原路径 bit-identical）。
    表未注册→[]（环境未启 sense_candidates 台账·向后兼容·同 read_experience_count 范式）。
    确定性：按 (sense_sid, sense_lid) 升序（NodeRef 升序 tiebreak·bit-identical·同 selection_pref_count 范式）。
    """
    assert_int(space_id, surface_hash, _where="read_sense_candidates.args")
    try:
        rows = backend.select(SENSE_CANDIDATES_TABLE, where={
            "space_id": space_id, "surface_hash": surface_hash,
        })
    except KeyError:
        return []   # 表未注册（caller 未 register_sense_candidates）·向后兼容
    out: list[tuple[tuple[int, int], int, int, int]] = []
    for r in rows:
        sense_ref = (r["sense_sid"], r["sense_lid"])
        out.append((sense_ref, r["base_count"], r["sc_sn"], r["sc_tn"]))
    out.sort(key=lambda x: x[0])   # NodeRef 升序（确定性 tiebreak·bit-identical）
    return out


def record_sense_token_seen(backend: StorageBackend, space_id: int,
                            surface_hash: int, sense_ref: tuple[int, int]) -> None:
    """记一次段内 token 出现（observe 调·sc_tn++·镜像 record_selection_pref_cooccur 范式·reward 不调此函数）。

    首次：insert(base_count=0, sc_sn=0, sc_tn=1)。
    已存在：sc_tn += 1（MUTABLE_MONOTONE·delta 固定 +1·无负·表纪律双保险）。
    base_count append-only 永不调（observe 路径不碰 base_count·sc_sn reward feed defer S4）。
    表未注册（bare fixture/未注册场景）→ KeyError 静默 skip（向后兼容·镜像 record_selection_pref_cooccur 范式）。
    """
    sid, lid = sense_ref
    assert_int(space_id, surface_hash, sid, lid, _where="record_sense_token_seen.args")
    try:
        existing = backend.select(SENSE_CANDIDATES_TABLE, where={
            "space_id": space_id, "surface_hash": surface_hash,
            "sense_sid": sid, "sense_lid": lid,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if not existing:
        backend.insert(SENSE_CANDIDATES_TABLE, {
            "space_id": space_id, "surface_hash": surface_hash,
            "sense_sid": sid, "sense_lid": lid,
            "base_count": 0, "sc_sn": 0, "sc_tn": 1,
        })
        return
    backend.update(SENSE_CANDIDATES_TABLE, where={
        "space_id": space_id, "surface_hash": surface_hash,
        "sense_sid": sid, "sense_lid": lid,
    }, set_={"sc_tn": ("+=", 1)})


def bootstrap_sense_candidates(backend: StorageBackend, concept_index,
                               sense_pairs: list[tuple[str, list[str]]],
                               *, space_id: int) -> int:
    """sense_candidates 批量 boot 种（刀6 片2·word→[sense1,sense2,...] surface 对 → ensure → 种 base_count）。

    入参 sense_pairs：(word_surface, [sense1_surface, sense2_surface, ...]) 列表·caller 不依赖语料 token 切片
      （boot 时种·早于 observe）·来自 sense_facts 文件 loader（§8.1c 来源① EPI_STRUCTURED 合规·类比刀0 IS_A）。

    每 (word, [senses])：surface_hash = sense_surface_hash(word) → 每 sense concept_index.ensure（TIER_PRIMARY·
      NODE_CONCEPT·不同 surface 不同 ref·解 N10）→ 查 (space_id, surface_hash, sense_sid, sense_lid) 已存在 skip
      （first-write-wins·base_count append-only 不重写·幂等防 resume 跨 run corrupt）→ insert base_count=1/sc_sn=0/sc_tn=0。

    **无文件零副作用硬守（bit-identical·P0·镜像 bootstrap_is_a_edges:119-120）**：sense_pairs 空 → 立即
      return 0·**绝不调 concept_index.ensure / select / insert**（无 ZERO_AI_LOCAL_DIR → resolve_sense_facts
      返 [] → 图与不接刀6 bit-identical·退化链 5 步·plan 决断 5）。

    返种行数。boot 写 base_count（先验）·sc_tn 留 0（sc_tn 是 observe 频次·observe 路径自写·boot 不碰）。

    铁律：纯整数（surface_hash + sense_ref + sc_*全整）/ 不写死（surface 来自外部文件·本函数只机制非语义）/
      §8.1c（sense_facts 来源①·统计台账非关系边·不涉三死刑）/ bit-identical（空 pairs 零副作用 + 幂等 skip）。
    诚实边界：surface/sense 真伪 = 外部数据责任（接地墙）·sense_candidates 不接 reward（守 CAUSES-only）·
      本表是地基非楼（理解侧 clone 消费者 read_sense_candidates·反 theater 用 IS_A 共祖选优非语义消歧·#479 墙）。
    """
    if not sense_pairs:
        return 0   # P0·无文件零副作用硬守（不调 ensure/select/insert·CI/生产 default bit-identical）
    assert_int(space_id, _where="bootstrap_sense_candidates.args")
    from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
    n = 0
    for word_surf, senses in sense_pairs:
        if not senses:
            continue   # 无 sense 候选跳（守确定性·空 senses 不种）
        sh = sense_surface_hash(word_surf)
        for sense_surf in senses:
            sense_ref = concept_index.ensure(
                sense_surf, space_id=space_id,
                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
            sid, lid = sense_ref
            # 幂等 skip（first-write-wins·base_count append-only 不重写·防 resume 跨 run corrupt）
            try:
                existing = backend.select(SENSE_CANDIDATES_TABLE, where={
                    "space_id": space_id, "surface_hash": sh,
                    "sense_sid": sid, "sense_lid": lid,
                }, limit=1)
            except KeyError:
                return 0   # 表未注册（caller 未 register）·向后兼容·已种 0 行
            if existing:
                continue   # 同 (word, sense) 已种→skip（幂等）
            backend.insert(SENSE_CANDIDATES_TABLE, {
                "space_id": space_id, "surface_hash": sh,
                "sense_sid": sid, "sense_lid": lid,
                "base_count": 1, "sc_sn": 0, "sc_tn": 0,
            })
            n += 1
    return n
