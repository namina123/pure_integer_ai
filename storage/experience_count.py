"""storage.experience_count — 概念维经验计数台账（三把钥匙公共 0→1 根因·§八-bis·阶段1 地基）。

experience_count = 边级 sn/tn 的**概念维对偶聚合**（非新机制）。同一 reward>0 episode 信号：
  - 边级（已活·H4 闭环·reward_propagate→edge_store.record_episode_result）落点=单条 CAUSES 边。
  - 概念维（本表）落点=单概念——"这概念在它出现的所有成功 episode 里攒了几次"。
→ 不需"发明概念级计数"·只需在已通环里多落一笔到概念行（修正分析二 §十一）。

**为何独立表非 concept_node 加列**（路 A·指引点1·4 先例）：concept_node 是核心 APPEND_ONLY 表（discipline.py:54
CORE_TABLES）·塞经验=污染核心永久层·违"概念纯净化"（legacy 概念阻断.md:42-48 已判路 A）。独立 core=False
扩展表（同 op_confidence/concept_identity/composes_attr/weaning_calibration 范式）不碰 concept_node 不变量。
experience_count 是 op_confidence（算子域概念级计数）的 **L 域对偶**。

**两源同表**（指引点4·用户认 2026-07-04）：
  base_freq  通识先验频次（append-only·录放层教师注入·reward 不调·镜像 edge_store.base_strength）
  e_sn/e_tn  经验成功/总数（MUTABLE_MONOTONE·reward feed·镜像 edge_store.sn/tn + op_confidence R1 符号）
消费者只读 effective_freq = base_freq + e_tn（总相遇频次·**两源同加**：通识基线 base_freq + 经验积累 e_tn·
**非列**·镜像边级 effective_weight=strength×rate 用 strength【经验维·可变】非 base_strength【先验·不变】
的对偶——经验维对偶体现在 e_tn 随 reward feed 增 / base_freq append-only 不变·两源永远共存）。
冷启动（经验≈0）effective_freq≈base_freq 通识主导·断奶后 base+exp 微调·两源永远共存不替换。

**reward CAUSES-only 真墙铁律**（修正分析五·P0-2）：reward 永走 CAUSES-only（防塌柱①·逐边塌缩有意防·非丢弃）·
**非因果经验走 experience_count 概念维对偶（非 reward 多头）**。本表是概念维对偶落点·不接 reward 多头·
不破防塌柱①。feed（reward>0 episode 聚合到概念行）在阶段2 propagate_reward 落点①·本表只提供存储+读写地基。

R1 episode 符号契约（同 reward_propagate.py:122-153 + op_confidence.py:19）：reward>0→e_sn++&e_tn++ /
reward≤0→e_tn++ only（e_sn 单调不降·率自然降·非 e_sn--）。

**复合 key 节奏**（指引点2·用户认·两刀）：第一刀单 key (space_id,local_id)（ctx_code/speaker_code 恒 0·
闭合 attractor 词终止阶段3）·第二刀复合 key (space_id,local_id,ctx_code,speaker_code)（解锁反讽固化层阶段6）。
ctx_code=domain<<24|modality<<16|task<<8|intent_type 位打包纯整数（ctx_tag 四维·reward_propagate.py:88-89）·
speaker_code=speaker int id。单 key 是退化（ctx=0,speaker=0）向后兼容·本阶段（第一刀）恒 0。

铁律：纯整数（base_freq/e_sn/e_tn/ctx_code/speaker_code 全 int·assert_int 守）/ MUTABLE_MONOTONE（表纪律·
delta 固定 +1 无负·表纪律双保险·base_freq append-only 由"record_base_freq 不 update 已存在行"公约守）/
append-only 行级（insert 一次 + e_sn/e_tn update·同 edge 表范式·不动 concept_node 不变量）/ 确定性（bit-identical）/
单向依赖（L0 storage·L8 formal_train 写·L4 ConceptGraph 读·皆向下）/ 不写死（schema 元定义列·计数器非语义规则）。
诚实边界：本表是地基非楼（解漂移补救1/attractor 越界②/freq 乘子/反讽固化①的信号源·不解钥匙③墙/钥匙①/静默学习/
记忆空间本体/世界态反讽/单次判定）·详见 doc/重来_experience_count落地设计指引.md §五。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

EXPERIENCE_COUNT_TABLE = "experience_count"

# 复合 key 退化默认（第一刀单 key·ctx/speaker 恒 0·第二刀阶段6 启用）
DEFAULT_CTX_CODE = 0
DEFAULT_SPEAKER_CODE = 0


def pack_ctx_code(domain: int, modality: int, task: int, intent_type: int) -> int:
    """ctx_tag 四维位打包为 ctx_code 单维纯整数（复合 key 第二刀·§点2·bit-identical）。

    公式 domain<<24 | modality<<16 | task<<8 | intent_type（schema 注释 :30 已有公式）。
    各维 ≤255（8 bit·types.py 枚举守：domain≤4 / modality≤7 / task=0 defer / intent_type≤3·不溢出污染相邻维）。
    task=0 defer（§十一6Q·InputPayload 无 task 字段·episode.py _ctx_tag 占位 0）。
    全 0 → 0 = DEFAULT_CTX_CODE（单 key 退化·第一刀恒 0·向后兼容）。
    caller：episode.py + reward_propagate.py 共享 import（ctx_tag 四维 → ctx_code·L5 cognition → L0 storage 向下）。
    """
    assert_int(domain, modality, task, intent_type, _where="pack_ctx_code.args")
    assert 0 <= domain <= 255 and 0 <= modality <= 255 \
        and 0 <= task <= 255 and 0 <= intent_type <= 255, (
        f"pack_ctx_code 各维须 ≤255（8 bit·防溢出污染相邻维）·got "
        f"domain={domain} modality={modality} task={task} intent_type={intent_type}")
    return (domain << 24) | (modality << 16) | (task << 8) | intent_type


_EXPERIENCE_COUNT_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("ctx_code", TYPE_INT),       # 语境位打包（第二刀启用·第一刀恒 0）
    ("speaker_code", TYPE_INT),   # speaker id（第二刀启用·第一刀恒 0）
    ("base_freq", TYPE_INT),      # 通识频次先验（append-only·录放层注入·reward 不调）
    ("e_sn", TYPE_INT),           # 经验成功数（MUTABLE_MONOTONE·reward>0 episode feed）
    ("e_tn", TYPE_INT),           # 经验总数（可升·任何参与 episode feed）
    ("observe_tn", TYPE_INT),     # 决策时 observe 计数（sign-agnostic·方案3 tn路 B4 β_arith 修法·dag_path add_active + attractor add_seed 写·reward 不调）
]
_EXPERIENCE_COUNT_INDEXES = [
    ("space_id", "local_id"),                                  # 第一刀单 key 主查询
    ("space_id", "local_id", "ctx_code", "speaker_code"),      # 第二刀复合 key
]


def register_experience_count(backend: StorageBackend) -> None:
    """注册 experience_count 扩展表（core=False·MUTABLE_MONOTONE·启动/用前调·幂等）。"""
    register_extension_table(backend, EXPERIENCE_COUNT_TABLE,
                             _EXPERIENCE_COUNT_COLUMNS,
                             disc.DISC_MUTABLE_MONOTONE, _EXPERIENCE_COUNT_INDEXES)


def _read_count_row(backend: StorageBackend, ref: tuple[int, int],
                    ctx_code: int, speaker_code: int) -> dict | None:
    """读单桶行 raw dict | None（read_experience_count / read_effective_freq 共享·避免 select 重复）。

    无行→None·表未注册→None（caller 未 register·向后兼容·同 read_op_confidence 范式）。
    """
    sid, lid = ref
    try:
        rows = backend.select(EXPERIENCE_COUNT_TABLE, where={
            "space_id": sid, "local_id": lid,
            "ctx_code": ctx_code, "speaker_code": speaker_code,
        }, limit=1)
    except KeyError:
        return None   # 表未注册（caller 未 register_experience_count）·向后兼容
    return rows[0] if rows else None   # 无行→None（冷启动）


def read_experience_count(backend: StorageBackend,
                          ref: tuple[int, int], *,
                          ctx_code: int = DEFAULT_CTX_CODE,
                          speaker_code: int = DEFAULT_SPEAKER_CODE,
                          observe_mode: bool = False
                          ) -> tuple[int, int, int] | None:
    """读概念经验计数 → (base_freq, e_sn, tn) | None（审计根治 [严重-2] 加 observe_mode 参）。

    无行=冷启动（未 feed 未注入·caller 判 None→effective_freq=0）。表未注册→None
    （环境未启 experience_count 台账·向后兼容·同 read_op_confidence 范式）。

    **审计根治 [严重-2]**：加 observe_mode 参·镜像 read_effective_freq observe_mode 切法。
      - observe_mode=False（默认·gate OFF·既有 bit-identical）→ 返 (base_freq, e_sn, e_tn)（e_tn reward episode 计数·β_arith 染）
      - observe_mode=True（gate FREQ_OBSERVE_MODE ON）→ 返 (base_freq, e_sn, observe_tn)（observe_tn 决策时计数·
        sign-agnostic·独立 reward 符号·避 β_arith·caller promote _experience_ok 用·success_rate = e_sn/(e_sn+observe_tn)
        rate 不恒 1/2·缓解 β_arith·不违 promote 三重②因 _experience_ok 是 D:11 专用闸非 _reward_ok 边级）。
    base_freq/e_sn 行为不变（两源永远共存·镜像 read_effective_freq observe_mode 仅切经验频次源）。
    """
    sid, lid = ref
    assert_int(sid, lid, ctx_code, speaker_code,
               _where="read_experience_count.args")
    r = _read_count_row(backend, ref, ctx_code, speaker_code)
    if r is None:
        return None
    tn = r["observe_tn"] if observe_mode else r["e_tn"]
    return (r["base_freq"], r["e_sn"], tn)


def read_effective_freq(backend: StorageBackend,
                        ref: tuple[int, int], *,
                        ctx_code: int = DEFAULT_CTX_CODE,
                        speaker_code: int = DEFAULT_SPEAKER_CODE,
                        observe_mode: bool = False) -> int:
    """消费者读 effective_freq = base_freq + 经验频次（总相遇频次·两源分桶同加·§点4）。

    经验频次源由 observe_mode 选（方案3 tn路·B4 β_arith 修法·gate FREQ_OBSERVE_MODE 守·caller 传参）：
      - observe_mode=False（gate OFF·既有 bit-identical）→ base_freq + e_tn（reward episode 计数·β_arith 塌缩·key 再细无效）。
      - observe_mode=True（gate ON）→ base_freq + observe_tn（决策时计数·sign-agnostic·独立 episode reward 符号·跨 episode 分化）。
    β_arith 病：reward>0 episode 同比 e_sn++&e_tn++ 致参与 concept 全同 e_tn·w_freq 塌缩·
    observe_tn 决策时写（dag_path add_active + attractor add_seed·非 episode 后批量）替 e_tn 作 w_freq 源。
    第二刀桶分离（守通识基线·防混淆频次·阶段6）：
      - base_freq 永从 (0,0) 通识桶读（context+speaker agnostic·record_base_freq 恒写 (0,0)·
        _inject_base_freq 在 formal_train stage 循环后跑恒默认 0 桶 → base_freq 只存 (0,0)）。
      - e_tn/observe_tn 从当前 (ctx_code,speaker_code) 经验桶读（reward feed / observe 写按当前 ctx 分桶）。
    退化（ctx=0 且 speaker=0）：base+经验 同一 (0,0) 行 = 同行读 bit-identical（一次 select·
      既有热路径零性能退化·第一刀/直接 record 调用方全过）。
    无行/表未注册→0（冷启动·消费者按 0 处理·attractor 词终止 θ_freq 不触发 / freq 乘子 0 贡献）。
    observe_mode 仅切经验频次源·base_freq 行为不变（两源永远共存·镜像边级 effective_weight 用 strength 的对偶）。
    """
    sid, lid = ref
    assert_int(sid, lid, ctx_code, speaker_code,
               _where="read_effective_freq.args")
    # 退化单 key 短路：ctx=0 且 speaker=0 → 同行读 base+经验（一次 select·既有路径零退化）
    if ctx_code == DEFAULT_CTX_CODE and speaker_code == DEFAULT_SPEAKER_CODE:
        r = _read_count_row(backend, ref, DEFAULT_CTX_CODE, DEFAULT_SPEAKER_CODE)
        if r is None:
            return 0
        freq_col = r["observe_tn"] if observe_mode else r["e_tn"]
        return r["base_freq"] + freq_col
    # 复合 key 非 0 桶：base 从 (0,0) 通识桶（恒 speaker-agnostic·为 #495 正确性）·经验从当前 (ctx,speaker) 经验桶
    base_r = _read_count_row(backend, ref, DEFAULT_CTX_CODE, DEFAULT_SPEAKER_CODE)
    base_freq = base_r["base_freq"] if base_r is not None else 0
    exp_r = _read_count_row(backend, ref, ctx_code, speaker_code)
    if exp_r is None:
        return base_freq
    freq_col = exp_r["observe_tn"] if observe_mode else exp_r["e_tn"]
    return base_freq + freq_col


def preload_effective_freq(backend: StorageBackend, *, ctx_code: int = DEFAULT_CTX_CODE,
                           speaker_code: int = DEFAULT_SPEAKER_CODE,
                           observe_mode: bool = False):
    """批量化预加载 → 返 lookup(ref)->int（镜像 read_effective_freq·dag_path_step 入口调一次）。

    perf round3 候选1：word_terminated 每节点 read_effective_freq（生产 ctx≠0 → 2 select/node）→
    dag_path 入口预加载两桶整表 select + dict lookup（O(V) once 替 O(V) per-node select）。
    observe agent + per-item agent 三角坐实 per-item reward 循环遍历全累积图（subgraph_edges == 全快照·
    M5 分页 defer）·item 656 时 4-6 万节点·per-node select 是小→大 n 真瓶颈（O(N²) 常数因子最好啃）。

    **bit-identical 全核证 SAFE**：
    - (sid,lid,ctx,sp) 唯一：record_base_freq/outcome/observe 全 check-then-insert/update upsert
      （select limit=1 查在→update / 不在→insert·单线程无双 insert）→ dict 一值一键 = select limit=1·序无关。
    - 读 pre-dag_path_step 状态：入口调快照全表。每节点 read 先于 own write（word_terminated 读 →
      record_experience_observe 同节点写）·每节点 topo_layers 访一次（Kahn/OI 唯一）·跨节点写不互扰
      （per-node 行）→ 预加载值 = 逐节点 select 当时值。
    - 表未注册（KeyError）→ lookup 恒返 0（镜像 read_effective_freq 无行→0）。
    - gate 稳定：FREQ_OBSERVE_MODE 入口 snapshot（formal_train try/finally 翻·step 内不变）。
    attractor _seed_weight **不**用此 cache（其读须见 post-write·走 live read_effective_freq）。
    """
    if ctx_code == DEFAULT_CTX_CODE and speaker_code == DEFAULT_SPEAKER_CODE:
        bucket = _preload_bucket(backend, DEFAULT_CTX_CODE, DEFAULT_SPEAKER_CODE)
        if bucket is None:
            return lambda ref: 0   # 表未注册 → 恒 0（镜像 read_effective_freq 无行→0）
        def _lookup_degenerate(ref):
            row = bucket.get(ref)
            if row is None:
                return 0
            base_freq, e_tn, observe_tn = row
            freq_col = observe_tn if observe_mode else e_tn
            return base_freq + freq_col
        return _lookup_degenerate
    base_bucket = _preload_bucket(backend, DEFAULT_CTX_CODE, DEFAULT_SPEAKER_CODE)
    exp_bucket = _preload_bucket(backend, ctx_code, speaker_code)
    def _lookup_production(ref):
        base_freq = 0
        if base_bucket is not None:
            base_row = base_bucket.get(ref)
            if base_row is not None:
                base_freq = base_row[0]
        if exp_bucket is None:
            return base_freq
        exp_row = exp_bucket.get(ref)
        if exp_row is None:
            return base_freq
        _, e_tn, observe_tn = exp_row
        freq_col = observe_tn if observe_mode else e_tn
        return base_freq + freq_col
    return _lookup_production


def _preload_bucket(backend: StorageBackend, ctx_code: int,
                    speaker_code: int) -> dict | None:
    """select 整桶 → {(sid,lid): (base_freq, e_tn, observe_tn)}（(sid,lid,ctx,sp) 唯一·一值一键）。

    存 tuple 非 row dict（省内存·只取 read_effective_freq 读的 3 列）。表未注册→None（caller 退化返 0）。
    """
    try:
        rows = backend.select(EXPERIENCE_COUNT_TABLE, where={
            "ctx_code": ctx_code, "speaker_code": speaker_code,
        })
    except KeyError:
        return None
    return {(r["space_id"], r["local_id"]): (r["base_freq"], r["e_tn"], r["observe_tn"])
            for r in rows}


def record_base_freq(backend: StorageBackend, *, ref: tuple[int, int],
                     base_freq: int, ctx_code: int = DEFAULT_CTX_CODE,
                     speaker_code: int = DEFAULT_SPEAKER_CODE) -> None:
    """注入通识频次先验 base_freq（录放层教师录制时调·append-only·reward 不调·镜像 edge_store.add
    写 base_strength=strength 初值）。

    行不存在→insert(base_freq, e_sn=0, e_tn=0)。行已存在→**幂等 skip**（base_freq append-only·
    first-write-wins·reward 路径永不调此函数）。公约：base_freq 在行创建时一次性写入·后续 e_sn/e_tn
    update 不碰 base_freq（同 edge 表 base_strength 不被 record_episode_result 调）。
    表未注册（bare fixture）→ KeyError 静默 skip（向后兼容·同 record_concept_identity 范式）。

    正常流：record_base_freq 在 observe/录制期（阶段1-2）调·先于 reward feed（阶段3）·故创建行时
    base_freq 已就位。异常流（reward feed 先建行 base_freq=0）→ skip·base_freq 留 0（该概念无通识先验·
    诚实降级·断奶后新概念本就无 base_freq 只靠 exp 自积累）。
    """
    sid, lid = ref
    assert_int(sid, lid, base_freq, ctx_code, speaker_code,
               _where="record_base_freq.args")
    try:
        existing = backend.select(EXPERIENCE_COUNT_TABLE, where={
            "space_id": sid, "local_id": lid,
            "ctx_code": ctx_code, "speaker_code": speaker_code,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if existing:
        return   # 幂等：行已存在·base_freq append-only 不重写（first-write-wins）
    backend.insert(EXPERIENCE_COUNT_TABLE, {
        "space_id": sid, "local_id": lid,
        "ctx_code": ctx_code, "speaker_code": speaker_code,
        "base_freq": base_freq, "e_sn": 0, "e_tn": 0, "observe_tn": 0,
    })


def record_experience_outcome(backend: StorageBackend, *, ref: tuple[int, int],
                               reward: int, ctx_code: int = DEFAULT_CTX_CODE,
                               speaker_code: int = DEFAULT_SPEAKER_CODE) -> None:
    """记一次 reward episode 对概念的经验结果（R1 episode 符号·镜像 op_confidence.record_op_outcome）。

    reward>0  → e_sn+=1, e_tn+=1（参与即成功·episode 级·e_sn 单调）
    reward≤0  → e_tn+=1 only（失败计数·e_sn 不降·率自然降·非 e_sn--）
    首次：insert with outcome applied（base_freq=0·e_sn=1 if reward>0 else 0·e_tn=1）·
      同 update 路径从 (base_freq,0,0) 起的终态（无独立 base 注入步·同 op_confidence 无独立 add 步）。
    守 MUTABLE_MONOTONE：delta 固定 +1（无负·MonotoneViolation 不可达·表纪律双保险）。
    base_freq append-only 永不调（reward 路径不碰 base_freq·防塌柱① reward CAUSES-only 铁律）。
    表未注册（bare fixture/未注册场景）→ KeyError 静默 skip（向后兼容·镜像 record_base_freq·三写函数兜底一致）。
    """
    sid, lid = ref
    assert_int(sid, lid, reward, ctx_code, speaker_code,
               _where="record_experience_outcome.args")
    try:
        existing = backend.select(EXPERIENCE_COUNT_TABLE, where={
            "space_id": sid, "local_id": lid,
            "ctx_code": ctx_code, "speaker_code": speaker_code,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture/未注册场景）·向后兼容·镜像 record_base_freq:141-142
    success = reward > 0
    if not existing:
        sn0 = 1 if success else 0
        backend.insert(EXPERIENCE_COUNT_TABLE, {
            "space_id": sid, "local_id": lid,
            "ctx_code": ctx_code, "speaker_code": speaker_code,
            "base_freq": 0, "e_sn": sn0, "e_tn": 1, "observe_tn": 0,
        })
        return
    set_: dict[str, tuple[str, int]] = {"e_tn": ("+=", 1)}
    if success:
        set_["e_sn"] = ("+=", 1)
    backend.update(EXPERIENCE_COUNT_TABLE, where={
        "space_id": sid, "local_id": lid,
        "ctx_code": ctx_code, "speaker_code": speaker_code,
    }, set_=set_)


def record_experience_observe(backend: StorageBackend, *, ref: tuple[int, int],
                              ctx_code: int = DEFAULT_CTX_CODE,
                              speaker_code: int = DEFAULT_SPEAKER_CODE) -> None:
    """记一次决策时 observe（sign-agnostic·observe_tn += 1·方案3 tn路 B4 β_arith 修法）。

    与 record_experience_outcome 区别：sign-agnostic（不接 reward·不分成功失败）·只 observe_tn += 1。
    解 β_arith：reward>0 episode 同比 e_sn++&e_tn++ 致 rate 塌缩·w_freq 概念间同·
    observe_tn 决策时写（dag_path add_active + attractor add_seed·非 episode 后批量）跨 episode 分化·
    独立 episode reward 符号（reward≤0 episode 的决策活动也计·反 e_tn reward>0 才计的塌缩）。
    首次：insert(base_freq=0, e_sn=0, e_tn=0, observe_tn=1)（镜像 record_experience_outcome 首次 insert·
      显式写全列避 SQLite 缺列 NULL·observe_tn 后续 += 1 不撞 NULL+1）。
    已存在：observe_tn += 1（MUTABLE_MONOTONE·delta 固定 +1·e_sn/e_tn/base_freq 不碰）。
    base_freq append-only 永不调（observe 路径不碰 base_freq·同 reward 路径·防塌柱① reward CAUSES-only 铁律）。
    表未注册（bare fixture/未注册场景）→ KeyError 静默 skip（向后兼容·镜像 record_base_freq/record_experience_outcome·三写函数兜底一致）。
    不接 reward：reward CAUSES-only 铁律守（observe 是概念维决策活动统计·非 reward 多头·不破防塌柱①）。
    """
    sid, lid = ref
    assert_int(sid, lid, ctx_code, speaker_code,
               _where="record_experience_observe.args")
    try:
        existing = backend.select(EXPERIENCE_COUNT_TABLE, where={
            "space_id": sid, "local_id": lid,
            "ctx_code": ctx_code, "speaker_code": speaker_code,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture/未注册场景）·向后兼容·镜像 record_base_freq/record_experience_outcome
    if not existing:
        backend.insert(EXPERIENCE_COUNT_TABLE, {
            "space_id": sid, "local_id": lid,
            "ctx_code": ctx_code, "speaker_code": speaker_code,
            "base_freq": 0, "e_sn": 0, "e_tn": 0, "observe_tn": 1,
        })
        return
    backend.update(EXPERIENCE_COUNT_TABLE, where={
        "space_id": sid, "local_id": lid,
        "ctx_code": ctx_code, "speaker_code": speaker_code,
    }, set_={"observe_tn": ("+=", 1)})
