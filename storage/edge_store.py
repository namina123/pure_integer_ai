"""storage.edge_store — edge 统一宽表 D1（§十五决策2 + 决策4 weight_p/q 不保留）。

D1 宽表（重来新基线·把散落窄表合进 edge）：
  端点：space_id_from, local_id_from, space_id_to, local_id_to（跨 space 复合键·决策1）
  类型：edge_type
  强度：strength（经验·reward 调·MUTABLE_MONOTONE 单调不降）
        base_strength（先验·append-only reward 不调·初始 DEFAULT_STRENGTH·tiebreak 读）
  信念：belief_p, belief_q（决策4：weight_p/q 是其冗余镜像·合并·砍 weight_p/q）
  经验：sn, tn（成功率计数·tn 可升·revisability 靠此非 tier 降级·低⑤）
  信任档：tier（PRIMARY/SHADOW 二级·§十二砍 SECONDARY·MUTABLE_MONOTONE 只升不降）
  provenance：source（数据源类型∈{CONCEPTNET,CODE,QA,BARE_TEXT,TEACHER,DERIVED,QUARANTINE}·非null）
              epistemic_origin（认识论来源·nullable·B8 与 source 正交）
  关系子类型：subtype（nullable·REFERS_TO 专属·PURE_ALIAS/METAPHOR/OCCURRENCE·S1·闭包纯净性）
  结构：order_index（PRECEDES 专属·可空·C4 句间序同域不同段 OFFSET）
        role（结构位置标签·可空）
  记忆时序：memory_time_attach（记忆边种类1时序附加·可空·C1 timestamp_seq）
  版本：content_version

砍：edge_tier 窄表、cg_edge_meta（合进 edge）、edge_verification_tier 代理、weight_p/weight_q。
索引：ix_edge_from/to/type/endpoint（吸收保留）。
MUTABLE_MONOTONE：strength/sn 单调不降；tier 只升不降；tn 可升（失败计数）。
H4：effective_weight=strength×rate 用 strength 非 base_strength。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.edge_types import is_registered_edge_type, EDGE_COOCCURS, EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW

DEFAULT_STRENGTH = 1  # base_strength 初始值（先验·reward 不调）

# ---- source 枚举（数据源类型·非null·§十五决策2/B8） ----
SOURCE_CONCEPTNET = 1
SOURCE_CODE = 2
SOURCE_QA = 3
SOURCE_BARE_TEXT = 4
SOURCE_TEACHER = 5
SOURCE_DERIVED = 6
SOURCE_QUARANTINE = 7   # 伴随检疫过闸（C9-ter·跨 space 检疫关联边 QUARANTINE_LINK 专属）
SOURCE_MATH = 8         # 算术域结构化源（Σ/Π/Recur/闭式 DSL·A3 兄弟件·doc/重来_算术域observe设计补充.md）
SOURCE_CHINESE_KB = 9   # ChineseSemanticKB（liuhuanyong·中文语义常用词典·公开文本+人工整理+机器抽取）·zh 反义/同义/简称/抽象/情态/否定 源

# ---- epistemic_origin 枚举（认识论来源·nullable·B8 与 source 正交） ----
EPI_STRUCTURED = 1      # 结构化源（代码/证明/文献/教师元定义）
EPI_CUE = 2             # 指向词锚定（§8.1c-bis 来源②）
EPI_LLM_CONFIRM = 3     # 断奶前 LLM 教师确认（断奶后退场判定）
# NULL for PRECEDES/COOCCURS/T_STEP/QUARANTINE_LINK

# ---- subtype 枚举（REFERS_TO 专属·nullable·S1 闭包纯净性） ----
SUBTYPE_PURE_ALIAS = 1    # 性质A 稳定同指·进纯同指闭包
SUBTYPE_METAPHOR = 2      # 喻称·不进闭包（拆三层）
SUBTYPE_OCCURRENCE = 3    # 性质B 语篇指代·occurrence token

# edge 宽表 D1 列（决策2 + 决策4）
EDGE_COLUMNS = [
    ("space_id_from", TYPE_INT),
    ("local_id_from", TYPE_INT),
    ("space_id_to", TYPE_INT),
    ("local_id_to", TYPE_INT),
    ("edge_type", TYPE_INT),
    ("strength", TYPE_INT),
    ("base_strength", TYPE_INT),
    ("belief_p", TYPE_INT),
    ("belief_q", TYPE_INT),
    ("sn", TYPE_INT),
    ("tn", TYPE_INT),
    ("tier", TYPE_INT),
    ("source", TYPE_INT),
    ("epistemic_origin", TYPE_INT),
    ("subtype", TYPE_INT),
    ("order_index", TYPE_INT),
    ("role", TYPE_INT),
    ("memory_time_attach", TYPE_INT),
    ("content_version", TYPE_INT),
]
EDGE_INDEXES = [
    ("space_id_from", "local_id_from"),  # ix_edge_from（query_from 无 et）
    ("space_id_to", "local_id_to"),       # ix_edge_to（query_to 无 et）
    ("edge_type",),                        # ix_edge_type（query_type）
    ("space_id_from", "local_id_from",
     "space_id_to", "local_id_to"),        # ix_edge_endpoint（add_cooccurs_dedup exact）
    ("tier",),
    ("source",),
    # 3-key 复合（2026-07-09·capability_exam cProfile·query_from/to(s,l,et) 热路径）：
    # query_from/to 带 edge_type 过滤是 D:11 word-concept readback（刀4/刀5 cue_type_of）主调用·
    # 旧仅 ix_edge_from(2 列)→返该节点全出边（高频词海量 COOCCURS）再 Python 过滤 et·浪费。
    # 3 列索引直命中 (s,l,et) 桶·覆盖最选择性·bit-identical-safe（索引只加速不改结果）。
    ("space_id_from", "local_id_from", "edge_type"),  # ix_edge_from_type（query_from(s,l,et)）
    ("space_id_to", "local_id_to", "edge_type"),       # ix_edge_to_type（query_to(s,l,et)）
]


def register_edge_table(backend: StorageBackend) -> None:
    """注册 edge 宽表 D1（核心表·启动调一次）。"""
    backend.register_table(
        "edge", EDGE_COLUMNS,
        disc.DISC_MUTABLE_MONOTONE, EDGE_INDEXES, core=True,
    )


class EdgeStore:
    """边宽表 D1 存储（经 backend 抽象·绝不写 raw SQL）。"""

    def __init__(self, backend: StorageBackend) -> None:
        self._b = backend
        # perf round3（2026-07-13）：COOCCURS 变更版本号·COOCCURS 插入/strength 更新时自增·
        # 供 hub_detect.compute_hub_set dirty-flag 缓存判失效（每代词 occurrence 全扫 COOCCURS = O(#pronoun×#COOCCURS)
        # → 缓存命 O(1)·失效即重算·O(n²)→O(n)）。**只 COOCCURS bump**（非通用 edge version）：代词解析自己 add
        # REFERS_TO/PROPERTY 边（refers_occurrence.py:134/177）·通用 version 会误失效零收益。纯整数单调计数·
        # 零存储语义影响·bit-identical（缓存仅加速·失效即 fresh 重算·永不 stale·尊重原 fresh-compute 设计 :68-69）。
        self._cooccurs_version: int = 0

    @property
    def cooccurs_version(self) -> int:
        """COOCCURS 变更版本号（单调自增·compute_hub_set cache 失效判据·读侧只读不写）。"""
        return self._cooccurs_version

    def _bump_if_cooccurs(self, edge_type: int) -> None:
        """COOCCURS 边变更 → bump 版本号（compute_hub_set dirty-flag cache 失效）。非 COOCCURS 边无操作。"""
        if edge_type == EDGE_COOCCURS:
            self._cooccurs_version += 1

    def add(self, *, space_id_from: int, local_id_from: int,
            space_id_to: int, local_id_to: int, edge_type: int,
            strength: int = DEFAULT_STRENGTH, source: int,
            tier: int = TIER_PRIMARY, epistemic_origin: int | None = None,
            subtype: int | None = None, order_index: int | None = None,
            role: int | None = None, memory_time_attach: int | None = None,
            belief_p: int = 0, belief_q: int = 1, sn: int = 0, tn: int = 0,
            content_version: int = 0) -> None:
        """插入边（append-only·base_strength=strength 初始·reward 不调 base_strength）。

        source 须非 null（数据源类型·B8）。epistemic_origin/subtype/order_index/role/
        memory_time_attach 可空（None→SQLite NULL）。
        """
        assert_int(space_id_from, local_id_from, space_id_to, local_id_to,
                   edge_type, strength, source, tier, belief_p, belief_q,
                   sn, tn, content_version, _where="EdgeStore.add")
        if not is_registered_edge_type(edge_type):
            raise ValueError(
                f"edge_type={edge_type} 未在 C9-bis 权威表登记（完备性 #1·"
                f"合法值见 edge_types.REGISTERED_EDGE_TYPES）"
            )
        self._b.insert("edge", {
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": edge_type,
            "strength": strength, "base_strength": strength,
            "belief_p": belief_p, "belief_q": belief_q,
            "sn": sn, "tn": tn, "tier": tier, "source": source,
            "epistemic_origin": epistemic_origin, "subtype": subtype,
            "order_index": order_index, "role": role,
            "memory_time_attach": memory_time_attach,
            "content_version": content_version,
        })
        self._bump_if_cooccurs(edge_type)   # perf round3：COOCCURS 插入 bump version（compute_hub_set cache 失效）

    def add_cooccurs_dedup(self, *, space_id_from: int, local_id_from: int,
                           space_id_to: int, local_id_to: int, edge_type: int,
                           source: int, tier: int = TIER_SHADOW) -> bool:
        """COOCCURS 去重 add：0 行→INSERT strength=1（返 True）/ 1 行→UPDATE strength+=1（返 False·仅 EDGE_COOCCURS）。

        解跨段重复 pair 堆叠（EdgeStore.add append-only 不去重·总收口 0.1·LIVE 病灶①）：同 (from,to,COOCCURS)
        跨段重复 pair·旧 add 每次插新行（vocab=50 COOCCURS 爆炸 9684）·真实语料跑不动。DEDUP 合并同 pair·
        strength=频次计数。reader（hub_degree/compute_hub_set/_cooccurs_count/collide_score）改读 strength 累加·
        gate OFF（旧 add strength 恒 1）累加=数行·完全等价 bit-identical；gate ON 频次正确。

        返 True=新建边（INSERT）/ False=已存在（UPDATE strength+=1）·caller build_cooccurs 仅对 True 计 n
        （built_edges=真实边数·dedup 后边数大降=解阻塞效果可观测·非配对数虚高）。

        **仅 EDGE_COOCCURS（code-level invariant·对抗审 P2-5）**：assert 强制 edge_type==EDGE_COOCCURS·
        防误用合并 PRECEDES（INSERT 不传 order_index 默认 None·腐蚀句间序）/ CAUSES（破 reward 分功语义）。
        caller build_cooccurs 已硬编码 EDGE_COOCCURS·assert 升防御。

        **纯净 gate-ON 假设（对抗审 P1-3）**：dedup 保证 run 内每 (from,to,EDGE_COOCCURS) ≤1 行。遇 ≥2 行
        （旧 append-only dump 跨 gate cursor 续训迁移场景）→ raise RuntimeError（防 add_strength 全量 +1 静默过计·
        append-only 禁 DELETE 不能合并已有重复行·跨 gate 迁移须重跑非 cursor 续训）。纯净 gate-ON（首次 run /
        同 gate 续训）永不触发。
        INSERT 路径 = self.add（strength=1·base_strength=1·append-only 先验）；UPDATE 路径 = self.add_strength
        （delta=1·MUTABLE_MONOTONE·base_strength 不动·H4 effective_weight 用 strength）。source 必传（B8）。
        """
        assert_int(space_id_from, local_id_from, space_id_to, local_id_to,
                   edge_type, source, tier, _where="EdgeStore.add_cooccurs_dedup")
        assert edge_type == EDGE_COOCCURS, (
            f"add_cooccurs_dedup 仅 EDGE_COOCCURS（防误用合并其他类型破语义）·got edge_type={edge_type}")
        rows = self._b.count("edge", where={
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": edge_type,
        })
        if rows == 0:
            self.add(space_id_from=space_id_from, local_id_from=local_id_from,
                     space_id_to=space_id_to, local_id_to=local_id_to,
                     edge_type=edge_type, strength=1, source=source,
                     epistemic_origin=None, tier=tier)
            return True
        if rows >= 2:
            # 旧 append-only 重复行（跨 gate cursor 续训迁移）·append-only 禁 DELETE 不能合并·
            # add_strength 全量 +1 会静默过计·fail-fast 防污染（须重跑非续训）。
            raise RuntimeError(
                f"add_cooccurs_dedup 遇 {rows} 行重复 COOCCURS 边 ({space_id_from},"
                f"{local_id_from})->({space_id_to},{local_id_to})·DEDUP 不合并旧 append-only 重复行"
                f"（append-only 禁 DELETE）·跨 gate cursor 续训须重跑非续训")
        self.add_strength(space_id_from=space_id_from, local_id_from=local_id_from,
                          space_id_to=space_id_to, local_id_to=local_id_to,
                          edge_type=edge_type, delta=1)
        return False

    def add_precedes_dedup(self, *, space_id_from: int, local_id_from: int,
                           space_id_to: int, local_id_to: int, edge_type: int,
                           source: int, order_index: int,
                           tier: int = TIER_PRIMARY) -> bool:
        """PRECEDES 跨 round 去重 add（mirror add_cooccurs_dedup·S2 dead-end 根因 §10.3·仅 EDGE_PRECEDES）。

        解 observe 跨 round 重建 PRECEDES 致 16× 重复（diag_de_dense·2256 vs 153 distinct·137 边组各 16×）。
        key = (from, to, EDGE_PRECEDES, order_index)——**含 order_index**（保同概念对多次出现的合法 pair 不误并·
        如 word"A"在 pos 3/50 各前于"B"→同 (from,to) 不同 oi 两行合法·不合并）。

        0 行 → INSERT strength=1（返 True）/ ≥1 行同 key → silent skip（返 False·不 raise）。
        **与 COOCCURS dedup 三差异**：(a) key 含 order_index（COOCCURS 无 oi）；
        (b) strength 恒 1（§7.1 结构真值·已存在 skip·非 COOCCURS UPDATE strength+=1 频次）；
        (c) ≥1 行 silent skip 非 raise（COOCCURS raise 防 strength 累加过计·PRECEDES strength 恒 1 无累加歧义·
        append-only 不删·silent skip 安全·无 cursor 续训迁移写入风险）。

        诚实边界：dedup 确定性 perf 16× + 数据卫生赢·**未必解 dead-end**（重复边同 from·AND 判定不变）·
        dedup 后重测 dead-end 定。不破 §7.1（strength=1·reward 永不调·propagate CAUSES-only 自动排除）。
        """
        assert_int(space_id_from, local_id_from, space_id_to, local_id_to,
                   edge_type, source, tier, order_index,
                   _where="EdgeStore.add_precedes_dedup")
        assert edge_type == EDGE_PRECEDES, (
            f"add_precedes_dedup 仅 EDGE_PRECEDES（防误用·got edge_type={edge_type}）")
        if self._b.count("edge", where={
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": EDGE_PRECEDES, "order_index": order_index,
        }) > 0:
            return False   # 同 (from,to,PRECEDES,oi) 已存在 → silent skip（strength 恒 1·不累加·不 raise）
        self.add(space_id_from=space_id_from, local_id_from=local_id_from,
                 space_id_to=space_id_to, local_id_to=local_id_to,
                 edge_type=EDGE_PRECEDES, strength=1, source=source,
                 epistemic_origin=None, order_index=order_index, tier=tier)
        return True

    def add_causes_dedup(self, *, space_id_from: int, local_id_from: int,
                         space_id_to: int, local_id_to: int, edge_type: int,
                         source: int, epistemic_origin: int,
                         tier: int = TIER_PRIMARY) -> bool:
        """CAUSES 跨 round 去重 add（mirror add_precedes_dedup·解 observe 16× 重复边膨胀·仅 EDGE_CAUSES）。

        解 observe 跨 round 重建 CAUSES 致 16× 重复（同 PRECEDES 16× bug·设计 CAUSES strength=reward 调涨
        学习性测度非频次·故 16× 重复违设计=bug）。key = (from, to, EDGE_CAUSES, source, epistemic_origin)--
        含 source+epistemic_origin（保 provenance·3 epistemic 源同三元组是合法不同边·同源同 epistemic 跨 round
        16× 是 bug）。下游 snapshot_strengths 按 (from,to) 覆写去重不区分源·故 16 行早被当 1 条·dedup key 含
        (source, epistemic_origin) 保写入侧 provenance 区分。

        0 行 -> INSERT strength=DEFAULT_STRENGTH（返 True）/ ≥1 行同 key -> silent skip（返 False·不 raise）。
        与 PRECEDES dedup 同（strength 恒 base·不累加·silent skip·不 raise·append-only 不删安全）。
        **reward 影响零**（终审 resolver 一锤定音·3 路径全证伪·审1"非零"判错）：① build_matrix sum 16× dup 权重
        致 PR matrix 变·**但 stepper.advance 不读 PR**（读 self.active+edge dict）·attractor 长 e_set 不回传
        stepper.active（构造时拷贝）·16 dup identical->min 选同 EdgeRef->path.edges 不变。② 假汇聚修致 struct_unit_refs
        变·但无 role_seq node 无输出（同）·有 role_seq 重复 OutputPart 但 J1/J2s/J3path/G 门全不变->reward 不变。
        ③ snapshot_strengths 覆写去重+record_episode_result 同 delta->strength_delta 同。dedup = 纯 perf（消 16× 边
        膨胀）+ 数据卫生（16× edge count 膨胀违设计）·gate OFF 守 CI bit-identical。cursor 续训 silent skip 残留旧
        16× 边（良性·mirror PRECEDES·假汇聚残留 benign·非 fresh run 仅部分修数据卫生）。
        gate CAUSES_DEDUP_MODE 默认 OFF 守 CI bit-identical（OFF 走旧 add 16×）。
        """
        assert_int(space_id_from, local_id_from, space_id_to, local_id_to,
                   edge_type, source, epistemic_origin, tier,
                   _where="EdgeStore.add_causes_dedup")
        assert edge_type == EDGE_CAUSES, (
            f"add_causes_dedup 仅 EDGE_CAUSES（防误用·got edge_type={edge_type}）")
        if self._b.count("edge", where={
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": EDGE_CAUSES, "source": source,
            "epistemic_origin": epistemic_origin,
        }) > 0:
            return False   # 同 (from,to,CAUSES,source,epistemic) 已存在 -> silent skip（strength 恒 base·不累加·不 raise）
        self.add(space_id_from=space_id_from, local_id_from=local_id_from,
                 space_id_to=space_id_to, local_id_to=local_id_to,
                 edge_type=EDGE_CAUSES, strength=DEFAULT_STRENGTH, source=source,
                 epistemic_origin=epistemic_origin, order_index=None, role=None, tier=tier)
        return True

    def get(self, *, space_id_from: int, local_id_from: int,
            space_id_to: int, local_id_to: int, edge_type: int) -> dict[str, Any] | None:
        """取单边（复合键·limit 1）。"""
        rows = self._b.select("edge", where={
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": edge_type,
        }, limit=1)
        return rows[0] if rows else None

    def set_tier(self, *, space_id_from: int, local_id_from: int,
                 space_id_to: int, local_id_to: int, edge_type: int,
                 new_tier: int) -> None:
        """边 tier flip（MUTABLE_MONOTONE 只升不降·SHADOW→PRIMARY·Stage 6 promote 消费）。

        promote 三重达则调此晋 PRIMARY。tier 降抛 MonotoneViolation（防 demotion 污染）。
        """
        cur = self.get(space_id_from=space_id_from, local_id_from=local_id_from,
                       space_id_to=space_id_to, local_id_to=local_id_to,
                       edge_type=edge_type)
        if cur is None:
            raise KeyError(f"set_tier: 边不存在 ({space_id_from},{local_id_from})->"
                           f"({space_id_to},{local_id_to}) et={edge_type}")
        old = cur["tier"]
        if new_tier < old:
            raise disc.MonotoneViolation(
                f"tier 须单调不降: old={old}, new={new_tier}"
            )
        if new_tier == old:
            return   # 幂等
        self._b.update("edge", where={
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": edge_type,
        }, set_={"tier": new_tier})

    def query_from(self, space_id: int, local_id: int,
                   edge_type: int | None = None) -> list[dict[str, Any]]:
        """查 from 端出边（可选按 edge_type 过滤）。"""
        where: dict[str, Any] = {"space_id_from": space_id, "local_id_from": local_id}
        if edge_type is not None:
            where["edge_type"] = edge_type
        return self._b.select("edge", where=where)

    def query_to(self, space_id: int, local_id: int,
                 edge_type: int | None = None) -> list[dict[str, Any]]:
        where: dict[str, Any] = {"space_id_to": space_id, "local_id_to": local_id}
        if edge_type is not None:
            where["edge_type"] = edge_type
        return self._b.select("edge", where=where)

    def query_type(self, edge_type: int) -> list[dict[str, Any]]:
        """查某类型的全部边（无端点过滤·单遍扫）。

        批量统计场景用（hub_detect.compute_hub_set 单遍建 degree map·vs per-ref query_from/to
        每次全表扫·2026-07-08 训练测试 perf 实测 7194×2 全表扫 = 276M 行 = 218s·改单遍根治）。
        """
        return self._b.select("edge", where={"edge_type": edge_type})

    def add_strength(self, *, space_id_from: int, local_id_from: int,
                     space_id_to: int, local_id_to: int, edge_type: int,
                     delta: int) -> None:
        """strength += delta（MUTABLE_MONOTONE·delta 须 ≥ 0·reward 反传落 CAUSES 头）。

        H4 effective_weight=strength×rate 用 strength 非 base_strength。
        """
        assert_int(delta, _where="add_strength.delta")
        if delta < 0:
            raise disc.MonotoneViolation(
                f"strength 须单调不降·delta 须 ≥ 0: delta={delta}"
            )
        self._b.update("edge", where={
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": edge_type,
        }, set_={"strength": ("+=", delta)})
        self._bump_if_cooccurs(edge_type)   # perf round3：COOCCURS strength 更新 bump（hub_degree 读 strength）

    def record_success(self, *, space_id_from: int, local_id_from: int,
                       space_id_to: int, local_id_to: int, edge_type: int,
                       success: bool) -> None:
        """sn++（成功）/ tn++（失败）。sn 单调不降；tn 可升（失败计数·revisability）。"""
        if success:
            self._b.update("edge", where={
                "space_id_from": space_id_from, "local_id_from": local_id_from,
                "space_id_to": space_id_to, "local_id_to": local_id_to,
                "edge_type": edge_type,
            }, set_={"sn": ("+=", 1)})
        else:
            self._b.update("edge", where={
                "space_id_from": space_id_from, "local_id_from": local_id_from,
                "space_id_to": space_id_to, "local_id_to": local_id_to,
                "edge_type": edge_type,
            }, set_={"tn": ("+=", 1)})

    def record_episode_result(self, *, space_id_from: int, local_id_from: int,
                              space_id_to: int, local_id_to: int, edge_type: int,
                              sn_delta: int, tn_delta: int,
                              strength_delta: int = 0) -> None:
        """卷二模块8 落点①：episode 级 reward 反传一次性更新 sn/tn/strength（R1）。

        R1 落盘：sn/tn 判定用 episode 级 reward 符号非边级 delta_reward。
          reward>0 → sn++ & tn++（参与即成功·成功是 episode 级）+ strength+=Δ（delta>0 才加）
          reward==0（judge veto）/ reward<0（死路）→ tn++ only
        守 MUTABLE_MONOTONE：sn_delta≥0（sn 单调不降）·strength_delta≥0（base_strength 不动·
          strength 单调）·tn_delta≥0（失败计数可升）。tn 无单调约束（revisability 靠此）。
        base_strength 先验 append-only 永不调（H4 effective_weight 用 strength 非 base_strength）。
        """
        assert_int(sn_delta, tn_delta, strength_delta,
                   _where="record_episode_result.deltas")
        if sn_delta < 0:
            raise disc.MonotoneViolation(
                f"sn 须单调不降·sn_delta 须 ≥ 0: {sn_delta}"
            )
        if strength_delta < 0:
            raise disc.MonotoneViolation(
                f"strength 须单调不降·strength_delta 须 ≥ 0: {strength_delta}"
            )
        if tn_delta < 0:
            raise ValueError(f"tn_delta 须 ≥ 0（失败计数可升不可降）: {tn_delta}")
        set_: dict[str, Any] = {}
        if sn_delta:
            set_["sn"] = ("+=", sn_delta)
        if tn_delta:
            set_["tn"] = ("+=", tn_delta)
        if strength_delta:
            set_["strength"] = ("+=", strength_delta)
        if not set_:
            return
        self._b.update("edge", where={
            "space_id_from": space_id_from, "local_id_from": local_id_from,
            "space_id_to": space_id_to, "local_id_to": local_id_to,
            "edge_type": edge_type,
        }, set_=set_)
        self._bump_if_cooccurs(edge_type)   # perf round3 防御性：COOCCURS strength 更新 bump（当前 reward 仅 feed
        #   CAUSES·reward_propagate.py:152 assert CAUSES-only·此 COOCCURS 分支生产不可达·纵深防御）
