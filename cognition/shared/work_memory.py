"""cognition.shared.work_memory — WorkMemory（跨卷工作记忆·§十四 J4 / 卷二过程）。

字段（规划 §一）：produced_refs / prior_topic_refs / pr_vector / round_id / weights /
  promoted_transition_targets / ctx / replay_candidates / exclude_refs。

卷一模块5（性质B pronoun）消费 prior_segments（FIFO window N=3·§十四J4低②）。
卷二/三消费其余字段（Stage 4/5 接线）。

纯整数：refs 是 ConceptRef tuple；pr_vector 是 {ref: Rational}（卷二填）。
无墙钟：时序靠 segment 序（observe 段序）非墙钟。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from pure_integer_ai.cognition.shared.types import ConceptRef

DEFAULT_PRONOUN_WINDOW = 3   # §十四J4 跨句 scope N=3 FIFO
# #729 M5 produced_refs FIFO cap（legacy_v1 _CONTEXT_PRODUCED_WINDOW=48 先例·oracle 标·
# 硬编码 module-level 常量守 bit-identical·禁 set_ 函数）。须 > 单段最大 unit 数（否则本段
# 已 append 的 unit 被截断删破去重语义）。
PRODUCED_REFS_WINDOW = 48
# #729 M5 章边界 carry → prior_topic_refs FIFO cap（legacy_v1 _CONTEXT_PRIOR_TOPIC_WINDOW=16
# 先例·oracle 标·硬编码守 bit-identical）。prior_topic_refs 是 M7 主题锚 stub 字段·#729 激活
# 作 ctx_refs 近似（slot_dispatch:100 已读·五环闭环）。首版 FIFO 末尾 N 近似·oracle 校准后改
# collide_score 高分 N 主题锚语义（设计文档决断4 字段语义错位诚实标）。
PRIOR_TOPIC_REFS_WINDOW = 16


@dataclass
class WorkMemory:
    """工作记忆（一个 episode / 一轮推理的瞬时状态）。

    prior_segments：FIFO 段序列·每段持 refs + seg 时序·供性质B pronoun 回溯前文。
    """

    round_id: int = 0
    produced_refs: list[ConceptRef] = field(default_factory=list)
    prior_topic_refs: list[ConceptRef] = field(default_factory=list)
    pr_vector: dict[ConceptRef, Any] = field(default_factory=dict)  # 卷二填 Rational
    weights: dict[int, int] = field(default_factory=dict)           # head_type → weight
    promoted_transition_targets: list[Any] = field(default_factory=list)
    ctx: tuple = ()                                                 # 多维 context_tag（F8）
    replay_candidates: list[Any] = field(default_factory=list)
    # B-PR4 动作词种子候选（doc §19·mirror replay_candidates 范式）：命令态（intent.type==INTENT_COMMAND）时
    # formal_train _run_reward_round 预算（_collect_action_seed_candidates·扫 segments 动作词 D:11 PRIMARY →
    # read_experience_count 洗净 sn==0 滤除 + rate-sort 降序）→ 存此处 → dag_path_step 入口读 + subgraph_nodes 过滤 +
    # append local_seeds/e_set（mirror #728 replay 扩张·PR 偏向动作拓扑邻域 §13.3）。gate ACTION_SEED_BIAS_MODE 守。
    # 默认 []（gate OFF / QUESTION intent → formal_train 不预算 → dag_path `if candidates:` 假 → 跳过 → bit-identical）。
    action_seed_candidates: list[Any] = field(default_factory=list)
    exclude_refs: set[ConceptRef] = field(default_factory=set)
    # ② fix（#733·J4 指代层3）：observe 段内代词悬空（resolve_pronoun_occurrence 返 None）→
    # _segment_dangling++ ·段末若 >0 标该段 struct_ref 进 dangling_units·judge check_closure ② 查
    # output.parts[*].unit ∈ dangling_units → J4=0 真碎句。绕旧 ② theater 4 重 kill switch
    # （produced_refs 是 struct_refs / surface_of 生产 None / out_edges 全类型 / EDGE_REFERS_TO 未 import）。
    dangling_units: set[ConceptRef] = field(default_factory=set)
    _segment_dangling: int = 0   # 段内悬空计数 temp（observe 段首 reset·段末归 dangling_units）
    # 层1 同段指代（factor E·2026-07-09·doc/重来_factorE_层1指代_intra_seg_设计）：段内已 normalize 的 token ref
    # （observe 段首 clear·token loop 每 token normalize 返回后 append·pronoun normalize 时含前序不含自身）。
    # resolve_pronoun_occurrence gate PRONOUN_INTRASEG_MODE ON 时读此作层1 候选源（同段前指·"动物...它们"同段）。
    # gate OFF 不读 → 无消费者 → 无可观察行为变 → bit-identical。append 不读 gate（cheap·无条件）。
    _current_segment_refs: list = field(default_factory=list)
    # 段 FIFO：每项 = (seg_seq, refs tuple)
    _segment_fifo: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_PRONOUN_WINDOW))
    # 维度桥 item→skeleton map（P1 G-PR2·COMPOSES_COMBINE_MODE ON 时 discovery 填·observe 读建 EDGE_INSTANTIATES 边·§十三-bis A.1）。
    # key=item_key（=id(collected_item)·稳定·raw per-round 重建 id 不稳故用 item_key）·value=skeleton_ref。gate OFF 不填不读→bit-identical。
    lang_skeleton_by_item: dict = field(default_factory=dict)
    # 维度桥 reader stub（P1 G-PR2·DIM_BRIDGE_READ_MODE ON 时 generate 读 binding on unit 记此·每 unit 重置）。
    # **P2 断桥 consumer stub**：P1 write-only 无消费者（值填充 VALUE_TRANSIT_MODE defer 断桥 #1053）·非 observability 信号
    # （无消费者≠observability·审2 MEDIUM-2 诚实标）。()=无 binding（default）·非空 tuple=skeleton_ref。consumer 落地后读此。
    last_dim_skeleton: tuple = ()
    # 对应泛化 readback→generation 桥 stash（CORRESPONDENCE_SLOT_MODE·doc/重来_对应泛化_readback_generation_桥·2026-07-17）。
    # generate 每 unit 读 unit→INSTANTIATES→skeleton→REALIZES→REL_* (rel_kind) + skeleton cue_sig (cue_slots)·
    # slot 循环设 current_slot_is_cue·dispatch_slot _correspondence_bonus 消费（第 8 路·(β) 独立轴·cue-slot-aware）。
    # **每 unit 全路径写（4 case 无漏清）**：gate OFF 不进块→default 守 / 无 INSTANTIATES→else 清双字段 / 有 INSTANTIATES+rel_kind=0→
    # elif 清 cue_slots / 有 INSTANTIATES+rel_kind!=0→设 cue_slots（length-guard 守·不等→∅）→无 stale state（审2 核证）。
    # default 0/∅/False（getattr 亦守·审1 LOW-1 契约级显式 default）·gate OFF 不读不写→bit-identical。
    current_rel_kind: int = 0
    current_cue_slots: frozenset = field(default_factory=frozenset)
    current_slot_is_cue: bool = False
    # 命门③ 候选 B（doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18·结构活化·功能词插补）：
    # current_cue_sig=cue token ConceptRef 序（tuple·None 占位=非 cue 位·与 role_seq 等长·length-guard 守）。
    # generate CORRESPONDENCE_SLOT_MODE 块 4 case 全路径设（审1 MED-1·无 stale state·镜像 current_rel_kind 范式）：
    #   ① gate OFF 不进块->default () 守 / ② 无 INSTANTIATES->else 设 () / ③ 有 INSTANTIATES+rel_kind=0->elif 设 ()
    #   / ④ 有 INSTANTIATES+rel_kind!=0+length-guard pass->设真 cue_sig 序（不等->()）。
    # dispatch_slot cue 位早 return 读 current_cue_sig[slot_idx]->surface_of 直出功能词·绕 collide。
    # current_slot_idx=当前 slot 序号（dispatch_slot 无 slot_idx 参数·走 workmem·审2 HIGH-1 修·generate 每 slot 设）。
    # default ()/0（getattr 亦守·审1 LOW-1 契约级显式 default）·gate OFF 不读不写->bit-identical。
    current_cue_sig: tuple = ()
    current_slot_idx: int = 0
    # 命门③ 候选 C（doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18·抽象活化·内容词按 slot IS_A LCA 类约束）：
    # current_slot_lcas=slot LCA ConceptRef 序（tuple·None 占位无约束位·与 role_seq 等长·length-guard 守）。
    # generate 独立 SLOT_LCA_CONSTRAINT_MODE 块 4 case 全路径设（审1 MED-1·无 stale state·非嵌 CORRESPONDENCE_SLOT_MODE·独立于 cue 链）：
    #   ① gate OFF 不进块->default () 守 / ② 无 INSTANTIATES->设 () / ③ length-guard fail->设 ()
    #   / ④ length-guard pass->设真 slot_lcas 序。
    # current_slot_lca=当前 slot_idx 的 LCA ConceptRef（()=无约束·mirror last_dim_skeleton 既有范式用 () 表 None·非 Optional·审1 LOW-2·
    # dispatch_slot 检查 `!= ()`·slot loop 每 slot 设·None 占位位亦设 ()）。
    # dispatch_slot 内容词位读 current_slot_lca->is_a_descendant_of(c, slot_lca) 过滤候选（reflexive-transitive）。
    # default ()（getattr 亦守 `getattr(workmem, "current_slot_lca", ())`·审1 LOW-1·minimal workmem 不崩·gate OFF 不读不写->bit-identical）。
    current_slot_lcas: tuple = ()
    current_slot_lca: tuple = ()

    def push_segment(self, seg_seq: int, refs: list[ConceptRef]) -> None:
        """段处理完入 FIFO（供后续段 pronoun 回溯前文·跨句 partial）。"""
        self._segment_fifo.append((seg_seq, tuple(refs)))

    def prior_segments(self, *, window: int = DEFAULT_PRONOUN_WINDOW,
                       fifo: bool = True) -> list[tuple[int, tuple[ConceptRef, ...]]]:
        """回溯前文段（FIFO·最近 window 段·时序衰减权重由调用方算）。

        fifo=True：最近段在前（近因优先）；False：远段在前。
        """
        items = list(self._segment_fifo)
        if fifo:
            items = list(reversed(items))   # 最近在前
        return items[:window]

    def recency_weight(self, seg_seq: int) -> int:
        """近因权重（纯整数·越近越大·时序衰减·seg_seq 大=近）。

        weight = max(1, WINDOW − dist)（dist = 当前最大 seg_seq − seg_seq·0=最近→最高）。
        线性衰减·窗口内 floor ≥1·窗口外（dist≥WINDOW）→ 1（弱但不零·I5 floor 守 ≥0）。
        旧版 decay**dist 公式对纯整反向（decay≥2 时远者反大）/ decay=1 恒 1 退自然序——stub #5 修。
        """
        if not self._segment_fifo:
            return 0
        cur = max(s for s, _ in self._segment_fifo)
        dist = cur - seg_seq
        if dist < 0:
            dist = 0
        return max(1, DEFAULT_PRONOUN_WINDOW - dist)

    def add_produced(self, ref: ConceptRef, *, window: int = PRODUCED_REFS_WINDOW) -> None:
        """累积已产出 ref（保序去重 + FIFO 截断防爆·确定性）。纯整数。

        #729 M5 produced_refs cap：window 满 del [:over] 截断保近期（复用 legacy_v1
        add_produced 范式·_archive/legacy_v1/pure_integer_ai/edge/nl_generate.py:306-313）。
        window 须 > 单段最大 unit 数（否则本段已 append 的 unit 被截断删破去重语义）。
        截断删头部最旧·新 append 在尾部·本段已 append 的 unit 不会被截断删。
        collide_score 用 set(ctx_refs) 自动去重（graph_view.py:148）·重复 ref 不影响分数。
        """
        if ref not in self.produced_refs:
            self.produced_refs.append(ref)
        over = len(self.produced_refs) - window
        if over > 0:
            del self.produced_refs[:over]   # 截断保近期（末 window 个）

    def add_prior_topic(self, ref: ConceptRef, *, window: int = PRIOR_TOPIC_REFS_WINDOW) -> None:
        """累积章末 anchor ref 进 prior_topic_refs（保序去重 + FIFO 截断防爆·确定性）。纯整数。

        #729 M5 章边界 carry：章末 parts 子集 snapshot 进 prior_topic_refs（复用 legacy_v1
        add_prior_topic 范式·_archive/legacy_v1/pure_integer_ai/edge/nl_generate.py:315-324）。
        prior_topic_refs 是 M7 主题锚 stub 字段·#729 激活作 ctx_refs 近似（slot_dispatch:100
        ctx_refs = prior_topic_refs + produced_refs·五环闭环消费者就位）。
        首版 FIFO 末尾 N 近似·oracle 校准后改 collide_score 高分 N 主题锚语义（设计文档决断4）。
        """
        if ref not in self.prior_topic_refs:
            self.prior_topic_refs.append(ref)
        over = len(self.prior_topic_refs) - window
        if over > 0:
            del self.prior_topic_refs[:over]   # 截断保近期（末 window 段主题）
