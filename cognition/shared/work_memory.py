"""cognition.shared.work_memory — WorkMemory（跨卷工作记忆·§十四 J4 / 卷二过程）。

字段（规划 §一）：produced_refs / prior_topic_refs / pr_vector / round_id / weights /
  promoted_transition_targets / ctx / replay_candidates / exclude_refs /
  attractor_state。

卷一模块5（性质B pronoun）消费 prior_segments（FIFO window N=3·§十四J4低②）。
卷二/三消费其余字段（Stage 4/5 接线）。

纯整数：refs 是 ConceptRef tuple；pr_vector 是 {ref: Rational}（卷二填）。
无墙钟：时序靠 segment 序（observe 段序）非墙钟。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    SCOPE_EPISODE,
    SCOPE_GENERATION,
    SCOPE_QUERY,
    SCOPE_SESSION,
    ScopeIdentity,
)

if TYPE_CHECKING:
    from pure_integer_ai.cognition.shared.attractor_state import AttractorState
    from pure_integer_ai.cognition.shared.formal_artifact import FormalArtifact
    from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
        ArtifactInvocationResult,
    )

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


class WorkMemoryScopeError(ValueError):
    """工作记忆生命周期或 scope 身份不满足契约时抛出的错误。"""


@dataclass
class WorkMemory:
    """按显式生命周期管理的瞬时工作记忆。

    `lang_skeleton_by_item` 是 session 级的稳定键桥接；其余字段分别由
    document、episode、query、generation 和 segment 边界管理。没有显式 scope
    时，生命周期方法拒绝运行；直接构造的旧单元测试仍可使用兼容字段路径。
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
    # key=item_key（稳定 document scope hash）·value=skeleton_ref。gate OFF 不填不读→bit-identical。
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
    # S-06：当前 episode 的 typed Artifact；query trace 与值本身分开清理。
    episode_artifacts: dict[ObjectIdentity, FormalArtifact] = field(
        default_factory=dict, init=False, repr=False)
    query_artifact_results: list[ArtifactInvocationResult] = field(
        default_factory=list, init=False, repr=False)
    # A-10：只保存当前 query 的动态方向状态，长期候选仍归 Core/Memory。
    attractor_state: AttractorState | None = field(
        default=None, init=False, repr=False)
    # 生命周期活动身份。身份只作边界和隔离判据，不替代图中的本体对象。
    active_session_scope: ScopeIdentity | None = field(default=None, init=False, repr=False)
    active_document_scope: ScopeIdentity | None = field(default=None, init=False, repr=False)
    active_episode_scope: ScopeIdentity | None = field(default=None, init=False, repr=False)
    active_query_scope: ScopeIdentity | None = field(default=None, init=False, repr=False)
    active_generation_scope: ScopeIdentity | None = field(default=None, init=False, repr=False)
    active_segment_index: int | None = field(default=None, init=False, repr=False)
    _occurrence_ordinal: int = field(default=0, init=False, repr=False)
    _pending_replay_document: ScopeIdentity | None = field(default=None, init=False, repr=False)
    _pending_replay_candidates: list[Any] = field(default_factory=list, init=False, repr=False)
    _pending_exclude_refs: set[ConceptRef] = field(default_factory=set, init=False, repr=False)
    _query_resources: list[Any] = field(default_factory=list, init=False, repr=False)

    @property
    def lifecycle_active(self) -> bool:
        """返回是否已进入显式 session 生命周期。"""
        return self.active_session_scope is not None

    @property
    def episode_active(self) -> bool:
        """返回是否存在可写的当前 episode。"""
        return self.active_episode_scope is not None

    def _require_scope(self, scope: ScopeIdentity | None, expected_kind: int,
                       where: str) -> ScopeIdentity:
        """校验边界身份存在且类型正确，防止无 scope 状态写入跨边界缓存。"""
        if scope is None:
            raise WorkMemoryScopeError(f"{where} 必须提供 ScopeIdentity")
        if not isinstance(scope, ScopeIdentity):
            raise WorkMemoryScopeError(f"{where} 必须是 ScopeIdentity")
        if scope.scope_kind != expected_kind:
            raise WorkMemoryScopeError(
                f"{where} scope_kind={scope.scope_kind}，期望 {expected_kind}")
        return scope

    def _require_active(self, scope: ScopeIdentity, active: ScopeIdentity | None,
                        where: str) -> None:
        """校验子边界只能在其已登记的父边界内打开。"""
        if active is None:
            raise WorkMemoryScopeError(f"{where} 的父生命周期尚未开始")
        if scope.parent != active:
            raise WorkMemoryScopeError(f"{where} 与当前父 scope 不一致")

    def _clear_generation_state(self) -> None:
        """清理一次生成期间的槽位和维度桥临时状态。"""
        self.last_dim_skeleton = ()
        self.current_rel_kind = 0
        self.current_cue_slots = frozenset()
        self.current_slot_is_cue = False
        self.current_cue_sig = ()
        self.current_slot_idx = 0
        self.current_slot_lcas = ()
        self.current_slot_lca = ()

    def _clear_query_state(self) -> None:
        """清理一次 query 的推理结果和上下文，保留文档级上下文。"""
        self._close_query_resources()
        self.pr_vector = {}
        self.weights = {}
        self.promoted_transition_targets.clear()
        self.ctx = ()
        self.query_artifact_results.clear()
        self.attractor_state = None
        self._clear_generation_state()

    def _clear_episode_state(self) -> None:
        """清理 episode 级事实，避免悬空、槽位和旧回放进入无关样本。"""
        self._close_query_resources()
        self.dangling_units.clear()
        self._segment_dangling = 0
        self._current_segment_refs.clear()
        self._segment_fifo.clear()
        self.pr_vector = {}
        self.weights = {}
        self.promoted_transition_targets.clear()
        self.ctx = ()
        self.action_seed_candidates.clear()
        self.episode_artifacts.clear()
        self.query_artifact_results.clear()
        self._clear_generation_state()

    def _clear_document_state(self) -> None:
        """清理文档级瞬时上下文和未消费的 episode 继承槽。"""
        self.produced_refs.clear()
        self.prior_topic_refs.clear()
        self._clear_episode_state()
        self.replay_candidates.clear()
        self.exclude_refs.clear()
        self._pending_replay_document = None
        self._pending_replay_candidates.clear()
        self._pending_exclude_refs.clear()
        self._occurrence_ordinal = 0

    def begin_session(self, scope: ScopeIdentity | None) -> None:
        """打开 session；session 重开时清掉所有上一运行的瞬时状态。"""
        scope = self._require_scope(scope, SCOPE_SESSION, "begin_session")
        if self.active_session_scope is not None:
            if self.active_session_scope == scope:
                return
            raise WorkMemoryScopeError("已有其他 session 未结束")
        self._clear_document_state()
        self.lang_skeleton_by_item.clear()
        self.active_session_scope = scope
        self.active_document_scope = None
        self.active_episode_scope = None
        self.active_query_scope = None
        self.active_generation_scope = None
        self.active_segment_index = None

    def end_session(self) -> None:
        """关闭 session；存在未关闭子生命周期时拒绝静默截断。"""
        if self.active_session_scope is None:
            return
        if any((self.active_document_scope, self.active_episode_scope,
                self.active_query_scope, self.active_generation_scope,
                self.active_segment_index is not None)):
            raise WorkMemoryScopeError("关闭 session 前必须结束所有子生命周期")
        self._clear_document_state()
        self.lang_skeleton_by_item.clear()
        self.active_session_scope = None

    def begin_document(self, scope: ScopeIdentity | None) -> None:
        """打开 document，并重置上一文档的段、产出、主题和 occurrence 状态。"""
        scope = self._require_scope(scope, SCOPE_DOCUMENT, "begin_document")
        if self.active_session_scope is None:
            raise WorkMemoryScopeError("begin_document 需要活动 session")
        if (scope.owner != self.active_session_scope.owner
                or scope.versions != self.active_session_scope.versions):
            raise WorkMemoryScopeError("document 与 session 的 owner/version 不一致")
        if self.active_document_scope is not None:
            raise WorkMemoryScopeError("已有 document 未结束")
        self._clear_document_state()
        self.active_document_scope = scope

    def end_document(self) -> None:
        """关闭 document，并清理所有不应流向下一文档的瞬时状态。"""
        if self.active_document_scope is None:
            return
        if any((self.active_episode_scope, self.active_query_scope,
                self.active_generation_scope, self.active_segment_index is not None)):
            raise WorkMemoryScopeError("关闭 document 前必须结束 episode/query/generation/segment")
        self._clear_document_state()
        self.active_document_scope = None

    def begin_episode(self, scope: ScopeIdentity | None, *, round_id: int | None = None) -> None:
        """打开 episode，并只恢复同一文档明确允许继承的回放结果。"""
        scope = self._require_scope(scope, SCOPE_EPISODE, "begin_episode")
        self._require_active(scope, self.active_document_scope, "begin_episode")
        if self.active_episode_scope is not None:
            raise WorkMemoryScopeError("已有 episode 未结束")
        self._clear_episode_state()
        if self._pending_replay_document == self.active_document_scope:
            self.replay_candidates.extend(self._pending_replay_candidates)
            self.exclude_refs.update(self._pending_exclude_refs)
        self._pending_replay_document = None
        self._pending_replay_candidates.clear()
        self._pending_exclude_refs.clear()
        self.active_episode_scope = scope
        if round_id is not None:
            self.round_id = round_id

    def end_episode(self) -> None:
        """关闭 episode，把回放结果转为下一同文档 episode 的显式继承槽。"""
        if self.active_episode_scope is None:
            return
        if any((self.active_query_scope, self.active_generation_scope,
                self.active_segment_index is not None)):
            raise WorkMemoryScopeError("关闭 episode 前必须结束 query/generation/segment")
        self._pending_replay_document = self.active_document_scope
        self._pending_replay_candidates = list(self.replay_candidates)
        self._pending_exclude_refs = set(self.exclude_refs)
        self._clear_episode_state()
        self.replay_candidates.clear()
        self.exclude_refs.clear()
        self.active_episode_scope = None

    def abort_episode(self) -> None:
        """异常中止 episode，丢弃所有待继承结果并恢复到干净文档状态。"""
        if self.active_segment_index is not None:
            self.active_segment_index = None
        self.active_generation_scope = None
        self.active_query_scope = None
        self._pending_replay_document = None
        self._pending_replay_candidates.clear()
        self._pending_exclude_refs.clear()
        self._clear_episode_state()
        self.replay_candidates.clear()
        self.exclude_refs.clear()
        self.active_episode_scope = None

    def begin_query(self, scope: ScopeIdentity | None) -> None:
        """打开 query，并清掉上一 query 的路径向量和生成临时状态。"""
        scope = self._require_scope(scope, SCOPE_QUERY, "begin_query")
        self._require_active(scope, self.active_episode_scope, "begin_query")
        if self.active_query_scope is not None:
            raise WorkMemoryScopeError("已有 query 未结束")
        self._clear_query_state()
        self.active_query_scope = scope

    def register_query_resource(self, resource: Any) -> None:
        """登记随当前 query 一起关闭的 context-local 资源。"""
        if self.active_query_scope is None:
            raise WorkMemoryScopeError("登记 query 资源需要活动 query")
        close = getattr(resource, "close", None)
        if not callable(close):
            raise TypeError("query resource 必须提供可调用 close")
        if any(item is resource for item in self._query_resources):
            raise WorkMemoryScopeError("同一 query resource 不得重复登记")
        self._query_resources.append(resource)

    def _close_query_resources(self) -> None:
        """按逆安装顺序关闭 query 资源，失败项保留以允许调用方重试。"""
        while self._query_resources:
            resource = self._query_resources[-1]
            resource.close()
            self._query_resources.pop()

    def end_query(self) -> None:
        """关闭 query；结果已由 Episode 复制后再清理临时状态。"""
        if self.active_query_scope is None:
            return
        if any((self.active_generation_scope, self.active_segment_index is not None)):
            raise WorkMemoryScopeError("关闭 query 前必须结束 generation/segment")
        self._clear_query_state()
        self.active_query_scope = None

    def begin_generation(self, scope: ScopeIdentity | None) -> None:
        """打开 generation，并保证槽位状态不会从上一代输出泄漏。"""
        scope = self._require_scope(scope, SCOPE_GENERATION, "begin_generation")
        self._require_active(scope, self.active_query_scope, "begin_generation")
        if self.active_generation_scope is not None:
            raise WorkMemoryScopeError("已有 generation 未结束")
        self._clear_generation_state()
        self.active_generation_scope = scope

    def end_generation(self) -> None:
        """关闭 generation 并清理槽位、维度桥和当前 slot 状态。"""
        if self.active_generation_scope is None:
            return
        self._clear_generation_state()
        self.active_generation_scope = None

    def begin_segment(self, segment_index: int) -> None:
        """打开段边界，清理段内候选并保留同文档段 FIFO。"""
        if self.active_episode_scope is None:
            raise WorkMemoryScopeError("begin_segment 需要活动 episode")
        if type(segment_index) is not int or segment_index < 0:
            raise WorkMemoryScopeError("segment_index 必须是非负整数")
        if self.active_segment_index is not None:
            raise WorkMemoryScopeError("已有 segment 未结束")
        self._segment_dangling = 0
        self._current_segment_refs.clear()
        self.active_segment_index = segment_index

    def begin_observation_state(self) -> None:
        """开始一次 observe，清理本次观察的悬空事实而保留文档段 FIFO。"""
        if self.active_episode_scope is None:
            raise WorkMemoryScopeError("begin_observation_state 需要活动 episode")
        if self.active_segment_index is not None:
            raise WorkMemoryScopeError("开始 observe 时仍有未结束 segment")
        self.dangling_units.clear()
        self._segment_dangling = 0
        self._current_segment_refs.clear()

    def end_segment(self, refs: list[ConceptRef]) -> None:
        """关闭段并登记段 FIFO，供后续段的 occurrence 解析回溯。"""
        if self.active_segment_index is None:
            raise WorkMemoryScopeError("end_segment 没有活动 segment")
        self.push_segment(self.active_segment_index, refs)
        self.active_segment_index = None

    def next_occurrence_ordinal(self) -> int:
        """分配当前文档内单调 occurrence 序号，重复概念不会因此合并。"""
        if self.active_segment_index is None:
            raise WorkMemoryScopeError("occurrence 必须位于活动 segment 内")
        self._occurrence_ordinal += 1
        return self._occurrence_ordinal

    def assert_episode_scope(self, scope: ScopeIdentity | None) -> None:
        """校验输入 scope 正是当前 episode，阻止观察写入错误上下文。"""
        scope = self._require_scope(scope, SCOPE_EPISODE, "assert_episode_scope")
        if self.active_episode_scope != scope:
            raise WorkMemoryScopeError("输入 scope 与当前 episode 不一致")

    def put_episode_artifact(self, artifact: FormalArtifact) -> None:
        """登记当前 episode 或其子 scope 的 Artifact，并拒绝身份碰撞和跨边界写入。"""
        from pure_integer_ai.cognition.shared.formal_artifact import FormalArtifact

        if self.active_episode_scope is None:
            raise WorkMemoryScopeError("写 Artifact 需要活动 episode")
        if not isinstance(artifact, FormalArtifact):
            raise TypeError("artifact 必须是 FormalArtifact")
        scope = artifact.scope
        if scope is None or not self._scope_descends_from(
                scope, self.active_episode_scope):
            raise WorkMemoryScopeError("Artifact 不属于当前 episode scope")
        existing = self.episode_artifacts.get(artifact.identity)
        if existing is not None and existing != artifact:
            raise WorkMemoryScopeError("同一 Artifact 身份对应不同运行期内容")
        self.episode_artifacts[artifact.identity] = artifact

    def get_episode_artifact(
            self, identity: ObjectIdentity,
            ) -> FormalArtifact | None:
        """在活动 episode 内按完整身份读取 Artifact，不接受裸 hash 或 local id。"""
        if self.active_episode_scope is None:
            raise WorkMemoryScopeError("读 Artifact 需要活动 episode")
        if not isinstance(identity, ObjectIdentity):
            raise TypeError("Artifact identity 必须是 ObjectIdentity")
        return self.episode_artifacts.get(identity)

    def record_artifact_result(self, result: ArtifactInvocationResult) -> None:
        """保存当前 query 的完整调用 trace，并把当前参数和值提升到 episode 槽。"""
        from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
            ArtifactInvocationResult,
        )

        if self.active_query_scope is None:
            raise WorkMemoryScopeError("记录 Artifact result 需要活动 query")
        if not isinstance(result, ArtifactInvocationResult):
            raise TypeError("result 必须是 ArtifactInvocationResult")
        if result.invocation.scope != self.active_query_scope:
            raise WorkMemoryScopeError("Artifact result 不属于当前 query")
        for argument in result.invocation.arguments:
            self.put_episode_artifact(argument.value)
        if result.invocation.expected is not None:
            self.put_episode_artifact(result.invocation.expected)
        if result.value is not None:
            self.put_episode_artifact(result.value)
        if result.proof is not None:
            self.put_episode_artifact(result.proof)
        self.query_artifact_results.append(result)

    def install_attractor_state(self, state: AttractorState) -> None:
        """把唯一 A-10 状态绑定到活动 query，拒绝跨 owner/session 或重复安装。"""
        from pure_integer_ai.cognition.shared.attractor_state import (
            AttractorState,
        )

        if self.active_query_scope is None:
            raise WorkMemoryScopeError("安装 AttractorState 需要活动 query")
        if not isinstance(state, AttractorState):
            raise TypeError("state 必须是 AttractorState")
        if state.scope != self.active_query_scope:
            raise WorkMemoryScopeError("AttractorState 不属于当前 query")
        if self.attractor_state is not None:
            raise WorkMemoryScopeError("当前 query 已安装 AttractorState")
        self.attractor_state = state

    def require_attractor_state(self) -> AttractorState:
        """返回当前 query 的 A-10 状态；未安装时拒绝隐式空状态。"""
        if self.active_query_scope is None:
            raise WorkMemoryScopeError("读取 AttractorState 需要活动 query")
        if self.attractor_state is None:
            raise WorkMemoryScopeError("当前 query 尚未安装 AttractorState")
        if self.attractor_state.scope != self.active_query_scope:
            raise WorkMemoryScopeError("AttractorState 与活动 query 漂移")
        return self.attractor_state

    @staticmethod
    def _scope_descends_from(
            scope: ScopeIdentity, ancestor: ScopeIdentity,
            ) -> bool:
        """沿显式 parent 链判断 scope 是否属于给定祖先，禁止只比 local_id。"""
        current: ScopeIdentity | None = scope
        while current is not None:
            if current == ancestor:
                return True
            current = current.parent
        return False

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
