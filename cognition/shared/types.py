"""cognition.shared.types — 跨卷共享类型（卷一/二/三公共依赖点）。

卷一（本 Stage）实装：InputPayload / Segment / ParsedSegment / ConceptRef / MultiRef /
  ObserveResult / SpaceContext / TrainingStage / 模态标记 / LangMarker / DomainMarker。
卷二/三类型（PathData / PathResult / DAGPath / Episode / G_meta）本 Stage 仅骨架占位——
  Stage 4/5 填实（避免反向阻塞卷一·守单向依赖）。

NodeRef / ConceptRef = tuple[int, int] = (space_id, local_id)（与 algorithm/vm 同形）。
纯整数：所有 id/标记是 int；文本只入伴随库（守"文本不入核心"）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion, CurriculumVersion, LogicalTime, ObjectIdentity, OwnerScope,
    ParserVersion, PrimitiveVersion, SourceRef, TypedRef, VersionBundle,
)

# ---- 编址 ----
NodeRef = tuple[int, int]          # (space_id, local_id)
ConceptRef = NodeRef
EdgeRef = tuple[int, int, int, int, int]  # (space_id_from, local_id_from, space_id_to, local_id_to, edge_type)


# ---- 模态标记（§7.4·一级语言/声/2D/3D·二级动画=2D+时间） ----
MODALITY_LANGUAGE = 1
MODALITY_AUDIO = 2      # 声（1d 时间）
MODALITY_2D = 3         # 2D 空间（静态）
MODALITY_3D = 4         # 3D 空间（静态）
MODALITY_ANIMATION = 5  # 动画（2D+时间）
MODALITY_CODE = 6       # 代码（A3·Python AST→COMPOSES 程序·doc/重来_A3_代码域observe设计补充.md）
MODALITY_ARITH = 7      # 算术（A3 兄弟件·数学记号 DSL→COMPOSES·doc/重来_算术域observe设计补充.md）

# 语言子模态（LangMarker）
LANG_ZH = 1
LANG_EN = 2
LANG_NONE = 0   # 非语言模态

# 域标记（DomainMarker）
DOMAIN_TEXT = 1
DOMAIN_CODE = 2
DOMAIN_MATH = 3
DOMAIN_BARE = 4

# reward 合法域（judge G5 correctness 承重域·methodology doc §二铁事实2）：DOMAIN_MATH+DOMAIN_CODE
# 有 vm_proof 自证机·reward 经 G5 与 correctness 对齐 = reward-legitimate。DOMAIN_TEXT / DOMAIN_BARE 无 vm_proof·
# reward 结构性 theater（G5 vacated·judge.py:43 _ARITH_DOMAINS 不含）·CAUSES edge reward 写有害（dead-end/veto→
# tn++ 惩罚唯一 reward-active 边·止血 #1146）。canonical 单一源 hoist 自 judge._ARITH_DOMAINS → shared
# （reward_propagate cognition.process 复用·避 process→result 顶层循环依赖·judge 保持 alias 不变）。
REWARD_LEGITIMATE_DOMAINS = frozenset({DOMAIN_CODE, DOMAIN_MATH})


# ---- 训练阶段（决定落点·模块8） ----
STAGE_TRAINING = 1              # ZERO_AI_MEMORY_ACTIVE=0·只核心养洁净
STAGE_POST_WEANING_READ = 2     # 训练后阅读·记忆一层（经伴随检疫晋升）
STAGE_USER_INTERACTION = 3      # 用户交互·记忆二层（全量·不经检疫）
STAGE_EXTERNAL_DEFINE = 4       # 外部 define/注入·伴随检疫

WEANING_PRE = 0     # 断奶前（LLM 教师走录放层合法·来源③）
WEANING_POST = 1    # 断奶后（LLM 退场·新元定义不注入核心）


# ---- 概念引用多挂（§B12·1:N·消歧在生成侧非摄入侧） ----
@dataclass(frozen=True)
class MultiRef:
    """同词多挂 1:N（多义 sense / 多概念）。activate_candidates 返全部·禁取首。"""

    refs: tuple[ConceptRef, ...]

    def __iter__(self):
        return iter(self.refs)

    def __len__(self) -> int:
        return len(self.refs)


# ---- 代码域规格契约（C6 生产闭环·A3 下游） ----

@dataclass(frozen=True)
class CodeSpec:
    """代码域一条测试用例规格（C6·R6 独立源·doc/重来_A3_代码域observe设计补充.md §二致命#2）。

    一函数多测试用例·spec 来自 corpus/CollectedItem（独立源·非教师录制·非学生 COMPOSES 编译）。
    vm_proof_fn 执行学生自建 COMPOSES 得结果·比 expected → 1/0/None（两路独立判 input→result·非 theater）。

    input_args : 函数参数值序（对应 FunctionDef.args 顺序·builder 分配 make_variable(0..n-1)）。
    expected   : (num, den) 期望返回值（Rational·纯整数·den≠0 fail-loud）。
    """

    input_args: tuple[int, ...]
    expected: tuple[int, int]

    def __post_init__(self) -> None:
        for a in self.input_args:
            assert_int(a, _where="CodeSpec.input_args")
        assert_int(self.expected[0], self.expected[1],
                   _where="CodeSpec.expected")
        assert_no_float(*self.input_args, self.expected[0], self.expected[1],
                        _where="CodeSpec")
        if self.expected[1] == 0:
            raise ValueError("CodeSpec: expected den 须非零（纯整数铁律）")


@dataclass(frozen=True)
class TransformHeldOut:
    """符号变换规则 held-out 验证对（符号数学扩展 Phase 3·doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis）。

    一条 held-out = (input_source, expected_source)：教师陈述的变换规则应用到 input_source（lambda DSL）
    → 产 output·cross-verify 执行 output == 执行 expected_source（统计验规则应用正确·stable≠correct·#479 守）。
    spec 来自 corpus（教师标·数据驱动·非硬编码·同 CodeSpec R6 独立源范式）。
    """
    input_source: str     # held-out 输入 lambda DSL（如 "lambda x: Pow(x,2)"）
    expected_source: str  # 期望输出 lambda DSL（如 "lambda x: 2*x"·cross-verify 执行等价基准）


@dataclass(frozen=True)
class TransformSpec:
    """符号变换规则规格（教师陈述·符号数学扩展 Phase 3）。

    一条变换规则 = (rule_name, lhs_source, rhs_source, held_out)：
      - rule_name : 规则名（register_transform_rule 键·如 "ddx_pow"）
      - lhs_source : LHS 模式 lambda DSL（如 "lambda b,n: Pow(b,n)"·PARAM 槽=通配符）
      - rhs_source : RHS 模板 lambda DSL（如 "lambda b,n: n*Pow(b,n-1)"·同 arg 序对齐 PARAM）
      - held_out : tuple[TransformHeldOut, ...] held-out 验证对（apply 规则到 input·cross-verify output==expected）。

    教师陈述模板（数据驱动·非硬编码 formula·humans 学法：从教师/课本学规则陈述+验证+应用+关联）。
    spec 来自 corpus（CollectedItem.transform_specs·R6 独立源·非教师录制·非学生编译）。
    """
    rule_name: str
    lhs_source: str
    rhs_source: str
    held_out: tuple[TransformHeldOut, ...] = ()


@dataclass(frozen=True)
class InverseRelationSpec:
    """运算间逆关系规格（教师陈述·S8 符号间运算关联·doc/重来_S8符号间关联机制设计_2026-07-15 §四/§五）。

    一条逆关系 = (relation_name, rule_a, rule_b, sample_sources)：两条独立变换规则 A 与 B 互逆
    （B∘A = identity @ 采样·构造性验证·反 theater 心脏）：
      - relation_name : 关系名（register_inverse_relation 键·如 "double_halve_inv"·确定性 surface __rel_inv_*）
      - rule_a : 规则 A TransformSpec（须 rule_name + lhs/rhs·held_out 可空·逆验证不依赖单规则 held-out）
      - rule_b : 规则 B TransformSpec（A 的逆·B∘A 须还原 A 域输入）
      - sample_sources : tuple[str, ...] 采样输入 e lambda DSL（B∘A 须还原这些 @ 探针·如 "lambda x: x+3"）。

    教师陈述（humans 学法：从教师/课本学"d/dx 和 ∫ 互逆"+构造验证·非纯归纳发现·research-grade defer）。
    数据驱动·非硬编码"+ −互逆"·code 是通用验证+存储机制·关系是数据。
    spec 来自 corpus（CollectedItem.inverse_relation_specs·single-source 教师·verify_source=SELF_PRODUCED·
    非 R6 两源·守"全自产不准停"·反 theater）。
    """
    relation_name: str
    rule_a: TransformSpec
    rule_b: TransformSpec
    sample_sources: tuple[str, ...] = ()


# ---- 输入层契约（6Q·§十一） ----
@dataclass
class Segment:
    """已切分段（文本=自然段 / 代码=函数体+token cap）。

    parsed 为预处理产物（parse_segment 填充·语言首版实·非语言骨架 defer）。
    首版语言：tokens/role_seq 由预处理（tokenize+emergent_role）填·observe 消费。
    """

    seg_id: int
    modality: int = MODALITY_LANGUAGE
    lang: int = LANG_ZH
    domain: int = DOMAIN_TEXT
    # 预处理产物（parse_segment 填·observe 读）——语言首版字段：
    tokens: list[str] = field(default_factory=list)
    role_seq: list[int] = field(default_factory=list)   # 每 token 的 role·对齐 tokens
    # 因果对（token index 对·预处理/指向词提取填）
    structured_causal_pairs: list[tuple[int, int]] = field(default_factory=list)
    cue_based_causal_pairs: list[tuple[int, int]] = field(default_factory=list)
    # IS_A 对（token index 对·系词提取填·致命3 来源②·child→parent）
    is_a_pairs: list[tuple[int, int]] = field(default_factory=list)
    # 时序 cue 对（刀 A·token index 对·A 先于 B·PRECEDES_CUE_FORWARD 提取·闭包传验序器·不入图）
    precedes_pairs: list[tuple[int, int]] = field(default_factory=list)
    # 数值等式声明（刀 B·(left_num, op_opcode, right_num, result_num) 4-int tuple·
    # NUM OP NUM 等于 NUM 提取·闭包传 numeric_proof_fn 检查·不入图·构造性检查 SELF_PRODUCED）
    numeric_claims: list[tuple[int, int, int, int]] = field(default_factory=list)
    # 全称量化声明（刀 C·(child_idx, parent_idx) token index 对·X 都是 Y·resolve 在验序器·
    # ConceptNet 外部源验·构造性验证 EXTERNAL·三值逻辑·不入图·详 doc/重来_刀C量化cue设计_2026-07-08.md）
    universal_claims: list[tuple[int, int]] = field(default_factory=list)
    # 存在量化声明（(child_idx, parent_idx) token index 对·有的 X 是 Y；本字段不携带真值，
    # 后续验证需要 MEMBER/nonempty/overlap/DISJOINT typed Evidence）。
    existential_claims: list[tuple[int, int]] = field(default_factory=list)
    # 属性命题声明（G1+#774·(subject_idx, attr_type_idx, value_idx, _reserved, polarity, modality) 6-int tuple·
    # "X 的 Y 是 Z" 提取·observe build_property_edges 建命题节点+PROPERTY 出边·G3b 读判同(subject,attr_type)
    # 多值结构矛盾·attr_type_idx<0=无 attr_type（"具有 Z"模式·首版 defer·build skip）·确定性 surface·入图非闭包传·
    # P0.3 命题节点扩展：polarity(0=肯定/1=否定)+modality(0=实然/1=必然□/2=可能◇/3=道义必然/4=道义可能)进 surface
    # 后缀（pol/mod>0 才加·default 0=既有命题 bit-identical）·B1 否定/B2 情态填值·build 防御读 claim[4]/[5] 缺省 0）
    property_claims: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)
    # 比较声明（刀 D·(left_num, cmp_opcode, right_num) 3-int tuple·NUM 比较OP NUM 提取·
    # 闭包传 comparison_proof_fn 验序（cross_compare）·不入图·构造性检查 SELF_PRODUCED·同刀 B 数值范式）
    comparison_claims: list[tuple[int, int, int]] = field(default_factory=list)
    # 相似声明（STEP5 PR4·(left_idx, right_idx) 2-int tuple·"X 像 Y" 提取·observe build_similar_edges
    # 建 EDGE_SIMILAR 边（X→Y·TIER_SHADOW·strength=1·非向量 D2 合规·dispatch_slot slot-filler 扩展候选））
    similar_claims: list[tuple[int, int]] = field(default_factory=list)
    has_negation: bool = False
    # has_condition/cond_p/cond_q 2026-07-09 删（CONDITION 写侧 YAGNI 清理·总收口 §五1.2·
    # 无 parser 设此字段+零读侧消费者·EDGE_CONDITION=7 保留注册登记但不激活）。
    has_implicit_causal_gap: bool = False
    implicit_causal_pair: tuple[int, int] | None = None
    # 代词（人称 anaphora·token index 集）
    pronoun_indices: set[int] = field(default_factory=set)
    # 空间模态图元（非语言·骨架 defer）
    spatial_primitives: list[Any] = field(default_factory=list)
    # 指向词/同位语线索（REFERS_TO 性质A 来源②·token index 对）
    alias_cue_pairs: list[tuple[int, int]] = field(default_factory=list)
    # 代码模态源码（MODALITY_CODE·A3·Python 源码字符串·observe code_observe 建 COMPOSES 树）
    # 语言模态为 None（A3 代码域专用·doc/重来_A3_代码域observe设计补充.md）
    code_source: str | None = None
    # 算术模态记号（MODALITY_ARITH·A3 兄弟件·lambda DSL 字符串·observe arith_observe 建 COMPOSES 树）
    # 语言/代码模态为 None（算术域专用·doc/重来_算术域observe设计补充.md）
    arith_source: str | None = None
    # 篇章结构序（缺口①·修正分析九v2·chapter_seq_table 独立扩展表·段 struct_ref 章节标记）
    # 机器可读结构源 parse 填（HTML h1-h6 / Markdown #/## / LaTeX \section / 代码 AST·文学卷章回 defer）·
    # 默认 0=无章节标记（无标记主流文本·退化同流水账·章节承载 defer 钥匙①）·向后兼容（既有 Segment 零改）
    chapter_seq: int = 0
    section_seq: int = 0
    token_spans: list[tuple[int, int]] = field(default_factory=list)
    document_token_indices: list[int] = field(default_factory=list)
    occurrence_ordinals: list[int] = field(default_factory=list)


@dataclass
class InputPayload:
    """observe 输入（§十一 6Q）。

    segments 已切分；source ∈ SOURCE_*（edge_store）；stage 决定落点；
    modality/lang/domain §7.4 标记。
    """

    segments: list[Segment]
    source: int
    stage: int
    modality: int = MODALITY_LANGUAGE
    lang: int = LANG_ZH
    domain: int = DOMAIN_TEXT
    weaning_phase: int = WEANING_PRE
    # 卷三 judge 自锚于输入（§十四自评命门破解）——intent + key_skeleton 由输入层填。
    intent: "IntentType" = field(default_factory=lambda: IntentType())
    key_skeleton: list[ConceptRef] = field(default_factory=list)   # J1 覆盖关键骨架子集
    item_key: int = 0   # 维度桥的 document scope registry 索引；哈希只作索引，完整身份见 scope_identity
    scope_identity: ScopeIdentity | None = None   # document/episode/query/generation 完整运行 scope；新断言生产路径应显式提供
    source_ref: SourceRef | None = None
    occurrence_scope_identity: ScopeIdentity | None = None   # 来源 occurrence 的稳定 scope；不得使用随 stage/round 变化的观察 episode scope
    raw_text: str | None = None
    speaker_identity: ObjectIdentity | None = None
    source_license_id: str | None = None
    source_batch_id: int | None = None

    def __post_init__(self) -> None:
        """核验显式来源、原文和 speaker 身份不会与旧整数 source 静默冲突。"""
        if self.source_ref is not None:
            if self.source_ref.source_kind != self.source:
                raise ValueError("InputPayload.source 与 SourceRef.source_kind 不一致")
        if self.occurrence_scope_identity is not None:
            if self.source_ref is None:
                raise ValueError("occurrence_scope_identity 必须同时携带 SourceRef")
            if self.occurrence_scope_identity.source != self.source_ref:
                raise ValueError("occurrence scope 必须指向同一 SourceRef")
        if self.raw_text is not None and not isinstance(self.raw_text, str):
            raise TypeError("InputPayload.raw_text 必须是字符串或 None")
        if (self.speaker_identity is not None
                and not isinstance(self.speaker_identity, ObjectIdentity)):
            raise TypeError("speaker_identity 必须是 ObjectIdentity 或 None")
        if (self.source_license_id is None) != (self.source_batch_id is None):
            raise ValueError("来源许可和 batch 必须同时声明或同时省略")
        if self.source_license_id is not None:
            if not isinstance(self.source_license_id, str) or not self.source_license_id:
                raise ValueError("source_license_id 必须是非空字符串")
            assert_int(self.source_batch_id, _where="InputPayload.source_batch_id")
            if type(self.source_batch_id) is not int or self.source_batch_id < 0:
                raise ValueError("source_batch_id 必须是非负严格整数")


@dataclass
class ObserveResult:
    """observe 产出（建好的图 + 落对的边 + defer 项 + 段结构概念 ref）。"""

    built_concepts: int = 0
    built_edges: int = 0
    deferred: list[str] = field(default_factory=list)
    # 段结构概念 ref 序（每段一个 struct_ref·inter-segment PRECEDES 串链·
    # formal_train 取首/末做 episode seed/sink + 全量做 key_skeleton·2026-07-02 落）
    struct_refs: list[ConceptRef] = field(default_factory=list)
    occurrence_refs: list[TypedRef] = field(default_factory=list)
    segment_occurrence_refs: list[list[TypedRef]] = field(default_factory=list)
    order_fact_assertion_hashes: list[int] = field(default_factory=list)
    span_refs: list[TypedRef] = field(default_factory=list)
    span_statement_assertion_hashes: list[int] = field(default_factory=list)
    prediction_results: list[Any] = field(default_factory=list)
    sense_candidate_traces: list[Any] = field(default_factory=list)
    semantic_course_run: Any = None


# ---- SpaceContext：observe 持有的三空间 + 阶段/开关 ----
@dataclass
class SpaceContext:
    """三空间上下文（observe 建图落点）。

    core         核心概念空间（纯整数·无衰减·训练期增长·训练后固化）。
    memory_read  记忆一层（阅读·带衰减·经伴随检疫晋升）。
    memory_interact 记忆二层（交互·全量·带时序）。
    companion    伴随库（原输入文本留档·sign=0 隔离）。
    memory_active ZERO_AI_MEMORY_ACTIVE（训练期 False·训练后 True·§十三）。
    """

    core: Any            # AbstractSpace
    memory_read: Any     # MemorySpace | None
    memory_interact: Any  # MemorySpace | None
    companion: Any       # CompanionSpace | None
    stage: int = STAGE_TRAINING
    memory_active: bool = False
    weaning_phase: int = WEANING_PRE


# ---- 卷二过程建模类型（Stage 4 填实·守单向依赖） ----

# 终点类型（terminal·§十三D-E3）
TERMINAL_REACHED_SINK = 1   # 达 sink ∧ J4 闭合（J4 卷三真判·卷二占位 true）
TERMINAL_DEAD_END = 2       # 死路（模块6 三条件任一）

# 死路 reward 常量（步进死路产负·§十三D-E3·防塌柱② greenfield）
REWARD_DEAD_END = -1

# 意图类型（§十一缺口#2·问句 sink=悬空槽 / 命令 sink=目标 / 陈述无 sink）
INTENT_QUESTION = 1     # 问句·sink=悬空槽（待填）
INTENT_COMMAND = 2      # 命令·sink=目标节点
INTENT_STATEMENT = 3    # 陈述·无 sink（仅建图·不步进取证）


@dataclass
class IntentType:
    """意图（步进终点判定 + judge 意图分类用）。

    sink    终点 ConceptRef | None（问句=悬空槽/命令=目标/陈述=None）。
    type    INTENT_*（context_tag 多维之一·落点② memory context_tag）。
    is_causal_reasoning              因果机制推理意图（J3 激活·G3a CAUSES 锚硬否决）。
    is_structural_sequence_reasoning 结构序推理意图（代码执行序/证明步骤序·J3 归零跳过·H3）。
    has_value_claim                  含值主张意图（G3b 反事实层a 激活·R4 写回核心）。

    三标志由输入层按 domain+type 判（§十四·首版 caller 填·oracle 标定后细化）。
    默认全 False = 事实 QA 非推理意图（J3 归零·G3a=1 跳过）。
    """

    type: int = INTENT_STATEMENT
    sink: ConceptRef | None = None
    is_causal_reasoning: bool = False
    is_structural_sequence_reasoning: bool = False
    has_value_claim: bool = False


@dataclass
class Step:
    """DAG-path 单步（模块4·F1 落盘）。

    node            步进到的节点。
    head            本步按头分发（PRECEDES AND / CAUSES OR）。
    selected_edges  选定边集（AND=全前驱边 / OR=选中前驱边·存非派生·§十四DAG-path契约）。
    """

    node: ConceptRef
    head: int
    selected_edges: list[EdgeRef] = field(default_factory=list)


@dataclass
class PathData:
    """卷二 DAG-path 步进产出（存非派生·§十四DAG-path契约·F1 落盘）。

    steps            list[Step]·步进序列。
    edges            list[EdgeRef]·选定边集（随 steps 存非派生·reward 反传读此）。
    struct_unit_refs 沿途汇聚点/结构单元 ref（F8 info_ref 链·G5 回溯路径记忆项）。
    """

    steps: list[Step] = field(default_factory=list)
    edges: list[EdgeRef] = field(default_factory=list)
    struct_unit_refs: list[ConceptRef] = field(default_factory=list)


@dataclass
class PathResult:
    """卷二路径结果（模块4 产出·.path 是 PathData·DAGPath 别名）。

    path         PathData（步进选择层·reward 反传读 path.edges 选定 CAUSES 边）。
    terminal     TERMINAL_*（REACHED_SINK / DEAD_END）。
    sink         终点 ConceptRef | None（REACHED_SINK 时填·DEAD_END 时 None）。
    topo_layers  FULL 相关子图拓扑层（结构展示层·生成侧读·superset of path）。
    convergence  汇聚点 map（{(node,head): (preds, conv_count)}·卷三识别结构单元）。
    source       起源 ConceptRef | None（topo_layers[0][0]·落点⑤ pass_reward 用）。
    """

    path: PathData = field(default_factory=PathData)
    terminal: int = TERMINAL_DEAD_END
    sink: ConceptRef | None = None
    topo_layers: list[list[ConceptRef]] = field(default_factory=list)
    convergence: dict = field(default_factory=dict)
    source: ConceptRef | None = None
    exploration_injected: bool = False   # 防塌柱③ proactive 注入标记（dag_path EXPLORATION_MODE 方差趋平时注入新种子）


# DAGPath 别名（卷二·.path → PathData）
DAGPath = PathResult


# ---- Layer0 外部锚门：verify episode 来源溯源（构造性检查≠构造性验证·防 cue 自产边 theater） ----
# 分层墙认知更正 §八b "找到就停纪律"：停止决策前查依据至少一路外部来源·全自产不准停。
# 标记在 **Episode**（非边）·不碰 #355 EDGE_PRECEDES epistemic_origin·刀A Option A 时序边不入图继续成立。
# 消费者：cognition/result/layer0_anchor.py 守门函数 + experiments/capability_exam.project_layer0。
VERIFY_SOURCE_NONE = 0          # 非 verify episode（reward 通道·judge 产·经验统计·不声称构造性）
VERIFY_SOURCE_EXTERNAL = 1      # 外部独立源 R6（vm_proof·expected 来自 corpus·真构造性验证·可驱动停止决策）
VERIFY_SOURCE_SELF_PRODUCED = 2 # 系统自产（time_seq·cue 对+token 序 single-source·构造性检查·非验证·全自产不准停）


@dataclass
class Episode:
    """卷二/三 episode 聚合层（模块9 产出·防塌/收敛验收消费 Episode 非OutputResult·F5）。

    G_meta 5字段 veto 写回（卷三 D1 跨卷最严重·R4 加 G3b/G5 须消费者同改）。
    reward 符号契约：judge 产 ≥0 / 步进死路产 <0 / propagate 接收可负（R1 episode 级）。
    """

    episode_id: int = 0
    run_id: int = 0
    input: Any = None
    output: Any = None
    reward: int = 0
    ref: ConceptRef | None = None             # sink ref
    terminal: int = TERMINAL_DEAD_END
    pr_vector: dict = field(default_factory=dict)   # 本 episode PR 向量（防塌柱③方差读）
    judge_G4_active: bool = False
    judge_G2p_active: bool = False
    judge_G3a_active: bool = False
    judge_G3b_active: bool = False
    judge_G5_active: bool = False
    judge_veto_count: int = 0
    dead_end_count: int = 0
    vetoed: bool = False
    exploration_injected: bool = False   # 防塌柱③ 本 episode 是否 proactive 注入新种子（anti_collapse 柱③ falsifiable 读）
    verify_source: int = VERIFY_SOURCE_NONE   # Layer0 外部锚门来源（default NONE·向后兼容·reward 通道 episode 不声称构造性·verify 通道由 formal_train 填 EXTERNAL/SELF_PRODUCED）


@dataclass
class GMeta:
    """卷三 judge 门因子 5字段（G4/G2p/G3a/G3b/G5·Stage 5 填实·R4 跨卷消费者同改）。

    各字段 True=该门 veto（reward=0）。vetoed = 任一门 True。
    卷二模块9 读 G4_vetoed 等填 Episode（D1 落盘·死路用 G_META_DEAD_END 全 False）。
    """

    G4: bool = False
    G2p: bool = False
    G3a: bool = False
    G3b: bool = False
    G5: bool = False

    @property
    def vetoed(self) -> bool:
        return self.G4 or self.G2p or self.G3a or self.G3b or self.G5

    # _vetoed 别名（模块9伪代码用 G_meta.G4_vetoed 命名·R4 落盘）
    @property
    def G4_vetoed(self) -> bool:
        return self.G4

    @property
    def G2p_vetoed(self) -> bool:
        return self.G2p

    @property
    def G3a_vetoed(self) -> bool:
        return self.G3a

    @property
    def G3b_vetoed(self) -> bool:
        return self.G3b

    @property
    def G5_vetoed(self) -> bool:
        return self.G5


# 死路 G_meta 常量（5字段全 False·同 judge 初始化·死路无 G veto·D1 落盘）
G_META_DEAD_END = GMeta()


# ---- 卷三结果建模类型（Stage 5 填实·守单向依赖） ----

# 血统来源（lineage·§十四路径填槽·回放标 DEF_REPLAY 不伪装）
LINEAGE_CONCEPT_FILL = 1    # 单概念→词形路径填槽（主）
LINEAGE_DEF_REPLAY = 2      # 记忆序列回放直出（逐槽原语·辅）
# 命门③ 候选 B（doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）：cue 位直出 cue token 功能词
# （dispatch_slot cue 位早 return·surface_of(cue_sig[slot_idx])·绕 collide/selection_pref/correspondence）。
# cue token 是结构活化（功能词）非内容词·不入 emitted_tokens/produced_refs（generate.py 守）·仅入 lineage 标血统。
# gate CUE_SLOT_FILL_MODE OFF = cue 位走 collide 返 LINEAGE_CONCEPT_FILL=1·lineage 值集退 {1,2}·bit-identical。
CUE_SLOT_FILL = 3          # cue 位 cue token 直出（功能词插补·结构活化·命门③ 候选 B）

# J3path PRECEDES:CAUSES 权重比占位（B1·oracle 标·序边贡献一个数量级低防堆链游戏）
J3_CAUSES_WEIGHT = 10
J3_PRECEDES_WEIGHT = 1


@dataclass
class RoleSlot:
    """role_seq 的一个槽位（模块2 逐槽分派消费）。

    ref                      槽位概念 ref（单概念填槽主路）。
    role                     role 标记（OBJECT/ACTION/答案 sink 槽等·§十一缺口#1 降字段）。
    filler_is_memory_sequence 记忆空间种类3 序列节点（True→回放直出 DEF_REPLAY）。
    """

    ref: ConceptRef
    role: int = 0
    filler_is_memory_sequence: bool = False


@dataclass
class OutputPart:
    """一个结构单元的输出（模块1·沿 DAG 拓扑序多部分编排）。"""

    unit: ConceptRef                       # 结构单元/汇聚点 ref
    words: list[str] = field(default_factory=list)
    # P0 #1040：段 token concept ref 序（slot.ref 派发的真 token·与 words 等长·gate DISPATCH_TOKEN_CHAIN_MODE
    # ON 时 generate 填）。双消费：(a) carry_to_workmem 段满/章边界 carry 写 prior_topic_refs 用 token 级 ctx
    # （解 collide/sel_pref/pronoun 的 unit-vs-token 错节点）·(b) 统计层产出度量（#1041·读 token concept 验
    # 产出真词非 truthiness·判据①度量腿）。gate OFF 空 list（默认·既有构造零改·bit-identical）。
    token_refs: list[ConceptRef] = field(default_factory=list)


@dataclass
class OutputResult:
    """卷三生成结果（模块1 产出·judge 消费）。

    parts        list[OutputPart]·沿 DAG 拓扑序单 pass。
    lineage      LineageMap = {(unit, slot_index): LINEAGE_*}·标血统不伪装。
    reached_sink sink 是否在产出结构单元中（judge G2p 读·模块1 填·F6 落盘）。
    """

    parts: list[OutputPart] = field(default_factory=list)
    lineage: dict[tuple[ConceptRef, int], int] = field(default_factory=dict)
    reached_sink: bool = False
    # 对话止血②（2026-07-18）：generate fill loop 段内词数超 MAX_WORDS_PER_PART 截断时记该 unit。
    # gate OUTPUT_LEN_CAP_MODE OFF = 永空（bit-identical）。独立字段不污染 lineage 语义（lineage 值集 gate OFF 仍 {1,2}·
# gate CUE_SLOT_FILL_MODE ON 扩 {1,2,3}·CUE_SLOT_FILL=3 独立·与 OUTPUT_LEN_CAP_MODE 不撞·后者用本字段）。
    truncated_units: set[ConceptRef] = field(default_factory=set)

    @property
    def words(self) -> list[str]:
        """扁平词序（judge/反传读 output.parts 词·F6）。"""
        out: list[str] = []
        for p in self.parts:
            out.extend(p.words)
        return out


@dataclass
class FloorActivation:
    """floor 端到端下游激活率测量结果（断奶 critical path 第 2 件·反 theater 首版机制层预验·doc/重来_floor_端到端下游激活率_2026-07-17）。

    纯读 `_measure_floor_activation`（cognition/result/floor_measure.py）产。镜像 generate.py:153-171 per-unit stash
    逻辑做**读侧后验重导**（不改 generate 写侧·bit-identical）·对 held-out OutputResult.parts 的 cue slot 测：
    学到的对应词（D:11 W→REL_*·tally→promote·桥 P3 新信号）是否在 cue slot 正确激活。

    activation_permille       cue slot 激活率 ×1000（cue_rel_of(token_refs[slot])==unit_rel_kind 占比·纯整 //）。
    false_positive_permille   distractor 误激活率 ×1000（cue slot 选了 cue_rel_of≠rel_kind 或无 D:11 的词）。
    measured                  total>0（measured-guard·空探针/not-run→False→anchor_pf 不过·防 stub-0 vacuous）。
    total / activated         原始计数（透明·防黑箱·调试用）。
    """
    activation_permille: int = 0
    false_positive_permille: int = 0
    measured: bool = False
    total: int = 0
    activated: int = 0


@dataclass
class JudgeWeights:
    """judge 加权 {w1 J1覆盖, w2 J2意图, w3 J3因果, w4 J4word 产出真词}（oracle 标定纯整数·断奶后冻结）。

    per intent_type（§十四权重确定）·首版 env 默认 + oracle 标定·非硬编码。
    w4（#1041 构造②）：产出真词覆盖率权重·默认 1。**H2 标定不动 w4**（oracle.calibrate_weights 网格搜
      w1/w2/w3·`JudgeWeights(w1=,w2=,w3=)` 构造 → w4 落默认）·w4 标定 defer。gate OUTPUT_WORD_REWARD_MODE
      OFF → J4word=0 → w4 值无关（主守 bit-identical）·ON → w4·J4word 进 reward（反映产出真词质量·判据②③）。
    """

    w1: int = 1
    w2: int = 1
    w3: int = 1
    w4: int = 1


@dataclass
class CollapseReport:
    """防塌三柱验收报告（模块4·防塌三柱缺一即塌）。

    pillar1_ok 结构 judge 非自媚（任一 G veto active·judge 在工作）。
    pillar2_ok 真负通路 active（judge veto + 死路 都 failure→tn++）。
    pillar3_ok 探索压力（方差够 或 seeded 探索注入·③最小版进首版）。
    failure_count   judge_veto_count + dead_end_count（M7 同口径）。
    neg_reward_count 死路负 reward 计数（负值只来自步进死路·诊断用）。
    """

    pillar1_ok: bool = False
    pillar2_ok: bool = False
    pillar3_ok: bool = False
    failure_count: int = 0
    neg_reward_count: int = 0


@dataclass
class ConvergenceReport:
    """收敛判据报告（模块5·含负通路活跃·假收敛识别）。

    steady_state     比率方差低+导通率平台+promote 平台+负通路活跃+非塌信号。
    real_convergence 真收敛（非塌信号 ∧ 负通路活跃 ∧ steady_state）。
    collapse_signal  塌信号（sn/tn→1+PR 方差→0+负通路 failure 计数=0·M7 同口径=假收敛）。
    neg_pathway_active 负通路活跃（failure_count_recent>0·M7 同口径含 judge veto+死路）·
                      D2 断奶硬前置（断奶须负通路活跃·防 reward 永正趋平伪满足）。
    """

    steady_state: bool = False
    real_convergence: bool = False
    collapse_signal: bool = False
    neg_pathway_active: bool = False
