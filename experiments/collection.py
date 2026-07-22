"""experiments.collection — 五类收集框架（§十二五类收集 + E5 graceful + E10 local_dir 首选）。

五类收集（数据摄入·预处理层产出 CollectedItem → formal_train 喂 observe 建图）：
  ① COLLECT_CAUSES    逻辑因果（ConceptNet/证明/代码 AST→CAUSES·结构化源·tier PRIMARY）
  ② COLLECT_PRECEDES  结构序（裸文本/代码/QA→PRECEDES+role_seq·tier PRIMARY）
  ③ COLLECT_ABSTRACT  抽象（ConceptNet/维基/教师元定义→IS_A/PROPERTY·tier PRIMARY）
  ④ COLLECT_COOCCURS  共现（裸文本段内→COOCCURS·tier SHADOW·不进默认建模）
  ⑤ COLLECT_NOISE     噪音隔离（provenance 贯穿四类·按源分级·SHADOW 持有不删→promote 晋 PRIMARY）

**CollectionSource 协议（可换·pluggable）**：外部源解析器（ConceptNet SDK/代码 AST/证明 LaTeX）
  实现协议注入·系统定义契约·外包件填实现（§7.3 接口归系统）。源解析是数据摄入非系统设计核心。

**E10 local_dir 首选**：LocalDirSource 读 ZERO_AI_LOCAL_DIR 本地裸文本文件（用户第三方下载器
  下到 K:\\数据集）·纯本地读 = 确定性 + 离线可复现·免 SDK 版本漂移/网络限流/镜像坑。优先序：
  local_dir > SDK > 跳过（记失败源）。LocalDirSource 是首选源·SDK 源作兜底。

**E5 graceful 降级**：单源失败不破坏训练——available()=False 跳过非崩·collect() 异常降级·
  失败源显式记录进 CollectionReport（不静默吞错）·训练继续（少一个源的数据·数据可再下）。
  **降级语义边界**：收集侧可降级（数据可再下）·replay 侧不可降级（录制不可重建·同 E4 显式报错）。

铁律：纯整数（CollectedItem 字段全整 + tokens str 列表·observe 消费）/ 确定性（local_dir 纯本地读
  bit-identical·源序确定）/ 不写死（五类映射 edge_type 非硬编码语义·源解析 pluggable）/
  外部只启发（源给数据·observe 建图判分流·源不注入边语义）。
诚实边界：源解析给候选非语义真伪（接地墙）/ local_dir 预处理是裸文本最小切分（tokenize/emergent_role
  首版 caller-fill·§十一 6Q defer）/ graceful 是软降级非硬兜底（数据可再下≠数据完整）/
  COOCCURS SHADOW 持有不删（promote 三重才晋 PRIMARY·非收集期判定）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.storage.edge_types import (
    EDGE_CAUSES, EDGE_PRECEDES, EDGE_IS_A, EDGE_PROPERTY, EDGE_COOCCURS,
)
from pure_integer_ai.storage.edge_store import (
    SOURCE_CONCEPTNET, SOURCE_CODE, SOURCE_MATH, SOURCE_QA, SOURCE_BARE_TEXT, SOURCE_TEACHER,
)
from pure_integer_ai.cognition.shared.types import (
    MODALITY_LANGUAGE, MODALITY_ARITH, MODALITY_CODE,
    LANG_ZH, LANG_EN, LANG_NONE,
    DOMAIN_TEXT, DOMAIN_CODE, DOMAIN_MATH,
    CodeSpec, ConceptRef, TransformSpec, TransformHeldOut, InverseRelationSpec,
)

# ---- 五类收集类型 ----
COLLECT_CAUSES = 1       # ① 逻辑因果 → CAUSES
COLLECT_PRECEDES = 2     # ② 结构序 → PRECEDES + role_seq
COLLECT_ABSTRACT = 3     # ③ 抽象 → IS_A / PROPERTY
COLLECT_COOCCURS = 4     # ④ 共现 → COOCCURS（SHADOW）
COLLECT_NOISE = 5        # ⑤ 噪音隔离（provenance 贯穿·非独立边类型）

COLLECT_TYPES: tuple[int, ...] = (
    COLLECT_CAUSES, COLLECT_PRECEDES, COLLECT_ABSTRACT, COLLECT_COOCCURS, COLLECT_NOISE,
)

# 五类 → 主建边类型（source_dist 审计 + tier 分级用·observe 实际按 segment 字段建图）
COLLECT_TYPE_EDGE: dict[int, int] = {
    COLLECT_CAUSES: EDGE_CAUSES,
    COLLECT_PRECEDES: EDGE_PRECEDES,
    COLLECT_ABSTRACT: EDGE_IS_A,
    COLLECT_COOCCURS: EDGE_COOCCURS,
    COLLECT_NOISE: EDGE_COOCCURS,   # 噪音挂共现 SHADOW（provenance 隔离）
}

# 五类默认 tier（①②③ PRIMARY 结构化源 / ④⑤ SHADOW 持有不删）
COLLECT_TYPE_TIER_PRIMARY: frozenset[int] = frozenset(
    {COLLECT_CAUSES, COLLECT_PRECEDES, COLLECT_ABSTRACT}
)


@dataclass
class CollectedItem:
    """单条收集项（预处理产出·formal_train 喂 observe 建图）。

    tokens         段 token 序（裸文本 tokenize / 结构化源重构为 token 序·observe 消费）。
    role_seq       每 token role（COLLECT_PRECEDES 填·代码 role=语句类型·emergent_role 冷启动）。
    causal_pairs   因果对 token index（COLLECT_CAUSES 填·结构化源/指向词提取）。
    is_a_pairs     IS_A 对 token index（COLLECT_ABSTRACT 填·ConceptNet IsA/系词提取·child→parent）。
    alias_cue_pairs 同指线索对（COLLECT_ABSTRACT/REFERS_TO 性质A 来源②·同位语/又名）。
    collect_type   五类之一（provenance 审计 + tier 分级）。
    source         SOURCE_*（edge source 列·按源分级）。
    strength       边 strength（共现=频次 / 结构化=1）。
    domain/lang/modality  §7.4 标记（三空间分流 + C1 防跨语言）。
    code_source    代码模态源码（MODALITY_CODE·Python 源码字符串·observe code_observe 建 COMPOSES 树）。
    code_specs     代码域测试用例规格序（C6·R6 独立源·一函数多 spec·vm_proof_fn 验执行 vs expected）。
                   语言模态留 ()（A3 代码域专用·doc/重来_A3_代码域observe设计补充.md §二致命#2）。
    arith_source   算术模态记号（MODALITY_ARITH·lambda DSL 字符串·observe arith_observe 建 COMPOSES 树·
                   Sigma/Prod/Recur/闭式·doc/重来_算术域observe设计补充.md）。
    arith_specs    算术域测试用例规格序（复用 CodeSpec·R6 独立源·vm_proof_fn 验执行 vs expected）。
                   语言/代码模态留 ()（算术域专用）。
    """

    tokens: list[str] = field(default_factory=list)
    raw_text: str | None = None   # L-01 正式词形 provider 输入；None 保留已分词/结构化来源，非 None 时 formal_train 在所有消费者前统一分词
    word_form_parse: Any = field(default=None, compare=False, repr=False)   # L-02 多边界 Hypothesis/Evidence 结果；tokens 只保存当前兼容 winner
    boundary_profile: Any = field(default=None, compare=False, repr=False)   # U-03 上游显式句界 Evidence；不得由 CollectedItem 自行按字符作用生成
    boundary_parse: Any = field(default=None, compare=False, repr=False)   # U-03 来源化候选全集；probe 预览与 training ledger 提交分离
    boundary_decision: Any = field(default=None, compare=False, repr=False)   # U-03 当前 active 图选择或纯预览决定；分段器只消费其 token cut 投影
    role_seq: list[int] = field(default_factory=list)
    causal_pairs: list[tuple[int, int]] = field(default_factory=list)
    is_a_pairs: list[tuple[int, int]] = field(default_factory=list)
    alias_cue_pairs: list[tuple[int, int]] = field(default_factory=list)
    collect_type: int = COLLECT_PRECEDES
    source: int = SOURCE_BARE_TEXT
    strength: int = 1
    domain: int = DOMAIN_TEXT
    lang: int = LANG_ZH
    modality: int = MODALITY_LANGUAGE
    code_source: str | None = None
    code_specs: tuple[CodeSpec, ...] = ()
    arith_source: str | None = None
    arith_specs: tuple[CodeSpec, ...] = ()
    arith_source_b: str | None = None   # Mode B cross-verify 参树 DSL（异 shape·迭代 vs 闭式·同函数第二表达·POST-weaning 统计一致加强腿·formal_train POST 路径 cross_verify_pair 激活·两路独立编译 execute_composes_value + rational.eq·None=退化 bit-identical·doc/重来_ModeB自洽设计补充.md §七）
    code_source_b: str | None = None   # Mode B cross-verify CODE 域参树 Python 源码（异 shape·同函数第二表达·对称 arith_source_b·POST-weaning 统计一致加强腿·build_composes_from_source 二次独立建·两路 execute_composes_value + rational.eq·None=退化 bit-identical·§施工序 1.2）
    action_specs: tuple[CodeSpec, ...] = ()   # 断桥 Phase A（P2 G-PR2/3·doc/重来_断桥设计refinement_2026-07-15）：教师标 I/O 例（数据驱动·**非硬编码**·language/action item 经此跨路径喂 synthesize_value 联合匹配·**spec→synthesis**·intent 分类=Phase B 动态构造器·Phase A 教师标 specs 已含 intent 语义·审2 F1/F2/F3 修回 design 原 dict[action_ref]）·默认 ()（无教师标→ACTION_BRIDGE_MODE 路径不进→bit-identical）·Phase B 动态 intent→spec 构造器 defer
    numeric_claims_flat: tuple[tuple[int, int, int, int], ...] = ()   # 断桥 Phase B 片1 数据桥（P2·doc/重来_断桥设计refinement_2026-07-15 §Phase B 片1）：observe 期 flatten raw.segments[*].numeric_claims（刀B extract_numeric_claims_gated 产·4-tuple `(left,op,right,result)`·mirror :386 code_struct_ref 捕获范式·ungated 纯缓存·NUMERIC_PROOF_MODE OFF→seg 空→flat 空→bit-identical）·`_run_task_driven_generate` Phase B block 读（gate ACTION_BRIDGE_CUE_MODE）→ CodeSpec 隐 op 联合匹配 synthesize_value。默认 ()（无 numeric cues→Phase B 路径不进→bit-identical）。
    expected_skeleton: ConceptRef | None = None   # S7 相0 钥匙③：教师标定该段应命中骨架 ref（跨 run 引用已注册算子 skeleton_ref·断奶前教师路径·None=退化 bit-identical·POST 退场）。**教师天花板**：主观标非闭式真理·可能多解/错标（错标→op_confidence sn=0 拉低 rate·反 theater 降权）·断奶后须相2/E1 接力（钥匙③墙≡#479）
    transform_specs: tuple[TransformSpec, ...] = ()   # 符号数学扩展 Phase 3（doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis）：教师陈述符号变换规则（rule_name+lhs/rhs lambda DSL+held-out 验证对·数据驱动**非硬编码**·humans 学法：从教师/课本学规则陈述+验证+应用+关联）·`_run_task_driven_generate` Phase 3 block 读（gate SYMBOLIC_TRANSFORM_MODE）→ register_transform_rule + apply held-out + cross-verify 执行等价 → 独立 task-driven episode（weaning-safe 决断 A·不替换 vm_proof·不碎 W7·同断桥 Phase A/B 范式）。默认 ()（无 transform_specs→Phase 3 路径不进→bit-identical）。
    inverse_relation_specs: tuple[InverseRelationSpec, ...] = ()   # S8 符号间运算关联（doc/重来_S8符号间关联机制设计_2026-07-15 §四/§七）：教师陈述逆关系（relation_name+rule_a/rule_b TransformSpec+sample_sources·两条独立变换规则互逆·数据驱动**非硬编码**·humans 学法：从教师/课本学"两规则互逆"+构造验证）·`_run_task_driven_generate` S8 block 读（gate SYMBOLIC_RELATION_MODE·SYMBOLIC_TRANSFORM 块后）→ register rule_a/b + register_inverse_relation + verify_inverse_relation（B∘A 还原 @ 采样·三值）→ verified 则独立 task-driven episode（weaning-safe 决断 A·不替换 vm_proof·不碎 W7·同 transform_specs 范式·verify_source=SELF_PRODUCED）。默认 ()（无 inverse_relation_specs→S8 路径不进→bit-identical）。
    code_struct_ref: ConceptRef | None = None   # #730 路径 W：observe 期建的 code COMPOSES 根（__prog_*·obs.struct_refs[0]）·
                   # task-driven 代码模态 unparse 读（候选 A·observe 建树一次·task-driven 纯读·幂等守 bit-identical·
                   # 确定性：code struct_ref=__prog_{stage}_{h63(code_source)}·跨 round 稳定·observe guard 防重 build）。
                   # 非 code 模态 / observe 未建树 → None（task-driven 跳过·诚实）。向后兼容默认 None（既有 item 零改）。
    source_ref: SourceRef | None = field(default=None, compare=False)   # 稳定来源记录；生产 source 应显式填，旧 fixture 可由 corpus_identity 补匿名来源
    document_scope_hash: int = field(default=0, compare=False, repr=False)   # identity registry 索引缓存；完整身份仍在 SourceRef/ScopeIdentity
    speaker_identity: ObjectIdentity | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        assert_int(self.collect_type, self.source, self.strength,
                   self.domain, self.lang, self.modality,
                   self.document_scope_hash,
                   _where="CollectedItem.__post_init__")
        if self.raw_text is not None and not isinstance(self.raw_text, str):
            raise TypeError("CollectedItem.raw_text 必须是字符串或 None")
        if self.source_ref is not None and self.source_ref.source_kind != self.source:
            raise ValueError("CollectedItem.source 与 SourceRef.source_kind 不一致")
        if (self.speaker_identity is not None
                and not isinstance(self.speaker_identity, ObjectIdentity)):
            raise TypeError("CollectedItem.speaker_identity 必须是 ObjectIdentity 或 None")


@runtime_checkable
class CollectionSource(Protocol):
    """收集源协议（可换·pluggable·§7.3 接口归系统）。

    外部源解析器（ConceptNet SDK / 代码 AST / 证明 LaTeX / 裸文本目录）实现此协议注入。
    available()=False → 跳过该源（E5 graceful·非崩）·collect() 异常 → 降级记失败源。
    """

    def name(self) -> str: ...
    def available(self) -> bool: ...
    def collect(self) -> list[CollectedItem]: ...


@dataclass
class CollectionReport:
    """收集报告（五类计数 + 失败源显式记录·E5 graceful 不静默吞错）。"""

    items: list[CollectedItem] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)
    counts_per_type: dict[int, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.items)


# ---- E10 local_dir 首选源 ----

def _resolve_local_dir(local_dir: str | None) -> str | None:
    """解析 local_dir（参数 > ZERO_AI_LOCAL_DIR env > None·E10 首选）。"""
    d = local_dir or os.environ.get("ZERO_AI_LOCAL_DIR")
    if d is None:
        return None
    return d if os.path.isdir(d) else None


def _split_paragraphs(text: str) -> list[str]:
    """裸文本切自然段（H7 段边界·文本段=自然段·语义连贯单元）。

    首版最小切分：按空行切段·段内按空白 tokenize。语言首版 caller-fill（§十一 6Q defer
    真 tokenize/emergent_role）。纯本地读 = 确定性 bit-identical。
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def _tokenize(text: str) -> list[str]:
    """生成未启用课程 provider 时使用的空白分词兼容结果。

    LocalDirSource 同时保留 raw_text；L-01 正式入口会在所有训练消费者前用课程 FMM
    覆盖本结果。未配置 provider 时继续按空白切，保持既有语料行为。
    """
    return [t for t in text.split() if t]


class LocalDirSource:
    """本地目录源（E10 首选·读 ZERO_AI_LOCAL_DIR 裸文本文件·纯本地读离线可复现）。

    目录下 *.txt 文件 → 每自然段一 CollectedItem（COLLECT_PRECEDES 裸文本序 +
    observe 内部建段内 COOCCURS SHADOW）·source=SOURCE_BARE_TEXT。
    优先序：local_dir > SDK > 跳过。SDK 源作兜底（local_dir 缺该源时）·SDK 失败走 E5 降级。
    """

    def __init__(self, local_dir: str | None = None, *,
                 lang: int = LANG_ZH, domain: int = DOMAIN_TEXT) -> None:
        self._dir = _resolve_local_dir(local_dir)
        self._lang = lang
        self._domain = domain

    def name(self) -> str:
        return f"local_dir:{self._dir or '<unset>'}"

    def available(self) -> bool:
        return self._dir is not None

    def collect(self) -> list[CollectedItem]:
        if self._dir is None:
            return []
        items: list[CollectedItem] = []
        # 确定性序：文件名升序（bit-identical·跨宿主一致）
        for fn in sorted(os.listdir(self._dir)):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(self._dir, fn)
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            file_source_id = Hasher("local_dir.source_file.v1").h63(
                (os.path.normcase(os.path.abspath(self._dir)), fn))
            if file_source_id == 0:
                file_source_id = 1
            for paragraph_index, para in enumerate(_split_paragraphs(text)):
                tokens = _tokenize(para)
                if not tokens:
                    continue
                items.append(CollectedItem(
                    tokens=tokens,
                    raw_text=para,
                    collect_type=COLLECT_PRECEDES,
                    source=SOURCE_BARE_TEXT,
                    strength=1,
                    lang=self._lang,
                    domain=self._domain,
                    source_ref=SourceRef(
                        SOURCE_BARE_TEXT,
                        file_source_id,
                        paragraph_index,
                        GLOBAL_OWNER_SCOPE,
                        VersionBundle(),
                    ),
                ))
        return items


class InMemorySource:
    """内存源（测试/确定性种子语料·bit-identical·非外部依赖）。"""

    def __init__(self, items: list[CollectedItem], *,
                 name: str = "in_memory") -> None:
        self._items = list(items)
        self._name = name

    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return True

    def collect(self) -> list[CollectedItem]:
        return list(self._items)


# ---- IS_A facts 本地文件 loader（刀0·来源① ConceptNet·E10 纯本地读·boot 时种 EDGE_ISA 边） ----

def load_is_a_facts_file(path: str) -> list[tuple[str, str]]:
    """读 IS_A facts 文件（E10 纯本地读·每行 "child parent"·ConceptNet IsA 三元组导出格式）。

    每行空白切·首段=child surface·末段=parent surface（中段忽略·容错·支持 "猫 是一种 动物" 带系词）。
    `#` 注释行 skip·空行 skip·格式错行（<2 段）skip + **不抛崩**（E5 graceful·数据可再下·错行不破训练）。
    自环（child==parent）skip（build_is_a_edge:57 亦跳·此处早跳省 ensure 副作用）。

    **非 core 数据**（合规守「不写死」）：core 永不 import 此文件·core 不知 pair 内容·loader 是 pluggable
    数据摄入（同 LocalDirSource 范式·§7.3 接口归系统）·生产 default 无文件→resolve 返空→boot 零副作用。
    与 cue_words frozenset（元定义非语义 enum 例外）不同范畴——cue_words 是 core 句法锚（非语义）·
    IS_A facts 文件是外部语义断言数据（ConceptNet 客观断言·非系统教师判断·经 build_is_a_edge 落 PRIMARY）。

    返 list[(child_surface, parent_surface)]（caller `is_a.bootstrap_is_a_edges` 建 EDGE_ISA 边）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：文件真伪/质量 = 外部数据责任（接地墙）·系统不判·只落边。
    """
    pairs: list[tuple[str, str]] = []
    try:
        # utf-8-sig 自动 strip BOM（防 BOM 前置致首行注释/数据识别失败·对抗审点 2）
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) < 2:
                    continue   # 格式错行（<2 段）skip（E5 graceful·不抛崩）
                child, parent = parts[0], parts[-1]
                if child == parent:
                    continue   # 自环无义·早跳省 ensure
                pairs.append((child, parent))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


# lang 常量 → 文件名后缀（is_a_facts_{suffix}.txt·刀0 boot 种边按 lang 分文件）
_LANG_FILE_SUFFIX: dict[int, str] = {
    LANG_ZH: "zh",
    LANG_EN: "en",
}


def resolve_is_a_facts(lang: int, local_dir: str | None = None) -> list[tuple[str, str]]:
    """解析某 lang 的 IS_A facts（E10 本地文件·ZERO_AI_LOCAL_DIR·文件不存在→空）。

    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 / lang 无映射 → 返 []
    （E5 graceful·bit-identical 守·CI/生产 default 无文件→空→bootstrap 零副作用→ancestor_map 空）。

    返 list[(child_surface, parent_surface)]（caller 调 `is_a.bootstrap_is_a_edges` 建 EDGE_ISA 边）。

    铁律：纯整数（lang int·返 surface str 对）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return []   # 无映射的 lang（LANG_NONE 等）→ 空
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, f"is_a_facts_{suffix}.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_is_a_facts_file(path)


# ---- mereology facts 本地文件 loader（T-L1d·客观序 gap·来源① 结构化·E10 纯本地读·boot 时种 EDGE_MEREOLOGY 边） ----

def load_mereology_facts_file(path: str) -> list[tuple[str, str]]:
    """读 MEREOLOGY facts 文件（E10 纯本地读·每行 "part whole"·ConceptNet PartOf / WordNet meronym 有向三元组导出格式）。

    每行空白切·首段=part surface·末段=whole surface（中段忽略·容错·支持 "part [cue] whole" 整词首末对）。
    **数据格式约定**：part 在前·whole 在末（ConceptNet PartOf = part TAB whole·干净导出）。**注意** mereology
    自然表达"X 是 Y 的一部分"把 whole 放中间（末是"的一部分"）·与 is_a"X 是一种 Y"(Y 末) 不同·故本 loader
    要求干净 "part whole" 对（whole 末）·数据导出须把 "X 是 Y 的一部分" 归一为 "X Y"（cue 不入文件）。
    `#` 注释行 skip·空行 skip·格式错行（<2 段）skip + **不抛崩**（E5 graceful·错行不破训练·镜像 load_is_a_facts_file）。
    自环（part==whole）skip（build_mereology_edge:58 亦跳·此处早跳省 ensure 副作用）。

    **非 core 数据**（合规守「不写死」·同 load_is_a_facts_file）：core 永不 import 此文件·core 不知 pair 内容·
    loader 是 pluggable 数据摄入·生产 default 无文件→resolve 返空→boot 零副作用。
    mereology facts 是外部部分-整体断言（ConceptNet PartOf 客观断言·非系统教师判断·经 build_mereology_edge 落 PRIMARY）。

    返 list[(part_surface, whole_surface)]（caller `mereology.bootstrap_mereology_edges` 建 EDGE_MEREOLOGY 边）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：文件真伪/质量 = 外部数据责任（接地墙）·系统不判·只落边。**MEREOLOGY≠IS_A**（部分-整体≠子集）·
      本 loader 只落 part→whole 有向边·与 is_a_facts 语义正交（cue_words REL_MEREOLOGY 路由已修正不再误入 IS_A_CUE）。
    """
    pairs: list[tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动 strip BOM（防 BOM 前置致首行识别失败）
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) < 2:
                    continue   # 格式错行（<2 段）skip（E5 graceful·不抛崩）
                part, whole = parts[0], parts[-1]
                if part == whole:
                    continue   # 自环无义·早跳省 ensure
                pairs.append((part, whole))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


def resolve_mereology_facts(lang: int, local_dir: str | None = None) -> list[tuple[str, str]]:
    """解析某 lang 的 MEREOLOGY facts（E10 本地文件·ZERO_AI_LOCAL_DIR/mereology_facts_{lang}.txt·文件不存在→空·镜像 resolve_is_a_facts）。

    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 / lang 无映射 → 返 []
    （E5 graceful·bit-identical 守·CI/生产 default 无文件→空→bootstrap 零副作用→零 MEREOLOGY 边）。

    返 list[(part_surface, whole_surface)]（caller 调 `mereology.bootstrap_mereology_edges` 建 EDGE_MEREOLOGY 边）。

    铁律：纯整数（lang int·返 surface str 对）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return []   # 无映射的 lang（LANG_NONE 等）→ 空
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, f"mereology_facts_{suffix}.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_mereology_facts_file(path)


# ---- antonym facts 本地文件 loader（T-L1e·客观序 gap·来源① 结构化·E10 纯本地读·boot 时种 EDGE_ANTONYM 边） ----

def load_antonym_facts_file(path: str) -> list[tuple[str, str]]:
    """读 ANTONYM facts 文件（E10 纯本地读·每行 "a b"·ConceptNet /r/Antonym / WordNet antonym 对导出格式·无序对称对）。

    每行空白切·首段=a surface·末段=b surface（中段忽略·容错）。**无序对称对**（a↔b·文件每对一次·canonical 序）。
    `#` 注释行 skip·空行 skip·格式错行（<2 段）skip + **不抛崩**（E5 graceful·镜像 load_mereology_facts_file）。
    自环（a==b·词非自身反义）skip（build_antonym_edge:58 亦跳·此处早跳省 ensure 副作用）。

    **非 core 数据**（合规守「不写死」·同 load_mereology_facts_file）：core 永不 import 此文件·loader pluggable·
    生产 default 无文件→resolve 返空→boot 零副作用。antonym facts 是外部反义对（ConceptNet/WordNet 客观断言·非系统教师判断）。

    **#479 + 非 verify_inverse**：antonym 对来自外部源·系统只落边不验真（语言反义 concept↔concept·代数 verify_inverse
    只验 transform·对词对返 None can't-verify·非偷渡 truth）。

    返 list[(a_surface, b_surface)]（caller `antonym.bootstrap_antonym_edges` 建 EDGE_ANTONYM 边·单边 a→b·reader 双向查）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：文件真伪/质量 = 外部数据责任（接地墙）·系统不判·只落边。
    """
    pairs: list[tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动 strip BOM
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) < 2:
                    continue   # 格式错行（<2 段）skip（E5 graceful·不抛崩）
                a, b = parts[0], parts[-1]
                if a == b:
                    continue   # 自环（词非自身反义）早跳省 ensure
                pairs.append((a, b))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


def resolve_antonym_facts(lang: int, local_dir: str | None = None) -> list[tuple[str, str]]:
    """解析某 lang 的 ANTONYM facts（E10 本地文件·ZERO_AI_LOCAL_DIR/antonym_facts_{lang}.txt·文件不存在→空·镜像 resolve_mereology_facts）。

    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 / lang 无映射 → 返 []
    （E5 graceful·bit-identical 守·CI/生产 default 无文件→空→bootstrap 零副作用→零 ANTONYM 边）。

    返 list[(a_surface, b_surface)]（caller 调 `antonym.bootstrap_antonym_edges` 建 EDGE_ANTONYM 边）。

    铁律：纯整数（lang int·返 surface str 对）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return []   # 无映射的 lang（LANG_NONE 等）→ 空
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, f"antonym_facts_{suffix}.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_antonym_facts_file(path)


def load_similar_facts_file(path: str) -> list[tuple[str, str]]:
    """读 SIMILAR facts 文件（E10 纯本地读·每行 "a b"·ChineseSemanticKB 同义关系库 / ConceptNet /r/Synonym 近义对导出格式·无序对称对）。

    镜像 load_antonym_facts_file（同格式·同清洗）：每行空白切·首段=a·末段=b（中段忽略容错）·无序对称对（a~b·canonical 序）。
    `#` 注释行 skip·空行 skip·格式错行（<2 段）skip + **不抛崩**（E5 graceful）·自环（a==b）skip。

    **非 core 数据**（守「不写死」·同 load_antonym_facts_file）：core 永不 import·loader pluggable·生产 default 无文件→resolve 返空→boot 零副作用。
    SIMILAR 机制全在（#898·build_similar_edges observe + dispatch_slot reader）·本 loader 补 boot-side 数据 ingest（镜像 antonym）。

    返 list[(a_surface, b_surface)]（caller `similar.bootstrap_similar_edges` 建 EDGE_SIMILAR 边·单边 a→b·reader 双向查）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：文件真伪/质量 = 外部数据责任（接地墙·#479）·系统不判·只落边（近义非证明）。
    """
    pairs: list[tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动 strip BOM
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) < 2:
                    continue   # 格式错行（<2 段）skip（E5 graceful·不抛崩）
                a, b = parts[0], parts[-1]
                if a == b:
                    continue   # 自环（词非自身近义）早跳省 ensure
                pairs.append((a, b))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


def resolve_similar_facts(lang: int, local_dir: str | None = None) -> list[tuple[str, str]]:
    """解析某 lang 的 SIMILAR facts（E10 本地文件·ZERO_AI_LOCAL_DIR/similar_facts_{lang}.txt·文件不存在→空·镜像 resolve_antonym_facts）。

    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 / lang 无映射 → 返 []
    （E5 graceful·bit-identical 守·CI/生产 default 无文件→空→bootstrap 零副作用→零 SIMILAR 边）。

    返 list[(a_surface, b_surface)]（caller 调 `similar.bootstrap_similar_edges` 建 EDGE_SIMILAR 边）。

    铁律：纯整数（lang int·返 surface str 对）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return []   # 无映射的 lang（LANG_NONE 等）→ 空
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, f"similar_facts_{suffix}.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_similar_facts_file(path)


def resolve_abstract_facts(lang: int, local_dir: str | None = None) -> list[tuple[str, str]]:
    """解析某 lang 的 ABSTRACT→IS_A 泛化 facts（E10 本地文件·ZERO_AI_LOCAL_DIR/abstract_facts_{lang}.txt·文件不存在→空）。

    **抽象 = IS_A 泛化**（specific→general = child→parent·#1133 纠偏：抽象=IS_A 非新 edge 类型·abstraction.py
    LCA/祖先图 + EDGE_IS_A + bootstrap_is_a_edges 全在·从始至今核心·撤回初版 EDGE_ABSTRACT）。复用 load_is_a_facts_file
    （同 "child parent" 格式·E5 graceful·abstract_facts 与 is_a_facts 同 schema）。

    **异 source provenance**（关键·避 MED-1 式污染）：caller 传 SOURCE_CHINESE_KB（ChineseSemanticKB 抽象关系库·#1133）·
    与 ConceptNet is_a_facts（SOURCE_CONCEPTNET·刀0 boot）分离——build_isa_ancestor_map_external 刀C 验证 filter
    source=SOURCE_CONCEPTNET·abstract 须另源 stamp CHINESE_KB 不污染该验证图。

    优先序：local_dir > ZERO_AI_LOCAL_DIR env > None。缺文件/lang 无映射 → []（E5·bit-identical·CI 无文件→空→零边）。
    返 list[(child_surface, parent_surface)]（caller `is_a.bootstrap_is_a_edges` 建 EDGE_IS_A 边·source=SOURCE_CHINESE_KB）。

    铁律：纯整数 lang/确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    诚实边界：泛化真伪 = 外部数据责任（接地墙·#479）·IS_A 非证明（proper subset 结构断言非 truth 验）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return []
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []
    path = os.path.join(d, f"abstract_facts_{suffix}.txt")
    if not os.path.isfile(path):
        return []
    return load_is_a_facts_file(path)   # 复用 is_a loader（同格式·abstract=IS_A 泛化·#1133 纠偏）


# ---- sense facts 本地文件 loader（刀6 件7·来源① 结构化·E10 纯本地读·boot 时种 sense_candidates 候选） ----

def load_sense_facts_file(path: str) -> list[tuple[str, list[str]]]:
    """读 sense facts 文件（E10 纯本地读·每行 "word sense1 sense2 ..."·多义 sense 候选词典导出格式）。

    每行空白切·首段=word surface（多义 token）·其余段=sense surfaces（不同 sense 不同 ConceptRef·解 N10）。
    `#` 注释行 skip·空行 skip·格式错行（<2 段·只有 word 无 sense）skip + **不抛崩**（E5 graceful·错行不破训练）。
    自环 sense（sense==word）skip（同 surface ensure 同 ref·非 sense 候选·早跳省 ensure 副作用）。

    **非 core 数据**（合规守「不写死」·同 load_is_a_facts_file）：core 永不 import 此文件·core 不知 pair 内容·
    loader 是 pluggable 数据摄入（§7.3 接口归系统）·生产 default 无文件→resolve 返空→boot 零副作用。
    sense facts 是外部多义词典（结构化源①·§8.1c 合规·类比 ConceptNet IsA）·非系统教师判断·非语义规则。

    返 list[(word_surface, [sense1_surface, sense2_surface, ...])]（caller `sense_candidates.bootstrap_sense_candidates`
    种 base_count·observe record_sense_token_seen 写 sc_tn·理解侧 recognize clone 选 sense）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：sense 真伪/覆盖度 = 外部数据责任（接地墙·#479 教师定义权）·系统不判·只落候选台账。
    """
    pairs: list[tuple[str, list[str]]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动 strip BOM
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) < 2:
                    continue   # 格式错行（只有 word 无 sense）skip（E5 graceful·不抛崩）
                word = parts[0]
                senses = [s for s in parts[1:] if s != word]   # 自环 sense 早跳（同 surface 同 ref·非 sense）
                if not senses:
                    continue   # 全自环 → 无 sense 候选 skip
                pairs.append((word, senses))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


def resolve_sense_facts(lang: int, local_dir: str | None = None) -> list[tuple[str, list[str]]]:
    """解析某 lang 的 sense facts（E10 本地文件·ZERO_AI_LOCAL_DIR·文件不存在→空·镜像 resolve_is_a_facts）。

    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 / lang 无映射 → 返 []
    （E5 graceful·bit-identical 守·CI/生产 default 无文件→空→bootstrap 零副作用→sense_candidates 表空）。

    返 list[(word_surface, [sense surfaces])]（caller 调 `sense_candidates.bootstrap_sense_candidates` 种 base_count）。

    铁律：纯整数（lang int·返 surface str 对）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return []   # 无映射的 lang（LANG_NONE 等）→ 空
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, f"sense_facts_{suffix}.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_sense_facts_file(path)


# ---- CAUSES facts 本地文件 loader（入手④·来源① ConceptNet·E10 纯本地读·boot 时种 EDGE_CAUSES 边） ----

def load_causes_facts_file(path: str) -> list[tuple[str, str]]:
    """读 CAUSES facts 文件（E10 纯本地读·每行 "cause effect"·ConceptNet Causes 有向三元组导出格式）。

    每行空白切·首段=cause surface·末段=effect surface（中段忽略·容错·支持 "雨 导致 地湿" 带指向词）。
    `#` 注释行 skip·空行 skip·格式错行（<2 段）skip + **不抛崩**（E5 graceful·数据可再下·错行不破训练）。
    自环（cause==effect）skip（_insert_causes:57 亦跳 a==b·此处早跳省 ensure 副作用）。

    **非 core 数据**（合规守「不写死」·同 load_is_a_facts_file）：core 永不 import 此文件·core 不知 pair 内容·
    loader 是 pluggable 数据摄入（同 LocalDirSource 范式·§7.3 接口归系统）·生产 default 无文件→resolve 返空
    →boot 零副作用。CAUSES facts 文件是外部因果断言数据（ConceptNet Causes 客观有向三元组·非系统教师判断·
    经 bootstrap_causes_edges 落 PRIMARY·EPI_STRUCTURED·来源①·§8.1c 合规）。

    返 list[(cause_surface, effect_surface)]（caller `causes.bootstrap_causes_edges` 建 EDGE_CAUSES 边）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：因果真伪/方向 = 外部数据责任（接地墙·ConceptNet 可错·stable≠correct·#479 墙）·系统不判·只落边。
    """
    pairs: list[tuple[str, str]] = []
    try:
        # utf-8-sig 自动 strip BOM（防 BOM 前置致首行注释/数据识别失败·同 load_is_a_facts_file）
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) < 2:
                    continue   # 格式错行（<2 段）skip（E5 graceful·不抛崩）
                cause, effect = parts[0], parts[-1]
                if cause == effect:
                    continue   # 自环无义·早跳省 ensure
                pairs.append((cause, effect))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


def resolve_causes_facts(lang: int, local_dir: str | None = None) -> list[tuple[str, str]]:
    """解析某 lang 的 CAUSES facts（E10 本地文件·ZERO_AI_LOCAL_DIR/causes_facts_{lang}.txt·镜像 resolve_is_a_facts）。

    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 / lang 无映射 → 返 []
    （E5 graceful·bit-identical 守·CI/生产 default 无文件→空→bootstrap 零副作用→零 CAUSES 外部边）。

    返 list[(cause_surface, effect_surface)]（caller 调 `causes.bootstrap_causes_edges` 建 EDGE_CAUSES 边）。

    铁律：纯整数（lang int·返 surface str 对）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return []   # 无映射的 lang（LANG_NONE 等）→ 空
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, f"causes_facts_{suffix}.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_causes_facts_file(path)


# ---- STRUCT_BIND 跨模态绑定对 loader（#478·来源 a 教师标注·E10 纯本地读·boot 时种 EDGE_STRUCT_BIND 边） ----
#
# 设计决断（doc/重来_任务0478_STRUCT_BIND_设计.md 决断 2 + 决断 4）：
#   - 通用跨模态槽位级绑定·非"算术↔语言专用"（用户 scope 收窄指令 1）。
#   - 来源 a 教师标注先行（minimal viable·墙内可做）·来源 b 跨模态结构对齐 defer（撞指称锚墙·#479 子问题）。
#   - **非 lang-keyed 单文件**（与 is_a_facts/sense_facts 异）：STRUCT_BIND 是跨模态绑定·绑定对横跨模态不属某 lang·
#     单一 struct_bind_pairs.txt（caller resolve 不带 lang）。
#   - slot_map 按位序（a:b·文本可解析·同 IS_A 容错范式）·非按 role 名（增耦合·决断 2 不推荐）。
#   - **name 是 skeleton 注册名**（DiscoveredOperator.name = `__op_disc_{h63}`·hash·非人类可读）·教师 corpus
#     须由 discover 录制产出（#731 真实语料准备链路）·非手写。决断 4"name 映射机制待实施期 loader 设计定"=>
#     loader 透传 name 字符串·name→skeleton_ref 解析在 formal_train boot caller（经 discovered_operators 索引）。


def load_struct_bind_pairs_file(path: str) -> list[tuple[str, str, list[tuple[int, int]]]]:
    """读 STRUCT_BIND 绑定对文件（E10 纯本地读·每行 "name_a name_b a:b c:d ..."·教师标注 discover 录制格式）。

    每行空白切·首段=模态A skeleton 注册名（DiscoveredOperator.name）·次段=模态B skeleton 注册名·
    其余段=slot_map 位序对（"a:b" = A 槽 a ↔ B 槽 b·按位序·决断 2）。`#` 注释行 skip·空行 skip·
    格式错行（<3 段·无 name_a/name_b/至少一对 slot_map）skip + **不抛崩**（E5 graceful·错行不破训练）。
    slot 对解析：以 ":" partition·两端 int≥0·解析失败 skip 该对（容错·同 IS_A 范式）·全对失败 skip 该行。

    **非 core 数据**（合规守「不写死」·同 load_is_a_facts_file/load_sense_facts_file）：core 永不 import 此
    文件·core 不知 pair 内容·loader 是 pluggable 数据摄入（§7.3 接口归系统）·生产 default 无文件→resolve 返空
    →boot 零副作用。STRUCT_BIND pairs 是外部教师标注（来源 a 结构化源①·§8.1c 合规·类比 ConceptNet IsA）·
    非系统教师判断·非语义规则·绑定真伪/对齐质量=外部数据责任（接地墙·#479·决断 2 来源 b defer）。

    返 list[(name_a, name_b, [(a_idx, b_idx), ...])]（caller `struct_bind.bootstrap_struct_bind_edges`
    经 collect_skeleton_slot_refs 解析 slot ref·name→skeleton_ref 经 discovered_operators 索引·决断 4）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：slot_map 位序是结构对齐非语义绑定（跨模态指称锚=接地墙·#479·来源 b defer）/ name 真伪/对齐=外部责任。
    """
    pairs: list[tuple[str, str, list[tuple[int, int]]]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动 strip BOM
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) < 3:
                    continue   # 格式错行（<3 段·须 name_a name_b + ≥1 slot 对）skip（E5 graceful·不抛崩）
                name_a, name_b = parts[0], parts[1]
                slot_map: list[tuple[int, int]] = []
                for tok in parts[2:]:
                    a, sep, b = tok.partition(":")
                    if not sep:
                        continue   # 非 "a:b" 形式 skip 该对（容错）
                    try:
                        ai, bi = int(a), int(b)
                    except ValueError:
                        continue   # 非整数 skip 该对
                    if ai < 0 or bi < 0:
                        continue   # 负 idx 无义 skip
                    slot_map.append((ai, bi))
                if not slot_map:
                    continue   # 全 slot 对解析失败 → 无有效绑定 skip
                pairs.append((name_a, name_b, slot_map))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


def resolve_struct_bind_pairs(local_dir: str | None = None
                              ) -> list[tuple[str, str, list[tuple[int, int]]]]:
    """解析 STRUCT_BIND 绑定对（E10 本地文件·ZERO_AI_LOCAL_DIR/struct_bind_pairs.txt·文件不存在→空·
    镜像 resolve_is_a_facts/resolve_sense_facts 范式·#478）。

    **非 lang-keyed**（与 resolve_is_a_facts/resolve_sense_facts 异）：STRUCT_BIND 跨模态绑定横跨模态不属某
    lang·单一 struct_bind_pairs.txt（无 lang 后缀·caller 不带 lang）。优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR
    env > None。目录/文件不存在 → 返 []（E5 graceful·bit-identical 守·CI/生产 default 无文件→空→boot 零副作用）。

    返 list[(name_a, name_b, [(a_idx, b_idx), ...])]（caller formal_train boot 经 discovered_operators 索引解析
    name→skeleton_ref + collect_skeleton_slot_refs 解析 slot ref → bootstrap_struct_bind_edges 建边·决断 4）。

    铁律：纯整数（slot idx int·返 name str 三元）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, "struct_bind_pairs.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_struct_bind_pairs_file(path)


# ---- alias facts 本地文件 loader（P0b·来源① 跨语言 PURE_ALIAS 桥·E10 纯本地读·boot 时种双向 PURE_ALIAS 边） ----

# alias_facts.txt 行内 lang 码 → LANG_* 常量（types.py:35-36·与 _LANG_FILE_SUFFIX 反向·显式枚举已知 lang）。
# 未知 lang 码 → None → 该行 skip（E5 graceful·守不写死：lang 码是外部数据非 core enum）。
_ALIAS_LANG_CODE: dict[str, int] = {
    "en": LANG_EN,
    "zh": LANG_ZH,
}


def load_alias_facts_file(path: str) -> list[tuple[str, int, str, int]]:
    """读 alias facts 文件（E10 纯本地读·每行 "surface_a lang_a surface_b lang_b"·跨语言/同义 PURE_ALIAS 对）。

    每行空白切·须 4 段：(surface_a, lang_a, surface_b, lang_b)·lang 码经 _ALIAS_LANG_CODE 映射 LANG_*。
    `#` 注释行 skip·空行 skip·格式错行（≠4 段）/ 未知 lang 码 / 自环（surface_a==surface_b 且 lang 相同）
    skip + **不抛崩**（E5 graceful·数据可再下·错行不破训练·镜像 load_is_a_facts_file）。

    **非 core 数据**（合规守「不写死」·同 load_is_a_facts_file / load_sense_facts_file）：core 永不 import 此文件·
    loader 是 pluggable 数据摄入（§7.3 接口归系统）·生产 default 无文件→resolve 返空→boot 零副作用。
    alias facts 是外部跨语言词典 / Wikidata QID 翻译等价（来源① 结构化·§8.1c 合规·EPI_STRUCTURED·
    非系统教师判断·非语义规则·类比 ConceptNet IsA）。

    返 list[(surface_a, lang_a, surface_b, lang_b)]（caller `alias_bridge.bootstrap_alias_edges` 建 NODE_WORD
    + MARK_LANG + 双向 REFERS_TO PURE_ALIAS 边·§3.1/§7.4 "苹果/apple 同节点"=等价类非同 local_id）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    诚实边界：alias 真伪（翻译对错）= 外部数据责任（接地墙·#479 教师定义权）·系统不判·只落 PURE_ALIAS 边。
    """
    pairs: list[tuple[str, int, str, int]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动 strip BOM（防首行识别失败·对抗审点）
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) != 4:
                    continue   # 格式错行（≠4 段）skip（E5 graceful·不抛崩）
                surf_a, lang_a_s, surf_b, lang_b_s = parts
                lang_a = _ALIAS_LANG_CODE.get(lang_a_s)
                lang_b = _ALIAS_LANG_CODE.get(lang_b_s)
                if lang_a is None or lang_b is None:
                    continue   # 未知 lang 码 skip（守不写死·lang 码外部数据非 core enum）
                if surf_a == surf_b and lang_a == lang_b:
                    continue   # 完全自环（同 surface 同 lang）无义·早跳省 ensure（build_refers_stable_edge:59 亦跳 a==b ref）
                pairs.append((surf_a, lang_a, surf_b, lang_b))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return pairs


def resolve_alias_facts(local_dir: str | None = None
                        ) -> list[tuple[str, int, str, int]]:
    """解析跨语言 alias facts（E10 本地文件·ZERO_AI_LOCAL_DIR/alias_facts.txt·文件不存在→空·
    镜像 resolve_struct_bind_pairs 范式·P0b）。

    **非 lang-keyed**（与 resolve_is_a_facts/resolve_sense_facts 异·同 resolve_struct_bind_pairs）：aliases 横跨
    语言不属某 lang·单一 alias_facts.txt（无 lang 后缀·caller 不带 lang）。优先序：local_dir 参数 >
    ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 → 返 []（E5 graceful·bit-identical 守·CI/生产 default
    无文件→空→bootstrap 零副作用→核心空间零 PURE_ALIAS 边→activate_candidates 退化现状）。

    返 list[(surface_a, lang_a, surface_b, lang_b)]（caller 调 `alias_bridge.bootstrap_alias_edges` 建双向边）。

    铁律：纯整数（lang int·返 surface str + lang 四元）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, "alias_facts.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_alias_facts_file(path)


# ---- number facts 本地文件 loader（language-grounding piece 1·语言→算数 桥数据源·E10 纯本地读·boot 种图边） ----
# doc/重来_语言通用接地_2026-07-16 §七。数字词→整数接地·**数据驱动 + 关联在图中**（用户铁律·非代码字典）：
# loader 读外部 number_facts.txt → bootstrap_number_grounding 建 整数概念(__int_{value}) + CORR_NUMERIC 值
# + 词 NODE_WORD —PURE_ALIAS 边→ 整数概念（图边 = 关联·镜像 apple↔苹果·可遍历）。core 永不 import·
# 生产 default 无文件 → resolve 返 [] → bootstrap 零副作用 → bit-identical（同 alias_facts 范式）。
_NUMBER_LANG_CODE: dict[str, int] = {"en": LANG_EN, "zh": LANG_ZH}   # 同 _ALIAS_LANG_CODE


def load_number_facts_file(path: str) -> list[tuple[str, int, int]]:
    """读 number facts 文件（E10 纯本地读·每行 "surface lang value"·数字词→整数接地）。

    每行空白切·须 3 段：(surface, lang, value)·lang 码经 _NUMBER_LANG_CODE 映射·value 须纯整数。
    `#` 注释行 skip·空行 skip·格式错行（≠3 段）/ 未知 lang 码 / value 非整 skip + 不抛崩（E5 graceful·
    镜像 load_alias_facts_file）。

    **非 core 数据**（守「不写死」·用户铁律·关联在图中非代码）：core 永不 import·loader pluggable·
    生产 default 无文件 → resolve 返 [] → bootstrap 零副作用。真值责任 = 外部数据（接地墙·#479）·系统不判。

    返 list[(surface, lang, value)]（caller `number_grounding.bootstrap_number_grounding` 建整数概念
    + CORR_NUMERIC + 词↔整数概念 PURE_ALIAS 图边）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    """
    facts: list[tuple[str, int, int]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig strip BOM（同 load_alias_facts_file）
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) != 3:
                    continue   # 格式错行 skip
                surf, lang_s, value_s = parts
                lang = _NUMBER_LANG_CODE.get(lang_s)
                if lang is None:
                    continue   # 未知 lang 码 skip
                try:
                    value = int(value_s)
                except ValueError:
                    continue   # value 非整 skip（守纯整数）
                facts.append((surf, lang, value))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩）
    return facts


def resolve_number_facts(local_dir: str | None = None
                         ) -> list[tuple[str, int, int]]:
    """解析数字词接地 facts（E10 本地文件·ZERO_AI_LOCAL_DIR/number_facts.txt·文件不存在→空·镜像 resolve_alias_facts）。

    **非 lang-keyed**（同 alias_facts·数字词横跨语言不属某 lang）：单一 number_facts.txt（无 lang 后缀）。
    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 → 返 []（E5 graceful·bit-identical 守·
    CI/生产 default 无文件 → 空 → bootstrap 零副作用 → 核心空间零整数概念/PURE_ALIAS 数字边 → bit-identical）。

    返 list[(surface, lang, value)]（caller 调 `number_grounding.bootstrap_number_grounding` 建图边 + 值）。

    铁律：纯整数（lang/value int·返 surface str）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, "number_facts.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_number_facts_file(path)


# ---- degree cues 本地文件 loader（#1134 程度→属性器 augment·degree 副词→Rational intensity·E10 纯本地读·boot 喂 cue_words cache） ----
# doc/重来_程度属性器intensity_2026-07-16.md + 权威 doc/重来_ChineseSemanticKB能力映射 §4.3。degree 副词 + intensity
# （很/非常/极其=2/1·较=3/2·稍=2/5·Rational·非 float·用户纠偏）·**非 core 数据**（守不写死·程度 cue+intensity 全来自
# 外部 degree_cues_{lang}.txt·core 不 import·loader pluggable·生产 default 无文件→resolve 返 {}→populate_degree_cues
# 空映射→is_degree_cue 恒 False→intensity 恒 1/1→bit-identical）。


def load_degree_cues_file(path: str) -> dict[str, tuple[int, int]]:
    """读 degree cues 文件（E10 纯本地读·每行 "cue<TAB>num/den"·degree 副词→Rational intensity）。

    每行空白切·须 2 段：(cue, num/den)。num/den 经 "/" split→两正整（den>0·num>0）。
    `#` 注释行 skip·空行 skip·格式错行（≠2 段）/ 无 "/" / num/den 非正整 / den≤0 / num≤0 skip + 不抛崩（E5 graceful·
    镜像 load_alias_facts_file）。**纯整数**（num/den 皆 int·Rational·非 float·用户纠偏「没说有理不行」）。

    返 dict[cue] = (num, den)（caller formal_train boot → cue_words.populate_degree_cues 喂 module cache）。

    铁律：纯本地读确定性 bit-identical / E5 graceful（错行/读错返空段不抛崩）/ 不写死（外部文件·core 不 import）。
    """
    mapping: dict[str, tuple[int, int]] = {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig strip BOM（同 load_alias_facts_file）
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split()
                if len(parts) != 2:
                    continue   # 格式错行 skip（cue + num/den 两段）
                cue, frac = parts
                if "/" not in frac:
                    continue   # 非 num/den 格式 skip
                num_s, _, den_s = frac.partition("/")
                try:
                    num = int(num_s)
                    den = int(den_s)
                except ValueError:
                    continue   # num/den 非整 skip（守纯整数）
                if den <= 0 or num <= 0:
                    continue   # intensity 正缩放（den>0·num>0·sign 经 >1/<1·非负非零）skip
                mapping[cue] = (num, den)
    except (OSError, UnicodeDecodeError):
        return {}   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩）
    return mapping


def resolve_degree_facts(lang: int, local_dir: str | None = None
                         ) -> dict[str, tuple[int, int]]:
    """解析 degree cues（E10 本地文件·ZERO_AI_LOCAL_DIR/degree_cues_{suffix}.txt·文件不存在→{}·镜像 resolve_*_facts lang-keyed·#1134）。

    lang-keyed（degree 副词语言特定·degree_cues_zh.txt）：suffix 经 _LANG_FILE_SUFFIX 映射（ZH→zh·EN→en·未知 lang→{}）。
    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 → 返 {}（E5 graceful·bit-identical 守·
    CI/生产 default 无文件 → {} → populate_degree_cues 空映射 → is_degree_cue 恒 False → intensity 恒 1/1 → bit-identical）。

    返 dict[cue] = (num, den)（caller formal_train boot → cue_words.populate_degree_cues(lang, mapping) 喂 cache）。

    铁律：纯整数（num/den int·lang int）/ 确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空 dict 非崩）。
    """
    suffix = _LANG_FILE_SUFFIX.get(lang)
    if suffix is None:
        return {}   # 未知 lang → 空（degree cue ZH 首版·EN defer）
    d = _resolve_local_dir(local_dir)
    if d is None:
        return {}   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, f"degree_cues_{suffix}.txt")
    if not os.path.isfile(path):
        return {}   # 文件不存在 → 空
    return load_degree_cues_file(path)


# ---- 符号数学语料本地文件 loader（Phase 0·S5-S8 训练课程数据摄入·E10 纯本地读·boot-inject 入 corpus） ----
#
# 设计决断（doc/重来_阶段断奶路线详设_2026-07-15 §二 Phase 0.1）：transform_rules.txt（S5-S7 教师陈述符号变换规则）
# + inverse_relations.txt（S8 逆关系）·单 cross-lang 风格文件（非 lang-keyed·同 alias_facts/struct_bind_pairs）·
# loader pluggable·生产 default 无文件→resolve 返 []→formal_train boot-inject 不 append CollectedItem→
# 核心空间零 transform_specs/inverse_relation_specs→bit-identical（同 alias_facts 范式）。
# **非 core 数据**（守不写死·同 alias_facts/sense_facts）：规则是教师陈述数据（TransformSpec/InverseRelationSpec·
# 外部文件·非硬编码 formula·code 是通用 register/apply/verify 机制·humans 学法）·core 永不 import。


def load_transform_rules_file(path: str) -> list[TransformSpec]:
    """读符号变换规则文件（E10·每行 TAB 分隔 4 字段·rule_name lhs rhs held_out·doc/重来_阶段断奶路线详设 §二）。

    每行：rule_name <TAB> lhs_source <TAB> rhs_source <TAB> held_out
      - lhs/rhs_source : lambda DSL 字符串（PARAM 按位置对齐·dsl 详 arith_observe·如 "lambda b,n: Pow(b,n)"）。
      - held_out       : 分号 ; 分隔的「input_source=>expected_source」对（空字段=无 held-out→()→不产 episode 反 theater）。
    `#` 注释行 skip·空行 skip·字段数≠4 / 字段空 / held-out 对缺 => / parse 异常 → **skip + 不抛崩**
    （E5 graceful·数据可再下·错行不破训练·镜像 load_alias_facts_file）。

    返 list[TransformSpec]（caller formal_train boot-inject 包 CollectedItem 挂 transform_specs·gate SYMBOLIC_TRANSFORM_MODE 消费）。
    **bit-identical**：CI/生产 default 无 ZERO_AI_LOCAL_DIR→resolve 返 []→不 inject→零 specs→bit-identical。

    铁律：纯本地读确定性 / E5 graceful（错行返空段不抛崩）/ 不写死（外部文件·规则数据驱动·core 不 import）。
    诚实边界：DSL 合法性由 build_composes_from_arith 在 formal_train 消费期 fail-loud（loader 只 parse 字符串·
    不验 DSL·malformed DSL→消费期 try/except skip 单 spec·不 abort run·既有守）。
    """
    rules: list[TransformSpec] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动 strip BOM
            for line in f:
                stripped = line.rstrip("\n").rstrip("\r")
                if not stripped.strip() or stripped.lstrip().startswith("#"):
                    continue   # 注释 / 空行 skip
                parts = stripped.split("\t")
                if len(parts) != 4:
                    continue   # 格式错行（≠4 TAB 字段）skip（E5 graceful·不抛崩）
                rule_name, lhs, rhs, held_field = (p.strip() for p in parts)
                if not rule_name or not lhs or not rhs:
                    continue   # 关键字段空 skip
                held_out: list[TransformHeldOut] = []
                if held_field:
                    for pair in held_field.split(";"):
                        pair = pair.strip()
                        if not pair or "=>" not in pair:
                            continue   # held-out 对缺 => 分隔 skip
                        in_src, exp_src = pair.split("=>", 1)
                        in_src, exp_src = in_src.strip(), exp_src.strip()
                        if not in_src or not exp_src:
                            continue
                        held_out.append(TransformHeldOut(in_src, exp_src))
                rules.append(TransformSpec(rule_name, lhs, rhs, tuple(held_out)))
    except (OSError, UnicodeDecodeError):
        return []   # 文件读错/编码错 → 返空段（E5 graceful·不抛崩训练）
    return rules


def resolve_transform_rules(local_dir: str | None = None
                            ) -> list[TransformSpec]:
    """解析符号变换规则（E10 本地文件·ZERO_AI_LOCAL_DIR/transform_rules.txt·文件不存在→空·
    镜像 resolve_alias_facts 范式·Phase 0.1）。

    **非 lang-keyed**（同 resolve_alias_facts/resolve_struct_bind_pairs）：符号数学规则横跨域不属某 lang·
    单一 transform_rules.txt（无 lang 后缀）。优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。
    目录/文件不存在 → 返 []（E5 graceful·bit-identical 守·CI/生产 default 无文件→空→boot-inject 不进）。

    返 list[TransformSpec]（caller formal_train boot-inject 包 CollectedItem 挂 transform_specs）。

    铁律：确定性（env 读一次·路径确定）/ E5 graceful（缺文件返空非崩）。
    """
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []   # 无 local_dir → 空（生产 default bit-identical 守）
    path = os.path.join(d, "transform_rules.txt")
    if not os.path.isfile(path):
        return []   # 文件不存在 → 空
    return load_transform_rules_file(path)


def load_inverse_relations_file(path: str) -> list[InverseRelationSpec]:
    """读逆关系文件（E10·每行 TAB 分隔 8 字段·doc/重来_阶段断奶路线详设 §二 + doc/重来_S8符号间关联机制设计）。

    每行：relation_name <TAB> a_name <TAB> a_lhs <TAB> a_rhs <TAB> b_name <TAB> b_lhs <TAB> b_rhs <TAB> samples
      - a_*/b_*    : 两条独立变换规则 A 与 B（各 rule_name + lhs/rhs lambda DSL·同 TransformSpec 范式）。
      - samples    : 分号 ; 分隔的采样输入 lambda（B∘A 须还原这些 @ 探针·arity≤6·小素数采样）。
    `#` 注释行 skip·空行 skip·字段数≠8 / 关键字段空 / parse 异常 → skip + 不抛崩（E5 graceful·镜像 load_alias_facts_file）。

    返 list[InverseRelationSpec]（caller formal_train boot-inject 包 CollectedItem 挂 inverse_relation_specs·
    gate SYMBOLIC_RELATION_MODE 消费）。a/b 的 held_out=()（逆验证不依赖单规则 held-out·B∘A 构造验证独立）。
    **bit-identical**：CI/生产 default 无文件→resolve 返 []→不 inject→bit-identical。

    铁律：纯本地读确定性 / E5 graceful / 不写死。诚实边界：DSL 合法性消费期 fail-loud 守（同 transform_rules）。
    """
    rels: list[InverseRelationSpec] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                stripped = line.rstrip("\n").rstrip("\r")
                if not stripped.strip() or stripped.lstrip().startswith("#"):
                    continue
                parts = stripped.split("\t")
                if len(parts) != 8:
                    continue
                rel_name, a_name, a_lhs, a_rhs, b_name, b_lhs, b_rhs, samp_field = \
                    (p.strip() for p in parts)
                if not rel_name or not a_name or not b_name:
                    continue
                samples = tuple(s.strip() for s in samp_field.split(";") if s.strip()) \
                    if samp_field else ()
                rule_a = TransformSpec(a_name, a_lhs, a_rhs, ())
                rule_b = TransformSpec(b_name, b_lhs, b_rhs, ())
                rels.append(InverseRelationSpec(rel_name, rule_a, rule_b, samples))
    except (OSError, UnicodeDecodeError):
        return []
    return rels


def resolve_inverse_relations(local_dir: str | None = None
                             ) -> list[InverseRelationSpec]:
    """解析逆关系（E10 本地文件·ZERO_AI_LOCAL_DIR/inverse_relations.txt·文件不存在→空·
    镜像 resolve_transform_rules 范式·Phase 0.1）。

    优先序：local_dir 参数 > ZERO_AI_LOCAL_DIR env > None。目录/文件不存在 → 返 []（E5 graceful·bit-identical 守）。
    返 list[InverseRelationSpec]（caller formal_train boot-inject 包 CollectedItem 挂 inverse_relation_specs）。

    铁律：确定性 / E5 graceful（缺文件返空非崩）。
    """
    d = _resolve_local_dir(local_dir)
    if d is None:
        return []
    path = os.path.join(d, "inverse_relations.txt")
    if not os.path.isfile(path):
        return []
    return load_inverse_relations_file(path)


# ---- 语料相关 KB 过滤（perf fix·KB vocab-edge 只留 ≥1 surface 在语料 vocab 的 pair） ----
#
# 设计决断（doc/重来_语料相关KB过滤_2026-07-16）：全量 KB（614k pair·similar 380k+alias 234k）对 656-paragraph
# 语料 84-99.5% out-of-corpus ballast（语料永不提及→observe/generate 不触→零学习信号）→ boot 95s + 训练图 660k
# 边 7x 慢 + causes_coverage ②结构 permille 分母稀释（capability_exam:545 文档已知·FAIL≠结构破裂）。
# 修：boot 时 KB vocab-edge（alias/similar/antonym/mereology/is_a/abstract）resolve 后过滤·只留 ≥1 surface 在
# 语料 vocab 的 pair。causes/sense 保留全量（causes 接 reward 反传·train 骨干·15k 小·非 ballast 主因；sense 小）。
# **随语料 scale**（大语料→vocab 大→多包）·非 hack·非截断·是「数据完备 for 语料」的**可辩护操作读法**（out-of-corpus
# pair 对本语料训练无值·非系统可学对象——语料不提的词无法从语料学·对抗审#2 H1：非「正确语义」独断·待用户确认·
# .full 备份可恢复全量·全量收集与 per-run 语料相关装载互补）。语料长大后重评（大语料→多留·自然 scale）。
# **bit-identical**：CI 无 ZERO_AI_LOCAL_DIR→resolve_*_facts 返 []→filter 空 list 返空（filter_pairs_to_vocab
# `not pairs` 短路）·零行为变。生产有文件→deterministic frozenset 过滤（vocab 由语料 token 确定性建）。
# **不写死**：vocab 来自语料（外部数据）·非 core enum·过滤是数据 scoping 非语义判定（守铁律）。


def corpus_relevant_vocab(corpus: list) -> frozenset[str]:
    """训练语料的 token vocab（language item 的 tokens·KB 语料相关过滤用）。

    返 frozenset[str]：所有 language 模态 CollectedItem 的 tokens 并集。arith/code 模态 tokens 不入
    （KB facts 是语言域 alias/similar/...·按 language 语料 vocab 过滤）。空语料 → 空 frozenset
    （filter_pairs_to_vocab 空 vocab 返原 pairs·守 bit-identical·但空语料无 training 亦无 boot vocab-edge）。

    铁律：确定性（frozenset·语料 token 确定）/ 纯 str（token 是 surface 文本）。
    """
    return frozenset(
        tok for it in corpus
        if getattr(it, "modality", None) == MODALITY_LANGUAGE
        and getattr(it, "tokens", None)
        for tok in it.tokens
    )


def filter_pairs_to_vocab(pairs: list, vocab: frozenset[str],
                          idx_a: int = 0, idx_b: int = 1) -> list:
    """留 ≥1 surface 在 vocab 的 pair（语料相关 KB 过滤·空 vocab/空 pairs 返原 pairs·守 bit-identical）。

    pairs 元素的 idx_a/idx_b 位是 surface str（alias 四元 (surf_a,lang_a,surf_b,lang_b)→idx 0,2 / 余二元 (a,b)→0,1）。
    ≥1 surface 在 vocab → 留（corpus word 的关系全保留·非 corpus word↔non-corpus 的 inert pair 去）。
    空 vocab（空语料 / CI 无文件 resolve 返 []）→ 返原 pairs（短路·守 bit-identical·无过滤副作用）。

    铁律：确定性（frozenset membership）/ 不写死（vocab 来自语料·过滤是 scoping 非语义）/ bit-identical（空短路）。
    """
    if not vocab or not pairs:
        return pairs   # 空 vocab/空 pairs → 不过滤（守 CI bit-identical + 空语料无意义）
    return [p for p in pairs if p[idx_a] in vocab or p[idx_b] in vocab]


# ---- 五类聚合 + E5 graceful 降级 ----

def collect_corpus(sources: list[CollectionSource]) -> CollectionReport:
    """五类收集聚合（多源合并 + E5 graceful 降级·失败源显式记录不静默吞错）。

    优先序由 sources 列表序表达（caller 排 local_dir > SDK）。每源 available()=False 跳过·
    collect() 异常 → 记 failed_sources 继续非崩。返 CollectionReport（items + 失败源 + 五类计数）。
    """
    report = CollectionReport()
    for src in sources:
        try:
            if not src.available():
                report.failed_sources.append(f"{src.name()}:unavailable")
                continue
            items = src.collect()
            from pure_integer_ai.experiments.corpus_identity import (
                assign_corpus_source_refs,
            )
            assign_corpus_source_refs(items, source_namespace=src.name())
            for it in items:
                report.items.append(it)
                report.counts_per_type[it.collect_type] = \
                    report.counts_per_type.get(it.collect_type, 0) + 1
        except Exception as exc:   # E5 graceful·单源失败不破坏训练（数据可再下）
            report.failed_sources.append(f"{src.name()}:{type(exc).__name__}")
    return report


# ---- #727 算术/代码种子 corpus loader（考核 harness 片4a·验③⑤⑥ fixture 限制） ----
#
# 设计决断（doc/重来_任务0727_corpus跑.md）：
#   - 算术 corpus：12 nullary mul 样本 → discover 2 + recognize 10 held-out + vm_proof 全验 →
#     generalization.rate_permille=1000 → ③ PASS（test_formal_train_wires_generalization 范式）。
#   - 代码 corpus：10 code items（n+n）+ 正确 code_specs → Mode A verify 10 episodes·reward=1 →
#     ⑤ × G5 attribution total=10 active=0（#723 归因表同步出·Mode A verify PRE·非 Mode B POST）。
#   - 坏算术 corpus（反 theater）：2 nullary mul + 全错 spec → held_out=0 → rate=0 → ③ FAIL。
#
# 诚实边界：
#   - Mode A verify（PRE·weaning_phase=WEANING_PRE 默认）·非 Mode B cross-verify（POST·须 weaning gates
#     过 + gate ON·test_mode_b_cross_verify 单测已验 4 态·formal_train POST 跑 defer 独立 session）。
#   - load_corpus(kind) 是种子语料（in-memory·确定性·bit-identical）·非真实语料（#731/#734）。
#   - stable≠correct：corpus 跑验机制活 + 统计 permille·非语义正确（#479 墙）。

CORPUS_KIND_ARITH = "arith"
CORPUS_KIND_ARITH_BAD = "arith_bad"
CORPUS_KIND_CODE = "code"
CORPUS_KIND_LANG = "lang"


def load_arith_corpus() -> list[CollectedItem]:
    """算术域种子 corpus（#727·12 参数化 square 样本·③ PASS + ⑤ Mode A task-driven PASS）。

    每项 lambda {p}: {p} * {p}（参数化·arity=1·变量同一性→square 骨架 PARAM_0 复用）+ 正确 arith_specs
    （input_args=(n,)·expected=n*n）。12 个异参数名（b..n·避保留 a/i·arith_observe._RESERVED）= 12 异 source 同 shape（square）。
    discover 取前 2（MIN_DISCOVER_SAMPLES=2）+ recognize 余 10 held-out + vm_proof 全验 →
    generalization.rate_permille=1000 → ③ PASS（test_stage9_structure_discover:881-885 范式·参数化 square）。
    Mode A verify（PRE）每项 1 episode·reward=1（spec 正确）→ ③ × G5 active=0。
    Mode A task-driven：square arity=1 匹配 spec input_args arity=1 → selected/verified=12 → ⑤ PASS（rate=1000）。
    """
    params = list("bcdefghjklmn")   # 12 distinct param names（避保留 a/i·arith_observe._RESERVED）→ 12 异 source 同 shape
    ns = list(range(5, 17))          # 5..16
    return [
        CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                      source=SOURCE_MATH, arith_source=f"lambda {p}: {p} * {p}",
                      arith_specs=(CodeSpec((n,), (n * n, 1)),))
        for p, n in zip(params, ns)
    ]


def load_arith_bad_corpus() -> list[CollectedItem]:
    """坏算术 corpus（#727 反 theater·2 参数化 square + 全错 spec → ③ FAIL + ⑤ FAIL）。

    2 参数化 square（b*b / c*c·同 shape）→ discover 取前 2 + recognize 余 0 → held_out=0 →
    rate_permille=(0*1000)//max(0,1)=0 → ③ FAIL（< 500）。arith_specs expected=999（全错·真值 25/36）
    → Mode A verify reward=0 → ③ × G5 active=total（全 vetoed）+ Mode A task-driven verified=0 → ⑤ FAIL。
    反 theater：与 load_arith_corpus() 同算子异 spec → ③ status 不同（PASS vs FAIL）·证 ③ 判据 corpus-sensitive。
    """
    return [
        CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                      source=SOURCE_MATH, arith_source="lambda b: b * b",
                      arith_specs=(CodeSpec((5,), (999, 1)),)),
        CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                      source=SOURCE_MATH, arith_source="lambda c: c * c",
                      arith_specs=(CodeSpec((6,), (999, 1)),)),
    ]


# ------ S1 diverse 算术语料（doc/重来_阶段断奶路线详设_2026-07-15 §三 S1）------
# 异参名池（避保留 a/i·arith_observe._RESERVED·同 load_arith_corpus:914 范式）。
_ARITH_PARAM_POOL = "bcdefghjklmnopqrstuvwxyz"   # 24 letters（无 a/i）

# 量级多样性值对（§三 S1 数据规则：小 1-9 / 中 10-99 / 大 100-9999 三档各 10·b≠0 守除法）。
_ARITH_DIV_VALS = (
    [(2, 3), (4, 5), (6, 7), (8, 9), (1, 4), (3, 8), (5, 2), (7, 6), (9, 4), (2, 8)]
    + [(12, 15), (27, 38), (46, 61), (83, 19), (50, 34), (71, 88), (23, 95), (64, 42), (18, 77), (39, 56)]
    + [(123, 456), (789, 211), (1000, 2345), (567, 890), (3456, 123), (7281, 994),
       (415, 6702), (8888, 111), (246, 1357), (9999, 1)]
)

_ARITH_OP_FAMILIES = ("+", "-", "*", "/")


def _distinct_arith_pairs(n: int) -> list[tuple[str, str]]:
    """产 n 个 distinct ordered (p, q) 参数对（p≠q·确定性枚举·bit-identical）。

    **异参名铁律**（反 theater·防 v1 smoke bug）：每 fixture 须用**异参数对**·非同 source 串。
    同 source 串（`lambda x,y: x+y` ×N）→ observe 内容哈希 dedup 成 1 树 → discover<MIN_DISCOVER_SAMPLES=2
    → 返 None → ③=0/verified=0（无泛化信号）。异参对 → N 异树 → discover fires → 真泛化。
    load_arith_corpus 12 square 用异参名(b..n)同此范式。
    """
    pairs: list[tuple[str, str]] = []
    pool = _ARITH_PARAM_POOL
    for i in range(len(pool)):
        for j in range(len(pool)):
            if i == j:
                continue
            pairs.append((pool[i], pool[j]))
            if len(pairs) >= n:
                return pairs
    return pairs


def _arith_div_expected(op: str, a: int, b: int) -> tuple[int, int]:
    """算子 expected (num, den) Rational 元组（vm_proof rational.eq 归一比·raw (a,b) 除法亦匹配）。"""
    if op == "+":
        return (a + b, 1)
    if op == "-":
        return (a - b, 1)
    if op == "*":
        return (a * b, 1)
    if op == "/":
        if b == 0:
            raise ValueError("除法 expected den=0（纯整数铁律·b≠0）")  # 早 fail-loud（审1 LOW-2·池不变式+vm_proof 三层纵深防御外加热守）
        return (a, b)   # raw (num, den)·rational.eq 归一比（6/2≡3/1）
    raise ValueError(f"未知算子族 op={op!r}·须 {_ARITH_OP_FAMILIES}")


def load_arith_family_corpus(op: str, n: int = 30) -> list[CollectedItem]:
    """S1 单算子族 diverse corpus（doc §三 S1·每族 ≥30 I/O 对·量级多样性 + 异参名）。

    **每算子独训**（§三 S1 风险②"单算子过窄——每算子独训防互扰"）：4 族(+−×÷) 各自独跑·非混合。
    合族 ③=1000（4 异骨架 ADD/SUB/MUL/DIV 各匹配本族 held-out·recognize 干净）但 **⑤ task-driven op
    选择干扰**（4 arity-2 op 竞争·cold-start 选首发现 op→verified<total）·单族独跑 ⑤verified=30/30 干净。
    caller 须 per-op 调本函数分跑（§三 防互扰·干扰在 ⑤ 非 ③）。
    （注：早期 smoke 见"合族 ③745"=旧 arith_bad(b,c) 撞 +族 source dedup artifact·已修·真 combined ③=1000。）

    每项 `lambda {p},{q}: {p}{op}{q}`（异参对·discover≥2 异树）+ 正确 arith_specs（input_args=(a,b)·
    expected=_arith_div_expected）。量级覆盖 small/medium/large（解 W8 fixture-trivial 边界·真泛化）。

    **反 theater**：异参名保 discover fires（同源 dedup<2→None·v1 bug 教训）·diverse 量级保非 trivial
    泛化（单 shape 同值=trivial 1000·diverse=真泛化压力）。教师 level 0（vm_proof 自锚·spec.expected R6）。
    **诚实边界**：整数 PbE=行为匹配非符号理解（stable≠correct）/ op_confidence 统计验非 truth。
    **★边界例 DEFER（§三 S1:89·非 silent omit·审2 HIGH）**：§三 S1 数据规则列边界例（恒等 a+0/a×1·零元
    a×0·负元 USub 字面负数·除法精确有理）。**本生成器仅覆盖 operand-binary 量级多样 + 除法精确有理
    （raw (a,b) rational.eq）**·**恒等/零元/负元 USub 三类边界例未实现（defer）**：负元须扩 DSL 支持 USub
    字面负数·零元/恒等须扩 _ARITH_DIV_VALS 含 0/1 + IMM-operand 异 shape（IMM-operand 异 operand_arity_hint
    →独立 discover group·非本 operand-binary shape）。本 defer 致单族 ③=1000（单 shape trivial-but-real 泛化）·
    加边界例 shape 变化方产 <1000 非 trivial。"边界"一词在此＝§三 S1 边界例（非 W8 fixture-trivial·防词汇混淆）。
    **held-out 20% DEFER（§三 S1:90）**：D4 探针切分由 caller formal_train config.probe_holdout 设（非 corpus 层）·
    本生成器供 30 样本可被切·discover 首 MIN_DISCOVER_SAMPLES/recognize 余是隐式 held-out 非 D4 probe 隔离。
    """
    if op not in _ARITH_OP_FAMILIES:
        raise ValueError(f"op={op!r} 须 {_ARITH_OP_FAMILIES}")
    pairs = _distinct_arith_pairs(n)
    vals = _ARITH_DIV_VALS[:n]
    if len(vals) < n:
        raise ValueError(f"n={n} 超 _ARITH_DIV_VALS={len(_ARITH_DIV_VALS)}·扩池或降 n")
    items: list[CollectedItem] = []
    for (p, q), (a, b) in zip(pairs, vals):
        items.append(CollectedItem(
            modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE, source=SOURCE_MATH,
            arith_source=f"lambda {p},{q}: {p}{op}{q}",
            arith_specs=(CodeSpec((a, b), _arith_div_expected(op, a, b)),)))
    return items


def load_arith_s1_corpus() -> list[CollectedItem]:
    """S1 全 4 算子族 diverse corpus（合族·+ arith_bad D2 负通路）。

    **caller 警示**：合族 ③=1000（recognize 干净·4 异骨架各匹配）但 **⑤ task-driven op 选择干扰**
    （verified<total·4 arity-2 op cold-start 竞争）。S1 课程序**须 per-op 分跑**（load_arith_family_corpus
    各族独训·⑤verified=30/30 干净）。本函数仅供合族 smoke / 干扰诊断 / corpus 总览。生产 S1 训练 = 4 次
    load_arith_family_corpus 分跑。（早期"合族 ③745"=旧 arith_bad source 碰撞 artifact·已修·真 ③=1000。）
    """
    items: list[CollectedItem] = []
    for op in _ARITH_OP_FAMILIES:
        items.extend(load_arith_family_corpus(op))
    # arith_bad（D2 负通路·错 expected→vm_proof reward=0→veto）。异参对 (o,p)/(q,r) 超 _distinct_arith_pairs(30)
    # 首 30 对→不撞 + 族/* 族 source（审2 LOW-1·防 observe dedup 同源致 D2 静默失效）·仍匹配 arity-2 + /* 族 op 被选→执行失败→veto。
    items.append(CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                               source=SOURCE_MATH, arith_source="lambda o,p: o+p",
                               arith_specs=(CodeSpec((2, 3), (999, 1)),)))
    items.append(CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                               source=SOURCE_MATH, arith_source="lambda q,r: q*r",
                               arith_specs=(CodeSpec((2, 3), (999, 1)),)))
    return items


def _distinct_arith_triples(n: int) -> list[tuple[str, str, str]]:
    """产 n 个 distinct ordered (p, q, r) 三参对（p/q/r 全异·无 a/i·确定性·bit-identical）。

    同 _distinct_arith_pairs 异参名铁律（防同源 dedup→discover<2→③=0）·扩 arity-3 复合用。
    """
    out: list[tuple[str, str, str]] = []
    pool = _ARITH_PARAM_POOL
    for i in range(len(pool)):
        for j in range(len(pool)):
            for k in range(len(pool)):
                if len({i, j, k}) == 3:   # p/q/r 全异
                    out.append((pool[i], pool[j], pool[k]))
                    if len(out) >= n:
                        return out
    return out


# S2 复合表达式 shape 表（§三 S2·优先级/多步链·每 shape = distinct 嵌套骨架·discover 按 shape 分组）。
# (name, source_template(p,q,r), expected(a,b,c)→(num,den))。arity-3。
_ARITH_S2_SHAPES = (
    ("addmul",   lambda p, q, r: f"({p}+{q})*{r}",   lambda a, b, c: ((a + b) * c, 1)),
    ("muladd",   lambda p, q, r: f"{p}*{q}+{r}",     lambda a, b, c: (a * b + c, 1)),
    ("submul",   lambda p, q, r: f"({p}-{q})*{r}",   lambda a, b, c: ((a - b) * c, 1)),
    ("muladdpa", lambda p, q, r: f"{p}*({q}+{r})",   lambda a, b, c: (a * (b + c), 1)),
    ("prec",     lambda p, q, r: f"{p}+{q}*{r}",     lambda a, b, c: (a + b * c, 1)),   # 优先级：* 先于 +
    ("addsub",   lambda p, q, r: f"({p}+{q})-{r}",   lambda a, b, c: (a + b - c, 1)),
)

# S2 值三参（7 组·全 small 1-9·非全同值·S2 重优先级/多步非量级·submul 用 a>b 保正）。
_ARITH_S2_VALS = [(7, 3, 4), (5, 1, 6), (8, 3, 2), (9, 5, 3), (6, 1, 2), (9, 2, 5), (8, 3, 4)]


def load_arith_s2_corpus(per_shape: int = 7) -> list[CollectedItem]:
    """S2 复合表达式 corpus（doc §三 S2·6 优先级/多步 shape × per_shape 异参 triple = 42+）。

    每 shape = distinct 嵌套骨架（addmul MUL(ADD,c) / muladd ADD(MUL,c) / submul / muladdpa / prec / addsub）·
    discover 按 (shape_signature, operand_arity_hint) 分组·每 shape ≥2 异参 triple→discover fires→③ 泛化。
    **smoke 验证 2026-07-16**：C1 (a+b)*c depth-2 嵌套 → ③=PASS 1000（discover 处理嵌套 BinOp·非机制 gap）。

    **异参名铁律 + 每算子独训**（同 S1）：复合 shape 各自独跑·caller 须 per-shape 分跑（异 opcode 嵌套
    同 arity-3 互扰·合跑 ⑤ op 选择干扰）。教师 level 0（vm_proof 自锚·expected 宿主算术 R6）。
    **诚实边界**：整数 PbE=行为匹配非符号理解（stable≠correct）/ 优先级是结构匹配非语义理解 /
    多项式（Pow x²+3x+2）defer（须 Pow DSL + arity-1·本版 arity-3 BinOp 复合）。
    **★多步链 arity-4/depth-3 DEFER（§三 S2:104·审2 F2·对称多项式 defer）**：规格例 `((a+b)*c)-d`（4 异参
    + depth-3 SUB(MUL(ADD,c),d)）未含·本版仅 depth-2 arity-3（addsub/submul 简化多步）。须扩 _distinct_arith_triples
    至 arity-4 + _ARITH_S2_VALS 四元组。**优先级区分（§三 S2 验收②·审2 F1）DEFER**：addmul(root MUL) vs prec(root ADD)
    shape_signature BFS 序异→discover 分异组非混同（机制保证·structure_discover:175-201,924-930）·**未单测锁定**（合训产
    ≥2 skeleton 测 defer·本版 per-shape 独训验 ③>0）。
    """
    triples = _distinct_arith_triples(per_shape)
    if len(triples) < per_shape:
        raise ValueError(f"per_shape={per_shape} 超 _ARITH_PARAM_POOL triple 池")
    vals = _ARITH_S2_VALS[:per_shape]
    if len(vals) < per_shape:
        raise ValueError(f"per_shape={per_shape} 超 _ARITH_S2_VALS={len(_ARITH_S2_VALS)}")
    items: list[CollectedItem] = []
    for _name, src_tmpl, exp_fn in _ARITH_S2_SHAPES:
        for (p, q, r), (a, b, c) in zip(triples, vals):
            items.append(CollectedItem(
                modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE, source=SOURCE_MATH,
                arith_source=f"lambda {p},{q},{r}: {src_tmpl(p, q, r)}",
                arith_specs=(CodeSpec((a, b, c), exp_fn(a, b, c)),)))
    return items


# ---- S4 迭代等价 corpus（Mode B cross-verify·doc/重来_S3S4迭代机制设计_2026-07-16 §四）----
# 闭式（直线 BinOp·arith_source）≡ 迭代（Sigma CTRL_WHILE·arith_source_b）·同函数异 shape·Mode B 两路独立
# execute_composes_value + rational.eq → agreement（formal_train POST 路径 :712-735·cross_verify_pair）。
# **★ Mode B 绕过 discover**（直 build 两树 + execute·S3 discover:217 排除 ctrl 在此路径不适用）→
# S4 迭代【验证】能力可达（S3 discover-泛化迭代仍 defer·同设计档选项 A）。每 form 各异参名（n/m/s·守
# 异参名铁律·防御性·Mode B discover-free 但不破坏）。expected=真值（PRE-fallback + 文档·POST 丢 expected）。

def _s4_sum(n: int) -> tuple[int, int]:
    """Σ_{k=1}^{n} k = n(n+1)/2（恒整·n(n+1) 偶）·真值。"""
    return (n * (n + 1) // 2, 1)


def _s4_sq(n: int) -> tuple[int, int]:
    """Σ_{k=1}^{n} k² = n(n+1)(2n+1)/6（恒整）·真值。"""
    return (n * (n + 1) * (2 * n + 1) // 6, 1)


def _s4_cube(n: int) -> tuple[int, int]:
    """Σ_{k=1}^{n} k³ = (n(n+1)/2)²（恒整·n(n+1)/2 整）·真值。"""
    t = n * (n + 1) // 2
    return (t * t, 1)


def _s4_odd(n: int) -> tuple[int, int]:
    """Σ_{k=1}^{n}(2k-1) = 1+3+...+(2n-1) = n²（首 n 个奇数和）·真值。"""
    return (n * n, 1)


def _s4_even(n: int) -> tuple[int, int]:
    """Σ_{k=1}^{n}(2k) = 2+4+...+2n = n(n+1)（首 n 个偶数和）·真值。"""
    return (n * (n + 1), 1)


def _s4_fact(n: int) -> tuple[int, int]:
    """n! = Π_{k=1}^{n} k（纯整数累乘·无 math import）·真值。"""
    f = 1
    for k in range(2, n + 1):
        f *= k
    return (f, 1)


# (name, param, source_a DSL, source_b DSL, 探针 n 序, expected_fn)。param 各异（n/m/s/o/e/f·无 a/i·守异参名铁律）。
# Σ 族：source_a=闭式直线 BinOp / source_b=迭代 Sigma（body=i/i*i/i*i*i/2*i-1/2*i 经 _build_expr 建 BinOp）。
# factorial 族：source_a=Prod(CTRL_WHILE 累乘) / source_b=Recur(CTRL_WHILE 递推累乘 a*i)·两迭代 builder 异 shape·
# 覆盖 Prod+Recur builder（审2 MED：非 Sigma-only breadth）。expected_fn: Callable[[int], tuple[int,int]]。
_ARITH_S4_FORMS: tuple[tuple[str, str, str, str, tuple[int, ...], Callable[[int], tuple[int, int]]], ...] = (
    # Σ k：闭式 n*(n+1)/2 vs Sigma(1,n,i)（W2 test_cross_verify_pair_agree 已证此对 AGREE·本表纳入 breadth）。
    ("sigma_sum",  "n", "n * (n + 1) / 2",
     "Sigma(1, n, i)",            (3, 4, 5, 6, 10), _s4_sum),
    # Σ k²：闭式 n(n+1)(2n+1)/6 vs Sigma(1,m,i*i)（body BinOp i*i）。
    ("sigma_sq",   "m", "(m * (m + 1) * (2 * m + 1)) / 6",
     "Sigma(1, m, i * i)",        (2, 3, 4, 5),     _s4_sq),
    # Σ k³：闭式 (n(n+1))²/4 vs Sigma(1,s,i*i*i)（body i*i*i）。
    ("sigma_cube", "s", "((s * (s + 1)) * (s * (s + 1))) / 4",
     "Sigma(1, s, i * i * i)",    (2, 3, 4),       _s4_cube),
    # Σ 奇数：闭式 n*n vs Sigma(1,o,2*i-1)（首 n 奇数和=n²·body 2*i-1 线性）。
    ("sigma_odd",  "o", "o * o",
     "Sigma(1, o, 2 * i - 1)",    (3, 4, 5, 6),     _s4_odd),
    # Σ 偶数：闭式 n*(n+1) vs Sigma(1,e,2*i)（首 n 偶数和=n(n+1)·body 2*i 线性）。
    ("sigma_even", "e", "e * (e + 1)",
     "Sigma(1, e, 2 * i)",        (3, 4, 5, 6),     _s4_even),
    # n!：Prod(1,f,i) 累乘 vs Recur(1,f,a*i) 递推累乘（两迭代 builder·覆盖 Prod+Recur·a=Recur acc·i=索引）。
    ("factorial",  "f", "Prod(1, f, i)",
     "Recur(1, f, a * i)",        (2, 3, 4, 5, 6),  _s4_fact),
)


def load_arith_s4_corpus() -> list[CollectedItem]:
    """S4 迭代等价 corpus（doc §三 S4 + doc/重来_S3S4迭代机制设计_2026-07-16 §四·Mode B cross-verify）。

    6 form = 两路同函数异 shape 等价对·Mode B POST 路径（formal_train:712-735）build root（source_a）+ root_b
    参树（source_b·异 builder 代码路径 R6 真守）+ cross_verify_pair 两路 execute_composes_value + rational.eq
    → reward=1 iff all_agree。覆盖**三个迭代 builder**：Sigma（Σ k/k²/k³/奇/偶·body 经 _build_expr 建 BinOp）
    + Prod（n! 累乘）+ Recur（n! 递推 a*i）。

    **★ S3/S4 课程序（设计档 §五·2026-07-16）**：S3 discover-迭代 **DONE**（option B·ctrl/store 已支持·见
    load_arith_s3_corpus）+ S4 Mode B DONE（本函数·两路等价验证）。迭代"学习(S3) + 验证(S4)"双能力齐。**Mode B 绕过
    discover**（直 build 两树+execute·无 discover 参与）→ Sigma+Prod+Recur 迭代【等价验证】能力经 S4 可达。S3
    discover【学习】leg（自主 LEARN 迭代算子·observe→discover 骨架→register→vm_proof 任意输入值验·doc/重来_S3S4迭代机制设计 §三-bis）。

    **权威 §三 S4 "≥10 对" 状态**：6 form（Sigma 5 body + Prod/Recur）·6/10·余机械 Σ 枚举（Σk⁴ 等同族）defer
    （机制跨三迭代 builder 已证·余为同机制机械扩展·非能力增量）·几何级数 defer（2^n 无多项式闭式·须 Pow 变量指数
    ·VM 不执行 Pow pattern·mode_b 须两路可执行）。

    **异参名**：每 form 各 param（n/m/s/o/e/f·无 a/i）·防御性守异参名铁律（Mode B discover-free·不破坏）。
    **教师 level 0**：vm_proof 自锚·expected 宿主算术 R6（PRE）/ POST 无 expected（丢·守 #479）。

    **诚实边界**：Mode B = 统计学内一致非真理（agreement 非 identity·stable≠correct·#479 墙内弱）·cross-verify
    是两路 agree 非 truth·仅验证非泛化（无 discover held-out·S3 discover-泛化迭代 defer）·两路同出 corpus（系统性
    中毒 agree wrong 仅 corpus 内冗余抓获·mode_b_cross_verify.py:25）·factorial 两路皆迭代（非闭式≡迭代·是两迭代
    builder 互验·R6 真守因 Prod/Recur 异 builder）。
    """
    items: list[CollectedItem] = []
    for _name, p, source_a, source_b, probes, exp_fn in _ARITH_S4_FORMS:
        specs = tuple(CodeSpec((n,), exp_fn(n)) for n in probes)
        items.append(CollectedItem(
            modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE, source=SOURCE_MATH,
            arith_source=f"lambda {p}: {source_a}",
            arith_source_b=f"lambda {p}: {source_b}",
            arith_specs=specs,
        ))
    return items


# ---- S3 discover-迭代语料（doc/重来_S3S4迭代机制设计 §三-bis·discover【学习】leg·2026-07-16）----

# 每 shape ≥ MIN_DISCOVER_SAMPLES(=2) 异参名 examples（distinct source string → distinct COMPOSES root·
# formal_train 按 arith_source 内容哈希建根绕 observe 撞 struct_ref·discover 对齐 ≥2 抽迭代骨架）+ held-out 余。
# 异参名铁律（同 S1/S4·防同源 dedup→<2→③=0）。覆盖三迭代 builder：Sigma(sum/sq) + Prod(factorial) + Recur(factorial)。
# 复用 _s4_* 算术真值函数（truth·非 S4 专属·DRY）。
_ARITH_S3_SHAPES: tuple[tuple[str, str, Callable[[int], tuple[int, int]], tuple[int, ...]], ...] = (
    # (shape_key, dsl_body_template({p}=lambda param), expected_fn, probe_vals)
    ("sigma_sum",  "Sigma(1, {p}, i)",       _s4_sum,  (7, 10, 12, 20)),
    ("sigma_sq",   "Sigma(1, {p}, i * i)",   _s4_sq,   (4, 5, 6, 10)),
    ("prod_fact",  "Prod(1, {p}, i)",        _s4_fact, (4, 5, 6, 7)),
    ("recur_fact", "Recur(1, {p}, a * i)",   _s4_fact, (4, 5, 6, 7)),
)
_ARITH_S3_PARAMS = ("n", "m", "k", "t", "u", "v")   # 异参名/shape（无 a/i）·discover 首2 + held-out 余


def load_arith_s3_shape(shape_key: str, n_examples: int = 4) -> list[CollectedItem]:
    """单 shape 异参名 examples（S3 discover 首 MIN_DISCOVER_SAMPLES + held-out 余·per-shape 独训范式·镜像 S1 family）。

    shape_key ∈ {sigma_sum, sigma_sq, prod_fact, recur_fact}。n_examples 个异参名（_ARITH_S3_PARAMS 首 n·无 a/i）。
    每 example = `lambda {p}: {dsl_body}` + specs（probe_vals 真值·宿主算术 R6）。
    formal_train 按 arith_source 内容哈希建独立根 → route_samples_for_discovery 首 K discover + 余 held-out recognize。
    """
    shape = next((s for s in _ARITH_S3_SHAPES if s[0] == shape_key), None)
    if shape is None:
        raise ValueError(f"unknown S3 shape: {shape_key}（∈ {[s[0] for s in _ARITH_S3_SHAPES]}）")
    _key, body_tmpl, exp_fn, probe_vals = shape
    items: list[CollectedItem] = []
    for p in _ARITH_S3_PARAMS[:n_examples]:
        specs = tuple(CodeSpec((n,), exp_fn(n)) for n in probe_vals)
        items.append(CollectedItem(
            modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE, source=SOURCE_MATH,
            arith_source=f"lambda {p}: {body_tmpl.format(p=p)}",
            arith_specs=specs,
        ))
    return items


def load_arith_s3_corpus(per_shape: int = 4) -> list[CollectedItem]:
    """S3 discover-迭代 corpus（4 shape × per_shape 异参名·discover+held-out·discover【学习】leg）。

    覆盖三迭代 builder：Sigma(sum/sq·CTRL_WHILE+STORE 累加) + Prod(factorial·累乘) + Recur(factorial·递推 a*i)。
    discover_skeleton（S3 §三-bis·ctrl/store 已支持）从 ≥2 异参名 examples 抽迭代骨架 → register 可 inline 复用
    → recognize held-out 新参名 → vm_proof 任意输入值验（sum(7)=28/sq/5!=120·test_discover_iteration 锁）。

    **★ 与 S4 区别**：S4 = Mode B 两路等价【验证】（source_a+source_b·agreement·绕过 discover）。
    S3 = discover【学习】（source_a only·多异参名 examples·discover 抽骨架+recognize held-out+vm_proof）。
    两者合 = 迭代"学习(S3) + 验证(S4)"双能力（doc/重来_S3S4迭代机制设计 §五）。

    **异参名铁律**：n/m/k/t distinct source string → distinct root（防同源 dedup→<2→③=0·同 S1）。
    **诚实边界**：结构归纳（K=2 样本）非符号证明·Rice 有限基底·held-out recognize = alpha-实例识别（同结构树匹配）
    ·真泛化 = 参数化骨架对任意输入值（n=1..20 验·test_internal_sid_no_param_collision_multi_n）·③ permille 构造性（#479 墙·stable≠correct）。
    生产 S3 训练 = per-shape 独 run_capability_exam（run_s3_training.py·报 ③+op_confidence·同 S1 范式）。
    """
    items: list[CollectedItem] = []
    for key, _body, _exp, _probes in _ARITH_S3_SHAPES:
        items.extend(load_arith_s3_shape(key, per_shape))
    return items


def load_code_corpus() -> list[CollectedItem]:
    """代码域种子 corpus（#727·10 code items·⑤ × G5 attribution real data + ⑥ DEAD_LEAK 一致性）。

    每项 def f(n): return n + n（BinOp Add·code_observe 支持）+ 正确 code_specs（expected=2n）。
    Mode A verify（PRE）每项 1 episode·reward=1（spec 正确）→ ⑤ × G5 total≥10 active=0（Mode A verify 非 Mode B）。
    ⑤ status=FAIL（result.generate 空 GenerateSummary·total_tasks=0·rate=0·#726 纠正4 follow-up：generate 永不 None）·
    ⑤ × G5 attribution 与 ⑤ status 分离（#723：status 读 result.generate·G5 cell 读 verify episodes·两路不同信号）。
    ⑥ FAIL（code 域无 CAUSES 边 → strength_delta_total=0）+ G3a/G3b DEAD_LEAK 一致性（#723 归因表·与 language 域同）。
    **诚实边界**：Mode A verify（PRE）·非 Mode B cross-verify（POST·须 weaning gates 过 + gate ON·test_mode_b_cross_verify
    单测已验 4 态·formal_train POST 跑 W2 done·run_weaning_arith weaning_phase=POST 激活·doc/重来_断奶阶段训练设计 W2）。
    """
    return [
        CollectedItem(modality=MODALITY_CODE, domain=DOMAIN_CODE, lang=LANG_NONE,
                      source=SOURCE_CODE, code_source="def f(n):\n  return n + n",
                      code_specs=(CodeSpec((n,), (n * 2, 1)),))
        for n in range(1, 11)   # 10 samples: 1..10
    ]


def load_corpus(kind: str) -> list[CollectedItem]:
    """按 kind 加载种子 corpus（#727 harness ·corpus 开关·确定性 in-memory）。

    kind ∈ {arith, arith_bad, code, lang}。lang 复用 _causal_multi_sent_item 范式（caller 自造·
    此处返空列表 + 提示·lang corpus 非本 loader scope·既有 test_capability_exam 已覆盖）。
    """
    if kind == CORPUS_KIND_ARITH:
        return load_arith_corpus()
    if kind == CORPUS_KIND_ARITH_BAD:
        return load_arith_bad_corpus()
    if kind == CORPUS_KIND_CODE:
        return load_code_corpus()
    if kind == CORPUS_KIND_LANG:
        return []   # lang corpus 既有 _causal_multi_sent_item 覆盖·本 loader 不重复
    raise ValueError(f"未知 corpus kind={kind}·须 {CORPUS_KIND_ARITH}/{CORPUS_KIND_ARITH_BAD}/"
                     f"{CORPUS_KIND_CODE}/{CORPUS_KIND_LANG}")


def source_dist_from_report(report: CollectionReport) -> dict[int, int]:
    """收集报告 → source 分布（edge source 计数·metrics source_dist 审计·五类来源审计）。"""
    dist: dict[int, int] = {}
    for it in report.items:
        dist[it.source] = dist.get(it.source, 0) + 1
    return dist
