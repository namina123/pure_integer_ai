"""config.gates — gate 二分（闭环核心 / 可选·live-read 模块属性·env 覆盖）。

§六 gate 二分：
  - 闭环核心 gate（§六默认 ON·新基线主线·重来无旧序可退 + OFF 则防塌柱①②失效）：
    JUDGE / ATTRACTOR / PR / CONVERGENCE_STEP / CAUSES_SINK / DISPATCH / A2
  - 可选 gate（§六默认 OFF·守回归 / 性能优化非承重）：
    CUE_EXTRACTOR_MODE / COOCCURS_WINDOW_MODE / 各 perf gate
    （原 PROCESS_REWARD_PROP / B3_LU / SPECTRAL 3 dead gate 已删 P3 #1054·见下删注）

**每阶段纪律**：gate 默认 OFF（回归 bit-identical）→ 写机制 → 单测验证 → ON 验承重 → 下一阶段。
故本文件中**尚未接线的 gate 一律默认 OFF**·其所属 Stage 接线并验承重后翻 ON（§六 end-state）。
Stage 2 VM dispatch（execute_composes_value·training 层 verify/task-driven 轮真实活·cognition 不调单向依赖守·
gate DISPATCH_MODE 装饰位零读取 OFF/ON 等价 bit-identical）。
其余 gate 随各 Stage 落地消费·此处定形名字避免后改。
**装饰位范式**（2026-07-11 孤儿审计核证）：DISPATCH/A2/PR/CONVERGENCE_STEP/CAUSES_SINK 五 gate 零 getattr 读取·
机制 production 无条件跑（dag_path 全程不翻 gate）·gate 名仅定形避免后改（同 JUDGE_MODE/GENERATE_MODE 承重件范式·
无 OFF 态·关机制必塌故不可读 gate）。ATTRACTOR_MODE 例外（真实 opt-in·dag_path:232 getattr 读·reward 阶段翻 ON）。

live-read：模块属性保存环境和测试基线；正式 runner 通过 context-local overlay 临时覆盖，
并发或嵌套运行互不串扰。旧测试仍可用 `gates.X = True` 修改进程基线。
"""
from __future__ import annotations

import contextvars
import os
import sys
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


def _flag(env_name: str, default: bool) -> bool:
    """env 覆盖（import 时读一次·测试可翻模块属性）。"""
    v = os.environ.get(env_name)
    if v is None:
        return default
    return v.strip() in ("1", "true", "TRUE", "True", "yes", "YES")


_GATE_OVERRIDES: contextvars.ContextVar[
    tuple[dict[str, bool], ...]
] = contextvars.ContextVar("zero_ai_gate_overrides", default=())


def push_gate_overrides(
        overrides: Mapping[str, bool],
        ) -> contextvars.Token[tuple[dict[str, bool], ...]]:
    """压入当前执行上下文的 gate 覆盖，并返回可精确复位的 token。

    覆盖只影响当前线程或异步上下文；模块属性仍保存进程基线，供环境配置和旧测试
    直接赋值。正式 runner 必须使用本接口，禁止临时改写模块全局值。
    """
    if not isinstance(overrides, Mapping):
        raise TypeError("gate overrides 必须是映射")
    module = sys.modules[__name__]
    normalized: dict[str, bool] = {}
    for name, value in overrides.items():
        if not isinstance(name, str) or not name.isupper():
            raise ValueError("gate 名必须是大写字符串")
        baseline = types.ModuleType.__getattribute__(module, name)
        if type(baseline) is not bool:
            raise ValueError(f"{name} 不是可覆盖的 bool gate")
        if type(value) is not bool:
            raise TypeError(f"{name} 覆盖值必须是 bool")
        normalized[name] = value
    current = _GATE_OVERRIDES.get()
    return _GATE_OVERRIDES.set((*current, normalized))


def reset_gate_overrides(
        token: contextvars.Token[tuple[dict[str, bool], ...]],
        ) -> None:
    """用配对 token 恢复 gate 覆盖栈，嵌套层不得越级复位。"""
    _GATE_OVERRIDES.reset(token)


@contextmanager
def gate_overrides(overrides: Mapping[str, bool]) -> Iterator[None]:
    """在一个词法作用域内应用 context-local gate 覆盖。"""
    token = push_gate_overrides(overrides)
    try:
        yield
    finally:
        reset_gate_overrides(token)


class _ContextLocalGateModule(types.ModuleType):
    """让既有 ``gates.X`` reader 优先读取当前上下文覆盖。"""

    def __getattribute__(self, name: str):
        if name.isupper():
            for layer in reversed(_GATE_OVERRIDES.get()):
                if name in layer:
                    return layer[name]
        return types.ModuleType.__getattribute__(self, name)


# ---- 闭环核心 gate（§六 end-state ON·接线前 OFF 守 stage 纪律） ----

# VM opcode dispatch（execute_composes_value→vm_core.execute→dispatch_binary·training 层 verify/task-driven 轮真实活）。
# cognition 不调 VM（单向依赖守·cognition=5→training=7·struct_bind.py 守）·gate 装饰位保留位（机制不读 gate·
# 零 getattr·无条件跑·同 JUDGE_MODE/GENERATE_MODE 承重件范式·无 OFF 态·OFF/ON 等价 bit-identical·gate 名定形避免后改）。
DISPATCH_MODE = _flag("ZERO_AI_DISPATCH_MODE", False)

# A2 拓扑分层步进调度（a2_layer/a2_layer_oi·dag_path 无条件调）。
# gate 装饰位保留位（机制不读 gate·零 getattr·无条件跑·同 JUDGE_MODE 承重件范式·OFF/ON 等价 bit-identical·gate 名定形避免后改）。
A2_MODE = _flag("ZERO_AI_A2_MODE", False)

# A3 PR wrapper（A3PRWrapper build/solve·dag_path 无条件调）。
# gate 装饰位保留位（机制不读 gate·零 getattr·无条件跑·同 JUDGE_MODE 承重件范式·OFF/ON 等价 bit-identical·gate 名定形避免后改）。
PR_MODE = _flag("ZERO_AI_PR_MODE", False)

# A3 PR 大 n 切 B2 定点迭代（perf·解 O(V³) 密集高斯炸弹·2026-07-13 上帝视角发现+设计本意接线）。
# **功能性 gate（机制读·非装饰位·异于上 PR_MODE）**：matrix.n > PR_B2_LARGE_N_THRESHOLD（a3_pr_wrapper）
# 时 solve 走 B2 定点迭代（稀疏 O(E×iters)·设计 docstring 本意"大 n 兜底"·但原只 PRSingular 回退·未按
# n 大小切→大 n 炸）替 B1 有理高斯（O(n³)+Rational 膨胀·n=656 不可行）。A/B 实测 B2 与 B1 rank 序一致
# （n=200/500 top-20 Jaccard=1.0·maxRelErr~1/B^k·消费者只相对比较→语义等价）。
# default OFF：CI 守 B1 精确 bit-identical；生产大 n 翻 ON（n=656 reward@scale 首次可行·用户"统计上做到即可"框架）。
# B2 近似（FixedQuotient·误差<1/B^k）·_fq_to_rational 转 Rational 累积·mode/exact 标 B2/False（诚实）。
PR_B2_LARGE_N_MODE = _flag("ZERO_AI_PR_B2_LARGE_N_MODE", False)

# 汇聚步进（a2_stepper·PRECEDES AND / CAUSES OR·dag_path 无条件消费）。
# gate 装饰位保留位（机制不读 gate·零 getattr·无条件跑·同 JUDGE_MODE 承重件范式·OFF/ON 等价 bit-identical·gate 名定形避免后改）。
CONVERGENCE_STEP_MODE = _flag("ZERO_AI_CONVERGENCE_STEP_MODE", False)

# CAUSES sink 终点门（dag_path·node==intent.sink 判定无条件）。
# gate 装饰位保留位（机制不读 gate·零 getattr·无条件跑·同 JUDGE_MODE 承重件范式·OFF/ON 等价 bit-identical·gate 名定形避免后改）。
CAUSES_SINK_MODE = _flag("ZERO_AI_CAUSES_SINK_MODE", False)

# Stage 4 消费：attractor 动态演化。§六 end-state ON。
ATTRACTOR_MODE = _flag("ZERO_AI_ATTRACTOR_MODE", False)

# Stage 5 judge 四判据·承重件(防塌柱①)永远 active·judge() 不读 gate·gate 装饰性保留位(无 OFF 态·关 judge 必塌)。
JUDGE_MODE = _flag("ZERO_AI_JUDGE_MODE", False)

# Stage 5 消费：防塌柱③ seeded 探索压力注入（PR 方差趋平时加新种子·③最小版进首版）。
# §六 end-state ON·OFF = 方差够时不注入 bit-identical（柱③ 仍在·proactive probe 关）。
EXPLORATION_MODE = _flag("ZERO_AI_EXPLORATION_MODE", False)

# Stage 5 生成路径填槽分派·承重件(闭环必跑)永远 active·generate_output/dispatch_slot 不读 gate·gate 装饰性保留位(无 OFF 态)。
GENERATE_MODE = _flag("ZERO_AI_GENERATE_MODE", False)

# P0 #1040（2026-07-14）·generate-dispatch 主缺口修复——slot.ref 派发 token concept（非 struct_ref）+
# ctx_refs 用 token（非 unit）。主缺口=generate 派发 struct_ref(NODE_CONCEPT `__seg_*`)·activate_candidates
# 零入向 REFERS_TO→[self]→surface_of(struct_ref)=`__seg_*` label→generate 永不产真词（probe n=4 实证
# 121584 words 100% `__seg_*`·候选恒=1）。根因=模型错配（struct_ref 是段结构锚·generate 却当词汇概念派发）。
# **修法（Path C 存储·非 PRECEDES walk）**：observe attach_token_seq 把段 resolved token concept ref 序存 struct_ref
# def_array（ref_space_id=token 真实 space·每 position 一行·repeat-safe·call-site 门控本 gate）·generate
# graph.read_token_seq 读 def_array ref_space_id!=0 → token concept 序→slot.ref=token concept。activate_candidates
# (token)→PURE_ALIAS 别名词形（多候选·P0b 活）/ [token] 单挂→surface_of(token)→P0a 码点真字。ctx_refs token 化
# （produced_refs/prior_topic_refs 唯一消费方 slot_dispatch·collide/sel_pref/pronoun 皆 token 级
# COOCCURS/selection_pref/pronoun_count·旧填 unit 致错节点=同一 disconnect 族）→6 路 scoring 真活。
# 解语言域统计独立产出腿（判据①编码接地 + ⑤跨语言汇聚·非 truth/非 can_ween·全统计层）。
# **为何非 walk**：PRECEDES walk 按 concept ref dedup·重复 token（功能词"的"跨 position 共享 concept ref）致丢
# position 漏 token·真语料炸。def_array 存储 per-position 不 dedup·repeat-safe。
# **bit-identical（gate OFF）**：observe call-site 不调 attach_token_seq + generate slot.ref=struct_ref + ctx=unit
# → 逐字现状（产 `__seg_*`/占位·6 路 scoring 退化）·CI 零回归。生产 formal_train try/finally 翻 ON（同
# ORDINAL_SURFACE_MODE 范式·两 gate 同翻：dispatch 出 token concept·surface_of 出真字）。实证（toy CAUSES 语料）：
# 翻 ON 产真词 [x,y。,z,w。] reward>0 veto=0 reached_sink=True（旧 walk 误滤 NODE_CONCEPT token 致空产 reward=0·Path C 解）。
DISPATCH_TOKEN_CHAIN_MODE = _flag("ZERO_AI_DISPATCH_TOKEN_CHAIN_MODE", False)

# P0 #1041（2026-07-14）·统计层产出度量——reward 信号 truthiness 校准。judge 加 J4word 加法项
# （产出真 token 覆盖率·output_measure.output_word_ratio 读 OutputPart.token_refs）·gate ON 时进 reward
# 公式（w4·J4word·判据②③信号质量·解 review-2 钉死：旧 slot_fill_rate `if w:` 只判非空→真词/__seg_* 同分）。
# **bit-identical（gate OFF）**：J4word 不计（主守）+ DISPATCH OFF token_refs 空→ratio=0（次守）→reward 逐字现状。
# 生产 formal_train try/finally 翻 ON（与 DISPATCH_TOKEN_CHAIN_MODE + ORDINAL_SURFACE_MODE 三 gate 同翻：
# dispatch 出 token concept·surface_of 出真字·reward 反映真词质量·系统产出端真活非 truthiness）。
OUTPUT_WORD_REWARD_MODE = _flag("ZERO_AI_OUTPUT_WORD_REWARD_MODE", False)

# Stage 6 消费：录放层教师（RecordableLLMTeacher·断奶前在位/断奶后退场）。
# §六 end-state：断奶前 ON（教师 ground-truth 喂 G5/C6 Mode A + define + CAUSES 确认③）/
#   断奶后 OFF（LLM 退场·自评无外部标准 D 墙）。OFF = judge self_proof_fn=None pass=1 占位 bit-identical。
TEACHER_MODE = _flag("ZERO_AI_TEACHER_MODE", False)

# Stage 6 消费：训练编排层（§十二五阶段 + oracle 标定 + promote + cursor 续训）。
# §六 end-state ON·OFF = 训练编排不接（单 episode 跑 bit-identical·无阶段门控）。
TRAINING_MODE = _flag("ZERO_AI_TRAINING_MODE", False)


# ---- 可选 gate（§六默认 OFF·噪音风险 / 性能优化非承重） ----

# P3 #1054 死码清理（2026-07-15）：删 3 dead gate（零 getattr 读 / body=pass·翻 ON no-op·违"无死码"）——
# PROCESS_REWARD_PROP_MODE（reward_propagate 落点⑤ body=pass·pass_reward 台账 defer Stage 6·设计意图保留在
#   reward_propagate.py 注释·非删设计）/ B3_LU_MODE（B3 LU decompose-once ungated 另活 perf round3·gate 死占位）/
# SPECTRAL_MODE（谱分析可选非承重·零读·设计意图见 doc/后继推理与图代数设计.md §四 + 伪代码分析/卷二_过程建模.md
# "谱分析 defer（首版有理高斯够）"·真实现时加新 gate·非此死占位）。3 gate 均不在生产 try/finally flip 列表·删除 bit-identical。
# 误引旧 doc 称 3 gate 者·引用前核证（已删·非 dormant）。

# 输入侧指向词/系词提取（致命3·CAUSES 来源② / IS_A 来源②·裸文本自产）。
# default OFF·守回归（OFF = cue_based_causal_pairs/is_a_pairs 空·等同现状）·ON 启裸文本自产。
# §六 end-state ON（破输入侧 D 墙·反馈腿 CAUSES 输入半边补全）。
CUE_EXTRACTOR_MODE = _flag("ZERO_AI_CUE_EXTRACTOR_MODE", False)

# 性能修复（2026-07-08 训练测试探索实测）：段内共现配对窗口化 O(L²)→O(L·K)（K=COOCCURS_WINDOW_K=2·cooccurs.py）。
# 解训练 scaling 爆炸（LLM 真语料 n=5 段=71s/13209 边/COOCCURS=10120·段内 i<j 全配对 C(53,2)=1378/段是主项）。
# OFF = i<j 全配对 bit-identical 现状；ON = i 仅配 j∈[i+1,i+K]（邻接+1-skip）。
# 生产 formal_train 入口 try/finally 翻 ON（镜像 CUE_EXTRACTOR_MODE·:1226 范式·单测 OFF 守回归）。
# 诚实边界：治段内 O(L²) 主项·跨段 append-only 重复由 COOCCURS_DEDUP_MODE（下）治·不动 EdgeStore.add
# （formal_train:515 invariant 不碰·DEDUP 走专用 add_cooccurs_dedup）。
COOCCURS_WINDOW_MODE = _flag("ZERO_AI_COOCCURS_WINDOW_MODE", False)

# 跨段去重（2026-07-08·总收口 0.1·解 LIVE 病灶①·阻塞 #734）：EdgeStore.add append-only 不去重·
# 同 (a,b,COOCCURS) 跨段重复 pair 堆叠（vocab=50 COOCCURS 爆炸 9684）·真实语料跑不动。
# OFF = build_cooccurs 走旧 add（append-only·strength 恒 1·reader 数行）= bit-identical 现状。
# ON  = build_cooccurs 走 add_cooccurs_dedup（SELECT→UPDATE strength+=1 / INSERT strength=1）·同 pair 合并·strength=频次。
# reader 协同：hub_degree/compute_hub_set/_cooccurs_count/collide_score 全改读 strength 累加（无 gate 分支·
# gate OFF strength 恒 1 累加=数行·完全等价 bit-identical）·collide_score 同改读 strength（gate OFF 频次边数 /
# gate ON strength=频次·两态同值·消歧判别力保持·非集合基数·对抗审 P1-1 纠偏原"不变"误述）。
# 生产 formal_train 入口 try/finally 翻 ON（镜像 COOCCURS_WINDOW_MODE·单测 OFF 守回归）。
# 只走 COOCCURS（PRECEDES order_index 异不可去重·COMPOSES invariant 不碰）。
COOCCURS_DEDUP_MODE = _flag("ZERO_AI_COOCCURS_DEDUP_MODE", False)

# PRECEDES 跨 round 去重（2026-07-09·S2 dead-end 根因 §10.3·doc/重来_PRECEDES_dedup_设计_2026-07-09）：
# EdgeStore.add append-only 不去重·observe 跨 ~16 round 重建 PRECEDES 致 16× 重复（n=1 k=2 实测 2256 vs 153 distinct·
# 137 边组各 16×）·mirror COOCCURS_DEDUP_MODE 范式给 PRECEDES。
# OFF = 3 builder 走旧 add（append-only 堆叠·strength 恒 1）= bit-identical 现状。
# ON  = 3 builder 走 add_precedes_dedup（SELECT (from,to,PRECEDES,order_index)→已存在 skip / 0 行 INSERT strength=1）。
# 与 COOCCURS dedup 关键差异：(a) key 含 order_index（保同概念对多次出现合法 pair·不误并）；
# (b) strength 恒 1（§7.1 结构真值·非频次·已存在 skip 不 +=1）；(c) ≥1 行 silent skip 非 raise（strength 恒 1 无累加歧义）。
# 诚实边界：dedup 是确定性 perf 16× + 数据卫生赢·**未必解 dead-end**（重复边同 from·AND 判定不变·
# dead-end 真因可能是 AND 语义对汇聚 node·dedup 后重测定）·不破 §7.1（strength=1·reward 永不调）。
# 生产 formal_train 入口 try/finally 翻 ON（镜像 COOCCURS_DEDUP_MODE·单测 OFF 守回归）。
PRECEDES_DEDUP_MODE = _flag("ZERO_AI_PRECEDES_DEDUP_MODE", False)

# CAUSES 跨 round 去重（2026-07-13·mirror PRECEDES_DEDUP_MODE·解 observe 16× 重复边膨胀·56% 墙大头）：
# EdgeStore.add append-only 不去重·observe 跨 ~16 round 重建 CAUSES 致 16× 重复（同 PRECEDES 16× bug·
# 设计重画.md:1004 CAUSES strength=reward 调涨学习性测度非频次·频次是 COOCCURS·故 16× 重复违设计=bug）。
# OFF = _insert_causes 走旧 add（append-only 堆叠 16×）= bit-identical 现状。
# ON  = _insert_causes 走 add_causes_dedup（SELECT (from,to,CAUSES,source,epistemic_origin)->已存在 skip / 0 行 INSERT）。
# 与 PRECEDES dedup 同（strength 恒 base·silent skip·不 raise）·key 含 source+epistemic_origin 保 provenance
# （3 epistemic 源同三元组是合法不同边·但同源同 epistemic 跨 round 16× 是 bug）。
# **reward 影响零**（终审 resolver 一锤定音·3 路径全证伪·审1"非零"判错）：① build_matrix（a3_personal_rank:107）
# sum 16× dup 权重->PR matrix 变->PR 向量变·**但 stepper.advance 不读 PR**（读 self.active+edge dict）·
# maybe_expand_attractor 长 e_set 不回传 stepper.active（构造时拷贝·后续独立增长）·16 dup identical->min 选同
# EdgeRef->path.edges 不变。② 假汇聚修（a2_stepper:203->struct_unit_refs 变）·无 role_seq node 无输出（同）·
# 有 role_seq node 跨层 append 产重复 OutputPart 但 J1(LCS 不增)/J2s(比率守 1)/J3path(读 path.edges)/G 门全不变
# ->reward 不变。③ snapshot_strengths 覆写去重+record_episode_result 同 delta->strength_delta 同。dedup 是纯 perf
# （消 16× 边膨胀·56% 墙大头）+ 数据卫生（16× edge count 膨胀违设计 strength=reward 学习性测度非频次）·
# gate OFF 守 CI bit-identical。cursor 续训 silent skip 残留 16× 边（良性·mirror PRECEDES·假汇聚残留 benign）。
# 生产 formal_train 入口 try/finally 翻 ON（镜像 PRECEDES_DEDUP_MODE·单测 OFF 守回归）。
CAUSES_DEDUP_MODE = _flag("ZERO_AI_CAUSES_DEDUP_MODE", False)

# PR 热区过滤（2026-07-13·perf·解 PR matrix 全图 8677²(n=656)·设计本意卷二:110 hotzone_adj radius·全图是 defer 意外态）：
# A3PRWrapper.build 内 BFS k-hop 从 local_seeds 出发（出边 only·PR 沿出边传播）·matrix.nodes/weighted 缩到 k-hop 子图。
# OFF = 全图 PR（matrix=全 subgraph_edges PRECEDES+CAUSES+OCCURRENCE·现状）= bit-identical CI。
# ON  = k=2 BFS·PR_MAX_NODES=2048 硬上界（超界按 BFS 近度+ConceptRef 序截断·确定 bit-identical）。
# **reward 影响零**（终审 resolver 证 PR matrix 不回流 path：stepper.advance 不读 PR·attractor 长 e_set 不回传
# stepper.active·16x/1x 同 EdgeRef·judge 不读 pr_vector）·仅变 PR 诊断（anti_collapse ③方差/convergence/exploration
# 候选池·不挂 stage_metric_gate）。③方差从全图稀释假阳性 fire 修为热区集中->修假阳性非破坏。
# B2 路径（生产 n>512）E 减 ~40x·与 round3 LU decompose-once 正交叠加。实施在 build 内（**非 formal_train:475**
# 过滤 subgraph_edges·:475 全边表留 stepper/topo/dead_end/attractor 5 消费者）。
# 诚实标注：①HOTZONE_MODE ON 须配合 PR_B2_LARGE_N_MODE ②k=2 oracle 起点须实测标定 ③③方差诊断语义变（修假阳性）
# ④attractor out-of-hotzone 候选 add_seed no-op≡seed_rank<θ_coh（零行为变·K_CAP=8<<热区池）⑤热区=拓扑邻近非语义 stable≠correct。
# 生产 formal_train 入口 try/finally 翻 ON（镜像 CAUSES_DEDUP_MODE·单测 OFF 守回归）。
HOTZONE_MODE = _flag("ZERO_AI_HOTZONE_MODE", False)

# PRECEDES AND→OR 语义改（2026-07-09·S2 dead-end factor A·doc/重来_PRECEDES_OR_语义改_设计_2026-07-09）：
# dag_path 生产 language-only（episode_loop 唯 production caller formal_train:459=path⑦·code 走 verify_round 绕
# dag_path）→ PRECEDES AND（def→use 完备性设计）production 死语义·language 重复词概念多前驱·AND 全 active
# 永不满足致 dead-end（post-dedup 5 前驱仅 1 active→BLOCKED→层尽 DEAD_END）。
# OFF = a2_stepper PRECEDES 走 AND（旧 def→use 完备性·bit-identical 现状·2 合成测试不翻 gate 走 AND 过）。
# ON  = a2_stepper PRECEDES 走 OR（任一前驱 active 即推进·返全活跃前驱边·解 language dead-end factor A）。
# **诚实边界**：(a)A 修不保证 REACHED_SINK——factor B（COVERAGE_THRESHOLD=500·coverage 口径）可能仍挡·
# A 修后须立即测 B（同 PR·否则 A 无生产效果=纸面闭合）；(b)OR 下通识词（word_terminated）后继可经其他活跃
# 前驱推进（非"仍 BLOCKED"·T7 gate ON 翻 REACHED_SINK）·word_terminated "后继 BLOCKED" 契约 under OR 削弱
# （language 正确行为·通识词不该阻塞路径）；(c)struct_unit_refs 语义 under OR 从"全前驱汇聚"变"活跃前驱汇聚"
# （generate 消费 benign·set 并集非查全到齐）；(d)dead-end 修 ≠ reward>0（G5 排语言=#479 正交挡）。
# 生产 formal_train 外层 try/finally 翻 ON（镜像 PRECEDES_DEDUP_MODE·:1558/:1771 范式·覆盖 episode_loop + H2）。
PRECEDES_OR_MODE = _flag("ZERO_AI_PRECEDES_OR_MODE", False)

# PRECEDES oi-first-occurrence 序遍历（2026-07-09·S2 dead-end factor C·doc/重来_F2_PRECEDES_oi遍历_设计_2026-07-09·v3）：
# factor C = language PRECEDES 概念成环（token 重复）·Kahn（a2_topology.py:94）丢所有环节点+下游（含 sink）·
# sink 永不可达 -> dag_path 对 language 恒 DEAD_END -> reward 恒 0（G2p=0）-> language 零学习。
# F2 = 弃 Kahn·改按 first-occurrence order_index 建层（含环节点·每节点访一次·acyclic by construction）。
# OFF = a2_layer（Kahn·current）= bit-identical 现状·1418 测零回归。
# ON  = a2_layer_oi（_build_topo_layers_oi·first_occ gap 检测 + normalized_max_in tiebreak·v3 解 inter-seg oi 不兼容）+
#       a2_stepper PRECEDES 走 OR（OI_MODE implies OR·§五bis step 10·OR 条件 `PRECEDES_OI_MODE or PRECEDES_OR_MODE`）。
# **v3 修（施工前 code 实读发现）**：inter-seg oi 巨值（seg_order_base+i×TOKEN_CAP_OFFSET·seg_order_base=全段 token 总数）
# 与 v2 first_occ=min(out-edge oi) 不兼容（last_token[i] 唯一出边=inter-seg 巨->first_occ 巨->后访·struct_ref[i+1] 反向先访·
# inter-seg 链断）。v3 first_occ gap 检测（min(out)<=max(in)+1 用 out·否则用 max(in)+1）+ tiebreak normalized_max_in
# （滤 inter-seg 巨入边）-> last_token[i] 排回段末·inter-seg 链前向·segment 0..n-2 全激活·sink 可达·reward>0 通路开。
# **诚实边界**：末段(n-1) tokens 在 sink(=struct_ref[-1]·末段首) 后不访·末段 CAUSES 丢（1 段损失·accept）；
# backward CAUSES 丢（§八代价1）；OR 死锁依赖既有 ATTRACTOR/EXPLORATION（§八代价3·生产 ON）。
# payoff v3≈MEDIUM（reachability 真解 + 多段 CAUSES 收集 + 生产探索破死锁 + reward>0 真流）。
# 生产 formal_train 外层 try/finally 翻 ON（镜像 PRECEDES_OR_MODE·:1558/:1771·生产 OI_MODE ON 时 DEDUP+OR 也 ON·三者叠加）。
PRECEDES_OI_MODE = _flag("ZERO_AI_PRECEDES_OI_MODE", False)

# 层1 同段指代候选（2026-07-09·factor E·doc/重来_factorE_层1指代_intra_seg_设计_2026-07-09）：
# factor E = 同段前指代词（"动物...它们"同段）无候选 → resolve_pronoun_occurrence 返 None → dangling →
# J4 ② fire → G4 veto → reward=0（language 零学习）。judge.py:58 注释声称"层1 单句指代已解析"是 theater——
# #733 只实施层3（OCCURRENCE 边）+ ② fix·层1（同段前序 token）候选生成从未实现 = 纸面闭合。
# OFF = resolve_pronoun_occurrence 层1 块跳过 → 候选源仅层2(跨段FIFO)+层3(OCCURRENCE) = current·bit-identical。
# ON  = 层1 候选源（当前段前序 token·work_memory._current_segment_refs·score=k+1 近因·hub_set 过滤）→
#       同段前指代词解析非悬空 → J4 ② 不 fire → reward>0（G2p F2 解·G3a/G3b/G5 语言域不激活）。
# **诚实边界**：层1 启发式近因非语义消解（"它们"→最近 token·stable≠correct·#479 墙）·reward>0=可训练非语义正确。
# 生产 formal_train 外层 try/finally 翻 ON（镜像 PRECEDES_OI_MODE·:1573/:1787·四 gate 叠加 DEDUP+OR+OI+INTRASEG）。
PRONOUN_INTRASEG_MODE = _flag("ZERO_AI_PRONOUN_INTRASEG_MODE", False)

# ---- 刀4 件2 验证下涌现关系学习（初心·4 子环·3 gate 默认 OFF 守 bit-identical） ----
# 设计 doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀4。3 gate 独立（细粒度回归隔离）。
# end-state：涌现实体接线验承重后逐 gate 翻 ON（formal_train 生产入口 try/finally 守·镜像 CUE_EXTRACTOR_MODE）。

# 子环1+2：涌现假设生成器（PRECEDES 链 connector 定位）+ 假设落 D:11 SHADOW 信号边。
# default OFF·守回归（OFF = 不涌现假设·不落 SHADOW D:11 边·等同刀3 后现状）。
EMERGENT_RELATION_HYPOTHESIS_MODE = _flag("ZERO_AI_EMERGENT_RELATION_HYPOTHESIS_MODE", False)

# 子环3：reward_propagate concept_targets 扩展（D:11 SHADOW 候选 word concept 进概念维 feed·
# 鸡生蛋破解·守 causes_edges 不变/D:11 边不进 record_episode_result·只 word ref 进 concept_targets）。
# default OFF·守回归（OFF = concept_targets 不扩展·等同现状·experience_count feed CAUSES-only 端点）。
EMERGENT_RELATION_FEED_MODE = _flag("ZERO_AI_EMERGENT_RELATION_FEED_MODE", False)

# 止血 #1146（methodology §五·reward 非 frame）：CAUSES edge reward 写按域过滤——reward-illegitimate 域
# （语言 DOMAIN_TEXT / bare·判据 = 不在 shared.REWARD_LEGITIMATE_DOMAINS·与 judge G5 激活域同源）剔出
# reward_propagate 落点① edge 写（dead-end/veto→tn++ 惩罚唯一 reward-active 边·有害）。default OFF 守回归
# （OFF → reward_propagate 落点① 逐字现状·CI 零语言 CAUSES 写 → 条件恒 False ·bit-identical）。
# 生产 formal_train try/finally 翻 ON（语言域 reward 退场·仅剔 edge 写·experience_count 概念维对偶保留）。
CAUSES_REWARD_DOMAIN_FILTER_MODE = _flag("ZERO_AI_CAUSES_REWARD_DOMAIN_FILTER_MODE", False)

# 决断5：cue_type_of D:11 readback（第二源·读 D:11 PRIMARY 边→REL_*→cue_type·反 theater 关键）。
# default OFF·守回归（OFF = cue_type_of 纯 frozenset·退化现状·lookup_word_concept 零生产 caller）。
# 冷启动退化：D:11 全 SHADOW（未 promote）→ 第二源返 None → 退化纯 frozenset。
EMERGENT_RELATION_CUE_READBACK_MODE = _flag("ZERO_AI_EMERGENT_RELATION_CUE_READBACK_MODE", False)

# STEP5 PR2：operator-level D:11 文字 alias 迁（arith_op_of/comparison_op_of 第二源·读 D:11 PRIMARY 边
# →OP_* concept→opcode·镜像 EMERGENT_RELATION_CUE_READBACK_MODE 两源范式·D6 开放类文字走 D:11 learnable）。
# default OFF·守回归（OFF = arith_op_of/comparison_op_of 纯 frozenset _ARITH_OP_WORDS/_COMPARISON_OP_WORDS·退化现状）。
# 冷启动退化：D:11 OP_* 无教师种子 → 第二源返 None → 退化纯 frozenset。反 theater：未验证 SHADOW 不注入。
OPERATOR_D11_READBACK_MODE = _flag("ZERO_AI_OPERATOR_D11_READBACK_MODE", False)

# 审计根治：modal-level D:11 文字 alias 迁（modal_op_of/is_modal_cue 第二源·读 D:11 PRIMARY 边
# →MODAL_KIND_* concept→modality·镜像 OPERATOR_D11_READBACK_MODE 两源范式·D6 模态种类归抽象空间后天可学习）。
# 解 [严重-1] _MODAL_CUES 换名字写死（"无 REL_MODALITY 故无 D:11"循环论证偷渡）·建 modal_kind concept +
# ATTR_MODAL_KIND=22 readback（镜像 OP_*）+ abstract_mark MARK_MODAL_KIND=5 D6 归属·不违 STOP（ATTR_* 非 TYPE_*）
# 不违 D6（abstract_mark 归属）。default OFF·守回归（OFF = modal_op_of/is_modal_cue 纯 frozenset _MODAL_CUES·退化现状）。
# 冷启动退化：D:11 MODAL_KIND 无教师种子 → 第二源返 None → 退化纯 frozenset。反 theater：未验证 SHADOW 不注入。
MODAL_D11_READBACK_MODE = _flag("ZERO_AI_MODAL_D11_READBACK_MODE", False)

# #940 否定词 D:11 readback 第二源（is_negation_cue 读 D:11 PRIMARY→TYPE_NEGATION concept·镜像 MODAL_D11_READBACK_MODE 两源范式）。
# 否定=符号域先天（TYPE_NEGATION=12·同 operator·非 modal 抽象空间）·复用 ATTR_SYMBOL_TYPE=17 不挂 abstract_mark·
# 激活 ensure_symbol_types（shadow→活·bootstrap_negation_signals 消费）。default OFF·守回归
# （OFF = is_negation_cue 纯 frozenset _NEGATION_CUES·退化现状）。冷启动退化：D:11 无教师种子 → 第二源返 False → 退化 frozenset。
# 反 theater：未验证 SHADOW 不注入（tier_filter=TIER_PRIMARY）。
NEGATION_D11_READBACK_MODE = _flag("ZERO_AI_NEGATION_D11_READBACK_MODE", False)

# B-PR1 action D:11 readback 第二源（is_action_intent_cue 读 D:11 PRIMARY→ACTION_INTENT_* concept·doc §16·镜像 NEGATION_D11_READBACK_MODE 两源范式）。
# 动作意图=符号域先天（INTENT_COMMAND_MOOD + ACTION_*·镜像 operator·非 modal 抽象空间）·挂 ATTR_OPERATION_INTENT=23
# 不挂 abstract_mark·ensure_action_primitives（bootstrap_action_signals 消费）。命令词（帮我/请→COMMAND_MOOD）+ 动作词（生成/计算→ACTION_*）。
# default OFF·守回归（OFF = is_action_intent_cue 纯 frozenset _ACTION_LEXICAL_CUE·退化现状）。冷启动退化：D:11 无教师种子→第二源返 False→退化 frozenset。
# 反 theater：未验证 SHADOW 不注入（tier_filter=TIER_PRIMARY）。**W7 命令判定复用此**（_has_action_intent 调 is_action_intent_cue·命中任一→type=COMMAND）。
ACTION_D11_READBACK_MODE = _flag("ZERO_AI_ACTION_D11_READBACK_MODE", False)

# B-PR2 动作意图经验回写（doc §17·施工序 §14.4 第3步·W7+B-PR1 词法/概念两层后的"经验"第三层）。
# formal_train._run_reward_round episode_loop 返后：D3（type==INTENT_COMMAND + terminal==REACHED_SINK）激活 →
# collect_action_intent_concepts 扫 segments tokens D:11 PRIMARY→distinct ACTION_* concept refs → record_experience_outcome
# 写 experience_count（reward 驱动 R1：reward>0→e_sn++&e_tn++ / reward==0→e_tn++·对偶 op_confidence·§16.4）。
# ctx_code 自动落 COMMAND 桶（pack_ctx_code 第4维 intent_type·§17.2）·B-PR3 读 ATTR 类别非率（gate③ D:11 存在性·§18.1 决断2·率 B-PR4+ 消费）。
# **D3 reward>0 = R1 成功臂非排除闸**（设计审 B CONFIRMED·§17.1 决断2）：硬排除→率恒1 β_arith 病→B-PR2 无意义·
# 故 reward==0（veto）亦写 e_tn++ only→率<1 有判别力·"后实验验证松紧"（§14.2）。
# **依赖链**：本 gate ON 但 INTENT_COMMAND_MODE OFF → type 永 QUESTION → D3 永假 → 零写（须 INTENT_COMMAND_MODE 同 ON 才活）。
# default OFF·守 CI 回归（OFF → hook 整段 skip → experience_count 零新增 → 1864 bit-identical）。
# **生产 try/finally 暂不翻**（B-PR3 未实施·写无消费者·独立 gate 留增量验证控制点·§16.7/§17.1 决断4）。
ACTION_EXPERIENCE_FEED_MODE = _flag("ZERO_AI_ACTION_EXPERIENCE_FEED_MODE", False)

# B-PR3 gate③ _intent_override 接通（doc §18·施工序 §14.4 第4步·断桥粗粒度 meta 内含）。
# dag_path.word_terminated 概念阻断三 gate 之③（首版返 0 未活·dag_path.py:93）：命令态（intent.type==INTENT_COMMAND）
# 动作词（D:11 PRIMARY 边到 ACTION_*/COMMAND_MOOD）不被 gate① freq 终止→留 path→dag_path 导向动作拓扑（§13.3 路由层）。
# **D:11 per-word 查找 + intent.type==COMMAND 闸**（§18.1 决断1·实施者推翻设计审"不查 intent"：语义[函数名/参数/注释全 intent 中心]
#   + 一致性[B-PR2 D3 要求 COMMAND §17.1·gate③ 同属动作链] + blast radius 仅 COMMAND[QUESTION 不路由动作执行·更安全]）。
# gate③ 不读 experience_count 率（§18.1 决断2·二元判定非排序·率消费者 B-PR4+ attractor seed 权重·纠正"读此率"stale）。
# **依赖链**：本 gate ON 但 INTENT_COMMAND_MODE OFF → intent.type 永 QUESTION → 决断1 intent 闸返 0 → gate③ 永空 override
#   （须 INTENT_COMMAND_MODE 同 ON 才活·三 action gate B-PR2~4 就绪后同翻 §16.7）。
# default OFF·守 CI 回归（OFF → _intent_override 首行早返 0 → word_terminated :138 不变 → 1881 bit-identical）。
# **生产 try/finally 暂不翻**（CHANNEL meta 无下游消费者·B-PR4 未实施·独立 gate 留增量验证控制点·§18.1 决断4）。
ACTION_INTENT_OVERRIDE_MODE = _flag("ZERO_AI_ACTION_INTENT_OVERRIDE_MODE", False)

# B-PR4 attractor 多节点种子偏向（doc §19·施工序 §14.4 第5步·路由层收口）。
# dag_path.dag_path_step 入口 local_seeds/e_set 扩张（mirror #728 replay·gate 守 + subgraph_nodes 过滤）：
# 命令态（intent.type==INTENT_COMMAND·formal_train 预算）动作词概念（D:11 源端 word concept·**非 ACTION_* 元概念**
# 后者不在 PR matrix→add_seed no-op theater·doc §19.0 矛盾 A）作多节点种子注入 attractor e₀ → PR 偏向动作拓扑邻域
# （§13.3·复用 attractor 不改数学·maybe_expand_attractor 松入严留不动）。
# **experience_count 率消费者在此接通**（§18.1 决断2"B-PR4+ 消费率"兑现）：_collect_action_seed_candidates 读
# read_experience_count(action_ref, COMMAND ctx_code)→**洗净 filter（sn==0 tested-never-verified 滤除·ACTIVE 率消费者·
# 非 theater）** + rate-sort survivors（dormant ordering·PR 交换律致纯排序行为惰性·defer top-K cap·结构对偶
# structure_discover:1144-1154 洗净+sort 两阶段）。不改 _seed_weight（freq/sp dock 既有乘子·率-value 进 PR defer 独立步）。
# **依赖链**：本 gate ON 但 INTENT_COMMAND_MODE OFF → intent.type 永 QUESTION → formal_train intent 守不预算
# → workmem.action_seed_candidates 空 → dag_path 跳过（须 INTENT_COMMAND_MODE 同 ON 才活）。
# default OFF·守 CI 回归（OFF → workmem 字段空 → dag_path `if candidates:` 假 → 跳过 → local_seeds==seeds → 1894 bit-identical）。
# **生产 try/finally 暂不翻**（下游 G 合成+维度桥未落·翻生产=改 COMMAND 路径无验证网·独立 gate 留增量验证控制点·
# §16.7 扩展"B-PR2~4 全就绪"四 action gate 同翻：INTENT_COMMAND_MODE+ACTION_EXPERIENCE_FEED_MODE+ACTION_INTENT_OVERRIDE_MODE+本）。
ACTION_SEED_BIAS_MODE = _flag("ZERO_AI_ACTION_SEED_BIAS_MODE", False)

# 相1 G-PR1 算术归纳合成（doc §三/§二十·施工序 §14.4 第7步[原 #8 提前·§14.4.1 纠偏]）。
# _run_verify_round PRE（formal_train）独立合成分支：synthesize_value 行为匹配搜索骨架池
# （load_discovered_operators·复用 execute_composes_value·两级搜索：直接 arity 匹配 + PARAM 绑定枚举）
# → 有匹配 reward=1 root=搜索产物 sink 重指 / 无匹配 reward=0（DISAGREE 牙·诚实·非伪造）。
# gate ON 跳过 vm_proof_fn 循环（synthesize_value 内部 execute 已行为匹配·非冗余双执行）。
# **断桥细粒度 defer 到相1后**（§14.4.1 三重 gap·相1 是断桥真消费者）。
# default OFF·守 CI 回归（OFF → _run_verify_round PRE vm_proof_fn 循环不变 → root=struct_refs[0] → 1909 bit-identical）。
# **生产 try/finally 暂不翻**（断桥+维度桥未落·翻生产=改 verify 路径须验证网·独立 gate 留增量验证控制点）。
VALUE_SYNTHESIZE_MODE = _flag("ZERO_AI_VALUE_SYNTHESIZE_MODE", False)

# STEP5 PR4：REL_SIMILAR EDGE_SIMILAR slot-filler 消费者（dispatch_slot 读 EDGE_SIMILAR 扩展 slot 候选）。
# default OFF·守回归（OFF = dispatch_slot 不扩展候选·退化现状·无 EDGE_SIMILAR 消费）。
# D2 合规：EDGE_SIMILAR 二元离散边·图遍历扩展候选·非相似度 SCORE·非向量。
SIMILAR_SLOT_MODE = _flag("ZERO_AI_SIMILAR_SLOT_MODE", False)

# 审计根治 [严重-3]：B6 生成侧 dispatch_slot pronoun scoring（读 pr_tn 加 slot 候选分·镜像 STEP5 PR4
# similar_candidates + observe 侧自消费范式·pair-key 对偶 observe 侧 pronoun→antecedent）。
# default OFF·守回归（OFF = dispatch_slot 不读 pr_tn·退化现状·无 pronoun 生成侧判别）。
# 与 PRONOUN_RESOLVE_COUNT_MODE（observe 侧读写门）分立——observe 写读用 PRONOUN_RESOLVE_COUNT_MODE·
# 生成侧读用 PRONOUN_SLOT_MODE·细粒度回归隔离（镜像 SELECTION_PREF_MODE 写 / GENERATE_SELECTION_PREF_MODE 读 分立）。
PRONOUN_SLOT_MODE = _flag("ZERO_AI_PRONOUN_SLOT_MODE", False)

# G2 修饰方向A（ 的-cue head/modifier 统计·source+read-time·doc/重来_G2_修饰方向A_设计_2026-07-15.md）。
# 中文 "X 的 Y" → X=modifier（ 的 前）·Y=head（ 的 后）·head 更 salient。
# source 侧（observe token loop·2-token lookback· 的-cue → observe_modification 写 modification_hist）gate-independent
# （表 inert 当无读·镜像 position_hist）·read 侧（dispatch_slot 第 6 路 head_pref_score·head 偏好 bonus cap 9·并入 sp 维
# 联合 _cap_sp cap 999·守 collide 主轴）用本 gate。
# default OFF·守回归（OFF = dispatch_slot 不读 head_pref·combine 逐字现状·modification_hist populated 但 inert→bit-identical）。
# 真消费者 dispatch_slot generate 选词·corpus_zh 富 的-修饰。诚实边界： 的-位置统计非语义中心语（#479 truth 墙）。
MODIFIER_DIRECTION_MODE = _flag("ZERO_AI_MODIFIER_DIRECTION_MODE", False)

# 刀5 件5：选择倾向共现统计 builder（selection_pref_count 表 sp_tn 写·§十 边约束·避免"我吃猫"/"石头追老鼠"）。
# default OFF·守回归（OFF = observe 不写 sp_tn·selection_pref_count 表全空·等同刀4 后现状）。
# PR 软加权 dock seed 向量 defer S4（seed=struct_ref 真阻塞·件5 只交付 stats 地基·反 theater 用 sp_tn count 区分）。
SELECTION_PREF_MODE = _flag("ZERO_AI_SELECTION_PREF_MODE", False)

# 刀6 件7：sense 多义管线修通（sense_candidates 表摄入侧写真 + 理解侧 recognize clone 选 sense）。
# default OFF·守回归（OFF = observe 不写 sc_tn·sense_lookup hook 返 None·MultiRef 不产·caller 建 COMPOSES
# 走 ensure(tok)·无 clone·sense_candidates 表空·等同刀5 后现状 bit-identical·退化链 5 步）。
# 用户拍板 Option B（修通 MultiRef 域内管线·非 theater substrate）·ON = sense_facts boot 种 +
# observe 真持久化多 sense + 理解侧 recognize clone 逐 sense 试（IS_A 共祖结构选优·非语义消歧·#479 墙不破）。
# sc_sn reward feed / PR docking / 多 token 笛卡尔积全 defer（地基非楼·同 selection_pref_count 诚实定位）。
SENSE_LOOKUP_MODE = _flag("ZERO_AI_SENSE_LOOKUP_MODE", False)

# S4 三乘子进 PR（selection_pref 维 dock PR seed·学习放开 6 刀后主线续接·doc/重来_学习放开整合设计_纠偏纠偏.md §5）。
# 2 gate 独立（细粒度回归隔离·镜像刀4/5/6 范式）。end-state：selection_pref 维 dock 验承重后翻 ON
# （formal_train 生产 reward 阶段 try/finally 守·同 ATTRACTOR_MODE/EXPLORATION_MODE 范式 :362-365）。

# 片2：_seed_weight 乘积 dock（w = mul(w_freq, w_sp)·sp_agg = sum_base + sum_sp_tn·attractor 扩张路径 token seed 真生效）。
# **纠偏（S4 Plan agent·doc L264）**：seed=struct_ref 是数据真空非机制阻塞（freq/selection_pref 维真生效走 attractor
# 扩张·ATTRACTOR_MODE ON reward 阶段·token concept seed 有 eff_freq/sp_agg）·乘子 dock 走 _seed_weight 权重缩放
# 不动 seed 节点集不破遍历。default OFF·守回归（OFF = w_sp 恒 ONE·_seed_weight = w_freq·落点 A 不变 bit-identical·等同刀6 后现状）。
SELECTION_PREF_DOCK_MODE = _flag("ZERO_AI_SELECTION_PREF_DOCK_MODE", False)

# 片3：sp_sn reward feed 第三条腿（reward_propagate 落点⑥·concept_targets 配对算 class_of·守 :131 CAUSES-only assert）。
# default OFF·守回归（OFF = 第三条腿整段跳过·reward_propagate 既有 5 落点不变 bit-identical·sp_sn 列永远=0 等同刀5 后现状）。
SELECTION_PREF_FEED_MODE = _flag("ZERO_AI_SELECTION_PREF_FEED_MODE", False)

# S4 决断 2 两层正交第二腿·生成侧精查（dispatch_slot 加 sel_pref 维·CLASS 级共现 boost·collide 全 0 时
# 未见 token 经 IS_A class 共现泛化）。combine = collide×SCORE_SCALE + min(sp, SCORE_SCALE-1)（collide 主轴·
# sel_pref 亚主轴 cap 守 1 个真 token 共现 > 999 个 class 共现）。gate OFF 时 dispatch_slot scored 不重算
# if 外短路·bit-identical 硬守（984 测零翻·graph.selection_pref_score 不调·零 IO·ancestor_map_cache 不 build）。
# **承重件零侵入**：dispatch_slot 签名零动·gate 在函数体 live-read（同 generate.py:31 范式·非 gate 内默认值）。
# default OFF·守回归·镜像刀4/5/6/S4 片1-3 四 gate 独立范式（写侧/PR 侧/reward 侧/生成侧）。
GENERATE_SELECTION_PREF_MODE = _flag("ZERO_AI_GENERATE_SELECTION_PREF_MODE", False)

# ---- perf round6（2026-07-13）：dag_edges scope 到 seed 前向可达子图（O(n²) 真杠杆） ----
# dag_path_step 在**全累积 dag_edges**（PRECEDES+CAUSES+T_STEP·随 item 累积 O(n) 涨）上建 path/topo →
# path 随 n 累积增长 → dispatch_slot/path-work O(n²)（6.6x/2x 实证·path-scope B 证 path≈active 无 off-path 死载
# → 真源是 path 自身累积涨）。gate ON → formal_train BFS 从当前 item seed 前向可达（out-edges）·只传 reachable
# edges 给 dag_path_step·path 不再累积全图。**bit-identical 条件**：dag_path 的 path 是 seed→sink 前向走·
# unreachable 节点 stepper 本就 no-op → reachable-only 的 path/topo（reachable 子集）== 全图的 path
# （AB hash 测·镜像 path-scope B 范式）。default OFF 守 CI·生产 try/finally 翻 ON。
DAG_PATH_REACHABLE_SCOPE_MODE = _flag("ZERO_AI_DAG_PATH_REACHABLE_SCOPE_MODE", False)

# ---- perf round8（2026-07-13）：语言 struct_ref 内容哈希（dispatch 累积止血·O(n^1.5) 主因） ----
# observe 段结构概念点 __seg_{stage}_{seg_idx}：seg_idx 每 observe 重置 → 跨 item 同 (stage,seg_idx)
# 撞同 struct_ref（concept_index.ensure 内容哈希去重）→ attach_role_seq 在同 struct_ref 反复 insert（无幂等）
# → 千段累积 ~4564 role → generate read_role_seq 返全长 → dispatch_slot 爆炸（n=12 实测 730K dispatch）。
# gate ON → 语言分支 h63(seg.tokens)（Segment.tokens=语言段"源"·镜像 code/arith 内容哈希机制·同句去重合并）。
# **bit-identical 条件**：单 item 内 seg_idx 本就唯一 + 段内容互异 → 内容哈希与 seg_idx 同给唯一 struct_ref
# → 单 item bit-identical；多 item 累积才发散（旧路径累积=bug·新路径挡跨句碰撞）。
# **诚实边界（2审 FINDING-A）**：哈希**机制**镜像 code/arith·**够用程度**不成立——code/arith continue 在
# attach_role_seq 前（不承载 role_seq）·语言会 attach·故本修只挡**跨 item 跨句**碰撞（主因~960×·n=12 dispatch
# 730K→~2.9K≈254×）·**同句/跨轮 re-observe 仍累积**（~16× 残留·baseline 既有）。真根因=attach_role_seq 无幂等·
# defer（first-write-wins guard·round8b）。n=656 真墙已转到 PR 大 matrix（matrix.n 650·见 round9）·非 dispatch。
# default OFF 守 CI·断奶/perf 跑设 ZERO_AI_STRUCT_REF_CONTENT_HASH_MODE=1（镜像 hotzone env 范式·非 try/finally）。
STRUCT_REF_CONTENT_HASH_MODE = _flag("ZERO_AI_STRUCT_REF_CONTENT_HASH_MODE", False)

# ---- 对话止血①（2026-07-18）：attach_role_seq/attach_token_seq 幂等守卫（兑现 round8b defer·见上 :412-413 真根因） ----
# gates.py:404-413 已诊断：attach_* 裸 backend.insert 无 first-write-wins -> 跨 item 同 struct_ref（content hash 撞）
# 反复 insert -> def_array 累积 ~16× -> read_role_seq 返全长 -> generate fill loop 产千词瀑布（冒烟 130段×935词）。
# STRUCT_REF_CONTENT_HASH_MODE（上）挡跨 item 跨句碰撞主因（~960×）·**同句/跨轮 re-observe 仍累积 ~16× 残留**（baseline）。
# 本 gate 兑现 round8b defer：attach_* 体内 first-write-wins per-row（backend.count 查全5列存在性·已存在 skip）。
# gate OFF = 裸 insert（行为同今·bit-identical 守 CI）/ ON = first-write-wins（冒烟/生产·解 ~16× 残留累积）。
# 镜像 STRUCT_REF_CONTENT_HASH_MODE env 范式（非 try/finally）。非零回归：formal_train 跨 episode re-observe 同 raw
# 场景 def_array 行数 16x->1x·generate parts.words 长度大降·J4word/output_word_ratio reward 信号变化（预期·止血目的）。
# 详见 doc/重来_对话止血_词瀑布降级_设计_2026-07-18.md。
ATTACH_SEQ_IDEMPOTENT_MODE = _flag("ZERO_AI_ATTACH_SEQ_IDEMPOTENT_MODE", False)

# ---- 对话止血②（2026-07-18）：generate fill loop 段内词数硬上限（①未覆盖路径兜底） ----
# generate.py:174 fill loop `for slot_idx, role in enumerate(role_seq)` 无段内长度上限 -> role_seq 累积时产千词瀑布。
# ①修幂等后 role_seq ≈ token_seq（实测 corpus_zh 单段 token max=120/p99=107/avg=69）·正常语料本 gate 不触发。
# 保留作①未覆盖路径（同 struct_ref 异 role_seq 长度变化·read 返 max 合并集）的产出上界保险。MAX_WORDS_PER_PART=256
# （generate.py 常量·max×2 真余量）。gate OFF = 无上限（行为同今·bit-identical）/ ON = 256 截断（冒烟/生产）。
# 截断触发时 OutputResult.truncated_units 记 unit（独立字段·不污染 lineage 语义）。详见止血设计档。
OUTPUT_LEN_CAP_MODE = _flag("ZERO_AI_OUTPUT_LEN_CAP_MODE", False)

# Mode B POST-weaning 异算法 cross-verify（#479 加强腿·统计学一致非 correctness）。学树（observe 建）+
# 参树（二次独立建·异 shape·R6 真守）→ cross_verify_pair 双路 execute_composes_value + rational.eq → reward = 1 iff all_agree。
# **模态对称**（§施工序 1.2）：ARITH arith_source_b / build_composes_from_arith · CODE code_source_b / build_composes_from_source·
# execute_composes_value + rational.eq 模态无关（两域都返 Rational）·corpus-agnostic。probes = spec.input_args（丢 expected·避教师 oracle）。
# reward 进 Episode metrics（verify propagate no-op·不落边 strength·不写 op_confidence）。default OFF·守回归
# （OFF + 参树 source_b 默认 None = POST 路径 reward=0 与现状 bit-identical·既有 CollectedItem 全无 arith_source_b/code_source_b）。
# 用户哲学重定向（2026-07-06）：不追求 correctness 真墙（Rice·须墙外 #478/#493）·只求统计学内一致——
# cross-verify 是 on-target 的统计一致性机制非妥协（agreement 非 identity·声称守）。
MODE_B_CROSS_VERIFY_MODE = _flag("ZERO_AI_MODE_B_CROSS_VERIFY_MODE", False)

# #732 G5-C 记忆项延迟晋升闸（memory_item consolidate gating·§十三:1108/1120 决断4）。
# G5-C = 记忆项 SHADOW/PENDING→PRIMARY 离线延迟晋升·数据源 memory_item SEG_EPISODIC 比率门
# sum(success_count)*1000//sum(count) ≥ θ_rw（caller 侧 by info_ref 聚合·#732 方案 d）。
# **三个 G5 同名不同物**：G5-A judge.py:167 自证机门因子（已 live·§十四:1245·本 gate 不碰）/ G5-B promote.py:_reward_ok
# 边级 promote（已 live·edge sn/tn·命名借用）/ G5-C 本 gate（记忆项 status flip·memory_item SEG_EPISODIC）。
# default OFF·守回归（OFF = consolidate 不 flip·memory_item count/sc 由 record_use 接线累加但 consolidate caller 不触发·
# 既有测试/formal_train bit-identical·test_stage11 直接调 reward_propagate 不经 _promote_eligible）。
# ON = STAGE4_PROMOTE_WEAN 末 _promote_eligible 后扫 memory_item by info_ref·G5-C 闸判达 → consolidate flip。
G5_C_CONSOLIDATE_MODE = _flag("ZERO_AI_G5_C_CONSOLIDATE_MODE", False)

# 维度桥（P1·G-PR2/3·§五 candidate 1 lightweight·composites_attr on __seg_*→skeleton_ref）。
# 翻 #965 monolithic defer（P1 增量落地 writer+reader 基建·**非"桥已活"**·value-transit consumer 断桥 P2 #1053 仍 defer）·gate-per-feature + default OFF + bit-identical。
# ★ COMPOSES 已生产 try/finally flip（formal_train·桥写侧激活·完成桥 #1167·让造句真用学到的 D:11 解"白学"·翻 Phase A.3 REJECT·
# doc/重来_对应机制生产激活_2026-07-17·2 对抗审 APPROVE-WITH-CONDITIONS 确认非 theater·§4.0 用户原则：底子合法+学习真+泛化）。
# DIM_BRIDGE_READ_MODE 仍 dormant 不 flip（P2 断桥 consumer stub·无 consumer·记 last_dim_skeleton 无效）。
# item-identity 映射（非 content-hash·code-verified·见 memory p1-dim-bridge-impl-derisk）。
# COMPOSES_COMBINE_MODE：observe 建 EDGE_INSTANTIATES on __seg_ struct_ref（writer·§十三-bis A.1）+ discovery 建 item→skeleton map·gate OFF→无 INSTANTIATES 边→generate.py:154 走空分支→bit-identical（FC12 直守）。
# DIM_BRIDGE_READ_MODE：generate 读 EDGE_INSTANTIATES on unit（reader·审1 MEDIUM-1 修：读 slot.ref 错位·DISPATCH_TOKEN_CHAIN_MODE ON 时 slot.ref=token 无边·边在 unit=struct_ref）。记 workmem.last_dim_skeleton（P2 断桥 consumer stub·P1 write-only 无消费者·非 observability 信号）。
# VALUE_TRANSIT_MODE（memory_space 写合成值·producer 侧）**defer 未加**（无 producer 写→dead gate 守 P3 纪律·断桥+相1 spec 后加）。
COMPOSES_COMBINE_MODE = _flag("ZERO_AI_COMPOSES_COMBINE_MODE", False)
DIM_BRIDGE_READ_MODE = _flag("ZERO_AI_DIM_BRIDGE_READ_MODE", False)
# Phase B 种子床扩（§十四-bis·mereology part-of 预序闭包 + PURE_ALIAS 等价闭包·self-gate at builder entry）：
# MEREOLOGY_CLOSURE_MODE：build_mereology_ancestor_map_external + whole_of reader（part-of 预序闭包·镜像 IS_A external）。
# PURE_ALIAS_CLOSURE_MODE：build_pure_alias_closure_external（PURE_ALIAS 等价类·transitive_closure 首个 live caller）。
# **dormant**：default OFF·self-gate（builder 入口 `if not gate: return {}`）·无生产 caller（基建·consumer 落 Phase D/E/F）·
#   bit-identical（gate OFF→零闭包→逐字现状）。诚实 seed-bed algebra（确定性派生·零学习内容·学习验 floor Phase F）。
MEREOLOGY_CLOSURE_MODE = _flag("ZERO_AI_MEREOLOGY_CLOSURE_MODE", False)
PURE_ALIAS_CLOSURE_MODE = _flag("ZERO_AI_PURE_ALIAS_CLOSURE_MODE", False)
# Phase C 二阶相似读侧（§十五-bis·shared-neighbor Jaccard·read-side 不存边·dispatch tiebreak consumer）：
# SIMILAR_SECOND_ORDER_MODE：_second_order_bonus(graph,c,ctx_refs)（per-candidate max·mirror _pronoun_bonus）并入 sp 维联合 _cap_sp。
# **read-side only**（不存 EDGE_SIMILAR·饱和=theater·词库类不可学）·确定性 computation over observed COOCCURS（非"学得"·同 collide_score）。
# default OFF·bit-identical（gate OFF→inline `if _so_gate else 0`→零分变→逐字现状）。对应 bootstrap = Phase D/E（非本 gate）。
SIMILAR_SECOND_ORDER_MODE = _flag("ZERO_AI_SIMILAR_SECOND_ORDER_MODE", False)
# Phase D §十六-bis D.1：REALIZES 边 writer（skeleton→__REL_SUBSET__·option-b oracle-pair-match labeled bed）。
# **外源 labeled bed**（forming-sample token-pair 命中外源 EDGE_IS_A·oracle 定 IS_A 非读 Cue·禁 pairs→渲染 Cue 泄漏）。
# default OFF·self-gate at writer（mirror Phase B self-gate）·bit-identical（gate OFF→零 REALIZES 边）。
# consumer Phase E·学习 claim 严禁前置·验 floor Phase F·D.1 ship ≠ Phase D done（D.2/D.3 frontier）。
REALIZES_MODE = _flag("ZERO_AI_REALIZES_MODE", False)
# Phase E §十八 condition 6a：cue 子聚类（skeleton 发现期按闭类 cue 分桶·route+auto_discover 双层·镜像 _cluster_by_lca）。
# **cue-保留结构类**（同 (sig,arity,abstract_sig) 异闭类 cue [苹果,是,甜] vs [糖,使,甜]·same-shape-same-LCA·唯 cue 区分→异名异骨架·破 cue 坍缩·REALIZES 可异标 IS_A/CAUSES）。
# sustainable-split（exposure-driven·无 frozenset·反 theater 命门）：某位按 token 划分得 ≥2 组各 ≥K → cue 位（闭类重复可持续·内容词各 1 不可持续→不拆·天然区分闭类 vs 开类）。
# **关系 label 走外源 oracle 非读 cue**（label_realizes·§十八 condition 6 复合键 (primitive-pattern,arity,position-dominance)·禁单 primitive 单射 relation·anti-theater）。
# default OFF·bit-identical（gate OFF→_cluster_by_cue 返 [(roots,())]→cue_sig=()→_shape_name 不加 payload→名同今）·
# 诚实 scope：6a cue-in-名+routing（§十八 partial·非 cue 作结构节点 model-reversal 6b defer）·consumer Phase E/F（REALIZES 异标 + floor P5 多关系 held-out）。
CUE_CLUSTER_MODE = _flag("ZERO_AI_CUE_CLUSTER_MODE", False)
# 对应泛化 v2（doc/重来_对应泛化_结构反推_学全_2026-07-17）：结构反推 tally → D:11 promote 结构匹配轨。
# 新词 W 落 REALIZES-R-skeleton cue slot（cue-blind tally·审2条件1 三路分离）→ tally (W,R) distinct forming-sample
# ≥K → 首次建 D:11 SHADOW（审2条件2·generator 关后唯一创建者）→ promote W→R D:11 PRIMARY（_structure_match_ok
# 唯一证据轨·审2 BLOCKER 1 删 ∨·experience/teacher 退场）。**非循环**（R 来自 REALIZES oracle·source==CONCEPTNET·非 cue）。
# **两 gate 共存（审2条件3）**：ON→结构匹配轨替代 generator+experience（formal_train 跳 _run_emergence_hook·
# promote D:11 只认 _structure_match_ok）；OFF→既有 generator+experience+teacher 照旧（knife4 旧路径回归守）。
# default OFF·bit-identical（gate OFF→_structure_match_ok 不被调·tally_cue_slot_matches 不调→零 D:11 翻→逐字现状）。
# 生产 try/finally 与 REALIZES_MODE + CUE_CLUSTER_MODE 同翻（三者共构结构反推·见 formal_train flip 块）。
ORACLE_PROMOTE_MODE = _flag("ZERO_AI_ORACLE_PROMOTE_MODE", False)
# 对应泛化 readback→generation 桥（doc/重来_对应泛化_readback_generation_桥_2026-07-17）：dispatch_slot 第 8 路
# correspondence bonus·让 v2 学到的 cue 词（D:11 W→REL_*·tally→promote）流入生成（学到的对应只识别不产出=白学命门）。
# **两跳 read**（unit→INSTANTIATES→skeleton→REALIZES→REL_*·审2 致命-1 修·挂 skeleton 非 unit·struct_ref 无 REALIZES 边）+
# read_cue_sig（skeleton PARAM slot ATTR_CUE_SIG 重建·runtime length-guard 守 slot 对齐）+ cue_rel_of（D:11
# source==SOURCE_BARE_TEXT AND tier==TIER_PRIMARY·审1 CONDITION A·**非 boot 种子 SOURCE_TEACHER**·禁复用 lookup_word_concept 只 tier 滤）。
# **(β) 独立轴**（CORR_BONUS=SCORE_SCALE+1=1001·不入 _cap_sp·审1 CONDITION B）：held-out cue 词(collide=0) 严格胜 collide=1(1000)·
# 输 collide=2+(2000+)守共现主轴·floor 可重复测稳定激活。**cue-slot-aware**（审1 CONDITION D·审2 证 ATTR_CUE_SIG 天然单 cue 位·
# _cluster_by_cue 单拆硬约束·bonus 不可能多 slot 触发）：bonus 仅 cue slot·非 cue slot 走 collide·反 theater。
# **非自证**（审2 核证 cognition/result/ 零 D:11 写点·生成纯 read·写点全在 tally/promote 阶段外）。
# default OFF·bit-identical（gate OFF→generate 不读 stash·workmem 三字段 default 守·dispatch_slot _corr_gate=False 不进分支→逐字现状）。
# 生产 try/finally 与 REALIZES_MODE + CUE_CLUSTER_MODE + ORACLE_PROMOTE_MODE 四 gate 共翻（学全机制+生成消费共构）。
CORRESPONDENCE_SLOT_MODE = _flag("ZERO_AI_CORRESPONDENCE_SLOT_MODE", False)
# 命门③ 候选 B：cue 位结构活化。无 learned relation cue 时直出骨架 cue；有 PRIMARY+BARE_TEXT D:11
# cue 时，骨架 cue 作为 fallback 与 learned cue 同池竞争，correspondence bonus 可真实改变 winner。
# **方法论纠偏契合**（用户）：功能词由 cue 聚簇（_cluster_by_cue sustainable-split）涌现（闭类 是/使 重复≥K 可持续拆 vs
# 开类内容词<K 不可持续拆·天然区分·无须词表·守 §十五 C5 禁硬编码功能词表）·主谓宾作 cue 位+内容词位涌现·非硬编码语法角色。
# workmem.current_cue_sig（tuple·None 占位非 cue 位）+ current_slot_idx（dispatch_slot 无 slot_idx 参数·走 workmem·审2 HIGH-1）
# 由 generate CORRESPONDENCE_SLOT_MODE 块 4 case 全路径设（无 stale state·审1 MED-1）。cue token 不入 emitted_tokens/
# produced_refs（generate.py:203 src!=CUE_SLOT_FILL 守·审2 HIGH-2·它是结构活化非内容词·不入 collide ctx 保信号质量）。
# **6 gate 同翻缺一链断**（审1 MED-4）：CUE_CLUSTER_MODE（写 cue_sig）+ COMPOSES_COMBINE_MODE（写 INSTANTIATES->read 非 None）
# + REALIZES_MODE（skeleton->REL_*->rel_kind!=0）+ CORRESPONDENCE_SLOT_MODE（stash current_cue_slots/cue_sig/slot_idx）
# + ORDINAL_SURFACE_MODE（surface_of）+ CUE_SLOT_FILL_MODE（fallback/lineage）。缺关系候选时仍稳定直出骨架 cue。
# default OFF·bit-identical（gate OFF->不注入骨架 cue fallback，不改 lineage，走既有候选链）。
# 生产 try/finally 与上述 5 gate 共翻（同 cue-slot-aware 家族·学全机制+生成消费共构）。
CUE_SLOT_FILL_MODE = _flag("ZERO_AI_CUE_SLOT_FILL_MODE", False)
# 命门③ 候选 C（doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）：slot_lca 抽象约束·内容词活化。
# dispatch_slot 内容词位（非 cue 位）按 slot 的 IS_A LCA 类过滤候选（c IS_A slot_lca 留·reflexive-transitive·
# c==slot_lca or slot_lca in ancestors(c)·非仅 nearest==slot_lca·c 可深 slot_lca 下多层·nearest 漏合法候选·design 决策）。
# slot_lca 由 _cluster_by_lca（structure_discover.py·set_lca 对簇内 slot 全 token 取最具体共同祖先）写 ATTR_SLOT_ROLE=9
# （CONCEPT_LEAF fresh 节点·与 ATTR_OPERAND 同节点第二 attr·absence=无类约束）·**已写盘零消费者**·候选 C = 加读侧+生成消费。
# **方法论纠偏契合**（用户）：抽象活化·内容词按学来的抽象类约束选词·非硬编码语法角色·主谓宾作下层涌现
# （abstraction.py:22-24"抽象层绝不用 role_seq/位置桶·须走 IS_A LCA 上卷"）。slot_lca 由 set_lca 涌现非词表（守 §十五 C5）。
# workmem.current_slot_lcas（tuple·()=无 INSTANTIATES/length-guard fail·ConceptRef|None 占位无约束位）+ current_slot_lca（tuple·()=无约束·mirror last_dim_skeleton 范式）
# 由 generate 独立 SLOT_LCA_CONSTRAINT_MODE 块 4 case 全路径设（无 stale state·审1 MED-1·非嵌 CORRESPONDENCE_SLOT_MODE·slot_lca 适用任意 INSTANTIATES 骨架非仅 R-skeleton·rel_kind!=0 非前置·独立于 cue 链）。
# **独立 gate 链**（design·审2 核）：COMPOSES_COMBINE_MODE（写 INSTANTIATES 边->read 非 None）+ SLOT_LCA_CONSTRAINT_MODE（读+过滤）= 2 gate·
# 不依赖 REALIZES/CUE_CLUSTER/CORRESPONDENCE_SLOT/CUE_SLOT_FILL/ORDINAL_SURFACE（ATTR_SLOT_ROLE 由 _cluster_by_lca 写非 CUE_CLUSTER 门）。
# **空集 fallback**（mirror hub filter `if cand_f:`）：slot_lca 类约束下候选空（未见过该类 token）-> 退化走 collide 不阻断生成。
# **混合 case 诚实边界**：CUE_SLOT_FILL_MODE OFF + SLOT_LCA_CONSTRAINT_MODE ON 时 cue 位仍走旧 LCA filter；
# 生产 B+C 共翻时 cue 候选竞争显式绕过内容词 LCA，避免功能词被错误排除。
# default OFF·bit-identical（gate OFF->dispatch 不进过滤分支->candidates 不变->走 collide 返 LINEAGE_CONCEPT_FILL=1·逐字现状·getattr 默认守 minimal workmem 不崩）。
# 生产 try/finally 与 B 6 gate 共翻（7 gate = B 6 + C 1·COMPOSES_COMBINE 共享不重计·C 独立 2 gate 也可活·生产 B+C 共翻最完整）。
SLOT_LCA_CONSTRAINT_MODE = _flag("ZERO_AI_SLOT_LCA_CONSTRAINT_MODE", False)
# floor 端到端下游激活率（断奶 critical path 第 2 件·反 theater 首版机制层预验·doc/重来_floor_端到端下游激活率_2026-07-17）：
# formal_train 生产 _measure_floor_pass orchestrator（observe〔probe_corpus held-out〕+auto_discover_operators〔**不调 tally**
# ·tally 仅训练路径 :3632 caller·held-out 测量侧禁污染 D:11〕+build _floor_path+generate_output〔gate ON〕）→ 纯读
# _measure_floor_activation（cognition/result·读侧重导 cue slot 激活率 + false-positive 臂 + measured-guard）。
# **学到的 cue 词（D:11 W→REL_*·tally→promote·桥 P3 新信号）在 held-out R-skeleton cue slot 正确激活率 ≥ LANG_FLOOR_ACTIVATION=500**。
# 镜像 W6 _run_simulated_offline_eval 隔离范式（扩展到语言域·W6 arith-only guard 不动·floor 独立实现 observe+discover）。
# **反 theater 终判心脏**：测真端到端 cue↔rel 下游激活（替 reward-ratio 假腿）·C-vs-L 真判别（闭包-only C 无 D:11→cue_rel_of=0→不激活）。
# default OFF·bit-identical（gate OFF→orchestrator 不跑·floor 三参数 default False/0/0·weaning.py anchor_pf floor_conjunct
#  `if FLOOR_ACTIVATION_MODE else True`→恒 True→anchor_pf 退既有→SW2/9/16 逐字过·审1 严重-1 守）。
# **env-gated**（同 STATISTICAL_WEANING_MODE·getattr 读·**不**在 formal_train try/finally 翻·它是 eval/measurement gate 非核心训练变换）：
# 生产 orchestrator `_measure_floor_pass`（observe〔probe_corpus held-out〕+auto_discover〔不调 tally〕+generate→measure_floor_activation）
# **defer 课程相位 piece 3**（须真 held-out split + 真 curriculum run 才能 e2e 验·设计档 §9）。piece 2 = measure 机制 + gate-gated verdict + FC1-8 fixture 预验。
# ★ **目前 inert**（post-impl 审 MEDIUM-1/2）：formal_train statistical 路径**未传 floor 三参数**（orchestrator defer piece 3）→
# gate ON 不带参→floor_measured=False→statistical_ready **silent veto**（非 measurement 失败·foot-gun）。piece 3 orchestrator
# 落地前须保持 OFF（env default OFF 守）。measure 机制本身已 FC1-8 fixture 预验（C-vs-L 判别/measured-guard/false-positive）。
FLOOR_ACTIVATION_MODE = _flag("ZERO_AI_FLOOR_ACTIVATION_MODE", False)
# 断桥 Phase A（P2 G-PR2/3 cross-path·doc/重来_断桥设计refinement_2026-07-15）：CollectedItem.action_specs（教师标
# action→CodeSpec）→ _run_task_driven_generate 调 synthesize_value（相1 消费者）→ 独立 task-driven episode。
# **weaning-safe 决断 A**：独立 episode·**不替换 vm_proof verify round**（不碎 W7·反 VALUE_SYNTHESIZE 翻 ON 教训）。
# **dormant**：default OFF·不在生产 try/finally flip（无教师 action_specs 数据时 inert·同维度桥范式）·Phase B 动态构造器后翻。
ACTION_BRIDGE_MODE = _flag("ZERO_AI_ACTION_BRIDGE_MODE", False)
# 断桥 Phase B 片1（P2 G-PR2/3 动态构造器·doc/重来_断桥设计refinement_2026-07-15 §Phase B 片1）：无教师标 action_specs
# 时·从 language text cues 动态构造 spec（numeric_claims 4-tuple `(left,op,right,result)` → CodeSpec 隐 op·
# input_args=(left,right)·expected=(result,1)·**op 隐藏**=synthesize 找算子非刀B 验算子·真合成）→ synthesize_value
# 联合匹配 → 独立 task-driven episode。input source = text cues（非 teacher·非 held-out·非硬编码·解 Phase B
# "撞 held-out"之谜：held-out 仅泛化验证须·构造+合成 NOW）。刀B 路由无冲突（刀B reward round 验证轴·断桥
# generate stage 合成轴·两 stage 分离·同 numeric item 得两 episode 不同轴）。数据桥 CollectedItem.numeric_claims_flat
# （observe 期 flatten seg.numeric_claims·mirror code_struct_ref·ungated 纯缓存·NUMERIC_PROOF_MODE OFF→flat 空）。
# **weaning-safe 决断 A**：独立 task-driven episode·不替换 vm_proof verify round·不碎 W7（同 Phase A）。
# **dormant**：default OFF·不在生产 try/finally flip（无 numeric_claims 时 inert·同 ACTION_BRIDGE_MODE 范式）·
# 片2 intent→spec 过滤 + dispatch 桥 + 骨架管线 + op_confidence 接线 defer。bit-identical（gate OFF + flat 空→不进）。
ACTION_BRIDGE_CUE_MODE = _flag("ZERO_AI_ACTION_BRIDGE_CUE_MODE", False)

# 符号数学能力扩展 Phase 3（doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis.7）：
# formal_train task-driven episode 集成符号变换规则（register 教师陈述规则 + apply held-out + cross-verify 执行等价
# → 独立 episode·weaning-safe 决断 A·不替换 vm_proof verify round·不碎 W7·同断桥 Phase A/B 范式）。
# **dormant**：default OFF·不在生产 try/finally flip（无 transform_specs 时 inert·同 ACTION_BRIDGE_* 范式）。
# 机制完整（Phase 1+2+2b·d/dx 可学）·本 gate 接生产训练接线。bit-identical（gate OFF + specs 空→不进）。
SYMBOLIC_TRANSFORM_MODE = _flag("ZERO_AI_SYMBOLIC_TRANSFORM_MODE", False)

# S8 符号间运算关联 Phase 1（doc/重来_S8符号间关联机制设计_2026-07-15 §七）：formal_train task-driven
# episode 集成运算间逆关系（register 教师陈述 rule_a/rule_b + register_inverse_relation + verify_inverse_relation
# B∘A 还原 @ 采样 → 独立 episode·weaning-safe 决断 A·不替换 vm_proof verify round·不碎 W7·同 SYMBOLIC_TRANSFORM_MODE 范式）。
# **逻辑依赖 SYMBOLIC_TRANSFORM_MODE**（逆关系规则 = 变换规则·机制建在 symbolic_transform 之上·两 gate 同 ON 才完整活·
# 镜像 INTENT_COMMAND_MODE 依赖 M1_INTENT_CLASSIFY_MODE 范式·非硬耦合·各 block 独立读各 specs）。
# **dormant**：default OFF·不在生产 try/finally flip（无 inverse_relation_specs 时 inert·同 SYMBOLIC_TRANSFORM_MODE/
# ACTION_BRIDGE_* 范式）·S8 课程续起时与 SYMBOLIC_TRANSFORM_MODE 同 flip。bit-identical（gate OFF + specs 空→不进）。
# Phase 1 = INVERSE only（链式法则 COMPOSITION defer Phase 2·恒等 IDENTITY 折化简规则）。
SYMBOLIC_RELATION_MODE = _flag("ZERO_AI_SYMBOLIC_RELATION_MODE", False)

# M1片2 intent 分类（替换 formal_train.py:366,1448 两处 IntentType(INTENT_QUESTION) 硬编码·
# doc/重来_M1片2_intent分类设计_2026-07-08.md）。cognition/understanding/intent_classify.classify_intent
# 派生 transient IntentType：is_causal_reasoning = 输入 segments 含因果对（与 causes.py 建边同源·解
# judge.py:224 G3a 死门·j3path 从永 0 到加权）·type/sink/其余 bool 维持现状（反 theater 收窄·M1 不动 type）。
# **type=COMMAND 由子 gate INTENT_COMMAND_MODE 控制**（W7·doc §15·见下）·M1 本身只控 classify_intent
# 是否被调（OFF=硬编码 INTENT_QUESTION + 三 bool 永 False·bit-identical）。
# **依赖 CUE_EXTRACTOR_MODE**（A1 #480 done·裸文本自产 CAUSES 来源②·非钥匙①）·is_causal 消费 EDGE_CAUSES。
# default OFF·守 CI 回归（OFF = 两处硬编码 IntentType(INTENT_QUESTION) + 三 bool 永 False·G3a/G3b dead·等同现状）。
# **★生产训练 try/finally 翻 ON**（镜像 ATTRACTOR/EXPLORATION/EMERGENT/CUE/SELECTION_PREF/MEMORY_REPLAY 六 gate·
# formal_train.py:1235 saved/flip + :1414 finally reset·否则 classify_intent 永不调 = is_causal 永假 =
# reward 退化核心病灶未修 = theater·审1 P0）。
M1_INTENT_CLASSIFY_MODE = _flag("ZERO_AI_M1_INTENT_CLASSIFY_MODE", False)

# W7+B-PR1 命令判定 type 分流（doc §16·真生成链路路由前置）。
# M1_INTENT_CLASSIFY_MODE 的子 gate：M1 控 classify_intent 是否被调·本 gate 控 type 是否可变 COMMAND。
# cognition/understanding/intent_classify._has_action_intent 扫 segment.tokens·命中动作意图词
# （命令词 帮我/请→COMMAND_MOOD OR 动作词 生成/计算→ACTION_*·doc §16.4·命令词 OR 动作词）→ type=INTENT_COMMAND。
# **走 D:11 二源**（doc §16.1 推翻 §15.1 纠正③：命令 mood 概念先天·命令词/动作词 alias 开放·D:11 学·
# 镜像 operator/modal/negation #940 范式·gate ACTION_D11_READBACK_MODE 控 readback 第二源）。
# 三态：M1 OFF/× → classify 不调 bit-identical；M1 ON/OFF → classify 跑 type 永 QUESTION（=当前 M1-ON）；
# M1 ON/ON → type 可变 COMMAND。
# default OFF·守 CI（OFF → _has_action_intent 返 False → type=INTENT_QUESTION·dag_path:302 早已 tuple 含
# COMMAND 零终止态差异·judge/dead_end 不读 type·bit-identical）。**生产 try/finally 暂不翻**（W7+B-PR1 无下游消费者·
# B-PR2~4 未实施·独立 gate 留 B-PR 落地增量验证控制点）。
INTENT_COMMAND_MODE = _flag("ZERO_AI_INTENT_COMMAND_MODE", False)

# 归一化半 A 功能词/hub 排除（read-time hub_degree ≥ θ 过滤 3 live 消费/污染点·
# doc/重来_归一化与功能词排除_设计_2026-07-08.md·对抗挖修正 §二序2·共现/跳跃机制生死前置）。
# cognition/shared/hub_detect.is_hub 读 COOCCURS 关联边总数（from+to 双向·绝对计数·无除法无 crash）。
# 3 点 read-time 排除（非 write-time·保 COOCCURS 数据避自举死锁 + 不碰 build_cooccurs/build_precedes）：
#   slot_dispatch collide_score caller（activate_candidates 后过滤 + ctx_refs 过滤·解"分子是曾经"排序污染）+
#   emergent_relation_signal._cooccurs_count（a/b hub→返 0·解伪产 REL_CAUSES 喂 reward 污染因果学习）+
#   refers_occurrence 候选生成（解"他"→"曾经"代词语义层污染）。
# **判据 hub_degree 是过渡统计判据·非 §十五 FUNCTION_CLASS(12) 学习边+reward 真·涌现形式**（θ oracle 软阈值·
# 解审2 铁律7 effective_freq "纯涌现"过称）·与 reward 动态无关（解审1 Q2 e_tn 时漂）·catch 高频实词 hub
# （解 Q4 freq/permille 只挡 closed-class 洞）·绝对计数无除法（解 Q7 corpus_total crash）。fresh compute
# 无 memo（避 stale-cache·COOCCURS 单调增）。
# default OFF·守 CI 回归（OFF = 3 点不过滤 = 现状 bit-identical 行为级·非 AST 级）。**★生产训练 try/finally
# 翻 ON**（镜像 M1/EMERGENT 范式·formal_train try + finally 复位·否则 gate 永不活→3 点不过滤=theater）。
# 半 B 共现比 lift 实施 defer 到统计深化 B session（无 live 消费者·预建=死码 theater·决断 B2）。
EXCLUDE_FUNCTION_MODE = _flag("ZERO_AI_EXCLUDE_FUNCTION_MODE", False)

# #478 STRUCT_BIND 跨模态槽位级绑定 boot 种边（来源 a 教师标注·§8.7-P2·决断 4 + 决断 8）。
# cognition/process/struct_bind.bootstrap_struct_bind_edges boot caller（formal_train lang discover 后·两模态
# skeleton ref 就位）经 resolve_struct_bind_pairs 读 ZERO_AI_LOCAL_DIR/struct_bind_pairs.txt → discovered_operators
# 索引 name→skeleton_ref + collect_skeleton_slot_refs 解析 slot ref → 建 EDGE_STRUCT_BIND 槽位级边。
# **boot-side gate-controlled deferred activation**（首例 boot-side gate·IS_A/sense/word_concept boot 无 gate 因
# lang-keyed + 空 pairs 短路已足·STRUCT_BIND 跨模态非 lang-keyed 单文件·gate 为显式 bit-identical 开关）。
# default OFF·守回归（OFF = boot caller 完全跳过·不调 resolve/bootstrap·CI===生产 default bit-identical·既有测零行为不变）。
# ON + 无文件 → resolve 返 [] → bootstrap 空 pairs 短路（双层守·gate + 空 pairs）。
# **不在生产入口 try/finally 翻 ON**（无真实教师 corpus·#731 defer·翻 ON 仅 no-op）·#731 真实语料就绪后翻 ON。
# 反 theater（决断 7）：本 gate OFF 不代表 theater——builder 真（test 验）·gate 控制 activate 时机·同 MODE_B 范式。
# 消费侧 reader 落 #730（generate 读 STRUCT_BIND 填语言槽·反 theater 锚闭环·解⑤长代码 NE）·#478 落地瞬态=形态 2 theater。
STRUCT_BIND_BOOT_MODE = _flag("ZERO_AI_STRUCT_BIND_BOOT_MODE", False)

# #730 长代码 generate 路径 W（代码模态 task-driven unparse COMPOSES→源码串·Mode A 构造性·解⑤ NE）。
# formal_train._run_task_driven_generate 代码模态分派：读 item.code_struct_ref（observe 期建 __prog_* 根·候选 A）→
# unparse_composes 序化 → ast bodies_match normalize == code_source normalize → verified·打包 OutputResult
# （parts 非空⟺ verified·反 theater ③下游读者锚）。
# **路径 W 自洽闭环**（决断 7·非 theater）：序化器（cognition/result/composes_unparse·片1）+ 真消费者 task-driven
# 代码模态 episode。**路径 X（generate.py 字面 + STRUCT_BIND reader）defer 独立 session**（维度桥阻断·§〇 纠偏 2）·
# STRUCT_BIND 边 theater 形态 2 在 #730 后仍存在（路径 W 不读 STRUCT_BIND）。
# default OFF·守 CI 回归（OFF = task-driven 只走 arith·代码模态 episode 不激活·零行为变·既有测 bit-identical）。
# **★生产训练 try/finally 翻 ON**（formal_train task-driven 调用点·镜像 ATTRACTOR/M1/归一化 范式·否则代码模态 episode
# 永不激活 = 序化器孤儿 = theater·审 P0）。ON = 代码模态 episode 激活·⑤ Mode A 部分（代码模态 task-driven 重建率）升 PASS。
# **诚实边界**：⑤ generate 字面部分（STRUCT_BIND reader）设计性 NE（路径 X defer·须维度桥）·capability_exam ⑤
# status footnote 显式标（子决断 1-bis·Mode A PASS 仅覆盖代码模态 task-driven 重建·非源码 generate）。
CODE_UNPARSE_MODE = _flag("ZERO_AI_CODE_UNPARSE_MODE", False)

# 刀 A 时序 cue verify-driven episode（语言域第一个 LIVE form_proof_fn·构造性检查层·形式 cue 扩展首刀）。
# formal_train.run_round_full 语言域路由分支（_is_verify_modality 之后）：语言域 + 任段 precedes_pairs 非空 +
# 本 gate ON → 走 occurrence-order adapter（验 cue 图 Kahn 无环·非事件时间、因果或 COMPOSES 执行）·
# 直调 time_seq_proof_fn 绕 judge（语言域 G5=DEAD_DESIGN·_ARITH_DOMAINS 门挡·非挂 G5）·reward=1 iff DAG 无环·
# 不落 strength（verify propagate no-op·镜像 :457）·**永不接 reward**（PRECEDES strength 恒 1·reward CAUSES-only）。
# **诚实边界**：构造性检查层（Kahn 验 DAG 无环·确定性）·非构造性验证（cue 对 + token 序 PRECEDES 均 single-source·
# 须 R6 独立源升验证·Layer0 下 session）。时序 cue 边不入图（Option A·闭包传·防 #355 provenance 冲突 + emergence 污染）·
# 持久化 defer Layer0 session。stable≠correct（DAG 无环≠语义时序正确·#479 墙）。
# default OFF·守 CI 回归（OFF = 路由不走 → 既有语言域 episode_loop 路径不变 bit-identical）。
# **★生产训练 try/finally 翻 ON**（镜像 COOCCURS_DEDUP_MODE 范式·formal_train try + finally 复位·
# 否则路由永不走 = 时序验序器孤儿 = theater）。
TIME_SEQ_PROOF_MODE = _flag("ZERO_AI_TIME_SEQ_PROOF_MODE", False)

# ---- 刀 B：语言域数值等式 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 A 时序） ----
# formal_train.run_round_full 语言域路由分支（_is_verify_modality + 时序分支之后·numeric priority over precedes）：
# 语言域 + 任段 numeric_claims 非空 + 本 gate ON → 走 _run_numeric_verify_round（镜像 occurrence-order adapter·
# 但验数值等式声明算术一致·直接整数算术·非 PRECEDES DAG·非 COMPOSES 执行）·直调 numeric_proof_fn 绕 judge·
# reward=1 iff 全声明算术一致·不落 strength（verify propagate no-op）·**永不接 reward**（数值声明不入图）。
# **诚实边界**：构造性检查层（整数算术 +,-,× 确定性）·非构造性验证（左式/右式数均 single-source·来自文本 cue 锚·
# 须 R6 独立源升验证·Layer0 标 SELF_PRODUCED·全自产不准停）。数值声明不入图（Option A·闭包传·同刀 A 防污染）。
# 为何直接整数算术非 execute_composes_value：平坦表达式直接算术即可·建 COMPOSES 树反图污染+无验证增益（详 numeric_proof.py）。
# stable≠correct（算术一致≠命题真·#479 墙·文本是否真陈述此算式是语义层）。
# default OFF·守 CI 回归（OFF = 路由不走 → 既有路径不变 bit-identical）。
# **★生产训练 try/finally 翻 ON**（镜像 TIME_SEQ_PROOF_MODE 范式·formal_train try + finally 复位·否则路由永不走 = 孤儿 = theater）。
NUMERIC_PROOF_MODE = _flag("ZERO_AI_NUMERIC_PROOF_MODE", False)

# ---- 刀 C：语言域全称量化 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 A/B） ----
# formal_train.run_round_full 语言域路由分支（numeric 分支后·precedes 前·numeric>universal>precedes 序）：
# 语言域 + 任段 universal_claims 非空 + 本 gate ON → 走 _run_universal_verify_round（镜像 _run_time_seq/
# numeric_verify_round·但验全称量化内涵分类子集 X⊆Y·ConceptNet 外部祖先图）·直调 universal_proof_fn 绕 judge·
# reward=1 iff 全声明 verified（ConceptNet 外部断言 child⊆parent）·不落 strength（verify propagate no-op）。
# **★构造性验证层·首个 EXTERNAL**（刀 A/B 是构造性检查 SELF_PRODUCED·刀 C 升验证·Layer0 external_verified
# 首个语言域 episode 计入·可驱动停止决策）。IS_A 来源须 ConceptNet（build_isa_ancestor_map_external·
# source=SOURCE_CONCEPTNET+epistemic=EPI_STRUCTURED 双 filter）·**非 cue 自产**（否则 single-source theater）。
# **三值逻辑**：外部祖先路径支持→reward=1 EXTERNAL；缺路径→None。显式反证 adapter 未接线前不产生 reward=0。
# **诚实边界**：构造性验证≠truth（ConceptNet 可错·外部源对齐非命题真·#479 墙·stable≠correct）·
# 刀 C=G5b 窄子域（内涵分类子集）·属性全称子域 defer Mode B（详 doc/重来_刀C量化cue设计_2026-07-08.md §六b）。
# external 依赖 ConceptNet 本地文件（CI/生产 default 无→ext_map 空→全 can't-verify→无 reward·非 theater）。
# default OFF·守 CI 回归（OFF = 路由不走 → 既有路径不变 bit-identical）。
# **★生产训练 try/finally 翻 ON**（镜像 TIME_SEQ/NUMERIC 范式·formal_train try + finally 复位·否则孤儿 = theater）。
UNIVERSAL_PROOF_MODE = _flag("ZERO_AI_UNIVERSAL_PROOF_MODE", False)

# ---- A1·STEP6：语言域存在量化 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 C） ----
# formal_train.run_round_full 语言域路由分支（universal 分支后·numeric>comparison>universal>existential>precedes 序）：
# 语言域 + 任段 existential_claims 非空 + 本 gate ON → 走 _run_existential_verify_round（镜像
# _run_universal_verify_round·验证 A∩B 非空声明）并直调 existential_proof_fn 绕 judge。
# 正证需要共同 MEMBER witness、显式 overlap 或已知非空共同子类；反证需要显式 DISJOINT。
# 当前 caller 只提供 SUBSET_EQ 祖先图，尚无 typed Evidence adapter，因此必须返回 None 且不产 EXTERNAL episode。
# D6 closed-class frozenset {有的,有些} 只负责 cue 识别，不携带真值或证明。
# default OFF·守 CI 回归（OFF = 路由不走 → 既有路径不变 bit-identical）。
# **★生产训练 try/finally 翻 ON**（镜像 UNIVERSAL 范式·formal_train try + finally 复位·否则孤儿 = theater）。
EXISTENTIAL_PROOF_MODE = _flag("ZERO_AI_EXISTENTIAL_PROOF_MODE", False)

# ---- G1 reification + #774 PROPERTY（命题节点承载 subject/attr_type/value 三元·G3b 真消费者·解 fork 架构不一致） ----
# 设计 doc/重来_G1reification_774PROPERTY_设计_2026-07-09.md（fork 分析 §5.3 实施 ready·选 b 避 generate 改动）。
# 命题节点 = (subject, attr_type) 对的载体·ATTR_PROPOSITION=11 标记·确定性 surface __prop_{subj}_{attr}·
# concept_index.ensure 去重·value 走 EDGE_PROPERTY 出边（命题节点→value 概念·core space）。#774 builder 在命题节点
# 建 PROPERTY 边·G3b 改全局扫命题节点 PROPERTY 出边判同(subject,attr_type)多值=结构矛盾（fork §3.2/§3.3 无假矛盾）。
# **G3b 真消费者**（反 theater）：has_value_claim=True（property_claims 非空·gate ON 时）激活 G3b·命题节点真有 PROPERTY
# 出边（builder 建）·G3b 真扫真判·非空集永返 1（fork §四选项 B theater 反例）。
# OFF = observe 不建命题节点/PROPERTY 边·extract_property_claims_gated 返 []·intent has_value_claim 永 False·G3b 不激活·
# 既有行为零变 bit-identical（既有 PROPERTY 边在 metaphor/pronoun token 概念·非 struct_ref·G3b 旧 part.unit 扫永空=返1·
# 新全局扫命题节点 OFF 无命题节点=返1·两态同值）。
# ON = formal_train try/finally 翻 ON（生产训练）·命题节点建·G3b 激活全局扫。
# **诚实边界**：reification 给表达力非验证力（命题 truth=#479 墙·G3b 只判结构矛盾层a·语义真对立层b/c #479 truth 关切 W2·**非 W1 D 物理接地墙**·provisional/可废止对立 E3 覆写+E4 推理引擎可达·只 definitive truth 撞 #479·defer）·
# 命题节点不进 dag_path/structure_units（判断层载体非路径层载体·零 J1/J2/J3 扰动）·attr_type 必须有（具有/has 无 attr_type 模式 defer）。
PROPOSITION_MODE = _flag("ZERO_AI_PROPOSITION_MODE", False)

# ---- B1 否定 polarity（P0.3 pol 进 surface·B1 否定 cue 抽取填 pol=1·"X 的 Y 不是 Z" → polarity=1） ----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §七 B1。否定词（不/没/非/无 + not/no/never·is_negation_cue）
# 在"是"前（j-1）→ extract_property_claims 否定窗口（negation_on 参数·_gated 传本 gate）·pol=1 命题节点建独立
# surface 后缀 _1_0（P0.3）·G3b 各判多值·对立非矛盾（P0.4 利好）。
# **二级 gate**：PROPOSITION_MODE OFF 时整体不建命题（extract_property_claims_gated 返 []）·本 gate 仅守否定窗口。
# OFF = negation_on=False → 既有肯定窗口·"不是"走错位 skip（bit-identical·既有行为零变）。
# ON = formal_train try/finally 翻 ON（生产训练·镜像 PROPOSITION_MODE 范式）·否定命题建 pol=1 节点。
# **诚实边界**：结构否定（polarity 标记）墙内·否定语用（言外否定"我不觉得他来了"=他没来）= W2 defer。
NEGATION_MODE = _flag("ZERO_AI_NEGATION_MODE", False)

# ---- B2 情态 modality（P0.3 mod 进 surface·B2 情态 cue 抽取填 modality 值·"X 的 Y 必然是 Z" → modality>0） ----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §七 B2。情态词（必然/可能/也许/必须/应该/可以·is_modal_cue/
# modal_op_of）在"是"前（j-1·同否定词槽位·与 negation 互斥·先查 modal）→ extract_property_claims 情态窗口
# （modality_on 参数·_gated 传本 gate）·modality 填值（0-4）·命题节点建独立 surface 后缀 _0_{mod}（P0.3）·G3b 各判多值。
# **三级 gate**：PROPOSITION_MODE 守整体·NEGATION_MODE 守否定窗口·MODALITY_MODE 守情态窗口。
# OFF = modality_on=False → 既有肯定窗口 modality=0·"必然是"走错位 skip（bit-identical·既有行为零变）。
# ON = formal_train try/finally 翻 ON（生产训练·镜像 NEGATION_MODE 范式）·情态命题建 mod>0 节点。
# **诚实边界**：T 公理形式层墙内（构造性检查·非 truth·情态比命题多一口气=定理有效性层有形式锚）·
# 实质情态真值（认识/规范 W2 + 动力 W1）defer。**D6**：closed-class 情态副词·开放变体（想必/势必/说不定）走 D:11 教师晋升
# （MODAL_D11_READBACK_MODE·见 :189·审计根治 [严重-1]·非 REL_MODALITY 走 modal_kind concept+ATTR_MODAL_KIND=22 readback）。
MODALITY_MODE = _flag("ZERO_AI_MODALITY_MODE", False)

# ---- 程度 intensity（#1134 程度→属性器 augment·degree 副词缩放命题值强度·"X 的 Y 是 非常 Z" → intensity=2/1） ----
# 设计 doc/重来_程度属性器intensity_2026-07-16.md + 权威 doc/重来_ChineseSemanticKB能力映射 §4.3。程度副词（很/非常/极其/较/稍·
# degree_intensity_of·**file-driven 来自 degree_cues_zh.txt·非 §九 code frozenset**·程度是语义强度非句法锚）在 copula 是 与 value
# 间（tokens[val_idx] 是 degree cue）→ extract_property_claims degree 窗口（degree_on 参数·_gated 传本 gate）·value 后移一位
# （degree 占 val_idx·真 value=val_idx+1）·intensity (num,den) 填值·命题节点建独立 surface 后缀 _i{num}_{den}（≠1/1）+ ATTR_PROP_INTENSITY=30。
# **四级 gate**：PROPOSITION_MODE 守整体·NEGATION/MODALITY/DEGREE 守各自窗口（degree 与 modal/negation 正交·可共存·首版 degree-only 窗口·复合 defer）。
# OFF = degree_on=False → 既有窗口 intensity 恒 1/1（无 _i 后缀·surface 同旧）·bit-identical·既有行为零变。
# ON = formal_train try/finally 翻 ON（生产训练·镜像 MODALITY_MODE 范式·boot 先 populate_degree_cues 喂 cache）·程度命题建 intensity 节点。
# **诚实边界**：intensity magnitude 暂无消费者（G3b 读 PROPERTY 出边 count 非量级·judge 只权 CAUSES/PRECEDES）·
# 数据完备（staged+loadable+field+提取 gate）非 demonstrated win·consumer defer（intensity-aware A1 聚合 / degree-comparison judge·revisit）。
DEGREE_MODE = _flag("ZERO_AI_DEGREE_MODE", False)

# ---- 语言域统计层断奶判定（#1143·5判据 formal gate·非 can_ween·绝不妥协·另建 verdict）----
# 设计 doc/重来_语言域统计层断奶_2026-07-16.md。另建判定（capability_exam FOOTNOTE_WEANING / weaning.py
# 「统计层持续学习就绪判据另建·未建 defer」）：5判据 + 5 反 theater 锚点·显式排除 E2(truth·#479)/
# D5(Mode B·语言域无等价)/D3(独立 GT·defer #731)·weaker-than-can_ween。
# OFF（默认）= formal_train 不算 statistical_weaning_ready（result 字段默认 False/None·CI bit-identical）。
# ON = 生产语言域 run（runner 翻 ON）算 language_statistical_weaning_check·measured-guard 守
# （未建测量 fadeout/heldout → 锚不过 → 诚实 False·非 theater）。
# **2审纠偏**（HIGH-1/2/3）：D2 硬 gate（closes permissive）+ fadeout/heldout measured-guard（防 stub-0 vacuous）。
STATISTICAL_WEANING_MODE = _flag("ZERO_AI_STATISTICAL_WEANING_MODE", False)

# ---- B4 频次维 observe_tn（方案3 tn路·β_arith 修法·STEP4 P1a）----
# 设计 doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §五 B4 + memory topic
# zero-ai-reboot-step4-b4-freq-p1-arith-collapse-2026-07-10（β_arith 病实测确认·consumer fire 但 w_freq 塌缩）。
# β_arith 病根：reward_propagate 落点① concept_targets episode 后写·reward>0 时同比 e_sn++&e_tn++ →
# 参与 concept 全同 e_tn·rate 全 1.0·key 再细无效·w_freq 概念间同（分化借 w_sp selection_pref 维）。
# 方案3 tn路：observe_tn 决策时写（sign-agnostic·独立 episode reward 符号）·跨 episode 分化·
# consumer read_effective_freq observe_mode=True 读 base+observe_tn（替 base+e_tn·e_tn β_arith 塌缩弃用）。
# 写时机：dag_path add_active 后 once-per-node（_node_activated flag·避多头过计）·path-reached token concept 写。
# **attractor add_seed 不另写**（对抗审 round2 隐患 B 修）：maybe_expand_attractor 仅在 dag_path `if sel:` 块内被
# path-reached node 调用 → attractor-expanded ⊆ path-reached → dag_path add_active 已覆盖·attractor 写致双写(+2)违 once-per-node。
# e_set 成员：struct_ref 初始 seed（OI_MODE ON 跳 add_active·数据真空 eff_freq=0 w_freq=ONE 正确）+
# attractor-expanded token concept（path-reached·dag_path 写 observe_tn·_seed_weight 读 eff_freq 分化）。
# 两 consumer：word_terminated（dag_path:130·读 path 节点 eff_freq 判通识终止）+ _seed_weight（a3_pr_wrapper:215·
# 读 e_set seed eff_freq 计算 w_freq）。promote _experience_ok 读 e_sn + (observe_tn if gate ON else e_tn) RATE（D:11 promote 专用·**审计根治 [严重-2] 改·gate ON observe_tn 替 e_tn 缓解 β_arith·非三重② _reward_ok·不动边级**）。
# **丢弃改点 record_base_freq first-non-zero-wins**（memory agent2 提议但违"诚实降级"locked 设计 +
# test_record_base_freq_after_outcome_skipped 锁定 first-write-wins·非 gate OFF 零回归·observe_tn 独立于 base_freq）。
# OFF = read_effective_freq observe_mode=False（base+e_tn 既有）+ dag_path/attractor 不写 observe_tn → bit-identical。
# ON = observe_mode=True（base+observe_tn）+ dag_path add_active/attractor add_seed 后写 observe_tn += 1。
# **★生产训练 try/finally 翻 ON**（镜像 ATTRACTOR_MODE·formal_train reward 阶段 try/finally·否则 observe_tn 永不写 = theater）。
# 诚实边界：observe_tn 是决策活动计数非语义频次（stable≠correct·#479 墙）·正反馈非死锁（K_CAP=8 + θ_freq=1000 双上界·
# word_terminated 读历史不含本 episode）·e_tn 仍写（reward feed 不变·observe_tn 只替 eff_freq 读源 + promote _experience_ok gate ON 替 e_tn·审计根治 [严重-2] β_arith 缓解）。
FREQ_OBSERVE_MODE = _flag("ZERO_AI_FREQ_OBSERVE_MODE", False)

# **方案3 tn路（B5 β_arith 修法）**：selection_pref 维 consumer 读门（SP_OBSERVE_MODE）。
# sp_tn 混 observe（record_selection_pref_cooccur 段内共现 sign-agnostic）+ reward（record_selection_pref_reward
# episode 末 reward>0 sp_sn++&sp_tn++）两路 → reward>0 episode 同 concept_targets 同比 sp_tn++ → rate 塌缩。
# sp_observe_tn 由 record_selection_pref_cooccur 同写（observe 路纯副本·sign-agnostic·独立 episode reward 符号）。
# OFF → consumer 读 base+sp_tn（既有 bit-identical·β_arith 染色）/ ON → 读 base+sp_observe_tn（避染·跨决策分化）。
# 写由既有 SELECTION_PREF_MODE 守（record_selection_pref_cooccur 调用门·build_selection_pref_count:72）·本 gate 只控制 consumer 读。
# 两 consumer：_seed_weight sp_agg（a3_pr_wrapper·PR 侧粗筛·读 sum_base+sum_sp_tn→sum_sp_observe_tn·不读 sp_sn·对偶 B4 w_freq）/
#   selection_pref_score（graph_view·生成侧精查·读 sp_sn+sp_tn→sp_sn+sp_observe_tn·sp_sn reward 路仍读·诚实边界染）。
# 诚实边界：sp_observe_tn 是段内共现决策活动计数非语义正确（stable≠correct·#479 墙）·sp_sn reward 路仍染（observe 只替 sp_tn·非根治）·
# reward CAUSES-only 守（observe 不接 reward·独立表统计非 reward feed·不破防塌柱①）。
# **★生产训练 try/finally 翻 ON**（镜像 FREQ_OBSERVE_MODE·stage loop 块）。
SP_OBSERVE_MODE = _flag("ZERO_AI_SP_OBSERVE_MODE", False)

# **方案3 tn+fn 路（B6 指代维 β_arith 修法）**：指代消解 count 表读写门（PRONOUN_RESOLVE_COUNT_MODE）。
# §九.2 病灶"J4 episode 级 bool·attribute 给谁不干净"——当前 resolve_pronoun_occurrence 悬空只 _segment_dangling++
# （段级 int 累加·非 per-occurrence·attribute 给谁不干净）。B6 observe 扩展 = resolve 内 per-occurrence 写
# pronoun_resolution_count（pr_tn=决策时 sign-agnostic / pr_fn=悬空 self-loop per-occurrence·独立 episode 符号·避 β_arith）。
# 写：record_pronoun_resolution_decision（pr_tn++·选 best antecedent）/ record_pronoun_resolution_dangling（pr_fn++·self-loop 悬空）。
# 读（consumer·生产非纸面闭合）：resolve_pronoun_occurrence 自消费读历史 pr_tn(pronoun, candidate) 加候选分
# （B0 dim consumer=refers_occurrence·observe 侧·reward>0 鲁棒·J4 bool veto 只查 dangling 不查 antecedent 质量·选哪个非悬空 antecedent 都 reward>0）。
# 件 C dispatch_slot pronoun scoring defer STEP6（§十二.5 主面件 C slot 级）。
# OFF → 不写不读 count 表 → resolve 候选排序 bit-identical（既有 _segment_dangling J4 veto 不变·factor E reward>0 不变）。
# ON → 写 count + 自消费读 pr_tn 加候选分（score += min(pr_tn, PR_TN_BONUS_CAP=3)·recency 尺度·不颠覆 layer3 effective_weight 1000）。
# 诚实边界：指代维 reward=J4 bool veto（非 graded·与 B4/B5 count 进 _seed_weight reward 加权不同）·count 不进 reward 公式·
#   consumer 在 observe 侧自消费·pr_sn 教师 P2 defer·pr_tn/pr_fn per-occurrence 决策时写非 episode 末（避 β_arith）·
#   代词消解结构非墙 vs sense 消歧 #479 真墙（§九.7.6 W2 拆分）·stable≠correct（"它们→最近 token 可能功能词"接地墙外）。
# **★生产训练 try/finally 翻 ON**（镜像 FREQ/SP_OBSERVE_MODE·stage loop 块）。
PRONOUN_RESOLVE_COUNT_MODE = _flag("ZERO_AI_PRONOUN_RESOLVE_COUNT_MODE", False)

# ---- 刀 D：语言域比较 cue verify-driven episode（self_proof_fn 独立通道·绕 judge·镜像刀 A/B/C） ----
# formal_train.run_round_full 语言域路由分支（numeric 分支后·universal 前·numeric>comparison>universal>precedes 序）：
# 语言域 + 任段 comparison_claims 非空 + 本 gate ON → 走 _run_comparison_verify_round（镜像 _run_numeric_verify_round·
# 但验比较声明算术序·cross_compare 交叉积·非整数算术·非 PRECEDES DAG·非 COMPOSES 执行）·直调 comparison_proof_fn 绕 judge·
# reward=1 iff 全声明比序一致·不落 strength（verify propagate no-op）·**永不接 reward**（比较声明不入图·闭包传）。
# **第 4 个 LIVE form_proof_fn**（刀 A 时序 / 刀 B 数值 / 刀 C 量化 / 刀 D 比较）·给 cross_compare 首个真**比较**消费者
# （既有 1 caller 非比较用途·分层墙 §四缝1·反 theater：机制获真消费者）。构造性检查 SELF_PRODUCED（数 single-source·
# 同刀 A/B·Layer0 全自产不准停·非刀 C 的 EXTERNAL 验证）。
# **doc "命题值比序" 降级 deferred**（设计 §三）：(B) 命题节点 value 比序须 ref→surface 反查（concept_index 无·概念节点
# 纯整数·surface 在 companion）= "大" scope → 本刀做 (A) 字面数值比序（操作数文本字面整数·不依赖 #774）。
# **诚实边界**：(B) 命题值比序 defer（须 ref→surface 基建 + 配对语料）·整数 operand only（分数 defer）·
# stable≠correct（比序一致 ≠ 命题真·#479 墙）·比较 OP 词不入 _CUE_WORDS（bit-identical 比刀 B 更 safe）。
# default OFF·守 CI 回归（OFF = 路由不走 → 既有路径不变 bit-identical）。
# **★生产训练 try/finally 翻 ON**（镜像 TIME_SEQ/NUMERIC/UNIVERSAL 范式·formal_train try + finally 复位·否则孤儿 = theater）。
COMPARISON_PROOF_MODE = _flag("ZERO_AI_COMPARISON_PROOF_MODE", False)


# P0a（2026-07-14·纠偏回合 round2 地基）：ordinal 码点 surface resolver。
# generate 的 surface_of 读 concept_correspondence 码点 → chr → 真实文本（解 A 偏离：生成吐 #1:42 占位）。
# 写侧（concept_index.ensure record_correspondence·码点入核心）ungated always-write（纯 additive·守 dump
# 完整性）·读侧（surface_of）gate 控：default OFF → surface_of 退 None → 占位（既有行为 bit-identical·CI=生产）。
# 生产 try/finally 翻 ON（formal_train ConceptGraph build 期·live-read·try+finally 复位守测试隔离）。
# blast radius 最小：h63 身份/local_id/dedup 全不动·只变 OutputPart.words（None/占位→真字·gate 控制）·
# reward/dag_path 不读显示文本（reward 活信号不变）·详见 plan velvet-juggling-garden.md。
ORDINAL_SURFACE_MODE = _flag("ZERO_AI_ORDINAL_SURFACE_MODE", False)


_THIS_MODULE = sys.modules[__name__]
if not isinstance(_THIS_MODULE, _ContextLocalGateModule):
    _THIS_MODULE.__class__ = _ContextLocalGateModule
