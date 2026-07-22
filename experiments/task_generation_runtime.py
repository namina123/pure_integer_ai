"""任务驱动生成、行为验证和生成汇总 runtime。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveredOperator,
    _OP_CONF_RATE_SCALE,
    load_discovered_operators,
)
from pure_integer_ai.cognition.shared.types import (
    CodeSpec,
    Episode,
    MODALITY_ARITH,
    MODALITY_CODE,
    MODALITY_LANGUAGE,
    OutputPart,
    OutputResult,
    TERMINAL_REACHED_SINK,
    VERIFY_SOURCE_EXTERNAL,
    VERIFY_SOURCE_SELF_PRODUCED,
)
from pure_integer_ai.config import gates
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.op_confidence import (
    read_op_confidence,
    record_op_outcome,
)
from pure_integer_ai.training.vm_proof import execute_composes_value

@dataclass
class GenerateSummary:
    """§8.7-全 生成侧 task-driven L8 episode 汇总（任务→选算子→执行骨架→验·§8.7-全）。

    **认识论增量（审计降级后·Mode A 构造性·非"外真非传递"）**：半环（§8.7-洗 done）测**自洽**——
    skeleton(recognized_params)==input()（骨架复现输入程序值·学生==学生·传递必然对正确算子）。
    task-driven 测**新 args 泛化探针 + 生成侧函数应用姿态**——skeleton(新 task args)==expected（骨架对新
    任务输入产答案·args 非学习输入→产新值非记忆复现·生成姿态 call 算子为函数 vs 半环识别姿态 align 程序）。
    **Mode A 构造性**：skeleton 派生自输入程序·故 expected=正确答案时 skeleton(args)==expected 构造性必然
    （传递经 skeleton 起源·同半环牙·抓获 PARAM 序错/编译发散/shape 异配·无新牙）·**非"外真非传递"**。
    真非传递外真 = Mode B（异算法闭式 O(1) vs 迭代 O(n)·断奶后 defer·§8.7-全 前置/范围）。

    total_tasks : 任务总数（arith_specs 总数·每 CodeSpec=(input_args,expected) 一任务）。
    selected    : 选到算子的任务数（arity 匹配候选·滤 tested-never-verified 后有候选）。
    verified    : 验过数（skeleton(input_args)==expected·Mode A 构造性·output.parts 非空⟺verified）。

    反 theater 4 锚点（审计后·①④ PASS·②③ 降级/修）：①行为真变（多候选读置信度择优·置 sn=0 产出变）②
    新消费（skeleton(新 args) vs expected·新 args 非学习输入·**Mode A 构造性·真非传递留 Mode B**）③下游读者
    （OutputResult.parts→metrics generate_verified 计数真读·**审计必修：metrics 读 parts 非 reward·parts 非空⟺
    verified·否则=(Z) theater 重引入**）④拒坏选好（不 fit task→tn++·fit→sn++·择优选 fit）。
    诚实边界：①单候选构造性必过（信号薄·同半环）·多候选信号实·信号强度依赖语料覆盖非墙；
    ②mul/square 不可分（置信度正交于变量同一性判别器·Half B arity 区分 arity·同 arity 同形仍不可分）；
    ③stable≠correct（VM 跑通≠语义对·结构正确非语义理解·接地墙算术版·不阻塞）；
    ④防双计=caller 责任（task 输入须 ≠ recognize held-out 程序·机制不强制·rate 不受影响·§8.7-全 F3）；
    ⑤闭环范围=生成侧 task-driven 探针（Mode A 构造性）+ 半环（自洽）= 生成侧 wash 环（对本模态·Mode A）·
    **真外真全环须 Mode B**·generate.py 字面路径仍须 STRUCT_BIND(VF)·独立后续。
    """
    total_tasks: int = 0
    selected: int = 0
    verified: int = 0
    # #1124 symbolic 子计数（additive·0=CI 无 symbolic specs bit-identical·formal_train _run_task_driven_generate 传）：
    # xform_verified=transform 规则 cross-verify verified 数（S5-S7）·inv_verified=inverse 关系 B∘A 还原 verified 数（S8）。
    # capability_exam.project_symbolic_measures → CapabilityReport.math_measures（反 theater：symbolic 学习可见非 invisible）。
    xform_verified: int = 0
    inv_verified: int = 0
    # #1124 M-1：symbolic 分母（spec §四 4.2 symbolic_cross_verify_rate permille 用）。
    xform_total: int = 0
    inv_total: int = 0

    @property
    def rate_permille(self) -> int:
        """验过率 ×1000（verified / total_tasks·total=0→0·纯整数无浮点）。"""
        return (self.verified * 1000) // max(self.total_tasks, 1)

def _run_task_driven_generate(ctx: TrainContext, corpus: list[CollectedItem],
                              all_ops: Sequence[DiscoveredOperator]
                              ) -> tuple[list[Episode], GenerateSummary]:
    """§8.7-全 生成侧全环·task-driven L8 episode：任务(input_args,expected) → 选算子 → 执行骨架
    → 外真验 vs expected → 写 op_confidence → 打包 OutputResult（6 步·不碰 generate.py L6·守单向 L8→L7/L0 向下）。

    **为何非 theater（审计降级后·Mode A 构造性·非"外真非传递"）**：半环（§8.7-洗 done）测自洽
    skeleton(recognized_params)==input()（学生==学生·传递必然对正确算子）·本函数测**新 args 泛化探针 +
    生成侧函数应用姿态**——skeleton(新 task args)==expected（生成姿态 call 算子为函数产答案·vs 半环识别姿态
    align 程序·args 非学习输入→产新值非记忆复现）。**Mode A 构造性**：skeleton 派生自输入程序·故 expected=正确答案时
    skeleton(args)==expected 构造性必然（传递经 skeleton 起源·同半环牙·无新牙）·**非"外真非传递"**。
    真非传递外真 = Mode B（异算法·断奶后 defer·§8.7-全 前置/范围）。task-driven 真增量 = 新 args 探针 + 生成姿态 +
    多候选 wash（①④真）·非伪需求·但认识论声称须诚实（Mode A 构造性·非外真）。

    **6 步机制**（§8.7-全）：
      1. task 输入 = CodeSpec(input_args, expected)（教师 Mode A 独立源·断奶前·vm_proof.py:91 R6）。
         任务来源 = arith item 的 arith_specs（同 item specs·设计决策点 (a)·不建输入程序选算子执行）。
      2. 选算子：候选 = arity==len(input_args) 的已发现算子·读 read_op_confidence → rate=sn*1000//max(tn,1)
         → 滤 tested-never-verified(sn==0·同 recognize_operators:730-732) → 稳定排序择优（同率保 BFS 序·bit-identical）。
      3. 执行：execute_composes_value(graph, op.skeleton_ref, input_args) → v_skel（L8→L7 OK·复用 :1108 既有）。
      4. 验（Mode A 构造性）：rational.eq(v_skel, make(*expected)) → verified（外比 expected·Mode A 构造性必然对正确算子）。
         None→not verified（诚实）。
      5. 写置信度：record_op_outcome(ref=op.name_ref, verified=eq)（L8→L0 OK·复用 :1159 既有·与 recognize 半环同表累积）。
      6. 打包 OutputResult（**审计必修·parts 非空⟺verified**·未验→parts=[]）→ Episode.output → metrics
         record_generate_round 读 e.output.parts 计数 generate_verified（真消费·非死写）。

    **反 theater 4 锚点**（审计后·①④ PASS·②③ 降级/修）：①行为真变（多候选读置信度择优·置 sn=0→选他者产出变）②
    新消费（skeleton(新 args) vs expected·新 args 非学习输入·**Mode A 构造性·真非传递留 Mode B**）③下游读者
    （OutputResult.parts→metrics generate_verified 计数真读·**metrics 读 parts 非 reward·parts 非空⟺verified·
    审计必修·否则=(Z) theater 重引入**）④拒坏选好（不 fit→tn++·fit→sn++·择优选 fit）。

    **防双计**（诚实边界④·审计 F3·**caller 责任·机制不强制**）：task_items 取所有有 arith_specs 的 arith item·
    若 held-out 程序项同携匹配 specs（input_args==程序 args）则同值同试验 sn/tn 双计·**但 rate=sn/tn 同比 inflate→
    rate 不变→选择不变·仅绝对计数虚增**。机制不强制 task 输入≠held-out（强制须跳过 recognize_roots 项的 specs·
    独立 follow-up）·caller 须确保 task specs 用 args ≠ held-out 程序。

    铁律：纯整数（input_args/expected/v_skel num/den 全 int·assert_int·Rational 经 make）/ bit-identical
    （rate 排序稳定 tiebreak BFS 序·同 recognize）/ 单向依赖（L8→L7 execute/L0 op_confidence 向下·不碰 L6 generate.py）/
    MUTABLE_MONOTONE（record_op_outcome 同表同键 UPDATE·R1 符号 sn 单调）/ 不写死（选算子=arity 结构匹配·rate 比较非硬编码）/
    核心无墙钟 / 不走外挂 LLM（断奶前教师 Mode A）/ 危险词禁 / IS_A 不涉。
    诚实边界：①单候选构造性必过（仅 mul 时 skeleton(args)==expected 构造性必过·信号薄·同半环）·多候选信号实；
    ②mul/square 不可分（同 task 两都 verified·置信度正交于变量同一性·Half B arity 区分）；③stable≠correct；
    ④防双计 caller 责任（见上·机制不强制·rate 不受影响）；⑤闭环=生成侧 task-driven 探针（Mode A）+ 半环（自洽）=
    生成侧 wash 环（对本模态·Mode A）·**真外真全环须 Mode B**·generate.py 字面须 STRUCT_BIND(VF)·独立后续。
    """
    assert isinstance(all_ops, Sequence), "_run_task_driven_generate.all_ops 须 Sequence[DiscoveredOperator]"
    # 候选按 arity 索引（len(input_args)==op.arity·结构匹配·非语义规则）
    by_arity: dict[int, list[DiscoveredOperator]] = {}
    for op in all_ops:
        by_arity.setdefault(op.arity, []).append(op)
    graph = ctx.concept_graph
    total_tasks = 0
    selected = 0
    verified = 0
    # S5-S8 symbolic 子计数（#1124·additive·镜像 GenerateSummary·0=CI 无 symbolic specs bit-identical）：
    # transform 规则 verified 数（S5-S7 SYMBOLIC_TRANSFORM 块）+ inverse 关系 verified 数（S8 SYMBOLIC_RELATION 块）。
    # 用于 capability_exam.project_symbolic_measures → CapabilityReport.math_measures（反 theater：symbolic 学习可见）。
    xform_verified = 0
    inv_verified = 0
    # #1124 M-1：symbolic 分母（spec §四 4.2 要求 rate permille·镜像 lang_rate_permille）。
    # xform_total=transform 规则总数（含 malformed/cross-verify 失败·分母）·inv_total=inverse 关系总数。
    xform_total = 0
    inv_total = 0
    episodes: list[Episode] = []
    # 任务来源 = arith item 的 arith_specs（同 item specs·设计决策点 (a)）·非 arith_source（不建输入程序）
    task_items = [it for it in corpus
                  if it.modality == MODALITY_ARITH and it.arith_specs]
    ep_id = 0
    for item in task_items:
        for spec in item.arith_specs:
            total_tasks += 1
            n_args = len(spec.input_args)   # CodeSpec.__post_init__ 已 assert_int 守 input_args/expected
            candidates = by_arity.get(n_args, [])
            # 洗净：滤 tested-never-verified (sn==0)·同 recognize_operators:730-732·cold-start(None)给机会不滤
            viable: list[tuple[int, DiscoveredOperator]] = []
            for op in candidates:
                conf = read_op_confidence(ctx.backend, op.name_ref)   # (sn,tn,strength)|None·纯读 L8→L0
                if conf is not None and conf[0] == 0:
                    continue   # tested-never-verified（sn==0=验过皆败）滤除·cold-start(None)给机会
                rate = ((conf[0] * _OP_CONF_RATE_SCALE // max(conf[1], 1))
                        if conf is not None else 0)   # cold-start→0（给机会但排序末位·verified 率高优先）
                viable.append((rate, op))
            if not viable:
                continue   # 无 arity 匹配候选 / 全滤 → 诚实跳过（不伪造·不计 selected/verified）
            # 稳定排序·同率保候选 BFS 序（=发现序·bit-identical）·reverse=True 不破稳定性（equal 保输入序）
            viable.sort(key=lambda x: x[0], reverse=True)
            op = viable[0][1]
            selected += 1
            # 执行骨架（L8→L7 execute_composes_value·复用 _verify_generalization:1108 既有调用·input_args int→(arg,1) Rational）
            v_skel = execute_composes_value(

                graph, op.skeleton_ref, tuple((a, 1) for a in spec.input_args))
            # 验（Mode A 构造性·expected=正确答案时 skeleton(args)==expected 构造性必然·同半环牙·
            # 真非传递外真留 Mode B·§8.7-全 决断标注）。rational.eq 外比（expected=教师 Mode A 独立源）。
            eq = (v_skel is not None
                  and rational.eq(v_skel, rational.make(spec.expected[0], spec.expected[1])))
            if eq:
                verified += 1
            # 写置信度（L8→L0·复用 _verify_generalization:1159 既有·与 recognize 半环同表累积·R1 符号 sn 单调）
            if op.name_ref != (0, 0):
                record_op_outcome(ctx.backend, ref=op.name_ref, verified=eq)
            # 打包 OutputResult（**反 theater ③下游读者锚·审计必修**：parts 非空 ⟺ verified·
            # 未验过/编译发散→parts=[] 不提交产出·metrics record_generate_round 读 e.output.parts 计数
            # generate_verified（真消费·非死写·否则=§8.7-洗-证伪 candidate (Z) theater 重引入）。
            # verified→parts=[算子名 ref + 产出值]·未验→parts=[]（产出值未提交·失败计数在 op_confidence tn）。
            if eq and v_skel is not None:
                output = OutputResult(parts=[OutputPart(
                    unit=op.name_ref, words=[f"{v_skel.num}/{v_skel.den}"])])
            else:
                output = OutputResult(parts=[])   # 未验过/None→不提交产出（诚实·metrics 读 parts 计数）
            ep = Episode(
                episode_id=ep_id, run_id=ep_id,
                input=None, output=output,
                reward=1 if eq else 0,
                ref=op.name_ref,
                terminal=TERMINAL_REACHED_SINK,
                pr_vector={},   # task-driven 不跑 dag_path_step·无 PR 向量（诚实）
                judge_G5_active=False,   # task-driven 不经 judge G5（外真验 = execute vs expected·非 G5 门）
                judge_veto_count=0 if eq else 1,
                dead_end_count=0,
                vetoed=(not eq),
                verify_source=VERIFY_SOURCE_EXTERNAL,   # Layer0：execute vs spec.expected R6 外部源（同 _run_verify_round·Mode A 构造性验证·可计 external_verified·2 审 P1-1 修）
            )
            episodes.append(ep)
            ep_id += 1
    # 断桥 Phase A（P2 G-PR2/3 cross-path·ACTION_BRIDGE_MODE ON·doc/重来_断桥设计refinement_2026-07-15）：
    # CollectedItem.action_specs（教师标 I/O 例·数据驱动**非硬编码**）→ synthesize_value **联合匹配**全 specs（PbE·
    # 一动作多 I/O 例共定一骨架·反 per-spec 碎·审2 F4）→ 独立 task-driven episode。断桥 cross-path：language/action
    # item 经 action_specs 跨路径喂 arith 骨架池合成（**spec→synthesis**·intent 分类=Phase B 动态构造器·Phase A 教师
    # 标 specs 已含 intent 语义·审2 F1/F2/F3 修回：design 原 dict[action_ref]+classify_intent 移 Phase B）。
    # **weaning-safe 决断 A**：独立 episode·**不替换 vm_proof verify round·不碎 W7**（反 VALUE_SYNTHESIZE 翻 ON 教训）。
    # gate OFF 或无 action_specs → 不进 → bit-identical。无匹配骨架 → 诚实 continue（同 arith no-viable :3512-3513）。
    # **不写 op_confidence**（断桥 cross-path 独立度量轴·teacher Mode A specs 构造性·reinforce 会 inflate·Phase B
    # held-out 真泛化后接·同 code-unparse :3590 范式）·故 unit/ref=skeleton_ref（合成骨架·非 arith name_ref·无
    # op_confidence 消费者·Phase B 接 op_confidence 时统一 name_ref·审1 MEDIUM-2/3）。selected==verified（synthesis
    # 返匹配皆已验·match=verified 内禀·非 arith 两段式·审1 LOW-3）。pool=load_discovered_operators（同 verify round
    # :653·persisted 发现骨架升序·all_ops 含本 run 非 persisted 不用·断桥/verify 匹配 persisted 发现骨架·审2 L2）。
    # Phase B（动态 intent→spec 构造器 + dispatch 桥 CHANNEL_*→VM/serializer/judge）defer。
    if getattr(gates, "ACTION_BRIDGE_MODE", False):
        # typed language owner 已由 G-00 至 G-04 负责输出；旧 action bridge 只保留给兼容链。
        _bridge_items = [
            it for it in corpus
            if it.action_specs and not (
                it.modality == MODALITY_LANGUAGE
                and ctx.language_generation_runtime is not None)
        ]
        if _bridge_items:   # 无 action_specs → 不 load pool（避空载·审1 LOW-4）
            from pure_integer_ai.training.value_synthesize import synthesize_value
            _bridge_pool = load_discovered_operators(ctx.backend, space_id=ctx.space_id)
            for _bitem in _bridge_items:
                total_tasks += 1   # 一 item = 一动作（联合 specs 共定·非 per-spec·审2 F4）
                _bmatches = synthesize_value(graph, _bridge_pool, tuple(_bitem.action_specs))
                if not _bmatches:
                    continue   # 无行为匹配骨架 → 诚实跳过（不伪造 episode·同 arith no-viable）
                _bsynth_root, _bbinding = _bmatches[0]   # 首匹配（pool 升序·bit-identical·多匹配 defer Phase B 排序·同相1 :656）
                selected += 1
                verified += 1   # match=verified 内禀（synthesize_value 已 execute+eq·审2 H4）
                # 实际产出值（re-execute spec[0]·非 expected·守 parts=actual 不变量·同 arith :3519/3536·审1 MEDIUM-1）：
                # match 保证 actual==spec[0].expected·re-execute 守未来 synthesize bug 不静默（_binding_param_values:67 镜像）。
                _bfirst = _bitem.action_specs[0]
                _bparam_vals = tuple((_bfirst.input_args[_bbinding[i]], 1)
                                     for i in range(len(_bbinding)))
                _bactual = execute_composes_value(graph, _bsynth_root, _bparam_vals)
                _boutput = OutputResult(parts=[OutputPart(
                    unit=_bsynth_root, words=[f"{_bactual.num}/{_bactual.den}"])])
                episodes.append(Episode(
                    episode_id=ep_id, run_id=ep_id,
                    input=None, output=_boutput,
                    reward=1, ref=_bsynth_root,
                    terminal=TERMINAL_REACHED_SINK,
                    pr_vector={}, judge_G5_active=False,
                    judge_veto_count=0, dead_end_count=0, vetoed=False,
                    verify_source=VERIFY_SOURCE_EXTERNAL,   # action_spec expected = 教师 Mode A 外部源（同 arith :3550）
                ))
                ep_id += 1
    # 断桥 Phase B 片1（P2 动态构造器·ACTION_BRIDGE_CUE_MODE ON·doc/重来_断桥设计refinement_2026-07-15 §Phase B 片1）：
    # 无教师 action_specs 时·从 language text cues 动态构造 spec：CollectedItem.numeric_claims_flat（刀B observe 期
    # flatten·4-tuple `(left,op,right,result)`）→ CodeSpec 隐 op（input_args=(left,right)·expected=(result,1)·**op 隐藏**
    # =synthesize 找算子非刀B 验算子·真合成）→ synthesize_value 联合匹配（同算子多 claim 共定一骨架·PbE·混算子无匹配
    # 诚实 skip）→ 独立 task-driven episode。input source = text cues（非 teacher·非 held-out·非硬编码·解 Phase B
    # "撞 held-out"之谜：held-out 仅泛化验证须·构造+合成 NOW）。
    # **刀B 无冲突**：刀B reward round 验证轴（用 op·:406 路由）·断桥 generate stage 合成轴（隐 op）·两 stage 分离·
    # 同 numeric item 得两 episode 不同轴（同 Phase A 与他处理共存范式）。断桥合成 ≠ 刀B 验证（合成产骨架学算子）。
    # **weaning-safe 决断 A**：独立 task-driven episode·不替换 vm_proof verify round·不碎 W7（同 Phase A）。
    # gate OFF 或无 numeric_claims_flat → 不进 → bit-identical。无匹配骨架 → 诚实 continue（同 Phase A :3576）。
    # **不写 op_confidence**（同 Phase A·独立度量轴·Phase B held-out 真泛化后接）。
    if getattr(gates, "ACTION_BRIDGE_CUE_MODE", False):
        _cue_items = [
            it for it in corpus
            if it.numeric_claims_flat and not (
                it.modality == MODALITY_LANGUAGE
                and ctx.language_generation_runtime is not None)
        ]
        if _cue_items:   # 无 numeric_claims_flat → 不 load pool（避空载·同 Phase A :3569）
            from pure_integer_ai.training.value_synthesize import synthesize_value
            _cue_pool = load_discovered_operators(ctx.backend, space_id=ctx.space_id)
            for _cueitem in _cue_items:
                # cue→spec：numeric_claim (left,op,right,result) → CodeSpec((left,right),(result,1))·op 隐藏（synthesize 找算子）
                _cue_specs = tuple(
                    CodeSpec(input_args=(c[0], c[2]), expected=(c[3], 1))
                    for c in _cueitem.numeric_claims_flat)
                total_tasks += 1   # 一 item = 一合成任务（联合 specs 共定·同 Phase A·审2 F4 范式）
                _cuematches = synthesize_value(graph, _cue_pool, _cue_specs)
                if not _cuematches:
                    continue   # 无行为匹配骨架 → 诚实跳过（同 Phase A·不伪造 episode）
                _csynth_root, _cbinding = _cuematches[0]   # 首匹配（pool 升序·bit-identical·多匹配排序 defer 片2）
                selected += 1
                verified += 1   # match=verified 内禀（synthesize_value 已 execute+eq·同 Phase A·审2 H4）
                # 实际产出值（re-execute spec[0]·非 expected·守 parts=actual 不变量·同 Phase A·审1 MEDIUM-1）：
                _cfirst = _cue_specs[0]
                _cparam_vals = tuple((_cfirst.input_args[_cbinding[i]], 1)
                                     for i in range(len(_cbinding)))
                _cactual = execute_composes_value(graph, _csynth_root, _cparam_vals)
                _coutput = OutputResult(parts=[OutputPart(
                    unit=_csynth_root, words=[f"{_cactual.num}/{_cactual.den}"])])
                episodes.append(Episode(
                    episode_id=ep_id, run_id=ep_id,
                    input=None, output=_coutput,
                    reward=1, ref=_csynth_root,
                    terminal=TERMINAL_REACHED_SINK,
                    pr_vector={}, judge_G5_active=False,
                    judge_veto_count=0, dead_end_count=0, vetoed=False,
                    verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # 审2 LOW-1 修：cue-derived spec.expected 来自 text cues
                    # （numeric_claims_flat·single-source·**非 R6 外部源**·同刀B SELF_PRODUCED :893·非 Phase A action_spec
                    # 教师标 R6 外部）→ SELF_PRODUCED 守"全自产不准停"（layer0_anchor.py:72 EXTERNAL 才计 external_verified
                    # 驱动停止决策·text-derived synthesis 不准驱动停止·反 theater）。
                ))
                ep_id += 1
    # 符号数学扩展 Phase 3（SYMBOLIC_TRANSFORM_MODE ON·doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis.7）：
    # CollectedItem.transform_specs（教师陈述符号变换规则·数据驱动**非硬编码**·humans 学法：从教师/课本学规则
    # 陈述+验证+应用+关联）→ register_transform_rule + apply held-out input + cross-verify 执行 output==expected
    # → 独立 task-driven episode。**weaning-safe 决断 A**：独立 task-driven episode·不替换 vm_proof verify round·
    # 不碎 W7（同断桥 Phase A/B）。**verify_source=SELF_PRODUCED**：规则应用+cross-verify single-source 自产自验
    # （非 R6 外部源·规则+held-out 来自 corpus 但验证是自产执行比对）·守"全自产不准停"（同断桥 Phase B 片1·反 theater）。
    # gate OFF 或无 transform_specs → 不进 → bit-identical。LHS 不匹配/cross-verify 失败 → 诚实 continue（不伪造 episode）。
    # **诚实边界**：cross-verify 单点采样（小素数探针 per arity·多采样点 refinement defer）·stable≠correct（#479 守）。
    if getattr(gates, "SYMBOLIC_TRANSFORM_MODE", False):
        _xform_items = [it for it in corpus if it.transform_specs]
        if _xform_items:
            import ast as _ast
            from pure_integer_ai.training.symbolic_transform import register_transform_rule, apply_transform
            from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith, UnsupportedConstruct
            from pure_integer_ai.storage.edge_store import SOURCE_MATH
            from pure_integer_ai.storage.node_store import NODE_CONCEPT
            _XFORM_PROBES = (2, 3, 5, 7, 11, 13)   # 小素数探针（cross-verify 采样点·per-slot·arity≤6）
            for _xitem in _xform_items:
                for _spec in _xitem.transform_specs:
                    total_tasks += 1   # 一规则 = 一学习任务（register + held-out 验）
                    xform_total += 1   # #1124 M-1 symbolic transform 分母（含失败/malformed）
                    # try/except 守（对抗审 Finding 1·mirror code_unparse :3753-3759）：malformed spec
                    # （build/apply/execute raise UnsupportedConstruct/ValueError/KeyError·如 Pow(x,0)→n-1=-1 负指数
                    # / DSL 解析错 / arity 不匹配）→ 诚实 skip 此 spec（不 abort 整个 run·守"诚实 continue"意图）。
                    try:
                        # build + register 规则（lhs/rhs lambda DSL → COMPOSES·教师陈述模板·非硬编码）
                        _lhs_ref = ctx.concept_index.ensure(
                            f"__xform_lhs_{_spec.rule_name}", space_id=ctx.space_id,
                            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                        build_composes_from_arith(_spec.lhs_source, concept_index=ctx.concept_index,
                            edge_store=ctx.edge_store, backend=ctx.backend,
                            space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_lhs_ref)
                        _rhs_ref = ctx.concept_index.ensure(
                            f"__xform_rhs_{_spec.rule_name}", space_id=ctx.space_id,
                            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                        build_composes_from_arith(_spec.rhs_source, concept_index=ctx.concept_index,
                            edge_store=ctx.edge_store, backend=ctx.backend,
                            space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_rhs_ref)
                        _rule_ref = register_transform_rule(
                            ctx.backend, ctx.concept_index, _spec.rule_name,
                            _lhs_ref, _rhs_ref, space_id=ctx.space_id)
                        # held-out cross-verify（apply 规则到 input·执行 output==expected·统计验规则应用正确）
                        _all_pass = bool(_spec.held_out)   # 无 held-out → 不验→不产 episode（反 theater）
                        _last_words = ["1/1"]
                        for _ho in _spec.held_out:
                            _in_ref = ctx.concept_index.ensure(
                                f"__xform_in_{_spec.rule_name}_{_ho.input_source}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_ho.input_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_in_ref)
                            _exp_ref = ctx.concept_index.ensure(
                                f"__xform_exp_{_spec.rule_name}_{_ho.expected_source}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)

                            build_composes_from_arith(_ho.expected_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_exp_ref)
                            _out_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                                space_id=ctx.space_id, source=SOURCE_MATH,
                                rule_name_ref=_rule_ref, input_ref=_in_ref)
                            if _out_ref is None:
                                _all_pass = False; break   # LHS 不匹配 → 诚实 skip（非 theater）
                            # cross-verify：arity 探针·执行 output==expected（stable≠correct·统计验非 truth）
                            _arity = len(_ast.parse(_ho.expected_source).body[0].value.args.args)
                            _probes = tuple((_XFORM_PROBES[i], 1) for i in range(_arity))
                            _vo = execute_composes_value(graph, _out_ref, _probes)
                            _ve = execute_composes_value(graph, _exp_ref, _probes)
                            if _vo is None or _ve is None or not rational.eq(_vo, _ve):
                                _all_pass = False; break   # cross-verify 失败 → 诚实 skip（反 theater·不伪造 verified）
                            _last_words = [f"{_vo.num}/{_vo.den}"]
                        if not _all_pass:
                            continue
                        selected += 1
                        verified += 1
                        xform_verified += 1   # #1124 symbolic transform 子计数（S5-S7）
                        episodes.append(Episode(
                            episode_id=ep_id, run_id=ep_id,
                            input=None, output=OutputResult(parts=[OutputPart(unit=_rule_ref, words=_last_words)]),
                            reward=1, ref=_rule_ref, terminal=TERMINAL_REACHED_SINK,
                            pr_vector={}, judge_G5_active=False,
                            judge_veto_count=0, dead_end_count=0, vetoed=False,
                            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # single-source 自产自验·不准驱动停止·反 theater
                        ))
                        ep_id += 1
                    except (UnsupportedConstruct, ValueError, KeyError):
                        continue   # malformed spec（build/apply/cross-verify raise）→ 诚实 skip·不 abort run
    # S8 符号间运算关联 Phase 1（SYMBOLIC_RELATION_MODE ON·doc/重来_S8符号间关联机制设计_2026-07-15 §七）：
    # CollectedItem.inverse_relation_specs（教师陈述逆关系·两条独立变换规则 A↔B 互逆·数据驱动**非硬编码**·humans 学法：
    # 从教师/课本学"两规则互逆"+构造验证·非纯归纳发现 research-grade defer）→ register rule_a + register rule_b +
    # register_inverse_relation + verify_inverse_relation（B∘A 还原 @ 采样·三值 True/False/None）→ verified 则独立
    # task-driven episode。**weaning-safe 决断 A**（同 SYMBOLIC_TRANSFORM 块·独立 episode·不替换 vm_proof·不碎 W7）。
    # **verify_source=SELF_PRODUCED**：两规则 single-source 教师·逆验证 self-consistency（非 R6 两源·非 truth）·
    # 守"全自产不准停"（同 transform_specs·反 theater）。gate OFF 或无 inverse_relation_specs → 不进 → bit-identical。
    # can't-verify(None)/falsified(False) → 诚实 continue（不伪造 episode·反 theater ③下游读者锚 parts 非空⟺verified）。
    # **诚实边界**：逆验证=统计非证明（采样还原 ≠ 数学逆·#479 守）·可复合约束（B LHS 须匹配 A 输出 shape·否则 can't-verify）。
    if getattr(gates, "SYMBOLIC_RELATION_MODE", False):
        _rel_items = [it for it in corpus if it.inverse_relation_specs]
        if _rel_items:
            from pure_integer_ai.training.symbolic_transform import register_transform_rule, load_transform_rule as _load_rule
            from pure_integer_ai.training.symbolic_relation import (
                register_inverse_relation, verify_inverse_relation, RELATION_KIND_INVERSE)
            from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith, UnsupportedConstruct
            from pure_integer_ai.storage.edge_store import SOURCE_MATH
            from pure_integer_ai.storage.node_store import NODE_CONCEPT
            for _ritem in _rel_items:
                for _rspec in _ritem.inverse_relation_specs:
                    total_tasks += 1   # 一逆关系 = 一学习任务（register 两规则 + 逆验证）
                    inv_total += 1   # #1124 M-1 symbolic inverse 分母（含失败/malformed）
                    # try/except 守（同 SYMBOLIC_TRANSFORM 块·mirror code_unparse :3753-3759）：malformed spec
                    # （build/register/verify raise UnsupportedConstruct/ValueError/KeyError·如 DSL 解析错 / arity 不匹配 /
                    # Pow 负指数）→ 诚实 skip 此 spec（不 abort 整个 run）。
                    try:
                        # build + register rule_a / rule_b（两变换规则·逆关系须先存在·镜像 SYMBOLIC_TRANSFORM 块建规则）。
                        # surface 键用 **rule_name**（非 relation_name·对抗审 MEDIUM）：register_transform_rule 按裸 rule_name
                        # 去重·同 rule_name 跨关系须映射同 lhs/rhs ConceptRef→幂等。**重 build 守**（对抗审 MEDIUM 深因）：
                        # build_composes_from_arith 重 build 进已填充 root 会破坏树（duplicate children）→ 共享 rule_name 的
                        # 第二关系重 build 致 apply 失败→不 verified。故 build 前查规则已注册（load_transform_rule）→ 复用跳 build。
                        from pure_integer_ai.training.symbolic_transform import load_transform_rule as _load_rule
                        _rule_refs = {}
                        for _rspec_rule in (_rspec.rule_a, _rspec.rule_b):
                            _rname_ref = ctx.concept_index.ensure(
                                _rspec_rule.rule_name, space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            if _load_rule(ctx.backend, _rname_ref) is not None:
                                _rule_refs[_rspec_rule.rule_name] = _rname_ref   # 已注册·幂等复用·跳 build（防重 build 树破坏）
                                continue
                            _rlhs = ctx.concept_index.ensure(
                                f"__rel_lhs_{_rspec_rule.rule_name}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_rspec_rule.lhs_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_rlhs)
                            _rrhs = ctx.concept_index.ensure(
                                f"__rel_rhs_{_rspec_rule.rule_name}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_rspec_rule.rhs_source, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_rrhs)
                            _rule_refs[_rspec_rule.rule_name] = register_transform_rule(
                                ctx.backend, ctx.concept_index, _rspec_rule.rule_name,
                                _rlhs, _rrhs, space_id=ctx.space_id)
                        # register 逆关系（KIND=INVERSE·挂关系名 concept + RULE_A + RULE_B）
                        _rel_ref = register_inverse_relation(
                            ctx.backend, ctx.concept_index, space_id=ctx.space_id,

                            name=_rspec.relation_name, kind=RELATION_KIND_INVERSE,
                            rule_a_ref=_rule_refs[_rspec.rule_a.rule_name],
                            rule_b_ref=_rule_refs[_rspec.rule_b.rule_name])
                        # build sample inputs e（B∘A 须还原这些 @ 探针）
                        _sample_refs = []
                        for _ss in _rspec.sample_sources:
                            _se = ctx.concept_index.ensure(
                                f"__rel_sample_{_rspec.relation_name}_{_ss}", space_id=ctx.space_id,
                                tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
                            build_composes_from_arith(_ss, concept_index=ctx.concept_index,
                                edge_store=ctx.edge_store, backend=ctx.backend,
                                space_id=ctx.space_id, source=SOURCE_MATH, root_ref=_se)
                            _sample_refs.append(_se)
                        # 构造验证 B∘A 还原（三值·反 theater 心脏）
                        _verified = verify_inverse_relation(
                            ctx.backend, ctx.concept_index, ctx.edge_store,
                            space_id=ctx.space_id, source=SOURCE_MATH,
                            relation_ref=_rel_ref, sample_inputs=_sample_refs)
                        if _verified is not True:
                            continue   # can't-verify(None)/falsified(False) → 诚实 skip（反 theater·不伪造 verified episode）
                        selected += 1
                        verified += 1
                        inv_verified += 1   # #1124 symbolic inverse 子计数（S8）
                        episodes.append(Episode(
                            episode_id=ep_id, run_id=ep_id,
                            input=None,
                            output=OutputResult(parts=[OutputPart(unit=_rel_ref, words=["inv:verified"])]),
                            reward=1, ref=_rel_ref, terminal=TERMINAL_REACHED_SINK,
                            pr_vector={}, judge_G5_active=False,
                            judge_veto_count=0, dead_end_count=0, vetoed=False,
                            verify_source=VERIFY_SOURCE_SELF_PRODUCED,   # 两规则 single-source 自产自验·不准驱动停止·反 theater
                        ))
                        ep_id += 1
                    except (UnsupportedConstruct, ValueError, KeyError):
                        continue   # malformed spec（build/register/verify raise）→ 诚实 skip·不 abort run
    # #730 路径 W：代码模态 task-driven episode（unparse COMPOSES→源码串·Mode A 构造性·gate CODE_UNPARSE_MODE）。
    # 与 arith execute 对称但走 unparse（非 execute·L8→L5 向下·不调 L7）：读 item.code_struct_ref（observe 期
    # __prog_* 根·候选 A·run_round_full observe 后捕获）→ unparse_composes 序化 → ast bodies_match normalize
    # == code_source normalize → verified（构造性必然·skeleton 派生自 code_source·同 arith skeleton(args)==expected·
    # formal_train.py:1944 范式·stable≠correct·非真生成·真生成须路径 X 跨模态 defer）。
    # 无算子选择（用 item 自身 observe 树·非 discover skeleton 抽象 PARAM 占位·unparse 得 PARAM 词非源码）·
    # 不写 op_confidence（非 operator-level·信号在 Episode reward + OutputResult.parts·metrics generate_verified 读）。
    # **反 theater ③下游读者锚**：parts 非空 ⟺ verified（未验/树缺→parts=[] 不提交产出·同 arith :2066-2070）。
    if getattr(gates, "CODE_UNPARSE_MODE", False):
        from pure_integer_ai.cognition.result.composes_unparse import unparse_composes
        from pure_integer_ai.cognition.result.ast_normalize import bodies_match
        code_items = [it for it in corpus
                      if it.modality == MODALITY_CODE and it.code_source]
        for item in code_items:
            if item.code_struct_ref is None:
                continue   # observe 未建树（理论不发生·防御）·诚实 skip（不计 total/selected/verified）
            total_tasks += 1
            selected += 1   # 树已建 = 选定（无算子择优·直接用 item observe 树·selected 语义=有可 unparse 树）
            # per-item try/except（审2 P1-2·镜像 arith v_skel=None 容错·单 item 序化异常不崩整个 run）：
            # 序化器 raise LoopClosureDefect(病态深/环) / ValueError(STORE 须 1 子·非支持形态)→诚实降级 not verified。
            try:
                unparsed = unparse_composes(graph, item.code_struct_ref)
                eq = bodies_match(unparsed, item.code_source)   # AST normalize 结构等价（Mode A 构造性验证）
            except (ValueError, RuntimeError) as _unparse_err:
                # LoopClosureDefect(RuntimeError 子) / ValueError → 序化异常·诚实 not verified（parts 空·reward=0）
                unparsed = "<unparse-failed>"
                eq = False
            if eq:
                verified += 1
            # 打包 OutputResult（反 theater ③：parts 非空 ⟺ verified·未验→parts=[]·metrics 读 parts 计 generate_verified）
            if eq:
                output = OutputResult(parts=[OutputPart(
                    unit=item.code_struct_ref, words=[unparsed])])
            else:
                output = OutputResult(parts=[])
            ep = Episode(
                episode_id=ep_id, run_id=ep_id,
                input=None, output=output,
                reward=1 if eq else 0,
                ref=item.code_struct_ref,
                terminal=TERMINAL_REACHED_SINK,
                pr_vector={},   # 代码模态 task-driven 不跑 dag_path_step·无 PR 向量（诚实·同 arith）
                judge_G5_active=False,   # unparse 验证非 judge G5 门（构造性重建 vs source·非 G5 门因子）
                judge_veto_count=0 if eq else 1,
                dead_end_count=0,
                vetoed=(not eq),
                verify_source=VERIFY_SOURCE_EXTERNAL,   # Layer0：unparse vs code_source R6 外部源（同 _run_verify_round·Mode A 构造性重建·可计 external_verified·2 审 P1-1 修）
            )
            episodes.append(ep)
            ep_id += 1
    summary = GenerateSummary(
        total_tasks=total_tasks, selected=selected, verified=verified,
        xform_verified=xform_verified, inv_verified=inv_verified,
        xform_total=xform_total, inv_total=inv_total)   # #1124 symbolic 子计数 + 分母（M-1）
    return episodes, summary

__all__ = [
    "GenerateSummary",
    "_run_task_driven_generate",
]
