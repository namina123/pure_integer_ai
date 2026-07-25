"""训练结果中可被领域 runtime 共享的稳定汇总类型。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveryRouteStats,
    StructureTallyStats,
)
from pure_integer_ai.experiments.lang_structure_metrics import (
    LanguageStructureStateStats,
)


@dataclass
class GeneralizationSummary:
    """序列3-min 验证半闭环汇总（识别 → vm_proof 验泛化·§8.7 反 theater + 学到能力证据）。

    发现骨架从发现集学到 → 识别 held-out 新输入（READ）→ vm_proof 独立验骨架绑参复现新输入值。
    verified/total_held_out = 泛化率（×1000·学到的能力覆盖多少 held-out 新输入·直接量化"学到能力"）。

    反 theater：识别 = 结构对齐（_align_walk）·vm_proof = VM 执行比对（execute_composes_value）·两路独立
    计算。**诚实定位**（对抗审计·勿过判）：对正确识别·骨架与输入结构同构 → 同值是构造性预期（非惊奇交叉验证）·
    vm_proof 真"牙"=抓获 PARAM 阅读序错位 / 编译发散 / shape 漏判结构异配（probe：SUB 错参 -47≠43 不 verified）。
    重执行本身 = 真 READ+应用消费（非 theater·非死写·识别产物 recognitions 现被 vm_proof 真消费·解 terminal 边界）。
    生成侧洗净循环反馈半闭环（§8.7-洗·2026-07-03 done）：本函数验结果现写算子置信度（op_confidence
    sn/tn/strength）→ recognize_operators 择优读（滤非泛化算子=洗净）·解 recognitions terminal·反 theater 半环。
    生成侧全环（generate.py 读置信度·OutputModel 路径填槽消费骨架）= 独立大切片 defer（须独立设计 pass）。
    **【2026-06-30 证伪·3 对抗智能体】** 字面机制当前架构不可达（generate.py L6→execute_composes_value L7
    向上违单向 + STRUCT_BIND 跨模态桥 VF defer 零 caller + 算术骨架无语言 surface）·算子域闭合环已由本半环
    （recognize↔verify+vm_proof+op_confidence）完成·vm_proof 是骨架执行值真消费者。"生成侧全环"对算术模态
    是伪需求·须 STRUCT_BIND VF 落地后才有意义。详见 doc/重来_结构发现设计补充.md §8.7-洗-证伪。
    """
    total_held_out: int = 0   # 留出的 held-out 新输入总数（识别候选池·= len(recognize_roots)）
    recognized: int = 0       # 命中已学骨架的 held-out 数（recognize_operators 产）
    verified: int = 0         # 命中中 vm_proof 验过（骨架绑参==输入值·两路独立）的数
    expected_verified: int = 0   # S7 相0 钥匙③：教师标定比对命中数（recognize 命中骨架 ref==item.expected_skeleton·断奶前教师路径·POST 退场·非 vm_proof）
    routing_stats: DiscoveryRouteStats | None = None
    tally_stats: StructureTallyStats | None = None
    structure_state: LanguageStructureStateStats | None = None

    @property
    def rate_permille(self) -> int:
        """泛化率 ×1000（verified / total_held_out·算术域 vm_proof 口径·total=0→0·纯整数无浮点）。"""
        return (self.verified * 1000) // max(self.total_held_out, 1)

    @property
    def lang_rate_permille(self) -> int:
        """S7 相1 钥匙③：语言含义命中率 ×1000（recognized / total_held_out·渐近判据·total=0→0·纯整数）。

        区别算术域 rate_permille（verified vm_proof 口径）：语言不可 vm_proof（钥匙③墙）·相1 用 recognized
        （recognize 结构对齐命中数）。渐近判据·非闭式真理。
        **消费者诚实边界**：相1 测试断言 + result.lang_generalization 字段暴露 + capability_exam.project_lang_measures
        observability 报告读取（#1041 构造③·单向 tap·**非闭环消费者**·无 decision/threshold/feedback）。recognize 择优读
        op_confidence（rate·已落）属**相0 半环**消费者·非本 property 消费者。本 property 与相0 op_confidence 半环当前
        **无机制耦合**。下游闭环消费者（metrics 显式读取 / weaning_ready 判据接线）仍 defer（相1 计算器先行·非纸面闭合）。
        """
        return (self.recognized * 1000) // max(self.total_held_out, 1)


__all__ = ["GeneralizationSummary"]
