"""training.universal_proof — 全称量化验序器（刀 C·语言域形式 cue·构造性**验证**层·首个 EXTERNAL）。

镜像 numeric_proof_fn_factory / time_seq_proof_fn_factory 范式·但验 **全称量化内涵分类子集 X⊆Y**·
数据来自 ConceptNet **外部源**（非系统自产）→ 构造性**验证**（verify_source=EXTERNAL·刀 A/B 是构造性检查
SELF_PRODUCED·刀 C 升验证·Layer0 external_verified 首个语言域计入）。

机制（Option A·量化声明不入图·闭包传外部祖先图·同刀 A/B 闭包非图范式）：
  - caller（_run_universal_verify_round）resolve 段 token→ConceptRef（concept_index.lookup·镜像 time_seq）
    + 构建**外部祖先图** build_isa_ancestor_map_external（仅 ConceptNet EPI_STRUCTURED 边·反 single-source theater）
    → factory 闭包捕获 ancestor_map + claims（(child_ref, parent_ref) 对）
  - 内层逐声明查 parent ∈ ancestor_map.get(child, set())·三值判定（verified/falsified/can't-verify）

返值（守 SelfProofFn 协议·judge.py:49·三参数 output/dag_path/graph 占位·内层只用闭包）：
  1 = 全声明 verified（ConceptNet 外部断言 X⊆Y·真构造性验证通过）
  0 = 任一声明 falsified（child/parent 均分类概念·ConceptNet 不认此子集·外部证伪）
  None = 任一声明 can't-verify 且无 falsified（child/parent 非分类概念·外部源不足·诚实弃权·守属性全称 G5b #479 墙）

**三值诚实逻辑（防误证伪属性全称·调和 G5b #479 墙·核心反 theater）**：
  - parent ∈ ancestors(child) → verified（ConceptNet 确认子集）
  - parent ∈ ext_concepts AND child ∈ ext_concepts AND parent ∉ ancestors(child) → falsified
    （两分类概念·ConceptNet 否认子集·外部证伪·如"所有鸟都是植物"）
  - parent ∉ ext_concepts 或 child ∉ ext_concepts → can't-verify（**非分类声明不准用分类源判**·
    如"所有鸟都会飞"·会飞∉ext_concepts→None 非证伪·守属性全称 #479 墙）
  ext_concepts = 外部图全 key ∪ 全祖先集（ConceptNet IsA 边涉及的概念·纯结构查询·判别靠外部源涌现非写死）

**诚实分层（构造性验证 ≠ truth·#479 墙守）**：
  - 构造性验证 ✅：IS_A 闭包查询确定性可执行 AND 数据来自 ConceptNet 外部 R6 独立源（非 cue 自产）
    → 真构造性验证（Mode B 范式·expected 独立源对齐）·Layer0 标 EXTERNAL（可驱动停止决策）
  - truth ❌：ConceptNet 可错·"鸟 IsA 动物"对齐外部断言 ≠ 命题真（语义内核层 #479 墙·stable≠correct）
  - **刀 C ≠ G5b 实现**：G5b 撞墙的是外延属性全称（"所有鸟都会飞"·遍历属性集+世界模型·defer Mode B）·
    刀 C 验内涵分类子集全称（"所有鸟都是动物"=X⊆Y·分类闭包·非墙）·属性全称子域三值 None 诚实弃权守墙·详
    doc/重来_刀C量化cue设计_2026-07-08.md §六b。

依赖方向：training(L7)→cognition(L5) algorithm/storage 向下·lint 允许（镜像 time_seq/numeric_proof.py）。
judge(cognition L5) 不 import training·universal_proof_fn 在 training 接线层建（守解耦）。

铁律：纯整数（ConceptRef=int 二元组·集合运算·无浮点）/ 确定性（集合查询·bit-identical）/ fail-loud
      （falsified→0·can't-verify→None 非静默放行）/ stable≠correct / 永不接 reward（self_proof_fn 通道）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.cognition.result.judge import SelfProofFn
from pure_integer_ai.cognition.shared.types import ConceptRef


def universal_proof_fn_factory(*, ancestor_map: dict[ConceptRef, set[ConceptRef]],
                               claims: list[tuple[ConceptRef, ConceptRef]]) -> SelfProofFn:
    """造全称量化验序器 fn（闭包捕获 ancestor_map + claims·镜像 numeric_proof_fn_factory）。

    ancestor_map : 外部 ConceptNet 祖先图 {child: {ancestors}}（build_isa_ancestor_map_external·
                   仅 source=SOURCE_CONCEPTNET 边·反 single-source theater·caller 构建·run-scoped）。
    claims : 全称量化声明 list·每元 (child_ref, parent_ref)·ConceptRef 二元组（caller resolve 自
             segments.universal_claims token index 对·concept_index.lookup·未概念化项 caller 已跳）。

    返 SelfProofFn(output, dag_path, graph) -> int|None：
      1（全 verified·ConceptNet 外部确认 X⊆Y）/ 0（任一 falsified·两分类概念 ConceptNet 否认子集）/
      None（任一 can't-verify 且无 falsified·非分类声明诚实弃权）。
      _run_universal_verify_round 直调（绕 judge·reward=1 iff r==1·r==0 falsified·r is None 弃权无 episode）。

    **构造性验证层**（诚实·首个 EXTERNAL·非 SELF_PRODUCED 检查）：ConceptNet 外部源·非系统自产。
    """
    # 防御性拷贝（同 numeric P2-3 范式）：闭包按引用捕获·防 caller 后续 mutation 改 fn 行为。
    # ancestor_map 深拷贝值集（dict 浅拷贝键 + set 拷贝值·防外部 dict/set mutation）·claims 浅拷贝（tuple 不可变）。
    ancestor_map = {k: set(v) for k, v in ancestor_map.items()}
    claims = list(claims)
    # ext_concepts = 外部图涉及的全部概念（key=有祖先的后代 ∪ 全祖先）·判别"是否分类概念"·纯结构涌现非写死
    ext_concepts: set[ConceptRef] = set(ancestor_map.keys())
    for _anc_set in ancestor_map.values():
        ext_concepts |= _anc_set

    def universal_proof_fn(output: Any, dag_path: Any, graph: Any) -> int | None:
        # output/dag_path/graph 占位（守 SelfProofFn 协议三参数·内层只用闭包·同 numeric/time_seq）
        if not claims:
            return None   # 无声明 → vacate（诚实退场·非 pass·非 theater）
        saw_cant_verify = False
        for (child, parent) in claims:
            if parent in ancestor_map.get(child, set()):
                continue   # verified（ConceptNet 外部断言 child⊆parent·真构造性验证）
            # 未 verified：判 falsified 还是 can't-verify（核心反 theater·守属性全称 #479 墙）
            if parent in ext_concepts and child in ext_concepts:
                return 0   # 两分类概念·ConceptNet 否认 child⊆parent → 外部证伪（如 鸟⊆植物）
            # parent 或 child 非分类概念（属性/谓词/未知·如 会飞∉ext_concepts）→ can't-verify（不准用分类源判）
            saw_cant_verify = True
        return None if saw_cant_verify else 1   # 全 verified→1 / 有 can't-verify→None（弃权·非证伪）

    return universal_proof_fn
