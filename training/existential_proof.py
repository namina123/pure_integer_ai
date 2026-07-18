"""training.existential_proof — 存在量化验序器（A1·STEP6·语言域形式 cue·构造性**验证**层·EXTERNAL·镜像刀 C）。

镜像 universal_proof_fn_factory 范式·但验 **存在量化 X∩Y≠∅**·数据来自 ConceptNet **外部源**（非系统自产）
→ 构造性**验证**（verify_source=EXTERNAL·同刀 C·Layer0 external_verified 计入·反 SELF_PRODUCED 全自产不准停）。

**★双向祖先（关键·非单纯 reversed·A1 与 ∀ 的核心差异）**：
  ∃ "有的 X 是 Y" = ∃x: X(x) ∧ Y(x) = X∩Y ≠ ∅·类层真 iff X⊆Y OR Y⊆X（其一为子集→小类实例即 X∩Y 样本）。
    - "有的鸟是企鹅"（企鹅⊆鸟 → 鸟 ∈ ancestors(企鹅) → reversed 命中）
    - "有的鸟是动物"（鸟⊆动物 → 动物 ∈ ancestors(鸟) → forward 命中·同 ∀）
  单向 reversed（仅 child ∈ ancestors(parent)）会**误证伪**"有的鸟是动物"（鸟∉ancestors(动物)·两分类→falsified·
  但"有的鸟是动物"为真）→ 必须**双向 OR**：`parent ∈ ancestors(child) OR child ∈ ancestors(parent)`。
  双向皆不命中 + 两分类概念 → falsified（X∩Y=∅·如"有的鸟是植物"）；非分类 → can't-verify（None·守 #479）。

机制（Option A·量化声明不入图·闭包传外部祖先图·同刀 C·防 #355 provenance 冲突 + emergence 污染）：
  - caller（_run_existential_verify_round）resolve 段 token→ConceptRef（concept_index.lookup·镜像 ∀）
    + 构建**外部祖先图** build_isa_ancestor_map_external（仅 ConceptNet EPI_STRUCTURED 边·反 single-source theater）
    → factory 闭包捕获 ancestor_map + claims（(child_ref, parent_ref) 对）
  - 内层逐声明查 `parent ∈ ancestors(child) OR child ∈ ancestors(parent)`·三值判定

返值（守 SelfProofFn 协议·judge.py:49·三参数 output/dag_path/graph 占位·内层只用闭包）：
  1 = 全声明 verified（ConceptNet 外部断言 X∩Y≠∅·真构造性验证通过）
  0 = 任一声明 falsified（两分类概念·ConceptNet 否认 X∩Y·双向皆不命中·外部证伪）
  None = 任一声明 can't-verify 且无 falsified（child/parent 非分类概念·外部源不足·诚实弃权·守属性 ∃ #479 墙）

**三值诚实逻辑（防误证伪属性 ∃·调和 #479 墙·核心反 theater·同 ∀）**：
  - parent ∈ ancestors(child) OR child ∈ ancestors(parent) → verified（ConceptNet 确认 X∩Y≠∅·双向其一子集）
  - parent ∈ ext_concepts AND child ∈ ext_concepts AND 双向皆不命中 → falsified
    （两分类概念·ConceptNet 否认 X⊆Y 且 Y⊆X → X∩Y=∅·外部证伪·如"有的鸟都是植物"）
  - parent ∉ ext_concepts 或 child ∉ ext_concepts → can't-verify（**非分类声明不准用分类源判**·
    如"有的鸟会飞"·会飞∉ext_concepts→None 非证伪·守属性 ∃ #479 墙）
  ext_concepts = 外部图全 key ∪ 全祖先集（ConceptNet IsA 边涉及的概念·纯结构查询·判别靠外部源涌现非写死）

**诚实分层（构造性验证 ≠ truth·#479 墙守·同 ∀）**：
  - 构造性验证 ✅：IS_A 闭包查询确定性可执行 AND 数据来自 ConceptNet 外部 R6 独立源（非 cue 自产）
    → 真构造性验证（Mode B 范式·expected 独立源对齐）·Layer0 标 EXTERNAL（可驱动停止决策）
  - truth ❌：ConceptNet 可错·"鸟 IsA 动物"对齐外部断言 ≠ 命题真（语义内核层 #479 墙·stable≠correct）
  - **严格实例存在 defer**：双向祖先验类层 X∩Y≠∅（子集关系）·严格实例非空=世界态 #479 墙 defer
    （ConceptNet 类图非实例存在断言·stable≠correct）

依赖方向：training(L7)→cognition(L5) algorithm/storage 向下·lint 允许（镜像 universal_proof.py）。
judge(cognition L5) 不 import training·existential_proof_fn 在 training 接线层建（守解耦）。

铁律：纯整数（ConceptRef=int 二元组·集合运算·无浮点）/ 确定性（集合查询·bit-identical）/ fail-loud
      （falsified→0·can't-verify→None 非静默放行）/ stable≠correct / 永不接 reward（self_proof_fn 通道）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.cognition.result.judge import SelfProofFn
from pure_integer_ai.cognition.shared.types import ConceptRef


def existential_proof_fn_factory(*, ancestor_map: dict[ConceptRef, set[ConceptRef]],
                                  claims: list[tuple[ConceptRef, ConceptRef]]) -> SelfProofFn:
    """造存在量化验序器 fn（闭包捕获 ancestor_map + claims·镜像 universal_proof_fn_factory·双向祖先）。

    ancestor_map : 外部 ConceptNet 祖先图 {child: {ancestors}}（build_isa_ancestor_map_external·
                   仅 source=SOURCE_CONCEPTNET 边·反 single-source theater·caller 构建·run-scoped）。
    claims : 存在量化声明 list·每元 (child_ref, parent_ref)·ConceptRef 二元组（caller resolve 自
             segments.existential_claims token index 对·concept_index.lookup·未概念化项 caller 已跳）。

    返 SelfProofFn(output, dag_path, graph) -> int|None：
      1（全 verified·ConceptNet 外部确认 X∩Y≠∅·双向其一子集）/ 0（任一 falsified·两分类概念 ConceptNet
      否认 X∩Y·双向皆不命中）/ None（任一 can't-verify 且无 falsified·非分类声明诚实弃权）。
      _run_existential_verify_round 直调（绕 judge·reward=1 iff r==1·r==0 falsified·r is None 弃权无 episode）。

    **构造性验证层**（诚实·EXTERNAL·同 ∀·非 SELF_PRODUCED 检查）：ConceptNet 外部源·非系统自产。
    **双向祖先**：verified iff `parent ∈ ancestors(child) OR child ∈ ancestors(parent)`（X⊆Y OR Y⊆X）。
    """
    # 防御性拷贝（同 universal P2-3 范式）：闭包按引用捕获·防 caller 后续 mutation 改 fn 行为。
    ancestor_map = {k: set(v) for k, v in ancestor_map.items()}
    claims = list(claims)
    # ext_concepts = 外部图涉及的全部概念（key=有祖先的后代 ∪ 全祖先）·判别"是否分类概念"·纯结构涌现非写死
    ext_concepts: set[ConceptRef] = set(ancestor_map.keys())
    for _anc_set in ancestor_map.values():
        ext_concepts |= _anc_set

    def existential_proof_fn(output: Any, dag_path: Any, graph: Any) -> int | None:
        # output/dag_path/graph 占位（守 SelfProofFn 协议三参数·内层只用闭包·同 universal/time_seq）
        if not claims:
            return None   # 无声明 → vacate（诚实退场·非 pass·非 theater）
        saw_cant_verify = False
        for (child, parent) in claims:
            # ★双向祖先：X⊆Y (parent ∈ ancestors(child)) OR Y⊆X (child ∈ ancestors(parent)) → X∩Y≠∅ verified
            if (parent in ancestor_map.get(child, set())
                    or child in ancestor_map.get(parent, set())):
                continue   # verified（ConceptNet 外部断言 X∩Y≠∅·双向其一子集·真构造性验证）
            # 未 verified：判 falsified 还是 can't-verify（核心反 theater·守属性 ∃ #479 墙·同 ∀）
            if parent in ext_concepts and child in ext_concepts:
                return 0   # 两分类概念·ConceptNet 否认 X⊆Y 且 Y⊆X → X∩Y=∅·外部证伪（如 鸟∩植物=∅）
            # parent 或 child 非分类概念（属性/谓词/未知·如 会飞∉ext_concepts）→ can't-verify（不准用分类源判）
            saw_cant_verify = True
        return None if saw_cant_verify else 1   # 全 verified→1 / 有 can't-verify→None（弃权·非证伪）

    return existential_proof_fn
