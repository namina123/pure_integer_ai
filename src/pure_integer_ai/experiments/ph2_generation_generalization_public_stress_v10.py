"""GG-03 V10 owner-independent public expansion probes.

This module extends the public executable Observation surface only.  It does
not contain accepted/rejected surfaces, semantic labels, or formal evaluator
artifacts.  The builder is deterministic so a fresh V10 run root can generate
its own transport without reusing a prior family.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_public_stress import (
    build_generation_generalization_public_stress_observations,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    DialogueEpisode,
    DialogueTurn,
    GroundedEvidence,
)


PUBLIC_V10_STRESS_BUDGET = GenerationGeneralizationEvaluationBudget(
    512, 4, 4, 96, 16)

PUBLIC_V10_STRESS_CASE_IDS = (
    "gg03-v10-public-answer-two-source-v1",
    "gg03-v10-public-answer-three-source-v1",
    "gg03-v10-public-answer-two-source-long-v1",
    "gg03-v10-public-clarify-three-candidate-v1",
    "gg03-v10-public-clarify-three-candidate-dialogue-v1",
    "gg03-v10-public-clarify-three-candidate-punctuation-v1",
    "gg03-v10-public-conflict-three-source-support-v1",
    "gg03-v10-public-conflict-three-source-refute-v1",
    "gg03-v10-public-conflict-three-source-dialogue-v1",
    "gg03-v10-public-reference-surface-variant-v1",
    "gg03-v10-public-reference-numeric-variant-v1",
    "gg03-v10-public-reference-long-variant-v1",
)


class GenerationGeneralizationPublicV10StressError(ValueError):
    """公开 V10 stress 构造或 inventory 发生漂移。"""


def _base_observations(
        source_path: str | Path,
        ) -> tuple[
            GenerationGeneralizationEvaluationObservation, ...]:
    """从已公开的 V9 stress builder 取得四类无 label 基形。"""
    base = build_generation_generalization_public_stress_observations(
        source_path)
    by_id = {item.episode_id: item for item in base}
    required = (
        "gg03-public-stress-answer-novel-v1",
        "gg03-public-stress-clarify-near-context-v1",
        "gg03-public-stress-conflict-novel-v1",
        "gg03-public-stress-reference-novel-v1",
    )
    if any(item not in by_id for item in required):
        raise GenerationGeneralizationPublicV10StressError(
            "V10 公开基形缺失")
    return tuple(by_id[item] for item in required)


def _answer_multi_source(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, source_ids: tuple[str, ...], context: str,
        question_surface: str,
        ) -> GenerationGeneralizationEvaluationObservation:
    """单命题 ANSWER 同时要求多个支持来源与多 citation。"""
    original = base.question.evidence[0]
    evidence = tuple(
        replace(
            original,
            evidence_id=f"{case_id}-evidence-{ordinal}",
            source_id=source_id,
            evidence_text=f"公开 V10 来源{ordinal}：{original.claim_text}",
        )
        for ordinal, source_id in enumerate(source_ids, start=1)
    )
    plan = replace(
        base.question.answer_plan,
        citation_source_ids=source_ids,
    )
    question = replace(
        base.question,
        context_surface=context,
        question_surface=question_surface,
        evidence=evidence,
        answer_plan=plan,
    )
    dialogue = replace(
        base.dialogue,
        turns=(
            *base.dialogue.turns[:-1],
            replace(base.dialogue.turns[-1], surface=question_surface),
        ),
    )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=dialogue,
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def _clarify_three_candidate(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, context: str, question_surface: str,
        three_turn: bool = False,
        ) -> GenerationGeneralizationEvaluationObservation:
    """把公开双候选 CLARIFY 扩展为三个同 scope 候选。"""
    scope = base.question.evidence_scope_id
    candidates = (
        ("v10-north", "北侧接口需要复核"),
        ("v10-south", "南侧接口需要复核"),
        ("v10-east", "东侧接口需要复核"),
    )
    evidence = tuple(
        GroundedEvidence(
            f"{case_id}-evidence-{ordinal}", proposition_id, "src-v10-clarify",
            scope, claim, f"公开 V10 候选{ordinal}：{claim}", 1, 0,
        )
        for ordinal, (proposition_id, claim) in enumerate(candidates, start=1)
    )
    question = replace(
        base.question,
        context_surface=context,
        question_surface=question_surface,
        evidence=evidence,
    )
    if three_turn:
        turns = (
            DialogueTurn(
                1, "USER", "请保留 V10 当前资料范围。",
                (base.question.evidence_scope_id,),
            ),
            DialogueTurn(
                2, "ASSISTANT", "资料范围已保留，请继续确认对象。",
                (base.question.response_scope_id,),
            ),
            DialogueTurn(
                3, "USER", question_surface,
                (base.question.evidence_scope_id,
                 base.question.response_scope_id),
            ),
        )
        dialogue = DialogueEpisode(turns, base.dialogue.active_scope_ids)
    else:
        dialogue = replace(
            base.dialogue,
            turns=(
                *base.dialogue.turns[:-1],
                replace(base.dialogue.turns[-1], surface=question_surface),
            ),
        )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=dialogue,
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def _conflict_three_source(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, support_count: int, context: str,
        question_surface: str, three_turn: bool = False,
        ) -> GenerationGeneralizationEvaluationObservation:
    """同一 Proposition 使用三个来源，并保留 support/refute 双向 stance。"""
    original = base.question.evidence[0]
    claim = original.claim_text
    source_ids = (
        "src-v10-conflict-a", "src-v10-conflict-b", "src-v10-conflict-c")
    evidence = tuple(
        GroundedEvidence(
            f"{case_id}-evidence-{ordinal}", original.proposition_id,
            source_id, original.scope_id, claim,
            f"公开 V10 冲突来源{ordinal}：{claim}",
            1 if ordinal <= support_count else 0,
            0 if ordinal <= support_count else 1,
        )
        for ordinal, source_id in enumerate(source_ids, start=1)
    )
    question = replace(
        base.question,
        context_surface=context,
        question_surface=question_surface,
        evidence=evidence,
    )
    dialogue = replace(
        base.dialogue,
        turns=(
            *base.dialogue.turns[:-1],
            replace(base.dialogue.turns[-1], surface=question_surface),
        ),
    )
    if three_turn:
        dialogue = DialogueEpisode(
            (
                DialogueTurn(
                    1, "USER", "请保留三来源 V10 范围。",
                    (base.question.evidence_scope_id,),
                ),
                DialogueTurn(
                    2, "ASSISTANT", "范围已保留，请判断来源是否一致。",
                    (base.question.response_scope_id,),
                ),
                DialogueTurn(
                    3, "USER", question_surface,
                    (base.question.evidence_scope_id,
                     base.question.response_scope_id),
                ),
            ),
            base.dialogue.active_scope_ids,
        )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=dialogue,
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def _reference_variant(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, context: str, question_surface: str,
        surfaces: tuple[str, str],
        ) -> GenerationGeneralizationEvaluationObservation:
    """保留双命题 reference contract，扩大词面与上下文组合。"""
    reference = base.reference_course
    if reference is None:
        raise GenerationGeneralizationPublicV10StressError(
            "V10 reference 基形缺失")
    if len(surfaces) != len(reference.options):
        raise GenerationGeneralizationPublicV10StressError(
            "V10 reference surface 分母漂移")
    reference = replace(
        reference,
        options=tuple(
            replace(option, reference_surface=surface)
            for option, surface in zip(reference.options, surfaces, strict=True)
        ),
    )
    question = replace(
        base.question,
        context_surface=context,
        question_surface=question_surface,
    )
    dialogue = replace(
        base.dialogue,
        turns=(
            *base.dialogue.turns[:-1],
            replace(base.dialogue.turns[-1], surface=question_surface),
        ),
    )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=dialogue,
        reference_course=reference,
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def build_generation_generalization_public_v10_stress_observations(
        source_path: str | Path,
        ) -> tuple[GenerationGeneralizationEvaluationObservation, ...]:
    """确定性构造 V10 公开扩展探针，覆盖四路径与六 requirement。"""
    answer, clarify, conflict, reference = _base_observations(source_path)
    observations = (
        _answer_multi_source(
            answer, case_id=PUBLIC_V10_STRESS_CASE_IDS[0],
            source_ids=("src-v10-answer-a", "src-v10-answer-b"),
            context="V10 两份独立记录均覆盖同一校准结果。",
            question_surface="两份记录是否共同支持校准后的稳定结论？",
        ),
        _answer_multi_source(
            answer, case_id=PUBLIC_V10_STRESS_CASE_IDS[1],
            source_ids=("src-v10-answer-a", "src-v10-answer-b", "src-v10-answer-c"),
            context="V10 三份独立记录以不同表述确认同一设备状态。",
            question_surface="三份记录是否共同支持该状态？",
        ),
        _answer_multi_source(
            answer, case_id=PUBLIC_V10_STRESS_CASE_IDS[2],
            source_ids=("src-v10-answer-long-a", "src-v10-answer-long-b"),
            context="V10 长上下文保留多个无关字段后，两份来源仍指向同一稳定结论。"
            * 3,
            question_surface="在保留长上下文后，两份来源是否仍共同支持该结论？",
        ),
        _clarify_three_candidate(
            clarify, case_id=PUBLIC_V10_STRESS_CASE_IDS[3],
            context="V10 同一资料范围内存在三个名称相近的接口候选。",
            question_surface="请确认当前问题指向北侧、南侧还是东侧接口？",
        ),
        _clarify_three_candidate(
            clarify, case_id=PUBLIC_V10_STRESS_CASE_IDS[4],
            context="V10 三候选对象跨越三轮对话，当前范围仍保持激活。",
            question_surface="结合前述三轮范围，当前指的是哪一个接口？",
            three_turn=True,
        ),
        _clarify_three_candidate(
            clarify, case_id=PUBLIC_V10_STRESS_CASE_IDS[5],
            context="V10 记录同时出现 A/B、甲乙和编号 17 等标记。",
            question_surface="这里的接口具体指北侧、南侧还是东侧？",
        ),
        _conflict_three_source(
            conflict, case_id=PUBLIC_V10_STRESS_CASE_IDS[6],
            support_count=2,
            context="V10 三份来源中两份支持启用状态，一份记录相反状态。",
            question_surface="三份来源下当前能否确定设备状态？",
        ),
        _conflict_three_source(
            conflict, case_id=PUBLIC_V10_STRESS_CASE_IDS[7],
            support_count=1,
            context="V10 三份来源中一份支持成功，另两份记录相反结果。",
            question_surface="在来源数量不对称时，结果是否仍可唯一确定？",
        ),
        _conflict_three_source(
            conflict, case_id=PUBLIC_V10_STRESS_CASE_IDS[8],
            support_count=2,
            context="V10 三来源矛盾信息跨越多轮对话但仍处于同一范围。",
            question_surface="结合前述三轮对话，现在可以确定结果吗？",
            three_turn=True,
        ),
        _reference_variant(
            reference, case_id=PUBLIC_V10_STRESS_CASE_IDS[9],
            context="V10 档案同时记录设施年代与入口位置，指代候选词面更长。",
            question_surface="请连续说明该公开设施的建成年代和入口位置。",
            surfaces=("该公开设施", "这座设施"),
        ),
        _reference_variant(
            reference, case_id=PUBLIC_V10_STRESS_CASE_IDS[10],
            context="V10 编号 17 档案保留容量与开放比例两个属性。",
            question_surface="请说明编号 17 设施的容量和开放比例。",
            surfaces=("编号 17 的设施", "该编号设施"),
        ),
        _reference_variant(
            reference, case_id=PUBLIC_V10_STRESS_CASE_IDS[11],
            context="V10 长文本保留多个属性后仍要求按原顺序连续回答。" * 3,
            question_surface="请完整说明该设施的两个属性并保持原顺序。",
            surfaces=("该公开设施本体", "前述设施"),
        ),
    )
    result = tuple(sorted(observations, key=lambda item: item.stable_key()))
    if (len(result) != len(PUBLIC_V10_STRESS_CASE_IDS)
            or {item.episode_id for item in result}
            != set(PUBLIC_V10_STRESS_CASE_IDS)
            or any(item.resource_budget != PUBLIC_V10_STRESS_BUDGET
                   for item in result)):
        raise GenerationGeneralizationPublicV10StressError(
            "V10 public stress inventory 漂移")
    return result


def generation_generalization_public_v10_stress_bytes(
        observations: tuple[GenerationGeneralizationEvaluationObservation, ...],
        ) -> bytes:
    """返回按 stable key 排序且不含 label surface 的 canonical JSONL。"""
    expected = tuple(sorted(observations, key=lambda item: item.stable_key()))
    if observations != expected or len(observations) != len(
            PUBLIC_V10_STRESS_CASE_IDS):
        raise GenerationGeneralizationPublicV10StressError(
            "V10 public stress 未冻结排序或分母")
    return b"".join(
        canonical_json_line(item.to_dict()) for item in observations)


__all__ = [
    "PUBLIC_V10_STRESS_BUDGET",
    "PUBLIC_V10_STRESS_CASE_IDS",
    "GenerationGeneralizationPublicV10StressError",
    "build_generation_generalization_public_v10_stress_observations",
    "generation_generalization_public_v10_stress_bytes",
]
