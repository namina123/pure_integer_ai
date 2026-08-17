"""GG-03 public diagnostics for the V10 capability failures.

The diagnostic inputs are label-free and intentionally narrow.  They isolate
candidate-count expansion and four-source conflict stance combinations before
any new code identity or formal family is considered.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_public_stress_v10 import (
    PUBLIC_V10_STRESS_BUDGET,
    build_generation_generalization_public_v10_stress_observations,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    DialogueEpisode,
    DialogueTurn,
    GroundedEvidence,
)


PUBLIC_V11_DIAGNOSTIC_CASE_IDS = (
    "gg03-v11-diagnostic-clarify-four-candidate-v1",
    "gg03-v11-diagnostic-clarify-four-candidate-multi-source-v1",
    "gg03-v11-diagnostic-conflict-four-source-mixed-v1",
    "gg03-v11-diagnostic-conflict-four-source-asymmetric-v1",
    "gg03-v11-diagnostic-clarify-permuted-evidence-v1",
    "gg03-v11-diagnostic-clarify-repeated-source-v1",
    "gg03-v11-diagnostic-conflict-repeated-source-opposing-v1",
    "gg03-v11-diagnostic-conflict-reordered-mixed-v1",
)


class GenerationGeneralizationPublicV11DiagnosticError(ValueError):
    """公开 V11 diagnostic 构造发生漂移。"""


def _base(
        source_path: str | Path,
        ) -> tuple[
            GenerationGeneralizationEvaluationObservation,
            GenerationGeneralizationEvaluationObservation,
        ]:
    observations = build_generation_generalization_public_v10_stress_observations(
        source_path)
    by_id = {item.episode_id: item for item in observations}
    try:
        clarify = by_id["gg03-v10-public-clarify-three-candidate-v1"]
        conflict = by_id["gg03-v10-public-conflict-three-source-support-v1"]
    except KeyError as error:
        raise GenerationGeneralizationPublicV11DiagnosticError(
            "V11 public diagnostic 基形缺失") from error
    return clarify, conflict


def _three_turn_dialogue(
        observation: GenerationGeneralizationEvaluationObservation,
        question_surface: str,
        ) -> DialogueEpisode:
    """给诊断输入增加确定的三轮 scope 形状。"""
    question = observation.question
    return DialogueEpisode(
        (
            DialogueTurn(
                1, "USER", "请保留 V11 当前资料范围。",
                (question.evidence_scope_id,),
            ),
            DialogueTurn(
                2, "ASSISTANT", "资料范围已保留，请继续限定对象。",
                (question.response_scope_id,),
            ),
            DialogueTurn(
                3, "USER", question_surface,
                (question.evidence_scope_id, question.response_scope_id),
            ),
        ),
        observation.dialogue.active_scope_ids,
    )


def _clarify(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, source_ids: tuple[str, ...], question_surface: str,
        ) -> GenerationGeneralizationEvaluationObservation:
    """构造四个候选，分别覆盖同源和多源 competition 形状。"""
    if len(source_ids) != 4:
        raise GenerationGeneralizationPublicV11DiagnosticError(
            "V11 CLARIFY source 分母非法")
    scope = base.question.evidence_scope_id
    candidates = (
        ("v11-clarify-north", "北侧接口需要复核"),
        ("v11-clarify-south", "南侧接口需要复核"),
        ("v11-clarify-east", "东侧接口需要复核"),
        ("v11-clarify-west", "西侧接口需要复核"),
    )
    evidence = tuple(
        GroundedEvidence(
            f"{case_id}-evidence-{ordinal}", proposition_id, source_id,
            scope, claim, f"公开 V11 候选{ordinal}：{claim}", 1, 0,
        )
        for ordinal, ((proposition_id, claim), source_id)
        in enumerate(zip(candidates, source_ids, strict=True), start=1)
    )
    question = replace(
        base.question,
        context_surface="V11 四候选对象处于同一资料范围，需保留三轮限定。",
        question_surface=question_surface,
        evidence=evidence,
    )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=_three_turn_dialogue(base, question_surface),
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def _conflict(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, stances: tuple[tuple[int, int], ...],
        question_surface: str,
        ) -> GenerationGeneralizationEvaluationObservation:
    """构造四来源 conflict，允许一个 source 同时承担双 stance。"""
    if len(stances) != 4:
        raise GenerationGeneralizationPublicV11DiagnosticError(
            "V11 CONFLICT stance 分母非法")
    original = base.question.evidence[0]
    source_ids = (
        f"{case_id}-source-a", f"{case_id}-source-b",
        f"{case_id}-source-c", f"{case_id}-source-d",
    )
    evidence = tuple(
        GroundedEvidence(
            f"{case_id}-evidence-{ordinal}", original.proposition_id,
            source_id, original.scope_id, original.claim_text,
            f"公开 V11 冲突来源{ordinal}：{original.claim_text}",
            support, refute,
        )
        for ordinal, (source_id, (support, refute))
        in enumerate(zip(source_ids, stances, strict=True), start=1)
    )
    question = replace(
        base.question,
        context_surface="V11 四份来源对同一命题形成更复杂的 support/refute 组合。",
        question_surface=question_surface,
        evidence=evidence,
    )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=_three_turn_dialogue(base, question_surface),
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def _rebound_evidence(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, source_ids: tuple[str, ...],
        stances: tuple[tuple[int, int], ...],
        reverse: bool = False,
        ) -> tuple[GroundedEvidence, ...]:
    """重排四条 label-free Evidence，同时保留 proposition/scope 归属。"""
    original = tuple(base.question.evidence)
    if len(source_ids) != 4 or len(stances) != 4 or len(original) != 4:
        raise GenerationGeneralizationPublicV11DiagnosticError(
            "V11 Evidence 重排分母非法")
    rows = tuple(
        (item, source_id, stance)
        for item, source_id, stance
        in zip(original, source_ids, stances, strict=True)
    )
    if reverse:
        rows = tuple(reversed(rows))
    return tuple(
        GroundedEvidence(
            f"{case_id}-evidence-{ordinal}", item.proposition_id,
            source_id, item.scope_id, item.claim_text,
            f"公开 V11 重排来源{ordinal}：{item.claim_text}",
            support, refute,
        )
        for ordinal, (item, source_id, (support, refute))
        in enumerate(rows, start=1)
    )


def _reordered(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, question_surface: str,
        source_ids: tuple[str, ...],
        stances: tuple[tuple[int, int], ...],
        reverse: bool,
        ) -> GenerationGeneralizationEvaluationObservation:
    """构造证据顺序和来源顺序均可能脱离输入顺序的诊断。"""
    question = replace(
        base.question,
        context_surface="V11 证据行顺序与来源归属刻意错开，但 scope 保持一致。",
        question_surface=question_surface,
        evidence=_rebound_evidence(
            base, case_id=case_id, source_ids=source_ids,
            stances=stances, reverse=reverse),
    )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=_three_turn_dialogue(base, question_surface),
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def build_generation_generalization_public_v11_diagnostic_observations(
        source_path: str | Path,
        ) -> tuple[GenerationGeneralizationEvaluationObservation, ...]:
    """返回八条只用于公开 actual runner 的诊断 Observation。"""
    clarify, conflict = _base(source_path)
    clarify_four = _clarify(
        clarify,
        case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[0],
        source_ids=("src-v11-clarify",) * 4,
        question_surface="四个候选中当前具体指哪一个接口？",
    )
    conflict_four = _conflict(
        conflict,
        case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[2],
        stances=((1, 1), (1, 0), (0, 1), (0, 1)),
        question_surface="四份来源中有一份同时记录两种方向，结果能否确定？",
    )
    result = (
        clarify_four,
        _clarify(
            clarify,
            case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[1],
            source_ids=(
                "src-v11-clarify-a", "src-v11-clarify-b",
                "src-v11-clarify-c", "src-v11-clarify-d",
            ),
            question_surface="结合三轮范围，四个来源候选中具体指哪一个接口？",
        ),
        conflict_four,
        _conflict(
            conflict,
            case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[3],
            stances=((1, 0), (0, 1), (0, 1), (0, 1)),
            question_surface="一份支持、三份反驳时，结果能否唯一确定？",
        ),
        _reordered(
            clarify_four,
            case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[4],
            source_ids=(
                "src-v11-clarify-z", "src-v11-clarify-a",
                "src-v11-clarify-y", "src-v11-clarify-b",
            ),
            stances=((1, 0),) * 4,
            reverse=True,
            question_surface="证据顺序被打乱后，四个候选仍指向哪一个接口？",
        ),
        _reordered(
            clarify_four,
            case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[5],
            source_ids=(
                "src-v11-clarify-shared-a", "src-v11-clarify-shared-a",
                "src-v11-clarify-shared-b", "src-v11-clarify-shared-b",
            ),
            stances=((1, 0),) * 4,
            reverse=False,
            question_surface="同一来源重复记录四个候选时，当前具体指哪一个接口？",
        ),
        _reordered(
            conflict_four,
            case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[6],
            source_ids=(
                "src-v11-conflict-repeat-a", "src-v11-conflict-repeat-a",
                "src-v11-conflict-repeat-b", "src-v11-conflict-repeat-b",
            ),
            stances=((1, 0), (0, 1), (1, 0), (0, 1)),
            reverse=True,
            question_surface="两份来源各自留下相反记录时，结果能否确定？",
        ),
        _reordered(
            conflict_four,
            case_id=PUBLIC_V11_DIAGNOSTIC_CASE_IDS[7],
            source_ids=(
                "src-v11-conflict-d", "src-v11-conflict-a",
                "src-v11-conflict-c", "src-v11-conflict-b",
            ),
            stances=((1, 1), (0, 1), (1, 0), (0, 1)),
            reverse=True,
            question_surface="混合立场且来源顺序变化时，能否保留完整不确定性？",
        ),
    )
    result = tuple(sorted(result, key=lambda item: item.stable_key()))
    if ({item.episode_id for item in result}
            != set(PUBLIC_V11_DIAGNOSTIC_CASE_IDS)):
        raise GenerationGeneralizationPublicV11DiagnosticError(
            "V11 diagnostic inventory 漂移")
    return result


__all__ = [
    "PUBLIC_V11_DIAGNOSTIC_CASE_IDS",
    "GenerationGeneralizationPublicV11DiagnosticError",
    "build_generation_generalization_public_v11_diagnostic_observations",
]
