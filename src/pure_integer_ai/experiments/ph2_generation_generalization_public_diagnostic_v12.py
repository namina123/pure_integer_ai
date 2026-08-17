"""GG-03 V12 public matrix for in-contract candidate and scope variation.

V12 stays inside the current GG03 formal contract.  It deliberately varies
CLARIFY candidate cardinality and dialogue scope history, while CONFLICT
continues to use one proposition with multiple source rows.  No evaluator
surface, private label, or formal-family artifact is authored here.
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


PUBLIC_V12_DIAGNOSTIC_CASE_IDS = (
    "gg03-v12-diagnostic-clarify-two-candidate-v1",
    "gg03-v12-diagnostic-clarify-three-candidate-multi-source-v1",
    "gg03-v12-diagnostic-clarify-four-candidate-long-scope-v1",
    "gg03-v12-diagnostic-clarify-four-candidate-scope-trace-v1",
    "gg03-v12-diagnostic-conflict-four-source-support-v1",
    "gg03-v12-diagnostic-conflict-four-source-refute-v1",
    "gg03-v12-diagnostic-conflict-four-source-mixed-scope-v1",
    "gg03-v12-diagnostic-conflict-four-source-asymmetric-scope-v1",
)


class GenerationGeneralizationPublicV12DiagnosticError(ValueError):
    """公开 V12 diagnostic 构造发生漂移。"""


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
        return (
            by_id["gg03-v10-public-clarify-three-candidate-v1"],
            by_id["gg03-v10-public-conflict-three-source-support-v1"],
        )
    except KeyError as error:
        raise GenerationGeneralizationPublicV12DiagnosticError(
            "V12 public diagnostic 基形缺失") from error


def _dialogue(
        observation: GenerationGeneralizationEvaluationObservation,
        question_surface: str,
        *,
        long_scope_trace: bool,
        ) -> DialogueEpisode:
    """构造可重复的 scope 历史，最后一轮仍精确绑定当前问题。"""
    question = observation.question
    if not long_scope_trace:
        turns = (
            DialogueTurn(
                1, "USER", "请保留 V12 当前资料范围。",
                (question.evidence_scope_id,),
            ),
            DialogueTurn(
                2, "ASSISTANT", "范围已保留，请继续限定对象。",
                (question.response_scope_id,),
            ),
            DialogueTurn(
                3, "USER", question_surface,
                (question.evidence_scope_id, question.response_scope_id),
            ),
        )
    else:
        turns = (
            DialogueTurn(1, "USER", "先记录旧资料范围。", (901,)),
            DialogueTurn(2, "ASSISTANT", "旧范围已记录。", (902,)),
            DialogueTurn(3, "USER", "再切换到当前证据范围。", (903,)),
            DialogueTurn(4, "ASSISTANT", "当前范围已切换。", (
                question.response_scope_id,),
            ),
            DialogueTurn(
                5, "USER", question_surface,
                (question.evidence_scope_id, question.response_scope_id),
            ),
        )
    return DialogueEpisode(
        turns,
        tuple(dict.fromkeys((
            *observation.dialogue.active_scope_ids,
            901, 902, 903,
        ))),
    )


def _clarify(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, candidate_count: int, question_surface: str,
        source_ids: tuple[str, ...], long_scope_trace: bool,
        ) -> GenerationGeneralizationEvaluationObservation:
    """构造 2/3/4 个无冲突可回答候选。"""
    if candidate_count not in {2, 3, 4} or len(source_ids) != candidate_count:
        raise GenerationGeneralizationPublicV12DiagnosticError(
            "V12 CLARIFY candidate/source 分母非法")
    names = (
        ("v12-north", "北侧接口需要复核"),
        ("v12-south", "南侧接口需要复核"),
        ("v12-east", "东侧接口需要复核"),
        ("v12-west", "西侧接口需要复核"),
    )[:candidate_count]
    scope = base.question.evidence_scope_id
    evidence = tuple(
        GroundedEvidence(
            f"{case_id}-evidence-{ordinal}", proposition_id, source_id,
            scope, claim, f"公开 V12 候选{ordinal}：{claim}", 1, 0,
        )
        for ordinal, ((proposition_id, claim), source_id)
        in enumerate(zip(names, source_ids, strict=True), start=1)
    )
    question = replace(
        base.question,
        context_surface=(
            "V12 候选数量、来源归属和历史 scope 均显式保留。"
            * (3 if long_scope_trace else 1)),
        question_surface=question_surface,
        evidence=evidence,
    )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=_dialogue(
            base, question_surface, long_scope_trace=long_scope_trace),
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def _conflict(
        base: GenerationGeneralizationEvaluationObservation,
        *, case_id: str, stances: tuple[tuple[int, int], ...],
        question_surface: str, long_scope_trace: bool,
        ) -> GenerationGeneralizationEvaluationObservation:
    """构造当前合同内的单 proposition、多来源 CONFLICT。"""
    if len(stances) != 4:
        raise GenerationGeneralizationPublicV12DiagnosticError(
            "V12 CONFLICT stance 分母非法")
    original = base.question.evidence[0]
    source_ids = tuple(f"{case_id}-source-{letter}" for letter in "abcd")
    evidence = tuple(
        GroundedEvidence(
            f"{case_id}-evidence-{ordinal}", original.proposition_id,
            source_id, original.scope_id, original.claim_text,
            f"公开 V12 冲突来源{ordinal}：{original.claim_text}",
            support, refute,
        )
        for ordinal, (source_id, (support, refute))
        in enumerate(zip(source_ids, stances, strict=True), start=1)
    )
    question = replace(
        base.question,
        context_surface=(
            "V12 单一命题保留四来源 support/refute 与 scope 历史。"
            * (3 if long_scope_trace else 1)),
        question_surface=question_surface,
        evidence=evidence,
    )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=_dialogue(
            base, question_surface, long_scope_trace=long_scope_trace),
        resource_budget=PUBLIC_V10_STRESS_BUDGET,
    )


def build_generation_generalization_public_v12_diagnostic_observations(
        source_path: str | Path,
        ) -> tuple[GenerationGeneralizationEvaluationObservation, ...]:
    """返回八条当前 GG03 合同内的 public actual-run 诊断。"""
    clarify, conflict = _base(source_path)
    result = (
        _clarify(
            clarify,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[0],
            candidate_count=2,
            source_ids=("src-v12-clarify-a", "src-v12-clarify-b"),
            long_scope_trace=False,
            question_surface="两个候选中当前具体指哪一个接口？",
        ),
        _clarify(
            clarify,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[1],
            candidate_count=3,
            source_ids=(
                "src-v12-clarify-a", "src-v12-clarify-a",
                "src-v12-clarify-b",
            ),
            long_scope_trace=False,
            question_surface="三候选且来源重复时，当前具体指哪一个接口？",
        ),
        _clarify(
            clarify,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[2],
            candidate_count=4,
            source_ids=(
                "src-v12-clarify-a", "src-v12-clarify-b",
                "src-v12-clarify-c", "src-v12-clarify-d",
            ),
            long_scope_trace=True,
            question_surface="经过多次 scope 切换后，四个候选中当前指哪一个？",
        ),
        _clarify(
            clarify,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[3],
            candidate_count=4,
            source_ids=(
                "src-v12-clarify-d", "src-v12-clarify-c",
                "src-v12-clarify-b", "src-v12-clarify-a",
            ),
            long_scope_trace=True,
            question_surface="来源倒序且经历 scope 轨迹后，仍指向哪一个接口？",
        ),
        _conflict(
            conflict,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[4],
            stances=((1, 0), (1, 0), (1, 0), (0, 1)),
            long_scope_trace=False,
            question_surface="三份支持、一份反驳时，结果能否确定？",
        ),
        _conflict(
            conflict,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[5],
            stances=((0, 1), (0, 1), (0, 1), (1, 0)),
            long_scope_trace=False,
            question_surface="一份支持、三份反驳时，结果能否确定？",
        ),
        _conflict(
            conflict,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[6],
            stances=((1, 1), (1, 0), (0, 1), (0, 1)),
            long_scope_trace=True,
            question_surface="多轮 scope 轨迹中混合立场能否完整保留？",
        ),
        _conflict(
            conflict,
            case_id=PUBLIC_V12_DIAGNOSTIC_CASE_IDS[7],
            stances=((0, 1), (1, 0), (0, 1), (0, 1)),
            long_scope_trace=True,
            question_surface="多轮 scope 轨迹下一份支持三份反驳时能否确定？",
        ),
    )
    result = tuple(sorted(result, key=lambda item: item.stable_key()))
    if ({item.episode_id for item in result}
            != set(PUBLIC_V12_DIAGNOSTIC_CASE_IDS)):
        raise GenerationGeneralizationPublicV12DiagnosticError(
            "V12 diagnostic inventory 漂移")
    return result


__all__ = [
    "PUBLIC_V12_DIAGNOSTIC_CASE_IDS",
    "GenerationGeneralizationPublicV12DiagnosticError",
    "build_generation_generalization_public_v12_diagnostic_observations",
]
