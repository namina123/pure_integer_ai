"""GG-03 owner-independent public stress Observation inventory。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
    read_generation_generalization_evaluation_observations,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    DialogueEpisode,
    DialogueTurn,
    GroundedAnswerEpisode,
    read_grounded_answer_episodes,
)


PUBLIC_STRESS_BUDGET = GenerationGeneralizationEvaluationBudget(
    512, 4, 4, 96, 16)
PUBLIC_STRESS_CASE_IDS = (
    "gg03-public-stress-answer-novel-v1",
    "gg03-public-stress-answer-numeric-v1",
    "gg03-public-stress-answer-no-forbidden-v1",
    "gg03-public-stress-clarify-near-context-v1",
    "gg03-public-stress-clarify-punctuation-v1",
    "gg03-public-stress-clarify-three-turn-v1",
    "gg03-public-stress-conflict-novel-v1",
    "gg03-public-stress-conflict-numeric-v1",
    "gg03-public-stress-conflict-three-turn-v1",
    "gg03-public-stress-reference-novel-v1",
    "gg03-public-stress-reference-numeric-v1",
    "gg03-public-stress-reference-long-v1",
)


class GenerationGeneralizationPublicStressError(ValueError):
    """公开 stress 来源、构造或规范发布发生漂移。"""


def _base_episodes(
        path: str | Path,
        ) -> tuple[
            GroundedAnswerEpisode,
            GroundedAnswerEpisode,
            GroundedAnswerEpisode,
            GroundedAnswerEpisode,
        ]:
    """从公开课程精确选出 ANSWER/CLARIFY/CONFLICT/REFERENCE 基形。"""
    episodes = read_grounded_answer_episodes(path)

    def select(response_act: str, *, reference: bool) -> GroundedAnswerEpisode:
        matches = tuple(
            item for item in episodes
            if (item.question.answer_plan.response_act == response_act
                and (item.reference_course is not None) == reference)
        )
        if len(matches) != 1:
            raise GenerationGeneralizationPublicStressError(
                "public stress 基形未唯一命中")
        return matches[0]

    return (
        select("ANSWER", reference=False),
        select("CLARIFY", reference=False),
        select("CONFLICT", reference=False),
        select("ANSWER", reference=True),
    )


def _base_observation(
        episode: GroundedAnswerEpisode, case_id: str,
        ) -> GenerationGeneralizationEvaluationObservation:
    """剥离 TRAIN surface label，只保留公开 executable 输入。"""
    return GenerationGeneralizationEvaluationObservation.from_held_out_episode(
        replace(episode, episode_id=f"{case_id}-base", split="held_out"),
        PUBLIC_STRESS_BUDGET,
    )


def _three_turn_dialogue(
        observation: GenerationGeneralizationEvaluationObservation,
        question_surface: str,
        ) -> DialogueEpisode:
    """构造不依赖历史课程正文的三轮公开对话形状。"""
    question = observation.question
    return DialogueEpisode(
        (
            DialogueTurn(
                1, "USER", "请先保留当前资料范围。",
                (question.evidence_scope_id,),
            ),
            DialogueTurn(
                2, "ASSISTANT", "资料范围已保留，请继续说明任务。",
                (question.response_scope_id,),
            ),
            DialogueTurn(
                3, "USER", question_surface,
                (question.evidence_scope_id, question.response_scope_id),
            ),
        ),
        observation.dialogue.active_scope_ids,
    )


def _rewrite(
        base: GenerationGeneralizationEvaluationObservation,
        *,
        case_id: str,
        context_surface: str,
        question_surface: str,
        claim_texts: tuple[str, ...],
        typed_intent: str,
        clear_forbidden: bool = False,
        three_turn: bool = False,
        reference_surfaces: tuple[str, ...] | None = None,
        ) -> GenerationGeneralizationEvaluationObservation:
    """改写公开词面并保留 typed Proposition/Evidence/source/scope 形状。"""
    proposition_ids = tuple(dict.fromkeys(
        item.proposition_id for item in base.question.evidence))
    if len(proposition_ids) != len(claim_texts):
        raise GenerationGeneralizationPublicStressError(
            "public stress claim 数与 Proposition 分母不一致")
    claims = dict(zip(proposition_ids, claim_texts, strict=True))
    evidence = tuple(
        replace(
            item,
            claim_text=claims[item.proposition_id],
            evidence_text=(
                f"公开压力证据{ordinal}："
                f"{claims[item.proposition_id]}"),
        )
        for ordinal, item in enumerate(base.question.evidence, start=1)
    )
    plan = base.question.answer_plan
    if clear_forbidden:
        plan = replace(plan, forbidden_claim_ids=())
    question = replace(
        base.question,
        typed_intent=typed_intent,
        context_surface=context_surface,
        question_surface=question_surface,
        evidence=evidence,
        answer_plan=plan,
    )
    dialogue = (
        _three_turn_dialogue(base, question_surface)
        if three_turn else replace(
            base.dialogue,
            turns=(
                *base.dialogue.turns[:-1],
                replace(
                    base.dialogue.turns[-1],
                    surface=question_surface,
                ),
            ),
        )
    )
    reference = base.reference_course
    if reference_surfaces is not None:
        if (reference is None
                or len(reference_surfaces) != len(reference.options)):
            raise GenerationGeneralizationPublicStressError(
                "public stress reference surface 分母不一致")
        reference = replace(
            reference,
            options=tuple(
                replace(option, reference_surface=surface)
                for option, surface in zip(
                    reference.options, reference_surfaces, strict=True)
            ),
        )
    return replace(
        base,
        episode_id=case_id,
        question=question,
        dialogue=dialogue,
        reference_course=reference,
    )


def build_generation_generalization_public_stress_observations(
        source_path: str | Path,
        ) -> tuple[GenerationGeneralizationEvaluationObservation, ...]:
    """由公开课程确定性构造四路径、六 requirement 的 12 条压力输入。"""
    answer_episode, clarify_episode, conflict_episode, reference_episode = (
        _base_episodes(source_path))
    answer = _base_observation(answer_episode, "stress-answer")
    clarify = _base_observation(clarify_episode, "stress-clarify")
    conflict = _base_observation(conflict_episode, "stress-conflict")
    reference = _base_observation(reference_episode, "stress-reference")

    observations = (
        _rewrite(
            answer,
            case_id=PUBLIC_STRESS_CASE_IDS[0],
            context_surface="公开传感器完成低温校准，记录保持在同一资料范围。",
            question_surface="校准后的传感器是否保持稳定？",
            claim_texts=("新型陶瓷传感器在低温校准后保持稳定读数",),
            typed_intent="PUBLIC_STRESS_READBACK_NOVEL",
        ),
        _rewrite(
            answer,
            case_id=PUBLIC_STRESS_CASE_IDS[1],
            context_surface="公开压力条件：负载45%；允许误差5%。",
            question_surface="该设备在45%负载下是否稳定？",
            claim_texts=("公开设备在45%负载下保持稳定；测量误差低于5%",),
            typed_intent="PUBLIC_STRESS_READBACK_NUMERIC",
        ),
        _rewrite(
            answer,
            case_id=PUBLIC_STRESS_CASE_IDS[2],
            context_surface="公开记录只给出一个得到支持的候选结论。",
            question_surface="唯一受支持的结论是什么？",
            claim_texts=("维护窗口将在周四上午开放",),
            typed_intent="PUBLIC_STRESS_READBACK_NO_FORBIDDEN",
            clear_forbidden=True,
        ),
        _rewrite(
            clarify,
            case_id=PUBLIC_STRESS_CASE_IDS[3],
            context_surface="边界背景" * 40,
            question_surface="请确认当前问题指向哪一个候选对象？",
            claim_texts=("北侧接口需要复核", "南侧接口需要复核"),
            typed_intent="PUBLIC_STRESS_CLARIFY_NEAR_CONTEXT",
        ),
        _rewrite(
            clarify,
            case_id=PUBLIC_STRESS_CASE_IDS[4],
            context_surface="公开记录同时出现A/B、甲：乙和编号17。",
            question_surface="这里的“接口”具体指A端还是B端？",
            claim_texts=("A端接口等待确认", "B端接口等待确认"),
            typed_intent="PUBLIC_STRESS_CLARIFY_PUNCTUATION",
        ),
        _rewrite(
            clarify,
            case_id=PUBLIC_STRESS_CASE_IDS[5],
            context_surface="公开多轮会话保留两个同名对象。",
            question_surface="请在前述范围内确认具体对象。",
            claim_texts=("第一对象处于开放状态", "第二对象处于开放状态"),
            typed_intent="PUBLIC_STRESS_CLARIFY_THREE_TURN",
            three_turn=True,
        ),
        _rewrite(
            conflict,
            case_id=PUBLIC_STRESS_CASE_IDS[6],
            context_surface="两份公开来源对同一状态给出相反记录。",
            question_surface="当前能否确定设备状态？",
            claim_texts=("设备当前处于启用状态",),
            typed_intent="PUBLIC_STRESS_CONFLICT_NOVEL",
        ),
        _rewrite(
            conflict,
            case_id=PUBLIC_STRESS_CASE_IDS[7],
            context_surface="来源甲记录成功率为91%；来源乙记录成功率为19%。",
            question_surface="成功率是否已经确定？",
            claim_texts=("本轮操作成功率已经确定",),
            typed_intent="PUBLIC_STRESS_CONFLICT_NUMERIC",
        ),
        _rewrite(
            conflict,
            case_id=PUBLIC_STRESS_CASE_IDS[8],
            context_surface="公开多轮会话中的两个来源仍然相互矛盾。",
            question_surface="结合前述对话，现在可以确定结果吗？",
            claim_texts=("当前结果可以被唯一确定",),
            typed_intent="PUBLIC_STRESS_CONFLICT_THREE_TURN",
            three_turn=True,
        ),
        _rewrite(
            reference,
            case_id=PUBLIC_STRESS_CASE_IDS[9],
            context_surface="公开档案连续记录同一设施的两个属性。",
            question_surface="请连续说明该设施的两个属性。",
            claim_texts=("设施建成于二零一八年", "设施位于东侧入口"),
            typed_intent="PUBLIC_STRESS_REFERENCE_NOVEL",
            reference_surfaces=("该设施", "设施"),
        ),
        _rewrite(
            reference,
            case_id=PUBLIC_STRESS_CASE_IDS[10],
            context_surface="公开档案记录编号17设施的容量与开放比例。",
            question_surface="请说明编号17设施的容量和开放比例。",
            claim_texts=("编号17设施容量为45单元", "编号17设施开放比例为95%"),
            typed_intent="PUBLIC_STRESS_REFERENCE_NUMERIC",
            reference_surfaces=("该设施", "编号17设施"),
        ),
        _rewrite(
            reference,
            case_id=PUBLIC_STRESS_CASE_IDS[11],
            context_surface="公开双命题长文本用于验证生成长度与查询耗时边界。",
            question_surface="请完整说明两个较长属性并保持引用清晰。",
            claim_texts=(
                "长距离引用结论一" * 9,
                "长距离引用结论二" * 9,
            ),
            typed_intent="PUBLIC_STRESS_REFERENCE_LONG",
            reference_surfaces=("该公开设施", "公开设施"),
        ),
    )
    result = tuple(sorted(observations, key=lambda item: item.stable_key()))
    if ({item.episode_id for item in result} != set(PUBLIC_STRESS_CASE_IDS)
            or len(result) != len(PUBLIC_STRESS_CASE_IDS)):
        raise GenerationGeneralizationPublicStressError(
            "public stress case inventory 漂移")
    return result


def generation_generalization_public_stress_bytes(
        observations: tuple[
            GenerationGeneralizationEvaluationObservation, ...],
        ) -> bytes:
    """返回按 stable key 排序的 canonical JSONL transport。"""
    if (not isinstance(observations, tuple)
            or not observations
            or observations != tuple(sorted(
                observations, key=lambda item: item.stable_key()))):
        raise GenerationGeneralizationPublicStressError(
            "public stress Observation 未冻结排序")
    return b"".join(
        canonical_json_line(item.to_dict()) for item in observations)


def publish_generation_generalization_public_stress_inventory(
        source_path: str | Path,
        target_path: str | Path,
        ) -> tuple[GenerationGeneralizationEvaluationObservation, ...]:
    """不可覆盖发布公开 stress JSONL，并严格回读逐字段 identity。"""
    target = Path(target_path).resolve()
    if target.exists():
        raise GenerationGeneralizationPublicStressError(
            "public stress inventory 已存在")
    observations = build_generation_generalization_public_stress_observations(
        source_path)
    with target.open("xb") as handle:
        handle.write(generation_generalization_public_stress_bytes(observations))
    if read_generation_generalization_evaluation_observations(target) != (
            observations):
        raise GenerationGeneralizationPublicStressError(
            "public stress inventory 回读漂移")
    return observations


__all__ = [
    "PUBLIC_STRESS_BUDGET",
    "PUBLIC_STRESS_CASE_IDS",
    "GenerationGeneralizationPublicStressError",
    "build_generation_generalization_public_stress_observations",
    "generation_generalization_public_stress_bytes",
    "publish_generation_generalization_public_stress_inventory",
]
