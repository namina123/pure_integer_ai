"""从 grounded-answer TRAIN Evidence 学习 claim 槽与 response-act 表面模式。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    GroundedAnswerTrainingBundle,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedQuestionEpisode,
    SurfaceRealization,
    verify_surface_realization,
)


PATTERN_LITERAL = "LITERAL"
PATTERN_CLAIM = "CLAIM"


# object-model: exception
class GroundedAnswerLearningError(ValueError):
    """训练标签不能形成受约束 pattern，或新 answer plan 无可用 surface。"""


def _positive(value: int, *, where: str) -> int:
    """核验正严格整数。"""
    if type(value) is not int or value <= 0:
        raise GroundedAnswerLearningError(f"{where} 必须是正严格整数")
    return value


def _pattern_id(parts: object) -> int:
    """从 pattern 规范值生成稳定正整数身份。"""
    value = int.from_bytes(
        hashlib.sha256(canonical_json_bytes(parts)).digest()[:8], "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SurfacePatternPart:
    """一个已学字面片段，或按 answer plan 序号取值的 claim 槽。"""

    kind: str
    literal: str = ""
    claim_ordinal: int = 0

    def __post_init__(self) -> None:
        if self.kind == PATTERN_LITERAL:
            if (not isinstance(self.literal, str) or not self.literal
                    or self.claim_ordinal != 0):
                raise GroundedAnswerLearningError("literal pattern part 非法")
        elif self.kind == PATTERN_CLAIM:
            if self.literal or type(self.claim_ordinal) is not int \
                    or self.claim_ordinal < 0:
                raise GroundedAnswerLearningError("claim pattern part 非法")
        else:
            raise GroundedAnswerLearningError("pattern part kind 未注册")

    def stable_value(self) -> list[object]:
        """返回用于 pattern 身份的规范值。"""
        return [self.kind, self.literal, self.claim_ordinal]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedSurfacePattern:
    """一个 response act、载体和 claim 数量下的可重用表面模式。"""

    pattern_id: int
    response_act: str
    carrier_kind: str
    claim_count: int
    parts: tuple[SurfacePatternPart, ...]
    support_episode_ids: tuple[str, ...]
    support_teacher_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _positive(self.pattern_id, where="pattern_id")
        if not isinstance(self.response_act, str) or not self.response_act:
            raise GroundedAnswerLearningError("pattern response_act 非法")
        if not isinstance(self.carrier_kind, str) or not self.carrier_kind:
            raise GroundedAnswerLearningError("pattern carrier_kind 非法")
        if type(self.claim_count) is not int or self.claim_count < 0:
            raise GroundedAnswerLearningError("pattern claim_count 非法")
        if (not isinstance(self.parts, tuple) or not self.parts
                or any(not isinstance(item, SurfacePatternPart)
                       for item in self.parts)):
            raise GroundedAnswerLearningError("pattern parts 必须非空")
        ordinals = tuple(
            item.claim_ordinal for item in self.parts
            if item.kind == PATTERN_CLAIM)
        if ordinals != tuple(range(self.claim_count)):
            raise GroundedAnswerLearningError(
                "pattern claim 槽必须按 answer plan 完整递增")
        if (not isinstance(self.support_episode_ids, tuple)
                or not self.support_episode_ids
                or any(not isinstance(item, str) or not item
                       for item in self.support_episode_ids)
                or self.support_episode_ids != tuple(sorted(
                    set(self.support_episode_ids)))):
            raise GroundedAnswerLearningError(
                "pattern support episode 非规范")
        if (not isinstance(self.support_teacher_keys, tuple)
                or not self.support_teacher_keys
                or any(not isinstance(key, tuple) or not key
                       or any(type(value) is not int for value in key)
                       for key in self.support_teacher_keys)
                or self.support_teacher_keys != tuple(sorted(
                    set(self.support_teacher_keys)))):
            raise GroundedAnswerLearningError(
                "pattern support teacher Evidence 非规范")

    def stable_value(self) -> dict[str, object]:
        """返回不含支持计数的 pattern 结构值。"""
        return {
            "carrier_kind": self.carrier_kind,
            "claim_count": self.claim_count,
            "parts": [item.stable_value() for item in self.parts],
            "response_act": self.response_act,
        }


def surface_pattern_structure_key(
        pattern: LearnedSurfacePattern,
        ) -> tuple[tuple[int, int], ...]:
    """返回忽略 literal 内容、保留 part/claim 序的结构键。"""
    if not isinstance(pattern, LearnedSurfacePattern):
        raise TypeError("surface pattern structure key 类型错误")
    return tuple(
        (0, 0) if part.kind == PATTERN_LITERAL
        else (1, part.claim_ordinal)
        for part in pattern.parts
    )


def surface_pattern_structure_id(pattern: LearnedSurfacePattern) -> int:
    """为 response act、载体、claim 数和 part 形状生成稳定结构身份。"""
    structure = {
        "carrier_kind": pattern.carrier_kind,
        "claim_count": pattern.claim_count,
        "parts": [list(item) for item in surface_pattern_structure_key(pattern)],
        "response_act": pattern.response_act,
    }
    return _pattern_id(structure)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerSurfaceModel:
    """由 TRAIN 合法表面形成的确定性 pattern 集。"""

    patterns: tuple[LearnedSurfacePattern, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.patterns, tuple) or not self.patterns
                or any(not isinstance(item, LearnedSurfacePattern)
                       for item in self.patterns)):
            raise GroundedAnswerLearningError("surface model patterns 不能为空")
        ids = tuple(item.pattern_id for item in self.patterns)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedAnswerLearningError(
                "surface model pattern id 必须唯一递增")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerLearningReport:
    """记录实际消费 episode、合法表面、pattern 和 slotted pattern 数。"""

    episode_count: int
    accepted_surface_count: int
    pattern_count: int
    slotted_pattern_count: int
    response_act_count: int

    def __post_init__(self) -> None:
        for name, value in (
                ("episode_count", self.episode_count),
                ("accepted_surface_count", self.accepted_surface_count),
                ("pattern_count", self.pattern_count),
                ("response_act_count", self.response_act_count)):
            _positive(value, where=f"learning_report.{name}")
        if (type(self.slotted_pattern_count) is not int
                or self.slotted_pattern_count < 0
                or self.slotted_pattern_count > self.pattern_count):
            raise GroundedAnswerLearningError(
                "learning_report slotted_pattern_count 非法")


def _claim_texts(observation: dict[str, object]) -> dict[str, str]:
    """从学生可见 Evidence 恢复每个 Proposition 唯一 claim surface。"""
    rows = observation["evidence"]
    if not isinstance(rows, list):
        raise GroundedAnswerLearningError("observation evidence 类型错误")
    grouped: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise GroundedAnswerLearningError("observation evidence row 类型错误")
        proposition_id = row.get("proposition_id")
        claim_text = row.get("claim_text")
        if not isinstance(proposition_id, str) or not isinstance(
                claim_text, str):
            raise GroundedAnswerLearningError("Evidence claim 字段错误")
        grouped.setdefault(proposition_id, set()).add(claim_text)
    result = {}
    for proposition_id, values in grouped.items():
        if len(values) != 1:
            raise GroundedAnswerLearningError(
                "同一 Proposition 存在多个 claim surface")
        result[proposition_id] = next(iter(values))
    return result


def _surface_parts(
        surface: str,
        response_act: str,
        ordered_claim_ids: tuple[str, ...],
        claim_texts: dict[str, str],
        ) -> tuple[SurfacePatternPart, ...]:
    """按 plan 顺序把合法 surface 分解为 literal 和 exact claim 槽。"""
    if response_act != "ANSWER":
        if ordered_claim_ids:
            raise GroundedAnswerLearningError(
                "非 ANSWER pattern 不得携带 claim")
        return (SurfacePatternPart(PATTERN_LITERAL, surface),)
    parts = []
    cursor = 0
    for ordinal, claim_id in enumerate(ordered_claim_ids):
        claim_text = claim_texts.get(claim_id)
        if claim_text is None:
            raise GroundedAnswerLearningError("answer plan claim 缺少 Evidence text")
        start = surface.find(claim_text, cursor)
        if start < 0:
            raise GroundedAnswerLearningError(
                "首轮 learner 只接受可精确定位的 claim surface")
        if start > cursor:
            parts.append(SurfacePatternPart(
                PATTERN_LITERAL, surface[cursor:start]))
        parts.append(SurfacePatternPart(
            PATTERN_CLAIM, claim_ordinal=ordinal))
        cursor = start + len(claim_text)
    if cursor < len(surface):
        parts.append(SurfacePatternPart(PATTERN_LITERAL, surface[cursor:]))
    if not parts:
        raise GroundedAnswerLearningError("ANSWER surface 未形成 pattern")
    return tuple(parts)


def learn_grounded_answer_surface_model(
        bundle: GroundedAnswerTrainingBundle,
        ) -> tuple[GroundedAnswerSurfaceModel, GroundedAnswerLearningReport]:
    """只从 teacher accepted surface 学习 pattern，不消费 rejected 文本。"""
    if not isinstance(bundle, GroundedAnswerTrainingBundle):
        raise TypeError("surface learner bundle 类型错误")
    observations = {item.stable_key: item for item in bundle.observations}
    supports: dict[
        tuple[str, str, int, tuple[tuple[object, ...], ...]],
        tuple[set[str], set[tuple[int, ...]]],
    ] = {}
    accepted_count = 0
    for teacher in bundle.teacher_evidence:
        observation_record = observations.get(teacher.observation_key)
        if observation_record is None:
            raise GroundedAnswerLearningError(
                "teacher Evidence 缺少 Observation")
        observation = observation_record.typed_payload.to_value()
        label = teacher.typed_evidence.to_value()
        episode_id = observation["episode_id"]
        if not isinstance(episode_id, str):
            raise GroundedAnswerLearningError("Observation episode_id 类型错误")
        plan = label["answer_plan"]
        surfaces = label["surface_realizations"]
        if not isinstance(plan, dict) or not isinstance(surfaces, dict):
            raise GroundedAnswerLearningError("teacher plan/surface 类型错误")
        response_act = plan["response_act"]
        raw_order = plan["ordered_claim_ids"]
        accepted = surfaces["accepted"]
        if (not isinstance(response_act, str)
                or not isinstance(raw_order, list)
                or not isinstance(accepted, list)):
            raise GroundedAnswerLearningError("teacher accepted contract 漂移")
        ordered_claim_ids = tuple(raw_order)
        claim_texts = _claim_texts(observation)
        for row in accepted:
            if (not isinstance(row, dict)
                    or not isinstance(row.get("realization"), dict)
                    or not isinstance(row.get("verification"), dict)):
                raise GroundedAnswerLearningError("accepted realization 类型错误")
            realization = row["realization"]
            verification = row["verification"]
            if (verification.get("verdict") != "PASS"
                    or verification.get("violations") != []):
                raise GroundedAnswerLearningError(
                    "accepted realization 未携带无失败 verification")
            surface = realization["surface"]
            carrier_kind = realization["carrier_kind"]
            if not isinstance(surface, str) or not isinstance(
                    carrier_kind, str):
                raise GroundedAnswerLearningError("accepted surface 字段错误")
            parts = _surface_parts(
                surface, response_act, ordered_claim_ids, claim_texts)
            part_key = tuple(tuple(item.stable_value()) for item in parts)
            signature = (
                response_act, carrier_kind, len(ordered_claim_ids), part_key)
            episode_ids, teacher_keys = supports.setdefault(
                signature, (set(), set()))
            episode_ids.add(episode_id)
            teacher_keys.add(tuple(teacher.stable_key.components))
            accepted_count += 1
    patterns = []
    for signature, support in supports.items():
        response_act, carrier_kind, claim_count, part_key = signature
        episode_ids, teacher_keys = support
        parts = tuple(SurfacePatternPart(
            str(item[0]), str(item[1]), int(item[2])) for item in part_key)
        stable_value = {
            "carrier_kind": carrier_kind,
            "claim_count": claim_count,
            "parts": [item.stable_value() for item in parts],
            "response_act": response_act,
        }
        patterns.append(LearnedSurfacePattern(
            _pattern_id(stable_value),
            response_act,
            carrier_kind,
            claim_count,
            parts,
            tuple(sorted(episode_ids)),
            tuple(sorted(teacher_keys)),
        ))
    model = GroundedAnswerSurfaceModel(tuple(sorted(
        patterns, key=lambda item: item.pattern_id)))
    report = GroundedAnswerLearningReport(
        len(bundle.observations),
        accepted_count,
        len(model.patterns),
        sum(any(part.kind == PATTERN_CLAIM for part in pattern.parts)
            for pattern in model.patterns),
        len({item.response_act for item in model.patterns}),
    )
    return model, report


def _question_claim_texts(
        question: GroundedQuestionEpisode,
        ) -> dict[str, str]:
    """从新问题 Evidence 恢复 plan claim 的唯一表面。"""
    grouped: dict[str, set[str]] = {}
    for evidence in question.evidence:
        grouped.setdefault(evidence.proposition_id, set()).add(
            evidence.claim_text)
    result = {}
    for claim_id in question.answer_plan.ordered_claim_ids:
        values = grouped.get(claim_id, set())
        if len(values) != 1:
            raise GroundedAnswerLearningError(
                "新 answer plan claim 缺少唯一 Evidence surface")
        result[claim_id] = next(iter(values))
    return result


def realize_grounded_answer_surfaces(
        model: GroundedAnswerSurfaceModel,
        question: GroundedQuestionEpisode,
        *, carrier_kind: str = "PLAIN_TEXT", max_surfaces: int = 8,
        ) -> tuple[SurfaceRealization, ...]:
    """以新 Evidence claim 填入已学槽，并只返回通过同一 verifier 的表面。"""
    if not isinstance(model, GroundedAnswerSurfaceModel):
        raise TypeError("surface realization model 类型错误")
    if not isinstance(question, GroundedQuestionEpisode):
        raise TypeError("surface realization question 类型错误")
    _positive(max_surfaces, where="max_surfaces")
    plan = question.answer_plan
    claim_texts = _question_claim_texts(question)
    surfaces = []
    seen = set()
    for pattern in model.patterns:
        if (pattern.response_act != plan.response_act
                or pattern.carrier_kind != carrier_kind
                or pattern.claim_count != len(plan.ordered_claim_ids)):
            continue
        units = []
        for part in pattern.parts:
            if part.kind == PATTERN_LITERAL:
                units.append(part.literal)
            else:
                claim_id = plan.ordered_claim_ids[part.claim_ordinal]
                units.append(claim_texts[claim_id])
        surface = "".join(units)
        if not surface or surface in seen:
            continue
        realization = SurfaceRealization(
            f"learned-{pattern.pattern_id}",
            surface,
            carrier_kind,
            plan.response_act,
            question.response_scope_id,
            plan.required_claim_ids,
            plan.citation_source_ids,
        )
        if not verify_surface_realization(question, realization).passed:
            raise GroundedAnswerLearningError(
                "已学 pattern 生成结果未通过同一 verifier")
        surfaces.append(realization)
        seen.add(surface)
        if len(surfaces) >= max_surfaces:
            break
    if not surfaces:
        raise GroundedAnswerLearningError("当前 answer plan 没有已学 surface pattern")
    return tuple(surfaces)


__all__ = [
    "GroundedAnswerLearningError",
    "GroundedAnswerLearningReport",
    "GroundedAnswerSurfaceModel",
    "LearnedSurfacePattern",
    "PATTERN_CLAIM",
    "PATTERN_LITERAL",
    "SurfacePatternPart",
    "learn_grounded_answer_surface_model",
    "realize_grounded_answer_surfaces",
    "surface_pattern_structure_id",
    "surface_pattern_structure_key",
]
