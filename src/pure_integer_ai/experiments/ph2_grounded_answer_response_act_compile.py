"""把已学 literal-only 非回答 pattern 编译为 G-02 response-act 模板。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.generation_response import (
    ResponseActGenerationTemplate,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    concept_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderSearchBudget,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    GroundedAnswerSurfaceModel,
    LearnedSurfacePattern,
    PATTERN_LITERAL,
)


_NAMESPACE = 20961
_NONANSWER_ACTS = frozenset({"UNKNOWN", "CLARIFY", "CONFLICT"})


# object-model: exception
class GroundedResponseActCompileError(ValueError):
    """已学 pattern 不能无损形成 response-act 模板。"""


def _stable_id(value: object) -> int:
    """从规范 JSON 值生成稳定正整数身份。"""
    result = int.from_bytes(
        hashlib.sha256(canonical_json_bytes(value)).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedResponseActCompileError(
            f"{where} 必须是非空严格整数 tuple")
    return value


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 stance 使用一等 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedResponseActCompileError(
            f"{where} 必须是 MinimalInstruction")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActCompileTarget:
    """显式绑定课程 response act、G-01 stance、语言分支和表示族。"""

    response_act: str
    stance: ObjectIdentity
    language_branch: ObjectIdentity
    representation_family: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.response_act not in _NONANSWER_ACTS:
            raise GroundedResponseActCompileError(
                "response-act compiler 只接受 UNKNOWN/CLARIFY/CONFLICT")
        _instruction(self.stance, where="response-act stance")
        if (not isinstance(self.language_branch, ObjectIdentity)
                or self.language_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise GroundedResponseActCompileError(
                "response-act language branch 类型错误")
        _strict_key(
            self.representation_family,
            where="response-act representation family",
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActVariant:
    """一个已学 literal pattern 的模板、R-01 表示、预算和任务归属。"""

    pattern_id: int
    response_act: str
    template: ResponseActGenerationTemplate
    structure_family: ObjectIdentity
    representation: ObjectIdentity
    task: ObjectIdentity
    task_requirement: ObjectIdentity
    task_result_key: tuple[int, ...]
    surface_instruction: ObjectIdentity
    surface_budget: AliasRouteSearchBudget
    order_budget: StructureOrderSearchBudget
    use_key_suffix: tuple[int, ...]
    support_episode_ids: tuple[str, ...]
    support_teacher_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.pattern_id) is not int or self.pattern_id <= 0:
            raise GroundedResponseActCompileError("response-act pattern id 非法")
        if self.response_act not in _NONANSWER_ACTS:
            raise GroundedResponseActCompileError("response-act variant 类型非法")
        if not isinstance(self.template, ResponseActGenerationTemplate):
            raise TypeError("response-act template 类型错误")
        if self.template.stance.object_kind != OBJECT_MINIMAL_INSTRUCTION:
            raise GroundedResponseActCompileError("response-act template stance 漂移")
        if (not isinstance(self.representation, ObjectIdentity)
                or self.representation.object_kind != OBJECT_REPRESENTATION):
            raise TypeError("response-act representation 类型错误")
        for value, where in (
                (self.task, "task"),
                (self.task_requirement, "task requirement"),
                (self.surface_instruction, "surface instruction")):
            _instruction(value, where=f"response-act {where}")
        _strict_key(self.task_result_key, where="response-act task result")
        _strict_key(self.use_key_suffix, where="response-act use suffix")
        if not isinstance(self.surface_budget, AliasRouteSearchBudget):
            raise TypeError("response-act surface budget 类型错误")
        if not isinstance(self.order_budget, StructureOrderSearchBudget):
            raise TypeError("response-act order budget 类型错误")
        if (not self.support_episode_ids
                or self.support_episode_ids != tuple(sorted(
                    set(self.support_episode_ids)))):
            raise GroundedResponseActCompileError(
                "response-act support episode 非规范")
        if (not self.support_teacher_keys
                or self.support_teacher_keys != tuple(sorted(
                    set(self.support_teacher_keys)))):
            raise GroundedResponseActCompileError(
                "response-act support teacher key 非规范")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActCompilation:
    """保存同一 response act 下全部可显式选择的 learned variants。"""

    target: GroundedResponseActCompileTarget
    variants: tuple[GroundedResponseActVariant, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target, GroundedResponseActCompileTarget):
            raise TypeError("response-act compilation target 类型错误")
        if (not isinstance(self.variants, tuple) or not self.variants
                or any(not isinstance(item, GroundedResponseActVariant)
                       for item in self.variants)):
            raise GroundedResponseActCompileError(
                "response-act variants 不能为空")
        ids = tuple(item.pattern_id for item in self.variants)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedResponseActCompileError(
                "response-act pattern id 必须唯一递增")
        if any(
                item.response_act != self.target.response_act
                or item.template.stance != self.target.stance
                or item.template.branch != self.target.language_branch
                for item in self.variants):
            raise GroundedResponseActCompileError(
                "response-act variant 与 compilation target 漂移")

    def select(self, pattern_id: int) -> GroundedResponseActVariant:
        """按调用者显式 pattern identity 返回唯一 variant。"""
        matches = tuple(
            item for item in self.variants if item.pattern_id == pattern_id)
        if len(matches) != 1:
            raise GroundedResponseActCompileError(
                "selected response-act pattern 不属于 compilation")
        return matches[0]


def _literal(pattern: LearnedSurfacePattern) -> str:
    """核验非回答 pattern 只有一个完整 literal part。"""
    if (pattern.response_act == "ANSWER" or pattern.claim_count != 0
            or len(pattern.parts) != 1
            or pattern.parts[0].kind != PATTERN_LITERAL):
        raise GroundedResponseActCompileError(
            "response-act compiler 只接受 literal-only 零 claim pattern")
    return pattern.parts[0].literal


def _variant(
        pattern: LearnedSurfacePattern,
        target: GroundedResponseActCompileTarget,
        ) -> GroundedResponseActVariant:
    """从单个 learned pattern 构造来源化模板和独占 R-01 表示要求。"""
    literal = _literal(pattern)
    branch = target.language_branch
    theory_id = _stable_id({
        "branch": list(branch.stable_key()),
        "pattern_id": pattern.pattern_id,
        "response_act": target.response_act,
        "stance": list(target.stance.stable_key()),
        "version": 1,
    })
    prefix = (_NAMESPACE, 1, theory_id, pattern.pattern_id)
    structure_family = structure_concept_identity(
        (*prefix, 1), owner=branch.owner, versions=branch.versions)
    structure = structure_concept_identity(
        (*prefix, 2), owner=branch.owner, versions=branch.versions)
    slot = StructureSlotDefinition(
        structure,
        structure_concept_identity(
            (*prefix, 3), owner=branch.owner, versions=branch.versions),
        role_identity(
            (*prefix, 4), owner=branch.owner, versions=branch.versions),
        concept_identity(
            (*prefix, 5), owner=branch.owner, versions=branch.versions),
    )
    template = ResponseActGenerationTemplate(
        branch,
        target.stance,
        structure_concept_identity(
            (*prefix, 6), owner=branch.owner, versions=branch.versions),
        slot,
        minimal_instruction_identity(
            (*prefix, 7), owner=branch.owner, versions=branch.versions),
        minimal_instruction_identity(
            (*prefix, 8), owner=branch.owner, versions=branch.versions),
    )
    representation = representation_identity(
        target.representation_family,
        tuple(ord(character) for character in literal),
        owner=branch.owner,
        versions=branch.versions,
    )
    result_key = integer_tuple_fingerprint(
        (
            *target.stance.stable_key(),
            *branch.stable_key(),
        ),
        domain="grounded.response.act.task.result.v1",
    )
    return GroundedResponseActVariant(
        pattern.pattern_id,
        pattern.response_act,
        template,
        structure_family,
        representation,
        minimal_instruction_identity(
            (*prefix, 9), owner=branch.owner, versions=branch.versions),
        minimal_instruction_identity(
            (*prefix, 10), owner=branch.owner, versions=branch.versions),
        result_key,
        minimal_instruction_identity(
            (*prefix, 11), owner=branch.owner, versions=branch.versions),
        AliasRouteSearchBudget(16, 16, 4),
        StructureOrderSearchBudget(4),
        (*prefix, 12),
        pattern.support_episode_ids,
        pattern.support_teacher_keys,
    )


def compile_grounded_response_act_patterns(
        model: GroundedAnswerSurfaceModel,
        target: GroundedResponseActCompileTarget,
        ) -> GroundedResponseActCompilation:
    """只从 learner 产物编译目标 response act，不读取原 surface label。"""
    if not isinstance(model, GroundedAnswerSurfaceModel):
        raise TypeError("response-act compiler model 类型错误")
    if not isinstance(target, GroundedResponseActCompileTarget):
        raise TypeError("response-act compiler target 类型错误")
    selected = tuple(
        pattern for pattern in model.patterns
        if pattern.response_act == target.response_act)
    if not selected:
        raise GroundedResponseActCompileError(
            "surface model 缺目标 response-act pattern")
    variants = tuple(sorted(
        (_variant(pattern, target) for pattern in selected),
        key=lambda item: item.pattern_id,
    ))
    return GroundedResponseActCompilation(target, variants)


__all__ = [
    "GroundedResponseActCompilation",
    "GroundedResponseActCompileError",
    "GroundedResponseActCompileTarget",
    "GroundedResponseActVariant",
    "compile_grounded_response_act_patterns",
]
