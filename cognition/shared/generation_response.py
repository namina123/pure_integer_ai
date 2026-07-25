"""为无可回答命题的 G-01 stance 建立可渲染的 G-02 response-act 结构。

本模块不解释 unknown、refuse 等具体语义，也不保存任何文字。调用方以
``(LanguageBranch, stance)`` 注入句式、S-07 槽、约束和原因；有已选命题时，
三个 router 原样委托既有语言结构 mapper，避免把问答入口变成第二套生成系统。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    DiscoursePlan,
    PlannedSentence,
    PropositionPlan,
    SyntaxLinearizationObligation,
    SyntaxPlan,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureSlotValue,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _identity(
        value: ObjectIdentity, *, label: str, kind: int | None = None,
        ) -> ObjectIdentity:
    """核验注入的一等对象及可选对象类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if kind is not None and value.object_kind != kind:
        raise ValueError(f"{label} 对象类型不匹配")
    return value


@dataclass(frozen=True)
class ResponseActGenerationTemplate:
    """描述一个 stance 在目标语言分支中的句式、槽、约束和来源化原因。"""

    branch: ObjectIdentity
    stance: ObjectIdentity
    sentence: ObjectIdentity
    slot: StructureSlotDefinition
    boundary: ObjectIdentity
    linearization_reason: ObjectIdentity
    constraints: tuple[ObjectIdentity, ...] = ()
    context: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        """核验模板只含注入身份，并保证约束、上下文确定且不重复。"""
        _identity(
            self.branch,
            label="response act branch",
            kind=OBJECT_LANGUAGE_BRANCH,
        )
        _identity(
            self.stance,
            label="response act stance",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        _identity(
            self.sentence,
            label="response act sentence",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        if not isinstance(self.slot, StructureSlotDefinition):
            raise TypeError("response act slot 必须是 StructureSlotDefinition")
        for identity, label in (
                (self.boundary, "response act boundary"),
                (self.linearization_reason,
                 "response act linearization reason")):
            _identity(identity, label=label, kind=OBJECT_MINIMAL_INSTRUCTION)
        for values, label in (
                (self.constraints, "response act constraints"),
                (self.context, "response act context")):
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, ObjectIdentity)
                           for item in values)):
                raise TypeError(f"{label} 必须是 ObjectIdentity tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} 不得重复")
        object.__setattr__(self, "constraints", tuple(sorted(
            self.constraints, key=ObjectIdentity.stable_key)))
        object.__setattr__(self, "context", tuple(sorted(
            self.context, key=ObjectIdentity.stable_key)))

    def stable_key(self) -> tuple[int, ...]:
        """返回分支、stance、句式、S-07 槽和全部生成约束的稳定键。"""
        slot = self.slot
        result = [
            *_packed(self.branch.stable_key()),
            *_packed(self.stance.stable_key()),
            *_packed(self.sentence.stable_key()),
            *_packed(slot.structure.stable_key()),
            *_packed(slot.slot.stable_key()),
            *_packed(slot.role.stable_key()),
            *_packed(slot.value_type.stable_key()),
            *_packed(self.boundary.stable_key()),
            *_packed(self.linearization_reason.stable_key()),
            len(self.constraints),
        ]
        for identity in self.constraints:
            result.extend(_packed(identity.stable_key()))
        result.append(len(self.context))
        for identity in self.context:
            result.extend(_packed(identity.stable_key()))
        return tuple(result)


class ResponseActGenerationRegistry:
    """按完整 LanguageBranch 和 stance 身份索引注入的 response-act 模板。"""

    def __init__(
            self, templates: tuple[ResponseActGenerationTemplate, ...],
            ) -> None:
        """建立不可歧义的模板索引，不从身份整数或文字推断用途。"""
        if (not isinstance(templates, tuple) or not templates
                or any(not isinstance(item, ResponseActGenerationTemplate)
                       for item in templates)):
            raise TypeError("response act templates 必须是非空模板 tuple")
        keys = tuple((item.branch, item.stance) for item in templates)
        if len(set(keys)) != len(keys):
            raise ValueError("同一 branch/stance 不得重复 response act 模板")
        self.templates = tuple(sorted(
            templates, key=lambda item: item.stable_key()))
        self._by_key = {
            (item.branch, item.stance): item for item in self.templates
        }

    def resolve(
            self, selection: AnswerContentSelection,
            ) -> ResponseActGenerationTemplate:
        """为无已选命题的 selection 返回唯一模板，缺失时明确失败。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("response act registry 需要 AnswerContentSelection")
        if selection.selected_candidate_keys:
            raise ValueError("有已选命题时不得改走 response act 模板")
        branch = selection.request.goal.target_branch
        if branch is None:
            raise ValueError("response act generation 缺少目标 LanguageBranch")
        template = self._by_key.get((branch, selection.stance))
        if template is None:
            raise LookupError("当前 branch/stance 没有 response act 模板")
        return template

    def stable_key(self) -> tuple[int, ...]:
        """返回全部 response-act 模板的确定性配置键。"""
        return (
            len(self.templates),
            *(value for item in self.templates
              for value in _packed(item.stable_key())),
        )


class ResponseActDiscourseRouter:
    """在普通命题篇章 mapper 与空内容 response-act 篇章之间路由。"""

    def __init__(self, delegate, registry: ResponseActGenerationRegistry) -> None:
        """绑定普通命题 mapper 和共享 response-act 模板注册表。"""
        if not hasattr(delegate, "plan"):
            raise TypeError("response act discourse delegate 缺少 plan")
        if not isinstance(registry, ResponseActGenerationRegistry):
            raise TypeError("response act discourse registry 类型错误")
        self.delegate = delegate
        self.registry = registry

    def plan(self, selection: AnswerContentSelection) -> DiscoursePlan:
        """有已选命题时委托原 mapper，否则建立空节点但有上下文的篇章计划。"""
        if selection.selected_candidate_keys:
            return self.delegate.plan(selection)
        template = self.registry.resolve(selection)
        return DiscoursePlan(
            selection.stable_key(),
            (),
            (),
            (),
            template.context,
        )


class ResponseActPropositionRouter:
    """在普通命题 mapper 与无命题 response-act 计划之间路由。"""

    def __init__(self, delegate, registry: ResponseActGenerationRegistry) -> None:
        """绑定普通命题 mapper，并与其他层共享同一模板注册表。"""
        if not hasattr(delegate, "plan"):
            raise TypeError("response act proposition delegate 缺少 plan")
        if not isinstance(registry, ResponseActGenerationRegistry):
            raise TypeError("response act proposition registry 类型错误")
        self.delegate = delegate
        self.registry = registry

    def plan(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            ) -> PropositionPlan:
        """有已选命题时委托原 mapper，否则核验模板并保留空命题计划。"""
        if selection.selected_candidate_keys:
            return self.delegate.plan(selection, discourse)
        self.registry.resolve(selection)
        if discourse.selection_key != selection.stable_key():
            raise ValueError("response act proposition 收到漂移 discourse")
        if discourse.candidate_keys or discourse.dependencies:
            raise ValueError("response act discourse 不得伪造命题节点或依赖")
        return PropositionPlan(selection.stable_key(), ())


class ResponseActSyntaxRouter:
    """在普通命题 syntax mapper 与实际 response-act 句式之间路由。"""

    def __init__(self, delegate, registry: ResponseActGenerationRegistry) -> None:
        """绑定普通 syntax mapper 和注入式 response-act 模板。"""
        if not hasattr(delegate, "plan"):
            raise TypeError("response act syntax delegate 缺少 plan")
        if not isinstance(registry, ResponseActGenerationRegistry):
            raise TypeError("response act syntax registry 类型错误")
        self.delegate = delegate
        self.registry = registry

    def plan(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            propositions: PropositionPlan,
            ) -> SyntaxPlan:
        """用 stance 本体填入 S-07 槽，使无答案状态真实进入顺序与 surface。"""
        if selection.selected_candidate_keys:
            return self.delegate.plan(selection, discourse, propositions)
        template = self.registry.resolve(selection)
        selection_key = selection.stable_key()
        if (discourse.selection_key != selection_key
                or propositions.selection_key != selection_key):
            raise ValueError("response act syntax 收到漂移上游计划")
        if discourse.candidate_keys or propositions.propositions:
            raise ValueError("response act syntax 不得消费伪造命题")
        value = StructureSlotValue(template.slot.slot, selection.stance)
        sentence = PlannedSentence(
            template.sentence,
            template.slot.structure,
            0,
            (),
            (template.slot,),
            (value,),
            (),
            template.boundary,
            selection.request.goal.source,
            selection.request.goal.scope,
            selection.stance,
        )
        obligation = SyntaxLinearizationObligation(
            template.sentence,
            template.slot.structure,
            (value,),
            template.constraints,
            template.context,
            template.linearization_reason,
            selection.request.goal.source,
            selection.request.goal.scope,
        )
        return SyntaxPlan(
            selection_key,
            (sentence,),
            (),
            (obligation,),
        )


__all__ = [
    "ResponseActDiscourseRouter",
    "ResponseActGenerationRegistry",
    "ResponseActGenerationTemplate",
    "ResponseActPropositionRouter",
    "ResponseActSyntaxRouter",
]
