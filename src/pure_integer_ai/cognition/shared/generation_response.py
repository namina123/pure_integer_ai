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
from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    DiscoursePlan,
    PlannedSentence,
    PlannedProposition,
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
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity


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
    content_slots: tuple[StructureSlotDefinition, ...] = ()
    stance_filler: ObjectIdentity | None = None

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
        if self.stance_filler is not None:
            _identity(self.stance_filler, label="response act stance filler")
            if not self.content_slots:
                raise ValueError("来源化 stance filler 只允许用于完整多槽结构")
        _identity(
            self.sentence,
            label="response act sentence",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        if not isinstance(self.slot, StructureSlotDefinition):
            raise TypeError("response act slot 必须是 StructureSlotDefinition")
        if (not isinstance(self.content_slots, tuple)
                or any(not isinstance(item, StructureSlotDefinition)
                       for item in self.content_slots)):
            raise TypeError("response act content slots 类型错误")
        slots = (self.slot, *self.content_slots)
        if (len({item.slot for item in slots}) != len(slots)
                or any(item.structure != self.slot.structure for item in slots)):
            raise ValueError("response act slots 必须互异并属于同一结构")
        object.__setattr__(self, "content_slots", tuple(sorted(
            self.content_slots, key=lambda item: item.slot.stable_key())))
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
        # Keep the existing single-slot identity; multi-slot graphs have an
        # explicit extension containing every slot definition.
        if self.content_slots:
            result.extend((2, len(self.content_slots)))
            for item in self.content_slots:
                for identity in (item.structure, item.slot, item.role, item.value_type):
                    result.extend(_packed(identity.stable_key()))
        if self.stance_filler is not None:
            result.extend((3, *_packed(self.stance_filler.stable_key())))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ResponseActGenerationBinding:
    """同次选择的完整多槽值与来源证明，不赋予候选事实真值。"""

    selection_key: tuple[int, ...]
    template_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    values: tuple[StructureSlotValue, ...]
    evidence: tuple[tuple[int, ...], ...]
    discourse_context: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        for key in (self.selection_key, self.template_key):
            if not isinstance(key, tuple) or not key or any(type(v) is not int for v in key):
                raise ValueError("response act binding key 必须为非空严格整数")
        if not isinstance(self.source, SourceRef) or not isinstance(self.scope, ScopeIdentity):
            raise TypeError("response act binding source/scope 类型错误")
        if (not isinstance(self.values, tuple) or not self.values
                or any(not isinstance(item, StructureSlotValue) for item in self.values)):
            raise TypeError("response act binding values 必须为非空槽值")
        if len({item.slot for item in self.values}) != len(self.values):
            raise ValueError("response act binding 不得重复槽位")
        if (not isinstance(self.evidence, tuple) or not self.evidence
                or any(not isinstance(key, tuple) or not key
                       or any(type(v) is not int for v in key) for key in self.evidence)):
            raise ValueError("response act binding 必须保存完整整数证据")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("response act binding 证据不得重复")
        object.__setattr__(self, "values", tuple(sorted(
            self.values, key=lambda item: item.slot.stable_key())))
        object.__setattr__(self, "evidence", tuple(sorted(self.evidence)))
        if (type(self.discourse_context) is not tuple
                or any(not isinstance(item, ObjectIdentity) for item in self.discourse_context)
                or len(set(self.discourse_context)) != len(self.discourse_context)):
            raise ValueError("response act discourse 必须是唯一的来源化对象")

    def stable_key(self) -> tuple[int, ...]:
        result = (
            1, *_packed(self.selection_key), *_packed(self.template_key),
            *_packed(self.source.stable_key()), *_packed(self.scope.stable_key()),
            len(self.values),
            *(v for item in self.values for key in (item.slot.stable_key(), item.filler.stable_key())
              for v in _packed(key)),
            len(self.evidence), *(v for key in self.evidence for v in _packed(key)),
        )
        if self.discourse_context:
            result += (2, len(self.discourse_context),
                       *(v for item in self.discourse_context for v in _packed(item.stable_key())))
        return result


class ResponseActGenerationRegistry:
    """按完整 LanguageBranch 和 stance 身份索引注入的 response-act 模板。"""

    def __init__(
            self, templates: tuple[ResponseActGenerationTemplate, ...],
            bindings: tuple[ResponseActGenerationBinding, ...] = (),
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
        if (not isinstance(bindings, tuple)
                or any(not isinstance(item, ResponseActGenerationBinding) for item in bindings)):
            raise TypeError("response act registry bindings 类型错误")
        binding_keys = tuple((item.selection_key, item.template_key) for item in bindings)
        if len(set(binding_keys)) != len(binding_keys):
            raise ValueError("同次 selection/template 不得私选竞争绑定")
        template_keys = {item.stable_key() for item in self.templates if item.content_slots}
        if any(item.template_key not in template_keys for item in bindings):
            raise ValueError("response act binding 必须属于已注册多槽结构")
        self.bindings = tuple(sorted(bindings, key=lambda item: item.stable_key()))
        self._bindings = {key: item for key, item in zip(binding_keys, bindings)}

    def resolve(
            self, selection: AnswerContentSelection,
            ) -> ResponseActGenerationTemplate:
        """按目标分支和 stance 返回唯一模板，不丢弃 G-01 保留的候选。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("response act registry 需要 AnswerContentSelection")
        branch = selection.request.goal.target_branch
        if branch is None:
            raise ValueError("response act generation 缺少目标 LanguageBranch")
        template = self._by_key.get((branch, selection.stance))
        if template is None:
            raise LookupError("当前 branch/stance 没有 response act 模板")
        return template

    def matches(self, selection: AnswerContentSelection) -> bool:
        """返回当前 selection 是否有显式 response-act 模板。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("response act registry 需要 AnswerContentSelection")
        branch = selection.request.goal.target_branch
        return branch is not None and (branch, selection.stance) in self._by_key

    def binding(self, selection: AnswerContentSelection) -> ResponseActGenerationBinding | None:
        """多槽结构只能消费同次选择的完整绑定；不借用旧请求或单槽表达。"""
        template = self.resolve(selection)
        if not template.content_slots:
            return None
        binding = self._bindings.get((selection.stable_key(), template.stable_key()))
        if binding is None:
            raise LookupError("多槽 response act 缺少同次来源化绑定")
        goal = selection.request.goal
        if binding.source != goal.source or binding.scope != goal.scope:
            raise ValueError("response act binding 不属于当前 source/scope")
        if {item.slot for item in binding.values} != {item.slot for item in template.content_slots}:
            raise ValueError("response act binding 必须完整覆盖内容槽")
        if any(item.filler == selection.stance for item in binding.values):
            raise ValueError("内容槽不得重复 stance 槽")
        return binding

    def stable_key(self) -> tuple[int, ...]:
        """返回全部 response-act 模板的确定性配置键。"""
        result = (
            len(self.templates),
            *(value for item in self.templates
              for value in _packed(item.stable_key())),
        )
        if self.bindings:
            result += (2, len(self.bindings), *(v for item in self.bindings
                                               for v in _packed(item.stable_key())))
        return result


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
        """已注册 stance 保留决策候选节点；其他 stance 委托普通 mapper。"""
        if not self.registry.matches(selection):
            return self.delegate.plan(selection)
        template = self.registry.resolve(selection)
        binding = self.registry.binding(selection)
        selected = set(selection.selected_candidate_keys)
        open_questions = tuple(
            obligation
            for candidate in selection.request.candidates
            if candidate.stable_key() in selected
            and candidate.reasoning is not None
            for obligation in candidate.reasoning.unresolved
        )
        return DiscoursePlan(
            selection.stable_key(),
            selection.selected_candidate_keys,
            (),
            open_questions,
            tuple(sorted({*template.context, *(binding.discourse_context if binding is not None else ())},
                         key=lambda item: item.stable_key())),
            declaration_source=None if binding is None else binding.source,
            declaration_trace=() if binding is None else binding.stable_key(),
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
        """已注册 stance 保留决策 Evidence；surface 由 syntax 显式抑制。"""
        if not self.registry.matches(selection):
            return self.delegate.plan(selection, discourse)
        self.registry.resolve(selection)
        if discourse.selection_key != selection.stable_key():
            raise ValueError("response act proposition 收到漂移 discourse")
        if (discourse.candidate_keys != selection.selected_candidate_keys
                or discourse.dependencies):
            raise ValueError("response act discourse 候选或依赖漂移")
        selected = set(selection.selected_candidate_keys)
        propositions = tuple(
            PlannedProposition(
                candidate.stable_key(),
                candidate.proposition,
                candidate.state,
                candidate.source,
                candidate.scope,
                candidate.evidence,
                candidate.hypotheses,
                (),
                candidate.observation_evidence,
            )
            for candidate in selection.request.candidates
            if (candidate.stable_key() in selected
                and isinstance(candidate, GenerationCandidate))
        )
        return PropositionPlan(selection.stable_key(), propositions)


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
        """用注册 stance 本体填槽，使保留候选的非回答状态也进入 surface。"""
        if not self.registry.matches(selection):
            return self.delegate.plan(selection, discourse, propositions)
        template = self.registry.resolve(selection)
        binding = self.registry.binding(selection)
        selection_key = selection.stable_key()
        if (discourse.selection_key != selection_key
                or propositions.selection_key != selection_key):
            raise ValueError("response act syntax 收到漂移上游计划")
        proposition_candidate_keys = {
            candidate.stable_key()
            for candidate in selection.request.candidates
            if (candidate.stable_key() in set(selection.selected_candidate_keys)
                and isinstance(candidate, GenerationCandidate))
        }
        if (discourse.candidate_keys != selection.selected_candidate_keys
                or {item.candidate_key for item in propositions.propositions}
                != proposition_candidate_keys):
            raise ValueError("response act syntax 决策候选 Evidence 漂移")
        value = StructureSlotValue(template.slot.slot, template.stance_filler or selection.stance)
        values = (value,) if binding is None else (value, *binding.values)
        if (discourse.declaration_source != (None if binding is None else binding.source)
                or discourse.declaration_trace != (() if binding is None else binding.stable_key())):
            raise ValueError("response act syntax 来源化绑定与 discourse 漂移")
        sentence = PlannedSentence(
            template.sentence,
            template.slot.structure,
            0,
            (),
            (template.slot, *template.content_slots),
            values,
            (),
            template.boundary,
            selection.request.goal.source,
            selection.request.goal.scope,
            selection.stance,
            response_act_value=value if template.stance_filler is not None else None,
        )
        obligation = SyntaxLinearizationObligation(
            template.sentence,
            template.slot.structure,
            values,
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
            selection.selected_candidate_keys,
        )


__all__ = [
    "ResponseActDiscourseRouter",
    "ResponseActGenerationRegistry",
    "ResponseActGenerationBinding",
    "ResponseActGenerationTemplate",
    "ResponseActPropositionRouter",
    "ResponseActSyntaxRouter",
]
