"""把单 claim grounded-answer pattern 编译成显式 connector variant。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_PROPOSITION,
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
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
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorSlotBinding,
    LanguageConnectorSurfaceDirective,
    LanguageConnectorSurfaceRuntimePolicy,
    LanguageConnectorTemplateRuntimePolicy,
    LanguageConnectorValueProtocol,
    LanguageGenerationConnector,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorRuntimePolicy,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedQuestionEpisode,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    GroundedAnswerSurfaceModel,
    LearnedSurfacePattern,
    PATTERN_CLAIM,
    PATTERN_LITERAL,
)


_NAMESPACE = 20916


# object-model: exception
class GroundedAnswerConnectorError(ValueError):
    """pattern 不能无损编译，或调用者未显式选择合法 variant。"""


def _text_id(text: str) -> int:
    """从 UTF-8 文本产生稳定正整数，仅作一等 literal 身份。"""
    if not isinstance(text, str) or not text:
        raise GroundedAnswerConnectorError("literal 文本不能为空")
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8],
                           "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


def _stable_id(value: object) -> int:
    """从规范 JSON 值生成稳定正整数身份。"""
    payload = canonical_json_bytes(value)
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _template_id(
        pattern: LearnedSurfacePattern,
        target: "GroundedAnswerConnectorTarget",
        ) -> int:
    """按 pattern 与 connector 精确匹配键生成理论身份。"""
    return _stable_id({
        "branch": list(target.language_branch.stable_key()),
        "pattern_id": pattern.pattern_id,
        "predicate": list(target.proposition.predicate.stable_key()),
        "proposition_structure": list(
            target.proposition.structure.stable_key()),
        "version": 1,
    })


def _choice_id(
        pattern: LearnedSurfacePattern,
        target: "GroundedAnswerConnectorTarget",
        ) -> int:
    """从理论 variant 与具体目标命题生成本次显式选择身份。"""
    return _stable_id({
        "branch": list(target.language_branch.stable_key()),
        "pattern_id": pattern.pattern_id,
        "proposition": list(target.proposition.stable_key()),
        "version": 1,
    })


def _claim_text(question: GroundedQuestionEpisode) -> str:
    """恢复单 claim ANSWER 的唯一 Evidence 表面。"""
    plan = question.answer_plan
    if plan.response_act != "ANSWER" or len(plan.ordered_claim_ids) != 1:
        raise GroundedAnswerConnectorError(
            "首轮 connector 只接受单 claim ANSWER")
    claim_id = plan.ordered_claim_ids[0]
    values = {
        item.claim_text for item in question.evidence
        if item.proposition_id == claim_id
    }
    if len(values) != 1:
        raise GroundedAnswerConnectorError("目标 claim 缺少唯一 Evidence 表面")
    return next(iter(values))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerConnectorTarget:
    """把新问题的单个 typed Proposition 绑定到目标语言分支。"""

    proposition: BoundProposition
    language_branch: ObjectIdentity
    representation_family: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("grounded connector proposition 类型错误")
        if (not isinstance(self.language_branch, ObjectIdentity)
                or self.language_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ValueError("grounded connector language branch 类型错误")
        if (not isinstance(self.representation_family, tuple)
                or not self.representation_family
                or any(type(value) is not int
                       for value in self.representation_family)):
            raise GroundedAnswerConnectorError(
                "representation family 必须是非空严格整数 tuple")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerAliasRequirement:
    """声明一个 slot filler 所需的 R-01 Unicode Representation。"""

    filler: ObjectIdentity
    slot: ObjectIdentity
    representation: ObjectIdentity
    part_kind: str
    part_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.filler, ObjectIdentity):
            raise TypeError("grounded alias filler 类型错误")
        if (not isinstance(self.slot, ObjectIdentity)
                or self.slot.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise TypeError("grounded alias slot 类型错误")
        if (not isinstance(self.representation, ObjectIdentity)
                or self.representation.object_kind != OBJECT_REPRESENTATION):
            raise TypeError("grounded alias representation 类型错误")
        if self.part_kind not in {PATTERN_LITERAL, PATTERN_CLAIM}:
            raise GroundedAnswerConnectorError("grounded alias part kind 非法")
        if type(self.part_ordinal) is not int or self.part_ordinal < 0:
            raise GroundedAnswerConnectorError("grounded alias ordinal 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerOrderRequirement:
    """声明 pattern 相邻 part 在 S-07 中必须保持的先后关系。"""

    constraint: ObjectIdentity
    before_slot: ObjectIdentity
    after_slot: ObjectIdentity

    def __post_init__(self) -> None:
        for value in (self.constraint, self.before_slot, self.after_slot):
            if (not isinstance(value, ObjectIdentity)
                    or value.object_kind != OBJECT_STRUCTURE_CONCEPT):
                raise TypeError("grounded order identity 类型错误")
        if self.before_slot == self.after_slot:
            raise GroundedAnswerConnectorError("grounded order 不得自环")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerPatternOption:
    """一个可由调用者显式采用的 lexical realization variant。"""

    choice_id: int
    pattern_id: int
    connector: ObjectIdentity
    support_episode_ids: tuple[str, ...]
    support_teacher_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.choice_id) is not int or self.choice_id <= 0:
            raise GroundedAnswerConnectorError("pattern choice id 非法")
        if type(self.pattern_id) is not int or self.pattern_id <= 0:
            raise GroundedAnswerConnectorError("pattern id 非法")
        if not isinstance(self.connector, ObjectIdentity):
            raise TypeError("pattern option connector 类型错误")
        if (not self.support_episode_ids
                or self.support_episode_ids != tuple(sorted(
                    set(self.support_episode_ids)))):
            raise GroundedAnswerConnectorError(
                "pattern option episode 追溯非规范")
        if (not isinstance(self.support_teacher_keys, tuple)
                or any(not isinstance(key, tuple) or not key
                       or any(type(value) is not int for value in key)
                       for key in self.support_teacher_keys)
                or not self.support_teacher_keys
                or self.support_teacher_keys != tuple(sorted(
                    set(self.support_teacher_keys)))):
            raise GroundedAnswerConnectorError("pattern option 缺少形成追溯")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerConnectorVariant:
    """一个 pattern 的 connector 理论、运行策略、alias 与顺序义务。"""

    option: GroundedAnswerPatternOption
    template: LanguageGenerationConnectorTemplate
    runtime_policy: LanguageGenerationConnectorRuntimePolicy
    aliases: tuple[GroundedAnswerAliasRequirement, ...]
    order_requirements: tuple[GroundedAnswerOrderRequirement, ...]

    def __post_init__(self) -> None:
        if self.option.connector != self.template.connector:
            raise GroundedAnswerConnectorError(
                "pattern option 与 connector template 漂移")
        if tuple(item.connector for item in self.runtime_policy.templates) != (
                self.template.connector,):
            raise GroundedAnswerConnectorError(
                "grounded runtime policy 未精确绑定单一 variant")
        if len(self.aliases) != len(self.template.slots):
            raise GroundedAnswerConnectorError("grounded alias 未逐槽覆盖")
        ordered_aliases = tuple(sorted(
            self.aliases, key=lambda item: item.part_ordinal))
        if tuple(item.part_ordinal for item in ordered_aliases) != tuple(
                range(len(ordered_aliases))):
            raise GroundedAnswerConnectorError("grounded alias ordinal 不连续")
        if {item.slot for item in ordered_aliases} != {
                item.slot for item in self.template.slots}:
            raise GroundedAnswerConnectorError("grounded alias slot 覆盖漂移")
        if len(self.order_requirements) != max(0, len(self.template.slots) - 1):
            raise GroundedAnswerConnectorError("grounded 顺序义务未覆盖相邻 part")
        expected_orders = tuple(
            (before.slot, after.slot)
            for before, after in zip(ordered_aliases, ordered_aliases[1:])
        )
        actual_orders = tuple(
            (item.before_slot, item.after_slot)
            for item in self.order_requirements)
        if actual_orders != expected_orders:
            raise GroundedAnswerConnectorError("grounded 顺序义务与 part 序漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerConnectorCompilation:
    """保存共同读取协议和全部可显式选择的合法 variant。"""

    value_protocol: LanguageConnectorValueProtocol
    variants: tuple[GroundedAnswerConnectorVariant, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.value_protocol, LanguageConnectorValueProtocol):
            raise TypeError("grounded connector value protocol 类型错误")
        if (not isinstance(self.variants, tuple) or not self.variants
                or any(not isinstance(item, GroundedAnswerConnectorVariant)
                       for item in self.variants)):
            raise GroundedAnswerConnectorError("grounded variants 不能为空")
        ids = tuple(item.option.pattern_id for item in self.variants)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedAnswerConnectorError("grounded variant id 非唯一递增")

    def select(self, pattern_id: int) -> GroundedAnswerConnectorVariant:
        """按调用者显式给出的 pattern identity 返回唯一 variant。"""
        if type(pattern_id) is not int or pattern_id <= 0:
            raise GroundedAnswerConnectorError("selected pattern id 非法")
        matches = tuple(
            item for item in self.variants
            if item.option.pattern_id == pattern_id)
        if len(matches) != 1:
            raise GroundedAnswerConnectorError(
                "selected pattern 不属于当前 grounded compilation")
        return matches[0]


def _value_protocol() -> LanguageConnectorValueProtocol:
    """建立本课程共享的四类最小 slot 读取指令。"""
    return LanguageConnectorValueProtocol(*tuple(
        minimal_instruction_identity((_NAMESPACE, 1, index))
        for index in range(1, 5)
    ))


def _variant(
        pattern: LearnedSurfacePattern,
        question: GroundedQuestionEpisode,
        target: GroundedAnswerConnectorTarget,
        surface_protocol: GenerationSurfaceProtocol,
        value_protocol: LanguageConnectorValueProtocol,
        ) -> GroundedAnswerConnectorVariant:
    """把一个已筛选单 claim pattern 无损编译为独立 connector。"""
    claim_text = _claim_text(question)
    pattern_key = (
        _NAMESPACE, 2, pattern.pattern_id, _template_id(pattern, target))
    connector = structure_concept_identity((*pattern_key, 1))
    structure = structure_concept_identity((*pattern_key, 2))
    value_type = concept_identity((*pattern_key, 3))
    slots = tuple(
        StructureSlotDefinition(
            structure,
            structure_concept_identity((*pattern_key, 10, index)),
            role_identity((*pattern_key, 11, index)),
            value_type,
        )
        for index, _part in enumerate(pattern.parts, start=1)
    )
    bindings = []
    directives = []
    runtime = []
    aliases = []
    for ordinal, (part, slot) in enumerate(
            zip(pattern.parts, slots, strict=True), start=1):
        if part.kind == PATTERN_CLAIM:
            source = value_protocol.proposition_source
            filler = target.proposition.template
            text = claim_text
            constant = None
        elif part.kind == PATTERN_LITERAL:
            source = value_protocol.constant_source
            filler = concept_identity((
                *pattern_key, 20, ordinal, _text_id(part.literal)))
            text = part.literal
            constant = filler
        else:
            raise GroundedAnswerConnectorError("pattern part kind 未注册")
        bindings.append(LanguageConnectorSlotBinding(
            structure_concept_identity((*pattern_key, 30, ordinal)),
            slot.slot,
            source,
            constant=constant,
        ))
        directives.append(LanguageConnectorSurfaceDirective(
            structure_concept_identity((*pattern_key, 40, ordinal)),
            slot.slot,
            surface_protocol.emit_action,
            minimal_instruction_identity((*pattern_key, 41, ordinal)),
            structure_concept_identity((*pattern_key, 42, ordinal)),
            (),
        ))
        runtime.append(LanguageConnectorSurfaceRuntimePolicy(
            slot.slot,
            (*pattern_key, 50, ordinal),
            AliasRouteSearchBudget(32, 32, 32),
            (*pattern_key, 51, ordinal),
        ))
        aliases.append(GroundedAnswerAliasRequirement(
            filler,
            slot.slot,
            representation_identity(
                target.representation_family,
                tuple(ord(char) for char in text),
            ),
            part.kind,
            ordinal - 1,
        ))
    orders = tuple(
        GroundedAnswerOrderRequirement(
            structure_concept_identity((*pattern_key, 60, index)),
            before.slot,
            after.slot,
        )
        for index, (before, after) in enumerate(
            zip(slots, slots[1:]), start=1)
    )
    template = LanguageGenerationConnectorTemplate(
        connector,
        target.language_branch,
        target.proposition.structure,
        target.proposition.predicate,
        structure_concept_identity((*pattern_key, 4)),
        structure,
        slots,
        tuple(bindings),
        structure_concept_identity((*pattern_key, 5)),
        tuple(item.constraint for item in orders),
        structure_concept_identity((*pattern_key, 6)),
        (),
        minimal_instruction_identity((*pattern_key, 7)),
        minimal_instruction_identity((*pattern_key, 8)),
        tuple(directives),
    )
    policy = LanguageGenerationConnectorRuntimePolicy(
        (*pattern_key, 70),
        StructureOrderSearchBudget(max(16, len(slots) * len(slots) * 2)),
        (LanguageConnectorTemplateRuntimePolicy(
            connector, tuple(runtime)),),
    )
    option = GroundedAnswerPatternOption(
        _choice_id(pattern, target),
        pattern.pattern_id,
        connector,
        pattern.support_episode_ids,
        pattern.support_teacher_keys,
    )
    return GroundedAnswerConnectorVariant(
        option, template, policy, tuple(aliases), orders)


def compile_grounded_answer_connectors(
        model: GroundedAnswerSurfaceModel,
        question: GroundedQuestionEpisode,
        target: GroundedAnswerConnectorTarget,
        surface_protocol: GenerationSurfaceProtocol,
        *,
        carrier_kind: str = "PLAIN_TEXT",
        ) -> GroundedAnswerConnectorCompilation:
    """编译全部合法单 claim ANSWER pattern，不按稳定序暗中采用任何一个。"""
    if not isinstance(model, GroundedAnswerSurfaceModel):
        raise TypeError("grounded connector model 类型错误")
    if not isinstance(question, GroundedQuestionEpisode):
        raise TypeError("grounded connector question 类型错误")
    if not isinstance(target, GroundedAnswerConnectorTarget):
        raise TypeError("grounded connector target 类型错误")
    if not isinstance(surface_protocol, GenerationSurfaceProtocol):
        raise TypeError("grounded connector surface protocol 类型错误")
    if target.proposition.template.object_kind != OBJECT_PROPOSITION:
        raise GroundedAnswerConnectorError("target 必须绑定 Proposition 本体")
    _claim_text(question)
    patterns = tuple(
        item for item in model.patterns
        if (item.response_act == "ANSWER"
            and item.carrier_kind == carrier_kind
            and item.claim_count == 1
            and sum(part.kind == PATTERN_CLAIM for part in item.parts) == 1)
    )
    if not patterns:
        raise GroundedAnswerConnectorError(
            "当前模型没有可编译的单 claim ANSWER pattern")
    value_protocol = _value_protocol()
    variants = tuple(sorted(
        (
            _variant(item, question, target, surface_protocol, value_protocol)
            for item in patterns
        ),
        key=lambda item: item.option.pattern_id,
    ))
    return GroundedAnswerConnectorCompilation(value_protocol, variants)


def build_grounded_answer_connector(
        compilation: GroundedAnswerConnectorCompilation,
        pattern_id: int,
        surface_protocol: GenerationSurfaceProtocol,
        ) -> tuple[GroundedAnswerConnectorVariant, LanguageGenerationConnector]:
    """按显式 pattern id 建立单模板 connector，杜绝歧义排序。"""
    if not isinstance(compilation, GroundedAnswerConnectorCompilation):
        raise TypeError("grounded connector compilation 类型错误")
    if not isinstance(surface_protocol, GenerationSurfaceProtocol):
        raise TypeError("grounded connector surface protocol 类型错误")
    variant = compilation.select(pattern_id)
    connector = LanguageGenerationConnector(
        LanguageGenerationConnectorRegistry(
            compilation.value_protocol,
            (variant.template,),
        ),
        variant.runtime_policy,
        surface_protocol,
    )
    return variant, connector


__all__ = [
    "GroundedAnswerAliasRequirement",
    "GroundedAnswerConnectorCompilation",
    "GroundedAnswerConnectorError",
    "GroundedAnswerConnectorTarget",
    "GroundedAnswerConnectorVariant",
    "GroundedAnswerOrderRequirement",
    "GroundedAnswerPatternOption",
    "build_grounded_answer_connector",
    "compile_grounded_answer_connectors",
]
