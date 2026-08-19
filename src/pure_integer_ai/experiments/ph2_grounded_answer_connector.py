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
    surface_pattern_structure_id,
    surface_pattern_structure_key,
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


def _structure_template_id(
        pattern: LearnedSurfacePattern,
        target: "GroundedAnswerConnectorTarget",
        ) -> int:
    """按 part 形状与 connector match key 生成共享结构理论身份。"""
    return _stable_id({
        "branch": list(target.language_branch.stable_key()),
        "predicate": list(target.proposition.predicate.stable_key()),
        "proposition_structure": list(
            target.proposition.structure.stable_key()),
        "structure_id": surface_pattern_structure_id(pattern),
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


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerClaimInput:
    """不依赖课程 episode 的单 claim typed 输入。

    claim surface 必须由调用方从本次实际候选/来源恢复；该对象不保存
    expected answer、response-act label、课程 episode 或 evaluator 字段。
    """

    claim_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.claim_text, str) or not self.claim_text:
            raise GroundedAnswerConnectorError(
                "generic answer claim surface 必须是非空字符串")


def _claim_text(
        question: GroundedQuestionEpisode | GroundedAnswerClaimInput,
        ) -> str:
    """恢复单 claim ANSWER 的唯一 Evidence 表面。"""
    if isinstance(question, GroundedAnswerClaimInput):
        return question.claim_text
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
    structure_id: int
    structure_key: tuple[tuple[int, int], ...]
    connector: ObjectIdentity
    support_episode_ids: tuple[str, ...]
    support_teacher_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.choice_id) is not int or self.choice_id <= 0:
            raise GroundedAnswerConnectorError("pattern choice id 非法")
        if type(self.pattern_id) is not int or self.pattern_id <= 0:
            raise GroundedAnswerConnectorError("pattern id 非法")
        if type(self.structure_id) is not int or self.structure_id <= 0:
            raise GroundedAnswerConnectorError("pattern structure id 非法")
        if (not isinstance(self.structure_key, tuple)
                or not self.structure_key
                or any(not isinstance(item, tuple) or len(item) != 2
                       or any(type(value) is not int for value in item)
                       for item in self.structure_key)):
            raise GroundedAnswerConnectorError("pattern structure key 非法")
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
class GroundedAnswerStructureOption:
    """一个可先于 lexical variant 显式选择的 surface 结构候选。"""

    structure_id: int
    structure_key: tuple[tuple[int, int], ...]
    structure: ObjectIdentity
    sentence: ObjectIdentity
    slots: tuple[StructureSlotDefinition, ...]
    pattern_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.structure_id) is not int or self.structure_id <= 0:
            raise GroundedAnswerConnectorError("structure option id 非法")
        if (not isinstance(self.structure_key, tuple)
                or not self.structure_key
                or len(self.structure_key) != len(self.slots)
                or any(not isinstance(item, tuple) or len(item) != 2
                       or any(type(value) is not int for value in item)
                       for item in self.structure_key)):
            raise GroundedAnswerConnectorError("structure option key 非法")
        for label, value in (
                ("structure", self.structure), ("sentence", self.sentence)):
            if (not isinstance(value, ObjectIdentity)
                    or value.object_kind != OBJECT_STRUCTURE_CONCEPT):
                raise TypeError(f"structure option {label} 类型错误")
        if (not isinstance(self.slots, tuple) or not self.slots
                or any(not isinstance(item, StructureSlotDefinition)
                       or item.structure != self.structure
                       for item in self.slots)):
            raise GroundedAnswerConnectorError("structure option slots 非法")
        if (not isinstance(self.pattern_ids, tuple) or not self.pattern_ids
                or any(type(value) is not int or value <= 0
                       for value in self.pattern_ids)
                or self.pattern_ids != tuple(sorted(set(self.pattern_ids)))):
            raise GroundedAnswerConnectorError(
                "structure option pattern ids 非规范")

    def choice_key(self) -> tuple[int, ...]:
        """返回不随 lexical 成员增减变化的结构本体竞争键。"""
        values = [self.structure_id, len(self.structure_key)]
        for item in self.structure_key:
            values.extend(item)
        values.extend((
            len(self.structure.stable_key()), *self.structure.stable_key(),
            len(self.sentence.stable_key()), *self.sentence.stable_key(),
            len(self.slots),
        ))
        for slot in self.slots:
            for identity in (
                    slot.structure, slot.slot, slot.role, slot.value_type):
                key = identity.stable_key()
                values.extend((len(key), *key))
        return tuple(values)

    def stable_key(self) -> tuple[int, ...]:
        """返回结构本体键与本次 compilation 的 lexical 成员。"""
        choice = self.choice_key()
        return (
            len(choice),
            *choice,
            len(self.pattern_ids),
            *self.pattern_ids,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerStructureSelection:
    """完整结构竞争集与调用者显式采用的唯一结构。"""

    options: tuple[GroundedAnswerStructureOption, ...]
    selected: GroundedAnswerStructureOption
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.options, tuple) or not self.options
                or any(not isinstance(item, GroundedAnswerStructureOption)
                       for item in self.options)):
            raise TypeError("grounded structure selection options 类型错误")
        ids = tuple(item.structure_id for item in self.options)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedAnswerConnectorError(
                "grounded structure options 非唯一递增")
        if self.selected not in self.options:
            raise GroundedAnswerConnectorError(
                "grounded selected structure 不属于竞争集")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(value) is not int for value in self.trace)):
            raise GroundedAnswerConnectorError(
                "grounded structure selection trace 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回全部结构候选、selected option 与显式选择 trace。"""
        values = [len(self.options)]
        for option in self.options:
            key = option.stable_key()
            values.extend((len(key), *key))
        selected = self.selected.stable_key()
        values.extend((len(selected), *selected, len(self.trace), *self.trace))
        return tuple(values)


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
    structures: tuple[GroundedAnswerStructureOption, ...]
    variants: tuple[GroundedAnswerConnectorVariant, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.value_protocol, LanguageConnectorValueProtocol):
            raise TypeError("grounded connector value protocol 类型错误")
        if (not isinstance(self.structures, tuple) or not self.structures
                or any(not isinstance(item, GroundedAnswerStructureOption)
                       for item in self.structures)):
            raise GroundedAnswerConnectorError("grounded structures 不能为空")
        structure_ids = tuple(item.structure_id for item in self.structures)
        if (structure_ids != tuple(sorted(structure_ids))
                or len(set(structure_ids)) != len(structure_ids)):
            raise GroundedAnswerConnectorError(
                "grounded structure id 非唯一递增")
        if (not isinstance(self.variants, tuple) or not self.variants
                or any(not isinstance(item, GroundedAnswerConnectorVariant)
                       for item in self.variants)):
            raise GroundedAnswerConnectorError("grounded variants 不能为空")
        ids = tuple(item.option.pattern_id for item in self.variants)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedAnswerConnectorError("grounded variant id 非唯一递增")
        for structure in self.structures:
            variants = tuple(
                item for item in self.variants
                if item.option.structure_id == structure.structure_id)
            if (tuple(item.option.pattern_id for item in variants)
                    != structure.pattern_ids
                    or any(item.option.structure_key != structure.structure_key
                           or item.template.structure != structure.structure
                           or item.template.sentence != structure.sentence
                           or item.template.slots != structure.slots
                           for item in variants)):
                raise GroundedAnswerConnectorError(
                    "grounded structure option 与 lexical variants 漂移")
        if {item.option.structure_id for item in self.variants} != set(
                structure_ids):
            raise GroundedAnswerConnectorError(
                "grounded structure 未精确覆盖 variants")

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

    def select_structure(
            self, structure_id: int,
            ) -> GroundedAnswerStructureOption:
        """按调用者显式 identity 返回唯一 structure option。"""
        if type(structure_id) is not int or structure_id <= 0:
            raise GroundedAnswerConnectorError("selected structure id 非法")
        matches = tuple(
            item for item in self.structures
            if item.structure_id == structure_id)
        if len(matches) != 1:
            raise GroundedAnswerConnectorError(
                "selected structure 不属于当前 grounded compilation")
        return matches[0]

    def select_within_structure(
            self,
            structure_id: int,
            pattern_id: int,
            ) -> GroundedAnswerConnectorVariant:
        """先核验 structure，再只在其 lexical 成员中采用 pattern。"""
        structure = self.select_structure(structure_id)
        if pattern_id not in structure.pattern_ids:
            raise GroundedAnswerConnectorError(
                "selected pattern 不属于已选 grounded structure")
        variant = self.select(pattern_id)
        if variant.option.structure_id != structure.structure_id:
            raise GroundedAnswerConnectorError(
                "selected pattern structure identity 漂移")
        return variant


def _value_protocol() -> LanguageConnectorValueProtocol:
    """建立本课程共享的四类最小 slot 读取指令。"""
    return LanguageConnectorValueProtocol(*tuple(
        minimal_instruction_identity((_NAMESPACE, 1, index))
        for index in range(1, 5)
    ))


def _variant(
        pattern: LearnedSurfacePattern,
        question: GroundedQuestionEpisode | GroundedAnswerClaimInput,
        target: GroundedAnswerConnectorTarget,
        surface_protocol: GenerationSurfaceProtocol,
        value_protocol: LanguageConnectorValueProtocol,
        ) -> GroundedAnswerConnectorVariant:
    """把一个已筛选单 claim pattern 无损编译为独立 connector。"""
    claim_text = _claim_text(question)
    pattern_key = (
        _NAMESPACE, 2, pattern.pattern_id, _template_id(pattern, target))
    structure_id = surface_pattern_structure_id(pattern)
    structure_key = (
        _NAMESPACE, 3, structure_id, _structure_template_id(pattern, target))
    connector = structure_concept_identity((*pattern_key, 1))
    structure = structure_concept_identity((*structure_key, 1))
    value_type = concept_identity((*structure_key, 2))
    slots = tuple(
        StructureSlotDefinition(
            structure,
            structure_concept_identity((*structure_key, 10, index)),
            role_identity((*structure_key, 11, index)),
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
            structure_concept_identity((*structure_key, 60, index)),
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
        structure_concept_identity((*structure_key, 3)),
        structure,
        slots,
        tuple(bindings),
        structure_concept_identity((*structure_key, 4)),
        tuple(item.constraint for item in orders),
        structure_concept_identity((*structure_key, 5)),
        (),
        minimal_instruction_identity((*structure_key, 6)),
        minimal_instruction_identity((*structure_key, 7)),
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
        structure_id,
        surface_pattern_structure_key(pattern),
        connector,
        pattern.support_episode_ids,
        pattern.support_teacher_keys,
    )
    return GroundedAnswerConnectorVariant(
        option, template, policy, tuple(aliases), orders)


def _structure_options(
        variants: tuple[GroundedAnswerConnectorVariant, ...],
        ) -> tuple[GroundedAnswerStructureOption, ...]:
    """把同一 part 形状的 lexical variants 汇入共享结构候选。"""
    groups: dict[int, list[GroundedAnswerConnectorVariant]] = {}
    for variant in variants:
        groups.setdefault(variant.option.structure_id, []).append(variant)
    options = []
    for structure_id, members in groups.items():
        ordered = tuple(sorted(
            members, key=lambda item: item.option.pattern_id))
        first = ordered[0]
        options.append(GroundedAnswerStructureOption(
            structure_id,
            first.option.structure_key,
            first.template.structure,
            first.template.sentence,
            first.template.slots,
            tuple(item.option.pattern_id for item in ordered),
        ))
    return tuple(sorted(options, key=lambda item: item.structure_id))


def compile_grounded_answer_connectors(
        model: GroundedAnswerSurfaceModel,
        question: GroundedQuestionEpisode | GroundedAnswerClaimInput,
        target: GroundedAnswerConnectorTarget,
        surface_protocol: GenerationSurfaceProtocol,
        *,
        carrier_kind: str = "PLAIN_TEXT",
        ) -> GroundedAnswerConnectorCompilation:
    """编译全部合法单 claim ANSWER pattern，不按稳定序暗中采用任何一个。"""
    if not isinstance(model, GroundedAnswerSurfaceModel):
        raise TypeError("grounded connector model 类型错误")
    if not isinstance(question, (GroundedQuestionEpisode,
                                 GroundedAnswerClaimInput)):
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
    return GroundedAnswerConnectorCompilation(
        value_protocol, _structure_options(variants), variants)


def select_grounded_answer_structure(
        compilation: GroundedAnswerConnectorCompilation,
        structure_id: int,
        ) -> GroundedAnswerStructureSelection:
    """显式采用一个结构竞争项，供 lexical 选择和 syntax mapper 消费。"""
    if not isinstance(compilation, GroundedAnswerConnectorCompilation):
        raise TypeError("grounded structure compilation 类型错误")
    selected = compilation.select_structure(structure_id)
    trace = (
        _NAMESPACE,
        90,
        structure_id,
        len(compilation.structures),
        *(item.structure_id for item in compilation.structures),
    )
    return GroundedAnswerStructureSelection(
        compilation.structures, selected, trace)


def build_grounded_answer_connector(
        compilation: GroundedAnswerConnectorCompilation,
        structure_id: int,
        pattern_id: int,
        surface_protocol: GenerationSurfaceProtocol,
        ) -> tuple[GroundedAnswerConnectorVariant, LanguageGenerationConnector]:
    """按显式 pattern id 建立单模板 connector，杜绝歧义排序。"""
    if not isinstance(compilation, GroundedAnswerConnectorCompilation):
        raise TypeError("grounded connector compilation 类型错误")
    if not isinstance(surface_protocol, GenerationSurfaceProtocol):
        raise TypeError("grounded connector surface protocol 类型错误")
    variant = compilation.select_within_structure(structure_id, pattern_id)
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
    "GroundedAnswerClaimInput",
    "GroundedAnswerConnectorCompilation",
    "GroundedAnswerConnectorError",
    "GroundedAnswerConnectorTarget",
    "GroundedAnswerConnectorVariant",
    "GroundedAnswerOrderRequirement",
    "GroundedAnswerPatternOption",
    "GroundedAnswerStructureOption",
    "GroundedAnswerStructureSelection",
    "build_grounded_answer_connector",
    "compile_grounded_answer_connectors",
    "select_grounded_answer_structure",
]
