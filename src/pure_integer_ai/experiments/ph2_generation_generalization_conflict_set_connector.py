"""把 ``CONFLICT_SET`` typed plan 编译为真实的多句 G-03 connector。

该模块只承接公开 typed contract 和调用者提供的真实 ``BoundProposition``；
不读取 private label，不改变旧单 proposition ``CONFLICT``，也不把 claim/source
字符串伪装成 Core 语义对象。每个 claim 形成一个句子：命题本体槽静默，claim
literal 槽通过 R-01 发出；句间顺序由 claim 顺序形成来源化篇章声明。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    DiscourseDependency,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorDiscourseDeclaration,
    LanguageConnectorOrdinalDefinition,
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
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetPlan,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_surface import (
    ConflictSetGeneratedSentence,
    generate_conflict_set_sentences,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderSearchBudget,
)


class ConflictSetConnectorCompileError(ValueError):
    """CONFLICT_SET connector 输入或模板闭包不完整。"""


@dataclass(frozen=True, slots=True)
class ConflictSetSentenceBinding:
    """把一个 declared claim 绑定到真实 Proposition 和公开 surface。"""

    claim_id: str
    proposition: BoundProposition
    claim_surface: str

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ConflictSetConnectorCompileError("claim_id 必须是非空文本")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("conflict sentence proposition 类型错误")
        if not isinstance(self.claim_surface, str) or not self.claim_surface:
            raise ConflictSetConnectorCompileError("claim_surface 必须是非空文本")


@dataclass(frozen=True, slots=True)
class ConflictSetSentenceCompilation:
    """一个 claim 的 connector template、槽和来源归属。"""

    claim_id: str
    proposition: BoundProposition
    source_ids: tuple[str, ...]
    claim_surface: str
    template: LanguageGenerationConnectorTemplate
    proposition_slot: ObjectIdentity
    claim_slot: ObjectIdentity
    claim_filler: ObjectIdentity
    claim_representation: ObjectIdentity
    order_constraint: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ConflictSetConnectorCompileError("sentence claim_id 非法")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("sentence proposition 类型错误")
        if (not isinstance(self.source_ids, tuple) or not self.source_ids
                or self.source_ids != tuple(sorted(set(self.source_ids)))):
            raise ConflictSetConnectorCompileError("sentence source_ids 非规范")
        if not all(isinstance(item, str) and item for item in self.source_ids):
            raise ConflictSetConnectorCompileError("sentence source_ids 含非法项")
        if not isinstance(self.claim_surface, str) or not self.claim_surface:
            raise ConflictSetConnectorCompileError("sentence claim_surface 非法")
        if not isinstance(self.template, LanguageGenerationConnectorTemplate):
            raise TypeError("sentence template 类型错误")
        if self.proposition_slot == self.claim_slot:
            raise ConflictSetConnectorCompileError("命题槽和 claim 槽不得相同")
        slots = {item.slot for item in self.template.slots}
        if {self.proposition_slot, self.claim_slot} != slots:
            raise ConflictSetConnectorCompileError("sentence slot 覆盖不完整")
        if not isinstance(self.claim_filler, ObjectIdentity):
            raise TypeError("sentence claim_filler 类型错误")
        if (not isinstance(self.claim_representation, ObjectIdentity)
                or self.claim_representation.object_kind
                != OBJECT_REPRESENTATION):
            raise TypeError("sentence claim_representation 类型错误")
        if self.template.constraints != (self.order_constraint,):
            raise ConflictSetConnectorCompileError("sentence 顺序义务未唯一注册")


class ConflictSetDiscourseDeclarations:
    """按 declared claim 顺序为真实 selected candidates 生成篇章声明。"""

    def __init__(
            self,
            bindings: tuple[ConflictSetSentenceBinding, ...],
            source: SourceRef,
            namespace: tuple[int, ...],
            ) -> None:
        if not isinstance(bindings, tuple) or not bindings:
            raise TypeError("conflict discourse bindings 必须非空 tuple")
        if any(not isinstance(item, ConflictSetSentenceBinding)
               for item in bindings):
            raise TypeError("conflict discourse bindings 类型错误")
        claim_ids = tuple(item.claim_id for item in bindings)
        if len(set(claim_ids)) != len(claim_ids):
            raise ConflictSetConnectorCompileError("claim_id 不得重复")
        proposition_keys = tuple(item.proposition.stable_key() for item in bindings)
        if len(set(proposition_keys)) != len(proposition_keys):
            raise ConflictSetConnectorCompileError("Proposition 不得重复")
        self._bindings = bindings
        self._claim_ids = claim_ids
        self._proposition_keys = proposition_keys
        if not isinstance(source, SourceRef):
            raise TypeError("conflict discourse source 类型错误")
        if (not isinstance(namespace, tuple) or not namespace
                or any(type(item) is not int for item in namespace)):
            raise ConflictSetConnectorCompileError(
                "conflict discourse namespace 非法")
        self._source = source
        self._namespace = namespace

    def declaration(
            self,
            selection: AnswerContentSelection,
            ) -> LanguageConnectorDiscourseDeclaration | None:
        """按 selected candidate 的真实 Proposition 恢复唯一 claim 顺序。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("conflict discourse selection 类型错误")
        selected_keys = set(selection.selected_candidate_keys)
        selected = tuple(
            candidate for candidate in selection.request.candidates
            if candidate.stable_key() in selected_keys
        )
        if len(selected) != len(selected_keys):
            raise ConflictSetConnectorCompileError("selection candidate 不可恢复")
        by_proposition = {candidate.proposition.stable_key(): candidate
                          for candidate in selected}
        if len(by_proposition) != len(selected):
            raise ConflictSetConnectorCompileError("selection Proposition 不得重复")
        if set(by_proposition) != set(self._proposition_keys):
            raise ConflictSetConnectorCompileError(
                "CONFLICT_SET selection 未精确覆盖 declared claim")
        ordered = tuple(by_proposition[key] for key in self._proposition_keys)
        candidate_keys = tuple(item.stable_key() for item in ordered)
        relation = structure_concept_identity((*self._namespace, 5, 1))
        reason = minimal_instruction_identity((*self._namespace, 5, 2))
        dependencies = tuple(
            DiscourseDependency(
                before.stable_key(),
                after.stable_key(),
                relation,
                reason,
                (*self._namespace, 5, 3, index),
            )
            for index, (before, after) in enumerate(
                zip(ordered, ordered[1:]), start=1)
        )
        return LanguageConnectorDiscourseDeclaration(
            candidate_keys,
            dependencies,
            self._source,
            (*self._namespace, 5, 4),
        )

    def state_key(self) -> tuple[int, ...]:
        """返回 claim/Proposition 顺序配置键。"""
        source_key = self._source.stable_key()
        result = [len(self._namespace), *self._namespace,
                  len(source_key), *source_key, len(self._claim_ids)]
        for claim_id, proposition_key in zip(
                self._claim_ids, self._proposition_keys, strict=True):
            claim_key = tuple(ord(char) for char in claim_id)
            result.extend((len(claim_key), *claim_key))
            result.extend((len(proposition_key), *proposition_key))
        return tuple(result)

    def clone_for_evaluation(self) -> "ConflictSetDiscourseDeclarations":
        """复制不可变声明配置，隔离评测读取状态。"""
        return ConflictSetDiscourseDeclarations(
            self._bindings, self._source, self._namespace)


@dataclass(frozen=True, slots=True)
class ConflictSetConnectorCompilation:
    """CONFLICT_SET 的真实多句 connector 和逐 claim 编译记录。"""

    plan: ConflictSetPlan
    language_branch: ObjectIdentity
    sentences: tuple[ConflictSetSentenceCompilation, ...]
    connector: LanguageGenerationConnector

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ConflictSetPlan):
            raise TypeError("compilation plan 类型错误")
        if (not isinstance(self.language_branch, ObjectIdentity)
                or self.language_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ConflictSetConnectorCompileError("language_branch 类型错误")
        if (not isinstance(self.sentences, tuple)
                or len(self.sentences) != len(self.plan.claim_ids)):
            raise ConflictSetConnectorCompileError("sentence 数量未覆盖 claim")
        if tuple(item.claim_id for item in self.sentences) != self.plan.claim_ids:
            raise ConflictSetConnectorCompileError("sentence 顺序漂移")
        if not isinstance(self.connector, LanguageGenerationConnector):
            raise TypeError("compilation connector 类型错误")

    def generate_surfaces(
            self,
            surfaces: tuple[str, ...],
            ) -> tuple[ConflictSetGeneratedSentence, ...]:
        """将一次公开 surface 结果绑定到本次 compilation 的 claim/source。"""
        return generate_conflict_set_sentences(self.plan, surfaces)

    @property
    def alias_pairs(self) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
        """返回 claim filler 到 Unicode representation 的完整安装对。"""
        return tuple(
            (item.claim_filler, item.claim_representation)
            for item in self.sentences
        )


@dataclass(frozen=True, slots=True)
class ConflictSetConnectorCompileRequest:
    """真实 connector compiler 的所有显式输入。"""

    plan: ConflictSetPlan
    bindings: tuple[ConflictSetSentenceBinding, ...]
    language_branch: ObjectIdentity
    surface_protocol: GenerationSurfaceProtocol
    representation_family: tuple[int, ...]
    discourse_source: SourceRef
    namespace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ConflictSetPlan):
            raise TypeError("compile request plan 类型错误")
        if (not isinstance(self.bindings, tuple)
                or len(self.bindings) != len(self.plan.claim_ids)
                or tuple(item.claim_id for item in self.bindings)
                != self.plan.claim_ids):
            raise ConflictSetConnectorCompileError(
                "bindings 必须按 plan claim 顺序精确覆盖")
        if len({item.proposition for item in self.bindings}) != len(self.bindings):
            raise ConflictSetConnectorCompileError("bindings Proposition 不得重复")
        if (not isinstance(self.language_branch, ObjectIdentity)
                or self.language_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ConflictSetConnectorCompileError("language_branch 类型错误")
        if not isinstance(self.surface_protocol, GenerationSurfaceProtocol):
            raise TypeError("surface_protocol 类型错误")
        if (not isinstance(self.representation_family, tuple)
                or not self.representation_family
                or any(type(item) is not int
                       for item in self.representation_family)):
            raise ConflictSetConnectorCompileError(
                "representation_family 必须是非空整数 tuple")
        if not isinstance(self.discourse_source, SourceRef):
            raise TypeError("discourse_source 类型错误")
        if (not isinstance(self.namespace, tuple) or not self.namespace
                or any(type(item) is not int for item in self.namespace)):
            raise ConflictSetConnectorCompileError("namespace 必须是非空整数 tuple")


def compile_conflict_set_connector(
        request: ConflictSetConnectorCompileRequest,
        ) -> ConflictSetConnectorCompilation:
    """把 typed 多命题计划接入真实 LanguageGenerationConnector。"""
    if not isinstance(request, ConflictSetConnectorCompileRequest):
        raise TypeError("compile request 类型错误")
    value_protocol = LanguageConnectorValueProtocol(*tuple(
        minimal_instruction_identity((*request.namespace, 1, index))
        for index in range(1, 5)
    ), ordinals=tuple(
        LanguageConnectorOrdinalDefinition(
            minimal_instruction_identity((*request.namespace, 2, index)), index)
        for index in range(len(request.bindings))
    ))
    templates = []
    policies = []
    sentence_compilations = []
    for sentence_ordinal, binding in enumerate(request.bindings, start=1):
        identity = (*request.namespace, 3, sentence_ordinal)
        sentence = structure_concept_identity((*identity, 1))
        structure = structure_concept_identity((*identity, 2))
        proposition_slot = structure_concept_identity((*identity, 3))
        claim_slot = structure_concept_identity((*identity, 4))
        value_type = concept_identity((*identity, 5))
        slots = (
            StructureSlotDefinition(
                structure, proposition_slot, role_identity((*identity, 6)),
                value_type),
            StructureSlotDefinition(
                structure, claim_slot, role_identity((*identity, 7)),
                value_type),
        )
        claim_constant = concept_identity((*identity, 8))
        bindings = (
            LanguageConnectorSlotBinding(
                structure_concept_identity((*identity, 9)),
                proposition_slot, value_protocol.proposition_source),
            LanguageConnectorSlotBinding(
                structure_concept_identity((*identity, 10)),
                claim_slot, value_protocol.constant_source,
                constant=claim_constant),
        )
        directives = (
            LanguageConnectorSurfaceDirective(
                structure_concept_identity((*identity, 11)),
                proposition_slot, request.surface_protocol.silent_action,
                minimal_instruction_identity((*identity, 12)),
                structure_concept_identity((*identity, 13)), ()),
            LanguageConnectorSurfaceDirective(
                structure_concept_identity((*identity, 14)),
                claim_slot, request.surface_protocol.emit_action,
                minimal_instruction_identity((*identity, 15)),
                structure_concept_identity((*identity, 16)), ()),
        )
        connector_id = structure_concept_identity((*identity, 17))
        order_constraint = structure_concept_identity((*identity, 18))
        template = LanguageGenerationConnectorTemplate(
            connector_id,
            request.language_branch,
            binding.proposition.structure,
            binding.proposition.predicate,
            sentence,
            structure,
            slots,
            bindings,
            structure_concept_identity((*identity, 19)),
            (order_constraint,),
            structure_concept_identity((*identity, 20)),
            (),
            minimal_instruction_identity((*identity, 21)),
            minimal_instruction_identity((*identity, 22)),
            directives,
        )
        templates.append(template)
        policies.append(LanguageConnectorTemplateRuntimePolicy(
            connector_id,
            (
                LanguageConnectorSurfaceRuntimePolicy(
                    proposition_slot, (*identity, 23, 1), None,
                    (*identity, 24, 1)),
                LanguageConnectorSurfaceRuntimePolicy(
                    claim_slot, (*identity, 23, 2),
                    AliasRouteSearchBudget(32, 32, 32), (*identity, 24, 2)),
            ),
        ))
        sentence_compilations.append(ConflictSetSentenceCompilation(
            binding.claim_id,
            binding.proposition,
            tuple(sorted({
                item.source_id for item in request.plan.evidence
                if item.claim_id == binding.claim_id
            })),
            binding.claim_surface,
            template,
            proposition_slot,
            claim_slot,
            claim_constant,
            representation_identity(
                request.representation_family,
                tuple(ord(char) for char in binding.claim_surface),
            ),
            order_constraint,
        ))
    connector = LanguageGenerationConnector(
        LanguageGenerationConnectorRegistry(value_protocol, tuple(templates)),
        LanguageGenerationConnectorRuntimePolicy(
            (*request.namespace, 4), StructureOrderSearchBudget(32), tuple(policies)),
        request.surface_protocol,
        discourse_declarations=ConflictSetDiscourseDeclarations(
            request.bindings, request.discourse_source, request.namespace),
    )
    return ConflictSetConnectorCompilation(
        request.plan,
        request.language_branch,
        tuple(sentence_compilations),
        connector,
    )


__all__ = [
    "ConflictSetConnectorCompilation",
    "ConflictSetConnectorCompileError",
    "ConflictSetConnectorCompileRequest",
    "ConflictSetDiscourseDeclarations",
    "ConflictSetSentenceBinding",
    "ConflictSetSentenceCompilation",
    "compile_conflict_set_connector",
]
