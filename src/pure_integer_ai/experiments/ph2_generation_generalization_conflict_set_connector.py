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
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
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
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
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
class ConflictSetSourceBinding:
    """把公开 source id 一一绑定到候选实际引用的 ``SourceRef``。"""

    source_id: str
    source: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ConflictSetConnectorCompileError("source_id 必须是非空文本")
        if not isinstance(self.source, SourceRef):
            raise TypeError("conflict source binding 类型错误")


def _candidate_source_stances(
        candidate: GenerationCandidate,
        ) -> dict[SourceRef, tuple[bool, bool]]:
    """从候选实际 Core/Memory Evidence 恢复逐来源 support/refute 位。"""
    states: dict[SourceRef, tuple[bool, bool]] = {}

    def accumulate(source: SourceRef, stance: int) -> None:
        support, refute = states.get(source, (False, False))
        states[source] = (
            support or stance == EVIDENCE_SUPPORT,
            refute or stance == EVIDENCE_REFUTE,
        )

    for evidence in candidate.evidence:
        accumulate(evidence.source, evidence.stance)
    for memory in candidate.memory_evidence:
        for source in memory.sources:
            accumulate(source.trace.source, source.trace.stance)
    return states


@dataclass(frozen=True, slots=True)
class ConflictSetSentenceBinding:
    """把一个 declared claim 绑定到真实候选、来源和公开 surface。"""

    claim_id: str
    candidate: GenerationCandidate
    claim_surface: str
    source_bindings: tuple[ConflictSetSourceBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ConflictSetConnectorCompileError("claim_id 必须是非空文本")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("conflict sentence candidate 类型错误")
        if not self.candidate.state.support or not self.candidate.state.refute:
            raise ConflictSetConnectorCompileError(
                "CONFLICT_SET candidate 必须同时携带 support/refute")
        if not isinstance(self.claim_surface, str) or not self.claim_surface:
            raise ConflictSetConnectorCompileError("claim_surface 必须是非空文本")
        if (not isinstance(self.source_bindings, tuple)
                or not self.source_bindings
                or any(not isinstance(item, ConflictSetSourceBinding)
                       for item in self.source_bindings)):
            raise ConflictSetConnectorCompileError(
                "source_bindings 必须是非空 typed tuple")
        source_ids = tuple(item.source_id for item in self.source_bindings)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ConflictSetConnectorCompileError(
                "source_bindings 必须按唯一 source_id 规范排序")
        sources = tuple(item.source for item in self.source_bindings)
        if len(set(sources)) != len(sources):
            raise ConflictSetConnectorCompileError(
                "source_id 到 SourceRef 必须一一映射")
        if set(sources) != set(self.candidate.citation_sources):
            raise ConflictSetConnectorCompileError(
                "source_bindings 未精确覆盖 candidate citation_sources")

    @property
    def proposition(self) -> BoundProposition:
        """返回真实候选的 bound Proposition。"""
        return self.candidate.proposition

    @property
    def source_ids(self) -> tuple[str, ...]:
        """返回与 source bindings 相同顺序的公开 source ids。"""
        return tuple(item.source_id for item in self.source_bindings)


@dataclass(frozen=True, slots=True)
class ConflictSetSentenceCompilation:
    """一个 claim 的 connector template、槽和来源归属。"""

    claim_id: str
    candidate: GenerationCandidate
    source_bindings: tuple[ConflictSetSourceBinding, ...]
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
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("sentence candidate 类型错误")
        if (not isinstance(self.source_bindings, tuple)
                or not self.source_bindings
                or any(not isinstance(item, ConflictSetSourceBinding)
                       for item in self.source_bindings)):
            raise ConflictSetConnectorCompileError(
                "sentence source_bindings 非规范")
        if tuple(item.source_id for item in self.source_bindings) != tuple(
                sorted({item.source_id for item in self.source_bindings})):
            raise ConflictSetConnectorCompileError(
                "sentence source_bindings 未规范排序")
        if set(item.source for item in self.source_bindings) != set(
                self.candidate.citation_sources):
            raise ConflictSetConnectorCompileError(
                "sentence source_bindings 与 candidate 来源漂移")
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

    @property
    def proposition(self) -> BoundProposition:
        """返回本句实际候选的 bound Proposition。"""
        return self.candidate.proposition

    @property
    def source_ids(self) -> tuple[str, ...]:
        """返回本句实际 SourceRef 对应的公开 source ids。"""
        return tuple(item.source_id for item in self.source_bindings)


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
        candidate_keys = tuple(item.candidate.stable_key() for item in bindings)
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ConflictSetConnectorCompileError("candidate 不得重复")
        self._bindings = bindings
        self._claim_ids = claim_ids
        self._candidate_keys = candidate_keys
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
        """按 selected candidate 的完整身份恢复唯一 claim 顺序。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("conflict discourse selection 类型错误")
        selected_keys = set(selection.selected_candidate_keys)
        selected = tuple(
            candidate for candidate in selection.request.candidates
            if candidate.stable_key() in selected_keys
        )
        if len(selected) != len(selected_keys):
            raise ConflictSetConnectorCompileError("selection candidate 不可恢复")
        by_candidate = {
            candidate.stable_key(): candidate for candidate in selected}
        if len(by_candidate) != len(selected):
            raise ConflictSetConnectorCompileError("selection candidate 不得重复")
        if set(by_candidate) != set(self._candidate_keys):
            raise ConflictSetConnectorCompileError(
                "CONFLICT_SET selection 未精确覆盖 declared claim")
        ordered = tuple(by_candidate[key] for key in self._candidate_keys)
        candidate_keys = tuple(item.stable_key() for item in ordered)
        relation = structure_concept_identity(
            (*self._namespace, 5, 1),
            owner=self._source.owner,
            versions=self._source.versions,
        )
        reason = minimal_instruction_identity(
            (*self._namespace, 5, 2),
            owner=self._source.owner,
            versions=self._source.versions,
        )
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
        """返回 claim/candidate 顺序配置键。"""
        source_key = self._source.stable_key()
        result = [len(self._namespace), *self._namespace,
                  len(source_key), *source_key, len(self._claim_ids)]
        for claim_id, candidate_key in zip(
                self._claim_ids, self._candidate_keys, strict=True):
            claim_key = tuple(ord(char) for char in claim_id)
            result.extend((len(claim_key), *claim_key))
            result.extend((len(candidate_key), *candidate_key))
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
                or any(not isinstance(item, ConflictSetSentenceBinding)
                       for item in self.bindings)
                or tuple(item.claim_id for item in self.bindings)
                != self.plan.claim_ids):
            raise ConflictSetConnectorCompileError(
                "bindings 必须按 plan claim 顺序精确覆盖")
        candidate_keys = tuple(
            item.candidate.stable_key() for item in self.bindings)
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ConflictSetConnectorCompileError("bindings candidate 不得重复")
        source_id_map: dict[str, SourceRef] = {}
        source_ref_map: dict[SourceRef, str] = {}
        for claim, binding in zip(
                self.plan.claims, self.bindings, strict=True):
            if binding.source_ids != claim.source_ids:
                raise ConflictSetConnectorCompileError(
                    "binding source ids 未精确覆盖 plan claim")
            states = _candidate_source_stances(binding.candidate)
            by_id = {item.source_id: item.source
                     for item in binding.source_bindings}
            support = tuple(sorted(
                source_id for source_id, source in by_id.items()
                if states.get(source, (False, False))[0]
            ))
            refute = tuple(sorted(
                source_id for source_id, source in by_id.items()
                if states.get(source, (False, False))[1]
            ))
            if (support != claim.support_source_ids
                    or refute != claim.refute_source_ids):
                raise ConflictSetConnectorCompileError(
                    "candidate 逐来源 stance 与 plan claim 不一致")
            for item in binding.source_bindings:
                if (source_id_map.setdefault(item.source_id, item.source)
                        != item.source):
                    raise ConflictSetConnectorCompileError(
                        "同一 source_id 不得映射不同 SourceRef")
                if (source_ref_map.setdefault(item.source, item.source_id)
                        != item.source_id):
                    raise ConflictSetConnectorCompileError(
                        "同一 SourceRef 不得映射不同 source_id")
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
        if (self.discourse_source.owner != self.language_branch.owner
                or self.discourse_source.versions
                != self.language_branch.versions):
            raise ConflictSetConnectorCompileError(
                "discourse_source 与 language_branch owner/version 不一致")
        if any(
                item.proposition.template.owner != self.language_branch.owner
                or item.proposition.template.versions
                != self.language_branch.versions
                for item in self.bindings):
            raise ConflictSetConnectorCompileError(
                "binding Proposition 与 language_branch owner/version 不一致")
        if (not isinstance(self.namespace, tuple) or not self.namespace
                or any(type(item) is not int for item in self.namespace)):
            raise ConflictSetConnectorCompileError("namespace 必须是非空整数 tuple")


def compile_conflict_set_connector(
        request: ConflictSetConnectorCompileRequest,
        ) -> ConflictSetConnectorCompilation:
    """把 typed 多命题计划接入真实 LanguageGenerationConnector。"""
    if not isinstance(request, ConflictSetConnectorCompileRequest):
        raise TypeError("compile request 类型错误")
    owner = request.language_branch.owner
    versions = request.language_branch.versions

    def instruction(key: tuple[int, ...]) -> ObjectIdentity:
        """建立继承 LanguageBranch owner/version 的 MinimalInstruction。"""
        return minimal_instruction_identity(key, owner=owner, versions=versions)

    def structure_identity(key: tuple[int, ...]) -> ObjectIdentity:
        """建立继承 LanguageBranch owner/version 的 StructureConcept。"""
        return structure_concept_identity(key, owner=owner, versions=versions)

    def concept(key: tuple[int, ...]) -> ObjectIdentity:
        """建立继承 LanguageBranch owner/version 的通用 Concept。"""
        return concept_identity(key, owner=owner, versions=versions)

    def role(key: tuple[int, ...]) -> ObjectIdentity:
        """建立继承 LanguageBranch owner/version 的 Role。"""
        return role_identity(key, owner=owner, versions=versions)

    value_protocol = LanguageConnectorValueProtocol(*tuple(
        instruction((*request.namespace, 1, index))
        for index in range(1, 5)
    ), ordinals=tuple(
        LanguageConnectorOrdinalDefinition(
            instruction((*request.namespace, 2, index)), index)
        for index in range(len(request.bindings))
    ))
    templates = []
    policies = []
    sentence_compilations = []
    for sentence_ordinal, binding in enumerate(request.bindings, start=1):
        identity = (*request.namespace, 3, sentence_ordinal)
        sentence = structure_identity((*identity, 1))
        structure = structure_identity((*identity, 2))
        proposition_slot = structure_identity((*identity, 3))
        claim_slot = structure_identity((*identity, 4))
        value_type = concept((*identity, 5))
        slots = (
            StructureSlotDefinition(
                structure, proposition_slot, role((*identity, 6)),
                value_type),
            StructureSlotDefinition(
                structure, claim_slot, role((*identity, 7)),
                value_type),
        )
        claim_constant = concept((*identity, 8))
        bindings = (
            LanguageConnectorSlotBinding(
                structure_identity((*identity, 9)),
                proposition_slot, value_protocol.proposition_source),
            LanguageConnectorSlotBinding(
                structure_identity((*identity, 10)),
                claim_slot, value_protocol.constant_source,
                constant=claim_constant),
        )
        directives = (
            LanguageConnectorSurfaceDirective(
                structure_identity((*identity, 11)),
                proposition_slot, request.surface_protocol.silent_action,
                instruction((*identity, 12)),
                structure_identity((*identity, 13)), ()),
            LanguageConnectorSurfaceDirective(
                structure_identity((*identity, 14)),
                claim_slot, request.surface_protocol.emit_action,
                instruction((*identity, 15)),
                structure_identity((*identity, 16)), ()),
        )
        connector_id = structure_identity((*identity, 17))
        order_constraint = structure_identity((*identity, 18))
        template = LanguageGenerationConnectorTemplate(
            connector_id,
            request.language_branch,
            binding.proposition.structure,
            binding.proposition.predicate,
            sentence,
            structure,
            slots,
            bindings,
            structure_identity((*identity, 19)),
            (order_constraint,),
            structure_identity((*identity, 20)),
            (),
            instruction((*identity, 21)),
            instruction((*identity, 22)),
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
            binding.candidate,
            binding.source_bindings,
            binding.claim_surface,
            template,
            proposition_slot,
            claim_slot,
            claim_constant,
            representation_identity(
                request.representation_family,
                tuple(ord(char) for char in binding.claim_surface),
                owner=owner,
                versions=versions,
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
    "ConflictSetSourceBinding",
    "ConflictSetSentenceBinding",
    "ConflictSetSentenceCompilation",
    "compile_conflict_set_connector",
]
