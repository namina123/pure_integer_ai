"""S-07 一等结构槽位和顺序约束的图拓扑。

本模块只保存可整体引用的 StructureConcept、Role、类型、约束参数和 H-06
Hypothesis 溯源。必要、偏好、可选、邻接、距离、条件和例外的具体含义均由
调用方注入的一等对象及后续 resolver 解释，通用图层不读取名称或键值语义。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_HYPOTHESIS,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_VALUE_TYPE_KINDS = frozenset({
    OBJECT_CONCEPT,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_ROLE,
})


class StructureOrderTopologyError(RuntimeError):
    """结构顺序图出现缺边、竞争端点或身份漂移。"""


def _require_identity(
        value: ObjectIdentity, *, where: str,
        object_kind: int | None = None) -> ObjectIdentity:
    """校验一等对象及可选类型，不解释对象 components 的领域含义。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if object_kind is not None and value.object_kind != object_kind:
        raise ValueError(f"{where} 对象类型不匹配")
    return value


def _require_identity_sequence(
        values: tuple[ObjectIdentity, ...], *, where: str,
        object_kind: int | None = None) -> tuple[ObjectIdentity, ...]:
    """校验无重复的一等对象序列，并保留调用方声明顺序。"""
    if not isinstance(values, tuple):
        raise TypeError(f"{where} 必须是 tuple")
    for index, value in enumerate(values):
        _require_identity(
            value, where=f"{where}[{index}]", object_kind=object_kind)
    if len({item.stable_key() for item in values}) != len(values):
        raise ValueError(f"{where} 不得包含重复对象")
    return values


@dataclass(frozen=True)
class StructureSlotDefinition:
    """一个结构成员槽位及其独立 Role 和值类型。"""

    structure: ObjectIdentity
    slot: ObjectIdentity
    role: ObjectIdentity
    value_type: ObjectIdentity

    def __post_init__(self) -> None:
        _require_identity(
            self.structure,
            where="StructureSlotDefinition.structure",
            object_kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _require_identity(
            self.slot,
            where="StructureSlotDefinition.slot",
            object_kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _require_identity(
            self.role,
            where="StructureSlotDefinition.role",
            object_kind=OBJECT_ROLE,
        )
        _require_identity(
            self.value_type, where="StructureSlotDefinition.value_type")
        if self.value_type.object_kind not in _VALUE_TYPE_KINDS:
            raise ValueError("slot value_type 必须是 Concept/StructureConcept/Role")
        if self.structure == self.slot:
            raise ValueError("structure 不得同时充当自身 slot")
        owners = {
            self.structure.owner,
            self.slot.owner,
            self.role.owner,
            self.value_type.owner,
        }
        if len(owners) != 1:
            raise ValueError("slot 定义的一等对象 owner 必须一致")


@dataclass(frozen=True)
class StructureOrderParameterDefinition:
    """顺序约束的一个一等参数绑定，参数 Role 与值对象分开保存。"""

    binding: ObjectIdentity
    role: ObjectIdentity
    value: ObjectIdentity

    def __post_init__(self) -> None:
        _require_identity(
            self.binding,
            where="StructureOrderParameterDefinition.binding",
            object_kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _require_identity(
            self.role,
            where="StructureOrderParameterDefinition.role",
            object_kind=OBJECT_ROLE,
        )
        _require_identity(
            self.value, where="StructureOrderParameterDefinition.value")
        if len({
                self.binding.owner,
                self.role.owner,
                self.value.owner,
                }) != 1:
            raise ValueError("constraint parameter 的一等对象 owner 必须一致")


@dataclass(frozen=True)
class StructureOrderConstraintDefinition:
    """一个可整体引用且带 H-06 溯源的结构局部顺序约束。"""

    constraint: ObjectIdentity
    language_branch: ObjectIdentity
    structure_family: ObjectIdentity
    structure: ObjectIdentity
    first_slot: ObjectIdentity
    second_slot: ObjectIdentity
    order_kind: ObjectIdentity
    constraint_kind: ObjectIdentity
    modality: ObjectIdentity
    context: ObjectIdentity
    conditions: tuple[ObjectIdentity, ...]
    exceptions: tuple[ObjectIdentity, ...]
    parameters: tuple[StructureOrderParameterDefinition, ...]
    hypothesis: HypothesisKey

    def __post_init__(self) -> None:
        for name, value, kind in (
                ("constraint", self.constraint, OBJECT_STRUCTURE_CONCEPT),
                ("language_branch", self.language_branch,
                 OBJECT_LANGUAGE_BRANCH),
                ("structure_family", self.structure_family,
                 OBJECT_STRUCTURE_CONCEPT),
                ("structure", self.structure, OBJECT_STRUCTURE_CONCEPT),
                ("first_slot", self.first_slot, OBJECT_STRUCTURE_CONCEPT),
                ("second_slot", self.second_slot, OBJECT_STRUCTURE_CONCEPT),
                ("order_kind", self.order_kind, OBJECT_CONCEPT),
                ("constraint_kind", self.constraint_kind, OBJECT_CONCEPT),
                ("modality", self.modality, OBJECT_CONCEPT),
                ("context", self.context, None)):
            _require_identity(
                value,
                where=f"StructureOrderConstraintDefinition.{name}",
                object_kind=kind,
            )
        _require_identity_sequence(
            self.conditions,
            where="StructureOrderConstraintDefinition.conditions",
        )
        _require_identity_sequence(
            self.exceptions,
            where="StructureOrderConstraintDefinition.exceptions",
        )
        if self.conditions != tuple(sorted(
                self.conditions, key=ObjectIdentity.stable_key)):
            raise ValueError("conditions 必须按完整对象稳定键规范化")
        if self.exceptions != tuple(sorted(
                self.exceptions, key=ObjectIdentity.stable_key)):
            raise ValueError("exceptions 必须按完整对象稳定键规范化")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters 必须是 tuple")
        if any(not isinstance(item, StructureOrderParameterDefinition)
               for item in self.parameters):
            raise TypeError("parameters 必须由 StructureOrderParameterDefinition 组成")
        if len({item.binding for item in self.parameters}) != len(
                self.parameters):
            raise ValueError("constraint parameter binding 不得重复")
        if len({item.role for item in self.parameters}) != len(self.parameters):
            raise ValueError("同一 constraint parameter Role 不得重复")
        if self.parameters != tuple(sorted(
                self.parameters,
                key=lambda item: item.binding.stable_key())):
            raise ValueError("parameters 必须按 binding 完整身份规范化")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        if self.first_slot == self.second_slot:
            raise ValueError("顺序约束的两个 slot 必须不同")
        identities = (
            self.constraint,
            self.language_branch,
            self.structure_family,
            self.structure,
            self.first_slot,
            self.second_slot,
            self.order_kind,
            self.constraint_kind,
            self.modality,
            self.context,
            *self.conditions,
            *self.exceptions,
            *(value for item in self.parameters
              for value in (item.binding, item.role, item.value)),
        )
        if any(item.owner != self.structure.owner for item in identities):
            raise ValueError("结构顺序定义的一等对象 owner 必须一致")

    @property
    def source(self):
        """返回聚合 H-06 Hypothesis 的版本化来源。"""
        return self.hypothesis.observation

    @property
    def scope(self) -> ScopeIdentity:
        """返回聚合 H-06 Hypothesis 的来源化 scope。"""
        return self.hypothesis.scope

    def canonical_conditions(self) -> tuple[ObjectIdentity, ...]:
        """按完整对象身份返回确定性 condition 集。"""
        return tuple(sorted(self.conditions, key=ObjectIdentity.stable_key))

    def canonical_exceptions(self) -> tuple[ObjectIdentity, ...]:
        """按完整对象身份返回确定性 exception 集。"""
        return tuple(sorted(self.exceptions, key=ObjectIdentity.stable_key))

    def canonical_parameters(
            self) -> tuple[StructureOrderParameterDefinition, ...]:
        """按 binding 完整身份返回确定性参数绑定。"""
        return tuple(sorted(
            self.parameters, key=lambda item: item.binding.stable_key()))


@dataclass(frozen=True)
class StructureOrderGraphPredicates:
    """结构顺序拓扑使用的开放 predicate 槽位。"""

    structure_language: TypedRef
    structure_family: TypedRef
    structure_slot: TypedRef
    slot_role: TypedRef
    slot_value_type: TypedRef
    structure_constraint: TypedRef
    constraint_structure: TypedRef
    constraint_first_slot: TypedRef
    constraint_second_slot: TypedRef
    constraint_order_kind: TypedRef
    constraint_kind: TypedRef
    constraint_modality: TypedRef
    constraint_context: TypedRef
    constraint_condition: TypedRef
    constraint_exception: TypedRef
    constraint_parameter: TypedRef
    parameter_role: TypedRef
    parameter_value: TypedRef
    constraint_hypothesis: TypedRef

    def refs(self) -> tuple[TypedRef, ...]:
        """按协议槽位顺序返回全部 predicate 引用。"""
        return (
            self.structure_language,
            self.structure_family,
            self.structure_slot,
            self.slot_role,
            self.slot_value_type,
            self.structure_constraint,
            self.constraint_structure,
            self.constraint_first_slot,
            self.constraint_second_slot,
            self.constraint_order_kind,
            self.constraint_kind,
            self.constraint_modality,
            self.constraint_context,
            self.constraint_condition,
            self.constraint_exception,
            self.constraint_parameter,
            self.parameter_role,
            self.parameter_value,
            self.constraint_hypothesis,
        )


@dataclass(frozen=True)
class MaterializedStructureSlot:
    """从权威图恢复的 slot 定义、引用和全部来源 statement。"""

    definition: StructureSlotDefinition
    slot: TypedRef
    role: TypedRef
    value_type: TypedRef
    statements: tuple[GraphStatement, ...]


@dataclass(frozen=True)
class MaterializedStructureOrderParameter:
    """从权威图恢复的约束参数绑定。"""

    definition: StructureOrderParameterDefinition
    binding: TypedRef
    role: TypedRef
    value: TypedRef
    statements: tuple[GraphStatement, ...]


@dataclass(frozen=True)
class MaterializedStructureOrderConstraint:
    """从权威图恢复的完整顺序约束和 provenance statement。"""

    definition: StructureOrderConstraintDefinition
    constraint: TypedRef
    parameters: tuple[MaterializedStructureOrderParameter, ...]
    statements: tuple[GraphStatement, ...]


@dataclass(frozen=True)
class MaterializedStructureOrder:
    """一个结构的一等成员和全部已定义顺序约束。"""

    structure: TypedRef
    language_branch: TypedRef
    structure_family: TypedRef
    slots: tuple[MaterializedStructureSlot, ...]
    constraints: tuple[MaterializedStructureOrderConstraint, ...]
    statements: tuple[GraphStatement, ...]


class StructureOrderGraph:
    """以 GraphOntology 为真源写入并恢复 S-07 结构顺序拓扑。"""

    def __init__(self, ontology: GraphOntology,
                 predicates: StructureOrderGraphPredicates) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if not isinstance(predicates, StructureOrderGraphPredicates):
            raise TypeError("predicates 必须是 StructureOrderGraphPredicates")
        self._ontology = ontology
        self._predicates = predicates
        self._validate_predicates()

    @property
    def ontology(self) -> GraphOntology:
        """返回当前 facade 绑定的权威图。"""
        return self._ontology

    @property
    def predicates(self) -> StructureOrderGraphPredicates:
        """返回调用方注入的结构顺序 predicate 协议。"""
        return self._predicates

    def define_constraint(
            self, slots: tuple[StructureSlotDefinition, ...],
            definition: StructureOrderConstraintDefinition, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int = 0, content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedStructureOrderConstraint:
        """整批预检后幂等追加结构、slot、constraint 和来源 statement。"""
        checked_slots = self._validate_definition(slots, definition)
        self._validate_metadata(
            definition,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        self._preflight_structure(definition)
        for slot in checked_slots:
            self._preflight_slot(slot)
        for parameter in definition.parameters:
            self._preflight_parameter(parameter)
        self._preflight_constraint(definition)

        metadata = (
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        structure = self._ontology.materialize(definition.structure)
        language = self._ontology.materialize(definition.language_branch)
        family = self._ontology.materialize(definition.structure_family)
        self._relate(
            self._predicates.structure_language,
            structure,
            language,
            metadata,
        )
        self._relate(
            self._predicates.structure_family,
            structure,
            family,
            metadata,
        )
        for slot_definition in checked_slots:
            slot = self._ontology.materialize(slot_definition.slot)
            role = self._ontology.materialize(slot_definition.role)
            value_type = self._ontology.materialize(slot_definition.value_type)
            self._relate(
                self._predicates.structure_slot,
                structure,
                slot,
                metadata,
            )
            self._relate(
                self._predicates.slot_role,
                slot,
                role,
                metadata,
            )
            self._relate(
                self._predicates.slot_value_type,
                slot,
                value_type,
                metadata,
            )

        constraint = self._ontology.materialize(definition.constraint)
        self._relate(
            self._predicates.structure_constraint,
            structure,
            constraint,
            metadata,
        )
        singleton_targets = (
            (self._predicates.constraint_structure, definition.structure),
            (self._predicates.constraint_first_slot, definition.first_slot),
            (self._predicates.constraint_second_slot, definition.second_slot),
            (self._predicates.constraint_order_kind, definition.order_kind),
            (self._predicates.constraint_kind, definition.constraint_kind),
            (self._predicates.constraint_modality, definition.modality),
            (self._predicates.constraint_context, definition.context),
            (self._predicates.constraint_hypothesis,
             definition.hypothesis.object_identity()),
        )
        for predicate, identity in singleton_targets:
            target = self._ontology.materialize(identity)
            self._relate(predicate, constraint, target, metadata)
        for identity in definition.canonical_conditions():
            self._relate(
                self._predicates.constraint_condition,
                constraint,
                self._ontology.materialize(identity),
                metadata,
            )
        for identity in definition.canonical_exceptions():
            self._relate(
                self._predicates.constraint_exception,
                constraint,
                self._ontology.materialize(identity),
                metadata,
            )
        for parameter in definition.canonical_parameters():
            binding = self._ontology.materialize(parameter.binding)
            self._relate(
                self._predicates.constraint_parameter,
                constraint,
                binding,
                metadata,
            )
            self._relate(
                self._predicates.parameter_role,
                binding,
                self._ontology.materialize(parameter.role),
                metadata,
            )
            self._relate(
                self._predicates.parameter_value,
                binding,
                self._ontology.materialize(parameter.value),
                metadata,
            )

        restored = self.read_constraint(constraint)
        if restored.definition != definition:
            raise StructureOrderTopologyError("写后恢复的 constraint 定义不一致")
        return restored

    def read_constraint(
            self, constraint: TypedRef,
            ) -> MaterializedStructureOrderConstraint:
        """从图中严格恢复一个 constraint 的单值槽、多值条件和参数。"""
        constraint_identity = self._ontology.identity_of(constraint)
        if constraint_identity.object_kind != OBJECT_STRUCTURE_CONCEPT:
            raise ValueError("constraint 必须是 StructureConcept")
        singleton_specs = (
            (self._predicates.constraint_structure, "structure"),
            (self._predicates.constraint_first_slot, "first slot"),
            (self._predicates.constraint_second_slot, "second slot"),
            (self._predicates.constraint_order_kind, "order kind"),
            (self._predicates.constraint_kind, "constraint kind"),
            (self._predicates.constraint_modality, "modality"),
            (self._predicates.constraint_context, "context"),
            (self._predicates.constraint_hypothesis, "Hypothesis"),
        )
        singleton = tuple(
            self._single_target(predicate, constraint, label=label)
            for predicate, label in singleton_specs
        )
        structure_ref = singleton[0][0]
        structure_identity = self._ontology.identity_of(structure_ref)
        language_ref, language_statements = self._single_target(
            self._predicates.structure_language,
            structure_ref,
            label="structure language",
        )
        family_ref, family_statements = self._single_target(
            self._predicates.structure_family,
            structure_ref,
            label="structure family",
        )
        membership = self._matching_statements(
            self._predicates.structure_constraint,
            subject=structure_ref,
            object_ref=constraint,
        )
        if not membership:
            raise StructureOrderTopologyError("constraint 缺少 structure 成员关系")
        incoming_structures = {
            self._ontology.identity_of(item.subject)
            for item in self._ontology.statements(
                predicate=self._predicates.structure_constraint,
                object_ref=constraint,
            )
        }
        if incoming_structures != {structure_identity}:
            raise StructureOrderTopologyError(
                "constraint 被多个 structure 引用或成员关系发生漂移")

        condition_targets, condition_statements = self._multi_targets(
            self._predicates.constraint_condition, constraint)
        exception_targets, exception_statements = self._multi_targets(
            self._predicates.constraint_exception, constraint)
        parameter_targets, parameter_statements = self._multi_targets(
            self._predicates.constraint_parameter, constraint)
        parameters = tuple(
            self._read_parameter(target)
            for target in parameter_targets
        )
        hypothesis_identity = self._ontology.identity_of(singleton[7][0])
        if hypothesis_identity.object_kind != OBJECT_HYPOTHESIS:
            raise StructureOrderTopologyError("constraint provenance 不是 Hypothesis")
        try:
            hypothesis = HypothesisKey.from_stable_key(
                hypothesis_identity.components)
        except (TypeError, ValueError) as exc:
            raise StructureOrderTopologyError(
                "constraint Hypothesis 完整身份无法恢复") from exc
        if hypothesis.object_identity() != hypothesis_identity:
            raise StructureOrderTopologyError("Hypothesis 图身份与候选键不一致")

        definition = StructureOrderConstraintDefinition(
            constraint_identity,
            self._ontology.identity_of(language_ref),
            self._ontology.identity_of(family_ref),
            structure_identity,
            self._ontology.identity_of(singleton[1][0]),
            self._ontology.identity_of(singleton[2][0]),
            self._ontology.identity_of(singleton[3][0]),
            self._ontology.identity_of(singleton[4][0]),
            self._ontology.identity_of(singleton[5][0]),
            self._ontology.identity_of(singleton[6][0]),
            tuple(self._ontology.identity_of(item)
                  for item in condition_targets),
            tuple(self._ontology.identity_of(item)
                  for item in exception_targets),
            tuple(item.definition for item in parameters),
            hypothesis,
        )
        statements = tuple(sorted(
            (
                *membership,
                *language_statements,
                *family_statements,
                *(statement for _, group in singleton for statement in group),
                *condition_statements,
                *exception_statements,
                *parameter_statements,
                *(statement for item in parameters
                  for statement in item.statements),
            ),
            key=lambda item: item.assertion_hash,
        ))
        return MaterializedStructureOrderConstraint(
            definition,
            constraint,
            parameters,
            statements,
        )

    def read_structure(self, structure: TypedRef) -> MaterializedStructureOrder:
        """恢复结构的 language/family、全部 slot 和已定义 constraint。"""
        identity = self._ontology.identity_of(structure)
        if identity.object_kind != OBJECT_STRUCTURE_CONCEPT:
            raise ValueError("structure 必须是 StructureConcept")
        language, language_statements = self._single_target(
            self._predicates.structure_language,
            structure,
            label="structure language",
        )
        family, family_statements = self._single_target(
            self._predicates.structure_family,
            structure,
            label="structure family",
        )
        slot_refs, slot_memberships = self._multi_targets(
            self._predicates.structure_slot, structure)
        constraint_refs, constraint_memberships = self._multi_targets(
            self._predicates.structure_constraint, structure)
        slots = tuple(self._read_slot(structure, item) for item in slot_refs)
        constraints = tuple(self.read_constraint(item) for item in constraint_refs)
        if any(item.definition.structure != identity for item in constraints):
            raise StructureOrderTopologyError(
                "structure 成员关系指向了其他结构的 constraint")
        statements = tuple(sorted(
            (
                *language_statements,
                *family_statements,
                *slot_memberships,
                *constraint_memberships,
                *(statement for item in slots for statement in item.statements),
                *(statement for item in constraints
                  for statement in item.statements),
            ),
            key=lambda item: item.assertion_hash,
        ))
        return MaterializedStructureOrder(
            structure,
            language,
            family,
            slots,
            constraints,
            statements,
        )

    def _read_slot(
            self, structure: TypedRef, slot: TypedRef,
            ) -> MaterializedStructureSlot:
        """恢复一个 structure 成员 slot 的唯一 Role 和 value type。"""
        membership = self._matching_statements(
            self._predicates.structure_slot,
            subject=structure,
            object_ref=slot,
        )
        if not membership:
            raise StructureOrderTopologyError("slot 缺少 structure 成员关系")
        role, role_statements = self._single_target(
            self._predicates.slot_role, slot, label="slot role")
        value_type, type_statements = self._single_target(
            self._predicates.slot_value_type,
            slot,
            label="slot value type",
        )
        definition = StructureSlotDefinition(
            self._ontology.identity_of(structure),
            self._ontology.identity_of(slot),
            self._ontology.identity_of(role),
            self._ontology.identity_of(value_type),
        )
        return MaterializedStructureSlot(
            definition,
            slot,
            role,
            value_type,
            tuple(sorted(
                (*membership, *role_statements, *type_statements),
                key=lambda item: item.assertion_hash,
            )),
        )

    def _read_parameter(
            self, binding: TypedRef,
            ) -> MaterializedStructureOrderParameter:
        """恢复 parameter binding 的唯一 Role 和值对象。"""
        role, role_statements = self._single_target(
            self._predicates.parameter_role,
            binding,
            label="parameter role",
        )
        value, value_statements = self._single_target(
            self._predicates.parameter_value,
            binding,
            label="parameter value",
        )
        definition = StructureOrderParameterDefinition(
            self._ontology.identity_of(binding),
            self._ontology.identity_of(role),
            self._ontology.identity_of(value),
        )
        return MaterializedStructureOrderParameter(
            definition,
            binding,
            role,
            value,
            tuple(sorted(
                (*role_statements, *value_statements),
                key=lambda item: item.assertion_hash,
            )),
        )

    def _validate_predicates(self) -> None:
        """核验全部拓扑 predicate 为当前图内互异 Concept。"""
        refs = self._predicates.refs()
        if any(not isinstance(item, TypedRef) for item in refs):
            raise TypeError("结构顺序 predicate 必须全部是 TypedRef")
        if len({item.stable_key() for item in refs}) != len(refs):
            raise ValueError("结构顺序 predicate 槽位必须互不相同")
        for ref in refs:
            if self._ontology.identity_of(ref).object_kind != OBJECT_CONCEPT:
                raise ValueError("结构顺序 predicate 必须是 Concept")

    @staticmethod
    def _validate_definition(
            slots: tuple[StructureSlotDefinition, ...],
            definition: StructureOrderConstraintDefinition,
            ) -> tuple[StructureSlotDefinition, ...]:
        """核验本次结构成员完整覆盖约束端点且无竞争 slot 定义。"""
        if not isinstance(definition, StructureOrderConstraintDefinition):
            raise TypeError("definition 必须是 StructureOrderConstraintDefinition")
        if not isinstance(slots, tuple) or not slots:
            raise ValueError("slots 必须是非空 tuple")
        if any(not isinstance(item, StructureSlotDefinition) for item in slots):
            raise TypeError("slots 必须由 StructureSlotDefinition 组成")
        by_slot: dict[ObjectIdentity, StructureSlotDefinition] = {}
        for item in slots:
            if item.structure != definition.structure:
                raise ValueError("slot 定义属于其他 StructureConcept")
            prior = by_slot.get(item.slot)
            if prior is not None and prior != item:
                raise ValueError("同一 slot 出现竞争 Role 或 value type")
            by_slot[item.slot] = item
        if len(by_slot) != len(slots):
            raise ValueError("同一次定义不得重复 slot")
        if (definition.first_slot not in by_slot
                or definition.second_slot not in by_slot):
            raise ValueError("约束左右 slot 必须出现在结构成员定义中")
        return tuple(sorted(
            slots, key=lambda item: item.slot.stable_key()))

    @staticmethod
    def _validate_metadata(
            definition: StructureOrderConstraintDefinition, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...]) -> None:
        """核验图 assertion 精确使用 H-06 aggregate scope 和开放整数元数据。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if scope != definition.scope:
            raise ValueError("结构顺序定义必须使用 H-06 aggregate scope")
        if scope.owner != definition.structure.owner:
            raise ValueError("结构顺序 scope 与 StructureConcept owner 不一致")
        if not isinstance(qualifiers, tuple):
            raise TypeError("qualifiers 必须是整数 tuple")
        assert_int(
            provenance_kind,
            epistemic_origin,
            content_version,
            *qualifiers,
            _where="StructureOrderGraph.define_constraint",
        )
        if type(provenance_kind) is not int or provenance_kind <= 0:
            raise ValueError("provenance_kind 必须为严格正整数")
        if type(epistemic_origin) is not int or epistemic_origin < 0:
            raise ValueError("epistemic_origin 必须为非负严格整数")
        if type(content_version) is not int or content_version < 0:
            raise ValueError("content_version 必须为非负严格整数")
        if any(type(item) is not int for item in qualifiers):
            raise ValueError("qualifiers 必须使用严格整数")

    def _preflight_structure(
            self, definition: StructureOrderConstraintDefinition) -> None:
        """允许追加成员和约束，但拒绝 structure 基础槽的部分或竞争拓扑。"""
        structure = self._ontology.resolve(definition.structure)
        if structure is None:
            return
        language = self._semantic_target_identities(
            self._predicates.structure_language, structure)
        family = self._semantic_target_identities(
            self._predicates.structure_family, structure)
        members = self._semantic_target_identities(
            self._predicates.structure_slot, structure)
        constraints = self._semantic_target_identities(
            self._predicates.structure_constraint, structure)
        if not language and not family:
            if members or constraints:
                raise StructureOrderTopologyError(
                    "structure 已有成员但缺少 language/family 基础槽")
            return
        if language != (definition.language_branch,):
            raise StructureOrderTopologyError("structure language 发生竞争或漂移")
        if family != (definition.structure_family,):
            raise StructureOrderTopologyError("structure family 发生竞争或漂移")

    def _preflight_slot(self, definition: StructureSlotDefinition) -> None:
        """核验既有 slot 是空壳或完全相同定义，拒绝部分修补。"""
        slot = self._ontology.resolve(definition.slot)
        if slot is None:
            return
        roles = self._semantic_target_identities(
            self._predicates.slot_role, slot)
        types = self._semantic_target_identities(
            self._predicates.slot_value_type, slot)
        if not roles and not types:
            memberships = self._ontology.statements(
                predicate=self._predicates.structure_slot,
                object_ref=slot,
            )
            if memberships:
                raise StructureOrderTopologyError(
                    "slot 已作为结构成员但缺少 Role/value type")
            return
        if roles != (definition.role,) or types != (definition.value_type,):
            raise StructureOrderTopologyError("slot Role/value type 发生部分或竞争拓扑")

    def _preflight_constraint(
            self, definition: StructureOrderConstraintDefinition) -> None:
        """核验 constraint instance 尚未定义或与完整声明逐槽一致。"""
        constraint = self._ontology.resolve(definition.constraint)
        if constraint is None:
            return
        predicates = (
            self._predicates.constraint_structure,
            self._predicates.constraint_first_slot,
            self._predicates.constraint_second_slot,
            self._predicates.constraint_order_kind,
            self._predicates.constraint_kind,
            self._predicates.constraint_modality,
            self._predicates.constraint_context,
            self._predicates.constraint_condition,
            self._predicates.constraint_exception,
            self._predicates.constraint_parameter,
            self._predicates.constraint_hypothesis,
        )
        has_topology = any(
            self._ontology.statements(predicate=item, subject=constraint)
            for item in predicates
        )
        if not has_topology:
            memberships = self._ontology.statements(
                predicate=self._predicates.structure_constraint,
                object_ref=constraint,
            )
            if memberships:
                raise StructureOrderTopologyError(
                    "constraint 已作为结构成员但缺少定义拓扑")
            return
        restored = self.read_constraint(constraint)
        if restored.definition != definition:
            raise StructureOrderTopologyError(
                "constraint instance 已绑定不同完整定义")

    def _preflight_parameter(
            self, definition: StructureOrderParameterDefinition) -> None:
        """核验既有 parameter binding 是空壳或完整相同定义。"""
        binding = self._ontology.resolve(definition.binding)
        if binding is None:
            return
        roles = self._semantic_target_identities(
            self._predicates.parameter_role, binding)
        values = self._semantic_target_identities(
            self._predicates.parameter_value, binding)
        if not roles and not values:
            memberships = self._ontology.statements(
                predicate=self._predicates.constraint_parameter,
                object_ref=binding,
            )
            if memberships:
                raise StructureOrderTopologyError(
                    "parameter 已作为约束成员但缺少 Role/value")
            return
        if roles != (definition.role,) or values != (definition.value,):
            raise StructureOrderTopologyError(
                "constraint parameter 发生部分或竞争拓扑")

    def _semantic_target_identities(
            self, predicate: TypedRef, subject: TypedRef,
            ) -> tuple[ObjectIdentity, ...]:
        """把同一语义端点的多来源 assertion 去重后返回完整对象身份。"""
        targets, _ = self._multi_targets(predicate, subject)
        return tuple(self._ontology.identity_of(item) for item in targets)

    def _single_target(
            self, predicate: TypedRef, subject: TypedRef, *, label: str,
            ) -> tuple[TypedRef, tuple[GraphStatement, ...]]:
        """读取唯一语义端点，允许该端点保留多条来源 assertion。"""
        targets, statements = self._multi_targets(predicate, subject)
        if len(targets) != 1:
            raise StructureOrderTopologyError(
                f"{label} 必须有唯一语义端点，实际 {len(targets)} 个")
        return targets[0], statements

    def _multi_targets(
            self, predicate: TypedRef, subject: TypedRef,
            ) -> tuple[tuple[TypedRef, ...], tuple[GraphStatement, ...]]:
        """按完整身份稳定排序多值端点，并保留全部 provenance statement。"""
        statements = self._ontology.statements(
            predicate=predicate, subject=subject)
        targets: dict[ObjectIdentity, TypedRef] = {}
        for statement in statements:
            identity = self._ontology.identity_of(statement.object)
            prior = targets.get(identity)
            if prior is not None and prior != statement.object:
                raise StructureOrderTopologyError("同一对象身份映射到多个图节点")
            targets[identity] = statement.object
        identities = tuple(sorted(targets, key=ObjectIdentity.stable_key))
        return (
            tuple(targets[item] for item in identities),
            tuple(sorted(statements, key=lambda item: item.assertion_hash)),
        )

    def _matching_statements(
            self, predicate: TypedRef, *, subject: TypedRef,
            object_ref: TypedRef) -> tuple[GraphStatement, ...]:
        """读取给定语义边的全部来源 assertion。"""
        return tuple(sorted(
            self._ontology.statements(
                predicate=predicate,
                subject=subject,
                object_ref=object_ref,
            ),
            key=lambda item: item.assertion_hash,
        ))

    def _relate(
            self, predicate: TypedRef, subject: TypedRef,
            object_ref: TypedRef,
            metadata: tuple[ScopeIdentity, int, int, int, tuple[int, ...]],
            ) -> GraphStatement:
        """用统一来源元数据追加一条结构定义 statement。"""
        scope, provenance, epistemic, content_version, qualifiers = metadata
        return self._ontology.relate(
            predicate,
            subject,
            object_ref,
            scope=scope,
            provenance_kind=provenance,
            epistemic_origin=epistemic,
            content_version=content_version,
            qualifiers=qualifiers,
        )


__all__ = [
    "MaterializedStructureOrder",
    "MaterializedStructureOrderConstraint",
    "MaterializedStructureOrderParameter",
    "MaterializedStructureSlot",
    "StructureOrderConstraintDefinition",
    "StructureOrderGraph",
    "StructureOrderGraphPredicates",
    "StructureOrderParameterDefinition",
    "StructureOrderTopologyError",
    "StructureSlotDefinition",
]
