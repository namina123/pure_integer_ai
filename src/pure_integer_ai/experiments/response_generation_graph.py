"""在既有 connector 图上持久化对话动作、来源和逐槽观察，不恢复整句答案。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_response import ResponseActGenerationTemplate
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_ATOM, OBJECT_SPAN, ObjectIdentity, SourceRef, concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order_consumer import StructureSlotValue
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorValueProtocol, LanguageGenerationConnectorTemplate,
)


RESPONSE_CONNECTOR_PROFILE = 91525

GENERIC_RESPONSE_GRAPH_ENTITY = 1
GENERIC_RESPONSE_GRAPH_EVENT = 2
GENERIC_RESPONSE_GRAPH_TOPIC = 3
GENERIC_RESPONSE_GRAPH_PROPOSITION = 4
GENERIC_RESPONSE_GRAPH_CONCEPT = 5
GENERIC_RESPONSE_GRAPH_ROLES = frozenset({
    GENERIC_RESPONSE_GRAPH_ENTITY,
    GENERIC_RESPONSE_GRAPH_EVENT,
    GENERIC_RESPONSE_GRAPH_TOPIC,
    GENERIC_RESPONSE_GRAPH_PROPOSITION,
    GENERIC_RESPONSE_GRAPH_CONCEPT,
})


def generic_response_context_role(branch: ObjectIdentity) -> ObjectIdentity:
    """返回 generic response-act 用来绑定当前 Memory 上下文的显式 Role。"""
    return role_identity(
        (RESPONSE_CONNECTOR_PROFILE, 6, 1),
        owner=branch.owner,
        versions=branch.versions,
    )


def generic_response_context_type(branch: ObjectIdentity) -> ObjectIdentity:
    """返回静默上下文槽的值类型；它不声明 Core 命题或语言表面。"""
    return concept_identity(
        (RESPONSE_CONNECTOR_PROFILE, 6, 2),
        owner=branch.owner,
        versions=branch.versions,
    )


def generic_response_context_marker(branch: ObjectIdentity) -> ObjectIdentity:
    """返回图内 generic response-act 上下文合同标记。"""
    return concept_identity(
        (RESPONSE_CONNECTOR_PROFILE, 6, 3),
        owner=branch.owner,
        versions=branch.versions,
    )


def generic_response_graph_role(
        branch: ObjectIdentity, category: int,
        ) -> ObjectIdentity:
    """Return the injected Role for one graph-derived response filler class."""
    if type(category) is not int or category not in GENERIC_RESPONSE_GRAPH_ROLES:
        raise ValueError("generic response graph role category is not registered")
    return role_identity(
        (RESPONSE_CONNECTOR_PROFILE, 7, category),
        owner=branch.owner,
        versions=branch.versions,
    )


def generic_response_graph_type(
        branch: ObjectIdentity, category: int,
        ) -> ObjectIdentity:
    """Return the slot type for a graph-derived response filler class."""
    if type(category) is not int or category not in GENERIC_RESPONSE_GRAPH_ROLES:
        raise ValueError("generic response graph type category is not registered")
    return concept_identity(
        (RESPONSE_CONNECTOR_PROFILE, 8, category),
        owner=branch.owner,
        versions=branch.versions,
    )


def response_predicate(branch: ObjectIdentity, index: int) -> ObjectIdentity:
    """冻结扩展关系槽位；语言内容及动作条件仍由训练图提供。"""
    if type(index) is not int or not 1 <= index <= 5:
        raise ValueError("对话 connector predicate 序号非法")
    return concept_identity((RESPONSE_CONNECTOR_PROFILE, 1, index),
                            owner=branch.owner, versions=branch.versions)


@dataclass(frozen=True, slots=True)
class RecoveredResponseConnector:
    """完整对话动作结构及训练观察；没有待回答的 Core 事实声明。"""

    template: LanguageGenerationConnectorTemplate
    stance: ObjectIdentity
    marker_slot: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    examples: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    evidence: tuple[tuple[int, ...], ...]
    condition_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验完整成员、来源与严格整数记录，不接受截断恢复对象。"""
        if (not isinstance(self.template, LanguageGenerationConnectorTemplate)
                or not isinstance(self.source, SourceRef) or not isinstance(self.scope, ScopeIdentity)
                or self.scope.owner != self.source.owner or self.scope.versions != self.source.versions):
            raise ValueError("对话 connector 来源或结构类型错误")
        if (self.marker_slot not in {item.slot for item in self.template.slots}
                or len(self.examples) != len(self.template.slots)
                or {slot for slot, _ in self.examples} != {item.slot for item in self.template.slots}
                or any(span.object_kind != OBJECT_SPAN
                       or SourceRef.from_stable_key(span.components[:11]) != self.source
                       for _, span in self.examples)
                or not self.evidence or self.evidence != tuple(sorted(set(self.evidence)))):
            raise ValueError("对话 connector 缺少完整逐槽来源及证据")
        if (type(self.condition_key) is not tuple
                or any(type(value) is not int or value < 0
                       for value in self.condition_key)):
            raise ValueError("对话 connector condition 必须是纯整数")
        self.stable_key()

    def bind_context(self, context, selection, values: LanguageConnectorValueProtocol):
        """按图内 Role/ordinal 和常量关系绑定既有三图上下文，保留全部形成证据。"""
        template = self.response_template(values)
        roles = tuple((item.slot, values.ordinal_value(item.ordinal))
                      for item in self.template.bindings if item.source == values.role_filler_source)
        constants = tuple(StructureSlotValue(item.slot, item.constant)
                          for item in self.template.bindings
                          if item.source == values.constant_source and item.slot != self.marker_slot)
        return context.bind_response_act(selection, template, roles, constants, (self.stable_key(),))

    def response_template(self, values: LanguageConnectorValueProtocol) -> ResponseActGenerationTemplate:
        """将训练图恢复成现有 G-02 三层 router 的完整多槽合同。"""
        slots = {item.slot: item for item in self.template.slots}
        bindings = {item.slot: item for item in self.template.bindings}
        marker = bindings[self.marker_slot]
        if (marker.source != values.constant_source or marker.constant is None
                or marker.constant.object_kind != OBJECT_LANGUAGE_ATOM):
            raise ValueError("对话动作 marker 必须是来源化语言原子，不是整句 stance 表示")
        return ResponseActGenerationTemplate(
            self.template.language_branch, self.stance, self.template.sentence,
            slots[self.marker_slot], self.template.boundary, self.template.linearization_reason,
            self.template.constraints, self.template.context,
            tuple(item for key, item in slots.items() if key != self.marker_slot), marker.constant,
        )

    def stable_key(self) -> tuple[int, ...]:
        """保留动作、完整 connector、来源、逐槽 Span 及全部形成证据。"""
        parts = (
            self.template.stable_key(), self.stance.stable_key(),
            self.marker_slot.stable_key(), self.source.stable_key(),
            self.scope.stable_key(),
            pack_record(1, *(pack_record(1, slot.stable_key(), span.stable_key())
                             for slot, span in self.examples)),
            pack_record(1, *self.evidence),
        )
        return pack_record(
            RESPONSE_CONNECTOR_PROFILE, *parts,
            *((pack_record(2, self.condition_key),) if self.condition_key else ()),
        )


def materialize_response_connector(graph, definition: RecoveredResponseConnector) -> None:
    """追加动作边与每个语言成员的实际 Span；connector/S-07 仍由原图负责。"""
    ontology = graph.ontology
    template = definition.template
    marker = definition.response_template(graph.value_protocol)
    if not marker.content_slots:
        raise ValueError("对话动作不能退回单槽整句课程")
    if (len(definition.examples) != len(template.slots)
            or {slot for slot, _ in definition.examples} != {item.slot for item in template.slots}
            or any(span.object_kind != OBJECT_SPAN
                   or SourceRef.from_stable_key(span.components[:11]) != definition.source
                   for _, span in definition.examples)):
        raise ValueError("对话观察必须逐槽保留完整来源 Span")
    if not definition.evidence or definition.evidence != tuple(sorted(set(definition.evidence))):
        raise ValueError("对话课程必须持有规范完整形成证据")
    qualifier_parts = (
        definition.source.stable_key(), pack_record(1, *definition.evidence))
    qualifiers = pack_record(
        RESPONSE_CONNECTOR_PROFILE, *qualifier_parts,
        *((pack_record(2, definition.condition_key),)
          if definition.condition_key else ()),
    )
    graph.materialize(template, scope=definition.scope, provenance_kind=RESPONSE_CONNECTOR_PROFILE,
                      content_version=1, qualifiers=qualifiers)
    predicates = tuple(ontology.materialize(response_predicate(template.language_branch, index))
                       for index in range(1, 6))
    root = ontology.materialize(template.connector)
    metadata = dict(scope=definition.scope, provenance_kind=RESPONSE_CONNECTOR_PROFILE,
                    content_version=1, qualifiers=qualifiers)
    for predicate, target in ((predicates[0], definition.stance),
                              (predicates[1], definition.marker_slot)):
        ontology.relate(predicate, root, ontology.materialize(target), **metadata)
    for slot, span in definition.examples:
        slot_ref = ontology.materialize(slot)
        ontology.relate(predicates[2], root, slot_ref, **metadata)
        ontology.relate(predicates[3], slot_ref, ontology.materialize(span), **metadata)
    ontology.relate(predicates[4], root, ontology.materialize(template.proposition_structure), **metadata)


def recover_response_connectors(graph, branch: ObjectIdentity) -> tuple[RecoveredResponseConnector, ...]:
    """只沿对话动作根恢复全部竞争结构；缺少扩展的旧模型返回空集合。"""
    ontology = graph.ontology
    predicates = tuple(ontology.resolve(response_predicate(branch, index)) for index in range(1, 6))
    if not any(predicates):
        return ()
    if not all(predicates):
        raise ValueError("对话 connector predicate 图部分缺失")
    results = []
    for row in ontology.statements(predicate=predicates[0]):
        template_record = graph.read(ontology.identity_of(row.subject))
        template = template_record.definition
        if template.language_branch != branch:
            continue
        fields = unpack_record(template_record.qualifiers, RESPONSE_CONNECTOR_PROFILE)
        if len(fields) not in {2, 3}:
            raise ValueError("对话 connector 来源记录不完整")
        source = SourceRef.from_stable_key(fields[0])
        evidence = unpack_record(fields[1], 1)
        if len(fields) == 2:
            condition_key = ()
        else:
            condition_fields = unpack_record(fields[2], 2)
            if len(condition_fields) != 1:
                raise ValueError("对话 connector condition 记录不完整")
            condition_key = condition_fields[0]
        statements = [row]

        def one(subject, predicate):
            """恢复唯一显式边；不以排序选择竞争端点。"""
            rows = ontology.statements(subject=subject, predicate=predicate)
            if len(rows) != 1:
                raise ValueError("对话 connector 缺边或存在竞争端点")
            statements.append(rows[0])
            return ontology.identity_of(rows[0].object)

        marker = one(row.subject, predicates[1])
        schema = one(row.subject, predicates[4])
        if schema != template.proposition_structure:
            raise ValueError("对话 connector 角色结构来源漂移")
        examples = []
        for member in ontology.statements(subject=row.subject, predicate=predicates[2]):
            statements.append(member)
            examples.append((ontology.identity_of(member.object), one(member.object, predicates[3])))
        if len(examples) != len(template.slots) or {slot for slot, _ in examples} != {
                item.slot for item in template.slots}:
            raise ValueError("对话 connector 缺少逐槽来源")
        for statement in statements:
            assertion = statement.assertion
            if (assertion.scope != template_record.scope or assertion.provenance_kind != RESPONSE_CONNECTOR_PROFILE
                    or assertion.content_version != 1 or assertion.qualifiers != template_record.qualifiers):
                raise ValueError("对话 connector 来源元数据漂移")
        result = RecoveredResponseConnector(
            template, ontology.identity_of(row.object), marker, source,
            template_record.scope,
            tuple(sorted(examples, key=lambda pair: pair[0].stable_key())),
            evidence, condition_key)
        result.response_template(graph.value_protocol)
        results.append(result)
    return tuple(sorted(results, key=lambda item: item.stable_key()))
